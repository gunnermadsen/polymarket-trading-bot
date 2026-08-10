//! Chainlink Data Streams BTC/USD provider-native one-minute OHLC ingestion.
//!
//! The history endpoint exposes only closed candles. Provider volume is
//! unsupported: the wire value must be zero and the durable fact remains NULL.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    fmt::Write as _,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use futures_util::StreamExt;
use reqwest::{Client, Response, StatusCode, Url};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::{FromRow, PgPool, Postgres, QueryBuilder, Transaction};
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    domain::{
        CaptureArtifact, DataGap, IngesterProfile, IngesterStrategy, IngesterStrategyKey,
        StrategyError, StrategyErrorKind,
    },
    persistence::{
        ArtifactBatch, ArtifactRepository, GapRepository, NewCaptureArtifact, NewDataGap,
        ProfileRepository, StrategyDegradation, StrategyProgress,
    },
    runtime::{StrategyFactory, StrategyFactoryError},
};

pub const STRATEGY_KEY: IngesterStrategyKey = IngesterStrategyKey::ChainlinkBtcusdOneMinuteOhlc;
pub const CONFIG_SCHEMA_VERSION: i32 = 1;
pub const CHECKPOINT_SCHEMA_VERSION: i32 = 1;

pub const LOGIN_ENV: &str = "MARKET_DATA_INGESTER_CHAINLINK_DATA_STREAMS_API_KEY";
pub const CANDLESTICK_API_KEY_ENV: &str =
    "MARKET_DATA_INGESTER_CHAINLINK_DATA_STREAMS_CANDLESTICK_API_KEY";

const SOURCE: &str = "chainlink_candlestick";
const SYMBOL: &str = "BTCUSD";
const RESOLUTION: &str = "1m";
const DEFAULT_BASE_URL: &str = "https://priceapi.dataengine.chain.link";
const MINUTE_SECONDS: i64 = 60;
const CHAINLINK_PRICE_SCALE: i128 = 1_000_000_000_000_000_000;
const MAX_AUTHORIZATION_BODY_BYTES: usize = 65_536;
const MAX_HISTORY_BODY_BYTES: usize = 2_097_152;
const GAP_REPAIRS_PER_POLL: i64 = 8;
const AUTHORIZATION_REFRESH_MARGIN_SECONDS: i64 = 30;
// Chainlink publishes a closed candle after one additional full minute.
const PROVIDER_PUBLICATION_DELAY_SECONDS: i64 = MINUTE_SECONDS * 2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ChainlinkBtcusdOneMinuteOhlcConfig {
    pub base_url: String,
    pub symbol: String,
    pub resolution: String,
    pub poll_interval_seconds: u64,
    pub startup_lookback_minutes: u16,
    pub overlap_minutes: u16,
    pub request_window_minutes: u16,
    pub artifact_window_seconds: i64,
    pub request_timeout_seconds: u64,
}

impl Default for ChainlinkBtcusdOneMinuteOhlcConfig {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_owned(),
            symbol: SYMBOL.to_owned(),
            resolution: RESOLUTION.to_owned(),
            poll_interval_seconds: 15,
            startup_lookback_minutes: 1_440,
            overlap_minutes: 5,
            request_window_minutes: 1_440,
            artifact_window_seconds: 3_600,
            request_timeout_seconds: 15,
        }
    }
}

impl ChainlinkBtcusdOneMinuteOhlcConfig {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let config = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            invalid_config(format!(
                "failed to decode Chainlink one-minute OHLC config: {error}"
            ))
        })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), StrategyFactoryError> {
        let url = Url::parse(&self.base_url)
            .map_err(|error| invalid_config(format!("base_url is invalid: {error}")))?;
        if self.base_url != DEFAULT_BASE_URL
            || url.scheme() != "https"
            || url.host_str() != Some("priceapi.dataengine.chain.link")
            || !url.username().is_empty()
            || url.password().is_some()
            || url.path() != "/"
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(invalid_config(
                "base_url must be the approved Chainlink Candlestick HTTPS origin",
            ));
        }
        if self.symbol != SYMBOL {
            return Err(invalid_config("symbol must be BTCUSD"));
        }
        if self.resolution != RESOLUTION {
            return Err(invalid_config("resolution must be 1m"));
        }
        if !(5..=300).contains(&self.poll_interval_seconds) {
            return Err(invalid_config(
                "poll_interval_seconds must be between 5 and 300",
            ));
        }
        if self.request_window_minutes == 0 || self.request_window_minutes > 1_440 {
            return Err(invalid_config(
                "request_window_minutes must be between 1 and 1440",
            ));
        }
        if self.startup_lookback_minutes == 0
            || self.startup_lookback_minutes > self.request_window_minutes
        {
            return Err(invalid_config(
                "startup_lookback_minutes must be positive and no greater than request_window_minutes",
            ));
        }
        if self.overlap_minutes == 0 || self.overlap_minutes > self.request_window_minutes {
            return Err(invalid_config(
                "overlap_minutes must be positive and no greater than request_window_minutes",
            ));
        }
        if self.artifact_window_seconds < MINUTE_SECONDS
            || self.artifact_window_seconds > 86_400
            || self.artifact_window_seconds % MINUTE_SECONDS != 0
        {
            return Err(invalid_config(
                "artifact_window_seconds must be a minute multiple between 60 and 86400",
            ));
        }
        if !(1..=60).contains(&self.request_timeout_seconds) {
            return Err(invalid_config(
                "request_timeout_seconds must be between 1 and 60",
            ));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct ChainlinkCandlestickCredentials {
    login: Arc<str>,
    api_key: Arc<str>,
}

impl ChainlinkCandlestickCredentials {
    fn from_environment() -> Result<Self, StrategyFactoryError> {
        Ok(Self {
            login: Arc::<str>::from(required_secret(LOGIN_ENV)?),
            api_key: Arc::<str>::from(required_secret(CANDLESTICK_API_KEY_ENV)?),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
struct CandleCheckpoint {
    last_open_timestamp: Option<DateTime<Utc>>,
}

impl CandleCheckpoint {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let checkpoint = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "invalid Chainlink one-minute OHLC checkpoint: {error}"
            ))
        })?;
        if checkpoint.last_open_timestamp.is_some_and(|timestamp| {
            timestamp.timestamp() < 0
                || timestamp.timestamp_subsec_nanos() != 0
                || timestamp.timestamp().rem_euclid(MINUTE_SECONDS) != 0
        }) {
            return Err(StrategyFactoryError::Construction(
                "Chainlink one-minute OHLC checkpoint must be a nonnegative whole UTC minute"
                    .to_owned(),
            ));
        }
        Ok(checkpoint)
    }

    fn to_value(last_open_timestamp: DateTime<Utc>) -> Value {
        json!({ "last_open_timestamp": last_open_timestamp })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ChainlinkBtcusdOneMinuteOhlcFactory;

impl StrategyFactory for ChainlinkBtcusdOneMinuteOhlcFactory {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    fn config_schema_version(&self) -> i32 {
        CONFIG_SCHEMA_VERSION
    }

    fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError> {
        ChainlinkBtcusdOneMinuteOhlcConfig::from_value(config).map(|_| ())
    }

    fn build(
        &self,
        profile: &IngesterProfile,
        pool: PgPool,
    ) -> Result<Box<dyn IngesterStrategy>, StrategyFactoryError> {
        if profile.strategy_key != STRATEGY_KEY {
            return Err(StrategyFactoryError::Construction(format!(
                "received profile for {}",
                profile.strategy_key
            )));
        }
        if profile.config_schema_version != CONFIG_SCHEMA_VERSION {
            return Err(StrategyFactoryError::Construction(format!(
                "Chainlink one-minute OHLC config schema must be {CONFIG_SCHEMA_VERSION}"
            )));
        }
        if profile.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(StrategyFactoryError::Construction(format!(
                "Chainlink one-minute OHLC checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}"
            )));
        }
        if profile.desired_generation <= 0 {
            return Err(StrategyFactoryError::Construction(
                "profile generation must be positive".to_owned(),
            ));
        }

        let config = ChainlinkBtcusdOneMinuteOhlcConfig::from_value(&profile.config)?;
        let config_snapshot = serde_json::to_value(&config).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "failed to encode effective Chainlink candle config: {error}"
            ))
        })?;
        let checkpoint = CandleCheckpoint::from_value(&profile.checkpoint)?;
        let lease_owner = profile.lease_owner.clone().ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease owner is missing".to_owned())
        })?;
        let lease_token = profile.lease_token.ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease token is missing".to_owned())
        })?;
        let credentials = ChainlinkCandlestickCredentials::from_environment()?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(config.request_timeout_seconds))
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("capitonic-market-data-ingester/0.1")
            .build()
            .map_err(|error| {
                StrategyFactoryError::Construction(format!(
                    "failed to build Chainlink Candlestick HTTP client: {error}"
                ))
            })?;

        Ok(Box::new(ChainlinkBtcusdOneMinuteOhlcStrategy {
            config,
            config_snapshot,
            profile_generation: profile.desired_generation,
            lease_owner: Arc::<str>::from(lease_owner),
            lease_token,
            initial_checkpoint: checkpoint,
            credentials,
            client,
            pool,
        }))
    }
}

pub struct ChainlinkBtcusdOneMinuteOhlcStrategy {
    config: ChainlinkBtcusdOneMinuteOhlcConfig,
    config_snapshot: Value,
    profile_generation: i64,
    lease_owner: Arc<str>,
    lease_token: Uuid,
    initial_checkpoint: CandleCheckpoint,
    credentials: ChainlinkCandlestickCredentials,
    client: Client,
    pool: PgPool,
}

struct CandleRunState {
    last_open: Option<DateTime<Utc>>,
    artifact: Option<CaptureArtifact>,
    access_token: Option<CachedAccessToken>,
}

#[derive(Clone)]
struct CachedAccessToken {
    value: Arc<str>,
    expires_at: DateTime<Utc>,
}

impl CachedAccessToken {
    fn is_usable_at(&self, now: DateTime<Utc>) -> bool {
        self.expires_at.timestamp()
            > now
                .timestamp()
                .saturating_add(AUTHORIZATION_REFRESH_MARGIN_SECONDS)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct OneMinuteCandle {
    open_timestamp: DateTime<Utc>,
    close_timestamp: DateTime<Utc>,
    received_at: DateTime<Utc>,
    open_price: Decimal,
    high_price: Decimal,
    low_price: Decimal,
    close_price: Decimal,
    payload_sha256: String,
}

impl OneMinuteCandle {
    fn from_wire_row(row: &[Value], received_at: DateTime<Utc>) -> Result<Self, StrategyError> {
        if row.len() != 6 {
            return Err(source_error(
                "chainlink_candle_wrong_row_shape",
                format!(
                    "Chainlink Candlestick row has {} fields, expected exactly 6",
                    row.len()
                ),
            ));
        }
        let timestamp = json_i64(&row[0], "timestamp")?;
        if timestamp < 0 || timestamp.rem_euclid(MINUTE_SECONDS) != 0 {
            return Err(source_error(
                "chainlink_candle_unaligned_timestamp",
                "Chainlink Candlestick timestamp is not a nonnegative UTC minute",
            ));
        }
        let open_timestamp = Utc.timestamp_opt(timestamp, 0).single().ok_or_else(|| {
            source_error(
                "chainlink_candle_timestamp_out_of_range",
                "Chainlink Candlestick timestamp is outside the supported range",
            )
        })?;
        let close_timestamp = open_timestamp + chrono::Duration::minutes(1);
        if close_timestamp > received_at {
            return Err(source_error(
                "chainlink_candle_not_closed",
                "Chainlink Candlestick row was received before its minute closed",
            ));
        }

        let open_price = scaled_price(&row[1], "open")?;
        let high_price = scaled_price(&row[2], "high")?;
        let low_price = scaled_price(&row[3], "low")?;
        let close_price = scaled_price(&row[4], "close")?;
        let provider_volume = decimal_value(&row[5], "volume")?;
        if provider_volume != Decimal::ZERO {
            return Err(source_error(
                "chainlink_candle_unsupported_volume",
                "Chainlink Candlestick returned nonzero unsupported volume",
            ));
        }
        if open_price <= Decimal::ZERO
            || high_price < open_price
            || high_price < close_price
            || high_price < low_price
            || low_price > open_price
            || low_price > close_price
        {
            return Err(source_error(
                "chainlink_candle_incoherent_ohlc",
                "Chainlink Candlestick OHLC prices are incoherent",
            ));
        }

        let mut candle = Self {
            open_timestamp,
            close_timestamp,
            received_at,
            open_price,
            high_price,
            low_price,
            close_price,
            payload_sha256: String::new(),
        };
        candle.payload_sha256 = candle.factual_payload_sha256();
        Ok(candle)
    }

    fn factual_payload_sha256(&self) -> String {
        let canonical = format!(
            "v1|source={SOURCE}|symbol={SYMBOL}|open_timestamp={}|close_timestamp={}|open_price={}|high_price={}|low_price={}|close_price={}|volume=unsupported",
            self.open_timestamp.timestamp(),
            self.close_timestamp.timestamp(),
            canonical_decimal(&self.open_price),
            canonical_decimal(&self.high_price),
            canonical_decimal(&self.low_price),
            canonical_decimal(&self.close_price),
        );
        sha256_hex(canonical.as_bytes())
    }

    fn factual_eq(&self, stored: &StoredCandle) -> bool {
        self.open_timestamp == stored.open_timestamp
            && self.close_timestamp == stored.close_timestamp
            && self.open_price == stored.open_price
            && self.high_price == stored.high_price
            && self.low_price == stored.low_price
            && self.close_price == stored.close_price
            && stored.volume.is_none()
            && !stored.volume_supported
            && self.payload_sha256 == stored.payload_sha256
    }
}

#[derive(Debug, FromRow)]
struct StoredCandle {
    open_timestamp: DateTime<Utc>,
    close_timestamp: DateTime<Utc>,
    open_price: Decimal,
    high_price: Decimal,
    low_price: Decimal,
    close_price: Decimal,
    volume: Option<Decimal>,
    volume_supported: bool,
    payload_sha256: String,
}

#[derive(Debug, FromRow)]
struct ArtifactChecksumRow {
    open_timestamp: DateTime<Utc>,
    payload_sha256: String,
}

#[derive(Debug)]
struct HistoryPage {
    candles: Vec<OneMinuteCandle>,
    received_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SourceGap {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorizationResponse {
    s: String,
    d: AuthorizationData,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorizationData {
    access_token: String,
    expiration: i64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoryResponse {
    s: String,
    candles: Vec<Vec<Value>>,
}

#[async_trait]
impl IngesterStrategy for ChainlinkBtcusdOneMinuteOhlcStrategy {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    async fn run(&self, shutdown: CancellationToken) -> Result<(), StrategyError> {
        let database_last_open = self.database_last_open().await?;
        if self.initial_checkpoint.last_open_timestamp.is_some() && database_last_open.is_none() {
            return Err(integrity_error(
                "chainlink_candle_checkpoint_without_fact",
                "Chainlink candle checkpoint exists without a durable source fact",
            ));
        }
        if let (Some(checkpoint), Some(database)) = (
            self.initial_checkpoint.last_open_timestamp,
            database_last_open,
        ) {
            if checkpoint > database {
                return Err(integrity_error(
                    "chainlink_candle_checkpoint_ahead",
                    format!(
                        "Chainlink candle checkpoint {checkpoint} is ahead of durable fact {database}"
                    ),
                ));
            }
        }

        let mut state = CandleRunState {
            last_open: database_last_open.or(self.initial_checkpoint.last_open_timestamp),
            artifact: None,
            access_token: None,
        };
        let mut ticker =
            tokio::time::interval(Duration::from_secs(self.config.poll_interval_seconds));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {
                    self.seal_artifact(&mut state, true).await?;
                    return Ok(());
                }
                _ = ticker.tick() => {
                    match self.capture_cycle(&mut state, &shutdown).await {
                        Ok(()) => {}
                        Err(error) if error.kind == StrategyErrorKind::Shutdown => {
                            self.seal_artifact(&mut state, true).await?;
                            return Ok(());
                        }
                        Err(error) if error.kind == StrategyErrorKind::LeaseLost => {
                            return self.finish_owned_drain(&mut state).await;
                        }
                        Err(error) if matches!(
                            error.kind,
                            StrategyErrorKind::TransientSource | StrategyErrorKind::TransientDatabase
                        ) => {
                            let marked = ProfileRepository::new(self.pool.clone())
                                .mark_degraded(
                                    STRATEGY_KEY,
                                    &self.lease_owner,
                                    self.lease_token,
                                    self.profile_generation,
                                    &StrategyDegradation {
                                        reason_code: error.code.to_owned(),
                                        reason_message: error.to_string(),
                                    },
                                )
                                .await
                                .map_err(|database| database_error(
                                    "chainlink_candle_degraded_state_failed",
                                    database,
                                ))?;
                            if !marked {
                                return self.finish_owned_drain(&mut state).await;
                            }
                            warn!(
                                strategy = %STRATEGY_KEY,
                                error_code = error.code,
                                error = %error,
                                "Chainlink one-minute OHLC poll failed and will retry"
                            );
                        }
                        Err(error) => {
                            if let Err(seal_error) = self.seal_artifact(&mut state, false).await {
                                warn!(
                                    strategy = %STRATEGY_KEY,
                                    error = %seal_error,
                                    "failed to seal Chainlink candle artifact after terminal failure"
                                );
                            }
                            return Err(error);
                        }
                    }
                }
            }
        }
    }
}

impl ChainlinkBtcusdOneMinuteOhlcStrategy {
    async fn database_last_open(&self) -> Result<Option<DateTime<Utc>>, StrategyError> {
        sqlx::query_scalar::<_, DateTime<Utc>>(
            r#"
            SELECT open_timestamp
            FROM market_data.chainlink_btcusd_one_minute_candles
            WHERE symbol = 'BTCUSD'
            ORDER BY open_timestamp DESC
            LIMIT 1
            "#,
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| database_error("chainlink_candle_cursor_read_failed", error))
    }

    async fn capture_cycle(
        &self,
        state: &mut CandleRunState,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        let latest_safe_open = latest_safely_published_open(Utc::now())?;
        if state
            .last_open
            .is_some_and(|cursor| cursor > latest_safe_open)
        {
            return Err(integrity_error(
                "chainlink_candle_cursor_in_future",
                format!(
                    "durable Chainlink candle cursor {:?} is newer than the latest safely published minute {latest_safe_open}",
                    state.last_open
                ),
            ));
        }

        let (request_start, mut gaps) = self.capture_window(state.last_open, latest_safe_open);
        let page = self
            .fetch_history_page(state, request_start, latest_safe_open, shutdown)
            .await?;
        gaps.extend(find_missing_ranges(
            &page.candles,
            request_start,
            latest_safe_open,
        ));
        gaps.sort_unstable();
        gaps.dedup();
        self.persist_capture(state, page, &gaps).await?;
        self.reconcile_gaps(state, shutdown).await
    }

    fn capture_window(
        &self,
        cursor: Option<DateTime<Utc>>,
        latest_safe_open: DateTime<Utc>,
    ) -> (DateTime<Utc>, Vec<SourceGap>) {
        let epoch = Utc.timestamp_opt(0, 0).single().expect("Unix epoch exists");
        match cursor {
            None => {
                let start = latest_safe_open
                    - chrono::Duration::minutes(i64::from(
                        self.config.startup_lookback_minutes.saturating_sub(1),
                    ));
                (start.max(epoch), Vec::new())
            }
            Some(cursor) => {
                let overlap_start = cursor
                    - chrono::Duration::minutes(i64::from(
                        self.config.overlap_minutes.saturating_sub(1),
                    ));
                let recent_start = latest_safe_open
                    - chrono::Duration::minutes(i64::from(
                        self.config.request_window_minutes.saturating_sub(1),
                    ));
                let request_start = overlap_start.max(recent_start).max(epoch);
                let skipped_start = cursor + chrono::Duration::minutes(1);
                let skipped_end = request_start - chrono::Duration::minutes(1);
                let gaps = if skipped_start <= skipped_end {
                    split_gap(
                        skipped_start,
                        skipped_end,
                        self.config.request_window_minutes,
                    )
                } else {
                    Vec::new()
                };
                (request_start, gaps)
            }
        }
    }

    async fn fetch_history_page(
        &self,
        state: &mut CandleRunState,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        shutdown: &CancellationToken,
    ) -> Result<HistoryPage, StrategyError> {
        if start > end
            || !is_minute_aligned(start)
            || !is_minute_aligned(end)
            || (end - start).num_minutes() >= i64::from(self.config.request_window_minutes)
        {
            return Err(integrity_error(
                "chainlink_candle_invalid_request_range",
                "Chainlink candle request range is not a bounded aligned minute window",
            ));
        }
        let latest_safe_open = latest_safely_published_open(Utc::now())?;
        if end > latest_safe_open {
            return Err(integrity_error(
                "chainlink_candle_open_request_range",
                "attempted to request a Chainlink candle newer than the provider publication cutoff",
            ));
        }

        for attempt in 0..2 {
            if state
                .access_token
                .as_ref()
                .is_none_or(|token| !token.is_usable_at(Utc::now()))
            {
                state.access_token = Some(self.authorize(shutdown).await?);
            }
            let token = state
                .access_token
                .as_ref()
                .expect("authorization installs an access token")
                .clone();
            let endpoint = format!(
                "{}/api/v1/history/rows",
                self.config.base_url.trim_end_matches('/')
            );
            let from = start.timestamp().to_string();
            let to = end
                .timestamp()
                .saturating_add(MINUTE_SECONDS - 1)
                .to_string();
            let request = self
                .client
                .get(endpoint)
                .bearer_auth(token.value.as_ref())
                .query(&[
                    ("symbol", self.config.symbol.as_str()),
                    ("resolution", self.config.resolution.as_str()),
                    ("from", from.as_str()),
                    ("to", to.as_str()),
                ])
                .send();
            let response = tokio::select! {
                _ = shutdown.cancelled() => return Err(shutdown_error()),
                response = request => response,
            }
            .map_err(|error| {
                source_error(
                    "chainlink_candle_history_request_failed",
                    format!("Chainlink Candlestick history request failed: {error}"),
                )
            })?;
            if matches!(
                response.status(),
                StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN
            ) {
                state.access_token = None;
                if attempt == 0 {
                    continue;
                }
                return Err(source_error(
                    "chainlink_candle_history_unauthorized",
                    "Chainlink Candlestick rejected a freshly authorized access token",
                ));
            }
            if !response.status().is_success() {
                return Err(source_error(
                    "chainlink_candle_history_status",
                    format!(
                        "Chainlink Candlestick history returned HTTP {}",
                        response.status()
                    ),
                ));
            }
            let body = read_bounded_body(
                response,
                MAX_HISTORY_BODY_BYTES,
                shutdown,
                "chainlink_candle_history_body_failed",
                "Chainlink Candlestick history",
            )
            .await?;
            // Receipt time is deliberately recorded only after the complete,
            // size-bounded provider body has arrived.
            let received_at = Utc::now();
            let candles = decode_history_body(&body, start, end, received_at)?;
            return Ok(HistoryPage {
                candles,
                received_at,
            });
        }
        unreachable!("authorization retry loop returns on its second attempt")
    }

    async fn authorize(
        &self,
        shutdown: &CancellationToken,
    ) -> Result<CachedAccessToken, StrategyError> {
        let endpoint = format!(
            "{}/api/v1/authorize",
            self.config.base_url.trim_end_matches('/')
        );
        let request = self
            .client
            .post(endpoint)
            .form(&[
                ("login", self.credentials.login.as_ref()),
                ("password", self.credentials.api_key.as_ref()),
            ])
            .send();
        let response = tokio::select! {
            _ = shutdown.cancelled() => return Err(shutdown_error()),
            response = request => response,
        }
        .map_err(|error| {
            source_error(
                "chainlink_candle_authorization_request_failed",
                format!("Chainlink Candlestick authorization request failed: {error}"),
            )
        })?;
        if !response.status().is_success() {
            return Err(source_error(
                "chainlink_candle_authorization_status",
                format!(
                    "Chainlink Candlestick authorization returned HTTP {}",
                    response.status()
                ),
            ));
        }
        let body = read_bounded_body(
            response,
            MAX_AUTHORIZATION_BODY_BYTES,
            shutdown,
            "chainlink_candle_authorization_body_failed",
            "Chainlink Candlestick authorization",
        )
        .await?;
        decode_authorization_body(&body, Utc::now())
    }

    async fn persist_capture(
        &self,
        state: &mut CandleRunState,
        page: HistoryPage,
        gaps: &[SourceGap],
    ) -> Result<(), StrategyError> {
        let artifact_id = self.ensure_artifact(state, page.received_at).await?;
        let timestamps: Vec<DateTime<Utc>> = page
            .candles
            .iter()
            .map(|candle| candle.open_timestamp)
            .collect();
        let mut transaction =
            self.pool.begin().await.map_err(|error| {
                database_error("chainlink_candle_transaction_begin_failed", error)
            })?;
        self.assert_lease_in(&mut transaction, false).await?;

        let inserted = self
            .persist_candles_in(&mut transaction, artifact_id, &page.candles, &timestamps)
            .await?;
        for gap in gaps {
            GapRepository::new(self.pool.clone())
                .detect_in(
                    &mut transaction,
                    &NewDataGap {
                        strategy_key: STRATEGY_KEY,
                        detected_artifact_id: Some(artifact_id),
                        gap_kind: "source_time".to_owned(),
                        reason_code: "chainlink_one_minute_candle_gap".to_owned(),
                        reason_message: Some(format!(
                            "missing provider one-minute candles from {} through {}",
                            gap.start, gap.end
                        )),
                        source_time_start: Some(gap.start),
                        source_time_end: Some(
                            gap.end + chrono::Duration::seconds(MINUTE_SECONDS - 1),
                        ),
                        start_cursor: Some(gap.start.timestamp().to_string()),
                        end_cursor: Some(gap.end.timestamp().to_string()),
                    },
                )
                .await
                .map_err(|error| database_error("chainlink_candle_gap_detect_failed", error))?;
        }

        if !gaps.is_empty() {
            let marked = ProfileRepository::new(self.pool.clone())
                .mark_degraded_in(
                    &mut transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                    &StrategyDegradation {
                        reason_code: "chainlink_candle_gap_repair".to_owned(),
                        reason_message: format!(
                            "tracking {} exact missing Chainlink candle range(s)",
                            gaps.len()
                        ),
                    },
                )
                .await
                .map_err(|error| database_error("chainlink_candle_gap_health_failed", error))?;
            if !marked {
                return Err(lease_lost_error());
            }
        }

        let mut artifact_after_commit = None;
        if !inserted.is_empty() {
            let inserted_candles: Vec<&OneMinuteCandle> = page
                .candles
                .iter()
                .filter(|candle| inserted.contains(&candle.open_timestamp))
                .collect();
            artifact_after_commit = Some(
                ArtifactRepository::new(self.pool.clone())
                    .record_batch_in(
                        &mut transaction,
                        artifact_id,
                        &artifact_batch(&inserted_candles),
                    )
                    .await
                    .map_err(|error| {
                        database_error("chainlink_candle_artifact_progress_failed", error)
                    })?
                    .ok_or_else(|| {
                        integrity_error(
                            "chainlink_candle_artifact_not_open",
                            format!("artifact {artifact_id} was not open during fact insert"),
                        )
                    })?,
            );
        }

        let next_last_open =
            page.candles
                .last()
                .map(|candle| candle.open_timestamp)
                .map(|page_last| {
                    state
                        .last_open
                        .map_or(page_last, |current| current.max(page_last))
                });
        if let Some(next_last_open) = next_last_open {
            let last_source_timestamp = page
                .candles
                .iter()
                .map(|candle| candle.close_timestamp)
                .max();
            let progressed = ProfileRepository::new(self.pool.clone())
                .record_progress_in(
                    &mut transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                    &StrategyProgress {
                        verified_record_count: page.candles.len() as i64,
                        checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
                        checkpoint: CandleCheckpoint::to_value(next_last_open),
                        last_source_event_at: last_source_timestamp,
                        last_provider_available_at: None,
                        source_watermark: last_source_timestamp,
                        availability_watermark: None,
                    },
                )
                .await
                .map_err(|error| {
                    database_error("chainlink_candle_profile_progress_failed", error)
                })?;
            if !progressed {
                return Err(lease_lost_error());
            }
        }

        transaction
            .commit()
            .await
            .map_err(|error| database_error("chainlink_candle_transaction_commit_failed", error))?;
        if let Some(artifact) = artifact_after_commit {
            state.artifact = Some(artifact);
        }
        if let Some(next_last_open) = next_last_open {
            state.last_open = Some(next_last_open);
        }
        Ok(())
    }

    async fn persist_candles_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        candles: &[OneMinuteCandle],
        timestamps: &[DateTime<Utc>],
    ) -> Result<BTreeSet<DateTime<Utc>>, StrategyError> {
        if candles.is_empty() {
            return Ok(BTreeSet::new());
        }
        let existing = self.load_existing_candles(transaction, timestamps).await?;
        let existing_by_open = unique_stored_candles(existing)?;
        let mut missing = Vec::new();
        for candle in candles {
            if let Some(stored) = existing_by_open.get(&candle.open_timestamp) {
                if !candle.factual_eq(stored) {
                    return Err(immutable_conflict(candle.open_timestamp));
                }
            } else {
                missing.push(candle);
            }
        }

        let inserted = self
            .insert_missing_candles(transaction, artifact_id, &missing)
            .await?;
        let durable =
            unique_stored_candles(self.load_existing_candles(transaction, timestamps).await?)?;
        for candle in candles {
            let Some(stored) = durable.get(&candle.open_timestamp) else {
                return Err(integrity_error(
                    "chainlink_candle_insert_missing",
                    format!(
                        "Chainlink candle {} was absent after insert",
                        candle.open_timestamp
                    ),
                ));
            };
            if !candle.factual_eq(stored) {
                return Err(immutable_conflict(candle.open_timestamp));
            }
        }
        Ok(inserted)
    }

    async fn load_existing_candles(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        timestamps: &[DateTime<Utc>],
    ) -> Result<Vec<StoredCandle>, StrategyError> {
        if timestamps.is_empty() {
            return Ok(Vec::new());
        }
        sqlx::query_as::<_, StoredCandle>(
            r#"
            SELECT open_timestamp, close_timestamp,
                   open_price, high_price, low_price, close_price,
                   volume, volume_supported,
                   payload_sha256::text AS payload_sha256
            FROM market_data.chainlink_btcusd_one_minute_candles
            WHERE symbol = 'BTCUSD'
              AND open_timestamp = ANY($1::timestamptz[])
            ORDER BY open_timestamp
            "#,
        )
        .bind(timestamps)
        .fetch_all(&mut **transaction)
        .await
        .map_err(|error| database_error("chainlink_candle_fact_read_failed", error))
    }

    async fn insert_missing_candles(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        missing: &[&OneMinuteCandle],
    ) -> Result<BTreeSet<DateTime<Utc>>, StrategyError> {
        if missing.is_empty() {
            return Ok(BTreeSet::new());
        }
        let mut builder = QueryBuilder::<Postgres>::new(
            r#"
            INSERT INTO market_data.chainlink_btcusd_one_minute_candles (
              source, symbol, open_timestamp, close_timestamp,
              provider_available_at, received_at,
              open_price, high_price, low_price, close_price,
              volume, volume_supported, payload_sha256,
              strategy_key, capture_artifact_id
            )
            "#,
        );
        builder.push_values(missing, |mut row, candle| {
            row.push_bind(SOURCE)
                .push_bind(SYMBOL)
                .push_bind(candle.open_timestamp)
                .push_bind(candle.close_timestamp)
                .push_bind(Option::<DateTime<Utc>>::None)
                .push_bind(candle.received_at)
                .push_bind(candle.open_price)
                .push_bind(candle.high_price)
                .push_bind(candle.low_price)
                .push_bind(candle.close_price)
                .push_bind(Option::<Decimal>::None)
                .push_bind(false)
                .push_bind(&candle.payload_sha256)
                .push_bind(STRATEGY_KEY.as_str())
                .push_bind(artifact_id);
        });
        builder.push(" ON CONFLICT (symbol, open_timestamp) DO NOTHING RETURNING open_timestamp");
        let inserted = builder
            .build_query_scalar::<DateTime<Utc>>()
            .fetch_all(&mut **transaction)
            .await
            .map_err(|error| database_error("chainlink_candle_fact_insert_failed", error))?;
        Ok(inserted.into_iter().collect())
    }

    fn artifact_window(&self, timestamp: DateTime<Utc>) -> (DateTime<Utc>, DateTime<Utc>) {
        let start_seconds = timestamp
            .timestamp()
            .div_euclid(self.config.artifact_window_seconds)
            .saturating_mul(self.config.artifact_window_seconds);
        let start = Utc
            .timestamp_opt(start_seconds, 0)
            .single()
            .expect("aligned artifact timestamp is representable");
        let end = Utc
            .timestamp_opt(
                start_seconds.saturating_add(self.config.artifact_window_seconds),
                0,
            )
            .single()
            .expect("bounded artifact timestamp is representable");
        (start, end)
    }

    async fn ensure_artifact(
        &self,
        state: &mut CandleRunState,
        received_at: DateTime<Utc>,
    ) -> Result<Uuid, StrategyError> {
        let (window_start, window_end) = self.artifact_window(received_at);
        let current_matches = state.artifact.as_ref().is_some_and(|artifact| {
            artifact.profile_generation == self.profile_generation
                && artifact.config_schema_version == CONFIG_SCHEMA_VERSION
                && artifact.config_snapshot == self.config_snapshot
                && artifact.capture_window_start == window_start
                && artifact.capture_window_end == window_end
        });
        if current_matches {
            return Ok(state
                .artifact
                .as_ref()
                .expect("matching artifact exists")
                .artifact_id);
        }
        if state.artifact.is_some() {
            self.seal_artifact(state, false).await?;
        }

        let repository = ArtifactRepository::new(self.pool.clone());
        if let Some(open) = repository
            .get_open(STRATEGY_KEY)
            .await
            .map_err(|error| database_error("chainlink_candle_artifact_read_failed", error))?
        {
            verify_artifact_generation(open.profile_generation, self.profile_generation)?;
            let reusable = open.profile_generation == self.profile_generation
                && open.config_schema_version == CONFIG_SCHEMA_VERSION
                && open.config_snapshot == self.config_snapshot
                && open.capture_window_start == window_start
                && open.capture_window_end == window_end;
            if open.profile_generation == self.profile_generation
                && (open.config_schema_version != CONFIG_SCHEMA_VERSION
                    || open.config_snapshot != self.config_snapshot)
            {
                return Err(integrity_error(
                    "chainlink_candle_artifact_config_conflict",
                    "open artifact has the current generation but different effective configuration",
                ));
            }
            state.artifact = Some(open);
            if reusable {
                return Ok(state
                    .artifact
                    .as_ref()
                    .expect("reusable artifact exists")
                    .artifact_id);
            }
            self.seal_artifact(state, false).await?;
        }

        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("chainlink_candle_artifact_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let artifact = repository
            .create_in(
                &mut transaction,
                &NewCaptureArtifact {
                    strategy_key: STRATEGY_KEY,
                    profile_generation: self.profile_generation,
                    config_schema_version: CONFIG_SCHEMA_VERSION,
                    config_snapshot: self.config_snapshot.clone(),
                    capture_window_start: window_start,
                    capture_window_end: window_end,
                    start_cursor: state
                        .last_open
                        .map(|timestamp| timestamp.timestamp().to_string()),
                },
            )
            .await
            .map_err(|error| database_error("chainlink_candle_artifact_create_failed", error))?;
        transaction
            .commit()
            .await
            .map_err(|error| database_error("chainlink_candle_artifact_commit_failed", error))?;
        let artifact_id = artifact.artifact_id;
        state.artifact = Some(artifact);
        info!(
            strategy = %STRATEGY_KEY,
            %artifact_id,
            %window_start,
            %window_end,
            "opened Chainlink one-minute OHLC capture artifact"
        );
        Ok(artifact_id)
    }

    async fn seal_artifact(
        &self,
        state: &mut CandleRunState,
        allow_draining_generation: bool,
    ) -> Result<(), StrategyError> {
        if let Some(artifact) = state.artifact.as_ref() {
            verify_artifact_generation(artifact.profile_generation, self.profile_generation)?;
        }
        let Some(artifact) = state.artifact.as_ref().cloned() else {
            let mut transaction = self.pool.begin().await.map_err(|error| {
                database_error("chainlink_candle_drain_transaction_failed", error)
            })?;
            self.assert_lease_in(&mut transaction, allow_draining_generation)
                .await?;
            transaction
                .commit()
                .await
                .map_err(|error| database_error("chainlink_candle_drain_commit_failed", error))?;
            return Ok(());
        };

        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("chainlink_candle_artifact_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, allow_draining_generation)
            .await?;
        let (content_sha256, end_cursor) =
            self.artifact_seal_in(&mut transaction, &artifact).await?;
        let completed = ArtifactRepository::new(self.pool.clone())
            .complete_in(
                &mut transaction,
                artifact.artifact_id,
                &content_sha256,
                end_cursor.as_deref().or(artifact.end_cursor.as_deref()),
            )
            .await
            .map_err(|error| database_error("chainlink_candle_artifact_complete_failed", error))?;
        if completed.is_none() {
            return Err(integrity_error(
                "chainlink_candle_artifact_not_open",
                format!(
                    "artifact {} was not open while sealing",
                    artifact.artifact_id
                ),
            ));
        }
        transaction
            .commit()
            .await
            .map_err(|error| database_error("chainlink_candle_artifact_commit_failed", error))?;
        state.artifact = None;
        info!(
            strategy = %STRATEGY_KEY,
            artifact_id = %artifact.artifact_id,
            record_count = artifact.record_count,
            %content_sha256,
            "sealed Chainlink one-minute OHLC capture artifact"
        );
        Ok(())
    }

    async fn artifact_seal_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact: &CaptureArtifact,
    ) -> Result<(String, Option<String>), StrategyError> {
        let rows = sqlx::query_as::<_, ArtifactChecksumRow>(
            r#"
            SELECT open_timestamp, payload_sha256::text AS payload_sha256
            FROM market_data.chainlink_btcusd_one_minute_candles
            WHERE capture_artifact_id = $1 AND strategy_key = $2
            ORDER BY open_timestamp
            "#,
        )
        .bind(artifact.artifact_id)
        .bind(STRATEGY_KEY.as_str())
        .fetch_all(&mut **transaction)
        .await
        .map_err(|error| database_error("chainlink_candle_checksum_read_failed", error))?;
        if i64::try_from(rows.len()).ok() != Some(artifact.record_count) {
            return Err(integrity_error(
                "chainlink_candle_artifact_count_mismatch",
                format!(
                    "artifact {} records {} rows but owns {} facts",
                    artifact.artifact_id,
                    artifact.record_count,
                    rows.len()
                ),
            ));
        }
        let mut hasher = Sha256::new();
        for row in &rows {
            hash_field(&mut hasher, &row.open_timestamp.timestamp().to_string());
            hash_field(&mut hasher, &row.payload_sha256);
        }
        Ok((
            digest_hex(hasher.finalize()),
            rows.last()
                .map(|row| row.open_timestamp.timestamp().to_string()),
        ))
    }

    async fn assert_lease_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        allow_draining_generation: bool,
    ) -> Result<(), StrategyError> {
        let profiles = ProfileRepository::new(self.pool.clone());
        let current = if allow_draining_generation {
            profiles
                .lock_owned_lease_in(
                    transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                )
                .await
        } else {
            profiles
                .lock_current_lease_in(
                    transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                )
                .await
        }
        .map_err(|error| database_error("chainlink_candle_lease_check_failed", error))?;
        if !current {
            return Err(lease_lost_error());
        }
        Ok(())
    }

    async fn finish_owned_drain(&self, state: &mut CandleRunState) -> Result<(), StrategyError> {
        self.seal_artifact(state, true).await?;
        info!(
            strategy = %STRATEGY_KEY,
            generation = self.profile_generation,
            "Chainlink one-minute OHLC strategy drained after a desired-state lease race"
        );
        Ok(())
    }

    async fn reconcile_gaps(
        &self,
        state: &mut CandleRunState,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        let gaps = GapRepository::new(self.pool.clone())
            .list_unresolved_limited(STRATEGY_KEY, GAP_REPAIRS_PER_POLL)
            .await
            .map_err(|error| database_error("chainlink_candle_gap_list_failed", error))?;
        for gap in gaps {
            if shutdown.is_cancelled() {
                return Err(shutdown_error());
            }
            let range = gap_source_range(&gap)?;
            let complete_before = self.gap_is_complete(range).await?;
            if !self.begin_gap_repair(gap.gap_id).await? {
                continue;
            }
            if complete_before {
                self.ensure_artifact(state, Utc::now()).await?;
            } else {
                let page = self
                    .fetch_history_page(state, range.start, range.end, shutdown)
                    .await?;
                self.persist_capture(state, page, &[]).await?;
            }
            if self.gap_is_complete(range).await? {
                self.complete_repair_artifact(state, gap.gap_id).await?;
            }
        }
        Ok(())
    }

    async fn begin_gap_repair(&self, gap_id: Uuid) -> Result<bool, StrategyError> {
        let mut transaction =
            self.pool.begin().await.map_err(|error| {
                database_error("chainlink_candle_gap_transaction_failed", error)
            })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let begun = GapRepository::new(self.pool.clone())
            .begin_repair_in(&mut transaction, gap_id)
            .await
            .map_err(|error| database_error("chainlink_candle_gap_begin_failed", error))?
            .is_some();
        transaction.commit().await.map_err(|error| {
            database_error("chainlink_candle_gap_transaction_commit_failed", error)
        })?;
        Ok(begun)
    }

    async fn gap_is_complete(&self, range: SourceGap) -> Result<bool, StrategyError> {
        sqlx::query_scalar::<_, bool>(
            r#"
            SELECT NOT EXISTS (
              SELECT 1
              FROM generate_series(
                $1::timestamptz,
                $2::timestamptz,
                INTERVAL '1 minute'
              ) AS expected(open_timestamp)
              LEFT JOIN market_data.chainlink_btcusd_one_minute_candles AS facts
                ON facts.open_timestamp = expected.open_timestamp
               AND facts.symbol = 'BTCUSD'
              WHERE facts.open_timestamp IS NULL
            )
            "#,
        )
        .bind(range.start)
        .bind(range.end)
        .fetch_one(&self.pool)
        .await
        .map_err(|error| database_error("chainlink_candle_gap_verify_failed", error))
    }

    async fn complete_repair_artifact(
        &self,
        state: &mut CandleRunState,
        gap_id: Uuid,
    ) -> Result<(), StrategyError> {
        let artifact = state.artifact.as_ref().cloned().ok_or_else(|| {
            integrity_error(
                "chainlink_candle_repair_artifact_missing",
                "gap repair completed without an open capture artifact",
            )
        })?;
        verify_artifact_generation(artifact.profile_generation, self.profile_generation)?;
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("chainlink_candle_artifact_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let (content_sha256, end_cursor) =
            self.artifact_seal_in(&mut transaction, &artifact).await?;
        ArtifactRepository::new(self.pool.clone())
            .complete_in(
                &mut transaction,
                artifact.artifact_id,
                &content_sha256,
                end_cursor.as_deref().or(artifact.end_cursor.as_deref()),
            )
            .await
            .map_err(|error| database_error("chainlink_candle_artifact_complete_failed", error))?
            .ok_or_else(|| {
                integrity_error(
                    "chainlink_candle_artifact_not_open",
                    format!(
                        "artifact {} was not open while sealing",
                        artifact.artifact_id
                    ),
                )
            })?;
        GapRepository::new(self.pool.clone())
            .mark_repaired_in(
                &mut transaction,
                gap_id,
                artifact.artifact_id,
                "chainlink_history_closed_candles_recovered",
                Some("all missing provider one-minute candles are durably present"),
            )
            .await
            .map_err(|error| database_error("chainlink_candle_gap_complete_failed", error))?
            .ok_or_else(|| {
                integrity_error(
                    "chainlink_candle_gap_not_repairing",
                    format!("data gap {gap_id} was not repairing during completion"),
                )
            })?;
        transaction
            .commit()
            .await
            .map_err(|error| database_error("chainlink_candle_artifact_commit_failed", error))?;
        state.artifact = None;
        info!(
            strategy = %STRATEGY_KEY,
            artifact_id = %artifact.artifact_id,
            %gap_id,
            %content_sha256,
            "sealed Chainlink candle repair artifact and resolved gap"
        );
        Ok(())
    }
}

fn decode_authorization_body(
    body: &[u8],
    received_at: DateTime<Utc>,
) -> Result<CachedAccessToken, StrategyError> {
    let authorization = serde_json::from_slice::<AuthorizationResponse>(body).map_err(|error| {
        source_error(
            "chainlink_candle_authorization_invalid_body",
            format!("invalid Chainlink Candlestick authorization response: {error}"),
        )
    })?;
    if authorization.s != "ok"
        || authorization.d.access_token.trim().is_empty()
        || authorization.d.access_token.len() > 16_384
    {
        return Err(source_error(
            "chainlink_candle_authorization_invalid_body",
            "Chainlink Candlestick authorization did not contain a bounded nonempty access token",
        ));
    }
    let expires_at = Utc
        .timestamp_opt(authorization.d.expiration, 0)
        .single()
        .ok_or_else(|| {
            source_error(
                "chainlink_candle_authorization_invalid_body",
                "Chainlink Candlestick authorization expiration is outside the supported range",
            )
        })?;
    let token = CachedAccessToken {
        value: Arc::<str>::from(authorization.d.access_token),
        expires_at,
    };
    if !token.is_usable_at(received_at) {
        return Err(source_error(
            "chainlink_candle_authorization_invalid_body",
            "Chainlink Candlestick authorization token expires within the refresh margin",
        ));
    }
    Ok(token)
}

fn decode_history_body(
    body: &[u8],
    requested_start: DateTime<Utc>,
    requested_end: DateTime<Utc>,
    received_at: DateTime<Utc>,
) -> Result<Vec<OneMinuteCandle>, StrategyError> {
    let payload = serde_json::from_slice::<HistoryResponse>(body).map_err(|error| {
        source_error(
            "chainlink_candle_history_invalid_body",
            format!("invalid Chainlink Candlestick history response: {error}"),
        )
    })?;
    if payload.s != "ok" {
        return Err(source_error(
            "chainlink_candle_history_non_ok",
            "Chainlink Candlestick history returned a non-ok status",
        ));
    }
    let expected_count = requested_end
        .signed_duration_since(requested_start)
        .num_minutes()
        .saturating_add(1);
    if i64::try_from(payload.candles.len()).unwrap_or(i64::MAX) > expected_count {
        return Err(source_error(
            "chainlink_candle_history_row_limit",
            format!(
                "Chainlink Candlestick returned {} rows for a {expected_count}-minute range",
                payload.candles.len()
            ),
        ));
    }
    let mut candles = payload
        .candles
        .iter()
        .map(|row| OneMinuteCandle::from_wire_row(row, received_at))
        .collect::<Result<Vec<_>, _>>()?;
    candles.sort_unstable_by_key(|candle| candle.open_timestamp);
    for candle in &candles {
        if candle.open_timestamp < requested_start || candle.open_timestamp > requested_end {
            return Err(source_error(
                "chainlink_candle_history_outside_range",
                format!(
                    "Chainlink Candlestick returned minute {} outside {requested_start} through {requested_end}",
                    candle.open_timestamp
                ),
            ));
        }
    }
    if candles
        .windows(2)
        .any(|pair| pair[0].open_timestamp >= pair[1].open_timestamp)
    {
        return Err(source_error(
            "chainlink_candle_history_duplicate_timestamp",
            "Chainlink Candlestick history contained duplicate timestamps",
        ));
    }
    Ok(candles)
}

async fn read_bounded_body(
    response: Response,
    maximum_bytes: usize,
    shutdown: &CancellationToken,
    error_code: &'static str,
    context: &'static str,
) -> Result<Vec<u8>, StrategyError> {
    if response
        .content_length()
        .is_some_and(|length| length > maximum_bytes as u64)
    {
        return Err(source_error(
            error_code,
            format!("{context} response exceeds {maximum_bytes} bytes"),
        ));
    }
    let capacity = response
        .content_length()
        .and_then(|length| usize::try_from(length).ok())
        .unwrap_or(0)
        .min(maximum_bytes);
    let mut body = Vec::with_capacity(capacity);
    let mut stream = response.bytes_stream();
    loop {
        let next = tokio::select! {
            _ = shutdown.cancelled() => return Err(shutdown_error()),
            next = stream.next() => next,
        };
        let Some(chunk) = next else {
            return Ok(body);
        };
        let chunk = chunk.map_err(|error| {
            source_error(
                error_code,
                format!("failed to stream {context} response: {error}"),
            )
        })?;
        let next_len = body.len().checked_add(chunk.len()).ok_or_else(|| {
            source_error(error_code, format!("{context} response length overflowed"))
        })?;
        if next_len > maximum_bytes {
            return Err(source_error(
                error_code,
                format!("{context} response exceeds {maximum_bytes} bytes"),
            ));
        }
        body.extend_from_slice(&chunk);
    }
}

fn find_missing_ranges(
    candles: &[OneMinuteCandle],
    requested_start: DateTime<Utc>,
    requested_end: DateTime<Utc>,
) -> Vec<SourceGap> {
    let mut gaps = Vec::new();
    let mut expected = requested_start;
    for candle in candles {
        if candle.open_timestamp > expected {
            gaps.push(SourceGap {
                start: expected,
                end: candle.open_timestamp - chrono::Duration::minutes(1),
            });
        }
        expected = candle.open_timestamp + chrono::Duration::minutes(1);
    }
    if expected <= requested_end {
        gaps.push(SourceGap {
            start: expected,
            end: requested_end,
        });
    }
    gaps
}

fn split_gap(start: DateTime<Utc>, end: DateTime<Utc>, maximum_minutes: u16) -> Vec<SourceGap> {
    let mut gaps = Vec::new();
    let mut next = start;
    let width = chrono::Duration::minutes(i64::from(maximum_minutes.saturating_sub(1)));
    while next <= end {
        let chunk_end = (next + width).min(end);
        gaps.push(SourceGap {
            start: next,
            end: chunk_end,
        });
        next = chunk_end + chrono::Duration::minutes(1);
    }
    gaps
}

fn artifact_batch(candles: &[&OneMinuteCandle]) -> ArtifactBatch {
    ArtifactBatch {
        inserted_record_count: candles.len() as i64,
        minimum_source_timestamp: candles.iter().map(|candle| candle.close_timestamp).min(),
        maximum_source_timestamp: candles.iter().map(|candle| candle.close_timestamp).max(),
        minimum_received_at: candles.iter().map(|candle| candle.received_at).min(),
        maximum_received_at: candles.iter().map(|candle| candle.received_at).max(),
        start_cursor: candles
            .first()
            .map(|candle| candle.open_timestamp.timestamp().to_string()),
        end_cursor: candles
            .last()
            .map(|candle| candle.open_timestamp.timestamp().to_string()),
    }
}

fn unique_stored_candles(
    candles: Vec<StoredCandle>,
) -> Result<BTreeMap<DateTime<Utc>, StoredCandle>, StrategyError> {
    let mut by_open = BTreeMap::new();
    for candle in candles {
        if by_open.insert(candle.open_timestamp, candle).is_some() {
            return Err(integrity_error(
                "chainlink_candle_duplicate_identity",
                "multiple durable rows share one BTCUSD one-minute open timestamp",
            ));
        }
    }
    Ok(by_open)
}

fn gap_source_range(gap: &DataGap) -> Result<SourceGap, StrategyError> {
    let start = gap
        .start_cursor
        .as_deref()
        .ok_or_else(|| {
            integrity_error(
                "chainlink_candle_gap_cursor_missing",
                format!("data gap {} has no start cursor", gap.gap_id),
            )
        })?
        .parse::<i64>()
        .map_err(|_| {
            integrity_error(
                "chainlink_candle_gap_cursor_invalid",
                format!("data gap {} has an invalid start cursor", gap.gap_id),
            )
        })?;
    let end = gap
        .end_cursor
        .as_deref()
        .ok_or_else(|| {
            integrity_error(
                "chainlink_candle_gap_cursor_missing",
                format!("data gap {} has no end cursor", gap.gap_id),
            )
        })?
        .parse::<i64>()
        .map_err(|_| {
            integrity_error(
                "chainlink_candle_gap_cursor_invalid",
                format!("data gap {} has an invalid end cursor", gap.gap_id),
            )
        })?;
    if start < 0
        || end < start
        || start.rem_euclid(MINUTE_SECONDS) != 0
        || end.rem_euclid(MINUTE_SECONDS) != 0
    {
        return Err(integrity_error(
            "chainlink_candle_gap_cursor_invalid",
            format!("data gap {} has invalid minute boundaries", gap.gap_id),
        ));
    }
    let start = Utc.timestamp_opt(start, 0).single().ok_or_else(|| {
        integrity_error(
            "chainlink_candle_gap_cursor_invalid",
            format!("data gap {} start cursor is out of range", gap.gap_id),
        )
    })?;
    let end = Utc.timestamp_opt(end, 0).single().ok_or_else(|| {
        integrity_error(
            "chainlink_candle_gap_cursor_invalid",
            format!("data gap {} end cursor is out of range", gap.gap_id),
        )
    })?;
    Ok(SourceGap { start, end })
}

fn latest_safely_published_open(now: DateTime<Utc>) -> Result<DateTime<Utc>, StrategyError> {
    let current_minute = now
        .timestamp()
        .div_euclid(MINUTE_SECONDS)
        .saturating_mul(MINUTE_SECONDS);
    Utc.timestamp_opt(
        current_minute.saturating_sub(PROVIDER_PUBLICATION_DELAY_SECONDS),
        0,
    )
    .single()
    .ok_or_else(|| {
        integrity_error(
            "chainlink_candle_closed_minute_out_of_range",
            "latest safely published Chainlink minute is outside the supported range",
        )
    })
}

fn is_minute_aligned(timestamp: DateTime<Utc>) -> bool {
    timestamp.timestamp_subsec_nanos() == 0 && timestamp.timestamp().rem_euclid(MINUTE_SECONDS) == 0
}

fn json_i64(value: &Value, field: &str) -> Result<i64, StrategyError> {
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|encoded| encoded.parse().ok()))
        .ok_or_else(|| {
            source_error(
                "chainlink_candle_invalid_integer",
                format!("Chainlink Candlestick {field} is not an integer"),
            )
        })
}

fn scaled_price(value: &Value, field: &str) -> Result<Decimal, StrategyError> {
    let raw = decimal_value(value, field)?;
    if raw <= Decimal::ZERO || !raw.fract().is_zero() {
        return Err(source_error(
            "chainlink_candle_invalid_scaled_price",
            format!("Chainlink Candlestick {field} must be a positive integer scaled by 1e18"),
        ));
    }
    let price = raw
        .checked_div(Decimal::from_i128_with_scale(CHAINLINK_PRICE_SCALE, 0))
        .ok_or_else(|| {
            source_error(
                "chainlink_candle_scale_overflow",
                format!("Chainlink Candlestick {field} scale overflowed"),
            )
        })?;
    validate_price_storage_bound(&price, field)?;
    Ok(price)
}

fn decimal_value(value: &Value, field: &str) -> Result<Decimal, StrategyError> {
    let encoded = match value {
        Value::Number(number) => number.to_string(),
        Value::String(string) => string.clone(),
        _ => {
            return Err(source_error(
                "chainlink_candle_invalid_decimal",
                format!("Chainlink Candlestick {field} is not numeric"),
            ));
        }
    };
    encoded
        .parse::<Decimal>()
        .or_else(|_| Decimal::from_scientific(&encoded))
        .map_err(|error| {
            source_error(
                "chainlink_candle_invalid_decimal",
                format!("Chainlink Candlestick {field} exceeds decimal capacity: {error}"),
            )
        })
}

fn validate_price_storage_bound(value: &Decimal, field: &str) -> Result<(), StrategyError> {
    let canonical = canonical_decimal(value);
    let integer_digits = canonical
        .split_once('.')
        .map_or(canonical.as_str(), |(integer, _)| integer)
        .trim_start_matches('0')
        .len()
        .max(1);
    if value.normalize().scale() > 18 || integer_digits > 20 {
        return Err(source_error(
            "chainlink_candle_decimal_out_of_range",
            format!("Chainlink Candlestick {field} exceeds numeric(38,18)"),
        ));
    }
    Ok(())
}

fn canonical_decimal(value: &Decimal) -> String {
    value.normalize().to_string()
}

fn required_secret(key: &'static str) -> Result<String, StrategyFactoryError> {
    let value = env::var(key).map_err(|_| {
        StrategyFactoryError::Construction(format!(
            "required service-owned secret {key} is not configured"
        ))
    })?;
    if value.trim().is_empty() {
        return Err(StrategyFactoryError::Construction(format!(
            "required service-owned secret {key} is empty"
        )));
    }
    Ok(value)
}

fn immutable_conflict(timestamp: DateTime<Utc>) -> StrategyError {
    integrity_error(
        "chainlink_candle_immutable_conflict",
        format!("durable Chainlink candle {timestamp} conflicts with provider facts"),
    )
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn sha256_hex(bytes: &[u8]) -> String {
    digest_hex(Sha256::digest(bytes))
}

fn digest_hex(digest: impl AsRef<[u8]>) -> String {
    let digest = digest.as_ref();
    let mut encoded = String::with_capacity(digest.len().saturating_mul(2));
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn invalid_config(message: impl Into<String>) -> StrategyFactoryError {
    StrategyFactoryError::InvalidConfiguration(message.into())
}

fn source_error(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::TransientSource, code, message)
}

fn database_error(code: &'static str, error: impl std::fmt::Display) -> StrategyError {
    StrategyError::new(
        StrategyErrorKind::TransientDatabase,
        code,
        error.to_string(),
    )
}

fn integrity_error(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::Integrity, code, message)
}

fn lease_lost_error() -> StrategyError {
    StrategyError::new(
        StrategyErrorKind::LeaseLost,
        "chainlink_candle_lease_lost",
        "Chainlink one-minute OHLC profile lease was lost before progress committed",
    )
}

fn verify_artifact_generation(
    artifact_generation: i64,
    strategy_generation: i64,
) -> Result<(), StrategyError> {
    if artifact_generation > strategy_generation {
        return Err(lease_lost_error());
    }
    Ok(())
}

fn shutdown_error() -> StrategyError {
    StrategyError::new(
        StrategyErrorKind::Shutdown,
        "chainlink_candle_shutdown",
        "Chainlink one-minute OHLC strategy shutdown requested",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const AUTHORIZATION_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/chainlink/authorization_v1.json");
    const HISTORY_FIXTURE: &[u8] =
        include_bytes!("../../../tests/fixtures/chainlink/one_minute_ohlc_history_v1.json");

    fn minute(seconds: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(seconds, 0)
            .single()
            .expect("fixture timestamp")
    }

    #[test]
    fn default_config_matches_the_seed_contract() {
        let config = ChainlinkBtcusdOneMinuteOhlcConfig::default();
        assert_eq!(config.base_url, DEFAULT_BASE_URL);
        assert_eq!(config.symbol, SYMBOL);
        assert_eq!(config.resolution, RESOLUTION);
        assert_eq!(config.poll_interval_seconds, 15);
        assert_eq!(config.startup_lookback_minutes, 1_440);
        assert_eq!(config.overlap_minutes, 5);
        assert_eq!(config.request_window_minutes, 1_440);
        assert_eq!(config.artifact_window_seconds, 3_600);
        assert_eq!(config.request_timeout_seconds, 15);
        config.validate().expect("default config is valid");
    }

    #[test]
    fn config_is_typed_strict_and_never_contains_credentials() {
        assert!(ChainlinkBtcusdOneMinuteOhlcConfig::from_value(&json!({
            "unexpected": true
        }))
        .is_err());
        let encoded = serde_json::to_value(ChainlinkBtcusdOneMinuteOhlcConfig::default())
            .expect("serialize config");
        let text = encoded.to_string();
        assert!(!text.contains("credential"));
        assert!(!text.contains("api_key"));
        assert!(!text.contains("password"));
    }

    #[test]
    fn credentialed_requests_reject_unapproved_origins() {
        for base_url in [
            "https://example.com",
            "https://priceapi.dataengine.chain.link.evil.example",
            "https://user@priceapi.dataengine.chain.link",
            "https://priceapi.dataengine.chain.link/alternate",
        ] {
            let config = ChainlinkBtcusdOneMinuteOhlcConfig {
                base_url: base_url.to_owned(),
                ..ChainlinkBtcusdOneMinuteOhlcConfig::default()
            };
            assert!(config.validate().is_err(), "accepted {base_url}");
        }
    }

    #[test]
    fn authorization_fixture_decodes_a_sanitized_token() {
        let token = decode_authorization_body(AUTHORIZATION_FIXTURE, minute(1_722_470_400))
            .expect("valid authorization");
        assert_eq!(token.value.as_ref(), "sanitized-fixture-access-token");
        assert_eq!(token.expires_at, minute(2_000_000_000));
        assert!(token.is_usable_at(minute(1_999_999_969)));
        assert!(!token.is_usable_at(minute(1_999_999_970)));
    }

    #[test]
    fn stale_authorization_expiration_is_rejected() {
        let mut payload: Value =
            serde_json::from_slice(AUTHORIZATION_FIXTURE).expect("fixture is JSON");
        payload["d"]["expiration"] = json!(1_722_470_430);
        let body = serde_json::to_vec(&payload).expect("encode mutated fixture");
        let error = decode_authorization_body(&body, minute(1_722_470_400))
            .err()
            .expect("token expiring at the refresh margin must fail");
        assert_eq!(error.code, "chainlink_candle_authorization_invalid_body");
    }

    #[test]
    fn invalid_authorization_expiration_is_rejected() {
        let mut payload: Value =
            serde_json::from_slice(AUTHORIZATION_FIXTURE).expect("fixture is JSON");
        payload["d"]["expiration"] = json!("not-an-epoch-second");
        let body = serde_json::to_vec(&payload).expect("encode mutated fixture");
        let error = decode_authorization_body(&body, minute(1_722_470_400))
            .err()
            .expect("noninteger expiration must fail");
        assert_eq!(error.code, "chainlink_candle_authorization_invalid_body");
    }

    #[test]
    fn history_fixture_decodes_exact_closed_minute_facts() {
        let start = minute(1_722_470_400);
        let end = minute(1_722_470_520);
        let received_at = minute(1_722_470_640);
        let candles = decode_history_body(HISTORY_FIXTURE, start, end, received_at)
            .expect("valid history fixture");
        assert_eq!(candles.len(), 3);
        assert_eq!(candles[0].open_timestamp, start);
        assert_eq!(
            candles[0].close_timestamp,
            start + chrono::Duration::minutes(1)
        );
        assert_eq!(canonical_decimal(&candles[0].open_price), "64000.12");
        assert_eq!(canonical_decimal(&candles[0].high_price), "64010.5");
        assert_eq!(canonical_decimal(&candles[0].low_price), "63999.75");
        assert_eq!(canonical_decimal(&candles[0].close_price), "64008.25");
        assert_eq!(candles[0].payload_sha256.len(), 64);
        assert!(find_missing_ranges(&candles, start, end).is_empty());
    }

    #[test]
    fn scientific_wire_price_is_scaled_by_exactly_eighteen_decimals() {
        let value = json!(6.123456789e22);
        assert_eq!(
            canonical_decimal(&scaled_price(&value, "price").expect("valid scaled price")),
            "61234.56789"
        );
    }

    #[test]
    fn nonzero_provider_volume_is_rejected_instead_of_invented() {
        let mut payload: Value = serde_json::from_slice(HISTORY_FIXTURE).expect("fixture is JSON");
        payload["candles"][0][5] = json!(1);
        let body = serde_json::to_vec(&payload).expect("encode mutated fixture");
        let error = decode_history_body(
            &body,
            minute(1_722_470_400),
            minute(1_722_470_520),
            minute(1_722_470_640),
        )
        .expect_err("nonzero unsupported volume must fail");
        assert_eq!(error.code, "chainlink_candle_unsupported_volume");
    }

    #[test]
    fn history_rows_must_have_exactly_six_values() {
        let mut payload: Value = serde_json::from_slice(HISTORY_FIXTURE).expect("fixture is JSON");
        payload["candles"][0]
            .as_array_mut()
            .expect("row is an array")
            .push(json!("extra"));
        let body = serde_json::to_vec(&payload).expect("encode mutated fixture");
        let error = decode_history_body(
            &body,
            minute(1_722_470_400),
            minute(1_722_470_520),
            minute(1_722_470_640),
        )
        .expect_err("wide row must fail");
        assert_eq!(error.code, "chainlink_candle_wrong_row_shape");
    }

    #[test]
    fn missing_minutes_form_precise_ranges() {
        let start = minute(1_722_470_400);
        let end = minute(1_722_470_700);
        let received_at = minute(1_722_470_820);
        let mut candles =
            decode_history_body(HISTORY_FIXTURE, start, minute(1_722_470_520), received_at)
                .expect("valid fixture");
        candles.remove(1);
        assert_eq!(
            find_missing_ranges(&candles, start, end),
            vec![
                SourceGap {
                    start: minute(1_722_470_460),
                    end: minute(1_722_470_460),
                },
                SourceGap {
                    start: minute(1_722_470_580),
                    end,
                },
            ]
        );
    }

    #[test]
    fn long_downtime_gap_is_split_into_bounded_repair_windows() {
        let start = minute(1_722_470_400);
        let end = start + chrono::Duration::minutes(7);
        assert_eq!(
            split_gap(start, end, 3),
            vec![
                SourceGap {
                    start,
                    end: start + chrono::Duration::minutes(2),
                },
                SourceGap {
                    start: start + chrono::Duration::minutes(3),
                    end: start + chrono::Duration::minutes(5),
                },
                SourceGap {
                    start: start + chrono::Duration::minutes(6),
                    end,
                },
            ]
        );
    }

    #[test]
    fn checkpoint_v1_requires_a_whole_utc_minute() {
        assert!(CandleCheckpoint::from_value(&json!({
            "last_open_timestamp": "2024-08-01T00:00:00Z"
        }))
        .is_ok());
        assert!(CandleCheckpoint::from_value(&json!({
            "last_open_timestamp": "2024-08-01T00:00:01Z"
        }))
        .is_err());
        assert!(CandleCheckpoint::from_value(&json!({
            "last_open_timestamp": null,
            "unexpected": true
        }))
        .is_err());
    }

    #[test]
    fn publication_cutoff_excludes_the_current_and_trailing_closed_minute() {
        let now = Utc
            .timestamp_opt(1_722_470_539, 999_999_999)
            .single()
            .expect("timestamp");
        assert_eq!(
            latest_safely_published_open(now).expect("safe provider minute"),
            minute(1_722_470_400)
        );
    }

    #[test]
    fn publication_cutoff_advances_only_at_a_utc_minute_boundary() {
        let before_boundary = Utc
            .timestamp_opt(1_722_470_579, 999_999_999)
            .single()
            .expect("timestamp");
        let at_boundary = minute(1_722_470_580);
        assert_eq!(
            latest_safely_published_open(before_boundary).expect("safe provider minute"),
            minute(1_722_470_400)
        );
        assert_eq!(
            latest_safely_published_open(at_boundary).expect("safe provider minute"),
            minute(1_722_470_460)
        );
    }

    #[test]
    fn provider_publication_tail_is_never_classified_as_a_gap() {
        let start = minute(1_722_470_400);
        let now = Utc
            .timestamp_opt(1_722_470_699, 999_999_999)
            .single()
            .expect("timestamp");
        let safe_end = latest_safely_published_open(now).expect("safe provider minute");
        let candles = decode_history_body(HISTORY_FIXTURE, start, safe_end, now)
            .expect("fixture spans the safely published range");
        assert_eq!(safe_end, minute(1_722_470_520));
        assert!(find_missing_ranges(&candles, start, safe_end).is_empty());
        assert_eq!(
            find_missing_ranges(&candles, start, safe_end + chrono::Duration::minutes(1)),
            vec![SourceGap {
                start: minute(1_722_470_580),
                end: minute(1_722_470_580),
            }]
        );
    }

    #[test]
    fn only_a_newer_artifact_generation_is_rejected() {
        assert!(verify_artifact_generation(8, 7).is_err());
        assert!(verify_artifact_generation(7, 7).is_ok());
        assert!(verify_artifact_generation(6, 7).is_ok());
    }
}
