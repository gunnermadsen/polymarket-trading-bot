//! Continuous Binance Spot BTCUSDT aggregate-trade ingestion.
//!
//! The provider's aggregate-trade identifier is the logical identity. Capture
//! metadata is lineage only; it is deliberately excluded from factual equality.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::{FromRow, PgPool, Postgres, QueryBuilder, Transaction};
use tokio::time::{Instant, MissedTickBehavior};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    domain::{
        CaptureArtifact, IngesterProfile, IngesterStrategy, IngesterStrategyKey, StrategyError,
        StrategyErrorKind,
    },
    persistence::{
        ArtifactBatch, ArtifactRepository, GapRepository, NewCaptureArtifact, NewDataGap,
        ProfileRepository, StrategyDegradation, StrategyProgress,
    },
    runtime::{StrategyFactory, StrategyFactoryError},
};

pub const STRATEGY_KEY: IngesterStrategyKey =
    IngesterStrategyKey::BinanceSpotBtcusdtAggregateTrades;
pub const CONFIG_SCHEMA_VERSION: i32 = 1;
pub const CHECKPOINT_SCHEMA_VERSION: i32 = 1;

const SOURCE: &str = "binance_spot";
const SYMBOL: &str = "BTCUSDT";
const DEFAULT_WEBSOCKET_URL: &str = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade";
const DEFAULT_REST_BASE_URL: &str = "https://data-api.binance.vision";
const MAX_REST_BODY_BYTES: usize = 1_048_576;
const MAX_PROVIDER_CLOCK_SKEW: chrono::Duration = chrono::Duration::minutes(5);
const ALLOWED_WEBSOCKET_URLS: [&str; 3] = [
    DEFAULT_WEBSOCKET_URL,
    "wss://stream.binance.com:443/ws/btcusdt@aggTrade",
    "wss://data-stream.binance.vision/ws/btcusdt@aggTrade",
];
const ALLOWED_REST_BASE_URLS: [&str; 2] = [DEFAULT_REST_BASE_URL, "https://api.binance.com"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct BinanceSpotAggregateTradesConfig {
    pub websocket_url: String,
    pub rest_base_url: String,
    pub batch_size: usize,
    pub flush_interval_ms: u64,
    pub rest_page_limit: usize,
    pub recovery_overlap_records: i64,
    pub read_idle_timeout_ms: u64,
    pub reconnect_initial_delay_ms: u64,
    pub reconnect_max_delay_ms: u64,
    pub artifact_window_seconds: i64,
}

impl Default for BinanceSpotAggregateTradesConfig {
    fn default() -> Self {
        Self {
            websocket_url: DEFAULT_WEBSOCKET_URL.to_owned(),
            rest_base_url: DEFAULT_REST_BASE_URL.to_owned(),
            batch_size: 500,
            flush_interval_ms: 250,
            rest_page_limit: 1_000,
            recovery_overlap_records: 100,
            read_idle_timeout_ms: 40_000,
            reconnect_initial_delay_ms: 1_000,
            reconnect_max_delay_ms: 30_000,
            artifact_window_seconds: 3_600,
        }
    }
}

impl BinanceSpotAggregateTradesConfig {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let config = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::InvalidConfiguration(format!(
                "invalid Binance aggregate-trade config: {error}"
            ))
        })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), StrategyFactoryError> {
        let invalid = |message: &str| {
            StrategyFactoryError::InvalidConfiguration(format!(
                "Binance aggregate-trade config {message}"
            ))
        };
        if !ALLOWED_WEBSOCKET_URLS.contains(&self.websocket_url.as_str()) {
            return Err(invalid(
                "websocket_url is not an approved BTCUSDT aggTrade endpoint",
            ));
        }
        if !ALLOWED_REST_BASE_URLS.contains(&self.rest_base_url.as_str()) {
            return Err(invalid(
                "rest_base_url is not an approved Binance HTTPS endpoint",
            ));
        }
        if !(1..=4_000).contains(&self.batch_size) {
            return Err(invalid("batch_size must be between 1 and 4000"));
        }
        if !(25..=10_000).contains(&self.flush_interval_ms) {
            return Err(invalid("flush_interval_ms must be between 25 and 10000"));
        }
        if !(1..=1_000).contains(&self.rest_page_limit) {
            return Err(invalid("rest_page_limit must be between 1 and 1000"));
        }
        if !(1..=100_000).contains(&self.recovery_overlap_records) {
            return Err(invalid(
                "recovery_overlap_records must be between 1 and 100000",
            ));
        }
        if !(5_000..=300_000).contains(&self.read_idle_timeout_ms) {
            return Err(invalid(
                "read_idle_timeout_ms must be between 5000 and 300000",
            ));
        }
        if !(50..=60_000).contains(&self.reconnect_initial_delay_ms)
            || self.reconnect_max_delay_ms < self.reconnect_initial_delay_ms
            || self.reconnect_max_delay_ms > 300_000
        {
            return Err(invalid(
                "reconnect delays must be ordered between 50 and 300000 milliseconds",
            ));
        }
        if !(60..=86_400).contains(&self.artifact_window_seconds) {
            return Err(invalid(
                "artifact_window_seconds must be between 60 and 86400",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct BinanceSpotAggregateTradesFactory;

impl StrategyFactory for BinanceSpotAggregateTradesFactory {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    fn config_schema_version(&self) -> i32 {
        CONFIG_SCHEMA_VERSION
    }

    fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError> {
        BinanceSpotAggregateTradesConfig::from_value(config).map(|_| ())
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
                "config schema must be {CONFIG_SCHEMA_VERSION}"
            )));
        }
        if profile.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(StrategyFactoryError::Construction(format!(
                "checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}"
            )));
        }
        if profile.desired_generation <= 0 {
            return Err(StrategyFactoryError::Construction(
                "profile generation must be positive".to_owned(),
            ));
        }

        let config = BinanceSpotAggregateTradesConfig::from_value(&profile.config)?;
        let config_snapshot = serde_json::to_value(&config).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "failed to encode effective configuration: {error}"
            ))
        })?;
        let checkpoint = AggregateTradeCheckpoint::from_value(&profile.checkpoint)?;
        let lease_owner = profile.lease_owner.clone().ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease owner is missing".to_owned())
        })?;
        let lease_token = profile.lease_token.ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease token is missing".to_owned())
        })?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_millis(config.read_idle_timeout_ms))
            .user_agent("capitonic-market-data-ingester/0.1")
            .build()
            .map_err(|error| {
                StrategyFactoryError::Construction(format!(
                    "failed to build Binance HTTP client: {error}"
                ))
            })?;

        Ok(Box::new(BinanceSpotAggregateTradesStrategy {
            config,
            config_snapshot,
            profile_generation: profile.desired_generation,
            lease_owner: Arc::<str>::from(lease_owner),
            lease_token,
            initial_last_id: checkpoint.last_aggregate_trade_id,
            pool,
            client,
        }))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
struct AggregateTradeCheckpoint {
    last_aggregate_trade_id: Option<i64>,
}

impl AggregateTradeCheckpoint {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let checkpoint = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "invalid aggregate-trade checkpoint: {error}"
            ))
        })?;
        if checkpoint.last_aggregate_trade_id.is_some_and(|id| id < 0) {
            return Err(StrategyFactoryError::Construction(
                "aggregate-trade checkpoint cannot be negative".to_owned(),
            ));
        }
        Ok(checkpoint)
    }

    fn to_value(last_aggregate_trade_id: i64) -> Value {
        json!({ "last_aggregate_trade_id": last_aggregate_trade_id })
    }
}

pub struct BinanceSpotAggregateTradesStrategy {
    config: BinanceSpotAggregateTradesConfig,
    config_snapshot: Value,
    profile_generation: i64,
    lease_owner: Arc<str>,
    lease_token: Uuid,
    initial_last_id: Option<i64>,
    pool: PgPool,
    client: Client,
}

#[derive(Debug)]
struct AggregateRunState {
    last_id: Option<i64>,
    artifact: Option<CaptureArtifact>,
}

#[derive(Debug, Clone, PartialEq)]
struct AggregateTrade {
    aggregate_trade_id: i64,
    trade_timestamp: DateTime<Utc>,
    provider_available_at: Option<DateTime<Utc>>,
    received_at: DateTime<Utc>,
    price: Decimal,
    quantity: Decimal,
    first_trade_id: i64,
    last_trade_id: i64,
    buyer_maker: bool,
    best_match: bool,
    payload_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireAggregateTrade {
    #[serde(rename = "e")]
    event_type: String,
    #[serde(rename = "E")]
    event_time_ms: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "a")]
    aggregate_trade_id: i64,
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "q")]
    quantity: String,
    #[serde(rename = "f")]
    first_trade_id: i64,
    #[serde(rename = "l")]
    last_trade_id: i64,
    #[serde(rename = "T")]
    trade_time_ms: i64,
    #[serde(rename = "m")]
    buyer_maker: bool,
    #[serde(rename = "M")]
    best_match: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRestAggregateTrade {
    #[serde(rename = "a")]
    aggregate_trade_id: i64,
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "q")]
    quantity: String,
    #[serde(rename = "f")]
    first_trade_id: i64,
    #[serde(rename = "l")]
    last_trade_id: i64,
    #[serde(rename = "T")]
    trade_time_ms: i64,
    #[serde(rename = "m")]
    buyer_maker: bool,
    #[serde(rename = "M")]
    best_match: bool,
}

impl AggregateTrade {
    fn from_websocket(text: &str, received_at: DateTime<Utc>) -> Result<Self, StrategyError> {
        let wire = serde_json::from_str::<WireAggregateTrade>(text).map_err(|error| {
            source_error(
                "binance_aggregate_trade_invalid_message",
                format!("failed to decode Binance aggregate trade: {error}"),
            )
        })?;
        if wire.event_type != "aggTrade" || wire.symbol != SYMBOL {
            return Err(source_error(
                "binance_aggregate_trade_wrong_stream",
                "received a non-BTCUSDT aggregate-trade event",
            ));
        }
        let provider_available_at = timestamp_millis(wire.event_time_ms, "event time")?;
        let trade_timestamp = timestamp_millis(wire.trade_time_ms, "trade time")?;
        if provider_available_at < trade_timestamp {
            return Err(source_error(
                "binance_aggregate_trade_invalid_time_order",
                "provider event time precedes aggregate trade time",
            ));
        }
        Self::validated(
            wire.aggregate_trade_id,
            trade_timestamp,
            Some(provider_available_at),
            received_at,
            &wire.price,
            &wire.quantity,
            wire.first_trade_id,
            wire.last_trade_id,
            wire.buyer_maker,
            wire.best_match,
        )
    }

    fn from_rest(
        wire: WireRestAggregateTrade,
        received_at: DateTime<Utc>,
    ) -> Result<Self, StrategyError> {
        Self::validated(
            wire.aggregate_trade_id,
            timestamp_millis(wire.trade_time_ms, "trade time")?,
            None,
            received_at,
            &wire.price,
            &wire.quantity,
            wire.first_trade_id,
            wire.last_trade_id,
            wire.buyer_maker,
            wire.best_match,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn validated(
        aggregate_trade_id: i64,
        trade_timestamp: DateTime<Utc>,
        provider_available_at: Option<DateTime<Utc>>,
        received_at: DateTime<Utc>,
        price: &str,
        quantity: &str,
        first_trade_id: i64,
        last_trade_id: i64,
        buyer_maker: bool,
        best_match: bool,
    ) -> Result<Self, StrategyError> {
        if aggregate_trade_id < 0 || first_trade_id < 0 || last_trade_id < first_trade_id {
            return Err(source_error(
                "binance_aggregate_trade_invalid_identity",
                "aggregate-trade identifiers are negative or reversed",
            ));
        }
        if trade_timestamp > received_at + MAX_PROVIDER_CLOCK_SKEW
            || provider_available_at
                .is_some_and(|available_at| available_at > received_at + MAX_PROVIDER_CLOCK_SKEW)
        {
            return Err(source_error(
                "binance_aggregate_trade_future_timestamp",
                "aggregate-trade provider timestamp exceeds the allowed clock skew",
            ));
        }
        let price = positive_decimal(price, "price")?;
        let quantity = positive_decimal(quantity, "quantity")?;
        let mut trade = Self {
            aggregate_trade_id,
            trade_timestamp,
            provider_available_at,
            received_at,
            price,
            quantity,
            first_trade_id,
            last_trade_id,
            buyer_maker,
            best_match,
            payload_sha256: String::new(),
        };
        trade.payload_sha256 = trade.factual_payload_sha256();
        Ok(trade)
    }

    fn factual_payload_sha256(&self) -> String {
        let canonical = format!(
            "v1|source={SOURCE}|symbol={SYMBOL}|aggregate_trade_id={}|trade_timestamp_ms={}|price={}|quantity={}|first_trade_id={}|last_trade_id={}|buyer_maker={}|best_match={}",
            self.aggregate_trade_id,
            self.trade_timestamp.timestamp_millis(),
            canonical_decimal(&self.price),
            canonical_decimal(&self.quantity),
            self.first_trade_id,
            self.last_trade_id,
            self.buyer_maker,
            self.best_match,
        );
        sha256_hex(canonical.as_bytes())
    }

    fn factual_eq(&self, existing: &StoredAggregateTrade) -> bool {
        self.aggregate_trade_id == existing.aggregate_trade_id
            && self.trade_timestamp == existing.trade_timestamp
            && self.price == existing.price
            && self.quantity == existing.quantity
            && self.first_trade_id == existing.first_trade_id
            && self.last_trade_id == existing.last_trade_id
            && self.buyer_maker == existing.buyer_maker
            && self.best_match == existing.best_match
            && self.payload_sha256 == existing.payload_sha256
    }

    fn same_facts(&self, other: &Self) -> bool {
        self.aggregate_trade_id == other.aggregate_trade_id
            && self.trade_timestamp == other.trade_timestamp
            && self.price == other.price
            && self.quantity == other.quantity
            && self.first_trade_id == other.first_trade_id
            && self.last_trade_id == other.last_trade_id
            && self.buyer_maker == other.buyer_maker
            && self.best_match == other.best_match
            && self.payload_sha256 == other.payload_sha256
    }
}

#[derive(Debug, FromRow)]
struct StoredAggregateTrade {
    aggregate_trade_id: i64,
    trade_timestamp: DateTime<Utc>,
    price: Decimal,
    quantity: Decimal,
    first_trade_id: i64,
    last_trade_id: i64,
    buyer_maker: bool,
    best_match: bool,
    payload_sha256: String,
}

#[derive(Debug, FromRow)]
struct AggregateArtifactChecksumRow {
    aggregate_trade_id: i64,
    payload_sha256: String,
}

struct AggregateArtifactSeal {
    content_sha256: String,
    end_cursor: Option<String>,
}

#[async_trait]
impl IngesterStrategy for BinanceSpotAggregateTradesStrategy {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    async fn run(&self, shutdown: CancellationToken) -> Result<(), StrategyError> {
        let database_last_id = self.database_last_id().await?;
        if self.initial_last_id.is_some() && database_last_id.is_none() {
            return Err(integrity_error(
                "binance_aggregate_trade_checkpoint_without_fact",
                "aggregate-trade checkpoint exists without a durable source fact",
            ));
        }
        if let (Some(checkpoint), Some(database)) = (self.initial_last_id, database_last_id) {
            if checkpoint > database {
                return Err(integrity_error(
                    "binance_aggregate_trade_checkpoint_ahead",
                    format!(
                        "aggregate-trade checkpoint {checkpoint} is ahead of durable fact {database}"
                    ),
                ));
            }
        }

        let mut state = AggregateRunState {
            last_id: database_last_id.or(self.initial_last_id),
            artifact: None,
        };
        let mut reconnect_delay = self.config.reconnect_initial_delay_ms;

        loop {
            if shutdown.is_cancelled() {
                self.seal_artifact(&mut state, true).await?;
                return Ok(());
            }

            let cursor_before_session = state.last_id;
            match self.capture_session(&mut state, &shutdown).await {
                Ok(()) => {
                    self.seal_artifact(&mut state, true).await?;
                    return Ok(());
                }
                Err(error) if error.kind == StrategyErrorKind::Shutdown => {
                    self.seal_artifact(&mut state, true).await?;
                    return Ok(());
                }
                Err(error)
                    if matches!(
                        error.kind,
                        StrategyErrorKind::TransientSource | StrategyErrorKind::TransientDatabase
                    ) =>
                {
                    if state.last_id != cursor_before_session {
                        reconnect_delay = self.config.reconnect_initial_delay_ms;
                    }
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
                        .map_err(|database| {
                            database_error(
                                "binance_aggregate_trade_degraded_state_failed",
                                database,
                            )
                        })?;
                    if !marked {
                        if shutdown.is_cancelled() {
                            self.seal_artifact(&mut state, true).await?;
                            return Ok(());
                        }
                        return Err(lease_lost_error());
                    }
                    warn!(
                        strategy = %STRATEGY_KEY,
                        error_code = error.code,
                        error = %error,
                        reconnect_delay_ms = reconnect_delay,
                        "Binance aggregate-trade session will reconnect"
                    );
                    tokio::select! {
                        _ = shutdown.cancelled() => {
                            self.seal_artifact(&mut state, true).await?;
                            return Ok(());
                        }
                        _ = tokio::time::sleep(Duration::from_millis(reconnect_delay)) => {}
                    }
                    reconnect_delay = reconnect_delay
                        .saturating_mul(2)
                        .min(self.config.reconnect_max_delay_ms);
                }
                Err(error) => {
                    if let Err(seal_error) = self.seal_artifact(&mut state, false).await {
                        warn!(
                            strategy = %STRATEGY_KEY,
                            error = %seal_error,
                            "failed to seal aggregate-trade artifact after terminal failure"
                        );
                    }
                    return Err(error);
                }
            }
        }
    }
}

impl BinanceSpotAggregateTradesStrategy {
    async fn database_last_id(&self) -> Result<Option<i64>, StrategyError> {
        sqlx::query_scalar::<_, i64>(
            r#"
            SELECT aggregate_trade_id
            FROM market_data.binance_spot_btcusdt_aggregate_trades
            WHERE symbol = 'BTCUSDT'
            ORDER BY aggregate_trade_id DESC, trade_timestamp DESC
            LIMIT 1
            "#,
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| database_error("binance_aggregate_trade_cursor_read_failed", error))
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
        .map_err(|error| database_error("binance_aggregate_trade_lease_check_failed", error))?;
        if !current {
            return Err(lease_lost_error());
        }
        Ok(())
    }

    fn artifact_window(&self, timestamp: DateTime<Utc>) -> (DateTime<Utc>, DateTime<Utc>) {
        let seconds = timestamp.timestamp();
        let start_seconds = seconds
            .div_euclid(self.config.artifact_window_seconds)
            .saturating_mul(self.config.artifact_window_seconds);
        let start = Utc
            .timestamp_opt(start_seconds, 0)
            .single()
            .expect("aligned UTC artifact timestamp is representable");
        let end = Utc
            .timestamp_opt(
                start_seconds.saturating_add(self.config.artifact_window_seconds),
                0,
            )
            .single()
            .expect("bounded UTC artifact timestamp is representable");
        (start, end)
    }

    async fn ensure_artifact(
        &self,
        state: &mut AggregateRunState,
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
        if let Some(open) = repository.get_open(STRATEGY_KEY).await.map_err(|error| {
            database_error("binance_aggregate_trade_artifact_read_failed", error)
        })? {
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
                    "binance_aggregate_trade_artifact_config_conflict",
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
            database_error("binance_aggregate_trade_artifact_transaction_failed", error)
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
                    start_cursor: state.last_id.map(|id| id.to_string()),
                },
            )
            .await
            .map_err(|error| {
                database_error("binance_aggregate_trade_artifact_create_failed", error)
            })?;
        transaction.commit().await.map_err(|error| {
            database_error("binance_aggregate_trade_artifact_commit_failed", error)
        })?;
        let artifact_id = artifact.artifact_id;
        state.artifact = Some(artifact);
        info!(
            strategy = %STRATEGY_KEY,
            %artifact_id,
            %window_start,
            %window_end,
            "opened aggregate-trade capture artifact"
        );
        Ok(artifact_id)
    }

    async fn seal_artifact(
        &self,
        state: &mut AggregateRunState,
        allow_draining_generation: bool,
    ) -> Result<(), StrategyError> {
        if let Some(artifact) = state.artifact.as_ref() {
            verify_artifact_generation(artifact.profile_generation, self.profile_generation)?;
        }
        let Some(artifact) = state.artifact.as_ref().cloned() else {
            return Ok(());
        };
        let seal = self.artifact_seal(&artifact).await?;
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("binance_aggregate_trade_artifact_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, allow_draining_generation)
            .await?;
        let completed = ArtifactRepository::new(self.pool.clone())
            .complete_in(
                &mut transaction,
                artifact.artifact_id,
                &seal.content_sha256,
                seal.end_cursor
                    .as_deref()
                    .or(artifact.end_cursor.as_deref()),
            )
            .await
            .map_err(|error| {
                database_error("binance_aggregate_trade_artifact_complete_failed", error)
            })?;
        if completed.is_none() {
            return Err(integrity_error(
                "binance_aggregate_trade_artifact_not_open",
                format!(
                    "artifact {} was not open while sealing",
                    artifact.artifact_id
                ),
            ));
        }
        transaction.commit().await.map_err(|error| {
            database_error("binance_aggregate_trade_artifact_commit_failed", error)
        })?;
        state.artifact = None;
        info!(
            strategy = %STRATEGY_KEY,
            artifact_id = %artifact.artifact_id,
            record_count = artifact.record_count,
            content_sha256 = %seal.content_sha256,
            "sealed aggregate-trade capture artifact"
        );
        Ok(())
    }

    async fn artifact_seal(
        &self,
        artifact: &CaptureArtifact,
    ) -> Result<AggregateArtifactSeal, StrategyError> {
        let rows = sqlx::query_as::<_, AggregateArtifactChecksumRow>(
            r#"
            SELECT aggregate_trade_id, payload_sha256::text AS payload_sha256
            FROM market_data.binance_spot_btcusdt_aggregate_trades
            WHERE capture_artifact_id = $1 AND strategy_key = $2
            ORDER BY aggregate_trade_id
            "#,
        )
        .bind(artifact.artifact_id)
        .bind(STRATEGY_KEY.as_str())
        .fetch_all(&self.pool)
        .await
        .map_err(|error| database_error("binance_aggregate_trade_checksum_read_failed", error))?;
        if i64::try_from(rows.len()).ok() != Some(artifact.record_count) {
            return Err(integrity_error(
                "binance_aggregate_trade_artifact_count_mismatch",
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
            hash_field(&mut hasher, &row.aggregate_trade_id.to_string());
            hash_field(&mut hasher, &row.payload_sha256);
        }
        Ok(AggregateArtifactSeal {
            content_sha256: digest_hex(hasher.finalize()),
            end_cursor: rows.last().map(|row| row.aggregate_trade_id.to_string()),
        })
    }

    async fn complete_repair_artifact(
        &self,
        state: &mut AggregateRunState,
        gap_id: Uuid,
    ) -> Result<(), StrategyError> {
        if let Some(artifact) = state.artifact.as_ref() {
            verify_artifact_generation(artifact.profile_generation, self.profile_generation)?;
        }
        let artifact = state.artifact.as_ref().cloned().ok_or_else(|| {
            integrity_error(
                "binance_aggregate_trade_repair_artifact_missing",
                "gap repair completed without an open capture artifact",
            )
        })?;
        let seal = self.artifact_seal(&artifact).await?;
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("binance_aggregate_trade_artifact_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let completed = ArtifactRepository::new(self.pool.clone())
            .complete_in(
                &mut transaction,
                artifact.artifact_id,
                &seal.content_sha256,
                seal.end_cursor
                    .as_deref()
                    .or(artifact.end_cursor.as_deref()),
            )
            .await
            .map_err(|error| {
                database_error("binance_aggregate_trade_artifact_complete_failed", error)
            })?;
        if completed.is_none() {
            return Err(integrity_error(
                "binance_aggregate_trade_artifact_not_open",
                format!(
                    "artifact {} was not open while sealing",
                    artifact.artifact_id
                ),
            ));
        }
        let repaired = GapRepository::new(self.pool.clone())
            .mark_repaired_in(
                &mut transaction,
                gap_id,
                artifact.artifact_id,
                "binance_rest_overlap_recovered",
                Some("all missing aggregate-trade identifiers were verified"),
            )
            .await
            .map_err(|error| {
                database_error("binance_aggregate_trade_gap_complete_failed", error)
            })?;
        if repaired.is_none() {
            return Err(integrity_error(
                "binance_aggregate_trade_gap_not_repairing",
                format!("data gap {gap_id} was not repairing during completion"),
            ));
        }
        transaction.commit().await.map_err(|error| {
            database_error("binance_aggregate_trade_artifact_commit_failed", error)
        })?;
        state.artifact = None;
        info!(
            strategy = %STRATEGY_KEY,
            artifact_id = %artifact.artifact_id,
            record_count = artifact.record_count,
            content_sha256 = %seal.content_sha256,
            %gap_id,
            "sealed aggregate-trade repair artifact and resolved gap"
        );
        Ok(())
    }

    async fn persist_trades(
        &self,
        state: &mut AggregateRunState,
        trades: Vec<AggregateTrade>,
    ) -> Result<(), StrategyError> {
        if trades.is_empty() {
            return Ok(());
        }

        let mut unique = BTreeMap::<i64, AggregateTrade>::new();
        for trade in trades {
            match unique.get_mut(&trade.aggregate_trade_id) {
                Some(existing) if !existing.same_facts(&trade) => {
                    return Err(integrity_error(
                        "binance_aggregate_trade_batch_conflict",
                        format!(
                            "provider supplied conflicting payloads for aggregate trade {}",
                            trade.aggregate_trade_id
                        ),
                    ));
                }
                Some(existing)
                    if existing.provider_available_at.is_none()
                        && trade.provider_available_at.is_some() =>
                {
                    *existing = trade;
                }
                Some(_) => {}
                None => {
                    unique.insert(trade.aggregate_trade_id, trade);
                }
            }
        }

        let mut windows = BTreeMap::<DateTime<Utc>, Vec<AggregateTrade>>::new();
        for trade in unique.into_values() {
            let (window_start, _) = self.artifact_window(trade.received_at);
            windows.entry(window_start).or_default().push(trade);
        }
        for trades in windows.into_values() {
            self.persist_artifact_batch(state, trades).await?;
        }
        Ok(())
    }

    async fn persist_artifact_batch(
        &self,
        state: &mut AggregateRunState,
        mut trades: Vec<AggregateTrade>,
    ) -> Result<(), StrategyError> {
        trades.sort_by_key(|trade| trade.aggregate_trade_id);
        let artifact_id = self
            .ensure_artifact(
                state,
                trades
                    .first()
                    .expect("non-empty aggregate-trade batch")
                    .received_at,
            )
            .await?;
        let ids: Vec<i64> = trades
            .iter()
            .map(|trade| trade.aggregate_trade_id)
            .collect();
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("binance_aggregate_trade_transaction_begin_failed", error)
        })?;

        // Timescale requires the partition key in a physical unique index. A
        // transaction-scoped advisory lock preserves Binance's global logical
        // identity (symbol, aggregate_trade_id) across time chunks.
        sqlx::query(
            r#"
            SELECT pg_advisory_xact_lock(
              hashtextextended('binance_spot:BTCUSDT:' || logical_id::text, 0)
            )
            FROM unnest($1::bigint[]) AS logical_id
            ORDER BY logical_id
            "#,
        )
        .bind(&ids)
        .fetch_all(&mut *transaction)
        .await
        .map_err(|error| database_error("binance_aggregate_trade_lock_failed", error))?;

        let existing = self.load_existing_trades(&mut transaction, &ids).await?;
        let mut stored_by_id = BTreeMap::<i64, StoredAggregateTrade>::new();
        for stored in existing {
            if stored_by_id
                .insert(stored.aggregate_trade_id, stored)
                .is_some()
            {
                return Err(integrity_error(
                    "binance_aggregate_trade_duplicate_logical_identity",
                    "multiple durable rows share one Binance aggregate-trade identifier",
                ));
            }
        }

        let mut missing = Vec::new();
        for trade in &trades {
            if let Some(stored) = stored_by_id.get(&trade.aggregate_trade_id) {
                if !trade.factual_eq(stored) {
                    return Err(integrity_error(
                        "binance_aggregate_trade_immutable_conflict",
                        format!(
                            "durable aggregate trade {} conflicts with provider payload",
                            trade.aggregate_trade_id
                        ),
                    ));
                }
            } else {
                missing.push(trade);
            }
        }

        let inserted_ids = self
            .insert_missing_trades(&mut transaction, artifact_id, &missing)
            .await?;

        // Re-read while locks are held. This verifies ON CONFLICT replays and
        // catches any non-compliant writer that bypassed the logical-ID lock.
        let durable = self.load_existing_trades(&mut transaction, &ids).await?;
        let mut durable_by_id = BTreeMap::<i64, StoredAggregateTrade>::new();
        for stored in durable {
            if durable_by_id
                .insert(stored.aggregate_trade_id, stored)
                .is_some()
            {
                return Err(integrity_error(
                    "binance_aggregate_trade_duplicate_logical_identity",
                    "multiple durable rows share one Binance aggregate-trade identifier",
                ));
            }
        }
        for trade in &trades {
            let Some(stored) = durable_by_id.get(&trade.aggregate_trade_id) else {
                return Err(integrity_error(
                    "binance_aggregate_trade_insert_missing",
                    format!(
                        "aggregate trade {} was absent after insert",
                        trade.aggregate_trade_id
                    ),
                ));
            };
            if !trade.factual_eq(stored) {
                return Err(integrity_error(
                    "binance_aggregate_trade_immutable_conflict",
                    format!(
                        "durable aggregate trade {} conflicts with provider payload",
                        trade.aggregate_trade_id
                    ),
                ));
            }
        }

        let inserted: Vec<&AggregateTrade> = trades
            .iter()
            .filter(|trade| inserted_ids.contains(&trade.aggregate_trade_id))
            .collect();
        let mut artifact_after_commit = None;
        if !inserted.is_empty() {
            let artifact = ArtifactRepository::new(self.pool.clone())
                .record_batch_in(
                    &mut transaction,
                    artifact_id,
                    &ArtifactBatch {
                        inserted_record_count: inserted.len() as i64,
                        minimum_source_timestamp: inserted
                            .iter()
                            .map(|trade| trade.trade_timestamp)
                            .min(),
                        maximum_source_timestamp: inserted
                            .iter()
                            .map(|trade| trade.trade_timestamp)
                            .max(),
                        minimum_received_at: inserted.iter().map(|trade| trade.received_at).min(),
                        maximum_received_at: inserted.iter().map(|trade| trade.received_at).max(),
                        start_cursor: inserted
                            .first()
                            .map(|trade| trade.aggregate_trade_id.to_string()),
                        end_cursor: inserted
                            .last()
                            .map(|trade| trade.aggregate_trade_id.to_string()),
                    },
                )
                .await
                .map_err(|error| {
                    database_error("binance_aggregate_trade_artifact_progress_failed", error)
                })?
                .ok_or_else(|| {
                    integrity_error(
                        "binance_aggregate_trade_artifact_not_open",
                        format!("artifact {artifact_id} was not open during fact insert"),
                    )
                })?;
            artifact_after_commit = Some(artifact);
        }

        let batch_last_id = trades
            .last()
            .expect("non-empty aggregate-trade batch")
            .aggregate_trade_id;
        let checkpoint_id = state
            .last_id
            .map_or(batch_last_id, |id| id.max(batch_last_id));
        let last_source_timestamp = trades.iter().map(|trade| trade.trade_timestamp).max();
        let last_provider_available_at = trades
            .iter()
            .filter_map(|trade| trade.provider_available_at)
            .max();
        let progressed = ProfileRepository::new(self.pool.clone())
            .record_progress_in(
                &mut transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &StrategyProgress {
                    verified_record_count: trades.len() as i64,
                    checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
                    checkpoint: AggregateTradeCheckpoint::to_value(checkpoint_id),
                    last_source_event_at: last_source_timestamp,
                    last_provider_available_at,
                    source_watermark: last_source_timestamp,
                    availability_watermark: last_provider_available_at,
                },
            )
            .await
            .map_err(|error| {
                database_error("binance_aggregate_trade_profile_progress_failed", error)
            })?;
        if !progressed {
            return Err(lease_lost_error());
        }
        transaction.commit().await.map_err(|error| {
            database_error("binance_aggregate_trade_transaction_commit_failed", error)
        })?;
        if let Some(artifact) = artifact_after_commit {
            state.artifact = Some(artifact);
        }
        state.last_id = Some(checkpoint_id);
        Ok(())
    }

    async fn load_existing_trades(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        ids: &[i64],
    ) -> Result<Vec<StoredAggregateTrade>, StrategyError> {
        sqlx::query_as::<_, StoredAggregateTrade>(
            r#"
            SELECT aggregate_trade_id, trade_timestamp, price, quantity,
                   first_trade_id, last_trade_id, buyer_maker, best_match,
                   payload_sha256::text AS payload_sha256
            FROM market_data.binance_spot_btcusdt_aggregate_trades
            WHERE symbol = 'BTCUSDT' AND aggregate_trade_id = ANY($1::bigint[])
            ORDER BY aggregate_trade_id, trade_timestamp
            "#,
        )
        .bind(ids)
        .fetch_all(&mut **transaction)
        .await
        .map_err(|error| database_error("binance_aggregate_trade_fact_read_failed", error))
    }

    async fn insert_missing_trades(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        missing: &[&AggregateTrade],
    ) -> Result<BTreeSet<i64>, StrategyError> {
        if missing.is_empty() {
            return Ok(BTreeSet::new());
        }
        let mut builder = QueryBuilder::<Postgres>::new(
            r#"
            INSERT INTO market_data.binance_spot_btcusdt_aggregate_trades (
              source, symbol, aggregate_trade_id, trade_timestamp,
              provider_available_at, received_at, price, quantity,
              first_trade_id, last_trade_id, buyer_maker, best_match,
              payload_sha256, strategy_key, capture_artifact_id
            )
            "#,
        );
        builder.push_values(missing, |mut row, trade| {
            row.push_bind(SOURCE)
                .push_bind(SYMBOL)
                .push_bind(trade.aggregate_trade_id)
                .push_bind(trade.trade_timestamp)
                .push_bind(trade.provider_available_at)
                .push_bind(trade.received_at)
                .push_bind(trade.price)
                .push_bind(trade.quantity)
                .push_bind(trade.first_trade_id)
                .push_bind(trade.last_trade_id)
                .push_bind(trade.buyer_maker)
                .push_bind(trade.best_match)
                .push_bind(&trade.payload_sha256)
                .push_bind(STRATEGY_KEY.as_str())
                .push_bind(artifact_id);
        });
        builder.push(
            " ON CONFLICT (symbol, trade_timestamp, aggregate_trade_id) DO NOTHING \
             RETURNING aggregate_trade_id",
        );
        let inserted = builder
            .build_query_scalar::<i64>()
            .fetch_all(&mut **transaction)
            .await
            .map_err(|error| database_error("binance_aggregate_trade_fact_insert_failed", error))?;
        Ok(inserted.into_iter().collect())
    }

    async fn capture_session(
        &self,
        state: &mut AggregateRunState,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        self.recover_overlap(state, shutdown).await?;
        if shutdown.is_cancelled() {
            return Err(shutdown_error());
        }

        let connection = tokio::select! {
            _ = shutdown.cancelled() => return Err(shutdown_error()),
            result = tokio::time::timeout(
                Duration::from_millis(self.config.read_idle_timeout_ms),
                connect_async(&self.config.websocket_url),
            ) => result,
        };
        let (mut socket, _) = connection
            .map_err(|_| {
                source_error(
                    "binance_aggregate_trade_connect_timeout",
                    "timed out connecting to Binance aggregate-trade websocket",
                )
            })?
            .map_err(|error| {
                source_error(
                    "binance_aggregate_trade_connect_failed",
                    format!("failed to connect to Binance aggregate-trade websocket: {error}"),
                )
            })?;
        info!(
            strategy = %STRATEGY_KEY,
            last_aggregate_trade_id = ?state.last_id,
            "connected to Binance aggregate-trade websocket"
        );

        let mut pending = Vec::<AggregateTrade>::with_capacity(self.config.batch_size);
        let mut flush = tokio::time::interval(Duration::from_millis(self.config.flush_interval_ms));
        flush.set_missed_tick_behavior(MissedTickBehavior::Skip);
        flush.tick().await;
        let idle = tokio::time::sleep(Duration::from_millis(self.config.read_idle_timeout_ms));
        tokio::pin!(idle);

        loop {
            tokio::select! {
                _ = shutdown.cancelled() => {
                    let flush_result = self
                        .persist_trades(state, std::mem::take(&mut pending))
                        .await;
                    let _ = socket.close(None).await;
                    if let Err(error) = flush_result {
                        if error.kind != StrategyErrorKind::LeaseLost {
                            return Err(error);
                        }
                    }
                    return Err(shutdown_error());
                }
                _ = flush.tick() => {
                    self.persist_trades(state, std::mem::take(&mut pending)).await?;
                }
                _ = &mut idle => {
                    self.persist_trades(state, std::mem::take(&mut pending)).await?;
                    return Err(source_error(
                        "binance_aggregate_trade_read_idle",
                        "Binance aggregate-trade websocket exceeded its read-idle timeout",
                    ));
                }
                next = socket.next() => {
                    idle.as_mut().reset(
                        Instant::now() + Duration::from_millis(self.config.read_idle_timeout_ms),
                    );
                    let message = match next {
                        Some(Ok(message)) => message,
                        Some(Err(error)) => {
                            self.persist_trades(state, std::mem::take(&mut pending)).await?;
                            return Err(source_error(
                                "binance_aggregate_trade_websocket_read_failed",
                                format!("failed to read Binance aggregate-trade websocket: {error}"),
                            ));
                        }
                        None => {
                            self.persist_trades(state, std::mem::take(&mut pending)).await?;
                            return Err(source_error(
                                "binance_aggregate_trade_websocket_ended",
                                "Binance aggregate-trade websocket ended",
                            ));
                        }
                    };
                    match message {
                        Message::Text(text) => {
                            let received_at = Utc::now();
                            let trade = match AggregateTrade::from_websocket(text.as_ref(), received_at) {
                                Ok(trade) => trade,
                                Err(error) => {
                                    self.persist_trades(state, std::mem::take(&mut pending)).await?;
                                    return Err(error);
                                }
                            };
                            let last_seen = pending
                                .last()
                                .map(|trade| trade.aggregate_trade_id)
                                .or(state.last_id);
                            if let Some(last_seen) = last_seen {
                                if trade.aggregate_trade_id > last_seen.saturating_add(1) {
                                    self.persist_trades(state, std::mem::take(&mut pending)).await?;
                                    self.repair_gap(
                                        state,
                                        last_seen.saturating_add(1),
                                        trade.aggregate_trade_id.saturating_sub(1),
                                        shutdown,
                                    )
                                    .await?;
                                }
                            }
                            pending.push(trade);
                            if pending.len() >= self.config.batch_size {
                                self.persist_trades(state, std::mem::take(&mut pending)).await?;
                            }
                        }
                        Message::Ping(payload) => {
                            if let Err(error) = socket.send(Message::Pong(payload)).await {
                                self.persist_trades(state, std::mem::take(&mut pending)).await?;
                                return Err(source_error(
                                    "binance_aggregate_trade_pong_failed",
                                    format!("failed to answer Binance websocket ping: {error}"),
                                ));
                            }
                        }
                        Message::Pong(_) => {}
                        Message::Close(frame) => {
                            self.persist_trades(state, std::mem::take(&mut pending)).await?;
                            return Err(source_error(
                                "binance_aggregate_trade_websocket_closed",
                                format!("Binance aggregate-trade websocket closed: {frame:?}"),
                            ));
                        }
                        Message::Binary(_) | Message::Frame(_) => {
                            self.persist_trades(state, std::mem::take(&mut pending)).await?;
                            return Err(source_error(
                                "binance_aggregate_trade_unexpected_frame",
                                "received an unexpected Binance websocket frame",
                            ));
                        }
                    }
                }
            }
        }
    }

    async fn recover_overlap(
        &self,
        state: &mut AggregateRunState,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        let original_cursor = state.last_id;
        let from_id = original_cursor.map(|last| {
            last.saturating_sub(self.config.recovery_overlap_records.saturating_sub(1))
                .max(0)
        });
        let first_limit = if from_id.is_none() {
            usize::try_from(self.config.recovery_overlap_records)
                .unwrap_or(self.config.rest_page_limit)
                .min(self.config.rest_page_limit)
        } else {
            self.config.rest_page_limit
        };
        let mut next_from = from_id;
        loop {
            let page = self
                .fetch_rest_page(next_from, first_limit, shutdown)
                .await?;
            if page.is_empty() {
                break;
            }
            validate_aggregate_page(&page)?;
            let page_len = page.len();
            let page_last = page
                .last()
                .expect("non-empty aggregate-trade REST page")
                .aggregate_trade_id;
            self.persist_trades(state, page).await?;
            if page_len < first_limit {
                break;
            }
            next_from = Some(page_last.checked_add(1).ok_or_else(|| {
                integrity_error(
                    "binance_aggregate_trade_cursor_overflow",
                    "aggregate-trade identifier overflowed during REST recovery",
                )
            })?);
        }
        Ok(())
    }

    async fn repair_gap(
        &self,
        state: &mut AggregateRunState,
        start_id: i64,
        end_id: i64,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        if start_id > end_id {
            return Ok(());
        }
        let artifact_id = self.ensure_artifact(state, Utc::now()).await?;
        let gap = NewDataGap {
            strategy_key: STRATEGY_KEY,
            detected_artifact_id: Some(artifact_id),
            gap_kind: "source_sequence".to_owned(),
            reason_code: "binance_aggregate_trade_id_jump".to_owned(),
            reason_message: Some(format!(
                "missing Binance aggregate-trade identifiers {start_id} through {end_id}"
            )),
            source_time_start: None,
            source_time_end: None,
            start_cursor: Some(start_id.to_string()),
            end_cursor: Some(end_id.to_string()),
        };
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("binance_aggregate_trade_gap_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let detection = GapRepository::new(self.pool.clone())
            .detect_in(&mut transaction, &gap)
            .await
            .map_err(|error| database_error("binance_aggregate_trade_gap_detect_failed", error))?;
        let degraded = ProfileRepository::new(self.pool.clone())
            .mark_degraded_in(
                &mut transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &StrategyDegradation {
                    reason_code: "binance_aggregate_trade_gap_repair".to_owned(),
                    reason_message: format!(
                        "repairing aggregate-trade identifiers {start_id} through {end_id}"
                    ),
                },
            )
            .await
            .map_err(|error| database_error("binance_aggregate_trade_gap_health_failed", error))?;
        if !degraded {
            return Err(lease_lost_error());
        }
        let repairing_gap = GapRepository::new(self.pool.clone())
            .begin_repair_in(&mut transaction, detection.gap.gap_id)
            .await
            .map_err(|error| database_error("binance_aggregate_trade_gap_begin_failed", error))?
            .unwrap_or_else(|| detection.gap.clone());
        transaction.commit().await.map_err(|error| {
            database_error(
                "binance_aggregate_trade_gap_transaction_commit_failed",
                error,
            )
        })?;

        let recovery = self
            .recover_exact_range(state, start_id, end_id, shutdown)
            .await;
        match recovery {
            Ok(()) => {
                self.complete_repair_artifact(state, detection.gap.gap_id)
                    .await?;
                Ok(())
            }
            Err(error)
                if error.code == "binance_aggregate_trade_range_unavailable"
                    && repairing_gap.repair_attempts >= 3 =>
            {
                let mut transaction = self.pool.begin().await.map_err(|database| {
                    database_error(
                        "binance_aggregate_trade_gap_terminal_transaction_failed",
                        database,
                    )
                })?;
                self.assert_lease_in(&mut transaction, false).await?;
                let terminal = GapRepository::new(self.pool.clone())
                    .mark_unrecoverable_in(
                        &mut transaction,
                        detection.gap.gap_id,
                        "binance_rest_range_unavailable_after_retries",
                        Some("Binance REST omitted the missing aggregate trade on three repair attempts"),
                    )
                    .await
                    .map_err(|database| {
                        database_error("binance_aggregate_trade_gap_unrecoverable_failed", database)
                    })?;
                if terminal.is_none() {
                    return Err(integrity_error(
                        "binance_aggregate_trade_gap_not_repairing",
                        format!(
                            "data gap {} was not open during terminalization",
                            detection.gap.gap_id
                        ),
                    ));
                }
                transaction.commit().await.map_err(|database| {
                    database_error(
                        "binance_aggregate_trade_gap_terminal_commit_failed",
                        database,
                    )
                })?;
                Err(integrity_error(
                    "binance_aggregate_trade_gap_unrecoverable",
                    format!(
                        "aggregate-trade gap {start_id} through {end_id} remained unavailable after {} attempts",
                        repairing_gap.repair_attempts
                    ),
                ))
            }
            Err(error) => Err(error),
        }
    }

    async fn recover_exact_range(
        &self,
        state: &mut AggregateRunState,
        start_id: i64,
        end_id: i64,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        let mut expected = start_id;
        while expected <= end_id {
            let remaining = end_id
                .checked_sub(expected)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| {
                    integrity_error(
                        "binance_aggregate_trade_range_overflow",
                        "aggregate-trade recovery range overflowed",
                    )
                })?;
            let limit = usize::try_from(remaining)
                .unwrap_or(self.config.rest_page_limit)
                .min(self.config.rest_page_limit);
            let page = self
                .fetch_rest_page(Some(expected), limit, shutdown)
                .await?;
            if page.is_empty()
                || page
                    .first()
                    .is_none_or(|trade| trade.aggregate_trade_id != expected)
            {
                return Err(source_error(
                    "binance_aggregate_trade_range_unavailable",
                    format!("Binance REST did not return aggregate trade {expected}"),
                ));
            }
            validate_aggregate_page(&page)?;
            for pair in page.windows(2) {
                if pair[1].aggregate_trade_id != pair[0].aggregate_trade_id.saturating_add(1) {
                    return Err(source_error(
                        "binance_aggregate_trade_range_unavailable",
                        format!(
                            "Binance REST skipped aggregate trade {}",
                            pair[0].aggregate_trade_id.saturating_add(1)
                        ),
                    ));
                }
            }
            let last = page
                .last()
                .expect("non-empty exact recovery page")
                .aggregate_trade_id;
            self.persist_trades(state, page).await?;
            if last >= end_id {
                break;
            }
            expected = last.checked_add(1).ok_or_else(|| {
                integrity_error(
                    "binance_aggregate_trade_cursor_overflow",
                    "aggregate-trade identifier overflowed during gap repair",
                )
            })?;
        }
        Ok(())
    }

    async fn fetch_rest_page(
        &self,
        from_id: Option<i64>,
        limit: usize,
        shutdown: &CancellationToken,
    ) -> Result<Vec<AggregateTrade>, StrategyError> {
        let url = format!(
            "{}/api/v3/aggTrades",
            self.config.rest_base_url.trim_end_matches('/')
        );
        let mut parameters = vec![("symbol", SYMBOL.to_owned()), ("limit", limit.to_string())];
        if let Some(from_id) = from_id {
            parameters.push(("fromId", from_id.to_string()));
        }
        let request = self.client.get(url).query(&parameters).send();
        let response = tokio::select! {
            _ = shutdown.cancelled() => return Err(shutdown_error()),
            response = request => response,
        }
        .map_err(|error| {
            source_error(
                "binance_aggregate_trade_rest_request_failed",
                format!("Binance aggregate-trade REST request failed: {error}"),
            )
        })?;
        if !response.status().is_success() {
            return Err(source_error(
                "binance_aggregate_trade_rest_status",
                format!(
                    "Binance aggregate-trade REST returned HTTP {}",
                    response.status()
                ),
            ));
        }
        let bytes = read_bounded_rest_body(response, shutdown).await?;
        let wire =
            serde_json::from_slice::<Vec<WireRestAggregateTrade>>(&bytes).map_err(|error| {
                source_error(
                    "binance_aggregate_trade_rest_invalid_body",
                    format!("failed to decode Binance aggregate-trade REST body: {error}"),
                )
            })?;
        if wire.len() > limit {
            return Err(source_error(
                "binance_aggregate_trade_rest_limit_exceeded",
                format!(
                    "Binance aggregate-trade REST returned {} rows for limit {limit}",
                    wire.len()
                ),
            ));
        }
        let received_at = Utc::now();
        wire.into_iter()
            .map(|trade| AggregateTrade::from_rest(trade, received_at))
            .collect()
    }
}

fn validate_aggregate_page(page: &[AggregateTrade]) -> Result<(), StrategyError> {
    for pair in page.windows(2) {
        if pair[1].aggregate_trade_id <= pair[0].aggregate_trade_id {
            return Err(source_error(
                "binance_aggregate_trade_rest_not_ordered",
                "Binance aggregate-trade REST rows were not strictly ordered by identifier",
            ));
        }
    }
    Ok(())
}

async fn read_bounded_rest_body(
    response: reqwest::Response,
    shutdown: &CancellationToken,
) -> Result<Vec<u8>, StrategyError> {
    let capacity = response
        .content_length()
        .and_then(|length| usize::try_from(length).ok())
        .unwrap_or(0)
        .min(MAX_REST_BODY_BYTES);
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
                "binance_aggregate_trade_rest_body_failed",
                format!("failed to stream Binance aggregate-trade REST body: {error}"),
            )
        })?;
        let next_len = body.len().checked_add(chunk.len()).ok_or_else(|| {
            source_error(
                "binance_aggregate_trade_rest_body_too_large",
                "Binance aggregate-trade REST body length overflowed",
            )
        })?;
        if next_len > MAX_REST_BODY_BYTES {
            return Err(source_error(
                "binance_aggregate_trade_rest_body_too_large",
                format!("Binance aggregate-trade REST body exceeds {MAX_REST_BODY_BYTES} bytes"),
            ));
        }
        body.extend_from_slice(&chunk);
    }
}

fn timestamp_millis(value: i64, field: &str) -> Result<DateTime<Utc>, StrategyError> {
    if value < 0 {
        return Err(source_error(
            "binance_aggregate_trade_invalid_timestamp",
            format!("Binance aggregate-trade {field} is negative"),
        ));
    }
    Utc.timestamp_millis_opt(value).single().ok_or_else(|| {
        source_error(
            "binance_aggregate_trade_invalid_timestamp",
            format!("Binance aggregate-trade {field} is out of range"),
        )
    })
}

fn positive_decimal(value: &str, field: &str) -> Result<Decimal, StrategyError> {
    validate_decimal_wire(value, field)?;
    let parsed = value.parse::<Decimal>().map_err(|error| {
        source_error(
            "binance_aggregate_trade_invalid_decimal",
            format!("Binance aggregate-trade {field} is invalid: {error}"),
        )
    })?;
    if parsed <= Decimal::ZERO {
        return Err(source_error(
            "binance_aggregate_trade_nonpositive_decimal",
            format!("Binance aggregate-trade {field} must be positive"),
        ));
    }
    validate_numeric_storage_bound(&parsed, field)?;
    Ok(parsed)
}

fn validate_decimal_wire(value: &str, field: &str) -> Result<(), StrategyError> {
    let mut dots = 0_u8;
    if value.is_empty()
        || value.starts_with('.')
        || value.ends_with('.')
        || value.bytes().any(|byte| {
            if byte == b'.' {
                dots = dots.saturating_add(1);
                dots > 1
            } else {
                !byte.is_ascii_digit()
            }
        })
    {
        return Err(source_error(
            "binance_aggregate_trade_invalid_decimal",
            format!("Binance aggregate-trade {field} is not a plain unsigned decimal"),
        ));
    }
    Ok(())
}

fn validate_numeric_storage_bound(value: &Decimal, field: &str) -> Result<(), StrategyError> {
    let canonical = canonical_decimal(value);
    let integer_digits = canonical
        .split_once('.')
        .map_or(canonical.as_str(), |(integer, _)| integer)
        .trim_start_matches('0')
        .len()
        .max(1);
    if value.normalize().scale() > 10 || integer_digits > 20 {
        return Err(source_error(
            "binance_aggregate_trade_decimal_out_of_range",
            format!("Binance aggregate-trade {field} exceeds numeric(30,10)"),
        ));
    }
    Ok(())
}

fn canonical_decimal(value: &Decimal) -> String {
    value.normalize().to_string()
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
        "binance_aggregate_trade_lease_lost",
        "aggregate-trade profile lease was lost before progress committed",
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
        "binance_aggregate_trade_shutdown",
        "aggregate-trade strategy shutdown requested",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const WEBSOCKET_FIXTURE: &str =
        include_str!("../../../tests/fixtures/binance/aggregate_trade_websocket_v1.json");
    const REST_FIXTURE: &str =
        include_str!("../../../tests/fixtures/binance/aggregate_trade_rest_v1.json");

    #[test]
    fn default_config_matches_the_bootstrap_profile() {
        let config = BinanceSpotAggregateTradesConfig::default();
        assert_eq!(config.websocket_url, DEFAULT_WEBSOCKET_URL);
        assert_eq!(config.rest_base_url, DEFAULT_REST_BASE_URL);
        assert_eq!(config.batch_size, 500);
        assert_eq!(config.flush_interval_ms, 250);
        assert_eq!(config.rest_page_limit, 1_000);
        assert_eq!(config.recovery_overlap_records, 100);
        assert_eq!(config.read_idle_timeout_ms, 40_000);
        assert_eq!(config.reconnect_initial_delay_ms, 1_000);
        assert_eq!(config.reconnect_max_delay_ms, 30_000);
        assert_eq!(config.artifact_window_seconds, 3_600);
        config.validate().expect("default config is valid");
    }

    #[test]
    fn config_rejects_unknown_fields_and_unapproved_endpoints() {
        let unknown = json!({ "unknown": true });
        assert!(BinanceSpotAggregateTradesConfig::from_value(&unknown).is_err());

        let mut config = serde_json::to_value(BinanceSpotAggregateTradesConfig::default())
            .expect("serialize config");
        config["rest_base_url"] = json!("https://example.invalid");
        assert!(BinanceSpotAggregateTradesConfig::from_value(&config).is_err());
    }

    #[test]
    fn websocket_fixture_decodes_source_native_identity() {
        let received_at = Utc.timestamp_millis_opt(1_722_470_400_130).unwrap();
        let trade = AggregateTrade::from_websocket(WEBSOCKET_FIXTURE, received_at)
            .expect("valid aggregate trade");
        assert_eq!(trade.aggregate_trade_id, 305_419_896);
        assert_eq!(trade.first_trade_id, 427_587_855);
        assert_eq!(trade.last_trade_id, 427_587_857);
        assert!(!trade.buyer_maker);
        assert!(trade.best_match);
        assert_eq!(trade.payload_sha256.len(), 64);
    }

    #[test]
    fn rest_and_websocket_have_identical_factual_hashes() {
        let received_at = Utc.timestamp_millis_opt(1_722_470_400_130).unwrap();
        let websocket = AggregateTrade::from_websocket(WEBSOCKET_FIXTURE, received_at)
            .expect("valid aggregate trade");
        let wire = serde_json::from_str::<Vec<WireRestAggregateTrade>>(REST_FIXTURE)
            .expect("valid REST fixture");
        let rest = AggregateTrade::from_rest(
            wire.into_iter().nth(1).expect("second REST row"),
            received_at + chrono::Duration::seconds(1),
        )
        .expect("valid REST aggregate trade");
        assert!(websocket.same_facts(&rest));
        assert_eq!(websocket.payload_sha256, rest.payload_sha256);
        assert!(rest.provider_available_at.is_none());
    }

    #[test]
    fn websocket_wire_shape_is_strict() {
        let mut value = serde_json::from_str::<Value>(WEBSOCKET_FIXTURE).expect("fixture JSON");
        value["unexpected"] = json!(1);
        let received_at = Utc.timestamp_millis_opt(1_722_470_400_130).unwrap();
        let error = AggregateTrade::from_websocket(&value.to_string(), received_at)
            .expect_err("unknown provider field must fail");
        assert_eq!(error.code, "binance_aggregate_trade_invalid_message");
    }

    #[test]
    fn rest_page_requires_strictly_increasing_ids() {
        let received_at = Utc.timestamp_millis_opt(1_722_470_400_130).unwrap();
        let wire = serde_json::from_str::<Vec<WireRestAggregateTrade>>(REST_FIXTURE)
            .expect("valid REST fixture");
        let mut trades: Vec<_> = wire
            .into_iter()
            .map(|row| AggregateTrade::from_rest(row, received_at).expect("valid REST row"))
            .collect();
        validate_aggregate_page(&trades).expect("ascending fixture");
        trades.reverse();
        assert!(validate_aggregate_page(&trades).is_err());
    }

    #[test]
    fn decimal_wire_values_are_plain_and_fit_the_fact_column() {
        assert!(positive_decimal("64000.12000000", "price").is_ok());
        assert!(positive_decimal("6.4e4", "price").is_err());
        assert!(positive_decimal("1.00000000001", "price").is_err());
    }

    #[test]
    fn an_older_generation_cannot_complete_a_newer_artifact() {
        assert!(verify_artifact_generation(8, 7).is_err());
        assert!(verify_artifact_generation(7, 7).is_ok());
        assert!(verify_artifact_generation(6, 7).is_ok());
    }
}
