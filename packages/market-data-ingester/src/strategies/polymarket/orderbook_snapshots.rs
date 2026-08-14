//! Continuous public Polymarket CLOB BTC five-minute order-book capture.
//!
//! The strategy independently discovers the previous, current, and successor
//! contracts from Gamma, reconstructs each subscribed token book from the
//! public market websocket, and persists only bounded factual samples. It does
//! not depend on the trading bot crate and intentionally contains no features,
//! labels, execution inputs, or model-delay assumptions.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt::Write as _,
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use chrono::{DateTime, TimeDelta, TimeZone, Utc};
use futures_util::{future::join_all, SinkExt, StreamExt};
use reqwest::{redirect::Policy, Client, StatusCode};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use sqlx::{FromRow, PgPool, Postgres, Transaction};
use tokio::{sync::mpsc, task::JoinHandle, time::Instant};
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{protocol::WebSocketConfig, Message},
};
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
    IngesterStrategyKey::PolymarketBtcFiveMinuteOrderbooks;
pub const CONFIG_SCHEMA_VERSION: i32 = 1;
pub const CHECKPOINT_SCHEMA_VERSION: i32 = 1;

const SOURCE: &str = "polymarket_clob_market";
const SERIES_SLUG: &str = "btc-up-or-down-5m";
const EVENT_SLUG_PREFIX: &str = "btc-updown-5m-";
const MARKET_INTERVAL_SECONDS: i64 = 300;
const DEFAULT_WEBSOCKET_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const DEFAULT_GAMMA_API_URL: &str = "https://gamma-api.polymarket.com";
const SAMPLING_POLICY_VERSION: &str = "polymarket-clob-btc-5m-orderbook-top-n-v1";
const SAMPLING_SELECTION: &str = "latest_valid_subscribed_market_book_at_aligned_wall_clock_slot";

const MAX_GAMMA_RESPONSE_BYTES: usize = 512 * 1024;
const MAX_WEBSOCKET_FRAME_BYTES: usize = 2 * 1024 * 1024;
const MAX_MESSAGES_PER_FRAME: usize = 64;
const MAX_CHANGES_PER_MESSAGE: usize = 20_000;
const MAX_IDENTIFIER_BYTES: usize = 512;
const MAX_SOURCE_HASH_BYTES: usize = 256;
const MAX_NUMERIC_BYTES: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct PolymarketBtcFiveMinuteOrderbooksConfig {
    pub websocket_url: String,
    pub gamma_api_url: String,
    pub sample_interval_ms: u64,
    pub top_n: usize,
    pub gamma_refresh_ms: u64,
    pub lookback_windows: usize,
    pub lookahead_windows: usize,
    pub successor_lead_ms: u64,
    pub contract_grace_ms: u64,
    pub connect_timeout_ms: u64,
    pub bootstrap_timeout_ms: u64,
    pub read_timeout_ms: u64,
    pub ping_interval_ms: u64,
    pub pong_timeout_ms: u64,
    pub reconnect_initial_ms: u64,
    pub reconnect_max_ms: u64,
    pub max_levels_per_side: usize,
    pub artifact_window_seconds: i64,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct OrderbookCheckpoint {
    schema_version: Option<i32>,
    sampled_at: Option<DateTime<Utc>>,
    connection_epoch: Option<Uuid>,
    artifact_id: Option<Uuid>,
    sampling_policy_sha256: Option<String>,
    books: Vec<OrderbookCheckpointBook>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OrderbookCheckpointBook {
    market_id: String,
    token_id: String,
    source_timestamp: DateTime<Utc>,
    ingest_sequence: i64,
    payload_sha256: String,
}

impl OrderbookCheckpoint {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let checkpoint = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "invalid Polymarket orderbook checkpoint: {error}"
            ))
        })?;
        if checkpoint
            .schema_version
            .is_some_and(|version| version != CHECKPOINT_SCHEMA_VERSION)
        {
            return Err(StrategyFactoryError::Construction(format!(
                "Polymarket orderbook checkpoint payload schema must be {CHECKPOINT_SCHEMA_VERSION}"
            )));
        }
        let metadata_count = [
            checkpoint.sampled_at.is_some(),
            checkpoint.connection_epoch.is_some(),
            checkpoint.artifact_id.is_some(),
            checkpoint.sampling_policy_sha256.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count();
        if metadata_count != 0 && metadata_count != 4 {
            return Err(StrategyFactoryError::Construction(
                "Polymarket orderbook checkpoint metadata must be wholly present or absent"
                    .to_owned(),
            ));
        }
        if (metadata_count == 0
            && (checkpoint.schema_version.is_some() || !checkpoint.books.is_empty()))
            || (metadata_count == 4 && checkpoint.schema_version != Some(CHECKPOINT_SCHEMA_VERSION))
        {
            return Err(StrategyFactoryError::Construction(
                "Polymarket orderbook checkpoint must be empty or a complete schema-v1 payload"
                    .to_owned(),
            ));
        }
        if checkpoint.books.len() > 6
            || checkpoint.books.iter().any(|book| {
                book.market_id.is_empty()
                    || book.market_id.len() > MAX_IDENTIFIER_BYTES
                    || book.token_id.is_empty()
                    || book.token_id.len() > MAX_IDENTIFIER_BYTES
                    || book.ingest_sequence <= 0
                    || book.payload_sha256.len() != 64
                    || !book
                        .payload_sha256
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                    || book.source_timestamp
                        > checkpoint.sampled_at.unwrap_or(book.source_timestamp)
            })
        {
            return Err(StrategyFactoryError::Construction(
                "Polymarket orderbook checkpoint contains invalid or excessive book cursors"
                    .to_owned(),
            ));
        }
        if checkpoint
            .sampling_policy_sha256
            .as_ref()
            .is_some_and(|hash| {
                hash.len() != 64
                    || !hash
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            })
        {
            return Err(StrategyFactoryError::Construction(
                "Polymarket checkpoint sampling-policy hash must be lowercase SHA-256".to_owned(),
            ));
        }
        Ok(checkpoint)
    }
}

impl Default for PolymarketBtcFiveMinuteOrderbooksConfig {
    fn default() -> Self {
        Self {
            websocket_url: DEFAULT_WEBSOCKET_URL.to_owned(),
            gamma_api_url: DEFAULT_GAMMA_API_URL.to_owned(),
            sample_interval_ms: 1_000,
            top_n: 20,
            gamma_refresh_ms: 5_000,
            lookback_windows: 1,
            lookahead_windows: 1,
            successor_lead_ms: 30_000,
            contract_grace_ms: 30_000,
            connect_timeout_ms: 10_000,
            bootstrap_timeout_ms: 15_000,
            read_timeout_ms: 40_000,
            ping_interval_ms: 10_000,
            pong_timeout_ms: 25_000,
            reconnect_initial_ms: 250,
            reconnect_max_ms: 30_000,
            max_levels_per_side: 10_000,
            artifact_window_seconds: 3_600,
        }
    }
}

impl PolymarketBtcFiveMinuteOrderbooksConfig {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let config = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::InvalidConfiguration(format!(
                "invalid Polymarket BTC five-minute orderbook config: {error}"
            ))
        })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), StrategyFactoryError> {
        let invalid =
            |message: &str| StrategyFactoryError::InvalidConfiguration(message.to_owned());
        if self.websocket_url != DEFAULT_WEBSOCKET_URL {
            return Err(invalid(
                "websocket_url must be the official public Polymarket CLOB market endpoint",
            ));
        }
        if self.gamma_api_url != DEFAULT_GAMMA_API_URL {
            return Err(invalid(
                "gamma_api_url must be the official public Polymarket Gamma origin",
            ));
        }
        if !(100..=60_000).contains(&self.sample_interval_ms) {
            return Err(invalid("sample_interval_ms must be between 100 and 60000"));
        }
        if self.top_n == 0 || self.top_n > 1_000 {
            return Err(invalid("top_n must be between 1 and 1000"));
        }
        if !(1_000..=60_000).contains(&self.gamma_refresh_ms) {
            return Err(invalid("gamma_refresh_ms must be between 1000 and 60000"));
        }
        if self.lookback_windows != 1 || self.lookahead_windows != 1 {
            return Err(invalid(
                "lookback_windows and lookahead_windows must both equal one",
            ));
        }
        if self.successor_lead_ms == 0
            || self.successor_lead_ms > MARKET_INTERVAL_SECONDS as u64 * 1_000
            || self.contract_grace_ms > MARKET_INTERVAL_SECONDS as u64 * 1_000
        {
            return Err(invalid(
                "successor_lead_ms must be positive and both market timing bounds must be at most 300000",
            ));
        }
        if !(1_000..=30_000).contains(&self.connect_timeout_ms)
            || !(1_000..=60_000).contains(&self.bootstrap_timeout_ms)
        {
            return Err(invalid(
                "connection and bootstrap timeouts must be within their bounded ranges",
            ));
        }
        if !(5_000..=120_000).contains(&self.read_timeout_ms)
            || !(1_000..=30_000).contains(&self.ping_interval_ms)
            || self.ping_interval_ms >= self.read_timeout_ms
            || self.pong_timeout_ms <= self.ping_interval_ms
            || self.pong_timeout_ms > 60_000
        {
            return Err(invalid(
                "heartbeat bounds require ping < PONG timeout <= 60000 and ping < read timeout <= 120000",
            ));
        }
        if !(50..=10_000).contains(&self.reconnect_initial_ms)
            || self.reconnect_max_ms < self.reconnect_initial_ms
            || self.reconnect_max_ms > 60_000
        {
            return Err(invalid(
                "reconnect bounds must be ordered within 50 and 60000 milliseconds",
            ));
        }
        if self.max_levels_per_side < self.top_n || self.max_levels_per_side > 10_000 {
            return Err(invalid(
                "max_levels_per_side must cover top_n and be at most 10000",
            ));
        }
        if !(60..=86_400).contains(&self.artifact_window_seconds)
            || 86_400 % self.artifact_window_seconds != 0
        {
            return Err(invalid(
                "artifact_window_seconds must divide one day and be between 60 and 86400",
            ));
        }
        Ok(())
    }

    fn sampling_policy(&self) -> Value {
        json!({
            "version": SAMPLING_POLICY_VERSION,
            "source": SOURCE,
            "market_interval_seconds": MARKET_INTERVAL_SECONDS,
            "sample_interval_ms": self.sample_interval_ms,
            "top_n": self.top_n,
            "selection": SAMPLING_SELECTION
        })
    }
}

#[derive(Debug, Default)]
pub struct PolymarketBtcFiveMinuteOrderbooksFactory;

impl StrategyFactory for PolymarketBtcFiveMinuteOrderbooksFactory {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    fn config_schema_version(&self) -> i32 {
        CONFIG_SCHEMA_VERSION
    }

    fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError> {
        PolymarketBtcFiveMinuteOrderbooksConfig::from_value(config).map(|_| ())
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
                "Polymarket orderbook config schema must be {CONFIG_SCHEMA_VERSION}"
            )));
        }
        if profile.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(StrategyFactoryError::Construction(format!(
                "Polymarket orderbook checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}"
            )));
        }
        if profile.desired_generation <= 0 {
            return Err(StrategyFactoryError::Construction(
                "profile generation must be positive".to_owned(),
            ));
        }
        let config = PolymarketBtcFiveMinuteOrderbooksConfig::from_value(&profile.config)?;
        let config_snapshot = serde_json::to_value(&config).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "failed to encode effective Polymarket orderbook config: {error}"
            ))
        })?;
        let _checkpoint = OrderbookCheckpoint::from_value(&profile.checkpoint)?;
        let lease_owner = profile.lease_owner.clone().ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease owner is missing".to_owned())
        })?;
        let lease_token = profile.lease_token.ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease token is missing".to_owned())
        })?;
        let client = Client::builder()
            .redirect(Policy::none())
            .connect_timeout(Duration::from_millis(config.connect_timeout_ms))
            .timeout(Duration::from_millis(config.connect_timeout_ms))
            .user_agent("capitonic-market-data-ingester/0.1")
            .build()
            .map_err(|error| {
                StrategyFactoryError::Construction(format!(
                    "failed to build bounded Polymarket HTTP client: {error}"
                ))
            })?;
        Ok(Box::new(PolymarketBtcFiveMinuteOrderbooksStrategy {
            config,
            config_snapshot,
            profile_generation: profile.desired_generation,
            lease_owner: Arc::<str>::from(lease_owner),
            lease_token,
            pool,
            client,
        }))
    }
}

pub struct PolymarketBtcFiveMinuteOrderbooksStrategy {
    config: PolymarketBtcFiveMinuteOrderbooksConfig,
    config_snapshot: Value,
    profile_generation: i64,
    lease_owner: Arc<str>,
    lease_token: Uuid,
    pool: PgPool,
    client: Client,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Outcome {
    Up,
    Down,
}

impl Outcome {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct MarketContract {
    event_slug: String,
    market_id: String,
    condition_id: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    up_token_id: String,
    down_token_id: String,
    tick_size: Decimal,
    received_at: DateTime<Utc>,
}

impl MarketContract {
    fn token_id(&self, outcome: Outcome) -> &str {
        match outcome {
            Outcome::Up => &self.up_token_id,
            Outcome::Down => &self.down_token_id,
        }
    }

    fn matches_wire_market(&self, value: &str) -> bool {
        self.market_id == value || self.condition_id == value
    }

    fn same_identity(&self, other: &Self) -> bool {
        self.event_slug == other.event_slug
            && self.market_id == other.market_id
            && self.condition_id == other.condition_id
            && self.window_start == other.window_start
            && self.window_end == other.window_end
            && self.up_token_id == other.up_token_id
            && self.down_token_id == other.down_token_id
            && self.tick_size == other.tick_size
    }
}

fn aligned_market_window(now: DateTime<Utc>) -> DateTime<Utc> {
    let seconds = now.timestamp().div_euclid(MARKET_INTERVAL_SECONDS) * MARKET_INTERVAL_SECONDS;
    Utc.timestamp_opt(seconds, 0)
        .single()
        .expect("an aligned UTC market timestamp is representable")
}

fn discovery_windows(
    now: DateTime<Utc>,
    config: &PolymarketBtcFiveMinuteOrderbooksConfig,
) -> Vec<DateTime<Utc>> {
    let current = aligned_market_window(now);
    let mut windows = Vec::with_capacity(
        config
            .lookback_windows
            .saturating_add(config.lookahead_windows)
            .saturating_add(1),
    );
    for offset in -(config.lookback_windows as i64)..=(config.lookahead_windows as i64) {
        windows.push(current + TimeDelta::seconds(offset * MARKET_INTERVAL_SECONDS));
    }
    windows
}

fn event_slug(window_start: DateTime<Utc>) -> String {
    format!("{EVENT_SLUG_PREFIX}{}", window_start.timestamp())
}

fn subscription_markets(
    discovered: &[MarketContract],
    now: DateTime<Utc>,
    config: &PolymarketBtcFiveMinuteOrderbooksConfig,
) -> Vec<MarketContract> {
    let current_start = aligned_market_window(now);
    let current_end = current_start + TimeDelta::seconds(MARKET_INTERVAL_SECONDS);
    let successor_from = current_end - TimeDelta::milliseconds(config.successor_lead_ms as i64);
    let previous_until = current_start + TimeDelta::milliseconds(config.contract_grace_ms as i64);
    discovered
        .iter()
        .filter(|market| {
            market.window_start == current_start
                || (market.window_end == current_start && now < previous_until)
                || (market.window_start == current_end && now >= successor_from)
        })
        .cloned()
        .collect()
}

fn parse_gamma_market(
    value: &Value,
    expected_window_start: DateTime<Utc>,
    received_at: DateTime<Utc>,
) -> Result<MarketContract, StrategyError> {
    let event = value.as_object().ok_or_else(|| {
        source_error(
            "polymarket_gamma_invalid_event",
            "Gamma event response must be an object",
        )
    })?;
    let expected_slug = event_slug(expected_window_start);
    let actual_slug = required_string(event, &["slug"], "Gamma event")?;
    if actual_slug != expected_slug {
        return Err(source_error(
            "polymarket_gamma_identity_mismatch",
            format!("Gamma event slug {actual_slug} did not match {expected_slug}"),
        ));
    }
    let series_slug = string_field(event, &["seriesSlug", "series_slug"])
        .or_else(|| series_slug_from_relation(event))
        .ok_or_else(|| {
            source_error(
                "polymarket_gamma_invalid_event",
                "Gamma event is missing its series slug",
            )
        })?;
    if series_slug != SERIES_SLUG {
        return Err(source_error(
            "polymarket_gamma_identity_mismatch",
            format!("Gamma event belongs to unexpected series {series_slug}"),
        ));
    }
    let window_start =
        datetime_field(event, &["eventStartTime", "startTime"])?.ok_or_else(|| {
            source_error(
                "polymarket_gamma_invalid_event",
                "Gamma event is missing its start time",
            )
        })?;
    if window_start != expected_window_start {
        return Err(source_error(
            "polymarket_gamma_identity_mismatch",
            "Gamma event start time did not match its slug epoch",
        ));
    }
    let markets = event
        .get("markets")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            source_error(
                "polymarket_gamma_invalid_event",
                "Gamma event is missing its markets array",
            )
        })?;
    if markets.len() != 1 {
        return Err(source_error(
            "polymarket_gamma_ambiguous_market",
            "BTC Up/Down five-minute event must contain exactly one market",
        ));
    }
    let market = markets[0].as_object().ok_or_else(|| {
        source_error(
            "polymarket_gamma_invalid_event",
            "Gamma event market must be an object",
        )
    })?;
    let window_end = datetime_field(market, &["endDate", "endDateIso"])?
        .or(datetime_field(event, &["endDate"])?)
        .ok_or_else(|| {
            source_error(
                "polymarket_gamma_invalid_event",
                "Gamma event is missing its market end time",
            )
        })?;
    if window_end - window_start != TimeDelta::seconds(MARKET_INTERVAL_SECONDS) {
        return Err(source_error(
            "polymarket_gamma_window_mismatch",
            "Gamma BTC market does not have an exact five-minute window",
        ));
    }
    let resolution_source = string_field(market, &["resolutionSource", "resolution_source"])
        .or_else(|| string_field(event, &["resolutionSource", "resolution_source"]))
        .ok_or_else(|| {
            source_error(
                "polymarket_gamma_invalid_event",
                "Gamma event is missing its resolution source",
            )
        })?;
    if !is_chainlink_btc_usd_source(&resolution_source) {
        return Err(source_error(
            "polymarket_gamma_resolution_source_mismatch",
            "Gamma BTC market resolution source is not Chainlink BTC/USD",
        ));
    }
    let outcomes = string_array_field(market, &["outcomes"])?;
    let token_ids = string_array_field(
        market,
        &["clobTokenIds", "clob_token_ids", "tokenIds", "token_ids"],
    )?;
    if outcomes.len() != 2 || token_ids.len() != 2 {
        return Err(source_error(
            "polymarket_gamma_outcome_mismatch",
            "Gamma BTC market must have exactly two outcomes and token IDs",
        ));
    }
    let mut up_token_id = None;
    let mut down_token_id = None;
    for (outcome, token_id) in outcomes.iter().zip(&token_ids) {
        validate_token_id(token_id)?;
        match outcome.trim().to_ascii_lowercase().as_str() {
            "up" if up_token_id.replace(token_id.clone()).is_none() => {}
            "down" if down_token_id.replace(token_id.clone()).is_none() => {}
            _ => {
                return Err(source_error(
                    "polymarket_gamma_outcome_mismatch",
                    "Gamma BTC market outcomes must map uniquely to Up and Down",
                ))
            }
        }
    }
    let up_token_id = up_token_id.expect("validated outcomes contain Up");
    let down_token_id = down_token_id.expect("validated outcomes contain Down");
    if up_token_id == down_token_id {
        return Err(source_error(
            "polymarket_gamma_token_collision",
            "Gamma BTC market token IDs must be distinct",
        ));
    }
    let market_id = required_string(market, &["id"], "Gamma market")?;
    let condition_id = required_string(market, &["conditionId", "condition_id"], "Gamma market")?;
    validate_identifier("event slug", &actual_slug)?;
    validate_market_id(&market_id)?;
    validate_condition_id(&condition_id)?;
    if market_id == condition_id {
        return Err(source_error(
            "polymarket_gamma_identifier_collision",
            "Gamma market and condition identifiers must be distinct",
        ));
    }
    let tick_size = decimal_field(
        market,
        &[
            "orderPriceMinTickSize",
            "minimumTickSize",
            "tickSize",
            "tick_size",
        ],
    )?
    .ok_or_else(|| {
        source_error(
            "polymarket_gamma_invalid_event",
            "Gamma market is missing its minimum tick size",
        )
    })?;
    if tick_size <= Decimal::ZERO || tick_size >= Decimal::ONE || tick_size.scale() > 8 {
        return Err(source_error(
            "polymarket_gamma_invalid_tick_size",
            "Gamma market minimum tick size must be strictly between zero and one",
        ));
    }
    Ok(MarketContract {
        event_slug: actual_slug,
        market_id,
        condition_id,
        window_start,
        window_end,
        up_token_id,
        down_token_id,
        tick_size,
        received_at: canonical_timestamp(received_at),
    })
}

fn validate_market_set(markets: &[MarketContract]) -> Result<(), StrategyError> {
    if markets.len() > 3 {
        return Err(integrity_error(
            "polymarket_contract_set_too_large",
            "Polymarket contract set exceeded previous/current/successor bound",
        ));
    }
    for (index, market) in markets.iter().enumerate() {
        for other in markets.iter().skip(index + 1) {
            if market.window_start == other.window_start {
                if !market.same_identity(other) {
                    return Err(integrity_error(
                        "polymarket_contract_identity_conflict",
                        "Gamma returned conflicting identities for one BTC window",
                    ));
                }
                continue;
            }
            let token_collision = [&market.up_token_id, &market.down_token_id]
                .into_iter()
                .any(|token| token == &other.up_token_id || token == &other.down_token_id);
            if token_collision {
                return Err(integrity_error(
                    "polymarket_contract_token_collision",
                    "distinct BTC windows reused a CLOB token identity",
                ));
            }
            if market.market_id == other.market_id
                || market.market_id == other.condition_id
                || market.condition_id == other.market_id
                || market.condition_id == other.condition_id
            {
                return Err(integrity_error(
                    "polymarket_contract_market_collision",
                    "distinct BTC windows reused a market identity",
                ));
            }
        }
    }
    Ok(())
}

async fn discover_markets(
    client: &Client,
    config: &PolymarketBtcFiveMinuteOrderbooksConfig,
    now: DateTime<Utc>,
) -> Result<Vec<MarketContract>, StrategyError> {
    let requests = discovery_windows(now, config)
        .into_iter()
        .map(|window| fetch_gamma_market(client, config, window));
    let mut markets = Vec::with_capacity(3);
    for result in join_all(requests).await {
        if let Some(market) = result? {
            markets.push(market);
        }
    }
    markets.sort_by_key(|market| market.window_start);
    validate_market_set(&markets)?;
    let current = aligned_market_window(now);
    if !markets.iter().any(|market| market.window_start == current) {
        return Err(source_error(
            "polymarket_current_contract_unavailable",
            format!("Gamma did not provide the current BTC contract {current}"),
        ));
    }
    Ok(markets)
}

async fn fetch_gamma_market(
    client: &Client,
    config: &PolymarketBtcFiveMinuteOrderbooksConfig,
    window_start: DateTime<Utc>,
) -> Result<Option<MarketContract>, StrategyError> {
    let slug = event_slug(window_start);
    let url = format!("{}/events/slug/{slug}", config.gamma_api_url);
    let response = client.get(url).send().await.map_err(|error| {
        source_error(
            "polymarket_gamma_request_failed",
            format!("failed to request Gamma event {slug}: {error}"),
        )
    })?;
    if response.status() == StatusCode::NOT_FOUND {
        return Ok(None);
    }
    let response = response.error_for_status().map_err(|error| {
        source_error(
            "polymarket_gamma_status_failed",
            format!("Gamma rejected event {slug}: {error}"),
        )
    })?;
    if response
        .content_length()
        .is_some_and(|length| length > MAX_GAMMA_RESPONSE_BYTES as u64)
    {
        return Err(source_error(
            "polymarket_gamma_body_too_large",
            "Gamma response exceeded the bounded body size",
        ));
    }
    let initial_capacity = response
        .content_length()
        .and_then(|length| usize::try_from(length).ok())
        .unwrap_or_default()
        .min(MAX_GAMMA_RESPONSE_BYTES);
    let mut body = Vec::with_capacity(initial_capacity);
    let mut chunks = response.bytes_stream();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.map_err(|error| {
            source_error(
                "polymarket_gamma_body_read_failed",
                format!("failed to read Gamma event {slug}: {error}"),
            )
        })?;
        if chunk.len() > MAX_GAMMA_RESPONSE_BYTES.saturating_sub(body.len()) {
            return Err(source_error(
                "polymarket_gamma_body_too_large",
                "Gamma response exceeded the bounded body size",
            ));
        }
        body.extend_from_slice(&chunk);
    }
    // Receipt is recorded only after the complete bounded response body arrived.
    let received_at = Utc::now();
    let value = serde_json::from_slice::<Value>(&body).map_err(|error| {
        source_error(
            "polymarket_gamma_decode_failed",
            format!("failed to decode Gamma event {slug}: {error}"),
        )
    })?;
    parse_gamma_market(&value, window_start, received_at).map(Some)
}

fn required_string(
    object: &Map<String, Value>,
    keys: &[&str],
    context: &str,
) -> Result<String, StrategyError> {
    string_field(object, keys).ok_or_else(|| {
        source_error(
            "polymarket_invalid_string_field",
            format!("{context} is missing required field {}", keys[0]),
        )
    })
}

fn string_field(object: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| match object.get(*key)? {
        Value::String(value) => {
            let value = value.trim();
            (!value.is_empty()).then(|| value.to_owned())
        }
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    })
}

fn string_array_field(
    object: &Map<String, Value>,
    keys: &[&str],
) -> Result<Vec<String>, StrategyError> {
    let value = keys
        .iter()
        .find_map(|key| object.get(*key))
        .ok_or_else(|| {
            source_error(
                "polymarket_gamma_invalid_array",
                format!("Gamma market is missing {}", keys[0]),
            )
        })?;
    let values = match value {
        Value::Array(values) => values.clone(),
        Value::String(value) => serde_json::from_str::<Vec<Value>>(value).map_err(|error| {
            source_error(
                "polymarket_gamma_invalid_array",
                format!("Gamma market field {} is invalid JSON: {error}", keys[0]),
            )
        })?,
        _ => {
            return Err(source_error(
                "polymarket_gamma_invalid_array",
                format!("Gamma market field {} must be an array", keys[0]),
            ))
        }
    };
    values
        .into_iter()
        .map(|value| match value {
            Value::String(value) if !value.trim().is_empty() => Ok(value.trim().to_owned()),
            Value::Number(value) => Ok(value.to_string()),
            _ => Err(source_error(
                "polymarket_gamma_invalid_array",
                format!("Gamma market field {} contains a non-string value", keys[0]),
            )),
        })
        .collect()
}

fn decimal_field(
    object: &Map<String, Value>,
    keys: &[&str],
) -> Result<Option<Decimal>, StrategyError> {
    let Some((key, value)) = keys
        .iter()
        .find_map(|key| object.get(*key).map(|value| (*key, value)))
    else {
        return Ok(None);
    };
    let raw = match value {
        Value::String(value) => value.as_str(),
        Value::Number(value) => {
            return Decimal::from_str(&value.to_string())
                .map(Some)
                .map_err(|error| {
                    source_error(
                        "polymarket_invalid_decimal",
                        format!("field {key} is not a decimal: {error}"),
                    )
                })
        }
        _ => {
            return Err(source_error(
                "polymarket_invalid_decimal",
                format!("field {key} is not a decimal"),
            ))
        }
    };
    if raw.len() > MAX_NUMERIC_BYTES {
        return Err(source_error(
            "polymarket_invalid_decimal",
            format!("field {key} exceeded the numeric length bound"),
        ));
    }
    Decimal::from_str(raw).map(Some).map_err(|error| {
        source_error(
            "polymarket_invalid_decimal",
            format!("field {key} is not a decimal: {error}"),
        )
    })
}

fn datetime_field(
    object: &Map<String, Value>,
    keys: &[&str],
) -> Result<Option<DateTime<Utc>>, StrategyError> {
    let Some(value) = string_field(object, keys) else {
        return Ok(None);
    };
    DateTime::parse_from_rfc3339(&value)
        .map(|timestamp| Some(canonical_timestamp(timestamp.with_timezone(&Utc))))
        .map_err(|error| {
            source_error(
                "polymarket_invalid_datetime",
                format!("field {} is not RFC3339: {error}", keys[0]),
            )
        })
}

fn series_slug_from_relation(object: &Map<String, Value>) -> Option<String> {
    object
        .get("series")?
        .as_array()?
        .iter()
        .filter_map(Value::as_object)
        .find_map(|series| string_field(series, &["slug"]))
}

fn is_chainlink_btc_usd_source(source: &str) -> bool {
    let normalized = source.trim().to_ascii_lowercase();
    (normalized.contains("chainlink") || normalized.contains("chain.link"))
        && (normalized.contains("btc-usd")
            || normalized.contains("btc/usd")
            || (normalized.contains("btc") && normalized.contains("usd")))
}

fn validate_identifier(name: &str, value: &str) -> Result<(), StrategyError> {
    if value.is_empty()
        || value.len() > MAX_IDENTIFIER_BYTES
        || value.chars().any(char::is_whitespace)
    {
        return Err(source_error(
            "polymarket_invalid_identifier",
            format!("{name} is empty, oversized, or contains whitespace"),
        ));
    }
    Ok(())
}

fn validate_market_id(value: &str) -> Result<(), StrategyError> {
    validate_identifier("market ID", value)?;
    if value.len() > 256 {
        return Err(source_error(
            "polymarket_invalid_market_id",
            "Gamma market ID exceeded the storage identity bound",
        ));
    }
    Ok(())
}

fn validate_condition_id(value: &str) -> Result<(), StrategyError> {
    if value.len() != 66
        || !value.starts_with("0x")
        || !value[2..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(source_error(
            "polymarket_invalid_condition_id",
            "Gamma condition ID must be a lowercase 32-byte hex value",
        ));
    }
    Ok(())
}

fn validate_token_id(value: &str) -> Result<(), StrategyError> {
    if value.is_empty() || value.len() > 100 || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(source_error(
            "polymarket_invalid_token_id",
            "Gamma token ID must be a bounded unsigned decimal integer",
        ));
    }
    Ok(())
}

fn canonical_timestamp(timestamp: DateTime<Utc>) -> DateTime<Utc> {
    DateTime::from_timestamp_micros(timestamp.timestamp_micros())
        .expect("a valid timestamp remains valid at microsecond precision")
}

#[derive(Debug, Clone, PartialEq)]
struct PriceLevel {
    price: Decimal,
    size: Decimal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BookSide {
    Bid,
    Ask,
}

#[derive(Debug, Clone, PartialEq)]
struct PriceChange {
    token_id: String,
    side: BookSide,
    price: Decimal,
    size: Decimal,
    source_hash: Option<String>,
    best_bid: Option<Decimal>,
    best_ask: Option<Decimal>,
}

#[derive(Debug, Clone, PartialEq)]
enum ClobMessage {
    Book {
        market_id: String,
        token_id: String,
        bids: Vec<PriceLevel>,
        asks: Vec<PriceLevel>,
        source_timestamp: DateTime<Utc>,
        source_hash: Option<String>,
    },
    PriceChange {
        market_id: String,
        changes: Vec<PriceChange>,
        source_timestamp: DateTime<Utc>,
    },
    BestBidAsk {
        market_id: String,
        token_id: String,
        best_bid: Option<Decimal>,
        best_ask: Option<Decimal>,
        source_timestamp: DateTime<Utc>,
    },
    TickSizeChange {
        market_id: String,
        token_id: String,
        old_tick_size: Decimal,
        new_tick_size: Decimal,
        source_timestamp: DateTime<Utc>,
    },
    Auxiliary {
        market_id: String,
        token_id: Option<String>,
        source_timestamp: DateTime<Utc>,
    },
    Control,
}

fn parse_clob_frame(bytes: &[u8], max_levels: usize) -> Result<Vec<ClobMessage>, StrategyError> {
    if bytes.len() > MAX_WEBSOCKET_FRAME_BYTES {
        return Err(source_error(
            "polymarket_clob_frame_too_large",
            "Polymarket CLOB websocket frame exceeded the bounded payload size",
        ));
    }
    let value = serde_json::from_slice::<Value>(bytes).map_err(|error| {
        source_error(
            "polymarket_clob_decode_failed",
            format!("failed to decode Polymarket CLOB websocket JSON: {error}"),
        )
    })?;
    let mut messages = Vec::new();
    parse_clob_value(&value, max_levels, &mut messages)?;
    Ok(messages)
}

fn parse_clob_value(
    value: &Value,
    max_levels: usize,
    messages: &mut Vec<ClobMessage>,
) -> Result<(), StrategyError> {
    if let Some(values) = value.as_array() {
        if values.len() > MAX_MESSAGES_PER_FRAME.saturating_sub(messages.len()) {
            return Err(source_error(
                "polymarket_clob_message_batch_too_large",
                "Polymarket CLOB frame exceeded the bounded message count",
            ));
        }
        for value in values {
            parse_clob_value(value, max_levels, messages)?;
        }
        return Ok(());
    }
    if messages.len() >= MAX_MESSAGES_PER_FRAME {
        return Err(source_error(
            "polymarket_clob_message_batch_too_large",
            "Polymarket CLOB frame exceeded the bounded message count",
        ));
    }
    let object = value.as_object().ok_or_else(|| {
        source_error(
            "polymarket_clob_invalid_message",
            "Polymarket CLOB message must be an object or array",
        )
    })?;
    let event_type = required_string(object, &["event_type"], "CLOB message")?;
    if event_type == "new_market" {
        messages.push(ClobMessage::Control);
        return Ok(());
    }
    let market_id = required_string(object, &["market"], "CLOB message")?;
    validate_identifier("wire market ID", &market_id)?;
    let source_timestamp = clob_timestamp(object)?;
    let parsed = match event_type.as_str() {
        "book" => {
            let token_id = required_string(object, &["asset_id"], "CLOB book")?;
            validate_identifier("wire token ID", &token_id)?;
            ClobMessage::Book {
                market_id,
                token_id,
                bids: parse_clob_levels(object.get("bids"), "bids", max_levels)?,
                asks: parse_clob_levels(object.get("asks"), "asks", max_levels)?,
                source_timestamp,
                source_hash: optional_source_hash(object, &["hash"])?,
            }
        }
        "price_change" => {
            let raw_changes = object
                .get("price_changes")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    source_error(
                        "polymarket_clob_invalid_price_change",
                        "CLOB price_change is missing price_changes",
                    )
                })?;
            if raw_changes.is_empty() || raw_changes.len() > MAX_CHANGES_PER_MESSAGE {
                return Err(source_error(
                    "polymarket_clob_invalid_price_change",
                    "CLOB price_change must contain a bounded nonempty changes array",
                ));
            }
            let mut changes = Vec::with_capacity(raw_changes.len());
            for raw in raw_changes {
                let change = raw.as_object().ok_or_else(|| {
                    source_error(
                        "polymarket_clob_invalid_price_change",
                        "CLOB price_change entry must be an object",
                    )
                })?;
                let token_id = required_string(change, &["asset_id"], "CLOB price change")?;
                validate_identifier("wire token ID", &token_id)?;
                let side = match required_string(change, &["side"], "CLOB price change")?
                    .to_ascii_uppercase()
                    .as_str()
                {
                    "BUY" => BookSide::Bid,
                    "SELL" => BookSide::Ask,
                    other => {
                        return Err(source_error(
                            "polymarket_clob_invalid_side",
                            format!("unsupported CLOB price_change side {other}"),
                        ))
                    }
                };
                let price = required_decimal(change, &["price"])?;
                let size = required_decimal(change, &["size"])?;
                if price <= Decimal::ZERO
                    || price >= Decimal::ONE
                    || price.scale() > 8
                    || size < Decimal::ZERO
                {
                    return Err(source_error(
                        "polymarket_clob_invalid_level",
                        "CLOB price change has an out-of-range price or negative size",
                    ));
                }
                changes.push(PriceChange {
                    token_id,
                    side,
                    price,
                    size,
                    source_hash: optional_source_hash(change, &["hash"])?,
                    best_bid: optional_decimal(change, &["best_bid"])?,
                    best_ask: optional_decimal(change, &["best_ask"])?,
                });
            }
            ClobMessage::PriceChange {
                market_id,
                changes,
                source_timestamp,
            }
        }
        "best_bid_ask" => {
            let token_id = required_string(object, &["asset_id"], "CLOB best_bid_ask")?;
            validate_identifier("wire token ID", &token_id)?;
            ClobMessage::BestBidAsk {
                market_id,
                token_id,
                best_bid: optional_decimal(object, &["best_bid"])?,
                best_ask: optional_decimal(object, &["best_ask"])?,
                source_timestamp,
            }
        }
        "tick_size_change" => {
            let token_id = required_string(object, &["asset_id"], "CLOB tick_size_change")?;
            validate_identifier("wire token ID", &token_id)?;
            ClobMessage::TickSizeChange {
                market_id,
                token_id,
                old_tick_size: required_decimal(object, &["old_tick_size"])?,
                new_tick_size: required_decimal(object, &["new_tick_size"])?,
                source_timestamp,
            }
        }
        "last_trade_price" => {
            let token_id = required_string(object, &["asset_id"], "CLOB last_trade_price")?;
            validate_identifier("wire token ID", &token_id)?;
            let price = required_decimal(object, &["price"])?;
            let size = required_decimal(object, &["size"])?;
            if price <= Decimal::ZERO || price >= Decimal::ONE || size <= Decimal::ZERO {
                return Err(source_error(
                    "polymarket_clob_invalid_trade",
                    "CLOB last trade has an invalid price or size",
                ));
            }
            ClobMessage::Auxiliary {
                market_id,
                token_id: Some(token_id),
                source_timestamp,
            }
        }
        "market_resolved" => {
            let token_id = required_string(object, &["winning_asset_id"], "CLOB market_resolved")?;
            validate_identifier("winning token ID", &token_id)?;
            let _winning_outcome =
                required_string(object, &["winning_outcome"], "CLOB market_resolved")?;
            ClobMessage::Auxiliary {
                market_id,
                token_id: Some(token_id),
                source_timestamp,
            }
        }
        other => {
            return Err(source_error(
                "polymarket_clob_unsupported_event",
                format!("unsupported Polymarket CLOB event type {other}"),
            ))
        }
    };
    messages.push(parsed);
    Ok(())
}

fn parse_clob_levels(
    value: Option<&Value>,
    field: &str,
    max_levels: usize,
) -> Result<Vec<PriceLevel>, StrategyError> {
    let values = value.and_then(Value::as_array).ok_or_else(|| {
        source_error(
            "polymarket_clob_invalid_book",
            format!("CLOB book is missing {field}"),
        )
    })?;
    if values.len() > max_levels {
        return Err(source_error(
            "polymarket_clob_book_too_large",
            format!("CLOB {field} exceeded the configured level bound"),
        ));
    }
    let mut levels = Vec::with_capacity(values.len());
    let mut prices = BTreeSet::new();
    for value in values {
        let object = value.as_object().ok_or_else(|| {
            source_error(
                "polymarket_clob_invalid_book",
                format!("CLOB {field} level must be an object"),
            )
        })?;
        let price = required_decimal(object, &["price"])?;
        let size = required_decimal(object, &["size"])?;
        if price <= Decimal::ZERO
            || price >= Decimal::ONE
            || price.scale() > 8
            || size < Decimal::ZERO
            || !prices.insert(price)
        {
            return Err(source_error(
                "polymarket_clob_invalid_level",
                format!("CLOB {field} contained an invalid or duplicate level"),
            ));
        }
        levels.push(PriceLevel { price, size });
    }
    Ok(levels)
}

fn required_decimal(object: &Map<String, Value>, keys: &[&str]) -> Result<Decimal, StrategyError> {
    decimal_field(object, keys)?.ok_or_else(|| {
        source_error(
            "polymarket_invalid_decimal",
            format!("missing required decimal field {}", keys[0]),
        )
    })
}

fn optional_decimal(
    object: &Map<String, Value>,
    keys: &[&str],
) -> Result<Option<Decimal>, StrategyError> {
    for key in keys {
        let Some(value) = object.get(*key) else {
            continue;
        };
        if value.is_null() {
            return Ok(None);
        }
        return decimal_field(object, &[*key]);
    }
    Ok(None)
}

fn optional_source_hash(
    object: &Map<String, Value>,
    keys: &[&str],
) -> Result<Option<String>, StrategyError> {
    let value = string_field(object, keys);
    if value.as_ref().is_some_and(|value| {
        value.len() > MAX_SOURCE_HASH_BYTES || value.chars().any(char::is_whitespace)
    }) {
        return Err(source_error(
            "polymarket_clob_invalid_source_hash",
            "CLOB source hash was oversized or contained whitespace",
        ));
    }
    Ok(value)
}

fn clob_timestamp(object: &Map<String, Value>) -> Result<DateTime<Utc>, StrategyError> {
    let raw = object.get("timestamp").ok_or_else(|| {
        source_error(
            "polymarket_clob_invalid_timestamp",
            "CLOB message is missing its timestamp",
        )
    })?;
    let milliseconds = match raw {
        Value::String(value) if value.len() <= 32 => value.parse::<i64>(),
        Value::Number(value) => value.to_string().parse::<i64>(),
        _ => {
            return Err(source_error(
                "polymarket_clob_invalid_timestamp",
                "CLOB timestamp must be an integer millisecond value",
            ))
        }
    }
    .map_err(|error| {
        source_error(
            "polymarket_clob_invalid_timestamp",
            format!("CLOB timestamp is not a valid integer: {error}"),
        )
    })?;
    Utc.timestamp_millis_opt(milliseconds)
        .single()
        .map(canonical_timestamp)
        .ok_or_else(|| {
            source_error(
                "polymarket_clob_invalid_timestamp",
                "CLOB timestamp is outside the representable UTC range",
            )
        })
}

#[derive(Debug, Clone)]
struct BookState {
    market: MarketContract,
    token_id: String,
    outcome: Outcome,
    tick_size: Decimal,
    bids: BTreeMap<Decimal, Decimal>,
    asks: BTreeMap<Decimal, Decimal>,
    bootstrapped: bool,
    source_timestamp: Option<DateTime<Utc>>,
    received_at: Option<DateTime<Utc>>,
    source_hash: Option<String>,
    ingest_sequence: i64,
}

impl BookState {
    fn new(market: MarketContract, outcome: Outcome) -> Self {
        let token_id = market.token_id(outcome).to_owned();
        let tick_size = market.tick_size;
        Self {
            market,
            token_id,
            outcome,
            tick_size,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            bootstrapped: false,
            source_timestamp: None,
            received_at: None,
            source_hash: None,
            ingest_sequence: 0,
        }
    }

    fn best_bid(&self) -> Option<Decimal> {
        self.bids.last_key_value().map(|(price, _)| *price)
    }

    fn best_ask(&self) -> Option<Decimal> {
        self.asks.first_key_value().map(|(price, _)| *price)
    }

    fn validate(&self, max_levels: usize) -> Result<(), StrategyError> {
        if self.bids.len() > max_levels || self.asks.len() > max_levels {
            return Err(source_error(
                "polymarket_clob_book_too_large",
                "reconstructed CLOB book exceeded the configured level bound",
            ));
        }
        if self.bids.iter().chain(&self.asks).any(|(price, size)| {
            *price <= Decimal::ZERO
                || *price >= Decimal::ONE
                || price.scale() > 8
                || *size <= Decimal::ZERO
        }) {
            return Err(source_error(
                "polymarket_clob_invalid_level",
                "reconstructed CLOB book contained an invalid level",
            ));
        }
        if matches!(
            (self.best_bid(), self.best_ask()),
            (Some(bid), Some(ask)) if bid >= ask
        ) {
            return Err(source_error(
                "polymarket_clob_crossed_book",
                "reconstructed CLOB book was crossed or locked",
            ));
        }
        Ok(())
    }

    fn is_stale(&self, timestamp: DateTime<Utc>) -> bool {
        self.source_timestamp.is_some_and(|last| timestamp < last)
    }

    fn reconcile_advertised_top(
        &mut self,
        best_bid: Option<Decimal>,
        best_ask: Option<Decimal>,
        max_levels: usize,
    ) -> Result<(), StrategyError> {
        validate_advertised_top(best_bid, best_ask)?;
        if let Some(best_bid) = best_bid {
            if best_bid.is_zero() {
                self.bids.clear();
            } else {
                self.bids.retain(|price, _| *price <= best_bid);
            }
        }
        if let Some(best_ask) = best_ask {
            if best_ask == Decimal::ONE {
                self.asks.clear();
            } else {
                self.asks.retain(|price, _| *price >= best_ask);
            }
        }
        self.validate(max_levels)?;
        self.verify_advertised_top(best_bid, best_ask)
    }

    fn verify_advertised_top(
        &self,
        best_bid: Option<Decimal>,
        best_ask: Option<Decimal>,
    ) -> Result<(), StrategyError> {
        validate_advertised_top(best_bid, best_ask)?;
        let bid_matches =
            best_bid.is_none_or(|expected| self.best_bid().unwrap_or(Decimal::ZERO) == expected);
        let ask_matches =
            best_ask.is_none_or(|expected| self.best_ask().unwrap_or(Decimal::ONE) == expected);
        if !bid_matches || !ask_matches {
            return Err(source_error(
                "polymarket_clob_top_mismatch",
                "CLOB advertised top did not match the reconstructed book",
            ));
        }
        Ok(())
    }
}

fn validate_advertised_top(
    best_bid: Option<Decimal>,
    best_ask: Option<Decimal>,
) -> Result<(), StrategyError> {
    if best_bid.is_some_and(|price| price < Decimal::ZERO || price >= Decimal::ONE)
        || best_ask.is_some_and(|price| price <= Decimal::ZERO || price > Decimal::ONE)
        || best_bid.is_some_and(|price| price.scale() > 8)
        || best_ask.is_some_and(|price| price.scale() > 8)
    {
        return Err(source_error(
            "polymarket_clob_invalid_advertised_top",
            "CLOB advertised top was outside its valid sentinel range",
        ));
    }
    Ok(())
}

fn levels_to_map(levels: Vec<PriceLevel>) -> BTreeMap<Decimal, Decimal> {
    levels
        .into_iter()
        .filter(|level| !level.size.is_zero())
        .map(|level| (level.price, level.size))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApplyOutcome {
    Applied,
    AwaitingSnapshot,
    IgnoredForeign,
    NonMutating,
}

#[derive(Debug)]
struct BookRegistry {
    connection_epoch: Uuid,
    books: HashMap<String, BookState>,
    next_sequence: i64,
}

impl BookRegistry {
    fn new(connection_epoch: Uuid, markets: &[MarketContract]) -> Result<Self, StrategyError> {
        let mut registry = Self {
            connection_epoch,
            books: HashMap::with_capacity(markets.len().saturating_mul(2)),
            next_sequence: 1,
        };
        registry.install_market_set(markets)?;
        Ok(registry)
    }

    fn reset_connection(&mut self, connection_epoch: Uuid) {
        self.connection_epoch = connection_epoch;
        self.next_sequence = 1;
        for book in self.books.values_mut() {
            book.bids.clear();
            book.asks.clear();
            book.bootstrapped = false;
            book.source_timestamp = None;
            book.received_at = None;
            book.source_hash = None;
            book.ingest_sequence = 0;
        }
    }

    fn install_market_set(&mut self, markets: &[MarketContract]) -> Result<(), StrategyError> {
        validate_market_set(markets)?;
        let desired_tokens = markets
            .iter()
            .flat_map(|market| [&market.up_token_id, &market.down_token_id])
            .cloned()
            .collect::<HashSet<_>>();
        for market in markets {
            for (token_id, outcome) in [
                (&market.up_token_id, Outcome::Up),
                (&market.down_token_id, Outcome::Down),
            ] {
                if let Some(existing) = self.books.get_mut(token_id) {
                    if existing.market.market_id != market.market_id
                        || existing.market.condition_id != market.condition_id
                        || existing.market.window_start != market.window_start
                        || existing.market.window_end != market.window_end
                        || existing.outcome != outcome
                    {
                        return Err(integrity_error(
                            "polymarket_contract_identity_conflict",
                            "desired CLOB contract conflicts with a registered token identity",
                        ));
                    }
                    if existing.tick_size != market.tick_size {
                        return Err(integrity_error(
                            "polymarket_contract_tick_race",
                            "Gamma tick size changed without a matching CLOB tick update",
                        ));
                    }
                    // Preserve the first causal discovery receipt for an unchanged identity.
                    let receipt = existing.market.received_at.min(market.received_at);
                    existing.market = market.clone();
                    existing.market.received_at = receipt;
                } else {
                    self.books
                        .insert(token_id.clone(), BookState::new(market.clone(), outcome));
                }
            }
        }
        self.books
            .retain(|token_id, _| desired_tokens.contains(token_id));
        Ok(())
    }

    fn all_bootstrapped(&self) -> bool {
        !self.books.is_empty() && self.books.values().all(|book| book.bootstrapped)
    }

    fn apply(
        &mut self,
        message: ClobMessage,
        received_at: DateTime<Utc>,
        max_levels: usize,
    ) -> Result<ApplyOutcome, StrategyError> {
        let received_at = canonical_timestamp(received_at);
        match message {
            ClobMessage::Book {
                market_id,
                token_id,
                bids,
                asks,
                source_timestamp,
                source_hash,
            } => {
                let Some(current) = self.books.get(&token_id) else {
                    return self.foreign_or_mismatch(&market_id, &token_id);
                };
                if !current.market.matches_wire_market(&market_id) {
                    return Err(source_error(
                        "polymarket_clob_market_mismatch",
                        "CLOB full book market did not match its registered token",
                    ));
                }
                if current.is_stale(source_timestamp) {
                    return Ok(ApplyOutcome::NonMutating);
                }
                let sequence = self.take_sequence()?;
                let current = self
                    .books
                    .get(&token_id)
                    .expect("book identity was checked before sequence allocation");
                let mut candidate = current.clone();
                candidate.bids = levels_to_map(bids);
                candidate.asks = levels_to_map(asks);
                candidate.bootstrapped = true;
                candidate.source_timestamp = Some(source_timestamp);
                candidate.received_at = Some(received_at);
                candidate.source_hash = source_hash;
                candidate.ingest_sequence = sequence;
                candidate.validate(max_levels)?;
                self.books.insert(token_id, candidate);
                Ok(ApplyOutcome::Applied)
            }
            ClobMessage::PriceChange {
                market_id,
                changes,
                source_timestamp,
            } => self.apply_price_changes(
                &market_id,
                changes,
                source_timestamp,
                received_at,
                max_levels,
            ),
            ClobMessage::BestBidAsk {
                market_id,
                token_id,
                best_bid,
                best_ask,
                source_timestamp,
            } => {
                let Some(book) = self.books.get(&token_id) else {
                    return self.foreign_or_mismatch(&market_id, &token_id);
                };
                if !book.market.matches_wire_market(&market_id) {
                    return Err(source_error(
                        "polymarket_clob_market_mismatch",
                        "CLOB best_bid_ask market did not match its registered token",
                    ));
                }
                if !book.bootstrapped {
                    return Ok(ApplyOutcome::AwaitingSnapshot);
                }
                if book.is_stale(source_timestamp) {
                    return Ok(ApplyOutcome::NonMutating);
                }
                // The venue does not sequence best_bid_ask relative to book and
                // price_change frames. Validate its values, but do not compare
                // the advisory event with a book that may already be newer.
                validate_advertised_top(best_bid, best_ask)?;
                Ok(ApplyOutcome::NonMutating)
            }
            ClobMessage::TickSizeChange {
                market_id,
                token_id,
                old_tick_size,
                new_tick_size,
                source_timestamp,
            } => {
                let Some(book) = self.books.get(&token_id) else {
                    return self.foreign_or_mismatch(&market_id, &token_id);
                };
                if !book.market.matches_wire_market(&market_id) {
                    return Err(source_error(
                        "polymarket_clob_market_mismatch",
                        "CLOB tick_size_change market did not match its registered token",
                    ));
                }
                if !book.bootstrapped {
                    return Ok(ApplyOutcome::AwaitingSnapshot);
                }
                if book.is_stale(source_timestamp) {
                    return Ok(ApplyOutcome::NonMutating);
                }
                if old_tick_size != book.tick_size
                    || new_tick_size <= Decimal::ZERO
                    || new_tick_size >= Decimal::ONE
                    || new_tick_size.scale() > 8
                {
                    return Err(source_error(
                        "polymarket_clob_tick_size_mismatch",
                        "CLOB tick-size transition did not match registered book state",
                    ));
                }
                let sequence = self.take_sequence()?;
                let book = self
                    .books
                    .get_mut(&token_id)
                    .expect("tick-size book identity was checked");
                book.tick_size = new_tick_size;
                book.market.tick_size = new_tick_size;
                book.source_timestamp = Some(source_timestamp);
                book.received_at = Some(received_at);
                // The tick event has no provider book hash. Once it advances
                // the causal timestamp, retaining the older hash would falsely
                // associate that hash with the newer state.
                book.source_hash = None;
                book.ingest_sequence = sequence;
                Ok(ApplyOutcome::Applied)
            }
            ClobMessage::Auxiliary {
                market_id,
                token_id,
                source_timestamp,
            } => {
                if let Some(token_id) = token_id {
                    let Some(book) = self.books.get(&token_id) else {
                        return self.foreign_or_mismatch(&market_id, &token_id);
                    };
                    if !book.market.matches_wire_market(&market_id) {
                        return Err(source_error(
                            "polymarket_clob_market_mismatch",
                            "CLOB auxiliary event market did not match its registered token",
                        ));
                    }
                    if book.is_stale(source_timestamp) {
                        return Ok(ApplyOutcome::NonMutating);
                    }
                }
                Ok(ApplyOutcome::NonMutating)
            }
            ClobMessage::Control => Ok(ApplyOutcome::NonMutating),
        }
    }

    fn apply_price_changes(
        &mut self,
        market_id: &str,
        changes: Vec<PriceChange>,
        source_timestamp: DateTime<Utc>,
        received_at: DateTime<Utc>,
        max_levels: usize,
    ) -> Result<ApplyOutcome, StrategyError> {
        let market_known = self
            .books
            .values()
            .any(|book| book.market.matches_wire_market(market_id));
        let mut grouped = BTreeMap::<String, Vec<PriceChange>>::new();
        for change in changes {
            grouped
                .entry(change.token_id.clone())
                .or_default()
                .push(change);
        }
        let mut candidates = Vec::with_capacity(grouped.len());
        for (token_id, token_changes) in grouped {
            let Some(book) = self.books.get(&token_id) else {
                if market_known {
                    return Err(source_error(
                        "polymarket_clob_token_mismatch",
                        "CLOB price_change used an unknown token for a subscribed market",
                    ));
                }
                continue;
            };
            if !book.market.matches_wire_market(market_id) {
                return Err(source_error(
                    "polymarket_clob_market_mismatch",
                    "CLOB price_change market did not match its registered token",
                ));
            }
            if !book.bootstrapped {
                // No part of a grouped provider update is installed until every
                // subscribed token group has an initial full snapshot.
                return Ok(ApplyOutcome::AwaitingSnapshot);
            }
            if book.is_stale(source_timestamp) {
                return Ok(ApplyOutcome::NonMutating);
            }
            let mut candidate = book.clone();
            for change in &token_changes {
                let side = match change.side {
                    BookSide::Bid => &mut candidate.bids,
                    BookSide::Ask => &mut candidate.asks,
                };
                if change.size.is_zero() {
                    side.remove(&change.price);
                } else {
                    side.insert(change.price, change.size);
                }
            }
            let best_bid = token_changes
                .iter()
                .rev()
                .find_map(|change| change.best_bid);
            let best_ask = token_changes
                .iter()
                .rev()
                .find_map(|change| change.best_ask);
            candidate.reconcile_advertised_top(best_bid, best_ask, max_levels)?;
            candidate.source_timestamp = Some(source_timestamp);
            candidate.received_at = Some(received_at);
            candidate.source_hash = token_changes
                .iter()
                .rev()
                .find_map(|change| change.source_hash.clone());
            candidates.push((token_id, candidate));
        }
        if candidates.is_empty() {
            return Ok(ApplyOutcome::IgnoredForeign);
        }
        // Allocate and commit only after every token group has validated, so a
        // malformed sibling change cannot leave a partially applied frame.
        for (_, candidate) in &mut candidates {
            candidate.ingest_sequence = self.take_sequence()?;
        }
        for (token_id, candidate) in candidates {
            self.books.insert(token_id, candidate);
        }
        Ok(ApplyOutcome::Applied)
    }

    fn foreign_or_mismatch(
        &self,
        market_id: &str,
        token_id: &str,
    ) -> Result<ApplyOutcome, StrategyError> {
        if self
            .books
            .values()
            .any(|book| book.market.matches_wire_market(market_id))
        {
            Err(source_error(
                "polymarket_clob_token_mismatch",
                format!("subscribed market received unexpected token {token_id}"),
            ))
        } else {
            Ok(ApplyOutcome::IgnoredForeign)
        }
    }

    fn take_sequence(&mut self) -> Result<i64, StrategyError> {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.checked_add(1).ok_or_else(|| {
            integrity_error(
                "polymarket_clob_sequence_overflow",
                "CLOB connection ingest sequence overflowed",
            )
        })?;
        Ok(sequence)
    }

    fn samples(&self, top_n: usize) -> Vec<BookSample> {
        let mut samples = self
            .books
            .values()
            .filter(|book| book.bootstrapped)
            .filter_map(|book| {
                Some(BookSample {
                    market: book.market.clone(),
                    token_id: book.token_id.clone(),
                    outcome: book.outcome,
                    tick_size: book.tick_size,
                    source_timestamp: book.source_timestamp?,
                    received_at: book.received_at?,
                    source_hash: book.source_hash.clone(),
                    ingest_sequence: book.ingest_sequence,
                    bids: book
                        .bids
                        .iter()
                        .rev()
                        .take(top_n)
                        .map(|(price, size)| [decimal_string(*price), decimal_string(*size)])
                        .collect(),
                    asks: book
                        .asks
                        .iter()
                        .take(top_n)
                        .map(|(price, size)| [decimal_string(*price), decimal_string(*size)])
                        .collect(),
                })
            })
            .collect::<Vec<_>>();
        samples.sort_by(|left, right| {
            (
                left.market.window_start,
                &left.market.market_id,
                left.outcome,
            )
                .cmp(&(
                    right.market.window_start,
                    &right.market.market_id,
                    right.outcome,
                ))
        });
        samples
    }
}

#[derive(Debug, Clone, PartialEq)]
struct BookSample {
    market: MarketContract,
    token_id: String,
    outcome: Outcome,
    tick_size: Decimal,
    source_timestamp: DateTime<Utc>,
    received_at: DateTime<Utc>,
    source_hash: Option<String>,
    ingest_sequence: i64,
    bids: Vec<[String; 2]>,
    asks: Vec<[String; 2]>,
}

fn decimal_string(value: Decimal) -> String {
    value.normalize().to_string()
}

fn clob_assets(markets: &[MarketContract]) -> Vec<String> {
    let mut assets = markets
        .iter()
        .flat_map(|market| [&market.up_token_id, &market.down_token_id])
        .cloned()
        .collect::<Vec<_>>();
    assets.sort_unstable();
    assets.dedup();
    assets
}

fn clob_subscription(markets: &[MarketContract]) -> String {
    json!({
        "assets_ids": clob_assets(markets),
        "type": "market",
        "custom_feature_enabled": true,
        "initial_dump": true
    })
    .to_string()
}

fn clob_subscription_operation(assets: &[String], subscribe: bool) -> String {
    if subscribe {
        json!({
            "assets_ids": assets,
            "operation": "subscribe",
            "custom_feature_enabled": true,
            "initial_dump": true
        })
    } else {
        json!({
            "assets_ids": assets,
            "operation": "unsubscribe"
        })
    }
    .to_string()
}

#[derive(Debug, PartialEq, Eq)]
struct SubscriptionDelta {
    added: Vec<String>,
    removed: Vec<String>,
}

fn subscription_delta(active: &[MarketContract], desired: &[MarketContract]) -> SubscriptionDelta {
    let active_assets = clob_assets(active);
    let desired_assets = clob_assets(desired);
    SubscriptionDelta {
        added: desired_assets
            .iter()
            .filter(|asset| active_assets.binary_search(asset).is_err())
            .cloned()
            .collect(),
        removed: active_assets
            .iter()
            .filter(|asset| desired_assets.binary_search(asset).is_err())
            .cloned()
            .collect(),
    }
}

#[derive(Debug, Clone)]
struct SnapshotFact {
    sampled_at: DateTime<Utc>,
    source_timestamp: DateTime<Utc>,
    provider_available_at: DateTime<Utc>,
    received_at: DateTime<Utc>,
    market_id: String,
    condition_id: String,
    event_slug: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    token_id: String,
    outcome: &'static str,
    connection_epoch: Uuid,
    ingest_sequence: i64,
    tick_size: Decimal,
    best_bid: Option<Decimal>,
    best_ask: Option<Decimal>,
    bid_depth: i32,
    ask_depth: i32,
    bids: Value,
    asks: Value,
    source_hash: Option<String>,
    book_sha256: String,
    sampling_policy: Value,
    sampling_policy_sha256: String,
    payload_sha256: String,
}

impl SnapshotFact {
    fn new(
        sample: BookSample,
        sampled_at: DateTime<Utc>,
        connection_epoch: Uuid,
        sampling_policy: &Value,
        sampling_policy_sha256: &str,
    ) -> Result<Self, StrategyError> {
        let sampled_at = canonical_timestamp(sampled_at);
        let received_at = sample.received_at.max(sample.market.received_at);
        if received_at > sampled_at {
            return Err(integrity_error(
                "polymarket_sample_causality_violation",
                "orderbook sample receipt timestamp exceeded its selection instant",
            ));
        }
        let bid_depth = i32::try_from(sample.bids.len()).map_err(|_| {
            integrity_error(
                "polymarket_sample_depth_overflow",
                "orderbook bid sample depth exceeded database range",
            )
        })?;
        let ask_depth = i32::try_from(sample.asks.len()).map_err(|_| {
            integrity_error(
                "polymarket_sample_depth_overflow",
                "orderbook ask sample depth exceeded database range",
            )
        })?;
        let best_bid = sample
            .bids
            .first()
            .map(|level| Decimal::from_str(&level[0]))
            .transpose()
            .map_err(|error| {
                integrity_error(
                    "polymarket_sample_serialization",
                    format!("failed to recover sampled best bid: {error}"),
                )
            })?;
        let best_ask = sample
            .asks
            .first()
            .map(|level| Decimal::from_str(&level[0]))
            .transpose()
            .map_err(|error| {
                integrity_error(
                    "polymarket_sample_serialization",
                    format!("failed to recover sampled best ask: {error}"),
                )
            })?;
        if matches!((best_bid, best_ask), (Some(bid), Some(ask)) if bid >= ask) {
            return Err(integrity_error(
                "polymarket_sample_crossed_book",
                "sampled CLOB top was crossed or locked",
            ));
        }
        let bids = serde_json::to_value(&sample.bids).map_err(|error| {
            integrity_error(
                "polymarket_sample_serialization",
                format!("failed to serialize CLOB bids: {error}"),
            )
        })?;
        let asks = serde_json::to_value(&sample.asks).map_err(|error| {
            integrity_error(
                "polymarket_sample_serialization",
                format!("failed to serialize CLOB asks: {error}"),
            )
        })?;
        let book_sha256 = hash_json(&json!({"bids": bids, "asks": asks}))?;
        let payload_sha256 = hash_json(&json!({
            "schema_version": 1,
            "sampled_at": sampled_at,
            "source": SOURCE,
            "source_timestamp": sample.source_timestamp,
            "market_id": sample.market.market_id,
            "condition_id": sample.market.condition_id,
            "event_slug": sample.market.event_slug,
            "window_start": sample.market.window_start,
            "window_end": sample.market.window_end,
            "token_id": sample.token_id,
            "outcome": sample.outcome.as_str(),
            "tick_size": decimal_string(sample.tick_size),
            "best_bid": best_bid.map(decimal_string),
            "best_ask": best_ask.map(decimal_string),
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "bids": bids,
            "asks": asks,
            "source_hash": sample.source_hash,
            "book_sha256": book_sha256,
            "sampling_policy_sha256": sampling_policy_sha256
        }))?;
        Ok(Self {
            sampled_at,
            source_timestamp: sample.source_timestamp,
            provider_available_at: sample.source_timestamp,
            received_at,
            market_id: sample.market.market_id,
            condition_id: sample.market.condition_id,
            event_slug: sample.market.event_slug,
            window_start: sample.market.window_start,
            window_end: sample.market.window_end,
            token_id: sample.token_id,
            outcome: sample.outcome.as_str(),
            connection_epoch,
            ingest_sequence: sample.ingest_sequence,
            tick_size: sample.tick_size,
            best_bid,
            best_ask,
            bid_depth,
            ask_depth,
            bids,
            asks,
            source_hash: sample.source_hash,
            book_sha256,
            sampling_policy: sampling_policy.clone(),
            sampling_policy_sha256: sampling_policy_sha256.to_owned(),
            payload_sha256,
        })
    }

    fn cursor(&self) -> String {
        format!(
            "sample:{}:{}:{}",
            self.sampled_at.timestamp_micros(),
            self.market_id,
            self.token_id
        )
    }
}

#[derive(Debug, FromRow)]
struct ExistingSnapshotFact {
    source_timestamp: DateTime<Utc>,
    provider_available_at: DateTime<Utc>,
    source: String,
    condition_id: String,
    event_slug: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    outcome: String,
    tick_size: Decimal,
    best_bid: Option<Decimal>,
    best_ask: Option<Decimal>,
    bid_depth: i32,
    ask_depth: i32,
    bids: Value,
    asks: Value,
    source_hash: Option<String>,
    book_sha256: String,
    sampling_policy: Value,
    payload_sha256: String,
}

impl ExistingSnapshotFact {
    fn matches(&self, fact: &SnapshotFact) -> bool {
        self.source_timestamp == fact.source_timestamp
            && self.provider_available_at == fact.provider_available_at
            && self.source == SOURCE
            && self.condition_id == fact.condition_id
            && self.event_slug == fact.event_slug
            && self.window_start == fact.window_start
            && self.window_end == fact.window_end
            && self.outcome == fact.outcome
            && self.tick_size == fact.tick_size
            && self.best_bid == fact.best_bid
            && self.best_ask == fact.best_ask
            && self.bid_depth == fact.bid_depth
            && self.ask_depth == fact.ask_depth
            && self.bids == fact.bids
            && self.asks == fact.asks
            && self.source_hash == fact.source_hash
            && self.book_sha256 == fact.book_sha256
            && self.sampling_policy == fact.sampling_policy
            && self.payload_sha256 == fact.payload_sha256
    }
}

#[derive(Debug, FromRow)]
struct ArtifactHashRow {
    sampled_at: DateTime<Utc>,
    market_id: String,
    token_id: String,
    payload_sha256: String,
}

#[derive(Debug)]
struct ArtifactSeal {
    content_sha256: String,
    end_cursor: Option<String>,
    record_count: i64,
}

#[derive(Debug)]
struct OpenArtifact {
    artifact: CaptureArtifact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArtifactSealFence {
    CurrentProfile,
    OwnedDrain,
}

struct GapObservation<'a> {
    kind: &'a str,
    code: &'a str,
    message: &'a str,
    source_start: Option<DateTime<Utc>>,
    source_end: Option<DateTime<Utc>>,
    start_cursor: Option<String>,
    end_cursor: Option<String>,
}

struct CaptureWriter {
    pool: PgPool,
    artifacts: ArtifactRepository,
    gaps: GapRepository,
    profiles: ProfileRepository,
    config: PolymarketBtcFiveMinuteOrderbooksConfig,
    config_snapshot: Value,
    profile_generation: i64,
    lease_owner: Arc<str>,
    lease_token: Uuid,
    sampling_policy: Value,
    sampling_policy_sha256: String,
    current: Option<OpenArtifact>,
}

impl CaptureWriter {
    fn new(strategy: &PolymarketBtcFiveMinuteOrderbooksStrategy) -> Result<Self, StrategyError> {
        let sampling_policy = strategy.config.sampling_policy();
        let sampling_policy_sha256 = hash_json(&sampling_policy)?;
        Ok(Self {
            pool: strategy.pool.clone(),
            artifacts: ArtifactRepository::new(strategy.pool.clone()),
            gaps: GapRepository::new(strategy.pool.clone()),
            profiles: ProfileRepository::new(strategy.pool.clone()),
            config: strategy.config.clone(),
            config_snapshot: strategy.config_snapshot.clone(),
            profile_generation: strategy.profile_generation,
            lease_owner: Arc::clone(&strategy.lease_owner),
            lease_token: strategy.lease_token,
            sampling_policy,
            sampling_policy_sha256,
            current: None,
        })
    }

    async fn initialize(&mut self) -> Result<(), StrategyError> {
        let Some(open) = self
            .artifacts
            .get_open(STRATEGY_KEY)
            .await
            .map_err(|error| {
                database_error(
                    "polymarket_artifact_load_failed",
                    format!("failed to load open Polymarket artifact: {error}"),
                )
            })?
        else {
            return Ok(());
        };
        let generation_changed =
            fence_open_artifact_generation(open.profile_generation, self.profile_generation)?;
        if !generation_changed
            && (open.config_schema_version != CONFIG_SCHEMA_VERSION
                || open.config_snapshot != self.config_snapshot)
        {
            return Err(integrity_error(
                "polymarket_artifact_config_mismatch",
                "open Polymarket artifact config differs within one profile generation",
            ));
        }
        let rotate = generation_changed || open.capture_window_end <= Utc::now();
        self.current = Some(OpenArtifact { artifact: open });
        if rotate {
            self.seal_current(ArtifactSealFence::CurrentProfile).await?;
        }
        Ok(())
    }

    async fn ensure_artifact(
        &mut self,
        sampled_at: DateTime<Utc>,
        start_cursor: &str,
    ) -> Result<(), StrategyError> {
        if let Some(current) = self.current.as_ref() {
            if sampled_at < current.artifact.capture_window_start {
                return Err(integrity_error(
                    "polymarket_sample_clock_regression",
                    "sample timestamp regressed before the current artifact window",
                ));
            }
            if sampled_at < current.artifact.capture_window_end {
                return Ok(());
            }
            self.seal_current(ArtifactSealFence::CurrentProfile).await?;
        }
        let (capture_window_start, capture_window_end) =
            aligned_artifact_window(sampled_at, self.config.artifact_window_seconds)?;
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error(
                "polymarket_artifact_create_transaction_failed",
                format!("failed to begin Polymarket artifact transaction: {error}"),
            )
        })?;
        self.lock_current_lease(&mut transaction, "creating a capture artifact")
            .await?;
        let artifact = self
            .artifacts
            .create_in(
                &mut transaction,
                &NewCaptureArtifact {
                    strategy_key: STRATEGY_KEY,
                    profile_generation: self.profile_generation,
                    config_schema_version: CONFIG_SCHEMA_VERSION,
                    config_snapshot: self.config_snapshot.clone(),
                    capture_window_start,
                    capture_window_end,
                    start_cursor: Some(start_cursor.to_owned()),
                },
            )
            .await
            .map_err(|error| {
                database_error(
                    "polymarket_artifact_create_failed",
                    format!("failed to create Polymarket artifact: {error}"),
                )
            })?;
        transaction.commit().await.map_err(|error| {
            database_error(
                "polymarket_artifact_create_commit_failed",
                format!("failed to commit Polymarket artifact creation: {error}"),
            )
        })?;
        self.current = Some(OpenArtifact { artifact });
        Ok(())
    }

    async fn seal_current(&mut self, fence: ArtifactSealFence) -> Result<(), StrategyError> {
        let Some(current) = self.current.as_ref() else {
            return Ok(());
        };
        let artifact_id = current.artifact.artifact_id;
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error(
                "polymarket_artifact_complete_transaction_failed",
                format!("failed to begin Polymarket artifact completion: {error}"),
            )
        })?;
        match fence {
            ArtifactSealFence::CurrentProfile => {
                self.lock_current_lease(&mut transaction, "completing a capture artifact")
                    .await?;
            }
            ArtifactSealFence::OwnedDrain => {
                self.lock_owned_lease(&mut transaction, "draining a capture artifact")
                    .await?;
            }
        }
        let durable = self
            .artifacts
            .get_in(&mut transaction, artifact_id)
            .await
            .map_err(|error| {
                database_error(
                    "polymarket_artifact_refresh_failed",
                    format!("failed to refresh Polymarket artifact: {error}"),
                )
            })?
            .ok_or_else(|| {
                integrity_error(
                    "polymarket_artifact_missing",
                    "open Polymarket artifact disappeared before completion",
                )
            })?;
        let seal = artifact_seal_in(&mut transaction, artifact_id).await?;
        if durable.record_count != seal.record_count {
            return Err(integrity_error(
                "polymarket_artifact_count_mismatch",
                format!(
                    "artifact {artifact_id} declares {} rows but contains {}",
                    durable.record_count, seal.record_count
                ),
            ));
        }
        let completed = self
            .artifacts
            .complete_in(
                &mut transaction,
                artifact_id,
                &seal.content_sha256,
                seal.end_cursor.as_deref(),
            )
            .await
            .map_err(|error| {
                database_error(
                    "polymarket_artifact_complete_failed",
                    format!("failed to complete Polymarket artifact: {error}"),
                )
            })?;
        if completed.is_none() {
            return Err(integrity_error(
                "polymarket_artifact_not_open",
                "Polymarket artifact was no longer open during completion",
            ));
        }
        transaction.commit().await.map_err(|error| {
            database_error(
                "polymarket_artifact_complete_commit_failed",
                format!("failed to commit Polymarket artifact completion: {error}"),
            )
        })?;
        self.current = None;
        info!(artifact_id = %artifact_id, record_count = seal.record_count, "completed Polymarket orderbook artifact");
        Ok(())
    }

    async fn seal_owned_drain(&mut self) -> Result<(), StrategyError> {
        if self.current.is_some() {
            return self.seal_current(ArtifactSealFence::OwnedDrain).await;
        }
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error(
                "polymarket_owned_drain_transaction_failed",
                format!("failed to begin empty Polymarket drain: {error}"),
            )
        })?;
        self.lock_owned_lease(&mut transaction, "confirming an empty owned drain")
            .await?;
        transaction.commit().await.map_err(|error| {
            database_error(
                "polymarket_owned_drain_commit_failed",
                format!("failed to commit empty Polymarket drain: {error}"),
            )
        })?;
        Ok(())
    }

    async fn persist_samples(
        &mut self,
        samples: Vec<BookSample>,
        sampled_at: DateTime<Utc>,
        connection_epoch: Uuid,
    ) -> Result<Vec<SnapshotFact>, StrategyError> {
        let mut facts = samples
            .into_iter()
            .map(|sample| {
                SnapshotFact::new(
                    sample,
                    sampled_at,
                    connection_epoch,
                    &self.sampling_policy,
                    &self.sampling_policy_sha256,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if facts.is_empty() {
            return Ok(facts);
        }
        facts.sort_by(|left, right| {
            (&left.market_id, &left.token_id).cmp(&(&right.market_id, &right.token_id))
        });
        let start_cursor = facts[0].cursor();
        self.ensure_artifact(sampled_at, &start_cursor).await?;
        let artifact_id = self
            .current
            .as_ref()
            .map(|current| current.artifact.artifact_id)
            .ok_or_else(|| {
                integrity_error(
                    "polymarket_artifact_missing",
                    "Polymarket samples have no open capture artifact",
                )
            })?;
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error(
                "polymarket_fact_transaction_failed",
                format!("failed to begin Polymarket fact transaction: {error}"),
            )
        })?;
        self.lock_current_lease(&mut transaction, "persisting orderbook samples")
            .await?;
        let mut advisory_keys = facts.iter().map(fact_advisory_key).collect::<Vec<_>>();
        advisory_keys.sort_unstable();
        advisory_keys.dedup();
        for key in advisory_keys {
            sqlx::query("SELECT pg_advisory_xact_lock($1)")
                .bind(key)
                .execute(&mut *transaction)
                .await
                .map_err(|error| {
                    database_error(
                        "polymarket_fact_lock_failed",
                        format!("failed to lock Polymarket sample identity: {error}"),
                    )
                })?;
        }
        let mut inserted = Vec::with_capacity(facts.len());
        for fact in &facts {
            let existing = sqlx::query_as::<_, ExistingSnapshotFact>(
                r#"
                SELECT source_timestamp, provider_available_at, source,
                       condition_id, event_slug, window_start, window_end,
                       outcome, tick_size, best_bid, best_ask, bid_depth,
                       ask_depth, bids, asks, source_hash, book_sha256,
                       sampling_policy, payload_sha256
                FROM market_data.polymarket_btc_five_minute_orderbook_snapshots
                WHERE sampled_at = $1
                  AND market_id = $2
                  AND token_id = $3
                  AND sampling_policy_sha256 = $4
                "#,
            )
            .bind(fact.sampled_at)
            .bind(&fact.market_id)
            .bind(&fact.token_id)
            .bind(&fact.sampling_policy_sha256)
            .fetch_optional(&mut *transaction)
            .await
            .map_err(|error| {
                database_error(
                    "polymarket_fact_lookup_failed",
                    format!("failed to check Polymarket sample identity: {error}"),
                )
            })?;
            match existing {
                Some(existing) if existing.matches(fact) => {}
                Some(existing) => {
                    return Err(integrity_error(
                        "polymarket_conflicting_sample",
                        format!(
                            "Polymarket sample {}:{} conflicts with payload {}",
                            fact.market_id, fact.token_id, existing.payload_sha256
                        ),
                    ));
                }
                None => {
                    insert_fact(&mut transaction, fact, artifact_id).await?;
                    inserted.push(fact);
                }
            }
        }
        let mut updated_artifact = None;
        if !inserted.is_empty() {
            let first_cursor = inserted[0].cursor();
            let last_cursor = inserted
                .last()
                .expect("inserted fact collection is nonempty")
                .cursor();
            let minimum_source = inserted.iter().map(|fact| fact.source_timestamp).min();
            let maximum_source = inserted.iter().map(|fact| fact.source_timestamp).max();
            let minimum_received = inserted.iter().map(|fact| fact.received_at).min();
            let maximum_received = inserted.iter().map(|fact| fact.received_at).max();
            updated_artifact = self
                .artifacts
                .record_batch_in(
                    &mut transaction,
                    artifact_id,
                    &ArtifactBatch {
                        inserted_record_count: i64::try_from(inserted.len()).map_err(|_| {
                            integrity_error(
                                "polymarket_artifact_count_overflow",
                                "Polymarket sample batch count exceeded database range",
                            )
                        })?,
                        minimum_source_timestamp: minimum_source,
                        maximum_source_timestamp: maximum_source,
                        minimum_received_at: minimum_received,
                        maximum_received_at: maximum_received,
                        start_cursor: Some(first_cursor),
                        end_cursor: Some(last_cursor),
                    },
                )
                .await
                .map_err(|error| {
                    database_error(
                        "polymarket_artifact_progress_failed",
                        format!("failed to update Polymarket artifact lineage: {error}"),
                    )
                })?;
            if updated_artifact.is_none() {
                return Err(integrity_error(
                    "polymarket_artifact_not_open",
                    "Polymarket artifact closed during sample transaction",
                ));
            }
        }
        let latest_source = facts.iter().map(|fact| fact.source_timestamp).max();
        let latest_availability = facts.iter().map(|fact| fact.provider_available_at).max();
        let progress = StrategyProgress {
            verified_record_count: i64::try_from(facts.len()).map_err(|_| {
                integrity_error(
                    "polymarket_progress_count_overflow",
                    "Polymarket verified sample count exceeded database range",
                )
            })?,
            checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
            checkpoint: json!({
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "sampled_at": sampled_at,
                "connection_epoch": connection_epoch,
                "artifact_id": artifact_id,
                "sampling_policy_sha256": self.sampling_policy_sha256,
                "books": facts.iter().map(|fact| json!({
                    "market_id": fact.market_id,
                    "token_id": fact.token_id,
                    "source_timestamp": fact.source_timestamp,
                    "ingest_sequence": fact.ingest_sequence,
                    "payload_sha256": fact.payload_sha256
                })).collect::<Vec<_>>()
            }),
            last_source_event_at: latest_source,
            last_provider_available_at: latest_availability,
            source_watermark: latest_source,
            availability_watermark: latest_availability,
        };
        let recorded = self
            .profiles
            .record_progress_in(
                &mut transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &progress,
            )
            .await
            .map_err(|error| {
                database_error(
                    "polymarket_profile_progress_failed",
                    format!("failed to update Polymarket profile progress: {error}"),
                )
            })?;
        if !recorded {
            return Err(lease_error(
                "polymarket_lease_lost",
                "Polymarket lease was lost before committing sample progress",
            ));
        }
        transaction.commit().await.map_err(|error| {
            database_error(
                "polymarket_fact_commit_failed",
                format!("failed to commit Polymarket samples: {error}"),
            )
        })?;
        if let Some(artifact) = updated_artifact {
            if let Some(current) = self.current.as_mut() {
                current.artifact = artifact;
            }
        }
        Ok(facts)
    }

    async fn record_gap(&mut self, gap: GapObservation<'_>) -> Result<(), StrategyError> {
        let (source_start, source_end) = match (gap.source_start, gap.source_end) {
            (Some(start), Some(end)) if end >= start => (Some(start), Some(end)),
            _ => (None, None),
        };
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error(
                "polymarket_gap_transaction_failed",
                format!("failed to begin Polymarket gap transaction: {error}"),
            )
        })?;
        self.lock_current_lease(&mut transaction, "recording a data gap")
            .await?;
        let detection = self
            .gaps
            .detect_in(
                &mut transaction,
                &NewDataGap {
                    strategy_key: STRATEGY_KEY,
                    detected_artifact_id: self
                        .current
                        .as_ref()
                        .map(|current| current.artifact.artifact_id),
                    gap_kind: gap.kind.to_owned(),
                    reason_code: gap.code.to_owned(),
                    reason_message: Some(gap.message.to_owned()),
                    source_time_start: source_start,
                    source_time_end: source_end,
                    start_cursor: gap.start_cursor,
                    end_cursor: gap.end_cursor,
                },
            )
            .await
            .map_err(|error| {
                database_error(
                    "polymarket_gap_persist_failed",
                    format!("failed to persist Polymarket gap: {error}"),
                )
            })?;
        let degraded = self
            .profiles
            .mark_degraded_in(
                &mut transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &StrategyDegradation {
                    reason_code: gap.code.to_owned(),
                    reason_message: gap.message.to_owned(),
                },
            )
            .await
            .map_err(|error| {
                database_error(
                    "polymarket_degraded_state_failed",
                    format!("failed to mark Polymarket strategy degraded: {error}"),
                )
            })?;
        if !degraded {
            return Err(lease_error(
                "polymarket_lease_lost",
                "Polymarket lease was lost while recording a data gap",
            ));
        }
        if !detection.gap.status.is_resolved() {
            let resolved = self
                .gaps
                .mark_unrecoverable_in(
                    &mut transaction,
                    detection.gap.gap_id,
                    "realtime_resubscribe_only",
                    Some(
                        "Polymarket public CLOB does not expose historical websocket book replay; continuity resumes from a new full snapshot",
                    ),
                )
                .await
                .map_err(|error| {
                    database_error(
                        "polymarket_gap_resolution_failed",
                        format!("failed to terminalize Polymarket gap: {error}"),
                    )
                })?;
            if resolved.is_none() {
                return Err(integrity_error(
                    "polymarket_gap_resolution_conflict",
                    "Polymarket gap changed before it could be marked unrecoverable",
                ));
            }
        }
        transaction.commit().await.map_err(|error| {
            database_error(
                "polymarket_gap_commit_failed",
                format!("failed to commit Polymarket gap: {error}"),
            )
        })?;
        Ok(())
    }

    async fn lock_current_lease(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        action: &str,
    ) -> Result<(), StrategyError> {
        let locked = self
            .profiles
            .lock_current_lease_in(
                transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
            )
            .await
            .map_err(|error| {
                database_error(
                    "polymarket_lease_lock_failed",
                    format!("failed to lock Polymarket profile lease: {error}"),
                )
            })?;
        if !locked {
            return Err(lease_error(
                "polymarket_lease_lost",
                format!("Polymarket lease was lost before {action}"),
            ));
        }
        Ok(())
    }

    async fn lock_owned_lease(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        action: &str,
    ) -> Result<(), StrategyError> {
        let locked = self
            .profiles
            .lock_owned_lease_in(
                transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
            )
            .await
            .map_err(|error| {
                database_error(
                    "polymarket_owned_lease_lock_failed",
                    format!("failed to lock owned Polymarket lease: {error}"),
                )
            })?;
        if !locked {
            return Err(lease_error(
                "polymarket_lease_lost",
                format!("Polymarket lease was lost before {action}"),
            ));
        }
        Ok(())
    }
}

async fn insert_fact(
    transaction: &mut Transaction<'_, Postgres>,
    fact: &SnapshotFact,
    artifact_id: Uuid,
) -> Result<(), StrategyError> {
    sqlx::query(
        r#"
        INSERT INTO market_data.polymarket_btc_five_minute_orderbook_snapshots (
          sampled_at, source_timestamp, provider_available_at, received_at,
          source, market_id, condition_id, event_slug, window_start,
          window_end, token_id, outcome, connection_epoch, ingest_sequence,
          tick_size, best_bid, best_ask, bid_depth, ask_depth, bids, asks,
          source_hash, book_sha256, sampling_policy,
          sampling_policy_sha256, payload_sha256, strategy_key,
          capture_artifact_id
        ) VALUES (
          $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
          $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24,
          $25, $26, $27, $28
        )
        "#,
    )
    .bind(fact.sampled_at)
    .bind(fact.source_timestamp)
    .bind(fact.provider_available_at)
    .bind(fact.received_at)
    .bind(SOURCE)
    .bind(&fact.market_id)
    .bind(&fact.condition_id)
    .bind(&fact.event_slug)
    .bind(fact.window_start)
    .bind(fact.window_end)
    .bind(&fact.token_id)
    .bind(fact.outcome)
    .bind(fact.connection_epoch)
    .bind(fact.ingest_sequence)
    .bind(fact.tick_size)
    .bind(fact.best_bid)
    .bind(fact.best_ask)
    .bind(fact.bid_depth)
    .bind(fact.ask_depth)
    .bind(&fact.bids)
    .bind(&fact.asks)
    .bind(fact.source_hash.as_deref())
    .bind(&fact.book_sha256)
    .bind(&fact.sampling_policy)
    .bind(&fact.sampling_policy_sha256)
    .bind(&fact.payload_sha256)
    .bind(STRATEGY_KEY.as_str())
    .bind(artifact_id)
    .execute(&mut **transaction)
    .await
    .map_err(|error| {
        database_error(
            "polymarket_fact_insert_failed",
            format!("failed to insert Polymarket orderbook sample: {error}"),
        )
    })?;
    Ok(())
}

async fn artifact_seal_in(
    transaction: &mut Transaction<'_, Postgres>,
    artifact_id: Uuid,
) -> Result<ArtifactSeal, StrategyError> {
    let rows = sqlx::query_as::<_, ArtifactHashRow>(
        r#"
        SELECT sampled_at, market_id, token_id, payload_sha256
        FROM market_data.polymarket_btc_five_minute_orderbook_snapshots
        WHERE capture_artifact_id = $1
        ORDER BY sampled_at, market_id, token_id, sampling_policy_sha256
        "#,
    )
    .bind(artifact_id)
    .fetch_all(&mut **transaction)
    .await
    .map_err(|error| {
        database_error(
            "polymarket_artifact_checksum_failed",
            format!("failed to compute Polymarket artifact checksum: {error}"),
        )
    })?;
    let record_count = i64::try_from(rows.len()).map_err(|_| {
        integrity_error(
            "polymarket_artifact_count_overflow",
            "Polymarket artifact row count exceeded database range",
        )
    })?;
    let mut hasher = Sha256::new();
    for row in &rows {
        append_artifact_hash(&mut hasher, &row.payload_sha256);
    }
    let end_cursor = rows.last().map(|row| {
        format!(
            "sample:{}:{}:{}",
            row.sampled_at.timestamp_micros(),
            row.market_id,
            row.token_id
        )
    });
    Ok(ArtifactSeal {
        content_sha256: encode_digest(hasher.finalize()),
        end_cursor,
        record_count,
    })
}

fn aligned_artifact_window(
    timestamp: DateTime<Utc>,
    window_seconds: i64,
) -> Result<(DateTime<Utc>, DateTime<Utc>), StrategyError> {
    let start_seconds = timestamp.timestamp().div_euclid(window_seconds) * window_seconds;
    let end_seconds = start_seconds.checked_add(window_seconds).ok_or_else(|| {
        integrity_error(
            "polymarket_artifact_window_overflow",
            "Polymarket artifact window overflowed",
        )
    })?;
    let start = Utc
        .timestamp_opt(start_seconds, 0)
        .single()
        .ok_or_else(|| {
            integrity_error(
                "polymarket_artifact_window_invalid",
                "Polymarket artifact start was outside UTC range",
            )
        })?;
    let end = Utc.timestamp_opt(end_seconds, 0).single().ok_or_else(|| {
        integrity_error(
            "polymarket_artifact_window_invalid",
            "Polymarket artifact end was outside UTC range",
        )
    })?;
    Ok((start, end))
}

fn fence_open_artifact_generation(
    artifact_generation: i64,
    strategy_generation: i64,
) -> Result<bool, StrategyError> {
    if artifact_generation > strategy_generation {
        return Err(lease_error(
            "polymarket_newer_artifact_generation",
            format!(
                "open Polymarket artifact generation {artifact_generation} is newer than owned generation {strategy_generation}"
            ),
        ));
    }
    Ok(artifact_generation < strategy_generation)
}

fn fact_advisory_key(fact: &SnapshotFact) -> i64 {
    let identity = format!(
        "{}:{}:{}:{}",
        fact.sampled_at.timestamp_micros(),
        fact.market_id,
        fact.token_id,
        fact.sampling_policy_sha256
    );
    let digest = Sha256::digest(identity.as_bytes());
    i64::from_be_bytes(digest[..8].try_into().expect("SHA-256 has eight bytes"))
}

fn hash_json(value: &Value) -> Result<String, StrategyError> {
    let canonical = canonical_json(value);
    let encoded = serde_json::to_vec(&canonical).map_err(|error| {
        integrity_error(
            "polymarket_hash_serialization",
            format!("failed to serialize canonical Polymarket payload: {error}"),
        )
    })?;
    Ok(encode_digest(Sha256::digest(encoded)))
}

fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut keys = object.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let mut canonical = Map::with_capacity(object.len());
            for key in keys {
                canonical.insert(key.clone(), canonical_json(&object[key]));
            }
            Value::Object(canonical)
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical_json).collect()),
        _ => value.clone(),
    }
}

fn append_artifact_hash(hasher: &mut Sha256, payload_sha256: &str) {
    hasher.update(payload_sha256.as_bytes());
    hasher.update(b"\n");
}

fn encode_digest(digest: impl AsRef<[u8]>) -> String {
    let digest = digest.as_ref();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

#[derive(Debug, Default)]
struct SamplingClock {
    last_bucket: Option<i64>,
    has_durable_sample: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SamplingClaim {
    should_sample: bool,
    missed_buckets: Option<(i64, i64)>,
}

impl SamplingClock {
    fn claim(&mut self, now: DateTime<Utc>, interval_ms: u64) -> SamplingClaim {
        let Ok(interval_ms) = i64::try_from(interval_ms) else {
            return SamplingClaim {
                should_sample: false,
                missed_buckets: None,
            };
        };
        let bucket = now.timestamp_millis().div_euclid(interval_ms);
        if self.last_bucket.is_some_and(|last| bucket <= last) {
            return SamplingClaim {
                should_sample: false,
                missed_buckets: None,
            };
        }
        let missed_buckets = self.last_bucket.and_then(|last| {
            (self.has_durable_sample && bucket > last.saturating_add(1))
                .then_some((last.saturating_add(1), bucket.saturating_sub(1)))
        });
        self.last_bucket = Some(bucket);
        SamplingClaim {
            should_sample: true,
            missed_buckets,
        }
    }

    fn mark_durable(&mut self) {
        self.has_durable_sample = true;
    }
}

#[derive(Debug, Default)]
struct Continuity {
    last_sampled_at: Option<DateTime<Utc>>,
    last_source_timestamp: Option<DateTime<Utc>>,
    last_received_at: Option<DateTime<Utc>>,
    last_cursor: Option<String>,
}

impl Continuity {
    fn observe(&mut self, facts: &[SnapshotFact]) {
        if let Some(first) = facts.first() {
            self.last_sampled_at = Some(first.sampled_at);
        }
        self.last_source_timestamp = facts
            .iter()
            .map(|fact| fact.source_timestamp)
            .max()
            .or(self.last_source_timestamp);
        self.last_received_at = facts
            .iter()
            .map(|fact| fact.received_at)
            .max()
            .or(self.last_received_at);
        self.last_cursor = facts.last().map(SnapshotFact::cursor);
    }
}

struct DiscoveryWorker {
    shutdown: CancellationToken,
    handle: JoinHandle<()>,
}

impl Drop for DiscoveryWorker {
    fn drop(&mut self) {
        self.shutdown.cancel();
        self.handle.abort();
    }
}

fn start_discovery_worker(
    client: Client,
    config: PolymarketBtcFiveMinuteOrderbooksConfig,
) -> (
    DiscoveryWorker,
    mpsc::Receiver<Result<Vec<MarketContract>, StrategyError>>,
) {
    let shutdown = CancellationToken::new();
    let worker_shutdown = shutdown.clone();
    let (sender, receiver) = mpsc::channel(1);
    let handle = tokio::spawn(async move {
        let interval = Duration::from_millis(config.gamma_refresh_ms);
        loop {
            tokio::select! {
                _ = worker_shutdown.cancelled() => return,
                _ = tokio::time::sleep(interval) => {}
            }
            let result = discover_markets(&client, &config, Utc::now()).await;
            tokio::select! {
                _ = worker_shutdown.cancelled() => return,
                sent = sender.send(result) => {
                    if sent.is_err() {
                        return;
                    }
                }
            }
        }
    });
    (DiscoveryWorker { shutdown, handle }, receiver)
}

#[async_trait]
impl IngesterStrategy for PolymarketBtcFiveMinuteOrderbooksStrategy {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    async fn run(&self, shutdown: CancellationToken) -> Result<(), StrategyError> {
        let mut writer = CaptureWriter::new(self)?;
        writer.initialize().await?;
        let mut sampling_clock = SamplingClock::default();
        let mut continuity = Continuity::default();
        let mut reconnect_delay = Duration::from_millis(self.config.reconnect_initial_ms);
        let maximum_reconnect_delay = Duration::from_millis(self.config.reconnect_max_ms);

        loop {
            if shutdown.is_cancelled() {
                writer.seal_owned_drain().await?;
                return Ok(());
            }
            let connection_epoch = Uuid::new_v4();
            let session_started_at = Instant::now();
            let result = self
                .capture_session(
                    &mut writer,
                    &mut sampling_clock,
                    &mut continuity,
                    connection_epoch,
                    &shutdown,
                )
                .await;
            match result {
                Ok(()) => {
                    writer.seal_owned_drain().await?;
                    return Ok(());
                }
                Err(error) if is_current_profile_lease_loss(&error) => {
                    match writer.seal_owned_drain().await {
                        Ok(()) => {
                            info!(
                                error_code = error.code,
                                "Polymarket strict write observed a requested profile drain"
                            );
                            shutdown.cancelled().await;
                            return Ok(());
                        }
                        Err(drain_error) if drain_error.kind == StrategyErrorKind::LeaseLost => {
                            return Err(error);
                        }
                        Err(drain_error) => return Err(drain_error),
                    }
                }
                Err(error) if error.kind == StrategyErrorKind::TransientSource => {
                    if !gap_was_recorded(error.code) {
                        if let Some(last_sampled_at) = continuity.last_sampled_at {
                            writer
                                .record_gap(GapObservation {
                                    kind: "transport_interruption",
                                    code: error.code,
                                    message: &error.message,
                                    source_start: continuity.last_source_timestamp,
                                    source_end: None,
                                    start_cursor: continuity.last_cursor.clone().or_else(|| {
                                        Some(format!(
                                            "after_sample:{}",
                                            last_sampled_at.timestamp_micros()
                                        ))
                                    }),
                                    end_cursor: None,
                                })
                                .await?;
                        }
                    }
                    warn!(
                        connection_epoch = %connection_epoch,
                        error_code = error.code,
                        error = %error,
                        "Polymarket CLOB session will reconnect with a fresh book epoch"
                    );
                    if session_started_at.elapsed() >= Duration::from_secs(60) {
                        reconnect_delay = Duration::from_millis(self.config.reconnect_initial_ms);
                    }
                    tokio::select! {
                        _ = shutdown.cancelled() => {
                            writer.seal_owned_drain().await?;
                            return Ok(());
                        }
                        _ = tokio::time::sleep(reconnect_delay) => {}
                    }
                    reconnect_delay = reconnect_delay
                        .checked_mul(2)
                        .unwrap_or(maximum_reconnect_delay)
                        .min(maximum_reconnect_delay);
                }
                Err(error) => return Err(error),
            }
        }
    }
}

impl PolymarketBtcFiveMinuteOrderbooksStrategy {
    async fn capture_session(
        &self,
        writer: &mut CaptureWriter,
        sampling_clock: &mut SamplingClock,
        continuity: &mut Continuity,
        connection_epoch: Uuid,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        let discovered = discover_markets(&self.client, &self.config, Utc::now()).await?;
        let mut active_markets = subscription_markets(&discovered, Utc::now(), &self.config);
        let websocket_config = WebSocketConfig::default()
            .read_buffer_size(64 * 1024)
            .write_buffer_size(16 * 1024)
            .max_write_buffer_size(64 * 1024)
            .max_message_size(Some(MAX_WEBSOCKET_FRAME_BYTES))
            .max_frame_size(Some(MAX_WEBSOCKET_FRAME_BYTES));
        let websocket = tokio::time::timeout(
            Duration::from_millis(self.config.connect_timeout_ms),
            connect_async_with_config(&self.config.websocket_url, Some(websocket_config), true),
        )
        .await
        .map_err(|_| {
            source_error(
                "polymarket_clob_connect_timeout",
                "timed out connecting to Polymarket CLOB websocket",
            )
        })?
        .map_err(|error| {
            source_error(
                "polymarket_clob_connect_failed",
                format!("failed to connect Polymarket CLOB websocket: {error}"),
            )
        })?
        .0;
        let (mut sink, mut stream) = websocket.split();
        send_websocket_message(
            &mut sink,
            Message::Text(clob_subscription(&active_markets).into()),
            Duration::from_millis(self.config.connect_timeout_ms),
        )
        .await?;

        let mut registry = BookRegistry::new(connection_epoch, &active_markets)?;
        // A new socket is always a new integrity epoch. No retained level may
        // receive a new-epoch delta before a complete token snapshot.
        registry.reset_connection(connection_epoch);
        let (_discovery_worker, mut discovery_updates) =
            start_discovery_worker(self.client.clone(), self.config.clone());
        let ping_interval = Duration::from_millis(self.config.ping_interval_ms);
        let mut ping = tokio::time::interval_at(Instant::now() + ping_interval, ping_interval);
        ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let sample_interval = Duration::from_millis(self.config.sample_interval_ms);
        let mut sample_tick = tokio::time::interval_at(
            Instant::now() + next_sample_delay(Utc::now(), self.config.sample_interval_ms),
            sample_interval,
        );
        sample_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let read_timeout = Duration::from_millis(self.config.read_timeout_ms);
        let mut read_deadline = Instant::now() + read_timeout;
        let mut pong_deadline = None;
        let mut bootstrap_deadline =
            Some(Instant::now() + Duration::from_millis(self.config.bootstrap_timeout_ms));
        let mut missing_current_since = None;

        loop {
            let disabled_deadline = Instant::now() + Duration::from_secs(86_400);
            let pong_sleep = tokio::time::sleep_until(pong_deadline.unwrap_or(disabled_deadline));
            let bootstrap_sleep =
                tokio::time::sleep_until(bootstrap_deadline.unwrap_or(disabled_deadline));
            tokio::pin!(pong_sleep, bootstrap_sleep);
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => return Ok(()),
                _ = &mut pong_sleep, if pong_deadline.is_some() => {
                    return Err(source_error(
                        "polymarket_clob_pong_timeout",
                        "Polymarket CLOB did not acknowledge the oldest text PING",
                    ));
                }
                _ = &mut bootstrap_sleep, if bootstrap_deadline.is_some() => {
                    let message = "Polymarket CLOB did not deliver every subscribed token's initial full book within the bootstrap bound";
                    writer.record_gap(GapObservation {
                        kind: "snapshot_bootstrap",
                        code: "polymarket_clob_bootstrap_timeout",
                        message,
                        source_start: None,
                        source_end: None,
                        start_cursor: continuity.last_cursor.clone().or_else(|| Some(format!("connection_epoch:{connection_epoch}"))),
                        end_cursor: Some("awaiting_initial_full_books".to_owned()),
                    }).await?;
                    return Err(source_error("polymarket_clob_bootstrap_timeout", message));
                }
                _ = tokio::time::sleep_until(read_deadline) => {
                    return Err(source_error(
                        "polymarket_clob_read_timeout",
                        "Polymarket CLOB websocket produced no frames before its read-idle deadline",
                    ));
                }
                _ = sample_tick.tick() => {
                    let scheduled_at = Utc::now();
                    let claim =
                        sampling_clock.claim(scheduled_at, self.config.sample_interval_ms);
                    if !claim.should_sample {
                        continue;
                    }
                    if let Some((first_missed, last_missed)) = claim.missed_buckets {
                        let message = format!(
                            "local sampler skipped aligned buckets {first_missed} through {last_missed}"
                        );
                        writer.record_gap(GapObservation {
                            kind: "local_sampling_cadence",
                            code: "polymarket_sampling_bucket_gap",
                            message: &message,
                            source_start: None,
                            source_end: None,
                            start_cursor: Some(format!("sampling_bucket:{first_missed}")),
                            end_cursor: Some(format!("sampling_bucket:{last_missed}")),
                        }).await?;
                    }
                    let current_window = aligned_market_window(scheduled_at);
                    let has_current = active_markets
                        .iter()
                        .any(|market| market.window_start == current_window);
                    if has_current {
                        missing_current_since = None;
                    } else {
                        let missing_since = missing_current_since.get_or_insert_with(Instant::now);
                        if missing_since.elapsed() >= Duration::from_millis(self.config.contract_grace_ms) {
                            let message = format!("current Polymarket contract {current_window} remained unavailable after the configured grace");
                            writer.record_gap(GapObservation {
                                kind: "contract_rotation",
                                code: "polymarket_current_contract_gap",
                                message: &message,
                                source_start: Some(current_window),
                                source_end: Some(current_window + TimeDelta::seconds(MARKET_INTERVAL_SECONDS)),
                                start_cursor: Some(format!("window_start:{}", current_window.timestamp())),
                                end_cursor: Some("current_contract_missing".to_owned()),
                            }).await?;
                            return Err(source_error("polymarket_current_contract_gap", message));
                        }
                    }
                    let samples = registry.samples(self.config.top_n);
                    // Capture the selection instant after copying immutable book
                    // values so received_at can never causally follow sampled_at.
                    let sampled_at = canonical_timestamp(Utc::now());
                    let facts = writer
                        .persist_samples(samples, sampled_at, connection_epoch)
                        .await?;
                    if !facts.is_empty() {
                        sampling_clock.mark_durable();
                    }
                    continuity.observe(&facts);
                }
                update = discovery_updates.recv() => {
                    let Some(update) = update else {
                        return Err(source_error(
                            "polymarket_gamma_worker_stopped",
                            "Polymarket Gamma discovery worker stopped unexpectedly",
                        ));
                    };
                    match update {
                        Ok(discovered_markets) => {
                            let desired_markets =
                                subscription_markets(&discovered_markets, Utc::now(), &self.config);
                            let delta = subscription_delta(&active_markets, &desired_markets);
                            if !delta.added.is_empty() {
                                send_websocket_message(
                                    &mut sink,
                                    Message::Text(clob_subscription_operation(&delta.added, true).into()),
                                    Duration::from_millis(self.config.connect_timeout_ms),
                                ).await?;
                            }
                            registry.install_market_set(&desired_markets)?;
                            if !delta.removed.is_empty() {
                                send_websocket_message(
                                    &mut sink,
                                    Message::Text(clob_subscription_operation(&delta.removed, false).into()),
                                    Duration::from_millis(self.config.connect_timeout_ms),
                                ).await?;
                            }
                            if !delta.added.is_empty() {
                                bootstrap_deadline = Some(
                                    Instant::now() + Duration::from_millis(self.config.bootstrap_timeout_ms)
                                );
                            }
                            active_markets = desired_markets;
                        }
                        Err(error) => {
                            warn!(
                                error_code = error.code,
                                error = %error,
                                "bounded Gamma refresh failed; retaining last verified subscription"
                            );
                        }
                    }
                }
                _ = ping.tick() => {
                    let sent_at = Instant::now();
                    send_websocket_message(
                        &mut sink,
                        Message::Text("PING".into()),
                        Duration::from_millis(self.config.connect_timeout_ms),
                    ).await?;
                    pong_deadline.get_or_insert(
                        sent_at + Duration::from_millis(self.config.pong_timeout_ms)
                    );
                }
                frame = stream.next() => {
                    read_deadline = Instant::now() + read_timeout;
                    let received_at = canonical_timestamp(Utc::now());
                    match frame {
                        Some(Ok(Message::Text(text))) => {
                            let value = text.as_str().trim();
                            if acknowledge_text_pong(value, &mut pong_deadline) {
                                continue;
                            }
                            if value.eq_ignore_ascii_case("PING") {
                                send_websocket_message(
                                    &mut sink,
                                    Message::Text("PONG".into()),
                                    Duration::from_millis(self.config.connect_timeout_ms),
                                ).await?;
                                continue;
                            }
                            if value.is_empty() {
                                continue;
                            }
                            self.apply_frame(
                                writer,
                                continuity,
                                &mut registry,
                                connection_epoch,
                                text.as_bytes(),
                                received_at,
                            ).await?;
                        }
                        Some(Ok(Message::Binary(bytes))) => {
                            self.apply_frame(
                                writer,
                                continuity,
                                &mut registry,
                                connection_epoch,
                                bytes.as_ref(),
                                received_at,
                            ).await?;
                        }
                        Some(Ok(Message::Ping(payload))) => {
                            send_websocket_message(
                                &mut sink,
                                Message::Pong(payload),
                                Duration::from_millis(self.config.connect_timeout_ms),
                            ).await?;
                        }
                        Some(Ok(Message::Pong(_))) => {
                            // The venue contract acknowledges text PING with text
                            // PONG. A protocol control PONG is not equivalent and
                            // cannot mask a missing venue acknowledgement.
                        }
                        Some(Ok(Message::Close(frame))) => {
                            return Err(source_error(
                                "polymarket_clob_closed",
                                format!("Polymarket CLOB websocket closed: {frame:?}"),
                            ));
                        }
                        Some(Ok(_)) => {}
                        Some(Err(error)) => {
                            return Err(source_error(
                                "polymarket_clob_read_failed",
                                format!("failed to read Polymarket CLOB websocket: {error}"),
                            ));
                        }
                        None => {
                            return Err(source_error(
                                "polymarket_clob_eof",
                                "Polymarket CLOB websocket ended",
                            ));
                        }
                    }
                    if registry.all_bootstrapped() {
                        bootstrap_deadline = None;
                    }
                }
            }
        }
    }
}

impl PolymarketBtcFiveMinuteOrderbooksStrategy {
    async fn apply_frame(
        &self,
        writer: &mut CaptureWriter,
        continuity: &Continuity,
        registry: &mut BookRegistry,
        connection_epoch: Uuid,
        bytes: &[u8],
        received_at: DateTime<Utc>,
    ) -> Result<(), StrategyError> {
        let messages = match parse_clob_frame(bytes, self.config.max_levels_per_side) {
            Ok(messages) => messages,
            Err(error) => {
                self.record_frame_gap(
                    writer,
                    continuity,
                    connection_epoch,
                    "message_decode",
                    &error,
                    None,
                )
                .await?;
                return Err(error);
            }
        };
        for message in messages {
            let source_timestamp = message_source_timestamp(&message);
            if let Err(error) =
                registry.apply(message, received_at, self.config.max_levels_per_side)
            {
                self.record_frame_gap(
                    writer,
                    continuity,
                    connection_epoch,
                    "book_integrity",
                    &error,
                    source_timestamp,
                )
                .await?;
                return Err(error);
            }
        }
        Ok(())
    }

    async fn record_frame_gap(
        &self,
        writer: &mut CaptureWriter,
        continuity: &Continuity,
        connection_epoch: Uuid,
        kind: &str,
        error: &StrategyError,
        source_end: Option<DateTime<Utc>>,
    ) -> Result<(), StrategyError> {
        writer
            .record_gap(GapObservation {
                kind,
                code: error.code,
                message: &error.message,
                source_start: continuity.last_source_timestamp,
                source_end,
                start_cursor: continuity
                    .last_cursor
                    .clone()
                    .or_else(|| Some(format!("connection_epoch:{connection_epoch}"))),
                end_cursor: source_end
                    .map(|timestamp| format!("source_timestamp:{}", timestamp.timestamp_micros())),
            })
            .await
    }
}

fn message_source_timestamp(message: &ClobMessage) -> Option<DateTime<Utc>> {
    match message {
        ClobMessage::Book {
            source_timestamp, ..
        }
        | ClobMessage::PriceChange {
            source_timestamp, ..
        }
        | ClobMessage::BestBidAsk {
            source_timestamp, ..
        }
        | ClobMessage::TickSizeChange {
            source_timestamp, ..
        }
        | ClobMessage::Auxiliary {
            source_timestamp, ..
        } => Some(*source_timestamp),
        ClobMessage::Control => None,
    }
}

fn next_sample_delay(now: DateTime<Utc>, interval_ms: u64) -> Duration {
    let interval_ms = i64::try_from(interval_ms).unwrap_or(i64::MAX);
    let remainder = now.timestamp_millis().rem_euclid(interval_ms);
    let delay = if remainder == 0 {
        interval_ms
    } else {
        interval_ms - remainder
    };
    Duration::from_millis(u64::try_from(delay).unwrap_or(1))
}

fn acknowledge_text_pong(text: &str, deadline: &mut Option<Instant>) -> bool {
    if text.trim().eq_ignore_ascii_case("PONG") {
        deadline.take();
        true
    } else {
        false
    }
}

async fn send_websocket_message<S>(
    sink: &mut S,
    message: Message,
    timeout: Duration,
) -> Result<(), StrategyError>
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::fmt::Display,
{
    tokio::time::timeout(timeout, sink.send(message))
        .await
        .map_err(|_| {
            source_error(
                "polymarket_clob_write_timeout",
                "timed out writing Polymarket CLOB websocket message",
            )
        })?
        .map_err(|error| {
            source_error(
                "polymarket_clob_write_failed",
                format!("failed to write Polymarket CLOB websocket message: {error}"),
            )
        })
}

fn gap_was_recorded(code: &str) -> bool {
    matches!(
        code,
        "polymarket_clob_bootstrap_timeout"
            | "polymarket_current_contract_gap"
            | "polymarket_sampling_bucket_gap"
            | "polymarket_clob_decode_failed"
            | "polymarket_clob_message_batch_too_large"
            | "polymarket_clob_invalid_message"
            | "polymarket_clob_invalid_book"
            | "polymarket_clob_invalid_level"
            | "polymarket_clob_book_too_large"
            | "polymarket_clob_invalid_price_change"
            | "polymarket_clob_invalid_side"
            | "polymarket_clob_invalid_timestamp"
            | "polymarket_clob_invalid_source_hash"
            | "polymarket_clob_unsupported_event"
            | "polymarket_clob_market_mismatch"
            | "polymarket_clob_token_mismatch"
            | "polymarket_clob_timestamp_regression"
            | "polymarket_clob_crossed_book"
            | "polymarket_clob_top_mismatch"
            | "polymarket_clob_invalid_advertised_top"
            | "polymarket_clob_tick_size_mismatch"
    )
}

fn is_current_profile_lease_loss(error: &StrategyError) -> bool {
    error.kind == StrategyErrorKind::LeaseLost && error.code == "polymarket_lease_lost"
}

fn source_error(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::TransientSource, code, message)
}

fn database_error(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::TransientDatabase, code, message)
}

fn integrity_error(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::Integrity, code, message)
}

fn lease_error(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::LeaseLost, code, message)
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use pretty_assertions::assert_eq;

    use super::*;

    const GAMMA_FIXTURE: &str =
        include_str!("../../../tests/fixtures/polymarket/gamma_btc_five_minute_event_v1.json");
    const BOOKS_FIXTURE: &str =
        include_str!("../../../tests/fixtures/polymarket/clob_books_v1.json");
    const CHANGE_FIXTURE: &str =
        include_str!("../../../tests/fixtures/polymarket/clob_price_change_v1.json");

    fn at(milliseconds: i64) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(milliseconds)
            .single()
            .expect("test timestamp")
    }

    fn fixture_market() -> MarketContract {
        let value = serde_json::from_str::<Value>(GAMMA_FIXTURE).expect("Gamma fixture JSON");
        parse_gamma_market(
            &value,
            Utc.timestamp_opt(1_783_902_600, 0)
                .single()
                .expect("window"),
            at(1_783_902_599_500),
        )
        .expect("Gamma fixture")
    }

    fn fixture_books() -> Vec<ClobMessage> {
        parse_clob_frame(BOOKS_FIXTURE.as_bytes(), 100).expect("book fixture")
    }

    fn bootstrapped_registry() -> BookRegistry {
        let market = fixture_market();
        let mut registry =
            BookRegistry::new(Uuid::new_v4(), std::slice::from_ref(&market)).expect("registry");
        for message in fixture_books() {
            assert_eq!(
                registry
                    .apply(message, at(1_783_902_600_130), 100)
                    .expect("full book"),
                ApplyOutcome::Applied
            );
        }
        registry
    }

    fn shifted_market(
        base: &MarketContract,
        offset: i64,
        digit: char,
        up: &str,
        down: &str,
    ) -> MarketContract {
        let window_start = base.window_start + TimeDelta::seconds(offset * 300);
        MarketContract {
            event_slug: event_slug(window_start),
            market_id: format!("market-{digit}"),
            condition_id: format!("0x{}", digit.to_string().repeat(64)),
            window_start,
            window_end: window_start + TimeDelta::seconds(300),
            up_token_id: up.to_owned(),
            down_token_id: down.to_owned(),
            tick_size: Decimal::new(1, 2),
            received_at: base.received_at,
        }
    }

    #[test]
    fn config_is_typed_public_and_provider_exact() {
        let config =
            PolymarketBtcFiveMinuteOrderbooksConfig::from_value(&json!({})).expect("defaults");
        assert_eq!(config.websocket_url, DEFAULT_WEBSOCKET_URL);
        assert_eq!(config.gamma_api_url, DEFAULT_GAMMA_API_URL);
        assert_eq!(config.ping_interval_ms, 10_000);
        assert_eq!(config.pong_timeout_ms, 25_000);
        assert_eq!(config.read_timeout_ms, 40_000);
        assert_eq!(config.sampling_policy()["selection"], SAMPLING_SELECTION);

        assert!(PolymarketBtcFiveMinuteOrderbooksConfig::from_value(&json!({
            "unknown": true
        }))
        .is_err());
        assert!(PolymarketBtcFiveMinuteOrderbooksConfig::from_value(&json!({
            "websocket_url": "wss://example.invalid/ws"
        }))
        .is_err());
        assert!(PolymarketBtcFiveMinuteOrderbooksConfig::from_value(&json!({
            "gamma_api_url": "https://example.invalid"
        }))
        .is_err());
        assert!(PolymarketBtcFiveMinuteOrderbooksConfig::from_value(&json!({
            "pong_timeout_ms": 10000
        }))
        .is_err());
        assert!(PolymarketBtcFiveMinuteOrderbooksConfig::from_value(&json!({
            "lookahead_windows": 2
        }))
        .is_err());
    }

    #[test]
    fn gamma_fixture_maps_outcomes_by_label_and_validates_identity() {
        let market = fixture_market();
        assert_eq!(market.event_slug, "btc-updown-5m-1783902600");
        assert_eq!(market.market_id, "market-sanitized-1");
        assert!(market.condition_id.starts_with("0x"));
        assert!(market.up_token_id.starts_with('1'));
        assert!(market.down_token_id.starts_with('2'));
        assert_eq!(market.tick_size, Decimal::new(1, 2));

        let value = serde_json::from_str::<Value>(GAMMA_FIXTURE).expect("fixture");
        assert!(parse_gamma_market(
            &value,
            market.window_start + TimeDelta::seconds(300),
            market.received_at
        )
        .is_err());

        let mut wrong_series = value.clone();
        wrong_series["seriesSlug"] = json!("eth-up-or-down-5m");
        assert!(
            parse_gamma_market(&wrong_series, market.window_start, market.received_at).is_err()
        );

        let mut bad_condition = value;
        bad_condition["markets"][0]["conditionId"] = json!("0xABC");
        assert!(
            parse_gamma_market(&bad_condition, market.window_start, market.received_at).is_err()
        );
    }

    #[test]
    fn sanitized_object_and_array_frames_parse_with_strict_bounds() {
        let books = fixture_books();
        assert_eq!(books.len(), 2);
        assert!(matches!(
            &books[0],
            ClobMessage::Book { bids, asks, .. } if bids.len() == 2 && asks.len() == 2
        ));
        assert!(matches!(
            &books[1],
            ClobMessage::Book { bids, asks, .. } if bids.is_empty() && asks.is_empty()
        ));

        let changes =
            parse_clob_frame(CHANGE_FIXTURE.as_bytes(), 100).expect("price-change fixture");
        assert!(matches!(
            &changes[0],
            ClobMessage::PriceChange { changes, .. }
                if changes.len() == 2
                    && changes[0].side == BookSide::Bid
                    && changes[1].size.is_zero()
        ));

        let duplicate = BOOKS_FIXTURE.replacen(
            r#"{"price": "0.47", "size": "12.5"}"#,
            r#"{"price": "0.48", "size": "12.5"}"#,
            1,
        );
        assert!(parse_clob_frame(duplicate.as_bytes(), 100).is_err());
        assert!(parse_clob_frame(BOOKS_FIXTURE.as_bytes(), 1).is_err());
        assert!(parse_clob_frame(&vec![b' '; MAX_WEBSOCKET_FRAME_BYTES + 1], 100).is_err());
    }

    #[test]
    fn full_snapshots_bootstrap_empty_and_shallow_books() {
        let registry = bootstrapped_registry();
        assert!(registry.all_bootstrapped());
        let samples = registry.samples(20);
        assert_eq!(samples.len(), 2);
        let up = samples
            .iter()
            .find(|sample| sample.outcome == Outcome::Up)
            .expect("Up sample");
        assert_eq!(
            up.bids,
            vec![
                ["0.48".to_owned(), "30".to_owned()],
                ["0.47".to_owned(), "12.5".to_owned()]
            ]
        );
        assert_eq!(
            up.asks,
            vec![
                ["0.52".to_owned(), "25".to_owned()],
                ["0.53".to_owned(), "11".to_owned()]
            ]
        );
        let down = samples
            .iter()
            .find(|sample| sample.outcome == Outcome::Down)
            .expect("Down sample");
        assert!(down.bids.is_empty());
        assert!(down.asks.is_empty());
    }

    #[test]
    fn deltas_wait_for_full_snapshot_and_reconnect_resets_every_level() {
        let market = fixture_market();
        let mut registry =
            BookRegistry::new(Uuid::new_v4(), std::slice::from_ref(&market)).expect("registry");
        let delta = parse_clob_frame(CHANGE_FIXTURE.as_bytes(), 100)
            .expect("fixture")
            .remove(0);
        assert_eq!(
            registry
                .apply(delta.clone(), at(1_783_902_601_130), 100)
                .expect("pre-snapshot delta"),
            ApplyOutcome::AwaitingSnapshot
        );
        assert!(registry.samples(20).is_empty());

        for message in fixture_books() {
            registry
                .apply(message, at(1_783_902_600_130), 100)
                .expect("snapshot");
        }
        assert!(!registry.samples(20).is_empty());
        let next_epoch = Uuid::new_v4();
        registry.reset_connection(next_epoch);
        assert_eq!(registry.connection_epoch, next_epoch);
        assert!(registry.samples(20).is_empty());
        assert_eq!(
            registry
                .apply(delta, at(1_783_902_601_130), 100)
                .expect("new-epoch delta"),
            ApplyOutcome::AwaitingSnapshot
        );
    }

    #[test]
    fn grouped_price_changes_are_atomic_across_tokens() {
        let mut registry = bootstrapped_registry();
        let market = fixture_market();
        let before = registry
            .books
            .get(&market.up_token_id)
            .expect("Up book")
            .clone();
        let message = ClobMessage::PriceChange {
            market_id: market.condition_id,
            source_timestamp: at(1_783_902_601_500),
            changes: vec![
                PriceChange {
                    token_id: market.up_token_id.clone(),
                    side: BookSide::Bid,
                    price: Decimal::new(49, 2),
                    size: Decimal::new(10, 0),
                    source_hash: Some("valid".to_owned()),
                    best_bid: Some(Decimal::new(49, 2)),
                    best_ask: Some(Decimal::new(52, 2)),
                },
                PriceChange {
                    token_id: market.down_token_id,
                    side: BookSide::Bid,
                    price: Decimal::new(11, 1),
                    size: Decimal::ONE,
                    source_hash: Some("invalid".to_owned()),
                    best_bid: None,
                    best_ask: None,
                },
            ],
        };
        assert!(registry.apply(message, at(1_783_902_601_510), 100).is_err());
        let after = registry.books.get(&market.up_token_id).expect("Up book");
        assert_eq!(after.bids, before.bids);
        assert_eq!(after.source_hash, before.source_hash);
        assert_eq!(after.ingest_sequence, before.ingest_sequence);
    }

    #[test]
    fn stale_timestamp_is_ignored_and_invalid_books_fail_closed_without_mutation() {
        let mut registry = bootstrapped_registry();
        let market = fixture_market();
        let before = registry
            .books
            .get(&market.up_token_id)
            .expect("Up book")
            .clone();

        let regressed = ClobMessage::Book {
            market_id: market.condition_id.clone(),
            token_id: market.up_token_id.clone(),
            bids: vec![],
            asks: vec![],
            source_timestamp: at(1_783_902_600_000),
            source_hash: None,
        };
        assert_eq!(
            registry
                .apply(regressed, at(1_783_902_601_000), 100)
                .expect("stale provider frame is ignored"),
            ApplyOutcome::NonMutating
        );
        let after_stale = registry.books.get(&market.up_token_id).expect("Up book");
        assert_eq!(after_stale.bids, before.bids);
        assert_eq!(after_stale.asks, before.asks);
        assert_eq!(after_stale.ingest_sequence, before.ingest_sequence);

        let advisory = ClobMessage::BestBidAsk {
            market_id: market.condition_id.clone(),
            token_id: market.up_token_id.clone(),
            best_bid: Some(Decimal::new(1, 1)),
            best_ask: Some(Decimal::new(9, 1)),
            source_timestamp: at(1_783_902_601_400),
        };
        assert_eq!(
            registry
                .apply(advisory, at(1_783_902_601_410), 100)
                .expect("unsequenced advertised top is advisory"),
            ApplyOutcome::NonMutating
        );

        let crossed = ClobMessage::Book {
            market_id: market.condition_id.clone(),
            token_id: market.up_token_id.clone(),
            bids: vec![PriceLevel {
                price: Decimal::new(6, 1),
                size: Decimal::ONE,
            }],
            asks: vec![PriceLevel {
                price: Decimal::new(5, 1),
                size: Decimal::ONE,
            }],
            source_timestamp: at(1_783_902_601_500),
            source_hash: None,
        };
        let error = registry
            .apply(crossed, at(1_783_902_601_510), 100)
            .expect_err("crossed");
        assert_eq!(error.code, "polymarket_clob_crossed_book");

        let mismatched = ClobMessage::PriceChange {
            market_id: market.condition_id,
            source_timestamp: at(1_783_902_601_600),
            changes: vec![PriceChange {
                token_id: market.up_token_id.clone(),
                side: BookSide::Bid,
                price: Decimal::new(47, 2),
                size: Decimal::ZERO,
                source_hash: None,
                best_bid: Some(Decimal::new(49, 2)),
                best_ask: Some(Decimal::new(52, 2)),
            }],
        };
        let error = registry
            .apply(mismatched, at(1_783_902_601_610), 100)
            .expect_err("top mismatch");
        assert_eq!(error.code, "polymarket_clob_top_mismatch");
        let after = registry.books.get(&market.up_token_id).expect("Up book");
        assert_eq!(after.bids, before.bids);
        assert_eq!(after.asks, before.asks);
    }

    #[test]
    fn tick_change_clears_hash_when_it_advances_causal_timestamp() {
        let mut registry = bootstrapped_registry();
        let market = fixture_market();
        assert!(registry
            .books
            .get(&market.up_token_id)
            .expect("Up book")
            .source_hash
            .is_some());
        registry
            .apply(
                ClobMessage::TickSizeChange {
                    market_id: market.condition_id,
                    token_id: market.up_token_id.clone(),
                    old_tick_size: Decimal::new(1, 2),
                    new_tick_size: Decimal::new(1, 3),
                    source_timestamp: at(1_783_902_601_700),
                },
                at(1_783_902_601_710),
                100,
            )
            .expect("tick change");
        let book = registry.books.get(&market.up_token_id).expect("Up book");
        assert_eq!(book.tick_size, Decimal::new(1, 3));
        assert_eq!(book.source_timestamp, Some(at(1_783_902_601_700)));
        assert!(book.source_hash.is_none());
    }

    #[test]
    fn subscription_payload_and_rotation_delta_use_assets_ids() {
        let current = fixture_market();
        let previous = shifted_market(&current, -1, '2', "3", "4");
        let successor = shifted_market(&current, 1, '3', "5", "6");
        let markets = vec![previous.clone(), current.clone(), successor.clone()];
        validate_market_set(&markets).expect("contract set");

        let payload =
            serde_json::from_str::<Value>(&clob_subscription(&markets)).expect("subscription JSON");
        assert_eq!(payload["type"], "market");
        assert_eq!(payload["initial_dump"], true);
        assert_eq!(
            payload["assets_ids"],
            json!([
                "111111111111111111111111111111111111111111111111111111111111111111",
                "222222222222222222222222222222222222222222222222222222222222222222",
                "3",
                "4",
                "5",
                "6"
            ])
        );
        assert!(payload.get("asset_ids").is_none());

        let desired = vec![current, successor];
        let delta = subscription_delta(&markets, &desired);
        assert_eq!(delta.added, Vec::<String>::new());
        assert_eq!(delta.removed, vec!["3".to_owned(), "4".to_owned()]);
        let operation =
            serde_json::from_str::<Value>(&clob_subscription_operation(&delta.removed, false))
                .expect("operation");
        assert_eq!(operation["operation"], "unsubscribe");
        assert_eq!(operation["assets_ids"], json!(["3", "4"]));
    }

    #[test]
    fn discovery_windows_are_exact_previous_current_successor() {
        let config = PolymarketBtcFiveMinuteOrderbooksConfig::default();
        let inside = at(1_783_902_731_999);
        let windows = discovery_windows(inside, &config);
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[1], aligned_market_window(inside));
        assert_eq!(windows[1] - windows[0], TimeDelta::seconds(300));
        assert_eq!(windows[2] - windows[1], TimeDelta::seconds(300));
    }

    #[test]
    fn successor_lead_and_previous_grace_bound_live_subscriptions() {
        let config = PolymarketBtcFiveMinuteOrderbooksConfig::default();
        let current = fixture_market();
        let previous = shifted_market(&current, -1, '2', "3", "4");
        let successor = shifted_market(&current, 1, '3', "5", "6");
        let discovered = vec![previous, current.clone(), successor.clone()];

        let opening = current.window_start + TimeDelta::seconds(1);
        let opening_markets = subscription_markets(&discovered, opening, &config);
        assert_eq!(opening_markets.len(), 2);
        assert!(opening_markets
            .iter()
            .any(|market| market.window_start == current.window_start - TimeDelta::seconds(300)));
        assert!(!opening_markets
            .iter()
            .any(|market| market.window_start == successor.window_start));

        let middle = current.window_start + TimeDelta::seconds(60);
        assert_eq!(
            subscription_markets(&discovered, middle, &config),
            vec![current.clone()]
        );

        let handoff = current.window_end - TimeDelta::seconds(29);
        let handoff_markets = subscription_markets(&discovered, handoff, &config);
        assert_eq!(handoff_markets, vec![current, successor]);
    }

    #[test]
    fn sample_hashes_are_stable_across_receipt_and_connection_metadata() {
        let registry = bootstrapped_registry();
        let sample = registry
            .samples(1)
            .into_iter()
            .find(|sample| sample.outcome == Outcome::Up)
            .expect("Up sample");
        let sampled_at = at(1_783_902_602_000);
        let policy = PolymarketBtcFiveMinuteOrderbooksConfig::default().sampling_policy();
        let policy_hash = hash_json(&policy).expect("policy hash");
        let first = SnapshotFact::new(
            sample.clone(),
            sampled_at,
            Uuid::new_v4(),
            &policy,
            &policy_hash,
        )
        .expect("first");
        let mut later_receipt = sample;
        later_receipt.received_at += TimeDelta::milliseconds(100);
        let second = SnapshotFact::new(
            later_receipt,
            sampled_at,
            Uuid::new_v4(),
            &policy,
            &policy_hash,
        )
        .expect("second");
        assert_ne!(first.received_at, second.received_at);
        assert_ne!(first.connection_epoch, second.connection_epoch);
        assert_eq!(first.book_sha256, second.book_sha256);
        assert_eq!(first.sampling_policy_sha256, second.sampling_policy_sha256);
        assert_eq!(first.payload_sha256, second.payload_sha256);

        let later_sample = SnapshotFact::new(
            registry
                .samples(1)
                .into_iter()
                .find(|sample| sample.outcome == Outcome::Up)
                .expect("Up sample"),
            sampled_at + TimeDelta::seconds(1),
            Uuid::new_v4(),
            &policy,
            &policy_hash,
        )
        .expect("later");
        assert_eq!(first.book_sha256, later_sample.book_sha256);
        assert_ne!(first.payload_sha256, later_sample.payload_sha256);
    }

    #[test]
    fn sample_causality_rejects_selection_before_receipt() {
        let registry = bootstrapped_registry();
        let sample = registry.samples(1).remove(0);
        let policy = PolymarketBtcFiveMinuteOrderbooksConfig::default().sampling_policy();
        let policy_hash = hash_json(&policy).expect("policy hash");
        let sampled_at = sample.received_at - TimeDelta::microseconds(1);
        let error = SnapshotFact::new(sample, sampled_at, Uuid::new_v4(), &policy, &policy_hash)
            .expect_err("causality violation");
        assert_eq!(error.code, "polymarket_sample_causality_violation");
    }

    #[test]
    fn sampling_clock_skips_repeated_and_missed_wall_slots() {
        let start = at(1_783_902_600_001);
        let mut clock = SamplingClock::default();
        assert_eq!(
            clock.claim(start, 1_000),
            SamplingClaim {
                should_sample: true,
                missed_buckets: None
            }
        );
        assert!(
            !clock
                .claim(start + TimeDelta::milliseconds(998), 1_000)
                .should_sample
        );
        assert!(
            clock
                .claim(start + TimeDelta::milliseconds(999), 1_000)
                .should_sample
        );
        assert_eq!(
            clock
                .claim(start + TimeDelta::milliseconds(5_999), 1_000)
                .missed_buckets,
            None,
            "no cadence gap is asserted before a durable sample"
        );
        assert!(
            !clock
                .claim(start + TimeDelta::milliseconds(6_500), 1_000)
                .should_sample
        );
        assert_eq!(
            next_sample_delay(at(1_783_902_600_250), 1_000),
            Duration::from_millis(750)
        );
    }

    #[test]
    fn sampling_clock_reports_bucket_jump_after_durable_sample() {
        let start = at(1_783_902_600_001);
        let mut clock = SamplingClock::default();
        assert!(clock.claim(start, 1_000).should_sample);
        clock.mark_durable();
        let previous = start.timestamp_millis().div_euclid(1_000);
        assert_eq!(
            clock
                .claim(start + TimeDelta::milliseconds(3_000), 1_000)
                .missed_buckets,
            Some((previous + 1, previous + 2))
        );
    }

    #[test]
    fn only_text_pong_acknowledges_the_venue_text_ping() {
        let first_deadline = Instant::now() + Duration::from_secs(25);
        let mut deadline = Some(first_deadline);
        assert!(!acknowledge_text_pong("not-pong", &mut deadline));
        assert_eq!(deadline, Some(first_deadline));
        assert!(acknowledge_text_pong(" PONG ", &mut deadline));
        assert_eq!(deadline, None);
    }

    #[test]
    fn canonical_hash_and_artifact_fences_are_deterministic() {
        let left = json!({"b": 2, "a": {"d": 4, "c": 3}});
        let right = json!({"a": {"c": 3, "d": 4}, "b": 2});
        assert_eq!(
            hash_json(&left).expect("left"),
            hash_json(&right).expect("right")
        );

        let mut first = Sha256::new();
        append_artifact_hash(&mut first, &"a".repeat(64));
        append_artifact_hash(&mut first, &"b".repeat(64));
        let mut second = Sha256::new();
        append_artifact_hash(&mut second, &"a".repeat(64));
        append_artifact_hash(&mut second, &"b".repeat(64));
        assert_eq!(
            encode_digest(first.finalize()),
            encode_digest(second.finalize())
        );

        assert!(!fence_open_artifact_generation(2, 2).expect("same generation"));
        assert!(fence_open_artifact_generation(1, 2).expect("new owner"));
        let error = fence_open_artifact_generation(3, 2).expect_err("stale owner must be fenced");
        assert_eq!(error.kind, StrategyErrorKind::LeaseLost);
    }
}
