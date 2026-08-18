//! Continuous Binance Spot BTCUSDT diff-depth capture.
//!
//! This strategy reconstructs a sequence-verified order book in memory and
//! persists only sampled, raw top-of-book snapshots. It intentionally contains
//! no feature engineering, labels, or model-availability assumptions.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt::Write as _,
    str::FromStr,
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
use sqlx::{FromRow, PgPool, Postgres, Transaction};
use tokio::{
    sync::mpsc,
    task::JoinHandle,
    time::{Instant, MissedTickBehavior},
};
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

pub const STRATEGY_KEY: IngesterStrategyKey = IngesterStrategyKey::BinanceSpotBtcusdtL2Snapshots;
pub const CONFIG_SCHEMA_VERSION: i32 = 1;
pub const CHECKPOINT_SCHEMA_VERSION: i32 = 1;
pub const SYMBOL: &str = "BTCUSDT";

const DEFAULT_WEBSOCKET_URL: &str = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms";
const ALTERNATE_WEBSOCKET_URL: &str = "wss://stream.binance.com:443/ws/btcusdt@depth@100ms";
const DEFAULT_REST_DEPTH_URL: &str = "https://api.binance.com/api/v3/depth";
const ALTERNATE_REST_DEPTH_URL: &str = "https://data-api.binance.vision/api/v3/depth";
const SAMPLING_POLICY_VERSION: &str = "binance-spot-btcusdt-l2-top-n-v1";
const MAX_UPDATE_LEVELS: usize = 10_000;
const MAX_UPDATE_FRAME_BYTES: usize = 2 * 1024 * 1024;
const MAX_SNAPSHOT_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

enum BinanceIoEvent {
    Update(BufferedUpdate),
    Failed(StrategyError),
}

struct BinanceIoWorker {
    shutdown: CancellationToken,
    handle: JoinHandle<()>,
}

impl Drop for BinanceIoWorker {
    fn drop(&mut self) {
        self.shutdown.cancel();
        self.handle.abort();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct BinanceSpotL2SnapshotConfig {
    pub symbol: String,
    pub websocket_url: String,
    pub rest_depth_url: String,
    pub rest_depth_limit: usize,
    pub top_n: usize,
    pub sample_interval_ms: u64,
    pub max_buffered_updates: usize,
    pub max_buffered_levels: usize,
    pub max_book_levels_per_side: usize,
    pub connect_timeout_ms: u64,
    pub bootstrap_timeout_ms: u64,
    pub read_timeout_ms: u64,
    pub ping_interval_ms: u64,
    pub reconnect_initial_ms: u64,
    pub reconnect_max_ms: u64,
    pub artifact_window_seconds: i64,
}

impl Default for BinanceSpotL2SnapshotConfig {
    fn default() -> Self {
        Self {
            symbol: SYMBOL.to_owned(),
            websocket_url: DEFAULT_WEBSOCKET_URL.to_owned(),
            rest_depth_url: DEFAULT_REST_DEPTH_URL.to_owned(),
            rest_depth_limit: 5_000,
            top_n: 20,
            sample_interval_ms: 1_000,
            max_buffered_updates: 4_096,
            max_buffered_levels: 200_000,
            max_book_levels_per_side: 100_000,
            connect_timeout_ms: 10_000,
            bootstrap_timeout_ms: 10_000,
            read_timeout_ms: 45_000,
            ping_interval_ms: 15_000,
            reconnect_initial_ms: 250,
            reconnect_max_ms: 30_000,
            artifact_window_seconds: 3_600,
        }
    }
}

impl BinanceSpotL2SnapshotConfig {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let config = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::InvalidConfiguration(format!(
                "invalid Binance spot L2 snapshot config: {error}"
            ))
        })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), StrategyFactoryError> {
        let invalid =
            |message: &str| StrategyFactoryError::InvalidConfiguration(message.to_owned());
        if self.symbol != SYMBOL {
            return Err(invalid("symbol must be BTCUSDT spot"));
        }
        if !matches!(
            self.websocket_url.as_str(),
            DEFAULT_WEBSOCKET_URL | ALTERNATE_WEBSOCKET_URL
        ) {
            return Err(invalid(
                "websocket_url must be an approved Binance BTCUSDT spot diff-depth endpoint",
            ));
        }
        if !matches!(
            self.rest_depth_url.as_str(),
            DEFAULT_REST_DEPTH_URL | ALTERNATE_REST_DEPTH_URL
        ) {
            return Err(invalid(
                "rest_depth_url must be an approved Binance spot depth endpoint",
            ));
        }
        if !(100..=5_000).contains(&self.rest_depth_limit) {
            return Err(invalid("rest_depth_limit must be between 100 and 5000"));
        }
        if self.top_n == 0 || self.top_n > 1_000 || self.top_n > self.rest_depth_limit {
            return Err(invalid(
                "top_n must be between 1 and 1000 and no larger than rest_depth_limit",
            ));
        }
        if !(100..=60_000).contains(&self.sample_interval_ms) {
            return Err(invalid("sample_interval_ms must be between 100 and 60000"));
        }
        if !(16..=65_536).contains(&self.max_buffered_updates) {
            return Err(invalid("max_buffered_updates must be between 16 and 65536"));
        }
        if self.max_buffered_levels < self.max_buffered_updates
            || self.max_buffered_levels > 1_000_000
        {
            return Err(invalid(
                "max_buffered_levels must be at least max_buffered_updates and at most 1000000",
            ));
        }
        if self.max_book_levels_per_side < self.rest_depth_limit
            || self.max_book_levels_per_side > 250_000
        {
            return Err(invalid(
                "max_book_levels_per_side must cover rest_depth_limit and be at most 250000",
            ));
        }
        if !(1_000..=30_000).contains(&self.connect_timeout_ms)
            || !(1_000..=30_000).contains(&self.bootstrap_timeout_ms)
        {
            return Err(invalid(
                "connect_timeout_ms and bootstrap_timeout_ms must be between 1000 and 30000",
            ));
        }
        if !(5_000..=120_000).contains(&self.read_timeout_ms)
            || self.ping_interval_ms < 1_000
            || self.ping_interval_ms >= self.read_timeout_ms
        {
            return Err(invalid(
                "ping_interval_ms must be at least 1000 and less than read_timeout_ms; read_timeout_ms must be at most 120000",
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
            "source": "binance_spot_diff_depth",
            "symbol": SYMBOL,
            "sample_interval_ms": self.sample_interval_ms,
            "sample_depth": self.top_n,
            "selection": "latest_contiguous_update_at_or_after_interval"
        })
    }
}

#[derive(Debug, Default)]
pub struct BinanceSpotL2SnapshotFactory;

impl StrategyFactory for BinanceSpotL2SnapshotFactory {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    fn config_schema_version(&self) -> i32 {
        CONFIG_SCHEMA_VERSION
    }

    fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError> {
        BinanceSpotL2SnapshotConfig::from_value(config).map(|_| ())
    }

    fn build(
        &self,
        profile: &IngesterProfile,
        pool: PgPool,
    ) -> Result<Box<dyn IngesterStrategy>, StrategyFactoryError> {
        let config = BinanceSpotL2SnapshotConfig::from_value(&profile.config)?;
        let lease_owner = profile.lease_owner.clone().ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease owner is missing".to_owned())
        })?;
        let lease_token = profile.lease_token.ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease token is missing".to_owned())
        })?;
        let client = Client::builder()
            .connect_timeout(Duration::from_millis(config.connect_timeout_ms))
            .timeout(Duration::from_millis(config.bootstrap_timeout_ms))
            .user_agent("capitonic-market-data-ingester/0.1")
            .build()
            .map_err(|error| {
                StrategyFactoryError::Construction(format!(
                    "failed to build Binance HTTP client: {error}"
                ))
            })?;
        Ok(Box::new(BinanceSpotL2SnapshotStrategy {
            config,
            config_snapshot: profile.config.clone(),
            profile_generation: profile.desired_generation,
            lease_owner: Arc::<str>::from(lease_owner),
            lease_token,
            pool,
            client,
        }))
    }
}

pub struct BinanceSpotL2SnapshotStrategy {
    config: BinanceSpotL2SnapshotConfig,
    config_snapshot: Value,
    profile_generation: i64,
    lease_owner: Arc<str>,
    lease_token: Uuid,
    pool: PgPool,
    client: Client,
}

#[derive(Debug, Clone, PartialEq)]
struct PriceLevel {
    price: Decimal,
    quantity: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
struct DepthUpdate {
    source_timestamp: DateTime<Utc>,
    first_update_id: u64,
    final_update_id: u64,
    bids: Vec<PriceLevel>,
    asks: Vec<PriceLevel>,
}

impl DepthUpdate {
    fn level_count(&self) -> usize {
        self.bids.len().saturating_add(self.asks.len())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct BufferedUpdate {
    update: DepthUpdate,
    received_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq)]
struct DepthSnapshot {
    last_update_id: u64,
    bids: Vec<PriceLevel>,
    asks: Vec<PriceLevel>,
}

#[derive(Debug, Deserialize)]
struct WireDepthUpdate {
    #[serde(rename = "e")]
    event_type: String,
    #[serde(rename = "E")]
    event_time_ms: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "U")]
    first_update_id: u64,
    #[serde(rename = "u")]
    final_update_id: u64,
    #[serde(rename = "b")]
    bids: Vec<[String; 2]>,
    #[serde(rename = "a")]
    asks: Vec<[String; 2]>,
}

#[derive(Debug, Deserialize)]
struct WireDepthSnapshot {
    #[serde(rename = "lastUpdateId")]
    last_update_id: u64,
    bids: Vec<[String; 2]>,
    asks: Vec<[String; 2]>,
}

fn parse_depth_update(text: &str) -> Result<DepthUpdate, StrategyError> {
    if text.len() > MAX_UPDATE_FRAME_BYTES {
        return Err(source_error(
            "binance_l2_update_frame_too_large",
            "Binance depth update frame exceeded the bounded payload size",
        ));
    }
    let wire = serde_json::from_str::<WireDepthUpdate>(text).map_err(|error| {
        source_error(
            "binance_l2_invalid_update",
            format!("failed to decode Binance spot depth update: {error}"),
        )
    })?;
    if wire.event_type != "depthUpdate" || wire.symbol != SYMBOL {
        return Err(source_error(
            "binance_l2_unexpected_stream",
            "Binance depth frame was not a BTCUSDT spot depthUpdate",
        ));
    }
    if wire.first_update_id > wire.final_update_id || wire.final_update_id > i64::MAX as u64 {
        return Err(source_error(
            "binance_l2_invalid_sequence",
            "Binance depth update contained an invalid update-id range",
        ));
    }
    let source_timestamp = Utc
        .timestamp_millis_opt(wire.event_time_ms)
        .single()
        .ok_or_else(|| {
            source_error(
                "binance_l2_invalid_timestamp",
                "Binance depth update contained an invalid event timestamp",
            )
        })?;
    let bids = parse_levels(wire.bids, true, "update bids")?;
    let asks = parse_levels(wire.asks, true, "update asks")?;
    if bids.is_empty() && asks.is_empty() {
        return Err(source_error(
            "binance_l2_empty_update",
            "Binance depth update contained no levels",
        ));
    }
    if bids.len().saturating_add(asks.len()) > MAX_UPDATE_LEVELS {
        return Err(source_error(
            "binance_l2_update_too_large",
            "Binance depth update exceeded the bounded level count",
        ));
    }
    Ok(DepthUpdate {
        source_timestamp,
        first_update_id: wire.first_update_id,
        final_update_id: wire.final_update_id,
        bids,
        asks,
    })
}

fn parse_depth_snapshot(wire: WireDepthSnapshot) -> Result<DepthSnapshot, StrategyError> {
    if wire.last_update_id > i64::MAX as u64 {
        return Err(source_error(
            "binance_l2_invalid_snapshot_sequence",
            "Binance depth snapshot update id exceeded database range",
        ));
    }
    let bids = parse_levels(wire.bids, false, "snapshot bids")?;
    let asks = parse_levels(wire.asks, false, "snapshot asks")?;
    if bids.is_empty() || asks.is_empty() || bids.len() > 5_000 || asks.len() > 5_000 {
        return Err(source_error(
            "binance_l2_invalid_snapshot_depth",
            "Binance depth snapshot must contain between 1 and 5000 levels per side",
        ));
    }
    Ok(DepthSnapshot {
        last_update_id: wire.last_update_id,
        bids,
        asks,
    })
}

fn parse_levels(
    rows: Vec<[String; 2]>,
    allow_zero: bool,
    name: &str,
) -> Result<Vec<PriceLevel>, StrategyError> {
    let mut levels = Vec::with_capacity(rows.len());
    let mut prices = BTreeSet::new();
    for [raw_price, raw_quantity] in rows {
        if raw_price.len() > 64 || raw_quantity.len() > 64 {
            return Err(source_error(
                "binance_l2_invalid_level",
                format!("Binance {name} contained an oversized numeric value"),
            ));
        }
        let price = Decimal::from_str(&raw_price).map_err(|error| {
            source_error(
                "binance_l2_invalid_level",
                format!("Binance {name} contained invalid price {raw_price}: {error}"),
            )
        })?;
        let quantity = Decimal::from_str(&raw_quantity).map_err(|error| {
            source_error(
                "binance_l2_invalid_level",
                format!("Binance {name} contained invalid quantity {raw_quantity}: {error}"),
            )
        })?;
        if price <= Decimal::ZERO
            || quantity < Decimal::ZERO
            || (!allow_zero && quantity.is_zero())
            || !prices.insert(price)
        {
            return Err(source_error(
                "binance_l2_invalid_level",
                format!("Binance {name} contained an invalid or repeated level"),
            ));
        }
        levels.push(PriceLevel { price, quantity });
    }
    Ok(levels)
}

#[derive(Debug, Default)]
struct OrderBook {
    bids: BTreeMap<Decimal, Decimal>,
    asks: BTreeMap<Decimal, Decimal>,
    last_update_id: Option<u64>,
    synchronized: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApplyOutcome {
    Applied {
        synchronized_now: bool,
    },
    Stale,
    Gap {
        expected: u64,
        first: u64,
        final_id: u64,
    },
}

impl OrderBook {
    fn install_snapshot(
        &mut self,
        snapshot: DepthSnapshot,
        max_levels_per_side: usize,
    ) -> Result<(), StrategyError> {
        self.bids.clear();
        self.asks.clear();
        for level in snapshot.bids {
            self.bids.insert(level.price, level.quantity);
        }
        for level in snapshot.asks {
            self.asks.insert(level.price, level.quantity);
        }
        self.last_update_id = Some(snapshot.last_update_id);
        self.synchronized = false;
        self.validate(max_levels_per_side)
    }

    fn apply(
        &mut self,
        update: &DepthUpdate,
        max_levels_per_side: usize,
    ) -> Result<ApplyOutcome, StrategyError> {
        let current = self.last_update_id.ok_or_else(|| {
            integrity_error(
                "binance_l2_snapshot_missing",
                "depth update cannot be applied before a REST snapshot",
            )
        })?;
        if update.final_update_id <= current {
            return Ok(ApplyOutcome::Stale);
        }
        let expected = current.checked_add(1).ok_or_else(|| {
            integrity_error(
                "binance_l2_sequence_overflow",
                "Binance depth update sequence overflowed",
            )
        })?;
        let contiguous = if self.synchronized {
            update.first_update_id == expected
        } else {
            update.first_update_id <= expected && update.final_update_id >= expected
        };
        if !contiguous {
            return Ok(ApplyOutcome::Gap {
                expected,
                first: update.first_update_id,
                final_id: update.final_update_id,
            });
        }

        apply_levels(&mut self.bids, &update.bids);
        apply_levels(&mut self.asks, &update.asks);
        self.validate(max_levels_per_side)?;
        let synchronized_now = !self.synchronized;
        self.synchronized = true;
        self.last_update_id = Some(update.final_update_id);
        Ok(ApplyOutcome::Applied { synchronized_now })
    }

    fn sample(&self, depth: usize) -> Result<BookSample, StrategyError> {
        if !self.synchronized {
            return Err(integrity_error(
                "binance_l2_not_synchronized",
                "cannot sample an unsynchronized Binance order book",
            ));
        }
        let bids = self
            .bids
            .iter()
            .rev()
            .take(depth)
            .map(|(price, quantity)| [price.to_string(), quantity.to_string()])
            .collect::<Vec<_>>();
        let asks = self
            .asks
            .iter()
            .take(depth)
            .map(|(price, quantity)| [price.to_string(), quantity.to_string()])
            .collect::<Vec<_>>();
        if bids.len() != depth || asks.len() != depth {
            return Err(source_error(
                "binance_l2_insufficient_depth",
                "synchronized Binance book did not contain the configured sample depth",
            ));
        }
        Ok(BookSample { bids, asks })
    }

    fn validate(&self, max_levels_per_side: usize) -> Result<(), StrategyError> {
        if self.bids.len() > max_levels_per_side || self.asks.len() > max_levels_per_side {
            return Err(source_error(
                "binance_l2_book_too_large",
                "Binance order book exceeded its bounded in-memory level count",
            ));
        }
        let best_bid = self.bids.last_key_value().map(|(price, _)| *price);
        let best_ask = self.asks.first_key_value().map(|(price, _)| *price);
        if best_bid.is_none() || best_ask.is_none() || best_bid >= best_ask {
            return Err(source_error(
                "binance_l2_crossed_book",
                "Binance order book was empty or crossed",
            ));
        }
        Ok(())
    }
}

fn apply_levels(book: &mut BTreeMap<Decimal, Decimal>, levels: &[PriceLevel]) {
    for level in levels {
        if level.quantity.is_zero() {
            book.remove(&level.price);
        } else {
            book.insert(level.price, level.quantity);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct BookSample {
    bids: Vec<[String; 2]>,
    asks: Vec<[String; 2]>,
}

#[derive(Debug, Default)]
struct SamplingClock {
    next_sample_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SamplingClaim {
    should_sample: bool,
    missed_slots: Option<(DateTime<Utc>, DateTime<Utc>)>,
}

impl SamplingClock {
    fn claim(&mut self, received_at: DateTime<Utc>, interval_ms: u64) -> SamplingClaim {
        let Ok(interval_ms) = i64::try_from(interval_ms) else {
            return SamplingClaim {
                should_sample: false,
                missed_slots: None,
            };
        };
        let interval = chrono::Duration::milliseconds(interval_ms);
        let Some(mut next) = self.next_sample_at else {
            self.next_sample_at = received_at.checked_add_signed(interval);
            return SamplingClaim {
                should_sample: true,
                missed_slots: None,
            };
        };
        if received_at < next {
            return SamplingClaim {
                should_sample: false,
                missed_slots: None,
            };
        }
        let first_missed = next;
        let mut last_missed = None;
        while next <= received_at {
            let Some(advanced) = next.checked_add_signed(interval) else {
                return SamplingClaim {
                    should_sample: false,
                    missed_slots: None,
                };
            };
            if advanced <= received_at {
                last_missed = Some(next);
            }
            next = advanced;
        }
        self.next_sample_at = Some(next);
        SamplingClaim {
            should_sample: true,
            missed_slots: last_missed.map(|last| (first_missed, last)),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Continuity {
    last_update_id: Option<u64>,
    last_source_timestamp: Option<DateTime<Utc>>,
    last_received_at: Option<DateTime<Utc>>,
}

#[derive(Debug)]
struct DeltaBuffer {
    updates: VecDeque<BufferedUpdate>,
    level_count: usize,
    max_updates: usize,
    max_levels: usize,
}

impl DeltaBuffer {
    fn new(max_updates: usize, max_levels: usize) -> Self {
        Self {
            updates: VecDeque::with_capacity(max_updates.min(4_096)),
            level_count: 0,
            max_updates,
            max_levels,
        }
    }

    fn push(&mut self, update: BufferedUpdate) -> Result<(), StrategyError> {
        let new_level_count = self
            .level_count
            .checked_add(update.update.level_count())
            .ok_or_else(|| {
                source_error(
                    "binance_l2_buffer_overflow",
                    "Binance bootstrap delta-buffer level count overflowed",
                )
            })?;
        if self.updates.len() >= self.max_updates || new_level_count > self.max_levels {
            return Err(source_error(
                "binance_l2_buffer_overflow",
                "Binance bootstrap delta buffer exceeded its configured bound",
            ));
        }
        self.level_count = new_level_count;
        self.updates.push_back(update);
        Ok(())
    }
}

struct CaptureWriter {
    pool: PgPool,
    artifacts: ArtifactRepository,
    gaps: GapRepository,
    profiles: ProfileRepository,
    config: BinanceSpotL2SnapshotConfig,
    config_snapshot: Value,
    profile_generation: i64,
    lease_owner: Arc<str>,
    lease_token: Uuid,
    sampling_policy: Value,
    sampling_policy_sha256: String,
    current: Option<OpenArtifact>,
}

#[derive(Debug)]
struct OpenArtifact {
    artifact: CaptureArtifact,
    content_hasher: Sha256,
    inserted_records: i64,
    end_cursor: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArtifactSealFence {
    CurrentProfile,
    OwnedDrain,
}

#[derive(Debug, Clone)]
struct SnapshotFact {
    source_timestamp: DateTime<Utc>,
    received_at: DateTime<Utc>,
    source_update_id: i64,
    connection_epoch: Uuid,
    sample_depth: i32,
    bids: Value,
    asks: Value,
    book_sha256: String,
    sampling_policy: Value,
    sampling_policy_sha256: String,
    payload_sha256: String,
}

impl SnapshotFact {
    fn new(
        update: &DepthUpdate,
        received_at: DateTime<Utc>,
        connection_epoch: Uuid,
        sample: &BookSample,
        sampling_policy: &Value,
        sampling_policy_sha256: &str,
    ) -> Result<Self, StrategyError> {
        let source_update_id = i64::try_from(update.final_update_id).map_err(|_| {
            integrity_error(
                "binance_l2_update_id_overflow",
                "Binance update id exceeded database range",
            )
        })?;
        let sample_depth = i32::try_from(sample.bids.len()).map_err(|_| {
            integrity_error(
                "binance_l2_sample_depth_overflow",
                "Binance sample depth exceeded database range",
            )
        })?;
        let bids = serde_json::to_value(&sample.bids).map_err(|error| {
            integrity_error(
                "binance_l2_sample_serialization",
                format!("failed to serialize Binance bid sample: {error}"),
            )
        })?;
        let asks = serde_json::to_value(&sample.asks).map_err(|error| {
            integrity_error(
                "binance_l2_sample_serialization",
                format!("failed to serialize Binance ask sample: {error}"),
            )
        })?;
        let book_sha256 = hash_json(&json!({"bids": bids, "asks": asks}))?;
        let payload_sha256 = hash_json(&json!({
            "schema_version": 1,
            "source_timestamp": update.source_timestamp,
            "symbol": SYMBOL,
            "source_update_id": source_update_id,
            "sample_depth": sample_depth,
            "bids": bids,
            "asks": asks,
            "book_sha256": book_sha256,
            "sampling_policy_sha256": sampling_policy_sha256
        }))?;
        Ok(Self {
            source_timestamp: update.source_timestamp,
            received_at,
            source_update_id,
            connection_epoch,
            sample_depth,
            bids,
            asks,
            book_sha256,
            sampling_policy: sampling_policy.clone(),
            sampling_policy_sha256: sampling_policy_sha256.to_owned(),
            payload_sha256,
        })
    }

    fn cursor(&self) -> String {
        format!("update_id:{}", self.source_update_id)
    }
}

#[derive(Debug, FromRow)]
struct ExistingSnapshotFact {
    source_timestamp: DateTime<Utc>,
    sample_depth: i32,
    book_sha256: String,
    sampling_policy_sha256: String,
    payload_sha256: String,
}

impl ExistingSnapshotFact {
    fn matches(&self, candidate: &SnapshotFact) -> bool {
        self.source_timestamp == candidate.source_timestamp
            && self.sample_depth == candidate.sample_depth
            && self.book_sha256 == candidate.book_sha256
            && self.sampling_policy_sha256 == candidate.sampling_policy_sha256
            && self.payload_sha256 == candidate.payload_sha256
    }
}

#[derive(Debug, FromRow)]
struct ArtifactHashRow {
    payload_sha256: String,
    source_update_id: i64,
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

impl CaptureWriter {
    fn new(strategy: &BinanceSpotL2SnapshotStrategy) -> Result<Self, StrategyError> {
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
                    "binance_l2_artifact_load_failed",
                    format!("failed to load open Binance L2 artifact: {error}"),
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
                "binance_l2_artifact_config_mismatch",
                "open Binance L2 artifact config differs within the same profile generation",
            ));
        }
        let state = self.rebuild_open_artifact(open).await?;
        let should_rotate = generation_changed || state.artifact.capture_window_end <= Utc::now();
        self.current = Some(state);
        if should_rotate {
            self.seal_current(ArtifactSealFence::CurrentProfile).await?;
        }
        Ok(())
    }

    async fn rebuild_open_artifact(
        &self,
        artifact: CaptureArtifact,
    ) -> Result<OpenArtifact, StrategyError> {
        let rows = sqlx::query_as::<_, ArtifactHashRow>(
            r#"
            SELECT payload_sha256, source_update_id
            FROM market_data.binance_spot_btcusdt_l2_snapshots
            WHERE capture_artifact_id = $1
            ORDER BY source_update_id, source_timestamp, sampling_policy_sha256
            "#,
        )
        .bind(artifact.artifact_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| {
            database_error(
                "binance_l2_artifact_rebuild_failed",
                format!("failed to rebuild Binance L2 artifact checksum: {error}"),
            )
        })?;
        if i64::try_from(rows.len()).ok() != Some(artifact.record_count) {
            return Err(integrity_error(
                "binance_l2_artifact_count_mismatch",
                format!(
                    "open Binance L2 artifact {} declares {} rows but contains {}",
                    artifact.artifact_id,
                    artifact.record_count,
                    rows.len()
                ),
            ));
        }
        let mut content_hasher = Sha256::new();
        let mut end_cursor = artifact.end_cursor.clone();
        for row in rows {
            append_artifact_hash(&mut content_hasher, &row.payload_sha256);
            end_cursor = Some(format!("update_id:{}", row.source_update_id));
        }
        Ok(OpenArtifact {
            inserted_records: artifact.record_count,
            artifact,
            content_hasher,
            end_cursor,
        })
    }

    async fn ensure_artifact(
        &mut self,
        received_at: DateTime<Utc>,
        start_cursor: &str,
    ) -> Result<(), StrategyError> {
        if let Some(current) = self.current.as_ref() {
            if received_at < current.artifact.capture_window_start {
                return Err(integrity_error(
                    "binance_l2_receipt_clock_regression",
                    "receipt timestamp regressed before the current artifact window",
                ));
            }
            if received_at < current.artifact.capture_window_end {
                return Ok(());
            }
            self.seal_current(ArtifactSealFence::CurrentProfile).await?;
        }

        let (capture_window_start, capture_window_end) =
            aligned_window(received_at, self.config.artifact_window_seconds)?;
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error(
                "binance_l2_artifact_create_transaction_failed",
                format!("failed to begin Binance L2 artifact transaction: {error}"),
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
                    "binance_l2_artifact_create_failed",
                    format!("failed to create Binance L2 capture artifact: {error}"),
                )
            })?;
        transaction.commit().await.map_err(|error| {
            database_error(
                "binance_l2_artifact_create_commit_failed",
                format!("failed to commit Binance L2 artifact creation: {error}"),
            )
        })?;
        self.current = Some(OpenArtifact {
            artifact,
            content_hasher: Sha256::new(),
            inserted_records: 0,
            end_cursor: None,
        });
        Ok(())
    }

    async fn seal_current(&mut self, fence: ArtifactSealFence) -> Result<(), StrategyError> {
        if self.current.is_none() {
            return Ok(());
        }
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error(
                "binance_l2_artifact_complete_transaction_failed",
                format!("failed to begin Binance L2 artifact completion: {error}"),
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
        let current = self
            .current
            .take()
            .expect("open artifact was checked before awaiting its lease fence");
        let content_sha256 = encode_digest(current.content_hasher.finalize());
        let completed = self
            .artifacts
            .complete_in(
                &mut transaction,
                current.artifact.artifact_id,
                &content_sha256,
                current.end_cursor.as_deref(),
            )
            .await
            .map_err(|error| {
                database_error(
                    "binance_l2_artifact_complete_failed",
                    format!("failed to complete Binance L2 capture artifact: {error}"),
                )
            })?;
        if completed.is_none() {
            return Err(integrity_error(
                "binance_l2_artifact_not_open",
                "Binance L2 capture artifact was no longer open during completion",
            ));
        }
        transaction.commit().await.map_err(|error| {
            database_error(
                "binance_l2_artifact_complete_commit_failed",
                format!("failed to commit Binance L2 artifact completion: {error}"),
            )
        })?;
        info!(
            artifact_id = %current.artifact.artifact_id,
            record_count = current.inserted_records,
            "completed Binance spot L2 capture artifact"
        );
        Ok(())
    }

    async fn seal_owned_drain(&mut self) -> Result<(), StrategyError> {
        if self.current.is_some() {
            return self.seal_current(ArtifactSealFence::OwnedDrain).await;
        }

        // A strict write can lose the current desired-generation fence before
        // any artifact exists. Still prove ownership of the applied generation
        // before treating that exit as a requested drain.
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error(
                "binance_l2_owned_drain_transaction_failed",
                format!("failed to begin Binance L2 owned-drain transaction: {error}"),
            )
        })?;
        self.lock_owned_lease(&mut transaction, "confirming an empty owned drain")
            .await?;
        transaction.commit().await.map_err(|error| {
            database_error(
                "binance_l2_owned_drain_commit_failed",
                format!("failed to commit Binance L2 owned-drain fence: {error}"),
            )
        })?;
        Ok(())
    }

    async fn persist(&mut self, fact: &SnapshotFact) -> Result<(), StrategyError> {
        let cursor = fact.cursor();
        self.ensure_artifact(fact.received_at, &cursor).await?;
        let artifact_id = self
            .current
            .as_ref()
            .map(|current| current.artifact.artifact_id)
            .ok_or_else(|| {
                integrity_error(
                    "binance_l2_artifact_missing",
                    "Binance L2 fact has no open capture artifact",
                )
            })?;
        let advisory_key = fact_advisory_key(fact);
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error(
                "binance_l2_transaction_begin_failed",
                format!("failed to begin Binance L2 fact transaction: {error}"),
            )
        })?;
        sqlx::query("SELECT pg_advisory_xact_lock($1)")
            .bind(advisory_key)
            .execute(&mut *transaction)
            .await
            .map_err(|error| {
                database_error(
                    "binance_l2_natural_key_lock_failed",
                    format!("failed to lock Binance L2 natural key: {error}"),
                )
            })?;

        let existing = sqlx::query_as::<_, ExistingSnapshotFact>(
            r#"
            SELECT source_timestamp, sample_depth, book_sha256,
                   sampling_policy_sha256, payload_sha256
            FROM market_data.binance_spot_btcusdt_l2_snapshots
            WHERE symbol = $1
              AND source_update_id = $2
              AND sampling_policy_sha256 = $3
            ORDER BY source_timestamp
            LIMIT 2
            "#,
        )
        .bind(SYMBOL)
        .bind(fact.source_update_id)
        .bind(&fact.sampling_policy_sha256)
        .fetch_all(&mut *transaction)
        .await
        .map_err(|error| {
            database_error(
                "binance_l2_fact_lookup_failed",
                format!("failed to check Binance L2 natural key: {error}"),
            )
        })?;

        let inserted = match existing.as_slice() {
            [] => {
                sqlx::query(
                    r#"
                    INSERT INTO market_data.binance_spot_btcusdt_l2_snapshots (
                      source_timestamp, received_at, symbol, source_update_id,
                      connection_epoch, sample_depth, bids, asks, book_sha256,
                      sampling_policy, sampling_policy_sha256, payload_sha256,
                      capture_artifact_id
                    ) VALUES (
                      $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
                    )
                    "#,
                )
                .bind(fact.source_timestamp)
                .bind(fact.received_at)
                .bind(SYMBOL)
                .bind(fact.source_update_id)
                .bind(fact.connection_epoch)
                .bind(fact.sample_depth)
                .bind(&fact.bids)
                .bind(&fact.asks)
                .bind(&fact.book_sha256)
                .bind(&fact.sampling_policy)
                .bind(&fact.sampling_policy_sha256)
                .bind(&fact.payload_sha256)
                .bind(artifact_id)
                .execute(&mut *transaction)
                .await
                .map_err(|error| {
                    database_error(
                        "binance_l2_fact_insert_failed",
                        format!("failed to insert Binance L2 snapshot fact: {error}"),
                    )
                })?;
                true
            }
            [stored] if stored.matches(fact) => false,
            [stored] => {
                return Err(integrity_error(
                    "binance_l2_conflicting_payload",
                    format!(
                        "Binance L2 natural key {} conflicts with stored payload {}",
                        fact.source_update_id, stored.payload_sha256
                    ),
                ));
            }
            _ => {
                return Err(integrity_error(
                    "binance_l2_duplicate_natural_key",
                    format!(
                        "Binance L2 natural key {} has multiple durable rows",
                        fact.source_update_id
                    ),
                ));
            }
        };

        if inserted {
            let artifact = self
                .artifacts
                .record_batch_in(
                    &mut transaction,
                    artifact_id,
                    &ArtifactBatch {
                        inserted_record_count: 1,
                        minimum_source_timestamp: Some(fact.source_timestamp),
                        maximum_source_timestamp: Some(fact.source_timestamp),
                        minimum_received_at: Some(fact.received_at),
                        maximum_received_at: Some(fact.received_at),
                        start_cursor: Some(cursor.clone()),
                        end_cursor: Some(cursor.clone()),
                    },
                )
                .await
                .map_err(|error| {
                    database_error(
                        "binance_l2_artifact_progress_failed",
                        format!("failed to update Binance L2 artifact lineage: {error}"),
                    )
                })?;
            if artifact.is_none() {
                return Err(integrity_error(
                    "binance_l2_artifact_not_open",
                    "Binance L2 artifact closed during fact transaction",
                ));
            }
        }

        let progress = StrategyProgress {
            verified_record_count: 1,
            checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
            checkpoint: json!({
                "connection_epoch": fact.connection_epoch,
                "source_update_id": fact.source_update_id,
                "source_timestamp": fact.source_timestamp,
                "received_at": fact.received_at,
                "sampling_policy_sha256": fact.sampling_policy_sha256,
                "artifact_id": artifact_id
            }),
            last_source_event_at: Some(fact.source_timestamp),
            last_provider_available_at: None,
            source_watermark: Some(fact.source_timestamp),
            availability_watermark: None,
        };
        let progress_recorded = self
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
                    "binance_l2_profile_progress_failed",
                    format!("failed to update Binance L2 profile progress: {error}"),
                )
            })?;
        if !progress_recorded {
            return Err(lease_error(
                "binance_l2_lease_lost",
                "Binance L2 lease was lost before committing fact progress",
            ));
        }
        transaction.commit().await.map_err(|error| {
            database_error(
                "binance_l2_transaction_commit_failed",
                format!("failed to commit Binance L2 fact transaction: {error}"),
            )
        })?;

        if inserted {
            let current = self.current.as_mut().ok_or_else(|| {
                integrity_error(
                    "binance_l2_artifact_missing",
                    "Binance L2 artifact disappeared after fact commit",
                )
            })?;
            append_artifact_hash(&mut current.content_hasher, &fact.payload_sha256);
            current.inserted_records += 1;
            current.end_cursor = Some(cursor);
        }
        Ok(())
    }

    async fn record_gap(&mut self, gap: GapObservation<'_>) -> Result<(), StrategyError> {
        let (source_start, source_end) = match (gap.source_start, gap.source_end) {
            (Some(start), Some(end)) if end >= start => (Some(start), Some(end)),
            _ => (None, None),
        };
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error(
                "binance_l2_gap_transaction_failed",
                format!("failed to begin Binance L2 gap transaction: {error}"),
            )
        })?;
        self.lock_current_lease(&mut transaction, "recording a data gap")
            .await?;
        let detected = self
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
                    "binance_l2_gap_persist_failed",
                    format!("failed to persist Binance L2 data gap: {error}"),
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
                    "binance_l2_degraded_state_failed",
                    format!("failed to mark Binance L2 strategy degraded: {error}"),
                )
            })?;
        if !degraded {
            return Err(lease_error(
                "binance_l2_lease_lost",
                "Binance L2 lease was lost while recording a data gap",
            ));
        }
        if !detected.gap.status.is_resolved() {
            let resolved = self
                .gaps
                .mark_unrecoverable_in(
                    &mut transaction,
                    detected.gap.gap_id,
                    "realtime_resnapshot_only",
                    Some(
                        "Binance does not expose historical diff-depth replay; continuity resumes from a new REST snapshot",
                    ),
                )
                .await
                .map_err(|error| {
                    database_error(
                        "binance_l2_gap_resolution_failed",
                        format!("failed to terminalize Binance L2 data gap: {error}"),
                    )
                })?;
            if resolved.is_none() {
                return Err(integrity_error(
                    "binance_l2_gap_resolution_conflict",
                    "Binance L2 gap changed before it could be marked unrecoverable",
                ));
            }
        }
        transaction.commit().await.map_err(|error| {
            database_error(
                "binance_l2_gap_commit_failed",
                format!("failed to commit Binance L2 data gap: {error}"),
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
                    "binance_l2_lease_lock_failed",
                    format!("failed to lock Binance L2 profile lease: {error}"),
                )
            })?;
        if !locked {
            return Err(lease_error(
                "binance_l2_lease_lost",
                format!("Binance L2 lease was lost before {action}"),
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
                    "binance_l2_owned_lease_lock_failed",
                    format!("failed to lock owned Binance L2 profile lease: {error}"),
                )
            })?;
        if !locked {
            return Err(lease_error(
                "binance_l2_lease_lost",
                format!("Binance L2 lease was lost before {action}"),
            ));
        }
        Ok(())
    }
}

fn fence_open_artifact_generation(
    artifact_generation: i64,
    strategy_generation: i64,
) -> Result<bool, StrategyError> {
    if artifact_generation > strategy_generation {
        return Err(lease_error(
            "binance_l2_newer_artifact_generation",
            format!(
                "open Binance L2 artifact generation {artifact_generation} is newer than owned generation {strategy_generation}"
            ),
        ));
    }
    Ok(artifact_generation < strategy_generation)
}

fn aligned_window(
    timestamp: DateTime<Utc>,
    window_seconds: i64,
) -> Result<(DateTime<Utc>, DateTime<Utc>), StrategyError> {
    let start_seconds = timestamp.timestamp().div_euclid(window_seconds) * window_seconds;
    let end_seconds = start_seconds.checked_add(window_seconds).ok_or_else(|| {
        integrity_error(
            "binance_l2_artifact_window_overflow",
            "Binance L2 artifact window overflowed",
        )
    })?;
    let start = Utc
        .timestamp_opt(start_seconds, 0)
        .single()
        .ok_or_else(|| {
            integrity_error(
                "binance_l2_artifact_window_invalid",
                "Binance L2 artifact start was outside UTC range",
            )
        })?;
    let end = Utc.timestamp_opt(end_seconds, 0).single().ok_or_else(|| {
        integrity_error(
            "binance_l2_artifact_window_invalid",
            "Binance L2 artifact end was outside UTC range",
        )
    })?;
    Ok((start, end))
}

fn fact_advisory_key(fact: &SnapshotFact) -> i64 {
    let value = format!(
        "{SYMBOL}:{}:{}",
        fact.source_update_id, fact.sampling_policy_sha256
    );
    let digest = Sha256::digest(value.as_bytes());
    i64::from_be_bytes(digest[..8].try_into().expect("SHA-256 has eight bytes"))
}

fn hash_json(value: &Value) -> Result<String, StrategyError> {
    let encoded = serde_json::to_vec(value).map_err(|error| {
        integrity_error(
            "binance_l2_hash_serialization",
            format!("failed to serialize Binance L2 hash payload: {error}"),
        )
    })?;
    Ok(encode_digest(Sha256::digest(encoded)))
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

#[async_trait]
impl IngesterStrategy for BinanceSpotL2SnapshotStrategy {
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
                                "Binance spot L2 strict write observed a requested profile drain"
                            );
                            shutdown.cancelled().await;
                            return Ok(());
                        }
                        Err(drain_error) if drain_error.kind == StrategyErrorKind::LeaseLost => {
                            warn!(
                                error = %error,
                                drain_error = %drain_error,
                                "Binance spot L2 no longer owns the applied generation during drain"
                            );
                            return Err(error);
                        }
                        Err(drain_error) => return Err(drain_error),
                    }
                }
                Err(error) if error.kind == StrategyErrorKind::TransientSource => {
                    if !gap_was_recorded(error.code) {
                        if let Some(last_update_id) = continuity.last_update_id {
                            writer
                                .record_gap(GapObservation {
                                    kind: "transport_interruption",
                                    code: error.code,
                                    message: &error.message,
                                    source_start: None,
                                    source_end: None,
                                    start_cursor: Some(format!("after_update_id:{last_update_id}")),
                                    end_cursor: None,
                                })
                                .await?;
                        }
                    }
                    warn!(
                        connection_epoch = %connection_epoch,
                        error_code = error.code,
                        error = %error,
                        "Binance spot L2 session will reconnect and resnapshot"
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

impl BinanceSpotL2SnapshotStrategy {
    async fn capture_session(
        &self,
        writer: &mut CaptureWriter,
        sampling_clock: &mut SamplingClock,
        continuity: &mut Continuity,
        connection_epoch: Uuid,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        let websocket = tokio::time::timeout(
            Duration::from_millis(self.config.connect_timeout_ms),
            connect_async(&self.config.websocket_url),
        )
        .await
        .map_err(|_| {
            source_error(
                "binance_l2_websocket_connect_timeout",
                "timed out connecting to Binance spot L2 websocket",
            )
        })?
        .map_err(|error| {
            source_error(
                "binance_l2_websocket_connect_failed",
                format!("failed to connect Binance spot L2 websocket: {error}"),
            )
        })?
        .0;
        let (mut sink, mut stream) = websocket.split();
        let read_timeout = Duration::from_millis(self.config.read_timeout_ms);
        let write_timeout = Duration::from_millis(self.config.connect_timeout_ms);
        let ping_interval = Duration::from_millis(self.config.ping_interval_ms);
        let (io_sender, mut io_receiver) =
            mpsc::channel::<BinanceIoEvent>(self.config.max_buffered_updates);
        let io_shutdown = CancellationToken::new();
        let worker_shutdown = io_shutdown.clone();
        let io_handle = tokio::spawn(async move {
            let mut ping = tokio::time::interval_at(Instant::now() + ping_interval, ping_interval);
            ping.set_missed_tick_behavior(MissedTickBehavior::Skip);
            let mut read_deadline = Instant::now() + read_timeout;
            loop {
                tokio::select! {
                    biased;
                    _ = worker_shutdown.cancelled() => return,
                    _ = tokio::time::sleep_until(read_deadline) => {
                        let _ = io_sender.send(BinanceIoEvent::Failed(source_error(
                            "binance_l2_websocket_read_timeout",
                            "Binance spot L2 websocket produced no frames before its read deadline",
                        ))).await;
                        return;
                    }
                    _ = ping.tick() => {
                        if let Err(error) = send_websocket_control(
                            &mut sink,
                            Message::Ping(Vec::new().into()),
                            write_timeout,
                        ).await {
                            let _ = io_sender.send(BinanceIoEvent::Failed(error)).await;
                            return;
                        }
                    }
                    frame = stream.next() => {
                        read_deadline = Instant::now() + read_timeout;
                        let event = match frame {
                            Some(Ok(Message::Text(text))) => {
                                let received_at = Utc::now();
                                match parse_depth_update(text.as_ref()) {
                                    Ok(update) => BinanceIoEvent::Update(BufferedUpdate {
                                        update,
                                        received_at,
                                    }),
                                    Err(error) => BinanceIoEvent::Failed(error),
                                }
                            }
                            Some(Ok(Message::Ping(payload))) => {
                                if let Err(error) = send_websocket_control(
                                    &mut sink,
                                    Message::Pong(payload),
                                    write_timeout,
                                ).await {
                                    let _ = io_sender.send(BinanceIoEvent::Failed(error)).await;
                                    return;
                                }
                                continue;
                            }
                            Some(Ok(Message::Pong(_))) => continue,
                            Some(Ok(Message::Close(frame))) => BinanceIoEvent::Failed(source_error(
                                "binance_l2_websocket_closed",
                                format!("Binance spot L2 websocket closed: {frame:?}"),
                            )),
                            Some(Ok(Message::Binary(_))) => BinanceIoEvent::Failed(source_error(
                                "binance_l2_binary_frame",
                                "Binance spot L2 websocket sent an unsupported binary frame",
                            )),
                            Some(Ok(_)) => continue,
                            Some(Err(error)) => BinanceIoEvent::Failed(source_error(
                                "binance_l2_websocket_read_failed",
                                format!("failed to read Binance spot L2 websocket: {error}"),
                            )),
                            None => BinanceIoEvent::Failed(source_error(
                                "binance_l2_websocket_eof",
                                "Binance spot L2 websocket ended",
                            )),
                        };
                        match io_sender.try_send(event) {
                            Ok(()) => {}
                            Err(mpsc::error::TrySendError::Closed(_)) => return,
                            Err(mpsc::error::TrySendError::Full(_)) => {
                                let _ = io_sender.send(BinanceIoEvent::Failed(source_error(
                                    "binance_l2_consumer_backpressure",
                                    "Binance spot L2 processing fell behind the bounded websocket buffer",
                                ))).await;
                                return;
                            }
                        }
                    }
                }
            }
        });
        let _io_worker = BinanceIoWorker {
            shutdown: io_shutdown,
            handle: io_handle,
        };
        let mut buffer = DeltaBuffer::new(
            self.config.max_buffered_updates,
            self.config.max_buffered_levels,
        );

        // The websocket is deliberately open before REST bootstrap begins.
        // Deltas received while REST is in flight are bounded and replayed
        // against lastUpdateId using Binance's U/u bridge protocol.
        let bootstrap = tokio::time::timeout(
            Duration::from_millis(self.config.bootstrap_timeout_ms),
            self.fetch_depth_snapshot(),
        );
        tokio::pin!(bootstrap);
        let snapshot = loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => return Ok(()),
                response = &mut bootstrap => {
                    break response.map_err(|_| {
                        source_error(
                            "binance_l2_bootstrap_timeout",
                            "timed out fetching Binance spot L2 REST snapshot",
                        )
                    })??;
                }
                event = io_receiver.recv() => {
                    match event {
                        Some(BinanceIoEvent::Update(buffered)) => {
                            let incoming_source_timestamp = buffered.update.source_timestamp;
                            let incoming_final_update_id = buffered.update.final_update_id;
                            if let Err(error) = buffer.push(buffered) {
                                let start = buffer.updates.front().map(|item| {
                                    format!("update_id:{}", item.update.first_update_id)
                                });
                                let end = Some(format!("update_id:{incoming_final_update_id}"));
                                writer.record_gap(GapObservation {
                                    kind: "bootstrap_buffer_overflow",
                                    code: error.code,
                                    message: &error.message,
                                    source_start: buffer.updates.front().map(|item| item.update.source_timestamp),
                                    source_end: Some(incoming_source_timestamp),
                                    start_cursor: start,
                                    end_cursor: end,
                                }).await?;
                                return Err(error);
                            }
                        }
                        Some(BinanceIoEvent::Failed(error)) => return Err(error),
                        None => return Err(source_error(
                            "binance_l2_io_worker_stopped",
                            "Binance spot L2 socket worker stopped during bootstrap",
                        )),
                    }
                }
            }
        };

        let mut book = OrderBook::default();
        if let Err(error) = book.install_snapshot(snapshot, self.config.max_book_levels_per_side) {
            writer
                .record_gap(GapObservation {
                    kind: "snapshot_integrity",
                    code: error.code,
                    message: &error.message,
                    source_start: continuity.last_source_timestamp,
                    source_end: None,
                    start_cursor: continuity
                        .last_update_id
                        .map(|id| format!("after_update_id:{id}")),
                    end_cursor: book
                        .last_update_id
                        .map(|id| format!("snapshot_update_id:{id}")),
                })
                .await?;
            return Err(error);
        }
        while let Some(buffered) = buffer.updates.pop_front() {
            self.apply_update(
                writer,
                sampling_clock,
                continuity,
                &mut book,
                connection_epoch,
                buffered,
            )
            .await?;
        }

        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => return Ok(()),
                event = io_receiver.recv() => {
                    match event {
                        Some(BinanceIoEvent::Update(buffered)) => {
                            self.apply_update(
                                writer,
                                sampling_clock,
                                continuity,
                                &mut book,
                                connection_epoch,
                                buffered,
                            ).await?;
                        }
                        Some(BinanceIoEvent::Failed(error)) => return Err(error),
                        None => return Err(source_error(
                            "binance_l2_io_worker_stopped",
                            "Binance spot L2 socket worker stopped unexpectedly",
                        )),
                    }
                }
            }
        }
    }

    async fn fetch_depth_snapshot(&self) -> Result<DepthSnapshot, StrategyError> {
        let limit = self.config.rest_depth_limit.to_string();
        let response = self
            .client
            .get(&self.config.rest_depth_url)
            .query(&[
                ("symbol", self.config.symbol.as_str()),
                ("limit", limit.as_str()),
            ])
            .send()
            .await
            .map_err(|error| {
                source_error(
                    "binance_l2_snapshot_request_failed",
                    format!("failed to request Binance spot L2 snapshot: {error}"),
                )
            })?
            .error_for_status()
            .map_err(|error| {
                source_error(
                    "binance_l2_snapshot_status_failed",
                    format!("Binance spot L2 snapshot returned an error status: {error}"),
                )
            })?;
        if response
            .content_length()
            .is_some_and(|length| length > MAX_SNAPSHOT_RESPONSE_BYTES as u64)
        {
            return Err(source_error(
                "binance_l2_snapshot_too_large",
                "Binance spot L2 snapshot response exceeded the bounded payload size",
            ));
        }
        let initial_capacity = response
            .content_length()
            .and_then(|length| usize::try_from(length).ok())
            .unwrap_or_default()
            .min(MAX_SNAPSHOT_RESPONSE_BYTES);
        let mut bytes = Vec::with_capacity(initial_capacity);
        let mut chunks = response.bytes_stream();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.map_err(|error| {
                source_error(
                    "binance_l2_snapshot_read_failed",
                    format!("failed to read Binance spot L2 snapshot: {error}"),
                )
            })?;
            if chunk.len() > MAX_SNAPSHOT_RESPONSE_BYTES.saturating_sub(bytes.len()) {
                return Err(source_error(
                    "binance_l2_snapshot_too_large",
                    "Binance spot L2 snapshot response exceeded the bounded payload size",
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        let wire = serde_json::from_slice::<WireDepthSnapshot>(&bytes).map_err(|error| {
            source_error(
                "binance_l2_snapshot_decode_failed",
                format!("failed to decode Binance spot L2 snapshot: {error}"),
            )
        })?;
        parse_depth_snapshot(wire)
    }

    async fn apply_update(
        &self,
        writer: &mut CaptureWriter,
        sampling_clock: &mut SamplingClock,
        continuity: &mut Continuity,
        book: &mut OrderBook,
        connection_epoch: Uuid,
        buffered: BufferedUpdate,
    ) -> Result<(), StrategyError> {
        let outcome = match book.apply(&buffered.update, self.config.max_book_levels_per_side) {
            Ok(outcome) => outcome,
            Err(error) => {
                writer
                    .record_gap(GapObservation {
                        kind: "book_integrity",
                        code: error.code,
                        message: &error.message,
                        source_start: continuity.last_source_timestamp,
                        source_end: Some(buffered.update.source_timestamp),
                        start_cursor: continuity
                            .last_update_id
                            .map(|id| format!("after_update_id:{id}")),
                        end_cursor: Some(format!("update_id:{}", buffered.update.first_update_id)),
                    })
                    .await?;
                return Err(error);
            }
        };
        match outcome {
            ApplyOutcome::Stale => return Ok(()),
            ApplyOutcome::Gap {
                expected,
                first,
                final_id,
            } => {
                let message = format!(
                    "expected Binance spot update {expected}, received range [{first}, {final_id}]"
                );
                writer
                    .record_gap(GapObservation {
                        kind: "source_sequence",
                        code: "binance_l2_sequence_gap",
                        message: &message,
                        source_start: continuity.last_source_timestamp,
                        source_end: Some(buffered.update.source_timestamp),
                        start_cursor: Some(format!("update_id:{expected}")),
                        end_cursor: Some(format!("update_id:{first}")),
                    })
                    .await?;
                return Err(source_error("binance_l2_sequence_gap", message));
            }
            ApplyOutcome::Applied { synchronized_now } => {
                if synchronized_now {
                    info!(
                        connection_epoch = %connection_epoch,
                        source_update_id = buffered.update.final_update_id,
                        "Binance spot L2 book synchronized"
                    );
                }
            }
        }

        continuity.last_update_id = Some(buffered.update.final_update_id);
        continuity.last_source_timestamp = Some(buffered.update.source_timestamp);
        continuity.last_received_at = Some(buffered.received_at);
        let sampling_claim =
            sampling_clock.claim(buffered.received_at, self.config.sample_interval_ms);
        if let Some((first_missed, last_missed)) = sampling_claim.missed_slots {
            let message = format!(
                "Binance L2 sampler skipped receipt slots {first_missed} through {last_missed}"
            );
            writer
                .record_gap(GapObservation {
                    kind: "local_sampling_cadence",
                    code: "binance_l2_sampling_slot_gap",
                    message: &message,
                    source_start: None,
                    source_end: None,
                    start_cursor: Some(format!(
                        "sampling_slot:{}",
                        first_missed.timestamp_millis()
                    )),
                    end_cursor: Some(format!("sampling_slot:{}", last_missed.timestamp_millis())),
                })
                .await?;
        }
        if sampling_claim.should_sample {
            let sample = match book.sample(self.config.top_n) {
                Ok(sample) => sample,
                Err(error) => {
                    writer
                        .record_gap(GapObservation {
                            kind: "book_integrity",
                            code: error.code,
                            message: &error.message,
                            source_start: continuity.last_source_timestamp,
                            source_end: Some(buffered.update.source_timestamp),
                            start_cursor: continuity
                                .last_update_id
                                .map(|id| format!("after_update_id:{id}")),
                            end_cursor: Some(format!(
                                "update_id:{}",
                                buffered.update.final_update_id
                            )),
                        })
                        .await?;
                    return Err(error);
                }
            };
            let fact = SnapshotFact::new(
                &buffered.update,
                buffered.received_at,
                connection_epoch,
                &sample,
                &writer.sampling_policy,
                &writer.sampling_policy_sha256,
            )?;
            writer.persist(&fact).await?;
        }
        Ok(())
    }
}

async fn send_websocket_control<S>(
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
                "binance_l2_websocket_write_timeout",
                "timed out writing Binance spot L2 websocket control frame",
            )
        })?
        .map_err(|error| {
            source_error(
                "binance_l2_websocket_write_failed",
                format!("failed to write Binance spot L2 websocket control frame: {error}"),
            )
        })
}

fn gap_was_recorded(code: &str) -> bool {
    matches!(
        code,
        "binance_l2_sequence_gap"
            | "binance_l2_sampling_slot_gap"
            | "binance_l2_buffer_overflow"
            | "binance_l2_book_too_large"
            | "binance_l2_crossed_book"
            | "binance_l2_insufficient_depth"
    )
}

fn is_current_profile_lease_loss(error: &StrategyError) -> bool {
    error.kind == StrategyErrorKind::LeaseLost && error.code == "binance_l2_lease_lost"
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
    use chrono::TimeDelta;
    use pretty_assertions::assert_eq;

    use super::*;

    const UPDATE_FIXTURE: &str =
        include_str!("../../../tests/fixtures/binance/spot_l2_depth_update.json");
    const SNAPSHOT_FIXTURE: &str =
        include_str!("../../../tests/fixtures/binance/spot_l2_depth_snapshot.json");

    fn snapshot() -> DepthSnapshot {
        let wire = serde_json::from_str::<WireDepthSnapshot>(SNAPSHOT_FIXTURE)
            .expect("snapshot fixture JSON");
        parse_depth_snapshot(wire).expect("snapshot fixture")
    }

    fn update(first: u64, final_id: u64, bid_quantity: &str) -> DepthUpdate {
        DepthUpdate {
            source_timestamp: Utc
                .timestamp_millis_opt(1_776_000_000_100 + final_id as i64)
                .single()
                .expect("timestamp"),
            first_update_id: first,
            final_update_id: final_id,
            bids: vec![PriceLevel {
                price: Decimal::from_str("100").expect("price"),
                quantity: Decimal::from_str(bid_quantity).expect("quantity"),
            }],
            asks: Vec::new(),
        }
    }

    #[test]
    fn config_is_typed_narrow_and_spot_only() {
        let defaults = BinanceSpotL2SnapshotConfig::from_value(&json!({})).expect("default config");
        assert_eq!(defaults.symbol, SYMBOL);
        assert_eq!(defaults.top_n, 20);
        assert!(BinanceSpotL2SnapshotConfig::from_value(&json!({
            "unknown": true
        }))
        .is_err());
        assert!(BinanceSpotL2SnapshotConfig::from_value(&json!({
            "symbol": "BTCUSD_PERP"
        }))
        .is_err());
        assert!(BinanceSpotL2SnapshotConfig::from_value(&json!({
            "ping_interval_ms": 45000,
            "read_timeout_ms": 45000
        }))
        .is_err());
    }

    #[test]
    fn sanitized_provider_fixtures_parse_strictly() {
        let parsed = parse_depth_update(UPDATE_FIXTURE).expect("update fixture");
        assert_eq!(parsed.first_update_id, 101);
        assert_eq!(parsed.final_update_id, 102);
        assert_eq!(parsed.level_count(), 3);
        assert!(parsed.bids[1].quantity.is_zero());

        let parsed_snapshot = snapshot();
        assert_eq!(parsed_snapshot.last_update_id, 100);
        assert_eq!(parsed_snapshot.bids.len(), 3);
        assert_eq!(parsed_snapshot.asks.len(), 3);

        let wrong_symbol = UPDATE_FIXTURE.replace("BTCUSDT", "ETHUSDT");
        assert!(parse_depth_update(&wrong_symbol).is_err());
        let inverted = UPDATE_FIXTURE.replace("\"U\": 101", "\"U\": 103");
        assert!(parse_depth_update(&inverted).is_err());
    }

    #[test]
    fn websocket_first_snapshot_bridge_is_strict_after_initial_overlap() {
        let mut book = OrderBook::default();
        book.install_snapshot(snapshot(), 100).expect("snapshot");

        assert_eq!(
            book.apply(&update(99, 100, "2"), 100)
                .expect("stale update"),
            ApplyOutcome::Stale
        );
        assert_eq!(
            book.apply(&update(99, 101, "2"), 100)
                .expect("overlap bridge"),
            ApplyOutcome::Applied {
                synchronized_now: true
            }
        );
        assert_eq!(
            book.apply(&update(102, 103, "3"), 100)
                .expect("contiguous update"),
            ApplyOutcome::Applied {
                synchronized_now: false
            }
        );
        assert_eq!(
            book.apply(&update(103, 104, "4"), 100)
                .expect("overlap is a gap after synchronization"),
            ApplyOutcome::Gap {
                expected: 104,
                first: 103,
                final_id: 104
            }
        );
    }

    #[test]
    fn zero_quantity_deletes_and_top_n_is_price_sorted() {
        let mut book = OrderBook::default();
        book.install_snapshot(snapshot(), 100).expect("snapshot");
        book.apply(&update(101, 101, "0"), 100)
            .expect("delete best bid");

        assert!(!book
            .bids
            .contains_key(&Decimal::from_str("100").expect("price")));
        let sample = book.sample(2).expect("top two");
        assert_eq!(
            sample.bids,
            vec![
                ["99.50".to_owned(), "3.00".to_owned()],
                ["99.00".to_owned(), "5.00".to_owned()]
            ]
        );
        assert_eq!(
            sample.asks,
            vec![
                ["100.50".to_owned(), "2.00".to_owned()],
                ["101.00".to_owned(), "4.00".to_owned()]
            ]
        );
    }

    #[test]
    fn bootstrap_delta_buffer_fails_closed_at_either_bound() {
        let buffered = BufferedUpdate {
            update: update(101, 101, "1"),
            received_at: Utc::now(),
        };
        let mut event_bounded = DeltaBuffer::new(1, 10);
        event_bounded.push(buffered.clone()).expect("first event");
        assert!(event_bounded.push(buffered.clone()).is_err());

        let mut level_bounded = DeltaBuffer::new(10, 1);
        level_bounded.push(buffered).expect("first level");
        assert!(level_bounded
            .push(BufferedUpdate {
                update: update(102, 102, "1"),
                received_at: Utc::now(),
            })
            .is_err());
    }

    #[test]
    fn sampling_clock_emits_at_most_once_per_interval() {
        let start = Utc
            .timestamp_millis_opt(1_776_000_000_000)
            .single()
            .expect("timestamp");
        let mut clock = SamplingClock::default();
        assert!(clock.claim(start, 1_000).should_sample);
        assert!(
            !clock
                .claim(start + TimeDelta::milliseconds(999), 1_000)
                .should_sample
        );
        assert!(
            clock
                .claim(start + TimeDelta::milliseconds(1_000), 1_000)
                .should_sample
        );
        let delayed = clock.claim(start + TimeDelta::milliseconds(5_500), 1_000);
        assert!(delayed.should_sample);
        assert_eq!(
            delayed.missed_slots,
            Some((
                start + TimeDelta::milliseconds(2_000),
                start + TimeDelta::milliseconds(4_000),
            ))
        );
        assert!(
            !clock
                .claim(start + TimeDelta::milliseconds(5_999), 1_000)
                .should_sample
        );
    }

    #[test]
    fn factual_hash_is_idempotent_across_capture_metadata() {
        let config = BinanceSpotL2SnapshotConfig {
            top_n: 2,
            ..Default::default()
        };
        let policy = config.sampling_policy();
        let policy_hash = hash_json(&policy).expect("policy hash");
        let mut book = OrderBook::default();
        book.install_snapshot(snapshot(), 100).expect("snapshot");
        let update = update(101, 101, "2");
        book.apply(&update, 100).expect("bridge");
        let sample = book.sample(2).expect("sample");
        let first = SnapshotFact::new(
            &update,
            Utc::now(),
            Uuid::new_v4(),
            &sample,
            &policy,
            &policy_hash,
        )
        .expect("first fact");
        let second = SnapshotFact::new(
            &update,
            Utc::now() + TimeDelta::seconds(1),
            Uuid::new_v4(),
            &sample,
            &policy,
            &policy_hash,
        )
        .expect("second fact");

        assert_eq!(first.book_sha256, second.book_sha256);
        assert_eq!(first.payload_sha256, second.payload_sha256);
        let stored = ExistingSnapshotFact {
            source_timestamp: first.source_timestamp,
            sample_depth: first.sample_depth,
            book_sha256: first.book_sha256.clone(),
            sampling_policy_sha256: first.sampling_policy_sha256.clone(),
            payload_sha256: first.payload_sha256.clone(),
        };
        assert!(stored.matches(&second));

        let mut changed = second;
        changed.book_sha256 = "0".repeat(64);
        assert!(!stored.matches(&changed));
    }

    #[test]
    fn artifact_checksum_and_window_alignment_are_deterministic() {
        let timestamp = Utc
            .timestamp_opt(1_776_003_723, 123_000_000)
            .single()
            .expect("timestamp");
        let (start, end) = aligned_window(timestamp, 3_600).expect("window");
        assert_eq!(start.timestamp() % 3_600, 0);
        assert_eq!(end - start, TimeDelta::hours(1));

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
    }

    #[test]
    fn only_current_profile_fence_loss_can_enter_owned_drain() {
        let requested_drain = lease_error(
            "binance_l2_lease_lost",
            "strict write observed a desired-generation change",
        );
        assert!(is_current_profile_lease_loss(&requested_drain));

        let newer_artifact = lease_error(
            "binance_l2_newer_artifact_generation",
            "a newer strategy owns the open artifact",
        );
        assert!(!is_current_profile_lease_loss(&newer_artifact));
        assert!(!is_current_profile_lease_loss(&source_error(
            "binance_l2_websocket_eof",
            "source disconnected",
        )));
    }

    #[test]
    fn an_older_strategy_cannot_seal_a_newer_generation_artifact() {
        let error = fence_open_artifact_generation(12, 11)
            .expect_err("newer artifact must fence stale strategy");
        assert_eq!(error.kind, StrategyErrorKind::LeaseLost);
        assert_eq!(error.code, "binance_l2_newer_artifact_generation");
        assert!(!fence_open_artifact_generation(11, 11).expect("same generation"));
        assert!(fence_open_artifact_generation(10, 11).expect("newer strategy"));
    }
}
