//! Continuous Binance Spot BTCUSDT provider-native one-second OHLCV ingestion.
//!
//! Only closed `@kline_1s` events and completed REST klines are persisted.
//! Missing seconds are gap evidence; this strategy never synthesizes candles.

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
use tokio::{
    io::{AsyncRead, AsyncWrite},
    sync::mpsc,
    task::JoinHandle,
};
use tokio_tungstenite::{connect_async, tungstenite::Message, WebSocketStream};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    domain::{
        CaptureArtifact, IngesterProfile, IngesterStrategyKey, RealtimeWorkerStrategy,
        StrategyError, StrategyErrorKind,
    },
    persistence::{
        ArtifactBatch, ArtifactRepository, GapRepository, NewCaptureArtifact, NewDataGap,
        ProfileRepository, StrategyDegradation, StrategyProgress,
    },
    runtime::{StrategyFactory, StrategyFactoryError},
};

pub const STRATEGY_KEY: IngesterStrategyKey = IngesterStrategyKey::BinanceSpotBtcusdtOneSecondOhlcv;
pub const CONFIG_SCHEMA_VERSION: i32 = 1;
pub const CHECKPOINT_SCHEMA_VERSION: i32 = 1;

const SOURCE: &str = "binance_spot";
const SYMBOL: &str = "BTCUSDT";
const INTERVAL: &str = "1s";
const DEFAULT_WEBSOCKET_URL: &str = "wss://stream.binance.com:9443/ws/btcusdt@kline_1s";
const DEFAULT_REST_BASE_URL: &str = "https://data-api.binance.vision";
const MAX_REST_BODY_BYTES: usize = 1_048_576;
const REST_BOUNDARY_STABILIZATION_DELAY: Duration = Duration::from_secs(10);
const PERSISTENCE_QUEUE_CAPACITY: usize = 1_024;
const MAX_PROVIDER_CLOCK_SKEW: chrono::Duration = chrono::Duration::minutes(5);
const ALLOWED_WEBSOCKET_URLS: [&str; 3] = [
    DEFAULT_WEBSOCKET_URL,
    "wss://stream.binance.com:443/ws/btcusdt@kline_1s",
    "wss://data-stream.binance.vision/ws/btcusdt@kline_1s",
];
const ALLOWED_REST_BASE_URLS: [&str; 2] = [DEFAULT_REST_BASE_URL, "https://api.binance.com"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct BinanceSpotOneSecondOhlcvConfig {
    pub websocket_url: String,
    pub rest_base_url: String,
    pub batch_size: usize,
    pub flush_interval_ms: u64,
    pub rest_page_limit: usize,
    pub recovery_overlap_seconds: i64,
    pub read_idle_timeout_ms: u64,
    pub reconnect_initial_delay_ms: u64,
    pub reconnect_max_delay_ms: u64,
    pub artifact_window_seconds: i64,
}

impl Default for BinanceSpotOneSecondOhlcvConfig {
    fn default() -> Self {
        Self {
            websocket_url: DEFAULT_WEBSOCKET_URL.to_owned(),
            rest_base_url: DEFAULT_REST_BASE_URL.to_owned(),
            batch_size: 250,
            flush_interval_ms: 250,
            rest_page_limit: 1_000,
            recovery_overlap_seconds: 60,
            read_idle_timeout_ms: 40_000,
            reconnect_initial_delay_ms: 1_000,
            reconnect_max_delay_ms: 30_000,
            artifact_window_seconds: 3_600,
        }
    }
}

impl BinanceSpotOneSecondOhlcvConfig {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let config = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::InvalidConfiguration(format!(
                "invalid Binance one-second OHLCV config: {error}"
            ))
        })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), StrategyFactoryError> {
        let invalid = |message: &str| {
            StrategyFactoryError::InvalidConfiguration(format!(
                "Binance one-second OHLCV config {message}"
            ))
        };
        if !ALLOWED_WEBSOCKET_URLS.contains(&self.websocket_url.as_str()) {
            return Err(invalid(
                "websocket_url is not an approved BTCUSDT kline endpoint",
            ));
        }
        if !ALLOWED_REST_BASE_URLS.contains(&self.rest_base_url.as_str()) {
            return Err(invalid(
                "rest_base_url is not an approved Binance HTTPS endpoint",
            ));
        }
        if !(1..=3_000).contains(&self.batch_size) {
            return Err(invalid("batch_size must be between 1 and 3000"));
        }
        if !(25..=10_000).contains(&self.flush_interval_ms) {
            return Err(invalid("flush_interval_ms must be between 25 and 10000"));
        }
        if !(1..=1_000).contains(&self.rest_page_limit) {
            return Err(invalid("rest_page_limit must be between 1 and 1000"));
        }
        if !(1..=86_400).contains(&self.recovery_overlap_seconds) {
            return Err(invalid(
                "recovery_overlap_seconds must be between 1 and 86400",
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
pub struct BinanceSpotOneSecondOhlcvFactory;

impl StrategyFactory for BinanceSpotOneSecondOhlcvFactory {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    fn config_schema_version(&self) -> i32 {
        CONFIG_SCHEMA_VERSION
    }

    fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError> {
        BinanceSpotOneSecondOhlcvConfig::from_value(config).map(|_| ())
    }

    fn build(
        &self,
        profile: &IngesterProfile,
        pool: PgPool,
    ) -> Result<Box<dyn RealtimeWorkerStrategy>, StrategyFactoryError> {
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

        let config = BinanceSpotOneSecondOhlcvConfig::from_value(&profile.config)?;
        let config_snapshot = serde_json::to_value(&config).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "failed to encode effective configuration: {error}"
            ))
        })?;
        let checkpoint = OhlcvCheckpoint::from_value(&profile.checkpoint)?;
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
        Ok(Box::new(BinanceSpotOneSecondOhlcvStrategy {
            config,
            config_snapshot,
            profile_generation: profile.desired_generation,
            lease_owner: Arc::<str>::from(lease_owner),
            lease_token,
            initial_last_open: checkpoint.last_open_timestamp,
            pool,
            client,
        }))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
struct OhlcvCheckpoint {
    last_open_timestamp: Option<DateTime<Utc>>,
}

impl OhlcvCheckpoint {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let checkpoint = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "invalid one-second OHLCV checkpoint: {error}"
            ))
        })?;
        if checkpoint
            .last_open_timestamp
            .is_some_and(|timestamp| timestamp.timestamp_subsec_nanos() != 0)
        {
            return Err(StrategyFactoryError::Construction(
                "one-second OHLCV checkpoint must align to a whole second".to_owned(),
            ));
        }
        Ok(checkpoint)
    }

    fn to_value(last_open_timestamp: DateTime<Utc>) -> Value {
        json!({ "last_open_timestamp": last_open_timestamp })
    }
}

pub struct BinanceSpotOneSecondOhlcvStrategy {
    config: BinanceSpotOneSecondOhlcvConfig,
    config_snapshot: Value,
    profile_generation: i64,
    lease_owner: Arc<str>,
    lease_token: Uuid,
    initial_last_open: Option<DateTime<Utc>>,
    pool: PgPool,
    client: Client,
}

#[derive(Debug)]
struct OhlcvRunState {
    last_open: Option<DateTime<Utc>>,
    artifact: Option<CaptureArtifact>,
}

struct SocketPump {
    shutdown: CancellationToken,
    task: JoinHandle<Result<(), StrategyError>>,
}

impl Drop for SocketPump {
    fn drop(&mut self) {
        self.shutdown.cancel();
        self.task.abort();
        crate::streaming::set_persistence_queue_depth(STRATEGY_KEY.as_str(), 0);
    }
}

async fn pump_websocket<S>(
    mut socket: WebSocketStream<S>,
    candle_tx: mpsc::Sender<OneSecondOhlcv>,
    shutdown: CancellationToken,
    read_idle_timeout: Duration,
) -> Result<(), StrategyError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let idle = tokio::time::sleep(read_idle_timeout);
    tokio::pin!(idle);
    loop {
        let next = tokio::select! {
            _ = shutdown.cancelled() => {
                let _ = socket.close(None).await;
                return Ok(());
            }
            _ = &mut idle => {
                return Err(source_error(
                    "binance_ohlcv_read_idle",
                    "Binance one-second kline websocket exceeded its read-idle timeout",
                ));
            }
            next = socket.next() => next,
        };
        idle.as_mut().reset(Instant::now() + read_idle_timeout);
        let message = match next {
            Some(Ok(message)) => message,
            Some(Err(error)) => {
                return Err(source_error(
                    "binance_ohlcv_websocket_read_failed",
                    format!("failed to read Binance one-second kline websocket: {error}"),
                ));
            }
            None => {
                return Err(source_error(
                    "binance_ohlcv_websocket_ended",
                    "Binance one-second kline websocket ended",
                ));
            }
        };
        match message {
            Message::Text(text) => {
                let received_at = Utc::now();
                let Some(candle) = OneSecondOhlcv::from_websocket(text.as_ref(), received_at)?
                else {
                    continue;
                };
                if candle_tx.capacity() == 0 {
                    crate::streaming::observe_persistence_queue_overflow(STRATEGY_KEY.as_str());
                    return Err(source_error(
                        "binance_ohlcv_persistence_backpressure",
                        "Binance one-second persistence queue reached capacity",
                    ));
                }
                crate::streaming::publish(
                    STRATEGY_KEY.as_str(),
                    candle.open_timestamp.timestamp_micros().to_string(),
                    candle.close_timestamp,
                    candle
                        .provider_available_at
                        .unwrap_or(candle.close_timestamp),
                    candle.received_at,
                    candle.payload_sha256.clone(),
                    true,
                    &candle,
                )
                .await;
                match candle_tx.try_send(candle) {
                    Ok(()) => crate::streaming::set_persistence_queue_depth(
                        STRATEGY_KEY.as_str(),
                        candle_tx.max_capacity() - candle_tx.capacity(),
                    ),
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        crate::streaming::observe_persistence_queue_overflow(STRATEGY_KEY.as_str());
                        return Err(source_error(
                            "binance_ohlcv_persistence_backpressure",
                            "Binance one-second persistence queue reached capacity",
                        ));
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => return Ok(()),
                }
            }
            Message::Ping(payload) => {
                crate::streaming::observe_websocket_ping(STRATEGY_KEY.as_str());
                let started = Instant::now();
                socket.send(Message::Pong(payload)).await.map_err(|error| {
                    source_error(
                        "binance_ohlcv_pong_failed",
                        format!("failed to answer Binance websocket ping: {error}"),
                    )
                })?;
                crate::streaming::observe_websocket_pong(STRATEGY_KEY.as_str(), started.elapsed());
            }
            Message::Pong(_) => {}
            Message::Close(frame) => {
                return Err(source_error(
                    "binance_ohlcv_websocket_closed",
                    format!("Binance one-second kline websocket closed: {frame:?}"),
                ));
            }
            Message::Binary(_) | Message::Frame(_) => {
                return Err(source_error(
                    "binance_ohlcv_unexpected_frame",
                    "received an unexpected Binance websocket frame",
                ));
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct OneSecondOhlcv {
    open_timestamp: DateTime<Utc>,
    close_timestamp: DateTime<Utc>,
    provider_available_at: Option<DateTime<Utc>>,
    received_at: DateTime<Utc>,
    open_price: Decimal,
    high_price: Decimal,
    low_price: Decimal,
    close_price: Decimal,
    base_volume: Decimal,
    quote_volume: Decimal,
    trade_count: i64,
    taker_buy_base_volume: Decimal,
    taker_buy_quote_volume: Decimal,
    payload_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireKlineEvent {
    #[serde(rename = "e")]
    event_type: String,
    #[serde(rename = "E")]
    event_time_ms: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "k")]
    kline: WireKline,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireKline {
    #[serde(rename = "t")]
    open_time_ms: i64,
    #[serde(rename = "T")]
    close_time_ms: i64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "i")]
    interval: String,
    #[serde(rename = "f")]
    _first_trade_id: i64,
    #[serde(rename = "L")]
    _last_trade_id: i64,
    #[serde(rename = "o")]
    open_price: String,
    #[serde(rename = "c")]
    close_price: String,
    #[serde(rename = "h")]
    high_price: String,
    #[serde(rename = "l")]
    low_price: String,
    #[serde(rename = "v")]
    base_volume: String,
    #[serde(rename = "n")]
    trade_count: i64,
    #[serde(rename = "x")]
    closed: bool,
    #[serde(rename = "q")]
    quote_volume: String,
    #[serde(rename = "V")]
    taker_buy_base_volume: String,
    #[serde(rename = "Q")]
    taker_buy_quote_volume: String,
    #[serde(rename = "B")]
    _ignore: String,
}

impl OneSecondOhlcv {
    fn from_websocket(
        text: &str,
        received_at: DateTime<Utc>,
    ) -> Result<Option<Self>, StrategyError> {
        let event = serde_json::from_str::<WireKlineEvent>(text).map_err(|error| {
            source_error(
                "binance_ohlcv_invalid_message",
                format!("failed to decode Binance one-second kline: {error}"),
            )
        })?;
        if event.event_type != "kline"
            || event.symbol != SYMBOL
            || event.kline.symbol != SYMBOL
            || event.kline.interval != INTERVAL
        {
            return Err(source_error(
                "binance_ohlcv_wrong_stream",
                "received a non-BTCUSDT one-second kline event",
            ));
        }
        validate_kline_window(event.kline.open_time_ms, event.kline.close_time_ms)?;
        if !event.kline.closed {
            return Ok(None);
        }
        let provider_available_at = timestamp_millis(event.event_time_ms, "event time")?;
        let close_timestamp = timestamp_millis(event.kline.close_time_ms, "close time")?;
        if provider_available_at < close_timestamp {
            return Err(source_error(
                "binance_ohlcv_invalid_time_order",
                "closed-kline event time precedes its close time",
            ));
        }
        Self::validated(
            event.kline.open_time_ms,
            event.kline.close_time_ms,
            Some(provider_available_at),
            received_at,
            &event.kline.open_price,
            &event.kline.high_price,
            &event.kline.low_price,
            &event.kline.close_price,
            &event.kline.base_volume,
            &event.kline.quote_volume,
            event.kline.trade_count,
            &event.kline.taker_buy_base_volume,
            &event.kline.taker_buy_quote_volume,
        )
        .map(Some)
    }

    fn from_rest(row: &[Value], received_at: DateTime<Utc>) -> Result<Self, StrategyError> {
        if row.len() != 12 {
            return Err(source_error(
                "binance_ohlcv_rest_wrong_shape",
                format!(
                    "Binance kline REST row has {} fields, expected 12",
                    row.len()
                ),
            ));
        }
        let open_time_ms = json_i64(&row[0], "open time")?;
        let close_time_ms = json_i64(&row[6], "close time")?;
        let _ignore = json_string(&row[11], "ignore")?;
        Self::validated(
            open_time_ms,
            close_time_ms,
            None,
            received_at,
            json_string(&row[1], "open price")?,
            json_string(&row[2], "high price")?,
            json_string(&row[3], "low price")?,
            json_string(&row[4], "close price")?,
            json_string(&row[5], "base volume")?,
            json_string(&row[7], "quote volume")?,
            json_i64(&row[8], "trade count")?,
            json_string(&row[9], "taker-buy base volume")?,
            json_string(&row[10], "taker-buy quote volume")?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn validated(
        open_time_ms: i64,
        close_time_ms: i64,
        provider_available_at: Option<DateTime<Utc>>,
        received_at: DateTime<Utc>,
        open_price: &str,
        high_price: &str,
        low_price: &str,
        close_price: &str,
        base_volume: &str,
        quote_volume: &str,
        trade_count: i64,
        taker_buy_base_volume: &str,
        taker_buy_quote_volume: &str,
    ) -> Result<Self, StrategyError> {
        validate_kline_window(open_time_ms, close_time_ms)?;
        let open_timestamp = timestamp_millis(open_time_ms, "open time")?;
        let close_timestamp = timestamp_millis(close_time_ms, "close time")?;
        if close_timestamp > received_at + MAX_PROVIDER_CLOCK_SKEW
            || provider_available_at
                .is_some_and(|available_at| available_at > received_at + MAX_PROVIDER_CLOCK_SKEW)
        {
            return Err(source_error(
                "binance_ohlcv_future_timestamp",
                "one-second kline provider timestamp exceeds the allowed clock skew",
            ));
        }
        let open_price = positive_decimal(open_price, "open price")?;
        let high_price = positive_decimal(high_price, "high price")?;
        let low_price = positive_decimal(low_price, "low price")?;
        let close_price = positive_decimal(close_price, "close price")?;
        if high_price < open_price
            || high_price < low_price
            || high_price < close_price
            || low_price > open_price
            || low_price > close_price
        {
            return Err(source_error(
                "binance_ohlcv_incoherent_prices",
                "Binance one-second kline OHLC prices are incoherent",
            ));
        }
        let base_volume = nonnegative_decimal(base_volume, "base volume")?;
        let quote_volume = nonnegative_decimal(quote_volume, "quote volume")?;
        let taker_buy_base_volume =
            nonnegative_decimal(taker_buy_base_volume, "taker-buy base volume")?;
        let taker_buy_quote_volume =
            nonnegative_decimal(taker_buy_quote_volume, "taker-buy quote volume")?;
        if trade_count < 0
            || taker_buy_base_volume > base_volume
            || taker_buy_quote_volume > quote_volume
        {
            return Err(source_error(
                "binance_ohlcv_incoherent_volume",
                "Binance one-second kline volume or trade count is incoherent",
            ));
        }
        let mut candle = Self {
            open_timestamp,
            close_timestamp,
            provider_available_at,
            received_at,
            open_price,
            high_price,
            low_price,
            close_price,
            base_volume,
            quote_volume,
            trade_count,
            taker_buy_base_volume,
            taker_buy_quote_volume,
            payload_sha256: String::new(),
        };
        candle.payload_sha256 = candle.factual_payload_sha256();
        Ok(candle)
    }

    fn factual_payload_sha256(&self) -> String {
        let canonical = format!(
            "v1|source={SOURCE}|symbol={SYMBOL}|open_timestamp_ms={}|close_timestamp_ms={}|open_price={}|high_price={}|low_price={}|close_price={}|base_volume={}|quote_volume={}|trade_count={}|taker_buy_base_volume={}|taker_buy_quote_volume={}",
            self.open_timestamp.timestamp_millis(),
            self.close_timestamp.timestamp_millis(),
            canonical_decimal(&self.open_price),
            canonical_decimal(&self.high_price),
            canonical_decimal(&self.low_price),
            canonical_decimal(&self.close_price),
            canonical_decimal(&self.base_volume),
            canonical_decimal(&self.quote_volume),
            self.trade_count,
            canonical_decimal(&self.taker_buy_base_volume),
            canonical_decimal(&self.taker_buy_quote_volume),
        );
        sha256_hex(canonical.as_bytes())
    }

    fn same_facts(&self, other: &Self) -> bool {
        self.open_timestamp == other.open_timestamp
            && self.close_timestamp == other.close_timestamp
            && self.open_price == other.open_price
            && self.high_price == other.high_price
            && self.low_price == other.low_price
            && self.close_price == other.close_price
            && self.base_volume == other.base_volume
            && self.quote_volume == other.quote_volume
            && self.trade_count == other.trade_count
            && self.taker_buy_base_volume == other.taker_buy_base_volume
            && self.taker_buy_quote_volume == other.taker_buy_quote_volume
            && self.payload_sha256 == other.payload_sha256
    }

    fn factual_eq(&self, stored: &StoredOhlcv) -> bool {
        self.open_timestamp == stored.open_timestamp
            && self.close_timestamp == stored.close_timestamp
            && self.open_price == stored.open_price
            && self.high_price == stored.high_price
            && self.low_price == stored.low_price
            && self.close_price == stored.close_price
            && self.base_volume == stored.base_volume
            && self.quote_volume == stored.quote_volume
            && self.trade_count == stored.trade_count
            && self.taker_buy_base_volume == stored.taker_buy_base_volume
            && self.taker_buy_quote_volume == stored.taker_buy_quote_volume
            && self.payload_sha256 == stored.payload_sha256
    }
}

#[derive(Debug, FromRow)]
struct StoredOhlcv {
    open_timestamp: DateTime<Utc>,
    close_timestamp: DateTime<Utc>,
    open_price: Decimal,
    high_price: Decimal,
    low_price: Decimal,
    close_price: Decimal,
    base_volume: Decimal,
    quote_volume: Decimal,
    trade_count: i64,
    taker_buy_base_volume: Decimal,
    taker_buy_quote_volume: Decimal,
    payload_sha256: String,
}

#[derive(Debug, FromRow)]
struct OhlcvArtifactChecksumRow {
    open_timestamp: DateTime<Utc>,
    payload_sha256: String,
}

struct OhlcvArtifactSeal {
    content_sha256: String,
    end_cursor: Option<String>,
}

#[async_trait]
impl RealtimeWorkerStrategy for BinanceSpotOneSecondOhlcvStrategy {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    async fn run(&self, shutdown: CancellationToken) -> Result<(), StrategyError> {
        let database_last_open = self.database_last_open().await?;
        if self.initial_last_open.is_some() && database_last_open.is_none() {
            return Err(integrity_error(
                "binance_ohlcv_checkpoint_without_fact",
                "OHLCV checkpoint exists without a durable source fact",
            ));
        }
        if let (Some(checkpoint), Some(database)) = (self.initial_last_open, database_last_open) {
            if checkpoint > database {
                return Err(integrity_error(
                    "binance_ohlcv_checkpoint_ahead",
                    format!("OHLCV checkpoint {checkpoint} is ahead of durable fact {database}"),
                ));
            }
        }
        let mut state = OhlcvRunState {
            last_open: database_last_open.or(self.initial_last_open),
            artifact: None,
        };
        let mut reconnect_delay = self.config.reconnect_initial_delay_ms;
        loop {
            if shutdown.is_cancelled() {
                self.seal_artifact(&mut state, true).await?;
                return Ok(());
            }
            let cursor_before_session = state.last_open;
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
                    crate::streaming::observe_source_reconnect(STRATEGY_KEY.as_str(), error.code);
                    if state.last_open != cursor_before_session {
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
                            database_error("binance_ohlcv_degraded_state_failed", database)
                        })?;
                    if !marked {
                        return self.finish_owned_drain(&mut state).await;
                    }
                    warn!(
                        strategy = %STRATEGY_KEY,
                        error_code = error.code,
                        error = %error,
                        reconnect_delay_ms = reconnect_delay,
                        "Binance one-second OHLCV session will reconnect"
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
                Err(error) if should_attempt_owned_drain(error.kind) => {
                    return self.finish_owned_drain(&mut state).await;
                }
                Err(error) => {
                    if let Err(seal_error) = self.seal_artifact(&mut state, false).await {
                        warn!(
                            strategy = %STRATEGY_KEY,
                            error = %seal_error,
                            "failed to seal OHLCV artifact after terminal failure"
                        );
                    }
                    return Err(error);
                }
            }
        }
    }
}

impl BinanceSpotOneSecondOhlcvStrategy {
    async fn database_last_open(&self) -> Result<Option<DateTime<Utc>>, StrategyError> {
        sqlx::query_scalar::<_, DateTime<Utc>>(
            r#"
            SELECT open_timestamp
            FROM market_data.binance_spot_btcusdt_one_second_ohlcv
            WHERE symbol = 'BTCUSDT'
            ORDER BY open_timestamp DESC
            LIMIT 1
            "#,
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| database_error("binance_ohlcv_cursor_read_failed", error))
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
        .map_err(|error| database_error("binance_ohlcv_lease_check_failed", error))?;
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
        state: &mut OhlcvRunState,
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
            .map_err(|error| database_error("binance_ohlcv_artifact_read_failed", error))?
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
                    "binance_ohlcv_artifact_config_conflict",
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
        let mut transaction =
            self.pool.begin().await.map_err(|error| {
                database_error("binance_ohlcv_artifact_transaction_failed", error)
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
                        .map(|timestamp| timestamp.timestamp_millis().to_string()),
                },
            )
            .await
            .map_err(|error| database_error("binance_ohlcv_artifact_create_failed", error))?;
        transaction
            .commit()
            .await
            .map_err(|error| database_error("binance_ohlcv_artifact_commit_failed", error))?;
        let artifact_id = artifact.artifact_id;
        state.artifact = Some(artifact);
        info!(
            strategy = %STRATEGY_KEY,
            %artifact_id,
            %window_start,
            %window_end,
            "opened one-second OHLCV capture artifact"
        );
        Ok(artifact_id)
    }

    async fn seal_artifact(
        &self,
        state: &mut OhlcvRunState,
        allow_draining_generation: bool,
    ) -> Result<(), StrategyError> {
        if let Some(artifact) = state.artifact.as_ref() {
            verify_artifact_generation(artifact.profile_generation, self.profile_generation)?;
        }
        let Some(artifact) = state.artifact.as_ref().cloned() else {
            if allow_draining_generation {
                let mut transaction = self.pool.begin().await.map_err(|error| {
                    database_error("binance_ohlcv_drain_transaction_failed", error)
                })?;
                self.assert_lease_in(&mut transaction, true).await?;
                transaction
                    .commit()
                    .await
                    .map_err(|error| database_error("binance_ohlcv_drain_commit_failed", error))?;
            }
            return Ok(());
        };
        let seal = self.artifact_seal(&artifact).await?;
        let mut transaction =
            self.pool.begin().await.map_err(|error| {
                database_error("binance_ohlcv_artifact_transaction_failed", error)
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
            .map_err(|error| database_error("binance_ohlcv_artifact_complete_failed", error))?;
        if completed.is_none() {
            return Err(integrity_error(
                "binance_ohlcv_artifact_not_open",
                format!(
                    "artifact {} was not open while sealing",
                    artifact.artifact_id
                ),
            ));
        }
        transaction
            .commit()
            .await
            .map_err(|error| database_error("binance_ohlcv_artifact_commit_failed", error))?;
        state.artifact = None;
        info!(
            strategy = %STRATEGY_KEY,
            artifact_id = %artifact.artifact_id,
            record_count = artifact.record_count,
            content_sha256 = %seal.content_sha256,
            "sealed one-second OHLCV capture artifact"
        );
        Ok(())
    }

    async fn finish_owned_drain(&self, state: &mut OhlcvRunState) -> Result<(), StrategyError> {
        self.seal_artifact(state, true).await?;
        info!(
            strategy = %STRATEGY_KEY,
            generation = self.profile_generation,
            "one-second OHLCV strategy drained after a desired-state lease race"
        );
        Ok(())
    }

    async fn artifact_seal(
        &self,
        artifact: &CaptureArtifact,
    ) -> Result<OhlcvArtifactSeal, StrategyError> {
        let rows = sqlx::query_as::<_, OhlcvArtifactChecksumRow>(
            r#"
            SELECT open_timestamp, payload_sha256::text AS payload_sha256
            FROM market_data.binance_spot_btcusdt_one_second_ohlcv
            WHERE capture_artifact_id = $1 AND strategy_key = $2
            ORDER BY open_timestamp
            "#,
        )
        .bind(artifact.artifact_id)
        .bind(STRATEGY_KEY.as_str())
        .fetch_all(&self.pool)
        .await
        .map_err(|error| database_error("binance_ohlcv_checksum_read_failed", error))?;
        if i64::try_from(rows.len()).ok() != Some(artifact.record_count) {
            return Err(integrity_error(
                "binance_ohlcv_artifact_count_mismatch",
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
            hash_field(
                &mut hasher,
                &row.open_timestamp.timestamp_millis().to_string(),
            );
            hash_field(&mut hasher, &row.payload_sha256);
        }
        Ok(OhlcvArtifactSeal {
            content_sha256: digest_hex(hasher.finalize()),
            end_cursor: rows
                .last()
                .map(|row| row.open_timestamp.timestamp_millis().to_string()),
        })
    }

    async fn complete_repair_artifact(
        &self,
        state: &mut OhlcvRunState,
        gap_id: Uuid,
    ) -> Result<(), StrategyError> {
        if let Some(artifact) = state.artifact.as_ref() {
            verify_artifact_generation(artifact.profile_generation, self.profile_generation)?;
        }
        let artifact = state.artifact.as_ref().cloned().ok_or_else(|| {
            integrity_error(
                "binance_ohlcv_repair_artifact_missing",
                "gap repair completed without an open capture artifact",
            )
        })?;
        let seal = self.artifact_seal(&artifact).await?;
        let mut transaction =
            self.pool.begin().await.map_err(|error| {
                database_error("binance_ohlcv_artifact_transaction_failed", error)
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
            .map_err(|error| database_error("binance_ohlcv_artifact_complete_failed", error))?;
        if completed.is_none() {
            return Err(integrity_error(
                "binance_ohlcv_artifact_not_open",
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
                "binance_rest_closed_klines_recovered",
                Some("all missing provider one-second klines were verified"),
            )
            .await
            .map_err(|error| database_error("binance_ohlcv_gap_complete_failed", error))?;
        if repaired.is_none() {
            return Err(integrity_error(
                "binance_ohlcv_gap_not_repairing",
                format!("data gap {gap_id} was not repairing during completion"),
            ));
        }
        transaction
            .commit()
            .await
            .map_err(|error| database_error("binance_ohlcv_artifact_commit_failed", error))?;
        state.artifact = None;
        info!(
            strategy = %STRATEGY_KEY,
            artifact_id = %artifact.artifact_id,
            record_count = artifact.record_count,
            content_sha256 = %seal.content_sha256,
            %gap_id,
            "sealed one-second OHLCV repair artifact and resolved gap"
        );
        Ok(())
    }

    async fn persist_candles(
        &self,
        state: &mut OhlcvRunState,
        candles: Vec<OneSecondOhlcv>,
    ) -> Result<(), StrategyError> {
        if candles.is_empty() {
            return Ok(());
        }
        let mut unique = BTreeMap::<DateTime<Utc>, OneSecondOhlcv>::new();
        for candle in candles {
            match unique.get_mut(&candle.open_timestamp) {
                Some(existing) if !existing.same_facts(&candle) => {
                    return Err(integrity_error(
                        "binance_ohlcv_batch_conflict",
                        format!(
                            "provider supplied conflicting payloads for kline {}",
                            candle.open_timestamp
                        ),
                    ));
                }
                Some(existing)
                    if existing.provider_available_at.is_none()
                        && candle.provider_available_at.is_some() =>
                {
                    *existing = candle;
                }
                Some(_) => {}
                None => {
                    unique.insert(candle.open_timestamp, candle);
                }
            }
        }
        let mut windows = BTreeMap::<DateTime<Utc>, Vec<OneSecondOhlcv>>::new();
        for candle in unique.into_values() {
            let (window_start, _) = self.artifact_window(candle.received_at);
            windows.entry(window_start).or_default().push(candle);
        }
        for candles in windows.into_values() {
            self.persist_artifact_batch(state, candles).await?;
        }
        Ok(())
    }

    async fn persist_artifact_batch(
        &self,
        state: &mut OhlcvRunState,
        mut candles: Vec<OneSecondOhlcv>,
    ) -> Result<(), StrategyError> {
        candles.sort_by_key(|candle| candle.open_timestamp);
        let artifact_id = self
            .ensure_artifact(
                state,
                candles.first().expect("non-empty OHLCV batch").received_at,
            )
            .await?;
        let timestamps: Vec<DateTime<Utc>> =
            candles.iter().map(|candle| candle.open_timestamp).collect();
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(|error| database_error("binance_ohlcv_transaction_begin_failed", error))?;
        let existing = self
            .load_existing_candles(&mut transaction, &timestamps)
            .await?;
        let mut existing_by_open = BTreeMap::<DateTime<Utc>, StoredOhlcv>::new();
        for stored in existing {
            if existing_by_open
                .insert(stored.open_timestamp, stored)
                .is_some()
            {
                return Err(integrity_error(
                    "binance_ohlcv_duplicate_identity",
                    "multiple durable rows share one BTCUSDT one-second open timestamp",
                ));
            }
        }
        let mut missing = Vec::new();
        for candle in &candles {
            if let Some(stored) = existing_by_open.get(&candle.open_timestamp) {
                if !candle.factual_eq(stored) {
                    return Err(integrity_error(
                        "binance_ohlcv_immutable_conflict",
                        format!(
                            "durable kline {} conflicts with provider payload",
                            candle.open_timestamp
                        ),
                    ));
                }
            } else {
                missing.push(candle);
            }
        }
        let inserted_timestamps = self
            .insert_missing_candles(&mut transaction, artifact_id, &missing)
            .await?;
        let durable = self
            .load_existing_candles(&mut transaction, &timestamps)
            .await?;
        let mut durable_by_open = BTreeMap::<DateTime<Utc>, StoredOhlcv>::new();
        for stored in durable {
            if durable_by_open
                .insert(stored.open_timestamp, stored)
                .is_some()
            {
                return Err(integrity_error(
                    "binance_ohlcv_duplicate_identity",
                    "multiple durable rows share one BTCUSDT one-second open timestamp",
                ));
            }
        }
        for candle in &candles {
            let Some(stored) = durable_by_open.get(&candle.open_timestamp) else {
                return Err(integrity_error(
                    "binance_ohlcv_insert_missing",
                    format!("kline {} was absent after insert", candle.open_timestamp),
                ));
            };
            if !candle.factual_eq(stored) {
                return Err(integrity_error(
                    "binance_ohlcv_immutable_conflict",
                    format!(
                        "durable kline {} conflicts with provider payload",
                        candle.open_timestamp
                    ),
                ));
            }
        }
        let inserted: Vec<&OneSecondOhlcv> = candles
            .iter()
            .filter(|candle| inserted_timestamps.contains(&candle.open_timestamp))
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
                            .map(|candle| candle.close_timestamp)
                            .min(),
                        maximum_source_timestamp: inserted
                            .iter()
                            .map(|candle| candle.close_timestamp)
                            .max(),
                        minimum_received_at: inserted.iter().map(|candle| candle.received_at).min(),
                        maximum_received_at: inserted.iter().map(|candle| candle.received_at).max(),
                        start_cursor: inserted
                            .first()
                            .map(|candle| candle.open_timestamp.timestamp_millis().to_string()),
                        end_cursor: inserted
                            .last()
                            .map(|candle| candle.open_timestamp.timestamp_millis().to_string()),
                    },
                )
                .await
                .map_err(|error| database_error("binance_ohlcv_artifact_progress_failed", error))?
                .ok_or_else(|| {
                    integrity_error(
                        "binance_ohlcv_artifact_not_open",
                        format!("artifact {artifact_id} was not open during fact insert"),
                    )
                })?;
            artifact_after_commit = Some(artifact);
        }
        let batch_last_open = candles
            .last()
            .expect("non-empty OHLCV batch")
            .open_timestamp;
        let checkpoint = state
            .last_open
            .map_or(batch_last_open, |current| current.max(batch_last_open));
        let last_source_timestamp = candles.iter().map(|candle| candle.close_timestamp).max();
        let last_provider_available_at = candles
            .iter()
            .filter_map(|candle| candle.provider_available_at)
            .max();
        let progressed = ProfileRepository::new(self.pool.clone())
            .record_progress_in(
                &mut transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &StrategyProgress {
                    verified_record_count: candles.len() as i64,
                    checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
                    checkpoint: OhlcvCheckpoint::to_value(checkpoint),
                    last_source_event_at: last_source_timestamp,
                    last_provider_available_at,
                    source_watermark: last_source_timestamp,
                    availability_watermark: last_provider_available_at,
                },
            )
            .await
            .map_err(|error| database_error("binance_ohlcv_profile_progress_failed", error))?;
        if !progressed {
            return Err(lease_lost_error());
        }
        transaction
            .commit()
            .await
            .map_err(|error| database_error("binance_ohlcv_transaction_commit_failed", error))?;
        if let Some(artifact) = artifact_after_commit {
            state.artifact = Some(artifact);
        }
        state.last_open = Some(checkpoint);
        Ok(())
    }

    async fn load_existing_candles(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        timestamps: &[DateTime<Utc>],
    ) -> Result<Vec<StoredOhlcv>, StrategyError> {
        sqlx::query_as::<_, StoredOhlcv>(
            r#"
            SELECT open_timestamp, close_timestamp,
                   open_price, high_price, low_price, close_price,
                   base_volume, quote_volume, trade_count,
                   taker_buy_base_volume, taker_buy_quote_volume,
                   payload_sha256::text AS payload_sha256
            FROM market_data.binance_spot_btcusdt_one_second_ohlcv
            WHERE symbol = 'BTCUSDT'
              AND open_timestamp = ANY($1::timestamptz[])
            ORDER BY open_timestamp
            "#,
        )
        .bind(timestamps)
        .fetch_all(&mut **transaction)
        .await
        .map_err(|error| database_error("binance_ohlcv_fact_read_failed", error))
    }

    async fn insert_missing_candles(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        missing: &[&OneSecondOhlcv],
    ) -> Result<BTreeSet<DateTime<Utc>>, StrategyError> {
        if missing.is_empty() {
            return Ok(BTreeSet::new());
        }
        let mut builder = QueryBuilder::<Postgres>::new(
            r#"
            INSERT INTO market_data.binance_spot_btcusdt_one_second_ohlcv (
              source, symbol, open_timestamp, close_timestamp,
              provider_available_at, received_at,
              open_price, high_price, low_price, close_price,
              base_volume, quote_volume, trade_count,
              taker_buy_base_volume, taker_buy_quote_volume,
              payload_sha256, strategy_key, capture_artifact_id
            )
            "#,
        );
        builder.push_values(missing, |mut row, candle| {
            row.push_bind(SOURCE)
                .push_bind(SYMBOL)
                .push_bind(candle.open_timestamp)
                .push_bind(candle.close_timestamp)
                .push_bind(candle.provider_available_at)
                .push_bind(candle.received_at)
                .push_bind(candle.open_price)
                .push_bind(candle.high_price)
                .push_bind(candle.low_price)
                .push_bind(candle.close_price)
                .push_bind(candle.base_volume)
                .push_bind(candle.quote_volume)
                .push_bind(candle.trade_count)
                .push_bind(candle.taker_buy_base_volume)
                .push_bind(candle.taker_buy_quote_volume)
                .push_bind(&candle.payload_sha256)
                .push_bind(STRATEGY_KEY.as_str())
                .push_bind(artifact_id);
        });
        builder.push(" ON CONFLICT (symbol, open_timestamp) DO NOTHING RETURNING open_timestamp");
        let inserted = builder
            .build_query_scalar::<DateTime<Utc>>()
            .fetch_all(&mut **transaction)
            .await
            .map_err(|error| database_error("binance_ohlcv_fact_insert_failed", error))?;
        Ok(inserted.into_iter().collect())
    }

    async fn capture_session(
        &self,
        state: &mut OhlcvRunState,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        self.recover_before_connect(state, shutdown).await?;
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
        let (socket, _) = connection
            .map_err(|_| {
                source_error(
                    "binance_ohlcv_connect_timeout",
                    "timed out connecting to Binance one-second kline websocket",
                )
            })?
            .map_err(|error| {
                source_error(
                    "binance_ohlcv_connect_failed",
                    format!("failed to connect to Binance one-second kline websocket: {error}"),
                )
            })?;
        info!(
            strategy = %STRATEGY_KEY,
            last_open_timestamp = ?state.last_open,
            "connected to Binance one-second kline websocket"
        );
        let (candle_tx, mut candle_rx) = mpsc::channel(PERSISTENCE_QUEUE_CAPACITY);
        let socket_shutdown = shutdown.child_token();
        let pump_shutdown = socket_shutdown.clone();
        let read_idle_timeout = Duration::from_millis(self.config.read_idle_timeout_ms);
        let socket_task = tokio::spawn(pump_websocket(
            socket,
            candle_tx,
            pump_shutdown,
            read_idle_timeout,
        ));
        let mut socket_pump = SocketPump {
            shutdown: socket_shutdown,
            task: socket_task,
        };
        let mut pending = Vec::<OneSecondOhlcv>::with_capacity(self.config.batch_size);
        let mut flush = tokio::time::interval(Duration::from_millis(self.config.flush_interval_ms));
        flush.set_missed_tick_behavior(MissedTickBehavior::Skip);
        flush.tick().await;

        loop {
            tokio::select! {
                _ = shutdown.cancelled() => {
                    let flush_result = self
                        .persist_candles_observed(state, std::mem::take(&mut pending))
                        .await;
                    if let Err(error) = flush_result {
                        if error.kind != StrategyErrorKind::LeaseLost {
                            return Err(error);
                        }
                    }
                    return Err(shutdown_error());
                }
                _ = flush.tick() => {
                    self.persist_candles_observed(state, std::mem::take(&mut pending)).await?;
                }
                Some(candle) = candle_rx.recv() => {
                    crate::streaming::set_persistence_queue_depth(
                        STRATEGY_KEY.as_str(),
                        candle_rx.len(),
                    );
                    self.accept_live_candle(state, &mut pending, candle, shutdown).await?;
                }
                pump_result = &mut socket_pump.task => {
                    while let Ok(candle) = candle_rx.try_recv() {
                        self.accept_live_candle(state, &mut pending, candle, shutdown).await?;
                    }
                    crate::streaming::set_persistence_queue_depth(STRATEGY_KEY.as_str(), 0);
                    self.persist_candles_observed(state, std::mem::take(&mut pending)).await?;
                    return match pump_result {
                        Ok(result) => result,
                        Err(error) => Err(source_error(
                            "binance_ohlcv_socket_task_failed",
                            format!("Binance one-second socket task failed: {error}"),
                        )),
                    };
                }
            }
        }
    }

    async fn accept_live_candle(
        &self,
        state: &mut OhlcvRunState,
        pending: &mut Vec<OneSecondOhlcv>,
        candle: OneSecondOhlcv,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        let last_seen = pending
            .last()
            .map(|pending| pending.open_timestamp)
            .or(state.last_open);
        if let Some(last_seen) = last_seen {
            let expected = last_seen + chrono::Duration::seconds(1);
            if candle.open_timestamp > expected {
                self.persist_candles_observed(state, std::mem::take(pending))
                    .await?;
                await_closed_boundary(
                    candle.open_timestamp - chrono::Duration::seconds(1),
                    shutdown,
                )
                .await?;
                self.repair_gap(
                    state,
                    expected,
                    candle.open_timestamp - chrono::Duration::seconds(1),
                    shutdown,
                )
                .await?;
            }
        }
        pending.push(candle);
        if pending.len() >= self.config.batch_size {
            self.persist_candles_observed(state, std::mem::take(pending))
                .await?;
        }
        Ok(())
    }

    async fn persist_candles_observed(
        &self,
        state: &mut OhlcvRunState,
        candles: Vec<OneSecondOhlcv>,
    ) -> Result<(), StrategyError> {
        if candles.is_empty() {
            return Ok(());
        }
        let started = Instant::now();
        let result = self.persist_candles(state, candles).await;
        crate::streaming::observe_persistence(STRATEGY_KEY.as_str(), started.elapsed());
        result
    }

    async fn recover_before_connect(
        &self,
        state: &mut OhlcvRunState,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        self.resume_unresolved_gaps(state, shutdown).await?;
        let last_closed = last_fully_closed_open(Utc::now())?;
        match state.last_open {
            None => {
                let start = last_closed
                    - chrono::Duration::seconds(
                        self.config.recovery_overlap_seconds.saturating_sub(1),
                    );
                self.recover_exact_range(state, start, last_closed, shutdown)
                    .await
            }
            Some(cursor) => {
                validate_closed_boundary(
                    cursor,
                    last_closed,
                    "binance_ohlcv_cursor_in_future",
                    "durable OHLCV cursor",
                )?;
                let overlap_start = cursor
                    - chrono::Duration::seconds(
                        self.config.recovery_overlap_seconds.saturating_sub(1),
                    );
                self.recover_exact_range(state, overlap_start, cursor, shutdown)
                    .await?;
                if cursor < last_closed {
                    self.repair_gap(
                        state,
                        cursor + chrono::Duration::seconds(1),
                        last_closed,
                        shutdown,
                    )
                    .await?;
                }
                Ok(())
            }
        }
    }

    async fn resume_unresolved_gaps(
        &self,
        state: &mut OhlcvRunState,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        let gaps = GapRepository::new(self.pool.clone())
            .list_unresolved_limited(STRATEGY_KEY, 1_000)
            .await
            .map_err(|error| database_error("binance_ohlcv_gap_resume_read_failed", error))?;
        for gap in gaps {
            if shutdown.is_cancelled() {
                return Err(shutdown_error());
            }
            if gap.reason_code != "binance_one_second_kline_gap" {
                return Err(integrity_error(
                    "binance_ohlcv_unknown_unresolved_gap",
                    format!(
                        "cannot resume unresolved OHLCV gap {} with reason {}",
                        gap.gap_id, gap.reason_code
                    ),
                ));
            }
            let start = gap.source_time_start.ok_or_else(|| {
                integrity_error(
                    "binance_ohlcv_gap_missing_range",
                    format!("unresolved OHLCV gap {} has no source start", gap.gap_id),
                )
            })?;
            let inclusive_end = gap.source_time_end.ok_or_else(|| {
                integrity_error(
                    "binance_ohlcv_gap_missing_range",
                    format!("unresolved OHLCV gap {} has no source end", gap.gap_id),
                )
            })?;
            let end_millis = inclusive_end.timestamp_millis().div_euclid(1_000) * 1_000;
            let end = Utc
                .timestamp_millis_opt(end_millis)
                .single()
                .ok_or_else(|| {
                    integrity_error(
                        "binance_ohlcv_gap_invalid_range",
                        format!(
                            "unresolved OHLCV gap {} has an invalid source end",
                            gap.gap_id
                        ),
                    )
                })?;
            if start.timestamp_subsec_nanos() != 0 || start > end {
                return Err(integrity_error(
                    "binance_ohlcv_gap_invalid_range",
                    format!("unresolved OHLCV gap {} is not second-aligned", gap.gap_id),
                ));
            }
            self.repair_gap(state, start, end, shutdown).await?;
        }
        Ok(())
    }

    async fn repair_gap(
        &self,
        state: &mut OhlcvRunState,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        if start > end {
            return Ok(());
        }
        let detected_artifact_id = self.ensure_artifact(state, Utc::now()).await?;
        let gap = NewDataGap {
            strategy_key: STRATEGY_KEY,
            detected_artifact_id: Some(detected_artifact_id),
            gap_kind: "source_time".to_owned(),
            reason_code: "binance_one_second_kline_gap".to_owned(),
            reason_message: Some(format!(
                "missing provider one-second klines from {start} through {end}"
            )),
            source_time_start: Some(start),
            source_time_end: Some(end + chrono::Duration::milliseconds(999)),
            start_cursor: Some(start.timestamp_millis().to_string()),
            end_cursor: Some(end.timestamp_millis().to_string()),
        };
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(|error| database_error("binance_ohlcv_gap_transaction_failed", error))?;
        self.assert_lease_in(&mut transaction, false).await?;
        let detection = GapRepository::new(self.pool.clone())
            .detect_in(&mut transaction, &gap)
            .await
            .map_err(|error| database_error("binance_ohlcv_gap_detect_failed", error))?;
        let degraded = ProfileRepository::new(self.pool.clone())
            .mark_degraded_in(
                &mut transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &StrategyDegradation {
                    reason_code: "binance_ohlcv_gap_repair".to_owned(),
                    reason_message: format!(
                        "repairing provider one-second klines from {start} through {end}"
                    ),
                },
            )
            .await
            .map_err(|error| database_error("binance_ohlcv_gap_health_failed", error))?;
        if !degraded {
            return Err(lease_lost_error());
        }
        let repairing_gap = GapRepository::new(self.pool.clone())
            .begin_repair_in(&mut transaction, detection.gap.gap_id)
            .await
            .map_err(|error| database_error("binance_ohlcv_gap_begin_failed", error))?
            .unwrap_or_else(|| detection.gap.clone());
        transaction.commit().await.map_err(|error| {
            database_error("binance_ohlcv_gap_transaction_commit_failed", error)
        })?;
        if detection.inserted {
            warn!(
                strategy = %STRATEGY_KEY,
                error_code = "binance_one_second_kline_gap",
                gap_id = %detection.gap.gap_id,
                start = %start,
                end = %end,
                "new Binance one-second candle gap detected"
            );
        }

        match self.recover_exact_range(state, start, end, shutdown).await {
            Ok(()) => {
                self.complete_repair_artifact(state, detection.gap.gap_id)
                    .await?;
                Ok(())
            }
            Err(error)
                if error.code == "binance_ohlcv_range_unavailable"
                    && repairing_gap.repair_attempts >= 3 =>
            {
                let mut transaction = self.pool.begin().await.map_err(|database| {
                    database_error("binance_ohlcv_gap_terminal_transaction_failed", database)
                })?;
                self.assert_lease_in(&mut transaction, false).await?;
                let terminal = GapRepository::new(self.pool.clone())
                    .mark_unrecoverable_in(
                        &mut transaction,
                        detection.gap.gap_id,
                        "binance_rest_closed_kline_unavailable_after_retries",
                        Some("Binance REST omitted a closed kline on three repair attempts"),
                    )
                    .await
                    .map_err(|database| {
                        database_error("binance_ohlcv_gap_unrecoverable_failed", database)
                    })?;
                if terminal.is_none() {
                    return Err(integrity_error(
                        "binance_ohlcv_gap_not_repairing",
                        format!(
                            "data gap {} was not open during terminalization",
                            detection.gap.gap_id
                        ),
                    ));
                }
                transaction.commit().await.map_err(|database| {
                    database_error("binance_ohlcv_gap_terminal_commit_failed", database)
                })?;
                warn!(
                    strategy = %STRATEGY_KEY,
                    error_code = "binance_ohlcv_gap_unrecoverable",
                    gap_id = %detection.gap.gap_id,
                    start = %start,
                    end = %end,
                    repair_attempts = repairing_gap.repair_attempts,
                    "Binance one-second candle gap repair became terminal"
                );
                Err(integrity_error(
                    "binance_ohlcv_gap_unrecoverable",
                    format!(
                        "OHLCV gap {start} through {end} remained unavailable after {} attempts",
                        repairing_gap.repair_attempts
                    ),
                ))
            }
            Err(error) => Err(error),
        }
    }

    async fn recover_exact_range(
        &self,
        state: &mut OhlcvRunState,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        if start > end {
            return Ok(());
        }
        let mut expected = start;
        while expected <= end {
            let remaining_seconds = (end - expected).num_seconds().saturating_add(1);
            let limit = usize::try_from(remaining_seconds)
                .unwrap_or(self.config.rest_page_limit)
                .min(self.config.rest_page_limit);
            let page = self.fetch_rest_page(expected, end, limit, shutdown).await?;
            if page.is_empty()
                || page
                    .first()
                    .is_none_or(|candle| candle.open_timestamp != expected)
            {
                return Err(source_error(
                    "binance_ohlcv_range_unavailable",
                    format!("Binance REST did not return closed kline {expected}"),
                ));
            }
            validate_contiguous_klines(&page)?;
            let last = page
                .last()
                .expect("non-empty exact OHLCV recovery page")
                .open_timestamp;
            if last > end {
                return Err(source_error(
                    "binance_ohlcv_rest_outside_range",
                    "Binance REST returned a kline beyond the requested closed range",
                ));
            }
            let page = self
                .stabilize_rest_boundary_page(expected, end, limit, page, shutdown)
                .await?;
            self.persist_candles(state, page).await?;
            if last >= end {
                break;
            }
            expected = last + chrono::Duration::seconds(1);
        }
        Ok(())
    }

    async fn stabilize_rest_boundary_page(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: usize,
        page: Vec<OneSecondOhlcv>,
        shutdown: &CancellationToken,
    ) -> Result<Vec<OneSecondOhlcv>, StrategyError> {
        let Some(wait) = rest_boundary_stabilization_wait(&page, Utc::now())? else {
            return Ok(page);
        };
        tokio::select! {
            _ = shutdown.cancelled() => return Err(shutdown_error()),
            _ = tokio::time::sleep(wait) => {}
        }
        let confirmed = self.fetch_rest_page(start, end, limit, shutdown).await?;
        validate_stable_rest_page(&page, &confirmed)?;
        Ok(confirmed)
    }

    async fn fetch_rest_page(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: usize,
        shutdown: &CancellationToken,
    ) -> Result<Vec<OneSecondOhlcv>, StrategyError> {
        let last_closed = last_fully_closed_open(Utc::now())?;
        validate_closed_boundary(
            end,
            last_closed,
            "binance_ohlcv_open_rest_range",
            "REST recovery range end",
        )?;
        let url = format!(
            "{}/api/v3/klines",
            self.config.rest_base_url.trim_end_matches('/')
        );
        let parameters = [
            ("symbol", SYMBOL.to_owned()),
            ("interval", INTERVAL.to_owned()),
            ("startTime", start.timestamp_millis().to_string()),
            ("endTime", end.timestamp_millis().to_string()),
            ("limit", limit.to_string()),
        ];
        let request = self.client.get(url).query(&parameters).send();
        let response = tokio::select! {
            _ = shutdown.cancelled() => return Err(shutdown_error()),
            response = request => response,
        }
        .map_err(|error| {
            source_error(
                "binance_ohlcv_rest_request_failed",
                format!("Binance one-second kline REST request failed: {error}"),
            )
        })?;
        if !response.status().is_success() {
            return Err(source_error(
                "binance_ohlcv_rest_status",
                format!(
                    "Binance one-second kline REST returned HTTP {}",
                    response.status()
                ),
            ));
        }
        let bytes = read_bounded_rest_body(response, shutdown).await?;
        let rows = serde_json::from_slice::<Vec<Vec<Value>>>(&bytes).map_err(|error| {
            source_error(
                "binance_ohlcv_rest_invalid_body",
                format!("failed to decode Binance one-second kline REST body: {error}"),
            )
        })?;
        if rows.len() > limit {
            return Err(source_error(
                "binance_ohlcv_rest_limit_exceeded",
                format!(
                    "Binance one-second kline REST returned {} rows for limit {limit}",
                    rows.len()
                ),
            ));
        }
        let received_at = Utc::now();
        rows.iter()
            .map(|row| OneSecondOhlcv::from_rest(row, received_at))
            .collect()
    }
}

fn validate_kline_window(open_time_ms: i64, close_time_ms: i64) -> Result<(), StrategyError> {
    if open_time_ms < 0
        || open_time_ms.rem_euclid(1_000) != 0
        || close_time_ms != open_time_ms.saturating_add(999)
    {
        return Err(source_error(
            "binance_ohlcv_invalid_window",
            "Binance one-second kline is not aligned to a closed 1000ms window",
        ));
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
                "binance_ohlcv_rest_body_failed",
                format!("failed to stream Binance one-second kline REST body: {error}"),
            )
        })?;
        let next_len = body.len().checked_add(chunk.len()).ok_or_else(|| {
            source_error(
                "binance_ohlcv_rest_body_too_large",
                "Binance one-second kline REST body length overflowed",
            )
        })?;
        if next_len > MAX_REST_BODY_BYTES {
            return Err(source_error(
                "binance_ohlcv_rest_body_too_large",
                format!("Binance one-second kline REST body exceeds {MAX_REST_BODY_BYTES} bytes"),
            ));
        }
        body.extend_from_slice(&chunk);
    }
}

fn validate_contiguous_klines(page: &[OneSecondOhlcv]) -> Result<(), StrategyError> {
    for pair in page.windows(2) {
        if pair[1].open_timestamp != pair[0].open_timestamp + chrono::Duration::seconds(1) {
            return Err(source_error(
                "binance_ohlcv_range_unavailable",
                format!(
                    "Binance REST omitted closed kline {}",
                    pair[0].open_timestamp + chrono::Duration::seconds(1)
                ),
            ));
        }
    }
    Ok(())
}

fn rest_boundary_stabilization_wait(
    page: &[OneSecondOhlcv],
    now: DateTime<Utc>,
) -> Result<Option<Duration>, StrategyError> {
    let Some(last) = page.last() else {
        return Ok(None);
    };
    let stable_at = last.close_timestamp
        + chrono::Duration::from_std(REST_BOUNDARY_STABILIZATION_DELAY).map_err(|_| {
            integrity_error(
                "binance_ohlcv_invalid_stabilization_delay",
                "REST boundary stabilization delay is out of range",
            )
        })?;
    if stable_at <= now {
        return Ok(None);
    }
    Ok(Some((stable_at - now).to_std().map_err(|_| {
        integrity_error(
            "binance_ohlcv_invalid_stabilization_wait",
            "REST boundary stabilization wait is negative or out of range",
        )
    })?))
}

fn validate_stable_rest_page(
    initial: &[OneSecondOhlcv],
    confirmed: &[OneSecondOhlcv],
) -> Result<(), StrategyError> {
    if initial.len() != confirmed.len()
        || initial.iter().zip(confirmed).any(|(left, right)| {
            left.open_timestamp != right.open_timestamp || !left.same_facts(right)
        })
    {
        return Err(source_error(
            "binance_ohlcv_rest_boundary_unstable",
            "Binance REST returned changing one-second kline facts near the live boundary",
        ));
    }
    Ok(())
}

fn last_fully_closed_open(now: DateTime<Utc>) -> Result<DateTime<Utc>, StrategyError> {
    let current_second = now
        .timestamp_millis()
        .div_euclid(1_000)
        .saturating_mul(1_000);
    timestamp_millis(
        current_second.saturating_sub(1_000),
        "last fully closed open time",
    )
}

fn validate_closed_boundary(
    requested: DateTime<Utc>,
    last_closed: DateTime<Utc>,
    error_code: &'static str,
    subject: &'static str,
) -> Result<(), StrategyError> {
    if requested <= last_closed {
        return Ok(());
    }
    let message =
        format!("{subject} {requested} is newer than last fully closed second {last_closed}");
    if requested <= last_closed + MAX_PROVIDER_CLOCK_SKEW {
        return Err(source_error(error_code, message));
    }
    Err(integrity_error(error_code, message))
}

async fn await_closed_boundary(
    requested: DateTime<Utc>,
    shutdown: &CancellationToken,
) -> Result<(), StrategyError> {
    let Some(wait) = closed_boundary_wait(requested, Utc::now())? else {
        return Ok(());
    };
    tokio::select! {
        _ = shutdown.cancelled() => Err(shutdown_error()),
        _ = tokio::time::sleep(wait) => Ok(()),
    }
}

fn closed_boundary_wait(
    requested: DateTime<Utc>,
    now: DateTime<Utc>,
) -> Result<Option<Duration>, StrategyError> {
    let last_closed = last_fully_closed_open(now)?;
    if requested <= last_closed {
        return Ok(None);
    }
    if requested > last_closed + MAX_PROVIDER_CLOCK_SKEW {
        return Err(integrity_error(
            "binance_ohlcv_live_gap_clock_lead",
            format!(
                "live gap recovery range end {requested} is newer than last fully closed second {last_closed}"
            ),
        ));
    }
    let close_at = requested + chrono::Duration::seconds(1);
    let wait = (close_at - now)
        .to_std()
        .map_err(|_| {
            integrity_error(
                "binance_ohlcv_invalid_close_boundary_wait",
                "live gap close-boundary wait is negative or out of range",
            )
        })?
        .saturating_add(Duration::from_millis(25));
    Ok(Some(wait))
}

fn timestamp_millis(value: i64, field: &str) -> Result<DateTime<Utc>, StrategyError> {
    if value < 0 {
        return Err(source_error(
            "binance_ohlcv_invalid_timestamp",
            format!("Binance one-second kline {field} is negative"),
        ));
    }
    Utc.timestamp_millis_opt(value).single().ok_or_else(|| {
        source_error(
            "binance_ohlcv_invalid_timestamp",
            format!("Binance one-second kline {field} is out of range"),
        )
    })
}

fn json_i64(value: &Value, field: &str) -> Result<i64, StrategyError> {
    value.as_i64().ok_or_else(|| {
        source_error(
            "binance_ohlcv_rest_wrong_type",
            format!("Binance kline REST {field} is not an integer"),
        )
    })
}

fn json_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, StrategyError> {
    value.as_str().ok_or_else(|| {
        source_error(
            "binance_ohlcv_rest_wrong_type",
            format!("Binance kline REST {field} is not a string"),
        )
    })
}

fn positive_decimal(value: &str, field: &str) -> Result<Decimal, StrategyError> {
    let parsed = decimal(value, field)?;
    if parsed <= Decimal::ZERO {
        return Err(source_error(
            "binance_ohlcv_nonpositive_decimal",
            format!("Binance one-second kline {field} must be positive"),
        ));
    }
    Ok(parsed)
}

fn nonnegative_decimal(value: &str, field: &str) -> Result<Decimal, StrategyError> {
    let parsed = decimal(value, field)?;
    if parsed < Decimal::ZERO {
        return Err(source_error(
            "binance_ohlcv_negative_decimal",
            format!("Binance one-second kline {field} cannot be negative"),
        ));
    }
    Ok(parsed)
}

fn decimal(value: &str, field: &str) -> Result<Decimal, StrategyError> {
    validate_decimal_wire(value, field)?;
    let parsed = value.parse::<Decimal>().map_err(|error| {
        source_error(
            "binance_ohlcv_invalid_decimal",
            format!("Binance one-second kline {field} is invalid: {error}"),
        )
    })?;
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
            "binance_ohlcv_invalid_decimal",
            format!("Binance one-second kline {field} is not a plain unsigned decimal"),
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
            "binance_ohlcv_decimal_out_of_range",
            format!("Binance one-second kline {field} exceeds numeric(30,10)"),
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
        "binance_ohlcv_lease_lost",
        "one-second OHLCV profile lease was lost before progress committed",
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

fn should_attempt_owned_drain(kind: StrategyErrorKind) -> bool {
    kind == StrategyErrorKind::LeaseLost
}

fn shutdown_error() -> StrategyError {
    StrategyError::new(
        StrategyErrorKind::Shutdown,
        "binance_ohlcv_shutdown",
        "one-second OHLCV strategy shutdown requested",
    )
}

#[cfg(test)]
mod tests {
    use tokio_tungstenite::tungstenite::protocol::Role;

    use super::*;

    const CLOSED_WEBSOCKET_FIXTURE: &str =
        include_str!("../../../tests/fixtures/binance/one_second_ohlcv_websocket_closed_v1.json");
    const OPEN_WEBSOCKET_FIXTURE: &str =
        include_str!("../../../tests/fixtures/binance/one_second_ohlcv_websocket_open_v1.json");
    const REST_FIXTURE: &str =
        include_str!("../../../tests/fixtures/binance/one_second_ohlcv_rest_v1.json");

    #[tokio::test]
    async fn websocket_pump_answers_ping_while_persistence_is_not_draining() {
        let (client_io, server_io) = tokio::io::duplex(16 * 1024);
        let client = WebSocketStream::from_raw_socket(client_io, Role::Client, None).await;
        let mut server = WebSocketStream::from_raw_socket(server_io, Role::Server, None).await;
        let (candle_tx, _candle_rx) = mpsc::channel(1);
        let shutdown = CancellationToken::new();
        let pump = tokio::spawn(pump_websocket(
            client,
            candle_tx,
            shutdown.clone(),
            Duration::from_secs(5),
        ));

        server
            .send(Message::Text(CLOSED_WEBSOCKET_FIXTURE.into()))
            .await
            .expect("send closed candle");
        server
            .send(Message::Ping(vec![1, 2, 3, 4].into()))
            .await
            .expect("send ping");

        let pong = tokio::time::timeout(Duration::from_millis(250), async {
            loop {
                match server.next().await {
                    Some(Ok(Message::Pong(payload))) => break payload,
                    Some(Ok(_)) => continue,
                    Some(Err(error)) => panic!("server websocket failed: {error}"),
                    None => panic!("server websocket ended before pong"),
                }
            }
        })
        .await
        .expect("socket pump must not wait for persistence before answering ping");
        assert_eq!(pong.as_ref(), &[1, 2, 3, 4]);

        shutdown.cancel();
        pump.await
            .expect("socket pump task")
            .expect("socket pump shutdown");
    }

    #[test]
    fn default_config_matches_the_bootstrap_profile() {
        let config = BinanceSpotOneSecondOhlcvConfig::default();
        assert_eq!(config.websocket_url, DEFAULT_WEBSOCKET_URL);
        assert_eq!(config.rest_base_url, DEFAULT_REST_BASE_URL);
        assert_eq!(config.batch_size, 250);
        assert_eq!(config.flush_interval_ms, 250);
        assert_eq!(config.rest_page_limit, 1_000);
        assert_eq!(config.recovery_overlap_seconds, 60);
        assert_eq!(config.read_idle_timeout_ms, 40_000);
        assert_eq!(config.reconnect_initial_delay_ms, 1_000);
        assert_eq!(config.reconnect_max_delay_ms, 30_000);
        assert_eq!(config.artifact_window_seconds, 3_600);
        config.validate().expect("default config is valid");
    }

    #[test]
    fn config_rejects_unknown_fields_and_unapproved_endpoints() {
        assert!(BinanceSpotOneSecondOhlcvConfig::from_value(&json!({
            "unexpected": true
        }))
        .is_err());
        let mut config = serde_json::to_value(BinanceSpotOneSecondOhlcvConfig::default())
            .expect("serialize config");
        config["websocket_url"] = json!("wss://example.invalid/ws/btcusdt@kline_1s");
        assert!(BinanceSpotOneSecondOhlcvConfig::from_value(&config).is_err());
    }

    #[test]
    fn closed_websocket_fixture_decodes_provider_candle() {
        let received_at = Utc.timestamp_millis_opt(1_722_470_401_010).unwrap();
        let candle = OneSecondOhlcv::from_websocket(CLOSED_WEBSOCKET_FIXTURE, received_at)
            .expect("valid closed kline")
            .expect("closed kline is emitted");
        assert_eq!(
            candle.open_timestamp,
            Utc.timestamp_millis_opt(1_722_470_400_000).unwrap()
        );
        assert_eq!(
            candle.close_timestamp,
            Utc.timestamp_millis_opt(1_722_470_400_999).unwrap()
        );
        assert_eq!(candle.trade_count, 5);
        assert_eq!(candle.payload_sha256.len(), 64);
    }

    #[test]
    fn open_websocket_fixture_is_not_a_fact() {
        let received_at = Utc.timestamp_millis_opt(1_722_470_400_510).unwrap();
        let candle = OneSecondOhlcv::from_websocket(OPEN_WEBSOCKET_FIXTURE, received_at)
            .expect("valid open kline update");
        assert!(candle.is_none());
    }

    #[test]
    fn rest_and_websocket_have_identical_factual_hashes() {
        let received_at = Utc.timestamp_millis_opt(1_722_470_401_010).unwrap();
        let websocket = OneSecondOhlcv::from_websocket(CLOSED_WEBSOCKET_FIXTURE, received_at)
            .expect("valid closed kline")
            .expect("closed kline");
        let rows =
            serde_json::from_str::<Vec<Vec<Value>>>(REST_FIXTURE).expect("valid REST fixture");
        let rest = OneSecondOhlcv::from_rest(&rows[0], received_at + chrono::Duration::seconds(1))
            .expect("valid REST kline");
        assert!(websocket.same_facts(&rest));
        assert_eq!(websocket.payload_sha256, rest.payload_sha256);
        assert!(rest.provider_available_at.is_none());
    }

    #[test]
    fn websocket_wire_shape_is_strict() {
        let mut value =
            serde_json::from_str::<Value>(CLOSED_WEBSOCKET_FIXTURE).expect("fixture JSON");
        value["unexpected"] = json!(1);
        let received_at = Utc.timestamp_millis_opt(1_722_470_401_010).unwrap();
        let error = OneSecondOhlcv::from_websocket(&value.to_string(), received_at)
            .expect_err("unknown provider field must fail");
        assert_eq!(error.code, "binance_ohlcv_invalid_message");
    }

    #[test]
    fn rest_fixture_is_contiguous_and_closed() {
        let rows =
            serde_json::from_str::<Vec<Vec<Value>>>(REST_FIXTURE).expect("valid REST fixture");
        let received_at = Utc.timestamp_millis_opt(1_722_470_402_010).unwrap();
        let candles: Vec<_> = rows
            .iter()
            .map(|row| OneSecondOhlcv::from_rest(row, received_at).expect("valid REST row"))
            .collect();
        validate_contiguous_klines(&candles).expect("contiguous provider rows");
        assert_eq!(
            candles[1].open_timestamp - candles[0].open_timestamp,
            chrono::Duration::seconds(1)
        );
    }

    #[test]
    fn bounded_close_boundary_clock_lead_is_transient() {
        let last_closed = Utc.timestamp_millis_opt(1_722_470_400_000).unwrap();
        let error = validate_closed_boundary(
            last_closed + chrono::Duration::seconds(2),
            last_closed,
            "binance_ohlcv_open_rest_range",
            "REST recovery range end",
        )
        .expect_err("bounded provider clock lead must wait for its close boundary");

        assert_eq!(error.kind, StrategyErrorKind::TransientSource);
        assert_eq!(error.code, "binance_ohlcv_open_rest_range");
    }

    #[test]
    fn excessive_close_boundary_clock_lead_remains_an_integrity_failure() {
        let last_closed = Utc.timestamp_millis_opt(1_722_470_400_000).unwrap();
        let error = validate_closed_boundary(
            last_closed + MAX_PROVIDER_CLOCK_SKEW + chrono::Duration::seconds(1),
            last_closed,
            "binance_ohlcv_cursor_in_future",
            "durable OHLCV cursor",
        )
        .expect_err("clock lead beyond the provider contract must fail closed");

        assert_eq!(error.kind, StrategyErrorKind::Integrity);
        assert_eq!(error.code, "binance_ohlcv_cursor_in_future");
    }

    #[test]
    fn live_gap_waits_until_the_requested_second_is_closed() {
        let now = Utc.timestamp_millis_opt(1_722_470_400_800).unwrap();
        let requested = Utc.timestamp_millis_opt(1_722_470_401_000).unwrap();

        assert_eq!(
            closed_boundary_wait(requested, now).expect("bounded clock lead"),
            Some(Duration::from_millis(1_225))
        );
    }

    #[test]
    fn live_gap_does_not_wait_for_an_already_closed_second() {
        let now = Utc.timestamp_millis_opt(1_722_470_402_100).unwrap();
        let requested = Utc.timestamp_millis_opt(1_722_470_401_000).unwrap();

        assert_eq!(
            closed_boundary_wait(requested, now).expect("closed boundary"),
            None
        );
    }

    #[test]
    fn near_boundary_rest_page_waits_until_the_confirmation_boundary() {
        let rows =
            serde_json::from_str::<Vec<Vec<Value>>>(REST_FIXTURE).expect("valid REST fixture");
        let received_at = Utc.timestamp_millis_opt(1_722_470_402_010).unwrap();
        let page: Vec<_> = rows
            .iter()
            .map(|row| OneSecondOhlcv::from_rest(row, received_at).expect("valid REST row"))
            .collect();
        let now = page.last().unwrap().close_timestamp + chrono::Duration::seconds(4);

        assert_eq!(
            rest_boundary_stabilization_wait(&page, now).expect("valid wait"),
            Some(Duration::from_secs(6))
        );
    }

    #[test]
    fn historical_rest_page_does_not_wait_for_confirmation() {
        let rows =
            serde_json::from_str::<Vec<Vec<Value>>>(REST_FIXTURE).expect("valid REST fixture");
        let received_at = Utc.timestamp_millis_opt(1_722_470_402_010).unwrap();
        let page: Vec<_> = rows
            .iter()
            .map(|row| OneSecondOhlcv::from_rest(row, received_at).expect("valid REST row"))
            .collect();
        let now = page.last().unwrap().close_timestamp + chrono::Duration::seconds(10);

        assert_eq!(
            rest_boundary_stabilization_wait(&page, now).expect("valid wait"),
            None
        );
    }

    #[test]
    fn changing_boundary_payload_is_transient_and_not_accepted() {
        let rows =
            serde_json::from_str::<Vec<Vec<Value>>>(REST_FIXTURE).expect("valid REST fixture");
        let received_at = Utc.timestamp_millis_opt(1_722_470_402_010).unwrap();
        let initial: Vec<_> = rows
            .iter()
            .map(|row| OneSecondOhlcv::from_rest(row, received_at).expect("valid REST row"))
            .collect();
        let mut confirmed = initial.clone();
        confirmed.last_mut().unwrap().close_price += Decimal::new(1, 8);

        let error = validate_stable_rest_page(&initial, &confirmed)
            .expect_err("changing near-boundary facts must retry");
        assert_eq!(error.kind, StrategyErrorKind::TransientSource);
        assert_eq!(error.code, "binance_ohlcv_rest_boundary_unstable");
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

    #[test]
    fn only_strict_lease_loss_routes_to_owned_drain() {
        assert!(should_attempt_owned_drain(StrategyErrorKind::LeaseLost));
        assert!(!should_attempt_owned_drain(
            StrategyErrorKind::TransientDatabase
        ));
        assert!(!should_attempt_owned_drain(StrategyErrorKind::Integrity));
    }
}
