use std::{collections::BTreeMap, fmt, str::FromStr};

use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub const BACKFILL_REQUEST_VERSION: i32 = 1;
pub const MAX_IDEMPOTENCY_KEY_BYTES: usize = 255;
pub const BINANCE_L2_HISTORICAL_START_EPOCH: i64 = 1_776_124_800;
pub const BINANCE_L2_HISTORICAL_END_EPOCH: i64 = 1_785_628_800;
pub const BINANCE_SPOT_L2_HISTORICAL_START_EPOCH: i64 = BINANCE_L2_HISTORICAL_START_EPOCH;
pub const BINANCE_SPOT_L2_HISTORICAL_END_EPOCH: i64 = BINANCE_L2_HISTORICAL_END_EPOCH;
pub const PMDATA_TWAP_HISTORICAL_START_EPOCH: i64 = 1_785_542_400;
pub const PMDATA_REFPRICE_HISTORICAL_START_EPOCH: i64 = 1_780_790_400;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BtcOutcome {
    Up,
    Down,
}

impl BtcOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BtcIntervalMarket {
    pub event_id: String,
    pub event_slug: String,
    pub series_slug: String,
    pub market_id: String,
    pub condition_id: String,
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    pub up_token_id: String,
    pub down_token_id: String,
    pub tick_size: Decimal,
    pub minimum_order_size: Option<Decimal>,
    pub resolution_source: String,
    pub active: bool,
    pub closed: bool,
    pub accepting_orders: bool,
    pub fees_enabled: bool,
    pub fee_schedule: Value,
    pub raw_payload: Value,
}

impl BtcIntervalMarket {
    pub fn token_id(&self, outcome: BtcOutcome) -> &str {
        match outcome {
            BtcOutcome::Up => &self.up_token_id,
            BtcOutcome::Down => &self.down_token_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IngesterKey {
    BtcFiveMinuteMarkets,
    BtcFiveMinuteResolutions,
    BinanceBtcusdtAggTrades,
    BinanceBtcusdtL2OneSecondFeatures,
    BinanceSpotBtcusdtL2OneSecondFeatures,
    BinanceBtcusdtOneSecondKlines,
    KrakenSpotBtcusdTradePrintsOneSecondOhlcv,
    PolymarketBtcFiveMinuteOrderbooks,
    PolymarketBtcFiveMinuteExecutionSnapshots,
    ChainlinkBtcusdReferenceTicks,
    ChainlinkBtcusdOneMinuteCandles,
    BinanceBtcusdtFiveMinuteOpenInterest,
    PolygonChainlinkBtcusdOracleRounds,
    #[serde(rename = "pmdata_chainlink_btcusd_refprice")]
    PmdataChainlinkBtcusdRefprice,
    #[serde(rename = "pmdata_chainlink_btcusd_twap_30s")]
    PmdataChainlinkBtcusdTwap30s,
    #[serde(rename = "pmdata_chainlink_btcusd_twap_60s")]
    PmdataChainlinkBtcusdTwap60s,
}

impl IngesterKey {
    pub const ALL: [Self; 16] = [
        Self::BtcFiveMinuteMarkets,
        Self::BtcFiveMinuteResolutions,
        Self::BinanceBtcusdtAggTrades,
        Self::BinanceBtcusdtL2OneSecondFeatures,
        Self::BinanceSpotBtcusdtL2OneSecondFeatures,
        Self::BinanceBtcusdtOneSecondKlines,
        Self::KrakenSpotBtcusdTradePrintsOneSecondOhlcv,
        Self::PolymarketBtcFiveMinuteOrderbooks,
        Self::PolymarketBtcFiveMinuteExecutionSnapshots,
        Self::ChainlinkBtcusdReferenceTicks,
        Self::ChainlinkBtcusdOneMinuteCandles,
        Self::BinanceBtcusdtFiveMinuteOpenInterest,
        Self::PolygonChainlinkBtcusdOracleRounds,
        Self::PmdataChainlinkBtcusdRefprice,
        Self::PmdataChainlinkBtcusdTwap30s,
        Self::PmdataChainlinkBtcusdTwap60s,
    ];

    pub const DEFAULT_WORKER: [Self; 13] = [
        Self::BtcFiveMinuteMarkets,
        Self::BtcFiveMinuteResolutions,
        Self::BinanceBtcusdtAggTrades,
        Self::BinanceBtcusdtL2OneSecondFeatures,
        Self::BinanceSpotBtcusdtL2OneSecondFeatures,
        Self::BinanceBtcusdtOneSecondKlines,
        Self::KrakenSpotBtcusdTradePrintsOneSecondOhlcv,
        Self::PolymarketBtcFiveMinuteOrderbooks,
        Self::PolymarketBtcFiveMinuteExecutionSnapshots,
        Self::ChainlinkBtcusdReferenceTicks,
        Self::ChainlinkBtcusdOneMinuteCandles,
        Self::BinanceBtcusdtFiveMinuteOpenInterest,
        Self::PolygonChainlinkBtcusdOracleRounds,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BtcFiveMinuteMarkets => "btc_five_minute_markets",
            Self::BtcFiveMinuteResolutions => "btc_five_minute_resolutions",
            Self::BinanceBtcusdtAggTrades => "binance_btcusdt_agg_trades",
            Self::BinanceBtcusdtL2OneSecondFeatures => "binance_btcusdt_l2_one_second_features",
            Self::BinanceSpotBtcusdtL2OneSecondFeatures => {
                "binance_spot_btcusdt_l2_one_second_features"
            }
            Self::BinanceBtcusdtOneSecondKlines => "binance_btcusdt_one_second_klines",
            Self::KrakenSpotBtcusdTradePrintsOneSecondOhlcv => {
                "kraken_spot_btcusd_trade_prints_one_second_ohlcv"
            }
            Self::PolymarketBtcFiveMinuteOrderbooks => "polymarket_btc_five_minute_orderbooks",
            Self::PolymarketBtcFiveMinuteExecutionSnapshots => {
                "polymarket_btc_five_minute_execution_snapshots"
            }
            Self::ChainlinkBtcusdReferenceTicks => "chainlink_btcusd_reference_ticks",
            Self::ChainlinkBtcusdOneMinuteCandles => "chainlink_btcusd_one_minute_candles",
            Self::BinanceBtcusdtFiveMinuteOpenInterest => {
                "binance_btcusdt_five_minute_open_interest"
            }
            Self::PolygonChainlinkBtcusdOracleRounds => "polygon_chainlink_btcusd_oracle_rounds",
            Self::PmdataChainlinkBtcusdRefprice => "pmdata_chainlink_btcusd_refprice",
            Self::PmdataChainlinkBtcusdTwap30s => "pmdata_chainlink_btcusd_twap_30s",
            Self::PmdataChainlinkBtcusdTwap60s => "pmdata_chainlink_btcusd_twap_60s",
        }
    }

    pub const fn supported_request_version(self) -> i32 {
        BACKFILL_REQUEST_VERSION
    }

    pub const fn accepts_new_requests(self) -> bool {
        !matches!(self, Self::PolymarketBtcFiveMinuteOrderbooks)
    }

    pub const fn alignment_seconds(self) -> i64 {
        match self {
            Self::BtcFiveMinuteMarkets | Self::BtcFiveMinuteResolutions => 300,
            Self::PolymarketBtcFiveMinuteOrderbooks
            | Self::PolymarketBtcFiveMinuteExecutionSnapshots => 3_600,
            Self::KrakenSpotBtcusdTradePrintsOneSecondOhlcv => 1,
            Self::BinanceBtcusdtAggTrades
            | Self::BinanceBtcusdtL2OneSecondFeatures
            | Self::BinanceSpotBtcusdtL2OneSecondFeatures
            | Self::BinanceBtcusdtOneSecondKlines
            | Self::ChainlinkBtcusdReferenceTicks
            | Self::ChainlinkBtcusdOneMinuteCandles
            | Self::BinanceBtcusdtFiveMinuteOpenInterest
            | Self::PolygonChainlinkBtcusdOracleRounds
            | Self::PmdataChainlinkBtcusdRefprice
            | Self::PmdataChainlinkBtcusdTwap30s
            | Self::PmdataChainlinkBtcusdTwap60s => 86_400,
        }
    }

    pub fn latest_complete_end(self, now: DateTime<Utc>) -> DateTime<Utc> {
        let alignment = self.alignment_seconds();
        let source_lag_seconds = match self {
            Self::PolymarketBtcFiveMinuteOrderbooks
            | Self::PolymarketBtcFiveMinuteExecutionSnapshots => 10 * 60,
            _ => 0,
        };
        let safe_now = now - chrono::Duration::seconds(source_lag_seconds);
        DateTime::from_timestamp(safe_now.timestamp().div_euclid(alignment) * alignment, 0)
            .expect("an aligned current UTC timestamp is representable")
    }
}

impl fmt::Display for IngesterKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for IngesterKey {
    type Err = BackfillRequestValidationError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "btc_five_minute_markets" => Ok(Self::BtcFiveMinuteMarkets),
            "btc_five_minute_resolutions" => Ok(Self::BtcFiveMinuteResolutions),
            "binance_btcusdt_agg_trades" => Ok(Self::BinanceBtcusdtAggTrades),
            "binance_btcusdt_l2_one_second_features" => Ok(Self::BinanceBtcusdtL2OneSecondFeatures),
            "binance_spot_btcusdt_l2_one_second_features" => {
                Ok(Self::BinanceSpotBtcusdtL2OneSecondFeatures)
            }
            "binance_btcusdt_one_second_klines" => Ok(Self::BinanceBtcusdtOneSecondKlines),
            "kraken_spot_btcusd_trade_prints_one_second_ohlcv" => {
                Ok(Self::KrakenSpotBtcusdTradePrintsOneSecondOhlcv)
            }
            "polymarket_btc_five_minute_orderbooks" => Ok(Self::PolymarketBtcFiveMinuteOrderbooks),
            "polymarket_btc_five_minute_execution_snapshots" => {
                Ok(Self::PolymarketBtcFiveMinuteExecutionSnapshots)
            }
            "chainlink_btcusd_reference_ticks" => Ok(Self::ChainlinkBtcusdReferenceTicks),
            "chainlink_btcusd_one_minute_candles" => Ok(Self::ChainlinkBtcusdOneMinuteCandles),
            "binance_btcusdt_five_minute_open_interest" => {
                Ok(Self::BinanceBtcusdtFiveMinuteOpenInterest)
            }
            "polygon_chainlink_btcusd_oracle_rounds" => {
                Ok(Self::PolygonChainlinkBtcusdOracleRounds)
            }
            "pmdata_chainlink_btcusd_refprice" => Ok(Self::PmdataChainlinkBtcusdRefprice),
            "pmdata_chainlink_btcusd_twap_30s" => Ok(Self::PmdataChainlinkBtcusdTwap30s),
            "pmdata_chainlink_btcusd_twap_60s" => Ok(Self::PmdataChainlinkBtcusdTwap60s),
            other => Err(BackfillRequestValidationError::new(format!(
                "unsupported ingester {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackfillRequest {
    pub ingester: IngesterKey,
    #[serde(default = "default_request_version")]
    pub request_version: i32,
    pub range_start: DateTime<Utc>,
    pub range_end: DateTime<Utc>,
    #[serde(default = "empty_object")]
    pub parameters: Value,
    pub idempotency_key: String,
}

impl BackfillRequest {
    pub fn validate(self) -> Result<ValidatedBackfillRequest, BackfillRequestValidationError> {
        let is_huggingface_goooddy = self.ingester
            == IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures
            && self.parameters.as_object().is_some_and(|parameters| {
                parameters.len() == 1
                    && parameters.get("strategy").and_then(Value::as_str)
                        == Some("huggingface_goooddy")
            });
        let is_local_orderbook_capacity = self.ingester
            == IngesterKey::PolymarketBtcFiveMinuteExecutionSnapshots
            && self.parameters.as_object().is_some_and(|parameters| {
                parameters.len() == 1
                    && parameters.get("source").and_then(Value::as_str) == Some("local_orderbook")
            });
        if !self.ingester.accepts_new_requests() {
            return Err(BackfillRequestValidationError::new(
                "raw PMXT orderbook ingestion is deprecated; use polymarket_btc_five_minute_execution_snapshots",
            ));
        }
        if self.request_version != self.ingester.supported_request_version() {
            return Err(BackfillRequestValidationError::new(format!(
                "ingester {} supports request_version {}, received {}",
                self.ingester,
                self.ingester.supported_request_version(),
                self.request_version
            )));
        }
        if self.range_end <= self.range_start {
            return Err(BackfillRequestValidationError::new(
                "range_end must be later than range_start",
            ));
        }
        let latest_complete_end = self.ingester.latest_complete_end(Utc::now());
        if self.range_end > latest_complete_end {
            return Err(BackfillRequestValidationError::new(
                "range_end must not exceed the latest complete historical source interval",
            ));
        }
        if !self.parameters.is_object() {
            return Err(BackfillRequestValidationError::new(
                "parameters must be a JSON object",
            ));
        }
        if self
            .parameters
            .as_object()
            .is_some_and(|parameters| !parameters.is_empty())
            && !is_huggingface_goooddy
            && !is_local_orderbook_capacity
        {
            return Err(BackfillRequestValidationError::new(format!(
                "ingester {} request_version {} does not accept parameters",
                self.ingester, self.request_version
            )));
        }

        let idempotency_key = self.idempotency_key.trim().to_string();
        if idempotency_key.is_empty() {
            return Err(BackfillRequestValidationError::new(
                "idempotency_key must not be empty",
            ));
        }
        if idempotency_key.len() > MAX_IDEMPOTENCY_KEY_BYTES {
            return Err(BackfillRequestValidationError::new(format!(
                "idempotency_key must be at most {MAX_IDEMPOTENCY_KEY_BYTES} bytes"
            )));
        }

        let alignment_seconds = self.ingester.alignment_seconds();
        validate_aligned_timestamp(self.range_start, alignment_seconds, "range_start")?;
        validate_aligned_timestamp(self.range_end, alignment_seconds, "range_end")?;
        match self.ingester {
            IngesterKey::BinanceBtcusdtL2OneSecondFeatures
                if self.range_start.timestamp() < BINANCE_L2_HISTORICAL_START_EPOCH
                    || self.range_end.timestamp() > BINANCE_L2_HISTORICAL_END_EPOCH =>
            {
                return Err(BackfillRequestValidationError::new(
                    "Binance BTCUSDT L2 historical requests must remain within [2026-04-14T00:00:00Z, 2026-08-02T00:00:00Z)",
                ));
            }
            IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures
                if self.range_start.timestamp() < BINANCE_SPOT_L2_HISTORICAL_START_EPOCH
                    || self.range_end.timestamp() > BINANCE_SPOT_L2_HISTORICAL_END_EPOCH =>
            {
                return Err(BackfillRequestValidationError::new(
                    "Binance spot BTCUSDT L2 historical requests must remain within [2026-04-14T00:00:00Z, 2026-08-02T00:00:00Z)",
                ));
            }
            IngesterKey::PmdataChainlinkBtcusdTwap30s
            | IngesterKey::PmdataChainlinkBtcusdTwap60s
                if self.range_start.timestamp() < PMDATA_TWAP_HISTORICAL_START_EPOCH =>
            {
                return Err(BackfillRequestValidationError::new(
                    "PMData Chainlink BTC/USD TWAP requests must start on or after 2026-08-01T00:00:00Z",
                ));
            }
            IngesterKey::PmdataChainlinkBtcusdRefprice
                if self.range_start.timestamp() < PMDATA_REFPRICE_HISTORICAL_START_EPOCH =>
            {
                return Err(BackfillRequestValidationError::new(
                    "PMData Chainlink BTC/USD RefPrice requests must start on or after 2026-06-07T00:00:00Z",
                ));
            }
            _ => {}
        }

        let seconds = (self.range_end - self.range_start).num_seconds();
        let work_unit_seconds =
            if self.ingester == IngesterKey::KrakenSpotBtcusdTradePrintsOneSecondOhlcv {
                86_400
            } else {
                alignment_seconds
            };
        let expected_work_units =
            u64::try_from(seconds.saturating_add(work_unit_seconds - 1) / work_unit_seconds)
                .map_err(|_| {
                    BackfillRequestValidationError::new(
                        "requested range contains too many work units",
                    )
                })?;
        if expected_work_units == 0 {
            return Err(BackfillRequestValidationError::new(
                "requested range does not contain a complete work unit",
            ));
        }
        if matches!(
            self.ingester,
            IngesterKey::BinanceBtcusdtL2OneSecondFeatures
                | IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures
                | IngesterKey::PmdataChainlinkBtcusdTwap30s
                | IngesterKey::PmdataChainlinkBtcusdTwap60s
                | IngesterKey::PmdataChainlinkBtcusdRefprice
        ) && !is_huggingface_goooddy
            && expected_work_units != 1
        {
            return Err(BackfillRequestValidationError::new(
                "daily archive requests must contain exactly one UTC-day shard",
            ));
        }
        if self.ingester == IngesterKey::KrakenSpotBtcusdTradePrintsOneSecondOhlcv
            && seconds > 86_400
        {
            return Err(BackfillRequestValidationError::new(
                "Kraken spot trade-print requests must not exceed one UTC day",
            ));
        }
        if is_huggingface_goooddy {
            let valid_range = (self.range_start.timestamp(), self.range_end.timestamp());
            if !matches!(
                valid_range,
                (1780444800, 1782864000) | (1782864000, 1785542400)
            ) {
                return Err(BackfillRequestValidationError::new(
                    "Hugging Face Goooddy requests must be one pinned source shard: [2026-06-03T00:00:00Z, 2026-07-01T00:00:00Z) or [2026-07-01T00:00:00Z, 2026-08-01T00:00:00Z)",
                ));
            }
        }

        Ok(ValidatedBackfillRequest {
            ingester: self.ingester,
            request_version: self.request_version,
            range_start: self.range_start,
            range_end: self.range_end,
            parameters: self.parameters,
            idempotency_key,
            expected_work_units,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidatedBackfillRequest {
    pub ingester: IngesterKey,
    pub request_version: i32,
    pub range_start: DateTime<Utc>,
    pub range_end: DateTime<Utc>,
    pub parameters: Value,
    pub idempotency_key: String,
    pub expected_work_units: u64,
}

impl ValidatedBackfillRequest {
    pub fn persisted_request(&self) -> Value {
        serde_json::json!({
            "ingester": self.ingester,
            "request_version": self.request_version,
            "range_start": self.range_start,
            "range_end": self.range_end,
            "parameters": self.parameters,
            "idempotency_key": self.idempotency_key,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackfillRequestValidationError {
    message: String,
}

impl BackfillRequestValidationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for BackfillRequestValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for BackfillRequestValidationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackfillJobStatus {
    Queued,
    Running,
    CancelRequested,
    Completed,
    Failed,
    Cancelled,
}

impl BackfillJobStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::CancelRequested => "cancel_requested",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

impl FromStr for BackfillJobStatus {
    type Err = BackfillRequestValidationError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "queued" => Ok(Self::Queued),
            "running" => Ok(Self::Running),
            "cancel_requested" => Ok(Self::CancelRequested),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            other => Err(BackfillRequestValidationError::new(format!(
                "unknown backfill job status {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillJob {
    pub job_id: Uuid,
    #[serde(rename = "ingester")]
    pub ingester_key: String,
    pub request_version: i32,
    pub status: BackfillJobStatus,
    pub range_start: Option<DateTime<Utc>>,
    pub range_end: Option<DateTime<Utc>>,
    pub idempotency_key: Option<String>,
    pub request: Value,
    pub progress: Value,
    pub checkpoint: Value,
    pub summary: Value,
    pub attempt: i32,
    pub max_attempts: i32,
    pub next_attempt_at: DateTime<Utc>,
    pub worker_id: Option<String>,
    #[serde(skip_serializing)]
    pub lease_token: Option<Uuid>,
    pub lease_expires_at: Option<DateTime<Utc>>,
    pub heartbeat_at: Option<DateTime<Utc>>,
    pub cancel_requested_at: Option<DateTime<Utc>>,
    pub requested_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
    pub updated_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lookback_days: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_trade_usd: Option<Decimal>,
}

impl BackfillJob {
    pub fn ingester(&self) -> Result<IngesterKey, BackfillRequestValidationError> {
        self.ingester_key.parse()
    }
}

#[derive(Debug, Clone)]
pub struct ClaimedJob {
    pub job: BackfillJob,
    pub worker_id: String,
    pub lease_token: Uuid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackfillEventLevel {
    Debug,
    Info,
    Warn,
    Error,
}

impl BackfillEventLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }
}

impl FromStr for BackfillEventLevel {
    type Err = BackfillRequestValidationError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "debug" => Ok(Self::Debug),
            "info" => Ok(Self::Info),
            "warn" => Ok(Self::Warn),
            "error" => Ok(Self::Error),
            other => Err(BackfillRequestValidationError::new(format!(
                "unknown backfill event level {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillJobEvent {
    pub event_id: Uuid,
    pub job_id: Uuid,
    pub timestamp_utc: DateTime<Utc>,
    pub level: BackfillEventLevel,
    pub message: String,
    pub metadata: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BackfillProgress {
    pub expected_work_units: u64,
    pub completed_work_units: u64,
    pub records_read: u64,
    pub records_committed: u64,
    pub bytes_downloaded: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_logical_key: Option<String>,
    #[serde(default = "empty_object")]
    pub details: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BackfillCheckpoint {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logical_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_date: Option<NaiveDate>,
    pub committed_record_ordinal: u64,
    #[serde(default = "empty_object")]
    pub details: Value,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BackfillJobSummary {
    pub expected_work_units: u64,
    pub completed_work_units: u64,
    pub records_read: u64,
    pub records_committed: u64,
    pub duplicate_records: u64,
    pub artifacts_completed: u64,
    pub artifacts_failed: u64,
    #[serde(default)]
    pub missing_by_reason: BTreeMap<String, u64>,
    #[serde(default = "empty_object")]
    pub details: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackfillFailureKind {
    Transient,
    Permanent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerControl {
    Continue,
    CancelRequested,
    LeaseLost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackfillArtifactStatus {
    Pending,
    Downloading,
    Downloaded,
    Verified,
    Ingesting,
    Completed,
    Failed,
}

impl BackfillArtifactStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Downloading => "downloading",
            Self::Downloaded => "downloaded",
            Self::Verified => "verified",
            Self::Ingesting => "ingesting",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

impl FromStr for BackfillArtifactStatus {
    type Err = BackfillRequestValidationError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "pending" => Ok(Self::Pending),
            "downloading" => Ok(Self::Downloading),
            "downloaded" => Ok(Self::Downloaded),
            "verified" => Ok(Self::Verified),
            "ingesting" => Ok(Self::Ingesting),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            other => Err(BackfillRequestValidationError::new(format!(
                "unknown backfill artifact status {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillArtifact {
    pub artifact_id: Uuid,
    pub job_id: Uuid,
    pub ingester_key: String,
    pub logical_key: String,
    pub provider: String,
    pub source_uri: String,
    pub source_date: Option<NaiveDate>,
    pub checksum_algorithm: String,
    pub expected_checksum: Option<String>,
    pub actual_checksum: Option<String>,
    pub compressed_bytes: Option<i64>,
    pub record_count: Option<i64>,
    pub minimum_source_timestamp: Option<DateTime<Utc>>,
    pub maximum_source_timestamp: Option<DateTime<Utc>>,
    pub status: BackfillArtifactStatus,
    pub metadata: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct ArtifactSpec {
    pub job_id: Uuid,
    pub ingester: IngesterKey,
    pub logical_key: String,
    pub provider: String,
    pub source_uri: String,
    pub source_date: Option<NaiveDate>,
    pub expected_checksum: Option<String>,
    pub metadata: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactDisposition {
    Process,
    AlreadyCompleted,
}

#[derive(Debug, Clone)]
pub struct PreparedArtifact {
    pub artifact: BackfillArtifact,
    pub disposition: ArtifactDisposition,
}

#[derive(Debug, Clone)]
pub struct ArtifactCompletion {
    pub actual_checksum: String,
    pub compressed_bytes: u64,
    pub record_count: u64,
    pub minimum_source_timestamp: Option<DateTime<Utc>>,
    pub maximum_source_timestamp: Option<DateTime<Utc>>,
    pub metadata: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BtcReferenceFactType {
    OpeningBoundary,
    FinalPrice,
}

impl BtcReferenceFactType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpeningBoundary => "opening_boundary",
            Self::FinalPrice => "final_price",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BtcReferenceFact {
    pub market_id: String,
    pub artifact_id: Uuid,
    pub fact_type: BtcReferenceFactType,
    pub value: Decimal,
    pub provider: String,
    pub source_effective_at: DateTime<Utc>,
    pub fetched_at: DateTime<Utc>,
    pub payload_sha256: String,
    pub evidence: Value,
}

#[derive(Debug, Clone)]
pub struct BtcResolutionCandidate {
    pub market: BtcIntervalMarket,
    pub official_outcome: Option<String>,
    pub official_winning_token_id: Option<String>,
    pub official_resolved_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchWriteResult {
    pub input_records: u64,
    pub inserted_records: u64,
    pub duplicate_records: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinanceAggregateTradeRecord {
    pub symbol: String,
    pub aggregate_trade_id: i64,
    pub price: Decimal,
    pub quantity: Decimal,
    pub first_trade_id: i64,
    pub last_trade_id: i64,
    pub trade_timestamp: DateTime<Utc>,
    pub buyer_maker: bool,
    pub best_match: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinanceOneSecondKlineRecord {
    pub symbol: String,
    pub open_timestamp: DateTime<Utc>,
    pub close_timestamp: DateTime<Utc>,
    pub open_price: Decimal,
    pub high_price: Decimal,
    pub low_price: Decimal,
    pub close_price: Decimal,
    pub base_volume: Decimal,
    pub quote_volume: Decimal,
    pub trade_count: i64,
    pub taker_buy_base_volume: Decimal,
    pub taker_buy_quote_volume: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinanceL2OneSecondFeature {
    pub symbol: String,
    pub second_start: DateTime<Utc>,
    pub source_event_timestamp: DateTime<Utc>,
    pub provider_received_at: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
    pub source_update_id: i64,
    pub feature_schema_version: String,
    pub quality_status: String,
    pub midpoint: Decimal,
    pub microprice: Decimal,
    pub spread_bps: Decimal,
    pub bid_depth_5: Decimal,
    pub ask_depth_5: Decimal,
    pub imbalance_5: Decimal,
    pub bid_depth_10: Decimal,
    pub ask_depth_10: Decimal,
    pub imbalance_10: Decimal,
    pub bid_depth_20: Decimal,
    pub ask_depth_20: Decimal,
    pub imbalance_20: Decimal,
    pub bid_depth_slope_20: Decimal,
    pub ask_depth_slope_20: Decimal,
    pub bid_depth_concentration_20: Decimal,
    pub ask_depth_concentration_20: Decimal,
    pub bid_quote_replenishment_1s: Decimal,
    pub ask_quote_replenishment_1s: Decimal,
    pub bid_quote_churn_1s: Decimal,
    pub ask_quote_churn_1s: Decimal,
    pub midpoint_change_bps_1s: Decimal,
    pub spread_bps_delta_1s: Decimal,
    pub depth_20_change_bps_1s: Decimal,
    pub imbalance_20_delta_1s: Decimal,
    pub midpoint_change_bps_5s: Decimal,
    pub spread_bps_delta_5s: Decimal,
    pub depth_20_change_bps_5s: Decimal,
    pub imbalance_20_delta_5s: Decimal,
    pub midpoint_change_bps_15s: Decimal,
    pub spread_bps_delta_15s: Decimal,
    pub depth_20_change_bps_15s: Decimal,
    pub imbalance_20_delta_15s: Decimal,
    pub midpoint_change_bps_30s: Decimal,
    pub spread_bps_delta_30s: Decimal,
    pub depth_20_change_bps_30s: Decimal,
    pub imbalance_20_delta_30s: Decimal,
    pub midpoint_change_bps_60s: Decimal,
    pub spread_bps_delta_60s: Decimal,
    pub depth_20_change_bps_60s: Decimal,
    pub imbalance_20_delta_60s: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BtcOrderbookMarketScope {
    pub market_id: String,
    pub condition_id: String,
    pub up_token_id: String,
    pub down_token_id: String,
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BtcOrderbookArchiveEvent {
    pub source_row_number: i64,
    pub provider_received_at: DateTime<Utc>,
    pub source_timestamp: DateTime<Utc>,
    pub condition_id: String,
    pub asset_id: String,
    pub event_type: String,
    pub bids: Option<Value>,
    pub asks: Option<Value>,
    pub price: Option<Decimal>,
    pub size: Option<Decimal>,
    pub side: Option<String>,
    pub best_bid: Option<Decimal>,
    pub best_ask: Option<Decimal>,
    pub fee_rate_bps: Option<i32>,
    pub transaction_hash: Option<String>,
    pub old_tick_size: Option<Decimal>,
    pub new_tick_size: Option<Decimal>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BtcExecutionSnapshot {
    pub market_id: String,
    pub sampled_at: DateTime<Utc>,
    pub up_source_row_number: Option<i64>,
    pub up_source_timestamp: Option<DateTime<Utc>>,
    pub up_provider_received_at: Option<DateTime<Utc>>,
    pub up_best_bid: Option<Decimal>,
    pub up_best_ask: Option<Decimal>,
    pub up_best_bid_size: Option<Decimal>,
    pub up_best_ask_size: Option<Decimal>,
    pub up_bid_depth: Option<Decimal>,
    pub up_ask_depth: Option<Decimal>,
    pub up_ask_vwap_1: Option<Decimal>,
    pub up_ask_vwap_5: Option<Decimal>,
    pub up_ask_vwap_10: Option<Decimal>,
    pub up_ask_vwap_15: Option<Decimal>,
    pub up_ask_vwap_20: Option<Decimal>,
    pub up_ask_vwap_25: Option<Decimal>,
    pub up_ask_vwap_30: Option<Decimal>,
    pub up_ask_vwap_40: Option<Decimal>,
    pub up_ask_vwap_50: Option<Decimal>,
    pub up_ask_vwap_75: Option<Decimal>,
    pub up_ask_vwap_100: Option<Decimal>,
    pub up_ask_vwap_125: Option<Decimal>,
    pub up_ask_vwap_150: Option<Decimal>,
    pub up_ask_vwap_175: Option<Decimal>,
    pub up_ask_vwap_200: Option<Decimal>,
    pub up_imbalance: Option<Decimal>,
    pub down_source_row_number: Option<i64>,
    pub down_source_timestamp: Option<DateTime<Utc>>,
    pub down_provider_received_at: Option<DateTime<Utc>>,
    pub down_best_bid: Option<Decimal>,
    pub down_best_ask: Option<Decimal>,
    pub down_best_bid_size: Option<Decimal>,
    pub down_best_ask_size: Option<Decimal>,
    pub down_bid_depth: Option<Decimal>,
    pub down_ask_depth: Option<Decimal>,
    pub down_ask_vwap_1: Option<Decimal>,
    pub down_ask_vwap_5: Option<Decimal>,
    pub down_ask_vwap_10: Option<Decimal>,
    pub down_ask_vwap_15: Option<Decimal>,
    pub down_ask_vwap_20: Option<Decimal>,
    pub down_ask_vwap_25: Option<Decimal>,
    pub down_ask_vwap_30: Option<Decimal>,
    pub down_ask_vwap_40: Option<Decimal>,
    pub down_ask_vwap_50: Option<Decimal>,
    pub down_ask_vwap_75: Option<Decimal>,
    pub down_ask_vwap_100: Option<Decimal>,
    pub down_ask_vwap_125: Option<Decimal>,
    pub down_ask_vwap_150: Option<Decimal>,
    pub down_ask_vwap_175: Option<Decimal>,
    pub down_ask_vwap_200: Option<Decimal>,
    pub down_imbalance: Option<Decimal>,
    pub quality_flags: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChainlinkBtcusdArchiveTick {
    pub feed_id: String,
    pub source_timestamp: DateTime<Utc>,
    pub valid_from_timestamp: DateTime<Utc>,
    pub price: Decimal,
    pub bid: Decimal,
    pub ask: Decimal,
    pub report_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PmdataChainlinkBtcusdTwapRecord {
    pub source_timestamp: DateTime<Utc>,
    pub provider_received_at: DateTime<Utc>,
    pub valid_from_timestamp: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub window_seconds: i16,
    pub twap_price: Decimal,
    pub full_accuracy_value: String,
    pub report_version: String,
    pub source_date: NaiveDate,
    pub archive_row_number: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PmdataChainlinkBtcusdRefpriceRecord {
    pub source_timestamp: DateTime<Utc>,
    pub provider_received_at: DateTime<Utc>,
    pub valid_from_timestamp: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub price: Decimal,
    pub bid: Option<Decimal>,
    pub ask: Option<Decimal>,
    pub report_version: Option<String>,
    pub canonical_row_sha256: String,
    pub source_date: NaiveDate,
    pub archive_row_number: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChainlinkBtcusdOneMinuteCandle {
    pub symbol: String,
    pub open_timestamp: DateTime<Utc>,
    pub close_timestamp: DateTime<Utc>,
    pub open_price: Decimal,
    pub high_price: Decimal,
    pub low_price: Decimal,
    pub close_price: Decimal,
    pub volume: Option<Decimal>,
    pub volume_supported: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinanceBtcusdtOpenInterestRecord {
    pub symbol: String,
    pub source_timestamp: DateTime<Utc>,
    pub period_seconds: i32,
    pub sum_open_interest: Decimal,
    pub sum_open_interest_value: Decimal,
    pub cmc_circulating_supply: Option<Decimal>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolygonChainlinkBtcusdOracleRound {
    pub chain_id: i64,
    pub feed_proxy_address: String,
    pub aggregator_address: String,
    pub phase_id: i32,
    pub aggregator_round_id: i64,
    pub source_timestamp: DateTime<Utc>,
    pub block_timestamp: DateTime<Utc>,
    pub answer_raw: Decimal,
    pub price: Decimal,
    pub decimals: i32,
    pub block_number: i64,
    pub block_hash: String,
    pub transaction_hash: String,
    pub log_index: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingReadiness {
    pub range_start: DateTime<Utc>,
    pub range_end: DateTime<Utc>,
    pub expected_markets: i64,
    pub valid_market_identities: i64,
    pub opening_boundaries: i64,
    pub final_prices: i64,
    pub official_outcomes: i64,
    pub aggregate_trade_covered_markets: i64,
    pub one_second_kline_covered_markets: i64,
    pub chainlink_covered_markets: i64,
    pub orderbook_covered_markets: i64,
    pub usable_markets: i64,
    pub aggregate_trade_min_timestamp: Option<DateTime<Utc>>,
    pub aggregate_trade_max_timestamp: Option<DateTime<Utc>>,
    pub one_second_kline_min_timestamp: Option<DateTime<Utc>>,
    pub one_second_kline_max_timestamp: Option<DateTime<Utc>>,
    pub chainlink_min_timestamp: Option<DateTime<Utc>>,
    pub chainlink_max_timestamp: Option<DateTime<Utc>>,
    pub orderbook_min_timestamp: Option<DateTime<Utc>>,
    pub orderbook_max_timestamp: Option<DateTime<Utc>>,
    pub missing_by_reason: BTreeMap<String, i64>,
    pub artifact_status_counts: BTreeMap<String, i64>,
}

fn default_request_version() -> i32 {
    BACKFILL_REQUEST_VERSION
}

fn empty_object() -> Value {
    serde_json::json!({})
}

fn validate_aligned_timestamp(
    timestamp: DateTime<Utc>,
    alignment_seconds: i64,
    field: &str,
) -> Result<(), BackfillRequestValidationError> {
    if timestamp.timestamp_subsec_nanos() != 0
        || timestamp.timestamp().rem_euclid(alignment_seconds) != 0
    {
        return Err(BackfillRequestValidationError::new(format!(
            "{field} must align to a {alignment_seconds}-second UTC boundary"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    fn timestamp(epoch: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(epoch, 0).single().unwrap()
    }

    fn request(ingester: IngesterKey, start: i64, end: i64) -> BackfillRequest {
        BackfillRequest {
            ingester,
            request_version: BACKFILL_REQUEST_VERSION,
            range_start: timestamp(start),
            range_end: timestamp(end),
            parameters: serde_json::json!({}),
            idempotency_key: " test-key ".to_string(),
        }
    }

    #[test]
    fn validates_five_minute_request_and_canonicalizes_idempotency_key() {
        let request = request(
            IngesterKey::BtcFiveMinuteMarkets,
            1_783_392_000,
            1_783_392_600,
        )
        .validate()
        .unwrap();
        assert_eq!(request.idempotency_key, "test-key");
        assert_eq!(request.expected_work_units, 2);
    }

    #[test]
    fn execution_snapshots_accept_only_the_explicit_local_orderbook_source() {
        let mut local = request(
            IngesterKey::PolymarketBtcFiveMinuteExecutionSnapshots,
            1_787_011_200,
            1_787_014_800,
        );
        local.parameters = serde_json::json!({"source": "local_orderbook"});
        assert!(local.validate().is_ok());

        let mut unsupported = request(
            IngesterKey::PolymarketBtcFiveMinuteExecutionSnapshots,
            1_787_011_200,
            1_787_014_800,
        );
        unsupported.parameters = serde_json::json!({"source": "other"});
        assert!(unsupported.validate().is_err());
    }

    #[test]
    fn binance_requests_require_whole_utc_days() {
        let aligned = request(
            IngesterKey::BinanceBtcusdtAggTrades,
            1_783_382_400,
            1_783_468_800,
        );
        assert!(aligned.validate().is_ok());

        let unaligned = request(
            IngesterKey::BinanceBtcusdtAggTrades,
            1_783_382_700,
            1_783_468_800,
        );
        assert!(unaligned.validate().is_err());

        let l2_aligned = request(
            IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
            1_783_382_400,
            1_783_468_800,
        )
        .validate()
        .unwrap();
        assert_eq!(l2_aligned.expected_work_units, 1);
        assert!(request(
            IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
            1_783_382_401,
            1_783_468_800,
        )
        .validate()
        .is_err());
    }

    #[test]
    fn binance_l2_accepts_the_first_and_last_historical_daily_shards() {
        const UTC_DAY_SECONDS: i64 = 86_400;

        for ingester in [
            IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
            IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures,
        ] {
            let april_14 = request(
                ingester,
                BINANCE_L2_HISTORICAL_START_EPOCH,
                BINANCE_L2_HISTORICAL_START_EPOCH + UTC_DAY_SECONDS,
            )
            .validate()
            .unwrap();
            assert_eq!(april_14.expected_work_units, 1);

            let august_1 = request(
                ingester,
                BINANCE_L2_HISTORICAL_END_EPOCH - UTC_DAY_SECONDS,
                BINANCE_L2_HISTORICAL_END_EPOCH,
            )
            .validate()
            .unwrap();
            assert_eq!(august_1.expected_work_units, 1);
        }
    }

    #[test]
    fn binance_l2_rejects_daily_shards_outside_the_historical_boundary() {
        const UTC_DAY_SECONDS: i64 = 86_400;

        for ingester in [
            IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
            IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures,
        ] {
            let april_13 = request(
                ingester,
                BINANCE_L2_HISTORICAL_START_EPOCH - UTC_DAY_SECONDS,
                BINANCE_L2_HISTORICAL_START_EPOCH,
            );
            assert!(april_13.validate().is_err());

            let august_2 = request(
                ingester,
                BINANCE_L2_HISTORICAL_END_EPOCH,
                BINANCE_L2_HISTORICAL_END_EPOCH + UTC_DAY_SECONDS,
            );
            assert!(august_2.validate().is_err());
        }
    }

    #[test]
    fn binance_l2_rejects_multi_day_and_sub_day_shards() {
        const UTC_DAY_SECONDS: i64 = 86_400;

        for ingester in [
            IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
            IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures,
        ] {
            let multi_day = request(
                ingester,
                BINANCE_L2_HISTORICAL_START_EPOCH,
                BINANCE_L2_HISTORICAL_START_EPOCH + (2 * UTC_DAY_SECONDS),
            );
            assert!(multi_day.validate().is_err());

            let sub_day = request(
                ingester,
                BINANCE_L2_HISTORICAL_START_EPOCH,
                BINANCE_L2_HISTORICAL_START_EPOCH + (UTC_DAY_SECONDS / 2),
            );
            assert!(sub_day.validate().is_err());
        }
    }

    #[test]
    fn source_work_units_use_their_native_archive_cadence() {
        let pmxt = request(
            IngesterKey::PolymarketBtcFiveMinuteExecutionSnapshots,
            1_783_382_400,
            1_783_386_000,
        )
        .validate()
        .unwrap();
        assert_eq!(pmxt.expected_work_units, 1);

        let chainlink = request(
            IngesterKey::ChainlinkBtcusdReferenceTicks,
            1_783_382_400,
            1_783_468_800,
        )
        .validate()
        .unwrap();
        assert_eq!(chainlink.expected_work_units, 1);

        let polygon_chainlink = request(
            IngesterKey::PolygonChainlinkBtcusdOracleRounds,
            1_783_382_400,
            1_783_468_800,
        )
        .validate()
        .unwrap();
        assert_eq!(polygon_chainlink.expected_work_units, 1);

        assert!(request(
            IngesterKey::PolymarketBtcFiveMinuteExecutionSnapshots,
            1_783_382_700,
            1_783_386_000,
        )
        .validate()
        .is_err());
        assert!(request(
            IngesterKey::PolymarketBtcFiveMinuteOrderbooks,
            1_783_382_400,
            1_783_386_000,
        )
        .validate()
        .is_err());
    }

    #[test]
    fn rejects_unknown_versions_parameters_and_empty_ranges() {
        let mut wrong_version = request(IngesterKey::BtcFiveMinuteResolutions, 0, 300);
        wrong_version.request_version = 2;
        assert!(wrong_version.validate().is_err());

        let mut parameters = request(IngesterKey::BtcFiveMinuteResolutions, 0, 300);
        parameters.parameters = serde_json::json!({"symbol": "BTCUSDT"});
        assert!(parameters.validate().is_err());

        assert!(request(IngesterKey::BtcFiveMinuteResolutions, 0, 0)
            .validate()
            .is_err());

        let future = Utc::now().timestamp() + 86_400;
        let aligned_future = future.div_euclid(300) * 300;
        assert!(request(
            IngesterKey::BtcFiveMinuteMarkets,
            aligned_future,
            aligned_future + 300,
        )
        .validate()
        .is_err());
    }

    #[test]
    fn stored_enum_names_are_stable() {
        for ingester in IngesterKey::ALL {
            assert_eq!(ingester.as_str().parse::<IngesterKey>().unwrap(), ingester);
            assert_eq!(
                serde_json::to_value(ingester).unwrap(),
                Value::String(ingester.as_str().to_string())
            );
        }
    }
}
