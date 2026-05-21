use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    pub event_id: String,
    pub market_id: String,
    pub outcome_group_id: Option<String>,
    pub question: String,
    pub category: Option<String>,
    pub active: bool,
    pub closed: bool,
    pub archived: bool,
    pub neg_risk: bool,
    pub neg_risk_augmented: bool,
    pub rules: Option<String>,
    pub end_date: Option<DateTime<Utc>>,
    pub underlying_key: String,
    pub resolution_score: i32,
    #[serde(default)]
    pub outcome_tokens: Vec<OutcomeToken>,
    pub raw: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeToken {
    pub market_id: String,
    pub token_id: String,
    pub outcome: String,
    pub side: TokenSide,
    pub condition_id: Option<String>,
    pub tick_size: Decimal,
    pub neg_risk: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokenSide {
    Yes,
    No,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalCandidate {
    pub signal_id: Uuid,
    #[serde(default)]
    pub process_id: Option<Uuid>,
    pub signal_type: SignalType,
    pub market_id: String,
    pub expected_edge: Decimal,
    pub threshold: Decimal,
    pub size: Decimal,
    pub status: SignalStatus,
    pub reject_reason: Option<String>,
    pub worst_case_loss: Option<Decimal>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookSnapshot {
    pub snapshot_id: Uuid,
    pub timestamp_utc: DateTime<Utc>,
    pub market_id: String,
    pub token_id: String,
    pub best_bid: Option<Decimal>,
    pub best_ask: Option<Decimal>,
    pub tick_size: Decimal,
    pub stale_level_count: i32,
    pub fresh_depth_bid: Decimal,
    pub fresh_depth_ask: Decimal,
    pub book: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionRecord {
    pub position_id: Uuid,
    pub market_id: String,
    pub token_id: String,
    pub underlying_key: String,
    pub status: String,
    pub size: Decimal,
    pub cost_basis: Decimal,
    pub worst_case_loss: Decimal,
    pub raw: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SignalType {
    CheapBasket,
    ExpensiveBasket,
    Conversion,
    WhaleFollow,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SignalStatus {
    Detected,
    Rejected,
    Submitted,
    Filled,
    Recovered,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRequest {
    pub client_order_id: Uuid,
    #[serde(default)]
    pub process_id: Option<Uuid>,
    pub market_id: String,
    pub token_id: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub price: Decimal,
    pub size: Decimal,
    pub signal_id: Option<Uuid>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OrderType {
    Fok,
    Gtc,
    Gtd,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRecord {
    pub order_id: String,
    pub request: OrderRequest,
    pub state: OrderState,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OrderState {
    Created,
    Submitted,
    Acknowledged,
    PartiallyFilled,
    Filled,
    CancelRequested,
    Cancelled,
    Rejected,
    Expired,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillRecord {
    pub fill_id: Uuid,
    #[serde(default)]
    pub process_id: Option<Uuid>,
    pub order_id: String,
    pub token_id: String,
    pub price: Decimal,
    pub size: Decimal,
    pub fee: Decimal,
    pub source: FillSource,
    pub filled_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FillSource {
    Sim,
    Paper,
    Live,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionRequest {
    pub conversion_id: Uuid,
    pub market_id: String,
    pub no_token_id: String,
    pub size: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionResult {
    pub conversion_id: Uuid,
    pub status: String,
    pub tx_hash: Option<String>,
    pub latency_ms: i64,
    pub gas_cost_usd: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataApiPosition {
    #[serde(default)]
    pub proxy_wallet: Option<String>,
    #[serde(default)]
    pub asset: Option<String>,
    #[serde(default)]
    pub condition_id: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub size: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub avg_price: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub initial_value: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub current_value: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub cash_pnl: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub percent_pnl: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub total_bought: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub realized_pnl: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub percent_realized_pnl: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub cur_price: Option<Decimal>,
    #[serde(default)]
    pub redeemable: Option<bool>,
    #[serde(default)]
    pub mergeable: Option<bool>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub slug: Option<String>,
    #[serde(default)]
    pub icon: Option<String>,
    #[serde(default)]
    pub event_slug: Option<String>,
    #[serde(default)]
    pub outcome: Option<String>,
    #[serde(default)]
    pub outcome_index: Option<i64>,
    #[serde(default)]
    pub opposite_outcome: Option<String>,
    #[serde(default)]
    pub opposite_asset: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
    #[serde(default)]
    pub negative_risk: Option<bool>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataApiClosedPosition {
    #[serde(default)]
    pub proxy_wallet: Option<String>,
    #[serde(default)]
    pub asset: Option<String>,
    #[serde(default)]
    pub condition_id: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub avg_price: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub total_bought: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub realized_pnl: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub cur_price: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_i64")]
    pub timestamp: Option<i64>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub slug: Option<String>,
    #[serde(default)]
    pub icon: Option<String>,
    #[serde(default)]
    pub event_slug: Option<String>,
    #[serde(default)]
    pub outcome: Option<String>,
    #[serde(default)]
    pub outcome_index: Option<i64>,
    #[serde(default)]
    pub opposite_outcome: Option<String>,
    #[serde(default)]
    pub opposite_asset: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataApiActivity {
    #[serde(default)]
    pub proxy_wallet: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_i64")]
    pub timestamp: Option<i64>,
    #[serde(default)]
    pub condition_id: Option<String>,
    #[serde(default, rename = "type")]
    pub activity_type: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub size: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub usdc_size: Option<Decimal>,
    #[serde(default)]
    pub transaction_hash: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub price: Option<Decimal>,
    #[serde(default)]
    pub asset: Option<String>,
    #[serde(default)]
    pub side: Option<String>,
    #[serde(default)]
    pub outcome_index: Option<i64>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub slug: Option<String>,
    #[serde(default)]
    pub icon: Option<String>,
    #[serde(default)]
    pub event_slug: Option<String>,
    #[serde(default)]
    pub outcome: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub pseudonym: Option<String>,
    #[serde(default)]
    pub bio: Option<String>,
    #[serde(default)]
    pub profile_image: Option<String>,
    #[serde(default)]
    pub profile_image_optimized: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataApiValue {
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub value: Option<Decimal>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

fn deserialize_optional_decimal<'de, D>(deserializer: D) -> Result<Option<Decimal>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(raw)) => raw
            .parse::<Decimal>()
            .map(Some)
            .map_err(serde::de::Error::custom),
        Some(serde_json::Value::Number(number)) => number
            .to_string()
            .parse::<Decimal>()
            .map(Some)
            .map_err(serde::de::Error::custom),
        Some(other) => Err(serde::de::Error::custom(format!(
            "expected decimal string or number, got {other}"
        ))),
    }
}

fn deserialize_optional_i64<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(raw)) => raw
            .parse::<i64>()
            .map(Some)
            .map_err(serde::de::Error::custom),
        Some(serde_json::Value::Number(number)) => number
            .as_i64()
            .ok_or_else(|| serde::de::Error::custom("expected integer timestamp"))
            .map(Some),
        Some(other) => Err(serde::de::Error::custom(format!(
            "expected integer timestamp string or number, got {other}"
        ))),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataApiTrade {
    #[serde(default)]
    pub proxy_wallet: Option<String>,
    #[serde(default)]
    pub side: Option<String>,
    #[serde(default)]
    pub asset: Option<String>,
    #[serde(default)]
    pub condition_id: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub size: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_decimal")]
    pub price: Option<Decimal>,
    #[serde(default, deserialize_with = "deserialize_optional_i64")]
    pub timestamp: Option<i64>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub slug: Option<String>,
    #[serde(default)]
    pub icon: Option<String>,
    #[serde(default)]
    pub event_slug: Option<String>,
    #[serde(default)]
    pub outcome: Option<String>,
    #[serde(default)]
    pub outcome_index: Option<i64>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub pseudonym: Option<String>,
    #[serde(default)]
    pub bio: Option<String>,
    #[serde(default)]
    pub profile_image: Option<String>,
    #[serde(default)]
    pub profile_image_optimized: Option<String>,
    #[serde(default)]
    pub transaction_hash: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackfillJobStatus {
    Queued,
    Running,
    CancelRequested,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillJob {
    pub job_id: Uuid,
    pub job_type: String,
    pub status: BackfillJobStatus,
    pub requested_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub cancel_requested_at: Option<DateTime<Utc>>,
    pub lookback_days: i32,
    pub min_trade_usd: Decimal,
    pub request: serde_json::Value,
    pub summary: serde_json::Value,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleTrade {
    pub trade_id: Uuid,
    pub proxy_wallet: String,
    pub asset: String,
    pub condition_id: Option<String>,
    pub market_id: Option<String>,
    pub side: String,
    pub outcome: Option<String>,
    pub price: Decimal,
    pub size: Decimal,
    pub cash_value: Decimal,
    pub timestamp_utc: DateTime<Utc>,
    pub title: Option<String>,
    pub slug: Option<String>,
    pub event_slug: Option<String>,
    pub transaction_hash: Option<String>,
    pub raw_payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletScore {
    pub proxy_wallet: String,
    pub score_version: String,
    pub resolved_markets: i32,
    pub total_trades: i32,
    pub total_volume: Decimal,
    pub realized_pnl: Decimal,
    pub roi: Decimal,
    pub win_rate: Decimal,
    pub avg_trade_size: Decimal,
    pub max_drawdown: Decimal,
    pub score: Decimal,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletPerformance {
    pub proxy_wallet: String,
    pub sample_updated_at: DateTime<Utc>,
    pub realized_pnl_usd: Decimal,
    pub total_bought_usd: Decimal,
    pub roi: Decimal,
    pub closed_positions: i32,
    pub winning_positions: i32,
    pub win_rate: Decimal,
    pub rank_score: Decimal,
    pub raw_payload: serde_json::Value,
    pub metadata: serde_json::Value,
}

impl WalletPerformance {
    pub fn with_computed_rank_score(mut self) -> Self {
        self.rank_score = self.realized_pnl_usd * self.roi;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeSignal {
    pub signal_id: Uuid,
    #[serde(default)]
    pub process_id: Option<Uuid>,
    pub timestamp_utc: DateTime<Utc>,
    pub proxy_wallet: String,
    pub wallet_score: Decimal,
    pub source_trade_id: Uuid,
    pub market_id: Option<String>,
    pub token_id: Option<String>,
    pub side: String,
    pub whale_price: Decimal,
    pub observed_price: Decimal,
    pub copy_size_usd: Decimal,
    pub reason: String,
    pub status: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingProcess {
    pub process_id: Uuid,
    pub name: String,
    pub process_type: String,
    pub status: String,
    pub enabled: bool,
    pub config: TradingProcessConfig,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub stopped_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
}

impl TradingProcess {
    pub fn effective_execution(&self) -> EffectiveProcessExecutionConfig {
        self.config.effective_execution()
    }

    pub fn effective_backfill(&self) -> EffectiveProcessBackfillConfig {
        self.config.effective_backfill()
    }

    pub fn effective_copy_trade(&self) -> EffectiveCopyTradeProcessConfig {
        self.config.effective_copy_trade()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TradingProcessConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ProcessExecutionConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub whale: Option<WhaleProcessConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copy_trade: Option<CopyTradeProcessConfig>,
    #[serde(default)]
    pub raw: serde_json::Value,
}

impl TradingProcessConfig {
    pub fn effective_execution(&self) -> EffectiveProcessExecutionConfig {
        let mut effective = EffectiveProcessExecutionConfig::default();
        if let Some(config) = &self.execution {
            if let Some(mode) = &config.mode {
                effective.mode = mode.clone();
            }
            effective.execute_signals = config.execute_signals;
            effective.live_capital = config.live_capital;
            if let Some(taker_fee_rate) = config.taker_fee_rate {
                effective.taker_fee_rate = taker_fee_rate;
            }
        }
        if self
            .execution
            .as_ref()
            .and_then(|config| config.mode.as_ref())
            .is_none()
        {
            if let Some(mode) = raw_string(&self.raw, &["execution", "mode"]) {
                effective.mode = mode.to_string();
            }
        }
        if self.execution.is_none() {
            if let Some(execute_signals) = raw_bool(&self.raw, &["execution", "execute_signals"]) {
                effective.execute_signals = execute_signals;
            }
            if let Some(live_capital) = raw_bool(&self.raw, &["execution", "live_capital"]) {
                effective.live_capital = live_capital;
            }
        }
        if self
            .execution
            .as_ref()
            .and_then(|config| config.taker_fee_rate)
            .is_none()
        {
            if let Some(taker_fee_rate) = raw_decimal(&self.raw, &["execution", "taker_fee_rate"]) {
                effective.taker_fee_rate = taker_fee_rate;
            }
        }
        effective
    }

    pub fn effective_backfill(&self) -> EffectiveProcessBackfillConfig {
        let mut effective = EffectiveProcessBackfillConfig::default();
        if let Some(config) = &self.whale {
            if let Some(backfill_enabled) = config.backfill_enabled {
                effective.backfill_enabled = backfill_enabled;
            }
            if let Some(live_enabled) = config.live_enabled {
                effective.live_enabled = live_enabled;
            }
            if let Some(lookback_days) = config.lookback_days {
                effective.lookback_days = lookback_days;
            }
            if let Some(min_trade_usd) = config.min_trade_usd {
                effective.min_trade_usd = min_trade_usd;
            }
            if let Some(page_limit) = config.page_limit {
                effective.page_limit = page_limit;
            }
            if let Some(max_pages) = config.max_pages {
                effective.max_pages = max_pages;
            }
            if let Some(live_page_limit) = config.live_page_limit {
                effective.live_page_limit = live_page_limit;
            }
            if let Some(live_max_pages) = config.live_max_pages {
                effective.live_max_pages = live_max_pages;
            }
            if let Some(live_poll_interval_secs) = config.live_poll_interval_secs {
                effective.live_poll_interval_secs = live_poll_interval_secs;
            }
            effective.wallets = config.wallets.clone();
            effective.market_ids = config.market_ids.clone();
        }
        if self
            .whale
            .as_ref()
            .and_then(|config| config.backfill_enabled)
            .is_none()
        {
            if let Some(backfill_enabled) = raw_bool(&self.raw, &["whale", "backfill_enabled"]) {
                effective.backfill_enabled = backfill_enabled;
            }
        }
        if self
            .whale
            .as_ref()
            .and_then(|config| config.live_enabled)
            .is_none()
        {
            if let Some(live_enabled) = raw_bool(&self.raw, &["whale", "live_enabled"]) {
                effective.live_enabled = live_enabled;
            }
        }
        if self
            .whale
            .as_ref()
            .and_then(|config| config.lookback_days)
            .is_none()
        {
            if let Some(lookback_days) = raw_u32(&self.raw, &["whale", "lookback_days"]) {
                effective.lookback_days = lookback_days;
            }
        }
        if self
            .whale
            .as_ref()
            .and_then(|config| config.min_trade_usd)
            .is_none()
        {
            if let Some(min_trade_usd) = raw_decimal(&self.raw, &["whale", "min_trade_usd"]) {
                effective.min_trade_usd = min_trade_usd;
            }
        }
        if self
            .whale
            .as_ref()
            .and_then(|config| config.page_limit)
            .is_none()
        {
            if let Some(page_limit) = raw_usize(&self.raw, &["whale", "page_limit"]) {
                effective.page_limit = page_limit;
            }
        }
        if self
            .whale
            .as_ref()
            .and_then(|config| config.max_pages)
            .is_none()
        {
            if let Some(max_pages) = raw_usize(&self.raw, &["whale", "max_pages"]) {
                effective.max_pages = max_pages;
            }
        }
        if self
            .whale
            .as_ref()
            .and_then(|config| config.live_page_limit)
            .is_none()
        {
            if let Some(live_page_limit) = raw_usize(&self.raw, &["whale", "live_page_limit"]) {
                effective.live_page_limit = live_page_limit;
            }
        }
        if self
            .whale
            .as_ref()
            .and_then(|config| config.live_max_pages)
            .is_none()
        {
            if let Some(live_max_pages) = raw_usize(&self.raw, &["whale", "live_max_pages"]) {
                effective.live_max_pages = live_max_pages;
            }
        }
        if self
            .whale
            .as_ref()
            .and_then(|config| config.live_poll_interval_secs)
            .is_none()
        {
            if let Some(live_poll_interval_secs) =
                raw_i64(&self.raw, &["whale", "live_poll_interval_secs"])
            {
                effective.live_poll_interval_secs = live_poll_interval_secs;
            }
        }
        effective
    }

    pub fn effective_copy_trade(&self) -> EffectiveCopyTradeProcessConfig {
        let mut effective = EffectiveCopyTradeProcessConfig::default();
        if let Some(config) = &self.copy_trade {
            if let Some(enabled) = config.enabled {
                effective.enabled = enabled;
            }
            if let Some(min_wallet_score) = config.min_wallet_score {
                effective.min_wallet_score = min_wallet_score;
            }
            if let Some(min_wallet_trades) = config.min_wallet_trades {
                effective.min_wallet_trades = min_wallet_trades;
            }
            if let Some(min_wallet_realized_pnl_usd) = config.min_wallet_realized_pnl_usd {
                effective.min_wallet_realized_pnl_usd = min_wallet_realized_pnl_usd;
            }
            if let Some(min_wallet_roi) = config.min_wallet_roi {
                effective.min_wallet_roi = min_wallet_roi;
            }
            if let Some(min_wallet_closed_positions) = config.min_wallet_closed_positions {
                effective.min_wallet_closed_positions = min_wallet_closed_positions;
            }
            if let Some(min_trade_usd) = config.min_trade_usd {
                effective.min_trade_usd = min_trade_usd;
            }
            if let Some(min_copy_size_usd) = config.min_copy_size_usd {
                effective.min_copy_size_usd = min_copy_size_usd;
            }
            if let Some(max_copy_size_usd) = config.max_copy_size_usd {
                effective.max_copy_size_usd = max_copy_size_usd;
            }
            if let Some(copy_size_fraction) = config.copy_size_fraction {
                effective.copy_size_fraction = copy_size_fraction;
            }
            if let Some(max_follow_lag_secs) = config.max_follow_lag_secs {
                effective.max_follow_lag_secs = max_follow_lag_secs;
            }
            if let Some(max_price_slippage_bps) = config.max_price_slippage_bps {
                effective.max_price_slippage_bps = max_price_slippage_bps;
            }
            if let Some(min_book_depth_usd) = config.min_book_depth_usd {
                effective.min_book_depth_usd = min_book_depth_usd;
            }
            if let Some(allow_sell_entries) = config.allow_sell_entries {
                effective.allow_sell_entries = allow_sell_entries;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.enabled)
            .is_none()
        {
            if let Some(enabled) = raw_bool(&self.raw, &["copy_trade", "enabled"]) {
                effective.enabled = enabled;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.min_wallet_score)
            .is_none()
        {
            if let Some(min_wallet_score) =
                raw_decimal(&self.raw, &["copy_trade", "min_wallet_score"])
            {
                effective.min_wallet_score = min_wallet_score;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.min_wallet_trades)
            .is_none()
        {
            if let Some(min_wallet_trades) =
                raw_i32(&self.raw, &["copy_trade", "min_wallet_trades"])
            {
                effective.min_wallet_trades = min_wallet_trades;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.min_wallet_realized_pnl_usd)
            .is_none()
        {
            if let Some(min_wallet_realized_pnl_usd) =
                raw_decimal(&self.raw, &["copy_trade", "min_wallet_realized_pnl_usd"])
            {
                effective.min_wallet_realized_pnl_usd = min_wallet_realized_pnl_usd;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.min_wallet_roi)
            .is_none()
        {
            if let Some(min_wallet_roi) = raw_decimal(&self.raw, &["copy_trade", "min_wallet_roi"])
            {
                effective.min_wallet_roi = min_wallet_roi;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.min_wallet_closed_positions)
            .is_none()
        {
            if let Some(min_wallet_closed_positions) =
                raw_i32(&self.raw, &["copy_trade", "min_wallet_closed_positions"])
            {
                effective.min_wallet_closed_positions = min_wallet_closed_positions;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.min_trade_usd)
            .is_none()
        {
            if let Some(min_trade_usd) = raw_decimal(&self.raw, &["copy_trade", "min_trade_usd"]) {
                effective.min_trade_usd = min_trade_usd;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.min_copy_size_usd)
            .is_none()
        {
            if let Some(min_copy_size_usd) =
                raw_decimal(&self.raw, &["copy_trade", "min_copy_size_usd"])
            {
                effective.min_copy_size_usd = min_copy_size_usd;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.max_copy_size_usd)
            .is_none()
        {
            if let Some(max_copy_size_usd) =
                raw_decimal(&self.raw, &["copy_trade", "max_copy_size_usd"])
            {
                effective.max_copy_size_usd = max_copy_size_usd;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.copy_size_fraction)
            .is_none()
        {
            if let Some(copy_size_fraction) =
                raw_decimal(&self.raw, &["copy_trade", "copy_size_fraction"])
            {
                effective.copy_size_fraction = copy_size_fraction;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.max_follow_lag_secs)
            .is_none()
        {
            if let Some(max_follow_lag_secs) =
                raw_i64(&self.raw, &["copy_trade", "max_follow_lag_secs"])
            {
                effective.max_follow_lag_secs = max_follow_lag_secs;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.max_price_slippage_bps)
            .is_none()
        {
            if let Some(max_price_slippage_bps) =
                raw_decimal(&self.raw, &["copy_trade", "max_price_slippage_bps"])
            {
                effective.max_price_slippage_bps = max_price_slippage_bps;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.min_book_depth_usd)
            .is_none()
        {
            if let Some(min_book_depth_usd) =
                raw_decimal(&self.raw, &["copy_trade", "min_book_depth_usd"])
            {
                effective.min_book_depth_usd = min_book_depth_usd;
            }
        }
        if self
            .copy_trade
            .as_ref()
            .and_then(|config| config.allow_sell_entries)
            .is_none()
        {
            if let Some(allow_sell_entries) =
                raw_bool(&self.raw, &["copy_trade", "allow_sell_entries"])
            {
                effective.allow_sell_entries = allow_sell_entries;
            }
        }
        if let Some(backtest_horizon_secs) =
            raw_i64(&self.raw, &["copy_trade", "backtest_horizon_secs"])
        {
            effective.backtest_horizon_secs = backtest_horizon_secs;
        }
        if let Some(taker_fee_rate) = raw_decimal(&self.raw, &["copy_trade", "taker_fee_rate"]) {
            effective.taker_fee_rate = taker_fee_rate;
        }
        effective
    }
}

fn raw_value<'a>(value: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
    path.iter().try_fold(value, |current, key| current.get(key))
}

fn raw_bool(value: &serde_json::Value, path: &[&str]) -> Option<bool> {
    raw_value(value, path).and_then(serde_json::Value::as_bool)
}

fn raw_string<'a>(value: &'a serde_json::Value, path: &[&str]) -> Option<&'a str> {
    raw_value(value, path).and_then(serde_json::Value::as_str)
}

fn raw_i32(value: &serde_json::Value, path: &[&str]) -> Option<i32> {
    raw_value(value, path)
        .and_then(serde_json::Value::as_i64)
        .and_then(|value| value.try_into().ok())
}

fn raw_i64(value: &serde_json::Value, path: &[&str]) -> Option<i64> {
    raw_value(value, path).and_then(serde_json::Value::as_i64)
}

fn raw_u32(value: &serde_json::Value, path: &[&str]) -> Option<u32> {
    raw_value(value, path)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| value.try_into().ok())
}

fn raw_usize(value: &serde_json::Value, path: &[&str]) -> Option<usize> {
    raw_value(value, path)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| value.try_into().ok())
}

fn raw_decimal(value: &serde_json::Value, path: &[&str]) -> Option<Decimal> {
    match raw_value(value, path)? {
        serde_json::Value::String(value) => value.parse().ok(),
        serde_json::Value::Number(value) => value.to_string().parse().ok(),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveProcessExecutionConfig {
    pub mode: String,
    pub execute_signals: bool,
    pub live_capital: bool,
    pub taker_fee_rate: Decimal,
}

impl Default for EffectiveProcessExecutionConfig {
    fn default() -> Self {
        Self {
            mode: "sim".to_string(),
            execute_signals: false,
            live_capital: false,
            taker_fee_rate: dec!(0.03),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessExecutionConfig {
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub execute_signals: bool,
    #[serde(default)]
    pub live_capital: bool,
    #[serde(default)]
    pub taker_fee_rate: Option<Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveProcessBackfillConfig {
    pub backfill_enabled: bool,
    pub live_enabled: bool,
    pub lookback_days: u32,
    pub min_trade_usd: Decimal,
    pub page_limit: usize,
    pub max_pages: usize,
    pub live_page_limit: usize,
    pub live_max_pages: usize,
    pub live_poll_interval_secs: i64,
    pub wallets: Vec<String>,
    pub market_ids: Vec<String>,
}

impl Default for EffectiveProcessBackfillConfig {
    fn default() -> Self {
        Self {
            backfill_enabled: true,
            live_enabled: false,
            lookback_days: 30,
            min_trade_usd: dec!(500),
            page_limit: 1000,
            max_pages: 10,
            live_page_limit: 100,
            live_max_pages: 1,
            live_poll_interval_secs: 15,
            wallets: Vec::new(),
            market_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleProcessConfig {
    #[serde(default)]
    pub backfill_enabled: Option<bool>,
    #[serde(default)]
    pub live_enabled: Option<bool>,
    #[serde(default)]
    pub lookback_days: Option<u32>,
    #[serde(default)]
    pub min_trade_usd: Option<Decimal>,
    #[serde(default)]
    pub page_limit: Option<usize>,
    #[serde(default)]
    pub max_pages: Option<usize>,
    #[serde(default)]
    pub live_page_limit: Option<usize>,
    #[serde(default)]
    pub live_max_pages: Option<usize>,
    #[serde(default)]
    pub live_poll_interval_secs: Option<i64>,
    #[serde(default)]
    pub wallets: Vec<String>,
    #[serde(default)]
    pub market_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveCopyTradeProcessConfig {
    pub enabled: bool,
    pub min_wallet_score: Decimal,
    pub min_wallet_trades: i32,
    pub min_wallet_realized_pnl_usd: Decimal,
    pub min_wallet_roi: Decimal,
    pub min_wallet_closed_positions: i32,
    pub min_trade_usd: Decimal,
    pub min_copy_size_usd: Decimal,
    pub max_copy_size_usd: Decimal,
    pub copy_size_fraction: Decimal,
    pub max_follow_lag_secs: i64,
    pub max_price_slippage_bps: Decimal,
    pub min_book_depth_usd: Decimal,
    pub backtest_horizon_secs: i64,
    pub taker_fee_rate: Decimal,
    pub allow_sell_entries: bool,
}

impl Default for EffectiveCopyTradeProcessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_wallet_score: dec!(0),
            min_wallet_trades: 0,
            min_wallet_realized_pnl_usd: dec!(100),
            min_wallet_roi: dec!(0.05),
            min_wallet_closed_positions: 3,
            min_trade_usd: dec!(500),
            min_copy_size_usd: dec!(2),
            max_copy_size_usd: dec!(2),
            copy_size_fraction: dec!(0.10),
            max_follow_lag_secs: 1800,
            max_price_slippage_bps: dec!(150),
            min_book_depth_usd: dec!(25),
            backtest_horizon_secs: 3600,
            taker_fee_rate: dec!(0.03),
            allow_sell_entries: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeProcessConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub min_wallet_score: Option<Decimal>,
    #[serde(default)]
    pub min_wallet_trades: Option<i32>,
    #[serde(default)]
    pub min_wallet_realized_pnl_usd: Option<Decimal>,
    #[serde(default)]
    pub min_wallet_roi: Option<Decimal>,
    #[serde(default)]
    pub min_wallet_closed_positions: Option<i32>,
    #[serde(default)]
    pub min_trade_usd: Option<Decimal>,
    #[serde(default)]
    pub min_copy_size_usd: Option<Decimal>,
    #[serde(default)]
    pub max_copy_size_usd: Option<Decimal>,
    #[serde(default)]
    pub copy_size_fraction: Option<Decimal>,
    #[serde(default)]
    pub max_follow_lag_secs: Option<i64>,
    #[serde(default)]
    pub max_price_slippage_bps: Option<Decimal>,
    #[serde(default)]
    pub min_book_depth_usd: Option<Decimal>,
    #[serde(default)]
    pub backtest_horizon_secs: Option<i64>,
    #[serde(default)]
    pub taker_fee_rate: Option<Decimal>,
    #[serde(default)]
    pub allow_sell_entries: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeBacktestRun {
    pub backtest_id: Uuid,
    pub job_id: Option<Uuid>,
    pub status: String,
    pub score_version: String,
    pub strategy_name: String,
    pub range_start: Option<DateTime<Utc>>,
    pub range_end: Option<DateTime<Utc>>,
    pub config: serde_json::Value,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeBacktestResult {
    pub result_id: Uuid,
    pub backtest_id: Uuid,
    pub timestamp_utc: DateTime<Utc>,
    pub wallet_count: i32,
    pub signal_count: i32,
    pub trade_count: i32,
    pub gross_pnl_usd: Decimal,
    pub net_pnl_usd: Decimal,
    pub roi: Option<Decimal>,
    pub max_drawdown: Option<Decimal>,
    pub win_rate: Option<Decimal>,
    pub result_summary: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletScoreCalibrationSnapshot {
    pub snapshot_id: Uuid,
    pub timestamp_utc: DateTime<Utc>,
    pub score_version: String,
    pub calibration_version: String,
    pub sample_start: Option<DateTime<Utc>>,
    pub sample_end: Option<DateTime<Utc>>,
    pub wallet_count: i32,
    pub trade_count: i32,
    pub feature_weights: serde_json::Value,
    pub thresholds: serde_json::Value,
    pub metrics: serde_json::Value,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhalePollCheckpoint {
    pub checkpoint_name: String,
    pub last_polled_at: Option<DateTime<Utc>>,
    pub next_cursor: Option<String>,
    pub last_trade_timestamp_utc: Option<DateTime<Utc>>,
    pub last_trade_id: Option<Uuid>,
    pub pages_seen: i64,
    pub trades_seen: i64,
    pub state: serde_json::Value,
}
