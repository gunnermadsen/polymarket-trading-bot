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
pub enum OrderIntent {
    Entry,
    Exit,
    RiskReduction,
    AdminManual,
}

impl OrderRequest {
    pub fn intent(&self) -> OrderIntent {
        for key in ["execution_intent", "intent", "purpose"] {
            let Some(value) = self.metadata.get(key).and_then(|value| value.as_str()) else {
                continue;
            };
            match value {
                "entry" | "whale_follow_entry" => return OrderIntent::Entry,
                "exit" | "whale_led_exit" => return OrderIntent::Exit,
                "risk_reduction" | "risk_reduce" => return OrderIntent::RiskReduction,
                "admin_manual" | "manual" => return OrderIntent::AdminManual,
                _ => continue,
            }
        }
        OrderIntent::Entry
    }
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
pub struct GammaMarketMetadata {
    pub cache_key: String,
    pub lookup_type: String,
    pub lookup_slug: String,
    pub event_slug: Option<String>,
    pub market_slug: Option<String>,
    pub gamma_event_id: Option<String>,
    pub gamma_market_id: Option<String>,
    pub category: Option<String>,
    pub series_slug: Option<String>,
    pub tag_slugs: Vec<String>,
    pub sport_key: Option<String>,
    pub taxonomy_segment: Option<String>,
    pub taxonomy_source: String,
    pub taxonomy_confidence: Decimal,
    pub taxonomy_version: String,
    pub raw_payload: serde_json::Value,
    pub fetched_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletTradeTaxonomyCandidate {
    pub trade_id: Uuid,
    pub title: Option<String>,
    pub slug: Option<String>,
    pub event_slug: Option<String>,
    pub market_id: Option<String>,
    pub condition_id: Option<String>,
    pub asset: String,
    pub raw_payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletTradeTaxonomyUpdate {
    pub trade_id: Uuid,
    pub taxonomy_segment: String,
    pub taxonomy_source: String,
    pub taxonomy_confidence: Decimal,
    pub taxonomy_version: String,
    pub taxonomy_metadata: serde_json::Value,
    pub taxonomy_fetched_at: DateTime<Utc>,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WalletScoreRefreshStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletScoreRefreshJob {
    pub queue_id: Uuid,
    pub proxy_wallet: String,
    pub score_version: String,
    pub status: WalletScoreRefreshStatus,
    pub refresh_reason: String,
    pub requested_at: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub attempt_count: i32,
    pub max_attempts: i32,
    pub last_error: Option<String>,
    pub request_metadata: serde_json::Value,
    pub result_metadata: serde_json::Value,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletSegmentPerformance {
    pub proxy_wallet: String,
    pub segment_key: String,
    pub score_version: String,
    pub classifier_version: String,
    pub score: Decimal,
    pub confidence: Decimal,
    pub closed_positions: i32,
    pub winning_positions: i32,
    pub losing_positions: i32,
    pub win_rate: Decimal,
    pub realized_pnl_usd: Decimal,
    pub total_bought_usd: Decimal,
    pub roi: Decimal,
    pub observed_trade_count: i32,
    pub observed_volume_usd: Decimal,
    pub sample_start: Option<DateTime<Utc>>,
    pub sample_end: Option<DateTime<Utc>>,
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
    pub process_scope: String,
    pub process_key: Option<String>,
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

    pub fn effective_expectancy_flow(&self) -> EffectiveExpectancyFlowProcessConfig {
        self.config.effective_expectancy_flow()
    }

    pub fn effective_exit_rules(&self) -> EffectiveProcessExitRulesConfig {
        self.config.effective_exit_rules()
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expectancy_flow: Option<ExpectancyFlowProcessConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_rules: Option<ProcessExitRulesConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mark_refresh: Option<MarkRefreshProcessConfig>,
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
            if let Some(max_open_notional_usd) = config.max_open_notional_usd {
                effective.max_open_notional_usd = max_open_notional_usd;
            }
            if let Some(max_open_notional_per_token_usd) = config.max_open_notional_per_token_usd {
                effective.max_open_notional_per_token_usd = max_open_notional_per_token_usd;
            }
            if let Some(max_open_notional_per_market_usd) = config.max_open_notional_per_market_usd
            {
                effective.max_open_notional_per_market_usd = max_open_notional_per_market_usd;
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
            if let Some(entry_pricing_mode) = &config.entry_pricing_mode {
                effective.entry_pricing_mode = entry_pricing_mode.clone();
            }
            if let Some(min_book_depth_usd) = config.min_book_depth_usd {
                effective.min_book_depth_usd = min_book_depth_usd;
            }
            if let Some(taker_fee_rate) = config.taker_fee_rate {
                effective.taker_fee_rate = taker_fee_rate;
            }
            if let Some(allow_sell_entries) = config.allow_sell_entries {
                effective.allow_sell_entries = allow_sell_entries;
            }
            if let Some(mrs_enabled) = config.mrs_enabled {
                effective.mrs_enabled = mrs_enabled;
            }
            if let Some(mrs_enforce) = config.mrs_enforce {
                effective.mrs_enforce = mrs_enforce;
            }
            if let Some(min_mrs_score) = config.min_mrs_score {
                effective.min_mrs_score = min_mrs_score;
            }
            if let Some(mrs_percentile_floor) = config.mrs_percentile_floor {
                effective.mrs_percentile_floor = mrs_percentile_floor;
            }
            if let Some(mrs_score_version) = &config.mrs_score_version {
                effective.mrs_score_version = mrs_score_version.clone();
            }
            if let Some(segment_scoring_enabled) = config.segment_scoring_enabled {
                effective.segment_scoring_enabled = segment_scoring_enabled;
            }
            if let Some(segment_scoring_mode) = &config.segment_scoring_mode {
                effective.segment_scoring_mode = segment_scoring_mode.clone();
            }
            if let Some(segment_score_version) = &config.segment_score_version {
                effective.segment_score_version = segment_score_version.clone();
            }
            if let Some(segment_classifier_version) = &config.segment_classifier_version {
                effective.segment_classifier_version = segment_classifier_version.clone();
            }
            if let Some(min_segment_score) = config.min_segment_score {
                effective.min_segment_score = min_segment_score;
            }
            if let Some(segment_mrs_percentile_floor) = config.segment_mrs_percentile_floor {
                effective.segment_mrs_percentile_floor = segment_mrs_percentile_floor;
            }
            if let Some(min_segment_confidence) = config.min_segment_confidence {
                effective.min_segment_confidence = min_segment_confidence;
            }
            if let Some(min_segment_closed_positions) = config.min_segment_closed_positions {
                effective.min_segment_closed_positions = min_segment_closed_positions;
            }
            if let Some(min_segment_win_rate) = config.min_segment_win_rate {
                effective.min_segment_win_rate = min_segment_win_rate;
            }
            if let Some(reject_negative_segment_roi_sample_size) =
                config.reject_negative_segment_roi_sample_size
            {
                effective.reject_negative_segment_roi_sample_size =
                    reject_negative_segment_roi_sample_size;
            }
            if let Some(hard_reject_segment_win_rate_below) =
                config.hard_reject_segment_win_rate_below
            {
                effective.hard_reject_segment_win_rate_below = hard_reject_segment_win_rate_below;
            }
            if let Some(hard_reject_segment_sample_size) = config.hard_reject_segment_sample_size {
                effective.hard_reject_segment_sample_size = hard_reject_segment_sample_size;
            }
            if let Some(unknown_segment_policy) = &config.unknown_segment_policy {
                effective.unknown_segment_policy = unknown_segment_policy.clone();
            }
            effective.segment_allowlist = config.segment_allowlist.clone();
            effective.segment_denylist = config.segment_denylist.clone();
            if let Some(entry_safety) = &config.entry_safety {
                if let Some(enabled) = entry_safety.enabled {
                    effective.entry_safety.enabled = enabled;
                }
                if let Some(min_time_to_expiry_secs) = entry_safety.min_time_to_expiry_secs {
                    effective.entry_safety.min_time_to_expiry_secs = min_time_to_expiry_secs;
                }
                if let Some(require_two_sided_book) = entry_safety.require_two_sided_book {
                    effective.entry_safety.require_two_sided_book = require_two_sided_book;
                }
                if let Some(max_spread_bps) = entry_safety.max_spread_bps {
                    effective.entry_safety.max_spread_bps = max_spread_bps;
                }
                if let Some(require_exit_depth) = entry_safety.require_exit_depth {
                    effective.entry_safety.require_exit_depth = require_exit_depth;
                }
                if let Some(exit_depth_size_fraction) = entry_safety.exit_depth_size_fraction {
                    effective.entry_safety.exit_depth_size_fraction = exit_depth_size_fraction;
                }
                if let Some(exit_depth_slippage_bps) = entry_safety.exit_depth_slippage_bps {
                    effective.entry_safety.exit_depth_slippage_bps = exit_depth_slippage_bps;
                }
                if let Some(min_entry_price) = entry_safety.min_entry_price {
                    effective.entry_safety.min_entry_price = min_entry_price;
                }
                if let Some(max_entry_price) = entry_safety.max_entry_price {
                    effective.entry_safety.max_entry_price = max_entry_price;
                }
            }
        }
        effective
    }

    pub fn effective_expectancy_flow(&self) -> EffectiveExpectancyFlowProcessConfig {
        let mut effective = EffectiveExpectancyFlowProcessConfig::default();
        if let Some(config) = &self.expectancy_flow {
            if let Some(enabled) = config.enabled {
                effective.enabled = enabled;
            }
            if let Some(enforce) = config.enforce {
                effective.enforce = enforce;
            }
            if let Some(score_version) = &config.score_version {
                effective.score_version = score_version.clone();
            }
            if let Some(recompute_lookback_days) = config.recompute_lookback_days {
                effective.recompute_lookback_days = recompute_lookback_days;
            }
            if let Some(min_trade_usd) = config.min_trade_usd {
                effective.min_trade_usd = min_trade_usd;
            }
            if let Some(horizon_secs) = config.horizon_secs {
                effective.horizon_secs = horizon_secs;
            }
            if let Some(max_snapshot_lag_secs) = config.max_snapshot_lag_secs {
                effective.max_snapshot_lag_secs = max_snapshot_lag_secs;
            }
            if let Some(price_bucket_bps) = config.price_bucket_bps {
                effective.price_bucket_bps = price_bucket_bps;
            }
            if let Some(min_cell_trades) = config.min_cell_trades {
                effective.min_cell_trades = min_cell_trades;
                effective.min_trades_per_cell = min_cell_trades;
            }
            if let Some(min_trades_per_cell) = config.min_trades_per_cell {
                effective.min_trades_per_cell = min_trades_per_cell;
            }
            if let Some(min_cell_covered) = config.min_cell_covered {
                effective.min_cell_covered = min_cell_covered;
            }
            if let Some(min_win_rate) = config.min_win_rate {
                effective.min_win_rate = min_win_rate;
            }
            if let Some(min_avg_roi) = config.min_avg_roi {
                effective.min_avg_roi = min_avg_roi;
            }
            if let Some(min_median_roi) = config.min_median_roi {
                effective.min_median_roi = min_median_roi;
            }
            if let Some(include_wallet_cells) = config.include_wallet_cells {
                effective.include_wallet_cells = include_wallet_cells;
            }
            if let Some(include_market_dimension) = config.include_market_dimension {
                effective.include_market_dimension = include_market_dimension;
            }
            if let Some(include_taxonomy_segment) = config.include_taxonomy_segment {
                effective.include_taxonomy_segment = include_taxonomy_segment;
            }
            if let Some(max_cells) = config.max_cells {
                effective.max_cells = max_cells;
            }
            effective.allowed_cells = config.allowed_cells.clone();
            effective.denied_cells = config.denied_cells.clone();
            if let Some(wallet_filter) = &config.wallet_filter {
                if let Some(enabled) = wallet_filter.enabled {
                    effective.wallet_filter.enabled = enabled;
                }
                if let Some(min_wallet_cell_covered) = wallet_filter.min_wallet_cell_covered {
                    effective.wallet_filter.min_wallet_cell_covered = min_wallet_cell_covered;
                }
                if let Some(min_wallet_cell_avg_roi) = wallet_filter.min_wallet_cell_avg_roi {
                    effective.wallet_filter.min_wallet_cell_avg_roi = min_wallet_cell_avg_roi;
                }
                if let Some(mrs_enabled) = wallet_filter.mrs_enabled {
                    effective.wallet_filter.mrs_enabled = mrs_enabled;
                }
                if let Some(mrs_enforce) = wallet_filter.mrs_enforce {
                    effective.wallet_filter.mrs_enforce = mrs_enforce;
                }
                if let Some(min_mrs_score) = wallet_filter.min_mrs_score {
                    effective.wallet_filter.min_mrs_score = min_mrs_score;
                }
                if let Some(mrs_percentile_floor) = wallet_filter.mrs_percentile_floor {
                    effective.wallet_filter.mrs_percentile_floor = mrs_percentile_floor;
                }
                if let Some(mrs_score_version) = &wallet_filter.mrs_score_version {
                    effective.wallet_filter.mrs_score_version = mrs_score_version.clone();
                }
            }
        }
        effective
    }

    pub fn effective_exit_rules(&self) -> EffectiveProcessExitRulesConfig {
        let mut effective = EffectiveProcessExitRulesConfig::default();
        if let Some(config) = &self.exit_rules {
            if let Some(take_profit) = &config.take_profit {
                if let Some(take_profit_enabled) = take_profit.take_profit_enabled {
                    effective.take_profit.take_profit_enabled = take_profit_enabled;
                }
                if let Some(take_profit_roi) = take_profit.take_profit_roi {
                    effective.take_profit.take_profit_roi = take_profit_roi;
                }
                if let Some(poll_interval_secs) = take_profit.poll_interval_secs {
                    effective.take_profit.poll_interval_secs = poll_interval_secs;
                }
                if let Some(min_hold_secs) = take_profit.min_hold_secs {
                    effective.take_profit.min_hold_secs = min_hold_secs;
                }
                if let Some(require_fresh_mark_secs) = take_profit.require_fresh_mark_secs {
                    effective.take_profit.require_fresh_mark_secs = require_fresh_mark_secs;
                }
                if let Some(exit_size_fraction) = take_profit.exit_size_fraction {
                    effective.take_profit.exit_size_fraction = exit_size_fraction;
                }
                if let Some(max_exit_slippage_bps) = take_profit.max_exit_slippage_bps {
                    effective.take_profit.max_exit_slippage_bps = max_exit_slippage_bps;
                }
                if let Some(exit_pricing_mode) = &take_profit.exit_pricing_mode {
                    effective.take_profit.exit_pricing_mode = exit_pricing_mode.clone();
                }
            }
            if let Some(stop_loss) = &config.stop_loss {
                if let Some(stop_loss_enabled) = stop_loss.stop_loss_enabled {
                    effective.stop_loss.stop_loss_enabled = stop_loss_enabled;
                }
                if let Some(stop_loss_roi) = stop_loss.stop_loss_roi {
                    effective.stop_loss.stop_loss_roi = stop_loss_roi;
                }
                if let Some(poll_interval_secs) = stop_loss.poll_interval_secs {
                    effective.stop_loss.poll_interval_secs = poll_interval_secs;
                }
                if let Some(min_hold_secs) = stop_loss.min_hold_secs {
                    effective.stop_loss.min_hold_secs = min_hold_secs;
                }
                if let Some(require_fresh_mark_secs) = stop_loss.require_fresh_mark_secs {
                    effective.stop_loss.require_fresh_mark_secs = require_fresh_mark_secs;
                }
                if let Some(exit_size_fraction) = stop_loss.exit_size_fraction {
                    effective.stop_loss.exit_size_fraction = exit_size_fraction;
                }
                if let Some(max_exit_slippage_bps) = stop_loss.max_exit_slippage_bps {
                    effective.stop_loss.max_exit_slippage_bps = max_exit_slippage_bps;
                }
                if let Some(exit_pricing_mode) = &stop_loss.exit_pricing_mode {
                    effective.stop_loss.exit_pricing_mode = exit_pricing_mode.clone();
                }
            }
        }
        effective
    }

    pub fn effective_mark_refresh(&self) -> EffectiveMarkRefreshProcessConfig {
        let mut effective = EffectiveMarkRefreshProcessConfig::default();
        if let Some(config) = &self.mark_refresh {
            if let Some(enabled) = config.enabled {
                effective.enabled = enabled;
            }
            if let Some(poll_interval_secs) = config.poll_interval_secs {
                effective.poll_interval_secs = poll_interval_secs;
            }
            if let Some(max_mark_age_secs) = config.max_mark_age_secs {
                effective.max_mark_age_secs = max_mark_age_secs;
            }
            if let Some(batch_size) = config.batch_size {
                effective.batch_size = batch_size;
            }
            if let Some(stale_only) = config.stale_only {
                effective.stale_only = stale_only;
            }
            if let Some(failure_backoff_secs) = config.failure_backoff_secs {
                effective.failure_backoff_secs = failure_backoff_secs;
            }
        }
        effective
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
    pub max_open_notional_usd: Decimal,
    pub max_open_notional_per_token_usd: Decimal,
    pub max_open_notional_per_market_usd: Decimal,
    pub copy_size_fraction: Decimal,
    pub max_follow_lag_secs: i64,
    pub max_price_slippage_bps: Decimal,
    pub entry_pricing_mode: String,
    pub min_book_depth_usd: Decimal,
    pub taker_fee_rate: Decimal,
    pub allow_sell_entries: bool,
    pub mrs_enabled: bool,
    pub mrs_enforce: bool,
    pub min_mrs_score: Decimal,
    pub mrs_percentile_floor: Decimal,
    pub mrs_score_version: String,
    pub segment_scoring_enabled: bool,
    pub segment_scoring_mode: String,
    pub segment_score_version: String,
    pub segment_classifier_version: String,
    pub min_segment_score: Decimal,
    pub segment_mrs_percentile_floor: Decimal,
    pub min_segment_confidence: Decimal,
    pub min_segment_closed_positions: i32,
    pub min_segment_win_rate: Decimal,
    pub reject_negative_segment_roi_sample_size: i32,
    pub hard_reject_segment_win_rate_below: Decimal,
    pub hard_reject_segment_sample_size: i32,
    pub unknown_segment_policy: String,
    pub segment_allowlist: Vec<String>,
    pub segment_denylist: Vec<String>,
    pub entry_safety: EffectiveEntrySafetyProcessConfig,
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
            max_open_notional_usd: dec!(20),
            max_open_notional_per_token_usd: Decimal::ZERO,
            max_open_notional_per_market_usd: Decimal::ZERO,
            copy_size_fraction: dec!(0.10),
            max_follow_lag_secs: 1800,
            max_price_slippage_bps: dec!(150),
            entry_pricing_mode: "signal_limit".to_string(),
            min_book_depth_usd: dec!(25),
            taker_fee_rate: dec!(0.03),
            allow_sell_entries: false,
            mrs_enabled: true,
            mrs_enforce: false,
            min_mrs_score: dec!(80),
            mrs_percentile_floor: dec!(0.80),
            mrs_score_version: "mrs_v1".to_string(),
            segment_scoring_enabled: false,
            segment_scoring_mode: "shadow".to_string(),
            segment_score_version: "mrs_segment_v1".to_string(),
            segment_classifier_version: "gamma_taxonomy_v1".to_string(),
            min_segment_score: dec!(50),
            segment_mrs_percentile_floor: dec!(0.95),
            min_segment_confidence: dec!(0.10),
            min_segment_closed_positions: 5,
            min_segment_win_rate: dec!(0.52),
            reject_negative_segment_roi_sample_size: 5,
            hard_reject_segment_win_rate_below: dec!(0.40),
            hard_reject_segment_sample_size: 10,
            unknown_segment_policy: "neutral".to_string(),
            segment_allowlist: Vec::new(),
            segment_denylist: Vec::new(),
            entry_safety: EffectiveEntrySafetyProcessConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveEntrySafetyProcessConfig {
    pub enabled: bool,
    pub min_time_to_expiry_secs: i64,
    pub require_two_sided_book: bool,
    pub max_spread_bps: Decimal,
    pub require_exit_depth: bool,
    pub exit_depth_size_fraction: Decimal,
    pub exit_depth_slippage_bps: Decimal,
    pub min_entry_price: Decimal,
    pub max_entry_price: Decimal,
}

impl Default for EffectiveEntrySafetyProcessConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_time_to_expiry_secs: 0,
            require_two_sided_book: false,
            max_spread_bps: Decimal::ZERO,
            require_exit_depth: false,
            exit_depth_size_fraction: dec!(1.0),
            exit_depth_slippage_bps: dec!(150),
            min_entry_price: Decimal::ZERO,
            max_entry_price: dec!(1.0),
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
    pub max_open_notional_usd: Option<Decimal>,
    #[serde(default)]
    pub max_open_notional_per_token_usd: Option<Decimal>,
    #[serde(default)]
    pub max_open_notional_per_market_usd: Option<Decimal>,
    #[serde(default)]
    pub copy_size_fraction: Option<Decimal>,
    #[serde(default)]
    pub max_follow_lag_secs: Option<i64>,
    #[serde(default)]
    pub max_price_slippage_bps: Option<Decimal>,
    #[serde(default)]
    pub entry_pricing_mode: Option<String>,
    #[serde(default)]
    pub min_book_depth_usd: Option<Decimal>,
    #[serde(default)]
    pub taker_fee_rate: Option<Decimal>,
    #[serde(default)]
    pub allow_sell_entries: Option<bool>,
    #[serde(default)]
    pub mrs_enabled: Option<bool>,
    #[serde(default)]
    pub mrs_enforce: Option<bool>,
    #[serde(default)]
    pub min_mrs_score: Option<Decimal>,
    #[serde(default)]
    pub mrs_percentile_floor: Option<Decimal>,
    #[serde(default)]
    pub mrs_score_version: Option<String>,
    #[serde(default)]
    pub segment_scoring_enabled: Option<bool>,
    #[serde(default)]
    pub segment_scoring_mode: Option<String>,
    #[serde(default)]
    pub segment_score_version: Option<String>,
    #[serde(default)]
    pub segment_classifier_version: Option<String>,
    #[serde(default)]
    pub min_segment_score: Option<Decimal>,
    #[serde(default)]
    pub segment_mrs_percentile_floor: Option<Decimal>,
    #[serde(default)]
    pub min_segment_confidence: Option<Decimal>,
    #[serde(default)]
    pub min_segment_closed_positions: Option<i32>,
    #[serde(default)]
    pub min_segment_win_rate: Option<Decimal>,
    #[serde(default)]
    pub reject_negative_segment_roi_sample_size: Option<i32>,
    #[serde(default)]
    pub hard_reject_segment_win_rate_below: Option<Decimal>,
    #[serde(default)]
    pub hard_reject_segment_sample_size: Option<i32>,
    #[serde(default)]
    pub unknown_segment_policy: Option<String>,
    #[serde(default)]
    pub segment_allowlist: Vec<String>,
    #[serde(default)]
    pub segment_denylist: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_safety: Option<EntrySafetyProcessConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveExpectancyFlowProcessConfig {
    pub enabled: bool,
    pub enforce: bool,
    pub score_version: String,
    pub recompute_lookback_days: i32,
    pub min_trade_usd: Decimal,
    pub horizon_secs: i64,
    pub max_snapshot_lag_secs: i64,
    pub price_bucket_bps: i32,
    pub min_cell_trades: i32,
    pub min_trades_per_cell: i32,
    pub min_cell_covered: i32,
    pub min_win_rate: Decimal,
    pub min_avg_roi: Decimal,
    pub min_median_roi: Decimal,
    pub include_wallet_cells: bool,
    pub include_market_dimension: bool,
    pub include_taxonomy_segment: bool,
    pub max_cells: i64,
    pub allowed_cells: Vec<String>,
    pub denied_cells: Vec<String>,
    pub wallet_filter: EffectiveExpectancyWalletFilterProcessConfig,
}

impl Default for EffectiveExpectancyFlowProcessConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            enforce: false,
            score_version: "expectancy_flow_v1".to_string(),
            recompute_lookback_days: 30,
            min_trade_usd: dec!(500),
            horizon_secs: 3600,
            max_snapshot_lag_secs: 900,
            price_bucket_bps: 2000,
            min_cell_trades: 50,
            min_trades_per_cell: 10,
            min_cell_covered: 25,
            min_win_rate: dec!(0.55),
            min_avg_roi: dec!(0.02),
            min_median_roi: Decimal::ZERO,
            include_wallet_cells: false,
            include_market_dimension: false,
            include_taxonomy_segment: true,
            max_cells: 1000,
            allowed_cells: Vec::new(),
            denied_cells: Vec::new(),
            wallet_filter: EffectiveExpectancyWalletFilterProcessConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveExpectancyWalletFilterProcessConfig {
    pub enabled: bool,
    pub min_wallet_cell_covered: i32,
    pub min_wallet_cell_avg_roi: Decimal,
    pub mrs_enabled: bool,
    pub mrs_enforce: bool,
    pub min_mrs_score: Decimal,
    pub mrs_percentile_floor: Decimal,
    pub mrs_score_version: String,
}

impl Default for EffectiveExpectancyWalletFilterProcessConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_wallet_cell_covered: 3,
            min_wallet_cell_avg_roi: Decimal::ZERO,
            mrs_enabled: true,
            mrs_enforce: false,
            min_mrs_score: dec!(80),
            mrs_percentile_floor: dec!(0.80),
            mrs_score_version: "mrs_v1".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectancyFlowProcessConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub enforce: Option<bool>,
    #[serde(default)]
    pub score_version: Option<String>,
    #[serde(default)]
    pub recompute_lookback_days: Option<i32>,
    #[serde(default)]
    pub min_trade_usd: Option<Decimal>,
    #[serde(default)]
    pub horizon_secs: Option<i64>,
    #[serde(default)]
    pub max_snapshot_lag_secs: Option<i64>,
    #[serde(default)]
    pub price_bucket_bps: Option<i32>,
    #[serde(default)]
    pub min_cell_trades: Option<i32>,
    #[serde(default)]
    pub min_trades_per_cell: Option<i32>,
    #[serde(default)]
    pub min_cell_covered: Option<i32>,
    #[serde(default)]
    pub min_win_rate: Option<Decimal>,
    #[serde(default)]
    pub min_avg_roi: Option<Decimal>,
    #[serde(default)]
    pub min_median_roi: Option<Decimal>,
    #[serde(default)]
    pub include_wallet_cells: Option<bool>,
    #[serde(default)]
    pub include_market_dimension: Option<bool>,
    #[serde(default)]
    pub include_taxonomy_segment: Option<bool>,
    #[serde(default)]
    pub max_cells: Option<i64>,
    #[serde(default)]
    pub allowed_cells: Vec<String>,
    #[serde(default)]
    pub denied_cells: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wallet_filter: Option<ExpectancyWalletFilterProcessConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectancyWalletFilterProcessConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub min_wallet_cell_covered: Option<i32>,
    #[serde(default)]
    pub min_wallet_cell_avg_roi: Option<Decimal>,
    #[serde(default)]
    pub mrs_enabled: Option<bool>,
    #[serde(default)]
    pub mrs_enforce: Option<bool>,
    #[serde(default)]
    pub min_mrs_score: Option<Decimal>,
    #[serde(default)]
    pub mrs_percentile_floor: Option<Decimal>,
    #[serde(default)]
    pub mrs_score_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntrySafetyProcessConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub min_time_to_expiry_secs: Option<i64>,
    #[serde(default)]
    pub require_two_sided_book: Option<bool>,
    #[serde(default)]
    pub max_spread_bps: Option<Decimal>,
    #[serde(default)]
    pub require_exit_depth: Option<bool>,
    #[serde(default)]
    pub exit_depth_size_fraction: Option<Decimal>,
    #[serde(default)]
    pub exit_depth_slippage_bps: Option<Decimal>,
    #[serde(default)]
    pub min_entry_price: Option<Decimal>,
    #[serde(default)]
    pub max_entry_price: Option<Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveProcessExitRulesConfig {
    pub take_profit: EffectiveTakeProfitExitRuleProcessConfig,
    pub stop_loss: EffectiveStopLossExitRuleProcessConfig,
}

impl Default for EffectiveProcessExitRulesConfig {
    fn default() -> Self {
        Self {
            take_profit: EffectiveTakeProfitExitRuleProcessConfig::default(),
            stop_loss: EffectiveStopLossExitRuleProcessConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveTakeProfitExitRuleProcessConfig {
    pub take_profit_enabled: bool,
    pub take_profit_roi: Decimal,
    pub poll_interval_secs: i64,
    pub min_hold_secs: i64,
    pub require_fresh_mark_secs: i64,
    pub exit_size_fraction: Decimal,
    pub max_exit_slippage_bps: Decimal,
    pub exit_pricing_mode: String,
}

impl Default for EffectiveTakeProfitExitRuleProcessConfig {
    fn default() -> Self {
        Self {
            take_profit_enabled: false,
            take_profit_roi: dec!(0.10),
            poll_interval_secs: 10,
            min_hold_secs: 60,
            require_fresh_mark_secs: 60,
            exit_size_fraction: dec!(1.0),
            max_exit_slippage_bps: dec!(150),
            exit_pricing_mode: "mark_limit".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessExitRulesConfig {
    #[serde(
        default,
        alias = "take-profit",
        skip_serializing_if = "Option::is_none"
    )]
    pub take_profit: Option<TakeProfitExitRuleProcessConfig>,
    #[serde(default, alias = "stop-loss", skip_serializing_if = "Option::is_none")]
    pub stop_loss: Option<StopLossExitRuleProcessConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TakeProfitExitRuleProcessConfig {
    #[serde(default)]
    pub take_profit_enabled: Option<bool>,
    #[serde(default)]
    pub take_profit_roi: Option<Decimal>,
    #[serde(default)]
    pub poll_interval_secs: Option<i64>,
    #[serde(default)]
    pub min_hold_secs: Option<i64>,
    #[serde(default)]
    pub require_fresh_mark_secs: Option<i64>,
    #[serde(default)]
    pub exit_size_fraction: Option<Decimal>,
    #[serde(default)]
    pub max_exit_slippage_bps: Option<Decimal>,
    #[serde(default)]
    pub exit_pricing_mode: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveStopLossExitRuleProcessConfig {
    pub stop_loss_enabled: bool,
    pub stop_loss_roi: Decimal,
    pub poll_interval_secs: i64,
    pub min_hold_secs: i64,
    pub require_fresh_mark_secs: i64,
    pub exit_size_fraction: Decimal,
    pub max_exit_slippage_bps: Decimal,
    pub exit_pricing_mode: String,
}

impl Default for EffectiveStopLossExitRuleProcessConfig {
    fn default() -> Self {
        Self {
            stop_loss_enabled: false,
            stop_loss_roi: dec!(-0.10),
            poll_interval_secs: 10,
            min_hold_secs: 60,
            require_fresh_mark_secs: 60,
            exit_size_fraction: dec!(1.0),
            max_exit_slippage_bps: dec!(150),
            exit_pricing_mode: "mark_limit".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopLossExitRuleProcessConfig {
    #[serde(default)]
    pub stop_loss_enabled: Option<bool>,
    #[serde(default)]
    pub stop_loss_roi: Option<Decimal>,
    #[serde(default)]
    pub poll_interval_secs: Option<i64>,
    #[serde(default)]
    pub min_hold_secs: Option<i64>,
    #[serde(default)]
    pub require_fresh_mark_secs: Option<i64>,
    #[serde(default)]
    pub exit_size_fraction: Option<Decimal>,
    #[serde(default)]
    pub max_exit_slippage_bps: Option<Decimal>,
    #[serde(default)]
    pub exit_pricing_mode: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveMarkRefreshProcessConfig {
    pub enabled: bool,
    pub poll_interval_secs: i64,
    pub max_mark_age_secs: i64,
    pub batch_size: i64,
    pub stale_only: bool,
    pub failure_backoff_secs: i64,
}

impl Default for EffectiveMarkRefreshProcessConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            poll_interval_secs: 15,
            max_mark_age_secs: 60,
            batch_size: 50,
            stale_only: true,
            failure_backoff_secs: 300,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkRefreshProcessConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub poll_interval_secs: Option<i64>,
    #[serde(default)]
    pub max_mark_age_secs: Option<i64>,
    #[serde(default)]
    pub batch_size: Option<i64>,
    #[serde(default)]
    pub stale_only: Option<bool>,
    #[serde(default)]
    pub failure_backoff_secs: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectancyFlowCell {
    pub process_id: Uuid,
    pub score_version: String,
    pub cell_key: String,
    pub dimensions: serde_json::Value,
    pub horizon_secs: i64,
    pub lookback_days: i32,
    pub sample_count: i32,
    pub winning_count: i32,
    pub losing_count: i32,
    pub observed_volume_usd: Decimal,
    pub realized_pnl_usd: Decimal,
    pub mean_price_delta: Decimal,
    pub mean_return: Decimal,
    pub win_rate: Decimal,
    pub expectancy: Decimal,
    pub confidence: Decimal,
    pub sample_start: Option<DateTime<Utc>>,
    pub sample_end: Option<DateTime<Utc>>,
    pub metadata: serde_json::Value,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectancyFlowWalletCell {
    pub process_id: Uuid,
    pub score_version: String,
    pub proxy_wallet: String,
    pub cell_key: String,
    pub dimensions: serde_json::Value,
    pub horizon_secs: i64,
    pub lookback_days: i32,
    pub sample_count: i32,
    pub winning_count: i32,
    pub losing_count: i32,
    pub observed_volume_usd: Decimal,
    pub realized_pnl_usd: Decimal,
    pub mean_price_delta: Decimal,
    pub mean_return: Decimal,
    pub win_rate: Decimal,
    pub expectancy: Decimal,
    pub confidence: Decimal,
    pub sample_start: Option<DateTime<Utc>>,
    pub sample_end: Option<DateTime<Utc>>,
    pub metadata: serde_json::Value,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExpectancyFlowRecomputeReport {
    pub process_id: Uuid,
    pub score_version: String,
    pub cells_recomputed: u64,
    pub wallet_cells_recomputed: u64,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_refresh_defaults_are_disabled_and_conservative() {
        let effective = TradingProcessConfig::default().effective_mark_refresh();

        assert!(!effective.enabled);
        assert_eq!(effective.poll_interval_secs, 15);
        assert_eq!(effective.max_mark_age_secs, 60);
        assert_eq!(effective.batch_size, 50);
        assert!(effective.stale_only);
        assert_eq!(effective.failure_backoff_secs, 300);
    }

    #[test]
    fn mark_refresh_config_overrides_defaults() {
        let config: TradingProcessConfig = serde_json::from_value(serde_json::json!({
            "mark_refresh": {
                "enabled": true,
                "poll_interval_secs": 5,
                "max_mark_age_secs": 30,
                "batch_size": 12,
                "stale_only": false,
                "failure_backoff_secs": 45
            }
        }))
        .expect("mark refresh config should deserialize");

        let effective = config.effective_mark_refresh();
        assert!(effective.enabled);
        assert_eq!(effective.poll_interval_secs, 5);
        assert_eq!(effective.max_mark_age_secs, 30);
        assert_eq!(effective.batch_size, 12);
        assert!(!effective.stale_only);
        assert_eq!(effective.failure_backoff_secs, 45);
    }

    #[test]
    fn expectancy_flow_defaults_are_disabled_and_process_configurable() {
        let effective = TradingProcessConfig::default().effective_expectancy_flow();

        assert!(!effective.enabled);
        assert_eq!(effective.score_version, "expectancy_flow_v1");
        assert_eq!(effective.recompute_lookback_days, 30);
        assert_eq!(effective.horizon_secs, 3600);
        assert_eq!(effective.max_snapshot_lag_secs, 900);
        assert_eq!(effective.price_bucket_bps, 2000);
        assert_eq!(effective.min_trades_per_cell, 10);
        assert!(!effective.include_wallet_cells);
        assert!(!effective.include_market_dimension);
        assert!(effective.include_taxonomy_segment);
        assert_eq!(effective.max_cells, 1000);
    }

    #[test]
    fn expectancy_flow_storage_config_overrides_defaults() {
        let config: TradingProcessConfig = serde_json::from_value(serde_json::json!({
            "expectancy_flow": {
                "enabled": true,
                "score_version": "expectancy_flow_v2",
                "recompute_lookback_days": 14,
                "horizon_secs": 7200,
                "max_snapshot_lag_secs": 300,
                "price_bucket_bps": 250,
                "min_trades_per_cell": 5,
                "include_wallet_cells": true,
                "include_market_dimension": true,
                "include_taxonomy_segment": false,
                "max_cells": 250
            }
        }))
        .expect("expectancy flow config should deserialize");

        let effective = config.effective_expectancy_flow();
        assert!(effective.enabled);
        assert_eq!(effective.score_version, "expectancy_flow_v2");
        assert_eq!(effective.recompute_lookback_days, 14);
        assert_eq!(effective.horizon_secs, 7200);
        assert_eq!(effective.max_snapshot_lag_secs, 300);
        assert_eq!(effective.price_bucket_bps, 250);
        assert_eq!(effective.min_trades_per_cell, 5);
        assert!(effective.include_wallet_cells);
        assert!(effective.include_market_dimension);
        assert!(!effective.include_taxonomy_segment);
        assert_eq!(effective.max_cells, 250);
    }

    #[test]
    fn copy_trade_entry_safety_config_overrides_defaults() {
        let config: TradingProcessConfig = serde_json::from_value(serde_json::json!({
            "copy_trade": {
                "entry_safety": {
                    "enabled": true,
                    "min_time_to_expiry_secs": 120,
                    "require_two_sided_book": true,
                    "max_spread_bps": "2500",
                    "require_exit_depth": true,
                    "exit_depth_size_fraction": "1.0",
                    "exit_depth_slippage_bps": "150",
                    "min_entry_price": "0.05",
                    "max_entry_price": "0.95"
                }
            }
        }))
        .expect("entry safety config should deserialize");

        let effective = config.effective_copy_trade();
        assert!(effective.entry_safety.enabled);
        assert_eq!(effective.entry_safety.min_time_to_expiry_secs, 120);
        assert!(effective.entry_safety.require_two_sided_book);
        assert_eq!(effective.entry_safety.max_spread_bps, dec!(2500));
        assert!(effective.entry_safety.require_exit_depth);
        assert_eq!(effective.entry_safety.exit_depth_size_fraction, dec!(1.0));
        assert_eq!(effective.entry_safety.exit_depth_slippage_bps, dec!(150));
        assert_eq!(effective.entry_safety.min_entry_price, dec!(0.05));
        assert_eq!(effective.entry_safety.max_entry_price, dec!(0.95));
    }

    #[test]
    fn copy_trade_entry_pricing_mode_overrides_default() {
        let default = TradingProcessConfig::default().effective_copy_trade();
        assert_eq!(default.entry_pricing_mode, "signal_limit");

        let config: TradingProcessConfig = serde_json::from_value(serde_json::json!({
            "copy_trade": {
                "entry_pricing_mode": "marketable_limit"
            }
        }))
        .expect("entry pricing config should deserialize");

        let effective = config.effective_copy_trade();
        assert_eq!(effective.entry_pricing_mode, "marketable_limit");
    }

    #[test]
    fn copy_trade_segment_lists_override_defaults() {
        let default = TradingProcessConfig::default().effective_copy_trade();
        assert!(default.segment_allowlist.is_empty());
        assert!(default.segment_denylist.is_empty());

        let config: TradingProcessConfig = serde_json::from_value(serde_json::json!({
            "copy_trade": {
                "segment_allowlist": ["politics.general"],
                "segment_denylist": ["crypto.bitcoin.short_interval", "sports.general"]
            }
        }))
        .expect("segment list config should deserialize");

        let effective = config.effective_copy_trade();
        assert_eq!(effective.segment_allowlist, vec!["politics.general"]);
        assert_eq!(
            effective.segment_denylist,
            vec!["crypto.bitcoin.short_interval", "sports.general"]
        );
    }

    #[test]
    fn expectancy_flow_config_overrides_defaults() {
        let default = TradingProcessConfig::default().effective_expectancy_flow();
        assert!(!default.enabled);
        assert!(!default.enforce);
        assert_eq!(default.min_trade_usd, dec!(500));
        assert_eq!(default.horizon_secs, 3600);
        assert!(!default.wallet_filter.enabled);
        assert!(default.wallet_filter.mrs_enabled);
        assert!(!default.wallet_filter.mrs_enforce);
        assert_eq!(default.wallet_filter.min_mrs_score, dec!(80));
        assert_eq!(default.wallet_filter.mrs_percentile_floor, dec!(0.80));
        assert_eq!(default.wallet_filter.mrs_score_version, "mrs_v1");

        let config: TradingProcessConfig = serde_json::from_value(serde_json::json!({
            "expectancy_flow": {
                "enabled": true,
                "enforce": true,
                "min_trade_usd": "250",
                "horizon_secs": 1800,
                "min_cell_trades": 75,
                "min_cell_covered": 30,
                "min_win_rate": "0.58",
                "min_avg_roi": "0.03",
                "min_median_roi": "0.01",
                "allowed_cells": ["sports.nba|buy|60-80c|3600"],
                "denied_cells": ["other|buy|60-80c|3600"],
                "wallet_filter": {
                    "enabled": true,
                    "min_wallet_cell_covered": 5,
                    "min_wallet_cell_avg_roi": "0.02",
                    "mrs_enabled": false,
                    "mrs_enforce": true,
                    "min_mrs_score": "92.5",
                    "mrs_percentile_floor": "0.91",
                    "mrs_score_version": "expectancy_flow_mrs_v2"
                }
            }
        }))
        .expect("expectancy flow config should deserialize");

        let effective = config.effective_expectancy_flow();
        assert!(effective.enabled);
        assert!(effective.enforce);
        assert_eq!(effective.min_trade_usd, dec!(250));
        assert_eq!(effective.horizon_secs, 1800);
        assert_eq!(effective.min_cell_trades, 75);
        assert_eq!(effective.min_cell_covered, 30);
        assert_eq!(effective.min_win_rate, dec!(0.58));
        assert_eq!(effective.min_avg_roi, dec!(0.03));
        assert_eq!(effective.min_median_roi, dec!(0.01));
        assert_eq!(effective.allowed_cells, vec!["sports.nba|buy|60-80c|3600"]);
        assert_eq!(effective.denied_cells, vec!["other|buy|60-80c|3600"]);
        assert!(effective.wallet_filter.enabled);
        assert_eq!(effective.wallet_filter.min_wallet_cell_covered, 5);
        assert_eq!(effective.wallet_filter.min_wallet_cell_avg_roi, dec!(0.02));
        assert!(!effective.wallet_filter.mrs_enabled);
        assert!(effective.wallet_filter.mrs_enforce);
        assert_eq!(effective.wallet_filter.min_mrs_score, dec!(92.5));
        assert_eq!(effective.wallet_filter.mrs_percentile_floor, dec!(0.91));
        assert_eq!(
            effective.wallet_filter.mrs_score_version,
            "expectancy_flow_mrs_v2"
        );
    }

    #[test]
    fn expectancy_flow_mrs_is_independent_from_copy_trade_mrs() {
        let copy_only_config: TradingProcessConfig = serde_json::from_value(serde_json::json!({
            "copy_trade": {
                "mrs_enabled": false,
                "mrs_enforce": true,
                "min_mrs_score": "10",
                "mrs_percentile_floor": "0.10",
                "mrs_score_version": "copy_trade_mrs"
            }
        }))
        .expect("copy-trade-only config should deserialize");

        let copy_only_expectancy_flow = copy_only_config.effective_expectancy_flow();
        assert!(copy_only_expectancy_flow.wallet_filter.mrs_enabled);
        assert!(!copy_only_expectancy_flow.wallet_filter.mrs_enforce);
        assert_eq!(
            copy_only_expectancy_flow.wallet_filter.min_mrs_score,
            dec!(80)
        );
        assert_eq!(
            copy_only_expectancy_flow.wallet_filter.mrs_percentile_floor,
            dec!(0.80)
        );
        assert_eq!(
            copy_only_expectancy_flow.wallet_filter.mrs_score_version,
            "mrs_v1"
        );

        let config: TradingProcessConfig = serde_json::from_value(serde_json::json!({
            "copy_trade": {
                "mrs_enabled": false,
                "mrs_enforce": true,
                "min_mrs_score": "10",
                "mrs_percentile_floor": "0.10",
                "mrs_score_version": "copy_trade_mrs"
            },
            "expectancy_flow": {
                "wallet_filter": {
                    "mrs_enabled": true,
                    "mrs_enforce": false,
                    "min_mrs_score": "95",
                    "mrs_percentile_floor": "0.95",
                    "mrs_score_version": "expectancy_flow_mrs"
                }
            }
        }))
        .expect("combined strategy config should deserialize");

        let copy_trade = config.effective_copy_trade();
        let expectancy_flow = config.effective_expectancy_flow();

        assert!(!copy_trade.mrs_enabled);
        assert!(copy_trade.mrs_enforce);
        assert_eq!(copy_trade.min_mrs_score, dec!(10));
        assert_eq!(copy_trade.mrs_percentile_floor, dec!(0.10));
        assert_eq!(copy_trade.mrs_score_version, "copy_trade_mrs");

        assert!(expectancy_flow.wallet_filter.mrs_enabled);
        assert!(!expectancy_flow.wallet_filter.mrs_enforce);
        assert_eq!(expectancy_flow.wallet_filter.min_mrs_score, dec!(95));
        assert_eq!(
            expectancy_flow.wallet_filter.mrs_percentile_floor,
            dec!(0.95)
        );
        assert_eq!(
            expectancy_flow.wallet_filter.mrs_score_version,
            "expectancy_flow_mrs"
        );
    }
}
