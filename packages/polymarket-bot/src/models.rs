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
                "entry" => return OrderIntent::Entry,
                "exit" => return OrderIntent::Exit,
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
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TradingProcessConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ProcessExecutionConfig>,
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
