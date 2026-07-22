use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
            mode: "unspecified".to_string(),
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

#[cfg(test)]
mod tests {
    use super::FillSource;

    #[test]
    fn retired_sim_fill_source_is_not_part_of_the_application_contract() {
        assert!(serde_json::from_str::<FillSource>(r#""sim""#).is_err());
        assert_eq!(
            serde_json::from_str::<FillSource>(r#""paper""#).unwrap(),
            FillSource::Paper
        );
        assert_eq!(
            serde_json::from_str::<FillSource>(r#""live""#).unwrap(),
            FillSource::Live
        );
    }
}
