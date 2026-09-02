use std::{fmt, str::FromStr};

use anyhow::{bail, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::FromRow;
use uuid::Uuid;

pub const KRAKEN_PROVIDER: &str = "kraken_futures";
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KrakenDataset {
    Instruments,
    FeeSchedules,
    TradeCandles,
    MarkCandles,
    SpotCandles,
    OpenInterest,
    FutureBasis,
    AggressorDifferential,
    TradeVolume,
    TradeCount,
    Cvd,
    LiquidationVolume,
    Spreads,
    Liquidity,
    Slippage,
    FundingRates,
}

impl KrakenDataset {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Instruments => "instruments",
            Self::FeeSchedules => "fee_schedules",
            Self::TradeCandles => "trade_candles",
            Self::MarkCandles => "mark_candles",
            Self::SpotCandles => "spot_candles",
            Self::OpenInterest => "open_interest",
            Self::FutureBasis => "future_basis",
            Self::AggressorDifferential => "aggressor_differential",
            Self::TradeVolume => "trade_volume",
            Self::TradeCount => "trade_count",
            Self::Cvd => "cvd",
            Self::LiquidationVolume => "liquidation_volume",
            Self::Spreads => "spreads",
            Self::Liquidity => "liquidity",
            Self::Slippage => "slippage",
            Self::FundingRates => "funding_rates",
        }
    }

    pub const fn analytics_slug(self) -> Option<&'static str> {
        match self {
            Self::OpenInterest => Some("open-interest"),
            Self::FutureBasis => Some("future-basis"),
            Self::AggressorDifferential => Some("aggressor-differential"),
            Self::TradeVolume => Some("trade-volume"),
            Self::TradeCount => Some("trade-count"),
            Self::Cvd => Some("cvd"),
            Self::LiquidationVolume => Some("liquidation-volume"),
            Self::Spreads => Some("spreads"),
            Self::Liquidity => Some("liquidity"),
            Self::Slippage => Some("slippage"),
            Self::FundingRates => Some("funding"),
            _ => None,
        }
    }

    pub const fn candle_kind(self) -> Option<&'static str> {
        match self {
            Self::TradeCandles => Some("trade"),
            Self::MarkCandles => Some("mark"),
            Self::SpotCandles => Some("spot"),
            _ => None,
        }
    }
}

impl fmt::Display for KrakenDataset {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for KrakenDataset {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let dataset = match value {
            "instruments" => Self::Instruments,
            "fee_schedules" => Self::FeeSchedules,
            "trade_candles" => Self::TradeCandles,
            "mark_candles" => Self::MarkCandles,
            "spot_candles" => Self::SpotCandles,
            "open_interest" => Self::OpenInterest,
            "future_basis" => Self::FutureBasis,
            "aggressor_differential" => Self::AggressorDifferential,
            "trade_volume" => Self::TradeVolume,
            "trade_count" => Self::TradeCount,
            "cvd" => Self::Cvd,
            "liquidation_volume" => Self::LiquidationVolume,
            "spreads" => Self::Spreads,
            "liquidity" => Self::Liquidity,
            "slippage" => Self::Slippage,
            "funding_rates" => Self::FundingRates,
            _ => bail!("unsupported Kraken dataset {value}"),
        };
        Ok(dataset)
    }
}

#[derive(Debug, Clone, FromRow)]
pub struct KrakenBackfillJob {
    pub job_id: Uuid,
    pub dataset: String,
    pub symbol: String,
    pub interval_seconds: i32,
    pub range_start: DateTime<Utc>,
    pub range_end: DateTime<Utc>,
}

impl KrakenBackfillJob {
    pub fn dataset(&self) -> Result<KrakenDataset> {
        self.dataset.parse()
    }
}

#[derive(Debug, Clone)]
pub struct LakeRecord {
    pub observed_at: DateTime<Utc>,
    pub payload: Value,
}

#[derive(Debug, Clone)]
pub enum NormalizedRows {
    Instruments(Vec<InstrumentRow>),
    FeeSchedules(Vec<FeeScheduleRow>),
    Candles(Vec<CandleRow>),
    Analytics(Vec<AnalyticsRow>),
    FundingRates(Vec<FundingRateRow>),
}

impl NormalizedRows {
    pub fn lake_records(&self) -> Vec<LakeRecord> {
        match self {
            Self::Instruments(rows) => rows
                .iter()
                .map(|row| LakeRecord {
                    observed_at: row.source_observed_at,
                    payload: row.raw_payload.clone(),
                })
                .collect(),
            Self::FeeSchedules(rows) => rows
                .iter()
                .map(|row| LakeRecord {
                    observed_at: row.source_observed_at,
                    payload: row.raw_payload.clone(),
                })
                .collect(),
            Self::Candles(rows) => rows
                .iter()
                .map(|row| LakeRecord {
                    observed_at: row.bucket_start,
                    payload: serde_json::json!({
                        "open": row.open,
                        "high": row.high,
                        "low": row.low,
                        "close": row.close,
                        "volume": row.volume,
                    }),
                })
                .collect(),
            Self::Analytics(rows) => rows
                .iter()
                .map(|row| LakeRecord {
                    observed_at: row.bucket_start,
                    payload: row.values.clone(),
                })
                .collect(),
            Self::FundingRates(rows) => rows
                .iter()
                .map(|row| LakeRecord {
                    observed_at: row.funding_time,
                    payload: serde_json::json!({
                        "funding_rate": row.funding_rate,
                        "relative_funding_rate": row.relative_funding_rate,
                    }),
                })
                .collect(),
        }
    }

    pub fn time_bounds(&self) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        let records = self.lake_records();
        let minimum = records.iter().map(|row| row.observed_at).min()?;
        let maximum = records.iter().map(|row| row.observed_at).max()?;
        Some((minimum, maximum))
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct InstrumentRow {
    pub symbol: String,
    pub instrument_type: String,
    pub tradeable: bool,
    pub tick_size: Option<Decimal>,
    pub contract_size: Option<Decimal>,
    pub base_currency: Option<String>,
    pub quote_currency: Option<String>,
    pub pair: Option<String>,
    pub contract_value_trade_precision: Option<i32>,
    pub max_position_size: Option<Decimal>,
    pub funding_rate_coefficient: Option<Decimal>,
    pub max_relative_funding_rate: Option<Decimal>,
    pub fee_schedule_uid: Option<String>,
    pub margin_levels: Value,
    pub retail_margin_levels: Value,
    pub margin_schedules: Value,
    pub raw_payload: Value,
    pub source_observed_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FeeScheduleRow {
    pub fee_schedule_uid: String,
    pub name: String,
    pub tiers: Value,
    pub raw_payload: Value,
    pub source_observed_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct CandleRow {
    pub bucket_start: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
}

#[derive(Debug, Clone)]
pub struct AnalyticsRow {
    pub bucket_start: DateTime<Utc>,
    pub values: Value,
}

#[derive(Debug, Clone)]
pub struct FundingRateRow {
    pub funding_time: DateTime<Utc>,
    pub funding_rate: Decimal,
    pub relative_funding_rate: Decimal,
}

#[derive(Debug, Clone)]
pub struct PublishedLakeObject {
    pub relative_path: String,
    pub sha256: String,
    pub byte_size: i64,
    pub row_count: i64,
}
