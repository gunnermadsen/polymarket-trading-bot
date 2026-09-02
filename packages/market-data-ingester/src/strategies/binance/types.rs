use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

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
pub struct BinanceBtcusdtOpenInterestRecord {
    pub symbol: String,
    pub source_timestamp: DateTime<Utc>,
    pub period_seconds: i32,
    pub sum_open_interest: Decimal,
    pub sum_open_interest_value: Decimal,
    pub cmc_circulating_supply: Option<Decimal>,
}
