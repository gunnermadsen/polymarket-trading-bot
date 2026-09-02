use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

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
