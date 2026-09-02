use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;

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
