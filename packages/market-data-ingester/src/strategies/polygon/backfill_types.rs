use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

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
