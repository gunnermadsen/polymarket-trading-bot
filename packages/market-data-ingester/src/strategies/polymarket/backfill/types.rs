use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone)]
pub struct ArchiveDownloadLimits {
    pub maximum_compressed_bytes: u64,
    pub chunk_idle_timeout: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct ArchiveCancellation {
    cancelled: Arc<AtomicBool>,
}

impl ArchiveCancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadedArchive {
    pub path: PathBuf,
    pub sha256: String,
    pub compressed_bytes: u64,
    pub reused_cache: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArchiveParseSummary {
    pub records: u64,
    pub batches: u64,
    pub maximum_batch_records: usize,
    pub minimum_timestamp: Option<DateTime<Utc>>,
    pub maximum_timestamp: Option<DateTime<Utc>>,
}
