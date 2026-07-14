use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const BTC_INTERVAL_SECONDS: i64 = 300;
pub const BTC_INTERVAL_SLUG_PREFIX: &str = "btc-updown-5m-";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BtcOutcome {
    Up,
    Down,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    pub fee_schedule: serde_json::Value,
    pub raw_payload: serde_json::Value,
}

impl BtcIntervalMarket {
    pub fn token_id(&self, outcome: BtcOutcome) -> &str {
        match outcome {
            BtcOutcome::Up => &self.up_token_id,
            BtcOutcome::Down => &self.down_token_id,
        }
    }

    pub fn is_trade_window(&self, now: DateTime<Utc>) -> bool {
        self.active
            && !self.closed
            && self.accepting_orders
            && now >= self.window_start
            && now < self.window_end
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferencePriceSource {
    DirectBinance,
    RtdsBinance,
    RtdsChainlink,
}

impl ReferencePriceSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectBinance => "direct_binance",
            Self::RtdsBinance => "rtds_binance",
            Self::RtdsChainlink => "rtds_chainlink",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferencePriceTick {
    pub tick_id: Uuid,
    pub dedup_key: String,
    pub source: ReferencePriceSource,
    pub symbol: String,
    pub price: Decimal,
    pub source_timestamp: DateTime<Utc>,
    pub envelope_timestamp: Option<DateTime<Utc>>,
    pub received_at: DateTime<Utc>,
    pub connection_id: Uuid,
    pub ingest_sequence: u64,
    pub source_event_id: Option<String>,
    pub raw_payload: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MarketFeedEventType {
    Book,
    PriceChange,
    BestBidAsk,
    TickSizeChange,
    LastTradePrice,
    MarketResolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedIntegrityStatus {
    Ok,
    PreSnapshot,
    Stale,
    OutOfOrder,
    DecodeError,
    CrossedBook,
    TopOfBookMismatch,
    UnknownToken,
    MarketMismatch,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarketFeedEvent {
    pub event_id: Uuid,
    pub market_id: String,
    pub token_id: Option<String>,
    pub event_type: MarketFeedEventType,
    pub source_timestamp: DateTime<Utc>,
    pub received_at: DateTime<Utc>,
    pub connection_id: Uuid,
    pub ingest_sequence: u64,
    pub source_hash: Option<String>,
    pub applied: bool,
    pub integrity_status: FeedIntegrityStatus,
    pub raw_payload: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderbookLevel {
    pub price: Decimal,
    pub size: Decimal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderbookCheckpoint {
    pub checkpoint_id: Uuid,
    pub market_id: String,
    pub token_id: String,
    pub source_timestamp: DateTime<Utc>,
    pub received_at: DateTime<Utc>,
    pub connection_id: Uuid,
    pub ingest_sequence: u64,
    pub source_hash: Option<String>,
    pub tick_size: Decimal,
    pub best_bid: Option<Decimal>,
    pub best_ask: Option<Decimal>,
    pub bids: Vec<OrderbookLevel>,
    pub asks: Vec<OrderbookLevel>,
    pub integrity_status: FeedIntegrityStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BookReadiness {
    pub market_id: String,
    pub token_id: String,
    /// Current CLOB websocket epoch. A book is never reusable across epochs.
    pub connection_id: Uuid,
    pub bootstrapped: bool,
    pub integrity_status: FeedIntegrityStatus,
    pub source_timestamp: Option<DateTime<Utc>>,
    pub received_at: Option<DateTime<Utc>>,
    pub best_bid: Option<Decimal>,
    pub best_ask: Option<Decimal>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceReadiness {
    pub source: ReferencePriceSource,
    pub source_timestamp: DateTime<Utc>,
    pub received_at: DateTime<Utc>,
    pub price: Decimal,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RealtimeState {
    pub current_market: Option<BtcIntervalMarket>,
    pub books: BTreeMap<String, BookReadiness>,
    pub reference_prices: BTreeMap<ReferencePriceSource, ReferencePriceTick>,
    pub resolved_outcome: Option<BtcOutcome>,
    pub last_updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Readiness {
    pub ready: bool,
    pub checked_at: DateTime<Utc>,
    pub market_slug: Option<String>,
    pub reasons: Vec<String>,
    pub books: Vec<BookReadiness>,
    pub sources: Vec<SourceReadiness>,
}
