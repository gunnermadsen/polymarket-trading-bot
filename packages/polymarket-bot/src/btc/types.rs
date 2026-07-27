use std::collections::{BTreeMap, VecDeque};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const BTC_INTERVAL_SECONDS: i64 = 300;
pub const BTC_INTERVAL_SLUG_PREFIX: &str = "btc-updown-5m-";
pub const BINANCE_ONE_SECOND_WINDOW_CAPACITY: usize = 305;

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinanceAggregateTrade {
    pub aggregate_trade_id: u64,
    pub price: Decimal,
    pub quantity: Decimal,
    pub first_trade_id: u64,
    pub last_trade_id: u64,
    pub transact_time: DateTime<Utc>,
    pub is_buyer_maker: bool,
}

impl BinanceAggregateTrade {
    pub fn trade_count(&self) -> u64 {
        self.last_trade_id
            .checked_sub(self.first_trade_id)
            .and_then(|count| count.checked_add(1))
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinanceOneSecondKline {
    pub open_timestamp: DateTime<Utc>,
    pub close_timestamp: DateTime<Utc>,
    pub open_price: Decimal,
    pub high_price: Decimal,
    pub low_price: Decimal,
    pub close_price: Decimal,
    pub base_volume: Decimal,
    pub quote_volume: Decimal,
    pub trade_count: u64,
    pub taker_buy_base_volume: Decimal,
    pub taker_buy_quote_volume: Decimal,
    pub first_aggregate_trade_id: u64,
    pub last_aggregate_trade_id: u64,
    pub first_source_timestamp: DateTime<Utc>,
    pub last_source_timestamp: DateTime<Utc>,
    pub max_received_at: DateTime<Utc>,
    pub source_complete: bool,
    pub synthetic: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BinanceOneSecondWindow {
    current: Option<BinanceOneSecondKline>,
    completed: VecDeque<BinanceOneSecondKline>,
}

impl BinanceOneSecondWindow {
    pub fn from_completed(klines: Vec<BinanceOneSecondKline>) -> Result<Self> {
        if klines.len() > BINANCE_ONE_SECOND_WINDOW_CAPACITY {
            bail!("Binance one-second kline bootstrap exceeds the bounded window");
        }
        for (index, kline) in klines.iter().enumerate() {
            validate_binance_one_second_kline(kline)?;
            if index > 0 && klines[index - 1].close_timestamp != kline.open_timestamp {
                bail!("Binance one-second kline bootstrap is not contiguous");
            }
        }
        Ok(Self {
            current: None,
            completed: klines.into(),
        })
    }

    pub fn current(&self) -> Option<&BinanceOneSecondKline> {
        self.current.as_ref()
    }

    pub fn completed(&self) -> &VecDeque<BinanceOneSecondKline> {
        &self.completed
    }

    pub fn clear(&mut self) {
        self.current = None;
        self.completed.clear();
    }

    pub fn update(
        &mut self,
        trade: &BinanceAggregateTrade,
        received_at: DateTime<Utc>,
    ) -> Result<()> {
        validate_binance_aggregate_trade(trade, received_at)?;
        let bucket_open = one_second_bucket(trade.transact_time)?;
        let Some(mut current) = self.current.take() else {
            self.current = Some(kline_from_trade(bucket_open, trade, received_at, false)?);
            return Ok(());
        };

        if bucket_open < current.open_timestamp
            || trade.aggregate_trade_id <= current.last_aggregate_trade_id
        {
            self.current = Some(current);
            bail!("Binance aggregate trade regressed within the one-second window");
        }
        let contiguous = current
            .last_aggregate_trade_id
            .checked_add(1)
            .is_some_and(|expected| expected == trade.aggregate_trade_id);
        if bucket_open == current.open_timestamp {
            apply_trade_to_kline(&mut current, trade, received_at, contiguous)?;
            self.current = Some(current);
            return Ok(());
        }

        // A missing aggregate-trade ID may belong to either side of the second boundary, so
        // conservatively invalidate the closing candle as well as the new sequence.
        current.source_complete &= contiguous;
        let carry_price = current.close_price;
        let carry_trade_id = current.last_aggregate_trade_id;
        let carry_source_timestamp = current.last_source_timestamp;
        let carry_received_at = current.max_received_at;
        let missing_seconds = (bucket_open - current.close_timestamp).num_seconds();
        if missing_seconds > BINANCE_ONE_SECOND_WINDOW_CAPACITY as i64 {
            self.clear();
            bail!("Binance aggregate-trade gap exceeds the bounded model window");
        }
        let mut next_open = current.close_timestamp;
        self.push_completed(current);
        while next_open < bucket_open {
            let next_close = next_open
                .checked_add_signed(chrono::Duration::seconds(1))
                .context("Binance one-second synthetic kline timestamp overflowed")?;
            self.push_completed(BinanceOneSecondKline {
                open_timestamp: next_open,
                close_timestamp: next_close,
                open_price: carry_price,
                high_price: carry_price,
                low_price: carry_price,
                close_price: carry_price,
                base_volume: Decimal::ZERO,
                quote_volume: Decimal::ZERO,
                trade_count: 0,
                taker_buy_base_volume: Decimal::ZERO,
                taker_buy_quote_volume: Decimal::ZERO,
                first_aggregate_trade_id: carry_trade_id,
                last_aggregate_trade_id: carry_trade_id,
                first_source_timestamp: carry_source_timestamp,
                last_source_timestamp: carry_source_timestamp,
                max_received_at: carry_received_at,
                source_complete: contiguous,
                synthetic: true,
            });
            next_open = next_close;
        }
        self.current = Some(kline_from_trade(
            bucket_open,
            trade,
            received_at,
            contiguous,
        )?);
        Ok(())
    }

    fn push_completed(&mut self, kline: BinanceOneSecondKline) {
        self.completed.push_back(kline);
        while self.completed.len() > BINANCE_ONE_SECOND_WINDOW_CAPACITY {
            self.completed.pop_front();
        }
    }
}

fn validate_binance_one_second_kline(kline: &BinanceOneSecondKline) -> Result<()> {
    if kline.close_timestamp - kline.open_timestamp != chrono::Duration::seconds(1)
        || kline.open_price <= Decimal::ZERO
        || kline.high_price < kline.open_price.max(kline.close_price)
        || kline.low_price > kline.open_price.min(kline.close_price)
        || kline.low_price <= Decimal::ZERO
        || kline.base_volume < Decimal::ZERO
        || kline.quote_volume < Decimal::ZERO
        || kline.taker_buy_base_volume < Decimal::ZERO
        || kline.taker_buy_base_volume > kline.base_volume
        || kline.taker_buy_quote_volume < Decimal::ZERO
        || kline.taker_buy_quote_volume > kline.quote_volume
        || kline.first_aggregate_trade_id > kline.last_aggregate_trade_id
        || kline.first_source_timestamp > kline.last_source_timestamp
    {
        bail!("Binance one-second kline bootstrap contains an invalid candle");
    }
    Ok(())
}

fn validate_binance_aggregate_trade(
    trade: &BinanceAggregateTrade,
    received_at: DateTime<Utc>,
) -> Result<()> {
    if trade.price <= Decimal::ZERO
        || trade.quantity <= Decimal::ZERO
        || trade.first_trade_id > trade.last_trade_id
        || trade.trade_count() == 0
        || trade.transact_time > received_at + chrono::Duration::seconds(2)
    {
        bail!("Binance aggregate trade cannot contribute to a one-second kline");
    }
    Ok(())
}

fn one_second_bucket(timestamp: DateTime<Utc>) -> Result<DateTime<Utc>> {
    let seconds = timestamp.timestamp();
    DateTime::from_timestamp(seconds, 0)
        .context("Binance aggregate trade timestamp is outside the supported range")
}

fn kline_from_trade(
    open_timestamp: DateTime<Utc>,
    trade: &BinanceAggregateTrade,
    received_at: DateTime<Utc>,
    source_complete: bool,
) -> Result<BinanceOneSecondKline> {
    let close_timestamp = open_timestamp
        .checked_add_signed(chrono::Duration::seconds(1))
        .context("Binance one-second kline timestamp overflowed")?;
    let quote_volume = trade
        .price
        .checked_mul(trade.quantity)
        .context("Binance aggregate trade quote volume overflowed")?;
    let (taker_buy_base_volume, taker_buy_quote_volume) = if trade.is_buyer_maker {
        (Decimal::ZERO, Decimal::ZERO)
    } else {
        (trade.quantity, quote_volume)
    };
    Ok(BinanceOneSecondKline {
        open_timestamp,
        close_timestamp,
        open_price: trade.price,
        high_price: trade.price,
        low_price: trade.price,
        close_price: trade.price,
        base_volume: trade.quantity,
        quote_volume,
        trade_count: trade.trade_count(),
        taker_buy_base_volume,
        taker_buy_quote_volume,
        first_aggregate_trade_id: trade.aggregate_trade_id,
        last_aggregate_trade_id: trade.aggregate_trade_id,
        first_source_timestamp: trade.transact_time,
        last_source_timestamp: trade.transact_time,
        max_received_at: received_at,
        source_complete,
        synthetic: false,
    })
}

fn apply_trade_to_kline(
    kline: &mut BinanceOneSecondKline,
    trade: &BinanceAggregateTrade,
    received_at: DateTime<Utc>,
    contiguous: bool,
) -> Result<()> {
    let quote_volume = trade
        .price
        .checked_mul(trade.quantity)
        .context("Binance aggregate trade quote volume overflowed")?;
    kline.high_price = kline.high_price.max(trade.price);
    kline.low_price = kline.low_price.min(trade.price);
    kline.close_price = trade.price;
    kline.base_volume = kline
        .base_volume
        .checked_add(trade.quantity)
        .context("Binance one-second base volume overflowed")?;
    kline.quote_volume = kline
        .quote_volume
        .checked_add(quote_volume)
        .context("Binance one-second quote volume overflowed")?;
    kline.trade_count = kline
        .trade_count
        .checked_add(trade.trade_count())
        .context("Binance one-second trade count overflowed")?;
    if !trade.is_buyer_maker {
        kline.taker_buy_base_volume = kline
            .taker_buy_base_volume
            .checked_add(trade.quantity)
            .context("Binance one-second taker-buy base volume overflowed")?;
        kline.taker_buy_quote_volume = kline
            .taker_buy_quote_volume
            .checked_add(quote_volume)
            .context("Binance one-second taker-buy quote volume overflowed")?;
    }
    kline.last_aggregate_trade_id = trade.aggregate_trade_id;
    kline.last_source_timestamp = trade.transact_time;
    kline.max_received_at = kline.max_received_at.max(received_at);
    kline.source_complete &= contiguous;
    Ok(())
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
    /// Internal inference-only accumulator. It is deliberately excluded from the
    /// public realtime-state JSON contract.
    #[serde(skip)]
    pub binance_one_second_window: BinanceOneSecondWindow,
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

#[cfg(test)]
mod tests {
    use rust_decimal_macros::dec;

    use super::*;

    fn at(milliseconds: i64) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(milliseconds).unwrap()
    }

    fn trade(
        aggregate_trade_id: u64,
        milliseconds: i64,
        price: Decimal,
        quantity: Decimal,
        first_trade_id: u64,
        last_trade_id: u64,
        is_buyer_maker: bool,
    ) -> BinanceAggregateTrade {
        BinanceAggregateTrade {
            aggregate_trade_id,
            price,
            quantity,
            first_trade_id,
            last_trade_id,
            transact_time: at(milliseconds),
            is_buyer_maker,
        }
    }

    #[test]
    fn one_second_window_builds_ohlcv_flow_and_zero_trade_gap_candles() {
        let mut window = BinanceOneSecondWindow::default();
        window
            .update(
                &trade(10, 100, dec!(100), dec!(2), 1000, 1001, false),
                at(150),
            )
            .unwrap();
        window
            .update(
                &trade(11, 900, dec!(105), dec!(1), 1002, 1002, true),
                at(950),
            )
            .unwrap();
        window
            .update(
                &trade(12, 2_100, dec!(103), dec!(0.5), 1003, 1003, false),
                at(2_150),
            )
            .unwrap();

        assert_eq!(window.completed().len(), 2);
        let first = &window.completed()[0];
        assert_eq!(first.open_price, dec!(100));
        assert_eq!(first.high_price, dec!(105));
        assert_eq!(first.low_price, dec!(100));
        assert_eq!(first.close_price, dec!(105));
        assert_eq!(first.base_volume, dec!(3));
        assert_eq!(first.quote_volume, dec!(305));
        assert_eq!(first.trade_count, 3);
        assert_eq!(first.taker_buy_base_volume, dec!(2));
        assert_eq!(first.taker_buy_quote_volume, dec!(200));
        assert!(!first.source_complete);

        let synthetic = &window.completed()[1];
        assert!(synthetic.synthetic);
        assert!(synthetic.source_complete);
        assert_eq!(synthetic.open_price, dec!(105));
        assert_eq!(synthetic.close_price, dec!(105));
        assert_eq!(synthetic.trade_count, 0);
        assert_eq!(synthetic.quote_volume, Decimal::ZERO);

        let current = window.current().unwrap();
        assert_eq!(current.open_timestamp, at(2_000));
        assert_eq!(current.open_price, dec!(103));
        assert!(current.source_complete);
    }

    #[test]
    fn aggregate_trade_id_gap_invalidates_both_sides_of_boundary() {
        let mut window = BinanceOneSecondWindow::default();
        window
            .update(
                &trade(20, 100, dec!(100), dec!(1), 2000, 2000, false),
                at(150),
            )
            .unwrap();
        window
            .update(
                &trade(21, 1_100, dec!(101), dec!(1), 2001, 2001, false),
                at(1_150),
            )
            .unwrap();
        window
            .update(
                &trade(23, 2_100, dec!(102), dec!(1), 2003, 2003, false),
                at(2_150),
            )
            .unwrap();

        assert!(!window.completed().back().unwrap().source_complete);
        assert!(!window.current().unwrap().source_complete);
    }

    #[test]
    fn oversized_aggregate_trade_gap_resets_without_unbounded_synthesis() {
        let mut window = BinanceOneSecondWindow::default();
        window
            .update(
                &trade(30, 100, dec!(100), dec!(1), 3000, 3000, false),
                at(150),
            )
            .unwrap();

        let gap = (BINANCE_ONE_SECOND_WINDOW_CAPACITY as i64 + 2) * 1_000;
        assert!(window
            .update(
                &trade(31, gap, dec!(101), dec!(1), 3001, 3001, false),
                at(gap + 50),
            )
            .is_err());
        assert!(window.current().is_none());
        assert!(window.completed().is_empty());
    }

    #[test]
    fn inference_window_is_not_part_of_realtime_state_json_contract() {
        let mut state = RealtimeState::default();
        state
            .binance_one_second_window
            .update(
                &trade(40, 100, dec!(100), dec!(1), 4000, 4000, false),
                at(150),
            )
            .unwrap();

        let serialized = serde_json::to_value(state).unwrap();

        assert!(serialized.get("binance_one_second_window").is_none());
    }

    #[test]
    #[ignore = "manual bounded-state clone latency benchmark"]
    fn benchmark_realtime_state_clone() {
        let mut state = RealtimeState::default();
        for second in 0..=BINANCE_ONE_SECOND_WINDOW_CAPACITY {
            let timestamp = second as i64 * 1_000 + 100;
            state
                .binance_one_second_window
                .update(
                    &trade(
                        second as u64 + 1,
                        timestamp,
                        dec!(100),
                        dec!(1),
                        second as u64 + 1,
                        second as u64 + 1,
                        false,
                    ),
                    at(timestamp + 50),
                )
                .unwrap();
        }
        let iterations = 100_000u128;
        let started_at = std::time::Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(state.clone());
        }
        let mean_clone_ns = started_at.elapsed().as_nanos() / iterations;
        eprintln!("bounded_realtime_state_mean_clone_ns={mean_clone_ns}");
    }
}
