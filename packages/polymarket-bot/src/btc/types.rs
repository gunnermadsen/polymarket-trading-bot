use std::collections::{BTreeMap, VecDeque};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::{prelude::ToPrimitive, Decimal};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{
    binance_spot_l2::BinanceSpotL2FeatureWindow,
    directional_external_runtime::DirectionalExternalState,
};

pub const BTC_INTERVAL_SECONDS: i64 = 300;
pub const BTC_INTERVAL_SLUG_PREFIX: &str = "btc-updown-5m-";
pub const BINANCE_ONE_SECOND_WINDOW_CAPACITY: usize = 305;
pub const BINANCE_PREWINDOW_SUMMARY_CAPACITY: usize = 12;
pub const BINANCE_ONE_SECOND_BOOTSTRAP_CAPACITY: usize =
    BINANCE_ONE_SECOND_WINDOW_CAPACITY + BINANCE_PREWINDOW_SUMMARY_CAPACITY * 300;
pub const CHAINLINK_TWAP_60_WINDOW_CAPACITY: usize = 600;

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

    pub fn is_interval_window(&self, now: DateTime<Utc>) -> bool {
        now >= self.window_start && now < self.window_end
    }

    pub fn is_trade_window(&self, now: DateTime<Utc>) -> bool {
        self.active && !self.closed && self.accepting_orders && self.is_interval_window(now)
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainlinkTwap60Point {
    pub price: Decimal,
    pub source_timestamp: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainlinkTwap60Window {
    points: VecDeque<ChainlinkTwap60Point>,
    capacity: usize,
}

impl Default for ChainlinkTwap60Window {
    fn default() -> Self {
        Self {
            points: VecDeque::with_capacity(CHAINLINK_TWAP_60_WINDOW_CAPACITY),
            capacity: CHAINLINK_TWAP_60_WINDOW_CAPACITY,
        }
    }
}

impl ChainlinkTwap60Window {
    pub fn observe(&mut self, point: ChainlinkTwap60Point) {
        if self
            .points
            .back()
            .is_some_and(|last| last.source_timestamp == point.source_timestamp)
        {
            self.points.pop_back();
        }
        self.points.push_back(point);
        while self.points.len() > self.capacity {
            self.points.pop_front();
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &ChainlinkTwap60Point> {
        self.points.iter()
    }
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct BinanceFiveMinuteSummary {
    pub window_start: DateTime<Utc>,
    pub open_available_close: f64,
    pub window_high: f64,
    pub window_low: f64,
    pub window_quote_volume: f64,
    pub window_trade_count: f64,
    pub window_taker_buy_quote_volume: f64,
    pub window_volatility_bps: f64,
    pub source_complete: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BinanceOneSecondWindow {
    current: Option<BinanceOneSecondKline>,
    completed: VecDeque<BinanceOneSecondKline>,
    completed_five_minute_summaries: VecDeque<BinanceFiveMinuteSummary>,
}

impl BinanceOneSecondWindow {
    pub fn from_completed(klines: Vec<BinanceOneSecondKline>) -> Result<Self> {
        if klines.len() > BINANCE_ONE_SECOND_BOOTSTRAP_CAPACITY {
            bail!("Binance one-second kline bootstrap exceeds the bounded window");
        }
        for (index, kline) in klines.iter().enumerate() {
            validate_binance_one_second_kline(kline)?;
            if index > 0 && klines[index - 1].close_timestamp != kline.open_timestamp {
                bail!("Binance one-second kline bootstrap is not contiguous");
            }
        }
        let mut window = Self::default();
        for kline in klines {
            window.push_completed(kline)?;
        }
        Ok(window)
    }

    pub fn current(&self) -> Option<&BinanceOneSecondKline> {
        self.current.as_ref()
    }

    pub fn completed(&self) -> &VecDeque<BinanceOneSecondKline> {
        &self.completed
    }

    pub(crate) fn completed_five_minute_summaries(&self) -> &VecDeque<BinanceFiveMinuteSummary> {
        &self.completed_five_minute_summaries
    }

    pub fn clear(&mut self) {
        self.current = None;
        self.completed.clear();
        self.completed_five_minute_summaries.clear();
    }

    pub fn update(
        &mut self,
        trade: &BinanceAggregateTrade,
        received_at: DateTime<Utc>,
    ) -> Result<()> {
        validate_binance_aggregate_trade(trade, received_at)?;
        let bucket_open = one_second_bucket(trade.transact_time)?;
        let Some(mut current) = self.current.take() else {
            let authoritative_bootstrap = self.completed.back().cloned();
            if let Some(previous) = authoritative_bootstrap.as_ref() {
                if bucket_open < previous.close_timestamp {
                    bail!(
                        "Binance aggregate trade predates the authoritative one-second bootstrap"
                    );
                }
                let missing_seconds = (bucket_open - previous.close_timestamp).num_seconds();
                if missing_seconds > BINANCE_ONE_SECOND_WINDOW_CAPACITY as i64 {
                    bail!("Binance aggregate-trade gap exceeds the bounded model window");
                }
                let mut next_open = previous.close_timestamp;
                let carry_trade_id = trade.aggregate_trade_id.saturating_sub(1);
                while next_open < bucket_open {
                    let next_close = next_open
                        .checked_add_signed(chrono::Duration::seconds(1))
                        .context("Binance bootstrap bridge timestamp overflowed")?;
                    self.push_completed(BinanceOneSecondKline {
                        open_timestamp: next_open,
                        close_timestamp: next_close,
                        open_price: previous.close_price,
                        high_price: previous.close_price,
                        low_price: previous.close_price,
                        close_price: previous.close_price,
                        base_volume: Decimal::ZERO,
                        quote_volume: Decimal::ZERO,
                        trade_count: 0,
                        taker_buy_base_volume: Decimal::ZERO,
                        taker_buy_quote_volume: Decimal::ZERO,
                        first_aggregate_trade_id: carry_trade_id,
                        last_aggregate_trade_id: carry_trade_id,
                        first_source_timestamp: previous.last_source_timestamp,
                        last_source_timestamp: previous.last_source_timestamp,
                        max_received_at: received_at,
                        source_complete: true,
                        synthetic: true,
                    })?;
                    next_open = next_close;
                }
            }
            self.current = Some(kline_from_trade(
                bucket_open,
                trade,
                received_at,
                authoritative_bootstrap.is_some(),
            )?);
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
        if !contiguous {
            self.current = Some(current);
            bail!("Binance aggregate-trade sequence gap requires authoritative recovery");
        }
        if bucket_open == current.open_timestamp {
            apply_trade_to_kline(&mut current, trade, received_at, contiguous)?;
            self.current = Some(current);
            return Ok(());
        }

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
        self.push_completed(current)?;
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
            })?;
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

    fn push_completed(&mut self, kline: BinanceOneSecondKline) -> Result<()> {
        let next_window_start = five_minute_window_start(kline.close_timestamp)?;
        if let Some(previous) = self.completed.back() {
            let previous_window_start = five_minute_window_start(previous.close_timestamp)?;
            if previous_window_start != next_window_start {
                if let Some(summary) =
                    summarize_completed_five_minute_window(&self.completed, previous_window_start)?
                {
                    self.completed_five_minute_summaries.push_back(summary);
                    while self.completed_five_minute_summaries.len()
                        > BINANCE_PREWINDOW_SUMMARY_CAPACITY
                    {
                        self.completed_five_minute_summaries.pop_front();
                    }
                }
            }
        }
        self.completed.push_back(kline);
        while self.completed.len() > BINANCE_ONE_SECOND_WINDOW_CAPACITY {
            self.completed.pop_front();
        }
        Ok(())
    }
}

fn five_minute_window_start(timestamp: DateTime<Utc>) -> Result<DateTime<Utc>> {
    let epoch_seconds = timestamp.timestamp();
    let window_start_seconds =
        epoch_seconds.div_euclid(BTC_INTERVAL_SECONDS) * BTC_INTERVAL_SECONDS;
    DateTime::from_timestamp(window_start_seconds, 0)
        .context("Binance five-minute summary timestamp is outside the supported range")
}

fn summarize_completed_five_minute_window(
    completed: &VecDeque<BinanceOneSecondKline>,
    window_start: DateTime<Utc>,
) -> Result<Option<BinanceFiveMinuteSummary>> {
    let mut candles = completed
        .iter()
        .filter(|candle| {
            five_minute_window_start(candle.close_timestamp)
                .is_ok_and(|candidate| candidate == window_start)
        })
        .collect::<Vec<_>>();
    if candles.len() != BTC_INTERVAL_SECONDS as usize {
        return Ok(None);
    }
    candles.sort_unstable_by_key(|candle| candle.open_timestamp);
    let expected_last_close = window_start
        .checked_add_signed(chrono::Duration::seconds(BTC_INTERVAL_SECONDS - 1))
        .context("Binance five-minute summary end timestamp overflowed")?;
    if candles[0].close_timestamp != window_start
        || candles.last().map(|candle| candle.close_timestamp) != Some(expected_last_close)
        || candles
            .windows(2)
            .any(|pair| pair[0].close_timestamp != pair[1].open_timestamp)
    {
        return Ok(None);
    }

    let mut closes = Vec::with_capacity(candles.len());
    let mut window_high = f64::NEG_INFINITY;
    let mut window_low = f64::INFINITY;
    let mut window_quote_volume = 0.0;
    let mut window_trade_count = 0.0;
    let mut window_taker_buy_quote_volume = 0.0;
    let mut source_complete = true;
    for candle in &candles {
        let close = summary_decimal(candle.close_price, "close_price")?;
        closes.push(close);
        window_high = window_high.max(summary_decimal(candle.high_price, "high_price")?);
        window_low = window_low.min(summary_decimal(candle.low_price, "low_price")?);
        window_quote_volume += summary_decimal(candle.quote_volume, "quote_volume")?;
        window_trade_count += candle.trade_count as f64;
        window_taker_buy_quote_volume +=
            summary_decimal(candle.taker_buy_quote_volume, "taker_buy_quote_volume")?;
        source_complete &= candle.source_complete;
    }
    let log_returns = closes
        .windows(2)
        .map(|pair| pair[1].ln() - pair[0].ln())
        .collect::<Vec<_>>();
    let mean = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
    let squared_deviations = log_returns
        .iter()
        .map(|value| {
            let deviation = value - mean;
            deviation * deviation
        })
        .sum::<f64>();
    let window_volatility_bps =
        (squared_deviations / (log_returns.len() - 1) as f64).sqrt() * 10_000.0;
    if !window_high.is_finite()
        || !window_low.is_finite()
        || !window_quote_volume.is_finite()
        || !window_trade_count.is_finite()
        || !window_taker_buy_quote_volume.is_finite()
        || !window_volatility_bps.is_finite()
    {
        bail!("Binance five-minute summary contains a non-finite value");
    }
    Ok(Some(BinanceFiveMinuteSummary {
        window_start,
        open_available_close: closes[0],
        window_high,
        window_low,
        window_quote_volume,
        window_trade_count,
        window_taker_buy_quote_volume,
        window_volatility_bps,
        source_complete,
    }))
}

fn summary_decimal(value: Decimal, field: &str) -> Result<f64> {
    value
        .to_f64()
        .filter(|value| value.is_finite())
        .with_context(|| format!("Binance five-minute summary {field} is not finite"))
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
    /// Local time at which the connected runtime observed this unchanged book state.
    /// This is internal qualification metadata and is not part of the serialized contract.
    #[serde(skip, default = "Utc::now")]
    pub observed_at: DateTime<Utc>,
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
    /// Time-window identity used only by operator displays. Trading continues to use
    /// `current_market`, whose active/closed/accepting flags fail closed independently.
    #[serde(skip)]
    pub display_market: Option<BtcIntervalMarket>,
    pub books: BTreeMap<String, BookReadiness>,
    pub reference_prices: BTreeMap<ReferencePriceSource, ReferencePriceTick>,
    #[serde(default)]
    pub primary_persistence_degraded: bool,
    /// Internal inference-only accumulator. It is deliberately excluded from the
    /// public realtime-state JSON contract.
    #[serde(skip)]
    pub binance_one_second_window: BinanceOneSecondWindow,
    /// Compact, inference-only Binance spot L2 feature history. The reconstructed
    /// full-depth book remains task-local and is never cloned into realtime state.
    #[serde(skip)]
    pub binance_spot_l2: BinanceSpotL2FeatureWindow,
    /// Bounded, inference-only external context shared by every directional-model process.
    #[serde(skip)]
    pub directional_external: DirectionalExternalState,
    /// Bounded display-only history from Polymarket's Chainlink BTC/USD TWAP 60s topic.
    #[serde(skip)]
    pub chainlink_twap_60: ChainlinkTwap60Window,
    pub resolved_outcome: Option<BtcOutcome>,
    pub last_updated_at: Option<DateTime<Utc>>,
}

impl RealtimeState {
    pub fn primary_persistence_available(&self) -> bool {
        !self.primary_persistence_degraded
    }
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

    #[test]
    fn primary_persistence_is_available_unless_degraded() {
        let mut state = RealtimeState::default();
        assert!(state.primary_persistence_available());

        state.primary_persistence_degraded = true;
        assert!(!state.primary_persistence_available());
    }

    #[test]
    fn chainlink_twap_window_is_bounded_and_replaces_duplicate_timestamp() {
        let mut window = ChainlinkTwap60Window {
            points: VecDeque::new(),
            capacity: 2,
        };
        let first_at = at(1_000);
        window.observe(ChainlinkTwap60Point {
            price: dec!(100),
            source_timestamp: first_at,
            available_at: first_at,
        });
        window.observe(ChainlinkTwap60Point {
            price: dec!(101),
            source_timestamp: first_at,
            available_at: first_at,
        });
        window.observe(ChainlinkTwap60Point {
            price: dec!(102),
            source_timestamp: at(2_000),
            available_at: at(2_000),
        });
        window.observe(ChainlinkTwap60Point {
            price: dec!(103),
            source_timestamp: at(3_000),
            available_at: at(3_000),
        });

        let points = window.iter().collect::<Vec<_>>();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].price, dec!(102));
        assert_eq!(points[1].price, dec!(103));
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
    fn aggregate_trade_id_gap_requires_authoritative_recovery() {
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
        let error = window
            .update(
                &trade(23, 2_100, dec!(102), dec!(1), 2003, 2003, false),
                at(2_150),
            )
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("sequence gap requires authoritative recovery"));
        assert_eq!(window.current().unwrap().last_aggregate_trade_id, 21);
        assert!(window.current().unwrap().source_complete);
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
    fn five_minute_summary_ring_rolls_over_and_clears_with_the_raw_window() {
        let current_window_start = DateTime::from_timestamp(4_200, 0).unwrap();
        let first_open = current_window_start - chrono::Duration::seconds(3_901);
        let mut window = BinanceOneSecondWindow::default();

        for offset in 0_u64..=3_901 {
            let bucket = first_open + chrono::Duration::seconds(offset as i64);
            let price = Decimal::new(100_000 + offset as i64, 3);
            window
                .update(
                    &BinanceAggregateTrade {
                        aggregate_trade_id: offset + 1,
                        price,
                        quantity: Decimal::ONE,
                        first_trade_id: offset + 1,
                        last_trade_id: offset + 1,
                        transact_time: bucket + chrono::Duration::milliseconds(100),
                        is_buyer_maker: offset % 2 == 0,
                    },
                    bucket + chrono::Duration::milliseconds(150),
                )
                .unwrap();
        }

        assert_eq!(window.completed().len(), BINANCE_ONE_SECOND_WINDOW_CAPACITY);
        assert_eq!(
            window.completed_five_minute_summaries().len(),
            BINANCE_PREWINDOW_SUMMARY_CAPACITY
        );
        assert_eq!(
            window
                .completed_five_minute_summaries()
                .front()
                .unwrap()
                .window_start,
            current_window_start - chrono::Duration::seconds(3_600)
        );
        assert_eq!(
            window
                .completed_five_minute_summaries()
                .back()
                .unwrap()
                .window_start,
            current_window_start - chrono::Duration::seconds(300)
        );
        assert!(window
            .completed_five_minute_summaries()
            .iter()
            .all(|summary| summary.source_complete));

        let far_bucket = current_window_start
            + chrono::Duration::seconds(BINANCE_ONE_SECOND_WINDOW_CAPACITY as i64 + 3);
        assert!(window
            .update(
                &BinanceAggregateTrade {
                    aggregate_trade_id: 3_903,
                    price: dec!(105),
                    quantity: Decimal::ONE,
                    first_trade_id: 3_903,
                    last_trade_id: 3_903,
                    transact_time: far_bucket + chrono::Duration::milliseconds(100),
                    is_buyer_maker: false,
                },
                far_bucket + chrono::Duration::milliseconds(150),
            )
            .is_err());
        assert!(window.current().is_none());
        assert!(window.completed().is_empty());
        assert!(window.completed_five_minute_summaries().is_empty());
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
        assert!(serialized.get("binance_spot_l2").is_none());
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
