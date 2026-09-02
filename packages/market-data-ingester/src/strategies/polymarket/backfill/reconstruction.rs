use std::collections::{BTreeMap, HashMap};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use serde_json::Value;

use super::types::{
    BtcExecutionSnapshot, BtcOrderbookArchiveEvent, BtcOrderbookMarketScope, BtcOutcome,
};

pub const EXECUTION_SNAPSHOT_SCHEMA_VERSION: &str = "btc5m-capacity-book-1-240s-v2";
pub const EXECUTION_SNAPSHOT_VWAP_QUANTITIES: [i64; 15] = [
    1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200,
];
pub const EXECUTION_SNAPSHOT_START_MILLIS: i64 = 1_000;
pub const EXECUTION_SNAPSHOT_END_MILLIS: i64 = 240_000;
pub const EXECUTION_SNAPSHOT_EARLY_END_MILLIS: i64 = 59_000;
pub const EXECUTION_SNAPSHOT_EARLY_INTERVAL_MILLIS: i64 = 1_000;
pub const EXECUTION_SNAPSHOT_LATER_INTERVAL_MILLIS: i64 = 5_000;
pub const EXECUTION_SNAPSHOTS_PER_MARKET: usize = 96;
pub const QUALITY_UP_MISSING: i32 = 1;
pub const QUALITY_DOWN_MISSING: i32 = 1 << 1;
pub const QUALITY_UP_STALE: i32 = 1 << 2;
pub const QUALITY_DOWN_STALE: i32 = 1 << 3;
pub const QUALITY_UP_CROSSED: i32 = 1 << 4;
pub const QUALITY_DOWN_CROSSED: i32 = 1 << 5;
pub const QUALITY_UP_INSUFFICIENT_DEPTH: i32 = 1 << 6;
pub const QUALITY_DOWN_INSUFFICIENT_DEPTH: i32 = 1 << 7;
pub const QUALITY_UP_INSUFFICIENT_DEPTH_10: i32 = 1 << 8;
pub const QUALITY_DOWN_INSUFFICIENT_DEPTH_10: i32 = 1 << 9;
pub const QUALITY_UP_INSUFFICIENT_DEPTH_15: i32 = 1 << 10;
pub const QUALITY_DOWN_INSUFFICIENT_DEPTH_15: i32 = 1 << 11;
pub const QUALITY_UP_INSUFFICIENT_DEPTH_20: i32 = 1 << 12;
pub const QUALITY_DOWN_INSUFFICIENT_DEPTH_20: i32 = 1 << 13;

const STALE_AFTER_MILLIS: i64 = 2_000;

#[derive(Debug, Clone, Default)]
struct BookState {
    initialized: bool,
    bids: BTreeMap<Decimal, Decimal>,
    asks: BTreeMap<Decimal, Decimal>,
    source_row_number: Option<i64>,
    source_timestamp: Option<DateTime<Utc>>,
    provider_received_at: Option<DateTime<Utc>>,
}

impl BookState {
    fn apply(&mut self, event: &BtcOrderbookArchiveEvent) -> Result<()> {
        match event.event_type.as_str() {
            "book" => {
                self.bids = parse_levels(event.bids.as_ref(), "bids")?;
                self.asks = parse_levels(event.asks.as_ref(), "asks")?;
                self.initialized = true;
            }
            "price_change" if self.initialized => {
                let price = event.price.context("price_change is missing price")?;
                let size = event.size.context("price_change is missing size")?;
                let levels = match event.side.as_deref() {
                    Some("buy") => &mut self.bids,
                    Some("sell") => &mut self.asks,
                    _ => bail!("price_change is missing a supported side"),
                };
                if size.is_zero() {
                    levels.remove(&price);
                } else {
                    levels.insert(price, size);
                }
            }
            "price_change" | "last_trade_price" | "tick_size_change" => return Ok(()),
            other => bail!("unsupported PMXT orderbook event type {other}"),
        }
        self.source_row_number = Some(event.source_row_number);
        self.source_timestamp = Some(event.source_timestamp);
        self.provider_received_at = Some(event.provider_received_at);
        Ok(())
    }
}

#[derive(Debug)]
struct MarketState {
    scope: BtcOrderbookMarketScope,
    up: OutcomeState,
    down: OutcomeState,
    emitted_samples: usize,
}

#[derive(Debug)]
struct OutcomeState {
    book: BookState,
    next_sample: DateTime<Utc>,
    samples: Vec<BookMeasures>,
    last_event_received_at: Option<DateTime<Utc>>,
}

impl OutcomeState {
    fn new(next_sample: DateTime<Utc>, book: BookState) -> Self {
        Self {
            book,
            next_sample,
            samples: Vec::with_capacity(EXECUTION_SNAPSHOTS_PER_MARKET),
            last_event_received_at: None,
        }
    }

    fn apply(&mut self, event: &BtcOrderbookArchiveEvent, window_end: DateTime<Utc>) -> Result<()> {
        if self
            .last_event_received_at
            .is_some_and(|previous| event.provider_received_at < previous)
        {
            bail!("PMXT events for one outcome token are not ordered by provider receipt time");
        }
        if self.last_event_received_at != Some(event.provider_received_at) {
            self.emit_before(event.provider_received_at, window_end);
            self.last_event_received_at = Some(event.provider_received_at);
        }
        self.book.apply(event)
    }

    #[allow(dead_code)]
    fn emit_through(&mut self, through: DateTime<Utc>, window_end: DateTime<Utc>) {
        while self.next_sample <= through && self.next_sample < window_end {
            self.samples.push(measure(&self.book, self.next_sample));
            self.next_sample = next_sample(self.next_sample, window_end);
        }
    }

    fn emit_before(&mut self, boundary: DateTime<Utc>, window_end: DateTime<Utc>) {
        while self.next_sample < boundary && self.next_sample < window_end {
            self.samples.push(measure(&self.book, self.next_sample));
            self.next_sample = next_sample(self.next_sample, window_end);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionMarketSeed {
    market_id: String,
    up: BookState,
    down: BookState,
}

#[derive(Debug)]
pub struct ExecutionSnapshotReconstructor {
    markets: Vec<MarketState>,
    asset_index: HashMap<String, (usize, BtcOutcome)>,
}

impl ExecutionSnapshotReconstructor {
    pub fn new(markets: Vec<BtcOrderbookMarketScope>) -> Result<Self> {
        Self::new_with_seed(markets, None)
    }

    pub fn new_with_seed(
        markets: Vec<BtcOrderbookMarketScope>,
        seed: Option<ExecutionMarketSeed>,
    ) -> Result<Self> {
        if markets.is_empty() {
            bail!("execution snapshot reconstruction requires market scope");
        }
        let mut states = Vec::with_capacity(markets.len());
        let mut asset_index = HashMap::with_capacity(markets.len() * 2);
        for scope in markets {
            if scope.window_end <= scope.window_start {
                bail!("execution snapshot market window must be non-empty");
            }
            let index = states.len();
            if asset_index
                .insert(scope.up_token_id.clone(), (index, BtcOutcome::Up))
                .is_some()
                || asset_index
                    .insert(scope.down_token_id.clone(), (index, BtcOutcome::Down))
                    .is_some()
            {
                bail!("execution snapshot token identity must be unique");
            }
            let seeded_books = seed
                .as_ref()
                .filter(|seed| seed.market_id == scope.market_id)
                .map(|seed| (seed.up.clone(), seed.down.clone()))
                .unwrap_or_default();
            states.push(MarketState {
                up: OutcomeState::new(decision_window_start(&scope), seeded_books.0),
                down: OutcomeState::new(decision_window_start(&scope), seeded_books.1),
                scope,
                emitted_samples: 0,
            });
        }
        Ok(Self {
            markets: states,
            asset_index,
        })
    }

    pub fn apply(
        &mut self,
        event: &BtcOrderbookArchiveEvent,
        _output: &mut Vec<BtcExecutionSnapshot>,
    ) -> Result<()> {
        if let Some((market_index, outcome)) = self.asset_index.get(&event.asset_id).copied() {
            let market = &mut self.markets[market_index];
            if event.condition_id != market.scope.condition_id {
                bail!("PMXT token was associated with an unexpected condition");
            }
            match outcome {
                BtcOutcome::Up => market.up.apply(event, decision_window_end(&market.scope))?,
                BtcOutcome::Down => market
                    .down
                    .apply(event, decision_window_end(&market.scope))?,
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn finish(&mut self, through: DateTime<Utc>, output: &mut Vec<BtcExecutionSnapshot>) {
        for market in &mut self.markets {
            let sample_end = decision_window_end(&market.scope);
            market.up.emit_through(through, sample_end);
            market.down.emit_through(through, sample_end);
            emit_joined_snapshots(market, output);
        }
    }

    pub fn finish_before(
        &mut self,
        boundary: DateTime<Utc>,
        output: &mut Vec<BtcExecutionSnapshot>,
    ) {
        for market in &mut self.markets {
            let sample_end = decision_window_end(&market.scope);
            market.up.emit_before(boundary, sample_end);
            market.down.emit_before(boundary, sample_end);
            emit_joined_snapshots(market, output);
        }
    }

    #[allow(dead_code)]
    pub fn market_seed(&self, market_id: &str) -> Option<ExecutionMarketSeed> {
        self.markets
            .iter()
            .find(|market| market.scope.market_id == market_id)
            .map(|market| ExecutionMarketSeed {
                market_id: market.scope.market_id.clone(),
                up: market.up.book.clone(),
                down: market.down.book.clone(),
            })
    }
}

fn emit_joined_snapshots(market: &mut MarketState, output: &mut Vec<BtcExecutionSnapshot>) {
    let available_samples = market.up.samples.len().min(market.down.samples.len());
    while market.emitted_samples < available_samples {
        let index = market.emitted_samples;
        let offset_millis = sample_offset_millis(index).unwrap_or(i64::MAX);
        let sampled_at = market.scope.window_start + Duration::milliseconds(offset_millis);
        output.push(snapshot(
            &market.scope,
            sampled_at,
            &market.up.samples[index],
            &market.down.samples[index],
        ));
        market.emitted_samples += 1;
    }
}

fn decision_window_start(scope: &BtcOrderbookMarketScope) -> DateTime<Utc> {
    scope.window_start + Duration::milliseconds(EXECUTION_SNAPSHOT_START_MILLIS)
}

fn decision_window_end(scope: &BtcOrderbookMarketScope) -> DateTime<Utc> {
    let next_sample_after_window =
        scope.window_start + Duration::milliseconds(EXECUTION_SNAPSHOT_END_MILLIS + 1);
    next_sample_after_window.min(scope.window_end)
}

fn sample_offset_millis(index: usize) -> Option<i64> {
    let index = i64::try_from(index).ok()?;
    let early_samples =
        EXECUTION_SNAPSHOT_EARLY_END_MILLIS / EXECUTION_SNAPSHOT_EARLY_INTERVAL_MILLIS;
    if index < early_samples {
        return Some((index + 1) * EXECUTION_SNAPSHOT_EARLY_INTERVAL_MILLIS);
    }
    Some(60_000 + (index - early_samples) * EXECUTION_SNAPSHOT_LATER_INTERVAL_MILLIS)
}

fn next_sample(current: DateTime<Utc>, window_end: DateTime<Utc>) -> DateTime<Utc> {
    let window_start = window_end - Duration::milliseconds(EXECUTION_SNAPSHOT_END_MILLIS + 1);
    let elapsed = current
        .signed_duration_since(window_start)
        .num_milliseconds();
    let interval = if elapsed <= EXECUTION_SNAPSHOT_EARLY_END_MILLIS {
        EXECUTION_SNAPSHOT_EARLY_INTERVAL_MILLIS
    } else {
        EXECUTION_SNAPSHOT_LATER_INTERVAL_MILLIS
    };
    current + Duration::milliseconds(interval)
}

#[derive(Debug, Default)]
struct BookMeasures {
    source_row_number: Option<i64>,
    source_timestamp: Option<DateTime<Utc>>,
    provider_received_at: Option<DateTime<Utc>>,
    best_bid: Option<Decimal>,
    best_ask: Option<Decimal>,
    best_bid_size: Option<Decimal>,
    best_ask_size: Option<Decimal>,
    bid_depth: Option<Decimal>,
    ask_depth: Option<Decimal>,
    ask_vwap_1: Option<Decimal>,
    ask_vwap_5: Option<Decimal>,
    ask_vwap_10: Option<Decimal>,
    ask_vwap_15: Option<Decimal>,
    ask_vwap_20: Option<Decimal>,
    ask_vwap_25: Option<Decimal>,
    ask_vwap_30: Option<Decimal>,
    ask_vwap_40: Option<Decimal>,
    ask_vwap_50: Option<Decimal>,
    ask_vwap_75: Option<Decimal>,
    ask_vwap_100: Option<Decimal>,
    ask_vwap_125: Option<Decimal>,
    ask_vwap_150: Option<Decimal>,
    ask_vwap_175: Option<Decimal>,
    ask_vwap_200: Option<Decimal>,
    imbalance: Option<Decimal>,
    missing: bool,
    stale: bool,
    crossed: bool,
    insufficient_depth: bool,
    insufficient_depth_10: bool,
    insufficient_depth_15: bool,
    insufficient_depth_20: bool,
}

fn snapshot(
    scope: &BtcOrderbookMarketScope,
    sampled_at: DateTime<Utc>,
    up: &BookMeasures,
    down: &BookMeasures,
) -> BtcExecutionSnapshot {
    let mut quality_flags = 0;
    if up.missing {
        quality_flags |= QUALITY_UP_MISSING;
    }
    if down.missing {
        quality_flags |= QUALITY_DOWN_MISSING;
    }
    if up.stale {
        quality_flags |= QUALITY_UP_STALE;
    }
    if down.stale {
        quality_flags |= QUALITY_DOWN_STALE;
    }
    if up.crossed {
        quality_flags |= QUALITY_UP_CROSSED;
    }
    if down.crossed {
        quality_flags |= QUALITY_DOWN_CROSSED;
    }
    if up.insufficient_depth {
        quality_flags |= QUALITY_UP_INSUFFICIENT_DEPTH;
    }
    if down.insufficient_depth {
        quality_flags |= QUALITY_DOWN_INSUFFICIENT_DEPTH;
    }
    if up.insufficient_depth_10 {
        quality_flags |= QUALITY_UP_INSUFFICIENT_DEPTH_10;
    }
    if down.insufficient_depth_10 {
        quality_flags |= QUALITY_DOWN_INSUFFICIENT_DEPTH_10;
    }
    if up.insufficient_depth_15 {
        quality_flags |= QUALITY_UP_INSUFFICIENT_DEPTH_15;
    }
    if down.insufficient_depth_15 {
        quality_flags |= QUALITY_DOWN_INSUFFICIENT_DEPTH_15;
    }
    if up.insufficient_depth_20 {
        quality_flags |= QUALITY_UP_INSUFFICIENT_DEPTH_20;
    }
    if down.insufficient_depth_20 {
        quality_flags |= QUALITY_DOWN_INSUFFICIENT_DEPTH_20;
    }
    BtcExecutionSnapshot {
        market_id: scope.market_id.clone(),
        sampled_at,
        up_source_row_number: up.source_row_number,
        up_source_timestamp: up.source_timestamp,
        up_provider_received_at: up.provider_received_at,
        up_best_bid: up.best_bid,
        up_best_ask: up.best_ask,
        up_best_bid_size: up.best_bid_size,
        up_best_ask_size: up.best_ask_size,
        up_bid_depth: up.bid_depth,
        up_ask_depth: up.ask_depth,
        up_ask_vwap_1: up.ask_vwap_1,
        up_ask_vwap_5: up.ask_vwap_5,
        up_ask_vwap_10: up.ask_vwap_10,
        up_ask_vwap_15: up.ask_vwap_15,
        up_ask_vwap_20: up.ask_vwap_20,
        up_ask_vwap_25: up.ask_vwap_25,
        up_ask_vwap_30: up.ask_vwap_30,
        up_ask_vwap_40: up.ask_vwap_40,
        up_ask_vwap_50: up.ask_vwap_50,
        up_ask_vwap_75: up.ask_vwap_75,
        up_ask_vwap_100: up.ask_vwap_100,
        up_ask_vwap_125: up.ask_vwap_125,
        up_ask_vwap_150: up.ask_vwap_150,
        up_ask_vwap_175: up.ask_vwap_175,
        up_ask_vwap_200: up.ask_vwap_200,
        up_imbalance: up.imbalance,
        down_source_row_number: down.source_row_number,
        down_source_timestamp: down.source_timestamp,
        down_provider_received_at: down.provider_received_at,
        down_best_bid: down.best_bid,
        down_best_ask: down.best_ask,
        down_best_bid_size: down.best_bid_size,
        down_best_ask_size: down.best_ask_size,
        down_bid_depth: down.bid_depth,
        down_ask_depth: down.ask_depth,
        down_ask_vwap_1: down.ask_vwap_1,
        down_ask_vwap_5: down.ask_vwap_5,
        down_ask_vwap_10: down.ask_vwap_10,
        down_ask_vwap_15: down.ask_vwap_15,
        down_ask_vwap_20: down.ask_vwap_20,
        down_ask_vwap_25: down.ask_vwap_25,
        down_ask_vwap_30: down.ask_vwap_30,
        down_ask_vwap_40: down.ask_vwap_40,
        down_ask_vwap_50: down.ask_vwap_50,
        down_ask_vwap_75: down.ask_vwap_75,
        down_ask_vwap_100: down.ask_vwap_100,
        down_ask_vwap_125: down.ask_vwap_125,
        down_ask_vwap_150: down.ask_vwap_150,
        down_ask_vwap_175: down.ask_vwap_175,
        down_ask_vwap_200: down.ask_vwap_200,
        down_imbalance: down.imbalance,
        quality_flags,
    }
}

fn measure(state: &BookState, sampled_at: DateTime<Utc>) -> BookMeasures {
    if !state.initialized {
        return BookMeasures {
            source_row_number: state.source_row_number,
            source_timestamp: state.source_timestamp,
            provider_received_at: state.provider_received_at,
            missing: true,
            ..BookMeasures::default()
        };
    }
    let best_bid = state.bids.last_key_value().map(|(price, _)| *price);
    let best_ask = state.asks.first_key_value().map(|(price, _)| *price);
    let crossed = best_bid.zip(best_ask).is_some_and(|(bid, ask)| bid >= ask);
    let bid_depth = state.bids.values().copied().sum::<Decimal>();
    let ask_depth = state.asks.values().copied().sum::<Decimal>();
    let total_depth = bid_depth + ask_depth;
    let stale = state.provider_received_at.is_none_or(|received_at| {
        sampled_at
            .signed_duration_since(received_at)
            .num_milliseconds()
            > STALE_AFTER_MILLIS
    });
    let [ask_vwap_1, ask_vwap_5, ask_vwap_10, ask_vwap_15, ask_vwap_20, ask_vwap_25, ask_vwap_30, ask_vwap_40, ask_vwap_50, ask_vwap_75, ask_vwap_100, ask_vwap_125, ask_vwap_150, ask_vwap_175, ask_vwap_200] =
        ask_vwaps(&state.asks, EXECUTION_SNAPSHOT_VWAP_QUANTITIES);
    BookMeasures {
        source_row_number: state.source_row_number,
        source_timestamp: state.source_timestamp,
        provider_received_at: state.provider_received_at,
        best_bid: (!crossed).then_some(best_bid).flatten(),
        best_ask: (!crossed).then_some(best_ask).flatten(),
        best_bid_size: (!crossed)
            .then(|| best_bid.and_then(|price| state.bids.get(&price).copied()))
            .flatten(),
        best_ask_size: (!crossed)
            .then(|| best_ask.and_then(|price| state.asks.get(&price).copied()))
            .flatten(),
        bid_depth: Some(bid_depth),
        ask_depth: Some(ask_depth),
        ask_vwap_1: (!crossed).then_some(ask_vwap_1).flatten(),
        ask_vwap_5: (!crossed).then_some(ask_vwap_5).flatten(),
        ask_vwap_10: (!crossed).then_some(ask_vwap_10).flatten(),
        ask_vwap_15: (!crossed).then_some(ask_vwap_15).flatten(),
        ask_vwap_20: (!crossed).then_some(ask_vwap_20).flatten(),
        ask_vwap_25: (!crossed).then_some(ask_vwap_25).flatten(),
        ask_vwap_30: (!crossed).then_some(ask_vwap_30).flatten(),
        ask_vwap_40: (!crossed).then_some(ask_vwap_40).flatten(),
        ask_vwap_50: (!crossed).then_some(ask_vwap_50).flatten(),
        ask_vwap_75: (!crossed).then_some(ask_vwap_75).flatten(),
        ask_vwap_100: (!crossed).then_some(ask_vwap_100).flatten(),
        ask_vwap_125: (!crossed).then_some(ask_vwap_125).flatten(),
        ask_vwap_150: (!crossed).then_some(ask_vwap_150).flatten(),
        ask_vwap_175: (!crossed).then_some(ask_vwap_175).flatten(),
        ask_vwap_200: (!crossed).then_some(ask_vwap_200).flatten(),
        imbalance: (!total_depth.is_zero()).then(|| (bid_depth - ask_depth) / total_depth),
        missing: best_bid.is_none() || best_ask.is_none(),
        stale,
        crossed,
        insufficient_depth: ask_vwap_1.is_none(),
        insufficient_depth_10: ask_vwap_10.is_none(),
        insufficient_depth_15: ask_vwap_15.is_none(),
        insufficient_depth_20: ask_vwap_20.is_none(),
    }
}

fn ask_vwaps<const N: usize>(
    levels: &BTreeMap<Decimal, Decimal>,
    target_quantities: [i64; N],
) -> [Option<Decimal>; N] {
    let mut results = [None; N];
    let mut target_index = 0;
    let mut cumulative_size = Decimal::ZERO;
    let mut cumulative_notional = Decimal::ZERO;
    for (price, size) in levels {
        let level_end = cumulative_size + *size;
        while target_index < N {
            let target = Decimal::from(target_quantities[target_index]);
            if target > level_end {
                break;
            }
            let consumed_at_level = target - cumulative_size;
            results[target_index] =
                Some((cumulative_notional + (*price * consumed_at_level)) / target);
            target_index += 1;
        }
        cumulative_size = level_end;
        cumulative_notional += *price * *size;
        if target_index == N {
            break;
        }
    }
    results
}

fn parse_levels(value: Option<&Value>, name: &str) -> Result<BTreeMap<Decimal, Decimal>> {
    let rows = value
        .and_then(Value::as_array)
        .with_context(|| format!("book event {name} must be an array"))?;
    let mut levels = BTreeMap::new();
    for row in rows {
        let pair = row
            .as_array()
            .with_context(|| format!("book event {name} level must be an array"))?;
        if pair.len() != 2 {
            bail!("book event {name} level must contain price and size");
        }
        let price = json_decimal(&pair[0], "price")?;
        let size = json_decimal(&pair[1], "size")?;
        if price < Decimal::ZERO || price > Decimal::ONE || size < Decimal::ZERO {
            bail!("book event {name} level is outside its valid range");
        }
        if size.is_zero() {
            levels.remove(&price);
        } else {
            levels.insert(price, size);
        }
    }
    Ok(levels)
}

fn json_decimal(value: &Value, name: &str) -> Result<Decimal> {
    let raw = value
        .as_str()
        .map(str::to_owned)
        .or_else(|| value.as_number().map(ToString::to_string))
        .with_context(|| format!("book level {name} must be numeric"))?;
    raw.parse()
        .with_context(|| format!("book level {name} is invalid"))
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use serde_json::json;

    use super::*;

    fn time(millis: i64) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(millis).single().unwrap()
    }

    fn scope() -> BtcOrderbookMarketScope {
        BtcOrderbookMarketScope {
            market_id: "market".to_string(),
            condition_id: "condition".to_string(),
            up_token_id: "up".to_string(),
            down_token_id: "down".to_string(),
            window_start: time(0),
            window_end: time(300_000),
        }
    }

    fn next_scope() -> BtcOrderbookMarketScope {
        BtcOrderbookMarketScope {
            market_id: "next-market".to_string(),
            condition_id: "condition".to_string(),
            up_token_id: "next-up".to_string(),
            down_token_id: "next-down".to_string(),
            window_start: time(300_000),
            window_end: time(600_000),
        }
    }

    fn book(asset_id: &str, received_at: i64, row: i64) -> BtcOrderbookArchiveEvent {
        BtcOrderbookArchiveEvent {
            source_row_number: row,
            provider_received_at: time(received_at),
            source_timestamp: time(received_at - 1),
            condition_id: "condition".to_string(),
            asset_id: asset_id.to_string(),
            event_type: "book".to_string(),
            bids: Some(json!([["0.40", "10"]])),
            asks: Some(json!([["0.45", "4"], ["0.50", "216"]])),
            price: None,
            size: None,
            side: None,
            best_bid: None,
            best_ask: None,
            fee_rate_bps: None,
            transaction_hash: None,
            old_tick_size: None,
            new_tick_size: None,
        }
    }

    #[test]
    fn reconstructs_causal_decision_window_snapshots_for_both_outcomes() {
        let mut reconstructor = ExecutionSnapshotReconstructor::new(vec![scope()]).unwrap();
        let mut snapshots = Vec::new();
        reconstructor
            .apply(&book("up", 900, 1), &mut snapshots)
            .unwrap();
        reconstructor
            .apply(&book("down", 900, 2), &mut snapshots)
            .unwrap();
        reconstructor.finish(time(300_000), &mut snapshots);
        assert_eq!(snapshots.len(), EXECUTION_SNAPSHOTS_PER_MARKET);
        assert_eq!(snapshots[0].sampled_at, time(1_000));
        assert_eq!(snapshots[58].sampled_at, time(59_000));
        assert_eq!(snapshots[59].sampled_at, time(60_000));
        assert_eq!(snapshots[95].sampled_at, time(240_000));
        assert_eq!(snapshots[0].quality_flags, 0);
        assert_eq!(snapshots[0].up_best_ask, Some(Decimal::new(45, 2)));
        assert_eq!(snapshots[0].up_ask_vwap_5, Some(Decimal::new(46, 2)));
        assert_eq!(snapshots[0].up_ask_vwap_20, Some(Decimal::new(49, 2)));
        assert_eq!(snapshots[0].up_ask_vwap_25, Some(Decimal::new(492, 3)));
        assert_eq!(snapshots[0].up_ask_vwap_200, Some(Decimal::new(499, 3)));
    }

    #[test]
    fn never_applies_an_event_before_its_provider_receipt_time() {
        let mut reconstructor = ExecutionSnapshotReconstructor::new(vec![scope()]).unwrap();
        let mut snapshots = Vec::new();
        reconstructor
            .apply(&book("up", 1_100, 1), &mut snapshots)
            .unwrap();
        reconstructor
            .apply(&book("down", 1_100, 2), &mut snapshots)
            .unwrap();
        reconstructor.finish(time(300_000), &mut snapshots);
        assert_ne!(snapshots[0].quality_flags & QUALITY_UP_MISSING, 0);
        assert_ne!(snapshots[0].quality_flags & QUALITY_DOWN_MISSING, 0);
        assert_eq!(snapshots[1].up_provider_received_at, Some(time(1_100)));
    }

    #[test]
    fn empty_event_window_emits_explicit_missing_book_observations() {
        let mut reconstructor = ExecutionSnapshotReconstructor::new(vec![scope()]).unwrap();
        let mut snapshots = Vec::new();

        reconstructor.finish(time(300_000), &mut snapshots);

        assert_eq!(snapshots.len(), EXECUTION_SNAPSHOTS_PER_MARKET);
        assert!(snapshots.iter().all(|snapshot| {
            snapshot.quality_flags & (QUALITY_UP_MISSING | QUALITY_DOWN_MISSING)
                == QUALITY_UP_MISSING | QUALITY_DOWN_MISSING
        }));
        assert!(snapshots
            .iter()
            .all(|snapshot| snapshot.up_best_ask.is_none() && snapshot.down_best_ask.is_none()));
    }

    #[test]
    fn accepts_provider_ordering_within_each_outcome_stream() {
        let mut reconstructor = ExecutionSnapshotReconstructor::new(vec![scope()]).unwrap();
        let mut snapshots = Vec::new();
        reconstructor
            .apply(&book("up", 1_100, 1), &mut snapshots)
            .unwrap();
        reconstructor
            .apply(&book("down", 900, 2), &mut snapshots)
            .unwrap();
        reconstructor.finish(time(300_000), &mut snapshots);

        assert_eq!(snapshots.len(), EXECUTION_SNAPSHOTS_PER_MARKET);
        assert_ne!(snapshots[0].quality_flags & QUALITY_UP_MISSING, 0);
        assert_eq!(snapshots[0].quality_flags & QUALITY_DOWN_MISSING, 0);
        assert_eq!(
            snapshots[3].quality_flags,
            QUALITY_UP_STALE | QUALITY_DOWN_STALE
        );
    }

    #[test]
    fn rejects_receipt_time_regression_within_one_outcome_stream() {
        let mut reconstructor = ExecutionSnapshotReconstructor::new(vec![scope()]).unwrap();
        let mut snapshots = Vec::new();
        reconstructor
            .apply(&book("up", 1_100, 1), &mut snapshots)
            .unwrap();
        let error = reconstructor
            .apply(&book("up", 1_000, 2), &mut snapshots)
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("one outcome token are not ordered"));
    }

    #[test]
    fn carries_the_next_market_seed_without_rereading_the_previous_hour() {
        let next = next_scope();
        let mut first = ExecutionSnapshotReconstructor::new(vec![scope(), next.clone()]).unwrap();
        let mut discarded = Vec::new();
        let mut next_up = book("next-up", 299_900, 1);
        next_up.condition_id = next.condition_id.clone();
        let mut next_down = book("next-down", 299_900, 2);
        next_down.condition_id = next.condition_id.clone();
        first.apply(&next_up, &mut discarded).unwrap();
        first.apply(&next_down, &mut discarded).unwrap();
        first.finish_before(time(300_000), &mut discarded);
        let seed = first.market_seed(&next.market_id).unwrap();

        let mut second =
            ExecutionSnapshotReconstructor::new_with_seed(vec![next], Some(seed)).unwrap();
        let mut snapshots = Vec::new();
        second.finish(time(302_000), &mut snapshots);
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0].sampled_at, time(301_000));
        assert_eq!(snapshots[0].quality_flags, 0);
        assert_eq!(
            snapshots[1].quality_flags,
            QUALITY_UP_STALE | QUALITY_DOWN_STALE
        );
        assert_eq!(snapshots[0].up_provider_received_at, Some(time(299_900)));
    }
}
