use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    str::FromStr,
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, TimeZone, Utc};
use rust_decimal::{Decimal, RoundingStrategy};
use serde_json::Value;
use uuid::Uuid;

pub use crate::ingestion::job::BinanceL2OneSecondFeature;

pub const BINANCE_SPOT_L2_SYMBOL: &str = "BTCUSDT";
pub const BINANCE_SPOT_L2_FEATURE_SCHEMA_VERSION: &str =
    "binance-spot-btcusdt-l2-one-second-features-v1";
pub const BINANCE_SPOT_L2_AVAILABILITY_OFFSET_MILLISECONDS: i64 = 100;
pub const BINANCE_SPOT_L2_MAX_SOURCE_AGE_MILLISECONDS: i64 = 1_000;
pub const BINANCE_SPOT_L2_MAX_INFERENCE_AGE_MILLISECONDS: i64 = 2_000;
pub const BINANCE_SPOT_L2_FEATURE_WINDOW_CAPACITY: usize = 64;
pub const BINANCE_SPOT_L2_MAX_UPDATE_LEVELS: usize = 10_000;
pub const BINANCE_SPOT_L2_MAX_SNAPSHOT_LEVELS_PER_SIDE: usize = 5_000;
pub const BINANCE_SPOT_L2_MIN_SNAPSHOT_LEVELS_PER_SIDE: usize = 100;
pub const BINANCE_SPOT_L2_MAX_BOOK_LEVELS_PER_SIDE: usize = 100_000;

const ROLLING_HORIZONS_SECONDS: [i64; 5] = [1, 5, 15, 30, 60];
const PERSISTED_DECIMAL_SCALE: u32 = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinanceSpotL2Side {
    Bid,
    Ask,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinanceSpotL2Level {
    pub price: Decimal,
    pub quantity: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinanceSpotL2DepthUpdate {
    pub event_time: DateTime<Utc>,
    pub first_update_id: u64,
    pub final_update_id: u64,
    pub bids: Vec<BinanceSpotL2Level>,
    pub asks: Vec<BinanceSpotL2Level>,
}

pub type BinanceSpotDepthUpdate = BinanceSpotL2DepthUpdate;

impl BinanceSpotL2DepthUpdate {
    pub fn level_count(&self) -> usize {
        self.bids.len().saturating_add(self.asks.len())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinanceSpotL2DepthSnapshot {
    pub last_update_id: u64,
    pub bids: Vec<BinanceSpotL2Level>,
    pub asks: Vec<BinanceSpotL2Level>,
}

pub fn parse_depth_update(text: &str) -> Result<BinanceSpotL2DepthUpdate> {
    let value: Value =
        serde_json::from_str(text).context("failed to decode Binance spot L2 JSON")?;
    let object = value
        .as_object()
        .context("Binance spot L2 update was not a JSON object")?;
    let event_type = object
        .get("e")
        .and_then(Value::as_str)
        .context("Binance spot L2 update omitted event type")?;
    if event_type != "depthUpdate" {
        bail!("Binance spot L2 frame had unsupported event type {event_type}");
    }
    let symbol = object
        .get("s")
        .and_then(Value::as_str)
        .context("Binance spot L2 update omitted symbol")?;
    if symbol != BINANCE_SPOT_L2_SYMBOL {
        bail!("Binance spot L2 update had unexpected symbol {symbol}");
    }
    let event_time_milliseconds = required_u64(object.get("E"), "event time")?;
    let event_time_milliseconds = i64::try_from(event_time_milliseconds)
        .context("Binance spot L2 event time exceeded UTC range")?;
    let event_time = Utc
        .timestamp_millis_opt(event_time_milliseconds)
        .single()
        .context("Binance spot L2 event time was invalid")?;
    let first_update_id = required_u64(object.get("U"), "first update id")?;
    let final_update_id = required_u64(object.get("u"), "final update id")?;
    if first_update_id > final_update_id {
        bail!("Binance spot L2 update id range was inverted");
    }
    let bids = parse_levels(object.get("b"), "bids", true)?;
    let asks = parse_levels(object.get("a"), "asks", true)?;
    if bids.len().saturating_add(asks.len()) > BINANCE_SPOT_L2_MAX_UPDATE_LEVELS {
        bail!("Binance spot L2 update exceeded its bounded price-level count");
    }
    if bids.is_empty() && asks.is_empty() {
        bail!("Binance spot L2 update contained no price levels");
    }
    Ok(BinanceSpotL2DepthUpdate {
        event_time,
        first_update_id,
        final_update_id,
        bids,
        asks,
    })
}

pub fn parse_depth_snapshot(value: &Value) -> Result<BinanceSpotL2DepthSnapshot> {
    let object = value
        .as_object()
        .context("Binance spot L2 snapshot was not a JSON object")?;
    let last_update_id = required_u64(object.get("lastUpdateId"), "snapshot update id")?;
    let bids = parse_levels(object.get("bids"), "snapshot bids", false)?;
    let asks = parse_levels(object.get("asks"), "snapshot asks", false)?;
    for (name, levels) in [("bid", &bids), ("ask", &asks)] {
        if levels.len() < BINANCE_SPOT_L2_MIN_SNAPSHOT_LEVELS_PER_SIDE {
            bail!("Binance spot L2 snapshot had fewer than 100 {name} levels");
        }
        if levels.len() > BINANCE_SPOT_L2_MAX_SNAPSHOT_LEVELS_PER_SIDE {
            bail!("Binance spot L2 snapshot exceeded 5000 {name} levels");
        }
    }
    Ok(BinanceSpotL2DepthSnapshot {
        last_update_id,
        bids,
        asks,
    })
}

fn required_u64(value: Option<&Value>, field: &str) -> Result<u64> {
    value
        .and_then(Value::as_u64)
        .with_context(|| format!("Binance spot L2 {field} was not an unsigned integer"))
}

fn parse_levels(
    value: Option<&Value>,
    field: &str,
    allow_zero: bool,
) -> Result<Vec<BinanceSpotL2Level>> {
    let rows = value
        .and_then(Value::as_array)
        .with_context(|| format!("Binance spot L2 {field} was not an array"))?;
    let mut levels = Vec::with_capacity(rows.len());
    let mut prices = BTreeSet::new();
    for row in rows {
        let row = row
            .as_array()
            .with_context(|| format!("Binance spot L2 {field} level was not an array"))?;
        if row.len() < 2 {
            bail!("Binance spot L2 {field} level omitted price or quantity");
        }
        let price = row[0]
            .as_str()
            .context("Binance spot L2 level price was not a string")
            .and_then(|value| Decimal::from_str(value).context("invalid Binance spot L2 price"))?;
        let quantity = row[1]
            .as_str()
            .context("Binance spot L2 level quantity was not a string")
            .and_then(|value| {
                Decimal::from_str(value).context("invalid Binance spot L2 quantity")
            })?;
        if price <= Decimal::ZERO || quantity < Decimal::ZERO || (!allow_zero && quantity.is_zero())
        {
            bail!("Binance spot L2 {field} contained a non-positive price or invalid quantity");
        }
        if !prices.insert(price) {
            bail!("Binance spot L2 {field} repeated a price level");
        }
        levels.push(BinanceSpotL2Level { price, quantity });
    }
    Ok(levels)
}

#[derive(Debug, Clone, Default, PartialEq)]
struct QuoteFlow {
    bid_replenishment: Decimal,
    ask_replenishment: Decimal,
    bid_churn: Decimal,
    ask_churn: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
struct BaseSecondState {
    second_start: DateTime<Utc>,
    source_event_timestamp: DateTime<Utc>,
    provider_received_at: DateTime<Utc>,
    available_at: DateTime<Utc>,
    source_update_id: i64,
    midpoint: Decimal,
    microprice: Decimal,
    spread_bps: Decimal,
    bid_depth_5: Decimal,
    ask_depth_5: Decimal,
    imbalance_5: Decimal,
    bid_depth_10: Decimal,
    ask_depth_10: Decimal,
    imbalance_10: Decimal,
    bid_depth_20: Decimal,
    ask_depth_20: Decimal,
    imbalance_20: Decimal,
    bid_depth_slope_20: Decimal,
    ask_depth_slope_20: Decimal,
    bid_depth_concentration_20: Decimal,
    ask_depth_concentration_20: Decimal,
    bid_quote_replenishment_1s: Decimal,
    ask_quote_replenishment_1s: Decimal,
    bid_quote_churn_1s: Decimal,
    ask_quote_churn_1s: Decimal,
}

#[derive(Debug, Clone)]
struct PendingSecond {
    second_start: DateTime<Utc>,
    latest_state: BaseSecondState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinanceSpotL2ApplyOutcome {
    Applied {
        features: Vec<BinanceL2OneSecondFeature>,
        synchronized_now: bool,
    },
    IgnoredStale,
    SequenceGap {
        expected_update_id: u64,
        first_update_id: u64,
        final_update_id: u64,
    },
}

pub type BinanceSpotL2UpdateOutcome = BinanceSpotL2ApplyOutcome;

#[derive(Debug, Default)]
pub struct BinanceSpotL2Engine {
    bids: BTreeMap<Decimal, Decimal>,
    asks: BTreeMap<Decimal, Decimal>,
    last_update_id: Option<u64>,
    synchronized: bool,
    last_source_event_timestamp: Option<DateTime<Utc>>,
    last_available_at: Option<DateTime<Utc>>,
    current_flow_second: Option<DateTime<Utc>>,
    current_flow: QuoteFlow,
    pending_second: Option<PendingSecond>,
    rolling: VecDeque<BaseSecondState>,
}

impl BinanceSpotL2Engine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn install_snapshot(
        &mut self,
        snapshot: BinanceSpotL2DepthSnapshot,
        _received_at: DateTime<Utc>,
    ) -> Result<()> {
        self.invalidate();
        for level in snapshot.bids {
            self.bids.insert(level.price, level.quantity);
        }
        for level in snapshot.asks {
            self.asks.insert(level.price, level.quantity);
        }
        self.require_bounded_book()?;
        self.require_valid_top_book()?;
        self.last_update_id = Some(snapshot.last_update_id);
        Ok(())
    }

    pub fn apply_update(
        &mut self,
        update: BinanceSpotL2DepthUpdate,
        provider_received_at: DateTime<Utc>,
    ) -> Result<BinanceSpotL2ApplyOutcome> {
        let current = self
            .last_update_id
            .context("Binance spot L2 update arrived before a snapshot")?;
        if update.final_update_id <= current {
            return Ok(BinanceSpotL2ApplyOutcome::IgnoredStale);
        }
        let expected = current
            .checked_add(1)
            .context("Binance spot L2 sequence overflow")?;
        if update.first_update_id > expected || update.final_update_id < expected {
            return Ok(BinanceSpotL2ApplyOutcome::SequenceGap {
                expected_update_id: expected,
                first_update_id: update.first_update_id,
                final_update_id: update.final_update_id,
            });
        }
        if self
            .last_source_event_timestamp
            .is_some_and(|previous| update.event_time < previous)
        {
            bail!("Binance spot L2 source timestamp regressed");
        }
        let offset = Duration::milliseconds(BINANCE_SPOT_L2_AVAILABILITY_OFFSET_MILLISECONDS);
        let reported_available_at = provider_received_at
            .max(update.event_time)
            .checked_add_signed(offset)
            .context("Binance spot L2 availability timestamp overflow")?;
        let available_at = self
            .last_available_at
            .map_or(reported_available_at, |previous| {
                previous.max(reported_available_at)
            });
        if available_at.signed_duration_since(update.event_time)
            > Duration::milliseconds(BINANCE_SPOT_L2_MAX_SOURCE_AGE_MILLISECONDS)
        {
            bail!("Binance spot L2 update exceeded the source-age qualification bound");
        }
        self.advance_flow_second(floor_utc_second(available_at))?;
        for level in &update.bids {
            self.apply_level(BinanceSpotL2Side::Bid, level)?;
        }
        for level in &update.asks {
            self.apply_level(BinanceSpotL2Side::Ask, level)?;
        }
        self.require_bounded_book()?;
        self.require_valid_top_book()?;
        let synchronized_now = !self.synchronized;
        self.synchronized = true;
        self.last_update_id = Some(update.final_update_id);
        self.last_source_event_timestamp = Some(update.event_time);
        self.last_available_at = Some(available_at);
        let source_update_id = i64::try_from(update.final_update_id)
            .context("Binance spot L2 update id exceeded storage range")?;
        let state = self
            .build_base_state(
                update.event_time,
                provider_received_at,
                available_at,
                source_update_id,
            )?
            .context("Binance spot L2 book did not contain a valid top 20")?;
        let features = self.observe_state(state)?;
        Ok(BinanceSpotL2ApplyOutcome::Applied {
            features,
            synchronized_now,
        })
    }

    pub fn advance_time(&mut self, now: DateTime<Utc>) -> Result<Vec<BinanceL2OneSecondFeature>> {
        let Some(pending) = self.pending_second.as_ref() else {
            return Ok(Vec::new());
        };
        if floor_utc_second(now) <= pending.second_start {
            return Ok(Vec::new());
        }
        self.finalize_pending_second()
    }

    pub fn invalidate(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.last_update_id = None;
        self.synchronized = false;
        self.last_source_event_timestamp = None;
        self.last_available_at = None;
        self.current_flow_second = None;
        self.current_flow = QuoteFlow::default();
        self.pending_second = None;
        self.rolling.clear();
    }

    pub fn synchronized(&self) -> bool {
        self.synchronized
    }

    pub fn update_id(&self) -> Option<u64> {
        self.last_update_id
    }

    pub fn level_counts(&self) -> (usize, usize) {
        (self.bids.len(), self.asks.len())
    }

    fn apply_level(&mut self, side: BinanceSpotL2Side, level: &BinanceSpotL2Level) -> Result<()> {
        let map = match side {
            BinanceSpotL2Side::Bid => &mut self.bids,
            BinanceSpotL2Side::Ask => &mut self.asks,
        };
        let previous = map.get(&level.price).copied().unwrap_or(Decimal::ZERO);
        if level.quantity.is_zero() {
            map.remove(&level.price);
        } else {
            map.insert(level.price, level.quantity);
        }
        let quantity_delta = level
            .quantity
            .checked_sub(previous)
            .context("Binance spot L2 quantity delta overflow")?;
        let quote_delta = level
            .price
            .checked_mul(quantity_delta.abs())
            .context("Binance spot L2 quote-flow multiplication overflow")?;
        let target = match (side, quantity_delta.is_sign_positive()) {
            (BinanceSpotL2Side::Bid, true) => &mut self.current_flow.bid_replenishment,
            (BinanceSpotL2Side::Ask, true) => &mut self.current_flow.ask_replenishment,
            (BinanceSpotL2Side::Bid, false) => &mut self.current_flow.bid_churn,
            (BinanceSpotL2Side::Ask, false) => &mut self.current_flow.ask_churn,
        };
        *target = target
            .checked_add(quote_delta)
            .context("Binance spot L2 quote-flow addition overflow")?;
        Ok(())
    }

    fn advance_flow_second(&mut self, second_start: DateTime<Utc>) -> Result<()> {
        if let Some(current) = self.current_flow_second {
            if second_start < current {
                bail!("Binance spot L2 availability time regressed");
            }
            if second_start > current {
                self.current_flow = QuoteFlow::default();
            }
        }
        self.current_flow_second = Some(second_start);
        Ok(())
    }

    fn observe_state(&mut self, state: BaseSecondState) -> Result<Vec<BinanceL2OneSecondFeature>> {
        if let Some(pending) = self.pending_second.as_mut() {
            if state.second_start < pending.second_start {
                bail!("Binance spot L2 feature time regressed");
            }
            if state.second_start == pending.second_start {
                pending.latest_state = state;
                return Ok(Vec::new());
            }
        }
        let features = self.finalize_pending_second()?;
        self.pending_second = Some(PendingSecond {
            second_start: state.second_start,
            latest_state: state,
        });
        Ok(features)
    }

    fn finalize_pending_second(&mut self) -> Result<Vec<BinanceL2OneSecondFeature>> {
        let Some(pending) = self.pending_second.take() else {
            return Ok(Vec::new());
        };
        let state = pending.latest_state;
        if self.rolling.back().is_some_and(|previous| {
            previous
                .second_start
                .checked_add_signed(Duration::seconds(1))
                != Some(state.second_start)
        }) {
            self.rolling.clear();
        }
        let feature = qualified_feature(&state, &self.rolling)?;
        self.rolling.push_back(state);
        while self.rolling.len() > 61 {
            self.rolling.pop_front();
        }
        Ok(feature.into_iter().collect())
    }

    fn require_bounded_book(&self) -> Result<()> {
        if self.bids.len() > BINANCE_SPOT_L2_MAX_BOOK_LEVELS_PER_SIDE
            || self.asks.len() > BINANCE_SPOT_L2_MAX_BOOK_LEVELS_PER_SIDE
        {
            bail!("Binance spot L2 book exceeded its memory safety bound");
        }
        Ok(())
    }

    fn require_valid_top_book(&self) -> Result<()> {
        let bids = self.bids.iter().rev().take(20).collect::<Vec<_>>();
        let asks = self.asks.iter().take(20).collect::<Vec<_>>();
        if bids.len() < 20 || asks.len() < 20 {
            bail!("Binance spot L2 book did not contain 20 levels per side");
        }
        if bids[0].0 >= asks[0].0 || bids[0].1 <= &Decimal::ZERO || asks[0].1 <= &Decimal::ZERO {
            bail!("Binance spot L2 book was crossed or had invalid top quantity");
        }
        Ok(())
    }

    fn build_base_state(
        &self,
        source_event_timestamp: DateTime<Utc>,
        provider_received_at: DateTime<Utc>,
        available_at: DateTime<Utc>,
        source_update_id: i64,
    ) -> Result<Option<BaseSecondState>> {
        let bids = self
            .bids
            .iter()
            .rev()
            .take(20)
            .map(|(price, quantity)| (*price, *quantity))
            .collect::<Vec<_>>();
        let asks = self
            .asks
            .iter()
            .take(20)
            .map(|(price, quantity)| (*price, *quantity))
            .collect::<Vec<_>>();
        if bids.len() < 20 || asks.len() < 20 {
            return Ok(None);
        }
        let best_bid = bids[0];
        let best_ask = asks[0];
        if best_bid.0 >= best_ask.0 || best_bid.1 <= Decimal::ZERO || best_ask.1 <= Decimal::ZERO {
            return Ok(None);
        }
        let midpoint = checked_div(
            checked_add(best_bid.0, best_ask.0, "midpoint addition")?,
            Decimal::from(2u32),
            "midpoint division",
        )?;
        let top_quantity = checked_add(best_bid.1, best_ask.1, "top quantity addition")?;
        let microprice = checked_div(
            checked_add(
                checked_mul(best_ask.0, best_bid.1, "microprice bid term")?,
                checked_mul(best_bid.0, best_ask.1, "microprice ask term")?,
                "microprice numerator",
            )?,
            top_quantity,
            "microprice division",
        )?;
        let spread_bps = basis_points_delta(best_ask.0, best_bid.0, midpoint)?;
        let (bid_depth_5, bid_depth_10, bid_depth_20) = tier_depths(&bids)?;
        let (ask_depth_5, ask_depth_10, ask_depth_20) = tier_depths(&asks)?;
        let imbalance_5 = imbalance(bid_depth_5, ask_depth_5)?;
        let imbalance_10 = imbalance(bid_depth_10, ask_depth_10)?;
        let imbalance_20 = imbalance(bid_depth_20, ask_depth_20)?;
        let bid_depth_slope_20 = checked_div(
            basis_points_delta(best_bid.0, bids[19].0, best_bid.0)?,
            bid_depth_20,
            "bid depth slope",
        )?;
        let ask_depth_slope_20 = checked_div(
            basis_points_delta(asks[19].0, best_ask.0, best_ask.0)?,
            ask_depth_20,
            "ask depth slope",
        )?;
        Ok(Some(BaseSecondState {
            second_start: floor_utc_second(available_at),
            source_event_timestamp,
            provider_received_at,
            available_at,
            source_update_id,
            midpoint,
            microprice,
            spread_bps,
            bid_depth_5,
            ask_depth_5,
            imbalance_5,
            bid_depth_10,
            ask_depth_10,
            imbalance_10,
            bid_depth_20,
            ask_depth_20,
            imbalance_20,
            bid_depth_slope_20,
            ask_depth_slope_20,
            bid_depth_concentration_20: checked_div(
                bid_depth_5,
                bid_depth_20,
                "bid depth concentration",
            )?,
            ask_depth_concentration_20: checked_div(
                ask_depth_5,
                ask_depth_20,
                "ask depth concentration",
            )?,
            bid_quote_replenishment_1s: self.current_flow.bid_replenishment,
            ask_quote_replenishment_1s: self.current_flow.ask_replenishment,
            bid_quote_churn_1s: self.current_flow.bid_churn,
            ask_quote_churn_1s: self.current_flow.ask_churn,
        }))
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BinanceSpotL2FeatureWindow {
    connection_id: Option<Uuid>,
    synchronized: bool,
    last_update_id: Option<u64>,
    features: VecDeque<BinanceL2OneSecondFeature>,
}

impl BinanceSpotL2FeatureWindow {
    pub fn clear_epoch(&mut self, connection_id: Uuid) {
        self.connection_id = Some(connection_id);
        self.synchronized = false;
        self.last_update_id = None;
        self.features.clear();
    }

    pub fn clear(&mut self) {
        self.connection_id = None;
        self.synchronized = false;
        self.last_update_id = None;
        self.features.clear();
    }

    pub fn mark_synchronized(&mut self, connection_id: Uuid, last_update_id: u64) -> Result<()> {
        if self.connection_id != Some(connection_id) {
            bail!("Binance spot L2 feature epoch changed before synchronization");
        }
        self.synchronized = true;
        self.last_update_id = Some(last_update_id);
        Ok(())
    }

    pub fn publish(
        &mut self,
        connection_id: Uuid,
        last_update_id: u64,
        feature: BinanceL2OneSecondFeature,
    ) -> Result<()> {
        if self.connection_id != Some(connection_id) || !self.synchronized {
            bail!("Binance spot L2 feature was published outside its synchronized epoch");
        }
        if self
            .features
            .back()
            .is_some_and(|previous| previous.second_start >= feature.second_start)
        {
            bail!("Binance spot L2 feature publication was not monotonic");
        }
        self.last_update_id = Some(last_update_id);
        self.features.push_back(feature);
        while self.features.len() > BINANCE_SPOT_L2_FEATURE_WINDOW_CAPACITY {
            self.features.pop_front();
        }
        Ok(())
    }

    pub fn latest_eligible(
        &self,
        feature_as_of: DateTime<Utc>,
    ) -> Option<&BinanceL2OneSecondFeature> {
        if !self.synchronized {
            return None;
        }
        let maximum_age = Duration::milliseconds(BINANCE_SPOT_L2_MAX_INFERENCE_AGE_MILLISECONDS);
        self.features.iter().rev().find(|feature| {
            let availability_age = feature_as_of.signed_duration_since(feature.available_at);
            let source_age = feature_as_of.signed_duration_since(feature.source_event_timestamp);
            availability_age > Duration::zero()
                && availability_age <= maximum_age
                && source_age > Duration::zero()
                && source_age <= maximum_age
        })
    }

    pub fn connection_id(&self) -> Option<Uuid> {
        self.connection_id
    }

    pub fn synchronized(&self) -> bool {
        self.synchronized
    }

    pub fn last_update_id(&self) -> Option<u64> {
        self.last_update_id
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

fn qualified_feature(
    current: &BaseSecondState,
    rolling: &VecDeque<BaseSecondState>,
) -> Result<Option<BinanceL2OneSecondFeature>> {
    let mut prior = Vec::with_capacity(ROLLING_HORIZONS_SECONDS.len());
    for horizon in ROLLING_HORIZONS_SECONDS {
        let expected = current
            .second_start
            .checked_sub_signed(Duration::seconds(horizon))
            .context("Binance spot L2 rolling horizon underflow")?;
        let Some(state) = rolling
            .iter()
            .rev()
            .find(|candidate| candidate.second_start == expected)
        else {
            return Ok(None);
        };
        prior.push(state);
    }
    let changes = prior
        .iter()
        .map(|state| rolling_changes(current, state))
        .collect::<Result<Vec<_>>>()?;
    let feature = BinanceL2OneSecondFeature {
        symbol: BINANCE_SPOT_L2_SYMBOL.to_owned(),
        second_start: current.second_start,
        source_event_timestamp: current.source_event_timestamp,
        provider_received_at: current.provider_received_at,
        available_at: current.available_at,
        source_update_id: current.source_update_id,
        feature_schema_version: BINANCE_SPOT_L2_FEATURE_SCHEMA_VERSION.to_owned(),
        quality_status: "qualified".to_owned(),
        midpoint: current.midpoint,
        microprice: current.microprice,
        spread_bps: current.spread_bps,
        bid_depth_5: current.bid_depth_5,
        ask_depth_5: current.ask_depth_5,
        imbalance_5: current.imbalance_5,
        bid_depth_10: current.bid_depth_10,
        ask_depth_10: current.ask_depth_10,
        imbalance_10: current.imbalance_10,
        bid_depth_20: current.bid_depth_20,
        ask_depth_20: current.ask_depth_20,
        imbalance_20: current.imbalance_20,
        bid_depth_slope_20: current.bid_depth_slope_20,
        ask_depth_slope_20: current.ask_depth_slope_20,
        bid_depth_concentration_20: current.bid_depth_concentration_20,
        ask_depth_concentration_20: current.ask_depth_concentration_20,
        bid_quote_replenishment_1s: current.bid_quote_replenishment_1s,
        ask_quote_replenishment_1s: current.ask_quote_replenishment_1s,
        bid_quote_churn_1s: current.bid_quote_churn_1s,
        ask_quote_churn_1s: current.ask_quote_churn_1s,
        midpoint_change_bps_1s: changes[0].0,
        spread_bps_delta_1s: changes[0].1,
        depth_20_change_bps_1s: changes[0].2,
        imbalance_20_delta_1s: changes[0].3,
        midpoint_change_bps_5s: changes[1].0,
        spread_bps_delta_5s: changes[1].1,
        depth_20_change_bps_5s: changes[1].2,
        imbalance_20_delta_5s: changes[1].3,
        midpoint_change_bps_15s: changes[2].0,
        spread_bps_delta_15s: changes[2].1,
        depth_20_change_bps_15s: changes[2].2,
        imbalance_20_delta_15s: changes[2].3,
        midpoint_change_bps_30s: changes[3].0,
        spread_bps_delta_30s: changes[3].1,
        depth_20_change_bps_30s: changes[3].2,
        imbalance_20_delta_30s: changes[3].3,
        midpoint_change_bps_60s: changes[4].0,
        spread_bps_delta_60s: changes[4].1,
        depth_20_change_bps_60s: changes[4].2,
        imbalance_20_delta_60s: changes[4].3,
    };
    Ok(Some(quantize_feature(feature)))
}

fn tier_depths(levels: &[(Decimal, Decimal)]) -> Result<(Decimal, Decimal, Decimal)> {
    let mut depth_5 = Decimal::ZERO;
    let mut depth_10 = Decimal::ZERO;
    let mut depth_20 = Decimal::ZERO;
    for (index, (_, quantity)) in levels.iter().enumerate().take(20) {
        depth_20 = checked_add(depth_20, *quantity, "depth-20 addition")?;
        if index < 10 {
            depth_10 = checked_add(depth_10, *quantity, "depth-10 addition")?;
        }
        if index < 5 {
            depth_5 = checked_add(depth_5, *quantity, "depth-5 addition")?;
        }
    }
    if depth_5 <= Decimal::ZERO || depth_10 <= Decimal::ZERO || depth_20 <= Decimal::ZERO {
        bail!("Binance spot L2 book depth was not positive");
    }
    Ok((depth_5, depth_10, depth_20))
}

fn imbalance(bid: Decimal, ask: Decimal) -> Result<Decimal> {
    checked_div(
        bid.checked_sub(ask)
            .context("Binance spot L2 imbalance subtraction overflow")?,
        checked_add(bid, ask, "imbalance denominator")?,
        "imbalance division",
    )
}

fn rolling_changes(
    current: &BaseSecondState,
    previous: &BaseSecondState,
) -> Result<(Decimal, Decimal, Decimal, Decimal)> {
    let current_depth = checked_add(
        current.bid_depth_20,
        current.ask_depth_20,
        "current rolling depth",
    )?;
    let previous_depth = checked_add(
        previous.bid_depth_20,
        previous.ask_depth_20,
        "previous rolling depth",
    )?;
    Ok((
        relative_change_bps(current.midpoint, previous.midpoint)?,
        current
            .spread_bps
            .checked_sub(previous.spread_bps)
            .context("Binance spot L2 spread delta overflow")?,
        relative_change_bps(current_depth, previous_depth)?,
        current
            .imbalance_20
            .checked_sub(previous.imbalance_20)
            .context("Binance spot L2 imbalance delta overflow")?,
    ))
}

fn relative_change_bps(current: Decimal, previous: Decimal) -> Result<Decimal> {
    basis_points_delta(current, previous, previous)
}

fn basis_points_delta(high: Decimal, low: Decimal, denominator: Decimal) -> Result<Decimal> {
    let difference = high
        .checked_sub(low)
        .context("Binance spot L2 basis-point subtraction overflow")?;
    checked_div(
        checked_mul(difference, Decimal::from(10_000u32), "basis-point scaling")?,
        denominator,
        "basis-point division",
    )
}

fn checked_add(left: Decimal, right: Decimal, operation: &str) -> Result<Decimal> {
    left.checked_add(right)
        .with_context(|| format!("Binance spot L2 {operation} overflow"))
}

fn checked_mul(left: Decimal, right: Decimal, operation: &str) -> Result<Decimal> {
    left.checked_mul(right)
        .with_context(|| format!("Binance spot L2 {operation} overflow"))
}

fn checked_div(numerator: Decimal, denominator: Decimal, operation: &str) -> Result<Decimal> {
    if denominator.is_zero() {
        bail!("Binance spot L2 {operation} divided by zero");
    }
    numerator
        .checked_div(denominator)
        .with_context(|| format!("Binance spot L2 {operation} overflow"))
}

fn quantize_feature(mut feature: BinanceL2OneSecondFeature) -> BinanceL2OneSecondFeature {
    macro_rules! quantize {
        ($($field:ident),+ $(,)?) => {
            $(
                feature.$field = feature.$field.round_dp_with_strategy(
                    PERSISTED_DECIMAL_SCALE,
                    RoundingStrategy::MidpointAwayFromZero,
                );
            )+
        };
    }
    quantize!(
        midpoint,
        microprice,
        spread_bps,
        bid_depth_5,
        ask_depth_5,
        imbalance_5,
        bid_depth_10,
        ask_depth_10,
        imbalance_10,
        bid_depth_20,
        ask_depth_20,
        imbalance_20,
        bid_depth_slope_20,
        ask_depth_slope_20,
        bid_depth_concentration_20,
        ask_depth_concentration_20,
        bid_quote_replenishment_1s,
        ask_quote_replenishment_1s,
        bid_quote_churn_1s,
        ask_quote_churn_1s,
        midpoint_change_bps_1s,
        spread_bps_delta_1s,
        depth_20_change_bps_1s,
        imbalance_20_delta_1s,
        midpoint_change_bps_5s,
        spread_bps_delta_5s,
        depth_20_change_bps_5s,
        imbalance_20_delta_5s,
        midpoint_change_bps_15s,
        spread_bps_delta_15s,
        depth_20_change_bps_15s,
        imbalance_20_delta_15s,
        midpoint_change_bps_30s,
        spread_bps_delta_30s,
        depth_20_change_bps_30s,
        imbalance_20_delta_30s,
        midpoint_change_bps_60s,
        spread_bps_delta_60s,
        depth_20_change_bps_60s,
        imbalance_20_delta_60s,
    );
    feature
}

fn floor_utc_second(timestamp: DateTime<Utc>) -> DateTime<Utc> {
    Utc.timestamp_opt(timestamp.timestamp(), 0)
        .single()
        .expect("valid UTC timestamps floor to valid UTC seconds")
}

#[cfg(test)]
mod tests {
    use chrono::TimeDelta;
    use rust_decimal_macros::dec;
    use serde_json::json;
    use tokio::sync::mpsc;

    use crate::ingestion::{
        binance_archive::ArchiveCancellation,
        cryptohft_binance_l2::{
            BinanceSpotL2RangeReplay, BinanceSpotL2ReplayEvent, BinanceSpotL2ReplayLevel,
            BinanceSpotL2ReplaySide, CryptoHftBinanceL2Config,
        },
    };

    use super::*;

    fn at(second: i64, milliseconds: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(second, 0).single().unwrap() + TimeDelta::milliseconds(milliseconds)
    }

    fn snapshot(last_update_id: u64) -> BinanceSpotL2DepthSnapshot {
        BinanceSpotL2DepthSnapshot {
            last_update_id,
            bids: (0..100)
                .map(|index| BinanceSpotL2Level {
                    price: dec!(100) - Decimal::from(index) / dec!(100),
                    quantity: Decimal::from(index + 1),
                })
                .collect(),
            asks: (0..100)
                .map(|index| BinanceSpotL2Level {
                    price: dec!(100.01) + Decimal::from(index) / dec!(100),
                    quantity: Decimal::from(index + 1),
                })
                .collect(),
        }
    }

    fn update(second: i64, update_id: u64, quantity: Decimal) -> BinanceSpotL2DepthUpdate {
        BinanceSpotL2DepthUpdate {
            event_time: at(second, 100),
            first_update_id: update_id,
            final_update_id: update_id,
            bids: vec![BinanceSpotL2Level {
                price: dec!(100),
                quantity,
            }],
            asks: Vec::new(),
        }
    }

    #[test]
    fn strict_parsers_validate_symbol_sequence_and_snapshot_depth() {
        let parsed = parse_depth_update(
            r#"{"e":"depthUpdate","E":1776000000100,"s":"BTCUSDT","U":11,"u":12,"b":[["100.0","2.0"]],"a":[["100.1","0.0"]]}"#,
        )
        .unwrap();
        assert_eq!(parsed.first_update_id, 11);
        assert_eq!(parsed.final_update_id, 12);
        assert_eq!(parsed.level_count(), 2);
        assert!(parse_depth_update(
            r#"{"e":"depthUpdate","E":1776000000100,"s":"ETHUSDT","U":11,"u":12,"b":[["100","1"]],"a":[]}"#,
        )
        .is_err());

        let levels = (0..100)
            .map(|index| json!([(10000 - index).to_string(), "1"]))
            .collect::<Vec<_>>();
        let asks = (0..100)
            .map(|index| json!([(10100 + index).to_string(), "1"]))
            .collect::<Vec<_>>();
        let parsed = parse_depth_snapshot(&json!({
            "lastUpdateId": 10,
            "bids": levels,
            "asks": asks,
        }))
        .unwrap();
        assert_eq!(parsed.last_update_id, 10);
    }

    #[test]
    fn snapshot_bridge_ignores_stale_accepts_overlap_and_rejects_gap() {
        let mut engine = BinanceSpotL2Engine::default();
        engine
            .install_snapshot(snapshot(100), at(1_776_000_000, 0))
            .unwrap();
        let mut stale = update(1_776_000_000, 100, dec!(2));
        stale.first_update_id = 99;
        assert_eq!(
            engine.apply_update(stale, at(1_776_000_000, 110)).unwrap(),
            BinanceSpotL2ApplyOutcome::IgnoredStale
        );
        let mut bridge = update(1_776_000_000, 102, dec!(2));
        bridge.first_update_id = 100;
        assert!(matches!(
            engine.apply_update(bridge, at(1_776_000_000, 110)).unwrap(),
            BinanceSpotL2ApplyOutcome::Applied {
                synchronized_now: true,
                ..
            }
        ));
        assert!(engine.synchronized());
        let gap = update(1_776_000_001, 104, dec!(3));
        assert!(matches!(
            engine.apply_update(gap, at(1_776_000_001, 110)).unwrap(),
            BinanceSpotL2ApplyOutcome::SequenceGap {
                expected_update_id: 103,
                ..
            }
        ));
    }

    #[test]
    fn engine_requires_sixty_contiguous_seconds_and_quantizes_features() {
        let start = 1_776_000_000;
        let mut engine = BinanceSpotL2Engine::default();
        engine.install_snapshot(snapshot(10), at(start, 0)).unwrap();
        let mut emitted = Vec::new();
        for offset in 0..=61 {
            let outcome = engine
                .apply_update(
                    update(
                        start + offset,
                        11 + offset as u64,
                        Decimal::from(2 + offset),
                    ),
                    at(start + offset, 110),
                )
                .unwrap();
            if let BinanceSpotL2ApplyOutcome::Applied { features, .. } = outcome {
                emitted.extend(features);
            }
        }
        assert_eq!(emitted.len(), 1);
        let feature = &emitted[0];
        assert_eq!(
            feature.feature_schema_version,
            BINANCE_SPOT_L2_FEATURE_SCHEMA_VERSION
        );
        assert_eq!(feature.quality_status, "qualified");
        assert!(feature.midpoint.scale() <= PERSISTED_DECIMAL_SCALE);
        assert_eq!(feature.midpoint_change_bps_60s, Decimal::ZERO);
    }

    #[test]
    fn feature_window_is_bounded_strictly_prior_and_epoch_scoped() {
        let connection_id = Uuid::new_v4();
        let mut window = BinanceSpotL2FeatureWindow::default();
        window.clear_epoch(connection_id);
        window.mark_synchronized(connection_id, 1).unwrap();
        let mut engine = BinanceSpotL2Engine::default();
        let start = 1_776_000_000;
        engine.install_snapshot(snapshot(10), at(start, 0)).unwrap();
        let mut feature = None;
        for offset in 0..=61 {
            if let BinanceSpotL2ApplyOutcome::Applied { features, .. } = engine
                .apply_update(
                    update(
                        start + offset,
                        11 + offset as u64,
                        Decimal::from(2 + offset),
                    ),
                    at(start + offset, 110),
                )
                .unwrap()
            {
                feature = features.into_iter().next().or(feature);
            }
        }
        let feature = feature.unwrap();
        let exact = feature.available_at;
        window.publish(connection_id, 72, feature.clone()).unwrap();
        assert!(window.latest_eligible(exact).is_none());
        assert!(window
            .latest_eligible(exact + TimeDelta::milliseconds(1))
            .is_some());
        assert!(window
            .latest_eligible(exact + TimeDelta::milliseconds(2_001))
            .is_none());
        window.clear_epoch(Uuid::new_v4());
        assert!(window.is_empty());
        assert!(!window.synchronized());
    }

    #[test]
    fn zero_quantity_updates_remove_levels_without_unbounded_state() {
        let start = 1_776_000_000;
        let mut engine = BinanceSpotL2Engine::default();
        engine.install_snapshot(snapshot(10), at(start, 0)).unwrap();
        let outcome = engine
            .apply_update(
                BinanceSpotL2DepthUpdate {
                    event_time: at(start, 100),
                    first_update_id: 11,
                    final_update_id: 11,
                    bids: vec![BinanceSpotL2Level {
                        price: dec!(99.01),
                        quantity: Decimal::ZERO,
                    }],
                    asks: Vec::new(),
                },
                at(start, 110),
            )
            .unwrap();
        assert!(matches!(outcome, BinanceSpotL2ApplyOutcome::Applied { .. }));
        assert_eq!(engine.level_counts(), (99, 100));
    }

    #[test]
    fn live_engine_matches_historical_spot_replay_feature_values() {
        let start = at(1_776_000_000, 0);
        let temporary = tempfile::tempdir().unwrap();
        let config = CryptoHftBinanceL2Config::new(
            temporary.path().join("archive"),
            temporary.path().join("work"),
        );
        let (sender, mut receiver) = mpsc::channel(4);
        let mut historical = BinanceSpotL2RangeReplay::new(
            &config,
            start,
            start + TimeDelta::seconds(120),
            1_000,
            sender,
            ArchiveCancellation::default(),
        )
        .unwrap();

        let source_snapshot = snapshot(10);
        let snapshot_levels = source_snapshot
            .bids
            .iter()
            .map(|level| BinanceSpotL2ReplayLevel {
                side: BinanceSpotL2ReplaySide::Bid,
                price: level.price,
                quantity: level.quantity,
            })
            .chain(
                source_snapshot
                    .asks
                    .iter()
                    .map(|level| BinanceSpotL2ReplayLevel {
                        side: BinanceSpotL2ReplaySide::Ask,
                        price: level.price,
                        quantity: level.quantity,
                    }),
            )
            .collect();
        historical
            .process(BinanceSpotL2ReplayEvent {
                event_time_ms: start.timestamp_millis(),
                first_update_id: None,
                final_update_id: None,
                last_update_id: Some(10),
                levels: snapshot_levels,
            })
            .unwrap();

        let mut live = BinanceSpotL2Engine::new();
        live.install_snapshot(source_snapshot, start).unwrap();
        let mut live_features = Vec::new();
        for offset in 0..=61 {
            let event_time = start + TimeDelta::seconds(offset) + TimeDelta::milliseconds(100);
            let update_id = 11 + u64::try_from(offset).unwrap();
            let quantity = Decimal::from(2 + offset);
            historical
                .process(BinanceSpotL2ReplayEvent {
                    event_time_ms: event_time.timestamp_millis(),
                    first_update_id: Some(i64::try_from(update_id).unwrap()),
                    final_update_id: Some(i64::try_from(update_id).unwrap()),
                    last_update_id: None,
                    levels: vec![BinanceSpotL2ReplayLevel {
                        side: BinanceSpotL2ReplaySide::Bid,
                        price: dec!(100),
                        quantity,
                    }],
                })
                .unwrap();
            if let BinanceSpotL2ApplyOutcome::Applied { features, .. } = live
                .apply_update(
                    BinanceSpotL2DepthUpdate {
                        event_time,
                        first_update_id: update_id,
                        final_update_id: update_id,
                        bids: vec![BinanceSpotL2Level {
                            price: dec!(100),
                            quantity,
                        }],
                        asks: Vec::new(),
                    },
                    event_time,
                )
                .unwrap()
            {
                live_features.extend(features);
            }
        }
        live_features.extend(live.advance_time(start + TimeDelta::seconds(63)).unwrap());
        historical.finish().unwrap();
        let historical_features = receiver.try_recv().unwrap();

        assert_eq!(live_features, historical_features);
        assert!(!live_features.is_empty());
    }
}
