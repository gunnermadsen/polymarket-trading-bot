use std::{
    collections::{BTreeMap, HashMap, HashSet},
    str::FromStr,
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use rust_decimal::{Decimal, RoundingStrategy};
use serde_json::Value;
use uuid::Uuid;

use super::types::{
    BinanceAggregateTrade, BookReadiness, BtcIntervalMarket, BtcOutcome, ChainlinkTwap60Point,
    FeedIntegrityStatus, OrderbookCheckpoint, OrderbookLevel, Readiness, RealtimeState,
    ReferencePriceSource, ReferencePriceTick, SourceReadiness,
};

const CHAINLINK_E18_SCALE: u64 = 1_000_000_000_000_000_000;

#[derive(Debug, Clone, PartialEq)]
pub enum ClobMessage {
    Book {
        market_id: String,
        token_id: String,
        bids: Vec<OrderbookLevel>,
        asks: Vec<OrderbookLevel>,
        source_timestamp: DateTime<Utc>,
        source_hash: Option<String>,
    },
    PriceChange {
        market_id: String,
        changes: Vec<PriceChange>,
        source_timestamp: DateTime<Utc>,
    },
    BestBidAsk {
        market_id: String,
        token_id: String,
        best_bid: Option<Decimal>,
        best_ask: Option<Decimal>,
        source_timestamp: DateTime<Utc>,
    },
    TickSizeChange {
        market_id: String,
        token_id: String,
        old_tick_size: Decimal,
        new_tick_size: Decimal,
        source_timestamp: DateTime<Utc>,
    },
    LastTradePrice {
        market_id: String,
        token_id: String,
        price: Decimal,
        size: Decimal,
        source_timestamp: DateTime<Utc>,
    },
    MarketResolved {
        market_id: String,
        winning_token_id: String,
        winning_outcome: String,
        source_timestamp: DateTime<Utc>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PriceChange {
    pub token_id: String,
    pub side: BookUpdateSide,
    pub price: Decimal,
    pub size: Decimal,
    pub source_hash: Option<String>,
    pub best_bid: Option<Decimal>,
    pub best_ask: Option<Decimal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookUpdateSide {
    Bid,
    Ask,
}

pub fn parse_clob_messages(value: &Value) -> Result<Vec<ClobMessage>> {
    if let Some(messages) = value.as_array() {
        let mut parsed = Vec::with_capacity(messages.len());
        for message in messages {
            parsed.extend(parse_clob_messages(message)?);
        }
        return Ok(parsed);
    }
    let object = value
        .as_object()
        .context("CLOB market websocket message must be an object or array")?;
    let event_type = required_string(object, &["event_type"])?;
    // Subscription-wide lifecycle notifications do not mutate a token book. The exact current
    // market set is owned by Gamma discovery, so these are benign control messages.
    if event_type == "new_market" {
        return Ok(Vec::new());
    }
    let source_timestamp = timestamp_field(object, &["timestamp"])?;
    let market_id = required_string(object, &["market"])?;
    let parsed = match event_type.as_str() {
        "book" => ClobMessage::Book {
            market_id,
            token_id: required_string(object, &["asset_id"])?,
            bids: parse_levels(object.get("bids"), "bids")?,
            asks: parse_levels(object.get("asks"), "asks")?,
            source_timestamp,
            source_hash: string_field(object, &["hash"]),
        },
        "price_change" => {
            let raw_changes = object
                .get("price_changes")
                .and_then(Value::as_array)
                .context("price_change message is missing price_changes")?;
            if raw_changes.is_empty() {
                bail!("price_change message contains no changes");
            }
            let mut changes = Vec::with_capacity(raw_changes.len());
            for raw in raw_changes {
                let change = raw
                    .as_object()
                    .context("price_change entry must be an object")?;
                changes.push(PriceChange {
                    token_id: required_string(change, &["asset_id"])?,
                    side: match required_string(change, &["side"])?
                        .to_ascii_uppercase()
                        .as_str()
                    {
                        "BUY" => BookUpdateSide::Bid,
                        "SELL" => BookUpdateSide::Ask,
                        side => bail!("unsupported CLOB price_change side {side}"),
                    },
                    price: required_decimal(change, &["price"])?,
                    size: required_decimal(change, &["size"])?,
                    source_hash: string_field(change, &["hash"]),
                    best_bid: optional_decimal_field(change, &["best_bid"])?,
                    best_ask: optional_decimal_field(change, &["best_ask"])?,
                });
            }
            ClobMessage::PriceChange {
                market_id,
                changes,
                source_timestamp,
            }
        }
        "best_bid_ask" => ClobMessage::BestBidAsk {
            market_id,
            token_id: required_string(object, &["asset_id"])?,
            best_bid: optional_decimal_field(object, &["best_bid"])?,
            best_ask: optional_decimal_field(object, &["best_ask"])?,
            source_timestamp,
        },
        "tick_size_change" => ClobMessage::TickSizeChange {
            market_id,
            token_id: required_string(object, &["asset_id"])?,
            old_tick_size: required_decimal(object, &["old_tick_size"])?,
            new_tick_size: required_decimal(object, &["new_tick_size"])?,
            source_timestamp,
        },
        "last_trade_price" => ClobMessage::LastTradePrice {
            market_id,
            token_id: required_string(object, &["asset_id"])?,
            price: required_decimal(object, &["price"])?,
            size: required_decimal(object, &["size"])?,
            source_timestamp,
        },
        "market_resolved" => ClobMessage::MarketResolved {
            market_id,
            winning_token_id: required_string(object, &["winning_asset_id"])?,
            winning_outcome: required_string(object, &["winning_outcome"])?,
            source_timestamp,
        },
        other => bail!("unsupported CLOB market event type {other}"),
    };
    Ok(vec![parsed])
}

pub fn parse_rtds_reference_tick(
    value: &Value,
    connection_id: Uuid,
    ingest_sequence: u64,
    received_at: DateTime<Utc>,
) -> Result<ReferencePriceTick> {
    let object = value
        .as_object()
        .context("RTDS message must be an object")?;
    let topic = required_string(object, &["topic"])?;
    let event_type = required_string(object, &["type"])?;
    if event_type != "update" {
        bail!("RTDS message is not a live price update");
    }
    let source = match topic.as_str() {
        "crypto_prices" => ReferencePriceSource::RtdsBinance,
        "crypto_prices_chainlink" => ReferencePriceSource::RtdsChainlink,
        other => bail!("unsupported RTDS price topic {other}"),
    };
    let payload = object
        .get("payload")
        .and_then(Value::as_object)
        .context("RTDS price update is missing payload")?;
    let raw_symbol = required_string(payload, &["symbol"])?;
    let expected_symbol = match source {
        ReferencePriceSource::RtdsBinance => "btcusdt",
        ReferencePriceSource::RtdsChainlink => "btc/usd",
        ReferencePriceSource::DirectBinance => unreachable!(),
    };
    if raw_symbol.to_ascii_lowercase() != expected_symbol {
        bail!("RTDS update has unexpected symbol {raw_symbol}");
    }
    let price = required_decimal(payload, &["value"])?;
    let source_timestamp = timestamp_field(payload, &["timestamp"])?;
    let envelope_timestamp = Some(timestamp_field(object, &["timestamp"])?);
    reference_tick(
        source,
        "BTCUSD",
        price,
        source_timestamp,
        envelope_timestamp,
        received_at,
        connection_id,
        ingest_sequence,
        None,
        value.clone(),
    )
}

pub(super) fn parse_rtds_chainlink_twap_60(
    value: &Value,
    received_at: DateTime<Utc>,
) -> Result<ChainlinkTwap60Point> {
    let object = value
        .as_object()
        .context("RTDS TWAP message must be an object")?;
    if required_string(object, &["topic"])? != "crypto_prices_twap_sixty"
        || required_string(object, &["type"])? != "update"
    {
        bail!("RTDS message is not a Chainlink 60-second TWAP update");
    }
    let payload = object
        .get("payload")
        .and_then(Value::as_object)
        .context("RTDS TWAP update is missing payload")?;
    let symbol = required_string(payload, &["symbol"])?;
    if !symbol.eq_ignore_ascii_case("btc/usd") {
        bail!("RTDS TWAP update has unexpected symbol {symbol}");
    }
    let window_seconds = payload
        .get("window_s")
        .and_then(Value::as_u64)
        .context("RTDS TWAP update is missing window_s")?;
    if window_seconds != 60 {
        bail!("RTDS TWAP update has unexpected window {window_seconds}s");
    }
    let fixed_point = required_decimal(payload, &["full_accuracy_value"])?;
    let price = fixed_point / Decimal::from(CHAINLINK_E18_SCALE);
    if price <= Decimal::ZERO {
        bail!("RTDS TWAP update price must be positive");
    }
    Ok(ChainlinkTwap60Point {
        price,
        source_timestamp: timestamp_field(payload, &["timestamp"])?,
        available_at: received_at,
    })
}

pub fn parse_binance_agg_trade(
    value: &Value,
    connection_id: Uuid,
    ingest_sequence: u64,
    received_at: DateTime<Utc>,
) -> Result<ReferencePriceTick> {
    parse_binance_agg_trade_with_details(value, connection_id, ingest_sequence, received_at)
        .map(|(tick, _)| tick)
}

pub fn parse_binance_agg_trade_with_details(
    value: &Value,
    connection_id: Uuid,
    ingest_sequence: u64,
    received_at: DateTime<Utc>,
) -> Result<(ReferencePriceTick, BinanceAggregateTrade)> {
    let trade = parse_binance_aggregate_trade(value)?;
    let object = value
        .as_object()
        .context("Binance aggregate trade must be an object")?;
    let envelope_timestamp = Some(timestamp_field(object, &["E"])?);
    let tick = reference_tick(
        ReferencePriceSource::DirectBinance,
        "BTCUSD",
        trade.price,
        trade.transact_time,
        envelope_timestamp,
        received_at,
        connection_id,
        ingest_sequence,
        Some(trade.aggregate_trade_id.to_string()),
        value.clone(),
    )?;
    Ok((tick, trade))
}

pub fn parse_binance_aggregate_trade(value: &Value) -> Result<BinanceAggregateTrade> {
    let object = value
        .as_object()
        .context("Binance aggregate trade must be an object")?;
    if required_string(object, &["e"])? != "aggTrade" {
        bail!("Binance message is not an aggregate trade");
    }
    let symbol = required_string(object, &["s"])?;
    if !symbol.eq_ignore_ascii_case("BTCUSDT") {
        bail!("Binance aggregate trade has unexpected symbol {symbol}");
    }
    let aggregate_trade_id = required_u64(object, &["a"])?;
    let price = required_decimal(object, &["p"])?;
    let quantity = required_decimal(object, &["q"])?;
    let first_trade_id = required_u64(object, &["f"])?;
    let last_trade_id = required_u64(object, &["l"])?;
    let transact_time = timestamp_field(object, &["T"])?;
    let is_buyer_maker = object
        .get("m")
        .and_then(Value::as_bool)
        .context("invalid boolean field m")?;
    if price <= Decimal::ZERO || quantity <= Decimal::ZERO || first_trade_id > last_trade_id {
        bail!("Binance aggregate trade has invalid price, quantity, or trade range");
    }
    Ok(BinanceAggregateTrade {
        aggregate_trade_id,
        price,
        quantity,
        first_trade_id,
        last_trade_id,
        transact_time,
        is_buyer_maker,
    })
}

fn reference_tick(
    source: ReferencePriceSource,
    symbol: &str,
    price: Decimal,
    source_timestamp: DateTime<Utc>,
    envelope_timestamp: Option<DateTime<Utc>>,
    received_at: DateTime<Utc>,
    connection_id: Uuid,
    ingest_sequence: u64,
    source_event_id: Option<String>,
    raw_payload: Value,
) -> Result<ReferencePriceTick> {
    // Keep the in-memory tick identical to its durable representation. PostgreSQL rounds
    // numeric(30,10) values away from zero at the midpoint and stores timestamptz values at
    // microsecond precision. Boundary ticks are compared exactly after persistence and restart,
    // so allowing additional live precision here would turn an idempotent insert into a false
    // immutable-data conflict.
    let price = price.round_dp_with_strategy(10, RoundingStrategy::MidpointAwayFromZero);
    ensure_positive_price(price)?;
    let source_timestamp = canonical_timestamp(source_timestamp);
    let envelope_timestamp = envelope_timestamp.map(canonical_timestamp);
    let received_at = canonical_timestamp(received_at);
    let dedup_key = format!(
        "{}:{}:{}:{}:{}",
        source.as_str(),
        symbol,
        source_timestamp.timestamp_millis(),
        source_event_id.as_deref().unwrap_or("-"),
        price.normalize()
    );
    Ok(ReferencePriceTick {
        tick_id: Uuid::new_v5(&Uuid::NAMESPACE_URL, dedup_key.as_bytes()),
        dedup_key,
        source,
        symbol: symbol.to_string(),
        price,
        source_timestamp,
        envelope_timestamp,
        received_at,
        connection_id,
        ingest_sequence,
        source_event_id,
        raw_payload,
    })
}

fn canonical_timestamp(timestamp: DateTime<Utc>) -> DateTime<Utc> {
    DateTime::from_timestamp_micros(timestamp.timestamp_micros())
        .expect("a valid DateTime must remain valid at microsecond precision")
}

#[derive(Debug, Clone)]
struct FeedBook {
    market_id: String,
    wire_market_id: String,
    token_id: String,
    outcome: BtcOutcome,
    tick_size: Decimal,
    bids: BTreeMap<Decimal, Decimal>,
    asks: BTreeMap<Decimal, Decimal>,
    bootstrapped: bool,
    integrity_status: FeedIntegrityStatus,
    source_timestamp: Option<DateTime<Utc>>,
    received_at: Option<DateTime<Utc>>,
    source_hash: Option<String>,
    ingest_sequence: u64,
}

impl FeedBook {
    fn new(
        market_id: String,
        wire_market_id: String,
        token_id: String,
        outcome: BtcOutcome,
        tick_size: Decimal,
    ) -> Self {
        Self {
            market_id,
            wire_market_id,
            token_id,
            outcome,
            tick_size,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            bootstrapped: false,
            integrity_status: FeedIntegrityStatus::PreSnapshot,
            source_timestamp: None,
            received_at: None,
            source_hash: None,
            ingest_sequence: 0,
        }
    }

    fn matches_market(&self, market_id: &str) -> bool {
        self.market_id == market_id || self.wire_market_id == market_id
    }

    fn best_bid(&self) -> Option<Decimal> {
        self.bids.keys().next_back().copied()
    }

    fn best_ask(&self) -> Option<Decimal> {
        self.asks.keys().next().copied()
    }

    fn validate(&mut self) {
        self.integrity_status = if matches!(
            (self.best_bid(), self.best_ask()),
            (Some(bid), Some(ask)) if bid >= ask
        ) {
            FeedIntegrityStatus::CrossedBook
        } else {
            FeedIntegrityStatus::Ok
        };
    }

    /// Reconcile only levels that the venue's authoritative top-of-book proves stale.
    /// Missing levels are never fabricated: if the advertised top is absent after pruning,
    /// the candidate is quarantined until a full snapshot repairs it.
    fn reconcile_advertised_top(
        &mut self,
        best_bid: Option<Decimal>,
        best_ask: Option<Decimal>,
        undo: &mut Vec<(BookUpdateSide, Decimal, Option<Decimal>)>,
    ) -> FeedIntegrityStatus {
        if best_bid.is_some_and(|price| price < Decimal::ZERO || price >= Decimal::ONE)
            || best_ask.is_some_and(|price| price <= Decimal::ZERO || price > Decimal::ONE)
        {
            self.integrity_status = FeedIntegrityStatus::DecodeError;
            return self.integrity_status;
        }

        if let Some(best_bid) = best_bid {
            if best_bid == Decimal::ZERO {
                let stale_prices = self.bids.keys().copied().collect::<Vec<_>>();
                for price in stale_prices {
                    let previous = self.bids.remove(&price);
                    undo.push((BookUpdateSide::Bid, price, previous));
                }
            } else {
                let stale_prices = self
                    .bids
                    .range((
                        std::ops::Bound::Excluded(best_bid),
                        std::ops::Bound::Unbounded,
                    ))
                    .map(|(price, _)| *price)
                    .collect::<Vec<_>>();
                for price in stale_prices {
                    let previous = self.bids.remove(&price);
                    undo.push((BookUpdateSide::Bid, price, previous));
                }
            }
        }
        if let Some(best_ask) = best_ask {
            if best_ask == Decimal::ONE {
                let stale_prices = self.asks.keys().copied().collect::<Vec<_>>();
                for price in stale_prices {
                    let previous = self.asks.remove(&price);
                    undo.push((BookUpdateSide::Ask, price, previous));
                }
            } else {
                let stale_prices = self
                    .asks
                    .range((
                        std::ops::Bound::Unbounded,
                        std::ops::Bound::Excluded(best_ask),
                    ))
                    .map(|(price, _)| *price)
                    .collect::<Vec<_>>();
                for price in stale_prices {
                    let previous = self.asks.remove(&price);
                    undo.push((BookUpdateSide::Ask, price, previous));
                }
            }
        }

        self.validate();
        if self.integrity_status != FeedIntegrityStatus::Ok {
            return self.integrity_status;
        }

        let bid_matches =
            best_bid.is_none_or(|expected| self.best_bid().unwrap_or(Decimal::ZERO) == expected);
        let ask_matches =
            best_ask.is_none_or(|expected| self.best_ask().unwrap_or(Decimal::ONE) == expected);
        if !bid_matches || !ask_matches {
            self.integrity_status = FeedIntegrityStatus::TopOfBookMismatch;
        }
        self.integrity_status
    }

    fn restore_levels(&mut self, undo: Vec<(BookUpdateSide, Decimal, Option<Decimal>)>) {
        for (side, price, previous) in undo.into_iter().rev() {
            let levels = match side {
                BookUpdateSide::Bid => &mut self.bids,
                BookUpdateSide::Ask => &mut self.asks,
            };
            if let Some(size) = previous {
                levels.insert(price, size);
            } else {
                levels.remove(&price);
            }
        }
    }

    fn readiness(&self, connection_id: Uuid) -> BookReadiness {
        BookReadiness {
            market_id: self.market_id.clone(),
            token_id: self.token_id.clone(),
            connection_id,
            bootstrapped: self.bootstrapped,
            integrity_status: self.integrity_status,
            source_timestamp: self.source_timestamp,
            received_at: self.received_at,
            best_bid: self.best_bid(),
            best_ask: self.best_ask(),
        }
    }

    fn checkpoint(&self, connection_id: Uuid) -> Option<OrderbookCheckpoint> {
        Some(OrderbookCheckpoint {
            checkpoint_id: Uuid::new_v4(),
            market_id: self.market_id.clone(),
            token_id: self.token_id.clone(),
            source_timestamp: self.source_timestamp?,
            received_at: self.received_at?,
            observed_at: Utc::now(),
            connection_id,
            ingest_sequence: self.ingest_sequence,
            source_hash: self.source_hash.clone(),
            tick_size: self.tick_size,
            best_bid: self.best_bid(),
            best_ask: self.best_ask(),
            bids: self
                .bids
                .iter()
                .rev()
                .map(|(price, size)| OrderbookLevel {
                    price: *price,
                    size: *size,
                })
                .collect(),
            asks: self
                .asks
                .iter()
                .map(|(price, size)| OrderbookLevel {
                    price: *price,
                    size: *size,
                })
                .collect(),
            integrity_status: self.integrity_status,
        })
    }
}

fn market_identifiers(market: &BtcIntervalMarket) -> [&str; 2] {
    [&market.market_id, &market.condition_id]
}

fn book_market_identifiers(book: &FeedBook) -> [&str; 2] {
    [&book.market_id, &book.wire_market_id]
}

fn validate_market_identity(market: &BtcIntervalMarket) -> Result<()> {
    if market.up_token_id == market.down_token_id {
        bail!(
            "orderbook market {} assigns both outcomes to token {}",
            market.market_id,
            market.up_token_id
        );
    }
    Ok(())
}

fn validate_desired_markets(markets: &[BtcIntervalMarket]) -> Result<()> {
    for (index, market) in markets.iter().enumerate() {
        validate_market_identity(market)?;
        for other in markets.iter().skip(index + 1) {
            validate_market_identity(other)?;
            let same_identity = market.market_id == other.market_id
                && market.condition_id == other.condition_id
                && market.up_token_id == other.up_token_id
                && market.down_token_id == other.down_token_id;
            if same_identity {
                continue;
            }
            let token_collision = [&market.up_token_id, &market.down_token_id]
                .into_iter()
                .any(|token_id| token_id == &other.up_token_id || token_id == &other.down_token_id);
            if token_collision {
                bail!(
                    "orderbook desired markets {} and {} reuse a token identity",
                    market.market_id,
                    other.market_id
                );
            }
            let identifier_collision = market_identifiers(market)
                .into_iter()
                .any(|identifier| market_identifiers(other).contains(&identifier));
            if identifier_collision {
                bail!(
                    "orderbook desired markets {} and {} reuse a market identity",
                    market.market_id,
                    other.market_id
                );
            }
        }
    }
    Ok(())
}

fn validate_market_against_books<'a>(
    market: &BtcIntervalMarket,
    books: impl Iterator<Item = &'a FeedBook>,
) -> Result<()> {
    validate_market_identity(market)?;
    for book in books {
        let expected_outcome = if book.token_id == market.up_token_id {
            Some(BtcOutcome::Up)
        } else if book.token_id == market.down_token_id {
            Some(BtcOutcome::Down)
        } else {
            None
        };
        let token_claimed = expected_outcome.is_some();
        let identifier_claimed = book_market_identifiers(book)
            .into_iter()
            .any(|existing| market_identifiers(market).contains(&existing));
        let same_market =
            book.market_id == market.market_id && book.wire_market_id == market.condition_id;
        let expected_token = same_market && expected_outcome == Some(book.outcome);
        if token_claimed && !same_market {
            bail!(
                "orderbook token {} is already owned by market {}",
                book.token_id,
                book.market_id
            );
        }
        if identifier_claimed && !expected_token {
            bail!(
                "orderbook market identity {} conflicts with registered token {}",
                market.market_id,
                book.token_id
            );
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct BookRegistry {
    connection_id: Uuid,
    books: HashMap<String, FeedBook>,
    next_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BookIdentityDiagnostic {
    pub reason: &'static str,
    pub token_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookApplyResult {
    pub(crate) token_id: Option<String>,
    pub(crate) source_timestamp: DateTime<Utc>,
    pub(crate) applied: bool,
    pub(crate) book_mutated: bool,
    pub(crate) integrity_status: FeedIntegrityStatus,
}

impl BookRegistry {
    pub fn new(connection_id: Uuid) -> Self {
        Self {
            connection_id,
            books: HashMap::new(),
            next_sequence: 1,
        }
    }

    pub fn connection_id(&self) -> Uuid {
        self.connection_id
    }

    /// Publishes only the books touched by one frame while preserving the rest of the active
    /// connection epoch. Subscription changes and connection promotion still replace the complete
    /// published registry at their explicit publication boundaries.
    pub(crate) fn publish_frame_books_from<'a>(
        &mut self,
        source: &BookRegistry,
        token_ids: impl IntoIterator<Item = &'a str>,
    ) {
        if self.connection_id != source.connection_id {
            *self = source.clone();
            return;
        }
        self.next_sequence = source.next_sequence;
        for token_id in token_ids {
            if let Some(book) = source.books.get(token_id) {
                self.books.insert(token_id.to_string(), book.clone());
            }
        }
    }

    pub fn register_market(&mut self, market: &BtcIntervalMarket) {
        self.try_register_market(market)
            .unwrap_or_else(|error| panic!("invalid orderbook market registration: {error}"));
    }

    /// Adds a market without replacing an existing live book. Market and token identifiers are
    /// immutable registry ownership boundaries; conflicting reuse fails before any state changes.
    pub fn try_register_market(&mut self, market: &BtcIntervalMarket) -> Result<bool> {
        self.validate_market_registration(market)?;
        let mut added = false;
        for (token_id, outcome) in [
            (&market.up_token_id, BtcOutcome::Up),
            (&market.down_token_id, BtcOutcome::Down),
        ] {
            if self.books.contains_key(token_id) {
                continue;
            }
            self.books.insert(
                token_id.clone(),
                FeedBook::new(
                    market.market_id.clone(),
                    market.condition_id.clone(),
                    token_id.clone(),
                    outcome,
                    market.tick_size,
                ),
            );
            added = true;
        }
        Ok(added)
    }

    /// Removes books outside the desired market set while preserving every retained book and the
    /// connection epoch. The desired identity set is validated before the registry is mutated.
    pub fn retain_markets(&mut self, desired: &[BtcIntervalMarket]) -> Result<usize> {
        self.validate_market_set(desired)?;
        let desired_tokens = desired
            .iter()
            .flat_map(|market| [&market.up_token_id, &market.down_token_id])
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let previous_len = self.books.len();
        self.books
            .retain(|token_id, _| desired_tokens.contains(token_id.as_str()));
        Ok(previous_len.saturating_sub(self.books.len()))
    }

    /// Validates a complete desired registry membership without mutating book state. Existing
    /// books are checked only when their token remains desired; retired books may therefore be
    /// removed before a replacement identity is registered.
    pub fn validate_market_set(&self, desired: &[BtcIntervalMarket]) -> Result<()> {
        validate_desired_markets(desired)?;
        let retained_tokens = desired
            .iter()
            .flat_map(|market| [&market.up_token_id, &market.down_token_id])
            .map(String::as_str)
            .collect::<HashSet<_>>();
        for market in desired {
            validate_market_against_books(
                market,
                self.books
                    .values()
                    .filter(|book| retained_tokens.contains(book.token_id.as_str())),
            )?;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.books.len()
    }

    pub fn is_empty(&self) -> bool {
        self.books.is_empty()
    }

    fn validate_market_registration(&self, market: &BtcIntervalMarket) -> Result<()> {
        validate_market_against_books(market, self.books.values())
    }

    /// A reconnect starts a new integrity epoch. Existing levels must never receive deltas from the
    /// new socket before its initial full snapshots arrive.
    pub fn reset_connection(&mut self, connection_id: Uuid) {
        self.connection_id = connection_id;
        self.next_sequence = 1;
        for book in self.books.values_mut() {
            reset_book_for_snapshot(book);
        }
    }

    /// Fail every bootstrapped book closed after a message-level CLOB decode failure. A later
    /// delta cannot clear this status; only a complete venue snapshot can restore readiness.
    pub fn quarantine(&mut self, status: FeedIntegrityStatus) {
        for book in self.books.values_mut().filter(|book| book.bootstrapped) {
            book.integrity_status = status;
        }
    }

    pub fn apply(
        &mut self,
        message: ClobMessage,
        received_at: DateTime<Utc>,
    ) -> Vec<BookApplyResult> {
        match message {
            ClobMessage::Book {
                market_id,
                token_id,
                bids,
                asks,
                source_timestamp,
                source_hash,
            } => {
                let sequence = self.take_sequence();
                let mut book_mutated = false;
                let status = if let Some(book) = self.books.get_mut(&token_id) {
                    if !book.matches_market(&market_id) {
                        FeedIntegrityStatus::MarketMismatch
                    } else {
                        book_mutated = true;
                        book.bids = levels_to_map(bids);
                        book.asks = levels_to_map(asks);
                        book.bootstrapped = true;
                        book.source_timestamp = Some(source_timestamp);
                        book.received_at = Some(received_at);
                        book.source_hash = source_hash.clone();
                        book.ingest_sequence = sequence;
                        book.validate();
                        book.integrity_status
                    }
                } else {
                    FeedIntegrityStatus::UnknownToken
                };
                vec![book_apply_result(
                    Some(token_id),
                    source_timestamp,
                    status == FeedIntegrityStatus::Ok,
                    book_mutated,
                    status,
                )]
            }
            ClobMessage::PriceChange {
                market_id,
                changes,
                source_timestamp,
            } => {
                let mut entries = Vec::with_capacity(changes.len());
                for change in changes {
                    let sequence = self.take_sequence();
                    entries.push((change, sequence));
                }

                let mut by_token: HashMap<String, Vec<usize>> = HashMap::new();
                for (index, (change, _)) in entries.iter().enumerate() {
                    by_token
                        .entry(change.token_id.clone())
                        .or_default()
                        .push(index);
                }
                let mut outcomes =
                    vec![(false, false, FeedIntegrityStatus::UnknownToken); entries.len()];

                for (token_id, indexes) in by_token {
                    let mut book_mutated = false;
                    let status = if let Some(book) = self.books.get_mut(&token_id) {
                        if !book.matches_market(&market_id) {
                            FeedIntegrityStatus::MarketMismatch
                        } else if !book.bootstrapped {
                            FeedIntegrityStatus::PreSnapshot
                        } else if book.integrity_status != FeedIntegrityStatus::Ok {
                            book.integrity_status
                        } else if book
                            .source_timestamp
                            .is_some_and(|last| source_timestamp < last)
                        {
                            FeedIntegrityStatus::OutOfOrder
                        } else if indexes.iter().any(|index| {
                            let change = &entries[*index].0;
                            change.price <= Decimal::ZERO
                                || change.price >= Decimal::ONE
                                || change.size < Decimal::ZERO
                        }) {
                            FeedIntegrityStatus::DecodeError
                        } else {
                            let mut undo = Vec::with_capacity(indexes.len());
                            for index in &indexes {
                                let change = &entries[*index].0;
                                let levels = match change.side {
                                    BookUpdateSide::Bid => &mut book.bids,
                                    BookUpdateSide::Ask => &mut book.asks,
                                };
                                let previous = levels.get(&change.price).copied();
                                undo.push((change.side, change.price, previous));
                                if change.size == Decimal::ZERO {
                                    levels.remove(&change.price);
                                } else {
                                    levels.insert(change.price, change.size);
                                }
                            }

                            let last_index = *indexes
                                .last()
                                .expect("a grouped price-change token must have an entry");
                            let best_bid = indexes
                                .iter()
                                .rev()
                                .find_map(|index| entries[*index].0.best_bid);
                            let best_ask = indexes
                                .iter()
                                .rev()
                                .find_map(|index| entries[*index].0.best_ask);
                            let status =
                                book.reconcile_advertised_top(best_bid, best_ask, &mut undo);
                            if status == FeedIntegrityStatus::Ok {
                                book_mutated = true;
                                book.source_timestamp = Some(source_timestamp);
                                book.received_at = Some(received_at);
                                book.source_hash = indexes
                                    .iter()
                                    .rev()
                                    .find_map(|index| entries[*index].0.source_hash.clone());
                                book.ingest_sequence = entries[last_index].1;
                            } else {
                                book.restore_levels(undo);
                            }
                            status
                        }
                    } else {
                        FeedIntegrityStatus::UnknownToken
                    };
                    if matches!(
                        status,
                        FeedIntegrityStatus::CrossedBook
                            | FeedIntegrityStatus::TopOfBookMismatch
                            | FeedIntegrityStatus::DecodeError
                    ) {
                        if let Some(current) = self.books.get_mut(&token_id) {
                            book_mutated = current.integrity_status != status;
                            current.integrity_status = status;
                        }
                    }
                    for index in indexes {
                        outcomes[index] = (status == FeedIntegrityStatus::Ok, book_mutated, status);
                    }
                }

                entries
                    .into_iter()
                    .zip(outcomes)
                    .map(|((change, _sequence), (applied, book_mutated, status))| {
                        book_apply_result(
                            Some(change.token_id),
                            source_timestamp,
                            applied,
                            book_mutated,
                            status,
                        )
                    })
                    .collect()
            }
            ClobMessage::BestBidAsk {
                market_id,
                token_id,
                best_bid: _,
                best_ask: _,
                source_timestamp,
            } => vec![self.non_mutating_result(market_id, token_id, source_timestamp)],
            ClobMessage::TickSizeChange {
                market_id,
                token_id,
                old_tick_size: _,
                new_tick_size,
                source_timestamp,
            } => {
                let sequence = self.take_sequence();
                let mut applied = false;
                let mut book_mutated = false;
                let status = if let Some(book) = self.books.get_mut(&token_id) {
                    if !book.matches_market(&market_id) {
                        FeedIntegrityStatus::MarketMismatch
                    } else if new_tick_size <= Decimal::ZERO {
                        FeedIntegrityStatus::DecodeError
                    } else if !book.bootstrapped {
                        book_mutated = true;
                        book.tick_size = new_tick_size;
                        book.ingest_sequence = sequence;
                        FeedIntegrityStatus::PreSnapshot
                    } else {
                        book_mutated = true;
                        book.tick_size = new_tick_size;
                        book.ingest_sequence = sequence;
                        applied = true;
                        book.integrity_status
                    }
                } else {
                    FeedIntegrityStatus::UnknownToken
                };
                vec![book_apply_result(
                    Some(token_id),
                    source_timestamp,
                    applied,
                    book_mutated,
                    status,
                )]
            }
            ClobMessage::LastTradePrice {
                market_id,
                token_id,
                price: _,
                size: _,
                source_timestamp,
            } => vec![self.non_mutating_result(market_id, token_id, source_timestamp)],
            ClobMessage::MarketResolved {
                market_id,
                winning_token_id,
                winning_outcome: _,
                source_timestamp,
            } => vec![self.non_mutating_result(market_id, winning_token_id, source_timestamp)],
        }
    }

    pub fn checkpoint(&self, token_id: &str) -> Option<OrderbookCheckpoint> {
        self.books.get(token_id)?.checkpoint(self.connection_id)
    }

    pub fn market_books_bootstrapped(&self, market: &BtcIntervalMarket) -> bool {
        [
            (&market.up_token_id, BtcOutcome::Up),
            (&market.down_token_id, BtcOutcome::Down),
        ]
        .into_iter()
        .all(|(token_id, outcome)| {
            self.books.get(token_id).is_some_and(|book| {
                book.market_id == market.market_id
                    && book.wire_market_id == market.condition_id
                    && book.token_id == *token_id
                    && book.outcome == outcome
                    && book.bootstrapped
                    && book.integrity_status == FeedIntegrityStatus::Ok
                    && book.source_timestamp.is_some()
                    && book.received_at.is_some()
            })
        })
    }

    pub fn market_books_structurally_ready(&self, market: &BtcIntervalMarket) -> bool {
        [
            (&market.up_token_id, BtcOutcome::Up),
            (&market.down_token_id, BtcOutcome::Down),
        ]
        .into_iter()
        .all(|(token_id, outcome)| {
            self.books.get(token_id).is_some_and(|book| {
                book.market_id == market.market_id
                    && book.wire_market_id == market.condition_id
                    && book.token_id == *token_id
                    && book.outcome == outcome
                    && book.bootstrapped
                    && book.integrity_status == FeedIntegrityStatus::Ok
                    && book.source_timestamp.is_some()
                    && book.received_at.is_some()
            })
        })
    }

    pub fn market_books_ready(
        &self,
        market: &BtcIntervalMarket,
        now: DateTime<Utc>,
        max_age: Duration,
    ) -> bool {
        [
            (&market.up_token_id, BtcOutcome::Up),
            (&market.down_token_id, BtcOutcome::Down),
        ]
        .into_iter()
        .all(|(token_id, outcome)| {
            self.books.get(token_id).is_some_and(|book| {
                book.market_id == market.market_id
                    && book.wire_market_id == market.condition_id
                    && book.token_id == *token_id
                    && book.outcome == outcome
                    && book.bootstrapped
                    && book.integrity_status == FeedIntegrityStatus::Ok
                    && book
                        .source_timestamp
                        .is_some_and(|timestamp| timestamp - now <= max_age)
                    && book.received_at.is_some_and(|timestamp| timestamp <= now)
                    && book.source_timestamp.zip(book.received_at).is_some_and(
                        |(source_timestamp, received_at)| received_at - source_timestamp <= max_age,
                    )
            })
        })
    }

    pub(crate) fn market_book_identity_diagnostic(
        &self,
        market: &BtcIntervalMarket,
    ) -> Option<BookIdentityDiagnostic> {
        for (token_id, outcome) in [
            (&market.up_token_id, BtcOutcome::Up),
            (&market.down_token_id, BtcOutcome::Down),
        ] {
            let Some(book) = self.books.get(token_id) else {
                continue;
            };
            let reason = if book.market_id != market.market_id {
                Some("canonical_market_identity_mismatch")
            } else if book.wire_market_id != market.condition_id {
                Some("wire_market_identity_mismatch")
            } else if book.token_id != *token_id {
                Some("token_identity_mismatch")
            } else if book.outcome != outcome {
                Some("outcome_identity_mismatch")
            } else {
                None
            };
            if let Some(reason) = reason {
                return Some(BookIdentityDiagnostic {
                    reason,
                    token_id: token_id.clone(),
                });
            }
        }
        None
    }

    pub fn book_readiness(&self) -> Vec<BookReadiness> {
        let mut books: Vec<_> = self
            .books
            .values()
            .map(|book| book.readiness(self.connection_id))
            .collect();
        books.sort_by(|left, right| left.token_id.cmp(&right.token_id));
        books
    }

    fn non_mutating_result(
        &mut self,
        market_id: String,
        token_id: String,
        source_timestamp: DateTime<Utc>,
    ) -> BookApplyResult {
        self.take_sequence();
        let (applied, status) = match self.books.get(&token_id) {
            Some(book) if !book.matches_market(&market_id) => {
                (false, FeedIntegrityStatus::MarketMismatch)
            }
            Some(book) if book.bootstrapped => (true, book.integrity_status),
            Some(_) => (false, FeedIntegrityStatus::PreSnapshot),
            None => (false, FeedIntegrityStatus::UnknownToken),
        };
        book_apply_result(Some(token_id), source_timestamp, applied, false, status)
    }

    fn take_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        sequence
    }
}

fn reset_book_for_snapshot(book: &mut FeedBook) {
    book.bids.clear();
    book.asks.clear();
    book.bootstrapped = false;
    book.integrity_status = FeedIntegrityStatus::PreSnapshot;
    book.source_timestamp = None;
    book.received_at = None;
    book.source_hash = None;
    book.ingest_sequence = 0;
}

impl RealtimeState {
    pub fn set_market(&mut self, market: BtcIntervalMarket) {
        self.set_current_market(Some(market));
    }

    pub fn set_current_market(&mut self, market: Option<BtcIntervalMarket>) {
        let changed = self.current_market.as_ref().map(|value| &value.market_id)
            != market.as_ref().map(|value| &value.market_id);
        self.current_market = market;
        if changed {
            self.resolved_outcome = None;
        }
    }

    pub fn update_books(&mut self, registry: &BookRegistry) {
        self.books = registry
            .book_readiness()
            .into_iter()
            .map(|book| (book.token_id.clone(), book))
            .collect();
    }

    pub fn update_reference_price(&mut self, tick: ReferencePriceTick) -> bool {
        let replace = self
            .reference_prices
            .get(&tick.source)
            .map(|current| tick.source_timestamp >= current.source_timestamp)
            .unwrap_or(true);
        if replace {
            self.last_updated_at = Some(tick.received_at);
            self.reference_prices.insert(tick.source, tick);
        }
        replace
    }

    pub fn apply_resolution(&mut self, winning_token_id: &str) -> bool {
        let Some(market_id) = self
            .current_market
            .as_ref()
            .map(|market| market.market_id.clone())
        else {
            return false;
        };
        self.apply_market_resolution(&market_id, winning_token_id)
    }

    pub fn apply_market_resolution(&mut self, market_id: &str, winning_token_id: &str) -> bool {
        let Some(market) = self.current_market.as_ref() else {
            return false;
        };
        if market_id != market.market_id && market_id != market.condition_id {
            return false;
        }
        let outcome = if winning_token_id == market.up_token_id {
            BtcOutcome::Up
        } else if winning_token_id == market.down_token_id {
            BtcOutcome::Down
        } else {
            return false;
        };
        self.resolved_outcome = Some(outcome);
        true
    }

    pub fn readiness(
        &self,
        now: DateTime<Utc>,
        max_book_age: Duration,
        max_reference_age: Duration,
    ) -> Readiness {
        let mut reasons = Vec::new();
        if !self.primary_persistence_available() {
            reasons.push("primary_persistence_unavailable".to_string());
        }
        let Some(market) = self.current_market.as_ref() else {
            reasons.push("missing_current_market".to_string());
            return Readiness {
                ready: false,
                checked_at: now,
                market_slug: None,
                reasons,
                books: self.books.values().cloned().collect(),
                sources: Vec::new(),
            };
        };
        if !market.is_trade_window(now) {
            reasons.push("market_not_in_trade_window".to_string());
        }
        for token_id in [&market.up_token_id, &market.down_token_id] {
            match self.books.get(token_id) {
                None => reasons.push(format!("missing_book:{token_id}")),
                Some(book) if !book.bootstrapped => {
                    reasons.push(format!("book_not_bootstrapped:{token_id}"))
                }
                Some(book) if book.integrity_status != FeedIntegrityStatus::Ok => {
                    reasons.push(format!("book_integrity:{token_id}"))
                }
                Some(book)
                    if book
                        .source_timestamp
                        .is_some_and(|timestamp| timestamp - now > max_book_age) =>
                {
                    reasons.push(format!("future_book_timestamp:{token_id}"))
                }
                Some(book) if book.received_at.is_none_or(|timestamp| timestamp > now) => {
                    reasons.push(format!("future_book_timestamp:{token_id}"))
                }
                Some(_) => {}
            }
        }
        for source in [
            ReferencePriceSource::DirectBinance,
            ReferencePriceSource::RtdsChainlink,
        ] {
            match self.reference_prices.get(&source) {
                None => reasons.push(format!("missing_reference:{}", source.as_str())),
                Some(tick) if tick.source_timestamp - now > max_reference_age => {
                    reasons.push(format!("future_reference:{}", source.as_str()))
                }
                Some(tick) if now - tick.received_at > max_reference_age => {
                    reasons.push(format!("stale_reference:{}", source.as_str()))
                }
                Some(_) => {}
            }
        }
        let sources = self
            .reference_prices
            .values()
            .map(|tick| SourceReadiness {
                source: tick.source,
                source_timestamp: tick.source_timestamp,
                received_at: tick.received_at,
                price: tick.price,
            })
            .collect();
        Readiness {
            ready: reasons.is_empty(),
            checked_at: now,
            market_slug: Some(market.event_slug.clone()),
            reasons,
            books: self.books.values().cloned().collect(),
            sources,
        }
    }
}

fn book_apply_result(
    token_id: Option<String>,
    source_timestamp: DateTime<Utc>,
    applied: bool,
    book_mutated: bool,
    integrity_status: FeedIntegrityStatus,
) -> BookApplyResult {
    BookApplyResult {
        token_id,
        source_timestamp,
        applied,
        book_mutated,
        integrity_status,
    }
}

fn levels_to_map(levels: Vec<OrderbookLevel>) -> BTreeMap<Decimal, Decimal> {
    levels
        .into_iter()
        .filter(|level| level.price > Decimal::ZERO && level.size > Decimal::ZERO)
        .map(|level| (level.price, level.size))
        .collect()
}

fn parse_levels(value: Option<&Value>, field: &str) -> Result<Vec<OrderbookLevel>> {
    let values = value
        .and_then(Value::as_array)
        .with_context(|| format!("CLOB book is missing {field}"))?;
    values
        .iter()
        .map(|value| {
            let object = value
                .as_object()
                .with_context(|| format!("CLOB {field} level must be an object"))?;
            let price = required_decimal(object, &["price"])?;
            let size = required_decimal(object, &["size"])?;
            if price <= Decimal::ZERO || price >= Decimal::ONE || size < Decimal::ZERO {
                bail!("CLOB book level has invalid price or size");
            }
            Ok(OrderbookLevel { price, size })
        })
        .collect()
}

fn required_string(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Result<String> {
    string_field(object, keys).with_context(|| format!("missing required field {}", keys[0]))
}

fn string_field(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| match object.get(*key)? {
        Value::String(value) => {
            let value = value.trim();
            (!value.is_empty()).then(|| value.to_string())
        }
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    })
}

fn decimal_field(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<Decimal> {
    keys.iter().find_map(|key| match object.get(*key)? {
        Value::String(value) => Decimal::from_str(value).ok(),
        Value::Number(value) => Decimal::from_str(&value.to_string()).ok(),
        _ => None,
    })
}

fn optional_decimal_field(
    object: &serde_json::Map<String, Value>,
    keys: &[&str],
) -> Result<Option<Decimal>> {
    for key in keys {
        let Some(value) = object.get(*key) else {
            continue;
        };
        if value.is_null() {
            return Ok(None);
        }
        let parsed = match value {
            Value::String(value) => Decimal::from_str(value).ok(),
            Value::Number(value) => Decimal::from_str(&value.to_string()).ok(),
            _ => None,
        }
        .with_context(|| format!("invalid decimal field {key}"))?;
        return Ok(Some(parsed));
    }
    Ok(None)
}

fn required_decimal(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Result<Decimal> {
    decimal_field(object, keys).with_context(|| format!("invalid decimal field {}", keys[0]))
}

fn required_u64(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Result<u64> {
    let value = required_string(object, keys)?;
    value
        .parse::<u64>()
        .with_context(|| format!("invalid unsigned integer field {}", keys[0]))
}

fn timestamp_field(
    object: &serde_json::Map<String, Value>,
    keys: &[&str],
) -> Result<DateTime<Utc>> {
    let raw = keys
        .iter()
        .find_map(|key| object.get(*key))
        .with_context(|| format!("missing timestamp field {}", keys[0]))?;
    let millis = match raw {
        Value::String(value) => value.parse::<i64>().ok(),
        Value::Number(value) => value.as_i64(),
        _ => None,
    }
    .with_context(|| format!("invalid timestamp field {}", keys[0]))?;
    DateTime::from_timestamp_millis(millis)
        .with_context(|| format!("timestamp field {} is out of range", keys[0]))
}

fn ensure_positive_price(price: Decimal) -> Result<()> {
    if price <= Decimal::ZERO {
        bail!("reference price must be positive");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use rust_decimal_macros::dec;

    use super::*;

    fn ts(millis: i64) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(millis).unwrap()
    }

    fn market() -> BtcIntervalMarket {
        BtcIntervalMarket {
            event_id: "event".to_string(),
            event_slug: "btc-updown-5m-1783902600".to_string(),
            series_slug: "btc-up-or-down-5m".to_string(),
            market_id: "market".to_string(),
            condition_id: "condition".to_string(),
            window_start: Utc.timestamp_opt(1_783_902_600, 0).unwrap(),
            window_end: Utc.timestamp_opt(1_783_902_900, 0).unwrap(),
            up_token_id: "up".to_string(),
            down_token_id: "down".to_string(),
            tick_size: dec!(0.01),
            minimum_order_size: Some(dec!(5)),
            resolution_source: "https://data.chain.link/streams/btc-usd".to_string(),
            active: true,
            closed: false,
            accepting_orders: true,
            fees_enabled: true,
            fee_schedule: serde_json::json!({}),
            raw_payload: serde_json::json!({}),
        }
    }

    fn numbered_market(index: u32) -> BtcIntervalMarket {
        let mut market = market();
        market.event_id = format!("event-{index}");
        market.event_slug = format!("btc-updown-5m-{index}");
        market.market_id = format!("market-{index}");
        market.condition_id = format!("condition-{index}");
        market.up_token_id = format!("up-{index}");
        market.down_token_id = format!("down-{index}");
        market
    }

    fn state_reference_tick(
        price: Decimal,
        source_timestamp: DateTime<Utc>,
        received_at: DateTime<Utc>,
        ingest_sequence: u64,
    ) -> ReferencePriceTick {
        reference_tick(
            ReferencePriceSource::DirectBinance,
            "BTCUSDT",
            price,
            source_timestamp,
            None,
            received_at,
            Uuid::nil(),
            ingest_sequence,
            None,
            serde_json::json!({}),
        )
        .unwrap()
    }

    #[test]
    fn parses_full_book_and_batched_price_changes() {
        let book = serde_json::json!({
            "event_type": "book",
            "market": "market",
            "asset_id": "up",
            "timestamp": "1783902700000",
            "hash": "h1",
            "bids": [{"price": ".48", "size": "30"}],
            "asks": [{"price": ".52", "size": "25"}]
        });
        let parsed = parse_clob_messages(&book).unwrap();
        assert!(matches!(&parsed[0], ClobMessage::Book { token_id, .. } if token_id == "up"));

        let changes = serde_json::json!({
            "event_type": "price_change",
            "market": "market",
            "timestamp": 1783902701000_i64,
            "price_changes": [{
                "asset_id": "up", "price": ".49", "size": "20", "side": "BUY", "hash": "h2",
                "best_bid": ".49", "best_ask": ".52"
            }, {
                "asset_id": "down", "price": ".53", "size": "0", "side": "SELL", "hash": "h3",
                "best_bid": ".47", "best_ask": ".54"
            }]
        });
        let parsed = parse_clob_messages(&changes).unwrap();
        assert!(
            matches!(&parsed[0], ClobMessage::PriceChange { changes, .. }
            if changes.len() == 2
                && changes[0].side == BookUpdateSide::Bid
                && changes[1].size == Decimal::ZERO
                && changes[0].best_bid == Some(dec!(0.49))
                && changes[1].best_ask == Some(dec!(0.54)))
        );
    }

    #[test]
    fn rejects_malformed_advertised_tops_and_out_of_range_snapshot_levels() {
        let malformed = serde_json::json!({
            "event_type": "price_change", "market": "market",
            "timestamp": 1783902701000_i64,
            "price_changes": [{
                "asset_id": "up", "price": ".49", "size": "20", "side": "BUY",
                "best_bid": "not-a-price", "best_ask": ".52"
            }]
        });
        assert!(parse_clob_messages(&malformed).is_err());

        let invalid_snapshot = serde_json::json!({
            "event_type": "book", "market": "market", "asset_id": "up",
            "timestamp": 1783902701000_i64,
            "bids": [{"price": "1", "size": "10"}], "asks": []
        });
        assert!(parse_clob_messages(&invalid_snapshot).is_err());
    }

    #[test]
    fn accepts_initial_message_arrays() {
        let value = serde_json::json!([{
            "event_type": "book", "market": "market", "asset_id": "up",
            "timestamp": "1783902700000", "bids": [], "asks": []
        }, {
            "event_type": "book", "market": "market", "asset_id": "down",
            "timestamp": "1783902700001", "bids": [], "asks": []
        }]);
        assert_eq!(parse_clob_messages(&value).unwrap().len(), 2);
    }

    #[test]
    fn ignores_new_market_control_messages() {
        let value = serde_json::json!({"event_type": "new_market"});
        assert!(parse_clob_messages(&value).unwrap().is_empty());
    }

    #[test]
    fn parses_auxiliary_clob_events() {
        let messages = [
            serde_json::json!({
                "event_type": "best_bid_ask", "market": "market", "asset_id": "up",
                "best_bid": ".4", "best_ask": ".6", "timestamp": "1783902700000"
            }),
            serde_json::json!({
                "event_type": "tick_size_change", "market": "market", "asset_id": "up",
                "old_tick_size": ".01", "new_tick_size": ".001", "timestamp": "1783902700000"
            }),
            serde_json::json!({
                "event_type": "market_resolved", "market": "market",
                "winning_asset_id": "up", "winning_outcome": "Up",
                "timestamp": "1783902700000"
            }),
        ];
        assert!(matches!(
            parse_clob_messages(&messages[0]).unwrap()[0],
            ClobMessage::BestBidAsk { .. }
        ));
        assert!(matches!(
            parse_clob_messages(&messages[1]).unwrap()[0],
            ClobMessage::TickSizeChange { .. }
        ));
        assert!(matches!(
            parse_clob_messages(&messages[2]).unwrap()[0],
            ClobMessage::MarketResolved { .. }
        ));
    }

    #[test]
    fn parses_rtds_sources_with_distinct_identity_and_timestamps() {
        let connection = Uuid::new_v4();
        let received = ts(1_783_902_701_250);
        let chainlink = parse_rtds_reference_tick(
            &serde_json::json!({
                "topic": "crypto_prices_chainlink",
                "type": "update",
                "timestamp": 1783902701200_i64,
                "payload": {
                    "symbol": "btc/usd", "timestamp": 1783902701100_i64, "value": 67234.50
                }
            }),
            connection,
            7,
            received,
        )
        .unwrap();
        assert_eq!(chainlink.source, ReferencePriceSource::RtdsChainlink);
        assert_eq!(chainlink.price, dec!(67234.50));
        assert_eq!(chainlink.source_timestamp, ts(1_783_902_701_100));
        assert_eq!(chainlink.received_at, received);

        let binance = parse_rtds_reference_tick(
            &serde_json::json!({
                "topic": "crypto_prices", "type": "update", "timestamp": 1783902701200_i64,
                "payload": {"symbol": "btcusdt", "timestamp": 1783902701150_i64, "value": "67235.1"}
            }),
            connection,
            8,
            received,
        )
        .unwrap();
        assert_eq!(binance.source, ReferencePriceSource::RtdsBinance);
    }

    #[test]
    fn parses_chainlink_twap_60_from_exact_e18_value() {
        let received = ts(1_783_902_701_250);
        let point = parse_rtds_chainlink_twap_60(
            &serde_json::json!({
                "topic": "crypto_prices_twap_sixty",
                "type": "update",
                "timestamp": 1783902701200_i64,
                "payload": {
                    "symbol": "btc/usd",
                    "value": 67234.5,
                    "full_accuracy_value": "67234501234567890123456",
                    "timestamp": 1783902701100_i64,
                    "window_s": 60
                }
            }),
            received,
        )
        .unwrap();

        assert_eq!(point.price, dec!(67234.501234567890123456));
        assert_eq!(point.source_timestamp, ts(1_783_902_701_100));
        assert_eq!(point.available_at, received);
    }

    #[test]
    fn rejects_a_non_sixty_second_twap_payload() {
        let result = parse_rtds_chainlink_twap_60(
            &serde_json::json!({
                "topic": "crypto_prices_twap_sixty",
                "type": "update",
                "payload": {
                    "symbol": "btc/usd",
                    "full_accuracy_value": "67234500000000000000000",
                    "timestamp": 1783902701100_i64,
                    "window_s": 30
                }
            }),
            ts(1_783_902_701_250),
        );

        assert!(result.is_err());
    }

    #[test]
    fn canonicalizes_live_reference_tick_before_identity_and_downstream_use() {
        let raw_payload = serde_json::json!({
            "payload": {"value": "62251.646396591175"}
        });
        let source_timestamp = Utc
            .timestamp_opt(1_783_902_701, 123_456_789)
            .single()
            .unwrap();
        let envelope_timestamp = Utc
            .timestamp_opt(1_783_902_701, 223_456_789)
            .single()
            .unwrap();
        let received_at = Utc
            .timestamp_opt(1_783_902_701, 323_456_789)
            .single()
            .unwrap();

        let tick = reference_tick(
            ReferencePriceSource::RtdsChainlink,
            "BTCUSD",
            dec!(62251.646396591175),
            source_timestamp,
            Some(envelope_timestamp),
            received_at,
            Uuid::new_v4(),
            41,
            None,
            raw_payload.clone(),
        )
        .unwrap();

        assert_eq!(tick.price, dec!(62251.6463965912));
        assert_eq!(tick.source_timestamp.timestamp_subsec_nanos(), 123_456_000);
        assert_eq!(
            tick.envelope_timestamp.unwrap().timestamp_subsec_nanos(),
            223_456_000
        );
        assert_eq!(tick.received_at.timestamp_subsec_nanos(), 323_456_000);
        assert_eq!(
            tick.dedup_key,
            "rtds_chainlink:BTCUSD:1783902701123:-:62251.6463965912"
        );
        assert_eq!(tick.raw_payload, raw_payload);

        // Delivery metadata is lineage, not source-event identity. A retransmission therefore
        // retains the same durable identity even when it arrives on another connection later.
        let replay = reference_tick(
            ReferencePriceSource::RtdsChainlink,
            "BTCUSD",
            dec!(62251.646396591175),
            source_timestamp,
            Some(envelope_timestamp),
            received_at + Duration::milliseconds(1),
            Uuid::new_v4(),
            1,
            None,
            serde_json::json!({}),
        )
        .unwrap();
        assert_eq!(tick.dedup_key, replay.dedup_key);
        assert_eq!(tick.tick_id, replay.tick_id);
        assert_ne!(tick.received_at, replay.received_at);
        assert_ne!(tick.ingest_sequence, replay.ingest_sequence);
    }

    #[test]
    fn advancing_reference_tick_replaces_authoritative_source_state() {
        let mut state = RealtimeState::default();
        let first_source_at = ts(1_783_902_701_100);
        let first_received_at = ts(1_783_902_701_200);
        assert!(state.update_reference_price(state_reference_tick(
            dec!(67234.5),
            first_source_at,
            first_received_at,
            1,
        )));

        let next_source_at = ts(1_783_902_701_300);
        let next_received_at = ts(1_783_902_701_400);
        assert!(state.update_reference_price(state_reference_tick(
            dec!(67235.1),
            next_source_at,
            next_received_at,
            2,
        )));

        let authoritative = state
            .reference_prices
            .get(&ReferencePriceSource::DirectBinance)
            .unwrap();
        assert_eq!(authoritative.source_timestamp, next_source_at);
        assert_eq!(authoritative.price, dec!(67235.1));
        assert_eq!(state.last_updated_at, Some(next_received_at));
    }

    #[test]
    fn equal_timestamp_reference_tick_preserves_existing_replacement_policy() {
        let mut state = RealtimeState::default();
        let source_at = ts(1_783_902_701_100);
        assert!(state.update_reference_price(state_reference_tick(
            dec!(67234.5),
            source_at,
            ts(1_783_902_701_200),
            1,
        )));

        let replacement_received_at = ts(1_783_902_701_300);
        assert!(state.update_reference_price(state_reference_tick(
            dec!(67235.1),
            source_at,
            replacement_received_at,
            2,
        )));

        let authoritative = state
            .reference_prices
            .get(&ReferencePriceSource::DirectBinance)
            .unwrap();
        assert_eq!(authoritative.source_timestamp, source_at);
        assert_eq!(authoritative.price, dec!(67235.1));
        assert_eq!(state.last_updated_at, Some(replacement_received_at));
    }

    #[test]
    fn stale_reference_tick_does_not_replace_authoritative_source_state() {
        let mut state = RealtimeState::default();
        let authoritative_source_at = ts(1_783_902_701_300);
        let authoritative_received_at = ts(1_783_902_701_400);
        assert!(state.update_reference_price(state_reference_tick(
            dec!(67235.1),
            authoritative_source_at,
            authoritative_received_at,
            2,
        )));

        assert!(!state.update_reference_price(state_reference_tick(
            dec!(67234.5),
            ts(1_783_902_701_100),
            ts(1_783_902_701_500),
            1,
        )));

        let authoritative = state
            .reference_prices
            .get(&ReferencePriceSource::DirectBinance)
            .unwrap();
        assert_eq!(authoritative.source_timestamp, authoritative_source_at);
        assert_eq!(authoritative.price, dec!(67235.1));
        assert_eq!(state.last_updated_at, Some(authoritative_received_at));
    }

    #[test]
    fn reference_price_rounding_matches_postgres_numeric_midpoints() {
        let tick = reference_tick(
            ReferencePriceSource::RtdsChainlink,
            "BTCUSD",
            dec!(1.00000000005),
            ts(1_783_902_701_100),
            None,
            ts(1_783_902_701_200),
            Uuid::new_v4(),
            1,
            None,
            serde_json::json!({}),
        )
        .unwrap();
        assert_eq!(tick.price, dec!(1.0000000001));
    }

    #[test]
    fn parses_direct_binance_aggregate_trade_idempotently() {
        let value = serde_json::json!({
            "e": "aggTrade", "E": 1783902701250_i64, "s": "BTCUSDT", "a": 12345,
            "p": "67236.12345678", "q": "0.5", "f": 12500, "l": 12502,
            "T": 1783902701200_i64, "m": false
        });
        let connection = Uuid::new_v4();
        let first = parse_binance_agg_trade(&value, connection, 1, ts(1_783_902_701_300)).unwrap();
        let second =
            parse_binance_agg_trade(&value, Uuid::new_v4(), 99, ts(1_783_902_702_000)).unwrap();
        assert_eq!(first.source, ReferencePriceSource::DirectBinance);
        assert_eq!(first.price, dec!(67236.12345678));
        assert_eq!(first.dedup_key, second.dedup_key);
        assert_eq!(first.tick_id, second.tick_id);

        let malformed = serde_json::json!({
            "e": "aggTrade", "E": 1783902701250_i64, "s": "BTCUSDT", "a": "invalid",
            "p": "67236.12345678", "q": "0.5", "f": 12500, "l": 12502,
            "T": 1783902701200_i64, "m": false
        });
        assert!(
            parse_binance_agg_trade(&malformed, Uuid::new_v4(), 100, ts(1_783_902_702_100),)
                .is_err()
        );
    }

    #[test]
    fn registry_quarantines_delta_before_full_snapshot() {
        let market = market();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market);
        let delta = ClobMessage::PriceChange {
            market_id: market.market_id.clone(),
            changes: vec![PriceChange {
                token_id: market.up_token_id.clone(),
                side: BookUpdateSide::Bid,
                price: dec!(0.48),
                size: dec!(10),
                source_hash: None,
                best_bid: Some(dec!(0.48)),
                best_ask: Some(dec!(0.52)),
            }],
            source_timestamp: ts(1_783_902_701_000),
        };
        let events = registry.apply(delta, ts(1_783_902_701_010));
        assert!(!events[0].applied);
        assert_eq!(events[0].integrity_status, FeedIntegrityStatus::PreSnapshot);
        assert!(!registry.book_readiness()[1].bootstrapped);
    }

    #[test]
    fn registry_accepts_condition_id_from_wire_and_keeps_canonical_market_id() {
        let market = market();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market);
        let events = registry.apply(
            ClobMessage::Book {
                market_id: market.condition_id.clone(),
                token_id: market.up_token_id.clone(),
                bids: vec![OrderbookLevel {
                    price: dec!(0.48),
                    size: dec!(10),
                }],
                asks: vec![OrderbookLevel {
                    price: dec!(0.52),
                    size: dec!(10),
                }],
                source_timestamp: ts(1_783_902_701_000),
                source_hash: None,
            },
            ts(1_783_902_701_005),
        );
        assert!(events[0].applied);
        assert_eq!(
            events[0].token_id.as_deref(),
            Some(market.up_token_id.as_str())
        );
        assert_eq!(
            registry.checkpoint(&market.up_token_id).unwrap().market_id,
            market.market_id
        );
    }

    fn seed_book(registry: &mut BookRegistry, token: &str, millis: i64) {
        registry.apply(
            ClobMessage::Book {
                market_id: "market".to_string(),
                token_id: token.to_string(),
                bids: vec![OrderbookLevel {
                    price: dec!(0.48),
                    size: dec!(10),
                }],
                asks: vec![OrderbookLevel {
                    price: dec!(0.52),
                    size: dec!(10),
                }],
                source_timestamp: ts(millis),
                source_hash: Some(format!("hash-{token}")),
            },
            ts(millis + 5),
        );
    }

    fn seed_market_book(
        registry: &mut BookRegistry,
        market: &BtcIntervalMarket,
        token_id: &str,
        millis: i64,
    ) {
        let events = registry.apply(
            ClobMessage::Book {
                market_id: market.market_id.clone(),
                token_id: token_id.to_string(),
                bids: vec![OrderbookLevel {
                    price: dec!(0.48),
                    size: dec!(10),
                }],
                asks: vec![OrderbookLevel {
                    price: dec!(0.52),
                    size: dec!(10),
                }],
                source_timestamp: ts(millis),
                source_hash: Some(format!("hash-{token_id}")),
            },
            ts(millis + 5),
        );
        assert!(events.iter().all(|event| event.applied));
    }

    fn assert_same_book_state(before: &OrderbookCheckpoint, after: &OrderbookCheckpoint) {
        assert_eq!(after.market_id, before.market_id);
        assert_eq!(after.token_id, before.token_id);
        assert_eq!(after.source_timestamp, before.source_timestamp);
        assert_eq!(after.received_at, before.received_at);
        assert_eq!(after.connection_id, before.connection_id);
        assert_eq!(after.ingest_sequence, before.ingest_sequence);
        assert_eq!(after.source_hash, before.source_hash);
        assert_eq!(after.tick_size, before.tick_size);
        assert_eq!(after.best_bid, before.best_bid);
        assert_eq!(after.best_ask, before.best_ask);
        assert_eq!(after.bids, before.bids);
        assert_eq!(after.asks, before.asks);
        assert_eq!(after.integrity_status, before.integrity_status);
    }

    #[test]
    fn market_registration_is_idempotent_and_rejects_identity_collisions() {
        let market = market();
        let connection_id = Uuid::new_v4();
        let mut registry = BookRegistry::new(connection_id);
        assert!(registry.try_register_market(&market).unwrap());
        seed_market_book(
            &mut registry,
            &market,
            &market.up_token_id,
            1_783_902_701_000,
        );
        seed_market_book(
            &mut registry,
            &market,
            &market.down_token_id,
            1_783_902_701_000,
        );
        let before = registry.checkpoint(&market.up_token_id).unwrap();
        let next_sequence = registry.next_sequence;

        let mut refreshed = market.clone();
        refreshed.tick_size = dec!(0.001);
        refreshed.raw_payload = serde_json::json!({"refreshed": true});
        registry.register_market(&refreshed);
        assert!(!registry.try_register_market(&refreshed).unwrap());
        assert_eq!(registry.connection_id(), connection_id);
        assert_eq!(registry.len(), 2);
        assert_eq!(registry.next_sequence, next_sequence);
        assert_same_book_state(&before, &registry.checkpoint(&market.up_token_id).unwrap());

        let mut token_collision = numbered_market(1);
        token_collision.up_token_id = market.up_token_id.clone();
        assert!(registry.try_register_market(&token_collision).is_err());

        let mut market_collision = market.clone();
        market_collision.up_token_id = "replacement-up".to_string();
        market_collision.down_token_id = "replacement-down".to_string();
        assert!(registry.try_register_market(&market_collision).is_err());

        let mut reversed_outcomes = market.clone();
        reversed_outcomes.up_token_id = market.down_token_id.clone();
        reversed_outcomes.down_token_id = market.up_token_id.clone();
        assert!(registry.try_register_market(&reversed_outcomes).is_err());
        assert!(registry
            .retain_markets(&[market.clone(), token_collision])
            .is_err());
        assert_eq!(registry.len(), 2);
        assert_same_book_state(&before, &registry.checkpoint(&market.up_token_id).unwrap());
    }

    #[test]
    fn market_set_validation_is_non_mutating_and_scopes_conflicts_to_retained_books() {
        let market = market();
        let connection_id = Uuid::new_v4();
        let mut registry = BookRegistry::new(connection_id);
        assert!(registry.try_register_market(&market).unwrap());
        seed_market_book(
            &mut registry,
            &market,
            &market.up_token_id,
            1_783_902_701_000,
        );
        let before = registry.checkpoint(&market.up_token_id).unwrap();

        let mut replacement = market.clone();
        replacement.up_token_id = "replacement-up".to_string();
        replacement.down_token_id = "replacement-down".to_string();
        registry
            .validate_market_set(std::slice::from_ref(&replacement))
            .unwrap();
        assert_eq!(registry.connection_id(), connection_id);
        assert_eq!(registry.len(), 2);
        assert_same_book_state(&before, &registry.checkpoint(&market.up_token_id).unwrap());

        let mut retained_token_collision = numbered_market(2);
        retained_token_collision.up_token_id = market.up_token_id.clone();
        assert!(registry
            .validate_market_set(std::slice::from_ref(&retained_token_collision))
            .is_err());
        assert_eq!(registry.len(), 2);
        assert_same_book_state(&before, &registry.checkpoint(&market.up_token_id).unwrap());

        assert_eq!(
            registry
                .retain_markets(std::slice::from_ref(&replacement))
                .unwrap(),
            2
        );
        assert!(registry.is_empty());
        assert!(registry.try_register_market(&replacement).unwrap());
        assert_eq!(registry.connection_id(), connection_id);
        assert_eq!(registry.len(), 2);
        assert!(registry
            .book_readiness()
            .iter()
            .all(|book| !book.bootstrapped
                && book.integrity_status == FeedIntegrityStatus::PreSnapshot));
    }

    #[test]
    fn desired_market_retention_preserves_active_state_and_isolates_removed_frames() {
        let removed_market = market();
        let active_market = numbered_market(1);
        let connection_id = Uuid::new_v4();
        let mut registry = BookRegistry::new(connection_id);
        assert!(registry.try_register_market(&removed_market).unwrap());
        seed_market_book(
            &mut registry,
            &removed_market,
            &removed_market.up_token_id,
            1_783_902_701_000,
        );
        seed_market_book(
            &mut registry,
            &removed_market,
            &removed_market.down_token_id,
            1_783_902_701_000,
        );

        assert!(registry.try_register_market(&active_market).unwrap());
        let added = registry
            .book_readiness()
            .into_iter()
            .filter(|book| book.market_id == active_market.market_id)
            .collect::<Vec<_>>();
        assert_eq!(added.len(), 2);
        assert!(added.iter().all(|book| {
            !book.bootstrapped && book.integrity_status == FeedIntegrityStatus::PreSnapshot
        }));
        seed_market_book(
            &mut registry,
            &active_market,
            &active_market.up_token_id,
            1_783_902_701_100,
        );
        seed_market_book(
            &mut registry,
            &active_market,
            &active_market.down_token_id,
            1_783_902_701_100,
        );
        let active_before = registry.checkpoint(&active_market.up_token_id).unwrap();

        assert_eq!(
            registry
                .retain_markets(std::slice::from_ref(&active_market))
                .unwrap(),
            2
        );
        assert_eq!(registry.connection_id(), connection_id);
        assert_eq!(registry.len(), 2);
        assert!(registry.checkpoint(&removed_market.up_token_id).is_none());

        let removed = registry.apply(
            ClobMessage::Book {
                market_id: removed_market.market_id.clone(),
                token_id: removed_market.up_token_id.clone(),
                bids: vec![OrderbookLevel {
                    price: dec!(0.70),
                    size: dec!(100),
                }],
                asks: vec![OrderbookLevel {
                    price: dec!(0.71),
                    size: dec!(100),
                }],
                source_timestamp: ts(1_783_902_701_200),
                source_hash: Some("retired-frame".to_string()),
            },
            ts(1_783_902_701_205),
        );
        assert_eq!(
            removed[0].integrity_status,
            FeedIntegrityStatus::UnknownToken
        );
        assert!(!removed[0].applied);

        let mismatched = registry.apply(
            ClobMessage::PriceChange {
                market_id: removed_market.market_id,
                changes: vec![PriceChange {
                    token_id: active_market.up_token_id.clone(),
                    side: BookUpdateSide::Bid,
                    price: dec!(0.60),
                    size: dec!(50),
                    source_hash: Some("spoofed-retired-frame".to_string()),
                    best_bid: Some(dec!(0.60)),
                    best_ask: Some(dec!(0.61)),
                }],
                source_timestamp: ts(1_783_902_701_210),
            },
            ts(1_783_902_701_215),
        );
        assert_eq!(
            mismatched[0].integrity_status,
            FeedIntegrityStatus::MarketMismatch
        );
        assert!(!mismatched[0].applied);
        assert_same_book_state(
            &active_before,
            &registry.checkpoint(&active_market.up_token_id).unwrap(),
        );
    }

    #[test]
    fn desired_market_retention_keeps_registry_memory_bounded() {
        let connection_id = Uuid::new_v4();
        let mut registry = BookRegistry::new(connection_id);
        let mut desired = Vec::new();

        for index in 1..=64 {
            let market = numbered_market(index);
            assert!(registry.try_register_market(&market).unwrap());
            desired.push(market);
            if desired.len() > 3 {
                desired.remove(0);
            }
            registry.retain_markets(&desired).unwrap();
            assert_eq!(registry.connection_id(), connection_id);
            assert_eq!(registry.len(), desired.len() * 2);
            assert!(registry.len() <= 6);
            for market in &desired {
                assert!(registry.book_readiness().iter().any(|book| {
                    book.token_id == market.up_token_id
                        && book.integrity_status == FeedIntegrityStatus::PreSnapshot
                }));
            }
        }
    }

    #[test]
    fn frame_publication_updates_only_touched_books() {
        let market = market();
        let connection_id = Uuid::new_v4();
        let mut active = BookRegistry::new(connection_id);
        active.register_market(&market);
        seed_book(&mut active, &market.up_token_id, 1_783_902_701_000);
        seed_book(&mut active, &market.down_token_id, 1_783_902_701_000);
        let mut published = active.clone();
        let down_before = published.checkpoint(&market.down_token_id).unwrap();

        let events = active.apply(
            ClobMessage::PriceChange {
                market_id: market.market_id.clone(),
                changes: vec![PriceChange {
                    token_id: market.up_token_id.clone(),
                    side: BookUpdateSide::Ask,
                    price: dec!(0.52),
                    size: dec!(20),
                    source_hash: Some("updated-up".to_string()),
                    best_bid: Some(dec!(0.48)),
                    best_ask: Some(dec!(0.52)),
                }],
                source_timestamp: ts(1_783_902_701_010),
            },
            ts(1_783_902_701_015),
        );
        assert!(events[0].applied);

        published.publish_frame_books_from(&active, [&market.up_token_id].map(String::as_str));

        assert_same_book_state(
            &active.checkpoint(&market.up_token_id).unwrap(),
            &published.checkpoint(&market.up_token_id).unwrap(),
        );
        assert_same_book_state(
            &down_before,
            &published.checkpoint(&market.down_token_id).unwrap(),
        );
    }

    #[test]
    fn market_book_health_requires_complete_causal_integrity_valid_pair() {
        let market = market();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market);
        let source_millis = 1_783_902_701_000;
        let max_age = Duration::milliseconds(20);
        let ready_at = ts(source_millis + 10);

        assert!(!registry.market_books_structurally_ready(&market));
        assert!(!registry.market_books_ready(&market, ready_at, max_age));
        seed_book(&mut registry, &market.up_token_id, source_millis);
        assert!(!registry.market_books_structurally_ready(&market));
        assert!(!registry.market_books_ready(&market, ready_at, max_age));
        seed_book(&mut registry, &market.down_token_id, source_millis);
        assert!(registry.market_books_structurally_ready(&market));
        assert!(registry.market_books_ready(&market, ready_at, max_age));
        assert!(registry.market_books_ready(&market, ts(source_millis + 21), max_age,));
        assert!(!registry.market_books_ready(&market, ts(source_millis - 21), max_age,));
        assert!(registry.market_books_structurally_ready(&market));

        registry.quarantine(FeedIntegrityStatus::Stale);
        assert!(!registry.market_books_structurally_ready(&market));
        assert!(!registry.market_books_ready(&market, ready_at, max_age));
    }

    #[test]
    fn market_book_identity_diagnostic_names_the_exact_hidden_field() {
        let market = market();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market);
        seed_book(&mut registry, &market.up_token_id, 1_783_902_701_000);
        seed_book(&mut registry, &market.down_token_id, 1_783_902_701_000);

        registry
            .books
            .get_mut(&market.up_token_id)
            .unwrap()
            .wire_market_id = "wrong-condition".to_string();
        assert_eq!(
            registry
                .market_book_identity_diagnostic(&market)
                .unwrap()
                .reason,
            "wire_market_identity_mismatch"
        );

        let up = registry.books.get_mut(&market.up_token_id).unwrap();
        up.wire_market_id = market.condition_id.clone();
        up.outcome = BtcOutcome::Down;
        assert_eq!(
            registry
                .market_book_identity_diagnostic(&market)
                .unwrap()
                .reason,
            "outcome_identity_mismatch"
        );
    }

    fn replace_book(
        registry: &mut BookRegistry,
        token: &str,
        bids: &[(Decimal, Decimal)],
        asks: &[(Decimal, Decimal)],
        millis: i64,
        source_hash: &str,
    ) -> Vec<BookApplyResult> {
        registry.apply(
            ClobMessage::Book {
                market_id: "market".to_string(),
                token_id: token.to_string(),
                bids: bids
                    .iter()
                    .map(|(price, size)| OrderbookLevel {
                        price: *price,
                        size: *size,
                    })
                    .collect(),
                asks: asks
                    .iter()
                    .map(|(price, size)| OrderbookLevel {
                        price: *price,
                        size: *size,
                    })
                    .collect(),
                source_timestamp: ts(millis),
                source_hash: Some(source_hash.to_string()),
            },
            ts(millis + 1),
        )
    }

    #[test]
    fn registry_reconciles_fragmented_same_hash_updates_without_false_crosses() {
        let market = market();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market);
        replace_book(
            &mut registry,
            "up",
            &[(dec!(0.43), dec!(10))],
            &[(dec!(0.44), dec!(10)), (dec!(0.46), dec!(10))],
            1_783_902_701_000,
            "before-up",
        );
        replace_book(
            &mut registry,
            "down",
            &[(dec!(0.56), dec!(10)), (dec!(0.54), dec!(10))],
            &[(dec!(0.57), dec!(10))],
            1_783_902_701_000,
            "before-down",
        );

        let first = registry.apply(
            ClobMessage::PriceChange {
                market_id: "market".to_string(),
                changes: vec![
                    PriceChange {
                        token_id: "up".to_string(),
                        side: BookUpdateSide::Bid,
                        price: dec!(0.45),
                        size: dec!(10),
                        source_hash: Some("transition".to_string()),
                        best_bid: Some(dec!(0.45)),
                        best_ask: Some(dec!(0.46)),
                    },
                    PriceChange {
                        token_id: "down".to_string(),
                        side: BookUpdateSide::Ask,
                        price: dec!(0.55),
                        size: dec!(10),
                        source_hash: Some("transition".to_string()),
                        best_bid: Some(dec!(0.54)),
                        best_ask: Some(dec!(0.55)),
                    },
                ],
                source_timestamp: ts(1_783_902_701_100),
            },
            ts(1_783_902_701_101),
        );
        assert!(first
            .iter()
            .all(|event| { event.applied && event.integrity_status == FeedIntegrityStatus::Ok }));

        let second = registry.apply(
            ClobMessage::PriceChange {
                market_id: "market".to_string(),
                changes: vec![
                    PriceChange {
                        token_id: "down".to_string(),
                        side: BookUpdateSide::Bid,
                        price: dec!(0.56),
                        size: Decimal::ZERO,
                        source_hash: Some("transition".to_string()),
                        best_bid: Some(dec!(0.54)),
                        best_ask: Some(dec!(0.55)),
                    },
                    PriceChange {
                        token_id: "up".to_string(),
                        side: BookUpdateSide::Ask,
                        price: dec!(0.44),
                        size: Decimal::ZERO,
                        source_hash: Some("transition".to_string()),
                        best_bid: Some(dec!(0.45)),
                        best_ask: Some(dec!(0.46)),
                    },
                ],
                source_timestamp: ts(1_783_902_701_100),
            },
            ts(1_783_902_701_102),
        );
        assert!(second
            .iter()
            .all(|event| { event.applied && event.integrity_status == FeedIntegrityStatus::Ok }));
        let up = registry.checkpoint("up").unwrap();
        let down = registry.checkpoint("down").unwrap();
        assert_eq!(
            (up.best_bid, up.best_ask),
            (Some(dec!(0.45)), Some(dec!(0.46)))
        );
        assert_eq!(
            (down.best_bid, down.best_ask),
            (Some(dec!(0.54)), Some(dec!(0.55)))
        );

        assert!(
            replace_book(
                &mut registry,
                "up",
                &[(dec!(0.45), dec!(10)), (dec!(0.43), dec!(10))],
                &[(dec!(0.46), dec!(10))],
                1_783_902_701_100,
                "transition",
            )[0]
            .applied
        );
        assert!(
            replace_book(
                &mut registry,
                "down",
                &[(dec!(0.54), dec!(10))],
                &[(dec!(0.55), dec!(10)), (dec!(0.57), dec!(10))],
                1_783_902_701_100,
                "transition",
            )[0]
            .applied
        );
    }

    #[test]
    fn registry_applies_same_token_changes_atomically_and_preserves_batch_hash() {
        let market = market();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market);
        seed_book(&mut registry, "up", 1_783_902_701_000);
        let events = registry.apply(
            ClobMessage::PriceChange {
                market_id: "market".to_string(),
                changes: vec![
                    PriceChange {
                        token_id: "up".to_string(),
                        side: BookUpdateSide::Bid,
                        price: dec!(0.53),
                        size: dec!(10),
                        source_hash: Some("atomic-batch".to_string()),
                        best_bid: Some(dec!(0.53)),
                        best_ask: Some(dec!(0.54)),
                    },
                    PriceChange {
                        token_id: "up".to_string(),
                        side: BookUpdateSide::Ask,
                        price: dec!(0.52),
                        size: Decimal::ZERO,
                        source_hash: None,
                        best_bid: Some(dec!(0.53)),
                        best_ask: Some(dec!(0.54)),
                    },
                    PriceChange {
                        token_id: "up".to_string(),
                        side: BookUpdateSide::Ask,
                        price: dec!(0.54),
                        size: dec!(10),
                        source_hash: None,
                        best_bid: Some(dec!(0.53)),
                        best_ask: Some(dec!(0.54)),
                    },
                ],
                source_timestamp: ts(1_783_902_701_100),
            },
            ts(1_783_902_701_105),
        );
        assert!(events.iter().all(|event| event.applied));
        let checkpoint = registry.checkpoint("up").unwrap();
        assert_eq!(checkpoint.best_bid, Some(dec!(0.53)));
        assert_eq!(checkpoint.best_ask, Some(dec!(0.54)));
        assert_eq!(checkpoint.source_hash.as_deref(), Some("atomic-batch"));
    }

    #[test]
    fn registry_quarantine_is_sticky_until_a_full_snapshot_repairs_it() {
        let market = market();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market);
        replace_book(
            &mut registry,
            "up",
            &[(dec!(0.48), dec!(10)), (dec!(0.47), dec!(10))],
            &[(dec!(0.52), dec!(10))],
            1_783_902_701_000,
            "before-mismatch",
        );

        let mismatch = registry.apply(
            ClobMessage::PriceChange {
                market_id: "market".to_string(),
                changes: vec![PriceChange {
                    token_id: "up".to_string(),
                    side: BookUpdateSide::Bid,
                    price: dec!(0.48),
                    size: Decimal::ZERO,
                    source_hash: Some("mismatch".to_string()),
                    best_bid: Some(dec!(0.46)),
                    best_ask: Some(dec!(0.52)),
                }],
                source_timestamp: ts(1_783_902_701_100),
            },
            ts(1_783_902_701_105),
        );
        assert_eq!(
            mismatch[0].integrity_status,
            FeedIntegrityStatus::TopOfBookMismatch
        );
        assert!(!mismatch[0].applied);
        let quarantined = registry.checkpoint("up").unwrap();
        assert_eq!(quarantined.best_bid, Some(dec!(0.48)));
        assert!(quarantined
            .bids
            .iter()
            .any(|level| level.price == dec!(0.47) && level.size == dec!(10)));
        assert_eq!(
            quarantined.integrity_status,
            FeedIntegrityStatus::TopOfBookMismatch
        );

        let later_delta = registry.apply(
            ClobMessage::PriceChange {
                market_id: "market".to_string(),
                changes: vec![PriceChange {
                    token_id: "up".to_string(),
                    side: BookUpdateSide::Bid,
                    price: dec!(0.49),
                    size: dec!(10),
                    source_hash: Some("later".to_string()),
                    best_bid: Some(dec!(0.49)),
                    best_ask: Some(dec!(0.52)),
                }],
                source_timestamp: ts(1_783_902_701_200),
            },
            ts(1_783_902_701_205),
        );
        assert_eq!(
            later_delta[0].integrity_status,
            FeedIntegrityStatus::TopOfBookMismatch
        );
        assert!(!later_delta[0].applied);
        assert_eq!(
            registry.checkpoint("up").unwrap().best_bid,
            Some(dec!(0.48))
        );

        let repaired = replace_book(
            &mut registry,
            "up",
            &[(dec!(0.49), dec!(10))],
            &[(dec!(0.53), dec!(10))],
            1_783_902_701_300,
            "repair",
        );
        assert!(repaired[0].applied);
        assert_eq!(
            registry.checkpoint("up").unwrap().integrity_status,
            FeedIntegrityStatus::Ok
        );
    }

    #[test]
    fn registry_decode_errors_roll_back_and_require_snapshot_recovery() {
        let market = market();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market);
        seed_book(&mut registry, "up", 1_783_902_701_000);
        let events = registry.apply(
            ClobMessage::PriceChange {
                market_id: "market".to_string(),
                changes: vec![
                    PriceChange {
                        token_id: "up".to_string(),
                        side: BookUpdateSide::Bid,
                        price: dec!(0.49),
                        size: dec!(10),
                        source_hash: Some("bad-batch".to_string()),
                        best_bid: Some(dec!(0.49)),
                        best_ask: Some(dec!(0.52)),
                    },
                    PriceChange {
                        token_id: "up".to_string(),
                        side: BookUpdateSide::Ask,
                        price: dec!(0.53),
                        size: dec!(-1),
                        source_hash: Some("bad-batch".to_string()),
                        best_bid: Some(dec!(0.49)),
                        best_ask: Some(dec!(0.52)),
                    },
                ],
                source_timestamp: ts(1_783_902_701_100),
            },
            ts(1_783_902_701_105),
        );
        assert!(events.iter().all(|event| {
            !event.applied && event.integrity_status == FeedIntegrityStatus::DecodeError
        }));
        let checkpoint = registry.checkpoint("up").unwrap();
        assert_eq!(
            (checkpoint.best_bid, checkpoint.best_ask),
            (Some(dec!(0.48)), Some(dec!(0.52)))
        );
        assert_eq!(
            checkpoint.integrity_status,
            FeedIntegrityStatus::DecodeError
        );

        let blocked = registry.apply(
            ClobMessage::PriceChange {
                market_id: "market".to_string(),
                changes: vec![PriceChange {
                    token_id: "up".to_string(),
                    side: BookUpdateSide::Bid,
                    price: Decimal::ONE,
                    size: dec!(1),
                    source_hash: None,
                    best_bid: Some(dec!(0.49)),
                    best_ask: Some(dec!(0.52)),
                }],
                source_timestamp: ts(1_783_902_701_200),
            },
            ts(1_783_902_701_205),
        );
        assert_eq!(
            blocked[0].integrity_status,
            FeedIntegrityStatus::DecodeError
        );
        assert!(!blocked[0].applied);

        assert!(
            replace_book(
                &mut registry,
                "up",
                &[(dec!(0.49), dec!(10))],
                &[(dec!(0.53), dec!(10))],
                1_783_902_701_300,
                "repair",
            )[0]
            .applied
        );
    }

    #[test]
    fn registry_message_decode_quarantine_requires_snapshot_recovery() {
        let market = market();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market);
        seed_book(&mut registry, "up", 1_783_902_701_000);
        registry.quarantine(FeedIntegrityStatus::DecodeError);
        assert_eq!(
            registry.checkpoint("up").unwrap().integrity_status,
            FeedIntegrityStatus::DecodeError
        );
        assert!(
            replace_book(
                &mut registry,
                "up",
                &[(dec!(0.48), dec!(10))],
                &[(dec!(0.52), dec!(10))],
                1_783_902_701_100,
                "repair",
            )[0]
            .applied
        );
    }

    #[test]
    fn registry_replaces_book_applies_zero_delete_and_rejects_out_of_order() {
        let market = market();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market);
        seed_book(&mut registry, "up", 1_783_902_701_000);
        let event = registry.apply(
            ClobMessage::PriceChange {
                market_id: "market".to_string(),
                changes: vec![PriceChange {
                    token_id: "up".to_string(),
                    side: BookUpdateSide::Bid,
                    price: dec!(0.48),
                    size: Decimal::ZERO,
                    source_hash: Some("next".to_string()),
                    best_bid: None,
                    best_ask: Some(dec!(0.52)),
                }],
                source_timestamp: ts(1_783_902_701_100),
            },
            ts(1_783_902_701_105),
        );
        assert!(event[0].applied);
        assert_eq!(registry.checkpoint("up").unwrap().best_bid, None);

        let old = registry.apply(
            ClobMessage::PriceChange {
                market_id: "market".to_string(),
                changes: vec![PriceChange {
                    token_id: "up".to_string(),
                    side: BookUpdateSide::Bid,
                    price: dec!(0.47),
                    size: dec!(1),
                    source_hash: None,
                    best_bid: None,
                    best_ask: None,
                }],
                source_timestamp: ts(1_783_902_700_000),
            },
            ts(1_783_902_701_200),
        );
        assert_eq!(old[0].integrity_status, FeedIntegrityStatus::OutOfOrder);
        assert!(!old[0].applied);
    }

    #[test]
    fn reconnect_requires_new_full_snapshots() {
        let market = market();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market);
        seed_book(&mut registry, "up", 1_783_902_701_000);
        assert!(registry.checkpoint("up").is_some());
        registry.reset_connection(Uuid::new_v4());
        assert!(registry.checkpoint("up").is_none());
        assert!(registry
            .book_readiness()
            .iter()
            .all(|book| !book.bootstrapped));
    }

    #[test]
    fn readiness_requires_window_books_and_primary_reference_sources() {
        let market = market();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market);
        seed_book(&mut registry, "up", 1_783_902_701_000);
        seed_book(&mut registry, "down", 1_783_902_701_000);
        let now = ts(1_783_902_701_500);
        let mut state = RealtimeState::default();
        state.set_market(market);
        state.update_books(&registry);
        for value in [serde_json::json!({
            "e": "aggTrade", "E": 1783902701400_i64, "s": "BTCUSDT", "a": 1,
            "p": "67000", "q": "1", "f": 1, "l": 1,
            "T": 1783902701400_i64, "m": false
        })] {
            state.update_reference_price(
                parse_binance_agg_trade(&value, Uuid::new_v4(), 1, now).unwrap(),
            );
        }
        let missing = state.readiness(now, Duration::seconds(2), Duration::seconds(2));
        assert!(!missing.ready);
        assert!(missing
            .reasons
            .iter()
            .any(|reason| reason == "missing_reference:rtds_chainlink"));

        state.update_reference_price(
            parse_rtds_reference_tick(
                &serde_json::json!({
                    "topic": "crypto_prices_chainlink", "type": "update",
                    "timestamp": 1783902701400_i64,
                    "payload": {"symbol": "btc/usd", "timestamp": 1783902701400_i64, "value": 67000}
                }),
                Uuid::new_v4(),
                2,
                now,
            )
            .unwrap(),
        );
        assert!(
            state
                .readiness(now, Duration::seconds(2), Duration::seconds(2))
                .ready
        );

        let stale_binance = state.readiness(
            now + Duration::milliseconds(2_001),
            Duration::seconds(2),
            Duration::seconds(2),
        );
        assert!(!stale_binance.ready);
        assert!(stale_binance
            .reasons
            .iter()
            .any(|reason| reason == "stale_reference:direct_binance"));

        let up_book = state.books.get_mut("up").unwrap();
        up_book.source_timestamp = Some(now - Duration::milliseconds(2_500));
        up_book.received_at = Some(now - Duration::milliseconds(1));
        let unchanged_book = state.readiness(now, Duration::seconds(2), Duration::seconds(2));
        assert!(unchanged_book.ready);
    }

    #[test]
    fn readiness_reports_primary_persistence_unavailable_without_a_market() {
        let now = ts(1_783_902_701_500);
        let state = RealtimeState {
            primary_persistence_degraded: true,
            ..RealtimeState::default()
        };

        let readiness = state.readiness(now, Duration::seconds(2), Duration::seconds(2));

        assert!(!readiness.ready);
        assert_eq!(
            readiness.reasons,
            vec![
                "primary_persistence_unavailable".to_string(),
                "missing_current_market".to_string(),
            ]
        );
    }

    #[test]
    fn resolution_maps_winning_token_without_position_assumptions() {
        let mut state = RealtimeState::default();
        state.set_market(market());
        assert!(state.apply_resolution("down"));
        assert_eq!(state.resolved_outcome, Some(BtcOutcome::Down));
        assert!(!state.apply_resolution("unknown"));
        assert_eq!(state.resolved_outcome, Some(BtcOutcome::Down));
        assert!(!state.apply_market_resolution("older-market", "up"));
        assert_eq!(state.resolved_outcome, Some(BtcOutcome::Down));
        state.set_current_market(None);
        assert_eq!(state.resolved_outcome, None);
    }
}
