use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{bail, Result};
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use uuid::Uuid;

use super::types::{
    BookReadiness, BtcIntervalMarket, BtcOutcome, FeedIntegrityStatus, OrderbookCheckpoint,
    OrderbookLevel, Readiness, RealtimeState, ReferencePriceSource, ReferencePriceTick,
    SourceReadiness,
};

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

    #[allow(clippy::too_many_arguments)]
    pub fn apply_canonical_snapshot(
        &mut self,
        connection_id: Uuid,
        market: &BtcIntervalMarket,
        token_id: &str,
        outcome: BtcOutcome,
        source_timestamp: DateTime<Utc>,
        received_at: DateTime<Utc>,
        ingest_sequence: u64,
        source_hash: Option<String>,
        bids: Vec<(Decimal, Decimal)>,
        asks: Vec<(Decimal, Decimal)>,
    ) -> Result<BookApplyResult> {
        if self.connection_id != connection_id {
            self.reset_connection(connection_id);
        }
        self.try_register_market(market)?;
        self.apply_canonical_snapshot_to_registered_market(
            &market.market_id,
            &market.condition_id,
            token_id,
            outcome,
            market.tick_size,
            source_timestamp,
            received_at,
            ingest_sequence,
            source_hash,
            bids,
            asks,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_canonical_snapshot_for_market_identity(
        &mut self,
        connection_id: Uuid,
        market_id: &str,
        condition_id: &str,
        up_token_id: &str,
        down_token_id: &str,
        token_id: &str,
        outcome: BtcOutcome,
        tick_size: Decimal,
        source_timestamp: DateTime<Utc>,
        received_at: DateTime<Utc>,
        ingest_sequence: u64,
        source_hash: Option<String>,
        bids: Vec<(Decimal, Decimal)>,
        asks: Vec<(Decimal, Decimal)>,
    ) -> Result<BookApplyResult> {
        if self.connection_id != connection_id {
            self.reset_connection(connection_id);
        }
        self.try_register_market_identity(
            market_id,
            condition_id,
            up_token_id,
            down_token_id,
            tick_size,
        )?;
        self.apply_canonical_snapshot_to_registered_market(
            market_id,
            condition_id,
            token_id,
            outcome,
            tick_size,
            source_timestamp,
            received_at,
            ingest_sequence,
            source_hash,
            bids,
            asks,
        )
    }

    fn try_register_market_identity(
        &mut self,
        market_id: &str,
        condition_id: &str,
        up_token_id: &str,
        down_token_id: &str,
        tick_size: Decimal,
    ) -> Result<()> {
        if up_token_id == down_token_id {
            bail!("orderbook market {market_id} assigns both outcomes to token {up_token_id}");
        }
        for book in self.books.values() {
            let expected_outcome = if book.token_id == up_token_id {
                Some(BtcOutcome::Up)
            } else if book.token_id == down_token_id {
                Some(BtcOutcome::Down)
            } else {
                None
            };
            let token_claimed = expected_outcome.is_some();
            let identifier_claimed = book.market_id == market_id
                || book.market_id == condition_id
                || book.wire_market_id == market_id
                || book.wire_market_id == condition_id;
            let same_market = book.market_id == market_id && book.wire_market_id == condition_id;
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
                    "orderbook market identity {market_id} conflicts with registered token {}",
                    book.token_id
                );
            }
        }
        for (token_id, outcome) in [
            (up_token_id, BtcOutcome::Up),
            (down_token_id, BtcOutcome::Down),
        ] {
            self.books.entry(token_id.to_owned()).or_insert_with(|| {
                FeedBook::new(
                    market_id.to_owned(),
                    condition_id.to_owned(),
                    token_id.to_owned(),
                    outcome,
                    tick_size,
                )
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_canonical_snapshot_to_registered_market(
        &mut self,
        market_id: &str,
        condition_id: &str,
        token_id: &str,
        outcome: BtcOutcome,
        tick_size: Decimal,
        source_timestamp: DateTime<Utc>,
        received_at: DateTime<Utc>,
        ingest_sequence: u64,
        source_hash: Option<String>,
        bids: Vec<(Decimal, Decimal)>,
        asks: Vec<(Decimal, Decimal)>,
    ) -> Result<BookApplyResult> {
        let Some(book) = self.books.get_mut(token_id) else {
            bail!("canonical snapshot token is not registered");
        };
        if book.outcome != outcome
            || book.market_id != market_id
            || book.wire_market_id != condition_id
        {
            bail!("canonical snapshot market identity mismatch");
        }
        if tick_size <= Decimal::ZERO || tick_size >= Decimal::ONE || tick_size.scale() > 8 {
            bail!("canonical snapshot tick size is invalid");
        }
        if book
            .source_timestamp
            .is_some_and(|current| source_timestamp < current)
        {
            return Ok(book_apply_result(
                Some(token_id.to_owned()),
                source_timestamp,
                false,
                false,
                FeedIntegrityStatus::OutOfOrder,
            ));
        }
        // The ingester validates venue tick-size transitions before emitting
        // canonical snapshots. Gamma discovery can therefore legitimately
        // register this exact market identity at an older tick before the
        // current CLOB snapshot arrives.
        book.tick_size = tick_size;
        book.bids = bids
            .into_iter()
            .filter(|(_, size)| !size.is_zero())
            .collect();
        book.asks = asks
            .into_iter()
            .filter(|(_, size)| !size.is_zero())
            .collect();
        book.bootstrapped = true;
        book.source_timestamp = Some(source_timestamp);
        book.received_at = Some(received_at);
        book.source_hash = source_hash;
        book.ingest_sequence = ingest_sequence;
        book.validate();
        self.next_sequence = self.next_sequence.max(ingest_sequence.saturating_add(1));
        Ok(book_apply_result(
            Some(token_id.to_owned()),
            source_timestamp,
            book.integrity_status == FeedIntegrityStatus::Ok,
            true,
            book.integrity_status,
        ))
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
                    && book.source_timestamp.is_some_and(|timestamp| {
                        timestamp - now <= max_age && now - timestamp <= max_age
                    })
                    && book
                        .received_at
                        .is_some_and(|timestamp| timestamp <= now && now - timestamp <= max_age)
                    && book.source_timestamp.zip(book.received_at).is_some_and(
                        |(source_timestamp, received_at)| received_at - source_timestamp <= max_age,
                    )
            })
        })
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
                Some(book)
                    if book
                        .source_timestamp
                        .is_none_or(|timestamp| now - timestamp > max_book_age) =>
                {
                    reasons.push(format!("stale_book_timestamp:{token_id}"))
                }
                Some(book) if book.received_at.is_none_or(|timestamp| timestamp > now) => {
                    reasons.push(format!("future_book_timestamp:{token_id}"))
                }
                Some(book)
                    if book
                        .received_at
                        .is_some_and(|timestamp| now - timestamp > max_book_age) =>
                {
                    reasons.push(format!("stale_book_receipt:{token_id}"))
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
