use std::collections::{BTreeMap, VecDeque};

use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use serde::{ser::SerializeStruct, Deserialize, Serialize, Serializer};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Level {
    pub price: Decimal,
    pub size: Decimal,
    pub observed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BookSide {
    Bid,
    Ask,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FillQuote {
    pub price: Decimal,
    pub size: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DepthWalk {
    pub total: Decimal,
    pub fills: Vec<FillQuote>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DepthSummary {
    pub fillable_size: Decimal,
    pub total: Decimal,
    pub avg_price: Option<Decimal>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LocalOrderBook {
    bids: BTreeMap<Decimal, Level>,
    asks: BTreeMap<Decimal, Level>,
    best_history: VecDeque<(DateTime<Utc>, Decimal)>,
}

impl Serialize for LocalOrderBook {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bids = self.levels(BookSide::Bid);
        let asks = self.levels(BookSide::Ask);
        let mut state = serializer.serialize_struct("LocalOrderBook", 3)?;
        state.serialize_field("bids", &bids)?;
        state.serialize_field("asks", &asks)?;
        state.serialize_field("best_history", &self.best_history)?;
        state.end()
    }
}

impl LocalOrderBook {
    pub fn upsert_level(
        &mut self,
        side: BookSide,
        price: Decimal,
        size: Decimal,
        observed_at: DateTime<Utc>,
    ) {
        let map = match side {
            BookSide::Bid => &mut self.bids,
            BookSide::Ask => &mut self.asks,
        };
        if size <= Decimal::ZERO {
            map.remove(&price);
            return;
        }
        let reset_age = map
            .get(&price)
            .map(|level| level.size != size)
            .unwrap_or(true);
        let stored_at = if reset_age {
            observed_at
        } else {
            map.get(&price)
                .map(|level| level.observed_at)
                .unwrap_or(observed_at)
        };
        map.insert(
            price,
            Level {
                price,
                size,
                observed_at: stored_at,
            },
        );
    }

    pub fn record_best_mid(&mut self, at: DateTime<Utc>) {
        if let Some(mid) = self.mid() {
            self.best_history.push_back((at, mid));
            while self.best_history.len() > 3600 {
                self.best_history.pop_front();
            }
        }
    }

    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.keys().next_back().copied()
    }

    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.keys().next().copied()
    }

    pub fn mid(&self) -> Option<Decimal> {
        Some((self.best_bid()? + self.best_ask()?) / Decimal::from(2))
    }

    pub fn depth_walk_buy(
        &self,
        target: Decimal,
        max_age: Duration,
        now: DateTime<Utc>,
    ) -> Option<DepthWalk> {
        self.depth_walk(BookSide::Ask, target, None, max_age, now, true)
    }

    pub fn depth_walk_buy_limit(
        &self,
        target: Decimal,
        limit_price: Decimal,
        max_age: Duration,
        now: DateTime<Utc>,
    ) -> Option<DepthWalk> {
        self.depth_walk(BookSide::Ask, target, Some(limit_price), max_age, now, true)
    }

    pub fn depth_walk_buy_limit_partial(
        &self,
        target: Decimal,
        limit_price: Decimal,
        max_age: Duration,
        now: DateTime<Utc>,
    ) -> Option<DepthWalk> {
        self.depth_walk(
            BookSide::Ask,
            target,
            Some(limit_price),
            max_age,
            now,
            false,
        )
    }

    pub fn depth_walk_sell(
        &self,
        target: Decimal,
        max_age: Duration,
        now: DateTime<Utc>,
    ) -> Option<DepthWalk> {
        self.depth_walk(BookSide::Bid, target, None, max_age, now, true)
    }

    pub fn depth_walk_sell_limit(
        &self,
        target: Decimal,
        limit_price: Decimal,
        max_age: Duration,
        now: DateTime<Utc>,
    ) -> Option<DepthWalk> {
        self.depth_walk(BookSide::Bid, target, Some(limit_price), max_age, now, true)
    }

    pub fn depth_walk_sell_limit_partial(
        &self,
        target: Decimal,
        limit_price: Decimal,
        max_age: Duration,
        now: DateTime<Utc>,
    ) -> Option<DepthWalk> {
        self.depth_walk(
            BookSide::Bid,
            target,
            Some(limit_price),
            max_age,
            now,
            false,
        )
    }

    pub fn limit_depth_summary(
        &self,
        side: BookSide,
        limit_price: Decimal,
        max_age: Duration,
        now: DateTime<Utc>,
    ) -> DepthSummary {
        let mut fillable_size = Decimal::ZERO;
        let mut total = Decimal::ZERO;
        for level in self.levels(side) {
            if now - level.observed_at > max_age {
                continue;
            }
            let crosses_limit = match side {
                BookSide::Ask => level.price <= limit_price,
                BookSide::Bid => level.price >= limit_price,
            };
            if !crosses_limit {
                continue;
            }
            fillable_size += level.size;
            total += level.size * level.price;
        }
        DepthSummary {
            fillable_size,
            total,
            avg_price: if fillable_size > Decimal::ZERO {
                Some(total / fillable_size)
            } else {
                None
            },
        }
    }

    pub fn fresh_depth_within_ticks(
        &self,
        side: BookSide,
        tick_size: Decimal,
        ticks: Decimal,
        max_age: Duration,
        now: DateTime<Utc>,
    ) -> Decimal {
        let Some(anchor) = (match side {
            BookSide::Bid => self.best_bid(),
            BookSide::Ask => self.best_ask(),
        }) else {
            return Decimal::ZERO;
        };
        let max_distance = tick_size * ticks;
        self.levels(side)
            .into_iter()
            .filter(|level| now - level.observed_at <= max_age)
            .filter(|level| match side {
                BookSide::Bid => anchor - level.price <= max_distance,
                BookSide::Ask => level.price - anchor <= max_distance,
            })
            .map(|level| level.size)
            .sum()
    }

    pub fn observed_price_velocity_per_second(&self, window: Duration) -> Decimal {
        let Some((latest_ts, latest_price)) = self.best_history.back().copied() else {
            return Decimal::ZERO;
        };
        let cutoff = latest_ts - window;
        let mut movements = Vec::new();
        let mut prev: Option<(DateTime<Utc>, Decimal)> = None;
        for (ts, price) in self
            .best_history
            .iter()
            .copied()
            .filter(|(ts, _)| *ts >= cutoff)
        {
            if let Some((prev_ts, prev_price)) = prev {
                let seconds = (ts - prev_ts).num_milliseconds().max(1) as i64;
                let movement =
                    (price - prev_price).abs() * Decimal::from(1000) / Decimal::from(seconds);
                movements.push(movement);
            }
            prev = Some((ts, price));
        }
        if movements.is_empty() {
            return latest_price * Decimal::ZERO;
        }
        movements.sort();
        let idx = ((movements.len() - 1) as f64 * 0.75).floor() as usize;
        movements[idx]
    }

    fn depth_walk(
        &self,
        side: BookSide,
        target: Decimal,
        limit_price: Option<Decimal>,
        max_age: Duration,
        now: DateTime<Utc>,
        require_full: bool,
    ) -> Option<DepthWalk> {
        if target <= Decimal::ZERO {
            return Some(DepthWalk {
                total: Decimal::ZERO,
                fills: vec![],
            });
        }
        let mut remaining = target;
        let mut total = Decimal::ZERO;
        let mut fills = Vec::new();
        for level in self.levels(side) {
            if now - level.observed_at > max_age {
                continue;
            }
            if let Some(limit_price) = limit_price {
                match side {
                    BookSide::Ask if level.price > limit_price => continue,
                    BookSide::Bid if level.price < limit_price => continue,
                    _ => {}
                }
            }
            let fill = remaining.min(level.size);
            if fill <= Decimal::ZERO {
                continue;
            }
            total += fill * level.price;
            fills.push(FillQuote {
                price: level.price,
                size: fill,
            });
            remaining -= fill;
            if remaining <= Decimal::ZERO {
                return Some(DepthWalk { total, fills });
            }
        }
        if !require_full && !fills.is_empty() {
            return Some(DepthWalk { total, fills });
        }
        None
    }

    fn levels(&self, side: BookSide) -> Vec<&Level> {
        match side {
            BookSide::Ask => self.asks.values().collect(),
            BookSide::Bid => self.bids.values().rev().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::Duration;
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    use super::{BookSide, LocalOrderBook};

    #[test]
    fn depth_walk_buy_excludes_stale_levels() {
        let now = chrono::Utc::now();
        let mut book = LocalOrderBook::default();
        book.upsert_level(
            BookSide::Ask,
            dec!(0.40),
            dec!(10),
            now - Duration::seconds(60),
        );
        book.upsert_level(BookSide::Ask, dec!(0.41), dec!(10), now);

        let quote = book
            .depth_walk_buy(dec!(10), Duration::seconds(45), now)
            .unwrap();
        assert_eq!(quote.total, dec!(4.10));
        assert_eq!(quote.fills[0].price, dec!(0.41));
    }

    #[test]
    fn depth_walk_sell_orders_bids_descending() {
        let now = chrono::Utc::now();
        let mut book = LocalOrderBook::default();
        book.upsert_level(BookSide::Bid, dec!(0.39), dec!(10), now);
        book.upsert_level(BookSide::Bid, dec!(0.40), dec!(5), now);

        let quote = book
            .depth_walk_sell(dec!(10), Duration::seconds(45), now)
            .unwrap();
        assert_eq!(quote.total, dec!(3.95));
        assert_eq!(quote.fills[0].price, dec!(0.40));
        assert_eq!(quote.fills[1].price, dec!(0.39));
    }

    #[test]
    fn limit_walk_rejects_prices_outside_limit() {
        let now = chrono::Utc::now();
        let mut book = LocalOrderBook::default();
        book.upsert_level(BookSide::Ask, dec!(0.41), dec!(5), now);
        book.upsert_level(BookSide::Ask, dec!(0.42), dec!(5), now);
        book.upsert_level(BookSide::Bid, dec!(0.39), dec!(5), now);
        book.upsert_level(BookSide::Bid, dec!(0.38), dec!(5), now);

        assert!(book
            .depth_walk_buy_limit(dec!(10), dec!(0.41), Duration::seconds(45), now)
            .is_none());
        assert!(book
            .depth_walk_sell_limit(dec!(10), dec!(0.39), Duration::seconds(45), now)
            .is_none());

        let buy = book
            .depth_walk_buy_limit(dec!(10), dec!(0.42), Duration::seconds(45), now)
            .unwrap();
        let sell = book
            .depth_walk_sell_limit(dec!(10), dec!(0.38), Duration::seconds(45), now)
            .unwrap();
        assert_eq!(buy.total, dec!(4.15));
        assert_eq!(sell.total, dec!(3.85));
    }

    #[test]
    fn partial_limit_walk_returns_available_crossing_depth() {
        let now = chrono::Utc::now();
        let mut book = LocalOrderBook::default();
        book.upsert_level(BookSide::Ask, dec!(0.41), dec!(5), now);
        book.upsert_level(BookSide::Ask, dec!(0.42), dec!(3), now);

        assert!(book
            .depth_walk_buy_limit(dec!(10), dec!(0.42), Duration::seconds(45), now)
            .is_none());

        let partial = book
            .depth_walk_buy_limit_partial(dec!(10), dec!(0.42), Duration::seconds(45), now)
            .unwrap();

        assert_eq!(partial.total, dec!(3.31));
        assert_eq!(partial.fills.len(), 2);
        assert_eq!(
            partial.fills.iter().map(|fill| fill.size).sum::<Decimal>(),
            dec!(8)
        );
    }

    #[test]
    fn limit_depth_summary_reports_available_crossing_depth() {
        let now = chrono::Utc::now();
        let mut book = LocalOrderBook::default();
        book.upsert_level(BookSide::Bid, dec!(0.40), dec!(5), now);
        book.upsert_level(BookSide::Bid, dec!(0.39), dec!(10), now);
        book.upsert_level(BookSide::Bid, dec!(0.38), dec!(20), now);

        let summary =
            book.limit_depth_summary(BookSide::Bid, dec!(0.39), Duration::seconds(45), now);

        assert_eq!(summary.fillable_size, dec!(15));
        assert_eq!(summary.total, dec!(5.90));
        assert_eq!(
            summary.avg_price,
            Some(dec!(0.3933333333333333333333333333))
        );
    }
}
