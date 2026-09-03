use std::collections::{BTreeMap, HashSet, VecDeque};

use anyhow::{bail, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

use super::directional_features::BTC_DIRECTIONAL_ORACLE_MAX_AGE_SECONDS;
use super::types::{ReferencePriceSource, ReferencePriceTick};

const RTDS_MID_CAPACITY: usize = 4_096;
const ORACLE_CAPACITY: usize = 256;
const OPEN_INTEREST_CAPACITY: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct ChainlinkMidPoint {
    pub source_timestamp: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
    pub price: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChainlinkRefPricePoint {
    pub source_timestamp: DateTime<Utc>,
    pub valid_from_timestamp: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
    pub price: Decimal,
    pub bid: Decimal,
    pub ask: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolygonOraclePoint {
    pub phase_id: u16,
    pub round_id: u64,
    pub source_timestamp: DateTime<Utc>,
    pub block_timestamp: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
    pub price: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinanceOpenInterestPoint {
    pub source_timestamp: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
    pub sum_open_interest: Decimal,
    pub sum_open_interest_value: Decimal,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DirectionalExternalSourceStatus {
    pub last_success_at: Option<DateTime<Utc>>,
    pub last_error_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DirectionalExternalState {
    pub chainlink_mid: VecDeque<ChainlinkMidPoint>,
    pub refprice: VecDeque<ChainlinkRefPricePoint>,
    pub oracle: VecDeque<PolygonOraclePoint>,
    pub open_interest: VecDeque<BinanceOpenInterestPoint>,
    pub source_status: BTreeMap<&'static str, DirectionalExternalSourceStatus>,
}

impl DirectionalExternalState {
    pub fn chainlink_candle_complete_minutes(&self, at: DateTime<Utc>) -> usize {
        const REQUIRED_MINUTES: i64 = 61;
        let latest_close_seconds = at.timestamp().div_euclid(60) * 60;
        let Some(latest_close) = DateTime::from_timestamp(latest_close_seconds, 0) else {
            return 0;
        };
        let earliest_open = latest_close - chrono::Duration::minutes(REQUIRED_MINUTES);
        self.chainlink_mid
            .iter()
            .filter(|point| {
                point.available_at <= at
                    && point.source_timestamp >= earliest_open
                    && point.source_timestamp < latest_close
            })
            .map(|point| point.source_timestamp.timestamp().div_euclid(60))
            .collect::<HashSet<_>>()
            .len()
            .min(REQUIRED_MINUTES as usize)
    }

    pub fn polygon_oracle_age_seconds(&self, at: DateTime<Utc>) -> Option<i64> {
        self.oracle
            .iter()
            .filter(|point| point.available_at <= at && point.block_timestamp <= at)
            .max_by_key(|point| (point.block_timestamp, point.phase_id, point.round_id))
            .map(|point| (at - point.block_timestamp).num_seconds().max(0))
    }

    pub fn polygon_oracle_ready(&self, at: DateTime<Utc>) -> bool {
        self.polygon_oracle_age_seconds(at)
            .is_some_and(|age| age <= BTC_DIRECTIONAL_ORACLE_MAX_AGE_SECONDS)
    }

    pub fn observe_rtds_chainlink(&mut self, tick: &ReferencePriceTick) -> Result<()> {
        if tick.source != ReferencePriceSource::RtdsChainlink {
            return Ok(());
        }
        if tick.price <= Decimal::ZERO {
            bail!("RTDS Chainlink price was invalid for directional features");
        }
        insert_bounded_first_seen(
            &mut self.chainlink_mid,
            ChainlinkMidPoint {
                source_timestamp: tick.source_timestamp,
                available_at: tick.received_at,
                price: tick.price,
            },
            RTDS_MID_CAPACITY,
            |point| point.source_timestamp,
        );
        self.record_success("chainlink_mid", tick.received_at);
        Ok(())
    }

    pub(crate) fn merge_oracle(&mut self, points: Vec<PolygonOraclePoint>, at: DateTime<Utc>) {
        for point in points {
            let identity = (point.phase_id, point.round_id);
            if self
                .oracle
                .iter()
                .any(|row| (row.phase_id, row.round_id) == identity)
            {
                continue;
            }
            self.oracle.push_back(point);
        }
        self.oracle
            .make_contiguous()
            .sort_by_key(|point| (point.block_timestamp, point.phase_id, point.round_id));
        while self.oracle.len() > ORACLE_CAPACITY {
            self.oracle.pop_front();
        }
        self.record_success("oracle", at);
    }

    pub(crate) fn merge_open_interest(
        &mut self,
        points: Vec<BinanceOpenInterestPoint>,
        at: DateTime<Utc>,
    ) {
        for point in points {
            if self
                .open_interest
                .iter()
                .any(|existing| existing.source_timestamp == point.source_timestamp)
            {
                continue;
            }
            self.open_interest.push_back(point);
        }
        self.open_interest
            .make_contiguous()
            .sort_by_key(|point| point.source_timestamp);
        while self.open_interest.len() > OPEN_INTEREST_CAPACITY {
            self.open_interest.pop_front();
        }
        self.record_success("open_interest", at);
    }

    fn record_success(&mut self, source: &'static str, at: DateTime<Utc>) {
        let status = self.source_status.entry(source).or_default();
        status.last_success_at = Some(at);
        status.last_error = None;
    }
}

fn insert_bounded_first_seen<T, F>(rows: &mut VecDeque<T>, value: T, capacity: usize, timestamp: F)
where
    F: Fn(&T) -> DateTime<Utc>,
{
    let key = timestamp(&value);
    if rows.back().is_none_or(|last| timestamp(last) < key) {
        rows.push_back(value);
    } else {
        let insert_at = match rows
            .make_contiguous()
            .binary_search_by_key(&key, &timestamp)
        {
            Ok(_) => return,
            Err(index) => index,
        };
        rows.insert(insert_at, value);
    }
    while rows.len() > capacity {
        rows.pop_front();
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use rust_decimal_macros::dec;

    use super::*;

    #[test]
    fn chainlink_history_counts_distinct_closed_minutes() {
        let at = Utc.with_ymd_and_hms(2026, 9, 3, 12, 1, 30).unwrap();
        let mut state = DirectionalExternalState::default();
        for minute in 0..61 {
            let source_timestamp = DateTime::from_timestamp(at.timestamp().div_euclid(60) * 60, 0)
                .unwrap()
                - chrono::Duration::minutes(minute + 1);
            state.chainlink_mid.push_back(ChainlinkMidPoint {
                source_timestamp,
                available_at: source_timestamp,
                price: dec!(100),
            });
        }
        assert_eq!(state.chainlink_candle_complete_minutes(at), 61);
    }
}
