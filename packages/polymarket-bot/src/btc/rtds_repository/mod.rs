//! Single shared, bounded RTDS history. Consumers receive immutable snapshots.
mod candles;
#[cfg(test)]
mod tests;
use crate::btc::{
    directional_features::{DirectionalChainlinkCandle, DirectionalFeatureError},
    types::{ReferencePriceSource, ReferencePriceTick},
};
use anyhow::{bail, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::collections::{HashSet, VecDeque};
const RTDS_MID_CAPACITY: usize = 4096;
#[derive(Debug, Clone, PartialEq)]
pub struct RtdsPoint {
    pub source_timestamp: DateTime<Utc>,
    pub available_at: DateTime<Utc>,
    pub price: Decimal,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RtdsRepository {
    points: VecDeque<RtdsPoint>,
}
impl RtdsRepository {
    /// Writer entry shared by database hydration and live ingestion.
    pub(crate) fn observe(&mut self, tick: &ReferencePriceTick) -> Result<()> {
        if tick.source != ReferencePriceSource::RtdsChainlink {
            return Ok(());
        }
        self.insert(RtdsPoint {
            source_timestamp: tick.source_timestamp,
            available_at: tick.received_at,
            price: tick.price,
        })
    }
    fn insert(&mut self, point: RtdsPoint) -> Result<()> {
        if point.price <= Decimal::ZERO {
            bail!("RTDS Chainlink price was invalid for directional features");
        }
        insert_bounded_first_seen(&mut self.points, point, RTDS_MID_CAPACITY, |p| {
            p.source_timestamp
        });
        Ok(())
    }
    /// Source-time ordered points, filtered by historical availability.
    pub fn points_as_of(&self, at: DateTime<Utc>) -> impl Iterator<Item = &RtdsPoint> {
        self.points
            .iter()
            .filter(move |p| p.available_at <= at && p.source_timestamp <= at)
    }
    /// Exact legacy RTDS recipe: 61 contiguous closed minutes, no synthetic gaps.
    pub fn closed_candles(
        &self,
        at: DateTime<Utc>,
    ) -> Result<Vec<DirectionalChainlinkCandle>, DirectionalFeatureError> {
        candles::closed_candles(self.points_as_of(at), at)
    }
    pub fn complete_minutes(&self, at: DateTime<Utc>) -> usize {
        const REQUIRED_MINUTES: i64 = 61;
        let latest_close_seconds = at.timestamp().div_euclid(60) * 60;
        let Some(latest_close) = DateTime::from_timestamp(latest_close_seconds, 0) else {
            return 0;
        };
        let earliest_open = latest_close - chrono::Duration::minutes(REQUIRED_MINUTES);
        self.points
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

    #[cfg(test)]
    pub(crate) fn insert_fixture(&mut self, point: RtdsPoint) {
        self.insert(point).unwrap();
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
