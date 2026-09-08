use std::collections::{BTreeMap, VecDeque};

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

use super::directional_features::BTC_DIRECTIONAL_ORACLE_MAX_AGE_SECONDS;

const ORACLE_CAPACITY: usize = 256;
const OPEN_INTEREST_CAPACITY: usize = 32;

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
    pub(crate) rtds: std::sync::Arc<super::rtds_repository::RtdsRepository>,
    pub refprice: VecDeque<ChainlinkRefPricePoint>,
    pub oracle: VecDeque<PolygonOraclePoint>,
    pub open_interest: VecDeque<BinanceOpenInterestPoint>,
    pub source_status: BTreeMap<&'static str, DirectionalExternalSourceStatus>,
}

impl DirectionalExternalState {
    pub fn rtds(&self) -> &super::rtds_repository::RtdsRepository {
        &self.rtds
    }

    /// Compatibility status projection; history is owned only by the repository.
    pub(crate) fn observe_rtds_chainlink(
        &mut self,
        tick: &super::types::ReferencePriceTick,
    ) -> anyhow::Result<()> {
        std::sync::Arc::make_mut(&mut self.rtds).observe(tick)?;
        if tick.source == super::types::ReferencePriceSource::RtdsChainlink {
            self.record_success("chainlink_mid", tick.received_at);
        }
        Ok(())
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
