use super::{backfill_runtime, support::PmdataTwapWindow};
use crate::{
    domain::{
        BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
        BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
    },
    strategies::backfill_support,
};
use async_trait::async_trait;
pub const STRATEGY_KEY: &str = "pmdata_chainlink_btcusd_twap_60s_backfill";
pub struct PmdataChainlinkBtcusdTwap60sBackfill {
    descriptor: StrategyDescriptor,
}
impl PmdataChainlinkBtcusdTwap60sBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        Ok(Self {
            descriptor: backfill_support::descriptor(
                STRATEGY_KEY,
                "PMData Chainlink BTC/USD 60-second TWAP backfill",
                "Collects PMData streams_twap60s daily Parquet archives",
            )?,
        })
    }
}
#[async_trait]
impl BackfillWorkerStrategy for PmdataChainlinkBtcusdTwap60sBackfill {
    fn descriptor(&self) -> &StrategyDescriptor {
        &self.descriptor
    }
    fn validate_request(
        &self,
        r: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        backfill_support::validate_empty_request(&self.descriptor, r)
    }
    fn plan_shards(
        &self,
        r: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        backfill_support::daily_shards(r, self.descriptor.maximum_shards)
    }
    async fn execute_backfill(
        &self,
        c: BackfillContext,
        s: BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        backfill_runtime::execute_twap(c, s, STRATEGY_KEY, PmdataTwapWindow::Seconds60).await
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::StrategyCapability;
    use chrono::{TimeZone, Utc};
    #[test]
    fn one_hour_backfill_only() {
        let s = PmdataChainlinkBtcusdTwap60sBackfill::new().unwrap();
        assert_eq!(
            s.descriptor().capabilities,
            vec![StrategyCapability::Backfill]
        );
        let start = Utc.with_ymd_and_hms(2026, 8, 2, 1, 0, 0).unwrap();
        let r = backfill_support::request(STRATEGY_KEY, start, start + chrono::Duration::hours(1));
        assert_eq!(
            s.plan_shards(&s.validate_request(&r).unwrap())
                .unwrap()
                .len(),
            1
        );
    }
}
