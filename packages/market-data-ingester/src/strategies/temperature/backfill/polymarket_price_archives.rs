use crate::{
    domain::{
        BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
        BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
    },
    strategies::{
        raw_archive,
        temperature::raw_support::{self, Support},
    },
};
use async_trait::async_trait;
pub const STRATEGY_KEY: &str = "polymarket_temperature_price_archives_backfill";
pub struct PolymarketTemperaturePriceArchivesBackfill {
    support: Support,
}
impl PolymarketTemperaturePriceArchivesBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        Ok(Self {
            support: Support::new(
                STRATEGY_KEY,
                "Polymarket temperature raw price archives",
                "Stores original CLOB temperature price-history responses",
                3660,
            )?,
        })
    }
}
#[async_trait]
impl BackfillWorkerStrategy for PolymarketTemperaturePriceArchivesBackfill {
    fn descriptor(&self) -> &StrategyDescriptor {
        self.support.descriptor()
    }
    fn validate_request(
        &self,
        r: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        self.support.validate(r)
    }
    fn plan_shards(
        &self,
        r: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        self.support.daily(r)
    }
    async fn execute_backfill(
        &self,
        c: BackfillContext,
        s: BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        let objects = raw_support::price_objects(&c, &s).await?;
        let mut v = vec![];
        for o in objects {
            v.push(raw_archive::store(&c, STRATEGY_KEY, &self.support.client, &o).await?);
        }
        Ok(raw_archive::combine(&s, v))
    }
}
