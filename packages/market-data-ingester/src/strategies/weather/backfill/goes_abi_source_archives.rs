use crate::{
    domain::{
        BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
        BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
    },
    strategies::{
        raw_archive,
        weather::raw_support::{self, Support},
    },
};
use async_trait::async_trait;
pub const STRATEGY_KEY: &str = "goes_abi_source_archives_backfill";
pub struct GoesAbiSourceArchivesBackfill {
    support: Support,
}
impl GoesAbiSourceArchivesBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        Ok(Self {
            support: Support::new(
                STRATEGY_KEY,
                "GOES ABI raw source archives",
                "Stores original NOAA GOES ABI NetCDF objects",
                17520,
            )?,
        })
    }
}
#[async_trait]
impl BackfillWorkerStrategy for GoesAbiSourceArchivesBackfill {
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
        self.support.hourly(r)
    }
    async fn execute_backfill(
        &self,
        c: BackfillContext,
        s: BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        let objects = raw_support::goes_objects(&self.support.client, &s).await?;
        let mut out = Vec::new();
        for o in objects {
            out.push(raw_archive::store(&c, STRATEGY_KEY, &self.support.client, &o).await?);
        }
        Ok(raw_archive::combine(&s, out))
    }
}
