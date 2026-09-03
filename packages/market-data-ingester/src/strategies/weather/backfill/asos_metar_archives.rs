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
pub const STRATEGY_KEY: &str = "asos_metar_archives_backfill";
pub struct AsosMetarArchivesBackfill {
    support: Support,
}
impl AsosMetarArchivesBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        Ok(Self {
            support: Support::new(
                STRATEGY_KEY,
                "ASOS METAR raw archives",
                "Stores original IEM ASOS/METAR CSV responses",
                3660,
            )?,
        })
    }
}
#[async_trait]
impl BackfillWorkerStrategy for AsosMetarArchivesBackfill {
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
        let objects = raw_support::asos_objects(&self.support.client, &s, false).await?;
        let mut out = Vec::new();
        for o in objects {
            out.push(raw_archive::store(&c, STRATEGY_KEY, &self.support.client, &o).await?);
        }
        Ok(raw_archive::combine(&s, out))
    }
}
