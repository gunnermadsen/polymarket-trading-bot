use async_trait::async_trait;

use crate::domain::{
    BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
    BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
};

use super::support::{BackfillSupport, Kind};

pub struct PolymarketBtcResolutionsBackfill(BackfillSupport);

impl PolymarketBtcResolutionsBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        BackfillSupport::new(Kind::Resolutions).map(Self)
    }
}

#[async_trait]
impl BackfillWorkerStrategy for PolymarketBtcResolutionsBackfill {
    fn descriptor(&self) -> &StrategyDescriptor {
        self.0.descriptor()
    }

    fn validate_request(
        &self,
        request: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        self.0.validate_request(request)
    }

    fn plan_shards(
        &self,
        request: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        self.0.plan_shards(request)
    }

    async fn execute_backfill(
        &self,
        context: BackfillContext,
        shard: BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        self.0.execute_resolutions(&context, &shard).await
    }
}
