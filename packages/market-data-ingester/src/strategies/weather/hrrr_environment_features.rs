use async_trait::async_trait;

use crate::domain::{
    BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
    BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
};

use super::support::WeatherBackfillSupport;

pub struct HrrrEnvironmentFeaturesBackfill {
    support: WeatherBackfillSupport,
}

impl HrrrEnvironmentFeaturesBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        Ok(Self {
            support: WeatherBackfillSupport::new(
                "hrrr_environment_features",
                "HRRR KLGA environmental features",
                "Collects historical HRRR environmental features for KLGA",
            )?,
        })
    }
}

#[async_trait]
impl BackfillWorkerStrategy for HrrrEnvironmentFeaturesBackfill {
    fn descriptor(&self) -> &StrategyDescriptor {
        self.support.descriptor()
    }

    fn validate_request(
        &self,
        request: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        self.support.validate_request(request)
    }

    fn plan_shards(
        &self,
        request: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        self.support.plan_shards(request)
    }

    async fn execute_backfill(
        &self,
        context: BackfillContext,
        shard: BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        self.support.execute_backfill(context, shard).await
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn retains_hrrr_strategy_identity() {
        let strategy = HrrrEnvironmentFeaturesBackfill::new().unwrap();
        assert_eq!(
            strategy.descriptor().strategy_key.as_ref(),
            "hrrr_environment_features"
        );
        let request: BackfillRequest = serde_json::from_value(json!({
            "strategy_key": "hrrr_environment_features",
            "range": {
                "start": "2020-01-01T00:00:00Z",
                "end": "2020-02-01T00:00:00Z"
            },
            "parameters": {}
        }))
        .unwrap();
        assert!(strategy.validate_request(&request).is_ok());
    }
}
