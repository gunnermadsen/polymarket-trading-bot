use async_trait::async_trait;

use crate::domain::{
    BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
    BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
};

use super::support::WeatherBackfillSupport;

pub struct GoesAbiKlgaFeaturesBackfill {
    support: WeatherBackfillSupport,
}

impl GoesAbiKlgaFeaturesBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        Ok(Self {
            support: WeatherBackfillSupport::new(
                "goes_abi_klga_features",
                "GOES ABI KLGA environmental features",
                "Collects historical GOES ABI satellite features for KLGA",
            )?,
        })
    }
}

#[async_trait]
impl BackfillWorkerStrategy for GoesAbiKlgaFeaturesBackfill {
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
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    use super::*;

    #[test]
    fn plans_monthly_shards() {
        let strategy = GoesAbiKlgaFeaturesBackfill::new().unwrap();
        let request: BackfillRequest = serde_json::from_value(json!({
            "strategy_key": "goes_abi_klga_features",
            "range": {
                "start": "2020-01-15T00:00:00Z",
                "end": "2020-03-02T00:00:00Z"
            },
            "parameters": {"feature_schema_version":"goes-klga-v2"}
        }))
        .unwrap();
        let validated = strategy.validate_request(&request).unwrap();
        let shards = strategy.plan_shards(&validated).unwrap();
        assert_eq!(shards.len(), 3);
        assert_eq!(
            shards[0].range_end,
            Utc.with_ymd_and_hms(2020, 2, 1, 0, 0, 0).unwrap()
        );
        assert_eq!(shards[2].range_end, request.range.end);
    }
}
