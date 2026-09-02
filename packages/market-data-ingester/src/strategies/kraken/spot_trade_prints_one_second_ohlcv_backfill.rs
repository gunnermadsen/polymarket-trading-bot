use async_trait::async_trait;

use crate::domain::{
    BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
    BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
};

use super::backfill_support;

pub const STRATEGY_KEY: &str = "kraken_spot_btcusd_trade_prints_one_second_ohlcv_backfill";

pub struct KrakenSpotBtcusdTradePrintsOneSecondOhlcvBackfill {
    descriptor: StrategyDescriptor,
}

impl KrakenSpotBtcusdTradePrintsOneSecondOhlcvBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        Ok(Self {
            descriptor: backfill_support::descriptor(
                STRATEGY_KEY,
                "Kraken Spot BTC/USD trade prints and one-second OHLCV backfill",
                "Collects Kraken Spot BTC/USD trades and materializes one-second OHLCV Parquet",
            )?,
        })
    }
}

#[async_trait]
impl BackfillWorkerStrategy for KrakenSpotBtcusdTradePrintsOneSecondOhlcvBackfill {
    fn descriptor(&self) -> &StrategyDescriptor {
        &self.descriptor
    }

    fn validate_request(
        &self,
        request: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        let validated = backfill_support::validate_request(&self.descriptor, request)?;
        let parameters: backfill_support::KrakenParameters =
            serde_json::from_value(validated.parameters.clone()).map_err(|error| {
                BackfillExecutionError::invalid("parameters_invalid", error.to_string())
            })?;
        if parameters.symbol != backfill_support::DEFAULT_SYMBOL
            || parameters.interval_seconds != backfill_support::DEFAULT_INTERVAL_SECONDS
        {
            return Err(BackfillExecutionError::invalid(
                "parameters_invalid",
                "Kraken Spot trade-print backfill uses the fixed XBTUSD source and 900-second control alignment",
            ));
        }
        Ok(validated)
    }

    fn plan_shards(
        &self,
        request: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        backfill_support::plan_time_shards(request, self.descriptor.maximum_shards)
    }

    async fn execute_backfill(
        &self,
        context: BackfillContext,
        shard: BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        backfill_support::execute_spot(context, shard, STRATEGY_KEY).await
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::*;
    use crate::domain::StrategyCapability;

    #[test]
    fn contract_is_backfill_only_and_daily_sharded() {
        let strategy = KrakenSpotBtcusdTradePrintsOneSecondOhlcvBackfill::new().unwrap();
        assert_eq!(
            strategy.descriptor().capabilities,
            vec![StrategyCapability::Backfill]
        );
        let start = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
        let request = backfill_support::test_request(
            STRATEGY_KEY,
            start,
            start + chrono::Duration::hours(49),
        );
        let validated = strategy.validate_request(&request).unwrap();
        assert_eq!(strategy.plan_shards(&validated).unwrap().len(), 3);
    }

    #[test]
    fn rejects_another_strategy_identity() {
        let strategy = KrakenSpotBtcusdTradePrintsOneSecondOhlcvBackfill::new().unwrap();
        let start = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
        let request = backfill_support::test_request(
            "kraken_trade_candles_backfill",
            start,
            start + chrono::Duration::hours(1),
        );
        assert!(strategy.validate_request(&request).is_err());
    }
}
