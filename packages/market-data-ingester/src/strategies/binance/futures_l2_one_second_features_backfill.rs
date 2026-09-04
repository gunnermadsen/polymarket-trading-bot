use async_trait::async_trait;

use super::{l2_backfill_support, l2_support::CryptoHftBinanceMarket};
use crate::domain::{
    BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
    BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
};
use crate::strategies::backfill_support;

pub const STRATEGY_KEY: &str = "binance_futures_btcusdt_l2_one_second_features_backfill";
const LEGACY_KEY: &str = "binance_btcusdt_l2_one_second_features";
const TARGET: &str = "polymarket.binance_btcusdt_l2_one_second_features";

pub struct BinanceFuturesL2OneSecondFeaturesBackfill {
    descriptor: StrategyDescriptor,
}

impl BinanceFuturesL2OneSecondFeaturesBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        Ok(Self { descriptor: backfill_support::descriptor(STRATEGY_KEY, "Binance Futures BTCUSDT L2 one-second features backfill", "Materializes historical Binance Futures BTCUSDT L2 one-second features from CryptoHFT archives")? })
    }
}

#[async_trait]
impl BackfillWorkerStrategy for BinanceFuturesL2OneSecondFeaturesBackfill {
    fn descriptor(&self) -> &StrategyDescriptor {
        &self.descriptor
    }
    fn validate_request(
        &self,
        request: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        backfill_support::validate_empty_request(&self.descriptor, request)
    }
    fn plan_shards(
        &self,
        request: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        backfill_support::daily_shards(request, self.descriptor.maximum_shards)
    }
    async fn execute_backfill(
        &self,
        context: BackfillContext,
        shard: BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        l2_backfill_support::execute(
            STRATEGY_KEY,
            LEGACY_KEY,
            TARGET,
            CryptoHftBinanceMarket::Futures,
            context,
            shard,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{BackfillWorkerStrategy, StrategyCapability};
    use chrono::{TimeZone, Utc};
    #[test]
    fn contract_is_backfill_only_and_shards_without_range_drift() {
        let strategy = BinanceFuturesL2OneSecondFeaturesBackfill::new().unwrap();
        assert_eq!(
            strategy.descriptor().capabilities,
            vec![StrategyCapability::Backfill]
        );
        let start = Utc.with_ymd_and_hms(2026, 7, 2, 4, 0, 0).unwrap();
        let end = start + chrono::Duration::hours(1);
        let request = strategy
            .validate_request(&backfill_support::request(STRATEGY_KEY, start, end))
            .unwrap();
        let shards = strategy.plan_shards(&request).unwrap();
        assert_eq!(shards.len(), 1);
        assert_eq!((shards[0].range_start, shards[0].range_end), (start, end));
    }
}
