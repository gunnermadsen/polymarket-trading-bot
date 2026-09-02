//! Kraken historical ingestion strategies.

macro_rules! define_kraken_futures_strategy {
    ($type_name:ident, $key:literal, $name:literal, $description:literal, $dataset:ident, time) => {
        define_kraken_futures_strategy!(@common $type_name, $key, $name, $description, $dataset, plan_time_shards);
    };
    ($type_name:ident, $key:literal, $name:literal, $description:literal, $dataset:ident, snapshot) => {
        define_kraken_futures_strategy!(@common $type_name, $key, $name, $description, $dataset, plan_snapshot_shard);
    };
    (@common $type_name:ident, $key:literal, $name:literal, $description:literal, $dataset:ident, $planner:ident) => {
        use async_trait::async_trait;
        use crate::domain::{
            BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest,
            BackfillShard, BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
        };
        use super::{backfill_support, types::KrakenDataset};

        pub const STRATEGY_KEY: &str = $key;

        pub struct $type_name {
            descriptor: StrategyDescriptor,
        }

        impl $type_name {
            pub fn new() -> Result<Self, BackfillExecutionError> {
                Ok(Self {
                    descriptor: backfill_support::descriptor(STRATEGY_KEY, $name, $description)?,
                })
            }
        }

        #[async_trait]
        impl BackfillWorkerStrategy for $type_name {
            fn descriptor(&self) -> &StrategyDescriptor {
                &self.descriptor
            }

            fn validate_request(
                &self,
                request: &BackfillRequest,
            ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
                backfill_support::validate_request(&self.descriptor, request)
            }

            fn plan_shards(
                &self,
                request: &ValidatedBackfillRequest,
            ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
                backfill_support::$planner(request, self.descriptor.maximum_shards)
            }

            async fn execute_backfill(
                &self,
                context: BackfillContext,
                shard: BackfillShard,
            ) -> Result<BackfillOutcome, BackfillExecutionError> {
                backfill_support::execute_futures(
                    context,
                    shard,
                    STRATEGY_KEY,
                    KrakenDataset::$dataset,
                )
                .await
            }
        }

        #[cfg(test)]
        mod tests {
            use chrono::{TimeZone, Utc};
            use super::*;
            use crate::domain::StrategyCapability;

            #[test]
            fn strategy_contract_is_backfill_only_and_validates_identity() {
                let strategy = $type_name::new().unwrap();
                assert_eq!(strategy.descriptor().strategy_key.as_ref(), STRATEGY_KEY);
                assert_eq!(strategy.descriptor().capabilities, vec![StrategyCapability::Backfill]);
                let start = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
                let request = backfill_support::test_request(
                    STRATEGY_KEY,
                    start,
                    start + chrono::Duration::hours(1),
                );
                let validated = strategy.validate_request(&request).unwrap();
                assert_eq!(strategy.plan_shards(&validated).unwrap().len(), 1);
                let mismatched = backfill_support::test_request(
                    "another_strategy",
                    start,
                    start + chrono::Duration::hours(1),
                );
                assert!(strategy.validate_request(&mismatched).is_err());
            }
        }
    };
}

mod aggressor_differential_backfill;
mod archive_client;
mod backfill_support;
mod cvd_backfill;
mod fee_schedules_backfill;
mod funding_rates_backfill;
mod future_basis_backfill;
mod instruments_backfill;
mod lake;
mod liquidation_volume_backfill;
mod liquidity_backfill;
mod mark_candles_backfill;
mod open_interest_backfill;
mod slippage_backfill;
mod spot_candles_backfill;
mod spot_support;
mod spot_trade_prints_one_second_ohlcv_backfill;
mod spreads_backfill;
mod trade_candles_backfill;
mod trade_count_backfill;
mod trade_volume_backfill;
mod types;

pub use aggressor_differential_backfill::KrakenAggressorDifferentialBackfill;
pub use cvd_backfill::KrakenCvdBackfill;
pub use fee_schedules_backfill::KrakenFeeSchedulesBackfill;
pub use funding_rates_backfill::KrakenFundingRatesBackfill;
pub use future_basis_backfill::KrakenFutureBasisBackfill;
pub use instruments_backfill::KrakenInstrumentsBackfill;
pub use liquidation_volume_backfill::KrakenLiquidationVolumeBackfill;
pub use liquidity_backfill::KrakenLiquidityBackfill;
pub use mark_candles_backfill::KrakenMarkCandlesBackfill;
pub use open_interest_backfill::KrakenOpenInterestBackfill;
pub use slippage_backfill::KrakenSlippageBackfill;
pub use spot_candles_backfill::KrakenSpotCandlesBackfill;
pub use spot_trade_prints_one_second_ohlcv_backfill::KrakenSpotBtcusdTradePrintsOneSecondOhlcvBackfill;
pub use spreads_backfill::KrakenSpreadsBackfill;
pub use trade_candles_backfill::KrakenTradeCandlesBackfill;
pub use trade_count_backfill::KrakenTradeCountBackfill;
pub use trade_volume_backfill::KrakenTradeVolumeBackfill;
