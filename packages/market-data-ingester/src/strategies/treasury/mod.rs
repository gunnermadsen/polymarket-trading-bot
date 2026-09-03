//! U.S. Treasury historical ingestion strategies.

macro_rules! define_treasury_strategy {
    ($type_name:ident, $key:literal, $name:literal, $description:literal, $dataset:ident) => {
        use crate::{
            domain::{
                BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest,
                BackfillShard, BackfillWorkerStrategy, StrategyDescriptor,
                ValidatedBackfillRequest,
            },
            strategies::economic::{backfill_support, types::EconomicDataset},
        };
        use async_trait::async_trait;

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
                backfill_support::validate_request(
                    &self.descriptor,
                    request,
                    EconomicDataset::$dataset,
                )
            }

            fn plan_shards(
                &self,
                request: &ValidatedBackfillRequest,
            ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
                backfill_support::plan_year_shards(
                    request,
                    self.descriptor.maximum_shards,
                    EconomicDataset::$dataset,
                )
            }

            async fn execute_backfill(
                &self,
                context: BackfillContext,
                shard: BackfillShard,
            ) -> Result<BackfillOutcome, BackfillExecutionError> {
                backfill_support::execute(context, shard, STRATEGY_KEY, EconomicDataset::$dataset)
                    .await
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use crate::domain::StrategyCapability;
            use chrono::{TimeZone, Utc};

            #[test]
            fn strategy_is_backfill_only_and_has_stable_identity() {
                let strategy = $type_name::new().unwrap();
                assert_eq!(strategy.descriptor().strategy_key.as_ref(), STRATEGY_KEY);
                assert_eq!(
                    strategy.descriptor().capabilities,
                    vec![StrategyCapability::Backfill]
                );
                let start = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
                let request = backfill_support::test_request(
                    STRATEGY_KEY,
                    start,
                    start + chrono::Duration::days(30),
                    EconomicDataset::$dataset,
                );
                let validated = strategy.validate_request(&request).unwrap();
                assert_eq!(strategy.plan_shards(&validated).unwrap().len(), 1);
            }
        }
    };
}

mod us_treasury_auctions_backfill;
mod us_treasury_debt_to_penny_backfill;
mod us_treasury_deposits_withdrawals_backfill;
mod us_treasury_operating_cash_balance_backfill;

pub use us_treasury_auctions_backfill::UsTreasuryAuctionsBackfill;
pub use us_treasury_debt_to_penny_backfill::UsTreasuryDebtToPennyBackfill;
pub use us_treasury_deposits_withdrawals_backfill::UsTreasuryDepositsWithdrawalsBackfill;
pub use us_treasury_operating_cash_balance_backfill::UsTreasuryOperatingCashBalanceBackfill;
