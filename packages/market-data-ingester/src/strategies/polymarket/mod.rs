//! Polymarket public market-data ingestion strategies.

mod backfill_types;
mod backfills;
#[cfg(test)]
mod backfills_tests;
pub mod chainlink_twap;
mod execution_snapshots;
pub mod market_contracts;
pub mod orderbook_snapshots;
mod pmxt_archive;
pub mod resolutions;

pub use backfills::PolymarketBtcBackfill;
pub use chainlink_twap::PolymarketChainlinkBtcusdTwapFactory;
pub use market_contracts::PolymarketBtcFiveMinuteMarketContractsFactory;
pub use orderbook_snapshots::PolymarketBtcFiveMinuteOrderbooksFactory;
pub use resolutions::PolymarketBtcFiveMinuteResolutionsFactory;
