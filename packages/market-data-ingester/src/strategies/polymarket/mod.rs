//! Polymarket public market-data ingestion strategies.

pub mod chainlink_twap;
pub mod market_contracts;
pub mod orderbook_snapshots;
pub mod resolutions;

pub use chainlink_twap::PolymarketChainlinkBtcusdTwapFactory;
pub use market_contracts::PolymarketBtcFiveMinuteMarketContractsFactory;
pub use orderbook_snapshots::PolymarketBtcFiveMinuteOrderbooksFactory;
pub use resolutions::PolymarketBtcFiveMinuteResolutionsFactory;
