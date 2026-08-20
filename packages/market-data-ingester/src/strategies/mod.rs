//! Provider-specific ingestion strategies.

use std::sync::Arc;

use crate::runtime::{StrategyFactory, StrategyFactoryError, StrategyRegistry};

pub mod binance;
pub mod chainlink;
pub mod polygon;
pub mod polymarket;

pub fn registry() -> Result<StrategyRegistry, StrategyFactoryError> {
    let factories: Vec<Arc<dyn StrategyFactory>> = vec![
        Arc::new(binance::BinanceSpotAggregateTradesFactory),
        Arc::new(binance::BinanceSpotOneSecondOhlcvFactory),
        Arc::new(binance::BinanceSpotL2SnapshotFactory),
        Arc::new(binance::BinanceFuturesOpenInterestFactory),
        Arc::new(chainlink::ChainlinkBtcusdReferencePriceFactory),
        Arc::new(chainlink::ChainlinkBtcusdOneMinuteOhlcFactory),
        Arc::new(polygon::PolygonChainlinkBtcusdOracleFactory),
        Arc::new(polymarket::PolymarketBtcFiveMinuteMarketContractsFactory),
        Arc::new(polymarket::PolymarketBtcFiveMinuteOrderbooksFactory),
        Arc::new(polymarket::PolymarketBtcFiveMinuteResolutionsFactory),
        Arc::new(polymarket::PolymarketChainlinkBtcusdTwapFactory),
    ];
    StrategyRegistry::from_factories(factories)
}
