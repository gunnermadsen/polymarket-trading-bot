//! Provider-specific ingestion strategies.

use std::sync::Arc;

use crate::runtime::{StrategyFactory, StrategyFactoryError, StrategyRegistry};

pub mod binance;
pub mod chainlink;
pub mod polygon;

pub fn registry() -> Result<StrategyRegistry, StrategyFactoryError> {
    let factories: Vec<Arc<dyn StrategyFactory>> = vec![
        Arc::new(binance::BinanceSpotAggregateTradesFactory),
        Arc::new(binance::BinanceSpotOneSecondOhlcvFactory),
        Arc::new(binance::BinanceSpotL2SnapshotFactory),
        Arc::new(binance::BinanceFuturesOpenInterestFactory),
        Arc::new(chainlink::ChainlinkBtcusdReferencePriceFactory),
        Arc::new(chainlink::ChainlinkBtcusdOneMinuteOhlcFactory),
        Arc::new(polygon::PolygonChainlinkBtcusdOracleFactory),
    ];
    StrategyRegistry::from_factories(factories)
}
