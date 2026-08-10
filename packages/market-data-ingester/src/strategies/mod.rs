//! Provider-specific ingestion strategies.

use std::sync::Arc;

use crate::runtime::{StrategyFactory, StrategyFactoryError, StrategyRegistry};

pub mod binance;

pub fn registry() -> Result<StrategyRegistry, StrategyFactoryError> {
    let factories: Vec<Arc<dyn StrategyFactory>> = vec![
        Arc::new(binance::BinanceSpotAggregateTradesFactory),
        Arc::new(binance::BinanceSpotOneSecondOhlcvFactory),
        Arc::new(binance::BinanceSpotL2SnapshotFactory),
        Arc::new(binance::BinanceFuturesOpenInterestFactory),
    ];
    StrategyRegistry::from_factories(factories)
}
