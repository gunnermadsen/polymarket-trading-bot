//! Provider-specific ingestion strategies.

use std::sync::Arc;

use crate::{
    domain::BackfillWorkerStrategy,
    runtime::{StrategyFactory, StrategyFactoryError, StrategyRegistry},
};

mod backfill_support;
pub mod binance;
pub mod chainlink;
pub mod pmdata;
pub mod polygon;
pub mod polymarket;
pub mod weather;

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
    let backfills: Vec<Arc<dyn BackfillWorkerStrategy>> = vec![
        Arc::new(
            binance::BinanceSpotAggregateTradesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            binance::BinanceFuturesFiveMinuteOpenInterestBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            binance::BinanceFuturesL2OneSecondFeaturesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            binance::BinanceSpotL2OneSecondFeaturesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            binance::BinanceSpotOneSecondOhlcvBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            weather::GoesAbiKlgaFeaturesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            weather::HrrrEnvironmentFeaturesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            chainlink::ChainlinkBtcusdReferenceTicksBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            chainlink::ChainlinkBtcusdOneMinuteCandlesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            polygon::PolygonChainlinkBtcusdOracleRoundsBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            pmdata::PmdataChainlinkBtcusdRefpriceBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            pmdata::PmdataChainlinkBtcusdTwap30sBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            pmdata::PmdataChainlinkBtcusdTwap60sBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            polymarket::PolymarketBtcMarketContractsBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            polymarket::PolymarketBtcResolutionsBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            polymarket::PolymarketBtcOrderbookEventsBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            polymarket::PolymarketBtcExecutionSnapshotsBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
    ];
    StrategyRegistry::from_factories(factories)?.with_backfills(backfills)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn realtime_and_backfill_strategy_identities_are_disjoint() {
        let registry = registry().unwrap();
        let realtime = registry
            .keys()
            .map(|key| key.as_str().to_owned())
            .collect::<BTreeSet<_>>();
        let backfills = registry
            .backfills()
            .map(|strategy| strategy.descriptor().strategy_key.to_string())
            .collect::<BTreeSet<_>>();
        assert!(realtime.is_disjoint(&backfills));
        for expected in [
            "binance_spot_btcusdt_aggregate_trades_backfill",
            "binance_futures_btcusdt_five_minute_open_interest_backfill",
            "binance_futures_btcusdt_l2_one_second_features_backfill",
            "binance_spot_btcusdt_l2_one_second_features_backfill",
            "binance_spot_btcusdt_one_second_ohlcv_backfill",
            "chainlink_btcusd_reference_ticks_backfill",
            "chainlink_btcusd_one_minute_candles_backfill",
            "polygon_chainlink_btcusd_oracle_rounds_backfill",
            "pmdata_chainlink_btcusd_refprice_backfill",
            "pmdata_chainlink_btcusd_twap_30s_backfill",
            "pmdata_chainlink_btcusd_twap_60s_backfill",
        ] {
            assert!(backfills.contains(expected), "missing {expected}");
        }
    }
}
