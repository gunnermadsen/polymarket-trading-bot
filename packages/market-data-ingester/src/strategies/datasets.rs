use crate::domain::DatasetKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrategyDatasetBinding {
    pub strategy_key: &'static str,
    pub dataset: DatasetKey,
}

/// Authoritative strategy-to-dataset mapping. Realtime and backfill are
/// collection modes; neither is allowed to define a second data product.
pub const STRATEGY_DATASETS: &[StrategyDatasetBinding] = &[
    binding(
        "binance_spot_btcusdt_aggregate_trades",
        DatasetKey::BinanceSpotAggregateTrades,
    ),
    binding(
        "binance_spot_btcusdt_aggregate_trades_backfill",
        DatasetKey::BinanceSpotAggregateTrades,
    ),
    binding(
        "binance_spot_btcusdt_one_second_ohlcv",
        DatasetKey::BinanceSpotOneSecondOhlcv,
    ),
    binding(
        "binance_spot_btcusdt_one_second_ohlcv_backfill",
        DatasetKey::BinanceSpotOneSecondOhlcv,
    ),
    binding(
        "binance_futures_btcusdt_open_interest",
        DatasetKey::BinanceFuturesOpenInterest,
    ),
    binding(
        "binance_futures_btcusdt_five_minute_open_interest_backfill",
        DatasetKey::BinanceFuturesOpenInterest,
    ),
    binding(
        "binance_spot_btcusdt_l2_snapshots",
        DatasetKey::BinanceSpotL2Snapshots,
    ),
    binding(
        "binance_spot_btcusdt_l2_one_second_features_backfill",
        DatasetKey::BinanceSpotL2OneSecondFeatures,
    ),
    binding(
        "coinapi_binance_spot_btcusdt_l2_one_second_features_backfill",
        DatasetKey::BinanceSpotL2OneSecondFeatures,
    ),
    binding(
        "binance_futures_btcusdt_l2_one_second_features_backfill",
        DatasetKey::BinanceFuturesL2OneSecondFeatures,
    ),
    binding(
        "chainlink_btcusd_reference_price",
        DatasetKey::ChainlinkBtcusdReferencePrices,
    ),
    binding(
        "chainlink_btcusd_reference_ticks_backfill",
        DatasetKey::ChainlinkBtcusdReferencePrices,
    ),
    binding(
        "pmdata_chainlink_btcusd_refprice_backfill",
        DatasetKey::ChainlinkBtcusdReferencePrices,
    ),
    binding(
        "chainlink_btcusd_one_minute_ohlc",
        DatasetKey::ChainlinkBtcusdOneMinuteCandles,
    ),
    binding(
        "chainlink_btcusd_one_minute_candles_backfill",
        DatasetKey::ChainlinkBtcusdOneMinuteCandles,
    ),
    binding(
        "polygon_chainlink_btcusd_oracle",
        DatasetKey::PolygonChainlinkBtcusdOracleRounds,
    ),
    binding(
        "polygon_chainlink_btcusd_oracle_rounds_backfill",
        DatasetKey::PolygonChainlinkBtcusdOracleRounds,
    ),
    binding(
        "pmdata_chainlink_btcusd_twap_30s_backfill",
        DatasetKey::PmdataChainlinkBtcusdTwap,
    ),
    binding(
        "pmdata_chainlink_btcusd_twap_60s_backfill",
        DatasetKey::PmdataChainlinkBtcusdTwap,
    ),
    binding(
        "polymarket_chainlink_btcusd_twap",
        DatasetKey::PolymarketChainlinkBtcusdTwap,
    ),
    binding(
        "polymarket_btc_five_minute_orderbooks",
        DatasetKey::PolymarketBtcFiveMinuteOrderbookSnapshots,
    ),
];

const fn binding(strategy_key: &'static str, dataset: DatasetKey) -> StrategyDatasetBinding {
    StrategyDatasetBinding {
        strategy_key,
        dataset,
    }
}

pub fn dataset_for_strategy(strategy_key: &str) -> Option<DatasetKey> {
    STRATEGY_DATASETS
        .iter()
        .find(|binding| binding.strategy_key == strategy_key)
        .map(|binding| binding.dataset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn each_strategy_has_exactly_one_dataset() {
        let keys = STRATEGY_DATASETS
            .iter()
            .map(|binding| binding.strategy_key)
            .collect::<BTreeSet<_>>();
        assert_eq!(keys.len(), STRATEGY_DATASETS.len());
        assert!(STRATEGY_DATASETS
            .iter()
            .all(|binding| crate::domain::contract_for(binding.dataset).is_some()));
    }

    #[test]
    fn aligned_collection_modes_resolve_to_one_dataset() {
        for (realtime, backfill) in [
            (
                "binance_spot_btcusdt_aggregate_trades",
                "binance_spot_btcusdt_aggregate_trades_backfill",
            ),
            (
                "binance_spot_btcusdt_one_second_ohlcv",
                "binance_spot_btcusdt_one_second_ohlcv_backfill",
            ),
            (
                "binance_futures_btcusdt_open_interest",
                "binance_futures_btcusdt_five_minute_open_interest_backfill",
            ),
            (
                "chainlink_btcusd_reference_price",
                "chainlink_btcusd_reference_ticks_backfill",
            ),
            (
                "chainlink_btcusd_reference_price",
                "pmdata_chainlink_btcusd_refprice_backfill",
            ),
            (
                "chainlink_btcusd_one_minute_ohlc",
                "chainlink_btcusd_one_minute_candles_backfill",
            ),
            (
                "polygon_chainlink_btcusd_oracle",
                "polygon_chainlink_btcusd_oracle_rounds_backfill",
            ),
        ] {
            assert_eq!(
                dataset_for_strategy(realtime),
                dataset_for_strategy(backfill)
            );
        }
    }
}
