//! Provider-specific ingestion strategies.

use std::sync::Arc;

use crate::{
    domain::{BackfillWorkerStrategy, DrainWorkerStrategy},
    runtime::{StrategyFactory, StrategyFactoryError, StrategyRegistry},
};

mod backfill_support;
pub mod binance;
pub mod chainlink;
mod datasets;
pub mod drains;
pub mod economic;
pub mod kraken;
pub mod pmdata;
pub mod polygon;
pub mod polymarket;
mod raw_archive;
pub mod temperature;
pub mod treasury;
pub mod weather;

pub use datasets::{dataset_for_strategy, StrategyDatasetBinding, STRATEGY_DATASETS};

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
            binance::CoinapiBinanceSpotL2OneSecondFeaturesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            binance::BinanceSpotOneSecondOhlcvBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            weather::GoesAbiSourceArchivesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            weather::HrrrSurfaceArchivesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            weather::AsosOneMinuteArchivesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            weather::AsosMetarArchivesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            temperature::PolymarketTemperatureMarketArchivesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            temperature::PolymarketTemperaturePriceArchivesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            temperature::PmxtPolymarketOrderbookArchivesBackfill::new()
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
            kraken::KrakenInstrumentsBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenFeeSchedulesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenTradeCandlesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenMarkCandlesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenSpotCandlesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenOpenInterestBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenFutureBasisBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenAggressorDifferentialBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenTradeVolumeBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenTradeCountBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenCvdBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenLiquidationVolumeBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenSpreadsBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenLiquidityBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenSlippageBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenFundingRatesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            kraken::KrakenSpotBtcusdTradePrintsOneSecondOhlcvBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            economic::FredEconomicSeriesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            economic::NewYorkFedReferenceRatesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            economic::NewYorkFedSomaHoldingsBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            economic::CftcLegacyFuturesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            economic::CftcTradersFinancialFuturesBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            treasury::UsTreasuryAuctionsBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            treasury::UsTreasuryDebtToPennyBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            treasury::UsTreasuryDepositsWithdrawalsBackfill::new()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            treasury::UsTreasuryOperatingCashBalanceBackfill::new()
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
    let drains: Vec<Arc<dyn DrainWorkerStrategy>> = vec![
        Arc::new(
            drains::BinanceAggregateTradesDrain::from_environment()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
        Arc::new(
            drains::PolymarketOrderbooksDrain::from_environment()
                .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?,
        ),
    ];
    StrategyRegistry::from_factories(factories)?
        .with_backfills(backfills)?
        .with_drains(drains)
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
            "kraken_instruments_backfill",
            "kraken_fee_schedules_backfill",
            "kraken_trade_candles_backfill",
            "kraken_mark_candles_backfill",
            "kraken_spot_candles_backfill",
            "kraken_open_interest_backfill",
            "kraken_future_basis_backfill",
            "kraken_aggressor_differential_backfill",
            "kraken_trade_volume_backfill",
            "kraken_trade_count_backfill",
            "kraken_cvd_backfill",
            "kraken_liquidation_volume_backfill",
            "kraken_spreads_backfill",
            "kraken_liquidity_backfill",
            "kraken_slippage_backfill",
            "kraken_funding_rates_backfill",
            "kraken_spot_btcusd_trade_prints_one_second_ohlcv_backfill",
            "fred_economic_series_backfill",
            "new_york_fed_reference_rates_backfill",
            "new_york_fed_soma_holdings_backfill",
            "cftc_legacy_futures_backfill",
            "cftc_traders_financial_futures_backfill",
            "us_treasury_auctions_backfill",
            "us_treasury_debt_to_penny_backfill",
            "us_treasury_deposits_withdrawals_backfill",
            "us_treasury_operating_cash_balance_backfill",
        ] {
            assert!(backfills.contains(expected), "missing {expected}");
        }
    }

    #[test]
    fn drain_is_opt_in_and_does_not_extend_backfill_contracts() {
        let registry = registry().unwrap();
        let drains = registry
            .drains()
            .map(|strategy| strategy.descriptor().strategy_key.as_ref())
            .collect::<Vec<_>>();
        assert_eq!(
            drains,
            vec![
                "binance_spot_btcusdt_aggregate_trades",
                "polymarket_btc_five_minute_orderbooks",
            ]
        );
        assert!(registry
            .backfill("binance_spot_btcusdt_aggregate_trades")
            .is_none());
        assert!(registry
            .backfill("binance_spot_btcusdt_aggregate_trades_backfill")
            .is_some());
    }

    #[test]
    fn every_dataset_binding_names_a_registered_strategy() {
        let registry = registry().unwrap();
        let realtime = registry
            .keys()
            .map(|key| key.as_str().to_owned())
            .collect::<BTreeSet<_>>();
        let backfills = registry
            .backfills()
            .map(|strategy| strategy.descriptor().strategy_key.to_string())
            .collect::<BTreeSet<_>>();
        for binding in STRATEGY_DATASETS {
            assert!(
                realtime.contains(binding.strategy_key) || backfills.contains(binding.strategy_key),
                "dataset binding names an unregistered strategy: {}",
                binding.strategy_key
            );
        }
    }
}
