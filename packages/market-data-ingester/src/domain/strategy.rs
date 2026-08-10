use std::{fmt, str::FromStr};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IngesterStrategyKey {
    BinanceSpotBtcusdtAggregateTrades,
    BinanceSpotBtcusdtOneSecondOhlcv,
    BinanceSpotBtcusdtL2Snapshots,
    BinanceFuturesBtcusdtOpenInterest,
    ChainlinkBtcusdReferencePrice,
    ChainlinkBtcusdOneMinuteOhlc,
    PolygonChainlinkBtcusdOracle,
    PolymarketBtcFiveMinuteMarketContracts,
    PolymarketBtcFiveMinuteOrderbooks,
    PolymarketBtcFiveMinuteResolutions,
}

impl IngesterStrategyKey {
    pub const ALL: [Self; 10] = [
        Self::BinanceSpotBtcusdtAggregateTrades,
        Self::BinanceSpotBtcusdtOneSecondOhlcv,
        Self::BinanceSpotBtcusdtL2Snapshots,
        Self::BinanceFuturesBtcusdtOpenInterest,
        Self::ChainlinkBtcusdReferencePrice,
        Self::ChainlinkBtcusdOneMinuteOhlc,
        Self::PolygonChainlinkBtcusdOracle,
        Self::PolymarketBtcFiveMinuteMarketContracts,
        Self::PolymarketBtcFiveMinuteOrderbooks,
        Self::PolymarketBtcFiveMinuteResolutions,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BinanceSpotBtcusdtAggregateTrades => "binance_spot_btcusdt_aggregate_trades",
            Self::BinanceSpotBtcusdtOneSecondOhlcv => "binance_spot_btcusdt_one_second_ohlcv",
            Self::BinanceSpotBtcusdtL2Snapshots => "binance_spot_btcusdt_l2_snapshots",
            Self::BinanceFuturesBtcusdtOpenInterest => "binance_futures_btcusdt_open_interest",
            Self::ChainlinkBtcusdReferencePrice => "chainlink_btcusd_reference_price",
            Self::ChainlinkBtcusdOneMinuteOhlc => "chainlink_btcusd_one_minute_ohlc",
            Self::PolygonChainlinkBtcusdOracle => "polygon_chainlink_btcusd_oracle",
            Self::PolymarketBtcFiveMinuteMarketContracts => {
                "polymarket_btc_five_minute_market_contracts"
            }
            Self::PolymarketBtcFiveMinuteOrderbooks => "polymarket_btc_five_minute_orderbooks",
            Self::PolymarketBtcFiveMinuteResolutions => "polymarket_btc_five_minute_resolutions",
        }
    }
}

impl fmt::Display for IngesterStrategyKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for IngesterStrategyKey {
    type Err = UnknownStrategyKey;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|key| key.as_str() == value)
            .ok_or_else(|| UnknownStrategyKey(value.to_owned()))
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("unknown ingester strategy key {0}")]
pub struct UnknownStrategyKey(String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyErrorKind {
    TransientSource,
    TransientDatabase,
    InvalidConfiguration,
    Integrity,
    LeaseLost,
    Shutdown,
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct StrategyError {
    pub kind: StrategyErrorKind,
    pub code: &'static str,
    pub message: String,
}

impl StrategyError {
    pub fn new(kind: StrategyErrorKind, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind,
            code,
            message: message.into(),
        }
    }
}

#[async_trait]
pub trait IngesterStrategy: Send + Sync {
    fn key(&self) -> IngesterStrategyKey;

    async fn run(&self, shutdown: CancellationToken) -> Result<(), StrategyError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_strategy_key_round_trips() {
        for expected in IngesterStrategyKey::ALL {
            assert_eq!(expected.as_str().parse(), Ok(expected));
        }
    }
}
