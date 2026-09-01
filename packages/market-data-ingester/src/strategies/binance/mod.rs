//! Binance spot and USD-M futures factual ingestion strategies.

mod aggregate_trades;
mod aggregate_trades_backfill;
mod futures_open_interest;
mod one_second_ohlcv;
mod spot_l2_snapshots;

pub use aggregate_trades::BinanceSpotAggregateTradesFactory;
pub use aggregate_trades_backfill::BinanceSpotAggregateTradesBackfill;
pub use futures_open_interest::BinanceFuturesOpenInterestFactory;
pub use one_second_ohlcv::BinanceSpotOneSecondOhlcvFactory;
pub use spot_l2_snapshots::BinanceSpotL2SnapshotFactory;
