//! TimescaleDB access for ingester-owned state and facts.

mod aggregate_trades;
mod artifacts;
mod backfills;
mod chainlink_reference_prices;
mod drains;
mod gaps;
mod profiles;

pub use aggregate_trades::{insert_binance_aggregate_trades, BinanceAggregateTradeWrite};
pub use artifacts::{
    ArtifactBatch, ArtifactPersistenceError, ArtifactRepository, NewCaptureArtifact,
};
pub use backfills::{
    BackfillJobEvent, BackfillJobRecord, BackfillRepository, ClaimedBackfillJob, WorkerRecord,
    WorkerRegistration,
};
pub use chainlink_reference_prices::{
    insert_chainlink_reference_prices, insert_pmdata_chainlink_reference_prices,
    ChainlinkReferencePriceWrite, ReferencePriceArtifact,
};
pub use drains::{ClaimedDrainJob, DrainJobRecord, DrainRepository};
pub use gaps::{GapDetection, GapPersistenceError, GapRepository, NewDataGap};
pub use profiles::{ProfileRepository, ProfileWriteError, StrategyDegradation, StrategyProgress};
