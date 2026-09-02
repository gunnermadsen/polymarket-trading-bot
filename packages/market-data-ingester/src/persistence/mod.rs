//! TimescaleDB access for ingester-owned state and facts.

mod aggregate_trades;
mod artifacts;
mod backfills;
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
pub use gaps::{GapDetection, GapPersistenceError, GapRepository, NewDataGap};
pub use profiles::{ProfileRepository, ProfileWriteError, StrategyDegradation, StrategyProgress};
