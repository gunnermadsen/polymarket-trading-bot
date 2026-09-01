//! TimescaleDB access for ingester-owned state and facts.

mod artifacts;
mod backfills;
mod gaps;
mod profiles;

pub use artifacts::{
    ArtifactBatch, ArtifactPersistenceError, ArtifactRepository, NewCaptureArtifact,
};
pub use backfills::{
    BackfillJobEvent, BackfillJobRecord, BackfillRepository, ClaimedBackfillJob, WorkerRecord,
    WorkerRegistration,
};
pub use gaps::{GapDetection, GapPersistenceError, GapRepository, NewDataGap};
pub use profiles::{ProfileRepository, ProfileWriteError, StrategyDegradation, StrategyProgress};
