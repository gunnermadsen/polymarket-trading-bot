//! Source-neutral ingestion contracts.

mod artifact;
mod backfill;
mod dataset;
mod drain;
mod gap;
mod profile;
mod strategy;
mod timestamps;
mod watermark;
mod worker_allocation;

pub use artifact::{ArtifactStatus, CaptureArtifact};
pub use backfill::{
    BackfillContext, BackfillExecutionError, BackfillFailureKind, BackfillOutcome, BackfillRange,
    BackfillRequest, BackfillShard, BackfillWorkerStrategy, ExecutionSelector, StrategyCapability,
    StrategyDescriptor, ValidatedBackfillRequest,
};
pub use dataset::*;
pub use drain::{
    DrainContext, DrainDescriptor, DrainExecutionError, DrainMode, DrainOutcome, DrainRequest,
    DrainWorkerStrategy,
};
pub use gap::{DataGap, GapStatus};
pub use profile::{DesiredState, HealthStatus, IngesterProfile, ObservedState};
pub use strategy::{IngesterStrategyKey, RealtimeWorkerStrategy, StrategyError, StrategyErrorKind};
pub use timestamps::FactualTimestamps;
pub use watermark::StrategyWatermarks;
pub use worker_allocation::{
    admits_backfill, admits_realtime, backfill_profile, realtime_profile, IsolationClass,
    WorkloadProfile, ALLOCATION_CONTRACT_VERSION, DEFAULT_REALTIME_SLOT_LIMIT,
    DEFAULT_WORKER_CAPACITY_UNITS,
};
