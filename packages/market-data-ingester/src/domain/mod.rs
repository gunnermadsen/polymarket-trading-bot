//! Source-neutral ingestion contracts.

mod artifact;
mod backfill;
mod gap;
mod profile;
mod strategy;
mod timestamps;
mod watermark;

pub use artifact::{ArtifactStatus, CaptureArtifact};
pub use backfill::{
    BackfillContext, BackfillExecutionError, BackfillFailureKind, BackfillOutcome, BackfillRequest,
    BackfillShard, BackfillWorkerStrategy, ExecutionSelector, StrategyCapability,
    StrategyDescriptor, ValidatedBackfillRequest,
};
pub use gap::{DataGap, GapStatus};
pub use profile::{DesiredState, HealthStatus, IngesterProfile, ObservedState};
pub use strategy::{IngesterStrategyKey, RealtimeWorkerStrategy, StrategyError, StrategyErrorKind};
pub use timestamps::FactualTimestamps;
pub use watermark::StrategyWatermarks;
