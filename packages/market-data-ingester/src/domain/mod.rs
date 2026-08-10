//! Source-neutral ingestion contracts.

mod artifact;
mod gap;
mod profile;
mod strategy;
mod timestamps;
mod watermark;

pub use artifact::{ArtifactStatus, CaptureArtifact};
pub use gap::{DataGap, GapStatus};
pub use profile::{DesiredState, HealthStatus, IngesterProfile, ObservedState};
pub use strategy::{IngesterStrategy, IngesterStrategyKey, StrategyError, StrategyErrorKind};
pub use timestamps::FactualTimestamps;
pub use watermark::StrategyWatermarks;
