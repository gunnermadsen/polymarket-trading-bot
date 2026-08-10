//! TimescaleDB access for ingester-owned state and facts.

mod artifacts;
mod gaps;
mod profiles;

pub use artifacts::{
    ArtifactBatch, ArtifactPersistenceError, ArtifactRepository, NewCaptureArtifact,
};
pub use gaps::{GapDetection, GapPersistenceError, GapRepository, NewDataGap};
pub use profiles::{ProfileRepository, ProfileWriteError, StrategyDegradation, StrategyProgress};
