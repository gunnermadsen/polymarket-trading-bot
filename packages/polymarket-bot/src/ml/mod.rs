//! Dependency-light, execution-isolated foundations for shadow ML scoring.
//!
//! This module intentionally has no dependency on trading or execution types. It consumes an
//! immutable feature vector and produces a shadow prediction that can only be persisted or
//! evaluated out of band.

mod artifact;
mod features;
mod inference;
mod shadow;

pub use artifact::{
    canonical_artifact, ArtifactError, LinearFeatureTransform, LinearLogitArtifact, MlTask,
    PriorKind,
};
pub use features::{
    canonical_feature_schema, canonical_feature_vector, sha256_hex, FeatureError, MlFeature,
    MlFeatureVector,
};
pub use inference::{score_linear_logit, InferenceError, LinearLogitScore};
pub use shadow::{ShadowPrediction, ShadowScorer};
