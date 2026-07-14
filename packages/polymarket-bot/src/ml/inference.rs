use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

use super::{
    artifact::ArtifactError, features::FeatureError, LinearLogitArtifact, MlFeatureVector,
};

const PROBABILITY_EPSILON: f64 = 1.0e-9;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearLogitScore {
    prior_probability: f64,
    probability: f64,
    residual_logit: f64,
}

impl LinearLogitScore {
    pub fn prior_probability(&self) -> f64 {
        self.prior_probability
    }

    pub fn probability(&self) -> f64 {
        self.probability
    }

    pub fn residual_logit(&self) -> f64 {
        self.residual_logit
    }
}

pub fn score_linear_logit(
    artifact: &LinearLogitArtifact,
    vector: &MlFeatureVector,
) -> Result<LinearLogitScore, InferenceError> {
    artifact.validate().map_err(InferenceError::Artifact)?;
    vector.validate_hashes().map_err(InferenceError::Feature)?;
    if artifact.feature_schema_version() != vector.schema_version()
        || artifact.feature_schema_sha256() != vector.schema_sha256()
    {
        return Err(InferenceError::FeatureSchemaMismatch {
            artifact_version: artifact.feature_schema_version().to_string(),
            vector_version: vector.schema_version().to_string(),
            artifact_hash: artifact.feature_schema_sha256().to_string(),
            vector_hash: vector.schema_sha256().to_string(),
        });
    }
    if artifact.features().len() != vector.features().len() {
        return Err(InferenceError::FeatureCountMismatch {
            expected: artifact.features().len(),
            actual: vector.features().len(),
        });
    }

    let mut residual_logit = artifact.intercept();
    for (transform, feature) in artifact.features().iter().zip(vector.features()) {
        if transform.name() != feature.name() {
            return Err(InferenceError::FeatureOrderMismatch {
                expected: transform.name().to_string(),
                actual: feature.name().to_string(),
            });
        }
        residual_logit +=
            transform.coefficient() * (feature.value() - transform.mean()) / transform.scale();
    }
    if !residual_logit.is_finite() {
        return Err(InferenceError::NonFiniteScore);
    }

    let prior = vector
        .prior_probability()
        .clamp(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON);
    let prior_logit = (prior / (1.0 - prior)).ln();
    let probability = stable_sigmoid(prior_logit + residual_logit);
    Ok(LinearLogitScore {
        prior_probability: vector.prior_probability(),
        probability,
        residual_logit,
    })
}

fn stable_sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[derive(Debug)]
pub enum InferenceError {
    Artifact(ArtifactError),
    Feature(FeatureError),
    FeatureSchemaMismatch {
        artifact_version: String,
        vector_version: String,
        artifact_hash: String,
        vector_hash: String,
    },
    FeatureCountMismatch {
        expected: usize,
        actual: usize,
    },
    FeatureOrderMismatch {
        expected: String,
        actual: String,
    },
    NonFiniteScore,
}

impl fmt::Display for InferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for InferenceError {}

#[cfg(test)]
mod tests {
    use super::super::{MlFeature, MlTask};

    use super::*;

    fn vector(prior: f64) -> MlFeatureVector {
        MlFeatureVector::new(
            "snapshot-1",
            "market-1",
            1_010,
            1_010,
            "features-v1",
            prior,
            vec![MlFeature::new("distance", 12.0, 1_000, 1_005).unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn schema_canary_reproduces_prior() {
        let artifact = LinearLogitArtifact::schema_canary_v0(
            MlTask::SettlementProbabilityResidual,
            "features-v1",
            ["distance"],
        )
        .unwrap();
        let score = score_linear_logit(&artifact, &vector(0.63)).unwrap();
        assert!((score.probability() - 0.63).abs() < 1.0e-12);
        assert_eq!(score.residual_logit(), 0.0);
    }

    #[test]
    fn mismatched_schema_is_rejected() {
        let artifact = LinearLogitArtifact::schema_canary_v0(
            MlTask::SettlementProbabilityResidual,
            "features-v2",
            ["distance"],
        )
        .unwrap();
        let error = score_linear_logit(&artifact, &vector(0.5)).unwrap_err();
        assert!(matches!(
            error,
            InferenceError::FeatureSchemaMismatch { .. }
        ));
    }
}
