use serde::{Deserialize, Serialize};

use super::{score_linear_logit, InferenceError, LinearLogitArtifact, MlFeatureVector, MlTask};

/// Persistable output of an execution-isolated shadow score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowPrediction {
    snapshot_id: String,
    market_id: String,
    feature_as_of_ms: i64,
    scored_at_ms: i64,
    task: MlTask,
    model_version: String,
    prior_probability: f64,
    probability: f64,
    residual_logit: f64,
    feature_vector_sha256: String,
    artifact_sha256: String,
}

impl ShadowPrediction {
    pub fn snapshot_id(&self) -> &str {
        &self.snapshot_id
    }

    pub fn market_id(&self) -> &str {
        &self.market_id
    }

    pub fn task(&self) -> MlTask {
        self.task
    }

    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    pub fn prior_probability(&self) -> f64 {
        self.prior_probability
    }

    pub fn probability(&self) -> f64 {
        self.probability
    }

    pub fn residual_logit(&self) -> f64 {
        self.residual_logit
    }

    pub fn feature_vector_sha256(&self) -> &str {
        &self.feature_vector_sha256
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }
}

#[derive(Debug, Clone)]
pub struct ShadowScorer {
    artifact: LinearLogitArtifact,
}

impl ShadowScorer {
    pub fn new(artifact: LinearLogitArtifact) -> Result<Self, InferenceError> {
        artifact.validate().map_err(InferenceError::Artifact)?;
        Ok(Self { artifact })
    }

    pub fn score(
        &self,
        vector: &MlFeatureVector,
        scored_at_ms: i64,
    ) -> Result<ShadowPrediction, InferenceError> {
        let score = score_linear_logit(&self.artifact, vector)?;
        Ok(ShadowPrediction {
            snapshot_id: vector.snapshot_id().to_string(),
            market_id: vector.market_id().to_string(),
            feature_as_of_ms: vector.feature_as_of_ms(),
            scored_at_ms,
            task: self.artifact.task(),
            model_version: self.artifact.model_version().to_string(),
            prior_probability: score.prior_probability(),
            probability: score.probability(),
            residual_logit: score.residual_logit(),
            feature_vector_sha256: vector.vector_sha256().to_string(),
            artifact_sha256: self.artifact.artifact_sha256().to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{MlFeature, MlTask};

    use super::*;

    #[test]
    fn ml_b_schema_canary_produces_shadow_only_prediction() {
        let artifact = LinearLogitArtifact::schema_canary_v0(
            MlTask::FokFillProbability,
            "execution-v1",
            ["queue_ahead"],
        )
        .unwrap();
        let scorer = ShadowScorer::new(artifact).unwrap();
        let vector = MlFeatureVector::new(
            "snapshot-1",
            "market-1",
            1_010,
            1_010,
            "execution-v1",
            0.25,
            vec![MlFeature::new("queue_ahead", 100.0, 999, 1_005).unwrap()],
        )
        .unwrap();
        let prediction = scorer.score(&vector, 1_020).unwrap();
        assert_eq!(prediction.task(), MlTask::FokFillProbability);
        assert!((prediction.probability() - 0.25).abs() < 1.0e-12);
    }
}
