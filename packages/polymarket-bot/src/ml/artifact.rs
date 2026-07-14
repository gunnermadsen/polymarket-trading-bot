use std::{collections::HashSet, error::Error, fmt};

use serde::{Deserialize, Serialize};

use super::features::{canonical_feature_schema, push_float, push_text, sha256_hex};

pub const LINEAR_LOGIT_ARTIFACT_VERSION: &str = "linear_logit_artifact_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlTask {
    SettlementProbabilityResidual,
    FokFillProbability,
    FokPostFillToxicityProbability,
}

impl MlTask {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SettlementProbabilityResidual => "settlement_probability_residual",
            Self::FokFillProbability => "fok_fill_probability",
            Self::FokPostFillToxicityProbability => "fok_post_fill_toxicity_probability",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorKind {
    ProvidedProbability,
}

impl PriorKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::ProvidedProbability => "provided_probability",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearFeatureTransform {
    name: String,
    mean: f64,
    scale: f64,
    coefficient: f64,
}

impl LinearFeatureTransform {
    pub fn new(
        name: impl Into<String>,
        mean: f64,
        scale: f64,
        coefficient: f64,
    ) -> Result<Self, ArtifactError> {
        let transform = Self {
            name: name.into(),
            mean,
            scale,
            coefficient,
        };
        transform.validate()?;
        Ok(transform)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn coefficient(&self) -> f64 {
        self.coefficient
    }

    fn validate(&self) -> Result<(), ArtifactError> {
        if self.name.trim().is_empty() {
            return Err(ArtifactError::EmptyFeatureName);
        }
        if !self.mean.is_finite() {
            return Err(ArtifactError::NonFiniteParameter(format!(
                "{}.mean",
                self.name
            )));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(ArtifactError::InvalidScale(self.name.clone()));
        }
        if !self.coefficient.is_finite() {
            return Err(ArtifactError::NonFiniteParameter(format!(
                "{}.coefficient",
                self.name
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearLogitArtifact {
    artifact_version: String,
    model_version: String,
    task: MlTask,
    prior_kind: PriorKind,
    feature_schema_version: String,
    feature_schema_sha256: String,
    dataset_manifest_sha256: Option<String>,
    trained_through_ms: Option<i64>,
    intercept: f64,
    features: Vec<LinearFeatureTransform>,
    artifact_sha256: String,
}

impl LinearLogitArtifact {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_version: impl Into<String>,
        task: MlTask,
        prior_kind: PriorKind,
        feature_schema_version: impl Into<String>,
        dataset_manifest_sha256: Option<String>,
        trained_through_ms: Option<i64>,
        intercept: f64,
        features: Vec<LinearFeatureTransform>,
    ) -> Result<Self, ArtifactError> {
        let feature_schema_version = feature_schema_version.into();
        let names = features.iter().map(|feature| feature.name());
        let feature_schema_sha256 =
            sha256_hex(&canonical_feature_schema(&feature_schema_version, names));
        let mut artifact = Self {
            artifact_version: LINEAR_LOGIT_ARTIFACT_VERSION.to_string(),
            model_version: model_version.into(),
            task,
            prior_kind,
            feature_schema_version,
            feature_schema_sha256,
            dataset_manifest_sha256,
            trained_through_ms,
            intercept,
            features,
            artifact_sha256: String::new(),
        };
        artifact.validate_content()?;
        artifact.artifact_sha256 = sha256_hex(&canonical_artifact(&artifact));
        Ok(artifact)
    }

    /// A pipeline-validation artifact. It contains no learned alpha and exactly reproduces the
    /// provided prior because every residual coefficient and the intercept are zero.
    pub fn schema_canary_v0<'a>(
        task: MlTask,
        feature_schema_version: impl Into<String>,
        feature_names: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, ArtifactError> {
        let features = feature_names
            .into_iter()
            .map(|name| LinearFeatureTransform::new(name, 0.0, 1.0, 0.0))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(
            "schema_canary_v0",
            task,
            PriorKind::ProvidedProbability,
            feature_schema_version,
            None,
            None,
            0.0,
            features,
        )
    }

    pub fn from_json(value: &str) -> Result<Self, ArtifactError> {
        let artifact: Self = serde_json::from_str(value)
            .map_err(|error| ArtifactError::InvalidJson(error.to_string()))?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn to_json(&self) -> Result<String, ArtifactError> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| ArtifactError::InvalidJson(error.to_string()))
    }

    pub fn validate(&self) -> Result<(), ArtifactError> {
        self.validate_content()?;
        let expected = sha256_hex(&canonical_artifact(self));
        if expected != self.artifact_sha256 {
            return Err(ArtifactError::ArtifactHashMismatch {
                expected,
                actual: self.artifact_sha256.clone(),
            });
        }
        Ok(())
    }

    fn validate_content(&self) -> Result<(), ArtifactError> {
        if self.artifact_version != LINEAR_LOGIT_ARTIFACT_VERSION {
            return Err(ArtifactError::UnsupportedArtifactVersion(
                self.artifact_version.clone(),
            ));
        }
        if self.model_version.trim().is_empty() {
            return Err(ArtifactError::EmptyModelVersion);
        }
        if self.feature_schema_version.trim().is_empty() {
            return Err(ArtifactError::EmptyFeatureSchemaVersion);
        }
        if !self.intercept.is_finite() {
            return Err(ArtifactError::NonFiniteParameter("intercept".to_string()));
        }
        if self.features.is_empty() {
            return Err(ArtifactError::EmptyFeatureList);
        }
        let mut names = HashSet::with_capacity(self.features.len());
        for feature in &self.features {
            feature.validate()?;
            if !names.insert(feature.name().to_string()) {
                return Err(ArtifactError::DuplicateFeatureName(
                    feature.name().to_string(),
                ));
            }
        }
        let schema = sha256_hex(&canonical_feature_schema(
            &self.feature_schema_version,
            self.features.iter().map(|feature| feature.name()),
        ));
        if schema != self.feature_schema_sha256 {
            return Err(ArtifactError::FeatureSchemaHashMismatch {
                expected: schema,
                actual: self.feature_schema_sha256.clone(),
            });
        }
        if let Some(hash) = &self.dataset_manifest_sha256 {
            if !is_sha256_hex(hash) {
                return Err(ArtifactError::InvalidDatasetManifestHash(hash.clone()));
            }
        }
        Ok(())
    }

    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    pub fn task(&self) -> MlTask {
        self.task
    }

    pub fn prior_kind(&self) -> PriorKind {
        self.prior_kind
    }

    pub fn feature_schema_version(&self) -> &str {
        &self.feature_schema_version
    }

    pub fn feature_schema_sha256(&self) -> &str {
        &self.feature_schema_sha256
    }

    pub fn dataset_manifest_sha256(&self) -> Option<&str> {
        self.dataset_manifest_sha256.as_deref()
    }

    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    pub fn features(&self) -> &[LinearFeatureTransform] {
        &self.features
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }
}

pub fn canonical_artifact(artifact: &LinearLogitArtifact) -> String {
    let mut output = String::from("linear_logit_artifact_canonical_v1\n");
    push_text(&mut output, "artifact_version", &artifact.artifact_version);
    push_text(&mut output, "model_version", &artifact.model_version);
    push_text(&mut output, "task", artifact.task.as_str());
    push_text(&mut output, "prior_kind", artifact.prior_kind.as_str());
    push_text(
        &mut output,
        "feature_schema_version",
        &artifact.feature_schema_version,
    );
    push_text(
        &mut output,
        "feature_schema_sha256",
        &artifact.feature_schema_sha256,
    );
    push_optional_text(
        &mut output,
        "dataset_manifest_sha256",
        artifact.dataset_manifest_sha256.as_deref(),
    );
    push_optional_i64(
        &mut output,
        "trained_through_ms",
        artifact.trained_through_ms,
    );
    push_float(&mut output, "intercept", artifact.intercept);
    output.push_str(&format!("feature_count={}\n", artifact.features.len()));
    for (index, feature) in artifact.features.iter().enumerate() {
        push_text(
            &mut output,
            &format!("feature[{index}].name"),
            feature.name(),
        );
        push_float(
            &mut output,
            &format!("feature[{index}].mean"),
            feature.mean(),
        );
        push_float(
            &mut output,
            &format!("feature[{index}].scale"),
            feature.scale(),
        );
        push_float(
            &mut output,
            &format!("feature[{index}].coefficient"),
            feature.coefficient(),
        );
    }
    output
}

fn push_optional_text(output: &mut String, key: &str, value: Option<&str>) {
    match value {
        Some(value) => {
            output.push_str(&format!("{key}.present=1\n"));
            push_text(output, key, value);
        }
        None => output.push_str(&format!("{key}.present=0\n")),
    }
}

fn push_optional_i64(output: &mut String, key: &str, value: Option<i64>) {
    match value {
        Some(value) => output.push_str(&format!("{key}.present=1\n{key}.value={value}\n")),
        None => output.push_str(&format!("{key}.present=0\n")),
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|character| character.is_ascii_hexdigit())
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArtifactError {
    InvalidJson(String),
    UnsupportedArtifactVersion(String),
    EmptyModelVersion,
    EmptyFeatureSchemaVersion,
    EmptyFeatureName,
    EmptyFeatureList,
    DuplicateFeatureName(String),
    NonFiniteParameter(String),
    InvalidScale(String),
    InvalidDatasetManifestHash(String),
    FeatureSchemaHashMismatch { expected: String, actual: String },
    ArtifactHashMismatch { expected: String, actual: String },
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for ArtifactError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_round_trip_preserves_and_validates_hash() {
        let artifact = LinearLogitArtifact::schema_canary_v0(
            MlTask::SettlementProbabilityResidual,
            "features-v1",
            ["distance", "seconds"],
        )
        .unwrap();
        // This fixture is shared with the standard-library Python implementation.
        assert_eq!(
            artifact.feature_schema_sha256(),
            "1502f8f90e7104b04834bbc01ab9675b16a70465e2c01eaadf0468fbd720f258"
        );
        assert_eq!(
            artifact.artifact_sha256(),
            "82c212a8bb1ce1705b6b643ecdffc0f5b13143549b71a88184498d08190f5aef"
        );
        let json = artifact.to_json().unwrap();
        let decoded = LinearLogitArtifact::from_json(&json).unwrap();
        assert_eq!(decoded, artifact);
    }

    #[test]
    fn tampered_json_is_rejected() {
        let artifact = LinearLogitArtifact::schema_canary_v0(
            MlTask::SettlementProbabilityResidual,
            "features-v1",
            ["distance"],
        )
        .unwrap();
        let mut json: serde_json::Value =
            serde_json::from_str(&artifact.to_json().unwrap()).unwrap();
        json["artifact_sha256"] = serde_json::Value::String("0".repeat(64));
        let error = LinearLogitArtifact::from_json(&json.to_string()).unwrap_err();
        assert!(matches!(error, ArtifactError::ArtifactHashMismatch { .. }));
    }

    #[test]
    fn ml_b_tasks_are_supported_as_shadow_artifacts() {
        for task in [
            MlTask::FokFillProbability,
            MlTask::FokPostFillToxicityProbability,
        ] {
            LinearLogitArtifact::schema_canary_v0(task, "execution-v1", ["queue_ahead"])
                .unwrap()
                .validate()
                .unwrap();
        }
    }
}
