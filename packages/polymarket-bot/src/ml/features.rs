use std::{collections::HashSet, error::Error, fmt};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MlFeature {
    name: String,
    value: f64,
    source_event_at_ms: i64,
    source_received_at_ms: i64,
}

impl MlFeature {
    pub fn new(
        name: impl Into<String>,
        value: f64,
        source_event_at_ms: i64,
        source_received_at_ms: i64,
    ) -> Result<Self, FeatureError> {
        let feature = Self {
            name: name.into(),
            value,
            source_event_at_ms,
            source_received_at_ms,
        };
        feature.validate()?;
        Ok(feature)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn source_event_at_ms(&self) -> i64 {
        self.source_event_at_ms
    }

    pub fn source_received_at_ms(&self) -> i64 {
        self.source_received_at_ms
    }

    fn validate(&self) -> Result<(), FeatureError> {
        if self.name.trim().is_empty() {
            return Err(FeatureError::EmptyFeatureName);
        }
        if !self.value.is_finite() {
            return Err(FeatureError::NonFiniteValue(self.name.clone()));
        }
        Ok(())
    }
}

/// An immutable, ordered feature vector that is independent of any market-specific module.
///
/// Feature order is part of the schema and is intentionally represented as a vector rather than a
/// JSON object. Each feature carries both source-event and local-receive cutoffs so callers cannot
/// construct a vector containing information that was unavailable at decision time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MlFeatureVector {
    snapshot_id: String,
    market_id: String,
    feature_as_of_ms: i64,
    feature_received_at_ms: i64,
    schema_version: String,
    schema_sha256: String,
    vector_sha256: String,
    prior_probability: f64,
    features: Vec<MlFeature>,
}

impl MlFeatureVector {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        snapshot_id: impl Into<String>,
        market_id: impl Into<String>,
        feature_as_of_ms: i64,
        feature_received_at_ms: i64,
        schema_version: impl Into<String>,
        prior_probability: f64,
        features: Vec<MlFeature>,
    ) -> Result<Self, FeatureError> {
        let snapshot_id = snapshot_id.into();
        let market_id = market_id.into();
        let schema_version = schema_version.into();
        validate_identity("snapshot_id", &snapshot_id)?;
        validate_identity("market_id", &market_id)?;
        validate_identity("schema_version", &schema_version)?;
        if !(0.0..=1.0).contains(&prior_probability) || !prior_probability.is_finite() {
            return Err(FeatureError::InvalidPriorProbability(prior_probability));
        }
        if features.is_empty() {
            return Err(FeatureError::EmptyFeatureVector);
        }
        if feature_received_at_ms > feature_as_of_ms {
            return Err(FeatureError::FeatureReceiptCutoffAfterDecision {
                feature_received_at_ms,
                feature_as_of_ms,
            });
        }

        let mut names = HashSet::with_capacity(features.len());
        for feature in &features {
            feature.validate()?;
            if !names.insert(feature.name().to_string()) {
                return Err(FeatureError::DuplicateFeatureName(
                    feature.name().to_string(),
                ));
            }
            if feature.source_event_at_ms() > feature_as_of_ms {
                return Err(FeatureError::FutureSourceEvent {
                    feature: feature.name().to_string(),
                    source_event_at_ms: feature.source_event_at_ms(),
                    feature_as_of_ms,
                });
            }
            if feature.source_received_at_ms() > feature_received_at_ms {
                return Err(FeatureError::FutureReceivedEvent {
                    feature: feature.name().to_string(),
                    source_received_at_ms: feature.source_received_at_ms(),
                    feature_received_at_ms,
                });
            }
        }

        let feature_names = features
            .iter()
            .map(|feature| feature.name.as_str())
            .collect::<Vec<_>>();
        let schema_sha256 = sha256_hex(&canonical_feature_schema(
            &schema_version,
            feature_names.iter().copied(),
        ));
        let vector_sha256 = sha256_hex(&canonical_feature_vector(
            &snapshot_id,
            &market_id,
            feature_as_of_ms,
            feature_received_at_ms,
            &schema_version,
            prior_probability,
            &features,
        ));
        Ok(Self {
            snapshot_id,
            market_id,
            feature_as_of_ms,
            feature_received_at_ms,
            schema_version,
            schema_sha256,
            vector_sha256,
            prior_probability,
            features,
        })
    }

    pub fn snapshot_id(&self) -> &str {
        &self.snapshot_id
    }

    pub fn market_id(&self) -> &str {
        &self.market_id
    }

    pub fn feature_as_of_ms(&self) -> i64 {
        self.feature_as_of_ms
    }

    pub fn feature_received_at_ms(&self) -> i64 {
        self.feature_received_at_ms
    }

    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    pub fn schema_sha256(&self) -> &str {
        &self.schema_sha256
    }

    pub fn vector_sha256(&self) -> &str {
        &self.vector_sha256
    }

    pub fn prior_probability(&self) -> f64 {
        self.prior_probability
    }

    pub fn features(&self) -> &[MlFeature] {
        &self.features
    }

    pub fn validate_hashes(&self) -> Result<(), FeatureError> {
        let names = self.features.iter().map(|feature| feature.name());
        let expected_schema = sha256_hex(&canonical_feature_schema(&self.schema_version, names));
        if expected_schema != self.schema_sha256 {
            return Err(FeatureError::SchemaHashMismatch {
                expected: expected_schema,
                actual: self.schema_sha256.clone(),
            });
        }
        let expected_vector = sha256_hex(&canonical_feature_vector(
            &self.snapshot_id,
            &self.market_id,
            self.feature_as_of_ms,
            self.feature_received_at_ms,
            &self.schema_version,
            self.prior_probability,
            &self.features,
        ));
        if expected_vector != self.vector_sha256 {
            return Err(FeatureError::VectorHashMismatch {
                expected: expected_vector,
                actual: self.vector_sha256.clone(),
            });
        }
        Ok(())
    }
}

pub fn canonical_feature_schema<'a>(
    schema_version: &str,
    feature_names: impl IntoIterator<Item = &'a str>,
) -> String {
    let names = feature_names.into_iter().collect::<Vec<_>>();
    let mut output = String::from("ml_feature_schema_v1\n");
    push_text(&mut output, "schema_version", schema_version);
    output.push_str(&format!("feature_count={}\n", names.len()));
    for (index, name) in names.into_iter().enumerate() {
        push_text(&mut output, &format!("feature[{index}]"), name);
    }
    output
}

#[allow(clippy::too_many_arguments)]
pub fn canonical_feature_vector(
    snapshot_id: &str,
    market_id: &str,
    feature_as_of_ms: i64,
    feature_received_at_ms: i64,
    schema_version: &str,
    prior_probability: f64,
    features: &[MlFeature],
) -> String {
    let mut output = String::from("ml_feature_vector_v1\n");
    push_text(&mut output, "snapshot_id", snapshot_id);
    push_text(&mut output, "market_id", market_id);
    output.push_str(&format!("feature_as_of_ms={feature_as_of_ms}\n"));
    output.push_str(&format!(
        "feature_received_at_ms={feature_received_at_ms}\n"
    ));
    push_text(&mut output, "schema_version", schema_version);
    push_float(&mut output, "prior_probability", prior_probability);
    output.push_str(&format!("feature_count={}\n", features.len()));
    for (index, feature) in features.iter().enumerate() {
        push_text(
            &mut output,
            &format!("feature[{index}].name"),
            feature.name(),
        );
        push_float(
            &mut output,
            &format!("feature[{index}].value"),
            feature.value(),
        );
        output.push_str(&format!(
            "feature[{index}].source_event_at_ms={}\n",
            feature.source_event_at_ms()
        ));
        output.push_str(&format!(
            "feature[{index}].source_received_at_ms={}\n",
            feature.source_received_at_ms()
        ));
    }
    output
}

pub fn sha256_hex(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) fn push_text(output: &mut String, key: &str, value: &str) {
    output.push_str(&format!("{key}.utf8_bytes={}:{}\n", value.len(), value));
}

pub(crate) fn push_float(output: &mut String, key: &str, value: f64) {
    output.push_str(&format!("{key}.f64_bits={:016x}\n", value.to_bits()));
}

fn validate_identity(field: &'static str, value: &str) -> Result<(), FeatureError> {
    if value.trim().is_empty() {
        return Err(FeatureError::EmptyIdentity(field));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeatureError {
    EmptyIdentity(&'static str),
    EmptyFeatureName,
    EmptyFeatureVector,
    DuplicateFeatureName(String),
    NonFiniteValue(String),
    InvalidPriorProbability(f64),
    FeatureReceiptCutoffAfterDecision {
        feature_received_at_ms: i64,
        feature_as_of_ms: i64,
    },
    FutureSourceEvent {
        feature: String,
        source_event_at_ms: i64,
        feature_as_of_ms: i64,
    },
    FutureReceivedEvent {
        feature: String,
        source_received_at_ms: i64,
        feature_received_at_ms: i64,
    },
    SchemaHashMismatch {
        expected: String,
        actual: String,
    },
    VectorHashMismatch {
        expected: String,
        actual: String,
    },
}

impl fmt::Display for FeatureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for FeatureError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature(name: &str, value: f64) -> MlFeature {
        MlFeature::new(name, value, 1_000, 1_010).unwrap()
    }

    #[test]
    fn schema_hash_changes_with_feature_order() {
        let left = sha256_hex(&canonical_feature_schema("v1", ["a", "b"]));
        let right = sha256_hex(&canonical_feature_schema("v1", ["b", "a"]));
        assert_ne!(left, right);
    }

    #[test]
    fn feature_vector_rejects_future_source_event() {
        let error = MlFeatureVector::new(
            "snapshot-1",
            "market-1",
            999,
            999,
            "v1",
            0.5,
            vec![feature("distance", 1.0)],
        )
        .unwrap_err();
        assert!(matches!(error, FeatureError::FutureSourceEvent { .. }));
    }

    #[test]
    fn vector_hash_is_deterministic() {
        let build = || {
            MlFeatureVector::new(
                "snapshot-1",
                "market-1",
                1_010,
                1_010,
                "v1",
                0.55,
                vec![feature("distance", 1.0), feature("seconds", 45.0)],
            )
            .unwrap()
        };
        assert_eq!(build().vector_sha256(), build().vector_sha256());
    }

    #[test]
    fn feature_receipt_cutoff_cannot_follow_decision_cutoff() {
        let error = MlFeatureVector::new(
            "snapshot-1",
            "market-1",
            1_010,
            1_011,
            "v1",
            0.5,
            vec![feature("distance", 1.0)],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            FeatureError::FeatureReceiptCutoffAfterDecision { .. }
        ));
    }
}
