use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::directional_features::{directional_feature_names, BTC_DIRECTIONAL_FEATURE_NAMES};

pub const BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION: &str = "btc_5m_directional_model_v1";
pub const BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION: &str =
    "btc-5m-directional-core-features-v2";
pub const BTC_DIRECTIONAL_MODEL_FAMILY: &str = "btc_5m_directional_model";
pub const BTC_DIRECTIONAL_MODEL_V1_KEY: &str =
    "btc-5m-directional-histogram-enriched-20260421-20260620-v1";
pub const BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256: &str =
    "e07160295cac3df02d9bec631e96ce34fa66a71a72d04e79e4bcf0fd0f8336b7";
pub const BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256: &str =
    "392aac87ecbc6704929b0ab91aed5de90dff26daf40edfa4e22d2c78a9d03dfc";
pub const BTC_DIRECTIONAL_MODEL_DIR_ENV: &str = "POLYMARKET_BTC_MODEL_DIR";
pub const DEFAULT_BTC_DIRECTIONAL_MODEL_DIR: &str = "/usr/local/share/polymarket-bot/models";

pub const RUNTIME_MODEL_SCHEMA_VERSION: &str = "capitonic-btc-directional-runtime-model-v1";
pub const RUNTIME_MODEL_TIME_BANDED_SCHEMA_VERSION: &str =
    "capitonic-btc-directional-runtime-model-v2";
pub const RUNTIME_MODEL_ASYMMETRIC_VALUE_SCHEMA_VERSION: &str =
    "capitonic-btc-asymmetric-value-runtime-model-v1";
pub const RUNTIME_MODEL_PAYOFF_AWARE_SCHEMA_VERSION: &str =
    "capitonic-btc-payoff-aware-runtime-model-v1";
pub const RUNTIME_MANIFEST_SCHEMA_VERSION: &str = "capitonic-btc-directional-runtime-manifest-v1";
pub const GOLDEN_VECTORS_SCHEMA_VERSION: &str = "capitonic-btc-directional-golden-vectors-v1";
pub const TIME_BANDED_GOLDEN_VECTORS_SCHEMA_VERSION: &str =
    "capitonic-btc-directional-golden-vectors-v2";
pub const ASYMMETRIC_VALUE_GOLDEN_VECTORS_SCHEMA_VERSION: &str =
    "capitonic-btc-asymmetric-value-golden-vectors-v1";
pub const PAYOFF_AWARE_GOLDEN_VECTORS_SCHEMA_VERSION: &str =
    "capitonic-btc-payoff-aware-golden-vectors-v1";
pub const BTC_DIRECTIONAL_MODEL_INPUT_CONTRACT: &str = "btc_directional_model_input_v1";
pub const BTC_ASYMMETRIC_VALUE_MODEL_INPUT_CONTRACT: &str = "btc_asymmetric_value_model_input_v1";

const MODEL_FILE_MAX_BYTES: u64 = 32 * 1024 * 1024;
const GOLDEN_VECTORS_FILE_MAX_BYTES: u64 = 64 * 1024 * 1024;
const SHA256_HEX_LENGTH: usize = 64;

#[allow(clippy::too_many_arguments)]
pub fn directional_model_input_sha256(
    selection: &RuntimeModelSelection,
    feature_schema_version: &str,
    market_id: &str,
    window_start: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
    seconds_elapsed: i64,
    feature_values: &[f64],
) -> Result<String> {
    let feature_names = if let Some(names) = directional_feature_names(feature_schema_version) {
        names
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>()
    } else if feature_schema_version.starts_with("btc-5m-payoff-aware-") {
        runtime_model(selection)?.feature_names().to_vec()
    } else {
        bail!("BTC directional model feature schema is not supported");
    };
    if feature_values.len() != feature_names.len() {
        bail!(
            "BTC directional model input has {} feature values; schema {} requires {}",
            feature_values.len(),
            feature_schema_version,
            feature_names.len()
        );
    }
    let payload = serde_json::json!({
        "contract": BTC_DIRECTIONAL_MODEL_INPUT_CONTRACT,
        "model_key": selection.model_key,
        "model_artifact_sha256": selection.artifact_sha256,
        "feature_schema_version": feature_schema_version,
        "feature_schema_sha256": selection.feature_schema_sha256,
        "feature_names": feature_names,
        "market_id": market_id,
        "window_start": window_start,
        "feature_as_of": feature_as_of,
        "seconds_elapsed": seconds_elapsed,
        "feature_values": feature_values,
    });
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&payload)?)
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn asymmetric_value_model_input_sha256(
    selection: &RuntimeModelSelection,
    feature_schema_version: &str,
    market_id: &str,
    window_start: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
    seconds_elapsed: i64,
    feature_values: &[f64],
    yes_ask_vwap: f64,
    no_ask_vwap: f64,
) -> Result<String> {
    let model = runtime_model(selection)?;
    if !model.is_asymmetric_value()
        || model.feature_schema_version() != feature_schema_version
        || feature_values.len() != model.feature_names().len()
        || !yes_ask_vwap.is_finite()
        || !no_ask_vwap.is_finite()
    {
        bail!("BTC asymmetric value model input contract is invalid");
    }
    let payload = serde_json::json!({
        "contract": BTC_ASYMMETRIC_VALUE_MODEL_INPUT_CONTRACT,
        "model_key": selection.model_key,
        "model_artifact_sha256": selection.artifact_sha256,
        "feature_schema_version": feature_schema_version,
        "feature_schema_sha256": selection.feature_schema_sha256,
        "feature_names": model.feature_names(),
        "market_id": market_id,
        "window_start": window_start,
        "feature_as_of": feature_as_of,
        "seconds_elapsed": seconds_elapsed,
        "feature_values": feature_values,
        "yes_ask_vwap": yes_ask_vwap,
        "no_ask_vwap": no_ask_vwap,
    });
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&payload)?)
    ))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModelSelection {
    pub model_key: String,
    pub artifact_sha256: String,
    pub feature_schema_sha256: String,
}

impl RuntimeModelSelection {
    pub fn validate(&self) -> Result<()> {
        validate_model_key(&self.model_key)?;
        validate_sha256("model artifact", &self.artifact_sha256)?;
        validate_sha256("feature schema", &self.feature_schema_sha256)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BtcDirectionalModelFeatureSnapshot {
    pub model_key: String,
    pub model_artifact_sha256: String,
    pub feature_schema_version: String,
    pub feature_schema_sha256: String,
    pub feature_as_of: DateTime<Utc>,
    pub seconds_elapsed: i64,
    pub feature_values: Vec<f64>,
    pub input_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeModelAction {
    Up,
    Down,
    NoTrade,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RuntimeModelScore {
    pub raw_logit: f64,
    pub probability_up: f64,
    pub confidence: f64,
    pub action: RuntimeModelAction,
    pub accepted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimePredictionPolicy {
    pub minimum_seconds_after_open: i64,
    pub maximum_seconds_after_open: i64,
    pub cadence_seconds: i64,
    pub early_end_second: Option<i64>,
    pub early_cadence_seconds: Option<i64>,
    pub late_start_second: Option<i64>,
}

impl RuntimePredictionPolicy {
    pub fn accepts(self, seconds_elapsed: i64) -> bool {
        if seconds_elapsed < self.minimum_seconds_after_open
            || seconds_elapsed > self.maximum_seconds_after_open
        {
            return false;
        }
        let (start, cadence) = match (self.early_end_second, self.early_cadence_seconds) {
            (Some(end), Some(cadence)) if seconds_elapsed <= end => {
                (self.minimum_seconds_after_open, cadence)
            }
            (Some(_), Some(_)) if self.late_start_second.is_some() => {
                let start = self.late_start_second.expect("checked above");
                if seconds_elapsed < start {
                    return false;
                }
                (start, self.cadence_seconds)
            }
            (Some(end), Some(_)) => (end + 1, self.cadence_seconds),
            _ => (self.minimum_seconds_after_open, self.cadence_seconds),
        };
        (seconds_elapsed - start) % cadence == 0
    }
}

#[derive(Debug)]
pub struct RuntimeDirectionalModel {
    model_key: String,
    artifact_sha256: String,
    feature_schema_version: String,
    feature_schema_sha256: String,
    deployment_scope: Option<String>,
    production_qualified: bool,
    live_capital_allowed: bool,
    feature_names: Vec<String>,
    imputation_medians: Vec<f64>,
    baseline_logit: f64,
    trees: Vec<RuntimeTree>,
    calibration_policy: RuntimeCalibrationPolicy,
    target: RuntimeTarget,
    decision: RuntimeDecision,
    prediction_policy: RuntimePredictionPolicy,
    asymmetric_value_calibration: Option<RuntimeAsymmetricValueCalibration>,
    payoff_model: Option<RuntimePayoffModel>,
}

impl RuntimeDirectionalModel {
    pub fn model_key(&self) -> &str {
        &self.model_key
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn feature_schema_version(&self) -> &str {
        &self.feature_schema_version
    }

    pub fn feature_schema_sha256(&self) -> &str {
        &self.feature_schema_sha256
    }

    pub fn deployment_scope(&self) -> Option<&str> {
        self.deployment_scope.as_deref()
    }

    pub fn production_qualified(&self) -> bool {
        self.production_qualified
    }

    /// Returns the immutable artifact authorization for live capital. Missing legacy metadata,
    /// paper-only artifacts, and unrecognized qualification combinations all fail closed.
    pub fn live_capital_allowed(&self) -> bool {
        self.live_capital_allowed
            && self.deployment_scope.as_deref().is_some_and(|scope| {
                (scope == "development_live_pilot" && !self.production_qualified)
                    || (scope != "paper_only" && self.production_qualified)
            })
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub fn imputation_medians(&self) -> &[f64] {
        &self.imputation_medians
    }

    pub fn prediction_policy(&self) -> RuntimePredictionPolicy {
        self.prediction_policy
    }

    pub fn is_asymmetric_value(&self) -> bool {
        self.asymmetric_value_calibration.is_some()
    }

    pub fn is_payoff_aware(&self) -> bool {
        self.payoff_model.is_some()
    }

    pub fn probability_up_threshold(&self) -> f64 {
        self.decision.probability_up_threshold
    }

    /// Returns the legacy global threshold, or the first configured threshold for a
    /// time-banded model. Inference callers should use `confidence_threshold_at`.
    pub fn confidence_threshold(&self) -> f64 {
        self.calibration_policy.first_confidence_threshold()
    }

    pub fn confidence_threshold_at(&self, seconds_elapsed: i64) -> Result<f64> {
        if !self.prediction_policy.accepts(seconds_elapsed) {
            bail!("BTC directional model elapsed time is outside its prediction policy");
        }
        Ok(self
            .calibration_policy
            .at(seconds_elapsed)?
            .confidence_threshold)
    }

    pub fn score_snapshot(
        &self,
        snapshot: &BtcDirectionalModelFeatureSnapshot,
    ) -> Result<RuntimeModelScore> {
        if snapshot.model_key != self.model_key
            || snapshot.model_artifact_sha256 != self.artifact_sha256
            || snapshot.feature_schema_version != self.feature_schema_version
            || snapshot.feature_schema_sha256 != self.feature_schema_sha256
        {
            bail!("BTC directional model feature snapshot identity does not match the model");
        }
        validate_sha256("feature input", &snapshot.input_sha256)?;
        if !self.prediction_policy.accepts(snapshot.seconds_elapsed) {
            bail!("BTC directional model feature snapshot is outside its prediction policy");
        }
        if let Some(payoff) = &self.payoff_model {
            let window_key = (snapshot.feature_as_of
                - chrono::Duration::seconds(snapshot.seconds_elapsed))
            .timestamp();
            return payoff.score_snapshot(
                &snapshot.feature_values,
                snapshot.seconds_elapsed,
                window_key,
            );
        }
        self.score_at_seconds(&snapshot.feature_values, snapshot.seconds_elapsed)
    }

    pub fn score_asymmetric_value_snapshot(
        &self,
        snapshot: &BtcDirectionalModelFeatureSnapshot,
        yes_ask_vwap: f64,
        no_ask_vwap: f64,
    ) -> Result<RuntimeModelScore> {
        if snapshot.model_key != self.model_key
            || snapshot.model_artifact_sha256 != self.artifact_sha256
            || snapshot.feature_schema_version != self.feature_schema_version
            || snapshot.feature_schema_sha256 != self.feature_schema_sha256
        {
            bail!("BTC asymmetric model feature snapshot identity does not match the model");
        }
        if !self.prediction_policy.accepts(snapshot.seconds_elapsed) {
            bail!("BTC asymmetric model feature snapshot is outside its prediction policy");
        }
        let calibration = self
            .asymmetric_value_calibration
            .as_ref()
            .context("BTC model does not carry asymmetric-value calibration")?;
        if snapshot.feature_values.len() != self.feature_names.len()
            || !yes_ask_vwap.is_finite()
            || !no_ask_vwap.is_finite()
            || !(0.0..=1.0).contains(&yes_ask_vwap)
            || !(0.0..=1.0).contains(&no_ask_vwap)
        {
            bail!("BTC asymmetric model input is invalid");
        }
        let mut raw_logit = self.baseline_logit;
        for tree in &self.trees {
            raw_logit += tree.score(&snapshot.feature_values, &self.imputation_medians)?;
        }
        if !raw_logit.is_finite() {
            bail!("BTC asymmetric model produced a non-finite raw logit");
        }
        let parent = calibration.parent_probability(raw_logit, snapshot.seconds_elapsed)?;
        let probability_up = calibration.coherent_probability(
            parent,
            snapshot.seconds_elapsed,
            yes_ask_vwap,
            no_ask_vwap,
        )?;
        let confidence = probability_up.max(1.0 - probability_up);
        let action = if probability_up >= 0.5 {
            RuntimeModelAction::Up
        } else {
            RuntimeModelAction::Down
        };
        Ok(RuntimeModelScore {
            raw_logit,
            probability_up,
            confidence,
            action,
            accepted: true,
        })
    }

    /// Scores an ordered feature vector without allocating on the inference path.
    ///
    /// Non-finite values follow the frozen training contract and use the corresponding
    /// feature median. This compatibility entrypoint is only valid for legacy models
    /// with one global calibration and confidence threshold.
    pub fn score(&self, features: &[f64]) -> Result<RuntimeModelScore> {
        let policy = self
            .calibration_policy
            .global()
            .context("BTC time-banded directional model scoring requires elapsed-time context")?;
        self.score_with_policy(features, policy)
    }

    /// Scores an ordered feature vector using the calibration and confidence threshold
    /// frozen for the supplied elapsed-time band.
    pub fn score_at_seconds(
        &self,
        features: &[f64],
        seconds_elapsed: i64,
    ) -> Result<RuntimeModelScore> {
        if !self.prediction_policy.accepts(seconds_elapsed) {
            bail!("BTC directional model elapsed time is outside its prediction policy");
        }
        if let Some(payoff) = &self.payoff_model {
            return payoff.score(features, seconds_elapsed);
        }
        let policy = self.calibration_policy.at(seconds_elapsed)?;
        self.score_with_policy(features, policy)
    }

    fn score_with_policy(
        &self,
        features: &[f64],
        policy: RuntimeCalibrationDecisionRef<'_>,
    ) -> Result<RuntimeModelScore> {
        if features.len() != self.feature_names.len() {
            bail!(
                "BTC directional model expected {} features but received {}",
                self.feature_names.len(),
                features.len()
            );
        }

        let mut raw_logit = self.baseline_logit;
        for tree in &self.trees {
            raw_logit += tree.score(features, &self.imputation_medians)?;
        }
        if !raw_logit.is_finite() {
            bail!("BTC directional model produced a non-finite raw logit");
        }

        let raw_probability = sigmoid(raw_logit);
        let clipped_probability = raw_probability.clamp(
            policy.calibration.input_probability_minimum,
            policy.calibration.input_probability_maximum,
        );
        let clipped_logit = (clipped_probability / (1.0 - clipped_probability)).ln();
        let calibrated_logit =
            (clipped_logit * policy.calibration.slope + policy.calibration.intercept).clamp(
                policy.calibration.output_logit_minimum,
                policy.calibration.output_logit_maximum,
            );
        let calibrated_target_probability = sigmoid(calibrated_logit);
        let Some(probability_up) = self
            .target
            .probability_up(features, calibrated_target_probability)
        else {
            return Ok(RuntimeModelScore {
                raw_logit,
                probability_up: 0.5,
                confidence: 0.5,
                action: RuntimeModelAction::NoTrade,
                accepted: false,
            });
        };
        let confidence = probability_up.max(1.0 - probability_up);
        if !probability_up.is_finite() || !confidence.is_finite() {
            bail!("BTC directional model produced a non-finite calibrated probability");
        }

        let action = if confidence < policy.confidence_threshold {
            RuntimeModelAction::NoTrade
        } else if probability_up >= self.decision.probability_up_threshold {
            RuntimeModelAction::Up
        } else {
            RuntimeModelAction::Down
        };
        Ok(RuntimeModelScore {
            raw_logit,
            probability_up,
            confidence,
            action,
            accepted: action != RuntimeModelAction::NoTrade,
        })
    }
}

#[derive(Debug)]
pub struct RuntimeModelRegistry {
    root: PathBuf,
    loaded: Mutex<HashMap<RuntimeModelSelection, Arc<RuntimeDirectionalModel>>>,
}

static RUNTIME_MODEL_REGISTRY: OnceLock<RuntimeModelRegistry> = OnceLock::new();

pub fn runtime_model(selection: &RuntimeModelSelection) -> Result<Arc<RuntimeDirectionalModel>> {
    RUNTIME_MODEL_REGISTRY
        .get_or_init(RuntimeModelRegistry::from_environment)
        .load(selection)
}

impl RuntimeModelRegistry {
    pub fn from_environment() -> Self {
        let root = env::var_os(BTC_DIRECTIONAL_MODEL_DIR_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(default_runtime_model_root);
        Self::new(root)
    }

    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            loaded: Mutex::new(HashMap::new()),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn load(&self, selection: &RuntimeModelSelection) -> Result<Arc<RuntimeDirectionalModel>> {
        selection.validate()?;
        if let Some(model) = self
            .loaded
            .lock()
            .map_err(|_| anyhow::anyhow!("BTC directional model registry lock was poisoned"))?
            .get(selection)
            .cloned()
        {
            return Ok(model);
        }

        let loaded = Arc::new(load_runtime_model(&self.root, selection)?);
        let mut cache = self
            .loaded
            .lock()
            .map_err(|_| anyhow::anyhow!("BTC directional model registry lock was poisoned"))?;
        Ok(cache
            .entry(selection.clone())
            .or_insert_with(|| loaded.clone())
            .clone())
    }
}

fn default_runtime_model_root() -> PathBuf {
    #[cfg(debug_assertions)]
    {
        let workspace_models = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(|packages| packages.join("btc-directional-model/runtime-models"));
        if let Some(workspace_models) = workspace_models.filter(|path| path.is_dir()) {
            return workspace_models;
        }
    }
    PathBuf::from(DEFAULT_BTC_DIRECTIONAL_MODEL_DIR)
}

#[derive(Debug)]
struct RuntimeTree {
    nodes: Vec<RuntimeTreeNode>,
}

impl RuntimeTree {
    fn score(&self, features: &[f64], medians: &[f64]) -> Result<f64> {
        let mut node_index = 0usize;
        // Validation guarantees an acyclic tree. Retain a hard runtime bound so corrupted
        // in-memory state cannot spin on the latency-sensitive inference path.
        for _ in 0..=self.nodes.len() {
            match self
                .nodes
                .get(node_index)
                .context("BTC directional model tree traversal left the validated node range")?
            {
                RuntimeTreeNode::Leaf { value } => return Ok(*value),
                RuntimeTreeNode::Split {
                    feature_index,
                    threshold,
                    missing_go_to_left: _,
                    left,
                    right,
                } => {
                    let raw = features[*feature_index];
                    let value = if raw.is_finite() {
                        raw
                    } else {
                        medians[*feature_index]
                    };
                    node_index = if value <= *threshold { *left } else { *right };
                }
            }
        }
        bail!("BTC directional model tree traversal exceeded the validated node bound")
    }

    fn score_native_missing(&self, features: &[f64]) -> Result<f64> {
        let mut node_index = 0usize;
        for _ in 0..=self.nodes.len() {
            match self
                .nodes
                .get(node_index)
                .context("BTC payoff tree traversal left its validated range")?
            {
                RuntimeTreeNode::Leaf { value } => return Ok(*value),
                RuntimeTreeNode::Split {
                    feature_index,
                    threshold,
                    missing_go_to_left,
                    left,
                    right,
                } => {
                    let value = features[*feature_index];
                    node_index = if !value.is_finite() {
                        if *missing_go_to_left {
                            *left
                        } else {
                            *right
                        }
                    } else if value <= *threshold {
                        *left
                    } else {
                        *right
                    };
                }
            }
        }
        bail!("BTC payoff tree traversal exceeded the validated node bound")
    }
}

#[derive(Debug)]
enum RuntimeTreeNode {
    Split {
        feature_index: usize,
        threshold: f64,
        missing_go_to_left: bool,
        left: usize,
        right: usize,
    },
    Leaf {
        value: f64,
    },
}

#[derive(Debug)]
struct RuntimeSubmodel {
    feature_indices: Vec<usize>,
    baseline: f64,
    probability_output: bool,
    trees: Vec<RuntimeTree>,
}

impl RuntimeSubmodel {
    fn score(&self, source: &[f64]) -> Result<f64> {
        let values = self
            .feature_indices
            .iter()
            .map(|index| source[*index])
            .collect::<Vec<_>>();
        let mut value = self.baseline;
        for tree in &self.trees {
            value += tree.score_native_missing(&values)?;
        }
        Ok(if self.probability_output {
            sigmoid(value)
        } else {
            value
        })
    }

    fn score_derived(&self, values: &[f64]) -> Result<f64> {
        let mut value = self.baseline;
        for tree in &self.trees {
            value += tree.score_native_missing(values)?;
        }
        Ok(if self.probability_output {
            sigmoid(value)
        } else {
            value
        })
    }
}

#[derive(Debug)]
struct RuntimeLogistic {
    coefficients: Vec<f64>,
    intercept: f64,
    means: Option<Vec<f64>>,
    scales: Option<Vec<f64>>,
}

impl RuntimeLogistic {
    fn probability(&self, values: &[f64]) -> Result<f64> {
        if values.len() != self.coefficients.len() {
            bail!("BTC payoff logistic input width is invalid");
        }
        let mut logit = self.intercept;
        for (index, raw) in values.iter().enumerate() {
            let mut value = if raw.is_finite() { *raw } else { 0.0 };
            if let (Some(means), Some(scales)) = (&self.means, &self.scales) {
                value = (value - means[index]) / scales[index];
            }
            logit += self.coefficients[index] * value;
        }
        Ok(sigmoid(logit))
    }
}

#[derive(Debug)]
struct RuntimePayoffExpert {
    start_second: i64,
    end_second_exclusive: i64,
    outcome: RuntimeSubmodel,
    calibrator: RuntimeLogistic,
}

impl RuntimePayoffExpert {
    fn probability(&self, features: &[f64]) -> Result<f64> {
        let raw = self.outcome.score(features)?.clamp(1e-7, 1.0 - 1e-7);
        self.calibrator.probability(&[(raw / (1.0 - raw)).ln()])
    }
}

#[derive(Debug)]
struct RuntimeQ5Threshold {
    enabled: bool,
    confidence: f64,
    edge: f64,
    admission: f64,
    payoff_lower_bound: f64,
}

#[derive(Debug)]
struct RuntimeFairValueThreshold {
    enabled: bool,
    confidence: f64,
    stress_edge: f64,
}

#[derive(Debug)]
enum RuntimePayoffModel {
    Q5 {
        feature_names: Vec<String>,
        price_edges: Vec<f64>,
        minimum_edges: Vec<f64>,
        experts: HashMap<String, RuntimePayoffExpert>,
        family_proxies: HashMap<String, RuntimePayoffExpert>,
        price_band_names: Vec<String>,
        price_calibrator: RuntimeLogistic,
        guard_penalties: HashMap<String, f64>,
        global_guard_penalty: f64,
        admission_feature_names: Vec<String>,
        admission_classifier: RuntimeSubmodel,
        admission_regressor: RuntimeSubmodel,
        thresholds: HashMap<String, RuntimeQ5Threshold>,
        history: Mutex<HashMap<i64, Vec<(i64, f64)>>>,
    },
    Middle {
        feature_names: Vec<String>,
        execution_reserve: f64,
        stress_slippage: f64,
        price_edges: Vec<f64>,
        outcome: RuntimeSubmodel,
        outcome_calibrator: RuntimeLogistic,
        correctness: RuntimeCorrectness,
        confidence_threshold: f64,
        stress_edge_threshold: f64,
        loss_threshold: Option<f64>,
        eligibility_indices: Vec<usize>,
        probability_modifier: Option<RuntimeProbabilityModifier>,
        loss_model: Option<RuntimeDerivedSubmodel>,
    },
    FairValue {
        feature_names: Vec<String>,
        stress_slippage: f64,
        outcome: RuntimeSubmodel,
        yes_cost_index: usize,
        no_cost_index: usize,
        cells: HashMap<String, RuntimeFairValueThreshold>,
    },
}

#[derive(Debug)]
struct RuntimeCorrectness {
    model: RuntimeLogistic,
    regime_indices: Vec<usize>,
    global_penalty: f64,
    locals: HashMap<String, RuntimeLocalCalibration>,
}

#[derive(Debug)]
struct RuntimeLocalCalibration {
    model: RuntimeLogistic,
    weight: f64,
    penalty: f64,
}

#[derive(Debug)]
struct RuntimeProbabilityModifier {
    feature_indices: Vec<usize>,
    model: RuntimeLogistic,
}

#[derive(Debug)]
struct RuntimeDerivedSubmodel {
    feature_names: Vec<String>,
    model: RuntimeSubmodel,
}

impl RuntimePayoffModel {
    fn score(&self, features: &[f64], seconds: i64) -> Result<RuntimeModelScore> {
        match self {
            Self::Q5 { .. } => self.score_q5(features, seconds, None),
            Self::Middle { .. } => self.score_middle(features, seconds),
            Self::FairValue { .. } => self.score_fair_value(features, seconds),
        }
    }

    fn score_snapshot(
        &self,
        features: &[f64],
        seconds: i64,
        window_key: i64,
    ) -> Result<RuntimeModelScore> {
        match self {
            Self::Q5 { .. } => self.score_q5(features, seconds, Some(window_key)),
            Self::Middle { .. } => self.score_middle(features, seconds),
            Self::FairValue { .. } => self.score_fair_value(features, seconds),
        }
    }

    fn score_q5(
        &self,
        features: &[f64],
        seconds: i64,
        window_key: Option<i64>,
    ) -> Result<RuntimeModelScore> {
        let Self::Q5 {
            feature_names,
            price_edges,
            minimum_edges,
            experts,
            family_proxies,
            price_band_names,
            price_calibrator,
            guard_penalties,
            global_guard_penalty,
            admission_feature_names,
            admission_classifier,
            admission_regressor,
            thresholds,
            history,
        } = self
        else {
            unreachable!()
        };
        if features.len() != feature_names.len() {
            bail!("BTC payoff-aware q5 feature width is invalid");
        }
        let band = if seconds < 90 {
            "early"
        } else if seconds < 180 {
            "mid"
        } else {
            "late"
        };
        let expert = experts.get(band).context("BTC q5 expert band is missing")?;
        if seconds < expert.start_second || seconds >= expert.end_second_exclusive {
            bail!("BTC q5 expert is outside its frozen time band");
        }
        let initial_up = expert.probability(features)?;
        let predicted_up = initial_up >= 0.5;
        let initial_selected = if predicted_up {
            initial_up
        } else {
            1.0 - initial_up
        };
        let directional = family_proxies
            .get("directional")
            .context("directional proxy missing")?
            .probability(features)?;
        let asymmetric = family_proxies
            .get("asymmetric")
            .context("asymmetric proxy missing")?
            .probability(features)?;
        let get = |name: &str| -> Result<f64> {
            feature_names
                .iter()
                .position(|value| value == name)
                .map(|index| features[index])
                .with_context(|| format!("BTC q5 runtime feature {name} is missing"))
        };
        let cost = if predicted_up {
            get("pm_yes_cost_per_share")?
        } else {
            get("pm_no_cost_per_share")?
        };
        let bucket = bucket_index(cost, price_edges);
        let band_hot = price_band_names
            .iter()
            .map(|name| f64::from(name == band))
            .collect::<Vec<_>>();
        let bucket_hot = (0..price_edges.len() - 1)
            .map(|index| f64::from(index == bucket))
            .collect::<Vec<_>>();
        let initial_logit = (initial_selected.clamp(1e-6, 1.0 - 1e-6)
            / (1.0 - initial_selected.clamp(1e-6, 1.0 - 1e-6)))
        .ln();
        let mut calibration = vec![initial_logit, f64::from(predicted_up)];
        calibration.extend_from_slice(&band_hot);
        calibration.extend_from_slice(&bucket_hot);
        calibration.extend(band_hot.iter().map(|hot| hot * initial_logit));
        calibration.extend(bucket_hot.iter().map(|hot| hot * initial_logit));
        let selected_probability = price_calibrator.probability(&calibration)?;
        let probability_up = if predicted_up {
            selected_probability
        } else {
            1.0 - selected_probability
        };
        let penalty = guard_penalties
            .get(&format!("{band}:{bucket}"))
            .copied()
            .unwrap_or(*global_guard_penalty);
        let conservative = (selected_probability - penalty).clamp(0.0, 1.0);
        let conservative_edge = conservative - cost;
        let mut derived = HashMap::new();
        derived.insert("probability_selected", selected_probability);
        derived.insert("confidence_margin", selected_probability - 0.5);
        derived.insert("selected_edge_5", selected_probability - cost);
        derived.insert("selected_cost_5", cost);
        derived.insert("directional_family_probability_up", directional);
        derived.insert("asymmetric_family_probability_up", asymmetric);
        derived.insert(
            "directional_family_disagreement",
            (probability_up - directional).abs(),
        );
        derived.insert(
            "asymmetric_family_disagreement",
            (probability_up - asymmetric).abs(),
        );
        derived.insert(
            "family_probability_spread",
            (directional - asymmetric).abs(),
        );
        derived.insert(
            "family_vote_agreement",
            f64::from((directional >= 0.5) == (asymmetric >= 0.5)),
        );
        let changes = if let Some(window_key) = window_key {
            let mut histories = history
                .lock()
                .map_err(|_| anyhow::anyhow!("BTC q5 probability history lock is poisoned"))?;
            histories.retain(|key, _| *key >= window_key - 600);
            let points = histories.entry(window_key).or_default();
            let mut changes = [f64::NAN; 3];
            for (index, lag) in [5, 15, 30].iter().enumerate() {
                if let Some((_, previous)) = points
                    .iter()
                    .find(|(previous_second, _)| *previous_second == seconds - lag)
                {
                    changes[index] = selected_probability - previous;
                }
            }
            if let Some(point) = points
                .iter_mut()
                .find(|(point_second, _)| *point_second == seconds)
            {
                point.1 = selected_probability;
            } else {
                points.push((seconds, selected_probability));
            }
            changes
        } else {
            [f64::NAN; 3]
        };
        derived.insert("probability_change_5s", changes[0]);
        derived.insert("probability_change_15s", changes[1]);
        derived.insert("probability_change_30s", changes[2]);
        derived.insert(
            "probability_instability_30s",
            changes
                .iter()
                .filter(|value| value.is_finite())
                .map(|value| value.abs())
                .reduce(f64::max)
                .unwrap_or(f64::NAN),
        );
        derived.insert("conservative_probability_selected", conservative);
        derived.insert("conservative_edge_5", conservative_edge);
        derived.insert("price_bucket_index", bucket as f64);
        let admission_values = admission_feature_names
            .iter()
            .map(|name| {
                derived
                    .get(name.as_str())
                    .copied()
                    .or_else(|| {
                        feature_names
                            .iter()
                            .position(|value| value == name)
                            .map(|index| features[index])
                    })
                    .unwrap_or(f64::NAN)
            })
            .collect::<Vec<_>>();
        let admission_probability = admission_classifier.score_derived(&admission_values)?;
        let payoff_edge = admission_regressor.score_derived(&admission_values)?;
        let threshold = thresholds
            .get(band)
            .context("BTC q5 threshold band is missing")?;
        let accepted = threshold.enabled
            && conservative >= threshold.confidence
            && conservative_edge >= threshold.edge
            && conservative_edge >= minimum_edges[bucket]
            && admission_probability >= threshold.admission
            && payoff_edge >= threshold.payoff_lower_bound;
        let action = if !accepted {
            RuntimeModelAction::NoTrade
        } else if predicted_up {
            RuntimeModelAction::Up
        } else {
            RuntimeModelAction::Down
        };
        Ok(RuntimeModelScore {
            raw_logit: (probability_up / (1.0 - probability_up)).ln(),
            probability_up,
            confidence: selected_probability,
            action,
            accepted,
        })
    }

    fn score_middle(&self, features: &[f64], seconds: i64) -> Result<RuntimeModelScore> {
        let Self::Middle {
            feature_names,
            execution_reserve,
            stress_slippage,
            price_edges,
            outcome,
            outcome_calibrator,
            correctness,
            confidence_threshold,
            stress_edge_threshold,
            loss_threshold,
            eligibility_indices,
            probability_modifier,
            loss_model,
        } = self
        else {
            unreachable!()
        };
        if features.len() != feature_names.len()
            || eligibility_indices
                .iter()
                .any(|index| !features[*index].is_finite())
        {
            return Ok(no_trade_score());
        }
        let raw_probability = outcome.score(features)?.clamp(1e-6, 1.0 - 1e-6);
        let mut probability_up =
            outcome_calibrator.probability(&[(raw_probability / (1.0 - raw_probability)).ln()])?;
        if let Some(modifier) = probability_modifier {
            let mut values = vec![(probability_up.clamp(1e-6, 1.0 - 1e-6)
                / (1.0 - probability_up.clamp(1e-6, 1.0 - 1e-6)))
            .ln()];
            values.extend(
                modifier
                    .feature_indices
                    .iter()
                    .map(|index| features[*index]),
            );
            probability_up = modifier.model.probability(&values)?;
        }
        let predicted_up = probability_up >= 0.5;
        let selected = if predicted_up {
            probability_up
        } else {
            1.0 - probability_up
        };
        let get = |name: &str| -> Result<f64> {
            feature_names
                .iter()
                .position(|value| value == name)
                .map(|index| features[index])
                .with_context(|| format!("BTC middle runtime feature {name} is missing"))
        };
        let price = if predicted_up {
            get("up_ask_vwap_5")?
        } else {
            get("down_ask_vwap_5")?
        };
        let fee_rate = get("fee_rate")?;
        let cost = price + fee_rate * price * (1.0 - price) + execution_reserve;
        let bucket = bucket_index(cost, price_edges);
        let cell = if seconds < 120 {
            "90-119"
        } else if seconds < 150 {
            "120-149"
        } else {
            "150-179"
        };
        let cells = ["90-119", "120-149", "150-179"];
        let logit =
            (selected.clamp(1e-6, 1.0 - 1e-6) / (1.0 - selected.clamp(1e-6, 1.0 - 1e-6))).ln();
        let cell_hot = cells
            .iter()
            .map(|value| f64::from(*value == cell))
            .collect::<Vec<_>>();
        let bucket_hot = (0..price_edges.len() - 1)
            .map(|index| f64::from(index == bucket))
            .collect::<Vec<_>>();
        let mut matrix = vec![logit, f64::from(predicted_up)];
        matrix.extend_from_slice(&cell_hot);
        matrix.extend_from_slice(&bucket_hot);
        matrix.extend(cell_hot.iter().map(|hot| hot * logit));
        matrix.extend(bucket_hot.iter().map(|hot| hot * logit));
        matrix.extend(correctness.regime_indices.iter().map(|index| {
            if features[*index].is_finite() {
                features[*index]
            } else {
                0.0
            }
        }));
        let global = correctness.model.probability(&matrix)?;
        let key = format!(
            "{cell}:{}:{bucket}",
            if predicted_up { "up" } else { "down" }
        );
        let (correct_probability, penalty) = if let Some(local) = correctness.locals.get(&key) {
            let local_probability = local.model.probability(&[logit])?;
            (
                local.weight * local_probability + (1.0 - local.weight) * global,
                local.weight * local.penalty + (1.0 - local.weight) * correctness.global_penalty,
            )
        } else {
            (global, correctness.global_penalty)
        };
        let lower = (correct_probability - penalty).clamp(0.0, 1.0);
        let stress_edge = lower - cost - stress_slippage;
        let loss = if let Some(loss) = loss_model {
            let values = derived_middle_values(
                &loss.feature_names,
                feature_names,
                features,
                selected,
                cost,
                selected - cost,
                lower,
                stress_edge,
                seconds,
            )?;
            loss.model.score_derived(&values)?
        } else {
            0.0
        };
        let accepted = lower >= *confidence_threshold
            && stress_edge >= *stress_edge_threshold
            && loss_threshold.is_none_or(|threshold| loss <= threshold);
        let action = if !accepted {
            RuntimeModelAction::NoTrade
        } else if predicted_up {
            RuntimeModelAction::Up
        } else {
            RuntimeModelAction::Down
        };
        Ok(RuntimeModelScore {
            raw_logit: (probability_up / (1.0 - probability_up)).ln(),
            probability_up,
            confidence: lower,
            action,
            accepted,
        })
    }

    fn score_fair_value(&self, features: &[f64], seconds: i64) -> Result<RuntimeModelScore> {
        let Self::FairValue {
            feature_names,
            stress_slippage,
            outcome,
            yes_cost_index,
            no_cost_index,
            cells,
        } = self
        else {
            unreachable!()
        };
        if features.len() != feature_names.len() {
            bail!("BTC fair-value feature width is invalid");
        }
        let probability_up = outcome.score(features)?.clamp(1e-6, 1.0 - 1e-6);
        let predicted_up = probability_up >= 0.5;
        let confidence = if predicted_up {
            probability_up
        } else {
            1.0 - probability_up
        };
        let cost = features[if predicted_up {
            *yes_cost_index
        } else {
            *no_cost_index
        }];
        if !cost.is_finite() {
            return Ok(no_trade_score());
        }
        let cell = fair_value_cell(seconds)
            .and_then(|name| cells.get(name))
            .context("BTC fair-value entry cell is missing")?;
        let accepted = cell.enabled
            && confidence >= cell.confidence
            && confidence - cost - stress_slippage >= cell.stress_edge;
        let action = if !accepted {
            RuntimeModelAction::NoTrade
        } else if predicted_up {
            RuntimeModelAction::Up
        } else {
            RuntimeModelAction::Down
        };
        Ok(RuntimeModelScore {
            raw_logit: (probability_up / (1.0 - probability_up)).ln(),
            probability_up,
            confidence,
            action,
            accepted,
        })
    }
}

fn fair_value_cell(seconds: i64) -> Option<&'static str> {
    match seconds {
        15..=89 => Some("early_15_89"),
        90..=119 => Some("middle_90_119"),
        120..=149 => Some("middle_120_149"),
        150..=179 => Some("middle_150_179"),
        180..=240 => Some("late_180_240"),
        _ => None,
    }
}

fn bucket_index(value: f64, edges: &[f64]) -> usize {
    edges[1..edges.len() - 1]
        .iter()
        .take_while(|edge| value >= **edge)
        .count()
}

fn no_trade_score() -> RuntimeModelScore {
    RuntimeModelScore {
        raw_logit: 0.0,
        probability_up: 0.5,
        confidence: 0.5,
        action: RuntimeModelAction::NoTrade,
        accepted: false,
    }
}

#[allow(clippy::too_many_arguments)]
fn derived_middle_values(
    names: &[String],
    feature_names: &[String],
    features: &[f64],
    selected: f64,
    cost: f64,
    edge: f64,
    lower: f64,
    stress_edge: f64,
    _seconds: i64,
) -> Result<Vec<f64>> {
    names
        .iter()
        .map(|name| match name.as_str() {
            "probability_selected" => Ok(selected),
            "selected_cost_5" => Ok(cost),
            "selected_edge_5" => Ok(edge),
            "lower_correctness_probability" => Ok(lower),
            "stress_edge_lower_bound" => Ok(stress_edge),
            other => feature_names
                .iter()
                .position(|value| value == other)
                .map(|index| features[index])
                .with_context(|| format!("BTC derived middle feature {other} is missing")),
        })
        .collect()
}

#[derive(Debug)]
struct RuntimeCalibration {
    slope: f64,
    intercept: f64,
    input_probability_minimum: f64,
    input_probability_maximum: f64,
    output_logit_minimum: f64,
    output_logit_maximum: f64,
}

#[derive(Debug)]
enum RuntimeCalibrationPolicy {
    Global {
        calibration: RuntimeCalibration,
        confidence_threshold: f64,
    },
    TimeBanded {
        bands: Vec<RuntimeTimeBand>,
    },
}

impl RuntimeCalibrationPolicy {
    fn global(&self) -> Option<RuntimeCalibrationDecisionRef<'_>> {
        match self {
            Self::Global {
                calibration,
                confidence_threshold,
            } => Some(RuntimeCalibrationDecisionRef {
                calibration,
                confidence_threshold: *confidence_threshold,
            }),
            Self::TimeBanded { .. } => None,
        }
    }

    fn at(&self, seconds_elapsed: i64) -> Result<RuntimeCalibrationDecisionRef<'_>> {
        match self {
            Self::Global {
                calibration,
                confidence_threshold,
            } => Ok(RuntimeCalibrationDecisionRef {
                calibration,
                confidence_threshold: *confidence_threshold,
            }),
            Self::TimeBanded { bands } => bands
                .iter()
                .find(|band| {
                    seconds_elapsed >= band.start_seconds
                        && seconds_elapsed < band.end_seconds_exclusive
                })
                .map(|band| RuntimeCalibrationDecisionRef {
                    calibration: &band.calibration,
                    confidence_threshold: band.confidence_threshold,
                })
                .context("BTC directional model elapsed time has no frozen calibration band"),
        }
    }

    fn first_confidence_threshold(&self) -> f64 {
        match self {
            Self::Global {
                confidence_threshold,
                ..
            } => *confidence_threshold,
            Self::TimeBanded { bands } => bands[0].confidence_threshold,
        }
    }
}

#[derive(Debug)]
struct RuntimeTimeBand {
    start_seconds: i64,
    end_seconds_exclusive: i64,
    calibration: RuntimeCalibration,
    confidence_threshold: f64,
}

#[derive(Debug)]
struct RuntimeAsymmetricValueCalibration {
    time_bands: Vec<RuntimeAsymmetricTimeBand>,
    side_price_cells: Vec<RuntimeAsymmetricPriceCell>,
}

#[derive(Debug)]
struct RuntimeAsymmetricTimeBand {
    start_seconds: i64,
    end_seconds_exclusive: i64,
    slope: f64,
    intercept: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeAsymmetricSide {
    Yes,
    No,
}

#[derive(Debug)]
struct RuntimeAsymmetricPriceCell {
    start_seconds: i64,
    end_seconds_exclusive: i64,
    minimum_price: f64,
    maximum_price: f64,
    side: RuntimeAsymmetricSide,
    slope: f64,
    intercept: f64,
}

impl RuntimeAsymmetricValueCalibration {
    fn parent_probability(&self, raw_logit: f64, seconds_elapsed: i64) -> Result<f64> {
        let band = self
            .time_bands
            .iter()
            .find(|band| {
                seconds_elapsed >= band.start_seconds
                    && seconds_elapsed < band.end_seconds_exclusive
            })
            .context("BTC asymmetric model has no calibration time band")?;
        Ok(sigmoid(
            (raw_logit * band.slope + band.intercept).clamp(-40.0, 40.0),
        ))
    }

    fn coherent_probability(
        &self,
        parent_yes: f64,
        seconds_elapsed: i64,
        yes_price: f64,
        no_price: f64,
    ) -> Result<f64> {
        let cell = |side, price| {
            self.side_price_cells.iter().find(|cell| {
                cell.side == side
                    && seconds_elapsed >= cell.start_seconds
                    && seconds_elapsed < cell.end_seconds_exclusive
                    && price >= cell.minimum_price
                    && (price < cell.maximum_price || (cell.maximum_price == 1.0 && price <= 1.0))
            })
        };
        let yes = cell(RuntimeAsymmetricSide::Yes, yes_price)
            .context("BTC asymmetric model has no YES price calibration cell")?;
        let no = cell(RuntimeAsymmetricSide::No, no_price)
            .context("BTC asymmetric model has no NO price calibration cell")?;
        let parent_yes = parent_yes.clamp(1e-9, 1.0 - 1e-9);
        let yes_logit = (parent_yes / (1.0 - parent_yes)).ln();
        let parent_no = 1.0 - parent_yes;
        let no_logit = (parent_no / (1.0 - parent_no)).ln();
        let coherent =
            0.5 * ((yes_logit * yes.slope + yes.intercept) - (no_logit * no.slope + no.intercept));
        let probability = sigmoid(coherent.clamp(-40.0, 40.0));
        if !probability.is_finite() {
            bail!("BTC asymmetric model produced a non-finite coherent probability");
        }
        Ok(probability)
    }
}

#[derive(Debug, Clone, Copy)]
struct RuntimeCalibrationDecisionRef<'a> {
    calibration: &'a RuntimeCalibration,
    confidence_threshold: f64,
}

#[derive(Debug)]
enum RuntimeTarget {
    OutcomeUp,
    PathPersistence {
        path_direction_feature_index: usize,
        zero_path_epsilon_bps: f64,
    },
}

impl RuntimeTarget {
    fn probability_up(&self, features: &[f64], calibrated_target_probability: f64) -> Option<f64> {
        match self {
            Self::OutcomeUp => Some(calibrated_target_probability),
            Self::PathPersistence {
                path_direction_feature_index,
                zero_path_epsilon_bps,
            } => {
                let path = features[*path_direction_feature_index];
                if !path.is_finite() || path.abs() <= *zero_path_epsilon_bps {
                    None
                } else if path > 0.0 {
                    Some(calibrated_target_probability)
                } else {
                    Some(1.0 - calibrated_target_probability)
                }
            }
        }
    }
}

#[derive(Debug)]
struct RuntimeDecision {
    probability_up_threshold: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeModelFile {
    schema_version: String,
    model_key: String,
    features: RuntimeFeatureFile,
    estimator: RuntimeEstimatorFile,
    calibration: Option<RuntimeCalibrationFile>,
    time_bands: Option<Vec<RuntimeTimeBandFile>>,
    target: Option<RuntimeTargetFile>,
    asymmetric_value_calibration: Option<RuntimeAsymmetricValueCalibrationFile>,
    payoff_model: Option<RuntimePayoffModelFile>,
    decision: RuntimeDecisionFile,
    prediction_policy: RuntimePredictionPolicyFile,
    provenance: serde_json::Value,
    deployment: Option<RuntimeDeploymentFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeDeploymentFile {
    scope: String,
    production_qualified: bool,
    live_capital_allowed: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeFeatureFile {
    schema_version: String,
    schema_sha256: String,
    names: Vec<String>,
    numeric_type: String,
    non_finite_policy: String,
    imputation_medians: Vec<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeEstimatorFile {
    #[serde(rename = "type")]
    estimator_type: String,
    class_order: [u8; 2],
    output: String,
    baseline_logit: f64,
    tree_values_include_learning_rate: bool,
    split_comparison: String,
    trees: Vec<RuntimeTreeFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeTreeFile {
    nodes: Vec<RuntimeTreeNodeFile>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum RuntimeTreeNodeFile {
    Split {
        feature_index: usize,
        threshold: f64,
        #[serde(rename = "missing_go_to_left")]
        _missing_go_to_left: bool,
        left: usize,
        right: usize,
    },
    Leaf {
        value: f64,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeCalibrationFile {
    #[serde(rename = "type")]
    calibration_type: String,
    slope: f64,
    intercept: f64,
    input_probability_clip: NumericRangeFile,
    output_logit_clip: NumericRangeFile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeTimeBandFile {
    name: String,
    start_seconds: i64,
    end_seconds_exclusive: i64,
    calibration: RuntimeCalibrationFile,
    confidence_threshold: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeAsymmetricValueCalibrationFile {
    time_bands: Vec<RuntimeAsymmetricTimeBandFile>,
    side_price_cells: Vec<RuntimeAsymmetricPriceCellFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeAsymmetricTimeBandFile {
    start_seconds: i64,
    end_seconds_exclusive: i64,
    slope: f64,
    intercept: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeAsymmetricPriceCellFile {
    start_seconds: i64,
    end_seconds_exclusive: i64,
    minimum_price: f64,
    maximum_price: f64,
    side: String,
    slope: f64,
    intercept: f64,
    #[serde(rename = "fitted")]
    _fitted: bool,
    #[serde(rename = "fallback")]
    _fallback: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum RuntimeTargetFile {
    OutcomeUp,
    PathPersistence {
        path_direction_feature: String,
        zero_path_epsilon_bps: f64,
        ineligible_action: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericRangeFile {
    minimum: f64,
    maximum: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeDecisionFile {
    probability_up_threshold: f64,
    confidence_threshold: Option<f64>,
    below_confidence_action: String,
    up_action: String,
    down_action: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimePredictionPolicyFile {
    #[serde(rename = "type")]
    policy_type: String,
    minimum_seconds_after_open: i64,
    maximum_seconds_after_open: i64,
    cadence_seconds: i64,
    #[serde(default)]
    early_end_second: Option<i64>,
    #[serde(default)]
    early_cadence_seconds: Option<i64>,
    #[serde(default)]
    late_start_second: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum RuntimePayoffModelFile {
    Q5 {
        execution_reserve_per_share: f64,
        stress_slippage_per_share: f64,
        price_bucket_edges: Vec<f64>,
        price_bucket_minimum_edges: Vec<f64>,
        experts: HashMap<String, RuntimePayoffExpertFile>,
        family_proxies: HashMap<String, RuntimePayoffExpertFile>,
        price_time_calibration: RuntimePriceTimeCalibrationFile,
        calibration_guard: RuntimePenaltyFile,
        admission: RuntimeAdmissionFile,
        thresholds: HashMap<String, RuntimeQ5ThresholdFile>,
    },
    Middle {
        execution_reserve_per_share: f64,
        stress_slippage_per_share: f64,
        price_bucket_edges: Vec<f64>,
        outcome: RuntimeSubmodelFile,
        outcome_calibrator: RuntimeLogisticFile,
        correctness: RuntimeCorrectnessFile,
        policy: RuntimeMiddlePolicyFile,
        eligibility_feature_indices: Vec<usize>,
        probability_modifier: Option<RuntimeProbabilityModifierFile>,
        loss_model: Option<RuntimeDerivedSubmodelFile>,
    },
    FairValue {
        stress_slippage_per_share: f64,
        outcome: RuntimeSubmodelFile,
        yes_cost_feature_index: usize,
        no_cost_feature_index: usize,
        policy: RuntimeFairValuePolicyFile,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeSubmodelFile {
    feature_indices: Vec<usize>,
    baseline: f64,
    output: String,
    trees: Vec<RuntimeTreeFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeLogisticFile {
    coefficients: Vec<f64>,
    intercept: f64,
    means: Option<Vec<f64>>,
    scales: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimePayoffExpertFile {
    start_second: i64,
    end_second_exclusive: i64,
    outcome: RuntimeSubmodelFile,
    calibrator: RuntimeLogisticFile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimePriceTimeCalibrationFile {
    band_names: Vec<String>,
    model: RuntimeLogisticFile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimePenaltyFile {
    penalties: HashMap<String, f64>,
    global_penalty: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeAdmissionFile {
    feature_names: Vec<String>,
    classifier: RuntimeSubmodelFile,
    regressor: RuntimeSubmodelFile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeQ5ThresholdFile {
    enabled: bool,
    confidence: f64,
    edge: f64,
    admission: f64,
    payoff_lower_bound: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeCorrectnessFile {
    model: RuntimeLogisticFile,
    regime_feature_indices: Vec<usize>,
    global_penalty: f64,
    locals: HashMap<String, RuntimeLocalCalibrationFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeLocalCalibrationFile {
    model: RuntimeLogisticFile,
    weight: f64,
    penalty: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeMiddlePolicyFile {
    confidence: f64,
    stress_edge: f64,
    loss_severity: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeFairValuePolicyFile {
    cells: HashMap<String, RuntimeFairValueThresholdFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeFairValueThresholdFile {
    enabled: bool,
    confidence: f64,
    stress_edge: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeProbabilityModifierFile {
    feature_indices: Vec<usize>,
    model: RuntimeLogisticFile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeDerivedSubmodelFile {
    feature_names: Vec<String>,
    model: RuntimeSubmodelFile,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeManifestFile {
    schema_version: String,
    model_key: String,
    model_file: String,
    model_sha256: String,
    golden_vectors_file: String,
    golden_vectors_sha256: String,
    feature_schema_version: String,
    feature_schema_sha256: String,
    source_freeze_manifest_sha256: String,
    source_training_model_sha256: String,
    deployment_scope: Option<String>,
    production_qualified: Option<bool>,
    live_capital_allowed: Option<bool>,
}

fn load_runtime_model(
    root: &Path,
    selection: &RuntimeModelSelection,
) -> Result<RuntimeDirectionalModel> {
    let directory = root.join(&selection.model_key);
    let manifest_bytes = read_bounded_file(
        &directory.join("manifest.json"),
        MODEL_FILE_MAX_BYTES,
        "runtime model manifest",
    )?;
    let manifest: RuntimeManifestFile = serde_json::from_slice(&manifest_bytes)
        .context("failed to decode BTC directional runtime model manifest")?;
    validate_manifest(&manifest, selection)?;

    let model_path = safe_child_file(&directory, &manifest.model_file)?;
    let model_bytes = read_bounded_file(&model_path, MODEL_FILE_MAX_BYTES, "runtime model")?;
    let artifact_sha256 = sha256_bytes(&model_bytes);
    if artifact_sha256 != manifest.model_sha256 || artifact_sha256 != selection.artifact_sha256 {
        bail!("BTC directional runtime model SHA-256 does not match its selection and manifest");
    }

    let golden_path = safe_child_file(&directory, &manifest.golden_vectors_file)?;
    let golden_bytes = read_bounded_file(
        &golden_path,
        GOLDEN_VECTORS_FILE_MAX_BYTES,
        "runtime model golden vectors",
    )?;
    if sha256_bytes(&golden_bytes) != manifest.golden_vectors_sha256 {
        bail!("BTC directional runtime model golden-vector SHA-256 does not match its manifest");
    }

    let file: RuntimeModelFile = serde_json::from_slice(&model_bytes)
        .context("failed to decode BTC directional runtime model")?;
    compile_runtime_model(file, manifest, selection, artifact_sha256)
}

fn validate_manifest(
    manifest: &RuntimeManifestFile,
    selection: &RuntimeModelSelection,
) -> Result<()> {
    if manifest.schema_version != RUNTIME_MANIFEST_SCHEMA_VERSION
        || manifest.model_key != selection.model_key
        || manifest.model_file != "model.json"
        || manifest.golden_vectors_file != "golden-vectors.json"
        || manifest.model_sha256 != selection.artifact_sha256
        || (directional_feature_names(&manifest.feature_schema_version).is_none()
            && !manifest
                .feature_schema_version
                .starts_with("btc-5m-asymmetric-")
            && !manifest
                .feature_schema_version
                .starts_with("btc-5m-payoff-aware-"))
        || manifest.feature_schema_sha256 != selection.feature_schema_sha256
    {
        bail!("BTC directional runtime model manifest does not match its frozen contract");
    }
    for (name, digest) in [
        ("model", manifest.model_sha256.as_str()),
        ("golden vectors", manifest.golden_vectors_sha256.as_str()),
        (
            "source freeze manifest",
            manifest.source_freeze_manifest_sha256.as_str(),
        ),
        (
            "source training model",
            manifest.source_training_model_sha256.as_str(),
        ),
    ] {
        validate_sha256(name, digest)?;
    }
    validate_manifest_deployment_metadata(manifest)?;
    Ok(())
}

fn validate_manifest_deployment_metadata(manifest: &RuntimeManifestFile) -> Result<()> {
    match (
        manifest.deployment_scope.as_deref(),
        manifest.production_qualified,
        manifest.live_capital_allowed,
    ) {
        (None, None, None) => Ok(()),
        (Some(scope), Some(production_qualified), Some(live_capital_allowed))
            if valid_deployment_metadata(scope, production_qualified, live_capital_allowed) =>
        {
            Ok(())
        }
        _ => bail!("BTC directional runtime model deployment metadata is invalid or incomplete"),
    }
}

fn valid_deployment_metadata(
    scope: &str,
    production_qualified: bool,
    live_capital_allowed: bool,
) -> bool {
    if scope.trim().is_empty() {
        return false;
    }
    match scope {
        "paper_only" => !production_qualified && !live_capital_allowed,
        "development_live_pilot" => !production_qualified && live_capital_allowed,
        _ => !live_capital_allowed || production_qualified,
    }
}

fn compile_runtime_model(
    file: RuntimeModelFile,
    manifest: RuntimeManifestFile,
    selection: &RuntimeModelSelection,
    artifact_sha256: String,
) -> Result<RuntimeDirectionalModel> {
    let legacy_schema = file.schema_version == RUNTIME_MODEL_SCHEMA_VERSION;
    let time_banded_schema = file.schema_version == RUNTIME_MODEL_TIME_BANDED_SCHEMA_VERSION;
    let asymmetric_schema = file.schema_version == RUNTIME_MODEL_ASYMMETRIC_VALUE_SCHEMA_VERSION;
    let payoff_schema = file.schema_version == RUNTIME_MODEL_PAYOFF_AWARE_SCHEMA_VERSION;
    if (!legacy_schema && !time_banded_schema && !asymmetric_schema && !payoff_schema)
        || file.model_key != selection.model_key
        || file.features.schema_version != manifest.feature_schema_version
        || file.features.schema_sha256 != manifest.feature_schema_sha256
        || file.features.numeric_type != "float64"
        || (file.features.non_finite_policy != "median_imputation"
            && !(payoff_schema && file.features.non_finite_policy == "native_missing_branch"))
    {
        bail!("BTC directional runtime model identity or feature contract is invalid");
    }
    if !file.provenance.is_object() {
        bail!("BTC directional runtime model provenance must be a JSON object");
    }
    let (deployment_scope, production_qualified, live_capital_allowed) = match (
        file.deployment.as_ref(),
        manifest.deployment_scope.as_deref(),
        manifest.production_qualified,
        manifest.live_capital_allowed,
    ) {
        (None, None, None, None) => (None, false, false),
        (Some(deployment), Some(scope), Some(production_qualified), Some(live_capital_allowed))
            if deployment.scope == scope
                && deployment.production_qualified == production_qualified
                && deployment.live_capital_allowed == live_capital_allowed
                && valid_deployment_metadata(
                    &deployment.scope,
                    deployment.production_qualified,
                    deployment.live_capital_allowed,
                ) =>
        {
            (
                Some(deployment.scope.clone()),
                deployment.production_qualified,
                deployment.live_capital_allowed,
            )
        }
        _ => {
            bail!("BTC directional runtime model deployment metadata does not match its manifest")
        }
    };
    let feature_count = file.features.names.len();
    if feature_count == 0
        || feature_count != file.features.imputation_medians.len()
        || file
            .features
            .names
            .iter()
            .any(|name| name.trim().is_empty())
        || file.features.names.iter().collect::<HashSet<_>>().len() != feature_count
        || file
            .features
            .imputation_medians
            .iter()
            .any(|value| !value.is_finite())
    {
        bail!("BTC directional runtime model feature contract is malformed");
    }
    if payoff_schema {
        let prediction_policy = compile_payoff_prediction_policy(&file.prediction_policy)?;
        let payoff_model = compile_payoff_model(
            file.payoff_model
                .context("BTC payoff-aware runtime model payload is missing")?,
            &file.features.names,
        )?;
        return Ok(RuntimeDirectionalModel {
            model_key: file.model_key,
            artifact_sha256,
            feature_schema_version: file.features.schema_version,
            feature_schema_sha256: file.features.schema_sha256,
            deployment_scope,
            production_qualified,
            live_capital_allowed,
            feature_names: file.features.names,
            imputation_medians: file.features.imputation_medians,
            baseline_logit: 0.0,
            trees: Vec::new(),
            calibration_policy: RuntimeCalibrationPolicy::Global {
                calibration: RuntimeCalibration {
                    slope: 1.0,
                    intercept: 0.0,
                    input_probability_minimum: 1e-9,
                    input_probability_maximum: 1.0 - 1e-9,
                    output_logit_minimum: -40.0,
                    output_logit_maximum: 40.0,
                },
                confidence_threshold: 0.5,
            },
            target: RuntimeTarget::OutcomeUp,
            decision: RuntimeDecision {
                probability_up_threshold: 0.5,
            },
            prediction_policy,
            asymmetric_value_calibration: None,
            payoff_model: Some(payoff_model),
        });
    } else if asymmetric_schema {
        validate_asymmetric_feature_order(&file.features.names)?;
    } else {
        validate_frozen_feature_order(&file.features.schema_version, &file.features.names)?;
    }

    let estimator = file.estimator;
    if estimator.estimator_type != "histogram_gradient_boosting_binary_classifier"
        || estimator.class_order != [0, 1]
        || estimator.output != "raw_logit"
        || !estimator.tree_values_include_learning_rate
        || estimator.split_comparison != "less_than_or_equal"
        || !estimator.baseline_logit.is_finite()
        || estimator.trees.is_empty()
    {
        bail!("BTC directional runtime model estimator contract is invalid");
    }
    let trees = estimator
        .trees
        .into_iter()
        .enumerate()
        .map(|(index, tree)| compile_tree(tree, feature_count, index))
        .collect::<Result<Vec<_>>>()?;

    let decision = file.decision;
    if !valid_open_probability(decision.probability_up_threshold)
        || decision.below_confidence_action != "no_trade"
        || decision.up_action != "up"
        || decision.down_action != "down"
    {
        bail!("BTC directional runtime model decision contract is invalid");
    }

    let prediction_policy = file.prediction_policy;
    let scheduled = asymmetric_schema && prediction_policy.policy_type == "scheduled";
    if (!scheduled && prediction_policy.policy_type != "first_confidence_crossing")
        || prediction_policy.minimum_seconds_after_open < 0
        || prediction_policy.maximum_seconds_after_open
            < prediction_policy.minimum_seconds_after_open
        || prediction_policy.maximum_seconds_after_open >= 300
        || prediction_policy.cadence_seconds <= 0
        || (!scheduled
            && (prediction_policy.maximum_seconds_after_open
                - prediction_policy.minimum_seconds_after_open)
                % prediction_policy.cadence_seconds
                != 0)
        || (scheduled
            && (prediction_policy.early_end_second != Some(59)
                || prediction_policy.early_cadence_seconds != Some(1)
                || prediction_policy.minimum_seconds_after_open != 1
                || prediction_policy.cadence_seconds != 5))
    {
        bail!("BTC directional runtime model prediction policy is invalid");
    }
    let prediction_policy = RuntimePredictionPolicy {
        minimum_seconds_after_open: prediction_policy.minimum_seconds_after_open,
        maximum_seconds_after_open: prediction_policy.maximum_seconds_after_open,
        cadence_seconds: prediction_policy.cadence_seconds,
        early_end_second: prediction_policy.early_end_second,
        early_cadence_seconds: prediction_policy.early_cadence_seconds,
        late_start_second: prediction_policy.late_start_second,
    };

    let (calibration_policy, target, asymmetric_value_calibration) = if legacy_schema {
        if file.time_bands.is_some() || file.target.is_some() {
            bail!("BTC directional runtime model v1 contains unsupported time-band metadata");
        }
        let calibration = file
            .calibration
            .context("BTC directional runtime model v1 requires one global calibration")?;
        let confidence_threshold = decision
            .confidence_threshold
            .context("BTC directional runtime model v1 requires one global confidence threshold")?;
        if !valid_open_probability(confidence_threshold) {
            bail!("BTC directional runtime model decision contract is invalid");
        }
        (
            RuntimeCalibrationPolicy::Global {
                calibration: compile_calibration(calibration)?,
                confidence_threshold,
            },
            RuntimeTarget::OutcomeUp,
            None,
        )
    } else if time_banded_schema {
        if file.calibration.is_some() || decision.confidence_threshold.is_some() {
            bail!(
                "BTC directional runtime model v2 must freeze calibration and confidence by time band"
            );
        }
        let bands = compile_time_bands(
            file.time_bands
                .context("BTC directional runtime model v2 requires time bands")?,
            prediction_policy,
        )?;
        let target = compile_runtime_target(
            file.target
                .context("BTC directional runtime model v2 requires a target contract")?,
            &file.features.names,
        )?;
        (RuntimeCalibrationPolicy::TimeBanded { bands }, target, None)
    } else {
        if file.calibration.is_some()
            || file.time_bands.is_some()
            || file.target.is_some()
            || decision.confidence_threshold.is_some()
        {
            bail!("BTC asymmetric runtime model contains directional calibration metadata");
        }
        let calibration = compile_asymmetric_value_calibration(
            file.asymmetric_value_calibration
                .context("BTC asymmetric runtime model requires its value calibration contract")?,
            prediction_policy,
        )?;
        (
            RuntimeCalibrationPolicy::Global {
                calibration: RuntimeCalibration {
                    slope: 1.0,
                    intercept: 0.0,
                    input_probability_minimum: 1e-9,
                    input_probability_maximum: 1.0 - 1e-9,
                    output_logit_minimum: -40.0,
                    output_logit_maximum: 40.0,
                },
                confidence_threshold: 0.5,
            },
            RuntimeTarget::OutcomeUp,
            Some(calibration),
        )
    };

    Ok(RuntimeDirectionalModel {
        model_key: file.model_key,
        artifact_sha256,
        feature_schema_version: file.features.schema_version,
        feature_schema_sha256: file.features.schema_sha256,
        deployment_scope,
        production_qualified,
        live_capital_allowed,
        feature_names: file.features.names,
        imputation_medians: file.features.imputation_medians,
        baseline_logit: estimator.baseline_logit,
        trees,
        calibration_policy,
        target,
        decision: RuntimeDecision {
            probability_up_threshold: decision.probability_up_threshold,
        },
        prediction_policy,
        asymmetric_value_calibration,
        payoff_model: None,
    })
}

fn compile_payoff_prediction_policy(
    file: &RuntimePredictionPolicyFile,
) -> Result<RuntimePredictionPolicy> {
    if file.policy_type != "first_confidence_crossing"
        || file.minimum_seconds_after_open < 0
        || file.maximum_seconds_after_open >= 300
        || file.maximum_seconds_after_open < file.minimum_seconds_after_open
        || file.cadence_seconds <= 0
    {
        bail!("BTC payoff-aware prediction policy is invalid");
    }
    if let Some(start) = file.late_start_second {
        if file.early_end_second.is_none()
            || file.early_cadence_seconds.is_none()
            || start <= file.early_end_second.unwrap_or_default()
            || start > file.maximum_seconds_after_open
        {
            bail!("BTC payoff-aware split prediction policy is invalid");
        }
    }
    Ok(RuntimePredictionPolicy {
        minimum_seconds_after_open: file.minimum_seconds_after_open,
        maximum_seconds_after_open: file.maximum_seconds_after_open,
        cadence_seconds: file.cadence_seconds,
        early_end_second: file.early_end_second,
        early_cadence_seconds: file.early_cadence_seconds,
        late_start_second: file.late_start_second,
    })
}

fn compile_logistic(file: RuntimeLogisticFile) -> Result<RuntimeLogistic> {
    let width = file.coefficients.len();
    if width == 0
        || !file.intercept.is_finite()
        || file.coefficients.iter().any(|value| !value.is_finite())
    {
        bail!("BTC payoff logistic model is invalid");
    }
    match (&file.means, &file.scales) {
        (None, None) => {}
        (Some(means), Some(scales))
            if means.len() == width
                && scales.len() == width
                && means.iter().all(|value| value.is_finite())
                && scales.iter().all(|value| value.is_finite() && *value > 0.0) => {}
        _ => bail!("BTC payoff logistic scaling contract is invalid"),
    }
    Ok(RuntimeLogistic {
        coefficients: file.coefficients,
        intercept: file.intercept,
        means: file.means,
        scales: file.scales,
    })
}

fn compile_submodel(file: RuntimeSubmodelFile, source_width: usize) -> Result<RuntimeSubmodel> {
    if file.feature_indices.is_empty()
        || file
            .feature_indices
            .iter()
            .any(|index| *index >= source_width)
        || !file.baseline.is_finite()
        || !matches!(file.output.as_str(), "probability" | "regression")
        || file.trees.is_empty()
    {
        bail!("BTC payoff histogram submodel is invalid");
    }
    let local_width = file.feature_indices.len();
    let trees = file
        .trees
        .into_iter()
        .enumerate()
        .map(|(index, tree)| compile_tree(tree, local_width, index))
        .collect::<Result<Vec<_>>>()?;
    Ok(RuntimeSubmodel {
        feature_indices: file.feature_indices,
        baseline: file.baseline,
        probability_output: file.output == "probability",
        trees,
    })
}

fn compile_expert(
    file: RuntimePayoffExpertFile,
    feature_count: usize,
) -> Result<RuntimePayoffExpert> {
    if file.start_second < 0 || file.end_second_exclusive <= file.start_second {
        bail!("BTC payoff expert time band is invalid");
    }
    Ok(RuntimePayoffExpert {
        start_second: file.start_second,
        end_second_exclusive: file.end_second_exclusive,
        outcome: compile_submodel(file.outcome, feature_count)?,
        calibrator: compile_logistic(file.calibrator)?,
    })
}

fn validate_payoff_numbers(values: &[f64]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        bail!("BTC payoff contract contains a non-finite value");
    }
    Ok(())
}

fn compile_payoff_model(
    file: RuntimePayoffModelFile,
    feature_names: &[String],
) -> Result<RuntimePayoffModel> {
    let feature_count = feature_names.len();
    match file {
        RuntimePayoffModelFile::Q5 {
            execution_reserve_per_share,
            stress_slippage_per_share,
            price_bucket_edges,
            price_bucket_minimum_edges,
            experts,
            family_proxies,
            price_time_calibration,
            calibration_guard,
            admission,
            thresholds,
        } => {
            validate_payoff_numbers(&[
                execution_reserve_per_share,
                stress_slippage_per_share,
                calibration_guard.global_penalty,
            ])?;
            validate_payoff_numbers(&price_bucket_edges)?;
            validate_payoff_numbers(&price_bucket_minimum_edges)?;
            if price_bucket_edges.len() < 2
                || price_bucket_minimum_edges.len() + 1 != price_bucket_edges.len()
            {
                bail!("BTC q5 price buckets are invalid");
            }
            let experts = experts
                .into_iter()
                .map(|(key, value)| Ok((key, compile_expert(value, feature_count)?)))
                .collect::<Result<HashMap<_, _>>>()?;
            let family_proxies = family_proxies
                .into_iter()
                .map(|(key, value)| Ok((key, compile_expert(value, feature_count)?)))
                .collect::<Result<HashMap<_, _>>>()?;
            let admission_width = admission.feature_names.len();
            let admission_classifier = compile_submodel(admission.classifier, admission_width)?;
            let admission_regressor = compile_submodel(admission.regressor, admission_width)?;
            let thresholds = thresholds
                .into_iter()
                .map(|(key, value)| {
                    validate_payoff_numbers(&[
                        value.confidence,
                        value.edge,
                        value.admission,
                        value.payoff_lower_bound,
                    ])?;
                    Ok((
                        key,
                        RuntimeQ5Threshold {
                            enabled: value.enabled,
                            confidence: value.confidence,
                            edge: value.edge,
                            admission: value.admission,
                            payoff_lower_bound: value.payoff_lower_bound,
                        },
                    ))
                })
                .collect::<Result<HashMap<_, _>>>()?;
            Ok(RuntimePayoffModel::Q5 {
                feature_names: feature_names.to_vec(),
                price_edges: price_bucket_edges,
                minimum_edges: price_bucket_minimum_edges,
                experts,
                family_proxies,
                price_band_names: price_time_calibration.band_names,
                price_calibrator: compile_logistic(price_time_calibration.model)?,
                guard_penalties: calibration_guard.penalties,
                global_guard_penalty: calibration_guard.global_penalty,
                admission_feature_names: admission.feature_names,
                admission_classifier,
                admission_regressor,
                thresholds,
                history: Mutex::new(HashMap::new()),
            })
        }
        RuntimePayoffModelFile::Middle {
            execution_reserve_per_share,
            stress_slippage_per_share,
            price_bucket_edges,
            outcome,
            outcome_calibrator,
            correctness,
            policy,
            eligibility_feature_indices,
            probability_modifier,
            loss_model,
        } => {
            validate_payoff_numbers(&[
                execution_reserve_per_share,
                stress_slippage_per_share,
                policy.confidence,
                policy.stress_edge,
                correctness.global_penalty,
            ])?;
            if eligibility_feature_indices
                .iter()
                .any(|index| *index >= feature_count)
            {
                bail!("BTC middle eligibility feature index is invalid");
            }
            let locals = correctness
                .locals
                .into_iter()
                .map(|(key, value)| {
                    Ok((
                        key,
                        RuntimeLocalCalibration {
                            model: compile_logistic(value.model)?,
                            weight: value.weight,
                            penalty: value.penalty,
                        },
                    ))
                })
                .collect::<Result<HashMap<_, _>>>()?;
            let probability_modifier = probability_modifier
                .map(|value| -> Result<_> {
                    if value
                        .feature_indices
                        .iter()
                        .any(|index| *index >= feature_count)
                    {
                        bail!("BTC middle modifier feature index is invalid");
                    }
                    Ok(RuntimeProbabilityModifier {
                        feature_indices: value.feature_indices,
                        model: compile_logistic(value.model)?,
                    })
                })
                .transpose()?;
            let loss_model = loss_model
                .map(|value| -> Result<_> {
                    let width = value.feature_names.len();
                    Ok(RuntimeDerivedSubmodel {
                        feature_names: value.feature_names,
                        model: compile_submodel(value.model, width)?,
                    })
                })
                .transpose()?;
            Ok(RuntimePayoffModel::Middle {
                feature_names: feature_names.to_vec(),
                execution_reserve: execution_reserve_per_share,
                stress_slippage: stress_slippage_per_share,
                price_edges: price_bucket_edges,
                outcome: compile_submodel(outcome, feature_count)?,
                outcome_calibrator: compile_logistic(outcome_calibrator)?,
                correctness: RuntimeCorrectness {
                    model: compile_logistic(correctness.model)?,
                    regime_indices: correctness.regime_feature_indices,
                    global_penalty: correctness.global_penalty,
                    locals,
                },
                confidence_threshold: policy.confidence,
                stress_edge_threshold: policy.stress_edge,
                loss_threshold: policy.loss_severity,
                eligibility_indices: eligibility_feature_indices,
                probability_modifier,
                loss_model,
            })
        }
        RuntimePayoffModelFile::FairValue {
            stress_slippage_per_share,
            outcome,
            yes_cost_feature_index,
            no_cost_feature_index,
            policy,
        } => {
            validate_payoff_numbers(&[stress_slippage_per_share])?;
            if yes_cost_feature_index >= feature_count
                || no_cost_feature_index >= feature_count
                || yes_cost_feature_index == no_cost_feature_index
            {
                bail!("BTC fair-value cost feature indices are invalid");
            }
            let expected_cells = HashSet::from([
                "early_15_89",
                "middle_90_119",
                "middle_120_149",
                "middle_150_179",
                "late_180_240",
            ]);
            if policy
                .cells
                .keys()
                .map(String::as_str)
                .collect::<HashSet<_>>()
                != expected_cells
            {
                bail!("BTC fair-value policy cells are invalid");
            }
            let cells = policy
                .cells
                .into_iter()
                .map(|(name, value)| {
                    validate_payoff_numbers(&[value.confidence, value.stress_edge])?;
                    if !(0.5..=1.0).contains(&value.confidence) {
                        bail!("BTC fair-value confidence threshold is invalid");
                    }
                    Ok((
                        name,
                        RuntimeFairValueThreshold {
                            enabled: value.enabled,
                            confidence: value.confidence,
                            stress_edge: value.stress_edge,
                        },
                    ))
                })
                .collect::<Result<HashMap<_, _>>>()?;
            Ok(RuntimePayoffModel::FairValue {
                feature_names: feature_names.to_vec(),
                stress_slippage: stress_slippage_per_share,
                outcome: compile_submodel(outcome, feature_count)?,
                yes_cost_index: yes_cost_feature_index,
                no_cost_index: no_cost_feature_index,
                cells,
            })
        }
    }
}

fn compile_calibration(file: RuntimeCalibrationFile) -> Result<RuntimeCalibration> {
    if file.calibration_type != "platt_logit"
        || !file.slope.is_finite()
        || !file.intercept.is_finite()
        || !valid_probability_clip(&file.input_probability_clip)
        || !valid_numeric_range(&file.output_logit_clip)
    {
        bail!("BTC directional runtime model calibration contract is invalid");
    }
    Ok(RuntimeCalibration {
        slope: file.slope,
        intercept: file.intercept,
        input_probability_minimum: file.input_probability_clip.minimum,
        input_probability_maximum: file.input_probability_clip.maximum,
        output_logit_minimum: file.output_logit_clip.minimum,
        output_logit_maximum: file.output_logit_clip.maximum,
    })
}

fn compile_asymmetric_value_calibration(
    file: RuntimeAsymmetricValueCalibrationFile,
    prediction_policy: RuntimePredictionPolicy,
) -> Result<RuntimeAsymmetricValueCalibration> {
    if file.time_bands.is_empty() || file.side_price_cells.is_empty() {
        bail!("BTC asymmetric value calibration contract is empty");
    }
    let mut expected_start = prediction_policy.minimum_seconds_after_open;
    let mut time_bands = Vec::with_capacity(file.time_bands.len());
    for band in file.time_bands {
        if band.start_seconds != expected_start
            || band.start_seconds >= band.end_seconds_exclusive
            || !band.slope.is_finite()
            || band.slope <= 0.0
            || !band.intercept.is_finite()
        {
            bail!("BTC asymmetric time calibration contract is invalid");
        }
        expected_start = band.end_seconds_exclusive;
        time_bands.push(RuntimeAsymmetricTimeBand {
            start_seconds: band.start_seconds,
            end_seconds_exclusive: band.end_seconds_exclusive,
            slope: band.slope,
            intercept: band.intercept,
        });
    }
    if expected_start != prediction_policy.maximum_seconds_after_open + 1 {
        bail!("BTC asymmetric time calibration does not cover its prediction policy");
    }
    let mut side_price_cells = Vec::with_capacity(file.side_price_cells.len());
    for cell in file.side_price_cells {
        let side = match cell.side.as_str() {
            "yes" => RuntimeAsymmetricSide::Yes,
            "no" => RuntimeAsymmetricSide::No,
            _ => bail!("BTC asymmetric price calibration side is invalid"),
        };
        if cell.start_seconds < prediction_policy.minimum_seconds_after_open
            || cell.end_seconds_exclusive > prediction_policy.maximum_seconds_after_open + 1
            || cell.start_seconds >= cell.end_seconds_exclusive
            || !cell.minimum_price.is_finite()
            || !cell.maximum_price.is_finite()
            || cell.minimum_price < 0.0
            || cell.maximum_price > 1.0
            || cell.minimum_price >= cell.maximum_price
            || !cell.slope.is_finite()
            || cell.slope <= 0.0
            || !cell.intercept.is_finite()
        {
            bail!("BTC asymmetric price calibration cell is invalid");
        }
        side_price_cells.push(RuntimeAsymmetricPriceCell {
            start_seconds: cell.start_seconds,
            end_seconds_exclusive: cell.end_seconds_exclusive,
            minimum_price: cell.minimum_price,
            maximum_price: cell.maximum_price,
            side,
            slope: cell.slope,
            intercept: cell.intercept,
        });
    }
    for band in &time_bands {
        for side in [RuntimeAsymmetricSide::Yes, RuntimeAsymmetricSide::No] {
            let mut cells = side_price_cells
                .iter()
                .filter(|cell| {
                    cell.side == side
                        && cell.start_seconds == band.start_seconds
                        && cell.end_seconds_exclusive == band.end_seconds_exclusive
                })
                .collect::<Vec<_>>();
            cells.sort_by(|left, right| left.minimum_price.total_cmp(&right.minimum_price));
            if cells.len() != 10
                || cells.first().is_none_or(|cell| cell.minimum_price != 0.0)
                || cells.last().is_none_or(|cell| cell.maximum_price != 1.0)
                || cells
                    .windows(2)
                    .any(|pair| pair[0].maximum_price != pair[1].minimum_price)
            {
                bail!("BTC asymmetric price calibration cells do not cover a side/time band");
            }
        }
    }
    Ok(RuntimeAsymmetricValueCalibration {
        time_bands,
        side_price_cells,
    })
}

fn compile_time_bands(
    files: Vec<RuntimeTimeBandFile>,
    prediction_policy: RuntimePredictionPolicy,
) -> Result<Vec<RuntimeTimeBand>> {
    if files.is_empty() {
        bail!("BTC directional runtime model time-band contract is empty");
    }
    let mut names = HashSet::with_capacity(files.len());
    let mut expected_start = prediction_policy.minimum_seconds_after_open;
    let mut bands = Vec::with_capacity(files.len());
    for file in files {
        if file.name.trim().is_empty()
            || !names.insert(file.name)
            || file.start_seconds != expected_start
            || file.start_seconds >= file.end_seconds_exclusive
            || file.end_seconds_exclusive > prediction_policy.maximum_seconds_after_open + 1
            || (file.start_seconds - prediction_policy.minimum_seconds_after_open)
                % prediction_policy.cadence_seconds
                != 0
            || !valid_open_probability(file.confidence_threshold)
            || file.confidence_threshold <= 0.5
        {
            bail!("BTC directional runtime model time-band contract is invalid");
        }
        expected_start = file.end_seconds_exclusive;
        bands.push(RuntimeTimeBand {
            start_seconds: file.start_seconds,
            end_seconds_exclusive: file.end_seconds_exclusive,
            calibration: compile_calibration(file.calibration)?,
            confidence_threshold: file.confidence_threshold,
        });
    }
    if expected_start != prediction_policy.maximum_seconds_after_open + 1 {
        bail!("BTC directional runtime model time bands do not cover its prediction policy");
    }
    Ok(bands)
}

fn compile_runtime_target(
    file: RuntimeTargetFile,
    feature_names: &[String],
) -> Result<RuntimeTarget> {
    match file {
        RuntimeTargetFile::OutcomeUp => Ok(RuntimeTarget::OutcomeUp),
        RuntimeTargetFile::PathPersistence {
            path_direction_feature,
            zero_path_epsilon_bps,
            ineligible_action,
        } => {
            if !zero_path_epsilon_bps.is_finite()
                || zero_path_epsilon_bps <= 0.0
                || ineligible_action != "no_trade"
            {
                bail!("BTC directional runtime model path-persistence target is invalid");
            }
            let path_direction_feature_index = feature_names
                .iter()
                .position(|name| name == &path_direction_feature)
                .context(
                    "BTC directional runtime model path-persistence direction feature is missing",
                )?;
            Ok(RuntimeTarget::PathPersistence {
                path_direction_feature_index,
                zero_path_epsilon_bps,
            })
        }
    }
}

fn validate_frozen_feature_order(
    feature_schema_version: &str,
    feature_names: &[String],
) -> Result<()> {
    let expected = directional_feature_names(feature_schema_version)
        .context("BTC directional runtime model feature schema is not supported")?;
    if feature_names.len() != expected.len()
        || feature_names
            .iter()
            .zip(expected.iter())
            .any(|(actual, expected)| actual.as_str() != *expected)
    {
        bail!(
            "BTC directional runtime model feature order does not match schema {}",
            feature_schema_version
        );
    }
    Ok(())
}

const ASYMMETRIC_PM_FEATURE_NAMES: [&str; 13] = [
    "pm_yes_cost_per_share",
    "pm_no_cost_per_share",
    "pm_yes_cost_logit",
    "pm_no_cost_logit",
    "pm_cost_overround",
    "pm_yes_minus_no_cost",
    "pm_yes_vwap_slippage",
    "pm_no_vwap_slippage",
    "pm_yes_depth_log",
    "pm_no_depth_log",
    "pm_depth_imbalance",
    "pm_yes_book_age_seconds",
    "pm_no_book_age_seconds",
];

const ASYMMETRIC_ORACLE_FEATURE_NAMES: [&str; 4] = [
    "oracle_return_from_window_open_bps",
    "oracle_round_age_seconds_scaled",
    "oracle_update_count_since_open_scaled",
    "binance_oracle_basis_bps",
];

pub const ASYMMETRIC_L2_FEATURE_NAMES: [&str; 40] = [
    "spot_l2_midpoint_to_kline_close_bps",
    "spot_l2_microprice_to_midpoint_bps",
    "spot_l2_spread_bps",
    "spot_l2_bid_depth_5_log",
    "spot_l2_ask_depth_5_log",
    "spot_l2_imbalance_5",
    "spot_l2_bid_depth_10_log",
    "spot_l2_ask_depth_10_log",
    "spot_l2_imbalance_10",
    "spot_l2_bid_depth_20_log",
    "spot_l2_ask_depth_20_log",
    "spot_l2_imbalance_20",
    "spot_l2_bid_depth_slope_20",
    "spot_l2_ask_depth_slope_20",
    "spot_l2_bid_depth_concentration_20",
    "spot_l2_ask_depth_concentration_20",
    "spot_l2_bid_quote_replenishment_1s_log",
    "spot_l2_ask_quote_replenishment_1s_log",
    "spot_l2_bid_quote_churn_1s_log",
    "spot_l2_ask_quote_churn_1s_log",
    "spot_l2_midpoint_change_1s_bps",
    "spot_l2_spread_change_1s_bps",
    "spot_l2_depth_20_change_1s_bps",
    "spot_l2_imbalance_20_change_1s",
    "spot_l2_midpoint_change_5s_bps",
    "spot_l2_spread_change_5s_bps",
    "spot_l2_depth_20_change_5s_bps",
    "spot_l2_imbalance_20_change_5s",
    "spot_l2_midpoint_change_15s_bps",
    "spot_l2_spread_change_15s_bps",
    "spot_l2_depth_20_change_15s_bps",
    "spot_l2_imbalance_20_change_15s",
    "spot_l2_midpoint_change_30s_bps",
    "spot_l2_spread_change_30s_bps",
    "spot_l2_depth_20_change_30s_bps",
    "spot_l2_imbalance_20_change_30s",
    "spot_l2_midpoint_change_60s_bps",
    "spot_l2_spread_change_60s_bps",
    "spot_l2_depth_20_change_60s_bps",
    "spot_l2_imbalance_20_change_60s",
];

fn validate_asymmetric_feature_order(feature_names: &[String]) -> Result<()> {
    let expected_external: &[&str] = match feature_names.len() {
        71 => &[],
        75 => &ASYMMETRIC_ORACLE_FEATURE_NAMES,
        111 => &ASYMMETRIC_L2_FEATURE_NAMES,
        _ => bail!("BTC asymmetric runtime model feature count is unsupported"),
    };
    let expected = BTC_DIRECTIONAL_FEATURE_NAMES
        .iter()
        .copied()
        .chain(expected_external.iter().copied())
        .chain(ASYMMETRIC_PM_FEATURE_NAMES.iter().copied());
    if feature_names.iter().map(String::as_str).ne(expected) {
        bail!("BTC asymmetric runtime model feature order is invalid");
    }
    Ok(())
}

fn compile_tree(
    tree: RuntimeTreeFile,
    feature_count: usize,
    tree_index: usize,
) -> Result<RuntimeTree> {
    if tree.nodes.is_empty() {
        bail!("BTC directional runtime model tree {tree_index} contains no nodes");
    }
    let node_count = tree.nodes.len();
    let mut parent_counts = vec![0u8; node_count];
    for (node_index, node) in tree.nodes.iter().enumerate() {
        match node {
            RuntimeTreeNodeFile::Split {
                feature_index,
                threshold,
                left,
                right,
                ..
            } => {
                if *feature_index >= feature_count
                    || !threshold.is_finite()
                    || *left >= node_count
                    || *right >= node_count
                    || *left == node_index
                    || *right == node_index
                    || *left == *right
                {
                    bail!(
                        "BTC directional runtime model tree {tree_index} has an invalid split node"
                    );
                }
                parent_counts[*left] = parent_counts[*left].saturating_add(1);
                parent_counts[*right] = parent_counts[*right].saturating_add(1);
            }
            RuntimeTreeNodeFile::Leaf { value } if !value.is_finite() => {
                bail!("BTC directional runtime model tree {tree_index} has a non-finite leaf");
            }
            RuntimeTreeNodeFile::Leaf { .. } => {}
        }
    }
    if parent_counts[0] != 0 || parent_counts[1..].iter().any(|count| *count != 1) {
        bail!("BTC directional runtime model tree {tree_index} is not a rooted tree");
    }

    let mut visited = vec![false; node_count];
    let mut stack = vec![0usize];
    while let Some(index) = stack.pop() {
        if std::mem::replace(&mut visited[index], true) {
            bail!("BTC directional runtime model tree {tree_index} contains a cycle");
        }
        if let RuntimeTreeNodeFile::Split { left, right, .. } = &tree.nodes[index] {
            stack.push(*right);
            stack.push(*left);
        }
    }
    if visited.iter().any(|visited| !visited) {
        bail!("BTC directional runtime model tree {tree_index} contains unreachable nodes");
    }

    let nodes = tree
        .nodes
        .into_iter()
        .map(|node| match node {
            RuntimeTreeNodeFile::Split {
                feature_index,
                threshold,
                _missing_go_to_left: missing_go_to_left,
                left,
                right,
            } => RuntimeTreeNode::Split {
                feature_index,
                threshold,
                missing_go_to_left,
                left,
                right,
            },
            RuntimeTreeNodeFile::Leaf { value } => RuntimeTreeNode::Leaf { value },
        })
        .collect();
    Ok(RuntimeTree { nodes })
}

fn read_bounded_file(path: &Path, maximum_bytes: u64, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to inspect BTC {label} {}", path.display()))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > maximum_bytes {
        bail!("BTC {label} size is outside its supported bound");
    }
    fs::read(path).with_context(|| format!("failed to read BTC {label} {}", path.display()))
}

fn safe_child_file(directory: &Path, file_name: &str) -> Result<PathBuf> {
    let candidate = Path::new(file_name);
    if candidate.components().count() != 1
        || candidate.file_name().and_then(|value| value.to_str()) != Some(file_name)
    {
        bail!("BTC directional runtime model manifest contains an unsafe file name");
    }
    Ok(directory.join(candidate))
}

fn validate_model_key(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || value.starts_with('-')
        || value.ends_with('-')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        bail!("BTC directional model key must be a lowercase dash-separated slug");
    }
    Ok(())
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != SHA256_HEX_LENGTH
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("BTC {label} SHA-256 must be 64 lowercase hexadecimal characters");
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_open_probability(value: f64) -> bool {
    value.is_finite() && value > 0.0 && value < 1.0
}

fn valid_probability_clip(range: &NumericRangeFile) -> bool {
    valid_open_probability(range.minimum)
        && valid_open_probability(range.maximum)
        && range.minimum < range.maximum
}

fn valid_numeric_range(range: &NumericRangeFile) -> bool {
    range.minimum.is_finite() && range.maximum.is_finite() && range.minimum < range.maximum
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::btc::directional_features::{
        BTC_DIRECTIONAL_FEATURE_COUNT, BTC_DIRECTIONAL_FEATURE_NAMES,
        BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_COUNT,
        BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_NAMES,
        BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
    };

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct GoldenVectorsFile {
        schema_version: String,
        model_key: String,
        #[serde(default)]
        feature_schema_version: Option<String>,
        feature_schema_sha256: String,
        vectors: Vec<GoldenVector>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct GoldenVector {
        id: String,
        source: Option<serde_json::Value>,
        seconds_elapsed: Option<i64>,
        feature_values: Vec<Option<f64>>,
        #[serde(default)]
        yes_ask_vwap: Option<f64>,
        #[serde(default)]
        no_ask_vwap: Option<f64>,
        expected: GoldenExpected,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct GoldenExpected {
        raw_logit: f64,
        probability_up: f64,
        confidence: f64,
        action: RuntimeModelAction,
    }

    fn test_model(confidence_threshold: f64) -> RuntimeDirectionalModel {
        RuntimeDirectionalModel {
            model_key: "btc-test-model".to_string(),
            artifact_sha256: "a".repeat(64),
            feature_schema_version: BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION.to_string(),
            feature_schema_sha256: "b".repeat(64),
            deployment_scope: None,
            production_qualified: false,
            live_capital_allowed: false,
            feature_names: vec!["signal".to_string()],
            imputation_medians: vec![0.0],
            baseline_logit: 0.0,
            trees: vec![RuntimeTree {
                nodes: vec![
                    RuntimeTreeNode::Split {
                        feature_index: 0,
                        threshold: 0.0,
                        missing_go_to_left: true,
                        left: 1,
                        right: 2,
                    },
                    RuntimeTreeNode::Leaf { value: -3.0 },
                    RuntimeTreeNode::Leaf { value: 3.0 },
                ],
            }],
            calibration_policy: RuntimeCalibrationPolicy::Global {
                calibration: RuntimeCalibration {
                    slope: 1.0,
                    intercept: 0.0,
                    input_probability_minimum: 1e-9,
                    input_probability_maximum: 0.999_999_999,
                    output_logit_minimum: -40.0,
                    output_logit_maximum: 40.0,
                },
                confidence_threshold,
            },
            target: RuntimeTarget::OutcomeUp,
            decision: RuntimeDecision {
                probability_up_threshold: 0.5,
            },
            prediction_policy: RuntimePredictionPolicy {
                minimum_seconds_after_open: 60,
                maximum_seconds_after_open: 240,
                cadence_seconds: 5,
                early_end_second: None,
                early_cadence_seconds: None,
                late_start_second: None,
            },
            asymmetric_value_calibration: None,
            payoff_model: None,
        }
    }

    fn calibration(intercept: f64) -> RuntimeCalibration {
        RuntimeCalibration {
            slope: 1.0,
            intercept,
            input_probability_minimum: 1e-9,
            input_probability_maximum: 0.999_999_999,
            output_logit_minimum: -40.0,
            output_logit_maximum: 40.0,
        }
    }

    fn calibration_file() -> RuntimeCalibrationFile {
        RuntimeCalibrationFile {
            calibration_type: "platt_logit".to_string(),
            slope: 1.0,
            intercept: 0.0,
            input_probability_clip: NumericRangeFile {
                minimum: 1e-9,
                maximum: 0.999_999_999,
            },
            output_logit_clip: NumericRangeFile {
                minimum: -40.0,
                maximum: 40.0,
            },
        }
    }

    fn time_banded_test_model() -> RuntimeDirectionalModel {
        let mut model = test_model(0.5);
        model.calibration_policy = RuntimeCalibrationPolicy::TimeBanded {
            bands: vec![
                RuntimeTimeBand {
                    start_seconds: 60,
                    end_seconds_exclusive: 90,
                    calibration: calibration(0.0),
                    confidence_threshold: 0.96,
                },
                RuntimeTimeBand {
                    start_seconds: 90,
                    end_seconds_exclusive: 241,
                    calibration: calibration(0.0),
                    confidence_threshold: 0.90,
                },
            ],
        };
        model
    }

    #[test]
    fn frozen_schema_rejects_reordered_or_replaced_features() {
        let expected = BTC_DIRECTIONAL_FEATURE_NAMES
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>();
        validate_frozen_feature_order(BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION, &expected)
            .unwrap();

        let mut reordered = expected.clone();
        reordered.swap(0, 1);
        assert!(validate_frozen_feature_order(
            BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION,
            &reordered,
        )
        .is_err());

        let mut replaced = expected;
        replaced[0] = "unexpected_feature".to_string();
        assert!(validate_frozen_feature_order(
            BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION,
            &replaced,
        )
        .is_err());

        let mature = BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_NAMES
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>();
        validate_frozen_feature_order(
            BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
            &mature,
        )
        .unwrap();
        assert!(validate_frozen_feature_order(
            BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION,
            &mature,
        )
        .is_err());
        assert!(validate_frozen_feature_order(
            BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
            &BTC_DIRECTIONAL_FEATURE_NAMES
                .iter()
                .map(|name| (*name).to_string())
                .collect::<Vec<_>>(),
        )
        .is_err());
        assert!(validate_frozen_feature_order("unsupported-schema", &[]).is_err());
    }

    #[test]
    fn paper_only_deployment_metadata_fails_closed_on_live_permissions() {
        assert!(valid_deployment_metadata("paper_only", false, false));
        assert!(!valid_deployment_metadata("paper_only", true, false));
        assert!(!valid_deployment_metadata("paper_only", false, true));
        assert!(!valid_deployment_metadata("production", false, true));
        assert!(valid_deployment_metadata("production", true, true));
        assert!(valid_deployment_metadata(
            "development_live_pilot",
            false,
            true
        ));
        assert!(!valid_deployment_metadata(
            "development_live_pilot",
            true,
            true
        ));
        assert!(!valid_deployment_metadata(" ", false, false));
    }

    #[test]
    fn runtime_model_live_capital_authorization_is_immutable_and_fail_closed() {
        let mut model = test_model(0.5);
        assert_eq!(model.deployment_scope(), None);
        assert!(!model.production_qualified());
        assert!(!model.live_capital_allowed());

        model.deployment_scope = Some("paper_only".to_string());
        model.production_qualified = true;
        model.live_capital_allowed = true;
        assert!(!model.live_capital_allowed());

        model.deployment_scope = Some("production".to_string());
        model.production_qualified = false;
        assert!(!model.live_capital_allowed());

        model.production_qualified = true;
        assert!(model.live_capital_allowed());

        model.deployment_scope = Some("development_live_pilot".to_string());
        model.production_qualified = false;
        assert!(model.live_capital_allowed());
    }

    #[test]
    fn input_hash_rejects_feature_count_mismatches_for_selected_schema() {
        let selection = RuntimeModelSelection {
            model_key: "btc-test-model".to_string(),
            artifact_sha256: "a".repeat(64),
            feature_schema_sha256: "b".repeat(64),
        };
        let window_start = "2026-07-28T12:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let feature_as_of = "2026-07-28T12:01:00Z".parse::<DateTime<Utc>>().unwrap();

        for (schema_version, expected_count) in [
            (
                BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION,
                BTC_DIRECTIONAL_FEATURE_COUNT,
            ),
            (
                BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
                BTC_DIRECTIONAL_MATURE_REVERSAL_FEATURE_COUNT,
            ),
        ] {
            assert!(directional_model_input_sha256(
                &selection,
                schema_version,
                "market",
                window_start,
                feature_as_of,
                60,
                &vec![0.0; expected_count],
            )
            .is_ok());

            let error = directional_model_input_sha256(
                &selection,
                schema_version,
                "market",
                window_start,
                feature_as_of,
                60,
                &vec![0.0; expected_count - 1],
            )
            .unwrap_err();
            assert!(
                error.to_string().contains(&format!(
                    "schema {schema_version} requires {expected_count}"
                )),
                "unexpected error: {error:#}"
            );
        }
    }

    #[test]
    fn score_is_allocation_free_in_shape_and_symmetric_in_direction() {
        let model = test_model(0.89);
        let up = model.score(&[1.0]).unwrap();
        let down = model.score(&[-1.0]).unwrap();

        assert_eq!(up.action, RuntimeModelAction::Up);
        assert_eq!(down.action, RuntimeModelAction::Down);
        assert!(up.accepted);
        assert!(down.accepted);
        assert!((up.probability_up - (1.0 - down.probability_up)).abs() < 1e-12);
    }

    #[test]
    fn below_confidence_maps_to_no_trade() {
        let model = test_model(0.99);
        let score = model.score(&[1.0]).unwrap();

        assert_eq!(score.action, RuntimeModelAction::NoTrade);
        assert!(!score.accepted);
    }

    #[test]
    fn time_banded_model_uses_elapsed_threshold_and_requires_time_context() {
        let model = time_banded_test_model();

        let early = model.score_at_seconds(&[1.0], 85).unwrap();
        let later = model.score_at_seconds(&[1.0], 90).unwrap();

        assert_eq!(early.action, RuntimeModelAction::NoTrade);
        assert_eq!(later.action, RuntimeModelAction::Up);
        assert_eq!(model.confidence_threshold_at(85).unwrap(), 0.96);
        assert_eq!(model.confidence_threshold_at(90).unwrap(), 0.90);
        assert!(model.score(&[1.0]).is_err());
        assert!(model.score_at_seconds(&[1.0], 89).is_err());
    }

    #[test]
    fn split_prediction_policy_disables_the_middle_interval() {
        let policy = RuntimePredictionPolicy {
            minimum_seconds_after_open: 15,
            maximum_seconds_after_open: 240,
            cadence_seconds: 5,
            early_end_second: Some(89),
            early_cadence_seconds: Some(1),
            late_start_second: Some(180),
        };

        for accepted in [15, 16, 89, 180, 185, 240] {
            assert!(
                policy.accepts(accepted),
                "second {accepted} must be eligible"
            );
        }
        for rejected in [14, 90, 179, 181, 241] {
            assert!(
                !policy.accepts(rejected),
                "second {rejected} must be ineligible"
            );
        }
    }

    #[test]
    fn time_band_contract_must_exactly_cover_prediction_policy() {
        let policy = RuntimePredictionPolicy {
            minimum_seconds_after_open: 60,
            maximum_seconds_after_open: 240,
            cadence_seconds: 5,
            early_end_second: None,
            early_cadence_seconds: None,
            late_start_second: None,
        };
        let valid = vec![
            RuntimeTimeBandFile {
                name: "60-89".to_string(),
                start_seconds: 60,
                end_seconds_exclusive: 90,
                calibration: calibration_file(),
                confidence_threshold: 0.91,
            },
            RuntimeTimeBandFile {
                name: "90-240".to_string(),
                start_seconds: 90,
                end_seconds_exclusive: 241,
                calibration: calibration_file(),
                confidence_threshold: 0.89,
            },
        ];
        assert!(compile_time_bands(valid, policy).is_ok());

        let gap = vec![
            RuntimeTimeBandFile {
                name: "60-89".to_string(),
                start_seconds: 60,
                end_seconds_exclusive: 90,
                calibration: calibration_file(),
                confidence_threshold: 0.91,
            },
            RuntimeTimeBandFile {
                name: "95-240".to_string(),
                start_seconds: 95,
                end_seconds_exclusive: 241,
                calibration: calibration_file(),
                confidence_threshold: 0.89,
            },
        ];
        assert!(compile_time_bands(gap, policy).is_err());
    }

    #[test]
    fn path_persistence_target_converts_by_raw_path_sign_and_rejects_zero_path() {
        let target = RuntimeTarget::PathPersistence {
            path_direction_feature_index: 0,
            zero_path_epsilon_bps: 1e-12,
        };

        assert_eq!(target.probability_up(&[2.0], 0.8), Some(0.8));
        assert!((target.probability_up(&[-2.0], 0.8).unwrap() - 0.2).abs() < f64::EPSILON);
        assert_eq!(target.probability_up(&[0.0], 0.8), None);
        assert_eq!(target.probability_up(&[f64::NAN], 0.8), None);

        let mut model = test_model(0.5);
        model.target = target;
        let ineligible = model.score(&[0.0]).unwrap();
        assert_eq!(ineligible.probability_up, 0.5);
        assert_eq!(ineligible.confidence, 0.5);
        assert_eq!(ineligible.action, RuntimeModelAction::NoTrade);
        assert!(!ineligible.accepted);
    }

    #[test]
    fn non_finite_input_uses_frozen_median() {
        let model = test_model(0.5);
        let missing = model.score(&[f64::NAN]).unwrap();
        let median = model.score(&[0.0]).unwrap();

        assert_eq!(missing, median);
        assert_eq!(missing.action, RuntimeModelAction::Down);
    }

    #[test]
    fn prediction_policy_accepts_only_frozen_cadence() {
        let policy = test_model(0.5).prediction_policy();

        assert!(policy.accepts(60));
        assert!(policy.accepts(240));
        assert!(!policy.accepts(59));
        assert!(!policy.accepts(61));
        assert!(!policy.accepts(245));
    }

    #[test]
    fn all_packaged_runtime_models_match_python_golden_vectors() {
        let local_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("btc-directional-model/runtime-models");
        let root = if local_root.is_dir() {
            local_root
        } else {
            PathBuf::from("/opt/polymarket-models")
        };
        let mut model_directories = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        model_directories.sort();
        assert!(
            !model_directories.is_empty(),
            "runtime model package must contain at least one model"
        );

        let registry = RuntimeModelRegistry::new(&root);
        for directory in model_directories {
            let manifest: RuntimeManifestFile =
                serde_json::from_slice(&fs::read(directory.join("manifest.json")).unwrap())
                    .unwrap();
            assert_eq!(
                directory.file_name().unwrap().to_string_lossy(),
                manifest.model_key
            );
            let selection = RuntimeModelSelection {
                model_key: manifest.model_key.clone(),
                artifact_sha256: manifest.model_sha256.clone(),
                feature_schema_sha256: manifest.feature_schema_sha256.clone(),
            };
            let model = registry.load(&selection).unwrap();
            assert_eq!(
                model.deployment_scope(),
                manifest.deployment_scope.as_deref()
            );
            assert_eq!(
                model.production_qualified(),
                manifest.production_qualified.unwrap_or(false)
            );
            let expected_live_capital_allowed =
                manifest.deployment_scope.as_deref().is_some_and(|scope| {
                    (scope == "development_live_pilot"
                        && !manifest.production_qualified.unwrap_or(false))
                        || (scope != "paper_only" && manifest.production_qualified.unwrap_or(false))
                }) && manifest.live_capital_allowed.unwrap_or(false);
            assert_eq!(model.live_capital_allowed(), expected_live_capital_allowed);
            let vectors: GoldenVectorsFile = serde_json::from_slice(
                &fs::read(directory.join(&manifest.golden_vectors_file)).unwrap(),
            )
            .unwrap();
            assert!(
                vectors.schema_version == GOLDEN_VECTORS_SCHEMA_VERSION
                    || vectors.schema_version == TIME_BANDED_GOLDEN_VECTORS_SCHEMA_VERSION
                    || vectors.schema_version == ASYMMETRIC_VALUE_GOLDEN_VECTORS_SCHEMA_VERSION
                    || vectors.schema_version == PAYOFF_AWARE_GOLDEN_VECTORS_SCHEMA_VERSION
            );
            assert_eq!(vectors.model_key, selection.model_key);
            if let Some(feature_schema_version) = vectors.feature_schema_version.as_deref() {
                assert_eq!(feature_schema_version, model.feature_schema_version());
            }
            assert_eq!(
                vectors.feature_schema_sha256,
                selection.feature_schema_sha256
            );

            for vector in vectors.vectors {
                let _source = vector.source;
                let features = vector
                    .feature_values
                    .into_iter()
                    .map(|value| value.unwrap_or(f64::NAN))
                    .collect::<Vec<_>>();
                let actual = match (vectors.schema_version.as_str(), vector.seconds_elapsed) {
                    (GOLDEN_VECTORS_SCHEMA_VERSION, None) => model.score(&features).unwrap(),
                    (TIME_BANDED_GOLDEN_VECTORS_SCHEMA_VERSION, Some(seconds_elapsed)) => {
                        model.score_at_seconds(&features, seconds_elapsed).unwrap()
                    }
                    (ASYMMETRIC_VALUE_GOLDEN_VECTORS_SCHEMA_VERSION, Some(seconds_elapsed)) => {
                        model
                            .score_asymmetric_value_snapshot(
                                &BtcDirectionalModelFeatureSnapshot {
                                    model_key: selection.model_key.clone(),
                                    model_artifact_sha256: selection.artifact_sha256.clone(),
                                    feature_schema_version: model
                                        .feature_schema_version()
                                        .to_string(),
                                    feature_schema_sha256: selection.feature_schema_sha256.clone(),
                                    feature_as_of: Utc::now(),
                                    seconds_elapsed,
                                    feature_values: features.clone(),
                                    input_sha256: "a".repeat(64),
                                },
                                vector.yes_ask_vwap.unwrap(),
                                vector.no_ask_vwap.unwrap(),
                            )
                            .unwrap()
                    }
                    (PAYOFF_AWARE_GOLDEN_VECTORS_SCHEMA_VERSION, Some(seconds_elapsed)) => {
                        model.score_at_seconds(&features, seconds_elapsed).unwrap()
                    }
                    _ => panic!(
                        "{}:{} golden-vector elapsed-time contract mismatch",
                        selection.model_key, vector.id
                    ),
                };
                assert!(
                    (actual.raw_logit - vector.expected.raw_logit).abs() < 1e-12,
                    "{}:{} raw logit mismatch: {actual:?}",
                    selection.model_key,
                    vector.id
                );
                assert!(
                    (actual.probability_up - vector.expected.probability_up).abs() < 1e-12,
                    "{}:{} probability mismatch: {actual:?}",
                    selection.model_key,
                    vector.id
                );
                assert!(
                    (actual.confidence - vector.expected.confidence).abs() < 1e-12,
                    "{}:{} confidence mismatch: {actual:?}",
                    selection.model_key,
                    vector.id
                );
                assert_eq!(
                    actual.action, vector.expected.action,
                    "{}:{} action mismatch",
                    selection.model_key, vector.id
                );
                assert_eq!(
                    actual.accepted,
                    actual.action != RuntimeModelAction::NoTrade
                );
            }
        }
    }
}
