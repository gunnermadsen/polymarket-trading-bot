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

use super::directional_features::BTC_DIRECTIONAL_FEATURE_NAMES;

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
pub const RUNTIME_MANIFEST_SCHEMA_VERSION: &str = "capitonic-btc-directional-runtime-manifest-v1";
pub const GOLDEN_VECTORS_SCHEMA_VERSION: &str = "capitonic-btc-directional-golden-vectors-v1";

const MODEL_FILE_MAX_BYTES: u64 = 32 * 1024 * 1024;
const GOLDEN_VECTORS_FILE_MAX_BYTES: u64 = 64 * 1024 * 1024;
const SHA256_HEX_LENGTH: usize = 64;

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
}

impl RuntimePredictionPolicy {
    pub fn accepts(self, seconds_elapsed: i64) -> bool {
        seconds_elapsed >= self.minimum_seconds_after_open
            && seconds_elapsed <= self.maximum_seconds_after_open
            && (seconds_elapsed - self.minimum_seconds_after_open) % self.cadence_seconds == 0
    }
}

#[derive(Debug)]
pub struct RuntimeDirectionalModel {
    model_key: String,
    artifact_sha256: String,
    feature_schema_version: String,
    feature_schema_sha256: String,
    feature_names: Vec<String>,
    imputation_medians: Vec<f64>,
    baseline_logit: f64,
    trees: Vec<RuntimeTree>,
    calibration: RuntimeCalibration,
    decision: RuntimeDecision,
    prediction_policy: RuntimePredictionPolicy,
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

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub fn imputation_medians(&self) -> &[f64] {
        &self.imputation_medians
    }

    pub fn prediction_policy(&self) -> RuntimePredictionPolicy {
        self.prediction_policy
    }

    pub fn probability_up_threshold(&self) -> f64 {
        self.decision.probability_up_threshold
    }

    pub fn confidence_threshold(&self) -> f64 {
        self.decision.confidence_threshold
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
        self.score(&snapshot.feature_values)
    }

    /// Scores an ordered feature vector without allocating on the inference path.
    ///
    /// Non-finite values follow the frozen training contract and use the corresponding
    /// feature median. The caller must provide the exact frozen feature order.
    pub fn score(&self, features: &[f64]) -> Result<RuntimeModelScore> {
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
            self.calibration.input_probability_minimum,
            self.calibration.input_probability_maximum,
        );
        let clipped_logit = (clipped_probability / (1.0 - clipped_probability)).ln();
        let calibrated_logit =
            (clipped_logit * self.calibration.slope + self.calibration.intercept).clamp(
                self.calibration.output_logit_minimum,
                self.calibration.output_logit_maximum,
            );
        let probability_up = sigmoid(calibrated_logit);
        let confidence = probability_up.max(1.0 - probability_up);
        if !probability_up.is_finite() || !confidence.is_finite() {
            bail!("BTC directional model produced a non-finite calibrated probability");
        }

        let action = if confidence < self.decision.confidence_threshold {
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
}

#[derive(Debug)]
enum RuntimeTreeNode {
    Split {
        feature_index: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
    Leaf {
        value: f64,
    },
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
struct RuntimeDecision {
    probability_up_threshold: f64,
    confidence_threshold: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeModelFile {
    schema_version: String,
    model_key: String,
    features: RuntimeFeatureFile,
    estimator: RuntimeEstimatorFile,
    calibration: RuntimeCalibrationFile,
    decision: RuntimeDecisionFile,
    prediction_policy: RuntimePredictionPolicyFile,
    provenance: serde_json::Value,
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
struct NumericRangeFile {
    minimum: f64,
    maximum: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeDecisionFile {
    probability_up_threshold: f64,
    confidence_threshold: f64,
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
        || manifest.feature_schema_version != BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION
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
    Ok(())
}

fn compile_runtime_model(
    file: RuntimeModelFile,
    manifest: RuntimeManifestFile,
    selection: &RuntimeModelSelection,
    artifact_sha256: String,
) -> Result<RuntimeDirectionalModel> {
    if file.schema_version != RUNTIME_MODEL_SCHEMA_VERSION
        || file.model_key != selection.model_key
        || file.features.schema_version != manifest.feature_schema_version
        || file.features.schema_sha256 != manifest.feature_schema_sha256
        || file.features.numeric_type != "float64"
        || file.features.non_finite_policy != "median_imputation"
    {
        bail!("BTC directional runtime model identity or feature contract is invalid");
    }
    if !file.provenance.is_object() {
        bail!("BTC directional runtime model provenance must be a JSON object");
    }
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
    validate_frozen_feature_order(&file.features.names)?;

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

    let calibration = file.calibration;
    if calibration.calibration_type != "platt_logit"
        || !calibration.slope.is_finite()
        || !calibration.intercept.is_finite()
        || !valid_probability_clip(&calibration.input_probability_clip)
        || !valid_numeric_range(&calibration.output_logit_clip)
    {
        bail!("BTC directional runtime model calibration contract is invalid");
    }

    let decision = file.decision;
    if !valid_open_probability(decision.probability_up_threshold)
        || !valid_open_probability(decision.confidence_threshold)
        || decision.below_confidence_action != "no_trade"
        || decision.up_action != "up"
        || decision.down_action != "down"
    {
        bail!("BTC directional runtime model decision contract is invalid");
    }

    let prediction_policy = file.prediction_policy;
    if prediction_policy.policy_type != "first_confidence_crossing"
        || prediction_policy.minimum_seconds_after_open < 0
        || prediction_policy.maximum_seconds_after_open
            < prediction_policy.minimum_seconds_after_open
        || prediction_policy.maximum_seconds_after_open >= 300
        || prediction_policy.cadence_seconds <= 0
        || (prediction_policy.maximum_seconds_after_open
            - prediction_policy.minimum_seconds_after_open)
            % prediction_policy.cadence_seconds
            != 0
    {
        bail!("BTC directional runtime model prediction policy is invalid");
    }

    Ok(RuntimeDirectionalModel {
        model_key: file.model_key,
        artifact_sha256,
        feature_schema_version: file.features.schema_version,
        feature_schema_sha256: file.features.schema_sha256,
        feature_names: file.features.names,
        imputation_medians: file.features.imputation_medians,
        baseline_logit: estimator.baseline_logit,
        trees,
        calibration: RuntimeCalibration {
            slope: calibration.slope,
            intercept: calibration.intercept,
            input_probability_minimum: calibration.input_probability_clip.minimum,
            input_probability_maximum: calibration.input_probability_clip.maximum,
            output_logit_minimum: calibration.output_logit_clip.minimum,
            output_logit_maximum: calibration.output_logit_clip.maximum,
        },
        decision: RuntimeDecision {
            probability_up_threshold: decision.probability_up_threshold,
            confidence_threshold: decision.confidence_threshold,
        },
        prediction_policy: RuntimePredictionPolicy {
            minimum_seconds_after_open: prediction_policy.minimum_seconds_after_open,
            maximum_seconds_after_open: prediction_policy.maximum_seconds_after_open,
            cadence_seconds: prediction_policy.cadence_seconds,
        },
    })
}

fn validate_frozen_feature_order(feature_names: &[String]) -> Result<()> {
    if feature_names.len() != BTC_DIRECTIONAL_FEATURE_NAMES.len()
        || feature_names
            .iter()
            .zip(BTC_DIRECTIONAL_FEATURE_NAMES)
            .any(|(actual, expected)| actual != expected)
    {
        bail!(
            "BTC directional runtime model feature order does not match schema {}",
            BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION
        );
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
                left,
                right,
                ..
            } => RuntimeTreeNode::Split {
                feature_index,
                threshold,
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

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct GoldenVectorsFile {
        schema_version: String,
        model_key: String,
        feature_schema_sha256: String,
        vectors: Vec<GoldenVector>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct GoldenVector {
        id: String,
        source: Option<serde_json::Value>,
        feature_values: Vec<Option<f64>>,
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
            feature_names: vec!["signal".to_string()],
            imputation_medians: vec![0.0],
            baseline_logit: 0.0,
            trees: vec![RuntimeTree {
                nodes: vec![
                    RuntimeTreeNode::Split {
                        feature_index: 0,
                        threshold: 0.0,
                        left: 1,
                        right: 2,
                    },
                    RuntimeTreeNode::Leaf { value: -3.0 },
                    RuntimeTreeNode::Leaf { value: 3.0 },
                ],
            }],
            calibration: RuntimeCalibration {
                slope: 1.0,
                intercept: 0.0,
                input_probability_minimum: 1e-9,
                input_probability_maximum: 0.999_999_999,
                output_logit_minimum: -40.0,
                output_logit_maximum: 40.0,
            },
            decision: RuntimeDecision {
                probability_up_threshold: 0.5,
                confidence_threshold,
            },
            prediction_policy: RuntimePredictionPolicy {
                minimum_seconds_after_open: 60,
                maximum_seconds_after_open: 240,
                cadence_seconds: 5,
            },
        }
    }

    #[test]
    fn frozen_schema_rejects_reordered_or_replaced_features() {
        let expected = BTC_DIRECTIONAL_FEATURE_NAMES
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>();
        validate_frozen_feature_order(&expected).unwrap();

        let mut reordered = expected.clone();
        reordered.swap(0, 1);
        assert!(validate_frozen_feature_order(&reordered).is_err());

        let mut replaced = expected;
        replaced[0] = "unexpected_feature".to_string();
        assert!(validate_frozen_feature_order(&replaced).is_err());
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
    fn packaged_runtime_model_matches_all_python_golden_vectors() {
        let local_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("btc-directional-model/runtime-models");
        let root = if local_root.is_dir() {
            local_root
        } else {
            PathBuf::from("/opt/polymarket-models")
        };
        let selection = RuntimeModelSelection {
            model_key: BTC_DIRECTIONAL_MODEL_V1_KEY.to_string(),
            artifact_sha256: BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256.to_string(),
            feature_schema_sha256: BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256.to_string(),
        };
        let registry = RuntimeModelRegistry::new(&root);
        let model = registry.load(&selection).unwrap();
        let vectors_path = root
            .join(BTC_DIRECTIONAL_MODEL_V1_KEY)
            .join("golden-vectors.json");
        let vectors: GoldenVectorsFile =
            serde_json::from_slice(&fs::read(vectors_path).unwrap()).unwrap();
        assert_eq!(vectors.schema_version, GOLDEN_VECTORS_SCHEMA_VERSION);
        assert_eq!(vectors.model_key, BTC_DIRECTIONAL_MODEL_V1_KEY);
        assert_eq!(
            vectors.feature_schema_sha256,
            BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256
        );

        for vector in vectors.vectors {
            let _source = vector.source;
            let features = vector
                .feature_values
                .into_iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            let actual = model.score(&features).unwrap();
            assert!(
                (actual.raw_logit - vector.expected.raw_logit).abs() < 1e-12,
                "{} raw logit mismatch: {actual:?}",
                vector.id
            );
            assert!(
                (actual.probability_up - vector.expected.probability_up).abs() < 1e-12,
                "{} probability mismatch: {actual:?}",
                vector.id
            );
            assert!(
                (actual.confidence - vector.expected.confidence).abs() < 1e-12,
                "{} confidence mismatch: {actual:?}",
                vector.id
            );
            assert_eq!(
                actual.action, vector.expected.action,
                "{} action mismatch",
                vector.id
            );
            assert_eq!(
                actual.accepted,
                actual.action != RuntimeModelAction::NoTrade
            );
        }
    }
}
