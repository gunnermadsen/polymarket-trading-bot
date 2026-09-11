//! Process-selected risk inference for already-approved BTC entry candidates.
//!
//! Risk packages are mounted beside directional packages.  Merely mounting a
//! package never activates it: a process opts in with one `risk_strategies`
//! selector.  The adapter consumes only the causal candidate snapshot already
//! assembled by the trading pipeline.
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, ensure, Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::btc::{
    directional_model::{
        compile_submodel, RuntimeSubmodel, RuntimeSubmodelFile, BTC_DIRECTIONAL_MODEL_DIR_ENV,
        DEFAULT_BTC_DIRECTIONAL_MODEL_DIR,
    },
    strategy::{BtcDecision, BtcStrategyPrediction},
    BtcFeatureSnapshot, BtcOutcome,
};

pub const RISK_STRATEGY_VERSION: &str = "capitonic-risk-strategy-v1";
pub const RISK_EVALUATION_VERSION: &str = "capitonic-risk-evaluation-v1";
pub const RISK_FEATURE_SCHEMA_VERSION: &str = "btc-risk-candidate-v1";
pub const RISK_FEATURE_NAMES: [&str; 12] = [
    "seconds_elapsed",
    "selected_probability",
    "confidence",
    "share_cost",
    "fee_per_share",
    "expected_edge",
    "side_is_up",
    "pm_vwap5_overround",
    "pm_selected_imbalance",
    "pm_selected_book_age_seconds",
    "chainlink_gap_bps",
    "realized_volatility",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskStrategySelection {
    pub version: String,
    pub model_key: String,
    pub artifact_sha256: String,
}

impl RiskStrategySelection {
    pub fn validate(&self) -> Result<()> {
        if self.version != RISK_STRATEGY_VERSION || self.model_key.trim().is_empty() {
            bail!("unsupported or incomplete risk strategy selector");
        }
        validate_sha256(&self.artifact_sha256)
    }
}

pub fn validate_selections(values: &[RiskStrategySelection]) -> Result<()> {
    if values.len() > 1 {
        bail!("risk_strategies v1 accepts at most one strategy");
    }
    if let Some(value) = values.first() {
        value.validate()?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskDisposition {
    Allow,
    Defer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskEvaluation {
    pub version: &'static str,
    pub model: RiskStrategySelection,
    pub policy: RiskPolicy,
    pub candidate: RiskCandidate,
    pub disposition: RiskDisposition,
    pub reason: &'static str,
    pub score: f64,
    pub threshold: f64,
    pub feature_as_of: DateTime<Utc>,
    pub input_sha256: String,
    pub inference_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskPolicy {
    pub qualified_start_second: i64,
    pub qualified_end_second: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCandidate {
    pub decision_id: uuid::Uuid,
    pub market_id: String,
    pub outcome: BtcOutcome,
    pub probability: f64,
    pub confidence: f64,
    pub executable_price: f64,
    pub expected_edge_per_share: f64,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: String,
    model_key: String,
    model_sha256: String,
    feature_schema_version: String,
    feature_names: Vec<String>,
    threshold: f64,
    qualified_start_second: i64,
    qualified_end_second: i64,
    deployment: serde_json::Value,
    provenance: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    model: RuntimeSubmodelFile,
    threshold: f64,
    qualified_start_second: i64,
    qualified_end_second: i64,
}

pub struct RuntimeRiskModel {
    selection: RiskStrategySelection,
    threshold: f64,
    start_second: i64,
    end_second: i64,
    model: RuntimeSubmodel,
}

impl RuntimeRiskModel {
    pub fn selection(&self) -> &RiskStrategySelection {
        &self.selection
    }

    pub fn evaluate(
        &self,
        snapshot: &BtcFeatureSnapshot,
        decision: &BtcDecision,
    ) -> Result<RiskEvaluation> {
        let started = std::time::Instant::now();
        let (values, seconds_elapsed) = feature_values(snapshot, decision)?;
        let input_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&values)?));
        let score = self.model.score(&values)?;
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            bail!("risk model produced invalid score");
        }
        let in_bucket = (self.start_second..=self.end_second).contains(&seconds_elapsed);
        let disposition = if in_bucket && score >= self.threshold {
            RiskDisposition::Defer
        } else {
            RiskDisposition::Allow
        };
        let reason = if !in_bucket {
            "outside_qualified_bucket"
        } else if disposition == RiskDisposition::Defer {
            "predicted_loss_risk"
        } else {
            "threshold_satisfied"
        };
        Ok(RiskEvaluation {
            version: RISK_EVALUATION_VERSION,
            model: self.selection.clone(),
            policy: RiskPolicy {
                qualified_start_second: self.start_second,
                qualified_end_second: self.end_second,
            },
            candidate: RiskCandidate {
                decision_id: decision.decision_id,
                market_id: snapshot.market_id.clone(),
                outcome: decision
                    .approved_intent
                    .as_ref()
                    .context("risk candidate disappeared")?
                    .outcome,
                probability: values[1],
                confidence: values[2],
                executable_price: values[3],
                expected_edge_per_share: values[5],
            },
            disposition,
            reason,
            score,
            threshold: self.threshold,
            feature_as_of: snapshot.observed_at,
            input_sha256,
            inference_seconds: started.elapsed().as_secs_f64(),
        })
    }
}

pub fn load(selection: &RiskStrategySelection) -> Result<Arc<RuntimeRiskModel>> {
    selection.validate()?;
    let root = std::env::var(BTC_DIRECTIONAL_MODEL_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_BTC_DIRECTIONAL_MODEL_DIR));
    load_from_root(selection, &root)
}

fn load_from_root(selection: &RiskStrategySelection, root: &Path) -> Result<Arc<RuntimeRiskModel>> {
    selection.validate()?;
    let package = root.join(&selection.model_key);
    ensure_package_path(root, &package)?;
    let manifest_bytes = bounded_read(&package.join("risk-manifest.json"))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        manifest.version == RISK_STRATEGY_VERSION,
        "risk manifest version is incompatible"
    );
    ensure!(
        manifest.model_key == selection.model_key,
        "risk manifest model key is incompatible"
    );
    ensure!(
        manifest.model_sha256 == selection.artifact_sha256,
        "risk manifest artifact identity is incompatible"
    );
    ensure!(
        manifest.feature_schema_version == RISK_FEATURE_SCHEMA_VERSION,
        "risk feature schema is incompatible"
    );
    ensure!(
        manifest.feature_names == RISK_FEATURE_NAMES.map(str::to_string),
        "risk feature order is incompatible"
    );
    ensure!(
        manifest.threshold.is_finite() && (0.0..=1.0).contains(&manifest.threshold),
        "risk threshold is incompatible"
    );
    ensure!(
        manifest.qualified_start_second >= 0
            && manifest.qualified_end_second >= manifest.qualified_start_second,
        "risk qualified bucket is incompatible"
    );
    let artifact_bytes = bounded_read(&package.join("risk-model.json"))?;
    let actual = format!("{:x}", Sha256::digest(&artifact_bytes));
    if actual != selection.artifact_sha256 {
        bail!("risk model artifact checksum mismatch");
    }
    let artifact: Artifact = serde_json::from_slice(&artifact_bytes)?;
    if artifact.threshold != manifest.threshold
        || artifact.qualified_start_second != manifest.qualified_start_second
        || artifact.qualified_end_second != manifest.qualified_end_second
    {
        bail!("risk manifest policy differs from immutable artifact");
    }
    Ok(Arc::new(RuntimeRiskModel {
        selection: selection.clone(),
        threshold: artifact.threshold,
        start_second: artifact.qualified_start_second,
        end_second: artifact.qualified_end_second,
        model: compile_submodel(artifact.model, RISK_FEATURE_NAMES.len())?,
    }))
}

fn feature_values(
    snapshot: &BtcFeatureSnapshot,
    decision: &BtcDecision,
) -> Result<(Vec<f64>, i64)> {
    let intent = decision
        .approved_intent
        .as_ref()
        .context("risk inference requires approved intent")?;
    let (probability, conservative_probability, executable_price, fee) =
        match decision.prediction.as_ref() {
            Some(BtcStrategyPrediction::DirectionalPrediction {
                probability,
                conservative_probability,
                executable_price,
                direct_taker_fee_per_share,
                ..
            }) => (
                probability.to_f64(),
                conservative_probability.to_f64(),
                executable_price.and_then(|v| v.to_f64()),
                direct_taker_fee_per_share.and_then(|v| v.to_f64()),
            ),
            _ => (
                decision
                    .fair_value
                    .as_ref()
                    .map(|v| {
                        if intent.outcome == BtcOutcome::Up {
                            v.up_probability
                        } else {
                            v.down_probability
                        }
                    })
                    .and_then(|v| v.to_f64()),
                None,
                None,
                None,
            ),
        };
    let probability = probability.context("risk candidate probability unavailable")?;
    let confidence = conservative_probability.unwrap_or_else(|| probability.max(1.0 - probability));
    let seconds_elapsed = (snapshot.observed_at - snapshot.window_start).num_seconds();
    let selected_book = if intent.outcome == BtcOutcome::Up {
        &snapshot.up_book
    } else {
        &snapshot.down_book
    };
    let overround = match (
        snapshot
            .up_book
            .executable_ask_vwap
            .and_then(|v| v.to_f64()),
        snapshot
            .down_book
            .executable_ask_vwap
            .and_then(|v| v.to_f64()),
    ) {
        (Some(up), Some(down)) => up + down - 1.0,
        _ => f64::NAN,
    };
    Ok((
        vec![
            seconds_elapsed as f64,
            probability,
            confidence,
            executable_price.unwrap_or(
                intent
                    .limit_price
                    .to_f64()
                    .context("risk share cost unavailable")?,
            ),
            fee.unwrap_or(0.0),
            intent
                .expected_net_edge_per_share
                .to_f64()
                .context("risk edge unavailable")?,
            if intent.outcome == BtcOutcome::Up {
                1.0
            } else {
                0.0
            },
            overround,
            selected_book
                .imbalance
                .and_then(|v| v.to_f64())
                .unwrap_or(f64::NAN),
            selected_book
                .age_ms
                .map(|v| v as f64 / 1000.0)
                .unwrap_or(f64::NAN),
            snapshot
                .chainlink_gap_bps
                .and_then(|v| v.to_f64())
                .unwrap_or(f64::NAN),
            snapshot
                .realized_volatility
                .and_then(|v| v.to_f64())
                .unwrap_or(f64::NAN),
        ],
        seconds_elapsed,
    ))
}

fn bounded_read(path: &Path) -> Result<Vec<u8>> {
    let metadata = fs::metadata(path)
        .with_context(|| format!("missing risk package file {}", path.display()))?;
    if metadata.len() > 16 * 1024 * 1024 {
        bail!("risk package file exceeds size limit");
    }
    Ok(fs::read(path)?)
}
fn ensure_package_path(root: &Path, package: &Path) -> Result<()> {
    let root = fs::canonicalize(root)?;
    let package = fs::canonicalize(package)?;
    if !package.starts_with(root) {
        bail!("risk package escapes model mount");
    }
    Ok(())
}
fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|v| v.is_ascii_hexdigit() && !v.is_ascii_uppercase())
    {
        bail!("invalid risk artifact sha256");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exported_packages_match_python_golden_predictions() {
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../btc-directional-model/runtime-models");
        let mut checked = 0;
        for entry in fs::read_dir(&root).unwrap() {
            let entry = entry.unwrap();
            let manifest_path = entry.path().join("risk-manifest.json");
            if !manifest_path.exists() {
                continue;
            }
            let manifest: Manifest =
                serde_json::from_slice(&fs::read(manifest_path).unwrap()).unwrap();
            let selection = RiskStrategySelection {
                version: manifest.version,
                model_key: manifest.model_key,
                artifact_sha256: manifest.model_sha256,
            };
            let model = load_from_root(&selection, &root).unwrap();
            let golden: serde_json::Value = serde_json::from_slice(
                &fs::read(entry.path().join("golden-predictions.json")).unwrap(),
            )
            .unwrap();
            for row in golden.as_array().unwrap() {
                let values = row["features"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(f64::NAN))
                    .collect::<Vec<_>>();
                let expected = row["expected_score"].as_f64().unwrap();
                let actual = model.model.score(&values).unwrap();
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "{}: {actual} != {expected}",
                    selection.model_key
                );
            }
            checked += 1;
        }
        assert_eq!(checked, 5);
    }
}
