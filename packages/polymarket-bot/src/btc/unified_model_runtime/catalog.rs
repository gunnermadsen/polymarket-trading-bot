//! Read-only discovery. A mounted package never activates a process by itself.
use super::risk::{self, RiskStrategySelection};
use crate::btc::directional_model::{runtime_model_registry, RuntimeModelSelection};
use anyhow::{Context, Result};
use serde::Serialize;
#[derive(Serialize)]
pub struct Entry {
    pub kind: String,
    pub model_key: String,
    pub selection: Option<RuntimeModelSelection>,
    pub risk_selection: Option<RiskStrategySelection>,
    pub compatible: bool,
    pub error: Option<String>,
    pub adapter: Option<String>,
    pub contract: Option<super::contract::ModelContract>,
    pub feature_schema_version: Option<String>,
    pub policy: Option<serde_json::Value>,
    pub schedule: Option<serde_json::Value>,
    pub supported_products: Vec<String>,
}
pub fn discover() -> Result<Vec<Entry>> {
    let registry = runtime_model_registry();
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(registry.root())
        .context("UMR model mount is unavailable")?
        .take(512)
    {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let key = entry.file_name().to_string_lossy().to_string();
        if key.starts_with('.') {
            continue;
        }
        let mut result = Entry {
            kind: "trade_strategy".into(),
            model_key: key.clone(),
            selection: None,
            risk_selection: None,
            compatible: false,
            error: None,
            adapter: None,
            contract: None,
            feature_schema_version: None,
            policy: None,
            schedule: None,
            supported_products: Vec::new(),
        };
        let risk_manifest_path = entry.path().join("risk-manifest.json");
        if risk_manifest_path.exists() {
            result.kind = "risk_strategy".into();
            let loaded = (|| -> Result<RiskStrategySelection> {
                if std::fs::metadata(&risk_manifest_path)?.len() > 1024 * 1024 {
                    anyhow::bail!("UMR risk manifest exceeds size limit");
                }
                let manifest: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&risk_manifest_path)?)?;
                let selection = RiskStrategySelection {
                    version: manifest["version"]
                        .as_str()
                        .context("missing risk version")?
                        .into(),
                    model_key: key.clone(),
                    artifact_sha256: manifest["model_sha256"]
                        .as_str()
                        .context("missing risk model checksum")?
                        .into(),
                };
                risk::load(&selection)?;
                Ok(selection)
            })();
            match loaded {
                Ok(selection) => {
                    result.risk_selection = Some(selection);
                    result.compatible = true;
                    result.adapter = Some("candidate_loss_risk".into());
                    result.feature_schema_version = Some(risk::RISK_FEATURE_SCHEMA_VERSION.into());
                }
                Err(error) => result.error = Some(error.to_string()),
            }
            entries.push(result);
            continue;
        }
        let loaded = (|| -> Result<_> {
            let path = entry.path().join("manifest.json");
            if std::fs::metadata(&path)?.len() > 1024 * 1024 {
                anyhow::bail!("UMR manifest exceeds size limit");
            }
            let m: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
            let selection = RuntimeModelSelection {
                model_key: key,
                artifact_sha256: m["model_sha256"]
                    .as_str()
                    .context("missing model checksum")?
                    .into(),
                feature_schema_sha256: m["feature_schema_sha256"]
                    .as_str()
                    .context("missing feature checksum")?
                    .into(),
            };
            let model = registry.load(&selection)?;
            Ok((selection, model))
        })();
        match loaded {
            Ok((selection, model)) => {
                result.selection = Some(selection);
                result.compatible = true;
                result.feature_schema_version = Some(model.feature_schema_version().into());
                let p = model.prediction_policy();
                result.schedule = Some(
                    serde_json::json!({"minimum_seconds_after_open":p.minimum_seconds_after_open,"maximum_seconds_after_open":p.maximum_seconds_after_open,"cadence_seconds":p.cadence_seconds}),
                );
                if let Some(adapter) = model.unified_adapter() {
                    result.adapter = Some(adapter.contract().adapter.clone());
                    result.contract = Some(adapter.contract().clone());
                    result.policy = Some(adapter.policy());
                    result.supported_products = adapter
                        .supported_products()
                        .iter()
                        .map(|v| (*v).into())
                        .collect();
                } else {
                    result.adapter = Some("legacy_directional".into());
                    result.contract = Some(super::adapters::legacy::contract(&model));
                }
            }
            Err(error) => result.error = Some(error.to_string()),
        }
        entries.push(result);
    }
    entries.sort_by(|a, b| a.model_key.cmp(&b.model_key));
    Ok(entries)
}
