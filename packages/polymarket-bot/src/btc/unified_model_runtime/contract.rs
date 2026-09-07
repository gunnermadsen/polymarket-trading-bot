use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const CONTRACT_VERSION: &str = "capitonic-unified-model-runtime-v1";
pub const EVALUATION_VERSION: &str = "capitonic-model-evaluation-v1";

/// Mathematical requirements are immutable artifact content. Process bindings may
/// satisfy these requirements, but cannot redefine their meaning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelContract {
    pub version: String,
    pub adapter: String,
    pub adapter_version: u32,
    pub inputs: Vec<InputContract>,
    pub probability_semantics: String,
    pub feature_clock: String,
    pub missing_policy: String,
    pub qualified_trade_size: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputContract {
    pub slot: String,
    pub product: String,
    pub semantics: String,
    pub required: bool,
    pub lookback_seconds: u32,
    pub maximum_age_ms: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceBinding {
    pub slot: String,
    pub product: String,
    pub semantics: String,
}

impl ModelContract {
    pub fn validate(&self) -> Result<()> {
        if self.version != CONTRACT_VERSION
            || self.adapter_version != 1
            || self.adapter.is_empty()
            || self.probability_semantics != "probability_up"
            || self.feature_clock != "closed_binance_second_as_of"
            || !matches!(
                self.missing_policy.as_str(),
                "legacy_preserved" | "native_missing_branch"
            )
            || self
                .qualified_trade_size
                .is_some_and(|v| !v.is_finite() || v <= 0.0)
        {
            bail!("unsupported UMR contract or capability");
        }
        let mut slots = HashSet::new();
        for input in &self.inputs {
            if input.slot.is_empty()
                || input.product.is_empty()
                || input.semantics.is_empty()
                || input.maximum_age_ms == 0
                || !slots.insert(&input.slot)
            {
                bail!("invalid or duplicate UMR input contract");
            }
        }
        Ok(())
    }

    pub fn validate_bindings(&self, bindings: &[SourceBinding], size: f64) -> Result<()> {
        self.validate()?;
        if self.qualified_trade_size.is_some_and(|v| size != v) {
            bail!("trade size differs from frozen model qualification");
        }
        let mut seen = HashSet::new();
        for binding in bindings {
            let input = self
                .inputs
                .iter()
                .find(|v| v.slot == binding.slot)
                .ok_or_else(|| anyhow::anyhow!("unknown UMR input slot: {}", binding.slot))?;
            if !seen.insert(&binding.slot)
                || input.product != binding.product
                || input.semantics != binding.semantics
            {
                bail!("incompatible UMR source binding: {}", binding.slot);
            }
        }
        for input in &self.inputs {
            if input.required && !seen.contains(&input.slot) {
                bail!("missing required UMR binding: {}", input.slot);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessBinding {
    pub version: String,
    pub sources: Vec<SourceBinding>,
    pub policy: serde_json::Value,
}
impl ProcessBinding {
    pub fn validate(&self, adapter: &dyn super::adapters::ModelAdapter, size: f64) -> Result<()> {
        if self.version != CONTRACT_VERSION || self.policy != adapter.policy() {
            bail!("UMR process policy differs from frozen qualification");
        }
        adapter.contract().validate_bindings(&self.sources, size)?;
        for source in &self.sources {
            if !adapter
                .supported_products()
                .contains(&source.product.as_str())
            {
                bail!(
                    "UMR product has no compatible realtime feature adapter: {}",
                    source.product
                );
            }
        }
        Ok(())
    }
}
