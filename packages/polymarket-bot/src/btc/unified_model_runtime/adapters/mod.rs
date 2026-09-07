//! Capability registration is the only shared edit needed for a new adapter.
//! Trading, telemetry, package discovery and process lifecycle consume these traits.
use super::contract::{ModelContract, ProcessBinding};
use crate::btc::{
    directional_model::RuntimeModelScore,
    types::{OrderbookCheckpoint, RealtimeState},
};
use anyhow::Result;
use chrono::{DateTime, Utc};
pub mod data;
mod feature_names;
pub mod features;
pub mod frozen_early_entry;
pub mod history;
pub mod legacy;
pub(crate) mod validation;

pub struct FeatureContext<'a> {
    pub state: &'a RealtimeState,
    pub binding: &'a ProcessBinding,
    pub market_id: &'a str,
    pub window_start: DateTime<Utc>,
    pub feature_as_of: DateTime<Utc>,
    pub up: &'a OrderbookCheckpoint,
    pub down: &'a OrderbookCheckpoint,
    pub fee_rate: f64,
    pub names: &'a [String],
}
/// Every process receives its own session. No process state belongs in a shared model.
pub trait FeatureSession: Send {
    fn observe_slot(&mut self, market: &str, seconds: i64);
    fn prepare(
        &mut self,
        adapter: &dyn ModelAdapter,
        context: &FeatureContext<'_>,
    ) -> Result<Vec<f64>>;
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct Evaluation {
    pub score: RuntimeModelScore,
    pub reason: &'static str,
    pub admission_probability: Option<f64>,
    pub predicted_stress_edge: Option<f64>,
    pub predicted_loss: Option<f64>,
    pub temporal_std: Option<f64>,
    pub temporal_agreement: Option<f64>,
}
/// Adapters own mathematics and declare input semantics. They cannot submit orders,
/// create streams, change authorization or emit model-specific monitoring schemas.
pub trait ModelAdapter: std::fmt::Debug + Send + Sync {
    fn contract(&self) -> &ModelContract;
    fn supported_products(&self) -> &'static [&'static str];
    fn policy(&self) -> serde_json::Value;
    fn evaluate(&self, features: &[f64], seconds: i64) -> Result<Evaluation>;
    fn directional_probability(&self, features: &[f64]) -> Result<f64>;
    fn requires_history(&self) -> bool;
    fn new_session(&self) -> Box<dyn FeatureSession>;
}
pub(crate) fn compile(
    definition: serde_json::Value,
    names: &[String],
) -> Result<Box<dyn ModelAdapter>> {
    match definition
        .pointer("/contract/adapter")
        .and_then(|v| v.as_str())
    {
        Some("frozen_early_entry") => Ok(Box::new(frozen_early_entry::Adapter::compile(
            serde_json::from_value(definition)?,
            names,
        )?)),
        _ => anyhow::bail!("unsupported UMR adapter capability"),
    }
}
