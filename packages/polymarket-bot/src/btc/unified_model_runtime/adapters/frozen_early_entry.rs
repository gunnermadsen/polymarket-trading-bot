use super::super::contract::ModelContract;
use crate::btc::directional_model::{
    compile_submodel, RuntimeModelAction, RuntimeModelScore, RuntimeSubmodel, RuntimeSubmodelFile,
};
use anyhow::{bail, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Definition {
    pub contract: ModelContract,
    pub outcome: RuntimeSubmodelFile,
    pub temporal: Vec<RuntimeSubmodelFile>,
    pub admission: Option<AdmissionDefinition>,
    pub policy: Policy,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AdmissionDefinition {
    pub profitable: RuntimeSubmodelFile,
    pub stress_edge: RuntimeSubmodelFile,
    pub loss_severity: RuntimeSubmodelFile,
}
#[derive(Debug, Clone, PartialEq, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub minimum_confidence: Option<f64>,
    pub minimum_edge: Option<f64>,
    pub maximum_share_cost: f64,
    pub minimum_admission_probability: Option<f64>,
    pub minimum_predicted_stress_edge: Option<f64>,
    pub maximum_predicted_loss: Option<f64>,
    pub maximum_temporal_std: Option<f64>,
    pub minimum_temporal_agreement: Option<f64>,
    pub execution_reserve_per_share: f64,
}
#[derive(Debug)]
pub struct Adapter {
    pub contract: ModelContract,
    names: Vec<String>,
    outcome: RuntimeSubmodel,
    temporal: Vec<RuntimeSubmodel>,
    admission: Option<(RuntimeSubmodel, RuntimeSubmodel, RuntimeSubmodel)>,
    policy: Policy,
}
use super::{Evaluation, FeatureContext, FeatureSession, ModelAdapter};
impl Adapter {
    pub(crate) fn compile(definition: Definition, names: &[String]) -> Result<Self> {
        definition.contract.validate()?;
        if names
            .iter()
            .any(|n| !super::feature_names::SUPPORTED.contains(&n.as_str()))
        {
            bail!("unsupported UMR feature capability");
        }
        if definition.contract.qualified_trade_size != Some(5.0) {
            bail!("frozen early-entry size qualification mismatch");
        }
        if definition.contract.adapter != "frozen_early_entry"
            || definition.contract.missing_policy != "native_missing_branch"
        {
            bail!("wrong UMR adapter contract");
        }
        validate_inputs(&definition.contract)?;
        let policy = &definition.policy;
        for value in [
            policy.minimum_confidence,
            policy.minimum_edge,
            Some(policy.maximum_share_cost),
            policy.minimum_admission_probability,
            policy.minimum_predicted_stress_edge,
            policy.maximum_predicted_loss,
            policy.maximum_temporal_std,
            policy.minimum_temporal_agreement,
            Some(policy.execution_reserve_per_share),
        ]
        .into_iter()
        .flatten()
        {
            if !value.is_finite() {
                bail!("nonfinite UMR policy");
            }
        }
        if !(0.0..=1.0).contains(&policy.maximum_share_cost)
            || policy.execution_reserve_per_share < 0.0
            || definition.temporal.len() > 8
        {
            bail!("invalid UMR policy bounds");
        }
        let learned = definition.admission.is_some();
        if learned != policy.minimum_admission_probability.is_some()
            || learned != policy.minimum_predicted_stress_edge.is_some()
            || learned != policy.maximum_predicted_loss.is_some()
            || definition.temporal.is_empty() == policy.maximum_temporal_std.is_some()
            || definition.temporal.is_empty() == policy.minimum_temporal_agreement.is_some()
        {
            bail!("incomplete UMR admission contract");
        }
        for name in ["up_ask_vwap_5", "down_ask_vwap_5", "fee_rate"] {
            if !names.iter().any(|v| v == name) {
                bail!("missing UMR execution feature: {name}");
            }
        }
        let mut derived = names.to_vec();
        for name in DERIVED_NAMES {
            if !derived.iter().any(|v| v == name) {
                derived.push(name.into());
            }
        }
        Ok(Self {
            contract: definition.contract,
            names: names.to_vec(),
            outcome: compile_submodel(definition.outcome, names.len())?,
            temporal: definition
                .temporal
                .into_iter()
                .map(|v| compile_submodel(v, names.len()))
                .collect::<Result<_>>()?,
            admission: definition
                .admission
                .map(|a| {
                    Ok::<_, anyhow::Error>((
                        compile_submodel(a.profitable, derived.len())?,
                        compile_submodel(a.stress_edge, derived.len())?,
                        compile_submodel(a.loss_severity, derived.len())?,
                    ))
                })
                .transpose()?,
            policy: definition.policy,
        })
    }
    pub fn requires_history(&self) -> bool {
        self.admission.is_some()
    }
    pub fn directional_probability(&self, features: &[f64]) -> Result<f64> {
        if features.len() != self.names.len() {
            bail!("UMR feature width mismatch");
        }
        let p = self.outcome.score(features)?;
        if !p.is_finite() {
            bail!("nonfinite UMR prediction");
        }
        Ok(p.clamp(1e-6, 1.0 - 1e-6))
    }
    pub fn evaluate(&self, features: &[f64], seconds: i64) -> Result<Evaluation> {
        if features.len() != self.names.len() || !(60..=89).contains(&seconds) || seconds % 5 != 0 {
            bail!("UMR input width or frozen schedule mismatch");
        }
        let p = self.outcome.score(features)?;
        if !p.is_finite() {
            bail!("nonfinite UMR probability");
        }
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        let get = |name: &str| {
            self.names
                .iter()
                .position(|v| v == name)
                .map(|i| features[i])
                .unwrap_or(f64::NAN)
        };
        let confidence = p.max(1.0 - p);
        let cost = get(if p >= 0.5 {
            "up_ask_vwap_5"
        } else {
            "down_ask_vwap_5"
        });
        let fee_rate = get("fee_rate");
        if !cost.is_finite()
            || !(0.0..=1.0).contains(&cost)
            || !fee_rate.is_finite()
            || fee_rate < 0.0
        {
            bail!("invalid UMR execution inputs");
        }
        let edge = confidence
            - cost
            - fee_rate * cost * (1.0 - cost)
            - self.policy.execution_reserve_per_share;
        let bucket = [0.65, 0.80, 0.95].iter().filter(|&&v| cost > v).count() as f64;
        let mut values = features.to_vec();
        let calculated = [p, confidence, seconds as f64, cost, edge, bucket];
        for (name, value) in DERIVED_NAMES.iter().zip(calculated) {
            if let Some(index) = self.names.iter().position(|v| v == name) {
                values[index] = value;
            } else {
                values.push(value);
            }
        }
        let mut result = Evaluation {
            score: RuntimeModelScore {
                raw_logit: (p / (1.0 - p)).ln(),
                probability_up: p,
                confidence,
                action: if p >= 0.5 {
                    RuntimeModelAction::Up
                } else {
                    RuntimeModelAction::Down
                },
                accepted: true,
            },
            reason: "accepted",
            admission_probability: None,
            predicted_stress_edge: None,
            predicted_loss: None,
            temporal_std: None,
            temporal_agreement: None,
        };
        let mut reject = |condition: bool, reason| {
            if condition && result.reason == "accepted" {
                result.reason = reason;
            }
        };
        reject(cost > self.policy.maximum_share_cost, "share_cost");
        reject(
            self.policy
                .minimum_confidence
                .is_some_and(|v| confidence < v),
            "confidence",
        );
        reject(
            self.policy.minimum_edge.is_some_and(|v| edge < v),
            "expected_edge",
        );
        if let Some((profitable, stress, loss)) = &self.admission {
            let a = profitable.score(&values)?;
            let e = stress.score(&values)?;
            let l = loss.score(&values)?;
            if ![a, e, l].iter().all(|v| v.is_finite()) {
                bail!("nonfinite UMR admission output");
            }
            let l = l.max(0.0);
            reject(
                a < self.policy.minimum_admission_probability.unwrap(),
                "admission_probability",
            );
            reject(
                e < self.policy.minimum_predicted_stress_edge.unwrap(),
                "predicted_stress_edge",
            );
            reject(
                l > self.policy.maximum_predicted_loss.unwrap(),
                "predicted_loss",
            );
            result.admission_probability = Some(a);
            result.predicted_stress_edge = Some(e);
            result.predicted_loss = Some(l);
        }
        if !self.temporal.is_empty() {
            let mut ps = vec![p];
            for model in &self.temporal {
                let value = model.score(features)?;
                if !value.is_finite() {
                    bail!("nonfinite UMR consensus output");
                }
                ps.push(value.clamp(1e-6, 1.0 - 1e-6));
            }
            if !ps.iter().all(|v| v.is_finite()) {
                bail!("nonfinite UMR consensus output");
            }
            let mean = ps.iter().sum::<f64>() / ps.len() as f64;
            let std = (ps.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ps.len() as f64).sqrt();
            let agreement =
                ps.iter().filter(|&&v| (v >= 0.5) == (p >= 0.5)).count() as f64 / ps.len() as f64;
            reject(
                std > self.policy.maximum_temporal_std.unwrap(),
                "temporal_dispersion",
            );
            reject(
                agreement < self.policy.minimum_temporal_agreement.unwrap(),
                "temporal_agreement",
            );
            result.temporal_std = Some(std);
            result.temporal_agreement = Some(agreement);
        }
        result.score.accepted = result.reason == "accepted";
        if !result.score.accepted {
            result.score.action = RuntimeModelAction::NoTrade;
        }
        Ok(result)
    }
}
pub const DERIVED_NAMES: [&str; 6] = [
    "probability",
    "confidence",
    "seconds_elapsed",
    "share_cost",
    "expected_edge",
    "price_bucket_index",
];

impl ModelAdapter for Adapter {
    fn supported_products(&self) -> &'static [&'static str] {
        &[
            "binance_spot_btcusdt_one_second_ohlcv",
            "polymarket_btc_five_minute_orderbooks",
            "polygon_chainlink_btcusd_oracle",
        ]
    }
    fn contract(&self) -> &ModelContract {
        &self.contract
    }
    fn policy(&self) -> serde_json::Value {
        serde_json::to_value(&self.policy).expect("validated finite policy")
    }
    fn evaluate(&self, features: &[f64], seconds: i64) -> Result<Evaluation> {
        Adapter::evaluate(self, features, seconds)
    }
    fn directional_probability(&self, features: &[f64]) -> Result<f64> {
        Adapter::directional_probability(self, features)
    }
    fn requires_history(&self) -> bool {
        Adapter::requires_history(self)
    }
    fn new_session(&self) -> Box<dyn FeatureSession> {
        Box::new(Session::default())
    }
}
#[derive(Default)]
struct Session {
    history: super::history::History,
}
impl FeatureSession for Session {
    fn observe_slot(&mut self, market: &str, seconds: i64) {
        self.history.observe_slot(market, seconds);
    }
    fn prepare(&mut self, adapter: &dyn ModelAdapter, c: &FeatureContext<'_>) -> Result<Vec<f64>> {
        let mut values = super::features::build(
            c.state,
            c.names,
            c.binding,
            c.window_start,
            c.feature_as_of,
            c.up,
            c.down,
            c.fee_rate,
        )?;
        self.history.prepare(
            c.market_id,
            (c.feature_as_of - c.window_start).num_seconds(),
            adapter,
            &mut values,
            c.names,
        )?;
        Ok(values)
    }
}

/// The capability owns recipe semantics; a manifest cannot rename a different
/// source or relax causal age/history requirements by asserting compatibility.
fn validate_inputs(contract: &ModelContract) -> Result<()> {
    for (slot, product, semantics, required, history, age) in [
        (
            "btc_seconds",
            "binance_spot_btcusdt_one_second_ohlcv",
            "binance_closed_seconds_prewindow_open_v1",
            true,
            301,
            5000,
        ),
        (
            "execution_book",
            "polymarket_btc_five_minute_orderbooks",
            "causal_vwap_five_shares_v1",
            true,
            2,
            2000,
        ),
        (
            "oracle",
            "polygon_chainlink_btcusd_oracle",
            "causal_oracle_rounds_v1",
            false,
            600,
            600000,
        ),
        (
            "candles",
            "chainlink_btcusd_one_minute_candles",
            "chainlink_ohlc_close_available_120s_v1",
            false,
            3660,
            120000,
        ),
    ] {
        let input = contract
            .inputs
            .iter()
            .find(|v| v.slot == slot)
            .ok_or_else(|| anyhow::anyhow!("missing frozen input declaration: {slot}"))?;
        if (
            input.product.as_str(),
            input.semantics.as_str(),
            input.required,
            input.lookback_seconds,
            input.maximum_age_ms,
        ) != (product, semantics, required, history, age)
        {
            bail!("frozen input capability mismatch: {slot}");
        }
    }
    for input in &contract.inputs {
        if !["btc_seconds", "execution_book", "oracle", "candles"].contains(&input.slot.as_str()) {
            let expected = match input.slot.as_str() {
                "spot_l2" => (
                    "binance_spot_btcusdt_l2_one_second_features",
                    "frozen_spot_l2_point_in_time_v1",
                ),
                "kraken_l2" => (
                    "kraken_btcusd_l2_updates",
                    "frozen_kraken_l2_point_in_time_v1",
                ),
                _ => bail!("unsupported frozen input slot"),
            };
            if (input.product.as_str(), input.semantics.as_str()) != expected
                || input.required
                || input.lookback_seconds != 60
                || input.maximum_age_ms != 5000
            {
                bail!("incompatible optional L2 input declaration");
            }
        }
    }
    Ok(())
}
