//! Package registration executes frozen reference cases before publishing readiness.
use crate::btc::directional_model::{BtcDirectionalModelFeatureSnapshot, RuntimeDirectionalModel};
use anyhow::{ensure, Context, Result};
pub(crate) fn verify(model: &RuntimeDirectionalModel, bytes: &[u8]) -> Result<()> {
    let data: serde_json::Value = serde_json::from_slice(bytes)?;
    let cases = data["vectors"]
        .as_array()
        .context("model reference vectors unavailable")?;
    ensure!(
        !cases.is_empty() && cases.len() <= 4096,
        "invalid reference vector count"
    );
    for case in cases {
        let values = case["feature_values"]
            .as_array()
            .context("reference input missing")?
            .iter()
            .map(|v| {
                if v.is_null() {
                    Ok(f64::NAN)
                } else {
                    v.as_f64().context("reference input is not numeric")
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let seconds = case["seconds_elapsed"].as_i64();
        let score = if model.is_asymmetric_value() {
            model.score_asymmetric_value_snapshot(
                &BtcDirectionalModelFeatureSnapshot {
                    model_key: model.model_key().into(),
                    model_artifact_sha256: model.artifact_sha256().into(),
                    feature_schema_version: model.feature_schema_version().into(),
                    feature_schema_sha256: model.feature_schema_sha256().into(),
                    feature_as_of: chrono::Utc::now(),
                    seconds_elapsed: seconds.context("reference clock missing")?,
                    feature_values: values.clone(),
                    input_sha256: "0".repeat(64),
                },
                case["yes_ask_vwap"]
                    .as_f64()
                    .context("reference UP cost missing")?,
                case["no_ask_vwap"]
                    .as_f64()
                    .context("reference DOWN cost missing")?,
            )?
        } else if let Some(seconds) = seconds {
            model.score_at_seconds(&values, seconds)?
        } else {
            model.score(&values)?
        };
        for (name, actual) in [
            ("probability_up", score.probability_up),
            ("confidence", score.confidence),
            ("raw_logit", score.raw_logit),
        ] {
            let expected = case["expected"][name]
                .as_f64()
                .context("reference expected output missing")?;
            ensure!(
                actual.is_finite() && (actual - expected).abs() < 1e-12,
                "model reference parity failed: {} {name}",
                model.model_key()
            );
        }
        ensure!(
            serde_json::to_value(score.action)? == case["expected"]["action"],
            "model reference action parity failed"
        );
        if let Some(adapter) = model.unified_adapter() {
            let result = serde_json::to_value(
                adapter.evaluate(&values, seconds.context("reference clock missing")?)?,
            )?;
            if let Some(expected) = case
                .pointer("/source/admission_expected")
                .and_then(|v| v.as_object())
            {
                for (name, value) in expected {
                    ensure!(
                        (result[name]
                            .as_f64()
                            .context("admission reference output missing")?
                            - value
                                .as_f64()
                                .context("admission reference expected missing")?)
                        .abs()
                            < 1e-12,
                        "model admission reference parity failed: {name}"
                    );
                }
            }
        }
    }
    Ok(())
}
