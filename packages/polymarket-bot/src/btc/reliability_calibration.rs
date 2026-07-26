use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub const SOURCE_PROCESS_ID: Uuid = Uuid::from_u128(0x5635682b4d7947869462668b2618a415);
pub const SOURCE_STRATEGY_VERSION: &str =
    "btc_5m_chainlink_persistence_calibrated_fair_value_research_v1";
pub const SOURCE_PROFILE_ID: &str = "btc5m-chainlink-persistence-calibrated-20260720-v1";
pub const SOURCE_PROFILE_SHA256: &str =
    "238e7ed1b3676a3da61be3b3c8f9d48d7d05cad92e158b921846b9bfe495c1a3";
pub const ANCHOR_SECONDS: [i32; 9] = [30, 60, 90, 120, 150, 180, 210, 240, 270];
pub const MAX_ANCHOR_LAG_SECONDS: f64 = 2.0;
pub const NEUTRAL_SLOPE: f64 = 0.85;
pub const MIN_SLOPE: f64 = 0.35;
pub const MAX_SLOPE: f64 = 1.35;
pub const MAX_ABS_LOGIT: f64 = 5.0;

#[derive(Debug, Clone, Deserialize)]
pub struct CalibrationInputRow {
    pub market_id: String,
    pub decision_id: Uuid,
    pub decision_at: DateTime<Utc>,
    pub window_start: DateTime<Utc>,
    pub label_available_at: DateTime<Utc>,
    pub outcome: String,
    pub action: String,
    #[serde(default)]
    pub reject_reason: Option<String>,
    pub uncalibrated_up_probability: f64,
    pub evidence_reliability: f64,
    pub chainlink_persistence_score: f64,
    pub binance_confirmation_score: f64,
    pub confirmation_deficiency_score: f64,
    pub realized_volatility: f64,
    pub seconds_to_close: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CalibrationSample {
    pub market_id: String,
    pub decision_id: Uuid,
    pub decision_at: DateTime<Utc>,
    pub window_start: DateTime<Utc>,
    pub anchor_seconds: i32,
    pub outcome_up: bool,
    pub action: String,
    pub reject_reason: Option<String>,
    pub base_logit: f64,
    pub reliability: f64,
    pub confirmed_quality: f64,
    pub confirmation_deficiency: f64,
    pub volatility_excess: f64,
    pub late_window_pressure: f64,
    pub market_weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationCoefficients {
    pub slope_intercept_adjustment: f64,
    pub reliability_weight: f64,
    pub confirmed_quality_weight: f64,
    pub confirmation_deficiency_penalty: f64,
    pub volatility_excess_penalty: f64,
    pub late_window_penalty: f64,
}

impl Default for CalibrationCoefficients {
    fn default() -> Self {
        Self {
            slope_intercept_adjustment: 0.0,
            reliability_weight: 0.0,
            confirmed_quality_weight: 0.0,
            confirmation_deficiency_penalty: 0.0,
            volatility_excess_penalty: 0.0,
            late_window_penalty: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CalibrationMetrics {
    pub markets: usize,
    pub samples: usize,
    pub log_loss: f64,
    pub brier_score: f64,
    pub accuracy: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CalibrationSplitReport {
    pub baseline: CalibrationMetrics,
    pub calibrated: CalibrationMetrics,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CalibrationFitReport {
    pub source_process_id: Uuid,
    pub source_strategy_version: String,
    pub source_profile_id: String,
    pub source_profile_sha256: String,
    pub generated_at: DateTime<Utc>,
    pub first_window_start: DateTime<Utc>,
    pub last_window_start: DateTime<Utc>,
    pub anchors_seconds: Vec<i32>,
    pub maximum_anchor_lag_seconds: f64,
    pub dataset_sha256: String,
    pub total_markets: usize,
    pub total_samples: usize,
    pub outcome_up_markets: usize,
    pub outcome_down_markets: usize,
    pub action_counts: BTreeMap<String, usize>,
    pub training_markets: usize,
    pub validation_markets: usize,
    pub holdout_markets: usize,
    pub regularization: f64,
    pub coefficients: CalibrationCoefficients,
    pub training: CalibrationSplitReport,
    pub validation: CalibrationSplitReport,
    pub holdout: CalibrationSplitReport,
    pub holdout_accepted: bool,
}

pub fn build_anchored_dataset(
    rows: impl IntoIterator<Item = CalibrationInputRow>,
) -> anyhow::Result<Vec<CalibrationSample>> {
    let mut selected: BTreeMap<(String, i32), CalibrationInputRow> = BTreeMap::new();
    for row in rows {
        validate_input_row(&row)?;
        if row.decision_at >= row.label_available_at {
            continue;
        }
        let elapsed = (row.decision_at - row.window_start).num_milliseconds() as f64 / 1_000.0;
        let Some(anchor) = ANCHOR_SECONDS.iter().copied().find(|anchor| {
            elapsed <= f64::from(*anchor) && f64::from(*anchor) - elapsed <= MAX_ANCHOR_LAG_SECONDS
        }) else {
            continue;
        };
        let key = (row.market_id.clone(), anchor);
        let replace = selected.get(&key).map_or(true, |current| {
            row.decision_at > current.decision_at
                || (row.decision_at == current.decision_at
                    && row.decision_id.as_bytes() < current.decision_id.as_bytes())
        });
        if replace {
            selected.insert(key, row);
        }
    }

    let mut counts = BTreeMap::<String, usize>::new();
    for (market_id, _) in selected.keys() {
        *counts.entry(market_id.clone()).or_default() += 1;
    }
    let mut samples = Vec::with_capacity(selected.len());
    for ((market_id, anchor_seconds), row) in selected {
        let confirmed_quality = (row.chainlink_persistence_score.max(0.0)
            * row.binance_confirmation_score.max(0.0)
            / 9.0)
            .clamp(0.0, 1.0);
        let confirmation_deficiency = (row.confirmation_deficiency_score / 3.0).clamp(0.0, 1.0);
        let volatility_excess =
            ((row.realized_volatility / 0.00012 - 1.0).max(0.0) / 3.0).clamp(0.0, 1.0);
        let late_window_pressure = (1.0 - row.seconds_to_close / 120.0).clamp(0.0, 1.0);
        samples.push(CalibrationSample {
            market_weight: 1.0 / *counts.get(&market_id).expect("selected market count") as f64,
            market_id,
            decision_id: row.decision_id,
            decision_at: row.decision_at,
            window_start: row.window_start,
            anchor_seconds,
            outcome_up: row.outcome.eq_ignore_ascii_case("up"),
            action: row.action,
            reject_reason: row.reject_reason,
            base_logit: logit(row.uncalibrated_up_probability).clamp(-MAX_ABS_LOGIT, MAX_ABS_LOGIT),
            reliability: row.evidence_reliability,
            confirmed_quality,
            confirmation_deficiency,
            volatility_excess,
            late_window_pressure,
        });
    }
    samples.sort_by(|left, right| {
        left.window_start
            .cmp(&right.window_start)
            .then_with(|| left.market_id.cmp(&right.market_id))
            .then_with(|| left.anchor_seconds.cmp(&right.anchor_seconds))
            .then_with(|| left.decision_id.cmp(&right.decision_id))
    });
    anyhow::ensure!(!samples.is_empty(), "calibration dataset is empty");
    Ok(samples)
}

fn validate_input_row(row: &CalibrationInputRow) -> anyhow::Result<()> {
    anyhow::ensure!(!row.market_id.trim().is_empty(), "market_id is empty");
    anyhow::ensure!(
        row.outcome.eq_ignore_ascii_case("up") || row.outcome.eq_ignore_ascii_case("down"),
        "outcome must be up or down"
    );
    for (name, value) in [
        (
            "uncalibrated_up_probability",
            row.uncalibrated_up_probability,
        ),
        ("evidence_reliability", row.evidence_reliability),
        (
            "chainlink_persistence_score",
            row.chainlink_persistence_score,
        ),
        ("binance_confirmation_score", row.binance_confirmation_score),
        (
            "confirmation_deficiency_score",
            row.confirmation_deficiency_score,
        ),
        ("realized_volatility", row.realized_volatility),
        ("seconds_to_close", row.seconds_to_close),
    ] {
        anyhow::ensure!(value.is_finite(), "{name} must be finite");
    }
    anyhow::ensure!(
        row.uncalibrated_up_probability > 0.0 && row.uncalibrated_up_probability < 1.0,
        "uncalibrated_up_probability must be inside (0,1)"
    );
    anyhow::ensure!(
        (0.0..=1.0).contains(&row.evidence_reliability),
        "evidence_reliability must be inside [0,1]"
    );
    anyhow::ensure!(
        row.realized_volatility > 0.0,
        "realized_volatility must be positive"
    );
    anyhow::ensure!(
        row.seconds_to_close > 0.0,
        "seconds_to_close must be positive"
    );
    Ok(())
}

pub fn conditional_slope(
    coefficients: CalibrationCoefficients,
    reliability: f64,
    confirmed_quality: f64,
    confirmation_deficiency: f64,
    volatility_excess: f64,
    late_window_pressure: f64,
) -> f64 {
    let theta = inverse_softplus(NEUTRAL_SLOPE)
        + coefficients.slope_intercept_adjustment
        + coefficients.reliability_weight * (2.0 * (reliability - 0.5)).clamp(-1.0, 1.0)
        + coefficients.confirmed_quality_weight * confirmed_quality.clamp(0.0, 1.0)
        - coefficients.confirmation_deficiency_penalty * confirmation_deficiency.clamp(0.0, 1.0)
        - coefficients.volatility_excess_penalty * volatility_excess.clamp(0.0, 1.0)
        - coefficients.late_window_penalty * late_window_pressure.clamp(0.0, 1.0);
    softplus(theta).clamp(MIN_SLOPE, MAX_SLOPE)
}

pub fn calibrate_probability(
    probability: f64,
    coefficients: CalibrationCoefficients,
    reliability: f64,
    confirmed_quality: f64,
    confirmation_deficiency: f64,
    volatility_excess: f64,
    late_window_pressure: f64,
) -> f64 {
    logistic(
        conditional_slope(
            coefficients,
            reliability,
            confirmed_quality,
            confirmation_deficiency,
            volatility_excess,
            late_window_pressure,
        ) * logit(probability).clamp(-MAX_ABS_LOGIT, MAX_ABS_LOGIT),
    )
}

pub fn fit(samples: &[CalibrationSample]) -> anyhow::Result<CalibrationFitReport> {
    anyhow::ensure!(!samples.is_empty(), "calibration dataset is empty");
    let market_order = ordered_markets(samples);
    anyhow::ensure!(
        market_order.len() >= 30,
        "at least 30 independent markets are required"
    );
    let train_end = (market_order.len() * 60) / 100;
    let validation_end = (market_order.len() * 80) / 100;
    let training_ids: BTreeSet<&str> = market_order[..train_end]
        .iter()
        .map(String::as_str)
        .collect();
    let validation_ids: BTreeSet<&str> = market_order[train_end..validation_end]
        .iter()
        .map(String::as_str)
        .collect();
    let holdout_ids: BTreeSet<&str> = market_order[validation_end..]
        .iter()
        .map(String::as_str)
        .collect();
    let training = select_markets(samples, &training_ids);
    let validation = select_markets(samples, &validation_ids);
    let holdout = select_markets(samples, &holdout_ids);

    let mut selected: Option<(f64, CalibrationCoefficients, (f64, f64, f64))> = None;
    for regularization in [0.001, 0.01, 0.1, 1.0, 10.0] {
        let coefficients = optimize(&training, regularization);
        let result = metrics(&validation, coefficients);
        let rank = (result.log_loss, result.brier_score, regularization);
        if selected.as_ref().map_or(true, |(_, _, current)| {
            compare_metric_rank(rank, *current) == Ordering::Less
        }) {
            selected = Some((regularization, coefficients, rank));
        }
    }
    let (regularization, _, _) = selected.expect("regularization grid is non-empty");
    let mut training_and_validation = training.clone();
    training_and_validation.extend(validation.iter().copied());
    let coefficients = optimize(&training_and_validation, regularization);
    let baseline = CalibrationCoefficients::default();
    let training_report = split_report(&training, baseline, coefficients);
    let validation_report = split_report(&validation, baseline, coefficients);
    let holdout_report = split_report(&holdout, baseline, coefficients);
    let holdout_accepted = holdout_report.calibrated.log_loss <= holdout_report.baseline.log_loss
        && holdout_report.calibrated.brier_score <= holdout_report.baseline.brier_score;

    let mut action_counts = BTreeMap::new();
    for sample in samples {
        *action_counts.entry(sample.action.clone()).or_default() += 1;
    }
    let outcomes = samples
        .iter()
        .map(|sample| (sample.market_id.as_str(), sample.outcome_up))
        .collect::<BTreeMap<_, _>>();
    Ok(CalibrationFitReport {
        source_process_id: SOURCE_PROCESS_ID,
        source_strategy_version: SOURCE_STRATEGY_VERSION.to_string(),
        source_profile_id: SOURCE_PROFILE_ID.to_string(),
        source_profile_sha256: SOURCE_PROFILE_SHA256.to_string(),
        generated_at: Utc::now(),
        first_window_start: samples.first().expect("non-empty samples").window_start,
        last_window_start: samples.last().expect("non-empty samples").window_start,
        anchors_seconds: ANCHOR_SECONDS.to_vec(),
        maximum_anchor_lag_seconds: MAX_ANCHOR_LAG_SECONDS,
        dataset_sha256: dataset_sha256(samples)?,
        total_markets: market_order.len(),
        total_samples: samples.len(),
        outcome_up_markets: outcomes.values().filter(|outcome| **outcome).count(),
        outcome_down_markets: outcomes.values().filter(|outcome| !**outcome).count(),
        action_counts,
        training_markets: training_ids.len(),
        validation_markets: validation_ids.len(),
        holdout_markets: holdout_ids.len(),
        regularization,
        coefficients,
        training: training_report,
        validation: validation_report,
        holdout: holdout_report,
        holdout_accepted,
    })
}

fn optimize(samples: &[&CalibrationSample], regularization: f64) -> CalibrationCoefficients {
    let mut parameters = [0.0_f64; 6];
    let mut first_moment = [0.0_f64; 6];
    let mut second_moment = [0.0_f64; 6];
    let total_weight = samples
        .iter()
        .map(|sample| sample.market_weight)
        .sum::<f64>();
    for iteration in 1..=4_000 {
        let mut gradient = [0.0_f64; 6];
        for sample in samples {
            let inputs = [
                1.0,
                (2.0 * (sample.reliability - 0.5)).clamp(-1.0, 1.0),
                sample.confirmed_quality,
                -sample.confirmation_deficiency,
                -sample.volatility_excess,
                -sample.late_window_pressure,
            ];
            let theta = inverse_softplus(NEUTRAL_SLOPE)
                + parameters
                    .iter()
                    .zip(inputs)
                    .map(|(parameter, input)| parameter * input)
                    .sum::<f64>();
            let raw_slope = softplus(theta);
            let slope = raw_slope.clamp(MIN_SLOPE, MAX_SLOPE);
            let predicted = logistic(slope * sample.base_logit);
            let slope_derivative = if (MIN_SLOPE..MAX_SLOPE).contains(&raw_slope) {
                logistic(theta)
            } else {
                0.0
            };
            let common = sample.market_weight
                * (predicted - f64::from(sample.outcome_up))
                * sample.base_logit
                * slope_derivative;
            for (index, input) in inputs.into_iter().enumerate() {
                gradient[index] += common * input;
            }
        }
        for index in 0..parameters.len() {
            gradient[index] =
                gradient[index] / total_weight + 2.0 * regularization * parameters[index];
            first_moment[index] = 0.9 * first_moment[index] + 0.1 * gradient[index];
            second_moment[index] =
                0.999 * second_moment[index] + 0.001 * gradient[index] * gradient[index];
            let first = first_moment[index] / (1.0 - 0.9_f64.powi(iteration));
            let second = second_moment[index] / (1.0 - 0.999_f64.powi(iteration));
            parameters[index] -= 0.02 * first / (second.sqrt() + 1e-8);
        }
        for parameter in &mut parameters[1..] {
            *parameter = parameter.max(0.0);
        }
    }
    CalibrationCoefficients {
        slope_intercept_adjustment: parameters[0],
        reliability_weight: parameters[1],
        confirmed_quality_weight: parameters[2],
        confirmation_deficiency_penalty: parameters[3],
        volatility_excess_penalty: parameters[4],
        late_window_penalty: parameters[5],
    }
}

fn split_report(
    samples: &[&CalibrationSample],
    baseline: CalibrationCoefficients,
    calibrated: CalibrationCoefficients,
) -> CalibrationSplitReport {
    CalibrationSplitReport {
        baseline: metrics(samples, baseline),
        calibrated: metrics(samples, calibrated),
    }
}

fn metrics(
    samples: &[&CalibrationSample],
    coefficients: CalibrationCoefficients,
) -> CalibrationMetrics {
    let markets = samples
        .iter()
        .map(|sample| sample.market_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let total_weight = samples
        .iter()
        .map(|sample| sample.market_weight)
        .sum::<f64>();
    let mut log_loss_sum = 0.0;
    let mut brier_sum = 0.0;
    let mut correct_sum = 0.0;
    for sample in samples {
        let probability = logistic(
            conditional_slope(
                coefficients,
                sample.reliability,
                sample.confirmed_quality,
                sample.confirmation_deficiency,
                sample.volatility_excess,
                sample.late_window_pressure,
            ) * sample.base_logit,
        )
        .clamp(1e-12, 1.0 - 1e-12);
        let outcome = f64::from(sample.outcome_up);
        log_loss_sum += sample.market_weight
            * -(outcome * probability.ln() + (1.0 - outcome) * (1.0 - probability).ln());
        brier_sum += sample.market_weight * (probability - outcome).powi(2);
        correct_sum += sample.market_weight * f64::from((probability >= 0.5) == sample.outcome_up);
    }
    CalibrationMetrics {
        markets,
        samples: samples.len(),
        log_loss: log_loss_sum / total_weight,
        brier_score: brier_sum / total_weight,
        accuracy: correct_sum / total_weight,
    }
}

fn compare_metric_rank(left: (f64, f64, f64), right: (f64, f64, f64)) -> Ordering {
    left.0
        .total_cmp(&right.0)
        .then_with(|| left.1.total_cmp(&right.1))
        .then_with(|| right.2.total_cmp(&left.2))
}

fn ordered_markets(samples: &[CalibrationSample]) -> Vec<String> {
    let mut markets = BTreeMap::<String, DateTime<Utc>>::new();
    for sample in samples {
        markets
            .entry(sample.market_id.clone())
            .and_modify(|at| *at = (*at).min(sample.window_start))
            .or_insert(sample.window_start);
    }
    let mut order = markets.into_iter().collect::<Vec<_>>();
    order.sort_by(|(left_id, left_at), (right_id, right_at)| {
        left_at.cmp(right_at).then_with(|| left_id.cmp(right_id))
    });
    order.into_iter().map(|(market_id, _)| market_id).collect()
}

fn select_markets<'a>(
    samples: &'a [CalibrationSample],
    market_ids: &BTreeSet<&str>,
) -> Vec<&'a CalibrationSample> {
    samples
        .iter()
        .filter(|sample| market_ids.contains(sample.market_id.as_str()))
        .collect()
}

fn dataset_sha256(samples: &[CalibrationSample]) -> anyhow::Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(samples)?)
    ))
}

fn logit(probability: f64) -> f64 {
    (probability / (1.0 - probability)).ln()
}

fn logistic(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn softplus(value: f64) -> f64 {
    if value > 30.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn inverse_softplus(value: f64) -> f64 {
    value.exp_m1().ln()
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    fn row(market: usize, second: i64, probability: f64, up: bool) -> CalibrationInputRow {
        let window_start = Utc.with_ymd_and_hms(2026, 7, 1, 0, 0, 0).unwrap()
            + chrono::Duration::minutes(5 * market as i64);
        CalibrationInputRow {
            market_id: format!("market-{market:03}"),
            decision_id: Uuid::from_u128((market * 1_000 + second as usize) as u128),
            decision_at: window_start + chrono::Duration::seconds(second),
            window_start,
            label_available_at: window_start + chrono::Duration::minutes(6),
            outcome: if up { "up" } else { "down" }.to_string(),
            action: "no_trade".to_string(),
            reject_reason: Some("insufficient_net_edge".to_string()),
            uncalibrated_up_probability: probability,
            evidence_reliability: 0.75,
            chainlink_persistence_score: 1.0,
            binance_confirmation_score: 1.0,
            confirmation_deficiency_score: 0.25,
            realized_volatility: 0.00012,
            seconds_to_close: 300.0 - second as f64,
        }
    }

    #[test]
    fn anchor_selection_is_causal_and_deterministic() {
        let older = row(1, 29, 0.60, true);
        let mut nearest = row(1, 30, 0.70, true);
        nearest.decision_id = Uuid::from_u128(1);
        let mut tie = nearest.clone();
        tie.decision_id = Uuid::from_u128(2);
        let after_anchor = row(1, 31, 0.99, true);
        let samples = build_anchored_dataset([after_anchor, tie, older, nearest.clone()]).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].decision_id, nearest.decision_id);
    }

    #[test]
    fn label_available_at_prevents_leakage() {
        let mut leaked = row(1, 30, 0.70, true);
        leaked.label_available_at = leaked.decision_at;
        assert!(build_anchored_dataset([leaked]).is_err());
    }

    #[test]
    fn neutral_calibration_matches_original_slope_and_is_symmetric() {
        for probability in [0.10, 0.30, 0.49, 0.51, 0.70, 0.90] {
            let up = calibrate_probability(
                probability,
                CalibrationCoefficients::default(),
                0.8,
                0.4,
                0.2,
                0.1,
                0.3,
            );
            let mirrored = calibrate_probability(
                1.0 - probability,
                CalibrationCoefficients::default(),
                0.8,
                0.4,
                0.2,
                0.1,
                0.3,
            );
            assert!((up - logistic(NEUTRAL_SLOPE * logit(probability))).abs() < 1e-12);
            assert!((up + mirrored - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn favorable_evidence_increases_and_adverse_evidence_reduces_confidence() {
        let coefficients = CalibrationCoefficients {
            slope_intercept_adjustment: 0.0,
            reliability_weight: 0.3,
            confirmed_quality_weight: 0.4,
            confirmation_deficiency_penalty: 0.5,
            volatility_excess_penalty: 0.4,
            late_window_penalty: 0.2,
        };
        let favorable = calibrate_probability(0.75, coefficients, 0.95, 1.0, 0.0, 0.0, 0.0);
        let adverse = calibrate_probability(0.75, coefficients, 0.10, 0.0, 1.0, 1.0, 1.0);
        assert!(favorable > adverse);
        assert!(favorable > 0.5 && adverse > 0.5);
    }

    #[test]
    fn fitting_and_dataset_hash_are_deterministic() {
        let rows = (0..60)
            .flat_map(|market| {
                let up = market % 2 == 0;
                [30, 60, 90]
                    .map(move |second| row(market, second, if up { 0.68 } else { 0.32 }, up))
            })
            .collect::<Vec<_>>();
        let samples = build_anchored_dataset(rows.clone()).unwrap();
        let repeated = build_anchored_dataset(rows).unwrap();
        assert_eq!(
            dataset_sha256(&samples).unwrap(),
            dataset_sha256(&repeated).unwrap()
        );
        assert_eq!(
            fit(&samples).unwrap().coefficients,
            fit(&samples).unwrap().coefficients
        );
    }
}
