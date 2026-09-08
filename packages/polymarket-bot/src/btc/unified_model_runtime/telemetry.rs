//! Stable, bounded process-scoped operational telemetry. No database reads on scrape.
use super::contract::EVALUATION_VERSION;
use crate::btc::directional_model::{RuntimeModelScore, RuntimeModelSelection};
use chrono::{DateTime, Utc};
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{Mutex, OnceLock},
    time::Instant,
};
use uuid::Uuid;

const MAX_PROCESSES: usize = 256;
const MAX_PENDING: usize = 2048;
const LATENCY_BUCKETS: [f64; 9] = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0];
#[derive(Clone, Default)]
struct Histogram {
    buckets: [u64; 9],
    count: u64,
    sum: f64,
}
impl Histogram {
    fn observe(&mut self, value: f64) {
        if !value.is_finite() || value < 0.0 {
            return;
        }
        self.count += 1;
        self.sum += value;
        for (i, b) in LATENCY_BUCKETS.iter().enumerate() {
            if value <= *b {
                self.buckets[i] += 1;
            }
        }
    }
}
#[derive(Clone, Serialize)]
pub struct PredictionRecord {
    pub contract_version: &'static str,
    pub process_id: Uuid,
    pub model: RuntimeModelSelection,
    pub run_id: Option<Uuid>,
    pub config_hash: String,
    pub execution_mode: String,
    pub feature_snapshot_id: Uuid,
    pub feature_as_of: DateTime<Utc>,
    pub input_sha256: String,
    pub score: RuntimeModelScore,
    pub inference_seconds: f64,
    pub admission: Option<serde_json::Value>,
}
#[derive(Clone)]
struct Pending {
    market: String,
    record: PredictionRecord,
}
#[derive(Clone, Default)]
struct Process {
    identity: Option<RuntimeModelSelection>,
    run_id: Option<Uuid>,
    config_hash: String,
    mode: String,
    enabled: bool,
    ready: bool,
    last_observation: f64,
    last_success: f64,
    counters: BTreeMap<(&'static str, String), u64>,
    gauges: BTreeMap<&'static str, f64>,
    histograms: BTreeMap<&'static str, Histogram>,
    latest: Option<PredictionRecord>,
    records: VecDeque<PredictionRecord>,
    pending: VecDeque<Pending>,
    last_market: String,
    last_eligible_market: String,
    last_inferred_market: String,
    last_admitted_market: String,
    calibration_count: [u64; 10],
    calibration_sum: [f64; 10],
    calibration_outcomes: [u64; 10],
}
#[derive(Default)]
struct Registry {
    processes: HashMap<Uuid, Process>,
    dropped: u64,
}
static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
fn registry() -> &'static Mutex<Registry> {
    REGISTRY.get_or_init(Default::default)
}
fn update(id: Uuid, f: impl FnOnce(&mut Process)) {
    let Ok(mut r) = registry().lock() else {
        return;
    };
    if !r.processes.contains_key(&id) && r.processes.len() >= MAX_PROCESSES {
        r.dropped += 1;
        return;
    }
    f(r.processes.entry(id).or_default());
}
fn increment(p: &mut Process, metric: &'static str, reason: &str) {
    *p.counters.entry((metric, reason.into())).or_default() += 1;
}
pub fn register(
    id: Uuid,
    run: Uuid,
    config: &str,
    mode: &str,
    selection: Option<&RuntimeModelSelection>,
) {
    update(id, |p| {
        if p.identity.as_ref() != selection || p.run_id != Some(run) || p.config_hash != config {
            *p = Process::default();
        }
        p.identity = selection.cloned();
        p.run_id = Some(run);
        p.config_hash = config.into();
        p.mode = mode.into();
        p.enabled = true;
        for (metric, reasons) in [
            ("opportunities", &["scheduled"][..]),
            (
                "book_input_failures",
                &[
                    "missing_snapshot",
                    "identity_mismatch",
                    "epoch_mismatch",
                    "invalid_integrity",
                    "future_timestamp",
                    "stale_source_timestamp",
                    "stale_received_timestamp",
                ][..],
            ),
            ("inferences", &["success", "error"][..]),
            ("feature_builds", &["success", "error"][..]),
            ("model_admission", &["accepted", "rejected"][..]),
            ("execution", &["filled", "rejected", "submitted"][..]),
            ("prediction_outcomes", &["correct", "incorrect"][..]),
            ("trade_outcomes", &["win", "loss", "push"][..]),
            (
                "markets",
                &["observed", "eligible", "inferred", "admitted"][..],
            ),
            ("telemetry_dropped", &["pending_prediction_capacity"][..]),
        ] {
            for reason in reasons {
                p.counters.entry((metric, (*reason).into())).or_default();
            }
        }
    });
    tracing::info!(event="umr_model_registered",process_id=%id,run_id=%run,config_hash=config,model_key=selection.map(|v|v.model_key.as_str()),"UMR process model registered");
}
pub fn enabled(id: Uuid, value: bool) {
    update(id, |p| {
        p.enabled = value;
        if !value {
            p.ready = false;
        }
    });
}
pub fn event(id: Uuid, metric: &'static str, reason: &str) {
    update(id, |p| increment(p, metric, reason));
}
/// Classify errors without placing arbitrary diagnostic text in metric labels.
pub fn failure(id: Uuid, stage: &'static str, detail: &str) {
    let lower = detail.to_ascii_lowercase();
    let reason = if lower.contains("schema") || lower.contains("width") {
        "schema_mismatch"
    } else if lower.contains("nonfinite") || lower.contains("non_finite") {
        "non_finite_output"
    } else if lower.contains("opening") || lower.contains("pre-window") {
        "opening_reference_unavailable"
    } else if lower.contains("book") {
        "orderbook_unavailable"
    } else if lower.contains("history") {
        "history_unavailable"
    } else if lower.contains("stale") || lower.contains("age") {
        "stale_source_data"
    } else if lower.contains("unavailable") || lower.contains("missing") {
        "missing_source_data"
    } else {
        "invalid_value"
    };
    event(
        id,
        if stage == "features" {
            "feature_failures"
        } else {
            "inference_failures"
        },
        reason,
    );
    update(id, |p| {
        tracing::warn!(event="umr_evaluation_failed",process_id=%id,run_id=?p.run_id,model_key=p.identity.as_ref().map(|v|v.model_key.as_str()),execution_mode=%p.mode,stage,reason,detail,"UMR evaluation unavailable");
    });
}
pub fn eligible_market(id: Uuid, market: &str) {
    update(id, |p| {
        if p.last_eligible_market != market {
            increment(p, "markets", "eligible");
            p.last_eligible_market = market.into();
        }
    });
}
pub fn duration(id: Uuid, stage: &'static str, value: f64) {
    update(id, |p| {
        p.histograms.entry(stage).or_default().observe(value)
    });
}
pub fn gauge(id: Uuid, name: &'static str, value: f64) {
    if value.is_finite() {
        update(id, |p| {
            p.gauges.insert(name, value);
        });
    }
}
pub fn readiness(id: Uuid, value: bool) {
    update(id, |p| {
        if p.ready != value {
            tracing::info!(event="umr_readiness_changed",process_id=%id,ready=value,"UMR process readiness changed");
        }
        p.ready = value;
    });
}
pub fn observation(id: Uuid, market: Option<&str>) {
    update(id, |p| {
        p.last_observation = Utc::now().timestamp_millis() as f64 / 1000.0;
        increment(p, "observations", "observed");
        if let Some(m) = market {
            if p.last_market != m {
                increment(p, "markets", "observed");
                p.last_market = m.into();
            }
        }
    });
}
#[allow(clippy::too_many_arguments)]
pub fn prediction(
    id: Uuid,
    snapshot_id: Uuid,
    market: &str,
    model: &RuntimeModelSelection,
    at: DateTime<Utc>,
    input: &str,
    score: RuntimeModelScore,
    seconds: f64,
    admission: Option<serde_json::Value>,
) {
    update(id, |p| {
        if p.latest
            .as_ref()
            .is_some_and(|r| r.feature_as_of == at && r.input_sha256 == input && r.model == *model)
        {
            return;
        }
        let record = PredictionRecord {
            contract_version: EVALUATION_VERSION,
            process_id: id,
            model: model.clone(),
            run_id: p.run_id,
            config_hash: p.config_hash.clone(),
            execution_mode: p.mode.clone(),
            feature_snapshot_id: snapshot_id,
            feature_as_of: at,
            input_sha256: input.into(),
            score,
            inference_seconds: seconds,
            admission,
        };
        p.last_success = Utc::now().timestamp_millis() as f64 / 1000.0;
        increment(p, "inferences", "success");
        if p.last_inferred_market != market {
            increment(p, "markets", "inferred");
            p.last_inferred_market = market.into();
        }
        if score.accepted && p.last_admitted_market != market {
            increment(p, "markets", "admitted");
            p.last_admitted_market = market.into();
        }
        increment(
            p,
            "probability_bin",
            &((score.probability_up * 10.0) as usize).min(9).to_string(),
        );
        increment(
            p,
            "confidence_bin",
            &((score.confidence * 10.0) as usize).min(9).to_string(),
        );
        increment(
            p,
            "actions",
            if !score.accepted {
                "abstain"
            } else if score.probability_up >= 0.5 {
                "up"
            } else {
                "down"
            },
        );
        if let Some(a) = record.admission.as_ref() {
            if let Some(reason) = a.get("reason").and_then(|v| v.as_str()) {
                increment(p, "admission_reasons", reason);
            }
            for (field, metric) in [
                ("admission_probability", "admission_probability"),
                ("predicted_stress_edge", "predicted_stress_edge"),
                ("predicted_loss", "predicted_loss"),
                ("temporal_std", "temporal_std"),
                ("temporal_agreement", "temporal_agreement"),
            ] {
                if let Some(v) = a.get(field).and_then(|v| v.as_f64()) {
                    p.gauges.insert(metric, v);
                }
            }
        }
        increment(
            p,
            "predictions",
            if score.probability_up >= 0.5 {
                "up"
            } else {
                "down"
            },
        );
        increment(
            p,
            "model_admission",
            if score.accepted {
                "accepted"
            } else {
                "rejected"
            },
        );
        p.histograms
            .entry("inference")
            .or_default()
            .observe(seconds);
        p.gauges.insert("probability_up", score.probability_up);
        p.gauges.insert("confidence", score.confidence);
        if p.pending.len() >= MAX_PENDING {
            p.pending.pop_front();
            increment(p, "telemetry_dropped", "pending_prediction_capacity");
        }
        p.pending.push_back(Pending {
            market: market.into(),
            record: record.clone(),
        });
        p.records.push_back(record.clone());
        while p.records.len() > 64 {
            p.records.pop_front();
        }
        p.latest = Some(record);
    });
}
pub fn prediction_record(id: Uuid, snapshot_id: Uuid) -> Option<PredictionRecord> {
    let r = registry().lock().ok()?;
    r.processes
        .get(&id)?
        .records
        .iter()
        .find(|v| v.feature_snapshot_id == snapshot_id)
        .cloned()
}
/// Resolutions are official observed facts. Remove pending predictions once so repeated
/// delivery cannot double count. Counters cover this instrumentation session only.
pub fn resolve(market: &str, up: bool) {
    let Ok(mut r) = registry().lock() else {
        return;
    };
    for p in r.processes.values_mut() {
        let mut retained = VecDeque::new();
        while let Some(item) = p.pending.pop_front() {
            if item.market != market {
                retained.push_back(item);
                continue;
            }
            let probability = item.record.score.probability_up;
            let correct = (probability >= 0.5) == up;
            increment(
                p,
                "prediction_outcomes",
                if correct { "correct" } else { "incorrect" },
            );
            *p.gauges.entry("brier_sum").or_default() += (probability - f64::from(up)).powi(2);
            *p.gauges.entry("brier_count").or_default() += 1.0;
            let bin = ((probability * 10.0) as usize).min(9);
            p.calibration_count[bin] += 1;
            p.calibration_sum[bin] += probability;
            p.calibration_outcomes[bin] += u64::from(up);
        }
        p.pending = retained;
    }
}
pub fn fill(id: Uuid, price: f64, size: f64, fee: f64, entry_second: f64, quoted: f64) {
    update(id, |p| {
        increment(p, "fills", "observed");
        for (name, value) in [
            ("fill_notional_usd", price * size),
            ("filled_shares", size),
            ("fill_fees_usd", fee),
            ("entry_seconds_sum", entry_second),
            ("entry_fill_count", 1.0),
            ("slippage_notional_usd", (price - quoted) * size),
        ] {
            *p.gauges.entry(name).or_default() += value;
        }
    });
}
pub fn settlement(id: Uuid, pnl: f64, fees: f64) {
    update(id, |p| {
        increment(
            p,
            "trade_outcomes",
            if pnl > 0.0 {
                "win"
            } else if pnl < 0.0 {
                "loss"
            } else {
                "push"
            },
        );
        let net = p.gauges.entry("realized_pnl_usd").or_default();
        *net += pnl;
        let net = *net;
        *p.gauges.entry("fees_usd").or_default() += fees;
        let high = p.gauges.entry("equity_high_usd").or_default();
        *high = high.max(net);
        let dd = *high - net;
        let max = p.gauges.entry("max_drawdown_usd").or_default();
        *max = max.max(dd);
        *p.gauges
            .entry(if pnl >= 0.0 {
                "gross_profit_usd"
            } else {
                "gross_loss_usd"
            })
            .or_default() += pnl.abs();
    });
}
pub struct ObservationGuard {
    id: Uuid,
    start: Instant,
    success: bool,
}
impl ObservationGuard {
    pub fn new(id: Uuid) -> Self {
        Self {
            id,
            start: Instant::now(),
            success: false,
        }
    }
    pub fn complete(&mut self) {
        self.success = true;
    }
}
impl Drop for ObservationGuard {
    fn drop(&mut self) {
        duration(self.id, "observation", self.start.elapsed().as_secs_f64());
        if !self.success {
            event(self.id, "observations_failed", "error");
        }
    }
}
fn escaped(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}
pub fn prometheus_metrics() -> String {
    use std::fmt::Write;
    let Ok(r) = registry().lock() else {
        return String::new();
    };
    let processes = r
        .processes
        .iter()
        .map(|(id, p)| {
            let mut gauges = p.gauges.clone();
            gauges.insert("pending_predictions", p.pending.len() as f64);
            (
                *id,
                Process {
                    identity: p.identity.clone(),
                    run_id: p.run_id,
                    config_hash: p.config_hash.clone(),
                    mode: p.mode.clone(),
                    enabled: p.enabled,
                    ready: p.ready,
                    last_observation: p.last_observation,
                    last_success: p.last_success,
                    counters: p.counters.clone(),
                    gauges,
                    histograms: p.histograms.clone(),
                    calibration_count: p.calibration_count,
                    calibration_sum: p.calibration_sum,
                    calibration_outcomes: p.calibration_outcomes,
                    ..Default::default()
                },
            )
        })
        .collect::<Vec<_>>();
    let dropped = r.dropped;
    drop(r);
    let mut out = String::new();
    let _=writeln!(out,"# HELP polymarket_umr_registry_dropped_total Process registrations exceeding telemetry capacity.\n# TYPE polymarket_umr_registry_dropped_total counter\npolymarket_umr_registry_dropped_total {dropped}");
    let mut declared = std::collections::HashSet::new();
    for (id, p) in processes {
        let labels = format!("process_id=\"{id}\"");
        for (name, value) in [
            ("enabled", f64::from(p.enabled)),
            ("runtime_ready", f64::from(p.ready)),
            ("last_observation_timestamp_seconds", p.last_observation),
            ("last_success_timestamp_seconds", p.last_success),
        ]
        .into_iter()
        .chain(p.gauges)
        {
            if declared.insert(name.to_string()) {
                let _=writeln!(out,"# HELP polymarket_umr_{name} UMR process {name}; session-scoped unless stated otherwise.\n# TYPE polymarket_umr_{name} gauge");
            }
            let _ = writeln!(out, "polymarket_umr_{name}{{{labels}}} {value}");
        }
        if let Some(m) = p.identity {
            if declared.insert("model_info".into()) {
                out.push_str("# HELP polymarket_umr_model_info Immutable active model and process configuration identity.\n# TYPE polymarket_umr_model_info gauge\n");
            }
            let _=writeln!(out,"polymarket_umr_model_info{{{labels},model_key=\"{}\",artifact_sha256=\"{}\",feature_schema_sha256=\"{}\",execution_mode=\"{}\",config_hash=\"{}\"}} 1",escaped(&m.model_key),escaped(&m.artifact_sha256),escaped(&m.feature_schema_sha256),escaped(&p.mode),escaped(&p.config_hash));
        }
        for ((name, reason), value) in p.counters {
            if declared.insert(format!("{name}_total")) {
                let _=writeln!(out,"# HELP polymarket_umr_{name}_total UMR {name} events by bounded reason.\n# TYPE polymarket_umr_{name}_total counter");
            }
            let _ = writeln!(
                out,
                "polymarket_umr_{name}_total{{{labels},reason=\"{}\"}} {value}",
                escaped(&reason)
            );
        }
        for (stage, h) in p.histograms {
            if declared.insert("stage_duration_seconds".into()) {
                out.push_str("# HELP polymarket_umr_stage_duration_seconds UMR stage latency in seconds.\n# TYPE polymarket_umr_stage_duration_seconds histogram\n");
            }
            for (i, b) in LATENCY_BUCKETS.iter().enumerate() {
                let _=writeln!(out,"polymarket_umr_stage_duration_seconds_bucket{{{labels},stage=\"{stage}\",le=\"{b}\"}} {}",h.buckets[i]);
            }
            let _=writeln!(out,"polymarket_umr_stage_duration_seconds_bucket{{{labels},stage=\"{stage}\",le=\"+Inf\"}} {}\npolymarket_umr_stage_duration_seconds_sum{{{labels},stage=\"{stage}\"}} {}\npolymarket_umr_stage_duration_seconds_count{{{labels},stage=\"{stage}\"}} {}",h.count,h.sum,h.count);
        }
        for bin in 0..10 {
            for (name, value) in [
                ("calibration_count", p.calibration_count[bin] as f64),
                ("calibration_probability_sum", p.calibration_sum[bin]),
                (
                    "calibration_up_outcomes",
                    p.calibration_outcomes[bin] as f64,
                ),
            ] {
                if declared.insert(name.into()) {
                    let _=writeln!(out,"# HELP polymarket_umr_{name} Resolved evaluation-weighted calibration bin {name}.\n# TYPE polymarket_umr_{name} gauge");
                }
                let _ = writeln!(
                    out,
                    "polymarket_umr_{name}{{{labels},bin=\"{bin}\"}} {value}"
                );
            }
        }
    }
    out
}
