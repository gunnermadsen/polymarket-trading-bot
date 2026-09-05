//! Process-owned execution evidence. Updates never query storage or change trading eligibility.
use super::types::OrderbookCheckpoint;
use chrono::{DateTime, Utc};
use std::{
    collections::BTreeMap,
    fmt::Write,
    sync::{
        atomic::{AtomicI64, AtomicU64, Ordering::Relaxed},
        Arc, Mutex, OnceLock, Weak,
    },
};
use uuid::Uuid;

static REGISTRY: OnceLock<Mutex<Vec<Weak<ExecutionFreshnessMetrics>>>> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
pub struct BookEvidence {
    pub source_at: DateTime<Utc>,
    pub received_at: DateTime<Utc>,
}

impl From<&OrderbookCheckpoint> for BookEvidence {
    fn from(book: &OrderbookCheckpoint) -> Self {
        Self {
            source_at: book.source_timestamp,
            received_at: book.received_at,
        }
    }
}

impl BookEvidence {
    fn ages(self, at: DateTime<Utc>) -> [i64; 3] {
        [
            (at - self.source_at).num_milliseconds(),
            (at - self.received_at).num_milliseconds(),
            (self.received_at - self.source_at).num_milliseconds(),
        ]
    }
    fn valid(self, at: DateTime<Utc>, limit: i64) -> bool {
        let bound = chrono::Duration::milliseconds(limit);
        self.source_at - at <= bound
            && at - self.source_at <= bound
            && self.received_at <= at
            && at - self.received_at <= bound
            && self.received_at - self.source_at <= bound
    }
}

#[derive(Debug)]
pub struct ExecutionFreshnessMetrics {
    process_id: Uuid,
    mode: &'static str,
    limit_ms: i64,
    attempts: AtomicU64,
    checks: AtomicU64,
    book_rejections: AtomicU64,
    reference_rejections: AtomicU64,
    executions: AtomicU64,
    violations: AtomicU64,
    missing: AtomicU64,
    last_execution_at: AtomicI64,
    last_check_at: AtomicI64,
    last_violation_at: AtomicI64,
    ages: [AtomicI64; 3],
    execution_ages: [AtomicI64; 3],
}

impl ExecutionFreshnessMetrics {
    pub fn new(process_id: Uuid, mode: &'static str, limit_ms: i64) -> Arc<Self> {
        let metrics = Arc::new(Self {
            process_id,
            mode,
            limit_ms,
            attempts: AtomicU64::new(0),
            checks: AtomicU64::new(0),
            book_rejections: AtomicU64::new(0),
            reference_rejections: AtomicU64::new(0),
            executions: AtomicU64::new(0),
            violations: AtomicU64::new(0),
            missing: AtomicU64::new(0),
            last_execution_at: AtomicI64::new(0),
            last_check_at: AtomicI64::new(0),
            last_violation_at: AtomicI64::new(0),
            execution_ages: std::array::from_fn(|_| AtomicI64::new(i64::MIN)),
            ages: std::array::from_fn(|_| AtomicI64::new(i64::MIN)),
        });
        let mut registry = REGISTRY
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        registry.retain(|m| m.strong_count() > 0);
        registry.push(Arc::downgrade(&metrics));
        metrics
    }
    pub fn attempt(&self) {
        self.attempts.fetch_add(1, Relaxed);
    }
    pub fn check(&self, book: Option<BookEvidence>, at: DateTime<Utc>) {
        self.checks.fetch_add(1, Relaxed);
        self.last_check_at.store(at.timestamp(), Relaxed);
        let ages = book.map(|b| b.ages(at)).unwrap_or([i64::MIN; 3]);
        for (target, value) in self.ages.iter().zip(ages) {
            target.store(value, Relaxed);
        }
    }
    pub fn reject_book(&self) {
        self.book_rejections.fetch_add(1, Relaxed);
    }
    pub fn reject_reference(&self) {
        self.reference_rejections.fetch_add(1, Relaxed);
    }

    /// Paper: the actual fill snapshot boundary. Live: immediately before the POST call;
    /// this counts attempts even if the venue rejects or the transport result is ambiguous.
    pub fn execution(&self, book: Option<BookEvidence>, at: DateTime<Utc>, order_id: Uuid) {
        for (target, value) in self
            .execution_ages
            .iter()
            .zip(book.map(|b| b.ages(at)).unwrap_or([i64::MIN; 3]))
        {
            target.store(value, Relaxed);
        }
        self.executions.fetch_add(1, Relaxed);
        self.last_execution_at.store(at.timestamp(), Relaxed);
        if !book.is_some_and(|b| b.valid(at, self.limit_ms)) {
            self.violations.fetch_add(1, Relaxed);
            self.last_violation_at.store(at.timestamp(), Relaxed);
            if book.is_none() {
                self.missing.fetch_add(1, Relaxed);
            }
            tracing::error!(process_id=%self.process_id, execution_mode=self.mode,
                client_order_id=%order_id, max_book_age_ms=self.limit_ms,
                ages_ms=?book.map(|b| b.ages(at)),
                "execution freshness invariant violated");
        }
    }

    #[cfg(test)]
    pub(super) fn counts(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.attempts.load(Relaxed),
            self.executions.load(Relaxed),
            self.violations.load(Relaxed),
            self.book_rejections.load(Relaxed),
            self.reference_rejections.load(Relaxed),
        )
    }

    fn write(&self, out: &mut String) {
        let labels = format!(
            "process_id=\"{}\",execution_mode=\"{}\"",
            self.process_id, self.mode
        );
        for (name, value) in [
            ("monitor_ready", 1),
            ("book_age_limit_milliseconds", self.limit_ms),
            ("attempts_total", self.attempts.load(Relaxed) as i64),
            ("book_checks_total", self.checks.load(Relaxed) as i64),
            (
                "book_rejections_total",
                self.book_rejections.load(Relaxed) as i64,
            ),
            (
                "reference_rejections_total",
                self.reference_rejections.load(Relaxed) as i64,
            ),
            ("executions_total", self.executions.load(Relaxed) as i64),
            ("violations_total", self.violations.load(Relaxed) as i64),
            ("missing_evidence_total", self.missing.load(Relaxed) as i64),
            (
                "last_execution_timestamp_seconds",
                self.last_execution_at.load(Relaxed),
            ),
            (
                "last_check_timestamp_seconds",
                self.last_check_at.load(Relaxed),
            ),
            (
                "last_violation_timestamp_seconds",
                self.last_violation_at.load(Relaxed),
            ),
        ] {
            let _ = writeln!(
                out,
                "polymarket_execution_freshness_{name}{{{labels}}} {value}"
            );
        }
        for (metric, ages) in [
            ("last_book_age_milliseconds", &self.ages),
            ("last_execution_book_age_milliseconds", &self.execution_ages),
        ] {
            for (kind, age) in ["source", "receipt", "source_to_receipt"].iter().zip(ages) {
                let age = age.load(Relaxed);
                if age != i64::MIN {
                    let _ = writeln!(out, "polymarket_execution_freshness_{metric}{{{labels},evidence=\"{kind}\"}} {age}");
                }
            }
        }
    }
}

/// Expected processes come from the manager independently of venue instrumentation. Idle
/// processes have zero counters and ready monitors; absence of orders is not missing telemetry.
pub fn prometheus_metrics(runtime: &serde_json::Value) -> String {
    let mut out = String::from("# HELP polymarket_execution_freshness_exporter_ready Execution freshness exporter is available.\n# TYPE polymarket_execution_freshness_exporter_ready gauge\npolymarket_execution_freshness_exporter_ready 1\n");
    for (name, help) in [
        (
            "attempts_total",
            "Unique order attempts after venue idempotency checks.",
        ),
        (
            "book_checks_total",
            "Canonical book snapshot checks; live can check twice per attempt.",
        ),
        (
            "book_rejections_total",
            "Checks rejected for stale or noncausal book evidence.",
        ),
        (
            "reference_rejections_total",
            "Checks rejected for stale or noncausal reference evidence.",
        ),
        (
            "executions_total",
            "Paper fills or live POST attempts, excluding previews and idempotent replays.",
        ),
        (
            "violations_total",
            "Executions with stale, noncausal or missing canonical book evidence.",
        ),
        (
            "missing_evidence_total",
            "Executions without a matching final book observation.",
        ),
        (
            "monitor_ready",
            "Process venue freshness monitor registered, including while idle.",
        ),
        (
            "expected",
            "Manager expects freshness instrumentation for this active process.",
        ),
        (
            "book_age_limit_milliseconds",
            "Process configured book freshness limit.",
        ),
        (
            "last_book_age_milliseconds",
            "Canonical checkpoint ages at the last book check, not current feed ages.",
        ),
        (
            "last_execution_book_age_milliseconds",
            "Canonical checkpoint ages at the last paper fill or local live POST attempt.",
        ),
        (
            "last_execution_timestamp_seconds",
            "Last paper fill or live POST attempt; zero before first execution.",
        ),
        (
            "last_check_timestamp_seconds",
            "Last book check; zero before first check.",
        ),
        (
            "last_violation_timestamp_seconds",
            "Last execution freshness violation; zero before first violation.",
        ),
    ] {
        let kind = if name.ends_with("_total") {
            "counter"
        } else {
            "gauge"
        };
        let _ = writeln!(out, "# HELP polymarket_execution_freshness_{name} {help}\n# TYPE polymarket_execution_freshness_{name} {kind}");
    }
    if let Some(processes) = runtime.get("processes").and_then(|p| p.as_array()) {
        for process in processes {
            if let (Some(id), Some(mode)) = (
                process
                    .get("process_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| Uuid::parse_str(s).ok()),
                process.get("execution_mode").and_then(|v| v.as_str()),
            ) {
                if matches!(mode, "paper" | "live") {
                    let _ = writeln!(out, "polymarket_execution_freshness_expected{{process_id=\"{id}\",execution_mode=\"{mode}\"}} 1");
                }
            }
        }
    }
    let mut registry = REGISTRY
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut current = BTreeMap::new();
    registry.retain(|weak| {
        if let Some(metrics) = weak.upgrade() {
            current.insert((metrics.process_id, metrics.mode), metrics);
            true
        } else {
            false
        }
    });
    drop(registry);
    for metrics in current.values() {
        metrics.write(&mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    #[test]
    fn executions_detect_stale_missing_and_noncausal_evidence_without_cross_process_effects() {
        let now = Utc::now();
        let a = ExecutionFreshnessMetrics::new(Uuid::new_v4(), "paper", 2000);
        let b = ExecutionFreshnessMetrics::new(Uuid::new_v4(), "live", 2000);
        let fresh = BookEvidence {
            source_at: now - Duration::seconds(2),
            received_at: now,
        };
        a.execution(Some(fresh), now, Uuid::new_v4());
        a.execution(Some(fresh), now + Duration::milliseconds(1), Uuid::new_v4());
        a.execution(None, now, Uuid::new_v4());
        a.execution(
            Some(BookEvidence {
                source_at: now,
                received_at: now + Duration::milliseconds(1),
            }),
            now,
            Uuid::new_v4(),
        );
        assert_eq!(a.executions.load(Relaxed), 4);
        assert_eq!(a.violations.load(Relaxed), 3);
        assert_eq!(a.missing.load(Relaxed), 1);
        assert_eq!(b.violations.load(Relaxed), 0);
        a.execution(Some(fresh), now, Uuid::new_v4());
        assert_eq!(a.violations.load(Relaxed), 3);
    }
    #[test]
    fn idle_restart_and_missing_monitor_are_distinguishable() {
        let id = Uuid::new_v4();
        let runtime = serde_json::json!({"processes":[{"process_id":id,"execution_mode":"paper"}]});
        let m = ExecutionFreshnessMetrics::new(id, "paper", 2000);
        let ready = format!("polymarket_execution_freshness_monitor_ready{{process_id=\"{id}\",execution_mode=\"paper\"}} 1");
        assert!(prometheus_metrics(&runtime).contains(&ready));
        drop(m);
        let missing = prometheus_metrics(&runtime);
        assert!(!missing.contains(&ready));
        assert!(missing.contains(&format!(
            "polymarket_execution_freshness_expected{{process_id=\"{id}\""
        )));
        let _restarted = ExecutionFreshnessMetrics::new(id, "paper", 2000);
        assert!(prometheus_metrics(&runtime).contains(&ready));
    }
}
