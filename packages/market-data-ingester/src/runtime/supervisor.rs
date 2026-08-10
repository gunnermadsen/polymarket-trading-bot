use std::{collections::BTreeMap, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use async_trait::async_trait;
use sqlx::PgPool;
use tokio::{
    sync::watch,
    task::{JoinError, JoinHandle},
    time::{Instant, MissedTickBehavior},
};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::{
    domain::{DesiredState, IngesterProfile, IngesterStrategyKey, StrategyError},
    persistence::ProfileRepository,
};

use super::StrategyRegistry;

const BASE_RETRY_DELAY: Duration = Duration::from_secs(5);
const MAX_RETRY_DELAY: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, Copy)]
pub(crate) struct SupervisorSettings {
    pub reconcile_interval: Duration,
    pub lease_duration: Duration,
    pub strategy_shutdown_timeout: Duration,
}

impl Default for SupervisorSettings {
    fn default() -> Self {
        Self {
            reconcile_interval: Duration::from_secs(1),
            lease_duration: Duration::from_secs(15),
            strategy_shutdown_timeout: Duration::from_secs(30),
        }
    }
}

impl SupervisorSettings {
    fn validate(self) -> Result<Self> {
        if self.reconcile_interval.is_zero() {
            anyhow::bail!("strategy reconciliation interval must be positive");
        }
        if self.strategy_shutdown_timeout.is_zero() {
            anyhow::bail!("strategy shutdown timeout must be positive");
        }
        let minimum_lease = self
            .reconcile_interval
            .checked_mul(3)
            .context("strategy reconciliation interval is too large")?;
        if self.lease_duration < minimum_lease {
            anyhow::bail!("strategy lease duration must span at least three reconciliations");
        }
        Ok(self)
    }
}

#[async_trait]
pub(crate) trait ProfileStore: Clone + Send + Sync + 'static {
    async fn list(&self) -> Result<Vec<IngesterProfile>>;

    async fn claim_lease(
        &self,
        key: IngesterStrategyKey,
        generation: i64,
        owner: &str,
        lease_duration: Duration,
    ) -> Result<Option<(IngesterProfile, Uuid)>>;

    async fn renew_lease(
        &self,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        lease_duration: Duration,
    ) -> Result<bool>;

    async fn mark_running(
        &self,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        generation: i64,
    ) -> Result<bool>;

    async fn mark_stopped(
        &self,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        generation: i64,
    ) -> Result<bool>;

    async fn mark_failed(
        &self,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        error_code: &str,
        error_message: &str,
    ) -> Result<bool>;
}

#[async_trait]
impl ProfileStore for ProfileRepository {
    async fn list(&self) -> Result<Vec<IngesterProfile>> {
        ProfileRepository::list(self).await
    }

    async fn claim_lease(
        &self,
        key: IngesterStrategyKey,
        generation: i64,
        owner: &str,
        lease_duration: Duration,
    ) -> Result<Option<(IngesterProfile, Uuid)>> {
        ProfileRepository::claim_lease(self, key, generation, owner, lease_duration).await
    }

    async fn renew_lease(
        &self,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        lease_duration: Duration,
    ) -> Result<bool> {
        ProfileRepository::renew_lease(self, key, owner, token, lease_duration).await
    }

    async fn mark_running(
        &self,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        generation: i64,
    ) -> Result<bool> {
        ProfileRepository::mark_running(self, key, owner, token, generation).await
    }

    async fn mark_stopped(
        &self,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        generation: i64,
    ) -> Result<bool> {
        ProfileRepository::mark_stopped(self, key, owner, token, generation).await
    }

    async fn mark_failed(
        &self,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        error_code: &str,
        error_message: &str,
    ) -> Result<bool> {
        ProfileRepository::mark_failed(self, key, owner, token, error_code, error_message).await
    }
}

pub(crate) struct StrategySupervisor<S = ProfileRepository> {
    profiles: S,
    registry: StrategyRegistry,
    pool: PgPool,
    instance: Arc<str>,
    settings: SupervisorSettings,
    active: BTreeMap<IngesterStrategyKey, ActiveStrategy>,
    retry_after: BTreeMap<IngesterStrategyKey, RetryAfter>,
    failure_streaks: BTreeMap<IngesterStrategyKey, FailureStreak>,
}

impl<S> StrategySupervisor<S>
where
    S: ProfileStore,
{
    pub fn new(
        profiles: S,
        registry: StrategyRegistry,
        pool: PgPool,
        instance: impl Into<Arc<str>>,
        settings: SupervisorSettings,
    ) -> Result<Self> {
        Ok(Self {
            profiles,
            registry,
            pool,
            instance: instance.into(),
            settings: settings.validate()?,
            active: BTreeMap::new(),
            retry_after: BTreeMap::new(),
            failure_streaks: BTreeMap::new(),
        })
    }

    pub async fn run(
        mut self,
        shutdown: CancellationToken,
        readiness: watch::Sender<bool>,
    ) -> Result<()> {
        let mut ticker = tokio::time::interval(self.settings.reconcile_interval);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => break,
                _ = ticker.tick() => {
                    match self.reconcile_once().await {
                        Ok(()) => {
                            let _ = readiness.send(true);
                        }
                        Err(error) => {
                            let _ = readiness.send(false);
                            error!(error = %error, "market-data ingester reconciliation failed");
                            self.request_stop_all(StopReason::LeaseSafety);
                        }
                    }
                }
            }
        }

        let _ = readiness.send(false);
        self.shutdown_active().await;
        Ok(())
    }

    async fn reconcile_once(&mut self) -> Result<()> {
        self.collect_finished().await?;
        self.abort_expired_stops().await?;

        let profiles = self
            .profiles
            .list()
            .await
            .context("failed to load ingester profiles for reconciliation")?;
        let profiles: BTreeMap<_, _> = profiles
            .into_iter()
            .map(|profile| (profile.strategy_key, profile))
            .collect();

        self.reconcile_active(&profiles).await?;
        self.collect_finished().await?;
        self.abort_expired_stops().await?;

        for profile in profiles.values() {
            if profile.desired_state != DesiredState::Running
                || self.active.contains_key(&profile.strategy_key)
                || self.retry_is_active(profile.strategy_key, profile.desired_generation)
            {
                continue;
            }
            self.start_profile(profile).await?;
        }

        Ok(())
    }

    async fn reconcile_active(
        &mut self,
        profiles: &BTreeMap<IngesterStrategyKey, IngesterProfile>,
    ) -> Result<()> {
        let active = self
            .active
            .iter()
            .map(|(key, running)| {
                (
                    *key,
                    running.generation,
                    running.lease_token,
                    running.next_renewal,
                )
            })
            .collect::<Vec<_>>();

        for (key, generation, lease_token, next_renewal) in active {
            if self
                .active
                .get(&key)
                .is_some_and(|running| running.stop.is_some())
            {
                continue;
            }

            let Some(profile) = profiles.get(&key) else {
                self.request_stop(key, StopReason::LeaseSafety);
                continue;
            };
            if profile.desired_state != DesiredState::Running
                || profile.desired_generation != generation
            {
                self.request_stop(key, StopReason::DesiredStateChanged);
                continue;
            }
            if profile.lease_owner.as_deref() != Some(self.instance.as_ref())
                || profile.lease_token != Some(lease_token)
                || !profile.lease_is_current(chrono::Utc::now())
            {
                self.request_stop(key, StopReason::LeaseSafety);
                continue;
            }
            if Instant::now() < next_renewal {
                continue;
            }

            let renewed = self
                .profiles
                .renew_lease(
                    key,
                    &self.instance,
                    lease_token,
                    self.settings.lease_duration,
                )
                .await
                .with_context(|| format!("failed to renew lease for strategy {key}"))?;
            if !renewed {
                self.request_stop(key, StopReason::LeaseSafety);
            } else if let Some(active) = self.active.get_mut(&key) {
                active.next_renewal = Instant::now() + self.settings.lease_duration / 3;
            }
        }
        Ok(())
    }

    async fn start_profile(&mut self, requested: &IngesterProfile) -> Result<()> {
        let Some((profile, lease_token)) = self
            .profiles
            .claim_lease(
                requested.strategy_key,
                requested.desired_generation,
                &self.instance,
                self.settings.lease_duration,
            )
            .await
            .with_context(|| {
                format!(
                    "failed to claim lease for strategy {}",
                    requested.strategy_key
                )
            })?
        else {
            return Ok(());
        };

        if let Err(error) = self.registry.validate_profile(&profile) {
            self.fail_start(&profile, lease_token, "invalid_profile", &error.to_string())
                .await?;
            return Ok(());
        }
        let Some(factory) = self.registry.factory(profile.strategy_key) else {
            self.fail_start(
                &profile,
                lease_token,
                "unsupported_strategy",
                "strategy is not compiled into this service",
            )
            .await?;
            return Ok(());
        };
        let strategy = match factory.build(&profile, self.pool.clone()) {
            Ok(strategy) => strategy,
            Err(error) => {
                self.fail_start(
                    &profile,
                    lease_token,
                    "strategy_construction_failed",
                    &error.to_string(),
                )
                .await?;
                return Ok(());
            }
        };
        if strategy.key() != profile.strategy_key {
            self.fail_start(
                &profile,
                lease_token,
                "strategy_identity_mismatch",
                "strategy factory returned a different strategy key",
            )
            .await?;
            return Ok(());
        }

        let marked_running = self
            .profiles
            .mark_running(
                profile.strategy_key,
                &self.instance,
                lease_token,
                profile.desired_generation,
            )
            .await
            .with_context(|| format!("failed to mark strategy {} running", profile.strategy_key))?;
        if !marked_running {
            warn!(strategy = %profile.strategy_key, "strategy lease was lost before startup");
            return Ok(());
        }

        let cancellation = CancellationToken::new();
        let strategy_cancellation = cancellation.clone();
        let key = profile.strategy_key;
        let task = tokio::spawn(async move { strategy.run(strategy_cancellation).await });
        self.active.insert(
            key,
            ActiveStrategy {
                generation: profile.desired_generation,
                lease_token,
                cancellation,
                task,
                stop: None,
                next_renewal: Instant::now() + self.settings.lease_duration / 3,
                stored_failure_count: profile.consecutive_failures.max(0) as u32,
            },
        );
        self.retry_after.remove(&key);
        info!(strategy = %key, generation = profile.desired_generation, "ingester strategy started");
        Ok(())
    }

    async fn fail_start(
        &mut self,
        profile: &IngesterProfile,
        lease_token: Uuid,
        code: &str,
        message: &str,
    ) -> Result<()> {
        let persisted = self
            .profiles
            .mark_failed(
                profile.strategy_key,
                &self.instance,
                lease_token,
                code,
                message,
            )
            .await
            .with_context(|| {
                format!(
                    "failed to persist startup failure for {}",
                    profile.strategy_key
                )
            })?;
        if persisted {
            self.defer_retry(
                profile.strategy_key,
                profile.desired_generation,
                profile.consecutive_failures.max(0) as u32,
            );
        }
        warn!(
            strategy = %profile.strategy_key,
            error_code = code,
            error = message,
            "ingester strategy startup rejected"
        );
        Ok(())
    }

    fn request_stop(&mut self, key: IngesterStrategyKey, reason: StopReason) {
        let Some(active) = self.active.get_mut(&key) else {
            return;
        };
        if active.stop.is_none() {
            active.cancellation.cancel();
            active.stop = Some(StopRequest {
                reason,
                deadline: Instant::now() + self.settings.strategy_shutdown_timeout,
            });
            info!(strategy = %key, ?reason, "ingester strategy stop requested");
        }
    }

    fn request_stop_all(&mut self, reason: StopReason) {
        for active in self.active.values_mut() {
            if active.stop.is_none() {
                active.cancellation.cancel();
                active.stop = Some(StopRequest {
                    reason,
                    deadline: Instant::now() + self.settings.strategy_shutdown_timeout,
                });
            }
        }
    }

    async fn collect_finished(&mut self) -> Result<()> {
        let finished = self
            .active
            .iter()
            .filter_map(|(key, active)| active.task.is_finished().then_some(*key))
            .collect::<Vec<_>>();
        for key in finished {
            self.finish(key).await?;
        }
        Ok(())
    }

    async fn abort_expired_stops(&mut self) -> Result<()> {
        let now = Instant::now();
        let expired = self
            .active
            .iter()
            .filter_map(|(key, active)| {
                active
                    .stop
                    .filter(|stop| stop.deadline <= now)
                    .map(|_| *key)
            })
            .collect::<Vec<_>>();
        for key in expired {
            if let Some(active) = self.active.get(&key) {
                active.task.abort();
                warn!(strategy = %key, "ingester strategy exceeded its shutdown deadline");
            }
            self.finish(key).await?;
        }
        Ok(())
    }

    async fn finish(&mut self, key: IngesterStrategyKey) -> Result<()> {
        let Some(active) = self.active.remove(&key) else {
            return Ok(());
        };
        let ActiveStrategy {
            generation,
            lease_token,
            task,
            stop,
            stored_failure_count,
            ..
        } = active;
        let outcome = task.await;

        if let Some(stop) = stop {
            let _ = self
                .profiles
                .mark_stopped(key, &self.instance, lease_token, generation)
                .await
                .with_context(|| format!("failed to persist stop state for strategy {key}"))?;
            info!(strategy = %key, reason = ?stop.reason, "ingester strategy stopped");
            return Ok(());
        }

        let (code, message) = unexpected_exit(&outcome);
        let persisted = self
            .profiles
            .mark_failed(key, &self.instance, lease_token, code, &message)
            .await
            .with_context(|| format!("failed to persist failure state for strategy {key}"))?;
        if persisted {
            self.defer_retry(key, generation, stored_failure_count);
        }
        warn!(strategy = %key, error_code = code, error = %message, "ingester strategy exited unexpectedly");
        Ok(())
    }

    fn defer_retry(
        &mut self,
        key: IngesterStrategyKey,
        generation: i64,
        stored_failure_count: u32,
    ) {
        let previous = self
            .failure_streaks
            .get(&key)
            .filter(|streak| streak.generation == generation)
            .map_or(0, |streak| streak.failures);
        let failures = previous.max(stored_failure_count).saturating_add(1);
        self.failure_streaks.insert(
            key,
            FailureStreak {
                generation,
                failures,
            },
        );
        self.retry_after.insert(
            key,
            RetryAfter {
                generation,
                instant: Instant::now() + retry_delay(key, failures),
            },
        );
    }

    fn retry_is_active(&mut self, key: IngesterStrategyKey, generation: i64) -> bool {
        let Some(retry) = self.retry_after.get(&key).copied() else {
            return false;
        };
        if retry.generation != generation || retry.instant <= Instant::now() {
            self.retry_after.remove(&key);
            false
        } else {
            true
        }
    }

    async fn shutdown_active(&mut self) {
        self.request_stop_all(StopReason::ServiceShutdown);
        let deadline = Instant::now() + self.settings.strategy_shutdown_timeout;

        while !self.active.is_empty() && Instant::now() < deadline {
            if let Err(error) = self.collect_finished().await {
                error!(error = %error, "failed to persist strategy state during shutdown");
            }
            if !self.active.is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }

        let remaining = self.active.keys().copied().collect::<Vec<_>>();
        for key in remaining {
            if let Some(active) = self.active.get(&key) {
                active.task.abort();
            }
            if let Err(error) = self.finish(key).await {
                error!(strategy = %key, error = %error, "failed to finalize strategy shutdown");
            }
        }
    }
}

struct ActiveStrategy {
    generation: i64,
    lease_token: Uuid,
    cancellation: CancellationToken,
    task: JoinHandle<Result<(), StrategyError>>,
    stop: Option<StopRequest>,
    next_renewal: Instant,
    stored_failure_count: u32,
}

#[derive(Debug, Clone, Copy)]
struct StopRequest {
    reason: StopReason,
    deadline: Instant,
}

#[derive(Debug, Clone, Copy)]
enum StopReason {
    DesiredStateChanged,
    LeaseSafety,
    ServiceShutdown,
}

#[derive(Debug, Clone, Copy)]
struct RetryAfter {
    generation: i64,
    instant: Instant,
}

#[derive(Debug, Clone, Copy)]
struct FailureStreak {
    generation: i64,
    failures: u32,
}

fn retry_delay(key: IngesterStrategyKey, failures: u32) -> Duration {
    let exponent = failures.saturating_sub(1).min(6);
    let multiplier = 1_u32 << exponent;
    let exponential = BASE_RETRY_DELAY
        .checked_mul(multiplier)
        .unwrap_or(MAX_RETRY_DELAY)
        .min(MAX_RETRY_DELAY);
    let jitter_milliseconds = key.as_str().bytes().fold(0_u64, |hash, byte| {
        hash.wrapping_mul(31).wrapping_add(u64::from(byte))
    }) % 1_000;
    exponential + Duration::from_millis(jitter_milliseconds)
}

fn unexpected_exit(
    outcome: &Result<Result<(), StrategyError>, JoinError>,
) -> (&'static str, String) {
    match outcome {
        Ok(Ok(())) => (
            "strategy_exited",
            "strategy returned without a shutdown request".to_owned(),
        ),
        Ok(Err(error)) => (error.code, error.to_string()),
        Err(error) if error.is_panic() => (
            "strategy_panicked",
            "strategy task panicked; details are available in service logs".to_owned(),
        ),
        Err(error) if error.is_cancelled() => (
            "strategy_cancelled",
            "strategy task was cancelled without a shutdown request".to_owned(),
        ),
        Err(error) => ("strategy_join_failed", error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    };

    use async_trait::async_trait;
    use chrono::Utc;
    use serde_json::{json, Value};
    use sqlx::postgres::PgPoolOptions;

    use crate::{
        domain::{HealthStatus, IngesterStrategy, ObservedState, StrategyError},
        runtime::{StrategyFactory, StrategyFactoryError},
    };

    use super::*;

    const KEY: IngesterStrategyKey = IngesterStrategyKey::BinanceSpotBtcusdtAggregateTrades;

    #[derive(Clone)]
    struct MemoryStore {
        profile: Arc<Mutex<IngesterProfile>>,
        renewals: Arc<AtomicUsize>,
    }

    impl MemoryStore {
        fn new(profile: IngesterProfile) -> Self {
            Self {
                profile: Arc::new(Mutex::new(profile)),
                renewals: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn update(&self, update: impl FnOnce(&mut IngesterProfile)) {
            update(&mut self.profile.lock().expect("profile lock"));
        }

        fn snapshot(&self) -> IngesterProfile {
            self.profile.lock().expect("profile lock").clone()
        }
    }

    #[async_trait]
    impl ProfileStore for MemoryStore {
        async fn list(&self) -> Result<Vec<IngesterProfile>> {
            Ok(vec![self.snapshot()])
        }

        async fn claim_lease(
            &self,
            key: IngesterStrategyKey,
            generation: i64,
            owner: &str,
            lease_duration: Duration,
        ) -> Result<Option<(IngesterProfile, Uuid)>> {
            let mut profile = self.profile.lock().expect("profile lock");
            if profile.strategy_key != key
                || profile.desired_state != DesiredState::Running
                || profile.desired_generation != generation
                || profile.lease_token.is_some()
            {
                return Ok(None);
            }
            let token = Uuid::new_v4();
            profile.lease_owner = Some(owner.to_owned());
            profile.lease_token = Some(token);
            profile.lease_expires_at = Some(
                Utc::now()
                    + chrono::Duration::from_std(lease_duration).expect("test lease duration"),
            );
            profile.observed_state = ObservedState::Starting;
            Ok(Some((profile.clone(), token)))
        }

        async fn renew_lease(
            &self,
            key: IngesterStrategyKey,
            owner: &str,
            token: Uuid,
            lease_duration: Duration,
        ) -> Result<bool> {
            let mut profile = self.profile.lock().expect("profile lock");
            let owned = profile.strategy_key == key
                && profile.desired_state == DesiredState::Running
                && profile.lease_owner.as_deref() == Some(owner)
                && profile.lease_token == Some(token);
            if owned {
                self.renewals.fetch_add(1, Ordering::SeqCst);
                profile.lease_expires_at = Some(
                    Utc::now()
                        + chrono::Duration::from_std(lease_duration).expect("test lease duration"),
                );
            }
            Ok(owned)
        }

        async fn mark_running(
            &self,
            key: IngesterStrategyKey,
            owner: &str,
            token: Uuid,
            generation: i64,
        ) -> Result<bool> {
            let mut profile = self.profile.lock().expect("profile lock");
            let owned = profile.strategy_key == key
                && profile.lease_owner.as_deref() == Some(owner)
                && profile.lease_token == Some(token);
            if owned {
                profile.observed_state = ObservedState::Running;
                profile.health_status = HealthStatus::Unknown;
                profile.applied_generation = Some(generation);
            }
            Ok(owned)
        }

        async fn mark_stopped(
            &self,
            key: IngesterStrategyKey,
            owner: &str,
            token: Uuid,
            generation: i64,
        ) -> Result<bool> {
            let mut profile = self.profile.lock().expect("profile lock");
            let owned = profile.strategy_key == key
                && profile.lease_owner.as_deref() == Some(owner)
                && profile.lease_token == Some(token);
            if owned {
                profile.observed_state = ObservedState::Stopped;
                profile.health_status = HealthStatus::Unknown;
                profile.applied_generation = Some(generation);
                profile.lease_owner = None;
                profile.lease_token = None;
                profile.lease_expires_at = None;
            }
            Ok(owned)
        }

        async fn mark_failed(
            &self,
            key: IngesterStrategyKey,
            owner: &str,
            token: Uuid,
            error_code: &str,
            error_message: &str,
        ) -> Result<bool> {
            let mut profile = self.profile.lock().expect("profile lock");
            let owned = profile.strategy_key == key
                && profile.lease_owner.as_deref() == Some(owner)
                && profile.lease_token == Some(token);
            if owned {
                profile.observed_state = ObservedState::Failed;
                profile.health_status = HealthStatus::Unhealthy;
                profile.last_error_code = Some(error_code.to_owned());
                profile.last_error_message = Some(error_message.to_owned());
                profile.lease_owner = None;
                profile.lease_token = None;
                profile.lease_expires_at = None;
            }
            Ok(owned)
        }
    }

    struct TestFactory {
        starts: Arc<AtomicUsize>,
        stops: Arc<AtomicUsize>,
        ignore_shutdown: bool,
    }

    impl StrategyFactory for TestFactory {
        fn key(&self) -> IngesterStrategyKey {
            KEY
        }

        fn config_schema_version(&self) -> i32 {
            1
        }

        fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError> {
            if config.get("valid") == Some(&Value::Bool(true)) {
                Ok(())
            } else {
                Err(StrategyFactoryError::InvalidConfiguration(
                    "valid must be true".to_owned(),
                ))
            }
        }

        fn build(
            &self,
            _profile: &IngesterProfile,
            _pool: PgPool,
        ) -> Result<Box<dyn IngesterStrategy>, StrategyFactoryError> {
            Ok(Box::new(TestStrategy {
                starts: Arc::clone(&self.starts),
                stops: Arc::clone(&self.stops),
                ignore_shutdown: self.ignore_shutdown,
            }))
        }
    }

    struct TestStrategy {
        starts: Arc<AtomicUsize>,
        stops: Arc<AtomicUsize>,
        ignore_shutdown: bool,
    }

    #[async_trait]
    impl IngesterStrategy for TestStrategy {
        fn key(&self) -> IngesterStrategyKey {
            KEY
        }

        async fn run(&self, shutdown: CancellationToken) -> Result<(), StrategyError> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            if self.ignore_shutdown {
                std::future::pending::<()>().await;
            } else {
                shutdown.cancelled().await;
            }
            self.stops.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn profile() -> IngesterProfile {
        let now = Utc::now();
        IngesterProfile {
            strategy_key: KEY,
            config_schema_version: 1,
            config: json!({"valid": true}),
            desired_state: DesiredState::Running,
            desired_generation: 1,
            observed_state: ObservedState::Stopped,
            health_status: HealthStatus::Unknown,
            applied_generation: None,
            checkpoint_schema_version: 1,
            checkpoint: json!({}),
            lease_owner: None,
            lease_token: None,
            lease_expires_at: None,
            heartbeat_at: None,
            started_at: None,
            stopped_at: None,
            last_source_event_at: None,
            last_provider_available_at: None,
            last_persisted_at: None,
            source_watermark: None,
            availability_watermark: None,
            consecutive_failures: 0,
            restart_count: 0,
            last_error_code: None,
            last_error_message: None,
            last_error_at: None,
            created_at: now,
            updated_at: now,
        }
    }

    fn supervisor(
        store: MemoryStore,
        starts: Arc<AtomicUsize>,
        stops: Arc<AtomicUsize>,
        ignore_shutdown: bool,
    ) -> StrategySupervisor<MemoryStore> {
        let registry = StrategyRegistry::from_factories([Arc::new(TestFactory {
            starts,
            stops,
            ignore_shutdown,
        }) as Arc<dyn StrategyFactory>])
        .expect("registry");
        let pool = PgPoolOptions::new()
            .connect_lazy("postgres://unused:unused@localhost/unused")
            .expect("lazy pool");
        StrategySupervisor::new(
            store,
            registry,
            pool,
            "test-instance",
            SupervisorSettings {
                reconcile_interval: Duration::from_millis(10),
                lease_duration: Duration::from_millis(50),
                strategy_shutdown_timeout: Duration::from_millis(25),
            },
        )
        .expect("supervisor")
    }

    #[tokio::test]
    async fn reconciles_start_stop_and_restart_generations() {
        let store = MemoryStore::new(profile());
        let starts = Arc::new(AtomicUsize::new(0));
        let stops = Arc::new(AtomicUsize::new(0));
        let mut supervisor = supervisor(
            store.clone(),
            Arc::clone(&starts),
            Arc::clone(&stops),
            false,
        );

        supervisor.reconcile_once().await.expect("start profile");
        tokio::task::yield_now().await;
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert_eq!(store.snapshot().observed_state, ObservedState::Running);

        store.update(|profile| {
            profile.desired_state = DesiredState::Stopped;
            profile.desired_generation = 2;
        });
        supervisor.reconcile_once().await.expect("request stop");
        tokio::task::yield_now().await;
        supervisor.reconcile_once().await.expect("finish stop");
        assert_eq!(stops.load(Ordering::SeqCst), 1);
        assert_eq!(store.snapshot().observed_state, ObservedState::Stopped);

        store.update(|profile| {
            profile.desired_state = DesiredState::Running;
            profile.desired_generation = 3;
        });
        supervisor.reconcile_once().await.expect("restart profile");
        tokio::task::yield_now().await;
        assert_eq!(starts.load(Ordering::SeqCst), 2);

        store.update(|profile| {
            profile.config = json!({"valid": true, "revision": 2});
            profile.desired_generation = 4;
        });
        supervisor
            .reconcile_once()
            .await
            .expect("request config restart");
        tokio::task::yield_now().await;
        supervisor
            .reconcile_once()
            .await
            .expect("apply config restart");
        tokio::task::yield_now().await;
        assert_eq!(starts.load(Ordering::SeqCst), 3);

        supervisor.shutdown_active().await;
    }

    #[tokio::test]
    async fn renews_leases_at_one_third_of_the_lease_duration() {
        let store = MemoryStore::new(profile());
        let starts = Arc::new(AtomicUsize::new(0));
        let stops = Arc::new(AtomicUsize::new(0));
        let mut supervisor = supervisor(
            store.clone(),
            Arc::clone(&starts),
            Arc::clone(&stops),
            false,
        );

        supervisor.reconcile_once().await.expect("start profile");
        supervisor
            .reconcile_once()
            .await
            .expect("reconcile before renewal");
        assert_eq!(store.renewals.load(Ordering::SeqCst), 0);

        tokio::time::sleep(Duration::from_millis(20)).await;
        supervisor
            .reconcile_once()
            .await
            .expect("reconcile after renewal threshold");
        assert_eq!(store.renewals.load(Ordering::SeqCst), 1);

        supervisor.shutdown_active().await;
    }

    #[tokio::test]
    async fn aborts_a_strategy_that_ignores_the_shutdown_deadline() {
        let store = MemoryStore::new(profile());
        let starts = Arc::new(AtomicUsize::new(0));
        let stops = Arc::new(AtomicUsize::new(0));
        let mut supervisor =
            supervisor(store.clone(), Arc::clone(&starts), Arc::clone(&stops), true);

        supervisor.reconcile_once().await.expect("start profile");
        tokio::task::yield_now().await;
        supervisor.shutdown_active().await;

        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert_eq!(stops.load(Ordering::SeqCst), 0);
        assert!(supervisor.active.is_empty());
        assert_eq!(store.snapshot().observed_state, ObservedState::Stopped);
    }

    #[test]
    fn retries_use_bounded_exponential_delay_with_stable_jitter() {
        let first = retry_delay(KEY, 1);
        let second = retry_delay(KEY, 2);
        let capped = retry_delay(KEY, 100);

        assert!(first >= Duration::from_secs(5));
        assert!(first < Duration::from_secs(6));
        assert_eq!(second - first, Duration::from_secs(5));
        assert!(capped >= MAX_RETRY_DELAY);
        assert!(capped < MAX_RETRY_DELAY + Duration::from_secs(1));
        assert_eq!(capped, retry_delay(KEY, 100));
    }
}
