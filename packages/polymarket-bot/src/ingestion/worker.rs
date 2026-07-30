use std::{
    env,
    path::PathBuf,
    time::{Duration, SystemTime},
};

use anyhow::{bail, Context, Result};
use sqlx::postgres::PgPoolOptions;
use tokio::{sync::watch, time::MissedTickBehavior};
use tracing::{error, info, warn};

use crate::config::AppConfig;

use super::{
    binance_archive::ArchiveCancellation,
    chainlink_archive::{
        ChainlinkArchiveConfig, ChainlinkCredentials, DEFAULT_CHAINLINK_BTCUSD_FEED_ID,
        DEFAULT_CHAINLINK_REST_URL,
    },
    executor::{IngestionExecutor, IngestionExecutorConfig},
    job::{BackfillEventLevel, BackfillFailureKind, BackfillJobSummary, ClaimedJob, WorkerControl},
    pmxt_archive::DEFAULT_PMXT_ARCHIVE_URL,
    polygon_chainlink_oracle::{
        PolygonChainlinkOracleConfig, DEFAULT_POLYGON_ARCHIVE_LOG_RPC_URL,
        DEFAULT_POLYGON_CHAINLINK_BTCUSD_PROXY, DEFAULT_POLYGON_RPC_URL,
    },
    repository::IngestionRepository,
};

const DEFAULT_BINANCE_ARCHIVE_BASE_URL: &str = "https://data.binance.vision";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackfillWorkerConfig {
    pub worker_id: String,
    pub cache_directory: PathBuf,
    pub poll_interval: Duration,
    pub lease_duration: Duration,
    pub heartbeat_interval: Duration,
    pub database_pool_connections: u32,
    pub batch_rows: usize,
    pub maximum_cache_bytes: u64,
    pub stale_cache_age: Duration,
}

impl BackfillWorkerConfig {
    pub fn from_env() -> Result<Self> {
        let config = Self {
            worker_id: env_string(
                "POLYMARKET_BACKFILL_WORKER_ID",
                "polymarket-backfill-worker",
            ),
            cache_directory: PathBuf::from(env_string(
                "POLYMARKET_BACKFILL_CACHE_DIR",
                "/var/lib/polymarket/backfill-cache",
            )),
            poll_interval: Duration::from_millis(env_u64(
                "POLYMARKET_BACKFILL_POLL_INTERVAL_MS",
                1_000,
            )?),
            lease_duration: Duration::from_secs(env_u64("POLYMARKET_BACKFILL_LEASE_SECS", 60)?),
            heartbeat_interval: Duration::from_secs(env_u64(
                "POLYMARKET_BACKFILL_HEARTBEAT_SECS",
                15,
            )?),
            database_pool_connections: env_u32("POLYMARKET_BACKFILL_DB_POOL_MAX_CONNECTIONS", 4)?,
            batch_rows: env_usize("POLYMARKET_BACKFILL_BATCH_ROWS", 4_000)?,
            maximum_cache_bytes: env_u64(
                "POLYMARKET_BACKFILL_CACHE_MAX_BYTES",
                20 * 1024 * 1024 * 1024,
            )?,
            stale_cache_age: Duration::from_secs(env_u64(
                "POLYMARKET_BACKFILL_CACHE_STALE_SECS",
                48 * 60 * 60,
            )?),
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.worker_id.trim().is_empty() {
            bail!("POLYMARKET_BACKFILL_WORKER_ID must not be empty");
        }
        if self.cache_directory.as_os_str().is_empty() {
            bail!("POLYMARKET_BACKFILL_CACHE_DIR must not be empty");
        }
        if self.poll_interval.is_zero() {
            bail!("POLYMARKET_BACKFILL_POLL_INTERVAL_MS must be positive");
        }
        if self.heartbeat_interval.is_zero() || self.lease_duration.is_zero() {
            bail!("backfill heartbeat and lease durations must be positive");
        }
        if self.heartbeat_interval.saturating_mul(2) >= self.lease_duration {
            bail!("backfill lease must exceed two heartbeat intervals");
        }
        if !(1..=16).contains(&self.database_pool_connections) {
            bail!("POLYMARKET_BACKFILL_DB_POOL_MAX_CONNECTIONS must be between 1 and 16");
        }
        if !(1..=4_000).contains(&self.batch_rows) {
            bail!("POLYMARKET_BACKFILL_BATCH_ROWS must be between 1 and 4000");
        }
        if self.maximum_cache_bytes < 4 * 1024 * 1024 * 1024 {
            bail!("POLYMARKET_BACKFILL_CACHE_MAX_BYTES must be at least 4 GiB");
        }
        if self.stale_cache_age < Duration::from_secs(60 * 60) {
            bail!("POLYMARKET_BACKFILL_CACHE_STALE_SECS must be at least one hour");
        }
        Ok(())
    }
}

pub struct BackfillWorker {
    repository: IngestionRepository,
    executor: IngestionExecutor,
    config: BackfillWorkerConfig,
}

impl BackfillWorker {
    pub async fn from_env() -> Result<Self> {
        let app = AppConfig::from_env()?;
        let config = BackfillWorkerConfig::from_env()?;
        prepare_cache_directory(&config).await?;
        let pool = PgPoolOptions::new()
            .max_connections(config.database_pool_connections)
            .connect(&app.postgres.database_url())
            .await
            .context("failed to connect backfill worker to Postgres")?;
        let repository = IngestionRepository::from_pool(pool);
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .user_agent("polymarket-bot-backfill-worker/1")
            .build()
            .context("failed to build backfill source client")?;
        let executor = IngestionExecutor::new(
            repository.clone(),
            client,
            IngestionExecutorConfig {
                gamma_base_url: app.gamma_base_url,
                clob_base_url: app.clob_base_url,
                binance_archive_base_url: DEFAULT_BINANCE_ARCHIVE_BASE_URL.to_string(),
                pmxt_archive_base_url: env_string(
                    "POLYMARKET_PMXT_ARCHIVE_BASE_URL",
                    DEFAULT_PMXT_ARCHIVE_URL,
                ),
                chainlink: ChainlinkArchiveConfig {
                    rest_url: env_string(
                        "POLYMARKET_CHAINLINK_DATA_STREAMS_REST_URL",
                        DEFAULT_CHAINLINK_REST_URL,
                    ),
                    feed_id: env_string(
                        "POLYMARKET_CHAINLINK_DATA_STREAMS_FEED_ID",
                        DEFAULT_CHAINLINK_BTCUSD_FEED_ID,
                    ),
                    page_limit: env_usize("POLYMARKET_CHAINLINK_DATA_STREAMS_PAGE_LIMIT", 1_000)?,
                    credentials: chainlink_credentials_from_env()?,
                },
                polygon_chainlink: PolygonChainlinkOracleConfig {
                    rpc_url: env_string("POLYMARKET_POLYGON_RPC_URL", DEFAULT_POLYGON_RPC_URL),
                    archive_log_rpc_url: env_string(
                        "POLYMARKET_POLYGON_ARCHIVE_LOG_RPC_URL",
                        DEFAULT_POLYGON_ARCHIVE_LOG_RPC_URL,
                    ),
                    feed_proxy_address: env_string(
                        "POLYMARKET_POLYGON_CHAINLINK_BTCUSD_PROXY",
                        DEFAULT_POLYGON_CHAINLINK_BTCUSD_PROXY,
                    ),
                    maximum_block_range: env_u64("POLYMARKET_POLYGON_RPC_MAX_BLOCK_RANGE", 100)?,
                },
                cache_directory: config.cache_directory.clone(),
                batch_rows: config.batch_rows,
                pmxt_prefetch_concurrency: env_usize("POLYMARKET_PMXT_PREFETCH_CONCURRENCY", 4)?,
                pmxt_prefetch_archives: env_usize("POLYMARKET_PMXT_PREFETCH_ARCHIVES", 48)?,
            },
        )?;
        Ok(Self {
            repository,
            executor,
            config,
        })
    }

    pub async fn run_until_shutdown(&self, mut shutdown: watch::Receiver<bool>) -> Result<()> {
        info!(worker_id = %self.config.worker_id, "backfill worker ready");
        loop {
            if *shutdown.borrow() {
                info!("backfill worker shutdown completed");
                return Ok(());
            }
            let claim = self
                .repository
                .claim_next(&self.config.worker_id, self.config.lease_duration)
                .await
                .context("failed to claim a backfill job")?;
            if let Some(claim) = claim {
                self.execute_claim(claim, shutdown.clone()).await;
                continue;
            }
            tokio::select! {
                _ = tokio::time::sleep(self.config.poll_interval) => {}
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        info!("backfill worker shutdown completed");
                        return Ok(());
                    }
                }
            }
        }
    }

    async fn execute_claim(&self, claim: ClaimedJob, shutdown: watch::Receiver<bool>) {
        let job_id = claim.job.job_id;
        info!(%job_id, ingester = %claim.job.ingester_key, attempt = claim.job.attempt, "backfill job claimed");
        let _ = self
            .repository
            .append_event(
                job_id,
                BackfillEventLevel::Info,
                "backfill job claimed",
                serde_json::json!({
                    "worker_id": self.config.worker_id,
                    "attempt": claim.job.attempt,
                }),
            )
            .await;

        let cancellation = ArchiveCancellation::default();
        let (stop_sender, stop_receiver) = watch::channel(false);
        let heartbeat = tokio::spawn(run_heartbeat(
            self.repository.clone(),
            claim.clone(),
            self.config.heartbeat_interval,
            self.config.lease_duration,
            cancellation.clone(),
            shutdown,
            stop_receiver,
        ));
        let execution = self.executor.execute(&claim, cancellation.clone()).await;
        let _ = stop_sender.send(true);
        let heartbeat_outcome = match heartbeat.await {
            Ok(outcome) => outcome,
            Err(error) => {
                error!(%job_id, error = %error, "backfill heartbeat task panicked");
                HeartbeatOutcome::LeaseLost
            }
        };

        let control = self
            .repository
            .is_cancel_requested(&claim)
            .await
            .unwrap_or(WorkerControl::LeaseLost);
        if heartbeat_outcome == HeartbeatOutcome::LeaseLost || control == WorkerControl::LeaseLost {
            warn!(%job_id, "backfill lease was lost; stale worker will not transition the job");
            return;
        }
        let summary = execution
            .as_ref()
            .ok()
            .cloned()
            .unwrap_or_else(|| summary_from_claim(&claim));
        if heartbeat_outcome == HeartbeatOutcome::CancelRequested
            || control == WorkerControl::CancelRequested
        {
            match self.repository.mark_cancelled(&claim, &summary).await {
                Ok(_) => info!(%job_id, "backfill job cancelled"),
                Err(error) => warn!(%job_id, error = %error, "failed to finalize cancellation"),
            }
            return;
        }
        if heartbeat_outcome == HeartbeatOutcome::Shutdown {
            self.retry_claim(&claim, "worker shutdown interrupted the job", &summary)
                .await;
            return;
        }

        match execution {
            Ok(summary) => match self.repository.complete(&claim, &summary).await {
                Ok(_) => info!(%job_id, "backfill job completed"),
                Err(error) => warn!(%job_id, error = %error, "failed to complete backfill job"),
            },
            Err(error) => {
                let retry_after = retry_delay(claim.job.attempt);
                let message = error.to_string();
                match self
                    .repository
                    .fail_or_retry(&claim, error.kind, &message, retry_after)
                    .await
                {
                    Ok(job) => {
                        warn!(%job_id, status = ?job.status, kind = ?error.kind, error = %message, "backfill job failed")
                    }
                    Err(transition_error) => {
                        warn!(%job_id, error = %transition_error, "failed to record backfill failure")
                    }
                }
            }
        }
    }

    async fn retry_claim(&self, claim: &ClaimedJob, message: &str, _summary: &BackfillJobSummary) {
        if let Err(error) = self
            .repository
            .fail_or_retry(
                claim,
                BackfillFailureKind::Transient,
                message,
                Duration::from_secs(1),
            )
            .await
        {
            warn!(job_id = %claim.job.job_id, error = %error, "failed to requeue interrupted job");
        }
    }
}

async fn prepare_cache_directory(config: &BackfillWorkerConfig) -> Result<()> {
    tokio::fs::create_dir_all(&config.cache_directory)
        .await
        .with_context(|| {
            format!(
                "failed to create backfill cache directory {}",
                config.cache_directory.display()
            )
        })?;
    let now = SystemTime::now();
    let mut directory = tokio::fs::read_dir(&config.cache_directory)
        .await
        .with_context(|| {
            format!(
                "failed to inspect backfill cache directory {}",
                config.cache_directory.display()
            )
        })?;
    let mut retained = Vec::new();
    while let Some(entry) = directory.next_entry().await? {
        let metadata = entry.metadata().await?;
        if !metadata.is_file() {
            continue;
        }
        let path = entry.path();
        let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let stale = now
            .duration_since(modified)
            .is_ok_and(|age| age >= config.stale_cache_age);
        let partial = path
            .extension()
            .is_some_and(|extension| extension == "part");
        if stale || partial {
            tokio::fs::remove_file(&path).await.with_context(|| {
                format!("failed to remove stale backfill cache {}", path.display())
            })?;
            continue;
        }
        retained.push((modified, metadata.len(), path));
    }
    retained.sort_by_key(|(modified, _, _)| *modified);
    let mut retained_bytes = retained
        .iter()
        .fold(0u64, |total, (_, size, _)| total.saturating_add(*size));
    for (_, size, path) in retained {
        if retained_bytes <= config.maximum_cache_bytes {
            break;
        }
        tokio::fs::remove_file(&path).await.with_context(|| {
            format!(
                "failed to enforce backfill cache quota for {}",
                path.display()
            )
        })?;
        retained_bytes = retained_bytes.saturating_sub(size);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeartbeatOutcome {
    Stopped,
    CancelRequested,
    LeaseLost,
    Shutdown,
}

async fn run_heartbeat(
    repository: IngestionRepository,
    claim: ClaimedJob,
    heartbeat_interval: Duration,
    lease_duration: Duration,
    cancellation: ArchiveCancellation,
    mut shutdown: watch::Receiver<bool>,
    mut stop: watch::Receiver<bool>,
) -> HeartbeatOutcome {
    if *stop.borrow() {
        return HeartbeatOutcome::Stopped;
    }
    if *shutdown.borrow() {
        cancellation.cancel();
        return HeartbeatOutcome::Shutdown;
    }

    let mut interval = tokio::time::interval(heartbeat_interval);
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    interval.tick().await;
    loop {
        tokio::select! {
            changed = stop.changed() => {
                if changed.is_err() || *stop.borrow() {
                    return HeartbeatOutcome::Stopped;
                }
            }
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    cancellation.cancel();
                    return HeartbeatOutcome::Shutdown;
                }
            }
            _ = interval.tick() => {
                if repository.heartbeat(&claim, lease_duration).await.is_err() {
                    cancellation.cancel();
                    return HeartbeatOutcome::LeaseLost;
                }
                match repository.is_cancel_requested(&claim).await {
                    Ok(WorkerControl::Continue) => {}
                    Ok(WorkerControl::CancelRequested) => {
                        cancellation.cancel();
                        return HeartbeatOutcome::CancelRequested;
                    }
                    Ok(WorkerControl::LeaseLost) | Err(_) => {
                        cancellation.cancel();
                        return HeartbeatOutcome::LeaseLost;
                    }
                }
            }
        }
    }
}

fn retry_delay(attempt: i32) -> Duration {
    let exponent = u32::try_from(attempt.saturating_sub(1).clamp(0, 5)).unwrap_or(0);
    Duration::from_secs(30u64.saturating_mul(2u64.saturating_pow(exponent)))
}

fn summary_from_claim(claim: &ClaimedJob) -> BackfillJobSummary {
    let progress =
        serde_json::from_value::<super::job::BackfillProgress>(claim.job.progress.clone())
            .unwrap_or_default();
    BackfillJobSummary {
        expected_work_units: progress.expected_work_units,
        completed_work_units: progress.completed_work_units,
        records_read: progress.records_read,
        records_committed: progress.records_committed,
        ..BackfillJobSummary::default()
    }
}

fn env_string(key: &str, default: &str) -> String {
    env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn chainlink_credentials_from_env() -> Result<Option<ChainlinkCredentials>> {
    let api_key = env::var("POLYMARKET_CHAINLINK_DATA_STREAMS_API_KEY")
        .ok()
        .filter(|value| !value.trim().is_empty());
    let api_secret = env::var("POLYMARKET_CHAINLINK_DATA_STREAMS_API_SECRET")
        .ok()
        .filter(|value| !value.trim().is_empty());
    match (api_key, api_secret) {
        (None, None) => Ok(None),
        (Some(api_key), Some(api_secret)) => Ok(Some(ChainlinkCredentials {
            api_key,
            api_secret,
        })),
        _ => bail!(
            "POLYMARKET_CHAINLINK_DATA_STREAMS_API_KEY and POLYMARKET_CHAINLINK_DATA_STREAMS_API_SECRET must be configured together"
        ),
    }
}

fn env_u64(key: &str, default: u64) -> Result<u64> {
    match env::var(key) {
        Ok(value) if !value.trim().is_empty() => value
            .parse::<u64>()
            .with_context(|| format!("{key} must be an unsigned integer")),
        _ => Ok(default),
    }
}

fn env_u32(key: &str, default: u32) -> Result<u32> {
    u32::try_from(env_u64(key, u64::from(default))?)
        .with_context(|| format!("{key} exceeds the supported range"))
}

fn env_usize(key: &str, default: usize) -> Result<usize> {
    usize::try_from(env_u64(key, u64::try_from(default).unwrap_or(u64::MAX))?)
        .with_context(|| format!("{key} exceeds the supported range"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BackfillWorkerConfig {
        BackfillWorkerConfig {
            worker_id: "worker".to_string(),
            cache_directory: PathBuf::from("/cache"),
            poll_interval: Duration::from_secs(1),
            lease_duration: Duration::from_secs(60),
            heartbeat_interval: Duration::from_secs(15),
            database_pool_connections: 4,
            batch_rows: 4_000,
            maximum_cache_bytes: 20 * 1024 * 1024 * 1024,
            stale_cache_age: Duration::from_secs(48 * 60 * 60),
        }
    }

    #[test]
    fn worker_config_rejects_unsafe_lease_and_batch_bounds() {
        let mut value = config();
        assert!(value.validate().is_ok());
        value.lease_duration = Duration::from_secs(30);
        assert!(value.validate().is_err());
        value = config();
        value.batch_rows = 4_001;
        assert!(value.validate().is_err());
    }

    #[test]
    fn retry_delay_is_bounded_exponential() {
        assert_eq!(retry_delay(1), Duration::from_secs(30));
        assert_eq!(retry_delay(2), Duration::from_secs(60));
        assert_eq!(retry_delay(99), Duration::from_secs(960));
    }
}
