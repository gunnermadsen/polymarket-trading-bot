use std::{env, path::PathBuf, time::Duration};

use anyhow::{bail, Context, Result};
use reqwest::Client;
use sqlx::postgres::PgPoolOptions;
use tokio::{sync::watch, task::JoinHandle, time::sleep};
use tracing::{error, info, warn};

use crate::config::AppConfig;

use super::{
    archive::{KrakenArchiveClient, DEFAULT_FUTURES_BASE_URL},
    job::ClaimedKrakenJob,
    lake::KrakenDataLake,
    repository::KrakenRepository,
};

#[derive(Debug, Clone)]
pub struct KrakenWorkerConfig {
    pub worker_id: String,
    pub data_lake_root: PathBuf,
    pub futures_base_url: String,
    pub poll_interval: Duration,
    pub lease_duration: Duration,
    pub heartbeat_interval: Duration,
    pub database_pool_connections: u32,
}

impl KrakenWorkerConfig {
    pub fn from_env() -> Result<Self> {
        let config = Self {
            worker_id: env_or("KRAKEN_BACKFILL_WORKER_ID", "kraken-backfill-worker-1"),
            data_lake_root: PathBuf::from(env_or("KRAKEN_DATA_LAKE_ROOT", "/var/lib/kraken-data")),
            futures_base_url: env_or("KRAKEN_FUTURES_BASE_URL", DEFAULT_FUTURES_BASE_URL),
            poll_interval: Duration::from_millis(env_u64(
                "KRAKEN_BACKFILL_POLL_INTERVAL_MS",
                1_000,
            )?),
            lease_duration: Duration::from_secs(env_u64(
                "KRAKEN_BACKFILL_LEASE_DURATION_SECS",
                120,
            )?),
            heartbeat_interval: Duration::from_secs(env_u64(
                "KRAKEN_BACKFILL_HEARTBEAT_INTERVAL_SECS",
                20,
            )?),
            database_pool_connections: env_u32("KRAKEN_BACKFILL_DB_POOL_MAX_CONNECTIONS", 1)?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self.worker_id.trim().is_empty() {
            bail!("KRAKEN_BACKFILL_WORKER_ID cannot be empty");
        }
        if !self.data_lake_root.is_absolute() {
            bail!("KRAKEN_DATA_LAKE_ROOT must be absolute");
        }
        if self.poll_interval.is_zero() {
            bail!("KRAKEN_BACKFILL_POLL_INTERVAL_MS must be positive");
        }
        if self.heartbeat_interval.is_zero() || self.lease_duration <= self.heartbeat_interval * 2 {
            bail!("Kraken lease must exceed twice the heartbeat interval");
        }
        if !(1..=4).contains(&self.database_pool_connections) {
            bail!("KRAKEN_BACKFILL_DB_POOL_MAX_CONNECTIONS must be between 1 and 4");
        }
        Ok(())
    }
}

pub struct KrakenBackfillWorker {
    config: KrakenWorkerConfig,
    repository: KrakenRepository,
    archive: KrakenArchiveClient,
    lake: KrakenDataLake,
}

impl KrakenBackfillWorker {
    pub async fn from_env() -> Result<Self> {
        let config = KrakenWorkerConfig::from_env()?;
        let app = AppConfig::from_env()?;
        let pool = PgPoolOptions::new()
            .max_connections(config.database_pool_connections)
            .connect(&app.postgres.database_url())
            .await
            .context("failed to connect Kraken worker to PostgreSQL")?;
        let repository = KrakenRepository::from_pool(pool);
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .user_agent("capitonic-kraken-archive/1.0")
            .build()
            .context("failed to build Kraken HTTP client")?;
        let archive =
            KrakenArchiveClient::new(client, repository.clone(), config.futures_base_url.clone())?;
        let lake = KrakenDataLake::new(config.data_lake_root.clone())?;
        Ok(Self {
            config,
            repository,
            archive,
            lake,
        })
    }

    pub async fn run_until_shutdown(&self, mut shutdown: watch::Receiver<bool>) -> Result<()> {
        info!(worker_id = %self.config.worker_id, "Kraken backfill worker started");
        loop {
            if *shutdown.borrow() {
                break;
            }
            match self
                .repository
                .claim_next(&self.config.worker_id, self.config.lease_duration)
                .await
            {
                Ok(Some(claim)) => self.run_claim(claim).await,
                Ok(None) => {
                    tokio::select! {
                        _ = sleep(self.config.poll_interval) => {}
                        changed = shutdown.changed() => {
                            if changed.is_err() || *shutdown.borrow() {
                                break;
                            }
                        }
                    }
                }
                Err(error) => {
                    error!(worker_id = %self.config.worker_id, error = %error, "Kraken job claim failed");
                    sleep(self.config.poll_interval).await;
                }
            }
        }
        self.repository
            .set_worker_stopped(&self.config.worker_id)
            .await?;
        info!(worker_id = %self.config.worker_id, "Kraken backfill worker stopped");
        Ok(())
    }

    async fn run_claim(&self, claim: ClaimedKrakenJob) {
        let heartbeat = self.spawn_heartbeat(claim.clone());
        let result = self.execute_claim(&claim).await;
        heartbeat.abort();
        if let Err(error) = result {
            warn!(
                worker_id = %self.config.worker_id,
                job_id = %claim.job.job_id,
                error = %error,
                "Kraken backfill job attempt failed"
            );
            if let Err(record_error) = self
                .repository
                .fail(&self.config.worker_id, &claim, &format!("{error:#}"))
                .await
            {
                error!(error = %record_error, "failed to persist Kraken job failure");
            }
        }
    }

    async fn execute_claim(&self, claim: &ClaimedKrakenJob) -> Result<()> {
        let fetched = self.archive.fetch(&claim.job).await?;
        self.repository
            .update_progress(claim, claim.job.expected_work_units / 2)
            .await?;
        let object = self.lake.publish(&claim.job, &fetched.rows).await?;
        self.repository
            .update_progress(claim, claim.job.expected_work_units * 3 / 4)
            .await?;
        let written = self
            .repository
            .persist(claim, &fetched.source_url, &fetched.rows, &object)
            .await?;
        self.repository
            .complete(&self.config.worker_id, claim, written)
            .await?;
        info!(
            worker_id = %self.config.worker_id,
            job_id = %claim.job.job_id,
            dataset = %claim.job.dataset,
            rows = written,
            "Kraken backfill job completed"
        );
        Ok(())
    }

    fn spawn_heartbeat(&self, claim: ClaimedKrakenJob) -> JoinHandle<()> {
        let repository = self.repository.clone();
        let worker_id = self.config.worker_id.clone();
        let heartbeat_interval = self.config.heartbeat_interval;
        let lease_duration = self.config.lease_duration;
        tokio::spawn(async move {
            loop {
                sleep(heartbeat_interval).await;
                if let Err(error) = repository
                    .heartbeat(&worker_id, &claim, lease_duration)
                    .await
                {
                    warn!(job_id = %claim.job.job_id, error = %error, "Kraken heartbeat stopped");
                    break;
                }
            }
        })
    }
}

fn env_or(name: &str, default: &str) -> String {
    env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_u64(name: &str, default: u64) -> Result<u64> {
    match env::var(name) {
        Ok(value) => value
            .parse()
            .with_context(|| format!("{name} must be an unsigned integer")),
        Err(_) => Ok(default),
    }
}

fn env_u32(name: &str, default: u32) -> Result<u32> {
    let value = env_u64(name, u64::from(default))?;
    u32::try_from(value).with_context(|| format!("{name} exceeded u32"))
}
