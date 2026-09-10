use std::{collections::BTreeMap, env, sync::Arc, time::Duration};

use anyhow::{bail, Context, Result};
use reqwest::{Client, StatusCode};
use serde::Serialize;
use serde_json::json;
use sqlx::PgPool;
use tokio::task::JoinSet;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::{
    domain::{
        BackfillContext, BackfillFailureKind, BackfillOutcome, BackfillShard,
        ALLOCATION_CONTRACT_VERSION, DEFAULT_REALTIME_SLOT_LIMIT, DEFAULT_WORKER_CAPACITY_UNITS,
    },
    persistence::{ClaimedBackfillJob, WorkerRegistration},
};

use super::StrategyRegistry;

pub struct BackfillWorkerRuntime {
    registry: StrategyRegistry,
    pool: PgPool,
    client: Client,
    master_url: String,
    admin_token: String,
    worker: WorkerRegistration,
}

impl BackfillWorkerRuntime {
    pub fn from_environment(registry: StrategyRegistry, pool: PgPool) -> Result<Self> {
        let master_url = required("INGESTER_MASTER_URL")?
            .trim_end_matches('/')
            .to_owned();
        if !master_url.starts_with("http://") && !master_url.starts_with("https://") {
            bail!("INGESTER_MASTER_URL must be an HTTP or HTTPS URL");
        }
        let admin_token = env::var("INGESTER_ADMIN_TOKEN")
            .or_else(|_| env::var("MARKET_DATA_INGESTER_ADMIN_TOKEN"))
            .context("INGESTER_ADMIN_TOKEN is required")?;
        let supported_strategies = registry
            .backfills()
            .map(|strategy| {
                (
                    strategy.descriptor().strategy_key.to_string(),
                    strategy.descriptor().strategy_contract_version,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let hostname = env::var("INGESTER_ADVERTISE_HOST")
            .or_else(|_| env::var("HOSTNAME"))
            .unwrap_or_else(|_| "ingester-worker".to_owned());
        let worker = WorkerRegistration {
            worker_id: env::var("INGESTER_WORKER_ID")
                .or_else(|_| env::var("INGESTER_INSTANCE"))
                .unwrap_or_else(|_| hostname.clone()),
            hostname,
            worker_contract_version: 1,
            supported_strategies,
            maximum_backfills: env::var("INGESTER_WORKER_MAX_BACKFILLS")
                .unwrap_or_else(|_| "1".to_owned())
                .parse()
                .context("INGESTER_WORKER_MAX_BACKFILLS must be an integer")?,
            capacity_units: env::var("INGESTER_WORKER_CAPACITY_UNITS")
                .unwrap_or_else(|_| DEFAULT_WORKER_CAPACITY_UNITS.to_string())
                .parse()
                .context("INGESTER_WORKER_CAPACITY_UNITS must be an integer")?,
            realtime_slot_limit: env::var("INGESTER_WORKER_REALTIME_SLOT_LIMIT")
                .unwrap_or_else(|_| DEFAULT_REALTIME_SLOT_LIMIT.to_string())
                .parse()
                .context("INGESTER_WORKER_REALTIME_SLOT_LIMIT must be an integer")?,
            allocation_contract_version: ALLOCATION_CONTRACT_VERSION,
            realtime_strategies: registry.keys().map(|key| key.as_str().to_owned()).collect(),
            image_digest: env::var("INGESTER_IMAGE_DIGEST")
                .unwrap_or_else(|_| "development".to_owned()),
            source_revision: env::var("INGESTER_GIT_REVISION")
                .unwrap_or_else(|_| "development".to_owned()),
            deployment_id: env::var("INGESTER_DEPLOYMENT_ID")
                .unwrap_or_else(|_| "default".to_owned()),
        };
        worker.validate()?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .user_agent("capitonic-ingester-worker/1")
            .build()?;
        Ok(Self {
            registry,
            pool,
            client,
            master_url,
            admin_token,
            worker,
        })
    }

    pub async fn run(self, shutdown: CancellationToken) -> Result<()> {
        let runtime = Arc::new(self);
        let mut registration_delay = Duration::from_secs(1);
        loop {
            match runtime.register().await {
                Ok(()) => break,
                Err(error) => warn!(
                    worker_id = %runtime.worker.worker_id,
                    %error,
                    retry_delay_ms = registration_delay.as_millis(),
                    "ingester worker registration unavailable; preserving realtime worker process"
                ),
            }
            tokio::select! {
                _ = shutdown.cancelled() => return Ok(()),
                _ = tokio::time::sleep(registration_delay) => {}
            }
            registration_delay = (registration_delay * 2).min(Duration::from_secs(30));
        }
        info!(worker_id=%runtime.worker.worker_id, capacity_units=runtime.worker.capacity_units, realtime_slot_limit=runtime.worker.realtime_slot_limit, "ingester worker registered");
        let mut idle = tokio::time::interval(Duration::from_secs(1));
        idle.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut active = JoinSet::new();
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => {
                    while active.join_next().await.is_some() {}
                    return Ok(());
                },
                completed = active.join_next(), if !active.is_empty() => {
                    if let Some(Err(error)) = completed {
                        warn!(error=%error, "backfill execution task failed");
                    }
                },
                _ = idle.tick() => {
                    if let Err(error) = runtime.worker_heartbeat().await {
                        warn!(error=%error, "ingester worker heartbeat failed");
                        continue;
                    }
                    while active.len() < usize::try_from(runtime.worker.maximum_backfills).unwrap_or(1) {
                        match runtime.claim().await {
                            Ok(Some(claim)) => {
                                let execution = Arc::clone(&runtime);
                                let execution_shutdown = shutdown.clone();
                                active.spawn(async move { execution.execute(claim, execution_shutdown).await });
                            }
                            Ok(None) => break,
                            Err(error) => {
                                warn!(error=%error, "ingester worker assignment request failed");
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    async fn register(&self) -> Result<()> {
        self.client
            .post(format!("{}/internal/workers/register", self.master_url))
            .bearer_auth(&self.admin_token)
            .json(&self.worker)
            .send()
            .await?
            .error_for_status()
            .context("master rejected worker registration")?;
        Ok(())
    }

    async fn worker_heartbeat(&self) -> Result<()> {
        self.client
            .post(format!(
                "{}/internal/workers/{}/heartbeat",
                self.master_url, self.worker.worker_id
            ))
            .bearer_auth(&self.admin_token)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    async fn claim(&self) -> Result<Option<ClaimedBackfillJob>> {
        let response = self
            .client
            .post(format!(
                "{}/internal/workers/{}/assignments",
                self.master_url, self.worker.worker_id
            ))
            .bearer_auth(&self.admin_token)
            .send()
            .await?
            .error_for_status()?;
        Ok(response.json().await?)
    }

    async fn execute(&self, claim: ClaimedBackfillJob, shutdown: CancellationToken) {
        let job_id = claim.job.job_id;
        let Some(strategy) = self.registry.backfill(&claim.job.strategy_key) else {
            let _ = self
                .fail(
                    &claim,
                    BackfillFailureKind::InvalidRequest,
                    "strategy_unavailable",
                    "assigned strategy is not registered on this worker",
                )
                .await;
            return;
        };
        if strategy.descriptor().strategy_contract_version != claim.job.strategy_contract_version {
            let _ = self
                .fail(
                    &claim,
                    BackfillFailureKind::InvalidRequest,
                    "strategy_contract_mismatch",
                    "assigned strategy contract version is incompatible",
                )
                .await;
            return;
        }
        let shard = BackfillShard {
            shard_key: claim
                .job
                .shard_key
                .clone()
                .unwrap_or_else(|| job_id.to_string()),
            range_start: claim.job.range_start,
            range_end: claim.job.range_end,
            parameters: claim
                .job
                .canonical_request
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({})),
        };
        let execution_shutdown = shutdown.child_token();
        let heartbeat_shutdown = execution_shutdown.clone();
        let heartbeat = self.heartbeat_loop(&claim, heartbeat_shutdown);
        tokio::pin!(heartbeat);
        let execution = strategy.execute_backfill(
            BackfillContext {
                pool: self.pool.clone(),
                job_id,
                lease_token: claim.lease_token,
                worker_id: Arc::from(self.worker.worker_id.clone()),
                shutdown: execution_shutdown.clone(),
            },
            shard,
        );
        tokio::pin!(execution);
        let result = tokio::select! {
            result = &mut execution => result,
            heartbeat_result = &mut heartbeat => {
                execution_shutdown.cancel();
                match heartbeat_result {
                    Ok(()) => Err(crate::domain::BackfillExecutionError::new(BackfillFailureKind::LeaseLost, "lease_lost", "job heartbeat ended")),
                    Err(error) => Err(crate::domain::BackfillExecutionError::new(BackfillFailureKind::LeaseLost, "lease_lost", error.to_string())),
                }
            }
            _ = shutdown.cancelled() => {
                execution_shutdown.cancel();
                Err(crate::domain::BackfillExecutionError::new(BackfillFailureKind::Cancelled, "worker_shutdown", "worker shutdown interrupted the job"))
            }
        };
        execution_shutdown.cancel();
        match result {
            Ok(outcome) => {
                if let Err(error) = self.complete(&claim, &outcome).await {
                    warn!(%job_id,error=%error,"failed to report backfill completion");
                }
            }
            Err(error) => {
                if let Err(report_error) = self
                    .fail(&claim, error.kind, error.code, &error.message)
                    .await
                {
                    warn!(%job_id,error=%report_error,"failed to report backfill failure");
                }
            }
        }
    }

    async fn heartbeat_loop(
        &self,
        claim: &ClaimedBackfillJob,
        shutdown: CancellationToken,
    ) -> Result<()> {
        let mut interval = tokio::time::interval(Duration::from_secs(15));
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        interval.tick().await;
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => return Ok(()),
                _ = interval.tick() => {
                    let response = self.client.post(format!("{}/internal/workers/{}/jobs/{}/heartbeat", self.master_url, self.worker.worker_id, claim.job.job_id))
                        .bearer_auth(&self.admin_token)
                        .json(&json!({"lease_token": claim.lease_token, "progress": {}, "checkpoint": {}}))
                        .send().await?;
                    if response.status() == StatusCode::CONFLICT { bail!("backfill lease was lost"); }
                    response.error_for_status()?;
                }
            }
        }
    }

    async fn complete(&self, claim: &ClaimedBackfillJob, outcome: &BackfillOutcome) -> Result<()> {
        self.post_job(
            claim,
            "complete",
            &json!({"lease_token": claim.lease_token, "outcome": outcome}),
        )
        .await
    }

    async fn fail(
        &self,
        claim: &ClaimedBackfillJob,
        kind: BackfillFailureKind,
        code: &str,
        message: &str,
    ) -> Result<()> {
        self.post_job(claim, "fail", &json!({"lease_token": claim.lease_token, "kind": kind, "code": code, "message": message})).await
    }

    async fn post_job(
        &self,
        claim: &ClaimedBackfillJob,
        operation: &str,
        body: &impl Serialize,
    ) -> Result<()> {
        self.client
            .post(format!(
                "{}/internal/workers/{}/jobs/{}/{}",
                self.master_url, self.worker.worker_id, claim.job.job_id, operation
            ))
            .bearer_auth(&self.admin_token)
            .json(body)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}

fn required(key: &str) -> Result<String> {
    let value = env::var(key).with_context(|| format!("{key} is required"))?;
    if value.trim().is_empty() {
        bail!("{key} must not be empty");
    }
    Ok(value)
}
