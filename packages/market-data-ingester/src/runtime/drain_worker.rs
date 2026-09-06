use super::StrategyRegistry;
use crate::{
    domain::{DrainContext, DrainExecutionError, DrainMode, DrainRequest, ExecutionSelector},
    persistence::{ClaimedDrainJob, DrainRepository},
};
use anyhow::{Context, Result};
use std::{env, sync::Arc, time::Duration};
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

pub struct DrainWorkerRuntime {
    registry: StrategyRegistry,
    repository: DrainRepository,
    pool: sqlx::PgPool,
    worker_id: String,
    deployment_id: String,
}
impl DrainWorkerRuntime {
    pub fn from_environment(
        registry: StrategyRegistry,
        strategy_pool: sqlx::PgPool,
        control_pool: sqlx::PgPool,
    ) -> Result<Self> {
        let worker_id = env::var("INGESTER_WORKER_ID")
            .or_else(|_| env::var("INGESTER_INSTANCE"))
            .or_else(|_| env::var("HOSTNAME"))
            .unwrap_or_else(|_| "ingester-worker".into());
        let deployment_id = env::var("INGESTER_DEPLOYMENT_ID").unwrap_or_else(|_| "default".into());
        if worker_id.trim().is_empty() || deployment_id.trim().is_empty() {
            anyhow::bail!("drain worker identity and deployment must be non-empty");
        }
        Ok(Self {
            repository: DrainRepository::new(control_pool),
            registry,
            pool: strategy_pool,
            worker_id,
            deployment_id,
        })
    }
    pub async fn run(self, shutdown: CancellationToken) -> Result<()> {
        let supported = self
            .registry
            .drains()
            .map(|s| s.descriptor().strategy_key.to_string())
            .collect::<Vec<_>>();
        if supported.is_empty() {
            return Ok(());
        }
        info!(worker_id=%self.worker_id,"ingester drain worker starting");
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tokio::select! {_=shutdown.cancelled()=>return Ok(()),_=interval.tick()=>match self.repository.claim(&self.worker_id,&self.deployment_id,&supported).await{Ok(Some(claim))=>self.execute(claim,shutdown.clone()).await,Ok(None)=>{},Err(error)=>warn!(error=%error,"drain assignment claim failed")}}
        }
    }
    async fn execute(&self, claim: ClaimedDrainJob, shutdown: CancellationToken) {
        let Some(strategy) = self.registry.drain(&claim.job.strategy_key) else {
            let _ = self
                .repository
                .fail(
                    claim.job.job_id,
                    claim.lease_token,
                    "drain_strategy_unavailable",
                    "assigned drain strategy is unavailable",
                    false,
                )
                .await;
            return;
        };
        if strategy.descriptor().contract_version != claim.job.strategy_contract_version {
            let _ = self
                .repository
                .fail(
                    claim.job.job_id,
                    claim.lease_token,
                    "drain_contract_mismatch",
                    "assigned drain contract version is incompatible",
                    false,
                )
                .await;
            return;
        }
        let request = DrainRequest {
            strategy_key: claim.job.strategy_key.clone(),
            cutoff: claim.job.cutoff,
            dry_run: claim.job.dry_run,
            mode: match claim.job.mode.as_str() {
                "drain" => DrainMode::Drain,
                "reconcile" => DrainMode::Reconcile,
                _ => {
                    let _ = self
                        .repository
                        .fail(
                            claim.job.job_id,
                            claim.lease_token,
                            "drain_mode_invalid",
                            "persisted drain mode is invalid",
                            false,
                        )
                        .await;
                    return;
                }
            },
            execution: ExecutionSelector {
                required_worker_id: claim.job.required_worker_id.clone(),
                required_deployment: claim.job.required_deployment.clone(),
            },
        };
        let execution_shutdown = shutdown.child_token();
        let heartbeat_shutdown = execution_shutdown.clone();
        let heartbeat = self.heartbeat_loop(&claim, heartbeat_shutdown);
        tokio::pin!(heartbeat);
        let execution = strategy.execute_drain(
            DrainContext {
                pool: self.pool.clone(),
                job_id: claim.job.job_id,
                lease_token: claim.lease_token,
                worker_id: Arc::from(self.worker_id.clone()),
                shutdown: execution_shutdown.clone(),
            },
            request,
        );
        tokio::pin!(execution);
        let result = tokio::select! {result=&mut execution=>result,result=&mut heartbeat=>{execution_shutdown.cancel();Err(DrainExecutionError::new("drain_lease_lost",result.err().map(|e|e.to_string()).unwrap_or_else(||"drain heartbeat ended".into()),true))},_=shutdown.cancelled()=>{execution_shutdown.cancel();Err(DrainExecutionError::new("drain_worker_shutdown","worker shutdown interrupted drain",true))}};
        execution_shutdown.cancel();
        match result {
            Ok(outcome) => {
                if !self
                    .repository
                    .complete(claim.job.job_id, claim.lease_token, &outcome)
                    .await
                    .unwrap_or(false)
                {
                    warn!(job_id=%claim.job.job_id,"drain completion lease was lost");
                }
            }
            Err(error) => {
                if let Err(report) = self
                    .repository
                    .fail(
                        claim.job.job_id,
                        claim.lease_token,
                        error.code,
                        &error.message,
                        error.retryable,
                    )
                    .await
                {
                    warn!(job_id=%claim.job.job_id,error=%report,"failed to report drain failure");
                }
            }
        }
    }
    async fn heartbeat_loop(
        &self,
        claim: &ClaimedDrainJob,
        shutdown: CancellationToken,
    ) -> Result<()> {
        let mut interval = tokio::time::interval(Duration::from_secs(15));
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        interval.tick().await;
        loop {
            tokio::select! {_=shutdown.cancelled()=>return Ok(()),_=interval.tick()=>if !self.repository.heartbeat(claim.job.job_id,claim.lease_token).await.context("heartbeat drain lease")?{anyhow::bail!("drain lease lost");}}
        }
    }
}
