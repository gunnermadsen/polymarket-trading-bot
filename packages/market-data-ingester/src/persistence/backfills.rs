use std::{collections::BTreeMap, time::Duration};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use sqlx::{FromRow, PgPool, Postgres, Transaction};
use uuid::Uuid;

use crate::domain::{
    backfill_profile, realtime_profile, BackfillFailureKind, BackfillOutcome, BackfillShard,
    IngesterStrategyKey, IsolationClass, ValidatedBackfillRequest, ALLOCATION_CONTRACT_VERSION,
};

const JOB_COLUMNS: &str = r#"
  job_id, parent_job_id, job_kind, strategy_key,
  strategy_contract_version, request_schema_version, canonical_request,
  request_hash, range_start, range_end, shard_key, status, attempt,
  max_attempts, next_attempt_at, assigned_worker_id, required_worker_id,
  required_deployment, lease_token, lease_expires_at, heartbeat_at,
  progress, checkpoint, verified_coverage, summary, last_error_kind,
  last_error_code, last_error_message, requested_at, started_at,
  completed_at, cancel_requested_at, allocation_units, assigned_worker_image_digest,
  assigned_worker_source_revision, created_at, updated_at
"#;

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct BackfillJobRecord {
    pub job_id: Uuid,
    pub parent_job_id: Option<Uuid>,
    pub job_kind: String,
    pub strategy_key: String,
    pub strategy_contract_version: i32,
    pub request_schema_version: i32,
    pub canonical_request: Value,
    pub request_hash: String,
    pub range_start: DateTime<Utc>,
    pub range_end: DateTime<Utc>,
    pub shard_key: Option<String>,
    pub status: String,
    pub attempt: i32,
    pub max_attempts: i32,
    pub next_attempt_at: DateTime<Utc>,
    pub assigned_worker_id: Option<String>,
    pub required_worker_id: Option<String>,
    pub required_deployment: Option<String>,
    pub lease_token: Option<Uuid>,
    pub lease_expires_at: Option<DateTime<Utc>>,
    pub heartbeat_at: Option<DateTime<Utc>>,
    pub progress: Value,
    pub checkpoint: Value,
    pub verified_coverage: Value,
    pub summary: Value,
    pub last_error_kind: Option<String>,
    pub last_error_code: Option<String>,
    pub last_error_message: Option<String>,
    pub requested_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub cancel_requested_at: Option<DateTime<Utc>>,
    pub allocation_units: i32,
    pub assigned_worker_image_digest: Option<String>,
    pub assigned_worker_source_revision: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimedBackfillJob {
    pub job: BackfillJobRecord,
    pub lease_token: Uuid,
}

#[derive(Debug, Clone, FromRow, Serialize)]
pub struct BackfillJobEvent {
    pub event_id: Uuid,
    pub job_id: Uuid,
    pub recorded_at: DateTime<Utc>,
    pub level: String,
    pub event_code: String,
    pub message: String,
    pub metadata: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerRegistration {
    pub worker_id: String,
    pub hostname: String,
    pub worker_contract_version: i32,
    pub supported_strategies: BTreeMap<String, i32>,
    pub maximum_backfills: i32,
    pub capacity_units: i32,
    pub realtime_slot_limit: i32,
    pub allocation_contract_version: i32,
    #[serde(default)]
    pub realtime_strategies: Vec<String>,
    pub image_digest: String,
    pub source_revision: String,
    pub deployment_id: String,
}

impl WorkerRegistration {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("worker_id", self.worker_id.as_str()),
            ("hostname", self.hostname.as_str()),
            ("image_digest", self.image_digest.as_str()),
            ("source_revision", self.source_revision.as_str()),
            ("deployment_id", self.deployment_id.as_str()),
        ] {
            if value.trim().is_empty() || value.len() > 255 {
                bail!("{name} must be non-empty and at most 255 bytes");
            }
        }
        if self.worker_contract_version <= 0
            || self.maximum_backfills <= 0
            || self.maximum_backfills > self.capacity_units
            || !(1..=32).contains(&self.capacity_units)
            || self.realtime_slot_limit != 1
            || self.allocation_contract_version != ALLOCATION_CONTRACT_VERSION
        {
            bail!("worker allocation capacity or contract is invalid");
        }
        if self.supported_strategies.is_empty()
            || self
                .supported_strategies
                .values()
                .any(|version| *version <= 0)
        {
            bail!("worker must register positive versions for at least one strategy");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, FromRow, Serialize)]
pub struct WorkerRecord {
    pub worker_id: String,
    pub hostname: String,
    pub worker_contract_version: i32,
    pub supported_strategies: Value,
    pub maximum_backfills: i32,
    pub active_backfills: i32,
    pub capacity_units: i32,
    pub realtime_slot_limit: i32,
    pub allocation_contract_version: i32,
    pub realtime_strategies: Value,
    pub image_digest: String,
    pub source_revision: String,
    pub deployment_id: String,
    pub lifecycle_state: String,
    pub started_at: DateTime<Utc>,
    pub heartbeat_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, FromRow, Serialize)]
pub struct WorkerAllocationRecord {
    pub worker_id: String,
    pub capacity_units: i32,
    pub allocated_units: i64,
    pub realtime_leases: i64,
    pub backfill_leases: i64,
}

#[derive(Debug, Clone, FromRow, Serialize)]
pub struct WorkerAllocationSummary {
    pub worker_id: String,
    pub hostname: String,
    pub worker_contract_version: i32,
    pub supported_strategies: Value,
    pub maximum_backfills: i32,
    pub active_backfills: i32,
    pub capacity_units: i32,
    pub realtime_slot_limit: i32,
    pub allocation_contract_version: i32,
    pub realtime_strategies: Value,
    pub image_digest: String,
    pub source_revision: String,
    pub deployment_id: String,
    pub lifecycle_state: String,
    pub started_at: DateTime<Utc>,
    pub heartbeat_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub allocated_units: i64,
    pub available_units: i64,
    pub realtime_leases: i64,
    pub backfill_leases: i64,
    pub heartbeat_fresh: bool,
    pub assigned_realtime_strategies: Value,
    pub assigned_backfills: Value,
}

#[derive(Clone)]
pub struct BackfillRepository {
    pool: PgPool,
}

impl BackfillRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn submit(
        &self,
        request: &ValidatedBackfillRequest,
        shards: &[BackfillShard],
    ) -> Result<BackfillJobRecord> {
        if shards.is_empty() {
            bail!("a backfill request must produce at least one shard");
        }
        let canonical = canonical_request(request);
        let allocation_units = backfill_profile(request.strategy_key.as_ref()).capacity_units;
        let request_hash = sha256_json(&canonical)?;
        let mut tx = self
            .pool
            .begin()
            .await
            .context("begin backfill submission")?;
        let parent_query = format!(
            r#"
            INSERT INTO ingester.backfill_jobs (
              job_kind, strategy_key, strategy_contract_version,
              request_schema_version, canonical_request, request_hash,
              range_start, range_end, required_worker_id, required_deployment, allocation_units
            ) VALUES ('request',$1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            ON CONFLICT (request_hash) WHERE job_kind='request' AND legacy_source IS NULL
            DO UPDATE SET updated_at=ingester.backfill_jobs.updated_at
            RETURNING {JOB_COLUMNS}
            "#,
        );
        let parent = sqlx::query_as::<_, BackfillJobRecord>(&parent_query)
            .bind(request.strategy_key.as_ref())
            .bind(request.strategy_contract_version)
            .bind(request.request_schema_version)
            .bind(&canonical)
            .bind(&request_hash)
            .bind(request.range_start)
            .bind(request.range_end)
            .bind(request.execution.required_worker_id.as_deref())
            .bind(request.execution.required_deployment.as_deref())
            .bind(allocation_units)
            .fetch_one(&mut *tx)
            .await
            .context("insert canonical backfill request")?;

        let existing: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM ingester.backfill_jobs WHERE parent_job_id=$1",
        )
        .bind(parent.job_id)
        .fetch_one(&mut *tx)
        .await?;
        if existing == 0 {
            for shard in shards {
                let shard_request = json!({
                    "strategy_key": request.strategy_key,
                    "strategy_contract_version": request.strategy_contract_version,
                    "request_schema_version": request.request_schema_version,
                    "range": {"start": shard.range_start, "end": shard.range_end},
                    "parameters": shard.parameters,
                    "shard_key": shard.shard_key,
                });
                let shard_hash = sha256_json(&shard_request)?;
                sqlx::query(
                    r#"
                    INSERT INTO ingester.backfill_jobs (
                      parent_job_id, job_kind, strategy_key, strategy_contract_version,
                      request_schema_version, canonical_request, request_hash,
                      range_start, range_end, shard_key, required_worker_id,
                      required_deployment, allocation_units
                    ) VALUES ($1,'shard',$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                    "#,
                )
                .bind(parent.job_id)
                .bind(request.strategy_key.as_ref())
                .bind(request.strategy_contract_version)
                .bind(request.request_schema_version)
                .bind(shard_request)
                .bind(shard_hash)
                .bind(shard.range_start)
                .bind(shard.range_end)
                .bind(&shard.shard_key)
                .bind(request.execution.required_worker_id.as_deref())
                .bind(request.execution.required_deployment.as_deref())
                .bind(allocation_units)
                .execute(&mut *tx)
                .await
                .context("insert canonical backfill shard")?;
            }
            append_event(
                &mut tx,
                parent.job_id,
                "info",
                "request_scheduled",
                "backfill request scheduled",
                json!({"shards": shards.len()}),
            )
            .await?;
        }
        tx.commit().await.context("commit backfill submission")?;
        Ok(parent)
    }

    pub async fn get(&self, job_id: Uuid) -> Result<Option<BackfillJobRecord>> {
        let query = format!("SELECT {JOB_COLUMNS} FROM ingester.backfill_jobs WHERE job_id=$1");
        Ok(sqlx::query_as(&query)
            .bind(job_id)
            .fetch_optional(&self.pool)
            .await?)
    }

    pub async fn list(&self, limit: i64) -> Result<Vec<BackfillJobRecord>> {
        let query = format!(
            "SELECT {JOB_COLUMNS} FROM ingester.backfill_jobs WHERE job_kind='request' ORDER BY requested_at DESC,job_id DESC LIMIT $1"
        );
        Ok(sqlx::query_as(&query)
            .bind(limit.clamp(1, 500))
            .fetch_all(&self.pool)
            .await?)
    }

    pub async fn children(&self, parent: Uuid) -> Result<Vec<BackfillJobRecord>> {
        let query = format!(
            "SELECT {JOB_COLUMNS} FROM ingester.backfill_jobs WHERE parent_job_id=$1 ORDER BY shard_key,job_id"
        );
        Ok(sqlx::query_as(&query)
            .bind(parent)
            .fetch_all(&self.pool)
            .await?)
    }

    pub async fn events(&self, job_id: Uuid, limit: i64) -> Result<Vec<BackfillJobEvent>> {
        Ok(sqlx::query_as(
            r#"
            SELECT event_id,job_id,recorded_at,level,event_code,message,metadata
            FROM ingester.backfill_job_events
            WHERE job_id=$1
            ORDER BY recorded_at DESC,event_id DESC LIMIT $2
            "#,
        )
        .bind(job_id)
        .bind(limit.clamp(1, 500))
        .fetch_all(&self.pool)
        .await?)
    }

    pub async fn retry(&self, job_id: Uuid) -> Result<Option<BackfillJobRecord>> {
        let mut tx = self.pool.begin().await?;
        let parent_id: Option<Uuid> = sqlx::query_scalar(
            "SELECT CASE WHEN job_kind='request' THEN job_id ELSE parent_job_id END FROM ingester.backfill_jobs WHERE job_id=$1 FOR UPDATE",
        )
        .bind(job_id)
        .fetch_optional(&mut *tx)
        .await?
        .flatten();
        let Some(parent_id) = parent_id else {
            return Ok(None);
        };
        let changed = sqlx::query(
            r#"
            UPDATE ingester.backfill_jobs SET
              status='queued',attempt=0,next_attempt_at=now(),completed_at=NULL,
              cancel_requested_at=NULL,assigned_worker_id=NULL,lease_token=NULL,
              lease_expires_at=NULL,heartbeat_at=NULL,last_error_kind=NULL,
              last_error_code=NULL,last_error_message=NULL,updated_at=now()
            WHERE parent_job_id=$1 AND status IN ('failed','cancelled')
            "#,
        )
        .bind(parent_id)
        .execute(&mut *tx)
        .await?;
        if changed.rows_affected() > 0 {
            sqlx::query("UPDATE ingester.backfill_jobs SET status='queued',completed_at=NULL,cancel_requested_at=NULL,updated_at=now() WHERE job_id=$1")
                .bind(parent_id).execute(&mut *tx).await?;
            append_event(
                &mut tx,
                parent_id,
                "info",
                "request_retried",
                "terminal backfill shards queued for retry",
                json!({"shards": changed.rows_affected()}),
            )
            .await?;
        }
        tx.commit().await?;
        self.get(parent_id).await
    }

    pub async fn cancel(&self, job_id: Uuid) -> Result<Option<BackfillJobRecord>> {
        let mut tx = self.pool.begin().await?;
        let parent_id: Option<Uuid> = sqlx::query_scalar(
            "SELECT CASE WHEN job_kind='request' THEN job_id ELSE parent_job_id END FROM ingester.backfill_jobs WHERE job_id=$1 FOR UPDATE",
        )
        .bind(job_id)
        .fetch_optional(&mut *tx)
        .await?
        .flatten();
        let Some(parent_id) = parent_id else {
            return Ok(None);
        };
        sqlx::query(
            r#"
            UPDATE ingester.backfill_jobs SET
              status=CASE WHEN status='queued' THEN 'cancelled' ELSE 'cancel_requested' END,
              cancel_requested_at=now(),
              completed_at=CASE WHEN status='queued' THEN now() ELSE completed_at END,
              updated_at=now()
            WHERE (job_id=$1 OR parent_job_id=$1) AND status IN ('queued','running')
            "#,
        )
        .bind(parent_id)
        .execute(&mut *tx)
        .await?;
        append_event(
            &mut tx,
            parent_id,
            "warn",
            "cancel_requested",
            "backfill cancellation requested",
            json!({}),
        )
        .await?;
        aggregate_parent(&mut tx, parent_id).await?;
        tx.commit().await?;
        self.get(parent_id).await
    }

    pub async fn register_worker(&self, worker: &WorkerRegistration) -> Result<WorkerRecord> {
        worker.validate()?;
        let strategies = serde_json::to_value(&worker.supported_strategies)?;
        let realtime = serde_json::to_value(&worker.realtime_strategies)?;
        Ok(sqlx::query_as(
            r#"
            INSERT INTO ingester.workers (
              worker_id,hostname,worker_contract_version,supported_strategies,
              maximum_backfills,realtime_strategies,image_digest,source_revision,deployment_id,
              capacity_units,realtime_slot_limit,allocation_contract_version
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (worker_id) DO UPDATE SET
              hostname=EXCLUDED.hostname,
              worker_contract_version=EXCLUDED.worker_contract_version,
              supported_strategies=EXCLUDED.supported_strategies,
              maximum_backfills=EXCLUDED.maximum_backfills,
              realtime_strategies=EXCLUDED.realtime_strategies,
              image_digest=EXCLUDED.image_digest,
              source_revision=EXCLUDED.source_revision,
              deployment_id=EXCLUDED.deployment_id,
              capacity_units=EXCLUDED.capacity_units,
              realtime_slot_limit=EXCLUDED.realtime_slot_limit,
              allocation_contract_version=EXCLUDED.allocation_contract_version,
              lifecycle_state='active',heartbeat_at=now(),updated_at=now()
            RETURNING *
            "#,
        )
        .bind(&worker.worker_id)
        .bind(&worker.hostname)
        .bind(worker.worker_contract_version)
        .bind(strategies)
        .bind(worker.maximum_backfills)
        .bind(realtime)
        .bind(&worker.image_digest)
        .bind(&worker.source_revision)
        .bind(&worker.deployment_id)
        .bind(worker.capacity_units)
        .bind(worker.realtime_slot_limit)
        .bind(worker.allocation_contract_version)
        .fetch_one(&self.pool)
        .await?)
    }

    pub async fn heartbeat_worker(&self, worker_id: &str) -> Result<bool> {
        let result = sqlx::query(
            r#"
            UPDATE ingester.workers worker SET
              heartbeat_at=now(), updated_at=now(),
              active_backfills=(
                SELECT count(*)::integer FROM ingester.backfill_jobs job
                WHERE job.assigned_worker_id=worker.worker_id
                  AND job.status IN ('running','cancel_requested')
              )
            WHERE worker_id=$1
            "#,
        )
        .bind(worker_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn list_workers(&self) -> Result<Vec<WorkerRecord>> {
        Ok(
            sqlx::query_as("SELECT * FROM ingester.workers ORDER BY worker_id")
                .fetch_all(&self.pool)
                .await?,
        )
    }

    pub async fn list_worker_allocation_summaries(&self) -> Result<Vec<WorkerAllocationSummary>> {
        Ok(sqlx::query_as(
            r#"
            WITH realtime AS (
              SELECT profile.lease_owner AS worker_id,
                count(*)::bigint AS leases,
                count(*)::bigint * 2 AS units,
                jsonb_agg(jsonb_build_object(
                  'strategy_key', profile.strategy_key,
                  'desired_generation', profile.desired_generation,
                  'applied_generation', profile.applied_generation,
                  'observed_state', profile.observed_state,
                  'health_status', profile.health_status,
                  'lease_expires_at', profile.lease_expires_at
                ) ORDER BY profile.strategy_key) AS assignments
              FROM ingester.profiles profile
              WHERE profile.lease_owner IS NOT NULL AND profile.lease_expires_at > now()
              GROUP BY profile.lease_owner
            ), backfill AS (
              SELECT job.assigned_worker_id AS worker_id,
                count(*)::bigint AS leases,
                COALESCE(sum(job.allocation_units),0)::bigint AS units,
                jsonb_agg(jsonb_build_object(
                  'job_id', job.job_id,
                  'strategy_key', job.strategy_key,
                  'status', job.status,
                  'allocation_units', job.allocation_units,
                  'lease_expires_at', job.lease_expires_at
                ) ORDER BY job.job_id) AS assignments
              FROM ingester.backfill_jobs job
              WHERE job.assigned_worker_id IS NOT NULL
                AND job.status IN ('running','cancel_requested')
                AND job.lease_expires_at > now()
              GROUP BY job.assigned_worker_id
            )
            SELECT worker.*,
              COALESCE(realtime.units,0) + COALESCE(backfill.units,0) AS allocated_units,
              GREATEST(0, worker.capacity_units::bigint - COALESCE(realtime.units,0) - COALESCE(backfill.units,0)) AS available_units,
              COALESCE(realtime.leases,0) AS realtime_leases,
              COALESCE(backfill.leases,0) AS backfill_leases,
              worker.lifecycle_state='active' AND worker.heartbeat_at > now()-interval '30 seconds' AS heartbeat_fresh,
              COALESCE(realtime.assignments,'[]'::jsonb) AS assigned_realtime_strategies,
              COALESCE(backfill.assignments,'[]'::jsonb) AS assigned_backfills
            FROM ingester.workers worker
            LEFT JOIN realtime ON realtime.worker_id=worker.worker_id
            LEFT JOIN backfill ON backfill.worker_id=worker.worker_id
            ORDER BY worker.worker_id
            "#,
        )
        .fetch_all(&self.pool)
        .await?)
    }

    pub async fn list_worker_allocations(&self) -> Result<Vec<WorkerAllocationRecord>> {
        Ok(sqlx::query_as(
            r#"
            SELECT worker.worker_id, worker.capacity_units,
              COALESCE(realtime.units,0) + COALESCE(backfill.units,0) AS allocated_units,
              COALESCE(realtime.leases,0) AS realtime_leases,
              COALESCE(backfill.leases,0) AS backfill_leases
            FROM ingester.workers worker
            LEFT JOIN LATERAL (
              SELECT count(*)::bigint AS leases, count(*)::bigint * 2 AS units
              FROM ingester.profiles profile
              WHERE profile.lease_owner=worker.worker_id AND profile.lease_expires_at>now()
            ) realtime ON true
            LEFT JOIN LATERAL (
              SELECT count(*)::bigint AS leases, COALESCE(sum(job.allocation_units),0)::bigint AS units
              FROM ingester.backfill_jobs job
              WHERE job.assigned_worker_id=worker.worker_id
                AND job.status IN ('running','cancel_requested') AND job.lease_expires_at>now()
            ) backfill ON true
            WHERE worker.lifecycle_state='active' AND worker.heartbeat_at>now()-interval '30 seconds'
            ORDER BY worker.worker_id
            "#,
        )
        .fetch_all(&self.pool)
        .await?)
    }

    pub async fn claim(
        &self,
        worker_id: &str,
        lease: Duration,
    ) -> Result<Option<ClaimedBackfillJob>> {
        let lease_seconds = i64::try_from(lease.as_secs()).context("lease duration overflow")?;
        let token = Uuid::new_v4();
        let mut tx = self.pool.begin().await?;
        let expired = sqlx::query_as::<_, (Uuid, Uuid, Option<String>)>(
            r#"
            WITH expired AS (
              SELECT job_id,parent_job_id,assigned_worker_id,status,attempt,max_attempts
              FROM ingester.backfill_jobs
              WHERE job_kind='shard' AND status IN ('running','cancel_requested')
                AND lease_expires_at<=now()
              FOR UPDATE SKIP LOCKED
            )
            UPDATE ingester.backfill_jobs job SET
              status=CASE
                WHEN expired.status='cancel_requested' THEN 'cancelled'
                WHEN expired.attempt>=expired.max_attempts THEN 'failed'
                ELSE 'queued'
              END,
              next_attempt_at=CASE WHEN expired.attempt>=expired.max_attempts THEN job.next_attempt_at ELSE now() END,
              completed_at=CASE WHEN expired.status='cancel_requested' OR expired.attempt>=expired.max_attempts THEN now() ELSE NULL END,
              last_error_kind=CASE WHEN expired.status='cancel_requested' THEN 'cancelled' ELSE 'lease_lost' END,
              last_error_code=CASE WHEN expired.status='cancel_requested' THEN 'cancelled_after_lease_expiry' ELSE 'worker_lease_expired' END,
              last_error_message='worker heartbeat expired before the shard reached a terminal state',
              assigned_worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,heartbeat_at=NULL,updated_at=now()
            FROM expired WHERE job.job_id=expired.job_id
            RETURNING job.job_id,job.parent_job_id,expired.assigned_worker_id
            "#,
        )
        .fetch_all(&mut *tx)
        .await?;
        let mut expired_parents = Vec::new();
        for (expired_job, parent, previous_worker) in expired {
            append_event(
                &mut tx,
                expired_job,
                "warn",
                "worker_lease_expired",
                "expired shard lease recovered by the master",
                json!({"worker_id": previous_worker}),
            )
            .await?;
            if !expired_parents.contains(&parent) {
                expired_parents.push(parent);
            }
        }
        for parent in expired_parents {
            aggregate_parent(&mut tx, parent).await?;
        }
        sqlx::query(
            r#"
            UPDATE ingester.workers worker SET active_backfills=(
              SELECT count(*)::integer FROM ingester.backfill_jobs job
              WHERE job.assigned_worker_id=worker.worker_id
                AND job.status IN ('running','cancel_requested')
            ),updated_at=now()
            WHERE worker_id=$1
            "#,
        )
        .bind(worker_id)
        .execute(&mut *tx)
        .await?;
        let worker = sqlx::query_as::<_, WorkerRecord>(
            "SELECT * FROM ingester.workers WHERE worker_id=$1 FOR UPDATE",
        )
        .bind(worker_id)
        .fetch_optional(&mut *tx)
        .await?;
        let Some(worker) = worker else {
            return Ok(None);
        };
        if worker.lifecycle_state != "active"
            || worker.heartbeat_at < Utc::now() - chrono::Duration::seconds(30)
            || worker.active_backfills >= worker.maximum_backfills
        {
            tx.commit().await?;
            return Ok(None);
        }
        let active_backfill_units: i64 = sqlx::query_scalar(
            "SELECT COALESCE(sum(allocation_units),0)::bigint FROM ingester.backfill_jobs WHERE assigned_worker_id=$1 AND status IN ('running','cancel_requested') AND lease_expires_at>now()",
        )
        .bind(worker_id)
        .fetch_one(&mut *tx)
        .await?;
        let realtime_key: Option<String> = sqlx::query_scalar(
            "SELECT strategy_key FROM ingester.profiles WHERE lease_owner=$1 AND lease_expires_at>now() ORDER BY strategy_key LIMIT 1",
        )
        .bind(worker_id)
        .fetch_optional(&mut *tx)
        .await?;
        let active_realtime = realtime_key
            .as_deref()
            .and_then(|key| key.parse::<IngesterStrategyKey>().ok())
            .map(realtime_profile);
        if active_realtime
            .is_some_and(|profile| profile.isolation == IsolationClass::LatencyCritical)
        {
            tx.commit().await?;
            return Ok(None);
        }
        let realtime_units = active_realtime.map_or(0, |profile| profile.capacity_units);
        let remaining_units = i64::from(worker.capacity_units)
            .saturating_sub(active_backfill_units)
            .saturating_sub(i64::from(realtime_units));
        if remaining_units <= 0 {
            tx.commit().await?;
            return Ok(None);
        }
        let query = r#"
            WITH candidate AS (
              SELECT job.job_id
              FROM ingester.backfill_jobs job
              WHERE job.job_kind='shard' AND job.status='queued'
                AND job.next_attempt_at <= now() AND job.attempt < job.max_attempts
                AND $2::jsonb ? job.strategy_key
                AND ($2::jsonb ->> job.strategy_key)::integer = job.strategy_contract_version
                AND (job.required_worker_id IS NULL OR job.required_worker_id=$1)
                AND (job.required_deployment IS NULL OR job.required_deployment=$3)
                AND job.allocation_units <= $8
                AND NOT (job.allocation_units = $9 AND ($10 OR $11 > 0))
              ORDER BY
                CASE
                  WHEN job.required_worker_id=$1 THEN 0
                  WHEN job.required_deployment=$3 THEN 1
                  ELSE 2
                END,
                job.requested_at,job.job_id
              FOR UPDATE SKIP LOCKED LIMIT 1
            )
            UPDATE ingester.backfill_jobs job SET
              status='running',attempt=attempt+1,assigned_worker_id=$1,
              lease_token=$4,lease_expires_at=now()+make_interval(secs=>$5),
              heartbeat_at=now(),started_at=COALESCE(started_at,now()),updated_at=now(),
              assigned_worker_image_digest=$6,assigned_worker_source_revision=$7,
              last_error_kind=NULL,last_error_code=NULL,last_error_message=NULL
            FROM candidate WHERE job.job_id=candidate.job_id
            RETURNING job.*
            "#;
        let job = sqlx::query_as::<_, BackfillJobRecord>(query)
            .bind(worker_id)
            .bind(&worker.supported_strategies)
            .bind(&worker.deployment_id)
            .bind(token)
            .bind(lease_seconds)
            .bind(&worker.image_digest)
            .bind(&worker.source_revision)
            .bind(remaining_units)
            .bind(worker.capacity_units)
            .bind(active_realtime.is_some())
            .bind(active_backfill_units)
            .fetch_optional(&mut *tx)
            .await?;
        if let Some(job) = &job {
            sqlx::query("UPDATE ingester.workers SET active_backfills=active_backfills+1,updated_at=now() WHERE worker_id=$1")
                .bind(worker_id).execute(&mut *tx).await?;
            append_event(
                &mut tx,
                job.job_id,
                "info",
                "job_assigned",
                "backfill shard assigned",
                json!({"worker_id": worker_id, "attempt": job.attempt}),
            )
            .await?;
            if let Some(parent) = job.parent_job_id {
                sqlx::query("UPDATE ingester.backfill_jobs SET status='running',started_at=COALESCE(started_at,now()),updated_at=now() WHERE job_id=$1 AND status='queued'")
                    .bind(parent).execute(&mut *tx).await?;
            }
        }
        tx.commit().await?;
        Ok(job.map(|job| ClaimedBackfillJob {
            job,
            lease_token: token,
        }))
    }

    pub async fn heartbeat_job(
        &self,
        worker_id: &str,
        job_id: Uuid,
        token: Uuid,
        lease: Duration,
        progress: &Value,
        checkpoint: &Value,
    ) -> Result<bool> {
        let seconds = i64::try_from(lease.as_secs())?;
        let result = sqlx::query(
            r#"
            UPDATE ingester.backfill_jobs SET heartbeat_at=now(),
              lease_expires_at=now()+make_interval(secs=>$4),progress=$5,checkpoint=$6,updated_at=now()
            WHERE job_id=$1 AND assigned_worker_id=$2 AND lease_token=$3
              AND status='running' AND lease_expires_at>now()
            "#,
        )
        .bind(job_id).bind(worker_id).bind(token).bind(seconds).bind(progress).bind(checkpoint)
        .execute(&self.pool).await?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn complete(
        &self,
        worker_id: &str,
        job_id: Uuid,
        token: Uuid,
        outcome: &BackfillOutcome,
    ) -> Result<bool> {
        let mut tx = self.pool.begin().await?;
        let parent: Option<Uuid> = sqlx::query_scalar(
            r#"
            UPDATE ingester.backfill_jobs SET status='completed',
              verified_coverage=$4,summary=$5,completed_at=now(),updated_at=now(),
              assigned_worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,heartbeat_at=NULL
            WHERE job_id=$1 AND assigned_worker_id=$2 AND lease_token=$3
              AND status='running' AND lease_expires_at>now()
            RETURNING parent_job_id
            "#,
        )
        .bind(job_id)
        .bind(worker_id)
        .bind(token)
        .bind(&outcome.verified_coverage)
        .bind(&outcome.summary)
        .fetch_optional(&mut *tx)
        .await?
        .flatten();
        let Some(parent) = parent else {
            tx.rollback().await.ok();
            return Ok(false);
        };
        sqlx::query("UPDATE ingester.workers SET active_backfills=GREATEST(0,active_backfills-1),updated_at=now() WHERE worker_id=$1")
            .bind(worker_id).execute(&mut *tx).await?;
        append_event(
            &mut tx,
            job_id,
            "info",
            "job_completed",
            "backfill shard completed",
            json!({"records_verified": outcome.records_verified}),
        )
        .await?;
        aggregate_parent(&mut tx, parent).await?;
        tx.commit().await?;
        Ok(true)
    }

    pub async fn fail(
        &self,
        worker_id: &str,
        job_id: Uuid,
        token: Uuid,
        kind: BackfillFailureKind,
        code: &str,
        message: &str,
    ) -> Result<bool> {
        let mut tx = self.pool.begin().await?;
        let parent: Option<Uuid> = sqlx::query_scalar(
            r#"
            UPDATE ingester.backfill_jobs SET
              status=CASE
                WHEN status='cancel_requested' OR $4='cancelled' THEN 'cancelled'
                WHEN attempt>=max_attempts OR $4 IN ('invalid_request','integrity') THEN 'failed'
                ELSE 'queued'
              END,
              next_attempt_at=CASE WHEN attempt>=max_attempts THEN next_attempt_at ELSE now()+make_interval(secs=>LEAST(900,5*(2^attempt))) END,
              last_error_kind=$4,last_error_code=left($5,128),last_error_message=left($6,4000),
              completed_at=CASE WHEN status='cancel_requested' OR $4='cancelled' OR attempt>=max_attempts OR $4 IN ('invalid_request','integrity') THEN now() ELSE NULL END,
              assigned_worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,heartbeat_at=NULL,updated_at=now()
            WHERE job_id=$1 AND assigned_worker_id=$2 AND lease_token=$3
              AND status IN ('running','cancel_requested')
            RETURNING parent_job_id
            "#,
        )
        .bind(job_id).bind(worker_id).bind(token).bind(kind.as_str()).bind(code).bind(message)
        .fetch_optional(&mut *tx).await?.flatten();
        let Some(parent) = parent else {
            tx.rollback().await.ok();
            return Ok(false);
        };
        sqlx::query("UPDATE ingester.workers SET active_backfills=GREATEST(0,active_backfills-1),updated_at=now() WHERE worker_id=$1")
            .bind(worker_id).execute(&mut *tx).await?;
        append_event(
            &mut tx,
            job_id,
            "error",
            "job_failed",
            "backfill shard failed",
            json!({"kind": kind.as_str(), "code": code, "message": message}),
        )
        .await?;
        aggregate_parent(&mut tx, parent).await?;
        tx.commit().await?;
        Ok(true)
    }
}

fn canonical_request(request: &ValidatedBackfillRequest) -> Value {
    canonicalize(json!({
        "strategy_key": request.strategy_key,
        "strategy_contract_version": request.strategy_contract_version,
        "request_schema_version": request.request_schema_version,
        "range": {"start": request.range_start, "end": request.range_end},
        "parameters": request.parameters,
        "execution": request.execution,
    }))
}

fn canonicalize(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let ordered: BTreeMap<_, _> = object.into_iter().collect();
            Value::Object(
                ordered
                    .into_iter()
                    .map(|(key, value)| (key, canonicalize(value)))
                    .collect::<Map<_, _>>(),
            )
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize).collect()),
        value => value,
    }
}

fn sha256_json(value: &Value) -> Result<String> {
    let encoded = serde_json::to_vec(value)?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

async fn append_event(
    tx: &mut Transaction<'_, Postgres>,
    job_id: Uuid,
    level: &str,
    code: &str,
    message: &str,
    metadata: Value,
) -> Result<()> {
    sqlx::query("INSERT INTO ingester.backfill_job_events (job_id,level,event_code,message,metadata) VALUES ($1,$2,$3,$4,$5)")
        .bind(job_id).bind(level).bind(code).bind(message).bind(metadata)
        .execute(&mut **tx).await?;
    Ok(())
}

async fn aggregate_parent(tx: &mut Transaction<'_, Postgres>, parent: Uuid) -> Result<()> {
    sqlx::query(
        r#"
        WITH child AS (
          SELECT
            count(*) FILTER (WHERE status='completed') AS completed,
            count(*) FILTER (WHERE status='failed') AS failed,
            count(*) FILTER (WHERE status='cancelled') AS cancelled,
            count(*) FILTER (WHERE status IN ('running','cancel_requested')) AS running,
            count(*) AS total
          FROM ingester.backfill_jobs WHERE parent_job_id=$1
        )
        UPDATE ingester.backfill_jobs parent SET
          status=CASE
            WHEN child.total=child.completed THEN 'completed'
            WHEN child.failed>0 AND child.completed+child.failed+child.cancelled=child.total THEN 'failed'
            WHEN child.cancelled=child.total THEN 'cancelled'
            WHEN child.running>0 THEN 'running'
            ELSE parent.status
          END,
          completed_at=CASE
            WHEN child.completed+child.failed+child.cancelled=child.total THEN now()
            ELSE parent.completed_at
          END,
          summary=jsonb_build_object(
            'total_shards',child.total,'completed_shards',child.completed,
            'failed_shards',child.failed,'cancelled_shards',child.cancelled
          ),updated_at=now()
        FROM child WHERE parent.job_id=$1
        "#,
    )
    .bind(parent).execute(&mut **tx).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_hash_ignores_object_key_order() {
        let first = json!({"b": 2, "a": {"d": 4, "c": 3}});
        let second = json!({"a": {"c": 3, "d": 4}, "b": 2});
        assert_eq!(
            sha256_json(&canonicalize(first)).unwrap(),
            sha256_json(&canonicalize(second)).unwrap()
        );
    }
}
