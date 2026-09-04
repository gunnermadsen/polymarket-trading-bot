use crate::domain::{DrainOutcome, DrainRequest};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

const COLUMNS: &str = "job_id,strategy_key,strategy_contract_version,cutoff,dry_run,status,required_worker_id,required_deployment,assigned_worker_id,lease_token,lease_expires_at,attempt,max_attempts,rows_exported,rows_removed,objects_published,bytes_written,summary,last_error_code,last_error_message,requested_at,started_at,completed_at,cancel_requested_at,updated_at";

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct DrainJobRecord {
    pub job_id: Uuid,
    pub strategy_key: String,
    pub strategy_contract_version: i32,
    pub cutoff: DateTime<Utc>,
    pub dry_run: bool,
    pub status: String,
    pub required_worker_id: Option<String>,
    pub required_deployment: Option<String>,
    pub assigned_worker_id: Option<String>,
    pub lease_token: Option<Uuid>,
    pub lease_expires_at: Option<DateTime<Utc>>,
    pub attempt: i32,
    pub max_attempts: i32,
    pub rows_exported: i64,
    pub rows_removed: i64,
    pub objects_published: i64,
    pub bytes_written: i64,
    pub summary: Value,
    pub last_error_code: Option<String>,
    pub last_error_message: Option<String>,
    pub requested_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub cancel_requested_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct ClaimedDrainJob {
    pub job: DrainJobRecord,
    pub lease_token: Uuid,
}

#[derive(Clone)]
pub struct DrainRepository {
    pool: PgPool,
}
impl DrainRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
    pub async fn submit(&self, request: &DrainRequest, version: i32) -> Result<DrainJobRecord> {
        let q=format!("INSERT INTO ingester.drain_jobs (strategy_key,strategy_contract_version,cutoff,dry_run,required_worker_id,required_deployment) VALUES ($1,$2,$3,$4,$5,$6) RETURNING {COLUMNS}");
        sqlx::query_as(&q)
            .bind(&request.strategy_key)
            .bind(version)
            .bind(request.cutoff)
            .bind(request.dry_run)
            .bind(request.execution.required_worker_id.as_deref())
            .bind(request.execution.required_deployment.as_deref())
            .fetch_one(&self.pool)
            .await
            .context("submit drain job")
    }
    pub async fn get(&self, id: Uuid) -> Result<Option<DrainJobRecord>> {
        let q = format!("SELECT {COLUMNS} FROM ingester.drain_jobs WHERE job_id=$1");
        Ok(sqlx::query_as(&q)
            .bind(id)
            .fetch_optional(&self.pool)
            .await?)
    }
    pub async fn list(&self, limit: i64) -> Result<Vec<DrainJobRecord>> {
        let q=format!("SELECT {COLUMNS} FROM ingester.drain_jobs ORDER BY requested_at DESC,job_id DESC LIMIT $1");
        Ok(sqlx::query_as(&q)
            .bind(limit.clamp(1, 500))
            .fetch_all(&self.pool)
            .await?)
    }
    pub async fn cancel(&self, id: Uuid) -> Result<Option<DrainJobRecord>> {
        let q=format!("UPDATE ingester.drain_jobs SET cancel_requested_at=clock_timestamp(),status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,updated_at=clock_timestamp() WHERE job_id=$1 AND status IN ('queued','running') RETURNING {COLUMNS}");
        Ok(sqlx::query_as(&q)
            .bind(id)
            .fetch_optional(&self.pool)
            .await?)
    }
    pub async fn retry(&self, id: Uuid) -> Result<Option<DrainJobRecord>> {
        let q=format!("UPDATE ingester.drain_jobs SET status='queued',assigned_worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,last_error_code=NULL,last_error_message=NULL,cancel_requested_at=NULL,updated_at=clock_timestamp() WHERE job_id=$1 AND status IN ('failed','cancelled') AND attempt<max_attempts RETURNING {COLUMNS}");
        Ok(sqlx::query_as(&q)
            .bind(id)
            .fetch_optional(&self.pool)
            .await?)
    }
    pub async fn claim(
        &self,
        worker: &str,
        deployment: &str,
        supported: &[String],
    ) -> Result<Option<ClaimedDrainJob>> {
        let mut tx = self.pool.begin().await?;
        let claim_columns = COLUMNS.replacen("job_id", "job.job_id", 1);
        let q = format!(
            r#"WITH candidate AS (SELECT job_id FROM ingester.drain_jobs WHERE strategy_key=ANY($3::text[]) AND cancel_requested_at IS NULL AND (required_worker_id IS NULL OR required_worker_id=$1) AND (required_deployment IS NULL OR required_deployment=$2) AND (status='queued' OR (status='running' AND lease_expires_at<clock_timestamp())) AND attempt<max_attempts ORDER BY requested_at,job_id FOR UPDATE SKIP LOCKED LIMIT 1) UPDATE ingester.drain_jobs job SET status='running',attempt=attempt+1,assigned_worker_id=$1,lease_token=gen_random_uuid(),lease_expires_at=clock_timestamp()+interval '45 seconds',started_at=COALESCE(started_at,clock_timestamp()),updated_at=clock_timestamp() FROM candidate WHERE job.job_id=candidate.job_id RETURNING {claim_columns}"#
        );
        let job: Option<DrainJobRecord> = sqlx::query_as(&q)
            .bind(worker)
            .bind(deployment)
            .bind(supported)
            .fetch_optional(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(job.map(|job| ClaimedDrainJob {
            lease_token: job.lease_token.expect("claimed drain lease"),
            job,
        }))
    }
    pub async fn heartbeat(&self, id: Uuid, lease: Uuid) -> Result<bool> {
        Ok(sqlx::query("UPDATE ingester.drain_jobs SET lease_expires_at=clock_timestamp()+interval '45 seconds',updated_at=clock_timestamp() WHERE job_id=$1 AND lease_token=$2 AND status='running' AND cancel_requested_at IS NULL").bind(id).bind(lease).execute(&self.pool).await?.rows_affected()==1)
    }
    pub async fn complete(&self, id: Uuid, lease: Uuid, o: &DrainOutcome) -> Result<bool> {
        Ok(sqlx::query("UPDATE ingester.drain_jobs SET status='completed',rows_exported=$3,rows_removed=$4,objects_published=$5,bytes_written=$6,summary=$7,completed_at=clock_timestamp(),lease_token=NULL,lease_expires_at=NULL,updated_at=clock_timestamp() WHERE job_id=$1 AND lease_token=$2 AND status='running' AND cancel_requested_at IS NULL").bind(id).bind(lease).bind(o.rows_exported).bind(o.rows_removed).bind(o.objects_published).bind(o.bytes_written).bind(&o.summary).execute(&self.pool).await?.rows_affected()==1)
    }
    pub async fn fail(
        &self,
        id: Uuid,
        lease: Uuid,
        code: &str,
        message: &str,
        retryable: bool,
    ) -> Result<bool> {
        Ok(sqlx::query("UPDATE ingester.drain_jobs SET status=CASE WHEN cancel_requested_at IS NOT NULL THEN 'cancelled' WHEN $5 AND attempt<max_attempts THEN 'queued' ELSE 'failed' END,last_error_code=$3,last_error_message=$4,assigned_worker_id=NULL,lease_token=NULL,lease_expires_at=NULL,updated_at=clock_timestamp() WHERE job_id=$1 AND lease_token=$2 AND status='running'").bind(id).bind(lease).bind(code).bind(message).bind(retryable).execute(&self.pool).await?.rows_affected()==1)
    }
}
