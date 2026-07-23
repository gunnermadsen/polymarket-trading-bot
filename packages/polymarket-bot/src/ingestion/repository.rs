use std::{collections::BTreeMap, str::FromStr, time::Duration};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde_json::Value;
use sqlx::{FromRow, PgPool, Postgres, QueryBuilder, Row, Transaction};
use uuid::Uuid;

use crate::ingestion::job::{
    ArtifactCompletion, ArtifactDisposition, ArtifactSpec, BackfillArtifact,
    BackfillArtifactStatus, BackfillCheckpoint, BackfillEventLevel, BackfillFailureKind,
    BackfillJob, BackfillJobEvent, BackfillJobStatus, BackfillJobSummary, BackfillProgress,
    BatchWriteResult, BinanceAggregateTradeRecord, BinanceOneSecondKlineRecord, BtcIntervalMarket,
    BtcOutcome, BtcReferenceFact, BtcResolutionCandidate, ClaimedJob, IngesterKey,
    PreparedArtifact, TrainingReadiness, ValidatedBackfillRequest, WorkerControl,
};

const MAX_DATABASE_BATCH_ROWS: usize = 4_000;

#[derive(Clone)]
pub struct IngestionRepository {
    pool: PgPool,
}

impl IngestionRepository {
    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn enqueue(&self, request: &ValidatedBackfillRequest) -> Result<BackfillJob> {
        let persisted_request = request.persisted_request();
        let progress = serde_json::to_value(BackfillProgress {
            expected_work_units: request.expected_work_units,
            ..BackfillProgress::default()
        })?;
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin backfill enqueue transaction")?;
        let inserted = sqlx::query_as::<_, BackfillJobRow>(
            r#"
            INSERT INTO polymarket.backfill_jobs (
              job_id, ingester_key, request_version, status, range_start, range_end,
              idempotency_key, request, progress, checkpoint, summary, attempt,
              max_attempts, next_attempt_at, requested_at, updated_at
            )
            VALUES (
              gen_random_uuid(), $1, $2, 'queued', $3, $4, $5, $6, $7,
              '{}'::jsonb, '{}'::jsonb, 0, 3, now(), now(), now()
            )
            ON CONFLICT (ingester_key, idempotency_key)
              WHERE idempotency_key IS NOT NULL
            DO NOTHING
            RETURNING job_id, ingester_key, request_version, status, range_start, range_end,
              idempotency_key, request, progress, checkpoint, summary, attempt, max_attempts,
              next_attempt_at, worker_id, lease_token, lease_expires_at, heartbeat_at,
              cancel_requested_at, requested_at, started_at, completed_at, error, updated_at,
              lookback_days, min_trade_usd
            "#,
        )
        .bind(request.ingester.as_str())
        .bind(request.request_version)
        .bind(request.range_start)
        .bind(request.range_end)
        .bind(&request.idempotency_key)
        .bind(&persisted_request)
        .bind(&progress)
        .fetch_optional(&mut *tx)
        .await
        .context("failed to enqueue backfill job")?;

        let (row, created) = if let Some(row) = inserted {
            (row, true)
        } else {
            let row = sqlx::query_as::<_, BackfillJobRow>(
                r#"
                SELECT job_id, ingester_key, request_version, status, range_start, range_end,
                  idempotency_key, request, progress, checkpoint, summary, attempt, max_attempts,
                  next_attempt_at, worker_id, lease_token, lease_expires_at, heartbeat_at,
                  cancel_requested_at, requested_at, started_at, completed_at, error, updated_at,
                  lookback_days, min_trade_usd
                FROM polymarket.backfill_jobs
                WHERE ingester_key = $1 AND idempotency_key = $2
                FOR UPDATE
                "#,
            )
            .bind(request.ingester.as_str())
            .bind(&request.idempotency_key)
            .fetch_one(&mut *tx)
            .await
            .context("failed to load idempotent backfill job")?;
            if row.request_version != request.request_version
                || row.range_start != Some(request.range_start)
                || row.range_end != Some(request.range_end)
                || row.request != persisted_request
            {
                bail!(
                    "idempotency key {} already belongs to a different {} request",
                    request.idempotency_key,
                    request.ingester
                );
            }
            (row, false)
        };

        if created {
            sqlx::query(
                r#"
                INSERT INTO polymarket.backfill_job_events (
                  event_id, job_id, timestamp_utc, level, message, metadata
                )
                VALUES (gen_random_uuid(), $1, now(), 'info', 'backfill job queued', $2)
                "#,
            )
            .bind(row.job_id)
            .bind(serde_json::json!({
                "ingester": request.ingester,
                "request_version": request.request_version,
                "range_start": request.range_start,
                "range_end": request.range_end,
            }))
            .execute(&mut *tx)
            .await
            .context("failed to record backfill enqueue event")?;
        }
        tx.commit()
            .await
            .context("failed to commit backfill enqueue transaction")?;
        row.try_into()
    }

    pub async fn list(&self, limit: i64) -> Result<Vec<BackfillJob>> {
        sqlx::query_as::<_, BackfillJobRow>(
            r#"
            SELECT job_id, ingester_key, request_version, status, range_start, range_end,
              idempotency_key, request, progress, checkpoint, summary, attempt, max_attempts,
              next_attempt_at, worker_id, lease_token, lease_expires_at, heartbeat_at,
              cancel_requested_at, requested_at, started_at, completed_at, error, updated_at,
              lookback_days, min_trade_usd
            FROM polymarket.backfill_jobs
            ORDER BY requested_at DESC, job_id DESC
            LIMIT $1
            "#,
        )
        .bind(limit.clamp(1, 500))
        .fetch_all(&self.pool)
        .await
        .context("failed to list backfill jobs")?
        .into_iter()
        .map(TryInto::try_into)
        .collect()
    }

    pub async fn get(&self, job_id: Uuid) -> Result<Option<BackfillJob>> {
        sqlx::query_as::<_, BackfillJobRow>(
            r#"
            SELECT job_id, ingester_key, request_version, status, range_start, range_end,
              idempotency_key, request, progress, checkpoint, summary, attempt, max_attempts,
              next_attempt_at, worker_id, lease_token, lease_expires_at, heartbeat_at,
              cancel_requested_at, requested_at, started_at, completed_at, error, updated_at,
              lookback_days, min_trade_usd
            FROM polymarket.backfill_jobs
            WHERE job_id = $1
            "#,
        )
        .bind(job_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to get backfill job")?
        .map(TryInto::try_into)
        .transpose()
    }

    pub async fn list_events(&self, job_id: Uuid, limit: i64) -> Result<Vec<BackfillJobEvent>> {
        sqlx::query_as::<_, BackfillJobEventRow>(
            r#"
            SELECT event_id, job_id, timestamp_utc, level, message, metadata
            FROM polymarket.backfill_job_events
            WHERE job_id = $1
            ORDER BY timestamp_utc DESC, event_id DESC
            LIMIT $2
            "#,
        )
        .bind(job_id)
        .bind(limit.clamp(1, 1_000))
        .fetch_all(&self.pool)
        .await
        .context("failed to list backfill job events")?
        .into_iter()
        .map(TryInto::try_into)
        .collect()
    }

    pub async fn append_event(
        &self,
        job_id: Uuid,
        level: BackfillEventLevel,
        message: &str,
        metadata: Value,
    ) -> Result<()> {
        require_json_object(&metadata, "event metadata")?;
        sqlx::query(
            r#"
            INSERT INTO polymarket.backfill_job_events (
              event_id, job_id, timestamp_utc, level, message, metadata
            )
            VALUES (gen_random_uuid(), $1, now(), $2, $3, $4)
            "#,
        )
        .bind(job_id)
        .bind(level.as_str())
        .bind(message)
        .bind(metadata)
        .execute(&self.pool)
        .await
        .context("failed to append backfill job event")?;
        Ok(())
    }

    pub async fn request_cancel(&self, job_id: Uuid) -> Result<Option<BackfillJob>> {
        sqlx::query_as::<_, BackfillJobRow>(
            r#"
            UPDATE polymarket.backfill_jobs
            SET status = CASE
                  WHEN status = 'queued' THEN 'cancelled'
                  WHEN status = 'running' THEN 'cancel_requested'
                  ELSE status
                END,
                cancel_requested_at = CASE
                  WHEN status IN ('queued', 'running') THEN COALESCE(cancel_requested_at, now())
                  ELSE cancel_requested_at
                END,
                completed_at = CASE WHEN status = 'queued' THEN now() ELSE completed_at END,
                worker_id = CASE WHEN status = 'queued' THEN NULL ELSE worker_id END,
                lease_token = CASE WHEN status = 'queued' THEN NULL ELSE lease_token END,
                lease_expires_at = CASE WHEN status = 'queued' THEN NULL ELSE lease_expires_at END,
                heartbeat_at = CASE WHEN status = 'queued' THEN NULL ELSE heartbeat_at END,
                updated_at = CASE WHEN status IN ('queued', 'running') THEN now() ELSE updated_at END
            WHERE job_id = $1
            RETURNING job_id, ingester_key, request_version, status, range_start, range_end,
              idempotency_key, request, progress, checkpoint, summary, attempt, max_attempts,
              next_attempt_at, worker_id, lease_token, lease_expires_at, heartbeat_at,
              cancel_requested_at, requested_at, started_at, completed_at, error, updated_at,
              lookback_days, min_trade_usd
            "#,
        )
        .bind(job_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to request backfill cancellation")?
        .map(TryInto::try_into)
        .transpose()
    }

    pub async fn claim_next(
        &self,
        worker_id: &str,
        lease_duration: Duration,
    ) -> Result<Option<ClaimedJob>> {
        let worker_id = worker_id.trim();
        if worker_id.is_empty() {
            bail!("backfill worker_id must not be empty");
        }
        let lease_seconds = positive_duration_seconds(lease_duration, "lease duration")?;
        let lease_token = Uuid::new_v4();
        let supported = IngesterKey::ALL
            .iter()
            .map(|key| key.as_str().to_string())
            .collect::<Vec<_>>();
        let row = sqlx::query_as::<_, BackfillJobRow>(
            r#"
            WITH candidate AS (
              SELECT job_id, status
              FROM polymarket.backfill_jobs
              WHERE ingester_key = ANY($1)
                AND (
                  (status = 'queued' AND next_attempt_at <= now() AND attempt < max_attempts)
                  OR
                  (status IN ('running', 'cancel_requested')
                    AND (lease_expires_at IS NULL OR lease_expires_at <= now()))
                )
              ORDER BY
                CASE WHEN status = 'cancel_requested' THEN 0 ELSE 1 END,
                next_attempt_at, requested_at, job_id
              LIMIT 1
              FOR UPDATE SKIP LOCKED
            )
            UPDATE polymarket.backfill_jobs AS job
            SET status = CASE
                  WHEN candidate.status = 'cancel_requested' THEN 'cancel_requested'
                  ELSE 'running'
                END,
                started_at = COALESCE(job.started_at, now()),
                completed_at = NULL,
                attempt = CASE
                  WHEN candidate.status = 'cancel_requested' THEN job.attempt
                  ELSE LEAST(job.attempt + 1, job.max_attempts)
                END,
                worker_id = $2,
                lease_token = $3,
                lease_expires_at = now() + ($4::bigint * interval '1 second'),
                heartbeat_at = now(),
                error = CASE WHEN candidate.status = 'queued' THEN NULL ELSE job.error END,
                updated_at = now()
            FROM candidate
            WHERE job.job_id = candidate.job_id
            RETURNING job.job_id, job.ingester_key, job.request_version, job.status,
              job.range_start, job.range_end, job.idempotency_key, job.request, job.progress,
              job.checkpoint, job.summary, job.attempt, job.max_attempts, job.next_attempt_at,
              job.worker_id, job.lease_token, job.lease_expires_at, job.heartbeat_at,
              job.cancel_requested_at, job.requested_at, job.started_at, job.completed_at,
              job.error, job.updated_at, job.lookback_days, job.min_trade_usd
            "#,
        )
        .bind(supported)
        .bind(worker_id)
        .bind(lease_token)
        .bind(lease_seconds)
        .fetch_optional(&self.pool)
        .await
        .context("failed to claim next backfill job")?;
        row.map(|row| {
            let job: BackfillJob = row.try_into()?;
            Ok(ClaimedJob {
                job,
                worker_id: worker_id.to_string(),
                lease_token,
            })
        })
        .transpose()
    }

    pub async fn heartbeat(&self, claim: &ClaimedJob, lease_duration: Duration) -> Result<()> {
        let lease_seconds = positive_duration_seconds(lease_duration, "lease duration")?;
        let affected = sqlx::query(
            r#"
            UPDATE polymarket.backfill_jobs
            SET heartbeat_at = now(),
                lease_expires_at = now() + ($4::bigint * interval '1 second'),
                updated_at = now()
            WHERE job_id = $1 AND worker_id = $2 AND lease_token = $3
              AND status IN ('running', 'cancel_requested')
              AND lease_expires_at > now()
            "#,
        )
        .bind(claim.job.job_id)
        .bind(&claim.worker_id)
        .bind(claim.lease_token)
        .bind(lease_seconds)
        .execute(&self.pool)
        .await
        .context("failed to heartbeat backfill job")?
        .rows_affected();
        require_fenced_update(affected, claim, "heartbeat")
    }

    pub async fn update_progress(
        &self,
        claim: &ClaimedJob,
        progress: &BackfillProgress,
        checkpoint: &BackfillCheckpoint,
    ) -> Result<()> {
        let progress = serde_json::to_value(progress)?;
        let checkpoint = serde_json::to_value(checkpoint)?;
        let affected = sqlx::query(
            r#"
            UPDATE polymarket.backfill_jobs
            SET progress = $4, checkpoint = $5, updated_at = now()
            WHERE job_id = $1 AND worker_id = $2 AND lease_token = $3
              AND status IN ('running', 'cancel_requested')
              AND lease_expires_at > now()
            "#,
        )
        .bind(claim.job.job_id)
        .bind(&claim.worker_id)
        .bind(claim.lease_token)
        .bind(progress)
        .bind(checkpoint)
        .execute(&self.pool)
        .await
        .context("failed to update backfill progress")?
        .rows_affected();
        require_fenced_update(affected, claim, "progress update")
    }

    pub async fn complete(
        &self,
        claim: &ClaimedJob,
        summary: &BackfillJobSummary,
    ) -> Result<BackfillJob> {
        let summary = serde_json::to_value(summary)?;
        let row = sqlx::query_as::<_, BackfillJobRow>(
            r#"
            UPDATE polymarket.backfill_jobs
            SET status = 'completed', completed_at = now(), summary = $4, error = NULL,
                worker_id = NULL, lease_token = NULL, lease_expires_at = NULL,
                heartbeat_at = NULL, updated_at = now()
            WHERE job_id = $1 AND worker_id = $2 AND lease_token = $3
              AND status = 'running' AND lease_expires_at > now()
            RETURNING job_id, ingester_key, request_version, status, range_start, range_end,
              idempotency_key, request, progress, checkpoint, summary, attempt, max_attempts,
              next_attempt_at, worker_id, lease_token, lease_expires_at, heartbeat_at,
              cancel_requested_at, requested_at, started_at, completed_at, error, updated_at,
              lookback_days, min_trade_usd
            "#,
        )
        .bind(claim.job.job_id)
        .bind(&claim.worker_id)
        .bind(claim.lease_token)
        .bind(summary)
        .fetch_optional(&self.pool)
        .await
        .context("failed to complete backfill job")?
        .ok_or_else(|| fenced_error(claim, "completion"))?;
        row.try_into()
    }

    pub async fn fail_or_retry(
        &self,
        claim: &ClaimedJob,
        kind: BackfillFailureKind,
        error: &str,
        retry_after: Duration,
    ) -> Result<BackfillJob> {
        let retry_seconds = i64::try_from(retry_after.as_secs())
            .context("backfill retry delay exceeds Postgres interval range")?;
        let permanent = kind == BackfillFailureKind::Permanent;
        let row = sqlx::query_as::<_, BackfillJobRow>(
            r#"
            UPDATE polymarket.backfill_jobs
            SET status = CASE
                  WHEN status = 'cancel_requested' THEN 'cancelled'
                  WHEN $4 OR attempt >= max_attempts THEN 'failed'
                  ELSE 'queued'
                END,
                completed_at = CASE
                  WHEN status = 'cancel_requested' OR $4 OR attempt >= max_attempts THEN now()
                  ELSE NULL
                END,
                next_attempt_at = CASE
                  WHEN status = 'cancel_requested' OR $4 OR attempt >= max_attempts
                    THEN next_attempt_at
                  ELSE now() + ($6::bigint * interval '1 second')
                END,
                error = $5,
                worker_id = NULL, lease_token = NULL, lease_expires_at = NULL,
                heartbeat_at = NULL, updated_at = now()
            WHERE job_id = $1 AND worker_id = $2 AND lease_token = $3
              AND status IN ('running', 'cancel_requested') AND lease_expires_at > now()
            RETURNING job_id, ingester_key, request_version, status, range_start, range_end,
              idempotency_key, request, progress, checkpoint, summary, attempt, max_attempts,
              next_attempt_at, worker_id, lease_token, lease_expires_at, heartbeat_at,
              cancel_requested_at, requested_at, started_at, completed_at, error, updated_at,
              lookback_days, min_trade_usd
            "#,
        )
        .bind(claim.job.job_id)
        .bind(&claim.worker_id)
        .bind(claim.lease_token)
        .bind(permanent)
        .bind(error)
        .bind(retry_seconds)
        .fetch_optional(&self.pool)
        .await
        .context("failed to record backfill failure")?
        .ok_or_else(|| fenced_error(claim, "failure transition"))?;
        row.try_into()
    }

    pub async fn mark_cancelled(
        &self,
        claim: &ClaimedJob,
        summary: &BackfillJobSummary,
    ) -> Result<BackfillJob> {
        let summary = serde_json::to_value(summary)?;
        let row = sqlx::query_as::<_, BackfillJobRow>(
            r#"
            UPDATE polymarket.backfill_jobs
            SET status = 'cancelled', cancel_requested_at = COALESCE(cancel_requested_at, now()),
                completed_at = now(), summary = $4, worker_id = NULL, lease_token = NULL,
                lease_expires_at = NULL, heartbeat_at = NULL, updated_at = now()
            WHERE job_id = $1 AND worker_id = $2 AND lease_token = $3
              AND status IN ('running', 'cancel_requested') AND lease_expires_at > now()
            RETURNING job_id, ingester_key, request_version, status, range_start, range_end,
              idempotency_key, request, progress, checkpoint, summary, attempt, max_attempts,
              next_attempt_at, worker_id, lease_token, lease_expires_at, heartbeat_at,
              cancel_requested_at, requested_at, started_at, completed_at, error, updated_at,
              lookback_days, min_trade_usd
            "#,
        )
        .bind(claim.job.job_id)
        .bind(&claim.worker_id)
        .bind(claim.lease_token)
        .bind(summary)
        .fetch_optional(&self.pool)
        .await
        .context("failed to cancel backfill job")?
        .ok_or_else(|| fenced_error(claim, "cancellation"))?;
        row.try_into()
    }

    pub async fn is_cancel_requested(&self, claim: &ClaimedJob) -> Result<WorkerControl> {
        let status = sqlx::query_scalar::<_, String>(
            r#"
            SELECT status
            FROM polymarket.backfill_jobs
            WHERE job_id = $1 AND worker_id = $2 AND lease_token = $3
              AND status IN ('running', 'cancel_requested') AND lease_expires_at > now()
            "#,
        )
        .bind(claim.job.job_id)
        .bind(&claim.worker_id)
        .bind(claim.lease_token)
        .fetch_optional(&self.pool)
        .await
        .context("failed to inspect backfill cancellation")?;
        Ok(match status.as_deref() {
            Some("running") => WorkerControl::Continue,
            Some("cancel_requested") => WorkerControl::CancelRequested,
            _ => WorkerControl::LeaseLost,
        })
    }

    pub async fn prepare_artifact(
        &self,
        claim: &ClaimedJob,
        spec: &ArtifactSpec,
    ) -> Result<PreparedArtifact> {
        if spec.job_id != claim.job.job_id || spec.ingester.as_str() != claim.job.ingester_key {
            bail!("artifact identity does not match the claimed backfill job");
        }
        validate_artifact_spec(spec)?;
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin artifact preparation transaction")?;
        require_active_lease(&mut tx, claim).await?;
        let existing = sqlx::query_as::<_, BackfillArtifactRow>(
            r#"
            SELECT artifact_id, job_id, ingester_key, logical_key, provider, source_uri,
              source_date, checksum_algorithm, expected_checksum, actual_checksum,
              compressed_bytes, record_count, minimum_source_timestamp, maximum_source_timestamp,
              status, metadata, created_at, updated_at, completed_at
            FROM polymarket.backfill_artifacts
            WHERE provider = $1 AND logical_key = $2
            FOR UPDATE
            "#,
        )
        .bind(&spec.provider)
        .bind(&spec.logical_key)
        .fetch_optional(&mut *tx)
        .await
        .context("failed to inspect existing backfill artifact")?;

        let (row, disposition) = if let Some(existing) = existing {
            validate_existing_artifact(&existing, spec)?;
            if existing.status == "completed" {
                (existing, ArtifactDisposition::AlreadyCompleted)
            } else if existing.job_id == spec.job_id && existing.status != "failed" {
                (existing, ArtifactDisposition::Process)
            } else {
                let row = sqlx::query_as::<_, BackfillArtifactRow>(
                    r#"
                    UPDATE polymarket.backfill_artifacts
                    SET job_id = $2, ingester_key = $3, source_uri = $4, source_date = $5,
                        expected_checksum = COALESCE(expected_checksum, $6), actual_checksum = NULL,
                        compressed_bytes = NULL, record_count = NULL,
                        minimum_source_timestamp = NULL, maximum_source_timestamp = NULL,
                        status = 'pending', metadata = metadata || $7, completed_at = NULL,
                        updated_at = now()
                    WHERE artifact_id = $1 AND status <> 'completed'
                    RETURNING artifact_id, job_id, ingester_key, logical_key, provider, source_uri,
                      source_date, checksum_algorithm, expected_checksum, actual_checksum,
                      compressed_bytes, record_count, minimum_source_timestamp,
                      maximum_source_timestamp, status, metadata, created_at, updated_at, completed_at
                    "#,
                )
                .bind(existing.artifact_id)
                .bind(spec.job_id)
                .bind(spec.ingester.as_str())
                .bind(&spec.source_uri)
                .bind(spec.source_date)
                .bind(&spec.expected_checksum)
                .bind(&spec.metadata)
                .fetch_one(&mut *tx)
                .await
                .context("failed to reset retryable backfill artifact")?;
                (row, ArtifactDisposition::Process)
            }
        } else {
            let row = sqlx::query_as::<_, BackfillArtifactRow>(
                r#"
                INSERT INTO polymarket.backfill_artifacts (
                  artifact_id, job_id, ingester_key, logical_key, provider, source_uri,
                  source_date, checksum_algorithm, expected_checksum, status, metadata
                )
                VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, 'sha256', $7, 'pending', $8)
                RETURNING artifact_id, job_id, ingester_key, logical_key, provider, source_uri,
                  source_date, checksum_algorithm, expected_checksum, actual_checksum,
                  compressed_bytes, record_count, minimum_source_timestamp,
                  maximum_source_timestamp, status, metadata, created_at, updated_at, completed_at
                "#,
            )
            .bind(spec.job_id)
            .bind(spec.ingester.as_str())
            .bind(&spec.logical_key)
            .bind(&spec.provider)
            .bind(&spec.source_uri)
            .bind(spec.source_date)
            .bind(&spec.expected_checksum)
            .bind(&spec.metadata)
            .fetch_one(&mut *tx)
            .await
            .context("failed to create backfill artifact")?;
            (row, ArtifactDisposition::Process)
        };
        tx.commit()
            .await
            .context("failed to commit artifact preparation")?;
        Ok(PreparedArtifact {
            artifact: row.try_into()?,
            disposition,
        })
    }

    pub async fn set_artifact_status(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        status: BackfillArtifactStatus,
        metadata: Value,
    ) -> Result<BackfillArtifact> {
        require_json_object(&metadata, "artifact metadata")?;
        if matches!(status, BackfillArtifactStatus::Completed) {
            bail!("use complete_artifact to complete an artifact");
        }
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        let current = sqlx::query_as::<_, BackfillArtifactRow>(
            r#"
            SELECT artifact_id, job_id, ingester_key, logical_key, provider, source_uri,
              source_date, checksum_algorithm, expected_checksum, actual_checksum,
              compressed_bytes, record_count, minimum_source_timestamp, maximum_source_timestamp,
              status, metadata, created_at, updated_at, completed_at
            FROM polymarket.backfill_artifacts
            WHERE artifact_id = $1 AND job_id = $2
            FOR UPDATE
            "#,
        )
        .bind(artifact_id)
        .bind(claim.job.job_id)
        .fetch_one(&mut *tx)
        .await
        .context("failed to load backfill artifact for status update")?;
        if current.status == "completed" {
            bail!("completed backfill artifact {artifact_id} is immutable");
        }
        let row = sqlx::query_as::<_, BackfillArtifactRow>(
            r#"
            UPDATE polymarket.backfill_artifacts
            SET status = $3, metadata = metadata || $4, updated_at = now()
            WHERE artifact_id = $1 AND job_id = $2 AND status <> 'completed'
            RETURNING artifact_id, job_id, ingester_key, logical_key, provider, source_uri,
              source_date, checksum_algorithm, expected_checksum, actual_checksum,
              compressed_bytes, record_count, minimum_source_timestamp, maximum_source_timestamp,
              status, metadata, created_at, updated_at, completed_at
            "#,
        )
        .bind(artifact_id)
        .bind(claim.job.job_id)
        .bind(status.as_str())
        .bind(metadata)
        .fetch_one(&mut *tx)
        .await
        .context("failed to update backfill artifact status")?;
        tx.commit().await?;
        row.try_into()
    }

    pub async fn complete_artifact(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        completion: &ArtifactCompletion,
    ) -> Result<BackfillArtifact> {
        validate_sha256(&completion.actual_checksum, "actual checksum")?;
        require_json_object(&completion.metadata, "artifact completion metadata")?;
        if completion.minimum_source_timestamp.is_some()
            != completion.maximum_source_timestamp.is_some()
        {
            bail!("artifact source timestamp bounds must both be present or both be absent");
        }
        if completion.minimum_source_timestamp > completion.maximum_source_timestamp {
            bail!("artifact maximum source timestamp precedes its minimum");
        }
        let compressed_bytes = i64::try_from(completion.compressed_bytes)
            .context("artifact compressed byte count exceeds Postgres bigint")?;
        let record_count = i64::try_from(completion.record_count)
            .context("artifact record count exceeds Postgres bigint")?;
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        let current = sqlx::query_as::<_, BackfillArtifactRow>(
            r#"
            SELECT artifact_id, job_id, ingester_key, logical_key, provider, source_uri,
              source_date, checksum_algorithm, expected_checksum, actual_checksum,
              compressed_bytes, record_count, minimum_source_timestamp, maximum_source_timestamp,
              status, metadata, created_at, updated_at, completed_at
            FROM polymarket.backfill_artifacts
            WHERE artifact_id = $1 AND job_id = $2
            FOR UPDATE
            "#,
        )
        .bind(artifact_id)
        .bind(claim.job.job_id)
        .fetch_one(&mut *tx)
        .await
        .context("failed to load backfill artifact for completion")?;
        if let Some(expected) = current.expected_checksum.as_deref() {
            if expected != completion.actual_checksum {
                bail!(
                    "artifact {artifact_id} checksum conflict: expected {expected}, received {}",
                    completion.actual_checksum
                );
            }
        }
        if current.status == "completed" {
            require_same_completed_artifact(&current, completion, compressed_bytes, record_count)?;
            tx.commit().await?;
            return current.try_into();
        }
        let row = sqlx::query_as::<_, BackfillArtifactRow>(
            r#"
            UPDATE polymarket.backfill_artifacts
            SET status = 'completed', actual_checksum = $3, compressed_bytes = $4,
                record_count = $5, minimum_source_timestamp = $6,
                maximum_source_timestamp = $7, metadata = metadata || $8,
                completed_at = now(), updated_at = now()
            WHERE artifact_id = $1 AND job_id = $2 AND status <> 'completed'
            RETURNING artifact_id, job_id, ingester_key, logical_key, provider, source_uri,
              source_date, checksum_algorithm, expected_checksum, actual_checksum,
              compressed_bytes, record_count, minimum_source_timestamp, maximum_source_timestamp,
              status, metadata, created_at, updated_at, completed_at
            "#,
        )
        .bind(artifact_id)
        .bind(claim.job.job_id)
        .bind(&completion.actual_checksum)
        .bind(compressed_bytes)
        .bind(record_count)
        .bind(completion.minimum_source_timestamp)
        .bind(completion.maximum_source_timestamp)
        .bind(&completion.metadata)
        .fetch_one(&mut *tx)
        .await
        .context("failed to complete backfill artifact")?;
        tx.commit().await?;
        row.try_into()
    }

    pub async fn fail_artifact(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        error: &str,
    ) -> Result<BackfillArtifact> {
        self.set_artifact_status(
            claim,
            artifact_id,
            BackfillArtifactStatus::Failed,
            serde_json::json!({"error": error}),
        )
        .await
    }

    pub async fn persist_market_fact(
        &self,
        claim: &ClaimedJob,
        fact: &BtcReferenceFact,
    ) -> Result<bool> {
        validate_market_fact(fact)?;
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        require_writable_artifact(&mut tx, claim, fact.artifact_id).await?;
        let inserted = sqlx::query(
            r#"
            INSERT INTO polymarket.btc_market_reference_facts (
              fact_id, market_id, artifact_id, fact_type, value, provider,
              source_effective_at, fetched_at, payload_sha256, evidence
            )
            VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (market_id, fact_type, provider) DO NOTHING
            "#,
        )
        .bind(&fact.market_id)
        .bind(fact.artifact_id)
        .bind(fact.fact_type.as_str())
        .bind(fact.value)
        .bind(&fact.provider)
        .bind(fact.source_effective_at)
        .bind(fact.fetched_at)
        .bind(&fact.payload_sha256)
        .bind(&fact.evidence)
        .execute(&mut *tx)
        .await
        .context("failed to persist BTC market reference fact")?
        .rows_affected()
            == 1;
        if !inserted {
            let existing = sqlx::query(
                r#"
                SELECT value, source_effective_at, payload_sha256
                FROM polymarket.btc_market_reference_facts
                WHERE market_id = $1 AND fact_type = $2 AND provider = $3
                "#,
            )
            .bind(&fact.market_id)
            .bind(fact.fact_type.as_str())
            .bind(&fact.provider)
            .fetch_one(&mut *tx)
            .await
            .context("failed to verify existing BTC market reference fact")?;
            let value: Decimal = existing.try_get("value")?;
            let effective_at: DateTime<Utc> = existing.try_get("source_effective_at")?;
            let payload_sha256: String = existing.try_get("payload_sha256")?;
            if value != fact.value
                || effective_at != fact.source_effective_at
                || payload_sha256 != fact.payload_sha256
            {
                bail!(
                    "immutable BTC {} fact conflict for market {}",
                    fact.fact_type.as_str(),
                    fact.market_id
                );
            }
        }
        tx.commit().await?;
        Ok(inserted)
    }

    pub async fn upsert_btc_interval_market(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        market: &BtcIntervalMarket,
    ) -> Result<()> {
        let minimum_order_size = market.minimum_order_size.unwrap_or(Decimal::ZERO);
        let (validation_status, validation_errors) = if market.minimum_order_size.is_some() {
            ("valid", serde_json::json!([]))
        } else {
            (
                "ineligible",
                serde_json::json!(["missing_minimum_order_size"]),
            )
        };
        let fee_rate = decimal_json_field(&market.fee_schedule, &["rate"]);
        let fee_exponent = integer_json_field(&market.fee_schedule, &["exponent"]);
        let fee_taker_only = bool_json_field(&market.fee_schedule, &["takerOnly", "taker_only"]);
        let question = market
            .raw_payload
            .pointer("/markets/0/question")
            .and_then(Value::as_str)
            .or_else(|| market.raw_payload.get("title").and_then(Value::as_str))
            .unwrap_or(&market.event_slug);
        let now = Utc::now();
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        require_writable_artifact(&mut tx, claim, artifact_id).await?;
        sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
            .bind(&market.event_slug)
            .execute(&mut *tx)
            .await
            .context("failed to lock ingested BTC interval market identity")?;
        let result = sqlx::query(
            r#"
            INSERT INTO polymarket.btc_interval_markets (
              market_id, event_id, event_slug, question, series_slug, window_start, window_end,
              condition_id, up_token_id, down_token_id, resolution_source, accepting_orders,
              active, closed, min_tick_size, min_order_size, fee_rate, fee_exponent,
              fee_taker_only, validation_status, validation_errors, discovered_at,
              last_refreshed_at, raw_payload
            )
            VALUES (
              $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,
              $22,$23,$24
            )
            ON CONFLICT (market_id) DO UPDATE SET
              question = EXCLUDED.question,
              accepting_orders = EXCLUDED.accepting_orders,
              active = EXCLUDED.active,
              closed = EXCLUDED.closed,
              min_tick_size = EXCLUDED.min_tick_size,
              min_order_size = EXCLUDED.min_order_size,
              fee_rate = EXCLUDED.fee_rate,
              fee_exponent = EXCLUDED.fee_exponent,
              fee_taker_only = EXCLUDED.fee_taker_only,
              validation_status = EXCLUDED.validation_status,
              validation_errors = EXCLUDED.validation_errors,
              last_refreshed_at = EXCLUDED.last_refreshed_at,
              raw_payload = EXCLUDED.raw_payload,
              updated_at = now()
            WHERE polymarket.btc_interval_markets.event_id = EXCLUDED.event_id
              AND polymarket.btc_interval_markets.event_slug = EXCLUDED.event_slug
              AND polymarket.btc_interval_markets.series_slug = EXCLUDED.series_slug
              AND polymarket.btc_interval_markets.window_start = EXCLUDED.window_start
              AND polymarket.btc_interval_markets.window_end = EXCLUDED.window_end
              AND polymarket.btc_interval_markets.condition_id = EXCLUDED.condition_id
              AND polymarket.btc_interval_markets.up_token_id = EXCLUDED.up_token_id
              AND polymarket.btc_interval_markets.down_token_id = EXCLUDED.down_token_id
              AND polymarket.btc_interval_markets.resolution_source = EXCLUDED.resolution_source
            "#,
        )
        .bind(&market.market_id)
        .bind(&market.event_id)
        .bind(&market.event_slug)
        .bind(question)
        .bind(&market.series_slug)
        .bind(market.window_start)
        .bind(market.window_end)
        .bind(&market.condition_id)
        .bind(&market.up_token_id)
        .bind(&market.down_token_id)
        .bind(&market.resolution_source)
        .bind(market.accepting_orders)
        .bind(market.active)
        .bind(market.closed)
        .bind(market.tick_size)
        .bind(minimum_order_size)
        .bind(fee_rate)
        .bind(fee_exponent)
        .bind(fee_taker_only)
        .bind(validation_status)
        .bind(validation_errors)
        .bind(now)
        .bind(now)
        .bind(&market.raw_payload)
        .execute(&mut *tx)
        .await
        .context("failed to upsert ingested BTC interval market")?;
        if result.rows_affected() != 1 {
            bail!(
                "immutable BTC market identity conflict for market {}",
                market.market_id
            );
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn persist_official_market_resolution(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        market: &BtcIntervalMarket,
        winning_token_id: &str,
        winning_outcome: BtcOutcome,
        observed_at: DateTime<Utc>,
        payload: &Value,
    ) -> Result<()> {
        if !payload.is_object() {
            bail!("official BTC resolution evidence must be a JSON object");
        }
        if market.token_id(winning_outcome) != winning_token_id {
            bail!("official BTC winner does not match the stored outcome token");
        }
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        require_writable_artifact(&mut tx, claim, artifact_id).await?;
        let row = sqlx::query(
            r#"
            SELECT window_end, official_outcome, official_resolved_at,
              official_winning_token_id, official_resolution_source,
              official_resolution_received_at, official_resolution_payload
            FROM polymarket.btc_interval_markets
            WHERE market_id = $1 AND condition_id = $2
            FOR UPDATE
            "#,
        )
        .bind(&market.market_id)
        .bind(&market.condition_id)
        .fetch_one(&mut *tx)
        .await
        .context("failed to locate ingested BTC market for official resolution")?;
        let window_end: DateTime<Utc> = row.try_get("window_end")?;
        if observed_at < window_end {
            bail!("official BTC resolution predates market close");
        }
        let outcome = winning_outcome.as_str();
        let existing_outcome: Option<String> = row.try_get("official_outcome")?;
        let existing_resolved_at: Option<DateTime<Utc>> = row.try_get("official_resolved_at")?;
        let existing_winner: Option<String> = row.try_get("official_winning_token_id")?;
        let existing_source: Option<String> = row.try_get("official_resolution_source")?;
        let existing_received_at: Option<DateTime<Utc>> =
            row.try_get("official_resolution_received_at")?;
        let existing_payload: Option<Value> = row.try_get("official_resolution_payload")?;
        let has_existing = existing_outcome.is_some()
            || existing_resolved_at.is_some()
            || existing_winner.is_some()
            || existing_source.is_some()
            || existing_received_at.is_some()
            || existing_payload.is_some();
        if has_existing
            && (existing_outcome.as_deref() != Some(outcome)
                || existing_winner.as_deref() != Some(winning_token_id))
        {
            bail!(
                "immutable official BTC resolution conflict for market {}",
                market.market_id
            );
        }
        sqlx::query(
            r#"
            UPDATE polymarket.btc_interval_markets
            SET official_outcome = COALESCE(official_outcome, $2),
                official_resolved_at = COALESCE(official_resolved_at, $3),
                official_winning_token_id = COALESCE(official_winning_token_id, $4),
                official_resolution_source = COALESCE(
                  official_resolution_source, 'clob_rest_reconciliation'
                ),
                official_resolution_received_at = COALESCE(
                  official_resolution_received_at, $3
                ),
                official_resolution_payload = COALESCE(official_resolution_payload, $5),
                resolved_outcome = COALESCE(resolved_outcome, $2),
                updated_at = now()
            WHERE market_id = $1
            "#,
        )
        .bind(&market.market_id)
        .bind(outcome)
        .bind(observed_at)
        .bind(winning_token_id)
        .bind(payload)
        .execute(&mut *tx)
        .await
        .context("failed to persist ingested BTC official resolution")?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn resolution_candidates(
        &self,
        range_start: DateTime<Utc>,
        range_end: DateTime<Utc>,
    ) -> Result<Vec<BtcResolutionCandidate>> {
        if range_end <= range_start {
            bail!("resolution candidate range must be non-empty");
        }
        sqlx::query_as::<_, BtcResolutionCandidateRow>(
            r#"
            SELECT event_id, event_slug, series_slug, market_id, condition_id,
              window_start, window_end, up_token_id, down_token_id, min_tick_size,
              min_order_size, resolution_source, accepting_orders, active, closed,
              fee_rate, fee_exponent, fee_taker_only, raw_payload, official_outcome,
              official_winning_token_id, official_resolved_at
            FROM polymarket.btc_interval_markets
            WHERE window_start >= $1 AND window_start < $2
            ORDER BY window_start, market_id
            "#,
        )
        .bind(range_start)
        .bind(range_end)
        .fetch_all(&self.pool)
        .await
        .context("failed to load BTC official resolution candidates")?
        .into_iter()
        .map(TryInto::try_into)
        .collect()
    }

    pub async fn insert_aggregate_trade_batch(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        records: &[BinanceAggregateTradeRecord],
    ) -> Result<BatchWriteResult> {
        if records.is_empty() {
            return Ok(BatchWriteResult::default());
        }
        if records.len() > MAX_DATABASE_BATCH_ROWS {
            bail!("aggregate-trade batch exceeds {MAX_DATABASE_BATCH_ROWS} rows");
        }
        validate_aggregate_trade_batch(records)?;
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        require_writable_artifact(&mut tx, claim, artifact_id).await?;

        let ids = records
            .iter()
            .map(|record| record.aggregate_trade_id)
            .collect::<Vec<_>>();
        let existing = sqlx::query_as::<_, ExistingAggregateTradeRow>(
            r#"
            SELECT symbol, trade_timestamp, aggregate_trade_id, price, quantity,
              first_trade_id, last_trade_id, buyer_maker, best_match, artifact_id
            FROM polymarket.binance_aggregate_trades
            WHERE symbol = 'BTCUSDT' AND aggregate_trade_id = ANY($1)
            "#,
        )
        .bind(&ids)
        .fetch_all(&mut *tx)
        .await
        .context("failed to inspect existing Binance aggregate trades")?;
        for stored in &existing {
            let candidate = records
                .iter()
                .find(|record| record.aggregate_trade_id == stored.aggregate_trade_id)
                .context("stored aggregate-trade identity was absent from its candidate batch")?;
            if !stored.same_as(candidate, artifact_id) {
                bail!(
                    "immutable Binance aggregate-trade conflict for {}:{}",
                    stored.symbol,
                    stored.aggregate_trade_id
                );
            }
        }

        let mut query = QueryBuilder::<Postgres>::new(
            "INSERT INTO polymarket.binance_aggregate_trades (symbol, trade_timestamp, \
             aggregate_trade_id, price, quantity, first_trade_id, last_trade_id, buyer_maker, \
             best_match, artifact_id) ",
        );
        query.push_values(records, |mut row, record| {
            row.push_bind(&record.symbol)
                .push_bind(record.trade_timestamp)
                .push_bind(record.aggregate_trade_id)
                .push_bind(record.price)
                .push_bind(record.quantity)
                .push_bind(record.first_trade_id)
                .push_bind(record.last_trade_id)
                .push_bind(record.buyer_maker)
                .push_bind(record.best_match)
                .push_bind(artifact_id);
        });
        query.push(" ON CONFLICT (symbol, trade_timestamp, aggregate_trade_id) DO NOTHING");
        let inserted = query
            .build()
            .execute(&mut *tx)
            .await
            .context("failed to persist Binance aggregate-trade batch")?
            .rows_affected();
        tx.commit().await?;
        let input = u64::try_from(records.len()).context("aggregate-trade batch size overflow")?;
        Ok(BatchWriteResult {
            input_records: input,
            inserted_records: inserted,
            duplicate_records: input.saturating_sub(inserted),
        })
    }

    pub async fn insert_one_second_kline_batch(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        records: &[BinanceOneSecondKlineRecord],
    ) -> Result<BatchWriteResult> {
        if records.is_empty() {
            return Ok(BatchWriteResult::default());
        }
        if records.len() > MAX_DATABASE_BATCH_ROWS {
            bail!("one-second-kline batch exceeds {MAX_DATABASE_BATCH_ROWS} rows");
        }
        validate_kline_batch(records)?;
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        require_writable_artifact(&mut tx, claim, artifact_id).await?;

        let timestamps = records
            .iter()
            .map(|record| record.open_timestamp)
            .collect::<Vec<_>>();
        let existing = sqlx::query_as::<_, ExistingKlineRow>(
            r#"
            SELECT symbol, open_timestamp, close_timestamp, open_price, high_price,
              low_price, close_price, base_volume, quote_volume, trade_count,
              taker_buy_base_volume, taker_buy_quote_volume, artifact_id
            FROM polymarket.binance_one_second_klines
            WHERE symbol = 'BTCUSDT' AND open_timestamp = ANY($1)
            "#,
        )
        .bind(&timestamps)
        .fetch_all(&mut *tx)
        .await
        .context("failed to inspect existing Binance one-second klines")?;
        for stored in &existing {
            let candidate = records
                .iter()
                .find(|record| record.open_timestamp == stored.open_timestamp)
                .context("stored kline identity was absent from its candidate batch")?;
            if !stored.same_as(candidate, artifact_id) {
                bail!(
                    "immutable Binance one-second-kline conflict for {}:{}",
                    stored.symbol,
                    stored.open_timestamp
                );
            }
        }

        let mut query = QueryBuilder::<Postgres>::new(
            "INSERT INTO polymarket.binance_one_second_klines (symbol, open_timestamp, \
             close_timestamp, open_price, high_price, low_price, close_price, base_volume, \
             quote_volume, trade_count, taker_buy_base_volume, taker_buy_quote_volume, \
             artifact_id) ",
        );
        query.push_values(records, |mut row, record| {
            row.push_bind(&record.symbol)
                .push_bind(record.open_timestamp)
                .push_bind(record.close_timestamp)
                .push_bind(record.open_price)
                .push_bind(record.high_price)
                .push_bind(record.low_price)
                .push_bind(record.close_price)
                .push_bind(record.base_volume)
                .push_bind(record.quote_volume)
                .push_bind(record.trade_count)
                .push_bind(record.taker_buy_base_volume)
                .push_bind(record.taker_buy_quote_volume)
                .push_bind(artifact_id);
        });
        query.push(" ON CONFLICT (symbol, open_timestamp) DO NOTHING");
        let inserted = query
            .build()
            .execute(&mut *tx)
            .await
            .context("failed to persist Binance one-second-kline batch")?
            .rows_affected();
        tx.commit().await?;
        let input = u64::try_from(records.len()).context("kline batch size overflow")?;
        Ok(BatchWriteResult {
            input_records: input,
            inserted_records: inserted,
            duplicate_records: input.saturating_sub(inserted),
        })
    }

    pub async fn training_readiness(
        &self,
        range_start: DateTime<Utc>,
        range_end: DateTime<Utc>,
    ) -> Result<TrainingReadiness> {
        if range_end <= range_start
            || range_start.timestamp().rem_euclid(300) != 0
            || range_end.timestamp().rem_euclid(300) != 0
        {
            bail!("training-readiness range must be non-empty and five-minute aligned");
        }
        let row = sqlx::query_as::<_, TrainingReadinessRow>(
            r#"
            WITH markets AS MATERIALIZED (
              SELECT market_id, window_start, window_end, validation_status, official_outcome
              FROM polymarket.btc_interval_markets
              WHERE window_start >= $1 AND window_start < $2
            ), facts AS MATERIALIZED (
              SELECT market_id,
                bool_or(fact_type = 'opening_boundary') AS opening_boundary,
                bool_or(fact_type = 'final_price') AS final_price
              FROM polymarket.btc_market_reference_facts
              WHERE source_effective_at >= $1 AND source_effective_at <= $2
              GROUP BY market_id
            ), coverage AS MATERIALIZED (
              SELECT m.market_id,
                EXISTS (
                  SELECT 1
                  FROM polymarket.binance_aggregate_trades t
                  JOIN polymarket.backfill_artifacts a USING (artifact_id)
                  WHERE t.symbol = 'BTCUSDT'
                    AND t.trade_timestamp >= m.window_start
                    AND t.trade_timestamp < m.window_end
                    AND a.status = 'completed'
                ) AS agg_covered,
                COALESCE((
                  SELECT count(*) = 300
                    AND min(k.open_timestamp) = m.window_start
                    AND max(k.open_timestamp) = m.window_end - interval '1 second'
                  FROM polymarket.binance_one_second_klines k
                  JOIN polymarket.backfill_artifacts a USING (artifact_id)
                  WHERE k.symbol = 'BTCUSDT'
                    AND k.open_timestamp >= m.window_start
                    AND k.open_timestamp < m.window_end
                    AND a.status = 'completed'
                ), false) AS kline_covered
              FROM markets m
            )
            SELECT
              count(*) FILTER (WHERE m.validation_status = 'valid')::bigint AS valid_market_identities,
              count(*) FILTER (WHERE COALESCE(f.opening_boundary, false))::bigint AS opening_boundaries,
              count(*) FILTER (WHERE COALESCE(f.final_price, false))::bigint AS final_prices,
              count(*) FILTER (WHERE m.official_outcome IS NOT NULL)::bigint AS official_outcomes,
              count(*) FILTER (WHERE c.agg_covered)::bigint AS aggregate_trade_covered_markets,
              count(*) FILTER (WHERE c.kline_covered)::bigint AS one_second_kline_covered_markets,
              count(*) FILTER (
                WHERE m.validation_status = 'valid' AND m.official_outcome IS NOT NULL
                  AND COALESCE(f.opening_boundary, false)
                  AND c.agg_covered AND c.kline_covered
              )::bigint AS usable_markets,
              (SELECT min(trade_timestamp) FROM polymarket.binance_aggregate_trades
                WHERE symbol = 'BTCUSDT' AND trade_timestamp >= $1 AND trade_timestamp < $2)
                AS aggregate_trade_min_timestamp,
              (SELECT max(trade_timestamp) FROM polymarket.binance_aggregate_trades
                WHERE symbol = 'BTCUSDT' AND trade_timestamp >= $1 AND trade_timestamp < $2)
                AS aggregate_trade_max_timestamp,
              (SELECT min(open_timestamp) FROM polymarket.binance_one_second_klines
                WHERE symbol = 'BTCUSDT' AND open_timestamp >= $1 AND open_timestamp < $2)
                AS one_second_kline_min_timestamp,
              (SELECT max(open_timestamp) FROM polymarket.binance_one_second_klines
                WHERE symbol = 'BTCUSDT' AND open_timestamp >= $1 AND open_timestamp < $2)
                AS one_second_kline_max_timestamp
            FROM markets m
            LEFT JOIN facts f USING (market_id)
            LEFT JOIN coverage c USING (market_id)
            "#,
        )
        .bind(range_start)
        .bind(range_end)
        .fetch_one(&self.pool)
        .await
        .context("failed to calculate BTC ML training readiness")?;
        let expected_markets = (range_end - range_start).num_seconds() / 300;
        let mut missing_by_reason = BTreeMap::new();
        missing_by_reason.insert(
            "invalid_or_missing_market_identity".to_string(),
            expected_markets.saturating_sub(row.valid_market_identities),
        );
        missing_by_reason.insert(
            "missing_opening_boundary".to_string(),
            expected_markets.saturating_sub(row.opening_boundaries),
        );
        missing_by_reason.insert(
            "missing_official_outcome".to_string(),
            expected_markets.saturating_sub(row.official_outcomes),
        );
        missing_by_reason.insert(
            "missing_aggregate_trades".to_string(),
            expected_markets.saturating_sub(row.aggregate_trade_covered_markets),
        );
        missing_by_reason.insert(
            "missing_one_second_klines".to_string(),
            expected_markets.saturating_sub(row.one_second_kline_covered_markets),
        );
        let artifact_rows = sqlx::query(
            r#"
            SELECT status, count(*)::bigint AS count
            FROM polymarket.backfill_artifacts
            WHERE ingester_key = ANY($1)
              AND (source_date IS NULL OR (source_date >= $2 AND source_date < $3))
            GROUP BY status
            "#,
        )
        .bind(
            IngesterKey::ALL
                .iter()
                .map(|key| key.as_str().to_string())
                .collect::<Vec<_>>(),
        )
        .bind(range_start.date_naive())
        .bind(range_end.date_naive())
        .fetch_all(&self.pool)
        .await
        .context("failed to summarize backfill artifact states")?;
        let artifact_status_counts = artifact_rows
            .into_iter()
            .map(|row| {
                Ok((
                    row.try_get::<String, _>("status")?,
                    row.try_get::<i64, _>("count")?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(TrainingReadiness {
            range_start,
            range_end,
            expected_markets,
            valid_market_identities: row.valid_market_identities,
            opening_boundaries: row.opening_boundaries,
            final_prices: row.final_prices,
            official_outcomes: row.official_outcomes,
            aggregate_trade_covered_markets: row.aggregate_trade_covered_markets,
            one_second_kline_covered_markets: row.one_second_kline_covered_markets,
            usable_markets: row.usable_markets,
            aggregate_trade_min_timestamp: row.aggregate_trade_min_timestamp,
            aggregate_trade_max_timestamp: row.aggregate_trade_max_timestamp,
            one_second_kline_min_timestamp: row.one_second_kline_min_timestamp,
            one_second_kline_max_timestamp: row.one_second_kline_max_timestamp,
            missing_by_reason,
            artifact_status_counts,
        })
    }
}

#[derive(Debug, FromRow)]
struct BackfillJobRow {
    job_id: Uuid,
    ingester_key: String,
    request_version: i32,
    status: String,
    range_start: Option<DateTime<Utc>>,
    range_end: Option<DateTime<Utc>>,
    idempotency_key: Option<String>,
    request: Value,
    progress: Value,
    checkpoint: Value,
    summary: Value,
    attempt: i32,
    max_attempts: i32,
    next_attempt_at: DateTime<Utc>,
    worker_id: Option<String>,
    lease_token: Option<Uuid>,
    lease_expires_at: Option<DateTime<Utc>>,
    heartbeat_at: Option<DateTime<Utc>>,
    cancel_requested_at: Option<DateTime<Utc>>,
    requested_at: DateTime<Utc>,
    started_at: Option<DateTime<Utc>>,
    completed_at: Option<DateTime<Utc>>,
    error: Option<String>,
    updated_at: DateTime<Utc>,
    lookback_days: Option<i32>,
    min_trade_usd: Option<Decimal>,
}

impl TryFrom<BackfillJobRow> for BackfillJob {
    type Error = anyhow::Error;

    fn try_from(row: BackfillJobRow) -> Result<Self> {
        Ok(Self {
            job_id: row.job_id,
            ingester_key: row.ingester_key,
            request_version: row.request_version,
            status: BackfillJobStatus::from_str(&row.status)?,
            range_start: row.range_start,
            range_end: row.range_end,
            idempotency_key: row.idempotency_key,
            request: row.request,
            progress: row.progress,
            checkpoint: row.checkpoint,
            summary: row.summary,
            attempt: row.attempt,
            max_attempts: row.max_attempts,
            next_attempt_at: row.next_attempt_at,
            worker_id: row.worker_id,
            lease_token: row.lease_token,
            lease_expires_at: row.lease_expires_at,
            heartbeat_at: row.heartbeat_at,
            cancel_requested_at: row.cancel_requested_at,
            requested_at: row.requested_at,
            started_at: row.started_at,
            completed_at: row.completed_at,
            error: row.error,
            updated_at: row.updated_at,
            lookback_days: row.lookback_days,
            min_trade_usd: row.min_trade_usd,
        })
    }
}

#[derive(Debug, FromRow)]
struct BackfillJobEventRow {
    event_id: Uuid,
    job_id: Uuid,
    timestamp_utc: DateTime<Utc>,
    level: String,
    message: String,
    metadata: Value,
}

impl TryFrom<BackfillJobEventRow> for BackfillJobEvent {
    type Error = anyhow::Error;

    fn try_from(row: BackfillJobEventRow) -> Result<Self> {
        Ok(Self {
            event_id: row.event_id,
            job_id: row.job_id,
            timestamp_utc: row.timestamp_utc,
            level: BackfillEventLevel::from_str(&row.level)?,
            message: row.message,
            metadata: row.metadata,
        })
    }
}

#[derive(Debug, FromRow)]
struct BackfillArtifactRow {
    artifact_id: Uuid,
    job_id: Uuid,
    ingester_key: String,
    logical_key: String,
    provider: String,
    source_uri: String,
    source_date: Option<NaiveDate>,
    checksum_algorithm: String,
    expected_checksum: Option<String>,
    actual_checksum: Option<String>,
    compressed_bytes: Option<i64>,
    record_count: Option<i64>,
    minimum_source_timestamp: Option<DateTime<Utc>>,
    maximum_source_timestamp: Option<DateTime<Utc>>,
    status: String,
    metadata: Value,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    completed_at: Option<DateTime<Utc>>,
}

impl TryFrom<BackfillArtifactRow> for BackfillArtifact {
    type Error = anyhow::Error;

    fn try_from(row: BackfillArtifactRow) -> Result<Self> {
        Ok(Self {
            artifact_id: row.artifact_id,
            job_id: row.job_id,
            ingester_key: row.ingester_key,
            logical_key: row.logical_key,
            provider: row.provider,
            source_uri: row.source_uri,
            source_date: row.source_date,
            checksum_algorithm: row.checksum_algorithm,
            expected_checksum: row.expected_checksum,
            actual_checksum: row.actual_checksum,
            compressed_bytes: row.compressed_bytes,
            record_count: row.record_count,
            minimum_source_timestamp: row.minimum_source_timestamp,
            maximum_source_timestamp: row.maximum_source_timestamp,
            status: BackfillArtifactStatus::from_str(&row.status)?,
            metadata: row.metadata,
            created_at: row.created_at,
            updated_at: row.updated_at,
            completed_at: row.completed_at,
        })
    }
}

#[derive(Debug, FromRow)]
struct BtcResolutionCandidateRow {
    event_id: String,
    event_slug: String,
    series_slug: String,
    market_id: String,
    condition_id: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    up_token_id: String,
    down_token_id: String,
    min_tick_size: Decimal,
    min_order_size: Decimal,
    resolution_source: String,
    accepting_orders: bool,
    active: bool,
    closed: bool,
    fee_rate: Option<Decimal>,
    fee_exponent: Option<i32>,
    fee_taker_only: Option<bool>,
    raw_payload: Value,
    official_outcome: Option<String>,
    official_winning_token_id: Option<String>,
    official_resolved_at: Option<DateTime<Utc>>,
}

impl TryFrom<BtcResolutionCandidateRow> for BtcResolutionCandidate {
    type Error = anyhow::Error;

    fn try_from(row: BtcResolutionCandidateRow) -> Result<Self> {
        let minimum_order_size = (row.min_order_size > Decimal::ZERO).then_some(row.min_order_size);
        Ok(Self {
            market: BtcIntervalMarket {
                event_id: row.event_id,
                event_slug: row.event_slug,
                series_slug: row.series_slug,
                market_id: row.market_id,
                condition_id: row.condition_id,
                window_start: row.window_start,
                window_end: row.window_end,
                up_token_id: row.up_token_id,
                down_token_id: row.down_token_id,
                tick_size: row.min_tick_size,
                minimum_order_size,
                resolution_source: row.resolution_source,
                active: row.active,
                closed: row.closed,
                accepting_orders: row.accepting_orders,
                fees_enabled: row.fee_rate.is_some(),
                fee_schedule: serde_json::json!({
                    "rate": row.fee_rate,
                    "exponent": row.fee_exponent,
                    "takerOnly": row.fee_taker_only,
                }),
                raw_payload: row.raw_payload,
            },
            official_outcome: row.official_outcome,
            official_winning_token_id: row.official_winning_token_id,
            official_resolved_at: row.official_resolved_at,
        })
    }
}

#[derive(Debug, FromRow)]
struct ExistingAggregateTradeRow {
    symbol: String,
    trade_timestamp: DateTime<Utc>,
    aggregate_trade_id: i64,
    price: Decimal,
    quantity: Decimal,
    first_trade_id: i64,
    last_trade_id: i64,
    buyer_maker: bool,
    best_match: bool,
    artifact_id: Uuid,
}

impl ExistingAggregateTradeRow {
    fn same_as(&self, row: &BinanceAggregateTradeRecord, artifact_id: Uuid) -> bool {
        self.symbol == row.symbol
            && self.trade_timestamp == row.trade_timestamp
            && self.aggregate_trade_id == row.aggregate_trade_id
            && self.price == row.price
            && self.quantity == row.quantity
            && self.first_trade_id == row.first_trade_id
            && self.last_trade_id == row.last_trade_id
            && self.buyer_maker == row.buyer_maker
            && self.best_match == row.best_match
            && self.artifact_id == artifact_id
    }
}

#[derive(Debug, FromRow)]
struct ExistingKlineRow {
    symbol: String,
    open_timestamp: DateTime<Utc>,
    close_timestamp: DateTime<Utc>,
    open_price: Decimal,
    high_price: Decimal,
    low_price: Decimal,
    close_price: Decimal,
    base_volume: Decimal,
    quote_volume: Decimal,
    trade_count: i64,
    taker_buy_base_volume: Decimal,
    taker_buy_quote_volume: Decimal,
    artifact_id: Uuid,
}

impl ExistingKlineRow {
    fn same_as(&self, row: &BinanceOneSecondKlineRecord, artifact_id: Uuid) -> bool {
        self.symbol == row.symbol
            && self.open_timestamp == row.open_timestamp
            && self.close_timestamp == row.close_timestamp
            && self.open_price == row.open_price
            && self.high_price == row.high_price
            && self.low_price == row.low_price
            && self.close_price == row.close_price
            && self.base_volume == row.base_volume
            && self.quote_volume == row.quote_volume
            && self.trade_count == row.trade_count
            && self.taker_buy_base_volume == row.taker_buy_base_volume
            && self.taker_buy_quote_volume == row.taker_buy_quote_volume
            && self.artifact_id == artifact_id
    }
}

#[derive(Debug, FromRow)]
struct TrainingReadinessRow {
    valid_market_identities: i64,
    opening_boundaries: i64,
    final_prices: i64,
    official_outcomes: i64,
    aggregate_trade_covered_markets: i64,
    one_second_kline_covered_markets: i64,
    usable_markets: i64,
    aggregate_trade_min_timestamp: Option<DateTime<Utc>>,
    aggregate_trade_max_timestamp: Option<DateTime<Utc>>,
    one_second_kline_min_timestamp: Option<DateTime<Utc>>,
    one_second_kline_max_timestamp: Option<DateTime<Utc>>,
}

async fn require_active_lease(
    transaction: &mut Transaction<'_, Postgres>,
    claim: &ClaimedJob,
) -> Result<()> {
    let valid = sqlx::query_scalar::<_, bool>(
        r#"
        SELECT EXISTS (
          SELECT 1 FROM polymarket.backfill_jobs
          WHERE job_id = $1 AND worker_id = $2 AND lease_token = $3
            AND status IN ('running', 'cancel_requested') AND lease_expires_at > now()
        )
        "#,
    )
    .bind(claim.job.job_id)
    .bind(&claim.worker_id)
    .bind(claim.lease_token)
    .fetch_one(&mut **transaction)
    .await
    .context("failed to validate active backfill lease")?;
    if !valid {
        return Err(fenced_error(claim, "database write"));
    }
    Ok(())
}

async fn require_writable_artifact(
    transaction: &mut Transaction<'_, Postgres>,
    claim: &ClaimedJob,
    artifact_id: Uuid,
) -> Result<()> {
    let valid = sqlx::query_scalar::<_, bool>(
        r#"
        SELECT EXISTS (
          SELECT 1 FROM polymarket.backfill_artifacts
          WHERE artifact_id = $1 AND job_id = $2 AND status = 'ingesting'
        )
        "#,
    )
    .bind(artifact_id)
    .bind(claim.job.job_id)
    .fetch_one(&mut **transaction)
    .await?;
    if !valid {
        bail!(
            "artifact {artifact_id} is not writable by job {}",
            claim.job.job_id
        );
    }
    Ok(())
}

fn decimal_json_field(value: &Value, keys: &[&str]) -> Option<Decimal> {
    let raw = keys.iter().find_map(|key| value.get(*key))?;
    match raw {
        Value::String(value) => Decimal::from_str(value).ok(),
        Value::Number(value) => Decimal::from_str(&value.to_string()).ok(),
        _ => None,
    }
}

fn integer_json_field(value: &Value, keys: &[&str]) -> Option<i32> {
    let raw = keys.iter().find_map(|key| value.get(*key))?;
    match raw {
        Value::Number(value) => value.as_i64().and_then(|value| i32::try_from(value).ok()),
        Value::String(value) => value.parse().ok(),
        _ => None,
    }
}

fn bool_json_field(value: &Value, keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|key| value.get(*key)?.as_bool())
}

fn validate_aggregate_trade_batch(records: &[BinanceAggregateTradeRecord]) -> Result<()> {
    let mut previous_id = None;
    for record in records {
        if record.symbol != "BTCUSDT"
            || record.aggregate_trade_id < 0
            || record.first_trade_id < 0
            || record.last_trade_id < record.first_trade_id
            || record.price <= Decimal::ZERO
            || record.quantity <= Decimal::ZERO
        {
            bail!("invalid Binance aggregate-trade record");
        }
        if previous_id.is_some_and(|previous| record.aggregate_trade_id <= previous) {
            bail!("aggregate-trade batch identifiers must be strictly increasing");
        }
        previous_id = Some(record.aggregate_trade_id);
    }
    Ok(())
}

fn validate_kline_batch(records: &[BinanceOneSecondKlineRecord]) -> Result<()> {
    let mut previous = None;
    for record in records {
        if record.symbol != "BTCUSDT"
            || record.open_price <= Decimal::ZERO
            || record.high_price < record.open_price
            || record.high_price < record.close_price
            || record.high_price < record.low_price
            || record.low_price > record.open_price
            || record.low_price > record.close_price
            || record.base_volume < Decimal::ZERO
            || record.quote_volume < Decimal::ZERO
            || record.trade_count < 0
            || record.taker_buy_base_volume < Decimal::ZERO
            || record.taker_buy_base_volume > record.base_volume
            || record.taker_buy_quote_volume < Decimal::ZERO
            || record.taker_buy_quote_volume > record.quote_volume
            || record.close_timestamp < record.open_timestamp
            || record.close_timestamp >= record.open_timestamp + chrono::Duration::seconds(1)
        {
            bail!("invalid Binance one-second-kline record");
        }
        if previous.is_some_and(|timestamp| record.open_timestamp <= timestamp) {
            bail!("kline batch timestamps must be strictly increasing");
        }
        previous = Some(record.open_timestamp);
    }
    Ok(())
}

fn validate_artifact_spec(spec: &ArtifactSpec) -> Result<()> {
    if spec.logical_key.trim().is_empty()
        || spec.provider.trim().is_empty()
        || spec.source_uri.trim().is_empty()
    {
        bail!("artifact identity fields must not be empty");
    }
    if let Some(checksum) = spec.expected_checksum.as_deref() {
        validate_sha256(checksum, "expected checksum")?;
    }
    require_json_object(&spec.metadata, "artifact metadata")
}

fn validate_existing_artifact(row: &BackfillArtifactRow, spec: &ArtifactSpec) -> Result<()> {
    if row.provider != spec.provider
        || row.logical_key != spec.logical_key
        || row.ingester_key != spec.ingester.as_str()
        || row.source_uri != spec.source_uri
        || row.source_date != spec.source_date
        || (row.expected_checksum.is_some()
            && spec.expected_checksum.is_some()
            && row.expected_checksum != spec.expected_checksum)
    {
        bail!(
            "immutable artifact identity conflict for {}:{}",
            spec.provider,
            spec.logical_key
        );
    }
    Ok(())
}

fn validate_market_fact(fact: &BtcReferenceFact) -> Result<()> {
    if fact.market_id.trim().is_empty()
        || fact.provider.trim().is_empty()
        || fact.value <= Decimal::ZERO
        || fact.fetched_at < fact.source_effective_at
    {
        bail!("invalid BTC market reference fact");
    }
    validate_sha256(&fact.payload_sha256, "fact payload checksum")?;
    require_json_object(&fact.evidence, "fact evidence")
}

fn require_same_completed_artifact(
    row: &BackfillArtifactRow,
    completion: &ArtifactCompletion,
    compressed_bytes: i64,
    record_count: i64,
) -> Result<()> {
    if row.actual_checksum.as_deref() != Some(completion.actual_checksum.as_str())
        || row.compressed_bytes != Some(compressed_bytes)
        || row.record_count != Some(record_count)
        || row.minimum_source_timestamp != completion.minimum_source_timestamp
        || row.maximum_source_timestamp != completion.maximum_source_timestamp
    {
        bail!(
            "completed artifact {} conflicts with repeated completion",
            row.artifact_id
        );
    }
    Ok(())
}

fn validate_sha256(value: &str, name: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a lowercase SHA-256 digest");
    }
    if value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        bail!("{name} must be a lowercase SHA-256 digest");
    }
    Ok(())
}

fn require_json_object(value: &Value, name: &str) -> Result<()> {
    if !value.is_object() {
        bail!("{name} must be a JSON object");
    }
    Ok(())
}

fn positive_duration_seconds(duration: Duration, name: &str) -> Result<i64> {
    let seconds = i64::try_from(duration.as_secs())
        .with_context(|| format!("{name} exceeds Postgres interval range"))?;
    if seconds == 0 {
        bail!("{name} must be at least one second");
    }
    Ok(seconds)
}

fn require_fenced_update(affected: u64, claim: &ClaimedJob, action: &str) -> Result<()> {
    if affected != 1 {
        return Err(fenced_error(claim, action));
    }
    Ok(())
}

fn fenced_error(claim: &ClaimedJob, action: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "backfill job {} lost lease fencing before {action}",
        claim.job.job_id
    )
}
