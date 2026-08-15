use std::{
    collections::{BTreeMap, HashSet},
    str::FromStr,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde_json::Value;
use sqlx::{FromRow, PgPool, Postgres, QueryBuilder, Row, Transaction};
use uuid::Uuid;

use crate::ingestion::cryptohft_binance_l2::{
    validate_representative_day_quality, validate_representative_day_quality_for_market,
    CryptoHftBinanceMarket,
};
use crate::ingestion::job::{
    ArtifactCompletion, ArtifactDisposition, ArtifactSpec, BackfillArtifact,
    BackfillArtifactStatus, BackfillCheckpoint, BackfillEventLevel, BackfillFailureKind,
    BackfillJob, BackfillJobEvent, BackfillJobStatus, BackfillJobSummary, BackfillProgress,
    BatchWriteResult, BinanceAggregateTradeRecord, BinanceBtcusdtOpenInterestRecord,
    BinanceL2OneSecondFeature, BinanceOneSecondKlineRecord, BtcExecutionSnapshot,
    BtcIntervalMarket, BtcOrderbookArchiveEvent, BtcOrderbookMarketScope, BtcOutcome,
    BtcReferenceFact, BtcResolutionCandidate, ChainlinkBtcusdArchiveTick,
    ChainlinkBtcusdOneMinuteCandle, ClaimedJob, IngesterKey, PolygonChainlinkBtcusdOracleRound,
    PreparedArtifact, TrainingReadiness, ValidatedBackfillRequest, WorkerControl,
};

const MAX_DATABASE_BATCH_ROWS: usize = 4_000;
const POSTGRES_MAX_BIND_PARAMETERS: usize = 65_535;
const ORDERBOOK_EVENT_INSERT_COLUMNS: usize = 18;
const MAX_ORDERBOOK_EVENT_INSERT_ROWS: usize =
    POSTGRES_MAX_BIND_PARAMETERS / ORDERBOOK_EVENT_INSERT_COLUMNS;
const EXECUTION_SNAPSHOT_INSERT_COLUMNS: usize = 35;
const MAX_EXECUTION_SNAPSHOT_INSERT_ROWS: usize =
    POSTGRES_MAX_BIND_PARAMETERS / EXECUTION_SNAPSHOT_INSERT_COLUMNS;
const POLYGON_CHAINLINK_INSERT_COLUMNS: usize = 15;
const MAX_POLYGON_CHAINLINK_INSERT_ROWS: usize =
    POSTGRES_MAX_BIND_PARAMETERS / POLYGON_CHAINLINK_INSERT_COLUMNS;
const BINANCE_L2_FEATURE_INSERT_COLUMNS: usize = 49;
const MAX_BINANCE_L2_FEATURE_INSERT_ROWS: usize = 1_000;
const BINANCE_FUTURES_L2_FEATURE_SCHEMA_VERSION: &str = "binance-btcusdt-l2-one-second-features-v1";
const BINANCE_SPOT_L2_FEATURE_SCHEMA_VERSION: &str =
    "binance-spot-btcusdt-l2-one-second-features-v1";

#[derive(Clone)]
pub struct IngestionRepository {
    pool: PgPool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrderbookEventCursor {
    pub provider_received_at: DateTime<Utc>,
    pub source_row_number: i64,
}

#[derive(Debug)]
pub struct RawOrderbookEventPage {
    pub events: Vec<BtcOrderbookArchiveEvent>,
    pub next_cursor: Option<OrderbookEventCursor>,
}

impl IngestionRepository {
    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn reserve_cryptohft_request_slot(&self) -> Result<DateTime<Utc>> {
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin CryptoHFT request-budget transaction")?;
        let row = sqlx::query(
            r#"
            SELECT GREATEST(next_request_at, clock_timestamp()) AS scheduled_at,
              spacing_milliseconds
            FROM polymarket.cryptohft_request_budget
            WHERE provider = 'cryptohftdata'
            FOR UPDATE
            "#,
        )
        .fetch_one(&mut *tx)
        .await
        .context("failed to lock the CryptoHFT request budget")?;
        let scheduled_at: DateTime<Utc> = row.try_get("scheduled_at")?;
        let spacing_milliseconds: i32 = row.try_get("spacing_milliseconds")?;
        if spacing_milliseconds != 1_100 {
            bail!("CryptoHFT request budget does not match the materialization contract");
        }
        let updated = sqlx::query(
            r#"
            UPDATE polymarket.cryptohft_request_budget
            SET next_request_at = $1 + ($2 * interval '1 millisecond'), updated_at = now()
            WHERE provider = 'cryptohftdata'
            "#,
        )
        .bind(scheduled_at)
        .bind(spacing_milliseconds)
        .execute(&mut *tx)
        .await
        .context("failed to reserve the CryptoHFT request slot")?;
        if updated.rows_affected() != 1 {
            bail!("CryptoHFT request budget row was not updated");
        }
        tx.commit()
            .await
            .context("failed to commit the CryptoHFT request slot")?;
        Ok(scheduled_at)
    }

    pub async fn require_binance_l2_representative_source_gate(
        &self,
        representative_logical_key: &str,
    ) -> Result<()> {
        let metadata = sqlx::query_scalar::<_, Value>(
            r#"
            SELECT metadata
            FROM polymarket.backfill_artifacts
            WHERE ingester_key = 'binance_btcusdt_l2_one_second_features'
              AND provider = 'cryptohftdata'
              AND logical_key = $1
              AND status = 'completed'
            "#,
        )
        .bind(representative_logical_key)
        .fetch_optional(&self.pool)
        .await
        .context("failed to inspect the Binance L2 representative source audit")?
        .with_context(|| {
            format!(
                "Binance L2 representative source audit {representative_logical_key} is not completed"
            )
        })?;
        validate_representative_day_quality(&metadata)
    }

    pub async fn require_binance_spot_l2_representative_source_gate(
        &self,
        representative_logical_key: &str,
    ) -> Result<()> {
        let metadata = sqlx::query_scalar::<_, Value>(
            r#"
            SELECT metadata
            FROM polymarket.backfill_artifacts
            WHERE ingester_key = 'binance_spot_btcusdt_l2_one_second_features'
              AND provider = 'cryptohftdata'
              AND logical_key = $1
              AND status = 'completed'
            "#,
        )
        .bind(representative_logical_key)
        .fetch_optional(&self.pool)
        .await
        .context("failed to inspect the Binance spot L2 representative source audit")?
        .with_context(|| {
            format!(
                "Binance spot L2 representative source audit {representative_logical_key} is not completed"
            )
        })?;
        validate_representative_day_quality_for_market(CryptoHftBinanceMarket::Spot, &metadata)
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

    pub async fn orderbook_market_scope(
        &self,
        file_start: DateTime<Utc>,
        file_end: DateTime<Utc>,
    ) -> Result<Vec<BtcOrderbookMarketScope>> {
        if file_end <= file_start {
            bail!("orderbook market scope range must be non-empty");
        }
        sqlx::query_as::<_, BtcOrderbookMarketScopeRow>(
            r#"
            SELECT market_id, condition_id, up_token_id, down_token_id, window_start, window_end
            FROM polymarket.btc_interval_markets
            WHERE validation_status = 'valid'
              AND window_start >= $1 AND window_start <= $2
            ORDER BY window_start, market_id
            "#,
        )
        .bind(file_start)
        .bind(file_end)
        .fetch_all(&self.pool)
        .await
        .context("failed to load BTC orderbook market scope")
        .map(|rows| rows.into_iter().map(Into::into).collect())
    }

    pub async fn execution_snapshot_market_scope(
        &self,
        window_start: DateTime<Utc>,
        window_end: DateTime<Utc>,
    ) -> Result<Vec<BtcOrderbookMarketScope>> {
        if window_end <= window_start {
            bail!("execution snapshot market scope range must be non-empty");
        }
        sqlx::query_as::<_, BtcOrderbookMarketScopeRow>(
            r#"
            SELECT market_id, condition_id, up_token_id, down_token_id, window_start, window_end
            FROM polymarket.btc_interval_markets
            WHERE validation_status = 'valid'
              AND window_start >= $1 AND window_start < $2
            ORDER BY window_start, market_id
            "#,
        )
        .bind(window_start)
        .bind(window_end)
        .fetch_all(&self.pool)
        .await
        .context("failed to load BTC execution snapshot market scope")
        .map(|rows| rows.into_iter().map(Into::into).collect())
    }

    pub async fn completed_raw_orderbook_artifacts(
        &self,
        logical_keys: &[String],
    ) -> Result<Vec<BackfillArtifact>> {
        if logical_keys.is_empty() {
            return Ok(Vec::new());
        }
        sqlx::query_as::<_, BackfillArtifactRow>(
            r#"
            SELECT a.artifact_id, a.job_id, a.ingester_key, a.logical_key, a.provider,
              a.source_uri, a.source_date, a.checksum_algorithm, a.expected_checksum,
              a.actual_checksum, a.compressed_bytes, a.record_count,
              a.minimum_source_timestamp, a.maximum_source_timestamp, a.status, a.metadata,
              a.created_at, a.updated_at, a.completed_at
            FROM polymarket.backfill_artifacts a
            WHERE a.ingester_key = 'polymarket_btc_five_minute_orderbooks'
              AND a.provider = 'pmxt_v2'
              AND a.logical_key = ANY($1)
              AND a.status = 'completed'
              AND EXISTS (
                SELECT 1
                FROM polymarket.btc_orderbook_archive_events e
                WHERE e.artifact_id = a.artifact_id
                LIMIT 1
              )
            ORDER BY a.logical_key
            "#,
        )
        .bind(logical_keys)
        .fetch_all(&self.pool)
        .await
        .context("failed to load completed raw PMXT artifacts")?
        .into_iter()
        .map(TryInto::try_into)
        .collect()
    }

    pub async fn raw_orderbook_event_page(
        &self,
        artifact_id: Uuid,
        condition_ids: &[String],
        cursor: Option<&OrderbookEventCursor>,
        limit: i64,
    ) -> Result<RawOrderbookEventPage> {
        if condition_ids.is_empty() {
            return Ok(RawOrderbookEventPage {
                events: Vec::new(),
                next_cursor: None,
            });
        }
        let cursor_received_at = cursor.map(|value| value.provider_received_at);
        let cursor_row_number = cursor.map(|value| value.source_row_number);
        let rows = sqlx::query_as::<_, ExistingOrderbookEventRow>(
            r#"
            SELECT artifact_id, source_row_number, provider_received_at, source_timestamp,
              condition_id, asset_id, event_type, bids, asks, price, size, side, best_bid,
              best_ask, fee_rate_bps, transaction_hash, old_tick_size, new_tick_size
            FROM polymarket.btc_orderbook_archive_events
            WHERE artifact_id = $1
              AND condition_id = ANY($2)
              AND (
                $3::timestamptz IS NULL
                OR (provider_received_at, source_row_number) > ($3, $4::bigint)
              )
            ORDER BY provider_received_at, source_row_number
            LIMIT $5
            "#,
        )
        .bind(artifact_id)
        .bind(condition_ids)
        .bind(cursor_received_at)
        .bind(cursor_row_number)
        .bind(limit.clamp(1, 20_000))
        .fetch_all(&self.pool)
        .await
        .context("failed to page raw PMXT events for compact reconstruction")?;
        let next_cursor = rows.last().map(|row| OrderbookEventCursor {
            provider_received_at: row.provider_received_at,
            source_row_number: row.source_row_number,
        });
        Ok(RawOrderbookEventPage {
            events: rows.into_iter().map(Into::into).collect(),
            next_cursor,
        })
    }

    pub async fn insert_execution_snapshot_batch(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        schema_version: &str,
        records: &[BtcExecutionSnapshot],
    ) -> Result<BatchWriteResult> {
        if records.is_empty() {
            return Ok(BatchWriteResult::default());
        }
        if records.len() > MAX_DATABASE_BATCH_ROWS {
            bail!("execution-snapshot batch exceeds {MAX_DATABASE_BATCH_ROWS} rows");
        }
        if schema_version.trim().is_empty() {
            bail!("execution-snapshot schema version must not be empty");
        }
        validate_execution_snapshot_batch(records)?;
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        require_writable_artifact(&mut tx, claim, artifact_id).await?;
        let mut inserted = 0u64;
        for chunk in records.chunks(MAX_EXECUTION_SNAPSHOT_INSERT_ROWS) {
            let mut query = QueryBuilder::<Postgres>::new(
                "INSERT INTO polymarket.btc_market_capacity_execution_snapshots (market_id, sampled_at, \
                 artifact_id, schema_version, up_source_row_number, up_source_timestamp, \
                 up_provider_received_at, up_best_bid, up_best_ask, up_best_bid_size, \
                 up_best_ask_size, up_bid_depth, up_ask_depth, up_ask_vwap_1, up_ask_vwap_5, \
                 up_ask_vwap_10, up_ask_vwap_15, up_ask_vwap_20, up_imbalance, \
                 down_source_row_number, down_source_timestamp, \
                 down_provider_received_at, down_best_bid, down_best_ask, down_best_bid_size, \
                 down_best_ask_size, down_bid_depth, down_ask_depth, down_ask_vwap_1, \
                 down_ask_vwap_5, down_ask_vwap_10, down_ask_vwap_15, down_ask_vwap_20, \
                 down_imbalance, quality_flags) ",
            );
            query.push_values(chunk, |mut row, record| {
                row.push_bind(&record.market_id)
                    .push_bind(record.sampled_at)
                    .push_bind(artifact_id)
                    .push_bind(schema_version)
                    .push_bind(record.up_source_row_number)
                    .push_bind(record.up_source_timestamp)
                    .push_bind(record.up_provider_received_at)
                    .push_bind(record.up_best_bid)
                    .push_bind(record.up_best_ask)
                    .push_bind(record.up_best_bid_size)
                    .push_bind(record.up_best_ask_size)
                    .push_bind(record.up_bid_depth)
                    .push_bind(record.up_ask_depth)
                    .push_bind(record.up_ask_vwap_1)
                    .push_bind(record.up_ask_vwap_5)
                    .push_bind(record.up_ask_vwap_10)
                    .push_bind(record.up_ask_vwap_15)
                    .push_bind(record.up_ask_vwap_20)
                    .push_bind(record.up_imbalance)
                    .push_bind(record.down_source_row_number)
                    .push_bind(record.down_source_timestamp)
                    .push_bind(record.down_provider_received_at)
                    .push_bind(record.down_best_bid)
                    .push_bind(record.down_best_ask)
                    .push_bind(record.down_best_bid_size)
                    .push_bind(record.down_best_ask_size)
                    .push_bind(record.down_bid_depth)
                    .push_bind(record.down_ask_depth)
                    .push_bind(record.down_ask_vwap_1)
                    .push_bind(record.down_ask_vwap_5)
                    .push_bind(record.down_ask_vwap_10)
                    .push_bind(record.down_ask_vwap_15)
                    .push_bind(record.down_ask_vwap_20)
                    .push_bind(record.down_imbalance)
                    .push_bind(record.quality_flags);
            });
            query.push(" ON CONFLICT (market_id, sampled_at) DO NOTHING");
            inserted = inserted.saturating_add(
                query
                    .build()
                    .execute(&mut *tx)
                    .await
                    .context("failed to persist compact execution snapshots")?
                    .rows_affected(),
            );
        }
        tx.commit().await?;
        batch_write_result(records.len(), inserted, "execution-snapshot")
    }

    pub async fn record_raw_orderbook_replacements(
        &self,
        claim: &ClaimedJob,
        replacement_artifact_id: Uuid,
        source_artifacts: &[BackfillArtifact],
    ) -> Result<()> {
        if source_artifacts.is_empty() {
            return Ok(());
        }
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        for source in source_artifacts {
            sqlx::query(
                r#"
                INSERT INTO polymarket.backfill_materialization_retention_events (
                  source_artifact_id, replacement_artifact_id, materialization, action,
                  source_record_count, metadata
                )
                VALUES ($1, $2, 'polymarket.btc_orderbook_archive_events', 'replaced', $3, $4)
                ON CONFLICT (source_artifact_id, materialization, action) DO NOTHING
                "#,
            )
            .bind(source.artifact_id)
            .bind(replacement_artifact_id)
            .bind(source.record_count.unwrap_or_default())
            .bind(serde_json::json!({
                "replacement_schema": "btc5m-decision-book-90-140s-5s-v1",
                "replacement_ingester": IngesterKey::PolymarketBtcFiveMinuteExecutionSnapshots,
            }))
            .execute(&mut *tx)
            .await
            .context("failed to record raw PMXT replacement lineage")?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn insert_orderbook_event_batch(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        records: &[BtcOrderbookArchiveEvent],
    ) -> Result<BatchWriteResult> {
        if records.is_empty() {
            return Ok(BatchWriteResult::default());
        }
        if records.len() > MAX_DATABASE_BATCH_ROWS {
            bail!("orderbook-event batch exceeds {MAX_DATABASE_BATCH_ROWS} rows");
        }
        validate_orderbook_event_batch(records)?;
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        require_writable_artifact(&mut tx, claim, artifact_id).await?;
        let row_numbers = records
            .iter()
            .map(|record| record.source_row_number)
            .collect::<Vec<_>>();
        let existing = sqlx::query_as::<_, ExistingOrderbookEventRow>(
            r#"
            SELECT artifact_id, source_row_number, provider_received_at, source_timestamp,
              condition_id, asset_id, event_type, bids, asks, price, size, side, best_bid,
              best_ask, fee_rate_bps, transaction_hash, old_tick_size, new_tick_size
            FROM polymarket.btc_orderbook_archive_events
            WHERE artifact_id = $1 AND source_row_number = ANY($2)
            "#,
        )
        .bind(artifact_id)
        .bind(&row_numbers)
        .fetch_all(&mut *tx)
        .await
        .context("failed to inspect existing PMXT orderbook events")?;
        for stored in &existing {
            let candidate = records
                .iter()
                .find(|record| record.source_row_number == stored.source_row_number)
                .context("stored PMXT row identity was absent from candidate batch")?;
            if !stored.same_as(candidate, artifact_id) {
                bail!(
                    "immutable PMXT orderbook-event conflict for {}:{}",
                    artifact_id,
                    stored.source_row_number
                );
            }
        }

        let mut inserted = 0u64;
        for chunk in records.chunks(MAX_ORDERBOOK_EVENT_INSERT_ROWS) {
            let mut query = QueryBuilder::<Postgres>::new(
                "INSERT INTO polymarket.btc_orderbook_archive_events (artifact_id, \
                 source_row_number, provider_received_at, source_timestamp, condition_id, asset_id, \
                 event_type, bids, asks, price, size, side, best_bid, best_ask, fee_rate_bps, \
                 transaction_hash, old_tick_size, new_tick_size) ",
            );
            query.push_values(chunk, |mut row, record| {
                row.push_bind(artifact_id)
                    .push_bind(record.source_row_number)
                    .push_bind(record.provider_received_at)
                    .push_bind(record.source_timestamp)
                    .push_bind(&record.condition_id)
                    .push_bind(&record.asset_id)
                    .push_bind(&record.event_type)
                    .push_bind(&record.bids)
                    .push_bind(&record.asks)
                    .push_bind(record.price)
                    .push_bind(record.size)
                    .push_bind(&record.side)
                    .push_bind(record.best_bid)
                    .push_bind(record.best_ask)
                    .push_bind(record.fee_rate_bps)
                    .push_bind(&record.transaction_hash)
                    .push_bind(record.old_tick_size)
                    .push_bind(record.new_tick_size);
            });
            query.push(
                " ON CONFLICT (artifact_id, source_row_number, provider_received_at) DO NOTHING",
            );
            inserted = inserted.saturating_add(
                query
                    .build()
                    .execute(&mut *tx)
                    .await
                    .context("failed to persist PMXT orderbook-event batch")?
                    .rows_affected(),
            );
        }
        tx.commit().await?;
        batch_write_result(records.len(), inserted, "orderbook-event")
    }

    pub async fn insert_chainlink_tick_batch(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        records: &[ChainlinkBtcusdArchiveTick],
    ) -> Result<BatchWriteResult> {
        if records.is_empty() {
            return Ok(BatchWriteResult::default());
        }
        if records.len() > MAX_DATABASE_BATCH_ROWS {
            bail!("Chainlink tick batch exceeds {MAX_DATABASE_BATCH_ROWS} rows");
        }
        validate_chainlink_tick_batch(records)?;
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        require_writable_artifact(&mut tx, claim, artifact_id).await?;
        let timestamps = records
            .iter()
            .map(|record| record.source_timestamp)
            .collect::<Vec<_>>();
        let existing = sqlx::query_as::<_, ExistingChainlinkTickRow>(
            r#"
            SELECT feed_id, source_timestamp, valid_from_timestamp, price, bid, ask,
              report_sha256, artifact_id
            FROM polymarket.chainlink_btcusd_archive_ticks
            WHERE feed_id = $1 AND source_timestamp = ANY($2)
            "#,
        )
        .bind(&records[0].feed_id)
        .bind(&timestamps)
        .fetch_all(&mut *tx)
        .await
        .context("failed to inspect existing Chainlink ticks")?;
        for stored in &existing {
            let candidate = records
                .iter()
                .find(|record| record.source_timestamp == stored.source_timestamp)
                .context("stored Chainlink tick identity was absent from candidate batch")?;
            if !stored.same_as(candidate, artifact_id) {
                bail!(
                    "immutable Chainlink tick conflict for {}:{}",
                    stored.feed_id,
                    stored.source_timestamp
                );
            }
        }

        let mut query = QueryBuilder::<Postgres>::new(
            "INSERT INTO polymarket.chainlink_btcusd_archive_ticks (feed_id, \
             source_timestamp, valid_from_timestamp, price, bid, ask, report_sha256, artifact_id) ",
        );
        query.push_values(records, |mut row, record| {
            row.push_bind(&record.feed_id)
                .push_bind(record.source_timestamp)
                .push_bind(record.valid_from_timestamp)
                .push_bind(record.price)
                .push_bind(record.bid)
                .push_bind(record.ask)
                .push_bind(&record.report_sha256)
                .push_bind(artifact_id);
        });
        query.push(" ON CONFLICT (feed_id, source_timestamp) DO NOTHING");
        let inserted = query
            .build()
            .execute(&mut *tx)
            .await
            .context("failed to persist Chainlink tick batch")?
            .rows_affected();
        tx.commit().await?;
        batch_write_result(records.len(), inserted, "Chainlink tick")
    }

    pub async fn insert_chainlink_candle_batch(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        records: &[ChainlinkBtcusdOneMinuteCandle],
    ) -> Result<BatchWriteResult> {
        if records.is_empty() {
            return Ok(BatchWriteResult::default());
        }
        if records.len() > MAX_DATABASE_BATCH_ROWS {
            bail!("Chainlink candle batch exceeds {MAX_DATABASE_BATCH_ROWS} rows");
        }
        validate_chainlink_candle_batch(records)?;
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        require_writable_artifact(&mut tx, claim, artifact_id).await?;
        let timestamps = records
            .iter()
            .map(|record| record.open_timestamp)
            .collect::<Vec<_>>();
        let existing = sqlx::query_as::<_, ExistingChainlinkCandleRow>(
            r#"
            SELECT symbol, open_timestamp, close_timestamp, open_price, high_price,
              low_price, close_price, volume, volume_supported, artifact_id
            FROM polymarket.chainlink_btcusd_one_minute_candles
            WHERE symbol = $1 AND open_timestamp = ANY($2)
            "#,
        )
        .bind(&records[0].symbol)
        .bind(&timestamps)
        .fetch_all(&mut *tx)
        .await
        .context("failed to inspect existing Chainlink candles")?;
        for stored in &existing {
            let candidate = records
                .iter()
                .find(|record| record.open_timestamp == stored.open_timestamp)
                .context("stored Chainlink candle identity was absent from candidate batch")?;
            if !stored.same_as(candidate, artifact_id) {
                bail!(
                    "immutable Chainlink candle conflict for {}:{}",
                    stored.symbol,
                    stored.open_timestamp
                );
            }
        }

        let mut query = QueryBuilder::<Postgres>::new(
            "INSERT INTO polymarket.chainlink_btcusd_one_minute_candles (symbol, \
             open_timestamp, close_timestamp, open_price, high_price, low_price, close_price, \
             volume, volume_supported, artifact_id) ",
        );
        query.push_values(records, |mut row, record| {
            row.push_bind(&record.symbol)
                .push_bind(record.open_timestamp)
                .push_bind(record.close_timestamp)
                .push_bind(record.open_price)
                .push_bind(record.high_price)
                .push_bind(record.low_price)
                .push_bind(record.close_price)
                .push_bind(record.volume)
                .push_bind(record.volume_supported)
                .push_bind(artifact_id);
        });
        query.push(" ON CONFLICT (symbol, open_timestamp) DO NOTHING");
        let inserted = query
            .build()
            .execute(&mut *tx)
            .await
            .context("failed to persist Chainlink candle batch")?
            .rows_affected();
        tx.commit().await?;
        batch_write_result(records.len(), inserted, "Chainlink candle")
    }

    pub async fn insert_binance_open_interest_batch(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        records: &[BinanceBtcusdtOpenInterestRecord],
    ) -> Result<BatchWriteResult> {
        if records.is_empty() {
            return Ok(BatchWriteResult::default());
        }
        if records.len() > MAX_DATABASE_BATCH_ROWS {
            bail!("Binance open-interest batch exceeds {MAX_DATABASE_BATCH_ROWS} rows");
        }
        validate_binance_open_interest_batch(records)?;
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        require_writable_artifact(&mut tx, claim, artifact_id).await?;
        let timestamps = records
            .iter()
            .map(|record| record.source_timestamp)
            .collect::<Vec<_>>();
        let existing = sqlx::query_as::<_, ExistingBinanceOpenInterestRow>(
            r#"
            SELECT symbol, source_timestamp, period_seconds, sum_open_interest,
              sum_open_interest_value, cmc_circulating_supply, artifact_id
            FROM polymarket.binance_btcusdt_five_minute_open_interest
            WHERE symbol = $1 AND source_timestamp = ANY($2)
            "#,
        )
        .bind(&records[0].symbol)
        .bind(&timestamps)
        .fetch_all(&mut *tx)
        .await
        .context("failed to inspect existing Binance open-interest rows")?;
        for stored in &existing {
            let candidate = records
                .iter()
                .find(|record| record.source_timestamp == stored.source_timestamp)
                .context("stored Binance open-interest identity was absent from candidate batch")?;
            if !stored.same_as(candidate, artifact_id) {
                bail!(
                    "immutable Binance open-interest conflict for {}:{}",
                    stored.symbol,
                    stored.source_timestamp
                );
            }
        }

        let mut query = QueryBuilder::<Postgres>::new(
            "INSERT INTO polymarket.binance_btcusdt_five_minute_open_interest (symbol, \
             source_timestamp, period_seconds, sum_open_interest, sum_open_interest_value, \
             cmc_circulating_supply, artifact_id) ",
        );
        query.push_values(records, |mut row, record| {
            row.push_bind(&record.symbol)
                .push_bind(record.source_timestamp)
                .push_bind(record.period_seconds)
                .push_bind(record.sum_open_interest)
                .push_bind(record.sum_open_interest_value)
                .push_bind(record.cmc_circulating_supply)
                .push_bind(artifact_id);
        });
        query.push(" ON CONFLICT (symbol, source_timestamp) DO NOTHING");
        let inserted = query
            .build()
            .execute(&mut *tx)
            .await
            .context("failed to persist Binance open-interest batch")?
            .rows_affected();
        tx.commit().await?;
        batch_write_result(records.len(), inserted, "Binance open-interest")
    }

    pub async fn insert_polygon_chainlink_oracle_round_batch(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        records: &[PolygonChainlinkBtcusdOracleRound],
    ) -> Result<BatchWriteResult> {
        if records.is_empty() {
            return Ok(BatchWriteResult::default());
        }
        if records.len() > MAX_POLYGON_CHAINLINK_INSERT_ROWS {
            bail!(
                "Polygon Chainlink oracle batch exceeds {MAX_POLYGON_CHAINLINK_INSERT_ROWS} rows"
            );
        }
        validate_polygon_chainlink_oracle_batch(records)?;
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        require_writable_artifact(&mut tx, claim, artifact_id).await?;
        let transaction_hashes = records
            .iter()
            .map(|record| record.transaction_hash.clone())
            .collect::<Vec<_>>();
        let existing = sqlx::query_as::<_, ExistingPolygonChainlinkOracleRoundRow>(
            r#"
            SELECT chain_id, feed_proxy_address, aggregator_address, phase_id,
              aggregator_round_id, source_timestamp, block_timestamp, answer_raw, price,
              decimals, block_number, block_hash, transaction_hash, log_index, artifact_id
            FROM polymarket.polygon_chainlink_btcusd_oracle_rounds
            WHERE feed_proxy_address = $1 AND transaction_hash = ANY($2)
            "#,
        )
        .bind(&records[0].feed_proxy_address)
        .bind(&transaction_hashes)
        .fetch_all(&mut *tx)
        .await
        .context("failed to inspect existing Polygon Chainlink oracle rounds")?;
        for stored in &existing {
            let candidate = records
                .iter()
                .find(|record| {
                    record.transaction_hash == stored.transaction_hash
                        && record.log_index == stored.log_index
                })
                .context(
                    "stored Polygon Chainlink oracle identity was absent from candidate batch",
                )?;
            if !stored.same_as(candidate, artifact_id) {
                bail!(
                    "immutable Polygon Chainlink oracle conflict for {}:{}",
                    stored.transaction_hash,
                    stored.log_index
                );
            }
        }

        let mut query = QueryBuilder::<Postgres>::new(
            "INSERT INTO polymarket.polygon_chainlink_btcusd_oracle_rounds (chain_id, \
             feed_proxy_address, aggregator_address, phase_id, aggregator_round_id, \
             source_timestamp, block_timestamp, answer_raw, price, decimals, block_number, \
             block_hash, transaction_hash, log_index, artifact_id) ",
        );
        query.push_values(records, |mut row, record| {
            row.push_bind(record.chain_id)
                .push_bind(&record.feed_proxy_address)
                .push_bind(&record.aggregator_address)
                .push_bind(record.phase_id)
                .push_bind(record.aggregator_round_id)
                .push_bind(record.source_timestamp)
                .push_bind(record.block_timestamp)
                .push_bind(record.answer_raw)
                .push_bind(record.price)
                .push_bind(record.decimals)
                .push_bind(record.block_number)
                .push_bind(&record.block_hash)
                .push_bind(&record.transaction_hash)
                .push_bind(record.log_index)
                .push_bind(artifact_id);
        });
        query.push(
            " ON CONFLICT (feed_proxy_address, source_timestamp, transaction_hash, log_index) \
             DO NOTHING",
        );
        let inserted = query
            .build()
            .execute(&mut *tx)
            .await
            .context("failed to persist Polygon Chainlink oracle round batch")?
            .rows_affected();
        tx.commit().await?;
        batch_write_result(records.len(), inserted, "Polygon Chainlink oracle round")
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

    pub async fn stage_binance_l2_one_second_feature_batch(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        records: &[BinanceL2OneSecondFeature],
    ) -> Result<BatchWriteResult> {
        require_claim_ingester(
            claim,
            IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
            "Binance futures L2",
        )?;
        if records.is_empty() {
            return Ok(BatchWriteResult::default());
        }
        if records.len() > MAX_BINANCE_L2_FEATURE_INSERT_ROWS {
            bail!("Binance L2 feature batch exceeds {MAX_BINANCE_L2_FEATURE_INSERT_ROWS} rows");
        }
        debug_assert!(
            MAX_BINANCE_L2_FEATURE_INSERT_ROWS * BINANCE_L2_FEATURE_INSERT_COLUMNS
                <= POSTGRES_MAX_BIND_PARAMETERS
        );
        validate_binance_l2_one_second_feature_batch(
            records,
            BINANCE_FUTURES_L2_FEATURE_SCHEMA_VERSION,
        )?;

        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        lock_writable_artifact_for_staging(&mut tx, claim, artifact_id).await?;

        let timestamps = records
            .iter()
            .map(|record| record.second_start)
            .collect::<Vec<_>>();
        let candidates = records
            .iter()
            .map(|record| (record.second_start, record))
            .collect::<BTreeMap<_, _>>();
        let existing = sqlx::query_as::<_, ExistingBinanceL2OneSecondFeatureRow>(
            r#"
            SELECT symbol, second_start, source_event_timestamp, provider_received_at,
              available_at, source_update_id, feature_schema_version, quality_status,
              midpoint, microprice, spread_bps, bid_depth_5, ask_depth_5, imbalance_5,
              bid_depth_10, ask_depth_10, imbalance_10, bid_depth_20, ask_depth_20,
              imbalance_20, bid_depth_slope_20, ask_depth_slope_20,
              bid_depth_concentration_20, ask_depth_concentration_20,
              bid_quote_replenishment_1s, ask_quote_replenishment_1s,
              bid_quote_churn_1s, ask_quote_churn_1s, midpoint_change_bps_1s,
              spread_bps_delta_1s, depth_20_change_bps_1s, imbalance_20_delta_1s,
              midpoint_change_bps_5s, spread_bps_delta_5s, depth_20_change_bps_5s,
              imbalance_20_delta_5s, midpoint_change_bps_15s, spread_bps_delta_15s,
              depth_20_change_bps_15s, imbalance_20_delta_15s,
              midpoint_change_bps_30s, spread_bps_delta_30s, depth_20_change_bps_30s,
              imbalance_20_delta_30s, midpoint_change_bps_60s, spread_bps_delta_60s,
              depth_20_change_bps_60s, imbalance_20_delta_60s, artifact_id
            FROM polymarket.binance_btcusdt_l2_one_second_features_staging
            WHERE artifact_id = $1 AND symbol = 'BTCUSDT' AND second_start = ANY($2)
            "#,
        )
        .bind(artifact_id)
        .bind(&timestamps)
        .fetch_all(&mut *tx)
        .await
        .context("failed to inspect existing Binance L2 one-second features")?;
        for stored in &existing {
            let candidate = candidates.get(&stored.second_start).with_context(|| {
                format!(
                    "stored Binance L2 feature identity {} was absent from its candidate batch",
                    stored.second_start
                )
            })?;
            if !stored.same_as(candidate, artifact_id) {
                bail!(
                    "immutable Binance L2 one-second feature conflict for {}:{}",
                    stored.symbol,
                    stored.second_start
                );
            }
        }

        let mut query = QueryBuilder::<Postgres>::new(
            "INSERT INTO polymarket.binance_btcusdt_l2_one_second_features_staging (symbol, \
             second_start, source_event_timestamp, provider_received_at, available_at, \
             source_update_id, feature_schema_version, quality_status, midpoint, microprice, \
             spread_bps, bid_depth_5, ask_depth_5, imbalance_5, bid_depth_10, ask_depth_10, \
             imbalance_10, bid_depth_20, ask_depth_20, imbalance_20, bid_depth_slope_20, \
             ask_depth_slope_20, bid_depth_concentration_20, ask_depth_concentration_20, \
             bid_quote_replenishment_1s, ask_quote_replenishment_1s, bid_quote_churn_1s, \
             ask_quote_churn_1s, midpoint_change_bps_1s, spread_bps_delta_1s, \
             depth_20_change_bps_1s, imbalance_20_delta_1s, midpoint_change_bps_5s, \
             spread_bps_delta_5s, depth_20_change_bps_5s, imbalance_20_delta_5s, \
             midpoint_change_bps_15s, spread_bps_delta_15s, depth_20_change_bps_15s, \
             imbalance_20_delta_15s, midpoint_change_bps_30s, spread_bps_delta_30s, \
             depth_20_change_bps_30s, imbalance_20_delta_30s, midpoint_change_bps_60s, \
             spread_bps_delta_60s, depth_20_change_bps_60s, imbalance_20_delta_60s, \
             artifact_id) ",
        );
        query.push_values(records, |mut row, record| {
            row.push_bind(&record.symbol)
                .push_bind(record.second_start)
                .push_bind(record.source_event_timestamp)
                .push_bind(record.provider_received_at)
                .push_bind(record.available_at)
                .push_bind(record.source_update_id)
                .push_bind(&record.feature_schema_version)
                .push_bind(&record.quality_status)
                .push_bind(record.midpoint)
                .push_bind(record.microprice)
                .push_bind(record.spread_bps)
                .push_bind(record.bid_depth_5)
                .push_bind(record.ask_depth_5)
                .push_bind(record.imbalance_5)
                .push_bind(record.bid_depth_10)
                .push_bind(record.ask_depth_10)
                .push_bind(record.imbalance_10)
                .push_bind(record.bid_depth_20)
                .push_bind(record.ask_depth_20)
                .push_bind(record.imbalance_20)
                .push_bind(record.bid_depth_slope_20)
                .push_bind(record.ask_depth_slope_20)
                .push_bind(record.bid_depth_concentration_20)
                .push_bind(record.ask_depth_concentration_20)
                .push_bind(record.bid_quote_replenishment_1s)
                .push_bind(record.ask_quote_replenishment_1s)
                .push_bind(record.bid_quote_churn_1s)
                .push_bind(record.ask_quote_churn_1s)
                .push_bind(record.midpoint_change_bps_1s)
                .push_bind(record.spread_bps_delta_1s)
                .push_bind(record.depth_20_change_bps_1s)
                .push_bind(record.imbalance_20_delta_1s)
                .push_bind(record.midpoint_change_bps_5s)
                .push_bind(record.spread_bps_delta_5s)
                .push_bind(record.depth_20_change_bps_5s)
                .push_bind(record.imbalance_20_delta_5s)
                .push_bind(record.midpoint_change_bps_15s)
                .push_bind(record.spread_bps_delta_15s)
                .push_bind(record.depth_20_change_bps_15s)
                .push_bind(record.imbalance_20_delta_15s)
                .push_bind(record.midpoint_change_bps_30s)
                .push_bind(record.spread_bps_delta_30s)
                .push_bind(record.depth_20_change_bps_30s)
                .push_bind(record.imbalance_20_delta_30s)
                .push_bind(record.midpoint_change_bps_60s)
                .push_bind(record.spread_bps_delta_60s)
                .push_bind(record.depth_20_change_bps_60s)
                .push_bind(record.imbalance_20_delta_60s)
                .push_bind(artifact_id);
        });
        query.push(" ON CONFLICT (artifact_id, symbol, second_start) DO NOTHING");
        let inserted = query
            .build()
            .execute(&mut *tx)
            .await
            .context("failed to stage Binance L2 one-second feature batch")?
            .rows_affected();
        tx.commit().await?;
        batch_write_result(
            records.len(),
            inserted,
            "staged Binance L2 one-second feature",
        )
    }

    pub async fn stage_binance_spot_l2_one_second_feature_batch(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        records: &[BinanceL2OneSecondFeature],
    ) -> Result<BatchWriteResult> {
        require_claim_ingester(
            claim,
            IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures,
            "Binance spot L2",
        )?;
        if records.is_empty() {
            return Ok(BatchWriteResult::default());
        }
        if records.len() > MAX_BINANCE_L2_FEATURE_INSERT_ROWS {
            bail!(
                "Binance spot L2 feature batch exceeds {MAX_BINANCE_L2_FEATURE_INSERT_ROWS} rows"
            );
        }
        debug_assert!(
            MAX_BINANCE_L2_FEATURE_INSERT_ROWS * BINANCE_L2_FEATURE_INSERT_COLUMNS
                <= POSTGRES_MAX_BIND_PARAMETERS
        );
        validate_binance_l2_one_second_feature_batch(
            records,
            BINANCE_SPOT_L2_FEATURE_SCHEMA_VERSION,
        )?;

        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        lock_writable_artifact_for_staging(&mut tx, claim, artifact_id).await?;

        let timestamps = records
            .iter()
            .map(|record| record.second_start)
            .collect::<Vec<_>>();
        let candidates = records
            .iter()
            .map(|record| (record.second_start, record))
            .collect::<BTreeMap<_, _>>();
        let existing = sqlx::query_as::<_, ExistingBinanceL2OneSecondFeatureRow>(
            r#"
            SELECT symbol, second_start, source_event_timestamp, provider_received_at,
              available_at, source_update_id, feature_schema_version, quality_status,
              midpoint, microprice, spread_bps, bid_depth_5, ask_depth_5, imbalance_5,
              bid_depth_10, ask_depth_10, imbalance_10, bid_depth_20, ask_depth_20,
              imbalance_20, bid_depth_slope_20, ask_depth_slope_20,
              bid_depth_concentration_20, ask_depth_concentration_20,
              bid_quote_replenishment_1s, ask_quote_replenishment_1s,
              bid_quote_churn_1s, ask_quote_churn_1s, midpoint_change_bps_1s,
              spread_bps_delta_1s, depth_20_change_bps_1s, imbalance_20_delta_1s,
              midpoint_change_bps_5s, spread_bps_delta_5s, depth_20_change_bps_5s,
              imbalance_20_delta_5s, midpoint_change_bps_15s, spread_bps_delta_15s,
              depth_20_change_bps_15s, imbalance_20_delta_15s,
              midpoint_change_bps_30s, spread_bps_delta_30s, depth_20_change_bps_30s,
              imbalance_20_delta_30s, midpoint_change_bps_60s, spread_bps_delta_60s,
              depth_20_change_bps_60s, imbalance_20_delta_60s, artifact_id
            FROM polymarket.binance_spot_btcusdt_l2_one_second_features_staging
            WHERE artifact_id = $1 AND symbol = 'BTCUSDT' AND second_start = ANY($2)
            "#,
        )
        .bind(artifact_id)
        .bind(&timestamps)
        .fetch_all(&mut *tx)
        .await
        .context("failed to inspect existing Binance spot L2 one-second features")?;
        for stored in &existing {
            let candidate = candidates.get(&stored.second_start).with_context(|| {
                format!(
                    "stored Binance spot L2 feature identity {} was absent from its candidate batch",
                    stored.second_start
                )
            })?;
            if !stored.same_as(candidate, artifact_id) {
                bail!(
                    "immutable Binance spot L2 one-second feature conflict for {}:{}",
                    stored.symbol,
                    stored.second_start
                );
            }
        }

        let mut query = QueryBuilder::<Postgres>::new(
            "INSERT INTO polymarket.binance_spot_btcusdt_l2_one_second_features_staging (symbol, \
             second_start, source_event_timestamp, provider_received_at, available_at, \
             source_update_id, feature_schema_version, quality_status, midpoint, microprice, \
             spread_bps, bid_depth_5, ask_depth_5, imbalance_5, bid_depth_10, ask_depth_10, \
             imbalance_10, bid_depth_20, ask_depth_20, imbalance_20, bid_depth_slope_20, \
             ask_depth_slope_20, bid_depth_concentration_20, ask_depth_concentration_20, \
             bid_quote_replenishment_1s, ask_quote_replenishment_1s, bid_quote_churn_1s, \
             ask_quote_churn_1s, midpoint_change_bps_1s, spread_bps_delta_1s, \
             depth_20_change_bps_1s, imbalance_20_delta_1s, midpoint_change_bps_5s, \
             spread_bps_delta_5s, depth_20_change_bps_5s, imbalance_20_delta_5s, \
             midpoint_change_bps_15s, spread_bps_delta_15s, depth_20_change_bps_15s, \
             imbalance_20_delta_15s, midpoint_change_bps_30s, spread_bps_delta_30s, \
             depth_20_change_bps_30s, imbalance_20_delta_30s, midpoint_change_bps_60s, \
             spread_bps_delta_60s, depth_20_change_bps_60s, imbalance_20_delta_60s, \
             artifact_id) ",
        );
        query.push_values(records, |mut row, record| {
            row.push_bind(&record.symbol)
                .push_bind(record.second_start)
                .push_bind(record.source_event_timestamp)
                .push_bind(record.provider_received_at)
                .push_bind(record.available_at)
                .push_bind(record.source_update_id)
                .push_bind(&record.feature_schema_version)
                .push_bind(&record.quality_status)
                .push_bind(record.midpoint)
                .push_bind(record.microprice)
                .push_bind(record.spread_bps)
                .push_bind(record.bid_depth_5)
                .push_bind(record.ask_depth_5)
                .push_bind(record.imbalance_5)
                .push_bind(record.bid_depth_10)
                .push_bind(record.ask_depth_10)
                .push_bind(record.imbalance_10)
                .push_bind(record.bid_depth_20)
                .push_bind(record.ask_depth_20)
                .push_bind(record.imbalance_20)
                .push_bind(record.bid_depth_slope_20)
                .push_bind(record.ask_depth_slope_20)
                .push_bind(record.bid_depth_concentration_20)
                .push_bind(record.ask_depth_concentration_20)
                .push_bind(record.bid_quote_replenishment_1s)
                .push_bind(record.ask_quote_replenishment_1s)
                .push_bind(record.bid_quote_churn_1s)
                .push_bind(record.ask_quote_churn_1s)
                .push_bind(record.midpoint_change_bps_1s)
                .push_bind(record.spread_bps_delta_1s)
                .push_bind(record.depth_20_change_bps_1s)
                .push_bind(record.imbalance_20_delta_1s)
                .push_bind(record.midpoint_change_bps_5s)
                .push_bind(record.spread_bps_delta_5s)
                .push_bind(record.depth_20_change_bps_5s)
                .push_bind(record.imbalance_20_delta_5s)
                .push_bind(record.midpoint_change_bps_15s)
                .push_bind(record.spread_bps_delta_15s)
                .push_bind(record.depth_20_change_bps_15s)
                .push_bind(record.imbalance_20_delta_15s)
                .push_bind(record.midpoint_change_bps_30s)
                .push_bind(record.spread_bps_delta_30s)
                .push_bind(record.depth_20_change_bps_30s)
                .push_bind(record.imbalance_20_delta_30s)
                .push_bind(record.midpoint_change_bps_60s)
                .push_bind(record.spread_bps_delta_60s)
                .push_bind(record.depth_20_change_bps_60s)
                .push_bind(record.imbalance_20_delta_60s)
                .push_bind(artifact_id);
        });
        query.push(" ON CONFLICT (artifact_id, symbol, second_start) DO NOTHING");
        let inserted = query
            .build()
            .execute(&mut *tx)
            .await
            .context("failed to stage Binance spot L2 one-second feature batch")?
            .rows_affected();
        tx.commit().await?;
        batch_write_result(
            records.len(),
            inserted,
            "staged Binance spot L2 one-second feature",
        )
    }

    pub async fn stage_binance_spot_l2_gap_feature_batch(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        records: &[BinanceL2OneSecondFeature],
    ) -> Result<BatchWriteResult> {
        require_claim_ingester(
            claim,
            IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures,
            "Binance spot L2",
        )?;
        if records.is_empty() {
            return Ok(BatchWriteResult::default());
        }
        let timestamps = records
            .iter()
            .map(|record| record.second_start)
            .collect::<Vec<_>>();
        let existing = sqlx::query_scalar::<_, DateTime<Utc>>(
            r#"
            SELECT second_start
            FROM polymarket.binance_spot_btcusdt_l2_one_second_features
            WHERE symbol = 'BTCUSDT' AND second_start = ANY($1)
            "#,
        )
        .bind(&timestamps)
        .fetch_all(&self.pool)
        .await
        .context("failed to inspect qualified Binance spot L2 seconds")?
        .into_iter()
        .collect::<HashSet<_>>();
        let gaps = records
            .iter()
            .filter(|record| !existing.contains(&record.second_start))
            .cloned()
            .collect::<Vec<_>>();
        let mut result = self
            .stage_binance_spot_l2_one_second_feature_batch(claim, artifact_id, &gaps)
            .await?;
        result.input_records = u64::try_from(records.len())?;
        result.duplicate_records = result
            .duplicate_records
            .saturating_add(u64::try_from(records.len().saturating_sub(gaps.len()))?);
        Ok(result)
    }

    pub async fn reset_binance_l2_feature_staging(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
    ) -> Result<()> {
        require_claim_ingester(
            claim,
            IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
            "Binance futures L2",
        )?;
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        lock_writable_artifact_for_staging(&mut tx, claim, artifact_id).await?;
        sqlx::query(
            "DELETE FROM polymarket.binance_btcusdt_l2_one_second_features_staging WHERE artifact_id = $1",
        )
        .bind(artifact_id)
        .execute(&mut *tx)
        .await
        .context("failed to reset staged Binance L2 features")?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn fail_binance_l2_artifact(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        error: &str,
    ) -> Result<BackfillArtifact> {
        require_claim_ingester(
            claim,
            IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
            "Binance futures L2",
        )?;
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
        .context("failed to lock Binance L2 artifact for failure cleanup")?;
        if current.ingester_key != IngesterKey::BinanceBtcusdtL2OneSecondFeatures.as_str() {
            bail!("artifact is not a Binance L2 materialization");
        }
        if current.status == "completed" {
            bail!("completed Binance L2 artifact {artifact_id} is immutable");
        }
        sqlx::query(
            "DELETE FROM polymarket.binance_btcusdt_l2_one_second_features_staging WHERE artifact_id = $1",
        )
        .bind(artifact_id)
        .execute(&mut *tx)
        .await
        .context("failed to clear staged Binance L2 features")?;
        let row = sqlx::query_as::<_, BackfillArtifactRow>(
            r#"
            UPDATE polymarket.backfill_artifacts
            SET status = 'failed', metadata = metadata || $3, updated_at = now()
            WHERE artifact_id = $1 AND job_id = $2 AND status <> 'completed'
            RETURNING artifact_id, job_id, ingester_key, logical_key, provider, source_uri,
              source_date, checksum_algorithm, expected_checksum, actual_checksum,
              compressed_bytes, record_count, minimum_source_timestamp, maximum_source_timestamp,
              status, metadata, created_at, updated_at, completed_at
            "#,
        )
        .bind(artifact_id)
        .bind(claim.job.job_id)
        .bind(serde_json::json!({"error": error}))
        .fetch_one(&mut *tx)
        .await
        .context("failed to mark Binance L2 artifact failed")?;
        tx.commit()
            .await
            .context("failed to commit Binance L2 failure cleanup")?;
        row.try_into()
    }

    pub async fn reset_binance_spot_l2_feature_staging(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
    ) -> Result<()> {
        require_claim_ingester(
            claim,
            IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures,
            "Binance spot L2",
        )?;
        let mut tx = self.pool.begin().await?;
        require_active_lease(&mut tx, claim).await?;
        lock_writable_artifact_for_staging(&mut tx, claim, artifact_id).await?;
        sqlx::query(
            "DELETE FROM polymarket.binance_spot_btcusdt_l2_one_second_features_staging WHERE artifact_id = $1",
        )
        .bind(artifact_id)
        .execute(&mut *tx)
        .await
        .context("failed to reset staged Binance spot L2 features")?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn fail_binance_spot_l2_artifact(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        error: &str,
    ) -> Result<BackfillArtifact> {
        require_claim_ingester(
            claim,
            IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures,
            "Binance spot L2",
        )?;
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
        .context("failed to lock Binance spot L2 artifact for failure cleanup")?;
        if current.ingester_key != IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures.as_str() {
            bail!("artifact is not a Binance spot L2 materialization");
        }
        if current.status == "completed" {
            bail!("completed Binance spot L2 artifact {artifact_id} is immutable");
        }
        sqlx::query(
            "DELETE FROM polymarket.binance_spot_btcusdt_l2_one_second_features_staging WHERE artifact_id = $1",
        )
        .bind(artifact_id)
        .execute(&mut *tx)
        .await
        .context("failed to clear staged Binance spot L2 features")?;
        let row = sqlx::query_as::<_, BackfillArtifactRow>(
            r#"
            UPDATE polymarket.backfill_artifacts
            SET status = 'failed', metadata = metadata || $3, updated_at = now()
            WHERE artifact_id = $1 AND job_id = $2 AND status <> 'completed'
            RETURNING artifact_id, job_id, ingester_key, logical_key, provider, source_uri,
              source_date, checksum_algorithm, expected_checksum, actual_checksum,
              compressed_bytes, record_count, minimum_source_timestamp, maximum_source_timestamp,
              status, metadata, created_at, updated_at, completed_at
            "#,
        )
        .bind(artifact_id)
        .bind(claim.job.job_id)
        .bind(serde_json::json!({"error": error}))
        .fetch_one(&mut *tx)
        .await
        .context("failed to mark Binance spot L2 artifact failed")?;
        tx.commit()
            .await
            .context("failed to commit Binance spot L2 failure cleanup")?;
        row.try_into()
    }

    pub async fn publish_binance_l2_one_second_features(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        completion: &ArtifactCompletion,
    ) -> Result<BackfillArtifact> {
        require_claim_ingester(
            claim,
            IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
            "Binance futures L2",
        )?;
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
        .context("failed to lock Binance L2 artifact for publication")?;
        if current.ingester_key != IngesterKey::BinanceBtcusdtL2OneSecondFeatures.as_str() {
            bail!("artifact is not a Binance L2 materialization");
        }
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
        if current.status != "ingesting" {
            bail!("Binance L2 artifact must be ingesting before atomic publication");
        }
        let source_date = current
            .source_date
            .context("Binance L2 artifact must have a source date")?;
        let day_start = source_date
            .and_hms_opt(0, 0, 0)
            .context("Binance L2 source date could not be aligned to UTC midnight")?
            .and_utc();
        let day_end = day_start + chrono::Duration::days(1);
        let staged_outside_day = sqlx::query_scalar::<_, bool>(
            r#"
            SELECT EXISTS (
              SELECT 1
              FROM polymarket.binance_btcusdt_l2_one_second_features_staging
              WHERE artifact_id = $1
                AND (second_start < $2 OR second_start >= $3)
            )
            "#,
        )
        .bind(artifact_id)
        .bind(day_start)
        .bind(day_end)
        .fetch_one(&mut *tx)
        .await
        .context("failed to validate the staged Binance L2 UTC day")?;
        if staged_outside_day {
            bail!("staged Binance L2 features escaped their artifact UTC day");
        }
        let staged_count = sqlx::query_scalar::<_, i64>(
            "SELECT count(*)::bigint FROM polymarket.binance_btcusdt_l2_one_second_features_staging WHERE artifact_id = $1",
        )
        .bind(artifact_id)
        .fetch_one(&mut *tx)
        .await
        .context("failed to count staged Binance L2 features")?;
        if staged_count != record_count {
            bail!(
                "staged Binance L2 row count {staged_count} did not match artifact count {record_count}"
            );
        }

        let inserted = sqlx::query(
            r#"
            INSERT INTO polymarket.binance_btcusdt_l2_one_second_features (
              symbol, second_start, source_event_timestamp, provider_received_at, available_at,
              source_update_id, feature_schema_version, quality_status, midpoint, microprice,
              spread_bps, bid_depth_5, ask_depth_5, imbalance_5, bid_depth_10, ask_depth_10,
              imbalance_10, bid_depth_20, ask_depth_20, imbalance_20, bid_depth_slope_20,
              ask_depth_slope_20, bid_depth_concentration_20, ask_depth_concentration_20,
              bid_quote_replenishment_1s, ask_quote_replenishment_1s, bid_quote_churn_1s,
              ask_quote_churn_1s, midpoint_change_bps_1s, spread_bps_delta_1s,
              depth_20_change_bps_1s, imbalance_20_delta_1s, midpoint_change_bps_5s,
              spread_bps_delta_5s, depth_20_change_bps_5s, imbalance_20_delta_5s,
              midpoint_change_bps_15s, spread_bps_delta_15s, depth_20_change_bps_15s,
              imbalance_20_delta_15s, midpoint_change_bps_30s, spread_bps_delta_30s,
              depth_20_change_bps_30s, imbalance_20_delta_30s, midpoint_change_bps_60s,
              spread_bps_delta_60s, depth_20_change_bps_60s, imbalance_20_delta_60s,
              artifact_id
            )
            SELECT symbol, second_start, source_event_timestamp, provider_received_at, available_at,
              source_update_id, feature_schema_version, quality_status, midpoint, microprice,
              spread_bps, bid_depth_5, ask_depth_5, imbalance_5, bid_depth_10, ask_depth_10,
              imbalance_10, bid_depth_20, ask_depth_20, imbalance_20, bid_depth_slope_20,
              ask_depth_slope_20, bid_depth_concentration_20, ask_depth_concentration_20,
              bid_quote_replenishment_1s, ask_quote_replenishment_1s, bid_quote_churn_1s,
              ask_quote_churn_1s, midpoint_change_bps_1s, spread_bps_delta_1s,
              depth_20_change_bps_1s, imbalance_20_delta_1s, midpoint_change_bps_5s,
              spread_bps_delta_5s, depth_20_change_bps_5s, imbalance_20_delta_5s,
              midpoint_change_bps_15s, spread_bps_delta_15s, depth_20_change_bps_15s,
              imbalance_20_delta_15s, midpoint_change_bps_30s, spread_bps_delta_30s,
              depth_20_change_bps_30s, imbalance_20_delta_30s, midpoint_change_bps_60s,
              spread_bps_delta_60s, depth_20_change_bps_60s, imbalance_20_delta_60s,
              artifact_id
            FROM polymarket.binance_btcusdt_l2_one_second_features_staging
            WHERE artifact_id = $1
            ORDER BY symbol, second_start
            "#,
        )
        .bind(artifact_id)
        .execute(&mut *tx)
        .await
        .context("failed to atomically publish Binance L2 features")?
        .rows_affected();
        if inserted != u64::try_from(record_count).context("negative Binance L2 record count")? {
            bail!("published Binance L2 row count did not match the staged row count");
        }

        let row = sqlx::query_as::<_, BackfillArtifactRow>(
            r#"
            UPDATE polymarket.backfill_artifacts
            SET status = 'completed', actual_checksum = $3, compressed_bytes = $4,
                record_count = $5, minimum_source_timestamp = $6,
                maximum_source_timestamp = $7, metadata = metadata || $8,
                completed_at = now(), updated_at = now()
            WHERE artifact_id = $1 AND job_id = $2 AND status = 'ingesting'
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
        .context("failed to complete Binance L2 artifact during publication")?;
        sqlx::query(
            "DELETE FROM polymarket.binance_btcusdt_l2_one_second_features_staging WHERE artifact_id = $1",
        )
        .bind(artifact_id)
        .execute(&mut *tx)
        .await
        .context("failed to clear published Binance L2 staging rows")?;
        tx.commit()
            .await
            .context("failed to commit atomic Binance L2 publication")?;
        row.try_into()
    }

    pub async fn publish_binance_spot_l2_one_second_features(
        &self,
        claim: &ClaimedJob,
        artifact_id: Uuid,
        completion: &ArtifactCompletion,
    ) -> Result<BackfillArtifact> {
        require_claim_ingester(
            claim,
            IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures,
            "Binance spot L2",
        )?;
        validate_sha256(&completion.actual_checksum, "actual checksum")?;
        require_json_object(&completion.metadata, "artifact completion metadata")?;
        let contract = completion
            .metadata
            .get("materialization_contract")
            .and_then(Value::as_str);
        let expected_provider = match contract {
            Some("cryptohft-binance-spot-btcusdt-l2-features-v1") => "cryptohftdata",
            Some("huggingface-goooddy-binance-spot-btcusdt-l2-features-v1") => "huggingface",
            _ => {
                bail!("Binance spot L2 artifact completion contract was invalid");
            }
        };
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
        .context("failed to lock Binance spot L2 artifact for publication")?;
        if current.ingester_key != IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures.as_str() {
            bail!("artifact is not a Binance spot L2 materialization");
        }
        if current.provider != expected_provider {
            bail!("Binance spot L2 artifact provider did not match its materialization contract");
        }
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
        if current.status != "ingesting" {
            bail!("Binance spot L2 artifact must be ingesting before atomic publication");
        }
        let (day_start, day_end) = if expected_provider == "huggingface" {
            (
                claim
                    .job
                    .range_start
                    .context("Hugging Face job was missing range_start")?,
                claim
                    .job
                    .range_end
                    .context("Hugging Face job was missing range_end")?,
            )
        } else {
            let source_date = current
                .source_date
                .context("Binance spot L2 artifact must have a source date")?;
            let day_start = source_date
                .and_hms_opt(0, 0, 0)
                .context("Binance spot L2 source date could not be aligned to UTC midnight")?
                .and_utc();
            (day_start, day_start + chrono::Duration::days(1))
        };
        let staged_outside_day = sqlx::query_scalar::<_, bool>(
            r#"
            SELECT EXISTS (
              SELECT 1
              FROM polymarket.binance_spot_btcusdt_l2_one_second_features_staging
              WHERE artifact_id = $1
                AND (second_start < $2 OR second_start >= $3)
            )
            "#,
        )
        .bind(artifact_id)
        .bind(day_start)
        .bind(day_end)
        .fetch_one(&mut *tx)
        .await
        .context("failed to validate the staged Binance spot L2 UTC day")?;
        if staged_outside_day {
            bail!("staged Binance spot L2 features escaped their artifact UTC day");
        }
        let staged_count = sqlx::query_scalar::<_, i64>(
            "SELECT count(*)::bigint FROM polymarket.binance_spot_btcusdt_l2_one_second_features_staging WHERE artifact_id = $1",
        )
        .bind(artifact_id)
        .fetch_one(&mut *tx)
        .await
        .context("failed to count staged Binance spot L2 features")?;
        if staged_count != record_count {
            bail!(
                "staged Binance spot L2 row count {staged_count} did not match artifact count {record_count}"
            );
        }

        let inserted = sqlx::query(
            r#"
            INSERT INTO polymarket.binance_spot_btcusdt_l2_one_second_features (
              symbol, second_start, source_event_timestamp, provider_received_at, available_at,
              source_update_id, feature_schema_version, quality_status, midpoint, microprice,
              spread_bps, bid_depth_5, ask_depth_5, imbalance_5, bid_depth_10, ask_depth_10,
              imbalance_10, bid_depth_20, ask_depth_20, imbalance_20, bid_depth_slope_20,
              ask_depth_slope_20, bid_depth_concentration_20, ask_depth_concentration_20,
              bid_quote_replenishment_1s, ask_quote_replenishment_1s, bid_quote_churn_1s,
              ask_quote_churn_1s, midpoint_change_bps_1s, spread_bps_delta_1s,
              depth_20_change_bps_1s, imbalance_20_delta_1s, midpoint_change_bps_5s,
              spread_bps_delta_5s, depth_20_change_bps_5s, imbalance_20_delta_5s,
              midpoint_change_bps_15s, spread_bps_delta_15s, depth_20_change_bps_15s,
              imbalance_20_delta_15s, midpoint_change_bps_30s, spread_bps_delta_30s,
              depth_20_change_bps_30s, imbalance_20_delta_30s, midpoint_change_bps_60s,
              spread_bps_delta_60s, depth_20_change_bps_60s, imbalance_20_delta_60s,
              artifact_id
            )
            SELECT symbol, second_start, source_event_timestamp, provider_received_at, available_at,
              source_update_id, feature_schema_version, quality_status, midpoint, microprice,
              spread_bps, bid_depth_5, ask_depth_5, imbalance_5, bid_depth_10, ask_depth_10,
              imbalance_10, bid_depth_20, ask_depth_20, imbalance_20, bid_depth_slope_20,
              ask_depth_slope_20, bid_depth_concentration_20, ask_depth_concentration_20,
              bid_quote_replenishment_1s, ask_quote_replenishment_1s, bid_quote_churn_1s,
              ask_quote_churn_1s, midpoint_change_bps_1s, spread_bps_delta_1s,
              depth_20_change_bps_1s, imbalance_20_delta_1s, midpoint_change_bps_5s,
              spread_bps_delta_5s, depth_20_change_bps_5s, imbalance_20_delta_5s,
              midpoint_change_bps_15s, spread_bps_delta_15s, depth_20_change_bps_15s,
              imbalance_20_delta_15s, midpoint_change_bps_30s, spread_bps_delta_30s,
              depth_20_change_bps_30s, imbalance_20_delta_30s, midpoint_change_bps_60s,
              spread_bps_delta_60s, depth_20_change_bps_60s, imbalance_20_delta_60s,
              artifact_id
            FROM polymarket.binance_spot_btcusdt_l2_one_second_features_staging
            WHERE artifact_id = $1
            ORDER BY symbol, second_start
            "#,
        )
        .bind(artifact_id)
        .execute(&mut *tx)
        .await
        .context("failed to atomically publish Binance spot L2 features")?
        .rows_affected();
        if inserted
            != u64::try_from(record_count).context("negative Binance spot L2 record count")?
        {
            bail!("published Binance spot L2 row count did not match the staged row count");
        }

        let row = sqlx::query_as::<_, BackfillArtifactRow>(
            r#"
            UPDATE polymarket.backfill_artifacts
            SET status = 'completed', actual_checksum = $3, compressed_bytes = $4,
                record_count = $5, minimum_source_timestamp = $6,
                maximum_source_timestamp = $7, metadata = metadata || $8,
                completed_at = now(), updated_at = now()
            WHERE artifact_id = $1 AND job_id = $2 AND status = 'ingesting'
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
        .context("failed to complete Binance spot L2 artifact during publication")?;
        sqlx::query(
            "DELETE FROM polymarket.binance_spot_btcusdt_l2_one_second_features_staging WHERE artifact_id = $1",
        )
        .bind(artifact_id)
        .execute(&mut *tx)
        .await
        .context("failed to clear published Binance spot L2 staging rows")?;
        tx.commit()
            .await
            .context("failed to commit atomic Binance spot L2 publication")?;
        row.try_into()
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
              SELECT market_id, condition_id, up_token_id, down_token_id, window_start,
                window_end, validation_status, official_outcome
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
                ), false) AS kline_covered,
                EXISTS (
                  SELECT 1
                  FROM polymarket.chainlink_btcusd_archive_ticks c
                  JOIN polymarket.backfill_artifacts a USING (artifact_id)
                  WHERE c.source_timestamp >= m.window_start
                    AND c.source_timestamp <= m.window_start + interval '5 seconds'
                    AND a.status = 'completed'
                ) AND EXISTS (
                  SELECT 1
                  FROM polymarket.chainlink_btcusd_archive_ticks c
                  JOIN polymarket.backfill_artifacts a USING (artifact_id)
                  WHERE c.source_timestamp >= m.window_end - interval '5 seconds'
                    AND c.source_timestamp < m.window_end
                    AND a.status = 'completed'
                ) AS chainlink_covered,
                EXISTS (
                  SELECT 1
                  FROM polymarket.btc_market_decision_execution_snapshots s
                  JOIN polymarket.backfill_artifacts a USING (artifact_id)
                  WHERE s.market_id = m.market_id
                    AND s.sampled_at >= m.window_start
                    AND s.sampled_at < m.window_end
                    AND a.status = 'completed'
                  GROUP BY s.schema_version
                  HAVING (
                    s.schema_version = 'btc5m-book-250ms-v1'
                    AND count(*) = 1200
                    AND min(s.sampled_at) = m.window_start
                    AND max(s.sampled_at) = m.window_end - interval '250 milliseconds'
                  ) OR (
                    s.schema_version = 'btc5m-decision-book-90-140s-5s-v1'
                    AND count(*) = 11
                    AND min(s.sampled_at) = m.window_start + interval '90 seconds'
                    AND max(s.sampled_at) = m.window_start + interval '140 seconds'
                  )
                ) AS orderbook_covered
              FROM markets m
            )
            SELECT
              count(*) FILTER (WHERE m.validation_status = 'valid')::bigint AS valid_market_identities,
              count(*) FILTER (WHERE COALESCE(f.opening_boundary, false))::bigint AS opening_boundaries,
              count(*) FILTER (WHERE COALESCE(f.final_price, false))::bigint AS final_prices,
              count(*) FILTER (WHERE m.official_outcome IS NOT NULL)::bigint AS official_outcomes,
              count(*) FILTER (WHERE c.agg_covered)::bigint AS aggregate_trade_covered_markets,
              count(*) FILTER (WHERE c.kline_covered)::bigint AS one_second_kline_covered_markets,
              count(*) FILTER (WHERE c.chainlink_covered)::bigint AS chainlink_covered_markets,
              count(*) FILTER (WHERE c.orderbook_covered)::bigint AS orderbook_covered_markets,
              count(*) FILTER (
                WHERE m.validation_status = 'valid' AND m.official_outcome IS NOT NULL
                  AND COALESCE(f.opening_boundary, false)
                  AND COALESCE(f.final_price, false)
                  AND c.kline_covered AND c.orderbook_covered
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
                AS one_second_kline_max_timestamp,
              (SELECT min(source_timestamp) FROM polymarket.chainlink_btcusd_archive_ticks
                WHERE source_timestamp >= $1 AND source_timestamp < $2)
                AS chainlink_min_timestamp,
              (SELECT max(source_timestamp) FROM polymarket.chainlink_btcusd_archive_ticks
                WHERE source_timestamp >= $1 AND source_timestamp < $2)
                AS chainlink_max_timestamp,
              (SELECT min(sampled_at) FROM polymarket.btc_market_decision_execution_snapshots
                WHERE sampled_at >= $1 AND sampled_at < $2)
                AS orderbook_min_timestamp,
              (SELECT max(sampled_at) FROM polymarket.btc_market_decision_execution_snapshots
                WHERE sampled_at >= $1 AND sampled_at < $2)
                AS orderbook_max_timestamp
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
            "missing_final_price".to_string(),
            expected_markets.saturating_sub(row.final_prices),
        );
        missing_by_reason.insert(
            "missing_official_outcome".to_string(),
            expected_markets.saturating_sub(row.official_outcomes),
        );
        missing_by_reason.insert(
            "missing_one_second_klines".to_string(),
            expected_markets.saturating_sub(row.one_second_kline_covered_markets),
        );
        missing_by_reason.insert(
            "missing_compact_execution_snapshots".to_string(),
            expected_markets.saturating_sub(row.orderbook_covered_markets),
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
            chainlink_covered_markets: row.chainlink_covered_markets,
            orderbook_covered_markets: row.orderbook_covered_markets,
            usable_markets: row.usable_markets,
            aggregate_trade_min_timestamp: row.aggregate_trade_min_timestamp,
            aggregate_trade_max_timestamp: row.aggregate_trade_max_timestamp,
            one_second_kline_min_timestamp: row.one_second_kline_min_timestamp,
            one_second_kline_max_timestamp: row.one_second_kline_max_timestamp,
            chainlink_min_timestamp: row.chainlink_min_timestamp,
            chainlink_max_timestamp: row.chainlink_max_timestamp,
            orderbook_min_timestamp: row.orderbook_min_timestamp,
            orderbook_max_timestamp: row.orderbook_max_timestamp,
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
struct ExistingBinanceL2OneSecondFeatureRow {
    symbol: String,
    second_start: DateTime<Utc>,
    source_event_timestamp: DateTime<Utc>,
    provider_received_at: DateTime<Utc>,
    available_at: DateTime<Utc>,
    source_update_id: i64,
    feature_schema_version: String,
    quality_status: String,
    midpoint: Decimal,
    microprice: Decimal,
    spread_bps: Decimal,
    bid_depth_5: Decimal,
    ask_depth_5: Decimal,
    imbalance_5: Decimal,
    bid_depth_10: Decimal,
    ask_depth_10: Decimal,
    imbalance_10: Decimal,
    bid_depth_20: Decimal,
    ask_depth_20: Decimal,
    imbalance_20: Decimal,
    bid_depth_slope_20: Decimal,
    ask_depth_slope_20: Decimal,
    bid_depth_concentration_20: Decimal,
    ask_depth_concentration_20: Decimal,
    bid_quote_replenishment_1s: Decimal,
    ask_quote_replenishment_1s: Decimal,
    bid_quote_churn_1s: Decimal,
    ask_quote_churn_1s: Decimal,
    midpoint_change_bps_1s: Decimal,
    spread_bps_delta_1s: Decimal,
    depth_20_change_bps_1s: Decimal,
    imbalance_20_delta_1s: Decimal,
    midpoint_change_bps_5s: Decimal,
    spread_bps_delta_5s: Decimal,
    depth_20_change_bps_5s: Decimal,
    imbalance_20_delta_5s: Decimal,
    midpoint_change_bps_15s: Decimal,
    spread_bps_delta_15s: Decimal,
    depth_20_change_bps_15s: Decimal,
    imbalance_20_delta_15s: Decimal,
    midpoint_change_bps_30s: Decimal,
    spread_bps_delta_30s: Decimal,
    depth_20_change_bps_30s: Decimal,
    imbalance_20_delta_30s: Decimal,
    midpoint_change_bps_60s: Decimal,
    spread_bps_delta_60s: Decimal,
    depth_20_change_bps_60s: Decimal,
    imbalance_20_delta_60s: Decimal,
    artifact_id: Uuid,
}

impl ExistingBinanceL2OneSecondFeatureRow {
    fn same_as(&self, row: &BinanceL2OneSecondFeature, artifact_id: Uuid) -> bool {
        self.symbol == row.symbol
            && self.second_start == row.second_start
            && self.source_event_timestamp == row.source_event_timestamp
            && self.provider_received_at == row.provider_received_at
            && self.available_at == row.available_at
            && self.source_update_id == row.source_update_id
            && self.feature_schema_version == row.feature_schema_version
            && self.quality_status == row.quality_status
            && self.midpoint == row.midpoint
            && self.microprice == row.microprice
            && self.spread_bps == row.spread_bps
            && self.bid_depth_5 == row.bid_depth_5
            && self.ask_depth_5 == row.ask_depth_5
            && self.imbalance_5 == row.imbalance_5
            && self.bid_depth_10 == row.bid_depth_10
            && self.ask_depth_10 == row.ask_depth_10
            && self.imbalance_10 == row.imbalance_10
            && self.bid_depth_20 == row.bid_depth_20
            && self.ask_depth_20 == row.ask_depth_20
            && self.imbalance_20 == row.imbalance_20
            && self.bid_depth_slope_20 == row.bid_depth_slope_20
            && self.ask_depth_slope_20 == row.ask_depth_slope_20
            && self.bid_depth_concentration_20 == row.bid_depth_concentration_20
            && self.ask_depth_concentration_20 == row.ask_depth_concentration_20
            && self.bid_quote_replenishment_1s == row.bid_quote_replenishment_1s
            && self.ask_quote_replenishment_1s == row.ask_quote_replenishment_1s
            && self.bid_quote_churn_1s == row.bid_quote_churn_1s
            && self.ask_quote_churn_1s == row.ask_quote_churn_1s
            && self.midpoint_change_bps_1s == row.midpoint_change_bps_1s
            && self.spread_bps_delta_1s == row.spread_bps_delta_1s
            && self.depth_20_change_bps_1s == row.depth_20_change_bps_1s
            && self.imbalance_20_delta_1s == row.imbalance_20_delta_1s
            && self.midpoint_change_bps_5s == row.midpoint_change_bps_5s
            && self.spread_bps_delta_5s == row.spread_bps_delta_5s
            && self.depth_20_change_bps_5s == row.depth_20_change_bps_5s
            && self.imbalance_20_delta_5s == row.imbalance_20_delta_5s
            && self.midpoint_change_bps_15s == row.midpoint_change_bps_15s
            && self.spread_bps_delta_15s == row.spread_bps_delta_15s
            && self.depth_20_change_bps_15s == row.depth_20_change_bps_15s
            && self.imbalance_20_delta_15s == row.imbalance_20_delta_15s
            && self.midpoint_change_bps_30s == row.midpoint_change_bps_30s
            && self.spread_bps_delta_30s == row.spread_bps_delta_30s
            && self.depth_20_change_bps_30s == row.depth_20_change_bps_30s
            && self.imbalance_20_delta_30s == row.imbalance_20_delta_30s
            && self.midpoint_change_bps_60s == row.midpoint_change_bps_60s
            && self.spread_bps_delta_60s == row.spread_bps_delta_60s
            && self.depth_20_change_bps_60s == row.depth_20_change_bps_60s
            && self.imbalance_20_delta_60s == row.imbalance_20_delta_60s
            && self.artifact_id == artifact_id
    }
}

#[derive(Debug, FromRow)]
struct BtcOrderbookMarketScopeRow {
    market_id: String,
    condition_id: String,
    up_token_id: String,
    down_token_id: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
}

impl From<BtcOrderbookMarketScopeRow> for BtcOrderbookMarketScope {
    fn from(row: BtcOrderbookMarketScopeRow) -> Self {
        Self {
            market_id: row.market_id,
            condition_id: row.condition_id,
            up_token_id: row.up_token_id,
            down_token_id: row.down_token_id,
            window_start: row.window_start,
            window_end: row.window_end,
        }
    }
}

#[derive(Debug, FromRow)]
struct ExistingOrderbookEventRow {
    artifact_id: Uuid,
    source_row_number: i64,
    provider_received_at: DateTime<Utc>,
    source_timestamp: DateTime<Utc>,
    condition_id: String,
    asset_id: String,
    event_type: String,
    bids: Option<Value>,
    asks: Option<Value>,
    price: Option<Decimal>,
    size: Option<Decimal>,
    side: Option<String>,
    best_bid: Option<Decimal>,
    best_ask: Option<Decimal>,
    fee_rate_bps: Option<i32>,
    transaction_hash: Option<String>,
    old_tick_size: Option<Decimal>,
    new_tick_size: Option<Decimal>,
}

impl ExistingOrderbookEventRow {
    fn same_as(&self, row: &BtcOrderbookArchiveEvent, artifact_id: Uuid) -> bool {
        self.artifact_id == artifact_id
            && self.source_row_number == row.source_row_number
            && self.provider_received_at == row.provider_received_at
            && self.source_timestamp == row.source_timestamp
            && self.condition_id == row.condition_id
            && self.asset_id == row.asset_id
            && self.event_type == row.event_type
            && self.bids == row.bids
            && self.asks == row.asks
            && self.price == row.price
            && self.size == row.size
            && self.side == row.side
            && self.best_bid == row.best_bid
            && self.best_ask == row.best_ask
            && self.fee_rate_bps == row.fee_rate_bps
            && self.transaction_hash == row.transaction_hash
            && self.old_tick_size == row.old_tick_size
            && self.new_tick_size == row.new_tick_size
    }
}

impl From<ExistingOrderbookEventRow> for BtcOrderbookArchiveEvent {
    fn from(row: ExistingOrderbookEventRow) -> Self {
        Self {
            source_row_number: row.source_row_number,
            provider_received_at: row.provider_received_at,
            source_timestamp: row.source_timestamp,
            condition_id: row.condition_id,
            asset_id: row.asset_id,
            event_type: row.event_type,
            bids: row.bids,
            asks: row.asks,
            price: row.price,
            size: row.size,
            side: row.side,
            best_bid: row.best_bid,
            best_ask: row.best_ask,
            fee_rate_bps: row.fee_rate_bps,
            transaction_hash: row.transaction_hash,
            old_tick_size: row.old_tick_size,
            new_tick_size: row.new_tick_size,
        }
    }
}

#[derive(Debug, FromRow)]
struct ExistingChainlinkTickRow {
    feed_id: String,
    source_timestamp: DateTime<Utc>,
    valid_from_timestamp: DateTime<Utc>,
    price: Decimal,
    bid: Decimal,
    ask: Decimal,
    report_sha256: String,
    artifact_id: Uuid,
}

#[derive(Debug, FromRow)]
struct ExistingChainlinkCandleRow {
    symbol: String,
    open_timestamp: DateTime<Utc>,
    close_timestamp: DateTime<Utc>,
    open_price: Decimal,
    high_price: Decimal,
    low_price: Decimal,
    close_price: Decimal,
    volume: Option<Decimal>,
    volume_supported: bool,
    artifact_id: Uuid,
}

#[derive(Debug, FromRow)]
struct ExistingBinanceOpenInterestRow {
    symbol: String,
    source_timestamp: DateTime<Utc>,
    period_seconds: i32,
    sum_open_interest: Decimal,
    sum_open_interest_value: Decimal,
    cmc_circulating_supply: Option<Decimal>,
    artifact_id: Uuid,
}

#[derive(Debug, FromRow)]
struct ExistingPolygonChainlinkOracleRoundRow {
    chain_id: i64,
    feed_proxy_address: String,
    aggregator_address: String,
    phase_id: i32,
    aggregator_round_id: i64,
    source_timestamp: DateTime<Utc>,
    block_timestamp: DateTime<Utc>,
    answer_raw: Decimal,
    price: Decimal,
    decimals: i32,
    block_number: i64,
    block_hash: String,
    transaction_hash: String,
    log_index: i32,
    artifact_id: Uuid,
}

impl ExistingPolygonChainlinkOracleRoundRow {
    fn same_as(&self, row: &PolygonChainlinkBtcusdOracleRound, artifact_id: Uuid) -> bool {
        self.chain_id == row.chain_id
            && self.feed_proxy_address == row.feed_proxy_address
            && self.aggregator_address == row.aggregator_address
            && self.phase_id == row.phase_id
            && self.aggregator_round_id == row.aggregator_round_id
            && self.source_timestamp == row.source_timestamp
            && self.block_timestamp == row.block_timestamp
            && self.answer_raw == row.answer_raw
            && self.price == row.price
            && self.decimals == row.decimals
            && self.block_number == row.block_number
            && self.block_hash == row.block_hash
            && self.transaction_hash == row.transaction_hash
            && self.log_index == row.log_index
            && self.artifact_id == artifact_id
    }
}

impl ExistingChainlinkTickRow {
    fn same_as(&self, row: &ChainlinkBtcusdArchiveTick, artifact_id: Uuid) -> bool {
        self.feed_id == row.feed_id
            && self.source_timestamp == row.source_timestamp
            && self.valid_from_timestamp == row.valid_from_timestamp
            && self.price == row.price
            && self.bid == row.bid
            && self.ask == row.ask
            && self.report_sha256 == row.report_sha256
            && self.artifact_id == artifact_id
    }
}

impl ExistingChainlinkCandleRow {
    fn same_as(&self, row: &ChainlinkBtcusdOneMinuteCandle, artifact_id: Uuid) -> bool {
        self.symbol == row.symbol
            && self.open_timestamp == row.open_timestamp
            && self.close_timestamp == row.close_timestamp
            && self.open_price == row.open_price
            && self.high_price == row.high_price
            && self.low_price == row.low_price
            && self.close_price == row.close_price
            && self.volume == row.volume
            && self.volume_supported == row.volume_supported
            && self.artifact_id == artifact_id
    }
}

impl ExistingBinanceOpenInterestRow {
    fn same_as(&self, row: &BinanceBtcusdtOpenInterestRecord, artifact_id: Uuid) -> bool {
        self.symbol == row.symbol
            && self.source_timestamp == row.source_timestamp
            && self.period_seconds == row.period_seconds
            && self.sum_open_interest == row.sum_open_interest
            && self.sum_open_interest_value == row.sum_open_interest_value
            && self.cmc_circulating_supply == row.cmc_circulating_supply
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
    chainlink_covered_markets: i64,
    orderbook_covered_markets: i64,
    usable_markets: i64,
    aggregate_trade_min_timestamp: Option<DateTime<Utc>>,
    aggregate_trade_max_timestamp: Option<DateTime<Utc>>,
    one_second_kline_min_timestamp: Option<DateTime<Utc>>,
    one_second_kline_max_timestamp: Option<DateTime<Utc>>,
    chainlink_min_timestamp: Option<DateTime<Utc>>,
    chainlink_max_timestamp: Option<DateTime<Utc>>,
    orderbook_min_timestamp: Option<DateTime<Utc>>,
    orderbook_max_timestamp: Option<DateTime<Utc>>,
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

fn require_claim_ingester(
    claim: &ClaimedJob,
    expected: IngesterKey,
    materialization: &str,
) -> Result<()> {
    if claim.job.ingester_key != expected.as_str() {
        bail!(
            "{materialization} repository operation rejected {} claim",
            claim.job.ingester_key
        );
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

async fn lock_writable_artifact_for_staging(
    transaction: &mut Transaction<'_, Postgres>,
    claim: &ClaimedJob,
    artifact_id: Uuid,
) -> Result<()> {
    let artifact = sqlx::query_scalar::<_, Uuid>(
        r#"
        SELECT artifact_id
        FROM polymarket.backfill_artifacts
        WHERE artifact_id = $1 AND job_id = $2 AND status = 'ingesting'
        FOR SHARE
        "#,
    )
    .bind(artifact_id)
    .bind(claim.job.job_id)
    .fetch_optional(&mut **transaction)
    .await
    .context("failed to lock writable Binance L2 artifact")?;
    if artifact.is_none() {
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

fn validate_binance_l2_one_second_feature_batch(
    records: &[BinanceL2OneSecondFeature],
    expected_feature_schema_version: &str,
) -> Result<()> {
    let valid_imbalance = |value: Decimal| value >= -Decimal::ONE && value <= Decimal::ONE;
    let valid_imbalance_delta =
        |value: Decimal| value >= Decimal::from(-2_i32) && value <= Decimal::from(2_i32);
    let valid_concentration = |value: Decimal| value >= Decimal::ZERO && value <= Decimal::ONE;
    let mut previous = None;
    for record in records {
        let numeric_values = [
            record.midpoint,
            record.microprice,
            record.spread_bps,
            record.bid_depth_5,
            record.ask_depth_5,
            record.imbalance_5,
            record.bid_depth_10,
            record.ask_depth_10,
            record.imbalance_10,
            record.bid_depth_20,
            record.ask_depth_20,
            record.imbalance_20,
            record.bid_depth_slope_20,
            record.ask_depth_slope_20,
            record.bid_depth_concentration_20,
            record.ask_depth_concentration_20,
            record.bid_quote_replenishment_1s,
            record.ask_quote_replenishment_1s,
            record.bid_quote_churn_1s,
            record.ask_quote_churn_1s,
            record.midpoint_change_bps_1s,
            record.spread_bps_delta_1s,
            record.depth_20_change_bps_1s,
            record.imbalance_20_delta_1s,
            record.midpoint_change_bps_5s,
            record.spread_bps_delta_5s,
            record.depth_20_change_bps_5s,
            record.imbalance_20_delta_5s,
            record.midpoint_change_bps_15s,
            record.spread_bps_delta_15s,
            record.depth_20_change_bps_15s,
            record.imbalance_20_delta_15s,
            record.midpoint_change_bps_30s,
            record.spread_bps_delta_30s,
            record.depth_20_change_bps_30s,
            record.imbalance_20_delta_30s,
            record.midpoint_change_bps_60s,
            record.spread_bps_delta_60s,
            record.depth_20_change_bps_60s,
            record.imbalance_20_delta_60s,
        ];
        if record.symbol != "BTCUSDT"
            || record.second_start.timestamp_subsec_nanos() != 0
            || record.source_event_timestamp > record.available_at
            || record.provider_received_at > record.available_at
            || record.available_at < record.second_start
            || record.available_at >= record.second_start + chrono::Duration::seconds(1)
            || record.source_update_id < 0
            || record.feature_schema_version != expected_feature_schema_version
            || record.quality_status != "qualified"
            || record.midpoint <= Decimal::ZERO
            || record.microprice <= Decimal::ZERO
            || record.spread_bps < Decimal::ZERO
            || record.bid_depth_5 <= Decimal::ZERO
            || record.ask_depth_5 <= Decimal::ZERO
            || record.bid_depth_10 < record.bid_depth_5
            || record.ask_depth_10 < record.ask_depth_5
            || record.bid_depth_20 < record.bid_depth_10
            || record.ask_depth_20 < record.ask_depth_10
            || !valid_imbalance(record.imbalance_5)
            || !valid_imbalance(record.imbalance_10)
            || !valid_imbalance(record.imbalance_20)
            || !valid_imbalance_delta(record.imbalance_20_delta_1s)
            || !valid_imbalance_delta(record.imbalance_20_delta_5s)
            || !valid_imbalance_delta(record.imbalance_20_delta_15s)
            || !valid_imbalance_delta(record.imbalance_20_delta_30s)
            || !valid_imbalance_delta(record.imbalance_20_delta_60s)
            || record.bid_depth_slope_20 < Decimal::ZERO
            || record.ask_depth_slope_20 < Decimal::ZERO
            || !valid_concentration(record.bid_depth_concentration_20)
            || !valid_concentration(record.ask_depth_concentration_20)
            || record.bid_quote_replenishment_1s < Decimal::ZERO
            || record.ask_quote_replenishment_1s < Decimal::ZERO
            || record.bid_quote_churn_1s < Decimal::ZERO
            || record.ask_quote_churn_1s < Decimal::ZERO
            || numeric_values.iter().any(|value| value.scale() > 10)
        {
            bail!("invalid qualified Binance L2 one-second feature record");
        }
        if previous.is_some_and(|timestamp| record.second_start <= timestamp) {
            bail!("Binance L2 feature timestamps must be strictly increasing within a batch");
        }
        previous = Some(record.second_start);
    }
    Ok(())
}

fn validate_orderbook_event_batch(records: &[BtcOrderbookArchiveEvent]) -> Result<()> {
    let mut previous = None;
    for record in records {
        if record.source_row_number < 0
            || !record.condition_id.starts_with("0x")
            || record.condition_id.len() != 66
            || record.asset_id.is_empty()
            || !matches!(
                record.event_type.as_str(),
                "book" | "price_change" | "last_trade_price" | "tick_size_change"
            )
            || record
                .price
                .is_some_and(|value| value < Decimal::ZERO || value > Decimal::ONE)
            || record.size.is_some_and(|value| value < Decimal::ZERO)
            || record
                .best_bid
                .is_some_and(|value| value < Decimal::ZERO || value > Decimal::ONE)
            || record
                .best_ask
                .is_some_and(|value| value < Decimal::ZERO || value > Decimal::ONE)
            || record
                .side
                .as_deref()
                .is_some_and(|side| !matches!(side, "buy" | "sell"))
            || (record.event_type == "book"
                && (!record.bids.as_ref().is_some_and(Value::is_array)
                    || !record.asks.as_ref().is_some_and(Value::is_array)))
        {
            bail!("invalid PMXT orderbook-event record");
        }
        if previous.is_some_and(|value| record.source_row_number <= value) {
            bail!("PMXT source row numbers must be strictly increasing within a batch");
        }
        previous = Some(record.source_row_number);
    }
    Ok(())
}

fn validate_execution_snapshot_batch(records: &[BtcExecutionSnapshot]) -> Result<()> {
    for record in records {
        let valid_price = |value: Option<Decimal>| {
            value.is_none_or(|value| value >= Decimal::ZERO && value <= Decimal::ONE)
        };
        let valid_size = |value: Option<Decimal>| value.is_none_or(|value| value >= Decimal::ZERO);
        if record.market_id.trim().is_empty()
            || record.sampled_at.timestamp_subsec_millis().rem_euclid(250) != 0
            || record.quality_flags < 0
            || record
                .up_provider_received_at
                .is_some_and(|value| value > record.sampled_at)
            || record
                .down_provider_received_at
                .is_some_and(|value| value > record.sampled_at)
            || !valid_price(record.up_best_bid)
            || !valid_price(record.up_best_ask)
            || !valid_price(record.down_best_bid)
            || !valid_price(record.down_best_ask)
            || !valid_price(record.up_ask_vwap_1)
            || !valid_price(record.up_ask_vwap_5)
            || !valid_price(record.up_ask_vwap_10)
            || !valid_price(record.down_ask_vwap_1)
            || !valid_price(record.down_ask_vwap_5)
            || !valid_price(record.down_ask_vwap_10)
            || !valid_price(record.up_ask_vwap_15)
            || !valid_price(record.up_ask_vwap_20)
            || !valid_price(record.down_ask_vwap_15)
            || !valid_price(record.down_ask_vwap_20)
            || !valid_size(record.up_best_bid_size)
            || !valid_size(record.up_best_ask_size)
            || !valid_size(record.up_bid_depth)
            || !valid_size(record.up_ask_depth)
            || !valid_size(record.down_best_bid_size)
            || !valid_size(record.down_best_ask_size)
            || !valid_size(record.down_bid_depth)
            || !valid_size(record.down_ask_depth)
        {
            bail!("invalid compact execution-snapshot record");
        }
    }
    Ok(())
}

fn validate_chainlink_tick_batch(records: &[ChainlinkBtcusdArchiveTick]) -> Result<()> {
    let feed_id = &records[0].feed_id;
    let mut previous = None;
    for record in records {
        if &record.feed_id != feed_id
            || record.feed_id.len() != 66
            || !record.feed_id.starts_with("0x")
            || record.valid_from_timestamp > record.source_timestamp
            || record.price <= Decimal::ZERO
            || record.bid <= Decimal::ZERO
            || record.ask <= Decimal::ZERO
            || record.bid > record.price
            || record.price > record.ask
        {
            bail!("invalid Chainlink BTC/USD tick record");
        }
        validate_sha256(&record.report_sha256, "Chainlink report checksum")?;
        if previous.is_some_and(|value| record.source_timestamp <= value) {
            bail!("Chainlink tick timestamps must be strictly increasing within a batch");
        }
        previous = Some(record.source_timestamp);
    }
    Ok(())
}

fn validate_chainlink_candle_batch(records: &[ChainlinkBtcusdOneMinuteCandle]) -> Result<()> {
    let mut previous = None;
    for record in records {
        if record.symbol != "BTCUSD"
            || record.open_timestamp.timestamp().rem_euclid(60) != 0
            || record.close_timestamp != record.open_timestamp + chrono::Duration::minutes(1)
            || record.open_price <= Decimal::ZERO
            || record.high_price < record.open_price
            || record.high_price < record.close_price
            || record.high_price < record.low_price
            || record.low_price > record.open_price
            || record.low_price > record.close_price
            || (record.volume_supported
                && record.volume.is_none_or(|volume| volume < Decimal::ZERO))
            || (!record.volume_supported && record.volume.is_some())
        {
            bail!("invalid Chainlink BTC/USD one-minute candle record");
        }
        if previous.is_some_and(|timestamp| record.open_timestamp <= timestamp) {
            bail!("Chainlink candle timestamps must be strictly increasing within a batch");
        }
        previous = Some(record.open_timestamp);
    }
    Ok(())
}

fn validate_binance_open_interest_batch(
    records: &[BinanceBtcusdtOpenInterestRecord],
) -> Result<()> {
    let mut previous = None;
    for record in records {
        if record.symbol != "BTCUSDT"
            || record.period_seconds != 300
            || record.source_timestamp.timestamp().rem_euclid(300) != 0
            || record.sum_open_interest < Decimal::ZERO
            || record.sum_open_interest_value < Decimal::ZERO
            || record
                .cmc_circulating_supply
                .is_some_and(|value| value < Decimal::ZERO)
        {
            bail!("invalid Binance BTCUSDT five-minute open-interest record");
        }
        if previous.is_some_and(|timestamp| record.source_timestamp <= timestamp) {
            bail!("Binance open-interest timestamps must be strictly increasing within a batch");
        }
        previous = Some(record.source_timestamp);
    }
    Ok(())
}

fn validate_polygon_chainlink_oracle_batch(
    records: &[PolygonChainlinkBtcusdOracleRound],
) -> Result<()> {
    let feed_proxy_address = &records[0].feed_proxy_address;
    let mut previous = None;
    for record in records {
        if record.chain_id != 137
            || &record.feed_proxy_address != feed_proxy_address
            || !valid_evm_address(&record.feed_proxy_address)
            || !valid_evm_address(&record.aggregator_address)
            || record.phase_id <= 0
            || record.aggregator_round_id <= 0
            || record.source_timestamp > record.block_timestamp
            || record.answer_raw <= Decimal::ZERO
            || record.answer_raw.scale() != 0
            || record.price <= Decimal::ZERO
            || !(0..=18).contains(&record.decimals)
            || record.block_number <= 0
            || !valid_evm_hash(&record.block_hash)
            || !valid_evm_hash(&record.transaction_hash)
            || record.log_index < 0
        {
            bail!("invalid Polygon Chainlink BTC/USD oracle round");
        }
        let expected_price = Decimal::from_i128_with_scale(
            record.answer_raw.mantissa(),
            u32::try_from(record.decimals).context("invalid Polygon Chainlink decimals")?,
        );
        if record.price != expected_price {
            bail!("invalid Polygon Chainlink BTC/USD scaled price");
        }
        let identity = (
            record.source_timestamp,
            record.block_number,
            record.log_index,
        );
        if previous.is_some_and(|value| identity <= value) {
            bail!("Polygon Chainlink oracle rounds must be strictly ordered within a batch");
        }
        previous = Some(identity);
    }
    Ok(())
}

fn valid_evm_address(value: &str) -> bool {
    value.len() == 42
        && value.starts_with("0x")
        && value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_evm_hash(value: &str) -> bool {
    value.len() == 66
        && value.starts_with("0x")
        && value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn batch_write_result(records: usize, inserted: u64, name: &str) -> Result<BatchWriteResult> {
    let input = u64::try_from(records).with_context(|| format!("{name} batch size overflow"))?;
    Ok(BatchWriteResult {
        input_records: input,
        inserted_records: inserted,
        duplicate_records: input.saturating_sub(inserted),
    })
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

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use rust_decimal_macros::dec;

    use super::*;

    fn valid_l2_feature() -> BinanceL2OneSecondFeature {
        let second_start = Utc.with_ymd_and_hms(2026, 3, 21, 0, 0, 0).single().unwrap();
        BinanceL2OneSecondFeature {
            symbol: "BTCUSDT".to_string(),
            second_start,
            source_event_timestamp: second_start + chrono::Duration::milliseconds(100),
            provider_received_at: second_start + chrono::Duration::milliseconds(150),
            available_at: second_start + chrono::Duration::milliseconds(250),
            source_update_id: 1,
            feature_schema_version: "binance-btcusdt-l2-one-second-features-v1".to_string(),
            quality_status: "qualified".to_string(),
            midpoint: dec!(50000),
            microprice: dec!(50000.01),
            spread_bps: dec!(0.2),
            bid_depth_5: dec!(1),
            ask_depth_5: dec!(1),
            imbalance_5: Decimal::ZERO,
            bid_depth_10: dec!(2),
            ask_depth_10: dec!(2),
            imbalance_10: Decimal::ZERO,
            bid_depth_20: dec!(3),
            ask_depth_20: dec!(3),
            imbalance_20: Decimal::ZERO,
            bid_depth_slope_20: dec!(0.1),
            ask_depth_slope_20: dec!(0.1),
            bid_depth_concentration_20: dec!(0.5),
            ask_depth_concentration_20: dec!(0.5),
            bid_quote_replenishment_1s: Decimal::ZERO,
            ask_quote_replenishment_1s: Decimal::ZERO,
            bid_quote_churn_1s: Decimal::ZERO,
            ask_quote_churn_1s: Decimal::ZERO,
            midpoint_change_bps_1s: Decimal::ZERO,
            spread_bps_delta_1s: Decimal::ZERO,
            depth_20_change_bps_1s: Decimal::ZERO,
            imbalance_20_delta_1s: Decimal::ZERO,
            midpoint_change_bps_5s: Decimal::ZERO,
            spread_bps_delta_5s: Decimal::ZERO,
            depth_20_change_bps_5s: Decimal::ZERO,
            imbalance_20_delta_5s: Decimal::ZERO,
            midpoint_change_bps_15s: Decimal::ZERO,
            spread_bps_delta_15s: Decimal::ZERO,
            depth_20_change_bps_15s: Decimal::ZERO,
            imbalance_20_delta_15s: Decimal::ZERO,
            midpoint_change_bps_30s: Decimal::ZERO,
            spread_bps_delta_30s: Decimal::ZERO,
            depth_20_change_bps_30s: Decimal::ZERO,
            imbalance_20_delta_30s: Decimal::ZERO,
            midpoint_change_bps_60s: Decimal::ZERO,
            spread_bps_delta_60s: Decimal::ZERO,
            depth_20_change_bps_60s: Decimal::ZERO,
            imbalance_20_delta_60s: Decimal::ZERO,
        }
    }

    #[test]
    fn validates_qualified_causal_l2_features() {
        assert!(validate_binance_l2_one_second_feature_batch(
            &[valid_l2_feature()],
            BINANCE_FUTURES_L2_FEATURE_SCHEMA_VERSION,
        )
        .is_ok());
    }

    #[test]
    fn keeps_spot_and_futures_feature_schemas_isolated() {
        let futures = valid_l2_feature();
        assert!(validate_binance_l2_one_second_feature_batch(
            &[futures.clone()],
            BINANCE_SPOT_L2_FEATURE_SCHEMA_VERSION,
        )
        .is_err());

        let mut spot = futures;
        spot.feature_schema_version = BINANCE_SPOT_L2_FEATURE_SCHEMA_VERSION.to_string();
        assert!(validate_binance_l2_one_second_feature_batch(
            &[spot.clone()],
            BINANCE_SPOT_L2_FEATURE_SCHEMA_VERSION,
        )
        .is_ok());
        assert!(validate_binance_l2_one_second_feature_batch(
            &[spot],
            BINANCE_FUTURES_L2_FEATURE_SCHEMA_VERSION,
        )
        .is_err());
    }

    #[test]
    fn rejects_noncausal_or_malformed_l2_features() {
        let mut noncausal = valid_l2_feature();
        noncausal.available_at = noncausal.second_start + chrono::Duration::seconds(1);
        assert!(validate_binance_l2_one_second_feature_batch(
            &[noncausal],
            BINANCE_FUTURES_L2_FEATURE_SCHEMA_VERSION,
        )
        .is_err());

        let mut non_nested = valid_l2_feature();
        non_nested.bid_depth_10 = dec!(0.5);
        assert!(validate_binance_l2_one_second_feature_batch(
            &[non_nested],
            BINANCE_FUTURES_L2_FEATURE_SCHEMA_VERSION,
        )
        .is_err());

        let mut out_of_bounds = valid_l2_feature();
        out_of_bounds.imbalance_20_delta_60s = dec!(2.1);
        assert!(validate_binance_l2_one_second_feature_batch(
            &[out_of_bounds],
            BINANCE_FUTURES_L2_FEATURE_SCHEMA_VERSION,
        )
        .is_err());

        let mut excessive_scale = valid_l2_feature();
        excessive_scale.midpoint = dec!(50000.00000000001);
        assert!(validate_binance_l2_one_second_feature_batch(
            &[excessive_scale],
            BINANCE_FUTURES_L2_FEATURE_SCHEMA_VERSION,
        )
        .is_err());
    }

    #[test]
    fn rejects_duplicate_l2_feature_seconds_within_a_batch() {
        let record = valid_l2_feature();
        assert!(validate_binance_l2_one_second_feature_batch(
            &[record.clone(), record],
            BINANCE_FUTURES_L2_FEATURE_SCHEMA_VERSION,
        )
        .is_err());
    }
}
