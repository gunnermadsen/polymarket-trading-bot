use std::{cmp::max, time::Duration};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::{PgPool, Postgres, QueryBuilder, Row, Transaction};
use tokio::time::sleep;
use uuid::Uuid;

use super::job::{
    ClaimedKrakenJob, KrakenBackfillJob, KrakenDataset, NormalizedRows, PublishedLakeObject,
    KRAKEN_PROVIDER,
};

const INSERT_BATCH_ROWS: usize = 1_000;

#[derive(Debug, Clone)]
pub struct KrakenRepository {
    pool: PgPool,
}

impl KrakenRepository {
    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn enqueue(
        &self,
        dataset: KrakenDataset,
        symbol: &str,
        interval_seconds: i32,
        range_start: DateTime<Utc>,
        range_end: DateTime<Utc>,
        expected_work_units: i64,
    ) -> Result<bool> {
        let inserted = sqlx::query(
            r#"
            INSERT INTO kraken.backfill_jobs (
              dataset, symbol, interval_seconds, range_start, range_end, expected_work_units
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (dataset, symbol, interval_seconds, range_start, range_end)
            DO NOTHING
            "#,
        )
        .bind(dataset.as_str())
        .bind(symbol)
        .bind(interval_seconds)
        .bind(range_start)
        .bind(range_end)
        .bind(expected_work_units)
        .execute(&self.pool)
        .await
        .context("failed to enqueue Kraken backfill job")?;
        Ok(inserted.rows_affected() == 1)
    }

    pub async fn claim_next(
        &self,
        worker_id: &str,
        lease_duration: Duration,
    ) -> Result<Option<ClaimedKrakenJob>> {
        let lease_token = Uuid::new_v4();
        let lease_seconds =
            i64::try_from(lease_duration.as_secs()).context("Kraken lease exceeded i64")?;
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin Kraken claim")?;
        let job = sqlx::query_as::<_, KrakenBackfillJob>(
            r#"
            WITH candidate AS (
              SELECT job_id
              FROM kraken.backfill_jobs
              WHERE (
                  status = 'queued'
                  AND next_attempt_at <= now()
                  AND attempt < max_attempts
                )
                OR (
                  status = 'running'
                  AND lease_expires_at < now()
                  AND attempt < max_attempts
                )
              ORDER BY range_start, dataset, job_id
              FOR UPDATE SKIP LOCKED
              LIMIT 1
            )
            UPDATE kraken.backfill_jobs AS jobs
            SET status = 'running',
                attempt = jobs.attempt + 1,
                worker_id = $1,
                lease_token = $2,
                lease_expires_at = now() + ($3 * interval '1 second'),
                processed_work_units = 0,
                error_message = NULL,
                started_at = COALESCE(jobs.started_at, now()),
                updated_at = now()
            FROM candidate
            WHERE jobs.job_id = candidate.job_id
            RETURNING jobs.job_id, jobs.dataset, jobs.symbol, jobs.interval_seconds,
              jobs.range_start, jobs.range_end, jobs.attempt, jobs.max_attempts,
              jobs.expected_work_units, jobs.processed_work_units
            "#,
        )
        .bind(worker_id)
        .bind(lease_token)
        .bind(lease_seconds)
        .fetch_optional(&mut *tx)
        .await
        .context("failed to claim Kraken backfill job")?;

        if let Some(job) = &job {
            upsert_worker(&mut tx, worker_id, "running", Some(job.job_id)).await?;
            insert_event(
                &mut tx,
                job.job_id,
                "info",
                "Kraken backfill job claimed",
                serde_json::json!({"worker_id": worker_id, "attempt": job.attempt}),
            )
            .await?;
        } else {
            upsert_worker(&mut tx, worker_id, "idle", None).await?;
        }
        tx.commit().await.context("failed to commit Kraken claim")?;
        Ok(job.map(|job| ClaimedKrakenJob { job, lease_token }))
    }

    pub async fn heartbeat(
        &self,
        worker_id: &str,
        claim: &ClaimedKrakenJob,
        lease_duration: Duration,
    ) -> Result<()> {
        let lease_seconds =
            i64::try_from(lease_duration.as_secs()).context("Kraken lease exceeded i64")?;
        let updated = sqlx::query(
            r#"
            UPDATE kraken.backfill_jobs
            SET lease_expires_at = now() + ($3 * interval '1 second'), updated_at = now()
            WHERE job_id = $1 AND lease_token = $2 AND status = 'running'
            "#,
        )
        .bind(claim.job.job_id)
        .bind(claim.lease_token)
        .bind(lease_seconds)
        .execute(&self.pool)
        .await
        .context("failed to heartbeat Kraken job")?;
        if updated.rows_affected() != 1 {
            bail!("Kraken job lease was lost");
        }
        sqlx::query("UPDATE kraken.worker_status SET heartbeat_at = now() WHERE worker_id = $1")
            .bind(worker_id)
            .execute(&self.pool)
            .await
            .context("failed to heartbeat Kraken worker")?;
        Ok(())
    }

    pub async fn update_progress(
        &self,
        claim: &ClaimedKrakenJob,
        processed_work_units: i64,
    ) -> Result<()> {
        let updated = sqlx::query(
            r#"
            UPDATE kraken.backfill_jobs
            SET processed_work_units = LEAST(expected_work_units, $3), updated_at = now()
            WHERE job_id = $1 AND lease_token = $2 AND status = 'running'
            "#,
        )
        .bind(claim.job.job_id)
        .bind(claim.lease_token)
        .bind(max(0, processed_work_units))
        .execute(&self.pool)
        .await
        .context("failed to update Kraken job progress")?;
        if updated.rows_affected() != 1 {
            bail!("Kraken job lease was lost while updating progress");
        }
        Ok(())
    }

    pub async fn acquire_request_slot(&self) -> Result<()> {
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin Kraken API budget transaction")?;
        let row = sqlx::query(
            r#"
            SELECT GREATEST(next_request_at, clock_timestamp()) AS scheduled_at,
              spacing_milliseconds
            FROM kraken.api_request_budget
            WHERE budget_key = 'futures_public_api'
            FOR UPDATE
            "#,
        )
        .fetch_one(&mut *tx)
        .await
        .context("failed to lock Kraken API request budget")?;
        let scheduled_at: DateTime<Utc> = row.try_get("scheduled_at")?;
        let spacing_milliseconds: i32 = row.try_get("spacing_milliseconds")?;
        sqlx::query(
            r#"
            UPDATE kraken.api_request_budget
            SET next_request_at = $1 + ($2 * interval '1 millisecond'), updated_at = now()
            WHERE budget_key = 'futures_public_api'
            "#,
        )
        .bind(scheduled_at)
        .bind(spacing_milliseconds)
        .execute(&mut *tx)
        .await
        .context("failed to reserve Kraken API request slot")?;
        tx.commit()
            .await
            .context("failed to commit Kraken API request slot")?;
        if let Ok(wait) = (scheduled_at - Utc::now()).to_std() {
            if !wait.is_zero() {
                sleep(wait).await;
            }
        }
        Ok(())
    }

    pub async fn persist(
        &self,
        claim: &ClaimedKrakenJob,
        source_url: &str,
        rows: &NormalizedRows,
        object: &PublishedLakeObject,
    ) -> Result<i64> {
        let dataset = claim.job.dataset()?;
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin Kraken publish")?;
        require_lease(&mut tx, claim).await?;
        let artifact_id = Uuid::new_v4();
        let bounds = rows.time_bounds();
        let inserted = sqlx::query(
            r#"
            INSERT INTO kraken.backfill_artifacts (
              artifact_id, job_id, provider, source_url, sha256, lake_relative_path,
              row_count, source_min_time, source_max_time, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (job_id) DO NOTHING
            "#,
        )
        .bind(artifact_id)
        .bind(claim.job.job_id)
        .bind(KRAKEN_PROVIDER)
        .bind(source_url)
        .bind(&object.sha256)
        .bind(&object.relative_path)
        .bind(object.row_count)
        .bind(bounds.map(|value| value.0))
        .bind(bounds.map(|value| value.1))
        .bind(serde_json::json!({"schema_version": 1}))
        .execute(&mut *tx)
        .await
        .context("failed to insert Kraken artifact")?;

        if inserted.rows_affected() == 0 {
            let existing: i64 = sqlx::query_scalar(
                "SELECT row_count FROM kraken.backfill_artifacts WHERE job_id = $1",
            )
            .bind(claim.job.job_id)
            .fetch_one(&mut *tx)
            .await
            .context("failed to load existing Kraken artifact")?;
            tx.commit()
                .await
                .context("failed to finish idempotent Kraken publish")?;
            return Ok(existing);
        }

        sqlx::query(
            r#"
            INSERT INTO kraken.parquet_objects (
              artifact_id, dataset, symbol, interval_seconds, lake_relative_path,
              sha256, byte_size, row_count
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#,
        )
        .bind(artifact_id)
        .bind(dataset.as_str())
        .bind(&claim.job.symbol)
        .bind(claim.job.interval_seconds)
        .bind(&object.relative_path)
        .bind(&object.sha256)
        .bind(object.byte_size)
        .bind(object.row_count)
        .execute(&mut *tx)
        .await
        .context("failed to insert Kraken Parquet object")?;

        let inserted_rows = match rows {
            NormalizedRows::Instruments(rows) => {
                for row in rows {
                    sqlx::query(
                        r#"
                        INSERT INTO kraken.instruments (
                          symbol, instrument_type, tradeable, tick_size, contract_size,
                          base_currency, quote_currency, pair, contract_value_trade_precision,
                          max_position_size, funding_rate_coefficient, max_relative_funding_rate,
                          fee_schedule_uid, margin_levels, retail_margin_levels, margin_schedules,
                          raw_payload, artifact_id, source_observed_at
                        )
                        VALUES (
                          $1, $2, $3, $4, $5, $6, $7, $8, $9,
                          $10, $11, $12, $13, $14, $15,
                          $16, $17, $18, $19
                        )
                        ON CONFLICT (symbol) DO UPDATE SET
                          instrument_type = EXCLUDED.instrument_type,
                          tradeable = EXCLUDED.tradeable,
                          tick_size = EXCLUDED.tick_size,
                          contract_size = EXCLUDED.contract_size,
                          base_currency = EXCLUDED.base_currency,
                          quote_currency = EXCLUDED.quote_currency,
                          pair = EXCLUDED.pair,
                          contract_value_trade_precision = EXCLUDED.contract_value_trade_precision,
                          max_position_size = EXCLUDED.max_position_size,
                          funding_rate_coefficient = EXCLUDED.funding_rate_coefficient,
                          max_relative_funding_rate = EXCLUDED.max_relative_funding_rate,
                          fee_schedule_uid = EXCLUDED.fee_schedule_uid,
                          margin_levels = EXCLUDED.margin_levels,
                          retail_margin_levels = EXCLUDED.retail_margin_levels,
                          margin_schedules = EXCLUDED.margin_schedules,
                          raw_payload = EXCLUDED.raw_payload,
                          artifact_id = EXCLUDED.artifact_id,
                          source_observed_at = EXCLUDED.source_observed_at,
                          ingested_at = now()
                        "#,
                    )
                    .bind(&row.symbol)
                    .bind(&row.instrument_type)
                    .bind(row.tradeable)
                    .bind(&row.tick_size)
                    .bind(&row.contract_size)
                    .bind(&row.base_currency)
                    .bind(&row.quote_currency)
                    .bind(&row.pair)
                    .bind(row.contract_value_trade_precision)
                    .bind(&row.max_position_size)
                    .bind(&row.funding_rate_coefficient)
                    .bind(&row.max_relative_funding_rate)
                    .bind(&row.fee_schedule_uid)
                    .bind(&row.margin_levels)
                    .bind(&row.retail_margin_levels)
                    .bind(&row.margin_schedules)
                    .bind(&row.raw_payload)
                    .bind(artifact_id)
                    .bind(row.source_observed_at)
                    .execute(&mut *tx)
                    .await
                    .context("failed to persist Kraken instrument")?;
                }
                i64::try_from(rows.len()).context("Kraken instrument count exceeded i64")?
            }
            NormalizedRows::FeeSchedules(rows) => {
                for row in rows {
                    sqlx::query(
                        r#"
                        INSERT INTO kraken.fee_schedules (
                          fee_schedule_uid, name, tiers, raw_payload,
                          artifact_id, source_observed_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (fee_schedule_uid) DO UPDATE SET
                          name = EXCLUDED.name,
                          tiers = EXCLUDED.tiers,
                          raw_payload = EXCLUDED.raw_payload,
                          artifact_id = EXCLUDED.artifact_id,
                          source_observed_at = EXCLUDED.source_observed_at,
                          ingested_at = now()
                        "#,
                    )
                    .bind(&row.fee_schedule_uid)
                    .bind(&row.name)
                    .bind(&row.tiers)
                    .bind(&row.raw_payload)
                    .bind(artifact_id)
                    .bind(row.source_observed_at)
                    .execute(&mut *tx)
                    .await
                    .context("failed to persist Kraken fee schedule")?;
                }
                i64::try_from(rows.len()).context("Kraken fee schedule count exceeded i64")?
            }
            NormalizedRows::Candles(rows) => {
                insert_candles(&mut tx, claim, artifact_id, dataset, rows).await?
            }
            NormalizedRows::Analytics(rows) => {
                insert_analytics(&mut tx, claim, artifact_id, dataset, rows).await?
            }
            NormalizedRows::FundingRates(rows) => {
                insert_funding(&mut tx, claim, artifact_id, rows).await?
            }
        };
        update_coverage(
            &mut tx,
            dataset,
            &claim.job.symbol,
            claim.job.interval_seconds,
            bounds,
            inserted_rows,
        )
        .await?;
        insert_event(
            &mut tx,
            claim.job.job_id,
            "info",
            "Kraken artifact published",
            serde_json::json!({
                "rows": inserted_rows,
                "lake_relative_path": object.relative_path,
                "sha256": object.sha256,
            }),
        )
        .await?;
        tx.commit()
            .await
            .context("failed to commit Kraken artifact")?;
        Ok(inserted_rows)
    }

    pub async fn complete(
        &self,
        worker_id: &str,
        claim: &ClaimedKrakenJob,
        rows_written: i64,
    ) -> Result<()> {
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin Kraken completion")?;
        let updated = sqlx::query(
            r#"
            UPDATE kraken.backfill_jobs
            SET status = 'completed',
                processed_work_units = expected_work_units,
                rows_written = $3,
                worker_id = NULL,
                lease_token = NULL,
                lease_expires_at = NULL,
                completed_at = now(),
                updated_at = now()
            WHERE job_id = $1 AND lease_token = $2 AND status = 'running'
            "#,
        )
        .bind(claim.job.job_id)
        .bind(claim.lease_token)
        .bind(rows_written)
        .execute(&mut *tx)
        .await
        .context("failed to complete Kraken job")?;
        if updated.rows_affected() != 1 {
            bail!("Kraken job lease was lost before completion");
        }
        sqlx::query(
            r#"
            UPDATE kraken.worker_status
            SET state = 'idle', current_job_id = NULL,
                jobs_completed = jobs_completed + 1,
                rows_written = rows_written + $2,
                heartbeat_at = now()
            WHERE worker_id = $1
            "#,
        )
        .bind(worker_id)
        .bind(rows_written)
        .execute(&mut *tx)
        .await
        .context("failed to update completed Kraken worker")?;
        tx.commit()
            .await
            .context("failed to commit Kraken completion")?;
        Ok(())
    }

    pub async fn fail(&self, worker_id: &str, claim: &ClaimedKrakenJob, error: &str) -> Result<()> {
        let terminal = claim.job.attempt >= claim.job.max_attempts;
        let retry_seconds = 2_i64.pow(u32::try_from(claim.job.attempt.min(8)).unwrap_or(8));
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin Kraken failure")?;
        let updated = sqlx::query(
            r#"
            UPDATE kraken.backfill_jobs
            SET status = CASE WHEN $3 THEN 'failed' ELSE 'queued' END,
                next_attempt_at = CASE
                  WHEN $3 THEN next_attempt_at
                  ELSE now() + ($4 * interval '1 second')
                END,
                worker_id = NULL,
                lease_token = NULL,
                lease_expires_at = NULL,
                error_message = left($5, 4000),
                completed_at = CASE WHEN $3 THEN now() ELSE NULL END,
                updated_at = now()
            WHERE job_id = $1 AND lease_token = $2 AND status = 'running'
            "#,
        )
        .bind(claim.job.job_id)
        .bind(claim.lease_token)
        .bind(terminal)
        .bind(retry_seconds)
        .bind(error)
        .execute(&mut *tx)
        .await
        .context("failed to record Kraken job failure")?;
        if updated.rows_affected() != 1 {
            tx.rollback().await.ok();
            return Ok(());
        }
        sqlx::query(
            r#"
            UPDATE kraken.worker_status
            SET state = 'idle', current_job_id = NULL,
                jobs_failed = jobs_failed + CASE WHEN $2 THEN 1 ELSE 0 END,
                heartbeat_at = now()
            WHERE worker_id = $1
            "#,
        )
        .bind(worker_id)
        .bind(terminal)
        .execute(&mut *tx)
        .await
        .context("failed to update failed Kraken worker")?;
        insert_event(
            &mut tx,
            claim.job.job_id,
            "error",
            if terminal {
                "Kraken backfill job failed"
            } else {
                "Kraken backfill job scheduled for retry"
            },
            serde_json::json!({"error": error, "attempt": claim.job.attempt}),
        )
        .await?;
        tx.commit()
            .await
            .context("failed to commit Kraken failure")?;
        Ok(())
    }

    pub async fn set_worker_stopped(&self, worker_id: &str) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO kraken.worker_status (worker_id, state, heartbeat_at)
            VALUES ($1, 'stopped', now())
            ON CONFLICT (worker_id) DO UPDATE
            SET state = 'stopped', current_job_id = NULL, heartbeat_at = now()
            "#,
        )
        .bind(worker_id)
        .execute(&self.pool)
        .await
        .context("failed to stop Kraken worker")?;
        Ok(())
    }
}

async fn require_lease(tx: &mut Transaction<'_, Postgres>, claim: &ClaimedKrakenJob) -> Result<()> {
    let owns_lease: bool = sqlx::query_scalar(
        r#"
        SELECT EXISTS (
          SELECT 1 FROM kraken.backfill_jobs
          WHERE job_id = $1 AND lease_token = $2
            AND status = 'running' AND lease_expires_at > now()
        )
        "#,
    )
    .bind(claim.job.job_id)
    .bind(claim.lease_token)
    .fetch_one(&mut **tx)
    .await
    .context("failed to verify Kraken job lease")?;
    if !owns_lease {
        bail!("Kraken job lease was lost before publication");
    }
    Ok(())
}

async fn upsert_worker(
    tx: &mut Transaction<'_, Postgres>,
    worker_id: &str,
    state: &str,
    current_job_id: Option<Uuid>,
) -> Result<()> {
    sqlx::query(
        r#"
        INSERT INTO kraken.worker_status (
          worker_id, state, current_job_id, heartbeat_at
        )
        VALUES ($1, $2, $3, now())
        ON CONFLICT (worker_id) DO UPDATE
        SET state = EXCLUDED.state,
            current_job_id = EXCLUDED.current_job_id,
            heartbeat_at = now()
        "#,
    )
    .bind(worker_id)
    .bind(state)
    .bind(current_job_id)
    .execute(&mut **tx)
    .await
    .context("failed to update Kraken worker status")?;
    Ok(())
}

async fn insert_event(
    tx: &mut Transaction<'_, Postgres>,
    job_id: Uuid,
    level: &str,
    message: &str,
    metadata: Value,
) -> Result<()> {
    sqlx::query(
        r#"
        INSERT INTO kraken.backfill_job_events (
          job_id, level, message, metadata
        )
        VALUES ($1, $2, $3, $4)
        "#,
    )
    .bind(job_id)
    .bind(level)
    .bind(message)
    .bind(metadata)
    .execute(&mut **tx)
    .await
    .context("failed to insert Kraken job event")?;
    Ok(())
}

async fn insert_candles(
    tx: &mut Transaction<'_, Postgres>,
    claim: &ClaimedKrakenJob,
    artifact_id: Uuid,
    dataset: KrakenDataset,
    rows: &[super::job::CandleRow],
) -> Result<i64> {
    let kind = dataset
        .candle_kind()
        .context("candle dataset omitted kind")?;
    let mut inserted = 0_u64;
    for chunk in rows.chunks(INSERT_BATCH_ROWS) {
        let mut builder = QueryBuilder::<Postgres>::new(
            "INSERT INTO kraken.market_candles (symbol, candle_kind, interval_seconds, \
             bucket_start, open, high, low, close, volume, artifact_id) ",
        );
        builder.push_values(chunk, |mut values, row| {
            values
                .push_bind(&claim.job.symbol)
                .push_bind(kind)
                .push_bind(claim.job.interval_seconds)
                .push_bind(row.bucket_start)
                .push_bind(row.open)
                .push_bind(row.high)
                .push_bind(row.low)
                .push_bind(row.close)
                .push_bind(row.volume)
                .push_bind(artifact_id);
        });
        builder.push(" ON CONFLICT DO NOTHING");
        inserted += builder
            .build()
            .execute(&mut **tx)
            .await
            .context("failed to bulk insert Kraken candles")?
            .rows_affected();
    }
    i64::try_from(inserted).context("Kraken candle count exceeded i64")
}

async fn insert_analytics(
    tx: &mut Transaction<'_, Postgres>,
    claim: &ClaimedKrakenJob,
    artifact_id: Uuid,
    dataset: KrakenDataset,
    rows: &[super::job::AnalyticsRow],
) -> Result<i64> {
    let mut inserted = 0_u64;
    for chunk in rows.chunks(INSERT_BATCH_ROWS) {
        let mut builder = QueryBuilder::<Postgres>::new(
            "INSERT INTO kraken.market_analytics (symbol, dataset, interval_seconds, \
             bucket_start, values, artifact_id) ",
        );
        builder.push_values(chunk, |mut values, row| {
            values
                .push_bind(&claim.job.symbol)
                .push_bind(dataset.as_str())
                .push_bind(claim.job.interval_seconds)
                .push_bind(row.bucket_start)
                .push_bind(&row.values)
                .push_bind(artifact_id);
        });
        builder.push(" ON CONFLICT DO NOTHING");
        inserted += builder
            .build()
            .execute(&mut **tx)
            .await
            .context("failed to bulk insert Kraken analytics")?
            .rows_affected();
    }
    i64::try_from(inserted).context("Kraken analytics count exceeded i64")
}

async fn insert_funding(
    tx: &mut Transaction<'_, Postgres>,
    claim: &ClaimedKrakenJob,
    artifact_id: Uuid,
    rows: &[super::job::FundingRateRow],
) -> Result<i64> {
    let mut inserted = 0_u64;
    for chunk in rows.chunks(INSERT_BATCH_ROWS) {
        let mut builder = QueryBuilder::<Postgres>::new(
            "INSERT INTO kraken.funding_rates (symbol, funding_time, funding_rate, \
             relative_funding_rate, artifact_id) ",
        );
        builder.push_values(chunk, |mut values, row| {
            values
                .push_bind(&claim.job.symbol)
                .push_bind(row.funding_time)
                .push_bind(row.funding_rate)
                .push_bind(row.relative_funding_rate)
                .push_bind(artifact_id);
        });
        builder.push(" ON CONFLICT DO NOTHING");
        inserted += builder
            .build()
            .execute(&mut **tx)
            .await
            .context("failed to bulk insert Kraken funding rates")?
            .rows_affected();
    }
    i64::try_from(inserted).context("Kraken funding rate count exceeded i64")
}

async fn update_coverage(
    tx: &mut Transaction<'_, Postgres>,
    dataset: KrakenDataset,
    symbol: &str,
    interval_seconds: i32,
    bounds: Option<(DateTime<Utc>, DateTime<Utc>)>,
    inserted_rows: i64,
) -> Result<()> {
    sqlx::query(
        r#"
        INSERT INTO kraken.ingestion_coverage (
          dataset, symbol, interval_seconds, earliest_observation,
          latest_observation, observed_rows, missing_buckets
        )
        VALUES ($1, $2, $3, $4, $5, $6, NULL)
        ON CONFLICT (dataset, symbol, interval_seconds) DO UPDATE
        SET earliest_observation = CASE
              WHEN EXCLUDED.earliest_observation IS NULL
                THEN kraken.ingestion_coverage.earliest_observation
              WHEN kraken.ingestion_coverage.earliest_observation IS NULL
                THEN EXCLUDED.earliest_observation
              ELSE LEAST(
                kraken.ingestion_coverage.earliest_observation,
                EXCLUDED.earliest_observation
              )
            END,
            latest_observation = CASE
              WHEN EXCLUDED.latest_observation IS NULL
                THEN kraken.ingestion_coverage.latest_observation
              WHEN kraken.ingestion_coverage.latest_observation IS NULL
                THEN EXCLUDED.latest_observation
              ELSE GREATEST(
                kraken.ingestion_coverage.latest_observation,
                EXCLUDED.latest_observation
              )
            END,
            observed_rows = kraken.ingestion_coverage.observed_rows + EXCLUDED.observed_rows,
            refreshed_at = now()
        "#,
    )
    .bind(dataset.as_str())
    .bind(symbol)
    .bind(interval_seconds)
    .bind(bounds.map(|value| value.0))
    .bind(bounds.map(|value| value.1))
    .bind(inserted_rows)
    .execute(&mut **tx)
    .await
    .context("failed to update Kraken ingestion coverage")?;
    Ok(())
}
