use std::{str::FromStr, time::Duration};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::{FromRow, PgConnection, PgPool, Postgres, Transaction};
use thiserror::Error;
use uuid::Uuid;

use crate::domain::{
    DesiredState, HealthStatus, IngesterProfile, IngesterStrategyKey, ObservedState,
};

const PROFILE_COLUMNS: &str = r#"
  strategy_key, config_schema_version, config, desired_state,
  desired_generation, observed_state, health_status, applied_generation,
  checkpoint_schema_version, checkpoint, lease_owner, lease_token,
  lease_expires_at, heartbeat_at, started_at, stopped_at,
  last_source_event_at, last_provider_available_at, last_persisted_at,
  source_watermark, availability_watermark, consecutive_failures,
  restart_count, last_error_code, last_error_message, last_error_at,
  created_at, updated_at
"#;

#[derive(Clone)]
pub struct ProfileRepository {
    pool: PgPool,
}

/// Durable source progress written on the same transaction as factual rows.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyProgress {
    /// Facts inserted or replayed and verified equal to an existing durable row.
    pub verified_record_count: i64,
    pub checkpoint_schema_version: i32,
    pub checkpoint: Value,
    pub last_source_event_at: Option<DateTime<Utc>>,
    pub last_provider_available_at: Option<DateTime<Utc>>,
    pub source_watermark: Option<DateTime<Utc>>,
    pub availability_watermark: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyDegradation {
    pub reason_code: String,
    pub reason_message: String,
}

impl StrategyProgress {
    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.verified_record_count > 0,
            "strategy progress requires at least one durably verified fact"
        );
        anyhow::ensure!(
            self.checkpoint_schema_version > 0,
            "checkpoint schema version must be positive"
        );
        anyhow::ensure!(
            self.checkpoint.is_object(),
            "strategy checkpoint must be a JSON object"
        );
        let encoded = serde_json::to_vec(&self.checkpoint)
            .context("failed to serialize strategy checkpoint")?;
        anyhow::ensure!(
            encoded.len() <= 8_192,
            "strategy checkpoint exceeds 8192 bytes"
        );
        Ok(())
    }
}

impl ProfileRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Locks and verifies the active profile lease for a short control-plane
    /// transaction that cannot use `record_progress_in` as its fence.
    pub async fn lock_current_lease_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        generation: i64,
    ) -> Result<bool> {
        let locked = sqlx::query_scalar::<_, String>(
            r#"
            SELECT strategy_key
            FROM ingester.profiles
            WHERE strategy_key = $1
              AND lease_owner = $2
              AND lease_token = $3
              AND lease_expires_at > now()
              AND desired_state = 'running'
              AND desired_generation = $4
              AND applied_generation = $4
            FOR UPDATE
            "#,
        )
        .bind(key.as_str())
        .bind(owner)
        .bind(token)
        .bind(generation)
        .fetch_optional(&mut **transaction)
        .await
        .context("failed to lock current ingester profile lease")?;
        Ok(locked.is_some())
    }

    /// Locks a still-owned lease while a superseded generation drains after a
    /// requested stop or restart. This intentionally ignores desired state and
    /// desired generation, but never ignores owner, token, expiry, or the
    /// generation that was actually applied to the running strategy.
    pub async fn lock_owned_lease_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        applied_generation: i64,
    ) -> Result<bool> {
        let locked = sqlx::query_scalar::<_, String>(
            r#"
            SELECT strategy_key
            FROM ingester.profiles
            WHERE strategy_key = $1
              AND lease_owner = $2
              AND lease_token = $3
              AND lease_expires_at > now()
              AND applied_generation = $4
            FOR UPDATE
            "#,
        )
        .bind(key.as_str())
        .bind(owner)
        .bind(token)
        .bind(applied_generation)
        .fetch_optional(&mut **transaction)
        .await
        .context("failed to lock owned ingester profile lease")?;
        Ok(locked.is_some())
    }

    pub async fn list(&self) -> Result<Vec<IngesterProfile>> {
        let query =
            format!("SELECT {PROFILE_COLUMNS} FROM ingester.profiles ORDER BY strategy_key");
        sqlx::query_as::<_, ProfileRow>(&query)
            .fetch_all(&self.pool)
            .await
            .context("failed to list ingester profiles")?
            .into_iter()
            .map(TryInto::try_into)
            .collect()
    }

    pub async fn get(&self, key: IngesterStrategyKey) -> Result<Option<IngesterProfile>> {
        let query =
            format!("SELECT {PROFILE_COLUMNS} FROM ingester.profiles WHERE strategy_key = $1");
        sqlx::query_as::<_, ProfileRow>(&query)
            .bind(key.as_str())
            .fetch_optional(&self.pool)
            .await
            .context("failed to load ingester profile")?
            .map(TryInto::try_into)
            .transpose()
    }

    pub async fn replace_config(
        &self,
        key: IngesterStrategyKey,
        expected_generation: i64,
        schema_version: i32,
        config: &Value,
    ) -> Result<IngesterProfile, ProfileWriteError> {
        let query = format!(
            r#"
            UPDATE ingester.profiles
            SET config_schema_version = $3,
                config = $4,
                desired_generation = desired_generation + 1,
                updated_at = now()
            WHERE strategy_key = $1 AND desired_generation = $2
            RETURNING {PROFILE_COLUMNS}
            "#
        );
        self.fetch_updated(
            sqlx::query_as::<_, ProfileRow>(&query)
                .bind(key.as_str())
                .bind(expected_generation)
                .bind(schema_version)
                .bind(config),
            key,
            expected_generation,
        )
        .await
    }

    pub async fn request_state(
        &self,
        key: IngesterStrategyKey,
        expected_generation: i64,
        desired_state: DesiredState,
        force_generation: bool,
    ) -> Result<IngesterProfile, ProfileWriteError> {
        let query = format!(
            r#"
            UPDATE ingester.profiles
            SET desired_state = $3,
                desired_generation = desired_generation
                  + CASE WHEN $4 OR desired_state IS DISTINCT FROM $3 THEN 1 ELSE 0 END,
                updated_at = now()
            WHERE strategy_key = $1 AND desired_generation = $2
            RETURNING {PROFILE_COLUMNS}
            "#
        );
        self.fetch_updated(
            sqlx::query_as::<_, ProfileRow>(&query)
                .bind(key.as_str())
                .bind(expected_generation)
                .bind(desired_state.as_str())
                .bind(force_generation),
            key,
            expected_generation,
        )
        .await
    }

    pub async fn claim_lease(
        &self,
        key: IngesterStrategyKey,
        generation: i64,
        owner: &str,
        lease_duration: Duration,
    ) -> Result<Option<(IngesterProfile, Uuid)>> {
        let token = Uuid::new_v4();
        let lease_milliseconds = duration_milliseconds(lease_duration)?;
        let query = format!(
            r#"
            UPDATE ingester.profiles
            SET lease_owner = $3,
                lease_token = $4,
                heartbeat_at = now(),
                lease_expires_at = now() + ($5 * INTERVAL '1 millisecond'),
                observed_state = 'starting',
                health_status = 'unknown',
                restart_count = restart_count + CASE
                  WHEN applied_generation IS NOT NULL
                    AND applied_generation <> desired_generation THEN 1
                  ELSE 0
                END,
                last_error_code = NULL,
                last_error_message = NULL,
                last_error_at = NULL,
                updated_at = now()
            WHERE strategy_key = $1
              AND desired_generation = $2
              AND desired_state = 'running'
              AND (lease_token IS NULL OR lease_expires_at <= now())
            RETURNING {PROFILE_COLUMNS}
            "#
        );
        sqlx::query_as::<_, ProfileRow>(&query)
            .bind(key.as_str())
            .bind(generation)
            .bind(owner)
            .bind(token)
            .bind(lease_milliseconds)
            .fetch_optional(&self.pool)
            .await
            .context("failed to claim ingester profile lease")?
            .map(TryInto::try_into)
            .transpose()
            .map(|profile| profile.map(|profile| (profile, token)))
    }

    pub async fn renew_lease(
        &self,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        lease_duration: Duration,
    ) -> Result<bool> {
        let lease_milliseconds = duration_milliseconds(lease_duration)?;
        let updated = sqlx::query_scalar::<_, String>(
            r#"
            UPDATE ingester.profiles
            SET heartbeat_at = now(),
                lease_expires_at = now() + ($4 * INTERVAL '1 millisecond'),
                updated_at = now()
            WHERE strategy_key = $1
              AND lease_owner = $2
              AND lease_token = $3
              AND lease_expires_at > now()
              AND desired_state = 'running'
            RETURNING strategy_key
            "#,
        )
        .bind(key.as_str())
        .bind(owner)
        .bind(token)
        .bind(lease_milliseconds)
        .fetch_optional(&self.pool)
        .await
        .context("failed to renew ingester profile lease")?;
        Ok(updated.is_some())
    }

    pub async fn mark_running(
        &self,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        generation: i64,
    ) -> Result<bool> {
        let updated = sqlx::query_scalar::<_, String>(
            r#"
            UPDATE ingester.profiles
            SET observed_state = 'running',
                health_status = 'unknown',
                applied_generation = $4,
                started_at = now(),
                stopped_at = NULL,
                updated_at = now()
            WHERE strategy_key = $1
              AND lease_owner = $2
              AND lease_token = $3
              AND lease_expires_at > now()
              AND desired_state = 'running'
              AND desired_generation = $4
            RETURNING strategy_key
            "#,
        )
        .bind(key.as_str())
        .bind(owner)
        .bind(token)
        .bind(generation)
        .fetch_optional(&self.pool)
        .await
        .context("failed to mark ingester profile running")?;
        Ok(updated.is_some())
    }

    /// Commits checkpoint, watermarks, and healthy state with source facts.
    ///
    /// A caller must invoke this on the transaction that inserted a fact or
    /// verified an idempotent replay against a durable row. This is the only
    /// path that clears consecutive failures.
    pub async fn record_progress_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        generation: i64,
        progress: &StrategyProgress,
    ) -> Result<bool> {
        progress.validate()?;
        let updated = sqlx::query_scalar::<_, String>(
            r#"
            WITH gap_health AS (
              SELECT
                EXISTS (
                SELECT 1
                FROM ingester.data_gaps
                WHERE strategy_key = $1
                  AND status IN ('open', 'repairing')
                ) AS has_unresolved_gap
            )
            UPDATE ingester.profiles
            SET checkpoint_schema_version = $5,
                checkpoint = $6,
                last_source_event_at = CASE
                  WHEN $7::timestamptz IS NULL THEN last_source_event_at
                  WHEN last_source_event_at IS NULL THEN $7
                  ELSE GREATEST(last_source_event_at, $7)
                END,
                last_provider_available_at = CASE
                  WHEN $8::timestamptz IS NULL THEN last_provider_available_at
                  WHEN last_provider_available_at IS NULL THEN $8
                  ELSE GREATEST(last_provider_available_at, $8)
                END,
                last_persisted_at = now(),
                source_watermark = CASE
                  WHEN $9::timestamptz IS NULL THEN source_watermark
                  WHEN source_watermark IS NULL THEN $9
                  ELSE GREATEST(source_watermark, $9)
                END,
                availability_watermark = CASE
                  WHEN $10::timestamptz IS NULL THEN availability_watermark
                  WHEN availability_watermark IS NULL THEN $10
                  ELSE GREATEST(availability_watermark, $10)
                END,
                observed_state = CASE
                  WHEN gap_health.has_unresolved_gap THEN 'degraded'
                  ELSE 'running'
                END,
                health_status = CASE
                  WHEN gap_health.has_unresolved_gap THEN 'degraded'
                  ELSE 'healthy'
                END,
                consecutive_failures = 0,
                last_error_code = CASE
                  WHEN gap_health.has_unresolved_gap THEN last_error_code
                  ELSE NULL
                END,
                last_error_message = CASE
                  WHEN gap_health.has_unresolved_gap THEN last_error_message
                  ELSE NULL
                END,
                last_error_at = CASE
                  WHEN gap_health.has_unresolved_gap THEN last_error_at
                  ELSE NULL
                END,
                updated_at = now()
            FROM gap_health
            WHERE strategy_key = $1
              AND lease_owner = $2
              AND lease_token = $3
              AND lease_expires_at > now()
              AND desired_state = 'running'
              AND desired_generation = $4
              AND applied_generation = $4
            RETURNING strategy_key
            "#,
        )
        .bind(key.as_str())
        .bind(owner)
        .bind(token)
        .bind(generation)
        .bind(progress.checkpoint_schema_version)
        .bind(&progress.checkpoint)
        .bind(progress.last_source_event_at)
        .bind(progress.last_provider_available_at)
        .bind(progress.source_watermark)
        .bind(progress.availability_watermark)
        .fetch_optional(&mut **transaction)
        .await
        .context("failed to commit ingester profile progress")?;
        Ok(updated.is_some())
    }

    pub async fn mark_degraded(
        &self,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        generation: i64,
        degradation: &StrategyDegradation,
    ) -> Result<bool> {
        let mut connection = self.pool.acquire().await?;
        mark_degraded_on(
            &mut connection,
            key,
            owner,
            token,
            generation,
            &degradation.reason_code,
            &degradation.reason_message,
        )
        .await
    }

    /// Marks health degraded on the same transaction that records a gap.
    pub async fn mark_degraded_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        generation: i64,
        degradation: &StrategyDegradation,
    ) -> Result<bool> {
        mark_degraded_on(
            &mut *transaction,
            key,
            owner,
            token,
            generation,
            &degradation.reason_code,
            &degradation.reason_message,
        )
        .await
    }

    pub async fn mark_stopped(
        &self,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        generation: i64,
    ) -> Result<bool> {
        let updated = sqlx::query_scalar::<_, String>(
            r#"
            UPDATE ingester.profiles
            SET observed_state = 'stopped',
                health_status = 'unknown',
                applied_generation = $4,
                stopped_at = now(),
                lease_owner = NULL,
                lease_token = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL,
                updated_at = now()
            WHERE strategy_key = $1 AND lease_owner = $2 AND lease_token = $3
            RETURNING strategy_key
            "#,
        )
        .bind(key.as_str())
        .bind(owner)
        .bind(token)
        .bind(generation)
        .fetch_optional(&self.pool)
        .await
        .context("failed to mark ingester profile stopped")?;
        Ok(updated.is_some())
    }

    pub async fn mark_failed(
        &self,
        key: IngesterStrategyKey,
        owner: &str,
        token: Uuid,
        error_code: &str,
        error_message: &str,
    ) -> Result<bool> {
        let bounded_message = bounded(error_message, 2048);
        let updated = sqlx::query_scalar::<_, String>(
            r#"
            UPDATE ingester.profiles
            SET observed_state = 'failed',
                health_status = 'unhealthy',
                consecutive_failures = consecutive_failures + 1,
                last_error_code = $4,
                last_error_message = $5,
                last_error_at = now(),
                stopped_at = now(),
                lease_owner = NULL,
                lease_token = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL,
                updated_at = now()
            WHERE strategy_key = $1 AND lease_owner = $2 AND lease_token = $3
            RETURNING strategy_key
            "#,
        )
        .bind(key.as_str())
        .bind(owner)
        .bind(token)
        .bind(error_code)
        .bind(bounded_message)
        .fetch_optional(&self.pool)
        .await
        .context("failed to persist ingester strategy failure")?;
        Ok(updated.is_some())
    }

    async fn fetch_updated<'q>(
        &self,
        query: sqlx::query::QueryAs<'q, sqlx::Postgres, ProfileRow, sqlx::postgres::PgArguments>,
        key: IngesterStrategyKey,
        expected_generation: i64,
    ) -> Result<IngesterProfile, ProfileWriteError> {
        if let Some(row) = query
            .fetch_optional(&self.pool)
            .await
            .map_err(ProfileWriteError::Database)?
        {
            return row.try_into().map_err(ProfileWriteError::Conversion);
        }
        let current = self
            .get(key)
            .await
            .map_err(ProfileWriteError::ReadCurrent)?;
        match current {
            None => Err(ProfileWriteError::NotFound(key)),
            Some(profile) => Err(ProfileWriteError::GenerationConflict {
                key,
                expected: expected_generation,
                actual: profile.desired_generation,
            }),
        }
    }
}

async fn mark_degraded_on(
    connection: &mut PgConnection,
    key: IngesterStrategyKey,
    owner: &str,
    token: Uuid,
    generation: i64,
    reason_code: &str,
    reason_message: &str,
) -> Result<bool> {
    anyhow::ensure!(
        !reason_code.trim().is_empty() && reason_code.len() <= 128,
        "degraded reason code must contain between 1 and 128 bytes"
    );
    let reason_message = bounded(reason_message, 2_048);
    let updated = sqlx::query_scalar::<_, String>(
        r#"
        UPDATE ingester.profiles
        SET observed_state = 'degraded',
            health_status = 'degraded',
            last_error_code = $5,
            last_error_message = $6,
            last_error_at = now(),
            updated_at = now()
        WHERE strategy_key = $1
          AND lease_owner = $2
          AND lease_token = $3
          AND lease_expires_at > now()
          AND desired_state = 'running'
          AND desired_generation = $4
          AND applied_generation = $4
        RETURNING strategy_key
        "#,
    )
    .bind(key.as_str())
    .bind(owner)
    .bind(token)
    .bind(generation)
    .bind(reason_code)
    .bind(reason_message)
    .fetch_optional(connection)
    .await
    .context("failed to mark ingester profile degraded")?;
    Ok(updated.is_some())
}

#[derive(Debug, Error)]
pub enum ProfileWriteError {
    #[error("ingester profile {0} does not exist")]
    NotFound(IngesterStrategyKey),
    #[error("ingester profile {key} generation conflict: expected {expected}, current {actual}")]
    GenerationConflict {
        key: IngesterStrategyKey,
        expected: i64,
        actual: i64,
    },
    #[error("profile database write failed: {0}")]
    Database(#[source] sqlx::Error),
    #[error("failed to load current profile after write conflict: {0}")]
    ReadCurrent(#[source] anyhow::Error),
    #[error("stored profile is invalid: {0}")]
    Conversion(#[source] anyhow::Error),
}

#[derive(Debug, FromRow)]
struct ProfileRow {
    strategy_key: String,
    config_schema_version: i32,
    config: Value,
    desired_state: String,
    desired_generation: i64,
    observed_state: String,
    health_status: String,
    applied_generation: Option<i64>,
    checkpoint_schema_version: i32,
    checkpoint: Value,
    lease_owner: Option<String>,
    lease_token: Option<Uuid>,
    lease_expires_at: Option<DateTime<Utc>>,
    heartbeat_at: Option<DateTime<Utc>>,
    started_at: Option<DateTime<Utc>>,
    stopped_at: Option<DateTime<Utc>>,
    last_source_event_at: Option<DateTime<Utc>>,
    last_provider_available_at: Option<DateTime<Utc>>,
    last_persisted_at: Option<DateTime<Utc>>,
    source_watermark: Option<DateTime<Utc>>,
    availability_watermark: Option<DateTime<Utc>>,
    consecutive_failures: i32,
    restart_count: i64,
    last_error_code: Option<String>,
    last_error_message: Option<String>,
    last_error_at: Option<DateTime<Utc>>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

impl TryFrom<ProfileRow> for IngesterProfile {
    type Error = anyhow::Error;

    fn try_from(row: ProfileRow) -> Result<Self> {
        Ok(Self {
            strategy_key: IngesterStrategyKey::from_str(&row.strategy_key)?,
            config_schema_version: row.config_schema_version,
            config: row.config,
            desired_state: parse_desired_state(&row.desired_state)?,
            desired_generation: row.desired_generation,
            observed_state: parse_observed_state(&row.observed_state)?,
            health_status: parse_health_status(&row.health_status)?,
            applied_generation: row.applied_generation,
            checkpoint_schema_version: row.checkpoint_schema_version,
            checkpoint: row.checkpoint,
            lease_owner: row.lease_owner,
            lease_token: row.lease_token,
            lease_expires_at: row.lease_expires_at,
            heartbeat_at: row.heartbeat_at,
            started_at: row.started_at,
            stopped_at: row.stopped_at,
            last_source_event_at: row.last_source_event_at,
            last_provider_available_at: row.last_provider_available_at,
            last_persisted_at: row.last_persisted_at,
            source_watermark: row.source_watermark,
            availability_watermark: row.availability_watermark,
            consecutive_failures: row.consecutive_failures,
            restart_count: row.restart_count,
            last_error_code: row.last_error_code,
            last_error_message: row.last_error_message,
            last_error_at: row.last_error_at,
            created_at: row.created_at,
            updated_at: row.updated_at,
        })
    }
}

fn parse_desired_state(value: &str) -> Result<DesiredState> {
    match value {
        "running" => Ok(DesiredState::Running),
        "stopped" => Ok(DesiredState::Stopped),
        other => anyhow::bail!("unknown desired state {other}"),
    }
}

fn parse_observed_state(value: &str) -> Result<ObservedState> {
    match value {
        "starting" => Ok(ObservedState::Starting),
        "running" => Ok(ObservedState::Running),
        "degraded" => Ok(ObservedState::Degraded),
        "restarting" => Ok(ObservedState::Restarting),
        "stopping" => Ok(ObservedState::Stopping),
        "stopped" => Ok(ObservedState::Stopped),
        "failed" => Ok(ObservedState::Failed),
        "unsupported" => Ok(ObservedState::Unsupported),
        other => anyhow::bail!("unknown observed state {other}"),
    }
}

fn parse_health_status(value: &str) -> Result<HealthStatus> {
    match value {
        "unknown" => Ok(HealthStatus::Unknown),
        "healthy" => Ok(HealthStatus::Healthy),
        "degraded" => Ok(HealthStatus::Degraded),
        "unhealthy" => Ok(HealthStatus::Unhealthy),
        other => anyhow::bail!("unknown health status {other}"),
    }
}

fn duration_milliseconds(duration: Duration) -> Result<i64> {
    i64::try_from(duration.as_millis()).context("lease duration exceeds PostgreSQL range")
}

fn bounded(value: &str, maximum_bytes: usize) -> &str {
    if value.len() <= maximum_bytes {
        return value;
    }
    let mut boundary = maximum_bytes;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    &value[..boundary]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_text_preserves_utf8_boundaries() {
        assert_eq!(bounded("abcdef", 4), "abcd");
        assert_eq!(bounded("ab💡cd", 5), "ab");
    }

    #[test]
    fn healthy_progress_requires_a_durable_fact() {
        let progress = StrategyProgress {
            verified_record_count: 0,
            checkpoint_schema_version: 1,
            checkpoint: serde_json::json!({"trade_id": 12}),
            last_source_event_at: None,
            last_provider_available_at: None,
            source_watermark: None,
            availability_watermark: None,
        };
        assert!(progress.validate().is_err());
    }
}
