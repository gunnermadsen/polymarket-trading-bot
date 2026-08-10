use std::fmt::Write as _;

use chrono::{DateTime, Duration, Utc};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sqlx::{FromRow, PgConnection, PgPool, Postgres, Transaction};
use thiserror::Error;
use uuid::Uuid;

use crate::domain::{ArtifactStatus, CaptureArtifact, IngesterStrategyKey};

const MAX_CONFIG_BYTES: usize = 16_384;
const MAX_CURSOR_BYTES: usize = 2_048;
const MAX_FAILURE_CODE_BYTES: usize = 128;
const MAX_FAILURE_MESSAGE_BYTES: usize = 2_048;
const MAX_CAPTURE_WINDOW: Duration = Duration::days(7);

const ARTIFACT_COLUMNS: &str = r#"
  artifact_id, strategy_key, profile_generation, config_schema_version,
  config_sha256, config_snapshot, capture_window_start, capture_window_end,
  minimum_source_timestamp, maximum_source_timestamp,
  minimum_received_at, maximum_received_at, start_cursor, end_cursor,
  record_count, content_sha256, status, failure_code, failure_message,
  created_at, updated_at, completed_at
"#;

#[derive(Debug, Clone, PartialEq)]
pub struct NewCaptureArtifact {
    pub strategy_key: IngesterStrategyKey,
    pub profile_generation: i64,
    pub config_schema_version: i32,
    pub config_snapshot: Value,
    pub capture_window_start: DateTime<Utc>,
    pub capture_window_end: DateTime<Utc>,
    pub start_cursor: Option<String>,
}

impl NewCaptureArtifact {
    fn validate(&self) -> Result<String, ArtifactPersistenceError> {
        if self.profile_generation <= 0 {
            return Err(ArtifactPersistenceError::InvalidInput(
                "profile generation must be positive".to_owned(),
            ));
        }
        if self.config_schema_version <= 0 {
            return Err(ArtifactPersistenceError::InvalidInput(
                "config schema version must be positive".to_owned(),
            ));
        }
        if !self.config_snapshot.is_object() {
            return Err(ArtifactPersistenceError::InvalidInput(
                "artifact config snapshot must be a JSON object".to_owned(),
            ));
        }
        let encoded = serde_json::to_vec(&self.config_snapshot)
            .map_err(ArtifactPersistenceError::SerializeConfig)?;
        if encoded.len() > MAX_CONFIG_BYTES {
            return Err(ArtifactPersistenceError::InvalidInput(format!(
                "artifact config snapshot exceeds {MAX_CONFIG_BYTES} bytes"
            )));
        }
        let window = self.capture_window_end - self.capture_window_start;
        if window <= Duration::zero() || window > MAX_CAPTURE_WINDOW {
            return Err(ArtifactPersistenceError::InvalidInput(
                "capture window must be greater than zero and at most seven days".to_owned(),
            ));
        }
        validate_optional_cursor("start cursor", self.start_cursor.as_deref())?;
        Ok(sha256_hex(&encoded))
    }
}

/// The rows actually inserted by one factual persistence transaction.
///
/// Callers must use the database `RETURNING` count, not the number received
/// from a provider, so an idempotent replay cannot inflate artifact lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactBatch {
    pub inserted_record_count: i64,
    pub minimum_source_timestamp: Option<DateTime<Utc>>,
    pub maximum_source_timestamp: Option<DateTime<Utc>>,
    pub minimum_received_at: Option<DateTime<Utc>>,
    pub maximum_received_at: Option<DateTime<Utc>>,
    pub start_cursor: Option<String>,
    pub end_cursor: Option<String>,
}

impl ArtifactBatch {
    fn validate(&self) -> Result<(), ArtifactPersistenceError> {
        if self.inserted_record_count <= 0 {
            return Err(ArtifactPersistenceError::InvalidInput(
                "artifact batch inserted record count must be positive".to_owned(),
            ));
        }
        validate_time_range(
            "source timestamp",
            self.minimum_source_timestamp,
            self.maximum_source_timestamp,
        )?;
        validate_time_range(
            "receipt timestamp",
            self.minimum_received_at,
            self.maximum_received_at,
        )?;
        validate_optional_cursor("start cursor", self.start_cursor.as_deref())?;
        validate_optional_cursor("end cursor", self.end_cursor.as_deref())?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct ArtifactRepository {
    pool: PgPool,
}

impl ArtifactRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create(
        &self,
        artifact: &NewCaptureArtifact,
    ) -> Result<CaptureArtifact, ArtifactPersistenceError> {
        let mut connection = self.pool.acquire().await?;
        create_on(&mut connection, artifact).await
    }

    pub async fn create_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact: &NewCaptureArtifact,
    ) -> Result<CaptureArtifact, ArtifactPersistenceError> {
        create_on(&mut *transaction, artifact).await
    }

    pub async fn get(
        &self,
        artifact_id: Uuid,
    ) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
        let mut connection = self.pool.acquire().await?;
        get_on(&mut connection, artifact_id).await
    }

    pub async fn get_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
    ) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
        get_on(&mut *transaction, artifact_id).await
    }

    pub async fn get_open(
        &self,
        strategy_key: IngesterStrategyKey,
    ) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
        let mut connection = self.pool.acquire().await?;
        get_open_on(&mut connection, strategy_key).await
    }

    pub async fn get_open_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        strategy_key: IngesterStrategyKey,
    ) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
        get_open_on(&mut *transaction, strategy_key).await
    }

    /// Lists open windows whose configured end has passed, allowing startup
    /// reconciliation to resume or explicitly fail them before opening another.
    pub async fn list_stale_open(
        &self,
        ended_before: DateTime<Utc>,
    ) -> Result<Vec<CaptureArtifact>, ArtifactPersistenceError> {
        let mut connection = self.pool.acquire().await?;
        list_stale_open_on(&mut connection, ended_before).await
    }

    pub async fn list_stale_open_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        ended_before: DateTime<Utc>,
    ) -> Result<Vec<CaptureArtifact>, ArtifactPersistenceError> {
        list_stale_open_on(&mut *transaction, ended_before).await
    }

    pub async fn record_batch(
        &self,
        artifact_id: Uuid,
        batch: &ArtifactBatch,
    ) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
        let mut connection = self.pool.acquire().await?;
        record_batch_on(&mut connection, artifact_id, batch).await
    }

    /// Updates capture lineage on the same transaction as source-fact inserts.
    pub async fn record_batch_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        batch: &ArtifactBatch,
    ) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
        record_batch_on(&mut *transaction, artifact_id, batch).await
    }

    pub async fn complete(
        &self,
        artifact_id: Uuid,
        content_sha256: &str,
        end_cursor: Option<&str>,
    ) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
        let mut connection = self.pool.acquire().await?;
        complete_on(&mut connection, artifact_id, content_sha256, end_cursor).await
    }

    pub async fn complete_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        content_sha256: &str,
        end_cursor: Option<&str>,
    ) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
        complete_on(&mut *transaction, artifact_id, content_sha256, end_cursor).await
    }

    pub async fn fail(
        &self,
        artifact_id: Uuid,
        failure_code: &str,
        failure_message: Option<&str>,
    ) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
        let mut connection = self.pool.acquire().await?;
        fail_on(&mut connection, artifact_id, failure_code, failure_message).await
    }

    pub async fn fail_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        failure_code: &str,
        failure_message: Option<&str>,
    ) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
        fail_on(
            &mut *transaction,
            artifact_id,
            failure_code,
            failure_message,
        )
        .await
    }
}

async fn create_on(
    connection: &mut PgConnection,
    artifact: &NewCaptureArtifact,
) -> Result<CaptureArtifact, ArtifactPersistenceError> {
    let config_sha256 = artifact.validate()?;
    let query = format!(
        r#"
        INSERT INTO ingester.capture_artifacts (
          strategy_key, profile_generation, config_schema_version,
          config_sha256, config_snapshot, capture_window_start,
          capture_window_end, start_cursor
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING {ARTIFACT_COLUMNS}
        "#
    );
    let row = sqlx::query_as::<_, ArtifactRow>(&query)
        .bind(artifact.strategy_key.as_str())
        .bind(artifact.profile_generation)
        .bind(artifact.config_schema_version)
        .bind(config_sha256)
        .bind(&artifact.config_snapshot)
        .bind(artifact.capture_window_start)
        .bind(artifact.capture_window_end)
        .bind(artifact.start_cursor.as_deref())
        .fetch_one(connection)
        .await?;
    row.try_into()
}

async fn get_on(
    connection: &mut PgConnection,
    artifact_id: Uuid,
) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
    let query =
        format!("SELECT {ARTIFACT_COLUMNS} FROM ingester.capture_artifacts WHERE artifact_id = $1");
    sqlx::query_as::<_, ArtifactRow>(&query)
        .bind(artifact_id)
        .fetch_optional(connection)
        .await?
        .map(TryInto::try_into)
        .transpose()
}

async fn get_open_on(
    connection: &mut PgConnection,
    strategy_key: IngesterStrategyKey,
) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
    let query = format!(
        "SELECT {ARTIFACT_COLUMNS} FROM ingester.capture_artifacts \
         WHERE strategy_key = $1 AND status = 'open'"
    );
    sqlx::query_as::<_, ArtifactRow>(&query)
        .bind(strategy_key.as_str())
        .fetch_optional(connection)
        .await?
        .map(TryInto::try_into)
        .transpose()
}

async fn list_stale_open_on(
    connection: &mut PgConnection,
    ended_before: DateTime<Utc>,
) -> Result<Vec<CaptureArtifact>, ArtifactPersistenceError> {
    let query = format!(
        "SELECT {ARTIFACT_COLUMNS} FROM ingester.capture_artifacts \
         WHERE status = 'open' AND capture_window_end <= $1 \
         ORDER BY capture_window_end, artifact_id"
    );
    sqlx::query_as::<_, ArtifactRow>(&query)
        .bind(ended_before)
        .fetch_all(connection)
        .await?
        .into_iter()
        .map(TryInto::try_into)
        .collect()
}

async fn record_batch_on(
    connection: &mut PgConnection,
    artifact_id: Uuid,
    batch: &ArtifactBatch,
) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
    batch.validate()?;
    let query = format!(
        r#"
        UPDATE ingester.capture_artifacts
        SET record_count = record_count + $2,
            minimum_source_timestamp = CASE
              WHEN minimum_source_timestamp IS NULL THEN $3
              WHEN $3 IS NULL THEN minimum_source_timestamp
              ELSE LEAST(minimum_source_timestamp, $3)
            END,
            maximum_source_timestamp = CASE
              WHEN maximum_source_timestamp IS NULL THEN $4
              WHEN $4 IS NULL THEN maximum_source_timestamp
              ELSE GREATEST(maximum_source_timestamp, $4)
            END,
            minimum_received_at = CASE
              WHEN minimum_received_at IS NULL THEN $5
              WHEN $5 IS NULL THEN minimum_received_at
              ELSE LEAST(minimum_received_at, $5)
            END,
            maximum_received_at = CASE
              WHEN maximum_received_at IS NULL THEN $6
              WHEN $6 IS NULL THEN maximum_received_at
              ELSE GREATEST(maximum_received_at, $6)
            END,
            start_cursor = COALESCE(start_cursor, $7),
            end_cursor = COALESCE($8, end_cursor),
            updated_at = now()
        WHERE artifact_id = $1 AND status = 'open'
        RETURNING {ARTIFACT_COLUMNS}
        "#
    );
    sqlx::query_as::<_, ArtifactRow>(&query)
        .bind(artifact_id)
        .bind(batch.inserted_record_count)
        .bind(batch.minimum_source_timestamp)
        .bind(batch.maximum_source_timestamp)
        .bind(batch.minimum_received_at)
        .bind(batch.maximum_received_at)
        .bind(batch.start_cursor.as_deref())
        .bind(batch.end_cursor.as_deref())
        .fetch_optional(connection)
        .await?
        .map(TryInto::try_into)
        .transpose()
}

async fn complete_on(
    connection: &mut PgConnection,
    artifact_id: Uuid,
    content_sha256: &str,
    end_cursor: Option<&str>,
) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
    validate_sha256("content SHA-256", content_sha256)?;
    validate_optional_cursor("end cursor", end_cursor)?;
    let query = format!(
        r#"
        UPDATE ingester.capture_artifacts
        SET content_sha256 = $2,
            end_cursor = COALESCE($3, end_cursor),
            status = 'completed',
            updated_at = now(),
            completed_at = now()
        WHERE artifact_id = $1 AND status = 'open'
        RETURNING {ARTIFACT_COLUMNS}
        "#
    );
    sqlx::query_as::<_, ArtifactRow>(&query)
        .bind(artifact_id)
        .bind(content_sha256)
        .bind(end_cursor)
        .fetch_optional(connection)
        .await?
        .map(TryInto::try_into)
        .transpose()
}

async fn fail_on(
    connection: &mut PgConnection,
    artifact_id: Uuid,
    failure_code: &str,
    failure_message: Option<&str>,
) -> Result<Option<CaptureArtifact>, ArtifactPersistenceError> {
    validate_nonempty_bounded("failure code", failure_code, MAX_FAILURE_CODE_BYTES)?;
    let bounded_message =
        failure_message.map(|value| truncate_utf8(value, MAX_FAILURE_MESSAGE_BYTES));
    let query = format!(
        r#"
        UPDATE ingester.capture_artifacts
        SET status = 'failed',
            failure_code = $2,
            failure_message = $3,
            updated_at = now(),
            completed_at = now()
        WHERE artifact_id = $1 AND status = 'open'
        RETURNING {ARTIFACT_COLUMNS}
        "#
    );
    sqlx::query_as::<_, ArtifactRow>(&query)
        .bind(artifact_id)
        .bind(failure_code)
        .bind(bounded_message)
        .fetch_optional(connection)
        .await?
        .map(TryInto::try_into)
        .transpose()
}

#[derive(Debug, FromRow)]
struct ArtifactRow {
    artifact_id: Uuid,
    strategy_key: String,
    profile_generation: i64,
    config_schema_version: i32,
    config_sha256: String,
    config_snapshot: Value,
    capture_window_start: DateTime<Utc>,
    capture_window_end: DateTime<Utc>,
    minimum_source_timestamp: Option<DateTime<Utc>>,
    maximum_source_timestamp: Option<DateTime<Utc>>,
    minimum_received_at: Option<DateTime<Utc>>,
    maximum_received_at: Option<DateTime<Utc>>,
    start_cursor: Option<String>,
    end_cursor: Option<String>,
    record_count: i64,
    content_sha256: Option<String>,
    status: String,
    failure_code: Option<String>,
    failure_message: Option<String>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    completed_at: Option<DateTime<Utc>>,
}

impl TryFrom<ArtifactRow> for CaptureArtifact {
    type Error = ArtifactPersistenceError;

    fn try_from(row: ArtifactRow) -> Result<Self, Self::Error> {
        let strategy_key = row.strategy_key.parse().map_err(|error| {
            ArtifactPersistenceError::StoredData(format!(
                "capture artifact contains an unknown strategy key: {error}"
            ))
        })?;
        let status = ArtifactStatus::from_database(&row.status).ok_or_else(|| {
            ArtifactPersistenceError::StoredData(format!(
                "capture artifact contains unknown status {}",
                row.status
            ))
        })?;
        Ok(Self {
            artifact_id: row.artifact_id,
            strategy_key,
            profile_generation: row.profile_generation,
            config_schema_version: row.config_schema_version,
            config_sha256: row.config_sha256,
            config_snapshot: row.config_snapshot,
            capture_window_start: row.capture_window_start,
            capture_window_end: row.capture_window_end,
            minimum_source_timestamp: row.minimum_source_timestamp,
            maximum_source_timestamp: row.maximum_source_timestamp,
            minimum_received_at: row.minimum_received_at,
            maximum_received_at: row.maximum_received_at,
            start_cursor: row.start_cursor,
            end_cursor: row.end_cursor,
            record_count: row.record_count,
            content_sha256: row.content_sha256,
            status,
            failure_code: row.failure_code,
            failure_message: row.failure_message,
            created_at: row.created_at,
            updated_at: row.updated_at,
            completed_at: row.completed_at,
        })
    }
}

#[derive(Debug, Error)]
pub enum ArtifactPersistenceError {
    #[error("invalid capture artifact input: {0}")]
    InvalidInput(String),
    #[error("failed to serialize capture configuration: {0}")]
    SerializeConfig(#[source] serde_json::Error),
    #[error("capture artifact database operation failed: {0}")]
    Database(#[from] sqlx::Error),
    #[error("stored capture artifact is invalid: {0}")]
    StoredData(String),
}

fn validate_time_range(
    name: &str,
    minimum: Option<DateTime<Utc>>,
    maximum: Option<DateTime<Utc>>,
) -> Result<(), ArtifactPersistenceError> {
    match (minimum, maximum) {
        (None, None) => Ok(()),
        (Some(minimum), Some(maximum)) if maximum >= minimum => Ok(()),
        (Some(_), Some(_)) => Err(ArtifactPersistenceError::InvalidInput(format!(
            "artifact {name} maximum precedes its minimum"
        ))),
        _ => Err(ArtifactPersistenceError::InvalidInput(format!(
            "artifact {name} range must be entirely present or absent"
        ))),
    }
}

fn validate_optional_cursor(
    name: &str,
    value: Option<&str>,
) -> Result<(), ArtifactPersistenceError> {
    if value.is_some_and(|value| value.len() > MAX_CURSOR_BYTES) {
        return Err(ArtifactPersistenceError::InvalidInput(format!(
            "artifact {name} exceeds {MAX_CURSOR_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_nonempty_bounded(
    name: &str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), ArtifactPersistenceError> {
    if value.trim().is_empty() || value.len() > maximum_bytes {
        return Err(ArtifactPersistenceError::InvalidInput(format!(
            "artifact {name} must contain between 1 and {maximum_bytes} bytes"
        )));
    }
    Ok(())
}

fn validate_sha256(name: &str, value: &str) -> Result<(), ArtifactPersistenceError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ArtifactPersistenceError::InvalidInput(format!(
            "artifact {name} must be a lowercase hexadecimal SHA-256"
        )));
    }
    Ok(())
}

fn truncate_utf8(value: &str, maximum_bytes: usize) -> &str {
    if value.len() <= maximum_bytes {
        return value;
    }
    let mut boundary = maximum_bytes;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    &value[..boundary]
}

fn sha256_hex(value: &[u8]) -> String {
    let digest = Sha256::digest(value);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use serde_json::json;

    use super::*;

    fn timestamp(seconds: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(seconds, 0).single().expect("valid time")
    }

    fn new_artifact() -> NewCaptureArtifact {
        NewCaptureArtifact {
            strategy_key: IngesterStrategyKey::BinanceSpotBtcusdtAggregateTrades,
            profile_generation: 1,
            config_schema_version: 1,
            config_snapshot: json!({"batch_size": 100, "flush_ms": 250}),
            capture_window_start: timestamp(1_000),
            capture_window_end: timestamp(4_600),
            start_cursor: Some("aggregate_trade_id:10".to_owned()),
        }
    }

    #[test]
    fn configuration_hash_is_deterministic() {
        let first = new_artifact().validate().expect("valid artifact");
        let mut reordered = new_artifact();
        reordered.config_snapshot = json!({"flush_ms": 250, "batch_size": 100});
        assert_eq!(first, reordered.validate().expect("valid artifact"));
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn capture_window_is_bounded() {
        let mut artifact = new_artifact();
        artifact.capture_window_end = artifact.capture_window_start;
        assert!(artifact.validate().is_err());

        artifact.capture_window_end = artifact.capture_window_start + Duration::days(8);
        assert!(artifact.validate().is_err());
    }

    #[test]
    fn batch_requires_complete_ordered_ranges() {
        let batch = ArtifactBatch {
            inserted_record_count: 2,
            minimum_source_timestamp: Some(timestamp(20)),
            maximum_source_timestamp: Some(timestamp(19)),
            minimum_received_at: None,
            maximum_received_at: None,
            start_cursor: None,
            end_cursor: None,
        };
        assert!(batch.validate().is_err());
    }

    #[test]
    fn bounded_diagnostics_preserve_utf8() {
        assert_eq!(truncate_utf8("ab💡cd", 5), "ab");
    }

    #[test]
    fn sha256_validation_rejects_uppercase_and_wrong_lengths() {
        assert!(validate_sha256("content", &"a".repeat(64)).is_ok());
        assert!(validate_sha256("content", &"A".repeat(64)).is_err());
        assert!(validate_sha256("content", &"a".repeat(63)).is_err());
    }

    #[test]
    fn empty_capture_has_a_valid_content_checksum() {
        let empty_sha256 = sha256_hex(b"");
        assert_eq!(
            empty_sha256,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert!(validate_sha256("content", &empty_sha256).is_ok());
    }
}
