use std::fmt::Write as _;

use chrono::{DateTime, SecondsFormat, Utc};
use sha2::{Digest, Sha256};
use sqlx::{FromRow, PgConnection, PgPool, Postgres, Transaction};
use thiserror::Error;
use uuid::Uuid;

use crate::domain::{DataGap, GapStatus, IngesterStrategyKey};

const MAX_GAP_KIND_BYTES: usize = 64;
const MAX_REASON_CODE_BYTES: usize = 128;
const MAX_MESSAGE_BYTES: usize = 2_048;
const MAX_CURSOR_BYTES: usize = 2_048;

const GAP_COLUMNS: &str = r#"
  gap_id, gap_fingerprint, strategy_key, detected_artifact_id,
  repair_artifact_id, gap_kind, reason_code, reason_message,
  source_time_start, source_time_end, start_cursor, end_cursor,
  status, repair_attempts, detected_at, repair_started_at, resolved_at,
  resolution_code, resolution_message, created_at, updated_at
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewDataGap {
    pub strategy_key: IngesterStrategyKey,
    pub detected_artifact_id: Option<Uuid>,
    pub gap_kind: String,
    pub reason_code: String,
    pub reason_message: Option<String>,
    pub source_time_start: Option<DateTime<Utc>>,
    pub source_time_end: Option<DateTime<Utc>>,
    pub start_cursor: Option<String>,
    pub end_cursor: Option<String>,
}

impl NewDataGap {
    fn prepare(&self) -> Result<PreparedGap, GapPersistenceError> {
        validate_nonempty_bounded("gap kind", &self.gap_kind, MAX_GAP_KIND_BYTES)?;
        validate_nonempty_bounded("reason code", &self.reason_code, MAX_REASON_CODE_BYTES)?;
        validate_time_range(self.source_time_start, self.source_time_end)?;
        validate_optional_cursor("start cursor", self.start_cursor.as_deref())?;
        validate_optional_cursor("end cursor", self.end_cursor.as_deref())?;
        if self.source_time_start.is_none()
            && self.start_cursor.is_none()
            && self.end_cursor.is_none()
        {
            return Err(GapPersistenceError::InvalidInput(
                "a data gap requires a source-time range or cursor boundary".to_owned(),
            ));
        }

        Ok(PreparedGap {
            fingerprint: gap_fingerprint(self),
            reason_message: self
                .reason_message
                .as_deref()
                .map(|message| truncate_utf8(message, MAX_MESSAGE_BYTES).to_owned()),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GapDetection {
    pub gap: DataGap,
    pub inserted: bool,
}

#[derive(Debug)]
struct PreparedGap {
    fingerprint: String,
    reason_message: Option<String>,
}

#[derive(Clone)]
pub struct GapRepository {
    pool: PgPool,
}

impl GapRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn detect(&self, gap: &NewDataGap) -> Result<GapDetection, GapPersistenceError> {
        let mut connection = self.pool.acquire().await?;
        detect_on(&mut connection, gap).await
    }

    /// Detects a gap on the caller's source-fact transaction.
    pub async fn detect_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        gap: &NewDataGap,
    ) -> Result<GapDetection, GapPersistenceError> {
        detect_on(&mut *transaction, gap).await
    }

    pub async fn get(&self, gap_id: Uuid) -> Result<Option<DataGap>, GapPersistenceError> {
        let mut connection = self.pool.acquire().await?;
        get_on(&mut connection, gap_id).await
    }

    pub async fn get_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        gap_id: Uuid,
    ) -> Result<Option<DataGap>, GapPersistenceError> {
        get_on(&mut *transaction, gap_id).await
    }

    pub async fn list_unresolved(
        &self,
        strategy_key: IngesterStrategyKey,
    ) -> Result<Vec<DataGap>, GapPersistenceError> {
        let mut connection = self.pool.acquire().await?;
        list_unresolved_on(&mut connection, strategy_key).await
    }

    pub async fn list_unresolved_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        strategy_key: IngesterStrategyKey,
    ) -> Result<Vec<DataGap>, GapPersistenceError> {
        list_unresolved_on(&mut *transaction, strategy_key).await
    }

    pub async fn begin_repair(&self, gap_id: Uuid) -> Result<Option<DataGap>, GapPersistenceError> {
        let mut connection = self.pool.acquire().await?;
        begin_repair_on(&mut connection, gap_id).await
    }

    pub async fn begin_repair_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        gap_id: Uuid,
    ) -> Result<Option<DataGap>, GapPersistenceError> {
        begin_repair_on(&mut *transaction, gap_id).await
    }

    pub async fn mark_repaired(
        &self,
        gap_id: Uuid,
        repair_artifact_id: Uuid,
        resolution_code: &str,
        resolution_message: Option<&str>,
    ) -> Result<Option<DataGap>, GapPersistenceError> {
        let mut connection = self.pool.acquire().await?;
        mark_repaired_on(
            &mut connection,
            gap_id,
            repair_artifact_id,
            resolution_code,
            resolution_message,
        )
        .await
    }

    pub async fn mark_repaired_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        gap_id: Uuid,
        repair_artifact_id: Uuid,
        resolution_code: &str,
        resolution_message: Option<&str>,
    ) -> Result<Option<DataGap>, GapPersistenceError> {
        mark_repaired_on(
            &mut *transaction,
            gap_id,
            repair_artifact_id,
            resolution_code,
            resolution_message,
        )
        .await
    }

    pub async fn mark_unrecoverable(
        &self,
        gap_id: Uuid,
        resolution_code: &str,
        resolution_message: Option<&str>,
    ) -> Result<Option<DataGap>, GapPersistenceError> {
        let mut connection = self.pool.acquire().await?;
        mark_unrecoverable_on(&mut connection, gap_id, resolution_code, resolution_message).await
    }

    pub async fn mark_unrecoverable_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        gap_id: Uuid,
        resolution_code: &str,
        resolution_message: Option<&str>,
    ) -> Result<Option<DataGap>, GapPersistenceError> {
        mark_unrecoverable_on(
            &mut *transaction,
            gap_id,
            resolution_code,
            resolution_message,
        )
        .await
    }
}

async fn detect_on(
    connection: &mut PgConnection,
    gap: &NewDataGap,
) -> Result<GapDetection, GapPersistenceError> {
    let prepared = gap.prepare()?;
    let insert = format!(
        r#"
        INSERT INTO ingester.data_gaps (
          gap_fingerprint, strategy_key, detected_artifact_id, gap_kind,
          reason_code, reason_message, source_time_start, source_time_end,
          start_cursor, end_cursor
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (gap_fingerprint) DO NOTHING
        RETURNING {GAP_COLUMNS}
        "#
    );
    let inserted = sqlx::query_as::<_, GapRow>(&insert)
        .bind(&prepared.fingerprint)
        .bind(gap.strategy_key.as_str())
        .bind(gap.detected_artifact_id)
        .bind(&gap.gap_kind)
        .bind(&gap.reason_code)
        .bind(prepared.reason_message)
        .bind(gap.source_time_start)
        .bind(gap.source_time_end)
        .bind(gap.start_cursor.as_deref())
        .bind(gap.end_cursor.as_deref())
        .fetch_optional(&mut *connection)
        .await?;
    if let Some(inserted) = inserted {
        return Ok(GapDetection {
            gap: inserted.try_into()?,
            inserted: true,
        });
    }

    let select = format!("SELECT {GAP_COLUMNS} FROM ingester.data_gaps WHERE gap_fingerprint = $1");
    let existing = sqlx::query_as::<_, GapRow>(&select)
        .bind(&prepared.fingerprint)
        .fetch_optional(connection)
        .await?
        .ok_or_else(|| {
            GapPersistenceError::StoredData(
                "gap fingerprint conflicted but no stored gap was visible".to_owned(),
            )
        })?;
    Ok(GapDetection {
        gap: existing.try_into()?,
        inserted: false,
    })
}

async fn get_on(
    connection: &mut PgConnection,
    gap_id: Uuid,
) -> Result<Option<DataGap>, GapPersistenceError> {
    let query = format!("SELECT {GAP_COLUMNS} FROM ingester.data_gaps WHERE gap_id = $1");
    sqlx::query_as::<_, GapRow>(&query)
        .bind(gap_id)
        .fetch_optional(connection)
        .await?
        .map(TryInto::try_into)
        .transpose()
}

async fn list_unresolved_on(
    connection: &mut PgConnection,
    strategy_key: IngesterStrategyKey,
) -> Result<Vec<DataGap>, GapPersistenceError> {
    let query = format!(
        "SELECT {GAP_COLUMNS} FROM ingester.data_gaps \
         WHERE strategy_key = $1 AND status IN ('open', 'repairing') \
         ORDER BY detected_at, gap_id"
    );
    sqlx::query_as::<_, GapRow>(&query)
        .bind(strategy_key.as_str())
        .fetch_all(connection)
        .await?
        .into_iter()
        .map(TryInto::try_into)
        .collect()
}

async fn begin_repair_on(
    connection: &mut PgConnection,
    gap_id: Uuid,
) -> Result<Option<DataGap>, GapPersistenceError> {
    let query = format!(
        r#"
        UPDATE ingester.data_gaps
        SET status = 'repairing',
            repair_attempts = repair_attempts + 1,
            repair_started_at = now(),
            updated_at = now()
        WHERE gap_id = $1 AND status IN ('open', 'repairing')
        RETURNING {GAP_COLUMNS}
        "#
    );
    sqlx::query_as::<_, GapRow>(&query)
        .bind(gap_id)
        .fetch_optional(connection)
        .await?
        .map(TryInto::try_into)
        .transpose()
}

async fn mark_repaired_on(
    connection: &mut PgConnection,
    gap_id: Uuid,
    repair_artifact_id: Uuid,
    resolution_code: &str,
    resolution_message: Option<&str>,
) -> Result<Option<DataGap>, GapPersistenceError> {
    let resolution_message = validate_resolution(resolution_code, resolution_message)?;
    let query = format!(
        r#"
        UPDATE ingester.data_gaps
        SET repair_artifact_id = $2,
            status = 'repaired',
            resolved_at = now(),
            resolution_code = $3,
            resolution_message = $4,
            updated_at = now()
        WHERE gap_id = $1 AND status = 'repairing'
        RETURNING {GAP_COLUMNS}
        "#
    );
    sqlx::query_as::<_, GapRow>(&query)
        .bind(gap_id)
        .bind(repair_artifact_id)
        .bind(resolution_code)
        .bind(resolution_message)
        .fetch_optional(connection)
        .await?
        .map(TryInto::try_into)
        .transpose()
}

async fn mark_unrecoverable_on(
    connection: &mut PgConnection,
    gap_id: Uuid,
    resolution_code: &str,
    resolution_message: Option<&str>,
) -> Result<Option<DataGap>, GapPersistenceError> {
    let resolution_message = validate_resolution(resolution_code, resolution_message)?;
    let query = format!(
        r#"
        UPDATE ingester.data_gaps
        SET status = 'unrecoverable',
            resolved_at = now(),
            resolution_code = $2,
            resolution_message = $3,
            updated_at = now()
        WHERE gap_id = $1 AND status IN ('open', 'repairing')
        RETURNING {GAP_COLUMNS}
        "#
    );
    sqlx::query_as::<_, GapRow>(&query)
        .bind(gap_id)
        .bind(resolution_code)
        .bind(resolution_message)
        .fetch_optional(connection)
        .await?
        .map(TryInto::try_into)
        .transpose()
}

#[derive(Debug, FromRow)]
struct GapRow {
    gap_id: Uuid,
    gap_fingerprint: String,
    strategy_key: String,
    detected_artifact_id: Option<Uuid>,
    repair_artifact_id: Option<Uuid>,
    gap_kind: String,
    reason_code: String,
    reason_message: Option<String>,
    source_time_start: Option<DateTime<Utc>>,
    source_time_end: Option<DateTime<Utc>>,
    start_cursor: Option<String>,
    end_cursor: Option<String>,
    status: String,
    repair_attempts: i32,
    detected_at: DateTime<Utc>,
    repair_started_at: Option<DateTime<Utc>>,
    resolved_at: Option<DateTime<Utc>>,
    resolution_code: Option<String>,
    resolution_message: Option<String>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

impl TryFrom<GapRow> for DataGap {
    type Error = GapPersistenceError;

    fn try_from(row: GapRow) -> Result<Self, Self::Error> {
        let strategy_key = row.strategy_key.parse().map_err(|error| {
            GapPersistenceError::StoredData(format!(
                "data gap contains an unknown strategy key: {error}"
            ))
        })?;
        let status = GapStatus::from_database(&row.status).ok_or_else(|| {
            GapPersistenceError::StoredData(format!(
                "data gap contains unknown status {}",
                row.status
            ))
        })?;
        Ok(Self {
            gap_id: row.gap_id,
            gap_fingerprint: row.gap_fingerprint,
            strategy_key,
            detected_artifact_id: row.detected_artifact_id,
            repair_artifact_id: row.repair_artifact_id,
            gap_kind: row.gap_kind,
            reason_code: row.reason_code,
            reason_message: row.reason_message,
            source_time_start: row.source_time_start,
            source_time_end: row.source_time_end,
            start_cursor: row.start_cursor,
            end_cursor: row.end_cursor,
            status,
            repair_attempts: row.repair_attempts,
            detected_at: row.detected_at,
            repair_started_at: row.repair_started_at,
            resolved_at: row.resolved_at,
            resolution_code: row.resolution_code,
            resolution_message: row.resolution_message,
            created_at: row.created_at,
            updated_at: row.updated_at,
        })
    }
}

#[derive(Debug, Error)]
pub enum GapPersistenceError {
    #[error("invalid data-gap input: {0}")]
    InvalidInput(String),
    #[error("data-gap database operation failed: {0}")]
    Database(#[from] sqlx::Error),
    #[error("stored data gap is invalid: {0}")]
    StoredData(String),
}

fn validate_time_range(
    start: Option<DateTime<Utc>>,
    end: Option<DateTime<Utc>>,
) -> Result<(), GapPersistenceError> {
    match (start, end) {
        (None, None) => Ok(()),
        (Some(start), Some(end)) if end >= start => Ok(()),
        (Some(_), Some(_)) => Err(GapPersistenceError::InvalidInput(
            "data-gap source-time end precedes its start".to_owned(),
        )),
        _ => Err(GapPersistenceError::InvalidInput(
            "data-gap source-time range must be entirely present or absent".to_owned(),
        )),
    }
}

fn validate_optional_cursor(name: &str, cursor: Option<&str>) -> Result<(), GapPersistenceError> {
    if cursor.is_some_and(|cursor| cursor.len() > MAX_CURSOR_BYTES) {
        return Err(GapPersistenceError::InvalidInput(format!(
            "data-gap {name} exceeds {MAX_CURSOR_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_nonempty_bounded(
    name: &str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), GapPersistenceError> {
    if value.trim().is_empty() || value.len() > maximum_bytes {
        return Err(GapPersistenceError::InvalidInput(format!(
            "data-gap {name} must contain between 1 and {maximum_bytes} bytes"
        )));
    }
    Ok(())
}

fn validate_resolution<'a>(
    code: &str,
    message: Option<&'a str>,
) -> Result<Option<&'a str>, GapPersistenceError> {
    validate_nonempty_bounded("resolution code", code, MAX_REASON_CODE_BYTES)?;
    Ok(message.map(|message| truncate_utf8(message, MAX_MESSAGE_BYTES)))
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

fn gap_fingerprint(gap: &NewDataGap) -> String {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, gap.strategy_key.as_str());
    hash_field(&mut hasher, &gap.gap_kind);
    hash_field(&mut hasher, &gap.reason_code);
    hash_optional_timestamp(&mut hasher, gap.source_time_start);
    hash_optional_timestamp(&mut hasher, gap.source_time_end);
    hash_optional_field(&mut hasher, gap.start_cursor.as_deref());
    hash_optional_field(&mut hasher, gap.end_cursor.as_deref());

    let mut encoded = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn hash_optional_timestamp(hasher: &mut Sha256, timestamp: Option<DateTime<Utc>>) {
    let encoded = timestamp.map(|timestamp| timestamp.to_rfc3339_opts(SecondsFormat::Nanos, true));
    hash_optional_field(hasher, encoded.as_deref());
}

fn hash_optional_field(hasher: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            hash_field(hasher, value);
        }
        None => hasher.update([0]),
    }
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    fn timestamp(seconds: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(seconds, 0).single().expect("valid time")
    }

    fn gap() -> NewDataGap {
        NewDataGap {
            strategy_key: IngesterStrategyKey::BinanceSpotBtcusdtL2Snapshots,
            detected_artifact_id: None,
            gap_kind: "sequence".to_owned(),
            reason_code: "depth_update_jump".to_owned(),
            reason_message: Some("missing depth update".to_owned()),
            source_time_start: Some(timestamp(100)),
            source_time_end: Some(timestamp(101)),
            start_cursor: Some("1000".to_owned()),
            end_cursor: Some("1007".to_owned()),
        }
    }

    #[test]
    fn fingerprint_is_stable_and_excludes_diagnostic_text() {
        let original = gap();
        let expected = original.prepare().expect("valid gap").fingerprint;
        let mut changed_message = original.clone();
        changed_message.reason_message = Some("different diagnostic".to_owned());
        assert_eq!(
            changed_message.prepare().expect("valid gap").fingerprint,
            expected
        );
    }

    #[test]
    fn fingerprint_changes_with_factual_boundary() {
        let original = gap().prepare().expect("valid gap").fingerprint;
        let mut changed = gap();
        changed.end_cursor = Some("1008".to_owned());
        assert_ne!(changed.prepare().expect("valid gap").fingerprint, original);
    }

    #[test]
    fn fingerprint_is_independent_of_capture_lineage() {
        let original = gap().prepare().expect("valid gap").fingerprint;
        let mut different_artifact = gap();
        different_artifact.detected_artifact_id = Some(Uuid::new_v4());
        assert_eq!(
            different_artifact.prepare().expect("valid gap").fingerprint,
            original
        );
    }

    #[test]
    fn gap_requires_a_complete_time_range() {
        let mut invalid = gap();
        invalid.source_time_end = None;
        assert!(invalid.prepare().is_err());
    }

    #[test]
    fn cursor_only_gap_is_valid() {
        let mut cursor_only = gap();
        cursor_only.source_time_start = None;
        cursor_only.source_time_end = None;
        assert!(cursor_only.prepare().is_ok());
    }

    #[test]
    fn bounded_diagnostics_preserve_utf8() {
        assert_eq!(truncate_utf8("ab💡cd", 5), "ab");
    }
}
