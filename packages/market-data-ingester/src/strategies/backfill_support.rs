use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use sqlx::{Postgres, Row, Transaction};
use uuid::Uuid;

use crate::domain::{
    BackfillContext, BackfillExecutionError, BackfillFailureKind, BackfillOutcome, BackfillRequest,
    BackfillShard, StrategyCapability, StrategyDescriptor, ValidatedBackfillRequest,
};

pub const CONTRACT_VERSION: i32 = 1;
pub const REQUEST_VERSION: i32 = 1;
pub const MAX_DAILY_SHARDS: usize = 3_650;

pub fn descriptor(
    key: &'static str,
    name: &'static str,
    description: &'static str,
) -> Result<StrategyDescriptor, BackfillExecutionError> {
    let descriptor = StrategyDescriptor {
        strategy_key: key.into(),
        name: name.into(),
        description: description.into(),
        capabilities: vec![StrategyCapability::Backfill],
        strategy_contract_version: CONTRACT_VERSION,
        request_schema_version: Some(REQUEST_VERSION),
        shardable: true,
        maximum_shards: MAX_DAILY_SHARDS,
    };
    descriptor.validate()?;
    Ok(descriptor)
}

pub fn validate_empty_request(
    descriptor: &StrategyDescriptor,
    request: &BackfillRequest,
) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
    if request.strategy_key != descriptor.strategy_key.as_ref() {
        return Err(BackfillExecutionError::invalid(
            "strategy_key_mismatch",
            "request strategy key does not match the backfill strategy",
        ));
    }
    if request.range.end <= request.range.start || request.range.end > Utc::now() {
        return Err(BackfillExecutionError::invalid(
            "range_invalid",
            "range must be increasing and may not end in the future",
        ));
    }
    if !request
        .parameters
        .as_object()
        .is_some_and(|value| value.is_empty())
    {
        return Err(BackfillExecutionError::invalid(
            "parameters_invalid",
            "this backfill strategy accepts no parameters",
        ));
    }
    request.execution.validate()?;
    Ok(ValidatedBackfillRequest {
        strategy_key: descriptor.strategy_key.clone(),
        strategy_contract_version: CONTRACT_VERSION,
        request_schema_version: REQUEST_VERSION,
        range_start: request.range.start,
        range_end: request.range.end,
        parameters: request.parameters.clone(),
        execution: request.execution.clone(),
    })
}

pub fn daily_shards(
    request: &ValidatedBackfillRequest,
    maximum: usize,
) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
    let mut cursor = request.range_start;
    let mut shards = Vec::new();
    while cursor < request.range_end {
        let next_midnight = cursor
            .date_naive()
            .succ_opt()
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .map(|value| value.and_utc())
            .ok_or_else(|| {
                BackfillExecutionError::invalid("range_overflow", "backfill shard range overflowed")
            })?;
        let end = next_midnight.min(request.range_end);
        shards.push(BackfillShard {
            shard_key: format!(
                "{}-{}",
                cursor.format("%Y%m%dT%H%M%SZ"),
                end.format("%Y%m%dT%H%M%SZ")
            ),
            range_start: cursor,
            range_end: end,
            parameters: json!({}),
        });
        if shards.len() > maximum {
            return Err(BackfillExecutionError::invalid(
                "too_many_shards",
                format!("request exceeds {maximum} daily shards"),
            ));
        }
        cursor = end;
    }
    Ok(shards)
}

pub async fn completed_outcome(
    context: &BackfillContext,
    strategy_key: &str,
    logical_key: &str,
) -> Result<Option<BackfillOutcome>, BackfillExecutionError> {
    let row = sqlx::query("SELECT record_count,minimum_source_timestamp,maximum_source_timestamp,metadata FROM ingester.backfill_artifacts WHERE strategy_key=$1 AND logical_key=$2 AND status='completed'")
        .bind(strategy_key).bind(logical_key).fetch_optional(&context.pool).await.map_err(database_error)?;
    Ok(row.map(|row| {
        let records = row.try_get::<Option<i64>, _>("record_count").ok().flatten().unwrap_or(0);
        BackfillOutcome { records_verified: records, verified_coverage: json!({
            "minimum_source_timestamp": row.try_get::<Option<DateTime<Utc>>, _>("minimum_source_timestamp").ok().flatten(),
            "maximum_source_timestamp": row.try_get::<Option<DateTime<Utc>>, _>("maximum_source_timestamp").ok().flatten(),
            "records_verified": records, "reused_artifact": true,
        }), summary: row.try_get::<Value, _>("metadata").unwrap_or_else(|_| json!({})) }
    }))
}

pub async fn create_artifact(
    context: &BackfillContext,
    strategy_key: &str,
    artifact_ingester_key: &str,
    logical_key: &str,
    provider: &str,
    source_uri: &str,
    durable_target: &str,
) -> Result<Uuid, BackfillExecutionError> {
    let mut tx = context.pool.begin().await.map_err(database_error)?;
    require_lease(&mut tx, context).await?;
    let artifact_id = sqlx::query_scalar(
        "INSERT INTO ingester.backfill_artifacts (job_id,strategy_key,ingester_key,logical_key,provider,source_uri,durable_target,status,metadata) VALUES ($1,$2,$3,$4,$5,$6,$7,'ingesting','{}'::jsonb) ON CONFLICT (strategy_key,logical_key) DO UPDATE SET job_id=EXCLUDED.job_id,status='ingesting',completed_at=NULL RETURNING artifact_id",
    ).bind(context.job_id).bind(strategy_key).bind(artifact_ingester_key).bind(logical_key).bind(provider).bind(source_uri).bind(durable_target)
      .fetch_one(&mut *tx).await.map_err(database_error)?;
    tx.commit().await.map_err(database_error)?;
    Ok(artifact_id)
}

pub struct ArtifactCompletion<'a> {
    pub artifact_id: Uuid,
    pub checksum: &'a str,
    pub byte_size: u64,
    pub record_count: i64,
    pub minimum: Option<DateTime<Utc>>,
    pub maximum: Option<DateTime<Utc>>,
    pub metadata: Value,
}

pub async fn complete_artifact(
    tx: &mut Transaction<'_, Postgres>,
    context: &BackfillContext,
    completion: ArtifactCompletion<'_>,
) -> Result<(), BackfillExecutionError> {
    require_lease(tx, context).await?;
    let updated = sqlx::query("UPDATE ingester.backfill_artifacts SET checksum=$2,byte_size=$3,record_count=$4,minimum_source_timestamp=$5,maximum_source_timestamp=$6,status='completed',metadata=$7,completed_at=now() WHERE artifact_id=$1 AND job_id=$8")
        .bind(completion.artifact_id).bind(completion.checksum)
        .bind(i64::try_from(completion.byte_size).map_err(|_| integrity("artifact byte size overflow"))?)
        .bind(completion.record_count).bind(completion.minimum).bind(completion.maximum)
        .bind(completion.metadata).bind(context.job_id).execute(&mut **tx).await.map_err(database_error)?;
    if updated.rows_affected() != 1 {
        return Err(integrity("backfill artifact ownership changed"));
    }
    Ok(())
}

pub async fn require_lease(
    tx: &mut Transaction<'_, Postgres>,
    context: &BackfillContext,
) -> Result<(), BackfillExecutionError> {
    let valid: bool = sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM ingester.backfill_jobs WHERE job_id=$1 AND assigned_worker_id=$2 AND lease_token=$3 AND status='running' AND lease_expires_at>now())")
        .bind(context.job_id).bind(context.worker_id.as_ref()).bind(context.lease_token)
        .fetch_one(&mut **tx).await.map_err(database_error)?;
    if !valid {
        return Err(BackfillExecutionError::new(
            BackfillFailureKind::LeaseLost,
            "lease_lost",
            "backfill job lease was lost",
        ));
    }
    Ok(())
}

pub fn outcome(records: i64, shard: &BackfillShard, summary: Value) -> BackfillOutcome {
    BackfillOutcome {
        records_verified: records,
        verified_coverage: json!({
            "requested_start": shard.range_start, "requested_end": shard.range_end, "records_verified": records,
        }),
        summary,
    }
}

pub fn database_error(error: sqlx::Error) -> BackfillExecutionError {
    BackfillExecutionError::new(
        BackfillFailureKind::TransientDatabase,
        "backfill_database",
        error.to_string(),
    )
}

pub fn source_error(error: impl std::fmt::Display) -> BackfillExecutionError {
    BackfillExecutionError::new(
        BackfillFailureKind::TransientSource,
        "backfill_source",
        error.to_string(),
    )
}

pub fn integrity(message: impl Into<String>) -> BackfillExecutionError {
    BackfillExecutionError::new(
        BackfillFailureKind::Integrity,
        "backfill_integrity",
        message,
    )
}

#[cfg(test)]
pub fn request(key: &str, start: DateTime<Utc>, end: DateTime<Utc>) -> BackfillRequest {
    BackfillRequest {
        strategy_key: key.to_owned(),
        range: crate::domain::BackfillRange { start, end },
        parameters: json!({}),
        execution: crate::domain::ExecutionSelector::default(),
    }
}
