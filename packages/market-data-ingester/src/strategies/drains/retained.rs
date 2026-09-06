use std::{path::Path, sync::Arc};

use async_trait::async_trait;
use chrono::{Duration, Utc};
use serde_json::json;
use tokio::fs;

use crate::domain::{
    DrainContext, DrainDescriptor, DrainExecutionError, DrainOutcome, DrainRequest,
};

use super::common::{
    archived_publications, db_error, existing_publication, invalid, verify_existing, Chunk,
    Publication,
};

pub struct RetainedDrainSpec {
    pub key: &'static str,
    pub relation: &'static str,
    pub schema: &'static str,
    pub table: &'static str,
    pub retention_days: Option<i64>,
}

#[async_trait]
pub trait RetainedDrainAdapter: Send + Sync {
    fn spec(&self) -> &RetainedDrainSpec;
    fn root(&self) -> &Path;
    async fn export_chunk(
        &self,
        context: &DrainContext,
        chunk: &Chunk,
    ) -> Result<Publication, DrainExecutionError>;
}

pub fn descriptor(spec: &RetainedDrainSpec) -> DrainDescriptor {
    DrainDescriptor {
        strategy_key: Arc::from(spec.key),
        relation: Arc::from(spec.relation),
        contract_version: 1,
    }
}

pub fn validate(
    adapter: &dyn RetainedDrainAdapter,
    request: &DrainRequest,
) -> Result<(), DrainExecutionError> {
    let spec = adapter.spec();
    if request.strategy_key != spec.key {
        return Err(invalid(
            "drain_strategy_mismatch",
            format!("request does not target the {} drain", spec.key),
        ));
    }
    request
        .execution
        .validate()
        .map_err(|error| invalid("drain_execution_selector_invalid", error.to_string()))?;
    if let Some(days) = spec.retention_days {
        if request.cutoff > Utc::now() - Duration::days(days) {
            return Err(invalid(
                "drain_retention_violation",
                format!("cutoff must retain at least {days} days of data"),
            ));
        }
    }
    Ok(())
}

pub async fn execute(
    adapter: &dyn RetainedDrainAdapter,
    context: DrainContext,
    request: DrainRequest,
) -> Result<DrainOutcome, DrainExecutionError> {
    validate(adapter, &request)?;
    let spec = adapter.spec();
    let snapshot_at = sqlx::query_scalar::<_, chrono::DateTime<Utc>>(
        "SELECT requested_at FROM ingester.drain_jobs WHERE job_id=$1",
    )
    .bind(context.job_id)
    .fetch_one(&context.pool)
    .await
    .map_err(db_error)?;
    let chunks = sqlx::query_as::<_, Chunk>(
        "SELECT chunk_schema,chunk_name,range_start,range_end \
         FROM timescaledb_information.chunks WHERE hypertable_schema=$1 \
         AND hypertable_name=$2 ORDER BY range_start,chunk_name",
    )
    .bind(spec.schema)
    .bind(spec.table)
    .fetch_all(&context.pool)
    .await
    .map_err(db_error)?;
    let copyable: Vec<_> = chunks
        .iter()
        .filter(|chunk| chunk.range_end <= snapshot_at)
        .cloned()
        .collect();
    let eligible = copyable
        .iter()
        .filter(|chunk| chunk.range_end <= request.cutoff)
        .count();
    let removable = if request.mode.removes_source_data() {
        eligible
    } else {
        0
    };
    if request.dry_run {
        return Ok(DrainOutcome {
            rows_exported: 0,
            rows_removed: 0,
            objects_published: 0,
            bytes_written: 0,
            summary: json!({
                "copyable_closed_chunks": copyable.len(),
                "cutoff": request.cutoff,
                "eligible_chunks": eligible,
                "mode": request.mode,
                "open_chunks": chunks.len() - copyable.len(),
                "relation": spec.relation,
                "retained_chunks": chunks.len() - removable,
                "retention_days": spec.retention_days,
                "snapshot_at": snapshot_at,
            }),
        });
    }
    fs::create_dir_all(adapter.root().join(".staging"))
        .await
        .map_err(super::common::io_error)?;
    let mut outcome = DrainOutcome {
        rows_exported: 0,
        rows_removed: 0,
        objects_published: 0,
        bytes_written: 0,
        summary: json!({}),
    };
    let archived = archived_publications(&context, spec.key).await?;
    for item in &archived {
        verify_existing(adapter.root(), &item.publication()).await?;
    }
    let mut verified_existing_objects = archived.len();
    let mut objects_created = 0usize;
    for chunk in copyable {
        if context.shutdown.is_cancelled() {
            return Err(DrainExecutionError::new(
                "drain_cancelled",
                "drain was cancelled",
                true,
            ));
        }
        let publication = match existing_publication(&context, spec.key, &chunk).await? {
            Some(publication) => {
                if !archived
                    .iter()
                    .any(|item| item.object_id == publication.object_id)
                {
                    verify_existing(adapter.root(), &publication).await?;
                    verified_existing_objects += 1;
                }
                publication
            }
            None => {
                objects_created += 1;
                adapter.export_chunk(&context, &chunk).await?
            }
        };
        outcome.rows_exported += publication.row_count;
        outcome.objects_published += 1;
        outcome.bytes_written += publication.byte_size;
        if request.mode.removes_source_data() && chunk.range_end <= request.cutoff {
            outcome.rows_removed +=
                sqlx::query_scalar::<_, i64>("SELECT ingester.remove_verified_drain_chunk($1,$2)")
                    .bind(publication.object_id)
                    .bind(&publication.sha256)
                    .fetch_one(&context.pool)
                    .await
                    .map_err(db_error)?;
        }
    }
    outcome.summary = json!({
        "cutoff": request.cutoff,
        "database_retained_from": chunks.iter().filter(|chunk| !request.mode.removes_source_data() || chunk.range_end > request.cutoff).map(|chunk| chunk.range_start).min(),
        "live_tail_from": chunks.iter().filter(|chunk| chunk.range_end > snapshot_at).map(|chunk| chunk.range_start).min(),
        "mode": request.mode,
        "objects_created": objects_created,
        "relation": spec.relation,
        "retention_days": spec.retention_days,
        "snapshot_at": snapshot_at,
        "ssd_complete_from": archived.iter().map(|item| item.source_start).min().or_else(|| chunks.iter().filter(|chunk| chunk.range_end <= snapshot_at).map(|chunk| chunk.range_start).min()),
        "ssd_complete_through": chunks.iter().filter(|chunk| chunk.range_end <= snapshot_at).map(|chunk| chunk.range_end).max().or_else(|| archived.iter().map(|item| item.source_end).max()),
        "verified_existing_objects": verified_existing_objects,
    });
    Ok(outcome)
}

pub async fn execute_strategy(
    adapter: &dyn RetainedDrainAdapter,
    context: DrainContext,
    request: DrainRequest,
) -> Result<DrainOutcome, DrainExecutionError> {
    execute(adapter, context, request).await
}

pub fn validate_strategy(
    adapter: &dyn RetainedDrainAdapter,
    request: &DrainRequest,
) -> Result<(), DrainExecutionError> {
    validate(adapter, request)
}
