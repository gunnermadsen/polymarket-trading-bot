use super::{
    common::{
        create_object, db_error, finish_writer, invalid, io_error, publish_file, start_writer,
        Chunk, Publication,
    },
    retained::{self, RetainedDrainAdapter, RetainedDrainSpec},
};
use crate::domain::{
    DrainContext, DrainDescriptor, DrainExecutionError, DrainOutcome, DrainRequest,
    DrainWorkerStrategy,
};
use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use futures_util::TryStreamExt;
use sqlx::{postgres::PgRow, Row};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::fs;
use uuid::Uuid;
const SPEC: RetainedDrainSpec = RetainedDrainSpec {
    key: "polymarket_btc_capacity_execution_snapshots",
    relation: "polymarket.btc_market_capacity_execution_snapshots",
    schema: "polymarket",
    table: "btc_market_capacity_execution_snapshots",
    retention_days: None,
};
const BATCH_ROWS: usize = 2_000;
const BACKFILL_KEY: &str = "polymarket_btc_five_minute_execution_snapshots_backfill";
const COLUMNS: [&str; 56] = [
    "market_id",
    "sampled_at",
    "artifact_id",
    "schema_version",
    "up_source_row_number",
    "up_source_timestamp",
    "up_provider_received_at",
    "up_best_bid",
    "up_best_ask",
    "up_best_bid_size",
    "up_best_ask_size",
    "up_bid_depth",
    "up_ask_depth",
    "up_ask_vwap_1",
    "up_ask_vwap_5",
    "up_ask_vwap_10",
    "up_imbalance",
    "down_source_row_number",
    "down_source_timestamp",
    "down_provider_received_at",
    "down_best_bid",
    "down_best_ask",
    "down_best_bid_size",
    "down_best_ask_size",
    "down_bid_depth",
    "down_ask_depth",
    "down_ask_vwap_1",
    "down_ask_vwap_5",
    "down_ask_vwap_10",
    "down_imbalance",
    "quality_flags",
    "created_at",
    "up_ask_vwap_15",
    "up_ask_vwap_20",
    "up_ask_vwap_25",
    "up_ask_vwap_30",
    "up_ask_vwap_40",
    "up_ask_vwap_50",
    "up_ask_vwap_75",
    "up_ask_vwap_100",
    "up_ask_vwap_125",
    "up_ask_vwap_150",
    "up_ask_vwap_175",
    "up_ask_vwap_200",
    "down_ask_vwap_15",
    "down_ask_vwap_20",
    "down_ask_vwap_25",
    "down_ask_vwap_30",
    "down_ask_vwap_40",
    "down_ask_vwap_50",
    "down_ask_vwap_75",
    "down_ask_vwap_100",
    "down_ask_vwap_125",
    "down_ask_vwap_150",
    "down_ask_vwap_175",
    "down_ask_vwap_200",
];
pub struct BtcCapacityExecutionSnapshotsDrain {
    descriptor: DrainDescriptor,
    root: PathBuf,
}
impl BtcCapacityExecutionSnapshotsDrain {
    pub fn from_environment() -> Result<Self, DrainExecutionError> {
        let root = PathBuf::from(
            std::env::var("INGESTER_BTC_CAPACITY_EXECUTION_SNAPSHOTS_LAKE_ROOT")
                .unwrap_or_else(|_| "/var/lib/btc-capacity-execution-snapshots".into()),
        );
        if !root.is_absolute() {
            return Err(invalid(
                "drain_root_invalid",
                "BTC capacity snapshot lake root must be absolute",
            ));
        }
        Ok(Self {
            descriptor: retained::descriptor(&SPEC),
            root,
        })
    }
}
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(
        COLUMNS
            .iter()
            .map(|name| Field::new(*name, DataType::Utf8, true))
            .collect::<Vec<_>>(),
    ))
}
fn batch(rows: Vec<PgRow>) -> Result<RecordBatch, DrainExecutionError> {
    let arrays = COLUMNS
        .iter()
        .enumerate()
        .map(|(i, _)| -> Result<ArrayRef, DrainExecutionError> {
            let values = rows
                .iter()
                .map(|r| r.try_get::<Option<String>, _>(i).map_err(db_error))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Arc::new(StringArray::from(values)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(schema(), arrays).map_err(|e| io_error(e.to_string()))
}
#[async_trait]
impl RetainedDrainAdapter for BtcCapacityExecutionSnapshotsDrain {
    fn spec(&self) -> &RetainedDrainSpec {
        &SPEC
    }
    fn root(&self) -> &Path {
        &self.root
    }
    async fn export_chunk(
        &self,
        c: &DrainContext,
        ch: &Chunk,
    ) -> Result<Publication, DrainExecutionError> {
        let id = create_object(c, SPEC.key, SPEC.relation, ch).await?;
        let staging = self.root.join(".staging").join(format!("{id}.parquet.tmp"));
        let _ = fs::remove_file(&staging).await;
        let (sender, writer) = start_writer(staging.clone(), schema());
        let mut stream=sqlx::query("SELECT market_id,sampled_at::text,artifact_id::text,schema_version,up_source_row_number::text,up_source_timestamp::text,up_provider_received_at::text,up_best_bid::text,up_best_ask::text,up_best_bid_size::text,up_best_ask_size::text,up_bid_depth::text,up_ask_depth::text,up_ask_vwap_1::text,up_ask_vwap_5::text,up_ask_vwap_10::text,up_imbalance::text,down_source_row_number::text,down_source_timestamp::text,down_provider_received_at::text,down_best_bid::text,down_best_ask::text,down_best_bid_size::text,down_best_ask_size::text,down_bid_depth::text,down_ask_depth::text,down_ask_vwap_1::text,down_ask_vwap_5::text,down_ask_vwap_10::text,down_imbalance::text,quality_flags::text,created_at::text,up_ask_vwap_15::text,up_ask_vwap_20::text,up_ask_vwap_25::text,up_ask_vwap_30::text,up_ask_vwap_40::text,up_ask_vwap_50::text,up_ask_vwap_75::text,up_ask_vwap_100::text,up_ask_vwap_125::text,up_ask_vwap_150::text,up_ask_vwap_175::text,up_ask_vwap_200::text,down_ask_vwap_15::text,down_ask_vwap_20::text,down_ask_vwap_25::text,down_ask_vwap_30::text,down_ask_vwap_40::text,down_ask_vwap_50::text,down_ask_vwap_75::text,down_ask_vwap_100::text,down_ask_vwap_125::text,down_ask_vwap_150::text,down_ask_vwap_175::text,down_ask_vwap_200::text FROM polymarket.btc_market_capacity_execution_snapshots WHERE sampled_at >= $1 AND sampled_at < $2 ORDER BY sampled_at,market_id").bind(ch.range_start).bind(ch.range_end).fetch(&c.pool);
        let mut rows = Vec::with_capacity(BATCH_ROWS);
        let mut count = 0;
        while let Some(row) = stream.try_next().await.map_err(db_error)? {
            rows.push(row);
            count += 1;
            if rows.len() == BATCH_ROWS {
                sender
                    .send(batch(std::mem::take(&mut rows))?)
                    .await
                    .map_err(|_| io_error("Parquet writer stopped"))?;
                tokio::time::sleep(std::time::Duration::from_millis(15)).await;
            }
        }
        if !rows.is_empty() {
            sender
                .send(batch(rows)?)
                .await
                .map_err(|_| io_error("Parquet writer stopped"))?;
        }
        finish_writer(sender, writer).await?;
        publish(c, &self.root, ch, id, staging, count).await
    }
}
async fn publish(
    c: &DrainContext,
    root: &Path,
    ch: &Chunk,
    id: Uuid,
    staging: PathBuf,
    count: i64,
) -> Result<Publication, DrainExecutionError> {
    let rel = format!(
        "verified-chunks-v1/year={}/month={}/day={}/{id}.parquet",
        ch.range_start.format("%Y"),
        ch.range_start.format("%m"),
        ch.range_start.format("%d")
    );
    let (sha, size) = publish_file(&staging, root, &rel, count).await?;
    sqlx::query_as("UPDATE ingester.drain_objects SET row_count=$2,relative_path=$3,sha256=$4,byte_size=$5,status='published',published_at=clock_timestamp(),updated_at=clock_timestamp() WHERE object_id=$1 AND status='staging' RETURNING object_id,row_count,relative_path,sha256::text,byte_size,status").bind(id).bind(count).bind(rel).bind(sha).bind(size).fetch_one(&c.pool).await.map_err(db_error)
}
#[async_trait]
impl DrainWorkerStrategy for BtcCapacityExecutionSnapshotsDrain {
    fn descriptor(&self) -> &DrainDescriptor {
        &self.descriptor
    }
    fn validate_request(&self, r: &DrainRequest) -> Result<(), DrainExecutionError> {
        retained::validate_strategy(self, r)
    }
    async fn execute_drain(
        &self,
        c: DrainContext,
        r: DrainRequest,
    ) -> Result<DrainOutcome, DrainExecutionError> {
        if !r.dry_run {
            let active=sqlx::query_scalar::<_,i64>("SELECT count(*)::bigint FROM ingester.backfill_jobs WHERE strategy_key=$1 AND status IN ('queued','running')").bind(BACKFILL_KEY).fetch_one(&c.pool).await.map_err(db_error)?;
            if active > 0 {
                return Err(DrainExecutionError::new(
                    "drain_backfill_active",
                    "capacity snapshot drain is blocked while its backfill is active",
                    true,
                ));
            }
        }
        retained::execute_strategy(self, c, r).await
    }
}
#[cfg(test)]
#[path = "tests/btc_capacity_execution_snapshots.rs"]
mod tests;
