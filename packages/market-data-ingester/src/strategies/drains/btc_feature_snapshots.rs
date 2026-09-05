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
    key: "polymarket_btc_feature_snapshots",
    relation: "polymarket.btc_feature_snapshots",
    schema: "polymarket",
    table: "btc_feature_snapshots",
    retention_days: Some(14),
};
const BATCH_ROWS: usize = 2_000;
const COLUMNS: [&str; 39] = [
    "snapshot_id",
    "feature_as_of",
    "received_at",
    "market_id",
    "window_start",
    "window_end",
    "feature_schema_version",
    "feature_hash",
    "chainlink_price",
    "chainlink_open_price",
    "binance_price",
    "seconds_to_close",
    "chainlink_gap_bps",
    "binance_return_1s_bps",
    "binance_return_5s_bps",
    "binance_return_30s_bps",
    "realized_vol_30s_bps",
    "basis_bps",
    "up_best_bid",
    "up_best_ask",
    "down_best_bid",
    "down_best_ask",
    "up_depth_ask",
    "down_depth_ask",
    "up_imbalance",
    "down_imbalance",
    "chainlink_age_ms",
    "binance_age_ms",
    "book_age_ms",
    "source_skew_ms",
    "fair_up_probability",
    "fair_up_lower",
    "fair_up_upper",
    "deterministic_logit",
    "readiness_status",
    "quality_flags",
    "features",
    "lineage",
    "created_at",
];
pub struct BtcFeatureSnapshotsDrain {
    descriptor: DrainDescriptor,
    root: PathBuf,
}
impl BtcFeatureSnapshotsDrain {
    pub fn from_environment() -> Result<Self, DrainExecutionError> {
        let root = PathBuf::from(
            std::env::var("INGESTER_BTC_FEATURE_SNAPSHOTS_LAKE_ROOT")
                .unwrap_or_else(|_| "/var/lib/btc-feature-snapshots".into()),
        );
        if !root.is_absolute() {
            return Err(invalid(
                "drain_root_invalid",
                "BTC feature snapshot lake root must be absolute",
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
        .map(|(index, _)| -> Result<ArrayRef, DrainExecutionError> {
            let values = rows
                .iter()
                .map(|row| row.try_get::<Option<String>, _>(index).map_err(db_error))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Arc::new(StringArray::from(values)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(schema(), arrays).map_err(|e| io_error(e.to_string()))
}
#[async_trait]
impl RetainedDrainAdapter for BtcFeatureSnapshotsDrain {
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
        let mut stream=sqlx::query("SELECT snapshot_id::text,feature_as_of::text,received_at::text,market_id,window_start::text,window_end::text,feature_schema_version,feature_hash,chainlink_price::text,chainlink_open_price::text,binance_price::text,seconds_to_close::text,chainlink_gap_bps::text,binance_return_1s_bps::text,binance_return_5s_bps::text,binance_return_30s_bps::text,realized_vol_30s_bps::text,basis_bps::text,up_best_bid::text,up_best_ask::text,down_best_bid::text,down_best_ask::text,up_depth_ask::text,down_depth_ask::text,up_imbalance::text,down_imbalance::text,chainlink_age_ms::text,binance_age_ms::text,book_age_ms::text,source_skew_ms::text,fair_up_probability::text,fair_up_lower::text,fair_up_upper::text,deterministic_logit::text,readiness_status,quality_flags::text,features::text,lineage::text,created_at::text FROM polymarket.btc_feature_snapshots WHERE feature_as_of >= $1 AND feature_as_of < $2 ORDER BY feature_as_of,snapshot_id").bind(ch.range_start).bind(ch.range_end).fetch(&c.pool);
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
impl DrainWorkerStrategy for BtcFeatureSnapshotsDrain {
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
        retained::execute_strategy(self, c, r).await
    }
}
#[cfg(test)]
#[path = "tests/btc_feature_snapshots.rs"]
mod tests;
