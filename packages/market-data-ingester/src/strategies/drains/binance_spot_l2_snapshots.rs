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
use arrow_array::{
    ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures_util::TryStreamExt;
use sqlx::FromRow;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::fs;
use uuid::Uuid;
const SPEC: RetainedDrainSpec = RetainedDrainSpec {
    key: "binance_spot_btcusdt_l2_snapshots",
    relation: "market_data.binance_spot_btcusdt_l2_snapshots",
    schema: "market_data",
    table: "binance_spot_btcusdt_l2_snapshots",
    retention_days: Some(14),
};
const BATCH_ROWS: usize = 2_000;
pub struct BinanceSpotL2SnapshotsDrain {
    descriptor: DrainDescriptor,
    root: PathBuf,
}
impl BinanceSpotL2SnapshotsDrain {
    pub fn from_environment() -> Result<Self, DrainExecutionError> {
        let root = PathBuf::from(
            std::env::var("INGESTER_BINANCE_SPOT_L2_SNAPSHOTS_LAKE_ROOT")
                .unwrap_or_else(|_| "/var/lib/binance-spot-l2-snapshots".into()),
        );
        if !root.is_absolute() {
            return Err(invalid(
                "drain_root_invalid",
                "Binance L2 lake root must be absolute",
            ));
        }
        Ok(Self {
            descriptor: retained::descriptor(&SPEC),
            root,
        })
    }
}
#[derive(FromRow)]
struct Row {
    source_timestamp: DateTime<Utc>,
    received_at: DateTime<Utc>,
    source: String,
    symbol: String,
    source_update_id: i64,
    connection_epoch: Uuid,
    sample_depth: i32,
    bids: serde_json::Value,
    asks: serde_json::Value,
    book_sha256: String,
    sampling_policy: serde_json::Value,
    sampling_policy_sha256: String,
    payload_sha256: String,
    strategy_key: String,
    capture_artifact_id: Uuid,
    ingested_at: DateTime<Utc>,
}
fn ts(n: &'static str) -> Field {
    Field::new(
        n,
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        false,
    )
}
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        ts("source_timestamp"),
        ts("received_at"),
        Field::new("source", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("source_update_id", DataType::Int64, false),
        Field::new("connection_epoch", DataType::Utf8, false),
        Field::new("sample_depth", DataType::Int32, false),
        Field::new("bids", DataType::Utf8, false),
        Field::new("asks", DataType::Utf8, false),
        Field::new("book_sha256", DataType::Utf8, false),
        Field::new("sampling_policy", DataType::Utf8, false),
        Field::new("sampling_policy_sha256", DataType::Utf8, false),
        Field::new("payload_sha256", DataType::Utf8, false),
        Field::new("strategy_key", DataType::Utf8, false),
        Field::new("capture_artifact_id", DataType::Utf8, false),
        ts("ingested_at"),
    ]))
}
fn batch(rows: Vec<Row>) -> Result<RecordBatch, DrainExecutionError> {
    let bids = rows.iter().map(|r| r.bids.to_string()).collect::<Vec<_>>();
    let asks = rows.iter().map(|r| r.asks.to_string()).collect::<Vec<_>>();
    let policies = rows
        .iter()
        .map(|r| r.sampling_policy.to_string())
        .collect::<Vec<_>>();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|r| r.source_timestamp.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|r| r.received_at.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.source.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.symbol.as_str()),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.source_update_id),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.connection_epoch.to_string()),
        )),
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|r| r.sample_depth),
        )),
        Arc::new(StringArray::from_iter_values(
            bids.iter().map(String::as_str),
        )),
        Arc::new(StringArray::from_iter_values(
            asks.iter().map(String::as_str),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.book_sha256.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            policies.iter().map(String::as_str),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.sampling_policy_sha256.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.payload_sha256.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.strategy_key.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.capture_artifact_id.to_string()),
        )),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|r| r.ingested_at.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
    ];
    RecordBatch::try_new(schema(), arrays).map_err(|e| io_error(e.to_string()))
}
#[async_trait]
impl RetainedDrainAdapter for BinanceSpotL2SnapshotsDrain {
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
        let mut stream=sqlx::query_as::<_,Row>("SELECT source_timestamp,received_at,source,symbol,source_update_id,connection_epoch,sample_depth,bids,asks,book_sha256,sampling_policy,sampling_policy_sha256,payload_sha256,strategy_key,capture_artifact_id,ingested_at FROM market_data.binance_spot_btcusdt_l2_snapshots WHERE source_timestamp >= $1 AND source_timestamp < $2 ORDER BY source_timestamp,source_update_id,connection_epoch").bind(ch.range_start).bind(ch.range_end).fetch(&c.pool);
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
impl DrainWorkerStrategy for BinanceSpotL2SnapshotsDrain {
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
#[path = "tests/binance_spot_l2_snapshots.rs"]
mod tests;
