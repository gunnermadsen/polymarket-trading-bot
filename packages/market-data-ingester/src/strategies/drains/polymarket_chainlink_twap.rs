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
    ArrayRef, Decimal128Array, Int16Array, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures_util::TryStreamExt;
use rust_decimal::Decimal;
use sqlx::FromRow;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::fs;
use uuid::Uuid;
const SPEC: RetainedDrainSpec = RetainedDrainSpec {
    key: "polymarket_chainlink_btcusd_twap",
    relation: "market_data.polymarket_chainlink_btcusd_twap",
    schema: "market_data",
    table: "polymarket_chainlink_btcusd_twap",
    retention_days: Some(14),
};
const BATCH_ROWS: usize = 10_000;
pub struct PolymarketChainlinkTwapDrain {
    descriptor: DrainDescriptor,
    root: PathBuf,
}
impl PolymarketChainlinkTwapDrain {
    pub fn from_environment() -> Result<Self, DrainExecutionError> {
        let root = PathBuf::from(
            std::env::var("INGESTER_POLYMARKET_CHAINLINK_TWAP_LAKE_ROOT")
                .unwrap_or_else(|_| "/var/lib/polymarket-chainlink-twap".into()),
        );
        if !root.is_absolute() {
            return Err(invalid(
                "drain_root_invalid",
                "Polymarket Chainlink TWAP lake root must be absolute",
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
    published_at: DateTime<Utc>,
    received_at: DateTime<Utc>,
    source: String,
    symbol: String,
    window_seconds: i16,
    twap_price: Decimal,
    full_accuracy_value: String,
    source_payload: serde_json::Value,
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
        ts("published_at"),
        ts("received_at"),
        Field::new("source", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("window_seconds", DataType::Int16, false),
        Field::new("twap_price", DataType::Decimal128(38, 18), false),
        Field::new("full_accuracy_value", DataType::Utf8, false),
        Field::new("source_payload", DataType::Utf8, false),
        Field::new("payload_sha256", DataType::Utf8, false),
        Field::new("strategy_key", DataType::Utf8, false),
        Field::new("capture_artifact_id", DataType::Utf8, false),
        ts("ingested_at"),
    ]))
}
fn decimal(mut v: Decimal) -> i128 {
    v.rescale(18);
    v.mantissa()
}
fn batch(rows: Vec<Row>) -> Result<RecordBatch, DrainExecutionError> {
    let payloads = rows
        .iter()
        .map(|r| r.source_payload.to_string())
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
                rows.iter().map(|r| r.published_at.timestamp_micros()),
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
        Arc::new(Int16Array::from_iter_values(
            rows.iter().map(|r| r.window_seconds),
        )),
        Arc::new(
            Decimal128Array::from_iter_values(rows.iter().map(|r| decimal(r.twap_price)))
                .with_precision_and_scale(38, 18)
                .map_err(|e| io_error(e.to_string()))?,
        ),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.full_accuracy_value.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            payloads.iter().map(String::as_str),
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
impl RetainedDrainAdapter for PolymarketChainlinkTwapDrain {
    fn spec(&self) -> &RetainedDrainSpec {
        &SPEC
    }
    fn root(&self) -> &Path {
        &self.root
    }
    async fn export_chunk(
        &self,
        context: &DrainContext,
        chunk: &Chunk,
    ) -> Result<Publication, DrainExecutionError> {
        let id = create_object(context, SPEC.key, SPEC.relation, chunk).await?;
        let staging = self.root.join(".staging").join(format!("{id}.parquet.tmp"));
        let _ = fs::remove_file(&staging).await;
        let (sender, writer) = start_writer(staging.clone(), schema());
        let mut stream=sqlx::query_as::<_,Row>("SELECT source_timestamp,published_at,received_at,source,symbol,window_seconds,twap_price,full_accuracy_value,source_payload,payload_sha256::text,strategy_key,capture_artifact_id,ingested_at FROM market_data.polymarket_chainlink_btcusd_twap WHERE source_timestamp >= $1 AND source_timestamp < $2 ORDER BY source_timestamp,source,symbol,window_seconds").bind(chunk.range_start).bind(chunk.range_end).fetch(&context.pool);
        let mut rows = Vec::with_capacity(BATCH_ROWS);
        let mut count = 0i64;
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
        publish(context, &self.root, chunk, id, staging, count).await
    }
}
async fn publish(
    context: &DrainContext,
    root: &Path,
    chunk: &Chunk,
    id: Uuid,
    staging: PathBuf,
    count: i64,
) -> Result<Publication, DrainExecutionError> {
    let relative = format!(
        "verified-chunks-v1/year={}/month={}/day={}/{id}.parquet",
        chunk.range_start.format("%Y"),
        chunk.range_start.format("%m"),
        chunk.range_start.format("%d")
    );
    let (sha, size) = publish_file(&staging, root, &relative, count).await?;
    sqlx::query_as("UPDATE ingester.drain_objects SET row_count=$2,relative_path=$3,sha256=$4,byte_size=$5,status='published',published_at=clock_timestamp(),updated_at=clock_timestamp() WHERE object_id=$1 AND status='staging' RETURNING object_id,row_count,relative_path,sha256::text,byte_size,status").bind(id).bind(count).bind(relative).bind(sha).bind(size).fetch_one(&context.pool).await.map_err(db_error)
}
#[async_trait]
impl DrainWorkerStrategy for PolymarketChainlinkTwapDrain {
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
#[path = "tests/polymarket_chainlink_twap.rs"]
mod tests;
