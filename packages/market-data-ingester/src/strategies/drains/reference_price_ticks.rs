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
    ArrayRef, Decimal128Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
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
    key: "polymarket_reference_price_ticks",
    relation: "polymarket.reference_price_ticks",
    schema: "polymarket",
    table: "reference_price_ticks",
    retention_days: Some(14),
};
const BATCH_ROWS: usize = 5_000;
pub struct ReferencePriceTicksDrain {
    descriptor: DrainDescriptor,
    root: PathBuf,
}
impl ReferencePriceTicksDrain {
    pub fn from_environment() -> Result<Self, DrainExecutionError> {
        let root = PathBuf::from(
            std::env::var("INGESTER_REFERENCE_PRICE_TICKS_LAKE_ROOT")
                .unwrap_or_else(|_| "/var/lib/reference-price-ticks".into()),
        );
        if !root.is_absolute() {
            return Err(invalid(
                "drain_root_invalid",
                "reference-price tick lake root must be absolute",
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
    tick_id: Uuid,
    source_timestamp: DateTime<Utc>,
    received_at: DateTime<Utc>,
    persisted_at: DateTime<Utc>,
    source: String,
    symbol: String,
    price: Decimal,
    envelope_timestamp: Option<DateTime<Utc>>,
    connection_id: Uuid,
    ingest_sequence: i64,
    source_event_id: Option<String>,
    dedup_key: String,
    clock_skew_ms: i64,
    integrity_status: String,
    raw_payload: serde_json::Value,
}
fn ts(n: &'static str, o: bool) -> Field {
    Field::new(
        n,
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        o,
    )
}
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("tick_id", DataType::Utf8, false),
        ts("source_timestamp", false),
        ts("received_at", false),
        ts("persisted_at", false),
        Field::new("source", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Decimal128(30, 10), false),
        ts("envelope_timestamp", true),
        Field::new("connection_id", DataType::Utf8, false),
        Field::new("ingest_sequence", DataType::Int64, false),
        Field::new("source_event_id", DataType::Utf8, true),
        Field::new("dedup_key", DataType::Utf8, false),
        Field::new("clock_skew_ms", DataType::Int64, false),
        Field::new("integrity_status", DataType::Utf8, false),
        Field::new("raw_payload", DataType::Utf8, false),
    ]))
}
fn decimal(mut v: Decimal) -> i128 {
    v.rescale(10);
    v.mantissa()
}
fn batch(rows: Vec<Row>) -> Result<RecordBatch, DrainExecutionError> {
    let payloads = rows
        .iter()
        .map(|r| r.raw_payload.to_string())
        .collect::<Vec<_>>();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.tick_id.to_string()),
        )),
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
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|r| r.persisted_at.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.source.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.symbol.as_str()),
        )),
        Arc::new(
            Decimal128Array::from_iter_values(rows.iter().map(|r| decimal(r.price)))
                .with_precision_and_scale(30, 10)
                .map_err(|e| io_error(e.to_string()))?,
        ),
        Arc::new(
            TimestampMicrosecondArray::from(
                rows.iter()
                    .map(|r| r.envelope_timestamp.map(|v| v.timestamp_micros()))
                    .collect::<Vec<_>>(),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.connection_id.to_string()),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.ingest_sequence),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|r| r.source_event_id.as_deref())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.dedup_key.as_str()),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.clock_skew_ms),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.integrity_status.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            payloads.iter().map(String::as_str),
        )),
    ];
    RecordBatch::try_new(schema(), arrays).map_err(|e| io_error(e.to_string()))
}
#[async_trait]
impl RetainedDrainAdapter for ReferencePriceTicksDrain {
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
        let mut stream=sqlx::query_as::<_,Row>("SELECT tick_id,source_timestamp,received_at,persisted_at,source,symbol,price,envelope_timestamp,connection_id,ingest_sequence,source_event_id,dedup_key,clock_skew_ms,integrity_status,raw_payload FROM polymarket.reference_price_ticks WHERE source_timestamp >= $1 AND source_timestamp < $2 ORDER BY source_timestamp,source,ingest_sequence,tick_id").bind(ch.range_start).bind(ch.range_end).fetch(&c.pool);
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
impl DrainWorkerStrategy for ReferencePriceTicksDrain {
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
#[path = "tests/reference_price_ticks.rs"]
mod tests;
