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
    ArrayRef, Date32Array, Decimal128Array, Int16Array, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use async_trait::async_trait;
use chrono::{DateTime, NaiveDate, Utc};
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
    key: "pmdata_chainlink_btcusd_twap",
    relation: "market_data.pmdata_chainlink_btcusd_twap",
    schema: "market_data",
    table: "pmdata_chainlink_btcusd_twap",
    retention_days: None,
};
const BATCH_ROWS: usize = 10_000;
pub struct PmdataChainlinkTwapDrain {
    descriptor: DrainDescriptor,
    root: PathBuf,
}
impl PmdataChainlinkTwapDrain {
    pub fn from_environment() -> Result<Self, DrainExecutionError> {
        let root = PathBuf::from(
            std::env::var("INGESTER_PMDATA_CHAINLINK_TWAP_LAKE_ROOT")
                .unwrap_or_else(|_| "/var/lib/pmdata-chainlink-twap".into()),
        );
        if !root.is_absolute() {
            return Err(invalid(
                "drain_root_invalid",
                "PMData Chainlink TWAP lake root must be absolute",
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
    provider_received_at: DateTime<Utc>,
    valid_from_timestamp: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    symbol: String,
    window_seconds: i16,
    twap_price: Decimal,
    full_accuracy_value: String,
    report_version: String,
    source_date: NaiveDate,
    archive_row_number: i64,
    artifact_id: Uuid,
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
        ts("provider_received_at"),
        ts("valid_from_timestamp"),
        ts("expires_at"),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("window_seconds", DataType::Int16, false),
        Field::new("twap_price", DataType::Decimal128(38, 18), false),
        Field::new("full_accuracy_value", DataType::Utf8, false),
        Field::new("report_version", DataType::Utf8, false),
        Field::new("source_date", DataType::Date32, false),
        Field::new("archive_row_number", DataType::Int64, false),
        Field::new("artifact_id", DataType::Utf8, false),
        ts("ingested_at"),
    ]))
}
fn decimal(mut v: Decimal) -> i128 {
    v.rescale(18);
    v.mantissa()
}
fn date(v: NaiveDate) -> i32 {
    (v - NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days() as i32
}
fn batch(rows: Vec<Row>) -> Result<RecordBatch, DrainExecutionError> {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|r| r.source_timestamp.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter()
                    .map(|r| r.provider_received_at.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter()
                    .map(|r| r.valid_from_timestamp.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|r| r.expires_at.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
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
            rows.iter().map(|r| r.report_version.as_str()),
        )),
        Arc::new(Date32Array::from_iter_values(
            rows.iter().map(|r| date(r.source_date)),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.archive_row_number),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.artifact_id.to_string()),
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
impl RetainedDrainAdapter for PmdataChainlinkTwapDrain {
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
        let object_id = create_object(context, SPEC.key, SPEC.relation, chunk).await?;
        let staging = self
            .root
            .join(".staging")
            .join(format!("{object_id}.parquet.tmp"));
        let _ = fs::remove_file(&staging).await;
        let (sender, writer) = start_writer(staging.clone(), schema());
        let mut stream=sqlx::query_as::<_,Row>("SELECT source_timestamp,provider_received_at,valid_from_timestamp,expires_at,symbol,window_seconds,twap_price,full_accuracy_value,report_version,source_date,archive_row_number,artifact_id,ingested_at FROM market_data.pmdata_chainlink_btcusd_twap WHERE source_timestamp >= $1 AND source_timestamp < $2 ORDER BY source_timestamp,source_date,archive_row_number").bind(chunk.range_start).bind(chunk.range_end).fetch(&context.pool);
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
        publish(context, &self.root, chunk, object_id, staging, count).await
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
impl DrainWorkerStrategy for PmdataChainlinkTwapDrain {
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
#[path = "tests/pmdata_chainlink_twap.rs"]
mod tests;
