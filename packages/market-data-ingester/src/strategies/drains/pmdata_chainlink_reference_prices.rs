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
    ArrayRef, Date32Array, Decimal128Array, Int64Array, RecordBatch, StringArray,
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
    key: "pmdata_chainlink_btcusd_reference_price",
    relation: "market_data.pmdata_chainlink_btcusd_reference_prices",
    schema: "market_data",
    table: "pmdata_chainlink_btcusd_reference_prices",
    retention_days: None,
};
const BATCH_ROWS: usize = 10_000;
pub struct PmdataChainlinkReferencePricesDrain {
    descriptor: DrainDescriptor,
    root: PathBuf,
}
impl PmdataChainlinkReferencePricesDrain {
    pub fn from_environment() -> Result<Self, DrainExecutionError> {
        let root = PathBuf::from(
            std::env::var("INGESTER_PMDATA_CHAINLINK_REFERENCE_PRICE_LAKE_ROOT")
                .unwrap_or_else(|_| "/var/lib/pmdata-chainlink-reference-prices".into()),
        );
        if !root.is_absolute() {
            return Err(invalid(
                "drain_root_invalid",
                "PMData Chainlink reference-price lake root must be absolute",
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
    source: String,
    feed_id: String,
    source_timestamp: DateTime<Utc>,
    valid_from_timestamp: Option<DateTime<Utc>>,
    provider_available_at: Option<DateTime<Utc>>,
    received_at: DateTime<Utc>,
    price: Decimal,
    bid: Option<Decimal>,
    ask: Option<Decimal>,
    report_sha256: String,
    payload_sha256: String,
    strategy_key: String,
    capture_artifact_id: Option<Uuid>,
    ingested_at: DateTime<Utc>,
    expires_at: Option<DateTime<Utc>>,
    report_version: Option<String>,
    source_date: Option<NaiveDate>,
    archive_row_number: Option<i64>,
    backfill_artifact_id: Option<Uuid>,
    report_hash_kind: String,
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
        Field::new("source", DataType::Utf8, false),
        Field::new("feed_id", DataType::Utf8, false),
        ts("source_timestamp", false),
        ts("valid_from_timestamp", true),
        ts("provider_available_at", true),
        ts("received_at", false),
        Field::new("price", DataType::Decimal128(38, 18), false),
        Field::new("bid", DataType::Decimal128(38, 18), true),
        Field::new("ask", DataType::Decimal128(38, 18), true),
        Field::new("report_sha256", DataType::Utf8, false),
        Field::new("payload_sha256", DataType::Utf8, false),
        Field::new("strategy_key", DataType::Utf8, false),
        Field::new("capture_artifact_id", DataType::Utf8, true),
        ts("ingested_at", false),
        ts("expires_at", true),
        Field::new("report_version", DataType::Utf8, true),
        Field::new("source_date", DataType::Date32, true),
        Field::new("archive_row_number", DataType::Int64, true),
        Field::new("backfill_artifact_id", DataType::Utf8, true),
        Field::new("report_hash_kind", DataType::Utf8, false),
    ]))
}
fn decimal(mut v: Decimal) -> i128 {
    v.rescale(18);
    v.mantissa()
}
fn opt_ts(v: Option<DateTime<Utc>>) -> Option<i64> {
    v.map(|x| x.timestamp_micros())
}
fn date(v: NaiveDate) -> i32 {
    (v - NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days() as i32
}
fn batch(rows: Vec<Row>) -> Result<RecordBatch, DrainExecutionError> {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.source.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.feed_id.as_str()),
        )),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|r| r.source_timestamp.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from(
                rows.iter()
                    .map(|r| opt_ts(r.valid_from_timestamp))
                    .collect::<Vec<_>>(),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from(
                rows.iter()
                    .map(|r| opt_ts(r.provider_available_at))
                    .collect::<Vec<_>>(),
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
            Decimal128Array::from_iter_values(rows.iter().map(|r| decimal(r.price)))
                .with_precision_and_scale(38, 18)
                .map_err(|e| io_error(e.to_string()))?,
        ),
        Arc::new(
            Decimal128Array::from(rows.iter().map(|r| r.bid.map(decimal)).collect::<Vec<_>>())
                .with_precision_and_scale(38, 18)
                .map_err(|e| io_error(e.to_string()))?,
        ),
        Arc::new(
            Decimal128Array::from(rows.iter().map(|r| r.ask.map(decimal)).collect::<Vec<_>>())
                .with_precision_and_scale(38, 18)
                .map_err(|e| io_error(e.to_string()))?,
        ),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.report_sha256.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.payload_sha256.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.strategy_key.as_str()),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|r| r.capture_artifact_id.map(|v| v.to_string()))
                .collect::<Vec<_>>(),
        )),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|r| r.ingested_at.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from(
                rows.iter()
                    .map(|r| opt_ts(r.expires_at))
                    .collect::<Vec<_>>(),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|r| r.report_version.as_deref())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Date32Array::from(
            rows.iter()
                .map(|r| r.source_date.map(date))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|r| r.archive_row_number)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|r| r.backfill_artifact_id.map(|v| v.to_string()))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.report_hash_kind.as_str()),
        )),
    ];
    RecordBatch::try_new(schema(), arrays).map_err(|e| io_error(e.to_string()))
}
#[async_trait]
impl RetainedDrainAdapter for PmdataChainlinkReferencePricesDrain {
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
        let mut stream=sqlx::query_as::<_,Row>("SELECT source,feed_id,source_timestamp,valid_from_timestamp,provider_available_at,received_at,price,bid,ask,report_sha256::text,payload_sha256::text,strategy_key,capture_artifact_id,ingested_at,expires_at,report_version,source_date,archive_row_number,backfill_artifact_id,report_hash_kind FROM market_data.pmdata_chainlink_btcusd_reference_prices WHERE source_timestamp >= $1 AND source_timestamp < $2 ORDER BY source_timestamp,source,feed_id,report_sha256").bind(ch.range_start).bind(ch.range_end).fetch(&c.pool);
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
impl DrainWorkerStrategy for PmdataChainlinkReferencePricesDrain {
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
#[path = "tests/pmdata_chainlink_reference_prices.rs"]
mod tests;
