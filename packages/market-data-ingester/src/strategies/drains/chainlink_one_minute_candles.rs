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
    ArrayRef, BooleanArray, Decimal128Array, RecordBatch, StringArray, TimestampMicrosecondArray,
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
    key: "chainlink_btcusd_one_minute_candles",
    relation: "market_data.chainlink_btcusd_one_minute_candles",
    schema: "market_data",
    table: "chainlink_btcusd_one_minute_candles",
    retention_days: Some(14),
};
const BATCH_ROWS: usize = 5_000;

pub struct ChainlinkOneMinuteCandlesDrain {
    descriptor: DrainDescriptor,
    root: PathBuf,
}
impl ChainlinkOneMinuteCandlesDrain {
    pub fn from_environment() -> Result<Self, DrainExecutionError> {
        let root = PathBuf::from(
            std::env::var("INGESTER_CHAINLINK_ONE_MINUTE_CANDLES_LAKE_ROOT")
                .unwrap_or_else(|_| "/var/lib/chainlink-one-minute-candles".into()),
        );
        if !root.is_absolute() {
            return Err(invalid(
                "drain_root_invalid",
                "Chainlink candle lake root must be absolute",
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
    symbol: String,
    open_timestamp: DateTime<Utc>,
    close_timestamp: DateTime<Utc>,
    provider_available_at: Option<DateTime<Utc>>,
    received_at: DateTime<Utc>,
    open_price: Decimal,
    high_price: Decimal,
    low_price: Decimal,
    close_price: Decimal,
    volume: Option<Decimal>,
    volume_supported: bool,
    payload_sha256: String,
    strategy_key: String,
    capture_artifact_id: Uuid,
    ingested_at: DateTime<Utc>,
}
fn ts(name: &'static str, nullable: bool) -> Field {
    Field::new(
        name,
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        nullable,
    )
}
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("source", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
        ts("open_timestamp", false),
        ts("close_timestamp", false),
        ts("provider_available_at", true),
        ts("received_at", false),
        Field::new("open_price", DataType::Decimal128(38, 18), false),
        Field::new("high_price", DataType::Decimal128(38, 18), false),
        Field::new("low_price", DataType::Decimal128(38, 18), false),
        Field::new("close_price", DataType::Decimal128(38, 18), false),
        Field::new("volume", DataType::Decimal128(38, 18), true),
        Field::new("volume_supported", DataType::Boolean, false),
        Field::new("payload_sha256", DataType::Utf8, false),
        Field::new("strategy_key", DataType::Utf8, false),
        Field::new("capture_artifact_id", DataType::Utf8, false),
        ts("ingested_at", false),
    ]))
}
fn decimal(mut value: Decimal) -> i128 {
    value.rescale(18);
    value.mantissa()
}
fn decimal_array<'a>(
    values: impl Iterator<Item = &'a Decimal>,
) -> Result<ArrayRef, DrainExecutionError> {
    Ok(Arc::new(
        Decimal128Array::from_iter_values(values.map(|v| decimal(*v)))
            .with_precision_and_scale(38, 18)
            .map_err(|e| io_error(e.to_string()))?,
    ))
}
fn batch(rows: Vec<Row>) -> Result<RecordBatch, DrainExecutionError> {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.source.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|r| r.symbol.as_str()),
        )),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|r| r.open_timestamp.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|r| r.close_timestamp.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from(
                rows.iter()
                    .map(|r| r.provider_available_at.map(|v| v.timestamp_micros()))
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
        decimal_array(rows.iter().map(|r| &r.open_price))?,
        decimal_array(rows.iter().map(|r| &r.high_price))?,
        decimal_array(rows.iter().map(|r| &r.low_price))?,
        decimal_array(rows.iter().map(|r| &r.close_price))?,
        Arc::new(
            Decimal128Array::from(
                rows.iter()
                    .map(|r| r.volume.map(decimal))
                    .collect::<Vec<_>>(),
            )
            .with_precision_and_scale(38, 18)
            .map_err(|e| io_error(e.to_string()))?,
        ),
        Arc::new(BooleanArray::from(
            rows.iter().map(|r| r.volume_supported).collect::<Vec<_>>(),
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
impl RetainedDrainAdapter for ChainlinkOneMinuteCandlesDrain {
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
        let mut stream = sqlx::query_as::<_, Row>("SELECT source,symbol,open_timestamp,close_timestamp,provider_available_at,received_at,open_price,high_price,low_price,close_price,volume,volume_supported,payload_sha256::text AS payload_sha256,strategy_key,capture_artifact_id,ingested_at FROM market_data.chainlink_btcusd_one_minute_candles WHERE open_timestamp >= $1 AND open_timestamp < $2 ORDER BY open_timestamp,source,symbol")
            .bind(ch.range_start).bind(ch.range_end).fetch(&c.pool);
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
impl DrainWorkerStrategy for ChainlinkOneMinuteCandlesDrain {
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
#[path = "tests/chainlink_one_minute_candles.rs"]
mod tests;
