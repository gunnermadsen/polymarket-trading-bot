use std::{
    path::{Path, PathBuf},
    sync::Arc,
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
use tokio::fs;
use uuid::Uuid;

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

const SPEC: RetainedDrainSpec = RetainedDrainSpec {
    key: "binance_spot_btcusdt_one_second_ohlcv",
    relation: "market_data.binance_spot_btcusdt_one_second_ohlcv",
    schema: "market_data",
    table: "binance_spot_btcusdt_one_second_ohlcv",
    retention_days: Some(14),
};
const BATCH_ROWS: usize = 20_000;

pub struct BinanceOneSecondOhlcvDrain {
    descriptor: DrainDescriptor,
    root: PathBuf,
}

impl BinanceOneSecondOhlcvDrain {
    pub fn from_environment() -> Result<Self, DrainExecutionError> {
        let root = PathBuf::from(
            std::env::var("INGESTER_BINANCE_ONE_SECOND_OHLCV_LAKE_ROOT")
                .unwrap_or_else(|_| "/var/lib/binance-one-second-ohlcv".into()),
        );
        if !root.is_absolute() {
            return Err(invalid(
                "drain_root_invalid",
                "Binance one-second OHLCV lake root must be absolute",
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
    base_volume: Decimal,
    quote_volume: Decimal,
    trade_count: i64,
    taker_buy_base_volume: Decimal,
    taker_buy_quote_volume: Decimal,
    payload_sha256: String,
    strategy_key: String,
    capture_artifact_id: Uuid,
    ingested_at: DateTime<Utc>,
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("source", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
        ts("open_timestamp", false),
        ts("close_timestamp", false),
        ts("provider_available_at", true),
        ts("received_at", false),
        dec("open_price"),
        dec("high_price"),
        dec("low_price"),
        dec("close_price"),
        dec("base_volume"),
        dec("quote_volume"),
        Field::new("trade_count", DataType::Int64, false),
        dec("taker_buy_base_volume"),
        dec("taker_buy_quote_volume"),
        Field::new("payload_sha256", DataType::Utf8, false),
        Field::new("strategy_key", DataType::Utf8, false),
        Field::new("capture_artifact_id", DataType::Utf8, false),
        ts("ingested_at", false),
    ]))
}
fn ts(name: &'static str, nullable: bool) -> Field {
    Field::new(
        name,
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        nullable,
    )
}
fn dec(name: &'static str) -> Field {
    Field::new(name, DataType::Decimal128(30, 10), false)
}
fn decimal(mut value: Decimal) -> i128 {
    value.rescale(10);
    value.mantissa()
}
fn decimal_array(rows: &[Row], get: fn(&Row) -> i128) -> Result<ArrayRef, DrainExecutionError> {
    Ok(Arc::new(
        Decimal128Array::from_iter_values(rows.iter().map(get))
            .with_precision_and_scale(30, 10)
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
        decimal_array(&rows, |r| decimal(r.open_price))?,
        decimal_array(&rows, |r| decimal(r.high_price))?,
        decimal_array(&rows, |r| decimal(r.low_price))?,
        decimal_array(&rows, |r| decimal(r.close_price))?,
        decimal_array(&rows, |r| decimal(r.base_volume))?,
        decimal_array(&rows, |r| decimal(r.quote_volume))?,
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.trade_count),
        )),
        decimal_array(&rows, |r| decimal(r.taker_buy_base_volume))?,
        decimal_array(&rows, |r| decimal(r.taker_buy_quote_volume))?,
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
impl RetainedDrainAdapter for BinanceOneSecondOhlcvDrain {
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
        let mut stream=sqlx::query_as::<_,Row>("SELECT source,symbol,open_timestamp,close_timestamp,provider_available_at,received_at,open_price,high_price,low_price,close_price,base_volume,quote_volume,trade_count,taker_buy_base_volume,taker_buy_quote_volume,payload_sha256::text,strategy_key,capture_artifact_id,ingested_at FROM market_data.binance_spot_btcusdt_one_second_ohlcv WHERE open_timestamp >= $1 AND open_timestamp < $2 ORDER BY open_timestamp,source,symbol").bind(chunk.range_start).bind(chunk.range_end).fetch(&context.pool);
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
    object_id: Uuid,
    staging: PathBuf,
    count: i64,
) -> Result<Publication, DrainExecutionError> {
    let relative = format!(
        "verified-chunks-v1/year={}/month={}/day={}/{object_id}.parquet",
        chunk.range_start.format("%Y"),
        chunk.range_start.format("%m"),
        chunk.range_start.format("%d")
    );
    let (sha, size) = publish_file(&staging, root, &relative, count).await?;
    sqlx::query_as("UPDATE ingester.drain_objects SET row_count=$2,relative_path=$3,sha256=$4,byte_size=$5,status='published',published_at=clock_timestamp(),updated_at=clock_timestamp() WHERE object_id=$1 AND status='staging' RETURNING object_id,row_count,relative_path,sha256::text,byte_size,status").bind(object_id).bind(count).bind(relative).bind(sha).bind(size).fetch_one(&context.pool).await.map_err(db_error)
}
#[async_trait]
impl DrainWorkerStrategy for BinanceOneSecondOhlcvDrain {
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
#[path = "tests/binance_one_second_ohlcv.rs"]
mod tests;
