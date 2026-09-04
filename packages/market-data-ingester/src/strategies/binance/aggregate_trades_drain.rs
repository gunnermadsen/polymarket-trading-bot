use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow_array::{
    ArrayRef, BooleanArray, Decimal128Array, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures_util::TryStreamExt;
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::{
        properties::WriterProperties,
        reader::{FileReader, SerializedFileReader},
    },
};
use rust_decimal::Decimal;
use serde_json::json;
use sha2::{Digest, Sha256};
use sqlx::{FromRow, Row};
use tokio::{fs, io::AsyncReadExt, sync::mpsc};
use uuid::Uuid;

use crate::domain::{
    DrainContext, DrainDescriptor, DrainExecutionError, DrainOutcome, DrainRequest,
    DrainWorkerStrategy,
};

const KEY: &str = "binance_spot_btcusdt_aggregate_trades";
const RELATION: &str = "market_data.binance_spot_btcusdt_aggregate_trades";
const BATCH_ROWS: usize = 50_000;

pub struct BinanceAggregateTradesDrain {
    descriptor: DrainDescriptor,
    root: PathBuf,
}

impl BinanceAggregateTradesDrain {
    pub fn from_environment() -> Result<Self, DrainExecutionError> {
        let root = PathBuf::from(
            std::env::var("INGESTER_BINANCE_AGG_TRADE_LAKE_ROOT")
                .unwrap_or_else(|_| "/var/lib/binance-aggregate-trades".into()),
        );
        if !root.is_absolute() {
            return Err(invalid(
                "drain_root_invalid",
                "Binance aggregate-trade lake root must be absolute",
            ));
        }
        Ok(Self {
            descriptor: DrainDescriptor {
                strategy_key: Arc::from(KEY),
                relation: Arc::from(RELATION),
                contract_version: 1,
            },
            root,
        })
    }
}

#[derive(Debug, FromRow)]
struct Chunk {
    chunk_schema: String,
    chunk_name: String,
    range_start: DateTime<Utc>,
    range_end: DateTime<Utc>,
}

#[derive(Debug, FromRow)]
struct Trade {
    source: String,
    symbol: String,
    aggregate_trade_id: i64,
    trade_timestamp: DateTime<Utc>,
    provider_available_at: Option<DateTime<Utc>>,
    received_at: DateTime<Utc>,
    price: Decimal,
    quantity: Decimal,
    first_trade_id: i64,
    last_trade_id: i64,
    buyer_maker: bool,
    best_match: bool,
    payload_sha256: String,
    strategy_key: String,
    capture_artifact_id: Uuid,
    ingested_at: DateTime<Utc>,
}

#[async_trait]
impl DrainWorkerStrategy for BinanceAggregateTradesDrain {
    fn descriptor(&self) -> &DrainDescriptor {
        &self.descriptor
    }
    fn validate_request(&self, request: &DrainRequest) -> Result<(), DrainExecutionError> {
        if request.strategy_key != KEY {
            return Err(invalid(
                "drain_strategy_mismatch",
                "request does not target the Binance aggregate-trade drain",
            ));
        }
        request
            .execution
            .validate()
            .map_err(|e| invalid("drain_execution_selector_invalid", e.to_string()))
    }
    async fn execute_drain(
        &self,
        context: DrainContext,
        request: DrainRequest,
    ) -> Result<DrainOutcome, DrainExecutionError> {
        self.validate_request(&request)?;
        if !request.dry_run {
            require_stopped(&context).await?;
            fs::create_dir_all(self.root.join(".staging"))
                .await
                .map_err(io_error)?;
        }
        let chunks=sqlx::query_as::<_,Chunk>("SELECT chunk_schema,chunk_name,range_start,range_end FROM timescaledb_information.chunks WHERE hypertable_schema='market_data' AND hypertable_name='binance_spot_btcusdt_aggregate_trades' ORDER BY range_start,chunk_name")
            .fetch_all(&context.pool).await.map_err(db_error)?;
        let cutoff_covers_all = chunks.iter().all(|chunk| chunk.range_end <= request.cutoff);
        if request.dry_run {
            return Ok(DrainOutcome {
                rows_exported: 0,
                rows_removed: 0,
                objects_published: 0,
                bytes_written: 0,
                summary: json!({"eligible_chunks":chunks.len(),"cutoff":request.cutoff,"relation":RELATION}),
            });
        }
        if !cutoff_covers_all {
            return Err(invalid(
                "drain_cutoff_incomplete",
                "cutoff must cover every source chunk because drain empties the complete relation",
            ));
        }
        let mut outcome = DrainOutcome {
            rows_exported: 0,
            rows_removed: 0,
            objects_published: 0,
            bytes_written: 0,
            summary: json!({}),
        };
        for chunk in chunks {
            if context.shutdown.is_cancelled() {
                return Err(DrainExecutionError::new(
                    "drain_cancelled",
                    "drain was cancelled",
                    true,
                ));
            }
            let published = existing_publication(&context, &chunk).await?;
            let publication = match published {
                Some(value) => {
                    verify_existing(&self.root, &value).await?;
                    value
                }
                None => export_chunk(&context, &self.root, &chunk).await?,
            };
            let removed: i64 = sqlx::query_scalar(
                "SELECT ingester.remove_verified_binance_aggregate_trade_chunk($1,$2)",
            )
            .bind(publication.object_id)
            .bind(&publication.sha256)
            .fetch_one(&context.pool)
            .await
            .map_err(db_error)?;
            outcome.rows_exported += publication.row_count;
            outcome.rows_removed += removed;
            outcome.objects_published += 1;
            outcome.bytes_written += publication.byte_size;
        }
        let remaining:i64=sqlx::query_scalar("SELECT count(*) FROM timescaledb_information.chunks WHERE hypertable_schema='market_data' AND hypertable_name='binance_spot_btcusdt_aggregate_trades'")
            .fetch_one(&context.pool).await.map_err(db_error)?;
        if remaining != 0 {
            return Err(invalid(
                "drain_relation_not_empty",
                format!("{remaining} source chunks remain after drain"),
            ));
        }
        let totals=sqlx::query("SELECT COALESCE(sum(row_count),0)::bigint,COALESCE(sum(byte_size),0)::bigint,count(*)::bigint FROM ingester.drain_objects WHERE job_id=$1 AND status='removed'")
            .bind(context.job_id).fetch_one(&context.pool).await.map_err(db_error)?;
        outcome.rows_exported = totals.try_get(0).map_err(db_error)?;
        outcome.rows_removed = outcome.rows_exported;
        outcome.bytes_written = totals.try_get(1).map_err(db_error)?;
        outcome.objects_published = totals.try_get(2).map_err(db_error)?;
        outcome.summary = json!({"relation":RELATION,"cutoff":request.cutoff,"drained":true});
        Ok(outcome)
    }
}

#[derive(Debug, FromRow)]
struct Publication {
    object_id: Uuid,
    row_count: i64,
    relative_path: String,
    sha256: String,
    byte_size: i64,
    status: String,
}

async fn require_stopped(context: &DrainContext) -> Result<(), DrainExecutionError> {
    let row=sqlx::query("SELECT desired_state::text,observed_state::text FROM ingester.profiles WHERE strategy_key=$1").bind(KEY).fetch_optional(&context.pool).await.map_err(db_error)?;
    let Some(row) = row else {
        return Err(invalid(
            "drain_profile_missing",
            "Binance aggregate-trade profile is missing",
        ));
    };
    let desired: String = row.try_get(0).map_err(db_error)?;
    let observed: String = row.try_get(1).map_err(db_error)?;
    if desired != "stopped" || observed != "stopped" {
        return Err(invalid(
            "drain_strategy_running",
            "full drain requires the realtime aggregate-trade strategy to be stopped",
        ));
    }
    Ok(())
}

async fn existing_publication(
    context: &DrainContext,
    chunk: &Chunk,
) -> Result<Option<Publication>, DrainExecutionError> {
    let value=sqlx::query_as::<_,Publication>("SELECT object_id,row_count,relative_path,sha256::text,byte_size,status FROM ingester.drain_objects WHERE strategy_key=$1 AND source_chunk_schema=$2 AND source_chunk_name=$3 AND status IN ('published','removed')")
        .bind(KEY).bind(&chunk.chunk_schema).bind(&chunk.chunk_name).fetch_optional(&context.pool).await.map_err(db_error)?;
    if value.as_ref().is_some_and(|v| v.status == "removed") {
        return Err(invalid(
            "drain_chunk_already_removed",
            "removed drain object still has a source chunk",
        ));
    }
    Ok(value)
}

async fn export_chunk(
    context: &DrainContext,
    root: &Path,
    chunk: &Chunk,
) -> Result<Publication, DrainExecutionError> {
    let object_id:Uuid=sqlx::query_scalar("INSERT INTO ingester.drain_objects(job_id,strategy_key,source_relation,source_chunk_schema,source_chunk_name,source_start,source_end) VALUES($1,$2,$3,$4,$5,$6,$7) ON CONFLICT(strategy_key,source_chunk_schema,source_chunk_name) DO UPDATE SET job_id=EXCLUDED.job_id,updated_at=clock_timestamp() WHERE ingester.drain_objects.status='staging' RETURNING object_id")
        .bind(context.job_id).bind(KEY).bind(RELATION).bind(&chunk.chunk_schema).bind(&chunk.chunk_name).bind(chunk.range_start).bind(chunk.range_end)
        .fetch_one(&context.pool).await.map_err(db_error)?;
    let staging = root
        .join(".staging")
        .join(format!("{object_id}.parquet.tmp"));
    let _ = fs::remove_file(&staging).await;
    let schema = trade_schema();
    let (sender, mut receiver) = mpsc::channel::<RecordBatch>(2);
    let output = staging.clone();
    let writer_schema = schema.clone();
    let writer = tokio::task::spawn_blocking(move || -> Result<(), String> {
        let file = File::create(&output).map_err(|e| e.to_string())?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(
                ZstdLevel::try_new(6).map_err(|e| e.to_string())?,
            ))
            .build();
        let mut writer =
            ArrowWriter::try_new(file, writer_schema, Some(props)).map_err(|e| e.to_string())?;
        while let Some(batch) = receiver.blocking_recv() {
            writer.write(&batch).map_err(|e| e.to_string())?;
        }
        let mut file = writer.into_inner().map_err(|e| e.to_string())?;
        use std::io::Write;
        file.flush().map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        Ok(())
    });
    let mut rows=sqlx::query_as::<_,Trade>("SELECT source,symbol,aggregate_trade_id,trade_timestamp,provider_available_at,received_at,price,quantity,first_trade_id,last_trade_id,buyer_maker,best_match,payload_sha256::text,strategy_key,capture_artifact_id,ingested_at FROM market_data.binance_spot_btcusdt_aggregate_trades WHERE trade_timestamp >= $1 AND trade_timestamp < $2 ORDER BY trade_timestamp,aggregate_trade_id")
        .bind(chunk.range_start).bind(chunk.range_end).fetch(&context.pool);
    let mut buffer = Vec::with_capacity(BATCH_ROWS);
    let mut count = 0i64;
    let mut min_id = None;
    let mut max_id = None;
    while let Some(row) = rows.try_next().await.map_err(db_error)? {
        min_id = Some(min_id.map_or(row.aggregate_trade_id, |v: i64| {
            v.min(row.aggregate_trade_id)
        }));
        max_id = Some(max_id.map_or(row.aggregate_trade_id, |v: i64| {
            v.max(row.aggregate_trade_id)
        }));
        buffer.push(row);
        count += 1;
        if buffer.len() == BATCH_ROWS {
            sender
                .send(to_batch(schema.clone(), std::mem::take(&mut buffer))?)
                .await
                .map_err(|_| io_error("Parquet writer stopped"))?;
        }
    }
    if !buffer.is_empty() {
        sender
            .send(to_batch(schema, std::mem::take(&mut buffer))?)
            .await
            .map_err(|_| io_error("Parquet writer stopped"))?;
    }
    drop(sender);
    writer
        .await
        .map_err(|e| io_error(e.to_string()))?
        .map_err(io_error)?;
    if count == 0 {
        let _ = fs::remove_file(&staging).await;
        return Err(invalid(
            "drain_empty_chunk",
            "eligible Timescale chunk contained no rows",
        ));
    }
    let (sha256, byte_size) = hash_file(&staging).await?;
    verify_parquet(&staging, count).await?;
    let partition = format!(
        "provider=binance_spot/dataset=aggregate_trades/symbol=BTCUSDT/year={}/month={}/day={}",
        chunk.range_start.format("%Y"),
        chunk.range_start.format("%m"),
        chunk.range_start.format("%d")
    );
    let directory = root.join(&partition);
    fs::create_dir_all(&directory).await.map_err(io_error)?;
    let name = format!("{}_{}_{}.parquet", min_id.unwrap(), max_id.unwrap(), sha256);
    let relative = format!("{partition}/{name}");
    let final_path = root.join(&relative);
    if final_path.exists() {
        verify_hash(&final_path, &sha256, byte_size).await?;
        fs::remove_file(&staging).await.map_err(io_error)?;
    } else {
        fs::rename(&staging, &final_path).await.map_err(io_error)?;
        sync_directory(directory).await?;
    }
    sqlx::query_as::<_,Publication>("UPDATE ingester.drain_objects SET row_count=$2,minimum_aggregate_trade_id=$3,maximum_aggregate_trade_id=$4,relative_path=$5,sha256=$6,byte_size=$7,status='published',published_at=clock_timestamp(),updated_at=clock_timestamp() WHERE object_id=$1 AND status='staging' RETURNING object_id,row_count,relative_path,sha256::text,byte_size,status")
        .bind(object_id).bind(count).bind(min_id).bind(max_id).bind(relative).bind(sha256).bind(byte_size).fetch_one(&context.pool).await.map_err(db_error)
}

fn trade_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("source", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("aggregate_trade_id", DataType::Int64, false),
        Field::new(
            "trade_timestamp",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new(
            "provider_available_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new(
            "received_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("price", DataType::Decimal128(30, 10), false),
        Field::new("quantity", DataType::Decimal128(30, 10), false),
        Field::new("first_trade_id", DataType::Int64, false),
        Field::new("last_trade_id", DataType::Int64, false),
        Field::new("buyer_maker", DataType::Boolean, false),
        Field::new("best_match", DataType::Boolean, false),
        Field::new("payload_sha256", DataType::Utf8, false),
        Field::new("strategy_key", DataType::Utf8, false),
        Field::new("capture_artifact_id", DataType::Utf8, false),
        Field::new(
            "ingested_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
    ]))
}

fn decimal(mut v: Decimal) -> i128 {
    v.rescale(10);
    v.mantissa()
}
fn to_batch(schema: Arc<Schema>, rows: Vec<Trade>) -> Result<RecordBatch, DrainExecutionError> {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|v| v.source.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|v| v.symbol.as_str()),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|v| v.aggregate_trade_id),
        )),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|v| v.trade_timestamp.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from(
                rows.iter()
                    .map(|v| v.provider_available_at.map(|x| x.timestamp_micros()))
                    .collect::<Vec<_>>(),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|v| v.received_at.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            Decimal128Array::from_iter_values(rows.iter().map(|v| decimal(v.price)))
                .with_precision_and_scale(30, 10)
                .map_err(|e| io_error(e.to_string()))?,
        ),
        Arc::new(
            Decimal128Array::from_iter_values(rows.iter().map(|v| decimal(v.quantity)))
                .with_precision_and_scale(30, 10)
                .map_err(|e| io_error(e.to_string()))?,
        ),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|v| v.first_trade_id),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|v| v.last_trade_id),
        )),
        Arc::new(BooleanArray::from_iter(
            rows.iter().map(|v| Some(v.buyer_maker)),
        )),
        Arc::new(BooleanArray::from_iter(
            rows.iter().map(|v| Some(v.best_match)),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|v| v.payload_sha256.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|v| v.strategy_key.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|v| v.capture_artifact_id.to_string()),
        )),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|v| v.ingested_at.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
    ];
    RecordBatch::try_new(schema, arrays).map_err(|e| io_error(e.to_string()))
}

async fn hash_file(path: &Path) -> Result<(String, i64), DrainExecutionError> {
    let mut file = fs::File::open(path).await.map_err(io_error)?;
    let mut hash = Sha256::new();
    let mut bytes = 0i64;
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf).await.map_err(io_error)?;
        if n == 0 {
            break;
        }
        bytes += n as i64;
        hash.update(&buf[..n]);
    }
    Ok((format!("{:x}", hash.finalize()), bytes))
}
async fn verify_hash(path: &Path, expected: &str, size: i64) -> Result<(), DrainExecutionError> {
    let (actual, bytes) = hash_file(path).await?;
    if actual != expected || bytes != size {
        return Err(invalid(
            "drain_object_conflict",
            "existing Parquet object hash or size differs",
        ));
    }
    Ok(())
}
async fn verify_existing(root: &Path, p: &Publication) -> Result<(), DrainExecutionError> {
    verify_hash(&root.join(&p.relative_path), &p.sha256, p.byte_size).await?;
    verify_parquet(&root.join(&p.relative_path), p.row_count).await
}
async fn verify_parquet(path: &Path, expected: i64) -> Result<(), DrainExecutionError> {
    let path = path.to_owned();
    let rows = tokio::task::spawn_blocking(move || {
        SerializedFileReader::new(File::open(path).map_err(|e| e.to_string())?)
            .map(|r| r.metadata().file_metadata().num_rows())
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| io_error(e.to_string()))?
    .map_err(io_error)?;
    if rows != expected {
        return Err(invalid(
            "drain_row_count_mismatch",
            format!("Parquet has {rows} rows, expected {expected}"),
        ));
    }
    Ok(())
}
async fn sync_directory(path: PathBuf) -> Result<(), DrainExecutionError> {
    tokio::task::spawn_blocking(move || File::open(path)?.sync_all())
        .await
        .map_err(|e| io_error(e.to_string()))?
        .map_err(io_error)
}
fn invalid(code: &'static str, message: impl Into<String>) -> DrainExecutionError {
    DrainExecutionError::new(code, message, false)
}
fn io_error(error: impl std::fmt::Display) -> DrainExecutionError {
    DrainExecutionError::new("drain_io_failed", error.to_string(), true)
}
fn db_error(error: impl std::fmt::Display) -> DrainExecutionError {
    DrainExecutionError::new("drain_database_failed", error.to_string(), true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ExecutionSelector;
    use arrow_array::Array;

    #[test]
    fn parquet_batch_preserves_the_complete_source_row() {
        let now = DateTime::parse_from_rfc3339("2026-09-04T12:34:56.123456Z")
            .unwrap()
            .with_timezone(&Utc);
        let row = Trade {
            source: "binance_spot".into(),
            symbol: "BTCUSDT".into(),
            aggregate_trade_id: 42,
            trade_timestamp: now,
            provider_available_at: Some(now),
            received_at: now,
            price: Decimal::new(60000123456789, 10),
            quantity: Decimal::new(123456789, 10),
            first_trade_id: 40,
            last_trade_id: 42,
            buyer_maker: true,
            best_match: false,
            payload_sha256: "a".repeat(64),
            strategy_key: KEY.into(),
            capture_artifact_id: Uuid::nil(),
            ingested_at: now,
        };
        let batch = to_batch(trade_schema(), vec![row]).unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 16);
        assert_eq!(
            batch.schema().field(6).data_type(),
            &DataType::Decimal128(30, 10)
        );
        assert_eq!(batch.schema().field(15).name(), "ingested_at");
        assert_eq!(batch.column(4).null_count(), 0);
    }

    #[test]
    fn only_the_registered_relation_is_accepted() {
        let strategy = BinanceAggregateTradesDrain::from_environment().unwrap();
        let request = DrainRequest {
            strategy_key: "market_data.anything".into(),
            cutoff: Utc::now(),
            dry_run: true,
            execution: ExecutionSelector::default(),
        };
        assert_eq!(
            strategy.validate_request(&request).unwrap_err().code,
            "drain_strategy_mismatch"
        );
    }
}
