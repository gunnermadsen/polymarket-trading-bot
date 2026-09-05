use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use futures_util::TryStreamExt;
use serde_json::json;
use sqlx::Row;
use tokio::fs;

use super::{
    binance_schema::{self, TradeRow},
    common::{
        create_object, db_error, existing_publication, finish_writer, invalid, io_error,
        publish_file, start_writer, verify_existing, Chunk, Publication,
    },
};
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
            .map_err(|error| invalid("drain_execution_selector_invalid", error.to_string()))
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
        let chunks = sqlx::query_as::<_, Chunk>("SELECT chunk_schema,chunk_name,range_start,range_end FROM timescaledb_information.chunks WHERE hypertable_schema='market_data' AND hypertable_name='binance_spot_btcusdt_aggregate_trades' ORDER BY range_start,chunk_name")
            .fetch_all(&context.pool).await.map_err(db_error)?;
        if request.dry_run {
            return Ok(DrainOutcome {
                rows_exported: 0,
                rows_removed: 0,
                objects_published: 0,
                bytes_written: 0,
                summary: json!({"eligible_chunks":chunks.len(),"cutoff":request.cutoff,"relation":RELATION}),
            });
        }
        if chunks.iter().any(|chunk| chunk.range_end > request.cutoff) {
            return Err(invalid(
                "drain_cutoff_incomplete",
                "cutoff must cover every source chunk because drain empties the complete relation",
            ));
        }
        for chunk in chunks {
            if context.shutdown.is_cancelled() {
                return Err(DrainExecutionError::new(
                    "drain_cancelled",
                    "drain was cancelled",
                    true,
                ));
            }
            let publication = match existing_publication(&context, KEY, &chunk).await? {
                Some(publication) => {
                    verify_existing(&self.root, &publication).await?;
                    publication
                }
                None => export_chunk(&context, &self.root, &chunk).await?,
            };
            sqlx::query_scalar::<_, i64>(
                "SELECT ingester.remove_verified_binance_aggregate_trade_chunk($1,$2)",
            )
            .bind(publication.object_id)
            .bind(&publication.sha256)
            .fetch_one(&context.pool)
            .await
            .map_err(db_error)?;
        }
        let remaining: i64 = sqlx::query_scalar("SELECT count(*) FROM timescaledb_information.chunks WHERE hypertable_schema='market_data' AND hypertable_name='binance_spot_btcusdt_aggregate_trades'")
            .fetch_one(&context.pool).await.map_err(db_error)?;
        if remaining != 0 {
            return Err(invalid(
                "drain_relation_not_empty",
                format!("{remaining} source chunks remain after drain"),
            ));
        }
        outcome(&context, request.cutoff).await
    }
}

async fn require_stopped(context: &DrainContext) -> Result<(), DrainExecutionError> {
    let row = sqlx::query("SELECT desired_state::text,observed_state::text FROM ingester.profiles WHERE strategy_key=$1")
        .bind(KEY).fetch_optional(&context.pool).await.map_err(db_error)?;
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

async fn export_chunk(
    context: &DrainContext,
    root: &std::path::Path,
    chunk: &Chunk,
) -> Result<Publication, DrainExecutionError> {
    let object_id = create_object(context, KEY, RELATION, chunk).await?;
    let staging = root
        .join(".staging")
        .join(format!("{object_id}.parquet.tmp"));
    let _ = fs::remove_file(&staging).await;
    let (sender, writer) = start_writer(staging.clone(), binance_schema::schema());
    let mut rows = sqlx::query_as::<_, TradeRow>("SELECT source,symbol,aggregate_trade_id,trade_timestamp,provider_available_at,received_at,price,quantity,first_trade_id,last_trade_id,buyer_maker,best_match,payload_sha256::text,strategy_key,capture_artifact_id,ingested_at FROM market_data.binance_spot_btcusdt_aggregate_trades WHERE trade_timestamp >= $1 AND trade_timestamp < $2 ORDER BY trade_timestamp,aggregate_trade_id")
        .bind(chunk.range_start).bind(chunk.range_end).fetch(&context.pool);
    let mut buffer = Vec::with_capacity(BATCH_ROWS);
    let mut count = 0i64;
    let mut minimum_id = None;
    let mut maximum_id = None;
    while let Some(row) = rows.try_next().await.map_err(db_error)? {
        minimum_id = Some(minimum_id.map_or(row.aggregate_trade_id, |value: i64| {
            value.min(row.aggregate_trade_id)
        }));
        maximum_id = Some(maximum_id.map_or(row.aggregate_trade_id, |value: i64| {
            value.max(row.aggregate_trade_id)
        }));
        buffer.push(row);
        count += 1;
        if buffer.len() == BATCH_ROWS {
            sender
                .send(binance_schema::to_batch(std::mem::take(&mut buffer))?)
                .await
                .map_err(|_| io_error("Parquet writer stopped"))?;
        }
    }
    if !buffer.is_empty() {
        sender
            .send(binance_schema::to_batch(buffer)?)
            .await
            .map_err(|_| io_error("Parquet writer stopped"))?;
    }
    finish_writer(sender, writer).await?;
    if count == 0 {
        let _ = fs::remove_file(&staging).await;
        return Err(invalid(
            "drain_empty_chunk",
            "eligible Timescale chunk contained no rows",
        ));
    }
    let partition = format!(
        "provider=binance_spot/dataset=aggregate_trades/symbol=BTCUSDT/year={}/month={}/day={}",
        chunk.range_start.format("%Y"),
        chunk.range_start.format("%m"),
        chunk.range_start.format("%d")
    );
    let relative = format!("{partition}/{object_id}.parquet");
    let (sha256, byte_size) = publish_file(&staging, root, &relative, count).await?;
    sqlx::query_as::<_, Publication>("UPDATE ingester.drain_objects SET row_count=$2,minimum_aggregate_trade_id=$3,maximum_aggregate_trade_id=$4,relative_path=$5,sha256=$6,byte_size=$7,status='published',published_at=clock_timestamp(),updated_at=clock_timestamp() WHERE object_id=$1 AND status='staging' RETURNING object_id,row_count,relative_path,sha256::text,byte_size,status")
        .bind(object_id).bind(count).bind(minimum_id).bind(maximum_id).bind(relative).bind(sha256).bind(byte_size).fetch_one(&context.pool).await.map_err(db_error)
}

async fn outcome(
    context: &DrainContext,
    cutoff: chrono::DateTime<chrono::Utc>,
) -> Result<DrainOutcome, DrainExecutionError> {
    let totals = sqlx::query("SELECT COALESCE(sum(row_count),0)::bigint,COALESCE(sum(byte_size),0)::bigint,count(*)::bigint FROM ingester.drain_objects WHERE job_id=$1 AND status='removed'")
        .bind(context.job_id).fetch_one(&context.pool).await.map_err(db_error)?;
    let rows = totals.try_get(0).map_err(db_error)?;
    Ok(DrainOutcome {
        rows_exported: rows,
        rows_removed: rows,
        bytes_written: totals.try_get(1).map_err(db_error)?,
        objects_published: totals.try_get(2).map_err(db_error)?,
        summary: json!({"relation":RELATION,"cutoff":cutoff,"drained":true}),
    })
}

#[cfg(test)]
#[path = "tests/binance_aggregate_trades.rs"]
mod tests;
