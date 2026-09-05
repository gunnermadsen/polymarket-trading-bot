use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use chrono::{Duration, Utc};
use futures_util::TryStreamExt;
use serde_json::json;
use sqlx::Row;
use tokio::fs;

use crate::domain::{
    DrainContext, DrainDescriptor, DrainExecutionError, DrainOutcome, DrainRequest,
    DrainWorkerStrategy,
};

use super::{
    common::{
        create_object, db_error, existing_publication, finish_writer, invalid, io_error,
        publish_file, start_writer, verify_existing, Chunk, Publication,
    },
    orderbook_schema::{self, OrderbookRow},
};

pub const KEY: &str = "polymarket_btc_five_minute_orderbooks";
pub const RELATION: &str = "polymarket.btc_five_minute_orderbook_snapshots";
const RETENTION_DAYS: i64 = 14;
const BATCH_ROWS: usize = 2_000;

pub struct PolymarketOrderbooksDrain {
    descriptor: DrainDescriptor,
    root: PathBuf,
}

impl PolymarketOrderbooksDrain {
    pub fn from_environment() -> Result<Self, DrainExecutionError> {
        let root = PathBuf::from(
            std::env::var("INGESTER_POLYMARKET_ORDERBOOK_LAKE_ROOT")
                .unwrap_or_else(|_| "/var/lib/polymarket-orderbooks".into()),
        );
        if !root.is_absolute() {
            return Err(invalid(
                "drain_root_invalid",
                "Polymarket orderbook lake root must be absolute",
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

    async fn chunks(&self, context: &DrainContext) -> Result<Vec<Chunk>, DrainExecutionError> {
        sqlx::query_as(
            "SELECT chunk_schema,chunk_name,range_start,range_end \
             FROM timescaledb_information.chunks WHERE hypertable_schema='polymarket' \
             AND hypertable_name='btc_five_minute_orderbook_snapshots' \
             ORDER BY range_start,chunk_name",
        )
        .fetch_all(&context.pool)
        .await
        .map_err(db_error)
    }

    async fn export_chunk(
        &self,
        context: &DrainContext,
        chunk: &Chunk,
    ) -> Result<Publication, DrainExecutionError> {
        let object_id = create_object(context, KEY, RELATION, chunk).await?;
        let staging = self
            .root
            .join(".staging")
            .join(format!("{object_id}.parquet.tmp"));
        let _ = fs::remove_file(&staging).await;
        let (sender, writer) = start_writer(staging.clone(), orderbook_schema::schema());
        let mut rows = sqlx::query_as::<_, OrderbookRow>(
            "SELECT sampled_at,source_timestamp,provider_available_at,received_at,source,\
             market_id,condition_id,event_slug,window_start,window_end,token_id,outcome,\
             connection_epoch,ingest_sequence,tick_size,best_bid,best_ask,bid_depth,ask_depth,\
             bids,asks,source_hash,book_sha256::text,sampling_policy,\
             sampling_policy_sha256::text,payload_sha256::text,strategy_key,\
             capture_artifact_id,ingested_at \
             FROM polymarket.btc_five_minute_orderbook_snapshots \
             WHERE sampled_at >= $1 AND sampled_at < $2 \
             ORDER BY sampled_at,market_id,token_id,sampling_policy_sha256",
        )
        .bind(chunk.range_start)
        .bind(chunk.range_end)
        .fetch(&context.pool);
        let mut buffer = Vec::with_capacity(BATCH_ROWS);
        let mut count = 0i64;
        while let Some(row) = rows.try_next().await.map_err(db_error)? {
            buffer.push(row);
            count += 1;
            if buffer.len() == BATCH_ROWS {
                sender
                    .send(orderbook_schema::to_batch(std::mem::take(&mut buffer))?)
                    .await
                    .map_err(|_| io_error("Parquet writer stopped"))?;
            }
        }
        if !buffer.is_empty() {
            sender
                .send(orderbook_schema::to_batch(buffer)?)
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
            "verified-chunks-v1/year={}/month={}/day={}",
            chunk.range_start.format("%Y"),
            chunk.range_start.format("%m"),
            chunk.range_start.format("%d"),
        );
        let relative_path = format!("{partition}/{object_id}.parquet");
        let (sha256, byte_size) = publish_file(&staging, &self.root, &relative_path, count).await?;
        sqlx::query_as::<_, Publication>(
            "UPDATE ingester.drain_objects SET row_count=$2,relative_path=$3,sha256=$4,\
             byte_size=$5,status='published',published_at=clock_timestamp(),\
             updated_at=clock_timestamp() WHERE object_id=$1 AND status='staging' \
             RETURNING object_id,row_count,relative_path,sha256::text,byte_size,status",
        )
        .bind(object_id)
        .bind(count)
        .bind(relative_path)
        .bind(sha256)
        .bind(byte_size)
        .fetch_one(&context.pool)
        .await
        .map_err(db_error)
    }
}

#[async_trait]
impl DrainWorkerStrategy for PolymarketOrderbooksDrain {
    fn descriptor(&self) -> &DrainDescriptor {
        &self.descriptor
    }

    fn validate_request(&self, request: &DrainRequest) -> Result<(), DrainExecutionError> {
        if request.strategy_key != KEY {
            return Err(invalid(
                "drain_strategy_mismatch",
                "request does not target the Polymarket orderbook drain",
            ));
        }
        request
            .execution
            .validate()
            .map_err(|error| invalid("drain_execution_selector_invalid", error.to_string()))?;
        let newest_allowed = Utc::now() - Duration::days(RETENTION_DAYS);
        if request.cutoff > newest_allowed {
            return Err(invalid(
                "drain_retention_violation",
                format!("cutoff must retain at least {RETENTION_DAYS} days of orderbook data"),
            ));
        }
        Ok(())
    }

    async fn execute_drain(
        &self,
        context: DrainContext,
        request: DrainRequest,
    ) -> Result<DrainOutcome, DrainExecutionError> {
        self.validate_request(&request)?;
        let chunks = self.chunks(&context).await?;
        let eligible: Vec<_> = chunks
            .iter()
            .filter(|chunk| chunk.range_end <= request.cutoff)
            .cloned()
            .collect();
        if request.dry_run {
            return Ok(DrainOutcome {
                rows_exported: 0,
                rows_removed: 0,
                objects_published: 0,
                bytes_written: 0,
                summary: json!({
                    "cutoff": request.cutoff,
                    "eligible_chunks": eligible.len(),
                    "relation": RELATION,
                    "retained_chunks": chunks.len() - eligible.len(),
                    "retention_days": RETENTION_DAYS,
                }),
            });
        }
        fs::create_dir_all(self.root.join(".staging"))
            .await
            .map_err(io_error)?;
        for chunk in eligible {
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
                None => self.export_chunk(&context, &chunk).await?,
            };
            sqlx::query_scalar::<_, i64>("SELECT ingester.remove_verified_drain_chunk($1,$2)")
                .bind(publication.object_id)
                .bind(&publication.sha256)
                .fetch_one(&context.pool)
                .await
                .map_err(db_error)?;
        }
        let totals = sqlx::query(
            "SELECT COALESCE(sum(row_count),0)::bigint,COALESCE(sum(byte_size),0)::bigint,\
             count(*)::bigint FROM ingester.drain_objects WHERE job_id=$1 AND status='removed'",
        )
        .bind(context.job_id)
        .fetch_one(&context.pool)
        .await
        .map_err(db_error)?;
        let rows: i64 = totals.try_get(0).map_err(db_error)?;
        Ok(DrainOutcome {
            rows_exported: rows,
            rows_removed: rows,
            bytes_written: totals.try_get(1).map_err(db_error)?,
            objects_published: totals.try_get(2).map_err(db_error)?,
            summary: json!({
                "cutoff": request.cutoff,
                "relation": RELATION,
                "retention_days": RETENTION_DAYS,
            }),
        })
    }
}

#[cfg(test)]
#[path = "tests/polymarket_orderbooks.rs"]
mod tests;
