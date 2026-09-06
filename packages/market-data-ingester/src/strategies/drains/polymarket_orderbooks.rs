use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use futures_util::TryStreamExt;
use tokio::fs;

use crate::domain::{
    DrainContext, DrainDescriptor, DrainExecutionError, DrainOutcome, DrainRequest,
    DrainWorkerStrategy,
};

use super::{
    common::{
        create_object, db_error, finish_writer, invalid, io_error, publish_file, start_writer,
        Chunk, Publication,
    },
    orderbook_schema::{self, OrderbookRow},
    retained::{self, RetainedDrainAdapter, RetainedDrainSpec},
};

pub const KEY: &str = "polymarket_btc_five_minute_orderbooks";
pub const RELATION: &str = "polymarket.btc_five_minute_orderbook_snapshots";
const RETENTION_DAYS: i64 = 14;
const BATCH_ROWS: usize = 2_000;
const SPEC: RetainedDrainSpec = RetainedDrainSpec {
    key: KEY,
    relation: RELATION,
    schema: "polymarket",
    table: "btc_five_minute_orderbook_snapshots",
    retention_days: Some(RETENTION_DAYS),
};

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

    async fn export_orderbook_chunk(
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
                tokio::time::sleep(std::time::Duration::from_millis(15)).await;
            }
        }
        if !buffer.is_empty() {
            sender
                .send(orderbook_schema::to_batch(buffer)?)
                .await
                .map_err(|_| io_error("Parquet writer stopped"))?;
        }
        finish_writer(sender, writer).await?;
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
impl RetainedDrainAdapter for PolymarketOrderbooksDrain {
    fn spec(&self) -> &RetainedDrainSpec {
        &SPEC
    }

    fn root(&self) -> &std::path::Path {
        &self.root
    }

    async fn export_chunk(
        &self,
        context: &DrainContext,
        chunk: &Chunk,
    ) -> Result<Publication, DrainExecutionError> {
        self.export_orderbook_chunk(context, chunk).await
    }
}

#[async_trait]
impl DrainWorkerStrategy for PolymarketOrderbooksDrain {
    fn descriptor(&self) -> &DrainDescriptor {
        &self.descriptor
    }

    fn validate_request(&self, request: &DrainRequest) -> Result<(), DrainExecutionError> {
        retained::validate_strategy(self, request)
    }

    async fn execute_drain(
        &self,
        context: DrainContext,
        request: DrainRequest,
    ) -> Result<DrainOutcome, DrainExecutionError> {
        retained::execute_strategy(self, context, request).await
    }
}

#[cfg(test)]
#[path = "tests/polymarket_orderbooks.rs"]
mod tests;
