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
use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use futures_util::TryStreamExt;
use sqlx::{postgres::PgRow, Row};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::fs;
use uuid::Uuid;
const SPEC: RetainedDrainSpec = RetainedDrainSpec {
    key: "polygon_chainlink_btcusd_oracle_rounds",
    relation: "market_data.polygon_chainlink_btcusd_oracle_rounds",
    schema: "market_data",
    table: "polygon_chainlink_btcusd_oracle_rounds",
    retention_days: Some(30),
};
const BATCH_ROWS: usize = 5_000;
const COLUMNS: [&str; 22] = [
    "source",
    "chain_id",
    "feed_proxy_address",
    "aggregator_address",
    "phase_id",
    "aggregator_round_id",
    "source_timestamp",
    "block_timestamp",
    "answer_raw",
    "price",
    "decimals",
    "block_number",
    "block_hash",
    "transaction_hash",
    "log_index",
    "provider_available_at",
    "received_at",
    "source_payload",
    "payload_sha256",
    "strategy_key",
    "capture_artifact_id",
    "ingested_at",
];
pub struct PolygonChainlinkOracleRoundsDrain {
    descriptor: DrainDescriptor,
    root: PathBuf,
}
impl PolygonChainlinkOracleRoundsDrain {
    pub fn from_environment() -> Result<Self, DrainExecutionError> {
        let root = PathBuf::from(
            std::env::var("INGESTER_POLYGON_CHAINLINK_ORACLE_ROUNDS_LAKE_ROOT")
                .unwrap_or_else(|_| "/var/lib/polygon-chainlink-oracle-rounds".into()),
        );
        if !root.is_absolute() {
            return Err(invalid(
                "drain_root_invalid",
                "Polygon Chainlink oracle lake root must be absolute",
            ));
        }
        Ok(Self {
            descriptor: retained::descriptor(&SPEC),
            root,
        })
    }
}
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(
        COLUMNS
            .iter()
            .map(|name| Field::new(*name, DataType::Utf8, true))
            .collect::<Vec<_>>(),
    ))
}
fn batch(rows: Vec<PgRow>) -> Result<RecordBatch, DrainExecutionError> {
    let arrays = COLUMNS
        .iter()
        .enumerate()
        .map(|(i, _)| -> Result<ArrayRef, DrainExecutionError> {
            let values = rows
                .iter()
                .map(|r| r.try_get::<Option<String>, _>(i).map_err(db_error))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Arc::new(StringArray::from(values)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(schema(), arrays).map_err(|e| io_error(e.to_string()))
}
#[async_trait]
impl RetainedDrainAdapter for PolygonChainlinkOracleRoundsDrain {
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
        let mut stream=sqlx::query("SELECT source,chain_id::text,feed_proxy_address,aggregator_address,phase_id::text,aggregator_round_id::text,source_timestamp::text,block_timestamp::text,answer_raw::text,price::text,decimals::text,block_number::text,block_hash,transaction_hash,log_index::text,provider_available_at::text,received_at::text,source_payload::text,payload_sha256::text,strategy_key,capture_artifact_id::text,ingested_at::text FROM market_data.polygon_chainlink_btcusd_oracle_rounds WHERE source_timestamp >= $1 AND source_timestamp < $2 ORDER BY source_timestamp,phase_id,aggregator_round_id,block_number,log_index").bind(ch.range_start).bind(ch.range_end).fetch(&c.pool);
        let mut rows = Vec::with_capacity(BATCH_ROWS);
        let mut count = 0;
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
impl DrainWorkerStrategy for PolygonChainlinkOracleRoundsDrain {
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
#[path = "tests/polygon_chainlink_oracle_rounds.rs"]
mod tests;
