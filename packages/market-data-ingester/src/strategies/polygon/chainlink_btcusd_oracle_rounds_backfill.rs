use super::{
    backfill_types::PolygonChainlinkBtcusdOracleRound,
    oracle_rounds_support::{
        PolygonChainlinkOracleConfig, DEFAULT_POLYGON_ARCHIVE_LOG_RPC_URL,
        DEFAULT_POLYGON_CHAINLINK_BTCUSD_PROXY, DEFAULT_POLYGON_RPC_URL,
        POLYGON_CHAINLINK_ORACLE_PROVIDER,
    },
};
use crate::{
    domain::{
        BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
        BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
    },
    strategies::{backfill_support, binance::archive_support::ArchiveCancellation},
};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use sqlx::{Postgres, QueryBuilder};
use std::time::Duration;
pub const STRATEGY_KEY: &str = "polygon_chainlink_btcusd_oracle_rounds_backfill";
const DURABLE_TARGET: &str = "market_data.polygon_chainlink_btcusd_oracle_rounds";
const CANONICAL_STRATEGY_KEY: &str = "polygon_chainlink_btcusd_oracle";
pub struct PolygonChainlinkBtcusdOracleRoundsBackfill {
    descriptor: StrategyDescriptor,
    client: Client,
    config: PolygonChainlinkOracleConfig,
}
impl PolygonChainlinkBtcusdOracleRoundsBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        let descriptor = backfill_support::descriptor(
            STRATEGY_KEY,
            "Polygon Chainlink BTC/USD oracle rounds backfill",
            "Collects historical AnswerUpdated events from the Polygon Chainlink BTC/USD feed",
        )?;
        let config = PolygonChainlinkOracleConfig {
            rpc_url: std::env::var("POLYGON_RPC_URL")
                .unwrap_or_else(|_| DEFAULT_POLYGON_RPC_URL.into()),
            archive_log_rpc_url: std::env::var("POLYGON_ARCHIVE_LOG_RPC_URL")
                .unwrap_or_else(|_| DEFAULT_POLYGON_ARCHIVE_LOG_RPC_URL.into()),
            feed_proxy_address: DEFAULT_POLYGON_CHAINLINK_BTCUSD_PROXY.into(),
            maximum_block_range: 30_000,
        };
        config.validate().map_err(backfill_support::source_error)?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .user_agent("capitonic-ingester-worker/1")
            .build()
            .map_err(backfill_support::source_error)?;
        Ok(Self {
            descriptor,
            client,
            config,
        })
    }
}
#[async_trait]
impl BackfillWorkerStrategy for PolygonChainlinkBtcusdOracleRoundsBackfill {
    fn descriptor(&self) -> &StrategyDescriptor {
        &self.descriptor
    }
    fn validate_request(
        &self,
        request: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        backfill_support::validate_empty_request(&self.descriptor, request)
    }
    fn plan_shards(
        &self,
        request: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        backfill_support::daily_shards(request, self.descriptor.maximum_shards)
    }
    async fn execute_backfill(
        &self,
        context: BackfillContext,
        shard: BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        let logical_key = format!(
            "{}:{}:{}",
            self.config.logical_key(shard.range_start.date_naive()),
            shard.range_start,
            shard.range_end
        );
        if let Some(outcome) =
            backfill_support::completed_outcome(&context, STRATEGY_KEY, &logical_key).await?
        {
            return Ok(outcome);
        }
        let date = shard.range_start.date_naive();
        let source_uri = self.config.source_uri(date);
        let artifact_id = backfill_support::create_artifact(
            &context,
            STRATEGY_KEY,
            "polygon_chainlink_btcusd_oracle_rounds",
            &logical_key,
            POLYGON_CHAINLINK_ORACLE_PROVIDER,
            &source_uri,
            DURABLE_TARGET,
        )
        .await?;
        let cancel = cancellation(&context);
        let fetched = self
            .config
            .fetch_day(&self.client, date, &cancel)
            .await
            .map_err(backfill_support::source_error)?;
        let records = fetched
            .records
            .into_iter()
            .filter(|r| {
                r.source_timestamp >= shard.range_start && r.source_timestamp < shard.range_end
            })
            .collect::<Vec<_>>();
        persist(
            &context,
            artifact_id,
            &records,
            &fetched.sha256,
            fetched.response_bytes,
            shard.range_start,
            shard.range_end,
        )
        .await?;
        Ok(backfill_support::outcome(
            records.len() as i64,
            &shard,
            json!({"provider":POLYGON_CHAINLINK_ORACLE_PROVIDER,"records_verified":records.len(),"start_block":fetched.start_block,"end_block":fetched.end_block,"phase_count":fetched.phase_count}),
        ))
    }
}
fn cancellation(context: &BackfillContext) -> ArchiveCancellation {
    let value = ArchiveCancellation::default();
    let watcher = value.clone();
    let shutdown = context.shutdown.clone();
    tokio::spawn(async move {
        shutdown.cancelled().await;
        watcher.cancel();
    });
    value
}
async fn persist(
    context: &BackfillContext,
    artifact_id: uuid::Uuid,
    records: &[PolygonChainlinkBtcusdOracleRound],
    checksum: &str,
    bytes: u64,
    capture_window_start: chrono::DateTime<chrono::Utc>,
    capture_window_end: chrono::DateTime<chrono::Utc>,
) -> Result<(), BackfillExecutionError> {
    let mut tx = context
        .pool
        .begin()
        .await
        .map_err(backfill_support::database_error)?;
    backfill_support::require_lease(&mut tx, context).await?;
    let received_at = chrono::Utc::now();
    ensure_capture_artifact(
        &mut tx,
        artifact_id,
        records,
        checksum,
        received_at,
        capture_window_start,
        capture_window_end,
    )
    .await?;
    for chunk in records.chunks(1_000) {
        let mut query=QueryBuilder::<Postgres>::new("INSERT INTO market_data.polygon_chainlink_btcusd_oracle_rounds (source,chain_id,feed_proxy_address,aggregator_address,phase_id,aggregator_round_id,source_timestamp,block_timestamp,answer_raw,price,decimals,block_number,block_hash,transaction_hash,log_index,provider_available_at,received_at,source_payload,payload_sha256,strategy_key,capture_artifact_id) ");
        query.push_values(chunk, |mut row, v| {
            row.push_bind("chainlink_polygon_data_feed")
                .push_bind(v.chain_id)
                .push_bind(&v.feed_proxy_address)
                .push_bind(&v.aggregator_address)
                .push_bind(v.phase_id)
                .push_bind(v.aggregator_round_id)
                .push_bind(v.source_timestamp)
                .push_bind(v.block_timestamp)
                .push_bind(v.answer_raw)
                .push_bind(v.price)
                .push_bind(v.decimals)
                .push_bind(v.block_number)
                .push_bind(&v.block_hash)
                .push_bind(&v.transaction_hash)
                .push_bind(v.log_index)
                .push_bind(v.block_timestamp)
                .push_bind(received_at)
                .push_bind(v.canonical_source_payload())
                .push_bind(v.canonical_payload_sha256())
                .push_bind(CANONICAL_STRATEGY_KEY)
                .push_bind(artifact_id);
        });
        query.push(" ON CONFLICT DO NOTHING");
        query
            .build()
            .execute(&mut *tx)
            .await
            .map_err(backfill_support::database_error)?;
    }
    let minimum = records.first().map(|r| r.source_timestamp);
    let maximum = records.last().map(|r| r.source_timestamp);
    backfill_support::complete_artifact(
        &mut tx,
        context,
        backfill_support::ArtifactCompletion {
            artifact_id,
            checksum,
            byte_size: bytes,
            record_count: records.len() as i64,
            minimum,
            maximum,
            metadata: json!({"provider":POLYGON_CHAINLINK_ORACLE_PROVIDER}),
        },
    )
    .await?;
    tx.commit().await.map_err(backfill_support::database_error)
}

async fn ensure_capture_artifact(
    tx: &mut sqlx::Transaction<'_, Postgres>,
    artifact_id: uuid::Uuid,
    records: &[PolygonChainlinkBtcusdOracleRound],
    checksum: &str,
    received_at: chrono::DateTime<chrono::Utc>,
    capture_window_start: chrono::DateTime<chrono::Utc>,
    capture_window_end: chrono::DateTime<chrono::Utc>,
) -> Result<(), BackfillExecutionError> {
    sqlx::query(
        r#"
      INSERT INTO ingester.capture_artifacts (
        artifact_id, strategy_key, profile_generation, config_schema_version,
        config_sha256, config_snapshot, capture_window_start, capture_window_end,
        minimum_source_timestamp, maximum_source_timestamp,
        minimum_received_at, maximum_received_at, record_count, content_sha256,
        status, created_at, updated_at, completed_at
      )
      SELECT $1, $2, profile.desired_generation, profile.config_schema_version,
        encode(digest(convert_to(profile.config::text, 'UTF8'), 'sha256'), 'hex'),
        profile.config, $3, $4, $5, $6, $7, $7, $8, $9,
        'completed', $7, $7, $7
      FROM ingester.profiles profile WHERE profile.strategy_key = $2
      ON CONFLICT DO NOTHING
    "#,
    )
    .bind(artifact_id)
    .bind(CANONICAL_STRATEGY_KEY)
    .bind(capture_window_start)
    .bind(capture_window_end)
    .bind(records.first().map(|row| row.source_timestamp))
    .bind(records.last().map(|row| row.source_timestamp))
    .bind(received_at)
    .bind(
        i64::try_from(records.len())
            .map_err(|_| backfill_support::integrity("record count overflow"))?,
    )
    .bind(checksum)
    .execute(&mut **tx)
    .await
    .map_err(backfill_support::database_error)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::StrategyCapability;
    use chrono::{TimeZone, Utc};
    #[test]
    fn contract_is_backfill_only_and_one_hour_is_one_shard() {
        let s = PolygonChainlinkBtcusdOracleRoundsBackfill::new().unwrap();
        assert_eq!(
            s.descriptor().capabilities,
            vec![StrategyCapability::Backfill]
        );
        let start = Utc.with_ymd_and_hms(2026, 8, 1, 1, 0, 0).unwrap();
        let r = backfill_support::request(STRATEGY_KEY, start, start + chrono::Duration::hours(1));
        assert_eq!(
            s.plan_shards(&s.validate_request(&r).unwrap())
                .unwrap()
                .len(),
            1
        );
    }
}
