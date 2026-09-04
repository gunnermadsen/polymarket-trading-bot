use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use sqlx::{Postgres, QueryBuilder};

use crate::{
    domain::{
        BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
        BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
    },
    strategies::{backfill_support, binance::archive_support::ArchiveCancellation},
};

use super::{
    backfill_types::ChainlinkBtcusdArchiveTick,
    reference_ticks_support::{
        ChainlinkArchiveConfig, ChainlinkCredentials, CHAINLINK_ARCHIVE_PROVIDER,
        DEFAULT_CHAINLINK_BTCUSD_FEED_ID, DEFAULT_CHAINLINK_REST_URL,
    },
};

pub const STRATEGY_KEY: &str = "chainlink_btcusd_reference_ticks_backfill";
const DURABLE_TARGET: &str = "market_data.chainlink_btcusd_reference_prices";

pub struct ChainlinkBtcusdReferenceTicksBackfill {
    descriptor: StrategyDescriptor,
    client: Client,
    config: ChainlinkArchiveConfig,
}

impl ChainlinkBtcusdReferenceTicksBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        let descriptor = backfill_support::descriptor(
            STRATEGY_KEY,
            "Chainlink BTC/USD reference ticks backfill",
            "Collects signed historical Chainlink Data Streams v3 BTC/USD reports",
        )?;
        let credentials = credentials(
            "MARKET_DATA_INGESTER_CHAINLINK_DATA_STREAMS_API_KEY",
            "MARKET_DATA_INGESTER_CHAINLINK_DATA_STREAMS_API_SECRET",
        );
        let config = ChainlinkArchiveConfig {
            rest_url: std::env::var("CHAINLINK_DATA_STREAMS_REST_URL")
                .unwrap_or_else(|_| DEFAULT_CHAINLINK_REST_URL.into()),
            feed_id: DEFAULT_CHAINLINK_BTCUSD_FEED_ID.into(),
            page_limit: 1_000,
            credentials,
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

fn credentials(key: &str, secret: &str) -> Option<ChainlinkCredentials> {
    let api_key = std::env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())?;
    let api_secret = std::env::var(secret)
        .ok()
        .filter(|value| !value.trim().is_empty())?;
    Some(ChainlinkCredentials {
        api_key,
        api_secret,
    })
}

#[async_trait]
impl BackfillWorkerStrategy for ChainlinkBtcusdReferenceTicksBackfill {
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
            "chainlink_btcusd_reference_ticks",
            &logical_key,
            CHAINLINK_ARCHIVE_PROVIDER,
            &source_uri,
            DURABLE_TARGET,
        )
        .await?;
        let cancellation = cancellation(&context);
        let fetched = self
            .config
            .fetch_day(&self.client, date, &cancellation)
            .await
            .map_err(backfill_support::source_error)?;
        let records = fetched
            .records
            .into_iter()
            .filter(|row| {
                row.source_timestamp >= shard.range_start && row.source_timestamp < shard.range_end
            })
            .collect::<Vec<_>>();
        persist(
            &context,
            artifact_id,
            &records,
            &fetched.sha256,
            fetched.response_bytes,
        )
        .await?;
        Ok(backfill_support::outcome(
            records.len() as i64,
            &shard,
            json!({"provider": CHAINLINK_ARCHIVE_PROVIDER, "records_verified": records.len()}),
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
    records: &[ChainlinkBtcusdArchiveTick],
    checksum: &str,
    bytes: u64,
) -> Result<(), BackfillExecutionError> {
    let mut tx = context
        .pool
        .begin()
        .await
        .map_err(backfill_support::database_error)?;
    backfill_support::require_lease(&mut tx, context).await?;
    for chunk in records.chunks(1_000) {
        let mut query = QueryBuilder::<Postgres>::new("INSERT INTO market_data.chainlink_btcusd_reference_prices (source,feed_id,source_timestamp,valid_from_timestamp,provider_available_at,received_at,price,bid,ask,report_sha256,payload_sha256,strategy_key,capture_artifact_id,backfill_artifact_id,report_hash_kind) ");
        query.push_values(chunk, |mut row, value| {
            row.push_bind("chainlink_data_streams")
                .push_bind(&value.feed_id)
                .push_bind(value.source_timestamp)
                .push_bind(value.valid_from_timestamp)
                .push_bind(value.source_timestamp)
                .push_bind(chrono::Utc::now())
                .push_bind(value.price)
                .push_bind(value.bid)
                .push_bind(value.ask)
                .push_bind(&value.report_sha256)
                .push_bind(&value.report_sha256)
                .push_bind(STRATEGY_KEY)
                .push_bind(Option::<uuid::Uuid>::None)
                .push_bind(artifact_id)
                .push_bind("signed_report");
        });
        query.push(" ON CONFLICT (feed_id,source_timestamp,report_sha256) DO NOTHING");
        query
            .build()
            .execute(&mut *tx)
            .await
            .map_err(backfill_support::database_error)?;
    }
    let minimum = records.first().map(|row| row.source_timestamp);
    let maximum = records.last().map(|row| row.source_timestamp);
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
            metadata: json!({"provider": CHAINLINK_ARCHIVE_PROVIDER}),
        },
    )
    .await?;
    tx.commit().await.map_err(backfill_support::database_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::StrategyCapability;
    use chrono::{TimeZone, Utc};
    #[test]
    fn contract_is_backfill_only_and_one_hour_is_one_shard() {
        let strategy = ChainlinkBtcusdReferenceTicksBackfill::new().unwrap();
        assert_eq!(
            strategy.descriptor().capabilities,
            vec![StrategyCapability::Backfill]
        );
        let start = Utc.with_ymd_and_hms(2026, 8, 1, 1, 0, 0).unwrap();
        let request =
            backfill_support::request(STRATEGY_KEY, start, start + chrono::Duration::hours(1));
        let shards = strategy
            .plan_shards(&strategy.validate_request(&request).unwrap())
            .unwrap();
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].range_start, start);
    }
}
