use super::{
    backfill_types::ChainlinkBtcusdOneMinuteCandle,
    one_minute_candles_support::{
        ChainlinkCandlestickConfig, ChainlinkCandlestickCredentials,
        CHAINLINK_CANDLESTICK_PROVIDER, DEFAULT_CHAINLINK_CANDLESTICK_BASE_URL,
        DEFAULT_CHAINLINK_CANDLESTICK_SYMBOL,
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

pub const STRATEGY_KEY: &str = "chainlink_btcusd_one_minute_candles_backfill";
const DURABLE_TARGET: &str = "polymarket.chainlink_btcusd_one_minute_candles";
pub struct ChainlinkBtcusdOneMinuteCandlesBackfill {
    descriptor: StrategyDescriptor,
    client: Client,
    config: ChainlinkCandlestickConfig,
}

impl ChainlinkBtcusdOneMinuteCandlesBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        let descriptor = backfill_support::descriptor(
            STRATEGY_KEY,
            "Chainlink BTC/USD one-minute candles backfill",
            "Collects historical BTC/USD candles from the Chainlink Candlestick API",
        )?;
        let login = std::env::var("MARKET_DATA_INGESTER_CHAINLINK_DATA_STREAMS_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty());
        let api_key =
            std::env::var("MARKET_DATA_INGESTER_CHAINLINK_DATA_STREAMS_CANDLESTICK_API_KEY")
                .ok()
                .filter(|v| !v.trim().is_empty());
        let credentials = login
            .zip(api_key)
            .map(|(login, api_key)| ChainlinkCandlestickCredentials { login, api_key });
        let config = ChainlinkCandlestickConfig {
            base_url: std::env::var("CHAINLINK_CANDLESTICK_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_CHAINLINK_CANDLESTICK_BASE_URL.into()),
            symbol: DEFAULT_CHAINLINK_CANDLESTICK_SYMBOL.into(),
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

#[async_trait]
impl BackfillWorkerStrategy for ChainlinkBtcusdOneMinuteCandlesBackfill {
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
            "chainlink_btcusd_one_minute_candles",
            &logical_key,
            CHAINLINK_CANDLESTICK_PROVIDER,
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
            .filter(|row| {
                row.open_timestamp >= shard.range_start && row.open_timestamp < shard.range_end
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
            json!({"provider":CHAINLINK_CANDLESTICK_PROVIDER,"records_verified":records.len()}),
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
    records: &[ChainlinkBtcusdOneMinuteCandle],
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
        let mut query=QueryBuilder::<Postgres>::new("INSERT INTO polymarket.chainlink_btcusd_one_minute_candles (symbol,open_timestamp,close_timestamp,open_price,high_price,low_price,close_price,volume,volume_supported,artifact_id) ");
        query.push_values(chunk, |mut row, value| {
            row.push_bind(&value.symbol)
                .push_bind(value.open_timestamp)
                .push_bind(value.close_timestamp)
                .push_bind(value.open_price)
                .push_bind(value.high_price)
                .push_bind(value.low_price)
                .push_bind(value.close_price)
                .push_bind(value.volume)
                .push_bind(value.volume_supported)
                .push_bind(artifact_id);
        });
        query.push(" ON CONFLICT (symbol,open_timestamp) DO NOTHING");
        query
            .build()
            .execute(&mut *tx)
            .await
            .map_err(backfill_support::database_error)?;
    }
    let minimum = records.first().map(|r| r.open_timestamp);
    let maximum = records.last().map(|r| r.open_timestamp);
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
            metadata: json!({"provider":CHAINLINK_CANDLESTICK_PROVIDER,"resolution":"1m"}),
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
        let strategy = ChainlinkBtcusdOneMinuteCandlesBackfill::new().unwrap();
        assert_eq!(
            strategy.descriptor().capabilities,
            vec![StrategyCapability::Backfill]
        );
        let start = Utc.with_ymd_and_hms(2026, 8, 1, 1, 0, 0).unwrap();
        let request =
            backfill_support::request(STRATEGY_KEY, start, start + chrono::Duration::hours(1));
        assert_eq!(
            strategy
                .plan_shards(&strategy.validate_request(&request).unwrap())
                .unwrap()
                .len(),
            1
        );
    }
}
