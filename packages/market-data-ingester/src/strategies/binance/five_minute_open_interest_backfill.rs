use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use sqlx::{Postgres, QueryBuilder};

use crate::domain::{
    BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
    BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
};
use crate::strategies::backfill_support::{self, ArtifactCompletion};

use super::{
    archive_support::ArchiveCancellation,
    open_interest_support::{
        BinanceOpenInterestConfig, BINANCE_OPEN_INTEREST_PROVIDER,
        DEFAULT_BINANCE_FUTURES_DATA_BASE_URL, DEFAULT_BINANCE_OPEN_INTEREST_SYMBOL,
    },
    types::BinanceBtcusdtOpenInterestRecord,
};

pub const STRATEGY_KEY: &str = "binance_futures_btcusdt_five_minute_open_interest_backfill";
const DURABLE_TARGET: &str = "polymarket.binance_btcusdt_five_minute_open_interest";

pub struct BinanceFuturesFiveMinuteOpenInterestBackfill {
    descriptor: StrategyDescriptor,
    client: Client,
    config: BinanceOpenInterestConfig,
}

impl BinanceFuturesFiveMinuteOpenInterestBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        let descriptor = backfill_support::descriptor(
            STRATEGY_KEY,
            "Binance Futures BTCUSDT five-minute open-interest backfill",
            "Collects historical Binance Futures BTCUSDT five-minute open interest",
        )?;
        let config = BinanceOpenInterestConfig {
            base_url: std::env::var("BINANCE_FUTURES_DATA_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_BINANCE_FUTURES_DATA_BASE_URL.to_owned()),
            symbol: DEFAULT_BINANCE_OPEN_INTEREST_SYMBOL.to_owned(),
        };
        config.validate().map_err(backfill_support::source_error)?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
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
impl BackfillWorkerStrategy for BinanceFuturesFiveMinuteOpenInterestBackfill {
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
        let date = shard.range_start.date_naive();
        let logical_key = format!(
            "binance-open-interest:BTCUSDT:5m:{}:{}",
            shard.range_start, shard.range_end
        );
        if let Some(outcome) =
            backfill_support::completed_outcome(&context, STRATEGY_KEY, &logical_key).await?
        {
            return Ok(outcome);
        }
        let source_uri = self.config.source_uri(date);
        let artifact_id = backfill_support::create_artifact(
            &context,
            STRATEGY_KEY,
            "binance_btcusdt_five_minute_open_interest",
            &logical_key,
            BINANCE_OPEN_INTEREST_PROVIDER,
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
            i64::try_from(records.len())
                .map_err(|_| backfill_support::integrity("record count overflow"))?,
            &shard,
            json!({
                "provider": BINANCE_OPEN_INTEREST_PROVIDER, "symbol": DEFAULT_BINANCE_OPEN_INTEREST_SYMBOL,
                "period_seconds": 300, "records_verified": records.len(),
            }),
        ))
    }
}

fn cancellation(context: &BackfillContext) -> ArchiveCancellation {
    let archive = ArchiveCancellation::default();
    let cloned = archive.clone();
    let shutdown = context.shutdown.clone();
    tokio::spawn(async move {
        shutdown.cancelled().await;
        cloned.cancel();
    });
    archive
}

async fn persist(
    context: &BackfillContext,
    artifact_id: uuid::Uuid,
    records: &[BinanceBtcusdtOpenInterestRecord],
    checksum: &str,
    response_bytes: u64,
) -> Result<(), BackfillExecutionError> {
    let mut tx = context
        .pool
        .begin()
        .await
        .map_err(backfill_support::database_error)?;
    backfill_support::require_lease(&mut tx, context).await?;
    for chunk in records.chunks(1_000) {
        let mut query = QueryBuilder::<Postgres>::new("INSERT INTO polymarket.binance_btcusdt_five_minute_open_interest (symbol,source_timestamp,period_seconds,sum_open_interest,sum_open_interest_value,cmc_circulating_supply,artifact_id) ");
        query.push_values(chunk, |mut row, value| {
            row.push_bind(&value.symbol)
                .push_bind(value.source_timestamp)
                .push_bind(value.period_seconds)
                .push_bind(value.sum_open_interest)
                .push_bind(value.sum_open_interest_value)
                .push_bind(value.cmc_circulating_supply)
                .push_bind(artifact_id);
        });
        query.push(" ON CONFLICT (symbol,source_timestamp) DO NOTHING");
        query
            .build()
            .execute(&mut *tx)
            .await
            .map_err(backfill_support::database_error)?;
    }
    let minimum = records.first().map(|row| row.source_timestamp);
    let maximum = records.last().map(|row| row.source_timestamp);
    backfill_support::complete_artifact(&mut tx, context, ArtifactCompletion { artifact_id, checksum, byte_size: response_bytes, record_count: i64::try_from(records.len()).map_err(|_| backfill_support::integrity("record count overflow"))?, minimum, maximum, metadata: json!({"period_seconds": 300, "source_day": minimum.map(|value| value.date_naive())}) }).await?;
    tx.commit()
        .await
        .map_err(backfill_support::database_error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{BackfillWorkerStrategy, StrategyCapability};
    use chrono::{TimeZone, Utc};

    #[test]
    fn contract_is_backfill_only_and_one_hour_is_one_shard() {
        let strategy = BinanceFuturesFiveMinuteOpenInterestBackfill::new().unwrap();
        assert_eq!(
            strategy.descriptor().capabilities,
            vec![StrategyCapability::Backfill]
        );
        let start = Utc.with_ymd_and_hms(2026, 7, 1, 1, 0, 0).unwrap();
        let validated = strategy
            .validate_request(&backfill_support::request(
                STRATEGY_KEY,
                start,
                start + chrono::Duration::hours(1),
            ))
            .unwrap();
        let shards = strategy.plan_shards(&validated).unwrap();
        assert_eq!(shards.len(), 1);
        assert_eq!(
            (shards[0].range_start, shards[0].range_end),
            (start, start + chrono::Duration::hours(1))
        );
    }
}
