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
const DURABLE_TARGET: &str = "market_data.binance_futures_btcusdt_open_interest";
const CANONICAL_STRATEGY_KEY: &str = "binance_futures_btcusdt_open_interest";

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
            "binance_futures_btcusdt_open_interest",
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
            shard.range_start,
            shard.range_end,
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
    capture_window_start: chrono::DateTime<chrono::Utc>,
    capture_window_end: chrono::DateTime<chrono::Utc>,
) -> Result<(), BackfillExecutionError> {
    let mut tx = context
        .pool
        .begin()
        .await
        .map_err(backfill_support::database_error)?;
    backfill_support::require_lease(&mut tx, context).await?;
    let capture_artifact_id = ensure_capture_artifact(
        &mut tx,
        artifact_id,
        records,
        checksum,
        capture_window_start,
        capture_window_end,
    )
    .await?;
    let received_at = chrono::Utc::now();
    for chunk in records.chunks(1_000) {
        let mut query = QueryBuilder::<Postgres>::new("INSERT INTO market_data.binance_futures_btcusdt_open_interest (source,symbol,source_timestamp,period_seconds,sum_open_interest,sum_open_interest_value,cmc_circulating_supply,provider_available_at,received_at,source_payload,payload_sha256,strategy_key,capture_artifact_id) ");
        query.push_values(chunk, |mut row, value| {
            row.push_bind("binance_usd_m_futures")
                .push_bind(&value.symbol)
                .push_bind(value.source_timestamp)
                .push_bind(value.period_seconds)
                .push_bind(value.sum_open_interest)
                .push_bind(value.sum_open_interest_value)
                .push_bind(value.cmc_circulating_supply)
                .push_bind(Option::<chrono::DateTime<chrono::Utc>>::None)
                .push_bind(received_at)
                .push_bind(value.canonical_source_payload())
                .push_bind(value.canonical_payload_sha256())
                .push_bind(CANONICAL_STRATEGY_KEY)
                .push_bind(capture_artifact_id);
        });
        query.push(" ON CONFLICT (source_timestamp,symbol,period_seconds) DO NOTHING");
        query
            .build()
            .execute(&mut *tx)
            .await
            .map_err(backfill_support::database_error)?;
    }
    for record in records {
        let stored = sqlx::query_as::<
            _,
            (
                rust_decimal::Decimal,
                rust_decimal::Decimal,
                Option<rust_decimal::Decimal>,
                String,
            ),
        >(
            r#"
            SELECT sum_open_interest, sum_open_interest_value,
                   cmc_circulating_supply, payload_sha256
            FROM market_data.binance_futures_btcusdt_open_interest
            WHERE source_timestamp = $1 AND symbol = $2 AND period_seconds = $3
            "#,
        )
        .bind(record.source_timestamp)
        .bind(&record.symbol)
        .bind(record.period_seconds)
        .fetch_one(&mut *tx)
        .await
        .map_err(backfill_support::database_error)?;
        if stored.0 != record.sum_open_interest
            || stored.1 != record.sum_open_interest_value
            || stored.2 != record.cmc_circulating_supply
            || stored.3 != record.canonical_payload_sha256()
        {
            return Err(backfill_support::integrity(format!(
                "immutable canonical open-interest conflict for {}:{}",
                record.symbol, record.source_timestamp
            )));
        }
    }
    let minimum = records.first().map(|row| row.source_timestamp);
    let maximum = records.last().map(|row| row.source_timestamp);
    backfill_support::complete_artifact(&mut tx, context, ArtifactCompletion { artifact_id, checksum, byte_size: response_bytes, record_count: i64::try_from(records.len()).map_err(|_| backfill_support::integrity("record count overflow"))?, minimum, maximum, metadata: json!({"period_seconds": 300, "source_day": minimum.map(|value| value.date_naive())}) }).await?;
    tx.commit()
        .await
        .map_err(backfill_support::database_error)?;
    Ok(())
}

async fn ensure_capture_artifact(
    tx: &mut sqlx::Transaction<'_, Postgres>,
    artifact_id: uuid::Uuid,
    records: &[BinanceBtcusdtOpenInterestRecord],
    checksum: &str,
    capture_window_start: chrono::DateTime<chrono::Utc>,
    capture_window_end: chrono::DateTime<chrono::Utc>,
) -> Result<uuid::Uuid, BackfillExecutionError> {
    let minimum = records.first().map(|row| row.source_timestamp);
    let maximum = records.last().map(|row| row.source_timestamp);
    let received_at = chrono::Utc::now();
    sqlx::query_scalar::<_, uuid::Uuid>(
        r#"
        INSERT INTO ingester.capture_artifacts (
          artifact_id, strategy_key, profile_generation, config_schema_version,
          config_sha256, config_snapshot, capture_window_start, capture_window_end,
          minimum_source_timestamp, maximum_source_timestamp,
          minimum_received_at, maximum_received_at, record_count, content_sha256,
          status, created_at, updated_at, completed_at
        )
        SELECT
          $1, $2, profile.desired_generation, profile.config_schema_version,
          encode(digest(convert_to(profile.config::text, 'UTF8'), 'sha256'), 'hex'),
          profile.config, $3, $4, $5, $6, $7, $7, $8, $9,
          'completed', $7, $7, $7
        FROM ingester.profiles profile
        WHERE profile.strategy_key = $2
        RETURNING artifact_id
        "#,
    )
    .bind(artifact_id)
    .bind(CANONICAL_STRATEGY_KEY)
    .bind(capture_window_start)
    .bind(capture_window_end)
    .bind(minimum)
    .bind(maximum)
    .bind(received_at)
    .bind(
        i64::try_from(records.len())
            .map_err(|_| backfill_support::integrity("record count overflow"))?,
    )
    .bind(checksum)
    .fetch_one(&mut **tx)
    .await
    .map_err(backfill_support::database_error)
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
