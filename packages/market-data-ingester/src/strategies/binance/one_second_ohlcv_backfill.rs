use std::{path::PathBuf, time::Duration};

use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use sqlx::{Postgres, QueryBuilder};

use crate::domain::{
    BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
    BackfillWorkerStrategy, StrategyDescriptor, ValidatedBackfillRequest,
};

use super::{
    archive_support::{
        self, ArchiveCancellation, ArchiveDownloadLimits, BinanceArchiveKind, BinanceArchiveSpec,
        BINANCE_ARCHIVE_PROVIDER,
    },
    backfill_support::{self, ArtifactCompletion},
    types::BinanceOneSecondKlineRecord,
};

pub const STRATEGY_KEY: &str = "binance_spot_btcusdt_one_second_ohlcv_backfill";
const DEFAULT_ARCHIVE_URL: &str = "https://data.binance.vision";
const DURABLE_TARGET: &str = "polymarket.binance_one_second_klines";

pub struct BinanceSpotOneSecondOhlcvBackfill {
    descriptor: StrategyDescriptor,
    client: Client,
    archive_url: String,
    cache_directory: PathBuf,
}

impl BinanceSpotOneSecondOhlcvBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        let descriptor = backfill_support::descriptor(
            STRATEGY_KEY,
            "Binance Spot BTCUSDT one-second OHLCV backfill",
            "Collects historical Binance Spot BTCUSDT one-second klines",
        )?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(120))
            .user_agent("capitonic-ingester-worker/1")
            .build()
            .map_err(backfill_support::source_error)?;
        Ok(Self {
            descriptor,
            client,
            archive_url: std::env::var("BINANCE_ARCHIVE_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_ARCHIVE_URL.to_owned()),
            cache_directory: std::env::var("INGESTER_BINANCE_CACHE_DIRECTORY")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/tmp/ingester-binance-cache")),
        })
    }
}

#[async_trait]
impl BackfillWorkerStrategy for BinanceSpotOneSecondOhlcvBackfill {
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
        let spec = BinanceArchiveSpec::new(
            &self.archive_url,
            BinanceArchiveKind::OneSecondKlines,
            shard.range_start.date_naive(),
        );
        let logical_key = format!(
            "{}:{}:{}",
            spec.logical_key, shard.range_start, shard.range_end
        );
        if let Some(outcome) =
            backfill_support::completed_outcome(&context, STRATEGY_KEY, &logical_key).await?
        {
            return Ok(outcome);
        }
        let artifact_id = backfill_support::create_artifact(
            &context,
            STRATEGY_KEY,
            "binance_btcusdt_one_second_klines",
            &logical_key,
            BINANCE_ARCHIVE_PROVIDER,
            &spec.source_uri,
            DURABLE_TARGET,
        )
        .await?;
        let cancellation = cancellation(&context);
        let checksum = archive_support::fetch_expected_checksum_with_cancellation(
            &self.client,
            &spec,
            &cancellation,
        )
        .await
        .map_err(backfill_support::source_error)?;
        let downloaded = archive_support::download_archive_with_cancellation(
            &self.client,
            &spec,
            &checksum,
            &self.cache_directory,
            &ArchiveDownloadLimits::default(),
            &cancellation,
        )
        .await
        .map_err(backfill_support::source_error)?;
        let (mut batches, parser) = archive_support::spawn_one_second_kline_parser_with_control(
            downloaded.path.clone(),
            spec,
            1_000,
            128 * 1024 * 1024 * 1024,
            cancellation,
        )
        .map_err(backfill_support::source_error)?;
        let mut records = Vec::new();
        while let Some(batch) = batches.recv().await {
            records.extend(batch.into_iter().filter(|row| {
                row.open_timestamp >= shard.range_start && row.open_timestamp < shard.range_end
            }));
        }
        let parsed = parser
            .await
            .map_err(|error| backfill_support::source_error(error.to_string()))?
            .map_err(backfill_support::source_error)?;
        persist(
            &context,
            artifact_id,
            &records,
            &downloaded.sha256,
            downloaded.compressed_bytes,
        )
        .await?;
        Ok(backfill_support::outcome(
            i64::try_from(records.len())
                .map_err(|_| backfill_support::integrity("record count overflow"))?,
            &shard,
            json!({
                "provider": BINANCE_ARCHIVE_PROVIDER, "source_day_records": parsed.records, "records_verified": records.len(), "reused_cache": downloaded.reused_cache,
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
    records: &[BinanceOneSecondKlineRecord],
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
        let mut query = QueryBuilder::<Postgres>::new("INSERT INTO polymarket.binance_one_second_klines (symbol,open_timestamp,close_timestamp,open_price,high_price,low_price,close_price,base_volume,quote_volume,trade_count,taker_buy_base_volume,taker_buy_quote_volume,artifact_id) ");
        query.push_values(chunk, |mut row, value| {
            row.push_bind(&value.symbol)
                .push_bind(value.open_timestamp)
                .push_bind(value.close_timestamp)
                .push_bind(value.open_price)
                .push_bind(value.high_price)
                .push_bind(value.low_price)
                .push_bind(value.close_price)
                .push_bind(value.base_volume)
                .push_bind(value.quote_volume)
                .push_bind(value.trade_count)
                .push_bind(value.taker_buy_base_volume)
                .push_bind(value.taker_buy_quote_volume)
                .push_bind(artifact_id);
        });
        query.push(" ON CONFLICT (symbol,open_timestamp) DO NOTHING");
        query
            .build()
            .execute(&mut *tx)
            .await
            .map_err(backfill_support::database_error)?;
    }
    let count = i64::try_from(records.len())
        .map_err(|_| backfill_support::integrity("record count overflow"))?;
    backfill_support::complete_artifact(
        &mut tx,
        context,
        ArtifactCompletion {
            artifact_id,
            checksum,
            byte_size: bytes,
            record_count: count,
            minimum: records.first().map(|row| row.open_timestamp),
            maximum: records.last().map(|row| row.open_timestamp),
            metadata: json!({"interval_seconds": 1}),
        },
    )
    .await?;
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
    fn contract_is_backfill_only_and_preserves_requested_hour() {
        let strategy = BinanceSpotOneSecondOhlcvBackfill::new().unwrap();
        assert_eq!(
            strategy.descriptor().capabilities,
            vec![StrategyCapability::Backfill]
        );
        let start = Utc.with_ymd_and_hms(2026, 7, 1, 2, 0, 0).unwrap();
        let request = strategy
            .validate_request(&backfill_support::request(
                STRATEGY_KEY,
                start,
                start + chrono::Duration::hours(1),
            ))
            .unwrap();
        let shards = strategy.plan_shards(&request).unwrap();
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].range_start, start);
        assert_eq!(shards[0].range_end, start + chrono::Duration::hours(1));
    }
}
