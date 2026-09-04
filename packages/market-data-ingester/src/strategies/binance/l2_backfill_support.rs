use std::{path::PathBuf, time::Duration};

use chrono::Timelike;
use reqwest::Client;
use serde_json::json;
use sha2::{Digest, Sha256};
use sqlx::{Postgres, QueryBuilder};
use uuid::Uuid;

use crate::domain::{BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillShard};

use super::{
    archive_support::ArchiveCancellation,
    backfill_support::{self, ArtifactCompletion},
    l2_support::{
        self, CryptoHftBinanceL2Config, CryptoHftBinanceMarket, CryptoHftDayParseRequest,
        CryptoHftHourlySpec, CRYPTOHFT_ARCHIVE_PROVIDER,
    },
    types::BinanceL2OneSecondFeature,
};

const BATCH_ROWS: usize = 1_000;

pub async fn execute(
    strategy_key: &str,
    legacy_ingester_key: &str,
    durable_target: &str,
    market: CryptoHftBinanceMarket,
    context: BackfillContext,
    shard: BackfillShard,
) -> Result<BackfillOutcome, BackfillExecutionError> {
    let logical_key = format!(
        "{}:{}:{}",
        l2_support::day_artifact_logical_key_for_market(market, shard.range_start.date_naive()),
        shard.range_start,
        shard.range_end
    );
    if let Some(outcome) =
        backfill_support::completed_outcome(&context, strategy_key, &logical_key).await?
    {
        return Ok(outcome);
    }
    let config = config()?;
    config
        .preflight()
        .await
        .map_err(backfill_support::source_error)?;
    let first =
        CryptoHftHourlySpec::new_for_market(&config, market, shard.range_start.date_naive(), 0)
            .map_err(backfill_support::source_error)?;
    let artifact_id = backfill_support::create_artifact(
        &context,
        strategy_key,
        legacy_ingester_key,
        &logical_key,
        CRYPTOHFT_ARCHIVE_PROVIDER,
        &first.source_uri,
        durable_target,
    )
    .await?;
    let cancellation = cancellation(&context);
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(120))
        .user_agent("capitonic-ingester-worker/1")
        .build()
        .map_err(backfill_support::source_error)?;
    let day = shard.range_start.date_naive();
    let mut target_archives = Vec::with_capacity(24);
    let mut compressed_bytes = 0u64;
    for hour in 0..24u8 {
        let spec = CryptoHftHourlySpec::new_for_market(&config, market, day, hour)
            .map_err(backfill_support::source_error)?;
        let manifest =
            l2_support::download_hour(&client, &context.pool, &config, &spec, &cancellation)
                .await
                .map_err(backfill_support::source_error)?;
        compressed_bytes = compressed_bytes.saturating_add(manifest.compressed_bytes);
        target_archives.push(manifest);
    }
    let mut context_hour = shard
        .range_start
        .date_naive()
        .and_hms_opt(0, 0, 0)
        .expect("UTC midnight is valid")
        .and_utc()
        - chrono::Duration::hours(1);
    let mut context_archives = Vec::new();
    for _ in 0..l2_support::MAX_CONTEXT_LOOKBACK_HOURS {
        if context_hour.timestamp() < l2_support::CRYPTOHFT_EARLIEST_CONTEXT_HOUR_EPOCH {
            break;
        }
        let spec = CryptoHftHourlySpec::new_for_market(
            &config,
            market,
            context_hour.date_naive(),
            u8::try_from(context_hour.hour())
                .map_err(|_| backfill_support::integrity("invalid context hour"))?,
        )
        .map_err(backfill_support::source_error)?;
        let manifest =
            l2_support::download_hour(&client, &context.pool, &config, &spec, &cancellation)
                .await
                .map_err(backfill_support::source_error)?;
        compressed_bytes = compressed_bytes.saturating_add(manifest.compressed_bytes);
        let has_bootstrap = manifest.validated_snapshot_events > 0;
        context_archives.push(manifest);
        if has_bootstrap {
            break;
        }
        context_hour -= chrono::Duration::hours(1);
    }
    if !context_archives
        .iter()
        .any(|manifest| manifest.validated_snapshot_events > 0)
    {
        return Err(backfill_support::integrity(
            "CryptoHFT context did not contain a validated full-depth snapshot",
        ));
    }
    let source_checksum = combined_checksum(context_archives.iter().chain(target_archives.iter()));
    let (mut batches, parser) = l2_support::spawn_day_parser(
        config,
        CryptoHftDayParseRequest {
            target_date: day,
            context_archives,
            target_archives,
            output_batch_rows: BATCH_ROWS,
            cancellation,
        },
    );
    let mut selected = Vec::new();
    while let Some(batch) = batches.recv().await {
        selected.extend(batch.into_iter().filter(|row| {
            row.second_start >= shard.range_start && row.second_start < shard.range_end
        }));
    }
    let parsed = parser
        .await
        .map_err(|error| backfill_support::source_error(error.to_string()))?
        .map_err(backfill_support::source_error)?;
    persist(&context, artifact_id, durable_target, &selected, &source_checksum, compressed_bytes, json!({
        "market": match market { CryptoHftBinanceMarket::Futures => "futures", CryptoHftBinanceMarket::Spot => "spot" },
        "materialization_contract": market.materialization_contract(), "feature_schema_version": market.feature_schema_version(),
        "source_raw_rows": parsed.raw_rows, "source_logical_events": parsed.logical_events, "source_emitted_feature_rows": parsed.emitted_feature_rows,
    })).await?;
    Ok(backfill_support::outcome(
        i64::try_from(selected.len())
            .map_err(|_| backfill_support::integrity("record count overflow"))?,
        &shard,
        json!({
            "provider": CRYPTOHFT_ARCHIVE_PROVIDER, "records_verified": selected.len(), "feature_schema_version": market.feature_schema_version(),
        }),
    ))
}

fn config() -> Result<CryptoHftBinanceL2Config, BackfillExecutionError> {
    let root = std::env::var("INGESTER_CRYPTOHFT_ARCHIVE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/var/lib/ingester/cryptohft"));
    let temporary = std::env::var("INGESTER_CRYPTOHFT_TEMPORARY_DIRECTORY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join("tmp"));
    let mut config = CryptoHftBinanceL2Config::new(root, temporary);
    config.base_url = std::env::var("CRYPTOHFT_BASE_URL")
        .unwrap_or_else(|_| l2_support::DEFAULT_CRYPTOHFT_BASE_URL.to_owned());
    config.validate().map_err(backfill_support::source_error)?;
    Ok(config)
}

fn cancellation(context: &BackfillContext) -> ArchiveCancellation {
    let value = ArchiveCancellation::default();
    let cloned = value.clone();
    let shutdown = context.shutdown.clone();
    tokio::spawn(async move {
        shutdown.cancelled().await;
        cloned.cancel();
    });
    value
}

fn combined_checksum<'a>(
    manifests: impl Iterator<Item = &'a l2_support::HourlyArchiveManifest>,
) -> String {
    let mut hasher = Sha256::new();
    for manifest in manifests {
        hasher.update(manifest.sha256.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub(super) async fn persist(
    context: &BackfillContext,
    artifact_id: Uuid,
    target: &str,
    records: &[BinanceL2OneSecondFeature],
    checksum: &str,
    bytes: u64,
    metadata: serde_json::Value,
) -> Result<(), BackfillExecutionError> {
    let staging = format!("{target}_staging");
    let mut tx = context
        .pool
        .begin()
        .await
        .map_err(backfill_support::database_error)?;
    backfill_support::require_lease(&mut tx, context).await?;
    for chunk in records.chunks(BATCH_ROWS) {
        let mut query = QueryBuilder::<Postgres>::new(format!("INSERT INTO {staging} (symbol,second_start,source_event_timestamp,provider_received_at,available_at,source_update_id,feature_schema_version,quality_status,midpoint,microprice,spread_bps,bid_depth_5,ask_depth_5,imbalance_5,bid_depth_10,ask_depth_10,imbalance_10,bid_depth_20,ask_depth_20,imbalance_20,bid_depth_slope_20,ask_depth_slope_20,bid_depth_concentration_20,ask_depth_concentration_20,bid_quote_replenishment_1s,ask_quote_replenishment_1s,bid_quote_churn_1s,ask_quote_churn_1s,midpoint_change_bps_1s,spread_bps_delta_1s,depth_20_change_bps_1s,imbalance_20_delta_1s,midpoint_change_bps_5s,spread_bps_delta_5s,depth_20_change_bps_5s,imbalance_20_delta_5s,midpoint_change_bps_15s,spread_bps_delta_15s,depth_20_change_bps_15s,imbalance_20_delta_15s,midpoint_change_bps_30s,spread_bps_delta_30s,depth_20_change_bps_30s,imbalance_20_delta_30s,midpoint_change_bps_60s,spread_bps_delta_60s,depth_20_change_bps_60s,imbalance_20_delta_60s,artifact_id) "));
        query.push_values(chunk, |mut row, r| {
            row.push_bind(&r.symbol)
                .push_bind(r.second_start)
                .push_bind(r.source_event_timestamp)
                .push_bind(r.provider_received_at)
                .push_bind(r.available_at)
                .push_bind(r.source_update_id)
                .push_bind(&r.feature_schema_version)
                .push_bind(&r.quality_status)
                .push_bind(r.midpoint)
                .push_bind(r.microprice)
                .push_bind(r.spread_bps)
                .push_bind(r.bid_depth_5)
                .push_bind(r.ask_depth_5)
                .push_bind(r.imbalance_5)
                .push_bind(r.bid_depth_10)
                .push_bind(r.ask_depth_10)
                .push_bind(r.imbalance_10)
                .push_bind(r.bid_depth_20)
                .push_bind(r.ask_depth_20)
                .push_bind(r.imbalance_20)
                .push_bind(r.bid_depth_slope_20)
                .push_bind(r.ask_depth_slope_20)
                .push_bind(r.bid_depth_concentration_20)
                .push_bind(r.ask_depth_concentration_20)
                .push_bind(r.bid_quote_replenishment_1s)
                .push_bind(r.ask_quote_replenishment_1s)
                .push_bind(r.bid_quote_churn_1s)
                .push_bind(r.ask_quote_churn_1s)
                .push_bind(r.midpoint_change_bps_1s)
                .push_bind(r.spread_bps_delta_1s)
                .push_bind(r.depth_20_change_bps_1s)
                .push_bind(r.imbalance_20_delta_1s)
                .push_bind(r.midpoint_change_bps_5s)
                .push_bind(r.spread_bps_delta_5s)
                .push_bind(r.depth_20_change_bps_5s)
                .push_bind(r.imbalance_20_delta_5s)
                .push_bind(r.midpoint_change_bps_15s)
                .push_bind(r.spread_bps_delta_15s)
                .push_bind(r.depth_20_change_bps_15s)
                .push_bind(r.imbalance_20_delta_15s)
                .push_bind(r.midpoint_change_bps_30s)
                .push_bind(r.spread_bps_delta_30s)
                .push_bind(r.depth_20_change_bps_30s)
                .push_bind(r.imbalance_20_delta_30s)
                .push_bind(r.midpoint_change_bps_60s)
                .push_bind(r.spread_bps_delta_60s)
                .push_bind(r.depth_20_change_bps_60s)
                .push_bind(r.imbalance_20_delta_60s)
                .push_bind(artifact_id);
        });
        query.push(" ON CONFLICT (artifact_id,symbol,second_start) DO NOTHING");
        query
            .build()
            .execute(&mut *tx)
            .await
            .map_err(backfill_support::database_error)?;
    }
    let count: i64 = sqlx::query_scalar(&format!(
        "SELECT count(*)::bigint FROM {staging} WHERE artifact_id=$1"
    ))
    .bind(artifact_id)
    .fetch_one(&mut *tx)
    .await
    .map_err(backfill_support::database_error)?;
    if count
        != i64::try_from(records.len())
            .map_err(|_| backfill_support::integrity("record count overflow"))?
    {
        return Err(backfill_support::integrity(
            "staged Binance L2 record count mismatch",
        ));
    }
    let columns = "symbol,second_start,source_event_timestamp,provider_received_at,available_at,source_update_id,feature_schema_version,quality_status,midpoint,microprice,spread_bps,bid_depth_5,ask_depth_5,imbalance_5,bid_depth_10,ask_depth_10,imbalance_10,bid_depth_20,ask_depth_20,imbalance_20,bid_depth_slope_20,ask_depth_slope_20,bid_depth_concentration_20,ask_depth_concentration_20,bid_quote_replenishment_1s,ask_quote_replenishment_1s,bid_quote_churn_1s,ask_quote_churn_1s,midpoint_change_bps_1s,spread_bps_delta_1s,depth_20_change_bps_1s,imbalance_20_delta_1s,midpoint_change_bps_5s,spread_bps_delta_5s,depth_20_change_bps_5s,imbalance_20_delta_5s,midpoint_change_bps_15s,spread_bps_delta_15s,depth_20_change_bps_15s,imbalance_20_delta_15s,midpoint_change_bps_30s,spread_bps_delta_30s,depth_20_change_bps_30s,imbalance_20_delta_30s,midpoint_change_bps_60s,spread_bps_delta_60s,depth_20_change_bps_60s,imbalance_20_delta_60s,artifact_id";
    sqlx::query(&format!("INSERT INTO {target} ({columns}) SELECT {columns} FROM {staging} WHERE artifact_id=$1 ORDER BY symbol,second_start ON CONFLICT (symbol,second_start) DO NOTHING")).bind(artifact_id).execute(&mut *tx).await.map_err(backfill_support::database_error)?;
    backfill_support::complete_artifact(
        &mut tx,
        context,
        ArtifactCompletion {
            artifact_id,
            checksum,
            byte_size: bytes,
            record_count: count,
            minimum: records.first().map(|r| r.source_event_timestamp),
            maximum: records.last().map(|r| r.source_event_timestamp),
            metadata,
        },
    )
    .await?;
    sqlx::query(&format!("DELETE FROM {staging} WHERE artifact_id=$1"))
        .bind(artifact_id)
        .execute(&mut *tx)
        .await
        .map_err(backfill_support::database_error)?;
    tx.commit()
        .await
        .map_err(backfill_support::database_error)?;
    Ok(())
}
