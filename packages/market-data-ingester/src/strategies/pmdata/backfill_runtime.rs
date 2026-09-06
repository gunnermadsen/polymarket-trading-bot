use super::{
    support::{
        self, PmdataTwapConfig, PmdataTwapWindow, DEFAULT_PMDATA_BASE_URL,
        PMDATA_REFPRICE_PROVIDER, PMDATA_TWAP_PROVIDER,
    },
    types::{PmdataChainlinkBtcusdRefpriceRecord, PmdataChainlinkBtcusdTwapRecord},
};
use crate::{
    domain::{BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillShard},
    persistence::{
        insert_pmdata_chainlink_reference_prices, ChainlinkReferencePriceWrite,
        ReferencePriceArtifact,
    },
    strategies::{backfill_support, binance::archive_support::ArchiveCancellation},
};
use serde_json::json;
use sqlx::{Postgres, QueryBuilder};
use std::path::PathBuf;

pub fn config() -> Result<PmdataTwapConfig, BackfillExecutionError> {
    let value = PmdataTwapConfig {
        base_url: std::env::var("PMDATA_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_PMDATA_BASE_URL.into()),
        api_key: std::env::var("PMDATA_API_KEY")
            .or_else(|_| std::env::var("POLYMARKET_PMDATA_API_KEY"))
            .ok()
            .filter(|v| !v.trim().is_empty()),
        archive_root: PathBuf::from(
            std::env::var("INGESTER_PMDATA_ARCHIVE_ROOT").unwrap_or_else(|_| "/tmp/pmdata".into()),
        ),
    };
    value.validate().map_err(backfill_support::source_error)?;
    Ok(value)
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

pub async fn execute_twap(
    context: BackfillContext,
    shard: BackfillShard,
    strategy_key: &str,
    window: PmdataTwapWindow,
) -> Result<BackfillOutcome, BackfillExecutionError> {
    let config = config()?;
    let date = shard.range_start.date_naive();
    let logical_key = format!(
        "{}:{}:{}",
        config.logical_key(date, window),
        shard.range_start,
        shard.range_end
    );
    if let Some(outcome) =
        backfill_support::completed_outcome(&context, strategy_key, &logical_key).await?
    {
        return Ok(outcome);
    }
    let source_uri = config.source_uri(date, window);
    let artifact_id = backfill_support::create_artifact(
        &context,
        strategy_key,
        strategy_key,
        &logical_key,
        PMDATA_TWAP_PROVIDER,
        &source_uri,
        "market_data.pmdata_chainlink_btcusd_twap",
    )
    .await?;
    let archive = config
        .ensure_archive(
            &reqwest::Client::new(),
            date,
            window,
            None,
            &cancellation(&context),
        )
        .await
        .map_err(backfill_support::source_error)?;
    let parsed = support::parse_archive(archive.path, date, window, 1_000)
        .await
        .map_err(backfill_support::source_error)?;
    let source_coverage = (parsed.minimum_timestamp, parsed.maximum_timestamp);
    let records = parsed
        .records
        .into_iter()
        .filter(|r| r.source_timestamp >= shard.range_start && r.source_timestamp < shard.range_end)
        .collect::<Vec<_>>();
    persist_twap(
        &context,
        artifact_id,
        &records,
        &archive.sha256,
        archive.bytes,
    )
    .await?;
    Ok(backfill_support::outcome(
        records.len() as i64,
        &shard,
        json!({"provider":PMDATA_TWAP_PROVIDER,"window_seconds":window.seconds(),"records_verified":records.len(),"source_day_minimum":source_coverage.0,"source_day_maximum":source_coverage.1}),
    ))
}

pub async fn execute_refprice(
    context: BackfillContext,
    shard: BackfillShard,
    strategy_key: &str,
) -> Result<BackfillOutcome, BackfillExecutionError> {
    let config = config()?;
    let date = shard.range_start.date_naive();
    let logical_key = format!(
        "{}:{}:{}",
        config.refprice_logical_key(date),
        shard.range_start,
        shard.range_end
    );
    if let Some(outcome) =
        backfill_support::completed_outcome(&context, strategy_key, &logical_key).await?
    {
        return Ok(outcome);
    }
    let source_uri = config.refprice_source_uri(date);
    let artifact_id = backfill_support::create_artifact(
        &context,
        strategy_key,
        strategy_key,
        &logical_key,
        PMDATA_REFPRICE_PROVIDER,
        &source_uri,
        "market_data.pmdata_chainlink_btcusd_reference_prices",
    )
    .await?;
    let archive = config
        .ensure_refprice_archive(&reqwest::Client::new(), date, None, &cancellation(&context))
        .await
        .map_err(backfill_support::source_error)?;
    let parsed = support::parse_refprice_archive(archive.path, date, 1_000)
        .await
        .map_err(backfill_support::source_error)?;
    let source_coverage = (parsed.minimum_timestamp, parsed.maximum_timestamp);
    let records = parsed
        .records
        .into_iter()
        .filter(|r| r.source_timestamp >= shard.range_start && r.source_timestamp < shard.range_end)
        .collect::<Vec<_>>();
    persist_refprice(
        &context,
        artifact_id,
        &records,
        &archive.sha256,
        archive.bytes,
        strategy_key,
    )
    .await?;
    Ok(backfill_support::outcome(
        records.len() as i64,
        &shard,
        json!({"provider":PMDATA_REFPRICE_PROVIDER,"records_verified":records.len(),"source_day_minimum":source_coverage.0,"source_day_maximum":source_coverage.1}),
    ))
}

async fn persist_twap(
    context: &BackfillContext,
    artifact_id: uuid::Uuid,
    records: &[PmdataChainlinkBtcusdTwapRecord],
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
        let mut query=QueryBuilder::<Postgres>::new("INSERT INTO market_data.pmdata_chainlink_btcusd_twap (source_timestamp,provider_received_at,valid_from_timestamp,expires_at,window_seconds,twap_price,full_accuracy_value,report_version,source_date,archive_row_number,artifact_id) ");
        query.push_values(chunk, |mut row, r| {
            row.push_bind(r.source_timestamp)
                .push_bind(r.provider_received_at)
                .push_bind(r.valid_from_timestamp)
                .push_bind(r.expires_at)
                .push_bind(r.window_seconds)
                .push_bind(r.twap_price)
                .push_bind(&r.full_accuracy_value)
                .push_bind(&r.report_version)
                .push_bind(r.source_date)
                .push_bind(r.archive_row_number)
                .push_bind(artifact_id);
        });
        query.push(" ON CONFLICT (source_timestamp,window_seconds) DO NOTHING");
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
            metadata: json!({"provider":PMDATA_TWAP_PROVIDER}),
        },
    )
    .await?;
    tx.commit().await.map_err(backfill_support::database_error)
}

async fn persist_refprice(
    context: &BackfillContext,
    artifact_id: uuid::Uuid,
    records: &[PmdataChainlinkBtcusdRefpriceRecord],
    checksum: &str,
    bytes: u64,
    strategy_key: &str,
) -> Result<(), BackfillExecutionError> {
    let mut tx = context
        .pool
        .begin()
        .await
        .map_err(backfill_support::database_error)?;
    backfill_support::require_lease(&mut tx, context).await?;
    for chunk in records.chunks(1_000) {
        let writes = chunk
            .iter()
            .map(|r| ChainlinkReferencePriceWrite {
                feed_id: "0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8",
                source_timestamp: r.source_timestamp,
                valid_from_timestamp: r.valid_from_timestamp,
                provider_available_at: Some(r.provider_received_at),
                received_at: r.provider_received_at,
                price: r.price,
                bid: r.bid,
                ask: r.ask,
                report_sha256: &r.canonical_row_sha256,
                payload_sha256: &r.canonical_row_sha256,
                artifact: ReferencePriceArtifact::Backfill(artifact_id),
                expires_at: r.expires_at,
                report_version: r.report_version.as_deref(),
                source_date: Some(r.source_date),
                archive_row_number: Some(r.archive_row_number),
                report_hash_kind: "canonical_archive_row",
            })
            .collect::<Vec<_>>();
        insert_pmdata_chainlink_reference_prices(&mut tx, strategy_key, &writes)
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
            metadata: json!({"provider":PMDATA_REFPRICE_PROVIDER}),
        },
    )
    .await?;
    tx.commit().await.map_err(backfill_support::database_error)
}
