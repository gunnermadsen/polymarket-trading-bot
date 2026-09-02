use std::{path::PathBuf, sync::Arc, time::Duration};

use chrono::{Datelike, TimeZone, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    domain::{
        BackfillContext, BackfillExecutionError, BackfillOutcome, BackfillRequest, BackfillShard,
        StrategyCapability, StrategyDescriptor, ValidatedBackfillRequest,
    },
    strategies::{backfill_support as ledger, binance::archive_support::ArchiveCancellation},
};

use super::{
    archive_client::{KrakenArchiveClient, DEFAULT_FUTURES_BASE_URL},
    lake::KrakenDataLake,
    spot_support::{
        fetch_and_publish_day, KrakenSpotTradeConfig, KRAKEN_SPOT_PROVIDER,
        KRAKEN_SPOT_SCHEMA_VERSION,
    },
    types::{KrakenBackfillJob, KrakenDataset},
};

pub const CONTRACT_VERSION: i32 = 1;
pub const REQUEST_VERSION: i32 = 1;
pub const MAX_SHARDS: usize = 3_650;
pub const DEFAULT_SYMBOL: &str = "PF_XBTUSD";
pub const DEFAULT_INTERVAL_SECONDS: i32 = 900;
const FUTURES_PROVIDER: &str = "kraken_futures";

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct KrakenParameters {
    pub symbol: String,
    pub interval_seconds: i32,
}

impl Default for KrakenParameters {
    fn default() -> Self {
        Self {
            symbol: DEFAULT_SYMBOL.to_owned(),
            interval_seconds: DEFAULT_INTERVAL_SECONDS,
        }
    }
}

pub fn descriptor(
    key: &'static str,
    name: &'static str,
    description: &'static str,
) -> Result<StrategyDescriptor, BackfillExecutionError> {
    let value = StrategyDescriptor {
        strategy_key: Arc::from(key),
        name: Arc::from(name),
        description: Arc::from(description),
        capabilities: vec![StrategyCapability::Backfill],
        strategy_contract_version: CONTRACT_VERSION,
        request_schema_version: Some(REQUEST_VERSION),
        shardable: true,
        maximum_shards: MAX_SHARDS,
    };
    value.validate()?;
    Ok(value)
}

pub fn validate_request(
    descriptor: &StrategyDescriptor,
    request: &BackfillRequest,
) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
    if request.strategy_key != descriptor.strategy_key.as_ref() {
        return Err(BackfillExecutionError::invalid(
            "strategy_key_mismatch",
            "request strategy key does not match the Kraken backfill strategy",
        ));
    }
    if request.range.end <= request.range.start || request.range.end > Utc::now() {
        return Err(BackfillExecutionError::invalid(
            "range_invalid",
            "range must be increasing and may not end in the future",
        ));
    }
    let parameters: KrakenParameters =
        serde_json::from_value(request.parameters.clone()).map_err(|error| {
            BackfillExecutionError::invalid(
                "parameters_invalid",
                format!("invalid Kraken parameters: {error}"),
            )
        })?;
    if parameters.symbol.is_empty()
        || parameters.symbol.len() > 32
        || !parameters
            .symbol
            .bytes()
            .all(|value| value.is_ascii_uppercase() || value.is_ascii_digit() || value == b'_')
    {
        return Err(BackfillExecutionError::invalid(
            "symbol_invalid",
            "Kraken symbol must contain only uppercase ASCII letters, digits, or underscores",
        ));
    }
    if parameters.interval_seconds <= 0 || 86_400 % parameters.interval_seconds != 0 {
        return Err(BackfillExecutionError::invalid(
            "interval_invalid",
            "interval_seconds must be positive and divide one UTC day evenly",
        ));
    }
    let interval = i64::from(parameters.interval_seconds);
    if request.range.start.timestamp().rem_euclid(interval) != 0
        || request.range.end.timestamp().rem_euclid(interval) != 0
    {
        return Err(BackfillExecutionError::invalid(
            "range_alignment_invalid",
            "Kraken range boundaries must align to interval_seconds",
        ));
    }
    request.execution.validate()?;
    Ok(ValidatedBackfillRequest {
        strategy_key: descriptor.strategy_key.clone(),
        strategy_contract_version: CONTRACT_VERSION,
        request_schema_version: REQUEST_VERSION,
        range_start: request.range.start,
        range_end: request.range.end,
        parameters: serde_json::to_value(parameters).map_err(|error| {
            BackfillExecutionError::invalid("parameters_invalid", error.to_string())
        })?,
        execution: request.execution.clone(),
    })
}

pub fn plan_time_shards(
    request: &ValidatedBackfillRequest,
    maximum: usize,
) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
    let mut shards = Vec::new();
    let mut cursor = request.range_start;
    while cursor < request.range_end {
        let next_day = Utc
            .with_ymd_and_hms(cursor.year(), cursor.month(), cursor.day(), 0, 0, 0)
            .single()
            .and_then(|midnight| midnight.checked_add_signed(chrono::Duration::days(1)))
            .ok_or_else(|| {
                BackfillExecutionError::invalid("range_overflow", "Kraken range overflowed")
            })?;
        let end = next_day.min(request.range_end);
        shards.push(BackfillShard {
            shard_key: format!(
                "{}-{}",
                cursor.format("%Y%m%dT%H%M%SZ"),
                end.format("%Y%m%dT%H%M%SZ")
            ),
            range_start: cursor,
            range_end: end,
            parameters: request.parameters.clone(),
        });
        if shards.len() > maximum {
            return Err(BackfillExecutionError::invalid(
                "too_many_shards",
                format!("request exceeds {maximum} Kraken shards"),
            ));
        }
        cursor = end;
    }
    Ok(shards)
}

pub fn plan_snapshot_shard(
    request: &ValidatedBackfillRequest,
    _maximum: usize,
) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
    Ok(vec![BackfillShard {
        shard_key: format!(
            "snapshot-{}-{}",
            request.range_start.format("%Y%m%dT%H%M%SZ"),
            request.range_end.format("%Y%m%dT%H%M%SZ")
        ),
        range_start: request.range_start,
        range_end: request.range_end,
        parameters: request.parameters.clone(),
    }])
}

fn parameters(shard: &BackfillShard) -> Result<KrakenParameters, BackfillExecutionError> {
    serde_json::from_value(shard.parameters.clone())
        .map_err(|error| BackfillExecutionError::invalid("parameters_invalid", error.to_string()))
}

fn cancellation(context: &BackfillContext) -> ArchiveCancellation {
    let cancellation = ArchiveCancellation::default();
    let watcher = cancellation.clone();
    let shutdown = context.shutdown.clone();
    tokio::spawn(async move {
        shutdown.cancelled().await;
        watcher.cancel();
    });
    cancellation
}

pub async fn execute_futures(
    context: BackfillContext,
    shard: BackfillShard,
    strategy_key: &str,
    dataset: KrakenDataset,
) -> Result<BackfillOutcome, BackfillExecutionError> {
    if context.shutdown.is_cancelled() {
        return Err(BackfillExecutionError::new(
            crate::domain::BackfillFailureKind::Cancelled,
            "backfill_cancelled",
            "Kraken backfill was cancelled",
        ));
    }
    let parameters = parameters(&shard)?;
    let logical_key = format!(
        "{}:{}:{}:{}:{}",
        dataset, parameters.symbol, parameters.interval_seconds, shard.range_start, shard.range_end
    );
    if let Some(outcome) = ledger::completed_outcome(&context, strategy_key, &logical_key).await? {
        return Ok(outcome);
    }
    let base_url = std::env::var("KRAKEN_FUTURES_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_FUTURES_BASE_URL.to_owned());
    let root = PathBuf::from(
        std::env::var("INGESTER_KRAKEN_DATA_ROOT")
            .unwrap_or_else(|_| "/var/lib/kraken-data".to_owned()),
    );
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60))
        .user_agent("capitonic-ingester-kraken/1.0")
        .build()
        .map_err(ledger::source_error)?;
    let archive = KrakenArchiveClient::new(client, base_url).map_err(ledger::source_error)?;
    let lake = KrakenDataLake::new(root.clone()).map_err(ledger::source_error)?;
    let local_job = KrakenBackfillJob {
        job_id: context.job_id,
        dataset: dataset.as_str().to_owned(),
        symbol: parameters.symbol.clone(),
        interval_seconds: parameters.interval_seconds,
        range_start: shard.range_start,
        range_end: shard.range_end,
    };
    let fetched = archive
        .fetch(&local_job)
        .await
        .map_err(ledger::source_error)?;
    if context.shutdown.is_cancelled() {
        return Err(BackfillExecutionError::new(
            crate::domain::BackfillFailureKind::Cancelled,
            "backfill_cancelled",
            "Kraken backfill was cancelled",
        ));
    }
    let object = lake
        .publish(&local_job, &fetched.rows)
        .await
        .map_err(ledger::source_error)?;
    let durable_target = root
        .join(&object.relative_path)
        .to_string_lossy()
        .to_string();
    let artifact_id = ledger::create_artifact(
        &context,
        strategy_key,
        strategy_key,
        &logical_key,
        FUTURES_PROVIDER,
        &fetched.source_url,
        &durable_target,
    )
    .await?;
    let bounds = fetched.rows.time_bounds();
    let (minimum, maximum) = bounds.unzip();
    let mut tx = context.pool.begin().await.map_err(ledger::database_error)?;
    ledger::complete_artifact(
        &mut tx,
        &context,
        ledger::ArtifactCompletion {
            artifact_id,
            checksum: &object.sha256,
            byte_size: u64::try_from(object.byte_size)
                .map_err(|_| ledger::integrity("Kraken artifact byte size was negative"))?,
            record_count: object.row_count,
            minimum,
            maximum,
            metadata: json!({
                "dataset": dataset.as_str(),
                "symbol": parameters.symbol,
                "interval_seconds": parameters.interval_seconds,
                "lake_relative_path": object.relative_path,
                "source_url": fetched.source_url,
            }),
        },
    )
    .await?;
    tx.commit().await.map_err(ledger::database_error)?;
    Ok(ledger::outcome(
        object.row_count,
        &shard,
        json!({
            "provider": FUTURES_PROVIDER,
            "dataset": dataset.as_str(),
            "records_verified": object.row_count,
            "durable_target": durable_target,
        }),
    ))
}

pub async fn execute_spot(
    context: BackfillContext,
    shard: BackfillShard,
    strategy_key: &str,
) -> Result<BackfillOutcome, BackfillExecutionError> {
    let logical_key = format!("XBTUSD:{}:{}", shard.range_start, shard.range_end);
    if let Some(outcome) = ledger::completed_outcome(&context, strategy_key, &logical_key).await? {
        return Ok(outcome);
    }
    let root = PathBuf::from(
        std::env::var("INGESTER_KRAKEN_DATA_ROOT")
            .unwrap_or_else(|_| "/var/lib/kraken-data".to_owned()),
    );
    let config = KrakenSpotTradeConfig {
        base_url: std::env::var("KRAKEN_SPOT_BASE_URL")
            .unwrap_or_else(|_| "https://api.kraken.com".to_owned()),
        lake_root: root.clone(),
        request_delay: Duration::from_millis(350),
    };
    let published = fetch_and_publish_day(
        &Client::new(),
        &config,
        shard.range_start,
        shard.range_end,
        &cancellation(&context),
    )
    .await
    .map_err(ledger::source_error)?;
    let durable_target = root.to_string_lossy().to_string();
    let source_uri = format!(
        "{}/0/public/Trades?pair=XBTUSD&since={}",
        config.base_url.trim_end_matches('/'),
        shard.range_start.timestamp_nanos_opt().unwrap_or_default()
    );
    let artifact_id = ledger::create_artifact(
        &context,
        strategy_key,
        strategy_key,
        &logical_key,
        KRAKEN_SPOT_PROVIDER,
        &source_uri,
        &durable_target,
    )
    .await?;
    let record_count = i64::try_from(published.trade_count)
        .map_err(|_| ledger::integrity("Kraken Spot trade count overflowed"))?;
    let byte_size = published.trade_bytes.saturating_add(published.candle_bytes);
    let mut tx = context.pool.begin().await.map_err(ledger::database_error)?;
    ledger::complete_artifact(
        &mut tx,
        &context,
        ledger::ArtifactCompletion {
            artifact_id,
            checksum: &published.combined_sha256,
            byte_size,
            record_count,
            minimum: Some(published.minimum_timestamp),
            maximum: Some(published.maximum_timestamp),
            metadata: json!({
                "schema_version": KRAKEN_SPOT_SCHEMA_VERSION,
                "trade_path": published.trade_path,
                "candle_path": published.candle_path,
                "trade_sha256": published.trade_sha256,
                "candle_sha256": published.candle_sha256,
                "trade_count": published.trade_count,
                "candle_count": published.candle_count,
                "pages": published.pages,
            }),
        },
    )
    .await?;
    tx.commit().await.map_err(ledger::database_error)?;
    Ok(ledger::outcome(
        record_count,
        &shard,
        json!({
            "provider": KRAKEN_SPOT_PROVIDER,
            "records_verified": record_count,
            "candles_verified": published.candle_count,
            "durable_target": durable_target,
        }),
    ))
}

#[cfg(test)]
pub fn test_request(
    key: &str,
    start: chrono::DateTime<Utc>,
    end: chrono::DateTime<Utc>,
) -> BackfillRequest {
    BackfillRequest {
        strategy_key: key.to_owned(),
        range: crate::domain::BackfillRange { start, end },
        parameters: serde_json::to_value(KrakenParameters::default())
            .unwrap_or(serde_json::Value::Null),
        execution: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use super::*;

    fn test_descriptor() -> StrategyDescriptor {
        descriptor(
            "kraken_test_backfill",
            "Kraken test",
            "Kraken test strategy",
        )
        .unwrap()
    }

    #[test]
    fn parameters_reject_unknown_fields_and_invalid_symbols() {
        let start = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
        let mut unknown = test_request(
            "kraken_test_backfill",
            start,
            start + chrono::Duration::hours(1),
        );
        unknown.parameters = json!({"symbol":"PF_XBTUSD","interval_seconds":900,"extra":true});
        assert!(validate_request(&test_descriptor(), &unknown).is_err());

        let mut lower = unknown;
        lower.parameters = json!({"symbol":"pf_xbtusd","interval_seconds":900});
        assert!(validate_request(&test_descriptor(), &lower).is_err());
    }

    #[test]
    fn validation_rejects_unaligned_ranges_and_intervals() {
        let start = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 1).unwrap();
        let request = test_request(
            "kraken_test_backfill",
            start,
            start + chrono::Duration::hours(1),
        );
        assert!(validate_request(&test_descriptor(), &request).is_err());

        let start = Utc.with_ymd_and_hms(2026, 8, 1, 0, 0, 0).unwrap();
        let mut invalid_interval = test_request(
            "kraken_test_backfill",
            start,
            start + chrono::Duration::hours(1),
        );
        invalid_interval.parameters = json!({"symbol":"PF_XBTUSD","interval_seconds":1000});
        assert!(validate_request(&test_descriptor(), &invalid_interval).is_err());
    }

    #[test]
    fn daily_shards_are_deterministic_and_bound_midnight() {
        let start = Utc.with_ymd_and_hms(2026, 8, 1, 12, 0, 0).unwrap();
        let request = test_request(
            "kraken_test_backfill",
            start,
            start + chrono::Duration::hours(37),
        );
        let validated = validate_request(&test_descriptor(), &request).unwrap();
        let first = plan_time_shards(&validated, MAX_SHARDS).unwrap();
        let second = plan_time_shards(&validated, MAX_SHARDS).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        assert_eq!(
            first[0].range_end,
            Utc.with_ymd_and_hms(2026, 8, 2, 0, 0, 0).unwrap()
        );
        assert_eq!(
            first[1].range_end,
            Utc.with_ymd_and_hms(2026, 8, 3, 0, 0, 0).unwrap()
        );
    }
}
