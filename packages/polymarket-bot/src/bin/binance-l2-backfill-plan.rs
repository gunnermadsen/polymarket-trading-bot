use std::env::{self, VarError};

use anyhow::{bail, Context, Result};
use chrono::{Duration as ChronoDuration, NaiveDate};
use polymarket_bot::{
    config::AppConfig,
    ingestion::{
        cryptohft_binance_l2::{
            day_artifact_logical_key, validate_representative_day_quality,
            BINANCE_L2_MATERIALIZATION_CONTRACT,
        },
        job::{BackfillJobStatus, BackfillRequest, IngesterKey, BACKFILL_REQUEST_VERSION},
        repository::IngestionRepository,
    },
};
use serde_json::json;
use sqlx::postgres::PgPoolOptions;
use tracing::info;
use tracing_subscriber::EnvFilter;

const RANGE_START: &str = "2026-04-14";
const RANGE_END: &str = "2026-08-02";
const EXPECTED_DAILY_SHARDS: usize = 110;
const RETRY_GENERATION_ENV: &str = "POLYMARKET_BINANCE_L2_RETRY_GENERATION";
const DEFAULT_RETRY_GENERATION: u32 = 1;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .init();

    let retry_generation = configured_retry_generation()?;
    let app = AppConfig::from_env()?;
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(&app.postgres.database_url())
        .await
        .context("failed to connect Binance L2 planner to PostgreSQL")?;
    let repository = IngestionRepository::from_pool(pool.clone());
    let (start, end, shard_dates) = approved_shard_dates()?;

    let representative_key = day_artifact_logical_key(start);
    let representative = sqlx::query_as::<_, (String, serde_json::Value)>(
        r#"
        SELECT status, metadata
        FROM polymarket.backfill_artifacts
        WHERE provider = 'cryptohftdata'
          AND logical_key = $1
        "#,
    )
    .bind(&representative_key)
    .fetch_optional(&pool)
    .await
    .context("failed to inspect the representative Binance L2 source audit")?;
    match representative {
        Some((status, metadata)) if status == "completed" => {
            validate_representative_day_quality(&metadata).context(
                "representative CryptoHFT source audit is not full-day qualified; refusing to enqueue the full Binance L2 backfill",
            )?;
        }
        _ => {
            let request = daily_request(start, retry_generation)?;
            let job = repository.enqueue(&request).await.context(
                "failed to enqueue or recover the representative Binance L2 source audit",
            )?;
            match job.status {
                BackfillJobStatus::Queued
                | BackfillJobStatus::Running
                | BackfillJobStatus::CancelRequested => {
                    info!(
                        job_id = %job.job_id,
                        shard_date = %start,
                        retry_generation,
                        status = ?job.status,
                        "representative Binance L2 source audit active; full plan remains gated"
                    );
                    return Ok(());
                }
                BackfillJobStatus::Completed
                | BackfillJobStatus::Failed
                | BackfillJobStatus::Cancelled => {
                    bail!(
                        "retry generation {retry_generation} resolves to terminal representative job {} ({:?}); increment {RETRY_GENERATION_ENV}",
                        job.job_id,
                        job.status
                    );
                }
            }
        }
    }

    let mut enqueued = 0u64;
    for date in shard_dates {
        let request = daily_request(date, retry_generation)?;
        let job = repository
            .enqueue(&request)
            .await
            .with_context(|| format!("failed to enqueue Binance L2 shard for {date}"))?;
        info!(job_id = %job.job_id, shard_date = %date, status = ?job.status, "Binance L2 shard ready");
        enqueued = enqueued.saturating_add(1);
    }

    info!(%start, %end, enqueued, retry_generation, "Binance L2 historical plan ready");
    Ok(())
}

fn daily_request(
    date: NaiveDate,
    retry_generation: u32,
) -> Result<polymarket_bot::ingestion::job::ValidatedBackfillRequest> {
    if retry_generation == 0 {
        bail!("Binance L2 retry generation must be positive");
    }
    let next = date + ChronoDuration::days(1);
    BackfillRequest {
        ingester: IngesterKey::BinanceBtcusdtL2OneSecondFeatures,
        request_version: BACKFILL_REQUEST_VERSION,
        range_start: date.and_hms_opt(0, 0, 0).unwrap().and_utc(),
        range_end: next.and_hms_opt(0, 0, 0).unwrap().and_utc(),
        parameters: json!({}),
        idempotency_key: format!(
            "binance-btcusdt-l2:{date}:{BINANCE_L2_MATERIALIZATION_CONTRACT}:retry-{retry_generation}"
        ),
    }
    .validate()
    .with_context(|| format!("invalid Binance L2 shard request for {date}"))
}

fn approved_shard_dates() -> Result<(NaiveDate, NaiveDate, Vec<NaiveDate>)> {
    let start = NaiveDate::parse_from_str(RANGE_START, "%Y-%m-%d")?;
    let end = NaiveDate::parse_from_str(RANGE_END, "%Y-%m-%d")?;
    if end <= start {
        bail!("Binance L2 backfill end must be later than start");
    }

    let mut dates = Vec::with_capacity(EXPECTED_DAILY_SHARDS);
    let mut date = start;
    while date < end {
        dates.push(date);
        date += ChronoDuration::days(1);
    }
    if dates.len() != EXPECTED_DAILY_SHARDS {
        bail!(
            "Binance L2 approved range must contain exactly {EXPECTED_DAILY_SHARDS} UTC-day shards; found {}",
            dates.len()
        );
    }

    Ok((start, end, dates))
}

fn configured_retry_generation() -> Result<u32> {
    match env::var(RETRY_GENERATION_ENV) {
        Ok(raw) => parse_retry_generation(Some(&raw)),
        Err(VarError::NotPresent) => parse_retry_generation(None),
        Err(VarError::NotUnicode(_)) => {
            bail!("{RETRY_GENERATION_ENV} must contain valid Unicode")
        }
    }
}

fn parse_retry_generation(raw: Option<&str>) -> Result<u32> {
    let Some(raw) = raw else {
        return Ok(DEFAULT_RETRY_GENERATION);
    };
    let generation = raw
        .trim()
        .parse::<u32>()
        .with_context(|| format!("{RETRY_GENERATION_ENV} must be a positive integer"))?;
    if generation == 0 {
        bail!("{RETRY_GENERATION_ENV} must be a positive integer");
    }
    Ok(generation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approved_range_contains_exactly_110_daily_shards() {
        let (start, end, dates) = approved_shard_dates().unwrap();

        assert_eq!(start, NaiveDate::from_ymd_opt(2026, 4, 14).unwrap());
        assert_eq!(end, NaiveDate::from_ymd_opt(2026, 8, 2).unwrap());
        assert_eq!(dates.len(), EXPECTED_DAILY_SHARDS);
        assert_eq!(dates.first().copied(), Some(start));
        assert_eq!(
            dates.last().copied(),
            Some(NaiveDate::from_ymd_opt(2026, 8, 1).unwrap())
        );
    }

    #[test]
    fn daily_request_identity_includes_contract_and_retry_generation() {
        let date = NaiveDate::from_ymd_opt(2026, 4, 14).unwrap();
        let first = daily_request(date, 1).unwrap();
        let retry = daily_request(date, 2).unwrap();

        assert_eq!(
            first.idempotency_key,
            format!("binance-btcusdt-l2:{date}:{BINANCE_L2_MATERIALIZATION_CONTRACT}:retry-1")
        );
        assert_ne!(first.idempotency_key, retry.idempotency_key);
        assert_eq!(
            day_artifact_logical_key(date),
            format!(
                "cryptohftdata:binance-futures:BTCUSDT:l2-day:{BINANCE_L2_MATERIALIZATION_CONTRACT}:{date}"
            )
        );
    }

    #[test]
    fn retry_generation_defaults_and_requires_a_positive_integer() {
        assert_eq!(parse_retry_generation(None).unwrap(), 1);
        assert_eq!(parse_retry_generation(Some(" 7 ")).unwrap(), 7);
        assert!(parse_retry_generation(Some("0")).is_err());
        assert!(parse_retry_generation(Some("not-a-number")).is_err());
    }
}
