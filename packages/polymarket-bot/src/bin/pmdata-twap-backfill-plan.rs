use std::env;

use anyhow::{bail, Context, Result};
use chrono::{Duration as ChronoDuration, NaiveDate, Utc};
use polymarket_bot::{
    config::AppConfig,
    ingestion::{
        job::{BackfillRequest, IngesterKey, BACKFILL_REQUEST_VERSION},
        repository::IngestionRepository,
    },
};
use serde_json::json;
use sqlx::postgres::PgPoolOptions;
use tracing::info;
use tracing_subscriber::EnvFilter;

const DEFAULT_START: &str = "2026-08-01";
const MATERIALIZATION_CONTRACT: &str = "pmdata-chainlink-btcusd-twap-v1";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .init();
    let start = configured_date("POLYMARKET_PMDATA_TWAP_START_DATE", DEFAULT_START)?;
    let default_end = Utc::now().date_naive().to_string();
    let end = configured_date("POLYMARKET_PMDATA_TWAP_END_DATE", &default_end)?;
    if end <= start {
        bail!("PMData TWAP end date must be later than start date");
    }

    let app = AppConfig::from_env()?;
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(&app.postgres.database_url())
        .await
        .context("failed to connect PMData TWAP planner to PostgreSQL")?;
    let repository = IngestionRepository::from_pool(pool);
    let mut date = start;
    let mut enqueued = 0u64;
    while date < end {
        for ingester in [
            IngesterKey::PmdataChainlinkBtcusdTwap30s,
            IngesterKey::PmdataChainlinkBtcusdTwap60s,
        ] {
            let request = daily_request(date, ingester)?;
            let job = repository.enqueue(&request).await.with_context(|| {
                format!(
                    "failed to enqueue PMData {} shard for {date}",
                    ingester.as_str()
                )
            })?;
            info!(
                job_id = %job.job_id,
                shard_date = %date,
                ingester = %ingester,
                status = ?job.status,
                "PMData TWAP shard ready"
            );
            enqueued = enqueued.saturating_add(1);
        }
        date += ChronoDuration::days(1);
    }
    info!(%start, %end, enqueued, "PMData TWAP historical plan ready");
    Ok(())
}

fn daily_request(
    date: NaiveDate,
    ingester: IngesterKey,
) -> Result<polymarket_bot::ingestion::job::ValidatedBackfillRequest> {
    if !matches!(
        ingester,
        IngesterKey::PmdataChainlinkBtcusdTwap30s | IngesterKey::PmdataChainlinkBtcusdTwap60s
    ) {
        bail!("PMData planner received a non-PMData ingester");
    }
    let next = date + ChronoDuration::days(1);
    BackfillRequest {
        ingester,
        request_version: BACKFILL_REQUEST_VERSION,
        range_start: date.and_hms_opt(0, 0, 0).unwrap().and_utc(),
        range_end: next.and_hms_opt(0, 0, 0).unwrap().and_utc(),
        parameters: json!({}),
        idempotency_key: format!("{}:{date}:{MATERIALIZATION_CONTRACT}", ingester.as_str()),
    }
    .validate()
    .with_context(|| format!("invalid PMData TWAP shard request for {date}"))
}

fn configured_date(key: &str, default: &str) -> Result<NaiveDate> {
    let value = env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default.to_string());
    NaiveDate::parse_from_str(&value, "%Y-%m-%d")
        .with_context(|| format!("{key} must be YYYY-MM-DD"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daily_requests_are_one_day_and_stable() {
        let date = NaiveDate::from_ymd_opt(2026, 8, 1).unwrap();
        let request = daily_request(date, IngesterKey::PmdataChainlinkBtcusdTwap60s).unwrap();
        assert_eq!(request.expected_work_units, 1);
        assert_eq!(
            request.range_end - request.range_start,
            ChronoDuration::days(1)
        );
        assert!(request.idempotency_key.contains("2026-08-01"));
    }
}
