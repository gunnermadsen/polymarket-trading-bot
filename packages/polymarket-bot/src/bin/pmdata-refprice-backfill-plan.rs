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

const DEFAULT_START: &str = "2026-06-07";
const MATERIALIZATION_CONTRACT: &str = "pmdata-chainlink-btcusd-refprice-v5";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .init();
    let start = configured_date("POLYMARKET_PMDATA_REFPRICE_START_DATE", DEFAULT_START)?;
    let default_end = Utc::now().date_naive().to_string();
    let end = configured_date("POLYMARKET_PMDATA_REFPRICE_END_DATE", &default_end)?;
    if end <= start {
        bail!("PMData RefPrice end date must be later than start date");
    }
    let app = AppConfig::from_env()?;
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(&app.postgres.database_url())
        .await
        .context("failed to connect PMData RefPrice planner to PostgreSQL")?;
    let repository = IngestionRepository::from_pool(pool);
    let mut date = start;
    let mut enqueued = 0u64;
    while date < end {
        let next = date + ChronoDuration::days(1);
        let request = BackfillRequest {
            ingester: IngesterKey::PmdataChainlinkBtcusdRefprice,
            request_version: BACKFILL_REQUEST_VERSION,
            range_start: date.and_hms_opt(0, 0, 0).unwrap().and_utc(),
            range_end: next.and_hms_opt(0, 0, 0).unwrap().and_utc(),
            parameters: json!({}),
            idempotency_key: format!(
                "pmdata_chainlink_btcusd_refprice:{date}:{MATERIALIZATION_CONTRACT}"
            ),
        }
        .validate()
        .with_context(|| format!("invalid PMData RefPrice shard request for {date}"))?;
        let job = repository
            .enqueue(&request)
            .await
            .with_context(|| format!("failed to enqueue PMData RefPrice shard for {date}"))?;
        info!(job_id = %job.job_id, shard_date = %date, status = ?job.status, "PMData RefPrice shard ready");
        enqueued = enqueued.saturating_add(1);
        date = next;
    }
    info!(%start, %end, enqueued, "PMData RefPrice historical plan ready");
    Ok(())
}

fn configured_date(key: &str, default: &str) -> Result<NaiveDate> {
    let value = env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default.to_string());
    NaiveDate::parse_from_str(&value, "%Y-%m-%d")
        .with_context(|| format!("{key} must be YYYY-MM-DD"))
}
