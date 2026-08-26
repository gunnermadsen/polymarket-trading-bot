use std::env;

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
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

const DEFAULT_START: &str = "2026-08-14T18:00:00Z";
const DEFAULT_END: &str = "2026-08-25T00:00:00Z";
const MATERIALIZATION_CONTRACT: &str = "local-orderbook-capacity-vwap-v1";
const RETRY_GENERATION_ENV: &str = "POLYMARKET_LOCAL_VWAP_RETRY_GENERATION";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .init();
    let start = configured_time("POLYMARKET_LOCAL_VWAP_START", DEFAULT_START)?;
    let end = configured_time("POLYMARKET_LOCAL_VWAP_END", DEFAULT_END)?;
    let retry_generation = env::var(RETRY_GENERATION_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "0".to_string())
        .parse::<u32>()
        .with_context(|| format!("{RETRY_GENERATION_ENV} must be an unsigned integer"))?;
    if end <= start {
        bail!("local VWAP end must be later than start");
    }
    let app = AppConfig::from_env()?;
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(&app.postgres.database_url())
        .await
        .context("failed to connect local VWAP planner to PostgreSQL")?;
    let repository = IngestionRepository::from_pool(pool);
    let mut hour = start;
    let mut enqueued = 0u64;
    while hour < end {
        let next_hour = hour + Duration::hours(1);
        let request = BackfillRequest {
            ingester: IngesterKey::PolymarketBtcFiveMinuteExecutionSnapshots,
            request_version: BACKFILL_REQUEST_VERSION,
            range_start: hour,
            range_end: next_hour,
            parameters: json!({"source": "local_orderbook"}),
            idempotency_key: format!(
                "polymarket_local_orderbook_capacity:{}:{MATERIALIZATION_CONTRACT}:retry-{retry_generation}",
                hour.format("%Y-%m-%dT%H"),
            ),
        }
        .validate()
        .with_context(|| format!("invalid local VWAP shard request for {hour}"))?;
        let job = repository
            .enqueue(&request)
            .await
            .with_context(|| format!("failed to enqueue local VWAP shard for {hour}"))?;
        info!(job_id = %job.job_id, shard_hour = %hour, status = ?job.status, "local VWAP shard ready");
        enqueued = enqueued.saturating_add(1);
        hour = next_hour;
    }
    info!(%start, %end, enqueued, retry_generation, "local VWAP historical plan ready");
    Ok(())
}

fn configured_time(key: &str, default: &str) -> Result<DateTime<Utc>> {
    env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
        .parse()
        .with_context(|| format!("{key} must be an RFC3339 timestamp"))
}
