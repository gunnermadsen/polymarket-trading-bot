use std::env;

use anyhow::{bail, Context, Result};
use chrono::{Duration as ChronoDuration, NaiveDate, TimeZone, Utc};
use polymarket_bot::{
    config::AppConfig,
    ingestion::{
        job::{BackfillRequest, IngesterKey, BACKFILL_REQUEST_VERSION},
        kraken_spot_trades::KRAKEN_SPOT_SCHEMA_VERSION,
        repository::IngestionRepository,
    },
};
use serde_json::json;
use sqlx::postgres::PgPoolOptions;
use tracing::info;
use tracing_subscriber::EnvFilter;

const FIXED_END_EXCLUSIVE: &str = "2026-08-31";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .init();
    let app = AppConfig::from_env()?;
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(&app.postgres.database_url())
        .await
        .context("failed to connect Kraken Spot planner to PostgreSQL")?;
    let repository = IngestionRepository::from_pool(pool);
    let start = tier_start()?;
    let fixed_end = NaiveDate::parse_from_str(FIXED_END_EXCLUSIVE, "%Y-%m-%d")?
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc();
    let now = Utc::now();
    let cutoff = fixed_end.min(
        Utc.timestamp_opt(now.timestamp(), 0)
            .single()
            .context("current UTC time is invalid")?,
    );
    let mut shard_start = start.and_hms_opt(0, 0, 0).unwrap().and_utc();
    let mut inserted = 0u64;
    while shard_start < cutoff {
        let shard_end = (shard_start + ChronoDuration::days(1)).min(cutoff);
        let request = BackfillRequest {
            ingester: IngesterKey::KrakenSpotBtcusdTradePrintsOneSecondOhlcv,
            request_version: BACKFILL_REQUEST_VERSION,
            range_start: shard_start,
            range_end: shard_end,
            parameters: json!({}),
            idempotency_key: format!(
                "kraken-spot-btcusd:{}:{}:{}",
                shard_start.format("%Y%m%dT%H%M%SZ"),
                shard_end.format("%Y%m%dT%H%M%SZ"),
                KRAKEN_SPOT_SCHEMA_VERSION
            ),
        }
        .validate()
        .with_context(|| format!("invalid Kraken Spot shard starting {shard_start}"))?;
        repository
            .enqueue(&request)
            .await
            .with_context(|| format!("failed to enqueue Kraken Spot shard {shard_start}"))?;
        inserted = inserted.saturating_add(1);
        shard_start = shard_end;
    }
    info!(%start, %cutoff, inserted, "Kraken Spot trade-print plan ready");
    Ok(())
}

fn tier_start() -> Result<NaiveDate> {
    let tier =
        env::var("POLYMARKET_KRAKEN_SPOT_BACKFILL_TIER").unwrap_or_else(|_| "april".to_string());
    let value = match tier.as_str() {
        "april" => "2026-04-01",
        "may" => "2026-05-01",
        "june" => "2026-06-01",
        _ => bail!("POLYMARKET_KRAKEN_SPOT_BACKFILL_TIER must be april, may, or june"),
    };
    NaiveDate::parse_from_str(value, "%Y-%m-%d").context("invalid Kraken Spot tier date")
}
