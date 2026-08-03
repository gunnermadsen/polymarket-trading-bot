use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use polymarket_bot::{
    config::AppConfig,
    ingestion::{
        huggingface_binance_l2::{
            HUGGINGFACE_GOOODDY_MATERIALIZATION_CONTRACT, HUGGINGFACE_GOOODDY_STRATEGY,
        },
        job::{BackfillRequest, IngesterKey, BACKFILL_REQUEST_VERSION},
        repository::IngestionRepository,
    },
};
use serde_json::json;
use sqlx::postgres::PgPoolOptions;
use tracing::info;
use tracing_subscriber::EnvFilter;

const SHARDS: [(&str, &str, &str); 2] = [
    ("2026-06", "2026-06-03T00:00:00Z", "2026-07-01T00:00:00Z"),
    ("2026-07", "2026-07-01T00:00:00Z", "2026-08-01T00:00:00Z"),
];

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
        .context("failed to connect Hugging Face Binance spot L2 planner to PostgreSQL")?;
    let repository = IngestionRepository::from_pool(pool);
    for (month, start, end) in SHARDS {
        let request = shard_request(month, start, end)?;
        let job = repository.enqueue(&request).await.with_context(|| {
            format!("failed to enqueue Hugging Face Binance spot L2 shard {month}")
        })?;
        info!(job_id = %job.job_id, source_month = month, status = ?job.status, "Hugging Face Binance spot L2 shard ready");
    }
    Ok(())
}

fn shard_request(
    month: &str,
    start: &str,
    end: &str,
) -> Result<polymarket_bot::ingestion::job::ValidatedBackfillRequest> {
    BackfillRequest {
        ingester: IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures,
        request_version: BACKFILL_REQUEST_VERSION,
        range_start: start.parse::<DateTime<Utc>>()?,
        range_end: end.parse::<DateTime<Utc>>()?,
        parameters: json!({"strategy": HUGGINGFACE_GOOODDY_STRATEGY}),
        idempotency_key: format!(
            "binance-spot-btcusdt-l2:huggingface-goooddy:{month}:{HUGGINGFACE_GOOODDY_MATERIALIZATION_CONTRACT}"
        ),
    }
    .validate()
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_contains_only_the_two_pinned_monthly_shards() {
        assert_eq!(SHARDS.len(), 2);
        for (month, start, end) in SHARDS {
            let request = shard_request(month, start, end).unwrap();
            assert_eq!(
                request.ingester,
                IngesterKey::BinanceSpotBtcusdtL2OneSecondFeatures
            );
            assert_eq!(
                request.parameters,
                json!({"strategy": HUGGINGFACE_GOOODDY_STRATEGY})
            );
            assert!(request.expected_work_units >= 28);
        }
    }
}
