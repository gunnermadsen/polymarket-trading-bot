use anyhow::{Context, Result};
use polymarket_bot::{
    config::AppConfig,
    kraken_ingestion::{
        job::DEFAULT_SYMBOL, planner::enqueue_historical_plan, repository::KrakenRepository,
    },
};
use sqlx::postgres::PgPoolOptions;
use tracing::info;
use tracing_subscriber::EnvFilter;

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
        .context("failed to connect Kraken planner to PostgreSQL")?;
    let repository = KrakenRepository::from_pool(pool);
    let symbol = std::env::var("KRAKEN_BACKFILL_SYMBOL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_SYMBOL.to_string())
        .to_ascii_uppercase();
    let inserted = enqueue_historical_plan(&repository, &symbol).await?;
    info!(symbol, inserted, "Kraken historical plan enqueued");
    Ok(())
}
