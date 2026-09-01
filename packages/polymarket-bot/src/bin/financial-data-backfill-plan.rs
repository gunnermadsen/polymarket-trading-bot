use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .json()
        .init();
    let inserted = polymarket_bot::financial_data_ingestion::enqueue_plan().await?;
    println!("enqueued {inserted} financial-data backfill jobs");
    Ok(())
}
