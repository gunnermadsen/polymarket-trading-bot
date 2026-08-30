use anyhow::Result;
use polymarket_bot::ingestion::kraken_spot_l2_archive::KrakenSpotL2ArchiveWorker;
use tokio::sync::watch;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .init();

    let worker = KrakenSpotL2ArchiveWorker::new()?;
    let (shutdown_sender, shutdown_receiver) = watch::channel(false);
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        info!("shutdown requested for Kraken Spot L2 backfill worker");
        let _ = shutdown_sender.send(true);
    });
    worker.run(shutdown_receiver).await
}
