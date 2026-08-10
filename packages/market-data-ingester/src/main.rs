use anyhow::Result;
use market_data_ingester::{bootstrap::Application, telemetry};

#[tokio::main]
async fn main() -> Result<()> {
    telemetry::initialize()?;
    Application::from_environment().await?.run().await
}
