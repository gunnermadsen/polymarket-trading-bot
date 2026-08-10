use anyhow::{Context, Result};
use sqlx::postgres::PgPoolOptions;
use tracing::info;

use super::{shutdown_signal, BootstrapSettings};

pub struct Application {
    settings: BootstrapSettings,
    pool: sqlx::PgPool,
}

impl Application {
    pub async fn from_environment() -> Result<Self> {
        let settings = BootstrapSettings::from_environment()?;
        let pool = PgPoolOptions::new()
            .max_connections(settings.database_pool_connections)
            .connect(&settings.database_url)
            .await
            .context("failed to connect market-data ingester to TimescaleDB")?;
        sqlx::query_scalar::<_, i32>("SELECT 1")
            .fetch_one(&pool)
            .await
            .context("market-data ingester database healthcheck failed")?;
        Ok(Self { settings, pool })
    }

    pub async fn run(self) -> Result<()> {
        info!(
            service_instance = %self.settings.service_instance,
            api_bind = %self.settings.api_bind,
            "market-data ingester foundation ready"
        );
        shutdown_signal().await;
        self.pool.close().await;
        info!("market-data ingester shutdown completed");
        Ok(())
    }
}
