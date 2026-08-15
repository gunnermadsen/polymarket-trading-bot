use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use sqlx::postgres::PgPoolOptions;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::{
    control::{ControlApi, ControlReadiness},
    persistence::ProfileRepository,
    runtime::{StrategyRegistry, StrategySupervisor, SupervisorSettings},
};

use super::{shutdown_signal, BootstrapSettings};

const COMPONENT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(45);

pub struct Application {
    settings: BootstrapSettings,
    pool: sqlx::PgPool,
    registry: StrategyRegistry,
}

impl Application {
    pub async fn from_environment() -> Result<Self> {
        let registry = crate::strategies::registry()
            .context("failed to register market-data ingestion strategies")?;
        Self::from_environment_with_registry(registry).await
    }

    pub async fn from_environment_with_registry(registry: StrategyRegistry) -> Result<Self> {
        let settings = BootstrapSettings::from_environment()?;
        let pool = PgPoolOptions::new()
            .max_connections(settings.database_pool_connections)
            .acquire_timeout(Duration::from_secs(5))
            .connect_with(settings.database.clone())
            .await
            .context("failed to connect market-data ingester to TimescaleDB")?;
        sqlx::query_scalar::<_, i32>("SELECT 1")
            .fetch_one(&pool)
            .await
            .context("market-data ingester database healthcheck failed")?;
        Ok(Self {
            settings,
            pool,
            registry,
        })
    }

    pub async fn run(self) -> Result<()> {
        let profiles = ProfileRepository::new(self.pool.clone());
        let (readiness_sender, readiness) = ControlReadiness::channel(false);
        let shutdown = CancellationToken::new();
        let api = ControlApi::new(
            profiles.clone(),
            self.registry.clone(),
            self.settings.admin_token.clone(),
            readiness,
        )?;
        let supervisor = StrategySupervisor::new(
            profiles,
            self.registry,
            self.pool.clone(),
            self.settings.service_instance.clone(),
            SupervisorSettings::default(),
        )?;

        info!(
            service_instance = %self.settings.service_instance,
            api_bind = %self.settings.api_bind,
            "market-data ingester starting"
        );

        let mut components = JoinSet::new();
        let api_shutdown = shutdown.clone();
        let api_bind = self.settings.api_bind;
        components.spawn(async move {
            (
                "control API",
                api.serve(api_bind, api_shutdown)
                    .await
                    .context("market-data ingester control API stopped"),
            )
        });
        let supervisor_shutdown = shutdown.clone();
        components.spawn(async move {
            (
                "strategy supervisor",
                supervisor
                    .run(supervisor_shutdown, readiness_sender)
                    .await
                    .context("market-data ingester strategy supervisor stopped"),
            )
        });

        let component_failure = tokio::select! {
            _ = shutdown_signal() => {
                info!("market-data ingester shutdown requested");
                None
            }
            completed = components.join_next() => Some(unexpected_component_exit(completed)),
        };

        shutdown.cancel();
        let shutdown_failure = match tokio::time::timeout(COMPONENT_SHUTDOWN_TIMEOUT, async {
            let mut first_failure = None;
            while let Some(completed) = components.join_next().await {
                if let Err(error) = component_result(completed) {
                    warn!(error = %error, "market-data ingester component failed during shutdown");
                    if first_failure.is_none() {
                        first_failure = Some(error);
                    }
                }
            }
            first_failure
        })
        .await
        {
            Ok(failure) => failure,
            Err(_) => {
                warn!("market-data ingester components exceeded the shutdown deadline");
                components.abort_all();
                while components.join_next().await.is_some() {}
                Some(anyhow!(
                    "market-data ingester components exceeded the shutdown deadline"
                ))
            }
        };

        self.pool.close().await;
        info!("market-data ingester shutdown completed");
        match component_failure {
            Some(error) => Err(error),
            None => match shutdown_failure {
                Some(error) => Err(error),
                None => Ok(()),
            },
        }
    }
}

fn unexpected_component_exit(
    completed: Option<Result<(&'static str, Result<()>), tokio::task::JoinError>>,
) -> anyhow::Error {
    match completed {
        Some(Ok((name, Ok(())))) => anyhow!("market-data ingester {name} exited unexpectedly"),
        Some(Ok((_name, Err(error)))) => error,
        Some(Err(error)) => anyhow!("market-data ingester component task failed: {error}"),
        None => anyhow!("all market-data ingester components exited unexpectedly"),
    }
}

fn component_result(
    completed: Result<(&'static str, Result<()>), tokio::task::JoinError>,
) -> Result<()> {
    match completed {
        Ok((_name, result)) => result,
        Err(error) if error.is_cancelled() => Ok(()),
        Err(error) => Err(anyhow!(
            "market-data ingester component task failed: {error}"
        )),
    }
}
