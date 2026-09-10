use std::{sync::Arc, time::Duration};

use anyhow::{anyhow, Context, Result};
use axum::{routing::get, Router};
use sqlx::postgres::PgPoolOptions;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::{
    control::{ControlApi, ControlReadiness},
    persistence::ProfileRepository,
    runtime::{
        BackfillWorkerRuntime, DrainWorkerRuntime, StrategyRegistry, StrategySupervisor,
        SupervisorSettings,
    },
    streaming::Publisher,
};

use super::{shutdown_signal, BootstrapSettings, IngesterMode};

const COMPONENT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(45);
const DATABASE_HEALTHCHECK_RETRY_INTERVAL: Duration = Duration::from_secs(1);

async fn await_pool_health(pool: &sqlx::PgPool, pool_name: &'static str) {
    loop {
        match sqlx::query_scalar::<_, i32>("SELECT 1")
            .fetch_one(pool)
            .await
        {
            Ok(_) => return,
            Err(error) => {
                warn!(
                    pool = pool_name,
                    error = %error,
                    retry_after_ms = DATABASE_HEALTHCHECK_RETRY_INTERVAL.as_millis(),
                    "market-data ingester database pool healthcheck deferred"
                );
                tokio::time::sleep(DATABASE_HEALTHCHECK_RETRY_INTERVAL).await;
            }
        }
    }
}

pub struct Application {
    settings: BootstrapSettings,
    strategy_pool: sqlx::PgPool,
    control_pool: sqlx::PgPool,
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
        let strategy_pool = PgPoolOptions::new()
            .max_connections(settings.database_pool_connections)
            .acquire_timeout(settings.database_acquire_timeout)
            .connect_with(settings.database.clone())
            .await
            .context("failed to connect market-data ingester strategy pool to TimescaleDB")?;
        let control_pool = PgPoolOptions::new()
            .max_connections(settings.control_database_pool_connections)
            .acquire_timeout(settings.control_database_acquire_timeout)
            .connect_with(settings.database.clone())
            .await
            .context("failed to connect market-data ingester control pool to TimescaleDB")?;
        await_pool_health(&strategy_pool, "strategy").await;
        await_pool_health(&control_pool, "control").await;
        Ok(Self {
            settings,
            strategy_pool,
            control_pool,
            registry,
        })
    }

    pub async fn run(self) -> Result<()> {
        match self.settings.mode {
            IngesterMode::Master => self.run_master().await,
            IngesterMode::Worker => self.run_worker().await,
        }
    }

    async fn run_master(self) -> Result<()> {
        let profiles = ProfileRepository::new(self.control_pool.clone());
        let (_readiness_sender, readiness) = ControlReadiness::channel(true);
        let shutdown = CancellationToken::new();
        let api = ControlApi::new(
            profiles.clone(),
            self.registry.clone(),
            self.settings.admin_token.clone(),
            readiness,
        )?;
        info!(
            service_instance = %self.settings.service_instance,
            api_bind = %self.settings.api_bind,
            strategy_database_pool_connections = self.settings.database_pool_connections,
            control_database_pool_connections = self.settings.control_database_pool_connections,
            "ingester master starting"
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
        let component_failure = tokio::select! {
            _ = shutdown_signal() => {
                info!("ingester master shutdown requested");
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

        self.strategy_pool.close().await;
        self.control_pool.close().await;
        info!("ingester master shutdown completed");
        match component_failure {
            Some(error) => Err(error),
            None => match shutdown_failure {
                Some(error) => Err(error),
                None => Ok(()),
            },
        }
    }

    async fn run_worker(self) -> Result<()> {
        let profiles = ProfileRepository::new(self.control_pool.clone());
        let (readiness_sender, _readiness) = ControlReadiness::channel(false);
        let supervisor = StrategySupervisor::new(
            profiles,
            self.registry.clone(),
            self.strategy_pool.clone(),
            self.settings.service_instance.clone(),
            SupervisorSettings::default(),
        )?;
        let backfills = BackfillWorkerRuntime::from_environment(
            self.registry.clone(),
            self.strategy_pool.clone(),
        )?;
        let drains = DrainWorkerRuntime::from_environment(
            self.registry,
            self.strategy_pool.clone(),
            self.control_pool.clone(),
        )?;
        let shutdown = CancellationToken::new();
        let publisher = Publisher::install(self.settings.service_instance.clone());
        info!(service_instance=%self.settings.service_instance, "ingester worker starting");
        let mut components = JoinSet::new();
        let realtime_shutdown = shutdown.clone();
        components.spawn(async move {
            (
                "realtime strategy supervisor",
                supervisor
                    .run(realtime_shutdown, readiness_sender)
                    .await
                    .context("ingester realtime supervisor stopped"),
            )
        });
        let drain_shutdown = shutdown.clone();
        components.spawn(async move {
            (
                "drain worker",
                drains
                    .run(drain_shutdown)
                    .await
                    .context("ingester drain worker stopped"),
            )
        });
        let stream_shutdown = shutdown.clone();
        let grpc_bind = self.settings.grpc_bind;
        let stream_token = Arc::<str>::from(self.settings.admin_token.clone());
        let stream_publisher = publisher.clone();
        components.spawn(async move {
            (
                "market-data gRPC server",
                stream_publisher
                    .serve(grpc_bind, stream_token, stream_shutdown)
                    .await
                    .context("ingester market-data gRPC server stopped"),
            )
        });
        let metrics_shutdown = shutdown.clone();
        let metrics_bind = self.settings.worker_metrics_bind;
        components.spawn(async move {
            let metrics_publisher = publisher.clone();
            let app = Router::new()
                .route("/health/live", get(|| async { "ok\n" }))
                .route(
                    "/prometheus/metrics",
                    get(move || {
                        let publisher = metrics_publisher.clone();
                        async move { publisher.render_metrics() }
                    }),
                );
            let result = async {
                let listener = tokio::net::TcpListener::bind(metrics_bind).await?;
                axum::serve(listener, app)
                    .with_graceful_shutdown(metrics_shutdown.cancelled_owned())
                    .await?;
                Ok(())
            }
            .await;
            ("worker telemetry API", result)
        });
        let backfill_shutdown = shutdown.clone();
        components.spawn(async move {
            (
                "backfill worker",
                backfills
                    .run(backfill_shutdown)
                    .await
                    .context("ingester backfill worker stopped"),
            )
        });
        let component_failure = tokio::select! {
            _ = shutdown_signal() => { info!("ingester worker shutdown requested"); None }
            completed = components.join_next() => Some(unexpected_component_exit(completed)),
        };
        shutdown.cancel();
        let shutdown_failure = match tokio::time::timeout(COMPONENT_SHUTDOWN_TIMEOUT, async {
            let mut first_failure = None;
            while let Some(completed) = components.join_next().await {
                if let Err(error) = component_result(completed) {
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
                components.abort_all();
                while components.join_next().await.is_some() {}
                Some(anyhow!(
                    "ingester worker components exceeded the shutdown deadline"
                ))
            }
        };
        self.strategy_pool.close().await;
        self.control_pool.close().await;
        info!("ingester worker shutdown completed");
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
