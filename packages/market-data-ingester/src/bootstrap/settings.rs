use std::{env, net::SocketAddr};

use anyhow::{bail, Context, Result};

const DEFAULT_API_BIND: &str = "0.0.0.0:8098";
const DEFAULT_POOL_CONNECTIONS: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BootstrapSettings {
    pub database_url: String,
    pub database_pool_connections: u32,
    pub service_instance: String,
    pub api_bind: SocketAddr,
}

impl BootstrapSettings {
    pub fn from_environment() -> Result<Self> {
        let database_url = required("DATABASE_URL")?;
        let service_instance = env::var("MARKET_DATA_INGESTER_INSTANCE")
            .unwrap_or_else(|_| "market-data-ingester-1".to_owned());
        let api_bind = env::var("MARKET_DATA_INGESTER_API_BIND")
            .unwrap_or_else(|_| DEFAULT_API_BIND.to_owned())
            .parse()
            .context("MARKET_DATA_INGESTER_API_BIND must be a socket address")?;
        let database_pool_connections = env::var("MARKET_DATA_INGESTER_DB_POOL_CONNECTIONS")
            .map_or(Ok(DEFAULT_POOL_CONNECTIONS), |value| {
                value
                    .parse::<u32>()
                    .context("MARKET_DATA_INGESTER_DB_POOL_CONNECTIONS must be an integer")
            })?;
        if service_instance.trim().is_empty() {
            bail!("MARKET_DATA_INGESTER_INSTANCE must not be empty");
        }
        if !(1..=8).contains(&database_pool_connections) {
            bail!("MARKET_DATA_INGESTER_DB_POOL_CONNECTIONS must be between 1 and 8");
        }
        Ok(Self {
            database_url,
            database_pool_connections,
            service_instance,
            api_bind,
        })
    }
}

fn required(key: &str) -> Result<String> {
    let value = env::var(key).with_context(|| format!("{key} is required"))?;
    if value.trim().is_empty() {
        bail!("{key} must not be empty");
    }
    Ok(value)
}
