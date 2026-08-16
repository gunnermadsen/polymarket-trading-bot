use std::{env, net::SocketAddr, time::Duration};

use anyhow::{bail, Context, Result};
use sqlx::postgres::{PgConnectOptions, PgSslMode};

const DEFAULT_API_BIND: &str = "0.0.0.0:8098";
const DEFAULT_POOL_CONNECTIONS: u32 = 4;
const DEFAULT_CONTROL_POOL_CONNECTIONS: u32 = 1;
const DEFAULT_DATABASE_ACQUIRE_TIMEOUT_SECS: u64 = 10;
const DEFAULT_CONTROL_DATABASE_ACQUIRE_TIMEOUT_SECS: u64 = 3;

#[derive(Clone)]
pub(crate) struct BootstrapSettings {
    pub database: PgConnectOptions,
    pub database_pool_connections: u32,
    pub control_database_pool_connections: u32,
    pub database_acquire_timeout: Duration,
    pub control_database_acquire_timeout: Duration,
    pub service_instance: String,
    pub api_bind: SocketAddr,
    pub admin_token: String,
}

impl BootstrapSettings {
    pub fn from_environment() -> Result<Self> {
        let database = database_options()?;
        let admin_token = required("MARKET_DATA_INGESTER_ADMIN_TOKEN")?;
        let service_instance = env::var("MARKET_DATA_INGESTER_INSTANCE")
            .unwrap_or_else(|_| "market-data-ingester-1".to_owned());
        let api_bind = env::var("MARKET_DATA_INGESTER_API_BIND")
            .unwrap_or_else(|_| DEFAULT_API_BIND.to_owned())
            .parse()
            .context("MARKET_DATA_INGESTER_API_BIND must be a socket address")?;
        let database_pool_connections = env_u32(
            "MARKET_DATA_INGESTER_DB_POOL_CONNECTIONS",
            DEFAULT_POOL_CONNECTIONS,
        )?;
        let control_database_pool_connections = env_u32(
            "MARKET_DATA_INGESTER_CONTROL_DB_POOL_CONNECTIONS",
            DEFAULT_CONTROL_POOL_CONNECTIONS,
        )?;
        let database_acquire_timeout = env_duration_secs(
            "MARKET_DATA_INGESTER_DB_ACQUIRE_TIMEOUT_SECS",
            DEFAULT_DATABASE_ACQUIRE_TIMEOUT_SECS,
        )?;
        let control_database_acquire_timeout = env_duration_secs(
            "MARKET_DATA_INGESTER_CONTROL_DB_ACQUIRE_TIMEOUT_SECS",
            DEFAULT_CONTROL_DATABASE_ACQUIRE_TIMEOUT_SECS,
        )?;
        if service_instance.trim().is_empty() {
            bail!("MARKET_DATA_INGESTER_INSTANCE must not be empty");
        }
        if !(1..=8).contains(&database_pool_connections) {
            bail!("MARKET_DATA_INGESTER_DB_POOL_CONNECTIONS must be between 1 and 8");
        }
        if !(1..=2).contains(&control_database_pool_connections) {
            bail!("MARKET_DATA_INGESTER_CONTROL_DB_POOL_CONNECTIONS must be between 1 and 2");
        }
        if !(Duration::from_secs(1)..=Duration::from_secs(60)).contains(&database_acquire_timeout) {
            bail!("MARKET_DATA_INGESTER_DB_ACQUIRE_TIMEOUT_SECS must be between 1 and 60");
        }
        if !(Duration::from_secs(1)..=Duration::from_secs(30))
            .contains(&control_database_acquire_timeout)
        {
            bail!("MARKET_DATA_INGESTER_CONTROL_DB_ACQUIRE_TIMEOUT_SECS must be between 1 and 30");
        }
        if !(32..=4096).contains(&admin_token.len()) {
            bail!("MARKET_DATA_INGESTER_ADMIN_TOKEN must be between 32 and 4096 bytes");
        }
        Ok(Self {
            database,
            database_pool_connections,
            control_database_pool_connections,
            database_acquire_timeout,
            control_database_acquire_timeout,
            service_instance,
            api_bind,
            admin_token,
        })
    }
}

fn env_u32(key: &str, default: u32) -> Result<u32> {
    env::var(key).map_or(Ok(default), |value| {
        value
            .parse::<u32>()
            .with_context(|| format!("{key} must be an integer"))
    })
}

fn env_duration_secs(key: &str, default: u64) -> Result<Duration> {
    env::var(key).map_or(Ok(Duration::from_secs(default)), |value| {
        value
            .parse::<u64>()
            .map(Duration::from_secs)
            .with_context(|| format!("{key} must be an integer"))
    })
}

fn database_options() -> Result<PgConnectOptions> {
    if let Ok(database_url) = env::var("DATABASE_URL") {
        if database_url.trim().is_empty() {
            bail!("DATABASE_URL must not be empty when provided");
        }
        return database_url
            .parse::<PgConnectOptions>()
            .context("DATABASE_URL must be a valid PostgreSQL connection URL");
    }

    let host = env_or("POSTGRES_HOST", "timescaledb-0");
    let port = env_or("POSTGRES_PORT", "5432")
        .parse::<u16>()
        .context("POSTGRES_PORT must be an integer between 1 and 65535")?;
    if port == 0 {
        bail!("POSTGRES_PORT must be between 1 and 65535");
    }
    let user = env_or("POSTGRES_USER", "postgres");
    let password = required("POSTGRES_PASSWORD")?;
    let database = env_or("POSTGRES_DB", "polymarket");
    let ssl_mode = parse_ssl_mode(&env_or("POSTGRES_SSL_MODE", "disable"))?;

    Ok(PgConnectOptions::new()
        .host(&host)
        .port(port)
        .username(&user)
        .password(&password)
        .database(&database)
        .ssl_mode(ssl_mode))
}

fn parse_ssl_mode(value: &str) -> Result<PgSslMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "disable" => Ok(PgSslMode::Disable),
        "allow" => Ok(PgSslMode::Allow),
        "prefer" => Ok(PgSslMode::Prefer),
        "require" => Ok(PgSslMode::Require),
        "verify-ca" | "verify_ca" => Ok(PgSslMode::VerifyCa),
        "verify-full" | "verify_full" => Ok(PgSslMode::VerifyFull),
        _ => bail!(
            "POSTGRES_SSL_MODE must be disable, allow, prefer, require, verify-ca, or verify-full"
        ),
    }
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_owned())
}

fn required(key: &str) -> Result<String> {
    let value = env::var(key).with_context(|| format!("{key} is required"))?;
    if value.trim().is_empty() {
        bail!("{key} must not be empty");
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_postgres_ssl_modes() {
        assert!(matches!(
            parse_ssl_mode("disable").expect("mode"),
            PgSslMode::Disable
        ));
        assert!(matches!(
            parse_ssl_mode("VERIFY-FULL").expect("mode"),
            PgSslMode::VerifyFull
        ));
        assert!(parse_ssl_mode("unsafe").is_err());
    }
}
