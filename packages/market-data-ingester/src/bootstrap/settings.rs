use std::{env, net::SocketAddr};

use anyhow::{bail, Context, Result};
use sqlx::postgres::{PgConnectOptions, PgSslMode};

const DEFAULT_API_BIND: &str = "0.0.0.0:8098";
const DEFAULT_POOL_CONNECTIONS: u32 = 4;

#[derive(Clone)]
pub(crate) struct BootstrapSettings {
    pub database: PgConnectOptions,
    pub database_pool_connections: u32,
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
        if !(32..=4096).contains(&admin_token.len()) {
            bail!("MARKET_DATA_INGESTER_ADMIN_TOKEN must be between 32 and 4096 bytes");
        }
        Ok(Self {
            database,
            database_pool_connections,
            service_instance,
            api_bind,
            admin_token,
        })
    }
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
