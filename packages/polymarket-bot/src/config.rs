use std::{env, fmt, time::Duration};

use crate::btc::{BtcHeartbeatConfig, DirectionalExternalRuntimeConfig};
use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub live: LiveExecutionConfig,
    pub gamma_base_url: String,
    pub clob_base_url: String,
    pub clob_ws_url: String,
    pub data_api_base_url: String,
    pub health_interval: Duration,
    pub postgres: PostgresConfig,
    pub http: HttpConfig,
    pub btc: BtcConfig,
    pub grafana_live: GrafanaLiveConfig,
}

#[derive(Clone)]
pub struct GrafanaLiveConfig {
    pub enabled: bool,
    pub push_url: String,
    pub publish_interval: Duration,
    pub bearer_token: Option<String>,
    pub basic_auth_username: Option<String>,
    pub basic_auth_password: Option<String>,
}

impl fmt::Debug for GrafanaLiveConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GrafanaLiveConfig")
            .field("enabled", &self.enabled)
            .field("push_url", &self.push_url)
            .field("publish_interval", &self.publish_interval)
            .field(
                "bearer_token",
                &self.bearer_token.as_ref().map(|_| "[redacted]"),
            )
            .field("basic_auth_username", &self.basic_auth_username)
            .field(
                "basic_auth_password",
                &self.basic_auth_password.as_ref().map(|_| "[redacted]"),
            )
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct BtcConfig {
    pub rtds_ws_url: String,
    pub binance_ws_url: String,
    pub binance_rest_base_url: String,
    pub data_source_heartbeat: BtcHeartbeatConfig,
    pub directional_external: DirectionalExternalRuntimeConfig,
}

#[derive(Debug, Clone)]
pub struct PostgresConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub user: String,
    pub password: String,
    pub ssl_mode: String,
    pub ssl_root_cert: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LiveExecutionConfig {
    pub user_ws_url: String,
    pub clob_api_base_url: String,
    pub user_ws_stale: Duration,
    pub reconcile_interval: Duration,
    pub stale_reconcile: Duration,
    pub clob_api_key: Option<String>,
    pub clob_secret: Option<String>,
    pub clob_passphrase: Option<String>,
    pub private_key: Option<String>,
    pub funder_address: Option<String>,
    pub signature_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct HttpConfig {
    pub enabled: bool,
    pub bind: String,
    pub admin_token: String,
}

impl AppConfig {
    pub fn from_env() -> Result<Self> {
        let clob_base_url = env_or("POLYMARKET_CLOB_BASE_URL", "https://clob.polymarket.com");
        let live = LiveExecutionConfig {
            user_ws_url: env_or(
                "POLYMARKET_LIVE_USER_WS_URL",
                "wss://ws-subscriptions-clob.polymarket.com/ws/user",
            ),
            clob_api_base_url: clob_base_url.clone(),
            user_ws_stale: Duration::from_secs(parse_u64("POLYMARKET_LIVE_USER_WS_STALE_SECS", 20)),
            reconcile_interval: Duration::from_secs(parse_u64(
                "POLYMARKET_LIVE_RECONCILE_INTERVAL_SECS",
                30,
            )),
            stale_reconcile: Duration::from_secs(parse_u64(
                "POLYMARKET_LIVE_STALE_RECONCILE_SECS",
                60,
            )),
            clob_api_key: first_non_empty_env(&["POLYMARKET_CLOB_API_KEY"]),
            clob_secret: first_non_empty_env(&["POLYMARKET_CLOB_SECRET"]),
            clob_passphrase: first_non_empty_env(&["POLYMARKET_CLOB_PASSPHRASE"]),
            private_key: first_non_empty_env(&["POLYMARKET_PRIVATE_KEY"]),
            funder_address: first_non_empty_env(&["POLYMARKET_FUNDER_ADDRESS"]),
            signature_type: first_non_empty_env(&["POLYMARKET_SIGNATURE_TYPE"]),
        };
        if live.live_auth_available() {
            live.validate_for_live()?;
        }

        let heartbeat_defaults = BtcHeartbeatConfig::default();
        let btc = BtcConfig {
            rtds_ws_url: env_or(
                "POLYMARKET_BTC_RTDS_WS_URL",
                "wss://ws-live-data.polymarket.com",
            ),
            binance_ws_url: env_or(
                "POLYMARKET_BTC_BINANCE_WS_URL",
                "wss://stream.binance.com/ws/btcusdt@aggTrade",
            ),
            binance_rest_base_url: env_or(
                "POLYMARKET_BTC_BINANCE_REST_BASE_URL",
                "https://data-api.binance.vision",
            ),
            data_source_heartbeat: BtcHeartbeatConfig {
                clob_interval: parse_positive_duration_secs(
                    "POLYMARKET_BTC_CLOB_HEARTBEAT_INTERVAL_SECS",
                    heartbeat_defaults.clob_interval.as_secs(),
                )?,
                clob_pong_timeout: parse_clob_pong_timeout_secs(
                    "POLYMARKET_BTC_CLOB_PONG_TIMEOUT_SECS",
                    heartbeat_defaults.clob_pong_timeout.as_secs(),
                )?,
                rtds_interval: parse_positive_duration_secs(
                    "POLYMARKET_BTC_RTDS_HEARTBEAT_INTERVAL_SECS",
                    heartbeat_defaults.rtds_interval.as_secs(),
                )?,
                binance_interval: parse_positive_duration_secs(
                    "POLYMARKET_BTC_BINANCE_HEARTBEAT_INTERVAL_SECS",
                    heartbeat_defaults.binance_interval.as_secs(),
                )?,
            },
            directional_external: DirectionalExternalRuntimeConfig::from_env()?,
        };
        let grafana_live = GrafanaLiveConfig {
            enabled: parse_bool("POLYMARKET_GRAFANA_LIVE_ENABLED", false),
            push_url: env_or(
                "POLYMARKET_GRAFANA_LIVE_PUSH_URL",
                "http://grafana:3000/api/live/push/polymarket",
            ),
            publish_interval: Duration::from_millis(parse_u64(
                "POLYMARKET_GRAFANA_LIVE_PUBLISH_INTERVAL_MS",
                1_000,
            )),
            bearer_token: first_non_empty_env(&["POLYMARKET_GRAFANA_LIVE_BEARER_TOKEN"]),
            basic_auth_username: first_non_empty_env(&[
                "POLYMARKET_GRAFANA_LIVE_USERNAME",
                "GRAFANA_ADMIN_USER",
            ]),
            basic_auth_password: first_non_empty_env(&[
                "POLYMARKET_GRAFANA_LIVE_PASSWORD",
                "GRAFANA_ADMIN_PASSWORD",
            ]),
        };
        if grafana_live.enabled {
            if !grafana_live.push_url.starts_with("http://")
                && !grafana_live.push_url.starts_with("https://")
            {
                bail!("POLYMARKET_GRAFANA_LIVE_PUSH_URL must be an HTTP(S) URL");
            }
            if grafana_live.publish_interval < Duration::from_millis(250)
                || grafana_live.publish_interval > Duration::from_secs(5)
            {
                bail!("POLYMARKET_GRAFANA_LIVE_PUBLISH_INTERVAL_MS must be between 250 and 5000");
            }
            let basic_auth_complete = grafana_live.basic_auth_username.is_some()
                && grafana_live.basic_auth_password.is_some();
            if grafana_live.bearer_token.is_none() && !basic_auth_complete {
                bail!(
                    "Grafana Live requires POLYMARKET_GRAFANA_LIVE_BEARER_TOKEN or complete Grafana basic-auth credentials"
                );
            }
        }
        Ok(Self {
            live,
            gamma_base_url: env_or(
                "POLYMARKET_GAMMA_BASE_URL",
                "https://gamma-api.polymarket.com",
            ),
            clob_base_url,
            clob_ws_url: env_or(
                "POLYMARKET_CLOB_WS_URL",
                "wss://ws-subscriptions-clob.polymarket.com/ws/market",
            ),
            data_api_base_url: env_or(
                "POLYMARKET_DATA_API_BASE_URL",
                "https://data-api.polymarket.com",
            ),
            health_interval: Duration::from_secs(parse_u64("POLYMARKET_HEALTH_INTERVAL_SECS", 30)),
            postgres: PostgresConfig {
                host: env_or("POSTGRES_HOST", "localhost"),
                port: parse_u16("POSTGRES_PORT", 5432),
                database: env_or("POSTGRES_DB", "polymarket"),
                user: env_or("POSTGRES_USER", "postgres"),
                password: required_env("POSTGRES_PASSWORD")?,
                ssl_mode: env_or("POSTGRES_SSL_MODE", "disable"),
                ssl_root_cert: first_non_empty_env(&["POSTGRES_SSL_CA_FILE", "PGSSLROOTCERT"]),
            },
            http: HttpConfig {
                enabled: parse_bool("POLYMARKET_HTTP_ENABLED", true),
                bind: env_or("POLYMARKET_HTTP_BIND", "0.0.0.0:8097"),
                admin_token: env_or("POLYMARKET_HTTP_ADMIN_TOKEN", "dev-polymarket-admin"),
            },
            btc,
            grafana_live,
        })
    }
}

impl LiveExecutionConfig {
    pub fn validate_for_live(&self) -> Result<()> {
        if self.user_ws_url.trim().is_empty() {
            bail!("POLYMARKET_LIVE_USER_WS_URL is required for live execution");
        }
        if self.clob_api_base_url.trim().is_empty() {
            bail!("POLYMARKET_CLOB_BASE_URL is required for live execution");
        }
        let mut missing = Vec::new();
        for (key, value) in [
            ("POLYMARKET_CLOB_API_KEY", &self.clob_api_key),
            ("POLYMARKET_CLOB_SECRET", &self.clob_secret),
            ("POLYMARKET_CLOB_PASSPHRASE", &self.clob_passphrase),
        ] {
            if value.is_none() {
                missing.push(key);
            }
        }
        if self.submit_auth_available() {
            for (key, value) in [
                ("POLYMARKET_PRIVATE_KEY", &self.private_key),
                ("POLYMARKET_FUNDER_ADDRESS", &self.funder_address),
                ("POLYMARKET_SIGNATURE_TYPE", &self.signature_type),
            ] {
                if value.is_none() {
                    missing.push(key);
                }
            }
        }
        missing.sort_unstable();
        missing.dedup();
        if !missing.is_empty() {
            bail!(
                "live mode missing required env vars: {}",
                missing.join(", ")
            );
        }
        Ok(())
    }

    pub fn clob_auth_available(&self) -> bool {
        self.user_ws_auth_available() && self.submit_auth_available()
    }

    pub fn live_auth_available(&self) -> bool {
        self.user_ws_auth_available() || self.submit_auth_available()
    }

    pub fn user_ws_auth_available(&self) -> bool {
        self.clob_api_key.is_some() && self.clob_secret.is_some() && self.clob_passphrase.is_some()
    }

    pub fn submit_auth_available(&self) -> bool {
        self.clob_api_key.is_some()
            && self.clob_secret.is_some()
            && self.clob_passphrase.is_some()
            && self.private_key.is_some()
            && self.funder_address.is_some()
            && self.signature_type.is_some()
    }
}

impl PostgresConfig {
    pub fn database_url(&self) -> String {
        let mut url = format!(
            "postgres://{}:{}@{}:{}/{}?sslmode={}",
            self.user, self.password, self.host, self.port, self.database, self.ssl_mode
        );
        if let Some(root_cert) = &self.ssl_root_cert {
            url.push_str("&sslrootcert=");
            url.push_str(root_cert);
        }
        url
    }
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn required_env(key: &str) -> Result<String> {
    match env::var(key).ok().filter(|value| !value.trim().is_empty()) {
        Some(value) => Ok(value),
        None => bail!("{key} is required"),
    }
}

fn first_non_empty_env(keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| env::var(key).ok().filter(|value| !value.trim().is_empty()))
}

fn parse_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn parse_u16(key: &str, default: u16) -> u16 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_positive_duration_secs(key: &str, default: u64) -> Result<Duration> {
    let value = env::var(key).ok().filter(|value| !value.trim().is_empty());
    positive_duration_secs(key, value.as_deref(), default)
}

fn positive_duration_secs(key: &str, value: Option<&str>, default: u64) -> Result<Duration> {
    bounded_positive_duration_secs(key, value, default, BtcHeartbeatConfig::MAX_INTERVAL_SECS)
}

fn parse_clob_pong_timeout_secs(key: &str, default: u64) -> Result<Duration> {
    bounded_positive_duration_secs(
        key,
        env::var(key).ok().as_deref(),
        default,
        BtcHeartbeatConfig::MAX_CLOB_PONG_TIMEOUT_SECS,
    )
}

fn bounded_positive_duration_secs(
    key: &str,
    value: Option<&str>,
    default: u64,
    maximum: u64,
) -> Result<Duration> {
    let seconds = match value {
        Some(value) => match value.parse::<u64>() {
            Ok(seconds) => seconds,
            Err(_) => bail!("{key} must be a positive integer number of seconds"),
        },
        None => default,
    };
    if seconds == 0 || seconds > maximum {
        bail!("{key} must be an integer between 1 and {maximum} seconds");
    }
    Ok(Duration::from_secs(seconds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heartbeat_intervals_parse_strictly_without_silent_fallbacks() {
        let key = "POLYMARKET_BTC_CLOB_HEARTBEAT_INTERVAL_SECS";
        assert_eq!(
            positive_duration_secs(key, None, 5).unwrap(),
            Duration::from_secs(5)
        );
        assert_eq!(
            positive_duration_secs(key, Some("7"), 5).unwrap(),
            Duration::from_secs(7)
        );
        assert!(positive_duration_secs(key, Some("0"), 5).is_err());
        assert!(positive_duration_secs(key, Some("invalid"), 5).is_err());
        assert!(positive_duration_secs(key, Some("-1"), 5).is_err());
        assert!(
            positive_duration_secs(key, Some("31"), BtcHeartbeatConfig::MAX_INTERVAL_SECS).is_err()
        );
        assert_eq!(
            parse_clob_pong_timeout_secs("POLYMARKET_BTC_CLOB_PONG_TIMEOUT_SECS", 25).unwrap(),
            Duration::from_secs(25)
        );
        assert!(bounded_positive_duration_secs(
            "POLYMARKET_BTC_CLOB_PONG_TIMEOUT_SECS",
            Some("61"),
            25,
            BtcHeartbeatConfig::MAX_CLOB_PONG_TIMEOUT_SECS,
        )
        .is_err());
    }
}
