use std::{env, time::Duration};

use anyhow::{bail, Result};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Sim,
    Paper,
    Live,
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub live: LiveExecutionConfig,
    pub gamma_base_url: String,
    pub clob_base_url: String,
    pub clob_ws_url: String,
    pub data_api_base_url: String,
    pub health_interval: Duration,
    pub postgres: PostgresConfig,
    pub risk: RiskConfig,
    pub http: HttpConfig,
    pub btc: BtcConfig,
}

#[derive(Debug, Clone)]
pub struct BtcConfig {
    pub realtime_enabled: bool,
    pub paper_enabled: bool,
    pub ml_shadow_enabled: bool,
    pub rtds_ws_url: String,
    pub binance_ws_url: String,
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
pub struct RiskConfig {
    pub taker_fee_rate: Decimal,
    pub max_simultaneous_conversions: usize,
}

#[derive(Debug, Clone)]
pub struct LiveExecutionConfig {
    pub order_submit_enabled: bool,
    pub max_order_notional_usd: Decimal,
    pub max_open_notional_usd: Decimal,
    pub max_daily_loss_usd: Decimal,
    pub max_open_positions: usize,
    pub require_exit_book: bool,
    pub require_idempotency_clean: bool,
    pub user_ws_enabled: bool,
    pub user_ws_url: String,
    pub user_ws_markets: Vec<String>,
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
        let live = LiveExecutionConfig {
            order_submit_enabled: parse_bool("POLYMARKET_LIVE_ORDER_SUBMIT_ENABLED", false),
            max_order_notional_usd: parse_decimal(
                "POLYMARKET_LIVE_MAX_ORDER_NOTIONAL_USD",
                dec!(2),
            ),
            max_open_notional_usd: parse_decimal("POLYMARKET_LIVE_MAX_OPEN_NOTIONAL_USD", dec!(20)),
            max_daily_loss_usd: parse_decimal("POLYMARKET_LIVE_MAX_DAILY_LOSS_USD", dec!(10)),
            max_open_positions: parse_usize("POLYMARKET_LIVE_MAX_OPEN_POSITIONS", 6),
            require_exit_book: parse_bool("POLYMARKET_LIVE_REQUIRE_EXIT_BOOK", true),
            require_idempotency_clean: parse_bool(
                "POLYMARKET_LIVE_REQUIRE_IDEMPOTENCY_CLEAN",
                true,
            ),
            user_ws_enabled: parse_bool("POLYMARKET_LIVE_USER_WS_ENABLED", false),
            user_ws_url: env_or(
                "POLYMARKET_LIVE_USER_WS_URL",
                "wss://ws-subscriptions-clob.polymarket.com/ws/user",
            ),
            user_ws_markets: parse_csv("POLYMARKET_LIVE_USER_WS_MARKETS"),
            clob_api_base_url: env_or(
                "POLYMARKET_LIVE_CLOB_BASE_URL",
                "https://clob.polymarket.com",
            ),
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
        if live.order_submit_enabled || live.live_auth_available() {
            live.validate_for_live()?;
        }

        let btc = BtcConfig {
            realtime_enabled: parse_bool("POLYMARKET_BTC_REALTIME_ENABLED", false),
            paper_enabled: parse_bool("POLYMARKET_BTC_PAPER_ENABLED", false),
            ml_shadow_enabled: parse_bool("POLYMARKET_BTC_ML_SHADOW_ENABLED", false),
            rtds_ws_url: env_or(
                "POLYMARKET_BTC_RTDS_WS_URL",
                "wss://ws-live-data.polymarket.com",
            ),
            binance_ws_url: env_or(
                "POLYMARKET_BTC_BINANCE_WS_URL",
                "wss://stream.binance.com:9443/ws/btcusdt@aggTrade",
            ),
        };
        if btc.paper_enabled && !btc.realtime_enabled {
            bail!("POLYMARKET_BTC_PAPER_ENABLED requires POLYMARKET_BTC_REALTIME_ENABLED");
        }
        if btc.ml_shadow_enabled && !btc.realtime_enabled {
            bail!("POLYMARKET_BTC_ML_SHADOW_ENABLED requires POLYMARKET_BTC_REALTIME_ENABLED");
        }
        if btc.realtime_enabled || btc.paper_enabled {
            let mut conflicting_flags = Vec::new();
            for (name, enabled) in [
                (
                    "POLYMARKET_LIVE_ORDER_SUBMIT_ENABLED",
                    live.order_submit_enabled,
                ),
                ("POLYMARKET_LIVE_USER_WS_ENABLED", live.user_ws_enabled),
            ] {
                if enabled {
                    conflicting_flags.push(name);
                }
            }
            if !conflicting_flags.is_empty() {
                bail!(
                    "BTC realtime/paper experiments require an isolated process; disable {}",
                    conflicting_flags.join(", ")
                );
            }
        }

        Ok(Self {
            live,
            gamma_base_url: env_or(
                "POLYMARKET_GAMMA_BASE_URL",
                "https://gamma-api.polymarket.com",
            ),
            clob_base_url: env_or("POLYMARKET_CLOB_BASE_URL", "https://clob.polymarket.com"),
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
            risk: RiskConfig {
                taker_fee_rate: parse_decimal("POLYMARKET_TAKER_FEE_RATE", dec!(0.03)),
                max_simultaneous_conversions: parse_usize(
                    "POLYMARKET_MAX_SIMULTANEOUS_CONVERSIONS",
                    3,
                ),
            },
            http: HttpConfig {
                enabled: parse_bool("POLYMARKET_HTTP_ENABLED", true),
                bind: env_or("POLYMARKET_HTTP_BIND", "0.0.0.0:8097"),
                admin_token: env_or("POLYMARKET_HTTP_ADMIN_TOKEN", "dev-polymarket-admin"),
            },
            btc,
        })
    }
}

impl LiveExecutionConfig {
    pub fn validate_for_live(&self) -> Result<()> {
        if self.max_order_notional_usd <= Decimal::ZERO {
            bail!("POLYMARKET_LIVE_MAX_ORDER_NOTIONAL_USD must be positive");
        }
        if self.max_order_notional_usd > dec!(2) {
            bail!("POLYMARKET_LIVE_MAX_ORDER_NOTIONAL_USD must be <= 2 for production canary");
        }
        if self.max_open_notional_usd <= Decimal::ZERO {
            bail!("POLYMARKET_LIVE_MAX_OPEN_NOTIONAL_USD must be positive");
        }
        if self.max_open_notional_usd > dec!(30) {
            bail!("POLYMARKET_LIVE_MAX_OPEN_NOTIONAL_USD must be <= 30 for production canary");
        }
        if self.max_open_positions == 0 || self.max_open_positions > 6 {
            bail!("POLYMARKET_LIVE_MAX_OPEN_POSITIONS must be between 1 and 6");
        }
        if self.max_daily_loss_usd <= Decimal::ZERO || self.max_daily_loss_usd > dec!(10) {
            bail!("POLYMARKET_LIVE_MAX_DAILY_LOSS_USD must be > 0 and <= 10");
        }
        if self.user_ws_enabled && self.user_ws_url.trim().is_empty() {
            bail!("POLYMARKET_LIVE_USER_WS_URL is required when user websocket is enabled");
        }
        if self.clob_api_base_url.trim().is_empty() {
            bail!("POLYMARKET_LIVE_CLOB_BASE_URL is required");
        }
        let mut missing = Vec::new();
        if self.user_ws_enabled {
            for (key, value) in [
                ("POLYMARKET_CLOB_API_KEY", &self.clob_api_key),
                ("POLYMARKET_CLOB_SECRET", &self.clob_secret),
                ("POLYMARKET_CLOB_PASSPHRASE", &self.clob_passphrase),
            ] {
                if value.is_none() {
                    missing.push(key);
                }
            }
        }
        if self.order_submit_enabled {
            for (key, value) in [
                ("POLYMARKET_CLOB_API_KEY", &self.clob_api_key),
                ("POLYMARKET_CLOB_SECRET", &self.clob_secret),
                ("POLYMARKET_CLOB_PASSPHRASE", &self.clob_passphrase),
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

fn parse_csv(key: &str) -> Vec<String> {
    env::var(key)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
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

fn parse_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_decimal(key: &str, default: Decimal) -> Decimal {
    env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
