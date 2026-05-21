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
    pub execution_mode: ExecutionMode,
    pub scan_enabled: bool,
    pub signal2_enabled: bool,
    pub signal3_enabled: bool,
    pub live_confirm: bool,
    pub live: LiveExecutionConfig,
    pub gamma_base_url: String,
    pub clob_base_url: String,
    pub clob_ws_url: String,
    pub data_api_base_url: String,
    pub scan_interval: Duration,
    pub health_interval: Duration,
    pub max_markets_per_scan: usize,
    pub postgres: PostgresConfig,
    pub risk: RiskConfig,
    pub http: HttpConfig,
    pub whale: WhaleConfig,
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
    pub daily_pnl_target_usd: Decimal,
    pub bootstrap_threshold: Decimal,
    pub taker_fee_rate: Decimal,
    pub target_size: Decimal,
    pub fill_confidence_discount_default: Decimal,
    pub fill_confidence_discount_tight: Decimal,
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

#[derive(Debug, Clone)]
pub struct WhaleConfig {
    pub backfill_enabled: bool,
    pub live_enabled: bool,
    pub copy_trade_enabled: bool,
    pub lookback_days: u32,
    pub min_trade_usd: Decimal,
    pub min_wallet_score: Decimal,
    pub min_wallet_trades: i32,
    pub min_wallet_realized_pnl_usd: Decimal,
    pub min_wallet_roi: Decimal,
    pub min_wallet_closed_positions: i32,
    pub min_copy_size_usd: Decimal,
    pub copy_size_fraction: Decimal,
    pub max_copy_size_usd: Decimal,
    pub min_liquidity_usd: Decimal,
    pub max_price_move_pct: Decimal,
    pub max_follow_lag: Duration,
    pub max_price_slippage_bps: Decimal,
    pub min_book_depth_usd: Decimal,
    pub backtest_horizon: Duration,
    pub copy_execute_enabled: bool,
    pub copy_allow_sell_entries: bool,
    pub live_poll_interval: Duration,
    pub live_page_limit: usize,
    pub live_max_pages: usize,
    pub max_pages: usize,
    pub exit_candidate_max_age: Duration,
}

impl AppConfig {
    pub fn from_env() -> Result<Self> {
        let execution_mode = match env_or("POLYMARKET_EXECUTION_MODE", "sim")
            .to_ascii_lowercase()
            .as_str()
        {
            "sim" => ExecutionMode::Sim,
            "paper" => ExecutionMode::Paper,
            "live" => ExecutionMode::Live,
            other => {
                bail!("unsupported POLYMARKET_EXECUTION_MODE={other}; expected sim, paper, or live")
            }
        };
        let live_confirm = parse_bool("POLYMARKET_LIVE_CONFIRM", false);
        if execution_mode == ExecutionMode::Live && !live_confirm {
            bail!("live mode requires POLYMARKET_LIVE_CONFIRM=true");
        }
        let live = LiveExecutionConfig {
            order_submit_enabled: parse_bool("POLYMARKET_LIVE_ORDER_SUBMIT_ENABLED", false),
            max_order_notional_usd: parse_decimal(
                "POLYMARKET_LIVE_MAX_ORDER_NOTIONAL_USD",
                dec!(2),
            ),
            max_open_notional_usd: parse_decimal("POLYMARKET_LIVE_MAX_OPEN_NOTIONAL_USD", dec!(30)),
            max_daily_loss_usd: parse_decimal("POLYMARKET_LIVE_MAX_DAILY_LOSS_USD", dec!(10)),
            max_open_positions: parse_usize("POLYMARKET_LIVE_MAX_OPEN_POSITIONS", 6),
            require_exit_book: parse_bool("POLYMARKET_LIVE_REQUIRE_EXIT_BOOK", true),
            require_idempotency_clean: parse_bool(
                "POLYMARKET_LIVE_REQUIRE_IDEMPOTENCY_CLEAN",
                true,
            ),
            user_ws_enabled: parse_bool("POLYMARKET_LIVE_USER_WS_ENABLED", true),
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
        if execution_mode == ExecutionMode::Live {
            live.validate_for_live()?;
        }

        Ok(Self {
            execution_mode,
            scan_enabled: parse_bool("POLYMARKET_SCAN_ENABLED", true),
            signal2_enabled: parse_bool("POLYMARKET_SIGNAL2_ENABLED", false),
            signal3_enabled: parse_bool("POLYMARKET_SIGNAL3_ENABLED", false),
            live_confirm,
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
            scan_interval: Duration::from_secs(parse_u64("POLYMARKET_SCAN_INTERVAL_SECS", 30)),
            health_interval: Duration::from_secs(parse_u64("POLYMARKET_HEALTH_INTERVAL_SECS", 30)),
            max_markets_per_scan: parse_usize("POLYMARKET_MAX_MARKETS_PER_SCAN", 50),
            postgres: PostgresConfig {
                host: env_or("POSTGRES_HOST", "localhost"),
                port: parse_u16("POSTGRES_PORT", 5432),
                database: env_or("POSTGRES_DB", "capitonic_timescale"),
                user: env_or("POSTGRES_USER", "postgres"),
                password: env_or("POSTGRES_PASSWORD", "postgres"),
                ssl_mode: env_or("POSTGRES_SSL_MODE", "disable"),
                ssl_root_cert: first_non_empty_env(&["POSTGRES_SSL_CA_FILE", "PGSSLROOTCERT"]),
            },
            risk: RiskConfig {
                daily_pnl_target_usd: parse_decimal("POLYMARKET_DAILY_PNL_TARGET_USD", dec!(1000)),
                bootstrap_threshold: parse_decimal("POLYMARKET_BOOTSTRAP_THRESHOLD", dec!(0.020)),
                taker_fee_rate: parse_decimal("POLYMARKET_TAKER_FEE_RATE", dec!(0.03)),
                target_size: parse_decimal("POLYMARKET_TARGET_SIZE", dec!(5)),
                fill_confidence_discount_default: dec!(0.80),
                fill_confidence_discount_tight: dec!(0.65),
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
            whale: WhaleConfig {
                backfill_enabled: parse_bool("POLYMARKET_WHALE_BACKFILL_ENABLED", true),
                live_enabled: parse_bool("POLYMARKET_WHALE_LIVE_ENABLED", false),
                copy_trade_enabled: parse_bool("POLYMARKET_COPY_TRADE_ENABLED", true),
                lookback_days: parse_u32("POLYMARKET_WHALE_BACKFILL_LOOKBACK_DAYS", 30),
                min_trade_usd: parse_decimal("POLYMARKET_WHALE_MIN_TRADE_USD", dec!(1000)),
                min_wallet_score: parse_decimal("POLYMARKET_COPY_MIN_WALLET_SCORE", dec!(70)),
                min_wallet_trades: parse_i32("POLYMARKET_COPY_MIN_WALLET_TRADES", 3),
                min_wallet_realized_pnl_usd: parse_decimal(
                    "POLYMARKET_WHALE_MIN_REALIZED_PNL_USD",
                    dec!(100),
                ),
                min_wallet_roi: parse_decimal("POLYMARKET_WHALE_MIN_ROI", dec!(0.05)),
                min_wallet_closed_positions: parse_i32("POLYMARKET_WHALE_MIN_CLOSED_POSITIONS", 3),
                min_copy_size_usd: parse_decimal("POLYMARKET_COPY_MIN_SIZE_USD", dec!(2)),
                copy_size_fraction: parse_decimal("POLYMARKET_COPY_SIZE_FRACTION", dec!(0.10)),
                max_copy_size_usd: parse_decimal("POLYMARKET_COPY_MAX_SIZE_USD", dec!(2)),
                min_liquidity_usd: parse_decimal("POLYMARKET_COPY_MIN_LIQUIDITY_USD", dec!(1000)),
                max_price_move_pct: parse_decimal("POLYMARKET_COPY_MAX_PRICE_MOVE_PCT", dec!(0.05)),
                max_follow_lag: Duration::from_secs(parse_u64(
                    "POLYMARKET_COPY_MAX_FOLLOW_LAG_SECS",
                    300,
                )),
                max_price_slippage_bps: parse_decimal(
                    "POLYMARKET_COPY_MAX_PRICE_SLIPPAGE_BPS",
                    dec!(150),
                ),
                min_book_depth_usd: parse_decimal("POLYMARKET_COPY_MIN_BOOK_DEPTH_USD", dec!(25)),
                backtest_horizon: Duration::from_secs(parse_u64(
                    "POLYMARKET_COPY_BACKTEST_HORIZON_SECS",
                    3600,
                )),
                copy_execute_enabled: parse_bool("POLYMARKET_COPY_EXECUTE_ENABLED", true),
                copy_allow_sell_entries: parse_bool("POLYMARKET_COPY_ALLOW_SELL_ENTRIES", false),
                live_poll_interval: Duration::from_secs(parse_u64(
                    "POLYMARKET_WHALE_LIVE_POLL_INTERVAL_SECS",
                    15,
                )),
                live_page_limit: parse_usize("POLYMARKET_WHALE_LIVE_PAGE_LIMIT", 100),
                live_max_pages: parse_usize("POLYMARKET_WHALE_LIVE_MAX_PAGES", 1),
                max_pages: parse_usize("POLYMARKET_WHALE_BACKFILL_MAX_PAGES", 10),
                exit_candidate_max_age: Duration::from_secs(parse_u64(
                    "POLYMARKET_EXIT_CANDIDATE_MAX_AGE_SECONDS",
                    900,
                )),
            },
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

fn parse_u32(key: &str, default: u32) -> u32 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_i32(key: &str, default: i32) -> i32 {
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
