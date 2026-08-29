//! Immutable observations of Polymarket BTC Up/Down five-minute contracts.
//!
//! Gamma responses contain high-frequency quote, volume, and liquidity fields. Those fields are
//! deliberately excluded from this strategy's factual projection so a five-second discovery poll
//! cannot turn transient trading state into unbounded contract revisions.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use futures_util::{stream, StreamExt};
use reqwest::{Client, Response, StatusCode, Url};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use sqlx::{FromRow, PgPool, Postgres, QueryBuilder, Transaction};
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    domain::{
        CaptureArtifact, IngesterProfile, IngesterStrategy, IngesterStrategyKey, StrategyError,
        StrategyErrorKind,
    },
    persistence::{
        ArtifactBatch, ArtifactRepository, GapRepository, NewCaptureArtifact, NewDataGap,
        ProfileRepository, StrategyDegradation, StrategyProgress,
    },
    runtime::{StrategyFactory, StrategyFactoryError},
};

pub const STRATEGY_KEY: IngesterStrategyKey =
    IngesterStrategyKey::PolymarketBtcFiveMinuteMarketContracts;
pub const CONFIG_SCHEMA_VERSION: i32 = 1;
pub const CHECKPOINT_SCHEMA_VERSION: i32 = 1;

pub(super) const INTERVAL_SECONDS: i64 = 300;
pub(super) const SLUG_PREFIX: &str = "btc-updown-5m-";
pub(super) const SERIES_SLUG: &str = "btc-up-or-down-5m";
const SOURCE: &str = "polymarket_gamma_rest";
const DEFAULT_GAMMA_BASE_URL: &str = "https://gamma-api.polymarket.com";
const MAX_RESPONSE_BYTES: usize = 512 * 1_024;
const MAX_WINDOWS_PER_CYCLE: i64 = 64;
const MAX_PARALLEL_REQUESTS: usize = 4;
const MAX_GAP_REPAIRS_PER_CYCLE: i64 = 8;
const GAP_RETRY_DELAY_SECONDS: i64 = 30;
const MAX_ABSENT_REPAIR_ATTEMPTS: i32 = 20;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct PolymarketBtcFiveMinuteMarketContractsConfig {
    pub gamma_base_url: String,
    pub poll_interval_seconds: u64,
    pub startup_lookback_windows: u16,
    pub lookahead_windows: u8,
    pub overlap_windows: u8,
    pub artifact_window_seconds: i64,
    pub request_timeout_seconds: u64,
}

impl Default for PolymarketBtcFiveMinuteMarketContractsConfig {
    fn default() -> Self {
        Self {
            gamma_base_url: DEFAULT_GAMMA_BASE_URL.to_owned(),
            poll_interval_seconds: 5,
            startup_lookback_windows: 12,
            lookahead_windows: 1,
            overlap_windows: 2,
            artifact_window_seconds: 3_600,
            request_timeout_seconds: 10,
        }
    }
}

impl PolymarketBtcFiveMinuteMarketContractsConfig {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let config = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            invalid_config(format!(
                "invalid Polymarket BTC five-minute market-contract config: {error}"
            ))
        })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), StrategyFactoryError> {
        validate_exact_origin(
            &self.gamma_base_url,
            DEFAULT_GAMMA_BASE_URL,
            "gamma_base_url",
        )?;
        if !(1..=300).contains(&self.poll_interval_seconds) {
            return Err(invalid_config(
                "poll_interval_seconds must be between 1 and 300",
            ));
        }
        if !(1..=288).contains(&self.startup_lookback_windows) {
            return Err(invalid_config(
                "startup_lookback_windows must be between 1 and 288",
            ));
        }
        if self.lookahead_windows > 2 {
            return Err(invalid_config("lookahead_windows must not exceed 2"));
        }
        if self.overlap_windows == 0 || self.overlap_windows > 12 {
            return Err(invalid_config("overlap_windows must be between 1 and 12"));
        }
        if self.overlap_windows < self.lookahead_windows {
            return Err(invalid_config(
                "overlap_windows must be greater than or equal to lookahead_windows",
            ));
        }
        if self.artifact_window_seconds < INTERVAL_SECONDS
            || self.artifact_window_seconds > 86_400
            || self.artifact_window_seconds % INTERVAL_SECONDS != 0
        {
            return Err(invalid_config(
                "artifact_window_seconds must be a five-minute multiple between 300 and 86400",
            ));
        }
        if !(1..=30).contains(&self.request_timeout_seconds) {
            return Err(invalid_config(
                "request_timeout_seconds must be between 1 and 30",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
struct ContractCheckpoint {
    last_window_start: Option<DateTime<Utc>>,
}

impl ContractCheckpoint {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let checkpoint = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "invalid Polymarket market-contract checkpoint: {error}"
            ))
        })?;
        if checkpoint
            .last_window_start
            .is_some_and(|timestamp| !is_aligned_window(timestamp))
        {
            return Err(StrategyFactoryError::Construction(
                "Polymarket market-contract checkpoint must be a nonnegative aligned UTC five-minute window"
                    .to_owned(),
            ));
        }
        Ok(checkpoint)
    }

    fn value(last_window_start: Option<DateTime<Utc>>) -> Value {
        json!({"last_window_start": last_window_start})
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BtcOutcome {
    Up,
    Down,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct MarketContract {
    pub(super) event_id: String,
    pub(super) event_slug: String,
    pub(super) series_slug: String,
    pub(super) market_id: String,
    pub(super) condition_id: String,
    pub(super) window_start: DateTime<Utc>,
    pub(super) window_end: DateTime<Utc>,
    pub(super) up_token_id: String,
    pub(super) down_token_id: String,
    pub(super) tick_size: Decimal,
    pub(super) minimum_order_size: Option<Decimal>,
    pub(super) resolution_source: String,
    pub(super) active: bool,
    pub(super) closed: bool,
    pub(super) accepting_orders: bool,
    pub(super) fees_enabled: bool,
    pub(super) fee_schedule: Value,
    pub(super) received_at: DateTime<Utc>,
    pub(super) source_payload: Value,
    pub(super) revision_sha256: String,
    pub(super) payload_sha256: String,
}

impl MarketContract {
    fn immutable_identity_eq(&self, stored: &StoredContract) -> bool {
        self.event_id == stored.event_id
            && self.event_slug == stored.event_slug
            && self.series_slug == stored.series_slug
            && self.market_id == stored.market_id
            && self.condition_id == stored.condition_id
            && self.window_start == stored.window_start
            && self.window_end == stored.window_end
            && self.up_token_id == stored.up_token_id
            && self.down_token_id == stored.down_token_id
            && self.resolution_source == stored.resolution_source
    }

    fn factual_eq(&self, stored: &StoredContract) -> bool {
        self.immutable_identity_eq(stored)
            && self.tick_size == stored.tick_size
            && self.minimum_order_size == stored.minimum_order_size
            && self.active == stored.active
            && self.closed == stored.closed
            && self.accepting_orders == stored.accepting_orders
            && self.fees_enabled == stored.fees_enabled
            && self.fee_schedule == stored.fee_schedule
            && self.source_payload == stored.source_payload
            && self.revision_sha256 == stored.revision_sha256
            && self.payload_sha256 == stored.payload_sha256
    }
}

#[derive(Debug, Clone, FromRow)]
struct StoredContract {
    event_id: String,
    event_slug: String,
    series_slug: String,
    market_id: String,
    condition_id: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    up_token_id: String,
    down_token_id: String,
    tick_size: Decimal,
    minimum_order_size: Option<Decimal>,
    resolution_source: String,
    active: bool,
    closed: bool,
    accepting_orders: bool,
    fees_enabled: bool,
    fee_schedule: Value,
    source_payload: Value,
    revision_sha256: String,
    payload_sha256: String,
}

#[derive(Debug, FromRow)]
struct ArtifactChecksumRow {
    market_id: String,
    revision_sha256: String,
    payload_sha256: String,
    window_start: DateTime<Utc>,
}

#[derive(Debug)]
enum WindowFetch {
    Found(Box<MarketContract>),
    Missing {
        window_start: DateTime<Utc>,
        observed_at: DateTime<Utc>,
    },
}

#[derive(Debug)]
struct DiscoveryPlan {
    windows: Vec<DateTime<Utc>>,
    scan_windows: Vec<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy)]
struct DueContractGap {
    gap_id: Uuid,
    window_start: DateTime<Utc>,
}

struct DueGapFetchOutcomes {
    transient: Vec<(Uuid, DateTime<Utc>)>,
    absent: Vec<(Uuid, DateTime<Utc>)>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PolymarketBtcFiveMinuteMarketContractsFactory;

impl StrategyFactory for PolymarketBtcFiveMinuteMarketContractsFactory {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    fn config_schema_version(&self) -> i32 {
        CONFIG_SCHEMA_VERSION
    }

    fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError> {
        PolymarketBtcFiveMinuteMarketContractsConfig::from_value(config).map(|_| ())
    }

    fn build(
        &self,
        profile: &IngesterProfile,
        pool: PgPool,
    ) -> Result<Box<dyn IngesterStrategy>, StrategyFactoryError> {
        if profile.strategy_key != STRATEGY_KEY {
            return Err(StrategyFactoryError::Construction(format!(
                "received profile for {}",
                profile.strategy_key
            )));
        }
        if profile.config_schema_version != CONFIG_SCHEMA_VERSION {
            return Err(StrategyFactoryError::Construction(format!(
                "Polymarket market-contract config schema must be {CONFIG_SCHEMA_VERSION}"
            )));
        }
        if profile.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(StrategyFactoryError::Construction(format!(
                "Polymarket market-contract checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}"
            )));
        }
        if profile.desired_generation <= 0 {
            return Err(StrategyFactoryError::Construction(
                "profile generation must be positive".to_owned(),
            ));
        }
        let config = PolymarketBtcFiveMinuteMarketContractsConfig::from_value(&profile.config)?;
        let config_snapshot = serde_json::to_value(&config).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "failed to encode effective Polymarket market-contract config: {error}"
            ))
        })?;
        let initial_checkpoint = ContractCheckpoint::from_value(&profile.checkpoint)?;
        let lease_owner = profile.lease_owner.clone().ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease owner is missing".to_owned())
        })?;
        let lease_token = profile.lease_token.ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease token is missing".to_owned())
        })?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(config.request_timeout_seconds.min(10)))
            .timeout(Duration::from_secs(config.request_timeout_seconds))
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("capitonic-market-data-ingester/0.1")
            .build()
            .map_err(|error| {
                StrategyFactoryError::Construction(format!(
                    "failed to build Polymarket Gamma HTTP client: {error}"
                ))
            })?;

        Ok(Box::new(PolymarketBtcFiveMinuteMarketContractsStrategy {
            config,
            config_snapshot,
            profile_generation: profile.desired_generation,
            lease_owner: Arc::<str>::from(lease_owner),
            lease_token,
            initial_checkpoint,
            client,
            pool,
        }))
    }
}

struct PolymarketBtcFiveMinuteMarketContractsStrategy {
    config: PolymarketBtcFiveMinuteMarketContractsConfig,
    config_snapshot: Value,
    profile_generation: i64,
    lease_owner: Arc<str>,
    lease_token: Uuid,
    initial_checkpoint: ContractCheckpoint,
    client: Client,
    pool: PgPool,
}

struct ContractRunState {
    last_window_start: Option<DateTime<Utc>>,
    artifact: Option<CaptureArtifact>,
}

#[async_trait]
impl IngesterStrategy for PolymarketBtcFiveMinuteMarketContractsStrategy {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    async fn run(&self, shutdown: CancellationToken) -> Result<(), StrategyError> {
        let recovered_frontier = if let Some(checkpoint) = self.initial_checkpoint.last_window_start
        {
            if !self.checkpoint_has_durable_evidence(checkpoint).await? {
                return Err(integrity_error(
                    "polymarket_contract_checkpoint_without_evidence",
                    format!("market-contract checkpoint {checkpoint} has no durable fact or gap evidence"),
                ));
            }
            Some(checkpoint)
        } else {
            self.recover_initial_scan_frontier(Utc::now()).await?
        };
        let mut state = ContractRunState {
            last_window_start: recovered_frontier,
            artifact: None,
        };
        let mut ticker =
            tokio::time::interval(Duration::from_secs(self.config.poll_interval_seconds));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {
                    self.seal_artifact(&mut state, true).await?;
                    return Ok(());
                }
                _ = ticker.tick() => {
                    match self.capture_cycle(&mut state, &shutdown).await {
                        Ok(()) => {}
                        Err(error) if error.kind == StrategyErrorKind::Shutdown => {
                            self.seal_artifact(&mut state, true).await?;
                            return Ok(());
                        }
                        Err(error) if error.kind == StrategyErrorKind::LeaseLost => {
                            return self.finish_owned_drain(&mut state).await;
                        }
                        Err(error) if matches!(
                            error.kind,
                            StrategyErrorKind::TransientSource | StrategyErrorKind::TransientDatabase
                        ) => {
                            let marked = ProfileRepository::new(self.pool.clone())
                                .mark_degraded(
                                    STRATEGY_KEY,
                                    &self.lease_owner,
                                    self.lease_token,
                                    self.profile_generation,
                                    &StrategyDegradation {
                                        reason_code: error.code.to_owned(),
                                        reason_message: error.to_string(),
                                    },
                                )
                                .await
                                .map_err(|database| database_error(
                                    "polymarket_contract_degraded_state_failed",
                                    database,
                                ))?;
                            if !marked {
                                return self.finish_owned_drain(&mut state).await;
                            }
                            warn!(
                                strategy = %STRATEGY_KEY,
                                error_code = error.code,
                                error = %error,
                                "Polymarket market-contract poll failed and will retry"
                            );
                        }
                        Err(error) => {
                            if let Err(seal_error) = self.seal_artifact(&mut state, false).await {
                                warn!(
                                    strategy = %STRATEGY_KEY,
                                    error = %seal_error,
                                    "failed to seal Polymarket market-contract artifact after terminal failure"
                                );
                            }
                            return Err(error);
                        }
                    }
                }
            }
        }
    }
}

impl PolymarketBtcFiveMinuteMarketContractsStrategy {
    async fn checkpoint_has_durable_evidence(
        &self,
        checkpoint: DateTime<Utc>,
    ) -> Result<bool, StrategyError> {
        sqlx::query_scalar::<_, bool>(
            r#"
            SELECT EXISTS (
              SELECT 1
              FROM market_data.polymarket_btc_five_minute_contracts
              WHERE window_start = $1
              LIMIT 1
            ) OR EXISTS (
              SELECT 1
              FROM ingester.data_gaps
              WHERE strategy_key = 'polymarket_btc_five_minute_market_contracts'
                AND reason_code = 'polymarket_gamma_contract_window_unavailable'
                AND source_time_start = $1
              LIMIT 1
            )
            "#,
        )
        .bind(checkpoint)
        .fetch_one(&self.pool)
        .await
        .map_err(|error| database_error("polymarket_contract_checkpoint_read_failed", error))
    }

    async fn recover_initial_scan_frontier(
        &self,
        now: DateTime<Utc>,
    ) -> Result<Option<DateTime<Utc>>, StrategyError> {
        let current = aligned_window_start(now);
        let end = current - chrono::Duration::seconds(INTERVAL_SECONDS);
        let start = current
            .checked_sub_signed(chrono::Duration::seconds(
                i64::from(self.config.startup_lookback_windows).saturating_mul(INTERVAL_SECONDS),
            ))
            .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().expect("Unix epoch exists"));
        if start > end {
            return Ok(None);
        }
        let rows = sqlx::query_scalar::<_, DateTime<Utc>>(
            r#"
            SELECT window_start
            FROM market_data.polymarket_btc_five_minute_contracts
            WHERE window_start BETWEEN $1 AND $2
            UNION
            SELECT source_time_start AS window_start
            FROM ingester.data_gaps
            WHERE strategy_key = 'polymarket_btc_five_minute_market_contracts'
              AND reason_code = 'polymarket_gamma_contract_window_unavailable'
              AND source_time_start BETWEEN $1 AND $2
            ORDER BY window_start
            "#,
        )
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| database_error("polymarket_contract_frontier_recovery_failed", error))?
        .into_iter()
        .collect::<BTreeSet<_>>();
        Ok(contiguous_evidence_frontier(start, end, &rows))
    }

    async fn capture_cycle(
        &self,
        state: &mut ContractRunState,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        let observed_at = Utc::now();
        let plan = discovery_window_plan(
            observed_at,
            state.last_window_start,
            self.config.startup_lookback_windows,
            self.config.lookahead_windows,
            self.config.overlap_windows,
        )?;
        let scan_windows = plan.scan_windows.clone();
        let due_gaps = self.select_due_gap_windows().await?;
        let due_by_window = due_gaps
            .iter()
            .map(|gap| (gap.window_start, gap.gap_id))
            .collect::<BTreeMap<_, _>>();
        let windows = plan
            .windows
            .into_iter()
            .chain(due_gaps.iter().map(|gap| gap.window_start))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mut fetches = stream::iter(windows.into_iter().map(|window_start| async move {
            (
                window_start,
                self.fetch_window(window_start, shutdown).await,
            )
        }))
        .buffer_unordered(MAX_PARALLEL_REQUESTS)
        .collect::<Vec<_>>()
        .await;
        fetches.sort_by_key(|(window_start, _)| *window_start);
        let mut contracts = Vec::new();
        let mut missing = Vec::new();
        let mut failed = Vec::new();
        let mut transient_gaps = Vec::new();
        let mut absent_gaps = Vec::new();
        let mut succeeded = BTreeSet::new();
        let mut first_error = None;
        for (window_start, fetch) in fetches {
            let fetch = match fetch {
                Ok(fetch) => {
                    succeeded.insert(window_start);
                    fetch
                }
                Err(error) => {
                    if error.kind == StrategyErrorKind::Shutdown {
                        return Err(error);
                    }
                    if window_start + chrono::Duration::seconds(INTERVAL_SECONDS) <= observed_at {
                        succeeded.insert(window_start);
                        failed.push((window_start, error.code, error.to_string()));
                    }
                    if let Some(gap_id) = due_by_window.get(&window_start) {
                        transient_gaps.push((*gap_id, window_start));
                    }
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                    continue;
                }
            };
            match fetch {
                WindowFetch::Found(contract) => contracts.push(*contract),
                WindowFetch::Missing {
                    window_start,
                    observed_at,
                } if window_start + chrono::Duration::seconds(INTERVAL_SECONDS) <= observed_at => {
                    missing.push(window_start);
                    if let Some(gap_id) = due_by_window.get(&window_start) {
                        absent_gaps.push((*gap_id, window_start));
                    }
                }
                WindowFetch::Missing { .. } => {}
            }
        }
        let mut next_scan_frontier = state.last_window_start;
        for window in scan_windows {
            if !succeeded.contains(&window) {
                break;
            }
            next_scan_frontier =
                Some(next_scan_frontier.map_or(window, |current| current.max(window)));
        }
        let recovered = contracts
            .iter()
            .filter_map(|contract| {
                due_by_window
                    .get(&contract.window_start)
                    .copied()
                    .map(|gap_id| (gap_id, contract.clone()))
            })
            .collect::<Vec<_>>();
        let normal_contracts = contracts
            .iter()
            .filter(|contract| !due_by_window.contains_key(&contract.window_start))
            .cloned()
            .collect::<Vec<_>>();
        let due_gap_outcomes = DueGapFetchOutcomes {
            transient: transient_gaps,
            absent: absent_gaps,
        };
        self.persist_cycle(
            state,
            &normal_contracts,
            &missing,
            &failed,
            &due_gap_outcomes,
            next_scan_frontier,
        )
        .await?;
        if !recovered.is_empty() && state.artifact.is_some() {
            self.seal_artifact(state, false).await?;
        }
        if !recovered.is_empty() {
            self.persist_recovered_gaps(state, &recovered, next_scan_frontier)
                .await?;
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(())
    }

    async fn select_due_gap_windows(&self) -> Result<Vec<DueContractGap>, StrategyError> {
        let mut transaction =
            self.pool.begin().await.map_err(|error| {
                database_error("polymarket_contract_gap_transaction_failed", error)
            })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let rows = sqlx::query_as::<_, (Uuid, Option<String>)>(
            r#"
            SELECT gap_id, start_cursor
            FROM ingester.data_gaps
            WHERE strategy_key = 'polymarket_btc_five_minute_market_contracts'
              AND reason_code = 'polymarket_gamma_contract_window_unavailable'
              AND status IN ('open', 'repairing')
              AND updated_at <= now() - ($1::bigint * INTERVAL '1 second')
            ORDER BY updated_at, detected_at, gap_id
            LIMIT $2
            FOR UPDATE SKIP LOCKED
            "#,
        )
        .bind(GAP_RETRY_DELAY_SECONDS)
        .bind(MAX_GAP_REPAIRS_PER_CYCLE)
        .fetch_all(&mut *transaction)
        .await
        .map_err(|error| database_error("polymarket_contract_gap_read_failed", error))?;
        let mut windows = Vec::with_capacity(rows.len());
        for (gap_id, cursor) in rows {
            let cursor = cursor.ok_or_else(|| {
                integrity_error(
                    "polymarket_contract_gap_cursor_missing",
                    format!("contract gap {gap_id} is missing its window cursor"),
                )
            })?;
            let seconds = cursor.parse::<i64>().map_err(|error| {
                integrity_error(
                    "polymarket_contract_gap_cursor_invalid",
                    format!("contract gap {gap_id} cursor is invalid: {error}"),
                )
            })?;
            let window = Utc.timestamp_opt(seconds, 0).single().ok_or_else(|| {
                integrity_error(
                    "polymarket_contract_gap_cursor_invalid",
                    format!("contract gap {gap_id} cursor is outside the supported range"),
                )
            })?;
            if !is_aligned_window(window) {
                return Err(integrity_error(
                    "polymarket_contract_gap_cursor_invalid",
                    format!("contract gap {gap_id} cursor is not a five-minute window"),
                ));
            }
            windows.push(DueContractGap {
                gap_id,
                window_start: window,
            });
        }
        transaction.commit().await.map_err(|error| {
            database_error("polymarket_contract_gap_transaction_commit_failed", error)
        })?;
        Ok(windows)
    }

    async fn fetch_window(
        &self,
        window_start: DateTime<Utc>,
        shutdown: &CancellationToken,
    ) -> Result<WindowFetch, StrategyError> {
        let slug = slug_for_window(window_start);
        let endpoint = format!("{}/events/slug/{slug}", self.config.gamma_base_url);
        let request = self.client.get(endpoint).send();
        let response = tokio::select! {
            _ = shutdown.cancelled() => return Err(shutdown_error()),
            response = request => response,
        }
        .map_err(|error| {
            source_error(
                "polymarket_contract_request_failed",
                format!("Gamma contract request for {slug} failed: {error}"),
            )
        })?;
        let status = response.status();
        if status == StatusCode::NOT_FOUND {
            return Ok(WindowFetch::Missing {
                window_start,
                observed_at: Utc::now(),
            });
        }
        if !status.is_success() {
            return Err(source_error(
                "polymarket_contract_http_status",
                format!("Gamma contract request for {slug} returned HTTP {status}"),
            ));
        }
        let body = read_bounded_body(response, MAX_RESPONSE_BYTES, shutdown).await?;
        let received_at = Utc::now();
        let value = serde_json::from_slice::<Value>(&body).map_err(|error| {
            source_error(
                "polymarket_contract_decode_failed",
                format!("Gamma contract response for {slug} was invalid JSON: {error}"),
            )
        })?;
        parse_gamma_contract(&value, window_start, received_at)
            .map(Box::new)
            .map(WindowFetch::Found)
    }

    async fn persist_cycle(
        &self,
        state: &mut ContractRunState,
        contracts: &[MarketContract],
        missing_windows: &[DateTime<Utc>],
        failed_windows: &[(DateTime<Utc>, &'static str, String)],
        due_gap_outcomes: &DueGapFetchOutcomes,
        next_scan_frontier: Option<DateTime<Utc>>,
    ) -> Result<(), StrategyError> {
        let artifact_id =
            if let Some(received_at) = contracts.iter().map(|row| row.received_at).max() {
                Some(self.ensure_artifact(state, received_at).await?)
            } else {
                None
            };
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polymarket_contract_transaction_begin_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;

        let inserted = if let Some(artifact_id) = artifact_id {
            self.persist_contracts_in(&mut transaction, artifact_id, contracts)
                .await?
        } else {
            BTreeSet::new()
        };

        for window_start in missing_windows {
            let detection = GapRepository::new(self.pool.clone())
                .detect_in(
                    &mut transaction,
                    &NewDataGap {
                        strategy_key: STRATEGY_KEY,
                        detected_artifact_id: artifact_id,
                        gap_kind: "source_window".to_owned(),
                        reason_code: "polymarket_gamma_contract_window_unavailable".to_owned(),
                        reason_message: Some(format!(
                            "Gamma did not expose ended BTC five-minute contract {}",
                            slug_for_window(*window_start)
                        )),
                        source_time_start: Some(*window_start),
                        source_time_end: Some(
                            *window_start
                                + chrono::Duration::seconds(INTERVAL_SECONDS.saturating_sub(1)),
                        ),
                        start_cursor: Some(window_start.timestamp().to_string()),
                        end_cursor: Some(window_start.timestamp().to_string()),
                    },
                )
                .await
                .map_err(|error| database_error("polymarket_contract_gap_detect_failed", error))?;
            if detection.inserted {
                warn!(strategy = %STRATEGY_KEY, error_code = "polymarket_gamma_contract_window_unavailable", gap_id = %detection.gap.gap_id, window_start = %window_start, "new Polymarket contract-window gap detected");
            }
        }

        for (window_start, failure_code, failure_message) in failed_windows {
            let detection = GapRepository::new(self.pool.clone())
                .detect_in(
                    &mut transaction,
                    &NewDataGap {
                        strategy_key: STRATEGY_KEY,
                        detected_artifact_id: artifact_id,
                        gap_kind: "source_window".to_owned(),
                        reason_code: "polymarket_gamma_contract_window_unavailable".to_owned(),
                        reason_message: Some(format!(
                            "Gamma BTC five-minute contract {} failed with {failure_code}: {failure_message}",
                            slug_for_window(*window_start)
                        )),
                        source_time_start: Some(*window_start),
                        source_time_end: Some(
                            *window_start
                                + chrono::Duration::seconds(INTERVAL_SECONDS.saturating_sub(1)),
                        ),
                        start_cursor: Some(window_start.timestamp().to_string()),
                        end_cursor: Some(window_start.timestamp().to_string()),
                    },
                )
                .await
                .map_err(|error| database_error("polymarket_contract_gap_detect_failed", error))?;
            if detection.inserted {
                warn!(strategy = %STRATEGY_KEY, error_code = "polymarket_gamma_contract_window_unavailable", gap_id = %detection.gap.gap_id, window_start = %window_start, "new Polymarket contract-window fetch gap detected");
            }
        }

        let transient_gap_ids = due_gap_outcomes
            .transient
            .iter()
            .map(|(gap_id, _)| *gap_id)
            .collect::<BTreeSet<_>>();
        let absent_gap_ids = due_gap_outcomes
            .absent
            .iter()
            .map(|(gap_id, _)| *gap_id)
            .collect::<BTreeSet<_>>();
        if transient_gap_ids.len() != due_gap_outcomes.transient.len()
            || absent_gap_ids.len() != due_gap_outcomes.absent.len()
            || !transient_gap_ids.is_disjoint(&absent_gap_ids)
        {
            return Err(integrity_error(
                "polymarket_contract_gap_outcome_conflict",
                "a due contract gap had duplicate or conflicting fetch outcomes",
            ));
        }

        if !due_gap_outcomes.transient.is_empty() {
            let deferred = sqlx::query_as::<_, (Uuid, Option<DateTime<Utc>>)>(
                r#"
                UPDATE ingester.data_gaps
                SET updated_at = now()
                WHERE gap_id = ANY($1::uuid[])
                  AND strategy_key = 'polymarket_btc_five_minute_market_contracts'
                  AND reason_code = 'polymarket_gamma_contract_window_unavailable'
                  AND status IN ('open', 'repairing')
                RETURNING gap_id, source_time_start
                "#,
            )
            .bind(transient_gap_ids.iter().copied().collect::<Vec<_>>())
            .fetch_all(&mut *transaction)
            .await
            .map_err(|error| database_error("polymarket_contract_gap_retry_defer_failed", error))?
            .into_iter()
            .collect::<BTreeMap<_, _>>();
            for (gap_id, window_start) in &due_gap_outcomes.transient {
                if deferred.get(gap_id).copied().flatten() != Some(*window_start) {
                    return Err(integrity_error(
                        "polymarket_contract_gap_retry_defer_mismatch",
                        format!(
                            "transient contract gap {gap_id} was absent or did not match window {window_start}"
                        ),
                    ));
                }
            }
        }

        let gaps = GapRepository::new(self.pool.clone());
        for (gap_id, window_start) in &due_gap_outcomes.absent {
            let attempted = gaps
                .begin_repair_in(&mut transaction, *gap_id)
                .await
                .map_err(|error| {
                    database_error("polymarket_contract_gap_absence_attempt_failed", error)
                })?
                .ok_or_else(|| {
                    integrity_error(
                        "polymarket_contract_gap_not_retryable",
                        format!("due contract gap {gap_id} was not retryable"),
                    )
                })?;
            if attempted.source_time_start != Some(*window_start) {
                return Err(integrity_error(
                    "polymarket_contract_gap_window_mismatch",
                    format!(
                        "absent contract gap {gap_id} does not match fetched window {window_start}"
                    ),
                ));
            }
            if attempted.repair_attempts >= MAX_ABSENT_REPAIR_ATTEMPTS {
                let terminal = gaps
                    .mark_unrecoverable_in(
                        &mut transaction,
                        *gap_id,
                        "polymarket_gamma_contract_permanently_absent",
                        Some(
                            "Gamma returned a successful not-found response for the ended contract window on every bounded repair attempt",
                        ),
                    )
                    .await
                    .map_err(|error| {
                        database_error("polymarket_contract_gap_terminalize_failed", error)
                    })?;
                if terminal.is_none() {
                    return Err(integrity_error(
                        "polymarket_contract_gap_not_terminalized",
                        format!("contract gap {gap_id} was not terminalized"),
                    ));
                }
                warn!(strategy = %STRATEGY_KEY, error_code = "polymarket_gamma_contract_permanently_absent", gap_id = %gap_id, window_start = %window_start, repair_attempts = attempted.repair_attempts, "Polymarket contract-window repair became terminal");
            }
        }

        if !missing_windows.is_empty() || !failed_windows.is_empty() {
            let marked = ProfileRepository::new(self.pool.clone())
                .mark_degraded_in(
                    &mut transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                    &StrategyDegradation {
                        reason_code: "polymarket_contract_gap_repair".to_owned(),
                        reason_message: format!(
                            "tracking {} unavailable ended Gamma contract window(s)",
                            missing_windows.len().saturating_add(failed_windows.len())
                        ),
                    },
                )
                .await
                .map_err(|error| database_error("polymarket_contract_gap_health_failed", error))?;
            if !marked {
                return Err(lease_lost_error());
            }
        }

        let mut artifact_after_commit = None;
        if let Some(artifact_id) = artifact_id {
            if !inserted.is_empty() {
                let inserted_rows = contracts
                    .iter()
                    .filter(|row| {
                        inserted.contains(&(row.market_id.clone(), row.revision_sha256.clone()))
                    })
                    .collect::<Vec<_>>();
                artifact_after_commit = Some(
                    ArtifactRepository::new(self.pool.clone())
                        .record_batch_in(
                            &mut transaction,
                            artifact_id,
                            &contract_artifact_batch(&inserted_rows),
                        )
                        .await
                        .map_err(|error| {
                            database_error("polymarket_contract_artifact_progress_failed", error)
                        })?
                        .ok_or_else(|| {
                            integrity_error(
                                "polymarket_contract_artifact_not_open",
                                format!(
                                    "artifact {artifact_id} was not open during contract insert"
                                ),
                            )
                        })?,
                );
            }
        }

        let checkpoint_persisted = if !contracts.is_empty() {
            let progressed = ProfileRepository::new(self.pool.clone())
                .record_progress_in(
                    &mut transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                    &StrategyProgress {
                        verified_record_count: contracts.len() as i64,
                        checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
                        checkpoint: ContractCheckpoint::value(next_scan_frontier),
                        last_source_event_at: contracts.iter().map(|row| row.window_start).max(),
                        last_provider_available_at: None,
                        source_watermark: contracts.iter().map(|row| row.window_start).max(),
                        availability_watermark: None,
                    },
                )
                .await
                .map_err(|error| {
                    database_error("polymarket_contract_profile_progress_failed", error)
                })?;
            if !progressed {
                return Err(lease_lost_error());
            }
            true
        } else if (!missing_windows.is_empty() || !failed_windows.is_empty())
            && next_scan_frontier.is_some()
        {
            let checkpointed = sqlx::query_scalar::<_, String>(
                r#"
                UPDATE ingester.profiles
                SET checkpoint_schema_version = $5,
                    checkpoint = $6,
                    updated_at = now()
                WHERE strategy_key = $1
                  AND lease_owner = $2
                  AND lease_token = $3
                  AND lease_expires_at > now()
                  AND desired_state = 'running'
                  AND desired_generation = $4
                  AND applied_generation = $4
                RETURNING strategy_key
                "#,
            )
            .bind(STRATEGY_KEY.as_str())
            .bind(self.lease_owner.as_ref())
            .bind(self.lease_token)
            .bind(self.profile_generation)
            .bind(CHECKPOINT_SCHEMA_VERSION)
            .bind(ContractCheckpoint::value(next_scan_frontier))
            .fetch_optional(&mut *transaction)
            .await
            .map_err(|error| database_error("polymarket_contract_gap_checkpoint_failed", error))?;
            if checkpointed.is_none() {
                return Err(lease_lost_error());
            }
            true
        } else {
            false
        };

        transaction.commit().await.map_err(|error| {
            database_error("polymarket_contract_transaction_commit_failed", error)
        })?;
        if let Some(artifact) = artifact_after_commit {
            state.artifact = Some(artifact);
        }
        if checkpoint_persisted {
            state.last_window_start = next_scan_frontier;
        }
        Ok(())
    }

    async fn persist_recovered_gaps(
        &self,
        state: &mut ContractRunState,
        recovered: &[(Uuid, MarketContract)],
        next_scan_frontier: Option<DateTime<Utc>>,
    ) -> Result<(), StrategyError> {
        if recovered.is_empty() {
            return Ok(());
        }
        if state.artifact.is_some() {
            return Err(integrity_error(
                "polymarket_contract_repair_artifact_not_dedicated",
                "contract gap repair requires a dedicated capture artifact",
            ));
        }
        let recovered_gap_ids = recovered
            .iter()
            .map(|(gap_id, _)| *gap_id)
            .collect::<BTreeSet<_>>();
        if recovered_gap_ids.len() != recovered.len() {
            return Err(integrity_error(
                "polymarket_contract_gap_recovery_duplicate",
                "a contract gap appeared more than once in one recovery batch",
            ));
        }
        let received_at = recovered
            .iter()
            .map(|(_, contract)| contract.received_at)
            .max()
            .expect("a nonempty recovery has a receipt timestamp");
        let artifact_id = self.ensure_artifact(state, received_at).await?;
        let contracts = recovered
            .iter()
            .map(|(_, contract)| contract.clone())
            .collect::<Vec<_>>();
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polymarket_contract_repair_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let inserted = self
            .persist_contracts_in(&mut transaction, artifact_id, &contracts)
            .await?;
        let mut artifact = state
            .artifact
            .as_ref()
            .filter(|artifact| artifact.artifact_id == artifact_id)
            .cloned()
            .ok_or_else(|| {
                integrity_error(
                    "polymarket_contract_repair_artifact_missing",
                    format!("repair artifact {artifact_id} is not owned by the run state"),
                )
            })?;
        if !inserted.is_empty() {
            let inserted_rows = contracts
                .iter()
                .filter(|row| {
                    inserted.contains(&(row.market_id.clone(), row.revision_sha256.clone()))
                })
                .collect::<Vec<_>>();
            artifact = ArtifactRepository::new(self.pool.clone())
                .record_batch_in(
                    &mut transaction,
                    artifact_id,
                    &contract_artifact_batch(&inserted_rows),
                )
                .await
                .map_err(|error| {
                    database_error("polymarket_contract_repair_artifact_progress_failed", error)
                })?
                .ok_or_else(|| {
                    integrity_error(
                        "polymarket_contract_repair_artifact_not_open",
                        format!("repair artifact {artifact_id} was not open during fact insert"),
                    )
                })?;
        }
        let gaps = GapRepository::new(self.pool.clone());
        for (gap_id, contract) in recovered {
            let attempted = gaps
                .begin_repair_in(&mut transaction, *gap_id)
                .await
                .map_err(|error| {
                    database_error("polymarket_contract_gap_repair_attempt_failed", error)
                })?
                .ok_or_else(|| {
                    integrity_error(
                        "polymarket_contract_gap_not_retryable",
                        format!("recovered contract gap {gap_id} was not retryable"),
                    )
                })?;
            if attempted.source_time_start != Some(contract.window_start) {
                return Err(integrity_error(
                    "polymarket_contract_gap_window_mismatch",
                    format!(
                        "recovered gap {gap_id} does not match factual window {}",
                        contract.window_start
                    ),
                ));
            }
        }
        let (content_sha256, end_cursor) = self
            .artifact_checksum_in(&mut transaction, &artifact)
            .await?;
        let completed = ArtifactRepository::new(self.pool.clone())
            .complete_in(
                &mut transaction,
                artifact_id,
                &content_sha256,
                end_cursor.as_deref().or(artifact.end_cursor.as_deref()),
            )
            .await
            .map_err(|error| {
                database_error("polymarket_contract_repair_artifact_complete_failed", error)
            })?;
        if completed.is_none() {
            return Err(integrity_error(
                "polymarket_contract_repair_artifact_not_open",
                format!("repair artifact {artifact_id} was not open while completing"),
            ));
        }
        for (gap_id, contract) in recovered {
            let repaired = gaps
                .mark_repaired_in(
                    &mut transaction,
                    *gap_id,
                    artifact_id,
                    "polymarket_gamma_contract_discovered",
                    Some(
                        "Gamma contract became available and its factual projection was durably verified",
                    ),
                )
                .await
                .map_err(|error| {
                    database_error("polymarket_contract_gap_repair_complete_failed", error)
                })?;
            let repaired = repaired.ok_or_else(|| {
                integrity_error(
                    "polymarket_contract_gap_not_repairing",
                    format!("recovered contract gap {gap_id} was not repairable"),
                )
            })?;
            if repaired.source_time_start != Some(contract.window_start) {
                return Err(integrity_error(
                    "polymarket_contract_gap_window_mismatch",
                    format!(
                        "claimed gap {gap_id} does not match recovered window {}",
                        contract.window_start
                    ),
                ));
            }
        }
        let progressed = ProfileRepository::new(self.pool.clone())
            .record_progress_in(
                &mut transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &StrategyProgress {
                    verified_record_count: contracts.len() as i64,
                    checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
                    checkpoint: ContractCheckpoint::value(next_scan_frontier),
                    last_source_event_at: contracts.iter().map(|row| row.window_start).max(),
                    last_provider_available_at: None,
                    source_watermark: contracts.iter().map(|row| row.window_start).max(),
                    availability_watermark: None,
                },
            )
            .await
            .map_err(|error| {
                database_error("polymarket_contract_repair_profile_progress_failed", error)
            })?;
        if !progressed {
            return Err(lease_lost_error());
        }
        transaction.commit().await.map_err(|error| {
            database_error(
                "polymarket_contract_repair_transaction_commit_failed",
                error,
            )
        })?;
        state.artifact = None;
        state.last_window_start = next_scan_frontier;
        info!(
            strategy = %STRATEGY_KEY,
            %artifact_id,
            repaired_gap_count = recovered.len(),
            %content_sha256,
            "completed dedicated Polymarket contract gap-repair artifact"
        );
        Ok(())
    }

    async fn persist_contracts_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        contracts: &[MarketContract],
    ) -> Result<BTreeSet<(String, String)>, StrategyError> {
        if contracts.is_empty() {
            return Ok(BTreeSet::new());
        }
        let market_ids = contracts
            .iter()
            .map(|contract| contract.market_id.clone())
            .collect::<Vec<_>>();
        let event_slugs = contracts
            .iter()
            .map(|contract| contract.event_slug.clone())
            .collect::<Vec<_>>();
        let event_ids = contracts
            .iter()
            .map(|contract| contract.event_id.clone())
            .collect::<Vec<_>>();
        let condition_ids = contracts
            .iter()
            .map(|contract| contract.condition_id.clone())
            .collect::<Vec<_>>();
        let token_ids = contracts
            .iter()
            .flat_map(|contract| [contract.up_token_id.clone(), contract.down_token_id.clone()])
            .collect::<Vec<_>>();
        validate_candidate_contract_identities(contracts)?;
        let existing = self
            .load_contracts_in(
                transaction,
                &market_ids,
                &event_ids,
                &event_slugs,
                &condition_ids,
                &token_ids,
            )
            .await?;
        validate_stored_contract_identities(contracts, &existing)?;
        let existing_by_revision = unique_contract_revisions(existing)?;
        let mut missing = Vec::new();
        for contract in contracts {
            let key = (contract.market_id.clone(), contract.revision_sha256.clone());
            if let Some(stored) = existing_by_revision.get(&key) {
                if !contract.factual_eq(stored) {
                    return Err(immutable_contract_conflict(contract));
                }
            } else {
                missing.push(contract);
            }
        }
        let inserted = self
            .insert_missing_contracts_in(transaction, artifact_id, &missing)
            .await?;
        let durable = unique_contract_revisions(
            self.load_contracts_in(
                transaction,
                &market_ids,
                &event_ids,
                &event_slugs,
                &condition_ids,
                &token_ids,
            )
            .await?,
        )?;
        for contract in contracts {
            let key = (contract.market_id.clone(), contract.revision_sha256.clone());
            let stored = durable.get(&key).ok_or_else(|| {
                integrity_error(
                    "polymarket_contract_insert_missing",
                    format!(
                        "contract revision {}:{} was absent after insert",
                        contract.market_id, contract.revision_sha256
                    ),
                )
            })?;
            if !contract.factual_eq(stored) {
                return Err(immutable_contract_conflict(contract));
            }
        }
        Ok(inserted)
    }

    async fn load_contracts_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        market_ids: &[String],
        event_ids: &[String],
        event_slugs: &[String],
        condition_ids: &[String],
        token_ids: &[String],
    ) -> Result<Vec<StoredContract>, StrategyError> {
        sqlx::query_as::<_, StoredContract>(
            r#"
            SELECT event_id, event_slug, series_slug, market_id, condition_id,
                   window_start, window_end, up_token_id, down_token_id,
                   tick_size, minimum_order_size, resolution_source,
                   active, closed, accepting_orders, fees_enabled, fee_schedule,
                   source_payload,
                   revision_sha256::text AS revision_sha256,
                   payload_sha256::text AS payload_sha256
            FROM market_data.polymarket_btc_five_minute_contracts
            WHERE market_id = ANY($1::text[])
               OR event_id = ANY($2::text[])
               OR event_slug = ANY($3::text[])
               OR condition_id = ANY($4::text[])
               OR up_token_id = ANY($5::text[])
               OR down_token_id = ANY($5::text[])
            ORDER BY market_id, revision_sha256
            "#,
        )
        .bind(market_ids)
        .bind(event_ids)
        .bind(event_slugs)
        .bind(condition_ids)
        .bind(token_ids)
        .fetch_all(&mut **transaction)
        .await
        .map_err(|error| database_error("polymarket_contract_fact_read_failed", error))
    }

    async fn insert_missing_contracts_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        missing: &[&MarketContract],
    ) -> Result<BTreeSet<(String, String)>, StrategyError> {
        if missing.is_empty() {
            return Ok(BTreeSet::new());
        }
        let mut builder = QueryBuilder::<Postgres>::new(
            r#"
            INSERT INTO market_data.polymarket_btc_five_minute_contracts (
              source, event_id, event_slug, series_slug, market_id, condition_id,
              window_start, window_end, up_token_id, down_token_id,
              tick_size, minimum_order_size, resolution_source,
              active, closed, accepting_orders, fees_enabled, fee_schedule,
              received_at, source_payload, revision_sha256, payload_sha256,
              strategy_key, capture_artifact_id
            )
            "#,
        );
        builder.push_values(missing, |mut row, contract| {
            row.push_bind(SOURCE)
                .push_bind(&contract.event_id)
                .push_bind(&contract.event_slug)
                .push_bind(&contract.series_slug)
                .push_bind(&contract.market_id)
                .push_bind(&contract.condition_id)
                .push_bind(contract.window_start)
                .push_bind(contract.window_end)
                .push_bind(&contract.up_token_id)
                .push_bind(&contract.down_token_id)
                .push_bind(contract.tick_size)
                .push_bind(contract.minimum_order_size)
                .push_bind(&contract.resolution_source)
                .push_bind(contract.active)
                .push_bind(contract.closed)
                .push_bind(contract.accepting_orders)
                .push_bind(contract.fees_enabled)
                .push_bind(&contract.fee_schedule)
                .push_bind(contract.received_at)
                .push_bind(&contract.source_payload)
                .push_bind(&contract.revision_sha256)
                .push_bind(&contract.payload_sha256)
                .push_bind(STRATEGY_KEY.as_str())
                .push_bind(artifact_id);
        });
        builder.push(
            " ON CONFLICT (market_id, revision_sha256) DO NOTHING \
             RETURNING market_id, revision_sha256::text",
        );
        let inserted = builder
            .build_query_as::<(String, String)>()
            .fetch_all(&mut **transaction)
            .await
            .map_err(|error| database_error("polymarket_contract_fact_insert_failed", error))?;
        Ok(inserted.into_iter().collect())
    }

    fn artifact_window(&self, timestamp: DateTime<Utc>) -> (DateTime<Utc>, DateTime<Utc>) {
        let start_seconds = timestamp
            .timestamp()
            .div_euclid(self.config.artifact_window_seconds)
            .saturating_mul(self.config.artifact_window_seconds);
        let start = Utc
            .timestamp_opt(start_seconds, 0)
            .single()
            .expect("aligned artifact timestamp is representable");
        let end = Utc
            .timestamp_opt(
                start_seconds.saturating_add(self.config.artifact_window_seconds),
                0,
            )
            .single()
            .expect("bounded artifact timestamp is representable");
        (start, end)
    }

    async fn ensure_artifact(
        &self,
        state: &mut ContractRunState,
        received_at: DateTime<Utc>,
    ) -> Result<Uuid, StrategyError> {
        let (window_start, window_end) = self.artifact_window(received_at);
        let reusable = state.artifact.as_ref().is_some_and(|artifact| {
            artifact.profile_generation == self.profile_generation
                && artifact.config_schema_version == CONFIG_SCHEMA_VERSION
                && artifact.config_snapshot == self.config_snapshot
                && artifact.capture_window_start == window_start
                && artifact.capture_window_end == window_end
        });
        if reusable {
            return Ok(state
                .artifact
                .as_ref()
                .expect("matching artifact exists")
                .artifact_id);
        }
        if state.artifact.is_some() {
            self.seal_artifact(state, false).await?;
        }
        let repository = ArtifactRepository::new(self.pool.clone());
        if let Some(open) = repository
            .get_open(STRATEGY_KEY)
            .await
            .map_err(|error| database_error("polymarket_contract_artifact_read_failed", error))?
        {
            verify_artifact_generation(open.profile_generation, self.profile_generation)?;
            let reusable = open.profile_generation == self.profile_generation
                && open.config_schema_version == CONFIG_SCHEMA_VERSION
                && open.config_snapshot == self.config_snapshot
                && open.capture_window_start == window_start
                && open.capture_window_end == window_end;
            if open.profile_generation == self.profile_generation
                && (open.config_schema_version != CONFIG_SCHEMA_VERSION
                    || open.config_snapshot != self.config_snapshot)
            {
                return Err(integrity_error(
                    "polymarket_contract_artifact_config_conflict",
                    "open artifact has the current generation but different effective configuration",
                ));
            }
            state.artifact = Some(open);
            if reusable {
                return Ok(state
                    .artifact
                    .as_ref()
                    .expect("reusable artifact exists")
                    .artifact_id);
            }
            self.seal_artifact(state, false).await?;
        }
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polymarket_contract_artifact_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let artifact = repository
            .create_in(
                &mut transaction,
                &NewCaptureArtifact {
                    strategy_key: STRATEGY_KEY,
                    profile_generation: self.profile_generation,
                    config_schema_version: CONFIG_SCHEMA_VERSION,
                    config_snapshot: self.config_snapshot.clone(),
                    capture_window_start: window_start,
                    capture_window_end: window_end,
                    start_cursor: state
                        .last_window_start
                        .map(|timestamp| timestamp.timestamp().to_string()),
                },
            )
            .await
            .map_err(|error| database_error("polymarket_contract_artifact_create_failed", error))?;
        transaction
            .commit()
            .await
            .map_err(|error| database_error("polymarket_contract_artifact_commit_failed", error))?;
        let artifact_id = artifact.artifact_id;
        state.artifact = Some(artifact);
        info!(
            strategy = %STRATEGY_KEY,
            %artifact_id,
            %window_start,
            %window_end,
            "opened Polymarket market-contract capture artifact"
        );
        Ok(artifact_id)
    }

    async fn seal_artifact(
        &self,
        state: &mut ContractRunState,
        allow_draining_generation: bool,
    ) -> Result<(), StrategyError> {
        let Some(artifact) = state.artifact.as_ref().cloned() else {
            let mut transaction = self.pool.begin().await.map_err(|error| {
                database_error("polymarket_contract_drain_transaction_failed", error)
            })?;
            self.assert_lease_in(&mut transaction, allow_draining_generation)
                .await?;
            transaction.commit().await.map_err(|error| {
                database_error("polymarket_contract_drain_commit_failed", error)
            })?;
            return Ok(());
        };
        verify_artifact_generation(artifact.profile_generation, self.profile_generation)?;
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polymarket_contract_artifact_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, allow_draining_generation)
            .await?;
        let (content_sha256, end_cursor) = self
            .artifact_checksum_in(&mut transaction, &artifact)
            .await?;
        let completed = ArtifactRepository::new(self.pool.clone())
            .complete_in(
                &mut transaction,
                artifact.artifact_id,
                &content_sha256,
                end_cursor.as_deref().or(artifact.end_cursor.as_deref()),
            )
            .await
            .map_err(|error| {
                database_error("polymarket_contract_artifact_complete_failed", error)
            })?;
        if completed.is_none() {
            return Err(integrity_error(
                "polymarket_contract_artifact_not_open",
                format!(
                    "artifact {} was not open while sealing",
                    artifact.artifact_id
                ),
            ));
        }
        transaction
            .commit()
            .await
            .map_err(|error| database_error("polymarket_contract_artifact_commit_failed", error))?;
        state.artifact = None;
        info!(
            strategy = %STRATEGY_KEY,
            artifact_id = %artifact.artifact_id,
            record_count = artifact.record_count,
            %content_sha256,
            "sealed Polymarket market-contract capture artifact"
        );
        Ok(())
    }

    async fn artifact_checksum_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact: &CaptureArtifact,
    ) -> Result<(String, Option<String>), StrategyError> {
        let rows = sqlx::query_as::<_, ArtifactChecksumRow>(
            r#"
            SELECT market_id,
                   revision_sha256::text AS revision_sha256,
                   payload_sha256::text AS payload_sha256,
                   window_start
            FROM market_data.polymarket_btc_five_minute_contracts
            WHERE capture_artifact_id = $1 AND strategy_key = $2
            ORDER BY window_start, market_id, revision_sha256
            "#,
        )
        .bind(artifact.artifact_id)
        .bind(STRATEGY_KEY.as_str())
        .fetch_all(&mut **transaction)
        .await
        .map_err(|error| database_error("polymarket_contract_checksum_read_failed", error))?;
        if i64::try_from(rows.len()).ok() != Some(artifact.record_count) {
            return Err(integrity_error(
                "polymarket_contract_artifact_count_mismatch",
                format!(
                    "artifact {} records {} rows but owns {} contract facts",
                    artifact.artifact_id,
                    artifact.record_count,
                    rows.len()
                ),
            ));
        }
        let mut hasher = Sha256::new();
        for row in &rows {
            hash_field(&mut hasher, &row.market_id);
            hash_field(&mut hasher, &row.revision_sha256);
            hash_field(&mut hasher, &row.payload_sha256);
        }
        Ok((
            digest_hex(hasher.finalize()),
            rows.last()
                .map(|row| row.window_start.timestamp().to_string()),
        ))
    }

    async fn assert_lease_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        allow_draining_generation: bool,
    ) -> Result<(), StrategyError> {
        let profiles = ProfileRepository::new(self.pool.clone());
        let current = if allow_draining_generation {
            profiles
                .lock_owned_lease_in(
                    transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                )
                .await
        } else {
            profiles
                .lock_current_lease_in(
                    transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                )
                .await
        }
        .map_err(|error| database_error("polymarket_contract_lease_check_failed", error))?;
        if !current {
            return Err(lease_lost_error());
        }
        Ok(())
    }

    async fn finish_owned_drain(&self, state: &mut ContractRunState) -> Result<(), StrategyError> {
        self.seal_artifact(state, true).await?;
        info!(
            strategy = %STRATEGY_KEY,
            generation = self.profile_generation,
            "Polymarket market-contract strategy drained after a desired-state lease race"
        );
        Ok(())
    }
}

pub(super) fn aligned_window_start(now: DateTime<Utc>) -> DateTime<Utc> {
    let seconds = now
        .timestamp()
        .div_euclid(INTERVAL_SECONDS)
        .saturating_mul(INTERVAL_SECONDS);
    Utc.timestamp_opt(seconds, 0)
        .single()
        .expect("an aligned UTC timestamp is representable")
}

pub(super) fn slug_for_window(window_start: DateTime<Utc>) -> String {
    format!("{SLUG_PREFIX}{}", window_start.timestamp())
}

pub(super) fn window_start_from_slug(slug: &str) -> Result<DateTime<Utc>, StrategyError> {
    let raw = slug.strip_prefix(SLUG_PREFIX).ok_or_else(|| {
        source_error(
            "polymarket_contract_invalid_slug",
            "Gamma event slug is not a BTC Up/Down five-minute slug",
        )
    })?;
    let seconds = raw.parse::<i64>().map_err(|error| {
        source_error(
            "polymarket_contract_invalid_slug",
            format!("Gamma event slug has an invalid epoch suffix: {error}"),
        )
    })?;
    let timestamp = Utc.timestamp_opt(seconds, 0).single().ok_or_else(|| {
        source_error(
            "polymarket_contract_invalid_slug",
            "Gamma event slug epoch is outside the supported range",
        )
    })?;
    if !is_aligned_window(timestamp) {
        return Err(source_error(
            "polymarket_contract_invalid_slug",
            "Gamma event slug epoch is not aligned to a five-minute boundary",
        ));
    }
    Ok(timestamp)
}

fn contiguous_evidence_frontier(
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    evidence: &BTreeSet<DateTime<Utc>>,
) -> Option<DateTime<Utc>> {
    let mut frontier = None;
    let mut window = start;
    while window <= end && evidence.contains(&window) {
        frontier = Some(window);
        let Some(next) = window.checked_add_signed(chrono::Duration::seconds(INTERVAL_SECONDS))
        else {
            break;
        };
        window = next;
    }
    frontier
}

fn discovery_window_plan(
    now: DateTime<Utc>,
    last_window_start: Option<DateTime<Utc>>,
    startup_lookback_windows: u16,
    lookahead_windows: u8,
    overlap_windows: u8,
) -> Result<DiscoveryPlan, StrategyError> {
    let current = aligned_window_start(now);
    let latest = current
        .checked_add_signed(chrono::Duration::seconds(
            i64::from(lookahead_windows).saturating_mul(INTERVAL_SECONDS),
        ))
        .ok_or_else(|| {
            integrity_error(
                "polymarket_contract_window_overflow",
                "lookahead overflowed",
            )
        })?;
    let scan_latest = current
        .checked_sub_signed(chrono::Duration::seconds(INTERVAL_SECONDS))
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().expect("Unix epoch exists"));
    let start = if let Some(last) = last_window_start {
        if !is_aligned_window(last) {
            return Err(integrity_error(
                "polymarket_contract_cursor_unaligned",
                "durable market-contract cursor is not an aligned five-minute window",
            ));
        }
        if last > scan_latest {
            return Err(integrity_error(
                "polymarket_contract_cursor_in_future",
                format!("durable scan cursor {last} is newer than last ended window {scan_latest}"),
            ));
        }
        last.checked_sub_signed(chrono::Duration::seconds(
            i64::from(overlap_windows).saturating_mul(INTERVAL_SECONDS),
        ))
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().expect("Unix epoch exists"))
    } else {
        current
            .checked_sub_signed(chrono::Duration::seconds(
                i64::from(startup_lookback_windows).saturating_mul(INTERVAL_SECONDS),
            ))
            .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().expect("Unix epoch exists"))
    };
    let available = (scan_latest - start)
        .num_seconds()
        .div_euclid(INTERVAL_SECONDS)
        .saturating_add(1);
    let count = available.min(MAX_WINDOWS_PER_CYCLE);
    let mut scan_windows = Vec::with_capacity(usize::try_from(count).unwrap_or_default());
    for index in 0..count {
        scan_windows.push(
            start
                .checked_add_signed(chrono::Duration::seconds(
                    index.saturating_mul(INTERVAL_SECONDS),
                ))
                .ok_or_else(|| {
                    integrity_error(
                        "polymarket_contract_window_overflow",
                        "discovery window overflowed",
                    )
                })?,
        );
    }
    let tail_start = current
        .checked_sub_signed(chrono::Duration::seconds(INTERVAL_SECONDS))
        .unwrap_or_else(|| Utc.timestamp_opt(0, 0).single().expect("Unix epoch exists"));
    let tail_count = (latest - tail_start)
        .num_seconds()
        .div_euclid(INTERVAL_SECONDS)
        .saturating_add(1);
    let mut windows = scan_windows.iter().copied().collect::<BTreeSet<_>>();
    for index in 0..tail_count {
        let window = tail_start
            .checked_add_signed(chrono::Duration::seconds(
                index.saturating_mul(INTERVAL_SECONDS),
            ))
            .ok_or_else(|| {
                integrity_error(
                    "polymarket_contract_window_overflow",
                    "realtime tail window overflowed",
                )
            })?;
        windows.insert(window);
    }
    Ok(DiscoveryPlan {
        windows: windows.into_iter().collect(),
        scan_windows,
    })
}

pub(super) fn parse_gamma_contract(
    value: &Value,
    expected_window_start: DateTime<Utc>,
    received_at: DateTime<Utc>,
) -> Result<MarketContract, StrategyError> {
    let event = value.as_object().ok_or_else(|| {
        source_error(
            "polymarket_contract_invalid_event",
            "Gamma event response must be an object",
        )
    })?;
    let event_slug = required_string(event, &["slug"])?;
    let slug_window = window_start_from_slug(&event_slug)?;
    if slug_window != expected_window_start {
        return Err(source_error(
            "polymarket_contract_window_mismatch",
            format!(
                "Gamma event slug window {slug_window} does not match requested window {expected_window_start}"
            ),
        ));
    }
    let series_slug = string_field(event, &["seriesSlug", "series_slug"])
        .or_else(|| series_slug_from_relation(event))
        .ok_or_else(|| {
            source_error(
                "polymarket_contract_missing_series",
                "Gamma event is missing its series slug",
            )
        })?;
    if series_slug != SERIES_SLUG {
        return Err(source_error(
            "polymarket_contract_series_mismatch",
            format!("Gamma event belongs to unexpected series {series_slug}"),
        ));
    }
    let window_start =
        datetime_field(event, &["eventStartTime", "startTime"]).ok_or_else(|| {
            source_error(
                "polymarket_contract_missing_window",
                "Gamma event is missing eventStartTime/startTime",
            )
        })?;
    if window_start != slug_window {
        return Err(source_error(
            "polymarket_contract_window_mismatch",
            "Gamma event start time does not match its slug epoch",
        ));
    }
    let markets = event
        .get("markets")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            source_error(
                "polymarket_contract_missing_market",
                "Gamma event is missing its markets array",
            )
        })?;
    if markets.len() != 1 {
        return Err(source_error(
            "polymarket_contract_ambiguous_market",
            "BTC Up/Down five-minute event must contain exactly one market",
        ));
    }
    let market = markets[0].as_object().ok_or_else(|| {
        source_error(
            "polymarket_contract_invalid_market",
            "Gamma event market must be an object",
        )
    })?;
    let market_start = datetime_field(market, &["eventStartTime"]);
    if market_start.is_some_and(|timestamp| timestamp != window_start) {
        return Err(source_error(
            "polymarket_contract_window_mismatch",
            "Gamma market eventStartTime does not match the event window",
        ));
    }
    let window_end = datetime_field(market, &["endDate", "endDateIso"])
        .or_else(|| datetime_field(event, &["endDate"]))
        .ok_or_else(|| {
            source_error(
                "polymarket_contract_missing_window",
                "Gamma event is missing its market end date",
            )
        })?;
    if window_end - window_start != chrono::Duration::seconds(INTERVAL_SECONDS) {
        return Err(source_error(
            "polymarket_contract_window_mismatch",
            "Gamma BTC Up/Down market does not have an exact five-minute window",
        ));
    }
    let resolution_source = string_field(market, &["resolutionSource", "resolution_source"])
        .or_else(|| string_field(event, &["resolutionSource", "resolution_source"]))
        .ok_or_else(|| {
            source_error(
                "polymarket_contract_missing_resolution_source",
                "Gamma event is missing its resolution source",
            )
        })?;
    if !is_chainlink_btc_usd_source(&resolution_source) {
        return Err(source_error(
            "polymarket_contract_resolution_source_mismatch",
            "Gamma BTC Up/Down resolution source is not Chainlink BTC/USD",
        ));
    }
    if resolution_source.is_empty() || resolution_source.len() > 2_048 {
        return Err(source_error(
            "polymarket_contract_resolution_source_too_large",
            "Gamma resolution source must contain between 1 and 2048 bytes",
        ));
    }
    let outcomes = string_array_field(market, &["outcomes"]).ok_or_else(|| {
        source_error(
            "polymarket_contract_missing_outcomes",
            "Gamma market is missing outcomes",
        )
    })?;
    let token_ids = string_array_field(
        market,
        &["clobTokenIds", "clob_token_ids", "tokenIds", "token_ids"],
    )
    .ok_or_else(|| {
        source_error(
            "polymarket_contract_missing_tokens",
            "Gamma market is missing CLOB token IDs",
        )
    })?;
    if outcomes.len() != 2 || token_ids.len() != 2 {
        return Err(source_error(
            "polymarket_contract_ambiguous_outcomes",
            "Gamma BTC Up/Down market must contain exactly two outcomes and token IDs",
        ));
    }
    let mut up_token_id = None;
    let mut down_token_id = None;
    for (outcome, token_id) in outcomes.iter().zip(&token_ids) {
        validate_token_id(token_id)?;
        match normalize_outcome(outcome)? {
            BtcOutcome::Up if up_token_id.replace(token_id.clone()).is_some() => {
                return Err(source_error(
                    "polymarket_contract_ambiguous_outcomes",
                    "Gamma market contains duplicate Up outcomes",
                ));
            }
            BtcOutcome::Down if down_token_id.replace(token_id.clone()).is_some() => {
                return Err(source_error(
                    "polymarket_contract_ambiguous_outcomes",
                    "Gamma market contains duplicate Down outcomes",
                ));
            }
            _ => {}
        }
    }
    let up_token_id = up_token_id.ok_or_else(|| {
        source_error(
            "polymarket_contract_missing_outcomes",
            "Gamma market is missing its Up token",
        )
    })?;
    let down_token_id = down_token_id.ok_or_else(|| {
        source_error(
            "polymarket_contract_missing_outcomes",
            "Gamma market is missing its Down token",
        )
    })?;
    if up_token_id == down_token_id {
        return Err(source_error(
            "polymarket_contract_duplicate_tokens",
            "Gamma market token IDs must be distinct",
        ));
    }
    let event_id = required_string(event, &["id"])?;
    let market_id = required_string(market, &["id"])?;
    let condition_id = required_string(market, &["conditionId", "condition_id"])?;
    validate_identifier("event ID", &event_id, 256)?;
    validate_identifier("market ID", &market_id, 256)?;
    validate_condition_id(&condition_id)?;
    let tick_size = decimal_field(
        market,
        &[
            "orderPriceMinTickSize",
            "minimumTickSize",
            "tickSize",
            "tick_size",
        ],
    )
    .ok_or_else(|| {
        source_error(
            "polymarket_contract_missing_tick_size",
            "Gamma market is missing its minimum tick size",
        )
    })?;
    if tick_size <= Decimal::ZERO || tick_size >= Decimal::ONE || !decimal_fits(tick_size, 18, 8) {
        return Err(source_error(
            "polymarket_contract_invalid_tick_size",
            "Gamma market minimum tick size must be between zero and one",
        ));
    }
    let minimum_order_size = decimal_field(market, &["orderMinSize", "minimumOrderSize"]);
    if minimum_order_size
        .is_some_and(|value| value <= Decimal::ZERO || !decimal_fits(value, 38, 18))
    {
        return Err(source_error(
            "polymarket_contract_invalid_minimum_order_size",
            "Gamma minimum order size must be positive when present",
        ));
    }
    let active = bool_field(market, &["active"])
        .or_else(|| bool_field(event, &["active"]))
        .unwrap_or(false);
    let closed = bool_field(market, &["closed"])
        .or_else(|| bool_field(event, &["closed"]))
        .unwrap_or(false);
    let accepting_orders =
        bool_field(market, &["acceptingOrders", "accepting_orders"]).unwrap_or(false);
    let fees_enabled = bool_field(market, &["feesEnabled", "fees_enabled"])
        .or_else(|| bool_field(event, &["feesEnabled", "fees_enabled"]))
        .unwrap_or(false);
    let fee_schedule = market
        .get("feeSchedule")
        .or_else(|| event.get("feeSchedule"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    if !fee_schedule.is_object() || canonical_json_bytes(&fee_schedule)?.len() > 8_192 {
        return Err(source_error(
            "polymarket_contract_invalid_fee_schedule",
            "Gamma fee schedule must be a bounded JSON object",
        ));
    }

    let source_payload = json!({
        "version": "polymarket-btc-5m-contract-v1",
        "source": SOURCE,
        "event_id": event_id,
        "event_slug": event_slug,
        "series_slug": series_slug,
        "market_id": market_id,
        "condition_id": condition_id,
        "window_start": window_start,
        "window_end": window_end,
        "up_token_id": up_token_id,
        "down_token_id": down_token_id,
        "tick_size": tick_size.normalize().to_string(),
        "minimum_order_size": minimum_order_size.map(|value| value.normalize().to_string()),
        "resolution_source": resolution_source,
        "active": active,
        "closed": closed,
        "accepting_orders": accepting_orders,
        "fees_enabled": fees_enabled,
        "fee_schedule": fee_schedule,
    });
    let payload_sha256 = hash_json(&source_payload)?;
    if canonical_json_bytes(&source_payload)?.len() > 65_536 {
        return Err(source_error(
            "polymarket_contract_projection_too_large",
            "canonical Gamma contract projection exceeds 65536 bytes",
        ));
    }
    Ok(MarketContract {
        event_id,
        event_slug,
        series_slug,
        market_id,
        condition_id,
        window_start,
        window_end,
        up_token_id,
        down_token_id,
        tick_size,
        minimum_order_size,
        resolution_source,
        active,
        closed,
        accepting_orders,
        fees_enabled,
        fee_schedule,
        received_at: microsecond_timestamp(received_at),
        source_payload,
        revision_sha256: payload_sha256.clone(),
        payload_sha256,
    })
}

pub(super) fn normalize_outcome(value: &str) -> Result<BtcOutcome, StrategyError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "up" => Ok(BtcOutcome::Up),
        "down" => Ok(BtcOutcome::Down),
        other => Err(source_error(
            "polymarket_contract_unexpected_outcome",
            format!("unexpected BTC interval outcome {other}"),
        )),
    }
}

fn validate_stored_contract_identities(
    contracts: &[MarketContract],
    stored: &[StoredContract],
) -> Result<(), StrategyError> {
    for candidate in contracts {
        for existing in stored {
            if contract_stored_namespace_collides(candidate, existing)
                && !candidate.immutable_identity_eq(existing)
            {
                return Err(integrity_error(
                    "polymarket_contract_identity_changed",
                    format!(
                        "Gamma reused or changed immutable identity namespace for market {} or event {}",
                        candidate.market_id, candidate.event_slug
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn validate_candidate_contract_identities(
    contracts: &[MarketContract],
) -> Result<(), StrategyError> {
    for (index, left) in contracts.iter().enumerate() {
        for right in &contracts[index.saturating_add(1)..] {
            if contract_candidate_namespace_collides(left, right)
                && !contract_candidates_have_same_identity(left, right)
            {
                return Err(integrity_error(
                    "polymarket_contract_batch_identity_collision",
                    format!(
                        "Gamma response batch reused an immutable identity across markets {} and {}",
                        left.market_id, right.market_id
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn contract_candidates_have_same_identity(left: &MarketContract, right: &MarketContract) -> bool {
    left.event_id == right.event_id
        && left.event_slug == right.event_slug
        && left.series_slug == right.series_slug
        && left.market_id == right.market_id
        && left.condition_id == right.condition_id
        && left.window_start == right.window_start
        && left.window_end == right.window_end
        && left.up_token_id == right.up_token_id
        && left.down_token_id == right.down_token_id
        && left.resolution_source == right.resolution_source
}

fn contract_candidate_namespace_collides(left: &MarketContract, right: &MarketContract) -> bool {
    left.market_id == right.market_id
        || left.event_id == right.event_id
        || left.event_slug == right.event_slug
        || left.condition_id == right.condition_id
        || left.up_token_id == right.up_token_id
        || left.up_token_id == right.down_token_id
        || left.down_token_id == right.up_token_id
        || left.down_token_id == right.down_token_id
}

fn contract_stored_namespace_collides(candidate: &MarketContract, stored: &StoredContract) -> bool {
    candidate.market_id == stored.market_id
        || candidate.event_id == stored.event_id
        || candidate.event_slug == stored.event_slug
        || candidate.condition_id == stored.condition_id
        || candidate.up_token_id == stored.up_token_id
        || candidate.up_token_id == stored.down_token_id
        || candidate.down_token_id == stored.up_token_id
        || candidate.down_token_id == stored.down_token_id
}

fn unique_contract_revisions(
    rows: Vec<StoredContract>,
) -> Result<BTreeMap<(String, String), StoredContract>, StrategyError> {
    let mut unique = BTreeMap::new();
    for row in rows {
        let key = (row.market_id.clone(), row.revision_sha256.clone());
        if unique.insert(key.clone(), row).is_some() {
            return Err(integrity_error(
                "polymarket_contract_duplicate_revision",
                format!(
                    "database returned duplicate contract revision {}:{}",
                    key.0, key.1
                ),
            ));
        }
    }
    Ok(unique)
}

fn contract_artifact_batch(contracts: &[&MarketContract]) -> ArtifactBatch {
    let minimum_source_timestamp = contracts.iter().map(|row| row.window_start).min();
    let maximum_source_timestamp = contracts.iter().map(|row| row.window_start).max();
    let minimum_received_at = contracts.iter().map(|row| row.received_at).min();
    let maximum_received_at = contracts.iter().map(|row| row.received_at).max();
    ArtifactBatch {
        inserted_record_count: contracts.len() as i64,
        minimum_source_timestamp,
        maximum_source_timestamp,
        minimum_received_at,
        maximum_received_at,
        start_cursor: minimum_source_timestamp.map(|timestamp| timestamp.timestamp().to_string()),
        end_cursor: maximum_source_timestamp.map(|timestamp| timestamp.timestamp().to_string()),
    }
}

async fn read_bounded_body(
    response: Response,
    maximum_bytes: usize,
    shutdown: &CancellationToken,
) -> Result<Vec<u8>, StrategyError> {
    if response
        .content_length()
        .is_some_and(|length| length > maximum_bytes as u64)
    {
        return Err(source_error(
            "polymarket_contract_body_too_large",
            format!("Gamma response exceeded {maximum_bytes} bytes"),
        ));
    }
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    loop {
        let next = tokio::select! {
            _ = shutdown.cancelled() => return Err(shutdown_error()),
            next = stream.next() => next,
        };
        let Some(chunk) = next else { break };
        let chunk = chunk.map_err(|error| {
            source_error(
                "polymarket_contract_body_failed",
                format!("failed reading Gamma response body: {error}"),
            )
        })?;
        if body.len().saturating_add(chunk.len()) > maximum_bytes {
            return Err(source_error(
                "polymarket_contract_body_too_large",
                format!("Gamma response exceeded {maximum_bytes} bytes"),
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn required_string(object: &Map<String, Value>, keys: &[&str]) -> Result<String, StrategyError> {
    string_field(object, keys).ok_or_else(|| {
        source_error(
            "polymarket_contract_missing_field",
            format!("Gamma response is missing required field {}", keys[0]),
        )
    })
}

fn string_field(object: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| match object.get(*key)? {
        Value::String(value) if !value.trim().is_empty() => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    })
}

fn string_array_field(object: &Map<String, Value>, keys: &[&str]) -> Option<Vec<String>> {
    keys.iter().find_map(|key| match object.get(*key)? {
        Value::Array(values) => values
            .iter()
            .map(|value| value.as_str().map(ToOwned::to_owned))
            .collect(),
        Value::String(value) => serde_json::from_str::<Vec<String>>(value).ok(),
        _ => None,
    })
}

fn decimal_field(object: &Map<String, Value>, keys: &[&str]) -> Option<Decimal> {
    keys.iter().find_map(|key| match object.get(*key)? {
        Value::String(value) => Decimal::from_str(value).ok(),
        Value::Number(value) => Decimal::from_str(&value.to_string()).ok(),
        _ => None,
    })
}

fn bool_field(object: &Map<String, Value>, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .find_map(|key| object.get(*key).and_then(Value::as_bool))
}

fn datetime_field(object: &Map<String, Value>, keys: &[&str]) -> Option<DateTime<Utc>> {
    keys.iter().find_map(|key| {
        let value = object.get(*key)?.as_str()?;
        DateTime::parse_from_rfc3339(value)
            .ok()
            .map(|timestamp| timestamp.with_timezone(&Utc))
    })
}

fn series_slug_from_relation(event: &Map<String, Value>) -> Option<String> {
    event
        .get("series")?
        .as_array()?
        .iter()
        .filter_map(Value::as_object)
        .find_map(|series| string_field(series, &["slug"]))
}

fn is_chainlink_btc_usd_source(source: &str) -> bool {
    let normalized = source.trim().to_ascii_lowercase();
    (normalized.contains("chainlink") || normalized.contains("chain.link"))
        && (normalized.contains("btc-usd")
            || normalized.contains("btc/usd")
            || (normalized.contains("btc") && normalized.contains("usd")))
}

fn validate_identifier(name: &str, value: &str, maximum_bytes: usize) -> Result<(), StrategyError> {
    if value.trim().is_empty() || value.len() > maximum_bytes {
        return Err(source_error(
            "polymarket_contract_invalid_identifier",
            format!("Gamma {name} must contain between 1 and {maximum_bytes} bytes"),
        ));
    }
    Ok(())
}

fn decimal_fits(value: Decimal, maximum_precision: u32, maximum_scale: u32) -> bool {
    let normalized = value.normalize();
    if normalized.scale() > maximum_scale {
        return false;
    }
    let digits = normalized.mantissa().unsigned_abs().to_string().len() as u32;
    let integer_digits = digits.saturating_sub(normalized.scale());
    digits <= maximum_precision && integer_digits <= maximum_precision.saturating_sub(maximum_scale)
}

fn validate_condition_id(value: &str) -> Result<(), StrategyError> {
    let valid = value.len() == 66
        && value.starts_with("0x")
        && value[2..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !valid {
        return Err(source_error(
            "polymarket_contract_invalid_condition_id",
            "Gamma condition ID must be lowercase 0x-prefixed 32-byte hexadecimal",
        ));
    }
    Ok(())
}

fn validate_token_id(value: &str) -> Result<(), StrategyError> {
    if value.is_empty() || value.len() > 100 || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(source_error(
            "polymarket_contract_invalid_token_id",
            "Gamma token ID must contain between 1 and 100 decimal digits",
        ));
    }
    Ok(())
}

fn is_aligned_window(timestamp: DateTime<Utc>) -> bool {
    timestamp.timestamp() >= 0
        && timestamp.timestamp_subsec_nanos() == 0
        && timestamp.timestamp().rem_euclid(INTERVAL_SECONDS) == 0
}

fn microsecond_timestamp(timestamp: DateTime<Utc>) -> DateTime<Utc> {
    DateTime::from_timestamp_micros(timestamp.timestamp_micros())
        .expect("a valid timestamp remains valid at microsecond precision")
}

fn validate_exact_origin(
    value: &str,
    expected: &str,
    field: &str,
) -> Result<(), StrategyFactoryError> {
    let parsed = Url::parse(value)
        .map_err(|error| invalid_config(format!("{field} is invalid: {error}")))?;
    if value != expected
        || parsed.scheme() != "https"
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.port().is_some()
        || parsed.path() != "/"
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(invalid_config(format!(
            "{field} must be the approved exact Polymarket HTTPS origin {expected}"
        )));
    }
    Ok(())
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, StrategyError> {
    serde_json::to_vec(value).map_err(|error| {
        integrity_error(
            "polymarket_contract_json_serialization",
            format!("failed to serialize canonical JSON: {error}"),
        )
    })
}

fn hash_json(value: &Value) -> Result<String, StrategyError> {
    canonical_json_bytes(value).map(|bytes| sha256_hex(&bytes))
}

fn sha256_hex(value: &[u8]) -> String {
    digest_hex(Sha256::digest(value))
}

fn digest_hex(digest: impl AsRef<[u8]>) -> String {
    let digest = digest.as_ref();
    let mut encoded = String::with_capacity(digest.len().saturating_mul(2));
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn verify_artifact_generation(actual: i64, expected: i64) -> Result<(), StrategyError> {
    if actual > expected {
        return Err(integrity_error(
            "polymarket_contract_artifact_generation_mismatch",
            format!("artifact generation {actual} is newer than profile generation {expected}"),
        ));
    }
    Ok(())
}

fn immutable_contract_conflict(contract: &MarketContract) -> StrategyError {
    integrity_error(
        "polymarket_contract_immutable_conflict",
        format!(
            "durable contract revision {}:{} differs from Gamma projection",
            contract.market_id, contract.revision_sha256
        ),
    )
}

fn invalid_config(message: impl Into<String>) -> StrategyFactoryError {
    StrategyFactoryError::InvalidConfiguration(message.into())
}

fn source_error(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::TransientSource, code, message)
}

fn database_error(code: &'static str, error: impl std::fmt::Display) -> StrategyError {
    StrategyError::new(
        StrategyErrorKind::TransientDatabase,
        code,
        error.to_string(),
    )
}

fn integrity_error(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::Integrity, code, message)
}

fn lease_lost_error() -> StrategyError {
    StrategyError::new(
        StrategyErrorKind::LeaseLost,
        "polymarket_contract_lease_lost",
        "Polymarket market-contract profile lease is no longer current",
    )
}

fn shutdown_error() -> StrategyError {
    StrategyError::new(
        StrategyErrorKind::Shutdown,
        "polymarket_contract_shutdown",
        "Polymarket market-contract strategy is shutting down",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/polymarket/gamma_btc_five_minute_event_v1.json"
        ))
        .expect("fixture is valid JSON")
    }

    fn fixture_window() -> DateTime<Utc> {
        Utc.timestamp_opt(1_783_902_600, 0).single().unwrap()
    }

    #[test]
    fn gamma_projection_maps_tokens_by_outcome_not_array_position() {
        let contract = parse_gamma_contract(&fixture(), fixture_window(), Utc::now()).unwrap();
        assert_eq!(contract.event_slug, "btc-updown-5m-1783902600");
        assert!(contract.up_token_id.starts_with('1'));
        assert!(contract.down_token_id.starts_with('2'));
        assert_eq!(contract.tick_size, Decimal::new(1, 2));
        assert_eq!(contract.minimum_order_size, None);
        assert_eq!(contract.revision_sha256, contract.payload_sha256);
        assert_eq!(
            contract.source_payload["version"],
            "polymarket-btc-5m-contract-v1"
        );
        assert!(contract.source_payload.get("volume").is_none());
        assert!(contract.source_payload.get("outcome_prices").is_none());
    }

    #[test]
    fn volatile_gamma_quote_fields_do_not_create_contract_revisions() {
        let mut changed = fixture();
        changed["markets"][0]["volume"] = json!("9999999");
        changed["markets"][0]["liquidity"] = json!("1");
        changed["markets"][0]["outcomePrices"] = json!("[\"0.99\",\"0.01\"]");
        let first = parse_gamma_contract(&fixture(), fixture_window(), Utc::now()).unwrap();
        let second = parse_gamma_contract(&changed, fixture_window(), Utc::now()).unwrap();
        assert_eq!(first.source_payload, second.source_payload);
        assert_eq!(first.revision_sha256, second.revision_sha256);
    }

    #[test]
    fn lifecycle_change_creates_a_new_narrow_revision() {
        let mut changed = fixture();
        changed["markets"][0]["closed"] = json!(true);
        changed["markets"][0]["acceptingOrders"] = json!(false);
        let first = parse_gamma_contract(&fixture(), fixture_window(), Utc::now()).unwrap();
        let second = parse_gamma_contract(&changed, fixture_window(), Utc::now()).unwrap();
        assert_ne!(first.revision_sha256, second.revision_sha256);
    }

    #[test]
    fn gamma_parser_rejects_wrong_series_window_and_resolution_source() {
        let mut wrong_series = fixture();
        wrong_series["seriesSlug"] = json!("btc-daily");
        assert!(parse_gamma_contract(&wrong_series, fixture_window(), Utc::now()).is_err());

        let wrong_window = fixture_window() + chrono::Duration::minutes(5);
        assert!(parse_gamma_contract(&fixture(), wrong_window, Utc::now()).is_err());

        let mut wrong_source = fixture();
        wrong_source["markets"][0]["resolutionSource"] = json!("Binance BTC/USDT");
        assert!(parse_gamma_contract(&wrong_source, fixture_window(), Utc::now()).is_err());
    }

    #[test]
    fn config_is_schema_v1_and_exact_origin_only() {
        let value =
            serde_json::to_value(PolymarketBtcFiveMinuteMarketContractsConfig::default()).unwrap();
        assert!(PolymarketBtcFiveMinuteMarketContractsConfig::from_value(&value).is_ok());

        let mut extra = value.clone();
        extra["labels"] = json!(true);
        assert!(PolymarketBtcFiveMinuteMarketContractsConfig::from_value(&extra).is_err());

        let mut redirected = value;
        redirected["gamma_base_url"] = json!("https://gamma-api.polymarket.com/redirect");
        assert!(PolymarketBtcFiveMinuteMarketContractsConfig::from_value(&redirected).is_err());
    }

    #[test]
    fn no_checkpoint_plans_the_configured_ended_lookback_and_realtime_tail() {
        let now = Utc.timestamp_opt(1_783_902_777, 0).single().unwrap();
        let current = aligned_window_start(now);
        let plan = discovery_window_plan(now, None, 12, 1, 2).unwrap();
        assert_eq!(
            plan.scan_windows.first(),
            Some(&(current - chrono::Duration::minutes(60)))
        );
        assert_eq!(
            plan.scan_windows.last(),
            Some(&(current - chrono::Duration::minutes(5)))
        );
        assert_eq!(plan.scan_windows.len(), 12);
        assert!(plan
            .windows
            .contains(&(current + chrono::Duration::minutes(5))));
    }

    #[test]
    fn stale_checkpoint_plans_only_the_oldest_bounded_prefix_plus_tail() {
        let now = Utc.timestamp_opt(1_783_902_777, 0).single().unwrap();
        let current = aligned_window_start(now);
        let stale = current - chrono::Duration::hours(12);
        let plan = discovery_window_plan(now, Some(stale), 12, 1, 2).unwrap();
        assert_eq!(
            plan.scan_windows.first(),
            Some(&(stale - chrono::Duration::minutes(10)))
        );
        assert_eq!(plan.scan_windows.len(), MAX_WINDOWS_PER_CYCLE as usize);
        assert!(plan.scan_windows.last().unwrap() < &(current - chrono::Duration::minutes(5)));
        assert!(plan
            .windows
            .contains(&(current - chrono::Duration::minutes(5))));
        assert!(plan.windows.contains(&current));
        assert!(plan
            .windows
            .contains(&(current + chrono::Duration::minutes(5))));
    }

    #[test]
    fn planner_retains_overlap_without_moving_scan_into_open_windows() {
        let now = Utc.timestamp_opt(1_783_902_777, 0).single().unwrap();
        let current = aligned_window_start(now);
        let cursor = current - chrono::Duration::minutes(15);
        let plan = discovery_window_plan(now, Some(cursor), 12, 1, 2).unwrap();
        assert_eq!(
            plan.scan_windows.first(),
            Some(&(cursor - chrono::Duration::minutes(10)))
        );
        assert_eq!(
            plan.scan_windows.last(),
            Some(&(current - chrono::Duration::minutes(5)))
        );
    }

    #[test]
    fn planner_rejects_a_scan_cursor_in_the_current_or_future_window() {
        let now = Utc.timestamp_opt(1_783_902_777, 0).single().unwrap();
        let current = aligned_window_start(now);
        assert!(discovery_window_plan(now, Some(current), 12, 1, 2).is_err());
        assert!(
            discovery_window_plan(now, Some(current + chrono::Duration::minutes(5)), 12, 1, 2)
                .is_err()
        );
    }

    #[test]
    fn empty_or_sparse_durable_evidence_does_not_skip_the_scan_frontier() {
        let start = fixture_window();
        let end = start + chrono::Duration::minutes(10);
        assert_eq!(
            contiguous_evidence_frontier(start, end, &BTreeSet::new()),
            None
        );

        let evidence = [start, end].into_iter().collect::<BTreeSet<_>>();
        assert_eq!(
            contiguous_evidence_frontier(start, end, &evidence),
            Some(start)
        );
    }

    #[test]
    fn decimal_validation_matches_database_integer_and_scale_capacity() {
        assert!(decimal_fits(
            Decimal::from_str("0.00000001").unwrap(),
            18,
            8
        ));
        assert!(!decimal_fits(
            Decimal::from_str("0.000000001").unwrap(),
            18,
            8
        ));
        assert!(decimal_fits(
            Decimal::from_str("99999999999999999999").unwrap(),
            38,
            18
        ));
        assert!(!decimal_fits(
            Decimal::from_str("100000000000000000000").unwrap(),
            38,
            18
        ));
    }

    #[test]
    fn decimal_formatting_does_not_create_a_false_contract_revision() {
        let mut formatted = fixture();
        formatted["markets"][0]["orderPriceMinTickSize"] = json!("0.010");
        let first = parse_gamma_contract(&fixture(), fixture_window(), Utc::now()).unwrap();
        let second = parse_gamma_contract(&formatted, fixture_window(), Utc::now()).unwrap();
        assert_eq!(first.tick_size, second.tick_size);
        assert_eq!(first.revision_sha256, second.revision_sha256);
    }
}
