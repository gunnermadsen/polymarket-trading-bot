//! Raw terminal evidence for Polymarket BTC Up/Down five-minute markets.
//!
//! This strategy deliberately persists provider evidence and narrow market identity only. It does
//! not derive training labels, features, or trading decisions. Gamma discovery is independent of
//! every other ingester profile and table; CLOB REST and the public market websocket are queried
//! directly so this process can be started, stopped, and repaired in isolation.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use futures_util::{stream, Sink, SinkExt, StreamExt};
use reqwest::{Client, Response, StatusCode, Url};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use sqlx::{FromRow, PgPool, Postgres, QueryBuilder, Transaction};
use tokio::{sync::mpsc, task::JoinHandle, time::Instant};
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{protocol::WebSocketConfig, Message},
};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    domain::{
        CaptureArtifact, DataGap, IngesterProfile, IngesterStrategyKey, RealtimeWorkerStrategy,
        StrategyError, StrategyErrorKind,
    },
    persistence::{
        ArtifactBatch, ArtifactRepository, GapRepository, NewCaptureArtifact, NewDataGap,
        ProfileRepository, StrategyDegradation, StrategyProgress,
    },
    runtime::{StrategyFactory, StrategyFactoryError},
};

use super::market_contracts::{
    aligned_window_start, parse_gamma_contract, slug_for_window, MarketContract, INTERVAL_SECONDS,
};

pub const STRATEGY_KEY: IngesterStrategyKey =
    IngesterStrategyKey::PolymarketBtcFiveMinuteResolutions;
pub const CONFIG_SCHEMA_VERSION: i32 = 1;
pub const CHECKPOINT_SCHEMA_VERSION: i32 = 1;

const DEFAULT_CLOB_BASE_URL: &str = "https://clob.polymarket.com";
const DEFAULT_CLOB_MARKET_ENDPOINT: &str = "markets";
const DEFAULT_GAMMA_BASE_URL: &str = "https://gamma-api.polymarket.com";
const DEFAULT_WEBSOCKET_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const MAX_HTTP_RESPONSE_BYTES: usize = 1_048_576;
const MAX_WEBSOCKET_FRAME_BYTES: usize = 1_048_576;
const MAX_MESSAGES_PER_FRAME: usize = 64;
const MAX_WINDOWS_PER_CYCLE: i64 = 64;
const MAX_PARALLEL_REQUESTS: usize = 4;
const MAX_GAP_REPAIRS_PER_CYCLE: i64 = 8;
const MAX_GAP_CANDIDATES_PER_CYCLE: i64 = 64;
const MAX_ABSENT_REPAIR_ATTEMPTS: i32 = 20;
const RESOLUTION_GAP_REASON: &str = "polymarket_resolution_unavailable";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct PolymarketBtcFiveMinuteResolutionsConfig {
    pub clob_base_url: String,
    pub clob_market_endpoint: String,
    pub gamma_base_url: String,
    pub websocket_url: String,
    pub poll_interval_seconds: u64,
    pub discovery_refresh_seconds: u64,
    pub startup_lookback_windows: u16,
    pub lookahead_windows: u8,
    pub gamma_fallback_grace_seconds: u64,
    pub retry_initial_seconds: u64,
    pub retry_max_seconds: u64,
    pub connect_timeout_ms: u64,
    pub read_timeout_ms: u64,
    pub ping_interval_ms: u64,
    pub pong_timeout_ms: u64,
    pub reconnect_initial_ms: u64,
    pub reconnect_max_ms: u64,
    pub artifact_window_seconds: i64,
    pub request_timeout_seconds: u64,
}

impl Default for PolymarketBtcFiveMinuteResolutionsConfig {
    fn default() -> Self {
        Self {
            clob_base_url: DEFAULT_CLOB_BASE_URL.to_owned(),
            clob_market_endpoint: DEFAULT_CLOB_MARKET_ENDPOINT.to_owned(),
            gamma_base_url: DEFAULT_GAMMA_BASE_URL.to_owned(),
            websocket_url: DEFAULT_WEBSOCKET_URL.to_owned(),
            poll_interval_seconds: 5,
            discovery_refresh_seconds: 5,
            startup_lookback_windows: 288,
            lookahead_windows: 1,
            gamma_fallback_grace_seconds: 120,
            retry_initial_seconds: 30,
            retry_max_seconds: 300,
            connect_timeout_ms: 10_000,
            read_timeout_ms: 40_000,
            ping_interval_ms: 10_000,
            pong_timeout_ms: 25_000,
            reconnect_initial_ms: 250,
            reconnect_max_ms: 30_000,
            artifact_window_seconds: 3_600,
            request_timeout_seconds: 10,
        }
    }
}

impl PolymarketBtcFiveMinuteResolutionsConfig {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let config = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            invalid_config(format!(
                "invalid Polymarket BTC five-minute resolution config: {error}"
            ))
        })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), StrategyFactoryError> {
        validate_exact_https_origin(&self.clob_base_url, DEFAULT_CLOB_BASE_URL, "clob_base_url")?;
        validate_exact_https_origin(
            &self.gamma_base_url,
            DEFAULT_GAMMA_BASE_URL,
            "gamma_base_url",
        )?;
        validate_exact_websocket_url(&self.websocket_url)?;
        if self.clob_market_endpoint != DEFAULT_CLOB_MARKET_ENDPOINT {
            return Err(invalid_config(
                "clob_market_endpoint must be the public CLOB markets endpoint",
            ));
        }
        if !(1..=300).contains(&self.poll_interval_seconds)
            || !(1..=300).contains(&self.discovery_refresh_seconds)
        {
            return Err(invalid_config(
                "poll and discovery refresh intervals must be between 1 and 300 seconds",
            ));
        }
        if !(1..=2_016).contains(&self.startup_lookback_windows) {
            return Err(invalid_config(
                "startup_lookback_windows must be between 1 and 2016",
            ));
        }
        if self.lookahead_windows > 2 {
            return Err(invalid_config("lookahead_windows must not exceed 2"));
        }
        if self.gamma_fallback_grace_seconds < 30 || self.gamma_fallback_grace_seconds > 3_600 {
            return Err(invalid_config(
                "gamma_fallback_grace_seconds must be between 30 and 3600",
            ));
        }
        if self.retry_initial_seconds == 0
            || self.retry_max_seconds < self.retry_initial_seconds
            || self.retry_max_seconds > 86_400
        {
            return Err(invalid_config(
                "retry bounds must be positive, ordered, and at most one day",
            ));
        }
        if !(1_000..=30_000).contains(&self.connect_timeout_ms)
            || !(5_000..=120_000).contains(&self.read_timeout_ms)
            || !(1_000..=30_000).contains(&self.ping_interval_ms)
            || self.ping_interval_ms >= self.read_timeout_ms
            || self.pong_timeout_ms <= self.ping_interval_ms
            || self.pong_timeout_ms > 60_000
        {
            return Err(invalid_config(
                "heartbeat bounds require ping < PONG timeout <= 60000 and ping < read timeout <= 120000",
            ));
        }
        if !(50..=10_000).contains(&self.reconnect_initial_ms)
            || self.reconnect_max_ms < self.reconnect_initial_ms
            || self.reconnect_max_ms > 60_000
        {
            return Err(invalid_config(
                "reconnect bounds must be ordered within 50 and 60000 milliseconds",
            ));
        }
        if self.artifact_window_seconds < INTERVAL_SECONDS
            || self.artifact_window_seconds > 86_400
            || 86_400 % self.artifact_window_seconds != 0
        {
            return Err(invalid_config(
                "artifact_window_seconds must divide one day and be between 300 and 86400",
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
struct ResolutionCheckpoint {
    last_scanned_window_start: Option<DateTime<Utc>>,
}

impl ResolutionCheckpoint {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let checkpoint = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "invalid Polymarket resolution checkpoint: {error}"
            ))
        })?;
        if checkpoint
            .last_scanned_window_start
            .is_some_and(|timestamp| !is_aligned_window(timestamp))
        {
            return Err(StrategyFactoryError::Construction(
                "Polymarket resolution checkpoint must be a nonnegative aligned UTC five-minute window"
                    .to_owned(),
            ));
        }
        Ok(checkpoint)
    }

    fn value(last_scanned_window_start: Option<DateTime<Utc>>) -> Value {
        json!({"last_scanned_window_start": last_scanned_window_start})
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MarketIdentity {
    market_id: String,
    condition_id: String,
    event_slug: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    up_token_id: String,
    down_token_id: String,
}

impl From<&MarketContract> for MarketIdentity {
    fn from(contract: &MarketContract) -> Self {
        Self {
            market_id: contract.market_id.clone(),
            condition_id: contract.condition_id.clone(),
            event_slug: contract.event_slug.clone(),
            window_start: contract.window_start,
            window_end: contract.window_end,
            up_token_id: contract.up_token_id.clone(),
            down_token_id: contract.down_token_id.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ResolutionSource {
    ClobWebsocket,
    ClobRestReconciliation,
    GammaRestReconciliation,
}

impl ResolutionSource {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ClobWebsocket => "clob_websocket",
            Self::ClobRestReconciliation => "clob_rest_reconciliation",
            Self::GammaRestReconciliation => "gamma_rest_reconciliation",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WinningOutcome {
    Up,
    Down,
}

impl WinningOutcome {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ResolutionFact {
    source: ResolutionSource,
    identity: MarketIdentity,
    winning_token_id: String,
    winning_outcome: WinningOutcome,
    source_timestamp: Option<DateTime<Utc>>,
    provider_available_at: Option<DateTime<Utc>>,
    received_at: DateTime<Utc>,
    source_payload: Value,
    revision_sha256: String,
    payload_sha256: String,
}

struct ResolutionEvidence {
    source: ResolutionSource,
    winning_token_id: String,
    winning_outcome: WinningOutcome,
    source_timestamp: Option<DateTime<Utc>>,
    received_at: DateTime<Utc>,
    source_payload: Value,
}

impl ResolutionFact {
    fn new(identity: MarketIdentity, evidence: ResolutionEvidence) -> Result<Self, StrategyError> {
        let received_at = microsecond_timestamp(evidence.received_at);
        let source_timestamp = evidence.source_timestamp.map(microsecond_timestamp);
        let provider_available_at = match evidence.source {
            ResolutionSource::ClobRestReconciliation => {
                if source_timestamp.is_some() {
                    return Err(integrity_error(
                        "polymarket_resolution_rest_source_time",
                        "CLOB REST resolution evidence cannot invent a provider timestamp",
                    ));
                }
                None
            }
            ResolutionSource::ClobWebsocket | ResolutionSource::GammaRestReconciliation => {
                Some(source_timestamp.ok_or_else(|| {
                    integrity_error(
                        "polymarket_resolution_missing_source_time",
                        "timestamped resolution evidence is missing its provider timestamp",
                    )
                })?)
            }
        };
        if received_at < identity.window_end
            || source_timestamp.is_some_and(|timestamp| timestamp < identity.window_end)
        {
            return Err(integrity_error(
                "polymarket_resolution_causality",
                "resolution evidence predates the end of its five-minute market window",
            ));
        }
        let expected_token = match evidence.winning_outcome {
            WinningOutcome::Up => &identity.up_token_id,
            WinningOutcome::Down => &identity.down_token_id,
        };
        if &evidence.winning_token_id != expected_token {
            return Err(integrity_error(
                "polymarket_resolution_winner_mismatch",
                "winning token does not map to the provider's winning Up/Down outcome",
            ));
        }
        if !evidence.source_payload.is_object() {
            return Err(integrity_error(
                "polymarket_resolution_payload_shape",
                "resolution source payload must be a JSON object",
            ));
        }
        let payload_bytes = canonical_json_bytes(&evidence.source_payload)?;
        if payload_bytes.len() > MAX_HTTP_RESPONSE_BYTES {
            return Err(source_error(
                "polymarket_resolution_payload_too_large",
                "canonical resolution evidence exceeds one mebibyte",
            ));
        }
        let payload_sha256 = sha256_hex(&payload_bytes);
        let revision_sha256 = hash_json(&json!({
            "schema_version": 1,
            "market_id": &identity.market_id,
            "condition_id": &identity.condition_id,
            "event_slug": &identity.event_slug,
            "window_start": identity.window_start,
            "window_end": identity.window_end,
            "up_token_id": &identity.up_token_id,
            "down_token_id": &identity.down_token_id,
            "winning_token_id": &evidence.winning_token_id,
            "winning_outcome": evidence.winning_outcome.as_str(),
        }))?;
        Ok(Self {
            source: evidence.source,
            identity,
            winning_token_id: evidence.winning_token_id,
            winning_outcome: evidence.winning_outcome,
            source_timestamp,
            provider_available_at,
            received_at,
            source_payload: evidence.source_payload,
            revision_sha256,
            payload_sha256,
        })
    }

    fn cursor(&self) -> String {
        format!(
            "resolution:{}:{}:{}",
            self.identity.window_start.timestamp(),
            self.identity.market_id,
            self.source.as_str()
        )
    }
}

#[derive(Debug, Clone, FromRow)]
struct StoredResolution {
    source: String,
    market_id: String,
    condition_id: String,
    event_slug: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    up_token_id: String,
    down_token_id: String,
    winning_token_id: String,
    winning_outcome: String,
    source_timestamp: Option<DateTime<Utc>>,
    provider_available_at: Option<DateTime<Utc>>,
    source_payload: Value,
    revision_sha256: String,
    payload_sha256: String,
}

impl StoredResolution {
    fn factual_eq(&self, fact: &ResolutionFact) -> bool {
        self.source == fact.source.as_str()
            && self.market_id == fact.identity.market_id
            && self.condition_id == fact.identity.condition_id
            && self.event_slug == fact.identity.event_slug
            && self.window_start == fact.identity.window_start
            && self.window_end == fact.identity.window_end
            && self.up_token_id == fact.identity.up_token_id
            && self.down_token_id == fact.identity.down_token_id
            && self.winning_token_id == fact.winning_token_id
            && self.winning_outcome == fact.winning_outcome.as_str()
            && self.source_timestamp == fact.source_timestamp
            && self.provider_available_at == fact.provider_available_at
            && self.source_payload == fact.source_payload
            && self.revision_sha256 == fact.revision_sha256
            && self.payload_sha256 == fact.payload_sha256
    }

    fn identity_eq(&self, fact: &ResolutionFact) -> bool {
        self.market_id == fact.identity.market_id
            && self.condition_id == fact.identity.condition_id
            && self.event_slug == fact.identity.event_slug
            && self.window_start == fact.identity.window_start
            && self.window_end == fact.identity.window_end
            && self.up_token_id == fact.identity.up_token_id
            && self.down_token_id == fact.identity.down_token_id
    }
}

#[derive(Debug, FromRow)]
struct ArtifactChecksumRow {
    source: String,
    market_id: String,
    payload_sha256: String,
    window_start: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy)]
struct DueResolutionGap {
    gap_id: Uuid,
    window_start: DateTime<Utc>,
}

#[derive(Debug)]
struct DiscoveryPlan {
    windows: Vec<DateTime<Utc>>,
    scan_windows: Vec<DateTime<Utc>>,
}

#[derive(Debug)]
enum ReconciliationResult {
    Resolved(Box<ResolutionFact>),
    Unavailable {
        window_start: DateTime<Utc>,
        reason: String,
    },
    Pending,
}

#[derive(Debug)]
enum ProducerNotice {
    Fact(Box<ResolutionFact>),
    Error(StrategyError),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PolymarketBtcFiveMinuteResolutionsFactory;

impl StrategyFactory for PolymarketBtcFiveMinuteResolutionsFactory {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    fn config_schema_version(&self) -> i32 {
        CONFIG_SCHEMA_VERSION
    }

    fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError> {
        PolymarketBtcFiveMinuteResolutionsConfig::from_value(config).map(|_| ())
    }

    fn build(
        &self,
        profile: &IngesterProfile,
        pool: PgPool,
    ) -> Result<Box<dyn RealtimeWorkerStrategy>, StrategyFactoryError> {
        if profile.strategy_key != STRATEGY_KEY {
            return Err(StrategyFactoryError::Construction(format!(
                "received profile for {}",
                profile.strategy_key
            )));
        }
        if profile.config_schema_version != CONFIG_SCHEMA_VERSION {
            return Err(StrategyFactoryError::Construction(format!(
                "Polymarket resolution config schema must be {CONFIG_SCHEMA_VERSION}"
            )));
        }
        if profile.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(StrategyFactoryError::Construction(format!(
                "Polymarket resolution checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}"
            )));
        }
        if profile.desired_generation <= 0 {
            return Err(StrategyFactoryError::Construction(
                "profile generation must be positive".to_owned(),
            ));
        }
        let config = PolymarketBtcFiveMinuteResolutionsConfig::from_value(&profile.config)?;
        let config_snapshot = serde_json::to_value(&config).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "failed to encode effective Polymarket resolution config: {error}"
            ))
        })?;
        let initial_checkpoint = ResolutionCheckpoint::from_value(&profile.checkpoint)?;
        let lease_owner = profile.lease_owner.clone().ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease owner is missing".to_owned())
        })?;
        let lease_token = profile.lease_token.ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease token is missing".to_owned())
        })?;
        let client = build_http_client(&config)?;
        Ok(Box::new(PolymarketBtcFiveMinuteResolutionsStrategy {
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

struct PolymarketBtcFiveMinuteResolutionsStrategy {
    config: PolymarketBtcFiveMinuteResolutionsConfig,
    config_snapshot: Value,
    profile_generation: i64,
    lease_owner: Arc<str>,
    lease_token: Uuid,
    initial_checkpoint: ResolutionCheckpoint,
    client: Client,
    pool: PgPool,
}

struct ResolutionRunState {
    last_scanned_window_start: Option<DateTime<Utc>>,
    artifact: Option<CaptureArtifact>,
    source_recheck_not_before: BTreeMap<(String, ResolutionSource), Instant>,
}

#[async_trait]
impl RealtimeWorkerStrategy for PolymarketBtcFiveMinuteResolutionsStrategy {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    async fn run(&self, shutdown: CancellationToken) -> Result<(), StrategyError> {
        let recovered_frontier =
            if let Some(checkpoint) = self.initial_checkpoint.last_scanned_window_start {
                if !self.checkpoint_has_durable_evidence(checkpoint).await? {
                    return Err(integrity_error(
                        "polymarket_resolution_checkpoint_without_evidence",
                        format!(
                        "resolution checkpoint {checkpoint} has no durable fact or gap evidence"
                    ),
                    ));
                }
                Some(checkpoint)
            } else {
                self.recover_initial_scan_frontier(Utc::now()).await?
            };
        let mut state = ResolutionRunState {
            last_scanned_window_start: recovered_frontier,
            artifact: None,
            source_recheck_not_before: BTreeMap::new(),
        };
        self.reconcile_open_artifact(&mut state).await?;
        let producer_shutdown = shutdown.child_token();
        let (producer, mut notices) = start_websocket_producer(
            self.client.clone(),
            self.config.clone(),
            producer_shutdown.clone(),
        );
        let mut ticker =
            tokio::time::interval(Duration::from_secs(self.config.poll_interval_seconds));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {
                    producer_shutdown.cancel();
                    await_producer(producer).await?;
                    self.seal_artifact(&mut state, true).await?;
                    return Ok(());
                }
                notice = notices.recv() => {
                    let Some(notice) = notice else {
                        producer_shutdown.cancel();
                        let producer_result = await_producer(producer).await;
                        let seal_result = self.seal_artifact(&mut state, false).await;
                        producer_result?;
                        seal_result?;
                        return Err(source_error(
                            "polymarket_resolution_websocket_worker_stopped",
                            "Polymarket resolution websocket worker stopped unexpectedly",
                        ));
                    };
                    let result = match notice {
                        ProducerNotice::Fact(fact) => {
                            self.persist_websocket_fact(&mut state, *fact).await
                        }
                        ProducerNotice::Error(error) => Err(error),
                    };
                    if let Err(error) = result {
                        match error.kind {
                            StrategyErrorKind::LeaseLost => {
                                producer_shutdown.cancel();
                                await_producer(producer).await?;
                                return self.finish_owned_drain(&mut state).await;
                            }
                            StrategyErrorKind::TransientSource | StrategyErrorKind::TransientDatabase => {
                                self.mark_degraded(&error).await?;
                                warn!(
                                    strategy = %STRATEGY_KEY,
                                    error_code = error.code,
                                    error = %error,
                                    "Polymarket resolution websocket path degraded and will retry"
                                );
                            }
                            _ => {
                                producer_shutdown.cancel();
                                await_producer(producer).await?;
                                let _ = self.seal_artifact(&mut state, false).await;
                                return Err(error);
                            }
                        }
                    }
                }
                _ = ticker.tick() => {
                    if let Err(error) = self.capture_cycle(&mut state, &shutdown).await {
                        match error.kind {
                            StrategyErrorKind::Shutdown => {
                                producer_shutdown.cancel();
                                await_producer(producer).await?;
                                self.seal_artifact(&mut state, true).await?;
                                return Ok(());
                            }
                            StrategyErrorKind::LeaseLost => {
                                producer_shutdown.cancel();
                                await_producer(producer).await?;
                                return self.finish_owned_drain(&mut state).await;
                            }
                            StrategyErrorKind::TransientSource | StrategyErrorKind::TransientDatabase => {
                                self.mark_degraded(&error).await?;
                                warn!(
                                    strategy = %STRATEGY_KEY,
                                    error_code = error.code,
                                    error = %error,
                                    "Polymarket resolution reconciliation degraded and will retry"
                                );
                            }
                            _ => {
                                producer_shutdown.cancel();
                                await_producer(producer).await?;
                                let _ = self.seal_artifact(&mut state, false).await;
                                return Err(error);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl PolymarketBtcFiveMinuteResolutionsStrategy {
    async fn reconcile_open_artifact(
        &self,
        state: &mut ResolutionRunState,
    ) -> Result<(), StrategyError> {
        let Some(open) = ArtifactRepository::new(self.pool.clone())
            .get_open(STRATEGY_KEY)
            .await
            .map_err(|error| {
                database_error("polymarket_resolution_artifact_recovery_failed", error)
            })?
        else {
            return Ok(());
        };
        verify_artifact_generation(open.profile_generation, self.profile_generation)?;
        if open.profile_generation == self.profile_generation
            && (open.config_schema_version != CONFIG_SCHEMA_VERSION
                || open.config_snapshot != self.config_snapshot)
        {
            return Err(integrity_error(
                "polymarket_resolution_artifact_config_conflict",
                "open current-generation artifact has a different effective configuration",
            ));
        }
        state.artifact = Some(open);
        self.seal_artifact(state, false).await
    }

    async fn checkpoint_has_durable_evidence(
        &self,
        checkpoint: DateTime<Utc>,
    ) -> Result<bool, StrategyError> {
        sqlx::query_scalar::<_, bool>(
            r#"
            SELECT EXISTS (
              SELECT 1
              FROM market_data.polymarket_btc_five_minute_resolutions
              WHERE window_start = $1
              LIMIT 1
            ) OR EXISTS (
              SELECT 1
              FROM ingester.data_gaps
              WHERE strategy_key = 'polymarket_btc_five_minute_resolutions'
                AND reason_code = 'polymarket_resolution_unavailable'
                AND source_time_start = $1
              LIMIT 1
            )
            "#,
        )
        .bind(checkpoint)
        .fetch_one(&self.pool)
        .await
        .map_err(|error| database_error("polymarket_resolution_checkpoint_read_failed", error))
    }

    async fn recover_initial_scan_frontier(
        &self,
        now: DateTime<Utc>,
    ) -> Result<Option<DateTime<Utc>>, StrategyError> {
        let latest = latest_matured_window(now, self.config.gamma_fallback_grace_seconds)?;
        let current = aligned_window_start(now);
        let start = current
            .checked_sub_signed(chrono::Duration::seconds(
                i64::from(self.config.startup_lookback_windows).saturating_mul(INTERVAL_SECONDS),
            ))
            .unwrap_or_else(unix_epoch);
        if start > latest {
            return Ok(None);
        }
        let rows = sqlx::query_scalar::<_, DateTime<Utc>>(
            r#"
            SELECT window_start
            FROM market_data.polymarket_btc_five_minute_resolutions
            WHERE window_start BETWEEN $1 AND $2
            UNION
            SELECT source_time_start AS window_start
            FROM ingester.data_gaps
            WHERE strategy_key = 'polymarket_btc_five_minute_resolutions'
              AND reason_code = 'polymarket_resolution_unavailable'
              AND source_time_start BETWEEN $1 AND $2
            ORDER BY window_start
            "#,
        )
        .bind(start)
        .bind(latest)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| database_error("polymarket_resolution_frontier_recovery_failed", error))?
        .into_iter()
        .collect::<BTreeSet<_>>();
        let mut frontier = None;
        let mut window = start;
        while window <= latest && rows.contains(&window) {
            frontier = Some(window);
            let Some(next) = window.checked_add_signed(chrono::Duration::seconds(INTERVAL_SECONDS))
            else {
                break;
            };
            window = next;
        }
        Ok(frontier)
    }

    async fn capture_cycle(
        &self,
        state: &mut ResolutionRunState,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        let observed_at = Utc::now();
        let plan = resolution_window_plan(
            observed_at,
            state.last_scanned_window_start,
            self.config.startup_lookback_windows,
            self.config.lookahead_windows,
            self.config.gamma_fallback_grace_seconds,
        )?;
        let scan_windows = plan.scan_windows.clone();
        let due_gaps = self.select_due_gaps().await?;
        let mut due_by_window = BTreeMap::new();
        for gap in &due_gaps {
            if due_by_window.insert(gap.window_start, gap.gap_id).is_some() {
                return Err(integrity_error(
                    "polymarket_resolution_duplicate_window_gap",
                    format!(
                        "multiple unresolved resolution gaps exist for window {}",
                        gap.window_start
                    ),
                ));
            }
        }
        let windows = plan
            .windows
            .into_iter()
            .chain(due_gaps.iter().map(|gap| gap.window_start))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let recheck = state.source_recheck_not_before.clone();
        let mut fetched = stream::iter(windows.into_iter().map(|window_start| {
            let is_gap_repair = due_by_window.contains_key(&window_start);
            let recheck = &recheck;
            async move {
                (
                    window_start,
                    self.reconcile_window(
                        window_start,
                        observed_at,
                        is_gap_repair,
                        recheck,
                        shutdown,
                    )
                    .await,
                )
            }
        }))
        .buffer_unordered(MAX_PARALLEL_REQUESTS)
        .collect::<Vec<_>>()
        .await;
        fetched.sort_by_key(|(window_start, _)| *window_start);

        let mut succeeded = BTreeSet::new();
        let mut normal_facts = Vec::new();
        let mut recovered = Vec::new();
        let mut unavailable = Vec::new();
        let mut absent_gap_ids = Vec::new();
        let mut first_error = None;
        for (window_start, result) in fetched {
            match result {
                Ok(ReconciliationResult::Resolved(fact)) => {
                    succeeded.insert(window_start);
                    if let Some(gap_id) = due_by_window.get(&window_start) {
                        recovered.push((*gap_id, *fact));
                    } else {
                        normal_facts.push(*fact);
                    }
                }
                Ok(ReconciliationResult::Unavailable {
                    window_start,
                    reason,
                }) => {
                    succeeded.insert(window_start);
                    if let Some(gap_id) = due_by_window.get(&window_start) {
                        absent_gap_ids.push(*gap_id);
                    } else {
                        unavailable.push((window_start, reason));
                    }
                }
                Ok(ReconciliationResult::Pending) => {}
                Err(error) if error.kind == StrategyErrorKind::Shutdown => return Err(error),
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }

        let mut next_scan_frontier = state.last_scanned_window_start;
        for window in scan_windows {
            if !succeeded.contains(&window) {
                break;
            }
            next_scan_frontier =
                Some(next_scan_frontier.map_or(window, |current| current.max(window)));
        }
        self.persist_normal_cycle(
            state,
            &normal_facts,
            &unavailable,
            &absent_gap_ids,
            next_scan_frontier,
        )
        .await?;
        if !recovered.is_empty() {
            self.seal_artifact(state, false).await?;
            self.persist_recovered_gaps(state, &recovered, next_scan_frontier)
                .await?;
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(())
    }

    async fn reconcile_window(
        &self,
        window_start: DateTime<Utc>,
        observed_at: DateTime<Utc>,
        is_gap_repair: bool,
        recheck: &BTreeMap<(String, ResolutionSource), Instant>,
        shutdown: &CancellationToken,
    ) -> Result<ReconciliationResult, StrategyError> {
        let Some((gamma, gamma_received_at, identity)) =
            fetch_gamma_identity(&self.client, &self.config, window_start, shutdown).await?
        else {
            return if window_is_mature(
                window_start,
                observed_at,
                self.config.gamma_fallback_grace_seconds,
            )? {
                Ok(ReconciliationResult::Unavailable {
                    window_start,
                    reason: format!(
                        "Gamma did not expose {} after its resolution grace",
                        slug_for_window(window_start)
                    ),
                })
            } else {
                Ok(ReconciliationResult::Pending)
            };
        };
        if observed_at < identity.window_end {
            return Ok(ReconciliationResult::Pending);
        }

        let clob_key = (
            identity.market_id.clone(),
            ResolutionSource::ClobRestReconciliation,
        );
        let clob_is_deferred = !is_gap_repair
            && recheck
                .get(&clob_key)
                .is_some_and(|not_before| *not_before > Instant::now());
        if !clob_is_deferred {
            if let Some(fact) =
                fetch_clob_rest_resolution(&self.client, &self.config, &identity, shutdown).await?
            {
                return Ok(ReconciliationResult::Resolved(Box::new(fact)));
            }
        } else {
            return Ok(ReconciliationResult::Pending);
        }

        if !window_is_mature(
            window_start,
            observed_at,
            self.config.gamma_fallback_grace_seconds,
        )? {
            return Ok(ReconciliationResult::Pending);
        }
        let gamma_key = (
            identity.market_id.clone(),
            ResolutionSource::GammaRestReconciliation,
        );
        if !is_gap_repair
            && recheck
                .get(&gamma_key)
                .is_some_and(|not_before| *not_before > Instant::now())
        {
            return Ok(ReconciliationResult::Pending);
        }
        if let Some(fact) = parse_gamma_resolution(
            &gamma,
            &identity,
            gamma_received_at,
            self.config.gamma_fallback_grace_seconds,
        )? {
            return Ok(ReconciliationResult::Resolved(Box::new(fact)));
        }
        Ok(ReconciliationResult::Unavailable {
            window_start,
            reason: format!(
                "CLOB and Gamma exposed no coherent terminal evidence for {} after the configured grace",
                identity.event_slug
            ),
        })
    }

    async fn select_due_gaps(&self) -> Result<Vec<DueResolutionGap>, StrategyError> {
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polymarket_resolution_gap_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let rows = sqlx::query_as::<_, (Uuid, Option<DateTime<Utc>>, Option<String>)>(
            r#"
            SELECT gap_id, source_time_start, start_cursor
            FROM ingester.data_gaps
            WHERE strategy_key = 'polymarket_btc_five_minute_resolutions'
              AND reason_code = 'polymarket_resolution_unavailable'
              AND status IN ('open', 'repairing')
              AND updated_at <= now() - (
                LEAST(
                  $2::numeric,
                  $1::numeric * power(2::numeric, LEAST(repair_attempts, 10))
                ) * INTERVAL '1 second'
              )
            ORDER BY repair_attempts, updated_at, detected_at, gap_id
            LIMIT $3
            FOR UPDATE SKIP LOCKED
            "#,
        )
        .bind(i64::try_from(self.config.retry_initial_seconds).unwrap_or(i64::MAX))
        .bind(i64::try_from(self.config.retry_max_seconds).unwrap_or(i64::MAX))
        .bind(MAX_GAP_CANDIDATES_PER_CYCLE)
        .fetch_all(&mut *transaction)
        .await
        .map_err(|error| database_error("polymarket_resolution_gap_read_failed", error))?;
        let mut due = Vec::new();
        for (gap_id, source_time_start, start_cursor) in rows {
            let window_start =
                parse_gap_window(gap_id, source_time_start, start_cursor.as_deref())?;
            due.push(DueResolutionGap {
                gap_id,
                window_start,
            });
            if due.len() >= MAX_GAP_REPAIRS_PER_CYCLE as usize {
                break;
            }
        }
        transaction.commit().await.map_err(|error| {
            database_error("polymarket_resolution_gap_transaction_commit_failed", error)
        })?;
        Ok(due)
    }

    async fn persist_websocket_fact(
        &self,
        state: &mut ResolutionRunState,
        fact: ResolutionFact,
    ) -> Result<(), StrategyError> {
        let checkpoint = state.last_scanned_window_start;
        self.persist_normal_cycle(state, &[fact], &[], &[], checkpoint)
            .await
    }

    async fn persist_normal_cycle(
        &self,
        state: &mut ResolutionRunState,
        facts: &[ResolutionFact],
        unavailable: &[(DateTime<Utc>, String)],
        absent_gap_ids: &[Uuid],
        next_scan_frontier: Option<DateTime<Utc>>,
    ) -> Result<(), StrategyError> {
        let artifact_id = if let Some(received_at) = facts.iter().map(|fact| fact.received_at).max()
        {
            Some(self.ensure_artifact(state, received_at).await?)
        } else {
            None
        };
        if facts.is_empty()
            && unavailable.is_empty()
            && absent_gap_ids.is_empty()
            && next_scan_frontier == state.last_scanned_window_start
        {
            return Ok(());
        }
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polymarket_resolution_transaction_begin_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let inserted = if let Some(artifact_id) = artifact_id {
            self.persist_facts_in(&mut transaction, artifact_id, facts)
                .await?
        } else {
            BTreeSet::new()
        };
        let gap_repository = GapRepository::new(self.pool.clone());
        for (window_start, reason) in unavailable {
            let detection = gap_repository
                .detect_in(
                    &mut transaction,
                    &NewDataGap {
                        strategy_key: STRATEGY_KEY,
                        detected_artifact_id: artifact_id,
                        gap_kind: "terminal_resolution".to_owned(),
                        reason_code: RESOLUTION_GAP_REASON.to_owned(),
                        reason_message: Some(reason.clone()),
                        source_time_start: Some(*window_start),
                        source_time_end: Some(
                            *window_start + chrono::Duration::seconds(INTERVAL_SECONDS),
                        ),
                        start_cursor: Some(window_start.timestamp().to_string()),
                        end_cursor: Some(window_start.timestamp().to_string()),
                    },
                )
                .await
                .map_err(|error| {
                    database_error("polymarket_resolution_gap_detect_failed", error)
                })?;
            if detection.inserted {
                warn!(strategy = %STRATEGY_KEY, error_code = RESOLUTION_GAP_REASON, gap_id = %detection.gap.gap_id, window_start = %window_start, "new Polymarket resolution gap detected");
            }
        }
        for gap_id in absent_gap_ids {
            let attempted = gap_repository
                .begin_repair_in(&mut transaction, *gap_id)
                .await
                .map_err(|error| database_error("polymarket_resolution_gap_attempt_failed", error))?
                .ok_or_else(|| {
                    integrity_error(
                        "polymarket_resolution_gap_not_retryable",
                        format!("due resolution gap {gap_id} was no longer unresolved"),
                    )
                })?;
            if attempted.repair_attempts >= MAX_ABSENT_REPAIR_ATTEMPTS {
                let terminal = gap_repository
                    .mark_unrecoverable_in(
                        &mut transaction,
                        *gap_id,
                        "polymarket_resolution_terminal_evidence_absent",
                        Some(
                            "CLOB and Gamma returned successful responses without coherent terminal evidence on every bounded repair attempt",
                        ),
                    )
                    .await
                    .map_err(|error| {
                        database_error("polymarket_resolution_gap_terminal_failed", error)
                    })?
                    .ok_or_else(|| {
                        integrity_error(
                            "polymarket_resolution_gap_terminal_race",
                            format!("resolution gap {gap_id} could not be terminalized"),
                        )
                    })?;
                warn!(strategy = %STRATEGY_KEY, error_code = "polymarket_resolution_terminal_evidence_absent", gap_id = %terminal.gap_id, repair_attempts = terminal.repair_attempts, "Polymarket resolution repair became terminal");
            }
        }

        let mut artifact_after_commit = None;
        if let Some(artifact_id) = artifact_id {
            if !inserted.is_empty() {
                let inserted_rows = facts
                    .iter()
                    .filter(|fact| inserted.contains(&fact_primary_key(fact)))
                    .collect::<Vec<_>>();
                artifact_after_commit = Some(
                    ArtifactRepository::new(self.pool.clone())
                        .record_batch_in(
                            &mut transaction,
                            artifact_id,
                            &resolution_artifact_batch(&inserted_rows),
                        )
                        .await
                        .map_err(|error| {
                            database_error("polymarket_resolution_artifact_progress_failed", error)
                        })?
                        .ok_or_else(|| {
                            integrity_error(
                                "polymarket_resolution_artifact_not_open",
                                format!("artifact {artifact_id} was not open during fact insert"),
                            )
                        })?,
                );
            }
        }
        if !facts.is_empty() {
            let progressed = ProfileRepository::new(self.pool.clone())
                .record_progress_in(
                    &mut transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                    &StrategyProgress {
                        verified_record_count: facts.len() as i64,
                        checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
                        checkpoint: ResolutionCheckpoint::value(next_scan_frontier),
                        last_source_event_at: facts
                            .iter()
                            .filter_map(|fact| fact.source_timestamp)
                            .max(),
                        last_provider_available_at: facts
                            .iter()
                            .filter_map(|fact| fact.provider_available_at)
                            .max(),
                        source_watermark: facts
                            .iter()
                            .filter_map(|fact| fact.source_timestamp)
                            .max(),
                        availability_watermark: facts
                            .iter()
                            .filter_map(|fact| fact.provider_available_at)
                            .max(),
                    },
                )
                .await
                .map_err(|error| {
                    database_error("polymarket_resolution_profile_progress_failed", error)
                })?;
            if !progressed {
                return Err(lease_lost_error());
            }
        } else {
            self.update_checkpoint_and_gap_health_in(&mut transaction, next_scan_frontier)
                .await?;
        }
        let has_unresolved_gaps = sqlx::query_scalar::<_, bool>(
            r#"
            SELECT EXISTS (
              SELECT 1
              FROM ingester.data_gaps
              WHERE strategy_key = $1
                AND status IN ('open', 'repairing')
            )
            "#,
        )
        .bind(STRATEGY_KEY.as_str())
        .fetch_one(&mut *transaction)
        .await
        .map_err(|error| database_error("polymarket_resolution_gap_health_read_failed", error))?;
        if has_unresolved_gaps && (!unavailable.is_empty() || !absent_gap_ids.is_empty()) {
            let marked = ProfileRepository::new(self.pool.clone())
                .mark_degraded_in(
                    &mut transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                    &StrategyDegradation {
                        reason_code: "polymarket_resolution_gap_repair".to_owned(),
                        reason_message: format!(
                            "tracking {} unresolved terminal market window(s)",
                            unavailable.len().saturating_add(absent_gap_ids.len())
                        ),
                    },
                )
                .await
                .map_err(|error| {
                    database_error("polymarket_resolution_gap_health_failed", error)
                })?;
            if !marked {
                return Err(lease_lost_error());
            }
        }
        transaction.commit().await.map_err(|error| {
            database_error("polymarket_resolution_transaction_commit_failed", error)
        })?;
        if let Some(artifact) = artifact_after_commit {
            state.artifact = Some(artifact);
        }
        state.last_scanned_window_start = next_scan_frontier;
        let recheck_at = Instant::now() + Duration::from_secs(self.config.retry_max_seconds);
        for fact in facts {
            state
                .source_recheck_not_before
                .insert((fact.identity.market_id.clone(), fact.source), recheck_at);
        }
        state
            .source_recheck_not_before
            .retain(|_, not_before| *not_before > Instant::now());
        Ok(())
    }

    async fn persist_recovered_gaps(
        &self,
        state: &mut ResolutionRunState,
        recovered: &[(Uuid, ResolutionFact)],
        next_scan_frontier: Option<DateTime<Utc>>,
    ) -> Result<(), StrategyError> {
        if recovered.is_empty() {
            return Ok(());
        }
        if state.artifact.is_some() {
            return Err(integrity_error(
                "polymarket_resolution_repair_artifact_not_dedicated",
                "resolution gap repair requires a dedicated capture artifact",
            ));
        }
        let received_at = recovered
            .iter()
            .map(|(_, fact)| fact.received_at)
            .max()
            .expect("a nonempty recovery has a receipt timestamp");
        let artifact_id = self.ensure_artifact(state, received_at).await?;
        let facts = recovered
            .iter()
            .map(|(_, fact)| fact.clone())
            .collect::<Vec<_>>();
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polymarket_resolution_repair_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let gaps = GapRepository::new(self.pool.clone());
        for (gap_id, fact) in recovered {
            let repairing = gaps
                .begin_repair_in(&mut transaction, *gap_id)
                .await
                .map_err(|error| database_error("polymarket_resolution_gap_begin_failed", error))?
                .ok_or_else(|| {
                    integrity_error(
                        "polymarket_resolution_gap_begin_race",
                        format!("resolution gap {gap_id} was no longer unresolved"),
                    )
                })?;
            validate_repair_gap(&repairing, fact.identity.window_start)?;
        }
        let inserted = self
            .persist_facts_in(&mut transaction, artifact_id, &facts)
            .await?;
        let mut artifact = state
            .artifact
            .as_ref()
            .filter(|artifact| artifact.artifact_id == artifact_id)
            .cloned()
            .ok_or_else(|| {
                integrity_error(
                    "polymarket_resolution_repair_artifact_missing",
                    format!("repair artifact {artifact_id} is not owned by the run state"),
                )
            })?;
        if !inserted.is_empty() {
            let inserted_rows = facts
                .iter()
                .filter(|fact| inserted.contains(&fact_primary_key(fact)))
                .collect::<Vec<_>>();
            artifact = ArtifactRepository::new(self.pool.clone())
                .record_batch_in(
                    &mut transaction,
                    artifact_id,
                    &resolution_artifact_batch(&inserted_rows),
                )
                .await
                .map_err(|error| {
                    database_error(
                        "polymarket_resolution_repair_artifact_progress_failed",
                        error,
                    )
                })?
                .ok_or_else(|| {
                    integrity_error(
                        "polymarket_resolution_repair_artifact_not_open",
                        format!("repair artifact {artifact_id} was not open during fact insert"),
                    )
                })?;
        }
        let (content_sha256, end_cursor) = self
            .artifact_checksum_in(&mut transaction, &artifact)
            .await?;
        ArtifactRepository::new(self.pool.clone())
            .complete_in(
                &mut transaction,
                artifact_id,
                &content_sha256,
                end_cursor.as_deref().or(artifact.end_cursor.as_deref()),
            )
            .await
            .map_err(|error| {
                database_error(
                    "polymarket_resolution_repair_artifact_complete_failed",
                    error,
                )
            })?
            .ok_or_else(|| {
                integrity_error(
                    "polymarket_resolution_repair_artifact_completion_race",
                    format!("repair artifact {artifact_id} was no longer open"),
                )
            })?;
        for (gap_id, _) in recovered {
            gaps.mark_repaired_in(
                &mut transaction,
                *gap_id,
                artifact_id,
                "polymarket_resolution_evidence_recovered",
                Some(
                    "coherent provider resolution evidence was durably inserted or replay-verified",
                ),
            )
            .await
            .map_err(|error| database_error("polymarket_resolution_gap_complete_failed", error))?
            .ok_or_else(|| {
                integrity_error(
                    "polymarket_resolution_gap_completion_race",
                    format!("resolution gap {gap_id} was not repairing"),
                )
            })?;
        }
        let progressed = ProfileRepository::new(self.pool.clone())
            .record_progress_in(
                &mut transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &StrategyProgress {
                    verified_record_count: facts.len() as i64,
                    checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
                    checkpoint: ResolutionCheckpoint::value(next_scan_frontier),
                    last_source_event_at: facts
                        .iter()
                        .filter_map(|fact| fact.source_timestamp)
                        .max(),
                    last_provider_available_at: facts
                        .iter()
                        .filter_map(|fact| fact.provider_available_at)
                        .max(),
                    source_watermark: facts.iter().filter_map(|fact| fact.source_timestamp).max(),
                    availability_watermark: facts
                        .iter()
                        .filter_map(|fact| fact.provider_available_at)
                        .max(),
                },
            )
            .await
            .map_err(|error| {
                database_error(
                    "polymarket_resolution_repair_profile_progress_failed",
                    error,
                )
            })?;
        if !progressed {
            return Err(lease_lost_error());
        }
        transaction.commit().await.map_err(|error| {
            database_error(
                "polymarket_resolution_repair_transaction_commit_failed",
                error,
            )
        })?;
        state.artifact = None;
        state.last_scanned_window_start = next_scan_frontier;
        let recheck_at = Instant::now() + Duration::from_secs(self.config.retry_max_seconds);
        for fact in &facts {
            state
                .source_recheck_not_before
                .insert((fact.identity.market_id.clone(), fact.source), recheck_at);
        }
        info!(
            strategy = %STRATEGY_KEY,
            %artifact_id,
            repaired_gap_count = recovered.len(),
            %content_sha256,
            "completed dedicated Polymarket resolution gap-repair artifact"
        );
        Ok(())
    }

    async fn persist_facts_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        facts: &[ResolutionFact],
    ) -> Result<BTreeSet<(String, String, String)>, StrategyError> {
        if facts.is_empty() {
            return Ok(BTreeSet::new());
        }
        validate_candidate_identities(facts)?;
        let market_ids = facts
            .iter()
            .map(|fact| fact.identity.market_id.clone())
            .collect::<Vec<_>>();
        let condition_ids = facts
            .iter()
            .map(|fact| fact.identity.condition_id.clone())
            .collect::<Vec<_>>();
        let event_slugs = facts
            .iter()
            .map(|fact| fact.identity.event_slug.clone())
            .collect::<Vec<_>>();
        let token_ids = facts
            .iter()
            .flat_map(|fact| {
                [
                    fact.identity.up_token_id.clone(),
                    fact.identity.down_token_id.clone(),
                ]
            })
            .collect::<Vec<_>>();
        let existing = self
            .load_facts_in(
                transaction,
                &market_ids,
                &condition_ids,
                &event_slugs,
                &token_ids,
            )
            .await?;
        validate_stored_identities(facts, &existing)?;
        let existing_by_key = unique_stored_facts(existing)?;
        let mut missing = Vec::new();
        for fact in facts {
            let key = fact_primary_key(fact);
            if let Some(stored) = existing_by_key.get(&key) {
                if !stored.factual_eq(fact) {
                    return Err(immutable_fact_conflict(fact));
                }
            } else {
                missing.push(fact);
            }
        }
        let inserted = self
            .insert_missing_facts_in(transaction, artifact_id, &missing)
            .await?;
        let durable = unique_stored_facts(
            self.load_facts_in(
                transaction,
                &market_ids,
                &condition_ids,
                &event_slugs,
                &token_ids,
            )
            .await?,
        )?;
        for fact in facts {
            let key = fact_primary_key(fact);
            let stored = durable.get(&key).ok_or_else(|| {
                integrity_error(
                    "polymarket_resolution_insert_missing",
                    format!(
                        "resolution evidence {}:{}:{} was absent after insert",
                        key.0, key.1, key.2
                    ),
                )
            })?;
            if !stored.factual_eq(fact) {
                return Err(immutable_fact_conflict(fact));
            }
        }
        Ok(inserted)
    }

    async fn load_facts_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        market_ids: &[String],
        condition_ids: &[String],
        event_slugs: &[String],
        token_ids: &[String],
    ) -> Result<Vec<StoredResolution>, StrategyError> {
        sqlx::query_as::<_, StoredResolution>(
            r#"
            SELECT source, market_id, condition_id, event_slug,
                   window_start, window_end, up_token_id, down_token_id,
                   winning_token_id, winning_outcome,
                   source_timestamp, provider_available_at, source_payload,
                   revision_sha256::text AS revision_sha256,
                   payload_sha256::text AS payload_sha256
            FROM market_data.polymarket_btc_five_minute_resolutions
            WHERE market_id = ANY($1::text[])
               OR condition_id = ANY($2::text[])
               OR event_slug = ANY($3::text[])
               OR up_token_id = ANY($4::text[])
               OR down_token_id = ANY($4::text[])
            ORDER BY market_id, source, payload_sha256
            "#,
        )
        .bind(market_ids)
        .bind(condition_ids)
        .bind(event_slugs)
        .bind(token_ids)
        .fetch_all(&mut **transaction)
        .await
        .map_err(|error| database_error("polymarket_resolution_fact_read_failed", error))
    }

    async fn insert_missing_facts_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        facts: &[&ResolutionFact],
    ) -> Result<BTreeSet<(String, String, String)>, StrategyError> {
        if facts.is_empty() {
            return Ok(BTreeSet::new());
        }
        let mut builder = QueryBuilder::<Postgres>::new(
            r#"
            INSERT INTO market_data.polymarket_btc_five_minute_resolutions (
              source, market_id, condition_id, event_slug,
              window_start, window_end, up_token_id, down_token_id,
              winning_token_id, winning_outcome,
              source_timestamp, provider_available_at, received_at,
              source_payload, revision_sha256, payload_sha256,
              strategy_key, capture_artifact_id
            )
            "#,
        );
        builder.push_values(facts, |mut row, fact| {
            row.push_bind(fact.source.as_str())
                .push_bind(&fact.identity.market_id)
                .push_bind(&fact.identity.condition_id)
                .push_bind(&fact.identity.event_slug)
                .push_bind(fact.identity.window_start)
                .push_bind(fact.identity.window_end)
                .push_bind(&fact.identity.up_token_id)
                .push_bind(&fact.identity.down_token_id)
                .push_bind(&fact.winning_token_id)
                .push_bind(fact.winning_outcome.as_str())
                .push_bind(fact.source_timestamp)
                .push_bind(fact.provider_available_at)
                .push_bind(fact.received_at)
                .push_bind(&fact.source_payload)
                .push_bind(&fact.revision_sha256)
                .push_bind(&fact.payload_sha256)
                .push_bind(STRATEGY_KEY.as_str())
                .push_bind(artifact_id);
        });
        builder.push(
            " ON CONFLICT (market_id, source, payload_sha256) DO NOTHING \
             RETURNING market_id, source, payload_sha256::text",
        );
        builder
            .build_query_as::<(String, String, String)>()
            .fetch_all(&mut **transaction)
            .await
            .map(|rows| rows.into_iter().collect())
            .map_err(|error| database_error("polymarket_resolution_fact_insert_failed", error))
    }

    async fn update_checkpoint_and_gap_health_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        next_scan_frontier: Option<DateTime<Utc>>,
    ) -> Result<(), StrategyError> {
        let updated = sqlx::query_scalar::<_, String>(
            r#"
            WITH gap_health AS (
              SELECT EXISTS (
                SELECT 1
                FROM ingester.data_gaps
                WHERE strategy_key = $1
                  AND status IN ('open', 'repairing')
              ) AS has_unresolved_gap
            )
            UPDATE ingester.profiles
            SET checkpoint_schema_version = $5,
                checkpoint = $6,
                observed_state = CASE
                  WHEN gap_health.has_unresolved_gap THEN 'degraded' ELSE 'running'
                END,
                health_status = CASE
                  WHEN gap_health.has_unresolved_gap THEN 'degraded' ELSE 'healthy'
                END,
                updated_at = now()
            FROM gap_health
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
        .bind(ResolutionCheckpoint::value(next_scan_frontier))
        .fetch_optional(&mut **transaction)
        .await
        .map_err(|error| database_error("polymarket_resolution_gap_checkpoint_failed", error))?;
        if updated.is_none() {
            return Err(lease_lost_error());
        }
        Ok(())
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
        state: &mut ResolutionRunState,
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
            .map_err(|error| database_error("polymarket_resolution_artifact_read_failed", error))?
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
                    "polymarket_resolution_artifact_config_conflict",
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
            database_error("polymarket_resolution_artifact_transaction_failed", error)
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
                        .last_scanned_window_start
                        .map(|timestamp| timestamp.timestamp().to_string()),
                },
            )
            .await
            .map_err(|error| {
                database_error("polymarket_resolution_artifact_create_failed", error)
            })?;
        transaction.commit().await.map_err(|error| {
            database_error("polymarket_resolution_artifact_commit_failed", error)
        })?;
        let artifact_id = artifact.artifact_id;
        state.artifact = Some(artifact);
        info!(
            strategy = %STRATEGY_KEY,
            %artifact_id,
            %window_start,
            %window_end,
            "opened Polymarket resolution capture artifact"
        );
        Ok(artifact_id)
    }

    async fn seal_artifact(
        &self,
        state: &mut ResolutionRunState,
        allow_draining_generation: bool,
    ) -> Result<(), StrategyError> {
        let Some(artifact) = state.artifact.as_ref().cloned() else {
            let mut transaction = self.pool.begin().await.map_err(|error| {
                database_error("polymarket_resolution_drain_transaction_failed", error)
            })?;
            self.assert_lease_in(&mut transaction, allow_draining_generation)
                .await?;
            transaction.commit().await.map_err(|error| {
                database_error("polymarket_resolution_drain_commit_failed", error)
            })?;
            return Ok(());
        };
        verify_artifact_generation(artifact.profile_generation, self.profile_generation)?;
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polymarket_resolution_artifact_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, allow_draining_generation)
            .await?;
        let (content_sha256, end_cursor) = self
            .artifact_checksum_in(&mut transaction, &artifact)
            .await?;
        ArtifactRepository::new(self.pool.clone())
            .complete_in(
                &mut transaction,
                artifact.artifact_id,
                &content_sha256,
                end_cursor.as_deref().or(artifact.end_cursor.as_deref()),
            )
            .await
            .map_err(|error| {
                database_error("polymarket_resolution_artifact_complete_failed", error)
            })?
            .ok_or_else(|| {
                integrity_error(
                    "polymarket_resolution_artifact_not_open",
                    format!(
                        "artifact {} was not open while sealing",
                        artifact.artifact_id
                    ),
                )
            })?;
        transaction.commit().await.map_err(|error| {
            database_error("polymarket_resolution_artifact_commit_failed", error)
        })?;
        state.artifact = None;
        info!(
            strategy = %STRATEGY_KEY,
            artifact_id = %artifact.artifact_id,
            record_count = artifact.record_count,
            %content_sha256,
            "sealed Polymarket resolution capture artifact"
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
            SELECT source, market_id, payload_sha256::text AS payload_sha256, window_start
            FROM market_data.polymarket_btc_five_minute_resolutions
            WHERE capture_artifact_id = $1 AND strategy_key = $2
            ORDER BY window_start, market_id, source, payload_sha256
            "#,
        )
        .bind(artifact.artifact_id)
        .bind(STRATEGY_KEY.as_str())
        .fetch_all(&mut **transaction)
        .await
        .map_err(|error| database_error("polymarket_resolution_checksum_read_failed", error))?;
        if i64::try_from(rows.len()).ok() != Some(artifact.record_count) {
            return Err(integrity_error(
                "polymarket_resolution_artifact_count_mismatch",
                format!(
                    "artifact {} records {} rows but owns {} resolution facts",
                    artifact.artifact_id,
                    artifact.record_count,
                    rows.len()
                ),
            ));
        }
        let mut hasher = Sha256::new();
        for row in &rows {
            hash_field(&mut hasher, &row.market_id);
            hash_field(&mut hasher, &row.source);
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
        .map_err(|error| database_error("polymarket_resolution_lease_check_failed", error))?;
        if !current {
            return Err(lease_lost_error());
        }
        Ok(())
    }

    async fn mark_degraded(&self, error: &StrategyError) -> Result<(), StrategyError> {
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
            .map_err(|database| {
                database_error("polymarket_resolution_degraded_state_failed", database)
            })?;
        if !marked {
            return Err(lease_lost_error());
        }
        Ok(())
    }

    async fn finish_owned_drain(
        &self,
        state: &mut ResolutionRunState,
    ) -> Result<(), StrategyError> {
        self.seal_artifact(state, true).await?;
        info!(
            strategy = %STRATEGY_KEY,
            generation = self.profile_generation,
            "Polymarket resolution strategy drained after a desired-state lease race"
        );
        Ok(())
    }
}

fn resolution_window_plan(
    now: DateTime<Utc>,
    last_scanned_window_start: Option<DateTime<Utc>>,
    startup_lookback_windows: u16,
    lookahead_windows: u8,
    gamma_fallback_grace_seconds: u64,
) -> Result<DiscoveryPlan, StrategyError> {
    let current = aligned_window_start(now);
    let latest_matured = latest_matured_window(now, gamma_fallback_grace_seconds)?;
    let scan_start = if let Some(last) = last_scanned_window_start {
        if !is_aligned_window(last) {
            return Err(integrity_error(
                "polymarket_resolution_cursor_unaligned",
                "durable resolution cursor is not an aligned five-minute window",
            ));
        }
        if last > current {
            return Err(integrity_error(
                "polymarket_resolution_cursor_in_future",
                format!("durable resolution cursor {last} is newer than current window {current}"),
            ));
        }
        last.checked_add_signed(chrono::Duration::seconds(INTERVAL_SECONDS))
            .ok_or_else(|| {
                integrity_error(
                    "polymarket_resolution_window_overflow",
                    "resolution cursor successor overflowed",
                )
            })?
    } else {
        current
            .checked_sub_signed(chrono::Duration::seconds(
                i64::from(startup_lookback_windows).saturating_mul(INTERVAL_SECONDS),
            ))
            .unwrap_or_else(unix_epoch)
    };
    let mut scan_windows = Vec::new();
    if scan_start <= latest_matured {
        let available = (latest_matured - scan_start)
            .num_seconds()
            .div_euclid(INTERVAL_SECONDS)
            .saturating_add(1);
        let count = available.min(MAX_WINDOWS_PER_CYCLE);
        for index in 0..count {
            scan_windows.push(
                scan_start
                    .checked_add_signed(chrono::Duration::seconds(
                        index.saturating_mul(INTERVAL_SECONDS),
                    ))
                    .ok_or_else(|| {
                        integrity_error(
                            "polymarket_resolution_window_overflow",
                            "resolution scan window overflowed",
                        )
                    })?,
            );
        }
    }
    let latest_tail = current
        .checked_add_signed(chrono::Duration::seconds(
            i64::from(lookahead_windows).saturating_mul(INTERVAL_SECONDS),
        ))
        .ok_or_else(|| {
            integrity_error(
                "polymarket_resolution_window_overflow",
                "resolution lookahead overflowed",
            )
        })?;
    let tail_start = current
        .checked_sub_signed(chrono::Duration::seconds(INTERVAL_SECONDS))
        .unwrap_or_else(unix_epoch);
    let tail_count = (latest_tail - tail_start)
        .num_seconds()
        .div_euclid(INTERVAL_SECONDS)
        .saturating_add(1);
    let mut windows = scan_windows.iter().copied().collect::<BTreeSet<_>>();
    for index in 0..tail_count {
        windows.insert(
            tail_start
                .checked_add_signed(chrono::Duration::seconds(
                    index.saturating_mul(INTERVAL_SECONDS),
                ))
                .ok_or_else(|| {
                    integrity_error(
                        "polymarket_resolution_window_overflow",
                        "resolution realtime tail overflowed",
                    )
                })?,
        );
    }
    Ok(DiscoveryPlan {
        windows: windows.into_iter().collect(),
        scan_windows,
    })
}

fn latest_matured_window(
    now: DateTime<Utc>,
    gamma_fallback_grace_seconds: u64,
) -> Result<DateTime<Utc>, StrategyError> {
    let current = aligned_window_start(now);
    let previous = current
        .checked_sub_signed(chrono::Duration::seconds(INTERVAL_SECONDS))
        .unwrap_or_else(unix_epoch);
    if window_is_mature(previous, now, gamma_fallback_grace_seconds)? {
        Ok(previous)
    } else {
        Ok(previous
            .checked_sub_signed(chrono::Duration::seconds(INTERVAL_SECONDS))
            .unwrap_or_else(unix_epoch))
    }
}

fn window_is_mature(
    window_start: DateTime<Utc>,
    now: DateTime<Utc>,
    grace_seconds: u64,
) -> Result<bool, StrategyError> {
    let grace = i64::try_from(grace_seconds).map_err(|_| {
        integrity_error(
            "polymarket_resolution_grace_overflow",
            "Gamma fallback grace is outside the supported timestamp range",
        )
    })?;
    let mature_at = window_start
        .checked_add_signed(chrono::Duration::seconds(
            INTERVAL_SECONDS.saturating_add(grace),
        ))
        .ok_or_else(|| {
            integrity_error(
                "polymarket_resolution_window_overflow",
                "resolution maturity timestamp overflowed",
            )
        })?;
    Ok(now >= mature_at)
}

async fn fetch_gamma_identity(
    client: &Client,
    config: &PolymarketBtcFiveMinuteResolutionsConfig,
    window_start: DateTime<Utc>,
    shutdown: &CancellationToken,
) -> Result<Option<(Value, DateTime<Utc>, MarketIdentity)>, StrategyError> {
    let slug = slug_for_window(window_start);
    let endpoint = format!("{}/events/slug/{slug}", config.gamma_base_url);
    let response = tokio::select! {
        _ = shutdown.cancelled() => return Err(shutdown_error()),
        response = client.get(endpoint).send() => response,
    }
    .map_err(|error| {
        source_error(
            "polymarket_resolution_gamma_request_failed",
            format!("Gamma resolution discovery for {slug} failed: {error}"),
        )
    })?;
    let status = response.status();
    let body = read_bounded_body(
        response,
        MAX_HTTP_RESPONSE_BYTES,
        "Gamma resolution discovery",
        shutdown,
    )
    .await?;
    let received_at = microsecond_timestamp(Utc::now());
    if status == StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !status.is_success() {
        return Err(source_error(
            "polymarket_resolution_gamma_http_status",
            format!("Gamma resolution discovery for {slug} returned HTTP {status}"),
        ));
    }
    let value = serde_json::from_slice::<Value>(&body).map_err(|error| {
        source_error(
            "polymarket_resolution_gamma_decode_failed",
            format!("Gamma resolution discovery for {slug} was invalid JSON: {error}"),
        )
    })?;
    let contract = parse_gamma_contract(&value, window_start, received_at)?;
    Ok(Some((value, received_at, MarketIdentity::from(&contract))))
}

async fn fetch_clob_rest_resolution(
    client: &Client,
    config: &PolymarketBtcFiveMinuteResolutionsConfig,
    identity: &MarketIdentity,
    shutdown: &CancellationToken,
) -> Result<Option<ResolutionFact>, StrategyError> {
    let endpoint = format!(
        "{}/{}/{}",
        config.clob_base_url, config.clob_market_endpoint, identity.condition_id
    );
    let response = tokio::select! {
        _ = shutdown.cancelled() => return Err(shutdown_error()),
        response = client.get(endpoint).send() => response,
    }
    .map_err(|error| {
        source_error(
            "polymarket_resolution_clob_request_failed",
            format!(
                "CLOB resolution reconciliation for {} failed: {error}",
                identity.condition_id
            ),
        )
    })?;
    let status = response.status();
    let body = read_bounded_body(
        response,
        MAX_HTTP_RESPONSE_BYTES,
        "CLOB resolution reconciliation",
        shutdown,
    )
    .await?;
    let received_at = microsecond_timestamp(Utc::now());
    if status == StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !status.is_success() {
        return Err(source_error(
            "polymarket_resolution_clob_http_status",
            format!(
                "CLOB resolution reconciliation for {} returned HTTP {status}",
                identity.condition_id
            ),
        ));
    }
    let value = serde_json::from_slice::<Value>(&body).map_err(|error| {
        source_error(
            "polymarket_resolution_clob_decode_failed",
            format!(
                "CLOB resolution reconciliation for {} was invalid JSON: {error}",
                identity.condition_id
            ),
        )
    })?;
    parse_clob_rest_resolution(&value, identity, received_at)
}

fn parse_clob_rest_resolution(
    value: &Value,
    identity: &MarketIdentity,
    received_at: DateTime<Utc>,
) -> Result<Option<ResolutionFact>, StrategyError> {
    let market = value.as_object().ok_or_else(|| {
        source_error(
            "polymarket_resolution_clob_invalid_market",
            "CLOB market reconciliation response must be an object",
        )
    })?;
    let condition_id = required_string(market, &["condition_id", "conditionId"], "CLOB market")?;
    if condition_id != identity.condition_id {
        return Err(source_error(
            "polymarket_resolution_clob_identity_mismatch",
            "CLOB reconciliation condition ID differs from Gamma discovery",
        ));
    }
    if let Some(slug) = string_field(market, &["market_slug", "marketSlug", "slug"]) {
        if slug != identity.event_slug {
            return Err(source_error(
                "polymarket_resolution_clob_identity_mismatch",
                "CLOB reconciliation slug differs from Gamma discovery",
            ));
        }
    }
    let closed = bool_field(market, &["closed"]);
    let accepting_orders = bool_field(market, &["accepting_orders", "acceptingOrders"]);
    if closed != Some(true) || accepting_orders != Some(false) {
        return Ok(None);
    }
    if bool_field(market, &["enable_order_book", "enableOrderBook"]) == Some(true) {
        return Err(source_error(
            "polymarket_resolution_clob_terminal_incoherent",
            "closed CLOB resolution evidence still enables its order book",
        ));
    }
    let tokens = market
        .get("tokens")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            source_error(
                "polymarket_resolution_clob_missing_tokens",
                "terminal CLOB market is missing its token array",
            )
        })?;
    if tokens.len() != 2 {
        return Err(source_error(
            "polymarket_resolution_clob_ambiguous_tokens",
            "terminal CLOB market must contain exactly two tokens",
        ));
    }
    let mut seen_tokens = BTreeSet::new();
    let mut projected_tokens = Vec::with_capacity(2);
    let mut winner = None;
    for token in tokens {
        let token = token.as_object().ok_or_else(|| {
            source_error(
                "polymarket_resolution_clob_invalid_token",
                "CLOB terminal token must be an object",
            )
        })?;
        let token_id = required_string(token, &["token_id", "tokenId"], "CLOB token")?;
        validate_token_id(&token_id, "CLOB token")?;
        if !seen_tokens.insert(token_id.clone()) {
            return Err(source_error(
                "polymarket_resolution_clob_ambiguous_tokens",
                "terminal CLOB market contains duplicate token IDs",
            ));
        }
        let outcome = parse_winning_outcome(&required_string(token, &["outcome"], "CLOB token")?)?;
        let expected = match outcome {
            WinningOutcome::Up => &identity.up_token_id,
            WinningOutcome::Down => &identity.down_token_id,
        };
        if &token_id != expected {
            return Err(source_error(
                "polymarket_resolution_clob_identity_mismatch",
                "CLOB token/outcome mapping differs from Gamma discovery",
            ));
        }
        let price = required_decimal(token, &["price"], "CLOB token")?;
        let is_winner = required_bool(token, &["winner"], "CLOB token")?;
        if (is_winner && price != Decimal::ONE) || (!is_winner && price != Decimal::ZERO) {
            return Err(source_error(
                "polymarket_resolution_clob_nonbinary_price",
                "CLOB terminal token winner flags must agree with exact binary prices",
            ));
        }
        if is_winner && winner.replace((token_id.clone(), outcome)).is_some() {
            return Err(source_error(
                "polymarket_resolution_clob_ambiguous_winner",
                "CLOB terminal market names more than one winning token",
            ));
        }
        projected_tokens.push(json!({
            "token_id": token_id,
            "outcome": outcome.as_str(),
            "price": price.normalize().to_string(),
            "winner": is_winner,
        }));
    }
    let (winning_token_id, winning_outcome) = winner.ok_or_else(|| {
        source_error(
            "polymarket_resolution_clob_missing_winner",
            "CLOB terminal market does not name a winning token",
        )
    })?;
    projected_tokens
        .sort_by(|left, right| left["outcome"].as_str().cmp(&right["outcome"].as_str()));
    let source_payload = json!({
        "schema_version": 1,
        "source": ResolutionSource::ClobRestReconciliation.as_str(),
        "condition_id": condition_id,
        "market_slug": identity.event_slug,
        "active": bool_field(market, &["active"]),
        "closed": closed,
        "accepting_orders": accepting_orders,
        "enable_order_book": bool_field(market, &["enable_order_book", "enableOrderBook"]),
        "tokens": projected_tokens,
    });
    ResolutionFact::new(
        identity.clone(),
        ResolutionEvidence {
            source: ResolutionSource::ClobRestReconciliation,
            winning_token_id,
            winning_outcome,
            source_timestamp: None,
            received_at,
            source_payload,
        },
    )
    .map(Some)
}

fn parse_gamma_resolution(
    value: &Value,
    identity: &MarketIdentity,
    received_at: DateTime<Utc>,
    grace_seconds: u64,
) -> Result<Option<ResolutionFact>, StrategyError> {
    if !window_is_mature(identity.window_start, received_at, grace_seconds)? {
        return Ok(None);
    }
    let event = value.as_object().ok_or_else(|| {
        source_error(
            "polymarket_resolution_gamma_invalid_event",
            "Gamma resolution response must be an object",
        )
    })?;
    let markets = event
        .get("markets")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            source_error(
                "polymarket_resolution_gamma_missing_market",
                "Gamma resolution response is missing its markets array",
            )
        })?;
    if markets.len() != 1 {
        return Err(source_error(
            "polymarket_resolution_gamma_ambiguous_market",
            "Gamma BTC five-minute event must contain exactly one market",
        ));
    }
    let market = markets[0].as_object().ok_or_else(|| {
        source_error(
            "polymarket_resolution_gamma_invalid_market",
            "Gamma resolution market must be an object",
        )
    })?;
    let event_closed = bool_field(event, &["closed"]);
    let market_closed = bool_field(market, &["closed"]);
    let accepting_orders = bool_field(market, &["acceptingOrders", "accepting_orders"]);
    let resolution_status = string_field(
        market,
        &[
            "umaResolutionStatus",
            "uma_resolution_status",
            "resolutionStatus",
        ],
    );
    let looks_terminal = event_closed == Some(true)
        || market_closed == Some(true)
        || resolution_status
            .as_deref()
            .is_some_and(|status| status.eq_ignore_ascii_case("resolved"));
    if !looks_terminal {
        return Ok(None);
    }
    if event_closed != Some(true)
        || market_closed != Some(true)
        || accepting_orders != Some(false)
        || !resolution_status
            .as_deref()
            .is_some_and(|status| status.eq_ignore_ascii_case("resolved"))
    {
        return Err(source_error(
            "polymarket_resolution_gamma_terminal_incoherent",
            "Gamma terminal lifecycle fields are not coherently closed, resolved, and non-accepting",
        ));
    }
    let event_closed_at = required_datetime(event, &["closedTime", "closed_time"], "Gamma event")?;
    let market_closed_at =
        required_datetime(market, &["closedTime", "closed_time"], "Gamma market")?;
    let resolved_at = required_datetime(
        market,
        &["umaEndDate", "uma_end_date", "resolvedAt"],
        "Gamma market",
    )?;
    if event_closed_at != market_closed_at || market_closed_at != resolved_at {
        return Err(source_error(
            "polymarket_resolution_gamma_timestamp_incoherent",
            "Gamma event close, market close, and resolution timestamps disagree",
        ));
    }
    if event_closed_at < identity.window_end {
        return Err(source_error(
            "polymarket_resolution_gamma_timestamp_incoherent",
            "Gamma terminal timestamp predates the market window end",
        ));
    }
    let outcomes = required_string_array(market, &["outcomes"], "Gamma market")?;
    let token_ids = required_string_array(
        market,
        &["clobTokenIds", "clob_token_ids", "tokenIds", "token_ids"],
        "Gamma market",
    )?;
    let prices =
        required_decimal_array(market, &["outcomePrices", "outcome_prices"], "Gamma market")?;
    if outcomes.len() != 2 || token_ids.len() != 2 || prices.len() != 2 {
        return Err(source_error(
            "polymarket_resolution_gamma_ambiguous_outcomes",
            "Gamma terminal market must contain exactly two outcomes, token IDs, and prices",
        ));
    }
    let mut projected = Vec::with_capacity(2);
    let mut winner = None;
    let mut seen = BTreeSet::new();
    for ((outcome, token_id), price) in outcomes.iter().zip(&token_ids).zip(&prices) {
        let outcome = parse_winning_outcome(outcome)?;
        validate_token_id(token_id, "Gamma token")?;
        if !seen.insert(token_id.clone()) {
            return Err(source_error(
                "polymarket_resolution_gamma_ambiguous_outcomes",
                "Gamma terminal market contains duplicate token IDs",
            ));
        }
        let expected = match outcome {
            WinningOutcome::Up => &identity.up_token_id,
            WinningOutcome::Down => &identity.down_token_id,
        };
        if token_id != expected {
            return Err(source_error(
                "polymarket_resolution_gamma_identity_mismatch",
                "Gamma terminal token/outcome mapping changed after discovery",
            ));
        }
        if *price != Decimal::ZERO && *price != Decimal::ONE {
            return Err(source_error(
                "polymarket_resolution_gamma_nonbinary_price",
                "Gamma terminal outcome prices must be exactly binary",
            ));
        }
        if *price == Decimal::ONE && winner.replace((token_id.clone(), outcome)).is_some() {
            return Err(source_error(
                "polymarket_resolution_gamma_ambiguous_winner",
                "Gamma terminal market has more than one unit-priced outcome",
            ));
        }
        projected.push(json!({
            "token_id": token_id,
            "outcome": outcome.as_str(),
            "price": price.normalize().to_string(),
        }));
    }
    let (winning_token_id, winning_outcome) = winner.ok_or_else(|| {
        source_error(
            "polymarket_resolution_gamma_missing_winner",
            "Gamma terminal market has no unit-priced outcome",
        )
    })?;
    projected.sort_by(|left, right| left["outcome"].as_str().cmp(&right["outcome"].as_str()));
    let source_payload = json!({
        "schema_version": 1,
        "source": ResolutionSource::GammaRestReconciliation.as_str(),
        "event": {
            "id": string_field(event, &["id"]),
            "slug": identity.event_slug,
            "closed": event_closed,
            "closed_time": event_closed_at,
        },
        "market": {
            "id": identity.market_id,
            "condition_id": identity.condition_id,
            "closed": market_closed,
            "accepting_orders": accepting_orders,
            "resolution_status": resolution_status.map(|status| status.to_ascii_lowercase()),
            "closed_time": market_closed_at,
            "resolved_at": resolved_at,
            "tokens": projected,
        }
    });
    ResolutionFact::new(
        identity.clone(),
        ResolutionEvidence {
            source: ResolutionSource::GammaRestReconciliation,
            winning_token_id,
            winning_outcome,
            source_timestamp: Some(resolved_at),
            received_at,
            source_payload,
        },
    )
    .map(Some)
}

#[derive(Debug, Clone, PartialEq)]
struct WireResolution {
    condition_id: String,
    winning_token_id: String,
    winning_outcome: WinningOutcome,
    source_timestamp: DateTime<Utc>,
    source_payload: Value,
}

fn parse_websocket_resolution_frame(bytes: &[u8]) -> Result<Vec<WireResolution>, StrategyError> {
    if bytes.len() > MAX_WEBSOCKET_FRAME_BYTES {
        return Err(source_error(
            "polymarket_resolution_websocket_frame_too_large",
            "Polymarket resolution websocket frame exceeded one mebibyte",
        ));
    }
    let value = serde_json::from_slice::<Value>(bytes).map_err(|error| {
        source_error(
            "polymarket_resolution_websocket_decode_failed",
            format!("failed to decode Polymarket resolution websocket frame: {error}"),
        )
    })?;
    let mut messages = Vec::new();
    let mut object_count = 0usize;
    parse_websocket_resolution_value(&value, &mut messages, &mut object_count)?;
    Ok(messages)
}

fn parse_websocket_resolution_value(
    value: &Value,
    messages: &mut Vec<WireResolution>,
    object_count: &mut usize,
) -> Result<(), StrategyError> {
    if let Some(values) = value.as_array() {
        if values.len() > MAX_MESSAGES_PER_FRAME.saturating_sub(*object_count) {
            return Err(source_error(
                "polymarket_resolution_websocket_batch_too_large",
                "Polymarket resolution websocket frame exceeded its bounded message count",
            ));
        }
        for value in values {
            parse_websocket_resolution_value(value, messages, object_count)?;
        }
        return Ok(());
    }
    let object = value.as_object().ok_or_else(|| {
        source_error(
            "polymarket_resolution_websocket_invalid_message",
            "Polymarket resolution websocket message must be an object or array",
        )
    })?;
    *object_count = (*object_count).saturating_add(1);
    if *object_count > MAX_MESSAGES_PER_FRAME {
        return Err(source_error(
            "polymarket_resolution_websocket_batch_too_large",
            "Polymarket resolution websocket frame exceeded its bounded message count",
        ));
    }
    let Some(event_type) = string_field(object, &["event_type"]) else {
        return Err(source_error(
            "polymarket_resolution_websocket_missing_event_type",
            "Polymarket websocket message is missing event_type",
        ));
    };
    if event_type != "market_resolved" {
        return Ok(());
    }
    let condition_id = required_string(object, &["market"], "CLOB market_resolved")?;
    validate_condition_id(&condition_id, "CLOB market_resolved")?;
    let winning_token_id = required_string(object, &["winning_asset_id"], "CLOB market_resolved")?;
    validate_token_id(&winning_token_id, "CLOB market_resolved")?;
    let winning_outcome = parse_winning_outcome(&required_string(
        object,
        &["winning_outcome"],
        "CLOB market_resolved",
    )?)?;
    let source_timestamp =
        required_millisecond_timestamp(object, &["timestamp"], "CLOB market_resolved")?;
    messages.push(WireResolution {
        condition_id: condition_id.clone(),
        winning_token_id: winning_token_id.clone(),
        winning_outcome,
        source_timestamp,
        source_payload: json!({
            "schema_version": 1,
            "source": ResolutionSource::ClobWebsocket.as_str(),
            "event_type": "market_resolved",
            "market": condition_id,
            "winning_asset_id": winning_token_id,
            "winning_outcome": winning_outcome.as_str(),
            "timestamp": source_timestamp.timestamp_millis().to_string(),
        }),
    });
    Ok(())
}

impl WireResolution {
    fn into_fact(
        self,
        identity: &MarketIdentity,
        received_at: DateTime<Utc>,
    ) -> Result<ResolutionFact, StrategyError> {
        if self.condition_id != identity.condition_id {
            return Err(integrity_error(
                "polymarket_resolution_websocket_identity_mismatch",
                "market_resolved condition ID differs from its subscription identity",
            ));
        }
        ResolutionFact::new(
            identity.clone(),
            ResolutionEvidence {
                source: ResolutionSource::ClobWebsocket,
                winning_token_id: self.winning_token_id,
                winning_outcome: self.winning_outcome,
                source_timestamp: Some(self.source_timestamp),
                received_at,
                source_payload: self.source_payload,
            },
        )
    }
}

fn start_websocket_producer(
    client: Client,
    config: PolymarketBtcFiveMinuteResolutionsConfig,
    shutdown: CancellationToken,
) -> (
    JoinHandle<Result<(), StrategyError>>,
    mpsc::Receiver<ProducerNotice>,
) {
    let (sender, receiver) = mpsc::channel(128);
    let handle =
        tokio::spawn(async move { run_websocket_producer(client, config, sender, shutdown).await });
    (handle, receiver)
}

async fn run_websocket_producer(
    client: Client,
    config: PolymarketBtcFiveMinuteResolutionsConfig,
    sender: mpsc::Sender<ProducerNotice>,
    shutdown: CancellationToken,
) -> Result<(), StrategyError> {
    let mut reconnect_delay = Duration::from_millis(config.reconnect_initial_ms);
    let maximum_reconnect_delay = Duration::from_millis(config.reconnect_max_ms);
    loop {
        if shutdown.is_cancelled() {
            return Ok(());
        }
        let started_at = Instant::now();
        match capture_websocket_session(&client, &config, &sender, &shutdown).await {
            Ok(()) if shutdown.is_cancelled() => return Ok(()),
            Ok(()) => {
                let error = source_error(
                    "polymarket_resolution_websocket_session_ended",
                    "Polymarket resolution websocket session ended without shutdown",
                );
                if sender.send(ProducerNotice::Error(error)).await.is_err() {
                    return Ok(());
                }
            }
            Err(error) if error.kind == StrategyErrorKind::Shutdown => return Ok(()),
            Err(error)
                if matches!(
                    error.kind,
                    StrategyErrorKind::TransientSource | StrategyErrorKind::TransientDatabase
                ) =>
            {
                if sender.send(ProducerNotice::Error(error)).await.is_err() {
                    return Ok(());
                }
            }
            Err(error) => return Err(error),
        }
        if started_at.elapsed() >= Duration::from_secs(60) {
            reconnect_delay = Duration::from_millis(config.reconnect_initial_ms);
        }
        tokio::select! {
            _ = shutdown.cancelled() => return Ok(()),
            _ = tokio::time::sleep(reconnect_delay) => {}
        }
        reconnect_delay = reconnect_delay
            .checked_mul(2)
            .unwrap_or(maximum_reconnect_delay)
            .min(maximum_reconnect_delay);
    }
}

async fn capture_websocket_session(
    client: &Client,
    config: &PolymarketBtcFiveMinuteResolutionsConfig,
    sender: &mpsc::Sender<ProducerNotice>,
    shutdown: &CancellationToken,
) -> Result<(), StrategyError> {
    let mut identities =
        discover_subscription_identities(client, config, Utc::now(), shutdown).await?;
    if identities.is_empty() {
        return Err(source_error(
            "polymarket_resolution_no_subscriptions",
            "Gamma exposed no BTC five-minute markets for the bounded websocket subscription set",
        ));
    }
    let websocket_config = WebSocketConfig::default()
        .read_buffer_size(64 * 1024)
        .write_buffer_size(16 * 1024)
        .max_write_buffer_size(64 * 1024)
        .max_message_size(Some(MAX_WEBSOCKET_FRAME_BYTES))
        .max_frame_size(Some(MAX_WEBSOCKET_FRAME_BYTES));
    let websocket = tokio::time::timeout(
        Duration::from_millis(config.connect_timeout_ms),
        connect_async_with_config(&config.websocket_url, Some(websocket_config), true),
    )
    .await
    .map_err(|_| {
        source_error(
            "polymarket_resolution_websocket_connect_timeout",
            "timed out connecting to the Polymarket resolution websocket",
        )
    })?
    .map_err(|error| {
        source_error(
            "polymarket_resolution_websocket_connect_failed",
            format!("failed to connect to the Polymarket resolution websocket: {error}"),
        )
    })?
    .0;
    let (mut sink, mut stream) = websocket.split();
    let mut active_assets = identity_assets(&identities);
    send_websocket_message(
        &mut sink,
        Message::Text(subscription_message(&active_assets).into()),
        Duration::from_millis(config.connect_timeout_ms),
    )
    .await?;

    let refresh_interval = Duration::from_secs(config.discovery_refresh_seconds);
    let mut refresh = tokio::time::interval_at(Instant::now() + refresh_interval, refresh_interval);
    refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let ping_interval = Duration::from_millis(config.ping_interval_ms);
    let mut ping = tokio::time::interval_at(Instant::now() + ping_interval, ping_interval);
    ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let read_timeout = Duration::from_millis(config.read_timeout_ms);
    let mut read_deadline = Instant::now() + read_timeout;
    let mut pong_deadline = None;

    loop {
        let disabled_deadline = Instant::now() + Duration::from_secs(86_400);
        let pong_sleep = tokio::time::sleep_until(pong_deadline.unwrap_or(disabled_deadline));
        tokio::pin!(pong_sleep);
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => return Ok(()),
            _ = &mut pong_sleep, if pong_deadline.is_some() => {
                return Err(source_error(
                    "polymarket_resolution_websocket_pong_timeout",
                    "Polymarket CLOB did not acknowledge the oldest text PING",
                ));
            }
            _ = tokio::time::sleep_until(read_deadline) => {
                return Err(source_error(
                    "polymarket_resolution_websocket_read_timeout",
                    "Polymarket resolution websocket produced no frames before its read-idle deadline",
                ));
            }
            _ = refresh.tick() => {
                match discover_subscription_identities(client, config, Utc::now(), shutdown).await {
                    Ok(desired) if !desired.is_empty() => {
                        let desired_assets = identity_assets(&desired);
                        let added = desired_assets
                            .difference(&active_assets)
                            .cloned()
                            .collect::<Vec<_>>();
                        let removed = active_assets
                            .difference(&desired_assets)
                            .cloned()
                            .collect::<Vec<_>>();
                        if !added.is_empty() {
                            send_websocket_message(
                                &mut sink,
                                Message::Text(subscription_operation(&added, true).into()),
                                Duration::from_millis(config.connect_timeout_ms),
                            ).await?;
                        }
                        for (condition_id, identity) in &desired {
                            identities.insert(condition_id.clone(), identity.clone());
                        }
                        let identity_cutoff = aligned_window_start(Utc::now())
                            - chrono::Duration::seconds(INTERVAL_SECONDS.saturating_mul(3));
                        identities.retain(|_, identity| identity.window_start >= identity_cutoff);
                        if !removed.is_empty() {
                            send_websocket_message(
                                &mut sink,
                                Message::Text(subscription_operation(&removed, false).into()),
                                Duration::from_millis(config.connect_timeout_ms),
                            ).await?;
                        }
                        active_assets = desired_assets;
                    }
                    Ok(_) => {
                        warn!(
                            strategy = %STRATEGY_KEY,
                            "Gamma refresh returned no subscription identities; retaining last verified set"
                        );
                    }
                    Err(error) if error.kind == StrategyErrorKind::Shutdown => return Ok(()),
                    Err(error) => {
                        warn!(
                            strategy = %STRATEGY_KEY,
                            error_code = error.code,
                            error = %error,
                            "bounded Gamma refresh failed; retaining last verified resolution subscription"
                        );
                    }
                }
            }
            _ = ping.tick() => {
                let sent_at = Instant::now();
                send_websocket_message(
                    &mut sink,
                    Message::Text("PING".into()),
                    Duration::from_millis(config.connect_timeout_ms),
                ).await?;
                pong_deadline.get_or_insert(
                    sent_at + Duration::from_millis(config.pong_timeout_ms)
                );
            }
            frame = stream.next() => {
                read_deadline = Instant::now() + read_timeout;
                let received_at = microsecond_timestamp(Utc::now());
                let bytes = match frame {
                    Some(Ok(Message::Text(text))) => {
                        let text = text.as_str().trim();
                        if acknowledge_text_pong(text, &mut pong_deadline) {
                            continue;
                        }
                        if text.eq_ignore_ascii_case("PING") {
                            send_websocket_message(
                                &mut sink,
                                Message::Text("PONG".into()),
                                Duration::from_millis(config.connect_timeout_ms),
                            ).await?;
                            continue;
                        }
                        if text.is_empty() {
                            continue;
                        }
                        text.as_bytes().to_vec()
                    }
                    Some(Ok(Message::Binary(bytes))) => bytes.to_vec(),
                    Some(Ok(Message::Ping(payload))) => {
                        send_websocket_message(
                            &mut sink,
                            Message::Pong(payload),
                            Duration::from_millis(config.connect_timeout_ms),
                        ).await?;
                        continue;
                    }
                    Some(Ok(Message::Pong(_))) => continue,
                    Some(Ok(Message::Close(frame))) => {
                        return Err(source_error(
                            "polymarket_resolution_websocket_closed",
                            format!("Polymarket resolution websocket closed: {frame:?}"),
                        ));
                    }
                    Some(Ok(_)) => continue,
                    Some(Err(error)) => {
                        return Err(source_error(
                            "polymarket_resolution_websocket_read_failed",
                            format!("failed reading the Polymarket resolution websocket: {error}"),
                        ));
                    }
                    None => {
                        return Err(source_error(
                            "polymarket_resolution_websocket_eof",
                            "Polymarket resolution websocket ended",
                        ));
                    }
                };
                for wire in parse_websocket_resolution_frame(&bytes)? {
                    let identity = identities.get(&wire.condition_id).ok_or_else(|| {
                        source_error(
                            "polymarket_resolution_websocket_unknown_market",
                            format!(
                                "market_resolved named unsubscribed condition {}",
                                wire.condition_id
                            ),
                        )
                    })?;
                    let fact = wire.into_fact(identity, received_at)?;
                    if sender
                        .send(ProducerNotice::Fact(Box::new(fact)))
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                }
            }
        }
    }
}

async fn discover_subscription_identities(
    client: &Client,
    config: &PolymarketBtcFiveMinuteResolutionsConfig,
    now: DateTime<Utc>,
    shutdown: &CancellationToken,
) -> Result<BTreeMap<String, MarketIdentity>, StrategyError> {
    let current = aligned_window_start(now);
    let start = current
        .checked_sub_signed(chrono::Duration::seconds(INTERVAL_SECONDS))
        .unwrap_or_else(unix_epoch);
    let count = i64::from(config.lookahead_windows).saturating_add(2);
    let windows = (0..count)
        .map(|index| {
            start
                .checked_add_signed(chrono::Duration::seconds(
                    index.saturating_mul(INTERVAL_SECONDS),
                ))
                .ok_or_else(|| {
                    integrity_error(
                        "polymarket_resolution_subscription_window_overflow",
                        "websocket subscription window overflowed",
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let results = stream::iter(windows.into_iter().map(|window_start| async move {
        fetch_gamma_identity(client, config, window_start, shutdown).await
    }))
    .buffer_unordered(MAX_PARALLEL_REQUESTS)
    .collect::<Vec<_>>()
    .await;
    let mut identities = BTreeMap::new();
    let mut first_error = None;
    for result in results {
        let identity = match result {
            Ok(Some((_, _, identity))) => identity,
            Ok(None) => continue,
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(error);
                }
                continue;
            }
        };
        if let Some(existing) = identities.insert(identity.condition_id.clone(), identity.clone()) {
            if existing != identity {
                return Err(integrity_error(
                    "polymarket_resolution_subscription_identity_collision",
                    "Gamma reused a CLOB condition ID across different five-minute markets",
                ));
            }
        }
    }
    if identities.is_empty() {
        if let Some(error) = first_error {
            return Err(error);
        }
    } else if let Some(error) = first_error {
        warn!(
            strategy = %STRATEGY_KEY,
            error_code = error.code,
            error = %error,
            "one bounded Gamma subscription window failed; retaining independently verified markets"
        );
    }
    validate_identity_namespace(identities.values().collect::<Vec<_>>().as_slice())?;
    Ok(identities)
}

fn identity_assets(identities: &BTreeMap<String, MarketIdentity>) -> BTreeSet<String> {
    identities
        .values()
        .flat_map(|identity| [identity.up_token_id.clone(), identity.down_token_id.clone()])
        .collect()
}

fn subscription_message(assets: &BTreeSet<String>) -> String {
    json!({
        "assets_ids": assets.iter().collect::<Vec<_>>(),
        "type": "market",
        "custom_feature_enabled": true,
        "initial_dump": true,
    })
    .to_string()
}

fn subscription_operation(assets: &[String], subscribe: bool) -> String {
    if subscribe {
        json!({
            "assets_ids": assets,
            "operation": "subscribe",
            "custom_feature_enabled": true,
            "initial_dump": true,
        })
    } else {
        json!({
            "assets_ids": assets,
            "operation": "unsubscribe",
        })
    }
    .to_string()
}

async fn send_websocket_message<S>(
    sink: &mut S,
    message: Message,
    timeout: Duration,
) -> Result<(), StrategyError>
where
    S: Sink<Message> + Unpin,
    S::Error: std::fmt::Display,
{
    tokio::time::timeout(timeout, sink.send(message))
        .await
        .map_err(|_| {
            source_error(
                "polymarket_resolution_websocket_write_timeout",
                "timed out writing to the Polymarket resolution websocket",
            )
        })?
        .map_err(|error| {
            source_error(
                "polymarket_resolution_websocket_write_failed",
                format!("failed writing to the Polymarket resolution websocket: {error}"),
            )
        })
}

fn acknowledge_text_pong(text: &str, deadline: &mut Option<Instant>) -> bool {
    if text.trim().eq_ignore_ascii_case("PONG") {
        deadline.take();
        true
    } else {
        false
    }
}

fn build_http_client(
    config: &PolymarketBtcFiveMinuteResolutionsConfig,
) -> Result<Client, StrategyFactoryError> {
    Client::builder()
        .connect_timeout(Duration::from_millis(config.connect_timeout_ms))
        .timeout(Duration::from_secs(config.request_timeout_seconds))
        .redirect(reqwest::redirect::Policy::none())
        .user_agent("capitonic-market-data-ingester/0.1")
        .build()
        .map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "failed to build Polymarket resolution HTTP client: {error}"
            ))
        })
}

async fn read_bounded_body(
    response: Response,
    maximum_bytes: usize,
    source: &str,
    shutdown: &CancellationToken,
) -> Result<Vec<u8>, StrategyError> {
    if response
        .content_length()
        .is_some_and(|length| length > maximum_bytes as u64)
    {
        return Err(source_error(
            "polymarket_resolution_http_body_too_large",
            format!("{source} response exceeded {maximum_bytes} bytes"),
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
                "polymarket_resolution_http_body_failed",
                format!("failed reading {source} response body: {error}"),
            )
        })?;
        if body.len().saturating_add(chunk.len()) > maximum_bytes {
            return Err(source_error(
                "polymarket_resolution_http_body_too_large",
                format!("{source} response exceeded {maximum_bytes} bytes"),
            ));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn fact_primary_key(fact: &ResolutionFact) -> (String, String, String) {
    (
        fact.identity.market_id.clone(),
        fact.source.as_str().to_owned(),
        fact.payload_sha256.clone(),
    )
}

fn validate_candidate_identities(facts: &[ResolutionFact]) -> Result<(), StrategyError> {
    for (index, left) in facts.iter().enumerate() {
        for right in &facts[index.saturating_add(1)..] {
            if identity_namespace_collides(&left.identity, &right.identity)
                && left.identity != right.identity
            {
                return Err(integrity_error(
                    "polymarket_resolution_batch_identity_collision",
                    format!(
                        "provider batch reused immutable identity between markets {} and {}",
                        left.identity.market_id, right.identity.market_id
                    ),
                ));
            }
            if fact_primary_key(left) == fact_primary_key(right)
                && !facts_replay_equivalent(left, right)
            {
                return Err(integrity_error(
                    "polymarket_resolution_batch_replay_conflict",
                    "same resolution evidence key appeared with different factual fields",
                ));
            }
        }
    }
    Ok(())
}

fn facts_replay_equivalent(left: &ResolutionFact, right: &ResolutionFact) -> bool {
    left.source == right.source
        && left.identity == right.identity
        && left.winning_token_id == right.winning_token_id
        && left.winning_outcome == right.winning_outcome
        && left.source_timestamp == right.source_timestamp
        && left.provider_available_at == right.provider_available_at
        && left.source_payload == right.source_payload
        && left.revision_sha256 == right.revision_sha256
        && left.payload_sha256 == right.payload_sha256
}

fn validate_stored_identities(
    facts: &[ResolutionFact],
    stored: &[StoredResolution],
) -> Result<(), StrategyError> {
    for fact in facts {
        for existing in stored {
            let stored_identity = MarketIdentity {
                market_id: existing.market_id.clone(),
                condition_id: existing.condition_id.clone(),
                event_slug: existing.event_slug.clone(),
                window_start: existing.window_start,
                window_end: existing.window_end,
                up_token_id: existing.up_token_id.clone(),
                down_token_id: existing.down_token_id.clone(),
            };
            if identity_namespace_collides(&fact.identity, &stored_identity)
                && !existing.identity_eq(fact)
            {
                return Err(integrity_error(
                    "polymarket_resolution_identity_changed",
                    format!(
                        "provider changed immutable identity namespace for market {}",
                        fact.identity.market_id
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn validate_identity_namespace(identities: &[&MarketIdentity]) -> Result<(), StrategyError> {
    for (index, left) in identities.iter().enumerate() {
        for right in &identities[index.saturating_add(1)..] {
            if identity_namespace_collides(left, right) && *left != *right {
                return Err(integrity_error(
                    "polymarket_resolution_subscription_identity_collision",
                    "Gamma reused an immutable identifier across websocket subscription markets",
                ));
            }
        }
    }
    Ok(())
}

fn identity_namespace_collides(left: &MarketIdentity, right: &MarketIdentity) -> bool {
    left.market_id == right.market_id
        || left.condition_id == right.condition_id
        || left.event_slug == right.event_slug
        || left.up_token_id == right.up_token_id
        || left.up_token_id == right.down_token_id
        || left.down_token_id == right.up_token_id
        || left.down_token_id == right.down_token_id
}

fn unique_stored_facts(
    rows: Vec<StoredResolution>,
) -> Result<BTreeMap<(String, String, String), StoredResolution>, StrategyError> {
    let mut unique = BTreeMap::new();
    for row in rows {
        let key = (
            row.market_id.clone(),
            row.source.clone(),
            row.payload_sha256.clone(),
        );
        if unique.insert(key.clone(), row).is_some() {
            return Err(integrity_error(
                "polymarket_resolution_duplicate_primary_key",
                format!(
                    "database returned duplicate resolution evidence {}:{}:{}",
                    key.0, key.1, key.2
                ),
            ));
        }
    }
    Ok(unique)
}

fn immutable_fact_conflict(fact: &ResolutionFact) -> StrategyError {
    integrity_error(
        "polymarket_resolution_immutable_conflict",
        format!(
            "durable resolution evidence {}:{}:{} differs from the replayed provider projection",
            fact.identity.market_id,
            fact.source.as_str(),
            fact.payload_sha256
        ),
    )
}

fn resolution_artifact_batch(facts: &[&ResolutionFact]) -> ArtifactBatch {
    let minimum_source_timestamp = facts.iter().filter_map(|fact| fact.source_timestamp).min();
    let maximum_source_timestamp = facts.iter().filter_map(|fact| fact.source_timestamp).max();
    let minimum_received_at = facts.iter().map(|fact| fact.received_at).min();
    let maximum_received_at = facts.iter().map(|fact| fact.received_at).max();
    let start_cursor = facts
        .iter()
        .min_by_key(|fact| fact.identity.window_start)
        .map(|fact| fact.cursor());
    let end_cursor = facts
        .iter()
        .max_by_key(|fact| fact.identity.window_start)
        .map(|fact| fact.cursor());
    ArtifactBatch {
        inserted_record_count: facts.len() as i64,
        minimum_source_timestamp,
        maximum_source_timestamp,
        minimum_received_at,
        maximum_received_at,
        start_cursor,
        end_cursor,
    }
}

fn parse_gap_window(
    gap_id: Uuid,
    source_time_start: Option<DateTime<Utc>>,
    start_cursor: Option<&str>,
) -> Result<DateTime<Utc>, StrategyError> {
    let source_time_start = source_time_start.ok_or_else(|| {
        integrity_error(
            "polymarket_resolution_gap_window_missing",
            format!("resolution gap {gap_id} is missing source_time_start"),
        )
    })?;
    if !is_aligned_window(source_time_start) {
        return Err(integrity_error(
            "polymarket_resolution_gap_window_invalid",
            format!("resolution gap {gap_id} has an unaligned source window"),
        ));
    }
    let cursor = start_cursor.ok_or_else(|| {
        integrity_error(
            "polymarket_resolution_gap_cursor_missing",
            format!("resolution gap {gap_id} is missing its start cursor"),
        )
    })?;
    let cursor_seconds = cursor.parse::<i64>().map_err(|error| {
        integrity_error(
            "polymarket_resolution_gap_cursor_invalid",
            format!("resolution gap {gap_id} cursor is invalid: {error}"),
        )
    })?;
    if cursor_seconds != source_time_start.timestamp() {
        return Err(integrity_error(
            "polymarket_resolution_gap_cursor_mismatch",
            format!("resolution gap {gap_id} cursor does not match source_time_start"),
        ));
    }
    Ok(source_time_start)
}

fn validate_repair_gap(gap: &DataGap, window_start: DateTime<Utc>) -> Result<(), StrategyError> {
    let expected_cursor = window_start.timestamp().to_string();
    let expected_end = window_start + chrono::Duration::seconds(INTERVAL_SECONDS);
    if gap.strategy_key != STRATEGY_KEY
        || gap.reason_code != RESOLUTION_GAP_REASON
        || gap.source_time_start != Some(window_start)
        || gap.source_time_end != Some(expected_end)
        || gap.start_cursor.as_deref() != Some(expected_cursor.as_str())
        || gap.end_cursor.as_deref() != Some(expected_cursor.as_str())
    {
        return Err(integrity_error(
            "polymarket_resolution_gap_repair_mismatch",
            format!(
                "claimed resolution gap {} does not match recovered window {window_start}",
                gap.gap_id
            ),
        ));
    }
    Ok(())
}

fn required_string(
    object: &Map<String, Value>,
    keys: &[&str],
    context: &str,
) -> Result<String, StrategyError> {
    string_field(object, keys).ok_or_else(|| {
        source_error(
            "polymarket_resolution_missing_field",
            format!("{context} is missing required field {}", keys[0]),
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

fn bool_field(object: &Map<String, Value>, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .find_map(|key| object.get(*key).and_then(Value::as_bool))
}

fn required_bool(
    object: &Map<String, Value>,
    keys: &[&str],
    context: &str,
) -> Result<bool, StrategyError> {
    bool_field(object, keys).ok_or_else(|| {
        source_error(
            "polymarket_resolution_missing_field",
            format!("{context} is missing required boolean field {}", keys[0]),
        )
    })
}

fn required_decimal(
    object: &Map<String, Value>,
    keys: &[&str],
    context: &str,
) -> Result<Decimal, StrategyError> {
    keys.iter()
        .find_map(|key| match object.get(*key)? {
            Value::String(value) => Decimal::from_str(value).ok(),
            Value::Number(value) => Decimal::from_str(&value.to_string()).ok(),
            _ => None,
        })
        .ok_or_else(|| {
            source_error(
                "polymarket_resolution_missing_field",
                format!("{context} is missing valid decimal field {}", keys[0]),
            )
        })
}

fn required_string_array(
    object: &Map<String, Value>,
    keys: &[&str],
    context: &str,
) -> Result<Vec<String>, StrategyError> {
    keys.iter()
        .find_map(|key| match object.get(*key)? {
            Value::Array(values) => values
                .iter()
                .map(|value| value.as_str().map(ToOwned::to_owned))
                .collect(),
            Value::String(value) => serde_json::from_str::<Vec<String>>(value).ok(),
            _ => None,
        })
        .ok_or_else(|| {
            source_error(
                "polymarket_resolution_missing_field",
                format!("{context} is missing valid string array {}", keys[0]),
            )
        })
}

fn required_decimal_array(
    object: &Map<String, Value>,
    keys: &[&str],
    context: &str,
) -> Result<Vec<Decimal>, StrategyError> {
    let raw = keys
        .iter()
        .find_map(|key| match object.get(*key)? {
            Value::Array(values) => Some(values.clone()),
            Value::String(value) => serde_json::from_str::<Vec<Value>>(value).ok(),
            _ => None,
        })
        .ok_or_else(|| {
            source_error(
                "polymarket_resolution_missing_field",
                format!("{context} is missing valid decimal array {}", keys[0]),
            )
        })?;
    raw.into_iter()
        .map(|value| {
            let parsed = match value {
                Value::String(value) => Decimal::from_str(&value).ok(),
                Value::Number(value) => Decimal::from_str(&value.to_string()).ok(),
                _ => None,
            };
            parsed.ok_or_else(|| {
                source_error(
                    "polymarket_resolution_invalid_decimal_array",
                    format!("{context} decimal array contains a non-decimal value"),
                )
            })
        })
        .collect()
}

fn required_datetime(
    object: &Map<String, Value>,
    keys: &[&str],
    context: &str,
) -> Result<DateTime<Utc>, StrategyError> {
    keys.iter()
        .find_map(|key| {
            DateTime::parse_from_rfc3339(object.get(*key)?.as_str()?)
                .ok()
                .map(|timestamp| microsecond_timestamp(timestamp.with_timezone(&Utc)))
        })
        .ok_or_else(|| {
            source_error(
                "polymarket_resolution_missing_timestamp",
                format!("{context} is missing valid timestamp {}", keys[0]),
            )
        })
}

fn required_millisecond_timestamp(
    object: &Map<String, Value>,
    keys: &[&str],
    context: &str,
) -> Result<DateTime<Utc>, StrategyError> {
    let value = required_string(object, keys, context)?;
    let milliseconds = value.parse::<i64>().map_err(|error| {
        source_error(
            "polymarket_resolution_invalid_timestamp",
            format!("{context} timestamp is not integer milliseconds: {error}"),
        )
    })?;
    if milliseconds < 1_000_000_000_000 {
        return Err(source_error(
            "polymarket_resolution_invalid_timestamp",
            format!("{context} timestamp must be Unix milliseconds"),
        ));
    }
    DateTime::from_timestamp_millis(milliseconds)
        .map(microsecond_timestamp)
        .ok_or_else(|| {
            source_error(
                "polymarket_resolution_invalid_timestamp",
                format!("{context} timestamp is outside the supported range"),
            )
        })
}

fn parse_winning_outcome(value: &str) -> Result<WinningOutcome, StrategyError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "up" => Ok(WinningOutcome::Up),
        "down" => Ok(WinningOutcome::Down),
        other => Err(source_error(
            "polymarket_resolution_unexpected_outcome",
            format!("unexpected BTC five-minute outcome {other}"),
        )),
    }
}

fn validate_condition_id(value: &str, context: &str) -> Result<(), StrategyError> {
    let valid = value.len() == 66
        && value.starts_with("0x")
        && value[2..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !valid {
        return Err(source_error(
            "polymarket_resolution_invalid_condition_id",
            format!("{context} condition ID must be lowercase 0x-prefixed hexadecimal"),
        ));
    }
    Ok(())
}

fn validate_token_id(value: &str, context: &str) -> Result<(), StrategyError> {
    if value.is_empty() || value.len() > 100 || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(source_error(
            "polymarket_resolution_invalid_token_id",
            format!("{context} token ID must contain between 1 and 100 decimal digits"),
        ));
    }
    Ok(())
}

fn validate_exact_https_origin(
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

fn validate_exact_websocket_url(value: &str) -> Result<(), StrategyFactoryError> {
    let parsed = Url::parse(value)
        .map_err(|error| invalid_config(format!("websocket_url is invalid: {error}")))?;
    if value != DEFAULT_WEBSOCKET_URL
        || parsed.scheme() != "wss"
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.port().is_some()
        || parsed.path() != "/ws/market"
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(invalid_config(
            "websocket_url must be the approved exact public Polymarket CLOB market endpoint",
        ));
    }
    Ok(())
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, StrategyError> {
    serde_json::to_vec(value).map_err(|error| {
        integrity_error(
            "polymarket_resolution_json_serialization",
            format!("failed to serialize canonical resolution JSON: {error}"),
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
            "polymarket_resolution_artifact_generation_mismatch",
            format!("artifact generation {actual} is newer than profile generation {expected}"),
        ));
    }
    Ok(())
}

fn is_aligned_window(timestamp: DateTime<Utc>) -> bool {
    timestamp.timestamp() >= 0
        && timestamp.timestamp_subsec_nanos() == 0
        && timestamp.timestamp().rem_euclid(INTERVAL_SECONDS) == 0
}

fn unix_epoch() -> DateTime<Utc> {
    Utc.timestamp_opt(0, 0).single().expect("Unix epoch exists")
}

fn microsecond_timestamp(timestamp: DateTime<Utc>) -> DateTime<Utc> {
    DateTime::from_timestamp_micros(timestamp.timestamp_micros())
        .expect("a valid timestamp remains valid at microsecond precision")
}

async fn await_producer(
    producer: JoinHandle<Result<(), StrategyError>>,
) -> Result<(), StrategyError> {
    producer.await.map_err(|error| {
        integrity_error(
            "polymarket_resolution_websocket_worker_join_failed",
            format!("Polymarket resolution websocket worker failed: {error}"),
        )
    })?
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
        "polymarket_resolution_lease_lost",
        "Polymarket resolution profile lease is no longer current",
    )
}

fn shutdown_error() -> StrategyError {
    StrategyError::new(
        StrategyErrorKind::Shutdown,
        "polymarket_resolution_shutdown",
        "Polymarket resolution strategy is shutting down",
    )
}

#[cfg(test)]
fn retry_delay_seconds(initial: u64, maximum: u64, repair_attempts: i32) -> u64 {
    let exponent = u32::try_from(repair_attempts.clamp(0, 10)).unwrap_or_default();
    initial
        .saturating_mul(2_u64.saturating_pow(exponent))
        .min(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gamma_fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/polymarket/gamma_btc_five_minute_resolved_event.json"
        ))
        .expect("Gamma resolution fixture is valid JSON")
    }

    fn clob_fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/polymarket/clob_rest_resolved_market.json"
        ))
        .expect("CLOB resolution fixture is valid JSON")
    }

    fn websocket_fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/polymarket/clob_websocket_market_resolved.json"
        ))
        .expect("CLOB websocket fixture is valid JSON")
    }

    fn window_start() -> DateTime<Utc> {
        Utc.timestamp_opt(1_783_902_600, 0).single().unwrap()
    }

    fn received_at() -> DateTime<Utc> {
        Utc.timestamp_opt(1_783_903_080, 0).single().unwrap()
    }

    fn identity() -> MarketIdentity {
        let contract = parse_gamma_contract(&gamma_fixture(), window_start(), received_at())
            .expect("resolved Gamma fixture remains a valid market contract");
        MarketIdentity::from(&contract)
    }

    #[test]
    fn clob_rest_maps_exact_binary_winner_without_source_timestamp() {
        let fact = parse_clob_rest_resolution(&clob_fixture(), &identity(), received_at())
            .unwrap()
            .unwrap();
        assert_eq!(fact.source, ResolutionSource::ClobRestReconciliation);
        assert_eq!(fact.winning_outcome, WinningOutcome::Up);
        assert_eq!(fact.winning_token_id, fact.identity.up_token_id);
        assert_eq!(fact.source_timestamp, None);
        assert_eq!(fact.provider_available_at, None);
        assert!(fact.source_payload.get("volume").is_none());
    }

    #[test]
    fn clob_projection_ignores_irrelevant_live_fields() {
        let mut changed = clob_fixture();
        changed["volume"] = json!("999999999");
        changed["liquidity"] = json!("1");
        changed["best_bid"] = json!("0.99");
        let first = parse_clob_rest_resolution(&clob_fixture(), &identity(), received_at())
            .unwrap()
            .unwrap();
        let second = parse_clob_rest_resolution(&changed, &identity(), received_at())
            .unwrap()
            .unwrap();
        assert_eq!(first.source_payload, second.source_payload);
        assert_eq!(first.payload_sha256, second.payload_sha256);
        assert_eq!(first.revision_sha256, second.revision_sha256);
    }

    #[test]
    fn clob_rest_rejects_ambiguous_or_nonbinary_terminal_tokens() {
        let mut two_winners = clob_fixture();
        two_winners["tokens"][1]["winner"] = json!(true);
        two_winners["tokens"][1]["price"] = json!(1);
        assert!(parse_clob_rest_resolution(&two_winners, &identity(), received_at()).is_err());

        let mut fractional = clob_fixture();
        fractional["tokens"][0]["price"] = json!("0.9");
        assert!(parse_clob_rest_resolution(&fractional, &identity(), received_at()).is_err());

        let mut unresolved = clob_fixture();
        unresolved["closed"] = json!(false);
        unresolved["accepting_orders"] = json!(true);
        assert!(
            parse_clob_rest_resolution(&unresolved, &identity(), received_at())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn gamma_fallback_requires_coherent_terminal_lifecycle_and_timestamps() {
        let fact = parse_gamma_resolution(&gamma_fixture(), &identity(), received_at(), 120)
            .unwrap()
            .unwrap();
        assert_eq!(fact.source, ResolutionSource::GammaRestReconciliation);
        assert_eq!(fact.winning_outcome, WinningOutcome::Up);
        assert_eq!(fact.source_timestamp, fact.provider_available_at);
        assert_eq!(
            fact.source_timestamp.unwrap(),
            Utc.timestamp_opt(1_783_902_965, 0).single().unwrap()
        );

        let mut incoherent = gamma_fixture();
        incoherent["markets"][0]["closedTime"] = json!("2026-07-13T00:36:06Z");
        assert!(parse_gamma_resolution(&incoherent, &identity(), received_at(), 120).is_err());

        let too_early = Utc.timestamp_opt(1_783_902_990, 0).single().unwrap();
        assert!(
            parse_gamma_resolution(&gamma_fixture(), &identity(), too_early, 120)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn genuine_winner_revisions_are_preserved_as_distinct_evidence() {
        let up = parse_gamma_resolution(&gamma_fixture(), &identity(), received_at(), 120)
            .unwrap()
            .unwrap();
        let mut revised = gamma_fixture();
        revised["markets"][0]["outcomePrices"] = json!("[\"0\",\"1\"]");
        let down = parse_gamma_resolution(&revised, &identity(), received_at(), 120)
            .unwrap()
            .unwrap();
        assert_eq!(up.winning_outcome, WinningOutcome::Up);
        assert_eq!(down.winning_outcome, WinningOutcome::Down);
        assert_ne!(up.revision_sha256, down.revision_sha256);
        assert_ne!(up.payload_sha256, down.payload_sha256);
        validate_candidate_identities(&[up, down]).unwrap();
    }

    #[test]
    fn semantic_revision_agrees_across_sources_for_the_same_winner() {
        let clob = parse_clob_rest_resolution(&clob_fixture(), &identity(), received_at())
            .unwrap()
            .unwrap();
        let gamma = parse_gamma_resolution(&gamma_fixture(), &identity(), received_at(), 120)
            .unwrap()
            .unwrap();
        assert_eq!(clob.revision_sha256, gamma.revision_sha256);
        assert_ne!(clob.payload_sha256, gamma.payload_sha256);
    }

    #[test]
    fn websocket_parser_accepts_object_and_array_and_maps_source_time() {
        let encoded = serde_json::to_vec(&websocket_fixture()).unwrap();
        let object = parse_websocket_resolution_frame(&encoded).unwrap();
        assert_eq!(object.len(), 1);
        let array = serde_json::to_vec(&json!([websocket_fixture(), websocket_fixture()])).unwrap();
        assert_eq!(parse_websocket_resolution_frame(&array).unwrap().len(), 2);
        let fact = object[0]
            .clone()
            .into_fact(&identity(), received_at())
            .unwrap();
        assert_eq!(fact.source, ResolutionSource::ClobWebsocket);
        assert_eq!(fact.source_timestamp, fact.provider_available_at);
        assert_eq!(fact.winning_outcome, WinningOutcome::Up);
    }

    #[test]
    fn websocket_parser_ignores_nonresolution_market_events_but_bounds_batches() {
        let book = json!({
            "event_type": "book",
            "market": identity().condition_id,
            "timestamp": "1783902965000",
            "bids": [],
            "asks": []
        });
        assert!(
            parse_websocket_resolution_frame(&serde_json::to_vec(&book).unwrap())
                .unwrap()
                .is_empty()
        );
        let oversized = Value::Array(
            (0..=MAX_MESSAGES_PER_FRAME)
                .map(|_| websocket_fixture())
                .collect(),
        );
        assert!(
            parse_websocket_resolution_frame(&serde_json::to_vec(&oversized).unwrap()).is_err()
        );
    }

    #[test]
    fn config_is_exact_schema_v1_with_no_credential_or_label_surface() {
        let value =
            serde_json::to_value(PolymarketBtcFiveMinuteResolutionsConfig::default()).unwrap();
        assert!(PolymarketBtcFiveMinuteResolutionsConfig::from_value(&value).is_ok());

        let mut extra = value.clone();
        extra["labels"] = json!(true);
        assert!(PolymarketBtcFiveMinuteResolutionsConfig::from_value(&extra).is_err());

        let mut wrong_rest = value.clone();
        wrong_rest["clob_market_endpoint"] = json!("clob-markets");
        assert!(PolymarketBtcFiveMinuteResolutionsConfig::from_value(&wrong_rest).is_err());

        let mut redirected = value;
        redirected["gamma_base_url"] = json!("https://gamma-api.polymarket.com/redirect");
        assert!(PolymarketBtcFiveMinuteResolutionsConfig::from_value(&redirected).is_err());
    }

    #[test]
    fn planner_bounds_stale_catchup_and_always_includes_realtime_tail() {
        let now = Utc.timestamp_opt(1_783_903_377, 0).single().unwrap();
        let current = aligned_window_start(now);
        let stale = current - chrono::Duration::days(2);
        let plan = resolution_window_plan(now, Some(stale), 288, 1, 120).unwrap();
        assert_eq!(plan.scan_windows.len(), MAX_WINDOWS_PER_CYCLE as usize);
        assert_eq!(
            plan.scan_windows.first(),
            Some(&(stale + chrono::Duration::minutes(5)))
        );
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
    fn planner_accepts_durable_cursor_newer_than_a_reconfigured_maturity_cutoff() {
        let now = Utc.timestamp_opt(1_783_903_377, 0).single().unwrap();
        let current = aligned_window_start(now);
        let durable_cursor = current - chrono::Duration::minutes(5);
        let plan = resolution_window_plan(now, Some(durable_cursor), 288, 1, 900).unwrap();

        assert!(plan.scan_windows.is_empty());
        assert!(plan.windows.contains(&durable_cursor));
        assert!(plan.windows.contains(&current));
        assert!(plan
            .windows
            .contains(&(current + chrono::Duration::minutes(5))));

        let future = current + chrono::Duration::minutes(5);
        assert!(resolution_window_plan(now, Some(future), 288, 1, 900).is_err());
    }

    #[test]
    fn retry_backoff_is_bounded_and_monotonic() {
        assert_eq!(retry_delay_seconds(30, 300, 0), 30);
        assert_eq!(retry_delay_seconds(30, 300, 1), 60);
        assert_eq!(retry_delay_seconds(30, 300, 3), 240);
        assert_eq!(retry_delay_seconds(30, 300, 4), 300);
        assert_eq!(retry_delay_seconds(30, 300, i32::MAX), 300);
    }

    #[test]
    fn canonical_payload_hash_is_stable_across_object_key_order() {
        let first: Value = serde_json::from_str(r#"{"b":1,"a":{"d":2,"c":3}}"#).unwrap();
        let second: Value = serde_json::from_str(r#"{"a":{"c":3,"d":2},"b":1}"#).unwrap();
        assert_eq!(hash_json(&first).unwrap(), hash_json(&second).unwrap());
    }

    #[test]
    fn artifact_generation_allows_takeover_of_older_but_rejects_newer() {
        assert!(verify_artifact_generation(4, 5).is_ok());
        assert!(verify_artifact_generation(5, 5).is_ok());
        assert!(verify_artifact_generation(6, 5).is_err());
    }

    #[test]
    fn websocket_subscription_enables_resolution_feature() {
        let assets = BTreeSet::from(["1".to_owned(), "2".to_owned()]);
        let value: Value = serde_json::from_str(&subscription_message(&assets)).unwrap();
        assert_eq!(value["type"], "market");
        assert_eq!(value["custom_feature_enabled"], true);
        assert_eq!(value["initial_dump"], true);
    }
}
