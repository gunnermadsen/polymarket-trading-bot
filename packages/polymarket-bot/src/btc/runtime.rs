use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Write},
    future::Future,
    panic::AssertUnwindSafe,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration as StdDuration,
};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use futures_util::{stream, FutureExt, SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::{
    net::TcpStream,
    sync::{mpsc, watch, RwLock},
    task::JoinHandle,
    time::{interval, interval_at, sleep, sleep_until, timeout, Instant, MissedTickBehavior},
};
use tokio_tungstenite::{connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream};
use uuid::Uuid;

use super::{
    feeds::{
        parse_binance_agg_trade, parse_clob_messages, parse_rtds_reference_tick, BookRegistry,
        ClobMessage,
    },
    market::{
        discovery_windows, parse_clob_rest_official_resolution, parse_gamma_btc_interval_event,
        slug_for_window, ClobRestOfficialResolution,
    },
    repository::{
        BtcMarketLabel, BtcOfficialResolutionWatch, BtcRepository, FeedSession,
        PersistedOfficialResolution,
    },
    types::{
        BtcIntervalMarket, BtcOutcome, FeedIntegrityStatus, MarketFeedEventType,
        OrderbookCheckpoint, Readiness, RealtimeState, ReferencePriceSource, ReferencePriceTick,
    },
};

const BOUNDARY_LABEL_VERSION: &str = "chainlink_first_tick_at_or_after_boundary_v1";
const CRITICAL_WRITE_ATTEMPTS: usize = 3;
const CRITICAL_WRITE_INITIAL_BACKOFF: StdDuration = StdDuration::from_millis(25);
const RTDS_HEARTBEAT_MESSAGE: &str = "ping";
const CLOB_CONNECT_TIMEOUT: StdDuration = StdDuration::from_secs(5);
const CLOB_SEND_TIMEOUT: StdDuration = StdDuration::from_secs(5);
const CLOB_BOOTSTRAP_TIMEOUT: StdDuration = StdDuration::from_secs(5);
const CLOB_PONG_TIMEOUT: StdDuration = StdDuration::from_secs(10);
const CLOB_READ_IDLE_TIMEOUT: StdDuration = StdDuration::from_secs(40);
const REFERENCE_CONNECT_TIMEOUT: StdDuration = StdDuration::from_secs(5);
const REFERENCE_SEND_TIMEOUT: StdDuration = StdDuration::from_secs(5);
// Chainlink updates can have legitimate multi-second gaps; this bound avoids
// reconnect churn while still detecting an unavailable required source quickly.
const RTDS_REQUIRED_DATA_TIMEOUT: StdDuration = StdDuration::from_secs(10);
const BINANCE_REQUIRED_DATA_TIMEOUT: StdDuration = StdDuration::from_secs(5);
const REFERENCE_PONG_TIMEOUT: StdDuration = StdDuration::from_secs(10);
const REFERENCE_READ_IDLE_TIMEOUT: StdDuration = StdDuration::from_secs(40);
const REFERENCE_STABLE_RESET_AFTER: StdDuration = StdDuration::from_secs(30);
const REFERENCE_RETRY_JITTER_PERCENT: u64 = 20;

type ClobSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BtcRuntimeConfig {
    pub enabled: bool,
    pub gamma_base_url: String,
    pub clob_rest_base_url: String,
    pub clob_ws_url: String,
    pub rtds_ws_url: String,
    pub binance_ws_url: String,
    pub discovery_interval: StdDuration,
    pub clob_heartbeat_interval: StdDuration,
    pub rtds_heartbeat_interval: StdDuration,
    pub binance_heartbeat_interval: StdDuration,
    pub reconnect_initial_delay: StdDuration,
    pub reconnect_max_delay: StdDuration,
    pub checkpoint_interval: StdDuration,
    pub strategy_interval: StdDuration,
    pub max_book_age: StdDuration,
    pub max_reference_age: StdDuration,
    pub boundary_tick_max_delay: StdDuration,
    pub official_resolution_audit_grace: StdDuration,
    pub official_resolution_watch_retention: StdDuration,
    pub writer_capacity: usize,
}

impl Default for BtcRuntimeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            gamma_base_url: "https://gamma-api.polymarket.com".to_string(),
            clob_rest_base_url: "https://clob.polymarket.com".to_string(),
            clob_ws_url: "wss://ws-subscriptions-clob.polymarket.com/ws/market".to_string(),
            rtds_ws_url: "wss://ws-live-data.polymarket.com".to_string(),
            binance_ws_url: "wss://stream.binance.com:9443/ws/btcusdt@aggTrade".to_string(),
            discovery_interval: StdDuration::from_secs(5),
            clob_heartbeat_interval: StdDuration::from_secs(10),
            rtds_heartbeat_interval: StdDuration::from_secs(5),
            binance_heartbeat_interval: StdDuration::from_secs(20),
            reconnect_initial_delay: StdDuration::from_secs(1),
            reconnect_max_delay: StdDuration::from_secs(30),
            checkpoint_interval: StdDuration::from_secs(1),
            strategy_interval: StdDuration::from_millis(250),
            max_book_age: StdDuration::from_secs(2),
            max_reference_age: StdDuration::from_secs(2),
            boundary_tick_max_delay: StdDuration::from_secs(5),
            official_resolution_audit_grace: StdDuration::from_secs(120),
            official_resolution_watch_retention: StdDuration::from_secs(3_600),
            writer_capacity: 8_192,
        }
    }
}

impl BtcRuntimeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.writer_capacity == 0 {
            bail!("BTC realtime writer capacity must be positive");
        }
        for (name, duration) in [
            ("discovery_interval", self.discovery_interval),
            ("clob_heartbeat_interval", self.clob_heartbeat_interval),
            ("rtds_heartbeat_interval", self.rtds_heartbeat_interval),
            (
                "binance_heartbeat_interval",
                self.binance_heartbeat_interval,
            ),
            ("reconnect_initial_delay", self.reconnect_initial_delay),
            ("reconnect_max_delay", self.reconnect_max_delay),
            ("checkpoint_interval", self.checkpoint_interval),
            ("strategy_interval", self.strategy_interval),
            ("max_book_age", self.max_book_age),
            ("max_reference_age", self.max_reference_age),
            ("boundary_tick_max_delay", self.boundary_tick_max_delay),
            (
                "official_resolution_audit_grace",
                self.official_resolution_audit_grace,
            ),
            (
                "official_resolution_watch_retention",
                self.official_resolution_watch_retention,
            ),
        ] {
            if duration.is_zero() {
                bail!("BTC realtime {name} must be positive");
            }
        }
        if self.reconnect_initial_delay > self.reconnect_max_delay {
            bail!("BTC realtime reconnect initial delay must not exceed its maximum");
        }
        if !self.gamma_base_url.starts_with("http") {
            bail!("BTC realtime Gamma endpoint must be HTTP(S)");
        }
        if !self.clob_rest_base_url.starts_with("http") {
            bail!("BTC realtime CLOB REST endpoint must be HTTP(S)");
        }
        let minimum_resolution_retention =
            StdDuration::from_secs(600).saturating_add(self.official_resolution_audit_grace);
        if self.official_resolution_watch_retention < minimum_resolution_retention {
            bail!("BTC resolution watch retention must cover two rollovers plus audit grace");
        }
        for (name, endpoint) in [
            ("CLOB", &self.clob_ws_url),
            ("RTDS", &self.rtds_ws_url),
            ("Binance", &self.binance_ws_url),
        ] {
            if !endpoint.starts_with("ws://") && !endpoint.starts_with("wss://") {
                bail!("BTC realtime {name} endpoint must be a websocket URL");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceDisconnectReason {
    Shutdown,
    ConnectTimeout,
    ConnectFailed,
    SubscriptionSendTimeout,
    SubscriptionSendFailed,
    HeartbeatSendTimeout,
    HeartbeatSendFailed,
    HeartbeatAckTimeout,
    RequiredDataIdleTimeout,
    ReadIdleTimeout,
    WebsocketEof,
    RemoteClose,
    TransportReadFailed,
    CriticalBoundaryIntegrity,
    CriticalBoundaryPersistence,
    CriticalWriterQueue,
    UnknownDisconnect,
}

impl ReferenceDisconnectReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Shutdown => "shutdown",
            Self::ConnectTimeout => "connect_timeout",
            Self::ConnectFailed => "connect_failed",
            Self::SubscriptionSendTimeout => "subscription_send_timeout",
            Self::SubscriptionSendFailed => "subscription_send_failed",
            Self::HeartbeatSendTimeout => "heartbeat_send_timeout",
            Self::HeartbeatSendFailed => "heartbeat_send_failed",
            Self::HeartbeatAckTimeout => "heartbeat_ack_timeout",
            Self::RequiredDataIdleTimeout => "required_data_idle_timeout",
            Self::ReadIdleTimeout => "read_idle_timeout",
            Self::WebsocketEof => "websocket_eof",
            Self::RemoteClose => "remote_close",
            Self::TransportReadFailed => "transport_read_failed",
            Self::CriticalBoundaryIntegrity => "critical_boundary_integrity",
            Self::CriticalBoundaryPersistence => "critical_boundary_persistence",
            Self::CriticalWriterQueue => "critical_writer_queue",
            Self::UnknownDisconnect => "unknown_disconnect",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReferenceTransportMetrics {
    pub connections_established: u64,
    pub healthy_connections: u64,
    pub connection_failures: u64,
    pub subscription_failures: u64,
    pub transport_disconnects: u64,
    pub watchdog_disconnects: u64,
    pub required_data_timeouts: u64,
    pub pong_timeouts: u64,
    pub read_timeouts: u64,
    pub immediate_recoveries_scheduled: u64,
    pub backoff_scheduled_milliseconds: u64,
    pub recovery_unavailable_milliseconds: u64,
    pub consecutive_failures: u32,
    pub connected_connection_epoch: Option<i32>,
    pub connected_connection_id: Option<Uuid>,
    pub active_connection_epoch: Option<i32>,
    pub active_connection_id: Option<Uuid>,
    pub last_connected_at: Option<DateTime<Utc>>,
    pub last_healthy_at: Option<DateTime<Utc>>,
    pub last_required_tick_at: Option<DateTime<Utc>>,
    pub last_disconnect_at: Option<DateTime<Utc>>,
    pub recovery_unavailable_since: Option<DateTime<Utc>>,
    pub last_disconnect_reason: Option<ReferenceDisconnectReason>,
    pub heartbeat_probes: u64,
    pub heartbeat_acknowledgements: u64,
    pub required_ticks_received: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BtcRuntimeMetrics {
    pub markets_discovered: u64,
    pub reference_ticks_received: u64,
    pub clob_messages_received: u64,
    pub feed_events_applied: u64,
    pub checkpoints_queued: u64,
    pub labels_created: u64,
    pub persistence_items_written: u64,
    pub persistence_errors: u64,
    pub strategy_errors: u64,
    pub decode_errors: u64,
    pub integrity_gaps: u64,
    pub dropped_messages: u64,
    pub reconnects: u64,
    pub clob_connections_established: u64,
    pub clob_subscription_updates: u64,
    pub clob_transport_disconnects: u64,
    pub clob_connection_failures: u64,
    pub clob_subscription_failures: u64,
    pub clob_bootstrap_failures: u64,
    pub clob_immediate_recoveries_scheduled: u64,
    pub clob_healthy_connections: u64,
    pub clob_backoff_scheduled_milliseconds: u64,
    pub clob_recovery_unavailable_milliseconds: u64,
    pub clob_consecutive_failures: u32,
    pub clob_connected_connection_epoch: Option<i32>,
    pub clob_connected_connection_id: Option<Uuid>,
    pub clob_active_connection_epoch: Option<i32>,
    pub clob_active_connection_id: Option<Uuid>,
    pub clob_last_connected_at: Option<DateTime<Utc>>,
    pub clob_last_healthy_at: Option<DateTime<Utc>>,
    pub clob_last_subscription_update_at: Option<DateTime<Utc>>,
    pub clob_last_disconnect_at: Option<DateTime<Utc>>,
    pub clob_recovery_unavailable_since: Option<DateTime<Utc>>,
    pub clob_last_disconnect_reason: Option<String>,
    pub clob_active_subscribed_assets: u64,
    pub rtds_transport: ReferenceTransportMetrics,
    pub binance_transport: ReferenceTransportMetrics,
    pub rtds_chainlink_ticks_received: u64,
    pub rtds_binance_ticks_received: u64,
    pub binance_ticks_received: u64,
    pub strategy_callbacks: u64,
    pub resolution_watches_active: u64,
    pub resolution_watches_rehydrated: u64,
    pub official_resolutions_websocket: u64,
    pub official_resolutions_rest: u64,
    pub resolution_reconciliation_errors: u64,
    pub resolution_watches_expired: u64,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BtcRuntimeStatus {
    pub enabled: bool,
    pub running: bool,
    pub readiness: Readiness,
    pub metrics: BtcRuntimeMetrics,
}

#[derive(Debug)]
struct ClobRecoveryWindow {
    since: Option<DateTime<Utc>>,
    started_at: Option<Instant>,
}

impl ClobRecoveryWindow {
    fn open(since: DateTime<Utc>, started_at: Instant) -> Self {
        Self {
            since: Some(since),
            started_at: Some(started_at),
        }
    }

    fn open_if_closed(&mut self, since: DateTime<Utc>, started_at: Instant) {
        if self.started_at.is_none() {
            self.since = Some(since);
            self.started_at = Some(started_at);
        }
    }

    fn close(&mut self, ended_at: Instant) -> u64 {
        self.since = None;
        self.started_at
            .take()
            .map(|started_at| duration_milliseconds(ended_at.duration_since(started_at)))
            .unwrap_or(0)
    }
}

#[derive(Debug, Default)]
struct ClobSubscriptionStats {
    updates: u64,
    active_assets: usize,
    last_updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
struct ClobSubscriptionDelta {
    added_assets: Vec<String>,
    removed_assets: Vec<String>,
    added_markets: Vec<BtcIntervalMarket>,
}

impl ClobSubscriptionDelta {
    fn is_empty(&self) -> bool {
        self.added_assets.is_empty() && self.removed_assets.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClobSubscriptionOperation {
    Subscribe,
    Unsubscribe,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ClobMarketIdentity {
    market_id: String,
    condition_id: String,
    up_token_id: String,
    down_token_id: String,
}

impl From<&BtcIntervalMarket> for ClobMarketIdentity {
    fn from(market: &BtcIntervalMarket) -> Self {
        Self {
            market_id: market.market_id.clone(),
            condition_id: market.condition_id.clone(),
            up_token_id: market.up_token_id.clone(),
            down_token_id: market.down_token_id.clone(),
        }
    }
}

impl ClobMarketIdentity {
    fn matches(&self, market: &BtcIntervalMarket) -> bool {
        self.market_id == market.market_id
            && self.condition_id == market.condition_id
            && self.up_token_id == market.up_token_id
            && self.down_token_id == market.down_token_id
    }
}

#[derive(Debug)]
struct ClobFeedWatchdog {
    bootstrap_market: Option<ClobMarketIdentity>,
    bootstrap_deadline: Option<Instant>,
    read_idle_deadline: Instant,
    pong_deadline: Option<Instant>,
    awaiting_text_pong: bool,
}

impl ClobFeedWatchdog {
    fn new(
        now: Instant,
        registry: &BookRegistry,
        markets: &[BtcIntervalMarket],
        checked_at: DateTime<Utc>,
        max_book_age: Duration,
    ) -> Self {
        let mut watchdog = Self {
            bootstrap_market: None,
            bootstrap_deadline: None,
            read_idle_deadline: now + CLOB_READ_IDLE_TIMEOUT,
            pong_deadline: None,
            awaiting_text_pong: false,
        };
        watchdog.refresh_bootstrap(now, registry, markets, checked_at, max_book_age);
        watchdog
    }

    fn on_frame(&mut self, now: Instant) {
        self.read_idle_deadline = now + CLOB_READ_IDLE_TIMEOUT;
    }

    fn arm_text_pong(&mut self, now: Instant) {
        self.awaiting_text_pong = true;
        self.pong_deadline = Some(now + CLOB_PONG_TIMEOUT);
    }

    fn acknowledge_text_pong(&mut self, text: &str) -> bool {
        if self.awaiting_text_pong && text == "PONG" {
            self.awaiting_text_pong = false;
            self.pong_deadline = None;
            true
        } else {
            false
        }
    }

    fn refresh_bootstrap(
        &mut self,
        now: Instant,
        registry: &BookRegistry,
        markets: &[BtcIntervalMarket],
        checked_at: DateTime<Utc>,
        max_book_age: Duration,
    ) {
        let Some(current_market) = unique_current_clob_market(markets, checked_at) else {
            self.bootstrap_market = None;
            self.bootstrap_deadline = None;
            return;
        };
        let identity_changed = self
            .bootstrap_market
            .as_ref()
            .is_none_or(|identity| !identity.matches(current_market));
        if identity_changed {
            self.bootstrap_market = Some(ClobMarketIdentity::from(current_market));
        }
        if registry.market_books_ready(current_market, checked_at, max_book_age) {
            self.bootstrap_deadline = None;
        } else if identity_changed {
            self.bootstrap_deadline = Some(now + CLOB_BOOTSTRAP_TIMEOUT);
        }
    }
}

#[derive(Debug)]
enum ClobSendFailure {
    Shutdown,
    Timeout,
    Transport(String),
}

async fn send_clob_text(
    socket: &mut ClobSocket,
    payload: String,
    shutdown: &mut watch::Receiver<bool>,
) -> std::result::Result<(), ClobSendFailure> {
    if *shutdown.borrow() {
        return Err(ClobSendFailure::Shutdown);
    }
    tokio::select! {
        biased;
        _ = shutdown.changed() => Err(ClobSendFailure::Shutdown),
        result = timeout(CLOB_SEND_TIMEOUT, socket.send(Message::Text(payload.into()))) => {
            match result {
                Err(_) => Err(ClobSendFailure::Timeout),
                Ok(Err(error)) => Err(ClobSendFailure::Transport(error.to_string())),
                Ok(Ok(())) => Ok(()),
            }
        }
    }
}

#[derive(Debug)]
struct ClobEpoch {
    connection_id: Uuid,
    connection_epoch: i32,
    socket: ClobSocket,
    registry: BookRegistry,
    markets: Vec<BtcIntervalMarket>,
    session: FeedSession,
    subscription_stats: ClobSubscriptionStats,
    connected_instant: Instant,
    watchdog: ClobFeedWatchdog,
    healthy_epoch: bool,
    books_usable: bool,
    pending_resolutions: HashMap<String, BufferedClobResolution>,
}

#[derive(Debug)]
struct BufferedClobResolution {
    message: ClobMessage,
    received_at: DateTime<Utc>,
}

impl ClobEpoch {
    fn refresh_private_health(
        &mut self,
        checked_at: DateTime<Utc>,
        checked_instant: Instant,
        max_book_age: Duration,
    ) {
        let was_usable = self.books_usable;
        self.watchdog.refresh_bootstrap(
            checked_instant,
            &self.registry,
            &self.markets,
            checked_at,
            max_book_age,
        );
        let books_usable =
            clob_epoch_ready(&self.registry, &self.markets, checked_at, max_book_age);
        if was_usable
            && !books_usable
            && self.watchdog.bootstrap_market.is_some()
            && self.watchdog.bootstrap_deadline.is_none()
        {
            self.watchdog.bootstrap_deadline = Some(checked_instant + CLOB_BOOTSTRAP_TIMEOUT);
        }
        self.books_usable = books_usable;
        self.healthy_epoch |= self.books_usable;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClobConnectFailureKind {
    Connect,
    Subscription,
    Identity,
    Shutdown,
}

#[derive(Debug)]
enum ClobConnectOutcome {
    Connected {
        epoch: ClobEpoch,
        connect_latency: StdDuration,
    },
    Failed {
        session: FeedSession,
        reason: String,
        kind: ClobConnectFailureKind,
    },
}

async fn connect_clob_epoch(
    config: BtcRuntimeConfig,
    desired_markets: Vec<BtcIntervalMarket>,
    connection_epoch: i32,
    mut shutdown: watch::Receiver<bool>,
) -> ClobConnectOutcome {
    let connection_id = Uuid::new_v4();
    let attempt_started_at = Instant::now();
    let mut session = new_session(
        connection_id,
        "polymarket_clob_market",
        &config.clob_ws_url,
        connection_epoch,
        Utc::now(),
    );
    let mut registry = BookRegistry::new(connection_id);
    if let Err(error) = register_clob_markets(&mut registry, &desired_markets) {
        session.disconnected_at = Some(Utc::now());
        session.disconnect_reason = Some(format!("invalid_subscription_identity:{error}"));
        return ClobConnectOutcome::Failed {
            session,
            reason: format!("invalid_subscription_identity:{error}"),
            kind: ClobConnectFailureKind::Identity,
        };
    }
    if *shutdown.borrow() {
        session.disconnected_at = Some(Utc::now());
        session.disconnect_reason = Some("shutdown".to_string());
        return ClobConnectOutcome::Failed {
            session,
            reason: "shutdown".to_string(),
            kind: ClobConnectFailureKind::Shutdown,
        };
    }
    let connect_result = tokio::select! {
        biased;
        _ = shutdown.changed() => None,
        result = timeout(CLOB_CONNECT_TIMEOUT, connect_async(&config.clob_ws_url)) => Some(result),
    };
    let (mut socket, _) = match connect_result {
        None => {
            session.disconnected_at = Some(Utc::now());
            session.disconnect_reason = Some("shutdown".to_string());
            return ClobConnectOutcome::Failed {
                session,
                reason: "shutdown".to_string(),
                kind: ClobConnectFailureKind::Shutdown,
            };
        }
        Some(Err(_)) => {
            session.disconnected_at = Some(Utc::now());
            session.disconnect_reason = Some("connect_timeout".to_string());
            return ClobConnectOutcome::Failed {
                session,
                reason: "connect_timeout".to_string(),
                kind: ClobConnectFailureKind::Connect,
            };
        }
        Some(Ok(Err(error))) => {
            let reason = format!("connect_failed:{error}");
            session.disconnected_at = Some(Utc::now());
            session.disconnect_reason = Some(reason.clone());
            return ClobConnectOutcome::Failed {
                session,
                reason,
                kind: ClobConnectFailureKind::Connect,
            };
        }
        Some(Ok(Ok(value))) => value,
    };
    let connected_at = Utc::now();
    let connected_instant = Instant::now();
    session.connected_at = Some(connected_at);
    match send_clob_text(
        &mut socket,
        clob_subscription(&desired_markets),
        &mut shutdown,
    )
    .await
    {
        Ok(()) => {}
        Err(ClobSendFailure::Shutdown) => {
            session.disconnected_at = Some(Utc::now());
            session.disconnect_reason = Some("shutdown".to_string());
            return ClobConnectOutcome::Failed {
                session,
                reason: "shutdown".to_string(),
                kind: ClobConnectFailureKind::Shutdown,
            };
        }
        Err(ClobSendFailure::Timeout) => {
            session.disconnected_at = Some(Utc::now());
            session.disconnect_reason = Some("subscription_send_timeout".to_string());
            return ClobConnectOutcome::Failed {
                session,
                reason: "subscription_send_timeout".to_string(),
                kind: ClobConnectFailureKind::Subscription,
            };
        }
        Err(ClobSendFailure::Transport(error)) => {
            let reason = format!("subscription_send_failed:{error}");
            session.disconnected_at = Some(Utc::now());
            session.disconnect_reason = Some(reason.clone());
            return ClobConnectOutcome::Failed {
                session,
                reason,
                kind: ClobConnectFailureKind::Subscription,
            };
        }
    }
    let watchdog_started = Instant::now();
    let watchdog = ClobFeedWatchdog::new(
        watchdog_started,
        &registry,
        &desired_markets,
        Utc::now(),
        chrono_duration(config.max_book_age),
    );
    let pending_resolution_capacity = desired_markets.len();
    ClobConnectOutcome::Connected {
        epoch: ClobEpoch {
            connection_id,
            connection_epoch,
            socket,
            registry,
            markets: desired_markets,
            session,
            subscription_stats: ClobSubscriptionStats::default(),
            connected_instant,
            watchdog,
            healthy_epoch: false,
            books_usable: false,
            pending_resolutions: HashMap::with_capacity(pending_resolution_capacity),
        },
        connect_latency: attempt_started_at.elapsed(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferenceFeedKind {
    Rtds,
    Binance,
}

impl ReferenceFeedKind {
    fn feed_name(self) -> &'static str {
        match self {
            Self::Rtds => "polymarket_rtds",
            Self::Binance => "binance_agg_trade",
        }
    }

    fn required_data_timeout(self) -> StdDuration {
        match self {
            Self::Rtds => RTDS_REQUIRED_DATA_TIMEOUT,
            Self::Binance => BINANCE_REQUIRED_DATA_TIMEOUT,
        }
    }

    fn retry_salt(self) -> u64 {
        match self {
            Self::Rtds => 0x5254_4453,
            Self::Binance => 0x4249_4e41,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferenceWatchdogTimeout {
    RequiredData,
    HeartbeatAck,
    ReadIdle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferenceDisconnectCause {
    ConnectFailure,
    SubscriptionFailure,
    WatchdogTimeout(ReferenceWatchdogTimeout),
    TransportFailure,
    Shutdown,
    CriticalPersistence,
}

impl ReferenceDisconnectCause {
    fn as_str(self) -> &'static str {
        match self {
            Self::ConnectFailure => "connect_failure",
            Self::SubscriptionFailure => "subscription_failure",
            Self::WatchdogTimeout(_) => "watchdog_timeout",
            Self::TransportFailure => "transport_failure",
            Self::Shutdown => "shutdown",
            Self::CriticalPersistence => "critical_persistence",
        }
    }
}

impl ReferenceDisconnectReason {
    fn cause(self) -> ReferenceDisconnectCause {
        match self {
            Self::Shutdown => ReferenceDisconnectCause::Shutdown,
            Self::ConnectTimeout | Self::ConnectFailed => ReferenceDisconnectCause::ConnectFailure,
            Self::SubscriptionSendTimeout | Self::SubscriptionSendFailed => {
                ReferenceDisconnectCause::SubscriptionFailure
            }
            Self::HeartbeatAckTimeout => {
                ReferenceDisconnectCause::WatchdogTimeout(ReferenceWatchdogTimeout::HeartbeatAck)
            }
            Self::RequiredDataIdleTimeout => {
                ReferenceDisconnectCause::WatchdogTimeout(ReferenceWatchdogTimeout::RequiredData)
            }
            Self::ReadIdleTimeout => {
                ReferenceDisconnectCause::WatchdogTimeout(ReferenceWatchdogTimeout::ReadIdle)
            }
            Self::CriticalBoundaryIntegrity
            | Self::CriticalBoundaryPersistence
            | Self::CriticalWriterQueue => ReferenceDisconnectCause::CriticalPersistence,
            Self::HeartbeatSendTimeout
            | Self::HeartbeatSendFailed
            | Self::WebsocketEof
            | Self::RemoteClose
            | Self::TransportReadFailed
            | Self::UnknownDisconnect => ReferenceDisconnectCause::TransportFailure,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferencePongExpectation {
    Binary([u8; 8]),
}

#[derive(Debug)]
struct ReferenceFeedWatchdog {
    required_data_deadline: Instant,
    read_idle_deadline: Instant,
    pong_deadline: Option<Instant>,
    stable_deadline: Option<Instant>,
    expected_pong: Option<ReferencePongExpectation>,
    stable: bool,
}

impl ReferenceFeedWatchdog {
    fn new(now: Instant, kind: ReferenceFeedKind) -> Self {
        Self {
            required_data_deadline: now + kind.required_data_timeout(),
            read_idle_deadline: now + REFERENCE_READ_IDLE_TIMEOUT,
            pong_deadline: None,
            stable_deadline: None,
            expected_pong: None,
            stable: false,
        }
    }

    fn on_frame(&mut self, now: Instant) {
        self.read_idle_deadline = now + REFERENCE_READ_IDLE_TIMEOUT;
    }

    fn on_required_tick(&mut self, now: Instant, kind: ReferenceFeedKind) {
        self.required_data_deadline = now + kind.required_data_timeout();
        if self.stable_deadline.is_none() && !self.stable {
            self.stable_deadline = Some(now + REFERENCE_STABLE_RESET_AFTER);
        }
    }

    fn arm_binary_pong(&mut self, now: Instant, payload: [u8; 8]) {
        self.expected_pong = Some(ReferencePongExpectation::Binary(payload));
        self.pong_deadline = Some(now + REFERENCE_PONG_TIMEOUT);
    }

    fn acknowledge_binary_pong(&mut self, payload: &[u8]) -> bool {
        let acknowledged = matches!(
            self.expected_pong,
            Some(ReferencePongExpectation::Binary(expected)) if expected.as_slice() == payload
        );
        if acknowledged {
            self.expected_pong = None;
            self.pong_deadline = None;
        }
        acknowledged
    }

    fn awaiting_pong(&self) -> bool {
        self.expected_pong.is_some()
    }

    fn mark_stable(&mut self) -> bool {
        if self.stable || self.stable_deadline.is_none() {
            return false;
        }
        self.stable = true;
        self.stable_deadline = None;
        true
    }
}

#[derive(Debug, Default)]
struct ReferenceSessionStats {
    healthy_epoch: bool,
    stable_epoch: bool,
    required_ticks: u64,
    heartbeat_probes: u64,
    heartbeat_acknowledgements: u64,
    last_required_tick_at: Option<DateTime<Utc>>,
    last_frame_at: Option<DateTime<Utc>>,
    last_pong_at: Option<DateTime<Utc>>,
    time_to_first_required_tick_milliseconds: Option<u64>,
    remote_close_code: Option<u16>,
}

#[derive(Debug, Default)]
struct ReferenceRetryState {
    consecutive_failures: u32,
}

impl ReferenceRetryState {
    fn record_failure(&mut self) -> u32 {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        self.consecutive_failures
    }

    fn reset(&mut self) {
        self.consecutive_failures = 0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferenceRetryAction {
    Stop,
    ImmediateRecovery,
    Backoff(StdDuration),
}

#[derive(Debug)]
struct ReferenceRecoveryWindow {
    since: Option<DateTime<Utc>>,
    started_at: Option<Instant>,
}

impl ReferenceRecoveryWindow {
    fn open(since: DateTime<Utc>, started_at: Instant) -> Self {
        Self {
            since: Some(since),
            started_at: Some(started_at),
        }
    }

    fn open_if_closed(&mut self, since: DateTime<Utc>, started_at: Instant) {
        if self.started_at.is_none() {
            self.since = Some(since);
            self.started_at = Some(started_at);
        }
    }

    fn close(&mut self, ended_at: Instant) -> u64 {
        self.since = None;
        self.started_at
            .take()
            .map(|started_at| duration_milliseconds(ended_at.duration_since(started_at)))
            .unwrap_or(0)
    }
}

pub type BtcRuntimeStatusInputs = (
    Arc<RwLock<RealtimeState>>,
    Arc<RwLock<BtcRuntimeMetrics>>,
    BtcRuntimeConfig,
    Arc<AtomicBool>,
);

#[derive(Debug, Clone)]
pub struct StrategyObservation {
    pub state: RealtimeState,
    pub readiness: Readiness,
}

#[async_trait]
pub trait BtcStrategyRunner: Send + Sync {
    async fn on_observation(&self, observation: StrategyObservation) -> Result<()>;

    /// Drains strategy-owned asynchronous work after feed tasks have stopped producing.
    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct NoopStrategyRunner;

#[async_trait]
impl BtcStrategyRunner for NoopStrategyRunner {
    async fn on_observation(&self, _observation: StrategyObservation) -> Result<()> {
        Ok(())
    }
}

pub struct BtcRuntime {
    config: BtcRuntimeConfig,
    repository: BtcRepository,
    books: Option<Arc<RwLock<BookRegistry>>>,
    state: Option<Arc<RwLock<RealtimeState>>>,
}

impl BtcRuntime {
    pub fn new(config: BtcRuntimeConfig, repository: BtcRepository) -> Self {
        Self {
            config,
            repository,
            books: None,
            state: None,
        }
    }

    /// Uses caller-owned shared state so every playbook observes one canonical feed runtime.
    pub fn with_shared_state(mut self, state: Arc<RwLock<RealtimeState>>) -> Self {
        self.state = Some(state);
        self
    }

    /// Uses a caller-owned registry so every paper venue reads the same arrival-time book state.
    pub fn with_shared_book_registry(mut self, books: Arc<RwLock<BookRegistry>>) -> Self {
        self.books = Some(books);
        self
    }

    pub async fn start(self) -> Result<BtcRuntimeHandle> {
        self.config.validate()?;
        self.repository.healthcheck().await?;

        let state = self
            .state
            .unwrap_or_else(|| Arc::new(RwLock::new(RealtimeState::default())));
        let books = self
            .books
            .unwrap_or_else(|| Arc::new(RwLock::new(BookRegistry::new(Uuid::new_v4()))));
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let running = Arc::new(AtomicBool::new(true));
        let boundaries = Arc::new(RwLock::new(BoundaryTracker::default()));
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (market_tx, market_rx) = watch::channel(Vec::<BtcIntervalMarket>::new());
        let (writer_tx, writer_rx) = mpsc::channel(self.config.writer_capacity);
        let tasks = vec![
            spawn_runtime_task(
                "writer",
                run_writer(self.repository.clone(), writer_rx, metrics.clone()),
                running.clone(),
                metrics.clone(),
            ),
            spawn_runtime_task(
                "discovery",
                run_discovery(
                    self.config.clone(),
                    self.repository.clone(),
                    market_tx,
                    state.clone(),
                    boundaries.clone(),
                    metrics.clone(),
                    shutdown_rx.clone(),
                ),
                running.clone(),
                metrics.clone(),
            ),
            spawn_runtime_task(
                "clob",
                run_clob_supervisor(
                    self.config.clone(),
                    self.repository.clone(),
                    market_rx,
                    writer_tx.clone(),
                    state.clone(),
                    books.clone(),
                    metrics.clone(),
                    shutdown_rx.clone(),
                ),
                running.clone(),
                metrics.clone(),
            ),
            spawn_runtime_task(
                "rtds",
                run_rtds_supervisor(
                    self.config.clone(),
                    self.repository.clone(),
                    writer_tx.clone(),
                    state.clone(),
                    boundaries,
                    metrics.clone(),
                    shutdown_rx.clone(),
                ),
                running.clone(),
                metrics.clone(),
            ),
            spawn_runtime_task(
                "binance",
                run_binance_supervisor(
                    self.config.clone(),
                    self.repository.clone(),
                    writer_tx.clone(),
                    state.clone(),
                    metrics.clone(),
                    shutdown_rx.clone(),
                ),
                running.clone(),
                metrics.clone(),
            ),
        ];
        drop(shutdown_rx);
        drop(writer_tx);

        Ok(BtcRuntimeHandle {
            enabled: self.config.enabled,
            shutdown: shutdown_tx,
            tasks,
            state,
            books,
            metrics,
            config: self.config,
            running,
        })
    }
}

fn spawn_runtime_task<F>(
    name: &'static str,
    future: F,
    running: Arc<AtomicBool>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
) -> JoinHandle<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    tokio::spawn(async move {
        let outcome = AssertUnwindSafe(future).catch_unwind().await;
        if running.swap(false, Ordering::Relaxed) {
            let reason = if outcome.is_err() {
                "panicked"
            } else {
                "exited"
            };
            let mut runtime_metrics = metrics.write().await;
            runtime_metrics
                .last_error
                .get_or_insert_with(|| format!("BTC runtime {name} task unexpectedly {reason}"));
        }
    })
}

pub struct BtcRuntimeHandle {
    enabled: bool,
    shutdown: watch::Sender<bool>,
    tasks: Vec<JoinHandle<()>>,
    state: Arc<RwLock<RealtimeState>>,
    books: Arc<RwLock<BookRegistry>>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
    config: BtcRuntimeConfig,
    running: Arc<AtomicBool>,
}

impl BtcRuntimeHandle {
    pub fn shared_state(&self) -> Arc<RwLock<RealtimeState>> {
        self.state.clone()
    }

    pub fn shared_book_registry(&self) -> Arc<RwLock<BookRegistry>> {
        self.books.clone()
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub fn status_inputs(&self) -> BtcRuntimeStatusInputs {
        (
            self.state.clone(),
            self.metrics.clone(),
            self.config.clone(),
            self.running.clone(),
        )
    }

    pub async fn status(&self) -> BtcRuntimeStatus {
        let state = self.state.read().await.clone();
        let readiness = state.readiness(
            Utc::now(),
            chrono_duration(self.config.max_book_age),
            chrono_duration(self.config.max_reference_age),
        );
        BtcRuntimeStatus {
            enabled: self.enabled,
            running: self.running.load(Ordering::Relaxed),
            readiness,
            metrics: self.metrics.read().await.clone(),
        }
    }

    pub async fn shutdown(mut self) -> Result<()> {
        self.running.store(false, Ordering::Relaxed);
        let _ = self.shutdown.send(true);
        let mut join_failures = Vec::new();
        for task in &mut self.tasks {
            if let Err(error) = task.await {
                join_failures.push(error.to_string());
            }
        }
        self.tasks.clear();
        if !join_failures.is_empty() {
            bail!("BTC runtime task join failed: {}", join_failures.join("; "));
        }
        let final_metrics = self.metrics.read().await.clone();
        if let Some(reason) = primary_runtime_failure(&final_metrics) {
            bail!("BTC runtime primary-path integrity failed: {reason}");
        }
        Ok(())
    }
}

impl Drop for BtcRuntimeHandle {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        let _ = self.shutdown.send(true);
        for task in &self.tasks {
            task.abort();
        }
    }
}

pub struct BtcPlaybookRuntimeHandle {
    shutdown: watch::Sender<bool>,
    task: Option<JoinHandle<()>>,
    state: Arc<RwLock<RealtimeState>>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
    config: BtcRuntimeConfig,
    running: Arc<AtomicBool>,
    strategy: Arc<dyn BtcStrategyRunner>,
}

impl BtcPlaybookRuntimeHandle {
    pub fn start(
        config: BtcRuntimeConfig,
        strategy: Arc<dyn BtcStrategyRunner>,
        state: Arc<RwLock<RealtimeState>>,
    ) -> Result<Self> {
        config.validate()?;
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let running = Arc::new(AtomicBool::new(true));
        let (shutdown, shutdown_rx) = watch::channel(false);
        let task = spawn_runtime_task(
            "playbook",
            run_strategy_loop(
                config.clone(),
                strategy.clone(),
                state.clone(),
                metrics.clone(),
                shutdown_rx,
            ),
            running.clone(),
            metrics.clone(),
        );
        Ok(Self {
            shutdown,
            task: Some(task),
            state,
            metrics,
            config,
            running,
            strategy,
        })
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub fn status_inputs(&self) -> BtcRuntimeStatusInputs {
        (
            self.state.clone(),
            self.metrics.clone(),
            self.config.clone(),
            self.running.clone(),
        )
    }

    pub async fn status(&self) -> BtcRuntimeStatus {
        runtime_status_from_inputs(
            self.state.clone(),
            self.metrics.clone(),
            self.config.clone(),
            self.running.clone(),
        )
        .await
    }

    pub async fn shutdown(mut self) -> Result<()> {
        self.running.store(false, Ordering::Relaxed);
        let _ = self.shutdown.send(true);
        if let Some(task) = self.task.take() {
            task.await.context("BTC playbook task join failed")?;
        }
        self.strategy
            .shutdown()
            .await
            .context("BTC playbook shutdown/drain failed")?;
        let final_metrics = self.metrics.read().await.clone();
        if let Some(reason) = primary_runtime_failure(&final_metrics) {
            bail!("BTC playbook primary-path integrity failed: {reason}");
        }
        Ok(())
    }
}

impl Drop for BtcPlaybookRuntimeHandle {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        let _ = self.shutdown.send(true);
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

pub async fn runtime_status_from_inputs(
    state: Arc<RwLock<RealtimeState>>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
    config: BtcRuntimeConfig,
    running: Arc<AtomicBool>,
) -> BtcRuntimeStatus {
    let state = state.read().await.clone();
    let readiness = state.readiness(
        Utc::now(),
        chrono_duration(config.max_book_age),
        chrono_duration(config.max_reference_age),
    );
    BtcRuntimeStatus {
        enabled: config.enabled,
        running: running.load(Ordering::Relaxed),
        readiness,
        metrics: metrics.read().await.clone(),
    }
}

#[derive(Debug)]
enum PersistItem {
    ReferenceTick(ReferencePriceTick),
    FeedEvent(super::types::MarketFeedEvent),
    Checkpoint(super::types::OrderbookCheckpoint),
}

async fn run_writer(
    repository: BtcRepository,
    mut receiver: mpsc::Receiver<PersistItem>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
) {
    while let Some(item) = receiver.recv().await {
        let result = match item {
            PersistItem::ReferenceTick(tick) => {
                repository.insert_reference_tick(&tick).await.map(|_| ())
            }
            PersistItem::FeedEvent(event) => repository.insert_feed_event(&event).await.map(|_| ()),
            PersistItem::Checkpoint(checkpoint) => repository
                .insert_orderbook_checkpoint(&checkpoint, "websocket_book")
                .await
                .map(|_| ()),
        };
        let mut metrics = metrics.write().await;
        match result {
            Ok(()) => {
                metrics.persistence_items_written =
                    metrics.persistence_items_written.saturating_add(1)
            }
            Err(error) => {
                metrics.persistence_errors = metrics.persistence_errors.saturating_add(1);
                metrics.last_error = Some(error.to_string());
                // An immutable realtime-paper experiment permits no primary-writer
                // failures. Exit so the shared runtime fails instead of allowing a
                // partially durable experiment to continue collecting.
                return;
            }
        }
    }
}

async fn enqueue(
    sender: &mpsc::Sender<PersistItem>,
    item: PersistItem,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
) -> bool {
    match sender.try_send(item) {
        Ok(()) => true,
        Err(error) => {
            let mut metrics = metrics.write().await;
            metrics.dropped_messages = metrics.dropped_messages.saturating_add(1);
            metrics.last_error = Some(format!("BTC persistence queue rejected item: {error}"));
            false
        }
    }
}

fn should_persist_feed_event(event: &super::types::MarketFeedEvent) -> bool {
    !event.applied
        || matches!(
            event.event_type,
            MarketFeedEventType::Book
                | MarketFeedEventType::TickSizeChange
                | MarketFeedEventType::MarketResolved
        )
}

async fn run_discovery(
    config: BtcRuntimeConfig,
    repository: BtcRepository,
    market_sender: watch::Sender<Vec<BtcIntervalMarket>>,
    state: Arc<RwLock<RealtimeState>>,
    boundaries: Arc<RwLock<BoundaryTracker>>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
    mut shutdown: watch::Receiver<bool>,
) {
    if !config.enabled {
        return;
    }
    let client = reqwest::Client::builder()
        .timeout(StdDuration::from_secs(5))
        .build()
        .unwrap_or_default();
    let resolution_retention = chrono_duration(config.official_resolution_watch_retention);
    if let Err(error) = repository
        .seed_recent_official_resolution_watches(Utc::now(), resolution_retention)
        .await
    {
        record_critical_persistence_error(&metrics, error).await;
        return;
    }
    match repository
        .load_unsettled_official_resolution_watches()
        .await
    {
        Ok(watches) => {
            metrics.write().await.resolution_watches_rehydrated = watches.len() as u64;
        }
        Err(error) => {
            record_critical_persistence_error(&metrics, error).await;
            return;
        }
    }
    let mut ticker = interval(config.discovery_interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = shutdown.changed() => break,
            _ = ticker.tick() => {
                let now = Utc::now();
                let fresh_markets = match discover_markets(&client, &config, now).await {
                    Ok(markets) => markets,
                    Err(error) => {
                        record_error(&metrics, error).await;
                        Vec::new()
                    }
                };
                for market in &fresh_markets {
                    if let Err(error) = repository.upsert_interval_market(market).await {
                        record_critical_persistence_error(&metrics, error).await;
                        return;
                    }
                    if let Err(error) = repository
                        .register_official_resolution_watch(market, now, resolution_retention)
                        .await
                    {
                        record_critical_persistence_error(&metrics, error).await;
                        return;
                    }
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.persistence_items_written = runtime_metrics
                        .persistence_items_written
                        .saturating_add(1);
                }

                // Only a fresh Gamma response is eligible to become tradable. Durable recovery
                // rows are subscription/audit inputs and can never re-open an old market.
                let current = fresh_markets
                    .iter()
                    .find(|market| market.is_trade_window(now))
                    .cloned();
                state.write().await.set_current_market(current);

                let watches = match repository
                    .load_unsettled_official_resolution_watches()
                    .await
                {
                    Ok(watches) => watches,
                    Err(error) => {
                        record_critical_persistence_error(&metrics, error).await;
                        return;
                    }
                };
                if let Err(error) = reconcile_official_resolution_watches(
                    &client,
                    &config,
                    &repository,
                    &watches,
                    state.clone(),
                    &metrics,
                    now,
                )
                .await
                {
                    record_critical_persistence_error(&metrics, error).await;
                    return;
                }
                let expired = match repository
                    .expire_overdue_official_resolution_watches(Utc::now())
                    .await
                {
                    Ok(expired) => expired,
                    Err(error) => {
                        record_critical_persistence_error(&metrics, error).await;
                        return;
                    }
                };
                if !expired.is_empty() {
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.resolution_watches_expired = runtime_metrics
                        .resolution_watches_expired
                        .saturating_add(expired.len() as u64);
                }
                let watches = match repository
                    .load_unsettled_official_resolution_watches()
                    .await
                {
                    Ok(watches) => watches,
                    Err(error) => {
                        record_critical_persistence_error(&metrics, error).await;
                        return;
                    }
                };
                let expired_unresolved = watches
                    .iter()
                    .filter(|watch| watch.status == "expired")
                    .map(|watch| watch.market.market_id.clone())
                    .collect::<Vec<_>>();
                if !expired_unresolved.is_empty() {
                    record_critical_persistence_error(
                        &metrics,
                        anyhow::anyhow!(
                            "expired unresolved BTC official-resolution watches: {}",
                            expired_unresolved.join(",")
                        ),
                    )
                    .await;
                    return;
                }

                let pending_markets = watches
                    .iter()
                    .filter(|watch| watch.status == "pending")
                    .map(|watch| watch.market.clone())
                    .collect::<Vec<_>>();
                let capacity = resolution_watch_capacity(config.official_resolution_watch_retention);
                if pending_markets.len() > capacity {
                    record_critical_persistence_error(
                        &metrics,
                        anyhow::anyhow!(
                            "BTC official-resolution watch capacity exceeded: {} > {}",
                            pending_markets.len(),
                            capacity
                        ),
                    )
                    .await;
                    return;
                }

                // Boundary recovery is broader than trading and independent of whether an
                // official result has arrived. Fresh previous/current/next plus durable pending
                // rows preserve local labels across restart and delayed settlement.
                let mut recovery_markets = pending_markets.clone();
                let mut recovery_ids = recovery_markets
                    .iter()
                    .map(|market| market.market_id.clone())
                    .collect::<HashSet<_>>();
                for market in &fresh_markets {
                    if recovery_ids.insert(market.market_id.clone()) {
                        recovery_markets.push(market.clone());
                    }
                }
                recovery_markets.sort_by_key(|market| market.window_start);
                let max_delay = chrono_duration(config.boundary_tick_max_delay);
                let durable = tokio::try_join!(
                    repository.load_market_open_references(&recovery_markets),
                    repository.load_market_close_references(&recovery_markets, max_delay),
                    repository.load_market_labels(&recovery_markets),
                );
                let (durable_opens, durable_closes, durable_labels) = match durable {
                    Ok(value) => value,
                    Err(error) => {
                        record_critical_persistence_error(&metrics, error).await;
                        return;
                    }
                };
                let hydration = {
                    let mut tracker = boundaries.write().await;
                    tracker.update_markets(&recovery_markets);
                    tracker
                        .hydrate_open_references(&durable_opens, max_delay)
                        .and_then(|_| tracker.hydrate_close_references(&durable_closes, max_delay))
                        .and_then(|_| tracker.hydrate_labels(&durable_labels))
                };
                if let Err(error) = hydration {
                    record_critical_persistence_error(&metrics, error).await;
                    return;
                }
                if let Err(error) = flush_pending_boundaries(
                    &repository,
                    &boundaries,
                    &metrics,
                    max_delay,
                )
                .await
                {
                    record_critical_persistence_error(&metrics, error).await;
                    return;
                }
                if !same_market_subscriptions(&market_sender.borrow(), &pending_markets) {
                    let _ = market_sender.send(pending_markets.clone());
                }
                let mut runtime_metrics = metrics.write().await;
                runtime_metrics.markets_discovered = fresh_markets.len() as u64;
                runtime_metrics.resolution_watches_active = pending_markets.len() as u64;
            }
        }
    }
}

async fn reconcile_official_resolution_watches(
    client: &reqwest::Client,
    config: &BtcRuntimeConfig,
    repository: &BtcRepository,
    watches: &[BtcOfficialResolutionWatch],
    state: Arc<RwLock<RealtimeState>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    now: DateTime<Utc>,
) -> Result<()> {
    let candidates = watches
        .iter()
        .filter(|watch| watch.market.window_end <= now)
        .cloned()
        .collect::<Vec<_>>();
    let results = stream::iter(candidates)
        .map(|watch| {
            let client = client.clone();
            let base_url = config.clob_rest_base_url.clone();
            async move {
                let result =
                    fetch_clob_rest_official_resolution(&client, &base_url, &watch.market).await;
                (watch, result)
            }
        })
        .buffer_unordered(4)
        .collect::<Vec<_>>()
        .await;

    for (watch, result) in results {
        match result {
            Ok(Some(resolution)) => {
                let persisted = persist_official_resolution_fact(
                    repository,
                    &resolution.market_id,
                    &resolution.winning_token_id,
                    outcome_display_name(resolution.winning_outcome),
                    resolution.observed_at,
                    "clob_rest_reconciliation",
                    resolution.observed_at,
                    &resolution.raw_payload,
                    metrics,
                )
                .await?;
                repository
                    .mark_official_resolution_watch_checked(
                        &persisted.market_id,
                        resolution.observed_at,
                        None,
                    )
                    .await?;
                state
                    .write()
                    .await
                    .apply_market_resolution(&persisted.market_id, &persisted.winning_token_id);
            }
            Ok(None) => {
                repository
                    .mark_official_resolution_watch_checked(
                        &watch.market.market_id,
                        Utc::now(),
                        None,
                    )
                    .await?;
            }
            Err(error) => {
                let message = format!("{error:#}");
                repository
                    .mark_official_resolution_watch_checked(
                        &watch.market.market_id,
                        Utc::now(),
                        Some(&message),
                    )
                    .await?;
                let mut runtime_metrics = metrics.write().await;
                runtime_metrics.resolution_reconciliation_errors = runtime_metrics
                    .resolution_reconciliation_errors
                    .saturating_add(1);
                runtime_metrics.last_error = Some(format!(
                    "BTC official resolution reconciliation failed for {}: {}",
                    watch.market.market_id, message
                ));
            }
        }
    }
    Ok(())
}

async fn fetch_clob_rest_official_resolution(
    client: &reqwest::Client,
    base_url: &str,
    market: &BtcIntervalMarket,
) -> Result<Option<ClobRestOfficialResolution>> {
    let url = format!(
        "{}/markets/{}",
        base_url.trim_end_matches('/'),
        market.condition_id
    );
    let value = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("failed to reconcile CLOB market {}", market.market_id))?
        .error_for_status()
        .with_context(|| format!("CLOB REST rejected market {}", market.market_id))?
        .json::<serde_json::Value>()
        .await
        .with_context(|| format!("failed to decode CLOB market {}", market.market_id))?;
    parse_clob_rest_official_resolution(&value, market, Utc::now())
}

fn resolution_watch_capacity(retention: StdDuration) -> usize {
    let intervals = retention.as_secs().saturating_add(299) / 300;
    usize::try_from(intervals.saturating_add(2)).unwrap_or(usize::MAX)
}

fn outcome_display_name(outcome: BtcOutcome) -> &'static str {
    match outcome {
        BtcOutcome::Up => "Up",
        BtcOutcome::Down => "Down",
    }
}

async fn discover_markets(
    client: &reqwest::Client,
    config: &BtcRuntimeConfig,
    now: DateTime<Utc>,
) -> Result<Vec<BtcIntervalMarket>> {
    let mut markets = Vec::new();
    for window_start in discovery_windows(now) {
        let slug = slug_for_window(window_start);
        let url = format!(
            "{}/events/slug/{}",
            config.gamma_base_url.trim_end_matches('/'),
            slug
        );
        let response = client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("failed to request BTC interval event {slug}"))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            continue;
        }
        let value = response
            .error_for_status()
            .with_context(|| format!("Gamma rejected BTC interval event {slug}"))?
            .json::<serde_json::Value>()
            .await
            .with_context(|| format!("failed to decode BTC interval event {slug}"))?;
        markets.push(parse_gamma_btc_interval_event(&value, window_start)?);
    }
    markets.sort_by_key(|market| market.window_start);
    Ok(markets)
}

#[derive(Debug)]
enum ClobEpochUpdateError {
    Shutdown,
    Recoverable(String),
    Critical(anyhow::Error),
}

async fn update_clob_epoch_subscriptions(
    epoch: &mut ClobEpoch,
    desired_markets: &[BtcIntervalMarket],
    repository: Option<&BtcRepository>,
    shutdown: &mut watch::Receiver<bool>,
    max_book_age: Duration,
) -> std::result::Result<ClobSubscriptionDelta, ClobEpochUpdateError> {
    epoch
        .registry
        .validate_market_set(desired_markets)
        .map_err(|error| {
            ClobEpochUpdateError::Recoverable(format!("invalid_subscription_transition:{error}"))
        })?;
    let delta = clob_subscription_delta(&epoch.markets, desired_markets);
    register_clob_markets(&mut epoch.registry, &delta.added_markets).map_err(|error| {
        ClobEpochUpdateError::Recoverable(format!("subscription_registration_failed:{error}"))
    })?;
    if !delta.added_assets.is_empty() {
        let payload =
            clob_subscription_operation(&delta.added_assets, ClobSubscriptionOperation::Subscribe);
        match send_clob_text(&mut epoch.socket, payload, shutdown).await {
            Ok(()) => {}
            Err(ClobSendFailure::Shutdown) => return Err(ClobEpochUpdateError::Shutdown),
            Err(ClobSendFailure::Timeout) => {
                return Err(ClobEpochUpdateError::Recoverable(
                    "dynamic_subscribe_timeout".to_string(),
                ));
            }
            Err(ClobSendFailure::Transport(error)) => {
                return Err(ClobEpochUpdateError::Recoverable(format!(
                    "dynamic_subscribe_failed:{error}"
                )));
            }
        }
    }
    if let Some(repository) = repository {
        if !delta.added_markets.is_empty() {
            acknowledge_clob_subscriptions(
                repository,
                &delta.added_markets,
                epoch.connection_id,
                Utc::now(),
            )
            .await
            .map_err(ClobEpochUpdateError::Critical)?;
        }
    }
    if !delta.removed_assets.is_empty() {
        let payload = clob_subscription_operation(
            &delta.removed_assets,
            ClobSubscriptionOperation::Unsubscribe,
        );
        match send_clob_text(&mut epoch.socket, payload, shutdown).await {
            Ok(()) => {}
            Err(ClobSendFailure::Shutdown) => return Err(ClobEpochUpdateError::Shutdown),
            Err(ClobSendFailure::Timeout) => {
                return Err(ClobEpochUpdateError::Recoverable(
                    "dynamic_unsubscribe_timeout".to_string(),
                ));
            }
            Err(ClobSendFailure::Transport(error)) => {
                return Err(ClobEpochUpdateError::Recoverable(format!(
                    "dynamic_unsubscribe_failed:{error}"
                )));
            }
        }
    }
    epoch
        .registry
        .retain_markets(desired_markets)
        .map_err(|error| {
            ClobEpochUpdateError::Recoverable(format!("subscription_retention_failed:{error}"))
        })?;
    epoch.pending_resolutions.retain(|market_id, _| {
        desired_markets
            .iter()
            .any(|market| market.market_id == *market_id)
    });
    epoch.markets = desired_markets.to_vec();
    let updated_at = Utc::now();
    if !delta.is_empty() {
        epoch.subscription_stats.updates = epoch.subscription_stats.updates.saturating_add(1);
        epoch.subscription_stats.active_assets = epoch.registry.len();
        epoch.subscription_stats.last_updated_at = Some(updated_at);
    }
    epoch.refresh_private_health(updated_at, Instant::now(), max_book_age);
    Ok(delta)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClobFrameAction {
    Continue,
    Disconnect,
}

async fn apply_active_clob_frame(
    epoch: &mut ClobEpoch,
    message: Message,
    repository: &BtcRepository,
    writer: &mpsc::Sender<PersistItem>,
    state: &Arc<RwLock<RealtimeState>>,
    shared_books: &Arc<RwLock<BookRegistry>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
) -> Result<ClobFrameAction> {
    epoch.watchdog.on_frame(Instant::now());
    let received_at = Utc::now();
    let parsed = match message {
        Message::Text(text) => {
            let pong_like = text.trim().eq_ignore_ascii_case("PONG");
            if epoch.watchdog.acknowledge_text_pong(text.as_str())
                || pong_like
                || text.trim().is_empty()
            {
                return Ok(ClobFrameAction::Continue);
            }
            serde_json::from_str::<serde_json::Value>(&text)
                .context("failed to decode CLOB websocket JSON")
                .and_then(|value| parse_clob_messages(&value))
        }
        Message::Binary(bytes) => serde_json::from_slice::<serde_json::Value>(&bytes)
            .context("failed to decode binary CLOB websocket JSON")
            .and_then(|value| parse_clob_messages(&value)),
        Message::Close(frame) => {
            epoch.session.disconnect_reason = Some(format!("remote_close:{frame:?}"));
            return Ok(ClobFrameAction::Disconnect);
        }
        _ => return Ok(ClobFrameAction::Continue),
    };
    epoch.session.messages_received = epoch.session.messages_received.saturating_add(1);
    {
        let mut runtime_metrics = metrics.write().await;
        runtime_metrics.clob_messages_received =
            runtime_metrics.clob_messages_received.saturating_add(1);
    }
    let messages = match parsed {
        Ok(messages) => messages,
        Err(error) => {
            epoch.registry.quarantine(FeedIntegrityStatus::DecodeError);
            epoch.session.decode_errors = epoch.session.decode_errors.saturating_add(1);
            {
                let mut published_books = shared_books.write().await;
                let mut shared = state.write().await;
                *published_books = epoch.registry.clone();
                shared.update_books(&epoch.registry);
                shared.last_updated_at = Some(received_at);
            }
            {
                let mut runtime_metrics = metrics.write().await;
                runtime_metrics.decode_errors = runtime_metrics.decode_errors.saturating_add(1);
            }
            record_error(metrics, error).await;
            return Ok(ClobFrameAction::Continue);
        }
    };
    let mut events = Vec::new();
    let mut resolutions = Vec::new();
    for message in messages {
        let resolution = persist_official_resolution(repository, &message, received_at, metrics)
            .await
            .context("critical CLOB official-resolution persistence failed")?;
        if let Some(resolution) = resolution {
            resolutions.push(resolution);
        }
        events.extend(epoch.registry.apply(message, received_at));
    }
    let applied_count = events.iter().filter(|event| event.applied).count();
    let integrity_gap_count = events.len().saturating_sub(applied_count);
    epoch.session.integrity_gaps = epoch
        .session
        .integrity_gaps
        .saturating_add(i64::try_from(integrity_gap_count).unwrap_or(i64::MAX));
    {
        let mut runtime_metrics = metrics.write().await;
        runtime_metrics.feed_events_applied = runtime_metrics
            .feed_events_applied
            .saturating_add(u64::try_from(applied_count).unwrap_or(u64::MAX));
        runtime_metrics.integrity_gaps = runtime_metrics
            .integrity_gaps
            .saturating_add(u64::try_from(integrity_gap_count).unwrap_or(u64::MAX));
    }
    let frame_changed = !events.is_empty() || !resolutions.is_empty();
    for event in events {
        if !should_persist_feed_event(&event) {
            continue;
        }
        if enqueue(writer, PersistItem::FeedEvent(event), metrics).await {
            epoch.session.messages_persisted = epoch.session.messages_persisted.saturating_add(1);
        } else {
            epoch.session.dropped_messages = epoch.session.dropped_messages.saturating_add(1);
            bail!("CLOB feed event persistence queue closed");
        }
    }
    if frame_changed {
        let mut published_books = shared_books.write().await;
        let mut shared = state.write().await;
        *published_books = epoch.registry.clone();
        shared.update_books(&epoch.registry);
        shared.last_updated_at = Some(received_at);
        for resolution in resolutions {
            shared.apply_market_resolution(&resolution.market_id, &resolution.winning_token_id);
        }
    }
    Ok(ClobFrameAction::Continue)
}

fn apply_private_clob_frame(epoch: &mut ClobEpoch, message: Message) -> ClobFrameAction {
    epoch.watchdog.on_frame(Instant::now());
    let received_at = Utc::now();
    let parsed = match message {
        Message::Text(text) => {
            let pong_like = text.trim().eq_ignore_ascii_case("PONG");
            if epoch.watchdog.acknowledge_text_pong(text.as_str())
                || pong_like
                || text.trim().is_empty()
            {
                return ClobFrameAction::Continue;
            }
            serde_json::from_str::<serde_json::Value>(&text)
                .context("failed to decode successor CLOB websocket JSON")
                .and_then(|value| parse_clob_messages(&value))
        }
        Message::Binary(bytes) => serde_json::from_slice::<serde_json::Value>(&bytes)
            .context("failed to decode successor binary CLOB websocket JSON")
            .and_then(|value| parse_clob_messages(&value)),
        Message::Close(frame) => {
            epoch.session.disconnect_reason = Some(format!("remote_close:{frame:?}"));
            return ClobFrameAction::Disconnect;
        }
        _ => return ClobFrameAction::Continue,
    };
    epoch.session.messages_received = epoch.session.messages_received.saturating_add(1);
    match parsed {
        Ok(messages) => {
            let mut integrity_gap = false;
            for message in messages {
                let resolution_market_id = if let ClobMessage::MarketResolved {
                    market_id,
                    winning_token_id,
                    ..
                } = &message
                {
                    epoch
                        .markets
                        .iter()
                        .find(|market| {
                            (market.market_id == *market_id || market.condition_id == *market_id)
                                && (market.up_token_id == *winning_token_id
                                    || market.down_token_id == *winning_token_id)
                        })
                        .map(|market| market.market_id.clone())
                } else {
                    None
                };
                if let Some(market_id) = resolution_market_id {
                    match epoch.pending_resolutions.entry(market_id) {
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(BufferedClobResolution {
                                message,
                                received_at,
                            });
                        }
                        std::collections::hash_map::Entry::Occupied(entry) => {
                            if !same_clob_resolution_outcome(&entry.get().message, &message) {
                                epoch.session.integrity_gaps =
                                    epoch.session.integrity_gaps.saturating_add(1);
                                epoch.session.disconnect_reason =
                                    Some("successor_resolution_conflict".to_string());
                                return ClobFrameAction::Disconnect;
                            }
                        }
                    }
                    continue;
                }
                for event in epoch.registry.apply(message, received_at) {
                    if !event.applied {
                        integrity_gap = true;
                        epoch.session.integrity_gaps =
                            epoch.session.integrity_gaps.saturating_add(1);
                    }
                }
            }
            if integrity_gap {
                epoch.session.disconnect_reason = Some("successor_integrity_gap".to_string());
                return ClobFrameAction::Disconnect;
            }
        }
        Err(error) => {
            epoch.registry.quarantine(FeedIntegrityStatus::DecodeError);
            epoch.session.decode_errors = epoch.session.decode_errors.saturating_add(1);
            epoch.session.disconnect_reason = Some("successor_decode_error".to_string());
            tracing::warn!(
                feed = "polymarket_clob_market",
                connection_id = %epoch.connection_id,
                connection_epoch = epoch.connection_epoch,
                reason = %error,
                "private CLOB successor frame failed to decode"
            );
            return ClobFrameAction::Disconnect;
        }
    }
    ClobFrameAction::Continue
}

fn same_clob_resolution_outcome(left: &ClobMessage, right: &ClobMessage) -> bool {
    match (left, right) {
        (
            ClobMessage::MarketResolved {
                winning_token_id: left_token,
                winning_outcome: left_outcome,
                ..
            },
            ClobMessage::MarketResolved {
                winning_token_id: right_token,
                winning_outcome: right_outcome,
                ..
            },
        ) => left_token == right_token && left_outcome == right_outcome,
        _ => false,
    }
}

async fn complete_clob_epoch(
    repository: &BtcRepository,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    epoch: &mut ClobEpoch,
    reason: String,
    cause: ClobDisconnectCause,
    retry_action: ClobRetryAction,
    consecutive_failures: u32,
) -> bool {
    let disconnected_at = Utc::now();
    epoch.session.disconnected_at = Some(disconnected_at);
    epoch.session.disconnect_reason = Some(reason.clone());
    epoch.session.metadata = clob_session_metadata(
        epoch.healthy_epoch,
        cause,
        retry_action,
        consecutive_failures,
        &epoch.subscription_stats,
    );
    log_clob_disconnect(
        epoch.connection_id,
        epoch.connection_epoch,
        Instant::now().duration_since(epoch.connected_instant),
        epoch.healthy_epoch,
        cause,
        retry_action,
        consecutive_failures,
        &reason,
    );
    finish_feed_session_or_fail(repository, &epoch.session, metrics).await
}

fn clob_disconnect_metrics(
    metrics: &mut BtcRuntimeMetrics,
    cause: ClobDisconnectCause,
    retry_action: ClobRetryAction,
) {
    metrics.reconnects = metrics.reconnects.saturating_add(1);
    clob_failure_metrics(metrics, cause);
    clob_retry_metrics(metrics, retry_action);
}

fn clob_candidate_attempt_metrics(
    metrics: &mut BtcRuntimeMetrics,
    cause: ClobDisconnectCause,
    retry_action: ClobRetryAction,
) {
    clob_failure_metrics(metrics, cause);
    clob_retry_metrics(metrics, retry_action);
}

fn clob_failure_metrics(metrics: &mut BtcRuntimeMetrics, cause: ClobDisconnectCause) {
    match cause {
        ClobDisconnectCause::ConnectFailure => {
            metrics.clob_connection_failures = metrics.clob_connection_failures.saturating_add(1);
        }
        ClobDisconnectCause::SubscriptionFailure => {
            metrics.clob_subscription_failures =
                metrics.clob_subscription_failures.saturating_add(1);
        }
        ClobDisconnectCause::BootstrapFailure => {
            metrics.clob_bootstrap_failures = metrics.clob_bootstrap_failures.saturating_add(1);
        }
        ClobDisconnectCause::TransportFailure => {
            metrics.clob_transport_disconnects =
                metrics.clob_transport_disconnects.saturating_add(1);
        }
        ClobDisconnectCause::Shutdown
        | ClobDisconnectCause::MarketWatchClosed
        | ClobDisconnectCause::CriticalPersistence => {}
    }
}

fn clob_retry_metrics(metrics: &mut BtcRuntimeMetrics, retry_action: ClobRetryAction) {
    match retry_action {
        ClobRetryAction::ImmediateRecovery => {
            metrics.clob_immediate_recoveries_scheduled = metrics
                .clob_immediate_recoveries_scheduled
                .saturating_add(1);
        }
        ClobRetryAction::Backoff(delay) => {
            metrics.clob_backoff_scheduled_milliseconds = metrics
                .clob_backoff_scheduled_milliseconds
                .saturating_add(duration_milliseconds(delay));
        }
        ClobRetryAction::Stop => {}
    }
}

#[allow(clippy::too_many_arguments)]
async fn promote_clob_epoch_if_ready(
    active: &mut Option<ClobEpoch>,
    successor: &mut Option<ClobEpoch>,
    desired_markets: &watch::Receiver<Vec<BtcIntervalMarket>>,
    repository: &BtcRepository,
    state: &Arc<RwLock<RealtimeState>>,
    shared_books: &Arc<RwLock<BookRegistry>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    max_book_age: Duration,
    consecutive_failures: &mut u32,
    recovery_window: &mut ClobRecoveryWindow,
) -> Result<bool> {
    if active.is_some() {
        return Ok(false);
    }
    let publication_boundary = Utc::now();
    let desired_before = desired_markets.borrow().clone();
    let Some(publication) = successor
        .as_ref()
        .map(|candidate| {
            prepare_clob_successor_publication(
                candidate,
                &desired_before,
                publication_boundary,
                max_book_age,
            )
        })
        .transpose()?
        .flatten()
    else {
        return Ok(false);
    };
    repository
        .insert_orderbook_checkpoint_pair(
            &publication.checkpoints,
            "websocket_book",
            publication_boundary,
        )
        .await
        .context("failed to persist CLOB successor checkpoint pair")?;

    let checked_after_checkpoint = Utc::now();
    if !successor.as_ref().is_some_and(|candidate| {
        clob_successor_publication_still_valid(
            candidate,
            &desired_markets.borrow(),
            &publication.current_market,
            checked_after_checkpoint,
            max_book_age,
        )
    }) {
        return Ok(false);
    }

    // Ownership transfers before subscription acknowledgement. From this point onward the
    // connection is either published as active or dropped fail-closed; it is never restored to
    // the private successor slot with durable acknowledgement attached to it.
    let mut promoted = successor
        .take()
        .expect("validated CLOB successor remains in its fixed slot");
    let buffered_resolutions = std::mem::take(&mut promoted.pending_resolutions)
        .into_values()
        .collect::<Vec<_>>();
    let checked_before_ack = Utc::now();
    let desired_before_ack = desired_markets.borrow().clone();
    if !clob_successor_publication_still_valid(
        &promoted,
        &desired_before_ack,
        &publication.current_market,
        checked_before_ack,
        max_book_age,
    ) {
        quarantine_published_clob_books(state, shared_books, checked_before_ack).await;
        if !complete_clob_epoch(
            repository,
            metrics,
            &mut promoted,
            "promotion_market_changed_after_checkpoint".to_string(),
            ClobDisconnectCause::SubscriptionFailure,
            ClobRetryAction::ImmediateRecovery,
            0,
        )
        .await
        {
            bail!("failed to finalize rejected CLOB promotion session");
        }
        return Ok(false);
    }
    if let Err(error) = acknowledge_clob_subscriptions(
        repository,
        &desired_before_ack,
        promoted.connection_id,
        checked_before_ack,
    )
    .await
    {
        quarantine_published_clob_books(state, shared_books, Utc::now()).await;
        let _ = complete_clob_epoch(
            repository,
            metrics,
            &mut promoted,
            "critical_promotion_subscription_ack".to_string(),
            ClobDisconnectCause::CriticalPersistence,
            ClobRetryAction::Stop,
            0,
        )
        .await;
        return Err(error).context("failed to acknowledge promoted CLOB successor subscriptions");
    }

    let mut published_books = shared_books.write().await;
    let mut shared = state.write().await;
    let published_at = Utc::now();
    let publication_valid = {
        let desired_at_publication = desired_markets.borrow();
        if clob_successor_publication_still_valid(
            &promoted,
            &desired_at_publication,
            &publication.current_market,
            published_at,
            max_book_age,
        ) {
            *published_books = publication.registry.clone();
            shared.update_books(&publication.registry);
            shared.last_updated_at = Some(published_at);
            true
        } else {
            published_books.quarantine(FeedIntegrityStatus::Stale);
            shared.update_books(&published_books);
            shared.last_updated_at = Some(published_at);
            false
        }
    };
    drop(shared);
    drop(published_books);
    if !publication_valid {
        if !complete_clob_epoch(
            repository,
            metrics,
            &mut promoted,
            "promotion_market_changed_after_ack".to_string(),
            ClobDisconnectCause::SubscriptionFailure,
            ClobRetryAction::ImmediateRecovery,
            0,
        )
        .await
        {
            bail!("failed to finalize post-ack CLOB promotion session");
        }
        return Ok(false);
    }

    promoted.healthy_epoch = true;
    promoted.books_usable = true;
    promoted.subscription_stats.active_assets = promoted.registry.len();
    let promoted_connection_id = promoted.connection_id;
    let promoted_connection_epoch = promoted.connection_epoch;
    let promoted_connected_at = promoted.session.connected_at;
    let promoted_assets = promoted.registry.len();
    *active = Some(promoted);
    {
        let mut runtime_metrics = metrics.write().await;
        runtime_metrics.persistence_items_written =
            runtime_metrics.persistence_items_written.saturating_add(2);
    }
    for buffered in buffered_resolutions {
        match persist_official_resolution(
            repository,
            &buffered.message,
            buffered.received_at,
            metrics,
        )
        .await
        {
            Ok(Some(resolution)) => {
                state
                    .write()
                    .await
                    .apply_market_resolution(&resolution.market_id, &resolution.winning_token_id);
            }
            Ok(None) => {}
            Err(error) => {
                let mut failed = active
                    .take()
                    .expect("published CLOB promotion remains in the active slot");
                quarantine_clob_books_on_disconnect(
                    &mut failed.registry,
                    state,
                    shared_books,
                    Utc::now(),
                )
                .await;
                let _ = complete_clob_epoch(
                    repository,
                    metrics,
                    &mut failed,
                    "critical_buffered_resolution_persistence".to_string(),
                    ClobDisconnectCause::CriticalPersistence,
                    ClobRetryAction::Stop,
                    0,
                )
                .await;
                return Err(error).context("failed to persist buffered CLOB successor resolution");
            }
        }
    }
    let post_flush_checked_at = Utc::now();
    if !active.as_ref().is_some_and(|epoch| {
        clob_successor_publication_still_valid(
            epoch,
            &desired_markets.borrow(),
            &publication.current_market,
            post_flush_checked_at,
            max_book_age,
        )
    }) {
        let mut failed = active
            .take()
            .expect("published CLOB promotion remains in the active slot");
        quarantine_clob_books_on_disconnect(
            &mut failed.registry,
            state,
            shared_books,
            post_flush_checked_at,
        )
        .await;
        if !complete_clob_epoch(
            repository,
            metrics,
            &mut failed,
            "promotion_stale_after_resolution_flush".to_string(),
            ClobDisconnectCause::TransportFailure,
            ClobRetryAction::ImmediateRecovery,
            0,
        )
        .await
        {
            bail!("failed to finalize stale CLOB promotion session");
        }
        return Ok(false);
    }
    record_clob_epoch_healthy(
        metrics,
        promoted_connection_id,
        promoted_connection_epoch,
        published_at,
        Instant::now(),
        true,
        consecutive_failures,
        recovery_window,
    )
    .await;
    {
        let mut runtime_metrics = metrics.write().await;
        runtime_metrics.clob_connected_connection_epoch = Some(promoted_connection_epoch);
        runtime_metrics.clob_connected_connection_id = Some(promoted_connection_id);
        runtime_metrics.clob_last_connected_at = promoted_connected_at;
        runtime_metrics.clob_active_subscribed_assets =
            u64::try_from(promoted_assets).unwrap_or(u64::MAX);
    }
    tracing::info!(
        feed = "polymarket_clob_market",
        connection_id = %promoted_connection_id,
        connection_epoch = promoted_connection_epoch,
        active_assets = promoted_assets,
        "CLOB successor promoted with an atomic ready-book handoff"
    );
    Ok(true)
}

async fn run_clob_supervisor(
    config: BtcRuntimeConfig,
    repository: BtcRepository,
    mut markets: watch::Receiver<Vec<BtcIntervalMarket>>,
    writer: mpsc::Sender<PersistItem>,
    state: Arc<RwLock<RealtimeState>>,
    shared_books: Arc<RwLock<BookRegistry>>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
    mut shutdown: watch::Receiver<bool>,
) {
    if !config.enabled {
        return;
    }
    let max_book_age = chrono_duration(config.max_book_age);
    let mut active: Option<ClobEpoch> = None;
    let mut successor: Option<ClobEpoch> = None;
    let mut connect_task: Option<JoinHandle<ClobConnectOutcome>> = None;
    let mut connection_epoch = 0i32;
    let mut consecutive_failures = 0u32;
    let mut successor_failures = 0u32;
    let mut successor_retry_at = Instant::now();
    let mut desired_connection_targets = markets.borrow().clone();
    let mut connecting_markets: Option<Vec<BtcIntervalMarket>> = None;
    let mut recovery_window = ClobRecoveryWindow::open(Utc::now(), Instant::now());
    metrics.write().await.clob_recovery_unavailable_since = recovery_window.since;
    let started_at = Instant::now();
    let mut heartbeat = interval_at(
        started_at + config.clob_heartbeat_interval,
        config.clob_heartbeat_interval,
    );
    let mut checkpoints = interval_at(
        started_at + config.checkpoint_interval,
        config.checkpoint_interval,
    );
    heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
    checkpoints.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut terminate = false;
    let dormant_deadline = Instant::now() + StdDuration::from_secs(86_400);
    let active_bootstrap_sleep = sleep_until(dormant_deadline);
    let active_pong_sleep = sleep_until(dormant_deadline);
    let active_read_sleep = sleep_until(dormant_deadline);
    let successor_bootstrap_sleep = sleep_until(dormant_deadline);
    let successor_pong_sleep = sleep_until(dormant_deadline);
    let successor_read_sleep = sleep_until(dormant_deadline);
    let retry_sleep = sleep_until(dormant_deadline);
    tokio::pin!(
        active_bootstrap_sleep,
        active_pong_sleep,
        active_read_sleep,
        successor_bootstrap_sleep,
        successor_pong_sleep,
        successor_read_sleep,
        retry_sleep
    );

    while !terminate {
        if *shutdown.borrow() {
            break;
        }
        if active.is_none() {
            match promote_clob_epoch_if_ready(
                &mut active,
                &mut successor,
                &markets,
                &repository,
                &state,
                &shared_books,
                &metrics,
                max_book_age,
                &mut consecutive_failures,
                &mut recovery_window,
            )
            .await
            {
                Ok(true) => {
                    successor_failures = 0;
                    successor_retry_at = Instant::now();
                    continue;
                }
                Ok(false) => {}
                Err(error) => {
                    quarantine_published_clob_books(&state, &shared_books, Utc::now()).await;
                    if let Some(mut failed) = successor.take() {
                        let _ = complete_clob_epoch(
                            &repository,
                            &metrics,
                            &mut failed,
                            "critical_promotion_persistence".to_string(),
                            ClobDisconnectCause::CriticalPersistence,
                            ClobRetryAction::Stop,
                            successor_failures,
                        )
                        .await;
                    }
                    record_critical_persistence_error(&metrics, error).await;
                    return;
                }
            }
        }
        if should_start_clob_successor(
            successor.is_some(),
            connect_task.is_some(),
            !markets.borrow().is_empty(),
            Instant::now(),
            successor_retry_at,
        ) {
            connection_epoch = connection_epoch.saturating_add(1);
            let connection_config = config.clone();
            let desired_markets = markets.borrow().clone();
            connecting_markets = Some(desired_markets.clone());
            let connection_shutdown = shutdown.clone();
            connect_task = Some(tokio::spawn(connect_clob_epoch(
                connection_config,
                desired_markets,
                connection_epoch,
                connection_shutdown,
            )));
        }

        let active_bootstrap_deadline = active
            .as_ref()
            .and_then(|epoch| epoch.watchdog.bootstrap_deadline);
        let active_pong_deadline = active
            .as_ref()
            .and_then(|epoch| epoch.watchdog.pong_deadline);
        let active_read_deadline = active
            .as_ref()
            .map(|epoch| epoch.watchdog.read_idle_deadline);
        let successor_bootstrap_deadline = successor
            .as_ref()
            .and_then(|epoch| epoch.watchdog.bootstrap_deadline);
        let successor_pong_deadline = successor
            .as_ref()
            .and_then(|epoch| epoch.watchdog.pong_deadline);
        let successor_read_deadline = successor
            .as_ref()
            .map(|epoch| epoch.watchdog.read_idle_deadline);
        let retry_deadline =
            (successor.is_none() && connect_task.is_none() && !markets.borrow().is_empty())
                .then_some(successor_retry_at);
        if let Some(deadline) = active_bootstrap_deadline {
            active_bootstrap_sleep.as_mut().reset(deadline);
        }
        if let Some(deadline) = active_pong_deadline {
            active_pong_sleep.as_mut().reset(deadline);
        }
        if let Some(deadline) = active_read_deadline {
            active_read_sleep.as_mut().reset(deadline);
        }
        if let Some(deadline) = successor_bootstrap_deadline {
            successor_bootstrap_sleep.as_mut().reset(deadline);
        }
        if let Some(deadline) = successor_pong_deadline {
            successor_pong_sleep.as_mut().reset(deadline);
        }
        if let Some(deadline) = successor_read_deadline {
            successor_read_sleep.as_mut().reset(deadline);
        }
        if let Some(deadline) = retry_deadline {
            retry_sleep.as_mut().reset(deadline);
        }

        let mut active_failure: Option<(String, ClobDisconnectCause, Option<anyhow::Error>)> = None;
        let mut successor_failure: Option<(String, ClobDisconnectCause)> = None;
        let mut shutdown_requested = false;

        tokio::select! {
            _ = shutdown.changed() => {
                shutdown_requested = true;
            }
            _ = &mut active_bootstrap_sleep, if active_bootstrap_deadline.is_some() => {
                active_failure = Some((
                    "bootstrap_book_timeout".to_string(),
                    ClobDisconnectCause::BootstrapFailure,
                    None,
                ));
            }
            _ = &mut active_pong_sleep, if active_pong_deadline.is_some() => {
                let cause = if active.as_ref().is_some_and(|epoch| epoch.healthy_epoch) {
                    ClobDisconnectCause::TransportFailure
                } else {
                    ClobDisconnectCause::BootstrapFailure
                };
                active_failure = Some(("heartbeat_ack_timeout".to_string(), cause, None));
            }
            _ = &mut active_read_sleep, if active_read_deadline.is_some() => {
                let cause = if active.as_ref().is_some_and(|epoch| epoch.healthy_epoch) {
                    ClobDisconnectCause::TransportFailure
                } else {
                    ClobDisconnectCause::BootstrapFailure
                };
                active_failure = Some(("read_idle_timeout".to_string(), cause, None));
            }
            _ = &mut successor_bootstrap_sleep, if successor_bootstrap_deadline.is_some() => {
                successor_failure = Some((
                    "bootstrap_book_timeout".to_string(),
                    ClobDisconnectCause::BootstrapFailure,
                ));
            }
            _ = &mut successor_pong_sleep, if successor_pong_deadline.is_some() => {
                let cause = if successor.as_ref().is_some_and(|epoch| epoch.healthy_epoch) {
                    ClobDisconnectCause::TransportFailure
                } else {
                    ClobDisconnectCause::BootstrapFailure
                };
                successor_failure = Some(("heartbeat_ack_timeout".to_string(), cause));
            }
            _ = &mut successor_read_sleep, if successor_read_deadline.is_some() => {
                let cause = if successor.as_ref().is_some_and(|epoch| epoch.healthy_epoch) {
                    ClobDisconnectCause::TransportFailure
                } else {
                    ClobDisconnectCause::BootstrapFailure
                };
                successor_failure = Some(("read_idle_timeout".to_string(), cause));
            }
            changed = markets.changed() => {
                match market_watch_disposition(&changed) {
                    MarketWatchDisposition::Terminate => terminate = true,
                    MarketWatchDisposition::UpdateSubscriptions => {
                        let desired_markets = markets.borrow().clone();
                        if !same_market_subscriptions(
                            &desired_connection_targets,
                            &desired_markets,
                        ) {
                            desired_connection_targets = desired_markets.clone();
                            successor_failures = 0;
                            successor_retry_at = Instant::now();
                            if connecting_markets.as_ref().is_some_and(|connecting| {
                                !same_market_subscriptions(connecting, &desired_markets)
                            }) {
                                if let Some(task) = connect_task.take() {
                                    task.abort();
                                    let _ = task.await;
                                }
                                connecting_markets = None;
                            }
                        }
                        if let Some(epoch) = active.as_mut() {
                            match update_clob_epoch_subscriptions(
                                epoch,
                                &desired_markets,
                                Some(&repository),
                                &mut shutdown,
                                max_book_age,
                            )
                            .await
                            {
                                Ok(delta) => {
                                    let updated_at = Utc::now();
                                    {
                                        let mut published_books = shared_books.write().await;
                                        let mut shared = state.write().await;
                                        *published_books = epoch.registry.clone();
                                        shared.update_books(&epoch.registry);
                                        shared.last_updated_at = Some(updated_at);
                                    }
                                    if !delta.is_empty() {
                                        let mut runtime_metrics = metrics.write().await;
                                        runtime_metrics.clob_subscription_updates = runtime_metrics
                                            .clob_subscription_updates
                                            .saturating_add(1);
                                        runtime_metrics.clob_active_subscribed_assets =
                                            u64::try_from(epoch.registry.len()).unwrap_or(u64::MAX);
                                        runtime_metrics.clob_last_subscription_update_at = Some(updated_at);
                                    }
                                }
                                Err(ClobEpochUpdateError::Shutdown) => shutdown_requested = true,
                                Err(ClobEpochUpdateError::Recoverable(reason)) => {
                                    active_failure = Some((
                                        reason,
                                        ClobDisconnectCause::SubscriptionFailure,
                                        None,
                                    ));
                                }
                                Err(ClobEpochUpdateError::Critical(error)) => {
                                    active_failure = Some((
                                        "critical_subscription_ack".to_string(),
                                        ClobDisconnectCause::CriticalPersistence,
                                        Some(error),
                                    ));
                                }
                            }
                        }
                        if let Some(epoch) = successor.as_mut() {
                            match update_clob_epoch_subscriptions(
                                epoch,
                                &desired_markets,
                                None,
                                &mut shutdown,
                                max_book_age,
                            )
                            .await
                            {
                                Ok(_) => {}
                                Err(ClobEpochUpdateError::Shutdown) => shutdown_requested = true,
                                Err(ClobEpochUpdateError::Recoverable(reason)) => {
                                    successor_failure = Some((
                                        reason,
                                        ClobDisconnectCause::SubscriptionFailure,
                                    ));
                                }
                                Err(ClobEpochUpdateError::Critical(_)) => {
                                    unreachable!("private CLOB successors never acknowledge subscriptions")
                                }
                            }
                        }
                    }
                }
            }
            message = async {
                active
                    .as_mut()
                    .expect("active CLOB socket branch is guarded")
                    .socket
                    .next()
                    .await
            }, if active.is_some() => {
                match message {
                    None => {
                        let cause = clob_disconnect_cause(
                            active
                                .as_ref()
                                .is_some_and(|epoch| epoch.healthy_epoch),
                            false,
                            false,
                            false,
                            false,
                        );
                        active_failure = Some(("websocket_eof".to_string(), cause, None));
                    }
                    Some(Err(error)) => {
                        let cause = if active.as_ref().is_some_and(|epoch| epoch.healthy_epoch) {
                            ClobDisconnectCause::TransportFailure
                        } else {
                            ClobDisconnectCause::BootstrapFailure
                        };
                        active_failure = Some((
                            format!("transport_read_failed:{error}"),
                            cause,
                            None,
                        ));
                    }
                    Some(Ok(message)) => {
                        let epoch = active.as_mut().expect("active epoch remains installed");
                        match apply_active_clob_frame(
                            epoch,
                            message,
                            &repository,
                            &writer,
                            &state,
                            &shared_books,
                            &metrics,
                        )
                        .await
                        {
                            Ok(ClobFrameAction::Disconnect) => {
                                active_failure = Some((
                                    epoch
                                        .session
                                        .disconnect_reason
                                        .clone()
                                        .unwrap_or_else(|| "remote_close".to_string()),
                                    if epoch.healthy_epoch {
                                        ClobDisconnectCause::TransportFailure
                                    } else {
                                        ClobDisconnectCause::BootstrapFailure
                                    },
                                    None,
                                ));
                            }
                            Ok(ClobFrameAction::Continue) => {
                                let checked_at = Utc::now();
                                epoch.watchdog.refresh_bootstrap(
                                    Instant::now(),
                                    &epoch.registry,
                                    &epoch.markets,
                                    checked_at,
                                    max_book_age,
                                );
                                update_clob_usability(
                                    &epoch.registry,
                                    &epoch.markets,
                                    checked_at,
                                    Instant::now(),
                                    max_book_age,
                                    &mut epoch.books_usable,
                                    &mut epoch.healthy_epoch,
                                    &metrics,
                                    epoch.connection_id,
                                    epoch.connection_epoch,
                                    &mut consecutive_failures,
                                    &mut recovery_window,
                                )
                                .await;
                                if !epoch.books_usable
                                    && successor.as_ref().is_some_and(|candidate| {
                                        clob_successor_ready(
                                            candidate,
                                            &markets.borrow(),
                                            checked_at,
                                            max_book_age,
                                        )
                                    })
                                {
                                    active_failure = Some((
                                        "active_book_stale".to_string(),
                                        ClobDisconnectCause::TransportFailure,
                                        None,
                                    ));
                                }
                            }
                            Err(error) => {
                                active_failure = Some((
                                    "critical_clob_persistence".to_string(),
                                    ClobDisconnectCause::CriticalPersistence,
                                    Some(error),
                                ));
                            }
                        }
                    }
                }
            }
            message = async {
                successor
                    .as_mut()
                    .expect("successor CLOB socket branch is guarded")
                    .socket
                    .next()
                    .await
            }, if successor.is_some() => {
                match message {
                    None => {
                        let cause = if successor.as_ref().is_some_and(|epoch| epoch.healthy_epoch) {
                            ClobDisconnectCause::TransportFailure
                        } else {
                            ClobDisconnectCause::BootstrapFailure
                        };
                        successor_failure = Some(("websocket_eof".to_string(), cause));
                    }
                    Some(Err(error)) => {
                        let cause = if successor.as_ref().is_some_and(|epoch| epoch.healthy_epoch) {
                            ClobDisconnectCause::TransportFailure
                        } else {
                            ClobDisconnectCause::BootstrapFailure
                        };
                        successor_failure = Some((
                            format!("transport_read_failed:{error}"),
                            cause,
                        ));
                    }
                    Some(Ok(message)) => {
                        let epoch = successor
                            .as_mut()
                            .expect("successor epoch remains installed");
                        if apply_private_clob_frame(epoch, message) == ClobFrameAction::Disconnect {
                            successor_failure = Some((
                                epoch
                                    .session
                                    .disconnect_reason
                                    .clone()
                                    .unwrap_or_else(|| "remote_close".to_string()),
                                if epoch.healthy_epoch {
                                    ClobDisconnectCause::TransportFailure
                                } else {
                                    ClobDisconnectCause::BootstrapFailure
                                },
                            ));
                        } else {
                            epoch.refresh_private_health(Utc::now(), Instant::now(), max_book_age);
                            if epoch.books_usable {
                                successor_failures = 0;
                                successor_retry_at = Instant::now();
                            }
                        }
                    }
                }
            }
            outcome = async {
                connect_task
                    .as_mut()
                    .expect("CLOB connect task branch is guarded")
                    .await
            }, if connect_task.is_some() => {
                connect_task = None;
                connecting_markets = None;
                match outcome {
                    Err(error) => {
                        successor_failures = successor_failures.saturating_add(1);
                        let delay = reconnect_backoff(&config, successor_failures);
                        successor_retry_at = Instant::now() + delay;
                        record_error(
                            &metrics,
                            anyhow::anyhow!("CLOB connection task failed: {error}"),
                        )
                        .await;
                        {
                            let mut runtime_metrics = metrics.write().await;
                            clob_candidate_attempt_metrics(
                                &mut runtime_metrics,
                                ClobDisconnectCause::ConnectFailure,
                                ClobRetryAction::Backoff(delay),
                            );
                        }
                    }
                    Ok(ClobConnectOutcome::Failed { mut session, reason, kind }) => {
                        if kind == ClobConnectFailureKind::Shutdown {
                            shutdown_requested = true;
                        } else {
                            successor_failures = successor_failures.saturating_add(1);
                            let delay = reconnect_backoff(&config, successor_failures);
                            successor_retry_at = Instant::now() + delay;
                            let cause = match kind {
                                ClobConnectFailureKind::Connect => ClobDisconnectCause::ConnectFailure,
                                ClobConnectFailureKind::Subscription => {
                                    ClobDisconnectCause::SubscriptionFailure
                                }
                                ClobConnectFailureKind::Identity => {
                                    ClobDisconnectCause::BootstrapFailure
                                }
                                ClobConnectFailureKind::Shutdown => unreachable!(),
                            };
                            session.metadata = clob_session_metadata(
                                false,
                                cause,
                                ClobRetryAction::Backoff(delay),
                                successor_failures,
                                &ClobSubscriptionStats::default(),
                            );
                            if start_feed_session_or_fail(&repository, &session, &metrics).await {
                                let _ =
                                    finish_feed_session_or_fail(&repository, &session, &metrics)
                                        .await;
                            }
                            let mut runtime_metrics = metrics.write().await;
                            clob_candidate_attempt_metrics(
                                &mut runtime_metrics,
                                cause,
                                ClobRetryAction::Backoff(delay),
                            );
                            tracing::warn!(
                                feed = "polymarket_clob_market",
                                %reason,
                                successor_failures,
                                retry_delay_ms = duration_milliseconds(delay),
                                active_present = active.is_some(),
                                "CLOB successor connection attempt failed"
                            );
                        }
                    }
                    Ok(ClobConnectOutcome::Connected { mut epoch, connect_latency }) => {
                        epoch.subscription_stats.active_assets = epoch.registry.len();
                        if !start_feed_session_or_fail(&repository, &epoch.session, &metrics).await {
                            successor_failures = successor_failures.saturating_add(1);
                            let delay = reconnect_backoff(&config, successor_failures);
                            successor_retry_at = Instant::now() + delay;
                            {
                                let mut runtime_metrics = metrics.write().await;
                                clob_retry_metrics(
                                    &mut runtime_metrics,
                                    ClobRetryAction::Backoff(delay),
                                );
                            }
                        } else if !same_market_subscriptions(&epoch.markets, &markets.borrow()) {
                            let _ = complete_clob_epoch(
                                &repository,
                                &metrics,
                                &mut epoch,
                                "subscription_target_changed_during_connect".to_string(),
                                ClobDisconnectCause::SubscriptionFailure,
                                ClobRetryAction::ImmediateRecovery,
                                successor_failures,
                            )
                            .await;
                            successor_failures = 0;
                            successor_retry_at = Instant::now();
                        } else {
                            {
                                let mut runtime_metrics = metrics.write().await;
                                runtime_metrics.clob_connections_established = runtime_metrics
                                    .clob_connections_established
                                    .saturating_add(1);
                            }
                            tracing::info!(
                                feed = "polymarket_clob_market",
                                connection_id = %epoch.connection_id,
                                connection_epoch = epoch.connection_epoch,
                                connect_latency_ms = duration_milliseconds(connect_latency),
                                active_present = active.is_some(),
                                "private CLOB successor connected"
                            );
                            successor = Some(epoch);
                        }
                    }
                }
            }
            _ = checkpoints.tick() => {
                let checked_at = Utc::now();
                if let Some(epoch) = successor.as_mut() {
                    epoch.refresh_private_health(checked_at, Instant::now(), max_book_age);
                    if epoch.books_usable {
                        successor_failures = 0;
                        successor_retry_at = Instant::now();
                    }
                }
                if let Some(epoch) = active.as_mut() {
                    epoch.watchdog.refresh_bootstrap(
                        Instant::now(),
                        &epoch.registry,
                        &epoch.markets,
                        checked_at,
                        max_book_age,
                    );
                    update_clob_usability(
                        &epoch.registry,
                        &epoch.markets,
                        checked_at,
                        Instant::now(),
                        max_book_age,
                        &mut epoch.books_usable,
                        &mut epoch.healthy_epoch,
                        &metrics,
                        epoch.connection_id,
                        epoch.connection_epoch,
                        &mut consecutive_failures,
                        &mut recovery_window,
                    )
                    .await;
                    if !epoch.books_usable
                        && successor.as_ref().is_some_and(|candidate| {
                            clob_successor_ready(
                                candidate,
                                &markets.borrow(),
                                checked_at,
                                max_book_age,
                            )
                        })
                    {
                        active_failure = Some((
                            "active_book_stale".to_string(),
                            ClobDisconnectCause::TransportFailure,
                            None,
                        ));
                    } else {
                        for market in &epoch.markets {
                            if !market.is_trade_window(checked_at) {
                                continue;
                            }
                            for token_id in [&market.up_token_id, &market.down_token_id] {
                                if let Some(checkpoint) = epoch.registry.checkpoint(token_id) {
                                    if enqueue(&writer, PersistItem::Checkpoint(checkpoint), &metrics).await {
                                        let mut runtime_metrics = metrics.write().await;
                                        runtime_metrics.checkpoints_queued = runtime_metrics
                                            .checkpoints_queued
                                            .saturating_add(1);
                                    } else {
                                        epoch.session.dropped_messages =
                                            epoch.session.dropped_messages.saturating_add(1);
                                        active_failure = Some((
                                            "critical_checkpoint_queue_closed".to_string(),
                                            ClobDisconnectCause::CriticalPersistence,
                                            Some(anyhow::anyhow!(
                                                "CLOB checkpoint persistence queue closed"
                                            )),
                                        ));
                                        break;
                                    }
                                }
                            }
                            if active_failure.is_some() {
                                break;
                            }
                        }
                    }
                }
            }
            _ = heartbeat.tick() => {
                if let Some(epoch) = active.as_mut() {
                    if !epoch.watchdog.awaiting_text_pong {
                        match send_clob_text(&mut epoch.socket, "PING".to_string(), &mut shutdown).await {
                            Ok(()) => epoch.watchdog.arm_text_pong(Instant::now()),
                            Err(ClobSendFailure::Shutdown) => shutdown_requested = true,
                            Err(ClobSendFailure::Timeout) => {
                                active_failure = Some((
                                    "heartbeat_send_timeout".to_string(),
                                    if epoch.healthy_epoch {
                                        ClobDisconnectCause::TransportFailure
                                    } else {
                                        ClobDisconnectCause::BootstrapFailure
                                    },
                                    None,
                                ));
                            }
                            Err(ClobSendFailure::Transport(error)) => {
                                active_failure = Some((
                                    format!("heartbeat_send_failed:{error}"),
                                    if epoch.healthy_epoch {
                                        ClobDisconnectCause::TransportFailure
                                    } else {
                                        ClobDisconnectCause::BootstrapFailure
                                    },
                                    None,
                                ));
                            }
                        }
                    }
                }
                if let Some(epoch) = successor.as_mut() {
                    if !epoch.watchdog.awaiting_text_pong {
                        match send_clob_text(&mut epoch.socket, "PING".to_string(), &mut shutdown).await {
                            Ok(()) => epoch.watchdog.arm_text_pong(Instant::now()),
                            Err(ClobSendFailure::Shutdown) => shutdown_requested = true,
                            Err(ClobSendFailure::Timeout) => {
                                successor_failure = Some((
                                    "heartbeat_send_timeout".to_string(),
                                    if epoch.healthy_epoch {
                                        ClobDisconnectCause::TransportFailure
                                    } else {
                                        ClobDisconnectCause::BootstrapFailure
                                    },
                                ));
                            }
                            Err(ClobSendFailure::Transport(error)) => {
                                successor_failure = Some((
                                    format!("heartbeat_send_failed:{error}"),
                                    if epoch.healthy_epoch {
                                        ClobDisconnectCause::TransportFailure
                                    } else {
                                        ClobDisconnectCause::BootstrapFailure
                                    },
                                ));
                            }
                        }
                    }
                }
            }
            _ = &mut retry_sleep, if retry_deadline.is_some() => {}
        }

        if shutdown_requested || terminate {
            break;
        }

        if let Some((reason, cause)) = successor_failure {
            if let Some(mut failed) = successor.take() {
                let retry_action = clob_candidate_retry_action(
                    &config,
                    failed.healthy_epoch,
                    &mut successor_failures,
                );
                successor_retry_at = match retry_action {
                    ClobRetryAction::ImmediateRecovery => Instant::now(),
                    ClobRetryAction::Backoff(delay) => Instant::now() + delay,
                    ClobRetryAction::Stop => unreachable!("failed successors always retry"),
                };
                let _ = complete_clob_epoch(
                    &repository,
                    &metrics,
                    &mut failed,
                    reason.clone(),
                    cause,
                    retry_action,
                    successor_failures,
                )
                .await;
                let mut runtime_metrics = metrics.write().await;
                clob_candidate_attempt_metrics(&mut runtime_metrics, cause, retry_action);
                tracing::warn!(
                    feed = "polymarket_clob_market",
                    %reason,
                    successor_failures,
                    immediate_retry = matches!(retry_action, ClobRetryAction::ImmediateRecovery),
                    active_present = active.is_some(),
                    "private CLOB successor retired"
                );
            }
        }

        if let Some((reason, cause, fatal_error)) = active_failure {
            let Some(mut failed) = active.take() else {
                continue;
            };
            let retry_action = clob_retry_action(
                &config,
                failed.healthy_epoch,
                cause == ClobDisconnectCause::CriticalPersistence,
                &mut consecutive_failures,
            );
            {
                let mut runtime_metrics = metrics.write().await;
                clob_disconnect_metrics(&mut runtime_metrics, cause, retry_action);
                runtime_metrics.clob_last_disconnect_at = Some(Utc::now());
                runtime_metrics.clob_last_disconnect_reason = Some(reason.clone());
            }
            if let Some(error) = fatal_error {
                quarantine_clob_books_on_disconnect(
                    &mut failed.registry,
                    &state,
                    &shared_books,
                    Utc::now(),
                )
                .await;
                let _ = complete_clob_epoch(
                    &repository,
                    &metrics,
                    &mut failed,
                    reason,
                    cause,
                    retry_action,
                    consecutive_failures,
                )
                .await;
                record_critical_persistence_error(&metrics, error).await;
                return;
            }
            match promote_clob_epoch_if_ready(
                &mut active,
                &mut successor,
                &markets,
                &repository,
                &state,
                &shared_books,
                &metrics,
                max_book_age,
                &mut consecutive_failures,
                &mut recovery_window,
            )
            .await
            {
                Ok(true) => {
                    successor_failures = 0;
                    successor_retry_at = Instant::now();
                    let _ = complete_clob_epoch(
                        &repository,
                        &metrics,
                        &mut failed,
                        reason,
                        cause,
                        retry_action,
                        consecutive_failures,
                    )
                    .await;
                }
                Ok(false) => {
                    let unavailable_at = Utc::now();
                    let unavailable_instant = Instant::now();
                    quarantine_clob_books_on_disconnect(
                        &mut failed.registry,
                        &state,
                        &shared_books,
                        unavailable_at,
                    )
                    .await;
                    record_clob_epoch_unavailable(
                        &metrics,
                        unavailable_at,
                        unavailable_instant,
                        &mut recovery_window,
                    )
                    .await;
                    {
                        let mut runtime_metrics = metrics.write().await;
                        clear_clob_connection_metrics(
                            &mut runtime_metrics,
                            unavailable_at,
                            &reason,
                            recovery_window.since,
                            consecutive_failures,
                        );
                    }
                    if successor.is_none() && connect_task.is_none() {
                        successor_failures = 0;
                        successor_retry_at = Instant::now();
                    }
                    let _ = complete_clob_epoch(
                        &repository,
                        &metrics,
                        &mut failed,
                        reason,
                        cause,
                        retry_action,
                        consecutive_failures,
                    )
                    .await;
                }
                Err(error) => {
                    quarantine_clob_books_on_disconnect(
                        &mut failed.registry,
                        &state,
                        &shared_books,
                        Utc::now(),
                    )
                    .await;
                    let _ = complete_clob_epoch(
                        &repository,
                        &metrics,
                        &mut failed,
                        reason,
                        cause,
                        retry_action,
                        consecutive_failures,
                    )
                    .await;
                    if let Some(mut failed_successor) = successor.take() {
                        let _ = complete_clob_epoch(
                            &repository,
                            &metrics,
                            &mut failed_successor,
                            "critical_promotion_persistence".to_string(),
                            ClobDisconnectCause::CriticalPersistence,
                            ClobRetryAction::Stop,
                            successor_failures,
                        )
                        .await;
                    }
                    record_critical_persistence_error(&metrics, error).await;
                    return;
                }
            }
        }
    }

    if let Some(task) = connect_task.take() {
        task.abort();
        let _ = task.await;
    }
    let stop_cause = if *shutdown.borrow() {
        ClobDisconnectCause::Shutdown
    } else {
        ClobDisconnectCause::MarketWatchClosed
    };
    let stop_reason = match stop_cause {
        ClobDisconnectCause::Shutdown => "shutdown",
        ClobDisconnectCause::MarketWatchClosed => "market_watch_closed",
        _ => unreachable!("CLOB supervisor only stops for shutdown or a closed market watch"),
    };
    if let Some(mut epoch) = active.take() {
        quarantine_clob_books_on_disconnect(&mut epoch.registry, &state, &shared_books, Utc::now())
            .await;
        let _ = complete_clob_epoch(
            &repository,
            &metrics,
            &mut epoch,
            stop_reason.to_string(),
            stop_cause,
            ClobRetryAction::Stop,
            consecutive_failures,
        )
        .await;
    }
    if let Some(mut epoch) = successor.take() {
        let _ = complete_clob_epoch(
            &repository,
            &metrics,
            &mut epoch,
            stop_reason.to_string(),
            stop_cause,
            ClobRetryAction::Stop,
            successor_failures,
        )
        .await;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarketWatchDisposition {
    UpdateSubscriptions,
    Terminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClobRetryAction {
    Stop,
    ImmediateRecovery,
    Backoff(StdDuration),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClobDisconnectCause {
    ConnectFailure,
    SubscriptionFailure,
    BootstrapFailure,
    TransportFailure,
    Shutdown,
    MarketWatchClosed,
    CriticalPersistence,
}

impl ClobDisconnectCause {
    fn as_str(self) -> &'static str {
        match self {
            Self::ConnectFailure => "connect_failure",
            Self::SubscriptionFailure => "subscription_failure",
            Self::BootstrapFailure => "bootstrap_failure",
            Self::TransportFailure => "transport_failure",
            Self::Shutdown => "shutdown",
            Self::MarketWatchClosed => "market_watch_closed",
            Self::CriticalPersistence => "critical_persistence",
        }
    }
}

fn clob_disconnect_cause(
    healthy_epoch: bool,
    subscription_failed: bool,
    market_watch_closed: bool,
    shutting_down: bool,
    fatal_persistence_error: bool,
) -> ClobDisconnectCause {
    if fatal_persistence_error {
        ClobDisconnectCause::CriticalPersistence
    } else if shutting_down {
        ClobDisconnectCause::Shutdown
    } else if market_watch_closed {
        ClobDisconnectCause::MarketWatchClosed
    } else if subscription_failed {
        ClobDisconnectCause::SubscriptionFailure
    } else if healthy_epoch {
        ClobDisconnectCause::TransportFailure
    } else {
        ClobDisconnectCause::BootstrapFailure
    }
}

fn clob_retry_action(
    config: &BtcRuntimeConfig,
    healthy_epoch: bool,
    stop: bool,
    consecutive_failures: &mut u32,
) -> ClobRetryAction {
    if stop {
        ClobRetryAction::Stop
    } else if healthy_epoch {
        ClobRetryAction::ImmediateRecovery
    } else {
        *consecutive_failures = consecutive_failures.saturating_add(1);
        ClobRetryAction::Backoff(reconnect_backoff(config, *consecutive_failures))
    }
}

fn clob_candidate_retry_action(
    config: &BtcRuntimeConfig,
    healthy_epoch: bool,
    consecutive_failures: &mut u32,
) -> ClobRetryAction {
    if healthy_epoch {
        *consecutive_failures = 0;
        ClobRetryAction::ImmediateRecovery
    } else {
        *consecutive_failures = consecutive_failures.saturating_add(1);
        ClobRetryAction::Backoff(reconnect_backoff(config, *consecutive_failures))
    }
}

fn market_watch_disposition(
    changed: &Result<(), watch::error::RecvError>,
) -> MarketWatchDisposition {
    if changed.is_ok() {
        MarketWatchDisposition::UpdateSubscriptions
    } else {
        MarketWatchDisposition::Terminate
    }
}

async fn quarantine_clob_books_on_disconnect(
    registry: &mut BookRegistry,
    state: &Arc<RwLock<RealtimeState>>,
    shared_books: &Arc<RwLock<BookRegistry>>,
    disconnected_at: DateTime<Utc>,
) {
    registry.quarantine(FeedIntegrityStatus::Stale);
    let mut published_books = shared_books.write().await;
    let mut shared = state.write().await;
    *published_books = registry.clone();
    shared.update_books(registry);
    shared.last_updated_at = Some(disconnected_at);
}

async fn quarantine_published_clob_books(
    state: &Arc<RwLock<RealtimeState>>,
    shared_books: &Arc<RwLock<BookRegistry>>,
    disconnected_at: DateTime<Utc>,
) {
    let mut published_books = shared_books.write().await;
    published_books.quarantine(FeedIntegrityStatus::Stale);
    let mut shared = state.write().await;
    shared.update_books(&published_books);
    shared.last_updated_at = Some(disconnected_at);
}

fn clob_epoch_ready(
    registry: &BookRegistry,
    markets: &[BtcIntervalMarket],
    now: DateTime<Utc>,
    max_age: Duration,
) -> bool {
    unique_current_clob_market(markets, now)
        .is_some_and(|market| registry.market_books_ready(market, now, max_age))
}

fn should_start_clob_successor(
    successor_present: bool,
    connect_in_flight: bool,
    has_desired_markets: bool,
    now: Instant,
    retry_at: Instant,
) -> bool {
    !successor_present && !connect_in_flight && has_desired_markets && now >= retry_at
}

#[cfg(test)]
fn clob_subscription_identities(markets: &[BtcIntervalMarket]) -> Vec<ClobMarketIdentity> {
    let mut identities = markets
        .iter()
        .map(ClobMarketIdentity::from)
        .collect::<Vec<_>>();
    identities.sort();
    identities.dedup();
    identities
}

#[cfg(test)]
fn same_clob_subscription_identity(
    left: &[BtcIntervalMarket],
    right: &[BtcIntervalMarket],
) -> bool {
    clob_subscription_identities(left) == clob_subscription_identities(right)
}

fn clob_successor_ready(
    successor: &ClobEpoch,
    desired_markets: &[BtcIntervalMarket],
    checked_at: DateTime<Utc>,
    max_book_age: Duration,
) -> bool {
    successor.connection_id == successor.registry.connection_id()
        && same_market_subscriptions(&successor.markets, desired_markets)
        && clob_epoch_ready(
            &successor.registry,
            desired_markets,
            checked_at,
            max_book_age,
        )
}

#[derive(Debug)]
struct ClobSuccessorPublication {
    current_market: ClobMarketIdentity,
    registry: BookRegistry,
    checkpoints: [OrderbookCheckpoint; 2],
}

fn prepare_clob_successor_publication(
    successor: &ClobEpoch,
    desired_markets: &[BtcIntervalMarket],
    publication_boundary: DateTime<Utc>,
    max_book_age: Duration,
) -> Result<Option<ClobSuccessorPublication>> {
    if !clob_successor_ready(
        successor,
        desired_markets,
        publication_boundary,
        max_book_age,
    ) {
        return Ok(None);
    }
    let Some(current_market) = unique_current_clob_market(desired_markets, publication_boundary)
    else {
        return Ok(None);
    };
    let frozen_registry = successor.registry.clone();
    let checkpoints = [
        frozen_registry
            .checkpoint(&current_market.up_token_id)
            .context("ready CLOB successor is missing its up checkpoint")?,
        frozen_registry
            .checkpoint(&current_market.down_token_id)
            .context("ready CLOB successor is missing its down checkpoint")?,
    ];
    if checkpoints.iter().any(|checkpoint| {
        checkpoint.connection_id != successor.connection_id
            || checkpoint.market_id != current_market.market_id
            || checkpoint.source_timestamp > publication_boundary
            || checkpoint.received_at > publication_boundary
    }) {
        return Ok(None);
    }
    if checkpoints[0].token_id != current_market.up_token_id
        || checkpoints[1].token_id != current_market.down_token_id
    {
        return Ok(None);
    }
    Ok(Some(ClobSuccessorPublication {
        current_market: ClobMarketIdentity::from(current_market),
        registry: frozen_registry,
        checkpoints,
    }))
}

fn clob_successor_publication_still_valid(
    successor: &ClobEpoch,
    desired_markets: &[BtcIntervalMarket],
    expected_current_market: &ClobMarketIdentity,
    checked_at: DateTime<Utc>,
    max_book_age: Duration,
) -> bool {
    clob_successor_ready(successor, desired_markets, checked_at, max_book_age)
        && unique_current_clob_market(desired_markets, checked_at)
            .is_some_and(|market| expected_current_market.matches(market))
}

fn unique_current_clob_market(
    markets: &[BtcIntervalMarket],
    now: DateTime<Utc>,
) -> Option<&BtcIntervalMarket> {
    let mut current = None;
    for market in markets.iter().filter(|market| market.is_trade_window(now)) {
        match current {
            None => current = Some(market),
            Some(existing)
                if existing.market_id == market.market_id
                    && existing.condition_id == market.condition_id
                    && existing.up_token_id == market.up_token_id
                    && existing.down_token_id == market.down_token_id => {}
            Some(_) => return None,
        }
    }
    current
}

async fn record_clob_epoch_healthy(
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    connection_id: Uuid,
    connection_epoch: i32,
    ready_at: DateTime<Utc>,
    ready_instant: Instant,
    first_healthy_transition: bool,
    consecutive_failures: &mut u32,
    recovery_window: &mut ClobRecoveryWindow,
) {
    *consecutive_failures = 0;
    let unavailable_milliseconds = recovery_window.close(ready_instant);
    {
        let mut runtime_metrics = metrics.write().await;
        if first_healthy_transition {
            runtime_metrics.clob_healthy_connections =
                runtime_metrics.clob_healthy_connections.saturating_add(1);
        }
        runtime_metrics.clob_recovery_unavailable_milliseconds = runtime_metrics
            .clob_recovery_unavailable_milliseconds
            .saturating_add(unavailable_milliseconds);
        runtime_metrics.clob_consecutive_failures = 0;
        runtime_metrics.clob_active_connection_epoch = Some(connection_epoch);
        runtime_metrics.clob_active_connection_id = Some(connection_id);
        runtime_metrics.clob_last_healthy_at = Some(ready_at);
        runtime_metrics.clob_recovery_unavailable_since = None;
    }
    tracing::info!(
        feed = "polymarket_clob_market",
        %connection_id,
        connection_epoch,
        unavailable_duration_ms = unavailable_milliseconds,
        "CLOB orderbooks became usable"
    );
}

async fn record_clob_epoch_unavailable(
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    unavailable_at: DateTime<Utc>,
    unavailable_instant: Instant,
    recovery_window: &mut ClobRecoveryWindow,
) {
    recovery_window.open_if_closed(unavailable_at, unavailable_instant);
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.clob_active_connection_epoch = None;
    runtime_metrics.clob_active_connection_id = None;
    runtime_metrics.clob_recovery_unavailable_since = recovery_window.since;
}

#[allow(clippy::too_many_arguments)]
async fn update_clob_usability(
    registry: &BookRegistry,
    active_markets: &[BtcIntervalMarket],
    checked_at: DateTime<Utc>,
    checked_instant: Instant,
    max_book_age: Duration,
    books_usable: &mut bool,
    healthy_epoch: &mut bool,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    connection_id: Uuid,
    connection_epoch: i32,
    consecutive_failures: &mut u32,
    recovery_window: &mut ClobRecoveryWindow,
) {
    let ready_now = clob_epoch_ready(registry, active_markets, checked_at, max_book_age);
    if ready_now && !*books_usable {
        let first_healthy_transition = !*healthy_epoch;
        *books_usable = true;
        *healthy_epoch = true;
        record_clob_epoch_healthy(
            metrics,
            connection_id,
            connection_epoch,
            checked_at,
            checked_instant,
            first_healthy_transition,
            consecutive_failures,
            recovery_window,
        )
        .await;
    } else if !ready_now && *books_usable {
        *books_usable = false;
        record_clob_epoch_unavailable(metrics, checked_at, checked_instant, recovery_window).await;
    }
}

fn clear_clob_connection_metrics(
    metrics: &mut BtcRuntimeMetrics,
    disconnected_at: DateTime<Utc>,
    reason: &str,
    recovery_unavailable_since: Option<DateTime<Utc>>,
    consecutive_failures: u32,
) {
    metrics.clob_connected_connection_epoch = None;
    metrics.clob_connected_connection_id = None;
    metrics.clob_active_connection_epoch = None;
    metrics.clob_active_connection_id = None;
    metrics.clob_active_subscribed_assets = 0;
    metrics.clob_last_disconnect_at = Some(disconnected_at);
    metrics.clob_last_disconnect_reason = Some(reason.to_string());
    metrics.clob_recovery_unavailable_since = recovery_unavailable_since;
    metrics.clob_consecutive_failures = consecutive_failures;
}

fn clob_session_metadata(
    healthy_epoch: bool,
    disconnect_cause: ClobDisconnectCause,
    retry_action: ClobRetryAction,
    consecutive_failures: u32,
    subscription_stats: &ClobSubscriptionStats,
) -> serde_json::Value {
    let (next_action, retry_delay) = match retry_action {
        ClobRetryAction::Stop => ("stop", None),
        ClobRetryAction::ImmediateRecovery => ("immediate_recovery", None),
        ClobRetryAction::Backoff(delay) => ("backoff", Some(delay)),
    };
    serde_json::json!({
        "healthy_epoch": healthy_epoch,
        "disconnect_cause": disconnect_cause.as_str(),
        "consecutive_failures": consecutive_failures,
        "retry_delay_ms": retry_delay.map(duration_milliseconds),
        "next_action": next_action,
        "subscription_updates": subscription_stats.updates,
        "active_subscribed_assets": subscription_stats.active_assets,
        "last_subscription_update_at": subscription_stats.last_updated_at,
    })
}

#[allow(clippy::too_many_arguments)]
fn log_clob_disconnect(
    connection_id: Uuid,
    connection_epoch: i32,
    connected_duration: StdDuration,
    healthy_epoch: bool,
    disconnect_cause: ClobDisconnectCause,
    retry_action: ClobRetryAction,
    consecutive_failures: u32,
    reason: &str,
) {
    let retry_delay_milliseconds = match retry_action {
        ClobRetryAction::Backoff(delay) => duration_milliseconds(delay),
        ClobRetryAction::Stop | ClobRetryAction::ImmediateRecovery => 0,
    };
    let immediate_recovery = matches!(retry_action, ClobRetryAction::ImmediateRecovery);
    let connected_duration_milliseconds = duration_milliseconds(connected_duration);
    match disconnect_cause {
        ClobDisconnectCause::Shutdown | ClobDisconnectCause::MarketWatchClosed => tracing::info!(
            feed = "polymarket_clob_market",
            %connection_id,
            connection_epoch,
            connected_duration_ms = connected_duration_milliseconds,
            healthy_epoch,
            disconnect_cause = disconnect_cause.as_str(),
            immediate_recovery,
            failure_streak = consecutive_failures,
            retry_delay_ms = retry_delay_milliseconds,
            reason,
            "CLOB websocket disconnected"
        ),
        ClobDisconnectCause::CriticalPersistence => tracing::error!(
            feed = "polymarket_clob_market",
            %connection_id,
            connection_epoch,
            connected_duration_ms = connected_duration_milliseconds,
            healthy_epoch,
            disconnect_cause = disconnect_cause.as_str(),
            immediate_recovery,
            failure_streak = consecutive_failures,
            retry_delay_ms = retry_delay_milliseconds,
            reason,
            "CLOB websocket disconnected"
        ),
        ClobDisconnectCause::ConnectFailure
        | ClobDisconnectCause::SubscriptionFailure
        | ClobDisconnectCause::BootstrapFailure
        | ClobDisconnectCause::TransportFailure => tracing::warn!(
            feed = "polymarket_clob_market",
            %connection_id,
            connection_epoch,
            connected_duration_ms = connected_duration_milliseconds,
            healthy_epoch,
            disconnect_cause = disconnect_cause.as_str(),
            immediate_recovery,
            failure_streak = consecutive_failures,
            retry_delay_ms = retry_delay_milliseconds,
            reason,
            "CLOB websocket disconnected"
        ),
    }
}

fn reference_transport_metrics_mut(
    metrics: &mut BtcRuntimeMetrics,
    kind: ReferenceFeedKind,
) -> &mut ReferenceTransportMetrics {
    match kind {
        ReferenceFeedKind::Rtds => &mut metrics.rtds_transport,
        ReferenceFeedKind::Binance => &mut metrics.binance_transport,
    }
}

fn reference_reconnect_delay(
    config: &BtcRuntimeConfig,
    consecutive_failures: u32,
    connection_epoch: i32,
    connection_id: Uuid,
    kind: ReferenceFeedKind,
) -> StdDuration {
    let base = reconnect_backoff(config, consecutive_failures);
    let base_milliseconds = duration_milliseconds(base);
    let spread = base_milliseconds.saturating_mul(REFERENCE_RETRY_JITTER_PERCENT) / 100;
    if spread == 0 {
        return base;
    }
    let width = spread.saturating_mul(2).saturating_add(1);
    let epoch = u64::try_from(connection_epoch).unwrap_or_default();
    let connection_bits = connection_id.as_u128();
    let connection_seed = (connection_bits as u64) ^ ((connection_bits >> 64) as u64);
    let mut seed = kind.retry_salt()
        ^ connection_seed
        ^ epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ u64::from(consecutive_failures).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    seed ^= seed >> 30;
    seed = seed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    seed ^= seed >> 27;
    let jittered = base_milliseconds
        .saturating_sub(spread)
        .saturating_add(seed % width);
    StdDuration::from_millis(jittered).min(config.reconnect_max_delay)
}

fn reference_retry_action(
    config: &BtcRuntimeConfig,
    stable_epoch: bool,
    stop: bool,
    retry_state: &mut ReferenceRetryState,
    connection_epoch: i32,
    connection_id: Uuid,
    kind: ReferenceFeedKind,
) -> ReferenceRetryAction {
    if stop {
        ReferenceRetryAction::Stop
    } else if stable_epoch {
        ReferenceRetryAction::ImmediateRecovery
    } else {
        let failures = retry_state.record_failure();
        ReferenceRetryAction::Backoff(reference_reconnect_delay(
            config,
            failures,
            connection_epoch,
            connection_id,
            kind,
        ))
    }
}

fn reference_tick_is_fresh(
    tick: &ReferencePriceTick,
    checked_at: DateTime<Utc>,
    max_age: Duration,
) -> bool {
    [tick.source_timestamp, tick.received_at]
        .into_iter()
        .all(|timestamp| timestamp - checked_at <= max_age && checked_at - timestamp <= max_age)
}

fn reference_tick_progresses(
    current: Option<&ReferencePriceTick>,
    tick: &ReferencePriceTick,
    kind: ReferenceFeedKind,
) -> bool {
    let required_source = match kind {
        ReferenceFeedKind::Rtds => ReferencePriceSource::RtdsChainlink,
        ReferenceFeedKind::Binance => ReferencePriceSource::DirectBinance,
    };
    tick.source == required_source && reference_tick_version_advances(current, tick)
}

fn reference_tick_version_advances(
    current: Option<&ReferencePriceTick>,
    tick: &ReferencePriceTick,
) -> bool {
    let direct_binance_event_id = match tick.source {
        ReferencePriceSource::DirectBinance => {
            let Some(event_id) = tick
                .source_event_id
                .as_deref()
                .and_then(|value| value.parse::<u64>().ok())
            else {
                return false;
            };
            Some(event_id)
        }
        ReferencePriceSource::RtdsChainlink | ReferencePriceSource::RtdsBinance => None,
    };
    let Some(current) = current else {
        return true;
    };
    if tick.source != current.source {
        return false;
    }
    match tick.source {
        ReferencePriceSource::RtdsChainlink | ReferencePriceSource::RtdsBinance => {
            tick.source_timestamp > current.source_timestamp
        }
        ReferencePriceSource::DirectBinance => {
            let Some(current_event_id) = current
                .source_event_id
                .as_deref()
                .and_then(|value| value.parse::<u64>().ok())
            else {
                return false;
            };
            let Some(next_event_id) = direct_binance_event_id else {
                return false;
            };
            tick.source_timestamp >= current.source_timestamp && next_event_id > current_event_id
        }
    }
}

fn update_reference_state_and_check_progress(
    state: &mut RealtimeState,
    tick: ReferencePriceTick,
    kind: ReferenceFeedKind,
    checked_at: DateTime<Utc>,
    max_age: Duration,
) -> bool {
    if !reference_tick_is_fresh(&tick, checked_at, max_age) {
        return false;
    }
    let current = state.reference_prices.get(&tick.source);
    if !reference_tick_version_advances(current, &tick) {
        return false;
    }
    let required_progress = reference_tick_progresses(current, &tick, kind);
    state.update_reference_price(tick) && required_progress
}

struct BoundedReferenceDetail {
    value: String,
    remaining_bytes: usize,
}

impl BoundedReferenceDetail {
    fn new(capacity: usize) -> Self {
        Self {
            value: String::with_capacity(capacity),
            remaining_bytes: capacity,
        }
    }
}

impl Write for BoundedReferenceDetail {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        if self.remaining_bytes == 0 {
            return Err(fmt::Error);
        }
        let mut accepted_bytes = value.len().min(self.remaining_bytes);
        while accepted_bytes > 0 && !value.is_char_boundary(accepted_bytes) {
            accepted_bytes -= 1;
        }
        self.value.push_str(&value[..accepted_bytes]);
        self.remaining_bytes -= accepted_bytes;
        if accepted_bytes < value.len() {
            Err(fmt::Error)
        } else {
            Ok(())
        }
    }
}

fn bounded_reference_detail(detail: impl fmt::Display) -> String {
    let mut bounded = BoundedReferenceDetail::new(256);
    let _ = write!(&mut bounded, "{detail}");
    bounded.value
}

fn reference_session_metadata(
    disconnect_reason: ReferenceDisconnectReason,
    disconnect_detail: Option<&str>,
    retry_action: ReferenceRetryAction,
    consecutive_failures: u32,
    connected_duration: Option<StdDuration>,
    stats: &ReferenceSessionStats,
) -> serde_json::Value {
    let (next_action, retry_delay) = match retry_action {
        ReferenceRetryAction::Stop => ("stop", None),
        ReferenceRetryAction::ImmediateRecovery => ("immediate_recovery", None),
        ReferenceRetryAction::Backoff(delay) => ("backoff", Some(delay)),
    };
    serde_json::json!({
        "disconnect_reason": disconnect_reason,
        "disconnect_detail": disconnect_detail,
        "disconnect_cause": disconnect_reason.cause().as_str(),
        "healthy_epoch": stats.healthy_epoch,
        "stable_epoch": stats.stable_epoch,
        "consecutive_failures": consecutive_failures,
        "retry_delay_ms": retry_delay.map(duration_milliseconds),
        "next_action": next_action,
        "connected_duration_ms": connected_duration.map(duration_milliseconds),
        "time_to_first_required_tick_ms": stats.time_to_first_required_tick_milliseconds,
        "required_tick_count": stats.required_ticks,
        "heartbeat_probes": stats.heartbeat_probes,
        "heartbeat_acknowledgements": stats.heartbeat_acknowledgements,
        "last_required_tick_at": stats.last_required_tick_at,
        "last_frame_at": stats.last_frame_at,
        "last_pong_at": stats.last_pong_at,
        "remote_close_code": stats.remote_close_code,
    })
}

fn record_reference_connected(
    metrics: &mut BtcRuntimeMetrics,
    kind: ReferenceFeedKind,
    connection_id: Uuid,
    connection_epoch: i32,
    connected_at: DateTime<Utc>,
    consecutive_failures: u32,
) {
    let transport = reference_transport_metrics_mut(metrics, kind);
    transport.connections_established = transport.connections_established.saturating_add(1);
    transport.connected_connection_epoch = Some(connection_epoch);
    transport.connected_connection_id = Some(connection_id);
    transport.last_connected_at = Some(connected_at);
    transport.consecutive_failures = consecutive_failures;
}

#[allow(clippy::too_many_arguments)]
fn record_reference_required_tick(
    metrics: &mut BtcRuntimeMetrics,
    kind: ReferenceFeedKind,
    connection_id: Uuid,
    connection_epoch: i32,
    required_at: DateTime<Utc>,
    first_healthy_transition: bool,
    unavailable_milliseconds: u64,
) {
    let transport = reference_transport_metrics_mut(metrics, kind);
    transport.required_ticks_received = transport.required_ticks_received.saturating_add(1);
    transport.last_required_tick_at = Some(required_at);
    if first_healthy_transition {
        transport.healthy_connections = transport.healthy_connections.saturating_add(1);
        transport.active_connection_epoch = Some(connection_epoch);
        transport.active_connection_id = Some(connection_id);
        transport.last_healthy_at = Some(required_at);
        transport.recovery_unavailable_milliseconds = transport
            .recovery_unavailable_milliseconds
            .saturating_add(unavailable_milliseconds);
        transport.recovery_unavailable_since = None;
    }
}

fn record_reference_stable(metrics: &mut BtcRuntimeMetrics, kind: ReferenceFeedKind) {
    reference_transport_metrics_mut(metrics, kind).consecutive_failures = 0;
}

#[allow(clippy::too_many_arguments)]
fn record_reference_disconnected(
    metrics: &mut BtcRuntimeMetrics,
    kind: ReferenceFeedKind,
    reason: ReferenceDisconnectReason,
    retry_action: ReferenceRetryAction,
    disconnected_at: DateTime<Utc>,
    unavailable_since: Option<DateTime<Utc>>,
    consecutive_failures: u32,
) {
    let transport = reference_transport_metrics_mut(metrics, kind);
    transport.connected_connection_epoch = None;
    transport.connected_connection_id = None;
    transport.active_connection_epoch = None;
    transport.active_connection_id = None;
    transport.last_disconnect_at = Some(disconnected_at);
    transport.last_disconnect_reason = Some(reason);
    transport.recovery_unavailable_since = unavailable_since;
    transport.consecutive_failures = consecutive_failures;
    match reason.cause() {
        ReferenceDisconnectCause::ConnectFailure => {
            transport.connection_failures = transport.connection_failures.saturating_add(1);
        }
        ReferenceDisconnectCause::SubscriptionFailure => {
            transport.subscription_failures = transport.subscription_failures.saturating_add(1);
        }
        ReferenceDisconnectCause::WatchdogTimeout(timeout) => {
            transport.watchdog_disconnects = transport.watchdog_disconnects.saturating_add(1);
            match timeout {
                ReferenceWatchdogTimeout::RequiredData => {
                    transport.required_data_timeouts =
                        transport.required_data_timeouts.saturating_add(1);
                }
                ReferenceWatchdogTimeout::HeartbeatAck => {
                    transport.pong_timeouts = transport.pong_timeouts.saturating_add(1);
                }
                ReferenceWatchdogTimeout::ReadIdle => {
                    transport.read_timeouts = transport.read_timeouts.saturating_add(1);
                }
            }
        }
        ReferenceDisconnectCause::TransportFailure => {
            transport.transport_disconnects = transport.transport_disconnects.saturating_add(1);
        }
        ReferenceDisconnectCause::Shutdown | ReferenceDisconnectCause::CriticalPersistence => {}
    }
    match retry_action {
        ReferenceRetryAction::ImmediateRecovery => {
            transport.immediate_recoveries_scheduled =
                transport.immediate_recoveries_scheduled.saturating_add(1);
        }
        ReferenceRetryAction::Backoff(delay) => {
            transport.backoff_scheduled_milliseconds = transport
                .backoff_scheduled_milliseconds
                .saturating_add(duration_milliseconds(delay));
        }
        ReferenceRetryAction::Stop => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn log_reference_disconnect(
    kind: ReferenceFeedKind,
    connection_id: Uuid,
    connection_epoch: i32,
    reason: ReferenceDisconnectReason,
    retry_action: ReferenceRetryAction,
    consecutive_failures: u32,
    connected_duration: Option<StdDuration>,
    detail: Option<&str>,
) {
    let retry_delay_milliseconds = match retry_action {
        ReferenceRetryAction::Backoff(delay) => duration_milliseconds(delay),
        ReferenceRetryAction::Stop | ReferenceRetryAction::ImmediateRecovery => 0,
    };
    let immediate_recovery = matches!(retry_action, ReferenceRetryAction::ImmediateRecovery);
    let connected_duration_milliseconds = connected_duration.map(duration_milliseconds);
    match reason.cause() {
        ReferenceDisconnectCause::Shutdown => tracing::info!(
            feed = kind.feed_name(),
            %connection_id,
            connection_epoch,
            reason = reason.as_str(),
            immediate_recovery,
            failure_streak = consecutive_failures,
            retry_delay_ms = retry_delay_milliseconds,
            connected_duration_ms = connected_duration_milliseconds,
            "reference websocket disconnected"
        ),
        ReferenceDisconnectCause::CriticalPersistence => tracing::error!(
            feed = kind.feed_name(),
            %connection_id,
            connection_epoch,
            reason = reason.as_str(),
            detail = detail.unwrap_or("none"),
            connected_duration_ms = connected_duration_milliseconds,
            "reference websocket stopped after critical persistence failure"
        ),
        _ => tracing::warn!(
            feed = kind.feed_name(),
            %connection_id,
            connection_epoch,
            reason = reason.as_str(),
            detail = detail.unwrap_or("none"),
            immediate_recovery,
            failure_streak = consecutive_failures,
            retry_delay_ms = retry_delay_milliseconds,
            connected_duration_ms = connected_duration_milliseconds,
            "reference websocket disconnected"
        ),
    }
}

async fn run_rtds_supervisor(
    config: BtcRuntimeConfig,
    repository: BtcRepository,
    writer: mpsc::Sender<PersistItem>,
    state: Arc<RwLock<RealtimeState>>,
    boundaries: Arc<RwLock<BoundaryTracker>>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
    mut shutdown: watch::Receiver<bool>,
) {
    if !config.enabled {
        return;
    }
    let kind = ReferenceFeedKind::Rtds;
    let mut reconnect_ordinal = 0i32;
    let mut retry_state = ReferenceRetryState::default();
    let mut recovery_window = ReferenceRecoveryWindow::open(Utc::now(), Instant::now());
    metrics
        .write()
        .await
        .rtds_transport
        .recovery_unavailable_since = recovery_window.since;
    loop {
        if *shutdown.borrow() {
            break;
        }
        reconnect_ordinal = reconnect_ordinal.saturating_add(1);
        let connection_id = Uuid::new_v4();
        let mut sequence = 0u64;
        let mut session = new_session(
            connection_id,
            "polymarket_rtds",
            &config.rtds_ws_url,
            reconnect_ordinal,
            Utc::now(),
        );
        let attempt_started = Instant::now();
        let connect_result = tokio::select! {
            biased;
            _ = shutdown.changed() => None,
            result = timeout(
                REFERENCE_CONNECT_TIMEOUT,
                connect_async(&config.rtds_ws_url),
            ) => Some(result),
        };
        let Some(connect_result) = connect_result else {
            break;
        };
        let (mut socket, _) = match connect_result {
            Ok(Ok(value)) => value,
            result => {
                let (reason, detail) = match result {
                    Ok(Err(error)) => (
                        ReferenceDisconnectReason::ConnectFailed,
                        Some(bounded_reference_detail(error)),
                    ),
                    Err(_) => (ReferenceDisconnectReason::ConnectTimeout, None),
                    Ok(Ok(_)) => unreachable!(),
                };
                let disconnected_at = Utc::now();
                let retry_action = reference_retry_action(
                    &config,
                    false,
                    false,
                    &mut retry_state,
                    reconnect_ordinal,
                    connection_id,
                    kind,
                );
                session.disconnected_at = Some(disconnected_at);
                session.disconnect_reason = Some(reason.as_str().to_string());
                session.metadata = reference_session_metadata(
                    reason,
                    detail.as_deref(),
                    retry_action,
                    retry_state.consecutive_failures,
                    None,
                    &ReferenceSessionStats::default(),
                );
                if !start_feed_session_or_fail(&repository, &session, &metrics).await
                    || !finish_feed_session_or_fail(&repository, &session, &metrics).await
                {
                    return;
                }
                {
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.reconnects = runtime_metrics.reconnects.saturating_add(1);
                    record_reference_disconnected(
                        &mut runtime_metrics,
                        kind,
                        reason,
                        retry_action,
                        disconnected_at,
                        recovery_window.since,
                        retry_state.consecutive_failures,
                    );
                }
                log_reference_disconnect(
                    kind,
                    connection_id,
                    reconnect_ordinal,
                    reason,
                    retry_action,
                    retry_state.consecutive_failures,
                    None,
                    detail.as_deref(),
                );
                let error_message = detail
                    .as_deref()
                    .map(|detail| format!("{}: {detail}", reason.as_str()))
                    .unwrap_or_else(|| reason.as_str().to_string());
                record_error(&metrics, anyhow::anyhow!(error_message)).await;
                let ReferenceRetryAction::Backoff(delay) = retry_action else {
                    unreachable!("connect failure must schedule bounded backoff");
                };
                if !wait_reconnect_backoff(delay, &mut shutdown).await {
                    break;
                }
                continue;
            }
        };
        let connected_at = Utc::now();
        let connected_instant = Instant::now();
        session.connected_at = Some(connected_at);
        if !start_feed_session_or_fail(&repository, &session, &metrics).await {
            return;
        }
        {
            let mut runtime_metrics = metrics.write().await;
            record_reference_connected(
                &mut runtime_metrics,
                kind,
                connection_id,
                reconnect_ordinal,
                connected_at,
                retry_state.consecutive_failures,
            );
        }
        tracing::info!(
            feed = kind.feed_name(),
            %connection_id,
            connection_epoch = reconnect_ordinal,
            failure_streak = retry_state.consecutive_failures,
            connect_latency_ms = duration_milliseconds(connected_instant.duration_since(attempt_started)),
            "reference websocket connected"
        );
        let mut fatal_persistence_error = None;
        let disconnect_reason: ReferenceDisconnectReason;
        let mut disconnect_detail = None;
        let mut stats = ReferenceSessionStats::default();
        let subscription_result = tokio::select! {
            biased;
            _ = shutdown.changed() => None,
            result = timeout(
                REFERENCE_SEND_TIMEOUT,
                socket.send(Message::Text(rtds_subscription().into())),
            ) => Some(result),
        };
        match subscription_result {
            None => disconnect_reason = ReferenceDisconnectReason::Shutdown,
            Some(Err(_)) => {
                disconnect_reason = ReferenceDisconnectReason::SubscriptionSendTimeout;
            }
            Some(Ok(Err(error))) => {
                disconnect_reason = ReferenceDisconnectReason::SubscriptionSendFailed;
                disconnect_detail = Some(bounded_reference_detail(error));
            }
            Some(Ok(Ok(()))) => {
                let watchdog_started = Instant::now();
                let mut watchdog = ReferenceFeedWatchdog::new(watchdog_started, kind);
                let mut heartbeat = interval_at(
                    watchdog_started + config.rtds_heartbeat_interval,
                    config.rtds_heartbeat_interval,
                );
                heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
                let required_data_sleep = sleep(kind.required_data_timeout());
                let read_idle_sleep = sleep(REFERENCE_READ_IDLE_TIMEOUT);
                let stable_sleep = sleep(REFERENCE_STABLE_RESET_AFTER);
                tokio::pin!(required_data_sleep, read_idle_sleep, stable_sleep);
                'connection: loop {
                    required_data_sleep
                        .as_mut()
                        .reset(watchdog.required_data_deadline);
                    read_idle_sleep.as_mut().reset(watchdog.read_idle_deadline);
                    if let Some(deadline) = watchdog.stable_deadline {
                        stable_sleep.as_mut().reset(deadline);
                    }
                    tokio::select! {
                        biased;
                        _ = shutdown.changed() => {
                            disconnect_reason = ReferenceDisconnectReason::Shutdown;
                            break;
                        }
                        _ = &mut required_data_sleep => {
                            disconnect_reason = ReferenceDisconnectReason::RequiredDataIdleTimeout;
                            break;
                        }
                        _ = &mut read_idle_sleep => {
                            disconnect_reason = ReferenceDisconnectReason::ReadIdleTimeout;
                            break;
                        }
                        _ = &mut stable_sleep, if watchdog.stable_deadline.is_some() => {
                            if watchdog.mark_stable() {
                                stats.stable_epoch = true;
                                retry_state.reset();
                                let mut runtime_metrics = metrics.write().await;
                                record_reference_stable(&mut runtime_metrics, kind);
                                drop(runtime_metrics);
                                tracing::info!(
                                    feed = kind.feed_name(),
                                    %connection_id,
                                    connection_epoch = reconnect_ordinal,
                                    "reference websocket reached stable data health"
                                );
                            }
                        }
                        _ = heartbeat.tick() => {
                            let send_result = tokio::select! {
                                biased;
                                _ = shutdown.changed() => None,
                                result = timeout(
                                    REFERENCE_SEND_TIMEOUT,
                                    socket.send(Message::Text(RTDS_HEARTBEAT_MESSAGE.into())),
                                ) => Some(result),
                            };
                            match send_result {
                                None => {
                                    disconnect_reason = ReferenceDisconnectReason::Shutdown;
                                    break;
                                }
                                Some(Err(_)) => {
                                    disconnect_reason = ReferenceDisconnectReason::HeartbeatSendTimeout;
                                    break;
                                }
                                Some(Ok(Err(error))) => {
                                    disconnect_reason = ReferenceDisconnectReason::HeartbeatSendFailed;
                                    disconnect_detail = Some(bounded_reference_detail(error));
                                    break;
                                }
                                Some(Ok(Ok(()))) => {
                                    stats.heartbeat_probes = stats.heartbeat_probes.saturating_add(1);
                                    let mut runtime_metrics = metrics.write().await;
                                    let transport = reference_transport_metrics_mut(
                                        &mut runtime_metrics,
                                        kind,
                                    );
                                    transport.heartbeat_probes =
                                        transport.heartbeat_probes.saturating_add(1);
                                }
                            }
                        }
                        message = socket.next() => {
                            let Some(message) = message else {
                                disconnect_reason = ReferenceDisconnectReason::WebsocketEof;
                                break;
                            };
                            match message {
                                Ok(message) => {
                                    let received_at = Utc::now();
                                    let received_instant = Instant::now();
                                    watchdog.on_frame(received_instant);
                                    stats.last_frame_at = Some(received_at);
                                    let rtds_pong = matches!(&message, Message::Pong(_))
                                        || matches!(
                                            &message,
                                            Message::Text(text)
                                                if text.trim().eq_ignore_ascii_case("pong")
                                        );
                                    if rtds_pong {
                                        stats.heartbeat_acknowledgements = stats
                                            .heartbeat_acknowledgements
                                            .saturating_add(1);
                                        stats.last_pong_at = Some(received_at);
                                        let mut runtime_metrics = metrics.write().await;
                                        let transport = reference_transport_metrics_mut(
                                            &mut runtime_metrics,
                                            kind,
                                        );
                                        transport.heartbeat_acknowledgements = transport
                                            .heartbeat_acknowledgements
                                            .saturating_add(1);
                                        continue;
                                    }
                                    match message {
                                        Message::Text(text) if text.trim().is_empty() => {}
                                        Message::Text(text) => {
                                            sequence = sequence.saturating_add(1);
                                            session.messages_received = session.messages_received.saturating_add(1);
                                            let parsed = serde_json::from_str::<serde_json::Value>(&text)
                                                .context("failed to decode RTDS JSON")
                                                .and_then(|value| {
                                                    if !is_rtds_reference_update(&value) {
                                                        return Ok(None);
                                                    }
                                                    parse_rtds_reference_tick(
                                                        &value, connection_id, sequence, received_at
                                                    )
                                                    .map(Some)
                                                });
                                            match parsed {
                                                Ok(Some(tick)) => {
                                                    let health_progress = {
                                                        let mut realtime = state.write().await;
                                                        update_reference_state_and_check_progress(
                                                            &mut realtime,
                                                            tick.clone(),
                                                            kind,
                                                            received_at,
                                                            chrono_duration(config.max_reference_age),
                                                        )
                                                    };
                                                    let first_healthy_transition =
                                                        health_progress && !stats.healthy_epoch;
                                                    let unavailable_milliseconds = if first_healthy_transition {
                                                        recovery_window.close(received_instant)
                                                    } else {
                                                        0
                                                    };
                                                    {
                                                        let mut runtime_metrics = metrics.write().await;
                                                        runtime_metrics.reference_ticks_received = runtime_metrics
                                                            .reference_ticks_received
                                                            .saturating_add(1);
                                                        match tick.source {
                                                            ReferencePriceSource::RtdsChainlink => {
                                                                runtime_metrics.rtds_chainlink_ticks_received =
                                                                    runtime_metrics.rtds_chainlink_ticks_received
                                                                        .saturating_add(1);
                                                            }
                                                            ReferencePriceSource::RtdsBinance => {
                                                                runtime_metrics.rtds_binance_ticks_received =
                                                                    runtime_metrics.rtds_binance_ticks_received
                                                                        .saturating_add(1);
                                                            }
                                                            ReferencePriceSource::DirectBinance => {}
                                                        }
                                                        if health_progress {
                                                            record_reference_required_tick(
                                                                &mut runtime_metrics,
                                                                kind,
                                                                connection_id,
                                                                reconnect_ordinal,
                                                                received_at,
                                                                first_healthy_transition,
                                                                unavailable_milliseconds,
                                                            );
                                                        }
                                                    }
                                                    if health_progress {
                                                        watchdog.on_required_tick(received_instant, kind);
                                                        stats.required_ticks = stats.required_ticks.saturating_add(1);
                                                        stats.last_required_tick_at = Some(received_at);
                                                        if first_healthy_transition {
                                                            stats.healthy_epoch = true;
                                                            stats.time_to_first_required_tick_milliseconds = Some(
                                                                duration_milliseconds(
                                                                    received_instant.duration_since(connected_instant),
                                                                ),
                                                            );
                                                            tracing::info!(
                                                                feed = kind.feed_name(),
                                                                %connection_id,
                                                                connection_epoch = reconnect_ordinal,
                                                                unavailable_duration_ms = unavailable_milliseconds,
                                                                "reference websocket began delivering required data"
                                                            );
                                                        }
                                                    }
                                                    if tick.source == ReferencePriceSource::RtdsChainlink {
                                                        let max_delay =
                                                            chrono_duration(config.boundary_tick_max_delay);
                                                        let observed = boundaries
                                                            .write()
                                                            .await
                                                            .observe_chainlink(&tick, max_delay);
                                                        if let Err(error) = observed {
                                                            disconnect_reason = ReferenceDisconnectReason::CriticalBoundaryIntegrity;
                                                            disconnect_detail = Some(bounded_reference_detail(&error));
                                                            fatal_persistence_error = Some(error);
                                                            break 'connection;
                                                        }
                                                        if let Err(error) = flush_pending_boundaries(
                                                            &repository,
                                                            &boundaries,
                                                            &metrics,
                                                            max_delay,
                                                        )
                                                        .await
                                                        {
                                                            disconnect_reason = ReferenceDisconnectReason::CriticalBoundaryPersistence;
                                                            disconnect_detail = Some(bounded_reference_detail(&error));
                                                            fatal_persistence_error = Some(error);
                                                            break 'connection;
                                                        }
                                                    }
                                                    if enqueue(&writer, PersistItem::ReferenceTick(tick), &metrics).await {
                                                        session.messages_persisted =
                                                            session.messages_persisted.saturating_add(1);
                                                    } else {
                                                        session.dropped_messages =
                                                            session.dropped_messages.saturating_add(1);
                                                        disconnect_reason = ReferenceDisconnectReason::CriticalWriterQueue;
                                                        fatal_persistence_error = Some(anyhow::anyhow!(
                                                            "RTDS reference persistence queue rejected item"
                                                        ));
                                                        break 'connection;
                                                    }
                                                }
                                                Ok(None) => {}
                                                Err(error) => {
                                                    session.decode_errors = session.decode_errors.saturating_add(1);
                                                    {
                                                        let mut runtime_metrics = metrics.write().await;
                                                        runtime_metrics.decode_errors =
                                                            runtime_metrics.decode_errors.saturating_add(1);
                                                    }
                                                    record_error(&metrics, error).await;
                                                }
                                            }
                                        }
                                        Message::Close(frame) => {
                                            if let Some(frame) = frame {
                                                stats.remote_close_code = Some(u16::from(frame.code));
                                                if !frame.reason.is_empty() {
                                                    disconnect_detail =
                                                        Some(bounded_reference_detail(frame.reason));
                                                }
                                            }
                                            disconnect_reason = ReferenceDisconnectReason::RemoteClose;
                                            break;
                                        }
                                        _ => {}
                                    }
                                }
                                Err(error) => {
                                    disconnect_reason = ReferenceDisconnectReason::TransportReadFailed;
                                    disconnect_detail = Some(bounded_reference_detail(error));
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        let disconnected_at = Utc::now();
        let disconnected_instant = Instant::now();
        recovery_window.open_if_closed(disconnected_at, disconnected_instant);
        let stop = matches!(
            disconnect_reason.cause(),
            ReferenceDisconnectCause::Shutdown | ReferenceDisconnectCause::CriticalPersistence
        );
        let retry_action = reference_retry_action(
            &config,
            stats.stable_epoch,
            stop,
            &mut retry_state,
            reconnect_ordinal,
            connection_id,
            kind,
        );
        let connected_duration = Some(disconnected_instant.duration_since(connected_instant));
        session.disconnected_at = Some(disconnected_at);
        session.disconnect_reason = Some(disconnect_reason.as_str().to_string());
        session.metadata = reference_session_metadata(
            disconnect_reason,
            disconnect_detail.as_deref(),
            retry_action,
            retry_state.consecutive_failures,
            connected_duration,
            &stats,
        );
        {
            let mut runtime_metrics = metrics.write().await;
            if retry_action != ReferenceRetryAction::Stop {
                runtime_metrics.reconnects = runtime_metrics.reconnects.saturating_add(1);
            }
            record_reference_disconnected(
                &mut runtime_metrics,
                kind,
                disconnect_reason,
                retry_action,
                disconnected_at,
                recovery_window.since,
                retry_state.consecutive_failures,
            );
        }
        log_reference_disconnect(
            kind,
            connection_id,
            reconnect_ordinal,
            disconnect_reason,
            retry_action,
            retry_state.consecutive_failures,
            connected_duration,
            disconnect_detail.as_deref(),
        );
        if !finish_feed_session_or_fail(&repository, &session, &metrics).await {
            return;
        }
        if let Some(error) = fatal_persistence_error {
            record_critical_persistence_error(&metrics, error).await;
            return;
        }
        match retry_action {
            ReferenceRetryAction::Stop => return,
            ReferenceRetryAction::ImmediateRecovery => continue,
            ReferenceRetryAction::Backoff(delay) => {
                if !wait_reconnect_backoff(delay, &mut shutdown).await {
                    break;
                }
            }
        }
    }
}

async fn run_binance_supervisor(
    config: BtcRuntimeConfig,
    repository: BtcRepository,
    writer: mpsc::Sender<PersistItem>,
    state: Arc<RwLock<RealtimeState>>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
    mut shutdown: watch::Receiver<bool>,
) {
    if !config.enabled {
        return;
    }
    let kind = ReferenceFeedKind::Binance;
    let mut reconnect_ordinal = 0i32;
    let mut retry_state = ReferenceRetryState::default();
    let mut recovery_window = ReferenceRecoveryWindow::open(Utc::now(), Instant::now());
    metrics
        .write()
        .await
        .binance_transport
        .recovery_unavailable_since = recovery_window.since;
    loop {
        if *shutdown.borrow() {
            break;
        }
        reconnect_ordinal = reconnect_ordinal.saturating_add(1);
        let connection_id = Uuid::new_v4();
        let mut sequence = 0u64;
        let mut session = new_session(
            connection_id,
            "binance_agg_trade",
            &config.binance_ws_url,
            reconnect_ordinal,
            Utc::now(),
        );
        let attempt_started = Instant::now();
        let connect_result = tokio::select! {
            biased;
            _ = shutdown.changed() => None,
            result = timeout(
                REFERENCE_CONNECT_TIMEOUT,
                connect_async(&config.binance_ws_url),
            ) => Some(result),
        };
        let Some(connect_result) = connect_result else {
            break;
        };
        let (mut socket, _) = match connect_result {
            Ok(Ok(value)) => value,
            result => {
                let (reason, detail) = match result {
                    Ok(Err(error)) => (
                        ReferenceDisconnectReason::ConnectFailed,
                        Some(bounded_reference_detail(error)),
                    ),
                    Err(_) => (ReferenceDisconnectReason::ConnectTimeout, None),
                    Ok(Ok(_)) => unreachable!(),
                };
                let disconnected_at = Utc::now();
                let retry_action = reference_retry_action(
                    &config,
                    false,
                    false,
                    &mut retry_state,
                    reconnect_ordinal,
                    connection_id,
                    kind,
                );
                session.disconnected_at = Some(disconnected_at);
                session.disconnect_reason = Some(reason.as_str().to_string());
                session.metadata = reference_session_metadata(
                    reason,
                    detail.as_deref(),
                    retry_action,
                    retry_state.consecutive_failures,
                    None,
                    &ReferenceSessionStats::default(),
                );
                if !start_feed_session_or_fail(&repository, &session, &metrics).await
                    || !finish_feed_session_or_fail(&repository, &session, &metrics).await
                {
                    return;
                }
                {
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.reconnects = runtime_metrics.reconnects.saturating_add(1);
                    record_reference_disconnected(
                        &mut runtime_metrics,
                        kind,
                        reason,
                        retry_action,
                        disconnected_at,
                        recovery_window.since,
                        retry_state.consecutive_failures,
                    );
                }
                log_reference_disconnect(
                    kind,
                    connection_id,
                    reconnect_ordinal,
                    reason,
                    retry_action,
                    retry_state.consecutive_failures,
                    None,
                    detail.as_deref(),
                );
                let error_message = detail
                    .as_deref()
                    .map(|detail| format!("{}: {detail}", reason.as_str()))
                    .unwrap_or_else(|| reason.as_str().to_string());
                record_error(&metrics, anyhow::anyhow!(error_message)).await;
                let ReferenceRetryAction::Backoff(delay) = retry_action else {
                    unreachable!("connect failure must schedule bounded backoff");
                };
                if !wait_reconnect_backoff(delay, &mut shutdown).await {
                    break;
                }
                continue;
            }
        };
        let connected_at = Utc::now();
        let connected_instant = Instant::now();
        session.connected_at = Some(connected_at);
        if !start_feed_session_or_fail(&repository, &session, &metrics).await {
            return;
        }
        {
            let mut runtime_metrics = metrics.write().await;
            record_reference_connected(
                &mut runtime_metrics,
                kind,
                connection_id,
                reconnect_ordinal,
                connected_at,
                retry_state.consecutive_failures,
            );
        }
        tracing::info!(
            feed = kind.feed_name(),
            %connection_id,
            connection_epoch = reconnect_ordinal,
            failure_streak = retry_state.consecutive_failures,
            connect_latency_ms = duration_milliseconds(connected_instant.duration_since(attempt_started)),
            "reference websocket connected"
        );
        let mut fatal_persistence_error = None;
        let disconnect_reason: ReferenceDisconnectReason;
        let mut disconnect_detail = None;
        let mut stats = ReferenceSessionStats::default();
        let watchdog_started = Instant::now();
        let mut watchdog = ReferenceFeedWatchdog::new(watchdog_started, kind);
        let mut heartbeat = interval_at(
            watchdog_started + config.binance_heartbeat_interval,
            config.binance_heartbeat_interval,
        );
        heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let required_data_sleep = sleep(kind.required_data_timeout());
        let read_idle_sleep = sleep(REFERENCE_READ_IDLE_TIMEOUT);
        let pong_sleep = sleep(REFERENCE_PONG_TIMEOUT);
        let stable_sleep = sleep(REFERENCE_STABLE_RESET_AFTER);
        tokio::pin!(
            required_data_sleep,
            read_idle_sleep,
            pong_sleep,
            stable_sleep
        );
        let mut heartbeat_sequence = 0u64;
        'connection: loop {
            required_data_sleep
                .as_mut()
                .reset(watchdog.required_data_deadline);
            read_idle_sleep.as_mut().reset(watchdog.read_idle_deadline);
            if let Some(deadline) = watchdog.pong_deadline {
                pong_sleep.as_mut().reset(deadline);
            }
            if let Some(deadline) = watchdog.stable_deadline {
                stable_sleep.as_mut().reset(deadline);
            }
            tokio::select! {
                biased;
                _ = shutdown.changed() => {
                    disconnect_reason = ReferenceDisconnectReason::Shutdown;
                    break;
                }
                _ = &mut required_data_sleep => {
                    disconnect_reason = ReferenceDisconnectReason::RequiredDataIdleTimeout;
                    break;
                }
                _ = &mut pong_sleep, if watchdog.pong_deadline.is_some() => {
                    disconnect_reason = ReferenceDisconnectReason::HeartbeatAckTimeout;
                    break;
                }
                _ = &mut read_idle_sleep => {
                    disconnect_reason = ReferenceDisconnectReason::ReadIdleTimeout;
                    break;
                }
                _ = &mut stable_sleep, if watchdog.stable_deadline.is_some() => {
                    if watchdog.mark_stable() {
                        stats.stable_epoch = true;
                        retry_state.reset();
                        let mut runtime_metrics = metrics.write().await;
                        record_reference_stable(&mut runtime_metrics, kind);
                        drop(runtime_metrics);
                        tracing::info!(
                            feed = kind.feed_name(),
                            %connection_id,
                            connection_epoch = reconnect_ordinal,
                            "reference websocket reached stable data health"
                        );
                    }
                }
                _ = heartbeat.tick() => {
                    if watchdog.awaiting_pong() {
                        continue;
                    }
                    heartbeat_sequence = heartbeat_sequence.saturating_add(1);
                    let payload = heartbeat_sequence.to_be_bytes();
                    let send_result = tokio::select! {
                        biased;
                        _ = shutdown.changed() => None,
                        result = timeout(
                            REFERENCE_SEND_TIMEOUT,
                            socket.send(Message::Ping(payload.to_vec().into())),
                        ) => Some(result),
                    };
                    match send_result {
                        None => {
                            disconnect_reason = ReferenceDisconnectReason::Shutdown;
                            break;
                        }
                        Some(Err(_)) => {
                            disconnect_reason = ReferenceDisconnectReason::HeartbeatSendTimeout;
                            break;
                        }
                        Some(Ok(Err(error))) => {
                            disconnect_reason = ReferenceDisconnectReason::HeartbeatSendFailed;
                            disconnect_detail = Some(bounded_reference_detail(error));
                            break;
                        }
                        Some(Ok(Ok(()))) => {
                            let sent_at = Instant::now();
                            watchdog.arm_binary_pong(sent_at, payload);
                            stats.heartbeat_probes = stats.heartbeat_probes.saturating_add(1);
                            let mut runtime_metrics = metrics.write().await;
                            let transport = reference_transport_metrics_mut(
                                &mut runtime_metrics,
                                kind,
                            );
                            transport.heartbeat_probes =
                                transport.heartbeat_probes.saturating_add(1);
                        }
                    }
                }
                message = socket.next() => {
                    let Some(message) = message else {
                        disconnect_reason = ReferenceDisconnectReason::WebsocketEof;
                        break;
                    };
                    match message {
                        Ok(message) => {
                            let received_at = Utc::now();
                            let received_instant = Instant::now();
                            watchdog.on_frame(received_instant);
                            stats.last_frame_at = Some(received_at);
                            match message {
                                Message::Text(text) => {
                                    sequence = sequence.saturating_add(1);
                                    session.messages_received =
                                        session.messages_received.saturating_add(1);
                                    let parsed = serde_json::from_str::<serde_json::Value>(&text)
                                        .context("failed to decode Binance aggregate trade JSON")
                                        .and_then(|value| parse_binance_agg_trade(
                                            &value,
                                            connection_id,
                                            sequence,
                                            received_at,
                                        ));
                                    match parsed {
                                        Ok(tick) => {
                                            let health_progress = {
                                                let mut realtime = state.write().await;
                                                update_reference_state_and_check_progress(
                                                    &mut realtime,
                                                    tick.clone(),
                                                    kind,
                                                    received_at,
                                                    chrono_duration(config.max_reference_age),
                                                )
                                            };
                                            let first_healthy_transition =
                                                health_progress && !stats.healthy_epoch;
                                            let unavailable_milliseconds =
                                                if first_healthy_transition {
                                                    recovery_window.close(received_instant)
                                                } else {
                                                    0
                                                };
                                            {
                                                let mut runtime_metrics = metrics.write().await;
                                                runtime_metrics.reference_ticks_received =
                                                    runtime_metrics
                                                        .reference_ticks_received
                                                        .saturating_add(1);
                                                runtime_metrics.binance_ticks_received =
                                                    runtime_metrics
                                                        .binance_ticks_received
                                                        .saturating_add(1);
                                                if health_progress {
                                                    record_reference_required_tick(
                                                        &mut runtime_metrics,
                                                        kind,
                                                        connection_id,
                                                        reconnect_ordinal,
                                                        received_at,
                                                        first_healthy_transition,
                                                        unavailable_milliseconds,
                                                    );
                                                }
                                            }
                                            if health_progress {
                                                watchdog.on_required_tick(received_instant, kind);
                                                stats.required_ticks =
                                                    stats.required_ticks.saturating_add(1);
                                                stats.last_required_tick_at = Some(received_at);
                                                if first_healthy_transition {
                                                    stats.healthy_epoch = true;
                                                    stats.time_to_first_required_tick_milliseconds =
                                                        Some(duration_milliseconds(
                                                            received_instant
                                                                .duration_since(connected_instant),
                                                        ));
                                                    tracing::info!(
                                                        feed = kind.feed_name(),
                                                        %connection_id,
                                                        connection_epoch = reconnect_ordinal,
                                                        unavailable_duration_ms = unavailable_milliseconds,
                                                        "reference websocket began delivering required data"
                                                    );
                                                }
                                            }
                                            if enqueue(
                                                &writer,
                                                PersistItem::ReferenceTick(tick),
                                                &metrics,
                                            )
                                            .await
                                            {
                                                session.messages_persisted =
                                                    session.messages_persisted.saturating_add(1);
                                            } else {
                                                session.dropped_messages =
                                                    session.dropped_messages.saturating_add(1);
                                                disconnect_reason =
                                                    ReferenceDisconnectReason::CriticalWriterQueue;
                                                fatal_persistence_error = Some(anyhow::anyhow!(
                                                    "Binance reference persistence queue rejected item"
                                                ));
                                                break 'connection;
                                            }
                                        }
                                        Err(error) => {
                                            session.decode_errors =
                                                session.decode_errors.saturating_add(1);
                                            {
                                                let mut runtime_metrics = metrics.write().await;
                                                runtime_metrics.decode_errors =
                                                    runtime_metrics.decode_errors.saturating_add(1);
                                            }
                                            record_error(&metrics, error).await;
                                        }
                                    }
                                }
                                Message::Pong(payload) => {
                                    if watchdog.acknowledge_binary_pong(payload.as_ref()) {
                                        stats.heartbeat_acknowledgements = stats
                                            .heartbeat_acknowledgements
                                            .saturating_add(1);
                                        stats.last_pong_at = Some(received_at);
                                        let mut runtime_metrics = metrics.write().await;
                                        let transport = reference_transport_metrics_mut(
                                            &mut runtime_metrics,
                                            kind,
                                        );
                                        transport.heartbeat_acknowledgements = transport
                                            .heartbeat_acknowledgements
                                            .saturating_add(1);
                                    }
                                }
                                Message::Close(frame) => {
                                    if let Some(frame) = frame {
                                        stats.remote_close_code = Some(u16::from(frame.code));
                                        if !frame.reason.is_empty() {
                                            disconnect_detail =
                                                Some(bounded_reference_detail(frame.reason));
                                        }
                                    }
                                    disconnect_reason = ReferenceDisconnectReason::RemoteClose;
                                    break;
                                }
                                _ => {}
                            }
                        }
                        Err(error) => {
                            disconnect_reason = ReferenceDisconnectReason::TransportReadFailed;
                            disconnect_detail = Some(bounded_reference_detail(error));
                            break;
                        }
                    }
                }
            }
        }
        let disconnected_at = Utc::now();
        let disconnected_instant = Instant::now();
        recovery_window.open_if_closed(disconnected_at, disconnected_instant);
        let stop = matches!(
            disconnect_reason.cause(),
            ReferenceDisconnectCause::Shutdown | ReferenceDisconnectCause::CriticalPersistence
        );
        let retry_action = reference_retry_action(
            &config,
            stats.stable_epoch,
            stop,
            &mut retry_state,
            reconnect_ordinal,
            connection_id,
            kind,
        );
        let connected_duration = Some(disconnected_instant.duration_since(connected_instant));
        session.disconnected_at = Some(disconnected_at);
        session.disconnect_reason = Some(disconnect_reason.as_str().to_string());
        session.metadata = reference_session_metadata(
            disconnect_reason,
            disconnect_detail.as_deref(),
            retry_action,
            retry_state.consecutive_failures,
            connected_duration,
            &stats,
        );
        {
            let mut runtime_metrics = metrics.write().await;
            if retry_action != ReferenceRetryAction::Stop {
                runtime_metrics.reconnects = runtime_metrics.reconnects.saturating_add(1);
            }
            record_reference_disconnected(
                &mut runtime_metrics,
                kind,
                disconnect_reason,
                retry_action,
                disconnected_at,
                recovery_window.since,
                retry_state.consecutive_failures,
            );
        }
        log_reference_disconnect(
            kind,
            connection_id,
            reconnect_ordinal,
            disconnect_reason,
            retry_action,
            retry_state.consecutive_failures,
            connected_duration,
            disconnect_detail.as_deref(),
        );
        if !finish_feed_session_or_fail(&repository, &session, &metrics).await {
            return;
        }
        if let Some(error) = fatal_persistence_error {
            record_critical_persistence_error(&metrics, error).await;
            return;
        }
        match retry_action {
            ReferenceRetryAction::Stop => return,
            ReferenceRetryAction::ImmediateRecovery => continue,
            ReferenceRetryAction::Backoff(delay) => {
                if !wait_reconnect_backoff(delay, &mut shutdown).await {
                    break;
                }
            }
        }
    }
}

async fn run_strategy_loop(
    config: BtcRuntimeConfig,
    strategy: Arc<dyn BtcStrategyRunner>,
    state: Arc<RwLock<RealtimeState>>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
    mut shutdown: watch::Receiver<bool>,
) {
    if !config.enabled {
        return;
    }
    let mut ticker = interval(config.strategy_interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut last_observed_at = None;
    loop {
        tokio::select! {
            _ = shutdown.changed() => break,
            _ = ticker.tick() => {
                let snapshot = state.read().await.clone();
                if snapshot.last_updated_at == last_observed_at {
                    continue;
                }
                last_observed_at = snapshot.last_updated_at;
                let readiness = snapshot.readiness(
                    Utc::now(),
                    chrono_duration(config.max_book_age),
                    chrono_duration(config.max_reference_age),
                );
                if let Err(error) = strategy
                    .on_observation(StrategyObservation { state: snapshot, readiness })
                    .await
                {
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.strategy_errors =
                        runtime_metrics.strategy_errors.saturating_add(1);
                    runtime_metrics.last_error = Some(error.to_string());
                    // The deterministic strategy and shared paper execution path
                    // are primary immutable experiment data. A callback failure
                    // invalidates the run, so let the task exit and fail the runtime.
                    return;
                } else {
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.strategy_callbacks =
                        runtime_metrics.strategy_callbacks.saturating_add(1);
                }
            }
        }
    }
}

async fn flush_pending_boundaries(
    repository: &BtcRepository,
    boundaries: &Arc<RwLock<BoundaryTracker>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    max_delay: Duration,
) -> Result<()> {
    loop {
        let candidates = boundaries.read().await.pending_candidates()?;
        if candidates.is_empty() {
            return Ok(());
        }
        for candidate in candidates {
            persist_boundary_candidate(repository, &candidate, max_delay).await?;
            let acknowledged = boundaries.write().await.acknowledge(&candidate)?;
            let mut runtime_metrics = metrics.write().await;
            runtime_metrics.persistence_items_written =
                runtime_metrics.persistence_items_written.saturating_add(1);
            if acknowledged && matches!(candidate, BoundaryCandidate::Label { .. }) {
                runtime_metrics.labels_created = runtime_metrics.labels_created.saturating_add(1);
            }
        }
    }
}

async fn persist_boundary_candidate(
    repository: &BtcRepository,
    candidate: &BoundaryCandidate,
    max_delay: Duration,
) -> Result<()> {
    let mut delay = CRITICAL_WRITE_INITIAL_BACKOFF;
    let mut last_error = None;
    for attempt in 1..=CRITICAL_WRITE_ATTEMPTS {
        let result = match candidate {
            BoundaryCandidate::Open { market_id, tick } => {
                repository
                    .persist_immutable_market_open(market_id, tick, max_delay)
                    .await
            }
            BoundaryCandidate::Close { market_id, tick } => {
                repository
                    .persist_boundary_close_tick(market_id, tick, max_delay)
                    .await
            }
            BoundaryCandidate::Label { label, close_tick } => {
                repository
                    .persist_immutable_market_label(label, close_tick, max_delay)
                    .await
            }
        };
        match result {
            Ok(()) => return Ok(()),
            Err(error) => last_error = Some(error),
        }
        if attempt < CRITICAL_WRITE_ATTEMPTS {
            sleep(delay).await;
            delay = delay.saturating_mul(2);
        }
    }
    Err(last_error.expect("critical boundary retry loop always runs")).with_context(|| {
        format!(
            "critical BTC {} persistence failed after {CRITICAL_WRITE_ATTEMPTS} attempts",
            boundary_candidate_name(candidate)
        )
    })
}

fn boundary_candidate_name(candidate: &BoundaryCandidate) -> &'static str {
    match candidate {
        BoundaryCandidate::Open { .. } => "opening-reference",
        BoundaryCandidate::Close { .. } => "closing-reference",
        BoundaryCandidate::Label { .. } => "market-label",
    }
}

async fn persist_official_resolution(
    repository: &BtcRepository,
    message: &ClobMessage,
    received_at: DateTime<Utc>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
) -> Result<Option<PersistedOfficialResolution>> {
    let ClobMessage::MarketResolved {
        market_id,
        winning_token_id,
        winning_outcome,
        source_timestamp,
        raw_payload,
    } = message
    else {
        return Ok(None);
    };

    persist_official_resolution_fact(
        repository,
        market_id,
        winning_token_id,
        winning_outcome,
        *source_timestamp,
        "clob_websocket",
        received_at,
        raw_payload,
        metrics,
    )
    .await
    .map(Some)
}

#[allow(clippy::too_many_arguments)]
async fn persist_official_resolution_fact(
    repository: &BtcRepository,
    market_or_condition_id: &str,
    winning_token_id: &str,
    winning_outcome: &str,
    source_timestamp: DateTime<Utc>,
    resolution_source: &str,
    received_at: DateTime<Utc>,
    raw_payload: &serde_json::Value,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
) -> Result<PersistedOfficialResolution> {
    let mut delay = CRITICAL_WRITE_INITIAL_BACKOFF;
    let mut last_error = None;
    let mut newly_recorded = false;
    for attempt in 1..=CRITICAL_WRITE_ATTEMPTS {
        match repository
            .persist_official_market_resolution(
                market_or_condition_id,
                winning_token_id,
                winning_outcome,
                source_timestamp,
                resolution_source,
                received_at,
                raw_payload,
            )
            .await
        {
            Ok(mut persisted) => {
                newly_recorded |= persisted.newly_recorded;
                match repository
                    .refresh_paper_experiments_for_market(&persisted.market_id)
                    .await
                {
                    Ok(_) => {
                        persisted.newly_recorded = newly_recorded;
                        let mut runtime_metrics = metrics.write().await;
                        runtime_metrics.persistence_items_written =
                            runtime_metrics.persistence_items_written.saturating_add(1);
                        if persisted.newly_recorded {
                            match resolution_source {
                                "clob_websocket" => {
                                    runtime_metrics.official_resolutions_websocket = runtime_metrics
                                        .official_resolutions_websocket
                                        .saturating_add(1)
                                }
                                "clob_rest_reconciliation" => {
                                    runtime_metrics.official_resolutions_rest =
                                        runtime_metrics.official_resolutions_rest.saturating_add(1)
                                }
                                _ => {}
                            }
                        }
                        return Ok(persisted);
                    }
                    Err(error) => last_error = Some(error),
                }
            }
            Err(error) => last_error = Some(error),
        }
        if attempt < CRITICAL_WRITE_ATTEMPTS {
            sleep(delay).await;
            delay = delay.saturating_mul(2);
        }
    }
    Err(last_error.expect("official resolution retry loop always runs")).with_context(|| {
        format!(
            "critical official BTC resolution persistence failed after {CRITICAL_WRITE_ATTEMPTS} attempts"
        )
    })
}

#[derive(Debug, Clone)]
struct BoundaryState {
    market: BtcIntervalMarket,
    open_tick: Option<ReferencePriceTick>,
    pending_open_tick: Option<ReferencePriceTick>,
    close_tick: Option<ReferencePriceTick>,
    close_tick_durable: bool,
    label: Option<BtcMarketLabel>,
}

#[derive(Debug, Clone)]
enum BoundaryCandidate {
    Open {
        market_id: String,
        tick: ReferencePriceTick,
    },
    Close {
        market_id: String,
        tick: ReferencePriceTick,
    },
    Label {
        label: BtcMarketLabel,
        close_tick: ReferencePriceTick,
    },
}

#[derive(Debug, Default)]
struct BoundaryTracker {
    markets: HashMap<String, BoundaryState>,
}

impl BoundaryTracker {
    fn update_markets(&mut self, markets: &[BtcIntervalMarket]) {
        let keep: HashSet<_> = markets
            .iter()
            .map(|market| market.market_id.clone())
            .collect();
        self.markets.retain(|market_id, _| keep.contains(market_id));
        for market in markets {
            self.markets
                .entry(market.market_id.clone())
                .or_insert_with(|| BoundaryState {
                    market: market.clone(),
                    open_tick: None,
                    pending_open_tick: None,
                    close_tick: None,
                    close_tick_durable: false,
                    label: None,
                })
                .market = market.clone();
        }
    }

    fn hydrate_open_references(
        &mut self,
        references: &[(String, ReferencePriceTick)],
        max_delay: Duration,
    ) -> Result<()> {
        for (market_id, tick) in references {
            let Some(boundary) = self.markets.get_mut(market_id) else {
                continue;
            };
            require_tracker_tick(
                tick,
                boundary.market.window_start,
                boundary.market.window_start + max_delay,
                "opening",
            )?;
            if let Some(existing) = boundary.open_tick.as_ref() {
                if !same_tracker_tick(existing, tick) {
                    bail!(
                        "durable BTC opening-reference conflict while hydrating market {market_id}"
                    );
                }
            } else if let Some(pending) = boundary.pending_open_tick.as_ref() {
                if !same_tracker_tick(pending, tick) {
                    bail!(
                        "pending BTC opening reference conflicts with durable state for market {market_id}"
                    );
                }
            }
            boundary.open_tick = Some(tick.clone());
            boundary.pending_open_tick = None;
        }
        Ok(())
    }

    fn hydrate_close_references(
        &mut self,
        references: &[(String, ReferencePriceTick)],
        max_delay: Duration,
    ) -> Result<()> {
        for (market_id, tick) in references {
            let Some(boundary) = self.markets.get_mut(market_id) else {
                continue;
            };
            require_tracker_tick(
                tick,
                boundary.market.window_end,
                boundary.market.window_end + max_delay,
                "closing",
            )?;
            if let Some(existing) = boundary.close_tick.as_ref() {
                if !same_tracker_tick(existing, tick) {
                    if boundary.label.is_some() || !tick_precedes(tick, existing) {
                        bail!(
                            "durable BTC closing-reference conflict while hydrating market {market_id}"
                        );
                    }
                }
            }
            boundary.close_tick = Some(tick.clone());
            boundary.close_tick_durable = true;
        }
        Ok(())
    }

    fn hydrate_labels(&mut self, labels: &[BtcMarketLabel]) -> Result<()> {
        for label in labels {
            let Some(boundary) = self.markets.get_mut(&label.market_id) else {
                continue;
            };
            if label.window_start != boundary.market.window_start
                || label.window_end != boundary.market.window_end
                || label.label_source != "rtds_chainlink"
            {
                bail!(
                    "durable BTC label identity conflicts while hydrating market {}",
                    label.market_id
                );
            }
            let open = boundary.open_tick.as_ref().with_context(|| {
                format!(
                    "durable BTC label has no recoverable opening tick for market {}",
                    label.market_id
                )
            })?;
            let close = boundary.close_tick.as_ref().with_context(|| {
                format!(
                    "durable BTC label has no recoverable closing tick for market {}",
                    label.market_id
                )
            })?;
            if label.open_price != open.price
                || label.source_open_timestamp != open.source_timestamp
                || label.close_price != close.price
                || label.source_close_timestamp != close.source_timestamp
            {
                bail!(
                    "durable BTC label boundary evidence conflicts while hydrating market {}",
                    label.market_id
                );
            }
            if let Some(existing) = boundary.label.as_ref() {
                if !same_tracker_label(existing, label) {
                    bail!(
                        "immutable BTC label conflict while hydrating market {}",
                        label.market_id
                    );
                }
            }
            boundary.label = Some(label.clone());
        }
        Ok(())
    }

    fn observe_chainlink(&mut self, tick: &ReferencePriceTick, max_delay: Duration) -> Result<()> {
        if tick.source != ReferencePriceSource::RtdsChainlink {
            return Ok(());
        }
        for boundary in self.markets.values_mut() {
            if tick.source_timestamp >= boundary.market.window_start
                && tick.source_timestamp <= boundary.market.window_start + max_delay
            {
                if let Some(open) = boundary.open_tick.as_ref() {
                    if tick_precedes(tick, open) && !same_tracker_tick(tick, open) {
                        bail!(
                            "an earlier BTC opening tick arrived after immutable acknowledgment for market {}",
                            boundary.market.market_id
                        );
                    }
                } else if match boundary.pending_open_tick.as_ref() {
                    None => true,
                    Some(pending) => tick_precedes(tick, pending),
                } {
                    boundary.pending_open_tick = Some(tick.clone());
                }
            }
            if tick.source_timestamp >= boundary.market.window_end
                && tick.source_timestamp <= boundary.market.window_end + max_delay
            {
                match boundary.close_tick.as_ref() {
                    Some(close) if tick_precedes(tick, close) => {
                        if boundary.label.is_some() {
                            bail!(
                                "an earlier BTC closing tick arrived after immutable label acknowledgment for market {}",
                                boundary.market.market_id
                            );
                        }
                        boundary.close_tick = Some(tick.clone());
                        boundary.close_tick_durable = false;
                    }
                    None => {
                        boundary.close_tick = Some(tick.clone());
                        boundary.close_tick_durable = false;
                    }
                    Some(_) => {}
                }
            }
        }
        Ok(())
    }

    fn pending_candidates(&self) -> Result<Vec<BoundaryCandidate>> {
        let mut candidates = Vec::new();
        for boundary in self.markets.values() {
            if let Some(tick) = boundary.pending_open_tick.as_ref() {
                candidates.push(BoundaryCandidate::Open {
                    market_id: boundary.market.market_id.clone(),
                    tick: tick.clone(),
                });
            }
            if let Some(tick) = boundary
                .close_tick
                .as_ref()
                .filter(|_| !boundary.close_tick_durable)
            {
                candidates.push(BoundaryCandidate::Close {
                    market_id: boundary.market.market_id.clone(),
                    tick: tick.clone(),
                });
            }
            if boundary.label.is_none() && boundary.close_tick_durable {
                if let (Some(open), Some(close)) =
                    (boundary.open_tick.as_ref(), boundary.close_tick.as_ref())
                {
                    candidates.push(BoundaryCandidate::Label {
                        label: boundary_label(&boundary.market, open, close),
                        close_tick: close.clone(),
                    });
                }
            }
        }
        Ok(candidates)
    }

    fn acknowledge(&mut self, candidate: &BoundaryCandidate) -> Result<bool> {
        match candidate {
            BoundaryCandidate::Open { market_id, tick } => {
                let boundary = self
                    .markets
                    .get_mut(market_id)
                    .with_context(|| format!("BTC opening candidate lost market {market_id}"))?;
                if let Some(open) = boundary.open_tick.as_ref() {
                    if same_tracker_tick(open, tick) {
                        return Ok(false);
                    }
                    bail!("BTC opening acknowledgment conflicts for market {market_id}");
                }
                let pending = boundary.pending_open_tick.as_ref().with_context(|| {
                    format!("BTC opening candidate disappeared before ack for market {market_id}")
                })?;
                if !same_tracker_tick(pending, tick) {
                    bail!("BTC opening candidate changed before ack for market {market_id}");
                }
                boundary.open_tick = Some(tick.clone());
                boundary.pending_open_tick = None;
                Ok(true)
            }
            BoundaryCandidate::Close { market_id, tick } => {
                let boundary = self
                    .markets
                    .get_mut(market_id)
                    .with_context(|| format!("BTC closing candidate lost market {market_id}"))?;
                let close = boundary.close_tick.as_ref().with_context(|| {
                    format!("BTC closing candidate disappeared before ack for market {market_id}")
                })?;
                if !same_tracker_tick(close, tick) {
                    bail!("BTC closing candidate changed before ack for market {market_id}");
                }
                if boundary.close_tick_durable {
                    return Ok(false);
                }
                boundary.close_tick_durable = true;
                Ok(true)
            }
            BoundaryCandidate::Label { label, .. } => {
                let boundary = self.markets.get_mut(&label.market_id).with_context(|| {
                    format!("BTC label candidate lost market {}", label.market_id)
                })?;
                if let Some(existing) = boundary.label.as_ref() {
                    if same_tracker_label(existing, label) {
                        return Ok(false);
                    }
                    bail!(
                        "BTC label acknowledgment conflicts for market {}",
                        label.market_id
                    );
                }
                boundary.label = Some(label.clone());
                Ok(true)
            }
        }
    }
}

fn require_tracker_tick(
    tick: &ReferencePriceTick,
    boundary_at: DateTime<Utc>,
    latest_at: DateTime<Utc>,
    boundary: &str,
) -> Result<()> {
    if tick.source != ReferencePriceSource::RtdsChainlink
        || tick.source_timestamp < boundary_at
        || tick.source_timestamp > latest_at
    {
        bail!("invalid durable BTC {boundary} boundary tick");
    }
    Ok(())
}

fn same_tracker_tick(left: &ReferencePriceTick, right: &ReferencePriceTick) -> bool {
    left.tick_id == right.tick_id
        && left.dedup_key == right.dedup_key
        && left.source == right.source
        && left.symbol == right.symbol
        && left.source_timestamp == right.source_timestamp
        && left.price == right.price
}

fn same_tracker_label(left: &BtcMarketLabel, right: &BtcMarketLabel) -> bool {
    left.market_id == right.market_id
        && left.window_start == right.window_start
        && left.window_end == right.window_end
        && left.open_price == right.open_price
        && left.close_price == right.close_price
        && left.outcome == right.outcome
        && left.label_source == right.label_source
        && left.label_version == right.label_version
        && left.source_open_timestamp == right.source_open_timestamp
        && left.source_close_timestamp == right.source_close_timestamp
}

fn tick_precedes(left: &ReferencePriceTick, right: &ReferencePriceTick) -> bool {
    (
        left.source_timestamp,
        left.received_at,
        left.ingest_sequence,
        left.tick_id,
    ) < (
        right.source_timestamp,
        right.received_at,
        right.ingest_sequence,
        right.tick_id,
    )
}

fn boundary_label(
    market: &BtcIntervalMarket,
    open_tick: &ReferencePriceTick,
    close_tick: &ReferencePriceTick,
) -> BtcMarketLabel {
    let outcome = if close_tick.price >= open_tick.price {
        BtcOutcome::Up
    } else {
        BtcOutcome::Down
    };
    BtcMarketLabel {
        market_id: market.market_id.clone(),
        window_start: market.window_start,
        window_end: market.window_end,
        open_price: open_tick.price,
        close_price: close_tick.price,
        outcome,
        label_source: "rtds_chainlink".to_string(),
        label_version: BOUNDARY_LABEL_VERSION.to_string(),
        source_open_timestamp: open_tick.source_timestamp,
        source_close_timestamp: close_tick.source_timestamp,
        label_available_at: close_tick.received_at,
        evidence: serde_json::json!({
            "open_tick_id": open_tick.tick_id,
            "close_tick_id": close_tick.tick_id,
            "open_received_at": open_tick.received_at,
            "close_received_at": close_tick.received_at,
            "rule": "up_when_close_greater_than_or_equal_to_open"
        }),
    }
}

fn clob_subscription(markets: &[BtcIntervalMarket]) -> String {
    serde_json::json!({
        "assets_ids": clob_asset_ids(markets),
        "type": "market",
        "custom_feature_enabled": true
    })
    .to_string()
}

fn clob_asset_ids(markets: &[BtcIntervalMarket]) -> Vec<String> {
    let mut assets = markets
        .iter()
        .flat_map(|market| [&market.up_token_id, &market.down_token_id])
        .cloned()
        .collect::<Vec<_>>();
    assets.sort_unstable();
    assets.dedup();
    assets
}

fn clob_subscription_delta(
    active: &[BtcIntervalMarket],
    desired: &[BtcIntervalMarket],
) -> ClobSubscriptionDelta {
    let active_assets = clob_asset_ids(active);
    let desired_assets = clob_asset_ids(desired);
    let added_assets = desired_assets
        .iter()
        .filter(|asset| active_assets.binary_search(asset).is_err())
        .cloned()
        .collect::<Vec<_>>();
    let removed_assets = active_assets
        .iter()
        .filter(|asset| desired_assets.binary_search(asset).is_err())
        .cloned()
        .collect::<Vec<_>>();
    let mut seen_market_ids = HashSet::with_capacity(desired.len());
    let added_markets = desired
        .iter()
        .filter(|market| {
            (added_assets.binary_search(&market.up_token_id).is_ok()
                || added_assets.binary_search(&market.down_token_id).is_ok())
                && seen_market_ids.insert(market.market_id.as_str())
        })
        .cloned()
        .collect();
    ClobSubscriptionDelta {
        added_assets,
        removed_assets,
        added_markets,
    }
}

fn clob_subscription_operation(assets: &[String], operation: ClobSubscriptionOperation) -> String {
    match operation {
        ClobSubscriptionOperation::Subscribe => serde_json::json!({
            "assets_ids": assets,
            "operation": "subscribe",
            "custom_feature_enabled": true
        }),
        ClobSubscriptionOperation::Unsubscribe => serde_json::json!({
            "assets_ids": assets,
            "operation": "unsubscribe"
        }),
    }
    .to_string()
}

fn register_clob_markets(registry: &mut BookRegistry, markets: &[BtcIntervalMarket]) -> Result<()> {
    for market in markets {
        registry
            .try_register_market(market)
            .with_context(|| format!("invalid CLOB market identity {}", market.market_id))?;
    }
    Ok(())
}

async fn acknowledge_clob_subscriptions(
    repository: &BtcRepository,
    markets: &[BtcIntervalMarket],
    connection_id: Uuid,
    subscribed_at: DateTime<Utc>,
) -> Result<()> {
    let mut market_ids = markets
        .iter()
        .map(|market| market.market_id.clone())
        .collect::<Vec<_>>();
    market_ids.sort_unstable();
    market_ids.dedup();
    repository
        .mark_official_resolution_watches_subscribed(&market_ids, connection_id, subscribed_at)
        .await?;
    Ok(())
}

fn same_market_subscriptions(left: &[BtcIntervalMarket], right: &[BtcIntervalMarket]) -> bool {
    let mut left_keys: Vec<_> = left
        .iter()
        .map(|market| {
            (
                &market.market_id,
                &market.condition_id,
                &market.up_token_id,
                &market.down_token_id,
                market.window_start,
                market.window_end,
            )
        })
        .collect();
    let mut right_keys: Vec<_> = right
        .iter()
        .map(|market| {
            (
                &market.market_id,
                &market.condition_id,
                &market.up_token_id,
                &market.down_token_id,
                market.window_start,
                market.window_end,
            )
        })
        .collect();
    left_keys.sort();
    right_keys.sort();
    left_keys == right_keys
}

fn rtds_subscription() -> String {
    serde_json::json!({
        "action": "subscribe",
        "subscriptions": [{
            "topic": "crypto_prices",
            "type": "update",
            "filters": "btcusdt"
        }, {
            "topic": "crypto_prices_chainlink",
            "type": "*",
            "filters": "{\"symbol\":\"btc/usd\"}"
        }]
    })
    .to_string()
}

fn is_rtds_reference_update(value: &serde_json::Value) -> bool {
    value.get("type").and_then(serde_json::Value::as_str) == Some("update")
        && matches!(
            value.get("topic").and_then(serde_json::Value::as_str),
            Some("crypto_prices" | "crypto_prices_chainlink")
        )
}

fn new_session(
    connection_id: Uuid,
    feed_name: &str,
    endpoint: &str,
    reconnect_ordinal: i32,
    started_at: DateTime<Utc>,
) -> FeedSession {
    FeedSession {
        connection_id,
        feed_name: feed_name.to_string(),
        endpoint: endpoint.to_string(),
        reconnect_ordinal,
        started_at,
        connected_at: None,
        disconnected_at: None,
        messages_received: 0,
        messages_persisted: 0,
        decode_errors: 0,
        integrity_gaps: 0,
        dropped_messages: 0,
        disconnect_reason: None,
        metadata: serde_json::json!({}),
    }
}

fn reconnect_backoff(config: &BtcRuntimeConfig, consecutive_failures: u32) -> StdDuration {
    let exponent = consecutive_failures.saturating_sub(1).min(10);
    let multiplier = 2u32.saturating_pow(exponent);
    config
        .reconnect_initial_delay
        .saturating_mul(multiplier)
        .min(config.reconnect_max_delay)
}

async fn wait_reconnect_backoff(delay: StdDuration, shutdown: &mut watch::Receiver<bool>) -> bool {
    if *shutdown.borrow() {
        return false;
    }
    tokio::select! {
        _ = shutdown.changed() => false,
        _ = sleep(delay) => true,
    }
}

fn duration_milliseconds(duration: StdDuration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

async fn record_error(metrics: &Arc<RwLock<BtcRuntimeMetrics>>, error: anyhow::Error) {
    metrics.write().await.last_error = Some(error.to_string());
}

async fn record_critical_persistence_error(
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    error: anyhow::Error,
) {
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.persistence_errors = runtime_metrics.persistence_errors.saturating_add(1);
    runtime_metrics.last_error = Some(format!("{error:#}"));
}

async fn start_feed_session_or_fail(
    repository: &BtcRepository,
    session: &FeedSession,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
) -> bool {
    match repository.start_feed_session(session).await {
        Ok(()) => true,
        Err(error) => {
            record_critical_persistence_error(
                metrics,
                error.context(format!(
                    "failed to durably start {} feed session {}",
                    session.feed_name, session.connection_id
                )),
            )
            .await;
            false
        }
    }
}

async fn finish_feed_session_or_fail(
    repository: &BtcRepository,
    session: &FeedSession,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
) -> bool {
    match repository.finish_feed_session(session).await {
        Ok(()) => true,
        Err(error) => {
            record_critical_persistence_error(
                metrics,
                error.context(format!(
                    "failed to durably finish {} feed session {}",
                    session.feed_name, session.connection_id
                )),
            )
            .await;
            false
        }
    }
}

fn primary_runtime_failure(metrics: &BtcRuntimeMetrics) -> Option<String> {
    if metrics.persistence_errors == 0
        && metrics.dropped_messages == 0
        && metrics.strategy_errors == 0
    {
        return None;
    }
    Some(format!(
        "persistence_errors={}, dropped_messages={}, strategy_errors={}, last_error={}",
        metrics.persistence_errors,
        metrics.dropped_messages,
        metrics.strategy_errors,
        metrics.last_error.as_deref().unwrap_or("none")
    ))
}

fn chrono_duration(duration: StdDuration) -> Duration {
    Duration::from_std(duration).unwrap_or_else(|_| Duration::seconds(i64::MAX / 1_000))
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use rust_decimal_macros::dec;

    use super::*;
    use crate::btc::types::OrderbookLevel;

    struct TaskDropSignal(Arc<AtomicBool>);

    impl Drop for TaskDropSignal {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }

    #[tokio::test]
    async fn dropping_runtime_handle_aborts_remaining_tasks() {
        let task_started = Arc::new(AtomicBool::new(false));
        let task_dropped = Arc::new(AtomicBool::new(false));
        let started = task_started.clone();
        let dropped = task_dropped.clone();
        let task = tokio::spawn(async move {
            let _drop_signal = TaskDropSignal(dropped);
            started.store(true, Ordering::Relaxed);
            std::future::pending::<()>().await;
        });
        while !task_started.load(Ordering::Relaxed) {
            tokio::task::yield_now().await;
        }

        let running = Arc::new(AtomicBool::new(true));
        let (shutdown, _) = watch::channel(false);
        let handle = BtcRuntimeHandle {
            enabled: true,
            shutdown,
            tasks: vec![task],
            state: Arc::new(RwLock::new(RealtimeState::default())),
            books: Arc::new(RwLock::new(BookRegistry::new(Uuid::new_v4()))),
            metrics: Arc::new(RwLock::new(BtcRuntimeMetrics::default())),
            config: BtcRuntimeConfig::default(),
            running: running.clone(),
        };

        drop(handle);
        for _ in 0..10 {
            if task_dropped.load(Ordering::Relaxed) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(!running.load(Ordering::Relaxed));
        assert!(task_dropped.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn cancelling_shutdown_aborts_tasks_that_have_not_joined() {
        let first_started = Arc::new(AtomicBool::new(false));
        let first_dropped = Arc::new(AtomicBool::new(false));
        let second_started = Arc::new(AtomicBool::new(false));
        let second_dropped = Arc::new(AtomicBool::new(false));
        let tasks = [
            (first_started.clone(), first_dropped.clone()),
            (second_started.clone(), second_dropped.clone()),
        ]
        .into_iter()
        .map(|(started, dropped)| {
            tokio::spawn(async move {
                let _drop_signal = TaskDropSignal(dropped);
                started.store(true, Ordering::Relaxed);
                std::future::pending::<()>().await;
            })
        })
        .collect::<Vec<_>>();
        while !first_started.load(Ordering::Relaxed) || !second_started.load(Ordering::Relaxed) {
            tokio::task::yield_now().await;
        }

        let running = Arc::new(AtomicBool::new(true));
        let (shutdown, _) = watch::channel(false);
        let handle = BtcRuntimeHandle {
            enabled: true,
            shutdown,
            tasks,
            state: Arc::new(RwLock::new(RealtimeState::default())),
            books: Arc::new(RwLock::new(BookRegistry::new(Uuid::new_v4()))),
            metrics: Arc::new(RwLock::new(BtcRuntimeMetrics::default())),
            config: BtcRuntimeConfig::default(),
            running: running.clone(),
        };

        assert!(
            tokio::time::timeout(StdDuration::from_millis(10), handle.shutdown())
                .await
                .is_err()
        );
        for _ in 0..20 {
            if first_dropped.load(Ordering::Relaxed) && second_dropped.load(Ordering::Relaxed) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(!running.load(Ordering::Relaxed));
        assert!(first_dropped.load(Ordering::Relaxed));
        assert!(second_dropped.load(Ordering::Relaxed));
    }

    fn market() -> BtcIntervalMarket {
        let window_start = Utc.timestamp_opt(1_783_902_600, 0).unwrap();
        BtcIntervalMarket {
            event_id: "event".to_string(),
            event_slug: "btc-updown-5m-1783902600".to_string(),
            series_slug: "btc-up-or-down-5m".to_string(),
            market_id: "market".to_string(),
            condition_id: "condition".to_string(),
            window_start,
            window_end: window_start + Duration::minutes(5),
            up_token_id: "up".to_string(),
            down_token_id: "down".to_string(),
            tick_size: dec!(0.01),
            minimum_order_size: Some(dec!(5)),
            resolution_source: "https://data.chain.link/streams/btc-usd".to_string(),
            active: true,
            closed: false,
            accepting_orders: true,
            fees_enabled: true,
            fee_schedule: serde_json::json!({}),
            raw_payload: serde_json::json!({}),
        }
    }

    fn tick(at: DateTime<Utc>, price: rust_decimal::Decimal) -> ReferencePriceTick {
        ReferencePriceTick {
            tick_id: Uuid::new_v4(),
            dedup_key: Uuid::new_v4().to_string(),
            source: ReferencePriceSource::RtdsChainlink,
            symbol: "BTCUSD".to_string(),
            price,
            source_timestamp: at,
            envelope_timestamp: Some(at),
            received_at: at + Duration::milliseconds(10),
            connection_id: Uuid::new_v4(),
            ingest_sequence: 1,
            source_event_id: None,
            raw_payload: serde_json::json!({}),
        }
    }

    fn reference_test_tick(
        source: ReferencePriceSource,
        source_timestamp: DateTime<Utc>,
        received_at: DateTime<Utc>,
        source_event_id: Option<&str>,
        ingest_sequence: u64,
    ) -> ReferencePriceTick {
        ReferencePriceTick {
            tick_id: Uuid::from_u128(u128::from(ingest_sequence).saturating_add(1)),
            dedup_key: format!("reference-test-{ingest_sequence}"),
            source,
            symbol: match source {
                ReferencePriceSource::RtdsChainlink => "BTCUSD",
                ReferencePriceSource::RtdsBinance | ReferencePriceSource::DirectBinance => {
                    "BTCUSDT"
                }
            }
            .to_string(),
            price: dec!(67_000),
            source_timestamp,
            envelope_timestamp: Some(source_timestamp),
            received_at,
            connection_id: Uuid::from_u128(0xfeed),
            ingest_sequence,
            source_event_id: source_event_id.map(str::to_string),
            raw_payload: serde_json::json!({}),
        }
    }

    fn ready_book_registry(market: &BtcIntervalMarket, source_at: DateTime<Utc>) -> BookRegistry {
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(market);
        for token_id in [&market.up_token_id, &market.down_token_id] {
            let events = registry.apply(
                ClobMessage::Book {
                    market_id: market.condition_id.clone(),
                    token_id: token_id.clone(),
                    bids: vec![OrderbookLevel {
                        price: dec!(0.48),
                        size: dec!(10),
                    }],
                    asks: vec![OrderbookLevel {
                        price: dec!(0.52),
                        size: dec!(10),
                    }],
                    source_timestamp: source_at,
                    source_hash: Some(format!("hash-{token_id}")),
                    raw_payload: serde_json::json!({}),
                },
                source_at + Duration::milliseconds(1),
            );
            assert!(events.iter().all(|event| event.applied));
        }
        registry
    }

    fn apply_ready_book_snapshot(
        registry: &mut BookRegistry,
        market: &BtcIntervalMarket,
        token_id: &str,
        source_at: DateTime<Utc>,
    ) {
        let events = registry.apply(
            ClobMessage::Book {
                market_id: market.condition_id.clone(),
                token_id: token_id.to_string(),
                bids: vec![OrderbookLevel {
                    price: dec!(0.48),
                    size: dec!(10),
                }],
                asks: vec![OrderbookLevel {
                    price: dec!(0.52),
                    size: dec!(10),
                }],
                source_timestamp: source_at,
                source_hash: Some(format!("hash-{token_id}")),
                raw_payload: serde_json::json!({}),
            },
            source_at + Duration::milliseconds(1),
        );
        assert!(events.iter().all(|event| event.applied));
    }

    async fn inert_clob_socket() -> ClobSocket {
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let (client, server) =
            tokio::join!(tokio::net::TcpStream::connect(address), listener.accept());
        let client = client.unwrap();
        let (server, _) = server.unwrap();
        let socket = WebSocketStream::from_raw_socket(
            MaybeTlsStream::Plain(client),
            tokio_tungstenite::tungstenite::protocol::Role::Client,
            None,
        )
        .await;
        drop(server);
        socket
    }

    async fn private_clob_epoch(
        market: BtcIntervalMarket,
        registry: BookRegistry,
        checked_at: DateTime<Utc>,
        max_book_age: Duration,
    ) -> ClobEpoch {
        let connection_id = registry.connection_id();
        let connected_instant = Instant::now();
        let watchdog = ClobFeedWatchdog::new(
            connected_instant,
            &registry,
            std::slice::from_ref(&market),
            checked_at,
            max_book_age,
        );
        ClobEpoch {
            connection_id,
            connection_epoch: 1,
            socket: inert_clob_socket().await,
            registry,
            markets: vec![market],
            session: new_session(
                connection_id,
                "polymarket_clob_market",
                "ws://127.0.0.1",
                1,
                checked_at,
            ),
            subscription_stats: ClobSubscriptionStats::default(),
            connected_instant,
            watchdog,
            healthy_epoch: false,
            books_usable: false,
            pending_resolutions: HashMap::new(),
        }
    }

    #[test]
    fn clob_subscription_identity_is_exact_and_order_independent() {
        let first = market();
        let mut second = first.clone();
        second.market_id = "market-next".to_string();
        second.condition_id = "condition-next".to_string();
        second.up_token_id = "up-next".to_string();
        second.down_token_id = "down-next".to_string();
        second.window_start += Duration::minutes(5);
        second.window_end += Duration::minutes(5);

        let mut refreshed_first = first.clone();
        refreshed_first.event_id = "refreshed-event".to_string();
        refreshed_first.tick_size = dec!(0.001);
        refreshed_first.minimum_order_size = Some(dec!(10));
        refreshed_first.fee_schedule = serde_json::json!({"maker": "updated"});
        refreshed_first.raw_payload = serde_json::json!({"refreshed": true});
        assert!(same_clob_subscription_identity(
            &[first.clone(), second.clone(), first.clone()],
            &[second.clone(), refreshed_first],
        ));

        for changed in [
            ("market", "changed-market"),
            ("condition", "changed-condition"),
            ("up", "changed-up"),
            ("down", "changed-down"),
        ] {
            let mut mutated = first.clone();
            match changed.0 {
                "market" => mutated.market_id = changed.1.to_string(),
                "condition" => mutated.condition_id = changed.1.to_string(),
                "up" => mutated.up_token_id = changed.1.to_string(),
                "down" => mutated.down_token_id = changed.1.to_string(),
                _ => unreachable!(),
            }
            assert!(!same_clob_subscription_identity(
                std::slice::from_ref(&first),
                &[mutated],
            ));
        }
    }

    #[test]
    fn successor_admission_enforces_two_connection_cap() {
        let now = Instant::now();
        assert!(should_start_clob_successor(false, false, true, now, now));
        assert!(!should_start_clob_successor(true, false, true, now, now));
        assert!(!should_start_clob_successor(false, true, true, now, now));
        assert!(!should_start_clob_successor(false, false, false, now, now));
        assert!(!should_start_clob_successor(
            false,
            false,
            true,
            now,
            now + StdDuration::from_millis(1),
        ));
    }

    #[test]
    fn healthy_successor_failure_retries_immediately_without_backoff_debt() {
        let config = BtcRuntimeConfig::default();
        let mut failures = 7;
        assert_eq!(
            clob_candidate_retry_action(&config, true, &mut failures),
            ClobRetryAction::ImmediateRecovery
        );
        assert_eq!(failures, 0);

        assert!(matches!(
            clob_candidate_retry_action(&config, false, &mut failures),
            ClobRetryAction::Backoff(delay) if delay == config.reconnect_initial_delay
        ));
        assert_eq!(failures, 1);
    }

    #[test]
    fn successor_attempt_metrics_do_not_overwrite_active_availability() {
        let connected_id = Uuid::new_v4();
        let disconnected_at = Utc::now();
        let mut metrics = BtcRuntimeMetrics {
            reconnects: 4,
            clob_connected_connection_epoch: Some(8),
            clob_connected_connection_id: Some(connected_id),
            clob_last_disconnect_at: Some(disconnected_at),
            clob_last_disconnect_reason: Some("active-history".to_string()),
            ..BtcRuntimeMetrics::default()
        };

        clob_candidate_attempt_metrics(
            &mut metrics,
            ClobDisconnectCause::TransportFailure,
            ClobRetryAction::ImmediateRecovery,
        );

        assert_eq!(metrics.reconnects, 4);
        assert_eq!(metrics.clob_connected_connection_epoch, Some(8));
        assert_eq!(metrics.clob_connected_connection_id, Some(connected_id));
        assert_eq!(metrics.clob_last_disconnect_at, Some(disconnected_at));
        assert_eq!(
            metrics.clob_last_disconnect_reason.as_deref(),
            Some("active-history")
        );
        assert_eq!(metrics.clob_transport_disconnects, 1);
        assert_eq!(metrics.clob_immediate_recoveries_scheduled, 1);
    }

    #[tokio::test]
    async fn private_successor_frames_remain_private() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        let max_book_age = Duration::seconds(2);
        let mut candidate_registry = BookRegistry::new(Uuid::new_v4());
        candidate_registry.register_market(&current);
        let mut candidate = private_clob_epoch(
            current.clone(),
            candidate_registry,
            checked_at,
            max_book_age,
        )
        .await;
        let mut public_registry = BookRegistry::new(Uuid::new_v4());
        public_registry.register_market(&current);
        let public_connection_id = public_registry.connection_id();
        let public_state = RealtimeState::default();
        let public_metrics = BtcRuntimeMetrics::default();
        let (_writer, mut reader) = mpsc::channel::<PersistItem>(4);

        let book = serde_json::json!({
            "event_type": "book",
            "market": current.condition_id,
            "asset_id": current.up_token_id,
            "timestamp": checked_at.timestamp_millis().to_string(),
            "hash": "candidate-up",
            "bids": [{"price": ".48", "size": "10"}],
            "asks": [{"price": ".52", "size": "10"}]
        });
        assert_eq!(
            apply_private_clob_frame(&mut candidate, Message::Text(book.to_string().into())),
            ClobFrameAction::Continue
        );
        assert!(candidate
            .registry
            .checkpoint(&current.up_token_id)
            .is_some());

        let resolution_received_before = Utc::now();
        let resolution_source_at = current.window_end + Duration::seconds(1);
        let resolved = serde_json::json!({
            "event_type": "market_resolved",
            "market": current.condition_id,
            "winning_asset_id": current.down_token_id,
            "winning_outcome": "Down",
            "timestamp": resolution_source_at.timestamp_millis().to_string()
        });
        assert_eq!(
            apply_private_clob_frame(&mut candidate, Message::Text(resolved.to_string().into()),),
            ClobFrameAction::Continue
        );
        let resolution_received_after = Utc::now();
        let buffered = candidate
            .pending_resolutions
            .get(&current.market_id)
            .expect("known private resolution remains buffered before promotion");
        assert!(buffered.received_at >= resolution_received_before);
        assert!(buffered.received_at <= resolution_received_after);
        assert!(matches!(
            &buffered.message,
            ClobMessage::MarketResolved { winning_token_id, .. }
                if winning_token_id == &current.down_token_id
        ));
        let first_resolution_receipt = buffered.received_at;
        let duplicate = serde_json::json!({
            "event_type": "market_resolved",
            "market": current.market_id,
            "winning_asset_id": current.down_token_id,
            "winning_outcome": "Down",
            "timestamp": (resolution_source_at + Duration::milliseconds(1))
                .timestamp_millis()
                .to_string()
        });
        assert_eq!(
            apply_private_clob_frame(&mut candidate, Message::Text(duplicate.to_string().into()),),
            ClobFrameAction::Continue
        );
        assert_eq!(
            candidate
                .pending_resolutions
                .get(&current.market_id)
                .expect("duplicate resolution preserves first observation")
                .received_at,
            first_resolution_receipt
        );

        assert_eq!(public_registry.connection_id(), public_connection_id);
        assert!(public_registry.checkpoint(&current.up_token_id).is_none());
        assert_eq!(public_state, RealtimeState::default());
        assert_eq!(public_metrics.clob_messages_received, 0);
        assert!(matches!(
            reader.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));

        assert_eq!(
            apply_private_clob_frame(
                &mut candidate,
                Message::Text("{malformed".to_string().into()),
            ),
            ClobFrameAction::Disconnect
        );
        assert!(public_registry.checkpoint(&current.up_token_id).is_none());
        assert_eq!(public_metrics.decode_errors, 0);
        assert!(matches!(
            reader.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn conflicting_private_resolution_retires_without_overwriting_first_fact() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&current);
        let mut candidate =
            private_clob_epoch(current.clone(), registry, checked_at, Duration::seconds(2)).await;
        let source_at = current.window_end + Duration::seconds(1);
        let first = serde_json::json!({
            "event_type": "market_resolved",
            "market": current.condition_id,
            "winning_asset_id": current.down_token_id,
            "winning_outcome": "Down",
            "timestamp": source_at.timestamp_millis().to_string()
        });
        assert_eq!(
            apply_private_clob_frame(&mut candidate, Message::Text(first.to_string().into())),
            ClobFrameAction::Continue
        );
        let first_received_at = candidate
            .pending_resolutions
            .get(&current.market_id)
            .expect("first resolution is buffered")
            .received_at;

        let conflicting = serde_json::json!({
            "event_type": "market_resolved",
            "market": current.market_id,
            "winning_asset_id": current.up_token_id,
            "winning_outcome": "Up",
            "timestamp": (source_at + Duration::milliseconds(1))
                .timestamp_millis()
                .to_string()
        });
        assert_eq!(
            apply_private_clob_frame(
                &mut candidate,
                Message::Text(conflicting.to_string().into()),
            ),
            ClobFrameAction::Disconnect
        );
        assert_eq!(
            candidate.session.disconnect_reason.as_deref(),
            Some("successor_resolution_conflict")
        );
        let preserved = candidate
            .pending_resolutions
            .get(&current.market_id)
            .expect("conflict preserves the first resolution fact");
        assert_eq!(preserved.received_at, first_received_at);
        assert!(matches!(
            &preserved.message,
            ClobMessage::MarketResolved { winning_token_id, .. }
                if winning_token_id == &current.down_token_id
        ));
    }

    #[tokio::test]
    async fn promotion_preparation_requires_exact_causal_book_pair() {
        let current = market();
        let publication_boundary = current.window_start + Duration::minutes(1);
        let max_book_age = Duration::seconds(2);
        let registry =
            ready_book_registry(&current, publication_boundary - Duration::milliseconds(1));
        let mut candidate = private_clob_epoch(
            current.clone(),
            registry,
            publication_boundary,
            max_book_age,
        )
        .await;
        let prepared = prepare_clob_successor_publication(
            &candidate,
            std::slice::from_ref(&current),
            publication_boundary,
            max_book_age,
        )
        .unwrap()
        .expect("fresh exact outcome pair should be promotable");
        assert!(prepared.current_market.matches(&current));
        assert_eq!(prepared.checkpoints[0].token_id, current.up_token_id);
        assert_eq!(prepared.checkpoints[1].token_id, current.down_token_id);

        let connection_id = candidate.connection_id;
        candidate.connection_id = Uuid::new_v4();
        assert!(prepare_clob_successor_publication(
            &candidate,
            std::slice::from_ref(&current),
            publication_boundary,
            max_book_age,
        )
        .unwrap()
        .is_none());
        candidate.connection_id = connection_id;

        let mut mutated_identity = current.clone();
        mutated_identity.up_token_id = "different-up".to_string();
        assert!(prepare_clob_successor_publication(
            &candidate,
            &[mutated_identity],
            publication_boundary,
            max_book_age,
        )
        .unwrap()
        .is_none());
        let mut shifted_window = current.clone();
        shifted_window.window_start += Duration::seconds(1);
        shifted_window.window_end += Duration::seconds(1);
        assert!(prepare_clob_successor_publication(
            &candidate,
            &[shifted_window],
            publication_boundary,
            max_book_age,
        )
        .unwrap()
        .is_none());
        assert!(prepare_clob_successor_publication(
            &candidate,
            std::slice::from_ref(&current),
            publication_boundary + max_book_age + Duration::milliseconds(1),
            max_book_age,
        )
        .unwrap()
        .is_none());

        let mut incomplete = BookRegistry::new(Uuid::new_v4());
        incomplete.register_market(&current);
        apply_ready_book_snapshot(
            &mut incomplete,
            &current,
            &current.up_token_id,
            publication_boundary - Duration::milliseconds(1),
        );
        candidate.connection_id = incomplete.connection_id();
        candidate.registry = incomplete;
        assert!(prepare_clob_successor_publication(
            &candidate,
            std::slice::from_ref(&current),
            publication_boundary,
            max_book_age,
        )
        .unwrap()
        .is_none());

        let future_registry =
            ready_book_registry(&current, publication_boundary + Duration::milliseconds(1));
        candidate.connection_id = future_registry.connection_id();
        candidate.registry = future_registry;
        assert!(prepare_clob_successor_publication(
            &candidate,
            std::slice::from_ref(&current),
            publication_boundary,
            max_book_age,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn clob_watchdog_tracks_read_and_pong_deadlines_independently() {
        let now = Instant::now();
        let checked_at = market().window_start + Duration::minutes(1);
        let registry = BookRegistry::new(Uuid::new_v4());
        let mut watchdog =
            ClobFeedWatchdog::new(now, &registry, &[], checked_at, Duration::seconds(2));
        let initial_read_deadline = now + CLOB_READ_IDLE_TIMEOUT;
        assert_eq!(watchdog.read_idle_deadline, initial_read_deadline);
        assert_eq!(watchdog.pong_deadline, None);

        let ping_at = now + StdDuration::from_secs(1);
        watchdog.arm_text_pong(ping_at);
        let pong_deadline = ping_at + CLOB_PONG_TIMEOUT;
        assert_eq!(watchdog.read_idle_deadline, initial_read_deadline);
        assert_eq!(watchdog.pong_deadline, Some(pong_deadline));

        let frame_at = now + StdDuration::from_secs(2);
        watchdog.on_frame(frame_at);
        assert_eq!(
            watchdog.read_idle_deadline,
            frame_at + CLOB_READ_IDLE_TIMEOUT
        );
        assert_eq!(watchdog.pong_deadline, Some(pong_deadline));
        assert!(watchdog.awaiting_text_pong);

        assert!(!watchdog.acknowledge_text_pong("pong"));
        assert!(!watchdog.acknowledge_text_pong("PONG "));
        assert_eq!(watchdog.pong_deadline, Some(pong_deadline));
        assert!(watchdog.acknowledge_text_pong("PONG"));
        assert_eq!(watchdog.pong_deadline, None);
        assert!(!watchdog.awaiting_text_pong);
        assert_eq!(
            watchdog.read_idle_deadline,
            frame_at + CLOB_READ_IDLE_TIMEOUT
        );
    }

    #[test]
    fn clob_watchdog_bootstrap_requires_both_current_books() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        let source_at = checked_at - Duration::milliseconds(1);
        let now = Instant::now();
        let max_book_age = Duration::seconds(2);
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&current);
        let mut watchdog = ClobFeedWatchdog::new(
            now,
            &registry,
            std::slice::from_ref(&current),
            checked_at,
            max_book_age,
        );
        let bootstrap_deadline = now + CLOB_BOOTSTRAP_TIMEOUT;
        assert_eq!(watchdog.bootstrap_deadline, Some(bootstrap_deadline));

        apply_ready_book_snapshot(&mut registry, &current, &current.up_token_id, source_at);
        watchdog.refresh_bootstrap(
            now + StdDuration::from_millis(1),
            &registry,
            std::slice::from_ref(&current),
            checked_at,
            max_book_age,
        );
        assert_eq!(watchdog.bootstrap_deadline, Some(bootstrap_deadline));

        apply_ready_book_snapshot(&mut registry, &current, &current.down_token_id, source_at);
        watchdog.refresh_bootstrap(
            now + StdDuration::from_millis(2),
            &registry,
            std::slice::from_ref(&current),
            checked_at,
            max_book_age,
        );
        assert_eq!(watchdog.bootstrap_deadline, None);
    }

    #[tokio::test]
    async fn once_ready_private_epoch_rearms_bootstrap_retirement_when_books_age_stale() {
        let current = market();
        let ready_at = current.window_start + Duration::minutes(1);
        let ready_instant = Instant::now();
        let max_book_age = Duration::seconds(2);
        let registry = ready_book_registry(&current, ready_at - Duration::milliseconds(1));
        let connection_id = registry.connection_id();
        let watchdog = ClobFeedWatchdog::new(
            ready_instant,
            &registry,
            std::slice::from_ref(&current),
            ready_at,
            max_book_age,
        );
        assert_eq!(watchdog.bootstrap_deadline, None);

        let mut epoch = ClobEpoch {
            connection_id,
            connection_epoch: 1,
            socket: inert_clob_socket().await,
            registry,
            markets: vec![current],
            session: new_session(
                connection_id,
                "polymarket_clob_market",
                "ws://127.0.0.1",
                1,
                ready_at,
            ),
            subscription_stats: ClobSubscriptionStats::default(),
            connected_instant: ready_instant,
            watchdog,
            healthy_epoch: true,
            books_usable: true,
            pending_resolutions: HashMap::new(),
        };
        let stale_at = ready_at + max_book_age + Duration::milliseconds(1);
        let stale_instant = ready_instant + StdDuration::from_millis(2_001);

        epoch.refresh_private_health(stale_at, stale_instant, max_book_age);

        let retirement_deadline = stale_instant + CLOB_BOOTSTRAP_TIMEOUT;
        assert!(!epoch.books_usable);
        assert!(epoch.healthy_epoch);
        assert_eq!(epoch.watchdog.bootstrap_deadline, Some(retirement_deadline));

        epoch.refresh_private_health(
            stale_at + Duration::milliseconds(100),
            stale_instant + StdDuration::from_millis(100),
            max_book_age,
        );
        assert_eq!(epoch.watchdog.bootstrap_deadline, Some(retirement_deadline));
    }

    #[test]
    fn clob_watchdog_omits_bootstrap_deadline_without_unique_current_market() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        let now = Instant::now();
        let max_book_age = Duration::seconds(2);
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&current);
        let mut watchdog = ClobFeedWatchdog::new(
            now,
            &registry,
            std::slice::from_ref(&current),
            checked_at,
            max_book_age,
        );
        assert!(watchdog.bootstrap_deadline.is_some());

        let mut conflicting = current.clone();
        conflicting.market_id = "other-market".to_string();
        conflicting.condition_id = "other-condition".to_string();
        conflicting.up_token_id = "other-up".to_string();
        conflicting.down_token_id = "other-down".to_string();
        watchdog.refresh_bootstrap(
            now + StdDuration::from_millis(1),
            &registry,
            &[current, conflicting],
            checked_at,
            max_book_age,
        );
        assert_eq!(watchdog.bootstrap_market, None);
        assert_eq!(watchdog.bootstrap_deadline, None);

        watchdog.refresh_bootstrap(
            now + StdDuration::from_millis(2),
            &registry,
            &[],
            checked_at,
            max_book_age,
        );
        assert_eq!(watchdog.bootstrap_market, None);
        assert_eq!(watchdog.bootstrap_deadline, None);
    }

    #[test]
    fn stale_or_prior_window_books_cannot_satisfy_current_bootstrap() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        let max_book_age = Duration::seconds(2);
        let mut prior = current.clone();
        prior.market_id = "prior-market".to_string();
        prior.condition_id = "prior-condition".to_string();
        prior.up_token_id = "prior-up".to_string();
        prior.down_token_id = "prior-down".to_string();
        prior.window_start -= Duration::minutes(5);
        prior.window_end -= Duration::minutes(5);

        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&prior);
        registry.register_market(&current);
        let fresh_source_at = checked_at - Duration::milliseconds(1);
        for token_id in [&prior.up_token_id, &prior.down_token_id] {
            apply_ready_book_snapshot(&mut registry, &prior, token_id, fresh_source_at);
        }
        assert!(registry.market_books_ready(&prior, checked_at, max_book_age));
        assert!(!clob_epoch_ready(
            &registry,
            &[prior.clone(), current.clone()],
            checked_at,
            max_book_age,
        ));

        let now = Instant::now();
        let mut watchdog = ClobFeedWatchdog::new(
            now,
            &registry,
            &[prior.clone(), current.clone()],
            checked_at,
            max_book_age,
        );
        assert_eq!(
            watchdog.bootstrap_market,
            Some(ClobMarketIdentity::from(&current))
        );
        let bootstrap_deadline = now + CLOB_BOOTSTRAP_TIMEOUT;
        assert_eq!(watchdog.bootstrap_deadline, Some(bootstrap_deadline));

        let stale_source_at = checked_at - max_book_age - Duration::milliseconds(1);
        for token_id in [&current.up_token_id, &current.down_token_id] {
            apply_ready_book_snapshot(&mut registry, &current, token_id, stale_source_at);
        }
        let active_markets = [prior, current];
        watchdog.refresh_bootstrap(
            now + StdDuration::from_millis(1),
            &registry,
            &active_markets,
            checked_at,
            max_book_age,
        );
        assert_eq!(watchdog.bootstrap_deadline, Some(bootstrap_deadline));
        assert!(!clob_epoch_ready(
            &registry,
            &active_markets,
            checked_at,
            max_book_age,
        ));
    }

    #[tokio::test]
    async fn clob_disconnect_quarantines_old_books_before_reconnect() {
        let market = market();
        let observed_at = market.window_start + Duration::seconds(30);
        let connection_id = Uuid::new_v4();
        let mut registry = BookRegistry::new(connection_id);
        registry.register_market(&market);
        for token_id in [&market.up_token_id, &market.down_token_id] {
            let events = registry.apply(
                ClobMessage::Book {
                    market_id: market.market_id.clone(),
                    token_id: token_id.clone(),
                    bids: vec![OrderbookLevel {
                        price: dec!(0.49),
                        size: dec!(100),
                    }],
                    asks: vec![OrderbookLevel {
                        price: dec!(0.51),
                        size: dec!(100),
                    }],
                    source_timestamp: observed_at,
                    source_hash: None,
                    raw_payload: serde_json::json!({}),
                },
                observed_at + Duration::milliseconds(5),
            );
            assert!(events.iter().all(|event| event.applied));
        }
        assert!(clob_epoch_ready(
            &registry,
            std::slice::from_ref(&market),
            observed_at + Duration::milliseconds(20),
            Duration::seconds(2),
        ));

        let mut realtime = RealtimeState::default();
        realtime.set_market(market.clone());
        realtime.update_books(&registry);
        realtime.update_reference_price(tick(observed_at, dec!(62_000)));
        let mut binance = tick(observed_at, dec!(62_000));
        binance.source = ReferencePriceSource::DirectBinance;
        binance.tick_id = Uuid::new_v4();
        realtime.update_reference_price(binance);
        let state = Arc::new(RwLock::new(realtime));
        let shared_books = Arc::new(RwLock::new(registry.clone()));
        let checked_at = observed_at + Duration::milliseconds(20);

        assert!(
            state
                .read()
                .await
                .readiness(checked_at, Duration::seconds(2), Duration::seconds(2))
                .ready
        );

        quarantine_clob_books_on_disconnect(&mut registry, &state, &shared_books, checked_at).await;

        let readiness = state.read().await.readiness(
            checked_at + Duration::milliseconds(1),
            Duration::seconds(2),
            Duration::seconds(2),
        );
        assert!(!readiness.ready);
        assert!(readiness
            .reasons
            .iter()
            .any(|reason| reason.starts_with("book_integrity:")));
        assert!(shared_books
            .read()
            .await
            .book_readiness()
            .iter()
            .all(|book| book.connection_id == connection_id
                && book.integrity_status == FeedIntegrityStatus::Stale));
        assert!(!clob_epoch_ready(
            &registry,
            std::slice::from_ref(&market),
            checked_at + Duration::milliseconds(1),
            Duration::seconds(2),
        ));
    }

    #[test]
    fn default_config_is_fail_closed_and_valid() {
        let config = BtcRuntimeConfig::default();
        assert!(!config.enabled);
        config.validate().unwrap();
    }

    #[tokio::test]
    async fn unexpected_task_exit_marks_runtime_not_running() {
        let running = Arc::new(AtomicBool::new(true));
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        spawn_runtime_task("test", async {}, running.clone(), metrics.clone())
            .await
            .unwrap();
        assert!(!running.load(Ordering::Relaxed));
        assert_eq!(
            metrics.read().await.last_error.as_deref(),
            Some("BTC runtime test task unexpectedly exited")
        );
    }

    #[tokio::test]
    async fn unexpected_task_exit_preserves_recorded_critical_error_chain() {
        let running = Arc::new(AtomicBool::new(true));
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics {
            last_error: Some("critical write failed: database unavailable".to_string()),
            ..BtcRuntimeMetrics::default()
        }));
        spawn_runtime_task("test", async {}, running.clone(), metrics.clone())
            .await
            .unwrap();
        assert!(!running.load(Ordering::Relaxed));
        assert_eq!(
            metrics.read().await.last_error.as_deref(),
            Some("critical write failed: database unavailable")
        );
    }

    #[derive(Debug)]
    struct FailingStrategyRunner;

    #[async_trait]
    impl BtcStrategyRunner for FailingStrategyRunner {
        async fn on_observation(&self, _observation: StrategyObservation) -> Result<()> {
            bail!("primary strategy persistence failed")
        }
    }

    #[tokio::test]
    async fn failing_playbook_does_not_stop_another_playbook_on_shared_state() {
        let config = BtcRuntimeConfig {
            enabled: true,
            strategy_interval: StdDuration::from_millis(1),
            ..BtcRuntimeConfig::default()
        };
        let state = RealtimeState {
            last_updated_at: Some(Utc::now()),
            ..RealtimeState::default()
        };
        let state = Arc::new(RwLock::new(state));
        let failing = BtcPlaybookRuntimeHandle::start(
            config.clone(),
            Arc::new(FailingStrategyRunner),
            state.clone(),
        )
        .unwrap();
        let healthy =
            BtcPlaybookRuntimeHandle::start(config, Arc::new(NoopStrategyRunner), state.clone())
                .unwrap();

        tokio::time::timeout(StdDuration::from_secs(1), async {
            while failing.is_running() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("failing playbook must terminate");

        assert!(!failing.is_running());
        assert!(healthy.is_running());
        assert!(Arc::ptr_eq(&failing.state, &healthy.state));
        let status = failing.metrics.read().await;
        assert_eq!(
            status.last_error.as_deref(),
            Some("primary strategy persistence failed")
        );
        assert_eq!(status.strategy_errors, 1);
        drop(status);

        assert!(failing.shutdown().await.is_err());
        healthy.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn primary_writer_queue_rejection_records_a_fatal_cause() {
        let (sender, receiver) = mpsc::channel(1);
        drop(receiver);
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let accepted = enqueue(
            &sender,
            PersistItem::ReferenceTick(tick(Utc::now(), dec!(100))),
            &metrics,
        )
        .await;

        assert!(!accepted);
        let status = metrics.read().await;
        assert_eq!(status.dropped_messages, 1);
        assert!(status
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("persistence queue rejected item")));
    }

    #[test]
    fn shutdown_failure_audit_rejects_every_primary_path_counter() {
        assert!(primary_runtime_failure(&BtcRuntimeMetrics::default()).is_none());
        for metrics in [
            BtcRuntimeMetrics {
                persistence_errors: 1,
                ..BtcRuntimeMetrics::default()
            },
            BtcRuntimeMetrics {
                dropped_messages: 1,
                ..BtcRuntimeMetrics::default()
            },
            BtcRuntimeMetrics {
                strategy_errors: 1,
                ..BtcRuntimeMetrics::default()
            },
        ] {
            assert!(primary_runtime_failure(&metrics).is_some());
        }
    }

    #[tokio::test]
    async fn runtime_status_exposes_canonical_subscription_metrics() {
        let updated_at = Utc.timestamp_opt(1_783_902_650, 0).unwrap();
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics {
            clob_subscription_updates: 3,
            clob_active_subscribed_assets: 8,
            clob_last_subscription_update_at: Some(updated_at),
            ..BtcRuntimeMetrics::default()
        }));
        let status = runtime_status_from_inputs(
            Arc::new(RwLock::new(RealtimeState::default())),
            metrics,
            BtcRuntimeConfig::default(),
            Arc::new(AtomicBool::new(true)),
        )
        .await;
        let value = serde_json::to_value(status).unwrap();
        assert_eq!(value["metrics"]["clob_subscription_updates"], 3);
        assert_eq!(value["metrics"]["clob_active_subscribed_assets"], 8);
        assert_eq!(
            value["metrics"]["clob_last_subscription_update_at"],
            serde_json::json!(updated_at)
        );
        assert!(value["metrics"].get("clob_planned_reconnects").is_none());
    }

    #[tokio::test]
    async fn runtime_status_exposes_reference_transport_metrics_by_feed() {
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics {
            rtds_transport: ReferenceTransportMetrics {
                connections_established: 11,
                required_ticks_received: 13,
                last_disconnect_reason: Some(ReferenceDisconnectReason::RequiredDataIdleTimeout),
                ..ReferenceTransportMetrics::default()
            },
            binance_transport: ReferenceTransportMetrics {
                connections_established: 17,
                required_ticks_received: 19,
                last_disconnect_reason: Some(ReferenceDisconnectReason::HeartbeatAckTimeout),
                ..ReferenceTransportMetrics::default()
            },
            rtds_chainlink_ticks_received: 23,
            rtds_binance_ticks_received: 29,
            binance_ticks_received: 31,
            ..BtcRuntimeMetrics::default()
        }));
        let status = runtime_status_from_inputs(
            Arc::new(RwLock::new(RealtimeState::default())),
            metrics,
            BtcRuntimeConfig::default(),
            Arc::new(AtomicBool::new(true)),
        )
        .await;
        let value = serde_json::to_value(status).unwrap();

        assert_eq!(
            value["metrics"]["rtds_transport"]["connections_established"],
            11
        );
        assert_eq!(
            value["metrics"]["rtds_transport"]["required_ticks_received"],
            13
        );
        assert_eq!(
            value["metrics"]["rtds_transport"]["last_disconnect_reason"],
            "required_data_idle_timeout"
        );
        assert_eq!(
            value["metrics"]["binance_transport"]["connections_established"],
            17
        );
        assert_eq!(
            value["metrics"]["binance_transport"]["required_ticks_received"],
            19
        );
        assert_eq!(
            value["metrics"]["binance_transport"]["last_disconnect_reason"],
            "heartbeat_ack_timeout"
        );
        assert_eq!(value["metrics"]["rtds_chainlink_ticks_received"], 23);
        assert_eq!(value["metrics"]["rtds_binance_ticks_received"], 29);
        assert_eq!(value["metrics"]["binance_ticks_received"], 31);
    }

    #[tokio::test]
    async fn closed_market_watch_terminates_while_changes_update_in_place() {
        let (sender, mut receiver) = watch::channel(Vec::<BtcIntervalMarket>::new());
        drop(sender);
        let changed = receiver.changed().await;
        assert_eq!(
            market_watch_disposition(&changed),
            MarketWatchDisposition::Terminate
        );

        let (sender, mut receiver) = watch::channel(Vec::<BtcIntervalMarket>::new());
        sender.send(vec![market()]).unwrap();
        let changed = receiver.changed().await;
        assert_eq!(
            market_watch_disposition(&changed),
            MarketWatchDisposition::UpdateSubscriptions
        );
    }

    #[test]
    fn clob_retry_backoff_uses_consecutive_failures_and_saturates() {
        let config = BtcRuntimeConfig::default();
        let delays = (1..=7)
            .map(|failures| reconnect_backoff(&config, failures))
            .collect::<Vec<_>>();
        assert_eq!(
            delays,
            vec![
                StdDuration::from_secs(1),
                StdDuration::from_secs(2),
                StdDuration::from_secs(4),
                StdDuration::from_secs(8),
                StdDuration::from_secs(16),
                StdDuration::from_secs(30),
                StdDuration::from_secs(30),
            ]
        );
        assert_eq!(
            reconnect_backoff(&config, u32::MAX),
            StdDuration::from_secs(30)
        );
    }

    #[test]
    fn disconnect_cleanup_clears_active_subscription_state_only() {
        let disconnected_at = Utc.timestamp_opt(1_783_902_660, 0).unwrap();
        let updated_at = disconnected_at - Duration::seconds(10);
        let mut metrics = BtcRuntimeMetrics {
            clob_subscription_updates: 3,
            clob_last_subscription_update_at: Some(updated_at),
            clob_active_subscribed_assets: 8,
            clob_connected_connection_epoch: Some(7),
            clob_connected_connection_id: Some(Uuid::new_v4()),
            clob_active_connection_epoch: Some(7),
            clob_active_connection_id: Some(Uuid::new_v4()),
            ..BtcRuntimeMetrics::default()
        };

        clear_clob_connection_metrics(
            &mut metrics,
            disconnected_at,
            "transport_reset",
            Some(disconnected_at),
            2,
        );

        assert_eq!(metrics.clob_active_subscribed_assets, 0);
        assert!(metrics.clob_connected_connection_id.is_none());
        assert!(metrics.clob_active_connection_id.is_none());
        assert_eq!(metrics.clob_subscription_updates, 3);
        assert_eq!(metrics.clob_last_subscription_update_at, Some(updated_at));
        assert_eq!(metrics.clob_consecutive_failures, 2);
    }

    #[test]
    fn clob_retry_classification_recovers_healthy_connections_immediately() {
        let config = BtcRuntimeConfig::default();
        let mut failures = 4;
        assert_eq!(
            clob_retry_action(&config, true, false, &mut failures),
            ClobRetryAction::ImmediateRecovery
        );
        assert_eq!(failures, 4);
        assert_eq!(
            clob_retry_action(&config, false, false, &mut failures),
            ClobRetryAction::Backoff(StdDuration::from_secs(16))
        );
        assert_eq!(failures, 5);
        assert_eq!(
            clob_disconnect_cause(false, false, false, true, false),
            ClobDisconnectCause::Shutdown
        );
        assert_eq!(
            clob_disconnect_cause(false, true, false, false, false),
            ClobDisconnectCause::SubscriptionFailure
        );
        assert_eq!(
            clob_disconnect_cause(true, false, false, false, false),
            ClobDisconnectCause::TransportFailure
        );
        assert_eq!(
            clob_retry_action(&config, false, true, &mut failures),
            ClobRetryAction::Stop
        );
        assert_eq!(failures, 5);
    }

    #[tokio::test]
    async fn healthy_clob_epoch_resets_failures_and_closes_unavailable_interval() {
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let ready_at = Utc.timestamp_opt(1_783_902_610, 0).unwrap();
        let recovery_started_at = Instant::now();
        let ready_instant = recovery_started_at + StdDuration::from_millis(750);
        let mut failures = 6;
        let mut recovery_window =
            ClobRecoveryWindow::open(ready_at - Duration::milliseconds(750), recovery_started_at);

        record_clob_epoch_healthy(
            &metrics,
            Uuid::new_v4(),
            9,
            ready_at,
            ready_instant,
            true,
            &mut failures,
            &mut recovery_window,
        )
        .await;

        assert_eq!(failures, 0);
        assert!(recovery_window.since.is_none());
        let healthy_metrics = metrics.read().await;
        assert_eq!(healthy_metrics.clob_healthy_connections, 1);
        assert_eq!(healthy_metrics.clob_recovery_unavailable_milliseconds, 750);
        assert_eq!(healthy_metrics.clob_consecutive_failures, 0);
        assert_eq!(healthy_metrics.clob_active_connection_epoch, Some(9));
        assert!(healthy_metrics.clob_active_connection_id.is_some());
        assert!(healthy_metrics.clob_recovery_unavailable_since.is_none());
        drop(healthy_metrics);

        let unavailable_at = ready_at + Duration::seconds(1);
        let unavailable_instant = ready_instant + StdDuration::from_millis(250);
        record_clob_epoch_unavailable(
            &metrics,
            unavailable_at,
            unavailable_instant,
            &mut recovery_window,
        )
        .await;
        assert!(metrics.read().await.clob_active_connection_id.is_none());
        record_clob_epoch_healthy(
            &metrics,
            Uuid::new_v4(),
            9,
            unavailable_at + Duration::milliseconds(250),
            unavailable_instant + StdDuration::from_millis(250),
            false,
            &mut failures,
            &mut recovery_window,
        )
        .await;
        let recovered_metrics = metrics.read().await;
        assert_eq!(recovered_metrics.clob_healthy_connections, 1);
        assert_eq!(
            recovered_metrics.clob_recovery_unavailable_milliseconds,
            1_000
        );
        drop(recovered_metrics);

        assert_eq!(
            clob_retry_action(&BtcRuntimeConfig::default(), true, false, &mut failures,),
            ClobRetryAction::ImmediateRecovery
        );
        failures = failures.saturating_add(1);
        assert_eq!(failures, 1);
        assert_eq!(
            reconnect_backoff(&BtcRuntimeConfig::default(), failures),
            StdDuration::from_secs(1)
        );
    }

    #[tokio::test]
    async fn decode_quarantine_revokes_active_clob_usability() {
        let market = market();
        let ready_at = market.window_start + Duration::minutes(2);
        let mut registry = ready_book_registry(&market, ready_at - Duration::milliseconds(10));
        let connection_id = registry.connection_id();
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let started = Instant::now();
        let mut recovery_window =
            ClobRecoveryWindow::open(ready_at - Duration::milliseconds(10), started);
        let mut books_usable = false;
        let mut healthy_epoch = false;
        let mut failures = 3;

        update_clob_usability(
            &registry,
            std::slice::from_ref(&market),
            ready_at,
            started + StdDuration::from_millis(10),
            Duration::seconds(2),
            &mut books_usable,
            &mut healthy_epoch,
            &metrics,
            connection_id,
            4,
            &mut failures,
            &mut recovery_window,
        )
        .await;
        assert!(books_usable);
        assert!(healthy_epoch);

        registry.quarantine(FeedIntegrityStatus::DecodeError);
        let quarantined_at = ready_at + Duration::milliseconds(20);
        update_clob_usability(
            &registry,
            std::slice::from_ref(&market),
            quarantined_at,
            started + StdDuration::from_millis(30),
            Duration::seconds(2),
            &mut books_usable,
            &mut healthy_epoch,
            &metrics,
            connection_id,
            4,
            &mut failures,
            &mut recovery_window,
        )
        .await;

        assert!(!books_usable);
        assert!(healthy_epoch);
        let status = metrics.read().await;
        assert!(status.clob_active_connection_id.is_none());
        assert_eq!(status.clob_recovery_unavailable_since, Some(quarantined_at));
    }

    #[tokio::test]
    async fn clock_aging_revokes_active_clob_usability_without_new_frames() {
        let market = market();
        let ready_at = market.window_start + Duration::minutes(2);
        let registry = ready_book_registry(&market, ready_at - Duration::milliseconds(10));
        let connection_id = registry.connection_id();
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let started = Instant::now();
        let mut recovery_window =
            ClobRecoveryWindow::open(ready_at - Duration::milliseconds(10), started);
        let mut books_usable = false;
        let mut healthy_epoch = false;
        let mut failures = 2;

        update_clob_usability(
            &registry,
            std::slice::from_ref(&market),
            ready_at,
            started + StdDuration::from_millis(10),
            Duration::seconds(2),
            &mut books_usable,
            &mut healthy_epoch,
            &metrics,
            connection_id,
            8,
            &mut failures,
            &mut recovery_window,
        )
        .await;
        assert!(books_usable);

        let stale_at = ready_at + Duration::milliseconds(2_001);
        update_clob_usability(
            &registry,
            std::slice::from_ref(&market),
            stale_at,
            started + StdDuration::from_millis(2_011),
            Duration::seconds(2),
            &mut books_usable,
            &mut healthy_epoch,
            &metrics,
            connection_id,
            8,
            &mut failures,
            &mut recovery_window,
        )
        .await;

        assert!(!books_usable);
        let status = metrics.read().await;
        assert!(status.clob_active_connection_epoch.is_none());
        assert_eq!(status.clob_recovery_unavailable_since, Some(stale_at));
    }

    #[tokio::test]
    async fn clob_reconnect_backoff_honors_requested_shutdown() {
        let (sender, mut shutdown) = watch::channel(false);
        sender.send(true).unwrap();

        let waited = tokio::time::timeout(
            StdDuration::from_millis(20),
            wait_reconnect_backoff(StdDuration::from_secs(30), &mut shutdown),
        )
        .await
        .expect("an already requested shutdown must interrupt reconnect backoff");

        assert!(!waited);
    }

    #[tokio::test]
    async fn critical_persistence_errors_retain_the_full_context_chain() {
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let error = anyhow::anyhow!("database unavailable").context("boundary write failed");
        record_critical_persistence_error(&metrics, error).await;
        assert_eq!(
            metrics.read().await.last_error.as_deref(),
            Some("boundary write failed: database unavailable")
        );
    }

    #[test]
    fn rejects_zero_writer_capacity_and_invalid_urls() {
        let mut config = BtcRuntimeConfig::default();
        config.writer_capacity = 0;
        assert!(config.validate().is_err());
        config.writer_capacity = 1;
        config.clob_ws_url = "https://not-a-websocket".to_string();
        assert!(config.validate().is_err());
        config.clob_ws_url = BtcRuntimeConfig::default().clob_ws_url;
        config.clob_rest_base_url = "ws://not-http".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn resolution_watch_capacity_covers_retention_plus_current_and_next() {
        assert_eq!(resolution_watch_capacity(StdDuration::from_secs(3_600)), 14);
        let mut config = BtcRuntimeConfig::default();
        config.official_resolution_watch_retention = StdDuration::from_secs(719);
        assert!(config.validate().is_err());
    }

    #[test]
    fn subscriptions_are_narrow_and_include_both_outcomes() {
        let market = market();
        let clob: serde_json::Value =
            serde_json::from_str(&clob_subscription(&[market.clone(), market])).unwrap();
        assert_eq!(clob["assets_ids"], serde_json::json!(["down", "up"]));
        let rtds: serde_json::Value = serde_json::from_str(&rtds_subscription()).unwrap();
        assert_eq!(rtds["subscriptions"][0]["filters"], "btcusdt");
    }

    #[test]
    fn dynamic_subscription_payloads_match_provider_contract() {
        let assets = vec!["down".to_string(), "up".to_string()];
        let subscribe: serde_json::Value = serde_json::from_str(&clob_subscription_operation(
            &assets,
            ClobSubscriptionOperation::Subscribe,
        ))
        .unwrap();
        assert_eq!(
            subscribe,
            serde_json::json!({
                "assets_ids": ["down", "up"],
                "operation": "subscribe",
                "custom_feature_enabled": true
            })
        );

        let unsubscribe: serde_json::Value = serde_json::from_str(&clob_subscription_operation(
            &assets,
            ClobSubscriptionOperation::Unsubscribe,
        ))
        .unwrap();
        assert_eq!(
            unsubscribe,
            serde_json::json!({
                "assets_ids": ["down", "up"],
                "operation": "unsubscribe"
            })
        );
    }

    #[test]
    fn subscription_delta_is_deterministic_and_deduplicated() {
        let retained = market();
        let mut added = retained.clone();
        added.market_id = "market-next".to_string();
        added.condition_id = "condition-next".to_string();
        added.up_token_id = "next-up".to_string();
        added.down_token_id = "next-down".to_string();
        added.window_start += Duration::minutes(5);
        added.window_end += Duration::minutes(5);

        let delta = clob_subscription_delta(
            std::slice::from_ref(&retained),
            &[added.clone(), retained.clone(), added.clone()],
        );
        assert_eq!(delta.added_assets, ["next-down", "next-up"]);
        assert!(delta.removed_assets.is_empty());
        assert_eq!(delta.added_markets.len(), 1);
        assert_eq!(delta.added_markets[0].market_id, added.market_id);

        let removal = clob_subscription_delta(&[retained, added.clone()], &[added]);
        assert!(removal.added_assets.is_empty());
        assert_eq!(removal.removed_assets, ["down", "up"]);
        assert!(removal.added_markets.is_empty());
    }

    #[test]
    fn ambiguous_current_market_set_fails_closed() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        assert_eq!(
            unique_current_clob_market(std::slice::from_ref(&current), checked_at)
                .map(|market| market.market_id.as_str()),
            Some("market")
        );
        assert!(
            unique_current_clob_market(&[current.clone(), current.clone()], checked_at).is_some()
        );

        let mut conflicting = current.clone();
        conflicting.market_id = "other-market".to_string();
        conflicting.condition_id = "other-condition".to_string();
        conflicting.up_token_id = "other-up".to_string();
        conflicting.down_token_id = "other-down".to_string();
        assert!(unique_current_clob_market(&[current, conflicting], checked_at).is_none());
    }

    #[test]
    fn clob_session_metadata_keeps_only_bounded_subscription_summary() {
        let updated_at = Utc.timestamp_opt(1_783_902_650, 0).unwrap();
        let stats = ClobSubscriptionStats {
            updates: 7,
            active_assets: 12,
            last_updated_at: Some(updated_at),
        };
        let metadata = clob_session_metadata(
            true,
            ClobDisconnectCause::TransportFailure,
            ClobRetryAction::ImmediateRecovery,
            0,
            &stats,
        );
        assert_eq!(metadata["subscription_updates"], 7);
        assert_eq!(metadata["active_subscribed_assets"], 12);
        assert_eq!(
            metadata["last_subscription_update_at"],
            serde_json::json!(updated_at)
        );
        assert!(metadata.get("subscription_history").is_none());
    }

    #[test]
    fn subscription_identity_ignores_dynamic_gamma_payload_fields() {
        let left = market();
        let mut refreshed = left.clone();
        refreshed.raw_payload = serde_json::json!({"volume": 1000});
        refreshed.accepting_orders = false;
        assert!(same_market_subscriptions(
            std::slice::from_ref(&left),
            std::slice::from_ref(&refreshed),
        ));
        let mut shifted_window = left.clone();
        shifted_window.window_start += Duration::seconds(1);
        shifted_window.window_end += Duration::seconds(1);
        assert!(!same_market_subscriptions(
            std::slice::from_ref(&left),
            &[shifted_window],
        ));
        refreshed.condition_id = "changed-condition".to_string();
        assert!(!same_market_subscriptions(&[left], &[refreshed]));
    }

    #[test]
    fn rtds_acknowledgements_are_not_treated_as_decode_errors() {
        assert!(!is_rtds_reference_update(&serde_json::json!({
            "topic": "crypto_prices_chainlink",
            "type": "subscribe",
            "payload": {}
        })));
        assert!(is_rtds_reference_update(&serde_json::json!({
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "payload": {}
        })));
    }

    #[test]
    fn reference_progress_requires_strict_chainlink_source_time() {
        let current_at = Utc.timestamp_opt(1_783_902_701, 0).unwrap();
        let current = reference_test_tick(
            ReferencePriceSource::RtdsChainlink,
            current_at,
            current_at + Duration::milliseconds(10),
            None,
            1,
        );
        let advancing = reference_test_tick(
            ReferencePriceSource::RtdsChainlink,
            current_at + Duration::milliseconds(1),
            current_at + Duration::milliseconds(11),
            None,
            2,
        );
        let equal = reference_test_tick(
            ReferencePriceSource::RtdsChainlink,
            current_at,
            current_at + Duration::milliseconds(12),
            None,
            3,
        );
        let regressing = reference_test_tick(
            ReferencePriceSource::RtdsChainlink,
            current_at - Duration::milliseconds(1),
            current_at + Duration::milliseconds(13),
            None,
            4,
        );
        let rtds_binance = reference_test_tick(
            ReferencePriceSource::RtdsBinance,
            current_at + Duration::milliseconds(2),
            current_at + Duration::milliseconds(14),
            None,
            5,
        );

        assert!(reference_tick_version_advances(None, &current));
        assert!(reference_tick_progresses(
            None,
            &current,
            ReferenceFeedKind::Rtds
        ));
        assert!(reference_tick_version_advances(Some(&current), &advancing));
        assert!(reference_tick_progresses(
            Some(&current),
            &advancing,
            ReferenceFeedKind::Rtds
        ));
        assert!(!reference_tick_version_advances(Some(&current), &equal));
        assert!(!reference_tick_progresses(
            Some(&current),
            &equal,
            ReferenceFeedKind::Rtds
        ));
        assert!(!reference_tick_version_advances(
            Some(&current),
            &regressing
        ));
        assert!(!reference_tick_progresses(
            Some(&current),
            &regressing,
            ReferenceFeedKind::Rtds
        ));
        assert!(reference_tick_version_advances(None, &rtds_binance));
        assert!(!reference_tick_progresses(
            None,
            &rtds_binance,
            ReferenceFeedKind::Rtds
        ));
    }

    #[test]
    fn binance_progress_requires_increasing_trade_id_and_non_regressing_time() {
        let current_at = Utc.timestamp_opt(1_783_902_701, 0).unwrap();
        let current = reference_test_tick(
            ReferencePriceSource::DirectBinance,
            current_at,
            current_at + Duration::milliseconds(10),
            Some("100"),
            1,
        );
        let same_time_next_id = reference_test_tick(
            ReferencePriceSource::DirectBinance,
            current_at,
            current_at + Duration::milliseconds(11),
            Some("101"),
            2,
        );
        let later_next_id = reference_test_tick(
            ReferencePriceSource::DirectBinance,
            current_at + Duration::milliseconds(1),
            current_at + Duration::milliseconds(12),
            Some("102"),
            3,
        );
        let replay = reference_test_tick(
            ReferencePriceSource::DirectBinance,
            current_at + Duration::milliseconds(1),
            current_at + Duration::milliseconds(13),
            Some("100"),
            4,
        );
        let regressing_id = reference_test_tick(
            ReferencePriceSource::DirectBinance,
            current_at + Duration::milliseconds(1),
            current_at + Duration::milliseconds(14),
            Some("99"),
            5,
        );
        let regressing_time = reference_test_tick(
            ReferencePriceSource::DirectBinance,
            current_at - Duration::milliseconds(1),
            current_at + Duration::milliseconds(15),
            Some("101"),
            6,
        );
        let unparseable = reference_test_tick(
            ReferencePriceSource::DirectBinance,
            current_at + Duration::milliseconds(1),
            current_at + Duration::milliseconds(16),
            Some("not-a-trade-id"),
            7,
        );
        let mut unparseable_current = current.clone();
        unparseable_current.source_event_id = None;

        assert!(!reference_tick_version_advances(None, &unparseable));
        assert!(!reference_tick_progresses(
            None,
            &unparseable,
            ReferenceFeedKind::Binance
        ));

        assert!(reference_tick_version_advances(
            Some(&current),
            &same_time_next_id
        ));
        assert!(reference_tick_progresses(
            Some(&current),
            &same_time_next_id,
            ReferenceFeedKind::Binance
        ));
        assert!(reference_tick_version_advances(
            Some(&current),
            &later_next_id
        ));
        assert!(reference_tick_progresses(
            Some(&current),
            &later_next_id,
            ReferenceFeedKind::Binance
        ));
        for rejected in [&replay, &regressing_id, &regressing_time, &unparseable] {
            assert!(!reference_tick_version_advances(Some(&current), rejected));
            assert!(!reference_tick_progresses(
                Some(&current),
                rejected,
                ReferenceFeedKind::Binance
            ));
        }
        assert!(!reference_tick_version_advances(
            Some(&unparseable_current),
            &same_time_next_id
        ));
        assert!(!reference_tick_progresses(
            Some(&unparseable_current),
            &same_time_next_id,
            ReferenceFeedKind::Binance
        ));

        let mut state = RealtimeState::default();
        assert!(!update_reference_state_and_check_progress(
            &mut state,
            unparseable,
            ReferenceFeedKind::Binance,
            current_at + Duration::milliseconds(20),
            Duration::seconds(2),
        ));
        assert!(state.reference_prices.is_empty());
        assert!(update_reference_state_and_check_progress(
            &mut state,
            current.clone(),
            ReferenceFeedKind::Binance,
            current_at + Duration::milliseconds(20),
            Duration::seconds(2),
        ));
        assert_eq!(
            state
                .reference_prices
                .get(&ReferencePriceSource::DirectBinance),
            Some(&current)
        );
    }

    #[test]
    fn out_of_window_reference_ticks_cannot_mutate_authoritative_state() {
        let checked_at = Utc.timestamp_opt(1_783_902_701, 0).unwrap();
        let max_age = Duration::seconds(2);
        let authoritative = reference_test_tick(
            ReferencePriceSource::RtdsChainlink,
            checked_at - Duration::milliseconds(100),
            checked_at - Duration::milliseconds(50),
            None,
            1,
        );
        let mut baseline = RealtimeState::default();
        assert!(baseline.update_reference_price(authoritative));

        let rejected_ticks = [
            reference_test_tick(
                ReferencePriceSource::RtdsChainlink,
                checked_at - max_age - Duration::milliseconds(1),
                checked_at,
                None,
                2,
            ),
            reference_test_tick(
                ReferencePriceSource::RtdsChainlink,
                checked_at + Duration::milliseconds(1),
                checked_at - max_age - Duration::milliseconds(1),
                None,
                3,
            ),
            reference_test_tick(
                ReferencePriceSource::RtdsChainlink,
                checked_at + max_age + Duration::milliseconds(1),
                checked_at,
                None,
                4,
            ),
            reference_test_tick(
                ReferencePriceSource::RtdsChainlink,
                checked_at + Duration::milliseconds(1),
                checked_at + max_age + Duration::milliseconds(1),
                None,
                5,
            ),
        ];

        for rejected in rejected_ticks {
            let mut state = baseline.clone();
            assert!(!update_reference_state_and_check_progress(
                &mut state,
                rejected,
                ReferenceFeedKind::Rtds,
                checked_at,
                max_age,
            ));
            assert_eq!(state, baseline);
        }
    }

    #[test]
    fn reference_state_rejects_replays_but_retains_non_required_rtds_data() {
        let checked_at = Utc.timestamp_opt(1_783_902_701, 0).unwrap();
        let max_age = Duration::seconds(2);
        let chainlink = reference_test_tick(
            ReferencePriceSource::RtdsChainlink,
            checked_at - Duration::milliseconds(100),
            checked_at - Duration::milliseconds(90),
            None,
            1,
        );
        let direct_binance = reference_test_tick(
            ReferencePriceSource::DirectBinance,
            checked_at - Duration::milliseconds(100),
            checked_at - Duration::milliseconds(80),
            Some("100"),
            2,
        );
        let mut state = RealtimeState::default();
        assert!(state.update_reference_price(chainlink.clone()));
        assert!(state.update_reference_price(direct_binance.clone()));
        let authoritative = state.clone();

        let equal_chainlink_replay = reference_test_tick(
            ReferencePriceSource::RtdsChainlink,
            chainlink.source_timestamp,
            checked_at - Duration::milliseconds(10),
            None,
            3,
        );
        assert!(!update_reference_state_and_check_progress(
            &mut state,
            equal_chainlink_replay,
            ReferenceFeedKind::Rtds,
            checked_at,
            max_age,
        ));
        assert_eq!(state, authoritative);

        let binance_id_replay = reference_test_tick(
            ReferencePriceSource::DirectBinance,
            direct_binance.source_timestamp + Duration::milliseconds(1),
            checked_at - Duration::milliseconds(5),
            Some("100"),
            4,
        );
        assert!(!update_reference_state_and_check_progress(
            &mut state,
            binance_id_replay,
            ReferenceFeedKind::Binance,
            checked_at,
            max_age,
        ));
        assert_eq!(state, authoritative);

        let rtds_binance = reference_test_tick(
            ReferencePriceSource::RtdsBinance,
            checked_at - Duration::milliseconds(4),
            checked_at - Duration::milliseconds(3),
            None,
            5,
        );
        assert!(!update_reference_state_and_check_progress(
            &mut state,
            rtds_binance.clone(),
            ReferenceFeedKind::Rtds,
            checked_at,
            max_age,
        ));
        assert_eq!(
            state
                .reference_prices
                .get(&ReferencePriceSource::RtdsBinance),
            Some(&rtds_binance)
        );
        assert_eq!(
            state
                .reference_prices
                .get(&ReferencePriceSource::RtdsChainlink),
            authoritative
                .reference_prices
                .get(&ReferencePriceSource::RtdsChainlink)
        );
        assert_eq!(
            state
                .reference_prices
                .get(&ReferencePriceSource::DirectBinance),
            authoritative
                .reference_prices
                .get(&ReferencePriceSource::DirectBinance)
        );
    }

    #[test]
    fn reference_watchdog_maintains_independent_deadlines_and_exact_pong_identity() {
        let started_at = Instant::now();
        let mut watchdog = ReferenceFeedWatchdog::new(started_at, ReferenceFeedKind::Rtds);
        assert_eq!(
            watchdog.required_data_deadline,
            started_at + RTDS_REQUIRED_DATA_TIMEOUT
        );
        assert_eq!(
            watchdog.read_idle_deadline,
            started_at + REFERENCE_READ_IDLE_TIMEOUT
        );
        assert!(watchdog.pong_deadline.is_none());
        assert!(watchdog.stable_deadline.is_none());

        let frame_at = started_at + StdDuration::from_secs(2);
        watchdog.on_frame(frame_at);
        assert_eq!(
            watchdog.read_idle_deadline,
            frame_at + REFERENCE_READ_IDLE_TIMEOUT
        );
        assert_eq!(
            watchdog.required_data_deadline,
            started_at + RTDS_REQUIRED_DATA_TIMEOUT
        );

        let data_at = started_at + StdDuration::from_secs(3);
        watchdog.on_required_tick(data_at, ReferenceFeedKind::Rtds);
        assert_eq!(
            watchdog.required_data_deadline,
            data_at + RTDS_REQUIRED_DATA_TIMEOUT
        );
        assert_eq!(
            watchdog.stable_deadline,
            Some(data_at + REFERENCE_STABLE_RESET_AFTER)
        );
        let stable_deadline = watchdog.stable_deadline;
        watchdog.on_required_tick(data_at + StdDuration::from_secs(1), ReferenceFeedKind::Rtds);
        assert_eq!(watchdog.stable_deadline, stable_deadline);

        let probe_at = started_at + StdDuration::from_secs(5);
        assert_eq!(RTDS_HEARTBEAT_MESSAGE, "ping");
        // RTDS requires a text keepalive but does not guarantee a correlated
        // acknowledgement. Required-data and read-idle deadlines detect loss.
        assert!(!watchdog.awaiting_pong());

        let expected = [1, 2, 3, 4, 5, 6, 7, 8];
        watchdog.arm_binary_pong(probe_at, expected);
        assert_eq!(
            watchdog.pong_deadline,
            Some(probe_at + REFERENCE_PONG_TIMEOUT)
        );
        assert!(watchdog.awaiting_pong());
        assert!(!watchdog.acknowledge_binary_pong(&expected[..7]));
        assert!(!watchdog.acknowledge_binary_pong(&[1, 2, 3, 4, 5, 6, 7, 9]));
        assert!(watchdog.awaiting_pong());
        assert!(watchdog.acknowledge_binary_pong(&expected));
        assert!(!watchdog.awaiting_pong());
        assert!(watchdog.pong_deadline.is_none());

        assert!(watchdog.mark_stable());
        assert!(watchdog.stable);
        assert!(watchdog.stable_deadline.is_none());
        assert!(!watchdog.mark_stable());
    }

    #[test]
    fn reference_retry_state_resets_and_classifies_recovery_without_extra_failures() {
        let config = BtcRuntimeConfig::default();
        let connection_id = Uuid::from_u128(1);
        let mut retry_state = ReferenceRetryState::default();
        let expected_delay =
            reference_reconnect_delay(&config, 1, 7, connection_id, ReferenceFeedKind::Rtds);

        assert_eq!(
            reference_retry_action(
                &config,
                false,
                false,
                &mut retry_state,
                7,
                connection_id,
                ReferenceFeedKind::Rtds,
            ),
            ReferenceRetryAction::Backoff(expected_delay)
        );
        assert_eq!(retry_state.consecutive_failures, 1);

        retry_state.reset();
        assert_eq!(retry_state.consecutive_failures, 0);
        assert_eq!(
            reference_retry_action(
                &config,
                true,
                false,
                &mut retry_state,
                8,
                connection_id,
                ReferenceFeedKind::Rtds,
            ),
            ReferenceRetryAction::ImmediateRecovery
        );
        assert_eq!(retry_state.consecutive_failures, 0);
        assert_eq!(
            reference_retry_action(
                &config,
                true,
                true,
                &mut retry_state,
                8,
                connection_id,
                ReferenceFeedKind::Rtds,
            ),
            ReferenceRetryAction::Stop
        );
        assert_eq!(retry_state.consecutive_failures, 0);
    }

    #[test]
    fn reference_retry_jitter_is_bounded_deterministic_and_connection_specific() {
        let config = BtcRuntimeConfig {
            reconnect_max_delay: StdDuration::from_secs(60),
            ..BtcRuntimeConfig::default()
        };
        let failures = 4;
        let connection_a = Uuid::from_u128(1);
        let connection_b = Uuid::from_u128(2);
        let first = reference_reconnect_delay(
            &config,
            failures,
            9,
            connection_a,
            ReferenceFeedKind::Binance,
        );
        let repeated = reference_reconnect_delay(
            &config,
            failures,
            9,
            connection_a,
            ReferenceFeedKind::Binance,
        );
        let other_connection = reference_reconnect_delay(
            &config,
            failures,
            9,
            connection_b,
            ReferenceFeedKind::Binance,
        );
        let base_milliseconds = duration_milliseconds(reconnect_backoff(&config, failures));
        let spread = base_milliseconds * REFERENCE_RETRY_JITTER_PERCENT / 100;
        let delay_milliseconds = duration_milliseconds(first);

        assert_eq!(first, repeated);
        assert!(delay_milliseconds >= base_milliseconds - spread);
        assert!(delay_milliseconds <= base_milliseconds + spread);
        assert_ne!(first, other_connection);
    }

    #[test]
    fn reference_disconnect_reason_serialization_is_stable() {
        let cases = [
            (ReferenceDisconnectReason::Shutdown, "shutdown"),
            (ReferenceDisconnectReason::ConnectTimeout, "connect_timeout"),
            (ReferenceDisconnectReason::ConnectFailed, "connect_failed"),
            (
                ReferenceDisconnectReason::SubscriptionSendTimeout,
                "subscription_send_timeout",
            ),
            (
                ReferenceDisconnectReason::SubscriptionSendFailed,
                "subscription_send_failed",
            ),
            (
                ReferenceDisconnectReason::HeartbeatSendTimeout,
                "heartbeat_send_timeout",
            ),
            (
                ReferenceDisconnectReason::HeartbeatSendFailed,
                "heartbeat_send_failed",
            ),
            (
                ReferenceDisconnectReason::HeartbeatAckTimeout,
                "heartbeat_ack_timeout",
            ),
            (
                ReferenceDisconnectReason::RequiredDataIdleTimeout,
                "required_data_idle_timeout",
            ),
            (
                ReferenceDisconnectReason::ReadIdleTimeout,
                "read_idle_timeout",
            ),
            (ReferenceDisconnectReason::WebsocketEof, "websocket_eof"),
            (ReferenceDisconnectReason::RemoteClose, "remote_close"),
            (
                ReferenceDisconnectReason::TransportReadFailed,
                "transport_read_failed",
            ),
            (
                ReferenceDisconnectReason::CriticalBoundaryIntegrity,
                "critical_boundary_integrity",
            ),
            (
                ReferenceDisconnectReason::CriticalBoundaryPersistence,
                "critical_boundary_persistence",
            ),
            (
                ReferenceDisconnectReason::CriticalWriterQueue,
                "critical_writer_queue",
            ),
            (
                ReferenceDisconnectReason::UnknownDisconnect,
                "unknown_disconnect",
            ),
        ];

        for (reason, expected) in cases {
            assert_eq!(reason.as_str(), expected);
            assert_eq!(serde_json::to_value(reason).unwrap(), expected);
        }
    }

    #[test]
    fn reference_detail_and_session_metadata_remain_bounded() {
        let bounded_unicode = bounded_reference_detail("é".repeat(200));
        assert_eq!(bounded_unicode.len(), 256);
        assert_eq!(bounded_unicode.chars().count(), 128);
        let split_boundary = bounded_reference_detail(format!("{}é", "x".repeat(255)));
        assert_eq!(split_boundary.len(), 255);
        assert!(split_boundary.is_char_boundary(split_boundary.len()));

        let observed_at = Utc.timestamp_opt(1_783_902_701, 0).unwrap();
        let stats = ReferenceSessionStats {
            healthy_epoch: true,
            stable_epoch: true,
            required_ticks: 7,
            heartbeat_probes: 3,
            heartbeat_acknowledgements: 2,
            last_required_tick_at: Some(observed_at),
            last_frame_at: Some(observed_at),
            last_pong_at: Some(observed_at),
            time_to_first_required_tick_milliseconds: Some(12),
            remote_close_code: Some(1001),
        };
        let metadata = reference_session_metadata(
            ReferenceDisconnectReason::RemoteClose,
            Some(&bounded_unicode),
            ReferenceRetryAction::Backoff(StdDuration::from_millis(1_250)),
            4,
            Some(StdDuration::from_secs(45)),
            &stats,
        );
        let fields = metadata.as_object().unwrap();

        assert_eq!(fields.len(), 17);
        assert!(fields
            .values()
            .all(|value| !value.is_array() && !value.is_object()));
        assert_eq!(metadata["disconnect_reason"], "remote_close");
        assert_eq!(metadata["disconnect_cause"], "transport_failure");
        assert_eq!(metadata["disconnect_detail"], bounded_unicode);
        assert_eq!(metadata["retry_delay_ms"], 1_250);
        assert_eq!(metadata["next_action"], "backoff");
    }

    #[test]
    fn boundary_tracker_uses_first_ticks_at_or_after_boundaries() {
        let market = market();
        let mut tracker = BoundaryTracker::default();
        tracker.update_markets(std::slice::from_ref(&market));
        tracker
            .observe_chainlink(
                &tick(market.window_start - Duration::milliseconds(1), dec!(100)),
                Duration::seconds(5),
            )
            .unwrap();
        assert!(tracker.pending_candidates().unwrap().is_empty());
        tracker
            .observe_chainlink(
                &tick(market.window_start + Duration::milliseconds(100), dec!(100)),
                Duration::seconds(5),
            )
            .unwrap();
        let open = tracker.pending_candidates().unwrap();
        assert!(matches!(open.as_slice(), [BoundaryCandidate::Open { .. }]));
        tracker.acknowledge(&open[0]).unwrap();
        tracker
            .observe_chainlink(
                &tick(market.window_end + Duration::milliseconds(100), dec!(100)),
                Duration::seconds(5),
            )
            .unwrap();
        let close = tracker.pending_candidates().unwrap();
        assert!(matches!(
            close.as_slice(),
            [BoundaryCandidate::Close { .. }]
        ));
        tracker.acknowledge(&close[0]).unwrap();
        let label = tracker.pending_candidates().unwrap();
        assert!(matches!(
            label.as_slice(),
            [BoundaryCandidate::Label {
                label: BtcMarketLabel {
                    outcome: BtcOutcome::Up,
                    ..
                },
                ..
            }]
        ));
        tracker.acknowledge(&label[0]).unwrap();
        assert!(tracker.pending_candidates().unwrap().is_empty());
    }

    #[test]
    fn boundary_tracker_refuses_late_open_ticks() {
        let market = market();
        let mut tracker = BoundaryTracker::default();
        tracker.update_markets(std::slice::from_ref(&market));
        tracker
            .observe_chainlink(
                &tick(market.window_start + Duration::seconds(6), dec!(100)),
                Duration::seconds(5),
            )
            .unwrap();
        assert!(tracker.pending_candidates().unwrap().is_empty());
    }

    #[test]
    fn boundary_tracker_hydrates_durable_open_after_restart() {
        let market = market();
        let open = tick(market.window_start + Duration::milliseconds(100), dec!(100));
        let mut tracker = BoundaryTracker::default();
        tracker.update_markets(std::slice::from_ref(&market));
        tracker
            .hydrate_open_references(
                &[(market.market_id.clone(), open.clone())],
                Duration::seconds(5),
            )
            .unwrap();
        let close = tick(market.window_end + Duration::milliseconds(100), dec!(101));
        tracker
            .hydrate_close_references(&[(market.market_id.clone(), close)], Duration::seconds(5))
            .unwrap();
        let items = tracker.pending_candidates().unwrap();
        assert!(matches!(
            items.as_slice(),
            [BoundaryCandidate::Label { label: BtcMarketLabel {
                source_open_timestamp,
                outcome: BtcOutcome::Up,
                ..
            }, .. }] if *source_open_timestamp == open.source_timestamp
        ));
    }

    #[test]
    fn boundary_tracker_converges_process_local_copies_of_one_source_tick() {
        let market = market();
        let pending = tick(market.window_start + Duration::milliseconds(100), dec!(100));
        let mut durable = pending.clone();
        durable.received_at += Duration::milliseconds(20);
        durable.connection_id = Uuid::new_v4();
        durable.ingest_sequence = pending.ingest_sequence + 10;

        let mut tracker = BoundaryTracker::default();
        tracker.update_markets(std::slice::from_ref(&market));
        tracker
            .observe_chainlink(&pending, Duration::seconds(5))
            .unwrap();
        tracker
            .hydrate_open_references(
                &[(market.market_id.clone(), durable.clone())],
                Duration::seconds(5),
            )
            .unwrap();

        let boundary = &tracker.markets[&market.market_id];
        assert_eq!(boundary.open_tick.as_ref(), Some(&durable));
        assert!(boundary.pending_open_tick.is_none());
    }

    #[test]
    fn boundary_tracker_retains_candidate_until_acknowledgment() {
        let market = market();
        let open = tick(market.window_start + Duration::milliseconds(50), dec!(100));
        let mut tracker = BoundaryTracker::default();
        tracker.update_markets(std::slice::from_ref(&market));
        tracker
            .observe_chainlink(&open, Duration::seconds(5))
            .unwrap();

        let first = tracker.pending_candidates().unwrap();
        let second = tracker.pending_candidates().unwrap();
        assert!(matches!(first.as_slice(), [BoundaryCandidate::Open { .. }]));
        assert!(matches!(
            second.as_slice(),
            [BoundaryCandidate::Open { .. }]
        ));
        tracker.acknowledge(&first[0]).unwrap();
        assert!(tracker.pending_candidates().unwrap().is_empty());
    }

    #[test]
    fn boundary_tracker_hydrates_existing_label_without_regenerating_it() {
        let market = market();
        let open = tick(market.window_start + Duration::milliseconds(50), dec!(100));
        let close = tick(market.window_end + Duration::milliseconds(50), dec!(99));
        let label = boundary_label(&market, &open, &close);
        let mut tracker = BoundaryTracker::default();
        tracker.update_markets(std::slice::from_ref(&market));
        tracker
            .hydrate_open_references(&[(market.market_id.clone(), open)], Duration::seconds(5))
            .unwrap();
        tracker
            .hydrate_close_references(&[(market.market_id.clone(), close)], Duration::seconds(5))
            .unwrap();
        tracker
            .hydrate_labels(std::slice::from_ref(&label))
            .unwrap();

        assert!(tracker.pending_candidates().unwrap().is_empty());
        assert_eq!(
            tracker.markets[&market.market_id].label.as_ref(),
            Some(&label)
        );
    }

    #[test]
    fn boundary_tracker_replaces_process_local_label_metadata_with_durable_label() {
        let market = market();
        let open = tick(market.window_start + Duration::milliseconds(50), dec!(100));
        let close = tick(market.window_end + Duration::milliseconds(50), dec!(99));
        let durable = boundary_label(&market, &open, &close);
        let mut process_local = durable.clone();
        process_local.label_available_at += Duration::milliseconds(25);
        process_local.evidence["close_received_at"] =
            serde_json::json!(process_local.label_available_at);

        let mut tracker = BoundaryTracker::default();
        tracker.update_markets(std::slice::from_ref(&market));
        tracker
            .hydrate_open_references(&[(market.market_id.clone(), open)], Duration::seconds(5))
            .unwrap();
        tracker
            .hydrate_close_references(&[(market.market_id.clone(), close)], Duration::seconds(5))
            .unwrap();
        tracker.markets.get_mut(&market.market_id).unwrap().label = Some(process_local);
        tracker
            .hydrate_labels(std::slice::from_ref(&durable))
            .unwrap();

        assert_eq!(
            tracker.markets[&market.market_id].label.as_ref(),
            Some(&durable)
        );
    }

    #[test]
    fn boundary_tracker_rejects_earlier_close_after_label_ack() {
        let market = market();
        let open = tick(market.window_start + Duration::milliseconds(50), dec!(100));
        let close = tick(market.window_end + Duration::milliseconds(200), dec!(101));
        let label = boundary_label(&market, &open, &close);
        let mut tracker = BoundaryTracker::default();
        tracker.update_markets(std::slice::from_ref(&market));
        tracker
            .hydrate_open_references(&[(market.market_id.clone(), open)], Duration::seconds(5))
            .unwrap();
        tracker
            .hydrate_close_references(&[(market.market_id.clone(), close)], Duration::seconds(5))
            .unwrap();
        tracker.hydrate_labels(&[label]).unwrap();

        let earlier = tick(market.window_end + Duration::milliseconds(100), dec!(101));
        assert!(tracker
            .observe_chainlink(&earlier, Duration::seconds(5))
            .is_err());
    }
}
