use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::{self, Write},
    future::Future,
    panic::AssertUnwindSafe,
    pin::Pin,
    str::FromStr,
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
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    net::TcpStream,
    sync::{mpsc, watch, RwLock},
    task::JoinHandle,
    time::{interval, interval_at, sleep, sleep_until, timeout, Instant, MissedTickBehavior},
};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{
        error::ProtocolError, handshake::client::Response as ClobHandshakeResponse,
        protocol::CloseFrame, Error, Message,
    },
    MaybeTlsStream, WebSocketStream,
};
use uuid::Uuid;

use super::{
    binance_spot_l2::{
        parse_depth_snapshot, parse_depth_update, BinanceL2OneSecondFeature,
        BinanceSpotDepthUpdate, BinanceSpotL2Engine, BinanceSpotL2UpdateOutcome,
        BINANCE_SPOT_L2_MAX_INFERENCE_AGE_MILLISECONDS,
    },
    directional_external_runtime::{
        run_directional_external_supervisor, DirectionalExternalRuntimeConfig,
    },
    feeds::{
        parse_binance_agg_trade_with_details, parse_clob_messages, parse_rtds_chainlink_twap_60,
        parse_rtds_reference_tick, BookRegistry, ClobMessage,
    },
    market::{
        discovery_windows, parse_clob_rest_official_resolution, parse_gamma_btc_interval_event,
        parse_gamma_rest_official_resolution, slug_for_window, ClobRestOfficialResolution,
        GammaRestOfficialResolution,
    },
    repository::{
        BtcMarketLabel, BtcOfficialResolutionWatch, BtcRepository, FeedSession,
        PersistedOfficialResolution,
    },
    types::{
        BinanceAggregateTrade, BinanceOneSecondKline, BinanceOneSecondWindow, BtcIntervalMarket,
        BtcOutcome, FeedIntegrityStatus, MarketFeedEvent, MarketFeedEventType, OrderbookCheckpoint,
        Readiness, RealtimeState, ReferencePriceSource, ReferencePriceTick,
        BINANCE_ONE_SECOND_BOOTSTRAP_CAPACITY,
    },
};

const BOUNDARY_LABEL_VERSION: &str = "chainlink_first_tick_at_or_after_boundary_v1";
const BOUNDARY_HYDRATION_RETRY_MAX_DELAY: StdDuration = StdDuration::from_secs(30);
const CRITICAL_WRITE_ATTEMPTS: usize = 3;
const CRITICAL_WRITE_INITIAL_BACKOFF: StdDuration = StdDuration::from_millis(25);
const PRIMARY_PERSISTENCE_RETRY_MAX_DELAY: StdDuration = StdDuration::from_secs(1);
const GAMMA_RESOLUTION_RETRY_INITIAL_BACKOFF: StdDuration = StdDuration::from_secs(30);
const GAMMA_RESOLUTION_RETRY_MAX_BACKOFF: StdDuration = StdDuration::from_secs(300);
const MAX_EXPIRED_RESOLUTION_RECONCILIATIONS_PER_TICK: usize = 4;
const RTDS_HEARTBEAT_MESSAGE: &str = "ping";
const CLOB_CONNECT_TIMEOUT: StdDuration = StdDuration::from_secs(5);
const CLOB_SEND_TIMEOUT: StdDuration = StdDuration::from_secs(5);
const CLOB_BOOTSTRAP_TIMEOUT: StdDuration = StdDuration::from_secs(5);
const CLOB_READ_IDLE_TIMEOUT: StdDuration = StdDuration::from_secs(40);
const CLOB_GRACEFUL_CLOSE_TIMEOUT: StdDuration = StdDuration::from_millis(100);
const CLOB_PROVENANCE_VALUE_MAX_BYTES: usize = 128;
const CLOB_ERROR_REASON_MAX_BYTES: usize = 256;
const REFERENCE_CONNECT_TIMEOUT: StdDuration = StdDuration::from_secs(5);
const REFERENCE_SEND_TIMEOUT: StdDuration = StdDuration::from_secs(5);
const BINANCE_MODEL_RECOVERY_HTTP_TIMEOUT: StdDuration = StdDuration::from_secs(10);
const BINANCE_MODEL_RECOVERY_RETRY_INTERVAL: StdDuration = StdDuration::from_secs(5);
const BINANCE_MODEL_RECOVERY_PAGE_LIMIT: usize = 1_000;
const BINANCE_MODEL_RECOVERY_BUFFER_CAPACITY: usize = 50_000;
const BINANCE_SPOT_L2_SNAPSHOT_TIMEOUT: StdDuration = StdDuration::from_secs(10);
const BINANCE_SPOT_L2_SNAPSHOT_LIMIT: usize = 5_000;
const BINANCE_SPOT_L2_SNAPSHOT_MAX_BYTES: usize = 8 * 1024 * 1024;
const BINANCE_SPOT_L2_BOOTSTRAP_EVENT_CAPACITY: usize = 4_096;
const BINANCE_SPOT_L2_BOOTSTRAP_LEVEL_CAPACITY: usize = 250_000;
const BINANCE_SPOT_L2_FEATURE_TICK: StdDuration = StdDuration::from_millis(100);
const REFERENCE_PONG_TIMEOUT: StdDuration = StdDuration::from_secs(10);
const REFERENCE_READ_IDLE_TIMEOUT: StdDuration = StdDuration::from_secs(40);
const REFERENCE_STABLE_RESET_AFTER: StdDuration = StdDuration::from_secs(30);
const REFERENCE_RETRY_JITTER_PERCENT: u64 = 20;

type ClobSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BtcHeartbeatConfig {
    pub clob_interval: StdDuration,
    pub clob_pong_timeout: StdDuration,
    pub rtds_interval: StdDuration,
    pub binance_interval: StdDuration,
}

impl Default for BtcHeartbeatConfig {
    fn default() -> Self {
        Self {
            // Polymarket's market channel requires a text PING every ten seconds.
            clob_interval: StdDuration::from_secs(10),
            // Permit two venue heartbeat periods plus bounded scheduling/network jitter.
            clob_pong_timeout: StdDuration::from_secs(25),
            rtds_interval: StdDuration::from_secs(5),
            binance_interval: StdDuration::from_secs(20),
        }
    }
}

impl BtcHeartbeatConfig {
    pub const MAX_INTERVAL_SECS: u64 = 30;
    pub const MAX_CLOB_PONG_TIMEOUT_SECS: u64 = 60;

    pub fn validate(self) -> Result<()> {
        for (name, interval) in [
            ("CLOB", self.clob_interval),
            ("RTDS", self.rtds_interval),
            ("Binance", self.binance_interval),
        ] {
            if interval.is_zero() || interval > StdDuration::from_secs(Self::MAX_INTERVAL_SECS) {
                bail!(
                    "BTC {name} heartbeat interval must be between 1 and {} seconds",
                    Self::MAX_INTERVAL_SECS
                );
            }
        }
        if self.clob_pong_timeout <= self.clob_interval
            || self.clob_pong_timeout > StdDuration::from_secs(Self::MAX_CLOB_PONG_TIMEOUT_SECS)
        {
            bail!(
                "BTC CLOB PONG timeout must exceed its heartbeat interval and be at most {} seconds",
                Self::MAX_CLOB_PONG_TIMEOUT_SECS
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BtcRuntimeConfig {
    pub enabled: bool,
    pub gamma_base_url: String,
    pub clob_rest_base_url: String,
    pub clob_ws_url: String,
    pub rtds_ws_url: String,
    pub binance_ws_url: String,
    pub binance_spot_l2_enabled: bool,
    pub binance_spot_l2_ws_url: String,
    pub binance_rest_base_url: String,
    pub discovery_interval: StdDuration,
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
            binance_ws_url: "wss://stream.binance.com/ws/btcusdt@aggTrade".to_string(),
            binance_spot_l2_enabled: false,
            binance_spot_l2_ws_url: "wss://stream.binance.com/ws/btcusdt@depth@100ms".to_string(),
            binance_rest_base_url: "https://data-api.binance.vision".to_string(),
            discovery_interval: StdDuration::from_secs(5),
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
        if !self.binance_rest_base_url.starts_with("http") {
            bail!("BTC realtime Binance REST endpoint must be HTTP(S)");
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
        if self.binance_spot_l2_enabled
            && !self.binance_spot_l2_ws_url.starts_with("ws://")
            && !self.binance_spot_l2_ws_url.starts_with("wss://")
        {
            bail!("BTC realtime Binance spot L2 endpoint must be a websocket URL");
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

/// Operational health for the shared, inference-only Binance spot L2 feed.
///
/// These counters deliberately live outside the persisted feed-session path: the
/// L2 feed supplies model features at runtime and does not own a database data
/// contract.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BinanceSpotL2RuntimeMetrics {
    pub enabled: bool,
    pub connections_established: u64,
    pub connection_failures: u64,
    pub reconnects: u64,
    pub snapshot_requests: u64,
    pub snapshot_successes: u64,
    pub snapshot_failures: u64,
    pub snapshot_behind_buffer: u64,
    pub updates_received: u64,
    pub updates_applied: u64,
    pub updates_discarded: u64,
    pub sequence_gaps: u64,
    pub bootstrap_buffer_overflows: u64,
    pub decode_errors: u64,
    pub ping_frames_received: u64,
    pub pong_frames_sent: u64,
    pub features_published: u64,
    pub consecutive_failures: u32,
    pub synchronized: bool,
    pub active_connection_id: Option<Uuid>,
    pub active_last_update_id: Option<u64>,
    pub last_connected_at: Option<DateTime<Utc>>,
    pub last_synchronized_at: Option<DateTime<Utc>>,
    pub last_update_at: Option<DateTime<Utc>>,
    pub last_feature_published_at: Option<DateTime<Utc>>,
    pub last_disconnect_at: Option<DateTime<Utc>>,
    pub recovery_unavailable_since: Option<DateTime<Utc>>,
    pub last_disconnect_reason: Option<String>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BtcRuntimeMetrics {
    pub markets_discovered: u64,
    pub reference_ticks_received: u64,
    pub clob_messages_received: u64,
    pub feed_events_applied: u64,
    pub checkpoints_queued: u64,
    pub labels_created: u64,
    pub finalized_boundary_ticks_quarantined: u64,
    pub finalized_boundary_outcome_conflicts: u64,
    pub persistence_items_written: u64,
    pub persistence_errors: u64,
    #[serde(default)]
    pub primary_persistence_retryable_errors: u64,
    #[serde(default)]
    pub primary_persistence_consecutive_failures: u32,
    #[serde(default)]
    pub primary_persistence_recoveries: u64,
    #[serde(default)]
    pub primary_persistence_queue_overflows: u64,
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
    #[serde(default)]
    pub clob_book_unavailable_reason: Option<String>,
    #[serde(default)]
    pub clob_book_unavailable_market_id: Option<String>,
    #[serde(default)]
    pub clob_book_unavailable_token_id: Option<String>,
    #[serde(default)]
    pub clob_book_unavailable_integrity_status: Option<FeedIntegrityStatus>,
    #[serde(default)]
    pub clob_book_unavailable_bootstrapped: Option<bool>,
    #[serde(default)]
    pub clob_book_unavailable_has_bid: Option<bool>,
    #[serde(default)]
    pub clob_book_unavailable_has_ask: Option<bool>,
    #[serde(default)]
    pub clob_book_unavailable_source_age_milliseconds: Option<i64>,
    #[serde(default)]
    pub clob_book_unavailable_receipt_age_milliseconds: Option<i64>,
    #[serde(default)]
    pub clob_book_unavailable_source_to_receive_lag_milliseconds: Option<i64>,
    pub clob_active_subscribed_assets: u64,
    pub clob_active_subscription_target_fingerprint_sha256: Option<String>,
    pub clob_active_peer_address: Option<String>,
    pub clob_active_edge_request_id: Option<String>,
    pub clob_active_edge_server: Option<String>,
    pub clob_active_handshake_date: Option<String>,
    pub clob_active_last_data_or_heartbeat_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub clob_active_last_inbound_frame_age_milliseconds: Option<u64>,
    #[serde(default)]
    pub clob_active_last_source_to_receive_lag_milliseconds: Option<i64>,
    pub clob_active_heartbeat_probes: u64,
    pub clob_active_heartbeat_acknowledgements: u64,
    pub clob_active_last_heartbeat_sent_at: Option<DateTime<Utc>>,
    pub clob_active_last_heartbeat_acknowledged_at: Option<DateTime<Utc>>,
    pub clob_active_last_heartbeat_send_lateness_milliseconds: u64,
    pub clob_active_max_heartbeat_send_lateness_milliseconds: u64,
    pub clob_active_last_pong_round_trip_milliseconds: Option<u64>,
    pub rtds_transport: ReferenceTransportMetrics,
    pub binance_transport: ReferenceTransportMetrics,
    #[serde(default)]
    pub binance_spot_l2: BinanceSpotL2RuntimeMetrics,
    pub rtds_chainlink_ticks_received: u64,
    pub rtds_binance_ticks_received: u64,
    pub binance_ticks_received: u64,
    pub binance_model_recovery_required: bool,
    pub binance_model_recovery_attempts: u64,
    pub binance_model_recovery_successes: u64,
    pub binance_model_recovery_failures: u64,
    pub binance_model_recovery_gap_resets: u64,
    pub binance_model_recovery_buffer_overflows: u64,
    pub binance_model_recovery_candles: u64,
    pub binance_model_recovery_last_duration_milliseconds: Option<u64>,
    pub binance_model_recovery_last_started_at: Option<DateTime<Utc>>,
    pub binance_model_recovery_last_completed_at: Option<DateTime<Utc>>,
    pub binance_model_recovery_last_error: Option<String>,
    pub strategy_callbacks: u64,
    pub resolution_watches_active: u64,
    pub resolution_watches_rehydrated: u64,
    pub official_resolutions_websocket: u64,
    pub official_resolutions_rest: u64,
    pub resolution_reconciliation_errors: u64,
    pub resolution_watches_expired: u64,
    #[serde(default)]
    pub boundary_hydration_read_errors: u64,
    #[serde(default)]
    pub boundary_hydration_consecutive_failures: u32,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BtcRuntimeStatus {
    pub enabled: bool,
    pub running: bool,
    pub readiness: Readiness,
    pub metrics: BtcRuntimeMetrics,
}

fn runtime_metrics_snapshot(
    mut metrics: BtcRuntimeMetrics,
    checked_at: DateTime<Utc>,
) -> BtcRuntimeMetrics {
    metrics.clob_active_last_inbound_frame_age_milliseconds = metrics
        .clob_active_last_data_or_heartbeat_at
        .filter(|received_at| *received_at <= checked_at)
        .map(|received_at| {
            u64::try_from((checked_at - received_at).num_milliseconds()).unwrap_or(u64::MAX)
        });
    metrics
}

#[derive(Debug)]
struct ClobRecoveryWindow {
    since: Option<DateTime<Utc>>,
    started_at: Option<Instant>,
    diagnostic: Option<ClobReadinessDiagnostic>,
}

impl ClobRecoveryWindow {
    fn open(since: DateTime<Utc>, started_at: Instant) -> Self {
        Self {
            since: Some(since),
            started_at: Some(started_at),
            diagnostic: None,
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
        self.diagnostic = None;
        self.started_at
            .take()
            .map(|started_at| duration_milliseconds(ended_at.duration_since(started_at)))
            .unwrap_or(0)
    }

    fn update_diagnostic(&mut self, diagnostic: ClobReadinessDiagnostic) -> bool {
        if self.diagnostic.as_ref() == Some(&diagnostic) {
            return false;
        }
        self.diagnostic = Some(diagnostic);
        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ClobReadinessDiagnostic {
    reason: &'static str,
    market_id: Option<String>,
    token_id: Option<String>,
    integrity_status: Option<FeedIntegrityStatus>,
    bootstrapped: Option<bool>,
    has_bid: Option<bool>,
    has_ask: Option<bool>,
    source_age_milliseconds: Option<i64>,
    receipt_age_milliseconds: Option<i64>,
    source_to_receive_lag_milliseconds: Option<i64>,
}

impl ClobReadinessDiagnostic {
    fn missing_current_market() -> Self {
        Self {
            reason: "current_market_not_unique",
            market_id: None,
            token_id: None,
            integrity_status: None,
            bootstrapped: None,
            has_bid: None,
            has_ask: None,
            source_age_milliseconds: None,
            receipt_age_milliseconds: None,
            source_to_receive_lag_milliseconds: None,
        }
    }

    fn transport_unavailable() -> Self {
        Self {
            reason: "transport_unavailable",
            market_id: None,
            token_id: None,
            integrity_status: None,
            bootstrapped: None,
            has_bid: None,
            has_ask: None,
            source_age_milliseconds: None,
            receipt_age_milliseconds: None,
            source_to_receive_lag_milliseconds: None,
        }
    }
}

#[derive(Debug, Default)]
struct ClobSubscriptionStats {
    updates: u64,
    active_assets: usize,
    ignored_foreign_events: u64,
    ignored_superseded_events: u64,
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
    pending_pong_probe_sent_at: Option<Instant>,
}

impl ClobFeedWatchdog {
    fn new(
        now: Instant,
        registry: &BookRegistry,
        markets: &[BtcIntervalMarket],
        checked_at: DateTime<Utc>,
    ) -> Self {
        let mut watchdog = Self {
            bootstrap_market: None,
            bootstrap_deadline: None,
            read_idle_deadline: now + CLOB_READ_IDLE_TIMEOUT,
            pong_deadline: None,
            pending_pong_probe_sent_at: None,
        };
        watchdog.refresh_bootstrap(now, registry, markets, checked_at);
        watchdog
    }

    fn on_frame(&mut self, now: Instant) {
        self.read_idle_deadline = now + CLOB_READ_IDLE_TIMEOUT;
        // Any inbound frame proves that the transport is alive. Some CLOB edge
        // connections continue delivering market data without echoing every
        // application-level PING, so a missed text PONG alone must not retire
        // an otherwise active connection.
        self.pong_deadline = None;
    }

    fn record_text_ping(&mut self, now: Instant, pong_timeout: StdDuration) {
        // Preserve the deadline of the oldest unacknowledged probe. Continuing the
        // documented PING cadence must not turn one missing PONG into an unbounded wait.
        self.pong_deadline.get_or_insert(now + pong_timeout);
        // Correlation is telemetry and remains pending when ordinary inbound traffic
        // independently proves transport liveness.
        self.pending_pong_probe_sent_at = Some(now);
    }

    fn acknowledge_text_pong(&mut self, text: &str, now: Instant) -> Option<StdDuration> {
        if !is_clob_text_pong(text) {
            return None;
        }
        self.pong_deadline = None;
        self.pending_pong_probe_sent_at
            .take()
            .map(|sent_at| now.saturating_duration_since(sent_at))
    }

    fn refresh_bootstrap(
        &mut self,
        now: Instant,
        registry: &BookRegistry,
        markets: &[BtcIntervalMarket],
        checked_at: DateTime<Utc>,
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
        if registry.market_books_bootstrapped(current_market) {
            self.bootstrap_deadline = None;
        } else if identity_changed {
            self.bootstrap_deadline = Some(now + CLOB_BOOTSTRAP_TIMEOUT);
        }
    }
}

fn is_clob_text_pong(text: &str) -> bool {
    text.trim().eq_ignore_ascii_case("PONG")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClobConnectionRole {
    Active,
    Successor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClobCloseAction {
    Skip,
    Initiate,
    AcknowledgeRemote,
}

impl ClobCloseAction {
    fn as_str(self) -> &'static str {
        match self {
            Self::Skip => "skip",
            Self::Initiate => "initiate",
            Self::AcknowledgeRemote => "acknowledge_remote",
        }
    }
}

impl ClobConnectionRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Successor => "successor",
        }
    }
}

#[derive(Debug, Default)]
struct ClobSocketProvenance {
    peer_address: Option<String>,
    edge_request_id: Option<String>,
    edge_server: Option<String>,
    handshake_date: Option<String>,
}

#[derive(Debug)]
struct ClobTransportErrorDetail {
    class: &'static str,
    io_kind: Option<String>,
    os_error_code: Option<i32>,
}

impl ClobTransportErrorDetail {
    fn websocket_eof() -> Self {
        Self {
            class: "websocket_eof",
            io_kind: None,
            os_error_code: None,
        }
    }
}

#[derive(Debug)]
struct ClobSocketTelemetry {
    role: ClobConnectionRole,
    provenance: ClobSocketProvenance,
    subscription_target_fingerprint_sha256: String,
    heartbeat_probes: u64,
    heartbeat_acknowledgements: u64,
    last_heartbeat_sent_at: Option<DateTime<Utc>>,
    last_heartbeat_sent_instant: Option<Instant>,
    last_heartbeat_acknowledged_at: Option<DateTime<Utc>>,
    last_heartbeat_acknowledged_instant: Option<Instant>,
    last_heartbeat_send_lateness: StdDuration,
    max_heartbeat_send_lateness: StdDuration,
    last_pong_round_trip: Option<StdDuration>,
    last_frame_at: Option<DateTime<Utc>>,
    last_frame_instant: Option<Instant>,
    last_data_or_heartbeat_at: Option<DateTime<Utc>>,
    remote_close_observed: bool,
    remote_close_code: Option<u16>,
    remote_close_reason: Option<String>,
    last_transport_error: Option<ClobTransportErrorDetail>,
}

impl ClobSocketTelemetry {
    fn new(markets: &[BtcIntervalMarket]) -> Self {
        Self {
            role: ClobConnectionRole::Successor,
            provenance: ClobSocketProvenance::default(),
            subscription_target_fingerprint_sha256: clob_subscription_fingerprint(markets),
            heartbeat_probes: 0,
            heartbeat_acknowledgements: 0,
            last_heartbeat_sent_at: None,
            last_heartbeat_sent_instant: None,
            last_heartbeat_acknowledged_at: None,
            last_heartbeat_acknowledged_instant: None,
            last_heartbeat_send_lateness: StdDuration::ZERO,
            max_heartbeat_send_lateness: StdDuration::ZERO,
            last_pong_round_trip: None,
            last_frame_at: None,
            last_frame_instant: None,
            last_data_or_heartbeat_at: None,
            remote_close_observed: false,
            remote_close_code: None,
            remote_close_reason: None,
            last_transport_error: None,
        }
    }

    fn record_frame(&mut self, received_at: DateTime<Utc>, received_instant: Instant) {
        self.last_frame_at = Some(received_at);
        self.last_frame_instant = Some(received_instant);
    }

    fn record_heartbeat_probe(
        &mut self,
        sent_at: DateTime<Utc>,
        sent_instant: Instant,
        scheduling_lateness: StdDuration,
    ) -> ClobHeartbeatProbeSample {
        self.heartbeat_probes = self.heartbeat_probes.saturating_add(1);
        self.last_heartbeat_sent_at = Some(sent_at);
        self.last_heartbeat_sent_instant = Some(sent_instant);
        self.last_heartbeat_send_lateness = scheduling_lateness;
        self.max_heartbeat_send_lateness =
            self.max_heartbeat_send_lateness.max(scheduling_lateness);
        ClobHeartbeatProbeSample {
            sent_at,
            scheduling_lateness,
        }
    }

    fn record_heartbeat_acknowledgement(
        &mut self,
        acknowledged_at: DateTime<Utc>,
        acknowledged_instant: Instant,
        round_trip: StdDuration,
    ) -> ClobHeartbeatAcknowledgementSample {
        self.heartbeat_acknowledgements = self.heartbeat_acknowledgements.saturating_add(1);
        self.last_heartbeat_acknowledged_at = Some(acknowledged_at);
        self.last_heartbeat_acknowledged_instant = Some(acknowledged_instant);
        self.last_data_or_heartbeat_at = Some(acknowledged_at);
        self.last_pong_round_trip = Some(round_trip);
        ClobHeartbeatAcknowledgementSample {
            acknowledged_at,
            round_trip: Some(round_trip),
        }
    }

    fn refresh_subscription_target(&mut self, markets: &[BtcIntervalMarket]) {
        self.subscription_target_fingerprint_sha256 = clob_subscription_fingerprint(markets);
    }

    fn mark_active(&mut self) {
        self.role = ClobConnectionRole::Active;
    }

    fn record_remote_close(&mut self, frame: Option<&CloseFrame>) {
        self.remote_close_observed = true;
        self.remote_close_code = frame.map(|frame| u16::from(frame.code));
        self.remote_close_reason = frame
            .map(|frame| frame.reason.as_str())
            .filter(|reason| !reason.is_empty())
            .map(bounded_clob_error_reason);
    }
}

#[derive(Debug, Clone, Copy)]
struct ClobHeartbeatProbeSample {
    sent_at: DateTime<Utc>,
    scheduling_lateness: StdDuration,
}

#[derive(Debug, Clone, Copy)]
struct ClobHeartbeatAcknowledgementSample {
    acknowledged_at: DateTime<Utc>,
    round_trip: Option<StdDuration>,
}

fn clob_transport_error_detail(error: &Error) -> ClobTransportErrorDetail {
    match error {
        Error::Protocol(ProtocolError::ResetWithoutClosingHandshake) => ClobTransportErrorDetail {
            class: "reset_without_closing_handshake",
            io_kind: None,
            os_error_code: None,
        },
        Error::Io(error) => ClobTransportErrorDetail {
            class: "io",
            io_kind: Some(format!("{:?}", error.kind())),
            os_error_code: error.raw_os_error(),
        },
        Error::ConnectionClosed => ClobTransportErrorDetail {
            class: "connection_closed",
            io_kind: None,
            os_error_code: None,
        },
        Error::AlreadyClosed => ClobTransportErrorDetail {
            class: "already_closed",
            io_kind: None,
            os_error_code: None,
        },
        Error::Tls(_) => ClobTransportErrorDetail {
            class: "tls",
            io_kind: None,
            os_error_code: None,
        },
        Error::Capacity(_) => ClobTransportErrorDetail {
            class: "capacity",
            io_kind: None,
            os_error_code: None,
        },
        Error::Protocol(_) => ClobTransportErrorDetail {
            class: "protocol",
            io_kind: None,
            os_error_code: None,
        },
        Error::WriteBufferFull(_) => ClobTransportErrorDetail {
            class: "write_buffer_full",
            io_kind: None,
            os_error_code: None,
        },
        Error::Utf8 => ClobTransportErrorDetail {
            class: "utf8",
            io_kind: None,
            os_error_code: None,
        },
        Error::AttackAttempt => ClobTransportErrorDetail {
            class: "attack_attempt",
            io_kind: None,
            os_error_code: None,
        },
        Error::Url(_) => ClobTransportErrorDetail {
            class: "url",
            io_kind: None,
            os_error_code: None,
        },
        Error::Http(_) => ClobTransportErrorDetail {
            class: "http",
            io_kind: None,
            os_error_code: None,
        },
        Error::HttpFormat(_) => ClobTransportErrorDetail {
            class: "http_format",
            io_kind: None,
            os_error_code: None,
        },
    }
}

fn bounded_clob_text(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    value[..end].to_string()
}

fn bounded_clob_error_reason(reason: &str) -> String {
    bounded_clob_text(reason, CLOB_ERROR_REASON_MAX_BYTES)
}

fn bounded_clob_response_header(
    response: &ClobHandshakeResponse,
    name: &'static str,
) -> Option<String> {
    let value = response.headers().get(name)?.to_str().ok()?;
    (!value.is_empty()).then(|| bounded_clob_text(value, CLOB_PROVENANCE_VALUE_MAX_BYTES))
}

fn clob_socket_provenance(
    socket: &ClobSocket,
    response: &ClobHandshakeResponse,
) -> ClobSocketProvenance {
    let peer_address = match socket.get_ref() {
        MaybeTlsStream::Plain(stream) => stream.peer_addr().ok(),
        MaybeTlsStream::Rustls(stream) => stream.get_ref().0.peer_addr().ok(),
        _ => None,
    }
    .map(|address| address.to_string());
    let mut provenance = clob_response_provenance(response);
    provenance.peer_address = peer_address;
    provenance
}

fn clob_response_provenance(response: &ClobHandshakeResponse) -> ClobSocketProvenance {
    ClobSocketProvenance {
        peer_address: None,
        edge_request_id: bounded_clob_response_header(response, "cf-ray"),
        edge_server: bounded_clob_response_header(response, "server"),
        handshake_date: bounded_clob_response_header(response, "date"),
    }
}

async fn record_active_clob_heartbeat_probe_metrics(
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    sample: ClobHeartbeatProbeSample,
) {
    let scheduling_lateness = duration_milliseconds(sample.scheduling_lateness);
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.clob_active_heartbeat_probes = runtime_metrics
        .clob_active_heartbeat_probes
        .saturating_add(1);
    runtime_metrics.clob_active_last_heartbeat_sent_at = Some(sample.sent_at);
    runtime_metrics.clob_active_last_heartbeat_send_lateness_milliseconds = scheduling_lateness;
    runtime_metrics.clob_active_max_heartbeat_send_lateness_milliseconds = runtime_metrics
        .clob_active_max_heartbeat_send_lateness_milliseconds
        .max(scheduling_lateness);
}

async fn record_active_clob_heartbeat_acknowledgement_metrics(
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    sample: ClobHeartbeatAcknowledgementSample,
) {
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.clob_active_heartbeat_acknowledgements = runtime_metrics
        .clob_active_heartbeat_acknowledgements
        .saturating_add(1);
    runtime_metrics.clob_active_last_heartbeat_acknowledged_at = Some(sample.acknowledged_at);
    runtime_metrics.clob_active_last_pong_round_trip_milliseconds =
        sample.round_trip.map(duration_milliseconds);
    runtime_metrics.clob_active_last_data_or_heartbeat_at = Some(sample.acknowledged_at);
}

fn publish_active_clob_socket_metrics(
    metrics: &mut BtcRuntimeMetrics,
    telemetry: &ClobSocketTelemetry,
) {
    metrics.clob_active_subscription_target_fingerprint_sha256 =
        Some(telemetry.subscription_target_fingerprint_sha256.clone());
    metrics.clob_active_peer_address = telemetry.provenance.peer_address.clone();
    metrics.clob_active_edge_request_id = telemetry.provenance.edge_request_id.clone();
    metrics.clob_active_edge_server = telemetry.provenance.edge_server.clone();
    metrics.clob_active_handshake_date = telemetry.provenance.handshake_date.clone();
    metrics.clob_active_last_data_or_heartbeat_at = telemetry.last_data_or_heartbeat_at;
    metrics.clob_active_heartbeat_probes = telemetry.heartbeat_probes;
    metrics.clob_active_heartbeat_acknowledgements = telemetry.heartbeat_acknowledgements;
    metrics.clob_active_last_heartbeat_sent_at = telemetry.last_heartbeat_sent_at;
    metrics.clob_active_last_heartbeat_acknowledged_at = telemetry.last_heartbeat_acknowledged_at;
    metrics.clob_active_last_heartbeat_send_lateness_milliseconds =
        duration_milliseconds(telemetry.last_heartbeat_send_lateness);
    metrics.clob_active_max_heartbeat_send_lateness_milliseconds =
        duration_milliseconds(telemetry.max_heartbeat_send_lateness);
    metrics.clob_active_last_pong_round_trip_milliseconds =
        telemetry.last_pong_round_trip.map(duration_milliseconds);
}

#[derive(Debug)]
enum ClobSendFailure {
    Shutdown,
    Timeout,
    Transport {
        error: String,
        detail: ClobTransportErrorDetail,
    },
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
                Ok(Err(error)) => Err(ClobSendFailure::Transport {
                    detail: clob_transport_error_detail(&error),
                    error: bounded_clob_error_reason(&error.to_string()),
                }),
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
    telemetry: ClobSocketTelemetry,
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
        self.watchdog
            .refresh_bootstrap(checked_instant, &self.registry, &self.markets, checked_at);
        self.books_usable =
            clob_epoch_ready(&self.registry, &self.markets, checked_at, max_book_age);
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
        epoch: Box<ClobEpoch>,
        connect_latency: StdDuration,
    },
    Failed(Box<ClobConnectFailure>),
}

#[derive(Debug)]
struct ClobConnectFailure {
    session: FeedSession,
    reason: String,
    kind: ClobConnectFailureKind,
    telemetry: ClobSocketTelemetry,
}

async fn connect_clob_epoch(
    config: BtcRuntimeConfig,
    desired_markets: Vec<BtcIntervalMarket>,
    connection_epoch: i32,
    mut shutdown: watch::Receiver<bool>,
) -> ClobConnectOutcome {
    let connection_id = Uuid::new_v4();
    let attempt_started_at = Instant::now();
    let mut telemetry = ClobSocketTelemetry::new(&desired_markets);
    let mut session = new_session(
        connection_id,
        "polymarket_clob_market",
        &config.clob_ws_url,
        connection_epoch,
        Utc::now(),
    );
    let mut registry = BookRegistry::new(connection_id);
    if let Err(error) = register_clob_markets(&mut registry, &desired_markets) {
        let reason = bounded_clob_error_reason(&format!("invalid_subscription_identity:{error}"));
        session.disconnected_at = Some(Utc::now());
        session.disconnect_reason = Some(reason.clone());
        return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
            session,
            reason,
            kind: ClobConnectFailureKind::Identity,
            telemetry,
        }));
    }
    if *shutdown.borrow() {
        session.disconnected_at = Some(Utc::now());
        session.disconnect_reason = Some("shutdown".to_string());
        return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
            session,
            reason: "shutdown".to_string(),
            kind: ClobConnectFailureKind::Shutdown,
            telemetry,
        }));
    }
    let connect_result = tokio::select! {
        biased;
        _ = shutdown.changed() => None,
        result = timeout(CLOB_CONNECT_TIMEOUT, connect_async(&config.clob_ws_url)) => Some(result),
    };
    let (mut socket, response) = match connect_result {
        None => {
            session.disconnected_at = Some(Utc::now());
            session.disconnect_reason = Some("shutdown".to_string());
            return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
                session,
                reason: "shutdown".to_string(),
                kind: ClobConnectFailureKind::Shutdown,
                telemetry,
            }));
        }
        Some(Err(_)) => {
            session.disconnected_at = Some(Utc::now());
            session.disconnect_reason = Some("connect_timeout".to_string());
            return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
                session,
                reason: "connect_timeout".to_string(),
                kind: ClobConnectFailureKind::Connect,
                telemetry,
            }));
        }
        Some(Ok(Err(error))) => {
            let reason = bounded_clob_error_reason(&format!("connect_failed:{error}"));
            if let Error::Http(response) = &error {
                telemetry.provenance = clob_response_provenance(response);
            }
            telemetry.last_transport_error = Some(clob_transport_error_detail(&error));
            session.disconnected_at = Some(Utc::now());
            session.disconnect_reason = Some(reason.clone());
            return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
                session,
                reason,
                kind: ClobConnectFailureKind::Connect,
                telemetry,
            }));
        }
        Some(Ok(Ok(value))) => value,
    };
    telemetry.provenance = clob_socket_provenance(&socket, &response);
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
            return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
                session,
                reason: "shutdown".to_string(),
                kind: ClobConnectFailureKind::Shutdown,
                telemetry,
            }));
        }
        Err(ClobSendFailure::Timeout) => {
            session.disconnected_at = Some(Utc::now());
            session.disconnect_reason = Some("subscription_send_timeout".to_string());
            return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
                session,
                reason: "subscription_send_timeout".to_string(),
                kind: ClobConnectFailureKind::Subscription,
                telemetry,
            }));
        }
        Err(ClobSendFailure::Transport { error, detail }) => {
            let reason = bounded_clob_error_reason(&format!("subscription_send_failed:{error}"));
            telemetry.last_transport_error = Some(detail);
            session.disconnected_at = Some(Utc::now());
            session.disconnect_reason = Some(reason.clone());
            return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
                session,
                reason,
                kind: ClobConnectFailureKind::Subscription,
                telemetry,
            }));
        }
    }
    let watchdog_started = Instant::now();
    let watchdog = ClobFeedWatchdog::new(watchdog_started, &registry, &desired_markets, Utc::now());
    let pending_resolution_capacity = desired_markets.len();
    ClobConnectOutcome::Connected {
        epoch: Box::new(ClobEpoch {
            connection_id,
            connection_epoch,
            socket,
            registry,
            markets: desired_markets,
            session,
            subscription_stats: ClobSubscriptionStats::default(),
            telemetry,
            connected_instant,
            watchdog,
            healthy_epoch: false,
            books_usable: false,
            pending_resolutions: HashMap::with_capacity(pending_resolution_capacity),
        }),
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
    read_idle_deadline: Instant,
    pong_deadline: Option<Instant>,
    stable_deadline: Option<Instant>,
    expected_pong: Option<ReferencePongExpectation>,
    stable: bool,
}

impl ReferenceFeedWatchdog {
    fn new(now: Instant) -> Self {
        Self {
            read_idle_deadline: now + REFERENCE_READ_IDLE_TIMEOUT,
            pong_deadline: None,
            stable_deadline: None,
            expected_pong: None,
            stable: false,
        }
    }

    fn on_frame(&mut self, now: Instant) {
        self.read_idle_deadline = now + REFERENCE_READ_IDLE_TIMEOUT;
        // Inbound traffic is authoritative transport-liveness evidence even
        // when an intermediary does not return the matching control PONG.
        self.pong_deadline = None;
        self.expected_pong = None;
    }

    fn on_required_tick(&mut self, now: Instant) {
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
    heartbeat: BtcHeartbeatConfig,
    repository: BtcRepository,
    books: Option<Arc<RwLock<BookRegistry>>>,
    state: Option<Arc<RwLock<RealtimeState>>>,
    directional_external: DirectionalExternalRuntimeConfig,
}

impl BtcRuntime {
    pub fn new(
        config: BtcRuntimeConfig,
        heartbeat: BtcHeartbeatConfig,
        repository: BtcRepository,
    ) -> Self {
        Self {
            config,
            heartbeat,
            repository,
            books: None,
            state: None,
            directional_external: DirectionalExternalRuntimeConfig::default(),
        }
    }

    /// Uses caller-owned shared state so every playbook observes one canonical feed runtime.
    pub fn with_shared_state(mut self, state: Arc<RwLock<RealtimeState>>) -> Self {
        self.state = Some(state);
        self
    }

    /// Binds global, secret-bearing source configuration outside the immutable process config.
    pub fn with_directional_external(mut self, config: DirectionalExternalRuntimeConfig) -> Self {
        self.directional_external = config;
        self
    }

    /// Uses a caller-owned registry so every paper venue reads the same arrival-time book state.
    pub fn with_shared_book_registry(mut self, books: Arc<RwLock<BookRegistry>>) -> Self {
        self.books = Some(books);
        self
    }

    pub async fn start(self) -> Result<BtcRuntimeHandle> {
        self.config.validate()?;
        self.heartbeat.validate()?;
        self.repository.healthcheck().await?;
        self.directional_external.validate()?;

        let state = self
            .state
            .unwrap_or_else(|| Arc::new(RwLock::new(RealtimeState::default())));
        if self.directional_external.enabled {
            let bootstrap_end = Utc::now();
            let bootstrap_start = bootstrap_end - chrono::Duration::minutes(62);
            match self
                .repository
                .load_directional_external_chainlink_mid_history(bootstrap_start, bootstrap_end)
                .await
            {
                Ok(ticks) => {
                    let mut realtime = state.write().await;
                    for tick in ticks {
                        if let Err(error) =
                            realtime.directional_external.observe_rtds_chainlink(&tick)
                        {
                            tracing::warn!(
                                error = %error,
                                "directional Chainlink midpoint bootstrap rejected a tick"
                            );
                        }
                    }
                }
                Err(error) => tracing::warn!(
                    error = %error,
                    "directional Chainlink midpoint bootstrap failed closed"
                ),
            }
        }
        let books = self
            .books
            .unwrap_or_else(|| Arc::new(RwLock::new(BookRegistry::new(Uuid::new_v4()))));
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let running = Arc::new(AtomicBool::new(true));
        let boundaries = Arc::new(RwLock::new(BoundaryTracker::default()));
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (market_tx, market_rx) = watch::channel(Vec::<BtcIntervalMarket>::new());
        let (writer_tx, writer_rx) = mpsc::channel(self.config.writer_capacity);
        let mut tasks = vec![
            spawn_runtime_task(
                "writer",
                run_writer(
                    self.repository.clone(),
                    writer_rx,
                    state.clone(),
                    metrics.clone(),
                    shutdown_rx.clone(),
                ),
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
                    self.heartbeat.clob_interval,
                    self.heartbeat.clob_pong_timeout,
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
                    self.heartbeat.rtds_interval,
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
                    self.heartbeat.binance_interval,
                    self.repository.clone(),
                    writer_tx.clone(),
                    state.clone(),
                    metrics.clone(),
                    shutdown_rx.clone(),
                ),
                running.clone(),
                metrics.clone(),
            ),
            spawn_runtime_task(
                "directional_external",
                run_directional_external_supervisor(
                    self.directional_external,
                    state.clone(),
                    shutdown_rx.clone(),
                ),
                running.clone(),
                metrics.clone(),
            ),
        ];
        if self.config.binance_spot_l2_enabled {
            tasks.push(spawn_runtime_task(
                "binance_spot_l2",
                run_binance_spot_l2_supervisor(
                    self.config.clone(),
                    state.clone(),
                    metrics.clone(),
                    shutdown_rx.clone(),
                ),
                running.clone(),
                metrics.clone(),
            ));
        }
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
        let checked_at = Utc::now();
        let readiness = state.readiness(
            checked_at,
            chrono_duration(self.config.max_book_age),
            chrono_duration(self.config.max_reference_age),
        );
        BtcRuntimeStatus {
            enabled: self.enabled,
            running: self.running.load(Ordering::Relaxed),
            readiness,
            metrics: runtime_metrics_snapshot(self.metrics.read().await.clone(), checked_at),
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
    let checked_at = Utc::now();
    let readiness = state.readiness(
        checked_at,
        chrono_duration(config.max_book_age),
        chrono_duration(config.max_reference_age),
    );
    BtcRuntimeStatus {
        enabled: config.enabled,
        running: running.load(Ordering::Relaxed),
        readiness,
        metrics: runtime_metrics_snapshot(metrics.read().await.clone(), checked_at),
    }
}

#[derive(Debug)]
enum PersistItem {
    ReferenceTick(ReferencePriceTick),
    FeedEvent(super::types::MarketFeedEvent),
    Checkpoint(super::types::OrderbookCheckpoint),
}

impl PersistItem {
    fn kind(&self) -> &'static str {
        match self {
            Self::ReferenceTick(_) => "reference tick",
            Self::FeedEvent(_) => "feed event",
            Self::Checkpoint(_) => "orderbook checkpoint",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PersistEnqueueOutcome {
    Queued,
    Saturated,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimaryPersistenceOutcome {
    Persisted,
    Shutdown,
    Fatal,
}

async fn run_writer(
    repository: BtcRepository,
    mut receiver: mpsc::Receiver<PersistItem>,
    state: Arc<RwLock<RealtimeState>>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
    mut shutdown: watch::Receiver<bool>,
) {
    while let Some(item) = receiver.recv().await {
        let outcome = persist_primary_item_with_retry(
            || persist_item(&repository, &item),
            item.kind(),
            &state,
            &metrics,
            &mut shutdown,
        )
        .await;
        match outcome {
            PrimaryPersistenceOutcome::Persisted => {}
            PrimaryPersistenceOutcome::Shutdown => {
                receiver.close();
                let abandoned_items = u64::try_from(receiver.len())
                    .unwrap_or(u64::MAX)
                    .saturating_add(1);
                record_primary_persistence_shutdown_abandonment(
                    &metrics,
                    item.kind(),
                    abandoned_items,
                )
                .await;
                return;
            }
            PrimaryPersistenceOutcome::Fatal => return,
        }
    }
}

async fn persist_item(repository: &BtcRepository, item: &PersistItem) -> Result<()> {
    match item {
        PersistItem::ReferenceTick(tick) => {
            repository.insert_reference_tick(tick).await.map(|_| ())
        }
        PersistItem::FeedEvent(event) => repository.insert_feed_event(event).await.map(|_| ()),
        PersistItem::Checkpoint(checkpoint) => repository
            .insert_orderbook_checkpoint(checkpoint, "websocket_book")
            .await
            .map(|_| ()),
    }
}

async fn persist_primary_item_with_retry<Operation, OperationFuture>(
    mut operation: Operation,
    item_kind: &'static str,
    state: &Arc<RwLock<RealtimeState>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    shutdown: &mut watch::Receiver<bool>,
) -> PrimaryPersistenceOutcome
where
    Operation: FnMut() -> OperationFuture,
    OperationFuture: Future<Output = Result<()>>,
{
    let mut recovering = false;
    loop {
        if recovering && *shutdown.borrow() {
            return PrimaryPersistenceOutcome::Shutdown;
        }
        // Never cancel an in-flight idempotent insert: a transport error can be
        // ambiguous about whether PostgreSQL committed it. Shutdown only
        // interrupts the bounded delay between complete attempts.
        let result = operation().await;
        match result {
            Ok(()) => {
                record_primary_persistence_success(state, metrics).await;
                return PrimaryPersistenceOutcome::Persisted;
            }
            Err(error) if is_retryable_primary_persistence_error(&error) => {
                let (entered_degraded_state, consecutive_failures) =
                    record_primary_persistence_retry(state, metrics).await;
                if entered_degraded_state {
                    tracing::warn!(
                        item_kind,
                        error = %error,
                        "primary persistence writer entered retry recovery"
                    );
                }
                let delay = primary_persistence_retry_delay(consecutive_failures);
                if !wait_reconnect_backoff(delay, shutdown).await {
                    return PrimaryPersistenceOutcome::Shutdown;
                }
                recovering = true;
            }
            Err(error) => {
                record_primary_persistence_fatal(state, metrics, item_kind, &error).await;
                return PrimaryPersistenceOutcome::Fatal;
            }
        }
    }
}

async fn record_primary_persistence_retry(
    state: &Arc<RwLock<RealtimeState>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
) -> (bool, u32) {
    let mut runtime_metrics = metrics.write().await;
    let entered_degraded_state = runtime_metrics.primary_persistence_consecutive_failures == 0;
    runtime_metrics.primary_persistence_retryable_errors = runtime_metrics
        .primary_persistence_retryable_errors
        .saturating_add(1);
    runtime_metrics.primary_persistence_consecutive_failures = runtime_metrics
        .primary_persistence_consecutive_failures
        .saturating_add(1);
    let consecutive_failures = runtime_metrics.primary_persistence_consecutive_failures;
    let health_changed = if entered_degraded_state {
        let mut realtime = state.write().await;
        if realtime.primary_persistence_degraded {
            false
        } else {
            realtime.primary_persistence_degraded = true;
            realtime.last_updated_at = Some(Utc::now());
            true
        }
    } else {
        false
    };
    (health_changed, consecutive_failures)
}

async fn record_primary_persistence_success(
    state: &Arc<RwLock<RealtimeState>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
) {
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.persistence_items_written =
        runtime_metrics.persistence_items_written.saturating_add(1);
    let recovered = runtime_metrics.primary_persistence_consecutive_failures > 0;
    if recovered {
        runtime_metrics.primary_persistence_recoveries = runtime_metrics
            .primary_persistence_recoveries
            .saturating_add(1);
    }
    runtime_metrics.primary_persistence_consecutive_failures = 0;
    if recovered && runtime_metrics.primary_persistence_queue_overflows == 0 {
        let mut realtime = state.write().await;
        if realtime.primary_persistence_degraded {
            realtime.primary_persistence_degraded = false;
            realtime.last_updated_at = Some(Utc::now());
            tracing::info!("primary persistence writer recovered");
        }
    }
}

async fn record_primary_persistence_fatal(
    state: &Arc<RwLock<RealtimeState>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    item_kind: &'static str,
    error: &anyhow::Error,
) {
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.persistence_errors = runtime_metrics.persistence_errors.saturating_add(1);
    runtime_metrics.last_error = Some(format!(
        "BTC primary {item_kind} persistence failed: {error:#}"
    ));
    let mut realtime = state.write().await;
    realtime.primary_persistence_degraded = true;
    realtime.last_updated_at = Some(Utc::now());
}

async fn record_primary_persistence_shutdown_abandonment(
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    item_kind: &'static str,
    abandoned_items: u64,
) {
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.dropped_messages = runtime_metrics
        .dropped_messages
        .saturating_add(abandoned_items);
    runtime_metrics.last_error.get_or_insert_with(|| {
        format!(
            "BTC primary persistence shutdown abandoned {abandoned_items} queued item(s), beginning with {item_kind}"
        )
    });
}

async fn enqueue(
    sender: &mpsc::Sender<PersistItem>,
    item: PersistItem,
    state: &Arc<RwLock<RealtimeState>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
) -> PersistEnqueueOutcome {
    let item_kind = item.kind();
    match sender.try_send(item) {
        Ok(()) => PersistEnqueueOutcome::Queued,
        Err(mpsc::error::TrySendError::Full(_)) => {
            let mut metrics = metrics.write().await;
            metrics.dropped_messages = metrics.dropped_messages.saturating_add(1);
            let first_overflow = metrics.primary_persistence_queue_overflows == 0;
            metrics.primary_persistence_queue_overflows = metrics
                .primary_persistence_queue_overflows
                .saturating_add(1);
            if first_overflow {
                metrics.last_error = Some(format!(
                    "BTC persistence queue saturated; dropped {item_kind}"
                ));
                let mut realtime = state.write().await;
                if !realtime.primary_persistence_degraded {
                    realtime.primary_persistence_degraded = true;
                    realtime.last_updated_at = Some(Utc::now());
                }
                tracing::error!(item_kind, "primary persistence queue saturated");
            }
            PersistEnqueueOutcome::Saturated
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            let mut metrics = metrics.write().await;
            metrics.dropped_messages = metrics.dropped_messages.saturating_add(1);
            metrics.last_error = Some(format!(
                "BTC persistence queue closed; rejected {item_kind}"
            ));
            let mut realtime = state.write().await;
            realtime.primary_persistence_degraded = true;
            realtime.last_updated_at = Some(Utc::now());
            PersistEnqueueOutcome::Closed
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
    let mut gamma_resolution_retries = HashMap::new();
    let mut boundary_hydration_retry_at = None;
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
                    &mut gamma_resolution_retries,
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
                    tracing::warn!(
                        market_ids = ?expired,
                        "BTC official-resolution watches exceeded retention and remain eligible for bounded reconciliation"
                    );
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
                publish_market_subscriptions_if_changed(&market_sender, &pending_markets);
                {
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.markets_discovered = fresh_markets.len() as u64;
                    runtime_metrics.resolution_watches_active = pending_markets.len() as u64;
                }
                if boundary_hydration_retry_at.is_some_and(|retry_at| Instant::now() < retry_at) {
                    continue;
                }
                let max_delay = chrono_duration(config.boundary_tick_max_delay);
                let durable = tokio::try_join!(
                    repository.load_market_open_references(&recovery_markets),
                    repository.load_market_close_references(&recovery_markets, max_delay),
                    repository.load_market_labels(&recovery_markets),
                );
                let (durable_opens, durable_closes, durable_labels) = match durable {
                    Ok(value) => value,
                    Err(error) => {
                        let failure = {
                            let mut runtime_metrics = metrics.write().await;
                            classify_boundary_hydration_read_failure(error, &mut runtime_metrics)
                        };
                        match failure {
                            BoundaryHydrationReadFailure::Retry {
                                error,
                                entered_degraded_state,
                                consecutive_failures,
                            } => {
                                let retry_delay = boundary_hydration_retry_delay(
                                    config.discovery_interval,
                                    consecutive_failures,
                                );
                                boundary_hydration_retry_at = Some(Instant::now() + retry_delay);
                                if entered_degraded_state {
                                    tracing::warn!(
                                        error = ?error,
                                        retry_delay_ms = duration_milliseconds(retry_delay),
                                        "BTC boundary hydration is temporarily unavailable; shared market data remains active"
                                    );
                                } else {
                                    tracing::debug!(
                                        error = ?error,
                                        retry_delay_ms = duration_milliseconds(retry_delay),
                                        "BTC boundary hydration retry remains unavailable"
                                    );
                                }
                                continue;
                            }
                            BoundaryHydrationReadFailure::Fatal(error) => {
                                record_critical_persistence_error(&metrics, error).await;
                                return;
                            }
                        }
                    }
                };
                let hydration = {
                    // Boundary observation remains fail-closed until every durable read has
                    // succeeded. Publishing CLOB subscriptions above is independent of this
                    // immutable boundary-state transition.
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
                boundary_hydration_retry_at = None;
                let recovered = {
                    let mut runtime_metrics = metrics.write().await;
                    mark_boundary_hydration_recovered(&mut runtime_metrics)
                };
                if recovered {
                    tracing::info!("BTC boundary hydration recovered without restarting shared market data");
                }
            }
        }
    }
}

#[derive(Debug)]
enum ReconciledOfficialResolution {
    Clob(ClobRestOfficialResolution),
    Gamma(GammaRestOfficialResolution),
}

#[derive(Debug, Clone, Copy)]
struct GammaResolutionRetry {
    next_attempt_at: DateTime<Utc>,
    backoff: StdDuration,
}

fn gamma_resolution_reconciliation_due(
    market: &BtcIntervalMarket,
    audit_grace: StdDuration,
    retry: Option<GammaResolutionRetry>,
    now: DateTime<Utc>,
) -> bool {
    now >= market.window_end + chrono_duration(audit_grace)
        && retry.map_or(true, |retry| retry.next_attempt_at <= now)
}

fn defer_gamma_resolution_retry(
    retries: &mut HashMap<String, GammaResolutionRetry>,
    market_id: &str,
    attempted_at: DateTime<Utc>,
) {
    let backoff = retries
        .get(market_id)
        .map(|retry| {
            retry
                .backoff
                .saturating_mul(2)
                .min(GAMMA_RESOLUTION_RETRY_MAX_BACKOFF)
        })
        .unwrap_or(GAMMA_RESOLUTION_RETRY_INITIAL_BACKOFF);
    retries.insert(
        market_id.to_string(),
        GammaResolutionRetry {
            next_attempt_at: attempted_at + chrono_duration(backoff),
            backoff,
        },
    );
}

async fn reconcile_official_resolution_watches(
    client: &reqwest::Client,
    config: &BtcRuntimeConfig,
    repository: &BtcRepository,
    watches: &[BtcOfficialResolutionWatch],
    state: Arc<RwLock<RealtimeState>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    gamma_retries: &mut HashMap<String, GammaResolutionRetry>,
    now: DateTime<Utc>,
) -> Result<()> {
    let unsettled_market_ids = watches
        .iter()
        .map(|watch| watch.market.market_id.as_str())
        .collect::<HashSet<_>>();
    gamma_retries.retain(|market_id, _| unsettled_market_ids.contains(market_id.as_str()));

    let mut candidates = Vec::with_capacity(watches.len());
    let mut expired_candidates = 0_usize;
    for watch in watches
        .iter()
        .filter(|watch| watch.market.window_end <= now)
    {
        let gamma_due = gamma_resolution_reconciliation_due(
            &watch.market,
            config.official_resolution_audit_grace,
            gamma_retries.get(&watch.market.market_id).copied(),
            now,
        );
        if watch.status == "expired" {
            if !gamma_due || expired_candidates >= MAX_EXPIRED_RESOLUTION_RECONCILIATIONS_PER_TICK {
                continue;
            }
            expired_candidates = expired_candidates.saturating_add(1);
        }
        candidates.push((watch.clone(), gamma_due));
    }
    let results = stream::iter(candidates)
        .map(|(watch, gamma_due)| {
            let client = client.clone();
            let clob_base_url = config.clob_rest_base_url.clone();
            let gamma_base_url = config.gamma_base_url.clone();
            async move {
                let mut gamma_attempted = false;
                let result = match fetch_clob_rest_official_resolution(
                    &client,
                    &clob_base_url,
                    &watch.market,
                )
                .await
                {
                    Ok(Some(resolution)) => {
                        Ok(Some(ReconciledOfficialResolution::Clob(resolution)))
                    }
                    Ok(None) if gamma_due => {
                        gamma_attempted = true;
                        fetch_gamma_rest_official_resolution(
                            &client,
                            &gamma_base_url,
                            &watch.market,
                        )
                        .await
                        .map(|resolution| resolution.map(ReconciledOfficialResolution::Gamma))
                    }
                    Ok(None) => Ok(None),
                    Err(error) => Err(error),
                };
                (watch, gamma_attempted, result)
            }
        })
        .buffer_unordered(4)
        .collect::<Vec<_>>()
        .await;

    for (watch, gamma_attempted, result) in results {
        match result {
            Ok(Some(ReconciledOfficialResolution::Clob(resolution))) => {
                gamma_retries.remove(&resolution.market_id);
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
            Ok(Some(ReconciledOfficialResolution::Gamma(resolution))) => {
                gamma_retries.remove(&resolution.market_id);
                let persisted = persist_official_resolution_fact(
                    repository,
                    &resolution.market_id,
                    &resolution.winning_token_id,
                    outcome_display_name(resolution.winning_outcome),
                    resolution.source_timestamp,
                    "gamma_rest_reconciliation",
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
                if gamma_attempted || watch.status == "expired" {
                    defer_gamma_resolution_retry(gamma_retries, &watch.market.market_id, now);
                }
                repository
                    .mark_official_resolution_watch_checked(
                        &watch.market.market_id,
                        Utc::now(),
                        None,
                    )
                    .await?;
            }
            Err(error) => {
                if gamma_attempted || watch.status == "expired" {
                    defer_gamma_resolution_retry(gamma_retries, &watch.market.market_id, now);
                }
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

async fn fetch_gamma_rest_official_resolution(
    client: &reqwest::Client,
    base_url: &str,
    market: &BtcIntervalMarket,
) -> Result<Option<GammaRestOfficialResolution>> {
    let url = format!(
        "{}/events/slug/{}",
        base_url.trim_end_matches('/'),
        market.event_slug
    );
    let value = client
        .get(&url)
        .send()
        .await
        .with_context(|| {
            format!(
                "failed to reconcile Gamma event {} for market {}",
                market.event_slug, market.market_id
            )
        })?
        .error_for_status()
        .with_context(|| format!("Gamma REST rejected event {}", market.event_slug))?
        .json::<serde_json::Value>()
        .await
        .with_context(|| format!("failed to decode Gamma event {}", market.event_slug))?;
    parse_gamma_rest_official_resolution(&value, market, Utc::now())
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
    if !delta.is_empty() {
        epoch.telemetry.refresh_subscription_target(desired_markets);
    }
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
            Err(ClobSendFailure::Transport { error, detail }) => {
                epoch.telemetry.last_transport_error = Some(detail);
                return Err(ClobEpochUpdateError::Recoverable(
                    bounded_clob_error_reason(&format!("dynamic_subscribe_failed:{error}")),
                ));
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
            Err(ClobSendFailure::Transport { error, detail }) => {
                epoch.telemetry.last_transport_error = Some(detail);
                return Err(ClobEpochUpdateError::Recoverable(
                    bounded_clob_error_reason(&format!("dynamic_unsubscribe_failed:{error}")),
                ));
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrivateClobEventDisposition {
    Accept,
    AwaitSnapshot,
    IgnoreForeign,
    IgnoreSuperseded,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrivateClobSubscriptionMatch {
    token: bool,
    market: bool,
    exact_identity: bool,
}

fn private_clob_subscription_match(
    markets: &[BtcIntervalMarket],
    event: &MarketFeedEvent,
) -> PrivateClobSubscriptionMatch {
    let token = event.token_id.as_deref().is_some_and(|token_id| {
        markets
            .iter()
            .any(|market| market.up_token_id == token_id || market.down_token_id == token_id)
    });
    let market = markets.iter().any(|market| {
        market.market_id == event.market_id || market.condition_id == event.market_id
    });
    let exact_identity = event.token_id.as_deref().is_some_and(|token_id| {
        markets.iter().any(|market| {
            (market.market_id == event.market_id || market.condition_id == event.market_id)
                && (market.up_token_id == token_id || market.down_token_id == token_id)
        })
    });
    PrivateClobSubscriptionMatch {
        token,
        market,
        exact_identity,
    }
}

fn private_clob_event_disposition(
    markets: &[BtcIntervalMarket],
    event: &MarketFeedEvent,
) -> PrivateClobEventDisposition {
    if event.applied && event.integrity_status == FeedIntegrityStatus::Ok {
        return PrivateClobEventDisposition::Accept;
    }

    let subscription_match = private_clob_subscription_match(markets, event);
    if !event.applied
        && event.integrity_status == FeedIntegrityStatus::OutOfOrder
        && subscription_match.exact_identity
        && event.event_type == MarketFeedEventType::PriceChange
    {
        return PrivateClobEventDisposition::IgnoreSuperseded;
    }
    if !event.applied
        && event.integrity_status == FeedIntegrityStatus::PreSnapshot
        && subscription_match.exact_identity
        && matches!(
            event.event_type,
            MarketFeedEventType::PriceChange
                | MarketFeedEventType::BestBidAsk
                | MarketFeedEventType::TickSizeChange
                | MarketFeedEventType::LastTradePrice
        )
    {
        return PrivateClobEventDisposition::AwaitSnapshot;
    }

    if !event.applied
        && event.integrity_status == FeedIntegrityStatus::UnknownToken
        && !subscription_match.token
        && !subscription_match.market
    {
        return PrivateClobEventDisposition::IgnoreForeign;
    }

    PrivateClobEventDisposition::Reject
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
    let received_instant = Instant::now();
    let received_at = Utc::now();
    let pong_round_trip = match &message {
        Message::Text(text) => epoch
            .watchdog
            .acknowledge_text_pong(text.as_str(), received_instant),
        _ => None,
    };
    epoch.watchdog.on_frame(received_instant);
    epoch.telemetry.record_frame(received_at, received_instant);
    let parsed = match message {
        Message::Text(text) => {
            let pong_like = is_clob_text_pong(text.as_str());
            if pong_like {
                if let Some(round_trip) = pong_round_trip {
                    let sample = epoch.telemetry.record_heartbeat_acknowledgement(
                        received_at,
                        received_instant,
                        round_trip,
                    );
                    record_active_clob_heartbeat_acknowledgement_metrics(metrics, sample).await;
                }
                return Ok(ClobFrameAction::Continue);
            }
            if text.trim().is_empty() {
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
            epoch.telemetry.record_remote_close(frame.as_ref());
            epoch.session.disconnect_reason = Some("remote_close".to_string());
            return Ok(ClobFrameAction::Disconnect);
        }
        _ => return Ok(ClobFrameAction::Continue),
    };
    epoch.telemetry.last_data_or_heartbeat_at = Some(received_at);
    epoch.session.messages_received = epoch.session.messages_received.saturating_add(1);
    {
        let mut runtime_metrics = metrics.write().await;
        runtime_metrics.clob_messages_received =
            runtime_metrics.clob_messages_received.saturating_add(1);
        runtime_metrics.clob_active_last_data_or_heartbeat_at = Some(received_at);
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
    let source_to_receive_lag_milliseconds = events
        .iter()
        .filter(|event| event.applied)
        .map(|event| {
            (event.received_at - event.source_timestamp)
                .num_milliseconds()
                .max(0)
        })
        .max();
    let frame_token_ids = events
        .iter()
        .filter_map(|event| event.token_id.as_deref())
        .fold(Vec::<String>::new(), |mut token_ids, token_id| {
            if !token_ids.iter().any(|existing| existing == token_id) {
                token_ids.push(token_id.to_string());
            }
            token_ids
        });
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
        if let Some(lag_milliseconds) = source_to_receive_lag_milliseconds {
            runtime_metrics.clob_active_last_source_to_receive_lag_milliseconds =
                Some(lag_milliseconds);
        }
    }
    let frame_changed = !events.is_empty() || !resolutions.is_empty();
    for event in events {
        if !should_persist_feed_event(&event) {
            continue;
        }
        match enqueue(writer, PersistItem::FeedEvent(event), state, metrics).await {
            PersistEnqueueOutcome::Queued => {
                epoch.session.messages_persisted =
                    epoch.session.messages_persisted.saturating_add(1);
            }
            PersistEnqueueOutcome::Saturated => {
                epoch.session.dropped_messages = epoch.session.dropped_messages.saturating_add(1);
            }
            PersistEnqueueOutcome::Closed => {
                epoch.session.dropped_messages = epoch.session.dropped_messages.saturating_add(1);
                bail!("CLOB feed event persistence queue closed");
            }
        }
    }
    if frame_changed {
        let mut published_books = shared_books.write().await;
        let mut shared = state.write().await;
        published_books
            .publish_frame_books_from(&epoch.registry, frame_token_ids.iter().map(String::as_str));
        shared.update_books(&epoch.registry);
        shared.last_updated_at = Some(received_at);
        for resolution in resolutions {
            shared.apply_market_resolution(&resolution.market_id, &resolution.winning_token_id);
        }
    }
    Ok(ClobFrameAction::Continue)
}

fn apply_private_clob_frame(epoch: &mut ClobEpoch, message: Message) -> ClobFrameAction {
    let received_instant = Instant::now();
    let received_at = Utc::now();
    let pong_round_trip = match &message {
        Message::Text(text) => epoch
            .watchdog
            .acknowledge_text_pong(text.as_str(), received_instant),
        _ => None,
    };
    epoch.watchdog.on_frame(received_instant);
    epoch.telemetry.record_frame(received_at, received_instant);
    let parsed = match message {
        Message::Text(text) => {
            let pong_like = is_clob_text_pong(text.as_str());
            if pong_like {
                if let Some(round_trip) = pong_round_trip {
                    epoch.telemetry.record_heartbeat_acknowledgement(
                        received_at,
                        received_instant,
                        round_trip,
                    );
                }
                return ClobFrameAction::Continue;
            }
            if text.trim().is_empty() {
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
            epoch.telemetry.record_remote_close(frame.as_ref());
            epoch.session.disconnect_reason = Some("remote_close".to_string());
            return ClobFrameAction::Disconnect;
        }
        _ => return ClobFrameAction::Continue,
    };
    epoch.telemetry.last_data_or_heartbeat_at = Some(received_at);
    epoch.session.messages_received = epoch.session.messages_received.saturating_add(1);
    match parsed {
        Ok(messages) => {
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
                                tracing::warn!(
                                    feed = "polymarket_clob_market",
                                    connection_id = %epoch.connection_id,
                                    connection_epoch = epoch.connection_epoch,
                                    market_id = %entry.key(),
                                    "private CLOB successor resolution conflict quarantined without retiring transport"
                                );
                            }
                        }
                    }
                    continue;
                }
                for event in epoch.registry.apply(message, received_at) {
                    match private_clob_event_disposition(&epoch.markets, &event) {
                        PrivateClobEventDisposition::Accept
                        | PrivateClobEventDisposition::AwaitSnapshot => {}
                        PrivateClobEventDisposition::IgnoreForeign => {
                            epoch.subscription_stats.ignored_foreign_events = epoch
                                .subscription_stats
                                .ignored_foreign_events
                                .saturating_add(1);
                        }
                        PrivateClobEventDisposition::IgnoreSuperseded => {
                            epoch.subscription_stats.ignored_superseded_events = epoch
                                .subscription_stats
                                .ignored_superseded_events
                                .saturating_add(1);
                        }
                        PrivateClobEventDisposition::Reject => {
                            let subscription_match =
                                private_clob_subscription_match(&epoch.markets, &event);
                            tracing::warn!(
                                feed = "polymarket_clob_market",
                                connection_id = %epoch.connection_id,
                                connection_epoch = epoch.connection_epoch,
                                event_type = ?event.event_type,
                                integrity_status = ?event.integrity_status,
                                token_matches_subscription = subscription_match.token,
                                market_matches_subscription = subscription_match.market,
                                "private CLOB successor integrity gap"
                            );
                            epoch.session.integrity_gaps =
                                epoch.session.integrity_gaps.saturating_add(1);
                        }
                    }
                }
            }
        }
        Err(error) => {
            epoch.registry.quarantine(FeedIntegrityStatus::DecodeError);
            epoch.session.decode_errors = epoch.session.decode_errors.saturating_add(1);
            tracing::warn!(
                feed = "polymarket_clob_market",
                connection_id = %epoch.connection_id,
                connection_epoch = epoch.connection_epoch,
                reason = %error,
                "private CLOB successor frame failed to decode; transport retained"
            );
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

#[derive(Debug)]
enum ClobCloseOutcome {
    Skipped,
    Completed,
    TimedOut,
    Failed(ClobTransportErrorDetail),
}

async fn attempt_clob_close<S>(
    socket: &mut WebSocketStream<S>,
    action: ClobCloseAction,
) -> ClobCloseOutcome
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let result = match action {
        ClobCloseAction::Skip => return ClobCloseOutcome::Skipped,
        ClobCloseAction::Initiate => timeout(CLOB_GRACEFUL_CLOSE_TIMEOUT, socket.close(None)).await,
        ClobCloseAction::AcknowledgeRemote => {
            timeout(CLOB_GRACEFUL_CLOSE_TIMEOUT, socket.flush()).await
        }
    };
    match result {
        Ok(Ok(())) => ClobCloseOutcome::Completed,
        Ok(Err(error)) => ClobCloseOutcome::Failed(clob_transport_error_detail(&error)),
        Err(_) => ClobCloseOutcome::TimedOut,
    }
}

fn clob_failure_close_action(telemetry: &ClobSocketTelemetry) -> ClobCloseAction {
    if telemetry.remote_close_observed {
        ClobCloseAction::AcknowledgeRemote
    } else {
        ClobCloseAction::Skip
    }
}

fn clob_unavailable_recovery_close_action() -> ClobCloseAction {
    ClobCloseAction::Skip
}

fn clob_handoff_close_action(
    cause: ClobDisconnectCause,
    telemetry: &ClobSocketTelemetry,
) -> ClobCloseAction {
    if telemetry.remote_close_observed {
        ClobCloseAction::AcknowledgeRemote
    } else if cause == ClobDisconnectCause::ReadinessRefresh {
        ClobCloseAction::Initiate
    } else {
        ClobCloseAction::Skip
    }
}

fn clob_stop_close_action(telemetry: &ClobSocketTelemetry) -> ClobCloseAction {
    if telemetry.remote_close_observed {
        ClobCloseAction::AcknowledgeRemote
    } else {
        ClobCloseAction::Initiate
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
    complete_clob_epoch_with_close(
        repository,
        metrics,
        epoch,
        reason,
        cause,
        retry_action,
        consecutive_failures,
        ClobCloseAction::Skip,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn complete_clob_epoch_with_close(
    repository: &BtcRepository,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    epoch: &mut ClobEpoch,
    reason: String,
    cause: ClobDisconnectCause,
    retry_action: ClobRetryAction,
    consecutive_failures: u32,
    close_action: ClobCloseAction,
) -> bool {
    let disconnected_at = Utc::now();
    let disconnected_instant = Instant::now();
    match attempt_clob_close(&mut epoch.socket, close_action).await {
        ClobCloseOutcome::Skipped => {}
        ClobCloseOutcome::Completed => {
            tracing::debug!(
                feed = "polymarket_clob_market",
                connection_id = %epoch.connection_id,
                connection_epoch = epoch.connection_epoch,
                connection_role = epoch.telemetry.role.as_str(),
                close_action = close_action.as_str(),
                "CLOB websocket close action completed"
            );
        }
        ClobCloseOutcome::TimedOut => {
            tracing::warn!(
                feed = "polymarket_clob_market",
                connection_id = %epoch.connection_id,
                connection_epoch = epoch.connection_epoch,
                connection_role = epoch.telemetry.role.as_str(),
                close_action = close_action.as_str(),
                timeout_ms = duration_milliseconds(CLOB_GRACEFUL_CLOSE_TIMEOUT),
                "CLOB websocket close action timed out"
            );
        }
        ClobCloseOutcome::Failed(detail) => {
            tracing::warn!(
                feed = "polymarket_clob_market",
                connection_id = %epoch.connection_id,
                connection_epoch = epoch.connection_epoch,
                connection_role = epoch.telemetry.role.as_str(),
                close_action = close_action.as_str(),
                transport_error_class = detail.class,
                transport_io_kind = ?detail.io_kind.as_deref(),
                transport_os_error_code = ?detail.os_error_code,
                "CLOB websocket close action failed"
            );
        }
    }
    let reason = bounded_clob_error_reason(&reason);
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
        disconnected_instant.duration_since(epoch.connected_instant),
        disconnected_instant,
        epoch.healthy_epoch,
        cause,
        retry_action,
        consecutive_failures,
        &reason,
        epoch.subscription_stats.active_assets,
        &epoch.telemetry,
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
        | ClobDisconnectCause::ReadinessRefresh
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

#[derive(Debug)]
enum ClobPromotionOutcome {
    NotReady,
    Promoted { retired: Option<Box<ClobEpoch>> },
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
) -> Result<ClobPromotionOutcome> {
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
        return Ok(ClobPromotionOutcome::NotReady);
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
        return Ok(ClobPromotionOutcome::NotReady);
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
        return Ok(ClobPromotionOutcome::NotReady);
    }
    if let Err(error) = acknowledge_clob_subscriptions(
        repository,
        &desired_before_ack,
        promoted.connection_id,
        checked_before_ack,
    )
    .await
    {
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
                let _ = complete_clob_epoch(
                    repository,
                    metrics,
                    &mut promoted,
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
    let checked_before_publication = Utc::now();
    if !clob_successor_publication_still_valid(
        &promoted,
        &desired_markets.borrow(),
        &publication.current_market,
        checked_before_publication,
        max_book_age,
    ) {
        if !complete_clob_epoch_with_close(
            repository,
            metrics,
            &mut promoted,
            "promotion_stale_before_publication".to_string(),
            ClobDisconnectCause::ReadinessRefresh,
            ClobRetryAction::ImmediateRecovery,
            0,
            clob_unavailable_recovery_close_action(),
        )
        .await
        {
            bail!("failed to finalize stale CLOB promotion session");
        }
        return Ok(ClobPromotionOutcome::NotReady);
    }

    let mut published_books = shared_books.write().await;
    let mut shared = state.write().await;
    let published_at = Utc::now();
    let publication_valid = {
        let desired_at_publication = desired_markets.borrow();
        clob_successor_publication_still_valid(
            &promoted,
            &desired_at_publication,
            &publication.current_market,
            published_at,
            max_book_age,
        )
    };
    if publication_valid {
        *published_books = publication.registry.clone();
        shared.update_books(&publication.registry);
        shared.last_updated_at = Some(published_at);
    }
    drop(shared);
    drop(published_books);
    if !publication_valid {
        if !complete_clob_epoch_with_close(
            repository,
            metrics,
            &mut promoted,
            "promotion_stale_at_publication".to_string(),
            ClobDisconnectCause::ReadinessRefresh,
            ClobRetryAction::ImmediateRecovery,
            0,
            clob_unavailable_recovery_close_action(),
        )
        .await
        {
            bail!("failed to finalize stale CLOB promotion session");
        }
        return Ok(ClobPromotionOutcome::NotReady);
    }

    promoted.healthy_epoch = true;
    promoted.books_usable = true;
    promoted.subscription_stats.active_assets = promoted.registry.len();
    let promoted_connection_id = promoted.connection_id;
    let promoted_connection_epoch = promoted.connection_epoch;
    let promoted_connected_at = promoted.session.connected_at;
    let promoted_assets = promoted.registry.len();
    promoted.telemetry.mark_active();
    let retired = active.replace(promoted);
    {
        let mut runtime_metrics = metrics.write().await;
        runtime_metrics.persistence_items_written =
            runtime_metrics.persistence_items_written.saturating_add(2);
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
    let promoted_telemetry = &active
        .as_ref()
        .expect("promoted CLOB epoch is installed as active")
        .telemetry;
    {
        let mut runtime_metrics = metrics.write().await;
        runtime_metrics.clob_connected_connection_epoch = Some(promoted_connection_epoch);
        runtime_metrics.clob_connected_connection_id = Some(promoted_connection_id);
        runtime_metrics.clob_last_connected_at = promoted_connected_at;
        runtime_metrics.clob_active_subscribed_assets =
            u64::try_from(promoted_assets).unwrap_or(u64::MAX);
        publish_active_clob_socket_metrics(&mut runtime_metrics, promoted_telemetry);
    }
    tracing::info!(
        feed = "polymarket_clob_market",
        connection_id = %promoted_connection_id,
        connection_epoch = promoted_connection_epoch,
        active_assets = promoted_assets,
        peer_address = ?promoted_telemetry.provenance.peer_address,
        edge_request_id = ?promoted_telemetry.provenance.edge_request_id,
        subscription_target_fingerprint_sha256 =
            %promoted_telemetry.subscription_target_fingerprint_sha256,
        "CLOB successor promoted with an atomic ready-book handoff"
    );
    Ok(ClobPromotionOutcome::Promoted {
        retired: retired.map(Box::new),
    })
}

async fn run_clob_supervisor(
    config: BtcRuntimeConfig,
    heartbeat_interval: StdDuration,
    pong_timeout: StdDuration,
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
    let mut successor_rapid_retry_allowed = true;
    let mut desired_connection_targets = markets.borrow().clone();
    let mut connecting_markets: Option<Vec<BtcIntervalMarket>> = None;
    let mut recovery_window = ClobRecoveryWindow::open(Utc::now(), Instant::now());
    metrics.write().await.clob_recovery_unavailable_since = recovery_window.since;
    let started_at = Instant::now();
    let mut heartbeat = interval_at(started_at + heartbeat_interval, heartbeat_interval);
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
                Ok(ClobPromotionOutcome::Promoted { retired }) => {
                    debug_assert!(retired.is_none());
                    successor_failures = 0;
                    successor_retry_at = Instant::now();
                    successor_rapid_retry_allowed = true;
                    continue;
                }
                Ok(ClobPromotionOutcome::NotReady) => {}
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
        let readiness_checked_at = Utc::now();
        let active_structurally_ready =
            clob_active_epoch_structurally_ready(active.as_ref(), readiness_checked_at);
        successor_retry_at = expedite_clob_candidate_retry(
            &config,
            active_structurally_ready,
            successor_rapid_retry_allowed,
            Instant::now(),
            successor_retry_at,
            &mut successor_failures,
        );
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
                if let Some(epoch) = active.as_mut() {
                    epoch.watchdog.bootstrap_deadline = None;
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.clob_bootstrap_failures = runtime_metrics
                        .clob_bootstrap_failures
                        .saturating_add(1);
                    runtime_metrics.last_error = Some(
                        "active CLOB books did not bootstrap before the readiness deadline; transport retained"
                            .to_string(),
                    );
                    tracing::warn!(
                        feed = "polymarket_clob_market",
                        connection_id = %epoch.connection_id,
                        connection_epoch = epoch.connection_epoch,
                        "active CLOB bootstrap deadline elapsed; transport retained"
                    );
                }
            }
            _ = &mut active_pong_sleep, if active_pong_deadline.is_some() => {
                active_failure = Some((
                    "heartbeat_ack_timeout".to_string(),
                    ClobDisconnectCause::TransportFailure,
                    None,
                ));
            }
            _ = &mut active_read_sleep, if active_read_deadline.is_some() => {
                active_failure = Some((
                    "read_idle_timeout".to_string(),
                    ClobDisconnectCause::TransportFailure,
                    None,
                ));
            }
            _ = &mut successor_bootstrap_sleep, if successor_bootstrap_deadline.is_some() => {
                if let Some(epoch) = successor.as_mut() {
                    epoch.watchdog.bootstrap_deadline = None;
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.clob_bootstrap_failures = runtime_metrics
                        .clob_bootstrap_failures
                        .saturating_add(1);
                    runtime_metrics.last_error = Some(
                        "successor CLOB books did not bootstrap before the readiness deadline; transport retained"
                            .to_string(),
                    );
                    tracing::warn!(
                        feed = "polymarket_clob_market",
                        connection_id = %epoch.connection_id,
                        connection_epoch = epoch.connection_epoch,
                        "successor CLOB bootstrap deadline elapsed; transport retained"
                    );
                }
            }
            _ = &mut successor_pong_sleep, if successor_pong_deadline.is_some() => {
                successor_failure = Some((
                    "heartbeat_ack_timeout".to_string(),
                    ClobDisconnectCause::TransportFailure,
                ));
            }
            _ = &mut successor_read_sleep, if successor_read_deadline.is_some() => {
                successor_failure = Some((
                    "read_idle_timeout".to_string(),
                    ClobDisconnectCause::TransportFailure,
                ));
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
                            successor_rapid_retry_allowed = true;
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
                                        runtime_metrics.clob_active_subscription_target_fingerprint_sha256 =
                                            Some(epoch.telemetry.subscription_target_fingerprint_sha256.clone());
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
                        active
                            .as_mut()
                            .expect("active epoch remains installed")
                            .telemetry
                            .last_transport_error = Some(ClobTransportErrorDetail::websocket_eof());
                        let cause = clob_disconnect_cause(
                            false,
                            false,
                            false,
                            false,
                        );
                        active_failure = Some(("websocket_eof".to_string(), cause, None));
                    }
                    Some(Err(error)) => {
                        active
                            .as_mut()
                            .expect("active epoch remains installed")
                            .telemetry
                            .last_transport_error = Some(clob_transport_error_detail(&error));
                        active_failure = Some((
                            bounded_clob_error_reason(&format!("transport_read_failed:{error}")),
                            ClobDisconnectCause::TransportFailure,
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
                                    ClobDisconnectCause::TransportFailure,
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
                        successor
                            .as_mut()
                            .expect("successor epoch remains installed")
                            .telemetry
                            .last_transport_error = Some(ClobTransportErrorDetail::websocket_eof());
                        successor_failure = Some((
                            "websocket_eof".to_string(),
                            ClobDisconnectCause::TransportFailure,
                        ));
                    }
                    Some(Err(error)) => {
                        successor
                            .as_mut()
                            .expect("successor epoch remains installed")
                            .telemetry
                            .last_transport_error = Some(clob_transport_error_detail(&error));
                        successor_failure = Some((
                            bounded_clob_error_reason(&format!("transport_read_failed:{error}")),
                            ClobDisconnectCause::TransportFailure,
                        ));
                    }
                    Some(Ok(message)) => {
                        let epoch = successor
                            .as_mut()
                            .expect("successor epoch remains installed");
                        let action = apply_private_clob_frame(epoch, message);
                        if action == ClobFrameAction::Disconnect {
                            successor_failure = Some((
                                epoch
                                    .session
                                    .disconnect_reason
                                    .clone()
                                    .unwrap_or_else(|| "remote_close".to_string()),
                                private_clob_disconnect_cause(
                                    epoch.session.disconnect_reason.as_deref(),
                                ),
                            ));
                        } else {
                            let checked_at = Utc::now();
                            epoch.refresh_private_health(
                                checked_at,
                                Instant::now(),
                                max_book_age,
                            );
                            if epoch.books_usable {
                                successor_failures = 0;
                                successor_retry_at = Instant::now();
                                successor_rapid_retry_allowed = true;
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
                        successor_rapid_retry_allowed = false;
                        let checked_at = Utc::now();
                        let active_structurally_ready_now =
                            clob_active_epoch_structurally_ready(active.as_ref(), checked_at);
                        let retry_action = clob_candidate_retry_action(
                            &config,
                            false,
                            active_structurally_ready_now,
                            successor_rapid_retry_allowed,
                            &mut successor_failures,
                        );
                        let ClobRetryAction::Backoff(delay) = retry_action else {
                            unreachable!("a failed connect task must use a bounded retry delay")
                        };
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
                                retry_action,
                            );
                        }
                    }
                    Ok(ClobConnectOutcome::Failed(failure)) => {
                        let ClobConnectFailure {
                            mut session,
                            reason,
                            kind,
                            telemetry,
                        } = *failure;
                        if kind == ClobConnectFailureKind::Shutdown {
                            shutdown_requested = true;
                        } else {
                            successor_rapid_retry_allowed =
                                kind != ClobConnectFailureKind::Identity;
                            let checked_at = Utc::now();
                            let active_structurally_ready_now =
                                clob_active_epoch_structurally_ready(active.as_ref(), checked_at);
                            let retry_action = clob_candidate_retry_action(
                                &config,
                                false,
                                active_structurally_ready_now,
                                successor_rapid_retry_allowed,
                                &mut successor_failures,
                            );
                            let ClobRetryAction::Backoff(delay) = retry_action else {
                                unreachable!("a failed connect attempt must use a bounded retry delay")
                            };
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
                                retry_action,
                                successor_failures,
                                &ClobSubscriptionStats::default(),
                            );
                            if start_feed_session_or_fail(&repository, &session, &metrics).await {
                                let _ =
                                    finish_feed_session_or_fail(&repository, &session, &metrics)
                                        .await;
                            }
                            {
                                let mut runtime_metrics = metrics.write().await;
                                clob_candidate_attempt_metrics(
                                    &mut runtime_metrics,
                                    cause,
                                    retry_action,
                                );
                            }
                            let transport_error = telemetry.last_transport_error.as_ref();
                            tracing::warn!(
                                feed = "polymarket_clob_market",
                                %reason,
                                successor_failures,
                                retry_delay_ms = duration_milliseconds(delay),
                                active_present = active.is_some(),
                                connection_role = telemetry.role.as_str(),
                                peer_address = ?telemetry.provenance.peer_address.as_deref(),
                                edge_request_id = ?telemetry.provenance.edge_request_id.as_deref(),
                                edge_server = ?telemetry.provenance.edge_server.as_deref(),
                                handshake_date = ?telemetry.provenance.handshake_date.as_deref(),
                                subscription_target_fingerprint_sha256 =
                                    %telemetry.subscription_target_fingerprint_sha256,
                                transport_error_class =
                                    ?transport_error.map(|error| error.class),
                                transport_io_kind =
                                    ?transport_error.and_then(|error| error.io_kind.as_deref()),
                                transport_os_error_code =
                                    ?transport_error.and_then(|error| error.os_error_code),
                                "CLOB successor connection attempt failed"
                            );
                        }
                    }
                    Ok(ClobConnectOutcome::Connected { mut epoch, connect_latency }) => {
                        epoch.subscription_stats.active_assets = epoch.registry.len();
                        if !start_feed_session_or_fail(&repository, &epoch.session, &metrics).await {
                            successor_rapid_retry_allowed = false;
                            let checked_at = Utc::now();
                            let active_structurally_ready_now =
                                clob_active_epoch_structurally_ready(active.as_ref(), checked_at);
                            let retry_action = clob_candidate_retry_action(
                                &config,
                                false,
                                active_structurally_ready_now,
                                successor_rapid_retry_allowed,
                                &mut successor_failures,
                            );
                            let ClobRetryAction::Backoff(delay) = retry_action else {
                                unreachable!("a failed session start must use a bounded retry delay")
                            };
                            successor_retry_at = Instant::now() + delay;
                            {
                                let mut runtime_metrics = metrics.write().await;
                                clob_retry_metrics(
                                    &mut runtime_metrics,
                                    retry_action,
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
                            successor_rapid_retry_allowed = true;
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
                                peer_address = ?epoch.telemetry.provenance.peer_address,
                                edge_request_id = ?epoch.telemetry.provenance.edge_request_id,
                                edge_server = ?epoch.telemetry.provenance.edge_server,
                                handshake_date = ?epoch.telemetry.provenance.handshake_date,
                                subscription_target_fingerprint_sha256 =
                                    %epoch.telemetry.subscription_target_fingerprint_sha256,
                                "private CLOB successor connected"
                            );
                            successor_rapid_retry_allowed = true;
                            successor = Some(*epoch);
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
                        successor_rapid_retry_allowed = true;
                    }
                }
                if let Some(epoch) = active.as_mut() {
                    epoch.watchdog.refresh_bootstrap(
                        Instant::now(),
                        &epoch.registry,
                        &epoch.markets,
                        checked_at,
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
                    if active_failure.is_none() {
                        for market in &epoch.markets {
                            if !market.is_trade_window(checked_at) {
                                continue;
                            }
                            for token_id in [&market.up_token_id, &market.down_token_id] {
                                if let Some(checkpoint) = epoch.registry.checkpoint(token_id) {
                                    match enqueue(
                                        &writer,
                                        PersistItem::Checkpoint(checkpoint),
                                        &state,
                                        &metrics,
                                    )
                                    .await
                                    {
                                        PersistEnqueueOutcome::Queued => {
                                            let mut runtime_metrics = metrics.write().await;
                                            runtime_metrics.checkpoints_queued = runtime_metrics
                                                .checkpoints_queued
                                                .saturating_add(1);
                                        }
                                        PersistEnqueueOutcome::Saturated => {
                                            epoch.session.dropped_messages =
                                                epoch.session.dropped_messages.saturating_add(1);
                                        }
                                        PersistEnqueueOutcome::Closed => {
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
                            }
                            if active_failure.is_some() {
                                break;
                            }
                        }
                    }
                }
            }
            scheduled_at = heartbeat.tick() => {
                let mut active_probe = None;
                if let Some(epoch) = active.as_mut() {
                    match send_clob_text(&mut epoch.socket, "PING".to_string(), &mut shutdown).await {
                            Ok(()) => {
                                let sent_instant = Instant::now();
                                epoch.watchdog.record_text_ping(sent_instant, pong_timeout);
                                active_probe = Some(epoch.telemetry.record_heartbeat_probe(
                                    Utc::now(),
                                    sent_instant,
                                    sent_instant.saturating_duration_since(scheduled_at),
                                ));
                            }
                            Err(ClobSendFailure::Shutdown) => shutdown_requested = true,
                            Err(ClobSendFailure::Timeout) => {
                                active_failure = Some((
                                    "heartbeat_send_timeout".to_string(),
                                    ClobDisconnectCause::TransportFailure,
                                    None,
                                ));
                            }
                            Err(ClobSendFailure::Transport { error, detail }) => {
                                epoch.telemetry.last_transport_error = Some(detail);
                                active_failure = Some((
                                    bounded_clob_error_reason(&format!(
                                        "heartbeat_send_failed:{error}"
                                    )),
                                    ClobDisconnectCause::TransportFailure,
                                    None,
                                ));
                            }
                    }
                }
                if let Some(epoch) = successor.as_mut() {
                    match send_clob_text(&mut epoch.socket, "PING".to_string(), &mut shutdown).await {
                            Ok(()) => {
                                let sent_instant = Instant::now();
                                epoch.watchdog.record_text_ping(sent_instant, pong_timeout);
                                epoch.telemetry.record_heartbeat_probe(
                                    Utc::now(),
                                    sent_instant,
                                    sent_instant.saturating_duration_since(scheduled_at),
                                );
                            }
                            Err(ClobSendFailure::Shutdown) => shutdown_requested = true,
                            Err(ClobSendFailure::Timeout) => {
                                successor_failure = Some((
                                    "heartbeat_send_timeout".to_string(),
                                    ClobDisconnectCause::TransportFailure,
                                ));
                            }
                            Err(ClobSendFailure::Transport { error, detail }) => {
                                epoch.telemetry.last_transport_error = Some(detail);
                                successor_failure = Some((
                                    bounded_clob_error_reason(&format!(
                                        "heartbeat_send_failed:{error}"
                                    )),
                                    ClobDisconnectCause::TransportFailure,
                                ));
                            }
                    }
                }
                if let Some(sample) = active_probe {
                    record_active_clob_heartbeat_probe_metrics(&metrics, sample).await;
                }
            }
            _ = &mut retry_sleep, if retry_deadline.is_some() => {}
        }

        if shutdown_requested || terminate {
            break;
        }

        if let Some((reason, cause)) = successor_failure {
            if let Some(mut failed) = successor.take() {
                let checked_at = Utc::now();
                let active_structurally_ready_now =
                    clob_active_epoch_structurally_ready(active.as_ref(), checked_at);
                successor_rapid_retry_allowed = true;
                let retry_action = clob_candidate_retry_action(
                    &config,
                    failed.healthy_epoch,
                    active_structurally_ready_now,
                    successor_rapid_retry_allowed,
                    &mut successor_failures,
                );
                successor_retry_at = match retry_action {
                    ClobRetryAction::ImmediateRecovery => Instant::now(),
                    ClobRetryAction::Backoff(delay) => Instant::now() + delay,
                    ClobRetryAction::Stop => unreachable!("failed successors always retry"),
                };
                let close_action = clob_failure_close_action(&failed.telemetry);
                let _ = complete_clob_epoch_with_close(
                    &repository,
                    &metrics,
                    &mut failed,
                    reason.clone(),
                    cause,
                    retry_action,
                    successor_failures,
                    close_action,
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
            if let Some(error) = fatal_error {
                let Some(mut failed) = active.take() else {
                    continue;
                };
                let retry_action = clob_retry_action(
                    &config,
                    failed.healthy_epoch,
                    true,
                    &mut consecutive_failures,
                );
                {
                    let mut runtime_metrics = metrics.write().await;
                    clob_disconnect_metrics(&mut runtime_metrics, cause, retry_action);
                    runtime_metrics.clob_last_disconnect_at = Some(Utc::now());
                    runtime_metrics.clob_last_disconnect_reason = Some(reason.clone());
                }
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
                Ok(ClobPromotionOutcome::Promoted { retired }) => {
                    successor_failures = 0;
                    successor_retry_at = Instant::now();
                    successor_rapid_retry_allowed = true;
                    debug_assert!(retired.is_some());
                    if let Some(mut failed) = retired {
                        let mut retired_failures = consecutive_failures;
                        let retry_action = clob_retry_action(
                            &config,
                            failed.healthy_epoch,
                            false,
                            &mut retired_failures,
                        );
                        {
                            let mut runtime_metrics = metrics.write().await;
                            clob_disconnect_metrics(&mut runtime_metrics, cause, retry_action);
                            runtime_metrics.clob_last_disconnect_at = Some(Utc::now());
                            runtime_metrics.clob_last_disconnect_reason = Some(reason.clone());
                        }
                        let close_action = clob_handoff_close_action(cause, &failed.telemetry);
                        let _ = complete_clob_epoch_with_close(
                            &repository,
                            &metrics,
                            &mut failed,
                            reason,
                            cause,
                            retry_action,
                            retired_failures,
                            close_action,
                        )
                        .await;
                    }
                }
                Ok(ClobPromotionOutcome::NotReady)
                    if cause == ClobDisconnectCause::ReadinessRefresh =>
                {
                    if successor.is_none() && connect_task.is_none() {
                        successor_failures = 0;
                        successor_retry_at = Instant::now();
                        successor_rapid_retry_allowed = true;
                    }
                }
                Ok(ClobPromotionOutcome::NotReady) => {
                    let Some(mut failed) = active.take() else {
                        continue;
                    };
                    let retry_action = clob_retry_action(
                        &config,
                        failed.healthy_epoch,
                        false,
                        &mut consecutive_failures,
                    );
                    {
                        let mut runtime_metrics = metrics.write().await;
                        clob_disconnect_metrics(&mut runtime_metrics, cause, retry_action);
                        runtime_metrics.clob_last_disconnect_at = Some(Utc::now());
                        runtime_metrics.clob_last_disconnect_reason = Some(reason.clone());
                    }
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
                        failed.connection_id,
                        failed.connection_epoch,
                        unavailable_at,
                        unavailable_instant,
                        ClobReadinessDiagnostic::transport_unavailable(),
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
                        successor_rapid_retry_allowed = true;
                    }
                    let close_action = clob_failure_close_action(&failed.telemetry);
                    let _ = complete_clob_epoch_with_close(
                        &repository,
                        &metrics,
                        &mut failed,
                        reason,
                        cause,
                        retry_action,
                        consecutive_failures,
                        close_action,
                    )
                    .await;
                }
                Err(error) => {
                    let Some(mut failed) = active.take() else {
                        record_critical_persistence_error(&metrics, error).await;
                        return;
                    };
                    let retry_action = clob_retry_action(
                        &config,
                        failed.healthy_epoch,
                        true,
                        &mut consecutive_failures,
                    );
                    {
                        let mut runtime_metrics = metrics.write().await;
                        clob_disconnect_metrics(&mut runtime_metrics, cause, retry_action);
                        runtime_metrics.clob_last_disconnect_at = Some(Utc::now());
                        runtime_metrics.clob_last_disconnect_reason = Some(reason.clone());
                    }
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
        let close_action = clob_stop_close_action(&epoch.telemetry);
        let _ = complete_clob_epoch_with_close(
            &repository,
            &metrics,
            &mut epoch,
            stop_reason.to_string(),
            stop_cause,
            ClobRetryAction::Stop,
            consecutive_failures,
            close_action,
        )
        .await;
    }
    if let Some(mut epoch) = successor.take() {
        let close_action = clob_stop_close_action(&epoch.telemetry);
        let _ = complete_clob_epoch_with_close(
            &repository,
            &metrics,
            &mut epoch,
            stop_reason.to_string(),
            stop_cause,
            ClobRetryAction::Stop,
            successor_failures,
            close_action,
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
    ReadinessRefresh,
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
            Self::ReadinessRefresh => "readiness_refresh",
            Self::Shutdown => "shutdown",
            Self::MarketWatchClosed => "market_watch_closed",
            Self::CriticalPersistence => "critical_persistence",
        }
    }
}

fn clob_disconnect_cause(
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
    } else {
        ClobDisconnectCause::TransportFailure
    }
}

fn private_clob_disconnect_cause(reason: Option<&str>) -> ClobDisconnectCause {
    if reason.is_some_and(|reason| reason.starts_with("remote_close")) {
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
    active_structurally_ready: bool,
    rapid_retry_allowed: bool,
    consecutive_failures: &mut u32,
) -> ClobRetryAction {
    if healthy_epoch {
        *consecutive_failures = 0;
        ClobRetryAction::ImmediateRecovery
    } else if !active_structurally_ready && rapid_retry_allowed {
        *consecutive_failures = 0;
        ClobRetryAction::Backoff(clob_unavailable_retry_delay(config))
    } else {
        *consecutive_failures = consecutive_failures.saturating_add(1);
        ClobRetryAction::Backoff(reconnect_backoff(config, *consecutive_failures))
    }
}

fn clob_unavailable_retry_delay(config: &BtcRuntimeConfig) -> StdDuration {
    config
        .reconnect_initial_delay
        .min(StdDuration::from_secs(1))
}

fn expedite_clob_candidate_retry(
    config: &BtcRuntimeConfig,
    active_structurally_ready: bool,
    rapid_retry_allowed: bool,
    now: Instant,
    retry_at: Instant,
    consecutive_failures: &mut u32,
) -> Instant {
    if active_structurally_ready || !rapid_retry_allowed {
        retry_at
    } else {
        *consecutive_failures = 0;
        retry_at.min(now + clob_unavailable_retry_delay(config))
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
    clob_epoch_readiness_diagnostic(registry, markets, now, max_age).is_none()
}

fn clob_epoch_readiness_diagnostic(
    registry: &BookRegistry,
    markets: &[BtcIntervalMarket],
    now: DateTime<Utc>,
    max_age: Duration,
) -> Option<ClobReadinessDiagnostic> {
    let Some(market) = unique_current_clob_market(markets, now) else {
        return Some(ClobReadinessDiagnostic::missing_current_market());
    };
    if registry.market_books_ready(market, now, max_age) {
        return None;
    }
    let books = registry.book_readiness();
    for token_id in [&market.up_token_id, &market.down_token_id] {
        let Some(book) = books.iter().find(|book| book.token_id == *token_id) else {
            return Some(ClobReadinessDiagnostic {
                reason: "missing_book",
                market_id: Some(market.market_id.clone()),
                token_id: Some(token_id.clone()),
                integrity_status: None,
                bootstrapped: None,
                has_bid: None,
                has_ask: None,
                source_age_milliseconds: None,
                receipt_age_milliseconds: None,
                source_to_receive_lag_milliseconds: None,
            });
        };
        let source_age_milliseconds = book
            .source_timestamp
            .map(|source_timestamp| (now - source_timestamp).num_milliseconds());
        let receipt_age_milliseconds = book
            .received_at
            .map(|received_at| (now - received_at).num_milliseconds());
        let source_to_receive_lag_milliseconds =
            book.source_timestamp
                .zip(book.received_at)
                .map(|(source_timestamp, received_at)| {
                    (received_at - source_timestamp).num_milliseconds()
                });
        let reason = if book.market_id != market.market_id {
            Some("book_identity_mismatch")
        } else if !book.bootstrapped {
            Some("book_not_bootstrapped")
        } else if book.integrity_status != FeedIntegrityStatus::Ok {
            Some("book_integrity")
        } else if book.source_timestamp.is_none() {
            Some("missing_source_timestamp")
        } else if source_age_milliseconds.is_some_and(|age| age < -max_age.num_milliseconds()) {
            Some("future_source_timestamp")
        } else if book.received_at.is_none() {
            Some("missing_received_at")
        } else if receipt_age_milliseconds.is_some_and(|age| age < 0) {
            Some("future_received_at")
        } else if source_to_receive_lag_milliseconds
            .is_some_and(|lag| lag > max_age.num_milliseconds())
        {
            Some("source_to_receive_lag")
        } else {
            None
        };
        if let Some(reason) = reason {
            return Some(ClobReadinessDiagnostic {
                reason,
                market_id: Some(market.market_id.clone()),
                token_id: Some(token_id.clone()),
                integrity_status: Some(book.integrity_status),
                bootstrapped: Some(book.bootstrapped),
                has_bid: Some(book.best_bid.is_some()),
                has_ask: Some(book.best_ask.is_some()),
                source_age_milliseconds,
                receipt_age_milliseconds,
                source_to_receive_lag_milliseconds,
            });
        }
    }
    Some(ClobReadinessDiagnostic {
        reason: "book_identity_mismatch",
        market_id: Some(market.market_id.clone()),
        token_id: None,
        integrity_status: None,
        bootstrapped: None,
        has_bid: None,
        has_ask: None,
        source_age_milliseconds: None,
        receipt_age_milliseconds: None,
        source_to_receive_lag_milliseconds: None,
    })
}

fn clob_epoch_structurally_ready(
    registry: &BookRegistry,
    markets: &[BtcIntervalMarket],
    now: DateTime<Utc>,
) -> bool {
    unique_current_clob_market(markets, now)
        .is_some_and(|market| registry.market_books_structurally_ready(market))
}

fn clob_active_epoch_structurally_ready(active: Option<&ClobEpoch>, now: DateTime<Utc>) -> bool {
    active.is_some_and(|epoch| clob_epoch_structurally_ready(&epoch.registry, &epoch.markets, now))
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
    if successor.connection_id != successor.registry.connection_id()
        || !same_market_subscriptions(&successor.markets, desired_markets)
    {
        return false;
    }
    let Some(current_market) = unique_current_clob_market(desired_markets, checked_at) else {
        return false;
    };
    successor
        .registry
        .market_books_ready(current_market, checked_at, max_book_age)
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
        clear_clob_book_unavailable_metrics(&mut runtime_metrics);
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
    connection_id: Uuid,
    connection_epoch: i32,
    unavailable_at: DateTime<Utc>,
    unavailable_instant: Instant,
    diagnostic: ClobReadinessDiagnostic,
    recovery_window: &mut ClobRecoveryWindow,
) {
    recovery_window.open_if_closed(unavailable_at, unavailable_instant);
    if !recovery_window.update_diagnostic(diagnostic.clone()) {
        return;
    }
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.clob_active_connection_epoch = None;
    runtime_metrics.clob_active_connection_id = None;
    runtime_metrics.clob_recovery_unavailable_since = recovery_window.since;
    runtime_metrics.clob_book_unavailable_reason = Some(diagnostic.reason.to_string());
    runtime_metrics.clob_book_unavailable_market_id = diagnostic.market_id.clone();
    runtime_metrics.clob_book_unavailable_token_id = diagnostic.token_id.clone();
    runtime_metrics.clob_book_unavailable_integrity_status = diagnostic.integrity_status;
    runtime_metrics.clob_book_unavailable_bootstrapped = diagnostic.bootstrapped;
    runtime_metrics.clob_book_unavailable_has_bid = diagnostic.has_bid;
    runtime_metrics.clob_book_unavailable_has_ask = diagnostic.has_ask;
    runtime_metrics.clob_book_unavailable_source_age_milliseconds =
        diagnostic.source_age_milliseconds;
    runtime_metrics.clob_book_unavailable_receipt_age_milliseconds =
        diagnostic.receipt_age_milliseconds;
    runtime_metrics.clob_book_unavailable_source_to_receive_lag_milliseconds =
        diagnostic.source_to_receive_lag_milliseconds;
    drop(runtime_metrics);
    tracing::warn!(
        feed = "polymarket_clob_market",
        %connection_id,
        connection_epoch,
        reason = diagnostic.reason,
        market_id = ?diagnostic.market_id,
        token_id = ?diagnostic.token_id,
        integrity_status = ?diagnostic.integrity_status,
        bootstrapped = ?diagnostic.bootstrapped,
        has_bid = ?diagnostic.has_bid,
        has_ask = ?diagnostic.has_ask,
        source_age_ms = ?diagnostic.source_age_milliseconds,
        receipt_age_ms = ?diagnostic.receipt_age_milliseconds,
        source_to_receive_lag_ms = ?diagnostic.source_to_receive_lag_milliseconds,
        "CLOB orderbooks became unavailable; transport retained"
    );
}

fn clear_clob_book_unavailable_metrics(metrics: &mut BtcRuntimeMetrics) {
    metrics.clob_book_unavailable_reason = None;
    metrics.clob_book_unavailable_market_id = None;
    metrics.clob_book_unavailable_token_id = None;
    metrics.clob_book_unavailable_integrity_status = None;
    metrics.clob_book_unavailable_bootstrapped = None;
    metrics.clob_book_unavailable_has_bid = None;
    metrics.clob_book_unavailable_has_ask = None;
    metrics.clob_book_unavailable_source_age_milliseconds = None;
    metrics.clob_book_unavailable_receipt_age_milliseconds = None;
    metrics.clob_book_unavailable_source_to_receive_lag_milliseconds = None;
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
    let unavailable =
        clob_epoch_readiness_diagnostic(registry, active_markets, checked_at, max_book_age);
    let ready_now = unavailable.is_none();
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
    } else if !ready_now {
        *books_usable = false;
        record_clob_epoch_unavailable(
            metrics,
            connection_id,
            connection_epoch,
            checked_at,
            checked_instant,
            unavailable.expect("unavailable CLOB epoch has a diagnostic"),
            recovery_window,
        )
        .await;
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
    metrics.clob_active_subscription_target_fingerprint_sha256 = None;
    metrics.clob_active_peer_address = None;
    metrics.clob_active_edge_request_id = None;
    metrics.clob_active_edge_server = None;
    metrics.clob_active_handshake_date = None;
    metrics.clob_active_last_data_or_heartbeat_at = None;
    metrics.clob_active_last_source_to_receive_lag_milliseconds = None;
    metrics.clob_active_heartbeat_probes = 0;
    metrics.clob_active_heartbeat_acknowledgements = 0;
    metrics.clob_active_last_heartbeat_sent_at = None;
    metrics.clob_active_last_heartbeat_acknowledged_at = None;
    metrics.clob_active_last_heartbeat_send_lateness_milliseconds = 0;
    metrics.clob_active_max_heartbeat_send_lateness_milliseconds = 0;
    metrics.clob_active_last_pong_round_trip_milliseconds = None;
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
        "ignored_foreign_events": subscription_stats.ignored_foreign_events,
        "ignored_superseded_events": subscription_stats.ignored_superseded_events,
        "last_subscription_update_at": subscription_stats.last_updated_at,
    })
}

#[allow(clippy::too_many_arguments)]
fn log_clob_disconnect(
    connection_id: Uuid,
    connection_epoch: i32,
    connected_duration: StdDuration,
    observed_at: Instant,
    healthy_epoch: bool,
    disconnect_cause: ClobDisconnectCause,
    retry_action: ClobRetryAction,
    consecutive_failures: u32,
    reason: &str,
    subscription_count: usize,
    telemetry: &ClobSocketTelemetry,
) {
    let retry_delay_milliseconds = match retry_action {
        ClobRetryAction::Backoff(delay) => duration_milliseconds(delay),
        ClobRetryAction::Stop | ClobRetryAction::ImmediateRecovery => 0,
    };
    let immediate_recovery = matches!(retry_action, ClobRetryAction::ImmediateRecovery);
    let connected_duration_milliseconds = duration_milliseconds(connected_duration);
    let last_ping_age_milliseconds = telemetry
        .last_heartbeat_sent_instant
        .map(|sent_at| duration_milliseconds(observed_at.saturating_duration_since(sent_at)));
    let last_pong_age_milliseconds =
        telemetry
            .last_heartbeat_acknowledged_instant
            .map(|acknowledged_at| {
                duration_milliseconds(observed_at.saturating_duration_since(acknowledged_at))
            });
    let last_frame_age_milliseconds = telemetry
        .last_frame_instant
        .map(|frame_at| duration_milliseconds(observed_at.saturating_duration_since(frame_at)));
    let transport_error_class = telemetry
        .last_transport_error
        .as_ref()
        .map(|error| error.class);
    let transport_io_kind = telemetry
        .last_transport_error
        .as_ref()
        .and_then(|error| error.io_kind.as_deref());
    let transport_os_error_code = telemetry
        .last_transport_error
        .as_ref()
        .and_then(|error| error.os_error_code);
    macro_rules! emit_disconnect {
        ($level:expr) => {
            tracing::event!(
                $level,
                feed = "polymarket_clob_market",
                %connection_id,
                connection_epoch,
                connected_duration_ms = connected_duration_milliseconds,
                healthy_epoch,
                disconnect_cause = disconnect_cause.as_str(),
                immediate_recovery,
                failure_streak = consecutive_failures,
                retry_delay_ms = retry_delay_milliseconds,
                connection_role = telemetry.role.as_str(),
                peer_address = ?telemetry.provenance.peer_address.as_deref(),
                edge_request_id = ?telemetry.provenance.edge_request_id.as_deref(),
                edge_server = ?telemetry.provenance.edge_server.as_deref(),
                handshake_date = ?telemetry.provenance.handshake_date.as_deref(),
                subscription_count,
                subscription_target_fingerprint_sha256 =
                    %telemetry.subscription_target_fingerprint_sha256,
                heartbeat_probes = telemetry.heartbeat_probes,
                heartbeat_acknowledgements = telemetry.heartbeat_acknowledgements,
                last_ping_age_ms = ?last_ping_age_milliseconds,
                last_pong_age_ms = ?last_pong_age_milliseconds,
                last_pong_round_trip_ms = ?telemetry.last_pong_round_trip.map(duration_milliseconds),
                last_heartbeat_send_lateness_ms =
                    duration_milliseconds(telemetry.last_heartbeat_send_lateness),
                max_heartbeat_send_lateness_ms =
                    duration_milliseconds(telemetry.max_heartbeat_send_lateness),
                last_frame_age_ms = ?last_frame_age_milliseconds,
                remote_close_observed = telemetry.remote_close_observed,
                remote_close_code = ?telemetry.remote_close_code,
                remote_close_reason = ?telemetry.remote_close_reason.as_deref(),
                transport_error_class = ?transport_error_class,
                transport_io_kind = ?transport_io_kind,
                transport_os_error_code = ?transport_os_error_code,
                reason,
                "CLOB websocket disconnected"
            )
        };
    }
    match disconnect_cause {
        ClobDisconnectCause::Shutdown
        | ClobDisconnectCause::MarketWatchClosed
        | ClobDisconnectCause::ReadinessRefresh => emit_disconnect!(tracing::Level::INFO),
        ClobDisconnectCause::CriticalPersistence => emit_disconnect!(tracing::Level::ERROR),
        ClobDisconnectCause::ConnectFailure
        | ClobDisconnectCause::SubscriptionFailure
        | ClobDisconnectCause::BootstrapFailure
        | ClobDisconnectCause::TransportFailure => emit_disconnect!(tracing::Level::WARN),
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

fn update_binance_reference_and_model_window(
    state: &mut RealtimeState,
    tick: ReferencePriceTick,
    trade: &BinanceAggregateTrade,
    received_at: DateTime<Utc>,
    max_reference_age: Duration,
) -> (bool, Option<anyhow::Error>) {
    let health_progress = update_reference_state_and_check_progress(
        state,
        tick,
        ReferenceFeedKind::Binance,
        received_at,
        max_reference_age,
    );
    let aggregation_error = state
        .binance_one_second_window
        .update(trade, received_at)
        .err();
    if aggregation_error.is_some() {
        // A malformed, regressing or discontinuous accumulator must fail closed for
        // model inference without degrading the canonical live price feed.
        state.binance_one_second_window.clear();
    }
    (health_progress, aggregation_error)
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
    heartbeat_interval: StdDuration,
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
                let mut watchdog = ReferenceFeedWatchdog::new(watchdog_started);
                let mut heartbeat =
                    interval_at(watchdog_started + heartbeat_interval, heartbeat_interval);
                heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
                let read_idle_sleep = sleep(REFERENCE_READ_IDLE_TIMEOUT);
                let stable_sleep = sleep(REFERENCE_STABLE_RESET_AFTER);
                tokio::pin!(read_idle_sleep, stable_sleep);
                'connection: loop {
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
                                            let decoded = serde_json::from_str::<serde_json::Value>(&text)
                                                .context("failed to decode RTDS JSON");
                                            if let Ok(value) = decoded.as_ref() {
                                                if is_rtds_twap_60_update(value) {
                                                    match parse_rtds_chainlink_twap_60(value, received_at) {
                                                        Ok(point) => state
                                                            .write()
                                                            .await
                                                            .chainlink_twap_60
                                                            .observe(point),
                                                        Err(error) => {
                                                            session.decode_errors = session.decode_errors.saturating_add(1);
                                                            {
                                                                let mut runtime_metrics = metrics.write().await;
                                                                runtime_metrics.decode_errors = runtime_metrics
                                                                    .decode_errors
                                                                    .saturating_add(1);
                                                            }
                                                            record_error(&metrics, error).await;
                                                        }
                                                    }
                                                    continue;
                                                }
                                            }
                                            let parsed = decoded.and_then(|value| {
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
                                                    let (health_progress, external_error) = {
                                                        let mut realtime = state.write().await;
                                                        let external_error = realtime
                                                            .directional_external
                                                            .observe_rtds_chainlink(&tick)
                                                            .err();
                                                        let health_progress = update_reference_state_and_check_progress(
                                                            &mut realtime,
                                                            tick.clone(),
                                                            kind,
                                                            received_at,
                                                            chrono_duration(config.max_reference_age),
                                                        );
                                                        (health_progress, external_error)
                                                    };
                                                    if let Some(error) = external_error {
                                                        tracing::warn!(
                                                            error = %error,
                                                            "RTDS Chainlink directional history rejected a tick"
                                                        );
                                                    }
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
                                                        watchdog.on_required_tick(received_instant);
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
                                                        let observation = boundaries
                                                            .write()
                                                            .await
                                                            .observe_chainlink(&tick, max_delay);
                                                        if !observation.finalized_late_ticks.is_empty() {
                                                            let quarantined = observation
                                                                .finalized_late_ticks
                                                                .len() as u64;
                                                            let outcome_conflicts = observation
                                                                .finalized_late_ticks
                                                                .iter()
                                                                .filter(|anomaly| {
                                                                    anomaly.changes_label_outcome
                                                                })
                                                                .count() as u64;
                                                            {
                                                                let mut runtime_metrics =
                                                                    metrics.write().await;
                                                                runtime_metrics
                                                                    .finalized_boundary_ticks_quarantined =
                                                                    runtime_metrics
                                                                        .finalized_boundary_ticks_quarantined
                                                                        .saturating_add(quarantined);
                                                                runtime_metrics
                                                                    .finalized_boundary_outcome_conflicts =
                                                                    runtime_metrics
                                                                        .finalized_boundary_outcome_conflicts
                                                                        .saturating_add(outcome_conflicts);
                                                            }
                                                            for anomaly in
                                                                observation.finalized_late_ticks
                                                            {
                                                                tracing::warn!(
                                                                    market_id = %anomaly.market_id,
                                                                    boundary = anomaly.boundary.as_str(),
                                                                    tick_id = %tick.tick_id,
                                                                    tick_source_timestamp =
                                                                        %tick.source_timestamp,
                                                                    acknowledged_source_timestamp =
                                                                        %anomaly.acknowledged_source_timestamp,
                                                                    changes_label_outcome =
                                                                        anomaly.changes_label_outcome,
                                                                    "late Chainlink boundary tick quarantined after immutable finalization"
                                                                );
                                                            }
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
                                                    match enqueue(
                                                        &writer,
                                                        PersistItem::ReferenceTick(tick),
                                                        &state,
                                                        &metrics,
                                                    )
                                                    .await
                                                    {
                                                        PersistEnqueueOutcome::Queued => {
                                                            session.messages_persisted = session
                                                                .messages_persisted
                                                                .saturating_add(1);
                                                        }
                                                        PersistEnqueueOutcome::Saturated => {
                                                            session.dropped_messages = session
                                                                .dropped_messages
                                                                .saturating_add(1);
                                                        }
                                                        PersistEnqueueOutcome::Closed => {
                                                            session.dropped_messages = session
                                                                .dropped_messages
                                                                .saturating_add(1);
                                                            disconnect_reason = ReferenceDisconnectReason::CriticalWriterQueue;
                                                            fatal_persistence_error = Some(anyhow::anyhow!(
                                                                "RTDS reference persistence queue closed"
                                                            ));
                                                            break 'connection;
                                                        }
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

struct BinanceModelRecovery {
    window: BinanceOneSecondWindow,
    candle_count: usize,
    completed_through: DateTime<Utc>,
}

async fn fetch_binance_model_history(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<BinanceModelRecovery> {
    let completed_through = DateTime::from_timestamp(Utc::now().timestamp(), 0)
        .context("Binance model-recovery cutoff is outside the supported range")?;
    let start = completed_through
        .checked_sub_signed(Duration::seconds(
            BINANCE_ONE_SECOND_BOOTSTRAP_CAPACITY as i64,
        ))
        .context("Binance model-recovery start timestamp underflowed")?;
    let final_open = completed_through
        .checked_sub_signed(Duration::seconds(1))
        .context("Binance model-recovery final candle timestamp underflowed")?;
    let mut next_open = start;
    let mut candles = Vec::with_capacity(BINANCE_ONE_SECOND_BOOTSTRAP_CAPACITY);

    while next_open <= final_open {
        let remaining = (final_open - next_open).num_seconds() as usize + 1;
        let limit = remaining.min(BINANCE_MODEL_RECOVERY_PAGE_LIMIT);
        let page_end = next_open
            .checked_add_signed(Duration::seconds(limit as i64 - 1))
            .context("Binance model-recovery page end overflowed")?;
        let response = client
            .get(format!("{}/api/v3/klines", base_url.trim_end_matches('/')))
            .query(&[
                ("symbol", "BTCUSDT".to_string()),
                ("interval", "1s".to_string()),
                ("startTime", next_open.timestamp_millis().to_string()),
                ("endTime", (page_end.timestamp_millis() + 999).to_string()),
                ("limit", limit.to_string()),
            ])
            .send()
            .await
            .context("failed to fetch Binance one-second model history")?
            .error_for_status()
            .context("Binance one-second model-history request failed")?;
        let rows = response
            .json::<serde_json::Value>()
            .await
            .context("failed to decode Binance one-second model history")?;
        let rows = rows
            .as_array()
            .context("Binance one-second model history was not an array")?;
        if rows.len() != limit {
            bail!(
                "Binance one-second model history returned {} rows; expected {limit}",
                rows.len()
            );
        }
        let recovered_at = Utc::now();
        for row in rows {
            let candle = parse_binance_rest_kline(row, recovered_at)?;
            if candle.open_timestamp != next_open {
                bail!(
                    "Binance one-second model history expected {} but received {}",
                    next_open,
                    candle.open_timestamp
                );
            }
            next_open = candle.close_timestamp;
            candles.push(candle);
        }
    }
    if candles.len() != BINANCE_ONE_SECOND_BOOTSTRAP_CAPACITY || next_open != completed_through {
        bail!("Binance one-second model history did not cover the complete recovery window");
    }
    let candle_count = candles.len();
    let window = BinanceOneSecondWindow::from_completed(candles)
        .context("Binance one-second model history failed validation")?;
    Ok(BinanceModelRecovery {
        window,
        candle_count,
        completed_through,
    })
}

fn parse_binance_rest_kline(
    value: &serde_json::Value,
    recovered_at: DateTime<Utc>,
) -> Result<BinanceOneSecondKline> {
    let row = value
        .as_array()
        .context("Binance one-second kline was not an array")?;
    if row.len() < 11 {
        bail!("Binance one-second kline omitted required fields");
    }
    let open_milliseconds = json_i64(&row[0], "open time")?;
    let venue_close_milliseconds = json_i64(&row[6], "close time")?;
    let open_timestamp = DateTime::from_timestamp_millis(open_milliseconds)
        .context("Binance one-second kline open time is outside the supported range")?;
    let close_timestamp = open_timestamp
        .checked_add_signed(Duration::seconds(1))
        .context("Binance one-second kline close time overflowed")?;
    if venue_close_milliseconds != close_timestamp.timestamp_millis() - 1 {
        bail!("Binance one-second kline has an invalid close time");
    }
    let trade_count = json_u64(&row[8], "trade count")?;
    Ok(BinanceOneSecondKline {
        open_timestamp,
        close_timestamp,
        open_price: json_decimal(&row[1], "open price")?,
        high_price: json_decimal(&row[2], "high price")?,
        low_price: json_decimal(&row[3], "low price")?,
        close_price: json_decimal(&row[4], "close price")?,
        base_volume: json_decimal(&row[5], "base volume")?,
        quote_volume: json_decimal(&row[7], "quote volume")?,
        trade_count,
        taker_buy_base_volume: json_decimal(&row[9], "taker-buy base volume")?,
        taker_buy_quote_volume: json_decimal(&row[10], "taker-buy quote volume")?,
        first_aggregate_trade_id: 0,
        last_aggregate_trade_id: 0,
        first_source_timestamp: open_timestamp,
        last_source_timestamp: close_timestamp - Duration::milliseconds(1),
        max_received_at: recovered_at,
        source_complete: true,
        synthetic: trade_count == 0,
    })
}

fn json_i64(value: &serde_json::Value, field: &str) -> Result<i64> {
    value
        .as_i64()
        .with_context(|| format!("Binance one-second kline {field} was not an integer"))
}

fn json_u64(value: &serde_json::Value, field: &str) -> Result<u64> {
    value
        .as_u64()
        .with_context(|| format!("Binance one-second kline {field} was not unsigned"))
}

fn json_decimal(value: &serde_json::Value, field: &str) -> Result<Decimal> {
    let value = value
        .as_str()
        .with_context(|| format!("Binance one-second kline {field} was not a string"))?;
    Decimal::from_str(value)
        .with_context(|| format!("Binance one-second kline {field} was invalid"))
}

fn replay_binance_recovery_buffer(
    recovery: &mut BinanceModelRecovery,
    buffer: &VecDeque<(BinanceAggregateTrade, DateTime<Utc>)>,
) -> Result<()> {
    for (trade, received_at) in buffer {
        if trade.transact_time < recovery.completed_through {
            continue;
        }
        recovery
            .window
            .update(trade, *received_at)
            .context("buffered Binance aggregate trade did not join recovered model history")?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinanceSpotL2DisconnectReason {
    Shutdown,
    ConnectTimeout,
    ConnectFailed,
    ReadIdleTimeout,
    PongSendTimeout,
    PongSendFailed,
    ServerShutdown,
    RemoteClose,
    WebsocketEof,
    TransportReadFailed,
}

impl BinanceSpotL2DisconnectReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Shutdown => "shutdown",
            Self::ConnectTimeout => "connect_timeout",
            Self::ConnectFailed => "connect_failed",
            Self::ReadIdleTimeout => "read_idle_timeout",
            Self::PongSendTimeout => "pong_send_timeout",
            Self::PongSendFailed => "pong_send_failed",
            Self::ServerShutdown => "server_shutdown",
            Self::RemoteClose => "remote_close",
            Self::WebsocketEof => "websocket_eof",
            Self::TransportReadFailed => "transport_read_failed",
        }
    }
}

#[derive(Debug)]
struct BinanceSpotL2ConnectionExit {
    reason: BinanceSpotL2DisconnectReason,
    detail: Option<String>,
    reached_synchronization: bool,
}

impl BinanceSpotL2ConnectionExit {
    fn new(
        reason: BinanceSpotL2DisconnectReason,
        detail: Option<String>,
        reached_synchronization: bool,
    ) -> Self {
        Self {
            reason,
            detail,
            reached_synchronization,
        }
    }
}

#[derive(Debug, Default)]
struct BinanceSpotL2BootstrapBuffer {
    updates: VecDeque<(BinanceSpotDepthUpdate, DateTime<Utc>)>,
    level_count: usize,
}

impl BinanceSpotL2BootstrapBuffer {
    fn push(&mut self, update: BinanceSpotDepthUpdate, received_at: DateTime<Utc>) -> Result<()> {
        let update_levels = update.level_count();
        let next_level_count = self
            .level_count
            .checked_add(update_levels)
            .context("Binance spot L2 bootstrap level count overflowed")?;
        if self.updates.len() >= BINANCE_SPOT_L2_BOOTSTRAP_EVENT_CAPACITY
            || next_level_count > BINANCE_SPOT_L2_BOOTSTRAP_LEVEL_CAPACITY
        {
            bail!("Binance spot L2 bootstrap buffer exceeded its bounded capacity");
        }
        self.updates.push_back((update, received_at));
        self.level_count = next_level_count;
        Ok(())
    }

    fn clear(&mut self) {
        self.updates.clear();
        self.level_count = 0;
    }
}

type BinanceSpotL2SnapshotFuture =
    Pin<Box<dyn Future<Output = Result<(String, DateTime<Utc>)>> + Send>>;

fn binance_spot_l2_snapshot_future(
    client: reqwest::Client,
    rest_base_url: String,
    retry_delay: StdDuration,
) -> BinanceSpotL2SnapshotFuture {
    Box::pin(async move {
        if !retry_delay.is_zero() {
            tokio::time::sleep(retry_delay).await;
        }
        let request_started = Instant::now();
        let endpoint = format!("{}/api/v3/depth", rest_base_url.trim_end_matches('/'));
        let query = [
            ("symbol", "BTCUSDT".to_string()),
            ("limit", BINANCE_SPOT_L2_SNAPSHOT_LIMIT.to_string()),
        ];
        let response = timeout(
            BINANCE_SPOT_L2_SNAPSHOT_TIMEOUT,
            client.get(endpoint).query(&query).send(),
        )
        .await
        .context("Binance spot L2 snapshot request timed out")??
        .error_for_status()
        .context("Binance spot L2 snapshot returned an error status")?;
        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("unknown")
            .to_string();
        let body = match timeout(BINANCE_SPOT_L2_SNAPSHOT_TIMEOUT, response.bytes()).await {
            Ok(Ok(body)) => body,
            Ok(Err(error)) => bail!(
                "Binance spot L2 snapshot body failed: status={status}, content_type={content_type}, elapsed_ms={}: {error}",
                duration_milliseconds(request_started.elapsed()),
            ),
            Err(_) => bail!(
                "Binance spot L2 snapshot body timed out: status={status}, content_type={content_type}, elapsed_ms={}",
                duration_milliseconds(request_started.elapsed()),
            ),
        };
        if body.len() > BINANCE_SPOT_L2_SNAPSHOT_MAX_BYTES {
            bail!("Binance spot L2 snapshot exceeded its response-size bound");
        }
        let received_at = Utc::now();
        let body = String::from_utf8(body.to_vec())
            .context("Binance spot L2 snapshot was not valid UTF-8")?;
        Ok((body, received_at))
    })
}

fn is_binance_spot_l2_server_shutdown(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|value| {
            value
                .get("e")
                .or_else(|| value.get("event"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .is_some_and(|event| {
            event.eq_ignore_ascii_case("serverShutdown")
                || event.eq_ignore_ascii_case("eventStreamTerminated")
        })
}

async fn clear_binance_spot_l2_runtime_state(
    state: &Arc<RwLock<RealtimeState>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
) {
    state.write().await.binance_spot_l2.clear();
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.binance_spot_l2.synchronized = false;
    runtime_metrics.binance_spot_l2.active_connection_id = None;
    runtime_metrics.binance_spot_l2.active_last_update_id = None;
}

async fn quarantine_binance_spot_l2_inference_state(
    state: &Arc<RwLock<RealtimeState>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    connection_id: Uuid,
    last_update_id: Option<u64>,
    reason: &str,
) {
    state
        .write()
        .await
        .binance_spot_l2
        .clear_epoch(connection_id);
    let unavailable_at = Utc::now();
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.binance_spot_l2.synchronized = false;
    runtime_metrics.binance_spot_l2.active_connection_id = Some(connection_id);
    runtime_metrics.binance_spot_l2.active_last_update_id = last_update_id;
    runtime_metrics.binance_spot_l2.last_error = Some(reason.to_string());
    runtime_metrics
        .binance_spot_l2
        .recovery_unavailable_since
        .get_or_insert(unavailable_at);
}

async fn publish_binance_spot_l2_features(
    state: &Arc<RwLock<RealtimeState>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    connection_id: Uuid,
    update_id: u64,
    features: Vec<BinanceL2OneSecondFeature>,
) -> Result<()> {
    if features.is_empty() {
        return Ok(());
    }
    let feature_count = u64::try_from(features.len()).unwrap_or(u64::MAX);
    {
        let mut realtime = state.write().await;
        for feature in features {
            realtime
                .binance_spot_l2
                .publish(connection_id, update_id, feature)
                .context("Binance spot L2 feature window rejected a feature")?;
        }
    }
    let published_at = Utc::now();
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.binance_spot_l2.features_published = runtime_metrics
        .binance_spot_l2
        .features_published
        .saturating_add(feature_count);
    runtime_metrics.binance_spot_l2.last_feature_published_at = Some(published_at);
    Ok(())
}

async fn record_binance_spot_l2_synchronized(
    engine: &BinanceSpotL2Engine,
    connection_id: Uuid,
    state: &Arc<RwLock<RealtimeState>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
) -> Result<()> {
    let synchronized_at = Utc::now();
    let update_id = engine
        .update_id()
        .context("Binance spot L2 synchronized without an update id")?;
    state
        .write()
        .await
        .binance_spot_l2
        .mark_synchronized(connection_id, update_id)?;
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.binance_spot_l2.synchronized = true;
    runtime_metrics.binance_spot_l2.active_last_update_id = engine.update_id();
    runtime_metrics.binance_spot_l2.last_synchronized_at = Some(synchronized_at);
    runtime_metrics.binance_spot_l2.last_error = None;
    runtime_metrics.binance_spot_l2.recovery_unavailable_since = None;
    runtime_metrics.binance_spot_l2.consecutive_failures = 0;
    Ok(())
}

async fn schedule_binance_spot_l2_snapshot_retry(
    snapshot_attempts: &mut u32,
    snapshot_request: &mut Option<BinanceSpotL2SnapshotFuture>,
    client: &reqwest::Client,
    config: &BtcRuntimeConfig,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
) {
    *snapshot_attempts = snapshot_attempts.saturating_add(1);
    let retry_delay = reconnect_backoff(config, *snapshot_attempts);
    *snapshot_request = Some(binance_spot_l2_snapshot_future(
        client.clone(),
        config.binance_rest_base_url.clone(),
        retry_delay,
    ));
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.binance_spot_l2.snapshot_requests = runtime_metrics
        .binance_spot_l2
        .snapshot_requests
        .saturating_add(1);
}

async fn run_binance_spot_l2_supervisor(
    config: BtcRuntimeConfig,
    state: Arc<RwLock<RealtimeState>>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
    mut shutdown: watch::Receiver<bool>,
) {
    if !config.enabled || !config.binance_spot_l2_enabled {
        return;
    }
    let unavailable_since = Utc::now();
    {
        let mut runtime_metrics = metrics.write().await;
        runtime_metrics.binance_spot_l2.enabled = true;
        runtime_metrics.binance_spot_l2.recovery_unavailable_since = Some(unavailable_since);
    }
    clear_binance_spot_l2_runtime_state(&state, &metrics).await;
    let client = reqwest::Client::builder()
        .timeout(BINANCE_SPOT_L2_SNAPSHOT_TIMEOUT)
        .build()
        .unwrap_or_default();
    let mut consecutive_failures = 0u32;

    loop {
        if *shutdown.borrow() {
            break;
        }
        let connection_id = Uuid::new_v4();
        let connect_result = tokio::select! {
            biased;
            _ = shutdown.changed() => None,
            result = timeout(
                REFERENCE_CONNECT_TIMEOUT,
                connect_async(&config.binance_spot_l2_ws_url),
            ) => Some(result),
        };
        let Some(connect_result) = connect_result else {
            break;
        };
        let (socket, _) = match connect_result {
            Ok(Ok(connected)) => connected,
            result => {
                consecutive_failures = consecutive_failures.saturating_add(1);
                let (reason, detail) = match result {
                    Ok(Err(error)) => (
                        BinanceSpotL2DisconnectReason::ConnectFailed,
                        Some(bounded_reference_detail(error)),
                    ),
                    Err(_) => (BinanceSpotL2DisconnectReason::ConnectTimeout, None),
                    Ok(Ok(_)) => unreachable!(),
                };
                let disconnected_at = Utc::now();
                {
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.binance_spot_l2.connection_failures = runtime_metrics
                        .binance_spot_l2
                        .connection_failures
                        .saturating_add(1);
                    runtime_metrics.binance_spot_l2.reconnects =
                        runtime_metrics.binance_spot_l2.reconnects.saturating_add(1);
                    runtime_metrics.binance_spot_l2.consecutive_failures = consecutive_failures;
                    runtime_metrics.binance_spot_l2.last_disconnect_at = Some(disconnected_at);
                    runtime_metrics.binance_spot_l2.last_disconnect_reason =
                        Some(reason.as_str().to_string());
                    runtime_metrics.binance_spot_l2.last_error = detail.clone();
                }
                let delay = reconnect_backoff(&config, consecutive_failures);
                tracing::warn!(
                    %connection_id,
                    reason = reason.as_str(),
                    error = detail.as_deref(),
                    backoff_ms = duration_milliseconds(delay),
                    "Binance spot L2 websocket connection failed"
                );
                if !wait_reconnect_backoff(delay, &mut shutdown).await {
                    break;
                }
                continue;
            }
        };
        let connected_at = Utc::now();
        state
            .write()
            .await
            .binance_spot_l2
            .clear_epoch(connection_id);
        {
            let mut runtime_metrics = metrics.write().await;
            runtime_metrics.binance_spot_l2.connections_established = runtime_metrics
                .binance_spot_l2
                .connections_established
                .saturating_add(1);
            runtime_metrics.binance_spot_l2.active_connection_id = Some(connection_id);
            runtime_metrics.binance_spot_l2.last_connected_at = Some(connected_at);
            runtime_metrics.binance_spot_l2.last_error = None;
        }
        tracing::info!(%connection_id, "Binance spot L2 websocket connected");
        let exit = run_binance_spot_l2_connection(
            socket,
            &client,
            &config,
            connection_id,
            &state,
            &metrics,
            &mut shutdown,
        )
        .await;

        // A reconstructed book or prior feature must never survive a transport,
        // sequence, or quality boundary.
        clear_binance_spot_l2_runtime_state(&state, &metrics).await;
        let disconnected_at = Utc::now();
        if exit.reached_synchronization {
            consecutive_failures = 0;
        }
        if exit.reason == BinanceSpotL2DisconnectReason::Shutdown {
            let mut runtime_metrics = metrics.write().await;
            runtime_metrics.binance_spot_l2.last_disconnect_at = Some(disconnected_at);
            runtime_metrics.binance_spot_l2.last_disconnect_reason =
                Some(exit.reason.as_str().to_string());
            break;
        }
        consecutive_failures = consecutive_failures.saturating_add(1);
        {
            let mut runtime_metrics = metrics.write().await;
            runtime_metrics.binance_spot_l2.reconnects =
                runtime_metrics.binance_spot_l2.reconnects.saturating_add(1);
            runtime_metrics.binance_spot_l2.consecutive_failures = consecutive_failures;
            runtime_metrics.binance_spot_l2.last_disconnect_at = Some(disconnected_at);
            runtime_metrics.binance_spot_l2.last_disconnect_reason =
                Some(exit.reason.as_str().to_string());
            runtime_metrics.binance_spot_l2.last_error = exit.detail.clone();
            runtime_metrics.binance_spot_l2.recovery_unavailable_since = Some(disconnected_at);
        }
        let delay = reconnect_backoff(&config, consecutive_failures);
        tracing::warn!(
            %connection_id,
            reason = exit.reason.as_str(),
            error = exit.detail.as_deref(),
            backoff_ms = duration_milliseconds(delay),
            "Binance spot L2 websocket disconnected; reconstruction quarantined"
        );
        if !wait_reconnect_backoff(delay, &mut shutdown).await {
            break;
        }
    }
    clear_binance_spot_l2_runtime_state(&state, &metrics).await;
}

async fn run_binance_spot_l2_connection(
    mut socket: ClobSocket,
    client: &reqwest::Client,
    config: &BtcRuntimeConfig,
    connection_id: Uuid,
    state: &Arc<RwLock<RealtimeState>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
    shutdown: &mut watch::Receiver<bool>,
) -> BinanceSpotL2ConnectionExit {
    let mut engine = BinanceSpotL2Engine::default();
    let mut bootstrap = BinanceSpotL2BootstrapBuffer::default();
    let mut snapshot_attempts = 0u32;
    let mut snapshot_request = Some(binance_spot_l2_snapshot_future(
        client.clone(),
        config.binance_rest_base_url.clone(),
        StdDuration::ZERO,
    ));
    let mut snapshot_installed = false;
    let mut reached_synchronization = false;
    let mut last_depth_update_instant: Option<Instant> = None;
    {
        let mut runtime_metrics = metrics.write().await;
        runtime_metrics.binance_spot_l2.snapshot_requests = runtime_metrics
            .binance_spot_l2
            .snapshot_requests
            .saturating_add(1);
    }
    let mut feature_tick = interval_at(
        Instant::now() + BINANCE_SPOT_L2_FEATURE_TICK,
        BINANCE_SPOT_L2_FEATURE_TICK,
    );
    feature_tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut read_idle_deadline = Instant::now() + REFERENCE_READ_IDLE_TIMEOUT;
    let read_idle_sleep = sleep_until(read_idle_deadline);
    tokio::pin!(read_idle_sleep);

    'connection: loop {
        read_idle_sleep.as_mut().reset(read_idle_deadline);
        tokio::select! {
            biased;
            _ = shutdown.changed() => {
                break 'connection BinanceSpotL2ConnectionExit::new(
                    BinanceSpotL2DisconnectReason::Shutdown,
                    None,
                    reached_synchronization,
                );
            }
            _ = &mut read_idle_sleep => {
                break 'connection BinanceSpotL2ConnectionExit::new(
                    BinanceSpotL2DisconnectReason::ReadIdleTimeout,
                    None,
                    reached_synchronization,
                );
            }
            snapshot_result = async {
                snapshot_request
                    .as_mut()
                    .expect("guarded Binance spot L2 snapshot request")
                    .await
            }, if snapshot_request.is_some() => {
                snapshot_request = None;
                let (body, snapshot_received_at) = match snapshot_result {
                    Ok(snapshot) => snapshot,
                    Err(error) => {
                        let detail = bounded_reference_detail(error);
                        {
                            let mut runtime_metrics = metrics.write().await;
                            runtime_metrics.binance_spot_l2.snapshot_failures = runtime_metrics
                                .binance_spot_l2
                                .snapshot_failures
                                .saturating_add(1);
                        }
                        engine.invalidate();
                        snapshot_installed = false;
                        schedule_binance_spot_l2_snapshot_retry(
                            &mut snapshot_attempts,
                            &mut snapshot_request,
                            client,
                            config,
                            metrics,
                        ).await;
                        tracing::warn!(
                            %connection_id,
                            error = %detail,
                            "Binance spot L2 snapshot failed; retrying without closing websocket"
                        );
                        continue;
                    }
                };
                let snapshot = match serde_json::from_str::<serde_json::Value>(&body)
                    .with_context(|| format!(
                        "failed to decode Binance spot L2 snapshot JSON: body_bytes={}",
                        body.len(),
                    ))
                    .and_then(|value| parse_depth_snapshot(&value))
                {
                    Ok(snapshot) => snapshot,
                    Err(error) => {
                        let detail = bounded_reference_detail(error);
                        {
                            let mut runtime_metrics = metrics.write().await;
                            runtime_metrics.binance_spot_l2.snapshot_failures = runtime_metrics
                                .binance_spot_l2
                                .snapshot_failures
                                .saturating_add(1);
                            runtime_metrics.binance_spot_l2.decode_errors = runtime_metrics
                                .binance_spot_l2
                                .decode_errors
                                .saturating_add(1);
                        }
                        engine.invalidate();
                        snapshot_installed = false;
                        schedule_binance_spot_l2_snapshot_retry(
                            &mut snapshot_attempts,
                            &mut snapshot_request,
                            client,
                            config,
                            metrics,
                        ).await;
                        tracing::warn!(
                            %connection_id,
                            error = %detail,
                            "Binance spot L2 snapshot decode failed; retrying without closing websocket"
                        );
                        continue;
                    }
                };
                if let Err(error) = engine.install_snapshot(snapshot, snapshot_received_at) {
                    let detail = bounded_reference_detail(error);
                    {
                        let mut runtime_metrics = metrics.write().await;
                        runtime_metrics.binance_spot_l2.snapshot_failures = runtime_metrics
                            .binance_spot_l2
                            .snapshot_failures
                            .saturating_add(1);
                    }
                    snapshot_installed = false;
                    schedule_binance_spot_l2_snapshot_retry(
                        &mut snapshot_attempts,
                        &mut snapshot_request,
                        client,
                        config,
                        metrics,
                    ).await;
                    tracing::warn!(
                        %connection_id,
                        error = %detail,
                        "Binance spot L2 snapshot was unusable; retrying without closing websocket"
                    );
                    continue;
                }
                snapshot_installed = true;

                // Replay remains tentative until the entire bounded buffer forms
                // one continuous sequence. No feature is published before that
                // atomic bootstrap succeeds.
                let mut tentative_features = Vec::new();
                let mut applied = 0u64;
                let mut discarded = 0u64;
                let mut replay_gap = None;
                for (update, received_at) in bootstrap.updates.iter().cloned() {
                    match engine.apply_update(update, received_at) {
                        Ok(BinanceSpotL2UpdateOutcome::Applied { features, .. }) => {
                            applied = applied.saturating_add(1);
                            tentative_features.extend(features);
                        }
                        Ok(BinanceSpotL2UpdateOutcome::AppliedUnqualified) => {
                            applied = applied.saturating_add(1);
                            discarded = discarded.saturating_add(1);
                            tentative_features.clear();
                        }
                        Ok(BinanceSpotL2UpdateOutcome::IgnoredStale) => {
                            discarded = discarded.saturating_add(1);
                        }
                        Ok(BinanceSpotL2UpdateOutcome::SequenceGap {
                            expected_update_id,
                            first_update_id,
                            final_update_id,
                        }) => {
                            replay_gap = Some(format!(
                                "expected update {expected_update_id}, received [{first_update_id}, {final_update_id}]"
                            ));
                            break;
                        }
                        Err(error) => {
                            replay_gap = Some(bounded_reference_detail(error));
                            break;
                        }
                    }
                }
                if let Some(detail) = replay_gap {
                    {
                        let mut runtime_metrics = metrics.write().await;
                        runtime_metrics.binance_spot_l2.sequence_gaps = runtime_metrics
                            .binance_spot_l2
                            .sequence_gaps
                            .saturating_add(1);
                        runtime_metrics.binance_spot_l2.snapshot_behind_buffer = runtime_metrics
                            .binance_spot_l2
                            .snapshot_behind_buffer
                            .saturating_add(1);
                    }
                    engine.invalidate();
                    snapshot_installed = false;
                    schedule_binance_spot_l2_snapshot_retry(
                        &mut snapshot_attempts,
                        &mut snapshot_request,
                        client,
                        config,
                        metrics,
                    ).await;
                    tracing::warn!(
                        %connection_id,
                        error = %detail,
                        "Binance spot L2 snapshot lagged buffered updates; retrying without closing websocket"
                    );
                    continue;
                }
                snapshot_attempts = 0;
                {
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.binance_spot_l2.snapshot_successes = runtime_metrics
                        .binance_spot_l2
                        .snapshot_successes
                        .saturating_add(1);
                    runtime_metrics.binance_spot_l2.updates_applied = runtime_metrics
                        .binance_spot_l2
                        .updates_applied
                        .saturating_add(applied);
                    runtime_metrics.binance_spot_l2.updates_discarded = runtime_metrics
                        .binance_spot_l2
                        .updates_discarded
                        .saturating_add(discarded);
                    runtime_metrics.binance_spot_l2.active_last_update_id = engine.update_id();
                }
                if engine.synchronized() {
                    if !reached_synchronization {
                        if let Err(error) = record_binance_spot_l2_synchronized(
                            &engine,
                            connection_id,
                            state,
                            metrics,
                        ).await {
                            let detail = bounded_reference_detail(error);
                            engine.quarantine_qualification();
                            quarantine_binance_spot_l2_inference_state(
                                state,
                                metrics,
                                connection_id,
                                engine.update_id(),
                                &detail,
                            ).await;
                            bootstrap.clear();
                            continue;
                        }
                        reached_synchronization = true;
                    }
                    let update_id = engine
                        .update_id()
                        .expect("synchronized Binance spot L2 engine has an update id");
                    if let Err(error) = publish_binance_spot_l2_features(
                        state,
                        metrics,
                        connection_id,
                        update_id,
                        tentative_features,
                    ).await {
                        let detail = bounded_reference_detail(error);
                        engine.quarantine_qualification();
                        quarantine_binance_spot_l2_inference_state(
                            state,
                            metrics,
                            connection_id,
                            engine.update_id(),
                            &detail,
                        ).await;
                        bootstrap.clear();
                        continue;
                    }
                    bootstrap.clear();
                }
            }
            _ = feature_tick.tick(), if engine.synchronized() => {
                let stale_after = StdDuration::from_millis(
                    u64::try_from(BINANCE_SPOT_L2_MAX_INFERENCE_AGE_MILLISECONDS)
                        .unwrap_or(2_000),
                );
                if last_depth_update_instant
                    .is_none_or(|last_update| last_update.elapsed() > stale_after)
                {
                    engine.quarantine_qualification();
                    quarantine_binance_spot_l2_inference_state(
                        state,
                        metrics,
                        connection_id,
                        engine.update_id(),
                        "Binance spot L2 source became stale",
                    ).await;
                    continue;
                }
                match engine.advance_time(Utc::now()) {
                    Ok(features) => {
                        let update_id = engine
                            .update_id()
                            .expect("synchronized Binance spot L2 engine has an update id");
                        if let Err(error) = publish_binance_spot_l2_features(
                            state,
                            metrics,
                            connection_id,
                            update_id,
                            features,
                        ).await {
                            let detail = bounded_reference_detail(error);
                            engine.quarantine_qualification();
                            quarantine_binance_spot_l2_inference_state(
                                state,
                                metrics,
                                connection_id,
                                engine.update_id(),
                                &detail,
                            ).await;
                            continue;
                        }
                        if !engine.synchronized() {
                            quarantine_binance_spot_l2_inference_state(
                                state,
                                metrics,
                                connection_id,
                                engine.update_id(),
                                "Binance spot L2 feature state became unqualified",
                            ).await;
                        }
                    }
                    Err(error) => {
                        let detail = bounded_reference_detail(error);
                        engine.quarantine_qualification();
                        quarantine_binance_spot_l2_inference_state(
                            state,
                            metrics,
                            connection_id,
                            engine.update_id(),
                            &detail,
                        ).await;
                    }
                }
            }
            message = socket.next() => {
                let Some(message) = message else {
                    break 'connection BinanceSpotL2ConnectionExit::new(
                        BinanceSpotL2DisconnectReason::WebsocketEof,
                        None,
                        reached_synchronization,
                    );
                };
                let message = match message {
                    Ok(message) => message,
                    Err(error) => {
                        break 'connection BinanceSpotL2ConnectionExit::new(
                            BinanceSpotL2DisconnectReason::TransportReadFailed,
                            Some(bounded_reference_detail(error)),
                            reached_synchronization,
                        );
                    }
                };
                let received_at = Utc::now();
                read_idle_deadline = Instant::now() + REFERENCE_READ_IDLE_TIMEOUT;
                match message {
                    Message::Text(text) => {
                        if is_binance_spot_l2_server_shutdown(&text) {
                            break 'connection BinanceSpotL2ConnectionExit::new(
                                BinanceSpotL2DisconnectReason::ServerShutdown,
                                None,
                                reached_synchronization,
                            );
                        }
                        let update = match parse_depth_update(&text) {
                            Ok(update) => update,
                            Err(error) => {
                                let detail = bounded_reference_detail(error);
                                {
                                    let mut runtime_metrics = metrics.write().await;
                                    runtime_metrics.binance_spot_l2.decode_errors = runtime_metrics
                                        .binance_spot_l2
                                        .decode_errors
                                        .saturating_add(1);
                                }
                                engine.invalidate();
                                bootstrap.clear();
                                snapshot_installed = false;
                                schedule_binance_spot_l2_snapshot_retry(
                                    &mut snapshot_attempts,
                                    &mut snapshot_request,
                                    client,
                                    config,
                                    metrics,
                                ).await;
                                quarantine_binance_spot_l2_inference_state(
                                    state,
                                    metrics,
                                    connection_id,
                                    None,
                                    &detail,
                                ).await;
                                continue 'connection;
                            }
                        };
                        {
                            let mut runtime_metrics = metrics.write().await;
                            runtime_metrics.binance_spot_l2.updates_received = runtime_metrics
                                .binance_spot_l2
                                .updates_received
                                .saturating_add(1);
                            runtime_metrics.binance_spot_l2.last_update_at = Some(received_at);
                        }
                        last_depth_update_instant = Some(Instant::now());
                        if !engine.synchronized() {
                            if let Err(error) = bootstrap.push(update.clone(), received_at) {
                                let detail = bounded_reference_detail(error);
                                {
                                    let mut runtime_metrics = metrics.write().await;
                                    runtime_metrics.binance_spot_l2.bootstrap_buffer_overflows =
                                        runtime_metrics
                                            .binance_spot_l2
                                            .bootstrap_buffer_overflows
                                            .saturating_add(1);
                                }
                                engine.invalidate();
                                bootstrap.clear();
                                snapshot_installed = false;
                                schedule_binance_spot_l2_snapshot_retry(
                                    &mut snapshot_attempts,
                                    &mut snapshot_request,
                                    client,
                                    config,
                                    metrics,
                                ).await;
                                quarantine_binance_spot_l2_inference_state(
                                    state,
                                    metrics,
                                    connection_id,
                                    None,
                                    &detail,
                                ).await;
                                continue;
                            }
                            if snapshot_request.is_some() {
                                continue;
                            }
                            if !snapshot_installed {
                                schedule_binance_spot_l2_snapshot_retry(
                                    &mut snapshot_attempts,
                                    &mut snapshot_request,
                                    client,
                                    config,
                                    metrics,
                                ).await;
                                continue;
                            }
                        }
                        let was_synchronized = engine.synchronized();
                        match engine.apply_update(update, received_at) {
                            Ok(BinanceSpotL2UpdateOutcome::Applied { features, .. }) => {
                                {
                                    let mut runtime_metrics = metrics.write().await;
                                    runtime_metrics.binance_spot_l2.updates_applied = runtime_metrics
                                        .binance_spot_l2
                                        .updates_applied
                                        .saturating_add(1);
                                    runtime_metrics.binance_spot_l2.active_last_update_id =
                                        engine.update_id();
                                }
                                if engine.synchronized() && !was_synchronized {
                                    bootstrap.clear();
                                    if let Err(error) = record_binance_spot_l2_synchronized(
                                        &engine,
                                        connection_id,
                                        state,
                                        metrics,
                                    ).await {
                                        let detail = bounded_reference_detail(error);
                                        engine.quarantine_qualification();
                                        quarantine_binance_spot_l2_inference_state(
                                            state,
                                            metrics,
                                            connection_id,
                                            engine.update_id(),
                                            &detail,
                                        ).await;
                                        continue;
                                    }
                                    reached_synchronization = true;
                                }
                                let update_id = engine
                                    .update_id()
                                    .expect("applied Binance spot L2 update has an update id");
                                if let Err(error) = publish_binance_spot_l2_features(
                                    state,
                                    metrics,
                                    connection_id,
                                    update_id,
                                    features,
                                ).await {
                                    let detail = bounded_reference_detail(error);
                                    engine.quarantine_qualification();
                                    quarantine_binance_spot_l2_inference_state(
                                        state,
                                        metrics,
                                        connection_id,
                                        engine.update_id(),
                                        &detail,
                                    ).await;
                                    continue;
                                }
                            }
                            Ok(BinanceSpotL2UpdateOutcome::AppliedUnqualified) => {
                                {
                                    let mut runtime_metrics = metrics.write().await;
                                    runtime_metrics.binance_spot_l2.updates_applied = runtime_metrics
                                        .binance_spot_l2
                                        .updates_applied
                                        .saturating_add(1);
                                    runtime_metrics.binance_spot_l2.updates_discarded =
                                        runtime_metrics
                                            .binance_spot_l2
                                            .updates_discarded
                                            .saturating_add(1);
                                    runtime_metrics.binance_spot_l2.active_last_update_id =
                                        engine.update_id();
                                }
                                quarantine_binance_spot_l2_inference_state(
                                    state,
                                    metrics,
                                    connection_id,
                                    engine.update_id(),
                                    "Binance spot L2 update exceeded the source-age qualification bound",
                                ).await;
                            }
                            Ok(BinanceSpotL2UpdateOutcome::IgnoredStale) => {
                                let mut runtime_metrics = metrics.write().await;
                                runtime_metrics.binance_spot_l2.updates_discarded = runtime_metrics
                                    .binance_spot_l2
                                    .updates_discarded
                                    .saturating_add(1);
                            }
                            Ok(BinanceSpotL2UpdateOutcome::SequenceGap {
                                expected_update_id,
                                first_update_id,
                                final_update_id,
                            }) => {
                                let detail = format!(
                                    "expected update {expected_update_id}, received [{first_update_id}, {final_update_id}]"
                                );
                                {
                                    let mut runtime_metrics = metrics.write().await;
                                    runtime_metrics.binance_spot_l2.sequence_gaps = runtime_metrics
                                        .binance_spot_l2
                                        .sequence_gaps
                                        .saturating_add(1);
                                }
                                engine.invalidate();
                                bootstrap.clear();
                                snapshot_installed = false;
                                schedule_binance_spot_l2_snapshot_retry(
                                    &mut snapshot_attempts,
                                    &mut snapshot_request,
                                    client,
                                    config,
                                    metrics,
                                ).await;
                                {
                                    let mut runtime_metrics = metrics.write().await;
                                    runtime_metrics.binance_spot_l2.snapshot_behind_buffer =
                                        runtime_metrics
                                            .binance_spot_l2
                                            .snapshot_behind_buffer
                                            .saturating_add(1);
                                }
                                quarantine_binance_spot_l2_inference_state(
                                    state,
                                    metrics,
                                    connection_id,
                                    None,
                                    &detail,
                                ).await;
                            }
                            Err(error) => {
                                let detail = bounded_reference_detail(error);
                                engine.invalidate();
                                bootstrap.clear();
                                snapshot_installed = false;
                                schedule_binance_spot_l2_snapshot_retry(
                                    &mut snapshot_attempts,
                                    &mut snapshot_request,
                                    client,
                                    config,
                                    metrics,
                                ).await;
                                quarantine_binance_spot_l2_inference_state(
                                    state,
                                    metrics,
                                    connection_id,
                                    None,
                                    &detail,
                                ).await;
                            }
                        }
                    }
                    Message::Ping(payload) => {
                        {
                            let mut runtime_metrics = metrics.write().await;
                            runtime_metrics.binance_spot_l2.ping_frames_received = runtime_metrics
                                .binance_spot_l2
                                .ping_frames_received
                                .saturating_add(1);
                        }
                        let send_result = tokio::select! {
                            biased;
                            _ = shutdown.changed() => None,
                            result = timeout(
                                REFERENCE_SEND_TIMEOUT,
                                socket.send(Message::Pong(payload)),
                            ) => Some(result),
                        };
                        match send_result {
                            None => {
                                break 'connection BinanceSpotL2ConnectionExit::new(
                                    BinanceSpotL2DisconnectReason::Shutdown,
                                    None,
                                    reached_synchronization,
                                );
                            }
                            Some(Err(_)) => {
                                break 'connection BinanceSpotL2ConnectionExit::new(
                                    BinanceSpotL2DisconnectReason::PongSendTimeout,
                                    None,
                                    reached_synchronization,
                                );
                            }
                            Some(Ok(Err(error))) => {
                                break 'connection BinanceSpotL2ConnectionExit::new(
                                    BinanceSpotL2DisconnectReason::PongSendFailed,
                                    Some(bounded_reference_detail(error)),
                                    reached_synchronization,
                                );
                            }
                            Some(Ok(Ok(()))) => {
                                let mut runtime_metrics = metrics.write().await;
                                runtime_metrics.binance_spot_l2.pong_frames_sent = runtime_metrics
                                    .binance_spot_l2
                                    .pong_frames_sent
                                    .saturating_add(1);
                            }
                        }
                    }
                    Message::Close(frame) => {
                        let detail = frame.and_then(|frame| {
                            (!frame.reason.is_empty())
                                .then(|| bounded_reference_detail(frame.reason))
                        });
                        break 'connection BinanceSpotL2ConnectionExit::new(
                            BinanceSpotL2DisconnectReason::RemoteClose,
                            detail,
                            reached_synchronization,
                        );
                    }
                    Message::Binary(_) => {
                        let detail = "Binance spot L2 websocket emitted an unexpected binary frame";
                        {
                            let mut runtime_metrics = metrics.write().await;
                            runtime_metrics.binance_spot_l2.decode_errors = runtime_metrics
                                .binance_spot_l2
                                .decode_errors
                                .saturating_add(1);
                        }
                        engine.invalidate();
                        bootstrap.clear();
                        snapshot_installed = false;
                        schedule_binance_spot_l2_snapshot_retry(
                            &mut snapshot_attempts,
                            &mut snapshot_request,
                            client,
                            config,
                            metrics,
                        ).await;
                        quarantine_binance_spot_l2_inference_state(
                            state,
                            metrics,
                            connection_id,
                            None,
                            detail,
                        ).await;
                    }
                    Message::Pong(_) | Message::Frame(_) => {}
                }
            }
        }
    }
}

async fn run_binance_supervisor(
    config: BtcRuntimeConfig,
    heartbeat_interval: StdDuration,
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
    let recovery_client = reqwest::Client::builder()
        .timeout(BINANCE_MODEL_RECOVERY_HTTP_TIMEOUT)
        .build()
        .unwrap_or_default();
    let mut model_recovery_required = true;
    {
        let mut runtime_metrics = metrics.write().await;
        runtime_metrics.binance_transport.recovery_unavailable_since = recovery_window.since;
        runtime_metrics.binance_model_recovery_required = true;
    }
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
        let mut watchdog = ReferenceFeedWatchdog::new(watchdog_started);
        let mut heartbeat = interval_at(watchdog_started + heartbeat_interval, heartbeat_interval);
        heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut model_recovery_retry = interval(BINANCE_MODEL_RECOVERY_RETRY_INTERVAL);
        model_recovery_retry.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut model_recovery_task: Option<JoinHandle<Result<BinanceModelRecovery>>> = None;
        let mut model_recovery_started = None;
        let mut model_recovery_buffer = VecDeque::with_capacity(1_024);
        let read_idle_sleep = sleep(REFERENCE_READ_IDLE_TIMEOUT);
        let pong_sleep = sleep(REFERENCE_PONG_TIMEOUT);
        let stable_sleep = sleep(REFERENCE_STABLE_RESET_AFTER);
        tokio::pin!(read_idle_sleep, pong_sleep, stable_sleep);
        let mut heartbeat_sequence = 0u64;
        'connection: loop {
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
                _ = model_recovery_retry.tick(), if model_recovery_required && model_recovery_task.is_none() => {
                    model_recovery_buffer.clear();
                    let client = recovery_client.clone();
                    let base_url = config.binance_rest_base_url.clone();
                    let started_at = Utc::now();
                    model_recovery_started = Some(Instant::now());
                    model_recovery_task = Some(tokio::spawn(async move {
                        fetch_binance_model_history(&client, &base_url).await
                    }));
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.binance_model_recovery_required = true;
                    runtime_metrics.binance_model_recovery_attempts = runtime_metrics
                        .binance_model_recovery_attempts
                        .saturating_add(1);
                    runtime_metrics.binance_model_recovery_last_started_at = Some(started_at);
                    runtime_metrics.binance_model_recovery_last_error = None;
                    tracing::info!(
                        %connection_id,
                        connection_epoch = reconnect_ordinal,
                        "started authoritative Binance model-history recovery"
                    );
                }
                recovery_result = async {
                    model_recovery_task
                        .as_mut()
                        .expect("guarded Binance model-recovery task")
                        .await
                }, if model_recovery_task.is_some() => {
                    model_recovery_task = None;
                    let recovery_duration = model_recovery_started
                        .take()
                        .map(|started| duration_milliseconds(started.elapsed()));
                    let recovered = match recovery_result {
                        Ok(Ok(mut recovery)) => {
                            replay_binance_recovery_buffer(
                                &mut recovery,
                                &model_recovery_buffer,
                            )
                            .map(|()| recovery)
                        }
                        Ok(Err(error)) => Err(error),
                        Err(error) => Err(anyhow::anyhow!(
                            "Binance model-recovery task failed: {error}"
                        )),
                    };
                    match recovered {
                        Ok(recovery) => {
                            let candle_count = recovery.candle_count;
                            let completed_through = recovery.completed_through;
                            state.write().await.binance_one_second_window = recovery.window;
                            model_recovery_buffer.clear();
                            model_recovery_required = false;
                            let completed_at = Utc::now();
                            let mut runtime_metrics = metrics.write().await;
                            runtime_metrics.binance_model_recovery_required = false;
                            runtime_metrics.binance_model_recovery_successes = runtime_metrics
                                .binance_model_recovery_successes
                                .saturating_add(1);
                            runtime_metrics.binance_model_recovery_candles = candle_count as u64;
                            runtime_metrics.binance_model_recovery_last_duration_milliseconds =
                                recovery_duration;
                            runtime_metrics.binance_model_recovery_last_completed_at =
                                Some(completed_at);
                            runtime_metrics.binance_model_recovery_last_error = None;
                            tracing::info!(
                                %connection_id,
                                connection_epoch = reconnect_ordinal,
                                candle_count,
                                %completed_through,
                                recovery_duration_ms = recovery_duration,
                                "atomically restored authoritative Binance model history"
                            );
                        }
                        Err(error) => {
                            let detail = error.to_string();
                            let mut runtime_metrics = metrics.write().await;
                            runtime_metrics.binance_model_recovery_required = true;
                            runtime_metrics.binance_model_recovery_failures = runtime_metrics
                                .binance_model_recovery_failures
                                .saturating_add(1);
                            runtime_metrics.binance_model_recovery_last_duration_milliseconds =
                                recovery_duration;
                            runtime_metrics.binance_model_recovery_last_error =
                                Some(detail.clone());
                            tracing::warn!(
                                %connection_id,
                                connection_epoch = reconnect_ordinal,
                                error = %detail,
                                "authoritative Binance model-history recovery failed"
                            );
                        }
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
                            let acknowledged_pong = match &message {
                                Message::Pong(payload) => {
                                    watchdog.acknowledge_binary_pong(payload.as_ref())
                                }
                                _ => false,
                            };
                            watchdog.on_frame(received_instant);
                            stats.last_frame_at = Some(received_at);
                            match message {
                                Message::Text(text) => {
                                    sequence = sequence.saturating_add(1);
                                    session.messages_received =
                                        session.messages_received.saturating_add(1);
                                    let parsed = serde_json::from_str::<serde_json::Value>(&text)
                                        .context("failed to decode Binance aggregate trade JSON")
                                        .and_then(|value| {
                                            parse_binance_agg_trade_with_details(
                                                &value,
                                                connection_id,
                                                sequence,
                                                received_at,
                                            )
                                        });
                                    match parsed {
                                        Ok((tick, trade)) => {
                                            let buffer_overflow = model_recovery_required
                                                && model_recovery_buffer.len()
                                                    >= BINANCE_MODEL_RECOVERY_BUFFER_CAPACITY;
                                            if buffer_overflow {
                                                if let Some(task) = model_recovery_task.take() {
                                                    task.abort();
                                                }
                                                model_recovery_started = None;
                                                model_recovery_buffer.clear();
                                                let mut runtime_metrics = metrics.write().await;
                                                runtime_metrics.binance_model_recovery_failures =
                                                    runtime_metrics
                                                        .binance_model_recovery_failures
                                                        .saturating_add(1);
                                                runtime_metrics
                                                    .binance_model_recovery_buffer_overflows =
                                                    runtime_metrics
                                                        .binance_model_recovery_buffer_overflows
                                                        .saturating_add(1);
                                                runtime_metrics.binance_model_recovery_last_error =
                                                    Some("live aggregate-trade recovery buffer exceeded its bounded capacity".to_string());
                                            } else if model_recovery_required {
                                                model_recovery_buffer
                                                    .push_back((trade.clone(), received_at));
                                            }
                                            let (health_progress, aggregation_error) = if model_recovery_required {
                                                let mut realtime = state.write().await;
                                                (
                                                    update_reference_state_and_check_progress(
                                                        &mut realtime,
                                                        tick.clone(),
                                                        ReferenceFeedKind::Binance,
                                                        received_at,
                                                        chrono_duration(config.max_reference_age),
                                                    ),
                                                    None,
                                                )
                                            } else {
                                                let mut realtime = state.write().await;
                                                update_binance_reference_and_model_window(
                                                    &mut realtime,
                                                    tick.clone(),
                                                    &trade,
                                                    received_at,
                                                    chrono_duration(config.max_reference_age),
                                                )
                                            };
                                            if let Some(error) = aggregation_error {
                                                model_recovery_required = true;
                                                model_recovery_buffer.clear();
                                                let mut runtime_metrics = metrics.write().await;
                                                runtime_metrics.binance_model_recovery_required = true;
                                                runtime_metrics.binance_model_recovery_gap_resets =
                                                    runtime_metrics
                                                        .binance_model_recovery_gap_resets
                                                        .saturating_add(1);
                                                runtime_metrics.binance_model_recovery_last_error =
                                                    Some(error.to_string());
                                                drop(runtime_metrics);
                                                tracing::warn!(
                                                    error = %error,
                                                    aggregate_trade_id = trade.aggregate_trade_id,
                                                    "quarantined invalid Binance model history for authoritative recovery"
                                                );
                                            }
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
                                                watchdog.on_required_tick(received_instant);
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
                                            match enqueue(
                                                &writer,
                                                PersistItem::ReferenceTick(tick),
                                                &state,
                                                &metrics,
                                            )
                                            .await
                                            {
                                                PersistEnqueueOutcome::Queued => {
                                                    session.messages_persisted = session
                                                        .messages_persisted
                                                        .saturating_add(1);
                                                }
                                                PersistEnqueueOutcome::Saturated => {
                                                    session.dropped_messages = session
                                                        .dropped_messages
                                                        .saturating_add(1);
                                                }
                                                PersistEnqueueOutcome::Closed => {
                                                    session.dropped_messages = session
                                                        .dropped_messages
                                                        .saturating_add(1);
                                                    disconnect_reason = ReferenceDisconnectReason::CriticalWriterQueue;
                                                    fatal_persistence_error = Some(anyhow::anyhow!(
                                                        "Binance reference persistence queue closed"
                                                    ));
                                                    break 'connection;
                                                }
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
                                Message::Pong(_) => {
                                    if acknowledged_pong {
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
        if let Some(task) = model_recovery_task.take() {
            task.abort();
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
                if !snapshot.primary_persistence_available() {
                    continue;
                }
                let readiness = snapshot.readiness(
                    Utc::now(),
                    chrono_duration(config.max_book_age),
                    chrono_duration(config.max_reference_age),
                );
                if let Err(error) = strategy
                    .on_observation(StrategyObservation { state: snapshot, readiness })
                    .await
                {
                    let error_chain = format!("{error:#}");
                    tracing::error!(
                        error = %error_chain,
                        "BTC strategy callback failed; terminating the trading process"
                    );
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.strategy_errors =
                        runtime_metrics.strategy_errors.saturating_add(1);
                    runtime_metrics.last_error = Some(error_chain);
                    // The deterministic strategy and shared paper execution path
                    // are primary immutable run data. A callback failure
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
        let candidates = boundaries
            .read()
            .await
            .pending_candidates(Utc::now(), max_delay)?;
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
                        "clob_rest_reconciliation" | "gamma_rest_reconciliation" => {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinalizedBoundaryKind {
    Open,
    Close,
}

impl FinalizedBoundaryKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Close => "close",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FinalizedBoundaryLateTick {
    market_id: String,
    boundary: FinalizedBoundaryKind,
    acknowledged_source_timestamp: DateTime<Utc>,
    changes_label_outcome: bool,
}

#[derive(Debug, Default)]
struct BoundaryObservation {
    finalized_late_ticks: Vec<FinalizedBoundaryLateTick>,
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

    fn observe_chainlink(
        &mut self,
        tick: &ReferencePriceTick,
        max_delay: Duration,
    ) -> BoundaryObservation {
        let mut observation = BoundaryObservation::default();
        if tick.source != ReferencePriceSource::RtdsChainlink {
            return observation;
        }
        for boundary in self.markets.values_mut() {
            if tick.source_timestamp >= boundary.market.window_start
                && tick.source_timestamp <= boundary.market.window_start + max_delay
            {
                if let Some(open) = boundary.open_tick.as_ref() {
                    if tick_precedes(tick, open) && !same_tracker_tick(tick, open) {
                        // The durable opening reference remains authoritative. Retain the late
                        // source tick for audit without invalidating current or future runtimes.
                        let changes_label_outcome = boundary
                            .close_tick
                            .as_ref()
                            .zip(boundary.label.as_ref())
                            .is_some_and(|(close, label)| {
                                boundary_outcome(tick.price, close.price) != label.outcome
                            });
                        observation
                            .finalized_late_ticks
                            .push(FinalizedBoundaryLateTick {
                                market_id: boundary.market.market_id.clone(),
                                boundary: FinalizedBoundaryKind::Open,
                                acknowledged_source_timestamp: open.source_timestamp,
                                changes_label_outcome,
                            });
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
                            // Once the label is durable, provider reordering is an auditable
                            // anomaly rather than permission to rewrite immutable market history.
                            let changes_label_outcome = boundary
                                .open_tick
                                .as_ref()
                                .zip(boundary.label.as_ref())
                                .is_some_and(|(open, label)| {
                                    boundary_outcome(open.price, tick.price) != label.outcome
                                });
                            observation
                                .finalized_late_ticks
                                .push(FinalizedBoundaryLateTick {
                                    market_id: boundary.market.market_id.clone(),
                                    boundary: FinalizedBoundaryKind::Close,
                                    acknowledged_source_timestamp: close.source_timestamp,
                                    changes_label_outcome,
                                });
                            continue;
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
        observation
    }

    fn pending_candidates(
        &self,
        observed_at: DateTime<Utc>,
        max_delay: Duration,
    ) -> Result<Vec<BoundaryCandidate>> {
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
            if boundary.label.is_none()
                && boundary.close_tick_durable
                // Wait through the complete eligible source-timestamp window so reordered
                // close ticks can converge before the market label becomes immutable.
                && observed_at >= boundary.market.window_end + max_delay
            {
                if let (Some(open), Some(close)) =
                    (boundary.open_tick.as_ref(), boundary.close_tick.as_ref())
                {
                    candidates.push(BoundaryCandidate::Label {
                        label: boundary_label(&boundary.market, open, close, observed_at),
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
    label_available_at: DateTime<Utc>,
) -> BtcMarketLabel {
    let outcome = boundary_outcome(open_tick.price, close_tick.price);
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
        label_available_at,
        evidence: serde_json::json!({
            "open_tick_id": open_tick.tick_id,
            "close_tick_id": close_tick.tick_id,
            "open_received_at": open_tick.received_at,
            "close_received_at": close_tick.received_at,
            "finalized_at": label_available_at,
            "rule": "up_when_close_greater_than_or_equal_to_open"
        }),
    }
}

fn boundary_outcome(open_price: Decimal, close_price: Decimal) -> BtcOutcome {
    if close_price >= open_price {
        BtcOutcome::Up
    } else {
        BtcOutcome::Down
    }
}

fn clob_subscription(markets: &[BtcIntervalMarket]) -> String {
    serde_json::json!({
        "assets_ids": clob_asset_ids(markets),
        "type": "market",
        "custom_feature_enabled": true,
        "initial_dump": true
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

fn clob_subscription_fingerprint(markets: &[BtcIntervalMarket]) -> String {
    let mut assets = Vec::with_capacity(markets.len().saturating_mul(2));
    for market in markets {
        assets.push(market.up_token_id.as_str());
        assets.push(market.down_token_id.as_str());
    }
    assets.sort_unstable();
    assets.dedup();
    let mut hasher = Sha256::new();
    for asset in assets {
        hasher.update(u64::try_from(asset.len()).unwrap_or(u64::MAX).to_be_bytes());
        hasher.update(asset.as_bytes());
    }
    format!("{:x}", hasher.finalize())
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
            "custom_feature_enabled": true,
            "initial_dump": true
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
    if left.len() != right.len() {
        return false;
    }
    if left
        .iter()
        .zip(right)
        .all(|(left, right)| same_market_subscription(left, right))
    {
        return true;
    }
    left.iter().all(|candidate| {
        let left_count = left
            .iter()
            .filter(|market| same_market_subscription(candidate, market))
            .count();
        let right_count = right
            .iter()
            .filter(|market| same_market_subscription(candidate, market))
            .count();
        left_count == right_count
    })
}

fn publish_market_subscriptions_if_changed(
    sender: &watch::Sender<Vec<BtcIntervalMarket>>,
    markets: &[BtcIntervalMarket],
) {
    if !same_market_subscriptions(&sender.borrow(), markets) {
        let _ = sender.send(markets.to_vec());
    }
}

fn same_market_subscription(left: &BtcIntervalMarket, right: &BtcIntervalMarket) -> bool {
    left.market_id == right.market_id
        && left.condition_id == right.condition_id
        && left.up_token_id == right.up_token_id
        && left.down_token_id == right.down_token_id
        && left.window_start == right.window_start
        && left.window_end == right.window_end
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
        }, {
            "topic": "crypto_prices_twap_sixty",
            "type": "update",
            "filters": "{\"symbol\":\"btc/usd\"}"
        }]
    })
    .to_string()
}

fn is_rtds_twap_60_update(value: &serde_json::Value) -> bool {
    value.get("type").and_then(serde_json::Value::as_str) == Some("update")
        && value.get("topic").and_then(serde_json::Value::as_str)
            == Some("crypto_prices_twap_sixty")
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

fn retryable_postgres_primary_persistence_sqlstate(code: &str) -> bool {
    code.starts_with("08")
        || code.starts_with("40")
        || code.starts_with("53")
        || matches!(
            code,
            "55P03" | "57014" | "57P01" | "57P02" | "57P03" | "57P05" | "58030"
        )
}

fn is_retryable_primary_persistence_error(error: &anyhow::Error) -> bool {
    let Some(sqlx_error) = error.downcast_ref::<sqlx::Error>() else {
        return false;
    };
    match sqlx_error {
        sqlx::Error::Database(error) => error
            .code()
            .as_deref()
            .is_some_and(retryable_postgres_primary_persistence_sqlstate),
        sqlx::Error::Io(_) | sqlx::Error::Tls(_) | sqlx::Error::PoolTimedOut => true,
        sqlx::Error::PoolClosed | sqlx::Error::WorkerCrashed => false,
        _ => false,
    }
}

fn primary_persistence_retry_delay(consecutive_failures: u32) -> StdDuration {
    let exponent = consecutive_failures.saturating_sub(1).min(10);
    CRITICAL_WRITE_INITIAL_BACKOFF
        .saturating_mul(2u32.saturating_pow(exponent))
        .min(PRIMARY_PERSISTENCE_RETRY_MAX_DELAY)
}

fn retryable_postgres_boundary_hydration_sqlstate(code: &str) -> bool {
    code.starts_with("08")
        || code.starts_with("40")
        || code.starts_with("53")
        || code.starts_with("58")
        || matches!(code, "55P03" | "57014" | "57P01" | "57P02" | "57P03")
}

fn boundary_hydration_retry_delay(
    discovery_interval: StdDuration,
    consecutive_failures: u32,
) -> StdDuration {
    let initial_delay = discovery_interval.max(StdDuration::from_secs(1));
    let maximum_delay = BOUNDARY_HYDRATION_RETRY_MAX_DELAY.max(initial_delay);
    let exponent = consecutive_failures.saturating_sub(1).min(10);
    initial_delay
        .saturating_mul(2u32.saturating_pow(exponent))
        .min(maximum_delay)
}

fn is_retryable_boundary_hydration_read_error(error: &anyhow::Error) -> bool {
    let Some(sqlx_error) = error.downcast_ref::<sqlx::Error>() else {
        return false;
    };
    match sqlx_error {
        sqlx::Error::Database(error) => error
            .code()
            .as_deref()
            .is_some_and(retryable_postgres_boundary_hydration_sqlstate),
        sqlx::Error::Io(_) | sqlx::Error::Tls(_) | sqlx::Error::PoolTimedOut => true,
        _ => false,
    }
}

#[derive(Debug)]
enum BoundaryHydrationReadFailure {
    Retry {
        error: anyhow::Error,
        entered_degraded_state: bool,
        consecutive_failures: u32,
    },
    Fatal(anyhow::Error),
}

fn classify_boundary_hydration_read_failure(
    error: anyhow::Error,
    metrics: &mut BtcRuntimeMetrics,
) -> BoundaryHydrationReadFailure {
    if !is_retryable_boundary_hydration_read_error(&error) {
        return BoundaryHydrationReadFailure::Fatal(error);
    }
    let entered_degraded_state = mark_boundary_hydration_read_failure(metrics);
    BoundaryHydrationReadFailure::Retry {
        error,
        entered_degraded_state,
        consecutive_failures: metrics.boundary_hydration_consecutive_failures,
    }
}

fn mark_boundary_hydration_read_failure(metrics: &mut BtcRuntimeMetrics) -> bool {
    let entered_degraded_state = metrics.boundary_hydration_consecutive_failures == 0;
    metrics.boundary_hydration_read_errors =
        metrics.boundary_hydration_read_errors.saturating_add(1);
    metrics.boundary_hydration_consecutive_failures = metrics
        .boundary_hydration_consecutive_failures
        .saturating_add(1);
    entered_degraded_state
}

fn mark_boundary_hydration_recovered(metrics: &mut BtcRuntimeMetrics) -> bool {
    let recovered = metrics.boundary_hydration_consecutive_failures > 0;
    metrics.boundary_hydration_consecutive_failures = 0;
    recovered
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

    #[test]
    fn binance_spot_l2_shutdown_events_are_detected_without_masking_depth_updates() {
        assert!(is_binance_spot_l2_server_shutdown(
            r#"{"e":"serverShutdown"}"#
        ));
        assert!(is_binance_spot_l2_server_shutdown(
            r#"{"event":"eventStreamTerminated"}"#
        ));
        assert!(!is_binance_spot_l2_server_shutdown(
            r#"{"e":"depthUpdate","s":"BTCUSDT"}"#
        ));
        assert!(!is_binance_spot_l2_server_shutdown("not-json"));
    }

    #[test]
    fn binance_spot_l2_bootstrap_buffer_fails_closed_without_eviction() {
        let update = BinanceSpotDepthUpdate {
            event_time: Utc.timestamp_opt(1_786_000_000, 0).unwrap(),
            first_update_id: 1,
            final_update_id: 1,
            bids: Vec::new(),
            asks: Vec::new(),
        };
        let mut buffer = BinanceSpotL2BootstrapBuffer::default();
        for _ in 0..BINANCE_SPOT_L2_BOOTSTRAP_EVENT_CAPACITY {
            buffer.push(update.clone(), update.event_time).unwrap();
        }
        assert_eq!(
            buffer.updates.len(),
            BINANCE_SPOT_L2_BOOTSTRAP_EVENT_CAPACITY
        );
        assert!(buffer.push(update.clone(), update.event_time).is_err());
        assert_eq!(
            buffer.updates.len(),
            BINANCE_SPOT_L2_BOOTSTRAP_EVENT_CAPACITY
        );
    }

    #[tokio::test]
    async fn binance_spot_l2_disconnect_clears_only_inference_state() {
        let connection_id = Uuid::new_v4();
        let state = Arc::new(RwLock::new(RealtimeState::default()));
        state
            .write()
            .await
            .binance_spot_l2
            .clear_epoch(connection_id);
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        {
            let mut runtime_metrics = metrics.write().await;
            runtime_metrics.binance_spot_l2.synchronized = true;
            runtime_metrics.binance_spot_l2.active_connection_id = Some(connection_id);
            runtime_metrics.binance_spot_l2.active_last_update_id = Some(10);
        }

        clear_binance_spot_l2_runtime_state(&state, &metrics).await;

        let realtime = state.read().await;
        assert_eq!(realtime.binance_spot_l2.connection_id(), None);
        assert!(!realtime.binance_spot_l2.synchronized());
        drop(realtime);
        let runtime_metrics = metrics.read().await;
        assert!(!runtime_metrics.binance_spot_l2.synchronized);
        assert_eq!(runtime_metrics.binance_spot_l2.active_connection_id, None);
        assert_eq!(runtime_metrics.binance_spot_l2.active_last_update_id, None);
    }

    #[test]
    fn binance_rest_kline_maps_to_live_model_candle_contract() {
        let open = Utc.timestamp_millis_opt(1_783_902_600_000).unwrap();
        let recovered_at = open + Duration::seconds(2);
        let candle = parse_binance_rest_kline(
            &serde_json::json!([
                open.timestamp_millis(),
                "100.00",
                "102.00",
                "99.00",
                "101.00",
                "3.00",
                open.timestamp_millis() + 999,
                "302.00",
                4,
                "2.00",
                "201.00",
                "0"
            ]),
            recovered_at,
        )
        .unwrap();

        assert_eq!(candle.open_timestamp, open);
        assert_eq!(candle.close_timestamp, open + Duration::seconds(1));
        assert_eq!(candle.trade_count, 4);
        assert_eq!(candle.taker_buy_quote_volume, dec!(201));
        assert!(candle.source_complete);
        assert!(!candle.synthetic);
    }

    #[test]
    fn authoritative_bootstrap_bridges_no_trade_seconds_before_live_replay() {
        let open = Utc.timestamp_opt(1_783_902_600, 0).unwrap();
        let recovered_at = open + Duration::seconds(1);
        let candle = parse_binance_rest_kline(
            &serde_json::json!([
                open.timestamp_millis(),
                "100",
                "100",
                "100",
                "100",
                "1",
                open.timestamp_millis() + 999,
                "100",
                1,
                "1",
                "100",
                "0"
            ]),
            recovered_at,
        )
        .unwrap();
        let mut window = BinanceOneSecondWindow::from_completed(vec![candle]).unwrap();
        let trade_at = open + Duration::seconds(3) + Duration::milliseconds(100);
        window
            .update(
                &BinanceAggregateTrade {
                    aggregate_trade_id: 42,
                    price: dec!(101),
                    quantity: dec!(0.5),
                    first_trade_id: 50,
                    last_trade_id: 50,
                    transact_time: trade_at,
                    is_buyer_maker: false,
                },
                trade_at + Duration::milliseconds(10),
            )
            .unwrap();

        assert_eq!(window.completed().len(), 3);
        assert!(window.completed()[1].synthetic);
        assert!(window.completed()[2].synthetic);
        assert!(window
            .completed()
            .iter()
            .all(|candle| candle.source_complete));
        assert!(window
            .current()
            .is_some_and(|candle| candle.source_complete));
    }

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
    ) -> ClobEpoch {
        let connection_id = registry.connection_id();
        let connected_instant = Instant::now();
        let watchdog = ClobFeedWatchdog::new(
            connected_instant,
            &registry,
            std::slice::from_ref(&market),
            checked_at,
        );
        let telemetry = ClobSocketTelemetry::new(std::slice::from_ref(&market));
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
            telemetry,
            connected_instant,
            watchdog,
            healthy_epoch: false,
            books_usable: false,
            pending_resolutions: HashMap::new(),
        }
    }

    #[test]
    fn remote_clob_close_keeps_stable_bounded_details() {
        let frame = CloseFrame {
            code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Away,
            reason: "planned maintenance".into(),
        };
        let mut telemetry = ClobSocketTelemetry::new(std::slice::from_ref(&market()));

        telemetry.record_remote_close(Some(&frame));

        assert!(telemetry.remote_close_observed);
        assert_eq!(telemetry.remote_close_code, Some(1001));
        assert_eq!(
            telemetry.remote_close_reason.as_deref(),
            Some("planned maintenance")
        );
    }

    #[test]
    fn clob_close_actions_are_explicit_and_failure_safe() {
        let mut telemetry = ClobSocketTelemetry::new(std::slice::from_ref(&market()));
        let causes = [
            ClobDisconnectCause::ConnectFailure,
            ClobDisconnectCause::SubscriptionFailure,
            ClobDisconnectCause::BootstrapFailure,
            ClobDisconnectCause::TransportFailure,
            ClobDisconnectCause::ReadinessRefresh,
            ClobDisconnectCause::Shutdown,
            ClobDisconnectCause::MarketWatchClosed,
            ClobDisconnectCause::CriticalPersistence,
        ];

        assert_eq!(clob_failure_close_action(&telemetry), ClobCloseAction::Skip);
        assert_eq!(
            clob_unavailable_recovery_close_action(),
            ClobCloseAction::Skip
        );
        assert_eq!(
            clob_stop_close_action(&telemetry),
            ClobCloseAction::Initiate
        );
        for cause in causes {
            let expected = if cause == ClobDisconnectCause::ReadinessRefresh {
                ClobCloseAction::Initiate
            } else {
                ClobCloseAction::Skip
            };
            assert_eq!(clob_handoff_close_action(cause, &telemetry), expected);
        }

        telemetry.record_remote_close(None);
        assert!(telemetry.remote_close_observed);
        assert_eq!(
            clob_failure_close_action(&telemetry),
            ClobCloseAction::AcknowledgeRemote
        );
        assert_eq!(
            clob_stop_close_action(&telemetry),
            ClobCloseAction::AcknowledgeRemote
        );
        for cause in causes {
            assert_eq!(
                clob_handoff_close_action(cause, &telemetry),
                ClobCloseAction::AcknowledgeRemote
            );
        }
    }

    #[tokio::test]
    async fn planned_clob_close_sends_one_close_frame() {
        let (client_io, server_io) = tokio::io::duplex(1_024);
        let mut client = WebSocketStream::from_raw_socket(
            client_io,
            tokio_tungstenite::tungstenite::protocol::Role::Client,
            None,
        )
        .await;
        let mut server = WebSocketStream::from_raw_socket(
            server_io,
            tokio_tungstenite::tungstenite::protocol::Role::Server,
            None,
        )
        .await;

        assert!(matches!(
            attempt_clob_close(&mut client, ClobCloseAction::Initiate).await,
            ClobCloseOutcome::Completed
        ));
        let frame = timeout(CLOB_GRACEFUL_CLOSE_TIMEOUT, server.next())
            .await
            .expect("planned close frame must arrive within the close bound")
            .expect("planned close must produce a frame")
            .expect("planned close frame must decode");
        assert!(matches!(frame, Message::Close(_)));
    }

    #[tokio::test]
    async fn remote_clob_close_is_acknowledged_by_flushing_the_queued_reply() {
        let (client_io, server_io) = tokio::io::duplex(1_024);
        let mut client = WebSocketStream::from_raw_socket(
            client_io,
            tokio_tungstenite::tungstenite::protocol::Role::Client,
            None,
        )
        .await;
        let mut server = WebSocketStream::from_raw_socket(
            server_io,
            tokio_tungstenite::tungstenite::protocol::Role::Server,
            None,
        )
        .await;
        server.send(Message::Close(None)).await.unwrap();
        assert!(matches!(
            client.next().await.unwrap().unwrap(),
            Message::Close(None)
        ));

        assert!(matches!(
            attempt_clob_close(&mut client, ClobCloseAction::AcknowledgeRemote).await,
            ClobCloseOutcome::Completed
        ));
        let acknowledgement = timeout(CLOB_GRACEFUL_CLOSE_TIMEOUT, server.next())
            .await
            .expect("remote close acknowledgement must arrive within the close bound")
            .expect("remote close acknowledgement must produce a frame")
            .expect("remote close acknowledgement must decode");
        assert!(matches!(acknowledgement, Message::Close(_)));
    }

    #[tokio::test]
    async fn clob_close_action_respects_the_hard_timeout() {
        use tokio::io::AsyncWriteExt;

        let (client_io, blocked_peer) = tokio::io::duplex(1);
        let mut client = WebSocketStream::from_raw_socket(
            client_io,
            tokio_tungstenite::tungstenite::protocol::Role::Client,
            None,
        )
        .await;
        client.get_mut().write_all(&[0]).await.unwrap();

        let started_at = Instant::now();
        assert!(matches!(
            attempt_clob_close(&mut client, ClobCloseAction::Initiate).await,
            ClobCloseOutcome::TimedOut
        ));
        assert!(started_at.elapsed() < StdDuration::from_millis(500));
        drop(blocked_peer);
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
    fn successor_retry_respects_structure_and_active_readiness() {
        let mut config = BtcRuntimeConfig::default();
        let mut failures = 7;
        assert_eq!(
            clob_candidate_retry_action(&config, true, true, true, &mut failures),
            ClobRetryAction::ImmediateRecovery
        );
        assert_eq!(failures, 0);

        assert!(matches!(
            clob_candidate_retry_action(&config, false, true, true, &mut failures),
            ClobRetryAction::Backoff(delay) if delay == config.reconnect_initial_delay
        ));
        assert_eq!(failures, 1);

        config.reconnect_initial_delay = StdDuration::from_secs(5);
        failures = 7;
        assert_eq!(
            clob_candidate_retry_action(&config, false, false, true, &mut failures),
            ClobRetryAction::Backoff(StdDuration::from_secs(1))
        );
        assert_eq!(failures, 0);

        failures = 7;
        assert_eq!(
            clob_candidate_retry_action(&config, false, false, false, &mut failures),
            ClobRetryAction::Backoff(config.reconnect_max_delay)
        );
        assert_eq!(failures, 8);

        let now = Instant::now();
        let retry_at = now + StdDuration::from_secs(30);
        failures = 9;
        assert_eq!(
            expedite_clob_candidate_retry(&config, false, true, now, retry_at, &mut failures),
            now + StdDuration::from_secs(1)
        );
        assert_eq!(failures, 0);

        failures = 4;
        assert_eq!(
            expedite_clob_candidate_retry(&config, true, true, now, retry_at, &mut failures),
            retry_at
        );
        assert_eq!(failures, 4);

        assert_eq!(
            expedite_clob_candidate_retry(&config, false, false, now, retry_at, &mut failures),
            retry_at
        );
        assert_eq!(failures, 4);
    }

    #[test]
    fn unchanged_book_remains_ready_on_structural_connection() {
        let current = market();
        let ready_at = current.window_start + Duration::minutes(1);
        let max_book_age = Duration::seconds(2);
        let registry = ready_book_registry(&current, ready_at - Duration::milliseconds(1));
        let markets = std::slice::from_ref(&current);

        assert!(clob_epoch_ready(&registry, markets, ready_at, max_book_age,));
        assert!(clob_epoch_structurally_ready(&registry, markets, ready_at));

        let stale_at = ready_at + max_book_age + Duration::milliseconds(1);
        assert!(clob_epoch_ready(&registry, markets, stale_at, max_book_age,));
        assert!(clob_epoch_structurally_ready(&registry, markets, stale_at));
    }

    #[test]
    fn clob_unavailability_diagnostic_names_the_affected_book_state() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        let mut registry = ready_book_registry(&current, checked_at - Duration::milliseconds(1));

        assert!(clob_epoch_readiness_diagnostic(
            &registry,
            std::slice::from_ref(&current),
            checked_at,
            Duration::seconds(2),
        )
        .is_none());

        registry.quarantine(FeedIntegrityStatus::TopOfBookMismatch);
        let diagnostic = clob_epoch_readiness_diagnostic(
            &registry,
            std::slice::from_ref(&current),
            checked_at,
            Duration::seconds(2),
        )
        .expect("quarantined book must expose a causal diagnostic");
        assert_eq!(diagnostic.reason, "book_integrity");
        assert_eq!(
            diagnostic.market_id.as_deref(),
            Some(current.market_id.as_str())
        );
        assert!(matches!(
            diagnostic.token_id.as_deref(),
            Some(token_id)
                if token_id == current.up_token_id.as_str()
                    || token_id == current.down_token_id.as_str()
        ));
        assert_eq!(
            diagnostic.integrity_status,
            Some(FeedIntegrityStatus::TopOfBookMismatch)
        );
        assert_eq!(diagnostic.bootstrapped, Some(true));
        assert_eq!(diagnostic.has_bid, Some(true));
        assert_eq!(diagnostic.has_ask, Some(true));
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

    #[test]
    fn readiness_refresh_is_not_counted_as_transport_failure() {
        let mut metrics = BtcRuntimeMetrics::default();
        clob_disconnect_metrics(
            &mut metrics,
            ClobDisconnectCause::ReadinessRefresh,
            ClobRetryAction::ImmediateRecovery,
        );

        assert_eq!(metrics.reconnects, 1);
        assert_eq!(metrics.clob_transport_disconnects, 0);
        assert_eq!(metrics.clob_bootstrap_failures, 0);
        assert_eq!(metrics.clob_immediate_recoveries_scheduled, 1);
    }

    #[tokio::test]
    async fn private_successor_frames_remain_private() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        let mut candidate_registry = BookRegistry::new(Uuid::new_v4());
        candidate_registry.register_market(&current);
        let mut candidate =
            private_clob_epoch(current.clone(), candidate_registry, checked_at).await;
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
            ClobFrameAction::Continue
        );
        assert!(public_registry.checkpoint(&current.up_token_id).is_none());
        assert_eq!(public_metrics.decode_errors, 0);
        assert!(matches!(
            reader.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn private_successor_waits_for_snapshot_and_ignores_only_foreign_events() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&current);
        let mut candidate = private_clob_epoch(current.clone(), registry, checked_at).await;
        let frame = serde_json::json!([{
            "event_type": "best_bid_ask",
            "market": current.condition_id.clone(),
            "asset_id": current.up_token_id.clone(),
            "best_bid": ".48",
            "best_ask": ".52",
            "timestamp": checked_at.timestamp_millis()
        }, {
            "event_type": "last_trade_price",
            "market": current.condition_id.clone(),
            "asset_id": current.up_token_id.clone(),
            "price": ".50",
            "size": "2",
            "timestamp": checked_at.timestamp_millis()
        }, {
            "event_type": "price_change",
            "market": current.condition_id.clone(),
            "timestamp": checked_at.timestamp_millis(),
            "price_changes": [{
                "asset_id": current.up_token_id.clone(),
                "price": ".49",
                "size": "20",
                "side": "BUY",
                "best_bid": ".49",
                "best_ask": ".52"
            }]
        }, {
            "event_type": "book",
            "market": current.condition_id.clone(),
            "asset_id": current.up_token_id.clone(),
            "timestamp": checked_at.timestamp_millis(),
            "bids": [{"price": ".48", "size": "10"}],
            "asks": [{"price": ".52", "size": "10"}]
        }]);

        assert_eq!(
            apply_private_clob_frame(&mut candidate, Message::Text(frame.to_string().into())),
            ClobFrameAction::Continue
        );
        assert_eq!(candidate.session.integrity_gaps, 0);
        assert!(candidate
            .registry
            .checkpoint(&current.up_token_id)
            .is_some());
        assert!(candidate
            .registry
            .checkpoint(&current.down_token_id)
            .is_none());
        assert!(candidate.watchdog.bootstrap_deadline.is_some());

        let foreign = serde_json::json!({
            "event_type": "last_trade_price",
            "market": "retired-market",
            "asset_id": "retired-token",
            "price": ".50",
            "size": "1",
            "timestamp": checked_at.timestamp_millis()
        });
        assert_eq!(
            apply_private_clob_frame(&mut candidate, Message::Text(foreign.to_string().into()),),
            ClobFrameAction::Continue
        );
        assert_eq!(candidate.subscription_stats.ignored_foreign_events, 1);
        assert_eq!(candidate.session.integrity_gaps, 0);

        let ambiguous = serde_json::json!({
            "event_type": "last_trade_price",
            "market": current.condition_id,
            "asset_id": "unexpected-token",
            "price": ".50",
            "size": "1",
            "timestamp": checked_at.timestamp_millis()
        });
        assert_eq!(
            apply_private_clob_frame(&mut candidate, Message::Text(ambiguous.to_string().into()),),
            ClobFrameAction::Continue
        );
        assert_eq!(candidate.session.integrity_gaps, 1);
        assert!(candidate.session.disconnect_reason.is_none());
    }

    #[test]
    fn successor_integrity_diagnostic_matches_token_and_market_independently() {
        let current = market();
        let mut next = current.clone();
        next.market_id = "next-market".to_string();
        next.condition_id = "next-condition".to_string();
        next.up_token_id = "next-up".to_string();
        next.down_token_id = "next-down".to_string();
        let received_at = current.window_start + Duration::minutes(1);
        let event = |market_id: &str, token_id: &str| MarketFeedEvent {
            event_id: Uuid::new_v4(),
            market_id: market_id.to_string(),
            token_id: Some(token_id.to_string()),
            event_type: MarketFeedEventType::LastTradePrice,
            source_timestamp: received_at,
            received_at,
            connection_id: Uuid::new_v4(),
            ingest_sequence: 1,
            source_hash: None,
            applied: false,
            integrity_status: FeedIntegrityStatus::UnknownToken,
            raw_payload: serde_json::Value::Null,
        };
        let markets = std::slice::from_ref(&current);

        assert_eq!(
            private_clob_subscription_match(
                markets,
                &event(&current.condition_id, &current.up_token_id),
            ),
            PrivateClobSubscriptionMatch {
                token: true,
                market: true,
                exact_identity: true,
            }
        );
        assert_eq!(
            private_clob_subscription_match(
                markets,
                &event(&current.condition_id, "unexpected-token"),
            ),
            PrivateClobSubscriptionMatch {
                token: false,
                market: true,
                exact_identity: false,
            }
        );
        assert_eq!(
            private_clob_subscription_match(
                markets,
                &event("unexpected-market", &current.down_token_id),
            ),
            PrivateClobSubscriptionMatch {
                token: true,
                market: false,
                exact_identity: false,
            }
        );
        assert_eq!(
            private_clob_subscription_match(
                &[current.clone(), next.clone()],
                &event(&current.condition_id, &next.up_token_id),
            ),
            PrivateClobSubscriptionMatch {
                token: true,
                market: true,
                exact_identity: false,
            }
        );
    }

    #[tokio::test]
    async fn private_successor_ignores_only_exact_superseded_price_changes() {
        let current = market();
        let snapshot_at = current.window_start + Duration::minutes(1);
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&current);
        let mut candidate = private_clob_epoch(current.clone(), registry, snapshot_at).await;
        let snapshots = serde_json::json!([{
            "event_type": "book",
            "market": current.condition_id.clone(),
            "asset_id": current.up_token_id.clone(),
            "timestamp": snapshot_at.timestamp_millis(),
            "hash": "up-snapshot",
            "bids": [{"price": ".48", "size": "10"}],
            "asks": [{"price": ".52", "size": "10"}]
        }, {
            "event_type": "book",
            "market": current.condition_id.clone(),
            "asset_id": current.down_token_id.clone(),
            "timestamp": snapshot_at.timestamp_millis(),
            "hash": "down-snapshot",
            "bids": [{"price": ".48", "size": "10"}],
            "asks": [{"price": ".52", "size": "10"}]
        }]);
        assert_eq!(
            apply_private_clob_frame(&mut candidate, Message::Text(snapshots.to_string().into()),),
            ClobFrameAction::Continue
        );

        let superseded_at = snapshot_at - Duration::milliseconds(2);
        let superseded = serde_json::json!({
            "event_type": "price_change",
            "market": current.condition_id.clone(),
            "timestamp": superseded_at.timestamp_millis(),
            "price_changes": [{
                "asset_id": current.up_token_id.clone(),
                "price": ".48",
                "size": "999",
                "side": "BUY",
                "best_bid": ".48",
                "best_ask": ".52"
            }, {
                "asset_id": current.down_token_id.clone(),
                "price": ".52",
                "size": "999",
                "side": "SELL",
                "best_bid": ".48",
                "best_ask": ".52"
            }]
        });
        assert_eq!(
            apply_private_clob_frame(&mut candidate, Message::Text(superseded.to_string().into()),),
            ClobFrameAction::Continue
        );
        assert_eq!(candidate.subscription_stats.ignored_superseded_events, 2);
        assert_eq!(candidate.session.integrity_gaps, 0);
        assert!(candidate.session.disconnect_reason.is_none());
        for token_id in [&current.up_token_id, &current.down_token_id] {
            let checkpoint = candidate
                .registry
                .checkpoint(token_id)
                .expect("newer authoritative snapshot remains available");
            assert_eq!(checkpoint.source_timestamp, snapshot_at);
            assert_eq!(checkpoint.bids[0].size, dec!(10));
            assert_eq!(checkpoint.asks[0].size, dec!(10));
        }

        let ambiguous = MarketFeedEvent {
            event_id: Uuid::new_v4(),
            market_id: current.market_id.clone(),
            token_id: Some("unexpected-token".to_string()),
            event_type: MarketFeedEventType::PriceChange,
            source_timestamp: superseded_at,
            received_at: snapshot_at,
            connection_id: Uuid::new_v4(),
            ingest_sequence: 1,
            source_hash: None,
            applied: false,
            integrity_status: FeedIntegrityStatus::OutOfOrder,
            raw_payload: serde_json::json!({}),
        };
        assert_eq!(
            private_clob_event_disposition(std::slice::from_ref(&current), &ambiguous),
            PrivateClobEventDisposition::Reject
        );
    }

    #[tokio::test]
    async fn private_successor_preserves_pre_snapshot_tick_and_rejects_market_mismatch() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&current);
        let mut tick_candidate = private_clob_epoch(current.clone(), registry, checked_at).await;
        let tick = serde_json::json!({
            "event_type": "tick_size_change",
            "market": current.condition_id.clone(),
            "asset_id": current.up_token_id.clone(),
            "old_tick_size": ".01",
            "new_tick_size": ".001",
            "timestamp": checked_at.timestamp_millis()
        });
        assert_eq!(
            apply_private_clob_frame(&mut tick_candidate, Message::Text(tick.to_string().into()),),
            ClobFrameAction::Continue
        );
        assert_eq!(tick_candidate.session.integrity_gaps, 0);
        assert!(tick_candidate
            .registry
            .checkpoint(&current.up_token_id)
            .is_none());
        let snapshot = serde_json::json!({
            "event_type": "book",
            "market": current.condition_id.clone(),
            "asset_id": current.up_token_id.clone(),
            "timestamp": (checked_at + Duration::milliseconds(1)).timestamp_millis(),
            "bids": [{"price": ".48", "size": "10"}],
            "asks": [{"price": ".52", "size": "10"}]
        });
        assert_eq!(
            apply_private_clob_frame(
                &mut tick_candidate,
                Message::Text(snapshot.to_string().into()),
            ),
            ClobFrameAction::Continue
        );
        assert_eq!(
            tick_candidate
                .registry
                .checkpoint(&current.up_token_id)
                .expect("snapshot should preserve pre-snapshot tick metadata")
                .tick_size,
            dec!(0.001)
        );

        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&current);
        let mut mismatch_candidate =
            private_clob_epoch(current.clone(), registry, checked_at).await;
        let mismatched_book = serde_json::json!({
            "event_type": "book",
            "market": "wrong-market",
            "asset_id": current.up_token_id,
            "timestamp": checked_at.timestamp_millis(),
            "bids": [{"price": ".48", "size": "10"}],
            "asks": [{"price": ".52", "size": "10"}]
        });
        assert_eq!(
            apply_private_clob_frame(
                &mut mismatch_candidate,
                Message::Text(mismatched_book.to_string().into()),
            ),
            ClobFrameAction::Continue
        );
        assert_eq!(mismatch_candidate.session.integrity_gaps, 1);
    }

    #[test]
    fn private_successor_rejects_applied_non_ok_events() {
        let current = market();
        let observed_at = current.window_start + Duration::minutes(1);
        let event = MarketFeedEvent {
            event_id: Uuid::new_v4(),
            market_id: current.market_id.clone(),
            token_id: Some(current.up_token_id.clone()),
            event_type: MarketFeedEventType::LastTradePrice,
            source_timestamp: observed_at,
            received_at: observed_at,
            connection_id: Uuid::new_v4(),
            ingest_sequence: 1,
            source_hash: None,
            applied: true,
            integrity_status: FeedIntegrityStatus::Stale,
            raw_payload: serde_json::json!({}),
        };
        assert_eq!(
            private_clob_event_disposition(std::slice::from_ref(&current), &event),
            PrivateClobEventDisposition::Reject
        );
    }

    #[tokio::test]
    async fn conflicting_private_resolution_is_quarantined_without_overwriting_first_fact() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&current);
        let mut candidate = private_clob_epoch(current.clone(), registry, checked_at).await;
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
            ClobFrameAction::Continue
        );
        assert!(candidate.session.disconnect_reason.is_none());
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
        let mut candidate =
            private_clob_epoch(current.clone(), registry, publication_boundary).await;
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
        .is_some());

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
        let mut watchdog = ClobFeedWatchdog::new(now, &registry, &[], checked_at);
        let initial_read_deadline = now + CLOB_READ_IDLE_TIMEOUT;
        assert_eq!(watchdog.read_idle_deadline, initial_read_deadline);
        assert_eq!(watchdog.pong_deadline, None);

        let ping_at = now + StdDuration::from_secs(1);
        let pong_timeout = StdDuration::from_secs(25);
        watchdog.record_text_ping(ping_at, pong_timeout);
        let pong_deadline = ping_at + pong_timeout;
        assert_eq!(watchdog.read_idle_deadline, initial_read_deadline);
        assert_eq!(watchdog.pong_deadline, Some(pong_deadline));

        let frame_at = now + StdDuration::from_secs(2);
        watchdog.on_frame(frame_at);
        assert_eq!(
            watchdog.read_idle_deadline,
            frame_at + CLOB_READ_IDLE_TIMEOUT
        );
        assert_eq!(watchdog.pong_deadline, None);
        assert_eq!(watchdog.pending_pong_probe_sent_at, Some(ping_at));
        let pong_at = frame_at + StdDuration::from_millis(7);
        assert_eq!(
            watchdog.acknowledge_text_pong(" pong \n", pong_at),
            Some(pong_at.saturating_duration_since(ping_at))
        );
        assert_eq!(watchdog.pending_pong_probe_sent_at, None);

        // A later probe starts a fresh liveness and correlation window.
        let later_ping_at = ping_at + StdDuration::from_secs(10);
        watchdog.record_text_ping(later_ping_at, pong_timeout);
        assert_eq!(watchdog.pong_deadline, Some(later_ping_at + pong_timeout));
        assert_eq!(
            watchdog.read_idle_deadline,
            frame_at + CLOB_READ_IDLE_TIMEOUT
        );
    }

    #[test]
    fn clob_socket_telemetry_tracks_heartbeat_timing_without_history() {
        let wall_clock = Utc.timestamp_opt(1_784_736_000, 0).unwrap();
        let sent_instant = Instant::now();
        let acknowledged_instant = sent_instant + StdDuration::from_millis(37);
        let mut telemetry = ClobSocketTelemetry::new(std::slice::from_ref(&market()));

        let probe = telemetry.record_heartbeat_probe(
            wall_clock,
            sent_instant,
            StdDuration::from_millis(12),
        );
        let acknowledgement = telemetry.record_heartbeat_acknowledgement(
            wall_clock + Duration::milliseconds(37),
            acknowledged_instant,
            StdDuration::from_millis(37),
        );
        telemetry.record_frame(
            wall_clock + Duration::milliseconds(37),
            acknowledged_instant,
        );

        assert_eq!(telemetry.role, ClobConnectionRole::Successor);
        assert_eq!(probe.scheduling_lateness, StdDuration::from_millis(12));
        assert_eq!(
            acknowledgement.round_trip,
            Some(StdDuration::from_millis(37))
        );
        assert_eq!(telemetry.heartbeat_probes, 1);
        assert_eq!(telemetry.heartbeat_acknowledgements, 1);
        assert_eq!(
            telemetry.max_heartbeat_send_lateness,
            StdDuration::from_millis(12)
        );
        assert_eq!(
            telemetry.last_pong_round_trip,
            Some(StdDuration::from_millis(37))
        );
        assert_eq!(
            telemetry.last_frame_at,
            Some(acknowledgement.acknowledged_at)
        );

        telemetry.mark_active();
        assert_eq!(telemetry.role, ClobConnectionRole::Active);
    }

    #[test]
    fn runtime_metrics_snapshot_reports_current_clob_inbound_age() {
        let checked_at = Utc.timestamp_opt(1_784_736_010, 0).unwrap();
        let metrics = BtcRuntimeMetrics {
            clob_active_last_data_or_heartbeat_at: Some(checked_at - Duration::milliseconds(1250)),
            ..BtcRuntimeMetrics::default()
        };

        let snapshot = runtime_metrics_snapshot(metrics, checked_at);

        assert_eq!(
            snapshot.clob_active_last_inbound_frame_age_milliseconds,
            Some(1250)
        );
    }

    #[test]
    fn clob_watchdog_bootstrap_requires_both_current_books() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        let source_at = checked_at - Duration::milliseconds(1);
        let now = Instant::now();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&current);
        let mut watchdog =
            ClobFeedWatchdog::new(now, &registry, std::slice::from_ref(&current), checked_at);
        let bootstrap_deadline = now + CLOB_BOOTSTRAP_TIMEOUT;
        assert_eq!(watchdog.bootstrap_deadline, Some(bootstrap_deadline));

        apply_ready_book_snapshot(&mut registry, &current, &current.up_token_id, source_at);
        watchdog.refresh_bootstrap(
            now + StdDuration::from_millis(1),
            &registry,
            std::slice::from_ref(&current),
            checked_at,
        );
        assert_eq!(watchdog.bootstrap_deadline, Some(bootstrap_deadline));

        apply_ready_book_snapshot(&mut registry, &current, &current.down_token_id, source_at);
        watchdog.refresh_bootstrap(
            now + StdDuration::from_millis(2),
            &registry,
            std::slice::from_ref(&current),
            checked_at,
        );
        assert_eq!(watchdog.bootstrap_deadline, None);
    }

    #[tokio::test]
    async fn unchanged_book_does_not_rearm_structural_bootstrap() {
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
        );
        assert_eq!(watchdog.bootstrap_deadline, None);

        let telemetry = ClobSocketTelemetry::new(std::slice::from_ref(&current));

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
            telemetry,
            connected_instant: ready_instant,
            watchdog,
            healthy_epoch: true,
            books_usable: true,
            pending_resolutions: HashMap::new(),
        };
        let stale_at = ready_at + max_book_age + Duration::milliseconds(1);
        let stale_instant = ready_instant + StdDuration::from_millis(2_001);

        epoch.refresh_private_health(stale_at, stale_instant, max_book_age);

        assert!(epoch.books_usable);
        assert!(epoch.healthy_epoch);
        assert_eq!(epoch.watchdog.bootstrap_deadline, None);

        epoch.refresh_private_health(
            stale_at + Duration::milliseconds(100),
            stale_instant + StdDuration::from_millis(100),
            max_book_age,
        );
        assert_eq!(epoch.watchdog.bootstrap_deadline, None);

        epoch.registry.quarantine(FeedIntegrityStatus::Stale);
        let structural_failure_at = stale_instant + StdDuration::from_millis(200);
        epoch.refresh_private_health(
            stale_at + Duration::milliseconds(200),
            structural_failure_at,
            max_book_age,
        );
        assert_eq!(epoch.watchdog.bootstrap_deadline, None);
        epoch.refresh_private_health(
            stale_at + Duration::milliseconds(300),
            structural_failure_at + StdDuration::from_millis(100),
            max_book_age,
        );
        assert_eq!(epoch.watchdog.bootstrap_deadline, None);
    }

    #[test]
    fn one_sided_snapshot_pair_is_structurally_ready_without_reconnecting() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        let now = Instant::now();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&current);
        let mut watchdog =
            ClobFeedWatchdog::new(now, &registry, std::slice::from_ref(&current), checked_at);
        let initial_deadline = now + CLOB_BOOTSTRAP_TIMEOUT;
        assert_eq!(watchdog.bootstrap_deadline, Some(initial_deadline));

        for (index, (token_id, bids, asks)) in [
            (
                &current.up_token_id,
                vec![OrderbookLevel {
                    price: dec!(0.99),
                    size: dec!(10),
                }],
                Vec::new(),
            ),
            (
                &current.down_token_id,
                Vec::new(),
                vec![OrderbookLevel {
                    price: dec!(0.01),
                    size: dec!(10),
                }],
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let events = registry.apply(
                ClobMessage::Book {
                    market_id: current.condition_id.clone(),
                    token_id: token_id.clone(),
                    bids,
                    asks,
                    source_timestamp: checked_at,
                    source_hash: Some(format!("one-sided-{token_id}")),
                    raw_payload: serde_json::json!({}),
                },
                checked_at + Duration::milliseconds(1),
            );
            assert!(events.iter().all(|event| event.applied));
            watchdog.refresh_bootstrap(
                now + StdDuration::from_millis(u64::try_from(index + 1).unwrap()),
                &registry,
                std::slice::from_ref(&current),
                checked_at,
            );
            if index == 0 {
                assert_eq!(watchdog.bootstrap_deadline, Some(initial_deadline));
            }
        }
        let readiness_checked_at = checked_at + Duration::milliseconds(2);
        assert!(registry.market_books_bootstrapped(&current));
        assert!(registry.market_books_structurally_ready(&current));
        assert!(registry.market_books_ready(&current, readiness_checked_at, Duration::seconds(2)));
        assert!(clob_epoch_ready(
            &registry,
            std::slice::from_ref(&current),
            readiness_checked_at,
            Duration::seconds(2),
        ));
        assert_eq!(watchdog.bootstrap_deadline, None);

        let mut replacement = current.clone();
        replacement.market_id = "replacement-market".to_string();
        replacement.condition_id = "replacement-condition".to_string();
        replacement.up_token_id = "replacement-up".to_string();
        replacement.down_token_id = "replacement-down".to_string();
        watchdog.refresh_bootstrap(
            now + StdDuration::from_millis(3),
            &registry,
            std::slice::from_ref(&replacement),
            checked_at,
        );
        assert_eq!(
            watchdog.bootstrap_deadline,
            Some(now + StdDuration::from_millis(3) + CLOB_BOOTSTRAP_TIMEOUT)
        );
    }

    #[test]
    fn clob_watchdog_omits_bootstrap_deadline_without_unique_current_market() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);
        let now = Instant::now();
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&current);
        let mut watchdog =
            ClobFeedWatchdog::new(now, &registry, std::slice::from_ref(&current), checked_at);
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
        );
        assert_eq!(watchdog.bootstrap_market, None);
        assert_eq!(watchdog.bootstrap_deadline, None);

        watchdog.refresh_bootstrap(
            now + StdDuration::from_millis(2),
            &registry,
            &[],
            checked_at,
        );
        assert_eq!(watchdog.bootstrap_market, None);
        assert_eq!(watchdog.bootstrap_deadline, None);
    }

    #[test]
    fn only_current_structural_books_satisfy_bootstrap() {
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
        );
        assert_eq!(watchdog.bootstrap_deadline, None);
        assert!(registry.market_books_structurally_ready(&active_markets[1]));
        assert!(clob_epoch_ready(
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
        assert!(!config.binance_spot_l2_enabled);
        assert_eq!(
            config.binance_spot_l2_ws_url,
            "wss://stream.binance.com/ws/btcusdt@depth@100ms"
        );
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

    #[test]
    fn boundary_hydration_retry_classification_is_limited_to_availability_failures() {
        for code in [
            "08006", "40001", "40P01", "53200", "53300", "55P03", "57014", "57P01", "57P02",
            "57P03", "58030",
        ] {
            assert!(
                retryable_postgres_boundary_hydration_sqlstate(code),
                "{code} must remain retryable"
            );
        }
        for code in ["22000", "23505", "42601", "42P01"] {
            assert!(
                !retryable_postgres_boundary_hydration_sqlstate(code),
                "{code} must remain fatal"
            );
        }

        let retryable = anyhow::Error::new(sqlx::Error::PoolTimedOut)
            .context("failed to load durable BTC closing references");
        assert!(is_retryable_boundary_hydration_read_error(&retryable));

        let fatal = anyhow::Error::new(sqlx::Error::ColumnNotFound("source".to_string()))
            .context("failed to decode durable BTC closing references");
        assert!(!is_retryable_boundary_hydration_read_error(&fatal));
        assert!(!is_retryable_boundary_hydration_read_error(
            &anyhow::Error::new(sqlx::Error::PoolClosed)
        ));
        assert!(!is_retryable_boundary_hydration_read_error(
            &anyhow::Error::new(sqlx::Error::WorkerCrashed)
        ));
        assert!(!is_retryable_boundary_hydration_read_error(
            &anyhow::anyhow!("durable BTC boundary identity conflict")
        ));
    }

    #[test]
    fn primary_persistence_retry_classification_is_limited_to_availability_failures() {
        for code in [
            "08006", "40001", "40P01", "53200", "53300", "55P03", "57014", "57P01", "57P02",
            "57P03", "57P05", "58030",
        ] {
            assert!(
                retryable_postgres_primary_persistence_sqlstate(code),
                "{code} must remain retryable"
            );
        }
        for code in ["22000", "23505", "42601", "42P01", "58000", "58P01"] {
            assert!(
                !retryable_postgres_primary_persistence_sqlstate(code),
                "{code} must remain fatal"
            );
        }

        let retryable = anyhow::Error::new(sqlx::Error::PoolTimedOut)
            .context("failed to persist BTC reference tick");
        assert!(is_retryable_primary_persistence_error(&retryable));

        for fatal in [
            sqlx::Error::PoolClosed,
            sqlx::Error::WorkerCrashed,
            sqlx::Error::ColumnNotFound("source".to_string()),
        ] {
            assert!(!is_retryable_primary_persistence_error(
                &anyhow::Error::new(fatal)
            ));
        }
        assert!(!is_retryable_primary_persistence_error(&anyhow::anyhow!(
            "primary persistence identity conflict"
        )));
    }

    #[test]
    fn primary_persistence_retry_delay_is_bounded_and_exponential() {
        assert_eq!(
            primary_persistence_retry_delay(1),
            StdDuration::from_millis(25)
        );
        assert_eq!(
            primary_persistence_retry_delay(2),
            StdDuration::from_millis(50)
        );
        assert_eq!(
            primary_persistence_retry_delay(6),
            StdDuration::from_millis(800)
        );
        assert_eq!(
            primary_persistence_retry_delay(7),
            PRIMARY_PERSISTENCE_RETRY_MAX_DELAY
        );
        assert_eq!(
            primary_persistence_retry_delay(u32::MAX),
            PRIMARY_PERSISTENCE_RETRY_MAX_DELAY
        );
    }

    #[tokio::test]
    async fn primary_persistence_retries_the_same_item_and_recovers() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state = Arc::new(RwLock::new(RealtimeState::default()));
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let (_shutdown_sender, mut shutdown) = watch::channel(false);

        let outcome = tokio::time::timeout(
            StdDuration::from_secs(1),
            persist_primary_item_with_retry(
                || {
                    let attempts = attempts.clone();
                    async move {
                        if attempts.fetch_add(1, Ordering::Relaxed) < 2 {
                            Err(anyhow::Error::new(sqlx::Error::PoolTimedOut))
                        } else {
                            Ok(())
                        }
                    }
                },
                "reference tick",
                &state,
                &metrics,
                &mut shutdown,
            ),
        )
        .await
        .expect("retry recovery must remain bounded");

        assert_eq!(outcome, PrimaryPersistenceOutcome::Persisted);
        assert_eq!(attempts.load(Ordering::Relaxed), 3);
        let status = metrics.read().await;
        assert_eq!(status.primary_persistence_retryable_errors, 2);
        assert_eq!(status.primary_persistence_consecutive_failures, 0);
        assert_eq!(status.primary_persistence_recoveries, 1);
        assert_eq!(status.persistence_items_written, 1);
        assert_eq!(status.persistence_errors, 0);
        drop(status);
        assert!(state.read().await.primary_persistence_available());
    }

    #[tokio::test]
    async fn primary_persistence_permanent_error_fails_closed_without_retry() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state = Arc::new(RwLock::new(RealtimeState::default()));
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let (_shutdown_sender, mut shutdown) = watch::channel(false);

        let outcome = persist_primary_item_with_retry(
            || {
                attempts.fetch_add(1, Ordering::Relaxed);
                std::future::ready(Err(anyhow::Error::new(sqlx::Error::ColumnNotFound(
                    "source".to_string(),
                ))))
            },
            "reference tick",
            &state,
            &metrics,
            &mut shutdown,
        )
        .await;

        assert_eq!(outcome, PrimaryPersistenceOutcome::Fatal);
        assert_eq!(attempts.load(Ordering::Relaxed), 1);
        let status = metrics.read().await;
        assert_eq!(status.persistence_errors, 1);
        assert_eq!(status.primary_persistence_retryable_errors, 0);
        drop(status);
        assert!(!state.read().await.primary_persistence_available());
    }

    #[tokio::test]
    async fn primary_persistence_shutdown_interrupts_retry_backoff() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state = Arc::new(RwLock::new(RealtimeState::default()));
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let (shutdown_sender, mut shutdown) = watch::channel(false);

        let outcome = tokio::time::timeout(StdDuration::from_secs(1), async {
            tokio::select! {
                outcome = persist_primary_item_with_retry(
                    || {
                        attempts.fetch_add(1, Ordering::Relaxed);
                        std::future::ready(Err(anyhow::Error::new(sqlx::Error::PoolTimedOut)))
                    },
                    "reference tick",
                    &state,
                    &metrics,
                    &mut shutdown,
                ) => outcome,
                _ = async {
                    while attempts.load(Ordering::Relaxed) == 0 {
                        tokio::task::yield_now().await;
                    }
                    shutdown_sender.send(true).unwrap();
                    std::future::pending::<()>().await;
                } => unreachable!("shutdown sender branch never completes"),
            }
        })
        .await
        .expect("shutdown must interrupt primary persistence retry backoff");

        assert_eq!(outcome, PrimaryPersistenceOutcome::Shutdown);
        assert_eq!(attempts.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.read().await.primary_persistence_retryable_errors, 1);
        assert_eq!(metrics.read().await.persistence_items_written, 0);
    }

    #[tokio::test]
    async fn primary_persistence_shutdown_does_not_skip_a_healthy_queued_write() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state = Arc::new(RwLock::new(RealtimeState::default()));
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let (_shutdown_sender, mut shutdown) = watch::channel(true);

        let outcome = persist_primary_item_with_retry(
            || {
                attempts.fetch_add(1, Ordering::Relaxed);
                std::future::ready(Ok(()))
            },
            "reference tick",
            &state,
            &metrics,
            &mut shutdown,
        )
        .await;

        assert_eq!(outcome, PrimaryPersistenceOutcome::Persisted);
        assert_eq!(attempts.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.read().await.persistence_items_written, 1);
    }

    #[tokio::test]
    async fn primary_persistence_shutdown_does_not_cancel_an_inflight_retry() {
        let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let retry_started = Arc::new(tokio::sync::Notify::new());
        let release_retry = Arc::new(tokio::sync::Notify::new());
        let state = Arc::new(RwLock::new(RealtimeState::default()));
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let (shutdown_sender, mut shutdown) = watch::channel(false);

        let outcome = tokio::time::timeout(StdDuration::from_secs(1), async {
            tokio::select! {
                outcome = persist_primary_item_with_retry(
                    || {
                        let attempt = attempts.fetch_add(1, Ordering::Relaxed);
                        let retry_started = retry_started.clone();
                        let release_retry = release_retry.clone();
                        async move {
                            if attempt == 0 {
                                Err(anyhow::Error::new(sqlx::Error::PoolTimedOut))
                            } else {
                                retry_started.notify_one();
                                release_retry.notified().await;
                                Ok(())
                            }
                        }
                    },
                    "reference tick",
                    &state,
                    &metrics,
                    &mut shutdown,
                ) => outcome,
                _ = async {
                    retry_started.notified().await;
                    shutdown_sender.send(true).unwrap();
                    release_retry.notify_one();
                    std::future::pending::<()>().await;
                } => unreachable!("retry controller branch never completes"),
            }
        })
        .await
        .expect("shutdown must not cancel an in-flight idempotent retry");

        assert_eq!(outcome, PrimaryPersistenceOutcome::Persisted);
        assert_eq!(attempts.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.read().await.persistence_items_written, 1);
        assert!(state.read().await.primary_persistence_available());
    }

    #[tokio::test]
    async fn primary_persistence_shutdown_abandonment_fails_the_integrity_audit() {
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));

        record_primary_persistence_shutdown_abandonment(&metrics, "feed event", 3).await;

        let status = metrics.read().await;
        assert_eq!(status.dropped_messages, 3);
        assert!(status
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("abandoned 3 queued item(s)")));
        assert!(primary_runtime_failure(&status).is_some());
    }

    #[test]
    fn boundary_hydration_retry_delay_is_bounded_and_exponential() {
        let discovery_interval = StdDuration::from_secs(5);
        assert_eq!(
            boundary_hydration_retry_delay(discovery_interval, 1),
            StdDuration::from_secs(5)
        );
        assert_eq!(
            boundary_hydration_retry_delay(discovery_interval, 2),
            StdDuration::from_secs(10)
        );
        assert_eq!(
            boundary_hydration_retry_delay(discovery_interval, 3),
            StdDuration::from_secs(20)
        );
        assert_eq!(
            boundary_hydration_retry_delay(discovery_interval, 4),
            BOUNDARY_HYDRATION_RETRY_MAX_DELAY
        );
        assert_eq!(
            boundary_hydration_retry_delay(discovery_interval, u32::MAX),
            BOUNDARY_HYDRATION_RETRY_MAX_DELAY
        );
    }

    #[test]
    fn retryable_boundary_hydration_control_does_not_poison_primary_runtime_health() {
        let mut metrics = BtcRuntimeMetrics::default();

        let first = classify_boundary_hydration_read_failure(
            anyhow::Error::new(sqlx::Error::PoolTimedOut)
                .context("failed to load durable BTC opening references"),
            &mut metrics,
        );
        assert!(matches!(
            first,
            BoundaryHydrationReadFailure::Retry {
                entered_degraded_state: true,
                consecutive_failures: 1,
                ..
            }
        ));

        let second = classify_boundary_hydration_read_failure(
            anyhow::Error::new(sqlx::Error::PoolTimedOut)
                .context("failed to load durable BTC closing references"),
            &mut metrics,
        );
        assert!(matches!(
            second,
            BoundaryHydrationReadFailure::Retry {
                entered_degraded_state: false,
                consecutive_failures: 2,
                ..
            }
        ));

        assert_eq!(metrics.boundary_hydration_read_errors, 2);
        assert_eq!(metrics.boundary_hydration_consecutive_failures, 2);
        assert_eq!(metrics.persistence_errors, 0);
        assert!(metrics.last_error.is_none());
        assert!(primary_runtime_failure(&metrics).is_none());

        assert!(mark_boundary_hydration_recovered(&mut metrics));
        assert_eq!(metrics.boundary_hydration_read_errors, 2);
        assert_eq!(metrics.boundary_hydration_consecutive_failures, 0);
        assert!(!mark_boundary_hydration_recovered(&mut metrics));
        assert!(primary_runtime_failure(&metrics).is_none());
    }

    #[test]
    fn market_subscription_publication_is_independent_of_boundary_hydration() {
        let market = market();
        let (sender, mut receiver) = watch::channel(Vec::new());

        publish_market_subscriptions_if_changed(&sender, std::slice::from_ref(&market));
        assert!(receiver.has_changed().unwrap());
        let published = receiver.borrow_and_update().clone();
        assert!(same_market_subscriptions(
            &published,
            std::slice::from_ref(&market)
        ));

        publish_market_subscriptions_if_changed(&sender, std::slice::from_ref(&market));
        assert!(!receiver.has_changed().unwrap());
    }

    #[derive(Debug)]
    struct FailingStrategyRunner;

    #[async_trait]
    impl BtcStrategyRunner for FailingStrategyRunner {
        async fn on_observation(&self, _observation: StrategyObservation) -> Result<()> {
            bail!("primary strategy persistence failed")
        }
    }

    #[derive(Debug, Default)]
    struct CountingStrategyRunner {
        callbacks: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl BtcStrategyRunner for CountingStrategyRunner {
        async fn on_observation(&self, _observation: StrategyObservation) -> Result<()> {
            self.callbacks.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[tokio::test]
    async fn degraded_primary_persistence_pauses_and_recovery_resumes_strategy_callbacks() {
        let config = BtcRuntimeConfig {
            enabled: true,
            strategy_interval: StdDuration::from_millis(1),
            ..BtcRuntimeConfig::default()
        };
        let updated_at = Utc::now();
        let state = Arc::new(RwLock::new(RealtimeState {
            primary_persistence_degraded: true,
            last_updated_at: Some(updated_at),
            ..RealtimeState::default()
        }));
        let strategy = Arc::new(CountingStrategyRunner::default());
        let handle = BtcPlaybookRuntimeHandle::start(config, strategy.clone(), state.clone())
            .expect("playbook must start");

        tokio::time::sleep(StdDuration::from_millis(20)).await;
        assert_eq!(strategy.callbacks.load(Ordering::Relaxed), 0);

        {
            let mut realtime = state.write().await;
            realtime.primary_persistence_degraded = false;
            realtime.last_updated_at = Some(updated_at + Duration::milliseconds(1));
        }
        tokio::time::timeout(StdDuration::from_secs(1), async {
            while strategy.callbacks.load(Ordering::Relaxed) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("strategy callbacks must resume after persistence recovery");

        handle.shutdown().await.unwrap();
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
    async fn primary_writer_queue_distinguishes_saturation_from_closure() {
        let state = Arc::new(RwLock::new(RealtimeState::default()));
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let (sender, _receiver) = mpsc::channel(1);
        sender
            .try_send(PersistItem::ReferenceTick(tick(Utc::now(), dec!(99))))
            .unwrap();
        let saturated = enqueue(
            &sender,
            PersistItem::ReferenceTick(tick(Utc::now(), dec!(100))),
            &state,
            &metrics,
        )
        .await;

        assert_eq!(saturated, PersistEnqueueOutcome::Saturated);
        let status = metrics.read().await;
        assert_eq!(status.dropped_messages, 1);
        assert_eq!(status.primary_persistence_queue_overflows, 1);
        assert!(status
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("persistence queue saturated")));
        drop(status);
        assert!(!state.read().await.primary_persistence_available());

        record_primary_persistence_success(&state, &metrics).await;
        assert!(
            !state.read().await.primary_persistence_available(),
            "a dropped durable item must latch the runtime fail-closed"
        );

        let closed_state = Arc::new(RwLock::new(RealtimeState::default()));
        let closed_metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let (closed_sender, closed_receiver) = mpsc::channel(1);
        drop(closed_receiver);
        let closed = enqueue(
            &closed_sender,
            PersistItem::ReferenceTick(tick(Utc::now(), dec!(101))),
            &closed_state,
            &closed_metrics,
        )
        .await;

        assert_eq!(closed, PersistEnqueueOutcome::Closed);
        let status = closed_metrics.read().await;
        assert_eq!(status.dropped_messages, 1);
        assert_eq!(status.primary_persistence_queue_overflows, 0);
        assert!(status
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("persistence queue closed")));
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
            clob_active_subscription_target_fingerprint_sha256: Some("a".repeat(64)),
            clob_active_peer_address: Some("203.0.113.10:443".to_string()),
            clob_active_edge_request_id: Some("LIM-example".to_string()),
            clob_active_last_data_or_heartbeat_at: Some(updated_at),
            clob_active_last_source_to_receive_lag_milliseconds: Some(23_783),
            clob_active_heartbeat_probes: 11,
            clob_active_heartbeat_acknowledgements: 10,
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
            value["metrics"]["clob_active_last_source_to_receive_lag_milliseconds"],
            23_783
        );
        assert_eq!(
            value["metrics"]["clob_active_subscription_target_fingerprint_sha256"],
            "a".repeat(64)
        );
        assert_eq!(
            value["metrics"]["clob_active_peer_address"],
            "203.0.113.10:443"
        );
        assert_eq!(
            value["metrics"]["clob_active_edge_request_id"],
            "LIM-example"
        );
        assert_eq!(value["metrics"]["clob_active_heartbeat_probes"], 11);
        assert_eq!(
            value["metrics"]["clob_active_heartbeat_acknowledgements"],
            10
        );
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
            clob_active_subscription_target_fingerprint_sha256: Some("a".repeat(64)),
            clob_active_peer_address: Some("203.0.113.10:443".to_string()),
            clob_active_edge_request_id: Some("LIM-example".to_string()),
            clob_active_edge_server: Some("cloudflare".to_string()),
            clob_active_handshake_date: Some("date".to_string()),
            clob_active_last_data_or_heartbeat_at: Some(updated_at),
            clob_active_last_source_to_receive_lag_milliseconds: Some(23_783),
            clob_active_heartbeat_probes: 4,
            clob_active_heartbeat_acknowledgements: 3,
            clob_active_last_heartbeat_sent_at: Some(updated_at),
            clob_active_last_heartbeat_acknowledged_at: Some(updated_at),
            clob_active_last_heartbeat_send_lateness_milliseconds: 7,
            clob_active_max_heartbeat_send_lateness_milliseconds: 9,
            clob_active_last_pong_round_trip_milliseconds: Some(11),
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
        assert!(metrics
            .clob_active_subscription_target_fingerprint_sha256
            .is_none());
        assert!(metrics.clob_active_peer_address.is_none());
        assert!(metrics.clob_active_edge_request_id.is_none());
        assert!(metrics.clob_active_edge_server.is_none());
        assert!(metrics.clob_active_handshake_date.is_none());
        assert!(metrics.clob_active_last_data_or_heartbeat_at.is_none());
        assert!(metrics
            .clob_active_last_source_to_receive_lag_milliseconds
            .is_none());
        assert_eq!(metrics.clob_active_heartbeat_probes, 0);
        assert_eq!(metrics.clob_active_heartbeat_acknowledgements, 0);
        assert!(metrics.clob_active_last_heartbeat_sent_at.is_none());
        assert!(metrics.clob_active_last_heartbeat_acknowledged_at.is_none());
        assert_eq!(
            metrics.clob_active_max_heartbeat_send_lateness_milliseconds,
            0
        );
        assert!(metrics
            .clob_active_last_pong_round_trip_milliseconds
            .is_none());
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
            clob_disconnect_cause(false, false, true, false),
            ClobDisconnectCause::Shutdown
        );
        assert_eq!(
            clob_disconnect_cause(true, false, false, false),
            ClobDisconnectCause::SubscriptionFailure
        );
        assert_eq!(
            clob_disconnect_cause(false, false, false, false),
            ClobDisconnectCause::TransportFailure
        );
        assert_eq!(
            private_clob_disconnect_cause(Some("remote_close:None")),
            ClobDisconnectCause::TransportFailure
        );
        assert_eq!(
            private_clob_disconnect_cause(Some("successor_integrity_gap")),
            ClobDisconnectCause::BootstrapFailure
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
            Uuid::new_v4(),
            9,
            unavailable_at,
            unavailable_instant,
            ClobReadinessDiagnostic::transport_unavailable(),
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
    async fn clock_aging_preserves_active_clob_usability_without_new_frames() {
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

        assert!(books_usable);
        assert!(clob_epoch_structurally_ready(
            &registry,
            std::slice::from_ref(&market),
            stale_at,
        ));
        let status = metrics.read().await;
        assert_eq!(status.clob_active_connection_epoch, Some(8));
        assert!(status.clob_recovery_unavailable_since.is_none());
        drop(status);

        let refreshed_at = stale_at + Duration::milliseconds(1);
        apply_ready_book_snapshot(&mut registry, &market, &market.up_token_id, refreshed_at);
        apply_ready_book_snapshot(&mut registry, &market, &market.down_token_id, refreshed_at);
        update_clob_usability(
            &registry,
            std::slice::from_ref(&market),
            refreshed_at + Duration::milliseconds(1),
            started + StdDuration::from_millis(2_013),
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
        assert!(healthy_epoch);
        assert_eq!(registry.connection_id(), connection_id);
        assert_eq!(failures, 0);
        let recovered = metrics.read().await;
        assert_eq!(recovered.clob_active_connection_id, Some(connection_id));
        assert_eq!(recovered.clob_active_connection_epoch, Some(8));
        assert!(recovered.clob_recovery_unavailable_since.is_none());
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

        let mut l2_config = BtcRuntimeConfig {
            binance_spot_l2_enabled: true,
            binance_spot_l2_ws_url: "https://not-a-websocket".to_string(),
            ..BtcRuntimeConfig::default()
        };
        assert!(l2_config.validate().is_err());
        l2_config.binance_spot_l2_enabled = false;
        l2_config.validate().unwrap();
    }

    #[test]
    fn heartbeat_configuration_is_system_owned_and_validated() {
        let heartbeat = BtcHeartbeatConfig::default();
        assert_eq!(heartbeat.clob_interval, StdDuration::from_secs(10));
        assert_eq!(heartbeat.clob_pong_timeout, StdDuration::from_secs(25));
        assert_eq!(heartbeat.rtds_interval, StdDuration::from_secs(5));
        assert_eq!(heartbeat.binance_interval, StdDuration::from_secs(20));
        heartbeat.validate().unwrap();

        let invalid = BtcHeartbeatConfig {
            clob_interval: StdDuration::ZERO,
            ..heartbeat
        };
        assert!(invalid.validate().is_err());
        let excessive = BtcHeartbeatConfig {
            binance_interval: StdDuration::from_secs(BtcHeartbeatConfig::MAX_INTERVAL_SECS + 1),
            ..heartbeat
        };
        assert!(excessive.validate().is_err());
        let invalid_pong_timeout = BtcHeartbeatConfig {
            clob_pong_timeout: heartbeat.clob_interval,
            ..heartbeat
        };
        assert!(invalid_pong_timeout.validate().is_err());
        assert_eq!(REFERENCE_PONG_TIMEOUT, StdDuration::from_secs(10));
    }

    #[test]
    fn resolution_watch_capacity_covers_retention_plus_current_and_next() {
        assert_eq!(resolution_watch_capacity(StdDuration::from_secs(3_600)), 14);
        let mut config = BtcRuntimeConfig::default();
        config.official_resolution_watch_retention = StdDuration::from_secs(719);
        assert!(config.validate().is_err());
    }

    #[test]
    fn gamma_resolution_fallback_honors_audit_grace_and_bounded_backoff() {
        let market = market();
        let grace = StdDuration::from_secs(120);
        let eligible_at = market.window_end + Duration::seconds(120);
        assert!(!gamma_resolution_reconciliation_due(
            &market,
            grace,
            None,
            eligible_at - Duration::milliseconds(1),
        ));
        assert!(gamma_resolution_reconciliation_due(
            &market,
            grace,
            None,
            eligible_at,
        ));

        let mut retries = HashMap::new();
        defer_gamma_resolution_retry(&mut retries, &market.market_id, eligible_at);
        let first = retries[&market.market_id];
        assert_eq!(first.next_attempt_at, eligible_at + Duration::seconds(30));
        assert!(!gamma_resolution_reconciliation_due(
            &market,
            grace,
            Some(first),
            first.next_attempt_at - Duration::milliseconds(1),
        ));
        assert!(gamma_resolution_reconciliation_due(
            &market,
            grace,
            Some(first),
            first.next_attempt_at,
        ));

        defer_gamma_resolution_retry(&mut retries, &market.market_id, first.next_attempt_at);
        let second = retries[&market.market_id];
        assert_eq!(second.backoff, StdDuration::from_secs(60));
        assert_eq!(
            second.next_attempt_at,
            first.next_attempt_at + Duration::seconds(60)
        );
    }

    #[test]
    fn subscriptions_are_narrow_and_include_both_outcomes() {
        let market = market();
        let clob: serde_json::Value =
            serde_json::from_str(&clob_subscription(&[market.clone(), market])).unwrap();
        assert_eq!(clob["assets_ids"], serde_json::json!(["down", "up"]));
        assert_eq!(clob["initial_dump"], true);
        let rtds: serde_json::Value = serde_json::from_str(&rtds_subscription()).unwrap();
        assert_eq!(rtds["subscriptions"][0]["filters"], "btcusdt");
        assert_eq!(
            rtds["subscriptions"][2],
            serde_json::json!({
                "topic": "crypto_prices_twap_sixty",
                "type": "update",
                "filters": "{\"symbol\":\"btc/usd\"}"
            })
        );
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
                "custom_feature_enabled": true,
                "initial_dump": true
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
    fn clob_subscription_fingerprint_is_stable_and_membership_sensitive() {
        let first = market();
        let mut second = first.clone();
        second.market_id = "market-next".to_string();
        second.condition_id = "condition-next".to_string();
        second.up_token_id = "next-up".to_string();
        second.down_token_id = "next-down".to_string();

        let expected = clob_subscription_fingerprint(&[first.clone(), second.clone()]);
        assert_eq!(expected.len(), 64);
        assert_eq!(
            expected,
            clob_subscription_fingerprint(&[second.clone(), first.clone(), first.clone()])
        );

        second.down_token_id = "different-down".to_string();
        assert_ne!(expected, clob_subscription_fingerprint(&[first, second]));
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
            ignored_foreign_events: 3,
            ignored_superseded_events: 5,
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
        assert_eq!(metadata["ignored_foreign_events"], 3);
        assert_eq!(metadata["ignored_superseded_events"], 5);
        assert_eq!(
            metadata["last_subscription_update_at"],
            serde_json::json!(updated_at)
        );
        assert!(metadata.get("subscription_history").is_none());
        for telemetry_field in [
            "connection_role",
            "peer_address",
            "edge_request_id",
            "heartbeat_probes",
            "subscription_target_fingerprint_sha256",
            "transport_error_class",
        ] {
            assert!(metadata.get(telemetry_field).is_none());
        }
    }

    #[test]
    fn clob_handshake_provenance_is_whitelisted_and_bounded() {
        let mut response = ClobHandshakeResponse::new(None);
        response.headers_mut().insert(
            "cf-ray",
            axum::http::HeaderValue::from_str(&"x".repeat(CLOB_PROVENANCE_VALUE_MAX_BYTES + 17))
                .unwrap(),
        );
        response
            .headers_mut()
            .insert("server", axum::http::HeaderValue::from_static("cloudflare"));
        response.headers_mut().insert(
            "date",
            axum::http::HeaderValue::from_static("Wed, 22 Jul 2026 16:00:00 GMT"),
        );
        response.headers_mut().insert(
            "set-cookie",
            axum::http::HeaderValue::from_static("secret=value"),
        );

        assert_eq!(
            bounded_clob_response_header(&response, "cf-ray")
                .unwrap()
                .len(),
            CLOB_PROVENANCE_VALUE_MAX_BYTES
        );
        assert_eq!(
            bounded_clob_response_header(&response, "server").as_deref(),
            Some("cloudflare")
        );
        let provenance = clob_response_provenance(&response);
        assert_eq!(provenance.edge_server.as_deref(), Some("cloudflare"));
        assert!(!format!("{provenance:?}").contains("secret=value"));
    }

    #[test]
    fn clob_transport_errors_keep_typed_bounded_classification() {
        let reset = clob_transport_error_detail(&Error::Protocol(
            ProtocolError::ResetWithoutClosingHandshake,
        ));
        assert_eq!(reset.class, "reset_without_closing_handshake");
        assert!(reset.io_kind.is_none());

        let io = clob_transport_error_detail(&Error::Io(std::io::Error::from_raw_os_error(104)));
        assert_eq!(io.class, "io");
        assert_eq!(io.os_error_code, Some(104));
        assert!(io.io_kind.as_ref().is_some_and(|kind| kind.len() < 64));

        let oversized = "é".repeat(CLOB_ERROR_REASON_MAX_BYTES);
        let bounded = bounded_clob_error_reason(&oversized);
        assert!(bounded.len() <= CLOB_ERROR_REASON_MAX_BYTES);
        assert!(bounded.is_char_boundary(bounded.len()));
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
        let mut second = left.clone();
        second.market_id = "next-market".to_string();
        second.condition_id = "next-condition".to_string();
        second.up_token_id = "next-up".to_string();
        second.down_token_id = "next-down".to_string();
        second.window_start += Duration::minutes(5);
        second.window_end += Duration::minutes(5);
        assert!(same_market_subscriptions(
            &[left.clone(), second.clone()],
            &[second.clone(), left.clone()],
        ));
        assert!(!same_market_subscriptions(
            &[left.clone(), left.clone()],
            &[left.clone(), second],
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
        assert!(is_rtds_twap_60_update(&serde_json::json!({
            "topic": "crypto_prices_twap_sixty",
            "type": "update",
            "payload": {}
        })));
        assert!(!is_rtds_reference_update(&serde_json::json!({
            "topic": "crypto_prices_twap_sixty",
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
    fn stale_sequential_binance_burst_still_completes_model_candles() {
        fn inputs(
            base: DateTime<Utc>,
            aggregate_trade_id: u64,
            source_offset_ms: i64,
            received_offset_ms: i64,
            ingest_sequence: u64,
        ) -> (ReferencePriceTick, BinanceAggregateTrade, DateTime<Utc>) {
            let source_timestamp = base + Duration::milliseconds(source_offset_ms);
            let received_at = base + Duration::milliseconds(received_offset_ms);
            let source_event_id = aggregate_trade_id.to_string();
            (
                reference_test_tick(
                    ReferencePriceSource::DirectBinance,
                    source_timestamp,
                    received_at,
                    Some(&source_event_id),
                    ingest_sequence,
                ),
                BinanceAggregateTrade {
                    aggregate_trade_id,
                    price: dec!(67_000),
                    quantity: dec!(0.1),
                    first_trade_id: aggregate_trade_id,
                    last_trade_id: aggregate_trade_id,
                    transact_time: source_timestamp,
                    is_buyer_maker: false,
                },
                received_at,
            )
        }

        let base = Utc.timestamp_opt(1_783_902_700, 0).unwrap();
        let max_reference_age = Duration::seconds(2);
        let mut state = RealtimeState::default();
        for (aggregate_trade_id, source_offset_ms, received_offset_ms, expected_progress) in [
            (100, 100, 150, true),
            (101, 1_100, 1_150, true),
            // These sequential trades arrived in a transport-buffered burst. They are too
            // old to replace the authoritative current price, but remain required history.
            (102, 2_100, 4_250, false),
            (103, 2_200, 4_260, false),
            (104, 4_300, 4_350, true),
        ] {
            let (tick, trade, received_at) = inputs(
                base,
                aggregate_trade_id,
                source_offset_ms,
                received_offset_ms,
                aggregate_trade_id,
            );
            let (health_progress, aggregation_error) = update_binance_reference_and_model_window(
                &mut state,
                tick,
                &trade,
                received_at,
                max_reference_age,
            );
            assert_eq!(health_progress, expected_progress);
            assert!(aggregation_error.is_none());
            if matches!(aggregate_trade_id, 102 | 103) {
                assert_eq!(
                    state
                        .reference_prices
                        .get(&ReferencePriceSource::DirectBinance)
                        .and_then(|tick| tick.source_event_id.as_deref()),
                    Some("101")
                );
            }
        }

        assert_eq!(
            state
                .reference_prices
                .get(&ReferencePriceSource::DirectBinance)
                .and_then(|tick| tick.source_event_id.as_deref()),
            Some("104")
        );
        for open_offset_seconds in [1, 2, 3] {
            let open_timestamp = base + Duration::seconds(open_offset_seconds);
            let candle = state
                .binance_one_second_window
                .completed()
                .iter()
                .find(|candle| candle.open_timestamp == open_timestamp)
                .expect("expected completed Binance model candle");
            assert!(candle.source_complete);
        }
        assert!(
            state
                .binance_one_second_window
                .current()
                .expect("expected current Binance model candle")
                .source_complete
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
        let mut watchdog = ReferenceFeedWatchdog::new(started_at);
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

        let data_at = started_at + StdDuration::from_secs(3);
        watchdog.on_required_tick(data_at);
        assert_eq!(
            watchdog.stable_deadline,
            Some(data_at + REFERENCE_STABLE_RESET_AFTER)
        );
        let stable_deadline = watchdog.stable_deadline;
        watchdog.on_required_tick(data_at + StdDuration::from_secs(1));
        assert_eq!(watchdog.stable_deadline, stable_deadline);

        let probe_at = started_at + StdDuration::from_secs(5);
        assert_eq!(RTDS_HEARTBEAT_MESSAGE, "ping");
        // RTDS requires a text keepalive but does not guarantee a correlated
        // acknowledgement. Frame-idle detection remains the transport watchdog;
        // required-data freshness is enforced at the trading boundary.
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

        watchdog.arm_binary_pong(probe_at + StdDuration::from_secs(1), expected);
        watchdog.on_frame(probe_at + StdDuration::from_secs(2));
        assert!(!watchdog.awaiting_pong());
        assert!(watchdog.pong_deadline.is_none());

        assert!(watchdog.mark_stable());
        assert!(watchdog.stable);
        assert!(watchdog.stable_deadline.is_none());
        assert!(!watchdog.mark_stable());
    }

    #[test]
    fn reference_watchdog_keeps_freshness_separate_from_transport_liveness() {
        let started_at = Instant::now();
        let mut watchdog = ReferenceFeedWatchdog::new(started_at);
        assert_eq!(
            watchdog.read_idle_deadline,
            started_at + REFERENCE_READ_IDLE_TIMEOUT
        );

        let data_at = started_at + StdDuration::from_secs(3);
        watchdog.on_required_tick(data_at);

        assert_eq!(
            watchdog.stable_deadline,
            Some(data_at + REFERENCE_STABLE_RESET_AFTER)
        );
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
        let max_delay = Duration::seconds(5);
        let mut tracker = BoundaryTracker::default();
        tracker.update_markets(std::slice::from_ref(&market));
        tracker.observe_chainlink(
            &tick(market.window_start - Duration::milliseconds(1), dec!(100)),
            max_delay,
        );
        assert!(tracker
            .pending_candidates(market.window_start, max_delay)
            .unwrap()
            .is_empty());
        tracker.observe_chainlink(
            &tick(market.window_start + Duration::milliseconds(100), dec!(100)),
            max_delay,
        );
        let open = tracker
            .pending_candidates(market.window_start, max_delay)
            .unwrap();
        assert!(matches!(open.as_slice(), [BoundaryCandidate::Open { .. }]));
        tracker.acknowledge(&open[0]).unwrap();
        tracker.observe_chainlink(
            &tick(market.window_end + Duration::milliseconds(100), dec!(100)),
            max_delay,
        );
        let close = tracker
            .pending_candidates(market.window_end + Duration::milliseconds(100), max_delay)
            .unwrap();
        assert!(matches!(
            close.as_slice(),
            [BoundaryCandidate::Close { .. }]
        ));
        tracker.acknowledge(&close[0]).unwrap();
        assert!(tracker
            .pending_candidates(
                market.window_end + max_delay - Duration::milliseconds(1),
                max_delay,
            )
            .unwrap()
            .is_empty());
        let finalized_at = market.window_end + max_delay;
        let label = tracker.pending_candidates(finalized_at, max_delay).unwrap();
        assert!(matches!(
            label.as_slice(),
            [BoundaryCandidate::Label {
                label: BtcMarketLabel {
                    outcome: BtcOutcome::Up,
                    label_available_at,
                    ..
                },
                ..
            }] if *label_available_at == finalized_at
        ));
        tracker.acknowledge(&label[0]).unwrap();
        assert!(tracker
            .pending_candidates(finalized_at, max_delay)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn boundary_tracker_refuses_late_open_ticks() {
        let market = market();
        let mut tracker = BoundaryTracker::default();
        tracker.update_markets(std::slice::from_ref(&market));
        tracker.observe_chainlink(
            &tick(market.window_start + Duration::seconds(6), dec!(100)),
            Duration::seconds(5),
        );
        assert!(tracker
            .pending_candidates(market.window_end, Duration::seconds(5))
            .unwrap()
            .is_empty());
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
        let max_delay = Duration::seconds(5);
        let items = tracker
            .pending_candidates(market.window_end + max_delay, max_delay)
            .unwrap();
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
        tracker.observe_chainlink(&pending, Duration::seconds(5));
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
        let max_delay = Duration::seconds(5);
        tracker.observe_chainlink(&open, max_delay);

        let first = tracker
            .pending_candidates(market.window_start, max_delay)
            .unwrap();
        let second = tracker
            .pending_candidates(market.window_start, max_delay)
            .unwrap();
        assert!(matches!(first.as_slice(), [BoundaryCandidate::Open { .. }]));
        assert!(matches!(
            second.as_slice(),
            [BoundaryCandidate::Open { .. }]
        ));
        tracker.acknowledge(&first[0]).unwrap();
        assert!(tracker
            .pending_candidates(market.window_start, max_delay)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn boundary_tracker_selects_earlier_close_before_label_finalization() {
        let market = market();
        let max_delay = Duration::seconds(5);
        let open = tick(market.window_start + Duration::milliseconds(50), dec!(100));
        let later_close = tick(market.window_end + Duration::milliseconds(200), dec!(101));
        let earlier_close = tick(market.window_end + Duration::milliseconds(100), dec!(99));
        let mut tracker = BoundaryTracker::default();
        tracker.update_markets(std::slice::from_ref(&market));
        tracker
            .hydrate_open_references(&[(market.market_id.clone(), open)], max_delay)
            .unwrap();

        tracker.observe_chainlink(&later_close, max_delay);
        let later_candidate = tracker
            .pending_candidates(market.window_end + Duration::milliseconds(200), max_delay)
            .unwrap();
        assert!(matches!(
            later_candidate.as_slice(),
            [BoundaryCandidate::Close { tick, .. }]
                if tick.source_timestamp == later_close.source_timestamp
        ));
        tracker.acknowledge(&later_candidate[0]).unwrap();

        let observation = tracker.observe_chainlink(&earlier_close, max_delay);
        assert!(observation.finalized_late_ticks.is_empty());
        let earlier_candidate = tracker
            .pending_candidates(market.window_end + Duration::milliseconds(300), max_delay)
            .unwrap();
        assert!(matches!(
            earlier_candidate.as_slice(),
            [BoundaryCandidate::Close { tick, .. }]
                if tick.source_timestamp == earlier_close.source_timestamp
        ));
        tracker.acknowledge(&earlier_candidate[0]).unwrap();

        let finalized_at = market.window_end + max_delay;
        let label_candidate = tracker.pending_candidates(finalized_at, max_delay).unwrap();
        assert!(matches!(
            label_candidate.as_slice(),
            [BoundaryCandidate::Label {
                label: BtcMarketLabel {
                    close_price,
                    source_close_timestamp,
                    outcome: BtcOutcome::Down,
                    ..
                },
                ..
            }] if *close_price == earlier_close.price
                && *source_close_timestamp == earlier_close.source_timestamp
        ));
    }

    #[test]
    fn boundary_tracker_hydrates_existing_label_without_regenerating_it() {
        let market = market();
        let open = tick(market.window_start + Duration::milliseconds(50), dec!(100));
        let close = tick(market.window_end + Duration::milliseconds(50), dec!(99));
        let max_delay = Duration::seconds(5);
        let finalized_at = market.window_end + max_delay;
        let label = boundary_label(&market, &open, &close, finalized_at);
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

        assert!(tracker
            .pending_candidates(finalized_at, max_delay)
            .unwrap()
            .is_empty());
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
        let durable = boundary_label(
            &market,
            &open,
            &close,
            market.window_end + Duration::seconds(5),
        );
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
    fn boundary_tracker_quarantines_earlier_close_after_label_ack() {
        let market = market();
        let open = tick(market.window_start + Duration::milliseconds(50), dec!(100));
        let close = tick(market.window_end + Duration::milliseconds(200), dec!(101));
        let label = boundary_label(
            &market,
            &open,
            &close,
            market.window_end + Duration::seconds(5),
        );
        let mut tracker = BoundaryTracker::default();
        tracker.update_markets(std::slice::from_ref(&market));
        tracker
            .hydrate_open_references(
                &[(market.market_id.clone(), open.clone())],
                Duration::seconds(5),
            )
            .unwrap();
        tracker
            .hydrate_close_references(
                &[(market.market_id.clone(), close.clone())],
                Duration::seconds(5),
            )
            .unwrap();
        tracker
            .hydrate_labels(std::slice::from_ref(&label))
            .unwrap();

        let earlier = tick(market.window_end + Duration::milliseconds(100), dec!(99));
        let observation = tracker.observe_chainlink(&earlier, Duration::seconds(5));

        assert_eq!(
            observation.finalized_late_ticks,
            vec![FinalizedBoundaryLateTick {
                market_id: market.market_id.clone(),
                boundary: FinalizedBoundaryKind::Close,
                acknowledged_source_timestamp: close.source_timestamp,
                changes_label_outcome: true,
            }]
        );
        let boundary = &tracker.markets[&market.market_id];
        assert_eq!(boundary.open_tick.as_ref(), Some(&open));
        assert_eq!(boundary.close_tick.as_ref(), Some(&close));
        assert_eq!(boundary.label.as_ref(), Some(&label));
    }

    #[test]
    fn boundary_tracker_quarantines_earlier_open_after_ack() {
        let market = market();
        let open = tick(market.window_start + Duration::milliseconds(200), dec!(100));
        let mut tracker = BoundaryTracker::default();
        tracker.update_markets(std::slice::from_ref(&market));
        tracker
            .hydrate_open_references(
                &[(market.market_id.clone(), open.clone())],
                Duration::seconds(5),
            )
            .unwrap();

        let earlier = tick(market.window_start + Duration::milliseconds(100), dec!(99));
        let observation = tracker.observe_chainlink(&earlier, Duration::seconds(5));

        assert_eq!(
            observation.finalized_late_ticks,
            vec![FinalizedBoundaryLateTick {
                market_id: market.market_id.clone(),
                boundary: FinalizedBoundaryKind::Open,
                acknowledged_source_timestamp: open.source_timestamp,
                changes_label_outcome: false,
            }]
        );
        assert_eq!(
            tracker.markets[&market.market_id].open_tick.as_ref(),
            Some(&open)
        );
    }
}
