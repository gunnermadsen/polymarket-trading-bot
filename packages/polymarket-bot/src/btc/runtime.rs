use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::{self, Write},
    future::Future,
    panic::AssertUnwindSafe,
    pin::Pin,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration as StdDuration,
};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use futures_util::{
    stream::{self, SplitSink, SplitStream},
    FutureExt, Sink, SinkExt, StreamExt,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    net::TcpStream,
    sync::{mpsc, watch, Mutex, OwnedSemaphorePermit, RwLock, Semaphore},
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
        parse_rtds_reference_tick, BookRegistry,
    },
    market::{
        discovery_windows, parse_clob_rest_official_resolution, parse_gamma_btc_interval_event,
        parse_gamma_rest_official_resolution, slug_for_window, ClobRestOfficialResolution,
        GammaRestOfficialResolution,
    },
    repository::{
        BtcMarketLabel, BtcOfficialResolutionWatch, BtcRepository, PersistedOfficialResolution,
    },
    types::{
        BinanceAggregateTrade, BinanceOneSecondKline, BinanceOneSecondWindow, BtcIntervalMarket,
        BtcOutcome, FeedIntegrityStatus, Readiness, RealtimeState, ReferencePriceSource,
        ReferencePriceTick, BINANCE_ONE_SECOND_BOOTSTRAP_CAPACITY,
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
const CLOB_INGRESS_FRAME_CAPACITY: usize = 2_048;
const CLOB_INGRESS_BYTE_CAPACITY: usize = 16 * 1024 * 1024;
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
type ClobSocketSink = SplitSink<ClobSocket, Message>;
type ClobSocketStream = SplitStream<ClobSocket>;

#[derive(Debug)]
struct ClobIngressFrame {
    message: Message,
    received_at: DateTime<Utc>,
    received_instant: Instant,
    sequence: u64,
    payload_bytes: usize,
    _byte_permit: Option<OwnedSemaphorePermit>,
}

#[derive(Debug)]
enum ClobIngressEvent {
    Frame(ClobIngressFrame),
    HeartbeatSent {
        observed_at: DateTime<Utc>,
        scheduled_at: Instant,
        sent_at: Instant,
    },
    TransportError {
        reason: String,
        detail: ClobTransportErrorDetail,
    },
    Eof,
}

#[derive(Debug)]
struct ClobIngressTransport {
    sink: Arc<Mutex<ClobSocketSink>>,
    events: mpsc::Receiver<ClobIngressEvent>,
    overflowed: Arc<AtomicBool>,
    receipt_clock_started_at: Instant,
    latest_receipt_elapsed_nanoseconds: Arc<AtomicU64>,
    reader_task: JoinHandle<()>,
    heartbeat_task: JoinHandle<()>,
}

impl ClobIngressTransport {
    fn latest_receipt_instant(&self) -> Option<Instant> {
        let encoded = self
            .latest_receipt_elapsed_nanoseconds
            .load(Ordering::Acquire);
        (encoded != 0).then(|| {
            self.receipt_clock_started_at + StdDuration::from_nanos(encoded.saturating_sub(1))
        })
    }
}

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
/// These counters are runtime telemetry only: the L2 feed supplies model
/// features at runtime and does not own a database data contract.
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
    #[serde(default)]
    pub clob_recovery_unavailable_age_milliseconds: Option<u64>,
    pub clob_last_disconnect_reason: Option<String>,
    #[serde(default)]
    pub clob_remote_close_1013_count: u64,
    #[serde(default)]
    pub clob_last_remote_close_1013_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub clob_last_remote_close_1013_age_milliseconds: Option<u64>,
    #[serde(default)]
    pub clob_last_remote_close_1013_reason: Option<String>,
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
    #[serde(default)]
    pub clob_active_ingress_frames: u64,
    #[serde(default)]
    pub clob_active_ingress_bytes: u64,
    #[serde(default)]
    pub clob_active_last_ingress_sequence: u64,
    #[serde(default)]
    pub clob_active_ingress_queue_depth: u64,
    #[serde(default)]
    pub clob_active_last_ingress_queue_dwell_milliseconds: u64,
    #[serde(default)]
    pub clob_active_max_ingress_queue_dwell_milliseconds: u64,
    #[serde(default)]
    pub clob_active_ingress_overflows: u64,
    #[serde(default)]
    pub clob_active_last_frame_processing_milliseconds: u64,
    #[serde(default)]
    pub clob_active_max_frame_processing_milliseconds: u64,
    #[serde(default)]
    pub clob_active_last_shared_books_lock_wait_milliseconds: u64,
    #[serde(default)]
    pub clob_active_max_shared_books_lock_wait_milliseconds: u64,
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
    metrics.clob_last_remote_close_1013_age_milliseconds = metrics
        .clob_last_remote_close_1013_at
        .filter(|closed_at| *closed_at <= checked_at)
        .map(|closed_at| {
            u64::try_from((checked_at - closed_at).num_milliseconds()).unwrap_or(u64::MAX)
        });
    metrics.clob_recovery_unavailable_age_milliseconds = metrics
        .clob_recovery_unavailable_since
        .filter(|unavailable_since| *unavailable_since <= checked_at)
        .map(|unavailable_since| {
            u64::try_from((checked_at - unavailable_since).num_milliseconds()).unwrap_or(u64::MAX)
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
        let causal_transition = self
            .diagnostic
            .as_ref()
            .is_none_or(|current| !current.same_cause(&diagnostic));
        self.diagnostic = Some(diagnostic);
        causal_transition
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
    fn same_cause(&self, other: &Self) -> bool {
        self.reason == other.reason
            && self.market_id == other.market_id
            && self.token_id == other.token_id
            && self.integrity_status == other.integrity_status
            && self.bootstrapped == other.bootstrapped
            && self.has_bid == other.has_bid
            && self.has_ask == other.has_ask
    }

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
    last_updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Default)]
struct ClobPendingFrameMetrics {
    messages_received: u64,
    events_applied: u64,
    integrity_gaps: u64,
    last_source_to_receive_lag_milliseconds: Option<i64>,
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
    last_transport_receipt_at: Option<Instant>,
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
            last_transport_receipt_at: None,
        };
        watchdog.refresh_bootstrap(now, registry, markets, checked_at);
        watchdog
    }

    fn observe_transport_receipt(&mut self, received_at: Option<Instant>) {
        let Some(received_at) = received_at else {
            return;
        };
        if self
            .last_transport_receipt_at
            .is_some_and(|observed_at| received_at <= observed_at)
        {
            return;
        }
        self.last_transport_receipt_at = Some(received_at);
        self.read_idle_deadline = received_at + CLOB_READ_IDLE_TIMEOUT;
        // Some CLOB edges deliver market data without echoing every text PING.
        // Only transport evidence received after the pending probe proves that
        // probe's connection remained alive.
        if self
            .pending_pong_probe_sent_at
            .is_some_and(|sent_at| received_at >= sent_at)
        {
            self.pong_deadline = None;
        }
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
        let sent_at = self.pending_pong_probe_sent_at?;
        if now < sent_at {
            return None;
        }
        self.observe_transport_receipt(Some(now));
        self.pending_pong_probe_sent_at = None;
        Some(now.saturating_duration_since(sent_at))
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
    ingress_frames: u64,
    ingress_bytes: u64,
    last_ingress_sequence: u64,
    ingress_queue_depth: usize,
    last_ingress_queue_dwell: StdDuration,
    max_ingress_queue_dwell: StdDuration,
    ingress_overflows: u64,
    last_frame_processing: StdDuration,
    max_frame_processing: StdDuration,
    last_shared_books_lock_wait: StdDuration,
    max_shared_books_lock_wait: StdDuration,
    remote_close_observed: bool,
    remote_close_code: Option<u16>,
    remote_close_reason: Option<String>,
    last_transport_error: Option<ClobTransportErrorDetail>,
}

impl ClobSocketTelemetry {
    fn new(markets: &[BtcIntervalMarket]) -> Self {
        Self {
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
            ingress_frames: 0,
            ingress_bytes: 0,
            last_ingress_sequence: 0,
            ingress_queue_depth: 0,
            last_ingress_queue_dwell: StdDuration::ZERO,
            max_ingress_queue_dwell: StdDuration::ZERO,
            ingress_overflows: 0,
            last_frame_processing: StdDuration::ZERO,
            max_frame_processing: StdDuration::ZERO,
            last_shared_books_lock_wait: StdDuration::ZERO,
            max_shared_books_lock_wait: StdDuration::ZERO,
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

    fn record_ingress_frame(&mut self, frame: &ClobIngressFrame, processing_at: Instant) {
        let queue_dwell = processing_at.saturating_duration_since(frame.received_instant);
        self.record_frame(frame.received_at, frame.received_instant);
        self.ingress_frames = self.ingress_frames.saturating_add(1);
        self.ingress_bytes = self
            .ingress_bytes
            .saturating_add(u64::try_from(frame.payload_bytes).unwrap_or(u64::MAX));
        self.last_ingress_sequence = frame.sequence;
        self.last_ingress_queue_dwell = queue_dwell;
        self.max_ingress_queue_dwell = self.max_ingress_queue_dwell.max(queue_dwell);
    }

    fn record_ingress_overflow(&mut self) {
        self.ingress_overflows = self.ingress_overflows.saturating_add(1);
    }

    fn record_frame_processing(
        &mut self,
        processing_duration: StdDuration,
        shared_books_lock_wait: StdDuration,
    ) {
        self.last_frame_processing = processing_duration;
        self.max_frame_processing = self.max_frame_processing.max(processing_duration);
        self.last_shared_books_lock_wait = shared_books_lock_wait;
        self.max_shared_books_lock_wait =
            self.max_shared_books_lock_wait.max(shared_books_lock_wait);
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

    fn record_remote_close(&mut self, frame: Option<&CloseFrame>) {
        self.remote_close_observed = true;
        self.remote_close_code = frame.map(|frame| u16::from(frame.code));
        self.remote_close_reason = frame
            .map(|frame| frame.reason.as_str())
            .filter(|reason| !reason.is_empty())
            .map(bounded_clob_error_reason);
    }
}

fn record_clob_remote_close_1013(
    metrics: &mut BtcRuntimeMetrics,
    telemetry: &ClobSocketTelemetry,
    observed_at: DateTime<Utc>,
) {
    if telemetry.remote_close_observed && telemetry.remote_close_code == Some(1013) {
        metrics.clob_remote_close_1013_count =
            metrics.clob_remote_close_1013_count.saturating_add(1);
        metrics.clob_last_remote_close_1013_at = Some(observed_at);
        metrics.clob_last_remote_close_1013_reason = telemetry.remote_close_reason.clone();
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

fn start_clob_ingress(
    socket: ClobSocket,
    heartbeat_interval: StdDuration,
    shutdown: watch::Receiver<bool>,
) -> ClobIngressTransport {
    let (sink, stream) = socket.split();
    let sink = Arc::new(Mutex::new(sink));
    let (sender, events) = mpsc::channel(CLOB_INGRESS_FRAME_CAPACITY);
    let byte_budget = Arc::new(Semaphore::new(CLOB_INGRESS_BYTE_CAPACITY));
    let overflowed = Arc::new(AtomicBool::new(false));
    let receipt_clock_started_at = Instant::now();
    let latest_receipt_elapsed_nanoseconds = Arc::new(AtomicU64::new(0));
    let reader_overflowed = Arc::clone(&overflowed);
    let reader_latest_receipt_elapsed_nanoseconds = Arc::clone(&latest_receipt_elapsed_nanoseconds);
    let reader_task = tokio::spawn(run_clob_ingress_reader(
        stream,
        sender.clone(),
        reader_overflowed,
        byte_budget,
        receipt_clock_started_at,
        reader_latest_receipt_elapsed_nanoseconds,
    ));
    let heartbeat_task = tokio::spawn(run_clob_heartbeat_sender(
        Arc::clone(&sink),
        sender,
        heartbeat_interval,
        shutdown,
    ));
    ClobIngressTransport {
        sink,
        events,
        overflowed,
        receipt_clock_started_at,
        latest_receipt_elapsed_nanoseconds,
        reader_task,
        heartbeat_task,
    }
}

async fn run_clob_heartbeat_sender(
    sink: Arc<Mutex<ClobSocketSink>>,
    sender: mpsc::Sender<ClobIngressEvent>,
    heartbeat_interval: StdDuration,
    mut shutdown: watch::Receiver<bool>,
) {
    let started_at = Instant::now();
    let mut heartbeat = interval_at(started_at + heartbeat_interval, heartbeat_interval);
    heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        let scheduled_at = tokio::select! {
            biased;
            _ = shutdown.changed() => return,
            scheduled_at = heartbeat.tick() => scheduled_at,
        };
        let mut socket = tokio::select! {
            biased;
            _ = shutdown.changed() => return,
            socket = sink.lock() => socket,
        };
        match send_clob_text(&mut *socket, "PING".to_string(), &mut shutdown).await {
            Ok(()) => {
                let sent_at = Instant::now();
                let event = ClobIngressEvent::HeartbeatSent {
                    observed_at: Utc::now(),
                    scheduled_at,
                    sent_at,
                };
                match sender.try_send(event) {
                    Ok(()) | Err(mpsc::error::TrySendError::Full(_)) => {}
                    Err(mpsc::error::TrySendError::Closed(_)) => return,
                }
            }
            Err(ClobSendFailure::Shutdown) => return,
            Err(ClobSendFailure::Timeout) => {
                let _ = sender
                    .send(ClobIngressEvent::TransportError {
                        reason: "heartbeat_send_timeout".to_string(),
                        detail: ClobTransportErrorDetail {
                            class: "send_timeout",
                            io_kind: None,
                            os_error_code: None,
                        },
                    })
                    .await;
                return;
            }
            Err(ClobSendFailure::Transport { error, detail }) => {
                let reason = bounded_clob_error_reason(&format!("heartbeat_send_failed:{error}"));
                let _ = sender
                    .send(ClobIngressEvent::TransportError { reason, detail })
                    .await;
                return;
            }
        }
    }
}

async fn run_clob_ingress_reader(
    mut stream: ClobSocketStream,
    sender: mpsc::Sender<ClobIngressEvent>,
    overflowed: Arc<AtomicBool>,
    byte_budget: Arc<Semaphore>,
    receipt_clock_started_at: Instant,
    latest_receipt_elapsed_nanoseconds: Arc<AtomicU64>,
) {
    let mut sequence = 0u64;
    loop {
        match stream.next().await {
            Some(Ok(message)) => {
                let received_instant = Instant::now();
                let received_at = Utc::now();
                let elapsed_nanoseconds = u64::try_from(
                    received_instant
                        .saturating_duration_since(receipt_clock_started_at)
                        .as_nanos(),
                )
                .unwrap_or(u64::MAX)
                .saturating_add(1);
                latest_receipt_elapsed_nanoseconds.store(elapsed_nanoseconds, Ordering::Release);
                sequence = sequence.saturating_add(1);
                let payload_bytes = message.len();
                let permit_count = match u32::try_from(payload_bytes.max(1)) {
                    Ok(value) => value,
                    Err(_) => {
                        overflowed.store(true, Ordering::Release);
                        return;
                    }
                };
                let byte_permit =
                    match Arc::clone(&byte_budget).try_acquire_many_owned(permit_count) {
                        Ok(permit) => permit,
                        Err(_) => {
                            overflowed.store(true, Ordering::Release);
                            return;
                        }
                    };
                let frame = ClobIngressEvent::Frame(ClobIngressFrame {
                    payload_bytes,
                    message,
                    received_at,
                    received_instant,
                    sequence,
                    _byte_permit: Some(byte_permit),
                });
                match sender.try_send(frame) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        overflowed.store(true, Ordering::Release);
                        return;
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => return,
                }
            }
            Some(Err(error)) => {
                let detail = clob_transport_error_detail(&error);
                let reason = bounded_clob_error_reason(&format!("transport_read_failed:{error}"));
                let _ = sender
                    .send(ClobIngressEvent::TransportError { reason, detail })
                    .await;
                return;
            }
            None => {
                let _ = sender.send(ClobIngressEvent::Eof).await;
                return;
            }
        }
    }
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
    metrics.clob_active_ingress_frames = telemetry.ingress_frames;
    metrics.clob_active_ingress_bytes = telemetry.ingress_bytes;
    metrics.clob_active_last_ingress_sequence = telemetry.last_ingress_sequence;
    metrics.clob_active_ingress_queue_depth =
        u64::try_from(telemetry.ingress_queue_depth).unwrap_or(u64::MAX);
    metrics.clob_active_last_ingress_queue_dwell_milliseconds =
        duration_milliseconds(telemetry.last_ingress_queue_dwell);
    metrics.clob_active_max_ingress_queue_dwell_milliseconds =
        duration_milliseconds(telemetry.max_ingress_queue_dwell);
    metrics.clob_active_ingress_overflows = telemetry.ingress_overflows;
    metrics.clob_active_last_frame_processing_milliseconds =
        duration_milliseconds(telemetry.last_frame_processing);
    metrics.clob_active_max_frame_processing_milliseconds =
        duration_milliseconds(telemetry.max_frame_processing);
    metrics.clob_active_last_shared_books_lock_wait_milliseconds =
        duration_milliseconds(telemetry.last_shared_books_lock_wait);
    metrics.clob_active_max_shared_books_lock_wait_milliseconds =
        duration_milliseconds(telemetry.max_shared_books_lock_wait);
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

async fn send_clob_text<S>(
    socket: &mut S,
    payload: String,
    shutdown: &mut watch::Receiver<bool>,
) -> std::result::Result<(), ClobSendFailure>
where
    S: Sink<Message, Error = Error> + Unpin,
{
    send_clob_text_with_timeout(socket, payload, shutdown, CLOB_SEND_TIMEOUT).await
}

async fn send_clob_text_with_timeout<S>(
    socket: &mut S,
    payload: String,
    shutdown: &mut watch::Receiver<bool>,
    send_timeout: StdDuration,
) -> std::result::Result<(), ClobSendFailure>
where
    S: Sink<Message, Error = Error> + Unpin,
{
    if *shutdown.borrow() {
        return Err(ClobSendFailure::Shutdown);
    }
    tokio::select! {
        biased;
        _ = shutdown.changed() => Err(ClobSendFailure::Shutdown),
        result = timeout(send_timeout, socket.send(Message::Text(payload.into()))) => {
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
    transport: ClobIngressTransport,
    registry: BookRegistry,
    markets: Vec<BtcIntervalMarket>,
    connected_at: DateTime<Utc>,
    subscription_stats: ClobSubscriptionStats,
    telemetry: ClobSocketTelemetry,
    connected_instant: Instant,
    watchdog: ClobFeedWatchdog,
    healthy_epoch: bool,
    books_usable: bool,
    pending_frame_metrics: ClobPendingFrameMetrics,
}

impl Drop for ClobEpoch {
    fn drop(&mut self) {
        self.transport.reader_task.abort();
        self.transport.heartbeat_task.abort();
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
    reason: String,
    kind: ClobConnectFailureKind,
    telemetry: ClobSocketTelemetry,
}

async fn connect_clob_epoch(
    config: BtcRuntimeConfig,
    desired_markets: Vec<BtcIntervalMarket>,
    connection_epoch: i32,
    heartbeat_interval: StdDuration,
    mut shutdown: watch::Receiver<bool>,
) -> ClobConnectOutcome {
    let connection_id = Uuid::new_v4();
    let attempt_started_at = Instant::now();
    let mut telemetry = ClobSocketTelemetry::new(&desired_markets);
    let mut registry = BookRegistry::new(connection_id);
    if let Err(error) = validate_clob_execution_market_set(&desired_markets) {
        let reason = bounded_clob_error_reason(&format!("invalid_subscription_target:{error}"));
        return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
            reason,
            kind: ClobConnectFailureKind::Identity,
            telemetry,
        }));
    }
    if let Err(error) = register_clob_markets(&mut registry, &desired_markets) {
        let reason = bounded_clob_error_reason(&format!("invalid_subscription_identity:{error}"));
        return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
            reason,
            kind: ClobConnectFailureKind::Identity,
            telemetry,
        }));
    }
    if *shutdown.borrow() {
        return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
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
            return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
                reason: "shutdown".to_string(),
                kind: ClobConnectFailureKind::Shutdown,
                telemetry,
            }));
        }
        Some(Err(_)) => {
            return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
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
            return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
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
    match send_clob_text(
        &mut socket,
        clob_subscription(&desired_markets),
        &mut shutdown,
    )
    .await
    {
        Ok(()) => {}
        Err(ClobSendFailure::Shutdown) => {
            return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
                reason: "shutdown".to_string(),
                kind: ClobConnectFailureKind::Shutdown,
                telemetry,
            }));
        }
        Err(ClobSendFailure::Timeout) => {
            return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
                reason: "subscription_send_timeout".to_string(),
                kind: ClobConnectFailureKind::Subscription,
                telemetry,
            }));
        }
        Err(ClobSendFailure::Transport { error, detail }) => {
            let reason = bounded_clob_error_reason(&format!("subscription_send_failed:{error}"));
            telemetry.last_transport_error = Some(detail);
            return ClobConnectOutcome::Failed(Box::new(ClobConnectFailure {
                reason,
                kind: ClobConnectFailureKind::Subscription,
                telemetry,
            }));
        }
    }
    let watchdog_started = Instant::now();
    let watchdog = ClobFeedWatchdog::new(watchdog_started, &registry, &desired_markets, Utc::now());
    ClobConnectOutcome::Connected {
        epoch: Box::new(ClobEpoch {
            connection_id,
            connection_epoch,
            transport: start_clob_ingress(socket, heartbeat_interval, shutdown.clone()),
            registry,
            markets: desired_markets,
            connected_at,
            subscription_stats: ClobSubscriptionStats::default(),
            telemetry,
            connected_instant,
            watchdog,
            healthy_epoch: false,
            books_usable: false,
            pending_frame_metrics: ClobPendingFrameMetrics::default(),
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
    Arc<RwLock<BookRegistry>>,
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
    async fn reconcile_if_due(&self) -> Result<()> {
        Ok(())
    }

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
            self.books.clone(),
            self.metrics.clone(),
            self.config.clone(),
            self.running.clone(),
        )
    }

    pub async fn status(&self) -> BtcRuntimeStatus {
        let state = realtime_snapshot(&self.state, &self.books).await;
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
    books: Arc<RwLock<BookRegistry>>,
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
        books: Arc<RwLock<BookRegistry>>,
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
                books.clone(),
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
            books,
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
            self.books.clone(),
            self.metrics.clone(),
            self.config.clone(),
            self.running.clone(),
        )
    }

    pub async fn status(&self) -> BtcRuntimeStatus {
        runtime_status_from_inputs(
            self.state.clone(),
            self.books.clone(),
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
    books: Arc<RwLock<BookRegistry>>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
    config: BtcRuntimeConfig,
    running: Arc<AtomicBool>,
) -> BtcRuntimeStatus {
    let state = realtime_snapshot(&state, &books).await;
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

async fn realtime_snapshot(
    state: &Arc<RwLock<RealtimeState>>,
    books: &Arc<RwLock<BookRegistry>>,
) -> RealtimeState {
    let mut snapshot = state.read().await.clone();
    let books = books.read().await;
    snapshot.update_books(&books);
    if let Some(book_received_at) = snapshot
        .books
        .values()
        .filter_map(|book| book.received_at)
        .max()
    {
        snapshot.last_updated_at = Some(
            snapshot
                .last_updated_at
                .map_or(book_received_at, |updated_at| {
                    updated_at.max(book_received_at)
                }),
        );
    }
    snapshot
}

#[derive(Debug)]
enum PersistItem {
    ReferenceTick(ReferencePriceTick),
    Checkpoint(super::types::OrderbookCheckpoint),
}

impl PersistItem {
    fn kind(&self) -> &'static str {
        match self {
            Self::ReferenceTick(_) => "reference tick",
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
    if recovered {
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
        Ok(()) => {
            if metrics.try_read().is_ok_and(|runtime_metrics| {
                runtime_metrics.primary_persistence_consecutive_failures == 0
            }) {
                match state.try_write() {
                    Ok(mut realtime) if realtime.primary_persistence_degraded => {
                        realtime.primary_persistence_degraded = false;
                        realtime.last_updated_at = Some(Utc::now());
                    }
                    Ok(_) | Err(_) => {}
                }
            }
            PersistEnqueueOutcome::Queued
        }
        Err(mpsc::error::TrySendError::Full(_)) => {
            let mut metrics = metrics.write().await;
            metrics.dropped_messages = metrics.dropped_messages.saturating_add(1);
            metrics.primary_persistence_queue_overflows = metrics
                .primary_persistence_queue_overflows
                .saturating_add(1);
            let mut realtime = state.write().await;
            if !realtime.primary_persistence_degraded {
                realtime.primary_persistence_degraded = true;
                realtime.last_updated_at = Some(Utc::now());
                metrics.last_error = Some(format!(
                    "BTC persistence queue saturated; dropped {item_kind}"
                ));
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
    let mut ticker = interval(config.discovery_interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut gamma_resolution_retries = HashMap::new();
    let mut boundary_hydration_retry_at = None;
    let mut resolution_watches_seeded = false;
    loop {
        tokio::select! {
            _ = shutdown.changed() => break,
            _ = ticker.tick() => {
                let now = Utc::now();
                let fresh_markets = match discover_markets(&client, &config, now).await {
                    Ok(markets) => Some(markets),
                    Err(error) => {
                        record_error(&metrics, error).await;
                        None
                    }
                };
                let fresh_market_slice = fresh_markets.as_deref().unwrap_or_default();
                let mut durable_fresh_market_ids = HashSet::with_capacity(fresh_market_slice.len());
                for market in fresh_market_slice {
                    if let Err(error) = repository.upsert_interval_market(market).await {
                        record_critical_persistence_error(&metrics, error).await;
                        continue;
                    }
                    if let Err(error) = repository
                        .register_official_resolution_watch(market, now, resolution_retention)
                        .await
                    {
                        record_critical_persistence_error(&metrics, error).await;
                        continue;
                    }
                    durable_fresh_market_ids.insert(market.market_id.clone());
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.persistence_items_written = runtime_metrics
                        .persistence_items_written
                        .saturating_add(1);
                }

                // Only Gamma responses can replace the current market; durable recovery rows
                // are subscription/audit inputs and can never re-open an old market. A failed
                // request is not evidence that the current market stopped trading, so retain
                // the last successful response. All order paths still enforce is_trade_window(),
                // preventing an expired retained market from authorizing a trade.
                if let Some(update) = current_market_update(fresh_markets.as_deref(), now) {
                    let tradable_is_durable = update.tradable.as_ref().is_none_or(|market| {
                        durable_fresh_market_ids.contains(&market.market_id)
                    });
                    let mut state = state.write().await;
                    state.display_market = update.display;
                    if tradable_is_durable {
                        let subscription = update.tradable.clone();
                        state.set_current_market(update.tradable);
                        drop(state);
                        publish_current_market_subscription_if_changed(
                            &market_sender,
                            subscription.as_ref(),
                        );
                    }
                }

                if !resolution_watches_seeded {
                    match repository
                        .seed_recent_official_resolution_watches(now, resolution_retention)
                        .await
                    {
                        Ok(_) => match repository
                            .load_unsettled_official_resolution_watches()
                            .await
                        {
                            Ok(watches) => {
                                metrics.write().await.resolution_watches_rehydrated =
                                    watches.len() as u64;
                                resolution_watches_seeded = true;
                            }
                            Err(error) => {
                                record_critical_persistence_error(&metrics, error).await;
                            }
                        },
                        Err(error) => {
                            record_critical_persistence_error(&metrics, error).await;
                        }
                    }
                }

                let watches = match repository
                    .load_unsettled_official_resolution_watches()
                    .await
                {
                    Ok(watches) => watches,
                    Err(error) => {
                        record_critical_persistence_error(&metrics, error).await;
                        continue;
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
                    continue;
                }
                let expired = match repository
                    .expire_overdue_official_resolution_watches(Utc::now())
                    .await
                {
                    Ok(expired) => expired,
                    Err(error) => {
                        record_critical_persistence_error(&metrics, error).await;
                        continue;
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
                        continue;
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
                    continue;
                }

                // Boundary recovery is broader than trading and independent of whether an
                // official result has arrived. Fresh previous/current/next plus durable pending
                // rows preserve local labels across restart and delayed settlement.
                let mut recovery_markets = pending_markets.clone();
                let mut recovery_ids = recovery_markets
                    .iter()
                    .map(|market| market.market_id.clone())
                    .collect::<HashSet<_>>();
                for market in fresh_market_slice {
                    if recovery_ids.insert(market.market_id.clone()) {
                        recovery_markets.push(market.clone());
                    }
                }
                recovery_markets.sort_by_key(|market| market.window_start);
                {
                    let mut runtime_metrics = metrics.write().await;
                    if let Some(fresh_markets) = fresh_markets.as_ref() {
                        runtime_metrics.markets_discovered = fresh_markets.len() as u64;
                    }
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
                                continue;
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
                    continue;
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
                    continue;
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

#[derive(Debug, Clone, PartialEq)]
struct CurrentMarketUpdate {
    tradable: Option<BtcIntervalMarket>,
    display: Option<BtcIntervalMarket>,
}

/// Returns `None` when discovery failed and the caller must retain its prior markets.
/// A successful response always returns an update. Display identity follows the clock
/// window while trading identity additionally honors active/closed/accepting flags.
fn current_market_update(
    fresh_markets: Option<&[BtcIntervalMarket]>,
    now: DateTime<Utc>,
) -> Option<CurrentMarketUpdate> {
    fresh_markets.map(|markets| CurrentMarketUpdate {
        tradable: markets
            .iter()
            .find(|market| market.is_trade_window(now))
            .cloned(),
        display: markets
            .iter()
            .find(|market| market.is_interval_window(now))
            .cloned(),
    })
}

#[derive(Debug)]
enum ClobEpochUpdateError {
    Shutdown,
    TargetRejected(String),
    Transport(String),
}

async fn update_clob_epoch_subscriptions(
    epoch: &mut ClobEpoch,
    desired_markets: &[BtcIntervalMarket],
    shutdown: &mut watch::Receiver<bool>,
) -> std::result::Result<ClobSubscriptionDelta, ClobEpochUpdateError> {
    validate_clob_execution_market_set(desired_markets)
        .map_err(|error| ClobEpochUpdateError::TargetRejected(error.to_string()))?;
    epoch
        .registry
        .validate_market_set(desired_markets)
        .map_err(|error| {
            ClobEpochUpdateError::TargetRejected(format!("invalid_subscription_transition:{error}"))
        })?;
    let delta = clob_subscription_delta(&epoch.markets, desired_markets);
    let mut next_registry = epoch.registry.clone();
    register_clob_markets(&mut next_registry, &delta.added_markets).map_err(|error| {
        ClobEpochUpdateError::TargetRejected(format!("subscription_registration_failed:{error}"))
    })?;
    next_registry
        .retain_markets(desired_markets)
        .map_err(|error| {
            ClobEpochUpdateError::TargetRejected(format!("subscription_retention_failed:{error}"))
        })?;
    if !delta.removed_assets.is_empty() {
        let payload = clob_subscription_operation(
            &delta.removed_assets,
            ClobSubscriptionOperation::Unsubscribe,
        );
        let mut sink = epoch.transport.sink.lock().await;
        match send_clob_text(&mut *sink, payload, shutdown).await {
            Ok(()) => {}
            Err(ClobSendFailure::Shutdown) => return Err(ClobEpochUpdateError::Shutdown),
            Err(ClobSendFailure::Timeout) => {
                return Err(ClobEpochUpdateError::Transport(
                    "dynamic_unsubscribe_timeout".to_string(),
                ));
            }
            Err(ClobSendFailure::Transport { error, detail }) => {
                epoch.telemetry.last_transport_error = Some(detail);
                return Err(ClobEpochUpdateError::Transport(bounded_clob_error_reason(
                    &format!("dynamic_unsubscribe_failed:{error}"),
                )));
            }
        }
    }
    if !delta.added_assets.is_empty() {
        let payload =
            clob_subscription_operation(&delta.added_assets, ClobSubscriptionOperation::Subscribe);
        let mut sink = epoch.transport.sink.lock().await;
        match send_clob_text(&mut *sink, payload, shutdown).await {
            Ok(()) => {}
            Err(ClobSendFailure::Shutdown) => return Err(ClobEpochUpdateError::Shutdown),
            Err(ClobSendFailure::Timeout) => {
                return Err(ClobEpochUpdateError::Transport(
                    "dynamic_subscribe_timeout".to_string(),
                ));
            }
            Err(ClobSendFailure::Transport { error, detail }) => {
                epoch.telemetry.last_transport_error = Some(detail);
                return Err(ClobEpochUpdateError::Transport(bounded_clob_error_reason(
                    &format!("dynamic_subscribe_failed:{error}"),
                )));
            }
        }
    }
    epoch.registry = next_registry;
    epoch.markets = desired_markets.to_vec();
    let updated_at = Utc::now();
    if !delta.is_empty() {
        epoch.telemetry.refresh_subscription_target(desired_markets);
        epoch.subscription_stats.updates = epoch.subscription_stats.updates.saturating_add(1);
        epoch.subscription_stats.active_assets = epoch.registry.len();
        epoch.subscription_stats.last_updated_at = Some(updated_at);
    }
    epoch
        .watchdog
        .refresh_bootstrap(Instant::now(), &epoch.registry, &epoch.markets, updated_at);
    Ok(delta)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClobFrameAction {
    Continue,
    Disconnect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClobFrameOutcome {
    action: ClobFrameAction,
    readiness_may_have_changed: bool,
}

impl ClobFrameOutcome {
    const fn continue_with(readiness_may_have_changed: bool) -> Self {
        Self {
            action: ClobFrameAction::Continue,
            readiness_may_have_changed,
        }
    }

    const fn disconnect() -> Self {
        Self {
            action: ClobFrameAction::Disconnect,
            readiness_may_have_changed: false,
        }
    }
}

async fn publish_clob_registry(registry: &BookRegistry, shared_books: &Arc<RwLock<BookRegistry>>) {
    *shared_books.write().await = registry.clone();
}

fn finish_active_clob_frame(
    epoch: &mut ClobEpoch,
    processing_started: Instant,
    shared_books_lock_wait: StdDuration,
) {
    epoch.telemetry.record_frame_processing(
        Instant::now().saturating_duration_since(processing_started),
        shared_books_lock_wait,
    );
    epoch.telemetry.ingress_queue_depth = epoch.transport.events.len();
}

async fn flush_clob_frame_metrics(epoch: &mut ClobEpoch, metrics: &Arc<RwLock<BtcRuntimeMetrics>>) {
    let pending = std::mem::take(&mut epoch.pending_frame_metrics);
    let mut runtime_metrics = metrics.write().await;
    runtime_metrics.clob_messages_received = runtime_metrics
        .clob_messages_received
        .saturating_add(pending.messages_received);
    runtime_metrics.feed_events_applied = runtime_metrics
        .feed_events_applied
        .saturating_add(pending.events_applied);
    runtime_metrics.integrity_gaps = runtime_metrics
        .integrity_gaps
        .saturating_add(pending.integrity_gaps);
    if let Some(lag_milliseconds) = pending.last_source_to_receive_lag_milliseconds {
        runtime_metrics.clob_active_last_source_to_receive_lag_milliseconds =
            Some(lag_milliseconds);
    }
    publish_active_clob_socket_metrics(&mut runtime_metrics, &epoch.telemetry);
}

async fn apply_active_clob_frame(
    epoch: &mut ClobEpoch,
    frame: ClobIngressFrame,
    shared_books: &Arc<RwLock<BookRegistry>>,
    metrics: &Arc<RwLock<BtcRuntimeMetrics>>,
) -> ClobFrameOutcome {
    let processing_started = Instant::now();
    epoch
        .telemetry
        .record_ingress_frame(&frame, processing_started);
    epoch.telemetry.last_shared_books_lock_wait = StdDuration::ZERO;
    let ClobIngressFrame {
        message,
        received_at,
        received_instant,
        ..
    } = frame;
    let pong_round_trip = match &message {
        Message::Text(text) => epoch
            .watchdog
            .acknowledge_text_pong(text.as_str(), received_instant),
        _ => None,
    };
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
                return ClobFrameOutcome::continue_with(false);
            }
            if text.trim().is_empty() {
                return ClobFrameOutcome::continue_with(false);
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
            return ClobFrameOutcome::disconnect();
        }
        _ => return ClobFrameOutcome::continue_with(false),
    };
    epoch.telemetry.last_data_or_heartbeat_at = Some(received_at);
    epoch.pending_frame_metrics.messages_received = epoch
        .pending_frame_metrics
        .messages_received
        .saturating_add(1);
    let messages = match parsed {
        Ok(messages) => messages,
        Err(error) => {
            epoch.registry.quarantine(FeedIntegrityStatus::DecodeError);
            let lock_wait_started = Instant::now();
            let mut published_books = shared_books.write().await;
            let shared_books_lock_wait =
                Instant::now().saturating_duration_since(lock_wait_started);
            *published_books = epoch.registry.clone();
            drop(published_books);
            {
                let mut runtime_metrics = metrics.write().await;
                runtime_metrics.decode_errors = runtime_metrics.decode_errors.saturating_add(1);
            }
            record_error(metrics, error).await;
            epoch.telemetry.last_shared_books_lock_wait = shared_books_lock_wait;
            return ClobFrameOutcome::continue_with(true);
        }
    };
    let mut applied_count = 0_u64;
    let mut integrity_gap_count = 0_u64;
    let mut source_to_receive_lag_milliseconds: Option<i64> = None;
    let mut mutated_token_ids = Vec::<String>::new();
    for message in messages {
        for event in epoch.registry.apply(message, received_at) {
            if event.applied {
                applied_count = applied_count.saturating_add(1);
                let lag_milliseconds = (received_at - event.source_timestamp)
                    .num_milliseconds()
                    .max(0);
                source_to_receive_lag_milliseconds = Some(
                    source_to_receive_lag_milliseconds
                        .unwrap_or_default()
                        .max(lag_milliseconds),
                );
            } else {
                integrity_gap_count = integrity_gap_count.saturating_add(1);
            }
            if event.book_mutated {
                if let Some(token_id) = event.token_id {
                    if !mutated_token_ids
                        .iter()
                        .any(|existing| existing == &token_id)
                    {
                        mutated_token_ids.push(token_id);
                    }
                }
            }
        }
    }
    epoch.pending_frame_metrics.events_applied = epoch
        .pending_frame_metrics
        .events_applied
        .saturating_add(applied_count);
    epoch.pending_frame_metrics.integrity_gaps = epoch
        .pending_frame_metrics
        .integrity_gaps
        .saturating_add(integrity_gap_count);
    if source_to_receive_lag_milliseconds.is_some() {
        epoch
            .pending_frame_metrics
            .last_source_to_receive_lag_milliseconds = source_to_receive_lag_milliseconds;
    }
    let mut shared_books_lock_wait = StdDuration::ZERO;
    if !mutated_token_ids.is_empty() {
        let lock_wait_started = Instant::now();
        let mut published_books = shared_books.write().await;
        shared_books_lock_wait = Instant::now().saturating_duration_since(lock_wait_started);
        published_books.publish_frame_books_from(
            &epoch.registry,
            mutated_token_ids.iter().map(String::as_str),
        );
    }
    epoch.telemetry.last_shared_books_lock_wait = shared_books_lock_wait;
    ClobFrameOutcome::continue_with(!mutated_token_ids.is_empty())
}

#[derive(Debug)]
enum ClobCloseOutcome {
    Skipped,
    Completed,
    TimedOut,
    Failed(ClobTransportErrorDetail),
}

async fn attempt_clob_close<S>(socket: &mut S, action: ClobCloseAction) -> ClobCloseOutcome
where
    S: Sink<Message, Error = Error> + Unpin,
{
    let result = match action {
        ClobCloseAction::Skip => return ClobCloseOutcome::Skipped,
        ClobCloseAction::Initiate => timeout(CLOB_GRACEFUL_CLOSE_TIMEOUT, socket.close()).await,
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

fn clob_stop_close_action(telemetry: &ClobSocketTelemetry) -> ClobCloseAction {
    if telemetry.remote_close_observed {
        ClobCloseAction::AcknowledgeRemote
    } else {
        ClobCloseAction::Initiate
    }
}

async fn complete_clob_epoch_with_close(
    epoch: &mut ClobEpoch,
    reason: String,
    cause: ClobDisconnectCause,
    retry_action: ClobRetryAction,
    consecutive_failures: u32,
    close_action: ClobCloseAction,
) {
    let disconnected_instant = Instant::now();
    epoch.transport.reader_task.abort();
    epoch.transport.heartbeat_task.abort();
    let mut sink = epoch.transport.sink.lock().await;
    match attempt_clob_close(&mut *sink, close_action).await {
        ClobCloseOutcome::Skipped => {}
        ClobCloseOutcome::Completed => {
            tracing::debug!(
                feed = "polymarket_clob_market",
                connection_id = %epoch.connection_id,
                connection_epoch = epoch.connection_epoch,
                close_action = close_action.as_str(),
                "CLOB websocket close action completed"
            );
        }
        ClobCloseOutcome::TimedOut => {
            tracing::warn!(
                feed = "polymarket_clob_market",
                connection_id = %epoch.connection_id,
                connection_epoch = epoch.connection_epoch,
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
                close_action = close_action.as_str(),
                transport_error_class = detail.class,
                transport_io_kind = ?detail.io_kind.as_deref(),
                transport_os_error_code = ?detail.os_error_code,
                "CLOB websocket close action failed"
            );
        }
    }
    let reason = bounded_clob_error_reason(&reason);
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
        ClobDisconnectCause::Shutdown => {}
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

async fn run_clob_supervisor(
    config: BtcRuntimeConfig,
    heartbeat_interval: StdDuration,
    pong_timeout: StdDuration,
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
    let mut connect_task: Option<JoinHandle<ClobConnectOutcome>> = None;
    let mut connecting_markets: Option<Vec<BtcIntervalMarket>> = None;
    let mut connection_epoch = 0i32;
    let mut consecutive_failures = 0u32;
    let mut retry_at = Instant::now();
    let mut recovery_window = ClobRecoveryWindow::open(Utc::now(), Instant::now());
    metrics.write().await.clob_recovery_unavailable_since = recovery_window.since;

    let started_at = Instant::now();
    let mut checkpoints = interval_at(
        started_at + config.checkpoint_interval,
        config.checkpoint_interval,
    );
    checkpoints.set_missed_tick_behavior(MissedTickBehavior::Delay);

    let dormant_deadline = Instant::now() + StdDuration::from_secs(86_400);
    let bootstrap_sleep = sleep_until(dormant_deadline);
    let pong_sleep = sleep_until(dormant_deadline);
    let read_sleep = sleep_until(dormant_deadline);
    let retry_sleep = sleep_until(dormant_deadline);
    tokio::pin!(bootstrap_sleep, pong_sleep, read_sleep, retry_sleep);

    let mut market_watch_open = true;
    loop {
        if *shutdown.borrow() {
            break;
        }

        if should_connect_clob(
            active.is_some(),
            connect_task.is_some(),
            !markets.borrow().is_empty(),
            Instant::now(),
            retry_at,
        ) {
            connection_epoch = connection_epoch.saturating_add(1);
            let desired_markets = markets.borrow().clone();
            connecting_markets = Some(desired_markets.clone());
            connect_task = Some(tokio::spawn(connect_clob_epoch(
                config.clone(),
                desired_markets,
                connection_epoch,
                heartbeat_interval,
                shutdown.clone(),
            )));
        }

        if let Some(epoch) = active.as_mut() {
            let latest_receipt = epoch.transport.latest_receipt_instant();
            epoch.watchdog.observe_transport_receipt(latest_receipt);
        }

        let bootstrap_deadline = active
            .as_ref()
            .and_then(|epoch| epoch.watchdog.bootstrap_deadline);
        let pong_deadline = active
            .as_ref()
            .and_then(|epoch| epoch.watchdog.pong_deadline);
        let read_deadline = active
            .as_ref()
            .map(|epoch| epoch.watchdog.read_idle_deadline);
        let retry_deadline =
            (active.is_none() && connect_task.is_none() && !markets.borrow().is_empty())
                .then_some(retry_at);

        if let Some(deadline) = bootstrap_deadline {
            bootstrap_sleep.as_mut().reset(deadline);
        }
        if let Some(deadline) = pong_deadline {
            pong_sleep.as_mut().reset(deadline);
        }
        if let Some(deadline) = read_deadline {
            read_sleep.as_mut().reset(deadline);
        }
        if let Some(deadline) = retry_deadline {
            retry_sleep.as_mut().reset(deadline);
        }

        let mut active_failure: Option<(String, ClobDisconnectCause)> = None;
        let mut shutdown_requested = false;

        tokio::select! {
            _ = shutdown.changed() => {
                shutdown_requested = true;
            }
            _ = &mut bootstrap_sleep, if bootstrap_deadline.is_some() => {
                if let Some(epoch) = active.as_mut() {
                    epoch.watchdog.bootstrap_deadline = None;
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.clob_bootstrap_failures = runtime_metrics
                        .clob_bootstrap_failures
                        .saturating_add(1);
                    runtime_metrics.last_error = Some(
                        "CLOB books did not bootstrap before the readiness deadline; transport retained"
                            .to_string(),
                    );
                    tracing::warn!(
                        feed = "polymarket_clob_market",
                        connection_id = %epoch.connection_id,
                        connection_epoch = epoch.connection_epoch,
                        "CLOB bootstrap deadline elapsed; transport retained"
                    );
                }
            }
            _ = &mut pong_sleep, if pong_deadline.is_some() => {
                if let Some(epoch) = active.as_mut() {
                    let latest_receipt = epoch.transport.latest_receipt_instant();
                    epoch.watchdog.observe_transport_receipt(latest_receipt);
                    if epoch
                        .watchdog
                        .pong_deadline
                        .is_some_and(|deadline| deadline <= Instant::now())
                    {
                        active_failure = Some((
                            "heartbeat_ack_timeout".to_string(),
                            ClobDisconnectCause::TransportFailure,
                        ));
                    }
                }
            }
            _ = &mut read_sleep, if read_deadline.is_some() => {
                if let Some(epoch) = active.as_mut() {
                    let latest_receipt = epoch.transport.latest_receipt_instant();
                    epoch.watchdog.observe_transport_receipt(latest_receipt);
                    if epoch.watchdog.read_idle_deadline <= Instant::now() {
                        active_failure = Some((
                            "read_idle_timeout".to_string(),
                            ClobDisconnectCause::TransportFailure,
                        ));
                    }
                }
            }
            changed = markets.changed(), if market_watch_open => {
                match market_watch_disposition(&changed) {
                    MarketWatchDisposition::RetainCurrent => {
                        market_watch_open = false;
                        tracing::warn!(
                            feed = "polymarket_clob_market",
                            "CLOB market discovery watch closed; retaining the last valid subscription"
                        );
                    }
                    MarketWatchDisposition::UpdateSubscriptions => {
                        let desired_markets = markets.borrow().clone();
                        if connecting_markets.as_ref().is_some_and(|connecting| {
                            !same_market_subscriptions(connecting, &desired_markets)
                        }) {
                            if let Some(task) = connect_task.take() {
                                task.abort();
                                let _ = task.await;
                            }
                            connecting_markets = None;
                            retry_at = Instant::now();
                        }
                        if let Some(epoch) = active.as_mut() {
                            match update_clob_epoch_subscriptions(
                                epoch,
                                &desired_markets,
                                &mut shutdown,
                            )
                            .await
                            {
                                Ok(delta) => {
                                    let updated_at = Utc::now();
                                    publish_clob_registry(&epoch.registry, &shared_books).await;
                                    flush_clob_frame_metrics(epoch, &metrics).await;
                                    if !delta.is_empty() {
                                        let mut runtime_metrics = metrics.write().await;
                                        runtime_metrics.clob_subscription_updates = runtime_metrics
                                            .clob_subscription_updates
                                            .saturating_add(1);
                                        runtime_metrics.clob_active_subscribed_assets =
                                            u64::try_from(epoch.registry.len()).unwrap_or(u64::MAX);
                                        runtime_metrics.clob_last_subscription_update_at =
                                            Some(updated_at);
                                        runtime_metrics
                                            .clob_active_subscription_target_fingerprint_sha256 =
                                            Some(
                                                epoch
                                                    .telemetry
                                                    .subscription_target_fingerprint_sha256
                                                    .clone(),
                                            );
                                    }
                                    update_clob_usability(
                                        &epoch.registry,
                                        &epoch.markets,
                                        updated_at,
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
                                Err(ClobEpochUpdateError::Shutdown) => shutdown_requested = true,
                                Err(ClobEpochUpdateError::Transport(reason)) => {
                                    active_failure = Some((
                                        reason,
                                        ClobDisconnectCause::SubscriptionFailure,
                                    ));
                                }
                                Err(ClobEpochUpdateError::TargetRejected(reason)) => {
                                    record_error(&metrics, anyhow::anyhow!(reason)).await;
                                }
                            }
                        }
                    }
                }
            }
            event = async {
                active
                    .as_mut()
                    .expect("CLOB socket branch is guarded")
                    .transport
                    .events
                    .recv()
                    .await
            }, if active.is_some() => {
                match event {
                    None => {
                        let epoch = active.as_mut().expect("CLOB epoch remains installed");
                        if epoch.transport.overflowed.load(Ordering::Acquire) {
                            epoch.telemetry.record_ingress_overflow();
                            let mut runtime_metrics = metrics.write().await;
                            publish_active_clob_socket_metrics(
                                &mut runtime_metrics,
                                &epoch.telemetry,
                            );
                            active_failure = Some((
                                "ingress_queue_overflow".to_string(),
                                ClobDisconnectCause::TransportFailure,
                            ));
                        } else {
                            epoch.telemetry.last_transport_error =
                                Some(ClobTransportErrorDetail::websocket_eof());
                            active_failure = Some((
                                "websocket_eof".to_string(),
                                ClobDisconnectCause::TransportFailure,
                            ));
                        }
                    }
                    Some(ClobIngressEvent::HeartbeatSent {
                        observed_at,
                        scheduled_at,
                        sent_at,
                    }) => {
                        let epoch = active.as_mut().expect("CLOB epoch remains installed");
                        epoch.watchdog.record_text_ping(sent_at, pong_timeout);
                        epoch
                            .watchdog
                            .observe_transport_receipt(epoch.transport.latest_receipt_instant());
                        let sample = epoch.telemetry.record_heartbeat_probe(
                            observed_at,
                            sent_at,
                            sent_at.saturating_duration_since(scheduled_at),
                        );
                        record_active_clob_heartbeat_probe_metrics(&metrics, sample).await;
                    }
                    Some(ClobIngressEvent::TransportError { reason, detail }) => {
                        active
                            .as_mut()
                            .expect("CLOB epoch remains installed")
                            .telemetry
                            .last_transport_error = Some(detail);
                        active_failure = Some((reason, ClobDisconnectCause::TransportFailure));
                    }
                    Some(ClobIngressEvent::Eof) => {
                        active
                            .as_mut()
                            .expect("CLOB epoch remains installed")
                            .telemetry
                            .last_transport_error =
                            Some(ClobTransportErrorDetail::websocket_eof());
                        active_failure = Some((
                            "websocket_eof".to_string(),
                            ClobDisconnectCause::TransportFailure,
                        ));
                    }
                    Some(ClobIngressEvent::Frame(frame)) => {
                        let epoch = active.as_mut().expect("CLOB epoch remains installed");
                        let processing_started = Instant::now();
                        let outcome = apply_active_clob_frame(
                            epoch,
                            frame,
                            &shared_books,
                            &metrics,
                        )
                        .await;
                        match outcome.action {
                            ClobFrameAction::Disconnect => {
                                let shared_books_lock_wait =
                                    epoch.telemetry.last_shared_books_lock_wait;
                                finish_active_clob_frame(
                                    epoch,
                                    processing_started,
                                    shared_books_lock_wait,
                                );
                                active_failure = Some((
                                    "remote_close".to_string(),
                                    ClobDisconnectCause::TransportFailure,
                                ));
                            }
                            ClobFrameAction::Continue => {
                                if outcome.readiness_may_have_changed {
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
                                let shared_books_lock_wait =
                                    epoch.telemetry.last_shared_books_lock_wait;
                                finish_active_clob_frame(
                                    epoch,
                                    processing_started,
                                    shared_books_lock_wait,
                                );
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
                        let retry_action =
                            clob_retry_action(&config, false, &mut consecutive_failures);
                        let ClobRetryAction::Backoff(delay) = retry_action else {
                            unreachable!("failed CLOB connect task must back off")
                        };
                        retry_at = Instant::now() + delay;
                        record_error(
                            &metrics,
                            anyhow::anyhow!("CLOB connection task failed: {error}"),
                        )
                        .await;
                        let mut runtime_metrics = metrics.write().await;
                        clob_failure_metrics(
                            &mut runtime_metrics,
                            ClobDisconnectCause::ConnectFailure,
                        );
                        clob_retry_metrics(&mut runtime_metrics, retry_action);
                    }
                    Ok(ClobConnectOutcome::Failed(failure)) => {
                        let ClobConnectFailure {
                            reason,
                            kind,
                            telemetry,
                        } = *failure;
                        if kind == ClobConnectFailureKind::Shutdown {
                            shutdown_requested = true;
                        } else {
                            let cause = match kind {
                                ClobConnectFailureKind::Connect => {
                                    ClobDisconnectCause::ConnectFailure
                                }
                                ClobConnectFailureKind::Subscription => {
                                    ClobDisconnectCause::SubscriptionFailure
                                }
                                ClobConnectFailureKind::Identity => {
                                    ClobDisconnectCause::BootstrapFailure
                                }
                                ClobConnectFailureKind::Shutdown => unreachable!(),
                            };
                            let retry_action =
                                clob_retry_action(&config, false, &mut consecutive_failures);
                            let ClobRetryAction::Backoff(delay) = retry_action else {
                                unreachable!("failed CLOB connect attempt must back off")
                            };
                            retry_at = Instant::now() + delay;
                            {
                                let mut runtime_metrics = metrics.write().await;
                                clob_failure_metrics(&mut runtime_metrics, cause);
                                clob_retry_metrics(&mut runtime_metrics, retry_action);
                            }
                            let transport_error = telemetry.last_transport_error.as_ref();
                            tracing::warn!(
                                feed = "polymarket_clob_market",
                                %reason,
                                consecutive_failures,
                                retry_delay_ms = duration_milliseconds(delay),
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
                                "CLOB connection attempt failed"
                            );
                        }
                    }
                    Ok(ClobConnectOutcome::Connected {
                        mut epoch,
                        connect_latency,
                    }) => {
                        epoch.subscription_stats.active_assets = epoch.registry.len();
                        if !same_market_subscriptions(&epoch.markets, &markets.borrow()) {
                            complete_clob_epoch_with_close(
                                &mut epoch,
                                "subscription_target_changed_during_connect".to_string(),
                                ClobDisconnectCause::SubscriptionFailure,
                                ClobRetryAction::ImmediateRecovery,
                                consecutive_failures,
                                ClobCloseAction::Initiate,
                            )
                            .await;
                            retry_at = Instant::now();
                        } else {
                            publish_clob_registry(&epoch.registry, &shared_books).await;
                            {
                                let mut runtime_metrics = metrics.write().await;
                                runtime_metrics.clob_connections_established = runtime_metrics
                                    .clob_connections_established
                                    .saturating_add(1);
                                runtime_metrics.clob_connected_connection_epoch =
                                    Some(epoch.connection_epoch);
                                runtime_metrics.clob_connected_connection_id =
                                    Some(epoch.connection_id);
                                runtime_metrics.clob_last_connected_at = Some(epoch.connected_at);
                                runtime_metrics.clob_active_subscribed_assets =
                                    u64::try_from(epoch.registry.len()).unwrap_or(u64::MAX);
                                publish_active_clob_socket_metrics(
                                    &mut runtime_metrics,
                                    &epoch.telemetry,
                                );
                            }
                            tracing::info!(
                                feed = "polymarket_clob_market",
                                connection_id = %epoch.connection_id,
                                connection_epoch = epoch.connection_epoch,
                                connect_latency_ms = duration_milliseconds(connect_latency),
                                peer_address = ?epoch.telemetry.provenance.peer_address,
                                edge_request_id = ?epoch.telemetry.provenance.edge_request_id,
                                edge_server = ?epoch.telemetry.provenance.edge_server,
                                handshake_date = ?epoch.telemetry.provenance.handshake_date,
                                subscription_target_fingerprint_sha256 =
                                    %epoch.telemetry.subscription_target_fingerprint_sha256,
                                "CLOB market websocket connected"
                            );
                            active = Some(*epoch);
                        }
                    }
                }
            }
            _ = checkpoints.tick() => {
                let checked_at = Utc::now();
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
                    flush_clob_frame_metrics(epoch, &metrics).await;
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
                                    PersistEnqueueOutcome::Saturated
                                    | PersistEnqueueOutcome::Closed => {}
                                }
                            }
                        }
                    }
                }
            }
            _ = &mut retry_sleep, if retry_deadline.is_some() => {}
        }

        if shutdown_requested {
            break;
        }

        if let Some((reason, cause)) = active_failure {
            let Some(mut failed) = active.take() else {
                continue;
            };
            flush_clob_frame_metrics(&mut failed, &metrics).await;
            let retry_action =
                clob_retry_action(&config, failed.healthy_epoch, &mut consecutive_failures);
            retry_at = match retry_action {
                ClobRetryAction::ImmediateRecovery => Instant::now(),
                ClobRetryAction::Backoff(delay) => Instant::now() + delay,
                ClobRetryAction::Stop => Instant::now(),
            };
            let unavailable_at = Utc::now();
            {
                let mut runtime_metrics = metrics.write().await;
                clob_disconnect_metrics(&mut runtime_metrics, cause, retry_action);
                runtime_metrics.clob_last_disconnect_at = Some(unavailable_at);
                runtime_metrics.clob_last_disconnect_reason = Some(reason.clone());
                record_clob_remote_close_1013(
                    &mut runtime_metrics,
                    &failed.telemetry,
                    unavailable_at,
                );
            }
            let unavailable_instant = Instant::now();
            quarantine_clob_books_on_disconnect(&mut failed.registry, &shared_books).await;
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
            let close_action = clob_failure_close_action(&failed.telemetry);
            complete_clob_epoch_with_close(
                &mut failed,
                reason,
                cause,
                retry_action,
                consecutive_failures,
                close_action,
            )
            .await;
        }
    }

    if let Some(task) = connect_task.take() {
        task.abort();
        let _ = task.await;
    }
    let stop_cause = ClobDisconnectCause::Shutdown;
    let stop_reason = "shutdown";
    if let Some(mut epoch) = active.take() {
        flush_clob_frame_metrics(&mut epoch, &metrics).await;
        quarantine_clob_books_on_disconnect(&mut epoch.registry, &shared_books).await;
        let close_action = clob_stop_close_action(&epoch.telemetry);
        complete_clob_epoch_with_close(
            &mut epoch,
            stop_reason.to_string(),
            stop_cause,
            ClobRetryAction::Stop,
            consecutive_failures,
            close_action,
        )
        .await;
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarketWatchDisposition {
    UpdateSubscriptions,
    RetainCurrent,
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
}

impl ClobDisconnectCause {
    fn as_str(self) -> &'static str {
        match self {
            Self::ConnectFailure => "connect_failure",
            Self::SubscriptionFailure => "subscription_failure",
            Self::BootstrapFailure => "bootstrap_failure",
            Self::TransportFailure => "transport_failure",
            Self::Shutdown => "shutdown",
        }
    }
}

fn clob_retry_action(
    config: &BtcRuntimeConfig,
    healthy_epoch: bool,
    consecutive_failures: &mut u32,
) -> ClobRetryAction {
    if healthy_epoch {
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
        MarketWatchDisposition::RetainCurrent
    }
}

async fn quarantine_clob_books_on_disconnect(
    registry: &mut BookRegistry,
    shared_books: &Arc<RwLock<BookRegistry>>,
) {
    registry.quarantine(FeedIntegrityStatus::Stale);
    publish_clob_registry(registry, shared_books).await;
}

#[cfg(test)]
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
            Some("canonical_market_identity_mismatch")
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
    if let Some(identity) = registry.market_book_identity_diagnostic(market) {
        return Some(ClobReadinessDiagnostic {
            reason: identity.reason,
            market_id: Some(market.market_id.clone()),
            token_id: Some(identity.token_id),
            integrity_status: None,
            bootstrapped: None,
            has_bid: None,
            has_ask: None,
            source_age_milliseconds: None,
            receipt_age_milliseconds: None,
            source_to_receive_lag_milliseconds: None,
        });
    }
    Some(ClobReadinessDiagnostic {
        reason: "unclassified_book_readiness_failure",
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

fn should_connect_clob(
    active_present: bool,
    connect_in_flight: bool,
    has_desired_markets: bool,
    now: Instant,
    retry_at: Instant,
) -> bool {
    !active_present && !connect_in_flight && has_desired_markets && now >= retry_at
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
    metrics.clob_active_ingress_frames = 0;
    metrics.clob_active_ingress_bytes = 0;
    metrics.clob_active_last_ingress_sequence = 0;
    metrics.clob_active_ingress_queue_depth = 0;
    metrics.clob_active_last_ingress_queue_dwell_milliseconds = 0;
    metrics.clob_active_max_ingress_queue_dwell_milliseconds = 0;
    metrics.clob_active_ingress_overflows = 0;
    metrics.clob_active_last_frame_processing_milliseconds = 0;
    metrics.clob_active_max_frame_processing_milliseconds = 0;
    metrics.clob_active_last_shared_books_lock_wait_milliseconds = 0;
    metrics.clob_active_max_shared_books_lock_wait_milliseconds = 0;
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
                ingress_frames = telemetry.ingress_frames,
                ingress_bytes = telemetry.ingress_bytes,
                last_ingress_sequence = telemetry.last_ingress_sequence,
                last_ingress_queue_dwell_ms =
                    duration_milliseconds(telemetry.last_ingress_queue_dwell),
                max_ingress_queue_dwell_ms =
                    duration_milliseconds(telemetry.max_ingress_queue_dwell),
                ingress_overflows = telemetry.ingress_overflows,
                last_frame_processing_ms = duration_milliseconds(telemetry.last_frame_processing),
                max_frame_processing_ms = duration_milliseconds(telemetry.max_frame_processing),
                last_shared_books_lock_wait_ms =
                    duration_milliseconds(telemetry.last_shared_books_lock_wait),
                max_shared_books_lock_wait_ms =
                    duration_milliseconds(telemetry.max_shared_books_lock_wait),
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
        ClobDisconnectCause::Shutdown => emit_disconnect!(tracing::Level::INFO),
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
                                                        PersistEnqueueOutcome::Queued
                                                        | PersistEnqueueOutcome::Saturated => {}
                                                        PersistEnqueueOutcome::Closed => {
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
                                                PersistEnqueueOutcome::Queued
                                                | PersistEnqueueOutcome::Saturated => {}
                                                PersistEnqueueOutcome::Closed => {
                                                    disconnect_reason = ReferenceDisconnectReason::CriticalWriterQueue;
                                                    fatal_persistence_error = Some(anyhow::anyhow!(
                                                        "Binance reference persistence queue closed"
                                                    ));
                                                    break 'connection;
                                                }
                                            }
                                        }
                                        Err(error) => {
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
    books: Arc<RwLock<BookRegistry>>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
    mut shutdown: watch::Receiver<bool>,
) {
    if !config.enabled {
        return;
    }
    let mut ticker = interval(config.strategy_interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut last_observation = None;
    loop {
        tokio::select! {
            _ = shutdown.changed() => break,
            _ = ticker.tick() => {
                if let Err(error) = strategy.reconcile_if_due().await {
                    tracing::warn!(
                        error = %error,
                        "BTC reconciliation maintenance failed; runtime remains active"
                    );
                }
                let snapshot = realtime_snapshot(&state, &books).await;
                let observation = (
                    snapshot.last_updated_at,
                    snapshot.books.values().cloned().collect::<Vec<_>>(),
                );
                if last_observation.as_ref() == Some(&observation) {
                    continue;
                }
                last_observation = Some(observation);
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
                    if is_retryable_strategy_database_error(&error) {
                        tracing::warn!(
                            error = %format!("{error:#}"),
                            "BTC strategy database dependency unavailable; skipping this evaluation and preserving the trading process"
                        );
                        last_observation = None;
                        continue;
                    }
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

fn validate_clob_execution_market_set(markets: &[BtcIntervalMarket]) -> Result<()> {
    if markets.len() > 1 {
        bail!("CLOB execution socket accepts at most one current market");
    }
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

fn publish_current_market_subscription_if_changed(
    sender: &watch::Sender<Vec<BtcIntervalMarket>>,
    market: Option<&BtcIntervalMarket>,
) {
    let markets = market.cloned().into_iter().collect::<Vec<_>>();
    if !same_market_subscriptions(&sender.borrow(), &markets) {
        let _ = sender.send(markets);
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

fn is_retryable_strategy_database_error(error: &anyhow::Error) -> bool {
    let Some(sqlx_error) = error.downcast_ref::<sqlx::Error>() else {
        return false;
    };
    match sqlx_error {
        sqlx::Error::Database(error) => {
            let message = error.message().trim();
            error
                .code()
                .as_deref()
                .is_some_and(retryable_postgres_primary_persistence_sqlstate)
                || message == "query_wait_timeout"
                || message.starts_with("query_wait_timeout:")
                || message == "sorry, too many clients already"
        }
        sqlx::Error::Io(_) | sqlx::Error::Tls(_) | sqlx::Error::PoolTimedOut => true,
        _ => false,
    }
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
    use crate::btc::feeds::ClobMessage;
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
            },
            source_at + Duration::milliseconds(1),
        );
        assert!(events.iter().all(|event| event.applied));
    }

    async fn clob_socket_pair() -> (ClobSocket, WebSocketStream<TcpStream>) {
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let (client, server) =
            tokio::join!(tokio::net::TcpStream::connect(address), listener.accept());
        let client = client.unwrap();
        let (server, _) = server.unwrap();
        let client = WebSocketStream::from_raw_socket(
            MaybeTlsStream::Plain(client),
            tokio_tungstenite::tungstenite::protocol::Role::Client,
            None,
        )
        .await;
        let server = WebSocketStream::from_raw_socket(
            server,
            tokio_tungstenite::tungstenite::protocol::Role::Server,
            None,
        )
        .await;
        (client, server)
    }

    fn start_test_clob_ingress(socket: ClobSocket) -> ClobIngressTransport {
        let (_shutdown_tx, shutdown) = watch::channel(false);
        start_clob_ingress(socket, StdDuration::from_secs(3_600), shutdown)
    }

    #[tokio::test]
    async fn clob_heartbeat_transmission_does_not_wait_for_frame_processing() {
        let (client, mut server) = clob_socket_pair().await;
        let (_shutdown_tx, shutdown) = watch::channel(false);
        let mut transport = start_clob_ingress(client, StdDuration::from_millis(20), shutdown);
        server
            .send(Message::Text("queued-frame".to_string().into()))
            .await
            .unwrap();
        timeout(StdDuration::from_secs(1), async {
            while transport.events.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let heartbeat = timeout(StdDuration::from_secs(1), server.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(heartbeat.into_text().unwrap(), "PING");

        let mut heartbeat_timing = None;
        while heartbeat_timing.is_none() {
            let event = timeout(StdDuration::from_secs(1), transport.events.recv())
                .await
                .unwrap()
                .unwrap();
            if let ClobIngressEvent::HeartbeatSent {
                scheduled_at,
                sent_at,
                ..
            } = event
            {
                heartbeat_timing = Some((scheduled_at, sent_at));
            }
        }
        let (scheduled_at, sent_at) = heartbeat_timing.unwrap();
        assert!(sent_at >= scheduled_at);
    }

    #[tokio::test]
    async fn clob_ingress_reader_drains_in_order_before_processing() {
        let (client, mut server) = clob_socket_pair().await;
        let mut transport = start_test_clob_ingress(client);
        let sent_before = Utc::now();
        let sent_before_instant = Instant::now();
        for payload in ["first", "second", "third"] {
            server
                .send(Message::Text(payload.to_string().into()))
                .await
                .unwrap();
        }
        tokio::time::sleep(StdDuration::from_millis(20)).await;

        assert_eq!(transport.events.len(), 3);
        let latest_receipt_before_dequeue = transport.latest_receipt_instant().unwrap();
        assert!(latest_receipt_before_dequeue >= sent_before_instant);
        assert!(latest_receipt_before_dequeue <= Instant::now());
        for (expected_sequence, expected_payload) in [(1, "first"), (2, "second"), (3, "third")] {
            let event = timeout(StdDuration::from_secs(1), transport.events.recv())
                .await
                .unwrap()
                .unwrap();
            let ClobIngressEvent::Frame(frame) = event else {
                panic!("expected an ingress frame")
            };
            assert_eq!(frame.sequence, expected_sequence);
            assert_eq!(frame.payload_bytes, expected_payload.len());
            assert_eq!(frame.message.into_text().unwrap(), expected_payload);
            assert!(frame.received_at >= sent_before);
            assert!(frame.received_at <= Utc::now());
        }
    }

    #[tokio::test]
    async fn clob_ingress_overflow_invalidates_instead_of_dropping_and_continuing() {
        let (client, mut server) = clob_socket_pair().await;
        let transport = start_test_clob_ingress(client);
        for _ in 0..=CLOB_INGRESS_FRAME_CAPACITY {
            if server
                .send(Message::Text("x".to_string().into()))
                .await
                .is_err()
            {
                break;
            }
        }

        timeout(StdDuration::from_secs(2), async {
            while !transport.overflowed.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        assert!(transport.overflowed.load(Ordering::Acquire));
        assert_eq!(transport.events.len(), CLOB_INGRESS_FRAME_CAPACITY);
        assert!(transport.reader_task.is_finished());
    }

    #[tokio::test]
    async fn clob_ingress_is_scoped_to_its_connection_epoch() {
        let (first_client, mut first_server) = clob_socket_pair().await;
        let first_transport = start_test_clob_ingress(first_client);
        first_server
            .send(Message::Text("old-epoch".to_string().into()))
            .await
            .unwrap();
        timeout(StdDuration::from_secs(1), async {
            while first_transport.events.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(first_transport);

        let (second_client, mut second_server) = clob_socket_pair().await;
        let mut second_transport = start_test_clob_ingress(second_client);
        second_server
            .send(Message::Text("new-epoch".to_string().into()))
            .await
            .unwrap();
        let event = timeout(StdDuration::from_secs(1), second_transport.events.recv())
            .await
            .unwrap()
            .unwrap();
        let ClobIngressEvent::Frame(frame) = event else {
            panic!("expected a new-epoch ingress frame")
        };
        assert_eq!(frame.sequence, 1);
        assert_eq!(frame.message.into_text().unwrap(), "new-epoch");
    }

    #[tokio::test]
    async fn direct_clob_frame_publishes_only_mutated_canonical_books() {
        let current = market();
        let received_at = current.window_start + Duration::seconds(100);
        let received_instant = Instant::now();
        let (socket, _server) = clob_socket_pair().await;
        let connection_id = Uuid::new_v4();
        let mut registry = BookRegistry::new(connection_id);
        registry.register_market(&current);
        let watchdog = ClobFeedWatchdog::new(
            received_instant,
            &registry,
            std::slice::from_ref(&current),
            received_at,
        );
        let mut epoch = ClobEpoch {
            connection_id,
            connection_epoch: 1,
            transport: start_test_clob_ingress(socket),
            registry: registry.clone(),
            markets: vec![current.clone()],
            connected_at: received_at,
            subscription_stats: ClobSubscriptionStats::default(),
            telemetry: ClobSocketTelemetry::new(std::slice::from_ref(&current)),
            connected_instant: received_instant,
            watchdog,
            healthy_epoch: false,
            books_usable: false,
            pending_frame_metrics: ClobPendingFrameMetrics::default(),
        };
        let shared_books = Arc::new(RwLock::new(registry));
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let payload = serde_json::json!({
            "event_type": "book",
            "market": current.market_id.clone(),
            "asset_id": current.up_token_id.clone(),
            "timestamp": received_at.timestamp_millis().to_string(),
            "bids": [{"price": "0.48", "size": "10"}],
            "asks": [{"price": "0.52", "size": "10"}]
        });

        let message = Message::Text(payload.to_string().into());
        let action = apply_active_clob_frame(
            &mut epoch,
            ClobIngressFrame {
                payload_bytes: message.len(),
                message,
                received_at,
                received_instant,
                sequence: 1,
                _byte_permit: None,
            },
            &shared_books,
            &metrics,
        )
        .await;

        assert_eq!(action.action, ClobFrameAction::Continue);
        assert!(action.readiness_may_have_changed);
        let shared_books_lock_wait = epoch.telemetry.last_shared_books_lock_wait;
        finish_active_clob_frame(&mut epoch, received_instant, shared_books_lock_wait);
        flush_clob_frame_metrics(&mut epoch, &metrics).await;
        let checkpoint = shared_books
            .read()
            .await
            .checkpoint(&current.up_token_id)
            .unwrap();
        assert_eq!(checkpoint.best_bid, Some(dec!(0.48)));
        assert_eq!(checkpoint.best_ask, Some(dec!(0.52)));
        {
            let runtime_metrics = metrics.read().await;
            assert_eq!(runtime_metrics.dropped_messages, 0);
            assert_eq!(runtime_metrics.clob_messages_received, 1);
            assert_eq!(runtime_metrics.feed_events_applied, 1);
            assert_eq!(runtime_metrics.clob_active_ingress_frames, 1);
            assert_eq!(runtime_metrics.clob_active_last_ingress_sequence, 1);
            assert_eq!(runtime_metrics.clob_active_ingress_overflows, 0);
        }

        let unpublished_connection_id = Uuid::new_v4();
        *shared_books.write().await = BookRegistry::new(unpublished_connection_id);
        let payload = serde_json::json!({
            "event_type": "last_trade_price",
            "market": current.market_id,
            "asset_id": current.up_token_id,
            "price": "0.50",
            "size": "1",
            "timestamp": received_at.timestamp_millis().to_string()
        });
        let message = Message::Text(payload.to_string().into());
        let outcome = apply_active_clob_frame(
            &mut epoch,
            ClobIngressFrame {
                payload_bytes: message.len(),
                message,
                received_at,
                received_instant,
                sequence: 2,
                _byte_permit: None,
            },
            &shared_books,
            &metrics,
        )
        .await;

        assert_eq!(outcome.action, ClobFrameAction::Continue);
        assert!(!outcome.readiness_may_have_changed);
        assert_eq!(
            shared_books.read().await.connection_id(),
            unpublished_connection_id
        );
        {
            let runtime_metrics = metrics.read().await;
            assert_eq!(runtime_metrics.clob_messages_received, 1);
            assert_eq!(runtime_metrics.feed_events_applied, 1);
        }
        flush_clob_frame_metrics(&mut epoch, &metrics).await;
        let runtime_metrics = metrics.read().await;
        assert_eq!(runtime_metrics.clob_messages_received, 2);
        assert_eq!(runtime_metrics.feed_events_applied, 2);
    }

    #[tokio::test]
    async fn current_market_rollover_unsubscribes_before_subscribing() {
        let current = market();
        let mut next = current.clone();
        next.market_id = "next-market".to_string();
        next.condition_id = "next-condition".to_string();
        next.up_token_id = "next-up".to_string();
        next.down_token_id = "next-down".to_string();
        next.window_start = current.window_end;
        next.window_end = next.window_start + Duration::minutes(5);
        let (socket, mut server) = clob_socket_pair().await;
        let connection_id = Uuid::new_v4();
        let mut registry = BookRegistry::new(connection_id);
        registry.register_market(&current);
        let connected_instant = Instant::now();
        let watchdog = ClobFeedWatchdog::new(
            connected_instant,
            &registry,
            std::slice::from_ref(&current),
            current.window_start,
        );
        let mut epoch = ClobEpoch {
            connection_id,
            connection_epoch: 1,
            transport: start_test_clob_ingress(socket),
            registry,
            markets: vec![current.clone()],
            connected_at: current.window_start,
            subscription_stats: ClobSubscriptionStats::default(),
            telemetry: ClobSocketTelemetry::new(std::slice::from_ref(&current)),
            connected_instant,
            watchdog,
            healthy_epoch: false,
            books_usable: false,
            pending_frame_metrics: ClobPendingFrameMetrics::default(),
        };
        let (_shutdown_tx, mut shutdown) = watch::channel(false);

        update_clob_epoch_subscriptions(&mut epoch, std::slice::from_ref(&next), &mut shutdown)
            .await
            .unwrap();

        let first = server.next().await.unwrap().unwrap().into_text().unwrap();
        let second = server.next().await.unwrap().unwrap().into_text().unwrap();
        let first: serde_json::Value = serde_json::from_str(&first).unwrap();
        let second: serde_json::Value = serde_json::from_str(&second).unwrap();
        assert_eq!(first["operation"], "unsubscribe");
        assert_eq!(first["assets_ids"], serde_json::json!(["down", "up"]));
        assert_eq!(second["operation"], "subscribe");
        assert_eq!(
            second["assets_ids"],
            serde_json::json!(["next-down", "next-up"])
        );
        assert_eq!(epoch.markets, vec![next]);
        assert_eq!(epoch.registry.len(), 2);
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
    fn remote_clob_close_1013_updates_only_its_telemetry() {
        let observed_at = Utc.timestamp_opt(1_784_736_010, 0).unwrap();
        let frame = CloseFrame {
            code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Again,
            reason: "service overloaded".into(),
        };
        let mut telemetry = ClobSocketTelemetry::new(std::slice::from_ref(&market()));
        let mut metrics = BtcRuntimeMetrics::default();
        telemetry.record_remote_close(Some(&frame));

        record_clob_remote_close_1013(&mut metrics, &telemetry, observed_at);

        assert_eq!(metrics.clob_remote_close_1013_count, 1);
        assert_eq!(metrics.clob_last_remote_close_1013_at, Some(observed_at));
        assert_eq!(
            metrics.clob_last_remote_close_1013_reason.as_deref(),
            Some("service overloaded")
        );

        let non_1013 = CloseFrame {
            code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Away,
            reason: "planned maintenance".into(),
        };
        telemetry.record_remote_close(Some(&non_1013));
        record_clob_remote_close_1013(&mut metrics, &telemetry, observed_at + Duration::seconds(1));

        assert_eq!(metrics.clob_remote_close_1013_count, 1);
        assert_eq!(metrics.clob_last_remote_close_1013_at, Some(observed_at));
        assert_eq!(
            metrics.clob_last_remote_close_1013_reason.as_deref(),
            Some("service overloaded")
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
            ClobDisconnectCause::Shutdown,
        ];

        assert_eq!(clob_failure_close_action(&telemetry), ClobCloseAction::Skip);
        assert_eq!(
            clob_stop_close_action(&telemetry),
            ClobCloseAction::Initiate
        );
        assert_eq!(causes.len(), 5);

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
    fn clob_connection_starts_only_without_an_active_socket() {
        let now = Instant::now();
        assert!(should_connect_clob(false, false, true, now, now));
        assert!(!should_connect_clob(true, false, true, now, now));
        assert!(!should_connect_clob(false, true, true, now, now));
        assert!(!should_connect_clob(false, false, false, now, now));
        assert!(!should_connect_clob(
            false,
            false,
            true,
            now,
            now + StdDuration::from_millis(1),
        ));
    }

    #[test]
    fn unchanged_book_remains_ready_on_structural_connection() {
        let current = market();
        let ready_at = current.window_start + Duration::minutes(1);
        let max_book_age = Duration::seconds(2);
        let registry = ready_book_registry(&current, ready_at - Duration::milliseconds(1));
        let markets = std::slice::from_ref(&current);

        assert!(
            clob_epoch_readiness_diagnostic(&registry, markets, ready_at, max_book_age).is_none()
        );

        let stale_at = ready_at + max_book_age + Duration::milliseconds(1);
        assert!(
            clob_epoch_readiness_diagnostic(&registry, markets, stale_at, max_book_age).is_none()
        );
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

    #[tokio::test]
    async fn delayed_book_pair_stays_unsafe_without_subscription_churn() {
        let mut current = market();
        let checked_at = Utc::now();
        current.window_start = checked_at - Duration::minutes(1);
        current.window_end = checked_at + Duration::minutes(4);
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&current);
        for token_id in [&current.up_token_id, &current.down_token_id] {
            let events = registry.apply(
                ClobMessage::Book {
                    market_id: current.condition_id.clone(),
                    token_id: token_id.clone(),
                    bids: vec![OrderbookLevel {
                        price: dec!(0.48),
                        size: dec!(10),
                    }],
                    asks: vec![OrderbookLevel {
                        price: dec!(0.52),
                        size: dec!(10),
                    }],
                    source_timestamp: checked_at - Duration::seconds(3),
                    source_hash: Some(format!("delayed-{token_id}")),
                },
                checked_at,
            );
            assert!(events.iter().all(|event| event.applied));
        }

        let connection_id = registry.connection_id();
        let (client, mut server) = clob_socket_pair().await;
        let connected_instant = Instant::now();
        let watchdog = ClobFeedWatchdog::new(
            connected_instant,
            &registry,
            std::slice::from_ref(&current),
            checked_at,
        );
        let mut epoch = ClobEpoch {
            connection_id,
            connection_epoch: 1,
            transport: start_test_clob_ingress(client),
            registry,
            markets: vec![current.clone()],
            connected_at: checked_at,
            subscription_stats: ClobSubscriptionStats::default(),
            telemetry: ClobSocketTelemetry::new(std::slice::from_ref(&current)),
            connected_instant,
            watchdog,
            healthy_epoch: false,
            books_usable: false,
            pending_frame_metrics: ClobPendingFrameMetrics::default(),
        };
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let mut failures = 0;
        let mut recovery_window = ClobRecoveryWindow::open(checked_at, connected_instant);

        update_clob_usability(
            &epoch.registry,
            &epoch.markets,
            checked_at,
            Instant::now(),
            Duration::seconds(2),
            &mut epoch.books_usable,
            &mut epoch.healthy_epoch,
            &metrics,
            epoch.connection_id,
            epoch.connection_epoch,
            &mut failures,
            &mut recovery_window,
        )
        .await;
        let delayed = clob_epoch_readiness_diagnostic(
            &epoch.registry,
            &epoch.markets,
            checked_at,
            Duration::seconds(2),
        )
        .expect("delayed books must remain unavailable");
        assert_eq!(delayed.reason, "source_to_receive_lag");
        assert!(!epoch.books_usable);
        assert!(epoch.registry.market_books_bootstrapped(&current));
        assert_eq!(epoch.subscription_stats.updates, 0);
        assert!(timeout(StdDuration::from_millis(25), server.next())
            .await
            .is_err());

        let recovered_at = checked_at + Duration::milliseconds(1);
        for token_id in [&current.up_token_id, &current.down_token_id] {
            let events = epoch.registry.apply(
                ClobMessage::Book {
                    market_id: current.condition_id.clone(),
                    token_id: token_id.clone(),
                    bids: vec![OrderbookLevel {
                        price: dec!(0.48),
                        size: dec!(10),
                    }],
                    asks: vec![OrderbookLevel {
                        price: dec!(0.52),
                        size: dec!(10),
                    }],
                    source_timestamp: recovered_at,
                    source_hash: Some(format!("fresh-{token_id}")),
                },
                recovered_at,
            );
            assert!(events.iter().all(|event| event.applied));
        }
        update_clob_usability(
            &epoch.registry,
            &epoch.markets,
            recovered_at,
            Instant::now(),
            Duration::seconds(2),
            &mut epoch.books_usable,
            &mut epoch.healthy_epoch,
            &metrics,
            epoch.connection_id,
            epoch.connection_epoch,
            &mut failures,
            &mut recovery_window,
        )
        .await;
        assert!(epoch.books_usable);
        assert!(epoch.healthy_epoch);
        assert!(clob_epoch_readiness_diagnostic(
            &epoch.registry,
            &epoch.markets,
            recovered_at,
            Duration::seconds(2),
        )
        .is_none());
        assert_eq!(epoch.subscription_stats.updates, 0);
        assert!(timeout(StdDuration::from_millis(25), server.next())
            .await
            .is_err());
    }
    #[test]
    fn clob_watchdog_uses_transport_receipt_time_for_liveness() {
        let now = Instant::now();
        let checked_at = market().window_start + Duration::minutes(1);
        let registry = BookRegistry::new(Uuid::new_v4());
        let mut watchdog = ClobFeedWatchdog::new(now, &registry, &[], checked_at);
        let initial_read_deadline = now + CLOB_READ_IDLE_TIMEOUT;
        assert_eq!(watchdog.read_idle_deadline, initial_read_deadline);
        assert_eq!(watchdog.pong_deadline, None);

        let pre_ping_frame_at = now + StdDuration::from_millis(500);
        watchdog.observe_transport_receipt(Some(pre_ping_frame_at));
        assert_eq!(
            watchdog.read_idle_deadline,
            pre_ping_frame_at + CLOB_READ_IDLE_TIMEOUT
        );

        let ping_at = now + StdDuration::from_secs(1);
        let pong_timeout = StdDuration::from_secs(25);
        watchdog.record_text_ping(ping_at, pong_timeout);
        let pong_deadline = ping_at + pong_timeout;
        assert_eq!(watchdog.pong_deadline, Some(pong_deadline));

        watchdog.observe_transport_receipt(Some(pre_ping_frame_at));
        assert_eq!(watchdog.pong_deadline, Some(pong_deadline));

        let post_ping_frame_at = now + StdDuration::from_secs(2);
        watchdog.observe_transport_receipt(Some(post_ping_frame_at));
        assert_eq!(
            watchdog.read_idle_deadline,
            post_ping_frame_at + CLOB_READ_IDLE_TIMEOUT
        );
        assert_eq!(watchdog.pong_deadline, None);
        assert_eq!(watchdog.pending_pong_probe_sent_at, Some(ping_at));
        let pong_at = post_ping_frame_at + StdDuration::from_millis(7);
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
            pong_at + CLOB_READ_IDLE_TIMEOUT
        );
    }

    #[test]
    fn clob_pong_acknowledgement_uses_its_transport_receipt_time() {
        let now = Instant::now();
        let checked_at = market().window_start + Duration::minutes(1);
        let registry = BookRegistry::new(Uuid::new_v4());
        let mut watchdog = ClobFeedWatchdog::new(now, &registry, &[], checked_at);
        let ping_at = now + StdDuration::from_secs(1);
        let pong_at = ping_at + StdDuration::from_millis(9);
        watchdog.record_text_ping(ping_at, StdDuration::from_secs(25));
        assert_eq!(
            watchdog.acknowledge_text_pong("PONG", pong_at),
            Some(StdDuration::from_millis(9))
        );
        assert_eq!(watchdog.pong_deadline, None);
        assert_eq!(
            watchdog.read_idle_deadline,
            pong_at + CLOB_READ_IDLE_TIMEOUT
        );
    }

    #[test]
    fn clob_stale_pong_cannot_acknowledge_a_later_probe() {
        let now = Instant::now();
        let checked_at = market().window_start + Duration::minutes(1);
        let registry = BookRegistry::new(Uuid::new_v4());
        let mut watchdog = ClobFeedWatchdog::new(now, &registry, &[], checked_at);
        let ping_at = now + StdDuration::from_secs(2);
        watchdog.record_text_ping(ping_at, StdDuration::from_secs(25));

        assert_eq!(
            watchdog.acknowledge_text_pong("PONG", ping_at - StdDuration::from_millis(1)),
            None
        );
        assert_eq!(watchdog.pending_pong_probe_sent_at, Some(ping_at));
        assert!(watchdog.pong_deadline.is_some());
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
    fn runtime_metrics_snapshot_reports_recent_1013_age() {
        let checked_at = Utc.timestamp_opt(1_784_736_010, 0).unwrap();
        let metrics = BtcRuntimeMetrics {
            clob_last_remote_close_1013_at: Some(checked_at - Duration::milliseconds(1250)),
            ..BtcRuntimeMetrics::default()
        };

        let snapshot = runtime_metrics_snapshot(metrics, checked_at);

        assert_eq!(
            snapshot.clob_last_remote_close_1013_age_milliseconds,
            Some(1250)
        );
    }

    #[test]
    fn runtime_metrics_snapshot_reports_current_clob_unavailable_age() {
        let checked_at = Utc.timestamp_opt(1_784_736_010, 0).unwrap();
        let metrics = BtcRuntimeMetrics {
            clob_recovery_unavailable_since: Some(checked_at - Duration::seconds(95)),
            ..BtcRuntimeMetrics::default()
        };

        let snapshot = runtime_metrics_snapshot(metrics, checked_at);

        assert_eq!(
            snapshot.clob_recovery_unavailable_age_milliseconds,
            Some(95_000)
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

    #[test]
    fn unchanged_book_does_not_rearm_structural_bootstrap() {
        let current = market();
        let ready_at = current.window_start + Duration::minutes(1);
        let ready_instant = Instant::now();
        let max_book_age = Duration::seconds(2);
        let mut registry = ready_book_registry(&current, ready_at - Duration::milliseconds(1));
        let mut watchdog = ClobFeedWatchdog::new(
            ready_instant,
            &registry,
            std::slice::from_ref(&current),
            ready_at,
        );
        assert_eq!(watchdog.bootstrap_deadline, None);

        let stale_at = ready_at + max_book_age + Duration::milliseconds(1);
        let stale_instant = ready_instant + StdDuration::from_millis(2_001);
        watchdog.refresh_bootstrap(
            stale_instant,
            &registry,
            std::slice::from_ref(&current),
            stale_at,
        );
        assert_eq!(watchdog.bootstrap_deadline, None);

        registry.quarantine(FeedIntegrityStatus::Stale);
        watchdog.refresh_bootstrap(
            stale_instant + StdDuration::from_millis(100),
            &registry,
            std::slice::from_ref(&current),
            stale_at + Duration::milliseconds(100),
        );
        assert_eq!(watchdog.bootstrap_deadline, None);
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

        quarantine_clob_books_on_disconnect(&mut registry, &shared_books).await;

        let readiness = realtime_snapshot(&state, &shared_books).await.readiness(
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
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics {
            primary_persistence_queue_overflows: 7,
            ..BtcRuntimeMetrics::default()
        }));
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
        assert_eq!(status.primary_persistence_queue_overflows, 7);
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

        record_primary_persistence_shutdown_abandonment(&metrics, "checkpoint", 3).await;

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

        publish_current_market_subscription_if_changed(&sender, Some(&market));
        assert!(receiver.has_changed().unwrap());
        let published = receiver.borrow_and_update().clone();
        assert_eq!(published.len(), 1);
        assert!(same_market_subscriptions(
            &published,
            std::slice::from_ref(&market)
        ));

        publish_current_market_subscription_if_changed(&sender, Some(&market));
        assert!(!receiver.has_changed().unwrap());

        publish_current_market_subscription_if_changed(&sender, None);
        assert!(receiver.has_changed().unwrap());
        assert!(receiver.borrow_and_update().is_empty());
    }

    #[test]
    fn execution_socket_rejects_more_than_one_market() {
        let current = market();
        let mut historical = current.clone();
        historical.market_id = "historical".to_string();
        historical.condition_id = "historical-condition".to_string();
        historical.up_token_id = "historical-up".to_string();
        historical.down_token_id = "historical-down".to_string();

        assert!(validate_clob_execution_market_set(std::slice::from_ref(&current)).is_ok());
        assert!(validate_clob_execution_market_set(&[current, historical]).is_err());
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
    struct TransientDatabaseFailureStrategyRunner {
        attempts: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl BtcStrategyRunner for TransientDatabaseFailureStrategyRunner {
        async fn on_observation(&self, _observation: StrategyObservation) -> Result<()> {
            if self.attempts.fetch_add(1, Ordering::Relaxed) == 0 {
                return Err(anyhow::Error::new(sqlx::Error::PoolTimedOut)
                    .context("failed to load BTC market fee schedule"));
            }
            Ok(())
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

    #[derive(Debug, Default)]
    struct RecoverableReconciliationStrategyRunner {
        reconciliation_attempts: std::sync::atomic::AtomicUsize,
        callbacks: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl BtcStrategyRunner for RecoverableReconciliationStrategyRunner {
        async fn reconcile_if_due(&self) -> Result<()> {
            self.reconciliation_attempts.fetch_add(1, Ordering::Relaxed);
            bail!("temporary reconciliation failure")
        }

        async fn on_observation(&self, _observation: StrategyObservation) -> Result<()> {
            self.callbacks.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[tokio::test]
    async fn reconciliation_maintenance_failure_never_terminates_strategy_runtime() {
        let config = BtcRuntimeConfig {
            enabled: true,
            strategy_interval: StdDuration::from_millis(1),
            ..BtcRuntimeConfig::default()
        };
        let state = Arc::new(RwLock::new(RealtimeState {
            last_updated_at: Some(Utc::now()),
            ..RealtimeState::default()
        }));
        let strategy = Arc::new(RecoverableReconciliationStrategyRunner::default());
        let books = Arc::new(RwLock::new(BookRegistry::new(Uuid::new_v4())));
        let handle =
            BtcPlaybookRuntimeHandle::start(config, strategy.clone(), state, books).unwrap();

        tokio::time::timeout(StdDuration::from_secs(1), async {
            while strategy.reconciliation_attempts.load(Ordering::Relaxed) < 2
                || strategy.callbacks.load(Ordering::Relaxed) == 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("reconciliation maintenance must retry independently of observations");

        assert!(handle.is_running());
        assert!(handle.metrics.read().await.last_error.is_none());
        handle.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn transient_strategy_database_failure_retries_without_terminating_process() {
        let config = BtcRuntimeConfig {
            enabled: true,
            strategy_interval: StdDuration::from_millis(1),
            ..BtcRuntimeConfig::default()
        };
        let state = Arc::new(RwLock::new(RealtimeState {
            last_updated_at: Some(Utc::now()),
            ..RealtimeState::default()
        }));
        let strategy = Arc::new(TransientDatabaseFailureStrategyRunner::default());
        let books = Arc::new(RwLock::new(BookRegistry::new(Uuid::new_v4())));
        let handle = BtcPlaybookRuntimeHandle::start(config, strategy.clone(), state, books)
            .expect("playbook must start");

        tokio::time::timeout(StdDuration::from_secs(1), async {
            while strategy.attempts.load(Ordering::Relaxed) < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the unchanged observation must retry after database recovery");

        assert!(handle.is_running());
        let status = handle.metrics.read().await;
        assert_eq!(status.strategy_errors, 0);
        assert_eq!(status.strategy_callbacks, 1);
        assert!(status.last_error.is_none());
        drop(status);
        handle.shutdown().await.unwrap();
    }

    #[test]
    fn immutable_order_identity_collision_is_not_a_retryable_database_failure() {
        let error = anyhow::anyhow!(
            "client_order_id collides with immutable order identity, result, or reference execution evidence"
        );
        assert!(!is_retryable_strategy_database_error(&error));
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
        let books = Arc::new(RwLock::new(BookRegistry::new(Uuid::new_v4())));
        let handle =
            BtcPlaybookRuntimeHandle::start(config, strategy.clone(), state.clone(), books)
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
        let books = Arc::new(RwLock::new(BookRegistry::new(Uuid::new_v4())));
        let failing = BtcPlaybookRuntimeHandle::start(
            config.clone(),
            Arc::new(FailingStrategyRunner),
            state.clone(),
            books.clone(),
        )
        .unwrap();
        let healthy = BtcPlaybookRuntimeHandle::start(
            config,
            Arc::new(NoopStrategyRunner),
            state.clone(),
            books,
        )
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
        let (sender, mut receiver) = mpsc::channel(1);
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

        receiver.recv().await.expect("queued item must drain");
        let recovered = enqueue(
            &sender,
            PersistItem::ReferenceTick(tick(Utc::now(), dec!(101))),
            &state,
            &metrics,
        )
        .await;
        assert_eq!(recovered, PersistEnqueueOutcome::Queued);
        assert!(state.read().await.primary_persistence_available());
        assert_eq!(metrics.read().await.primary_persistence_queue_overflows, 1);

        let saturated_again = enqueue(
            &sender,
            PersistItem::ReferenceTick(tick(Utc::now(), dec!(102))),
            &state,
            &metrics,
        )
        .await;
        assert_eq!(saturated_again, PersistEnqueueOutcome::Saturated);
        assert!(!state.read().await.primary_persistence_available());
        assert_eq!(metrics.read().await.primary_persistence_queue_overflows, 2);

        let closed_state = Arc::new(RwLock::new(RealtimeState::default()));
        let closed_metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let (closed_sender, closed_receiver) = mpsc::channel(1);
        drop(closed_receiver);
        let closed = enqueue(
            &closed_sender,
            PersistItem::ReferenceTick(tick(Utc::now(), dec!(103))),
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
            clob_active_ingress_frames: 12,
            clob_active_ingress_bytes: 4_096,
            clob_active_last_ingress_sequence: 12,
            clob_active_ingress_queue_depth: 3,
            clob_active_last_ingress_queue_dwell_milliseconds: 7,
            clob_active_max_ingress_queue_dwell_milliseconds: 11,
            clob_active_ingress_overflows: 1,
            clob_active_last_frame_processing_milliseconds: 5,
            clob_active_max_frame_processing_milliseconds: 9,
            clob_active_last_shared_books_lock_wait_milliseconds: 2,
            clob_active_max_shared_books_lock_wait_milliseconds: 4,
            clob_active_heartbeat_probes: 11,
            clob_active_heartbeat_acknowledgements: 10,
            clob_remote_close_1013_count: 2,
            clob_last_remote_close_1013_at: Some(updated_at),
            clob_last_remote_close_1013_reason: Some("service overloaded".to_string()),
            ..BtcRuntimeMetrics::default()
        }));
        let status = runtime_status_from_inputs(
            Arc::new(RwLock::new(RealtimeState::default())),
            Arc::new(RwLock::new(BookRegistry::new(Uuid::new_v4()))),
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
        assert_eq!(value["metrics"]["clob_active_ingress_frames"], 12);
        assert_eq!(value["metrics"]["clob_active_ingress_bytes"], 4_096);
        assert_eq!(value["metrics"]["clob_active_last_ingress_sequence"], 12);
        assert_eq!(value["metrics"]["clob_active_ingress_queue_depth"], 3);
        assert_eq!(
            value["metrics"]["clob_active_max_ingress_queue_dwell_milliseconds"],
            11
        );
        assert_eq!(value["metrics"]["clob_active_ingress_overflows"], 1);
        assert_eq!(
            value["metrics"]["clob_active_max_frame_processing_milliseconds"],
            9
        );
        assert_eq!(
            value["metrics"]["clob_active_max_shared_books_lock_wait_milliseconds"],
            4
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
        assert_eq!(value["metrics"]["clob_remote_close_1013_count"], 2);
        assert_eq!(
            value["metrics"]["clob_last_remote_close_1013_at"],
            serde_json::json!(updated_at)
        );
        assert_eq!(
            value["metrics"]["clob_last_remote_close_1013_reason"],
            "service overloaded"
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
            Arc::new(RwLock::new(BookRegistry::new(Uuid::new_v4()))),
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
    async fn closed_market_watch_retains_current_while_changes_update_in_place() {
        let (sender, mut receiver) = watch::channel(Vec::<BtcIntervalMarket>::new());
        drop(sender);
        let changed = receiver.changed().await;
        assert_eq!(
            market_watch_disposition(&changed),
            MarketWatchDisposition::RetainCurrent
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
            clob_active_ingress_frames: 12,
            clob_active_ingress_bytes: 4_096,
            clob_active_last_ingress_sequence: 12,
            clob_active_ingress_queue_depth: 3,
            clob_active_last_ingress_queue_dwell_milliseconds: 7,
            clob_active_max_ingress_queue_dwell_milliseconds: 11,
            clob_active_ingress_overflows: 1,
            clob_active_last_frame_processing_milliseconds: 5,
            clob_active_max_frame_processing_milliseconds: 9,
            clob_active_last_shared_books_lock_wait_milliseconds: 2,
            clob_active_max_shared_books_lock_wait_milliseconds: 4,
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
        assert_eq!(metrics.clob_active_ingress_frames, 0);
        assert_eq!(metrics.clob_active_ingress_bytes, 0);
        assert_eq!(metrics.clob_active_last_ingress_sequence, 0);
        assert_eq!(metrics.clob_active_ingress_queue_depth, 0);
        assert_eq!(metrics.clob_active_last_ingress_queue_dwell_milliseconds, 0);
        assert_eq!(metrics.clob_active_max_ingress_queue_dwell_milliseconds, 0);
        assert_eq!(metrics.clob_active_ingress_overflows, 0);
        assert_eq!(metrics.clob_active_last_frame_processing_milliseconds, 0);
        assert_eq!(metrics.clob_active_max_frame_processing_milliseconds, 0);
        assert_eq!(
            metrics.clob_active_last_shared_books_lock_wait_milliseconds,
            0
        );
        assert_eq!(
            metrics.clob_active_max_shared_books_lock_wait_milliseconds,
            0
        );
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
            clob_retry_action(&config, true, &mut failures),
            ClobRetryAction::ImmediateRecovery
        );
        assert_eq!(failures, 4);
        assert_eq!(
            clob_retry_action(&config, false, &mut failures),
            ClobRetryAction::Backoff(StdDuration::from_secs(16))
        );
        assert_eq!(failures, 5);
    }

    #[test]
    fn clob_recovery_diagnostic_ignores_changing_age_samples_for_transitions() {
        let started = Instant::now();
        let since = Utc.timestamp_opt(1_783_902_610, 0).unwrap();
        let mut recovery_window = ClobRecoveryWindow::open(since, started);
        let initial = ClobReadinessDiagnostic {
            reason: "source_to_receive_lag",
            market_id: Some("market-1".to_string()),
            token_id: Some("token-1".to_string()),
            integrity_status: Some(FeedIntegrityStatus::Ok),
            bootstrapped: Some(true),
            has_bid: Some(true),
            has_ask: Some(true),
            source_age_milliseconds: Some(2_001),
            receipt_age_milliseconds: Some(1),
            source_to_receive_lag_milliseconds: Some(2_000),
        };

        assert!(recovery_window.update_diagnostic(initial.clone()));

        let mut later_sample = initial.clone();
        later_sample.source_age_milliseconds = Some(35_000);
        later_sample.receipt_age_milliseconds = Some(0);
        later_sample.source_to_receive_lag_milliseconds = Some(35_000);
        assert!(!recovery_window.update_diagnostic(later_sample.clone()));
        assert_eq!(recovery_window.diagnostic, Some(later_sample.clone()));

        let mut different_token = later_sample;
        different_token.token_id = Some("token-2".to_string());
        assert!(recovery_window.update_diagnostic(different_token));
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
            clob_retry_action(&BtcRuntimeConfig::default(), true, &mut failures,),
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
        assert!(registry.market_books_structurally_ready(&market));
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
        assert_eq!(
            clob,
            serde_json::json!({
                "assets_ids": ["down", "up"],
                "type": "market",
                "initial_dump": true
            })
        );
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
    fn discovery_failure_retains_current_market_but_success_can_clear_it() {
        let current = market();
        let checked_at = current.window_start + Duration::minutes(1);

        assert_eq!(current_market_update(None, checked_at), None);
        assert_eq!(
            current_market_update(Some(std::slice::from_ref(&current)), checked_at),
            Some(CurrentMarketUpdate {
                tradable: Some(current.clone()),
                display: Some(current.clone()),
            })
        );

        let mut closed = current;
        closed.closed = true;
        closed.accepting_orders = false;
        assert_eq!(
            current_market_update(Some(std::slice::from_ref(&closed)), checked_at),
            Some(CurrentMarketUpdate {
                tradable: None,
                display: Some(closed),
            })
        );
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
    fn reference_detail_remains_bounded() {
        let bounded_unicode = bounded_reference_detail("é".repeat(200));
        assert_eq!(bounded_unicode.len(), 256);
        assert_eq!(bounded_unicode.chars().count(), 128);
        let split_boundary = bounded_reference_detail(format!("{}é", "x".repeat(255)));
        assert_eq!(split_boundary.len(), 255);
        assert!(split_boundary.is_char_boundary(split_boundary.len()));
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
