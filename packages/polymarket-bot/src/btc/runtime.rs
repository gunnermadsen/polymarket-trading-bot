use std::{
    collections::{HashMap, HashSet},
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
    sync::{mpsc, watch, RwLock},
    task::JoinHandle,
    time::{interval, sleep, Instant, MissedTickBehavior},
};
use tokio_tungstenite::{connect_async, tungstenite::Message};
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
        BtcIntervalMarket, BtcOutcome, FeedIntegrityStatus, MarketFeedEventType, Readiness,
        RealtimeState, ReferencePriceSource, ReferencePriceTick,
    },
};

const BOUNDARY_LABEL_VERSION: &str = "chainlink_first_tick_at_or_after_boundary_v1";
const CRITICAL_WRITE_ATTEMPTS: usize = 3;
const CRITICAL_WRITE_INITIAL_BACKOFF: StdDuration = StdDuration::from_millis(25);

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
    let mut reconnect_ordinal = 0i32;
    let mut consecutive_failures = 0u32;
    let mut recovery_window = ClobRecoveryWindow::open(Utc::now(), Instant::now());
    metrics.write().await.clob_recovery_unavailable_since = recovery_window.since;
    loop {
        if *shutdown.borrow() {
            break;
        }
        if markets.borrow().is_empty() {
            tokio::select! {
                _ = shutdown.changed() => break,
                changed = markets.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    continue;
                }
            }
        }
        reconnect_ordinal = reconnect_ordinal.saturating_add(1);
        let connection_id = Uuid::new_v4();
        let started_at = Utc::now();
        let attempt_started_at = Instant::now();
        let mut session = new_session(
            connection_id,
            "polymarket_clob_market",
            &config.clob_ws_url,
            reconnect_ordinal,
            started_at,
        );
        let outcome = connect_async(&config.clob_ws_url).await;
        let (mut socket, _) = match outcome {
            Ok(value) => value,
            Err(error) => {
                let disconnected_at = Utc::now();
                let reason = error.to_string();
                consecutive_failures = consecutive_failures.saturating_add(1);
                let delay = reconnect_backoff(&config, consecutive_failures);
                session.disconnected_at = Some(disconnected_at);
                session.disconnect_reason = Some(reason.clone());
                session.metadata = clob_session_metadata(
                    false,
                    ClobDisconnectCause::ConnectFailure,
                    ClobRetryAction::Backoff(delay),
                    consecutive_failures,
                    &ClobSubscriptionStats::default(),
                );
                if !start_feed_session_or_fail(&repository, &session, &metrics).await
                    || !finish_feed_session_or_fail(&repository, &session, &metrics).await
                {
                    return;
                }
                {
                    let delay_milliseconds = duration_milliseconds(delay);
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.clob_connection_failures =
                        runtime_metrics.clob_connection_failures.saturating_add(1);
                    runtime_metrics.clob_backoff_scheduled_milliseconds = runtime_metrics
                        .clob_backoff_scheduled_milliseconds
                        .saturating_add(delay_milliseconds);
                    clear_clob_connection_metrics(
                        &mut runtime_metrics,
                        disconnected_at,
                        &reason,
                        recovery_window.since,
                        consecutive_failures,
                    );
                }
                tracing::warn!(
                    feed = "polymarket_clob_market",
                    %connection_id,
                    connection_epoch = reconnect_ordinal,
                    connect_latency_ms = duration_milliseconds(attempt_started_at.elapsed()),
                    failure_streak = consecutive_failures,
                    retry_delay_ms = duration_milliseconds(delay),
                    reason = %reason,
                    "CLOB websocket connection attempt failed"
                );
                record_error(&metrics, error.into()).await;
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
            runtime_metrics.clob_connections_established = runtime_metrics
                .clob_connections_established
                .saturating_add(1);
            runtime_metrics.clob_connected_connection_epoch = Some(reconnect_ordinal);
            runtime_metrics.clob_connected_connection_id = Some(connection_id);
            runtime_metrics.clob_last_connected_at = Some(connected_at);
            runtime_metrics.clob_consecutive_failures = consecutive_failures;
        }
        tracing::info!(
            feed = "polymarket_clob_market",
            %connection_id,
            connection_epoch = reconnect_ordinal,
            failure_streak = consecutive_failures,
            connect_latency_ms = duration_milliseconds(connected_instant.duration_since(attempt_started_at)),
            "CLOB websocket connected"
        );
        let mut terminate_supervisor = false;
        let mut fatal_persistence_error = None;
        let mut subscription_failed = false;
        let mut healthy_epoch = false;
        let mut books_usable = false;
        // The durable watch ledger, not the three-window Gamma cache, owns this subscription set.
        // That keeps delayed outcomes subscribed across multiple rollovers and process restarts.
        let mut active_markets = markets.borrow().iter().cloned().collect::<Vec<_>>();
        let mut subscription_stats = ClobSubscriptionStats::default();
        let mut registry = BookRegistry::new(connection_id);
        let registration_succeeded = match register_clob_markets(&mut registry, &active_markets) {
            Ok(()) => true,
            Err(error) => {
                session.disconnect_reason = Some(format!("invalid_subscription_identity:{error}"));
                subscription_failed = true;
                record_error(&metrics, error).await;
                false
            }
        };
        if registration_succeeded {
            *shared_books.write().await = registry.clone();
            {
                let mut shared = state.write().await;
                shared.update_books(&registry);
                shared.last_updated_at = Some(Utc::now());
            }
        }
        if !registration_succeeded {
            // The connection is closed below. No subscription is sent for an invalid identity set.
        } else if let Err(error) = socket
            .send(Message::Text(clob_subscription(&active_markets).into()))
            .await
        {
            subscription_failed = true;
            session.disconnect_reason = Some(error.to_string());
        } else if let Err(error) =
            acknowledge_clob_subscriptions(&repository, &active_markets, connection_id, Utc::now())
                .await
        {
            session.disconnect_reason = Some(format!("critical_subscription_ack:{error}"));
            fatal_persistence_error = Some(error);
        } else {
            subscription_stats.active_assets = registry.len();
            metrics.write().await.clob_active_subscribed_assets =
                u64::try_from(subscription_stats.active_assets).unwrap_or(u64::MAX);
            let mut heartbeat = interval(config.clob_heartbeat_interval);
            let mut checkpoints = interval(config.checkpoint_interval);
            heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
            checkpoints.set_missed_tick_behavior(MissedTickBehavior::Delay);
            'connection: loop {
                tokio::select! {
                    _ = shutdown.changed() => {
                        session.disconnect_reason = Some("shutdown".to_string());
                        break;
                    }
                    changed = markets.changed() => {
                        match market_watch_disposition(&changed) {
                            MarketWatchDisposition::UpdateSubscriptions => {
                                let desired_markets = markets.borrow().clone();
                                if let Err(error) = registry.validate_market_set(&desired_markets) {
                                    subscription_failed = true;
                                    session.disconnect_reason = Some(format!(
                                        "invalid_subscription_transition:{error}"
                                    ));
                                    record_error(&metrics, error).await;
                                    break 'connection;
                                }
                                let delta = clob_subscription_delta(
                                    &active_markets,
                                    &desired_markets,
                                );
                                if let Err(error) = register_clob_markets(
                                    &mut registry,
                                    &delta.added_markets,
                                ) {
                                    subscription_failed = true;
                                    session.disconnect_reason = Some(format!(
                                        "subscription_registration_failed:{error}"
                                    ));
                                    record_error(&metrics, error).await;
                                    break 'connection;
                                }
                                if !delta.added_assets.is_empty() {
                                    if let Err(error) = socket
                                        .send(Message::Text(
                                            clob_subscription_operation(
                                                &delta.added_assets,
                                                ClobSubscriptionOperation::Subscribe,
                                            )
                                            .into(),
                                        ))
                                        .await
                                    {
                                        subscription_failed = true;
                                        session.disconnect_reason = Some(format!(
                                            "dynamic_subscribe_failed:{error}"
                                        ));
                                        break 'connection;
                                    }
                                }
                                if !delta.added_markets.is_empty() {
                                    if let Err(error) = acknowledge_clob_subscriptions(
                                        &repository,
                                        &delta.added_markets,
                                        connection_id,
                                        Utc::now(),
                                    )
                                    .await
                                    {
                                        session.disconnect_reason = Some(format!(
                                            "critical_subscription_ack:{error}"
                                        ));
                                        fatal_persistence_error = Some(error);
                                        break 'connection;
                                    }
                                }
                                if !delta.removed_assets.is_empty() {
                                    if let Err(error) = socket
                                        .send(Message::Text(
                                            clob_subscription_operation(
                                                &delta.removed_assets,
                                                ClobSubscriptionOperation::Unsubscribe,
                                            )
                                            .into(),
                                        ))
                                        .await
                                    {
                                        subscription_failed = true;
                                        session.disconnect_reason = Some(format!(
                                            "dynamic_unsubscribe_failed:{error}"
                                        ));
                                        break 'connection;
                                    }
                                }
                                if let Err(error) = registry.retain_markets(&desired_markets) {
                                    subscription_failed = true;
                                    session.disconnect_reason = Some(format!(
                                        "subscription_retention_failed:{error}"
                                    ));
                                    record_error(&metrics, error).await;
                                    break 'connection;
                                }

                                active_markets = desired_markets;
                                let updated_at = Utc::now();
                                let update_instant = Instant::now();
                                if !delta.is_empty() {
                                    subscription_stats.updates =
                                        subscription_stats.updates.saturating_add(1);
                                    subscription_stats.active_assets = registry.len();
                                    subscription_stats.last_updated_at = Some(updated_at);
                                }
                                *shared_books.write().await = registry.clone();
                                {
                                    let mut shared = state.write().await;
                                    shared.update_books(&registry);
                                    shared.last_updated_at = Some(updated_at);
                                }
                                if !delta.is_empty() {
                                    {
                                        let mut runtime_metrics = metrics.write().await;
                                        runtime_metrics.clob_subscription_updates = runtime_metrics
                                            .clob_subscription_updates
                                            .saturating_add(1);
                                        runtime_metrics.clob_active_subscribed_assets = u64::try_from(
                                            subscription_stats.active_assets,
                                        )
                                        .unwrap_or(u64::MAX);
                                        runtime_metrics.clob_last_subscription_update_at =
                                            Some(updated_at);
                                    }
                                    tracing::info!(
                                        feed = "polymarket_clob_market",
                                        %connection_id,
                                        connection_epoch = reconnect_ordinal,
                                        added_assets = delta.added_assets.len(),
                                        removed_assets = delta.removed_assets.len(),
                                        active_assets = subscription_stats.active_assets,
                                        "CLOB websocket subscriptions updated in place"
                                    );
                                }

                                update_clob_usability(
                                    &registry,
                                    &active_markets,
                                    updated_at,
                                    update_instant,
                                    chrono_duration(config.max_book_age),
                                    &mut books_usable,
                                    &mut healthy_epoch,
                                    &metrics,
                                    connection_id,
                                    reconnect_ordinal,
                                    &mut consecutive_failures,
                                    &mut recovery_window,
                                )
                                .await;
                            }
                            MarketWatchDisposition::Terminate => {
                                terminate_supervisor = true;
                                session.disconnect_reason = Some("market_watch_closed".to_string());
                                break 'connection;
                            }
                        }
                    }
                    _ = heartbeat.tick() => {
                        if let Err(error) = socket.send(Message::Text("PING".into())).await {
                            session.disconnect_reason = Some(error.to_string());
                            break;
                        }
                    }
                    _ = checkpoints.tick() => {
                        let checked_at = Utc::now();
                        update_clob_usability(
                            &registry,
                            &active_markets,
                            checked_at,
                            Instant::now(),
                            chrono_duration(config.max_book_age),
                            &mut books_usable,
                            &mut healthy_epoch,
                            &metrics,
                            connection_id,
                            reconnect_ordinal,
                            &mut consecutive_failures,
                            &mut recovery_window,
                        )
                        .await;
                        let checkpoint_markets = active_markets.clone();
                        for market in &checkpoint_markets {
                            if !market.is_trade_window(checked_at) {
                                continue;
                            }
                            for token_id in [&market.up_token_id, &market.down_token_id] {
                                if let Some(checkpoint) = registry.checkpoint(token_id) {
                                    if enqueue(&writer, PersistItem::Checkpoint(checkpoint), &metrics).await {
                                        let mut runtime_metrics = metrics.write().await;
                                        runtime_metrics.checkpoints_queued =
                                            runtime_metrics.checkpoints_queued.saturating_add(1);
                                    } else {
                                        session.dropped_messages =
                                            session.dropped_messages.saturating_add(1);
                                        session.disconnect_reason = Some(
                                            "critical_checkpoint_queue_closed".to_string(),
                                        );
                                        fatal_persistence_error = Some(anyhow::anyhow!(
                                            "CLOB checkpoint persistence queue closed"
                                        ));
                                        break 'connection;
                                    }
                                }
                            }
                        }
                    }
                    message = socket.next() => {
                        let Some(message) = message else {
                            session.disconnect_reason = Some("websocket_eof".to_string());
                            break;
                        };
                        match message {
                            Ok(Message::Text(text)) => {
                                if text.trim().eq_ignore_ascii_case("PONG") || text.trim().is_empty() {
                                    continue;
                                }
                                let received_at = Utc::now();
                                session.messages_received = session.messages_received.saturating_add(1);
                                {
                                    let mut runtime_metrics = metrics.write().await;
                                    runtime_metrics.clob_messages_received =
                                        runtime_metrics.clob_messages_received.saturating_add(1);
                                }
                                match serde_json::from_str::<serde_json::Value>(&text)
                                    .context("failed to decode CLOB websocket JSON")
                                    .and_then(|value| parse_clob_messages(&value))
                                {
                                    Ok(messages) => {
                                        for message in messages {
                                            let resolution = match persist_official_resolution(
                                                &repository,
                                                &message,
                                                received_at,
                                                &metrics,
                                            )
                                            .await
                                            {
                                                Ok(resolution) => resolution,
                                                Err(error) => {
                                                    session.disconnect_reason = Some(format!(
                                                        "critical_official_resolution:{error}"
                                                    ));
                                                    fatal_persistence_error = Some(error);
                                                    break 'connection;
                                                }
                                            };
                                            let events = registry.apply(message, received_at);
                                            *shared_books.write().await = registry.clone();
                                            {
                                                let mut shared = state.write().await;
                                                shared.update_books(&registry);
                                                shared.last_updated_at = Some(received_at);
                                                if let Some(resolution) = resolution {
                                                    shared.apply_market_resolution(
                                                        &resolution.market_id,
                                                        &resolution.winning_token_id,
                                                    );
                                                }
                                            }
                                            for event in events {
                                                if event.applied {
                                                    let mut runtime_metrics = metrics.write().await;
                                                    runtime_metrics.feed_events_applied =
                                                        runtime_metrics.feed_events_applied.saturating_add(1);
                                                } else {
                                                    session.integrity_gaps =
                                                        session.integrity_gaps.saturating_add(1);
                                                    let mut runtime_metrics = metrics.write().await;
                                                    runtime_metrics.integrity_gaps =
                                                        runtime_metrics.integrity_gaps.saturating_add(1);
                                                }
                                                if !should_persist_feed_event(&event) {
                                                    continue;
                                                }
                                                if enqueue(&writer, PersistItem::FeedEvent(event), &metrics).await {
                                                    session.messages_persisted =
                                                        session.messages_persisted.saturating_add(1);
                                                } else {
                                                    session.dropped_messages =
                                                        session.dropped_messages.saturating_add(1);
                                                    session.disconnect_reason = Some(
                                                        "critical_feed_event_queue_closed".to_string(),
                                                    );
                                                    fatal_persistence_error = Some(anyhow::anyhow!(
                                                        "CLOB feed event persistence queue closed"
                                                    ));
                                                    break 'connection;
                                                }
                                            }
                                        }
                                    }
                                    Err(error) => {
                                        registry.quarantine(FeedIntegrityStatus::DecodeError);
                                        *shared_books.write().await = registry.clone();
                                        {
                                            let mut shared = state.write().await;
                                            shared.update_books(&registry);
                                            shared.last_updated_at = Some(received_at);
                                        }
                                        session.decode_errors = session.decode_errors.saturating_add(1);
                                        {
                                            let mut runtime_metrics = metrics.write().await;
                                            runtime_metrics.decode_errors =
                                                runtime_metrics.decode_errors.saturating_add(1);
                                        }
                                        record_error(&metrics, error).await;
                                    }
                                }
                                update_clob_usability(
                                    &registry,
                                    &active_markets,
                                    Utc::now(),
                                    Instant::now(),
                                    chrono_duration(config.max_book_age),
                                    &mut books_usable,
                                    &mut healthy_epoch,
                                    &metrics,
                                    connection_id,
                                    reconnect_ordinal,
                                    &mut consecutive_failures,
                                    &mut recovery_window,
                                )
                                .await;
                            }
                            Ok(Message::Binary(bytes)) => {
                                let received_at = Utc::now();
                                session.messages_received = session.messages_received.saturating_add(1);
                                match serde_json::from_slice::<serde_json::Value>(&bytes)
                                    .context("failed to decode binary CLOB websocket JSON")
                                    .and_then(|value| parse_clob_messages(&value))
                                {
                                    Ok(messages) => {
                                        for message in messages {
                                            let resolution = match persist_official_resolution(
                                                &repository,
                                                &message,
                                                received_at,
                                                &metrics,
                                            )
                                            .await
                                            {
                                                Ok(resolution) => resolution,
                                                Err(error) => {
                                                    session.disconnect_reason = Some(format!(
                                                        "critical_official_resolution:{error}"
                                                    ));
                                                    fatal_persistence_error = Some(error);
                                                    break 'connection;
                                                }
                                            };
                                            for event in registry.apply(message, received_at) {
                                                if should_persist_feed_event(&event) {
                                                    if !enqueue(
                                                        &writer,
                                                        PersistItem::FeedEvent(event),
                                                        &metrics,
                                                    )
                                                    .await
                                                    {
                                                        session.dropped_messages = session
                                                            .dropped_messages
                                                            .saturating_add(1);
                                                        session.disconnect_reason = Some(
                                                            "critical_feed_event_queue_closed"
                                                                .to_string(),
                                                        );
                                                        fatal_persistence_error = Some(
                                                            anyhow::anyhow!(
                                                                "CLOB feed event persistence queue closed"
                                                            ),
                                                        );
                                                        break 'connection;
                                                    }
                                                }
                                            }
                                            if let Some(resolution) = resolution {
                                                state.write().await.apply_market_resolution(
                                                    &resolution.market_id,
                                                    &resolution.winning_token_id,
                                                );
                                            }
                                        }
                                        *shared_books.write().await = registry.clone();
                                        let mut shared = state.write().await;
                                        shared.update_books(&registry);
                                        shared.last_updated_at = Some(received_at);
                                    }
                                    Err(error) => {
                                        registry.quarantine(FeedIntegrityStatus::DecodeError);
                                        *shared_books.write().await = registry.clone();
                                        {
                                            let mut shared = state.write().await;
                                            shared.update_books(&registry);
                                            shared.last_updated_at = Some(received_at);
                                        }
                                        session.decode_errors = session.decode_errors.saturating_add(1);
                                        {
                                            let mut runtime_metrics = metrics.write().await;
                                            runtime_metrics.decode_errors =
                                                runtime_metrics.decode_errors.saturating_add(1);
                                        }
                                        record_error(&metrics, error).await;
                                    }
                                }
                                update_clob_usability(
                                    &registry,
                                    &active_markets,
                                    Utc::now(),
                                    Instant::now(),
                                    chrono_duration(config.max_book_age),
                                    &mut books_usable,
                                    &mut healthy_epoch,
                                    &metrics,
                                    connection_id,
                                    reconnect_ordinal,
                                    &mut consecutive_failures,
                                    &mut recovery_window,
                                )
                                .await;
                            }
                            Ok(Message::Close(frame)) => {
                                session.disconnect_reason = Some(format!("remote_close:{frame:?}"));
                                break;
                            }
                            Ok(_) => {}
                            Err(error) => {
                                session.disconnect_reason = Some(error.to_string());
                                break;
                            }
                        }
                    }
                }
            }
        }
        let disconnected_at = Utc::now();
        let disconnected_instant = Instant::now();
        session.disconnected_at = Some(disconnected_at);
        recovery_window.open_if_closed(disconnected_at, disconnected_instant);
        let reason = session
            .disconnect_reason
            .clone()
            .unwrap_or_else(|| "unknown_disconnect".to_string());
        let disconnect_cause = clob_disconnect_cause(
            healthy_epoch,
            subscription_failed,
            terminate_supervisor,
            *shutdown.borrow() || reason == "shutdown",
            fatal_persistence_error.is_some(),
        );
        let retry_action = clob_retry_action(
            &config,
            healthy_epoch,
            matches!(
                disconnect_cause,
                ClobDisconnectCause::Shutdown
                    | ClobDisconnectCause::MarketWatchClosed
                    | ClobDisconnectCause::CriticalPersistence
            ),
            &mut consecutive_failures,
        );
        let immediate_recovery = matches!(retry_action, ClobRetryAction::ImmediateRecovery);
        let retry_delay = match retry_action {
            ClobRetryAction::Backoff(delay) => Some(delay),
            ClobRetryAction::Stop | ClobRetryAction::ImmediateRecovery => None,
        };
        session.metadata = clob_session_metadata(
            healthy_epoch,
            disconnect_cause,
            retry_action,
            consecutive_failures,
            &subscription_stats,
        );
        quarantine_clob_books_on_disconnect(&mut registry, &state, &shared_books, disconnected_at)
            .await;
        {
            let mut runtime_metrics = metrics.write().await;
            clear_clob_connection_metrics(
                &mut runtime_metrics,
                disconnected_at,
                &reason,
                recovery_window.since,
                consecutive_failures,
            );
            if retry_action != ClobRetryAction::Stop {
                runtime_metrics.reconnects = runtime_metrics.reconnects.saturating_add(1);
            }
            match disconnect_cause {
                ClobDisconnectCause::TransportFailure => {
                    runtime_metrics.clob_transport_disconnects =
                        runtime_metrics.clob_transport_disconnects.saturating_add(1);
                }
                ClobDisconnectCause::SubscriptionFailure => {
                    runtime_metrics.clob_subscription_failures =
                        runtime_metrics.clob_subscription_failures.saturating_add(1);
                }
                ClobDisconnectCause::BootstrapFailure => {
                    runtime_metrics.clob_bootstrap_failures =
                        runtime_metrics.clob_bootstrap_failures.saturating_add(1);
                }
                ClobDisconnectCause::ConnectFailure
                | ClobDisconnectCause::Shutdown
                | ClobDisconnectCause::MarketWatchClosed
                | ClobDisconnectCause::CriticalPersistence => {}
            }
            if immediate_recovery {
                runtime_metrics.clob_immediate_recoveries_scheduled = runtime_metrics
                    .clob_immediate_recoveries_scheduled
                    .saturating_add(1);
            }
            if let Some(delay) = retry_delay {
                runtime_metrics.clob_backoff_scheduled_milliseconds = runtime_metrics
                    .clob_backoff_scheduled_milliseconds
                    .saturating_add(duration_milliseconds(delay));
            }
        }
        log_clob_disconnect(
            connection_id,
            reconnect_ordinal,
            disconnected_instant.duration_since(connected_instant),
            healthy_epoch,
            disconnect_cause,
            retry_action,
            consecutive_failures,
            &reason,
        );
        if !finish_feed_session_or_fail(&repository, &session, &metrics).await {
            return;
        }
        if let Some(error) = fatal_persistence_error {
            record_critical_persistence_error(&metrics, error).await;
            return;
        }
        match retry_action {
            ClobRetryAction::Stop => return,
            ClobRetryAction::ImmediateRecovery => continue,
            ClobRetryAction::Backoff(delay) => {
                if !wait_reconnect_backoff(delay, &mut shutdown).await {
                    break;
                }
            }
        }
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
    {
        let mut shared = state.write().await;
        shared.update_books(registry);
        shared.last_updated_at = Some(disconnected_at);
    }
    *shared_books.write().await = registry.clone();
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
    let mut reconnect_ordinal = 0i32;
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
        let (mut socket, _) = match connect_async(&config.rtds_ws_url).await {
            Ok(value) => value,
            Err(error) => {
                session.disconnected_at = Some(Utc::now());
                session.disconnect_reason = Some(error.to_string());
                if !start_feed_session_or_fail(&repository, &session, &metrics).await
                    || !finish_feed_session_or_fail(&repository, &session, &metrics).await
                {
                    return;
                }
                record_error(&metrics, error.into()).await;
                if !reconnect_delay(&config, reconnect_ordinal, &mut shutdown).await {
                    break;
                }
                continue;
            }
        };
        session.connected_at = Some(Utc::now());
        if !start_feed_session_or_fail(&repository, &session, &metrics).await {
            return;
        }
        let mut fatal_persistence_error = None;
        if let Err(error) = socket.send(Message::Text(rtds_subscription().into())).await {
            session.disconnect_reason = Some(error.to_string());
        } else {
            let mut heartbeat = interval(config.rtds_heartbeat_interval);
            heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
            'connection: loop {
                tokio::select! {
                    _ = shutdown.changed() => {
                        session.disconnect_reason = Some("shutdown".to_string());
                        break;
                    }
                    _ = heartbeat.tick() => {
                        if let Err(error) = socket.send(Message::Text("PING".into())).await {
                            session.disconnect_reason = Some(error.to_string());
                            break;
                        }
                    }
                    message = socket.next() => {
                        let Some(message) = message else {
                            session.disconnect_reason = Some("websocket_eof".to_string());
                            break;
                        };
                        match message {
                            Ok(Message::Text(text))
                                if !text.trim().eq_ignore_ascii_case("PONG")
                                    && !text.trim().is_empty() =>
                            {
                                let received_at = Utc::now();
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
                                        state.write().await.update_reference_price(tick.clone());
                                        {
                                            let mut runtime_metrics = metrics.write().await;
                                            runtime_metrics.reference_ticks_received =
                                                runtime_metrics.reference_ticks_received.saturating_add(1);
                                        }
                                        if tick.source == ReferencePriceSource::RtdsChainlink {
                                            let max_delay =
                                                chrono_duration(config.boundary_tick_max_delay);
                                            let observed = boundaries
                                                .write()
                                                .await
                                                .observe_chainlink(&tick, max_delay);
                                            if let Err(error) = observed {
                                                session.disconnect_reason =
                                                    Some(format!("critical_boundary_integrity:{error}"));
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
                                                session.disconnect_reason =
                                                    Some(format!("critical_boundary_persistence:{error}"));
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
                                            return;
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
                            Ok(Message::Close(frame)) => {
                                session.disconnect_reason = Some(format!("remote_close:{frame:?}"));
                                break;
                            }
                            Ok(_) => {}
                            Err(error) => {
                                session.disconnect_reason = Some(error.to_string());
                                break;
                            }
                        }
                    }
                }
            }
        }
        session.disconnected_at = Some(Utc::now());
        if !finish_feed_session_or_fail(&repository, &session, &metrics).await {
            return;
        }
        if let Some(error) = fatal_persistence_error {
            record_critical_persistence_error(&metrics, error).await;
            return;
        }
        {
            let mut runtime_metrics = metrics.write().await;
            runtime_metrics.reconnects = runtime_metrics.reconnects.saturating_add(1);
        }
        if !reconnect_delay(&config, reconnect_ordinal, &mut shutdown).await {
            break;
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
    let mut reconnect_ordinal = 0i32;
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
        let (mut socket, _) = match connect_async(&config.binance_ws_url).await {
            Ok(value) => value,
            Err(error) => {
                session.disconnected_at = Some(Utc::now());
                session.disconnect_reason = Some(error.to_string());
                if !start_feed_session_or_fail(&repository, &session, &metrics).await
                    || !finish_feed_session_or_fail(&repository, &session, &metrics).await
                {
                    return;
                }
                record_error(&metrics, error.into()).await;
                if !reconnect_delay(&config, reconnect_ordinal, &mut shutdown).await {
                    break;
                }
                continue;
            }
        };
        session.connected_at = Some(Utc::now());
        if !start_feed_session_or_fail(&repository, &session, &metrics).await {
            return;
        }
        let mut heartbeat = interval(config.binance_heartbeat_interval);
        heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    session.disconnect_reason = Some("shutdown".to_string());
                    break;
                }
                _ = heartbeat.tick() => {
                    if let Err(error) = socket.send(Message::Ping(Vec::new().into())).await {
                        session.disconnect_reason = Some(error.to_string());
                        break;
                    }
                }
                message = socket.next() => {
                    let Some(message) = message else {
                        session.disconnect_reason = Some("websocket_eof".to_string());
                        break;
                    };
                    match message {
                        Ok(Message::Text(text)) => {
                            let received_at = Utc::now();
                            sequence = sequence.saturating_add(1);
                            session.messages_received = session.messages_received.saturating_add(1);
                            let parsed = serde_json::from_str::<serde_json::Value>(&text)
                                .context("failed to decode Binance aggregate trade JSON")
                                .and_then(|value| parse_binance_agg_trade(
                                    &value, connection_id, sequence, received_at
                                ));
                            match parsed {
                                Ok(tick) => {
                                    state.write().await.update_reference_price(tick.clone());
                                    {
                                        let mut runtime_metrics = metrics.write().await;
                                        runtime_metrics.reference_ticks_received =
                                            runtime_metrics.reference_ticks_received.saturating_add(1);
                                    }
                                    if enqueue(&writer, PersistItem::ReferenceTick(tick), &metrics).await {
                                        session.messages_persisted =
                                            session.messages_persisted.saturating_add(1);
                                    } else {
                                        session.dropped_messages =
                                            session.dropped_messages.saturating_add(1);
                                        return;
                                    }
                                }
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
                        Ok(Message::Close(frame)) => {
                            session.disconnect_reason = Some(format!("remote_close:{frame:?}"));
                            break;
                        }
                        Ok(_) => {}
                        Err(error) => {
                            session.disconnect_reason = Some(error.to_string());
                            break;
                        }
                    }
                }
            }
        }
        session.disconnected_at = Some(Utc::now());
        if !finish_feed_session_or_fail(&repository, &session, &metrics).await {
            return;
        }
        {
            let mut runtime_metrics = metrics.write().await;
            runtime_metrics.reconnects = runtime_metrics.reconnects.saturating_add(1);
        }
        if !reconnect_delay(&config, reconnect_ordinal, &mut shutdown).await {
            break;
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

async fn reconnect_delay(
    config: &BtcRuntimeConfig,
    reconnect_ordinal: i32,
    shutdown: &mut watch::Receiver<bool>,
) -> bool {
    let failures = u32::try_from(reconnect_ordinal).unwrap_or(u32::MAX);
    wait_reconnect_backoff(reconnect_backoff(config, failures), shutdown).await
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
