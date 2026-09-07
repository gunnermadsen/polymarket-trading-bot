use std::{
    future::Future,
    panic::AssertUnwindSafe,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex as StdMutex,
    },
    time::Duration as StdDuration,
};

use crate::market_data_stream::{
    legacy_default_sources, MarketDataStreamRuntime, SourceSelector, StreamMetrics,
    PRODUCT_BINANCE_1S, PRODUCT_BINANCE_OPEN_INTEREST, PRODUCT_CHAINLINK,
};
use anyhow::{bail, ensure, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{watch, RwLock},
    task::JoinHandle,
    time::{interval, sleep, MissedTickBehavior},
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use super::{
    directional_external_runtime::BinanceOpenInterestPoint,
    feeds::BookRegistry,
    repository::BtcRepository,
    types::{
        BinanceOneSecondKline, BinanceOneSecondWindow, Readiness, RealtimeState,
        ReferencePriceTick, BINANCE_ONE_SECOND_BOOTSTRAP_CAPACITY,
        BINANCE_PREWINDOW_SUMMARY_CAPACITY,
    },
};

const DIRECTIONAL_CHAINLINK_HYDRATION_RETRY_MAX_DELAY: StdDuration = StdDuration::from_secs(300);
const DIRECTIONAL_OPEN_INTEREST_BOOTSTRAP_POINTS: usize = 13;
const DIRECTIONAL_OPEN_INTEREST_BOOTSTRAP_LOOKBACK_MINUTES: i64 = 70;
const BINANCE_ONE_SECOND_BOOTSTRAP_LOOKBACK_MINUTES: i64 = 66;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BtcRuntimeConfig {
    pub enabled: bool,
    pub strategy_interval: StdDuration,
    pub max_book_age: StdDuration,
    pub max_reference_age: StdDuration,
}

impl Default for BtcRuntimeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy_interval: StdDuration::from_millis(250),
            max_book_age: StdDuration::from_secs(2),
            max_reference_age: StdDuration::from_secs(2),
        }
    }
}

impl BtcRuntimeConfig {
    pub fn validate(&self) -> Result<()> {
        for (name, duration) in [
            ("strategy_interval", self.strategy_interval),
            ("max_book_age", self.max_book_age),
            ("max_reference_age", self.max_reference_age),
        ] {
            if duration.is_zero() {
                bail!("BTC realtime {name} must be positive");
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BtcRuntimeMetrics {
    pub strategy_callbacks: u64,
    pub strategy_errors: u64,
    pub persistence_errors: u64,
    pub dropped_messages: u64,
    #[serde(default)]
    pub rtds_chainlink_candle_window_ready: bool,
    #[serde(default)]
    pub rtds_chainlink_candle_complete_minutes: u64,
    #[serde(default)]
    pub polygon_oracle_ready: bool,
    #[serde(default)]
    pub polygon_oracle_age_seconds: Option<u64>,
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
    state: &RealtimeState,
    checked_at: DateTime<Utc>,
) -> BtcRuntimeMetrics {
    let complete_minutes = state
        .directional_external
        .rtds()
        .complete_minutes(checked_at);
    metrics.rtds_chainlink_candle_complete_minutes =
        u64::try_from(complete_minutes).unwrap_or(u64::MAX);
    metrics.rtds_chainlink_candle_window_ready = complete_minutes == 61;
    metrics.polygon_oracle_ready = state.directional_external.polygon_oracle_ready(checked_at);
    metrics.polygon_oracle_age_seconds = state
        .directional_external
        .polygon_oracle_age_seconds(checked_at)
        .and_then(|age| u64::try_from(age).ok());
    metrics
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
    sources: Vec<SourceSelector>,
}

async fn apply_directional_chainlink_history(
    state: &Arc<RwLock<RealtimeState>>,
    ticks: Vec<ReferencePriceTick>,
) -> usize {
    let mut realtime = state.write().await;
    for tick in ticks {
        if let Err(error) = realtime.directional_external.observe_rtds_chainlink(&tick) {
            tracing::warn!(
                error = %error,
                "directional Chainlink midpoint bootstrap rejected a tick"
            );
        }
    }
    realtime
        .directional_external
        .rtds()
        .complete_minutes(Utc::now())
}

async fn run_directional_chainlink_hydration_recovery(
    repository: BtcRepository,
    state: Arc<RwLock<RealtimeState>>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut retry_delay = StdDuration::from_secs(1);
    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    return;
                }
            }
            _ = sleep(retry_delay) => {}
        }
        let bootstrap_end = Utc::now();
        let bootstrap_start = bootstrap_end - chrono::Duration::minutes(62);
        match repository
            .load_directional_external_chainlink_mid_history(bootstrap_start, bootstrap_end)
            .await
        {
            Ok(ticks) => {
                let tick_count = ticks.len();
                let complete_minutes = apply_directional_chainlink_history(&state, ticks).await;
                if complete_minutes >= 61 {
                    tracing::info!(
                        tick_count,
                        complete_minutes,
                        "directional Chainlink midpoint bootstrap recovered"
                    );
                    let _ = shutdown.changed().await;
                    return;
                }
                tracing::warn!(
                    tick_count,
                    complete_minutes,
                    retry_delay_ms = retry_delay.as_millis(),
                    "directional Chainlink midpoint bootstrap remains incomplete"
                );
            }
            Err(error) => tracing::warn!(
                error = %error,
                retry_delay_ms = retry_delay.as_millis(),
                "directional Chainlink midpoint bootstrap retry failed"
            ),
        }
        retry_delay = (retry_delay * 2).min(DIRECTIONAL_CHAINLINK_HYDRATION_RETRY_MAX_DELAY);
    }
}

async fn apply_directional_open_interest_history(
    state: &Arc<RwLock<RealtimeState>>,
    points: Vec<BinanceOpenInterestPoint>,
    hydrated_at: DateTime<Utc>,
) -> usize {
    let mut realtime = state.write().await;
    realtime
        .directional_external
        .merge_open_interest(points, hydrated_at);
    realtime.directional_external.open_interest.len()
}

async fn run_directional_open_interest_hydration_recovery(
    repository: BtcRepository,
    state: Arc<RwLock<RealtimeState>>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut retry_delay = StdDuration::from_secs(1);
    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    return;
                }
            }
            _ = sleep(retry_delay) => {}
        }
        let bootstrap_end = Utc::now();
        let bootstrap_start = bootstrap_end
            - chrono::Duration::minutes(DIRECTIONAL_OPEN_INTEREST_BOOTSTRAP_LOOKBACK_MINUTES);
        match repository
            .load_directional_external_open_interest_history(bootstrap_start, bootstrap_end)
            .await
        {
            Ok(points) => {
                let point_count =
                    apply_directional_open_interest_history(&state, points, bootstrap_end).await;
                if point_count >= DIRECTIONAL_OPEN_INTEREST_BOOTSTRAP_POINTS {
                    tracing::info!(
                        point_count,
                        "directional Binance open-interest bootstrap recovered"
                    );
                    let _ = shutdown.changed().await;
                    return;
                }
                tracing::warn!(
                    point_count,
                    retry_delay_ms = retry_delay.as_millis(),
                    "directional Binance open-interest bootstrap remains incomplete"
                );
            }
            Err(error) => tracing::warn!(
                error = %error,
                retry_delay_ms = retry_delay.as_millis(),
                "directional Binance open-interest bootstrap retry failed"
            ),
        }
        retry_delay = (retry_delay * 2).min(DIRECTIONAL_CHAINLINK_HYDRATION_RETRY_MAX_DELAY);
    }
}

fn merge_binance_one_second_history(
    window: &mut BinanceOneSecondWindow,
    mut hydrated: Vec<BinanceOneSecondKline>,
) -> Result<()> {
    hydrated.extend(window.completed().iter().cloned());
    hydrated.sort_unstable_by_key(|candle| candle.open_timestamp);
    let mut canonical: Vec<BinanceOneSecondKline> = Vec::with_capacity(hydrated.len());
    for candle in hydrated {
        if let Some(previous) = canonical.last() {
            if previous.open_timestamp == candle.open_timestamp {
                ensure!(
                    previous == &candle,
                    "Binance one-second bootstrap conflicts with live runtime candle"
                );
                continue;
            }
        }
        canonical.push(candle);
    }
    let contiguous_start = canonical
        .windows(2)
        .rposition(|pair| pair[0].close_timestamp != pair[1].open_timestamp)
        .map_or(0, |gap| gap + 1);
    canonical.drain(..contiguous_start);
    if canonical.len() > BINANCE_ONE_SECOND_BOOTSTRAP_CAPACITY {
        canonical.drain(..canonical.len() - BINANCE_ONE_SECOND_BOOTSTRAP_CAPACITY);
    }
    *window = BinanceOneSecondWindow::from_completed(canonical)?;
    Ok(())
}

async fn apply_binance_one_second_history(
    state: &Arc<RwLock<RealtimeState>>,
    candles: Vec<BinanceOneSecondKline>,
) -> Result<(usize, usize)> {
    let mut realtime = state.write().await;
    merge_binance_one_second_history(&mut realtime.binance_one_second_window, candles)?;
    Ok((
        realtime.binance_one_second_window.completed().len(),
        realtime
            .binance_one_second_window
            .completed_five_minute_summaries()
            .len(),
    ))
}

async fn run_binance_one_second_hydration_recovery(
    repository: BtcRepository,
    state: Arc<RwLock<RealtimeState>>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut retry_delay = StdDuration::from_secs(1);
    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    return;
                }
            }
            _ = sleep(retry_delay) => {}
        }
        let bootstrap_end = Utc::now();
        let bootstrap_start = bootstrap_end
            - chrono::Duration::minutes(BINANCE_ONE_SECOND_BOOTSTRAP_LOOKBACK_MINUTES);
        match repository
            .load_binance_one_second_history(bootstrap_start, bootstrap_end)
            .await
        {
            Ok(candles) => match apply_binance_one_second_history(&state, candles).await {
                Ok((candle_count, summary_count))
                    if summary_count >= BINANCE_PREWINDOW_SUMMARY_CAPACITY =>
                {
                    tracing::info!(
                        candle_count,
                        summary_count,
                        "Binance one-second runtime bootstrap recovered"
                    );
                    let _ = shutdown.changed().await;
                    return;
                }
                Ok((candle_count, summary_count)) => tracing::warn!(
                    candle_count,
                    summary_count,
                    retry_delay_ms = retry_delay.as_millis(),
                    "Binance one-second runtime bootstrap remains incomplete"
                ),
                Err(error) => tracing::warn!(
                    error = %error,
                    retry_delay_ms = retry_delay.as_millis(),
                    "Binance one-second runtime bootstrap merge failed"
                ),
            },
            Err(error) => tracing::warn!(
                error = %error,
                retry_delay_ms = retry_delay.as_millis(),
                "Binance one-second runtime bootstrap retry failed"
            ),
        }
        retry_delay = (retry_delay * 2).min(DIRECTIONAL_CHAINLINK_HYDRATION_RETRY_MAX_DELAY);
    }
}

impl BtcRuntime {
    pub fn new(config: BtcRuntimeConfig, repository: BtcRepository) -> Self {
        Self {
            config,
            repository,
            books: None,
            state: None,
            sources: legacy_default_sources(),
        }
    }

    pub fn with_shared_state(mut self, state: Arc<RwLock<RealtimeState>>) -> Self {
        self.state = Some(state);
        self
    }

    pub fn with_shared_book_registry(mut self, books: Arc<RwLock<BookRegistry>>) -> Self {
        self.books = Some(books);
        self
    }

    pub fn with_sources(mut self, sources: Vec<SourceSelector>) -> Self {
        self.sources = sources;
        self
    }

    pub async fn start(mut self) -> Result<BtcRuntimeHandle> {
        include_shared_rtds(&mut self.sources);
        self.config.validate()?;
        self.repository.healthcheck().await?;
        let state = self
            .state
            .unwrap_or_else(|| Arc::new(RwLock::new(RealtimeState::default())));
        let mut binance_one_second_hydration_failed = false;
        if self
            .sources
            .iter()
            .any(|source| source.key == PRODUCT_BINANCE_1S)
        {
            let bootstrap_end = Utc::now();
            let bootstrap_start = bootstrap_end
                - chrono::Duration::minutes(BINANCE_ONE_SECOND_BOOTSTRAP_LOOKBACK_MINUTES);
            match self
                .repository
                .load_binance_one_second_history(bootstrap_start, bootstrap_end)
                .await
            {
                Ok(candles) => match apply_binance_one_second_history(&state, candles).await {
                    Ok((_, summary_count))
                        if summary_count >= BINANCE_PREWINDOW_SUMMARY_CAPACITY => {}
                    Ok((candle_count, summary_count)) => {
                        binance_one_second_hydration_failed = true;
                        tracing::warn!(
                            candle_count,
                            summary_count,
                            "Binance one-second runtime bootstrap is incomplete"
                        );
                    }
                    Err(error) => {
                        binance_one_second_hydration_failed = true;
                        tracing::warn!(
                            error = %error,
                            "Binance one-second runtime bootstrap merge failed"
                        );
                    }
                },
                Err(error) => {
                    binance_one_second_hydration_failed = true;
                    tracing::warn!(
                        error = %error,
                        "Binance one-second runtime bootstrap failed"
                    );
                }
            }
        }
        let mut directional_chainlink_hydration_failed = false;
        {
            let bootstrap_end = Utc::now();
            let bootstrap_start = bootstrap_end - chrono::Duration::minutes(62);
            match self
                .repository
                .load_directional_external_chainlink_mid_history(bootstrap_start, bootstrap_end)
                .await
            {
                Ok(ticks) => {
                    let complete_minutes = apply_directional_chainlink_history(&state, ticks).await;
                    if complete_minutes < 61 {
                        directional_chainlink_hydration_failed = true;
                        tracing::warn!(
                            complete_minutes,
                            "directional Chainlink midpoint bootstrap is incomplete"
                        );
                    }
                }
                Err(error) => {
                    directional_chainlink_hydration_failed = true;
                    tracing::warn!(
                        error = %error,
                        "directional Chainlink midpoint bootstrap failed"
                    );
                }
            }
        }
        let mut directional_open_interest_hydration_failed = false;
        if self
            .sources
            .iter()
            .any(|source| source.key == PRODUCT_BINANCE_OPEN_INTEREST)
        {
            let bootstrap_end = Utc::now();
            let bootstrap_start = bootstrap_end
                - chrono::Duration::minutes(DIRECTIONAL_OPEN_INTEREST_BOOTSTRAP_LOOKBACK_MINUTES);
            match self
                .repository
                .load_directional_external_open_interest_history(bootstrap_start, bootstrap_end)
                .await
            {
                Ok(points) => {
                    let point_count =
                        apply_directional_open_interest_history(&state, points, bootstrap_end)
                            .await;
                    if point_count < DIRECTIONAL_OPEN_INTEREST_BOOTSTRAP_POINTS {
                        directional_open_interest_hydration_failed = true;
                        tracing::warn!(
                            point_count,
                            "directional Binance open-interest bootstrap is incomplete"
                        );
                    }
                }
                Err(error) => {
                    directional_open_interest_hydration_failed = true;
                    tracing::warn!(
                        error = %error,
                        "directional Binance open-interest bootstrap failed"
                    );
                }
            }
        }

        let books = self
            .books
            .unwrap_or_else(|| Arc::new(RwLock::new(BookRegistry::new(Uuid::new_v4()))));
        let metrics = Arc::new(RwLock::new(BtcRuntimeMetrics::default()));
        let running = Arc::new(AtomicBool::new(true));
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let open_interest_hydration_started =
            Arc::new(AtomicBool::new(includes_open_interest(&self.sources)));
        let binance_one_second_hydration_started =
            Arc::new(AtomicBool::new(includes_binance_one_second(&self.sources)));
        let (sources_tx, sources_rx) = watch::channel(self.sources);
        let stream_shutdown = CancellationToken::new();
        let stream_shutdown_task = stream_shutdown.clone();
        let stream_metrics = Arc::new(StreamMetrics::default());
        crate::market_data_stream::install_metrics(stream_metrics.clone());
        let master_url = std::env::var("INGESTER_MASTER_URL")
            .context("INGESTER_MASTER_URL is required for BTC market data")?;
        let stream_token = std::env::var("INGESTER_ADMIN_TOKEN")
            .or_else(|_| std::env::var("MARKET_DATA_INGESTER_ADMIN_TOKEN"))
            .context("INGESTER_ADMIN_TOKEN is required for BTC market data")?;
        let stream_runtime = MarketDataStreamRuntime::new(
            master_url,
            stream_token,
            "polymarket-bot".to_owned(),
            self.repository.clone(),
            state.clone(),
            books.clone(),
            stream_metrics,
        )?;
        let watch_shutdown = stream_shutdown.clone();
        let mut stream_shutdown_rx = shutdown_rx.clone();
        tokio::spawn(async move {
            let _ = stream_shutdown_rx.changed().await;
            watch_shutdown.cancel();
        });
        let mut tasks = vec![spawn_runtime_task(
            "market_data_grpc",
            stream_runtime.run(sources_rx, stream_shutdown_task),
            running.clone(),
            metrics.clone(),
        )];
        if binance_one_second_hydration_failed {
            tasks.push(spawn_runtime_task(
                "binance_one_second_hydration",
                run_binance_one_second_hydration_recovery(
                    self.repository.clone(),
                    state.clone(),
                    shutdown_rx.clone(),
                ),
                running.clone(),
                metrics.clone(),
            ));
        }
        if directional_chainlink_hydration_failed {
            tasks.push(spawn_runtime_task(
                "directional_chainlink_hydration",
                run_directional_chainlink_hydration_recovery(
                    self.repository.clone(),
                    state.clone(),
                    shutdown_rx.clone(),
                ),
                running.clone(),
                metrics.clone(),
            ));
        }
        if directional_open_interest_hydration_failed {
            tasks.push(spawn_runtime_task(
                "directional_open_interest_hydration",
                run_directional_open_interest_hydration_recovery(
                    self.repository.clone(),
                    state.clone(),
                    shutdown_rx.clone(),
                ),
                running.clone(),
                metrics.clone(),
            ));
        }
        drop(shutdown_rx);

        Ok(BtcRuntimeHandle {
            shutdown: shutdown_tx,
            tasks,
            state,
            books,
            metrics,
            config: self.config,
            running,
            sources: sources_tx,
            repository: self.repository,
            open_interest_hydration_started,
            dynamic_open_interest_hydration_task: StdMutex::new(None),
            binance_one_second_hydration_started,
            dynamic_binance_one_second_hydration_task: StdMutex::new(None),
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

fn includes_open_interest(sources: &[SourceSelector]) -> bool {
    sources
        .iter()
        .any(|source| source.key == PRODUCT_BINANCE_OPEN_INTEREST)
}

// The shared repository stays current even when no active model requires RTDS.
// Preserve any explicit consumer selector; the default must not gate other models.
fn include_shared_rtds(sources: &mut Vec<SourceSelector>) {
    if !sources.iter().any(|source| source.key == PRODUCT_CHAINLINK) {
        sources.push(SourceSelector {
            key: PRODUCT_CHAINLINK.to_owned(),
            contract_version: 1,
            required: false,
            maximum_age_ms: None,
            require_sequence_integrity: true,
        });
    }
}

fn includes_binance_one_second(sources: &[SourceSelector]) -> bool {
    sources
        .iter()
        .any(|source| source.key == PRODUCT_BINANCE_1S)
}

fn claim_dynamic_open_interest_hydration(
    previous: &[SourceSelector],
    next: &[SourceSelector],
    hydration_started: &AtomicBool,
) -> bool {
    !includes_open_interest(previous)
        && includes_open_interest(next)
        && hydration_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
}

fn claim_dynamic_binance_one_second_hydration(
    previous: &[SourceSelector],
    next: &[SourceSelector],
    hydration_started: &AtomicBool,
) -> bool {
    !includes_binance_one_second(previous)
        && includes_binance_one_second(next)
        && hydration_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
}

pub struct BtcRuntimeHandle {
    shutdown: watch::Sender<bool>,
    tasks: Vec<JoinHandle<()>>,
    state: Arc<RwLock<RealtimeState>>,
    books: Arc<RwLock<BookRegistry>>,
    metrics: Arc<RwLock<BtcRuntimeMetrics>>,
    config: BtcRuntimeConfig,
    running: Arc<AtomicBool>,
    sources: watch::Sender<Vec<SourceSelector>>,
    repository: BtcRepository,
    open_interest_hydration_started: Arc<AtomicBool>,
    dynamic_open_interest_hydration_task: StdMutex<Option<JoinHandle<()>>>,
    binance_one_second_hydration_started: Arc<AtomicBool>,
    dynamic_binance_one_second_hydration_task: StdMutex<Option<JoinHandle<()>>>,
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

    pub fn update_sources(&self, mut sources: Vec<SourceSelector>) {
        include_shared_rtds(&mut sources);
        let hydrate_open_interest = {
            let previous = self.sources.borrow();
            claim_dynamic_open_interest_hydration(
                previous.as_slice(),
                &sources,
                &self.open_interest_hydration_started,
            )
        };
        let hydrate_binance_one_second = {
            let previous = self.sources.borrow();
            claim_dynamic_binance_one_second_hydration(
                previous.as_slice(),
                &sources,
                &self.binance_one_second_hydration_started,
            )
        };
        if self.sources.borrow().as_slice() != sources.as_slice() {
            self.sources.send_replace(sources);
        }
        if hydrate_open_interest {
            let task = spawn_runtime_task(
                "directional_open_interest_hydration",
                run_directional_open_interest_hydration_recovery(
                    self.repository.clone(),
                    self.state.clone(),
                    self.shutdown.subscribe(),
                ),
                self.running.clone(),
                self.metrics.clone(),
            );
            let previous = self
                .dynamic_open_interest_hydration_task
                .lock()
                .expect("dynamic open-interest hydration task lock")
                .replace(task);
            debug_assert!(previous.is_none());
        }
        if hydrate_binance_one_second {
            let task = spawn_runtime_task(
                "binance_one_second_hydration",
                run_binance_one_second_hydration_recovery(
                    self.repository.clone(),
                    self.state.clone(),
                    self.shutdown.subscribe(),
                ),
                self.running.clone(),
                self.metrics.clone(),
            );
            let previous = self
                .dynamic_binance_one_second_hydration_task
                .lock()
                .expect("dynamic Binance one-second hydration task lock")
                .replace(task);
            debug_assert!(previous.is_none());
        }
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
        let mut join_failures = Vec::new();
        for task in &mut self.tasks {
            if let Err(error) = task.await {
                join_failures.push(error.to_string());
            }
        }
        self.tasks.clear();
        let dynamic_hydration = self
            .dynamic_open_interest_hydration_task
            .lock()
            .expect("dynamic open-interest hydration task lock")
            .take();
        if let Some(task) = dynamic_hydration {
            if let Err(error) = task.await {
                join_failures.push(error.to_string());
            }
        }
        let dynamic_binance_hydration = self
            .dynamic_binance_one_second_hydration_task
            .lock()
            .expect("dynamic Binance one-second hydration task lock")
            .take();
        if let Some(task) = dynamic_binance_hydration {
            if let Err(error) = task.await {
                join_failures.push(error.to_string());
            }
        }
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
        if let Some(task) = self
            .dynamic_open_interest_hydration_task
            .lock()
            .expect("dynamic open-interest hydration task lock")
            .as_ref()
        {
            task.abort();
        }
        if let Some(task) = self
            .dynamic_binance_one_second_hydration_task
            .lock()
            .expect("dynamic Binance one-second hydration task lock")
            .as_ref()
        {
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
        metrics: runtime_metrics_snapshot(metrics.read().await.clone(), &state, checked_at),
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
                    tracing::warn!(error = %error, "BTC reconciliation maintenance failed; runtime remains active");
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
                    Utc::now(), chrono_duration(config.max_book_age),
                    chrono_duration(config.max_reference_age),
                );
                if let Err(error) = strategy.on_observation(StrategyObservation { state: snapshot, readiness }).await {
                    if is_retryable_strategy_database_error(&error) {
                        tracing::warn!(
                            error = %format!("{error:#}"),
                            "BTC strategy database dependency unavailable; skipping this evaluation and preserving the trading process"
                        );
                        last_observation = None;
                        continue;
                    }
                    let error_chain = format!("{error:#}");
                    tracing::error!(error = %error_chain, "BTC strategy callback failed; terminating the trading process");
                    let mut runtime_metrics = metrics.write().await;
                    runtime_metrics.strategy_errors = runtime_metrics.strategy_errors.saturating_add(1);
                    runtime_metrics.last_error = Some(error_chain);
                    return;
                }
                let mut runtime_metrics = metrics.write().await;
                runtime_metrics.strategy_callbacks = runtime_metrics.strategy_callbacks.saturating_add(1);
            }
        }
    }
}

fn retryable_postgres_sqlstate(code: &str) -> bool {
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
                .is_some_and(retryable_postgres_sqlstate)
                || message == "query_wait_timeout"
                || message.starts_with("query_wait_timeout:")
                || message == "sorry, too many clients already"
        }
        sqlx::Error::Io(_) | sqlx::Error::Tls(_) | sqlx::Error::PoolTimedOut => true,
        _ => false,
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
    use super::*;
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    #[test]
    fn shared_rtds_subscription_is_unique_and_preserves_explicit_consumers() {
        let mut sources = Vec::new();
        include_shared_rtds(&mut sources);
        include_shared_rtds(&mut sources);
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].key, PRODUCT_CHAINLINK);
        assert!(!sources[0].required);
        let mut explicit = legacy_default_sources();
        let before = explicit.clone();
        include_shared_rtds(&mut explicit);
        assert_eq!(explicit, before);
    }

    #[tokio::test]
    async fn shared_rtds_seed_restores_consumer_reads_and_existing_readiness_metrics() {
        let at = DateTime::from_timestamp(Utc::now().timestamp().div_euclid(60) * 60, 0).unwrap();
        let state = Arc::new(RwLock::new(RealtimeState::default()));
        let ticks = (1..=62)
            .map(|minute| {
                let timestamp = at - chrono::Duration::minutes(minute);
                ReferencePriceTick {
                    tick_id: Uuid::new_v4(),
                    dedup_key: minute.to_string(),
                    source: super::super::types::ReferencePriceSource::RtdsChainlink,
                    symbol: "BTCUSD".into(),
                    price: dec!(100),
                    source_timestamp: timestamp,
                    envelope_timestamp: Some(timestamp),
                    received_at: timestamp,
                    connection_id: Uuid::nil(),
                    ingest_sequence: minute as u64,
                    source_event_id: None,
                    raw_payload: serde_json::Value::Null,
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            apply_directional_chainlink_history(&state, ticks.clone()).await,
            61
        );
        assert_eq!(apply_directional_chainlink_history(&state, ticks).await, 61);
        let first_consumer = state.read().await.clone();
        let second_consumer = state.read().await.clone();
        assert!(Arc::ptr_eq(
            &first_consumer.directional_external.rtds,
            &second_consumer.directional_external.rtds
        ));
        assert_eq!(
            first_consumer
                .directional_external
                .rtds()
                .closed_candles(at)
                .unwrap()
                .len(),
            61
        );
        let metrics = runtime_metrics_snapshot(BtcRuntimeMetrics::default(), &second_consumer, at);
        assert_eq!(metrics.rtds_chainlink_candle_complete_minutes, 61);
        assert!(metrics.rtds_chainlink_candle_window_ready);
        assert_eq!(
            first_consumer
                .directional_external
                .rtds()
                .points_as_of(at)
                .count(),
            62
        );
    }

    fn one_second_candle(open_timestamp: DateTime<Utc>) -> BinanceOneSecondKline {
        BinanceOneSecondKline {
            open_timestamp,
            close_timestamp: open_timestamp + chrono::Duration::seconds(1),
            open_price: dec!(100),
            high_price: dec!(101),
            low_price: dec!(99),
            close_price: dec!(100),
            base_volume: dec!(1),
            quote_volume: dec!(100),
            trade_count: 1,
            taker_buy_base_volume: dec!(0.5),
            taker_buy_quote_volume: dec!(50),
            first_aggregate_trade_id: 0,
            last_aggregate_trade_id: 0,
            first_source_timestamp: open_timestamp,
            last_source_timestamp: open_timestamp + chrono::Duration::milliseconds(999),
            max_received_at: open_timestamp + chrono::Duration::milliseconds(1200),
            source_complete: true,
            synthetic: false,
        }
    }

    #[test]
    fn runtime_config_rejects_zero_durations() {
        let mut config = BtcRuntimeConfig::default();
        config.strategy_interval = StdDuration::ZERO;
        assert!(config.validate().is_err());
    }

    #[test]
    fn primary_runtime_failure_is_scoped_to_active_runtime_failures() {
        assert!(primary_runtime_failure(&BtcRuntimeMetrics::default()).is_none());
        assert!(primary_runtime_failure(&BtcRuntimeMetrics {
            strategy_errors: 1,
            ..BtcRuntimeMetrics::default()
        })
        .is_some());
    }

    #[tokio::test]
    async fn open_interest_hydration_and_live_updates_share_one_contiguous_state() {
        let start = DateTime::from_timestamp(1_788_436_800, 0).unwrap();
        let point = |index: i64| BinanceOpenInterestPoint {
            source_timestamp: start + chrono::Duration::minutes(index * 5),
            available_at: start
                + chrono::Duration::minutes(index * 5)
                + chrono::Duration::seconds(1),
            sum_open_interest: dec!(100) + Decimal::from(index),
            sum_open_interest_value: dec!(1000) + Decimal::from(index),
        };
        let state = Arc::new(RwLock::new(RealtimeState::default()));
        let hydrated = (0..13).map(point).collect::<Vec<_>>();

        assert_eq!(
            apply_directional_open_interest_history(
                &state,
                hydrated,
                start + chrono::Duration::minutes(61),
            )
            .await,
            13
        );
        state
            .write()
            .await
            .directional_external
            .merge_open_interest(vec![point(13)], start + chrono::Duration::minutes(66));
        let state = state.read().await;
        assert_eq!(state.directional_external.open_interest.len(), 14);
        assert_eq!(
            state
                .directional_external
                .open_interest
                .back()
                .unwrap()
                .source_timestamp,
            start + chrono::Duration::minutes(65)
        );
    }

    #[test]
    fn seven_to_eight_source_union_claims_open_interest_hydration_exactly_once() {
        let eight_sources = legacy_default_sources();
        let seven_sources = eight_sources
            .iter()
            .filter(|source| source.key != PRODUCT_BINANCE_OPEN_INTEREST)
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(seven_sources.len(), 7);
        assert_eq!(eight_sources.len(), 8);
        let hydration_started = AtomicBool::new(false);

        assert!(claim_dynamic_open_interest_hydration(
            &seven_sources,
            &eight_sources,
            &hydration_started,
        ));
        assert!(!claim_dynamic_open_interest_hydration(
            &seven_sources,
            &eight_sources,
            &hydration_started,
        ));
        assert!(!claim_dynamic_open_interest_hydration(
            &eight_sources,
            &seven_sources,
            &hydration_started,
        ));
        assert!(!claim_dynamic_open_interest_hydration(
            &seven_sources,
            &eight_sources,
            &hydration_started,
        ));
    }

    #[test]
    fn binance_one_second_hydration_preserves_prewindow_and_accepts_live_continuation() {
        let start = DateTime::from_timestamp(1_788_436_800, 0).unwrap();
        let hydrated = (0..BINANCE_ONE_SECOND_BOOTSTRAP_CAPACITY)
            .map(|index| one_second_candle(start + chrono::Duration::seconds(index as i64)))
            .collect::<Vec<_>>();
        let mut window = BinanceOneSecondWindow::default();

        merge_binance_one_second_history(&mut window, hydrated).unwrap();
        assert_eq!(window.completed().len(), 305);
        assert_eq!(
            window.completed_five_minute_summaries().len(),
            BINANCE_PREWINDOW_SUMMARY_CAPACITY
        );
        let next_open = window.completed().back().unwrap().close_timestamp;
        window
            .observe_completed(one_second_candle(next_open))
            .unwrap();
        assert_eq!(window.completed().back().unwrap().open_timestamp, next_open);
        assert_eq!(
            window.completed_five_minute_summaries().len(),
            BINANCE_PREWINDOW_SUMMARY_CAPACITY
        );
        assert!(window
            .observe_completed(one_second_candle(next_open + chrono::Duration::seconds(2)))
            .is_err());
    }

    #[test]
    fn binance_one_second_hydration_discards_history_before_latest_gap() {
        let start = DateTime::from_timestamp(1_788_436_800, 0).unwrap();
        let mut window = BinanceOneSecondWindow::default();
        merge_binance_one_second_history(
            &mut window,
            vec![
                one_second_candle(start),
                one_second_candle(start + chrono::Duration::seconds(1)),
                one_second_candle(start + chrono::Duration::seconds(3)),
                one_second_candle(start + chrono::Duration::seconds(4)),
            ],
        )
        .unwrap();

        assert_eq!(window.completed().len(), 2);
        assert_eq!(
            window.completed().front().unwrap().open_timestamp,
            start + chrono::Duration::seconds(3)
        );
    }

    #[test]
    fn dynamic_binance_one_second_selector_claims_hydration_exactly_once() {
        let eight_sources = legacy_default_sources();
        let seven_sources = eight_sources
            .iter()
            .filter(|source| source.key != PRODUCT_BINANCE_1S)
            .cloned()
            .collect::<Vec<_>>();
        let hydration_started = AtomicBool::new(false);

        assert!(claim_dynamic_binance_one_second_hydration(
            &seven_sources,
            &eight_sources,
            &hydration_started,
        ));
        assert!(!claim_dynamic_binance_one_second_hydration(
            &seven_sources,
            &eight_sources,
            &hydration_started,
        ));
    }
}
