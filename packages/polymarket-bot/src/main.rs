use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use polymarket_bot::{
    backfill::{
        enqueue_and_spawn_with_venue, fetch_closed_positions_for_wallet, fetch_whale_trade_page,
        run_copy_trade_signal_engine, BackfillMode, CopyTradeRunConfig, WhaleBackfillRequest,
    },
    clob::ClobClient,
    config::{AppConfig, ExecutionMode},
    copytrade::CopyTradeConfig,
    data_api::DataApiClient,
    events::ServiceEvent,
    execution::{live::LiveVenue, sim::SimVenue, ExecutionVenue, LiveVenueStatus},
    gamma::GammaClient,
    http as control_http,
    http::{
        BackfillJobResponse, BackfillJobsResponse, CancelBackfillJobResponse, ControlApi,
        HttpError, MetricsResponse,
    },
    risk::{RiskLimits, RiskState},
    scanner::{scan_markets_for_signal1, ScannerConfig, ScannerCycleReport},
    store::Store,
    trade_pnl::{mark_trade_pnl_now_with_config, refresh_trade_pnl_with_config, TradePnlConfig},
    wallets::score_closed_position_performance,
};
use tokio::time::MissedTickBehavior;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;

const SCAN_CYCLE_TIMEOUT: Duration = Duration::from_secs(8);
const WHALE_POLL_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Default, Clone)]
struct RuntimeMetrics {
    scans: u64,
    markets_seen: u64,
    markets_persisted: u64,
    tokens_persisted: u64,
    books_persisted: u64,
    signals_inserted: u64,
    positive_signals: u64,
    orders_inserted: u64,
    fills_inserted: u64,
    positions_upserted: u64,
    whale_live_poll_cycles: u64,
    whale_live_trades_seen: u64,
    copy_trade_signals: u64,
    copy_trade_orders: u64,
    copy_trade_fills: u64,
    whale_live_poll_errors: u64,
    scan_errors: u64,
    started_at: chrono::DateTime<Utc>,
}

impl RuntimeMetrics {
    fn new() -> Self {
        Self {
            started_at: Utc::now(),
            ..Self::default()
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "scans": self.scans,
            "markets_seen": self.markets_seen,
            "markets_persisted": self.markets_persisted,
            "tokens_persisted": self.tokens_persisted,
            "books_persisted": self.books_persisted,
            "signals_inserted": self.signals_inserted,
            "positive_signals": self.positive_signals,
            "orders_inserted": self.orders_inserted,
            "fills_inserted": self.fills_inserted,
            "positions_upserted": self.positions_upserted,
            "whale_live_poll_cycles": self.whale_live_poll_cycles,
            "whale_live_trades_seen": self.whale_live_trades_seen,
            "copy_trade_signals": self.copy_trade_signals,
            "copy_trade_orders": self.copy_trade_orders,
            "copy_trade_fills": self.copy_trade_fills,
            "whale_live_poll_errors": self.whale_live_poll_errors,
            "scan_errors": self.scan_errors,
            "uptime_secs": (Utc::now() - self.started_at).num_seconds()
        })
    }
}

struct RuntimeControl {
    store: Store,
    data_api: DataApiClient,
    venue: Arc<dyn ExecutionVenue>,
    metrics: Arc<Mutex<RuntimeMetrics>>,
    default_lookback_days: u32,
    default_min_trade_usd: rust_decimal::Decimal,
    default_copy_trade_config: CopyTradeConfig,
    default_copy_execute_enabled: bool,
    default_max_pages: usize,
    trade_pnl_config: TradePnlConfig,
}

#[derive(Debug, Default)]
struct ScanRunReport {
    markets_seen: usize,
    markets_persisted: usize,
    persist_errors: usize,
    scanner: ScannerCycleReport,
}

#[async_trait]
impl ControlApi for RuntimeControl {
    async fn metrics(&self) -> Result<MetricsResponse, HttpError> {
        let counters = self
            .metrics
            .lock()
            .map_err(|_| HttpError::internal("metrics lock poisoned"))?
            .to_json();
        Ok(MetricsResponse {
            service: "polymarket-bot".to_string(),
            captured_at: Utc::now(),
            counters,
            gauges: serde_json::json!({}),
        })
    }

    async fn start_whales_backfill(
        &self,
        request: control_http::BackfillWhalesRequest,
    ) -> Result<BackfillJobResponse, HttpError> {
        let backfill_request = WhaleBackfillRequest {
            lookback_days: request
                .lookback_days
                .unwrap_or(self.default_lookback_days as i32)
                .max(0) as u32,
            min_trade_usd: request.min_trade_usd.unwrap_or(self.default_min_trade_usd),
            market_ids: request.market_ids,
            event_ids: Vec::new(),
            wallets: request.wallet_addresses,
            mode: request.mode.unwrap_or(BackfillMode::Full),
            dry_run: request.dry_run,
            limit: request.page_limit.unwrap_or(1000),
            max_pages: request.max_pages.unwrap_or(self.default_max_pages),
            copy_min_wallet_score: self.default_copy_trade_config.min_wallet_score,
            copy_size_fraction: self.default_copy_trade_config.copy_size_fraction,
            copy_max_size_usd: self.default_copy_trade_config.max_copy_size_usd,
            copy_trade_config: Some(self.default_copy_trade_config.clone()),
            execute_signals: false,
        };
        let job_id = enqueue_and_spawn_with_venue(
            self.store.clone(),
            self.data_api.clone(),
            Some(self.venue.clone()),
            backfill_request,
        )
        .await
        .map_err(|error| HttpError::internal(error.to_string()))?;
        let job = self
            .store
            .get_backfill_job(job_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(BackfillJobResponse { job })
    }

    async fn start_copy_trade_backtest(
        &self,
        request: control_http::CopyTradeBacktestRequest,
    ) -> Result<BackfillJobResponse, HttpError> {
        let mut copy_trade_config = self.default_copy_trade_config.clone();
        if let Some(min_wallet_score) = request.min_wallet_score {
            copy_trade_config.min_wallet_score = min_wallet_score;
        }
        if let Some(copy_size_fraction) = request.copy_size_fraction {
            copy_trade_config.copy_size_fraction = copy_size_fraction;
        }
        if let Some(max_copy_size_usd) = request.max_copy_size_usd {
            copy_trade_config.max_copy_size_usd = max_copy_size_usd;
        }
        let backfill_request = WhaleBackfillRequest {
            lookback_days: request
                .lookback_days
                .unwrap_or(self.default_lookback_days as i32)
                .max(0) as u32,
            min_trade_usd: request.min_trade_usd.unwrap_or(self.default_min_trade_usd),
            market_ids: Vec::new(),
            event_ids: Vec::new(),
            wallets: Vec::new(),
            mode: BackfillMode::CopyTradeBacktest,
            dry_run: request.dry_run,
            limit: request.page_limit.unwrap_or(1000),
            max_pages: request.max_pages.unwrap_or(self.default_max_pages),
            copy_min_wallet_score: copy_trade_config.min_wallet_score,
            copy_size_fraction: copy_trade_config.copy_size_fraction,
            copy_max_size_usd: copy_trade_config.max_copy_size_usd,
            copy_trade_config: Some(copy_trade_config),
            execute_signals: request.execute_signals && self.default_copy_execute_enabled,
        };
        let job_id = enqueue_and_spawn_with_venue(
            self.store.clone(),
            self.data_api.clone(),
            Some(self.venue.clone()),
            backfill_request,
        )
        .await
        .map_err(|error| HttpError::internal(error.to_string()))?;
        let job = self
            .store
            .get_backfill_job(job_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(BackfillJobResponse { job })
    }

    async fn start_copy_trade_calibration(
        &self,
        request: control_http::CopyTradeCalibrationRequest,
    ) -> Result<BackfillJobResponse, HttpError> {
        let backfill_request = WhaleBackfillRequest {
            lookback_days: request
                .lookback_days
                .unwrap_or(self.default_lookback_days as i32)
                .max(0) as u32,
            min_trade_usd: request.min_trade_usd.unwrap_or(self.default_min_trade_usd),
            market_ids: Vec::new(),
            event_ids: Vec::new(),
            wallets: Vec::new(),
            mode: BackfillMode::CopyTradeBacktest,
            dry_run: true,
            limit: request.page_limit.unwrap_or(1000),
            max_pages: request.max_pages.unwrap_or(self.default_max_pages),
            copy_min_wallet_score: self.default_copy_trade_config.min_wallet_score,
            copy_size_fraction: self.default_copy_trade_config.copy_size_fraction,
            copy_max_size_usd: self.default_copy_trade_config.max_copy_size_usd,
            copy_trade_config: Some(self.default_copy_trade_config.clone()),
            execute_signals: false,
        };
        let job_id = enqueue_and_spawn_with_venue(
            self.store.clone(),
            self.data_api.clone(),
            Some(self.venue.clone()),
            backfill_request,
        )
        .await
        .map_err(|error| HttpError::internal(error.to_string()))?;
        let job = self
            .store
            .get_backfill_job(job_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(BackfillJobResponse { job })
    }

    async fn list_backfill_jobs(&self) -> Result<BackfillJobsResponse, HttpError> {
        let jobs = self
            .store
            .list_backfill_jobs(50)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(BackfillJobsResponse { jobs })
    }

    async fn get_backfill_job(&self, job_id: uuid::Uuid) -> Result<BackfillJobResponse, HttpError> {
        let job = self
            .store
            .get_backfill_job(job_id)
            .await
            .map_err(|_| HttpError::not_found("backfill job not found"))?;
        Ok(BackfillJobResponse { job })
    }

    async fn cancel_backfill_job(
        &self,
        job_id: uuid::Uuid,
    ) -> Result<CancelBackfillJobResponse, HttpError> {
        let job = self
            .store
            .request_backfill_cancel(job_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(CancelBackfillJobResponse {
            job_id,
            status: job.status,
            cancel_requested: true,
        })
    }

    async fn trade_pnl_summary(&self) -> Result<serde_json::Value, HttpError> {
        let summary = self
            .store
            .trade_pnl_summary()
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(summary)
    }

    async fn trade_pnl_wallets(
        &self,
        request: control_http::TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError> {
        self.store
            .wallet_trade_performance_rows(request.limit.unwrap_or(50).clamp(1, 500))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn trade_pnl_open_positions(
        &self,
        request: control_http::TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError> {
        self.store
            .open_trade_position_rows(request.limit.unwrap_or(50).clamp(1, 500))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn trade_pnl_recent_exits(
        &self,
        request: control_http::TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError> {
        self.store
            .recent_trade_exit_rows(request.limit.unwrap_or(50).clamp(1, 500))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn trade_pnl_backfill(&self) -> Result<serde_json::Value, HttpError> {
        let report = refresh_trade_pnl_with_config(
            &self.store,
            Some(self.venue.as_ref()),
            &self.trade_pnl_config,
        )
        .await
        .map_err(|error| HttpError::internal(error.to_string()))?;
        serde_json::to_value(report).map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn trade_pnl_mark_now(&self) -> Result<serde_json::Value, HttpError> {
        let report = mark_trade_pnl_now_with_config(
            &self.store,
            Some(self.venue.as_ref()),
            &self.trade_pnl_config,
        )
        .await
        .map_err(|error| HttpError::internal(error.to_string()))?;
        serde_json::to_value(report).map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_status(&self) -> Result<LiveVenueStatus, HttpError> {
        self.venue
            .live_status()
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_halt(&self) -> Result<serde_json::Value, HttpError> {
        let disable_result = self
            .venue
            .set_live_entries_enabled(false, Some("manual_live_halt".to_string()))
            .await;
        let cancel_result = self.venue.cancel_all().await;
        let reconcile_result = self.venue.reconcile().await;
        self.store
            .insert_service_event(&ServiceEvent::new(
                "manual_live_halt",
                serde_json::json!({
                    "disable_result": disable_result.as_ref().map(|status| serde_json::json!({"entries_enabled": status.entries_enabled, "reason": status.reason})).unwrap_or_else(|error| serde_json::json!({"error": error.to_string()})),
                    "cancel_result": cancel_result.as_ref().map(|count| serde_json::json!({"cancelled": count})).unwrap_or_else(|error| serde_json::json!({"error": error.to_string()})),
                    "reconcile_result": reconcile_result.as_ref().map(|report| serde_json::json!({
                        "open_orders": report.open_orders,
                        "mismatches_found": report.mismatches_found,
                        "unresolved_count": report.unresolved_count
                    })).unwrap_or_else(|error| serde_json::json!({"error": error.to_string()}))
                }),
            ))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(serde_json::json!({
            "halted": true,
            "cancelled": cancel_result.ok(),
            "reconciled": reconcile_result.ok()
        }))
    }

    async fn live_reconcile(&self) -> Result<serde_json::Value, HttpError> {
        let report = self
            .venue
            .reconcile()
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        serde_json::to_value(report).map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_set_entries_enabled(&self, enabled: bool) -> Result<LiveVenueStatus, HttpError> {
        self.venue
            .set_live_entries_enabled(enabled, (!enabled).then(|| "manual_disable".to_string()))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    install_tls_crypto_provider();
    init_tracing();

    let config = AppConfig::from_env()?;
    info!(
        service = "polymarket-bot",
        mode = ?config.execution_mode,
        scan_enabled = config.scan_enabled,
        signal2_enabled = config.signal2_enabled,
        signal3_enabled = config.signal3_enabled,
        "starting Polymarket bot"
    );

    let store = Store::connect(&config.postgres).await?;
    store.healthcheck().await?;
    store
        .insert_service_event(&ServiceEvent::new(
            "service_started",
            serde_json::json!({
                "mode": format!("{:?}", config.execution_mode).to_ascii_lowercase(),
                "scan_enabled": config.scan_enabled,
                "kafka_required": false
            }),
        ))
        .await?;

    let gamma = GammaClient::new(config.gamma_base_url.clone());
    let clob = ClobClient::new(config.clob_base_url.clone());
    let data_api = DataApiClient::new(config.data_api_base_url.clone());
    let venue: Arc<dyn ExecutionVenue> = match config.execution_mode {
        ExecutionMode::Sim => Arc::new(SimVenue::with_clob(
            clob.clone(),
            config.risk.taker_fee_rate,
        )),
        ExecutionMode::Live => Arc::new(LiveVenue::new(
            config.live_confirm,
            config.live.clone(),
            config.live.clob_api_base_url.clone(),
            store.clone(),
        )?),
    };
    let scanner_config = ScannerConfig {
        target_size: config.risk.target_size,
        taker_fee_rate: config.risk.taker_fee_rate,
        bootstrap_threshold: config.risk.bootstrap_threshold,
        max_quote_age: chrono::Duration::seconds(45),
        min_event_markets: 2,
    };
    let risk_limits = RiskLimits {
        daily_pnl_target_usd: config.risk.daily_pnl_target_usd,
        ..RiskLimits::default()
    };
    let risk_state = RiskState::default();
    let trade_pnl_config = TradePnlConfig {
        exit_candidate_max_age: chrono::Duration::from_std(config.whale.exit_candidate_max_age)
            .unwrap_or_else(|_| chrono::Duration::seconds(900)),
    };

    let mut metrics = RuntimeMetrics::new();
    let shared_metrics = Arc::new(Mutex::new(metrics.clone()));
    if config.http.enabled {
        let control: control_http::SharedControlApi = Arc::new(RuntimeControl {
            store: store.clone(),
            data_api: data_api.clone(),
            venue: venue.clone(),
            metrics: shared_metrics.clone(),
            default_lookback_days: config.whale.lookback_days,
            default_min_trade_usd: config.whale.min_trade_usd,
            default_copy_trade_config: CopyTradeConfig::from(&config.whale),
            default_copy_execute_enabled: config.whale.copy_execute_enabled,
            default_max_pages: config.whale.max_pages,
            trade_pnl_config: trade_pnl_config.clone(),
        });
        let app = control_http::router(control, config.http.admin_token.clone());
        let bind = config.http.bind.clone();
        tokio::spawn(async move {
            match tokio::net::TcpListener::bind(&bind).await {
                Ok(listener) => {
                    info!(bind = %bind, "Polymarket control HTTP server listening");
                    if let Err(error) = axum::serve(listener, app).await {
                        error!(error = %error, "Polymarket control HTTP server failed");
                    }
                }
                Err(error) => {
                    error!(error = %error, bind = %bind, "failed to bind Polymarket control HTTP server")
                }
            }
        });
    }
    let mut scan_interval = tokio::time::interval(config.scan_interval);
    scan_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut health_interval = tokio::time::interval(config.health_interval);
    health_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut whale_live_interval = tokio::time::interval(config.whale.live_poll_interval);
    whale_live_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut seen_live_whale_trades = HashSet::new();

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            biased;

            _ = &mut shutdown => {
                warn!("shutdown signal received; cancelling open orders before exit");
                if let Err(error) = venue.cancel_all().await {
                    error!(error = %error, "cancel_all failed during shutdown");
                }
                store.insert_service_event(&ServiceEvent::new("service_stopped", serde_json::json!({}))).await.ok();
                break;
            }

            _ = health_interval.tick() => {
                let reconciliation = venue.reconcile().await;
                if let Err(error) = store.healthcheck().await {
                    error!(error = %error, "database healthcheck failed");
                }
                match reconciliation {
                    Ok(report) => {
                        info!(
                            target: "metrics",
                            scans = metrics.scans,
                            markets_seen = metrics.markets_seen,
                            markets_persisted = metrics.markets_persisted,
                            tokens_persisted = metrics.tokens_persisted,
                            books_persisted = metrics.books_persisted,
                            signals_inserted = metrics.signals_inserted,
                            positive_signals = metrics.positive_signals,
                            orders_inserted = metrics.orders_inserted,
                            fills_inserted = metrics.fills_inserted,
                            positions_upserted = metrics.positions_upserted,
                            scan_errors = metrics.scan_errors,
                            open_orders = report.open_orders,
                            balances_checked = report.balances_checked,
                            uptime_secs = (Utc::now() - metrics.started_at).num_seconds(),
                            "polymarket bot liveness ok"
                        );
                        store.record_daily_metric("runtime", serde_json::json!({
                            "scans": metrics.scans,
                            "markets_seen": metrics.markets_seen,
                            "markets_persisted": metrics.markets_persisted,
                            "tokens_persisted": metrics.tokens_persisted,
                            "books_persisted": metrics.books_persisted,
                            "signals_inserted": metrics.signals_inserted,
                            "positive_signals": metrics.positive_signals,
                            "orders_inserted": metrics.orders_inserted,
                            "fills_inserted": metrics.fills_inserted,
                            "positions_upserted": metrics.positions_upserted,
                            "scan_errors": metrics.scan_errors,
                            "open_orders": report.open_orders
                        })).await.ok();
                        if let Ok(mut shared) = shared_metrics.lock() {
                            *shared = metrics.clone();
                        }
                    }
                    Err(error) => {
                        warn!(error = %error, "venue reconciliation failed");
                    }
                }
            }

            _ = scan_interval.tick(), if config.scan_enabled => {
                metrics.scans = metrics.scans.saturating_add(1);
                let scan = tokio::time::timeout(
                    SCAN_CYCLE_TIMEOUT,
                    run_scan_once(
                        &gamma,
                        &clob,
                        &store,
                        venue.as_ref(),
                        &scanner_config,
                        &risk_limits,
                        &risk_state,
                        config.max_markets_per_scan,
                    ),
                )
                .await;
                match scan {
                    Ok(Ok(scan_run)) => {
                        let scan_report = scan_run.scanner;
                        metrics.markets_seen = metrics
                            .markets_seen
                            .saturating_add(scan_run.markets_seen as u64);
                        metrics.markets_persisted = metrics
                            .markets_persisted
                            .saturating_add(scan_run.markets_persisted as u64);
                        metrics.tokens_persisted = metrics
                            .tokens_persisted
                            .saturating_add(scan_report.tokens_persisted as u64);
                        metrics.books_persisted = metrics
                            .books_persisted
                            .saturating_add(scan_report.books_persisted as u64);
                        metrics.signals_inserted = metrics
                            .signals_inserted
                            .saturating_add(scan_report.signals_inserted as u64);
                        metrics.positive_signals = metrics
                            .positive_signals
                            .saturating_add(scan_report.positive_signals as u64);
                        metrics.orders_inserted = metrics
                            .orders_inserted
                            .saturating_add(scan_report.orders_inserted as u64);
                        metrics.fills_inserted = metrics
                            .fills_inserted
                            .saturating_add(scan_report.fills_inserted as u64);
                        metrics.positions_upserted = metrics
                            .positions_upserted
                            .saturating_add(scan_report.positions_upserted as u64);
                        metrics.scan_errors = metrics
                            .scan_errors
                            .saturating_add((scan_report.errors + scan_run.persist_errors) as u64);
                    }
                    Ok(Err(error)) => {
                        metrics.scan_errors = metrics.scan_errors.saturating_add(1);
                        warn!(error = %error, "Gamma scan failed");
                    }
                    Err(_) => {
                        metrics.scan_errors = metrics.scan_errors.saturating_add(1);
                        warn!("market scan timed out");
                    }
                }
            }

            _ = whale_live_interval.tick(), if config.whale.live_enabled && config.whale.copy_trade_enabled => {
                metrics.whale_live_poll_cycles = metrics.whale_live_poll_cycles.saturating_add(1);
                let poll = tokio::time::timeout(
                    WHALE_POLL_TIMEOUT,
                    poll_live_whales_once(
                        &store,
                        &data_api,
                        venue.as_ref(),
                        &config,
                        &mut seen_live_whale_trades,
                    ),
                )
                .await;
                match poll {
                    Ok(Ok(copy_summary)) => {
                        metrics.whale_live_trades_seen = metrics
                            .whale_live_trades_seen
                            .saturating_add(copy_summary.trades_evaluated as u64);
                        metrics.copy_trade_signals = metrics
                            .copy_trade_signals
                            .saturating_add(copy_summary.signals_inserted as u64);
                        metrics.copy_trade_orders = metrics
                            .copy_trade_orders
                            .saturating_add(copy_summary.orders_inserted as u64);
                        metrics.copy_trade_fills = metrics
                            .copy_trade_fills
                            .saturating_add(copy_summary.fills_inserted as u64);
                    }
                    Ok(Err(error)) => {
                        metrics.whale_live_poll_errors = metrics
                            .whale_live_poll_errors
                            .saturating_add(1);
                        warn!(error = %error, "live whale polling failed");
                    }
                    Err(_) => {
                        metrics.whale_live_poll_errors = metrics
                            .whale_live_poll_errors
                            .saturating_add(1);
                        warn!("live whale polling timed out");
                    }
                }
            }

            _ = tokio::time::sleep(Duration::from_secs(3600)), if !config.scan_enabled => {
                info!("scan disabled; service idling");
            }
        }
    }

    Ok(())
}

async fn poll_live_whales_once(
    store: &Store,
    data_api: &DataApiClient,
    venue: &dyn ExecutionVenue,
    config: &AppConfig,
    seen_trade_keys: &mut HashSet<String>,
) -> Result<polymarket_bot::backfill::CopyTradeRunSummary> {
    if config.execution_mode == ExecutionMode::Live && config.whale.copy_execute_enabled {
        let live_status = venue.live_status().await?;
        if !live_status.entries_enabled {
            debug!(
                reason = live_status.reason.as_deref().unwrap_or("live_not_ready"),
                "skipping live whale poll until live entries are enabled"
            );
            return Ok(polymarket_bot::backfill::CopyTradeRunSummary::default());
        }
    }

    let mut trades = Vec::new();
    for page in 0..config.whale.live_max_pages {
        let offset = page * config.whale.live_page_limit;
        let page_trades = fetch_whale_trade_page(
            data_api,
            config.whale.live_page_limit,
            offset,
            config.whale.min_trade_usd,
        )
        .await?;
        if page_trades.is_empty() {
            break;
        }
        for mut trade in page_trades {
            let key = trade.transaction_hash.clone().unwrap_or_else(|| {
                format!(
                    "{}:{}:{}:{}",
                    trade.proxy_wallet, trade.asset, trade.timestamp_utc, trade.cash_value
                )
            });
            if !seen_trade_keys.insert(key) {
                continue;
            }
            store.ensure_whale_wallet(&trade).await?;
            if store.upsert_whale_trade(&trade).await? {
                store.record_wallet_observed_trade(&trade).await?;
            } else if let Some(existing_trade) = store.fetch_whale_trade_by_identity(&trade).await?
            {
                trade = existing_trade;
            } else if let Some(existing_trade) =
                store.fetch_whale_trade_by_copy_identity(&trade).await?
            {
                trade = existing_trade;
            } else {
                debug!(
                    proxy_wallet = %trade.proxy_wallet,
                    asset = %trade.asset,
                    timestamp_utc = %trade.timestamp_utc,
                    "skipping live copy-trade signal because persisted source trade could not be resolved"
                );
                continue;
            }
            trades.push(trade);
        }
    }

    ensure_live_wallet_performance(store, data_api, &trades, config.whale.live_page_limit, 1)
        .await?;

    run_copy_trade_signal_engine(
        store,
        Some(venue),
        &trades,
        &CopyTradeRunConfig {
            copy_trade: CopyTradeConfig::from(&config.whale),
            execute_signals: config.whale.copy_execute_enabled,
        },
        false,
    )
    .await
}

async fn ensure_live_wallet_performance(
    store: &Store,
    data_api: &DataApiClient,
    trades: &[polymarket_bot::models::WhaleTrade],
    page_limit: usize,
    max_pages: usize,
) -> Result<()> {
    let mut checked_wallets = HashSet::new();
    for trade in trades {
        if !checked_wallets.insert(trade.proxy_wallet.clone()) {
            continue;
        }
        if store
            .fetch_latest_wallet_performance(&trade.proxy_wallet)
            .await?
            .is_some()
        {
            continue;
        }
        let positions = fetch_closed_positions_for_wallet(
            data_api,
            &trade.proxy_wallet,
            page_limit.max(50),
            max_pages.max(1),
        )
        .await?;
        let performance_score = score_closed_position_performance(&trade.proxy_wallet, &positions);
        let performance = performance_score
            .clone()
            .into_wallet_performance(Utc::now(), serde_json::to_value(&positions)?);
        let wallet_score = performance_score.into_wallet_score();
        store.upsert_wallet_performance(&performance).await?;
        store.upsert_wallet_score(&wallet_score).await?;
    }
    Ok(())
}

async fn run_scan_once(
    gamma: &GammaClient,
    clob: &ClobClient,
    store: &Store,
    venue: &dyn ExecutionVenue,
    scanner_config: &ScannerConfig,
    risk_limits: &RiskLimits,
    risk_state: &RiskState,
    max_markets_per_scan: usize,
) -> Result<ScanRunReport> {
    let markets = gamma.fetch_active_events(max_markets_per_scan).await?;
    let mut report = ScanRunReport {
        markets_seen: markets.len(),
        ..ScanRunReport::default()
    };

    for market in &markets {
        if market.closed || market.archived || !market.active {
            continue;
        }
        if let Err(error) = store.insert_market(market).await {
            report.persist_errors += 1;
            warn!(
                error = %error,
                market_id = %market.market_id,
                "failed to persist market"
            );
        } else {
            report.markets_persisted += 1;
        }
    }

    report.scanner = scan_markets_for_signal1(
        &markets,
        clob,
        store,
        venue,
        scanner_config,
        risk_limits,
        risk_state,
    )
    .await;

    Ok(report)
}

fn install_tls_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .json()
        .with_current_span(false)
        .with_span_list(false)
        .init();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            warn!(error = %error, "failed to install ctrl-c handler");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => warn!(error = %error, "failed to install terminate signal handler"),
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
