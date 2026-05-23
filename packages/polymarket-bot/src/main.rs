use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{bail, Result};
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
    execution::{
        live::LiveVenue, sim::SimVenue, ExecutionVenue, LiveIdentityDiagnostics, LiveVenueStatus,
    },
    gamma::GammaClient,
    http as control_http,
    http::{
        BackfillJobResponse, BackfillJobsResponse, CancelBackfillJobResponse, ControlApi,
        HttpError, MetricsResponse, TradingProcessResetResponse, TradingProcessResponse,
        TradingProcessStatusResponse, TradingProcessesResponse,
    },
    models::{TradingProcess, WhalePollCheckpoint},
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
const WHALE_POLL_SCHEDULER_INTERVAL: Duration = Duration::from_secs(5);

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
    clob: ClobClient,
    venues: ExecutionVenues,
    metrics: Arc<Mutex<RuntimeMetrics>>,
    default_process_id: Option<uuid::Uuid>,
    trade_pnl_config: TradePnlConfig,
}

#[derive(Clone)]
struct ExecutionVenues {
    sim: Arc<dyn ExecutionVenue>,
    paper: Arc<dyn ExecutionVenue>,
    live: Option<Arc<dyn ExecutionVenue>>,
    health: Arc<dyn ExecutionVenue>,
}

impl ExecutionVenues {
    fn for_mode(&self, mode: ExecutionMode) -> Result<Arc<dyn ExecutionVenue>> {
        match mode {
            ExecutionMode::Sim => Ok(self.sim.clone()),
            ExecutionMode::Paper => Ok(self.paper.clone()),
            ExecutionMode::Live => self
                .live
                .clone()
                .ok_or_else(|| anyhow::anyhow!("live execution is not configured")),
        }
    }
}

#[derive(Debug, Clone)]
struct ProcessRuntimeConfig {
    process_id: uuid::Uuid,
    execution_mode: ExecutionMode,
    execute_signals: bool,
    backfill_enabled: bool,
    live_enabled: bool,
    lookback_days: u32,
    min_trade_usd: rust_decimal::Decimal,
    page_limit: usize,
    max_pages: usize,
    live_page_limit: usize,
    live_max_pages: usize,
    live_poll_interval_secs: i64,
    wallets: Vec<String>,
    market_ids: Vec<String>,
    copy_trade: CopyTradeConfig,
}

#[derive(Debug, Default)]
struct ScanRunReport {
    markets_seen: usize,
    markets_persisted: usize,
    persist_errors: usize,
    scanner: ScannerCycleReport,
}

impl RuntimeControl {
    async fn resolve_process_config(
        &self,
        process_id: Option<uuid::Uuid>,
    ) -> Result<ProcessRuntimeConfig, HttpError> {
        let process = if let Some(process_id) = process_id.or(self.default_process_id) {
            self.store
                .get_trading_process(process_id)
                .await
                .map_err(|error| HttpError::internal(error.to_string()))?
                .ok_or_else(|| HttpError::not_found("trading process not found"))?
        } else {
            self.store
                .list_trading_processes(100)
                .await
                .map_err(|error| HttpError::internal(error.to_string()))?
                .into_iter()
                .find(|process| {
                    process.process_type == "copy_trade"
                        && process.enabled
                        && process.status == "running"
                        && process
                            .metadata
                            .get("source")
                            .and_then(|value| value.as_str())
                            == Some("infra/processes")
                })
                .ok_or_else(|| {
                    HttpError::bad_request(
                        "process_id is required until an infra trading process is running",
                    )
                })?
        };
        runtime_config_from_process(&process)
            .map_err(|error| HttpError::bad_request(error.to_string()))
    }
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
        let process_config = self.resolve_process_config(request.process_id).await?;
        if !process_config.backfill_enabled {
            return Err(HttpError::bad_request(
                "trading process whale backfill is disabled",
            ));
        }
        let venue = self
            .venues
            .for_mode(process_config.execution_mode)
            .map_err(|error| HttpError::bad_request(error.to_string()))?;
        let backfill_request = WhaleBackfillRequest {
            lookback_days: request
                .lookback_days
                .unwrap_or(process_config.lookback_days as i32)
                .max(0) as u32,
            min_trade_usd: request
                .min_trade_usd
                .unwrap_or(process_config.min_trade_usd),
            market_ids: if request.market_ids.is_empty() {
                process_config.market_ids.clone()
            } else {
                request.market_ids
            },
            event_ids: Vec::new(),
            wallets: if request.wallet_addresses.is_empty() {
                process_config.wallets.clone()
            } else {
                request.wallet_addresses
            },
            mode: request.mode.unwrap_or(BackfillMode::Full),
            dry_run: request.dry_run,
            limit: request.page_limit.unwrap_or(process_config.page_limit),
            max_pages: request.max_pages.unwrap_or(process_config.max_pages),
            process_id: Some(process_config.process_id),
            copy_min_wallet_score: process_config.copy_trade.min_wallet_score,
            copy_size_fraction: process_config.copy_trade.copy_size_fraction,
            copy_max_size_usd: process_config.copy_trade.max_copy_size_usd,
            copy_trade_config: Some(process_config.copy_trade),
            execute_signals: false,
        };
        let job_id = enqueue_and_spawn_with_venue(
            self.store.clone(),
            self.data_api.clone(),
            Some(venue),
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
        let process_config = self.resolve_process_config(request.process_id).await?;
        let venue = self
            .venues
            .for_mode(process_config.execution_mode)
            .map_err(|error| HttpError::bad_request(error.to_string()))?;
        let mut copy_trade_config = process_config.copy_trade;
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
                .unwrap_or(process_config.lookback_days as i32)
                .max(0) as u32,
            min_trade_usd: request
                .min_trade_usd
                .unwrap_or(process_config.min_trade_usd),
            market_ids: process_config.market_ids,
            event_ids: Vec::new(),
            wallets: process_config.wallets,
            mode: BackfillMode::CopyTradeBacktest,
            dry_run: request.dry_run,
            limit: request.page_limit.unwrap_or(process_config.page_limit),
            max_pages: request.max_pages.unwrap_or(process_config.max_pages),
            process_id: Some(process_config.process_id),
            copy_min_wallet_score: copy_trade_config.min_wallet_score,
            copy_size_fraction: copy_trade_config.copy_size_fraction,
            copy_max_size_usd: copy_trade_config.max_copy_size_usd,
            copy_trade_config: Some(copy_trade_config),
            execute_signals: request.execute_signals && process_config.execute_signals,
        };
        let job_id = enqueue_and_spawn_with_venue(
            self.store.clone(),
            self.data_api.clone(),
            Some(venue),
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
        let process_config = self.resolve_process_config(request.process_id).await?;
        let venue = self
            .venues
            .for_mode(process_config.execution_mode)
            .map_err(|error| HttpError::bad_request(error.to_string()))?;
        let backfill_request = WhaleBackfillRequest {
            lookback_days: request
                .lookback_days
                .unwrap_or(process_config.lookback_days as i32)
                .max(0) as u32,
            min_trade_usd: request
                .min_trade_usd
                .unwrap_or(process_config.min_trade_usd),
            market_ids: process_config.market_ids,
            event_ids: Vec::new(),
            wallets: process_config.wallets,
            mode: BackfillMode::CopyTradeBacktest,
            dry_run: true,
            limit: request.page_limit.unwrap_or(process_config.page_limit),
            max_pages: request.max_pages.unwrap_or(process_config.max_pages),
            process_id: Some(process_config.process_id),
            copy_min_wallet_score: process_config.copy_trade.min_wallet_score,
            copy_size_fraction: process_config.copy_trade.copy_size_fraction,
            copy_max_size_usd: process_config.copy_trade.max_copy_size_usd,
            copy_trade_config: Some(process_config.copy_trade),
            execute_signals: false,
        };
        let job_id = enqueue_and_spawn_with_venue(
            self.store.clone(),
            self.data_api.clone(),
            Some(venue),
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

    async fn trade_pnl_mark_health(
        &self,
        request: control_http::TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError> {
        self.store
            .trade_pnl_mark_health(
                request.process_id,
                request.limit.unwrap_or(50).clamp(1, 500),
            )
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
            Some(self.venues.health.as_ref()),
            Some(&self.clob),
            &self.trade_pnl_config,
        )
        .await
        .map_err(|error| HttpError::internal(error.to_string()))?;
        serde_json::to_value(report).map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn trade_pnl_mark_now(&self) -> Result<serde_json::Value, HttpError> {
        let report = mark_trade_pnl_now_with_config(
            &self.store,
            Some(self.venues.health.as_ref()),
            Some(&self.clob),
            &self.trade_pnl_config,
        )
        .await
        .map_err(|error| HttpError::internal(error.to_string()))?;
        serde_json::to_value(report).map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_status(&self) -> Result<LiveVenueStatus, HttpError> {
        self.venues
            .for_mode(ExecutionMode::Live)
            .map_err(|error| HttpError::bad_request(error.to_string()))?
            .live_status()
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_identity_diagnostics(&self) -> Result<LiveIdentityDiagnostics, HttpError> {
        self.venues
            .for_mode(ExecutionMode::Live)
            .map_err(|error| HttpError::bad_request(error.to_string()))?
            .live_identity_diagnostics()
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_halt(&self) -> Result<serde_json::Value, HttpError> {
        let live = self
            .venues
            .for_mode(ExecutionMode::Live)
            .map_err(|error| HttpError::bad_request(error.to_string()))?;
        let disable_result = live
            .set_live_entries_enabled(false, Some("manual_live_halt".to_string()))
            .await;
        let cancel_result = live.cancel_all().await;
        let reconcile_result = live.reconcile().await;
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
            .venues
            .for_mode(ExecutionMode::Live)
            .map_err(|error| HttpError::bad_request(error.to_string()))?
            .reconcile()
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        serde_json::to_value(report).map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_set_entries_enabled(&self, enabled: bool) -> Result<LiveVenueStatus, HttpError> {
        self.venues
            .for_mode(ExecutionMode::Live)
            .map_err(|error| HttpError::bad_request(error.to_string()))?
            .set_live_entries_enabled(enabled, (!enabled).then(|| "manual_disable".to_string()))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn create_trading_process(
        &self,
        request: control_http::CreateTradingProcessRequest,
    ) -> Result<TradingProcessResponse, HttpError> {
        let name = request.name.trim();
        if name.is_empty() {
            return Err(HttpError::bad_request("trading process name is required"));
        }
        let process_type = request.process_type.trim();
        if process_type.is_empty() {
            return Err(HttpError::bad_request("trading process type is required"));
        }
        let process_scope = request.process_scope.trim();
        if process_scope.is_empty() {
            return Err(HttpError::bad_request("trading process scope is required"));
        }
        let process_key = request.process_key.as_deref().map(str::trim);
        if matches!(process_key, Some("")) {
            return Err(HttpError::bad_request(
                "trading process key cannot be empty",
            ));
        }
        let process = self
            .store
            .create_trading_process(
                name,
                process_type,
                process_scope,
                process_key,
                request.enabled,
                request.config,
                request.metadata,
            )
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(TradingProcessResponse { process })
    }

    async fn list_trading_processes(
        &self,
        request: control_http::ListTradingProcessesRequest,
    ) -> Result<TradingProcessesResponse, HttpError> {
        let processes = self
            .store
            .list_trading_processes(request.limit.unwrap_or(50).clamp(1, 500))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(TradingProcessesResponse { processes })
    }

    async fn get_trading_process(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcessResponse, HttpError> {
        let process = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        Ok(TradingProcessResponse { process })
    }

    async fn upsert_trading_process_by_key(
        &self,
        process_key: String,
        request: control_http::UpsertTradingProcessByKeyRequest,
    ) -> Result<TradingProcessResponse, HttpError> {
        let key = process_key.trim();
        if key.is_empty() {
            return Err(HttpError::bad_request("trading process key is required"));
        }
        let name = request.name.trim();
        if name.is_empty() {
            return Err(HttpError::bad_request("trading process name is required"));
        }
        let process_type = request.process_type.trim();
        if process_type.is_empty() {
            return Err(HttpError::bad_request("trading process type is required"));
        }
        let process_scope = request.process_scope.trim();
        if process_scope.is_empty() {
            return Err(HttpError::bad_request("trading process scope is required"));
        }
        let status = request.status.trim();
        if status.is_empty() {
            return Err(HttpError::bad_request("trading process status is required"));
        }
        let process = self
            .store
            .upsert_trading_process_by_key(
                name,
                process_type,
                process_scope,
                key,
                request.enabled,
                status,
                request.config,
                request.metadata,
            )
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(TradingProcessResponse { process })
    }

    async fn get_trading_process_status(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcessStatusResponse, HttpError> {
        let status = self
            .store
            .trading_process_status(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        Ok(TradingProcessStatusResponse { process_id, status })
    }

    async fn update_trading_process(
        &self,
        process_id: uuid::Uuid,
        request: control_http::UpdateTradingProcessRequest,
    ) -> Result<TradingProcessResponse, HttpError> {
        let name = request.name.as_deref().map(str::trim);
        if matches!(name, Some("")) {
            return Err(HttpError::bad_request(
                "trading process name cannot be empty",
            ));
        }
        let process_type = request.process_type.as_deref().map(str::trim);
        if matches!(process_type, Some("")) {
            return Err(HttpError::bad_request(
                "trading process type cannot be empty",
            ));
        }
        let process_scope = request.process_scope.as_deref().map(str::trim);
        if matches!(process_scope, Some("")) {
            return Err(HttpError::bad_request(
                "trading process scope cannot be empty",
            ));
        }
        let process_key = request
            .process_key
            .as_ref()
            .map(|key| key.as_deref().map(str::trim));
        if matches!(process_key, Some(Some(""))) {
            return Err(HttpError::bad_request(
                "trading process key cannot be empty",
            ));
        }
        let status = request.status.as_deref().map(str::trim);
        if matches!(status, Some("")) {
            return Err(HttpError::bad_request(
                "trading process status cannot be empty",
            ));
        }
        let process = self
            .store
            .update_trading_process(
                process_id,
                name,
                process_type,
                process_scope,
                process_key,
                request.enabled,
                status,
                request.config,
                request.metadata,
            )
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        Ok(TradingProcessResponse { process })
    }

    async fn start_trading_process(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcessResponse, HttpError> {
        let process = self
            .store
            .start_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        Ok(TradingProcessResponse { process })
    }

    async fn stop_trading_process(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcessResponse, HttpError> {
        let process = self
            .store
            .stop_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        Ok(TradingProcessResponse { process })
    }

    async fn reset_trading_process_simulation(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcessResetResponse, HttpError> {
        let process = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        let execution_mode = process.config.effective_execution().mode;
        if execution_mode != "sim" {
            return Err(HttpError::bad_request(
                "only sim trading processes can be reset through this endpoint",
            ));
        }
        let report = self
            .store
            .reset_trading_process_data(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        Ok(TradingProcessResetResponse { report })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    install_tls_crypto_provider();
    init_tracing();

    let config = AppConfig::from_env()?;
    info!(
        service = "polymarket-bot",
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
                "execution_control": "trade_processes",
                "scan_enabled": config.scan_enabled,
                "kafka_required": false
            }),
        ))
        .await?;

    let gamma = GammaClient::new(config.gamma_base_url.clone());
    let clob = ClobClient::new(config.clob_base_url.clone());
    let data_api = DataApiClient::new(config.data_api_base_url.clone());
    let sim_venue: Arc<dyn ExecutionVenue> = Arc::new(SimVenue::with_clob(
        clob.clone(),
        config.risk.taker_fee_rate,
    ));
    let paper_venue: Arc<dyn ExecutionVenue> = Arc::new(SimVenue::paper_with_clob(
        clob.clone(),
        config.risk.taker_fee_rate,
    ));
    let live_venue: Option<Arc<dyn ExecutionVenue>> = if config.live.live_auth_available() {
        Some(Arc::new(LiveVenue::new(
            config.live.clone(),
            config.live.clob_api_base_url.clone(),
            store.clone(),
        )?))
    } else {
        None
    };
    let venues = ExecutionVenues {
        sim: sim_venue.clone(),
        paper: paper_venue.clone(),
        live: live_venue.clone(),
        health: sim_venue.clone(),
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
        ..TradePnlConfig::default()
    };

    let mut metrics = RuntimeMetrics::new();
    let shared_metrics = Arc::new(Mutex::new(metrics.clone()));
    if config.http.enabled {
        let control: control_http::SharedControlApi = Arc::new(RuntimeControl {
            store: store.clone(),
            data_api: data_api.clone(),
            clob: clob.clone(),
            venues: venues.clone(),
            metrics: shared_metrics.clone(),
            default_process_id: None,
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
    let mut whale_live_interval = tokio::time::interval(WHALE_POLL_SCHEDULER_INTERVAL);
    whale_live_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut seen_live_whale_trades_by_process: HashMap<uuid::Uuid, HashSet<String>> =
        HashMap::new();
    let mut last_live_whale_poll_by_process: HashMap<uuid::Uuid, chrono::DateTime<Utc>> =
        HashMap::new();

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            biased;

            _ = &mut shutdown => {
                warn!("shutdown signal received; cancelling open orders before exit");
                if let Err(error) = venues.health.cancel_all().await {
                    error!(error = %error, "cancel_all failed during shutdown");
                }
                store.insert_service_event(&ServiceEvent::new("service_stopped", serde_json::json!({}))).await.ok();
                break;
            }

            _ = health_interval.tick() => {
                let reconciliation = venues.health.reconcile().await;
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
                        venues.health.as_ref(),
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

            _ = whale_live_interval.tick() => {
                metrics.whale_live_poll_cycles = metrics.whale_live_poll_cycles.saturating_add(1);
                match store.list_trading_processes(100).await {
                    Ok(processes) => {
                        for process in processes {
                            if process.process_type != "copy_trade" || !process.enabled || process.status != "running" {
                                continue;
                            }
                            let runtime_config = match runtime_config_from_process(&process) {
                                Ok(config) => config,
                                Err(error) => {
                                    metrics.whale_live_poll_errors = metrics
                                        .whale_live_poll_errors
                                        .saturating_add(1);
                                    warn!(error = %error, process_id = %process.process_id, "invalid trading process config for live whale polling");
                                    continue;
                                }
                            };
                            let now = Utc::now();
                            let poll_interval_secs = runtime_config.live_poll_interval_secs.max(1);
                            if last_live_whale_poll_by_process
                                .get(&process.process_id)
                                .map(|last_polled_at| {
                                    (now - *last_polled_at).num_seconds() < poll_interval_secs
                                })
                                .unwrap_or(false)
                            {
                                continue;
                            }
                            last_live_whale_poll_by_process.insert(process.process_id, now);
                            let seen_live_whale_trades = seen_live_whale_trades_by_process
                                .entry(process.process_id)
                                .or_default();
                            let poll = tokio::time::timeout(
                                WHALE_POLL_TIMEOUT,
                                poll_live_whales_once(
                                    &store,
                                    &data_api,
                                    &clob,
                                    &venues,
                                    process.process_id,
                                    seen_live_whale_trades,
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
                                    if let Err(error) = store
                                        .upsert_whale_poll_checkpoint(&WhalePollCheckpoint {
                                            checkpoint_name: process.process_id.to_string(),
                                            last_polled_at: Some(Utc::now()),
                                            next_cursor: None,
                                            last_trade_timestamp_utc: None,
                                            last_trade_id: None,
                                            pages_seen: runtime_config.live_max_pages as i64,
                                            trades_seen: copy_summary.trades_evaluated as i64,
                                            state: serde_json::json!({
                                                "status": "ok",
                                                "signals_inserted": copy_summary.signals_inserted,
                                                "orders_inserted": copy_summary.orders_inserted,
                                                "fills_inserted": copy_summary.fills_inserted,
                                                "rejections": copy_summary.rejections
                                            }),
                                        })
                                        .await
                                    {
                                        warn!(error = %error, process_id = %process.process_id, "failed to record live whale poll checkpoint");
                                    }
                                }
                                Ok(Err(error)) => {
                                    metrics.whale_live_poll_errors = metrics
                                        .whale_live_poll_errors
                                        .saturating_add(1);
                                    warn!(error = %error, process_id = %process.process_id, "live whale polling failed");
                                    if let Err(checkpoint_error) = store
                                        .upsert_whale_poll_checkpoint(&WhalePollCheckpoint {
                                            checkpoint_name: process.process_id.to_string(),
                                            last_polled_at: Some(Utc::now()),
                                            next_cursor: None,
                                            last_trade_timestamp_utc: None,
                                            last_trade_id: None,
                                            pages_seen: runtime_config.live_max_pages as i64,
                                            trades_seen: 0,
                                            state: serde_json::json!({
                                                "status": "error",
                                                "error": error.to_string()
                                            }),
                                        })
                                        .await
                                    {
                                        warn!(error = %checkpoint_error, process_id = %process.process_id, "failed to record live whale poll error checkpoint");
                                    }
                                }
                                Err(_) => {
                                    metrics.whale_live_poll_errors = metrics
                                        .whale_live_poll_errors
                                        .saturating_add(1);
                                    warn!(process_id = %process.process_id, "live whale polling timed out");
                                    if let Err(checkpoint_error) = store
                                        .upsert_whale_poll_checkpoint(&WhalePollCheckpoint {
                                            checkpoint_name: process.process_id.to_string(),
                                            last_polled_at: Some(Utc::now()),
                                            next_cursor: None,
                                            last_trade_timestamp_utc: None,
                                            last_trade_id: None,
                                            pages_seen: runtime_config.live_max_pages as i64,
                                            trades_seen: 0,
                                            state: serde_json::json!({
                                                "status": "timeout"
                                            }),
                                        })
                                        .await
                                    {
                                        warn!(error = %checkpoint_error, process_id = %process.process_id, "failed to record live whale poll timeout checkpoint");
                                    }
                                }
                            }
                        }
                    }
                    Err(error) => {
                        metrics.whale_live_poll_errors = metrics
                            .whale_live_poll_errors
                            .saturating_add(1);
                        warn!(error = %error, "failed to list trading processes for live whale polling");
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
    clob: &ClobClient,
    venues: &ExecutionVenues,
    process_id: uuid::Uuid,
    seen_trade_keys: &mut HashSet<String>,
) -> Result<polymarket_bot::backfill::CopyTradeRunSummary> {
    let Some(process) = store.get_trading_process(process_id).await? else {
        warn!(
            process_id = %process_id,
            "skipping live whale poll because trading process is missing"
        );
        return Ok(polymarket_bot::backfill::CopyTradeRunSummary::default());
    };
    if !process.enabled || process.status != "running" {
        debug!(
            process_id = %process_id,
            status = %process.status,
            enabled = process.enabled,
            "skipping live whale poll because trading process is not running"
        );
        return Ok(polymarket_bot::backfill::CopyTradeRunSummary::default());
    }
    let runtime_config = runtime_config_from_process(&process)?;
    if !runtime_config.live_enabled || !runtime_config.copy_trade.enabled {
        debug!(
            process_id = %process_id,
            live_enabled = runtime_config.live_enabled,
            copy_trade_enabled = runtime_config.copy_trade.enabled,
            "skipping live whale poll because process config disabled it"
        );
        return Ok(polymarket_bot::backfill::CopyTradeRunSummary::default());
    }
    let venue = venues.for_mode(runtime_config.execution_mode)?;

    if runtime_config.execution_mode == ExecutionMode::Live && runtime_config.execute_signals {
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
    for page in 0..runtime_config.live_max_pages {
        let offset = page * runtime_config.live_page_limit;
        let page_trades = fetch_whale_trade_page(
            data_api,
            runtime_config.live_page_limit,
            offset,
            runtime_config.min_trade_usd,
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

    ensure_live_wallet_performance(store, data_api, &trades, runtime_config.live_page_limit, 1)
        .await?;

    run_copy_trade_signal_engine(
        store,
        Some(venue.as_ref()),
        Some(clob),
        &trades,
        &CopyTradeRunConfig {
            process_id: Some(process_id),
            copy_trade: runtime_config.copy_trade,
            execute_signals: runtime_config.execute_signals,
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

fn runtime_config_from_process(process: &TradingProcess) -> Result<ProcessRuntimeConfig> {
    let execution = process.config.effective_execution();
    let backfill = process.config.effective_backfill();
    let copy_trade = process.config.effective_copy_trade();
    Ok(ProcessRuntimeConfig {
        process_id: process.process_id,
        execution_mode: parse_process_execution_mode(&execution.mode)?,
        execute_signals: execution.execute_signals,
        backfill_enabled: backfill.backfill_enabled,
        live_enabled: backfill.live_enabled,
        lookback_days: backfill.lookback_days,
        min_trade_usd: backfill.min_trade_usd,
        page_limit: backfill.page_limit,
        max_pages: backfill.max_pages,
        live_page_limit: backfill.live_page_limit,
        live_max_pages: backfill.live_max_pages,
        live_poll_interval_secs: backfill.live_poll_interval_secs,
        wallets: backfill.wallets,
        market_ids: backfill.market_ids,
        copy_trade: CopyTradeConfig {
            enabled: copy_trade.enabled,
            min_wallet_score: copy_trade.min_wallet_score,
            min_wallet_trades: copy_trade.min_wallet_trades,
            min_wallet_realized_pnl_usd: copy_trade.min_wallet_realized_pnl_usd,
            min_wallet_roi: copy_trade.min_wallet_roi,
            min_wallet_closed_positions: copy_trade.min_wallet_closed_positions,
            min_trade_usd: copy_trade.min_trade_usd,
            min_copy_size_usd: copy_trade.min_copy_size_usd,
            max_copy_size_usd: copy_trade.max_copy_size_usd,
            copy_size_fraction: copy_trade.copy_size_fraction,
            max_follow_lag_secs: copy_trade.max_follow_lag_secs,
            max_price_slippage_bps: copy_trade.max_price_slippage_bps,
            min_book_depth_usd: copy_trade.min_book_depth_usd,
            backtest_horizon_secs: copy_trade.backtest_horizon_secs,
            taker_fee_rate: copy_trade.taker_fee_rate,
            allow_sell_entries: copy_trade.allow_sell_entries,
        },
    })
}

fn parse_process_execution_mode(mode: &str) -> Result<ExecutionMode> {
    match mode.to_ascii_lowercase().as_str() {
        "sim" => Ok(ExecutionMode::Sim),
        "paper" => Ok(ExecutionMode::Paper),
        "live" => Ok(ExecutionMode::Live),
        other => bail!(
            "unsupported trading process execution mode={other}; expected sim, paper, or live"
        ),
    }
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
