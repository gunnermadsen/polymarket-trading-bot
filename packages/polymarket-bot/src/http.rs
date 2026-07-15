use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{header::AUTHORIZATION, HeaderMap, Request, StatusCode},
    middleware::{from_fn_with_state, Next},
    response::{IntoResponse, Response},
    routing::{get, post, put},
    Json, Router,
};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use crate::account_reconcile::{AccountReconcileReport, AccountReconcileRequest};

use crate::{
    backfill::BackfillMode,
    execution::{
        LiveIdentityDiagnostics, LiveOrderDryRunDiagnostics, LiveOrderDryRunRequest,
        LivePoly1271FunderProbeRequest, LivePoly1271FunderProbeResponse, LiveVenueStatus,
        LiveWalletAddressDiagnostics,
    },
    ingestion::job::{
        BackfillJob as IngestionBackfillJob, BackfillJobEvent as IngestionBackfillJobEvent,
        BackfillJobStatus as IngestionBackfillJobStatus,
        BackfillRequest as IngestionBackfillRequest, IngesterKey, TrainingReadiness,
    },
    models::{BackfillJob, BackfillJobStatus, BacktestRun, TradingProcess, TradingProcessConfig},
    replay::{BacktestReplayQueued, BacktestReplayRequest},
    store::TradingProcessResetReport,
};

const SERVICE_NAME: &str = "polymarket-bot";

pub type SharedControlApi = Arc<dyn ControlApi>;

#[derive(Clone)]
pub struct HttpState {
    control: SharedControlApi,
}

impl HttpState {
    pub fn new(control: SharedControlApi) -> Self {
        Self { control }
    }
}

#[derive(Debug, Clone)]
pub struct AdminAuth {
    bearer_token: String,
}

impl AdminAuth {
    pub fn new(bearer_token: impl Into<String>) -> Self {
        Self {
            bearer_token: bearer_token.into(),
        }
    }
}

#[async_trait]
pub trait ControlApi: Send + Sync + 'static {
    async fn health(&self) -> Result<HealthResponse, HttpError> {
        Ok(HealthResponse {
            service: SERVICE_NAME.to_string(),
            status: HealthStatus::Ok,
            checked_at: Utc::now(),
        })
    }

    async fn metrics(&self) -> Result<MetricsResponse, HttpError>;

    async fn btc_realtime_status(&self) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "BTC realtime status is not wired",
        ))
    }

    async fn btc_paper_experiment_status(&self) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "BTC paper experiment status is not wired",
        ))
    }

    async fn start_whales_backfill(
        &self,
        request: BackfillWhalesRequest,
    ) -> Result<BackfillJobResponse, HttpError>;

    async fn start_copy_trade_backtest(
        &self,
        request: CopyTradeBacktestRequest,
    ) -> Result<BackfillJobResponse, HttpError>;

    async fn start_copy_trade_calibration(
        &self,
        request: CopyTradeCalibrationRequest,
    ) -> Result<BackfillJobResponse, HttpError>;

    async fn replay_existing_copy_trades(
        &self,
        request: CopyTradeReplayRequest,
    ) -> Result<serde_json::Value, HttpError>;

    async fn start_backtest_replay(
        &self,
        request: BacktestReplayRequest,
    ) -> Result<BacktestReplayQueued, HttpError>;

    async fn get_backtest_run(
        &self,
        backtest_run_id: Uuid,
    ) -> Result<BacktestRunResponse, HttpError>;

    async fn list_backtest_runs(
        &self,
        request: ListBacktestRunsRequest,
    ) -> Result<BacktestRunsResponse, HttpError>;

    async fn list_backfill_jobs(&self) -> Result<BackfillJobsResponse, HttpError>;

    async fn get_backfill_job(&self, job_id: Uuid) -> Result<BackfillJobResponse, HttpError>;

    async fn cancel_backfill_job(
        &self,
        job_id: Uuid,
    ) -> Result<CancelBackfillJobResponse, HttpError>;

    async fn enqueue_ingestion_backfill(
        &self,
        _request: IngestionBackfillRequest,
    ) -> Result<IngestionBackfillEnqueueResponse, HttpError> {
        Err(HttpError::not_implemented(
            "generic backfill enqueue is not wired",
        ))
    }

    async fn list_ingestion_backfills(
        &self,
        _request: ListIngestionBackfillsRequest,
    ) -> Result<IngestionBackfillJobsResponse, HttpError> {
        Err(HttpError::not_implemented(
            "generic backfill registry is not wired",
        ))
    }

    async fn get_ingestion_backfill(
        &self,
        _job_id: Uuid,
    ) -> Result<IngestionBackfillJobResponse, HttpError> {
        Err(HttpError::not_implemented(
            "generic backfill registry is not wired",
        ))
    }

    async fn list_ingestion_backfill_events(
        &self,
        _job_id: Uuid,
        _request: ListIngestionBackfillEventsRequest,
    ) -> Result<IngestionBackfillEventsResponse, HttpError> {
        Err(HttpError::not_implemented(
            "generic backfill event registry is not wired",
        ))
    }

    async fn cancel_ingestion_backfill(
        &self,
        _job_id: Uuid,
    ) -> Result<IngestionBackfillCancelResponse, HttpError> {
        Err(HttpError::not_implemented(
            "generic backfill cancellation is not wired",
        ))
    }

    async fn ingestion_training_readiness(
        &self,
        _request: IngestionReadinessRequest,
    ) -> Result<TrainingReadiness, HttpError> {
        Err(HttpError::not_implemented(
            "BTC training-data readiness is not wired",
        ))
    }

    async fn trade_pnl_summary(&self) -> Result<serde_json::Value, HttpError>;

    async fn trade_pnl_wallets(
        &self,
        request: TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError>;

    async fn trade_pnl_open_positions(
        &self,
        request: TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError>;

    async fn trade_pnl_mark_health(
        &self,
        request: TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError>;

    async fn trade_pnl_recent_exits(
        &self,
        request: TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError>;

    async fn trade_pnl_backfill(&self) -> Result<serde_json::Value, HttpError>;

    async fn trade_pnl_mark_now(&self) -> Result<serde_json::Value, HttpError>;

    async fn recompute_mrs_scores(
        &self,
        request: MrsRecomputeRequest,
    ) -> Result<serde_json::Value, HttpError>;

    async fn recompute_mrs_segment_scores(
        &self,
        request: MrsRecomputeRequest,
    ) -> Result<serde_json::Value, HttpError>;

    async fn enqueue_wallet_score_refresh(
        &self,
        _request: WalletScoreRefreshEnqueueRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "wallet score refresh enqueue is not wired",
        ))
    }

    async fn process_wallet_score_refresh(
        &self,
        _request: WalletScoreRefreshProcessRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "wallet score refresh processing is not wired",
        ))
    }

    async fn list_wallet_score_refresh_jobs(
        &self,
        _request: WalletScoreRefreshJobsRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "wallet score refresh job listing is not wired",
        ))
    }

    async fn mrs_segment_summary(
        &self,
        request: TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError>;

    async fn recompute_expectancy_flow(
        &self,
        _request: ExpectancyFlowRecomputeRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "expectancy flow recompute is not wired",
        ))
    }

    async fn expectancy_flow_cells(
        &self,
        _request: ExpectancyFlowCellsRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "expectancy flow cell listing is not wired",
        ))
    }

    async fn expectancy_flow_wallet_cells(
        &self,
        _request: ExpectancyFlowWalletCellsRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "expectancy flow wallet cell listing is not wired",
        ))
    }

    async fn gamma_taxonomy_status(&self) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "Gamma taxonomy status is not wired",
        ))
    }

    async fn gamma_taxonomy_backfill(
        &self,
        _request: GammaTaxonomyBackfillRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "Gamma taxonomy backfill is not wired",
        ))
    }

    async fn live_status(&self) -> Result<LiveVenueStatus, HttpError>;

    async fn live_identity_diagnostics(&self) -> Result<LiveIdentityDiagnostics, HttpError>;

    async fn live_wallet_address_diagnostics(
        &self,
        candidate_addresses: Vec<String>,
    ) -> Result<LiveWalletAddressDiagnostics, HttpError>;

    async fn live_order_dry_run(
        &self,
        request: LiveOrderDryRunRequest,
    ) -> Result<LiveOrderDryRunDiagnostics, HttpError>;

    async fn live_poly1271_funder_probe(
        &self,
        request: LivePoly1271FunderProbeRequest,
    ) -> Result<LivePoly1271FunderProbeResponse, HttpError>;

    async fn live_halt(&self) -> Result<serde_json::Value, HttpError>;

    async fn live_reconcile(&self) -> Result<serde_json::Value, HttpError>;

    async fn live_account_reconcile(
        &self,
        _request: AccountReconcileRequest,
    ) -> Result<AccountReconcileReport, HttpError> {
        Err(HttpError::not_implemented(
            "live account reconciliation is not wired",
        ))
    }

    async fn live_set_entries_enabled(&self, enabled: bool) -> Result<LiveVenueStatus, HttpError>;

    async fn create_trading_process(
        &self,
        _request: CreateTradingProcessRequest,
    ) -> Result<TradingProcessResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }

    async fn list_trading_processes(
        &self,
        _request: ListTradingProcessesRequest,
    ) -> Result<TradingProcessesResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }

    async fn get_trading_process(
        &self,
        _process_id: Uuid,
    ) -> Result<TradingProcessResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }

    async fn upsert_trading_process_by_key(
        &self,
        _process_key: String,
        _request: UpsertTradingProcessByKeyRequest,
    ) -> Result<TradingProcessResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }

    async fn get_trading_process_status(
        &self,
        _process_id: Uuid,
    ) -> Result<TradingProcessStatusResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading process status is not wired",
        ))
    }

    async fn preview_trading_process_start(
        &self,
        _process_id: Uuid,
    ) -> Result<TradingProcessStartPreviewResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading process start preview is not wired",
        ))
    }

    async fn update_trading_process(
        &self,
        _process_id: Uuid,
        _request: UpdateTradingProcessRequest,
    ) -> Result<TradingProcessResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }

    async fn start_trading_process(
        &self,
        _process_id: Uuid,
    ) -> Result<TradingProcessResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }

    async fn stop_trading_process(
        &self,
        _process_id: Uuid,
    ) -> Result<TradingProcessResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }

    async fn reset_trading_process_simulation(
        &self,
        _process_id: Uuid,
    ) -> Result<TradingProcessResetResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading process reset is not wired",
        ))
    }
}

#[derive(Debug, Default)]
pub struct PlaceholderControlApi;

#[async_trait]
impl ControlApi for PlaceholderControlApi {
    async fn metrics(&self) -> Result<MetricsResponse, HttpError> {
        Err(HttpError::not_implemented("metrics provider is not wired"))
    }

    async fn start_whales_backfill(
        &self,
        _request: BackfillWhalesRequest,
    ) -> Result<BackfillJobResponse, HttpError> {
        Err(HttpError::not_implemented(
            "whales backfill runner is not wired",
        ))
    }

    async fn start_copy_trade_backtest(
        &self,
        _request: CopyTradeBacktestRequest,
    ) -> Result<BackfillJobResponse, HttpError> {
        Err(HttpError::not_implemented(
            "copy-trade backtest runner is not wired",
        ))
    }

    async fn start_copy_trade_calibration(
        &self,
        _request: CopyTradeCalibrationRequest,
    ) -> Result<BackfillJobResponse, HttpError> {
        Err(HttpError::not_implemented(
            "copy-trade calibration runner is not wired",
        ))
    }

    async fn list_backfill_jobs(&self) -> Result<BackfillJobsResponse, HttpError> {
        Err(HttpError::not_implemented(
            "backfill job registry is not wired",
        ))
    }

    async fn get_backfill_job(&self, _job_id: Uuid) -> Result<BackfillJobResponse, HttpError> {
        Err(HttpError::not_implemented(
            "backfill job registry is not wired",
        ))
    }

    async fn cancel_backfill_job(
        &self,
        _job_id: Uuid,
    ) -> Result<CancelBackfillJobResponse, HttpError> {
        Err(HttpError::not_implemented(
            "backfill job cancellation is not wired",
        ))
    }

    async fn trade_pnl_summary(&self) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented("trade PnL summary is not wired"))
    }

    async fn trade_pnl_wallets(
        &self,
        _request: TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "trade PnL wallets are not wired",
        ))
    }

    async fn trade_pnl_open_positions(
        &self,
        _request: TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "trade PnL open positions are not wired",
        ))
    }

    async fn trade_pnl_mark_health(
        &self,
        _request: TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "trade PnL mark health is not wired",
        ))
    }

    async fn trade_pnl_recent_exits(
        &self,
        _request: TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented("trade PnL exits are not wired"))
    }

    async fn trade_pnl_backfill(&self) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "trade PnL backfill is not wired",
        ))
    }

    async fn trade_pnl_mark_now(&self) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented("trade PnL marking is not wired"))
    }

    async fn replay_existing_copy_trades(
        &self,
        _request: CopyTradeReplayRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented("copy-trade replay is not wired"))
    }

    async fn start_backtest_replay(
        &self,
        _request: BacktestReplayRequest,
    ) -> Result<BacktestReplayQueued, HttpError> {
        Err(HttpError::not_implemented("backtest replay is not wired"))
    }

    async fn get_backtest_run(
        &self,
        _backtest_run_id: Uuid,
    ) -> Result<BacktestRunResponse, HttpError> {
        Err(HttpError::not_implemented(
            "backtest run lookup is not wired",
        ))
    }

    async fn list_backtest_runs(
        &self,
        _request: ListBacktestRunsRequest,
    ) -> Result<BacktestRunsResponse, HttpError> {
        Err(HttpError::not_implemented(
            "backtest run listing is not wired",
        ))
    }

    async fn recompute_mrs_scores(
        &self,
        _request: MrsRecomputeRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented("MRS recompute is not wired"))
    }

    async fn recompute_mrs_segment_scores(
        &self,
        _request: MrsRecomputeRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "MRS segment recompute is not wired",
        ))
    }

    async fn enqueue_wallet_score_refresh(
        &self,
        _request: WalletScoreRefreshEnqueueRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "wallet score refresh enqueue is not wired",
        ))
    }

    async fn process_wallet_score_refresh(
        &self,
        _request: WalletScoreRefreshProcessRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "wallet score refresh processing is not wired",
        ))
    }

    async fn list_wallet_score_refresh_jobs(
        &self,
        _request: WalletScoreRefreshJobsRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "wallet score refresh job listing is not wired",
        ))
    }

    async fn mrs_segment_summary(
        &self,
        _request: TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented(
            "MRS segment summary is not wired",
        ))
    }

    async fn live_status(&self) -> Result<LiveVenueStatus, HttpError> {
        Err(HttpError::not_implemented("live status is not wired"))
    }

    async fn live_identity_diagnostics(&self) -> Result<LiveIdentityDiagnostics, HttpError> {
        Err(HttpError::not_implemented(
            "live identity diagnostics are not wired",
        ))
    }

    async fn live_wallet_address_diagnostics(
        &self,
        _candidate_addresses: Vec<String>,
    ) -> Result<LiveWalletAddressDiagnostics, HttpError> {
        Err(HttpError::not_implemented(
            "live wallet diagnostics are not wired",
        ))
    }

    async fn live_order_dry_run(
        &self,
        _request: LiveOrderDryRunRequest,
    ) -> Result<LiveOrderDryRunDiagnostics, HttpError> {
        Err(HttpError::not_implemented(
            "live order dry-run is not wired",
        ))
    }

    async fn live_poly1271_funder_probe(
        &self,
        _request: LivePoly1271FunderProbeRequest,
    ) -> Result<LivePoly1271FunderProbeResponse, HttpError> {
        Err(HttpError::not_implemented(
            "live POLY_1271 funder probe is not wired",
        ))
    }

    async fn live_halt(&self) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented("live halt is not wired"))
    }

    async fn live_reconcile(&self) -> Result<serde_json::Value, HttpError> {
        Err(HttpError::not_implemented("live reconcile is not wired"))
    }

    async fn live_set_entries_enabled(&self, _enabled: bool) -> Result<LiveVenueStatus, HttpError> {
        Err(HttpError::not_implemented("live entry toggle is not wired"))
    }

    async fn create_trading_process(
        &self,
        _request: CreateTradingProcessRequest,
    ) -> Result<TradingProcessResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }

    async fn list_trading_processes(
        &self,
        _request: ListTradingProcessesRequest,
    ) -> Result<TradingProcessesResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }

    async fn get_trading_process(
        &self,
        _process_id: Uuid,
    ) -> Result<TradingProcessResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }

    async fn upsert_trading_process_by_key(
        &self,
        _process_key: String,
        _request: UpsertTradingProcessByKeyRequest,
    ) -> Result<TradingProcessResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }

    async fn get_trading_process_status(
        &self,
        _process_id: Uuid,
    ) -> Result<TradingProcessStatusResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading process status is not wired",
        ))
    }

    async fn preview_trading_process_start(
        &self,
        _process_id: Uuid,
    ) -> Result<TradingProcessStartPreviewResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading process start preview is not wired",
        ))
    }

    async fn update_trading_process(
        &self,
        _process_id: Uuid,
        _request: UpdateTradingProcessRequest,
    ) -> Result<TradingProcessResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }

    async fn start_trading_process(
        &self,
        _process_id: Uuid,
    ) -> Result<TradingProcessResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }

    async fn stop_trading_process(
        &self,
        _process_id: Uuid,
    ) -> Result<TradingProcessResponse, HttpError> {
        Err(HttpError::not_implemented(
            "trading processes are not wired",
        ))
    }
}

pub fn router(control: SharedControlApi, admin_bearer_token: impl Into<String>) -> Router {
    let state = HttpState::new(control);
    let admin_auth = AdminAuth::new(admin_bearer_token);

    let admin_routes = Router::new()
        .route("/strategy/btc-5m/readiness", get(btc_realtime_status))
        .route(
            "/strategy/btc-5m/paper-experiment",
            get(btc_paper_experiment_status),
        )
        .route("/backfill/whales", post(legacy_copy_endpoint_gone))
        .route("/copy-trade/backtest", post(legacy_copy_endpoint_gone))
        .route(
            "/copy-trade/replay-existing",
            post(legacy_copy_endpoint_gone),
        )
        .route("/backtests/replay", post(legacy_copy_endpoint_gone))
        .route("/backtests", get(list_backtest_runs))
        .route("/backtests/:backtest_run_id", get(get_backtest_run))
        .route("/copy-trade/calibration", post(legacy_copy_endpoint_gone))
        .route("/backfill/ingesters", get(list_ingesters))
        .route(
            "/backfill/jobs",
            get(list_ingestion_backfills).post(enqueue_ingestion_backfill),
        )
        .route("/backfill/jobs/:job_id", get(get_ingestion_backfill))
        .route(
            "/backfill/jobs/:job_id/events",
            get(list_ingestion_backfill_events),
        )
        .route(
            "/backfill/jobs/:job_id/cancel",
            post(cancel_ingestion_backfill),
        )
        .route(
            "/backfill/readiness/btc-five-minute-training",
            get(ingestion_training_readiness),
        )
        .route("/pnl/stats", get(trade_pnl_summary))
        .route("/trades/pnl/summary", get(trade_pnl_summary))
        .route("/trades/pnl/wallets", get(trade_pnl_wallets))
        .route("/trades/pnl/open", get(trade_pnl_open_positions))
        .route("/trades/pnl/mark-health", get(trade_pnl_mark_health))
        .route("/trades/pnl/recent-exits", get(trade_pnl_recent_exits))
        .route("/trades/pnl/backfill", post(trade_pnl_backfill))
        .route("/trades/pnl/mark-now", post(trade_pnl_mark_now))
        .route("/wallets/mrs/recompute", post(recompute_mrs_scores))
        .route(
            "/wallets/scoring/refresh/enqueue",
            post(enqueue_wallet_score_refresh),
        )
        .route(
            "/wallets/scoring/refresh/process",
            post(process_wallet_score_refresh),
        )
        .route(
            "/wallets/scoring/refresh/jobs",
            get(list_wallet_score_refresh_jobs),
        )
        .route("/wallets/mrs/segments", get(mrs_segment_summary))
        .route(
            "/wallets/mrs/segments/recompute",
            post(recompute_mrs_segment_scores),
        )
        .route(
            "/copy-trade/expectancy-flow/recompute",
            post(recompute_expectancy_flow),
        )
        .route(
            "/copy-trade/expectancy-flow/cells",
            get(expectancy_flow_cells),
        )
        .route(
            "/copy-trade/expectancy-flow/wallet-cells",
            get(expectancy_flow_wallet_cells),
        )
        .route("/gamma/taxonomy/status", get(gamma_taxonomy_status))
        .route("/gamma/taxonomy/backfill", post(gamma_taxonomy_backfill))
        .route("/live/status", get(live_status))
        .route("/live/diagnostics", get(live_identity_diagnostics))
        .route(
            "/live/wallet-diagnostics",
            get(live_wallet_address_diagnostics),
        )
        .route("/live/order-dry-run", post(live_order_dry_run))
        .route(
            "/live/poly1271-funder-probe",
            post(live_poly1271_funder_probe),
        )
        .route("/live/halt", post(live_halt))
        .route("/live/reconcile", post(live_reconcile))
        .route("/live/account-reconcile", post(live_account_reconcile))
        .route("/live/entries/enable", post(live_entries_enable))
        .route("/live/entries/disable", post(live_entries_disable))
        .route(
            "/trading-processes",
            get(list_trading_processes).post(create_trading_process),
        )
        .route(
            "/trading-processes/by-key/:process_key",
            put(upsert_trading_process_by_key),
        )
        .route(
            "/trading-processes/:process_id/status",
            get(get_trading_process_status),
        )
        .route(
            "/trading-processes/:process_id/start-preview",
            get(preview_trading_process_start),
        )
        .route(
            "/trading-processes/:process_id",
            get(get_trading_process).patch(update_trading_process),
        )
        .route(
            "/trading-processes/:process_id/start",
            post(start_trading_process),
        )
        .route(
            "/trading-processes/:process_id/stop",
            post(stop_trading_process),
        )
        .route(
            "/trading-processes/:process_id/reset-simulation",
            post(reset_trading_process_simulation),
        )
        .route_layer(from_fn_with_state(admin_auth, require_admin_bearer));

    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .nest("/admin", admin_routes)
        .with_state(state)
}

async fn health(State(state): State<HttpState>) -> Result<Json<HealthResponse>, HttpError> {
    state.control.health().await.map(Json)
}

async fn metrics(State(state): State<HttpState>) -> Result<Json<MetricsResponse>, HttpError> {
    state.control.metrics().await.map(Json)
}

async fn btc_realtime_status(
    State(state): State<HttpState>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.btc_realtime_status().await.map(Json)
}

async fn btc_paper_experiment_status(
    State(state): State<HttpState>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.btc_paper_experiment_status().await.map(Json)
}

async fn legacy_copy_endpoint_gone() -> Result<Json<serde_json::Value>, HttpError> {
    Err(HttpError::gone(
        "legacy copy-trade and wallet-address workflows are disabled",
    ))
}

async fn get_backtest_run(
    State(state): State<HttpState>,
    Path(backtest_run_id): Path<Uuid>,
) -> Result<Json<BacktestRunResponse>, HttpError> {
    state
        .control
        .get_backtest_run(backtest_run_id)
        .await
        .map(Json)
}

async fn list_backtest_runs(
    State(state): State<HttpState>,
    Query(request): Query<ListBacktestRunsRequest>,
) -> Result<Json<BacktestRunsResponse>, HttpError> {
    state.control.list_backtest_runs(request).await.map(Json)
}

async fn enqueue_ingestion_backfill(
    State(state): State<HttpState>,
    Json(request): Json<IngestionBackfillRequest>,
) -> Result<Json<IngestionBackfillEnqueueResponse>, HttpError> {
    state
        .control
        .enqueue_ingestion_backfill(request)
        .await
        .map(Json)
}

async fn list_ingesters() -> Json<IngesterListResponse> {
    Json(IngesterListResponse {
        ingesters: IngesterKey::ALL
            .into_iter()
            .map(|key| IngesterDescription {
                key,
                request_version: key.supported_request_version(),
                range_alignment_seconds: if key.is_binance() { 86_400 } else { 300 },
            })
            .collect(),
    })
}

async fn list_ingestion_backfills(
    State(state): State<HttpState>,
    Query(request): Query<ListIngestionBackfillsRequest>,
) -> Result<Json<IngestionBackfillJobsResponse>, HttpError> {
    state
        .control
        .list_ingestion_backfills(request)
        .await
        .map(Json)
}

async fn get_ingestion_backfill(
    State(state): State<HttpState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<IngestionBackfillJobResponse>, HttpError> {
    state.control.get_ingestion_backfill(job_id).await.map(Json)
}

async fn list_ingestion_backfill_events(
    State(state): State<HttpState>,
    Path(job_id): Path<Uuid>,
    Query(request): Query<ListIngestionBackfillEventsRequest>,
) -> Result<Json<IngestionBackfillEventsResponse>, HttpError> {
    state
        .control
        .list_ingestion_backfill_events(job_id, request)
        .await
        .map(Json)
}

async fn cancel_ingestion_backfill(
    State(state): State<HttpState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<IngestionBackfillCancelResponse>, HttpError> {
    state
        .control
        .cancel_ingestion_backfill(job_id)
        .await
        .map(Json)
}

async fn ingestion_training_readiness(
    State(state): State<HttpState>,
    Query(request): Query<IngestionReadinessRequest>,
) -> Result<Json<TrainingReadiness>, HttpError> {
    state
        .control
        .ingestion_training_readiness(request)
        .await
        .map(Json)
}

async fn trade_pnl_summary(
    State(state): State<HttpState>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.trade_pnl_summary().await.map(Json)
}

async fn trade_pnl_wallets(
    State(state): State<HttpState>,
    Query(request): Query<TradePnlListRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.trade_pnl_wallets(request).await.map(Json)
}

async fn trade_pnl_open_positions(
    State(state): State<HttpState>,
    Query(request): Query<TradePnlListRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state
        .control
        .trade_pnl_open_positions(request)
        .await
        .map(Json)
}

async fn trade_pnl_mark_health(
    State(state): State<HttpState>,
    Query(request): Query<TradePnlListRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.trade_pnl_mark_health(request).await.map(Json)
}

async fn trade_pnl_recent_exits(
    State(state): State<HttpState>,
    Query(request): Query<TradePnlListRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state
        .control
        .trade_pnl_recent_exits(request)
        .await
        .map(Json)
}

async fn trade_pnl_backfill(
    State(state): State<HttpState>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.trade_pnl_backfill().await.map(Json)
}

async fn trade_pnl_mark_now(
    State(state): State<HttpState>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.trade_pnl_mark_now().await.map(Json)
}

async fn recompute_mrs_scores(
    State(state): State<HttpState>,
    Json(request): Json<MrsRecomputeRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.recompute_mrs_scores(request).await.map(Json)
}

async fn recompute_mrs_segment_scores(
    State(state): State<HttpState>,
    Json(request): Json<MrsRecomputeRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state
        .control
        .recompute_mrs_segment_scores(request)
        .await
        .map(Json)
}

async fn enqueue_wallet_score_refresh(
    State(state): State<HttpState>,
    Json(request): Json<WalletScoreRefreshEnqueueRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state
        .control
        .enqueue_wallet_score_refresh(request)
        .await
        .map(Json)
}

async fn process_wallet_score_refresh(
    State(state): State<HttpState>,
    Json(request): Json<WalletScoreRefreshProcessRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state
        .control
        .process_wallet_score_refresh(request)
        .await
        .map(Json)
}

async fn list_wallet_score_refresh_jobs(
    State(state): State<HttpState>,
    Query(request): Query<WalletScoreRefreshJobsRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state
        .control
        .list_wallet_score_refresh_jobs(request)
        .await
        .map(Json)
}

async fn mrs_segment_summary(
    State(state): State<HttpState>,
    Query(request): Query<TradePnlListRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.mrs_segment_summary(request).await.map(Json)
}

async fn recompute_expectancy_flow(
    State(state): State<HttpState>,
    Json(request): Json<ExpectancyFlowRecomputeRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state
        .control
        .recompute_expectancy_flow(request)
        .await
        .map(Json)
}

async fn expectancy_flow_cells(
    State(state): State<HttpState>,
    Query(request): Query<ExpectancyFlowCellsRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.expectancy_flow_cells(request).await.map(Json)
}

async fn expectancy_flow_wallet_cells(
    State(state): State<HttpState>,
    Query(request): Query<ExpectancyFlowWalletCellsRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state
        .control
        .expectancy_flow_wallet_cells(request)
        .await
        .map(Json)
}

async fn gamma_taxonomy_status(
    State(state): State<HttpState>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.gamma_taxonomy_status().await.map(Json)
}

async fn gamma_taxonomy_backfill(
    State(state): State<HttpState>,
    Json(request): Json<GammaTaxonomyBackfillRequest>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state
        .control
        .gamma_taxonomy_backfill(request)
        .await
        .map(Json)
}

async fn live_status(State(state): State<HttpState>) -> Result<Json<LiveVenueStatus>, HttpError> {
    state.control.live_status().await.map(Json)
}

async fn live_identity_diagnostics(
    State(state): State<HttpState>,
) -> Result<Json<LiveIdentityDiagnostics>, HttpError> {
    state.control.live_identity_diagnostics().await.map(Json)
}

async fn live_wallet_address_diagnostics(
    State(state): State<HttpState>,
    Query(request): Query<LiveWalletDiagnosticsRequest>,
) -> Result<Json<LiveWalletAddressDiagnostics>, HttpError> {
    state
        .control
        .live_wallet_address_diagnostics(request.candidate_addresses())
        .await
        .map(Json)
}

async fn live_order_dry_run(
    State(state): State<HttpState>,
    Json(request): Json<LiveOrderDryRunRequest>,
) -> Result<Json<LiveOrderDryRunDiagnostics>, HttpError> {
    state.control.live_order_dry_run(request).await.map(Json)
}

async fn live_poly1271_funder_probe(
    State(state): State<HttpState>,
    Json(request): Json<LivePoly1271FunderProbeRequest>,
) -> Result<Json<LivePoly1271FunderProbeResponse>, HttpError> {
    state
        .control
        .live_poly1271_funder_probe(request)
        .await
        .map(Json)
}

async fn live_halt(State(state): State<HttpState>) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.live_halt().await.map(Json)
}

async fn live_reconcile(
    State(state): State<HttpState>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.live_reconcile().await.map(Json)
}

async fn live_account_reconcile(
    State(state): State<HttpState>,
    Json(request): Json<AccountReconcileRequest>,
) -> Result<Json<AccountReconcileReport>, HttpError> {
    state
        .control
        .live_account_reconcile(request)
        .await
        .map(Json)
}

async fn live_entries_enable(
    State(state): State<HttpState>,
) -> Result<Json<LiveVenueStatus>, HttpError> {
    state.control.live_set_entries_enabled(true).await.map(Json)
}

async fn live_entries_disable(
    State(state): State<HttpState>,
) -> Result<Json<LiveVenueStatus>, HttpError> {
    state
        .control
        .live_set_entries_enabled(false)
        .await
        .map(Json)
}

async fn create_trading_process(
    State(state): State<HttpState>,
    Json(request): Json<CreateTradingProcessRequest>,
) -> Result<Json<TradingProcessResponse>, HttpError> {
    state
        .control
        .create_trading_process(request)
        .await
        .map(Json)
}

async fn list_trading_processes(
    State(state): State<HttpState>,
    Query(request): Query<ListTradingProcessesRequest>,
) -> Result<Json<TradingProcessesResponse>, HttpError> {
    state
        .control
        .list_trading_processes(request)
        .await
        .map(Json)
}

async fn get_trading_process(
    State(state): State<HttpState>,
    Path(process_id): Path<Uuid>,
) -> Result<Json<TradingProcessResponse>, HttpError> {
    state
        .control
        .get_trading_process(process_id)
        .await
        .map(Json)
}

async fn upsert_trading_process_by_key(
    State(state): State<HttpState>,
    Path(process_key): Path<String>,
    Json(request): Json<UpsertTradingProcessByKeyRequest>,
) -> Result<Json<TradingProcessResponse>, HttpError> {
    state
        .control
        .upsert_trading_process_by_key(process_key, request)
        .await
        .map(Json)
}

async fn get_trading_process_status(
    State(state): State<HttpState>,
    Path(process_id): Path<Uuid>,
) -> Result<Json<TradingProcessStatusResponse>, HttpError> {
    state
        .control
        .get_trading_process_status(process_id)
        .await
        .map(Json)
}

async fn preview_trading_process_start(
    State(state): State<HttpState>,
    Path(process_id): Path<Uuid>,
) -> Result<Json<TradingProcessStartPreviewResponse>, HttpError> {
    state
        .control
        .preview_trading_process_start(process_id)
        .await
        .map(Json)
}

async fn update_trading_process(
    State(state): State<HttpState>,
    Path(process_id): Path<Uuid>,
    Json(request): Json<UpdateTradingProcessRequest>,
) -> Result<Json<TradingProcessResponse>, HttpError> {
    state
        .control
        .update_trading_process(process_id, request)
        .await
        .map(Json)
}

async fn start_trading_process(
    State(state): State<HttpState>,
    Path(process_id): Path<Uuid>,
) -> Result<Json<TradingProcessResponse>, HttpError> {
    state
        .control
        .start_trading_process(process_id)
        .await
        .map(Json)
}

async fn stop_trading_process(
    State(state): State<HttpState>,
    Path(process_id): Path<Uuid>,
) -> Result<Json<TradingProcessResponse>, HttpError> {
    state
        .control
        .stop_trading_process(process_id)
        .await
        .map(Json)
}

async fn reset_trading_process_simulation(
    State(state): State<HttpState>,
    Path(process_id): Path<Uuid>,
) -> Result<Json<TradingProcessResetResponse>, HttpError> {
    state
        .control
        .reset_trading_process_simulation(process_id)
        .await
        .map(Json)
}

async fn require_admin_bearer(
    State(auth): State<AdminAuth>,
    headers: HeaderMap,
    request: Request<Body>,
    next: Next,
) -> Result<Response, HttpError> {
    let expected_token = auth.bearer_token.trim();
    let authorized = if expected_token.is_empty() {
        false
    } else {
        let expected = format!("Bearer {expected_token}");
        headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .map(|value| value == expected)
            .unwrap_or(false)
    };

    if authorized {
        Ok(next.run(request).await)
    } else {
        Err(HttpError::unauthorized(
            "missing or invalid admin bearer token",
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub service: String,
    pub status: HealthStatus,
    pub checked_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Ok,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsResponse {
    pub service: String,
    pub captured_at: DateTime<Utc>,
    #[serde(default)]
    pub counters: serde_json::Value,
    #[serde(default)]
    pub gauges: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillWhalesRequest {
    #[serde(default)]
    pub process_id: Option<Uuid>,
    pub lookback_days: Option<i32>,
    pub min_trade_usd: Option<Decimal>,
    pub page_limit: Option<usize>,
    pub max_pages: Option<usize>,
    #[serde(default)]
    pub mode: Option<BackfillMode>,
    #[serde(default)]
    pub wallet_addresses: Vec<String>,
    #[serde(default)]
    pub market_ids: Vec<String>,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub request: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeBacktestRequest {
    #[serde(default)]
    pub process_id: Option<Uuid>,
    pub lookback_days: Option<i32>,
    pub min_trade_usd: Option<Decimal>,
    pub min_wallet_score: Option<Decimal>,
    pub copy_size_fraction: Option<Decimal>,
    pub max_copy_size_usd: Option<Decimal>,
    pub page_limit: Option<usize>,
    pub max_pages: Option<usize>,
    #[serde(default)]
    pub execute_signals: bool,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeCalibrationRequest {
    #[serde(default)]
    pub process_id: Option<Uuid>,
    pub lookback_days: Option<i32>,
    pub min_trade_usd: Option<Decimal>,
    pub page_limit: Option<usize>,
    pub max_pages: Option<usize>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeReplayRequest {
    #[serde(default)]
    pub process_id: Option<Uuid>,
    pub lookback_days: Option<i64>,
    pub limit: Option<usize>,
    #[serde(default)]
    pub execute_signals: bool,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradePnlListRequest {
    #[serde(default)]
    pub process_id: Option<Uuid>,
    pub limit: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MrsRecomputeRequest {
    pub lookback_days: Option<i64>,
    pub limit: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletScoreRefreshEnqueueRequest {
    #[serde(default)]
    pub wallets: Vec<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub score_version: Option<String>,
    #[serde(default)]
    pub segment_score_version: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletScoreRefreshProcessRequest {
    #[serde(default)]
    pub wallets: Vec<String>,
    #[serde(default)]
    pub use_queue: Option<bool>,
    pub lookback_days: Option<i64>,
    pub page_limit: Option<usize>,
    pub max_pages: Option<usize>,
    pub limit: Option<i64>,
    #[serde(default)]
    pub refresh_percentiles: bool,
    #[serde(default)]
    pub score_version: Option<String>,
    #[serde(default)]
    pub segment_score_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletScoreRefreshJobsRequest {
    #[serde(default)]
    pub status: Option<String>,
    pub limit: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectancyFlowRecomputeRequest {
    #[serde(default)]
    pub process_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectancyFlowCellsRequest {
    #[serde(default)]
    pub process_id: Option<Uuid>,
    pub limit: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectancyFlowWalletCellsRequest {
    #[serde(default)]
    pub process_id: Option<Uuid>,
    pub proxy_wallet: String,
    pub limit: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammaTaxonomyBackfillRequest {
    pub limit: Option<i64>,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub fallback_keywords: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LiveWalletDiagnosticsRequest {
    #[serde(default)]
    pub address: Option<String>,
    #[serde(default)]
    pub candidate_address: Option<String>,
}

impl LiveWalletDiagnosticsRequest {
    fn candidate_addresses(&self) -> Vec<String> {
        self.address
            .iter()
            .chain(self.candidate_address.iter())
            .flat_map(|value| value.split(','))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTradingProcessRequest {
    pub name: String,
    #[serde(default = "default_process_type")]
    pub process_type: String,
    #[serde(default = "default_process_scope")]
    pub process_scope: String,
    #[serde(default)]
    pub process_key: Option<String>,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub config: TradingProcessConfig,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateTradingProcessRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub process_type: Option<String>,
    #[serde(default)]
    pub process_scope: Option<String>,
    #[serde(default)]
    pub process_key: Option<Option<String>>,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub config: Option<TradingProcessConfig>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListTradingProcessesRequest {
    pub limit: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListBacktestRunsRequest {
    pub limit: Option<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListIngestionBackfillsRequest {
    pub limit: Option<i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListIngestionBackfillEventsRequest {
    pub limit: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionReadinessRequest {
    pub range_start: DateTime<Utc>,
    pub range_end: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertTradingProcessByKeyRequest {
    pub name: String,
    #[serde(default = "default_process_type")]
    pub process_type: String,
    #[serde(default = "default_process_scope")]
    pub process_scope: String,
    pub enabled: bool,
    pub status: String,
    #[serde(default)]
    pub config: TradingProcessConfig,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillJobResponse {
    pub job: BackfillJob,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillJobsResponse {
    pub jobs: Vec<BackfillJob>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelBackfillJobResponse {
    pub job_id: Uuid,
    pub status: BackfillJobStatus,
    pub cancel_requested: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionBackfillEnqueueResponse {
    pub job_id: Uuid,
    pub ingester: String,
    pub status: IngestionBackfillJobStatus,
    pub requested_at: DateTime<Utc>,
}

impl From<IngestionBackfillJob> for IngestionBackfillEnqueueResponse {
    fn from(job: IngestionBackfillJob) -> Self {
        Self {
            job_id: job.job_id,
            ingester: job.ingester_key,
            status: job.status,
            requested_at: job.requested_at,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionBackfillJobResponse {
    pub job: IngestionBackfillJob,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionBackfillJobsResponse {
    pub jobs: Vec<IngestionBackfillJob>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionBackfillEventsResponse {
    pub job_id: Uuid,
    pub events: Vec<IngestionBackfillJobEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionBackfillCancelResponse {
    pub job_id: Uuid,
    pub status: IngestionBackfillJobStatus,
    pub cancel_requested: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngesterDescription {
    pub key: IngesterKey,
    pub request_version: i32,
    pub range_alignment_seconds: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngesterListResponse {
    pub ingesters: Vec<IngesterDescription>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingProcessResponse {
    pub process: TradingProcess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingProcessesResponse {
    pub processes: Vec<TradingProcess>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingProcessStatusResponse {
    pub process_id: Uuid,
    pub status: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingProcessStartPreviewResponse {
    pub process_id: Uuid,
    pub experiment_id: Uuid,
    pub experiment_key: String,
    pub preregistration_sha256: String,
    pub config_hash: String,
    pub frozen_process_config: TradingProcessConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingProcessResetResponse {
    pub report: TradingProcessResetReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestRunResponse {
    pub backtest_run: BacktestRun,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestRunsResponse {
    pub backtest_runs: Vec<BacktestRun>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
}

fn default_process_type() -> String {
    "copy_trade".to_string()
}

fn default_process_scope() -> String {
    "default".to_string()
}

#[derive(Debug, Clone)]
pub struct HttpError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl HttpError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "bad_request", message)
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "unauthorized", message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found", message)
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, "conflict", message)
    }

    pub fn gone(message: impl Into<String>) -> Self {
        Self::new(StatusCode::GONE, "gone", message)
    }

    pub fn not_implemented(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_IMPLEMENTED, "not_implemented", message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", message)
    }

    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }
}

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        let body = ErrorResponse {
            error: ErrorBody {
                code: self.code.to_string(),
                message: self.message,
            },
        };
        (self.status, Json(body)).into_response()
    }
}
