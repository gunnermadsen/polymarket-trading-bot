use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{header::AUTHORIZATION, HeaderMap, Request, StatusCode},
    middleware::{from_fn_with_state, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    backfill::BackfillMode,
    execution::LiveVenueStatus,
    models::{BackfillJob, BackfillJobStatus},
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
    async fn metrics(&self) -> Result<MetricsResponse, HttpError>;

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

    async fn list_backfill_jobs(&self) -> Result<BackfillJobsResponse, HttpError>;

    async fn get_backfill_job(&self, job_id: Uuid) -> Result<BackfillJobResponse, HttpError>;

    async fn cancel_backfill_job(
        &self,
        job_id: Uuid,
    ) -> Result<CancelBackfillJobResponse, HttpError>;

    async fn trade_pnl_summary(&self) -> Result<serde_json::Value, HttpError>;

    async fn trade_pnl_wallets(
        &self,
        request: TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError>;

    async fn trade_pnl_open_positions(
        &self,
        request: TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError>;

    async fn trade_pnl_recent_exits(
        &self,
        request: TradePnlListRequest,
    ) -> Result<serde_json::Value, HttpError>;

    async fn trade_pnl_backfill(&self) -> Result<serde_json::Value, HttpError>;

    async fn trade_pnl_mark_now(&self) -> Result<serde_json::Value, HttpError>;

    async fn live_status(&self) -> Result<LiveVenueStatus, HttpError>;

    async fn live_halt(&self) -> Result<serde_json::Value, HttpError>;

    async fn live_reconcile(&self) -> Result<serde_json::Value, HttpError>;

    async fn live_set_entries_enabled(&self, enabled: bool) -> Result<LiveVenueStatus, HttpError>;
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

    async fn live_status(&self) -> Result<LiveVenueStatus, HttpError> {
        Err(HttpError::not_implemented("live status is not wired"))
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
}

pub fn router(control: SharedControlApi, admin_bearer_token: impl Into<String>) -> Router {
    let state = HttpState::new(control);
    let admin_auth = AdminAuth::new(admin_bearer_token);

    let admin_routes = Router::new()
        .route("/backfill/whales", post(start_whales_backfill))
        .route("/copy-trade/backtest", post(start_copy_trade_backtest))
        .route(
            "/copy-trade/calibration",
            post(start_copy_trade_calibration),
        )
        .route("/backfill/jobs", get(list_backfill_jobs))
        .route("/backfill/jobs/:job_id", get(get_backfill_job))
        .route("/backfill/jobs/:job_id/cancel", post(cancel_backfill_job))
        .route("/pnl/stats", get(trade_pnl_summary))
        .route("/trades/pnl/summary", get(trade_pnl_summary))
        .route("/trades/pnl/wallets", get(trade_pnl_wallets))
        .route("/trades/pnl/open", get(trade_pnl_open_positions))
        .route("/trades/pnl/recent-exits", get(trade_pnl_recent_exits))
        .route("/trades/pnl/backfill", post(trade_pnl_backfill))
        .route("/trades/pnl/mark-now", post(trade_pnl_mark_now))
        .route("/live/status", get(live_status))
        .route("/live/halt", post(live_halt))
        .route("/live/reconcile", post(live_reconcile))
        .route("/live/entries/enable", post(live_entries_enable))
        .route("/live/entries/disable", post(live_entries_disable))
        .route_layer(from_fn_with_state(admin_auth, require_admin_bearer));

    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .nest("/admin", admin_routes)
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        service: SERVICE_NAME.to_string(),
        status: HealthStatus::Ok,
        checked_at: Utc::now(),
    })
}

async fn metrics(State(state): State<HttpState>) -> Result<Json<MetricsResponse>, HttpError> {
    state.control.metrics().await.map(Json)
}

async fn start_whales_backfill(
    State(state): State<HttpState>,
    Json(request): Json<BackfillWhalesRequest>,
) -> Result<Json<BackfillJobResponse>, HttpError> {
    state.control.start_whales_backfill(request).await.map(Json)
}

async fn start_copy_trade_backtest(
    State(state): State<HttpState>,
    Json(request): Json<CopyTradeBacktestRequest>,
) -> Result<Json<BackfillJobResponse>, HttpError> {
    state
        .control
        .start_copy_trade_backtest(request)
        .await
        .map(Json)
}

async fn start_copy_trade_calibration(
    State(state): State<HttpState>,
    Json(request): Json<CopyTradeCalibrationRequest>,
) -> Result<Json<BackfillJobResponse>, HttpError> {
    state
        .control
        .start_copy_trade_calibration(request)
        .await
        .map(Json)
}

async fn list_backfill_jobs(
    State(state): State<HttpState>,
) -> Result<Json<BackfillJobsResponse>, HttpError> {
    state.control.list_backfill_jobs().await.map(Json)
}

async fn get_backfill_job(
    State(state): State<HttpState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<BackfillJobResponse>, HttpError> {
    state.control.get_backfill_job(job_id).await.map(Json)
}

async fn cancel_backfill_job(
    State(state): State<HttpState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<CancelBackfillJobResponse>, HttpError> {
    state.control.cancel_backfill_job(job_id).await.map(Json)
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

async fn live_status(State(state): State<HttpState>) -> Result<Json<LiveVenueStatus>, HttpError> {
    state.control.live_status().await.map(Json)
}

async fn live_halt(State(state): State<HttpState>) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.live_halt().await.map(Json)
}

async fn live_reconcile(
    State(state): State<HttpState>,
) -> Result<Json<serde_json::Value>, HttpError> {
    state.control.live_reconcile().await.map(Json)
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
    pub lookback_days: Option<i32>,
    pub min_trade_usd: Option<Decimal>,
    pub page_limit: Option<usize>,
    pub max_pages: Option<usize>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradePnlListRequest {
    pub limit: Option<i64>,
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
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
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
