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
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use crate::account_reconcile::{AccountReconcileReport, AccountReconcileRequest};

use crate::{
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
    models::{TradingProcess, TradingProcessConfig},
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

    async fn complete_trading_process(
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
        .route("/trading-processes", get(list_trading_processes))
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
            "/trading-processes/:process_id/complete",
            post(complete_trading_process),
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
                range_alignment_seconds: key.alignment_seconds(),
                accepts_new_requests: key.accepts_new_requests(),
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

async fn complete_trading_process(
    State(state): State<HttpState>,
    Path(process_id): Path<Uuid>,
) -> Result<Json<TradingProcessResponse>, HttpError> {
    state
        .control
        .complete_trading_process(process_id)
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
#[serde(deny_unknown_fields)]
pub struct UpdateTradingProcessRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub config: Option<TradingProcessConfig>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListTradingProcessesRequest {
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
    pub process_type: String,
    pub process_scope: String,
    pub enabled: bool,
    pub status: String,
    #[serde(default)]
    pub config: TradingProcessConfig,
    #[serde(default)]
    pub metadata: serde_json::Value,
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
    pub accepts_new_requests: bool,
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
    pub run_id: Uuid,
    pub run_key: String,
    pub preregistration_sha256: String,
    pub config_hash: String,
    pub frozen_process_config: TradingProcessConfig,
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
