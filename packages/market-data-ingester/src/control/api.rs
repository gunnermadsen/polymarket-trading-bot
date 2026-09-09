use std::{
    collections::{BTreeMap, BTreeSet},
    net::SocketAddr,
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE},
        HeaderMap, Request, StatusCode,
    },
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::{Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use subtle::ConstantTimeEq;
use tokio::{net::TcpListener, sync::watch};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    coverage::{self, CoverageTarget},
    domain::{
        BackfillFailureKind, BackfillOutcome, BackfillRequest, DesiredState, DrainRequest,
        IngesterProfile, IngesterStrategyKey,
    },
    persistence::{
        BackfillJobEvent, BackfillJobRecord, BackfillRepository, ClaimedBackfillJob,
        DrainJobRecord, DrainRepository, ProfileRepository, WorkerAllocationSummary, WorkerRecord,
        WorkerRegistration,
    },
    runtime::StrategyRegistry,
    strategies::{dataset_for_strategy, STRATEGY_DATASETS},
};

use super::error::ApiError;
use super::metrics;

const MAX_REQUEST_BODY_BYTES: usize = 32 * 1024;
const MAX_CONFIG_BYTES: usize = 16 * 1024;

#[derive(Clone)]
pub struct ControlApi {
    state: ApiState,
}

impl ControlApi {
    pub fn new(
        profiles: ProfileRepository,
        registry: StrategyRegistry,
        admin_token: String,
        readiness: ControlReadiness,
    ) -> Result<Self> {
        if admin_token.len() < 32 {
            anyhow::bail!("market-data ingester administrative token must be at least 32 bytes");
        }
        Ok(Self {
            state: ApiState {
                backfills: BackfillRepository::new(profiles.pool().clone()),
                drains: DrainRepository::new(profiles.pool().clone()),
                profiles,
                registry,
                admin_token: Arc::<str>::from(admin_token),
                readiness,
            },
        })
    }

    pub fn router(&self) -> Router {
        let admin = Router::new()
            .route("/strategy/all", get(list_strategies))
            .route("/strategy/:strategy_key", get(get_strategy))
            .route("/backfills", get(list_backfills).post(submit_backfill))
            .route("/backfills/:job_id", get(get_backfill))
            .route("/backfills/:job_id/cancel", post(cancel_backfill))
            .route("/backfills/:job_id/retry", post(retry_backfill))
            .route("/backfills/:job_id/events", get(get_backfill_events))
            .route("/drains", get(list_drains).post(submit_drain))
            .route("/drains/:job_id", get(get_drain))
            .route("/drains/:job_id/cancel", post(cancel_drain))
            .route("/drains/:job_id/retry", post(retry_drain))
            .route("/datasets/coverage", get(dataset_coverage))
            .route("/workers", get(list_workers))
            .route("/stream/routes", post(resolve_stream_routes))
            .route("/internal/workers/register", post(register_worker))
            .route(
                "/internal/workers/:worker_id/heartbeat",
                post(heartbeat_worker),
            )
            .route(
                "/internal/workers/:worker_id/assignments",
                post(claim_assignment),
            )
            .route(
                "/internal/workers/:worker_id/jobs/:job_id/heartbeat",
                post(heartbeat_job),
            )
            .route(
                "/internal/workers/:worker_id/jobs/:job_id/complete",
                post(complete_job),
            )
            .route(
                "/internal/workers/:worker_id/jobs/:job_id/fail",
                post(fail_job),
            )
            .route("/ingesters", get(list_profiles))
            .route("/ingesters/:strategy_key", get(get_profile))
            .route(
                "/ingesters/:strategy_key/config",
                get(get_profile_config).put(replace_profile_config),
            )
            .route("/ingesters/:strategy_key/start", post(start_profile))
            .route("/ingesters/:strategy_key/stop", post(stop_profile))
            .route("/ingesters/:strategy_key/restart", post(restart_profile))
            .route_layer(middleware::from_fn_with_state(
                self.state.clone(),
                require_admin,
            ));

        Router::new()
            .route("/health/live", get(liveness))
            .route("/health/ready", get(readiness))
            .route("/prometheus/metrics", get(prometheus_metrics))
            .merge(admin)
            .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
            .with_state(self.state.clone())
    }

    pub async fn serve(self, bind: SocketAddr, shutdown: CancellationToken) -> Result<()> {
        let listener = TcpListener::bind(bind)
            .await
            .with_context(|| format!("failed to bind market-data ingester API to {bind}"))?;
        axum::serve(listener, self.router())
            .with_graceful_shutdown(shutdown.cancelled_owned())
            .await
            .context("market-data ingester API failed")
    }
}

async fn dataset_coverage(
    State(state): State<ApiState>,
) -> Result<Json<coverage::CoverageReport>, ApiError> {
    let targets = coverage_targets(&state.registry)?;
    coverage::detect(state.profiles.pool(), targets)
        .await
        .map(Json)
        .map_err(ApiError::internal)
}

fn coverage_targets(registry: &StrategyRegistry) -> Result<Vec<CoverageTarget>, ApiError> {
    let drains_by_relation = registry
        .drains()
        .map(|strategy| {
            let descriptor = strategy.descriptor();
            (
                descriptor.relation.to_string(),
                descriptor.strategy_key.to_string(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut targets = BTreeMap::<String, CoverageTarget>::new();
    for strategy in registry.backfills() {
        let strategy_key = strategy.descriptor().strategy_key.to_string();
        let dataset = dataset_for_strategy(&strategy_key);
        let contract = dataset.and_then(crate::domain::contract_for);
        let product_key = dataset.map_or_else(|| strategy_key.clone(), |key| key.to_string());
        let relation = contract.map(|contract| contract.canonical_table.to_owned());
        let drain_strategy_key = relation
            .as_ref()
            .and_then(|relation| drains_by_relation.get(relation))
            .cloned();
        let gap_strategy_keys = dataset.map_or_else(
            || vec![strategy_key.clone()],
            |dataset| {
                STRATEGY_DATASETS
                    .iter()
                    .filter(|binding| binding.dataset == dataset)
                    .map(|binding| binding.strategy_key.to_owned())
                    .collect()
            },
        );
        let target = targets
            .entry(product_key.clone())
            .or_insert_with(|| CoverageTarget {
                product_key,
                relation: relation.clone(),
                backfill_strategy_keys: Vec::new(),
                drain_strategy_key: drain_strategy_key.clone(),
                gap_strategy_keys,
            });
        if target.relation != relation || target.drain_strategy_key != drain_strategy_key {
            return Err(ApiError::internal(
                "backfill strategies for one dataset declare incompatible coverage storage",
            ));
        }
        target.backfill_strategy_keys.push(strategy_key);
        if let Some(drain_strategy_key) = &target.drain_strategy_key {
            if !target.gap_strategy_keys.contains(drain_strategy_key) {
                target.gap_strategy_keys.push(drain_strategy_key.clone());
            }
        }
    }
    Ok(targets.into_values().collect())
}

#[derive(Clone)]
struct ApiState {
    backfills: BackfillRepository,
    drains: DrainRepository,
    profiles: ProfileRepository,
    registry: StrategyRegistry,
    admin_token: Arc<str>,
    readiness: ControlReadiness,
}

async fn submit_drain(
    State(state): State<ApiState>,
    Json(request): Json<DrainRequest>,
) -> Result<(StatusCode, Json<DrainJobRecord>), ApiError> {
    let strategy = state.registry.drain(&request.strategy_key).ok_or_else(|| {
        ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_drain_strategy",
            "strategy is not registered for drain",
        )
    })?;
    strategy
        .validate_request(&request)
        .map_err(|e| ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, e.code, e.message))?;
    let job = state
        .drains
        .submit(&request, strategy.descriptor().contract_version)
        .await
        .map_err(ApiError::internal)?;
    Ok((StatusCode::ACCEPTED, Json(job)))
}

#[derive(Debug, Deserialize)]
struct DrainListQuery {
    limit: Option<i64>,
}
async fn list_drains(
    State(state): State<ApiState>,
    Query(query): Query<DrainListQuery>,
) -> Result<Json<Vec<DrainJobRecord>>, ApiError> {
    state
        .drains
        .list(query.limit.unwrap_or(100))
        .await
        .map(Json)
        .map_err(ApiError::internal)
}
async fn get_drain(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<DrainJobRecord>, ApiError> {
    state
        .drains
        .get(job_id)
        .await
        .map_err(ApiError::internal)?
        .map(Json)
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::NOT_FOUND,
                "drain_not_found",
                "drain job was not found",
            )
        })
}
async fn cancel_drain(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<DrainJobRecord>, ApiError> {
    state
        .drains
        .cancel(job_id)
        .await
        .map_err(ApiError::internal)?
        .map(Json)
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::CONFLICT,
                "drain_not_cancellable",
                "drain job is not cancellable",
            )
        })
}
async fn retry_drain(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<DrainJobRecord>, ApiError> {
    state
        .drains
        .retry(job_id)
        .await
        .map_err(ApiError::internal)?
        .map(Json)
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::CONFLICT,
                "drain_not_retryable",
                "drain job is not retryable",
            )
        })
}

async fn list_strategies(State(state): State<ApiState>) -> Json<Vec<String>> {
    let mut keys = state
        .registry
        .keys()
        .map(|key| key.as_str().to_owned())
        .collect::<BTreeSet<_>>();
    keys.extend(
        state
            .registry
            .backfills()
            .map(|strategy| strategy.descriptor().strategy_key.to_string()),
    );
    keys.extend(
        state
            .registry
            .drains()
            .map(|strategy| strategy.descriptor().strategy_key.to_string()),
    );
    Json(keys.into_iter().collect())
}

async fn get_strategy(
    State(state): State<ApiState>,
    Path(strategy_key): Path<String>,
) -> Result<Json<Value>, ApiError> {
    if let Some(strategy) = state.registry.drain(&strategy_key) {
        return serde_json::to_value(strategy.descriptor())
            .map(Json)
            .map_err(ApiError::internal);
    }
    if let Some(strategy) = state.registry.backfill(&strategy_key) {
        return serde_json::to_value(strategy.descriptor())
            .map(Json)
            .map_err(ApiError::internal);
    }
    if let Ok(key) = IngesterStrategyKey::from_str(&strategy_key) {
        if state.registry.factory(key).is_some() {
            return Ok(Json(serde_json::json!({
                "strategy_key": key.as_str(),
                "name": key.as_str(),
                "description": "realtime collection strategy",
                "capabilities": ["realtime"],
                "strategy_contract_version": 1,
                "request_schema_version": null,
                "shardable": false,
                "maximum_shards": 0
            })));
        }
    }
    Err(ApiError::new(
        StatusCode::NOT_FOUND,
        "strategy_not_found",
        "strategy is not registered",
    ))
}

async fn submit_backfill(
    State(state): State<ApiState>,
    Json(request): Json<BackfillRequest>,
) -> Result<(StatusCode, Json<BackfillJobRecord>), ApiError> {
    let strategy = state
        .registry
        .backfill(&request.strategy_key)
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unsupported_strategy",
                "strategy is not registered for backfills",
            )
        })?;
    let validated = strategy.validate_request(&request).map_err(|error| {
        ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, error.code, error.message)
    })?;
    let shards = strategy.plan_shards(&validated).map_err(|error| {
        ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, error.code, error.message)
    })?;
    let job = state
        .backfills
        .submit(&validated, &shards)
        .await
        .map_err(ApiError::internal)?;
    Ok((StatusCode::ACCEPTED, Json(job)))
}

#[derive(Debug, Deserialize)]
struct BackfillListQuery {
    limit: Option<i64>,
}

async fn list_backfills(
    State(state): State<ApiState>,
    Query(query): Query<BackfillListQuery>,
) -> Result<Json<Vec<BackfillJobRecord>>, ApiError> {
    state
        .backfills
        .list(query.limit.unwrap_or(100))
        .await
        .map(Json)
        .map_err(ApiError::internal)
}

#[derive(Serialize)]
struct BackfillDetail {
    job: BackfillJobRecord,
    shards: Vec<BackfillJobRecord>,
}

async fn get_backfill(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<BackfillDetail>, ApiError> {
    let job = state
        .backfills
        .get(job_id)
        .await
        .map_err(ApiError::internal)?
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::NOT_FOUND,
                "backfill_not_found",
                "backfill job was not found",
            )
        })?;
    let parent = job.parent_job_id.unwrap_or(job.job_id);
    let shards = state
        .backfills
        .children(parent)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(BackfillDetail { job, shards }))
}

async fn cancel_backfill(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<BackfillJobRecord>, ApiError> {
    state
        .backfills
        .cancel(job_id)
        .await
        .map_err(ApiError::internal)?
        .map(Json)
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::NOT_FOUND,
                "backfill_not_found",
                "backfill job was not found",
            )
        })
}

async fn retry_backfill(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
) -> Result<Json<BackfillJobRecord>, ApiError> {
    state
        .backfills
        .retry(job_id)
        .await
        .map_err(ApiError::internal)?
        .map(Json)
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::NOT_FOUND,
                "backfill_not_found",
                "backfill job was not found",
            )
        })
}

async fn get_backfill_events(
    State(state): State<ApiState>,
    Path(job_id): Path<Uuid>,
    Query(query): Query<BackfillListQuery>,
) -> Result<Json<Vec<BackfillJobEvent>>, ApiError> {
    if state
        .backfills
        .get(job_id)
        .await
        .map_err(ApiError::internal)?
        .is_none()
    {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "backfill_not_found",
            "backfill job was not found",
        ));
    }
    state
        .backfills
        .events(job_id, query.limit.unwrap_or(100))
        .await
        .map(Json)
        .map_err(ApiError::internal)
}

async fn list_workers(
    State(state): State<ApiState>,
) -> Result<Json<Vec<WorkerAllocationSummary>>, ApiError> {
    state
        .backfills
        .list_worker_allocation_summaries()
        .await
        .map(Json)
        .map_err(ApiError::internal)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamRouteRequest {
    products: Vec<StreamProductRequest>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StreamProductRequest {
    key: String,
    contract_version: u32,
}

#[derive(Debug, Serialize)]
struct StreamRouteResponse {
    routes: Vec<WorkerStreamRoute>,
    unresolved: Vec<StreamProductRejection>,
}

#[derive(Debug, Serialize)]
struct WorkerStreamRoute {
    worker_id: String,
    endpoint: String,
    source_revision: String,
    products: Vec<StreamProductRequest>,
}

#[derive(Debug, Serialize)]
struct StreamProductRejection {
    product: StreamProductRequest,
    reason: &'static str,
}

async fn resolve_stream_routes(
    State(state): State<ApiState>,
    Json(request): Json<StreamRouteRequest>,
) -> Result<Json<StreamRouteResponse>, ApiError> {
    if request.products.is_empty() || request.products.len() > IngesterStrategyKey::ALL.len() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_stream_products",
            "stream route request must contain at least one registered product",
        ));
    }
    let profiles = state.profiles.list().await.map_err(ApiError::internal)?;
    let workers = state
        .backfills
        .list_workers()
        .await
        .map_err(ApiError::internal)?;
    let now = Utc::now();
    let mut grouped = std::collections::BTreeMap::<String, WorkerStreamRoute>::new();
    let mut unresolved = Vec::new();
    for product in request.products {
        // One Polymarket RTDS strategy owns both products carried by its single
        // upstream socket. Product selectors are intentionally not constrained
        // to a one-product-per-strategy topology.
        let key = if product.key == "polymarket_rtds_chainlink_reference_price" {
            IngesterStrategyKey::PolymarketChainlinkBtcusdTwap
        } else if let Ok(key) = IngesterStrategyKey::from_str(&product.key) {
            key
        } else {
            unresolved.push(StreamProductRejection {
                product,
                reason: "unknown_product",
            });
            continue;
        };
        if product.contract_version != crate::streaming::CONTRACT_VERSION {
            unresolved.push(StreamProductRejection {
                product,
                reason: "contract_version_mismatch",
            });
            continue;
        }
        let Some(profile) = profiles.iter().find(|profile| profile.strategy_key == key) else {
            unresolved.push(StreamProductRejection {
                product,
                reason: "profile_not_found",
            });
            continue;
        };
        let Some(owner) = profile
            .lease_owner
            .as_deref()
            .filter(|_| profile.lease_is_current(now))
        else {
            unresolved.push(StreamProductRejection {
                product,
                reason: "no_current_owner",
            });
            continue;
        };
        let Some(worker) = workers.iter().find(|worker| {
            worker.worker_id == owner
                && worker.lifecycle_state == "active"
                && worker.heartbeat_at >= now - ChronoDuration::seconds(30)
                && worker
                    .realtime_strategies
                    .as_array()
                    .is_some_and(|strategies| {
                        strategies
                            .iter()
                            .any(|strategy| strategy.as_str() == Some(key.as_str()))
                    })
        }) else {
            unresolved.push(StreamProductRejection {
                product,
                reason: "owner_unavailable",
            });
            continue;
        };
        grouped
            .entry(worker.worker_id.clone())
            .or_insert_with(|| WorkerStreamRoute {
                worker_id: worker.worker_id.clone(),
                endpoint: format!("http://{}:50051", worker.hostname),
                source_revision: worker.source_revision.clone(),
                products: Vec::new(),
            })
            .products
            .push(product);
    }
    Ok(Json(StreamRouteResponse {
        routes: grouped.into_values().collect(),
        unresolved,
    }))
}

async fn register_worker(
    State(state): State<ApiState>,
    Json(worker): Json<WorkerRegistration>,
) -> Result<Json<WorkerRecord>, ApiError> {
    state
        .backfills
        .register_worker(&worker)
        .await
        .map(Json)
        .map_err(ApiError::internal)
}

async fn heartbeat_worker(
    State(state): State<ApiState>,
    Path(worker_id): Path<String>,
) -> Result<StatusCode, ApiError> {
    if state
        .backfills
        .heartbeat_worker(&worker_id)
        .await
        .map_err(ApiError::internal)?
    {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "worker_not_found",
            "worker is not registered",
        ))
    }
}

async fn claim_assignment(
    State(state): State<ApiState>,
    Path(worker_id): Path<String>,
) -> Result<Json<Option<ClaimedBackfillJob>>, ApiError> {
    state
        .backfills
        .claim(&worker_id, Duration::from_secs(60))
        .await
        .map(Json)
        .map_err(ApiError::internal)
}

#[derive(Deserialize)]
struct JobHeartbeatRequest {
    lease_token: Uuid,
    #[serde(default = "empty_json_object")]
    progress: Value,
    #[serde(default = "empty_json_object")]
    checkpoint: Value,
}

fn empty_json_object() -> Value {
    Value::Object(Default::default())
}

async fn heartbeat_job(
    State(state): State<ApiState>,
    Path((worker_id, job_id)): Path<(String, Uuid)>,
    Json(request): Json<JobHeartbeatRequest>,
) -> Result<StatusCode, ApiError> {
    if state
        .backfills
        .heartbeat_job(
            &worker_id,
            job_id,
            request.lease_token,
            Duration::from_secs(60),
            &request.progress,
            &request.checkpoint,
        )
        .await
        .map_err(ApiError::internal)?
    {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::new(
            StatusCode::CONFLICT,
            "lease_lost",
            "backfill job lease was lost",
        ))
    }
}

#[derive(Deserialize)]
struct CompleteJobRequest {
    lease_token: Uuid,
    outcome: BackfillOutcome,
}

async fn complete_job(
    State(state): State<ApiState>,
    Path((worker_id, job_id)): Path<(String, Uuid)>,
    Json(request): Json<CompleteJobRequest>,
) -> Result<StatusCode, ApiError> {
    if state
        .backfills
        .complete(&worker_id, job_id, request.lease_token, &request.outcome)
        .await
        .map_err(ApiError::internal)?
    {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::new(
            StatusCode::CONFLICT,
            "lease_lost",
            "backfill job lease was lost",
        ))
    }
}

#[derive(Deserialize)]
struct FailJobRequest {
    lease_token: Uuid,
    kind: BackfillFailureKind,
    code: String,
    message: String,
}

async fn fail_job(
    State(state): State<ApiState>,
    Path((worker_id, job_id)): Path<(String, Uuid)>,
    Json(request): Json<FailJobRequest>,
) -> Result<StatusCode, ApiError> {
    if state
        .backfills
        .fail(
            &worker_id,
            job_id,
            request.lease_token,
            request.kind,
            &request.code,
            &request.message,
        )
        .await
        .map_err(ApiError::internal)?
    {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::new(
            StatusCode::CONFLICT,
            "lease_lost",
            "backfill job lease was lost",
        ))
    }
}

#[derive(Clone)]
pub struct ControlReadiness {
    receiver: watch::Receiver<bool>,
}

impl ControlReadiness {
    pub fn channel(initial: bool) -> (watch::Sender<bool>, Self) {
        let (sender, receiver) = watch::channel(initial);
        (sender, Self { receiver })
    }

    fn is_ready(&self) -> bool {
        *self.receiver.borrow()
    }
}

async fn require_admin(
    State(state): State<ApiState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if authorized(request.headers(), &state.admin_token) {
        next.run(request).await
    } else {
        ApiError::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "valid bearer authentication is required",
        )
        .into_response()
    }
}

fn authorized(headers: &HeaderMap, expected: &str) -> bool {
    let Some(value) = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let Some(candidate) = value.strip_prefix("Bearer ") else {
        return false;
    };
    candidate.len() == expected.len() && bool::from(candidate.as_bytes().ct_eq(expected.as_bytes()))
}

async fn liveness() -> StatusCode {
    StatusCode::OK
}

async fn readiness(State(state): State<ApiState>) -> StatusCode {
    if state.readiness.is_ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn prometheus_metrics(State(state): State<ApiState>) -> Result<Response, ApiError> {
    let profiles = state.profiles.list().await.map_err(ApiError::internal)?;
    let allocations = state
        .backfills
        .list_worker_allocations()
        .await
        .map_err(ApiError::internal)?;
    let body = metrics::render(&profiles, &allocations, state.readiness.is_ready())
        .map_err(ApiError::internal)?;
    Ok((
        [(
            CONTENT_TYPE,
            "application/openmetrics-text; version=1.0.0; charset=utf-8",
        )],
        body,
    )
        .into_response())
}

async fn list_profiles(
    State(state): State<ApiState>,
) -> Result<Json<Vec<IngesterProfile>>, ApiError> {
    state
        .profiles
        .list()
        .await
        .map(Json)
        .map_err(ApiError::internal)
}

async fn get_profile(
    State(state): State<ApiState>,
    Path(strategy_key): Path<String>,
) -> Result<Json<IngesterProfile>, ApiError> {
    let key = parse_key(&strategy_key)?;
    load_profile(&state, key).await.map(Json)
}

async fn get_profile_config(
    State(state): State<ApiState>,
    Path(strategy_key): Path<String>,
) -> Result<Json<ProfileConfigResponse>, ApiError> {
    let profile = load_profile(&state, parse_key(&strategy_key)?).await?;
    Ok(Json(ProfileConfigResponse {
        strategy_key: profile.strategy_key,
        desired_generation: profile.desired_generation,
        config_schema_version: profile.config_schema_version,
        config: profile.config,
    }))
}

async fn replace_profile_config(
    State(state): State<ApiState>,
    Path(strategy_key): Path<String>,
    headers: HeaderMap,
    Json(request): Json<ReplaceConfigRequest>,
) -> Result<Json<IngesterProfile>, ApiError> {
    let key = parse_key(&strategy_key)?;
    let expected_generation = expected_generation(&headers)?;
    if serde_json::to_vec(&request.config)
        .map_err(ApiError::internal)?
        .len()
        > MAX_CONFIG_BYTES
    {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "configuration_too_large",
            "strategy configuration must not exceed 16 KiB",
        ));
    }
    let current = load_profile(&state, key).await?;
    let factory = state.registry.factory(key).ok_or_else(|| {
        ApiError::new(
            StatusCode::CONFLICT,
            "unsupported_strategy",
            format!("strategy {key} is not compiled into this service"),
        )
    })?;
    if request.config_schema_version != factory.config_schema_version() {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "config_schema_mismatch",
            format!(
                "strategy {key} requires config schema version {}, received {}",
                factory.config_schema_version(),
                request.config_schema_version
            ),
        ));
    }
    factory.validate_config(&request.config).map_err(|error| {
        ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_configuration",
            error.to_string(),
        )
    })?;
    if current.desired_generation != expected_generation {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "generation_conflict",
            format!(
                "expected generation {expected_generation}, current generation is {}",
                current.desired_generation
            ),
        ));
    }
    state
        .profiles
        .replace_config(
            key,
            expected_generation,
            request.config_schema_version,
            &request.config,
        )
        .await
        .map(Json)
        .map_err(Into::into)
}

async fn start_profile(
    State(state): State<ApiState>,
    Path(strategy_key): Path<String>,
    headers: HeaderMap,
) -> Result<Json<LifecycleResponse>, ApiError> {
    request_lifecycle(
        &state,
        &strategy_key,
        &headers,
        DesiredState::Running,
        false,
    )
    .await
}

async fn stop_profile(
    State(state): State<ApiState>,
    Path(strategy_key): Path<String>,
    headers: HeaderMap,
) -> Result<Json<LifecycleResponse>, ApiError> {
    request_lifecycle(
        &state,
        &strategy_key,
        &headers,
        DesiredState::Stopped,
        false,
    )
    .await
}

async fn restart_profile(
    State(state): State<ApiState>,
    Path(strategy_key): Path<String>,
    headers: HeaderMap,
) -> Result<Json<LifecycleResponse>, ApiError> {
    request_lifecycle(&state, &strategy_key, &headers, DesiredState::Running, true).await
}

async fn request_lifecycle(
    state: &ApiState,
    strategy_key: &str,
    headers: &HeaderMap,
    desired_state: DesiredState,
    force_generation: bool,
) -> Result<Json<LifecycleResponse>, ApiError> {
    let key = parse_key(strategy_key)?;
    if desired_state == DesiredState::Running {
        let profile = load_profile(state, key).await?;
        state.registry.validate_profile(&profile).map_err(|error| {
            ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_profile",
                error.to_string(),
            )
        })?;
    }
    let expected_generation = expected_generation(headers)?;
    let profile = state
        .profiles
        .request_state(key, expected_generation, desired_state, force_generation)
        .await?;
    Ok(Json(LifecycleResponse {
        strategy_key: key,
        desired_state: profile.desired_state,
        desired_generation: profile.desired_generation,
    }))
}

async fn load_profile(
    state: &ApiState,
    key: IngesterStrategyKey,
) -> Result<IngesterProfile, ApiError> {
    state
        .profiles
        .get(key)
        .await
        .map_err(ApiError::internal)?
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::NOT_FOUND,
                "profile_not_found",
                format!("ingester profile {key} does not exist"),
            )
        })
}

fn parse_key(value: &str) -> Result<IngesterStrategyKey, ApiError> {
    IngesterStrategyKey::from_str(value).map_err(|error| {
        ApiError::new(StatusCode::NOT_FOUND, "unknown_strategy", error.to_string())
    })
}

fn expected_generation(headers: &HeaderMap) -> Result<i64, ApiError> {
    let value = headers
        .get("if-match")
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::PRECONDITION_REQUIRED,
                "generation_required",
                "If-Match with the current desired generation is required",
            )
        })?;
    let unquoted = value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(value);
    let generation = unquoted.parse::<i64>().map_err(|_| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_generation",
            "If-Match must contain a positive integer generation",
        )
    })?;
    if generation <= 0 {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_generation",
            "If-Match must contain a positive integer generation",
        ));
    }
    Ok(generation)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplaceConfigRequest {
    config_schema_version: i32,
    config: Value,
}

#[derive(Debug, Serialize)]
struct ProfileConfigResponse {
    strategy_key: IngesterStrategyKey,
    desired_generation: i64,
    config_schema_version: i32,
    config: Value,
}

#[derive(Debug, Serialize)]
struct LifecycleResponse {
    strategy_key: IngesterStrategyKey,
    desired_state: DesiredState,
    desired_generation: i64,
}

#[cfg(test)]
mod tests {
    use axum::http::{header::AUTHORIZATION, HeaderValue, Method};
    use chrono::TimeZone;
    use serde_json::json;
    use sqlx::postgres::PgPoolOptions;
    use tower::ServiceExt;

    use super::*;

    #[test]
    fn bearer_authentication_requires_an_exact_token() {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_static("Bearer 01234567890123456789012345678901"),
        );
        assert!(authorized(&headers, "01234567890123456789012345678901"));
        assert!(!authorized(&headers, "11234567890123456789012345678901"));
    }

    #[test]
    fn generation_accepts_quoted_etags() {
        let mut headers = HeaderMap::new();
        headers.insert("if-match", HeaderValue::from_static("\"42\""));
        assert_eq!(expected_generation(&headers).expect("generation"), 42);
    }

    #[test]
    fn coverage_targets_are_discovered_from_the_backfill_catalog() {
        let registry = crate::strategies::registry().expect("strategy registry");
        let targets = coverage_targets(&registry).expect("coverage targets");
        let products = targets
            .iter()
            .map(|target| target.product_key.as_str())
            .collect::<BTreeSet<_>>();

        assert!(products.contains("binance_futures_btcusdt_open_interest"));
        for product in [
            "kraken_instruments_backfill",
            "kraken_fee_schedules_backfill",
            "kraken_trade_candles_backfill",
            "kraken_mark_candles_backfill",
            "kraken_spot_candles_backfill",
            "kraken_open_interest_backfill",
            "kraken_future_basis_backfill",
            "kraken_aggressor_differential_backfill",
            "kraken_trade_volume_backfill",
            "kraken_trade_count_backfill",
            "kraken_cvd_backfill",
            "kraken_liquidation_volume_backfill",
            "kraken_spreads_backfill",
            "kraken_liquidity_backfill",
            "kraken_slippage_backfill",
            "kraken_funding_rates_backfill",
            "kraken_spot_btcusd_trade_prints_one_second_ohlcv_backfill",
        ] {
            assert!(
                products.contains(product),
                "missing coverage product {product}"
            );
        }
        assert_eq!(
            targets
                .iter()
                .flat_map(|target| target.backfill_strategy_keys.iter())
                .collect::<BTreeSet<_>>()
                .len(),
            registry.backfills().count()
        );

        let open_interest = targets
            .iter()
            .find(|target| target.product_key == "binance_futures_btcusdt_open_interest")
            .expect("open-interest coverage target");
        assert!(open_interest
            .gap_strategy_keys
            .iter()
            .any(|key| key == "binance_futures_btcusdt_open_interest"));
        assert!(open_interest
            .gap_strategy_keys
            .iter()
            .any(|key| key == "binance_futures_btcusdt_five_minute_open_interest_backfill"));

        let kraken = targets
            .iter()
            .find(|target| target.product_key == "kraken_open_interest_backfill")
            .expect("Kraken coverage target");
        assert!(kraken.relation.is_none());
        assert_eq!(
            kraken.backfill_strategy_keys,
            ["kraken_open_interest_backfill"]
        );
    }

    #[test]
    fn worker_allocation_summary_preserves_existing_fields_and_adds_live_allocation() {
        let timestamp = Utc.timestamp_opt(1_788_000_000, 0).single().unwrap();
        let summary = WorkerAllocationSummary {
            worker_id: "worker-1".to_owned(),
            hostname: "ingester-worker-1".to_owned(),
            worker_contract_version: 1,
            supported_strategies: json!({"strategy": 1}),
            maximum_backfills: 2,
            active_backfills: 0,
            capacity_units: 4,
            realtime_slot_limit: 1,
            allocation_contract_version: 1,
            realtime_strategies: json!(["binance_spot_btcusdt_l2_snapshots"]),
            image_digest: "sha256:image".to_owned(),
            source_revision: "revision".to_owned(),
            deployment_id: "deployment".to_owned(),
            lifecycle_state: "active".to_owned(),
            started_at: timestamp,
            heartbeat_at: timestamp,
            updated_at: timestamp,
            allocated_units: 2,
            available_units: 2,
            realtime_leases: 1,
            backfill_leases: 0,
            heartbeat_fresh: true,
            assigned_realtime_strategies: json!([{
                "strategy_key": "binance_spot_btcusdt_l2_snapshots",
                "desired_generation": 9,
                "applied_generation": 9,
                "observed_state": "running",
                "health_status": "healthy",
                "lease_expires_at": timestamp
            }]),
            assigned_backfills: json!([]),
        };

        let value = serde_json::to_value(summary).unwrap();
        for existing in [
            "worker_id",
            "hostname",
            "worker_contract_version",
            "supported_strategies",
            "maximum_backfills",
            "active_backfills",
            "capacity_units",
            "realtime_slot_limit",
            "allocation_contract_version",
            "realtime_strategies",
            "image_digest",
            "source_revision",
            "deployment_id",
            "lifecycle_state",
            "started_at",
            "heartbeat_at",
            "updated_at",
        ] {
            assert!(
                value.get(existing).is_some(),
                "missing existing field {existing}"
            );
        }
        assert_eq!(value["allocated_units"], 2);
        assert_eq!(value["available_units"], 2);
        assert_eq!(value["realtime_leases"], 1);
        assert_eq!(value["backfill_leases"], 0);
        assert_eq!(value["heartbeat_fresh"], true);
        assert_eq!(
            value["assigned_realtime_strategies"][0]["desired_generation"],
            9
        );
        assert_eq!(value["assigned_backfills"], json!([]));
    }

    #[tokio::test]
    async fn administrative_routes_reject_missing_authentication_before_database_access() {
        let pool = PgPoolOptions::new()
            .connect_lazy("postgres://unused:unused@localhost/unused")
            .expect("lazy pool");
        let (_readiness_sender, readiness) = ControlReadiness::channel(false);
        let api = ControlApi::new(
            ProfileRepository::new(pool),
            StrategyRegistry::default(),
            "01234567890123456789012345678901".to_owned(),
            readiness,
        )
        .expect("control API");

        let response = api
            .router()
            .oneshot(
                Request::builder()
                    .uri("/ingesters")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn readiness_tracks_supervisor_state() {
        let pool = PgPoolOptions::new()
            .connect_lazy("postgres://unused:unused@localhost/unused")
            .expect("lazy pool");
        let (readiness_sender, readiness) = ControlReadiness::channel(false);
        let api = ControlApi::new(
            ProfileRepository::new(pool),
            StrategyRegistry::default(),
            "01234567890123456789012345678901".to_owned(),
            readiness,
        )
        .expect("control API");

        let unavailable = api
            .router()
            .oneshot(
                Request::builder()
                    .uri("/health/ready")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(unavailable.status(), StatusCode::SERVICE_UNAVAILABLE);

        readiness_sender.send(true).expect("readiness receiver");
        let available = api
            .router()
            .oneshot(
                Request::builder()
                    .uri("/health/ready")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(available.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn config_updates_reject_payloads_over_the_database_limit_before_access() {
        let pool = PgPoolOptions::new()
            .connect_lazy("postgres://unused:unused@localhost/unused")
            .expect("lazy pool");
        let (_readiness_sender, readiness) = ControlReadiness::channel(false);
        let api = ControlApi::new(
            ProfileRepository::new(pool),
            StrategyRegistry::default(),
            "01234567890123456789012345678901".to_owned(),
            readiness,
        )
        .expect("control API");
        let body = serde_json::json!({
            "config_schema_version": 1,
            "config": {"value": "x".repeat(MAX_CONFIG_BYTES + 1)}
        })
        .to_string();

        let response = api
            .router()
            .oneshot(
                Request::builder()
                    .method(Method::PUT)
                    .uri(format!(
                        "/ingesters/{KEY}/config",
                        KEY = IngesterStrategyKey::BinanceSpotBtcusdtAggregateTrades
                    ))
                    .header(AUTHORIZATION, "Bearer 01234567890123456789012345678901")
                    .header("if-match", "1")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }
}
