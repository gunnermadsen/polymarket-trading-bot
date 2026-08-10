use std::{net::SocketAddr, str::FromStr, sync::Arc};

use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Path, State},
    http::{header::AUTHORIZATION, HeaderMap, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use subtle::ConstantTimeEq;
use tokio::{net::TcpListener, sync::watch};
use tokio_util::sync::CancellationToken;

use crate::{
    domain::{DesiredState, IngesterProfile, IngesterStrategyKey},
    persistence::ProfileRepository,
    runtime::StrategyRegistry,
};

use super::error::ApiError;

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
                profiles,
                registry,
                admin_token: Arc::<str>::from(admin_token),
                readiness,
            },
        })
    }

    pub fn router(&self) -> Router {
        let admin = Router::new()
            .route("/v1/ingesters", get(list_profiles))
            .route("/v1/ingesters/:strategy_key", get(get_profile))
            .route(
                "/v1/ingesters/:strategy_key/config",
                get(get_profile_config).put(replace_profile_config),
            )
            .route("/v1/ingesters/:strategy_key/start", post(start_profile))
            .route("/v1/ingesters/:strategy_key/stop", post(stop_profile))
            .route("/v1/ingesters/:strategy_key/restart", post(restart_profile))
            .route_layer(middleware::from_fn_with_state(
                self.state.clone(),
                require_admin,
            ));

        Router::new()
            .route("/health/live", get(liveness))
            .route("/health/ready", get(readiness))
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

#[derive(Clone)]
struct ApiState {
    profiles: ProfileRepository,
    registry: StrategyRegistry,
    admin_token: Arc<str>,
    readiness: ControlReadiness,
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
                    .uri("/v1/ingesters")
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
                        "/v1/ingesters/{KEY}/config",
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
