use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chrono::Utc;
use polymarket_bot::{
    btc::{
        runtime_status_from_inputs, BookRegistry, BtcDecisionStrategyConfig,
        BtcEntryAdmissionConfig, BtcPaperExperimentConfig, BtcPaperExperimentRunner,
        BtcPlaybookRuntimeHandle, BtcRepository, BtcRuntime, BtcRuntimeConfig, BtcRuntimeHandle,
        BtcStrategyConfig, PaperPreviewConfig, PaperVenue as BtcPaperVenue, PaperVenueConfig,
        BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_SCHEMA_VERSION,
        BTC_CHAINLINK_PATH_CONDITIONED_STRATEGY_VERSION,
        BTC_CHAINLINK_PERSISTENCE_CALIBRATED_FEATURE_SCHEMA_VERSION,
        BTC_CHAINLINK_PERSISTENCE_CALIBRATED_STRATEGY_VERSION, BTC_FEATURE_SCHEMA_VERSION,
        BTC_MARKET_ANCHORED_DIRECTIONAL_PREDICTION_STRATEGY_VERSION,
        BTC_MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION, BTC_STRATEGY_VERSION,
        BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION,
    },
    config::{AppConfig, BtcConfig},
    data_api::DataApiClient,
    events::ServiceEvent,
    execution::{
        live::LiveVenue, ExecutionVenue, LiveIdentityDiagnostics, LiveOrderDryRunDiagnostics,
        LiveOrderDryRunRequest, LivePoly1271FunderProbeRequest, LivePoly1271FunderProbeResponse,
        LiveVenueStatus, LiveWalletAddressDiagnostics,
    },
    grafana_live::{CountdownSnapshot, GrafanaLivePublisher},
    http as control_http,
    http::{
        ControlApi, HealthResponse, HealthStatus, HttpError, IngestionBackfillCancelResponse,
        IngestionBackfillEnqueueResponse, IngestionBackfillEventsResponse,
        IngestionBackfillJobResponse, IngestionBackfillJobsResponse, MetricsResponse,
        TradingProcessResponse, TradingProcessStartPreviewResponse, TradingProcessStatusResponse,
        TradingProcessesResponse,
    },
    ingestion::{
        job::BackfillRequest as IngestionBackfillRequest, repository::IngestionRepository,
    },
    models::{ProcessExecutionConfig, TradingProcess, TradingProcessConfig},
    store::Store,
};
use rust_decimal_macros::dec;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use sqlx::{postgres::PgPoolOptions, PgPool};
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

const BTC_PIPELINE_VERSION: &str = "btc_realtime_paper_pipeline_v11";
const BTC_PROCESS_SCHEMA_VERSION: &str = "btc_realtime_paper_process_v2";
const SELECTABLE_BTC_PIPELINE_VERSION: &str = "btc_realtime_paper_pipeline_v12";
const SELECTABLE_BTC_PROCESS_SCHEMA_VERSION: &str = "btc_realtime_paper_process_v3";
const LEGACY_BTC_PROCESS_SCHEMA_VERSION: &str = "btc_realtime_paper_process_v1";
const BTC_PROCESS_TYPE: &str = "btc_5m";
const BTC_PROCESS_SCOPE: &str = "realtime_paper";
const BTC_RUNTIME_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(45);
const COMPILED_SOURCE_IDENTITY: &str = env!("POLYMARKET_COMPILED_SOURCE_ID");

async fn preflight_btc_experiment_identity(
    pool: &PgPool,
    _process_id: uuid::Uuid,
    experiment_key: &str,
    experiment_id: uuid::Uuid,
) -> Result<()> {
    let identity_exists = sqlx::query_scalar::<_, bool>(
        r#"
        SELECT EXISTS (
          SELECT 1
          FROM polymarket.btc_paper_experiments
          WHERE experiment_id = $1 OR name = $2
        )
        "#,
    )
    .bind(experiment_id)
    .bind(experiment_key)
    .fetch_one(pool)
    .await
    .context("failed to preflight immutable BTC experiment identity")?;
    if identity_exists {
        bail!(
            "BTC experiment identity {experiment_key} already exists; every new explicit cohort start requires a new experiment key"
        );
    }
    Ok(())
}

async fn mark_btc_cohort_terminal(
    pool: &PgPool,
    experiment_id: uuid::Uuid,
    process_id: uuid::Uuid,
    status: &str,
    reason: &str,
    allow_missing_experiment: bool,
) -> Result<()> {
    if !matches!(status, "stopped" | "failed") || reason.trim().is_empty() {
        bail!("invalid BTC cohort terminal status or reason");
    }
    let mut tx = pool
        .begin()
        .await
        .context("failed to begin BTC cohort terminal transaction")?;
    let experiment_update = sqlx::query(
        r#"
        UPDATE polymarket.btc_paper_experiments
        SET status = $2,
            stopped_at = now(),
            stop_reason = $3,
            updated_at = now()
        WHERE experiment_id = $1 AND status = 'running'
        "#,
    )
    .bind(experiment_id)
    .bind(status)
    .bind(reason)
    .execute(&mut *tx)
    .await
    .context("failed to mark BTC experiment terminal")?;
    if experiment_update.rows_affected() == 0 {
        let existing_status = sqlx::query_scalar::<_, String>(
            r#"
            SELECT status
            FROM polymarket.btc_paper_experiments
            WHERE experiment_id = $1
            "#,
        )
        .bind(experiment_id)
        .fetch_optional(&mut *tx)
        .await
        .context("failed to inspect existing BTC experiment terminal state")?;
        match existing_status {
            Some(existing_status) if existing_status == status => {}
            Some(existing_status) => {
                bail!(
                    "cannot overwrite BTC experiment terminal state {existing_status} with {status}"
                );
            }
            None if allow_missing_experiment => {}
            None => bail!("BTC experiment disappeared during terminal transition"),
        }
    }
    let process_update = sqlx::query(
        r#"
        UPDATE polymarket.trading_processes
        SET status = $2,
            enabled = false,
            stopped_at = now(),
            stop_reason = $3,
            last_error = CASE WHEN $2 = 'failed' THEN $3 ELSE last_error END,
            updated_at = now()
        WHERE process_id = $1
          AND status IN ('starting','running','stopping')
        "#,
    )
    .bind(process_id)
    .bind(status)
    .bind(reason)
    .execute(&mut *tx)
    .await
    .context("failed to mark BTC process terminal")?;
    if process_update.rows_affected() == 0 {
        let existing_status = sqlx::query_scalar::<_, String>(
            r#"
            SELECT status
            FROM polymarket.trading_processes
            WHERE process_id = $1
            "#,
        )
        .bind(process_id)
        .fetch_optional(&mut *tx)
        .await
        .context("failed to inspect existing BTC process terminal state")?;
        match existing_status {
            Some(existing_status) if existing_status == status => {}
            Some(existing_status)
                if allow_missing_experiment
                    && matches!(
                        existing_status.as_str(),
                        "created" | "stopped" | "failed" | "completed" | "expired"
                    ) => {}
            Some(existing_status) => {
                bail!("cannot overwrite BTC process terminal state {existing_status} with {status}")
            }
            None => bail!("BTC process disappeared during terminal transition"),
        }
    }
    tx.commit()
        .await
        .context("failed to commit BTC cohort terminal transaction")?;
    Ok(())
}

#[derive(Debug, Clone)]
struct BtcProcessManagerConfig {
    btc: BtcConfig,
    gamma_base_url: String,
    clob_rest_base_url: String,
    clob_ws_url: String,
}

fn shared_market_data_config_compatible(left: &BtcRuntimeConfig, right: &BtcRuntimeConfig) -> bool {
    left.gamma_base_url == right.gamma_base_url
        && left.clob_rest_base_url == right.clob_rest_base_url
        && left.clob_ws_url == right.clob_ws_url
        && left.rtds_ws_url == right.rtds_ws_url
        && left.binance_ws_url == right.binance_ws_url
        && left.discovery_interval == right.discovery_interval
        && left.reconnect_initial_delay == right.reconnect_initial_delay
        && left.reconnect_max_delay == right.reconnect_max_delay
        && left.checkpoint_interval == right.checkpoint_interval
        && left.boundary_tick_max_delay == right.boundary_tick_max_delay
        && left.official_resolution_audit_grace == right.official_resolution_audit_grace
        && left.official_resolution_watch_retention == right.official_resolution_watch_retention
        && left.writer_capacity == right.writer_capacity
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct BtcRealtimePaperControlConfig {
    schema_version: String,
    next_experiment_key: String,
    preregistration_sha256: String,
    strategy: serde_json::Value,
    entry_admission: Option<BtcEntryAdmissionConfig>,
    runtime: BtcProcessRuntimeControl,
    paper: BtcProcessPaperControl,
}

impl Default for BtcRealtimePaperControlConfig {
    fn default() -> Self {
        Self {
            schema_version: String::new(),
            next_experiment_key: String::new(),
            preregistration_sha256: String::new(),
            strategy: serde_json::json!({}),
            entry_admission: None,
            runtime: BtcProcessRuntimeControl::default(),
            paper: BtcProcessPaperControl::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BtcDefinitionUse {
    ExplicitStart,
    DurableResume,
}

fn parse_btc_process_control(
    mut value: serde_json::Value,
    definition_use: BtcDefinitionUse,
) -> Result<BtcRealtimePaperControlConfig, HttpError> {
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| HttpError::bad_request("BTC process schema_version is required"))?;

    if schema_version == LEGACY_BTC_PROCESS_SCHEMA_VERSION {
        if definition_use != BtcDefinitionUse::DurableResume {
            return Err(HttpError::bad_request(format!(
                "BTC process schema_version {LEGACY_BTC_PROCESS_SCHEMA_VERSION} is resume-only; new and explicitly restarted processes must use {BTC_PROCESS_SCHEMA_VERSION} or {SELECTABLE_BTC_PROCESS_SCHEMA_VERSION}"
            )));
        }
        let object = value.as_object_mut().ok_or_else(|| {
            HttpError::bad_request("BTC process control configuration must be an object")
        })?;
        if let Some(retired) = object.remove("ml_shadow") {
            let retired = retired.as_object().ok_or_else(|| {
                HttpError::bad_request("legacy ml_shadow compatibility value must be an object")
            })?;
            if retired.keys().any(|key| key != "enabled")
                || retired
                    .get("enabled")
                    .is_some_and(|enabled| !enabled.is_boolean())
            {
                return Err(HttpError::bad_request(
                    "legacy ml_shadow compatibility value may contain only a boolean enabled field",
                ));
            }
        }
        object.insert(
            "schema_version".to_string(),
            serde_json::Value::String(BTC_PROCESS_SCHEMA_VERSION.to_string()),
        );
    } else if !matches!(
        schema_version,
        BTC_PROCESS_SCHEMA_VERSION | SELECTABLE_BTC_PROCESS_SCHEMA_VERSION
    ) {
        return Err(HttpError::bad_request(format!(
            "BTC process schema_version must be {BTC_PROCESS_SCHEMA_VERSION} or {SELECTABLE_BTC_PROCESS_SCHEMA_VERSION}"
        )));
    }

    serde_json::from_value(value).map_err(|error| {
        HttpError::bad_request(format!(
            "invalid process config.raw.btc_realtime_paper: {error}"
        ))
    })
}

fn resolve_btc_strategy(
    control: &BtcRealtimePaperControlConfig,
) -> Result<BtcStrategyConfig, HttpError> {
    let strategy_overrides = control.strategy.as_object().ok_or_else(|| {
        HttpError::bad_request("btc_realtime_paper.strategy must be a JSON object")
    })?;
    let is_selectable_contract = control.schema_version == SELECTABLE_BTC_PROCESS_SCHEMA_VERSION;

    if is_selectable_contract {
        if !strategy_overrides.contains_key("decision_strategy") {
            return Err(HttpError::bad_request(format!(
                "BTC process schema_version {SELECTABLE_BTC_PROCESS_SCHEMA_VERSION} requires strategy.decision_strategy"
            )));
        }
        for compiled_or_legacy_key in [
            "strategy_version",
            "feature_schema_version",
            "volatility_continuation",
        ] {
            if strategy_overrides.contains_key(compiled_or_legacy_key) {
                return Err(HttpError::bad_request(format!(
                    "BTC strategy setting {compiled_or_legacy_key} cannot be supplied with schema_version {SELECTABLE_BTC_PROCESS_SCHEMA_VERSION}"
                )));
            }
        }
    } else if strategy_overrides.contains_key("decision_strategy") {
        return Err(HttpError::bad_request(format!(
            "BTC strategy.decision_strategy requires schema_version {SELECTABLE_BTC_PROCESS_SCHEMA_VERSION}"
        )));
    }

    let mut strategy_value = serde_json::to_value(BtcStrategyConfig::default())
        .map_err(|error| HttpError::internal(error.to_string()))?;
    let strategy_object = strategy_value
        .as_object_mut()
        .ok_or_else(|| HttpError::internal("default BTC strategy did not serialize as object"))?;
    strategy_object.insert(
        "volatility_continuation".to_string(),
        serde_json::Value::Null,
    );
    strategy_object.insert("decision_strategy".to_string(), serde_json::Value::Null);
    for (key, value) in strategy_overrides {
        let Some(slot) = strategy_object.get_mut(key) else {
            return Err(HttpError::bad_request(format!(
                "unsupported BTC strategy setting {key}"
            )));
        };
        *slot = value.clone();
    }

    if is_selectable_contract {
        let selection: BtcDecisionStrategyConfig = serde_json::from_value(
            strategy_overrides
                .get("decision_strategy")
                .cloned()
                .expect("selectable contract requires a decision strategy"),
        )
        .map_err(|error| {
            HttpError::bad_request(format!("invalid BTC decision strategy selection: {error}"))
        })?;
        let (strategy_version, feature_schema_version) = match selection {
            BtcDecisionStrategyConfig::ChainlinkFairValue {} => {
                (BTC_STRATEGY_VERSION, BTC_FEATURE_SCHEMA_VERSION)
            }
            BtcDecisionStrategyConfig::ChainlinkPersistenceCalibratedFairValue { .. } => (
                BTC_CHAINLINK_PERSISTENCE_CALIBRATED_STRATEGY_VERSION,
                BTC_CHAINLINK_PERSISTENCE_CALIBRATED_FEATURE_SCHEMA_VERSION,
            ),
            BtcDecisionStrategyConfig::ChainlinkPathConditionedFairValue { .. } => (
                BTC_CHAINLINK_PATH_CONDITIONED_STRATEGY_VERSION,
                BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_SCHEMA_VERSION,
            ),
            BtcDecisionStrategyConfig::VolatilityContinuation { .. } => (
                BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION,
                BTC_FEATURE_SCHEMA_VERSION,
            ),
            BtcDecisionStrategyConfig::MarketAnchoredFairValue { .. } => (
                BTC_MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION,
                BTC_FEATURE_SCHEMA_VERSION,
            ),
            BtcDecisionStrategyConfig::MarketAnchoredDirectionalPrediction { .. } => (
                BTC_MARKET_ANCHORED_DIRECTIONAL_PREDICTION_STRATEGY_VERSION,
                BTC_FEATURE_SCHEMA_VERSION,
            ),
        };
        strategy_object.insert(
            "strategy_version".to_string(),
            serde_json::Value::String(strategy_version.to_string()),
        );
        strategy_object.insert(
            "feature_schema_version".to_string(),
            serde_json::Value::String(feature_schema_version.to_string()),
        );
    }

    let strategy: BtcStrategyConfig = serde_json::from_value(strategy_value).map_err(|error| {
        HttpError::bad_request(format!("invalid BTC strategy settings: {error}"))
    })?;
    strategy
        .validate()
        .map_err(|error| HttpError::bad_request(error.to_string()))?;
    let compiled_identity_valid = matches!(
        (
            strategy.strategy_version.as_str(),
            strategy.feature_schema_version.as_str()
        ),
        (BTC_STRATEGY_VERSION, BTC_FEATURE_SCHEMA_VERSION)
            | (
                BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION,
                BTC_FEATURE_SCHEMA_VERSION
            )
            | (
                BTC_MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION,
                BTC_FEATURE_SCHEMA_VERSION
            )
            | (
                BTC_MARKET_ANCHORED_DIRECTIONAL_PREDICTION_STRATEGY_VERSION,
                BTC_FEATURE_SCHEMA_VERSION
            )
            | (
                BTC_CHAINLINK_PERSISTENCE_CALIBRATED_STRATEGY_VERSION,
                BTC_CHAINLINK_PERSISTENCE_CALIBRATED_FEATURE_SCHEMA_VERSION
            )
            | (
                BTC_CHAINLINK_PATH_CONDITIONED_STRATEGY_VERSION,
                BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_SCHEMA_VERSION
            )
    );
    if !compiled_identity_valid {
        return Err(HttpError::bad_request(
            "BTC strategy and feature schema versions are compiled identities and cannot be overridden",
        ));
    }
    Ok(strategy)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct BtcProcessRuntimeControl {
    strategy_interval_ms: u64,
    official_resolution_audit_grace_secs: u64,
    official_resolution_watch_retention_secs: u64,
}

impl Default for BtcProcessRuntimeControl {
    fn default() -> Self {
        Self {
            strategy_interval_ms: 1_000,
            official_resolution_audit_grace_secs: 120,
            official_resolution_watch_retention_secs: 3_600,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct BtcProcessPaperControl {
    arrival_latency_ms: u64,
    visible_depth_haircut: rust_decimal::Decimal,
    starting_collateral_usd: rust_decimal::Decimal,
    stress_previews: Vec<BtcProcessPaperPreviewControl>,
}

impl Default for BtcProcessPaperControl {
    fn default() -> Self {
        Self {
            arrival_latency_ms: 150,
            visible_depth_haircut: dec!(0.80),
            starting_collateral_usd: dec!(1000),
            stress_previews: vec![
                BtcProcessPaperPreviewControl {
                    scenario_key: "latency_300ms_depth_65pct".to_string(),
                    arrival_latency_ms: 300,
                    visible_depth_haircut: dec!(0.65),
                },
                BtcProcessPaperPreviewControl {
                    scenario_key: "latency_600ms_depth_50pct".to_string(),
                    arrival_latency_ms: 600,
                    visible_depth_haircut: dec!(0.50),
                },
            ],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct BtcProcessPaperPreviewControl {
    scenario_key: String,
    arrival_latency_ms: u64,
    visible_depth_haircut: rust_decimal::Decimal,
}

#[derive(Clone)]
struct ResolvedBtcProcessDefinition {
    control: BtcRealtimePaperControlConfig,
    strategy: BtcStrategyConfig,
    entry_admission: Option<BtcEntryAdmissionConfig>,
    runtime: BtcRuntimeConfig,
    paper_venue: PaperVenueConfig,
    paper_stress_previews: Vec<PaperPreviewConfig>,
}

struct PreparedBtcStartDefinition {
    experiment_id: uuid::Uuid,
    experiment_key: String,
    preregistration_sha256: String,
    strategy: BtcStrategyConfig,
    entry_admission: Option<BtcEntryAdmissionConfig>,
    runtime: BtcRuntimeConfig,
    paper_venue: PaperVenueConfig,
    paper_stress_previews: Vec<PaperPreviewConfig>,
    frozen_process_config: TradingProcessConfig,
    config_hash: String,
}

fn validate_btc_start_eligibility(
    process: &TradingProcess,
    realtime_enabled: bool,
    paper_enabled: bool,
) -> Result<(), HttpError> {
    validate_btc_process_capability(process, realtime_enabled, paper_enabled)?;
    if process.enabled || matches!(process.status.as_str(), "starting" | "running" | "stopping") {
        return Err(HttpError::conflict(
            "BTC trading process already claims an active lifecycle",
        ));
    }
    if !matches!(
        process.status.as_str(),
        "created" | "stopped" | "failed" | "completed"
    ) {
        return Err(HttpError::bad_request(format!(
            "BTC trading process cannot start from status {}",
            process.status
        )));
    }
    Ok(())
}

fn validate_btc_process_capability(
    process: &TradingProcess,
    realtime_enabled: bool,
    paper_enabled: bool,
) -> Result<(), HttpError> {
    if !BtcProcessManager::is_managed_process(process) {
        return Err(HttpError::bad_request(
            "process is not a managed BTC realtime-paper process",
        ));
    }
    if !realtime_enabled || !paper_enabled {
        return Err(HttpError::bad_request(
            "BTC realtime-paper capability is disabled for this deployment",
        ));
    }
    let execution = process.effective_execution();
    if execution.mode != "paper" {
        return Err(HttpError::bad_request(
            "BTC realtime-paper process execution.mode must be paper",
        ));
    }
    if !execution.execute_signals {
        return Err(HttpError::bad_request(
            "BTC realtime-paper process execution.execute_signals must be true",
        ));
    }
    if execution.live_capital {
        return Err(HttpError::bad_request(
            "BTC realtime-paper process execution.live_capital must be false",
        ));
    }
    Ok(())
}

fn prepare_btc_start_definition(
    resolved: ResolvedBtcProcessDefinition,
) -> Result<PreparedBtcStartDefinition, HttpError> {
    let ResolvedBtcProcessDefinition {
        control,
        strategy,
        entry_admission,
        runtime,
        paper_venue,
        paper_stress_previews,
    } = resolved;
    let (pipeline_version, process_schema_version) = match control.schema_version.as_str() {
        BTC_PROCESS_SCHEMA_VERSION => (BTC_PIPELINE_VERSION, BTC_PROCESS_SCHEMA_VERSION),
        SELECTABLE_BTC_PROCESS_SCHEMA_VERSION => (
            SELECTABLE_BTC_PIPELINE_VERSION,
            SELECTABLE_BTC_PROCESS_SCHEMA_VERSION,
        ),
        schema_version => {
            return Err(HttpError::internal(format!(
                "resolved unsupported BTC process schema {schema_version}"
            )))
        }
    };
    let experiment_key = control.next_experiment_key;
    let preregistration_sha256 = control.preregistration_sha256;
    let experiment_id = uuid::Uuid::new_v5(
        &uuid::Uuid::NAMESPACE_URL,
        format!("polymarket-bot/btc-paper/{experiment_key}").as_bytes(),
    );
    let mut frozen_raw = serde_json::json!({
        "pipeline_version": pipeline_version,
        "process_schema_version": process_schema_version,
        "preregistration_sha256": &preregistration_sha256,
        "build": {
            "package_version": env!("CARGO_PKG_VERSION"),
            "compiled_source_identity": COMPILED_SOURCE_IDENTITY,
        },
        "strategy": &strategy,
        "runtime": &runtime,
        "paper": {
            "execution_enabled": true,
            "venue": &paper_venue,
            "stress_previews": &paper_stress_previews,
        }
    });
    if let Some(entry_admission) = entry_admission.as_ref() {
        frozen_raw
            .as_object_mut()
            .expect("BTC frozen process config is an object")
            .insert(
                "entry_admission".to_string(),
                serde_json::to_value(entry_admission)
                    .map_err(|error| HttpError::internal(error.to_string()))?,
            );
    }
    let frozen_process_config = TradingProcessConfig {
        execution: Some(ProcessExecutionConfig {
            mode: Some("paper".to_string()),
            execute_signals: true,
            live_capital: false,
            taker_fee_rate: None,
        }),
        raw: frozen_raw,
        ..TradingProcessConfig::default()
    };
    let config_hash = hash_btc_frozen_process_config(&frozen_process_config)?;
    Ok(PreparedBtcStartDefinition {
        experiment_id,
        experiment_key,
        preregistration_sha256,
        strategy,
        entry_admission,
        runtime,
        paper_venue,
        paper_stress_previews,
        frozen_process_config,
        config_hash,
    })
}

fn hash_btc_frozen_process_config(
    frozen_process_config: &TradingProcessConfig,
) -> Result<String, HttpError> {
    Ok(format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(frozen_process_config)
                .map_err(|error| HttpError::internal(error.to_string()))?
        )
    ))
}

fn resume_process_contract_projection(mut config: serde_json::Value) -> serde_json::Value {
    if let Some(raw) = config
        .get_mut("raw")
        .and_then(serde_json::Value::as_object_mut)
    {
        if let Some(build) = raw
            .get_mut("build")
            .and_then(serde_json::Value::as_object_mut)
        {
            build.remove("compiled_source_identity");
        }
        if let Some(runtime) = raw
            .get_mut("runtime")
            .and_then(serde_json::Value::as_object_mut)
        {
            runtime.remove("clob_heartbeat_interval");
            runtime.remove("rtds_heartbeat_interval");
            runtime.remove("binance_heartbeat_interval");
        }
        if raw
            .get("process_schema_version")
            .and_then(serde_json::Value::as_str)
            == Some(LEGACY_BTC_PROCESS_SCHEMA_VERSION)
        {
            raw.insert(
                "process_schema_version".to_string(),
                serde_json::Value::String(BTC_PROCESS_SCHEMA_VERSION.to_string()),
            );
            raw.remove("ml_shadow");
        }
    }
    config
}

struct ActiveBtcPlaybook {
    process_id: uuid::Uuid,
    experiment_id: uuid::Uuid,
    experiment_key: String,
    config_hash: String,
    runtime: BtcPlaybookRuntimeHandle,
}

struct SharedBtcRuntime {
    config: BtcRuntimeConfig,
    runtime: BtcRuntimeHandle,
}

#[derive(Debug, Clone)]
struct PendingBtcTerminal {
    process_id: uuid::Uuid,
    experiment_id: uuid::Uuid,
    experiment_key: String,
    config_hash: String,
    terminal_status: String,
    terminal_reason: String,
    allow_missing_experiment: bool,
}

#[derive(Clone)]
struct BtcProcessManager {
    store: Store,
    pool: PgPool,
    repository: BtcRepository,
    config: BtcProcessManagerConfig,
    shutting_down: Arc<AtomicBool>,
    transition: Arc<tokio::sync::Mutex<()>>,
    active_playbooks: Arc<tokio::sync::Mutex<HashMap<uuid::Uuid, ActiveBtcPlaybook>>>,
    shared_runtime: Arc<tokio::sync::Mutex<Option<SharedBtcRuntime>>>,
    terminal_pending: Arc<tokio::sync::Mutex<HashMap<uuid::Uuid, PendingBtcTerminal>>>,
}

impl BtcProcessManager {
    fn new(
        store: Store,
        pool: PgPool,
        repository: BtcRepository,
        config: BtcProcessManagerConfig,
    ) -> Self {
        Self {
            store,
            pool,
            repository,
            config,
            shutting_down: Arc::new(AtomicBool::new(false)),
            transition: Arc::new(tokio::sync::Mutex::new(())),
            active_playbooks: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            shared_runtime: Arc::new(tokio::sync::Mutex::new(None)),
            terminal_pending: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    async fn ensure_shared_runtime(
        &self,
        config: &BtcRuntimeConfig,
    ) -> Result<
        (
            Arc<tokio::sync::RwLock<polymarket_bot::btc::RealtimeState>>,
            Arc<tokio::sync::RwLock<BookRegistry>>,
        ),
        HttpError,
    > {
        let active_playbooks = self.active_playbooks.lock().await.len();
        let retired_runtime = {
            let mut shared = self.shared_runtime.lock().await;
            if let Some(existing) = shared.as_ref() {
                let compatible = shared_market_data_config_compatible(&existing.config, config);
                if compatible && existing.runtime.is_running() {
                    return Ok((
                        existing.runtime.shared_state(),
                        existing.runtime.shared_book_registry(),
                    ));
                }
                if active_playbooks > 0 {
                    let message = if compatible {
                        "BTC shared market-data runtime is not running"
                    } else {
                        "BTC playbook market-data runtime settings differ from the active shared runtime"
                    };
                    return Err(HttpError::conflict(message));
                }
            }
            shared.take()
        };
        if let Some(retired) = retired_runtime {
            match tokio::time::timeout(BTC_RUNTIME_SHUTDOWN_TIMEOUT, retired.runtime.shutdown())
                .await
            {
                Ok(Ok(())) => {}
                Ok(Err(error)) => warn!(
                    error = ?error,
                    "retired BTC shared market-data runtime reported an integrity failure"
                ),
                Err(_) => warn!(
                    timeout_secs = BTC_RUNTIME_SHUTDOWN_TIMEOUT.as_secs(),
                    "timed out retiring BTC shared market-data runtime"
                ),
            }
        }

        let state = Arc::new(tokio::sync::RwLock::new(
            polymarket_bot::btc::RealtimeState::default(),
        ));
        let books = Arc::new(tokio::sync::RwLock::new(BookRegistry::new(
            uuid::Uuid::new_v4(),
        )));
        let heartbeat = self.config.btc.data_source_heartbeat;
        let runtime = BtcRuntime::new(config.clone(), heartbeat, self.repository.clone())
            .with_shared_state(state.clone())
            .with_shared_book_registry(books.clone())
            .start()
            .await
            .map_err(|error| {
                HttpError::internal(format!(
                    "failed to start shared BTC market-data runtime: {error:#}"
                ))
            })?;
        info!(
            clob_heartbeat_interval_secs = heartbeat.clob_interval.as_secs(),
            rtds_heartbeat_interval_secs = heartbeat.rtds_interval.as_secs(),
            binance_heartbeat_interval_secs = heartbeat.binance_interval.as_secs(),
            "BTC shared market-data runtime started with global heartbeat configuration"
        );
        let mut shared = self.shared_runtime.lock().await;
        debug_assert!(shared.is_none());
        *shared = Some(SharedBtcRuntime {
            config: config.clone(),
            runtime,
        });
        Ok((state, books))
    }

    async fn shutdown_shared_runtime_if_idle(&self) {
        if !self.active_playbooks.lock().await.is_empty() {
            return;
        }
        let shared = { self.shared_runtime.lock().await.take() };
        let Some(shared) = shared else {
            return;
        };
        match tokio::time::timeout(BTC_RUNTIME_SHUTDOWN_TIMEOUT, shared.runtime.shutdown()).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => warn!(
                error = ?error,
                "idle BTC shared market-data runtime reported an integrity failure during shutdown"
            ),
            Err(_) => warn!(
                timeout_secs = BTC_RUNTIME_SHUTDOWN_TIMEOUT.as_secs(),
                "timed out shutting down idle BTC shared market-data runtime"
            ),
        }
    }

    fn is_managed_process(process: &TradingProcess) -> bool {
        Self::is_managed_identity(&process.process_type, &process.process_scope)
    }

    fn is_managed_identity(process_type: &str, process_scope: &str) -> bool {
        process_type == BTC_PROCESS_TYPE && process_scope == BTC_PROCESS_SCOPE
    }

    async fn record_event(
        &self,
        process_id: uuid::Uuid,
        level: &str,
        event_type: &str,
        message: &str,
        metadata: serde_json::Value,
    ) {
        if let Err(event_error) = self
            .store
            .record_trading_process_event(process_id, level, event_type, Some(message), metadata)
            .await
        {
            warn!(
                error = %event_error,
                process_id = %process_id,
                event_type,
                "failed to persist trading process lifecycle event"
            );
        }
    }

    fn validate_start_definition(
        &self,
        process: &TradingProcess,
    ) -> Result<ResolvedBtcProcessDefinition, HttpError> {
        self.validate_definition(process, BtcDefinitionUse::ExplicitStart)
    }

    fn validate_resume_definition(
        &self,
        process: &TradingProcess,
    ) -> Result<ResolvedBtcProcessDefinition, HttpError> {
        self.validate_definition(process, BtcDefinitionUse::DurableResume)
    }

    fn validate_definition(
        &self,
        process: &TradingProcess,
        definition_use: BtcDefinitionUse,
    ) -> Result<ResolvedBtcProcessDefinition, HttpError> {
        if definition_use == BtcDefinitionUse::ExplicitStart {
            validate_btc_start_eligibility(
                process,
                self.config.btc.realtime_enabled,
                self.config.btc.paper_enabled,
            )?;
        } else {
            validate_btc_process_capability(
                process,
                self.config.btc.realtime_enabled,
                self.config.btc.paper_enabled,
            )?;
        }
        if process
            .config
            .execution
            .as_ref()
            .and_then(|execution| execution.taker_fee_rate)
            .is_some()
        {
            return Err(HttpError::bad_request(
                "BTC realtime-paper config accepts only execution safety fields and raw.btc_realtime_paper settings",
            ));
        }
        let raw = process.config.raw.as_object().ok_or_else(|| {
            HttpError::bad_request("BTC process config.raw must be a JSON object")
        })?;
        if raw.len() != 1 || !raw.contains_key("btc_realtime_paper") {
            return Err(HttpError::bad_request(
                "BTC process config.raw may contain only btc_realtime_paper",
            ));
        }
        let control_value = process
            .config
            .raw
            .get("btc_realtime_paper")
            .cloned()
            .ok_or_else(|| {
                HttpError::bad_request(
                    "process config.raw.btc_realtime_paper is required before start",
                )
            })?;
        let mut control = parse_btc_process_control(control_value, definition_use)?;
        control.next_experiment_key = control.next_experiment_key.trim().to_string();
        control.preregistration_sha256 = control.preregistration_sha256.trim().to_ascii_lowercase();
        if control.next_experiment_key.is_empty() {
            return Err(HttpError::bad_request(
                "next_experiment_key must not be empty",
            ));
        }
        if control.next_experiment_key.len() > 128
            || !control
                .next_experiment_key
                .bytes()
                .enumerate()
                .all(|(index, byte)| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || (index > 0 && matches!(byte, b'.' | b'_' | b'-'))
                })
        {
            return Err(HttpError::bad_request(
                "next_experiment_key must be a lowercase slug of at most 128 characters",
            ));
        }
        if control.preregistration_sha256.len() != 64
            || !control
                .preregistration_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(HttpError::bad_request(
                "preregistration_sha256 must be a 64-character hexadecimal digest",
            ));
        }
        let strategy = resolve_btc_strategy(&control)?;
        if let Some(entry_admission) = control.entry_admission.as_ref() {
            entry_admission
                .validate()
                .map_err(|error| HttpError::bad_request(error.to_string()))?;
        }
        if strategy
            .min_seconds_after_open
            .checked_add(strategy.min_seconds_before_close)
            .is_none_or(|entry_gate_seconds| entry_gate_seconds >= 300)
        {
            return Err(HttpError::bad_request(
                "BTC entry timing gates leave no tradable portion of a five-minute window",
            ));
        }
        if !(1..=60_000).contains(&control.runtime.strategy_interval_ms)
            || !(1..=86_400).contains(&control.runtime.official_resolution_audit_grace_secs)
            || !(600..=604_800).contains(&control.runtime.official_resolution_watch_retention_secs)
        {
            return Err(HttpError::bad_request(
                "BTC runtime timings exceed the supported process safety bounds",
            ));
        }
        let runtime = BtcRuntimeConfig {
            enabled: true,
            gamma_base_url: self.config.gamma_base_url.clone(),
            clob_rest_base_url: self.config.clob_rest_base_url.clone(),
            clob_ws_url: self.config.clob_ws_url.clone(),
            rtds_ws_url: self.config.btc.rtds_ws_url.clone(),
            binance_ws_url: self.config.btc.binance_ws_url.clone(),
            strategy_interval: Duration::from_millis(control.runtime.strategy_interval_ms),
            max_book_age: Duration::from_millis(strategy.max_book_age_ms as u64),
            max_reference_age: Duration::from_millis(strategy.max_reference_age_ms as u64),
            boundary_tick_max_delay: Duration::from_millis(
                strategy.max_chainlink_open_delay_ms as u64,
            ),
            official_resolution_audit_grace: Duration::from_secs(
                control.runtime.official_resolution_audit_grace_secs,
            ),
            official_resolution_watch_retention: Duration::from_secs(
                control.runtime.official_resolution_watch_retention_secs,
            ),
            ..BtcRuntimeConfig::default()
        };
        runtime
            .validate()
            .map_err(|error| HttpError::bad_request(error.to_string()))?;
        if !(1..=60_000).contains(&control.paper.arrival_latency_ms)
            || control.paper.starting_collateral_usd <= rust_decimal::Decimal::ZERO
            || control.paper.starting_collateral_usd > dec!(1000000)
        {
            return Err(HttpError::bad_request(
                "BTC paper arrival latency and starting collateral must be positive",
            ));
        }
        let paper_venue = PaperVenueConfig {
            arrival_latency: Duration::from_millis(control.paper.arrival_latency_ms),
            visible_depth_haircut: control.paper.visible_depth_haircut,
            max_book_age: chrono::Duration::milliseconds(strategy.max_book_age_ms),
            starting_collateral_usd: control.paper.starting_collateral_usd,
        };
        paper_venue
            .validate()
            .map_err(|error| HttpError::bad_request(error.to_string()))?;
        if let Some(high_water_mark) = control
            .entry_admission
            .as_ref()
            .and_then(|admission| admission.daily_realized_pnl_high_water_mark.as_ref())
        {
            high_water_mark
                .validate_against_starting_collateral(control.paper.starting_collateral_usd)
                .map_err(|error| HttpError::bad_request(error.to_string()))?;
        }
        let mut preview_keys = HashSet::new();
        let mut paper_stress_previews = Vec::with_capacity(control.paper.stress_previews.len());
        for preview in &control.paper.stress_previews {
            if !(1..=60_000).contains(&preview.arrival_latency_ms) {
                return Err(HttpError::bad_request(
                    "BTC paper stress-preview latency must be positive",
                ));
            }
            let resolved = PaperPreviewConfig {
                scenario_key: preview.scenario_key.trim().to_string(),
                arrival_latency: Duration::from_millis(preview.arrival_latency_ms),
                visible_depth_haircut: preview.visible_depth_haircut,
            };
            resolved
                .validate()
                .map_err(|error| HttpError::bad_request(error.to_string()))?;
            if !preview_keys.insert(resolved.scenario_key.clone()) {
                return Err(HttpError::bad_request(format!(
                    "duplicate BTC paper stress-preview scenario {}",
                    resolved.scenario_key
                )));
            }
            paper_stress_previews.push(resolved);
        }
        Ok(ResolvedBtcProcessDefinition {
            entry_admission: control.entry_admission.clone(),
            control,
            strategy,
            runtime,
            paper_venue,
            paper_stress_previews,
        })
    }

    fn prepare_start_definition(
        &self,
        process: &TradingProcess,
    ) -> Result<PreparedBtcStartDefinition, HttpError> {
        prepare_btc_start_definition(self.validate_start_definition(process)?)
    }

    fn prepare_resume_definition(
        &self,
        process: &TradingProcess,
    ) -> Result<PreparedBtcStartDefinition, HttpError> {
        prepare_btc_start_definition(self.validate_resume_definition(process)?)
    }

    async fn ensure_start_slot_available(&self, process_id: uuid::Uuid) -> Result<(), HttpError> {
        if let Some(pending) = self.terminal_pending.lock().await.get(&process_id) {
            return Err(HttpError::conflict(format!(
                "BTC experiment {} still has a pending terminal transition",
                pending.experiment_key
            )));
        }
        if let Some(active) = self.active_playbooks.lock().await.get(&process_id) {
            return Err(HttpError::conflict(format!(
                "BTC process {} is already running experiment {}",
                active.process_id, active.experiment_key
            )));
        }
        Ok(())
    }

    async fn preview_start(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcessStartPreviewResponse, HttpError> {
        if self.shutting_down.load(Ordering::Acquire) {
            return Err(HttpError::conflict(
                "BTC runtime cannot start while the service is shutting down",
            ));
        }
        let _transition_guard = self.transition.lock().await;
        if self.shutting_down.load(Ordering::Acquire) {
            return Err(HttpError::conflict(
                "BTC runtime cannot start while the service is shutting down",
            ));
        }
        self.ensure_start_slot_available(process_id).await?;
        let process = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        let prepared = self.prepare_start_definition(&process)?;
        preflight_btc_experiment_identity(
            &self.pool,
            process_id,
            &prepared.experiment_key,
            prepared.experiment_id,
        )
        .await
        .map_err(|error| HttpError::conflict(error.to_string()))?;
        Ok(TradingProcessStartPreviewResponse {
            process_id,
            experiment_id: prepared.experiment_id,
            experiment_key: prepared.experiment_key,
            preregistration_sha256: prepared.preregistration_sha256,
            config_hash: prepared.config_hash,
            frozen_process_config: prepared.frozen_process_config,
        })
    }

    async fn start_process(&self, process_id: uuid::Uuid) -> Result<TradingProcess, HttpError> {
        if self.shutting_down.load(Ordering::Acquire) {
            return Err(HttpError::conflict(
                "BTC runtime cannot start while the service is shutting down",
            ));
        }
        let transition_guard = self.transition.clone().lock_owned().await;
        if self.shutting_down.load(Ordering::Acquire) {
            return Err(HttpError::conflict(
                "BTC runtime cannot start while the service is shutting down",
            ));
        }
        let manager = self.clone();
        tokio::spawn(async move {
            let _transition_guard = transition_guard;
            manager.start_process_locked(process_id).await
        })
        .await
        .map_err(|error| {
            HttpError::internal(format!("BTC start transition task failed: {error}"))
        })?
    }

    async fn resume_durable_processes(&self) -> Result<Vec<TradingProcess>, HttpError> {
        if self.shutting_down.load(Ordering::Acquire) {
            return Err(HttpError::conflict(
                "BTC runtime cannot resume while the service is shutting down",
            ));
        }
        let _transition_guard = self.transition.lock().await;
        let candidates = sqlx::query_scalar::<_, uuid::Uuid>(
            r#"
            SELECT p.process_id
            FROM polymarket.trading_processes p
            JOIN polymarket.btc_paper_experiments e
              ON e.process_id = p.process_id
             AND e.name = p.config #>> '{raw,btc_realtime_paper,next_experiment_key}'
            WHERE p.process_type = 'btc_5m'
              AND p.process_scope = 'realtime_paper'
              AND p.config #>> '{raw,btc_realtime_paper,schema_version}' IN ($1, $2, $3)
              AND p.enabled
              AND p.status IN ('starting','running','stopping')
              AND p.stopped_at IS NULL
              AND e.status = 'running'
              AND e.stopped_at IS NULL
            ORDER BY e.started_at DESC
            "#,
        )
        .bind(SELECTABLE_BTC_PROCESS_SCHEMA_VERSION)
        .bind(BTC_PROCESS_SCHEMA_VERSION)
        .bind(LEGACY_BTC_PROCESS_SCHEMA_VERSION)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| HttpError::internal(error.to_string()))?;
        if candidates.is_empty() {
            let durable_claims = sqlx::query_scalar::<_, i64>(
                r#"
                SELECT COUNT(DISTINCT p.process_id)
                FROM polymarket.trading_processes p
                LEFT JOIN polymarket.btc_paper_experiments e
                  ON e.process_id = p.process_id
                 AND e.status = 'running'
                 AND e.stopped_at IS NULL
                WHERE p.process_type = 'btc_5m'
                  AND p.process_scope = 'realtime_paper'
                  AND (
                    (p.enabled AND p.status IN ('starting','running','stopping') AND p.stopped_at IS NULL)
                    OR e.experiment_id IS NOT NULL
                  )
                "#,
            )
            .fetch_one(&self.pool)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
            if durable_claims > 0 {
                return Err(HttpError::conflict(
                    "durable BTC realtime-paper state exists but cannot be reattached exactly; database state was left unchanged",
                ));
            }
            return Ok(Vec::new());
        }

        let mut resumed = Vec::with_capacity(candidates.len());
        for process_id in candidates {
            self.ensure_start_slot_available(process_id).await?;
            resumed.push(self.resume_durable_process_locked(process_id).await?);
        }
        Ok(resumed)
    }

    async fn resume_durable_process_locked(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcess, HttpError> {
        let process = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        let prepared = self.prepare_resume_definition(&process)?;
        let (config_hash, frozen_process_config_value) = sqlx::query_as::<
            _,
            (String, serde_json::Value),
        >(
            r#"
                SELECT config_hash, config
                FROM polymarket.btc_paper_experiments
                WHERE experiment_id = $1
                  AND name = $2
                  AND process_id = $3
                  AND status = 'running'
                  AND stopped_at IS NULL
                "#,
        )
        .bind(prepared.experiment_id)
        .bind(&prepared.experiment_key)
        .bind(process_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| HttpError::internal(error.to_string()))?
        .ok_or_else(|| {
            HttpError::conflict("durable BTC experiment disappeared before runtime reattachment")
        })?;
        let current_frozen_process_config =
            serde_json::to_value(&prepared.frozen_process_config)
                .map_err(|error| HttpError::internal(error.to_string()))?;
        if resume_process_contract_projection(current_frozen_process_config)
            != resume_process_contract_projection(frozen_process_config_value.clone())
        {
            return Err(HttpError::conflict(
                "durable BTC experiment parameters changed and cannot be resumed by this process definition",
            ));
        }
        let PreparedBtcStartDefinition {
            experiment_id,
            experiment_key,
            preregistration_sha256,
            strategy,
            entry_admission,
            runtime: runtime_config,
            paper_venue: paper_venue_config,
            paper_stress_previews,
            frozen_process_config: _,
            config_hash: current_config_hash,
        } = prepared;
        self.record_event(
            process_id,
            "info",
            "btc_runtime_resuming",
            "BTC realtime-paper runtime resume accepted",
            serde_json::json!({
                "experiment_id": experiment_id,
                "experiment_key": &experiment_key,
                "preregistration_sha256": &preregistration_sha256,
                "config_hash": &config_hash,
                "current_definition_config_hash": &current_config_hash,
                "compiled_source_identity": COMPILED_SOURCE_IDENTITY,
            }),
        )
        .await;

        let startup_result: Result<BtcPlaybookRuntimeHandle> = async {
            let (state, books) = self
                .ensure_shared_runtime(&runtime_config)
                .await
                .map_err(|error| anyhow::anyhow!("{error:?}"))?;
            let paper_venue = BtcPaperVenue::new_with_reference_execution_guard(
                books.clone(),
                paper_venue_config,
                strategy.max_depth_participation,
                process_id,
                chrono::Duration::milliseconds(strategy.max_reference_age_ms),
            )?;
            let experiment = Arc::new(BtcPaperExperimentRunner::new(
                self.repository.clone(),
                self.store.clone(),
                paper_venue,
                BtcPaperExperimentConfig {
                    experiment_id,
                    experiment_name: experiment_key.clone(),
                    process_id,
                    config_hash: config_hash.clone(),
                    frozen_process_config: frozen_process_config_value,
                    strategy,
                    entry_admission,
                    execution_enabled: true,
                    paper_stress_previews,
                },
            )?);
            experiment
                .resume()
                .await
                .context("failed to reattach immutable BTC experiment before feed resume")?;
            BtcPlaybookRuntimeHandle::start(runtime_config, experiment, state)
        }
        .await;
        let runtime = match startup_result {
            Ok(runtime) => runtime,
            Err(resume_error) => {
                return Err(HttpError::internal(format!(
                    "BTC durable runtime resume failed without mutating lifecycle state: {resume_error:#}"
                )));
            }
        };
        self.active_playbooks.lock().await.insert(
            process_id,
            ActiveBtcPlaybook {
                process_id,
                experiment_id,
                experiment_key: experiment_key.clone(),
                config_hash: config_hash.clone(),
                runtime,
            },
        );
        self.record_event(
            process_id,
            "info",
            "btc_runtime_resumed",
            "BTC realtime-paper runtime resumed after service restart",
            serde_json::json!({
                "experiment_id": experiment_id,
                "experiment_key": &experiment_key,
                "config_hash": &config_hash,
                "current_definition_config_hash": &current_config_hash,
                "compiled_source_identity": COMPILED_SOURCE_IDENTITY,
            }),
        )
        .await;
        info!(
            process_id = %process_id,
            experiment_id = %experiment_id,
            experiment_key = %experiment_key,
            config_hash = %config_hash,
            "durable BTC realtime-paper runtime resumed"
        );
        Ok(process)
    }

    async fn start_process_locked(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcess, HttpError> {
        self.ensure_start_slot_available(process_id).await?;
        let process = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        let PreparedBtcStartDefinition {
            experiment_id,
            experiment_key,
            preregistration_sha256,
            strategy,
            entry_admission,
            runtime: runtime_config,
            paper_venue: paper_venue_config,
            paper_stress_previews,
            frozen_process_config,
            config_hash,
        } = self.prepare_start_definition(&process)?;
        preflight_btc_experiment_identity(&self.pool, process_id, &experiment_key, experiment_id)
            .await
            .map_err(|error| HttpError::conflict(error.to_string()))?;
        let frozen_process_config_value = serde_json::to_value(&frozen_process_config)
            .map_err(|error| HttpError::internal(error.to_string()))?;

        let start_pending = PendingBtcTerminal {
            process_id,
            experiment_id,
            experiment_key: experiment_key.clone(),
            config_hash: config_hash.clone(),
            terminal_status: "failed".to_string(),
            terminal_reason: "btc_start_transition_interrupted".to_string(),
            allow_missing_experiment: true,
        };
        self.terminal_pending
            .lock()
            .await
            .insert(process_id, start_pending.clone());
        match self
            .store
            .update_trading_process_status(process_id, "starting", true, None)
            .await
        {
            Ok(Some(_)) => {}
            Ok(None) => {
                let reason = "BTC process disappeared while entering starting state".to_string();
                return Err(self
                    .terminalize_start_failure_locked(start_pending, reason)
                    .await);
            }
            Err(error) => {
                let reason = format!("failed to mark BTC process starting: {error:#}");
                return Err(self
                    .terminalize_start_failure_locked(start_pending, reason)
                    .await);
            }
        }
        self.record_event(
            process_id,
            "info",
            "btc_runtime_starting",
            "BTC realtime-paper runtime start accepted",
            serde_json::json!({
                "experiment_id": experiment_id,
                "experiment_key": &experiment_key,
                "preregistration_sha256": &preregistration_sha256,
            }),
        )
        .await;

        let startup_result: Result<BtcPlaybookRuntimeHandle> = async {
            let (state, books) = self
                .ensure_shared_runtime(&runtime_config)
                .await
                .map_err(|error| anyhow::anyhow!("{error:?}"))?;
            let paper_venue = BtcPaperVenue::new_with_reference_execution_guard(
                books.clone(),
                paper_venue_config,
                strategy.max_depth_participation,
                process_id,
                chrono::Duration::milliseconds(strategy.max_reference_age_ms),
            )?;
            let experiment = Arc::new(BtcPaperExperimentRunner::new(
                self.repository.clone(),
                self.store.clone(),
                paper_venue,
                BtcPaperExperimentConfig {
                    experiment_id,
                    experiment_name: experiment_key.clone(),
                    process_id,
                    config_hash: config_hash.clone(),
                    frozen_process_config: frozen_process_config_value,
                    strategy: strategy.clone(),
                    entry_admission: entry_admission.clone(),
                    execution_enabled: true,
                    paper_stress_previews: paper_stress_previews.clone(),
                },
            )?);
            experiment
                .initialize()
                .await
                .context("failed to initialize immutable BTC experiment before feed startup")?;
            BtcPlaybookRuntimeHandle::start(runtime_config, experiment, state)
        }
        .await;

        let runtime = match startup_result {
            Ok(runtime) => runtime,
            Err(startup_error) => {
                let reason = format!("btc_startup_failed: {startup_error:#}");
                return Err(self
                    .terminalize_start_failure_locked(start_pending, reason)
                    .await);
            }
        };

        let running_process = match self
            .store
            .update_trading_process_status(process_id, "running", true, None)
            .await
        {
            Ok(Some(process)) => process,
            Ok(None) => {
                let reason = "BTC process disappeared while runtime was starting".to_string();
                let _ =
                    tokio::time::timeout(BTC_RUNTIME_SHUTDOWN_TIMEOUT, runtime.shutdown()).await;
                return Err(self
                    .terminalize_start_failure_locked(start_pending, reason)
                    .await);
            }
            Err(update_error) => {
                let reason = format!("failed to mark BTC process running: {update_error:#}");
                let _ =
                    tokio::time::timeout(BTC_RUNTIME_SHUTDOWN_TIMEOUT, runtime.shutdown()).await;
                return Err(self
                    .terminalize_start_failure_locked(start_pending, reason)
                    .await);
            }
        };

        self.active_playbooks.lock().await.insert(
            process_id,
            ActiveBtcPlaybook {
                process_id,
                experiment_id,
                experiment_key: experiment_key.clone(),
                config_hash: config_hash.clone(),
                runtime,
            },
        );
        let mut pending_guard = self.terminal_pending.lock().await;
        if pending_guard
            .get(&process_id)
            .is_some_and(|pending| pending.experiment_id == experiment_id)
        {
            pending_guard.remove(&process_id);
        }
        drop(pending_guard);
        self.record_event(
            process_id,
            "info",
            "btc_runtime_started",
            "BTC realtime-paper runtime is running",
            serde_json::json!({
                "experiment_id": experiment_id,
                "experiment_key": &experiment_key,
                "config_hash": &config_hash,
            }),
        )
        .await;
        info!(
            process_id = %process_id,
            experiment_id = %experiment_id,
            experiment_key = %experiment_key,
            config_hash = %config_hash,
            compiled_source_identity = COMPILED_SOURCE_IDENTITY,
            "API-owned BTC realtime-paper runtime started"
        );
        Ok(running_process)
    }

    async fn terminalize_start_failure_locked(
        &self,
        mut pending: PendingBtcTerminal,
        reason: String,
    ) -> HttpError {
        pending.terminal_status = "failed".to_string();
        pending.terminal_reason = reason.clone();
        self.terminal_pending
            .lock()
            .await
            .insert(pending.process_id, pending.clone());
        self.record_event(
            pending.process_id,
            "error",
            "btc_runtime_start_failed",
            &reason,
            serde_json::json!({
                "experiment_id": pending.experiment_id,
                "experiment_key": pending.experiment_key,
                "config_hash": pending.config_hash,
            }),
        )
        .await;
        let result = self.finalize_pending_locked(pending).await;
        self.shutdown_shared_runtime_if_idle().await;
        match result {
            Ok(_) => HttpError::internal(reason),
            Err(persistence_error) => persistence_error,
        }
    }

    async fn stop_process(
        &self,
        process_id: uuid::Uuid,
        reason: &str,
    ) -> Result<TradingProcess, HttpError> {
        self.stop_process_for_generation(process_id, None, reason, false)
            .await
    }

    async fn stop_process_for_generation(
        &self,
        process_id: uuid::Uuid,
        expected_experiment_id: Option<uuid::Uuid>,
        reason: &str,
        runtime_failed: bool,
    ) -> Result<TradingProcess, HttpError> {
        let transition_guard = self.transition.clone().lock_owned().await;
        let manager = self.clone();
        let reason = reason.to_string();
        tokio::spawn(async move {
            let _transition_guard = transition_guard;
            manager
                .stop_process_locked(process_id, expected_experiment_id, &reason, runtime_failed)
                .await
        })
        .await
        .map_err(|error| HttpError::internal(format!("BTC stop transition task failed: {error}")))?
    }

    async fn stop_process_locked(
        &self,
        process_id: uuid::Uuid,
        expected_experiment_id: Option<uuid::Uuid>,
        reason: &str,
        runtime_failed: bool,
    ) -> Result<TradingProcess, HttpError> {
        let process = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        if !Self::is_managed_process(&process) {
            return Err(HttpError::bad_request(
                "process is not a managed BTC realtime-paper process",
            ));
        }

        let pending = { self.terminal_pending.lock().await.get(&process_id).cloned() };
        if let Some(pending) = pending {
            if expected_experiment_id.is_some_and(|expected| expected != pending.experiment_id) {
                return Ok(process);
            }
            return self.finalize_pending_locked(pending).await;
        }
        let active_identity = self
            .active_playbooks
            .lock()
            .await
            .get(&process_id)
            .map(|active| (active.experiment_id, active.experiment_key.clone()));
        let Some((active_experiment_id, active_experiment_key)) = active_identity else {
            if process.enabled
                || matches!(process.status.as_str(), "starting" | "running" | "stopping")
            {
                return Err(HttpError::conflict(
                    "BTC process claims to be active but this service owns no runtime handle",
                ));
            }
            return Ok(process);
        };
        if expected_experiment_id.is_some_and(|expected| expected != active_experiment_id) {
            return Ok(process);
        }
        self.store
            .update_trading_process_status(process_id, "stopping", true, None)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        self.record_event(
            process_id,
            "info",
            "btc_runtime_stopping",
            "BTC realtime-paper runtime stop accepted",
            serde_json::json!({
                "experiment_id": active_experiment_id,
                "experiment_key": active_experiment_key,
                "reason": reason,
            }),
        )
        .await;

        let active = self
            .active_playbooks
            .lock()
            .await
            .remove(&process_id)
            .expect("active BTC playbook exists while lifecycle transition is held");
        let provisional_pending = PendingBtcTerminal {
            process_id: active.process_id,
            experiment_id: active.experiment_id,
            experiment_key: active.experiment_key.clone(),
            config_hash: active.config_hash.clone(),
            terminal_status: "failed".to_string(),
            terminal_reason: format!("{reason}; stop_transition_interrupted"),
            allow_missing_experiment: false,
        };
        self.terminal_pending
            .lock()
            .await
            .insert(process_id, provisional_pending);
        let shutdown_result =
            tokio::time::timeout(BTC_RUNTIME_SHUTDOWN_TIMEOUT, active.runtime.shutdown()).await;
        let shutdown_failure = match shutdown_result {
            Ok(Ok(())) => None,
            Ok(Err(shutdown_error)) => {
                Some(format!("btc_runtime_shutdown_failed: {shutdown_error:#}"))
            }
            Err(_) => Some(format!(
                "btc_runtime_shutdown_timeout_after_{}s",
                BTC_RUNTIME_SHUTDOWN_TIMEOUT.as_secs()
            )),
        };
        let terminal_status = if runtime_failed || shutdown_failure.is_some() {
            "failed"
        } else {
            "stopped"
        };
        let terminal_reason = shutdown_failure
            .map(|failure| format!("{reason}; {failure}"))
            .unwrap_or_else(|| reason.to_string());
        let pending = PendingBtcTerminal {
            process_id: active.process_id,
            experiment_id: active.experiment_id,
            experiment_key: active.experiment_key,
            config_hash: active.config_hash,
            terminal_status: terminal_status.to_string(),
            terminal_reason,
            allow_missing_experiment: false,
        };
        self.terminal_pending
            .lock()
            .await
            .insert(process_id, pending.clone());
        let result = self.finalize_pending_locked(pending).await;
        self.shutdown_shared_runtime_if_idle().await;
        result
    }

    async fn finalize_pending_locked(
        &self,
        pending: PendingBtcTerminal,
    ) -> Result<TradingProcess, HttpError> {
        if let Err(terminal_error) = mark_btc_cohort_terminal(
            &self.pool,
            pending.experiment_id,
            pending.process_id,
            &pending.terminal_status,
            &pending.terminal_reason,
            pending.allow_missing_experiment,
        )
        .await
        {
            let persistence_reason = format!(
                "{}; terminal_persistence_pending: {terminal_error:#}",
                pending.terminal_reason
            );
            self.record_event(
                pending.process_id,
                "error",
                "btc_runtime_terminal_persistence_pending",
                &persistence_reason,
                serde_json::json!({
                    "experiment_id": pending.experiment_id,
                    "experiment_key": pending.experiment_key,
                    "config_hash": pending.config_hash,
                    "desired_terminal_status": pending.terminal_status,
                }),
            )
            .await;
            return Err(HttpError::internal(persistence_reason));
        }
        let mut pending_guard = self.terminal_pending.lock().await;
        if pending_guard
            .get(&pending.process_id)
            .is_some_and(|current| current.experiment_id == pending.experiment_id)
        {
            pending_guard.remove(&pending.process_id);
        }
        drop(pending_guard);
        self.record_event(
            pending.process_id,
            if pending.terminal_status == "stopped" {
                "info"
            } else {
                "error"
            },
            if pending.terminal_status == "stopped" {
                "btc_runtime_stopped"
            } else {
                "btc_runtime_stop_failed"
            },
            &pending.terminal_reason,
            serde_json::json!({
                "experiment_id": pending.experiment_id,
                "experiment_key": pending.experiment_key,
                "config_hash": pending.config_hash,
            }),
        )
        .await;
        self.store
            .get_trading_process(pending.process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))
    }

    async fn quiesce_for_shutdown(&self, reason: &str) -> Result<(), HttpError> {
        self.begin_shutdown();
        // Wait behind any already-admitted lifecycle transition, then inspect
        // state while owning the same gate so no start can publish afterward.
        let _transition_guard = self.transition.lock().await;
        let pending = self.terminal_pending.lock().await.values().next().cloned();
        if let Some(pending) = pending {
            return Err(HttpError::conflict(format!(
                "BTC experiment {} has a pending API lifecycle transition; service shutdown left durable state unchanged",
                pending.experiment_key
            )));
        }
        let active_runs = self
            .active_playbooks
            .lock()
            .await
            .drain()
            .map(|(_, active)| active)
            .collect::<Vec<_>>();
        let mut failures = Vec::new();
        for active in active_runs {
            let process_id = active.process_id;
            let experiment_id = active.experiment_id;
            let experiment_key = active.experiment_key.clone();
            let config_hash = active.config_hash.clone();
            let shutdown_result =
                tokio::time::timeout(BTC_RUNTIME_SHUTDOWN_TIMEOUT, active.runtime.shutdown()).await;
            if let Some(shutdown_failure) = match shutdown_result {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(format!("btc_runtime_shutdown_failed: {error:#}")),
                Err(_) => Some(format!(
                    "btc_runtime_shutdown_timeout_after_{}s",
                    BTC_RUNTIME_SHUTDOWN_TIMEOUT.as_secs()
                )),
            } {
                failures.push(format!("{process_id}: {shutdown_failure}"));
                continue;
            }
            self.record_event(
                process_id,
                "info",
                "btc_runtime_suspended",
                "BTC realtime-paper runtime suspended for service shutdown",
                serde_json::json!({
                    "experiment_id": experiment_id,
                    "experiment_key": experiment_key,
                    "config_hash": config_hash,
                    "reason": reason,
                    "resume_on_service_restart": true,
                }),
            )
            .await;
            info!(
                process_id = %process_id,
                experiment_id = %experiment_id,
                "BTC realtime-paper runtime suspended with durable resume intent"
            );
        }
        let shared_runtime = { self.shared_runtime.lock().await.take() };
        if let Some(shared) = shared_runtime {
            let shutdown_result =
                tokio::time::timeout(BTC_RUNTIME_SHUTDOWN_TIMEOUT, shared.runtime.shutdown()).await;
            if let Some(shutdown_failure) = match shutdown_result {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(format!("shared_btc_runtime_shutdown_failed: {error:#}")),
                Err(_) => Some(format!(
                    "shared_btc_runtime_shutdown_timeout_after_{}s",
                    BTC_RUNTIME_SHUTDOWN_TIMEOUT.as_secs()
                )),
            } {
                failures.push(shutdown_failure);
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(HttpError::internal(format!(
                "BTC runtimes did not quiesce cleanly during service shutdown; durable lifecycle state was left unchanged: {}",
                failures.join("; ")
            )))
        }
    }

    fn begin_shutdown(&self) {
        self.shutting_down.store(true, Ordering::Release);
    }

    async fn grafana_countdown_snapshot(
        &self,
        observed_at: chrono::DateTime<Utc>,
    ) -> CountdownSnapshot {
        let active_processes = self.active_playbooks.lock().await.len();
        let state = self
            .shared_runtime
            .lock()
            .await
            .as_ref()
            .map(|shared| shared.runtime.shared_state());
        let current_market = match state {
            Some(state) => state.read().await.current_market.clone(),
            None => None,
        };
        let markets = current_market
            .map(|market| vec![market; active_processes])
            .unwrap_or_default();
        CountdownSnapshot::resolve(observed_at, active_processes, markets)
    }

    async fn reconcile_failed_runtime(&self) {
        let pending = self
            .terminal_pending
            .lock()
            .await
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for pending in pending {
            if let Err(stop_error) = self
                .stop_process_for_generation(
                    pending.process_id,
                    Some(pending.experiment_id),
                    &pending.terminal_reason,
                    pending.terminal_status == "failed",
                )
                .await
            {
                error!(
                    error = ?stop_error,
                    process_id = %pending.process_id,
                    experiment_id = %pending.experiment_id,
                    "failed to retry pending BTC terminal transition"
                );
            }
        }
        let process_ids = self
            .active_playbooks
            .lock()
            .await
            .keys()
            .copied()
            .collect::<Vec<_>>();
        let shared_status_inputs = self
            .shared_runtime
            .lock()
            .await
            .as_ref()
            .map(|shared| shared.runtime.status_inputs());
        let shared_status = match shared_status_inputs {
            Some((state, metrics, config, running)) => {
                Some(runtime_status_from_inputs(state, metrics, config, running).await)
            }
            None => None,
        };
        let shared_running = shared_status.as_ref().is_some_and(|status| status.running);
        if !process_ids.is_empty() && !shared_running {
            let failure_reason = format!(
                "btc_shared_market_data_runtime_failed: {}",
                shared_status
                    .as_ref()
                    .and_then(|status| status.metrics.last_error.as_deref())
                    .unwrap_or("shared market-data runtime handle is unavailable")
            );
            for process_id in process_ids {
                let experiment_id = self
                    .active_playbooks
                    .lock()
                    .await
                    .get(&process_id)
                    .map(|active| active.experiment_id);
                let Some(experiment_id) = experiment_id else {
                    continue;
                };
                if let Err(stop_error) = self
                    .stop_process_for_generation(
                        process_id,
                        Some(experiment_id),
                        &failure_reason,
                        true,
                    )
                    .await
                {
                    error!(
                        error = ?stop_error,
                        process_id = %process_id,
                        experiment_id = %experiment_id,
                        "failed to terminalize playbook after shared market-data failure"
                    );
                }
            }
            return;
        }
        for process_id in process_ids {
            let status_input = {
                let active_guard = self.active_playbooks.lock().await;
                let Some(active) = active_guard.get(&process_id) else {
                    continue;
                };
                (active.experiment_id, active.runtime.status_inputs())
            };
            let (experiment_id, (state, metrics, config, running)) = status_input;
            let status = runtime_status_from_inputs(state, metrics, config, running).await;
            let runtime_running = status.running;
            let last_error = status.metrics.last_error;
            if runtime_running {
                match self
                    .store
                    .heartbeat_active_trading_process(process_id)
                    .await
                {
                    Ok(true) => continue,
                    Ok(false) => {}
                    Err(heartbeat_error) => {
                        warn!(
                            error = %heartbeat_error,
                            process_id = %process_id,
                            experiment_id = %experiment_id,
                            "failed to persist BTC manager heartbeat"
                        );
                        continue;
                    }
                }
            }
            let reason = last_error.unwrap_or_else(|| {
                if runtime_running {
                    "manager heartbeat rejected because the durable process is not active"
                        .to_string()
                } else {
                    "BTC runtime child task stopped unexpectedly".to_string()
                }
            });
            let terminal_reason = format!("btc_runtime_failed: {reason}");
            if let Err(stop_error) = self
                .stop_process_for_generation(
                    process_id,
                    Some(experiment_id),
                    &terminal_reason,
                    true,
                )
                .await
            {
                error!(
                    error = ?stop_error,
                    process_id = %process_id,
                    experiment_id = %experiment_id,
                    "failed to terminalize BTC runtime child failure"
                );
            }
        }
    }

    async fn runtime_status(&self) -> serde_json::Value {
        let process_ids = self
            .active_playbooks
            .lock()
            .await
            .keys()
            .copied()
            .collect::<Vec<_>>();
        let mut processes = Vec::with_capacity(process_ids.len());
        for process_id in process_ids {
            processes.push(self.runtime_status_for_process(process_id).await);
        }
        let shared_inputs = self
            .shared_runtime
            .lock()
            .await
            .as_ref()
            .map(|shared| shared.runtime.status_inputs());
        let shared_market_data = match shared_inputs {
            Some((state, metrics, config, running)) => serde_json::to_value(
                runtime_status_from_inputs(state, metrics, config, running).await,
            )
            .unwrap_or_else(|_| serde_json::json!({"running": false})),
            None => serde_json::json!({
                "enabled": self.config.btc.realtime_enabled,
                "running": false,
                "readiness": {"ready": false, "reasons": ["shared_market_data_inactive"]}
            }),
        };
        serde_json::json!({
            "capability_enabled": self.config.btc.realtime_enabled,
            "active": !processes.is_empty(),
            "active_process_count": processes.len(),
            "shared_market_data": shared_market_data,
            "processes": processes,
        })
    }

    async fn runtime_status_for_process(&self, process_id: uuid::Uuid) -> serde_json::Value {
        let active = self
            .active_playbooks
            .lock()
            .await
            .get(&process_id)
            .map(|active| {
                (
                    active.process_id,
                    active.experiment_id,
                    active.experiment_key.clone(),
                    active.config_hash.clone(),
                    active.runtime.status_inputs(),
                )
            });
        if let Some((process_id, experiment_id, experiment_key, config_hash, inputs)) = active {
            let (state, metrics, config, running) = inputs;
            let runtime = runtime_status_from_inputs(state, metrics, config, running).await;
            return serde_json::json!({
                "capability_enabled": self.config.btc.realtime_enabled,
                "active": true,
                "process_id": process_id,
                "experiment_id": experiment_id,
                "experiment_key": experiment_key,
                "config_hash": config_hash,
                "runtime": runtime,
            });
        }
        let pending = self.terminal_pending.lock().await.get(&process_id).cloned();
        if let Some(pending) = pending {
            return serde_json::json!({
                "capability_enabled": self.config.btc.realtime_enabled,
                "active": false,
                "running": false,
                "lifecycle_state": "terminal_pending",
                "process_id": pending.process_id,
                "experiment_id": pending.experiment_id,
                "experiment_key": pending.experiment_key,
                "config_hash": pending.config_hash,
                "desired_terminal_status": pending.terminal_status,
                "terminal_reason": pending.terminal_reason,
                "readiness": {"ready": false, "reasons": ["terminal_persistence_pending"]}
            });
        }
        serde_json::json!({
            "capability_enabled": self.config.btc.realtime_enabled,
            "active": false,
            "running": false,
            "process_id": process_id,
            "readiness": {"ready": false, "reasons": ["trading_process_inactive"]}
        })
    }

    async fn paper_experiment_status(&self) -> Result<serde_json::Value, HttpError> {
        let process_ids = self
            .active_playbooks
            .lock()
            .await
            .keys()
            .copied()
            .collect::<Vec<_>>();
        let mut processes = Vec::with_capacity(process_ids.len());
        for process_id in process_ids {
            processes.push(self.paper_experiment_status_for_process(process_id).await?);
        }
        Ok(serde_json::json!({
            "configured": true,
            "active": !processes.is_empty(),
            "active_process_count": processes.len(),
            "processes": processes,
        }))
    }

    async fn paper_experiment_status_for_process(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<serde_json::Value, HttpError> {
        let active = self
            .active_playbooks
            .lock()
            .await
            .get(&process_id)
            .map(|active| {
                (
                    active.process_id,
                    active.experiment_id,
                    active.experiment_key.clone(),
                )
            });
        let Some((active_process_id, experiment_id, experiment_key)) = active else {
            let pending = self.terminal_pending.lock().await.get(&process_id).cloned();
            if let Some(pending) = pending {
                return Ok(serde_json::json!({
                    "configured": true,
                    "active": false,
                    "status": "terminal_pending",
                    "process_id": pending.process_id,
                    "experiment_id": pending.experiment_id,
                    "experiment_key": pending.experiment_key,
                    "desired_terminal_status": pending.terminal_status,
                    "terminal_reason": pending.terminal_reason,
                }));
            }
            return Ok(serde_json::json!({
                "configured": true,
                "active": false,
                "status": "inactive",
                "process_id": process_id,
            }));
        };
        let experiment = self
            .repository
            .paper_experiment_status(experiment_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(serde_json::json!({
            "configured": true,
            "active": true,
            "process_id": active_process_id,
            "experiment_id": experiment_id,
            "experiment_key": experiment_key,
            "experiment": experiment,
        }))
    }
}

#[derive(Debug, Default, Clone)]
struct RuntimeMetrics {
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
            "uptime_secs": (Utc::now() - self.started_at).num_seconds()
        })
    }
}

struct RuntimeControl {
    store: Store,
    ingestion: IngestionRepository,
    live_venue: Option<Arc<dyn ExecutionVenue>>,
    metrics: Arc<Mutex<RuntimeMetrics>>,
    btc_manager: Option<BtcProcessManager>,
}

impl RuntimeControl {
    fn live_venue(&self) -> Result<Arc<dyn ExecutionVenue>, HttpError> {
        self.live_venue
            .clone()
            .ok_or_else(|| HttpError::bad_request("live execution is not configured"))
    }
}

#[async_trait]
impl ControlApi for RuntimeControl {
    async fn health(&self) -> Result<HealthResponse, HttpError> {
        self.store
            .healthcheck()
            .await
            .map_err(|error| HttpError::internal(format!("database unhealthy: {error}")))?;
        Ok(HealthResponse {
            service: "polymarket-bot".to_string(),
            status: HealthStatus::Ok,
            checked_at: Utc::now(),
        })
    }

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

    async fn btc_realtime_status(&self) -> Result<serde_json::Value, HttpError> {
        let Some(manager) = &self.btc_manager else {
            return Ok(serde_json::json!({
                "capability_enabled": false,
                "active": false,
                "running": false,
                "readiness": {"ready": false, "reasons": ["btc_realtime_capability_disabled"]}
            }));
        };
        Ok(manager.runtime_status().await)
    }

    async fn btc_paper_experiment_status(&self) -> Result<serde_json::Value, HttpError> {
        let Some(manager) = &self.btc_manager else {
            return Ok(serde_json::json!({"configured": false, "status": "disabled"}));
        };
        manager.paper_experiment_status().await
    }

    async fn enqueue_ingestion_backfill(
        &self,
        request: IngestionBackfillRequest,
    ) -> Result<IngestionBackfillEnqueueResponse, HttpError> {
        let request = request
            .validate()
            .map_err(|error| HttpError::bad_request(error.to_string()))?;
        let job = self.ingestion.enqueue(&request).await.map_err(|error| {
            let message = error.to_string();
            if message.contains("idempotency key") {
                HttpError::conflict(message)
            } else {
                HttpError::internal(message)
            }
        })?;
        Ok(job.into())
    }

    async fn list_ingestion_backfills(
        &self,
        request: control_http::ListIngestionBackfillsRequest,
    ) -> Result<IngestionBackfillJobsResponse, HttpError> {
        let jobs = self
            .ingestion
            .list(request.limit.unwrap_or(50).clamp(1, 200))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(IngestionBackfillJobsResponse { jobs })
    }

    async fn get_ingestion_backfill(
        &self,
        job_id: uuid::Uuid,
    ) -> Result<IngestionBackfillJobResponse, HttpError> {
        let job = self
            .ingestion
            .get(job_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("backfill job not found"))?;
        Ok(IngestionBackfillJobResponse { job })
    }

    async fn list_ingestion_backfill_events(
        &self,
        job_id: uuid::Uuid,
        request: control_http::ListIngestionBackfillEventsRequest,
    ) -> Result<IngestionBackfillEventsResponse, HttpError> {
        if self
            .ingestion
            .get(job_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .is_none()
        {
            return Err(HttpError::not_found("backfill job not found"));
        }
        let events = self
            .ingestion
            .list_events(job_id, request.limit.unwrap_or(100).clamp(1, 500))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(IngestionBackfillEventsResponse { job_id, events })
    }

    async fn cancel_ingestion_backfill(
        &self,
        job_id: uuid::Uuid,
    ) -> Result<IngestionBackfillCancelResponse, HttpError> {
        let job = self
            .ingestion
            .request_cancel(job_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("backfill job not found"))?;
        Ok(IngestionBackfillCancelResponse {
            job_id,
            status: job.status,
            cancel_requested: matches!(
                job.status,
                polymarket_bot::ingestion::job::BackfillJobStatus::CancelRequested
                    | polymarket_bot::ingestion::job::BackfillJobStatus::Cancelled
            ),
        })
    }

    async fn ingestion_training_readiness(
        &self,
        request: control_http::IngestionReadinessRequest,
    ) -> Result<polymarket_bot::ingestion::job::TrainingReadiness, HttpError> {
        self.ingestion
            .training_readiness(request.range_start, request.range_end)
            .await
            .map_err(|error| HttpError::bad_request(error.to_string()))
    }

    async fn live_status(&self) -> Result<LiveVenueStatus, HttpError> {
        self.live_venue()?
            .live_status()
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_identity_diagnostics(&self) -> Result<LiveIdentityDiagnostics, HttpError> {
        self.live_venue()?
            .live_identity_diagnostics()
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_wallet_address_diagnostics(
        &self,
        candidate_addresses: Vec<String>,
    ) -> Result<LiveWalletAddressDiagnostics, HttpError> {
        self.live_venue()?
            .live_wallet_address_diagnostics(candidate_addresses)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_order_dry_run(
        &self,
        request: LiveOrderDryRunRequest,
    ) -> Result<LiveOrderDryRunDiagnostics, HttpError> {
        self.live_venue()?
            .live_order_dry_run(request)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_poly1271_funder_probe(
        &self,
        request: LivePoly1271FunderProbeRequest,
    ) -> Result<LivePoly1271FunderProbeResponse, HttpError> {
        self.live_venue()?
            .live_poly1271_funder_probe(request)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_halt(&self) -> Result<serde_json::Value, HttpError> {
        let live = self.live_venue()?;
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
            .live_venue()?
            .reconcile()
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        serde_json::to_value(report).map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_account_reconcile(
        &self,
        request: control_http::AccountReconcileRequest,
    ) -> Result<control_http::AccountReconcileReport, HttpError> {
        self.live_venue()?
            .live_account_reconcile(request)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_set_entries_enabled(&self, enabled: bool) -> Result<LiveVenueStatus, HttpError> {
        self.live_venue()?
            .set_live_entries_enabled(enabled, (!enabled).then(|| "manual_disable".to_string()))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
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
        let process_scope = request.process_scope.trim();
        if !BtcProcessManager::is_managed_identity(process_type, process_scope) {
            return Err(HttpError::bad_request(
                "only btc_5m/realtime_paper process definitions are supported",
            ));
        }
        let status = request.status.trim();
        if status.is_empty() {
            return Err(HttpError::bad_request("trading process status is required"));
        }
        if request.enabled || matches!(status, "starting" | "running" | "stopping") {
            return Err(HttpError::bad_request(
                "BTC lifecycle cannot be activated through PUT; use the process /start endpoint",
            ));
        }
        if !matches!(status, "created" | "stopped" | "failed" | "completed") {
            return Err(HttpError::bad_request(
                "BTC definition status must be created, stopped, failed, or completed while inactive",
            ));
        }
        // Keep the lifecycle lock through the definition read and write. This
        // prevents /start from freezing the old definition while this request
        // concurrently replaces it.
        let lifecycle_guard = match &self.btc_manager {
            Some(manager) => Some(manager.transition.lock().await),
            None => None,
        };
        let manager = self.btc_manager.as_ref().ok_or_else(|| {
            HttpError::bad_request("BTC realtime-paper capability is disabled for this deployment")
        })?;
        let now = Utc::now();
        manager.validate_start_definition(&TradingProcess {
            process_id: uuid::Uuid::nil(),
            name: name.to_string(),
            process_type: BTC_PROCESS_TYPE.to_string(),
            process_scope: BTC_PROCESS_SCOPE.to_string(),
            process_key: Some(key.to_string()),
            status: status.to_string(),
            enabled: false,
            config: request.config.clone(),
            metadata: request.metadata.clone(),
            created_at: now,
            updated_at: now,
            started_at: None,
            stopped_at: Some(now),
            last_error: None,
        })?;
        let existing = self
            .store
            .get_btc_realtime_paper_process_by_key(key)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        if let Some(existing) = existing {
            if status != existing.status {
                return Err(HttpError::bad_request(
                    "BTC status is lifecycle-owned; stable-key PUT must preserve the current status",
                ));
            }
            let runtime_active = match &self.btc_manager {
                Some(manager) => manager
                    .active_playbooks
                    .lock()
                    .await
                    .contains_key(&existing.process_id),
                None => false,
            };
            let terminal_pending = match &self.btc_manager {
                Some(manager) => manager
                    .terminal_pending
                    .lock()
                    .await
                    .contains_key(&existing.process_id),
                None => false,
            };
            if runtime_active
                || terminal_pending
                || existing.enabled
                || matches!(
                    existing.status.as_str(),
                    "starting" | "running" | "stopping"
                )
            {
                return Err(HttpError::conflict(
                    "stop the BTC trading process through its /stop endpoint before reconfiguring it",
                ));
            }
        } else if status != "created" {
            return Err(HttpError::bad_request(
                "a new BTC process definition must be created with status=created",
            ));
        }
        let process = self
            .store
            .upsert_btc_realtime_paper_process_by_key(
                name,
                key,
                status,
                request.config,
                request.metadata,
            )
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        drop(lifecycle_guard);
        Ok(TradingProcessResponse { process })
    }

    async fn get_trading_process_status(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcessStatusResponse, HttpError> {
        let process = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        let mut status = self
            .store
            .trading_process_status(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        if BtcProcessManager::is_managed_process(&process) {
            if let Some(object) = status.as_object_mut() {
                object.insert(
                    "aggregate_scope".to_string(),
                    serde_json::Value::String("process_lifetime".to_string()),
                );
                let runtime = match &self.btc_manager {
                    Some(manager) => manager.runtime_status_for_process(process_id).await,
                    None => serde_json::json!({
                        "capability_enabled": false,
                        "active": false,
                        "running": false
                    }),
                };
                object.insert("btc_runtime".to_string(), runtime);
                if let Some(manager) = &self.btc_manager {
                    object.insert(
                        "active_experiment".to_string(),
                        manager
                            .paper_experiment_status_for_process(process_id)
                            .await?,
                    );
                }
            }
        }
        Ok(TradingProcessStatusResponse { process_id, status })
    }

    async fn preview_trading_process_start(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcessStartPreviewResponse, HttpError> {
        let current = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        if !BtcProcessManager::is_managed_process(&current) {
            return Err(HttpError::bad_request(
                "start preview is available only for managed BTC realtime-paper processes",
            ));
        }
        let manager = self.btc_manager.as_ref().ok_or_else(|| {
            HttpError::bad_request("BTC realtime-paper capability is disabled for this deployment")
        })?;
        manager.preview_start(process_id).await
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
        let lifecycle_guard = match &self.btc_manager {
            Some(manager) => Some(manager.transition.lock().await),
            None => None,
        };
        let current = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        if !BtcProcessManager::is_managed_process(&current) {
            return Err(HttpError::bad_request(
                "only managed BTC realtime-paper process definitions can be updated",
            ));
        }
        let runtime_active = match &self.btc_manager {
            Some(manager) => manager
                .active_playbooks
                .lock()
                .await
                .contains_key(&process_id),
            None => false,
        };
        let terminal_pending = match &self.btc_manager {
            Some(manager) => manager
                .terminal_pending
                .lock()
                .await
                .contains_key(&process_id),
            None => false,
        };
        if runtime_active
            || terminal_pending
            || current.enabled
            || matches!(current.status.as_str(), "starting" | "running" | "stopping")
        {
            return Err(HttpError::conflict(
                "stop the BTC trading process through its /stop endpoint before reconfiguring it",
            ));
        }
        if let Some(config) = &request.config {
            let manager = self.btc_manager.as_ref().ok_or_else(|| {
                HttpError::bad_request(
                    "BTC realtime-paper capability is disabled for this deployment",
                )
            })?;
            let mut candidate = current.clone();
            candidate.config = config.clone();
            manager.validate_start_definition(&candidate)?;
        }
        let process = self
            .store
            .update_btc_realtime_paper_process_definition(
                process_id,
                name,
                request.config,
                request.metadata,
            )
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        drop(lifecycle_guard);
        Ok(TradingProcessResponse { process })
    }

    async fn start_trading_process(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcessResponse, HttpError> {
        let current = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        if !BtcProcessManager::is_managed_process(&current) {
            return Err(HttpError::bad_request(
                "only managed BTC realtime-paper processes can be started",
            ));
        }
        let manager = self.btc_manager.as_ref().ok_or_else(|| {
            HttpError::bad_request("BTC realtime-paper capability is disabled for this deployment")
        })?;
        let process = manager.start_process(process_id).await?;
        Ok(TradingProcessResponse { process })
    }

    async fn stop_trading_process(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcessResponse, HttpError> {
        let current = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        if !BtcProcessManager::is_managed_process(&current) {
            return Err(HttpError::bad_request(
                "only managed BTC realtime-paper processes can be stopped",
            ));
        }
        let manager = self.btc_manager.as_ref().ok_or_else(|| {
            HttpError::bad_request("BTC realtime-paper capability is disabled for this deployment")
        })?;
        let process = manager.stop_process(process_id, "api_stop").await?;
        Ok(TradingProcessResponse { process })
    }
}

async fn run_grafana_live_countdown(
    manager: BtcProcessManager,
    publisher: GrafanaLivePublisher,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) {
    let mut ticker = tokio::time::interval(publisher.publish_interval());
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut last_publish_error: Option<String> = None;
    info!(
        channel = polymarket_bot::grafana_live::COUNTDOWN_CHANNEL,
        "Grafana Live BTC market countdown publisher started"
    );
    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    break;
                }
            }
            _ = ticker.tick() => {
                let snapshot = manager.grafana_countdown_snapshot(Utc::now()).await;
                match publisher.publish(&snapshot).await {
                    Ok(()) => {
                        if last_publish_error.take().is_some() {
                            info!("Grafana Live BTC market countdown publishing recovered");
                        }
                    }
                    Err(publish_error) => {
                        let message = format!("{publish_error:#}");
                        if last_publish_error.as_deref() != Some(message.as_str()) {
                            warn!(
                                error = %message,
                                "Grafana Live BTC market countdown publish failed; retrying"
                            );
                            last_publish_error = Some(message);
                        }
                    }
                }
            }
        }
    }
    info!("Grafana Live BTC market countdown publisher stopped");
}

#[tokio::main]
async fn main() -> Result<()> {
    install_tls_crypto_provider();
    init_tracing();

    let config = AppConfig::from_env()?;
    info!(
        service = "polymarket-bot",
        live_order_submit_enabled = config.live.order_submit_enabled,
        live_user_ws_enabled = config.live.user_ws_enabled,
        btc_realtime_enabled = config.btc.realtime_enabled,
        btc_paper_enabled = config.btc.paper_enabled,
        btc_clob_heartbeat_interval_secs = config.btc.data_source_heartbeat.clob_interval.as_secs(),
        btc_rtds_heartbeat_interval_secs = config.btc.data_source_heartbeat.rtds_interval.as_secs(),
        btc_binance_heartbeat_interval_secs =
            config.btc.data_source_heartbeat.binance_interval.as_secs(),
        grafana_live_enabled = config.grafana_live.enabled,
        compiled_source_identity = COMPILED_SOURCE_IDENTITY,
        "starting Polymarket bot"
    );

    let pool = PgPoolOptions::new()
        .max_connections(8)
        .connect(&config.postgres.database_url())
        .await
        .context("failed to connect Polymarket application to Postgres")?;
    let store = Store::from_pool(pool.clone());
    store.healthcheck().await?;
    store
        .insert_service_event(&ServiceEvent::new(
            "service_started",
            serde_json::json!({
                "execution_control": "trade_processes",
                "live_order_submit_enabled": config.live.order_submit_enabled,
                "live_user_ws_enabled": config.live.user_ws_enabled,
                "btc_realtime_enabled": config.btc.realtime_enabled,
                "btc_paper_enabled": config.btc.paper_enabled,
                "btc_clob_heartbeat_interval_secs": config.btc.data_source_heartbeat.clob_interval.as_secs(),
                "btc_rtds_heartbeat_interval_secs": config.btc.data_source_heartbeat.rtds_interval.as_secs(),
                "btc_binance_heartbeat_interval_secs": config.btc.data_source_heartbeat.binance_interval.as_secs(),
                "grafana_live_enabled": config.grafana_live.enabled,
                "compiled_source_identity": COMPILED_SOURCE_IDENTITY,
                "kafka_required": false
            }),
        ))
        .await?;

    let data_api = DataApiClient::new(config.data_api_base_url.clone());
    let live_venue: Option<Arc<dyn ExecutionVenue>> = if config.live.live_auth_available() {
        Some(Arc::new(LiveVenue::new(
            config.live.clone(),
            config.live.clob_api_base_url.clone(),
            store.clone(),
            data_api.clone(),
        )?))
    } else {
        None
    };
    let btc_manager = if config.btc.realtime_enabled {
        let repository = BtcRepository::from_pool(pool.clone());
        Some(BtcProcessManager::new(
            store.clone(),
            pool.clone(),
            repository,
            BtcProcessManagerConfig {
                btc: config.btc.clone(),
                gamma_base_url: config.gamma_base_url.clone(),
                clob_rest_base_url: config.clob_base_url.clone(),
                clob_ws_url: config.clob_ws_url.clone(),
            },
        ))
    } else {
        None
    };
    if let Some(manager) = &btc_manager {
        if let Err(error) = manager.resume_durable_processes().await {
            bail!(
                "failed to resume durable BTC realtime-paper state; database lifecycle state was left unchanged: {error:?}"
            );
        }
    }

    let (grafana_live_shutdown_tx, grafana_live_shutdown_rx) = tokio::sync::watch::channel(false);
    let grafana_live_task = if config.grafana_live.enabled {
        let manager = btc_manager
            .clone()
            .context("Grafana Live countdown requires the BTC process manager")?;
        let publisher = GrafanaLivePublisher::new(config.grafana_live.clone())?;
        Some(tokio::spawn(run_grafana_live_countdown(
            manager,
            publisher,
            grafana_live_shutdown_rx,
        )))
    } else {
        None
    };

    let metrics = RuntimeMetrics::new();
    let shared_metrics = Arc::new(Mutex::new(metrics.clone()));
    if config.http.enabled {
        let ingestion = IngestionRepository::from_pool(pool.clone());
        let control: control_http::SharedControlApi = Arc::new(RuntimeControl {
            store: store.clone(),
            ingestion,
            live_venue: live_venue.clone(),
            metrics: shared_metrics.clone(),
            btc_manager: btc_manager.clone(),
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
    let mut health_interval = tokio::time::interval(config.health_interval);
    health_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            biased;

            _ = &mut shutdown => {
                warn!("shutdown signal received");
                let _ = grafana_live_shutdown_tx.send(true);
                if let Some(manager) = &btc_manager {
                    if let Err(error) = manager.quiesce_for_shutdown("service_shutdown").await {
                        error!(
                            error = ?error,
                            "failed to stop active BTC runtime during service shutdown"
                        );
                    }
                }
                store.insert_service_event(&ServiceEvent::new("service_stopped", serde_json::json!({}))).await.ok();
                break;
            }

            _ = health_interval.tick() => {
                if let Some(manager) = &btc_manager {
                    manager.reconcile_failed_runtime().await;
                }
                if let Err(error) = store.healthcheck().await {
                    error!(error = %error, "database healthcheck failed");
                }
                info!(
                    target: "metrics",
                    uptime_secs = (Utc::now() - metrics.started_at).num_seconds(),
                    "polymarket bot liveness ok"
                );
                if let Ok(mut shared) = shared_metrics.lock() {
                    *shared = metrics.clone();
                }
            }

        }
    }

    if let Some(task) = grafana_live_task {
        if let Err(join_error) = task.await {
            warn!(error = %join_error, "Grafana Live countdown publisher task did not join cleanly");
        }
    }

    Ok(())
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

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    fn eligible_btc_process() -> TradingProcess {
        let now = Utc::now();
        TradingProcess {
            process_id: uuid::Uuid::new_v4(),
            name: "BTC realtime paper".to_string(),
            process_type: "btc_5m".to_string(),
            process_scope: "realtime_paper".to_string(),
            process_key: Some("btc-5m-realtime-paper".to_string()),
            status: "created".to_string(),
            enabled: false,
            config: TradingProcessConfig {
                execution: Some(ProcessExecutionConfig {
                    mode: Some("paper".to_string()),
                    execute_signals: true,
                    live_capital: false,
                    taker_fee_rate: None,
                }),
                ..TradingProcessConfig::default()
            },
            metadata: serde_json::json!({}),
            created_at: now,
            updated_at: now,
            started_at: None,
            stopped_at: None,
            last_error: None,
        }
    }

    fn prepared_btc_definition_with_default_runtime() -> PreparedBtcStartDefinition {
        prepare_btc_start_definition(ResolvedBtcProcessDefinition {
            control: BtcRealtimePaperControlConfig {
                schema_version: BTC_PROCESS_SCHEMA_VERSION.to_string(),
                next_experiment_key: "btc-5m-heartbeat-resume-test".to_string(),
                preregistration_sha256: "a".repeat(64),
                ..BtcRealtimePaperControlConfig::default()
            },
            strategy: BtcStrategyConfig::default(),
            entry_admission: None,
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        })
        .unwrap()
    }

    #[test]
    fn btc_run_identity_is_owned_by_process_control_config() {
        let control: BtcRealtimePaperControlConfig = serde_json::from_value(serde_json::json!({
            "schema_version": BTC_PROCESS_SCHEMA_VERSION,
            "next_experiment_key": "btc-5m-paper-20260713-c",
            "preregistration_sha256": "a".repeat(64),
        }))
        .unwrap();
        assert_eq!(control.schema_version, BTC_PROCESS_SCHEMA_VERSION);
        assert_eq!(control.next_experiment_key, "btc-5m-paper-20260713-c");
        assert_eq!(control.preregistration_sha256.len(), 64);
        assert_eq!(control.runtime.strategy_interval_ms, 1_000);
        assert_eq!(control.paper.arrival_latency_ms, 150);
        assert_eq!(control.paper.visible_depth_haircut, dec!(0.80));
        assert!(control.entry_admission.is_none());
    }

    #[test]
    fn btc_process_control_rejects_silently_ignored_fields() {
        let result = serde_json::from_value::<BtcRealtimePaperControlConfig>(serde_json::json!({
            "schema_version": BTC_PROCESS_SCHEMA_VERSION,
            "next_experiment_key": "btc-5m-paper-20260713-c",
            "preregistration_sha256": "a".repeat(64),
            "unknown_setting": true,
        }));
        assert!(result.is_err());

        let retired_ml =
            serde_json::from_value::<BtcRealtimePaperControlConfig>(serde_json::json!({
                    "schema_version": BTC_PROCESS_SCHEMA_VERSION,
                    "next_experiment_key": "btc-5m-paper-20260713-c",
                    "preregistration_sha256": "a".repeat(64),
                    "ml_shadow": {"enabled": true},
            }));
        assert!(retired_ml.is_err());

        for field in [
            "clob_heartbeat_interval",
            "rtds_heartbeat_interval",
            "binance_heartbeat_interval",
        ] {
            let mut process_control = serde_json::json!({
                "schema_version": BTC_PROCESS_SCHEMA_VERSION,
                "next_experiment_key": "btc-5m-paper-20260713-c",
                "preregistration_sha256": "a".repeat(64),
                "runtime": {},
            });
            process_control["runtime"][field] = serde_json::json!(5);
            assert!(
                serde_json::from_value::<BtcRealtimePaperControlConfig>(process_control).is_err()
            );
        }
    }

    #[test]
    fn retired_v1_ml_field_is_accepted_only_for_durable_resume() {
        let legacy = serde_json::json!({
            "schema_version": LEGACY_BTC_PROCESS_SCHEMA_VERSION,
            "next_experiment_key": "btc-5m-paper-20260713-c",
            "preregistration_sha256": "a".repeat(64),
            "ml_shadow": {"enabled": true},
        });
        assert!(
            parse_btc_process_control(legacy.clone(), BtcDefinitionUse::ExplicitStart).is_err()
        );
        let resumed = parse_btc_process_control(legacy, BtcDefinitionUse::DurableResume).unwrap();
        assert_eq!(resumed.schema_version, BTC_PROCESS_SCHEMA_VERSION);
    }

    #[test]
    fn selectable_v3_requires_and_resolves_an_explicit_strategy() {
        let control = parse_btc_process_control(
            serde_json::json!({
                "schema_version": SELECTABLE_BTC_PROCESS_SCHEMA_VERSION,
                "next_experiment_key": "btc-5m-selectable-paper-v1",
                "preregistration_sha256": "d".repeat(64),
                "strategy": {
                    "decision_strategy": {"type": "chainlink_fair_value"}
                }
            }),
            BtcDefinitionUse::ExplicitStart,
        )
        .unwrap();
        let strategy = resolve_btc_strategy(&control).unwrap();
        assert_eq!(strategy.strategy_version, BTC_STRATEGY_VERSION);
        assert_eq!(
            strategy.decision_strategy,
            Some(BtcDecisionStrategyConfig::ChainlinkFairValue {})
        );

        let missing = BtcRealtimePaperControlConfig {
            schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
            strategy: serde_json::json!({}),
            ..BtcRealtimePaperControlConfig::default()
        };
        assert!(resolve_btc_strategy(&missing).is_err());
    }

    #[test]
    fn selectable_v3_resolves_nested_continuation_configuration() {
        let continuation = polymarket_bot::btc::BtcVolatilityContinuationConfig::default();
        let control = BtcRealtimePaperControlConfig {
            schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
            strategy: serde_json::json!({
                "decision_strategy": {
                    "type": "volatility_continuation",
                    "config": continuation
                }
            }),
            ..BtcRealtimePaperControlConfig::default()
        };
        let strategy = resolve_btc_strategy(&control).unwrap();

        assert_eq!(
            strategy.strategy_version,
            BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION
        );
        assert!(matches!(
            strategy.decision_strategy,
            Some(BtcDecisionStrategyConfig::VolatilityContinuation { .. })
        ));
        assert!(strategy.volatility_continuation.is_none());
    }

    #[test]
    fn selectable_v3_resolves_and_freezes_market_anchored_profile() {
        let profile_id = polymarket_bot::btc::BTC_MARKET_ANCHORED_RESEARCH_PROFILE_ID;
        let profile_sha256 = polymarket_bot::btc::BTC_MARKET_ANCHORED_RESEARCH_PROFILE_SHA256;
        let control = BtcRealtimePaperControlConfig {
            schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
            next_experiment_key: "btc-5m-market-anchored-preview".to_string(),
            preregistration_sha256: "f".repeat(64),
            strategy: serde_json::json!({
                "decision_strategy": {
                    "type": "market_anchored_fair_value",
                    "profile_id": profile_id,
                    "profile_sha256": profile_sha256
                },
                "min_entry_price": "0.30"
            }),
            ..BtcRealtimePaperControlConfig::default()
        };
        let strategy = resolve_btc_strategy(&control).unwrap();
        assert_eq!(
            strategy.strategy_version,
            BTC_MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION
        );
        let chainlink_hash = prepare_btc_start_definition(ResolvedBtcProcessDefinition {
            control: BtcRealtimePaperControlConfig {
                schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
                next_experiment_key: control.next_experiment_key.clone(),
                preregistration_sha256: control.preregistration_sha256.clone(),
                strategy: serde_json::json!({
                    "decision_strategy": {"type": "chainlink_fair_value"},
                    "min_entry_price": "0.30"
                }),
                ..BtcRealtimePaperControlConfig::default()
            },
            strategy: BtcStrategyConfig {
                decision_strategy: Some(BtcDecisionStrategyConfig::ChainlinkFairValue {}),
                min_entry_price: dec!(0.30),
                ..BtcStrategyConfig::default()
            },
            entry_admission: None,
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        })
        .unwrap()
        .config_hash;
        let prepared = prepare_btc_start_definition(ResolvedBtcProcessDefinition {
            control,
            strategy,
            entry_admission: None,
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        })
        .unwrap();

        assert_ne!(prepared.config_hash, chainlink_hash);
        assert_eq!(
            prepared.frozen_process_config.raw["strategy"]["decision_strategy"],
            serde_json::json!({
                "type": "market_anchored_fair_value",
                "profile_id": profile_id,
                "profile_sha256": profile_sha256
            })
        );
    }

    #[test]
    fn selectable_v3_resolves_and_freezes_chainlink_persistence_profile() {
        let profile_id = polymarket_bot::btc::BTC_CHAINLINK_PERSISTENCE_CALIBRATED_PROFILE_ID;
        let profile_sha256 =
            polymarket_bot::btc::BTC_CHAINLINK_PERSISTENCE_CALIBRATED_PROFILE_SHA256;
        let control = BtcRealtimePaperControlConfig {
            schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
            next_experiment_key: "btc-5m-chainlink-persistence-preview".to_string(),
            preregistration_sha256: "9".repeat(64),
            strategy: serde_json::json!({
                "decision_strategy": {
                    "type": "chainlink_persistence_calibrated_fair_value",
                    "profile_id": profile_id,
                    "profile_sha256": profile_sha256
                },
                "min_entry_price": "0.30"
            }),
            ..BtcRealtimePaperControlConfig::default()
        };
        let strategy = resolve_btc_strategy(&control).unwrap();
        assert_eq!(
            strategy.strategy_version,
            BTC_CHAINLINK_PERSISTENCE_CALIBRATED_STRATEGY_VERSION
        );
        assert_eq!(
            strategy.feature_schema_version,
            BTC_CHAINLINK_PERSISTENCE_CALIBRATED_FEATURE_SCHEMA_VERSION
        );
        assert!(matches!(
            strategy.decision_strategy,
            Some(BtcDecisionStrategyConfig::ChainlinkPersistenceCalibratedFairValue { .. })
        ));
        let prepared = prepare_btc_start_definition(ResolvedBtcProcessDefinition {
            control,
            strategy,
            entry_admission: None,
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        })
        .unwrap();
        assert_eq!(
            prepared.frozen_process_config.raw["strategy"]["decision_strategy"],
            serde_json::json!({
                "type": "chainlink_persistence_calibrated_fair_value",
                "profile_id": profile_id,
                "profile_sha256": profile_sha256
            })
        );

        let bad_control = BtcRealtimePaperControlConfig {
            schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
            strategy: serde_json::json!({
                "decision_strategy": {
                    "type": "chainlink_persistence_calibrated_fair_value",
                    "profile_id": profile_id,
                    "profile_sha256": "0".repeat(64)
                }
            }),
            ..BtcRealtimePaperControlConfig::default()
        };
        assert!(resolve_btc_strategy(&bad_control).is_err());
    }

    #[test]
    fn selectable_v3_resolves_and_freezes_chainlink_path_conditioned_profile() {
        let profile_id = polymarket_bot::btc::BTC_CHAINLINK_PATH_CONDITIONED_PROFILE_ID;
        let profile_sha256 = polymarket_bot::btc::BTC_CHAINLINK_PATH_CONDITIONED_PROFILE_SHA256;
        let control = BtcRealtimePaperControlConfig {
            schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
            next_experiment_key: "btc-5m-chainlink-path-conditioned-preview".to_string(),
            preregistration_sha256: "8".repeat(64),
            strategy: serde_json::json!({
                "decision_strategy": {
                    "type": "chainlink_path_conditioned_fair_value",
                    "profile_id": profile_id,
                    "profile_sha256": profile_sha256
                },
                "min_entry_price": "0.30"
            }),
            ..BtcRealtimePaperControlConfig::default()
        };
        let strategy = resolve_btc_strategy(&control).unwrap();
        assert_eq!(
            strategy.strategy_version,
            BTC_CHAINLINK_PATH_CONDITIONED_STRATEGY_VERSION
        );
        assert_eq!(
            strategy.feature_schema_version,
            BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_SCHEMA_VERSION
        );
        assert!(matches!(
            strategy.decision_strategy,
            Some(BtcDecisionStrategyConfig::ChainlinkPathConditionedFairValue { .. })
        ));
        let prepared = prepare_btc_start_definition(ResolvedBtcProcessDefinition {
            control,
            strategy,
            entry_admission: None,
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        })
        .unwrap();
        assert_eq!(
            prepared.frozen_process_config.raw["strategy"]["decision_strategy"],
            serde_json::json!({
                "type": "chainlink_path_conditioned_fair_value",
                "profile_id": profile_id,
                "profile_sha256": profile_sha256
            })
        );

        let bad_control = BtcRealtimePaperControlConfig {
            schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
            strategy: serde_json::json!({
                "decision_strategy": {
                    "type": "chainlink_path_conditioned_fair_value",
                    "profile_id": profile_id,
                    "profile_sha256": "0".repeat(64)
                }
            }),
            ..BtcRealtimePaperControlConfig::default()
        };
        assert!(resolve_btc_strategy(&bad_control).is_err());
    }

    #[test]
    fn selectable_v3_resolves_and_freezes_directional_prediction_contract() {
        let profile_id = polymarket_bot::btc::BTC_MARKET_ANCHORED_RESEARCH_PROFILE_ID;
        let profile_sha256 = polymarket_bot::btc::BTC_MARKET_ANCHORED_RESEARCH_PROFILE_SHA256;
        let directional_config = polymarket_bot::btc::BtcDirectionalPredictionConfig::default();
        assert_eq!(directional_config.min_conservative_probability, dec!(0.75));

        let directional_control = BtcRealtimePaperControlConfig {
            schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
            next_experiment_key: "btc-5m-directional-prediction-preview".to_string(),
            preregistration_sha256: "e".repeat(64),
            strategy: serde_json::json!({
                "decision_strategy": {
                    "type": "market_anchored_directional_prediction",
                    "profile_id": profile_id,
                    "profile_sha256": profile_sha256,
                    "config": directional_config
                },
                "min_entry_price": "0.30"
            }),
            ..BtcRealtimePaperControlConfig::default()
        };
        let strategy = resolve_btc_strategy(&directional_control).unwrap();
        assert_eq!(
            strategy.strategy_version,
            BTC_MARKET_ANCHORED_DIRECTIONAL_PREDICTION_STRATEGY_VERSION
        );
        assert!(matches!(
            strategy.decision_strategy.as_ref(),
            Some(BtcDecisionStrategyConfig::MarketAnchoredDirectionalPrediction {
                config: polymarket_bot::btc::BtcDirectionalPredictionConfig {
                    min_conservative_probability: value
                },
                ..
            }) if *value == dec!(0.75)
        ));

        let legacy_control = BtcRealtimePaperControlConfig {
            strategy: serde_json::json!({
                "decision_strategy": {
                    "type": "market_anchored_fair_value",
                    "profile_id": profile_id,
                    "profile_sha256": profile_sha256
                },
                "min_entry_price": "0.30"
            }),
            ..directional_control.clone()
        };
        let legacy_strategy = resolve_btc_strategy(&legacy_control).unwrap();
        let legacy_prepared = prepare_btc_start_definition(ResolvedBtcProcessDefinition {
            control: legacy_control,
            strategy: legacy_strategy,
            entry_admission: None,
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        })
        .unwrap();
        let directional_prepared = prepare_btc_start_definition(ResolvedBtcProcessDefinition {
            control: directional_control,
            strategy,
            entry_admission: None,
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        })
        .unwrap();

        assert_ne!(
            directional_prepared.config_hash,
            legacy_prepared.config_hash
        );
        assert_eq!(
            directional_prepared.frozen_process_config.raw["strategy"]["decision_strategy"],
            serde_json::json!({
                "type": "market_anchored_directional_prediction",
                "profile_id": profile_id,
                "profile_sha256": profile_sha256,
                "config": {"min_conservative_probability": "0.75"}
            })
        );
    }

    #[test]
    fn selectable_v3_rejects_unknown_market_anchored_profile() {
        let control = BtcRealtimePaperControlConfig {
            schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
            strategy: serde_json::json!({
                "decision_strategy": {
                    "type": "market_anchored_fair_value",
                    "profile_id": polymarket_bot::btc::BTC_MARKET_ANCHORED_RESEARCH_PROFILE_ID,
                    "profile_sha256": "0".repeat(64)
                }
            }),
            ..BtcRealtimePaperControlConfig::default()
        };

        assert!(resolve_btc_strategy(&control).is_err());
    }

    #[test]
    fn selectable_v3_rejects_invalid_directional_prediction_contract() {
        let profile_id = polymarket_bot::btc::BTC_MARKET_ANCHORED_RESEARCH_PROFILE_ID;
        let profile_sha256 = polymarket_bot::btc::BTC_MARKET_ANCHORED_RESEARCH_PROFILE_SHA256;
        for decision_strategy in [
            serde_json::json!({
                "type": "market_anchored_directional_prediction",
                "profile_id": "unknown-market-profile",
                "profile_sha256": profile_sha256,
                "config": {"min_conservative_probability": "0.75"}
            }),
            serde_json::json!({
                "type": "market_anchored_directional_prediction",
                "profile_id": profile_id,
                "profile_sha256": "0".repeat(64),
                "config": {"min_conservative_probability": "0.75"}
            }),
            serde_json::json!({
                "type": "market_anchored_directional_prediction",
                "profile_id": profile_id,
                "profile_sha256": profile_sha256,
                "config": {"min_conservative_probability": "0.70"}
            }),
            serde_json::json!({
                "type": "market_anchored_directional_prediction",
                "profile_id": profile_id,
                "profile_sha256": profile_sha256
            }),
            serde_json::json!({
                "type": "market_anchored_directional_prediction",
                "profile_id": profile_id,
                "profile_sha256": profile_sha256,
                "config": {
                    "min_conservative_probability": "0.75",
                    "unexpected": true
                }
            }),
        ] {
            let control = BtcRealtimePaperControlConfig {
                schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
                strategy: serde_json::json!({"decision_strategy": decision_strategy}),
                ..BtcRealtimePaperControlConfig::default()
            };
            assert!(resolve_btc_strategy(&control).is_err());
        }
    }

    #[test]
    fn legacy_v2_rejects_selector_and_v3_rejects_conflicting_or_unknown_fields() {
        let legacy_with_selector = BtcRealtimePaperControlConfig {
            schema_version: BTC_PROCESS_SCHEMA_VERSION.to_string(),
            strategy: serde_json::json!({
                "decision_strategy": {"type": "chainlink_fair_value"}
            }),
            ..BtcRealtimePaperControlConfig::default()
        };
        assert!(resolve_btc_strategy(&legacy_with_selector).is_err());

        let mut invalid_continuation =
            serde_json::to_value(polymarket_bot::btc::BtcVolatilityContinuationConfig::default())
                .unwrap();
        invalid_continuation
            .as_object_mut()
            .unwrap()
            .insert("unexpected".to_string(), serde_json::json!(true));

        for strategy in [
            serde_json::json!({
                "strategy_version": BTC_STRATEGY_VERSION,
                "decision_strategy": {"type": "chainlink_fair_value"}
            }),
            serde_json::json!({
                "feature_schema_version": BTC_FEATURE_SCHEMA_VERSION,
                "decision_strategy": {"type": "chainlink_fair_value"}
            }),
            serde_json::json!({
                "volatility_continuation": null,
                "decision_strategy": {"type": "chainlink_fair_value"}
            }),
            serde_json::json!({
                "unknown_setting": true,
                "decision_strategy": {"type": "chainlink_fair_value"}
            }),
            serde_json::json!({
                "decision_strategy": {
                    "type": "chainlink_fair_value",
                    "unexpected": true
                }
            }),
            serde_json::json!({
                "decision_strategy": {"type": "unknown_strategy"}
            }),
            serde_json::json!({
                "decision_strategy": {
                    "type": "volatility_continuation",
                    "config": invalid_continuation
                }
            }),
        ] {
            let control = BtcRealtimePaperControlConfig {
                schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
                strategy,
                ..BtcRealtimePaperControlConfig::default()
            };
            assert!(resolve_btc_strategy(&control).is_err());
        }
    }

    #[test]
    fn btc_start_eligibility_rejects_generic_active_and_invalid_definitions() {
        let mut generic = eligible_btc_process();
        generic.process_type = "copy_trade".to_string();
        assert!(validate_btc_start_eligibility(&generic, true, true).is_err());

        let mut active = eligible_btc_process();
        active.status = "running".to_string();
        active.enabled = true;
        assert!(validate_btc_start_eligibility(&active, true, true).is_err());

        let mut invalid = eligible_btc_process();
        invalid.config.execution.as_mut().unwrap().mode = Some("live".to_string());
        assert!(validate_btc_start_eligibility(&invalid, true, true).is_err());

        assert!(validate_btc_start_eligibility(&eligible_btc_process(), true, true).is_ok());
    }

    #[test]
    fn btc_start_preparation_is_deterministic_and_freezes_exact_execution_config() {
        let experiment_key = "btc-5m-paper-20260713-preview";
        let preregistration_sha256 = "b".repeat(64);
        let resolved = ResolvedBtcProcessDefinition {
            control: BtcRealtimePaperControlConfig {
                schema_version: BTC_PROCESS_SCHEMA_VERSION.to_string(),
                next_experiment_key: experiment_key.to_string(),
                preregistration_sha256: preregistration_sha256.clone(),
                ..BtcRealtimePaperControlConfig::default()
            },
            strategy: BtcStrategyConfig::default(),
            entry_admission: None,
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        };

        let first = prepare_btc_start_definition(resolved.clone()).unwrap();
        let second = prepare_btc_start_definition(resolved).unwrap();
        let expected_experiment_id = uuid::Uuid::new_v5(
            &uuid::Uuid::NAMESPACE_URL,
            format!("polymarket-bot/btc-paper/{experiment_key}").as_bytes(),
        );
        let serialized_config = serde_json::to_vec(&first.frozen_process_config).unwrap();
        let expected_config_hash = format!("{:x}", Sha256::digest(serialized_config));

        assert_eq!(first.experiment_id, expected_experiment_id);
        assert_eq!(first.experiment_id, second.experiment_id);
        assert_eq!(first.config_hash, expected_config_hash);
        assert_eq!(first.config_hash, second.config_hash);
        let frozen_runtime = first.frozen_process_config.raw["runtime"]
            .as_object()
            .unwrap();
        assert!(!frozen_runtime.contains_key("clob_heartbeat_interval"));
        assert!(!frozen_runtime.contains_key("rtds_heartbeat_interval"));
        assert!(!frozen_runtime.contains_key("binance_heartbeat_interval"));
        assert_eq!(
            serde_json::to_value(&first.frozen_process_config).unwrap(),
            serde_json::to_value(&second.frozen_process_config).unwrap()
        );
        assert_eq!(
            first.frozen_process_config.raw["preregistration_sha256"],
            preregistration_sha256
        );
        assert_eq!(
            first.frozen_process_config.raw["process_schema_version"],
            BTC_PROCESS_SCHEMA_VERSION
        );
        assert_eq!(
            first.frozen_process_config.raw["pipeline_version"],
            BTC_PIPELINE_VERSION
        );
        assert!(first.frozen_process_config.raw["strategy"]
            .get("decision_strategy")
            .is_none());
        assert!(first.frozen_process_config.raw.get("ml_shadow").is_none());
        assert!(first
            .frozen_process_config
            .raw
            .get("entry_admission")
            .is_none());
        assert_eq!(
            first
                .frozen_process_config
                .execution
                .as_ref()
                .and_then(|execution| execution.mode.as_deref()),
            Some("paper")
        );
    }

    #[test]
    fn btc_start_preparation_freezes_configured_entry_admission() {
        let entry_admission = BtcEntryAdmissionConfig {
            loss_regime_confidence_floor: polymarket_bot::btc::LossRegimeConfidenceFloorConfig {
                schema_version: polymarket_bot::btc::LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION
                    .to_string(),
                activation_consecutive_candidate_losses: 2,
                min_conservative_probability: dec!(0.50),
                release_consecutive_candidate_wins: 1,
            },
            daily_realized_pnl_high_water_mark: None,
            shadow_predictive_regime_circuit_breaker: None,
        };
        let resolved = ResolvedBtcProcessDefinition {
            control: BtcRealtimePaperControlConfig {
                schema_version: BTC_PROCESS_SCHEMA_VERSION.to_string(),
                next_experiment_key: "btc-5m-paper-admission-preview".to_string(),
                preregistration_sha256: "c".repeat(64),
                entry_admission: Some(entry_admission.clone()),
                ..BtcRealtimePaperControlConfig::default()
            },
            strategy: BtcStrategyConfig::default(),
            entry_admission: Some(entry_admission.clone()),
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        };

        let prepared = prepare_btc_start_definition(resolved).unwrap();
        assert_eq!(prepared.entry_admission, Some(entry_admission.clone()));
        assert_eq!(
            prepared.frozen_process_config.raw["entry_admission"],
            serde_json::to_value(entry_admission).unwrap()
        );
    }

    #[test]
    fn btc_start_preparation_freezes_process_scoped_high_water_mark() {
        let entry_admission = BtcEntryAdmissionConfig {
            loss_regime_confidence_floor: polymarket_bot::btc::LossRegimeConfidenceFloorConfig {
                schema_version: polymarket_bot::btc::LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION
                    .to_string(),
                activation_consecutive_candidate_losses: 2,
                min_conservative_probability: dec!(0.50),
                release_consecutive_candidate_wins: 1,
            },
            daily_realized_pnl_high_water_mark: Some(
                polymarket_bot::btc::DailyRealizedPnlHighWaterMarkConfig {
                    schema_version:
                        polymarket_bot::btc::DAILY_REALIZED_PNL_HIGH_WATER_MARK_SCHEMA_VERSION
                            .to_string(),
                    activation_realized_pnl_usd: dec!(5),
                    max_drawdown_from_high_water_mark_usd: dec!(5),
                },
            ),
            shadow_predictive_regime_circuit_breaker: None,
        };
        let resolved = ResolvedBtcProcessDefinition {
            control: BtcRealtimePaperControlConfig {
                schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
                next_experiment_key: "btc-5m-paper-hwm-preview".to_string(),
                preregistration_sha256: "d".repeat(64),
                entry_admission: Some(entry_admission.clone()),
                ..BtcRealtimePaperControlConfig::default()
            },
            strategy: BtcStrategyConfig {
                decision_strategy: Some(BtcDecisionStrategyConfig::ChainlinkFairValue {}),
                ..BtcStrategyConfig::default()
            },
            entry_admission: Some(entry_admission.clone()),
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        };

        let prepared = prepare_btc_start_definition(resolved).unwrap();
        assert_eq!(prepared.entry_admission, Some(entry_admission.clone()));
        assert_eq!(
            prepared.frozen_process_config.raw["entry_admission"]
                ["daily_realized_pnl_high_water_mark"]["schema_version"],
            polymarket_bot::btc::DAILY_REALIZED_PNL_HIGH_WATER_MARK_SCHEMA_VERSION
        );
        assert_eq!(
            prepared.frozen_process_config.raw["entry_admission"],
            serde_json::to_value(entry_admission).unwrap()
        );
    }

    #[test]
    fn btc_start_preparation_freezes_process_scoped_shadow_circuit_breaker() {
        let entry_admission = BtcEntryAdmissionConfig {
            loss_regime_confidence_floor: polymarket_bot::btc::LossRegimeConfidenceFloorConfig {
                schema_version: polymarket_bot::btc::LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION
                    .to_string(),
                activation_consecutive_candidate_losses: 2,
                min_conservative_probability: dec!(0.50),
                release_consecutive_candidate_wins: 1,
            },
            daily_realized_pnl_high_water_mark: None,
            shadow_predictive_regime_circuit_breaker: Some(
                polymarket_bot::btc::ShadowPredictiveRegimeCircuitBreakerConfig {
                    schema_version:
                        polymarket_bot::btc::SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION
                            .to_string(),
                    mode: polymarket_bot::btc::SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE
                        .to_string(),
                    rolling_resolved_market_window: 20,
                    minimum_resolved_markets: 20,
                    degradation_brier_score_threshold: dec!(0.23),
                    degradation_overconfidence_gap_threshold: dec!(0.12),
                    degradation_confirmation_markets: 2,
                    recovery_brier_score_threshold: dec!(0.21),
                    recovery_overconfidence_gap_threshold: dec!(0.05),
                    recovery_confirmation_markets: 2,
                }
                .into(),
            ),
        };
        let resolved = ResolvedBtcProcessDefinition {
            control: BtcRealtimePaperControlConfig {
                schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
                next_experiment_key: "btc-5m-paper-shadow-breaker-preview".to_string(),
                preregistration_sha256: "f".repeat(64),
                entry_admission: Some(entry_admission.clone()),
                ..BtcRealtimePaperControlConfig::default()
            },
            strategy: BtcStrategyConfig::default(),
            entry_admission: Some(entry_admission.clone()),
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        };

        let prepared = prepare_btc_start_definition(resolved).unwrap();
        assert_eq!(prepared.entry_admission, Some(entry_admission.clone()));
        assert_eq!(
            prepared.frozen_process_config.raw["entry_admission"]
                ["shadow_predictive_regime_circuit_breaker"]["mode"],
            "shadow"
        );
        assert_eq!(
            prepared.frozen_process_config.raw["entry_admission"],
            serde_json::to_value(entry_admission).unwrap()
        );
    }

    #[test]
    fn btc_start_preparation_freezes_v2_shadow_breaker_under_existing_key() {
        let entry_admission = BtcEntryAdmissionConfig {
            loss_regime_confidence_floor: polymarket_bot::btc::LossRegimeConfidenceFloorConfig {
                schema_version: polymarket_bot::btc::LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION
                    .to_string(),
                activation_consecutive_candidate_losses: 2,
                min_conservative_probability: dec!(0.50),
                release_consecutive_candidate_wins: 1,
            },
            daily_realized_pnl_high_water_mark: None,
            shadow_predictive_regime_circuit_breaker: Some(
                polymarket_bot::btc::ShadowPredictiveRegimeCircuitBreakerV2Config {
                    schema_version:
                        polymarket_bot::btc::SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_V2_SCHEMA_VERSION
                            .to_string(),
                    mode: polymarket_bot::btc::SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE
                        .to_string(),
                    fast_resolved_market_window: 4,
                    slow_resolved_market_window: 20,
                    minimum_resolved_markets: 20,
                    max_evidence_gap_seconds: 900,
                    degradation_fast_brier_score_threshold: dec!(0.27),
                    degradation_fast_minus_slow_threshold: dec!(0.02),
                    degradation_slow_brier_score_threshold: dec!(0.25),
                    degradation_confirmation_markets: 2,
                    recovery_fast_brier_score_threshold: dec!(0.25),
                    recovery_fast_minus_slow_ceiling: dec!(0),
                    recovery_confirmation_markets: 2,
                }
                .into(),
            ),
        };
        let resolved = ResolvedBtcProcessDefinition {
            control: BtcRealtimePaperControlConfig {
                schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
                next_experiment_key: "btc-5m-paper-shadow-breaker-v2-preview".to_string(),
                preregistration_sha256: "a".repeat(64),
                entry_admission: Some(entry_admission.clone()),
                ..BtcRealtimePaperControlConfig::default()
            },
            strategy: BtcStrategyConfig::default(),
            entry_admission: Some(entry_admission.clone()),
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        };

        let prepared = prepare_btc_start_definition(resolved).unwrap();
        let frozen = &prepared.frozen_process_config.raw["entry_admission"]
            ["shadow_predictive_regime_circuit_breaker"];
        assert_eq!(
            frozen["schema_version"],
            polymarket_bot::btc::SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_V2_SCHEMA_VERSION
        );
        assert!(frozen.get("V1").is_none());
        assert!(frozen.get("V2").is_none());
        assert_eq!(
            prepared.frozen_process_config.raw["entry_admission"],
            serde_json::to_value(entry_admission).unwrap()
        );
    }

    #[test]
    fn selectable_start_preparation_freezes_v3_contract_identity() {
        let strategy = BtcStrategyConfig {
            decision_strategy: Some(BtcDecisionStrategyConfig::ChainlinkFairValue {}),
            ..BtcStrategyConfig::default()
        };
        let resolved = ResolvedBtcProcessDefinition {
            control: BtcRealtimePaperControlConfig {
                schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
                next_experiment_key: "btc-5m-selectable-preview".to_string(),
                preregistration_sha256: "e".repeat(64),
                strategy: serde_json::json!({
                    "decision_strategy": {"type": "chainlink_fair_value"}
                }),
                ..BtcRealtimePaperControlConfig::default()
            },
            strategy,
            entry_admission: None,
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        };

        let prepared = prepare_btc_start_definition(resolved).unwrap();
        assert_eq!(
            prepared.frozen_process_config.raw["process_schema_version"],
            SELECTABLE_BTC_PROCESS_SCHEMA_VERSION
        );
        assert_eq!(
            prepared.frozen_process_config.raw["pipeline_version"],
            SELECTABLE_BTC_PIPELINE_VERSION
        );
        assert_eq!(
            prepared.frozen_process_config.raw["strategy"]["decision_strategy"]["type"],
            "chainlink_fair_value"
        );
    }

    #[test]
    fn shared_market_data_accepts_playbook_only_runtime_differences() {
        let shared = BtcRuntimeConfig::default();
        let mut playbook = shared.clone();
        playbook.strategy_interval = Duration::from_millis(500);
        playbook.max_book_age = Duration::from_secs(3);
        playbook.max_reference_age = Duration::from_secs(4);

        assert!(shared_market_data_config_compatible(&shared, &playbook));

        playbook.writer_capacity += 1;
        assert!(!shared_market_data_config_compatible(&shared, &playbook));
    }

    #[test]
    fn btc_resume_process_contract_ignores_all_system_heartbeat_metadata() {
        let prepared = prepared_btc_definition_with_default_runtime();
        let current = serde_json::to_value(&prepared.frozen_process_config).unwrap();
        assert_eq!(
            resume_process_contract_projection(current.clone()),
            resume_process_contract_projection(current.clone())
        );

        for historical_value in [
            serde_json::Value::Null,
            serde_json::json!("10s"),
            serde_json::to_value(Duration::ZERO).unwrap(),
            serde_json::to_value(Duration::from_secs(5)).unwrap(),
            serde_json::to_value(Duration::from_secs(10)).unwrap(),
            serde_json::to_value(Duration::from_secs(6)).unwrap(),
            serde_json::to_value(Duration::from_secs(30)).unwrap(),
            serde_json::to_value(Duration::new(10, 1)).unwrap(),
        ] {
            let mut durable = current.clone();
            for field in [
                "clob_heartbeat_interval",
                "rtds_heartbeat_interval",
                "binance_heartbeat_interval",
            ] {
                durable["raw"]["runtime"][field] = historical_value.clone();
            }
            assert_eq!(
                resume_process_contract_projection(current.clone()),
                resume_process_contract_projection(durable)
            );
        }
    }

    #[test]
    fn btc_resume_system_heartbeat_metadata_does_not_relax_process_parameters() {
        let prepared = prepared_btc_definition_with_default_runtime();
        let current = serde_json::to_value(&prepared.frozen_process_config).unwrap();
        let mut durable = serde_json::to_value(&prepared.frozen_process_config).unwrap();
        durable["raw"]["runtime"]["clob_heartbeat_interval"] = serde_json::json!("ignored");
        durable["raw"]["runtime"]["rtds_heartbeat_interval"] = serde_json::json!(5);
        durable["raw"]["runtime"]["binance_heartbeat_interval"] = serde_json::json!(20);
        durable["raw"]["runtime"]["writer_capacity"] = serde_json::json!(1);

        assert_ne!(
            resume_process_contract_projection(current),
            resume_process_contract_projection(durable)
        );
    }

    #[test]
    fn btc_resume_preserves_frozen_parameters_across_service_rebuilds() {
        let durable = serde_json::json!({
            "raw": {
                "process_schema_version": LEGACY_BTC_PROCESS_SCHEMA_VERSION,
                "ml_shadow": {
                    "ml_a_enabled": true,
                    "ml_b_enabled": true,
                    "execution_authority": false
                },
                "build": {
                    "package_version": "0.1.0",
                    "compiled_source_identity": "tree-sha256:old"
                },
                "strategy": {"threshold": "0.03"}
            }
        });
        let rebuilt = serde_json::json!({
            "raw": {
                "process_schema_version": BTC_PROCESS_SCHEMA_VERSION,
                "build": {
                    "package_version": "0.1.0",
                    "compiled_source_identity": "tree-sha256:new"
                },
                "strategy": {"threshold": "0.03"}
            }
        });
        assert_eq!(
            resume_process_contract_projection(durable.clone()),
            resume_process_contract_projection(rebuilt)
        );

        let changed_parameters = serde_json::json!({
            "raw": {
                "process_schema_version": BTC_PROCESS_SCHEMA_VERSION,
                "build": {
                    "package_version": "0.1.0",
                    "compiled_source_identity": "tree-sha256:new"
                },
                "strategy": {"threshold": "0.04"}
            }
        });
        assert_ne!(
            resume_process_contract_projection(durable),
            resume_process_contract_projection(changed_parameters)
        );
    }

    #[test]
    fn readme_btc_process_contract_matches_v2_parser() {
        let readme = include_str!("../../../README.md");
        let contract = readme
            .split("<!-- btc-5m-process-v2:start -->")
            .nth(1)
            .and_then(|tail| tail.split("<!-- btc-5m-process-v2:end -->").next())
            .expect("README must contain the BTC v2 process contract example")
            .trim()
            .strip_prefix("```json")
            .and_then(|json| json.trim().strip_suffix("```"))
            .expect("README BTC process contract must be a JSON code block");
        let request: control_http::UpsertTradingProcessByKeyRequest =
            serde_json::from_str(contract).unwrap();
        let control = request
            .config
            .raw
            .get("btc_realtime_paper")
            .cloned()
            .unwrap();
        parse_btc_process_control(control, BtcDefinitionUse::ExplicitStart).unwrap();

        let now = Utc::now();
        let process = TradingProcess {
            process_id: uuid::Uuid::nil(),
            name: request.name,
            process_type: request.process_type,
            process_scope: request.process_scope,
            process_key: Some("btc-5m-chainlink-paper".to_string()),
            status: request.status,
            enabled: request.enabled,
            config: request.config,
            metadata: request.metadata,
            created_at: now,
            updated_at: now,
            started_at: None,
            stopped_at: None,
            last_error: None,
        };
        validate_btc_start_eligibility(&process, true, true).unwrap();
    }

    #[test]
    fn readme_selectable_btc_process_contract_matches_v3_resolver() {
        let readme = include_str!("../../../README.md");
        let contract = readme
            .split("<!-- btc-5m-process-v3:start -->")
            .nth(1)
            .and_then(|tail| tail.split("<!-- btc-5m-process-v3:end -->").next())
            .expect("README must contain the BTC v3 process contract example")
            .trim()
            .strip_prefix("```json")
            .and_then(|json| json.trim().strip_suffix("```"))
            .expect("README BTC v3 process contract must be a JSON code block");
        let request: control_http::UpsertTradingProcessByKeyRequest =
            serde_json::from_str(contract).unwrap();
        let control = parse_btc_process_control(
            request
                .config
                .raw
                .get("btc_realtime_paper")
                .cloned()
                .unwrap(),
            BtcDefinitionUse::ExplicitStart,
        )
        .unwrap();
        let strategy = resolve_btc_strategy(&control).unwrap();

        assert_eq!(
            control.schema_version,
            SELECTABLE_BTC_PROCESS_SCHEMA_VERSION
        );
        assert_eq!(
            strategy.strategy_version,
            BTC_MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION
        );
        assert_eq!(
            strategy.attribution().unwrap().profile_sha256,
            Some(polymarket_bot::btc::BTC_MARKET_ANCHORED_RESEARCH_PROFILE_SHA256)
        );

        let now = Utc::now();
        let process = TradingProcess {
            process_id: uuid::Uuid::nil(),
            name: request.name,
            process_type: request.process_type,
            process_scope: request.process_scope,
            process_key: Some("btc-5m-market-anchored-research".to_string()),
            status: request.status,
            enabled: request.enabled,
            config: request.config,
            metadata: request.metadata,
            created_at: now,
            updated_at: now,
            started_at: None,
            stopped_at: None,
            last_error: None,
        };
        validate_btc_start_eligibility(&process, true, true).unwrap();
    }
}
