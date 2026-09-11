use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chrono::Utc;
use polymarket_bot::btc::unified_model_runtime::risk::{
    self as risk_runtime, RiskStrategySelection,
};
use polymarket_bot::grafana_live::MarketPathPublicationState;
use polymarket_bot::{
    btc::{
        process_runtime_readiness, runtime_model, runtime_status_from_inputs, BookRegistry,
        BtcDecisionStrategyConfig, BtcDirectionalModelEntryPolicy, BtcEntryAdmissionConfig,
        BtcExecutionLifecycle, BtcExecutionMode, BtcLiveExecutionAdapter, BtcPlaybookRuntimeHandle,
        BtcProcessConfig, BtcProcessRunner, BtcRepository, BtcRuntime, BtcRuntimeConfig,
        BtcRuntimeHandle, BtcStrategyConfig, LiveExecutionLifecycle, PaperExecutionLifecycle,
        PaperPreviewConfig, PaperVenue as BtcPaperVenue, PaperVenueConfig, RuntimeModelSelection,
        BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_VERSION, BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION,
    },
    config::AppConfig,
    data_api::DataApiClient,
    events::ServiceEvent,
    execution::{
        live::LiveVenue, ExecutionVenue, LiveIdentityDiagnostics, LiveOrderDryRunDiagnostics,
        LiveOrderDryRunRequest, LivePoly1271FunderProbeRequest, LivePoly1271FunderProbeResponse,
        LiveVenueStatus, LiveWalletAddressDiagnostics,
    },
    grafana_live::{
        CountdownSnapshot, EntryPermission, GrafanaLivePublisher, ProcessEntryPermission,
        TradingEntryStatusSnapshot,
    },
    http as control_http,
    http::{
        ControlApi, EntryStatusRequest, HealthResponse, HealthStatus, HttpError, MetricsResponse,
        TradingProcessLivePreflightResponse, TradingProcessResponse,
        TradingProcessStartPreviewResponse, TradingProcessStatusResponse, TradingProcessesResponse,
    },
    market_data_stream::{legacy_default_sources, SourceSelector},
    models::{
        EffectiveProcessExecutionConfig, ProcessExecutionConfig, TradingProcess,
        TradingProcessConfig,
    },
    store::Store,
};
use rust_decimal::Decimal;
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
const BTC_LIVE_QUIESCE_TIMEOUT: Duration = Duration::from_secs(10);
const BTC_LIVE_CONTROL_TIMEOUT: Duration = Duration::from_secs(20);
const GRAFANA_STATUS_READ_TIMEOUT: Duration = Duration::from_millis(25);
const COMPILED_SOURCE_IDENTITY: &str = env!("POLYMARKET_COMPILED_SOURCE_ID");

fn should_resume_configured_live_entries(
    process: &TradingProcess,
    execution: &EffectiveProcessExecutionConfig,
) -> bool {
    process.enabled
        && process.status == "running"
        && execution.mode == "live"
        && execution.execute_signals
        && execution.live_capital
}

async fn preflight_btc_run_identity(
    repository: &BtcRepository,
    run_key: &str,
    run_id: uuid::Uuid,
) -> Result<()> {
    let identity_exists = repository
        .run_manifest_exists(run_id, run_key)
        .await
        .context("failed to preflight immutable BTC run identity")?;
    if identity_exists {
        bail!(
            "BTC run identity {run_key} already exists; every new explicit start requires a globally unique run key"
        );
    }
    Ok(())
}

async fn mark_btc_process_terminal(
    pool: &PgPool,
    process_id: uuid::Uuid,
    status: &str,
    reason: &str,
    allow_inactive_process: bool,
) -> Result<()> {
    validate_btc_process_terminal_request(status, reason)?;
    let mut tx = pool
        .begin()
        .await
        .context("failed to begin BTC process terminal transaction")?;
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
                if allow_inactive_process
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
        .context("failed to commit BTC process terminal transaction")?;
    Ok(())
}

fn validate_btc_process_terminal_request(status: &str, reason: &str) -> Result<()> {
    if !matches!(status, "stopped" | "failed" | "completed") || reason.trim().is_empty() {
        bail!("invalid BTC process terminal status or reason");
    }
    Ok(())
}

#[derive(Clone)]
struct BtcProcessManagerConfig {
    live_venue: Option<Arc<LiveVenue>>,
    live_reconcile_interval: Duration,
}

fn shared_market_data_config_compatible(left: &BtcRuntimeConfig, right: &BtcRuntimeConfig) -> bool {
    let _ = (left, right);
    true
}

fn shared_runtime_recovery_required(active_process_count: usize, shared_running: bool) -> bool {
    active_process_count > 0 && !shared_running
}

async fn invalidate_shared_market_data_evidence(
    state: &Arc<tokio::sync::RwLock<polymarket_bot::btc::RealtimeState>>,
    books: &Arc<tokio::sync::RwLock<BookRegistry>>,
) {
    *state.write().await = polymarket_bot::btc::RealtimeState::default();
    *books.write().await = BookRegistry::new(uuid::Uuid::new_v4());
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct BtcRealtimePaperControlConfig {
    schema_version: String,
    #[serde(default)]
    playbook_version: Option<String>,
    #[serde(default)]
    sources: Vec<SourceSelector>,
    next_experiment_key: String,
    preregistration_sha256: String,
    strategy: serde_json::Value,
    entry_admission: Option<BtcEntryAdmissionConfig>,
    #[serde(default)]
    risk_strategies: Vec<RiskStrategySelection>,
    runtime: BtcProcessRuntimeControl,
    paper: BtcProcessPaperControl,
}

impl Default for BtcRealtimePaperControlConfig {
    fn default() -> Self {
        Self {
            schema_version: String::new(),
            playbook_version: None,
            sources: Vec::new(),
            next_experiment_key: String::new(),
            preregistration_sha256: String::new(),
            strategy: serde_json::json!({}),
            entry_admission: None,
            risk_strategies: Vec::new(),
            runtime: BtcProcessRuntimeControl::default(),
            paper: BtcProcessPaperControl::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BtcDefinitionUse {
    InactiveDefinition,
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

    let mut control: BtcRealtimePaperControlConfig =
        serde_json::from_value(value).map_err(|error| {
            HttpError::bad_request(format!(
                "invalid process config.raw.btc_realtime_paper: {error}"
            ))
        })?;
    if control.sources.is_empty() {
        if definition_use == BtcDefinitionUse::DurableResume {
            control.sources = legacy_default_sources();
        } else {
            return Err(HttpError::bad_request(
                "BTC process sources are required; update the playbook to version v1.2",
            ));
        }
    } else if control.playbook_version.as_deref() != Some("v1.2") {
        return Err(HttpError::bad_request(
            "BTC process sources require playbook_version v1.2",
        ));
    }
    for source in &control.sources {
        source
            .validate()
            .map_err(|error| HttpError::bad_request(error.to_string()))?;
    }
    let unique = control
        .sources
        .iter()
        .map(|source| source.key.as_str())
        .collect::<HashSet<_>>();
    if unique.len() != control.sources.len() {
        return Err(HttpError::bad_request("BTC process sources must be unique"));
    }
    Ok(control)
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
        for compiled_or_legacy_key in ["strategy_version", "feature_schema_version"] {
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
    strategy_object.insert("decision_strategy".to_string(), serde_json::Value::Null);
    strategy_object.insert("unified_model".to_string(), serde_json::Value::Null);
    strategy_object.insert(
        "max_directional_feature_age_ms".to_string(),
        serde_json::Value::Null,
    );
    strategy_object.insert(
        "required_model_feeds".to_string(),
        serde_json::Value::Array(Vec::new()),
    );
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
        let (strategy_version, feature_schema_version) = match &selection {
            BtcDecisionStrategyConfig::BtcDirectionalModel {
                model_key,
                artifact_sha256,
                feature_schema_sha256,
            } => {
                let model = runtime_model(&RuntimeModelSelection {
                    model_key: model_key.clone(),
                    artifact_sha256: artifact_sha256.clone(),
                    feature_schema_sha256: feature_schema_sha256.clone(),
                })
                .map_err(|error| {
                    HttpError::bad_request(format!(
                        "invalid BTC directional model selection: {error}"
                    ))
                })?;
                (
                    BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION.to_string(),
                    model.feature_schema_version().to_string(),
                )
            }
            BtcDecisionStrategyConfig::BtcAsymmetricValueModel {
                model_key,
                artifact_sha256,
                feature_schema_sha256,
            } => {
                let model = runtime_model(&RuntimeModelSelection {
                    model_key: model_key.clone(),
                    artifact_sha256: artifact_sha256.clone(),
                    feature_schema_sha256: feature_schema_sha256.clone(),
                })
                .map_err(|error| {
                    HttpError::bad_request(format!(
                        "invalid BTC asymmetric value model selection: {error}"
                    ))
                })?;
                if !model.is_asymmetric_value() {
                    return Err(HttpError::bad_request(
                        "selected model is not an asymmetric value artifact",
                    ));
                }
                (
                    BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_VERSION.to_string(),
                    model.feature_schema_version().to_string(),
                )
            }
        };
        strategy_object.insert(
            "strategy_version".to_string(),
            serde_json::Value::String(strategy_version),
        );
        strategy_object.insert(
            "feature_schema_version".to_string(),
            serde_json::Value::String(feature_schema_version),
        );
    }

    let strategy: BtcStrategyConfig = serde_json::from_value(strategy_value).map_err(|error| {
        HttpError::bad_request(format!("invalid BTC strategy settings: {error}"))
    })?;
    strategy
        .validate()
        .map_err(|error| HttpError::bad_request(error.to_string()))?;
    let directional_model_identity_valid = match strategy.decision_strategy.as_ref() {
        Some(BtcDecisionStrategyConfig::BtcDirectionalModel {
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        }) if strategy.strategy_version == BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION => {
            runtime_model(&RuntimeModelSelection {
                model_key: model_key.clone(),
                artifact_sha256: artifact_sha256.clone(),
                feature_schema_sha256: feature_schema_sha256.clone(),
            })
            .is_ok_and(|model| model.feature_schema_version() == strategy.feature_schema_version)
        }
        Some(BtcDecisionStrategyConfig::BtcAsymmetricValueModel {
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        }) if strategy.strategy_version == BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_VERSION => {
            runtime_model(&RuntimeModelSelection {
                model_key: model_key.clone(),
                artifact_sha256: artifact_sha256.clone(),
                feature_schema_sha256: feature_schema_sha256.clone(),
            })
            .is_ok_and(|model| {
                model.is_asymmetric_value()
                    && model.feature_schema_version() == strategy.feature_schema_version
            })
        }
        _ => false,
    };
    if !directional_model_identity_valid {
        return Err(HttpError::bad_request(
            "BTC strategy and feature schema versions are compiled identities and cannot be overridden",
        ));
    }
    Ok(strategy)
}

fn validate_directional_model_entry_policy(
    strategy: &BtcStrategyConfig,
    entry_policy: BtcDirectionalModelEntryPolicy,
) -> Result<(), HttpError> {
    if entry_policy == BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction
        && !matches!(
            strategy.decision_strategy.as_ref(),
            Some(BtcDecisionStrategyConfig::BtcDirectionalModel { .. })
        )
    {
        return Err(HttpError::bad_request(
            "paper.directional_model_entry_policy execute_directional_prediction requires the BTC directional-model strategy",
        ));
    }
    Ok(())
}

fn validate_btc_entry_timing(strategy: &BtcStrategyConfig) -> Result<(), HttpError> {
    let fixed_120_directional_model = matches!(
        strategy.decision_strategy.as_ref(),
        Some(BtcDecisionStrategyConfig::BtcDirectionalModel { .. })
    ) && strategy.min_seconds_after_open == 120
        && strategy.min_seconds_before_close == 180;
    let timing_valid = strategy.min_seconds_after_open >= 0
        && strategy.min_seconds_before_close > 0
        && strategy
            .min_seconds_after_open
            .checked_add(strategy.min_seconds_before_close)
            .is_some_and(|entry_gate_seconds| {
                entry_gate_seconds < 300
                    || (entry_gate_seconds == 300 && fixed_120_directional_model)
            });
    if !timing_valid {
        return Err(HttpError::bad_request(
            "BTC entry timing gates leave no tradable portion of a five-minute window",
        ));
    }
    Ok(())
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
    directional_model_entry_policy: BtcDirectionalModelEntryPolicy,
    stress_previews: Vec<BtcProcessPaperPreviewControl>,
}

impl Default for BtcProcessPaperControl {
    fn default() -> Self {
        Self {
            arrival_latency_ms: 150,
            visible_depth_haircut: dec!(0.80),
            starting_collateral_usd: dec!(1000),
            directional_model_entry_policy:
                BtcDirectionalModelEntryPolicy::RequirePositiveDirectEdge,
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
    risk_strategies: Vec<RiskStrategySelection>,
    runtime: BtcRuntimeConfig,
    paper_venue: PaperVenueConfig,
    paper_stress_previews: Vec<PaperPreviewConfig>,
}

struct PreparedBtcStartDefinition {
    run_id: uuid::Uuid,
    run_key: String,
    preregistration_sha256: String,
    strategy: BtcStrategyConfig,
    sources: Vec<SourceSelector>,
    entry_admission: Option<BtcEntryAdmissionConfig>,
    risk_strategies: Vec<RiskStrategySelection>,
    directional_model_entry_policy: BtcDirectionalModelEntryPolicy,
    runtime: BtcRuntimeConfig,
    paper_venue: PaperVenueConfig,
    paper_stress_previews: Vec<PaperPreviewConfig>,
    execution_mode: BtcExecutionMode,
    execution: EffectiveProcessExecutionConfig,
    frozen_process_config: TradingProcessConfig,
    config_hash: String,
}

fn merge_source_selectors(
    selectors: impl IntoIterator<Item = SourceSelector>,
) -> Result<Vec<SourceSelector>, HttpError> {
    let mut by_key: BTreeMap<String, SourceSelector> = BTreeMap::new();
    for selector in selectors {
        if let Some(existing) = by_key.get(&selector.key) {
            // An optional consumer must not change an established required
            // subscription. Adapter freshness checks remain process-local.
            if existing.contract_version == selector.contract_version
                && existing.required != selector.required
            {
                if selector.required {
                    by_key.insert(selector.key.clone(), selector);
                }
                continue;
            }
            if existing != &selector {
                return Err(HttpError::conflict(format!(
                    "BTC source {} has incompatible selector settings across active processes",
                    selector.key
                )));
            }
            continue;
        }
        by_key.insert(selector.key.clone(), selector);
    }
    Ok(by_key.into_values().collect())
}

const BTC_LIVE_EXECUTION_FRESHNESS_LIMIT_MS: i64 = 2_000;

fn validate_btc_start_eligibility(process: &TradingProcess) -> Result<(), HttpError> {
    validate_btc_process_capability(process)?;
    validate_btc_execution_activation(process)?;
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

fn validate_btc_process_capability(process: &TradingProcess) -> Result<(), HttpError> {
    if !BtcProcessManager::is_managed_process(process) {
        return Err(HttpError::bad_request(
            "process is not a managed BTC realtime execution process",
        ));
    }
    let execution = process.effective_execution();
    validate_optional_execution_controls(&execution)?;
    match execution.mode.as_str() {
        "paper" => {
            if execution.live_capital {
                return Err(HttpError::bad_request(
                    "BTC paper execution cannot enable live capital",
                ));
            }
            if !execution.execute_signals {
                return Err(HttpError::bad_request(
                    "BTC paper execution requires execution.execute_signals=true",
                ));
            }
            if execution.account_ref.is_some() {
                return Err(HttpError::bad_request(
                    "BTC paper execution cannot select a live account_ref",
                ));
            }
        }
        "live" => {
            let account_ref = execution
                .account_ref
                .as_deref()
                .map(str::trim)
                .unwrap_or("");
            if account_ref.is_empty()
                || account_ref.len() > 128
                || !account_ref.bytes().all(|byte| {
                    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':')
                })
            {
                return Err(HttpError::bad_request(
                    "BTC live execution requires a bounded account_ref slug",
                ));
            }
            if execution.execute_signals != execution.live_capital {
                return Err(HttpError::bad_request(
                    "BTC live execution must enable execute_signals and live_capital together",
                ));
            }
        }
        _ => {
            return Err(HttpError::bad_request(
                "BTC execution.mode must be paper or live",
            ));
        }
    }
    Ok(())
}

fn validate_optional_execution_controls(
    execution: &EffectiveProcessExecutionConfig,
) -> Result<(), HttpError> {
    for (name, value, maximum) in [
        (
            "max_order_notional_usd",
            execution.max_order_notional_usd,
            dec!(5),
        ),
        (
            "max_open_notional_usd",
            execution.max_open_notional_usd,
            dec!(30),
        ),
        ("max_daily_loss_usd", execution.max_daily_loss_usd, dec!(10)),
    ] {
        if value.is_some_and(|value| value <= Decimal::ZERO || value > maximum) {
            return Err(HttpError::bad_request(format!(
                "BTC execution.{name} must be greater than zero and at most {maximum}"
            )));
        }
    }
    if execution
        .max_open_positions
        .is_some_and(|value| value == 0 || value > 6)
    {
        return Err(HttpError::bad_request(
            "BTC execution.max_open_positions must be between 1 and 6",
        ));
    }
    Ok(())
}

fn validate_btc_execution_activation(process: &TradingProcess) -> Result<(), HttpError> {
    let execution = process.effective_execution();
    match execution.mode.as_str() {
        "paper" if execution.execute_signals && !execution.live_capital => Ok(()),
        "live" if execution.execute_signals && execution.live_capital => Ok(()),
        "paper" => Err(HttpError::bad_request(
            "BTC paper start requires execution.execute_signals=true",
        )),
        "live" => Err(HttpError::bad_request(
            "BTC live start requires execution.execute_signals=true and execution.live_capital=true; credential-only definitions cannot start",
        )),
        _ => Err(HttpError::bad_request(
            "BTC execution.mode must be paper or live",
        )),
    }
}

fn validate_btc_live_model_authorization(strategy: &BtcStrategyConfig) -> Result<(), HttpError> {
    let (model_key, artifact_sha256, feature_schema_sha256, require_asymmetric_value) =
        match strategy.decision_strategy.as_ref() {
            Some(BtcDecisionStrategyConfig::BtcDirectionalModel {
                model_key,
                artifact_sha256,
                feature_schema_sha256,
            }) => (model_key, artifact_sha256, feature_schema_sha256, false),
            Some(BtcDecisionStrategyConfig::BtcAsymmetricValueModel {
                model_key,
                artifact_sha256,
                feature_schema_sha256,
            }) => (model_key, artifact_sha256, feature_schema_sha256, true),
            _ => {
                return Err(HttpError::bad_request(
                    "BTC live execution currently requires an immutable model artifact",
                ));
            }
        };
    let model = runtime_model(&RuntimeModelSelection {
        model_key: model_key.clone(),
        artifact_sha256: artifact_sha256.clone(),
        feature_schema_sha256: feature_schema_sha256.clone(),
    })
    .map_err(|error| HttpError::bad_request(format!("invalid live model artifact: {error}")))?;
    if require_asymmetric_value && !model.is_asymmetric_value() {
        return Err(HttpError::bad_request(
            "BTC asymmetric-value live execution requires an asymmetric-value artifact",
        ));
    }
    if !model.live_capital_allowed() {
        return Err(HttpError::conflict(format!(
            "model {} is not authorized for live capital (deployment_scope={}, production_qualified={})",
            model.model_key(),
            model.deployment_scope().unwrap_or("unspecified"),
            model.production_qualified()
        )));
    }
    Ok(())
}

fn validate_btc_live_execution_freshness(strategy: &BtcStrategyConfig) -> Result<(), HttpError> {
    if strategy.max_reference_age_ms > BTC_LIVE_EXECUTION_FRESHNESS_LIMIT_MS
        || strategy.max_book_age_ms > BTC_LIVE_EXECUTION_FRESHNESS_LIMIT_MS
    {
        return Err(HttpError::bad_request(format!(
            "BTC live execution requires max_reference_age_ms and max_book_age_ms at or below {BTC_LIVE_EXECUTION_FRESHNESS_LIMIT_MS}"
        )));
    }
    Ok(())
}

#[cfg(test)]
fn prepare_btc_start_definition(
    resolved: ResolvedBtcProcessDefinition,
) -> Result<PreparedBtcStartDefinition, HttpError> {
    prepare_btc_start_definition_for_execution(
        resolved,
        &EffectiveProcessExecutionConfig {
            mode: "paper".to_string(),
            execute_signals: true,
            live_capital: false,
            account_ref: None,
            taker_fee_rate: dec!(0.03),
            max_order_notional_usd: None,
            max_open_notional_usd: None,
            max_open_positions: None,
            max_daily_loss_usd: None,
            require_exit_book: None,
        },
    )
}

fn prepare_btc_start_definition_for_execution(
    resolved: ResolvedBtcProcessDefinition,
    execution: &EffectiveProcessExecutionConfig,
) -> Result<PreparedBtcStartDefinition, HttpError> {
    let ResolvedBtcProcessDefinition {
        control,
        strategy,
        entry_admission,
        risk_strategies,
        runtime,
        paper_venue,
        paper_stress_previews,
    } = resolved;
    let directional_model_entry_policy = control.paper.directional_model_entry_policy;
    let sources = control.sources.clone();
    risk_runtime::validate_selections(&risk_strategies)
        .map_err(|error| HttpError::bad_request(error.to_string()))?;
    if let Some(binding) = &strategy.unified_model {
        for input in &binding.sources {
            if !sources.iter().any(|source| source.key == input.product) {
                return Err(HttpError::bad_request(format!(
                    "UMR binding {} requires process stream {}",
                    input.slot, input.product
                )));
            }
        }
    }
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
    let run_key = control.next_experiment_key;
    let preregistration_sha256 = control.preregistration_sha256;
    let execution_mode = match execution.mode.as_str() {
        "paper" => BtcExecutionMode::Paper,
        "live" => BtcExecutionMode::Live,
        _ => {
            return Err(HttpError::bad_request(
                "BTC execution.mode must be paper or live",
            ))
        }
    };
    if execution_mode == BtcExecutionMode::Live {
        validate_btc_live_execution_freshness(&strategy)?;
    }
    let run_namespace = match execution_mode {
        BtcExecutionMode::Paper => "polymarket-bot/btc-paper",
        BtcExecutionMode::Live => "polymarket-bot/btc-live",
    };
    let run_id = uuid::Uuid::new_v5(
        &uuid::Uuid::NAMESPACE_URL,
        format!("{run_namespace}/{run_key}").as_bytes(),
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
        "playbook_version": "v1.2",
        "sources": &sources,
        "runtime": &runtime,
        "paper": {
            "execution_enabled": true,
            "venue": &paper_venue,
            "stress_previews": &paper_stress_previews,
        }
    });
    if directional_model_entry_policy != BtcDirectionalModelEntryPolicy::RequirePositiveDirectEdge {
        frozen_raw["paper"]
            .as_object_mut()
            .expect("BTC frozen paper config is an object")
            .insert(
                "directional_model_entry_policy".to_string(),
                serde_json::to_value(directional_model_entry_policy)
                    .map_err(|error| HttpError::internal(error.to_string()))?,
            );
    }
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
    if !risk_strategies.is_empty() {
        frozen_raw
            .as_object_mut()
            .expect("BTC frozen process config is an object")
            .insert(
                "risk_strategies".to_string(),
                serde_json::to_value(&risk_strategies)
                    .map_err(|error| HttpError::internal(error.to_string()))?,
            );
    }
    if execution_mode == BtcExecutionMode::Live {
        frozen_raw["paper"]["execution_enabled"] = serde_json::Value::Bool(false);
        frozen_raw
            .as_object_mut()
            .expect("BTC frozen process config is an object")
            .insert(
                "live".to_string(),
                serde_json::json!({
                    "execution_enabled": true,
                    "account_ref": execution.account_ref,
                }),
            );
    }
    let frozen_process_config = TradingProcessConfig {
        execution: Some(ProcessExecutionConfig {
            mode: Some(execution_mode.as_str().to_string()),
            execute_signals: execution.execute_signals,
            live_capital: execution.live_capital,
            account_ref: execution.account_ref.clone(),
            taker_fee_rate: None,
            max_order_notional_usd: execution.max_order_notional_usd,
            max_open_notional_usd: execution.max_open_notional_usd,
            max_open_positions: execution.max_open_positions,
            max_daily_loss_usd: execution.max_daily_loss_usd,
            require_exit_book: execution.require_exit_book,
        }),
        raw: frozen_raw,
        ..TradingProcessConfig::default()
    };
    let config_hash = hash_btc_frozen_process_config(&frozen_process_config)?;
    Ok(PreparedBtcStartDefinition {
        run_id,
        run_key,
        preregistration_sha256,
        strategy,
        sources,
        entry_admission,
        risk_strategies,
        directional_model_entry_policy,
        runtime,
        paper_venue,
        paper_stress_previews,
        execution_mode,
        execution: execution.clone(),
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
            for retired_system_field in [
                "gamma_base_url",
                "clob_rest_base_url",
                "clob_ws_url",
                "rtds_ws_url",
                "binance_ws_url",
                "binance_rest_base_url",
                "discovery_interval",
                "reconnect_initial_delay",
                "reconnect_max_delay",
                "checkpoint_interval",
                "boundary_tick_max_delay",
                "official_resolution_audit_grace",
                "official_resolution_watch_retention",
                "writer_capacity",
                "clob_heartbeat_interval",
                "rtds_heartbeat_interval",
                "binance_heartbeat_interval",
                "binance_spot_l2_enabled",
                "binance_spot_l2_ws_url",
            ] {
                runtime.remove(retired_system_field);
            }
        }
        raw.remove("playbook_version");
        raw.remove("sources");
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

fn selector_only_config_change(
    current: &TradingProcessConfig,
    candidate: &TradingProcessConfig,
) -> Result<bool, HttpError> {
    fn remove_selectors(value: &mut serde_json::Value) {
        if let Some(control) = value
            .get_mut("raw")
            .and_then(|raw| raw.get_mut("btc_realtime_paper"))
            .and_then(serde_json::Value::as_object_mut)
        {
            control.remove("playbook_version");
            control.remove("sources");
        }
    }
    let mut current =
        serde_json::to_value(current).map_err(|error| HttpError::internal(error.to_string()))?;
    let mut candidate =
        serde_json::to_value(candidate).map_err(|error| HttpError::internal(error.to_string()))?;
    remove_selectors(&mut current);
    remove_selectors(&mut candidate);
    Ok(current == candidate)
}

struct ActiveBtcPlaybook {
    process_id: uuid::Uuid,
    run_id: uuid::Uuid,
    run_key: String,
    config_hash: String,
    execution_mode: BtcExecutionMode,
    strategy: BtcStrategyConfig,
    sources: Vec<SourceSelector>,
    live_venue: Option<Arc<LiveVenue>>,
    runtime: BtcPlaybookRuntimeHandle,
}

struct BtcExecutionComponents {
    venue: Arc<dyn ExecutionVenue>,
    lifecycle: Arc<dyn BtcExecutionLifecycle>,
    live_venue: Option<Arc<LiveVenue>>,
}

struct SharedBtcRuntime {
    config: BtcRuntimeConfig,
    state: Arc<tokio::sync::RwLock<polymarket_bot::btc::RealtimeState>>,
    books: Arc<tokio::sync::RwLock<BookRegistry>>,
    runtime: Option<BtcRuntimeHandle>,
}

#[derive(Debug, Clone)]
struct PendingBtcTerminal {
    process_id: uuid::Uuid,
    run_id: uuid::Uuid,
    run_key: String,
    config_hash: String,
    terminal_status: String,
    terminal_reason: String,
    allow_inactive_process: bool,
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
    shared_runtime_startup: Arc<tokio::sync::Mutex<()>>,
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
            shared_runtime_startup: Arc::new(tokio::sync::Mutex::new(())),
            terminal_pending: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    fn execution_components(
        &self,
        execution_mode: BtcExecutionMode,
        execution: &EffectiveProcessExecutionConfig,
        process_id: uuid::Uuid,
        books: Arc<tokio::sync::RwLock<BookRegistry>>,
        paper_venue_config: PaperVenueConfig,
        strategy: &BtcStrategyConfig,
    ) -> Result<BtcExecutionComponents> {
        let max_directional_feature_age = strategy
            .effective_max_directional_feature_age_ms()?
            .map(chrono::Duration::milliseconds);
        match execution_mode {
            BtcExecutionMode::Paper => {
                let paper_venue = Arc::new(
                    BtcPaperVenue::new_with_reference_execution_guard_and_controls(
                        books,
                        paper_venue_config,
                        strategy.max_depth_participation,
                        execution.clone(),
                        process_id,
                        chrono::Duration::milliseconds(strategy.max_reference_age_ms),
                        max_directional_feature_age,
                    )?,
                );
                let venue: Arc<dyn ExecutionVenue> = paper_venue.clone();
                let lifecycle: Arc<dyn BtcExecutionLifecycle> =
                    Arc::new(PaperExecutionLifecycle::new(paper_venue));
                Ok(BtcExecutionComponents {
                    venue,
                    lifecycle,
                    live_venue: None,
                })
            }
            BtcExecutionMode::Live => {
                execution
                    .account_ref
                    .as_deref()
                    .context("live BTC execution is missing account_ref")?;
                let global = self
                    .config
                    .live_venue
                    .as_ref()
                    .context("live BTC execution credentials are not configured")?;
                let live_venue = Arc::new(global.bind_process(process_id, execution)?);
                let delegate: Arc<dyn ExecutionVenue> = live_venue.clone();
                let venue: Arc<dyn ExecutionVenue> = Arc::new(BtcLiveExecutionAdapter::new(
                    delegate,
                    books,
                    process_id,
                    chrono::Duration::milliseconds(strategy.max_reference_age_ms),
                    max_directional_feature_age,
                    chrono::Duration::milliseconds(strategy.max_book_age_ms),
                    strategy.max_depth_participation,
                    execution.require_exit_book.unwrap_or(false),
                )?);
                let lifecycle: Arc<dyn BtcExecutionLifecycle> =
                    Arc::new(LiveExecutionLifecycle::new(
                        venue.clone(),
                        self.config.live_reconcile_interval,
                    )?);
                Ok(BtcExecutionComponents {
                    venue,
                    lifecycle,
                    live_venue: Some(live_venue),
                })
            }
        }
    }

    async fn ensure_shared_runtime(
        &self,
        config: &BtcRuntimeConfig,
        sources: &[SourceSelector],
    ) -> Result<
        (
            Arc<tokio::sync::RwLock<polymarket_bot::btc::RealtimeState>>,
            Arc<tokio::sync::RwLock<BookRegistry>>,
        ),
        HttpError,
    > {
        // Durable processes resume concurrently after a container restart.
        // Serialize only shared transport initialization so they converge on
        // one selector-union consumer instead of racing to create one each.
        let _startup_guard = self.shared_runtime_startup.lock().await;
        let active_guard = self.active_playbooks.lock().await;
        let active_playbooks = active_guard.len();
        let source_union = merge_source_selectors(
            active_guard
                .values()
                .flat_map(|playbook| playbook.sources.clone())
                .chain(sources.iter().cloned()),
        )?;
        drop(active_guard);
        let retired_runtime = {
            let mut shared = self.shared_runtime.lock().await;
            if let Some(existing) = shared.as_ref() {
                let compatible = shared_market_data_config_compatible(&existing.config, config);
                if compatible
                    && existing
                        .runtime
                        .as_ref()
                        .is_some_and(BtcRuntimeHandle::is_running)
                {
                    if let Some(runtime) = existing.runtime.as_ref() {
                        runtime.update_sources(source_union.clone());
                    }
                    return Ok((existing.state.clone(), existing.books.clone()));
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
            if let Some(runtime) = retired.runtime {
                match tokio::time::timeout(BTC_RUNTIME_SHUTDOWN_TIMEOUT, runtime.shutdown()).await {
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
        }

        let state = Arc::new(tokio::sync::RwLock::new(
            polymarket_bot::btc::RealtimeState::default(),
        ));
        let books = Arc::new(tokio::sync::RwLock::new(BookRegistry::new(
            uuid::Uuid::new_v4(),
        )));
        let runtime = BtcRuntime::new(config.clone(), self.repository.clone())
            .with_sources(source_union)
            .with_shared_state(state.clone())
            .with_shared_book_registry(books.clone())
            .start()
            .await
            .map_err(|error| {
                HttpError::internal(format!(
                    "failed to start shared BTC market-data runtime: {error:#}"
                ))
            })?;
        info!("BTC shared market-data gRPC runtime started");
        let mut shared = self.shared_runtime.lock().await;
        debug_assert!(shared.is_none());
        *shared = Some(SharedBtcRuntime {
            config: config.clone(),
            state: state.clone(),
            books: books.clone(),
            runtime: Some(runtime),
        });
        Ok((state, books))
    }

    async fn refresh_shared_sources(&self) -> Result<(), HttpError> {
        let source_union = merge_source_selectors(
            self.active_playbooks
                .lock()
                .await
                .values()
                .flat_map(|playbook| playbook.sources.clone()),
        )?;
        if source_union.is_empty() {
            return Ok(());
        }
        if let Some(runtime) = self
            .shared_runtime
            .lock()
            .await
            .as_ref()
            .and_then(|shared| shared.runtime.as_ref())
        {
            runtime.update_sources(source_union);
        }
        Ok(())
    }

    async fn recover_shared_runtime(&self) {
        if self.shutting_down.load(Ordering::Acquire) {
            return;
        }
        let _transition_guard = self.transition.lock().await;
        if self.shutting_down.load(Ordering::Acquire)
            || self.active_playbooks.lock().await.is_empty()
        {
            return;
        }

        let (config, state, books, failed_runtime) = {
            let mut shared_guard = self.shared_runtime.lock().await;
            let Some(shared) = shared_guard.as_mut() else {
                error!(
                    "BTC shared market-data runtime handle is unavailable; preserving active process configuration"
                );
                return;
            };
            if shared
                .runtime
                .as_ref()
                .is_some_and(BtcRuntimeHandle::is_running)
            {
                return;
            }
            (
                shared.config.clone(),
                shared.state.clone(),
                shared.books.clone(),
                shared.runtime.take(),
            )
        };

        if let Some(runtime) = failed_runtime {
            match tokio::time::timeout(BTC_RUNTIME_SHUTDOWN_TIMEOUT, runtime.shutdown()).await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => warn!(
                    error = ?error,
                    "failed BTC shared market-data runtime reported an integrity failure during recovery"
                ),
                Err(_) => warn!(
                    timeout_secs = BTC_RUNTIME_SHUTDOWN_TIMEOUT.as_secs(),
                    "timed out retiring failed BTC shared market-data runtime during recovery"
                ),
            }
        }

        invalidate_shared_market_data_evidence(&state, &books).await;

        let sources = self
            .active_playbooks
            .lock()
            .await
            .values()
            .flat_map(|playbook| playbook.sources.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let recovery = BtcRuntime::new(config.clone(), self.repository.clone())
            .with_sources(sources)
            .with_shared_state(state)
            .with_shared_book_registry(books)
            .start()
            .await;
        match recovery {
            Ok(runtime) => {
                let mut shared_guard = self.shared_runtime.lock().await;
                let Some(shared) = shared_guard.as_mut() else {
                    warn!("recovered BTC shared market-data runtime lost its manager slot");
                    drop(runtime);
                    return;
                };
                shared.runtime = Some(runtime);
                drop(shared_guard);
                info!(
                    active_process_count = self.active_playbooks.lock().await.len(),
                    "BTC shared market-data runtime recovered without changing durable process state"
                );
            }
            Err(error) => warn!(
                error = ?error,
                "BTC shared market-data runtime recovery deferred; active process configuration remains enabled"
            ),
        }
    }

    async fn shutdown_shared_runtime_if_idle(&self) {
        if !self.active_playbooks.lock().await.is_empty() {
            return;
        }
        let shared = { self.shared_runtime.lock().await.take() };
        let Some(shared) = shared else {
            return;
        };
        let Some(runtime) = shared.runtime else {
            return;
        };
        match tokio::time::timeout(BTC_RUNTIME_SHUTDOWN_TIMEOUT, runtime.shutdown()).await {
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

    fn validate_inactive_definition(
        &self,
        process: &TradingProcess,
    ) -> Result<ResolvedBtcProcessDefinition, HttpError> {
        self.validate_definition(process, BtcDefinitionUse::InactiveDefinition)
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
        match definition_use {
            BtcDefinitionUse::ExplicitStart => validate_btc_start_eligibility(process)?,
            BtcDefinitionUse::DurableResume => {
                validate_btc_process_capability(process)?;
                validate_btc_execution_activation(process)?;
            }
            BtcDefinitionUse::InactiveDefinition => validate_btc_process_capability(process)?,
        }
        if process
            .config
            .execution
            .as_ref()
            .and_then(|execution| execution.taker_fee_rate)
            .is_some()
        {
            return Err(HttpError::bad_request(
                "BTC realtime execution config accepts only execution safety fields and raw.btc_realtime_paper settings",
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
        if process.effective_execution().mode == "live"
            && definition_use != BtcDefinitionUse::InactiveDefinition
        {
            validate_btc_live_model_authorization(&strategy)?;
        }
        validate_directional_model_entry_policy(
            &strategy,
            control.paper.directional_model_entry_policy,
        )?;
        if let Some(entry_admission) = control.entry_admission.as_ref() {
            entry_admission
                .validate()
                .map_err(|error| HttpError::bad_request(error.to_string()))?;
        }
        risk_runtime::validate_selections(&control.risk_strategies)
            .map_err(|error| HttpError::bad_request(error.to_string()))?;
        validate_btc_entry_timing(&strategy)?;
        if !(1..=60_000).contains(&control.runtime.strategy_interval_ms) {
            return Err(HttpError::bad_request(
                "BTC runtime timings exceed the supported process safety bounds",
            ));
        }
        let runtime = BtcRuntimeConfig {
            enabled: true,
            strategy_interval: Duration::from_millis(control.runtime.strategy_interval_ms),
            max_book_age: Duration::from_millis(strategy.max_book_age_ms as u64),
            max_reference_age: Duration::from_millis(strategy.max_reference_age_ms as u64),
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
            risk_strategies: control.risk_strategies.clone(),
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
        prepare_btc_start_definition_for_execution(
            self.validate_start_definition(process)?,
            &process.effective_execution(),
        )
    }

    fn prepare_resume_definition(
        &self,
        process: &TradingProcess,
    ) -> Result<PreparedBtcStartDefinition, HttpError> {
        prepare_btc_start_definition_for_execution(
            self.validate_resume_definition(process)?,
            &process.effective_execution(),
        )
    }

    async fn live_preflight(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcessLivePreflightResponse, HttpError> {
        let process = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        if process.enabled || matches!(process.status.as_str(), "starting" | "running" | "stopping")
        {
            return Err(HttpError::conflict(
                "credential preflight requires an inactive trading process",
            ));
        }
        self.validate_inactive_definition(&process)?;
        let execution = process.effective_execution();
        if execution.mode != "live" {
            return Err(HttpError::bad_request(
                "process-scoped live preflight requires execution.mode=live",
            ));
        }
        let account_ref = execution
            .account_ref
            .as_deref()
            .ok_or_else(|| HttpError::internal("validated live process is missing account_ref"))?
            .to_string();
        let global = self.config.live_venue.as_ref().cloned().ok_or_else(|| {
            HttpError::bad_request("live execution credentials are not configured")
        })?;
        let venue = Arc::new(
            global
                .bind_process(process_id, &execution)
                .map_err(|error| HttpError::bad_request(error.to_string()))?,
        );
        let identity = venue
            .live_identity_diagnostics()
            .await
            .map_err(|error| HttpError::bad_request(error.to_string()))?;
        let reconciliation_result = venue.reconcile().await;
        let (reconciliation, reconciliation_error) = match reconciliation_result {
            Ok(report) => (Some(report), None),
            Err(error) => {
                let mut message = error.to_string();
                message.truncate(512);
                (None, Some(message))
            }
        };
        let status = venue
            .live_status()
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        let credential_connectivity_ready = identity.credentials_present
            && identity.account_identity_valid
            && identity.account_identity_fingerprint_sha256.is_some()
            && identity.api_keys_readable
            && identity.balance_allowance_readable
            && identity.balance_allowance_error.is_none()
            && identity
                .collateral_balance
                .as_deref()
                .and_then(|balance| balance.parse::<Decimal>().ok())
                .is_some_and(|balance| balance > Decimal::ZERO)
            && identity.open_orders_readable
            && identity.signer_address.is_some()
            && identity.configured_funder_address.is_some()
            && identity.resolved_signature_type.is_some()
            && identity.authenticated_client_address.is_some();
        let reconciliation_ready = reconciliation.as_ref().is_some_and(|report| {
            report.balances_checked
                && report.mismatches_found == 0
                && report.unresolved_count == 0
                && status.process_accounting_proven
        });
        let trading_disabled = !execution.execute_signals
            && !execution.live_capital
            && !status.order_submit_enabled
            && !status.entries_enabled;
        let mut reasons = Vec::with_capacity(3);
        if !credential_connectivity_ready {
            reasons.push("credential_connectivity_failed".to_string());
        }
        if !reconciliation_ready {
            reasons.push("process_reconciliation_not_clean".to_string());
        }
        if !trading_disabled {
            reasons.push("credential_preflight_requires_trading_disabled".to_string());
        }
        let ready = reasons.is_empty();
        let checked_at = Utc::now();
        self.record_event(
            process_id,
            if ready { "info" } else { "warn" },
            "btc_live_preflight_completed",
            "BTC live credential and reconciliation preflight completed without order submission",
            serde_json::json!({
                "account_ref": &account_ref,
                "ready": ready,
                "credential_connectivity_ready": credential_connectivity_ready,
                "reconciliation_ready": reconciliation_ready,
                "trading_disabled": trading_disabled,
                "reasons": &reasons,
                "checked_at": checked_at,
            }),
        )
        .await;
        Ok(TradingProcessLivePreflightResponse {
            process_id,
            account_ref,
            credential_connectivity_ready,
            reconciliation_ready,
            trading_disabled,
            ready,
            reasons,
            identity,
            status,
            reconciliation,
            reconciliation_error,
            checked_at,
        })
    }

    async fn set_live_entries_enabled(
        &self,
        process_id: uuid::Uuid,
        enabled: bool,
    ) -> Result<LiveVenueStatus, HttpError> {
        let _transition_guard = self.transition.lock().await;
        self.set_live_entries_enabled_locked(process_id, enabled)
            .await
    }

    async fn set_live_entries_enabled_locked(
        &self,
        process_id: uuid::Uuid,
        enabled: bool,
    ) -> Result<LiveVenueStatus, HttpError> {
        let (venue, active_run_id, active_run_key, active_config_hash) = {
            let active = self.active_playbooks.lock().await;
            let active = active
                .get(&process_id)
                .filter(|active| active.execution_mode == BtcExecutionMode::Live)
                .ok_or_else(|| HttpError::conflict("active live process runtime not found"))?;
            let venue = active
                .live_venue
                .clone()
                .ok_or_else(|| HttpError::internal("active live runtime omitted its venue"))?;
            (
                venue,
                active.run_id,
                active.run_key.clone(),
                active.config_hash.clone(),
            )
        };
        if enabled {
            let process = self
                .store
                .get_trading_process(process_id)
                .await
                .map_err(|error| HttpError::internal(error.to_string()))?
                .ok_or_else(|| HttpError::not_found("trading process not found"))?;
            if !process.enabled || process.status != "running" {
                return Err(HttpError::conflict(
                    "live entries require the exact active process to be enabled and running",
                ));
            }
            let prepared = self.prepare_resume_definition(&process)?;
            if prepared.run_id != active_run_id || prepared.run_key != active_run_key {
                return Err(HttpError::conflict(
                    "live process definition no longer matches the active run identity",
                ));
            }
            let manifest = self
                .repository
                .load_run_manifest(process_id, active_run_id, &active_run_key)
                .await
                .map_err(|error| HttpError::internal(error.to_string()))?
                .ok_or_else(|| HttpError::conflict("active live run manifest is missing"))?;
            if manifest.config_hash != active_config_hash
                || resume_process_contract_projection(
                    serde_json::to_value(&prepared.frozen_process_config)
                        .map_err(|error| HttpError::internal(error.to_string()))?,
                ) != resume_process_contract_projection(manifest.frozen_process_config)
            {
                return Err(HttpError::conflict(
                    "live process definition or manifest drifted from the active runtime",
                ));
            }
            let active_still_matches = self
                .active_playbooks
                .lock()
                .await
                .get(&process_id)
                .is_some_and(|active| {
                    active.execution_mode == BtcExecutionMode::Live
                        && active.run_id == active_run_id
                        && active.config_hash == active_config_hash
                        && active.live_venue.is_some()
                });
            if !active_still_matches {
                return Err(HttpError::conflict(
                    "active live runtime changed during entry-enable validation",
                ));
            }
        }
        // The venue owns the authoritative close/drain/reconcile/generation-CAS sequence.
        // Bound the whole operation instead of performing an earlier, non-authoritative
        // reconciliation that could go stale before the gate is reopened.
        let status = tokio::time::timeout(
            BTC_LIVE_CONTROL_TIMEOUT,
            venue.set_live_entries_enabled(
                enabled,
                (!enabled).then(|| "process_manual_disable".to_string()),
            ),
        )
        .await
        .map_err(|_| {
            HttpError::conflict(format!(
                "live entry transition timed out after {}s; entries remain fail-closed",
                BTC_LIVE_CONTROL_TIMEOUT.as_secs()
            ))
        })?
        .map_err(|error| HttpError::conflict(error.to_string()))?;
        if enabled && !status.entries_enabled {
            let _ = venue
                .set_live_entries_enabled(
                    false,
                    Some("entry_enable_preconditions_failed".to_string()),
                )
                .await;
            return Err(HttpError::conflict(format!(
                "live entry enable preconditions failed: {}",
                status.reason.as_deref().unwrap_or("unknown")
            )));
        }
        self.record_event(
            process_id,
            "warn",
            if enabled {
                "btc_live_entries_enabled"
            } else {
                "btc_live_entries_disabled"
            },
            if enabled {
                "BTC live entries manually enabled after strict reconciliation"
            } else {
                "BTC live entries manually disabled"
            },
            serde_json::json!({
                "entries_enabled": status.entries_enabled,
                "reason": &status.reason,
                "source": "manual_process_control",
            }),
        )
        .await;
        Ok(status)
    }

    async fn ensure_start_slot_available(&self, process_id: uuid::Uuid) -> Result<(), HttpError> {
        if let Some(pending) = self.terminal_pending.lock().await.get(&process_id) {
            return Err(HttpError::conflict(format!(
                "BTC run {} still has a pending terminal transition",
                pending.run_key
            )));
        }
        if let Some(active) = self.active_playbooks.lock().await.get(&process_id) {
            return Err(HttpError::conflict(format!(
                "BTC process {} is already running execution run {}",
                active.process_id, active.run_key
            )));
        }
        Ok(())
    }

    async fn quiesce_live_venue(venue: Option<&Arc<LiveVenue>>, reason: &str) -> Option<String> {
        let Some(venue) = venue else {
            return None;
        };
        let quiesce = async {
            let mut failures = Vec::with_capacity(3);
            if let Err(error) = venue
                .set_live_entries_enabled(false, Some(reason.to_string()))
                .await
            {
                failures.push(format!("live_entry_disable_failed: {error:#}"));
            }
            if let Err(error) = venue.cancel_all().await {
                failures.push(format!("live_order_cancel_failed: {error:#}"));
            }
            match venue.reconcile().await {
                Ok(report) if report.open_orders == 0 && report.unresolved_count == 0 => {}
                Ok(report) => warn!(
                    open_orders = report.open_orders,
                    unresolved_count = report.unresolved_count,
                    "live stop reconciliation remains unresolved without changing terminal intent"
                ),
                Err(error) => warn!(
                    error = %error,
                    "live stop reconciliation failed without changing terminal intent"
                ),
            }
            failures
        };
        match tokio::time::timeout(BTC_LIVE_QUIESCE_TIMEOUT, quiesce).await {
            Ok(failures) => (!failures.is_empty()).then(|| failures.join("; ")),
            Err(_) => Some(format!(
                "live_quiesce_timeout_after_{}s",
                BTC_LIVE_QUIESCE_TIMEOUT.as_secs()
            )),
        }
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
        preflight_btc_run_identity(&self.repository, &prepared.run_key, prepared.run_id)
            .await
            .map_err(|error| HttpError::conflict(error.to_string()))?;
        Ok(TradingProcessStartPreviewResponse {
            process_id,
            run_id: prepared.run_id,
            run_key: prepared.run_key,
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
            JOIN LATERAL (
              SELECT event.timestamp_utc
              FROM polymarket.trading_process_events event
              WHERE event.process_id = p.process_id
                AND event.event_type = 'btc_run_manifest'
                AND event.metadata #>> '{run_key}' =
                    p.config #>> '{raw,btc_realtime_paper,next_experiment_key}'
              ORDER BY event.timestamp_utc DESC, event.created_at DESC
              LIMIT 1
            ) run ON true
            WHERE p.process_type = 'btc_5m'
              AND p.process_scope = 'realtime_paper'
              AND p.config #>> '{raw,btc_realtime_paper,schema_version}' IN ($1, $2, $3)
              AND p.enabled
              AND p.status IN ('starting','running','stopping')
              AND p.stopped_at IS NULL
            ORDER BY run.timestamp_utc DESC
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
                SELECT COUNT(*)
                FROM polymarket.trading_processes p
                WHERE p.process_type = 'btc_5m'
                  AND p.process_scope = 'realtime_paper'
                  AND p.enabled
                  AND p.status IN ('starting','running','stopping')
                  AND p.stopped_at IS NULL
                "#,
            )
            .fetch_one(&self.pool)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
            if durable_claims > 0 {
                return Err(HttpError::conflict(
                    "durable BTC realtime execution state exists but cannot be reattached exactly; database state was left unchanged",
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
        let manifest = self
            .repository
            .load_run_manifest(process_id, prepared.run_id, &prepared.run_key)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| {
                HttpError::conflict("durable BTC run disappeared before runtime reattachment")
            })?;
        let config_hash = manifest.config_hash;
        let frozen_process_config_value = manifest.frozen_process_config;
        let current_frozen_process_config =
            serde_json::to_value(&prepared.frozen_process_config)
                .map_err(|error| HttpError::internal(error.to_string()))?;
        if resume_process_contract_projection(current_frozen_process_config)
            != resume_process_contract_projection(frozen_process_config_value.clone())
        {
            return Err(HttpError::conflict(
                "durable BTC run parameters changed and cannot be resumed by this process definition",
            ));
        }
        let PreparedBtcStartDefinition {
            run_id,
            run_key,
            preregistration_sha256,
            strategy,
            sources,
            entry_admission,
            risk_strategies,
            directional_model_entry_policy,
            runtime: runtime_config,
            paper_venue: paper_venue_config,
            paper_stress_previews,
            execution_mode,
            execution,
            frozen_process_config: _,
            config_hash: current_config_hash,
        } = prepared;
        self.record_event(
            process_id,
            "info",
            "btc_runtime_resuming",
            "BTC realtime execution runtime resume accepted",
            serde_json::json!({
                "run_id": run_id,
                "run_key": &run_key,
                "preregistration_sha256": &preregistration_sha256,
                "config_hash": &config_hash,
                "current_definition_config_hash": &current_config_hash,
                "compiled_source_identity": COMPILED_SOURCE_IDENTITY,
            }),
        )
        .await;
        let active_strategy = strategy.clone();

        let startup_result: Result<(BtcPlaybookRuntimeHandle, Option<Arc<LiveVenue>>)> = async {
            let (state, books) = self
                .ensure_shared_runtime(&runtime_config, &sources)
                .await
                .map_err(|error| anyhow::anyhow!("{error:?}"))?;
            let components = self.execution_components(
                execution_mode,
                &execution,
                process_id,
                books.clone(),
                paper_venue_config,
                &strategy,
            )?;
            let live_process_venue = components.live_venue.clone();
            if should_resume_configured_live_entries(&process, &execution) {
                live_process_venue
                    .as_ref()
                    .context("configured live resume omitted its process venue")?
                    .restore_configured_entries_after_restart()
                    .await
                    .context("failed to restore configured live authorization before resume")?;
            }
            let process_runner = Arc::new(
                BtcProcessRunner::new_with_execution(
                    self.repository.clone(),
                    self.store.clone(),
                    components.venue,
                    books.clone(),
                    components.lifecycle,
                    BtcProcessConfig {
                        run_id,
                        run_key: run_key.clone(),
                        process_id,
                        config_hash: config_hash.clone(),
                        frozen_process_config: frozen_process_config_value,
                        strategy,
                        entry_admission,
                        risk_strategies,
                        directional_model_entry_policy,
                        execution_enabled: true,
                        paper_stress_previews,
                    },
                )?
                .with_primary_persistence_state(state.clone()),
            );
            process_runner
                .resume()
                .await
                .context("failed to reattach immutable BTC run before feed resume")?;
            Ok((
                BtcPlaybookRuntimeHandle::start(runtime_config, process_runner, state, books)?,
                live_process_venue,
            ))
        }
        .await;
        let (runtime, live_process_venue) = match startup_result {
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
                run_id,
                run_key: run_key.clone(),
                config_hash: config_hash.clone(),
                execution_mode,
                strategy: active_strategy,
                sources,
                live_venue: live_process_venue,
                runtime,
            },
        );
        self.refresh_shared_sources().await?;
        self.record_event(
            process_id,
            "info",
            "btc_runtime_resumed",
            "BTC realtime execution runtime resumed after service restart",
            serde_json::json!({
                "run_id": run_id,
                "run_key": &run_key,
                "config_hash": &config_hash,
                "current_definition_config_hash": &current_config_hash,
                "compiled_source_identity": COMPILED_SOURCE_IDENTITY,
            }),
        )
        .await;
        info!(
            process_id = %process_id,
            run_id = %run_id,
            run_key = %run_key,
            config_hash = %config_hash,
            "durable BTC realtime execution runtime resumed"
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
            run_id,
            run_key,
            preregistration_sha256,
            strategy,
            sources,
            entry_admission,
            risk_strategies,
            directional_model_entry_policy,
            runtime: runtime_config,
            paper_venue: paper_venue_config,
            paper_stress_previews,
            execution_mode,
            execution,
            frozen_process_config,
            config_hash,
        } = self.prepare_start_definition(&process)?;
        preflight_btc_run_identity(&self.repository, &run_key, run_id)
            .await
            .map_err(|error| HttpError::conflict(error.to_string()))?;
        let frozen_process_config_value = serde_json::to_value(&frozen_process_config)
            .map_err(|error| HttpError::internal(error.to_string()))?;

        let start_pending = PendingBtcTerminal {
            process_id,
            run_id,
            run_key: run_key.clone(),
            config_hash: config_hash.clone(),
            terminal_status: "failed".to_string(),
            terminal_reason: "btc_start_transition_interrupted".to_string(),
            allow_inactive_process: true,
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
            "BTC realtime execution runtime start accepted",
            serde_json::json!({
                "run_id": run_id,
                "run_key": &run_key,
                "preregistration_sha256": &preregistration_sha256,
            }),
        )
        .await;

        let startup_result: Result<(BtcPlaybookRuntimeHandle, Option<Arc<LiveVenue>>)> = async {
            let (state, books) = self
                .ensure_shared_runtime(&runtime_config, &sources)
                .await
                .map_err(|error| anyhow::anyhow!("{error:?}"))?;
            let components = self.execution_components(
                execution_mode,
                &execution,
                process_id,
                books.clone(),
                paper_venue_config,
                &strategy,
            )?;
            let live_process_venue = components.live_venue.clone();
            let process_runner = Arc::new(
                BtcProcessRunner::new_with_execution(
                    self.repository.clone(),
                    self.store.clone(),
                    components.venue,
                    books.clone(),
                    components.lifecycle,
                    BtcProcessConfig {
                        run_id,
                        run_key: run_key.clone(),
                        process_id,
                        config_hash: config_hash.clone(),
                        frozen_process_config: frozen_process_config_value,
                        strategy: strategy.clone(),
                        entry_admission: entry_admission.clone(),
                        risk_strategies: risk_strategies.clone(),
                        directional_model_entry_policy,
                        execution_enabled: true,
                        paper_stress_previews: paper_stress_previews.clone(),
                    },
                )?
                .with_primary_persistence_state(state.clone()),
            );
            process_runner
                .initialize()
                .await
                .context("failed to initialize immutable BTC run before feed startup")?;
            Ok((
                BtcPlaybookRuntimeHandle::start(runtime_config, process_runner, state, books)?,
                live_process_venue,
            ))
        }
        .await;

        let (runtime, live_process_venue) = match startup_result {
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
                run_id,
                run_key: run_key.clone(),
                config_hash: config_hash.clone(),
                execution_mode,
                strategy,
                sources,
                live_venue: live_process_venue,
                runtime,
            },
        );
        self.refresh_shared_sources().await?;
        let mut pending_guard = self.terminal_pending.lock().await;
        if pending_guard
            .get(&process_id)
            .is_some_and(|pending| pending.run_id == run_id)
        {
            pending_guard.remove(&process_id);
        }
        drop(pending_guard);
        self.record_event(
            process_id,
            "info",
            "btc_runtime_started",
            "BTC realtime execution runtime is running",
            serde_json::json!({
                "run_id": run_id,
                "run_key": &run_key,
                "config_hash": &config_hash,
            }),
        )
        .await;
        info!(
            process_id = %process_id,
            run_id = %run_id,
            run_key = %run_key,
            config_hash = %config_hash,
            compiled_source_identity = COMPILED_SOURCE_IDENTITY,
            "API-owned BTC realtime execution runtime started"
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
                "run_id": pending.run_id,
                "run_key": pending.run_key,
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
        self.stop_process_for_generation(process_id, None, reason, false, "stopped")
            .await
    }

    async fn complete_process(
        &self,
        process_id: uuid::Uuid,
        reason: &str,
    ) -> Result<TradingProcess, HttpError> {
        self.stop_process_for_generation(process_id, None, reason, false, "completed")
            .await
    }

    async fn stop_process_for_generation(
        &self,
        process_id: uuid::Uuid,
        expected_run_id: Option<uuid::Uuid>,
        reason: &str,
        runtime_failed: bool,
        successful_terminal_status: &str,
    ) -> Result<TradingProcess, HttpError> {
        if !matches!(successful_terminal_status, "stopped" | "completed") {
            return Err(HttpError::internal(
                "unsupported successful BTC terminal status",
            ));
        }
        let transition_guard = self.transition.clone().lock_owned().await;
        let manager = self.clone();
        let reason = reason.to_string();
        let successful_terminal_status = successful_terminal_status.to_string();
        tokio::spawn(async move {
            let _transition_guard = transition_guard;
            manager
                .stop_process_locked(
                    process_id,
                    expected_run_id,
                    &reason,
                    runtime_failed,
                    &successful_terminal_status,
                )
                .await
        })
        .await
        .map_err(|error| HttpError::internal(format!("BTC stop transition task failed: {error}")))?
    }

    async fn stop_process_locked(
        &self,
        process_id: uuid::Uuid,
        expected_run_id: Option<uuid::Uuid>,
        reason: &str,
        runtime_failed: bool,
        successful_terminal_status: &str,
    ) -> Result<TradingProcess, HttpError> {
        let process = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        if !Self::is_managed_process(&process) {
            return Err(HttpError::bad_request(
                "process is not a managed BTC realtime execution process",
            ));
        }

        let pending = { self.terminal_pending.lock().await.get(&process_id).cloned() };
        if let Some(pending) = pending {
            if expected_run_id.is_some_and(|expected| expected != pending.run_id) {
                return Ok(process);
            }
            return self.finalize_pending_locked(pending).await;
        }
        let active_identity = self
            .active_playbooks
            .lock()
            .await
            .get(&process_id)
            .map(|active| (active.run_id, active.run_key.clone()));
        let Some((active_run_id, active_run_key)) = active_identity else {
            if process.enabled
                || matches!(process.status.as_str(), "starting" | "running" | "stopping")
            {
                return Err(HttpError::conflict(
                    "BTC process claims to be active but this service owns no runtime handle",
                ));
            }
            if successful_terminal_status == "completed" && process.status != "completed" {
                return Err(HttpError::conflict(
                    "only an active or already completed BTC process can be completed",
                ));
            }
            return Ok(process);
        };
        if expected_run_id.is_some_and(|expected| expected != active_run_id) {
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
            "BTC realtime execution runtime stop accepted",
            serde_json::json!({
                "run_id": active_run_id,
                "run_key": active_run_key,
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
        let remaining_sources = merge_source_selectors(
            self.active_playbooks
                .lock()
                .await
                .values()
                .flat_map(|playbook| playbook.sources.clone()),
        )?;
        if let Some(shared) = self.shared_runtime.lock().await.as_ref() {
            if let Some(runtime) = shared.runtime.as_ref() {
                if !remaining_sources.is_empty() {
                    runtime.update_sources(remaining_sources);
                }
            }
        }
        let provisional_pending = PendingBtcTerminal {
            process_id: active.process_id,
            run_id: active.run_id,
            run_key: active.run_key.clone(),
            config_hash: active.config_hash.clone(),
            terminal_status: "failed".to_string(),
            terminal_reason: format!("{reason}; stop_transition_interrupted"),
            allow_inactive_process: false,
        };
        self.terminal_pending
            .lock()
            .await
            .insert(process_id, provisional_pending);
        let live_quiesce_failure =
            Self::quiesce_live_venue(active.live_venue.as_ref(), "process_runtime_stopping").await;
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
        let terminal_status =
            if runtime_failed || live_quiesce_failure.is_some() || shutdown_failure.is_some() {
                "failed"
            } else {
                successful_terminal_status
            };
        let mut terminal_failures = Vec::with_capacity(2);
        if let Some(failure) = live_quiesce_failure {
            terminal_failures.push(failure);
        }
        if let Some(failure) = shutdown_failure {
            terminal_failures.push(failure);
        }
        let terminal_reason = if terminal_failures.is_empty() {
            reason.to_string()
        } else {
            format!("{reason}; {}", terminal_failures.join("; "))
        };
        let pending = PendingBtcTerminal {
            process_id: active.process_id,
            run_id: active.run_id,
            run_key: active.run_key,
            config_hash: active.config_hash,
            terminal_status: terminal_status.to_string(),
            terminal_reason,
            allow_inactive_process: false,
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
        if let Err(terminal_error) = mark_btc_process_terminal(
            &self.pool,
            pending.process_id,
            &pending.terminal_status,
            &pending.terminal_reason,
            pending.allow_inactive_process,
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
                    "run_id": pending.run_id,
                    "run_key": pending.run_key,
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
            .is_some_and(|current| current.run_id == pending.run_id)
        {
            pending_guard.remove(&pending.process_id);
        }
        drop(pending_guard);
        let (level, event_type) = match pending.terminal_status.as_str() {
            "stopped" => ("info", "btc_runtime_stopped"),
            "completed" => ("info", "btc_runtime_completed"),
            _ => ("error", "btc_runtime_stop_failed"),
        };
        self.record_event(
            pending.process_id,
            level,
            event_type,
            &pending.terminal_reason,
            serde_json::json!({
                "run_id": pending.run_id,
                "run_key": pending.run_key,
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
                "BTC run {} has a pending API lifecycle transition; service shutdown left durable state unchanged",
                pending.run_key
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
            let run_id = active.run_id;
            let run_key = active.run_key.clone();
            let config_hash = active.config_hash.clone();
            if let Some(failure) =
                Self::quiesce_live_venue(active.live_venue.as_ref(), "service_shutdown").await
            {
                failures.push(format!("{process_id}: {failure}"));
            }
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
                "BTC realtime execution runtime suspended for service shutdown",
                serde_json::json!({
                    "run_id": run_id,
                    "run_key": run_key,
                    "config_hash": config_hash,
                    "reason": reason,
                    "resume_on_service_restart": true,
                }),
            )
            .await;
            info!(
                process_id = %process_id,
                run_id = %run_id,
                "BTC realtime execution runtime suspended with durable resume intent"
            );
        }
        let shared_runtime = { self.shared_runtime.lock().await.take() };
        if let Some(shared) = shared_runtime {
            if let Some(runtime) = shared.runtime {
                let shutdown_result =
                    tokio::time::timeout(BTC_RUNTIME_SHUTDOWN_TIMEOUT, runtime.shutdown()).await;
                if let Some(shutdown_failure) = match shutdown_result {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => {
                        Some(format!("shared_btc_runtime_shutdown_failed: {error:#}"))
                    }
                    Err(_) => Some(format!(
                        "shared_btc_runtime_shutdown_timeout_after_{}s",
                        BTC_RUNTIME_SHUTDOWN_TIMEOUT.as_secs()
                    )),
                } {
                    failures.push(shutdown_failure);
                }
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
            .map(|shared| shared.state.clone());
        let current_market = match state {
            Some(state) => state.read().await.display_market.clone(),
            None => None,
        };
        let markets = current_market
            .map(|market| vec![market; active_processes])
            .unwrap_or_default();
        CountdownSnapshot::resolve(observed_at, active_processes, markets)
    }

    async fn grafana_entry_status_snapshot(
        &self,
        observed_at: chrono::DateTime<Utc>,
    ) -> Result<TradingEntryStatusSnapshot> {
        let processes = self.store.list_observable_btc_processes().await?;
        let active = self
            .active_playbooks
            .lock()
            .await
            .iter()
            .map(|(process_id, active)| {
                (
                    *process_id,
                    (
                        active.execution_mode,
                        active.runtime.is_running(),
                        active.live_venue.clone(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();
        let mut permissions = Vec::with_capacity(processes.len());
        for process in processes {
            let execution = process.effective_execution();
            let permission = if !process.enabled || !execution.execute_signals {
                EntryPermission::disabled()
            } else if matches!(
                process.status.as_str(),
                "stopping" | "stopped" | "failed" | "expired"
            ) {
                EntryPermission::stopped()
            } else if process.status != "running" {
                EntryPermission::unknown(Some(format!("durable_process_{}", process.status)))
            } else {
                match active.get(&process.process_id) {
                    None => EntryPermission::unknown(Some("runtime_not_attached".to_string())),
                    Some((_, false, _)) => {
                        EntryPermission::unknown(Some("runtime_not_running".to_string()))
                    }
                    Some((BtcExecutionMode::Paper, true, _)) if execution.mode == "paper" => {
                        EntryPermission::enabled()
                    }
                    Some((BtcExecutionMode::Live, true, Some(venue)))
                        if execution.mode == "live" =>
                    {
                        match tokio::time::timeout(GRAFANA_STATUS_READ_TIMEOUT, venue.live_status())
                            .await
                        {
                            Ok(Ok(status)) if status.entries_enabled => EntryPermission::enabled(),
                            Ok(Ok(status)) => EntryPermission::blocked(status.reason),
                            Ok(Err(_)) => EntryPermission::unknown(Some(
                                "live_status_unavailable".to_string(),
                            )),
                            Err(_) => EntryPermission::unknown(Some(
                                "live_status_read_timeout".to_string(),
                            )),
                        }
                    }
                    Some((BtcExecutionMode::Live, true, None)) if execution.mode == "live" => {
                        EntryPermission::unknown(Some("live_venue_unavailable".to_string()))
                    }
                    Some(_) => EntryPermission::unknown(Some(
                        "runtime_execution_mode_mismatch".to_string(),
                    )),
                }
            };
            permissions.push(ProcessEntryPermission {
                process_id: process.process_id,
                permission,
            });
        }
        Ok(TradingEntryStatusSnapshot::new(observed_at, permissions))
    }

    async fn grafana_market_path_observation(
        &self,
    ) -> Option<(
        polymarket_bot::btc::BtcIntervalMarket,
        Vec<polymarket_bot::btc::ChainlinkTwap60Point>,
    )> {
        let state = self
            .shared_runtime
            .lock()
            .await
            .as_ref()
            .map(|shared| shared.state.clone());
        let Some(state) = state else {
            return None;
        };
        let (market, twap_history) = {
            let state = state.read().await;
            let Some(market) = state.display_market.clone() else {
                return None;
            };
            let twap_history = state.chainlink_twap_60.iter().cloned().collect::<Vec<_>>();
            (market, twap_history)
        };
        Some((market, twap_history))
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
            let _transition_guard = self.transition.lock().await;
            let current = self
                .terminal_pending
                .lock()
                .await
                .get(&pending.process_id)
                .filter(|current| current.run_id == pending.run_id)
                .cloned();
            let Some(current) = current else {
                continue;
            };
            if let Err(stop_error) = self.finalize_pending_locked(current).await {
                error!(
                    error = ?stop_error,
                    process_id = %pending.process_id,
                    run_id = %pending.run_id,
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
            .and_then(|shared| shared.runtime.as_ref().map(BtcRuntimeHandle::status_inputs));
        let shared_status = match shared_status_inputs {
            Some((state, books, metrics, config, running)) => {
                Some(runtime_status_from_inputs(state, books, metrics, config, running).await)
            }
            None => None,
        };
        let shared_running = shared_status.as_ref().is_some_and(|status| status.running);
        if shared_runtime_recovery_required(process_ids.len(), shared_running) {
            warn!(
                reason = shared_status
                    .as_ref()
                    .and_then(|status| status.metrics.last_error.as_deref())
                    .unwrap_or("shared market-data runtime handle is unavailable"),
                active_process_count = process_ids.len(),
                "BTC shared market-data runtime is unavailable; preserving process state and attempting recovery"
            );
            self.recover_shared_runtime().await;
            return;
        }
        for process_id in process_ids {
            let status_input = {
                let active_guard = self.active_playbooks.lock().await;
                let Some(active) = active_guard.get(&process_id) else {
                    continue;
                };
                (active.run_id, active.runtime.status_inputs())
            };
            let (run_id, (state, books, metrics, config, running)) = status_input;
            let status = runtime_status_from_inputs(state, books, metrics, config, running).await;
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
                            run_id = %run_id,
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
                    Some(run_id),
                    &terminal_reason,
                    true,
                    "stopped",
                )
                .await
            {
                error!(
                    error = ?stop_error,
                    process_id = %process_id,
                    run_id = %run_id,
                    "failed to terminalize BTC runtime child failure"
                );
            }
        }
    }

    async fn runtime_status(&self) -> serde_json::Value {
        let active_processes = self
            .active_playbooks
            .lock()
            .await
            .values()
            .map(|active| (active.process_id, active.execution_mode))
            .collect::<Vec<_>>();
        let mut processes = Vec::with_capacity(active_processes.len());
        for (process_id, execution_mode) in active_processes {
            processes.push(
                self.runtime_status_for_process(process_id, Some(execution_mode.as_str()))
                    .await,
            );
        }
        let shared_inputs = self
            .shared_runtime
            .lock()
            .await
            .as_ref()
            .and_then(|shared| shared.runtime.as_ref().map(BtcRuntimeHandle::status_inputs));
        let shared_market_data = match shared_inputs {
            Some((state, books, metrics, config, running)) => serde_json::to_value(
                runtime_status_from_inputs(state, books, metrics, config, running).await,
            )
            .unwrap_or_else(|_| serde_json::json!({"running": false})),
            None => serde_json::json!({
                "enabled": true,
                "running": false,
                "readiness": {"ready": false, "reasons": ["shared_market_data_inactive"]}
            }),
        };
        serde_json::json!({
            "capability_enabled": true,
            "active": !processes.is_empty(),
            "active_process_count": processes.len(),
            "shared_market_data": shared_market_data,
            "processes": processes,
        })
    }

    async fn runtime_status_for_process(
        &self,
        process_id: uuid::Uuid,
        configured_execution_mode: Option<&str>,
    ) -> serde_json::Value {
        let active = self
            .active_playbooks
            .lock()
            .await
            .get(&process_id)
            .map(|active| {
                (
                    active.process_id,
                    active.run_id,
                    active.run_key.clone(),
                    active.config_hash.clone(),
                    active.execution_mode,
                    active.strategy.clone(),
                    active.live_venue.clone(),
                    active.runtime.status_inputs(),
                )
            });
        if let Some((
            process_id,
            run_id,
            run_key,
            config_hash,
            execution_mode,
            strategy,
            live_venue,
            inputs,
        )) = active
        {
            let (state, books, metrics, config, running) = inputs;
            let mut runtime =
                runtime_status_from_inputs(state, books, metrics, config, running).await;
            runtime.readiness = process_runtime_readiness(&strategy, &runtime.readiness);
            let mut status = serde_json::json!({
                "capability_enabled": true,
                "active": true,
                "process_id": process_id,
                "run_id": run_id,
                "run_key": run_key,
                "config_hash": config_hash,
                "execution_mode": execution_mode.as_str(),
                "runtime": runtime,
            });
            if execution_mode == BtcExecutionMode::Live {
                let live_status = match live_venue {
                    Some(venue) => venue
                        .live_status()
                        .await
                        .and_then(|status| {
                            serde_json::to_value(status).map_err(anyhow::Error::from)
                        })
                        .unwrap_or_else(|_| {
                            serde_json::json!({
                                "entries_enabled": false,
                                "reason": "bound_live_status_unavailable"
                            })
                        }),
                    None => serde_json::json!({
                        "entries_enabled": false,
                        "reason": "bound_live_venue_unavailable"
                    }),
                };
                status
                    .as_object_mut()
                    .expect("BTC runtime status is an object")
                    .insert("live_status".to_string(), live_status);
            }
            return status;
        }
        let configured_execution_mode = configured_execution_mode.unwrap_or("unknown");
        let pending = self.terminal_pending.lock().await.get(&process_id).cloned();
        if let Some(pending) = pending {
            return serde_json::json!({
                "capability_enabled": true,
                "active": false,
                "running": false,
                "lifecycle_state": "terminal_pending",
                "process_id": pending.process_id,
                "run_id": pending.run_id,
                "run_key": pending.run_key,
                "config_hash": pending.config_hash,
                "execution_mode": configured_execution_mode,
                "desired_terminal_status": pending.terminal_status,
                "terminal_reason": pending.terminal_reason,
                "readiness": {"ready": false, "reasons": ["terminal_persistence_pending"]}
            });
        }
        serde_json::json!({
            "capability_enabled": true,
            "active": false,
            "running": false,
            "process_id": process_id,
            "execution_mode": configured_execution_mode,
            "readiness": {"ready": false, "reasons": ["trading_process_inactive"]}
        })
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
    live_venue: Option<Arc<LiveVenue>>,
    metrics: Arc<Mutex<RuntimeMetrics>>,
    btc_manager: Option<BtcProcessManager>,
}

impl RuntimeControl {
    fn live_venue(&self) -> Result<Arc<dyn ExecutionVenue>, HttpError> {
        self.live_venue
            .clone()
            .map(|venue| venue as Arc<dyn ExecutionVenue>)
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

    async fn prometheus_metrics(&self) -> Result<String, HttpError> {
        let runtime = match &self.btc_manager {
            Some(manager) => manager.runtime_status().await,
            None => serde_json::json!({}),
        };
        let metrics = runtime
            .get("shared_market_data")
            .and_then(|value| value.get("metrics"));
        let candle_ready = metrics
            .and_then(|value| value.get("rtds_chainlink_candle_window_ready"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let complete_minutes = metrics
            .and_then(|value| value.get("rtds_chainlink_candle_complete_minutes"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let oracle_ready = metrics
            .and_then(|value| value.get("polygon_oracle_ready"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let oracle_age = metrics
            .and_then(|value| value.get("polygon_oracle_age_seconds"))
            .and_then(serde_json::Value::as_u64);

        let mut output = String::from(
            "# HELP polymarket_btc_rtds_chainlink_candle_window_ready Whether the required 61 closed RTDS Chainlink candle minutes are available.\n\
# TYPE polymarket_btc_rtds_chainlink_candle_window_ready gauge\n",
        );
        output.push_str(&format!(
            "polymarket_btc_rtds_chainlink_candle_window_ready {}\n",
            u8::from(candle_ready)
        ));
        output.push_str(
            "# HELP polymarket_btc_rtds_chainlink_candle_complete_minutes Complete closed RTDS Chainlink candle minutes in the required window.\n\
# TYPE polymarket_btc_rtds_chainlink_candle_complete_minutes gauge\n",
        );
        output.push_str(&format!(
            "polymarket_btc_rtds_chainlink_candle_complete_minutes {complete_minutes}\n"
        ));
        output.push_str(
            "# HELP polymarket_btc_polygon_oracle_ready Whether a causally valid and fresh Polygon oracle round is available.\n\
# TYPE polymarket_btc_polygon_oracle_ready gauge\n",
        );
        output.push_str(&format!(
            "polymarket_btc_polygon_oracle_ready {}\n",
            u8::from(oracle_ready)
        ));
        output.push_str(
            "# HELP polymarket_btc_polygon_oracle_age_seconds Age in seconds of the latest accepted Polygon oracle round.\n\
# TYPE polymarket_btc_polygon_oracle_age_seconds gauge\n",
        );
        if let Some(oracle_age) = oracle_age {
            output.push_str(&format!(
                "polymarket_btc_polygon_oracle_age_seconds {oracle_age}\n"
            ));
        }
        output.push_str(&polymarket_bot::market_data_stream::prometheus_metrics());
        output.push_str(&polymarket_bot::btc::execution_freshness::prometheus_metrics(&runtime));
        output
            .push_str(&polymarket_bot::btc::unified_model_runtime::telemetry::prometheus_metrics());
        Ok(output)
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

    async fn btc_entry_status(
        &self,
        request: EntryStatusRequest,
    ) -> Result<polymarket_bot::grafana_live::EntryStatusSelection, HttpError> {
        let manager = self.btc_manager.as_ref().ok_or_else(|| {
            HttpError::internal("BTC realtime capability is disabled for this deployment")
        })?;
        let snapshot = manager
            .grafana_entry_status_snapshot(Utc::now())
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(snapshot.select(&request.scope, request.process_id))
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
        let _transition_guard = match &self.btc_manager {
            Some(manager) => Some(manager.transition.lock().await),
            None => None,
        };
        let live = self.live_venue()?;
        let disable_result = match tokio::time::timeout(
            BTC_LIVE_CONTROL_TIMEOUT,
            live.set_live_entries_enabled(false, Some("manual_live_halt".to_string())),
        )
        .await
        {
            Ok(result) => result.map_err(|error| error.to_string()),
            Err(_) => Err(format!(
                "live disable timed out after {}s",
                BTC_LIVE_CONTROL_TIMEOUT.as_secs()
            )),
        };
        let cancel_result =
            match tokio::time::timeout(BTC_LIVE_CONTROL_TIMEOUT, live.cancel_all()).await {
                Ok(result) => result.map_err(|error| error.to_string()),
                Err(_) => Err(format!(
                    "live cancel-all timed out after {}s",
                    BTC_LIVE_CONTROL_TIMEOUT.as_secs()
                )),
            };
        let reconcile_result =
            match tokio::time::timeout(BTC_LIVE_CONTROL_TIMEOUT, live.reconcile()).await {
                Ok(result) => result.map_err(|error| error.to_string()),
                Err(_) => Err(format!(
                    "live reconciliation timed out after {}s",
                    BTC_LIVE_CONTROL_TIMEOUT.as_secs()
                )),
            };
        let halted = disable_result
            .as_ref()
            .is_ok_and(|status| !status.entries_enabled)
            && cancel_result.is_ok()
            && reconcile_result
                .as_ref()
                .is_ok_and(|report| report.open_orders == 0 && report.unresolved_count == 0);
        self.store
            .insert_service_event(&ServiceEvent::new(
                "manual_live_halt",
                serde_json::json!({
                    "halted": halted,
                    "disable_result": disable_result.as_ref().map(|status| serde_json::json!({"entries_enabled": status.entries_enabled, "reason": status.reason})).unwrap_or_else(|error| serde_json::json!({"error": error})),
                    "cancel_result": cancel_result.as_ref().map(|count| serde_json::json!({"cancelled": count})).unwrap_or_else(|error| serde_json::json!({"error": error})),
                    "reconcile_result": reconcile_result.as_ref().map(|report| serde_json::json!({
                        "open_orders": report.open_orders,
                        "mismatches_found": report.mismatches_found,
                        "unresolved_count": report.unresolved_count
                    })).unwrap_or_else(|error| serde_json::json!({"error": error}))
                }),
            ))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(serde_json::json!({
            "halted": halted,
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
        if enabled {
            return Err(HttpError::bad_request(
                "wallet-wide live entry enable is disabled; enable entries on an active process-scoped runtime",
            ));
        }
        let _transition_guard = match &self.btc_manager {
            Some(manager) => Some(manager.transition.lock().await),
            None => None,
        };
        self.live_venue()?
            .set_live_entries_enabled(enabled, (!enabled).then(|| "manual_disable".to_string()))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn trading_process_live_preflight(
        &self,
        process_id: uuid::Uuid,
    ) -> Result<TradingProcessLivePreflightResponse, HttpError> {
        let manager = self.btc_manager.as_ref().ok_or_else(|| {
            HttpError::bad_request("BTC realtime capability is disabled for this deployment")
        })?;
        manager.live_preflight(process_id).await
    }

    async fn set_trading_process_live_entries_enabled(
        &self,
        process_id: uuid::Uuid,
        enabled: bool,
    ) -> Result<LiveVenueStatus, HttpError> {
        let manager = self.btc_manager.as_ref().ok_or_else(|| {
            HttpError::bad_request("BTC realtime capability is disabled for this deployment")
        })?;
        manager.set_live_entries_enabled(process_id, enabled).await
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
            HttpError::bad_request("BTC realtime capability is disabled for this deployment")
        })?;
        let now = Utc::now();
        manager.validate_inactive_definition(&TradingProcess {
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
                let execution_mode = process.effective_execution().mode;
                let runtime = match &self.btc_manager {
                    Some(manager) => {
                        manager
                            .runtime_status_for_process(process_id, Some(&execution_mode))
                            .await
                    }
                    None => serde_json::json!({
                        "capability_enabled": false,
                        "active": false,
                        "running": false,
                        "execution_mode": execution_mode
                    }),
                };
                object.insert("btc_runtime".to_string(), runtime);
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
                "start preview is available only for managed BTC realtime execution processes",
            ));
        }
        let manager = self.btc_manager.as_ref().ok_or_else(|| {
            HttpError::bad_request("BTC realtime capability is disabled for this deployment")
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
                "only managed BTC realtime execution process definitions can be updated",
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
        let active_selector_update = runtime_active
            && request.name.is_none()
            && request.metadata.is_none()
            && request.config.as_ref().is_some_and(|config| {
                selector_only_config_change(&current.config, config).unwrap_or(false)
            });
        if terminal_pending
            || current.enabled && !active_selector_update
            || matches!(current.status.as_str(), "starting" | "stopping")
        {
            return Err(HttpError::conflict(
                "stop the BTC trading process through its /stop endpoint before reconfiguring it",
            ));
        }
        if let Some(config) = &request.config {
            let manager = self.btc_manager.as_ref().ok_or_else(|| {
                HttpError::bad_request("BTC realtime capability is disabled for this deployment")
            })?;
            let mut candidate = current.clone();
            candidate.config = config.clone();
            if active_selector_update {
                manager.validate_resume_definition(&candidate)?;
            } else {
                manager.validate_inactive_definition(&candidate)?;
            }
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
        if active_selector_update {
            let manager = self.btc_manager.as_ref().expect("BTC manager exists");
            let sources = manager
                .validate_resume_definition(&process)?
                .control
                .sources;
            {
                let mut active = manager.active_playbooks.lock().await;
                active
                    .get_mut(&process_id)
                    .expect("active selector update retains its runtime")
                    .sources = sources;
                let union = merge_source_selectors(
                    active
                        .values()
                        .flat_map(|playbook| playbook.sources.clone()),
                )?;
                if let Some(runtime) = manager
                    .shared_runtime
                    .lock()
                    .await
                    .as_ref()
                    .and_then(|shared| shared.runtime.as_ref())
                {
                    runtime.update_sources(union);
                }
            }
        }
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
                "only managed BTC realtime execution processes can be started",
            ));
        }
        let manager = self.btc_manager.as_ref().ok_or_else(|| {
            HttpError::bad_request("BTC realtime capability is disabled for this deployment")
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
                "only managed BTC realtime execution processes can be stopped",
            ));
        }
        let manager = self.btc_manager.as_ref().ok_or_else(|| {
            HttpError::bad_request("BTC realtime capability is disabled for this deployment")
        })?;
        let process = manager.stop_process(process_id, "api_stop").await?;
        Ok(TradingProcessResponse { process })
    }

    async fn complete_trading_process(
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
                "only managed BTC realtime execution processes can be completed",
            ));
        }
        let manager = self.btc_manager.as_ref().ok_or_else(|| {
            HttpError::bad_request("BTC realtime capability is disabled for this deployment")
        })?;
        let process = manager.complete_process(process_id, "api_complete").await?;
        Ok(TradingProcessResponse { process })
    }
}

async fn run_grafana_live(
    manager: BtcProcessManager,
    publisher: GrafanaLivePublisher,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) {
    let mut ticker = tokio::time::interval(publisher.publish_interval());
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut last_countdown_error: Option<String> = None;
    let mut last_market_path_error: Option<String> = None;
    let mut last_entry_status_error: Option<String> = None;
    info!(
        countdown_channel = polymarket_bot::grafana_live::COUNTDOWN_CHANNEL,
        market_path_channel = polymarket_bot::grafana_live::MARKET_PATH_CHANNEL,
        entry_status_channel = polymarket_bot::grafana_live::ENTRY_STATUS_CHANNEL,
        "Grafana Live BTC market publishers started"
    );
    let mut market_path_state = MarketPathPublicationState::default();
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
                        if last_countdown_error.take().is_some() {
                            info!("Grafana Live BTC market countdown publishing recovered");
                        }
                    }
                    Err(publish_error) => {
                        let message = format!("{publish_error:#}");
                        if last_countdown_error.as_deref() != Some(message.as_str()) {
                            warn!(
                                error = %message,
                                "Grafana Live BTC market countdown publish failed; retrying"
                            );
                            last_countdown_error = Some(message);
                        }
                    }
                }

                let observed_at = Utc::now();
                let market_path_observation = manager.grafana_market_path_observation().await;
                let market_path_result = match market_path_state
                    .observe(observed_at, market_path_observation)
                {
                    Some(snapshot) => publisher.publish_market_path(&snapshot).await,
                    None => Ok(()),
                };
                match market_path_result {
                    Ok(()) => {
                        if last_market_path_error.take().is_some() {
                            info!("Grafana Live BTC market path publishing recovered");
                        }
                    }
                    Err(publish_error) => {
                        let message = format!("{publish_error:#}");
                        if last_market_path_error.as_deref() != Some(message.as_str()) {
                            warn!(
                                error = %message,
                                "Grafana Live BTC market path publish failed; retrying"
                            );
                            last_market_path_error = Some(message);
                        }
                    }
                }

                let entry_status_result = match manager
                    .grafana_entry_status_snapshot(Utc::now())
                    .await
                {
                    Ok(snapshot) => publisher.publish_entry_status(&snapshot).await,
                    Err(error) => Err(error),
                };
                match entry_status_result {
                    Ok(()) => {
                        if last_entry_status_error.take().is_some() {
                            info!("Grafana Live trading entry status publishing recovered");
                        }
                    }
                    Err(publish_error) => {
                        let message = format!("{publish_error:#}");
                        if last_entry_status_error.as_deref() != Some(message.as_str()) {
                            warn!(
                                error = %message,
                                "Grafana Live trading entry status publish failed; retrying"
                            );
                            last_entry_status_error = Some(message);
                        }
                    }
                }
            }
        }
    }
    info!("Grafana Live BTC market publishers stopped");
}

#[tokio::main]
async fn main() -> Result<()> {
    install_tls_crypto_provider();
    init_tracing();

    let config = AppConfig::from_env()?;
    let live_user_ws_enabled = config.live.user_ws_auth_available();
    info!(
        service = "polymarket-bot",
        live_order_submit_enabled = false,
        live_user_ws_enabled,
        btc_realtime_enabled = true,
        btc_paper_enabled = true,
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
                "live_order_submit_enabled": false,
                "live_user_ws_enabled": live_user_ws_enabled,
                "btc_realtime_enabled": true,
                "btc_paper_enabled": true,
                "grafana_live_enabled": config.grafana_live.enabled,
                "compiled_source_identity": COMPILED_SOURCE_IDENTITY,
                "kafka_required": false
            }),
        ))
        .await?;

    let data_api = DataApiClient::new(config.data_api_base_url.clone());
    let live_venue: Option<Arc<LiveVenue>> = if config.live.live_auth_available() {
        Some(Arc::new(LiveVenue::new(
            config.live.clone(),
            config.live.clob_api_base_url.clone(),
            store.clone(),
            data_api.clone(),
        )?))
    } else {
        None
    };
    let btc_manager = {
        let repository = BtcRepository::from_pool(pool.clone());
        Some(BtcProcessManager::new(
            store.clone(),
            pool.clone(),
            repository,
            BtcProcessManagerConfig {
                live_venue: live_venue.clone(),
                live_reconcile_interval: config.live.reconcile_interval,
            },
        ))
    };
    if let Some(manager) = &btc_manager {
        if let Err(error) = manager.resume_durable_processes().await {
            bail!(
                "failed to resume durable BTC realtime execution state; database lifecycle state was left unchanged: {error:?}"
            );
        }
    }

    let (grafana_live_shutdown_tx, grafana_live_shutdown_rx) = tokio::sync::watch::channel(false);
    let grafana_live_task = if config.grafana_live.enabled {
        let manager = btc_manager
            .clone()
            .context("Grafana Live countdown requires the BTC process manager")?;
        let publisher = GrafanaLivePublisher::new(config.grafana_live.clone())?;
        Some(tokio::spawn(run_grafana_live(
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
        let control: control_http::SharedControlApi = Arc::new(RuntimeControl {
            store: store.clone(),
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
    #[test]
    fn optional_model_source_preserves_required_subscription_in_either_order() {
        let required: super::SourceSelector =
            serde_json::from_value(serde_json::json!("polygon_chainlink_btcusd_oracle")).unwrap();
        let optional: super::SourceSelector = serde_json::from_value(serde_json::json!({
            "key": "polygon_chainlink_btcusd_oracle", "required": false,
            "maximum_age_ms": 600000, "require_sequence_integrity": false
        }))
        .unwrap();
        for selectors in [
            vec![required.clone(), optional.clone()],
            vec![optional.clone(), required.clone()],
        ] {
            assert_eq!(
                super::merge_source_selectors(selectors).unwrap(),
                vec![required.clone()]
            );
        }
        let mut conflicting = optional.clone();
        conflicting.contract_version = 2;
        assert!(super::merge_source_selectors([required.clone(), conflicting]).is_err());
        let mut conflicting = required.clone();
        conflicting.maximum_age_ms = Some(1000);
        assert!(super::merge_source_selectors([required, conflicting]).is_err());
    }
    use super::*;
    use polymarket_bot::btc::BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION;

    #[tokio::test]
    async fn inactive_runtime_status_reports_configured_mode_without_unbound_live_status() {
        let pool = PgPoolOptions::new()
            .connect_lazy("postgres://postgres:postgres@localhost/polymarket")
            .unwrap();
        let manager = BtcProcessManager::new(
            Store::from_pool(pool.clone()),
            pool.clone(),
            BtcRepository::from_pool(pool),
            BtcProcessManagerConfig {
                live_venue: None,
                live_reconcile_interval: Duration::from_secs(1),
            },
        );

        let status = manager
            .runtime_status_for_process(uuid::Uuid::new_v4(), Some("live"))
            .await;

        assert_eq!(status["active"], false);
        assert_eq!(status["execution_mode"], "live");
        assert!(status.get("live_status").is_none());
    }

    #[test]
    fn btc_process_terminal_validation_accepts_all_supported_terminal_states() {
        for status in ["stopped", "failed", "completed"] {
            validate_btc_process_terminal_request(status, "api_transition").unwrap();
        }
        assert!(validate_btc_process_terminal_request("running", "api_transition").is_err());
        assert!(validate_btc_process_terminal_request("completed", " ").is_err());
    }

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
                    account_ref: None,
                    taker_fee_rate: None,
                    ..ProcessExecutionConfig::default()
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

    #[test]
    fn durable_resume_uses_only_explicit_running_live_capital_authorization() {
        let mut process = eligible_btc_process();
        process.enabled = true;
        process.status = "running".to_string();
        let configured = process.config.execution.as_mut().unwrap();
        configured.mode = Some("live".to_string());
        configured.live_capital = true;
        configured.account_ref = Some("polymarket-primary".to_string());
        let execution = process.effective_execution();
        assert!(should_resume_configured_live_entries(&process, &execution));

        process.enabled = false;
        assert!(!should_resume_configured_live_entries(&process, &execution));
        process.enabled = true;
        process.status = "stopping".to_string();
        assert!(!should_resume_configured_live_entries(&process, &execution));
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
            risk_strategies: Vec::new(),
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
        assert_eq!(
            control.paper.directional_model_entry_policy,
            BtcDirectionalModelEntryPolicy::RequirePositiveDirectEdge
        );
        assert!(control.entry_admission.is_none());
        assert!(control.risk_strategies.is_empty());
    }

    #[test]
    fn btc_process_control_accepts_one_explicit_risk_strategy() {
        let control: BtcRealtimePaperControlConfig = serde_json::from_value(serde_json::json!({
            "schema_version": BTC_PROCESS_SCHEMA_VERSION,
            "next_experiment_key": "btc-5m-risk-contract",
            "preregistration_sha256": "a".repeat(64),
            "risk_strategies": [{
                "version": "capitonic-risk-strategy-v1",
                "model_key": "risk-model",
                "artifact_sha256": "b".repeat(64)
            }]
        }))
        .unwrap();
        risk_runtime::validate_selections(&control.risk_strategies).unwrap();
        let mut duplicated = control.risk_strategies.clone();
        duplicated.push(duplicated[0].clone());
        assert!(risk_runtime::validate_selections(&duplicated).is_err());
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
    fn selectable_v3_resolves_native_directional_model_as_the_only_strategy() {
        let control = BtcRealtimePaperControlConfig {
            schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
            strategy: serde_json::json!({
                "decision_strategy": {
                    "type": "btc_directional_model",
                    "model_key": polymarket_bot::btc::BTC_DIRECTIONAL_MODEL_V1_KEY,
                    "artifact_sha256":
                        polymarket_bot::btc::BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256,
                    "feature_schema_sha256":
                        polymarket_bot::btc::BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256
                },
                "min_seconds_after_open": 60,
                "min_seconds_before_close": 60,
                "max_directional_feature_age_ms": 5000
            }),
            ..BtcRealtimePaperControlConfig::default()
        };

        let strategy = resolve_btc_strategy(&control).unwrap();

        assert_eq!(
            strategy.strategy_version,
            BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION
        );
        assert_eq!(
            strategy.feature_schema_version,
            BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION
        );
        assert!(matches!(
            strategy.decision_strategy,
            Some(BtcDecisionStrategyConfig::BtcDirectionalModel { .. })
        ));
        assert_eq!(strategy.max_directional_feature_age_ms, Some(5_000));
    }

    #[test]
    fn selectable_v3_resolves_asymmetric_value_model_without_directional_entry_policy() {
        let mut control = BtcRealtimePaperControlConfig {
            schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
            strategy: serde_json::json!({
                "decision_strategy": {
                    "type": "btc_asymmetric_value_model",
                    "model_key": "btc-5m-asymmetric-core-paper-20260805-v1",
                    "artifact_sha256":
                        "4379aee1ab04b382425b76f2c8f32e80c86299a149cd9994c2515b8138de9813",
                    "feature_schema_sha256":
                        "633033efb069dfb54a5f7834ab355ea380bd01c5774b800fddb528322e1e73dd"
                },
                "required_model_feeds": [
                    {"feed": "binance_btcusdt_one_second_v1", "maximum_age_ms": 1000},
                    {"feed": "polymarket_btc5m_clob_execution_v1", "maximum_age_ms": 2000}
                ],
                "min_seconds_after_open": 1,
                "min_seconds_before_close": 244,
                "max_directional_feature_age_ms": 1000,
                "min_entry_price": "0.20",
                "max_entry_price": "0.30",
                "spread_reserve_fraction": "0",
                "slippage_reserve_bps": "0",
                "latency_reserve_per_share": "0.01",
                "min_net_edge_per_share": "0.03",
                "min_net_edge_usd": "0.15"
            }),
            ..BtcRealtimePaperControlConfig::default()
        };

        let strategy = resolve_btc_strategy(&control).unwrap();

        assert_eq!(
            strategy.strategy_version,
            BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_VERSION
        );
        assert!(matches!(
            strategy.decision_strategy,
            Some(BtcDecisionStrategyConfig::BtcAsymmetricValueModel { .. })
        ));
        assert_eq!(strategy.required_model_feeds.len(), 2);

        control.strategy["decision_strategy"] = serde_json::json!({
            "type": "btc_asymmetric_value_model",
            "model_key": "btc-5m-asymmetric-core-oracle-live-pilot-20260814",
            "artifact_sha256":
                "c87dd4ca07903000f5ccff2c3691b9541aee38f389cceee7c9432383ec0e0a0b",
            "feature_schema_sha256":
                "fe2a5aaee3df1ef899d2553712555091aa29b7481b3fed7805ba140dc8aa5014"
        });
        control.strategy["required_model_feeds"] = serde_json::json!([
            {"feed": "binance_btcusdt_one_second_v1", "maximum_age_ms": 1000},
            {"feed": "polymarket_btc5m_clob_execution_v1", "maximum_age_ms": 2000},
            {"feed": "chainlink_btcusd_oracle_v1", "maximum_age_ms": 300000}
        ]);
        let live_strategy = resolve_btc_strategy(&control).unwrap();
        validate_btc_live_model_authorization(&live_strategy).unwrap();
    }

    #[test]
    fn live_model_authorization_accepts_only_the_promoted_asymmetric_artifact() {
        let strategy = |model_key: &str, artifact_sha256: &str| BtcStrategyConfig {
            decision_strategy: Some(BtcDecisionStrategyConfig::BtcAsymmetricValueModel {
                model_key: model_key.to_string(),
                artifact_sha256: artifact_sha256.to_string(),
                feature_schema_sha256:
                    "fe2a5aaee3df1ef899d2553712555091aa29b7481b3fed7805ba140dc8aa5014".to_string(),
            }),
            ..BtcStrategyConfig::default()
        };

        validate_btc_live_model_authorization(&strategy(
            "btc-5m-asymmetric-core-oracle-live-pilot-20260814",
            "c87dd4ca07903000f5ccff2c3691b9541aee38f389cceee7c9432383ec0e0a0b",
        ))
        .unwrap();

        assert!(validate_btc_live_model_authorization(&strategy(
            "btc-5m-asymmetric-core-oracle-paper-20260805-v1",
            "2c91e894356f6fee7fe9514e24c39da6e11602ffcb7f961f64848bde72418db9",
        ))
        .is_err());
    }

    #[test]
    fn directional_model_validation_entry_policy_is_paper_only_and_frozen() {
        let mut control = BtcRealtimePaperControlConfig {
            schema_version: SELECTABLE_BTC_PROCESS_SCHEMA_VERSION.to_string(),
            next_experiment_key: "btc-5m-directional-model-validation-test".to_string(),
            preregistration_sha256: "e".repeat(64),
            strategy: serde_json::json!({
                "decision_strategy": {
                    "type": "btc_directional_model",
                    "model_key": polymarket_bot::btc::BTC_DIRECTIONAL_MODEL_V1_KEY,
                    "artifact_sha256":
                        polymarket_bot::btc::BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256,
                    "feature_schema_sha256":
                        polymarket_bot::btc::BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256
                },
                "min_seconds_after_open": 60,
                "min_seconds_before_close": 60
            }),
            ..BtcRealtimePaperControlConfig::default()
        };
        control.paper.directional_model_entry_policy =
            BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction;
        let strategy = resolve_btc_strategy(&control).unwrap();
        validate_directional_model_entry_policy(
            &strategy,
            BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction,
        )
        .unwrap();
        assert!(validate_directional_model_entry_policy(
            &BtcStrategyConfig::default(),
            BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction,
        )
        .is_err());

        let resolved = |control: BtcRealtimePaperControlConfig| ResolvedBtcProcessDefinition {
            control,
            strategy: strategy.clone(),
            entry_admission: None,
            risk_strategies: Vec::new(),
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        };
        let prepared = prepare_btc_start_definition(resolved(control.clone())).unwrap();
        assert_eq!(
            prepared.directional_model_entry_policy,
            BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction
        );
        assert_eq!(
            prepared.frozen_process_config.raw["paper"]["directional_model_entry_policy"],
            serde_json::json!("execute_directional_prediction")
        );

        control.paper.directional_model_entry_policy =
            BtcDirectionalModelEntryPolicy::RequirePositiveDirectEdge;
        let default_prepared = prepare_btc_start_definition(resolved(control)).unwrap();
        assert!(default_prepared.frozen_process_config.raw["paper"]
            .get("directional_model_entry_policy")
            .is_none());
        assert_ne!(prepared.config_hash, default_prepared.config_hash);

        let mut live_capital = eligible_btc_process();
        live_capital.config.execution.as_mut().unwrap().live_capital = true;
        assert!(validate_btc_process_capability(&live_capital).is_err());

        let mut live_mode = eligible_btc_process();
        live_mode.config.execution.as_mut().unwrap().mode = Some("live".to_string());
        assert!(validate_btc_process_capability(&live_mode).is_err());
    }

    #[test]
    fn optional_execution_controls_are_mode_independent_and_validated_when_present() {
        let mut process = eligible_btc_process();
        let execution = process.config.execution.as_mut().unwrap();
        execution.max_order_notional_usd = Some(dec!(5));
        execution.max_open_notional_usd = Some(dec!(20));
        execution.max_open_positions = Some(6);
        execution.max_daily_loss_usd = Some(dec!(10));
        execution.require_exit_book = Some(true);
        validate_btc_process_capability(&process).unwrap();

        let execution = process.config.execution.as_mut().unwrap();
        execution.mode = Some("live".to_string());
        execution.execute_signals = true;
        execution.live_capital = true;
        execution.account_ref = Some("polymarket-primary".to_string());
        validate_btc_process_capability(&process).unwrap();

        process
            .config
            .execution
            .as_mut()
            .unwrap()
            .max_order_notional_usd = Some(dec!(5.01));
        assert!(validate_btc_process_capability(&process).is_err());
    }

    #[test]
    fn entry_timing_accepts_only_the_fixed_120_zero_width_model_window() {
        let mut fixed_120 = BtcStrategyConfig {
            min_seconds_after_open: 120,
            min_seconds_before_close: 180,
            decision_strategy: Some(BtcDecisionStrategyConfig::BtcDirectionalModel {
                model_key: "btc-fixed-120".to_string(),
                artifact_sha256: "a".repeat(64),
                feature_schema_sha256: "b".repeat(64),
            }),
            ..BtcStrategyConfig::default()
        };
        validate_btc_entry_timing(&fixed_120).unwrap();

        fixed_120.min_seconds_after_open = 125;
        fixed_120.min_seconds_before_close = 175;
        assert!(validate_btc_entry_timing(&fixed_120).is_err());

        fixed_120.min_seconds_after_open = 121;
        fixed_120.min_seconds_before_close = 180;
        assert!(validate_btc_entry_timing(&fixed_120).is_err());

        fixed_120.min_seconds_after_open = 120;
        fixed_120.min_seconds_before_close = 180;
        fixed_120.decision_strategy = None;
        assert!(validate_btc_entry_timing(&fixed_120).is_err());
    }

    #[test]
    fn btc_start_eligibility_rejects_generic_active_and_invalid_definitions() {
        let mut generic = eligible_btc_process();
        generic.process_type = "copy_trade".to_string();
        assert!(validate_btc_start_eligibility(&generic).is_err());

        let mut active = eligible_btc_process();
        active.status = "running".to_string();
        active.enabled = true;
        assert!(validate_btc_start_eligibility(&active).is_err());

        let mut invalid = eligible_btc_process();
        invalid.config.execution.as_mut().unwrap().mode = Some("live".to_string());
        assert!(validate_btc_start_eligibility(&invalid).is_err());

        assert!(validate_btc_start_eligibility(&eligible_btc_process()).is_ok());
    }

    #[test]
    fn live_credential_only_definition_is_valid_but_cannot_start() {
        let mut process = eligible_btc_process();
        process.config.execution = Some(ProcessExecutionConfig {
            mode: Some("live".to_string()),
            execute_signals: false,
            live_capital: false,
            account_ref: Some("polymarket-primary".to_string()),
            taker_fee_rate: None,
            ..ProcessExecutionConfig::default()
        });

        validate_btc_process_capability(&process).unwrap();
        assert!(validate_btc_start_eligibility(&process).is_err());

        let execution = process.config.execution.as_mut().unwrap();
        execution.execute_signals = true;
        execution.live_capital = true;
        validate_btc_start_eligibility(&process).unwrap();
    }

    #[test]
    fn live_start_preparation_has_a_distinct_venue_identity_and_frozen_seam() {
        let run_key = "btc-5m-live-cutover-preview";
        let resolved = ResolvedBtcProcessDefinition {
            control: BtcRealtimePaperControlConfig {
                schema_version: BTC_PROCESS_SCHEMA_VERSION.to_string(),
                next_experiment_key: run_key.to_string(),
                preregistration_sha256: "c".repeat(64),
                ..BtcRealtimePaperControlConfig::default()
            },
            strategy: BtcStrategyConfig::default(),
            entry_admission: None,
            risk_strategies: Vec::new(),
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        };
        let execution = EffectiveProcessExecutionConfig {
            mode: "live".to_string(),
            execute_signals: true,
            live_capital: true,
            account_ref: Some("polymarket-primary".to_string()),
            taker_fee_rate: dec!(0.03),
            ..EffectiveProcessExecutionConfig::default()
        };

        let prepared = prepare_btc_start_definition_for_execution(resolved, &execution).unwrap();
        assert_eq!(prepared.execution_mode, BtcExecutionMode::Live);
        assert_eq!(
            prepared.execution.account_ref.as_deref(),
            Some("polymarket-primary")
        );
        assert_eq!(
            prepared.run_id,
            uuid::Uuid::new_v5(
                &uuid::Uuid::NAMESPACE_URL,
                format!("polymarket-bot/btc-live/{run_key}").as_bytes(),
            )
        );
        assert_eq!(
            prepared
                .frozen_process_config
                .execution
                .as_ref()
                .unwrap()
                .mode
                .as_deref(),
            Some("live")
        );
        assert_eq!(
            prepared.frozen_process_config.raw["paper"]["execution_enabled"],
            false
        );
        assert_eq!(
            prepared.frozen_process_config.raw["live"],
            serde_json::json!({
                "execution_enabled": true,
                "account_ref": "polymarket-primary"
            })
        );
    }

    #[test]
    fn live_start_cannot_relax_the_two_second_execution_freshness_contract() {
        let execution = EffectiveProcessExecutionConfig {
            mode: "live".to_string(),
            execute_signals: true,
            live_capital: true,
            account_ref: Some("polymarket-primary".to_string()),
            taker_fee_rate: dec!(0.03),
            ..EffectiveProcessExecutionConfig::default()
        };
        let resolved = |strategy: BtcStrategyConfig| ResolvedBtcProcessDefinition {
            control: BtcRealtimePaperControlConfig {
                schema_version: BTC_PROCESS_SCHEMA_VERSION.to_string(),
                next_experiment_key: "btc-live-freshness-contract".to_string(),
                preregistration_sha256: "d".repeat(64),
                ..BtcRealtimePaperControlConfig::default()
            },
            strategy,
            entry_admission: None,
            risk_strategies: Vec::new(),
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        };

        let mut relaxed_reference = BtcStrategyConfig::default();
        relaxed_reference.max_reference_age_ms = BTC_LIVE_EXECUTION_FRESHNESS_LIMIT_MS + 1;
        assert!(prepare_btc_start_definition_for_execution(
            resolved(relaxed_reference.clone()),
            &execution,
        )
        .is_err());
        assert!(
            prepare_btc_start_definition(resolved(relaxed_reference)).is_ok(),
            "the live-only ceiling must not break existing paper definitions"
        );

        let mut relaxed_book = BtcStrategyConfig::default();
        relaxed_book.max_book_age_ms = BTC_LIVE_EXECUTION_FRESHNESS_LIMIT_MS + 1;
        assert!(
            prepare_btc_start_definition_for_execution(resolved(relaxed_book), &execution,)
                .is_err()
        );

        assert!(prepare_btc_start_definition_for_execution(
            resolved(BtcStrategyConfig::default()),
            &execution,
        )
        .is_ok());
    }

    #[test]
    fn btc_start_preparation_is_deterministic_and_freezes_exact_execution_config() {
        let run_key = "btc-5m-paper-20260713-preview";
        let preregistration_sha256 = "b".repeat(64);
        let resolved = ResolvedBtcProcessDefinition {
            control: BtcRealtimePaperControlConfig {
                schema_version: BTC_PROCESS_SCHEMA_VERSION.to_string(),
                next_experiment_key: run_key.to_string(),
                preregistration_sha256: preregistration_sha256.clone(),
                ..BtcRealtimePaperControlConfig::default()
            },
            strategy: BtcStrategyConfig::default(),
            entry_admission: None,
            risk_strategies: Vec::new(),
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
        };

        let first = prepare_btc_start_definition(resolved.clone()).unwrap();
        let second = prepare_btc_start_definition(resolved).unwrap();
        let expected_run_id = uuid::Uuid::new_v5(
            &uuid::Uuid::NAMESPACE_URL,
            format!("polymarket-bot/btc-paper/{run_key}").as_bytes(),
        );
        let serialized_config = serde_json::to_vec(&first.frozen_process_config).unwrap();
        let expected_config_hash = format!("{:x}", Sha256::digest(serialized_config));

        assert_eq!(first.run_id, expected_run_id);
        assert_eq!(first.run_id, second.run_id);
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
    fn shared_market_data_accepts_playbook_runtime_differences() {
        let shared = BtcRuntimeConfig::default();
        let mut playbook = shared.clone();
        playbook.strategy_interval = Duration::from_millis(500);
        playbook.max_book_age = Duration::from_secs(3);
        playbook.max_reference_age = Duration::from_secs(4);

        assert!(shared_market_data_config_compatible(&shared, &playbook));

        playbook.strategy_interval += Duration::from_millis(1);
        assert!(shared_market_data_config_compatible(&shared, &playbook));
    }

    #[test]
    fn shared_feed_loss_requests_recovery_only_for_active_processes() {
        assert!(shared_runtime_recovery_required(17, false));
        assert!(!shared_runtime_recovery_required(17, true));
        assert!(!shared_runtime_recovery_required(0, false));
    }

    #[tokio::test]
    async fn shared_feed_recovery_invalidates_evidence_without_rebinding_consumers() {
        let state = Arc::new(tokio::sync::RwLock::new(
            polymarket_bot::btc::RealtimeState {
                primary_persistence_degraded: true,
                last_updated_at: Some(Utc::now()),
                ..polymarket_bot::btc::RealtimeState::default()
            },
        ));
        let original_connection_id = uuid::Uuid::new_v4();
        let books = Arc::new(tokio::sync::RwLock::new(BookRegistry::new(
            original_connection_id,
        )));
        let state_consumer = state.clone();
        let books_consumer = books.clone();

        invalidate_shared_market_data_evidence(&state, &books).await;

        assert!(Arc::ptr_eq(&state, &state_consumer));
        assert!(Arc::ptr_eq(&books, &books_consumer));
        assert_eq!(
            *state.read().await,
            polymarket_bot::btc::RealtimeState::default()
        );
        assert_ne!(books.read().await.connection_id(), original_connection_id);
    }

    #[test]
    fn btc_resume_process_contract_ignores_system_feed_transport_metadata() {
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
                "gamma_base_url",
                "clob_rest_base_url",
                "clob_ws_url",
                "rtds_ws_url",
                "clob_heartbeat_interval",
                "rtds_heartbeat_interval",
                "binance_heartbeat_interval",
                "binance_ws_url",
                "binance_spot_l2_enabled",
                "binance_spot_l2_ws_url",
                "binance_rest_base_url",
                "discovery_interval",
                "reconnect_initial_delay",
                "reconnect_max_delay",
                "checkpoint_interval",
                "boundary_tick_max_delay",
                "official_resolution_audit_grace",
                "official_resolution_watch_retention",
                "writer_capacity",
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
        durable["raw"]["runtime"]["strategy_interval"] =
            serde_json::to_value(Duration::from_secs(1)).unwrap();

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
        validate_btc_start_eligibility(&process).unwrap();
    }
    #[test]
    fn umr_paper_templates_use_existing_process_resolution_and_start_contract() {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("infra/processes");
        let mut checked = 0;
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if !path.to_string_lossy().ends_with("-umr-20260902.json") {
                continue;
            }
            let process: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            let control = parse_btc_process_control(
                process["config"]["raw"]["btc_realtime_paper"].clone(),
                BtcDefinitionUse::InactiveDefinition,
            )
            .unwrap();
            let strategy = resolve_btc_strategy(&control).unwrap();
            strategy.validate().unwrap();
            validate_btc_entry_timing(&strategy).unwrap();
            validate_directional_model_entry_policy(
                &strategy,
                control.paper.directional_model_entry_policy,
            )
            .unwrap();
            assert!(validate_btc_live_model_authorization(&strategy).is_err());
            let resolved = ResolvedBtcProcessDefinition {
                control,
                strategy,
                entry_admission: None,
                risk_strategies: Vec::new(),
                runtime: BtcRuntimeConfig {
                    enabled: true,
                    ..BtcRuntimeConfig::default()
                },
                paper_venue: PaperVenueConfig::default(),
                paper_stress_previews: Vec::new(),
            };
            let first = prepare_btc_start_definition(resolved.clone()).unwrap();
            let resumed = prepare_btc_start_definition(resolved).unwrap();
            assert_eq!(first.run_id, resumed.run_id);
            assert_eq!(first.config_hash, resumed.config_hash);
            checked += 1;
        }
        assert_eq!(checked, 5);
    }
}
