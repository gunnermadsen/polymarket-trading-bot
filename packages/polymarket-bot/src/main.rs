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
        BookRegistry, BtcPaperExperimentConfig, BtcPaperExperimentRunner, BtcRepository,
        BtcRuntime, BtcRuntimeConfig, BtcRuntimeHandle, BtcStrategyConfig, PaperPreviewConfig,
        PaperVenue as BtcPaperVenue, PaperVenueConfig, BTC_FEATURE_SCHEMA_VERSION,
        BTC_STRATEGY_VERSION, BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION,
    },
    clob::ClobClient,
    config::{AppConfig, BtcConfig, ExecutionMode},
    data_api::{ClosedPositionsQuery, DataApiClient},
    events::ServiceEvent,
    execution::{
        live::LiveVenue, sim::SimVenue, ExecutionVenue, LiveIdentityDiagnostics,
        LiveOrderDryRunDiagnostics, LiveOrderDryRunRequest, LivePoly1271FunderProbeRequest,
        LivePoly1271FunderProbeResponse, LiveVenueStatus, LiveWalletAddressDiagnostics,
    },
    gamma::GammaClient,
    http as control_http,
    http::{
        ControlApi, HealthResponse, HealthStatus, HttpError, IngestionBackfillCancelResponse,
        IngestionBackfillEnqueueResponse, IngestionBackfillEventsResponse,
        IngestionBackfillJobResponse, IngestionBackfillJobsResponse, MetricsResponse,
        TradingProcessResetResponse, TradingProcessResponse, TradingProcessStartPreviewResponse,
        TradingProcessStatusResponse, TradingProcessesResponse,
    },
    ingestion::{
        job::BackfillRequest as IngestionBackfillRequest, repository::IngestionRepository,
    },
    models::{DataApiClosedPosition, ProcessExecutionConfig, TradingProcess, TradingProcessConfig},
    risk::{RiskLimits, RiskState},
    scanner::{scan_markets_for_signal1, ScannerConfig, ScannerCycleReport},
    store::Store,
    taxonomy::taxonomy_update_from_metadata,
    wallets::{score_closed_position_performance, score_mrs, MrsScoreInput},
};
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{postgres::PgPoolOptions, PgPool};
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

const SCAN_CYCLE_TIMEOUT: Duration = Duration::from_secs(8);
const BTC_PIPELINE_VERSION: &str = "btc_realtime_paper_pipeline_v11";
const BTC_PROCESS_SCHEMA_VERSION: &str = "btc_realtime_paper_process_v1";
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

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct BtcRealtimePaperControlConfig {
    schema_version: String,
    next_experiment_key: String,
    preregistration_sha256: String,
    strategy: serde_json::Value,
    runtime: BtcProcessRuntimeControl,
    paper: BtcProcessPaperControl,
    ml_shadow: BtcProcessMlShadowControl,
}

impl Default for BtcRealtimePaperControlConfig {
    fn default() -> Self {
        Self {
            schema_version: String::new(),
            next_experiment_key: String::new(),
            preregistration_sha256: String::new(),
            strategy: serde_json::json!({}),
            runtime: BtcProcessRuntimeControl::default(),
            paper: BtcProcessPaperControl::default(),
            ml_shadow: BtcProcessMlShadowControl::default(),
        }
    }
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

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct BtcProcessMlShadowControl {
    enabled: bool,
}

impl Default for BtcProcessMlShadowControl {
    fn default() -> Self {
        Self { enabled: false }
    }
}

#[derive(Clone)]
struct ResolvedBtcProcessDefinition {
    control: BtcRealtimePaperControlConfig,
    strategy: BtcStrategyConfig,
    runtime: BtcRuntimeConfig,
    paper_venue: PaperVenueConfig,
    paper_stress_previews: Vec<PaperPreviewConfig>,
    ml_shadow_enabled: bool,
}

struct PreparedBtcStartDefinition {
    experiment_id: uuid::Uuid,
    experiment_key: String,
    preregistration_sha256: String,
    strategy: BtcStrategyConfig,
    runtime: BtcRuntimeConfig,
    paper_venue: PaperVenueConfig,
    paper_stress_previews: Vec<PaperPreviewConfig>,
    ml_shadow_enabled: bool,
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
        runtime,
        paper_venue,
        paper_stress_previews,
        ml_shadow_enabled,
    } = resolved;
    let experiment_key = control.next_experiment_key;
    let preregistration_sha256 = control.preregistration_sha256;
    let experiment_id = uuid::Uuid::new_v5(
        &uuid::Uuid::NAMESPACE_URL,
        format!("polymarket-bot/btc-paper/{experiment_key}").as_bytes(),
    );
    let frozen_raw = serde_json::json!({
        "pipeline_version": BTC_PIPELINE_VERSION,
        "process_schema_version": BTC_PROCESS_SCHEMA_VERSION,
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
        },
        "ml_shadow": {
            "ml_a_enabled": ml_shadow_enabled,
            "ml_b_enabled": ml_shadow_enabled,
            "execution_authority": false,
        }
    });
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
    let config_hash = format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&frozen_process_config)
                .map_err(|error| HttpError::internal(error.to_string()))?
        )
    );
    Ok(PreparedBtcStartDefinition {
        experiment_id,
        experiment_key,
        preregistration_sha256,
        strategy,
        runtime,
        paper_venue,
        paper_stress_previews,
        ml_shadow_enabled,
        frozen_process_config,
        config_hash,
    })
}

fn resume_config_without_compiled_source_identity(
    mut config: serde_json::Value,
) -> serde_json::Value {
    if let Some(build) = config
        .pointer_mut("/raw/build")
        .and_then(serde_json::Value::as_object_mut)
    {
        build.remove("compiled_source_identity");
    }
    config
}

struct ActiveBtcRun {
    process_id: uuid::Uuid,
    experiment_id: uuid::Uuid,
    experiment_key: String,
    config_hash: String,
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
    active: Arc<tokio::sync::Mutex<HashMap<uuid::Uuid, ActiveBtcRun>>>,
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
            active: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            terminal_pending: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    fn is_managed_process(process: &TradingProcess) -> bool {
        process.process_type == "btc_5m" && process.process_scope == "realtime_paper"
    }

    fn is_managed_identity(process_type: &str, process_scope: &str) -> bool {
        process_type == "btc_5m" && process_scope == "realtime_paper"
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
        self.validate_definition(process, true)
    }

    fn validate_resume_definition(
        &self,
        process: &TradingProcess,
    ) -> Result<ResolvedBtcProcessDefinition, HttpError> {
        self.validate_definition(process, false)
    }

    fn validate_definition(
        &self,
        process: &TradingProcess,
        require_inactive: bool,
    ) -> Result<ResolvedBtcProcessDefinition, HttpError> {
        if require_inactive {
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
        let mut control: BtcRealtimePaperControlConfig = serde_json::from_value(control_value)
            .map_err(|error| {
                HttpError::bad_request(format!(
                    "invalid process config.raw.btc_realtime_paper: {error}"
                ))
            })?;
        if control.schema_version != BTC_PROCESS_SCHEMA_VERSION {
            return Err(HttpError::bad_request(format!(
                "BTC process schema_version must be {BTC_PROCESS_SCHEMA_VERSION}"
            )));
        }
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
        let strategy_overrides = control.strategy.as_object().ok_or_else(|| {
            HttpError::bad_request("btc_realtime_paper.strategy must be a JSON object")
        })?;
        let mut strategy_value = serde_json::to_value(BtcStrategyConfig::default())
            .map_err(|error| HttpError::internal(error.to_string()))?;
        let strategy_object = strategy_value.as_object_mut().ok_or_else(|| {
            HttpError::internal("default BTC strategy did not serialize as object")
        })?;
        strategy_object.insert(
            "volatility_continuation".to_string(),
            serde_json::Value::Null,
        );
        for (key, value) in strategy_overrides {
            let Some(slot) = strategy_object.get_mut(key) else {
                return Err(HttpError::bad_request(format!(
                    "unsupported BTC strategy setting {key}"
                )));
            };
            *slot = value.clone();
        }
        let strategy: BtcStrategyConfig =
            serde_json::from_value(strategy_value).map_err(|error| {
                HttpError::bad_request(format!("invalid BTC strategy settings: {error}"))
            })?;
        strategy
            .validate()
            .map_err(|error| HttpError::bad_request(error.to_string()))?;
        if !matches!(
            strategy.strategy_version.as_str(),
            BTC_STRATEGY_VERSION | BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION
        ) || strategy.feature_schema_version != BTC_FEATURE_SCHEMA_VERSION
        {
            return Err(HttpError::bad_request(
                "BTC strategy and feature schema versions are compiled identities and cannot be overridden",
            ));
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
        if control.ml_shadow.enabled && !self.config.btc.ml_shadow_enabled {
            return Err(HttpError::bad_request(
                "BTC ML shadow was requested but is disabled by the deployment capability gate",
            ));
        }
        let ml_shadow_enabled = control.ml_shadow.enabled;
        Ok(ResolvedBtcProcessDefinition {
            control,
            strategy,
            runtime,
            paper_venue,
            paper_stress_previews,
            ml_shadow_enabled,
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
        if let Some(active) = self.active.lock().await.get(&process_id) {
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
              AND p.config #>> '{raw,btc_realtime_paper,schema_version}' = $1
              AND p.enabled
              AND p.status IN ('starting','running','stopping')
              AND p.stopped_at IS NULL
              AND e.status = 'running'
              AND e.stopped_at IS NULL
            ORDER BY e.started_at DESC
            "#,
        )
        .bind(BTC_PROCESS_SCHEMA_VERSION)
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
        let PreparedBtcStartDefinition {
            experiment_id,
            experiment_key,
            preregistration_sha256,
            strategy,
            runtime: runtime_config,
            paper_venue: paper_venue_config,
            paper_stress_previews,
            ml_shadow_enabled,
            frozen_process_config,
            config_hash: current_config_hash,
        } = self.prepare_resume_definition(&process)?;
        let current_frozen_process_config = serde_json::to_value(&frozen_process_config)
            .map_err(|error| HttpError::internal(error.to_string()))?;
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
        .bind(experiment_id)
        .bind(&experiment_key)
        .bind(process_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| HttpError::internal(error.to_string()))?
        .ok_or_else(|| {
            HttpError::conflict("durable BTC experiment disappeared before runtime reattachment")
        })?;
        if resume_config_without_compiled_source_identity(current_frozen_process_config)
            != resume_config_without_compiled_source_identity(frozen_process_config_value.clone())
        {
            return Err(HttpError::conflict(
                "durable BTC experiment parameters changed and cannot be resumed by this process definition",
            ));
        }
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

        let startup_result: Result<BtcRuntimeHandle> = async {
            let books = Arc::new(tokio::sync::RwLock::new(BookRegistry::new(
                uuid::Uuid::new_v4(),
            )));
            let paper_venue = BtcPaperVenue::new(books.clone(), paper_venue_config)?;
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
                    execution_enabled: true,
                    paper_stress_previews,
                    ml_a_shadow_enabled: ml_shadow_enabled,
                    ml_b_shadow_enabled: ml_shadow_enabled,
                },
            )?);
            experiment
                .resume()
                .await
                .context("failed to reattach immutable BTC experiment before feed resume")?;
            BtcRuntime::new(runtime_config, self.repository.clone())
                .with_shared_book_registry(books)
                .with_strategy_runner(experiment)
                .start()
                .await
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
        self.active.lock().await.insert(
            process_id,
            ActiveBtcRun {
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
            runtime: runtime_config,
            paper_venue: paper_venue_config,
            paper_stress_previews,
            ml_shadow_enabled,
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

        let startup_result: Result<BtcRuntimeHandle> = async {
            let books = Arc::new(tokio::sync::RwLock::new(BookRegistry::new(
                uuid::Uuid::new_v4(),
            )));
            let paper_venue = BtcPaperVenue::new(books.clone(), paper_venue_config)?;
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
                    execution_enabled: true,
                    paper_stress_previews: paper_stress_previews.clone(),
                    ml_a_shadow_enabled: ml_shadow_enabled,
                    ml_b_shadow_enabled: ml_shadow_enabled,
                },
            )?);
            experiment
                .initialize()
                .await
                .context("failed to initialize immutable BTC experiment before feed startup")?;
            BtcRuntime::new(runtime_config, self.repository.clone())
                .with_shared_book_registry(books)
                .with_strategy_runner(experiment)
                .start()
                .await
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

        self.active.lock().await.insert(
            process_id,
            ActiveBtcRun {
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
        match self.finalize_pending_locked(pending).await {
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
        let mut active_guard = self.active.lock().await;
        // Re-read after acquiring the lifecycle lock so concurrent, duplicate
        // stops observe the terminal state produced by the first request.
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

        if let Some(pending) = self.terminal_pending.lock().await.get(&process_id).cloned() {
            if expected_experiment_id.is_some_and(|expected| expected != pending.experiment_id) {
                return Ok(process);
            }
            drop(active_guard);
            return self.finalize_pending_locked(pending).await;
        }
        let Some(active) = active_guard.get(&process_id) else {
            if process.enabled
                || matches!(process.status.as_str(), "starting" | "running" | "stopping")
            {
                return Err(HttpError::conflict(
                    "BTC process claims to be active but this service owns no runtime handle",
                ));
            }
            return Ok(process);
        };
        if expected_experiment_id.is_some_and(|expected| expected != active.experiment_id) {
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
                "experiment_id": active.experiment_id,
                "experiment_key": active.experiment_key,
                "reason": reason,
            }),
        )
        .await;

        let active = active_guard
            .remove(&process_id)
            .expect("active BTC runtime exists while manager lock is held");
        drop(active_guard);
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
        self.finalize_pending_locked(pending).await
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
            .active
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
        let process_ids = self.active.lock().await.keys().copied().collect::<Vec<_>>();
        for process_id in process_ids {
            let snapshot = {
                let active_guard = self.active.lock().await;
                let Some(active) = active_guard.get(&process_id) else {
                    continue;
                };
                let status = active.runtime.status().await;
                (
                    active.experiment_id,
                    status.running,
                    status.metrics.last_error,
                )
            };
            let (experiment_id, runtime_running, last_error) = snapshot;
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
        let process_ids = self.active.lock().await.keys().copied().collect::<Vec<_>>();
        let mut processes = Vec::with_capacity(process_ids.len());
        for process_id in process_ids {
            processes.push(self.runtime_status_for_process(process_id).await);
        }
        serde_json::json!({
            "capability_enabled": self.config.btc.realtime_enabled,
            "active": !processes.is_empty(),
            "active_process_count": processes.len(),
            "processes": processes,
        })
    }

    async fn runtime_status_for_process(&self, process_id: uuid::Uuid) -> serde_json::Value {
        let active_guard = self.active.lock().await;
        if let Some(active) = active_guard.get(&process_id) {
            let runtime = active.runtime.status().await;
            return serde_json::json!({
                "capability_enabled": self.config.btc.realtime_enabled,
                "active": true,
                "process_id": active.process_id,
                "experiment_id": active.experiment_id,
                "experiment_key": active.experiment_key,
                "config_hash": active.config_hash,
                "runtime": runtime,
            });
        }
        drop(active_guard);
        if let Some(pending) = self.terminal_pending.lock().await.get(&process_id) {
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
        let process_ids = self.active.lock().await.keys().copied().collect::<Vec<_>>();
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
        let active = self.active.lock().await;
        let Some(active) = active.get(&process_id) else {
            drop(active);
            if let Some(pending) = self.terminal_pending.lock().await.get(&process_id) {
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
            .paper_experiment_status(active.experiment_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(serde_json::json!({
            "configured": true,
            "active": true,
            "process_id": active.process_id,
            "experiment_id": active.experiment_id,
            "experiment_key": active.experiment_key,
            "experiment": experiment,
        }))
    }
}

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
            "scan_errors": self.scan_errors,
            "uptime_secs": (Utc::now() - self.started_at).num_seconds()
        })
    }
}

struct RuntimeControl {
    store: Store,
    ingestion: IngestionRepository,
    gamma: GammaClient,
    data_api: DataApiClient,
    venues: ExecutionVenues,
    metrics: Arc<Mutex<RuntimeMetrics>>,
    btc_manager: Option<BtcProcessManager>,
}

#[derive(Clone)]
struct ExecutionVenues {
    sim: Arc<dyn ExecutionVenue>,
    paper: Arc<dyn ExecutionVenue>,
    live: Option<Arc<dyn ExecutionVenue>>,
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

#[derive(Debug, Default)]
struct ScanRunReport {
    markets_seen: usize,
    markets_persisted: usize,
    persist_errors: usize,
    scanner: ScannerCycleReport,
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

    async fn recompute_mrs_scores(
        &self,
        request: control_http::MrsRecomputeRequest,
    ) -> Result<serde_json::Value, HttpError> {
        let lookback_days = request.lookback_days.unwrap_or(150).clamp(1, 365);
        let limit = request.limit.unwrap_or(20_000).clamp(1, 100_000);
        let since = Utc::now() - chrono::Duration::days(lookback_days);
        let updated = self
            .store
            .recompute_mrs_scores_from_existing(since, limit)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        let top_scores = self
            .store
            .fetch_top_mrs_scores(10)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(serde_json::json!({
            "score_version": polymarket_bot::wallets::MRS_SCORE_VERSION,
            "updated_wallets": updated,
            "lookback_days": lookback_days,
            "limit": limit,
            "top_scores": top_scores
        }))
    }

    async fn recompute_mrs_segment_scores(
        &self,
        request: control_http::MrsRecomputeRequest,
    ) -> Result<serde_json::Value, HttpError> {
        let lookback_days = request.lookback_days.unwrap_or(150).clamp(1, 365);
        let limit = request.limit.unwrap_or(20_000).clamp(1, 100_000);
        let since = Utc::now() - chrono::Duration::days(lookback_days);
        let updated = self
            .store
            .recompute_wallet_segment_v2_scores_from_existing(since, limit)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        let summary = self
            .store
            .wallet_segment_summary(20)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(serde_json::json!({
            "score_version": polymarket_bot::segments::MRS_SEGMENT_V2_SCORE_VERSION,
            "classifier_version": polymarket_bot::segments::GAMMA_SEGMENT_CLASSIFIER_VERSION,
            "updated_segments": updated,
            "lookback_days": lookback_days,
            "limit": limit,
            "summary": summary
        }))
    }

    async fn enqueue_wallet_score_refresh(
        &self,
        request: control_http::WalletScoreRefreshEnqueueRequest,
    ) -> Result<serde_json::Value, HttpError> {
        if request.wallets.is_empty() {
            return Err(HttpError::bad_request(
                "wallets must contain at least one wallet",
            ));
        }
        let score_version = request
            .score_version
            .unwrap_or_else(|| polymarket_bot::wallets::MRS_SCORE_VERSION.to_string());
        let segment_score_version = request
            .segment_score_version
            .unwrap_or_else(|| polymarket_bot::segments::MRS_SEGMENT_V2_SCORE_VERSION.to_string());
        validate_supported_score_versions(&score_version, &segment_score_version)?;
        let reason = request
            .reason
            .as_deref()
            .map(str::trim)
            .filter(|reason| !reason.is_empty())
            .unwrap_or("admin_requested");
        let mut jobs = Vec::new();
        for wallet in request.wallets.iter().take(1_000) {
            let wallet = wallet.trim();
            if wallet.is_empty() {
                continue;
            }
            let job = self
                .store
                .enqueue_wallet_score_refresh(
                    wallet,
                    &score_version,
                    &segment_score_version,
                    reason,
                    None,
                    request.metadata.clone(),
                )
                .await
                .map_err(|error| HttpError::internal(error.to_string()))?;
            jobs.push(job);
        }
        Ok(serde_json::json!({
            "enqueued": jobs.len(),
            "score_version": score_version,
            "segment_score_version": segment_score_version,
            "jobs": jobs
        }))
    }

    async fn process_wallet_score_refresh(
        &self,
        request: control_http::WalletScoreRefreshProcessRequest,
    ) -> Result<serde_json::Value, HttpError> {
        let lookback_days = request.lookback_days.unwrap_or(150).clamp(1, 365);
        let page_limit = request.page_limit.unwrap_or(50).clamp(1, 250);
        let max_pages = request.max_pages.unwrap_or(1).clamp(1, 10);
        let limit = request.limit.unwrap_or(25).clamp(1, 250);
        let score_version = request
            .score_version
            .unwrap_or_else(|| polymarket_bot::wallets::MRS_SCORE_VERSION.to_string());
        let segment_score_version = request
            .segment_score_version
            .unwrap_or_else(|| polymarket_bot::segments::MRS_SEGMENT_V2_SCORE_VERSION.to_string());
        validate_supported_score_versions(&score_version, &segment_score_version)?;

        let mut claimed_jobs = Vec::new();
        let wallets = if request.wallets.is_empty() {
            if request.use_queue.unwrap_or(true) {
                claimed_jobs = self
                    .store
                    .claim_wallet_score_refresh_jobs(limit)
                    .await
                    .map_err(|error| HttpError::internal(error.to_string()))?;
                claimed_jobs
                    .iter()
                    .map(|job| job.proxy_wallet.clone())
                    .collect::<Vec<_>>()
            } else {
                return Err(HttpError::bad_request(
                    "wallets must be provided when use_queue is false",
                ));
            }
        } else {
            request
                .wallets
                .iter()
                .take(limit as usize)
                .map(|wallet| wallet.trim().to_ascii_lowercase())
                .filter(|wallet| !wallet.is_empty())
                .collect::<Vec<_>>()
        };

        let since = Utc::now() - chrono::Duration::days(lookback_days);
        let mut processed = 0u64;
        let mut failed = 0u64;
        let mut segment_scores_updated = 0u64;
        let mut results = Vec::new();

        for wallet in wallets {
            let queue_id = claimed_jobs
                .iter()
                .find(|job| job.proxy_wallet.eq_ignore_ascii_case(&wallet))
                .map(|job| job.queue_id);
            match recompute_single_wallet_scores(
                &self.store,
                &self.data_api,
                &wallet,
                since,
                lookback_days,
                page_limit,
                max_pages,
                false,
            )
            .await
            {
                Ok(report) => {
                    processed = processed.saturating_add(1);
                    segment_scores_updated =
                        segment_scores_updated.saturating_add(report.segment_scores_updated);
                    if let Some(queue_id) = queue_id {
                        self.store
                            .complete_wallet_score_refresh_job(
                                queue_id,
                                serde_json::to_value(&report)
                                    .map_err(|error| HttpError::internal(error.to_string()))?,
                            )
                            .await
                            .map_err(|error| HttpError::internal(error.to_string()))?;
                    }
                    results.push(
                        serde_json::to_value(report)
                            .map_err(|error| HttpError::internal(error.to_string()))?,
                    );
                }
                Err(error) => {
                    failed = failed.saturating_add(1);
                    if let Some(queue_id) = queue_id {
                        if let Err(fail_error) = self
                            .store
                            .fail_wallet_score_refresh_job(queue_id, &error.to_string())
                            .await
                        {
                            warn!(
                                error = %fail_error,
                                queue_id = %queue_id,
                                "failed to update wallet score refresh job failure"
                            );
                        }
                    }
                    results.push(serde_json::json!({
                        "proxy_wallet": wallet,
                        "status": "failed",
                        "error": error.to_string()
                    }));
                }
            }
        }

        if request.refresh_percentiles && segment_scores_updated > 0 {
            self.store
                .refresh_wallet_segment_v2_percentiles()
                .await
                .map_err(|error| HttpError::internal(error.to_string()))?;
        }

        Ok(serde_json::json!({
            "status": "completed",
            "processed_wallets": processed,
            "failed_wallets": failed,
            "segment_scores_updated": segment_scores_updated,
            "lookback_days": lookback_days,
            "page_limit": page_limit,
            "max_pages": max_pages,
            "refresh_percentiles": request.refresh_percentiles,
            "results": results
        }))
    }

    async fn list_wallet_score_refresh_jobs(
        &self,
        request: control_http::WalletScoreRefreshJobsRequest,
    ) -> Result<serde_json::Value, HttpError> {
        let status = request
            .status
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let jobs = self
            .store
            .list_wallet_score_refresh_jobs(status, request.limit.unwrap_or(50))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        Ok(serde_json::json!({ "jobs": jobs }))
    }

    async fn mrs_segment_summary(
        &self,
        request: control_http::WalletRowsRequest,
    ) -> Result<serde_json::Value, HttpError> {
        self.store
            .wallet_segment_summary(request.limit.unwrap_or(20).clamp(1, 100))
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn gamma_taxonomy_status(&self) -> Result<serde_json::Value, HttpError> {
        self.store
            .wallet_trade_taxonomy_status()
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn gamma_taxonomy_backfill(
        &self,
        request: control_http::GammaTaxonomyBackfillRequest,
    ) -> Result<serde_json::Value, HttpError> {
        let limit = request.limit.unwrap_or(500).clamp(1, 5_000);
        let candidates = self
            .store
            .fetch_unresolved_wallet_trade_taxonomy_candidates(limit)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?;
        let mut gamma_event_hits = 0usize;
        let mut gamma_market_hits = 0usize;
        let mut dry_run_updates = 0usize;
        let mut updated_trades = 0u64;
        let mut errors = Vec::new();

        for candidate in &candidates {
            let mut metadata = None;
            if let Some(event_slug) = candidate
                .event_slug
                .as_deref()
                .filter(|slug| !slug.is_empty())
            {
                metadata = self
                    .store
                    .fetch_gamma_market_metadata_by_lookup("event_slug", event_slug)
                    .await
                    .map_err(|error| HttpError::internal(error.to_string()))?;
                if metadata.is_none() {
                    match self.gamma.fetch_event_taxonomy_by_slug(event_slug).await {
                        Ok(Some(fetched)) => {
                            if !request.dry_run {
                                self.store
                                    .upsert_gamma_market_metadata(&fetched)
                                    .await
                                    .map_err(|error| HttpError::internal(error.to_string()))?;
                            }
                            metadata = Some(fetched);
                        }
                        Ok(None) => {}
                        Err(error) => errors.push(serde_json::json!({
                            "trade_id": candidate.trade_id,
                            "lookup": "event_slug",
                            "slug": event_slug,
                            "error": error.to_string()
                        })),
                    }
                }
                if metadata.is_some() {
                    gamma_event_hits += 1;
                }
            }

            if metadata.is_none() {
                if let Some(slug) = candidate.slug.as_deref().filter(|slug| !slug.is_empty()) {
                    metadata = self
                        .store
                        .fetch_gamma_market_metadata_by_lookup("market_slug", slug)
                        .await
                        .map_err(|error| HttpError::internal(error.to_string()))?;
                    if metadata.is_none() {
                        match self.gamma.fetch_market_taxonomy_by_slug(slug).await {
                            Ok(Some(fetched)) => {
                                if !request.dry_run {
                                    self.store
                                        .upsert_gamma_market_metadata(&fetched)
                                        .await
                                        .map_err(|error| HttpError::internal(error.to_string()))?;
                                }
                                metadata = Some(fetched);
                            }
                            Ok(None) => {}
                            Err(error) => errors.push(serde_json::json!({
                                "trade_id": candidate.trade_id,
                                "lookup": "market_slug",
                                "slug": slug,
                                "error": error.to_string()
                            })),
                        }
                    }
                    if metadata.is_some() {
                        gamma_market_hits += 1;
                    }
                }
            }

            let update = metadata
                .as_ref()
                .and_then(|metadata| taxonomy_update_from_metadata(candidate, metadata));

            if let Some(update) = update {
                if request.dry_run {
                    dry_run_updates += 1;
                } else {
                    updated_trades += self
                        .store
                        .update_wallet_trade_taxonomy(&update)
                        .await
                        .map_err(|error| HttpError::internal(error.to_string()))?;
                }
            }
        }

        let error_count = errors.len();
        Ok(serde_json::json!({
            "taxonomy_version": polymarket_bot::taxonomy::GAMMA_TAXONOMY_VERSION,
            "dry_run": request.dry_run,
            "fallback_keywords": false,
            "fallback_keywords_ignored": request.fallback_keywords,
            "candidates": candidates.len(),
            "gamma_event_hits": gamma_event_hits,
            "gamma_market_hits": gamma_market_hits,
            "fallback_hits": 0,
            "dry_run_updates": dry_run_updates,
            "updated_trades": updated_trades,
            "errors": errors,
            "error_count": error_count
        }))
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

    async fn live_wallet_address_diagnostics(
        &self,
        candidate_addresses: Vec<String>,
    ) -> Result<LiveWalletAddressDiagnostics, HttpError> {
        self.venues
            .for_mode(ExecutionMode::Live)
            .map_err(|error| HttpError::bad_request(error.to_string()))?
            .live_wallet_address_diagnostics(candidate_addresses)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_order_dry_run(
        &self,
        request: LiveOrderDryRunRequest,
    ) -> Result<LiveOrderDryRunDiagnostics, HttpError> {
        self.venues
            .for_mode(ExecutionMode::Live)
            .map_err(|error| HttpError::bad_request(error.to_string()))?
            .live_order_dry_run(request)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
    }

    async fn live_poly1271_funder_probe(
        &self,
        request: LivePoly1271FunderProbeRequest,
    ) -> Result<LivePoly1271FunderProbeResponse, HttpError> {
        self.venues
            .for_mode(ExecutionMode::Live)
            .map_err(|error| HttpError::bad_request(error.to_string()))?
            .live_poly1271_funder_probe(request)
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

    async fn live_account_reconcile(
        &self,
        request: control_http::AccountReconcileRequest,
    ) -> Result<control_http::AccountReconcileReport, HttpError> {
        self.venues
            .for_mode(ExecutionMode::Live)
            .map_err(|error| HttpError::bad_request(error.to_string()))?
            .live_account_reconcile(request)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))
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
        if BtcProcessManager::is_managed_identity(process_type, process_scope) {
            return Err(HttpError::bad_request(
                "create is disabled for managed BTC definitions; use stable-key PUT, then the process /start endpoint",
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
        if BtcProcessManager::is_managed_identity(process_type, process_scope) {
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
                HttpError::bad_request(
                    "BTC realtime-paper capability is disabled for this deployment",
                )
            })?;
            let now = Utc::now();
            manager.validate_start_definition(&TradingProcess {
                process_id: uuid::Uuid::nil(),
                name: name.to_string(),
                process_type: process_type.to_string(),
                process_scope: process_scope.to_string(),
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
                .get_trading_process_by_key(process_type, process_scope, key)
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
                        .active
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
            drop(lifecycle_guard);
            return Ok(TradingProcessResponse { process });
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
        let proposed_type = process_type.unwrap_or(current.process_type.as_str());
        let proposed_scope = process_scope.unwrap_or(current.process_scope.as_str());
        let managed_now = BtcProcessManager::is_managed_process(&current);
        let managed_after = BtcProcessManager::is_managed_identity(proposed_type, proposed_scope);
        if !managed_now && managed_after {
            return Err(HttpError::bad_request(
                "an existing generic process cannot be converted into a managed BTC process; create a dedicated stopped BTC definition",
            ));
        }
        if managed_now || managed_after {
            let runtime_active = match &self.btc_manager {
                Some(manager) => manager.active.lock().await.contains_key(&process_id),
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
            if request.enabled.is_some() || request.status.is_some() {
                return Err(HttpError::bad_request(
                    "BTC enabled/status fields are lifecycle-owned; use the /start and /stop endpoints",
                ));
            }
            if managed_now {
                if process_type.is_some_and(|value| value != current.process_type)
                    || process_scope.is_some_and(|value| value != current.process_scope)
                    || process_key.is_some()
                {
                    return Err(HttpError::bad_request(
                        "BTC process type, scope, and stable key are immutable; name and config remain mutable while stopped",
                    ));
                }
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
        if BtcProcessManager::is_managed_process(&current) {
            let manager = self.btc_manager.as_ref().ok_or_else(|| {
                HttpError::bad_request(
                    "BTC realtime-paper capability is disabled for this deployment",
                )
            })?;
            let process = manager.start_process(process_id).await?;
            return Ok(TradingProcessResponse { process });
        }
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
        let current = self
            .store
            .get_trading_process(process_id)
            .await
            .map_err(|error| HttpError::internal(error.to_string()))?
            .ok_or_else(|| HttpError::not_found("trading process not found"))?;
        if BtcProcessManager::is_managed_process(&current) {
            let manager = self.btc_manager.as_ref().ok_or_else(|| {
                HttpError::bad_request(
                    "BTC realtime-paper capability is disabled for this deployment",
                )
            })?;
            let process = manager.stop_process(process_id, "api_stop").await?;
            return Ok(TradingProcessResponse { process });
        }
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

#[derive(Debug, Serialize)]
struct WalletScoreRefreshProcessReport {
    proxy_wallet: String,
    closed_positions: usize,
    mrs_score: rust_decimal::Decimal,
    segment_scores_updated: u64,
}

fn validate_supported_score_versions(
    score_version: &str,
    segment_score_version: &str,
) -> Result<(), HttpError> {
    if score_version != polymarket_bot::wallets::MRS_SCORE_VERSION {
        return Err(HttpError::bad_request(format!(
            "unsupported score_version {score_version}"
        )));
    }
    if segment_score_version != polymarket_bot::segments::MRS_SEGMENT_V2_SCORE_VERSION {
        return Err(HttpError::bad_request(format!(
            "unsupported segment_score_version {segment_score_version}"
        )));
    }
    Ok(())
}

fn merge_json(mut left: serde_json::Value, right: serde_json::Value) -> serde_json::Value {
    if let (Some(left), Some(right)) = (left.as_object_mut(), right.as_object()) {
        for (key, value) in right {
            left.insert(key.clone(), value.clone());
        }
    }
    left
}

async fn fetch_closed_positions_for_wallet(
    data_api: &DataApiClient,
    wallet: &str,
    page_limit: usize,
    max_pages: usize,
) -> Result<Vec<DataApiClosedPosition>> {
    let page_limit = page_limit.max(1);
    let max_pages = max_pages.max(1);
    let mut wallet_positions = Vec::new();
    for page in 0..max_pages {
        let mut query = ClosedPositionsQuery::for_user(wallet);
        query.limit = Some(page_limit);
        query.offset = Some(page * page_limit);
        let positions = match data_api.fetch_closed_positions(&query).await {
            Ok(positions) => positions,
            Err(error) => {
                warn!(
                    error = %error,
                    wallet,
                    page,
                    "failed to fetch closed positions for wallet; continuing"
                );
                break;
            }
        };
        let fetched = positions.len();
        wallet_positions.extend(positions);
        if fetched < page_limit {
            break;
        }
    }
    Ok(wallet_positions)
}

async fn recompute_single_wallet_scores(
    store: &Store,
    data_api: &DataApiClient,
    proxy_wallet: &str,
    since: chrono::DateTime<Utc>,
    lookback_days: i64,
    page_limit: usize,
    max_pages: usize,
    refresh_percentiles_inline: bool,
) -> Result<WalletScoreRefreshProcessReport> {
    let wallet = proxy_wallet.trim().to_ascii_lowercase();
    if wallet.is_empty() {
        bail!("proxy wallet is required");
    }
    store
        .ensure_wallet_address(
            &wallet,
            Utc::now(),
            serde_json::json!({
                "source": "wallet_score_refresh"
            }),
        )
        .await?;
    let positions =
        fetch_closed_positions_for_wallet(data_api, &wallet, page_limit, max_pages).await?;
    let performance_score = score_closed_position_performance(&wallet, &positions);
    let performance = performance_score
        .clone()
        .into_wallet_performance(Utc::now(), serde_json::to_value(&positions)?);
    store.upsert_wallet_performance(&performance).await?;
    let observed_stats = store
        .fetch_wallet_observed_trade_stats(&wallet, since)
        .await?;
    let mut mrs_input = MrsScoreInput::from(&performance_score);
    mrs_input.observed_trade_count = observed_stats.observed_trade_count;
    mrs_input.observed_volume_usd = observed_stats.observed_volume_usd;
    mrs_input.observed_market_count = observed_stats.observed_market_count;
    mrs_input.avg_trade_size = observed_stats.avg_trade_size;
    let mut mrs_score = score_mrs(mrs_input).into_wallet_score();
    mrs_score.metadata = merge_json(
        mrs_score.metadata,
        serde_json::json!({
            "source": "wallet_score_refresh_job",
            "lookback_days": lookback_days,
            "sample_start": observed_stats.sample_start,
            "sample_end": observed_stats.sample_end
        }),
    );
    store.upsert_wallet_score(&mrs_score).await?;
    let segment_scores_updated = store
        .recompute_wallet_segment_v2_scores_for_wallets(
            std::slice::from_ref(&wallet),
            since,
            refresh_percentiles_inline,
        )
        .await?;
    let report = WalletScoreRefreshProcessReport {
        proxy_wallet: wallet,
        closed_positions: positions.len(),
        mrs_score: mrs_score.score,
        segment_scores_updated,
    };
    Ok(report)
}

#[tokio::main]
async fn main() -> Result<()> {
    install_tls_crypto_provider();
    init_tracing();

    let config = AppConfig::from_env()?;
    info!(
        service = "polymarket-bot",
        scan_enabled = config.scan_enabled,
        live_order_submit_enabled = config.live.order_submit_enabled,
        live_user_ws_enabled = config.live.user_ws_enabled,
        btc_realtime_enabled = config.btc.realtime_enabled,
        btc_paper_enabled = config.btc.paper_enabled,
        btc_ml_shadow_enabled = config.btc.ml_shadow_enabled,
        compiled_source_identity = COMPILED_SOURCE_IDENTITY,
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
                "live_order_submit_enabled": config.live.order_submit_enabled,
                "live_user_ws_enabled": config.live.user_ws_enabled,
                "btc_realtime_enabled": config.btc.realtime_enabled,
                "btc_paper_enabled": config.btc.paper_enabled,
                "btc_ml_shadow_enabled": config.btc.ml_shadow_enabled,
                "compiled_source_identity": COMPILED_SOURCE_IDENTITY,
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
            data_api.clone(),
        )?))
    } else {
        None
    };
    let venues = ExecutionVenues {
        sim: sim_venue.clone(),
        paper: paper_venue.clone(),
        live: live_venue.clone(),
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
    let btc_manager = if config.btc.realtime_enabled {
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .connect(&config.postgres.database_url())
            .await
            .context("failed to connect BTC process manager to Postgres")?;
        let repository = BtcRepository::from_pool(pool.clone());
        Some(BtcProcessManager::new(
            store.clone(),
            pool,
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

    let mut metrics = RuntimeMetrics::new();
    let shared_metrics = Arc::new(Mutex::new(metrics.clone()));
    if config.http.enabled {
        let ingestion = IngestionRepository::connect(&config.postgres)
            .await
            .context("failed to connect backfill ingestion repository")?;
        let control: control_http::SharedControlApi = Arc::new(RuntimeControl {
            store: store.clone(),
            ingestion,
            gamma: gamma.clone(),
            data_api: data_api.clone(),
            venues: venues.clone(),
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
    let mut scan_interval = tokio::time::interval(config.scan_interval);
    scan_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut health_interval = tokio::time::interval(config.health_interval);
    health_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            biased;

            _ = &mut shutdown => {
                warn!("shutdown signal received");
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
                let reconciliation = venues.sim.reconcile().await;
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
                        venues.sim.as_ref(),
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

            _ = tokio::time::sleep(Duration::from_secs(3600)), if !config.scan_enabled => {
                info!("scan disabled; service idling");
            }
        }
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
        assert!(!control.ml_shadow.enabled);
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
            runtime: BtcRuntimeConfig {
                enabled: true,
                ..BtcRuntimeConfig::default()
            },
            paper_venue: PaperVenueConfig::default(),
            paper_stress_previews: Vec::new(),
            ml_shadow_enabled: false,
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
            first
                .frozen_process_config
                .execution
                .as_ref()
                .and_then(|execution| execution.mode.as_deref()),
            Some("paper")
        );
    }

    #[test]
    fn btc_runtime_ownership_is_independent_per_process_id() {
        let first_process_id = uuid::Uuid::new_v4();
        let second_process_id = uuid::Uuid::new_v4();
        let mut owned_runtimes = HashMap::new();
        let mut pending_terminals = HashMap::new();

        owned_runtimes.insert(first_process_id, "first-runtime");
        assert!(owned_runtimes.contains_key(&first_process_id));
        assert!(!owned_runtimes.contains_key(&second_process_id));

        owned_runtimes.insert(second_process_id, "second-runtime");
        pending_terminals.insert(first_process_id, "first-terminal");
        assert_eq!(owned_runtimes.len(), 2);
        assert!(pending_terminals.contains_key(&first_process_id));
        assert!(!pending_terminals.contains_key(&second_process_id));

        owned_runtimes.remove(&first_process_id);
        pending_terminals.remove(&first_process_id);
        assert_eq!(
            owned_runtimes.get(&second_process_id),
            Some(&"second-runtime")
        );
        assert!(pending_terminals.is_empty());
    }

    #[test]
    fn btc_resume_preserves_frozen_parameters_across_service_rebuilds() {
        let durable = serde_json::json!({
            "raw": {
                "build": {
                    "package_version": "0.1.0",
                    "compiled_source_identity": "tree-sha256:old"
                },
                "strategy": {"threshold": "0.03"}
            }
        });
        let rebuilt = serde_json::json!({
            "raw": {
                "build": {
                    "package_version": "0.1.0",
                    "compiled_source_identity": "tree-sha256:new"
                },
                "strategy": {"threshold": "0.03"}
            }
        });
        assert_eq!(
            resume_config_without_compiled_source_identity(durable.clone()),
            resume_config_without_compiled_source_identity(rebuilt)
        );

        let changed_parameters = serde_json::json!({
            "raw": {
                "build": {
                    "package_version": "0.1.0",
                    "compiled_source_identity": "tree-sha256:new"
                },
                "strategy": {"threshold": "0.04"}
            }
        });
        assert_ne!(
            resume_config_without_compiled_source_identity(durable),
            resume_config_without_compiled_source_identity(changed_parameters)
        );
    }
}
