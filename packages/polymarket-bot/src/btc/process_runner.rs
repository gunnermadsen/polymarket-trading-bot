use super::unified_model_runtime::telemetry as umr_telemetry;
use std::{
    collections::HashSet,
    str::FromStr,
    sync::{Arc, Mutex as StdMutex},
};

use anyhow::{ensure, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::{prelude::ToPrimitive, Decimal};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    sync::{Mutex, OnceCell, RwLock},
    time::{Duration as TokioDuration, Instant},
};
use tracing::warn;
use uuid::Uuid;

use crate::{
    execution::{execute_order_plan, ExecutionVenue, OrderPlan, OrderPlanReport},
    models::{OrderRequest, OrderSide, OrderState, OrderType},
    store::Store,
};

use super::{
    admission::{
        AdmissionDisposition, BtcEntryAdmissionConfig, DailyRealizedPnlHighWaterMarkEvaluation,
        LossRegimeConfidenceFloorEvaluation, LossRegimeConfidenceFloorState,
        LossRegimeConfidenceFloorTransition, ProposedEntryExposure,
    },
    asymmetric_value_features::build_asymmetric_value_feature_snapshot,
    directional_external_runtime::DirectionalExternalState,
    directional_features::{
        build_directional_features_for_schema_with_external, build_payoff_aware_feature_values,
        directional_external_feature_requirements, directional_schema_requires_opening_boundary,
        DirectionalBinanceOpenInterest, DirectionalChainlinkCandle, DirectionalChainlinkRefPrice,
        DirectionalExternalFeatureInputs, DirectionalFeatureError, DirectionalFeatureVector,
        DirectionalOracleRound,
    },
    directional_model::{
        directional_model_input_sha256, runtime_model, BtcDirectionalModelFeatureSnapshot,
        RuntimeModelSelection, RuntimePredictionPolicy, BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION,
    },
    execution_guard::{BtcExecutionFreshnessBounds, BtcReferenceExecutionGuard},
    execution_lifecycle::{BtcExecutionLifecycle, BtcExecutionMode, PaperExecutionLifecycle},
    feed_contract::BtcModelFeedId,
    feeds::BookRegistry,
    paper::{PaperPreviewConfig, PaperVenue, PAPER_DYNAMIC_FEE_RATE_METADATA_KEY},
    repository::{BtcPointInTimeInputs, BtcRepository},
    runtime::{BtcStrategyRunner, StrategyObservation},
    strategy::{
        BtcDecision, BtcDecisionAction, BtcDecisionStrategyConfig, BtcDirectionalModelEntryPolicy,
        BtcFeatureLineage, BtcFeatureSnapshot, BtcInputWindowLineage, BtcOutcomeBookFeatures,
        BtcRejectReason, BtcStrategyConfig, BtcStrategyPrediction, DeterministicBtcStrategy,
        BTC_DIRECTIONAL_MODEL_STRATEGY_FAMILY, BTC_FEATURE_LINEAGE_VERSION,
    },
    types::{
        BtcIntervalMarket, BtcOutcome, FeedIntegrityStatus, OrderbookCheckpoint, RealtimeState,
        ReferencePriceSource, ReferencePriceTick,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BtcProcessConfig {
    pub process_id: Uuid,
    pub run_id: Uuid,
    pub run_key: String,
    pub config_hash: String,
    /// Full immutable `TradingProcessConfig` snapshot for this execution run.
    /// The reusable process definition may be changed after the run stops, so
    /// evidence consumers must read this run-owned value instead.
    pub frozen_process_config: serde_json::Value,
    pub strategy: BtcStrategyConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_admission: Option<BtcEntryAdmissionConfig>,
    #[serde(
        default,
        skip_serializing_if = "BtcDirectionalModelEntryPolicy::is_default"
    )]
    pub directional_model_entry_policy: BtcDirectionalModelEntryPolicy,
    pub execution_enabled: bool,
    pub paper_stress_previews: Vec<PaperPreviewConfig>,
}

pub type BtcPaperProcessConfig = BtcProcessConfig;

#[derive(Debug, Default)]
struct LossRegimeAdmissionRuntime {
    state: LossRegimeConfidenceFloorState,
    evaluated_market_id: Option<String>,
}

#[derive(Debug, Default)]
struct DirectionalModelProcessRuntime {
    market_id: Option<String>,
    last_candidate_at: Option<DateTime<Utc>>,
    pending_candidate_at: Option<DateTime<Utc>>,
    in_flight_candidate_at: Option<DateTime<Utc>>,
    opportunity_sealed: bool,
    rehydrated: bool,
}

impl DirectionalModelProcessRuntime {
    fn claim(
        &mut self,
        market_id: &str,
        latest_feature_as_of: DateTime<Utc>,
    ) -> Option<(DateTime<Utc>, bool)> {
        if self.market_id.as_deref() != Some(market_id) {
            self.market_id = Some(market_id.to_string());
            self.last_candidate_at = None;
            self.pending_candidate_at = None;
            self.in_flight_candidate_at = None;
            self.opportunity_sealed = false;
            self.rehydrated = false;
        }
        if self.opportunity_sealed
            || self
                .last_candidate_at
                .is_some_and(|last_candidate_at| last_candidate_at >= latest_feature_as_of)
            || self.in_flight_candidate_at.is_some()
        {
            return None;
        }
        let feature_as_of = *self
            .pending_candidate_at
            .get_or_insert(latest_feature_as_of);
        self.in_flight_candidate_at = Some(feature_as_of);
        let requires_rehydration = !self.rehydrated;
        Some((feature_as_of, requires_rehydration))
    }

    fn mark_rehydrated(&mut self, market_id: &str, feature_as_of: DateTime<Utc>) {
        if self.market_id.as_deref() == Some(market_id)
            && self.in_flight_candidate_at == Some(feature_as_of)
        {
            self.rehydrated = true;
        }
    }

    fn complete(
        &mut self,
        market_id: &str,
        feature_as_of: DateTime<Utc>,
        seal_opportunity: bool,
    ) -> bool {
        if self.market_id.as_deref() != Some(market_id)
            || self.in_flight_candidate_at != Some(feature_as_of)
        {
            return false;
        }
        self.last_candidate_at = Some(feature_as_of);
        self.pending_candidate_at = None;
        self.in_flight_candidate_at = None;
        self.rehydrated = true;
        if seal_opportunity {
            self.opportunity_sealed = true;
        }
        true
    }

    fn release(&mut self, market_id: &str, feature_as_of: DateTime<Utc>) {
        if self.market_id.as_deref() == Some(market_id)
            && self.in_flight_candidate_at == Some(feature_as_of)
        {
            self.in_flight_candidate_at = None;
        }
    }
}

struct DirectionalModelCandidateLease<'a> {
    runtime: &'a StdMutex<DirectionalModelProcessRuntime>,
    market_id: String,
    feature_as_of: DateTime<Utc>,
    requires_rehydration: bool,
    completed: bool,
}

impl DirectionalModelCandidateLease<'_> {
    fn feature_as_of(&self) -> DateTime<Utc> {
        self.feature_as_of
    }

    fn requires_rehydration(&self) -> bool {
        self.requires_rehydration
    }

    fn mark_rehydrated(&self) -> Result<()> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("BTC directional model process lock was poisoned"))?;
        runtime.mark_rehydrated(&self.market_id, self.feature_as_of);
        Ok(())
    }

    fn complete(mut self, seal_opportunity: bool) -> Result<()> {
        let completed = {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| anyhow::anyhow!("BTC directional model process lock was poisoned"))?;
            runtime.complete(&self.market_id, self.feature_as_of, seal_opportunity)
        };
        anyhow::ensure!(
            completed,
            "BTC directional model candidate lease no longer owns its runtime candidate"
        );
        self.completed = true;
        Ok(())
    }
}

impl Drop for DirectionalModelCandidateLease<'_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if let Ok(mut runtime) = self.runtime.lock() {
            runtime.release(&self.market_id, self.feature_as_of);
        }
    }
}

fn directional_feature_error_metadata(error: &DirectionalFeatureError) -> serde_json::Value {
    serde_json::json!({
        "code": error.code(),
        "detail": error.to_string(),
    })
}

#[derive(Debug)]
struct DirectionalExternalDecisionSnapshot {
    oracle_rounds: Vec<DirectionalOracleRound>,
    refprice_reports: Vec<DirectionalChainlinkRefPrice>,
    chainlink_candles: Vec<DirectionalChainlinkCandle>,
    open_interest: Vec<DirectionalBinanceOpenInterest>,
}

impl DirectionalExternalDecisionSnapshot {
    fn inputs(&self) -> DirectionalExternalFeatureInputs<'_> {
        DirectionalExternalFeatureInputs {
            oracle_rounds: &self.oracle_rounds,
            refprice_reports: &self.refprice_reports,
            chainlink_candles: &self.chainlink_candles,
            open_interest: &self.open_interest,
        }
    }
}

fn directional_external_decision_snapshot(
    state: &DirectionalExternalState,
    feature_as_of: DateTime<Utc>,
    schema_version: &str,
) -> Result<Option<DirectionalExternalDecisionSnapshot>, DirectionalFeatureError> {
    let requirements = directional_external_feature_requirements(schema_version);
    if requirements == Default::default() {
        return Ok(None);
    }

    let oracle_rounds = if requirements.oracle {
        state
            .oracle
            .iter()
            .filter(|point| {
                point.source_timestamp <= feature_as_of
                    && point.block_timestamp <= feature_as_of
                    && point.available_at <= feature_as_of
            })
            .map(|point| {
                let aggregator_round_id = i64::try_from(point.round_id).map_err(|_| {
                    external_snapshot_error(
                        "polygon_oracle",
                        "runtime round identity exceeded the feature contract",
                    )
                })?;
                if point.phase_id == 0 || aggregator_round_id <= 0 {
                    return Err(external_snapshot_error(
                        "polygon_oracle",
                        "runtime round identity was invalid",
                    ));
                }
                Ok(DirectionalOracleRound {
                    phase_id: i32::from(point.phase_id),
                    aggregator_round_id,
                    source_timestamp: point.source_timestamp,
                    block_timestamp: point.block_timestamp,
                    // latestRoundData exposes the signed round identity but not its transaction
                    // provenance. Preserve that distinction instead of fabricating block fields.
                    block_number: None,
                    log_index: None,
                    price: external_decimal_value(
                        point.price,
                        "polygon_oracle",
                        "runtime price was invalid",
                    )?,
                    available_at: point.available_at,
                })
            })
            .collect::<Result<Vec<_>, DirectionalFeatureError>>()?
    } else {
        Vec::new()
    };

    let refprice_reports = if requirements.refprice {
        state
            .refprice
            .iter()
            .filter(|point| {
                point.source_timestamp <= feature_as_of && point.available_at <= feature_as_of
            })
            .map(|point| {
                Ok(DirectionalChainlinkRefPrice {
                    source_timestamp: point.source_timestamp,
                    valid_from_timestamp: point.valid_from_timestamp,
                    price: external_decimal_value(
                        point.price,
                        "chainlink_refprice",
                        "runtime price was invalid",
                    )?,
                    bid: external_decimal_value(
                        point.bid,
                        "chainlink_refprice",
                        "runtime bid was invalid",
                    )?,
                    ask: external_decimal_value(
                        point.ask,
                        "chainlink_refprice",
                        "runtime ask was invalid",
                    )?,
                    available_at: point.available_at,
                })
            })
            .collect::<Result<Vec<_>, DirectionalFeatureError>>()?
    } else {
        Vec::new()
    };

    let chainlink_candles = if requirements.chainlink_candles {
        state.rtds().closed_candles(feature_as_of)?
    } else {
        Vec::new()
    };

    let open_interest = if requirements.open_interest {
        state
            .open_interest
            .iter()
            .filter(|point| {
                point.source_timestamp < feature_as_of && point.available_at <= feature_as_of
            })
            .map(|point| {
                Ok(DirectionalBinanceOpenInterest {
                    source_timestamp: point.source_timestamp,
                    period_seconds: 300,
                    sum_open_interest: external_decimal_value(
                        point.sum_open_interest,
                        "binance_open_interest",
                        "runtime open interest was invalid",
                    )?,
                    sum_open_interest_value: external_decimal_value(
                        point.sum_open_interest_value,
                        "binance_open_interest",
                        "runtime open-interest value was invalid",
                    )?,
                    available_at: point.available_at,
                })
            })
            .collect::<Result<Vec<_>, DirectionalFeatureError>>()?
    } else {
        Vec::new()
    };

    Ok(Some(DirectionalExternalDecisionSnapshot {
        oracle_rounds,
        refprice_reports,
        chainlink_candles,
        open_interest,
    }))
}

fn external_decimal_value(
    value: Decimal,
    source: &'static str,
    reason: &'static str,
) -> Result<Decimal, DirectionalFeatureError> {
    if value <= Decimal::ZERO {
        return Err(external_snapshot_error(source, reason));
    }
    Ok(value)
}

fn external_snapshot_error(source: &'static str, reason: &'static str) -> DirectionalFeatureError {
    DirectionalFeatureError::ExternalFeatureUnavailable { source, reason }
}

fn complete_directional_model_candidate(
    candidate: &mut Option<DirectionalModelCandidateLease<'_>>,
    seal_opportunity: bool,
) -> Result<()> {
    let Some(candidate) = candidate.take() else {
        return Ok(());
    };
    candidate.complete(seal_opportunity)
}

struct EntryAdmissionEvaluation {
    disposition: AdmissionDisposition,
    evidence: serde_json::Value,
}

pub struct BtcProcessRunner {
    repository: BtcRepository,
    store: Store,
    execution_venue: Arc<dyn ExecutionVenue>,
    execution_lifecycle: Arc<dyn BtcExecutionLifecycle>,
    book_registry: Arc<tokio::sync::RwLock<BookRegistry>>,
    config: BtcProcessConfig,
    max_directional_feature_age_ms: Option<i64>,
    initialized: OnceCell<()>,
    loss_regime_admission: Mutex<Option<LossRegimeAdmissionRuntime>>,
    high_water_mark_entry_submission: Mutex<()>,
    execution_reconcile_started_at: Mutex<Option<Instant>>,
    directional_model_runtime: StdMutex<DirectionalModelProcessRuntime>,
    unified_session:
        StdMutex<Option<Box<dyn super::unified_model_runtime::adapters::FeatureSession>>>,
    primary_persistence_state: Option<Arc<RwLock<RealtimeState>>>,
}

pub type BtcPaperProcessRunner = BtcProcessRunner;

impl BtcProcessRunner {
    /// Backward-compatible paper constructor. New process wiring should use
    /// `new_with_execution` so venue choice occurs only at the composition boundary.
    pub fn new(
        repository: BtcRepository,
        store: Store,
        paper_venue: PaperVenue,
        config: BtcProcessConfig,
    ) -> Result<Self> {
        let paper_venue = Arc::new(paper_venue);
        let book_registry = paper_venue.registry();
        let execution_venue: Arc<dyn ExecutionVenue> = paper_venue.clone();
        let execution_lifecycle: Arc<dyn BtcExecutionLifecycle> =
            Arc::new(PaperExecutionLifecycle::new(paper_venue));
        Self::new_with_execution(
            repository,
            store,
            execution_venue,
            book_registry,
            execution_lifecycle,
            config,
        )
    }

    pub fn new_with_execution(
        repository: BtcRepository,
        store: Store,
        execution_venue: Arc<dyn ExecutionVenue>,
        book_registry: Arc<tokio::sync::RwLock<BookRegistry>>,
        execution_lifecycle: Arc<dyn BtcExecutionLifecycle>,
        config: BtcProcessConfig,
    ) -> Result<Self> {
        if config.run_key.trim().is_empty() || config.config_hash.trim().is_empty() {
            anyhow::bail!("BTC run identity and config hash must not be empty");
        }
        if !config.frozen_process_config.is_object() {
            anyhow::bail!("BTC run frozen process config must be a JSON object");
        }
        ensure!(
            !execution_lifecycle.reconcile_interval().is_zero(),
            "BTC execution reconcile interval must be positive"
        );
        if let Some(frozen_mode) = config
            .frozen_process_config
            .pointer("/execution/mode")
            .and_then(serde_json::Value::as_str)
        {
            ensure!(
                frozen_mode == execution_lifecycle.mode().as_str(),
                "BTC execution lifecycle mode does not match the frozen process config"
            );
        }
        config.strategy.validate()?;
        let max_directional_feature_age_ms =
            config.strategy.effective_max_directional_feature_age_ms()?;
        if config.strategy.attribution().is_none() {
            anyhow::bail!("BTC execution run strategy attribution is invalid");
        }
        validate_directional_model_entry_policy(
            &config.strategy,
            config.directional_model_entry_policy,
            config.execution_enabled,
        )?;
        if let Some(entry_admission) = config.entry_admission.as_ref() {
            entry_admission.validate()?;
        }
        let mut preview_keys = HashSet::new();
        for preview in &config.paper_stress_previews {
            preview.validate()?;
            if !preview_keys.insert(preview.scenario_key.as_str()) {
                anyhow::bail!(
                    "duplicate BTC paper stress-preview scenario {}",
                    preview.scenario_key
                );
            }
        }
        let unified_session = directional_model_selection(&config.strategy)
            .map(|selection| runtime_model(&selection))
            .transpose()?
            .and_then(|model| model.unified_adapter().map(|a| a.new_session()));
        umr_telemetry::register(
            config.process_id,
            config.run_id,
            &config.config_hash,
            execution_lifecycle.mode().as_str(),
            directional_model_selection(&config.strategy).as_ref(),
        );
        Ok(Self {
            repository,
            store,
            execution_venue,
            execution_lifecycle,
            book_registry,
            loss_regime_admission: Mutex::new(
                config
                    .entry_admission
                    .as_ref()
                    .map(|_| LossRegimeAdmissionRuntime::default()),
            ),
            high_water_mark_entry_submission: Mutex::new(()),
            config,
            max_directional_feature_age_ms,
            initialized: OnceCell::new(),
            execution_reconcile_started_at: Mutex::new(None),
            directional_model_runtime: StdMutex::new(DirectionalModelProcessRuntime::default()),
            unified_session: StdMutex::new(unified_session),
            primary_persistence_state: None,
        })
    }

    pub fn with_primary_persistence_state(mut self, state: Arc<RwLock<RealtimeState>>) -> Self {
        self.primary_persistence_state = Some(state);
        self
    }

    async fn primary_persistence_available(&self) -> bool {
        let Some(state) = self.primary_persistence_state.as_ref() else {
            return !self.config.execution_enabled;
        };
        state.read().await.primary_persistence_available()
    }

    pub fn execution_mode(&self) -> BtcExecutionMode {
        self.execution_lifecycle.mode()
    }

    fn claim_directional_model_candidate(
        &self,
        market_id: &str,
        feature_as_of: DateTime<Utc>,
    ) -> Result<Option<DirectionalModelCandidateLease<'_>>> {
        let claim = self
            .directional_model_runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("BTC directional model process lock was poisoned"))?
            .claim(market_id, feature_as_of);
        Ok(claim.map(
            |(feature_as_of, requires_rehydration)| DirectionalModelCandidateLease {
                runtime: &self.directional_model_runtime,
                market_id: market_id.to_string(),
                feature_as_of,
                requires_rehydration,
                completed: false,
            },
        ))
    }

    async fn insert_process_strategy_decision(
        &self,
        market_id: &str,
        decision: &BtcDecision,
        entry_admission_evidence: Option<&serde_json::Value>,
        order_plan_id: Option<Uuid>,
        status: &str,
    ) -> Result<bool> {
        let inserted = self
            .repository
            .insert_strategy_decision(
                self.config.process_id,
                self.config.run_id,
                &self.config.config_hash,
                market_id,
                &self.config.strategy.strategy_version,
                decision,
                entry_admission_evidence,
                order_plan_id,
                self.execution_mode(),
                status,
            )
            .await?;
        if inserted {
            umr_telemetry::event(self.config.process_id, "decisions", status);
        }
        Ok(inserted)
    }

    /// Claims the immutable execution-run identity before feeds begin.
    pub async fn initialize(&self) -> Result<()> {
        self.initialize_with_existing_identity(false).await
    }

    /// Reattaches an already-running immutable execution run.
    pub async fn resume(&self) -> Result<()> {
        self.initialize_with_existing_identity(true).await
    }

    async fn initialize_with_existing_identity(&self, resume: bool) -> Result<()> {
        self.initialized
            .get_or_try_init(|| async {
                if resume {
                    self.repository
                        .verify_run_manifest(
                            self.config.process_id,
                            self.config.run_id,
                            &self.config.run_key,
                            &self.config.config_hash,
                            &self.config.frozen_process_config,
                        )
                        .await?;
                    self.execution_lifecycle
                        .resume_run(&self.repository, self.config.process_id, self.config.run_id)
                        .await?;
                } else {
                    self.repository
                        .claim_run_manifest(
                            self.config.process_id,
                            self.config.run_id,
                            &self.config.run_key,
                            &self.config.config_hash,
                            &self.config.frozen_process_config,
                        )
                        .await?;
                }
                self.initialize_entry_admission(resume).await?;
                if let Err(error) = self.force_refresh_settlement_and_reconcile().await {
                    warn!(
                        process_id = %self.config.process_id,
                        run_id = %self.config.run_id,
                        error = %error,
                        "initial reconciliation deferred; runtime remains active for automatic recovery"
                    );
                }
                Ok::<(), anyhow::Error>(())
            })
            .await?;
        Ok(())
    }

    async fn initialize_entry_admission(&self, resume: bool) -> Result<()> {
        let Some(entry_admission) = self.config.entry_admission.as_ref() else {
            return Ok(());
        };
        let as_of = Utc::now();
        let floor = &entry_admission.loss_regime_confidence_floor;
        let candidates = self
            .repository
            .load_resolved_loss_regime_candidates(
                self.config.process_id,
                &self.config.config_hash,
                as_of,
            )
            .await?;
        let state = LossRegimeConfidenceFloorState::from_candidates(floor, &candidates);
        let initialized_state = state.clone();
        *self.loss_regime_admission.lock().await = Some(LossRegimeAdmissionRuntime {
            state,
            evaluated_market_id: None,
        });
        self.record_entry_admission_event(
            "btc_entry_admission_initialized",
            "loss-regime confidence-floor admission initialized",
            serde_json::json!({
                "resume": resume,
                "config": entry_admission,
                "entry_admission_config_hash": floor.config_hash()?,
                "state": initialized_state,
            }),
        )
        .await;
        Ok(())
    }

    async fn evaluate_loss_regime_admission(
        &self,
        decision: &BtcDecision,
        as_of: DateTime<Utc>,
    ) -> Result<Option<LossRegimeConfidenceFloorEvaluation>> {
        let Some(entry_admission) = self.config.entry_admission.as_ref() else {
            return Ok(None);
        };
        let floor = &entry_admission.loss_regime_confidence_floor;
        let selected_probability = selected_conservative_probability(decision)
            .context("approved BTC intent is missing its selected conservative probability")?;
        let market_id = decision
            .approved_intent
            .as_ref()
            .map(|intent| intent.market_id.as_str())
            .context("approved BTC intent is missing its market identity")?;
        {
            let admission = self.loss_regime_admission.lock().await;
            let runtime = admission
                .as_ref()
                .context("configured loss-regime admission state was not initialized")?;
            if runtime.evaluated_market_id.as_deref() == Some(market_id) {
                return Ok(Some(runtime.state.evaluate(floor, selected_probability)?));
            }
        }
        let candidates = self
            .repository
            .load_resolved_loss_regime_candidates(
                self.config.process_id,
                &self.config.config_hash,
                as_of,
            )
            .await?;
        let mut rebuilt_state = LossRegimeConfidenceFloorState::default();
        let mut last_transition = None;
        for candidate in candidates {
            if let Some(transition) = rebuilt_state.apply_candidate(floor, &candidate) {
                last_transition = Some((transition, candidate, rebuilt_state.clone()));
            }
        }
        let (evaluation, transition_event) = {
            let mut admission = self.loss_regime_admission.lock().await;
            let runtime = admission
                .as_mut()
                .context("configured loss-regime admission state was not initialized")?;
            let active_changed = runtime.state.active != rebuilt_state.active;
            runtime.state = rebuilt_state;
            runtime.evaluated_market_id = Some(market_id.to_string());
            (
                runtime.state.evaluate(floor, selected_probability)?,
                active_changed.then_some(last_transition).flatten(),
            )
        };
        if let Some((transition, candidate, state)) = transition_event {
            let (event_type, message) = match transition {
                LossRegimeConfidenceFloorTransition::Activated => (
                    "btc_loss_regime_confidence_floor_activated",
                    "loss-regime confidence floor activated",
                ),
                LossRegimeConfidenceFloorTransition::Released => (
                    "btc_loss_regime_confidence_floor_released",
                    "loss-regime confidence floor released",
                ),
            };
            self.record_entry_admission_event(
                event_type,
                message,
                serde_json::json!({
                    "entry_admission_config_hash": floor.config_hash()?,
                    "candidate": candidate,
                    "state": state,
                }),
            )
            .await;
        }
        Ok(Some(evaluation))
    }

    async fn evaluate_entry_admission(
        &self,
        decision: &BtcDecision,
        as_of: DateTime<Utc>,
        fee_rate: Decimal,
    ) -> Result<Option<EntryAdmissionEvaluation>> {
        let Some(entry_admission) = self.config.entry_admission.as_ref() else {
            return Ok(None);
        };
        let loss_regime = self
            .evaluate_loss_regime_admission(decision, as_of)
            .await?
            .context("configured loss-regime admission did not produce an evaluation")?;
        let high_water_mark = match entry_admission.daily_realized_pnl_high_water_mark.as_ref() {
            Some(config) => {
                let intent = decision
                    .approved_intent
                    .as_ref()
                    .context("approved BTC intent is missing for high-water-mark admission")?;
                let proposed =
                    ProposedEntryExposure::new(intent.size, intent.limit_price, fee_rate)?;
                let state = self
                    .repository
                    .load_daily_realized_pnl_high_water_mark_state(self.config.process_id, as_of)
                    .await?;
                Some(state.evaluate(config, &proposed)?)
            }
            None => None,
        };
        Ok(Some(combine_entry_admission_evaluations(
            self.config.process_id,
            loss_regime,
            high_water_mark,
        )?))
    }

    async fn record_entry_admission_event(
        &self,
        event_type: &str,
        message: &str,
        metadata: serde_json::Value,
    ) {
        if let Err(error) = self
            .store
            .record_trading_process_event(
                self.config.process_id,
                "info",
                event_type,
                Some(message),
                metadata,
            )
            .await
        {
            warn!(
                error = %error,
                process_id = %self.config.process_id,
                event_type,
                "failed to persist BTC entry-admission event"
            );
        }
    }

    pub async fn shutdown(&self) -> Result<()> {
        umr_telemetry::enabled(self.config.process_id, false);
        if let Err(error) = self.force_refresh_settlement_and_reconcile().await {
            warn!(
                process_id = %self.config.process_id,
                run_id = %self.config.run_id,
                error = %error,
                "shutdown reconciliation failed without changing durable process authorization"
            );
        }
        Ok(())
    }

    pub fn shared_book_registry(&self) -> Arc<tokio::sync::RwLock<BookRegistry>> {
        self.book_registry.clone()
    }

    async fn refresh_settlement_and_reconcile(&self) -> Result<()> {
        self.execution_lifecycle
            .reconcile_run(
                &self.repository,
                self.config.process_id,
                self.config.run_id,
                &self.config.config_hash,
            )
            .await
    }

    async fn force_refresh_settlement_and_reconcile(&self) -> Result<()> {
        let mut last_started_at = self.execution_reconcile_started_at.lock().await;
        let previous = *last_started_at;
        *last_started_at = Some(Instant::now());
        let result = self.refresh_settlement_and_reconcile().await;
        if result.is_err() {
            *last_started_at = previous;
        }
        result
    }

    async fn refresh_settlement_and_reconcile_if_due(&self) -> Result<()> {
        let Ok(mut last_started_at) = self.execution_reconcile_started_at.try_lock() else {
            return Ok(());
        };
        let now = Instant::now();
        if !execution_reconcile_due(
            *last_started_at,
            now,
            self.execution_lifecycle.reconcile_interval(),
        ) {
            return Ok(());
        }
        let previous = *last_started_at;
        *last_started_at = Some(now);
        let result = self.refresh_settlement_and_reconcile().await;
        if result.is_err() {
            *last_started_at = previous;
        }
        result
    }

    async fn observe(&self, observation: StrategyObservation) -> Result<()> {
        if !self.primary_persistence_available().await {
            umr_telemetry::readiness(self.config.process_id, false);
            umr_telemetry::event(self.config.process_id, "skipped", "persistence_unavailable");
            return Ok(());
        }
        self.initialize().await?;
        let high_water_mark_configured = self
            .config
            .entry_admission
            .as_ref()
            .and_then(|admission| admission.daily_realized_pnl_high_water_mark.as_ref())
            .is_some();
        let _high_water_mark_entry_guard = if high_water_mark_configured {
            Some(self.high_water_mark_entry_submission.lock().await)
        } else {
            None
        };
        if let Err(error) = self.refresh_settlement_and_reconcile_if_due().await {
            warn!(
                process_id = %self.config.process_id,
                run_id = %self.config.run_id,
                error = %error,
                "periodic reconciliation deferred; strategy runtime remains active"
            );
        }

        let Some(market) = observation.state.current_market.as_ref() else {
            umr_telemetry::event(self.config.process_id, "skipped", "no_market");
            return Ok(());
        };
        // Keep durable point-in-time inputs on the same immutable observation boundary used by
        // runtime readiness. Initialization, reconciliation and admission must not move the
        // feature timestamp forward while feeds continue advancing.
        let observed_at = observation.readiness.checked_at;
        let scoped_readiness =
            process_runtime_readiness(&self.config.strategy, &observation.readiness);
        for reason in &scoped_readiness.reasons {
            umr_telemetry::event(
                self.config.process_id,
                "readiness_blocks",
                reason.split(':').next().unwrap_or("unknown"),
            );
        }
        if !scoped_readiness.ready {
            umr_telemetry::readiness(self.config.process_id, false);
        }
        let directional_selection = directional_model_selection(&self.config.strategy);
        let mut directional_candidate = None;
        let mut directional_opening_reference = None;
        let feature_started = Instant::now();
        let (snapshot_identity_at, directional_model, mut directional_model_feature_error) =
            if let Some(selection) = directional_selection.as_ref() {
                let Some(latest_feature_as_of) = observation
                    .state
                    .binance_one_second_window
                    .completed()
                    .back()
                    .map(|candle| candle.close_timestamp)
                else {
                    umr_telemetry::readiness(self.config.process_id, false);
                    umr_telemetry::event(
                        self.config.process_id,
                        "skipped",
                        "binance_history_unavailable",
                    );
                    return Ok(());
                };
                let model = runtime_model(selection)
                    .context("failed to resolve configured BTC directional model")?;
                let policy = model.prediction_policy();
                let Some((feature_as_of, candidate_seconds_elapsed)) =
                    latest_directional_model_candidate(
                        policy,
                        market.window_start,
                        latest_feature_as_of,
                    )
                else {
                    umr_telemetry::event(self.config.process_id, "skipped", "outside_schedule");
                    return Ok(());
                };
                if feature_as_of > observed_at {
                    umr_telemetry::event(self.config.process_id, "skipped", "future_candidate");
                    return Ok(());
                }
                if !policy.accepts(candidate_seconds_elapsed) {
                    return Ok(());
                }
                let Some(candidate) =
                    self.claim_directional_model_candidate(&market.market_id, feature_as_of)?
                else {
                    umr_telemetry::event(
                        self.config.process_id,
                        "skipped",
                        "candidate_already_claimed",
                    );
                    return Ok(());
                };
                umr_telemetry::event(self.config.process_id, "opportunities", "scheduled");
                umr_telemetry::eligible_market(self.config.process_id, &market.market_id);
                if let Some(session) = self
                    .unified_session
                    .lock()
                    .map_err(|_| anyhow::anyhow!("UMR session lock poisoned"))?
                    .as_mut()
                {
                    session.observe_slot(&market.market_id, candidate_seconds_elapsed);
                }
                let feature_as_of = candidate.feature_as_of();
                let candidate_seconds_elapsed = (feature_as_of - market.window_start).num_seconds();
                if !policy.accepts(candidate_seconds_elapsed) {
                    return Ok(());
                }
                if candidate.requires_rehydration() {
                    let has_entry = self
                        .repository
                        .process_has_entry(self.config.process_id, &market.market_id)
                        .await?;
                    candidate.mark_rehydrated()?;
                    if has_entry {
                        candidate.complete(true)?;
                        return Ok(());
                    }
                }
                if model.unified_adapter().is_none()
                    && directional_schema_requires_opening_boundary(model.feature_schema_version())
                {
                    directional_opening_reference = self
                        .repository
                        .load_market_opening_reference(
                            market,
                            feature_as_of,
                            chrono::Duration::milliseconds(
                                self.config.strategy.max_chainlink_open_delay_ms,
                            ),
                        )
                        .await?;
                }
                if model.is_asymmetric_value() || model.is_payoff_aware() {
                    directional_candidate = Some(candidate);
                    (feature_as_of, None, None)
                } else {
                    let opening_boundary = directional_opening_reference
                        .as_ref()
                        .map(|tick| tick.price);
                    let features = match directional_external_decision_snapshot(
                        &observation.state.directional_external,
                        feature_as_of,
                        model.feature_schema_version(),
                    )
                    .and_then(|external| {
                        let external_inputs = external.as_ref().map(|snapshot| snapshot.inputs());
                        build_directional_features_for_schema_with_external(
                            &observation.state.binance_one_second_window,
                            market.window_start,
                            feature_as_of,
                            model.feature_schema_version(),
                            opening_boundary,
                            external_inputs.as_ref(),
                        )
                    }) {
                        Ok(features) => {
                            directional_candidate = Some(candidate);
                            (
                                feature_as_of,
                                Some(build_directional_model_feature_snapshot(
                                    selection, market, features,
                                )?),
                                None,
                            )
                        }
                        Err(error) => {
                            directional_candidate = Some(candidate);
                            (
                                feature_as_of,
                                None,
                                Some(directional_feature_error_metadata(&error)),
                            )
                        }
                    };
                    features
                }
            } else {
                (observed_at, None, None)
            };
        let mut inputs = if directional_selection.is_some() {
            directional_model_execution_inputs_from_runtime(
                &observation.state,
                self.config.process_id,
                market,
                observed_at,
                snapshot_identity_at,
                chrono::Duration::milliseconds(self.config.strategy.max_reference_age_ms),
                chrono::Duration::milliseconds(self.config.strategy.max_book_age_ms),
            )?
        } else {
            let clob_connection_id = observation_clob_connection_id(market, &observation.readiness);
            self.repository
                .load_point_in_time_inputs(
                    market,
                    observed_at,
                    chrono::Duration::milliseconds(
                        self.config.strategy.max_chainlink_open_delay_ms,
                    ),
                    chrono::Duration::milliseconds(self.config.strategy.max_reference_age_ms),
                    chrono::Duration::milliseconds(self.config.strategy.max_book_age_ms),
                    clob_connection_id,
                )
                .await?
        };
        if directional_opening_reference.is_some() {
            inputs.chainlink_open = directional_opening_reference;
        }
        let mut snapshot = build_snapshot(
            self.config.process_id,
            market,
            observed_at,
            &inputs,
            self.config.strategy.target_size,
            &self.config.strategy.feature_schema_version,
            snapshot_identity_at,
            directional_model,
        );
        if let Some(selection) = directional_selection.as_ref() {
            let model = runtime_model(selection)
                .context("failed to resolve configured BTC model for feature construction")?;
            if model.is_asymmetric_value() && snapshot.directional_model.is_none() {
                match build_asymmetric_value_feature_snapshot(
                    selection,
                    &model,
                    &observation.state,
                    market,
                    snapshot_identity_at,
                    &snapshot,
                ) {
                    Ok(features) => snapshot.directional_model = Some(features),
                    Err(error) => {
                        directional_model_feature_error = Some(serde_json::json!({
                            "code": "asymmetric_value_features_unavailable",
                            "detail": error.to_string(),
                        }));
                    }
                }
            }
            if model.is_payoff_aware() && snapshot.directional_model.is_none() {
                let payoff_features = (|| -> Result<BtcDirectionalModelFeatureSnapshot> {
                    if let Some(adapter) = model.unified_adapter() {
                        let context = super::unified_model_runtime::adapters::FeatureContext {
                            state: &observation.state,
                            names: model.feature_names(),
                            binding: self
                                .config
                                .strategy
                                .unified_model
                                .as_ref()
                                .context("UMR binding unavailable")?,
                            market_id: &market.market_id,
                            window_start: market.window_start,
                            feature_as_of: snapshot_identity_at,
                            up: inputs.up_book.as_ref().context("UMR UP book unavailable")?,
                            down: inputs
                                .down_book
                                .as_ref()
                                .context("UMR DOWN book unavailable")?,
                            fee_rate: inputs
                                .fee_rate
                                .and_then(|v| v.to_f64())
                                .context("UMR fee unavailable")?,
                        };
                        let values = self
                            .unified_session
                            .lock()
                            .map_err(|_| anyhow::anyhow!("UMR session lock poisoned"))?
                            .as_mut()
                            .context("UMR session unavailable")?
                            .prepare(adapter, &context)?;
                        let input_sha256 = directional_model_input_sha256(
                            selection,
                            model.feature_schema_version(),
                            &market.market_id,
                            market.window_start,
                            snapshot_identity_at,
                            (snapshot_identity_at - market.window_start).num_seconds(),
                            &values,
                        )?;
                        return Ok(BtcDirectionalModelFeatureSnapshot {
                            model_key: selection.model_key.clone(),
                            model_artifact_sha256: selection.artifact_sha256.clone(),
                            feature_schema_version: model.feature_schema_version().into(),
                            feature_schema_sha256: selection.feature_schema_sha256.clone(),
                            feature_as_of: snapshot_identity_at,
                            seconds_elapsed: (snapshot_identity_at - market.window_start)
                                .num_seconds(),
                            feature_values: values,
                            input_sha256,
                        });
                    }
                    let opening_boundary = inputs
                        .chainlink_open
                        .as_ref()
                        .context("payoff-aware model opening boundary is unavailable")?
                        .price;
                    let external = directional_external_decision_snapshot(
                        &observation.state.directional_external,
                        snapshot_identity_at,
                        model.feature_schema_version(),
                    )?
                    .context("payoff-aware external feature snapshot is unavailable")?;
                    let values = build_payoff_aware_feature_values(
                        &observation.state.binance_one_second_window,
                        market.window_start,
                        snapshot_identity_at,
                        opening_boundary,
                        &external.inputs(),
                        inputs
                            .up_book
                            .as_ref()
                            .context("payoff-aware UP book is unavailable")?,
                        inputs
                            .down_book
                            .as_ref()
                            .context("payoff-aware DOWN book is unavailable")?,
                        inputs
                            .fee_rate
                            .and_then(|value| value.to_f64())
                            .context("payoff-aware fee is unavailable")?,
                        model.feature_names(),
                    )?;
                    let input_sha256 = directional_model_input_sha256(
                        selection,
                        model.feature_schema_version(),
                        &market.market_id,
                        market.window_start,
                        snapshot_identity_at,
                        (snapshot_identity_at - market.window_start).num_seconds(),
                        &values,
                    )?;
                    Ok(BtcDirectionalModelFeatureSnapshot {
                        model_key: selection.model_key.clone(),
                        model_artifact_sha256: selection.artifact_sha256.clone(),
                        feature_schema_version: model.feature_schema_version().to_string(),
                        feature_schema_sha256: selection.feature_schema_sha256.clone(),
                        feature_as_of: snapshot_identity_at,
                        seconds_elapsed: (snapshot_identity_at - market.window_start).num_seconds(),
                        feature_values: values,
                        input_sha256,
                    })
                })();
                match payoff_features {
                    Ok(features) => snapshot.directional_model = Some(features),
                    Err(error) => {
                        directional_model_feature_error = Some(serde_json::json!({
                            "code": "payoff_aware_features_unavailable",
                            "detail": error.to_string(),
                        }))
                    }
                }
            }
        }
        umr_telemetry::duration(
            self.config.process_id,
            "features",
            feature_started.elapsed().as_secs_f64(),
        );
        umr_telemetry::event(
            self.config.process_id,
            "feature_builds",
            if directional_model_feature_error.is_some() {
                "error"
            } else {
                "success"
            },
        );
        if let Some(error) = directional_model_feature_error.as_ref() {
            umr_telemetry::failure(
                self.config.process_id,
                "features",
                error
                    .get("detail")
                    .and_then(|v| v.as_str())
                    .unwrap_or("missing_source_data"),
            );
        }
        if let Some(features) = snapshot.directional_model.as_ref() {
            umr_telemetry::gauge(
                self.config.process_id,
                "feature_age_seconds",
                (observed_at - features.feature_as_of).num_milliseconds() as f64 / 1000.0,
            );
            umr_telemetry::gauge(
                self.config.process_id,
                "missing_feature_fraction",
                features
                    .feature_values
                    .iter()
                    .filter(|v| !v.is_finite())
                    .count() as f64
                    / features.feature_values.len().max(1) as f64,
            );
        }
        let decision_started = Instant::now();
        let mut decision = DeterministicBtcStrategy::evaluate_with_directional_model_entry_policy(
            &self.config.strategy,
            &snapshot,
            self.config.directional_model_entry_policy,
        );
        umr_telemetry::duration(
            self.config.process_id,
            "strategy",
            decision_started.elapsed().as_secs_f64(),
        );
        enforce_runtime_readiness(&mut decision, &observation.readiness, &self.config.strategy);
        umr_telemetry::readiness(
            self.config.process_id,
            scoped_readiness.ready
                && directional_model_feature_error.is_none()
                && decision.prediction.is_some(),
        );
        if let Some(reason) = decision.reject_reason.as_ref() {
            umr_telemetry::event(
                self.config.process_id,
                "strategy_rejections",
                reason.as_str(),
            );
        }
        if directional_model_feature_error.is_some() {
            decision.action = BtcDecisionAction::NoTrade;
            decision.reject_reason = Some(BtcRejectReason::DirectionalFeaturesUnavailable);
            decision.fair_value = None;
            decision.up_edge = None;
            decision.down_edge = None;
            decision.approved_intent = None;
            decision.prediction = None;
        }
        let existing_process_entry = if decision.approved_intent.is_some() {
            self.repository
                .process_has_entry(self.config.process_id, &snapshot.market_id)
                .await?
        } else {
            false
        };
        if existing_process_entry {
            decision.action = BtcDecisionAction::NoTrade;
            decision.reject_reason = Some(BtcRejectReason::ExistingProcessEntry);
            decision.approved_intent = None;
        }
        let feature_hash = sha256_json(&snapshot)?;
        let quality_flags =
            snapshot_quality_flags(&snapshot, &observation.readiness, &self.config.strategy);
        let readiness_status = if quality_flags.is_empty() {
            "ready"
        } else {
            "not_ready"
        };
        let feature_inserted = self
            .repository
            .insert_feature_snapshot(
                &snapshot,
                decision.fair_value.as_ref(),
                &feature_hash,
                readiness_status,
                &serde_json::json!({
                    "runtime_readiness": observation.readiness,
                    "chainlink_quality_ok": snapshot.chainlink_quality_ok,
                    "binance_quality_ok": snapshot.binance_quality_ok,
                    "directional_model_feature_error": directional_model_feature_error,
                    "quality_flags": quality_flags,
                }),
            )
            .await?;
        if !feature_inserted && directional_selection.is_none() {
            return Ok(());
        }

        let Some(intent) = decision.approved_intent.clone() else {
            self.insert_process_strategy_decision(
                &snapshot.market_id,
                &decision,
                None,
                None,
                "rejected",
            )
            .await?;
            complete_directional_model_candidate(
                &mut directional_candidate,
                existing_process_entry,
            )?;
            return Ok(());
        };

        if !self.config.execution_enabled {
            self.insert_process_strategy_decision(
                &snapshot.market_id,
                &decision,
                None,
                None,
                "shadow_only",
            )
            .await?;
            complete_directional_model_candidate(&mut directional_candidate, true)?;
            return Ok(());
        }

        let entry_admission = self
            .evaluate_entry_admission(
                &decision,
                observed_at,
                snapshot.fee_rate.unwrap_or_default(),
            )
            .await?;
        let entry_admission_evidence = entry_admission
            .as_ref()
            .map(|evaluation| &evaluation.evidence);
        if entry_admission
            .as_ref()
            .is_some_and(|evaluation| evaluation.disposition == AdmissionDisposition::Defer)
        {
            self.insert_process_strategy_decision(
                &snapshot.market_id,
                &decision,
                entry_admission_evidence,
                None,
                "admission_blocked",
            )
            .await?;
            complete_directional_model_candidate(&mut directional_candidate, false)?;
            return Ok(());
        }

        let plan_id = Uuid::new_v5(
            &Uuid::NAMESPACE_URL,
            format!("btc-paper-plan:{}", intent.intent_id).as_bytes(),
        );
        let fee_rate = snapshot.fee_rate.unwrap_or_default();
        let order_metadata = btc_entry_order_metadata(
            &self.config.strategy,
            &intent,
            decision.prediction.as_ref(),
            decision.decision_id,
            self.config.process_id,
            self.config.run_id,
            fee_rate,
        )?;
        let client_order_id = Uuid::new_v5(
            &Uuid::NAMESPACE_URL,
            format!("btc-paper-order:{}", intent.intent_id).as_bytes(),
        );
        let mut request = OrderRequest {
            client_order_id,
            process_id: Some(self.config.process_id),
            market_id: intent.market_id.clone(),
            token_id: intent.token_id.clone(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: intent.limit_price,
            size: intent.size,
            metadata: order_metadata,
        };
        let reference_execution_guard = BtcReferenceExecutionGuard::from_snapshot(
            &snapshot,
            &decision,
            &intent,
            &request,
            &feature_hash,
            fee_rate,
            BtcExecutionFreshnessBounds {
                max_reference_age_ms: self.config.strategy.max_reference_age_ms,
                max_directional_feature_age_ms: self.max_directional_feature_age_ms,
            },
        )?;
        reference_execution_guard.insert_into_metadata(&mut request.metadata)?;
        if !self.primary_persistence_available().await {
            return Ok(());
        }
        self.insert_process_strategy_decision(
            &snapshot.market_id,
            &decision,
            entry_admission_evidence,
            Some(plan_id),
            // This write-ahead state deliberately is not treated as an entry. The
            // adjacent persistence gate below may still defer venue mutation.
            "execution_pending",
        )
        .await?;
        let preview_futures = self
            .config
            .paper_stress_previews
            .iter()
            .map(|preview| self.execution_lifecycle.preview_order(&request, preview));
        let primary_request = request.clone();
        if !self.primary_persistence_available().await {
            return Ok(());
        }
        // Once this durable reservation succeeds, execution must proceed. A
        // crash after authorization cannot permit a different entry for the
        // same process and market on resume.
        self.repository
            .authorize_pending_strategy_decision(
                self.config.process_id,
                self.config.run_id,
                decision.decision_id,
                decision.evaluated_at,
            )
            .await?;
        complete_directional_model_candidate(&mut directional_candidate, true)?;
        let (report, preview_results) = tokio::join!(
            execute_order_plan(
                self.execution_venue.as_ref(),
                OrderPlan {
                    plan_id,
                    orders: vec![primary_request],
                },
            ),
            futures_util::future::join_all(preview_futures),
        );
        let execution_mode = self.execution_lifecycle.mode();
        let report = report.with_context(|| {
            format!("BTC {} OrderPlan execution failed", execution_mode.as_str())
        })?;
        let mut stress_previews = Vec::with_capacity(preview_results.len());
        for (config, result) in self
            .config
            .paper_stress_previews
            .iter()
            .zip(preview_results)
        {
            match result {
                Ok(Some(result)) => stress_previews.push(serde_json::json!({
                    "status": "observed",
                    "result": result,
                })),
                Ok(None) => stress_previews.push(serde_json::json!({
                    "status": "unavailable",
                    "scenario_key": config.scenario_key,
                    "execution_mode": execution_mode.as_str(),
                })),
                Err(error) => {
                    warn!(
                        error = %error,
                        scenario_key = %config.scenario_key,
                        decision_id = %decision.decision_id,
                        "non-mutating BTC paper stress preview failed"
                    );
                    stress_previews.push(serde_json::json!({
                        "status": "error",
                        "scenario_key": config.scenario_key,
                        "error": error.to_string(),
                    }));
                }
            }
        }
        self.store.persist_order_plan_report(&report).await?;
        for fill in &report.fills {
            if let (Some(price), Some(size), Some(fee)) =
                (fill.price.to_f64(), fill.size.to_f64(), fill.fee.to_f64())
            {
                let quoted = if fill.token_id == snapshot.up_book.token_id {
                    snapshot.up_book.executable_ask_vwap
                } else {
                    snapshot.down_book.executable_ask_vwap
                }
                .and_then(|v| v.to_f64())
                .unwrap_or(price);
                umr_telemetry::fill(
                    self.config.process_id,
                    price,
                    size,
                    fee,
                    (observed_at - market.window_start).num_milliseconds() as f64 / 1000.0,
                    quoted,
                );
            }
        }
        let primary_order = report
            .orders
            .first()
            .context("BTC OrderPlan report omitted its primary order")?;
        let primary_state = primary_order.state;
        let filled = primary_state == OrderState::Filled;
        let execution_status = decision_execution_status(primary_state);
        umr_telemetry::event(self.config.process_id, "execution", execution_status);
        if filled {
            umr_telemetry::gauge(
                self.config.process_id,
                "last_entry_seconds",
                (observed_at - market.window_start).num_milliseconds() as f64 / 1000.0,
            );
            if let Some(cost) = snapshot
                .up_book
                .executable_ask_vwap
                .filter(|_| intent.outcome == BtcOutcome::Up)
                .or(snapshot
                    .down_book
                    .executable_ask_vwap
                    .filter(|_| intent.outcome == BtcOutcome::Down))
                .and_then(|v| v.to_f64())
            {
                umr_telemetry::gauge(self.config.process_id, "last_entry_quote_usd", cost);
            }
        }
        let mut execution_reject_reason = primary_order
            .request
            .metadata
            .get("reject_reason")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        if execution_reject_reason.is_none()
            && matches!(execution_mode, BtcExecutionMode::Live)
            && primary_state == OrderState::Rejected
        {
            execution_reject_reason = self
                .store
                .order_venue_reject_reason(
                    self.config.process_id,
                    primary_order.request.client_order_id,
                )
                .await?;
        }
        let execution_metadata = execution_result_metadata(
            execution_mode,
            &report,
            &self.config.config_hash,
            stress_previews,
        );
        self.repository
            .update_strategy_decision_execution(
                self.config.process_id,
                self.config.run_id,
                decision.decision_id,
                decision.evaluated_at,
                execution_status,
                execution_reject_reason.as_deref(),
                execution_metadata,
            )
            .await?;
        if filled {
            if let Err(error) = self.force_refresh_settlement_and_reconcile().await {
                warn!(
                    process_id = %self.config.process_id,
                    run_id = %self.config.run_id,
                    error = %error,
                    "post-fill reconciliation deferred; strategy runtime remains active"
                );
            }
        }
        Ok(())
    }
}

fn decision_execution_status(state: OrderState) -> &'static str {
    match state {
        OrderState::Filled => "filled",
        OrderState::Rejected | OrderState::Cancelled | OrderState::Expired => "rejected",
        OrderState::Created
        | OrderState::Submitted
        | OrderState::Acknowledged
        | OrderState::PartiallyFilled
        | OrderState::CancelRequested
        | OrderState::Unknown => "submitted",
    }
}

fn execution_result_metadata(
    mode: BtcExecutionMode,
    report: &OrderPlanReport,
    config_hash: &str,
    stress_previews: Vec<serde_json::Value>,
) -> serde_json::Value {
    let order_plan = serde_json::json!({
        "plan_id": report.plan_id,
        "orders": &report.orders,
        "fills": &report.fills,
        "reconciliation": &report.reconciliation,
    });
    let previews = serde_json::json!({
        "telemetry_only": true,
        "influenced_primary_execution": false,
        "primary_config_hash": config_hash,
        "scenarios": stress_previews,
    });
    match mode {
        // Preserve the durable paper evidence shape for existing processes and resumable runs.
        BtcExecutionMode::Paper => serde_json::json!({
            "paper_order_plan": order_plan,
            "paper_stress_previews": previews,
        }),
        BtcExecutionMode::Live => serde_json::json!({
            "execution_mode": mode.as_str(),
            "order_plan": order_plan,
            "stress_previews": previews,
        }),
    }
}

fn combine_entry_admission_evaluations(
    process_id: Uuid,
    loss_regime: LossRegimeConfidenceFloorEvaluation,
    high_water_mark: Option<DailyRealizedPnlHighWaterMarkEvaluation>,
) -> Result<EntryAdmissionEvaluation> {
    let loss_deferred = loss_regime.disposition == AdmissionDisposition::Defer;
    let high_water_mark_deferred = high_water_mark
        .as_ref()
        .is_some_and(|evaluation| evaluation.disposition == AdmissionDisposition::Defer);
    let disposition = if loss_deferred || high_water_mark_deferred {
        AdmissionDisposition::Defer
    } else {
        AdmissionDisposition::Allow
    };

    let evidence = match high_water_mark {
        None => serde_json::to_value(loss_regime)?,
        Some(high_water_mark) => {
            let mut blocking_policies = Vec::new();
            if loss_deferred {
                blocking_policies.push("loss_regime_confidence_floor");
            }
            if high_water_mark_deferred {
                blocking_policies.push("daily_realized_pnl_high_water_mark");
            }
            serde_json::json!({
                "evidence_version": "btc_entry_admission_v2",
                "process_id": process_id,
                "disposition": disposition,
                "blocking_policies": blocking_policies,
                "loss_regime_confidence_floor": loss_regime,
                "daily_realized_pnl_high_water_mark": high_water_mark,
            })
        }
    };
    Ok(EntryAdmissionEvaluation {
        disposition,
        evidence,
    })
}

fn btc_entry_order_metadata(
    strategy: &BtcStrategyConfig,
    intent: &super::strategy::ApprovedIntent,
    prediction: Option<&BtcStrategyPrediction>,
    decision_id: Uuid,
    process_id: Uuid,
    run_id: Uuid,
    fee_rate: Decimal,
) -> Result<serde_json::Value> {
    ensure!(
        intent.process_id == process_id,
        "BTC order metadata process ownership does not match its execution scope"
    );
    let attribution = strategy
        .attribution()
        .context("BTC execution run strategy attribution became invalid")?;
    let mut metadata = serde_json::json!({
        "execution_intent": "entry",
        "strategy": attribution.family,
        "strategy_version": intent.strategy_version,
        "feature_schema_version": intent.feature_schema_version,
        "feature_snapshot_id": intent.feature_snapshot_id,
        "decision_id": decision_id,
        "process_id": process_id,
        "run_id": run_id,
        "outcome": intent.outcome,
        "expected_net_edge": intent.expected_net_edge,
        PAPER_DYNAMIC_FEE_RATE_METADATA_KEY: fee_rate,
    });
    if let (Some(profile_id), Some(profile_sha256)) =
        (attribution.profile_id, attribution.profile_sha256)
    {
        let metadata = metadata
            .as_object_mut()
            .context("BTC order strategy metadata must be an object")?;
        metadata.insert("profile_id".to_string(), profile_id.into());
        metadata.insert("profile_sha256".to_string(), profile_sha256.into());
    }
    if let Some(prediction) = prediction {
        let BtcStrategyPrediction::DirectionalPrediction {
            outcome,
            entry_policy,
            ..
        } = prediction
        else {
            anyhow::bail!("BTC entry order cannot carry a no-prediction result");
        };
        ensure!(
            matches!(attribution.family, BTC_DIRECTIONAL_MODEL_STRATEGY_FAMILY)
                && *outcome == intent.outcome,
            "BTC directional prediction attribution does not match its entry intent"
        );
        if *entry_policy == BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction {
            ensure!(
                attribution.family == BTC_DIRECTIONAL_MODEL_STRATEGY_FAMILY
                    && intent.strategy_version == BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION,
                "BTC directional prediction execution policy requires directional-model attribution"
            );
        }
        let metadata = metadata
            .as_object_mut()
            .context("BTC order strategy metadata must be an object")?;
        metadata.insert("prediction".to_string(), serde_json::to_value(prediction)?);
    }
    Ok(metadata)
}

fn selected_conservative_probability(decision: &BtcDecision) -> Option<Decimal> {
    match decision.action {
        BtcDecisionAction::BuyUp => decision
            .up_edge
            .as_ref()
            .map(|edge| edge.conservative_probability),
        BtcDecisionAction::BuyDown => decision
            .down_edge
            .as_ref()
            .map(|edge| edge.conservative_probability),
        BtcDecisionAction::NoTrade => None,
    }
}

fn execution_reconcile_due(
    last_started_at: Option<Instant>,
    now: Instant,
    interval: TokioDuration,
) -> bool {
    last_started_at
        .is_none_or(|last_started_at| now.saturating_duration_since(last_started_at) >= interval)
}

fn snapshot_quality_flags(
    snapshot: &BtcFeatureSnapshot,
    readiness: &super::types::Readiness,
    config: &BtcStrategyConfig,
) -> Vec<String> {
    let mut flags = Vec::new();
    let directional_model = runtime_model_configured(config);
    if !runtime_readiness_satisfied(config, readiness) {
        flags.push("runtime_not_ready".to_string());
        flags.extend(
            readiness
                .reasons
                .iter()
                .filter(|reason| runtime_readiness_reason_required(config, reason))
                .map(|reason| format!("runtime:{reason}")),
        );
    }
    if directional_model && snapshot.directional_model.is_none() {
        flags.push("missing_directional_model_features".to_string());
    }
    if !directional_model {
        match snapshot.chainlink_age_ms {
            None => flags.push("missing_chainlink".to_string()),
            Some(age) if age < 0 => flags.push("future_chainlink_receipt".to_string()),
            Some(age) if age > config.max_reference_age_ms => {
                flags.push("stale_chainlink".to_string())
            }
            Some(_) => {}
        }
    }
    match snapshot.binance_age_ms {
        None => flags.push("missing_binance".to_string()),
        Some(age) if age < 0 => flags.push("future_binance_receipt".to_string()),
        Some(age) if age > config.max_reference_age_ms => flags.push("stale_binance".to_string()),
        Some(_) => {}
    }
    if !directional_model
        && snapshot
            .source_skew_ms
            .is_none_or(|skew| skew > config.max_source_skew_ms)
    {
        flags.push("source_timestamp_skew".to_string());
    }
    for (name, book) in [("up", &snapshot.up_book), ("down", &snapshot.down_book)] {
        if book.connection_id.is_none() {
            flags.push(format!("missing_{name}_book_epoch"));
        }
        if book.integrity_status != FeedIntegrityStatus::Ok {
            flags.push(format!("{name}_book_integrity"));
        }
        if book.best_bid.is_none() || book.best_ask.is_none() {
            flags.push(format!("missing_{name}_two_sided_book"));
        }
        if matches!((book.best_bid, book.best_ask), (Some(bid), Some(ask)) if bid >= ask) {
            flags.push(format!("crossed_{name}_book"));
        }
        match book.age_ms {
            None => flags.push(format!("missing_{name}_book_age")),
            Some(age) if age < 0 => flags.push(format!("future_{name}_book_receipt")),
            Some(age) if age > config.max_book_age_ms => {
                flags.push(format!("stale_{name}_book"));
            }
            Some(_) => {}
        }
    }
    if snapshot.lineage.up_book_connection_id.is_some()
        && snapshot.lineage.down_book_connection_id.is_some()
        && snapshot.lineage.up_book_connection_id != snapshot.lineage.down_book_connection_id
    {
        flags.push("mixed_book_connection_epochs".to_string());
    }
    let expected_up_epoch = readiness
        .books
        .iter()
        .find(|book| book.token_id == snapshot.up_book.token_id)
        .map(|book| book.connection_id);
    let expected_down_epoch = readiness
        .books
        .iter()
        .find(|book| book.token_id == snapshot.down_book.token_id)
        .map(|book| book.connection_id);
    match (expected_up_epoch, expected_down_epoch) {
        (Some(up), Some(down)) if up == down => {
            if snapshot.lineage.up_book_connection_id != Some(up)
                || snapshot.lineage.down_book_connection_id != Some(up)
            {
                flags.push("book_connection_epoch_mismatch".to_string());
            }
        }
        _ => flags.push("runtime_book_epoch_unavailable".to_string()),
    }
    if snapshot.fees_enabled {
        match (snapshot.fee_rate, snapshot.fee_rate_observed_at) {
            (None, _) => flags.push("missing_fee_rate".to_string()),
            (Some(rate), _) if rate < Decimal::ZERO || rate > config.max_fee_rate => {
                flags.push("invalid_fee_rate".to_string())
            }
            (_, None) => flags.push("missing_fee_observed_at".to_string()),
            (_, Some(observed_at))
                if (snapshot.observed_at - observed_at).num_milliseconds()
                    > config.max_fee_age_ms =>
            {
                flags.push("stale_fee_rate".to_string())
            }
            _ => {}
        }
    }
    flags.sort();
    flags.dedup();
    flags
}

fn enforce_runtime_readiness(
    decision: &mut BtcDecision,
    readiness: &super::types::Readiness,
    config: &BtcStrategyConfig,
) {
    if !runtime_readiness_satisfied(config, readiness) {
        decision.action = BtcDecisionAction::NoTrade;
        decision.reject_reason = Some(BtcRejectReason::RuntimeNotReady);
        decision.approved_intent = None;
    }
}

fn runtime_readiness_satisfied(
    config: &BtcStrategyConfig,
    readiness: &super::types::Readiness,
) -> bool {
    if readiness.ready {
        return true;
    }
    if readiness.reasons.is_empty() {
        return false;
    }
    readiness
        .reasons
        .iter()
        .all(|reason| !runtime_readiness_reason_required(config, reason))
}

pub fn process_runtime_readiness(
    config: &BtcStrategyConfig,
    readiness: &super::types::Readiness,
) -> super::types::Readiness {
    let mut process_readiness = readiness.clone();
    let had_reasons = !process_readiness.reasons.is_empty();
    process_readiness
        .reasons
        .retain(|reason| runtime_readiness_reason_required(config, reason));
    process_readiness.ready =
        readiness.ready || (had_reasons && process_readiness.reasons.is_empty());
    process_readiness
}

fn runtime_readiness_reason_required(config: &BtcStrategyConfig, reason: &str) -> bool {
    if !runtime_model_configured(config) {
        return true;
    }
    if chainlink_reference_readiness_reason(reason) {
        return false;
    }
    if direct_binance_reference_readiness_reason(reason) {
        return config.required_model_feeds.is_empty()
            || config
                .required_model_feeds
                .iter()
                .any(|requirement| requirement.feed == BtcModelFeedId::BinanceBtcusdtOneSecondV1);
    }
    true
}

fn chainlink_reference_readiness_reason(reason: &str) -> bool {
    let Some((kind, source)) = reason.split_once(':') else {
        return false;
    };
    matches!(
        kind,
        "missing_reference" | "future_reference" | "stale_reference"
    ) && source == ReferencePriceSource::RtdsChainlink.as_str()
}

fn direct_binance_reference_readiness_reason(reason: &str) -> bool {
    let Some((kind, source)) = reason.split_once(':') else {
        return false;
    };
    matches!(
        kind,
        "missing_reference" | "future_reference" | "stale_reference"
    ) && source == ReferencePriceSource::DirectBinance.as_str()
}

fn observation_clob_connection_id(
    market: &BtcIntervalMarket,
    readiness: &super::types::Readiness,
) -> Option<Uuid> {
    let up = readiness
        .books
        .iter()
        .find(|book| book.token_id == market.up_token_id)?;
    let down = readiness
        .books
        .iter()
        .find(|book| book.token_id == market.down_token_id)?;
    (up.connection_id == down.connection_id).then_some(up.connection_id)
}

fn directional_model_selection(config: &BtcStrategyConfig) -> Option<RuntimeModelSelection> {
    match config.decision_strategy.as_ref()? {
        BtcDecisionStrategyConfig::BtcDirectionalModel {
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        }
        | BtcDecisionStrategyConfig::BtcAsymmetricValueModel {
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        } => Some(RuntimeModelSelection {
            model_key: model_key.clone(),
            artifact_sha256: artifact_sha256.clone(),
            feature_schema_sha256: feature_schema_sha256.clone(),
        }),
    }
}

fn directional_model_configured(config: &BtcStrategyConfig) -> bool {
    matches!(
        config.decision_strategy.as_ref(),
        Some(BtcDecisionStrategyConfig::BtcDirectionalModel { .. })
    )
}

fn runtime_model_configured(config: &BtcStrategyConfig) -> bool {
    matches!(
        config.decision_strategy.as_ref(),
        Some(BtcDecisionStrategyConfig::BtcDirectionalModel { .. })
            | Some(BtcDecisionStrategyConfig::BtcAsymmetricValueModel { .. })
    )
}

fn validate_directional_model_entry_policy(
    strategy: &BtcStrategyConfig,
    entry_policy: BtcDirectionalModelEntryPolicy,
    execution_enabled: bool,
) -> Result<()> {
    if entry_policy == BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction
        && (!directional_model_configured(strategy) || !execution_enabled)
    {
        anyhow::bail!(
            "BTC directional prediction execution policy requires an executing directional-model process"
        );
    }
    Ok(())
}

fn latest_directional_model_candidate(
    policy: RuntimePredictionPolicy,
    window_start: DateTime<Utc>,
    latest_feature_as_of: DateTime<Utc>,
) -> Option<(DateTime<Utc>, i64)> {
    let latest_seconds_elapsed = (latest_feature_as_of - window_start).num_seconds();
    if latest_seconds_elapsed < policy.minimum_seconds_after_open || policy.cadence_seconds <= 0 {
        return None;
    }
    let (schedule_start, cadence) = match (policy.early_end_second, policy.early_cadence_seconds) {
        (Some(end), Some(early_cadence)) if latest_seconds_elapsed <= end => {
            (policy.minimum_seconds_after_open, early_cadence)
        }
        (Some(end), Some(_)) => (end + 1, policy.cadence_seconds),
        _ => (policy.minimum_seconds_after_open, policy.cadence_seconds),
    };
    if latest_seconds_elapsed < schedule_start || cadence <= 0 {
        return None;
    }
    let seconds_elapsed = (schedule_start
        + ((latest_seconds_elapsed - schedule_start) / cadence) * cadence)
        .min(policy.maximum_seconds_after_open);
    policy.accepts(seconds_elapsed).then(|| {
        (
            window_start + chrono::Duration::seconds(seconds_elapsed),
            seconds_elapsed,
        )
    })
}

fn build_directional_model_feature_snapshot(
    selection: &RuntimeModelSelection,
    market: &BtcIntervalMarket,
    features: DirectionalFeatureVector,
) -> Result<BtcDirectionalModelFeatureSnapshot> {
    let input_sha256 = directional_model_input_sha256(
        selection,
        features.schema_version(),
        &market.market_id,
        market.window_start,
        features.feature_as_of,
        i64::from(features.seconds_elapsed),
        &features.values,
    )?;
    Ok(BtcDirectionalModelFeatureSnapshot {
        model_key: selection.model_key.clone(),
        model_artifact_sha256: selection.artifact_sha256.clone(),
        feature_schema_version: features.schema_version().to_string(),
        feature_schema_sha256: selection.feature_schema_sha256.clone(),
        feature_as_of: features.feature_as_of,
        seconds_elapsed: i64::from(features.seconds_elapsed),
        feature_values: features.values,
        input_sha256,
    })
}

#[async_trait]
impl BtcStrategyRunner for BtcProcessRunner {
    async fn reconcile_if_due(&self) -> Result<()> {
        if self.initialized.get().is_none() {
            return Ok(());
        }
        if let Err(error) = self.refresh_settlement_and_reconcile_if_due().await {
            warn!(
                process_id = %self.config.process_id,
                run_id = %self.config.run_id,
                error = %error,
                "timer-driven reconciliation deferred; strategy runtime remains active"
            );
        }
        Ok(())
    }

    async fn on_observation(&self, observation: StrategyObservation) -> Result<()> {
        umr_telemetry::observation(
            self.config.process_id,
            observation
                .state
                .current_market
                .as_ref()
                .map(|m| m.market_id.as_str()),
        );
        let mut guard = umr_telemetry::ObservationGuard::new(self.config.process_id);
        let result = self.observe(observation).await;
        if result.is_ok() {
            guard.complete();
        }
        result
    }

    async fn shutdown(&self) -> Result<()> {
        BtcProcessRunner::shutdown(self).await
    }
}

fn build_snapshot(
    process_id: Uuid,
    market: &BtcIntervalMarket,
    observed_at: DateTime<Utc>,
    inputs: &BtcPointInTimeInputs,
    target_size: Decimal,
    feature_schema_version: &str,
    snapshot_identity_at: DateTime<Utc>,
    directional_model: Option<BtcDirectionalModelFeatureSnapshot>,
) -> BtcFeatureSnapshot {
    let chainlink_open = inputs.chainlink_open.as_ref();
    let chainlink = inputs.chainlink_current.as_ref();
    let binance = inputs.binance_history.last();
    let snapshot_id = Uuid::new_v5(
        &Uuid::NAMESPACE_URL,
        format!(
            "btc-feature:{}:{}:{}",
            process_id,
            market.market_id,
            snapshot_identity_at.timestamp_micros()
        )
        .as_bytes(),
    );
    let chainlink_gap_bps = chainlink_open.zip(chainlink).and_then(|(open, current)| {
        ratio_return(current.price, open.price).map(|value| value * Decimal::from(10_000))
    });
    let latest_binance = binance.map(|tick| tick.price);
    let binance_return_1s = return_over(&inputs.binance_history, observed_at, 1);
    let binance_return_5s = return_over(&inputs.binance_history, observed_at, 5);
    let binance_return_30s = return_over(&inputs.binance_history, observed_at, 30);
    let realized_volatility = realized_volatility(&inputs.binance_history);
    let basis = normalized_basis_bps(inputs);
    let up_book = book_features(
        BtcOutcome::Up,
        &market.up_token_id,
        inputs.up_book.as_ref(),
        observed_at,
        target_size,
    );
    let down_book = book_features(
        BtcOutcome::Down,
        &market.down_token_id,
        inputs.down_book.as_ref(),
        observed_at,
        target_size,
    );
    let chainlink_age_ms =
        chainlink.map(|tick| (observed_at - tick.received_at).num_milliseconds());
    let binance_age_ms = binance.map(|tick| (observed_at - tick.received_at).num_milliseconds());
    let source_skew_ms = chainlink.zip(binance).map(|(left, right)| {
        (left.source_timestamp - right.source_timestamp)
            .num_milliseconds()
            .abs()
    });
    BtcFeatureSnapshot {
        snapshot_id,
        process_id,
        observed_at,
        feature_schema_version: feature_schema_version.to_string(),
        market_id: market.market_id.clone(),
        event_slug: market.event_slug.clone(),
        window_start: market.window_start,
        window_end: market.window_end,
        market_active: market.active,
        market_closed: market.closed,
        accepting_orders: market.accepting_orders,
        tick_size: market.tick_size,
        minimum_order_size: market.minimum_order_size,
        resolution_source: market.resolution_source.clone(),
        chainlink_open_price: chainlink_open.map(|tick| tick.price),
        chainlink_price: chainlink.map(|tick| tick.price),
        binance_price: latest_binance,
        chainlink_gap_bps,
        chainlink_return_5s: None,
        chainlink_return_15s: None,
        chainlink_return_30s: None,
        chainlink_path_efficiency_30s: None,
        chainlink_path_tick_count_30s: None,
        chainlink_realized_volatility_5s: None,
        chainlink_realized_volatility_30s: None,
        binance_return_1s,
        binance_return_5s,
        binance_return_30s,
        realized_volatility,
        binance_chainlink_basis_bps: basis,
        chainlink_age_ms,
        binance_age_ms,
        source_skew_ms,
        chainlink_quality_ok: chainlink_age_ms.map(|age| age >= 0).unwrap_or(false),
        binance_quality_ok: binance_age_ms.map(|age| age >= 0).unwrap_or(false),
        up_book,
        down_book,
        fees_enabled: market.fees_enabled,
        fee_rate: inputs.fee_rate,
        fee_rate_observed_at: inputs.fee_observed_at,
        directional_model,
        lineage: BtcFeatureLineage {
            lineage_version: BTC_FEATURE_LINEAGE_VERSION.to_string(),
            chainlink_open_tick_id: chainlink_open.map(|tick| tick.tick_id),
            chainlink_tick_id: chainlink.map(|tick| tick.tick_id),
            binance_tick_id: binance.map(|tick| tick.tick_id),
            up_book_checkpoint_id: inputs.up_book.as_ref().map(|book| book.checkpoint_id),
            down_book_checkpoint_id: inputs.down_book.as_ref().map(|book| book.checkpoint_id),
            chainlink_open_source_timestamp: chainlink_open.map(|tick| tick.source_timestamp),
            chainlink_open_received_at: chainlink_open.map(|tick| tick.received_at),
            chainlink_source_timestamp: chainlink.map(|tick| tick.source_timestamp),
            chainlink_received_at: chainlink.map(|tick| tick.received_at),
            chainlink_anchor_15s_tick_id: None,
            chainlink_anchor_15s_source_timestamp: None,
            chainlink_anchor_15s_received_at: None,
            chainlink_anchor_15s_ingest_sequence: None,
            chainlink_anchor_15s_effective_lookback_ms: None,
            chainlink_anchor_30s_tick_id: None,
            chainlink_anchor_30s_source_timestamp: None,
            chainlink_anchor_30s_received_at: None,
            chainlink_anchor_30s_ingest_sequence: None,
            chainlink_anchor_30s_effective_lookback_ms: None,
            binance_source_timestamp: binance.map(|tick| tick.source_timestamp),
            binance_received_at: binance.map(|tick| tick.received_at),
            chainlink_ingest_sequence: chainlink.map(|tick| tick.ingest_sequence),
            chainlink_open_ingest_sequence: chainlink_open.map(|tick| tick.ingest_sequence),
            binance_ingest_sequence: binance.map(|tick| tick.ingest_sequence),
            up_book_ingest_sequence: inputs.up_book.as_ref().map(|book| book.ingest_sequence),
            down_book_ingest_sequence: inputs.down_book.as_ref().map(|book| book.ingest_sequence),
            up_book_connection_id: inputs.up_book.as_ref().map(|book| book.connection_id),
            down_book_connection_id: inputs.down_book.as_ref().map(|book| book.connection_id),
            chainlink_history: input_window_lineage(&inputs.chainlink_history),
            binance_history: input_window_lineage(&inputs.binance_history),
        },
    }
}

fn directional_model_execution_inputs_from_runtime(
    state: &RealtimeState,
    process_id: Uuid,
    market: &BtcIntervalMarket,
    as_of: DateTime<Utc>,
    feature_as_of: DateTime<Utc>,
    max_reference_age: chrono::Duration,
    max_book_age: chrono::Duration,
) -> Result<BtcPointInTimeInputs> {
    ensure!(
        max_reference_age > chrono::Duration::zero() && max_book_age > chrono::Duration::zero(),
        "directional-model runtime input freshness bounds must be positive"
    );
    let reference_fresh_since = as_of
        .checked_sub_signed(max_reference_age)
        .context("directional-model reference freshness bound is outside the timestamp range")?;

    let binance = state
        .reference_prices
        .get(&ReferencePriceSource::DirectBinance)
        .filter(|tick| {
            tick.source_timestamp >= reference_fresh_since
                && tick.source_timestamp <= as_of
                && tick.received_at >= reference_fresh_since
                && tick.received_at <= as_of
        })
        .cloned();
    // Use the immutable observation, never a registry advanced by concurrent feeds.
    let fresh_book = |token_id: &str| {
        let selected = state.unified_book_history.observation_book(
            &market.market_id,
            token_id,
            state.books.get(token_id),
            as_of,
            max_book_age,
        );
        match selected {
            Ok(book) => Some(book.clone()),
            Err(error) => {
                umr_telemetry::event(process_id, "book_input_failures", error.reason);
                tracing::warn!(event="umr_book_input_unavailable", %process_id,
                    market_id=%market.market_id, token_id, observation_at=%as_of, %feature_as_of,
                    expected_epoch=?state.books.get(token_id).map(|v| v.connection_id),
                    reason=error.reason, source_timestamp=?error.source_timestamp,
                    received_at=?error.received_at,
                    "Observation book unavailable");
                None
            }
        }
    };
    let fee_rate = market
        .fees_enabled
        .then(|| decimal_json_field(&market.fee_schedule, &["rate"]))
        .flatten();

    Ok(BtcPointInTimeInputs {
        chainlink_open: None,
        chainlink_current: None,
        chainlink_history: Vec::new(),
        binance_history: binance.into_iter().collect(),
        up_book: fresh_book(&market.up_token_id),
        down_book: fresh_book(&market.down_token_id),
        fee_rate,
        // The fee schedule is an immutable term of this five-minute market contract. Its
        // effective boundary is therefore the market open, rather than the arrival time of an
        // unrelated realtime update.
        fee_observed_at: fee_rate.map(|_| market.window_start),
    })
}

fn decimal_json_field(value: &serde_json::Value, keys: &[&str]) -> Option<Decimal> {
    keys.iter().find_map(|key| match value.get(*key)? {
        serde_json::Value::String(value) => Decimal::from_str(value).ok(),
        serde_json::Value::Number(value) => Decimal::from_str(&value.to_string()).ok(),
        _ => None,
    })
}

fn book_features(
    outcome: BtcOutcome,
    token_id: &str,
    checkpoint: Option<&OrderbookCheckpoint>,
    observed_at: DateTime<Utc>,
    target_size: Decimal,
) -> BtcOutcomeBookFeatures {
    let Some(checkpoint) = checkpoint else {
        return BtcOutcomeBookFeatures {
            outcome,
            token_id: token_id.to_string(),
            best_bid: None,
            best_ask: None,
            executable_ask_vwap: None,
            marketable_limit_price: None,
            quoted_size: Decimal::ZERO,
            bid_depth: Decimal::ZERO,
            ask_depth: Decimal::ZERO,
            imbalance: None,
            source_timestamp: None,
            received_at: None,
            age_ms: None,
            integrity_status: FeedIntegrityStatus::PreSnapshot,
            connection_id: None,
        };
    };
    let bid_depth: Decimal = checkpoint.bids.iter().map(|level| level.size).sum();
    let ask_depth: Decimal = checkpoint.asks.iter().map(|level| level.size).sum();
    let total_depth = bid_depth + ask_depth;
    let imbalance = (total_depth > Decimal::ZERO).then(|| (bid_depth - ask_depth) / total_depth);
    let mut asks = checkpoint.asks.clone();
    asks.sort_by(|left, right| left.price.cmp(&right.price));
    let mut remaining = target_size;
    let mut quoted = Decimal::ZERO;
    let mut notional = Decimal::ZERO;
    let mut limit = None;
    for level in asks {
        if remaining <= Decimal::ZERO {
            break;
        }
        let take = remaining.min(level.size);
        quoted += take;
        notional += take * level.price;
        remaining -= take;
        if take > Decimal::ZERO {
            limit = Some(level.price);
        }
    }
    BtcOutcomeBookFeatures {
        outcome,
        token_id: token_id.to_string(),
        best_bid: checkpoint.best_bid,
        best_ask: checkpoint.best_ask,
        executable_ask_vwap: (quoted > Decimal::ZERO).then(|| notional / quoted),
        marketable_limit_price: limit,
        quoted_size: quoted,
        bid_depth,
        ask_depth,
        imbalance,
        source_timestamp: Some(checkpoint.source_timestamp),
        received_at: Some(checkpoint.received_at),
        age_ms: Some(
            (observed_at - checkpoint.source_timestamp)
                .max(observed_at - checkpoint.received_at)
                .num_milliseconds(),
        ),
        integrity_status: checkpoint.integrity_status,
        connection_id: Some(checkpoint.connection_id),
    }
}

fn input_window_lineage(history: &[ReferencePriceTick]) -> BtcInputWindowLineage {
    let Some(first) = history.first() else {
        return BtcInputWindowLineage::default();
    };
    let last = history.last().expect("non-empty history has a final tick");
    let mut hasher = Sha256::new();
    for tick in history {
        hasher.update(tick.tick_id.as_bytes());
        hasher.update(tick.source_timestamp.timestamp_micros().to_le_bytes());
        hasher.update(tick.received_at.timestamp_micros().to_le_bytes());
        hasher.update(tick.ingest_sequence.to_le_bytes());
        let normalized_price = tick.price.normalize().to_string();
        hasher.update((normalized_price.len() as u64).to_le_bytes());
        hasher.update(normalized_price.as_bytes());
    }
    BtcInputWindowLineage {
        first_tick_id: Some(first.tick_id),
        last_tick_id: Some(last.tick_id),
        first_source_timestamp: Some(first.source_timestamp),
        last_source_timestamp: Some(last.source_timestamp),
        tick_count: history.len(),
        max_received_at: history.iter().map(|tick| tick.received_at).max(),
        input_sha256: Some(format!("{:x}", hasher.finalize())),
    }
}

fn return_over(
    history: &[ReferencePriceTick],
    observed_at: DateTime<Utc>,
    seconds: i64,
) -> Option<Decimal> {
    let latest = history.last()?;
    let target = observed_at - chrono::Duration::seconds(seconds);
    let prior = history
        .iter()
        .rev()
        .find(|tick| tick.source_timestamp <= target)?;
    ratio_return(latest.price, prior.price)
}

fn ratio_return(current: Decimal, prior: Decimal) -> Option<Decimal> {
    (current > Decimal::ZERO && prior > Decimal::ZERO).then(|| current / prior - Decimal::ONE)
}

fn realized_volatility(history: &[ReferencePriceTick]) -> Option<Decimal> {
    if history.len() < 3 {
        return None;
    }
    realized_volatility_from_path(history)
}

fn realized_volatility_from_path(history: &[ReferencePriceTick]) -> Option<Decimal> {
    let elapsed = (history.last()?.source_timestamp - history.first()?.source_timestamp)
        .num_milliseconds() as f64
        / 1_000.0;
    if elapsed <= 0.0 {
        return None;
    }
    let sum_squared = history
        .windows(2)
        .filter_map(|pair| {
            let prior = pair[0].price.to_f64()?;
            let current = pair[1].price.to_f64()?;
            (prior > 0.0 && current > 0.0).then(|| (current / prior).ln().powi(2))
        })
        .sum::<f64>();
    Decimal::from_f64_retain((sum_squared / elapsed).sqrt())
}

fn normalized_basis_bps(inputs: &BtcPointInTimeInputs) -> Option<Decimal> {
    let current_binance = inputs.binance_history.last()?;
    let current_chainlink = inputs.chainlink_current.as_ref()?;
    let raw = ratio_return(current_binance.price, current_chainlink.price)? * Decimal::from(10_000);
    let mut history = inputs
        .chainlink_history
        .iter()
        .filter_map(|chainlink| {
            let binance = inputs
                .binance_history
                .iter()
                .rev()
                .find(|tick| tick.source_timestamp <= chainlink.source_timestamp)?;
            ratio_return(binance.price, chainlink.price).map(|value| value * Decimal::from(10_000))
        })
        .collect::<Vec<_>>();
    if history.is_empty() {
        return Some(raw);
    }
    history.sort();
    let neutral = history[history.len() / 2];
    Some(raw - neutral)
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use rust_decimal_macros::dec;

    #[test]
    fn directional_feature_failures_persist_a_stable_machine_code() {
        let window_start = Utc.timestamp_opt(1_783_902_600, 0).unwrap();
        let metadata = directional_feature_error_metadata(
            &DirectionalFeatureError::IncompletePrewindowHistory { window_start },
        );

        assert_eq!(metadata["code"], "incomplete_prewindow_history");
        assert!(metadata["detail"]
            .as_str()
            .is_some_and(|detail| detail.contains("is incomplete")));
    }

    use crate::btc::{
        admission::{
            DailyRealizedPnlCredit, DailyRealizedPnlHighWaterMarkConfig,
            DailyRealizedPnlHighWaterMarkState, LossRegimeConfidenceFloorConfig,
            LossRegimeConfidenceFloorState, ProposedEntryExposure, UnsettledEntryExposure,
            DAILY_REALIZED_PNL_HIGH_WATER_MARK_SCHEMA_VERSION,
            LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION,
        },
        directional_external_runtime::{
            BinanceOpenInterestPoint, ChainlinkRefPricePoint, DirectionalExternalState,
            PolygonOraclePoint,
        },
        directional_features::{
            BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_SCHEMA_VERSION,
            BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION,
        },
        directional_model::{
            BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION, BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256,
            BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256, BTC_DIRECTIONAL_MODEL_V1_KEY,
        },
        strategy::{
            ApprovedIntent, BtcDecisionStrategyConfig, BtcStrategyConfig,
            BTC_FEATURE_SCHEMA_VERSION,
        },
        types::{BookReadiness, Readiness},
        BtcModelFeedRequirement,
    };

    fn external_midpoint_state(feature_as_of: DateTime<Utc>) -> DirectionalExternalState {
        let latest_close =
            DateTime::from_timestamp(feature_as_of.timestamp().div_euclid(60) * 60, 0).unwrap();
        let earliest_open = latest_close - chrono::Duration::minutes(61);
        let mut state = DirectionalExternalState::default();
        for minute in 0..61_i64 {
            let open = earliest_open + chrono::Duration::minutes(minute);
            let base = Decimal::new(6_000_000 + minute * 100, 2);
            Arc::make_mut(&mut state.rtds).insert_fixture(crate::btc::rtds_repository::RtdsPoint {
                source_timestamp: open + chrono::Duration::seconds(5),
                available_at: open + chrono::Duration::seconds(6),
                price: base,
            });
            Arc::make_mut(&mut state.rtds).insert_fixture(crate::btc::rtds_repository::RtdsPoint {
                source_timestamp: open + chrono::Duration::seconds(50),
                available_at: open + chrono::Duration::seconds(51),
                price: base + dec!(1.25),
            });
        }
        state
    }

    #[test]
    fn runtime_chainlink_candles_use_only_available_closed_midpoints() {
        let feature_as_of = Utc.with_ymd_and_hms(2026, 8, 1, 12, 34, 30).unwrap();
        let mut state = external_midpoint_state(feature_as_of);
        let latest_close =
            DateTime::from_timestamp(feature_as_of.timestamp().div_euclid(60) * 60, 0).unwrap();
        Arc::make_mut(&mut state.rtds).insert_fixture(crate::btc::rtds_repository::RtdsPoint {
            source_timestamp: latest_close - chrono::Duration::seconds(5),
            available_at: feature_as_of + chrono::Duration::milliseconds(1),
            price: dec!(999999),
        });

        let candles = state.rtds().closed_candles(feature_as_of).unwrap();

        assert_eq!(candles.len(), 61);
        assert_eq!(candles.last().unwrap().close_timestamp, latest_close);
        assert_eq!(candles.last().unwrap().high_price, dec!(60061.25));
        assert_eq!(candles.last().unwrap().close_price, dec!(60061.25));
        assert!(candles
            .iter()
            .all(|candle| candle.available_at <= feature_as_of));
    }

    #[test]
    fn runtime_chainlink_candles_fail_closed_on_one_missing_minute() {
        let feature_as_of = Utc.with_ymd_and_hms(2026, 8, 1, 12, 34, 30).unwrap();
        let mut state = external_midpoint_state(feature_as_of);
        let latest_close =
            DateTime::from_timestamp(feature_as_of.timestamp().div_euclid(60) * 60, 0).unwrap();
        let missing_open = latest_close - chrono::Duration::minutes(20);
        let retained = state
            .rtds()
            .points_as_of(feature_as_of)
            .filter(|point| {
                point.source_timestamp < missing_open
                    || point.source_timestamp >= missing_open + chrono::Duration::minutes(1)
            })
            .cloned()
            .collect::<Vec<_>>();
        state.rtds = Arc::default();
        for point in retained {
            Arc::make_mut(&mut state.rtds).insert_fixture(point);
        }

        let error = state.rtds().closed_candles(feature_as_of).unwrap_err();

        assert_eq!(error.code(), "external_feature_unavailable");
        assert!(error.to_string().contains("61 contiguous closed"));
    }

    #[test]
    fn external_decision_snapshot_preserves_source_values_and_archive_absence() {
        let feature_as_of = Utc.with_ymd_and_hms(2026, 8, 1, 12, 34, 30).unwrap();
        let mut state = external_midpoint_state(feature_as_of);
        let oracle_at = feature_as_of - chrono::Duration::seconds(10);
        state.oracle.push_back(PolygonOraclePoint {
            phase_id: 7,
            round_id: 42,
            source_timestamp: oracle_at,
            block_timestamp: oracle_at,
            available_at: oracle_at + chrono::Duration::milliseconds(250),
            price: dec!(63123.12345678),
        });
        let ref_at = feature_as_of - chrono::Duration::seconds(1);
        let valid_from = ref_at - chrono::Duration::milliseconds(250);
        state.refprice.push_back(ChainlinkRefPricePoint {
            source_timestamp: ref_at,
            valid_from_timestamp: valid_from,
            available_at: ref_at + chrono::Duration::milliseconds(500),
            price: dec!(63124.123456789012345678),
            bid: dec!(63123.9),
            ask: dec!(63124.3),
        });
        let oi_at = feature_as_of - chrono::Duration::minutes(5);
        state.open_interest.push_back(BinanceOpenInterestPoint {
            source_timestamp: oi_at,
            available_at: oi_at + chrono::Duration::seconds(1),
            sum_open_interest: dec!(123456.12345678),
            sum_open_interest_value: dec!(7654321.12345678),
        });

        let snapshot = directional_external_decision_snapshot(
            &state,
            feature_as_of,
            BTC_DIRECTIONAL_BOUNDARY_ORACLE_CHAINLINK_REFPRICE_CANDLE_OI_FEATURE_SCHEMA_VERSION,
        )
        .unwrap()
        .unwrap();

        assert_eq!(snapshot.oracle_rounds[0].block_number, None);
        assert_eq!(snapshot.oracle_rounds[0].log_index, None);
        assert_eq!(snapshot.oracle_rounds[0].price, dec!(63123.12345678));
        assert_eq!(
            snapshot.refprice_reports[0].valid_from_timestamp,
            valid_from
        );
        assert_eq!(
            snapshot.refprice_reports[0].price,
            dec!(63124.123456789012345678)
        );
        assert_eq!(
            snapshot.open_interest[0].sum_open_interest,
            dec!(123456.12345678)
        );
    }

    #[test]
    fn existing_directional_schema_does_not_require_external_state() {
        let feature_as_of = Utc.with_ymd_and_hms(2026, 8, 1, 12, 34, 30).unwrap();
        assert!(directional_external_decision_snapshot(
            &DirectionalExternalState::default(),
            feature_as_of,
            BTC_DIRECTIONAL_FEATURE_SCHEMA_VERSION,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn directional_model_candidate_recovers_latest_eligible_cadence() {
        let window_start = Utc.with_ymd_and_hms(2026, 7, 27, 12, 0, 0).unwrap();
        let policy = RuntimePredictionPolicy {
            minimum_seconds_after_open: 60,
            maximum_seconds_after_open: 240,
            cadence_seconds: 5,
            early_end_second: None,
            early_cadence_seconds: None,
            late_start_second: None,
        };

        assert_eq!(
            latest_directional_model_candidate(
                policy,
                window_start,
                window_start + chrono::Duration::seconds(67),
            ),
            Some((window_start + chrono::Duration::seconds(65), 65))
        );
        assert_eq!(
            latest_directional_model_candidate(
                policy,
                window_start,
                window_start + chrono::Duration::seconds(300),
            ),
            Some((window_start + chrono::Duration::seconds(240), 240))
        );
        assert_eq!(
            latest_directional_model_candidate(
                policy,
                window_start,
                window_start + chrono::Duration::seconds(59),
            ),
            None
        );
    }

    #[test]
    fn directional_model_process_runtime_advances_candidates_until_opportunity_is_sealed() {
        let first = Utc.with_ymd_and_hms(2026, 7, 27, 12, 1, 0).unwrap();
        let mut runtime = DirectionalModelProcessRuntime::default();

        assert_eq!(runtime.claim("market-a", first), Some((first, true)));
        assert_eq!(runtime.claim("market-a", first), None);
        runtime.release("market-a", first);
        assert_eq!(
            runtime.claim("market-a", first + chrono::Duration::seconds(5)),
            Some((first, true))
        );
        runtime.mark_rehydrated("market-a", first);
        runtime.release("market-a", first);
        assert_eq!(
            runtime.claim("market-a", first + chrono::Duration::seconds(5)),
            Some((first, false))
        );
        assert!(runtime.complete("market-a", first, false));
        assert_eq!(runtime.claim("market-a", first), None);
        assert_eq!(
            runtime.claim("market-a", first + chrono::Duration::seconds(10)),
            Some((first + chrono::Duration::seconds(10), false))
        );
        assert!(runtime.complete("market-a", first + chrono::Duration::seconds(10), true));
        assert_eq!(
            runtime.claim("market-a", first + chrono::Duration::seconds(15)),
            None
        );
        assert_eq!(runtime.claim("market-b", first), Some((first, true)));
    }

    #[test]
    fn directional_model_rehydration_seals_only_an_existing_entry() {
        let first = Utc.with_ymd_and_hms(2026, 8, 15, 12, 1, 0).unwrap();
        let next = first + chrono::Duration::seconds(5);

        let mut rejected_prediction = DirectionalModelProcessRuntime::default();
        assert_eq!(
            rejected_prediction.claim("market-a", first),
            Some((first, true))
        );
        rejected_prediction.mark_rehydrated("market-a", first);
        assert!(rejected_prediction.complete("market-a", first, false));
        assert_eq!(
            rejected_prediction.claim("market-a", next),
            Some((next, false))
        );

        let mut existing_entry = DirectionalModelProcessRuntime::default();
        assert_eq!(existing_entry.claim("market-a", first), Some((first, true)));
        existing_entry.mark_rehydrated("market-a", first);
        assert!(existing_entry.complete("market-a", first, true));
        assert_eq!(existing_entry.claim("market-a", next), None);
    }

    #[test]
    fn directional_model_runtime_state_is_isolated_per_trading_process_runner() {
        let first = Utc.with_ymd_and_hms(2026, 7, 27, 12, 1, 0).unwrap();
        let mut strategy_only = DirectionalModelProcessRuntime::default();
        let mut loss_floor = DirectionalModelProcessRuntime::default();

        assert_eq!(
            strategy_only.claim("shared-market", first),
            Some((first, true))
        );
        assert_eq!(
            loss_floor.claim("shared-market", first),
            Some((first, true))
        );
        assert!(strategy_only.complete("shared-market", first, true));
        assert!(loss_floor.complete("shared-market", first, false));

        let next = first + chrono::Duration::seconds(5);
        assert_eq!(strategy_only.claim("shared-market", next), None);
        assert_eq!(loss_floor.claim("shared-market", next), Some((next, false)));
    }

    fn metadata_intent(strategy_version: &str) -> ApprovedIntent {
        ApprovedIntent {
            intent_id: Uuid::from_u128(201),
            process_id: Uuid::from_u128(202),
            feature_snapshot_id: Uuid::from_u128(203),
            market_id: "market".to_string(),
            window_start: Utc::now(),
            outcome: BtcOutcome::Up,
            token_id: "up".to_string(),
            limit_price: dec!(0.50),
            size: dec!(5),
            expected_net_edge: dec!(0.10),
            expected_net_edge_per_share: dec!(0.02),
            strategy_version: strategy_version.to_string(),
            feature_schema_version: BTC_FEATURE_SCHEMA_VERSION.to_string(),
        }
    }

    #[test]
    fn order_metadata_binds_validation_policy_to_directional_model() {
        let config = BtcStrategyConfig {
            strategy_version: BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION.to_string(),
            feature_schema_version: BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION.to_string(),
            decision_strategy: Some(BtcDecisionStrategyConfig::BtcDirectionalModel {
                model_key: BTC_DIRECTIONAL_MODEL_V1_KEY.to_string(),
                artifact_sha256: BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256.to_string(),
                feature_schema_sha256: BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256.to_string(),
            }),
            ..BtcStrategyConfig::default()
        };
        let prediction = BtcStrategyPrediction::DirectionalPrediction {
            outcome: BtcOutcome::Down,
            probability: dec!(0.92),
            conservative_probability: dec!(0.92),
            minimum_conservative_probability: dec!(0.89),
            probability_uncertainty: Decimal::ZERO,
            executable_price: Some(dec!(0.95)),
            direct_taker_fee_per_share: Some(dec!(0.002)),
            direct_net_edge_per_share: Some(dec!(-0.032)),
            entry_policy: BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction,
        };
        let mut intent = metadata_intent(&config.strategy_version);
        intent.outcome = BtcOutcome::Down;

        let metadata = btc_entry_order_metadata(
            &config,
            &intent,
            Some(&prediction),
            Uuid::from_u128(204),
            Uuid::from_u128(202),
            Uuid::from_u128(205),
            dec!(0.03),
        )
        .unwrap();

        assert_eq!(metadata["strategy"], BTC_DIRECTIONAL_MODEL_STRATEGY_FAMILY);
        assert_eq!(
            metadata["strategy_version"],
            BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION
        );
        assert_eq!(
            metadata["prediction"]["entry_policy"],
            "execute_directional_prediction"
        );
    }

    #[test]
    fn paper_capital_reconciliation_is_due_at_five_second_boundaries() {
        let started_at = Instant::now();
        let interval = TokioDuration::from_secs(5);
        assert!(execution_reconcile_due(None, started_at, interval));
        assert!(!execution_reconcile_due(
            Some(started_at),
            started_at + TokioDuration::from_millis(4_999),
            interval,
        ));
        assert!(execution_reconcile_due(
            Some(started_at),
            started_at + TokioDuration::from_secs(5),
            interval,
        ));
    }

    #[test]
    fn asynchronous_order_states_remain_submitted_until_reconciliation_finalizes_them() {
        for state in [
            OrderState::Created,
            OrderState::Submitted,
            OrderState::Acknowledged,
            OrderState::PartiallyFilled,
            OrderState::CancelRequested,
            OrderState::Unknown,
        ] {
            assert_eq!(decision_execution_status(state), "submitted");
        }
        assert_eq!(decision_execution_status(OrderState::Filled), "filled");
        for state in [
            OrderState::Rejected,
            OrderState::Cancelled,
            OrderState::Expired,
        ] {
            assert_eq!(decision_execution_status(state), "rejected");
        }
    }

    #[test]
    fn execution_result_metadata_preserves_paper_shape_and_names_live_evidence() {
        let report = OrderPlanReport {
            plan_id: Uuid::from_u128(601),
            orders: Vec::new(),
            fills: Vec::new(),
            reconciliation: crate::execution::ReconciliationReport {
                open_orders: 0,
                balances_checked: true,
                mismatches_found: 0,
                unresolved_count: 0,
                checked_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
            },
        };
        let paper =
            execution_result_metadata(BtcExecutionMode::Paper, &report, "paper-config", Vec::new());
        assert!(paper.get("paper_order_plan").is_some());
        assert!(paper.get("paper_stress_previews").is_some());
        assert!(paper.get("execution_mode").is_none());
        assert!(paper.get("order_plan").is_none());

        let live =
            execution_result_metadata(BtcExecutionMode::Live, &report, "live-config", Vec::new());
        assert_eq!(live["execution_mode"], "live");
        assert!(live.get("order_plan").is_some());
        assert!(live.get("stress_previews").is_some());
        assert!(live.get("paper_order_plan").is_none());
    }

    #[test]
    fn absent_high_water_mark_preserves_legacy_admission_evidence_shape() {
        let floor = LossRegimeConfidenceFloorConfig {
            schema_version: LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION.to_string(),
            activation_consecutive_candidate_losses: 2,
            min_conservative_probability: dec!(0.50),
            release_consecutive_candidate_wins: 1,
        };
        let loss = LossRegimeConfidenceFloorState::default()
            .evaluate(&floor, dec!(0.60))
            .unwrap();
        let expected = serde_json::to_value(&loss).unwrap();
        let combined =
            combine_entry_admission_evaluations(Uuid::from_u128(300), loss, None).unwrap();

        assert_eq!(combined.disposition, AdmissionDisposition::Allow);
        assert_eq!(combined.evidence, expected);
    }

    #[test]
    fn high_water_mark_defer_wins_and_keeps_both_policy_evaluations() {
        let floor = LossRegimeConfidenceFloorConfig {
            schema_version: LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION.to_string(),
            activation_consecutive_candidate_losses: 2,
            min_conservative_probability: dec!(0.50),
            release_consecutive_candidate_wins: 1,
        };
        let loss = LossRegimeConfidenceFloorState::default()
            .evaluate(&floor, dec!(0.60))
            .unwrap();
        let process_id = Uuid::from_u128(301);
        let as_of = Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap();
        let state = DailyRealizedPnlHighWaterMarkState::from_evidence(
            process_id,
            as_of,
            vec![DailyRealizedPnlCredit {
                settlement_id: Uuid::from_u128(302),
                order_id: "credited-order".to_string(),
                credited_at: as_of - chrono::Duration::minutes(1),
                net_pnl_usd: dec!(6),
            }],
            vec![UnsettledEntryExposure {
                order_id: "unsettled-order".to_string(),
                fill_ids: vec![Uuid::from_u128(303)],
                entry_debit_usd: dec!(1),
            }],
        )
        .unwrap();
        let proposed = ProposedEntryExposure::new(dec!(5), dec!(1), Decimal::ZERO).unwrap();
        let high_water_mark = state
            .evaluate(
                &DailyRealizedPnlHighWaterMarkConfig {
                    schema_version: DAILY_REALIZED_PNL_HIGH_WATER_MARK_SCHEMA_VERSION.to_string(),
                    activation_realized_pnl_usd: dec!(5),
                    max_drawdown_from_high_water_mark_usd: dec!(5),
                },
                &proposed,
            )
            .unwrap();
        let combined =
            combine_entry_admission_evaluations(process_id, loss, Some(high_water_mark)).unwrap();

        assert_eq!(combined.disposition, AdmissionDisposition::Defer);
        assert_eq!(combined.evidence["process_id"], process_id.to_string());
        assert_eq!(
            combined.evidence["blocking_policies"],
            serde_json::json!(["daily_realized_pnl_high_water_mark"])
        );
        assert!(combined
            .evidence
            .get("loss_regime_confidence_floor")
            .is_some());
        assert!(combined
            .evidence
            .get("daily_realized_pnl_high_water_mark")
            .is_some());
    }

    #[test]
    fn realized_volatility_is_positive_for_moving_prices() {
        let start = Utc::now();
        let ticks = [dec!(100), dec!(101), dec!(99)]
            .into_iter()
            .enumerate()
            .map(|(index, price)| ReferencePriceTick {
                tick_id: Uuid::new_v4(),
                dedup_key: index.to_string(),
                source: super::super::types::ReferencePriceSource::DirectBinance,
                symbol: "btcusdt".to_string(),
                price,
                source_timestamp: start + chrono::Duration::seconds(index as i64),
                envelope_timestamp: None,
                received_at: start + chrono::Duration::seconds(index as i64),
                connection_id: Uuid::new_v4(),
                ingest_sequence: index as u64,
                source_event_id: None,
                raw_payload: serde_json::json!({}),
            })
            .collect::<Vec<_>>();
        assert!(realized_volatility(&ticks).unwrap() > Decimal::ZERO);
    }

    #[test]
    fn directional_model_execution_inputs_preserve_only_required_point_in_time_lineage() {
        let window_start = Utc.with_ymd_and_hms(2026, 7, 27, 12, 0, 0).unwrap();
        let observed_at = window_start + chrono::Duration::seconds(180);
        let connection_id = Uuid::from_u128(501);
        let market = BtcIntervalMarket {
            event_id: "event".to_string(),
            event_slug: "btc-updown-5m-1785153600".to_string(),
            series_slug: "btc-up-or-down-5m".to_string(),
            market_id: "model-market".to_string(),
            condition_id: "condition".to_string(),
            window_start,
            window_end: window_start + chrono::Duration::seconds(300),
            up_token_id: "up".to_string(),
            down_token_id: "down".to_string(),
            tick_size: dec!(0.01),
            minimum_order_size: Some(dec!(1)),
            resolution_source: "chainlink".to_string(),
            active: true,
            closed: false,
            accepting_orders: true,
            fees_enabled: true,
            fee_schedule: serde_json::json!({}),
            raw_payload: serde_json::json!({}),
        };
        let binance = ReferencePriceTick {
            tick_id: Uuid::from_u128(502),
            dedup_key: "binance-current".to_string(),
            source: super::super::types::ReferencePriceSource::DirectBinance,
            symbol: "BTCUSD".to_string(),
            price: dec!(118000),
            source_timestamp: observed_at - chrono::Duration::milliseconds(10),
            envelope_timestamp: None,
            received_at: observed_at - chrono::Duration::milliseconds(8),
            connection_id: Uuid::from_u128(503),
            ingest_sequence: 504,
            source_event_id: None,
            raw_payload: serde_json::json!({}),
        };
        let checkpoint = |outcome: &str, checkpoint_id: u128, sequence| OrderbookCheckpoint {
            checkpoint_id: Uuid::from_u128(checkpoint_id),
            market_id: market.market_id.clone(),
            token_id: outcome.to_string(),
            source_timestamp: observed_at - chrono::Duration::milliseconds(6),
            received_at: observed_at - chrono::Duration::milliseconds(4),
            observed_at: observed_at - chrono::Duration::milliseconds(2),
            connection_id,
            ingest_sequence: sequence,
            source_hash: None,
            tick_size: dec!(0.01),
            best_bid: Some(dec!(0.39)),
            best_ask: Some(dec!(0.40)),
            bids: vec![super::super::types::OrderbookLevel {
                price: dec!(0.39),
                size: dec!(10),
            }],
            asks: vec![super::super::types::OrderbookLevel {
                price: dec!(0.40),
                size: dec!(10),
            }],
            integrity_status: FeedIntegrityStatus::Ok,
        };
        let inputs = BtcPointInTimeInputs {
            chainlink_open: None,
            chainlink_current: None,
            chainlink_history: Vec::new(),
            binance_history: vec![binance.clone()],
            up_book: Some(checkpoint("up", 505, 506)),
            down_book: Some(checkpoint("down", 507, 508)),
            fee_rate: Some(dec!(0.25)),
            fee_observed_at: Some(observed_at - chrono::Duration::milliseconds(2)),
        };
        let directional_model = BtcDirectionalModelFeatureSnapshot {
            model_key: BTC_DIRECTIONAL_MODEL_V1_KEY.to_string(),
            model_artifact_sha256: BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256.to_string(),
            feature_schema_version: BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION.to_string(),
            feature_schema_sha256: BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256.to_string(),
            feature_as_of: observed_at,
            seconds_elapsed: 180,
            feature_values: Vec::new(),
            input_sha256: "a".repeat(64),
        };

        let snapshot = build_snapshot(
            Uuid::from_u128(509),
            &market,
            observed_at,
            &inputs,
            dec!(5),
            BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION,
            observed_at,
            Some(directional_model),
        );

        assert_eq!(snapshot.chainlink_open_price, None);
        assert_eq!(snapshot.chainlink_price, None);
        assert_eq!(snapshot.chainlink_age_ms, None);
        assert_eq!(snapshot.binance_price, Some(binance.price));
        assert_eq!(snapshot.binance_return_1s, None);
        assert_eq!(snapshot.binance_return_5s, None);
        assert_eq!(snapshot.binance_return_30s, None);
        assert_eq!(snapshot.realized_volatility, None);
        assert_eq!(snapshot.binance_chainlink_basis_bps, None);
        assert_eq!(snapshot.lineage.binance_tick_id, Some(binance.tick_id));
        assert_eq!(
            snapshot.lineage.binance_source_timestamp,
            Some(binance.source_timestamp)
        );
        assert_eq!(
            snapshot.lineage.binance_received_at,
            Some(binance.received_at)
        );
        assert_eq!(
            snapshot.lineage.binance_ingest_sequence,
            Some(binance.ingest_sequence)
        );
        assert_eq!(
            snapshot.lineage.up_book_checkpoint_id,
            Some(Uuid::from_u128(505))
        );
        assert_eq!(
            snapshot.lineage.down_book_checkpoint_id,
            Some(Uuid::from_u128(507))
        );
        assert_eq!(snapshot.lineage.up_book_connection_id, Some(connection_id));
        assert_eq!(
            snapshot.lineage.down_book_connection_id,
            Some(connection_id)
        );
        assert_eq!(snapshot.up_book.executable_ask_vwap, Some(dec!(0.40)));
        assert_eq!(snapshot.down_book.executable_ask_vwap, Some(dec!(0.40)));
        assert_eq!(snapshot.fee_rate, Some(dec!(0.25)));
    }

    #[test]
    fn directional_execution_inputs_preserve_observation_during_registry_advance() {
        let window_start = Utc.with_ymd_and_hms(2026, 9, 3, 12, 0, 0).unwrap();
        let as_of = window_start + chrono::Duration::seconds(30);
        let market = BtcIntervalMarket {
            event_id: "event".to_string(),
            event_slug: "btc-updown-5m".to_string(),
            series_slug: "btc-up-or-down-5m".to_string(),
            market_id: "market".to_string(),
            condition_id: "condition".to_string(),
            window_start,
            window_end: window_start + chrono::Duration::minutes(5),
            up_token_id: "up".to_string(),
            down_token_id: "down".to_string(),
            tick_size: dec!(0.01),
            minimum_order_size: Some(dec!(1)),
            resolution_source: "chainlink".to_string(),
            active: true,
            closed: false,
            accepting_orders: true,
            fees_enabled: true,
            fee_schedule: serde_json::json!({"rate": "0.25"}),
            raw_payload: serde_json::json!({}),
        };
        let epoch = Uuid::from_u128(700);
        let mut registry = BookRegistry::new(epoch);
        for (token_id, outcome, sequence) in
            [("up", BtcOutcome::Up, 1), ("down", BtcOutcome::Down, 2)]
        {
            registry
                .apply_canonical_snapshot(
                    epoch,
                    &market,
                    token_id,
                    outcome,
                    as_of - chrono::Duration::milliseconds(20),
                    as_of - chrono::Duration::milliseconds(10),
                    sequence,
                    None,
                    vec![(dec!(0.49), dec!(10))],
                    vec![(dec!(0.50), dec!(10))],
                )
                .unwrap();
        }
        let binance = ReferencePriceTick {
            tick_id: Uuid::from_u128(701),
            dedup_key: "binance".to_string(),
            source: ReferencePriceSource::DirectBinance,
            symbol: "BTCUSDT".to_string(),
            price: dec!(118000),
            source_timestamp: as_of - chrono::Duration::milliseconds(20),
            envelope_timestamp: None,
            received_at: as_of - chrono::Duration::milliseconds(10),
            connection_id: Uuid::from_u128(702),
            ingest_sequence: 3,
            source_event_id: Some("kline".to_string()),
            raw_payload: serde_json::json!({}),
        };
        let mut state = RealtimeState::default();
        state.update_reference_price(binance.clone());
        state.update_books(&registry);
        for token in ["up", "down"] {
            Arc::make_mut(&mut state.unified_book_history)
                .observe(registry.checkpoint(token).unwrap());
        }
        // The observation is immutable while incoming publications advance the registry.
        registry
            .apply_canonical_snapshot(
                epoch,
                &market,
                "up",
                BtcOutcome::Up,
                as_of + chrono::Duration::milliseconds(1),
                as_of + chrono::Duration::milliseconds(2),
                4,
                None,
                vec![(dec!(0.59), dec!(10))],
                vec![(dec!(0.60), dec!(10))],
            )
            .unwrap();

        let inputs = directional_model_execution_inputs_from_runtime(
            &state,
            Uuid::nil(),
            &market,
            as_of,
            as_of,
            chrono::Duration::seconds(2),
            chrono::Duration::seconds(2),
        )
        .unwrap();

        assert_eq!(inputs.binance_history, vec![binance]);
        assert_eq!(inputs.up_book.as_ref().unwrap().token_id, "up");
        assert_eq!(inputs.down_book.as_ref().unwrap().token_id, "down");
        assert_eq!(inputs.up_book.as_ref().unwrap().connection_id, epoch);
        assert_eq!(inputs.up_book.as_ref().unwrap().best_ask, Some(dec!(0.50)));
        assert_eq!(
            registry.checkpoint("up").unwrap().best_ask,
            Some(dec!(0.60))
        );
        assert_eq!(inputs.fee_rate, Some(dec!(0.25)));
        assert_eq!(inputs.fee_observed_at, Some(window_start));

        let future_as_of = as_of - chrono::Duration::milliseconds(15);
        let filtered = directional_model_execution_inputs_from_runtime(
            &state,
            Uuid::nil(),
            &market,
            future_as_of,
            future_as_of,
            chrono::Duration::seconds(2),
            chrono::Duration::seconds(2),
        )
        .unwrap();
        assert!(filtered.binance_history.is_empty());
        assert!(filtered.up_book.is_none());
        assert!(filtered.down_book.is_none());
    }

    #[test]
    fn book_walk_records_full_target_limit() {
        let now = Utc::now();
        let checkpoint = OrderbookCheckpoint {
            checkpoint_id: Uuid::new_v4(),
            market_id: "m".to_string(),
            token_id: "up".to_string(),
            source_timestamp: now,
            received_at: now,
            observed_at: now,
            connection_id: Uuid::new_v4(),
            ingest_sequence: 1,
            source_hash: None,
            tick_size: dec!(0.01),
            best_bid: Some(dec!(0.49)),
            best_ask: Some(dec!(0.50)),
            bids: vec![],
            asks: vec![
                super::super::types::OrderbookLevel {
                    price: dec!(0.50),
                    size: dec!(2),
                },
                super::super::types::OrderbookLevel {
                    price: dec!(0.51),
                    size: dec!(3),
                },
            ],
            integrity_status: FeedIntegrityStatus::Ok,
        };
        let features = book_features(BtcOutcome::Up, "up", Some(&checkpoint), now, dec!(5));
        assert_eq!(features.quoted_size, dec!(5));
        assert_eq!(features.marketable_limit_price, Some(dec!(0.51)));
        assert_eq!(features.executable_ask_vwap, Some(dec!(0.506)));

        let mut delayed = checkpoint.clone();
        delayed.source_timestamp = now - chrono::Duration::milliseconds(2_500);
        delayed.received_at = now;
        let delayed_features = book_features(BtcOutcome::Up, "up", Some(&delayed), now, dec!(5));
        assert_eq!(delayed_features.age_ms, Some(2_500));

        let mut empty = checkpoint;
        empty.best_bid = None;
        empty.best_ask = None;
        empty.bids.clear();
        empty.asks.clear();
        let empty_features = book_features(BtcOutcome::Up, "up", Some(&empty), now, dec!(5));
        assert_eq!(empty_features.imbalance, None);
        assert_eq!(empty_features.executable_ask_vwap, None);
        assert_eq!(empty_features.quoted_size, Decimal::ZERO);
    }

    #[test]
    fn runtime_not_ready_revokes_an_otherwise_approved_intent() {
        let now = Utc::now();
        let prediction = BtcStrategyPrediction::DirectionalPrediction {
            outcome: BtcOutcome::Up,
            probability: dec!(0.82),
            conservative_probability: dec!(0.78),
            minimum_conservative_probability: dec!(0.75),
            probability_uncertainty: dec!(0.04),
            executable_price: Some(dec!(0.72)),
            direct_taker_fee_per_share: Some(dec!(0.01)),
            direct_net_edge_per_share: Some(dec!(0.09)),
            entry_policy: BtcDirectionalModelEntryPolicy::default(),
        };
        let mut decision = BtcDecision {
            decision_id: Uuid::new_v4(),
            process_id: Uuid::new_v4(),
            feature_snapshot_id: Uuid::new_v4(),
            evaluated_at: now,
            action: BtcDecisionAction::BuyUp,
            reject_reason: None,
            fair_value: None,
            up_edge: None,
            down_edge: None,
            approved_intent: Some(ApprovedIntent {
                intent_id: Uuid::new_v4(),
                process_id: Uuid::new_v4(),
                feature_snapshot_id: Uuid::new_v4(),
                market_id: "market".to_string(),
                window_start: now,
                outcome: BtcOutcome::Up,
                token_id: "up".to_string(),
                limit_price: dec!(0.5),
                size: dec!(5),
                expected_net_edge: dec!(0.1),
                expected_net_edge_per_share: dec!(0.02),
                strategy_version: "test".to_string(),
                feature_schema_version: super::super::strategy::BTC_FEATURE_SCHEMA_VERSION
                    .to_string(),
            }),
            prediction: Some(prediction.clone()),
        };
        enforce_runtime_readiness(
            &mut decision,
            &Readiness::default(),
            &BtcStrategyConfig::default(),
        );
        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(
            decision.reject_reason,
            Some(BtcRejectReason::RuntimeNotReady)
        );
        assert!(decision.approved_intent.is_none());
        assert_eq!(decision.prediction, Some(prediction));
    }

    #[test]
    fn directional_model_runtime_readiness_ignores_only_chainlink_reference_failures() {
        let now = Utc::now();
        let prediction = BtcStrategyPrediction::DirectionalPrediction {
            outcome: BtcOutcome::Up,
            probability: dec!(0.92),
            conservative_probability: dec!(0.92),
            minimum_conservative_probability: dec!(0.89),
            probability_uncertainty: Decimal::ZERO,
            executable_price: Some(dec!(0.80)),
            direct_taker_fee_per_share: Some(dec!(0.01)),
            direct_net_edge_per_share: Some(dec!(0.11)),
            entry_policy: BtcDirectionalModelEntryPolicy::default(),
        };
        let approved = BtcDecision {
            decision_id: Uuid::new_v4(),
            process_id: Uuid::new_v4(),
            feature_snapshot_id: Uuid::new_v4(),
            evaluated_at: now,
            action: BtcDecisionAction::BuyUp,
            reject_reason: None,
            fair_value: None,
            up_edge: None,
            down_edge: None,
            approved_intent: Some(ApprovedIntent {
                intent_id: Uuid::new_v4(),
                process_id: Uuid::new_v4(),
                feature_snapshot_id: Uuid::new_v4(),
                market_id: "market".to_string(),
                window_start: now,
                outcome: BtcOutcome::Up,
                token_id: "up".to_string(),
                limit_price: dec!(0.80),
                size: dec!(5),
                expected_net_edge: dec!(0.55),
                expected_net_edge_per_share: dec!(0.11),
                strategy_version: "test".to_string(),
                feature_schema_version: BTC_FEATURE_SCHEMA_VERSION.to_string(),
            }),
            prediction: Some(prediction),
        };
        let model_config = BtcStrategyConfig {
            decision_strategy: Some(BtcDecisionStrategyConfig::BtcDirectionalModel {
                model_key: BTC_DIRECTIONAL_MODEL_V1_KEY.to_string(),
                artifact_sha256: BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256.to_string(),
                feature_schema_sha256: BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256.to_string(),
            }),
            ..BtcStrategyConfig::default()
        };
        for reason in [
            "missing_reference:rtds_chainlink",
            "future_reference:rtds_chainlink",
            "stale_reference:rtds_chainlink",
        ] {
            let readiness = Readiness {
                ready: false,
                checked_at: now,
                reasons: vec![reason.to_string()],
                ..Readiness::default()
            };
            let mut decision = approved.clone();
            enforce_runtime_readiness(&mut decision, &readiness, &model_config);
            assert_eq!(decision.action, BtcDecisionAction::BuyUp);
            assert!(decision.approved_intent.is_some());

            let mut legacy = approved.clone();
            enforce_runtime_readiness(&mut legacy, &readiness, &BtcStrategyConfig::default());
            assert_eq!(legacy.action, BtcDecisionAction::NoTrade);
            assert_eq!(legacy.reject_reason, Some(BtcRejectReason::RuntimeNotReady));
        }
        for reason in [
            "missing_reference:direct_binance",
            "stale_reference:direct_binance",
            "missing_book:up",
            "book_integrity:up",
            "market_not_in_trade_window",
            "primary_persistence_unavailable",
        ] {
            let readiness = Readiness {
                ready: false,
                checked_at: now,
                reasons: vec![reason.to_string()],
                ..Readiness::default()
            };
            let mut decision = approved.clone();
            enforce_runtime_readiness(&mut decision, &readiness, &model_config);
            assert_eq!(decision.action, BtcDecisionAction::NoTrade);
            assert_eq!(
                decision.reject_reason,
                Some(BtcRejectReason::RuntimeNotReady)
            );
            assert!(decision.approved_intent.is_none());
        }
    }

    #[test]
    fn declared_model_feeds_isolate_unrelated_reference_failures() {
        let now = Utc::now();
        let readiness = Readiness {
            ready: false,
            checked_at: now,
            reasons: vec!["stale_reference:direct_binance".to_string()],
            ..Readiness::default()
        };
        let model = BtcDecisionStrategyConfig::BtcDirectionalModel {
            model_key: BTC_DIRECTIONAL_MODEL_V1_KEY.to_string(),
            artifact_sha256: BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256.to_string(),
            feature_schema_sha256: BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256.to_string(),
        };
        let l2_only = BtcStrategyConfig {
            decision_strategy: Some(model.clone()),
            required_model_feeds: vec![BtcModelFeedRequirement {
                feed: BtcModelFeedId::BinanceBtcusdtL2V1,
                maximum_age_ms: 2_000,
                require_sequence_integrity: true,
            }],
            ..BtcStrategyConfig::default()
        };
        let one_second = BtcStrategyConfig {
            decision_strategy: Some(model),
            required_model_feeds: vec![BtcModelFeedRequirement {
                feed: BtcModelFeedId::BinanceBtcusdtOneSecondV1,
                maximum_age_ms: 1_000,
                require_sequence_integrity: false,
            }],
            ..BtcStrategyConfig::default()
        };

        assert!(runtime_readiness_satisfied(&l2_only, &readiness));
        assert!(!runtime_readiness_satisfied(&one_second, &readiness));
        let isolated = process_runtime_readiness(&l2_only, &readiness);
        assert!(isolated.ready);
        assert!(isolated.reasons.is_empty());
        let required = process_runtime_readiness(&one_second, &readiness);
        assert!(!required.ready);
        assert_eq!(required.reasons, readiness.reasons);
    }

    #[test]
    fn directional_model_validation_entry_policy_requires_executing_model_runner() {
        let model_config = BtcStrategyConfig {
            decision_strategy: Some(BtcDecisionStrategyConfig::BtcDirectionalModel {
                model_key: BTC_DIRECTIONAL_MODEL_V1_KEY.to_string(),
                artifact_sha256: BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256.to_string(),
                feature_schema_sha256: BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256.to_string(),
            }),
            ..BtcStrategyConfig::default()
        };
        assert!(validate_directional_model_entry_policy(
            &model_config,
            BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction,
            true,
        )
        .is_ok());
        assert!(validate_directional_model_entry_policy(
            &model_config,
            BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction,
            false,
        )
        .is_err());
        assert!(validate_directional_model_entry_policy(
            &BtcStrategyConfig::default(),
            BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction,
            true,
        )
        .is_err());
        assert!(validate_directional_model_entry_policy(
            &BtcStrategyConfig::default(),
            BtcDirectionalModelEntryPolicy::RequirePositiveDirectEdge,
            true,
        )
        .is_ok());
    }

    #[test]
    fn reconnect_epoch_requires_both_books_from_the_same_connection() {
        let market = BtcIntervalMarket {
            event_id: "event".to_string(),
            event_slug: "btc-updown-5m-0".to_string(),
            series_slug: "btc-up-or-down-5m".to_string(),
            market_id: "market".to_string(),
            condition_id: "condition".to_string(),
            window_start: Utc::now(),
            window_end: Utc::now() + chrono::Duration::minutes(5),
            up_token_id: "up".to_string(),
            down_token_id: "down".to_string(),
            tick_size: dec!(0.01),
            minimum_order_size: Some(dec!(1)),
            resolution_source: "chainlink".to_string(),
            active: true,
            closed: false,
            accepting_orders: true,
            fees_enabled: true,
            fee_schedule: serde_json::json!({}),
            raw_payload: serde_json::json!({}),
        };
        let first_epoch = Uuid::new_v4();
        let second_epoch = Uuid::new_v4();
        let book = |token_id: &str, connection_id| BookReadiness {
            market_id: market.market_id.clone(),
            token_id: token_id.to_string(),
            connection_id,
            bootstrapped: true,
            integrity_status: FeedIntegrityStatus::Ok,
            source_timestamp: Some(Utc::now()),
            received_at: Some(Utc::now()),
            best_bid: Some(dec!(0.49)),
            best_ask: Some(dec!(0.50)),
        };
        let mixed = Readiness {
            books: vec![book("up", first_epoch), book("down", second_epoch)],
            ..Readiness::default()
        };
        assert_eq!(observation_clob_connection_id(&market, &mixed), None);

        let coherent = Readiness {
            books: vec![book("up", second_epoch), book("down", second_epoch)],
            ..Readiness::default()
        };
        assert_eq!(
            observation_clob_connection_id(&market, &coherent),
            Some(second_epoch)
        );
    }
}
