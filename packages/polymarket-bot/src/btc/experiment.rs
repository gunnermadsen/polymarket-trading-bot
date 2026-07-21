use std::{
    collections::HashSet,
    sync::{Arc, Mutex as StdMutex},
};

use anyhow::{ensure, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::{prelude::ToPrimitive, Decimal};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    sync::{Mutex, OnceCell},
    time::{Duration as TokioDuration, Instant},
};
use tracing::warn;
use uuid::Uuid;

use crate::{
    execution::{execute_order_plan, OrderPlan},
    models::{OrderRequest, OrderSide, OrderState, OrderType},
    store::Store,
};

use super::{
    admission::{
        AdmissionDisposition, BtcEntryAdmissionConfig, DailyRealizedPnlHighWaterMarkEvaluation,
        LossRegimeConfidenceFloorEvaluation, LossRegimeConfidenceFloorState,
        LossRegimeConfidenceFloorTransition, ProposedEntryExposure,
        ShadowPredictiveRegimeCandidate, ShadowPredictiveRegimeCircuitBreakerConfig,
        ShadowPredictiveRegimeEvaluation, ShadowPredictiveRegimeState,
        ShadowPredictiveRegimeTransition,
    },
    execution_guard::BtcReferenceExecutionGuard,
    paper::{PaperPreviewConfig, PaperVenue, PAPER_DYNAMIC_FEE_RATE_METADATA_KEY},
    repository::{BtcPointInTimeInputs, BtcRepository},
    runtime::{BtcStrategyRunner, StrategyObservation},
    strategy::{
        BtcDecision, BtcDecisionAction, BtcFeatureLineage, BtcFeatureSnapshot,
        BtcInputWindowLineage, BtcOutcomeBookFeatures, BtcRejectReason, BtcStrategyConfig,
        BtcStrategyPrediction, DeterministicBtcStrategy,
        BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_LINEAGE_VERSION,
        BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_SCHEMA_VERSION,
        BTC_CHAINLINK_PERSISTENCE_CALIBRATED_FEATURE_SCHEMA_VERSION, BTC_FEATURE_LINEAGE_VERSION,
        BTC_MARKET_ANCHORED_DIRECTIONAL_PREDICTION_STRATEGY_FAMILY,
    },
    types::{
        BtcIntervalMarket, BtcOutcome, FeedIntegrityStatus, OrderbookCheckpoint, ReferencePriceTick,
    },
};

const PAPER_CAPITAL_RECONCILE_INTERVAL: TokioDuration = TokioDuration::from_secs(5);
const SHADOW_PREDICTIVE_REGIME_MAX_REPLAY_CANDIDATES: u32 = 10_000;
const SHADOW_PREDICTIVE_REGIME_REPLAY_FETCH_CANDIDATES: u32 =
    SHADOW_PREDICTIVE_REGIME_MAX_REPLAY_CANDIDATES + 1;
const SHADOW_PREDICTIVE_REGIME_TRANSITION_EVENT_NAMESPACE: Uuid =
    Uuid::from_u128(0x8f0d_73b4_4e62_5b31_9a77_21cf_09d8_6a42);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BtcPaperExperimentConfig {
    pub experiment_id: Uuid,
    pub experiment_name: String,
    pub process_id: Uuid,
    pub config_hash: String,
    /// Full immutable `TradingProcessConfig` snapshot for this experiment run.
    /// The reusable process definition may be changed after the run stops, so
    /// evidence consumers must read this run-owned value instead.
    pub frozen_process_config: serde_json::Value,
    pub strategy: BtcStrategyConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_admission: Option<BtcEntryAdmissionConfig>,
    pub execution_enabled: bool,
    pub paper_stress_previews: Vec<PaperPreviewConfig>,
}

#[derive(Debug, Default)]
struct LossRegimeAdmissionRuntime {
    state: LossRegimeConfidenceFloorState,
    evaluated_market_id: Option<String>,
}

struct ShadowPredictiveRegimeAdmissionRuntime {
    state: ShadowPredictiveRegimeState,
    state_hydrated: bool,
    evaluated_market_id: Option<String>,
    attempted_market_id: Option<String>,
    refresh_in_progress: bool,
    telemetry_error: Option<String>,
}

#[derive(Default)]
struct ShadowPredictiveRegimeRefreshTasks {
    stopping: bool,
    handles: Vec<tokio::task::JoinHandle<()>>,
}

type ShadowPredictiveRegimeTransitionEvidence = (
    ShadowPredictiveRegimeTransition,
    ShadowPredictiveRegimeCandidate,
    ShadowPredictiveRegimeState,
);

fn shadow_predictive_regime_refresh_pending(
    runtime: &ShadowPredictiveRegimeAdmissionRuntime,
    market_id: &str,
) -> bool {
    runtime.refresh_in_progress
        || !runtime.state_hydrated
        || runtime.evaluated_market_id.as_deref() != Some(market_id)
}

fn unavailable_shadow_predictive_regime_evaluation(
    process_id: Uuid,
    config: &ShadowPredictiveRegimeCircuitBreakerConfig,
    as_of: DateTime<Utc>,
    telemetry_error: String,
) -> Result<ShadowPredictiveRegimeEvaluation> {
    let mut evaluation =
        ShadowPredictiveRegimeState::new(process_id, config)?.evaluate(config, as_of)?;
    evaluation.state_checkpoint_eligible = false;
    evaluation.refresh_pending = true;
    evaluation.telemetry_error = Some(telemetry_error);
    Ok(evaluation)
}

fn shadow_predictive_regime_transition_event_id(
    process_id: Uuid,
    breaker_config_hash: &str,
    event_type: &str,
    state_evidence_sha256: &str,
) -> Uuid {
    let identity =
        format!("{process_id}:{breaker_config_hash}:{event_type}:{state_evidence_sha256}");
    Uuid::new_v5(
        &SHADOW_PREDICTIVE_REGIME_TRANSITION_EVENT_NAMESPACE,
        identity.as_bytes(),
    )
}

struct EntryAdmissionEvaluation {
    disposition: AdmissionDisposition,
    evidence: serde_json::Value,
}

pub struct BtcPaperExperimentRunner {
    repository: BtcRepository,
    store: Store,
    paper_venue: PaperVenue,
    config: BtcPaperExperimentConfig,
    initialized: OnceCell<()>,
    loss_regime_admission: Mutex<Option<LossRegimeAdmissionRuntime>>,
    shadow_predictive_regime_admission:
        Arc<StdMutex<Option<ShadowPredictiveRegimeAdmissionRuntime>>>,
    shadow_predictive_regime_refresh_tasks: StdMutex<ShadowPredictiveRegimeRefreshTasks>,
    high_water_mark_entry_submission: Mutex<()>,
    paper_capital_reconcile_started_at: Mutex<Option<Instant>>,
}

impl BtcPaperExperimentRunner {
    pub fn new(
        repository: BtcRepository,
        store: Store,
        paper_venue: PaperVenue,
        config: BtcPaperExperimentConfig,
    ) -> Result<Self> {
        if config.experiment_name.trim().is_empty() || config.config_hash.trim().is_empty() {
            anyhow::bail!("BTC paper experiment identity and config hash must not be empty");
        }
        if !config.frozen_process_config.is_object() {
            anyhow::bail!("BTC paper experiment frozen process config must be a JSON object");
        }
        config.strategy.validate()?;
        if config.strategy.attribution().is_none() {
            anyhow::bail!("BTC paper experiment strategy attribution is invalid");
        }
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
        let shadow_predictive_regime_admission = config
            .entry_admission
            .as_ref()
            .and_then(|entry_admission| {
                entry_admission
                    .shadow_predictive_regime_circuit_breaker
                    .as_ref()
            })
            .map(|shadow| {
                Ok::<_, anyhow::Error>(ShadowPredictiveRegimeAdmissionRuntime {
                    state: ShadowPredictiveRegimeState::new(config.process_id, shadow)?,
                    state_hydrated: false,
                    evaluated_market_id: None,
                    attempted_market_id: None,
                    refresh_in_progress: false,
                    telemetry_error: None,
                })
            })
            .transpose()?;
        Ok(Self {
            repository,
            store,
            paper_venue,
            loss_regime_admission: Mutex::new(
                config
                    .entry_admission
                    .as_ref()
                    .map(|_| LossRegimeAdmissionRuntime::default()),
            ),
            shadow_predictive_regime_admission: Arc::new(StdMutex::new(
                shadow_predictive_regime_admission,
            )),
            shadow_predictive_regime_refresh_tasks: StdMutex::new(
                ShadowPredictiveRegimeRefreshTasks::default(),
            ),
            high_water_mark_entry_submission: Mutex::new(()),
            config,
            initialized: OnceCell::new(),
            paper_capital_reconcile_started_at: Mutex::new(None),
        })
    }

    /// Claims the immutable experiment identity before feeds begin.
    pub async fn initialize(&self) -> Result<()> {
        self.initialize_with_existing_identity(false).await
    }

    /// Reattaches an already-running immutable experiment.
    pub async fn resume(&self) -> Result<()> {
        self.initialize_with_existing_identity(true).await
    }

    async fn initialize_with_existing_identity(&self, resume: bool) -> Result<()> {
        self.initialized
            .get_or_try_init(|| async {
                if resume {
                    self.repository
                        .verify_resumable_paper_experiment(
                            self.config.experiment_id,
                            &self.config.experiment_name,
                            self.config.process_id,
                            &self.config.strategy.strategy_version,
                            &self.config.strategy.feature_schema_version,
                            &self.config.config_hash,
                            &self.config.frozen_process_config,
                        )
                        .await?;
                    let state = self
                        .repository
                        .paper_venue_resume_state(self.config.experiment_id)
                        .await?;
                    self.paper_venue
                        .rehydrate_capital(
                            state.entry_debits_usd,
                            state.settlement_credits_usd,
                            state.order_count,
                            state.fill_count,
                            state.credited_settlement_ids,
                        )
                        .await?;
                } else {
                    self.repository
                        .ensure_paper_experiment(
                            self.config.experiment_id,
                            &self.config.experiment_name,
                            self.config.process_id,
                            &self.config.strategy.strategy_version,
                            &self.config.strategy.feature_schema_version,
                            &self.config.config_hash,
                            &self.config.frozen_process_config,
                        )
                        .await?;
                }
                self.initialize_entry_admission(resume).await?;
                self.force_refresh_settlement_and_reconcile().await?;
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
        let shadow_initialized_state = if entry_admission
            .shadow_predictive_regime_circuit_breaker
            .is_some()
        {
            self.shadow_predictive_regime_admission
                .lock()
                .ok()
                .and_then(|admission| admission.as_ref().map(|runtime| runtime.state.clone()))
        } else {
            None
        };
        if entry_admission
            .shadow_predictive_regime_circuit_breaker
            .is_none()
        {
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
        } else {
            self.record_entry_admission_event(
                "btc_entry_admission_initialized",
                "BTC entry admission initialized with non-blocking shadow predictive-regime hydration pending",
                serde_json::json!({
                    "resume": resume,
                    "config": entry_admission,
                    "entry_admission_config_hash": floor.config_hash()?,
                    "state": initialized_state,
                    "shadow_predictive_regime_state": shadow_initialized_state,
                    "shadow_predictive_regime_hydration_pending": true,
                }),
            )
            .await;
        }
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

    fn cached_shadow_predictive_regime_evaluation(
        &self,
        market_id: &str,
        as_of: DateTime<Utc>,
    ) -> Option<ShadowPredictiveRegimeEvaluation> {
        let config = self
            .config
            .entry_admission
            .as_ref()
            .and_then(|entry_admission| {
                entry_admission
                    .shadow_predictive_regime_circuit_breaker
                    .as_ref()
            })?;
        let admission = match self.shadow_predictive_regime_admission.try_lock() {
            Ok(admission) => admission,
            Err(error) => {
                let telemetry_error =
                    format!("shadow predictive-regime cache lock was unavailable: {error}");
                warn!(
                    error = %telemetry_error,
                    process_id = %self.config.process_id,
                    "shadow predictive-regime cache lock failed open"
                );
                return unavailable_shadow_predictive_regime_evaluation(
                    self.config.process_id,
                    config,
                    as_of,
                    telemetry_error,
                )
                .map_err(|fallback_error| {
                    warn!(
                        error = %fallback_error,
                        process_id = %self.config.process_id,
                        "shadow predictive-regime fallback evaluation failed open"
                    );
                })
                .ok();
            }
        };
        let Some(runtime) = admission.as_ref() else {
            let telemetry_error =
                "shadow predictive-regime cache runtime was unavailable".to_string();
            warn!(
                process_id = %self.config.process_id,
                "shadow predictive-regime cache was unavailable and failed open"
            );
            return unavailable_shadow_predictive_regime_evaluation(
                self.config.process_id,
                config,
                as_of,
                telemetry_error,
            )
            .map_err(|fallback_error| {
                warn!(
                    error = %fallback_error,
                    process_id = %self.config.process_id,
                    "shadow predictive-regime fallback evaluation failed open"
                );
            })
            .ok();
        };
        match runtime.state.evaluate(config, as_of) {
            Ok(mut evaluation) => {
                evaluation.state_checkpoint_eligible = runtime.state_hydrated;
                evaluation.telemetry_error = runtime.telemetry_error.clone();
                evaluation.refresh_pending =
                    shadow_predictive_regime_refresh_pending(runtime, market_id);
                Some(evaluation)
            }
            Err(error) => {
                let telemetry_error = format!(
                    "shadow predictive-regime cached state could not be evaluated: {error}"
                );
                warn!(
                    error = %telemetry_error,
                    process_id = %self.config.process_id,
                    "shadow predictive-regime cached evaluation failed open"
                );
                unavailable_shadow_predictive_regime_evaluation(
                    self.config.process_id,
                    config,
                    as_of,
                    telemetry_error,
                )
                .map_err(|fallback_error| {
                    warn!(
                        error = %fallback_error,
                        process_id = %self.config.process_id,
                        "shadow predictive-regime fallback evaluation failed open"
                    );
                })
                .ok()
            }
        }
    }

    fn schedule_shadow_predictive_regime_refresh(&self, market_id: &str, as_of: DateTime<Utc>) {
        let Some(config) = self
            .config
            .entry_admission
            .as_ref()
            .and_then(|entry_admission| {
                entry_admission
                    .shadow_predictive_regime_circuit_breaker
                    .clone()
            })
        else {
            return;
        };
        let mut refresh_tasks = match self.shadow_predictive_regime_refresh_tasks.lock() {
            Ok(refresh_tasks) => refresh_tasks,
            Err(error) => {
                warn!(
                    error = %error,
                    process_id = %self.config.process_id,
                    "shadow predictive-regime task tracker failed open"
                );
                return;
            }
        };
        if refresh_tasks.stopping {
            return;
        }
        refresh_tasks.handles.retain(|handle| !handle.is_finished());
        let breaker_config_hash = match config.config_hash() {
            Ok(config_hash) => config_hash,
            Err(error) => {
                warn!(
                    error = %error,
                    process_id = %self.config.process_id,
                    "shadow predictive-regime config hash failed open"
                );
                return;
            }
        };
        let base_state = {
            let mut admission = match self.shadow_predictive_regime_admission.try_lock() {
                Ok(admission) => admission,
                Err(error) => {
                    warn!(
                        error = %error,
                        process_id = %self.config.process_id,
                        "shadow predictive-regime refresh lock failed open"
                    );
                    return;
                }
            };
            let Some(runtime) = admission.as_mut() else {
                warn!(
                    process_id = %self.config.process_id,
                    "shadow predictive-regime refresh state was unavailable"
                );
                return;
            };
            if runtime.refresh_in_progress
                || runtime.attempted_market_id.as_deref() == Some(market_id)
            {
                return;
            }
            runtime.refresh_in_progress = true;
            runtime.attempted_market_id = Some(market_id.to_string());
            runtime.state_hydrated.then(|| runtime.state.clone())
        };

        let repository = self.repository.clone();
        let store = self.store.clone();
        let runtime = self.shadow_predictive_regime_admission.clone();
        let process_id = self.config.process_id;
        let market_id = market_id.to_string();
        let handle = tokio::spawn(async move {
            let refreshed = load_shadow_predictive_regime_state(
                &repository,
                process_id,
                &config,
                as_of,
                base_state,
            )
            .await
            .and_then(|(state, transitions)| {
                let evaluation = state.evaluate(&config, as_of)?;
                Ok((state, evaluation, transitions))
            });

            let events = match refreshed {
                Ok((state, evaluation, transitions)) => {
                    let mut admission = match runtime.lock() {
                        Ok(admission) => admission,
                        Err(error) => {
                            warn!(
                                error = %error,
                                process_id = %process_id,
                                "shadow predictive-regime refresh result lock failed open"
                            );
                            return;
                        }
                    };
                    let Some(runtime) = admission.as_mut() else {
                        warn!(
                            process_id = %process_id,
                            "shadow predictive-regime refresh result had no runtime state"
                        );
                        return;
                    };
                    runtime.state = state;
                    runtime.state_hydrated = true;
                    runtime.evaluated_market_id = Some(market_id.clone());
                    runtime.refresh_in_progress = false;
                    runtime.telemetry_error = None;
                    drop(admission);
                    transitions
                        .into_iter()
                        .filter_map(|(transition, candidate, transition_state)| {
                            let (event_type, message) = match transition {
                                ShadowPredictiveRegimeTransition::DegradationConfirmed => (
                                    "btc_shadow_predictive_regime_degradation_confirmed",
                                    "shadow predictive-regime degradation confirmed",
                                ),
                                ShadowPredictiveRegimeTransition::RecoveryConfirmed => (
                                    "btc_shadow_predictive_regime_recovery_confirmed",
                                    "shadow predictive-regime recovery confirmed",
                                ),
                            };
                            let state_evidence_sha256 = match transition_state
                                .evidence_sha256(&config)
                            {
                                Ok(state_evidence_sha256) => state_evidence_sha256,
                                Err(error) => {
                                    warn!(
                                        error = %error,
                                        process_id = %process_id,
                                        event_type,
                                        "shadow predictive-regime transition identity failed open"
                                    );
                                    return None;
                                }
                            };
                            let event_id = shadow_predictive_regime_transition_event_id(
                                process_id,
                                &breaker_config_hash,
                                event_type,
                                &state_evidence_sha256,
                            );
                            let transition_timestamp_utc = candidate.label_available_at;
                            Some((
                                Some((event_id, transition_timestamp_utc)),
                                event_type,
                                message,
                                serde_json::json!({
                                    "transition_event_id": event_id,
                                    "shadow_predictive_regime_config_hash": breaker_config_hash,
                                    "transition_state_evidence_sha256": state_evidence_sha256,
                                    "candidate": candidate,
                                    "transition_state": transition_state,
                                    "refresh_evaluation": &evaluation,
                                }),
                            ))
                        })
                        .collect::<Vec<_>>()
                }
                Err(error) => {
                    let error = error.to_string();
                    warn!(
                        error = %error,
                        process_id = %process_id,
                        market_id,
                        "shadow predictive-regime refresh failed without changing admission"
                    );
                    if let Ok(mut admission) = runtime.lock() {
                        if let Some(runtime) = admission.as_mut() {
                            runtime.state_hydrated = false;
                            runtime.refresh_in_progress = false;
                            runtime.telemetry_error = Some(error.clone());
                        }
                    }
                    vec![(
                        None,
                        "btc_shadow_predictive_regime_telemetry_error",
                        "shadow predictive-regime telemetry failed without changing admission",
                        serde_json::json!({
                            "shadow_predictive_regime_config_hash": breaker_config_hash,
                            "market_id": market_id,
                            "error": error,
                        }),
                    )]
                }
            };

            for (event_identity, event_type, message, metadata) in events {
                let persisted = match event_identity {
                    Some((event_id, timestamp_utc)) => store
                        .record_trading_process_event_idempotent(
                            event_id,
                            timestamp_utc,
                            process_id,
                            "info",
                            event_type,
                            Some(message),
                            metadata,
                        )
                        .await
                        .map(|_| ()),
                    None => {
                        store
                            .record_trading_process_event(
                                process_id,
                                "info",
                                event_type,
                                Some(message),
                                metadata,
                            )
                            .await
                    }
                };
                if let Err(error) = persisted {
                    warn!(
                        error = %error,
                        process_id = %process_id,
                        event_type,
                        "failed to persist shadow predictive-regime event"
                    );
                }
            }
        });
        refresh_tasks.handles.push(handle);
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
        let shadow_predictive_regime = decision.approved_intent.as_ref().and_then(|intent| {
            self.cached_shadow_predictive_regime_evaluation(&intent.market_id, as_of)
        });
        Ok(Some(combine_entry_admission_evaluations(
            self.config.process_id,
            loss_regime,
            high_water_mark,
            shadow_predictive_regime,
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
        let refresh_handles = {
            let mut refresh_tasks = self
                .shadow_predictive_regime_refresh_tasks
                .lock()
                .map_err(|error| {
                    anyhow::anyhow!(
                        "shadow predictive-regime task tracker was poisoned during shutdown: {error}"
                    )
                })?;
            refresh_tasks.stopping = true;
            std::mem::take(&mut refresh_tasks.handles)
        };
        for handle in &refresh_handles {
            handle.abort();
        }
        for handle in refresh_handles {
            if let Err(error) = handle.await {
                if !error.is_cancelled() {
                    warn!(
                        error = %error,
                        process_id = %self.config.process_id,
                        "shadow predictive-regime refresh task failed during shutdown"
                    );
                }
            }
        }
        self.force_refresh_settlement_and_reconcile().await?;
        Ok(())
    }

    pub fn shared_book_registry(&self) -> Arc<tokio::sync::RwLock<super::feeds::BookRegistry>> {
        self.paper_venue.registry()
    }

    async fn refresh_settlement_and_reconcile(&self) -> Result<()> {
        self.repository
            .refresh_paper_experiment_settlement(self.config.experiment_id)
            .await?;
        self.reconcile_paper_capital().await
    }

    async fn force_refresh_settlement_and_reconcile(&self) -> Result<()> {
        let mut last_started_at = self.paper_capital_reconcile_started_at.lock().await;
        let previous = *last_started_at;
        *last_started_at = Some(Instant::now());
        let result = self.refresh_settlement_and_reconcile().await;
        if result.is_err() {
            *last_started_at = previous;
        }
        result
    }

    async fn refresh_settlement_and_reconcile_if_due(&self) -> Result<()> {
        let Ok(mut last_started_at) = self.paper_capital_reconcile_started_at.try_lock() else {
            return Ok(());
        };
        let now = Instant::now();
        if !paper_capital_reconcile_due(*last_started_at, now) {
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

    async fn reconcile_paper_capital(&self) -> Result<()> {
        let pending = self
            .repository
            .discover_pending_paper_settlements(self.config.experiment_id)
            .await?;
        for settlement in pending {
            let credit = self
                .paper_venue
                .apply_settlement_credit(settlement.settlement_id, settlement.payout)
                .await?;
            let evidence = serde_json::json!({
                "evidence_version": "btc_paper_capital_credit_v1",
                "settlement_id": settlement.settlement_id,
                "experiment_id": settlement.experiment_id,
                "process_id": settlement.process_id,
                "order_id": settlement.order_id,
                "market_id": settlement.market_id,
                "token_id": settlement.token_id,
                "fill_ids": settlement.fill_ids,
                "official_outcome": settlement.official_outcome,
                "official_winning_token_id": settlement.official_winning_token_id,
                "official_resolution_received_at": settlement.official_resolution_received_at,
                "official_resolution_source": settlement.official_resolution_source,
                "filled_size": settlement.filled_size,
                "entry_notional": settlement.entry_notional,
                "entry_fees": settlement.entry_fees,
                "payout": settlement.payout,
                "net_pnl": settlement.net_pnl,
                "venue_credit": credit,
                "credited_by_config_hash": self.config.config_hash,
            });
            let marked = self
                .repository
                .mark_paper_settlement_credited(
                    self.config.experiment_id,
                    settlement.settlement_id,
                    &evidence,
                )
                .await?;
            if !marked {
                warn!(
                    settlement_id = %settlement.settlement_id,
                    experiment_id = %self.config.experiment_id,
                    "BTC paper settlement was already credited by a concurrent reconciliation"
                );
            }
        }

        let venue = self.paper_venue.status().await;
        let ledger = self
            .repository
            .paper_settlement_ledger_summary(self.config.experiment_id)
            .await?;
        self.repository
            .update_paper_capital_runtime_summary(
                self.config.experiment_id,
                &serde_json::json!({
                    "status_version": "btc_paper_capital_v1",
                    "venue": venue,
                    "ledger": ledger,
                    "reconciled_at": Utc::now(),
                    "official_payout_only": true,
                }),
            )
            .await
    }

    async fn observe(&self, observation: StrategyObservation) -> Result<()> {
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
        self.refresh_settlement_and_reconcile_if_due().await?;

        let Some(market) = observation.state.current_market.as_ref() else {
            return Ok(());
        };
        // Keep durable point-in-time inputs on the same immutable observation boundary used by
        // runtime readiness. Initialization, reconciliation and admission must not move the
        // feature timestamp forward while feeds continue advancing.
        let observed_at = observation.readiness.checked_at;
        self.schedule_shadow_predictive_regime_refresh(&market.market_id, observed_at);
        let clob_connection_id = observation_clob_connection_id(market, &observation.readiness);
        let inputs = self
            .repository
            .load_point_in_time_inputs(
                market,
                observed_at,
                chrono::Duration::milliseconds(self.config.strategy.max_chainlink_open_delay_ms),
                chrono::Duration::milliseconds(self.config.strategy.max_reference_age_ms),
                chrono::Duration::milliseconds(self.config.strategy.max_book_age_ms),
                clob_connection_id,
            )
            .await?;
        let snapshot = build_snapshot(
            self.config.process_id,
            market,
            observed_at,
            &inputs,
            self.config.strategy.target_size,
            &self.config.strategy.feature_schema_version,
        );
        let mut decision = DeterministicBtcStrategy::evaluate(&self.config.strategy, &snapshot);
        enforce_runtime_readiness(&mut decision, &observation.readiness);
        if decision.approved_intent.is_some()
            && self
                .repository
                .process_has_entry(self.config.process_id, &snapshot.market_id)
                .await?
        {
            decision.action = BtcDecisionAction::NoTrade;
            decision.reject_reason = Some(BtcRejectReason::ExistingExperimentEntry);
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
        if !self
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
                    "quality_flags": quality_flags,
                }),
            )
            .await?
        {
            return Ok(());
        }

        let Some(intent) = decision.approved_intent.clone() else {
            self.repository
                .insert_strategy_decision(
                    self.config.experiment_id,
                    &self.config.config_hash,
                    &snapshot.market_id,
                    &self.config.strategy.strategy_version,
                    &decision,
                    None,
                    None,
                    "rejected",
                )
                .await?;
            self.repository
                .increment_experiment_counts(self.config.experiment_id, 1, 1, 0)
                .await?;
            return Ok(());
        };

        if !self.config.execution_enabled {
            self.repository
                .insert_strategy_decision(
                    self.config.experiment_id,
                    &self.config.config_hash,
                    &snapshot.market_id,
                    &self.config.strategy.strategy_version,
                    &decision,
                    None,
                    None,
                    "shadow_only",
                )
                .await?;
            self.repository
                .increment_experiment_counts(self.config.experiment_id, 1, 1, 0)
                .await?;
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
            self.repository
                .insert_strategy_decision(
                    self.config.experiment_id,
                    &self.config.config_hash,
                    &snapshot.market_id,
                    &self.config.strategy.strategy_version,
                    &decision,
                    entry_admission_evidence,
                    None,
                    "admission_blocked",
                )
                .await?;
            self.repository
                .increment_experiment_counts(self.config.experiment_id, 1, 1, 0)
                .await?;
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
            self.config.experiment_id,
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
            signal_id: None,
            metadata: order_metadata,
        };
        let reference_execution_guard = BtcReferenceExecutionGuard::from_snapshot(
            &snapshot,
            &decision,
            &intent,
            &request,
            &feature_hash,
            fee_rate,
            self.config.strategy.max_reference_age_ms,
        )?;
        reference_execution_guard.insert_into_metadata(&mut request.metadata)?;
        self.repository
            .insert_strategy_decision(
                self.config.experiment_id,
                &self.config.config_hash,
                &snapshot.market_id,
                &self.config.strategy.strategy_version,
                &decision,
                entry_admission_evidence,
                Some(plan_id),
                "approved",
            )
            .await?;
        let preview_futures = self
            .config
            .paper_stress_previews
            .iter()
            .map(|preview| self.paper_venue.preview_order(&request, preview));
        let primary_request = request.clone();
        let (report, preview_results) = tokio::join!(
            execute_order_plan(
                &self.paper_venue,
                OrderPlan {
                    plan_id,
                    orders: vec![primary_request],
                },
            ),
            futures_util::future::join_all(preview_futures),
        );
        let report = report.context("BTC paper OrderPlan execution failed")?;
        let mut stress_previews = Vec::with_capacity(preview_results.len());
        for (config, result) in self
            .config
            .paper_stress_previews
            .iter()
            .zip(preview_results)
        {
            match result {
                Ok(result) => stress_previews.push(serde_json::json!({
                    "status": "observed",
                    "result": result,
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
        let filled = report
            .orders
            .first()
            .map(|order| order.state == OrderState::Filled)
            .unwrap_or(false);
        let execution_reject_reason = report
            .orders
            .first()
            .and_then(|order| order.request.metadata.get("reject_reason"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        self.repository
            .update_strategy_decision_execution(
                decision.decision_id,
                decision.evaluated_at,
                if filled { "filled" } else { "rejected" },
                execution_reject_reason.as_deref(),
                serde_json::json!({
                    "paper_order_plan": {
                        "plan_id": report.plan_id,
                        "orders": report.orders,
                        "fills": report.fills,
                        "reconciliation": report.reconciliation,
                    },
                    "paper_stress_previews": {
                        "telemetry_only": true,
                        "influenced_primary_execution": false,
                        "primary_config_hash": self.config.config_hash,
                        "scenarios": stress_previews,
                    }
                }),
            )
            .await?;
        self.repository
            .increment_experiment_counts(self.config.experiment_id, 1, 1, i64::from(filled))
            .await?;
        if filled {
            self.force_refresh_settlement_and_reconcile().await?;
        }
        Ok(())
    }
}

async fn load_shadow_predictive_regime_state(
    repository: &BtcRepository,
    process_id: Uuid,
    config: &ShadowPredictiveRegimeCircuitBreakerConfig,
    as_of: DateTime<Utc>,
    base_state: Option<ShadowPredictiveRegimeState>,
) -> Result<(
    ShadowPredictiveRegimeState,
    Vec<ShadowPredictiveRegimeTransitionEvidence>,
)> {
    let breaker_config_hash = config.config_hash()?;
    let prior_state = match base_state {
        Some(state) => {
            state.validate(config)?;
            state
        }
        None => match repository
            .load_latest_shadow_predictive_regime_state(
                process_id,
                Some(&breaker_config_hash),
                as_of,
            )
            .await?
        {
            Some(state) => {
                state.validate(config)?;
                state
            }
            None => ShadowPredictiveRegimeState::new(process_id, config)?,
        },
    };
    let candidates = repository
        .load_shadow_predictive_regime_candidates(
            process_id,
            None,
            as_of,
            SHADOW_PREDICTIVE_REGIME_REPLAY_FETCH_CANDIDATES,
        )
        .await?;
    reconcile_shadow_predictive_regime_state(process_id, config, Some(prior_state), &candidates)
}

fn reconcile_shadow_predictive_regime_state(
    process_id: Uuid,
    config: &ShadowPredictiveRegimeCircuitBreakerConfig,
    prior_state: Option<ShadowPredictiveRegimeState>,
    candidates: &[ShadowPredictiveRegimeCandidate],
) -> Result<(
    ShadowPredictiveRegimeState,
    Vec<ShadowPredictiveRegimeTransitionEvidence>,
)> {
    if candidates.len()
        > usize::try_from(SHADOW_PREDICTIVE_REGIME_MAX_REPLAY_CANDIDATES).unwrap_or(usize::MAX)
    {
        anyhow::bail!(
            "shadow predictive-regime canonical replay exceeded its bounded {}-candidate history; complete history is required to reconcile late causal evidence",
            SHADOW_PREDICTIVE_REGIME_MAX_REPLAY_CANDIDATES
        );
    }

    if let Some(state) = prior_state.as_ref() {
        state.validate(config)?;
        if state.process_id != process_id {
            anyhow::bail!(
                "shadow predictive-regime prior state process_id does not match the requested process"
            );
        }
    }

    let mut rebuilt_state = ShadowPredictiveRegimeState::new(process_id, config)?;
    let mut replay_transitions = Vec::new();
    for candidate in candidates {
        if let Some(transition) = rebuilt_state.apply_candidate(config, candidate)? {
            replay_transitions.push((transition, candidate.clone(), rebuilt_state.clone()));
        }
    }

    let Some(prior_state) = prior_state else {
        return Ok((rebuilt_state, replay_transitions));
    };
    let prior_count = usize::try_from(prior_state.resolved_markets_observed)
        .context("shadow predictive-regime prior observation count does not fit in memory")?;
    let retained_count = prior_state.rolling_candidates.len();
    let prior_is_canonical_prefix = retained_count <= prior_count
        && prior_count <= candidates.len()
        && prior_state.rolling_candidates.as_slice()
            == &candidates[prior_count - retained_count..prior_count];

    if prior_is_canonical_prefix {
        let mut resumed_state = prior_state.clone();
        let mut incremental_transitions = Vec::new();
        for candidate in &candidates[prior_count..] {
            if let Some(transition) = resumed_state.apply_candidate(config, candidate)? {
                incremental_transitions.push((
                    transition,
                    candidate.clone(),
                    resumed_state.clone(),
                ));
            }
        }
        if resumed_state == rebuilt_state {
            return Ok((rebuilt_state, incremental_transitions));
        }
    }

    if prior_state.degraded == rebuilt_state.degraded {
        return Ok((rebuilt_state, Vec::new()));
    }
    let expected_transition = if rebuilt_state.degraded {
        ShadowPredictiveRegimeTransition::DegradationConfirmed
    } else {
        ShadowPredictiveRegimeTransition::RecoveryConfirmed
    };
    let corrective_transition = replay_transitions
        .into_iter()
        .rev()
        .find(|(transition, _, _)| *transition == expected_transition)
        .context(
            "shadow predictive-regime canonical replay changed terminal state without a matching transition",
        )?;
    Ok((rebuilt_state, vec![corrective_transition]))
}

fn combine_entry_admission_evaluations(
    process_id: Uuid,
    loss_regime: LossRegimeConfidenceFloorEvaluation,
    high_water_mark: Option<DailyRealizedPnlHighWaterMarkEvaluation>,
    shadow_predictive_regime: Option<ShadowPredictiveRegimeEvaluation>,
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

    let evidence = match (high_water_mark, shadow_predictive_regime) {
        (None, None) => serde_json::to_value(loss_regime)?,
        (Some(high_water_mark), None) => {
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
        (high_water_mark, Some(shadow_predictive_regime)) => {
            let mut blocking_policies = Vec::new();
            if loss_deferred {
                blocking_policies.push("loss_regime_confidence_floor");
            }
            if high_water_mark_deferred {
                blocking_policies.push("daily_realized_pnl_high_water_mark");
            }
            let shadow_would_block_policies = if shadow_predictive_regime.would_defer {
                vec!["shadow_predictive_regime_circuit_breaker"]
            } else {
                Vec::new()
            };
            serde_json::json!({
                "evidence_version": "btc_entry_admission_v3",
                "process_id": process_id,
                "disposition": disposition,
                "blocking_policies": blocking_policies,
                "shadow_would_block_policies": shadow_would_block_policies,
                "loss_regime_confidence_floor": loss_regime,
                "daily_realized_pnl_high_water_mark": high_water_mark,
                "shadow_predictive_regime_circuit_breaker": shadow_predictive_regime,
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
    experiment_id: Uuid,
    fee_rate: Decimal,
) -> Result<serde_json::Value> {
    let attribution = strategy
        .attribution()
        .context("BTC paper experiment strategy attribution became invalid")?;
    let mut metadata = serde_json::json!({
        "execution_intent": "entry",
        "strategy": attribution.family,
        "strategy_version": intent.strategy_version,
        "feature_schema_version": intent.feature_schema_version,
        "feature_snapshot_id": intent.feature_snapshot_id,
        "decision_id": decision_id,
        "process_id": intent.process_id,
        "experiment_id": experiment_id,
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
        let BtcStrategyPrediction::DirectionalPrediction { outcome, .. } = prediction else {
            anyhow::bail!("BTC entry order cannot carry a no-prediction result");
        };
        ensure!(
            attribution.family == BTC_MARKET_ANCHORED_DIRECTIONAL_PREDICTION_STRATEGY_FAMILY
                && *outcome == intent.outcome,
            "BTC directional prediction attribution does not match its entry intent"
        );
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

fn paper_capital_reconcile_due(last_started_at: Option<Instant>, now: Instant) -> bool {
    last_started_at.is_none_or(|last_started_at| {
        now.saturating_duration_since(last_started_at) >= PAPER_CAPITAL_RECONCILE_INTERVAL
    })
}

fn snapshot_quality_flags(
    snapshot: &BtcFeatureSnapshot,
    readiness: &super::types::Readiness,
    config: &BtcStrategyConfig,
) -> Vec<String> {
    let mut flags = Vec::new();
    if !readiness.ready {
        flags.push("runtime_not_ready".to_string());
        flags.extend(
            readiness
                .reasons
                .iter()
                .map(|reason| format!("runtime:{reason}")),
        );
    }
    match snapshot.chainlink_age_ms {
        None => flags.push("missing_chainlink".to_string()),
        Some(age) if age < 0 => flags.push("future_chainlink_receipt".to_string()),
        Some(age) if age > config.max_reference_age_ms => flags.push("stale_chainlink".to_string()),
        Some(_) => {}
    }
    match snapshot.binance_age_ms {
        None => flags.push("missing_binance".to_string()),
        Some(age) if age < 0 => flags.push("future_binance_receipt".to_string()),
        Some(age) if age > config.max_reference_age_ms => flags.push("stale_binance".to_string()),
        Some(_) => {}
    }
    if snapshot
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

fn enforce_runtime_readiness(decision: &mut BtcDecision, readiness: &super::types::Readiness) {
    if !readiness.ready {
        decision.action = BtcDecisionAction::NoTrade;
        decision.reject_reason = Some(BtcRejectReason::RuntimeNotReady);
        decision.approved_intent = None;
    }
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

#[async_trait]
impl BtcStrategyRunner for BtcPaperExperimentRunner {
    async fn on_observation(&self, observation: StrategyObservation) -> Result<()> {
        self.observe(observation).await
    }

    async fn shutdown(&self) -> Result<()> {
        BtcPaperExperimentRunner::shutdown(self).await
    }
}

fn build_snapshot(
    process_id: Uuid,
    market: &BtcIntervalMarket,
    observed_at: DateTime<Utc>,
    inputs: &BtcPointInTimeInputs,
    target_size: Decimal,
    feature_schema_version: &str,
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
            observed_at.timestamp_micros()
        )
        .as_bytes(),
    );
    let chainlink_gap_bps = chainlink_open.zip(chainlink).and_then(|(open, current)| {
        ratio_return(current.price, open.price).map(|value| value * Decimal::from(10_000))
    });
    let path_conditioned =
        feature_schema_version == BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_SCHEMA_VERSION;
    let chainlink_anchor_5s = path_conditioned
        .then(|| point_in_time_return_anchor(&inputs.chainlink_history, chainlink, observed_at, 5))
        .flatten();
    let chainlink_anchor_15s = path_conditioned
        .then(|| point_in_time_return_anchor(&inputs.chainlink_history, chainlink, observed_at, 15))
        .flatten();
    let chainlink_anchor_30s = path_conditioned
        .then(|| point_in_time_return_anchor(&inputs.chainlink_history, chainlink, observed_at, 30))
        .flatten();
    let chainlink_return_5s = if path_conditioned {
        chainlink_anchor_5s.map(|anchor| anchor.return_value)
    } else if feature_schema_version == BTC_CHAINLINK_PERSISTENCE_CALIBRATED_FEATURE_SCHEMA_VERSION
    {
        return_over(&inputs.chainlink_history, observed_at, 5)
    } else {
        None
    };
    let chainlink_return_15s = chainlink_anchor_15s.map(|anchor| anchor.return_value);
    let chainlink_return_30s = chainlink_anchor_30s.map(|anchor| anchor.return_value);
    let chainlink_path_5s = chainlink_anchor_5s.and_then(|anchor| {
        chainlink.map(|current| {
            point_in_time_path(&inputs.chainlink_history, anchor.tick, current, observed_at)
        })
    });
    let chainlink_path_30s = chainlink_anchor_30s.and_then(|anchor| {
        chainlink.map(|current| {
            point_in_time_path(&inputs.chainlink_history, anchor.tick, current, observed_at)
        })
    });
    let chainlink_path_efficiency_30s = chainlink_path_30s.as_deref().and_then(path_efficiency);
    let chainlink_path_tick_count_30s = chainlink_path_30s.as_ref().map(Vec::len);
    let chainlink_realized_volatility_5s = chainlink_path_5s
        .as_deref()
        .and_then(path_realized_volatility);
    let chainlink_realized_volatility_30s = chainlink_path_30s
        .as_deref()
        .and_then(path_realized_volatility);
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
        chainlink_return_5s,
        chainlink_return_15s,
        chainlink_return_30s,
        chainlink_path_efficiency_30s,
        chainlink_path_tick_count_30s,
        chainlink_realized_volatility_5s,
        chainlink_realized_volatility_30s,
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
        lineage: BtcFeatureLineage {
            lineage_version: if path_conditioned {
                BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_LINEAGE_VERSION.to_string()
            } else {
                BTC_FEATURE_LINEAGE_VERSION.to_string()
            },
            chainlink_open_tick_id: chainlink_open.map(|tick| tick.tick_id),
            chainlink_tick_id: chainlink.map(|tick| tick.tick_id),
            binance_tick_id: binance.map(|tick| tick.tick_id),
            up_book_checkpoint_id: inputs.up_book.as_ref().map(|book| book.checkpoint_id),
            down_book_checkpoint_id: inputs.down_book.as_ref().map(|book| book.checkpoint_id),
            chainlink_open_source_timestamp: chainlink_open.map(|tick| tick.source_timestamp),
            chainlink_open_received_at: chainlink_open.map(|tick| tick.received_at),
            chainlink_source_timestamp: chainlink.map(|tick| tick.source_timestamp),
            chainlink_received_at: chainlink.map(|tick| tick.received_at),
            chainlink_anchor_15s_tick_id: chainlink_anchor_15s.map(|anchor| anchor.tick.tick_id),
            chainlink_anchor_15s_source_timestamp: chainlink_anchor_15s
                .map(|anchor| anchor.tick.source_timestamp),
            chainlink_anchor_15s_received_at: chainlink_anchor_15s
                .map(|anchor| anchor.tick.received_at),
            chainlink_anchor_15s_ingest_sequence: chainlink_anchor_15s
                .map(|anchor| anchor.tick.ingest_sequence),
            chainlink_anchor_15s_effective_lookback_ms: chainlink_anchor_15s
                .map(|anchor| anchor.effective_lookback_ms),
            chainlink_anchor_30s_tick_id: chainlink_anchor_30s.map(|anchor| anchor.tick.tick_id),
            chainlink_anchor_30s_source_timestamp: chainlink_anchor_30s
                .map(|anchor| anchor.tick.source_timestamp),
            chainlink_anchor_30s_received_at: chainlink_anchor_30s
                .map(|anchor| anchor.tick.received_at),
            chainlink_anchor_30s_ingest_sequence: chainlink_anchor_30s
                .map(|anchor| anchor.tick.ingest_sequence),
            chainlink_anchor_30s_effective_lookback_ms: chainlink_anchor_30s
                .map(|anchor| anchor.effective_lookback_ms),
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
        age_ms: Some((observed_at - checkpoint.received_at).num_milliseconds()),
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

const CHAINLINK_PATH_ANCHOR_TOLERANCE_MS: i64 = 5_000;

#[derive(Debug, Clone, Copy)]
struct PointInTimeReturnAnchor<'a> {
    tick: &'a ReferencePriceTick,
    return_value: Decimal,
    effective_lookback_ms: i64,
}

fn reference_tick_order_key(
    tick: &ReferencePriceTick,
) -> (DateTime<Utc>, DateTime<Utc>, u64, Uuid) {
    (
        tick.source_timestamp,
        tick.received_at,
        tick.ingest_sequence,
        tick.tick_id,
    )
}

fn point_in_time_return_anchor<'a>(
    history: &'a [ReferencePriceTick],
    current: Option<&ReferencePriceTick>,
    observed_at: DateTime<Utc>,
    seconds: i64,
) -> Option<PointInTimeReturnAnchor<'a>> {
    let current = current
        .filter(|tick| tick.source_timestamp <= observed_at && tick.received_at <= observed_at)?;
    let latest_causal = history
        .iter()
        .filter(|tick| tick.source_timestamp <= observed_at && tick.received_at <= observed_at)
        .max_by_key(|tick| reference_tick_order_key(tick))?;
    if latest_causal.tick_id != current.tick_id {
        return None;
    }
    let target = observed_at - chrono::Duration::seconds(seconds);
    let anchor = history
        .iter()
        .filter(|tick| {
            tick.source_timestamp <= target
                && tick.received_at <= observed_at
                && tick.source_timestamp <= observed_at
        })
        .max_by_key(|tick| reference_tick_order_key(tick))?;
    let anchor_staleness_ms = (target - anchor.source_timestamp).num_milliseconds();
    if !(0..=CHAINLINK_PATH_ANCHOR_TOLERANCE_MS).contains(&anchor_staleness_ms)
        || current.source_timestamp <= anchor.source_timestamp
    {
        return None;
    }
    Some(PointInTimeReturnAnchor {
        tick: anchor,
        return_value: ratio_return(current.price, anchor.price)?,
        effective_lookback_ms: (current.source_timestamp - anchor.source_timestamp)
            .num_milliseconds(),
    })
}

fn point_in_time_path(
    history: &[ReferencePriceTick],
    anchor: &ReferencePriceTick,
    current: &ReferencePriceTick,
    observed_at: DateTime<Utc>,
) -> Vec<ReferencePriceTick> {
    let anchor_key = reference_tick_order_key(anchor);
    let current_key = reference_tick_order_key(current);
    let mut path = history
        .iter()
        .filter(|tick| {
            let key = reference_tick_order_key(tick);
            key >= anchor_key
                && key <= current_key
                && tick.source_timestamp <= observed_at
                && tick.received_at <= observed_at
        })
        .cloned()
        .collect::<Vec<_>>();
    if !path.iter().any(|tick| tick.tick_id == anchor.tick_id) {
        path.push(anchor.clone());
    }
    if current.source_timestamp <= observed_at
        && current.received_at <= observed_at
        && !path.iter().any(|tick| tick.tick_id == current.tick_id)
    {
        path.push(current.clone());
    }
    path.sort_by_key(reference_tick_order_key);
    path.dedup_by_key(|tick| tick.tick_id);
    path
}

fn path_efficiency(history: &[ReferencePriceTick]) -> Option<Decimal> {
    let first = history.first()?.price;
    let last = history.last()?.price;
    if first <= Decimal::ZERO || last <= Decimal::ZERO || history.len() < 2 {
        return None;
    }
    let net = (last - first).abs();
    let travelled = history
        .windows(2)
        .map(|pair| (pair[1].price - pair[0].price).abs())
        .sum::<Decimal>();
    Some(if travelled > Decimal::ZERO {
        (net / travelled).clamp(Decimal::ZERO, Decimal::ONE)
    } else {
        Decimal::ZERO
    })
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

fn path_realized_volatility(history: &[ReferencePriceTick]) -> Option<Decimal> {
    if history.len() < 2 {
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

    use crate::btc::{
        admission::{
            DailyRealizedPnlCredit, DailyRealizedPnlHighWaterMarkConfig,
            DailyRealizedPnlHighWaterMarkState, LossRegimeConfidenceFloorConfig,
            LossRegimeConfidenceFloorState, ProposedEntryExposure, ShadowPredictiveRegimeCandidate,
            ShadowPredictiveRegimeCircuitBreakerConfig, ShadowPredictiveRegimeState,
            UnsettledEntryExposure, DAILY_REALIZED_PNL_HIGH_WATER_MARK_SCHEMA_VERSION,
            LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION,
            SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE,
            SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION,
        },
        strategy::{
            ApprovedIntent, BtcDecisionStrategyConfig, BtcDirectionalPredictionConfig,
            BtcStrategyConfig, BtcVolatilityContinuationConfig,
            BTC_CHAINLINK_FAIR_VALUE_STRATEGY_FAMILY,
            BTC_CHAINLINK_PERSISTENCE_CALIBRATED_FEATURE_SCHEMA_VERSION,
            BTC_CHAINLINK_PERSISTENCE_CALIBRATED_PROFILE_ID,
            BTC_CHAINLINK_PERSISTENCE_CALIBRATED_PROFILE_SHA256,
            BTC_CHAINLINK_PERSISTENCE_CALIBRATED_STRATEGY_FAMILY,
            BTC_CHAINLINK_PERSISTENCE_CALIBRATED_STRATEGY_VERSION, BTC_FEATURE_SCHEMA_VERSION,
            BTC_MARKET_ANCHORED_DIRECTIONAL_PREDICTION_STRATEGY_VERSION,
            BTC_MARKET_ANCHORED_FAIR_VALUE_STRATEGY_FAMILY,
            BTC_MARKET_ANCHORED_RESEARCH_PROFILE_ID, BTC_MARKET_ANCHORED_RESEARCH_PROFILE_SHA256,
            BTC_MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION,
            BTC_VOLATILITY_CONTINUATION_STRATEGY_FAMILY,
            BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION,
        },
        types::{BookReadiness, Readiness},
    };

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
    fn order_metadata_attributes_each_strategy_and_only_candidate_profile() {
        let chainlink_config = BtcStrategyConfig::default();
        let chainlink = btc_entry_order_metadata(
            &chainlink_config,
            &metadata_intent(&chainlink_config.strategy_version),
            None,
            Uuid::from_u128(204),
            Uuid::from_u128(205),
            dec!(0.03),
        )
        .unwrap();
        assert_eq!(
            chainlink["strategy"],
            BTC_CHAINLINK_FAIR_VALUE_STRATEGY_FAMILY
        );
        assert!(chainlink.get("profile_id").is_none());
        assert!(chainlink.get("profile_sha256").is_none());
        assert!(chainlink.get("prediction").is_none());

        let continuation_config = BtcStrategyConfig {
            strategy_version: BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION.to_string(),
            volatility_continuation: Some(BtcVolatilityContinuationConfig::default()),
            ..BtcStrategyConfig::default()
        };
        let continuation = btc_entry_order_metadata(
            &continuation_config,
            &metadata_intent(&continuation_config.strategy_version),
            None,
            Uuid::from_u128(204),
            Uuid::from_u128(205),
            dec!(0.03),
        )
        .unwrap();
        assert_eq!(
            continuation["strategy"],
            BTC_VOLATILITY_CONTINUATION_STRATEGY_FAMILY
        );
        assert!(continuation.get("profile_id").is_none());
        assert!(continuation.get("prediction").is_none());

        let candidate_config = BtcStrategyConfig {
            strategy_version: BTC_MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION.to_string(),
            decision_strategy: Some(BtcDecisionStrategyConfig::MarketAnchoredFairValue {
                profile_id: BTC_MARKET_ANCHORED_RESEARCH_PROFILE_ID.to_string(),
                profile_sha256: BTC_MARKET_ANCHORED_RESEARCH_PROFILE_SHA256.to_string(),
            }),
            ..BtcStrategyConfig::default()
        };
        let candidate = btc_entry_order_metadata(
            &candidate_config,
            &metadata_intent(&candidate_config.strategy_version),
            None,
            Uuid::from_u128(204),
            Uuid::from_u128(205),
            dec!(0.03),
        )
        .unwrap();
        assert_eq!(
            candidate["strategy"],
            BTC_MARKET_ANCHORED_FAIR_VALUE_STRATEGY_FAMILY
        );
        assert_eq!(
            candidate["profile_id"],
            BTC_MARKET_ANCHORED_RESEARCH_PROFILE_ID
        );
        assert_eq!(
            candidate["profile_sha256"],
            BTC_MARKET_ANCHORED_RESEARCH_PROFILE_SHA256
        );
        assert!(candidate.get("prediction").is_none());
    }

    #[test]
    fn order_metadata_attributes_chainlink_persistence_profile() {
        let config = BtcStrategyConfig {
            strategy_version: BTC_CHAINLINK_PERSISTENCE_CALIBRATED_STRATEGY_VERSION.to_string(),
            feature_schema_version: BTC_CHAINLINK_PERSISTENCE_CALIBRATED_FEATURE_SCHEMA_VERSION
                .to_string(),
            decision_strategy: Some(
                BtcDecisionStrategyConfig::ChainlinkPersistenceCalibratedFairValue {
                    profile_id: BTC_CHAINLINK_PERSISTENCE_CALIBRATED_PROFILE_ID.to_string(),
                    profile_sha256: BTC_CHAINLINK_PERSISTENCE_CALIBRATED_PROFILE_SHA256.to_string(),
                },
            ),
            ..BtcStrategyConfig::default()
        };
        let metadata = btc_entry_order_metadata(
            &config,
            &metadata_intent(&config.strategy_version),
            None,
            Uuid::from_u128(204),
            Uuid::from_u128(205),
            dec!(0.03),
        )
        .unwrap();

        assert_eq!(
            metadata["strategy"],
            BTC_CHAINLINK_PERSISTENCE_CALIBRATED_STRATEGY_FAMILY
        );
        assert_eq!(
            metadata["strategy_version"],
            BTC_CHAINLINK_PERSISTENCE_CALIBRATED_STRATEGY_VERSION
        );
        assert_eq!(
            metadata["profile_id"],
            BTC_CHAINLINK_PERSISTENCE_CALIBRATED_PROFILE_ID
        );
        assert_eq!(
            metadata["profile_sha256"],
            BTC_CHAINLINK_PERSISTENCE_CALIBRATED_PROFILE_SHA256
        );
    }

    #[test]
    fn order_metadata_includes_directional_prediction_evidence() {
        let config = BtcStrategyConfig {
            strategy_version: BTC_MARKET_ANCHORED_DIRECTIONAL_PREDICTION_STRATEGY_VERSION
                .to_string(),
            decision_strategy: Some(
                BtcDecisionStrategyConfig::MarketAnchoredDirectionalPrediction {
                    profile_id: BTC_MARKET_ANCHORED_RESEARCH_PROFILE_ID.to_string(),
                    profile_sha256: BTC_MARKET_ANCHORED_RESEARCH_PROFILE_SHA256.to_string(),
                    config: BtcDirectionalPredictionConfig::default(),
                },
            ),
            ..BtcStrategyConfig::default()
        };
        let prediction = BtcStrategyPrediction::DirectionalPrediction {
            outcome: BtcOutcome::Up,
            probability: dec!(0.82),
            conservative_probability: dec!(0.78),
            minimum_conservative_probability: dec!(0.75),
            probability_uncertainty: dec!(0.04),
            executable_price: Some(dec!(0.72)),
            direct_taker_fee_per_share: Some(dec!(0.01)),
            direct_net_edge_per_share: Some(dec!(0.09)),
        };

        let metadata = btc_entry_order_metadata(
            &config,
            &metadata_intent(&config.strategy_version),
            Some(&prediction),
            Uuid::from_u128(204),
            Uuid::from_u128(205),
            dec!(0.03),
        )
        .unwrap();

        assert_eq!(
            metadata["prediction"],
            serde_json::json!({
                "status": "directional_prediction",
                "outcome": "up",
                "probability": "0.82",
                "conservative_probability": "0.78",
                "minimum_conservative_probability": "0.75",
                "probability_uncertainty": "0.04",
                "executable_price": "0.72",
                "direct_taker_fee_per_share": "0.01",
                "direct_net_edge_per_share": "0.09",
            })
        );
        assert_eq!(
            metadata["strategy"],
            BTC_MARKET_ANCHORED_DIRECTIONAL_PREDICTION_STRATEGY_FAMILY
        );
        assert_eq!(
            metadata["strategy_version"],
            BTC_MARKET_ANCHORED_DIRECTIONAL_PREDICTION_STRATEGY_VERSION
        );
        assert_eq!(
            metadata["profile_id"],
            BTC_MARKET_ANCHORED_RESEARCH_PROFILE_ID
        );
    }

    #[test]
    fn paper_capital_reconciliation_is_due_at_five_second_boundaries() {
        let started_at = Instant::now();
        assert!(paper_capital_reconcile_due(None, started_at));
        assert!(!paper_capital_reconcile_due(
            Some(started_at),
            started_at + TokioDuration::from_millis(4_999)
        ));
        assert!(paper_capital_reconcile_due(
            Some(started_at),
            started_at + TokioDuration::from_secs(5)
        ));
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
            combine_entry_admission_evaluations(Uuid::from_u128(300), loss, None, None).unwrap();

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
            combine_entry_admission_evaluations(process_id, loss, Some(high_water_mark), None)
                .unwrap();

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
    fn shadow_would_defer_is_evidence_only_and_never_changes_admission() {
        let floor = LossRegimeConfidenceFloorConfig {
            schema_version: LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION.to_string(),
            activation_consecutive_candidate_losses: 2,
            min_conservative_probability: dec!(0.50),
            release_consecutive_candidate_wins: 1,
        };
        let loss = LossRegimeConfidenceFloorState::default()
            .evaluate(&floor, dec!(0.60))
            .unwrap();
        let shadow_config = ShadowPredictiveRegimeCircuitBreakerConfig {
            schema_version: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION.to_string(),
            mode: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE.to_string(),
            rolling_resolved_market_window: 20,
            minimum_resolved_markets: 20,
            degradation_brier_score_threshold: dec!(0.23),
            degradation_overconfidence_gap_threshold: dec!(0.12),
            degradation_confirmation_markets: 2,
            recovery_brier_score_threshold: dec!(0.21),
            recovery_overconfidence_gap_threshold: dec!(0.05),
            recovery_confirmation_markets: 2,
        };
        let process_id = Uuid::from_u128(304);
        let start = Utc.with_ymd_and_hms(2026, 7, 20, 0, 0, 0).unwrap();
        let candidates = (0..21)
            .map(|index| {
                let label_available_at = start + chrono::Duration::minutes(index * 5);
                ShadowPredictiveRegimeCandidate {
                    market_id: format!("market-{index}"),
                    decision_id: Uuid::from_u128(u128::try_from(index + 1).unwrap()),
                    decision_outcome: BtcOutcome::Up,
                    resolved_outcome: BtcOutcome::Down,
                    selected_point_probability: dec!(0.90),
                    decision_at: label_available_at - chrono::Duration::minutes(4),
                    label_available_at,
                }
            })
            .collect::<Vec<_>>();
        let state =
            ShadowPredictiveRegimeState::from_candidates(process_id, &shadow_config, &candidates)
                .unwrap();
        let shadow = state
            .evaluate(
                &shadow_config,
                candidates.last().unwrap().label_available_at,
            )
            .unwrap();
        assert!(shadow.would_defer);
        assert_eq!(shadow.disposition, AdmissionDisposition::Allow);

        let combined =
            combine_entry_admission_evaluations(process_id, loss, None, Some(shadow)).unwrap();

        assert_eq!(combined.disposition, AdmissionDisposition::Allow);
        assert_eq!(
            combined.evidence["evidence_version"],
            "btc_entry_admission_v3"
        );
        assert_eq!(
            combined.evidence["blocking_policies"],
            serde_json::json!([])
        );
        assert_eq!(
            combined.evidence["shadow_would_block_policies"],
            serde_json::json!(["shadow_predictive_regime_circuit_breaker"])
        );
        assert_eq!(
            combined.evidence["shadow_predictive_regime_circuit_breaker"]["disposition"],
            "allow"
        );
    }

    #[test]
    fn shadow_canonical_replay_incorporates_late_candidate_before_saved_cursor() {
        let config = ShadowPredictiveRegimeCircuitBreakerConfig {
            schema_version: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION.to_string(),
            mode: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE.to_string(),
            rolling_resolved_market_window: 20,
            minimum_resolved_markets: 20,
            degradation_brier_score_threshold: dec!(0.23),
            degradation_overconfidence_gap_threshold: dec!(0.12),
            degradation_confirmation_markets: 2,
            recovery_brier_score_threshold: dec!(0.21),
            recovery_overconfidence_gap_threshold: dec!(0.05),
            recovery_confirmation_markets: 2,
        };
        let process_id = Uuid::from_u128(307);
        let start = Utc.with_ymd_and_hms(2026, 7, 20, 0, 0, 0).unwrap();
        let candidate = |index: i64| {
            let label_available_at = start + chrono::Duration::minutes(index * 5);
            ShadowPredictiveRegimeCandidate {
                market_id: format!("replay-market-{index}"),
                decision_id: Uuid::from_u128(u128::try_from(index + 1).unwrap()),
                decision_outcome: BtcOutcome::Up,
                resolved_outcome: BtcOutcome::Down,
                selected_point_probability: dec!(0.50),
                decision_at: label_available_at - chrono::Duration::minutes(4),
                label_available_at,
            }
        };
        let original = (0..20).map(candidate).collect::<Vec<_>>();
        let prior_state =
            ShadowPredictiveRegimeState::from_candidates(process_id, &config, &original).unwrap();
        assert!(!prior_state.degraded);
        assert_eq!(prior_state.consecutive_degradation_markets, 1);

        let late_label_available_at = start + chrono::Duration::minutes(92);
        let late_candidate = ShadowPredictiveRegimeCandidate {
            market_id: "replay-market-late".to_string(),
            decision_id: Uuid::from_u128(10_000),
            decision_outcome: BtcOutcome::Up,
            resolved_outcome: BtcOutcome::Down,
            selected_point_probability: dec!(0.50),
            decision_at: late_label_available_at - chrono::Duration::minutes(4),
            label_available_at: late_label_available_at,
        };
        let mut corrected_history = original[..19].to_vec();
        corrected_history.push(late_candidate.clone());
        corrected_history.push(original[19].clone());

        let expected =
            ShadowPredictiveRegimeState::from_candidates(process_id, &config, &corrected_history)
                .unwrap();
        let (reconciled, transitions) = reconcile_shadow_predictive_regime_state(
            process_id,
            &config,
            Some(prior_state),
            &corrected_history,
        )
        .unwrap();

        assert_eq!(reconciled, expected);
        assert_eq!(reconciled.resolved_markets_observed, 21);
        assert!(reconciled.degraded);
        assert!(reconciled
            .rolling_candidates
            .iter()
            .any(|candidate| candidate.decision_id == late_candidate.decision_id));
        assert_eq!(transitions.len(), 1);
        assert_eq!(
            transitions[0].0,
            ShadowPredictiveRegimeTransition::DegradationConfirmed
        );

        let transition_event_type = "btc_shadow_predictive_regime_degradation_confirmed";
        let transition_state_hash = transitions[0].2.evidence_sha256(&config).unwrap();
        let transition_event_id = shadow_predictive_regime_transition_event_id(
            process_id,
            &config.config_hash().unwrap(),
            transition_event_type,
            &transition_state_hash,
        );
        let empty_restart_state = ShadowPredictiveRegimeState::new(process_id, &config).unwrap();
        let (_, restart_transitions) = reconcile_shadow_predictive_regime_state(
            process_id,
            &config,
            Some(empty_restart_state),
            &corrected_history,
        )
        .unwrap();
        assert_eq!(restart_transitions.len(), 1);
        let restart_state_hash = restart_transitions[0].2.evidence_sha256(&config).unwrap();
        assert_eq!(
            shadow_predictive_regime_transition_event_id(
                process_id,
                &config.config_hash().unwrap(),
                transition_event_type,
                &restart_state_hash,
            ),
            transition_event_id
        );

        let (repeated, repeated_transitions) = reconcile_shadow_predictive_regime_state(
            process_id,
            &config,
            Some(reconciled.clone()),
            &corrected_history,
        )
        .unwrap();
        assert_eq!(repeated, reconciled);
        assert!(repeated_transitions.is_empty());
    }

    #[test]
    fn shadow_canonical_replay_fails_when_complete_history_exceeds_bound() {
        let config = ShadowPredictiveRegimeCircuitBreakerConfig {
            schema_version: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION.to_string(),
            mode: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE.to_string(),
            rolling_resolved_market_window: 20,
            minimum_resolved_markets: 20,
            degradation_brier_score_threshold: dec!(0.23),
            degradation_overconfidence_gap_threshold: dec!(0.12),
            degradation_confirmation_markets: 2,
            recovery_brier_score_threshold: dec!(0.21),
            recovery_overconfidence_gap_threshold: dec!(0.05),
            recovery_confirmation_markets: 2,
        };
        let process_id = Uuid::from_u128(308);
        let start = Utc.with_ymd_and_hms(2026, 7, 20, 0, 0, 0).unwrap();
        let mut candidates = (0..SHADOW_PREDICTIVE_REGIME_MAX_REPLAY_CANDIDATES)
            .map(|index| {
                let index = i64::from(index);
                let label_available_at = start + chrono::Duration::minutes(index * 5);
                ShadowPredictiveRegimeCandidate {
                    market_id: format!("bounded-replay-market-{index}"),
                    decision_id: Uuid::from_u128(u128::try_from(index + 1).unwrap()),
                    decision_outcome: BtcOutcome::Up,
                    resolved_outcome: BtcOutcome::Down,
                    selected_point_probability: dec!(0.50),
                    decision_at: label_available_at - chrono::Duration::minutes(4),
                    label_available_at,
                }
            })
            .collect::<Vec<_>>();

        let (at_bound, _) =
            reconcile_shadow_predictive_regime_state(process_id, &config, None, &candidates)
                .unwrap();
        assert_eq!(
            at_bound.resolved_markets_observed,
            u64::from(SHADOW_PREDICTIVE_REGIME_MAX_REPLAY_CANDIDATES)
        );

        let index = i64::from(SHADOW_PREDICTIVE_REGIME_MAX_REPLAY_CANDIDATES);
        let label_available_at = start + chrono::Duration::minutes(index * 5);
        candidates.push(ShadowPredictiveRegimeCandidate {
            market_id: format!("bounded-replay-market-{index}"),
            decision_id: Uuid::from_u128(u128::try_from(index + 1).unwrap()),
            decision_outcome: BtcOutcome::Up,
            resolved_outcome: BtcOutcome::Down,
            selected_point_probability: dec!(0.50),
            decision_at: label_available_at - chrono::Duration::minutes(4),
            label_available_at,
        });
        let error =
            reconcile_shadow_predictive_regime_state(process_id, &config, None, &candidates)
                .unwrap_err();
        assert!(error.to_string().contains("complete history is required"));
    }

    #[test]
    fn shadow_refresh_pending_tracks_hydration_and_market_identity() {
        let config = ShadowPredictiveRegimeCircuitBreakerConfig {
            schema_version: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION.to_string(),
            mode: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE.to_string(),
            rolling_resolved_market_window: 20,
            minimum_resolved_markets: 20,
            degradation_brier_score_threshold: dec!(0.23),
            degradation_overconfidence_gap_threshold: dec!(0.12),
            degradation_confirmation_markets: 2,
            recovery_brier_score_threshold: dec!(0.21),
            recovery_overconfidence_gap_threshold: dec!(0.05),
            recovery_confirmation_markets: 2,
        };
        let mut runtime = ShadowPredictiveRegimeAdmissionRuntime {
            state: ShadowPredictiveRegimeState::new(Uuid::from_u128(305), &config).unwrap(),
            state_hydrated: false,
            evaluated_market_id: None,
            attempted_market_id: None,
            refresh_in_progress: false,
            telemetry_error: None,
        };

        assert!(shadow_predictive_regime_refresh_pending(
            &runtime, "market-a"
        ));
        runtime.state_hydrated = true;
        runtime.evaluated_market_id = Some("market-a".to_string());
        assert!(!shadow_predictive_regime_refresh_pending(
            &runtime, "market-a"
        ));
        assert!(shadow_predictive_regime_refresh_pending(
            &runtime, "market-b"
        ));
        runtime.refresh_in_progress = true;
        assert!(shadow_predictive_regime_refresh_pending(
            &runtime, "market-a"
        ));
    }

    #[test]
    fn unavailable_shadow_cache_remains_visible_and_fail_open() {
        let config = ShadowPredictiveRegimeCircuitBreakerConfig {
            schema_version: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION.to_string(),
            mode: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE.to_string(),
            rolling_resolved_market_window: 20,
            minimum_resolved_markets: 20,
            degradation_brier_score_threshold: dec!(0.23),
            degradation_overconfidence_gap_threshold: dec!(0.12),
            degradation_confirmation_markets: 2,
            recovery_brier_score_threshold: dec!(0.21),
            recovery_overconfidence_gap_threshold: dec!(0.05),
            recovery_confirmation_markets: 2,
        };
        let evaluation = unavailable_shadow_predictive_regime_evaluation(
            Uuid::from_u128(306),
            &config,
            Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap(),
            "cache busy".to_string(),
        )
        .unwrap();

        assert_eq!(evaluation.disposition, AdmissionDisposition::Allow);
        assert!(!evaluation.would_defer);
        assert!(evaluation.refresh_pending);
        assert!(!evaluation.state_checkpoint_eligible);
        assert_eq!(evaluation.telemetry_error.as_deref(), Some("cache busy"));
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
    fn path_efficiency_is_exact_bounded_and_flat_safe() {
        let start = Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap();
        let ticks = |prices: &[Decimal]| {
            prices
                .iter()
                .enumerate()
                .map(|(index, price)| ReferencePriceTick {
                    tick_id: Uuid::from_u128(350 + index as u128),
                    dedup_key: index.to_string(),
                    source: super::super::types::ReferencePriceSource::RtdsChainlink,
                    symbol: "btcusd".to_string(),
                    price: *price,
                    source_timestamp: start + chrono::Duration::seconds(index as i64),
                    envelope_timestamp: None,
                    received_at: start + chrono::Duration::seconds(index as i64),
                    connection_id: Uuid::from_u128(360),
                    ingest_sequence: index as u64,
                    source_event_id: None,
                    raw_payload: serde_json::json!({}),
                })
                .collect::<Vec<_>>()
        };

        let moving = ticks(&[dec!(100), dec!(102), dec!(101), dec!(103)]);
        assert_eq!(path_efficiency(&moving), Some(dec!(0.6)));
        let flat = ticks(&[dec!(100), dec!(100)]);
        assert_eq!(path_efficiency(&flat), Some(Decimal::ZERO));
        assert!(path_realized_volatility(&moving).unwrap() > Decimal::ZERO);
    }

    #[test]
    fn chainlink_path_features_are_additive_and_schema_scoped() {
        let window_start = Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap();
        let observed_at = window_start + chrono::Duration::seconds(180);
        let tick = |index: u128,
                    source: super::super::types::ReferencePriceSource,
                    price: Decimal,
                    at: DateTime<Utc>| ReferencePriceTick {
            tick_id: Uuid::from_u128(index),
            dedup_key: index.to_string(),
            source,
            symbol: "btcusd".to_string(),
            price,
            source_timestamp: at,
            envelope_timestamp: None,
            received_at: at,
            connection_id: Uuid::from_u128(400),
            ingest_sequence: index as u64,
            source_event_id: None,
            raw_payload: serde_json::json!({}),
        };
        let chainlink_open = tick(
            401,
            super::super::types::ReferencePriceSource::RtdsChainlink,
            dec!(100),
            window_start,
        );
        let chainlink_prior = tick(
            402,
            super::super::types::ReferencePriceSource::RtdsChainlink,
            dec!(100),
            observed_at - chrono::Duration::seconds(6),
        );
        let chainlink_15s_anchor = tick(
            408,
            super::super::types::ReferencePriceSource::RtdsChainlink,
            dec!(99.8),
            observed_at - chrono::Duration::seconds(16),
        );
        let chainlink_30s_anchor = tick(
            409,
            super::super::types::ReferencePriceSource::RtdsChainlink,
            dec!(99.5),
            observed_at - chrono::Duration::seconds(31),
        );
        let chainlink_mid_1 = tick(
            410,
            super::super::types::ReferencePriceSource::RtdsChainlink,
            dec!(100.5),
            observed_at - chrono::Duration::seconds(4),
        );
        let chainlink_mid_2 = tick(
            411,
            super::super::types::ReferencePriceSource::RtdsChainlink,
            dec!(100.2),
            observed_at - chrono::Duration::seconds(2),
        );
        let chainlink_current = tick(
            403,
            super::super::types::ReferencePriceSource::RtdsChainlink,
            dec!(101),
            observed_at,
        );
        let binance_prior = tick(
            404,
            super::super::types::ReferencePriceSource::DirectBinance,
            dec!(100),
            observed_at - chrono::Duration::seconds(31),
        );
        let binance_current = tick(
            405,
            super::super::types::ReferencePriceSource::DirectBinance,
            dec!(101),
            observed_at,
        );
        let inputs = BtcPointInTimeInputs {
            chainlink_open: Some(chainlink_open),
            chainlink_current: Some(chainlink_current.clone()),
            chainlink_history: vec![
                chainlink_30s_anchor,
                chainlink_15s_anchor,
                chainlink_prior,
                chainlink_mid_1,
                chainlink_mid_2,
                chainlink_current,
            ],
            binance_history: vec![binance_prior, binance_current],
            up_book: None,
            down_book: None,
            fee_rate: None,
            fee_observed_at: None,
        };
        let market = BtcIntervalMarket {
            event_id: "event".to_string(),
            event_slug: "btc-updown-5m-1784548800".to_string(),
            series_slug: "btc-up-or-down-5m".to_string(),
            market_id: "market".to_string(),
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

        let v2 = build_snapshot(
            Uuid::from_u128(406),
            &market,
            observed_at,
            &inputs,
            dec!(5),
            super::super::strategy::BTC_FEATURE_SCHEMA_VERSION,
        );
        assert_eq!(v2.chainlink_return_5s, None);
        assert!(serde_json::to_value(&v2)
            .unwrap()
            .get("chainlink_return_5s")
            .is_none());

        let v3 = build_snapshot(
            Uuid::from_u128(407),
            &market,
            observed_at,
            &inputs,
            dec!(5),
            BTC_CHAINLINK_PERSISTENCE_CALIBRATED_FEATURE_SCHEMA_VERSION,
        );
        assert_eq!(v3.chainlink_return_5s, Some(dec!(0.01)));
        assert_eq!(
            v3.feature_schema_version,
            BTC_CHAINLINK_PERSISTENCE_CALIBRATED_FEATURE_SCHEMA_VERSION
        );
        assert!(serde_json::to_value(&v3)
            .unwrap()
            .get("chainlink_return_15s")
            .is_none());

        let v4 = build_snapshot(
            Uuid::from_u128(412),
            &market,
            observed_at,
            &inputs,
            dec!(5),
            BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_SCHEMA_VERSION,
        );
        let v4_repeat = build_snapshot(
            Uuid::from_u128(412),
            &market,
            observed_at,
            &inputs,
            dec!(5),
            BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_SCHEMA_VERSION,
        );
        assert_eq!(v4, v4_repeat);
        assert_eq!(v4.process_id, Uuid::from_u128(412));
        assert_eq!(v4.chainlink_return_5s, Some(dec!(0.01)));
        assert_eq!(
            v4.chainlink_return_15s,
            Some(dec!(101) / dec!(99.8) - Decimal::ONE)
        );
        assert_eq!(
            v4.chainlink_return_30s,
            Some(dec!(101) / dec!(99.5) - Decimal::ONE)
        );
        assert_eq!(
            v4.lineage.lineage_version,
            BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_LINEAGE_VERSION
        );
        assert_eq!(
            v4.lineage.chainlink_anchor_15s_tick_id,
            Some(Uuid::from_u128(408))
        );
        assert_eq!(v4.lineage.chainlink_anchor_15s_ingest_sequence, Some(408));
        assert_eq!(
            v4.lineage.chainlink_anchor_15s_effective_lookback_ms,
            Some(16_000)
        );
        assert_eq!(
            v4.lineage.chainlink_anchor_30s_tick_id,
            Some(Uuid::from_u128(409))
        );
        assert_eq!(v4.lineage.chainlink_anchor_30s_ingest_sequence, Some(409));
        assert_eq!(
            v4.lineage.chainlink_anchor_30s_effective_lookback_ms,
            Some(31_000)
        );
        assert_eq!(v4.chainlink_path_tick_count_30s, Some(6));
        assert!(v4.chainlink_path_efficiency_30s > Some(Decimal::ZERO));
        assert!(v4.chainlink_path_efficiency_30s <= Some(Decimal::ONE));
        assert!(v4.chainlink_realized_volatility_5s > Some(Decimal::ZERO));
        assert!(v4.chainlink_realized_volatility_30s > Some(Decimal::ZERO));
    }

    #[test]
    fn path_anchor_excludes_future_received_ticks_and_rejects_stale_history() {
        let observed_at = Utc.with_ymd_and_hms(2026, 7, 20, 12, 3, 0).unwrap();
        let tick = |index: u128, price: Decimal, source_offset: i64, received_offset: i64| {
            ReferencePriceTick {
                tick_id: Uuid::from_u128(index),
                dedup_key: index.to_string(),
                source: super::super::types::ReferencePriceSource::RtdsChainlink,
                symbol: "btcusd".to_string(),
                price,
                source_timestamp: observed_at + chrono::Duration::seconds(source_offset),
                envelope_timestamp: None,
                received_at: observed_at + chrono::Duration::seconds(received_offset),
                connection_id: Uuid::from_u128(420),
                ingest_sequence: index as u64,
                source_event_id: None,
                raw_payload: serde_json::json!({}),
            }
        };
        let causal_anchor = tick(421, dec!(100), -31, -31);
        let future_received_anchor = tick(422, dec!(90), -30, 1);
        let current = tick(423, dec!(101), 0, 0);
        let history = vec![causal_anchor, future_received_anchor, current.clone()];

        let selected = point_in_time_return_anchor(&history, Some(&current), observed_at, 30)
            .expect("causal anchor inside tolerance");
        assert_eq!(selected.tick.tick_id, Uuid::from_u128(421));
        assert_eq!(selected.effective_lookback_ms, 31_000);

        let stale = vec![tick(424, dec!(100), -36, -36), current.clone()];
        assert!(point_in_time_return_anchor(&stale, Some(&current), observed_at, 30).is_none());

        let future_current = tick(425, dec!(101), 0, 1);
        assert!(
            point_in_time_return_anchor(&history, Some(&future_current), observed_at, 30).is_none()
        );
    }

    #[test]
    fn path_begins_at_the_exact_selected_anchor_in_full_tick_order() {
        let observed_at = Utc.with_ymd_and_hms(2026, 7, 20, 12, 3, 0).unwrap();
        let source_at = observed_at - chrono::Duration::seconds(30);
        let tick =
            |index: u128, price: Decimal, source_timestamp, ingest_sequence| ReferencePriceTick {
                tick_id: Uuid::from_u128(index),
                dedup_key: index.to_string(),
                source: super::super::types::ReferencePriceSource::RtdsChainlink,
                symbol: "btcusd".to_string(),
                price,
                source_timestamp,
                envelope_timestamp: None,
                received_at: source_timestamp,
                connection_id: Uuid::from_u128(430),
                ingest_sequence,
                source_event_id: None,
                raw_payload: serde_json::json!({}),
            };
        let superseded_same_time = tick(431, dec!(80), source_at, 431);
        let selected_anchor = tick(432, dec!(100), source_at, 432);
        let current = tick(433, dec!(101), observed_at, 433);
        let history = vec![
            superseded_same_time,
            selected_anchor.clone(),
            current.clone(),
        ];

        let selected = point_in_time_return_anchor(&history, Some(&current), observed_at, 30)
            .expect("latest full-order target tick is selected");
        assert_eq!(selected.tick.tick_id, selected_anchor.tick_id);
        assert_eq!(selected.return_value, dec!(0.01));

        let path = point_in_time_path(&history, selected.tick, &current, observed_at);
        assert_eq!(
            path.iter().map(|tick| tick.tick_id).collect::<Vec<_>>(),
            vec![selected_anchor.tick_id, current.tick_id]
        );
        assert_eq!(path_efficiency(&path), Some(Decimal::ONE));
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
        enforce_runtime_readiness(&mut decision, &Readiness::default());
        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(
            decision.reject_reason,
            Some(BtcRejectReason::RuntimeNotReady)
        );
        assert!(decision.approved_intent.is_none());
        assert_eq!(decision.prediction, Some(prediction));
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
