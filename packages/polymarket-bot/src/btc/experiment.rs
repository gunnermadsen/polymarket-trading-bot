use std::{collections::HashSet, sync::Arc};

use anyhow::{Context, Result};
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
        AdmissionDisposition, BtcEntryAdmissionConfig, LossRegimeConfidenceFloorEvaluation,
        LossRegimeConfidenceFloorState, LossRegimeConfidenceFloorTransition,
    },
    paper::{PaperPreviewConfig, PaperVenue, PAPER_DYNAMIC_FEE_RATE_METADATA_KEY},
    repository::{BtcPointInTimeInputs, BtcRepository},
    runtime::{BtcStrategyRunner, StrategyObservation},
    strategy::{
        BtcDecision, BtcDecisionAction, BtcFeatureLineage, BtcFeatureSnapshot,
        BtcInputWindowLineage, BtcOutcomeBookFeatures, BtcRejectReason, BtcStrategyConfig,
        DeterministicBtcStrategy, BTC_FEATURE_LINEAGE_VERSION,
    },
    types::{
        BtcIntervalMarket, BtcOutcome, FeedIntegrityStatus, OrderbookCheckpoint, ReferencePriceTick,
    },
};

const PAPER_CAPITAL_RECONCILE_INTERVAL: TokioDuration = TokioDuration::from_secs(5);

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

pub struct BtcPaperExperimentRunner {
    repository: BtcRepository,
    store: Store,
    paper_venue: PaperVenue,
    config: BtcPaperExperimentConfig,
    initialized: OnceCell<()>,
    loss_regime_admission: Mutex<Option<LossRegimeAdmissionRuntime>>,
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
        let floor = &entry_admission.loss_regime_confidence_floor;
        let candidates = self
            .repository
            .load_resolved_loss_regime_candidates(
                self.config.process_id,
                &self.config.config_hash,
                Utc::now(),
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

    async fn evaluate_entry_admission(
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
        self.refresh_settlement_and_reconcile_if_due().await?;

        let Some(market) = observation.state.current_market.as_ref() else {
            return Ok(());
        };
        let observed_at = Utc::now();
        let clob_connection_id = observation_clob_connection_id(market, &observation.readiness);
        let inputs = self
            .repository
            .load_point_in_time_inputs(
                market,
                observed_at,
                chrono::Duration::milliseconds(self.config.strategy.max_chainlink_open_delay_ms),
                clob_connection_id,
            )
            .await?;
        let snapshot = build_snapshot(
            self.config.process_id,
            market,
            observed_at,
            &inputs,
            self.config.strategy.target_size,
        );
        let mut decision = DeterministicBtcStrategy::evaluate(&self.config.strategy, &snapshot);
        enforce_runtime_readiness(&mut decision, &observation.readiness);
        if decision.approved_intent.is_some()
            && self
                .repository
                .experiment_has_entry(self.config.experiment_id, &snapshot.market_id)
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
            .evaluate_entry_admission(&decision, observed_at)
            .await?;
        let entry_admission_evidence = entry_admission
            .as_ref()
            .map(serde_json::to_value)
            .transpose()?;
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
                    entry_admission_evidence.as_ref(),
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
        let order_metadata = btc_entry_order_metadata(
            &self.config.strategy,
            &intent,
            decision.decision_id,
            self.config.experiment_id,
            snapshot.fee_rate.unwrap_or_default(),
        )?;
        let request = OrderRequest {
            client_order_id: Uuid::new_v5(
                &Uuid::NAMESPACE_URL,
                format!("btc-paper-order:{}", intent.intent_id).as_bytes(),
            ),
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
        self.repository
            .insert_strategy_decision(
                self.config.experiment_id,
                &self.config.config_hash,
                &snapshot.market_id,
                &self.config.strategy.strategy_version,
                &decision,
                entry_admission_evidence.as_ref(),
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

fn btc_entry_order_metadata(
    strategy: &BtcStrategyConfig,
    intent: &super::strategy::ApprovedIntent,
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
        feature_schema_version: super::strategy::BTC_FEATURE_SCHEMA_VERSION.to_string(),
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
    use rust_decimal_macros::dec;

    use crate::btc::{
        strategy::{
            ApprovedIntent, BtcDecisionStrategyConfig, BtcStrategyConfig,
            BtcVolatilityContinuationConfig, BTC_CHAINLINK_FAIR_VALUE_STRATEGY_FAMILY,
            BTC_FEATURE_SCHEMA_VERSION, BTC_MARKET_ANCHORED_FAIR_VALUE_STRATEGY_FAMILY,
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

        let continuation_config = BtcStrategyConfig {
            strategy_version: BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION.to_string(),
            volatility_continuation: Some(BtcVolatilityContinuationConfig::default()),
            ..BtcStrategyConfig::default()
        };
        let continuation = btc_entry_order_metadata(
            &continuation_config,
            &metadata_intent(&continuation_config.strategy_version),
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
        };
        enforce_runtime_readiness(&mut decision, &Readiness::default());
        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(
            decision.reject_reason,
            Some(BtcRejectReason::RuntimeNotReady)
        );
        assert!(decision.approved_intent.is_none());
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
