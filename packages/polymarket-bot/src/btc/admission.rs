use std::collections::HashSet;

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::fees::dynamic_crypto_taker_fee;

use super::{
    predictive_regime_v2::ShadowPredictiveRegimeCircuitBreakerConfigSelector, types::BtcOutcome,
};

pub const LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION: &str = "loss_regime_confidence_floor_v1";
pub const DAILY_REALIZED_PNL_HIGH_WATER_MARK_SCHEMA_VERSION: &str =
    "daily_realized_pnl_high_water_mark_v1";
pub const SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION: &str =
    "shadow_predictive_regime_circuit_breaker_v1";
pub const SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE: &str = "shadow";

pub const HIGH_WATER_MARK_NOT_ARMED_REASON: &str = "high_water_mark_not_armed";
pub const WITHIN_HIGH_WATER_MARK_RISK_BUDGET_REASON: &str = "within_high_water_mark_risk_budget";
pub const HIGH_WATER_MARK_RISK_BUDGET_EXCEEDED_REASON: &str =
    "high_water_mark_risk_budget_exceeded";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BtcEntryAdmissionConfig {
    pub loss_regime_confidence_floor: LossRegimeConfidenceFloorConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub daily_realized_pnl_high_water_mark: Option<DailyRealizedPnlHighWaterMarkConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_predictive_regime_circuit_breaker:
        Option<ShadowPredictiveRegimeCircuitBreakerConfigSelector>,
}

impl BtcEntryAdmissionConfig {
    pub fn validate(&self) -> Result<()> {
        self.loss_regime_confidence_floor.validate()?;
        if let Some(config) = self.daily_realized_pnl_high_water_mark.as_ref() {
            config.validate()?;
        }
        if let Some(config) = self.shadow_predictive_regime_circuit_breaker.as_ref() {
            config.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowPredictiveRegimeCircuitBreakerConfig {
    pub schema_version: String,
    pub mode: String,
    pub rolling_resolved_market_window: u32,
    pub minimum_resolved_markets: u32,
    pub degradation_brier_score_threshold: Decimal,
    pub degradation_overconfidence_gap_threshold: Decimal,
    pub degradation_confirmation_markets: u32,
    pub recovery_brier_score_threshold: Decimal,
    pub recovery_overconfidence_gap_threshold: Decimal,
    pub recovery_confirmation_markets: u32,
}

impl ShadowPredictiveRegimeCircuitBreakerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION {
            bail!(
                "shadow predictive-regime circuit-breaker schema_version must be {}",
                SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION
            );
        }
        if self.mode != SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE {
            bail!("shadow predictive-regime circuit-breaker mode must be shadow");
        }
        if !(20..=10_000).contains(&self.rolling_resolved_market_window) {
            bail!("rolling_resolved_market_window must be between 20 and 10000");
        }
        if self.minimum_resolved_markets < 20
            || self.minimum_resolved_markets > self.rolling_resolved_market_window
        {
            bail!(
                "minimum_resolved_markets must be at least 20 and cannot exceed rolling_resolved_market_window"
            );
        }
        if !(1..=100).contains(&self.degradation_confirmation_markets) {
            bail!("degradation_confirmation_markets must be between 1 and 100");
        }
        if !(1..=100).contains(&self.recovery_confirmation_markets) {
            bail!("recovery_confirmation_markets must be between 1 and 100");
        }
        for (name, value) in [
            (
                "degradation_brier_score_threshold",
                self.degradation_brier_score_threshold,
            ),
            (
                "degradation_overconfidence_gap_threshold",
                self.degradation_overconfidence_gap_threshold,
            ),
            (
                "recovery_brier_score_threshold",
                self.recovery_brier_score_threshold,
            ),
            (
                "recovery_overconfidence_gap_threshold",
                self.recovery_overconfidence_gap_threshold,
            ),
        ] {
            if !(Decimal::ZERO..=Decimal::ONE).contains(&value) {
                bail!("{name} must be between 0 and 1");
            }
        }
        if self.recovery_brier_score_threshold >= self.degradation_brier_score_threshold {
            bail!("recovery_brier_score_threshold must be below degradation_brier_score_threshold");
        }
        if self.recovery_overconfidence_gap_threshold
            >= self.degradation_overconfidence_gap_threshold
        {
            bail!(
                "recovery_overconfidence_gap_threshold must be below degradation_overconfidence_gap_threshold"
            );
        }
        Ok(())
    }

    pub fn config_hash(&self) -> Result<String> {
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DailyRealizedPnlHighWaterMarkConfig {
    pub schema_version: String,
    pub activation_realized_pnl_usd: Decimal,
    pub max_drawdown_from_high_water_mark_usd: Decimal,
}

impl DailyRealizedPnlHighWaterMarkConfig {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != DAILY_REALIZED_PNL_HIGH_WATER_MARK_SCHEMA_VERSION {
            bail!(
                "daily realized-PnL high-water-mark schema_version must be {}",
                DAILY_REALIZED_PNL_HIGH_WATER_MARK_SCHEMA_VERSION
            );
        }
        if self.activation_realized_pnl_usd <= Decimal::ZERO {
            bail!("activation_realized_pnl_usd must be positive");
        }
        if self.max_drawdown_from_high_water_mark_usd <= Decimal::ZERO {
            bail!("max_drawdown_from_high_water_mark_usd must be positive");
        }
        if self.max_drawdown_from_high_water_mark_usd > self.activation_realized_pnl_usd {
            bail!(
                "max_drawdown_from_high_water_mark_usd cannot exceed activation_realized_pnl_usd"
            );
        }
        Ok(())
    }

    pub fn validate_against_starting_collateral(
        &self,
        starting_collateral_usd: Decimal,
    ) -> Result<()> {
        self.validate()?;
        if starting_collateral_usd <= Decimal::ZERO {
            bail!("paper starting collateral must be positive");
        }
        if self.activation_realized_pnl_usd > starting_collateral_usd
            || self.max_drawdown_from_high_water_mark_usd > starting_collateral_usd
        {
            bail!("high-water-mark amounts cannot exceed paper starting collateral");
        }
        Ok(())
    }

    pub fn config_hash(&self) -> Result<String> {
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DailyRealizedPnlCredit {
    pub settlement_id: Uuid,
    pub order_id: String,
    pub credited_at: DateTime<Utc>,
    pub net_pnl_usd: Decimal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnsettledEntryExposure {
    pub order_id: String,
    pub fill_ids: Vec<Uuid>,
    pub entry_debit_usd: Decimal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposedEntryExposure {
    pub size: Decimal,
    pub limit_price: Decimal,
    pub fee_rate: Decimal,
    entry_debit_usd: Decimal,
}

impl ProposedEntryExposure {
    pub fn new(size: Decimal, limit_price: Decimal, fee_rate: Decimal) -> Result<Self> {
        if size <= Decimal::ZERO {
            bail!("proposed entry size must be positive");
        }
        if limit_price <= Decimal::ZERO || limit_price > Decimal::ONE {
            bail!("proposed entry limit price must be in (0, 1]");
        }
        if fee_rate < Decimal::ZERO || fee_rate > Decimal::ONE {
            bail!("proposed entry fee rate must be between 0 and 1");
        }
        let fee = dynamic_crypto_taker_fee(size, fee_rate, limit_price);
        Ok(Self {
            size,
            limit_price,
            fee_rate,
            entry_debit_usd: size * limit_price + fee,
        })
    }

    pub fn entry_debit_usd(&self) -> Decimal {
        self.entry_debit_usd
    }

    fn validate(&self) -> Result<()> {
        let canonical = Self::new(self.size, self.limit_price, self.fee_rate)?;
        if canonical.entry_debit_usd != self.entry_debit_usd {
            bail!("proposed entry exposure debit is inconsistent with its price and fee inputs");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DailyRealizedPnlHighWaterMarkState {
    pub process_id: Uuid,
    pub as_of: DateTime<Utc>,
    pub period_start_utc: DateTime<Utc>,
    pub period_end_utc: DateTime<Utc>,
    pub credited_settlement_count: u64,
    pub daily_realized_pnl_usd: Decimal,
    pub high_water_mark_usd: Decimal,
    pub unsettled_order_count: u64,
    pub unsettled_entry_debit_usd: Decimal,
    pub last_credited_at: Option<DateTime<Utc>>,
    pub last_credited_settlement_id: Option<Uuid>,
    pub state_evidence_sha256: String,
}

impl DailyRealizedPnlHighWaterMarkState {
    pub fn from_evidence(
        process_id: Uuid,
        as_of: DateTime<Utc>,
        mut credits: Vec<DailyRealizedPnlCredit>,
        mut unsettled_exposures: Vec<UnsettledEntryExposure>,
    ) -> Result<Self> {
        let period_start_utc = as_of
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .context("failed to derive UTC high-water-mark period start")?
            .and_utc();
        let period_end_utc = period_start_utc + Duration::days(1);

        credits.sort_by_key(|credit| (credit.credited_at, credit.settlement_id));
        unsettled_exposures.sort_by(|left, right| left.order_id.cmp(&right.order_id));

        let mut settlement_ids = HashSet::new();
        let mut credited_order_ids = HashSet::new();
        let mut running_pnl = Decimal::ZERO;
        let mut high_water_mark = Decimal::ZERO;
        for credit in &credits {
            if credit.order_id.trim().is_empty() {
                bail!("credited high-water-mark evidence has an empty order id");
            }
            if credit.credited_at < period_start_utc
                || credit.credited_at >= period_end_utc
                || credit.credited_at > as_of
            {
                bail!("credited high-water-mark evidence is outside its causal UTC period");
            }
            if !settlement_ids.insert(credit.settlement_id) {
                bail!("credited high-water-mark evidence contains a duplicate settlement");
            }
            if !credited_order_ids.insert(credit.order_id.clone()) {
                bail!("credited high-water-mark evidence contains a duplicate order");
            }
            running_pnl += credit.net_pnl_usd;
            high_water_mark = high_water_mark.max(running_pnl);
        }

        let mut unsettled_order_ids = HashSet::new();
        let mut unsettled_entry_debit_usd = Decimal::ZERO;
        for exposure in &mut unsettled_exposures {
            if exposure.order_id.trim().is_empty() || exposure.entry_debit_usd <= Decimal::ZERO {
                bail!("unsettled high-water-mark exposure must have an order and positive debit");
            }
            exposure.fill_ids.sort();
            if exposure.fill_ids.is_empty()
                || exposure.fill_ids.windows(2).any(|pair| pair[0] == pair[1])
            {
                bail!("unsettled high-water-mark exposure has missing or duplicate fills");
            }
            if credited_order_ids.contains(&exposure.order_id)
                || !unsettled_order_ids.insert(exposure.order_id.clone())
            {
                bail!("high-water-mark evidence does not canonically classify each order");
            }
            unsettled_entry_debit_usd += exposure.entry_debit_usd;
        }

        #[derive(Serialize)]
        struct StateEvidence<'a> {
            process_id: Uuid,
            as_of: DateTime<Utc>,
            period_start_utc: DateTime<Utc>,
            period_end_utc: DateTime<Utc>,
            credits: &'a [DailyRealizedPnlCredit],
            unsettled_exposures: &'a [UnsettledEntryExposure],
        }
        let state_evidence_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&StateEvidence {
                process_id,
                as_of,
                period_start_utc,
                period_end_utc,
                credits: &credits,
                unsettled_exposures: &unsettled_exposures,
            })?)
        );
        let last_credit = credits.last();
        Ok(Self {
            process_id,
            as_of,
            period_start_utc,
            period_end_utc,
            credited_settlement_count: u64::try_from(credits.len()).unwrap_or(u64::MAX),
            daily_realized_pnl_usd: running_pnl,
            high_water_mark_usd: high_water_mark,
            unsettled_order_count: u64::try_from(unsettled_exposures.len()).unwrap_or(u64::MAX),
            unsettled_entry_debit_usd,
            last_credited_at: last_credit.map(|credit| credit.credited_at),
            last_credited_settlement_id: last_credit.map(|credit| credit.settlement_id),
            state_evidence_sha256,
        })
    }

    pub fn evaluate(
        &self,
        config: &DailyRealizedPnlHighWaterMarkConfig,
        proposed_entry: &ProposedEntryExposure,
    ) -> Result<DailyRealizedPnlHighWaterMarkEvaluation> {
        config.validate()?;
        proposed_entry.validate()?;
        let armed = self.high_water_mark_usd >= config.activation_realized_pnl_usd;
        let protected_floor_usd =
            armed.then(|| self.high_water_mark_usd - config.max_drawdown_from_high_water_mark_usd);
        let remaining_risk_budget_before_proposal_usd = protected_floor_usd
            .map(|floor| self.daily_realized_pnl_usd - floor - self.unsettled_entry_debit_usd);
        let projected_worst_case_pnl_after_usd = self.daily_realized_pnl_usd
            - self.unsettled_entry_debit_usd
            - proposed_entry.entry_debit_usd();
        let (disposition, reason) = match protected_floor_usd {
            None => (
                AdmissionDisposition::Allow,
                HIGH_WATER_MARK_NOT_ARMED_REASON,
            ),
            Some(floor) if projected_worst_case_pnl_after_usd >= floor => (
                AdmissionDisposition::Allow,
                WITHIN_HIGH_WATER_MARK_RISK_BUDGET_REASON,
            ),
            Some(_) => (
                AdmissionDisposition::Defer,
                HIGH_WATER_MARK_RISK_BUDGET_EXCEEDED_REASON,
            ),
        };

        Ok(DailyRealizedPnlHighWaterMarkEvaluation {
            schema_version: config.schema_version.clone(),
            config_hash: config.config_hash()?,
            process_id: self.process_id,
            as_of: self.as_of,
            period_start_utc: self.period_start_utc,
            period_end_utc: self.period_end_utc,
            activation_realized_pnl_usd: config.activation_realized_pnl_usd,
            max_drawdown_from_high_water_mark_usd: config.max_drawdown_from_high_water_mark_usd,
            credited_settlement_count: self.credited_settlement_count,
            daily_realized_pnl_usd: self.daily_realized_pnl_usd,
            high_water_mark_usd: self.high_water_mark_usd,
            armed,
            protected_floor_usd,
            unsettled_order_count: self.unsettled_order_count,
            unsettled_entry_debit_usd: self.unsettled_entry_debit_usd,
            proposed_entry_debit_usd: proposed_entry.entry_debit_usd(),
            remaining_risk_budget_before_proposal_usd,
            projected_worst_case_pnl_after_usd,
            disposition,
            reason: reason.to_string(),
            last_credited_at: self.last_credited_at,
            last_credited_settlement_id: self.last_credited_settlement_id,
            state_evidence_sha256: self.state_evidence_sha256.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DailyRealizedPnlHighWaterMarkEvaluation {
    pub schema_version: String,
    pub config_hash: String,
    pub process_id: Uuid,
    pub as_of: DateTime<Utc>,
    pub period_start_utc: DateTime<Utc>,
    pub period_end_utc: DateTime<Utc>,
    pub activation_realized_pnl_usd: Decimal,
    pub max_drawdown_from_high_water_mark_usd: Decimal,
    pub credited_settlement_count: u64,
    pub daily_realized_pnl_usd: Decimal,
    pub high_water_mark_usd: Decimal,
    pub armed: bool,
    pub protected_floor_usd: Option<Decimal>,
    pub unsettled_order_count: u64,
    pub unsettled_entry_debit_usd: Decimal,
    pub proposed_entry_debit_usd: Decimal,
    pub remaining_risk_budget_before_proposal_usd: Option<Decimal>,
    pub projected_worst_case_pnl_after_usd: Decimal,
    pub disposition: AdmissionDisposition,
    pub reason: String,
    pub last_credited_at: Option<DateTime<Utc>>,
    pub last_credited_settlement_id: Option<Uuid>,
    pub state_evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LossRegimeConfidenceFloorConfig {
    pub schema_version: String,
    pub activation_consecutive_candidate_losses: u32,
    pub min_conservative_probability: Decimal,
    pub release_consecutive_candidate_wins: u32,
}

impl LossRegimeConfidenceFloorConfig {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION {
            bail!(
                "loss-regime confidence-floor schema_version must be {}",
                LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION
            );
        }
        if !(1..=100).contains(&self.activation_consecutive_candidate_losses) {
            bail!("activation_consecutive_candidate_losses must be between 1 and 100");
        }
        if !(Decimal::ZERO..=Decimal::ONE).contains(&self.min_conservative_probability) {
            bail!("min_conservative_probability must be between 0 and 1");
        }
        if !(1..=100).contains(&self.release_consecutive_candidate_wins) {
            bail!("release_consecutive_candidate_wins must be between 1 and 100");
        }
        Ok(())
    }

    pub fn config_hash(&self) -> Result<String> {
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LossRegimeCandidate {
    pub market_id: String,
    pub decision_id: Uuid,
    pub decision_outcome: BtcOutcome,
    pub resolved_outcome: BtcOutcome,
    pub decision_at: DateTime<Utc>,
    pub label_available_at: DateTime<Utc>,
}

impl LossRegimeCandidate {
    pub fn won(&self) -> bool {
        self.decision_outcome == self.resolved_outcome
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossRegimeConfidenceFloorTransition {
    Activated,
    Released,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LossRegimeConfidenceFloorState {
    pub active: bool,
    pub consecutive_candidate_losses: u32,
    pub consecutive_candidate_wins: u32,
    pub candidates_observed: u64,
    pub state_as_of_market_id: Option<String>,
    pub state_as_of_label_available_at: Option<DateTime<Utc>>,
}

impl LossRegimeConfidenceFloorState {
    pub fn from_candidates(
        config: &LossRegimeConfidenceFloorConfig,
        candidates: &[LossRegimeCandidate],
    ) -> Self {
        let mut state = Self::default();
        for candidate in candidates {
            state.apply_candidate(config, candidate);
        }
        state
    }

    pub fn apply_candidate(
        &mut self,
        config: &LossRegimeConfidenceFloorConfig,
        candidate: &LossRegimeCandidate,
    ) -> Option<LossRegimeConfidenceFloorTransition> {
        self.candidates_observed = self.candidates_observed.saturating_add(1);
        self.state_as_of_market_id = Some(candidate.market_id.clone());
        self.state_as_of_label_available_at = Some(candidate.label_available_at);

        if candidate.won() {
            self.consecutive_candidate_losses = 0;
            self.consecutive_candidate_wins = self.consecutive_candidate_wins.saturating_add(1);
            if self.active
                && self.consecutive_candidate_wins >= config.release_consecutive_candidate_wins
            {
                self.active = false;
                return Some(LossRegimeConfidenceFloorTransition::Released);
            }
            return None;
        }

        self.consecutive_candidate_wins = 0;
        self.consecutive_candidate_losses = self.consecutive_candidate_losses.saturating_add(1);
        if !self.active
            && self.consecutive_candidate_losses >= config.activation_consecutive_candidate_losses
        {
            self.active = true;
            return Some(LossRegimeConfidenceFloorTransition::Activated);
        }
        None
    }

    pub fn evaluate(
        &self,
        config: &LossRegimeConfidenceFloorConfig,
        selected_conservative_probability: Decimal,
    ) -> Result<LossRegimeConfidenceFloorEvaluation> {
        let (disposition, reason) = if !self.active {
            (AdmissionDisposition::Allow, "loss_regime_inactive")
        } else if selected_conservative_probability >= config.min_conservative_probability {
            (AdmissionDisposition::Allow, "confidence_floor_satisfied")
        } else {
            (
                AdmissionDisposition::Defer,
                "below_loss_regime_confidence_floor",
            )
        };
        Ok(LossRegimeConfidenceFloorEvaluation {
            schema_version: config.schema_version.clone(),
            config_hash: config.config_hash()?,
            active: self.active,
            consecutive_candidate_losses: self.consecutive_candidate_losses,
            consecutive_candidate_wins: self.consecutive_candidate_wins,
            candidates_observed: self.candidates_observed,
            selected_conservative_probability,
            min_conservative_probability: config.min_conservative_probability,
            disposition,
            reason: reason.to_string(),
            state_as_of_market_id: self.state_as_of_market_id.clone(),
            state_as_of_label_available_at: self.state_as_of_label_available_at,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowPredictiveRegimeCandidate {
    pub market_id: String,
    pub decision_id: Uuid,
    pub decision_outcome: BtcOutcome,
    pub resolved_outcome: BtcOutcome,
    pub selected_point_probability: Decimal,
    pub decision_at: DateTime<Utc>,
    pub label_available_at: DateTime<Utc>,
}

impl ShadowPredictiveRegimeCandidate {
    pub fn won(&self) -> bool {
        self.decision_outcome == self.resolved_outcome
    }

    pub fn validate(&self) -> Result<()> {
        if self.market_id.trim().is_empty() {
            bail!("shadow predictive-regime candidate market_id cannot be empty");
        }
        if self.decision_id.is_nil() {
            bail!("shadow predictive-regime candidate decision_id cannot be nil");
        }
        if !(Decimal::ZERO..=Decimal::ONE).contains(&self.selected_point_probability) {
            bail!("selected_point_probability must be between 0 and 1");
        }
        if self.decision_at >= self.label_available_at {
            bail!("shadow predictive-regime candidate label must follow its decision");
        }
        Ok(())
    }

    pub fn evidence_sha256(&self, process_id: Uuid) -> Result<String> {
        if process_id.is_nil() {
            bail!("shadow predictive-regime evidence process_id cannot be nil");
        }
        self.validate()?;
        #[derive(Serialize)]
        struct CandidateEvidence<'a> {
            process_id: Uuid,
            candidate: &'a ShadowPredictiveRegimeCandidate,
        }
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&CandidateEvidence {
                process_id,
                candidate: self,
            })?)
        ))
    }

    fn brier_score(&self) -> Decimal {
        let observed = if self.won() {
            Decimal::ONE
        } else {
            Decimal::ZERO
        };
        let error = self.selected_point_probability - observed;
        error * error
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowPredictiveRegimeTransition {
    DegradationConfirmed,
    RecoveryConfirmed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowPredictiveRegimeState {
    pub process_id: Uuid,
    pub config_hash: String,
    pub degraded: bool,
    pub consecutive_degradation_markets: u32,
    pub consecutive_recovery_markets: u32,
    pub resolved_markets_observed: u64,
    pub rolling_candidates: Vec<ShadowPredictiveRegimeCandidate>,
    pub state_as_of_market_id: Option<String>,
    pub state_as_of_decision_id: Option<Uuid>,
    pub state_as_of_decision_at: Option<DateTime<Utc>>,
    pub state_as_of_label_available_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ShadowPredictiveRegimeMetrics {
    resolved_market_count: u32,
    wins: u32,
    mean_selected_point_probability: Decimal,
    empirical_accuracy: Decimal,
    brier_score: Decimal,
    calibration_gap: Decimal,
    one_sided_overconfidence_gap: Decimal,
}

impl ShadowPredictiveRegimeState {
    pub fn new(
        process_id: Uuid,
        config: &ShadowPredictiveRegimeCircuitBreakerConfig,
    ) -> Result<Self> {
        config.validate()?;
        if process_id.is_nil() {
            bail!("shadow predictive-regime state process_id cannot be nil");
        }
        Ok(Self {
            process_id,
            config_hash: config.config_hash()?,
            degraded: false,
            consecutive_degradation_markets: 0,
            consecutive_recovery_markets: 0,
            resolved_markets_observed: 0,
            rolling_candidates: Vec::new(),
            state_as_of_market_id: None,
            state_as_of_decision_id: None,
            state_as_of_decision_at: None,
            state_as_of_label_available_at: None,
        })
    }

    pub fn from_candidates(
        process_id: Uuid,
        config: &ShadowPredictiveRegimeCircuitBreakerConfig,
        candidates: &[ShadowPredictiveRegimeCandidate],
    ) -> Result<Self> {
        let mut state = Self::new(process_id, config)?;
        for candidate in candidates {
            state.apply_candidate(config, candidate)?;
        }
        Ok(state)
    }

    pub fn resume_from_state(
        config: &ShadowPredictiveRegimeCircuitBreakerConfig,
        mut state: Self,
        candidates: &[ShadowPredictiveRegimeCandidate],
    ) -> Result<Self> {
        state.validate(config)?;
        for candidate in candidates {
            state.apply_candidate(config, candidate)?;
        }
        Ok(state)
    }

    pub fn candidate_is_newer(
        &self,
        config: &ShadowPredictiveRegimeCircuitBreakerConfig,
        candidate: &ShadowPredictiveRegimeCandidate,
    ) -> Result<bool> {
        self.validate(config)?;
        candidate.validate()?;
        Ok(self
            .rolling_candidates
            .last()
            .is_none_or(|last| candidate_ordering_key(candidate) > candidate_ordering_key(last)))
    }

    pub fn validate(&self, config: &ShadowPredictiveRegimeCircuitBreakerConfig) -> Result<()> {
        config.validate()?;
        if self.process_id.is_nil() {
            bail!("shadow predictive-regime state process_id cannot be nil");
        }
        if self.config_hash != config.config_hash()? {
            bail!("shadow predictive-regime state config hash does not match configuration");
        }
        let rolling_window = usize::try_from(config.rolling_resolved_market_window)
            .context("rolling_resolved_market_window does not fit in memory")?;
        if self.rolling_candidates.len() > rolling_window {
            bail!("shadow predictive-regime state exceeds its rolling window");
        }
        if self.resolved_markets_observed
            < u64::try_from(self.rolling_candidates.len()).unwrap_or(u64::MAX)
        {
            bail!("shadow predictive-regime state observation count is inconsistent");
        }
        if self.consecutive_degradation_markets > config.degradation_confirmation_markets
            || self.consecutive_recovery_markets > config.recovery_confirmation_markets
            || (self.consecutive_degradation_markets > 0 && self.consecutive_recovery_markets > 0)
        {
            bail!("shadow predictive-regime confirmation counters are inconsistent");
        }

        let state_cursor = (
            self.state_as_of_market_id.as_deref(),
            self.state_as_of_decision_id,
            self.state_as_of_decision_at,
            self.state_as_of_label_available_at,
        );
        if self.resolved_markets_observed == 0 {
            if !self.rolling_candidates.is_empty()
                || state_cursor != (None, None, None, None)
                || self.degraded
                || self.consecutive_degradation_markets != 0
                || self.consecutive_recovery_markets != 0
            {
                bail!("empty shadow predictive-regime state is inconsistent");
            }
            return Ok(());
        }
        if self.rolling_candidates.is_empty() {
            bail!("observed shadow predictive-regime state must retain rolling evidence");
        }

        let mut market_ids = HashSet::new();
        let mut decision_ids = HashSet::new();
        let mut previous: Option<&ShadowPredictiveRegimeCandidate> = None;
        for candidate in &self.rolling_candidates {
            candidate.validate()?;
            if !market_ids.insert(candidate.market_id.as_str()) {
                bail!("shadow predictive-regime state contains a duplicate market");
            }
            if !decision_ids.insert(candidate.decision_id) {
                bail!("shadow predictive-regime state contains a duplicate decision");
            }
            if let Some(previous) = previous {
                if candidate_ordering_key(candidate) <= candidate_ordering_key(previous) {
                    bail!("shadow predictive-regime candidates are not in causal order");
                }
            }
            previous = Some(candidate);
        }

        let last = self
            .rolling_candidates
            .last()
            .context("observed shadow predictive-regime state has no final candidate")?;
        if self.state_as_of_market_id.as_deref() != Some(last.market_id.as_str())
            || self.state_as_of_decision_id != Some(last.decision_id)
            || self.state_as_of_decision_at != Some(last.decision_at)
            || self.state_as_of_label_available_at != Some(last.label_available_at)
        {
            bail!("shadow predictive-regime state cursor does not match rolling evidence");
        }
        Ok(())
    }

    pub fn apply_candidate(
        &mut self,
        config: &ShadowPredictiveRegimeCircuitBreakerConfig,
        candidate: &ShadowPredictiveRegimeCandidate,
    ) -> Result<Option<ShadowPredictiveRegimeTransition>> {
        self.validate(config)?;
        candidate.validate()?;
        if let Some(last) = self.rolling_candidates.last() {
            if candidate_ordering_key(candidate) <= candidate_ordering_key(last) {
                bail!("shadow predictive-regime candidate is duplicate or out of causal order");
            }
        }

        self.resolved_markets_observed = self
            .resolved_markets_observed
            .checked_add(1)
            .context("shadow predictive-regime observation count overflowed")?;
        self.rolling_candidates.push(candidate.clone());
        let rolling_window = usize::try_from(config.rolling_resolved_market_window)
            .context("rolling_resolved_market_window does not fit in memory")?;
        if self.rolling_candidates.len() > rolling_window {
            self.rolling_candidates.remove(0);
        }
        self.state_as_of_market_id = Some(candidate.market_id.clone());
        self.state_as_of_decision_id = Some(candidate.decision_id);
        self.state_as_of_decision_at = Some(candidate.decision_at);
        self.state_as_of_label_available_at = Some(candidate.label_available_at);

        let Some(metrics) = self.current_metrics(config) else {
            self.consecutive_degradation_markets = 0;
            self.consecutive_recovery_markets = 0;
            return Ok(None);
        };
        let degradation_condition_met = metrics.brier_score
            >= config.degradation_brier_score_threshold
            && metrics.one_sided_overconfidence_gap
                >= config.degradation_overconfidence_gap_threshold;
        let recovery_condition_met = metrics.brier_score <= config.recovery_brier_score_threshold
            && metrics.one_sided_overconfidence_gap <= config.recovery_overconfidence_gap_threshold;

        if self.degraded {
            self.consecutive_degradation_markets = 0;
            if recovery_condition_met {
                self.consecutive_recovery_markets =
                    self.consecutive_recovery_markets.saturating_add(1);
                if self.consecutive_recovery_markets >= config.recovery_confirmation_markets {
                    self.degraded = false;
                    return Ok(Some(ShadowPredictiveRegimeTransition::RecoveryConfirmed));
                }
            } else {
                self.consecutive_recovery_markets = 0;
            }
            return Ok(None);
        }

        self.consecutive_recovery_markets = 0;
        if degradation_condition_met {
            self.consecutive_degradation_markets =
                self.consecutive_degradation_markets.saturating_add(1);
            if self.consecutive_degradation_markets >= config.degradation_confirmation_markets {
                self.degraded = true;
                return Ok(Some(ShadowPredictiveRegimeTransition::DegradationConfirmed));
            }
        } else {
            self.consecutive_degradation_markets = 0;
        }
        Ok(None)
    }

    pub fn evaluate(
        &self,
        config: &ShadowPredictiveRegimeCircuitBreakerConfig,
        as_of: DateTime<Utc>,
    ) -> Result<ShadowPredictiveRegimeEvaluation> {
        self.validate(config)?;
        if self
            .state_as_of_label_available_at
            .is_some_and(|label_available_at| as_of < label_available_at)
        {
            bail!("shadow predictive-regime evaluation cannot precede its resolved evidence");
        }

        let metrics = self.current_metrics(config);
        let degradation_condition_met = metrics.is_some_and(|metrics| {
            metrics.brier_score >= config.degradation_brier_score_threshold
                && metrics.one_sided_overconfidence_gap
                    >= config.degradation_overconfidence_gap_threshold
        });
        let recovery_condition_met = metrics.is_some_and(|metrics| {
            metrics.brier_score <= config.recovery_brier_score_threshold
                && metrics.one_sided_overconfidence_gap
                    <= config.recovery_overconfidence_gap_threshold
        });
        let reason = if metrics.is_none() {
            "shadow_predictive_regime_warming_up"
        } else if self.degraded && self.consecutive_recovery_markets > 0 {
            "shadow_predictive_regime_degraded_recovery_pending"
        } else if self.degraded {
            "shadow_predictive_regime_degraded"
        } else if self.consecutive_degradation_markets > 0 {
            "shadow_predictive_regime_degradation_pending"
        } else {
            "shadow_predictive_regime_healthy"
        };
        let state_evidence_sha256 = self.evidence_sha256(config)?;

        #[derive(Serialize)]
        struct EvaluationEvidence<'a> {
            schema_version: &'a str,
            config_hash: &'a str,
            mode: &'a str,
            process_id: Uuid,
            as_of: DateTime<Utc>,
            state_evidence_sha256: &'a str,
            degraded: bool,
            would_defer: bool,
            degradation_condition_met: bool,
            recovery_condition_met: bool,
            reason: &'a str,
        }
        let config_hash = config.config_hash()?;
        let evaluation_evidence_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&EvaluationEvidence {
                schema_version: &config.schema_version,
                config_hash: &config_hash,
                mode: &config.mode,
                process_id: self.process_id,
                as_of,
                state_evidence_sha256: &state_evidence_sha256,
                degraded: self.degraded,
                would_defer: self.degraded,
                degradation_condition_met,
                recovery_condition_met,
                reason,
            })?)
        );

        Ok(ShadowPredictiveRegimeEvaluation {
            schema_version: config.schema_version.clone(),
            config_hash,
            mode: config.mode.clone(),
            process_id: self.process_id,
            as_of,
            shadow_only: true,
            sample_ready: metrics.is_some(),
            rolling_resolved_market_count: metrics.map(|metrics| metrics.resolved_market_count),
            resolved_markets_observed: self.resolved_markets_observed,
            rolling_wins: metrics.map(|metrics| metrics.wins),
            mean_selected_point_probability: metrics
                .map(|metrics| metrics.mean_selected_point_probability),
            empirical_accuracy: metrics.map(|metrics| metrics.empirical_accuracy),
            brier_score: metrics.map(|metrics| metrics.brier_score),
            calibration_gap: metrics.map(|metrics| metrics.calibration_gap),
            one_sided_overconfidence_gap: metrics
                .map(|metrics| metrics.one_sided_overconfidence_gap),
            degradation_condition_met,
            recovery_condition_met,
            degraded: self.degraded,
            consecutive_degradation_markets: self.consecutive_degradation_markets,
            consecutive_recovery_markets: self.consecutive_recovery_markets,
            would_defer: self.degraded,
            disposition: AdmissionDisposition::Allow,
            reason: reason.to_string(),
            state_as_of_market_id: self.state_as_of_market_id.clone(),
            state_as_of_decision_id: self.state_as_of_decision_id,
            state_as_of_label_available_at: self.state_as_of_label_available_at,
            state_evidence_sha256,
            evaluation_evidence_sha256,
            state: self.clone(),
            state_checkpoint_eligible: false,
            telemetry_error: None,
            refresh_pending: false,
        })
    }

    pub fn evidence_sha256(
        &self,
        config: &ShadowPredictiveRegimeCircuitBreakerConfig,
    ) -> Result<String> {
        self.validate(config)?;
        #[derive(Serialize)]
        struct StateEvidence<'a> {
            schema_version: &'a str,
            config_hash: String,
            state: &'a ShadowPredictiveRegimeState,
        }
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&StateEvidence {
                schema_version: &config.schema_version,
                config_hash: config.config_hash()?,
                state: self,
            })?)
        ))
    }

    fn current_metrics(
        &self,
        config: &ShadowPredictiveRegimeCircuitBreakerConfig,
    ) -> Option<ShadowPredictiveRegimeMetrics> {
        let count = u32::try_from(self.rolling_candidates.len()).ok()?;
        if count < config.minimum_resolved_markets {
            return None;
        }
        let denominator = Decimal::from(count);
        let wins = u32::try_from(
            self.rolling_candidates
                .iter()
                .filter(|candidate| candidate.won())
                .count(),
        )
        .ok()?;
        let mean_selected_point_probability = self
            .rolling_candidates
            .iter()
            .map(|candidate| candidate.selected_point_probability)
            .sum::<Decimal>()
            / denominator;
        let empirical_accuracy = Decimal::from(wins) / denominator;
        let brier_score = self
            .rolling_candidates
            .iter()
            .map(ShadowPredictiveRegimeCandidate::brier_score)
            .sum::<Decimal>()
            / denominator;
        let calibration_gap = mean_selected_point_probability - empirical_accuracy;
        let one_sided_overconfidence_gap = calibration_gap.max(Decimal::ZERO);
        Some(ShadowPredictiveRegimeMetrics {
            resolved_market_count: count,
            wins,
            mean_selected_point_probability,
            empirical_accuracy,
            brier_score,
            calibration_gap,
            one_sided_overconfidence_gap,
        })
    }
}

fn candidate_ordering_key(
    candidate: &ShadowPredictiveRegimeCandidate,
) -> (DateTime<Utc>, DateTime<Utc>, Uuid, &str) {
    (
        candidate.label_available_at,
        candidate.decision_at,
        candidate.decision_id,
        candidate.market_id.as_str(),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionDisposition {
    Allow,
    Defer,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LossRegimeConfidenceFloorEvaluation {
    pub schema_version: String,
    pub config_hash: String,
    pub active: bool,
    pub consecutive_candidate_losses: u32,
    pub consecutive_candidate_wins: u32,
    pub candidates_observed: u64,
    pub selected_conservative_probability: Decimal,
    pub min_conservative_probability: Decimal,
    pub disposition: AdmissionDisposition,
    pub reason: String,
    pub state_as_of_market_id: Option<String>,
    pub state_as_of_label_available_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowPredictiveRegimeEvaluation {
    pub schema_version: String,
    pub config_hash: String,
    pub mode: String,
    pub process_id: Uuid,
    pub as_of: DateTime<Utc>,
    pub shadow_only: bool,
    pub sample_ready: bool,
    pub rolling_resolved_market_count: Option<u32>,
    pub resolved_markets_observed: u64,
    pub rolling_wins: Option<u32>,
    pub mean_selected_point_probability: Option<Decimal>,
    pub empirical_accuracy: Option<Decimal>,
    pub brier_score: Option<Decimal>,
    /// Signed mean forecast probability minus empirical accuracy. Positive values are
    /// overconfidence; negative values are underconfidence.
    pub calibration_gap: Option<Decimal>,
    pub one_sided_overconfidence_gap: Option<Decimal>,
    pub degradation_condition_met: bool,
    pub recovery_condition_met: bool,
    pub degraded: bool,
    pub consecutive_degradation_markets: u32,
    pub consecutive_recovery_markets: u32,
    pub would_defer: bool,
    pub disposition: AdmissionDisposition,
    pub reason: String,
    pub state_as_of_market_id: Option<String>,
    pub state_as_of_decision_id: Option<Uuid>,
    pub state_as_of_label_available_at: Option<DateTime<Utc>>,
    pub state_evidence_sha256: String,
    pub evaluation_evidence_sha256: String,
    pub state: ShadowPredictiveRegimeState,
    /// True only after runtime hydration has verified this snapshot as a durable replay checkpoint.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub state_checkpoint_eligible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry_error: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub refresh_pending: bool,
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, TimeZone};
    use rust_decimal_macros::dec;

    use crate::btc::{
        ShadowPredictiveRegimeCircuitBreakerV2Config,
        SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_V2_SCHEMA_VERSION,
    };

    use super::*;

    fn config() -> LossRegimeConfidenceFloorConfig {
        LossRegimeConfidenceFloorConfig {
            schema_version: LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION.to_string(),
            activation_consecutive_candidate_losses: 2,
            min_conservative_probability: dec!(0.50),
            release_consecutive_candidate_wins: 1,
        }
    }

    fn candidate(index: i64, won: bool) -> LossRegimeCandidate {
        let at = Utc::now() + Duration::minutes(index * 5);
        LossRegimeCandidate {
            market_id: format!("market-{index}"),
            decision_id: Uuid::new_v4(),
            decision_outcome: BtcOutcome::Up,
            resolved_outcome: if won {
                BtcOutcome::Up
            } else {
                BtcOutcome::Down
            },
            decision_at: at - Duration::minutes(4),
            label_available_at: at,
        }
    }

    fn shadow_config() -> ShadowPredictiveRegimeCircuitBreakerConfig {
        ShadowPredictiveRegimeCircuitBreakerConfig {
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
        }
    }

    fn shadow_v2_config() -> ShadowPredictiveRegimeCircuitBreakerV2Config {
        ShadowPredictiveRegimeCircuitBreakerV2Config {
            schema_version: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_V2_SCHEMA_VERSION.to_string(),
            mode: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE.to_string(),
            fast_resolved_market_window: 4,
            slow_resolved_market_window: 20,
            minimum_resolved_markets: 20,
            max_evidence_gap_seconds: 900,
            degradation_fast_brier_score_threshold: dec!(0.27),
            degradation_fast_minus_slow_threshold: dec!(0.02),
            degradation_slow_brier_score_threshold: dec!(0.25),
            degradation_confirmation_markets: 2,
            recovery_fast_brier_score_threshold: dec!(0.25),
            recovery_fast_minus_slow_ceiling: Decimal::ZERO,
            recovery_confirmation_markets: 2,
        }
    }

    fn shadow_candidate(
        index: i64,
        selected_point_probability: Decimal,
        won: bool,
    ) -> ShadowPredictiveRegimeCandidate {
        let label_available_at = Utc.with_ymd_and_hms(2026, 7, 20, 0, 0, 0).single().unwrap()
            + Duration::minutes(index * 5);
        ShadowPredictiveRegimeCandidate {
            market_id: format!("shadow-market-{index}"),
            decision_id: Uuid::from_u128(u128::try_from(index + 1).unwrap()),
            decision_outcome: BtcOutcome::Up,
            resolved_outcome: if won {
                BtcOutcome::Up
            } else {
                BtcOutcome::Down
            },
            selected_point_probability,
            decision_at: label_available_at - Duration::minutes(4),
            label_available_at,
        }
    }

    fn high_water_mark_config() -> DailyRealizedPnlHighWaterMarkConfig {
        DailyRealizedPnlHighWaterMarkConfig {
            schema_version: DAILY_REALIZED_PNL_HIGH_WATER_MARK_SCHEMA_VERSION.to_string(),
            activation_realized_pnl_usd: dec!(5),
            max_drawdown_from_high_water_mark_usd: dec!(5),
        }
    }

    fn high_water_mark_as_of() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0)
            .single()
            .unwrap()
    }

    fn credit(
        index: u128,
        credited_at: DateTime<Utc>,
        net_pnl_usd: Decimal,
    ) -> DailyRealizedPnlCredit {
        DailyRealizedPnlCredit {
            settlement_id: Uuid::from_u128(index),
            order_id: format!("order-{index}"),
            credited_at,
            net_pnl_usd,
        }
    }

    #[test]
    fn two_losses_activate_and_one_win_releases() {
        let config = config();
        let mut state = LossRegimeConfidenceFloorState::default();
        assert_eq!(state.apply_candidate(&config, &candidate(1, false)), None);
        assert_eq!(
            state.apply_candidate(&config, &candidate(2, false)),
            Some(LossRegimeConfidenceFloorTransition::Activated)
        );
        assert!(state.active);
        assert_eq!(
            state.apply_candidate(&config, &candidate(3, true)),
            Some(LossRegimeConfidenceFloorTransition::Released)
        );
        assert!(!state.active);
    }

    #[test]
    fn intervening_win_resets_activation_streak() {
        let config = config();
        let state = LossRegimeConfidenceFloorState::from_candidates(
            &config,
            &[candidate(1, false), candidate(2, true), candidate(3, false)],
        );
        assert!(!state.active);
        assert_eq!(state.consecutive_candidate_losses, 1);
    }

    #[test]
    fn active_state_survives_losses_and_respects_multi_win_release() {
        let mut config = config();
        config.release_consecutive_candidate_wins = 2;
        let mut state = LossRegimeConfidenceFloorState::from_candidates(
            &config,
            &[candidate(1, false), candidate(2, false)],
        );
        assert!(state.active);
        assert_eq!(state.apply_candidate(&config, &candidate(3, false)), None);
        assert!(state.active);
        assert_eq!(state.apply_candidate(&config, &candidate(4, true)), None);
        assert!(state.active);
        assert_eq!(state.apply_candidate(&config, &candidate(5, false)), None);
        assert_eq!(state.apply_candidate(&config, &candidate(6, true)), None);
        assert_eq!(
            state.apply_candidate(&config, &candidate(7, true)),
            Some(LossRegimeConfidenceFloorTransition::Released)
        );
        assert!(!state.active);
    }

    #[test]
    fn late_causal_candidate_requires_full_ordered_replay() {
        let config = config();
        let first_loss = candidate(2, false);
        let second_loss = candidate(3, false);
        let late_earlier_win = candidate(1, true);

        let mut incorrectly_appended = LossRegimeConfidenceFloorState::from_candidates(
            &config,
            &[first_loss.clone(), second_loss.clone()],
        );
        incorrectly_appended.apply_candidate(&config, &late_earlier_win);
        assert!(!incorrectly_appended.active);

        let causally_replayed = LossRegimeConfidenceFloorState::from_candidates(
            &config,
            &[late_earlier_win, first_loss, second_loss],
        );
        assert!(causally_replayed.active);
    }

    #[test]
    fn active_floor_defers_below_half_and_allows_half() {
        let config = config();
        let state = LossRegimeConfidenceFloorState::from_candidates(
            &config,
            &[candidate(1, false), candidate(2, false)],
        );
        assert_eq!(
            state.evaluate(&config, dec!(0.499999)).unwrap().disposition,
            AdmissionDisposition::Defer
        );
        assert_eq!(
            state.evaluate(&config, dec!(0.50)).unwrap().disposition,
            AdmissionDisposition::Allow
        );
    }

    #[test]
    fn inactive_floor_never_defers() {
        let config = config();
        let evaluation = LossRegimeConfidenceFloorState::default()
            .evaluate(&config, dec!(0.10))
            .unwrap();
        assert_eq!(evaluation.disposition, AdmissionDisposition::Allow);
        assert!(!evaluation.active);
    }

    #[test]
    fn configuration_is_strict_and_bounded() {
        assert!(config().validate().is_ok());

        let mut invalid = config();
        invalid.schema_version = "other".to_string();
        assert!(invalid.validate().is_err());

        let mut invalid = config();
        invalid.activation_consecutive_candidate_losses = 0;
        assert!(invalid.validate().is_err());

        let mut invalid = config();
        invalid.release_consecutive_candidate_wins = 101;
        assert!(invalid.validate().is_err());

        let mut invalid = config();
        invalid.min_conservative_probability = dec!(1.01);
        assert!(invalid.validate().is_err());

        let unknown = serde_json::json!({
            "schema_version": LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION,
            "activation_consecutive_candidate_losses": 2,
            "min_conservative_probability": "0.50",
            "release_consecutive_candidate_wins": 1,
            "mode": "observe"
        });
        assert!(serde_json::from_value::<LossRegimeConfidenceFloorConfig>(unknown).is_err());
    }

    #[test]
    fn optional_high_water_mark_config_is_strict_and_omitted_when_absent() {
        let entry_admission = BtcEntryAdmissionConfig {
            loss_regime_confidence_floor: config(),
            daily_realized_pnl_high_water_mark: None,
            shadow_predictive_regime_circuit_breaker: None,
        };
        let serialized = serde_json::to_value(&entry_admission).unwrap();
        assert!(serialized
            .get("daily_realized_pnl_high_water_mark")
            .is_none());

        let mut invalid = high_water_mark_config();
        invalid.schema_version = "other".to_string();
        assert!(invalid.validate().is_err());
        let mut invalid = high_water_mark_config();
        invalid.activation_realized_pnl_usd = Decimal::ZERO;
        assert!(invalid.validate().is_err());
        let mut invalid = high_water_mark_config();
        invalid.max_drawdown_from_high_water_mark_usd = dec!(5.01);
        assert!(invalid.validate().is_err());
        assert!(high_water_mark_config()
            .validate_against_starting_collateral(dec!(4.99))
            .is_err());

        let unknown = serde_json::json!({
            "schema_version": DAILY_REALIZED_PNL_HIGH_WATER_MARK_SCHEMA_VERSION,
            "activation_realized_pnl_usd": "5.00",
            "max_drawdown_from_high_water_mark_usd": "5.00",
            "reset_timezone": "UTC"
        });
        assert!(serde_json::from_value::<DailyRealizedPnlHighWaterMarkConfig>(unknown).is_err());
    }

    #[test]
    fn proposed_entry_exposure_uses_limit_price_notional_and_dynamic_fee() {
        let exposure = ProposedEntryExposure::new(dec!(5), dec!(0.40), dec!(0.25)).unwrap();
        assert_eq!(exposure.entry_debit_usd(), dec!(2.30));
        assert_eq!(
            ProposedEntryExposure::new(dec!(5), Decimal::ONE, dec!(1))
                .unwrap()
                .entry_debit_usd(),
            dec!(5)
        );
        assert!(ProposedEntryExposure::new(Decimal::ZERO, dec!(0.4), dec!(0.25)).is_err());
        assert!(ProposedEntryExposure::new(dec!(5), dec!(1.01), dec!(0.25)).is_err());
        assert!(ProposedEntryExposure::new(dec!(5), dec!(0.4), dec!(1.01)).is_err());
    }

    #[test]
    fn daily_state_replays_credited_pnl_in_causal_order_and_hashes_canonically() {
        let as_of = high_water_mark_as_of();
        let credits = vec![
            credit(3, as_of - Duration::minutes(5), dec!(-2)),
            credit(1, as_of - Duration::minutes(15), dec!(5)),
            credit(2, as_of - Duration::minutes(10), dec!(3)),
        ];
        let exposures = vec![UnsettledEntryExposure {
            order_id: "open-order".to_string(),
            fill_ids: vec![Uuid::from_u128(12), Uuid::from_u128(11)],
            entry_debit_usd: dec!(1.25),
        }];

        let left = DailyRealizedPnlHighWaterMarkState::from_evidence(
            Uuid::from_u128(100),
            as_of,
            credits.clone(),
            exposures.clone(),
        )
        .unwrap();
        let right = DailyRealizedPnlHighWaterMarkState::from_evidence(
            Uuid::from_u128(100),
            as_of,
            credits.into_iter().rev().collect(),
            exposures,
        )
        .unwrap();

        assert_eq!(left.daily_realized_pnl_usd, dec!(6));
        assert_eq!(left.high_water_mark_usd, dec!(8));
        assert_eq!(left.unsettled_entry_debit_usd, dec!(1.25));
        assert_eq!(left.last_credited_settlement_id, Some(Uuid::from_u128(3)));
        assert_eq!(left.state_evidence_sha256, right.state_evidence_sha256);
    }

    #[test]
    fn high_water_mark_risk_budget_allows_floor_equality_and_defers_below_it() {
        let as_of = high_water_mark_as_of();
        let state = DailyRealizedPnlHighWaterMarkState::from_evidence(
            Uuid::from_u128(100),
            as_of,
            vec![
                credit(1, as_of - Duration::minutes(10), dec!(8)),
                credit(2, as_of - Duration::minutes(5), dec!(-2)),
            ],
            vec![UnsettledEntryExposure {
                order_id: "open-order".to_string(),
                fill_ids: vec![Uuid::from_u128(11)],
                entry_debit_usd: dec!(1),
            }],
        )
        .unwrap();
        let at_floor = ProposedEntryExposure::new(dec!(2), Decimal::ONE, Decimal::ZERO).unwrap();
        let below_floor =
            ProposedEntryExposure::new(dec!(2.01), Decimal::ONE, Decimal::ZERO).unwrap();

        let allowed = state
            .evaluate(&high_water_mark_config(), &at_floor)
            .unwrap();
        assert!(allowed.armed);
        assert_eq!(allowed.protected_floor_usd, Some(dec!(3)));
        assert_eq!(allowed.projected_worst_case_pnl_after_usd, dec!(3));
        assert_eq!(allowed.disposition, AdmissionDisposition::Allow);
        assert_eq!(allowed.reason, WITHIN_HIGH_WATER_MARK_RISK_BUDGET_REASON);

        let deferred = state
            .evaluate(&high_water_mark_config(), &below_floor)
            .unwrap();
        assert_eq!(deferred.disposition, AdmissionDisposition::Defer);
        assert_eq!(deferred.reason, HIGH_WATER_MARK_RISK_BUDGET_EXCEEDED_REASON);
    }

    #[test]
    fn unarmed_high_water_mark_allows_without_spending_a_nonexistent_floor() {
        let as_of = high_water_mark_as_of();
        let state = DailyRealizedPnlHighWaterMarkState::from_evidence(
            Uuid::from_u128(100),
            as_of,
            vec![credit(1, as_of - Duration::minutes(5), dec!(4.99))],
            Vec::new(),
        )
        .unwrap();
        let proposed = ProposedEntryExposure::new(dec!(5), dec!(0.95), dec!(1)).unwrap();
        let evaluation = state
            .evaluate(&high_water_mark_config(), &proposed)
            .unwrap();

        assert!(!evaluation.armed);
        assert_eq!(evaluation.protected_floor_usd, None);
        assert_eq!(evaluation.disposition, AdmissionDisposition::Allow);
        assert_eq!(evaluation.reason, HIGH_WATER_MARK_NOT_ARMED_REASON);
    }

    #[test]
    fn daily_state_rejects_noncausal_and_noncanonical_evidence() {
        let as_of = high_water_mark_as_of();
        assert!(DailyRealizedPnlHighWaterMarkState::from_evidence(
            Uuid::from_u128(100),
            as_of,
            vec![credit(1, as_of + Duration::seconds(1), dec!(1))],
            Vec::new(),
        )
        .is_err());

        let duplicated_order = vec![
            credit(1, as_of - Duration::minutes(2), dec!(1)),
            DailyRealizedPnlCredit {
                settlement_id: Uuid::from_u128(2),
                order_id: "order-1".to_string(),
                credited_at: as_of - Duration::minutes(1),
                net_pnl_usd: dec!(1),
            },
        ];
        assert!(DailyRealizedPnlHighWaterMarkState::from_evidence(
            Uuid::from_u128(100),
            as_of,
            duplicated_order,
            Vec::new(),
        )
        .is_err());
    }

    #[test]
    fn shadow_config_is_explicit_strict_and_hysteretic() {
        let config = shadow_config();
        assert!(config.validate().is_ok());

        let mut invalid = config.clone();
        invalid.mode = "enforced".to_string();
        assert!(invalid.validate().is_err());

        let mut invalid = config.clone();
        invalid.rolling_resolved_market_window = 19;
        assert!(invalid.validate().is_err());

        let mut invalid = config.clone();
        invalid.minimum_resolved_markets = 19;
        assert!(invalid.validate().is_err());

        let mut invalid = config.clone();
        invalid.recovery_brier_score_threshold = invalid.degradation_brier_score_threshold;
        assert!(invalid.validate().is_err());

        let mut invalid = config.clone();
        invalid.recovery_overconfidence_gap_threshold =
            invalid.degradation_overconfidence_gap_threshold;
        assert!(invalid.validate().is_err());

        let unknown = serde_json::json!({
            "schema_version": SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION,
            "mode": "shadow",
            "rolling_resolved_market_window": 20,
            "minimum_resolved_markets": 20,
            "degradation_brier_score_threshold": "0.23",
            "degradation_overconfidence_gap_threshold": "0.12",
            "degradation_confirmation_markets": 2,
            "recovery_brier_score_threshold": "0.21",
            "recovery_overconfidence_gap_threshold": "0.05",
            "recovery_confirmation_markets": 2,
            "enforce": false
        });
        assert!(
            serde_json::from_value::<ShadowPredictiveRegimeCircuitBreakerConfig>(unknown).is_err()
        );
    }

    #[test]
    fn absent_shadow_config_preserves_entry_admission_serialization() {
        let entry_admission = BtcEntryAdmissionConfig {
            loss_regime_confidence_floor: config(),
            daily_realized_pnl_high_water_mark: None,
            shadow_predictive_regime_circuit_breaker: None,
        };
        let serialized = serde_json::to_value(entry_admission).unwrap();
        assert!(serialized
            .get("shadow_predictive_regime_circuit_breaker")
            .is_none());
        assert_eq!(serialized.as_object().unwrap().len(), 1);
    }

    #[test]
    fn versioned_shadow_config_preserves_exact_v1_wire_shape_and_hash() {
        let shadow = shadow_config();
        let entry_admission = BtcEntryAdmissionConfig {
            loss_regime_confidence_floor: config(),
            daily_realized_pnl_high_water_mark: None,
            shadow_predictive_regime_circuit_breaker: Some(shadow.clone().into()),
        };

        entry_admission.validate().unwrap();
        let serialized = serde_json::to_value(&entry_admission).unwrap();
        assert_eq!(
            serialized["shadow_predictive_regime_circuit_breaker"],
            serde_json::to_value(&shadow).unwrap()
        );
        assert!(serialized["shadow_predictive_regime_circuit_breaker"]
            .get("V1")
            .is_none());
        assert_eq!(
            entry_admission
                .shadow_predictive_regime_circuit_breaker
                .as_ref()
                .unwrap()
                .config_hash()
                .unwrap(),
            "45545d1b10183a6c33eb559134575c739e93766fc06837cc5db95bc13ec0480f"
        );

        let restored: BtcEntryAdmissionConfig = serde_json::from_value(serialized).unwrap();
        assert_eq!(restored, entry_admission);
    }

    #[test]
    fn versioned_shadow_config_parses_v2_under_existing_strict_key() {
        let shadow = shadow_v2_config();
        let entry_admission = BtcEntryAdmissionConfig {
            loss_regime_confidence_floor: config(),
            daily_realized_pnl_high_water_mark: None,
            shadow_predictive_regime_circuit_breaker: Some(shadow.clone().into()),
        };
        let serialized = serde_json::to_value(&entry_admission).unwrap();

        let restored: BtcEntryAdmissionConfig = serde_json::from_value(serialized.clone()).unwrap();
        assert!(restored
            .shadow_predictive_regime_circuit_breaker
            .as_ref()
            .unwrap()
            .as_v2()
            .is_some());
        assert_eq!(
            serialized["shadow_predictive_regime_circuit_breaker"],
            serde_json::to_value(shadow).unwrap()
        );

        let mut unknown = serialized;
        unknown["shadow_predictive_regime_circuit_breaker"]["enforce"] = serde_json::json!(false);
        assert!(serde_json::from_value::<BtcEntryAdmissionConfig>(unknown).is_err());
    }

    #[test]
    fn shadow_degradation_requires_brier_and_overconfidence_for_two_updates() {
        let config = shadow_config();
        let process_id = Uuid::from_u128(700);
        let mut state = ShadowPredictiveRegimeState::new(process_id, &config).unwrap();

        for index in 0..19 {
            assert_eq!(
                state
                    .apply_candidate(&config, &shadow_candidate(index, dec!(0.90), false))
                    .unwrap(),
                None
            );
        }
        assert_eq!(state.consecutive_degradation_markets, 0);
        assert!(!state.degraded);

        assert_eq!(
            state
                .apply_candidate(&config, &shadow_candidate(19, dec!(0.90), false))
                .unwrap(),
            None
        );
        assert_eq!(state.consecutive_degradation_markets, 1);
        assert!(!state.degraded);

        assert_eq!(
            state
                .apply_candidate(&config, &shadow_candidate(20, dec!(0.90), false))
                .unwrap(),
            Some(ShadowPredictiveRegimeTransition::DegradationConfirmed)
        );
        let evaluation = state
            .evaluate(
                &config,
                shadow_candidate(20, dec!(0.90), false).label_available_at,
            )
            .unwrap();
        assert!(evaluation.degradation_condition_met);
        assert!(evaluation.degraded);
        assert!(evaluation.would_defer);
        assert_eq!(evaluation.disposition, AdmissionDisposition::Allow);
        assert_eq!(evaluation.process_id, process_id);
        assert_eq!(evaluation.rolling_resolved_market_count, Some(20));
        assert_eq!(evaluation.brier_score, Some(dec!(0.81)));
        assert_eq!(evaluation.calibration_gap, Some(dec!(0.90)));
        assert_eq!(evaluation.one_sided_overconfidence_gap, Some(dec!(0.90)));
        assert_eq!(state.rolling_candidates.len(), 20);
        assert_eq!(state.resolved_markets_observed, 21);

        let serialized = serde_json::to_value(&evaluation).unwrap();
        assert!(serialized.get("experiment_id").is_none());
        assert_eq!(serialized["state"]["process_id"], process_id.to_string());
    }

    #[test]
    fn shadow_brier_without_one_sided_overconfidence_does_not_degrade() {
        let config = shadow_config();
        let mut state = ShadowPredictiveRegimeState::new(Uuid::from_u128(701), &config).unwrap();
        for index in 0..22 {
            state
                .apply_candidate(
                    &config,
                    &shadow_candidate(index, dec!(0.50), index % 2 == 0),
                )
                .unwrap();
        }
        let evaluation = state
            .evaluate(
                &config,
                shadow_candidate(21, dec!(0.50), false).label_available_at,
            )
            .unwrap();
        assert_eq!(evaluation.brier_score, Some(dec!(0.25)));
        assert_eq!(evaluation.calibration_gap, Some(Decimal::ZERO));
        assert_eq!(evaluation.one_sided_overconfidence_gap, Some(Decimal::ZERO));
        assert!(!evaluation.degradation_condition_met);
        assert!(!evaluation.degraded);
        assert!(!evaluation.would_defer);
        assert_eq!(evaluation.disposition, AdmissionDisposition::Allow);
    }

    #[test]
    fn shadow_recovery_requires_two_updates_and_neutral_zone_preserves_degradation() {
        let config = shadow_config();
        let mut state = ShadowPredictiveRegimeState::new(Uuid::from_u128(702), &config).unwrap();
        for index in 0..21 {
            state
                .apply_candidate(&config, &shadow_candidate(index, dec!(0.90), false))
                .unwrap();
        }
        assert!(state.degraded);

        for index in 21..39 {
            assert_eq!(
                state
                    .apply_candidate(&config, &shadow_candidate(index, dec!(0.99), true))
                    .unwrap(),
                None
            );
        }
        let neutral = state
            .evaluate(
                &config,
                shadow_candidate(38, dec!(0.99), true).label_available_at,
            )
            .unwrap();
        assert!(!neutral.degradation_condition_met);
        assert!(!neutral.recovery_condition_met);
        assert!(neutral.degraded);
        assert_eq!(neutral.disposition, AdmissionDisposition::Allow);

        assert_eq!(
            state
                .apply_candidate(&config, &shadow_candidate(39, dec!(0.99), true))
                .unwrap(),
            None
        );
        let pending = state
            .evaluate(
                &config,
                shadow_candidate(39, dec!(0.99), true).label_available_at,
            )
            .unwrap();
        assert!(pending.recovery_condition_met);
        assert_eq!(pending.consecutive_recovery_markets, 1);
        assert!(pending.degraded);
        assert_eq!(pending.disposition, AdmissionDisposition::Allow);

        assert_eq!(
            state
                .apply_candidate(&config, &shadow_candidate(40, dec!(0.99), true))
                .unwrap(),
            Some(ShadowPredictiveRegimeTransition::RecoveryConfirmed)
        );
        let recovered = state
            .evaluate(
                &config,
                shadow_candidate(40, dec!(0.99), true).label_available_at,
            )
            .unwrap();
        assert!(!recovered.degraded);
        assert!(!recovered.would_defer);
        assert_eq!(recovered.disposition, AdmissionDisposition::Allow);
    }

    #[test]
    fn shadow_state_resume_is_process_owned_config_bound_and_causal() {
        let config = shadow_config();
        let process_id = Uuid::from_u128(703);
        let candidates = (0..20)
            .map(|index| shadow_candidate(index, dec!(0.90), false))
            .collect::<Vec<_>>();
        let state =
            ShadowPredictiveRegimeState::from_candidates(process_id, &config, &candidates).unwrap();
        let serialized = serde_json::to_vec(&state).unwrap();
        let restored: ShadowPredictiveRegimeState = serde_json::from_slice(&serialized).unwrap();
        assert!(!restored
            .candidate_is_newer(&config, &candidates[19])
            .unwrap());
        assert!(restored
            .candidate_is_newer(&config, &shadow_candidate(20, dec!(0.90), false))
            .unwrap());
        let resumed = ShadowPredictiveRegimeState::resume_from_state(
            &config,
            restored,
            &[shadow_candidate(20, dec!(0.90), false)],
        )
        .unwrap();
        assert!(resumed.degraded);
        assert_eq!(resumed.process_id, process_id);

        assert!(ShadowPredictiveRegimeState::resume_from_state(
            &config,
            resumed.clone(),
            &[shadow_candidate(20, dec!(0.90), false)]
        )
        .is_err());

        let mut altered_config = config.clone();
        altered_config.degradation_brier_score_threshold = dec!(0.24);
        assert!(
            ShadowPredictiveRegimeState::resume_from_state(&altered_config, resumed, &[]).is_err()
        );
    }
}
