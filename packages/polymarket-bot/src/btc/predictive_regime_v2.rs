use std::collections::{HashSet, VecDeque};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::{
    admission::{
        AdmissionDisposition, ShadowPredictiveRegimeCircuitBreakerConfig,
        ShadowPredictiveRegimeTransition, SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE,
        SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION,
    },
    types::BtcOutcome,
};

pub const SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_V2_SCHEMA_VERSION: &str =
    "shadow_predictive_regime_circuit_breaker_v2";

#[derive(Debug, Clone, PartialEq)]
pub enum ShadowPredictiveRegimeCircuitBreakerConfigSelector {
    V1(ShadowPredictiveRegimeCircuitBreakerConfig),
    V2(ShadowPredictiveRegimeCircuitBreakerV2Config),
}

impl ShadowPredictiveRegimeCircuitBreakerConfigSelector {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::V1(config) => config.validate(),
            Self::V2(config) => config.validate(),
        }
    }

    pub fn schema_version(&self) -> &str {
        match self {
            Self::V1(config) => &config.schema_version,
            Self::V2(config) => &config.schema_version,
        }
    }

    pub fn mode(&self) -> &str {
        match self {
            Self::V1(config) => &config.mode,
            Self::V2(config) => &config.mode,
        }
    }

    pub fn config_hash(&self) -> Result<String> {
        match self {
            Self::V1(config) => config.config_hash(),
            Self::V2(config) => config.config_hash(),
        }
    }

    pub fn as_v1(&self) -> Option<&ShadowPredictiveRegimeCircuitBreakerConfig> {
        match self {
            Self::V1(config) => Some(config),
            Self::V2(_) => None,
        }
    }

    pub fn as_v2(&self) -> Option<&ShadowPredictiveRegimeCircuitBreakerV2Config> {
        match self {
            Self::V1(_) => None,
            Self::V2(config) => Some(config),
        }
    }
}

impl From<ShadowPredictiveRegimeCircuitBreakerConfig>
    for ShadowPredictiveRegimeCircuitBreakerConfigSelector
{
    fn from(config: ShadowPredictiveRegimeCircuitBreakerConfig) -> Self {
        Self::V1(config)
    }
}

impl From<ShadowPredictiveRegimeCircuitBreakerV2Config>
    for ShadowPredictiveRegimeCircuitBreakerConfigSelector
{
    fn from(config: ShadowPredictiveRegimeCircuitBreakerV2Config) -> Self {
        Self::V2(config)
    }
}

impl Serialize for ShadowPredictiveRegimeCircuitBreakerConfigSelector {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::V1(config) => config.serialize(serializer),
            Self::V2(config) => config.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ShadowPredictiveRegimeCircuitBreakerConfigSelector {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let schema_version = value
            .get("schema_version")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                D::Error::custom("predictive-regime breaker schema_version is required")
            })?;
        match schema_version {
            SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION => {
                serde_json::from_value(value)
                    .map(Self::V1)
                    .map_err(D::Error::custom)
            }
            SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_V2_SCHEMA_VERSION => {
                serde_json::from_value(value)
                    .map(Self::V2)
                    .map_err(D::Error::custom)
            }
            other => Err(D::Error::custom(format!(
                "unsupported predictive-regime breaker schema_version {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowPredictiveRegimeCircuitBreakerV2Config {
    pub schema_version: String,
    pub mode: String,
    pub fast_resolved_market_window: u32,
    pub slow_resolved_market_window: u32,
    pub minimum_resolved_markets: u32,
    pub max_evidence_gap_seconds: u32,
    pub degradation_fast_brier_score_threshold: Decimal,
    pub degradation_fast_minus_slow_threshold: Decimal,
    pub degradation_slow_brier_score_threshold: Decimal,
    pub degradation_confirmation_markets: u32,
    pub recovery_fast_brier_score_threshold: Decimal,
    pub recovery_fast_minus_slow_ceiling: Decimal,
    pub recovery_confirmation_markets: u32,
}

impl ShadowPredictiveRegimeCircuitBreakerV2Config {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_V2_SCHEMA_VERSION {
            bail!(
                "predictive-regime circuit-breaker v2 schema_version must be {}",
                SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_V2_SCHEMA_VERSION
            );
        }
        if self.mode != SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE {
            bail!("predictive-regime circuit-breaker v2 mode must be shadow");
        }
        if !(2..=10_000).contains(&self.fast_resolved_market_window) {
            bail!("fast_resolved_market_window must be between 2 and 10000");
        }
        if !(20..=10_000).contains(&self.slow_resolved_market_window) {
            bail!("slow_resolved_market_window must be between 20 and 10000");
        }
        if self.fast_resolved_market_window > self.slow_resolved_market_window {
            bail!("fast_resolved_market_window cannot exceed slow_resolved_market_window");
        }
        if self.minimum_resolved_markets < 20
            || self.minimum_resolved_markets > self.slow_resolved_market_window
        {
            bail!(
                "minimum_resolved_markets must be at least 20 and cannot exceed slow_resolved_market_window"
            );
        }
        if !(300..=86_400).contains(&self.max_evidence_gap_seconds) {
            bail!("max_evidence_gap_seconds must be between 300 and 86400");
        }
        if !(1..=100).contains(&self.degradation_confirmation_markets) {
            bail!("degradation_confirmation_markets must be between 1 and 100");
        }
        if !(1..=100).contains(&self.recovery_confirmation_markets) {
            bail!("recovery_confirmation_markets must be between 1 and 100");
        }
        for (name, value) in [
            (
                "degradation_fast_brier_score_threshold",
                self.degradation_fast_brier_score_threshold,
            ),
            (
                "degradation_slow_brier_score_threshold",
                self.degradation_slow_brier_score_threshold,
            ),
            (
                "recovery_fast_brier_score_threshold",
                self.recovery_fast_brier_score_threshold,
            ),
        ] {
            if !(Decimal::ZERO..=Decimal::ONE).contains(&value) {
                bail!("{name} must be between 0 and 1");
            }
        }
        if !(Decimal::ZERO..=Decimal::ONE).contains(&self.degradation_fast_minus_slow_threshold) {
            bail!("degradation_fast_minus_slow_threshold must be between 0 and 1");
        }
        if !(Decimal::NEGATIVE_ONE..=Decimal::ZERO).contains(&self.recovery_fast_minus_slow_ceiling)
        {
            bail!("recovery_fast_minus_slow_ceiling must be between -1 and 0");
        }
        if self.recovery_fast_brier_score_threshold >= self.degradation_fast_brier_score_threshold {
            bail!(
                "recovery_fast_brier_score_threshold must be below degradation_fast_brier_score_threshold"
            );
        }
        if self.recovery_fast_minus_slow_ceiling >= self.degradation_fast_minus_slow_threshold {
            bail!(
                "recovery_fast_minus_slow_ceiling must be below degradation_fast_minus_slow_threshold"
            );
        }
        Ok(())
    }

    pub fn config_hash(&self) -> Result<String> {
        self.validate()?;
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)))
    }

    fn fast_window(&self) -> Result<usize> {
        usize::try_from(self.fast_resolved_market_window)
            .context("fast_resolved_market_window does not fit in memory")
    }

    fn slow_window(&self) -> Result<usize> {
        usize::try_from(self.slow_resolved_market_window)
            .context("slow_resolved_market_window does not fit in memory")
    }

    fn evidence_gap(&self) -> Duration {
        Duration::seconds(i64::from(self.max_evidence_gap_seconds))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowPredictiveRegimeV2CandidateSource {
    ActualPaperFill,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowPredictiveRegimeV2Candidate {
    pub market_id: String,
    pub decision_id: Uuid,
    pub snapshot_id: Uuid,
    pub order_id: String,
    pub fill_id: Uuid,
    pub source: ShadowPredictiveRegimeV2CandidateSource,
    pub decision_outcome: BtcOutcome,
    pub resolved_outcome: BtcOutcome,
    pub selected_point_probability: Decimal,
    pub decision_at: DateTime<Utc>,
    pub fill_at: DateTime<Utc>,
    pub label_available_at: DateTime<Utc>,
}

impl ShadowPredictiveRegimeV2Candidate {
    pub fn won(&self) -> bool {
        self.decision_outcome == self.resolved_outcome
    }

    pub fn validate(&self) -> Result<()> {
        if self.market_id.trim().is_empty() {
            bail!("predictive-regime v2 candidate market_id cannot be empty");
        }
        if self.decision_id.is_nil() || self.snapshot_id.is_nil() || self.fill_id.is_nil() {
            bail!("predictive-regime v2 candidate identifiers cannot be nil");
        }
        if self.order_id.trim().is_empty() {
            bail!("predictive-regime v2 candidate order_id cannot be empty");
        }
        if !(Decimal::ZERO..=Decimal::ONE).contains(&self.selected_point_probability) {
            bail!("selected_point_probability must be between 0 and 1");
        }
        if self.decision_at > self.fill_at || self.fill_at >= self.label_available_at {
            bail!("predictive-regime v2 candidate timestamps are not causal");
        }
        Ok(())
    }

    pub fn evidence_sha256(&self, process_id: Uuid) -> Result<String> {
        if process_id.is_nil() {
            bail!("predictive-regime v2 evidence process_id cannot be nil");
        }
        self.validate()?;
        #[derive(Serialize)]
        struct Evidence<'a> {
            process_id: Uuid,
            candidate: &'a ShadowPredictiveRegimeV2Candidate,
        }
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&Evidence {
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

    fn confidence_weighted_miss(&self) -> Decimal {
        if self.won() {
            Decimal::ZERO
        } else {
            self.selected_point_probability
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ShadowPredictiveRegimeV2Metrics {
    slow_count: u32,
    fast_count: u32,
    slow_wins: u32,
    mean_selected_point_probability: Decimal,
    empirical_accuracy: Decimal,
    slow_brier_score: Decimal,
    fast_brier_score: Decimal,
    fast_minus_slow_brier: Decimal,
    fast_confidence_weighted_miss: Decimal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowPredictiveRegimeV2State {
    pub process_id: Uuid,
    pub config_hash: String,
    pub degraded: bool,
    pub consecutive_degradation_markets: u32,
    pub consecutive_recovery_markets: u32,
    pub resolved_markets_observed: u64,
    pub slow_candidates: VecDeque<ShadowPredictiveRegimeV2Candidate>,
    pub fresh_evidence_count: u32,
    pub slow_probability_sum: Decimal,
    pub slow_win_count: u32,
    pub slow_brier_sum: Decimal,
    pub fast_brier_sum: Decimal,
    pub fast_confidence_weighted_miss_sum: Decimal,
    pub state_as_of_market_id: Option<String>,
    pub state_as_of_decision_id: Option<Uuid>,
    pub state_as_of_fill_id: Option<Uuid>,
    pub state_as_of_decision_at: Option<DateTime<Utc>>,
    pub state_as_of_label_available_at: Option<DateTime<Utc>>,
}

impl ShadowPredictiveRegimeV2State {
    pub fn new(
        process_id: Uuid,
        config: &ShadowPredictiveRegimeCircuitBreakerV2Config,
    ) -> Result<Self> {
        config.validate()?;
        if process_id.is_nil() {
            bail!("predictive-regime v2 state process_id cannot be nil");
        }
        Ok(Self {
            process_id,
            config_hash: config.config_hash()?,
            degraded: false,
            consecutive_degradation_markets: 0,
            consecutive_recovery_markets: 0,
            resolved_markets_observed: 0,
            slow_candidates: VecDeque::with_capacity(config.slow_window()?),
            fresh_evidence_count: 0,
            slow_probability_sum: Decimal::ZERO,
            slow_win_count: 0,
            slow_brier_sum: Decimal::ZERO,
            fast_brier_sum: Decimal::ZERO,
            fast_confidence_weighted_miss_sum: Decimal::ZERO,
            state_as_of_market_id: None,
            state_as_of_decision_id: None,
            state_as_of_fill_id: None,
            state_as_of_decision_at: None,
            state_as_of_label_available_at: None,
        })
    }

    pub fn from_candidates(
        process_id: Uuid,
        config: &ShadowPredictiveRegimeCircuitBreakerV2Config,
        candidates: &[ShadowPredictiveRegimeV2Candidate],
    ) -> Result<Self> {
        let mut state = Self::new(process_id, config)?;
        for candidate in candidates {
            state.apply_candidate(config, candidate)?;
        }
        Ok(state)
    }

    pub fn resume_from_state(
        config: &ShadowPredictiveRegimeCircuitBreakerV2Config,
        mut state: Self,
        candidates: &[ShadowPredictiveRegimeV2Candidate],
    ) -> Result<Self> {
        state.validate(config)?;
        for candidate in candidates {
            state.apply_candidate(config, candidate)?;
        }
        Ok(state)
    }

    pub fn candidate_is_newer(
        &self,
        config: &ShadowPredictiveRegimeCircuitBreakerV2Config,
        candidate: &ShadowPredictiveRegimeV2Candidate,
    ) -> Result<bool> {
        self.validate_runtime_header(config)?;
        candidate.validate()?;
        Ok(self
            .slow_candidates
            .back()
            .is_none_or(|last| candidate_ordering_key(candidate) > candidate_ordering_key(last)))
    }

    pub fn validate(&self, config: &ShadowPredictiveRegimeCircuitBreakerV2Config) -> Result<()> {
        self.validate_runtime_header(config)?;
        if self.resolved_markets_observed == 0 {
            if !self.slow_candidates.is_empty()
                || self.fresh_evidence_count != 0
                || self.slow_probability_sum != Decimal::ZERO
                || self.slow_win_count != 0
                || self.slow_brier_sum != Decimal::ZERO
                || self.fast_brier_sum != Decimal::ZERO
                || self.fast_confidence_weighted_miss_sum != Decimal::ZERO
                || self.degraded
                || self.consecutive_degradation_markets != 0
                || self.consecutive_recovery_markets != 0
                || self.state_as_of_market_id.is_some()
                || self.state_as_of_decision_id.is_some()
                || self.state_as_of_fill_id.is_some()
                || self.state_as_of_decision_at.is_some()
                || self.state_as_of_label_available_at.is_some()
            {
                bail!("empty predictive-regime v2 state is inconsistent");
            }
            return Ok(());
        }
        let last = self
            .slow_candidates
            .back()
            .context("observed predictive-regime v2 state has no final candidate")?;
        let mut market_ids = HashSet::new();
        let mut decision_ids = HashSet::new();
        let mut fill_ids = HashSet::new();
        let mut previous: Option<&ShadowPredictiveRegimeV2Candidate> = None;
        let mut slow_probability_sum = Decimal::ZERO;
        let mut slow_win_count = 0_u32;
        let mut slow_brier_sum = Decimal::ZERO;
        for candidate in &self.slow_candidates {
            candidate.validate()?;
            if !market_ids.insert(candidate.market_id.as_str())
                || !decision_ids.insert(candidate.decision_id)
                || !fill_ids.insert(candidate.fill_id)
            {
                bail!("predictive-regime v2 state contains duplicate evidence");
            }
            if previous.is_some_and(|prior| {
                candidate_ordering_key(candidate) <= candidate_ordering_key(prior)
            }) {
                bail!("predictive-regime v2 candidates are not in causal order");
            }
            slow_probability_sum += candidate.selected_point_probability;
            slow_win_count = slow_win_count.saturating_add(u32::from(candidate.won()));
            slow_brier_sum += candidate.brier_score();
            previous = Some(candidate);
        }
        let expected_fresh_count = self.expected_fresh_evidence_count(config)?;
        if self.fresh_evidence_count != expected_fresh_count {
            bail!("predictive-regime v2 fresh evidence count is inconsistent");
        }
        let fast_count = usize::try_from(self.fresh_evidence_count)
            .context("fresh predictive-regime v2 evidence count does not fit in memory")?;
        let expected_fast = self.slow_candidates.iter().rev().take(fast_count);
        let mut fast_brier_sum = Decimal::ZERO;
        let mut fast_confidence_weighted_miss_sum = Decimal::ZERO;
        for candidate in expected_fast {
            fast_brier_sum += candidate.brier_score();
            fast_confidence_weighted_miss_sum += candidate.confidence_weighted_miss();
        }
        if self.slow_probability_sum != slow_probability_sum
            || self.slow_win_count != slow_win_count
            || self.slow_brier_sum != slow_brier_sum
            || self.fast_brier_sum != fast_brier_sum
            || self.fast_confidence_weighted_miss_sum != fast_confidence_weighted_miss_sum
        {
            bail!("predictive-regime v2 rolling aggregates are inconsistent");
        }
        if self.state_as_of_market_id.as_deref() != Some(last.market_id.as_str())
            || self.state_as_of_decision_id != Some(last.decision_id)
            || self.state_as_of_fill_id != Some(last.fill_id)
            || self.state_as_of_decision_at != Some(last.decision_at)
            || self.state_as_of_label_available_at != Some(last.label_available_at)
        {
            bail!("predictive-regime v2 state cursor does not match rolling evidence");
        }
        Ok(())
    }

    fn validate_runtime_header(
        &self,
        config: &ShadowPredictiveRegimeCircuitBreakerV2Config,
    ) -> Result<()> {
        config.validate()?;
        if self.process_id.is_nil() {
            bail!("predictive-regime v2 state process_id cannot be nil");
        }
        if self.config_hash != config.config_hash()? {
            bail!("predictive-regime v2 state config hash does not match configuration");
        }
        if self.slow_candidates.len() > config.slow_window()?
            || self.fresh_evidence_count > config.fast_resolved_market_window
        {
            bail!("predictive-regime v2 state exceeds its configured windows");
        }
        if self.resolved_markets_observed
            < u64::try_from(self.slow_candidates.len()).unwrap_or(u64::MAX)
        {
            bail!("predictive-regime v2 state observation count is inconsistent");
        }
        if self.slow_win_count > u32::try_from(self.slow_candidates.len()).unwrap_or(u32::MAX)
            || self.slow_probability_sum < Decimal::ZERO
            || self.slow_brier_sum < Decimal::ZERO
            || self.fast_brier_sum < Decimal::ZERO
            || self.fast_confidence_weighted_miss_sum < Decimal::ZERO
        {
            bail!("predictive-regime v2 rolling aggregates are out of bounds");
        }
        if self.consecutive_degradation_markets > config.degradation_confirmation_markets
            || self.consecutive_recovery_markets > config.recovery_confirmation_markets
            || (self.consecutive_degradation_markets > 0 && self.consecutive_recovery_markets > 0)
        {
            bail!("predictive-regime v2 confirmation counters are inconsistent");
        }
        let confirmation_state_reachable = if self.degraded {
            (self.consecutive_degradation_markets == config.degradation_confirmation_markets
                && self.consecutive_recovery_markets == 0)
                || (self.consecutive_degradation_markets == 0
                    && self.consecutive_recovery_markets < config.recovery_confirmation_markets)
        } else {
            (self.consecutive_recovery_markets == config.recovery_confirmation_markets
                && self.consecutive_degradation_markets == 0)
                || (self.consecutive_recovery_markets == 0
                    && self.consecutive_degradation_markets
                        < config.degradation_confirmation_markets)
        };
        if !confirmation_state_reachable {
            bail!("predictive-regime v2 confirmation state is unreachable");
        }
        Ok(())
    }

    fn expected_fresh_evidence_count(
        &self,
        config: &ShadowPredictiveRegimeCircuitBreakerV2Config,
    ) -> Result<u32> {
        let mut count = 0_u32;
        let mut newer: Option<&ShadowPredictiveRegimeV2Candidate> = None;
        for candidate in self.slow_candidates.iter().rev() {
            if count >= config.fast_resolved_market_window {
                break;
            }
            if newer.is_some_and(|next| {
                next.label_available_at - candidate.label_available_at > config.evidence_gap()
            }) {
                break;
            }
            count = count
                .checked_add(1)
                .context("predictive-regime v2 fresh evidence count overflowed")?;
            newer = Some(candidate);
        }
        Ok(count)
    }

    pub fn apply_candidate(
        &mut self,
        config: &ShadowPredictiveRegimeCircuitBreakerV2Config,
        candidate: &ShadowPredictiveRegimeV2Candidate,
    ) -> Result<Option<ShadowPredictiveRegimeTransition>> {
        self.validate_runtime_header(config)?;
        candidate.validate()?;
        if let Some(last) = self.slow_candidates.back() {
            if candidate_ordering_key(candidate) <= candidate_ordering_key(last) {
                bail!("predictive-regime v2 candidate is duplicate or out of causal order");
            }
            if candidate.label_available_at - last.label_available_at > config.evidence_gap() {
                self.fresh_evidence_count = 0;
                self.fast_brier_sum = Decimal::ZERO;
                self.fast_confidence_weighted_miss_sum = Decimal::ZERO;
                self.consecutive_degradation_markets = 0;
                self.consecutive_recovery_markets = 0;
            }
        }

        self.resolved_markets_observed = self
            .resolved_markets_observed
            .checked_add(1)
            .context("predictive-regime v2 observation count overflowed")?;
        let fast_window = config.fast_window()?;
        if usize::try_from(self.fresh_evidence_count).unwrap_or(usize::MAX) >= fast_window {
            let leaving_fast_index = self
                .slow_candidates
                .len()
                .checked_sub(fast_window)
                .context("predictive-regime v2 fast window is inconsistent")?;
            let leaving_fast = self
                .slow_candidates
                .get(leaving_fast_index)
                .context("predictive-regime v2 fast window has no outgoing candidate")?;
            self.fast_brier_sum -= leaving_fast.brier_score();
            self.fast_confidence_weighted_miss_sum -= leaving_fast.confidence_weighted_miss();
        }
        if self.slow_candidates.len() >= config.slow_window()? {
            let leaving_slow = self
                .slow_candidates
                .pop_front()
                .context("predictive-regime v2 slow window has no outgoing candidate")?;
            self.slow_probability_sum -= leaving_slow.selected_point_probability;
            self.slow_win_count = self
                .slow_win_count
                .checked_sub(u32::from(leaving_slow.won()))
                .context("predictive-regime v2 slow win count underflowed")?;
            self.slow_brier_sum -= leaving_slow.brier_score();
        }
        self.slow_candidates.push_back(candidate.clone());
        self.slow_probability_sum += candidate.selected_point_probability;
        self.slow_win_count = self
            .slow_win_count
            .checked_add(u32::from(candidate.won()))
            .context("predictive-regime v2 slow win count overflowed")?;
        self.slow_brier_sum += candidate.brier_score();
        self.fast_brier_sum += candidate.brier_score();
        self.fast_confidence_weighted_miss_sum += candidate.confidence_weighted_miss();
        self.fresh_evidence_count = self
            .fresh_evidence_count
            .saturating_add(1)
            .min(config.fast_resolved_market_window);
        self.state_as_of_market_id = Some(candidate.market_id.clone());
        self.state_as_of_decision_id = Some(candidate.decision_id);
        self.state_as_of_fill_id = Some(candidate.fill_id);
        self.state_as_of_decision_at = Some(candidate.decision_at);
        self.state_as_of_label_available_at = Some(candidate.label_available_at);

        let Some(metrics) = self.current_metrics(config) else {
            self.consecutive_degradation_markets = 0;
            self.consecutive_recovery_markets = 0;
            return Ok(None);
        };
        let degradation_condition_met = degradation_condition(config, metrics);
        let recovery_condition_met = recovery_condition(config, metrics);

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
        config: &ShadowPredictiveRegimeCircuitBreakerV2Config,
        as_of: DateTime<Utc>,
    ) -> Result<ShadowPredictiveRegimeV2Evaluation> {
        self.validate_runtime_header(config)?;
        if self
            .state_as_of_label_available_at
            .is_some_and(|label_available_at| as_of < label_available_at)
        {
            bail!("predictive-regime v2 evaluation cannot precede its resolved evidence");
        }
        let metrics = self.current_metrics(config);
        let evidence_fresh = self
            .state_as_of_label_available_at
            .is_some_and(|label_at| as_of - label_at <= config.evidence_gap());
        let sample_ready = metrics.is_some() && evidence_fresh;
        let degradation_condition_met =
            sample_ready && metrics.is_some_and(|metrics| degradation_condition(config, metrics));
        let recovery_condition_met =
            sample_ready && metrics.is_some_and(|metrics| recovery_condition(config, metrics));
        let reason = if metrics.is_none() {
            "shadow_predictive_regime_v2_warming_up"
        } else if !evidence_fresh {
            "shadow_predictive_regime_v2_stale_evidence"
        } else if self.degraded && self.consecutive_recovery_markets > 0 {
            "shadow_predictive_regime_v2_degraded_recovery_pending"
        } else if self.degraded {
            "shadow_predictive_regime_v2_degraded"
        } else if self.consecutive_degradation_markets > 0 {
            "shadow_predictive_regime_v2_degradation_pending"
        } else {
            "shadow_predictive_regime_v2_healthy"
        };
        let state_evidence_sha256 = self.evidence_sha256(config)?;
        let config_hash = config.config_hash()?;
        let would_defer = self.degraded;

        #[derive(Serialize)]
        struct EvaluationEvidence<'a> {
            schema_version: &'a str,
            config_hash: &'a str,
            mode: &'a str,
            process_id: Uuid,
            as_of: DateTime<Utc>,
            state_evidence_sha256: &'a str,
            evidence_fresh: bool,
            degraded: bool,
            would_defer: bool,
            degradation_condition_met: bool,
            recovery_condition_met: bool,
            reason: &'a str,
        }
        let evaluation_evidence_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&EvaluationEvidence {
                schema_version: &config.schema_version,
                config_hash: &config_hash,
                mode: &config.mode,
                process_id: self.process_id,
                as_of,
                state_evidence_sha256: &state_evidence_sha256,
                evidence_fresh,
                degraded: self.degraded,
                would_defer,
                degradation_condition_met,
                recovery_condition_met,
                reason,
            })?)
        );

        Ok(ShadowPredictiveRegimeV2Evaluation {
            schema_version: config.schema_version.clone(),
            config_hash,
            mode: config.mode.clone(),
            process_id: self.process_id,
            as_of,
            shadow_only: true,
            sample_ready,
            evidence_fresh,
            slow_resolved_market_count: metrics.map(|metrics| metrics.slow_count),
            fast_resolved_market_count: metrics.map(|metrics| metrics.fast_count),
            resolved_markets_observed: self.resolved_markets_observed,
            rolling_wins: metrics.map(|metrics| metrics.slow_wins),
            mean_selected_point_probability: metrics
                .map(|metrics| metrics.mean_selected_point_probability),
            empirical_accuracy: metrics.map(|metrics| metrics.empirical_accuracy),
            brier_score: metrics.map(|metrics| metrics.slow_brier_score),
            slow_brier_score: metrics.map(|metrics| metrics.slow_brier_score),
            fast_brier_score: metrics.map(|metrics| metrics.fast_brier_score),
            fast_minus_slow_brier: metrics.map(|metrics| metrics.fast_minus_slow_brier),
            fast_confidence_weighted_miss: metrics
                .map(|metrics| metrics.fast_confidence_weighted_miss),
            degradation_condition_met,
            recovery_condition_met,
            degraded: self.degraded,
            consecutive_degradation_markets: self.consecutive_degradation_markets,
            consecutive_recovery_markets: self.consecutive_recovery_markets,
            would_defer,
            disposition: AdmissionDisposition::Allow,
            reason: reason.to_string(),
            state_as_of_market_id: self.state_as_of_market_id.clone(),
            state_as_of_decision_id: self.state_as_of_decision_id,
            state_as_of_fill_id: self.state_as_of_fill_id,
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
        config: &ShadowPredictiveRegimeCircuitBreakerV2Config,
    ) -> Result<String> {
        self.validate(config)?;
        #[derive(Serialize)]
        struct StateEvidence<'a> {
            schema_version: &'a str,
            config_hash: String,
            state: &'a ShadowPredictiveRegimeV2State,
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
        config: &ShadowPredictiveRegimeCircuitBreakerV2Config,
    ) -> Option<ShadowPredictiveRegimeV2Metrics> {
        let slow_count = u32::try_from(self.slow_candidates.len()).ok()?;
        if slow_count < config.minimum_resolved_markets
            || self.fresh_evidence_count < config.fast_resolved_market_window
        {
            return None;
        }
        let fast_count = self.fresh_evidence_count;
        let slow_denominator = Decimal::from(slow_count);
        let fast_denominator = Decimal::from(fast_count);
        let slow_wins = self.slow_win_count;
        let mean_selected_point_probability = self.slow_probability_sum / slow_denominator;
        let empirical_accuracy = Decimal::from(slow_wins) / slow_denominator;
        let slow_brier_score = self.slow_brier_sum / slow_denominator;
        let fast_brier_score = self.fast_brier_sum / fast_denominator;
        let fast_confidence_weighted_miss =
            self.fast_confidence_weighted_miss_sum / fast_denominator;
        Some(ShadowPredictiveRegimeV2Metrics {
            slow_count,
            fast_count,
            slow_wins,
            mean_selected_point_probability,
            empirical_accuracy,
            slow_brier_score,
            fast_brier_score,
            fast_minus_slow_brier: fast_brier_score - slow_brier_score,
            fast_confidence_weighted_miss,
        })
    }
}

fn degradation_condition(
    config: &ShadowPredictiveRegimeCircuitBreakerV2Config,
    metrics: ShadowPredictiveRegimeV2Metrics,
) -> bool {
    metrics.fast_brier_score >= config.degradation_fast_brier_score_threshold
        && (metrics.fast_minus_slow_brier >= config.degradation_fast_minus_slow_threshold
            || metrics.slow_brier_score >= config.degradation_slow_brier_score_threshold)
}

fn recovery_condition(
    config: &ShadowPredictiveRegimeCircuitBreakerV2Config,
    metrics: ShadowPredictiveRegimeV2Metrics,
) -> bool {
    metrics.fast_brier_score <= config.recovery_fast_brier_score_threshold
        && metrics.fast_minus_slow_brier <= config.recovery_fast_minus_slow_ceiling
}

fn candidate_ordering_key(
    candidate: &ShadowPredictiveRegimeV2Candidate,
) -> (DateTime<Utc>, DateTime<Utc>, Uuid, &str) {
    (
        candidate.label_available_at,
        candidate.fill_at,
        candidate.fill_id,
        candidate.market_id.as_str(),
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowPredictiveRegimeV2Evaluation {
    pub schema_version: String,
    pub config_hash: String,
    pub mode: String,
    pub process_id: Uuid,
    pub as_of: DateTime<Utc>,
    pub shadow_only: bool,
    pub sample_ready: bool,
    pub evidence_fresh: bool,
    pub slow_resolved_market_count: Option<u32>,
    pub fast_resolved_market_count: Option<u32>,
    pub resolved_markets_observed: u64,
    pub rolling_wins: Option<u32>,
    pub mean_selected_point_probability: Option<Decimal>,
    pub empirical_accuracy: Option<Decimal>,
    pub brier_score: Option<Decimal>,
    pub slow_brier_score: Option<Decimal>,
    pub fast_brier_score: Option<Decimal>,
    pub fast_minus_slow_brier: Option<Decimal>,
    pub fast_confidence_weighted_miss: Option<Decimal>,
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
    pub state_as_of_fill_id: Option<Uuid>,
    pub state_as_of_label_available_at: Option<DateTime<Utc>>,
    pub state_evidence_sha256: String,
    pub evaluation_evidence_sha256: String,
    pub state: ShadowPredictiveRegimeV2State,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub state_checkpoint_eligible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry_error: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub refresh_pending: bool,
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use rust_decimal_macros::dec;

    use super::*;

    fn v1_config() -> ShadowPredictiveRegimeCircuitBreakerConfig {
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

    fn v2_config() -> ShadowPredictiveRegimeCircuitBreakerV2Config {
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

    fn candidate(index: i64, probability: Decimal, won: bool) -> ShadowPredictiveRegimeV2Candidate {
        let label_available_at = Utc.with_ymd_and_hms(2026, 7, 22, 0, 0, 1).single().unwrap()
            + Duration::minutes(index * 5);
        ShadowPredictiveRegimeV2Candidate {
            market_id: format!("market-{index}"),
            decision_id: Uuid::from_u128(u128::try_from(index + 1).unwrap()),
            snapshot_id: Uuid::from_u128(u128::try_from(index + 10_001).unwrap()),
            order_id: format!("order-{index}"),
            fill_id: Uuid::from_u128(u128::try_from(index + 20_001).unwrap()),
            source: ShadowPredictiveRegimeV2CandidateSource::ActualPaperFill,
            decision_outcome: BtcOutcome::Up,
            resolved_outcome: if won {
                BtcOutcome::Up
            } else {
                BtcOutcome::Down
            },
            selected_point_probability: probability,
            decision_at: label_available_at - Duration::minutes(4),
            fill_at: label_available_at - Duration::minutes(3),
            label_available_at,
        }
    }

    #[test]
    fn selector_preserves_exact_v1_wire_contract_and_hash() {
        let selector = ShadowPredictiveRegimeCircuitBreakerConfigSelector::V1(v1_config());
        assert_eq!(
            serde_json::to_string(&selector).unwrap(),
            r#"{"schema_version":"shadow_predictive_regime_circuit_breaker_v1","mode":"shadow","rolling_resolved_market_window":20,"minimum_resolved_markets":20,"degradation_brier_score_threshold":"0.23","degradation_overconfidence_gap_threshold":"0.12","degradation_confirmation_markets":2,"recovery_brier_score_threshold":"0.21","recovery_overconfidence_gap_threshold":"0.05","recovery_confirmation_markets":2}"#
        );
        assert_eq!(
            selector.config_hash().unwrap(),
            "45545d1b10183a6c33eb559134575c739e93766fc06837cc5db95bc13ec0480f"
        );
        let restored: ShadowPredictiveRegimeCircuitBreakerConfigSelector =
            serde_json::from_str(&serde_json::to_string(&selector).unwrap()).unwrap();
        assert_eq!(restored, selector);
    }

    #[test]
    fn selector_dispatches_by_schema_and_rejects_mixed_contracts() {
        let v2 = serde_json::to_value(v2_config()).unwrap();
        assert!(matches!(
            serde_json::from_value::<ShadowPredictiveRegimeCircuitBreakerConfigSelector>(
                v2.clone()
            )
            .unwrap(),
            ShadowPredictiveRegimeCircuitBreakerConfigSelector::V2(_)
        ));

        let mut mixed = v2;
        mixed["rolling_resolved_market_window"] = serde_json::json!(20);
        assert!(
            serde_json::from_value::<ShadowPredictiveRegimeCircuitBreakerConfigSelector>(mixed)
                .is_err()
        );

        let mut wrong_schema = serde_json::to_value(v1_config()).unwrap();
        wrong_schema["schema_version"] = serde_json::json!("unknown");
        assert!(
            serde_json::from_value::<ShadowPredictiveRegimeCircuitBreakerConfigSelector>(
                wrong_schema
            )
            .is_err()
        );
    }

    #[test]
    fn v2_arms_and_recovers_from_fast_relative_brier_without_signed_gap() {
        let config = v2_config();
        let mut state = ShadowPredictiveRegimeV2State::new(Uuid::from_u128(900), &config).unwrap();

        for index in 0..20 {
            state
                .apply_candidate(&config, &candidate(index, dec!(0.50), index % 2 == 0))
                .unwrap();
        }
        assert!(!state.degraded);

        assert_eq!(
            state
                .apply_candidate(&config, &candidate(20, dec!(0.90), false))
                .unwrap(),
            None
        );
        assert_eq!(state.consecutive_degradation_markets, 1);
        assert_eq!(
            state
                .apply_candidate(&config, &candidate(21, dec!(0.90), false))
                .unwrap(),
            Some(ShadowPredictiveRegimeTransition::DegradationConfirmed)
        );
        let degraded = state
            .evaluate(&config, candidate(21, dec!(0.90), false).label_available_at)
            .unwrap();
        assert!(degraded.degraded);
        assert!(degraded.degradation_condition_met);
        assert_eq!(degraded.disposition, AdmissionDisposition::Allow);
        assert_eq!(degraded.fast_confidence_weighted_miss, Some(dec!(0.575)));

        for index in 22..25 {
            assert_eq!(
                state
                    .apply_candidate(&config, &candidate(index, dec!(0.99), true))
                    .unwrap(),
                None
            );
        }
        assert_eq!(state.consecutive_recovery_markets, 1);
        assert_eq!(
            state
                .apply_candidate(&config, &candidate(25, dec!(0.99), true))
                .unwrap(),
            Some(ShadowPredictiveRegimeTransition::RecoveryConfirmed)
        );
        assert!(!state.degraded);
    }

    #[test]
    fn v2_evidence_gap_resets_fast_confirmation_but_never_recovers_on_time_alone() {
        let config = v2_config();
        let mut state = ShadowPredictiveRegimeV2State::new(Uuid::from_u128(901), &config).unwrap();
        for index in 0..21 {
            state
                .apply_candidate(&config, &candidate(index, dec!(0.90), false))
                .unwrap();
        }
        assert!(state.degraded);

        let stale_as_of = state.state_as_of_label_available_at.unwrap() + Duration::minutes(16);
        let stale = state.evaluate(&config, stale_as_of).unwrap();
        assert!(!stale.evidence_fresh);
        assert!(!stale.sample_ready);
        assert!(stale.degraded);
        assert!(stale.would_defer);
        assert_eq!(stale.disposition, AdmissionDisposition::Allow);

        let mut post_gap = candidate(22, dec!(0.99), true);
        post_gap.label_available_at = stale_as_of + Duration::minutes(1);
        post_gap.decision_at = post_gap.label_available_at - Duration::minutes(4);
        post_gap.fill_at = post_gap.label_available_at - Duration::minutes(3);
        assert_eq!(state.apply_candidate(&config, &post_gap).unwrap(), None);
        assert_eq!(state.fresh_evidence_count, 1);
        assert_eq!(state.consecutive_recovery_markets, 0);
        assert!(
            !state
                .evaluate(&config, post_gap.label_available_at)
                .unwrap()
                .sample_ready
        );
        assert!(state.degraded);
    }

    #[test]
    fn v2_rolling_aggregates_remain_exact_across_eviction_and_evidence_gaps() {
        let config = v2_config();
        let mut state = ShadowPredictiveRegimeV2State::new(Uuid::from_u128(902), &config).unwrap();

        for index in 0..45 {
            let probability = match index % 3 {
                0 => dec!(0.55),
                1 => dec!(0.72),
                _ => dec!(0.91),
            };
            state
                .apply_candidate(&config, &candidate(index, probability, index % 4 != 0))
                .unwrap();
            state.validate(&config).unwrap();
        }
        assert_eq!(state.slow_candidates.len(), 20);
        assert_eq!(state.fresh_evidence_count, 4);

        let restored: ShadowPredictiveRegimeV2State =
            serde_json::from_value(serde_json::to_value(&state).unwrap()).unwrap();
        restored.validate(&config).unwrap();

        for index in 100..104 {
            state
                .apply_candidate(&config, &candidate(index, dec!(0.63), true))
                .unwrap();
            state.validate(&config).unwrap();
        }
        assert_eq!(state.fresh_evidence_count, 4);
        assert!(
            state
                .evaluate(&config, candidate(103, dec!(0.63), true).label_available_at)
                .unwrap()
                .sample_ready
        );
    }

    #[test]
    fn v2_state_validation_rejects_unreachable_confirmation_counters() {
        let config = v2_config();
        let mut healthy =
            ShadowPredictiveRegimeV2State::new(Uuid::from_u128(903), &config).unwrap();
        healthy.consecutive_degradation_markets = config.degradation_confirmation_markets;
        assert!(healthy.validate(&config).is_err());

        let mut degraded =
            ShadowPredictiveRegimeV2State::new(Uuid::from_u128(904), &config).unwrap();
        degraded.degraded = true;
        degraded.consecutive_recovery_markets = config.recovery_confirmation_markets;
        assert!(degraded.validate(&config).is_err());
    }
}
