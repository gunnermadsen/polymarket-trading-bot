use anyhow::{bail, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::types::BtcOutcome;

pub const LOSS_REGIME_CONFIDENCE_FLOOR_SCHEMA_VERSION: &str = "loss_regime_confidence_floor_v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BtcEntryAdmissionConfig {
    pub loss_regime_confidence_floor: LossRegimeConfidenceFloorConfig,
}

impl BtcEntryAdmissionConfig {
    pub fn validate(&self) -> Result<()> {
        self.loss_regime_confidence_floor.validate()
    }
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

#[cfg(test)]
mod tests {
    use chrono::Duration;
    use rust_decimal_macros::dec;

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
}
