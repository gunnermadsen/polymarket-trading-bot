use std::sync::OnceLock;

use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{
    decimal_from_f64, normal_cdf, positive, BtcFeatureSnapshot, BtcRejectReason, BtcStrategyConfig,
    FairValueEstimate,
};

pub(super) const STRATEGY_VERSION: &str =
    "btc_5m_chainlink_persistence_calibrated_fair_value_research_v1";
pub(super) const FEATURE_SCHEMA_VERSION: &str = "btc_5m_features_v3";
pub(super) const PROFILE_ID: &str = "btc5m-chainlink-persistence-calibrated-20260720-v1";
pub(super) const PROFILE_SHA256: &str =
    "238e7ed1b3676a3da61be3b3c8f9d48d7d05cad92e158b921846b9bfe495c1a3";

const PROFILE_ARTIFACT: &str =
    include_str!("profiles/btc5m-chainlink-persistence-calibrated-20260720-v1.json");
const PROFILE_STATUS: &str = "research_only";
const ESTIMATOR: &str = "chainlink_persistence_calibrated_fair_value_v1";
const PARAMETER_SOURCE: &str =
    "fixed_research_heuristic_informed_by_confirmed_vs_unconfirmed_analysis_20260719";
const CALIBRATION: &str = "fixed_logit_shrink_research_v1";
const UNCERTAINTY_POLICY: &str = "existing_btc_v1_plus_reliability_research_heuristic";

static COMPILED_PROFILE: OnceLock<Result<ChainlinkPersistenceCalibratedProfile, BtcRejectReason>> =
    OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ProfileSelection {
    profile_id: String,
    profile_sha256: String,
}

impl ProfileSelection {
    pub(super) fn new(profile_id: impl Into<String>, profile_sha256: impl Into<String>) -> Self {
        Self {
            profile_id: profile_id.into(),
            profile_sha256: profile_sha256.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ChainlinkPersistenceCalibratedProfile {
    profile_id: String,
    status: String,
    feature_schema_version: String,
    estimator: String,
    parameter_source: String,
    probability_floor: Decimal,
    evidence_input_clip: Decimal,
    external_evidence_parameters: ExternalEvidenceParameters,
    reliability_parameters: ReliabilityParameters,
    recent_evidence_parameters: RecentEvidenceParameters,
    calibration: String,
    calibration_parameters: CalibrationParameters,
    uncertainty_policy: String,
    uncertainty_parameters: UncertaintyParameters,
    research_evidence: ResearchEvidence,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalEvidenceParameters {
    volatility_floor_per_sqrt_second: Decimal,
    basis_lead_weight: Decimal,
    momentum_1s_weight: Decimal,
    momentum_5s_weight: Decimal,
    momentum_30s_weight: Decimal,
    max_lead_sigma_fraction: Decimal,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReliabilityParameters {
    intercept: Decimal,
    confirmation_sufficiency_strength: Decimal,
    confirmed_strength_weight: Decimal,
    unconfirmed_strength_penalty: Decimal,
    fading_penalty: Decimal,
    opposition_penalty: Decimal,
    volatility_reference_per_sqrt_second: Decimal,
    volatility_excess_penalty: Decimal,
    late_window_reference_seconds: Decimal,
    late_window_penalty: Decimal,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecentEvidenceParameters {
    persistence_weight: Decimal,
    confirmation_weight: Decimal,
    confirmed_conflict_weight: Decimal,
    confirmation_deficiency_weight: Decimal,
    max_abs_z_adjustment: Decimal,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibrationParameters {
    intercept: Decimal,
    slope: Decimal,
    max_abs_logit: Decimal,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct UncertaintyParameters {
    base_probability_uncertainty: Decimal,
    basis_uncertainty_weight: Decimal,
    feed_age_uncertainty_per_second: Decimal,
    max_baseline_probability_uncertainty: Decimal,
    reliability_uncertainty_weight: Decimal,
    volatility_excess_uncertainty_weight: Decimal,
    late_window_uncertainty_weight: Decimal,
    max_total_probability_uncertainty: Decimal,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResearchEvidence {
    analysis_cutoff: String,
    confirmed_state: String,
    unconfirmed_state: String,
    pooled_confirmed_wins: u32,
    pooled_confirmed_candidates: u32,
    pooled_unconfirmed_wins: u32,
    pooled_unconfirmed_candidates: u32,
    pooled_accuracy_delta_percentage_points: Decimal,
    exact_zero_confirmation_wins: u32,
    exact_zero_confirmation_candidates: u32,
    exact_zero_confirmation_realized_pnl_usd: Decimal,
    calibration_claim: String,
}

pub(super) fn resolve_profile(
    selection: &ProfileSelection,
) -> Result<&'static ChainlinkPersistenceCalibratedProfile, BtcRejectReason> {
    if selection.profile_id != PROFILE_ID || selection.profile_sha256 != PROFILE_SHA256 {
        return Err(BtcRejectReason::InvalidConfiguration);
    }
    match COMPILED_PROFILE.get_or_init(|| {
        if sha256_hex(PROFILE_ARTIFACT.as_bytes()) != PROFILE_SHA256 {
            return Err(BtcRejectReason::InvalidConfiguration);
        }
        decode_and_validate_profile(PROFILE_ARTIFACT)
    }) {
        Ok(profile) => Ok(profile),
        Err(reason) => Err(*reason),
    }
}

pub(super) fn estimate(
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
    profile: &ChainlinkPersistenceCalibratedProfile,
) -> Result<FairValueEstimate, BtcRejectReason> {
    validate_strategy_config(config, profile)?;

    let open = positive(
        snapshot.chainlink_open_price,
        BtcRejectReason::MissingChainlinkOpen,
    )?;
    let chainlink = positive(
        snapshot.chainlink_price,
        BtcRejectReason::MissingChainlinkPrice,
    )?;
    positive(snapshot.binance_price, BtcRejectReason::MissingBinancePrice)?;
    let chainlink_return_5s = snapshot
        .chainlink_return_5s
        .ok_or(BtcRejectReason::MissingChainlinkReturns)?;
    let return_1s = snapshot
        .binance_return_1s
        .ok_or(BtcRejectReason::MissingBinanceReturns)?;
    let return_5s = snapshot
        .binance_return_5s
        .ok_or(BtcRejectReason::MissingBinanceReturns)?;
    let return_30s = snapshot
        .binance_return_30s
        .ok_or(BtcRejectReason::MissingBinanceReturns)?;
    let volatility = positive(
        snapshot.realized_volatility,
        BtcRejectReason::MissingRealizedVolatility,
    )?
    .max(
        profile
            .external_evidence_parameters
            .volatility_floor_per_sqrt_second,
    );
    let basis_bps = snapshot
        .binance_chainlink_basis_bps
        .ok_or(BtcRejectReason::MissingBasis)?;

    let seconds_to_close_ms = (snapshot.window_end - snapshot.observed_at).num_milliseconds();
    if seconds_to_close_ms <= 0 {
        return Err(BtcRejectReason::OutsideEntryWindow);
    }
    let seconds_to_close = Decimal::from(seconds_to_close_ms) / dec!(1000);
    let tau = seconds_to_close_ms as f64 / 1000.0;
    let open_f64 = open.to_f64().ok_or(BtcRejectReason::InvalidFeatureValue)?;
    let chainlink_f64 = chainlink
        .to_f64()
        .ok_or(BtcRejectReason::InvalidFeatureValue)?;
    let chainlink_log_gap = decimal_from_f64((chainlink_f64 / open_f64).ln())?;
    let terminal_volatility = decimal_from_f64(
        volatility
            .to_f64()
            .ok_or(BtcRejectReason::InvalidFeatureValue)?
            * tau.sqrt(),
    )?;
    let horizon_volatility = decimal_from_f64(
        volatility
            .to_f64()
            .ok_or(BtcRejectReason::InvalidFeatureValue)?
            * 5.0_f64.sqrt(),
    )?;
    if terminal_volatility <= Decimal::ZERO || horizon_volatility <= Decimal::ZERO {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }

    let evidence = &profile.external_evidence_parameters;
    let raw_lead = evidence.basis_lead_weight * basis_bps / dec!(10000)
        + evidence.momentum_1s_weight * return_1s
        + evidence.momentum_5s_weight * return_5s
        + evidence.momentum_30s_weight * return_30s;
    let max_lead = terminal_volatility * evidence.max_lead_sigma_fraction;
    let lead_adjustment = raw_lead.clamp(-max_lead, max_lead);
    let level_z = (chainlink_log_gap + lead_adjustment) / terminal_volatility;
    let direction = if chainlink_log_gap > Decimal::ZERO {
        Decimal::ONE
    } else if chainlink_log_gap < Decimal::ZERO {
        -Decimal::ONE
    } else if lead_adjustment < Decimal::ZERO {
        -Decimal::ONE
    } else {
        Decimal::ONE
    };
    let input_clip = profile.evidence_input_clip;
    let persistence =
        (direction * chainlink_return_5s / horizon_volatility).clamp(-input_clip, input_clip);
    let confirmation = (direction * return_5s / horizon_volatility).clamp(-input_clip, input_clip);
    let strengthening = persistence.max(Decimal::ZERO);
    let same_direction_confirmation = confirmation.max(Decimal::ZERO);
    let confirmed_strength = strengthening * same_direction_confirmation;
    let confirmation_conflict = strengthening * (-confirmation).max(Decimal::ZERO);
    let fading = (-persistence).max(Decimal::ZERO);
    let opposition = (-confirmation).max(Decimal::ZERO);

    let reliability = &profile.reliability_parameters;
    let confirmation_deficiency_fraction = (Decimal::ONE
        - same_direction_confirmation / reliability.confirmation_sufficiency_strength)
        .clamp(Decimal::ZERO, Decimal::ONE);
    let confirmation_deficiency = strengthening * confirmation_deficiency_fraction;
    let volatility_excess = (volatility / reliability.volatility_reference_per_sqrt_second
        - Decimal::ONE)
        .clamp(Decimal::ZERO, input_clip);
    let late_window_pressure = (Decimal::ONE
        - seconds_to_close / reliability.late_window_reference_seconds)
        .clamp(Decimal::ZERO, Decimal::ONE);
    let reliability_logit = reliability.intercept
        + reliability.confirmed_strength_weight * confirmed_strength
        - reliability.unconfirmed_strength_penalty * confirmation_deficiency
        - reliability.fading_penalty * fading
        - reliability.opposition_penalty * opposition
        - reliability.volatility_excess_penalty * volatility_excess
        - reliability.late_window_penalty * late_window_pressure;
    let evidence_reliability = logistic(reliability_logit)?;

    let recent = &profile.recent_evidence_parameters;
    let recent_evidence_adjustment = (recent.persistence_weight * persistence
        + recent.confirmation_weight * confirmation
        + recent.confirmed_conflict_weight * (confirmed_strength - confirmation_conflict)
        - recent.confirmation_deficiency_weight * confirmation_deficiency)
        .clamp(-recent.max_abs_z_adjustment, recent.max_abs_z_adjustment);
    let effective_z = evidence_reliability * level_z + direction * recent_evidence_adjustment;
    let uncalibrated_up_probability = decimal_from_f64(normal_cdf(
        effective_z
            .to_f64()
            .ok_or(BtcRejectReason::InvalidFeatureValue)?,
    ))?
    .clamp(
        profile.probability_floor,
        Decimal::ONE - profile.probability_floor,
    );
    let up_probability = calibrate_probability(uncalibrated_up_probability, profile)?;

    let max_age_ms = snapshot
        .chainlink_age_ms
        .unwrap_or_default()
        .max(snapshot.binance_age_ms.unwrap_or_default())
        .max(0);
    let age_seconds = Decimal::from(max_age_ms) / dec!(1000);
    let uncertainty = &profile.uncertainty_parameters;
    let baseline_uncertainty = (uncertainty.base_probability_uncertainty
        + uncertainty.basis_uncertainty_weight * (basis_bps / dec!(10000)).abs()
        + uncertainty.feed_age_uncertainty_per_second * age_seconds)
        .clamp(
            Decimal::ZERO,
            uncertainty.max_baseline_probability_uncertainty,
        );
    let probability_uncertainty = (baseline_uncertainty
        + uncertainty.reliability_uncertainty_weight * (Decimal::ONE - evidence_reliability)
        + uncertainty.volatility_excess_uncertainty_weight * volatility_excess
        + uncertainty.late_window_uncertainty_weight * late_window_pressure)
        .clamp(Decimal::ZERO, uncertainty.max_total_probability_uncertainty);
    let up_lower_bound = (up_probability - probability_uncertainty).max(profile.probability_floor);
    let up_upper_bound =
        (up_probability + probability_uncertainty).min(Decimal::ONE - profile.probability_floor);

    Ok(FairValueEstimate {
        up_probability,
        down_probability: Decimal::ONE - up_probability,
        up_lower_bound,
        up_upper_bound,
        down_lower_bound: Decimal::ONE - up_upper_bound,
        down_upper_bound: Decimal::ONE - up_lower_bound,
        z_score: effective_z,
        chainlink_log_gap,
        lead_adjustment,
        terminal_volatility,
        probability_uncertainty,
        market_up_prior: None,
        signed_logit_adjustment: None,
        chainlink_persistence_score: Some(persistence),
        binance_confirmation_score: Some(confirmation),
        confirmation_deficiency_score: Some(confirmation_deficiency),
        evidence_reliability: Some(evidence_reliability),
        uncalibrated_up_probability: Some(uncalibrated_up_probability),
        path_recent_log_return_30s: None,
        path_prior_log_gap_30s: None,
        path_mature_gap_support: None,
        path_fresh_aligned_impulse: None,
        path_fresh_impulse_concentration: None,
        path_efficiency_score: None,
        path_choppiness_score: None,
        path_volatility_expansion_score: None,
        projected_terminal_gap: None,
        open_crossing_score: None,
        pre_path_up_probability: None,
        path_probability_delta: None,
        path_uncertainty_increment: None,
        estimator_id: None,
        estimator_profile_id: None,
        estimator_profile_sha256: None,
    })
}

pub(super) fn validate_strategy_config(
    config: &BtcStrategyConfig,
    profile: &ChainlinkPersistenceCalibratedProfile,
) -> Result<(), BtcRejectReason> {
    let evidence = &profile.external_evidence_parameters;
    let uncertainty = &profile.uncertainty_parameters;
    if config.feature_schema_version == profile.feature_schema_version
        && config.probability_floor == profile.probability_floor
        && config.volatility_floor_per_sqrt_second == evidence.volatility_floor_per_sqrt_second
        && config.basis_lead_weight == evidence.basis_lead_weight
        && config.momentum_1s_weight == evidence.momentum_1s_weight
        && config.momentum_5s_weight == evidence.momentum_5s_weight
        && config.momentum_30s_weight == evidence.momentum_30s_weight
        && config.max_lead_sigma_fraction == evidence.max_lead_sigma_fraction
        && config.base_probability_uncertainty == uncertainty.base_probability_uncertainty
        && config.basis_uncertainty_weight == uncertainty.basis_uncertainty_weight
        && config.feed_age_uncertainty_per_second == uncertainty.feed_age_uncertainty_per_second
        && config.max_probability_uncertainty == uncertainty.max_baseline_probability_uncertainty
    {
        Ok(())
    } else {
        Err(BtcRejectReason::InvalidConfiguration)
    }
}

fn calibrate_probability(
    probability: Decimal,
    profile: &ChainlinkPersistenceCalibratedProfile,
) -> Result<Decimal, BtcRejectReason> {
    let probability_f64 = probability
        .to_f64()
        .ok_or(BtcRejectReason::InvalidFeatureValue)?;
    let raw_logit = decimal_from_f64((probability_f64 / (1.0 - probability_f64)).ln())?;
    let calibration = &profile.calibration_parameters;
    let calibrated_logit = calibration.intercept
        + calibration.slope
            * raw_logit.clamp(-calibration.max_abs_logit, calibration.max_abs_logit);
    Ok(logistic(calibrated_logit)?.clamp(
        profile.probability_floor,
        Decimal::ONE - profile.probability_floor,
    ))
}

fn logistic(value: Decimal) -> Result<Decimal, BtcRejectReason> {
    let value = value.to_f64().ok_or(BtcRejectReason::InvalidFeatureValue)?;
    decimal_from_f64(1.0 / (1.0 + (-value).exp()))
}

fn decode_and_validate_profile(
    artifact: &str,
) -> Result<ChainlinkPersistenceCalibratedProfile, BtcRejectReason> {
    let profile: ChainlinkPersistenceCalibratedProfile =
        serde_json::from_str(artifact).map_err(|_| BtcRejectReason::InvalidConfiguration)?;
    let valid = profile.profile_id == PROFILE_ID
        && profile.status == PROFILE_STATUS
        && profile.feature_schema_version == FEATURE_SCHEMA_VERSION
        && profile.estimator == ESTIMATOR
        && profile.parameter_source == PARAMETER_SOURCE
        && profile.probability_floor == dec!(0.01)
        && profile.evidence_input_clip == dec!(3.0)
        && profile.external_evidence_parameters
            == (ExternalEvidenceParameters {
                volatility_floor_per_sqrt_second: dec!(0.00005),
                basis_lead_weight: dec!(0.25),
                momentum_1s_weight: dec!(0.05),
                momentum_5s_weight: dec!(0.10),
                momentum_30s_weight: dec!(0.10),
                max_lead_sigma_fraction: dec!(0.25),
            })
        && profile.reliability_parameters
            == (ReliabilityParameters {
                intercept: dec!(1.25),
                confirmation_sufficiency_strength: dec!(1.00),
                confirmed_strength_weight: dec!(1.00),
                unconfirmed_strength_penalty: dec!(1.50),
                fading_penalty: dec!(1.00),
                opposition_penalty: dec!(1.25),
                volatility_reference_per_sqrt_second: dec!(0.00012),
                volatility_excess_penalty: dec!(0.30),
                late_window_reference_seconds: dec!(120),
                late_window_penalty: dec!(0.15),
            })
        && profile.recent_evidence_parameters
            == (RecentEvidenceParameters {
                persistence_weight: dec!(0.10),
                confirmation_weight: dec!(0.15),
                confirmed_conflict_weight: dec!(0.05),
                confirmation_deficiency_weight: dec!(0.10),
                max_abs_z_adjustment: dec!(0.25),
            })
        && profile.calibration == CALIBRATION
        && profile.calibration_parameters
            == (CalibrationParameters {
                intercept: Decimal::ZERO,
                slope: dec!(0.85),
                max_abs_logit: dec!(5.0),
            })
        && profile.uncertainty_policy == UNCERTAINTY_POLICY
        && profile.uncertainty_parameters
            == (UncertaintyParameters {
                base_probability_uncertainty: dec!(0.015),
                basis_uncertainty_weight: Decimal::ONE,
                feed_age_uncertainty_per_second: dec!(0.002),
                max_baseline_probability_uncertainty: dec!(0.10),
                reliability_uncertainty_weight: dec!(0.04),
                volatility_excess_uncertainty_weight: dec!(0.01),
                late_window_uncertainty_weight: dec!(0.01),
                max_total_probability_uncertainty: dec!(0.15),
            })
        && profile.research_evidence
            == (ResearchEvidence {
                analysis_cutoff: "2026-07-19T22:55:00Z".to_string(),
                confirmed_state: "chainlink_gap_strengthening_and_binance_same_direction"
                    .to_string(),
                unconfirmed_state: "chainlink_gap_strengthening_without_binance_same_direction"
                    .to_string(),
                pooled_confirmed_wins: 54,
                pooled_confirmed_candidates: 79,
                pooled_unconfirmed_wins: 32,
                pooled_unconfirmed_candidates: 70,
                pooled_accuracy_delta_percentage_points: dec!(22.64),
                exact_zero_confirmation_wins: 11,
                exact_zero_confirmation_candidates: 42,
                exact_zero_confirmation_realized_pnl_usd: dec!(-35.27),
                calibration_claim: "none_fixed_research_heuristic".to_string(),
            });
    if valid {
        Ok(profile)
    } else {
        Err(BtcRejectReason::InvalidConfiguration)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, TimeZone, Utc};
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    use super::*;
    use crate::btc::{
        BtcFeatureLineage, BtcInputWindowLineage, BtcOutcome, BtcOutcomeBookFeatures,
        FeedIntegrityStatus, BTC_FEATURE_LINEAGE_VERSION,
    };

    fn config() -> BtcStrategyConfig {
        BtcStrategyConfig {
            strategy_version: STRATEGY_VERSION.to_string(),
            feature_schema_version: FEATURE_SCHEMA_VERSION.to_string(),
            ..BtcStrategyConfig::default()
        }
    }

    fn profile() -> &'static ChainlinkPersistenceCalibratedProfile {
        resolve_profile(&ProfileSelection::new(PROFILE_ID, PROFILE_SHA256)).unwrap()
    }

    fn snapshot() -> BtcFeatureSnapshot {
        let window_start = Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap();
        let observed_at = window_start + Duration::seconds(180);
        let source_at = observed_at - Duration::milliseconds(100);
        BtcFeatureSnapshot {
            snapshot_id: Uuid::from_u128(301),
            process_id: Uuid::from_u128(302),
            observed_at,
            feature_schema_version: FEATURE_SCHEMA_VERSION.to_string(),
            market_id: "btc-persistence-research".to_string(),
            event_slug: "btc-updown-5m-1784548800".to_string(),
            window_start,
            window_end: window_start + Duration::seconds(300),
            market_active: true,
            market_closed: false,
            accepting_orders: true,
            tick_size: dec!(0.01),
            minimum_order_size: Some(dec!(1)),
            resolution_source: "chainlink btc/usd".to_string(),
            chainlink_open_price: Some(dec!(100000)),
            chainlink_price: Some(dec!(100100)),
            binance_price: Some(dec!(100110)),
            chainlink_gap_bps: Some(dec!(10)),
            chainlink_return_5s: Some(dec!(0.00020)),
            chainlink_return_15s: None,
            chainlink_return_30s: None,
            chainlink_path_efficiency_30s: None,
            chainlink_path_tick_count_30s: None,
            chainlink_realized_volatility_5s: None,
            chainlink_realized_volatility_30s: None,
            binance_return_1s: Some(dec!(0.00005)),
            binance_return_5s: Some(dec!(0.00020)),
            binance_return_30s: Some(dec!(0.00030)),
            realized_volatility: Some(dec!(0.00012)),
            binance_chainlink_basis_bps: Some(dec!(1)),
            chainlink_age_ms: Some(100),
            binance_age_ms: Some(100),
            source_skew_ms: Some(25),
            chainlink_quality_ok: true,
            binance_quality_ok: true,
            up_book: book(BtcOutcome::Up, "up", dec!(0.50), dec!(0.51)),
            down_book: book(BtcOutcome::Down, "down", dec!(0.48), dec!(0.49)),
            fees_enabled: true,
            fee_rate: Some(dec!(0.03)),
            fee_rate_observed_at: Some(observed_at - Duration::minutes(1)),
            lineage: BtcFeatureLineage {
                lineage_version: BTC_FEATURE_LINEAGE_VERSION.to_string(),
                chainlink_open_tick_id: Some(Uuid::from_u128(303)),
                chainlink_tick_id: Some(Uuid::from_u128(304)),
                binance_tick_id: Some(Uuid::from_u128(305)),
                up_book_checkpoint_id: Some(Uuid::from_u128(306)),
                down_book_checkpoint_id: Some(Uuid::from_u128(307)),
                chainlink_open_source_timestamp: Some(window_start),
                chainlink_open_received_at: Some(window_start),
                chainlink_source_timestamp: Some(source_at),
                chainlink_received_at: Some(source_at),
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
                binance_source_timestamp: Some(source_at),
                binance_received_at: Some(source_at),
                chainlink_ingest_sequence: Some(10),
                chainlink_open_ingest_sequence: Some(9),
                binance_ingest_sequence: Some(11),
                up_book_ingest_sequence: Some(12),
                down_book_ingest_sequence: Some(13),
                up_book_connection_id: Some(Uuid::from_u128(308)),
                down_book_connection_id: Some(Uuid::from_u128(308)),
                chainlink_history: BtcInputWindowLineage::default(),
                binance_history: BtcInputWindowLineage::default(),
            },
        }
    }

    fn book(
        outcome: BtcOutcome,
        token_id: &str,
        bid: Decimal,
        ask: Decimal,
    ) -> BtcOutcomeBookFeatures {
        let observed_at = Utc.with_ymd_and_hms(2026, 7, 20, 12, 3, 0).unwrap();
        BtcOutcomeBookFeatures {
            outcome,
            token_id: token_id.to_string(),
            best_bid: Some(bid),
            best_ask: Some(ask),
            executable_ask_vwap: Some(ask),
            marketable_limit_price: Some(ask),
            quoted_size: dec!(5),
            bid_depth: dec!(100),
            ask_depth: dec!(100),
            imbalance: Some(Decimal::ZERO),
            source_timestamp: Some(observed_at - Duration::milliseconds(100)),
            received_at: Some(observed_at - Duration::milliseconds(100)),
            age_ms: Some(100),
            integrity_status: FeedIntegrityStatus::Ok,
            connection_id: Some(Uuid::from_u128(308)),
        }
    }

    #[test]
    fn profile_identity_hash_and_schema_fail_closed() {
        assert_eq!(sha256_hex(PROFILE_ARTIFACT.as_bytes()), PROFILE_SHA256);
        assert!(resolve_profile(&ProfileSelection::new(PROFILE_ID, PROFILE_SHA256)).is_ok());
        assert_eq!(
            resolve_profile(&ProfileSelection::new("wrong", PROFILE_SHA256)),
            Err(BtcRejectReason::InvalidConfiguration)
        );
        assert_eq!(
            resolve_profile(&ProfileSelection::new(PROFILE_ID, "0".repeat(64))),
            Err(BtcRejectReason::InvalidConfiguration)
        );
        let wrong_schema = PROFILE_ARTIFACT.replace(FEATURE_SCHEMA_VERSION, "wrong_features");
        assert_eq!(
            decode_and_validate_profile(&wrong_schema),
            Err(BtcRejectReason::InvalidConfiguration)
        );
    }

    #[test]
    fn confirmation_preserves_more_confidence_than_conflict() {
        let confirmed = estimate(&config(), &snapshot(), profile()).unwrap();
        let mut conflict_snapshot = snapshot();
        conflict_snapshot.binance_return_5s = Some(dec!(-0.00020));
        let conflict = estimate(&config(), &conflict_snapshot, profile()).unwrap();

        assert!(confirmed.evidence_reliability > conflict.evidence_reliability);
        assert!(confirmed.up_probability > conflict.up_probability);
        assert!(confirmed.probability_uncertainty < conflict.probability_uncertainty);
    }

    #[test]
    fn exact_zero_confirmation_is_continuously_treated_as_unconfirmed() {
        let confirmed = estimate(&config(), &snapshot(), profile()).unwrap();
        let mut zero_confirmation_snapshot = snapshot();
        zero_confirmation_snapshot.binance_return_5s = Some(Decimal::ZERO);
        let unconfirmed = estimate(&config(), &zero_confirmation_snapshot, profile()).unwrap();

        assert_eq!(unconfirmed.binance_confirmation_score, Some(Decimal::ZERO));
        assert!(unconfirmed.confirmation_deficiency_score > Some(Decimal::ZERO));
        assert!(
            confirmed.confirmation_deficiency_score < unconfirmed.confirmation_deficiency_score
        );
        assert!(confirmed.evidence_reliability > unconfirmed.evidence_reliability);
        assert!(confirmed.up_probability > unconfirmed.up_probability);
        assert!(confirmed.probability_uncertainty < unconfirmed.probability_uncertainty);
    }

    #[test]
    fn fading_chainlink_evidence_reduces_confidence() {
        let strengthening = estimate(&config(), &snapshot(), profile()).unwrap();
        let mut fading_snapshot = snapshot();
        fading_snapshot.chainlink_return_5s = Some(dec!(-0.00020));
        let fading = estimate(&config(), &fading_snapshot, profile()).unwrap();

        assert!(strengthening.evidence_reliability > fading.evidence_reliability);
        assert!(strengthening.up_probability > fading.up_probability);
        assert!(strengthening.probability_uncertainty < fading.probability_uncertainty);
    }

    #[test]
    fn estimator_is_deterministic_complementary_and_book_imbalance_independent() {
        let snapshot = snapshot();
        let first = estimate(&config(), &snapshot, profile()).unwrap();
        let second = estimate(&config(), &snapshot, profile()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.up_probability + first.down_probability, Decimal::ONE);
        assert_eq!(first.up_lower_bound + first.down_upper_bound, Decimal::ONE);
        assert_eq!(first.up_upper_bound + first.down_lower_bound, Decimal::ONE);
        assert!(first.up_lower_bound <= first.up_probability);
        assert!(first.up_probability <= first.up_upper_bound);

        let mut changed = snapshot;
        changed.up_book.imbalance = Some(dec!(-0.99));
        changed.down_book.imbalance = Some(dec!(0.99));
        assert_eq!(first, estimate(&config(), &changed, profile()).unwrap());
    }
}
