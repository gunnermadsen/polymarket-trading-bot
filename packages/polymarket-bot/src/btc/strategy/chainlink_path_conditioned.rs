use std::sync::OnceLock;

use chrono::{DateTime, Duration, Utc};
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
    "btc_5m_chainlink_path_conditioned_fair_value_research_v1";
pub(super) const FEATURE_SCHEMA_VERSION: &str = "btc_5m_features_v4";
pub(super) const FEATURE_LINEAGE_VERSION: &str = "btc_5m_feature_lineage_v3";
pub(super) const PROFILE_ID: &str = "btc5m-chainlink-path-conditioned-20260720-v1";
pub(super) const PROFILE_SHA256: &str =
    "f18648c68d602093dd6d1c47ea0fb3ca3833a7492623bf0641fe0980c054ffe7";

const PROFILE_ARTIFACT: &str =
    include_str!("profiles/btc5m-chainlink-path-conditioned-20260720-v1.json");
const PROFILE_STATUS: &str = "research_only";
const ESTIMATOR: &str = "chainlink_path_conditioned_fair_value_v1";
const PARAMETER_SOURCE: &str = "fixed_research_heuristic_path_conditioning_hypothesis_20260720";
const CALIBRATION: &str = "fixed_logit_shrink_research_v1";
const UNCERTAINTY_POLICY: &str =
    "existing_btc_v1_plus_reliability_and_path_uncertainty_research_heuristic";

static COMPILED_PROFILE: OnceLock<Result<ChainlinkPathConditionedProfile, BtcRejectReason>> =
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
pub(super) struct ChainlinkPathConditionedProfile {
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
    path_parameters: PathParameters,
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
struct PathParameters {
    recent_path_lookback_seconds: i64,
    short_reversal_lookback_seconds: i64,
    max_anchor_lag_ms: i64,
    minimum_path_tick_count: usize,
    volatility_scaled_gap_floor_sigma_fraction: Decimal,
    mature_gap_weight: Decimal,
    fresh_aligned_impulse_weight: Decimal,
    opposing_impulse_weight: Decimal,
    minimum_path_gap_scale: Decimal,
    fresh_impulse_reliability_penalty: Decimal,
    choppiness_reliability_penalty: Decimal,
    volatility_expansion_reliability_penalty: Decimal,
    max_volatility_expansion: Decimal,
    max_abs_crossing_score: Decimal,
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
    max_pre_path_probability_uncertainty: Decimal,
    fresh_impulse_uncertainty_weight: Decimal,
    choppiness_uncertainty_weight: Decimal,
    volatility_expansion_uncertainty_weight: Decimal,
    max_total_probability_uncertainty: Decimal,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResearchEvidence {
    base_profile_id: String,
    base_profile_sha256: String,
    source_process_id: String,
    analysis_cutoff: String,
    target_failure_mode: String,
    parameter_claim: String,
    validation_requirement: String,
    calibration_claim: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct EvidenceScores {
    persistence: Decimal,
    confirmation: Decimal,
    confirmation_deficiency: Decimal,
    volatility_excess: Decimal,
    late_window_pressure: Decimal,
    reliability_logit: Decimal,
    recent_adjustment: Decimal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PathProjection {
    recent_log_return_30s: Decimal,
    prior_log_gap_30s: Decimal,
    mature_gap_support: Decimal,
    fresh_aligned_impulse: Decimal,
    fresh_impulse_concentration: Decimal,
    path_efficiency: Decimal,
    choppiness: Decimal,
    volatility_expansion: Decimal,
    projected_terminal_gap: Decimal,
    open_crossing_score: Decimal,
}

pub(super) fn resolve_profile(
    selection: &ProfileSelection,
) -> Result<&'static ChainlinkPathConditionedProfile, BtcRejectReason> {
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

pub(super) fn validate_strategy_config(
    config: &BtcStrategyConfig,
    profile: &ChainlinkPathConditionedProfile,
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

pub(super) fn validate_snapshot(
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
    profile: &ChainlinkPathConditionedProfile,
) -> Result<(), BtcRejectReason> {
    if config.feature_schema_version != FEATURE_SCHEMA_VERSION
        || snapshot.feature_schema_version != FEATURE_SCHEMA_VERSION
    {
        return Err(BtcRejectReason::FeatureSchemaMismatch);
    }
    let path = &profile.path_parameters;
    let returns = [
        snapshot
            .chainlink_return_5s
            .ok_or(BtcRejectReason::MissingChainlinkReturns)?,
        snapshot
            .chainlink_return_15s
            .ok_or(BtcRejectReason::MissingChainlinkReturns)?,
        snapshot
            .chainlink_return_30s
            .ok_or(BtcRejectReason::MissingChainlinkReturns)?,
    ];
    if returns.iter().any(|value| *value <= -Decimal::ONE) {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    let efficiency = snapshot
        .chainlink_path_efficiency_30s
        .ok_or(BtcRejectReason::MissingChainlinkReturns)?;
    let tick_count = snapshot
        .chainlink_path_tick_count_30s
        .ok_or(BtcRejectReason::MissingLineage)?;
    let short_volatility = snapshot
        .chainlink_realized_volatility_5s
        .ok_or(BtcRejectReason::MissingRealizedVolatility)?;
    let long_volatility = snapshot
        .chainlink_realized_volatility_30s
        .ok_or(BtcRejectReason::MissingRealizedVolatility)?;
    if !(Decimal::ZERO..=Decimal::ONE).contains(&efficiency)
        || tick_count < path.minimum_path_tick_count
        || short_volatility < Decimal::ZERO
        || long_volatility < Decimal::ZERO
    {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    let lineage = &snapshot.lineage;
    if lineage.lineage_version != FEATURE_LINEAGE_VERSION {
        return Err(BtcRejectReason::MissingLineage);
    }
    let current_source = lineage
        .chainlink_source_timestamp
        .ok_or(BtcRejectReason::MissingLineage)?;
    let current_received = lineage
        .chainlink_received_at
        .ok_or(BtcRejectReason::MissingLineage)?;
    if current_source > snapshot.observed_at || current_received > snapshot.observed_at {
        return Err(BtcRejectReason::FutureInputTimestamp);
    }
    validate_anchor(
        snapshot.observed_at,
        current_source,
        path.short_reversal_lookback_seconds,
        path.max_anchor_lag_ms,
        lineage.chainlink_anchor_15s_tick_id.is_some(),
        lineage.chainlink_anchor_15s_source_timestamp,
        lineage.chainlink_anchor_15s_received_at,
        lineage.chainlink_anchor_15s_ingest_sequence.is_some(),
        lineage.chainlink_anchor_15s_effective_lookback_ms,
    )?;
    validate_anchor(
        snapshot.observed_at,
        current_source,
        path.recent_path_lookback_seconds,
        path.max_anchor_lag_ms,
        lineage.chainlink_anchor_30s_tick_id.is_some(),
        lineage.chainlink_anchor_30s_source_timestamp,
        lineage.chainlink_anchor_30s_received_at,
        lineage.chainlink_anchor_30s_ingest_sequence.is_some(),
        lineage.chainlink_anchor_30s_effective_lookback_ms,
    )?;
    if lineage.chainlink_anchor_15s_source_timestamp < lineage.chainlink_anchor_30s_source_timestamp
    {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_anchor(
    observed_at: DateTime<Utc>,
    current_source: DateTime<Utc>,
    lookback_seconds: i64,
    max_anchor_lag_ms: i64,
    has_tick_id: bool,
    anchor_source: Option<DateTime<Utc>>,
    anchor_received: Option<DateTime<Utc>>,
    has_ingest_sequence: bool,
    effective_lookback_ms: Option<i64>,
) -> Result<(), BtcRejectReason> {
    if !has_tick_id || !has_ingest_sequence {
        return Err(BtcRejectReason::MissingLineage);
    }
    let source = anchor_source.ok_or(BtcRejectReason::MissingLineage)?;
    let received = anchor_received.ok_or(BtcRejectReason::MissingLineage)?;
    let effective = effective_lookback_ms.ok_or(BtcRejectReason::MissingLineage)?;
    if source > observed_at || received > observed_at {
        return Err(BtcRejectReason::FutureInputTimestamp);
    }
    let target = observed_at - Duration::seconds(lookback_seconds);
    let anchor_lag_ms = (target - source).num_milliseconds();
    let expected_effective = (current_source - source).num_milliseconds();
    if !(0..=max_anchor_lag_ms).contains(&anchor_lag_ms)
        || effective <= 0
        || effective != expected_effective
    {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    Ok(())
}

pub(super) fn estimate(
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
    profile: &ChainlinkPathConditionedProfile,
) -> Result<FairValueEstimate, BtcRejectReason> {
    validate_strategy_config(config, profile)?;
    validate_snapshot(config, snapshot, profile)?;

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
    let chainlink_return_15s = snapshot
        .chainlink_return_15s
        .ok_or(BtcRejectReason::MissingChainlinkReturns)?;
    let chainlink_return_30s = snapshot
        .chainlink_return_30s
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
    let evidence = &profile.external_evidence_parameters;
    let volatility = positive(
        snapshot.realized_volatility,
        BtcRejectReason::MissingRealizedVolatility,
    )?
    .max(evidence.volatility_floor_per_sqrt_second);
    let short_path_volatility = snapshot
        .chainlink_realized_volatility_5s
        .ok_or(BtcRejectReason::MissingRealizedVolatility)?;
    let long_path_volatility = snapshot
        .chainlink_realized_volatility_30s
        .ok_or(BtcRejectReason::MissingRealizedVolatility)?;
    let path_efficiency = snapshot
        .chainlink_path_efficiency_30s
        .ok_or(BtcRejectReason::MissingChainlinkReturns)?;
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
    let recent_log_return_15s = log_one_plus(chainlink_return_15s)?;
    let recent_log_return_30s = log_one_plus(chainlink_return_30s)?;
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

    let raw_lead = evidence.basis_lead_weight * basis_bps / dec!(10000)
        + evidence.momentum_1s_weight * return_1s
        + evidence.momentum_5s_weight * return_5s
        + evidence.momentum_30s_weight * return_30s;
    let max_lead = terminal_volatility * evidence.max_lead_sigma_fraction;
    let lead_adjustment = raw_lead.clamp(-max_lead, max_lead);
    let current_direction = direction_for(chainlink_log_gap, lead_adjustment);
    let projection = project_path(
        chainlink_log_gap,
        recent_log_return_15s,
        recent_log_return_30s,
        path_efficiency,
        short_path_volatility,
        long_path_volatility,
        lead_adjustment,
        current_direction,
        profile,
    )?;

    let baseline_scores = evidence_scores(
        current_direction,
        chainlink_return_5s,
        return_5s,
        volatility,
        horizon_volatility,
        seconds_to_close,
        profile,
    )?;
    let baseline_reliability = logistic(baseline_scores.reliability_logit)?;
    let baseline_level_z = (chainlink_log_gap + lead_adjustment) / terminal_volatility;
    let baseline_effective_z = baseline_reliability * baseline_level_z
        + current_direction * baseline_scores.recent_adjustment;
    let baseline_uncalibrated = normal_probability(baseline_effective_z, profile)?;
    let pre_path_up_probability = calibrate_probability(baseline_uncalibrated, profile)?;

    let projected_direction = direction_for(projection.projected_terminal_gap, Decimal::ZERO);
    let path_scores = evidence_scores(
        projected_direction,
        chainlink_return_5s,
        return_5s,
        volatility,
        horizon_volatility,
        seconds_to_close,
        profile,
    )?;
    let path = &profile.path_parameters;
    let path_reliability_logit = path_scores.reliability_logit
        - path.fresh_impulse_reliability_penalty * projection.fresh_impulse_concentration
        - path.choppiness_reliability_penalty * projection.choppiness
        - path.volatility_expansion_reliability_penalty * projection.volatility_expansion;
    let evidence_reliability = logistic(path_reliability_logit)?;
    let base_path_z = projection.projected_terminal_gap / terminal_volatility;
    let effective_z = if projection.projected_terminal_gap == Decimal::ZERO {
        Decimal::ZERO
    } else {
        let effective_abs_z = (evidence_reliability * base_path_z.abs()
            + path_scores.recent_adjustment)
            .max(Decimal::ZERO);
        projected_direction * effective_abs_z
    };
    let uncalibrated_up_probability =
        path_probability(projection.projected_terminal_gap, effective_z, profile)?;
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
    let pre_path_probability_uncertainty = (baseline_uncertainty
        + uncertainty.reliability_uncertainty_weight * (Decimal::ONE - evidence_reliability)
        + uncertainty.volatility_excess_uncertainty_weight * path_scores.volatility_excess
        + uncertainty.late_window_uncertainty_weight * path_scores.late_window_pressure)
        .clamp(
            Decimal::ZERO,
            uncertainty.max_pre_path_probability_uncertainty,
        );
    let path_uncertainty_increment = uncertainty.fresh_impulse_uncertainty_weight
        * projection.fresh_impulse_concentration
        + uncertainty.choppiness_uncertainty_weight * projection.choppiness
        + uncertainty.volatility_expansion_uncertainty_weight * projection.volatility_expansion;
    let probability_uncertainty = (pre_path_probability_uncertainty + path_uncertainty_increment)
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
        chainlink_persistence_score: Some(path_scores.persistence),
        binance_confirmation_score: Some(path_scores.confirmation),
        confirmation_deficiency_score: Some(path_scores.confirmation_deficiency),
        evidence_reliability: Some(evidence_reliability),
        uncalibrated_up_probability: Some(uncalibrated_up_probability),
        path_recent_log_return_30s: Some(projection.recent_log_return_30s),
        path_prior_log_gap_30s: Some(projection.prior_log_gap_30s),
        path_mature_gap_support: Some(projection.mature_gap_support),
        path_fresh_aligned_impulse: Some(projection.fresh_aligned_impulse),
        path_fresh_impulse_concentration: Some(projection.fresh_impulse_concentration),
        path_efficiency_score: Some(projection.path_efficiency),
        path_choppiness_score: Some(projection.choppiness),
        path_volatility_expansion_score: Some(projection.volatility_expansion),
        projected_terminal_gap: Some(projection.projected_terminal_gap),
        open_crossing_score: Some(projection.open_crossing_score),
        pre_path_up_probability: Some(pre_path_up_probability),
        path_probability_delta: Some(up_probability - pre_path_up_probability),
        path_uncertainty_increment: Some(path_uncertainty_increment),
        estimator_id: Some(ESTIMATOR.to_string()),
        estimator_profile_id: Some(PROFILE_ID.to_string()),
        estimator_profile_sha256: Some(PROFILE_SHA256.to_string()),
    })
}

#[allow(clippy::too_many_arguments)]
fn project_path(
    chainlink_log_gap: Decimal,
    recent_log_return_15s: Decimal,
    recent_log_return_30s: Decimal,
    path_efficiency: Decimal,
    short_path_volatility: Decimal,
    long_path_volatility: Decimal,
    lead_adjustment: Decimal,
    current_direction: Decimal,
    profile: &ChainlinkPathConditionedProfile,
) -> Result<PathProjection, BtcRejectReason> {
    let path = &profile.path_parameters;
    let evidence = &profile.external_evidence_parameters;
    let long_volatility_floor = long_path_volatility.max(evidence.volatility_floor_per_sqrt_second);
    let sqrt_30 = decimal_from_f64(30.0_f64.sqrt())?;
    let sigma_30 = long_volatility_floor * sqrt_30;
    let denominator = chainlink_log_gap
        .abs()
        .max(sigma_30 * path.volatility_scaled_gap_floor_sigma_fraction);
    if denominator <= Decimal::ZERO {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }

    let prior_log_gap_30s = chainlink_log_gap - recent_log_return_30s;
    let mature_gap_support = (current_direction * prior_log_gap_30s).max(Decimal::ZERO);
    let fresh_aligned_impulse = (current_direction * recent_log_return_30s).max(Decimal::ZERO);
    let opposing_30s = (-current_direction * recent_log_return_30s).max(Decimal::ZERO);
    let opposing_15s = (-current_direction * recent_log_return_15s).max(Decimal::ZERO);
    let opposing_impulse = opposing_30s.max(opposing_15s);
    let fresh_impulse_concentration =
        (fresh_aligned_impulse / denominator).clamp(Decimal::ZERO, Decimal::ONE);

    let path_activity_scale = long_path_volatility * sqrt_30;
    let recent_activity = (recent_log_return_30s.abs().max(path_activity_scale) / denominator)
        .clamp(Decimal::ZERO, Decimal::ONE);
    let choppiness =
        ((Decimal::ONE - path_efficiency) * recent_activity).clamp(Decimal::ZERO, Decimal::ONE);
    let path_quality = Decimal::ONE - (Decimal::ONE - path.minimum_path_gap_scale) * choppiness;
    let support = (path.mature_gap_weight * mature_gap_support
        + path.fresh_aligned_impulse_weight * fresh_aligned_impulse)
        .min(chainlink_log_gap.abs());
    let projected_chainlink_gap = current_direction
        * (path_quality * support - path.opposing_impulse_weight * opposing_impulse);
    let projected_terminal_gap = projected_chainlink_gap + lead_adjustment;
    let open_crossing_score = (-current_direction * projected_terminal_gap / denominator)
        .clamp(-path.max_abs_crossing_score, path.max_abs_crossing_score);
    let volatility_expansion = (short_path_volatility / long_volatility_floor - Decimal::ONE)
        .clamp(Decimal::ZERO, path.max_volatility_expansion);

    Ok(PathProjection {
        recent_log_return_30s,
        prior_log_gap_30s,
        mature_gap_support,
        fresh_aligned_impulse,
        fresh_impulse_concentration,
        path_efficiency,
        choppiness,
        volatility_expansion,
        projected_terminal_gap,
        open_crossing_score,
    })
}

#[allow(clippy::too_many_arguments)]
fn evidence_scores(
    direction: Decimal,
    chainlink_return_5s: Decimal,
    binance_return_5s: Decimal,
    volatility: Decimal,
    horizon_volatility: Decimal,
    seconds_to_close: Decimal,
    profile: &ChainlinkPathConditionedProfile,
) -> Result<EvidenceScores, BtcRejectReason> {
    if direction.abs() != Decimal::ONE || horizon_volatility <= Decimal::ZERO {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    let input_clip = profile.evidence_input_clip;
    let persistence =
        (direction * chainlink_return_5s / horizon_volatility).clamp(-input_clip, input_clip);
    let confirmation =
        (direction * binance_return_5s / horizon_volatility).clamp(-input_clip, input_clip);
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

    let recent = &profile.recent_evidence_parameters;
    let recent_adjustment = (recent.persistence_weight * persistence
        + recent.confirmation_weight * confirmation
        + recent.confirmed_conflict_weight * (confirmed_strength - confirmation_conflict)
        - recent.confirmation_deficiency_weight * confirmation_deficiency)
        .clamp(-recent.max_abs_z_adjustment, recent.max_abs_z_adjustment);

    Ok(EvidenceScores {
        persistence,
        confirmation,
        confirmation_deficiency,
        volatility_excess,
        late_window_pressure,
        reliability_logit,
        recent_adjustment,
    })
}

fn direction_for(primary: Decimal, fallback: Decimal) -> Decimal {
    if primary > Decimal::ZERO {
        Decimal::ONE
    } else if primary < Decimal::ZERO {
        -Decimal::ONE
    } else if fallback < Decimal::ZERO {
        -Decimal::ONE
    } else {
        Decimal::ONE
    }
}

fn log_one_plus(value: Decimal) -> Result<Decimal, BtcRejectReason> {
    if value <= -Decimal::ONE {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    let value = value.to_f64().ok_or(BtcRejectReason::InvalidFeatureValue)?;
    decimal_from_f64(value.ln_1p())
}

fn normal_probability(
    z_score: Decimal,
    profile: &ChainlinkPathConditionedProfile,
) -> Result<Decimal, BtcRejectReason> {
    Ok(decimal_from_f64(normal_cdf(
        z_score
            .to_f64()
            .ok_or(BtcRejectReason::InvalidFeatureValue)?,
    ))?
    .clamp(
        profile.probability_floor,
        Decimal::ONE - profile.probability_floor,
    ))
}

fn path_probability(
    projected_terminal_gap: Decimal,
    z_score: Decimal,
    profile: &ChainlinkPathConditionedProfile,
) -> Result<Decimal, BtcRejectReason> {
    if projected_terminal_gap == Decimal::ZERO || z_score == Decimal::ZERO {
        Ok(dec!(0.5))
    } else {
        normal_probability(z_score, profile)
    }
}

fn calibrate_probability(
    probability: Decimal,
    profile: &ChainlinkPathConditionedProfile,
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
) -> Result<ChainlinkPathConditionedProfile, BtcRejectReason> {
    let profile: ChainlinkPathConditionedProfile =
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
        && profile.path_parameters
            == (PathParameters {
                recent_path_lookback_seconds: 30,
                short_reversal_lookback_seconds: 15,
                max_anchor_lag_ms: 5_000,
                minimum_path_tick_count: 2,
                volatility_scaled_gap_floor_sigma_fraction: dec!(0.10),
                mature_gap_weight: dec!(1.00),
                fresh_aligned_impulse_weight: dec!(0.35),
                opposing_impulse_weight: dec!(1.25),
                minimum_path_gap_scale: dec!(0.65),
                fresh_impulse_reliability_penalty: dec!(0.75),
                choppiness_reliability_penalty: dec!(0.50),
                volatility_expansion_reliability_penalty: dec!(0.25),
                max_volatility_expansion: dec!(3.0),
                max_abs_crossing_score: dec!(3.0),
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
                max_pre_path_probability_uncertainty: dec!(0.15),
                fresh_impulse_uncertainty_weight: dec!(0.03),
                choppiness_uncertainty_weight: dec!(0.02),
                volatility_expansion_uncertainty_weight: dec!(0.01),
                max_total_probability_uncertainty: dec!(0.18),
            })
        && profile.research_evidence
            == (ResearchEvidence {
                base_profile_id: "btc5m-chainlink-persistence-calibrated-20260720-v1".to_string(),
                base_profile_sha256:
                    "238e7ed1b3676a3da61be3b3c8f9d48d7d05cad92e158b921846b9bfe495c1a3".to_string(),
                source_process_id: "39183e9d-6af4-4671-b985-e6d1284ba28d".to_string(),
                analysis_cutoff: "2026-07-19T22:55:00Z".to_string(),
                target_failure_mode: "overconfident_fresh_impulse_and_late_reversal_predictions"
                    .to_string(),
                parameter_claim: "fixed_research_heuristic_not_fitted".to_string(),
                validation_requirement: "prospective_process_id_scoped_ab".to_string(),
                calibration_claim: "none_forward_validation_required".to_string(),
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
        FeedIntegrityStatus,
    };

    fn config() -> BtcStrategyConfig {
        BtcStrategyConfig {
            strategy_version: STRATEGY_VERSION.to_string(),
            feature_schema_version: FEATURE_SCHEMA_VERSION.to_string(),
            ..BtcStrategyConfig::default()
        }
    }

    fn profile() -> &'static ChainlinkPathConditionedProfile {
        resolve_profile(&ProfileSelection::new(PROFILE_ID, PROFILE_SHA256)).unwrap()
    }

    fn snapshot() -> BtcFeatureSnapshot {
        let window_start = Utc.with_ymd_and_hms(2026, 7, 20, 12, 0, 0).unwrap();
        let observed_at = window_start + Duration::seconds(180);
        let current_source = observed_at - Duration::milliseconds(100);
        let anchor_15s = observed_at - Duration::seconds(15);
        let anchor_30s = observed_at - Duration::seconds(30);
        BtcFeatureSnapshot {
            snapshot_id: Uuid::from_u128(501),
            process_id: Uuid::from_u128(502),
            observed_at,
            feature_schema_version: FEATURE_SCHEMA_VERSION.to_string(),
            market_id: "btc-path-research".to_string(),
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
            chainlink_return_15s: Some(dec!(0.00030)),
            chainlink_return_30s: Some(dec!(0.00040)),
            chainlink_path_efficiency_30s: Some(dec!(0.90)),
            chainlink_path_tick_count_30s: Some(31),
            chainlink_realized_volatility_5s: Some(dec!(0.00012)),
            chainlink_realized_volatility_30s: Some(dec!(0.00012)),
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
                lineage_version: FEATURE_LINEAGE_VERSION.to_string(),
                chainlink_open_tick_id: Some(Uuid::from_u128(503)),
                chainlink_tick_id: Some(Uuid::from_u128(504)),
                binance_tick_id: Some(Uuid::from_u128(505)),
                up_book_checkpoint_id: Some(Uuid::from_u128(506)),
                down_book_checkpoint_id: Some(Uuid::from_u128(507)),
                chainlink_open_source_timestamp: Some(window_start),
                chainlink_open_received_at: Some(window_start),
                chainlink_source_timestamp: Some(current_source),
                chainlink_received_at: Some(current_source),
                chainlink_anchor_15s_tick_id: Some(Uuid::from_u128(508)),
                chainlink_anchor_15s_source_timestamp: Some(anchor_15s),
                chainlink_anchor_15s_received_at: Some(anchor_15s),
                chainlink_anchor_15s_ingest_sequence: Some(15),
                chainlink_anchor_15s_effective_lookback_ms: Some(14_900),
                chainlink_anchor_30s_tick_id: Some(Uuid::from_u128(509)),
                chainlink_anchor_30s_source_timestamp: Some(anchor_30s),
                chainlink_anchor_30s_received_at: Some(anchor_30s),
                chainlink_anchor_30s_ingest_sequence: Some(14),
                chainlink_anchor_30s_effective_lookback_ms: Some(29_900),
                binance_source_timestamp: Some(current_source),
                binance_received_at: Some(current_source),
                chainlink_ingest_sequence: Some(20),
                chainlink_open_ingest_sequence: Some(10),
                binance_ingest_sequence: Some(21),
                up_book_ingest_sequence: Some(22),
                down_book_ingest_sequence: Some(23),
                up_book_connection_id: Some(Uuid::from_u128(510)),
                down_book_connection_id: Some(Uuid::from_u128(510)),
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
            connection_id: Some(Uuid::from_u128(510)),
        }
    }

    fn neutralize_external_lead(snapshot: &mut BtcFeatureSnapshot) {
        snapshot.binance_return_1s = Some(Decimal::ZERO);
        snapshot.binance_return_5s = Some(Decimal::ZERO);
        snapshot.binance_return_30s = Some(Decimal::ZERO);
        snapshot.binance_chainlink_basis_bps = Some(Decimal::ZERO);
    }

    fn stable_mature_snapshot() -> BtcFeatureSnapshot {
        let mut value = snapshot();
        value.chainlink_return_15s = Some(Decimal::ZERO);
        value.chainlink_return_30s = Some(Decimal::ZERO);
        value.chainlink_path_efficiency_30s = Some(Decimal::ZERO);
        value.chainlink_realized_volatility_5s = Some(Decimal::ZERO);
        value.chainlink_realized_volatility_30s = Some(Decimal::ZERO);
        value
    }

    #[test]
    fn profile_identity_hash_schema_and_parameters_fail_closed() {
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
        let changed_weight = PROFILE_ARTIFACT.replace(
            "\"fresh_aligned_impulse_weight\": \"0.35\"",
            "\"fresh_aligned_impulse_weight\": \"0.36\"",
        );
        assert_eq!(
            decode_and_validate_profile(&changed_weight),
            Err(BtcRejectReason::InvalidConfiguration)
        );
        let unknown = PROFILE_ARTIFACT.replace(
            "\"status\": \"research_only\",",
            "\"status\": \"research_only\",\n  \"unknown\": true,",
        );
        assert_eq!(
            decode_and_validate_profile(&unknown),
            Err(BtcRejectReason::InvalidConfiguration)
        );
    }

    #[test]
    fn stable_mature_gap_preserves_pre_path_probability_and_uncertainty() {
        let estimate = estimate(&config(), &stable_mature_snapshot(), profile()).unwrap();

        assert_eq!(
            estimate.pre_path_up_probability,
            Some(estimate.up_probability)
        );
        assert_eq!(
            estimate.path_fresh_impulse_concentration,
            Some(Decimal::ZERO)
        );
        assert_eq!(estimate.path_choppiness_score, Some(Decimal::ZERO));
        assert_eq!(estimate.path_uncertainty_increment, Some(Decimal::ZERO));
        assert_eq!(estimate.path_probability_delta, Some(Decimal::ZERO));
    }

    #[test]
    fn pre_path_uncertainty_retains_the_v1_cap_before_path_increment() {
        let mut saturated = stable_mature_snapshot();
        saturated.realized_volatility = Some(dec!(0.00050));
        saturated.binance_chainlink_basis_bps = Some(dec!(1000));
        saturated.chainlink_return_5s = Some(dec!(-0.01));
        saturated.binance_return_5s = Some(dec!(-0.01));
        saturated.window_end = saturated.observed_at + Duration::seconds(30);

        let mature = estimate(&config(), &saturated, profile()).unwrap();
        assert_eq!(mature.path_uncertainty_increment, Some(Decimal::ZERO));
        assert_eq!(mature.probability_uncertainty, dec!(0.15));

        saturated.chainlink_return_15s = Some(dec!(0.001));
        saturated.chainlink_return_30s = Some(dec!(0.001));
        saturated.chainlink_path_efficiency_30s = Some(Decimal::ONE);
        let fresh = estimate(&config(), &saturated, profile()).unwrap();
        assert!(fresh.path_uncertainty_increment > Some(dec!(0.029)));
        assert!(fresh.probability_uncertainty > dec!(0.179));
        assert!(fresh.probability_uncertainty <= dec!(0.18));
    }

    #[test]
    fn fresh_impulse_is_continuously_less_confident_than_mature_support() {
        let mature = estimate(&config(), &stable_mature_snapshot(), profile()).unwrap();
        let mut fresh_snapshot = snapshot();
        fresh_snapshot.chainlink_return_15s = Some(dec!(0.00080));
        fresh_snapshot.chainlink_return_30s = Some(dec!(0.00100));
        fresh_snapshot.chainlink_path_efficiency_30s = Some(Decimal::ONE);
        fresh_snapshot.chainlink_realized_volatility_5s = Some(dec!(0.00012));
        fresh_snapshot.chainlink_realized_volatility_30s = Some(dec!(0.00012));
        let fresh = estimate(&config(), &fresh_snapshot, profile()).unwrap();

        assert!(fresh.up_probability > dec!(0.5));
        assert!(fresh.up_probability < mature.up_probability);
        assert!(fresh.probability_uncertainty > mature.probability_uncertainty);
        assert!(fresh.path_fresh_impulse_concentration > Some(dec!(0.99)));

        let mut half_fresh_snapshot = fresh_snapshot;
        half_fresh_snapshot.chainlink_return_30s = Some(dec!(0.00050));
        let half_fresh = estimate(&config(), &half_fresh_snapshot, profile()).unwrap();
        assert!(fresh.up_probability < half_fresh.up_probability);
        assert!(half_fresh.up_probability < mature.up_probability);
    }

    #[test]
    fn fresh_aligned_impulse_alone_cannot_flip_the_current_side() {
        let projection = project_path(
            dec!(0.001),
            dec!(0.002),
            dec!(0.003),
            Decimal::ONE,
            dec!(0.00012),
            dec!(0.00012),
            Decimal::ZERO,
            Decimal::ONE,
            profile(),
        )
        .unwrap();

        assert!(projection.projected_terminal_gap >= Decimal::ZERO);
        assert!(projection.open_crossing_score <= Decimal::ZERO);
    }

    #[test]
    fn reversal_occurs_only_after_projected_terminal_gap_crosses_zero() {
        let mut below = snapshot();
        neutralize_external_lead(&mut below);
        below.chainlink_return_15s = Some(dec!(-0.00010));
        below.chainlink_return_30s = Some(Decimal::ZERO);
        below.chainlink_path_efficiency_30s = Some(Decimal::ONE);
        let below = estimate(&config(), &below, profile()).unwrap();
        assert!(below.projected_terminal_gap > Some(Decimal::ZERO));
        assert!(below.up_probability > dec!(0.5));

        let mut crossed = snapshot();
        neutralize_external_lead(&mut crossed);
        crossed.chainlink_return_15s = Some(dec!(-0.00200));
        crossed.chainlink_return_30s = Some(Decimal::ZERO);
        crossed.chainlink_path_efficiency_30s = Some(Decimal::ONE);
        let crossed = estimate(&config(), &crossed, profile()).unwrap();
        assert!(crossed.projected_terminal_gap < Some(Decimal::ZERO));
        assert!(crossed.open_crossing_score > Some(Decimal::ZERO));
        assert!(crossed.up_probability < dec!(0.5));
    }

    #[test]
    fn exact_crossing_boundary_is_probability_neutral() {
        let projection = project_path(
            dec!(0.001),
            dec!(-0.0008),
            Decimal::ZERO,
            Decimal::ONE,
            dec!(0.00012),
            dec!(0.00012),
            Decimal::ZERO,
            Decimal::ONE,
            profile(),
        )
        .unwrap();
        assert_eq!(projection.projected_terminal_gap, Decimal::ZERO);

        let uncalibrated =
            path_probability(projection.projected_terminal_gap, Decimal::ZERO, profile()).unwrap();
        assert!(normal_probability(Decimal::ZERO, profile()).unwrap() > dec!(0.5));
        let up_probability = calibrate_probability(uncalibrated, profile()).unwrap();
        assert_eq!(uncalibrated, dec!(0.5));
        assert_eq!(up_probability, dec!(0.5));
        assert_eq!(Decimal::ONE - up_probability, dec!(0.5));
        assert_eq!(
            path_probability(dec!(-0.001), Decimal::ZERO, profile()).unwrap(),
            dec!(0.5)
        );
    }

    #[test]
    fn active_choppiness_reduces_confidence_but_flat_maturity_is_not_penalized() {
        let mut efficient = snapshot();
        efficient.chainlink_path_efficiency_30s = Some(Decimal::ONE);
        let efficient = estimate(&config(), &efficient, profile()).unwrap();

        let mut choppy = snapshot();
        choppy.chainlink_path_efficiency_30s = Some(Decimal::ZERO);
        let choppy = estimate(&config(), &choppy, profile()).unwrap();
        assert!(choppy.up_probability < efficient.up_probability);
        assert!(choppy.probability_uncertainty > efficient.probability_uncertainty);
        assert!(choppy.path_choppiness_score > Some(Decimal::ZERO));

        let mut flat_efficient = stable_mature_snapshot();
        flat_efficient.chainlink_path_efficiency_30s = Some(Decimal::ONE);
        let flat_efficient = estimate(&config(), &flat_efficient, profile()).unwrap();
        let flat_zero_efficiency =
            estimate(&config(), &stable_mature_snapshot(), profile()).unwrap();
        assert_eq!(
            flat_efficient.up_probability,
            flat_zero_efficiency.up_probability
        );
        assert_eq!(
            flat_efficient.probability_uncertainty,
            flat_zero_efficiency.probability_uncertainty
        );
        assert_eq!(
            flat_efficient.projected_terminal_gap,
            flat_zero_efficiency.projected_terminal_gap
        );
        assert_eq!(flat_efficient.path_choppiness_score, Some(Decimal::ZERO));
        assert_eq!(
            flat_zero_efficiency.path_choppiness_score,
            Some(Decimal::ZERO)
        );
    }

    #[test]
    fn volatility_expansion_reduces_reliability_and_adds_uncertainty() {
        let normal = estimate(&config(), &snapshot(), profile()).unwrap();
        let mut expanded_snapshot = snapshot();
        expanded_snapshot.chainlink_realized_volatility_5s = Some(dec!(0.00048));
        let expanded = estimate(&config(), &expanded_snapshot, profile()).unwrap();

        assert!(expanded.evidence_reliability < normal.evidence_reliability);
        assert!(expanded.probability_uncertainty > normal.probability_uncertainty);
        assert_eq!(expanded.path_volatility_expansion_score, Some(dec!(3.0)));
    }

    #[test]
    fn path_projection_is_up_down_symmetric() {
        let up = project_path(
            dec!(0.0010),
            dec!(-0.0002),
            dec!(0.0004),
            dec!(0.7),
            dec!(0.00015),
            dec!(0.00010),
            dec!(0.00005),
            Decimal::ONE,
            profile(),
        )
        .unwrap();
        let down = project_path(
            dec!(-0.0010),
            dec!(0.0002),
            dec!(-0.0004),
            dec!(0.7),
            dec!(0.00015),
            dec!(0.00010),
            dec!(-0.00005),
            -Decimal::ONE,
            profile(),
        )
        .unwrap();

        assert_eq!(up.projected_terminal_gap, -down.projected_terminal_gap);
        assert_eq!(up.open_crossing_score, down.open_crossing_score);
        assert_eq!(
            up.fresh_impulse_concentration,
            down.fresh_impulse_concentration
        );
        assert_eq!(up.choppiness, down.choppiness);
    }

    #[test]
    fn path_inputs_and_lineage_fail_closed() {
        let mut missing_return = snapshot();
        missing_return.chainlink_return_30s = None;
        assert_eq!(
            estimate(&config(), &missing_return, profile()),
            Err(BtcRejectReason::MissingChainlinkReturns)
        );

        let mut stale_anchor = snapshot();
        let stale = stale_anchor.observed_at - Duration::seconds(36);
        let current = stale_anchor.lineage.chainlink_source_timestamp.unwrap();
        stale_anchor.lineage.chainlink_anchor_30s_source_timestamp = Some(stale);
        stale_anchor.lineage.chainlink_anchor_30s_received_at = Some(stale);
        stale_anchor
            .lineage
            .chainlink_anchor_30s_effective_lookback_ms =
            Some((current - stale).num_milliseconds());
        assert_eq!(
            estimate(&config(), &stale_anchor, profile()),
            Err(BtcRejectReason::InvalidFeatureValue)
        );

        let mut future_anchor = snapshot();
        future_anchor.lineage.chainlink_anchor_15s_received_at =
            Some(future_anchor.observed_at + Duration::milliseconds(1));
        assert_eq!(
            estimate(&config(), &future_anchor, profile()),
            Err(BtcRejectReason::FutureInputTimestamp)
        );
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

        let mut changed = snapshot;
        changed.up_book.imbalance = Some(dec!(-0.99));
        changed.down_book.imbalance = Some(dec!(0.99));
        assert_eq!(first, estimate(&config(), &changed, profile()).unwrap());
    }
}
