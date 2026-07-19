use std::sync::OnceLock;

use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{
    decimal_from_f64, estimate_chainlink_fair_value, BtcFeatureSnapshot, BtcRejectReason,
    BtcStrategyConfig, FairValueEstimate, BTC_FEATURE_SCHEMA_VERSION,
};

pub(super) const MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION: &str =
    "btc_5m_market_anchored_fair_value_research_v1";

pub(super) const PROFILE_ID: &str = "btc5m-market-anchored-research-20260718-v1";
pub(super) const PROFILE_SHA256: &str =
    "5f84df367641cd6f6144ef20f8e6044f4de43710f9caa16a7da7e9e8f888c99a";
const PROFILE_ARTIFACT: &str =
    include_str!("profiles/btc5m-market-anchored-research-20260718-v1.json");
const PROFILE_STATUS: &str = "research_only";
const MARKET_PRIOR: &str = "normalized_two_sided_best_bid_ask_midpoint";
const EXTERNAL_EVIDENCE: &str =
    "existing_chainlink_gap_plus_bounded_lead_over_terminal_volatility_z";
const CALIBRATION: &str = "identity";
const UNCERTAINTY_POLICY: &str = "existing_btc_v1";

static COMPILED_PROFILE: OnceLock<Result<MarketAnchoredProfile, BtcRejectReason>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MarketAnchoredProfileSelection {
    pub(super) profile_id: String,
    pub(super) profile_sha256: String,
}

impl MarketAnchoredProfileSelection {
    pub(super) fn new(profile_id: impl Into<String>, profile_sha256: impl Into<String>) -> Self {
        Self {
            profile_id: profile_id.into(),
            profile_sha256: profile_sha256.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MarketAnchoredProfile {
    profile_id: String,
    status: String,
    feature_schema_version: String,
    market_prior: String,
    external_evidence: String,
    external_z_clip: Decimal,
    max_abs_logit_adjustment: Decimal,
    probability_floor: Decimal,
    external_evidence_parameters: ExternalEvidenceParameters,
    calibration: String,
    uncertainty_policy: String,
    uncertainty_parameters: UncertaintyParameters,
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
struct UncertaintyParameters {
    base_probability_uncertainty: Decimal,
    basis_uncertainty_weight: Decimal,
    feed_age_uncertainty_per_second: Decimal,
    max_probability_uncertainty: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct MarketAnchoredEstimate {
    pub(super) fair_value: FairValueEstimate,
    pub(super) market_up_prior: Decimal,
    pub(super) signed_logit_adjustment: Decimal,
}

pub(super) fn resolve_profile(
    selection: &MarketAnchoredProfileSelection,
) -> Result<&'static MarketAnchoredProfile, BtcRejectReason> {
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
    profile: &MarketAnchoredProfile,
) -> Result<MarketAnchoredEstimate, BtcRejectReason> {
    validate_strategy_config(config, profile)?;
    // Reuse the baseline's evidence calculation so the candidate differs only in how that
    // evidence updates the independently observed market prior.
    let baseline = estimate_chainlink_fair_value(config, snapshot)?;

    let up_midpoint = midpoint(
        snapshot.up_book.best_bid,
        snapshot.up_book.best_ask,
        BtcRejectReason::MissingUpBook,
    )?;
    let down_midpoint = midpoint(
        snapshot.down_book.best_bid,
        snapshot.down_book.best_ask,
        BtcRejectReason::MissingDownBook,
    )?;
    let midpoint_sum = up_midpoint + down_midpoint;
    if midpoint_sum <= Decimal::ZERO {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    let market_up_prior = (up_midpoint / midpoint_sum).clamp(
        profile.probability_floor,
        Decimal::ONE - profile.probability_floor,
    );

    let external_z = baseline.z_score;
    let clipped_external_z = external_z.clamp(-profile.external_z_clip, profile.external_z_clip);
    let signed_logit_adjustment = profile.max_abs_logit_adjustment * clipped_external_z;

    let prior_f64 = market_up_prior
        .to_f64()
        .ok_or(BtcRejectReason::InvalidFeatureValue)?;
    let prior_logit = (prior_f64 / (1.0 - prior_f64)).ln();
    let adjusted_logit = prior_logit
        + signed_logit_adjustment
            .to_f64()
            .ok_or(BtcRejectReason::InvalidFeatureValue)?;
    let up_probability = decimal_from_f64(1.0 / (1.0 + (-adjusted_logit).exp()))?.clamp(
        profile.probability_floor,
        Decimal::ONE - profile.probability_floor,
    );

    let probability_uncertainty = baseline.probability_uncertainty;
    let up_lower_bound = (up_probability - probability_uncertainty).max(profile.probability_floor);
    let up_upper_bound =
        (up_probability + probability_uncertainty).min(Decimal::ONE - profile.probability_floor);

    Ok(MarketAnchoredEstimate {
        fair_value: FairValueEstimate {
            up_probability,
            down_probability: Decimal::ONE - up_probability,
            up_lower_bound,
            up_upper_bound,
            down_lower_bound: Decimal::ONE - up_upper_bound,
            down_upper_bound: Decimal::ONE - up_lower_bound,
            z_score: external_z,
            chainlink_log_gap: baseline.chainlink_log_gap,
            lead_adjustment: baseline.lead_adjustment,
            terminal_volatility: baseline.terminal_volatility,
            probability_uncertainty,
            market_up_prior: Some(market_up_prior),
            signed_logit_adjustment: Some(signed_logit_adjustment),
        },
        market_up_prior,
        signed_logit_adjustment,
    })
}

pub(super) fn validate_strategy_config(
    config: &BtcStrategyConfig,
    profile: &MarketAnchoredProfile,
) -> Result<(), BtcRejectReason> {
    if config.feature_schema_version == profile.feature_schema_version
        && config.probability_floor == profile.probability_floor
        && config.volatility_floor_per_sqrt_second
            == profile
                .external_evidence_parameters
                .volatility_floor_per_sqrt_second
        && config.basis_lead_weight == profile.external_evidence_parameters.basis_lead_weight
        && config.momentum_1s_weight == profile.external_evidence_parameters.momentum_1s_weight
        && config.momentum_5s_weight == profile.external_evidence_parameters.momentum_5s_weight
        && config.momentum_30s_weight == profile.external_evidence_parameters.momentum_30s_weight
        && config.max_lead_sigma_fraction
            == profile.external_evidence_parameters.max_lead_sigma_fraction
        && config.base_probability_uncertainty
            == profile.uncertainty_parameters.base_probability_uncertainty
        && config.basis_uncertainty_weight
            == profile.uncertainty_parameters.basis_uncertainty_weight
        && config.feed_age_uncertainty_per_second
            == profile
                .uncertainty_parameters
                .feed_age_uncertainty_per_second
        && config.max_probability_uncertainty
            == profile.uncertainty_parameters.max_probability_uncertainty
    {
        Ok(())
    } else {
        Err(BtcRejectReason::InvalidConfiguration)
    }
}

fn midpoint(
    bid: Option<Decimal>,
    ask: Option<Decimal>,
    missing_reason: BtcRejectReason,
) -> Result<Decimal, BtcRejectReason> {
    let bid = bid.ok_or(missing_reason)?;
    let ask = ask.ok_or(missing_reason)?;
    if bid < Decimal::ZERO
        || ask > Decimal::ONE
        || ask <= Decimal::ZERO
        || bid >= Decimal::ONE
        || bid > ask
    {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    Ok((bid + ask) / dec!(2))
}

fn decode_and_validate_profile(artifact: &str) -> Result<MarketAnchoredProfile, BtcRejectReason> {
    let profile: MarketAnchoredProfile =
        serde_json::from_str(artifact).map_err(|_| BtcRejectReason::InvalidConfiguration)?;
    let valid = profile.profile_id == PROFILE_ID
        && profile.status == PROFILE_STATUS
        && profile.feature_schema_version == BTC_FEATURE_SCHEMA_VERSION
        && profile.market_prior == MARKET_PRIOR
        && profile.external_evidence == EXTERNAL_EVIDENCE
        && profile.external_z_clip == Decimal::ONE
        && profile.max_abs_logit_adjustment == dec!(0.40)
        && profile.probability_floor == dec!(0.01)
        && profile.external_evidence_parameters
            == (ExternalEvidenceParameters {
                volatility_floor_per_sqrt_second: dec!(0.00005),
                basis_lead_weight: dec!(0.25),
                momentum_1s_weight: dec!(0.05),
                momentum_5s_weight: dec!(0.10),
                momentum_30s_weight: dec!(0.10),
                max_lead_sigma_fraction: dec!(0.25),
            })
        && profile.calibration == CALIBRATION
        && profile.uncertainty_policy == UNCERTAINTY_POLICY
        && profile.uncertainty_parameters
            == (UncertaintyParameters {
                base_probability_uncertainty: dec!(0.015),
                basis_uncertainty_weight: Decimal::ONE,
                feed_age_uncertainty_per_second: dec!(0.002),
                max_probability_uncertainty: dec!(0.10),
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

    fn snapshot() -> BtcFeatureSnapshot {
        let window_start = Utc.with_ymd_and_hms(2026, 7, 18, 12, 0, 0).unwrap();
        let observed_at = window_start + Duration::seconds(180);
        let source_at = observed_at - Duration::milliseconds(100);
        BtcFeatureSnapshot {
            snapshot_id: Uuid::from_u128(101),
            process_id: Uuid::from_u128(102),
            observed_at,
            feature_schema_version: BTC_FEATURE_SCHEMA_VERSION.to_string(),
            market_id: "btc-market-research".to_string(),
            event_slug: "btc-updown-5m-1784376000".to_string(),
            window_start,
            window_end: window_start + Duration::seconds(300),
            market_active: true,
            market_closed: false,
            accepting_orders: true,
            tick_size: dec!(0.01),
            minimum_order_size: Some(dec!(1)),
            resolution_source: "chainlink btc/usd".to_string(),
            chainlink_open_price: Some(dec!(100000)),
            chainlink_price: Some(dec!(100000)),
            binance_price: Some(dec!(100000)),
            chainlink_gap_bps: Some(Decimal::ZERO),
            binance_return_1s: Some(Decimal::ZERO),
            binance_return_5s: Some(Decimal::ZERO),
            binance_return_30s: Some(Decimal::ZERO),
            realized_volatility: Some(dec!(0.00012)),
            binance_chainlink_basis_bps: Some(Decimal::ZERO),
            chainlink_age_ms: Some(100),
            binance_age_ms: Some(100),
            source_skew_ms: Some(25),
            chainlink_quality_ok: true,
            binance_quality_ok: true,
            up_book: book(BtcOutcome::Up, "up", dec!(0.59), dec!(0.61), dec!(0.2)),
            down_book: book(BtcOutcome::Down, "down", dec!(0.39), dec!(0.41), dec!(-0.2)),
            fees_enabled: true,
            fee_rate: Some(dec!(0.03)),
            fee_rate_observed_at: Some(observed_at - Duration::minutes(1)),
            lineage: BtcFeatureLineage {
                lineage_version: "btc_5m_feature_lineage_v2".to_string(),
                chainlink_open_tick_id: Some(Uuid::from_u128(103)),
                chainlink_tick_id: Some(Uuid::from_u128(104)),
                binance_tick_id: Some(Uuid::from_u128(105)),
                up_book_checkpoint_id: Some(Uuid::from_u128(106)),
                down_book_checkpoint_id: Some(Uuid::from_u128(107)),
                chainlink_open_source_timestamp: Some(window_start),
                chainlink_open_received_at: Some(window_start),
                chainlink_source_timestamp: Some(source_at),
                chainlink_received_at: Some(source_at),
                binance_source_timestamp: Some(source_at),
                binance_received_at: Some(source_at),
                chainlink_ingest_sequence: Some(10),
                chainlink_open_ingest_sequence: Some(9),
                binance_ingest_sequence: Some(11),
                up_book_ingest_sequence: Some(12),
                down_book_ingest_sequence: Some(13),
                up_book_connection_id: Some(Uuid::from_u128(108)),
                down_book_connection_id: Some(Uuid::from_u128(108)),
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
        imbalance: Decimal,
    ) -> BtcOutcomeBookFeatures {
        let observed_at = Utc.with_ymd_and_hms(2026, 7, 18, 12, 3, 0).unwrap();
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
            imbalance: Some(imbalance),
            source_timestamp: Some(observed_at - Duration::milliseconds(100)),
            received_at: Some(observed_at - Duration::milliseconds(100)),
            age_ms: Some(100),
            integrity_status: FeedIntegrityStatus::Ok,
            connection_id: Some(Uuid::from_u128(108)),
        }
    }

    fn estimate_snapshot(snapshot: &BtcFeatureSnapshot) -> MarketAnchoredEstimate {
        let selection = MarketAnchoredProfileSelection::new(PROFILE_ID, PROFILE_SHA256);
        let profile = resolve_profile(&selection).unwrap();
        estimate(&BtcStrategyConfig::default(), snapshot, profile).unwrap()
    }

    fn as_f64(value: Decimal) -> f64 {
        value.to_f64().unwrap()
    }

    fn assert_close(left: Decimal, right: Decimal, tolerance: f64) {
        assert!((as_f64(left) - as_f64(right)).abs() <= tolerance);
    }

    #[test]
    fn neutral_evidence_preserves_normalized_market_midpoint() {
        let result = estimate_snapshot(&snapshot());

        assert_eq!(result.market_up_prior, dec!(0.6));
        assert_close(result.fair_value.up_probability, dec!(0.6), 1e-12);
        assert_eq!(result.signed_logit_adjustment, Decimal::ZERO);
    }

    #[test]
    fn normalization_uses_both_outcome_midpoints() {
        let mut snapshot = snapshot();
        snapshot.up_book.best_bid = Some(dec!(0.59));
        snapshot.up_book.best_ask = Some(dec!(0.61));
        snapshot.down_book.best_bid = Some(dec!(0.29));
        snapshot.down_book.best_ask = Some(dec!(0.31));

        let result = estimate_snapshot(&snapshot);

        assert_close(result.market_up_prior, dec!(0.6666666666666667), 1e-12);
        assert_close(
            result.fair_value.up_probability,
            result.market_up_prior,
            1e-12,
        );
    }

    #[test]
    fn mirrored_market_and_external_evidence_are_complementary() {
        let mut positive = snapshot();
        positive.binance_chainlink_basis_bps = Some(dec!(10));
        let positive_result = estimate_snapshot(&positive);

        let mut negative = snapshot();
        negative.up_book = book(BtcOutcome::Up, "up", dec!(0.39), dec!(0.41), dec!(0.7));
        negative.down_book = book(BtcOutcome::Down, "down", dec!(0.59), dec!(0.61), dec!(-0.7));
        negative.binance_chainlink_basis_bps = Some(dec!(-10));
        let negative_result = estimate_snapshot(&negative);

        assert_close(
            positive_result.fair_value.up_probability,
            Decimal::ONE - negative_result.fair_value.up_probability,
            1e-12,
        );
        assert_close(
            positive_result.signed_logit_adjustment,
            -negative_result.signed_logit_adjustment,
            1e-12,
        );
    }

    #[test]
    fn external_logit_adjustment_is_capped_by_research_profile() {
        let mut snapshot = snapshot();
        snapshot.chainlink_price = Some(dec!(102000));
        let result = estimate_snapshot(&snapshot);

        assert!(result.fair_value.z_score > Decimal::ONE);
        assert_eq!(result.signed_logit_adjustment, dec!(0.40));
        let probability = as_f64(result.fair_value.up_probability);
        let prior = as_f64(result.market_up_prior);
        let observed_adjustment =
            (probability / (1.0 - probability)).ln() - (prior / (1.0 - prior)).ln();
        assert!((observed_adjustment - 0.40).abs() <= 1e-12);
    }

    #[test]
    fn probabilities_and_bounds_are_finite_complementary_and_ordered() {
        let mut snapshot = snapshot();
        snapshot.chainlink_price = Some(dec!(100080));
        snapshot.binance_chainlink_basis_bps = Some(dec!(1));
        let estimate = estimate_snapshot(&snapshot).fair_value;

        for value in [
            estimate.up_probability,
            estimate.down_probability,
            estimate.up_lower_bound,
            estimate.up_upper_bound,
            estimate.down_lower_bound,
            estimate.down_upper_bound,
            estimate.z_score,
            estimate.chainlink_log_gap,
            estimate.lead_adjustment,
            estimate.terminal_volatility,
            estimate.probability_uncertainty,
        ] {
            assert!(as_f64(value).is_finite());
        }
        assert_eq!(
            estimate.up_probability + estimate.down_probability,
            Decimal::ONE
        );
        assert_eq!(
            estimate.up_lower_bound + estimate.down_upper_bound,
            Decimal::ONE
        );
        assert_eq!(
            estimate.up_upper_bound + estimate.down_lower_bound,
            Decimal::ONE
        );
        assert!(estimate.up_lower_bound <= estimate.up_probability);
        assert!(estimate.up_probability <= estimate.up_upper_bound);
        assert!(estimate.down_lower_bound <= estimate.down_probability);
        assert!(estimate.down_probability <= estimate.down_upper_bound);
    }

    #[test]
    fn book_imbalance_does_not_influence_probability() {
        let baseline = estimate_snapshot(&snapshot());
        let mut changed = snapshot();
        changed.up_book.imbalance = Some(dec!(-0.99));
        changed.down_book.imbalance = Some(dec!(0.99));

        let changed = estimate_snapshot(&changed);

        assert_eq!(baseline, changed);
    }

    #[test]
    fn estimator_is_deterministic_for_identical_inputs() {
        let snapshot = snapshot();
        let first = estimate_snapshot(&snapshot);
        let second = estimate_snapshot(&snapshot);

        assert_eq!(first, second);
    }

    #[test]
    fn profile_resolver_rejects_wrong_id_hash_and_schema() {
        assert_eq!(PROFILE_ID, "btc5m-market-anchored-research-20260718-v1");
        assert_eq!(
            PROFILE_SHA256,
            "5f84df367641cd6f6144ef20f8e6044f4de43710f9caa16a7da7e9e8f888c99a"
        );
        let wrong_id = MarketAnchoredProfileSelection::new("wrong", PROFILE_SHA256);
        let wrong_hash = MarketAnchoredProfileSelection::new(PROFILE_ID, "0".repeat(64));
        assert_eq!(
            resolve_profile(&wrong_id),
            Err(BtcRejectReason::InvalidConfiguration)
        );
        assert_eq!(
            resolve_profile(&wrong_hash),
            Err(BtcRejectReason::InvalidConfiguration)
        );

        let wrong_schema =
            PROFILE_ARTIFACT.replace(BTC_FEATURE_SCHEMA_VERSION, "btc_5m_features_incompatible");
        assert_eq!(
            decode_and_validate_profile(&wrong_schema),
            Err(BtcRejectReason::InvalidConfiguration)
        );

        let selection = MarketAnchoredProfileSelection::new(PROFILE_ID, PROFILE_SHA256);
        let profile = resolve_profile(&selection).unwrap();
        let changed_weights = BtcStrategyConfig {
            momentum_5s_weight: dec!(0.11),
            ..BtcStrategyConfig::default()
        };
        assert_eq!(
            validate_strategy_config(&changed_weights, profile),
            Err(BtcRejectReason::InvalidConfiguration)
        );
    }
}
