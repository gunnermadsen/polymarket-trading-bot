use chrono::{DateTime, Utc};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use crate::fees::dynamic_crypto_taker_fee;

use super::{
    directional_model::{
        asymmetric_value_model_input_sha256, directional_model_input_sha256, runtime_model,
        BtcDirectionalModelFeatureSnapshot, RuntimeModelAction, RuntimeModelScore,
        RuntimeModelSelection, BTC_DIRECTIONAL_MODEL_FAMILY,
        BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION,
    },
    feed_contract::{validate_feed_requirements, BtcModelFeedRequirement},
    types::{BtcOutcome, FeedIntegrityStatus},
};

pub const BTC_FEATURE_SCHEMA_VERSION: &str = "btc_5m_features_v2";
pub const BTC_FEATURE_LINEAGE_VERSION: &str = "btc_5m_feature_lineage_v2";
pub const BTC_DIRECTIONAL_MODEL_STRATEGY_FAMILY: &str = BTC_DIRECTIONAL_MODEL_FAMILY;
pub const BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_VERSION: &str = "btc_5m_asymmetric_value_model_v1";
pub const BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_FAMILY: &str = "btc_5m_asymmetric_value_model";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BtcDirectionalModelEntryPolicy {
    #[default]
    RequirePositiveDirectEdge,
    ExecuteDirectionalPrediction,
}

impl BtcDirectionalModelEntryPolicy {
    pub(crate) fn is_default(&self) -> bool {
        *self == Self::RequirePositiveDirectEdge
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum BtcDecisionStrategyConfig {
    BtcDirectionalModel {
        model_key: String,
        artifact_sha256: String,
        feature_schema_sha256: String,
    },
    BtcAsymmetricValueModel {
        model_key: String,
        artifact_sha256: String,
        feature_schema_sha256: String,
    },
}

/// Immutable, process-owned parameters for the deterministic baseline.
///
/// Runtime configuration should be hashed and frozen when an execution run starts. The strategy
/// deliberately excludes Polymarket order-book imbalance from its probability estimator. Book
/// features are used only to determine executability and net edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BtcStrategyConfig {
    pub strategy_version: String,
    pub feature_schema_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_strategy: Option<BtcDecisionStrategyConfig>,
    /// Additive declaration for model-owned feature requirements. Existing strategies leave
    /// this empty and retain their established runtime data path.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_model_feeds: Vec<BtcModelFeedRequirement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unified_model: Option<super::unified_model_runtime::contract::ProcessBinding>,
    pub target_size: Decimal,
    pub min_seconds_after_open: i64,
    pub min_seconds_before_close: i64,
    pub max_reference_age_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_directional_feature_age_ms: Option<i64>,
    pub max_chainlink_open_delay_ms: i64,
    pub max_book_age_ms: i64,
    pub max_source_skew_ms: i64,
    pub max_fee_age_ms: i64,
    pub min_entry_price: Decimal,
    pub max_entry_price: Decimal,
    pub max_depth_participation: Decimal,
    pub volatility_floor_per_sqrt_second: Decimal,
    pub probability_floor: Decimal,
    pub basis_lead_weight: Decimal,
    pub momentum_1s_weight: Decimal,
    pub momentum_5s_weight: Decimal,
    pub momentum_30s_weight: Decimal,
    pub max_lead_sigma_fraction: Decimal,
    pub base_probability_uncertainty: Decimal,
    pub basis_uncertainty_weight: Decimal,
    pub feed_age_uncertainty_per_second: Decimal,
    pub max_probability_uncertainty: Decimal,
    pub spread_reserve_fraction: Decimal,
    pub slippage_reserve_bps: Decimal,
    pub latency_reserve_per_share: Decimal,
    pub min_net_edge_per_share: Decimal,
    pub min_net_edge_usd: Decimal,
    pub max_fee_rate: Decimal,
}

impl Default for BtcStrategyConfig {
    fn default() -> Self {
        Self {
            strategy_version: String::new(),
            feature_schema_version: String::new(),
            decision_strategy: None,
            required_model_feeds: Vec::new(),
            unified_model: None,
            target_size: dec!(5),
            min_seconds_after_open: 15,
            min_seconds_before_close: 20,
            max_reference_age_ms: 2_000,
            max_directional_feature_age_ms: None,
            max_chainlink_open_delay_ms: 5_000,
            max_book_age_ms: 2_000,
            max_source_skew_ms: 1_000,
            max_fee_age_ms: 3_600_000,
            min_entry_price: dec!(0.05),
            max_entry_price: dec!(0.95),
            max_depth_participation: dec!(0.25),
            volatility_floor_per_sqrt_second: dec!(0.00005),
            probability_floor: dec!(0.01),
            basis_lead_weight: dec!(0.25),
            momentum_1s_weight: dec!(0.05),
            momentum_5s_weight: dec!(0.10),
            momentum_30s_weight: dec!(0.10),
            max_lead_sigma_fraction: dec!(0.25),
            base_probability_uncertainty: dec!(0.015),
            basis_uncertainty_weight: dec!(1),
            feed_age_uncertainty_per_second: dec!(0.002),
            max_probability_uncertainty: dec!(0.10),
            spread_reserve_fraction: dec!(0.10),
            slippage_reserve_bps: dec!(25),
            latency_reserve_per_share: dec!(0.005),
            min_net_edge_per_share: dec!(0.015),
            min_net_edge_usd: dec!(0.02),
            max_fee_rate: dec!(1),
        }
    }
}

impl BtcStrategyConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        validate_config(self)
            .map_err(|_| anyhow::anyhow!("invalid BTC deterministic strategy configuration"))
    }

    pub fn effective_max_directional_feature_age_ms(&self) -> anyhow::Result<Option<i64>> {
        let strategy = ResolvedBtcDecisionStrategy::resolve(self)
            .map_err(|_| anyhow::anyhow!("invalid BTC decision strategy configuration"))?;
        let (model_key, artifact_sha256, feature_schema_sha256) = match strategy {
            ResolvedBtcDecisionStrategy::BtcDirectionalModel {
                model_key,
                artifact_sha256,
                feature_schema_sha256,
            }
            | ResolvedBtcDecisionStrategy::BtcAsymmetricValueModel {
                model_key,
                artifact_sha256,
                feature_schema_sha256,
            } => (model_key, artifact_sha256, feature_schema_sha256),
        };
        let selection =
            btc_directional_model_selection(model_key, artifact_sha256, feature_schema_sha256);
        let model = runtime_model(&selection)?;
        let cadence_seconds = model
            .prediction_policy()
            .early_cadence_seconds
            .unwrap_or(model.prediction_policy().cadence_seconds);
        let max_age = self
            .max_directional_feature_age_ms
            .unwrap_or(cadence_seconds * 1_000);
        anyhow::ensure!(
            max_age > 0 && max_age <= cadence_seconds * 1_000,
            "BTC directional feature age bound must be positive and no greater than model cadence"
        );
        Ok(Some(max_age))
    }

    pub fn attribution(&self) -> Option<BtcStrategyAttribution<'_>> {
        let family = match ResolvedBtcDecisionStrategy::resolve(self).ok()? {
            ResolvedBtcDecisionStrategy::BtcDirectionalModel { .. } => {
                BTC_DIRECTIONAL_MODEL_STRATEGY_FAMILY
            }
            ResolvedBtcDecisionStrategy::BtcAsymmetricValueModel { .. } => {
                BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_FAMILY
            }
        };
        let (profile_id, profile_sha256) = match self.decision_strategy.as_ref() {
            Some(BtcDecisionStrategyConfig::BtcDirectionalModel {
                model_key,
                artifact_sha256,
                ..
            })
            | Some(BtcDecisionStrategyConfig::BtcAsymmetricValueModel {
                model_key,
                artifact_sha256,
                ..
            }) => (Some(model_key.as_str()), Some(artifact_sha256.as_str())),
            None => (None, None),
        };
        Some(BtcStrategyAttribution {
            family,
            strategy_version: &self.strategy_version,
            profile_id,
            profile_sha256,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BtcStrategyAttribution<'a> {
    pub family: &'static str,
    pub strategy_version: &'a str,
    pub profile_id: Option<&'a str>,
    pub profile_sha256: Option<&'a str>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BtcInputWindowLineage {
    pub first_tick_id: Option<Uuid>,
    pub last_tick_id: Option<Uuid>,
    pub first_source_timestamp: Option<DateTime<Utc>>,
    pub last_source_timestamp: Option<DateTime<Utc>>,
    pub tick_count: usize,
    pub max_received_at: Option<DateTime<Utc>>,
    /// SHA-256 of the ordered tick identity, event/receipt times, sequence and normalized price.
    pub input_sha256: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BtcFeatureLineage {
    pub lineage_version: String,
    pub chainlink_open_tick_id: Option<Uuid>,
    pub chainlink_tick_id: Option<Uuid>,
    pub binance_tick_id: Option<Uuid>,
    pub up_book_checkpoint_id: Option<Uuid>,
    pub down_book_checkpoint_id: Option<Uuid>,
    pub chainlink_open_source_timestamp: Option<DateTime<Utc>>,
    pub chainlink_open_received_at: Option<DateTime<Utc>>,
    pub chainlink_source_timestamp: Option<DateTime<Utc>>,
    pub chainlink_received_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_anchor_15s_tick_id: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_anchor_15s_source_timestamp: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_anchor_15s_received_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_anchor_15s_ingest_sequence: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_anchor_15s_effective_lookback_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_anchor_30s_tick_id: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_anchor_30s_source_timestamp: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_anchor_30s_received_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_anchor_30s_ingest_sequence: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_anchor_30s_effective_lookback_ms: Option<i64>,
    pub binance_source_timestamp: Option<DateTime<Utc>>,
    pub binance_received_at: Option<DateTime<Utc>>,
    pub chainlink_ingest_sequence: Option<u64>,
    pub chainlink_open_ingest_sequence: Option<u64>,
    pub binance_ingest_sequence: Option<u64>,
    pub up_book_ingest_sequence: Option<u64>,
    pub down_book_ingest_sequence: Option<u64>,
    pub up_book_connection_id: Option<Uuid>,
    pub down_book_connection_id: Option<Uuid>,
    pub chainlink_history: BtcInputWindowLineage,
    pub binance_history: BtcInputWindowLineage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BtcOutcomeBookFeatures {
    pub outcome: BtcOutcome,
    pub token_id: String,
    pub best_bid: Option<Decimal>,
    pub best_ask: Option<Decimal>,
    /// VWAP of a depth walk for `quoted_size` at the observation timestamp.
    pub executable_ask_vwap: Option<Decimal>,
    /// Worst ask used by the depth walk and therefore the marketable FOK limit.
    pub marketable_limit_price: Option<Decimal>,
    pub quoted_size: Decimal,
    pub bid_depth: Decimal,
    pub ask_depth: Decimal,
    /// Recorded for observability and future ML, but intentionally unused by fair value.
    pub imbalance: Option<Decimal>,
    pub source_timestamp: Option<DateTime<Utc>>,
    pub received_at: Option<DateTime<Utc>>,
    pub age_ms: Option<i64>,
    pub integrity_status: FeedIntegrityStatus,
    /// Socket epoch that produced the durable checkpoint used by this snapshot.
    pub connection_id: Option<Uuid>,
}

/// A complete point-in-time feature vector. No strategy method mutates it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BtcFeatureSnapshot {
    pub snapshot_id: Uuid,
    pub process_id: Uuid,
    pub observed_at: DateTime<Utc>,
    pub feature_schema_version: String,
    pub market_id: String,
    pub event_slug: String,
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    pub market_active: bool,
    pub market_closed: bool,
    pub accepting_orders: bool,
    pub tick_size: Decimal,
    pub minimum_order_size: Option<Decimal>,
    pub resolution_source: String,
    pub chainlink_open_price: Option<Decimal>,
    pub chainlink_price: Option<Decimal>,
    pub binance_price: Option<Decimal>,
    pub chainlink_gap_bps: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_return_5s: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_return_15s: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_return_30s: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_path_efficiency_30s: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_path_tick_count_30s: Option<usize>,
    /// Chainlink realized log-return volatility measured per square-root second.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_realized_volatility_5s: Option<Decimal>,
    /// Chainlink realized log-return volatility measured per square-root second.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_realized_volatility_30s: Option<Decimal>,
    pub binance_return_1s: Option<Decimal>,
    pub binance_return_5s: Option<Decimal>,
    pub binance_return_30s: Option<Decimal>,
    /// Realized log-return volatility measured per square-root second.
    pub realized_volatility: Option<Decimal>,
    /// Current Binance/Chainlink basis minus its rolling neutral basis, in bps.
    pub binance_chainlink_basis_bps: Option<Decimal>,
    pub chainlink_age_ms: Option<i64>,
    pub binance_age_ms: Option<i64>,
    pub source_skew_ms: Option<i64>,
    pub chainlink_quality_ok: bool,
    pub binance_quality_ok: bool,
    pub up_book: BtcOutcomeBookFeatures,
    pub down_book: BtcOutcomeBookFeatures,
    pub fees_enabled: bool,
    pub fee_rate: Option<Decimal>,
    pub fee_rate_observed_at: Option<DateTime<Utc>>,
    pub lineage: BtcFeatureLineage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directional_model: Option<BtcDirectionalModelFeatureSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FairValueEstimate {
    pub up_probability: Decimal,
    pub down_probability: Decimal,
    pub up_lower_bound: Decimal,
    pub up_upper_bound: Decimal,
    pub down_lower_bound: Decimal,
    pub down_upper_bound: Decimal,
    pub z_score: Decimal,
    pub chainlink_log_gap: Decimal,
    pub lead_adjustment: Decimal,
    pub terminal_volatility: Decimal,
    pub probability_uncertainty: Decimal,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub market_up_prior: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signed_logit_adjustment: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chainlink_persistence_score: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binance_confirmation_score: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirmation_deficiency_score: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_reliability: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncalibrated_up_probability: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_recent_log_return_30s: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_prior_log_gap_30s: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_mature_gap_support: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_fresh_aligned_impulse: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_fresh_impulse_concentration: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_efficiency_score: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_choppiness_score: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_volatility_expansion_score: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected_terminal_gap: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_crossing_score: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_path_up_probability: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_probability_delta: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_uncertainty_increment: Option<Decimal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimator_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimator_profile_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimator_profile_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutcomeEdge {
    pub outcome: BtcOutcome,
    pub token_id: String,
    pub conservative_probability: Decimal,
    pub executable_price: Decimal,
    pub marketable_limit_price: Decimal,
    pub size: Decimal,
    pub gross_edge: Decimal,
    pub taker_fee: Decimal,
    pub spread_reserve: Decimal,
    pub slippage_reserve: Decimal,
    pub latency_reserve: Decimal,
    pub net_edge: Decimal,
    pub net_edge_per_share: Decimal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovedIntent {
    pub intent_id: Uuid,
    pub process_id: Uuid,
    pub feature_snapshot_id: Uuid,
    pub market_id: String,
    pub window_start: DateTime<Utc>,
    pub outcome: BtcOutcome,
    pub token_id: String,
    pub limit_price: Decimal,
    pub size: Decimal,
    pub expected_net_edge: Decimal,
    pub expected_net_edge_per_share: Decimal,
    pub strategy_version: String,
    pub feature_schema_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BtcDecisionAction {
    NoTrade,
    BuyUp,
    BuyDown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BtcRejectReason {
    InvalidConfiguration,
    FeatureSchemaMismatch,
    MarketInactive,
    MarketClosed,
    OrdersNotAccepted,
    OutsideEntryWindow,
    InvalidMarketIdentity,
    InvalidResolutionSource,
    MissingChainlinkOpen,
    MissingChainlinkPrice,
    MissingChainlinkGap,
    MissingChainlinkReturns,
    MissingBinancePrice,
    MissingBinanceReturns,
    MissingRealizedVolatility,
    MissingBasis,
    ChainlinkFeedUnhealthy,
    BinanceFeedUnhealthy,
    MissingLineage,
    FutureInputTimestamp,
    InvalidDirectionalFeatureTimestamp,
    FutureDirectionalFeatures,
    StaleDirectionalFeatures,
    DirectionalFeaturesUnavailable,
    StaleChainlinkFeed,
    StaleBinanceFeed,
    SourceTimestampSkew,
    MissingUpBook,
    MissingDownBook,
    BookFeedUnhealthy,
    StaleBook,
    CrossedBook,
    MissingFeeRate,
    StaleFeeRate,
    InvalidFeeRate,
    InvalidFeatureValue,
    PriceOutsideBounds,
    InsufficientDepth,
    BelowMinimumOrderSize,
    EqualEdge,
    EdgeBelowThreshold,
    PredictionConfidenceBelowThreshold,
    PredictionDirectEdgeNonPositive,
    VolatilityRegimeBelowThreshold,
    ContinuationSignalUnconfirmed,
    MarketPriorOutsideBounds,
    ExistingProcessEntry,
    RuntimeNotReady,
}

impl BtcRejectReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidConfiguration => "invalid_configuration",
            Self::FeatureSchemaMismatch => "feature_schema_mismatch",
            Self::MarketInactive => "market_inactive",
            Self::MarketClosed => "market_closed",
            Self::OrdersNotAccepted => "orders_not_accepted",
            Self::OutsideEntryWindow => "outside_entry_window",
            Self::InvalidMarketIdentity => "invalid_market_identity",
            Self::InvalidResolutionSource => "invalid_resolution_source",
            Self::MissingChainlinkOpen => "missing_chainlink_open",
            Self::MissingChainlinkPrice => "missing_chainlink_price",
            Self::MissingChainlinkGap => "missing_chainlink_gap",
            Self::MissingChainlinkReturns => "missing_chainlink_returns",
            Self::MissingBinancePrice => "missing_binance_price",
            Self::MissingBinanceReturns => "missing_binance_returns",
            Self::MissingRealizedVolatility => "missing_realized_volatility",
            Self::MissingBasis => "missing_basis",
            Self::ChainlinkFeedUnhealthy => "chainlink_feed_unhealthy",
            Self::BinanceFeedUnhealthy => "binance_feed_unhealthy",
            Self::MissingLineage => "missing_lineage",
            Self::FutureInputTimestamp => "future_input_timestamp",
            Self::InvalidDirectionalFeatureTimestamp => "invalid_directional_feature_timestamp",
            Self::FutureDirectionalFeatures => "future_directional_features",
            Self::StaleDirectionalFeatures => "stale_directional_features",
            Self::DirectionalFeaturesUnavailable => "directional_features_unavailable",
            Self::StaleChainlinkFeed => "stale_chainlink_feed",
            Self::StaleBinanceFeed => "stale_binance_feed",
            Self::SourceTimestampSkew => "source_timestamp_skew",
            Self::MissingUpBook => "missing_up_book",
            Self::MissingDownBook => "missing_down_book",
            Self::BookFeedUnhealthy => "book_feed_unhealthy",
            Self::StaleBook => "stale_book",
            Self::CrossedBook => "crossed_book",
            Self::MissingFeeRate => "missing_fee_rate",
            Self::StaleFeeRate => "stale_fee_rate",
            Self::InvalidFeeRate => "invalid_fee_rate",
            Self::InvalidFeatureValue => "invalid_feature_value",
            Self::PriceOutsideBounds => "price_outside_bounds",
            Self::InsufficientDepth => "insufficient_depth",
            Self::BelowMinimumOrderSize => "below_minimum_order_size",
            Self::EqualEdge => "equal_edge",
            Self::EdgeBelowThreshold => "edge_below_threshold",
            Self::PredictionConfidenceBelowThreshold => "prediction_confidence_below_threshold",
            Self::PredictionDirectEdgeNonPositive => "prediction_direct_edge_non_positive",
            Self::VolatilityRegimeBelowThreshold => "volatility_regime_below_threshold",
            Self::ContinuationSignalUnconfirmed => "continuation_signal_unconfirmed",
            Self::MarketPriorOutsideBounds => "market_prior_outside_bounds",
            // Preserve the immutable decision-evidence value written by existing processes.
            Self::ExistingProcessEntry => "existing_experiment_entry",
            Self::RuntimeNotReady => "runtime_not_ready",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum BtcStrategyPrediction {
    NoPrediction {
        reason: BtcRejectReason,
        minimum_conservative_probability: Decimal,
        up_probability: Decimal,
        down_probability: Decimal,
        up_conservative_probability: Decimal,
        down_conservative_probability: Decimal,
    },
    DirectionalPrediction {
        outcome: BtcOutcome,
        probability: Decimal,
        conservative_probability: Decimal,
        minimum_conservative_probability: Decimal,
        probability_uncertainty: Decimal,
        executable_price: Option<Decimal>,
        direct_taker_fee_per_share: Option<Decimal>,
        direct_net_edge_per_share: Option<Decimal>,
        #[serde(
            default,
            skip_serializing_if = "BtcDirectionalModelEntryPolicy::is_default"
        )]
        entry_policy: BtcDirectionalModelEntryPolicy,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BtcDecision {
    pub decision_id: Uuid,
    pub process_id: Uuid,
    pub feature_snapshot_id: Uuid,
    pub evaluated_at: DateTime<Utc>,
    pub action: BtcDecisionAction,
    pub reject_reason: Option<BtcRejectReason>,
    pub fair_value: Option<FairValueEstimate>,
    pub up_edge: Option<OutcomeEdge>,
    pub down_edge: Option<OutcomeEdge>,
    pub approved_intent: Option<ApprovedIntent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prediction: Option<BtcStrategyPrediction>,
}

#[derive(Debug, Clone, PartialEq)]
struct BtcStrategyEstimate {
    fair_value: FairValueEstimate,
}

#[derive(Debug, Clone, Copy)]
enum ResolvedBtcDecisionStrategy<'a> {
    BtcDirectionalModel {
        model_key: &'a str,
        artifact_sha256: &'a str,
        feature_schema_sha256: &'a str,
    },
    BtcAsymmetricValueModel {
        model_key: &'a str,
        artifact_sha256: &'a str,
        feature_schema_sha256: &'a str,
    },
}

impl<'a> ResolvedBtcDecisionStrategy<'a> {
    fn resolve(config: &'a BtcStrategyConfig) -> Result<Self, BtcRejectReason> {
        if let Some(selection) = config.decision_strategy.as_ref() {
            return match selection {
                BtcDecisionStrategyConfig::BtcDirectionalModel {
                    model_key,
                    artifact_sha256,
                    feature_schema_sha256,
                } if config.strategy_version == BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION => {
                    let selection = RuntimeModelSelection {
                        model_key: model_key.clone(),
                        artifact_sha256: artifact_sha256.clone(),
                        feature_schema_sha256: feature_schema_sha256.clone(),
                    };
                    match runtime_model(&selection) {
                        Ok(model)
                            if model.feature_schema_version() == config.feature_schema_version
                                && !model.is_asymmetric_value() =>
                        {
                            Ok(Self::BtcDirectionalModel {
                                model_key,
                                artifact_sha256,
                                feature_schema_sha256,
                            })
                        }
                        _ => Err(BtcRejectReason::InvalidConfiguration),
                    }
                }
                BtcDecisionStrategyConfig::BtcAsymmetricValueModel {
                    model_key,
                    artifact_sha256,
                    feature_schema_sha256,
                } if config.strategy_version == BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_VERSION => {
                    let selection = RuntimeModelSelection {
                        model_key: model_key.clone(),
                        artifact_sha256: artifact_sha256.clone(),
                        feature_schema_sha256: feature_schema_sha256.clone(),
                    };
                    match runtime_model(&selection) {
                        Ok(model)
                            if model.feature_schema_version() == config.feature_schema_version
                                && model.is_asymmetric_value() =>
                        {
                            Ok(Self::BtcAsymmetricValueModel {
                                model_key,
                                artifact_sha256,
                                feature_schema_sha256,
                            })
                        }
                        _ => Err(BtcRejectReason::InvalidConfiguration),
                    }
                }
                _ => Err(BtcRejectReason::InvalidConfiguration),
            };
        }
        Err(BtcRejectReason::InvalidConfiguration)
    }

    fn estimate(
        self,
        _config: &BtcStrategyConfig,
        snapshot: &BtcFeatureSnapshot,
    ) -> Result<BtcStrategyEstimate, BtcRejectReason> {
        match self {
            Self::BtcDirectionalModel {
                model_key,
                artifact_sha256,
                feature_schema_sha256,
            } => Ok(BtcStrategyEstimate {
                fair_value: estimate_btc_directional_model(
                    snapshot,
                    model_key,
                    artifact_sha256,
                    feature_schema_sha256,
                )?,
            }),
            Self::BtcAsymmetricValueModel {
                model_key,
                artifact_sha256,
                feature_schema_sha256,
            } => Ok(BtcStrategyEstimate {
                fair_value: estimate_btc_asymmetric_value_model(
                    snapshot,
                    model_key,
                    artifact_sha256,
                    feature_schema_sha256,
                )?,
            }),
        }
    }
}

pub struct DeterministicBtcStrategy;

struct BtcDirectionalModelEstimate {
    fair_value: FairValueEstimate,
    score: RuntimeModelScore,
    payoff_aware: bool,
}

impl DeterministicBtcStrategy {
    pub fn evaluate(config: &BtcStrategyConfig, snapshot: &BtcFeatureSnapshot) -> BtcDecision {
        Self::evaluate_with_directional_model_entry_policy(
            config,
            snapshot,
            BtcDirectionalModelEntryPolicy::RequirePositiveDirectEdge,
        )
    }

    pub(crate) fn evaluate_with_directional_model_entry_policy(
        config: &BtcStrategyConfig,
        snapshot: &BtcFeatureSnapshot,
        directional_model_entry_policy: BtcDirectionalModelEntryPolicy,
    ) -> BtcDecision {
        let decision_id = deterministic_decision_id(config, snapshot);
        if let Err(reason) = validate_config(config) {
            return rejected(decision_id, snapshot, reason, None, None, None);
        }

        let strategy = match ResolvedBtcDecisionStrategy::resolve(config) {
            Ok(strategy) => strategy,
            Err(reason) => return rejected(decision_id, snapshot, reason, None, None, None),
        };
        if let ResolvedBtcDecisionStrategy::BtcDirectionalModel {
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        } = strategy
        {
            if let Err(reason) = validate_btc_directional_model_snapshot(
                config,
                snapshot,
                model_key,
                artifact_sha256,
                feature_schema_sha256,
            ) {
                return rejected(decision_id, snapshot, reason, None, None, None);
            }
            let estimate = match score_btc_directional_model(
                snapshot,
                model_key,
                artifact_sha256,
                feature_schema_sha256,
            ) {
                Ok(value) => value,
                Err(reason) => return rejected(decision_id, snapshot, reason, None, None, None),
            };
            let minimum = btc_directional_model_confidence_threshold(
                snapshot,
                model_key,
                artifact_sha256,
                feature_schema_sha256,
            )
            .unwrap_or(Decimal::ONE);
            if estimate.payoff_aware
                && (!estimate.score.accepted
                    || estimate.score.action == RuntimeModelAction::NoTrade)
            {
                return rejected_with_prediction(
                    decision_id,
                    snapshot,
                    BtcRejectReason::PredictionConfidenceBelowThreshold,
                    Some(estimate.fair_value.clone()),
                    None,
                    None,
                    BtcStrategyPrediction::NoPrediction {
                        reason: BtcRejectReason::PredictionConfidenceBelowThreshold,
                        minimum_conservative_probability: minimum,
                        up_probability: estimate.fair_value.up_probability,
                        down_probability: estimate.fair_value.down_probability,
                        up_conservative_probability: estimate.fair_value.up_lower_bound,
                        down_conservative_probability: estimate.fair_value.down_lower_bound,
                    },
                );
            }
            if estimate.payoff_aware
                && !(config.unified_model.is_some()
                    && estimate.score.action == RuntimeModelAction::Up
                    && estimate.fair_value.up_probability == estimate.fair_value.down_probability)
                && !matches!(
                    (
                        estimate.score.action,
                        estimate
                            .fair_value
                            .up_probability
                            .cmp(&estimate.fair_value.down_probability)
                    ),
                    (RuntimeModelAction::Up, std::cmp::Ordering::Greater)
                        | (RuntimeModelAction::Down, std::cmp::Ordering::Less)
                )
            {
                return rejected(
                    decision_id,
                    snapshot,
                    BtcRejectReason::InvalidFeatureValue,
                    Some(estimate.fair_value),
                    None,
                    None,
                );
            }
            let mut decision = build_directional_prediction_decision(
                config,
                minimum,
                snapshot,
                decision_id,
                estimate.fair_value,
                directional_model_entry_policy,
            );
            if matches!(
                decision.prediction,
                Some(BtcStrategyPrediction::DirectionalPrediction { .. })
            ) {
                if let Err(reason) = validate_snapshot(config, snapshot, true) {
                    if let Some(prediction) = decision.prediction.take() {
                        return rejected_with_prediction(
                            decision_id,
                            snapshot,
                            reason,
                            decision.fair_value,
                            decision.up_edge,
                            decision.down_edge,
                            prediction,
                        );
                    }
                    return rejected(
                        decision_id,
                        snapshot,
                        BtcRejectReason::InvalidFeatureValue,
                        decision.fair_value,
                        decision.up_edge,
                        decision.down_edge,
                    );
                }
            }
            return decision;
        }
        if let Err(reason) = validate_snapshot(config, snapshot, false) {
            return rejected(decision_id, snapshot, reason, None, None, None);
        }
        let estimate = match strategy.estimate(config, snapshot) {
            Ok(value) => value,
            Err(reason) => return rejected(decision_id, snapshot, reason, None, None, None),
        };
        match strategy {
            ResolvedBtcDecisionStrategy::BtcDirectionalModel { .. } => {
                unreachable!("BTC directional model is evaluated before execution validation")
            }
            ResolvedBtcDecisionStrategy::BtcAsymmetricValueModel { .. } => {
                build_decision_from_estimate(config, snapshot, decision_id, estimate)
            }
        }
    }
}

fn build_directional_prediction_decision(
    config: &BtcStrategyConfig,
    minimum: Decimal,
    snapshot: &BtcFeatureSnapshot,
    decision_id: Uuid,
    fair_value: FairValueEstimate,
    entry_policy: BtcDirectionalModelEntryPolicy,
) -> BtcDecision {
    let (outcome, probability, conservative_probability, book) = if fair_value.up_probability
        > fair_value.down_probability
        || (config.unified_model.is_some()
            && fair_value.up_probability == fair_value.down_probability)
    {
        (
            BtcOutcome::Up,
            fair_value.up_probability,
            fair_value.up_lower_bound,
            &snapshot.up_book,
        )
    } else if fair_value.down_probability > fair_value.up_probability {
        (
            BtcOutcome::Down,
            fair_value.down_probability,
            fair_value.down_lower_bound,
            &snapshot.down_book,
        )
    } else {
        return rejected_with_prediction(
            decision_id,
            snapshot,
            BtcRejectReason::PredictionConfidenceBelowThreshold,
            Some(fair_value.clone()),
            None,
            None,
            BtcStrategyPrediction::NoPrediction {
                reason: BtcRejectReason::PredictionConfidenceBelowThreshold,
                minimum_conservative_probability: minimum,
                up_probability: fair_value.up_probability,
                down_probability: fair_value.down_probability,
                up_conservative_probability: fair_value.up_lower_bound,
                down_conservative_probability: fair_value.down_lower_bound,
            },
        );
    };

    if conservative_probability < minimum {
        return rejected_with_prediction(
            decision_id,
            snapshot,
            BtcRejectReason::PredictionConfidenceBelowThreshold,
            Some(fair_value.clone()),
            None,
            None,
            BtcStrategyPrediction::NoPrediction {
                reason: BtcRejectReason::PredictionConfidenceBelowThreshold,
                minimum_conservative_probability: minimum,
                up_probability: fair_value.up_probability,
                down_probability: fair_value.down_probability,
                up_conservative_probability: fair_value.up_lower_bound,
                down_conservative_probability: fair_value.down_lower_bound,
            },
        );
    }

    let fee_rate = if snapshot.fees_enabled {
        snapshot.fee_rate.unwrap_or(Decimal::ZERO)
    } else {
        Decimal::ZERO
    };
    let executable_price = book.executable_ask_vwap;
    let direct_taker_fee_per_share = executable_price.map(|price| {
        dynamic_crypto_taker_fee(config.target_size, fee_rate, price) / config.target_size
    });
    let direct_net_edge_per_share = executable_price
        .zip(direct_taker_fee_per_share)
        .map(|(price, fee)| probability - price - fee);
    let prediction = BtcStrategyPrediction::DirectionalPrediction {
        outcome,
        probability,
        conservative_probability,
        minimum_conservative_probability: minimum,
        probability_uncertainty: fair_value.probability_uncertainty,
        executable_price,
        direct_taker_fee_per_share,
        direct_net_edge_per_share,
        entry_policy,
    };

    let selected = match quote_outcome_edge(
        config,
        book,
        conservative_probability,
        fee_rate,
        snapshot.minimum_order_size,
    ) {
        Ok(edge) => edge,
        Err(reason) => {
            return rejected_with_prediction(
                decision_id,
                snapshot,
                reason,
                Some(fair_value),
                None,
                None,
                prediction,
            )
        }
    };
    let direct_net_edge_per_share = direct_net_edge_per_share.unwrap_or(Decimal::MIN);
    let (up_edge, down_edge) = match outcome {
        BtcOutcome::Up => (Some(selected.clone()), None),
        BtcOutcome::Down => (None, Some(selected.clone())),
    };
    if entry_policy == BtcDirectionalModelEntryPolicy::RequirePositiveDirectEdge
        && direct_net_edge_per_share <= Decimal::ZERO
    {
        return rejected_with_prediction(
            decision_id,
            snapshot,
            BtcRejectReason::PredictionDirectEdgeNonPositive,
            Some(fair_value),
            up_edge,
            down_edge,
            prediction,
        );
    }

    let mut decision = approved(
        decision_id,
        config,
        snapshot,
        fair_value,
        up_edge,
        down_edge,
        selected,
    );
    if let Some(intent) = decision.approved_intent.as_mut() {
        intent.expected_net_edge_per_share = direct_net_edge_per_share;
        intent.expected_net_edge = direct_net_edge_per_share * intent.size;
    }
    decision.prediction = Some(prediction);
    decision
}

fn build_decision_from_estimate(
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
    decision_id: Uuid,
    estimate: BtcStrategyEstimate,
) -> BtcDecision {
    let BtcStrategyEstimate { fair_value } = estimate;
    let fee_rate = if snapshot.fees_enabled {
        snapshot.fee_rate.unwrap_or(Decimal::ZERO)
    } else {
        Decimal::ZERO
    };

    let up_edge = quote_outcome_edge(
        config,
        &snapshot.up_book,
        fair_value.up_lower_bound,
        fee_rate,
        snapshot.minimum_order_size,
    );
    let down_edge = quote_outcome_edge(
        config,
        &snapshot.down_book,
        fair_value.down_lower_bound,
        fee_rate,
        snapshot.minimum_order_size,
    );

    let (up_edge, down_edge) = match (up_edge, down_edge) {
        (Ok(up), Ok(down)) => (up, down),
        (Err(up_reason), Err(down_reason)) => {
            let reason = if up_reason == down_reason {
                up_reason
            } else {
                preferred_execution_reject(up_reason, down_reason)
            };
            return rejected(decision_id, snapshot, reason, Some(fair_value), None, None);
        }
        (Err(_), Ok(down)) => {
            return finish_selected_edge(
                config,
                snapshot,
                decision_id,
                fair_value,
                None,
                Some(down.clone()),
                down,
            )
        }
        (Ok(up), Err(_)) => {
            return finish_selected_edge(
                config,
                snapshot,
                decision_id,
                fair_value,
                Some(up.clone()),
                None,
                up,
            )
        }
    };

    if up_edge.net_edge == down_edge.net_edge {
        return rejected(
            decision_id,
            snapshot,
            BtcRejectReason::EqualEdge,
            Some(fair_value),
            Some(up_edge),
            Some(down_edge),
        );
    }

    let selected = if up_edge.net_edge > down_edge.net_edge {
        up_edge.clone()
    } else {
        down_edge.clone()
    };
    if !edge_passes(config, &selected) {
        return rejected(
            decision_id,
            snapshot,
            BtcRejectReason::EdgeBelowThreshold,
            Some(fair_value),
            Some(up_edge),
            Some(down_edge),
        );
    }

    approved(
        decision_id,
        config,
        snapshot,
        fair_value,
        Some(up_edge),
        Some(down_edge),
        selected,
    )
}

pub fn estimate_fair_value(
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
) -> Result<FairValueEstimate, BtcRejectReason> {
    match ResolvedBtcDecisionStrategy::resolve(config)? {
        ResolvedBtcDecisionStrategy::BtcDirectionalModel {
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        } => estimate_btc_directional_model(
            snapshot,
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        ),
        ResolvedBtcDecisionStrategy::BtcAsymmetricValueModel {
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        } => estimate_btc_asymmetric_value_model(
            snapshot,
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        ),
    }
}

fn btc_directional_model_selection(
    model_key: &str,
    artifact_sha256: &str,
    feature_schema_sha256: &str,
) -> RuntimeModelSelection {
    RuntimeModelSelection {
        model_key: model_key.to_string(),
        artifact_sha256: artifact_sha256.to_string(),
        feature_schema_sha256: feature_schema_sha256.to_string(),
    }
}

fn btc_directional_model_confidence_threshold(
    snapshot: &BtcFeatureSnapshot,
    model_key: &str,
    artifact_sha256: &str,
    feature_schema_sha256: &str,
) -> Result<Decimal, BtcRejectReason> {
    let selection =
        btc_directional_model_selection(model_key, artifact_sha256, feature_schema_sha256);
    let model = runtime_model(&selection).map_err(|_| BtcRejectReason::InvalidConfiguration)?;
    let features = snapshot
        .directional_model
        .as_ref()
        .ok_or(BtcRejectReason::MissingBinanceReturns)?;
    decimal_from_f64(
        model
            .confidence_threshold_at(features.seconds_elapsed)
            .map_err(|_| BtcRejectReason::OutsideEntryWindow)?,
    )
}

fn estimate_btc_directional_model(
    snapshot: &BtcFeatureSnapshot,
    model_key: &str,
    artifact_sha256: &str,
    feature_schema_sha256: &str,
) -> Result<FairValueEstimate, BtcRejectReason> {
    Ok(
        score_btc_directional_model(snapshot, model_key, artifact_sha256, feature_schema_sha256)?
            .fair_value,
    )
}

fn score_btc_directional_model(
    snapshot: &BtcFeatureSnapshot,
    model_key: &str,
    artifact_sha256: &str,
    feature_schema_sha256: &str,
) -> Result<BtcDirectionalModelEstimate, BtcRejectReason> {
    let selection =
        btc_directional_model_selection(model_key, artifact_sha256, feature_schema_sha256);
    let model = runtime_model(&selection).map_err(|_| BtcRejectReason::InvalidConfiguration)?;
    let features = snapshot
        .directional_model
        .as_ref()
        .ok_or(BtcRejectReason::MissingBinanceReturns)?;
    let started = std::time::Instant::now();
    let inference = (|| -> Result<_, BtcRejectReason> {
        Ok(if let Some(adapter) = model.unified_adapter() {
            model
                .validate_snapshot_identity(features)
                .map_err(|error| {
                    super::unified_model_runtime::telemetry::failure(
                        snapshot.process_id,
                        "inference",
                        &error.to_string(),
                    );
                    BtcRejectReason::InvalidFeatureValue
                })?;
            let result = adapter
                .evaluate(&features.feature_values, features.seconds_elapsed)
                .map_err(|error| {
                    super::unified_model_runtime::telemetry::failure(
                        snapshot.process_id,
                        "inference",
                        &error.to_string(),
                    );
                    BtcRejectReason::InvalidFeatureValue
                })?;
            (result.score, serde_json::to_value(result).ok())
        } else {
            (
                model.score_snapshot(features).map_err(|error| {
                    super::unified_model_runtime::telemetry::failure(
                        snapshot.process_id,
                        "inference",
                        &error.to_string(),
                    );
                    BtcRejectReason::InvalidFeatureValue
                })?,
                None,
            )
        })
    })();
    let (score, admission) = inference.inspect_err(|_| {
        super::unified_model_runtime::telemetry::event(snapshot.process_id, "inferences", "error");
    })?;
    super::unified_model_runtime::telemetry::prediction(
        snapshot.process_id,
        snapshot.snapshot_id,
        &snapshot.market_id,
        &selection,
        features.feature_as_of,
        &features.input_sha256,
        score,
        started.elapsed().as_secs_f64(),
        admission,
    );
    let up_probability = decimal_from_f64(score.probability_up)?;
    let down_probability = Decimal::ONE - up_probability;
    let raw_logit = decimal_from_f64(score.raw_logit)?;
    let fair_value = FairValueEstimate {
        up_probability,
        down_probability,
        up_lower_bound: up_probability,
        up_upper_bound: up_probability,
        down_lower_bound: down_probability,
        down_upper_bound: down_probability,
        z_score: raw_logit,
        chainlink_log_gap: Decimal::ZERO,
        lead_adjustment: Decimal::ZERO,
        terminal_volatility: snapshot.realized_volatility.unwrap_or(Decimal::ZERO),
        probability_uncertainty: Decimal::ZERO,
        market_up_prior: None,
        signed_logit_adjustment: Some(raw_logit),
        chainlink_persistence_score: None,
        binance_confirmation_score: None,
        confirmation_deficiency_score: None,
        evidence_reliability: None,
        uncalibrated_up_probability: None,
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
        estimator_id: Some(BTC_DIRECTIONAL_MODEL_STRATEGY_FAMILY.to_string()),
        estimator_profile_id: Some(model_key.to_string()),
        estimator_profile_sha256: Some(artifact_sha256.to_string()),
    };
    Ok(BtcDirectionalModelEstimate {
        fair_value,
        score,
        payoff_aware: model.is_payoff_aware(),
    })
}

fn estimate_btc_asymmetric_value_model(
    snapshot: &BtcFeatureSnapshot,
    model_key: &str,
    artifact_sha256: &str,
    feature_schema_sha256: &str,
) -> Result<FairValueEstimate, BtcRejectReason> {
    let selection =
        btc_directional_model_selection(model_key, artifact_sha256, feature_schema_sha256);
    let model = runtime_model(&selection).map_err(|_| BtcRejectReason::InvalidConfiguration)?;
    let features = snapshot
        .directional_model
        .as_ref()
        .ok_or(BtcRejectReason::MissingBinanceReturns)?;
    let yes_ask_vwap = snapshot
        .up_book
        .executable_ask_vwap
        .and_then(|value| value.to_f64())
        .ok_or(BtcRejectReason::MissingUpBook)?;
    let no_ask_vwap = snapshot
        .down_book
        .executable_ask_vwap
        .and_then(|value| value.to_f64())
        .ok_or(BtcRejectReason::MissingDownBook)?;
    let score = model
        .score_asymmetric_value_snapshot(features, yes_ask_vwap, no_ask_vwap)
        .map_err(|_| BtcRejectReason::InvalidFeatureValue)?;
    let up_probability = decimal_from_f64(score.probability_up)?;
    let down_probability = Decimal::ONE - up_probability;
    let raw_logit = decimal_from_f64(score.raw_logit)?;
    Ok(FairValueEstimate {
        up_probability,
        down_probability,
        up_lower_bound: up_probability,
        up_upper_bound: up_probability,
        down_lower_bound: down_probability,
        down_upper_bound: down_probability,
        z_score: raw_logit,
        chainlink_log_gap: Decimal::ZERO,
        lead_adjustment: Decimal::ZERO,
        terminal_volatility: snapshot.realized_volatility.unwrap_or(Decimal::ZERO),
        probability_uncertainty: Decimal::ZERO,
        market_up_prior: None,
        signed_logit_adjustment: Some(raw_logit),
        chainlink_persistence_score: None,
        binance_confirmation_score: None,
        confirmation_deficiency_score: None,
        evidence_reliability: None,
        uncalibrated_up_probability: None,
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
        estimator_id: Some(BTC_ASYMMETRIC_VALUE_MODEL_STRATEGY_FAMILY.to_string()),
        estimator_profile_id: Some(model_key.to_string()),
        estimator_profile_sha256: Some(artifact_sha256.to_string()),
    })
}

fn validate_config(config: &BtcStrategyConfig) -> Result<(), BtcRejectReason> {
    if !config.required_model_feeds.is_empty()
        && validate_feed_requirements(&config.required_model_feeds).is_err()
    {
        return Err(BtcRejectReason::InvalidConfiguration);
    }
    let strategy_contract_valid = match ResolvedBtcDecisionStrategy::resolve(config) {
        Ok(ResolvedBtcDecisionStrategy::BtcDirectionalModel {
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        }) => {
            let selection =
                btc_directional_model_selection(model_key, artifact_sha256, feature_schema_sha256);
            runtime_model(&selection).is_ok_and(|model| {
                let policy = model.prediction_policy();
                (match (model.unified_adapter(), config.unified_model.as_ref()) {
                    (Some(adapter), Some(binding)) => config
                        .target_size
                        .to_f64()
                        .is_some_and(|size| binding.validate(adapter, size).is_ok()),
                    (None, None) => true,
                    _ => false,
                }) && model.feature_schema_version() == config.feature_schema_version
                    && policy.minimum_seconds_after_open == config.min_seconds_after_open
                    && 300 - policy.maximum_seconds_after_open == config.min_seconds_before_close
                    && policy.cadence_seconds > 0
                    && config.max_directional_feature_age_ms.is_none_or(|max_age| {
                        max_age > 0 && max_age <= policy.cadence_seconds * 1_000
                    })
                    && config.max_reference_age_ms == config.max_book_age_ms
                    && model.probability_up_threshold() == 0.5
                    && (model.is_payoff_aware() || model.confidence_threshold() > 0.5)
                    && model.confidence_threshold() < 1.0
            })
        }
        Ok(ResolvedBtcDecisionStrategy::BtcAsymmetricValueModel {
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        }) => {
            let selection =
                btc_directional_model_selection(model_key, artifact_sha256, feature_schema_sha256);
            runtime_model(&selection).is_ok_and(|model| {
                let policy = model.prediction_policy();
                let minimum_cadence_seconds = policy
                    .early_cadence_seconds
                    .unwrap_or(policy.cadence_seconds);
                model.is_asymmetric_value()
                    && model.feature_schema_version() == config.feature_schema_version
                    && config.min_seconds_after_open >= policy.minimum_seconds_after_open
                    && 300 - config.min_seconds_before_close <= policy.maximum_seconds_after_open
                    && minimum_cadence_seconds > 0
                    && config.max_directional_feature_age_ms.is_none_or(|max_age| {
                        max_age > 0 && max_age <= minimum_cadence_seconds * 1_000
                    })
                    && config.max_reference_age_ms == config.max_book_age_ms
            })
        }
        Err(_) => false,
    };
    let valid = !config.strategy_version.trim().is_empty()
        && !config.feature_schema_version.trim().is_empty()
        && config.target_size > Decimal::ZERO
        && config.min_seconds_after_open >= 0
        && config.min_seconds_before_close > 0
        && config.max_reference_age_ms > 0
        && config
            .max_directional_feature_age_ms
            .is_none_or(|max_age| max_age > 0)
        && config.max_chainlink_open_delay_ms > 0
        && config.max_book_age_ms > 0
        && config.max_source_skew_ms >= 0
        && config.max_fee_age_ms > 0
        && config.min_entry_price > Decimal::ZERO
        && config.max_entry_price < Decimal::ONE
        && config.min_entry_price < config.max_entry_price
        && config.max_depth_participation > Decimal::ZERO
        && config.max_depth_participation <= Decimal::ONE
        && config.volatility_floor_per_sqrt_second > Decimal::ZERO
        && config.probability_floor > Decimal::ZERO
        && config.probability_floor < dec!(0.5)
        && config.basis_lead_weight >= Decimal::ZERO
        && config.momentum_1s_weight >= Decimal::ZERO
        && config.momentum_5s_weight >= Decimal::ZERO
        && config.momentum_30s_weight >= Decimal::ZERO
        && config.max_lead_sigma_fraction >= Decimal::ZERO
        && config.base_probability_uncertainty >= Decimal::ZERO
        && config.basis_uncertainty_weight >= Decimal::ZERO
        && config.feed_age_uncertainty_per_second >= Decimal::ZERO
        && config.max_probability_uncertainty >= config.base_probability_uncertainty
        && config.max_probability_uncertainty < dec!(0.5)
        && config.spread_reserve_fraction >= Decimal::ZERO
        && config.slippage_reserve_bps >= Decimal::ZERO
        && config.latency_reserve_per_share >= Decimal::ZERO
        && config.min_net_edge_per_share >= Decimal::ZERO
        && config.min_net_edge_usd >= Decimal::ZERO
        && config.max_fee_rate >= Decimal::ZERO
        && strategy_contract_valid;
    if valid {
        Ok(())
    } else {
        Err(BtcRejectReason::InvalidConfiguration)
    }
}

fn validate_snapshot(
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
    directional_model_input_validated: bool,
) -> Result<(), BtcRejectReason> {
    if snapshot.feature_schema_version != config.feature_schema_version {
        return Err(BtcRejectReason::FeatureSchemaMismatch);
    }
    if !snapshot.market_active {
        return Err(BtcRejectReason::MarketInactive);
    }
    if snapshot.market_closed {
        return Err(BtcRejectReason::MarketClosed);
    }
    if !snapshot.accepting_orders {
        return Err(BtcRejectReason::OrdersNotAccepted);
    }
    let resolved = ResolvedBtcDecisionStrategy::resolve(config)?;
    let (model_key, artifact_sha256, feature_schema_sha256) = match resolved {
        ResolvedBtcDecisionStrategy::BtcDirectionalModel {
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        }
        | ResolvedBtcDecisionStrategy::BtcAsymmetricValueModel {
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        } => (model_key, artifact_sha256, feature_schema_sha256),
    };
    if !directional_model_input_validated {
        validate_btc_directional_model_snapshot(
            config,
            snapshot,
            model_key,
            artifact_sha256,
            feature_schema_sha256,
        )?;
    }
    if matches!(
        resolved,
        ResolvedBtcDecisionStrategy::BtcAsymmetricValueModel { .. }
    ) {
        let earliest_entry =
            snapshot.window_start + chrono::Duration::seconds(config.min_seconds_after_open);
        let latest_entry =
            snapshot.window_end - chrono::Duration::seconds(config.min_seconds_before_close);
        if snapshot.observed_at < earliest_entry || snapshot.observed_at >= latest_entry {
            return Err(BtcRejectReason::OutsideEntryWindow);
        }
    }
    if snapshot.market_id.trim().is_empty()
        || snapshot.event_slug.trim().is_empty()
        || snapshot.up_book.token_id.trim().is_empty()
        || snapshot.down_book.token_id.trim().is_empty()
        || snapshot.up_book.token_id == snapshot.down_book.token_id
        || snapshot.up_book.outcome != BtcOutcome::Up
        || snapshot.down_book.outcome != BtcOutcome::Down
    {
        return Err(BtcRejectReason::InvalidMarketIdentity);
    }
    if snapshot.tick_size <= Decimal::ZERO
        || snapshot
            .minimum_order_size
            .map(|size| size <= Decimal::ZERO)
            .unwrap_or(false)
    {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    let resolution_source = snapshot.resolution_source.to_ascii_lowercase();
    if !resolution_source.contains("chainlink") && !resolution_source.contains("chain.link") {
        return Err(BtcRejectReason::InvalidResolutionSource);
    }
    validate_book(
        config,
        snapshot,
        &snapshot.up_book,
        BtcRejectReason::MissingUpBook,
    )?;
    validate_book(
        config,
        snapshot,
        &snapshot.down_book,
        BtcRejectReason::MissingDownBook,
    )?;
    validate_btc_directional_model_execution_lineage(snapshot)?;
    validate_fee(config, snapshot)?;
    Ok(())
}

fn validate_btc_directional_model_snapshot(
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
    model_key: &str,
    artifact_sha256: &str,
    feature_schema_sha256: &str,
) -> Result<(), BtcRejectReason> {
    if snapshot.feature_schema_version != config.feature_schema_version {
        return Err(BtcRejectReason::FeatureSchemaMismatch);
    }
    let features = snapshot
        .directional_model
        .as_ref()
        .ok_or(BtcRejectReason::MissingBinanceReturns)?;
    if features.model_key != model_key
        || features.model_artifact_sha256 != artifact_sha256
        || features.feature_schema_version != config.feature_schema_version
        || features.feature_schema_sha256 != feature_schema_sha256
    {
        return Err(BtcRejectReason::FeatureSchemaMismatch);
    }
    let selection =
        btc_directional_model_selection(model_key, artifact_sha256, feature_schema_sha256);
    let model = runtime_model(&selection).map_err(|_| BtcRejectReason::InvalidConfiguration)?;
    if !model.prediction_policy().accepts(features.seconds_elapsed) {
        return Err(BtcRejectReason::OutsideEntryWindow);
    }
    let expected_feature_as_of =
        snapshot.window_start + chrono::Duration::seconds(features.seconds_elapsed);
    let feature_age_ms = (snapshot.observed_at - features.feature_as_of).num_milliseconds();
    if features.feature_as_of != expected_feature_as_of {
        return Err(BtcRejectReason::InvalidDirectionalFeatureTimestamp);
    }
    if feature_age_ms < 0 {
        return Err(BtcRejectReason::FutureDirectionalFeatures);
    }
    let cadence_seconds = model
        .prediction_policy()
        .early_cadence_seconds
        .filter(|_| {
            model
                .prediction_policy()
                .early_end_second
                .is_some_and(|end| features.seconds_elapsed <= end)
        })
        .unwrap_or(model.prediction_policy().cadence_seconds);
    let max_directional_feature_age_ms = config
        .max_directional_feature_age_ms
        .unwrap_or(cadence_seconds * 1_000);
    if feature_age_ms > max_directional_feature_age_ms {
        return Err(BtcRejectReason::StaleDirectionalFeatures);
    }
    let input_sha256 = if model.is_asymmetric_value() {
        asymmetric_value_model_input_sha256(
            &selection,
            &features.feature_schema_version,
            &snapshot.market_id,
            snapshot.window_start,
            features.feature_as_of,
            features.seconds_elapsed,
            &features.feature_values,
            snapshot
                .up_book
                .executable_ask_vwap
                .and_then(|value| value.to_f64())
                .ok_or(BtcRejectReason::MissingUpBook)?,
            snapshot
                .down_book
                .executable_ask_vwap
                .and_then(|value| value.to_f64())
                .ok_or(BtcRejectReason::MissingDownBook)?,
        )
    } else {
        directional_model_input_sha256(
            &selection,
            &features.feature_schema_version,
            &snapshot.market_id,
            snapshot.window_start,
            features.feature_as_of,
            features.seconds_elapsed,
            &features.feature_values,
        )
    }
    .map_err(|_| BtcRejectReason::InvalidFeatureValue)?;
    if input_sha256 != features.input_sha256 {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    Ok(())
}

fn validate_btc_directional_model_execution_lineage(
    snapshot: &BtcFeatureSnapshot,
) -> Result<(), BtcRejectReason> {
    let binance_tick_id = snapshot
        .lineage
        .binance_tick_id
        .ok_or(BtcRejectReason::MissingLineage)?;
    let binance_source_timestamp = snapshot
        .lineage
        .binance_source_timestamp
        .ok_or(BtcRejectReason::MissingLineage)?;
    let binance_received_at = snapshot
        .lineage
        .binance_received_at
        .ok_or(BtcRejectReason::MissingLineage)?;
    let binance_ingest_sequence = snapshot
        .lineage
        .binance_ingest_sequence
        .ok_or(BtcRejectReason::MissingLineage)?;
    let up_checkpoint_id = snapshot
        .lineage
        .up_book_checkpoint_id
        .ok_or(BtcRejectReason::MissingLineage)?;
    let down_checkpoint_id = snapshot
        .lineage
        .down_book_checkpoint_id
        .ok_or(BtcRejectReason::MissingLineage)?;
    let up_ingest_sequence = snapshot
        .lineage
        .up_book_ingest_sequence
        .ok_or(BtcRejectReason::MissingLineage)?;
    let down_ingest_sequence = snapshot
        .lineage
        .down_book_ingest_sequence
        .ok_or(BtcRejectReason::MissingLineage)?;
    let up_connection_id = snapshot
        .lineage
        .up_book_connection_id
        .ok_or(BtcRejectReason::MissingLineage)?;
    let down_connection_id = snapshot
        .lineage
        .down_book_connection_id
        .ok_or(BtcRejectReason::MissingLineage)?;
    if binance_tick_id.is_nil()
        || up_checkpoint_id.is_nil()
        || down_checkpoint_id.is_nil()
        || binance_ingest_sequence == 0
        || up_ingest_sequence == 0
        || down_ingest_sequence == 0
        || up_connection_id.is_nil()
        || up_connection_id != down_connection_id
        || snapshot.up_book.connection_id != Some(up_connection_id)
        || snapshot.down_book.connection_id != Some(down_connection_id)
    {
        return Err(BtcRejectReason::MissingLineage);
    }
    if binance_source_timestamp > snapshot.observed_at || binance_received_at > snapshot.observed_at
    {
        return Err(BtcRejectReason::FutureInputTimestamp);
    }
    Ok(())
}

fn validate_book(
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
    book: &BtcOutcomeBookFeatures,
    missing_reason: BtcRejectReason,
) -> Result<(), BtcRejectReason> {
    let bid = book.best_bid.ok_or(missing_reason)?;
    let ask = book.best_ask.ok_or(missing_reason)?;
    let vwap = book.executable_ask_vwap.ok_or(missing_reason)?;
    let limit = book.marketable_limit_price.ok_or(missing_reason)?;
    let source_timestamp = book
        .source_timestamp
        .ok_or(BtcRejectReason::MissingLineage)?;
    let received_at = book.received_at.ok_or(BtcRejectReason::MissingLineage)?;
    if source_timestamp > snapshot.observed_at || received_at > snapshot.observed_at {
        return Err(BtcRejectReason::FutureInputTimestamp);
    }
    if book.integrity_status != FeedIntegrityStatus::Ok {
        return Err(BtcRejectReason::BookFeedUnhealthy);
    }
    validate_age(
        book.age_ms,
        config.max_book_age_ms,
        BtcRejectReason::StaleBook,
    )?;
    if bid <= Decimal::ZERO
        || ask <= Decimal::ZERO
        || vwap <= Decimal::ZERO
        || limit <= Decimal::ZERO
        || bid >= ask
        || ask > Decimal::ONE
        || vwap > Decimal::ONE
        || limit > Decimal::ONE
        || vwap < ask
        || limit < vwap
    {
        return Err(if bid >= ask {
            BtcRejectReason::CrossedBook
        } else {
            BtcRejectReason::InvalidFeatureValue
        });
    }
    Ok(())
}

fn validate_fee(
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
) -> Result<(), BtcRejectReason> {
    if !snapshot.fees_enabled {
        return Ok(());
    }
    let fee_rate = snapshot.fee_rate.ok_or(BtcRejectReason::MissingFeeRate)?;
    if fee_rate < Decimal::ZERO || fee_rate > config.max_fee_rate {
        return Err(BtcRejectReason::InvalidFeeRate);
    }
    let observed_at = snapshot
        .fee_rate_observed_at
        .ok_or(BtcRejectReason::MissingFeeRate)?;
    if observed_at > snapshot.observed_at {
        return Err(BtcRejectReason::FutureInputTimestamp);
    }
    let age_ms = (snapshot.observed_at - observed_at).num_milliseconds();
    if age_ms > config.max_fee_age_ms {
        return Err(BtcRejectReason::StaleFeeRate);
    }
    Ok(())
}

fn quote_outcome_edge(
    config: &BtcStrategyConfig,
    book: &BtcOutcomeBookFeatures,
    conservative_probability: Decimal,
    fee_rate: Decimal,
    minimum_order_size: Option<Decimal>,
) -> Result<OutcomeEdge, BtcRejectReason> {
    let bid = book.best_bid.ok_or(match book.outcome {
        BtcOutcome::Up => BtcRejectReason::MissingUpBook,
        BtcOutcome::Down => BtcRejectReason::MissingDownBook,
    })?;
    let executable_price = book.executable_ask_vwap.ok_or(match book.outcome {
        BtcOutcome::Up => BtcRejectReason::MissingUpBook,
        BtcOutcome::Down => BtcRejectReason::MissingDownBook,
    })?;
    let limit = book.marketable_limit_price.ok_or(match book.outcome {
        BtcOutcome::Up => BtcRejectReason::MissingUpBook,
        BtcOutcome::Down => BtcRejectReason::MissingDownBook,
    })?;
    if executable_price < config.min_entry_price
        || executable_price > config.max_entry_price
        || limit > config.max_entry_price
    {
        return Err(BtcRejectReason::PriceOutsideBounds);
    }
    if book.quoted_size < config.target_size
        || book.ask_depth * config.max_depth_participation < config.target_size
    {
        return Err(BtcRejectReason::InsufficientDepth);
    }
    if let Some(minimum_order_size) = minimum_order_size {
        if config.target_size < minimum_order_size {
            return Err(BtcRejectReason::BelowMinimumOrderSize);
        }
    }

    let size = config.target_size;
    let gross_edge = (conservative_probability - executable_price) * size;
    let taker_fee = dynamic_crypto_taker_fee(size, fee_rate, executable_price);
    let spread_reserve =
        (executable_price - bid).max(Decimal::ZERO) * config.spread_reserve_fraction * size;
    let slippage_reserve = executable_price * size * config.slippage_reserve_bps / dec!(10000);
    let latency_reserve = config.latency_reserve_per_share * size;
    let net_edge = gross_edge - taker_fee - spread_reserve - slippage_reserve - latency_reserve;

    Ok(OutcomeEdge {
        outcome: book.outcome,
        token_id: book.token_id.clone(),
        conservative_probability,
        executable_price,
        marketable_limit_price: limit,
        size,
        gross_edge,
        taker_fee,
        spread_reserve,
        slippage_reserve,
        latency_reserve,
        net_edge,
        net_edge_per_share: net_edge / size,
    })
}

fn finish_selected_edge(
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
    decision_id: Uuid,
    fair_value: FairValueEstimate,
    up_edge: Option<OutcomeEdge>,
    down_edge: Option<OutcomeEdge>,
    selected: OutcomeEdge,
) -> BtcDecision {
    if !edge_passes(config, &selected) {
        return rejected(
            decision_id,
            snapshot,
            BtcRejectReason::EdgeBelowThreshold,
            Some(fair_value),
            up_edge,
            down_edge,
        );
    }
    approved(
        decision_id,
        config,
        snapshot,
        fair_value,
        up_edge,
        down_edge,
        selected,
    )
}

fn approved(
    decision_id: Uuid,
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
    fair_value: FairValueEstimate,
    up_edge: Option<OutcomeEdge>,
    down_edge: Option<OutcomeEdge>,
    selected: OutcomeEdge,
) -> BtcDecision {
    let action = match selected.outcome {
        BtcOutcome::Up => BtcDecisionAction::BuyUp,
        BtcOutcome::Down => BtcDecisionAction::BuyDown,
    };
    let intent_id = Uuid::new_v5(
        &Uuid::NAMESPACE_URL,
        format!(
            "polymarket-bot:{}:{}:{}:{}:{}:{}:entry",
            config.strategy_version,
            snapshot.process_id,
            snapshot.market_id,
            snapshot.window_start.timestamp(),
            snapshot.snapshot_id,
            match selected.outcome {
                BtcOutcome::Up => "up",
                BtcOutcome::Down => "down",
            }
        )
        .as_bytes(),
    );
    BtcDecision {
        decision_id,
        process_id: snapshot.process_id,
        feature_snapshot_id: snapshot.snapshot_id,
        evaluated_at: snapshot.observed_at,
        action,
        reject_reason: None,
        fair_value: Some(fair_value),
        up_edge,
        down_edge,
        approved_intent: Some(ApprovedIntent {
            intent_id,
            process_id: snapshot.process_id,
            feature_snapshot_id: snapshot.snapshot_id,
            market_id: snapshot.market_id.clone(),
            window_start: snapshot.window_start,
            outcome: selected.outcome,
            token_id: selected.token_id,
            limit_price: selected.marketable_limit_price,
            size: selected.size,
            expected_net_edge: selected.net_edge,
            expected_net_edge_per_share: selected.net_edge_per_share,
            strategy_version: config.strategy_version.clone(),
            feature_schema_version: config.feature_schema_version.clone(),
        }),
        prediction: None,
    }
}

fn rejected(
    decision_id: Uuid,
    snapshot: &BtcFeatureSnapshot,
    reason: BtcRejectReason,
    fair_value: Option<FairValueEstimate>,
    up_edge: Option<OutcomeEdge>,
    down_edge: Option<OutcomeEdge>,
) -> BtcDecision {
    BtcDecision {
        decision_id,
        process_id: snapshot.process_id,
        feature_snapshot_id: snapshot.snapshot_id,
        evaluated_at: snapshot.observed_at,
        action: BtcDecisionAction::NoTrade,
        reject_reason: Some(reason),
        fair_value,
        up_edge,
        down_edge,
        approved_intent: None,
        prediction: None,
    }
}

fn rejected_with_prediction(
    decision_id: Uuid,
    snapshot: &BtcFeatureSnapshot,
    reason: BtcRejectReason,
    fair_value: Option<FairValueEstimate>,
    up_edge: Option<OutcomeEdge>,
    down_edge: Option<OutcomeEdge>,
    prediction: BtcStrategyPrediction,
) -> BtcDecision {
    let mut decision = rejected(
        decision_id,
        snapshot,
        reason,
        fair_value,
        up_edge,
        down_edge,
    );
    decision.prediction = Some(prediction);
    decision
}

fn edge_passes(config: &BtcStrategyConfig, edge: &OutcomeEdge) -> bool {
    edge.net_edge >= config.min_net_edge_usd
        && edge.net_edge_per_share >= config.min_net_edge_per_share
}

fn validate_age(
    age_ms: Option<i64>,
    max_age_ms: i64,
    stale_reason: BtcRejectReason,
) -> Result<(), BtcRejectReason> {
    let age_ms = age_ms.ok_or(BtcRejectReason::MissingLineage)?;
    if age_ms < 0 {
        return Err(BtcRejectReason::FutureInputTimestamp);
    }
    if age_ms > max_age_ms {
        return Err(stale_reason);
    }
    Ok(())
}

fn deterministic_decision_id(config: &BtcStrategyConfig, snapshot: &BtcFeatureSnapshot) -> Uuid {
    Uuid::new_v5(
        &Uuid::NAMESPACE_URL,
        format!(
            "polymarket-bot:{}:{}:{}:{}",
            config.strategy_version, snapshot.process_id, snapshot.market_id, snapshot.snapshot_id
        )
        .as_bytes(),
    )
}

fn decimal_from_f64(value: f64) -> Result<Decimal, BtcRejectReason> {
    if !value.is_finite() {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    Decimal::from_f64(value).ok_or(BtcRejectReason::InvalidFeatureValue)
}

fn preferred_execution_reject(left: BtcRejectReason, right: BtcRejectReason) -> BtcRejectReason {
    fn rank(reason: BtcRejectReason) -> u8 {
        match reason {
            BtcRejectReason::InsufficientDepth => 0,
            BtcRejectReason::BelowMinimumOrderSize => 1,
            BtcRejectReason::PriceOutsideBounds => 2,
            _ => 3,
        }
    }
    if rank(left) <= rank(right) {
        left
    } else {
        right
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use chrono::{Duration, TimeZone};

    use super::super::directional_model::{
        BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION, BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256,
        BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256, BTC_DIRECTIONAL_MODEL_V1_KEY,
    };
    use super::*;
    use crate::{
        btc::execution_guard::{
            BtcExecutionFreshnessBounds, BtcReferenceExecutionGuard,
            BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION,
        },
        models::{OrderRequest, OrderSide, OrderType},
    };

    fn snapshot() -> BtcFeatureSnapshot {
        let window_start = Utc.with_ymd_and_hms(2026, 7, 12, 12, 0, 0).unwrap();
        let observed_at = window_start + Duration::seconds(180);
        let source_at = observed_at - Duration::milliseconds(100);
        BtcFeatureSnapshot {
            snapshot_id: Uuid::from_u128(1),
            process_id: Uuid::from_u128(2),
            observed_at,
            feature_schema_version: BTC_FEATURE_SCHEMA_VERSION.to_string(),
            market_id: "btc-market-1".to_string(),
            event_slug: "btc-updown-5m-1783857600".to_string(),
            window_start,
            window_end: window_start + Duration::seconds(300),
            market_active: true,
            market_closed: false,
            accepting_orders: true,
            tick_size: dec!(0.01),
            minimum_order_size: Some(dec!(1)),
            resolution_source: "chainlink btc/usd".to_string(),
            chainlink_open_price: Some(dec!(100000)),
            chainlink_price: Some(dec!(100080)),
            binance_price: Some(dec!(100090)),
            chainlink_gap_bps: Some(dec!(8)),
            chainlink_return_5s: None,
            chainlink_return_15s: None,
            chainlink_return_30s: None,
            chainlink_path_efficiency_30s: None,
            chainlink_path_tick_count_30s: None,
            chainlink_realized_volatility_5s: None,
            chainlink_realized_volatility_30s: None,
            binance_return_1s: Some(dec!(0.00005)),
            binance_return_5s: Some(dec!(0.00010)),
            binance_return_30s: Some(dec!(0.00020)),
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
            directional_model: None,
            lineage: BtcFeatureLineage {
                lineage_version: BTC_FEATURE_LINEAGE_VERSION.to_string(),
                chainlink_open_tick_id: Some(Uuid::from_u128(3)),
                chainlink_tick_id: Some(Uuid::from_u128(4)),
                binance_tick_id: Some(Uuid::from_u128(5)),
                up_book_checkpoint_id: Some(Uuid::from_u128(6)),
                down_book_checkpoint_id: Some(Uuid::from_u128(7)),
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
                up_book_connection_id: Some(Uuid::from_u128(8)),
                down_book_connection_id: Some(Uuid::from_u128(8)),
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
        let observed_at = Utc.with_ymd_and_hms(2026, 7, 12, 12, 3, 0).unwrap();
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
            imbalance: Some(dec!(0.2)),
            source_timestamp: Some(observed_at - Duration::milliseconds(100)),
            received_at: Some(observed_at - Duration::milliseconds(100)),
            age_ms: Some(100),
            integrity_status: FeedIntegrityStatus::Ok,
            connection_id: Some(Uuid::from_u128(8)),
        }
    }

    fn directional_model_config() -> BtcStrategyConfig {
        BtcStrategyConfig {
            strategy_version: BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION.to_string(),
            feature_schema_version: BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION.to_string(),
            decision_strategy: Some(BtcDecisionStrategyConfig::BtcDirectionalModel {
                model_key: BTC_DIRECTIONAL_MODEL_V1_KEY.to_string(),
                artifact_sha256: BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256.to_string(),
                feature_schema_sha256: BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256.to_string(),
            }),
            min_seconds_after_open: 60,
            min_seconds_before_close: 60,
            ..BtcStrategyConfig::default()
        }
    }

    #[test]
    fn payoff_aware_directional_model_uses_internal_admission_threshold() {
        let config = BtcStrategyConfig {
            strategy_version: BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION.to_string(),
            feature_schema_version: "btc-5m-payoff-aware-q5-features-v1".to_string(),
            decision_strategy: Some(BtcDecisionStrategyConfig::BtcDirectionalModel {
                model_key: "btc-5m-payoff-aware-q5-paper-20260820".to_string(),
                artifact_sha256: "ddb2c1cfedb9132d1120dfcdb405ab7737d808b124306e36da8f850ef51d803e"
                    .to_string(),
                feature_schema_sha256:
                    "eccc1787cfbb14f6faacc7ec20be6b2d6d16ca7fdac7c685be86410d63581158".to_string(),
            }),
            min_seconds_after_open: 15,
            min_seconds_before_close: 60,
            max_directional_feature_age_ms: Some(1_000),
            ..BtcStrategyConfig::default()
        };

        assert!(config.validate().is_ok());
    }

    fn payoff_model_case(
        directory_name: &str,
        vector_action: &str,
        force_missing_features: bool,
    ) -> (BtcStrategyConfig, BtcFeatureSnapshot) {
        let workspace_models = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("btc-directional-model/runtime-models");
        let root = if workspace_models.is_dir() {
            workspace_models
        } else {
            Path::new("/opt/polymarket-models").to_path_buf()
        };
        let directory = root.join(directory_name);
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("manifest.json")).unwrap()).unwrap();
        let golden: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("golden-vectors.json")).unwrap())
                .unwrap();
        let vector = golden["vectors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|vector| vector["expected"]["action"].as_str() == Some(vector_action))
            .unwrap();
        let model_key = manifest["model_key"].as_str().unwrap();
        let artifact_sha256 = manifest["model_sha256"].as_str().unwrap();
        let feature_schema_version = manifest["feature_schema_version"].as_str().unwrap();
        let feature_schema_sha256 = manifest["feature_schema_sha256"].as_str().unwrap();
        let selection =
            btc_directional_model_selection(model_key, artifact_sha256, feature_schema_sha256);
        let model = runtime_model(&selection).unwrap();
        let policy = model.prediction_policy();
        let seconds_elapsed = vector["seconds_elapsed"].as_i64().unwrap();
        let mut feature_values = vector["feature_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap_or(f64::NAN))
            .collect::<Vec<_>>();
        if force_missing_features {
            feature_values.fill(f64::NAN);
        }

        let config = BtcStrategyConfig {
            strategy_version: BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION.to_string(),
            feature_schema_version: feature_schema_version.to_string(),
            decision_strategy: Some(BtcDecisionStrategyConfig::BtcDirectionalModel {
                model_key: model_key.to_string(),
                artifact_sha256: artifact_sha256.to_string(),
                feature_schema_sha256: feature_schema_sha256.to_string(),
            }),
            min_seconds_after_open: policy.minimum_seconds_after_open,
            min_seconds_before_close: 300 - policy.maximum_seconds_after_open,
            max_directional_feature_age_ms: Some(
                policy
                    .early_cadence_seconds
                    .unwrap_or(policy.cadence_seconds)
                    * 1_000,
            ),
            max_entry_price: dec!(0.995),
            ..BtcStrategyConfig::default()
        };
        let mut snapshot = snapshot();
        snapshot.feature_schema_version = feature_schema_version.to_string();
        snapshot.observed_at = snapshot.window_start + Duration::seconds(seconds_elapsed);
        let source_at = snapshot.observed_at - Duration::milliseconds(100);
        snapshot.fee_rate_observed_at = Some(source_at);
        snapshot.lineage.chainlink_source_timestamp = Some(source_at);
        snapshot.lineage.chainlink_received_at = Some(source_at);
        snapshot.lineage.binance_source_timestamp = Some(source_at);
        snapshot.lineage.binance_received_at = Some(source_at);
        for book in [&mut snapshot.up_book, &mut snapshot.down_book] {
            book.source_timestamp = Some(source_at);
            book.received_at = Some(source_at);
        }
        let feature_as_of = snapshot.observed_at;
        let input_sha256 = directional_model_input_sha256(
            &selection,
            feature_schema_version,
            &snapshot.market_id,
            snapshot.window_start,
            feature_as_of,
            seconds_elapsed,
            &feature_values,
        )
        .unwrap();
        snapshot.directional_model = Some(BtcDirectionalModelFeatureSnapshot {
            model_key: model_key.to_string(),
            model_artifact_sha256: artifact_sha256.to_string(),
            feature_schema_version: feature_schema_version.to_string(),
            feature_schema_sha256: feature_schema_sha256.to_string(),
            feature_as_of,
            seconds_elapsed,
            feature_values,
            input_sha256,
        });
        (config, snapshot)
    }

    #[test]
    fn payoff_models_apply_native_admission_before_directional_execution() {
        let (config, snapshot) =
            payoff_model_case("btc-5m-payoff-aware-q5-paper-20260820", "no_trade", false);
        let decision = DeterministicBtcStrategy::evaluate_with_directional_model_entry_policy(
            &config,
            &snapshot,
            BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction,
        );
        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert!(decision.approved_intent.is_none());

        for directory in [
            "btc-5m-chainlink-regime-calibrated-paper-20260820",
            "btc-5m-chainlink-stratified-payoff-paper-20260820",
            "btc-5m-chainlink-full-combined-paper-20260820",
            "btc-5m-specialist-distilled-fair-value-paper-20260823-v1",
        ] {
            let (config, snapshot) = payoff_model_case(directory, "down", false);
            let accepted = DeterministicBtcStrategy::evaluate_with_directional_model_entry_policy(
                &config,
                &snapshot,
                BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction,
            );
            assert_eq!(accepted.action, BtcDecisionAction::BuyDown, "{directory}");
            assert!(accepted.approved_intent.is_some(), "{directory}");

            let (config, snapshot) = payoff_model_case(directory, "down", true);
            let rejected = DeterministicBtcStrategy::evaluate_with_directional_model_entry_policy(
                &config,
                &snapshot,
                BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction,
            );
            assert_eq!(rejected.action, BtcDecisionAction::NoTrade, "{directory}");
            assert!(rejected.approved_intent.is_none(), "{directory}");
        }
    }

    fn packaged_directional_model_vector(action: &str) -> Vec<f64> {
        let workspace_models = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("btc-directional-model/runtime-models");
        let root = if workspace_models.is_dir() {
            workspace_models
        } else {
            Path::new("/opt/polymarket-models").to_path_buf()
        };
        let path = root
            .join(BTC_DIRECTIONAL_MODEL_V1_KEY)
            .join("golden-vectors.json");
        let payload: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        payload["vectors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|vector| vector["expected"]["action"].as_str() == Some(action))
            .unwrap()["feature_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap_or(f64::NAN))
            .collect()
    }

    fn directional_model_snapshot(action: &str) -> BtcFeatureSnapshot {
        let mut snapshot = snapshot();
        snapshot.feature_schema_version = BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION.to_string();
        let mut features = BtcDirectionalModelFeatureSnapshot {
            model_key: BTC_DIRECTIONAL_MODEL_V1_KEY.to_string(),
            model_artifact_sha256: BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256.to_string(),
            feature_schema_version: BTC_DIRECTIONAL_MODEL_FEATURE_SCHEMA_VERSION.to_string(),
            feature_schema_sha256: BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256.to_string(),
            feature_as_of: snapshot.observed_at,
            seconds_elapsed: 180,
            feature_values: packaged_directional_model_vector(action),
            input_sha256: String::new(),
        };
        let selection = btc_directional_model_selection(
            BTC_DIRECTIONAL_MODEL_V1_KEY,
            BTC_DIRECTIONAL_MODEL_V1_ARTIFACT_SHA256,
            BTC_DIRECTIONAL_MODEL_V1_FEATURE_SCHEMA_SHA256,
        );
        features.input_sha256 = directional_model_input_sha256(
            &selection,
            &features.feature_schema_version,
            &snapshot.market_id,
            snapshot.window_start,
            features.feature_as_of,
            features.seconds_elapsed,
            &features.feature_values,
        )
        .unwrap();
        snapshot.directional_model = Some(features);
        snapshot
    }

    #[test]
    fn directional_model_abstain_maps_to_existing_no_trade_contract() {
        let config = directional_model_config();
        let snapshot = directional_model_snapshot("no_trade");

        for entry_policy in [
            BtcDirectionalModelEntryPolicy::RequirePositiveDirectEdge,
            BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction,
        ] {
            let decision = DeterministicBtcStrategy::evaluate_with_directional_model_entry_policy(
                &config,
                &snapshot,
                entry_policy,
            );

            assert_eq!(decision.action, BtcDecisionAction::NoTrade);
            assert_eq!(
                decision.reject_reason,
                Some(BtcRejectReason::PredictionConfidenceBelowThreshold)
            );
            assert!(matches!(
                decision.prediction,
                Some(BtcStrategyPrediction::NoPrediction { .. })
            ));
            assert!(decision.approved_intent.is_none());
        }
    }

    #[test]
    fn directional_model_paper_validation_executes_real_prediction_with_signed_negative_edge() {
        for (action, expected_outcome, expected_action) in [
            ("up", BtcOutcome::Up, BtcDecisionAction::BuyUp),
            ("down", BtcOutcome::Down, BtcDecisionAction::BuyDown),
        ] {
            let mut config = directional_model_config();
            config.max_entry_price = dec!(0.995);
            let mut snapshot = directional_model_snapshot(action);
            match expected_outcome {
                BtcOutcome::Up => {
                    snapshot.up_book = book(BtcOutcome::Up, "up", dec!(0.98), dec!(0.99));
                }
                BtcOutcome::Down => {
                    snapshot.down_book = book(BtcOutcome::Down, "down", dec!(0.98), dec!(0.99));
                }
            }

            let default_decision = DeterministicBtcStrategy::evaluate(&config, &snapshot);
            assert_eq!(default_decision.action, BtcDecisionAction::NoTrade);
            assert_eq!(
                default_decision.reject_reason,
                Some(BtcRejectReason::PredictionDirectEdgeNonPositive)
            );
            let default_prediction = default_decision.prediction.as_ref().unwrap();
            assert!(
                serde_json::to_value(default_prediction)
                    .unwrap()
                    .get("entry_policy")
                    .is_none(),
                "the default policy must preserve the historical prediction contract"
            );

            let validation_decision =
                DeterministicBtcStrategy::evaluate_with_directional_model_entry_policy(
                    &config,
                    &snapshot,
                    BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction,
                );
            assert_eq!(validation_decision.action, expected_action);
            assert_eq!(validation_decision.reject_reason, None);
            assert!(validation_decision
                .approved_intent
                .as_ref()
                .is_some_and(|intent| intent.outcome == expected_outcome
                    && intent.expected_net_edge_per_share < Decimal::ZERO
                    && intent.expected_net_edge < Decimal::ZERO));
            assert!(matches!(
                validation_decision.prediction,
                Some(BtcStrategyPrediction::DirectionalPrediction {
                    outcome,
                    direct_net_edge_per_share: Some(edge),
                    entry_policy:
                        BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction,
                    ..
                }) if outcome == expected_outcome && edge < Decimal::ZERO
            ));
            assert_eq!(
                serde_json::to_value(validation_decision.prediction.as_ref().unwrap()).unwrap()
                    ["entry_policy"],
                serde_json::json!("execute_directional_prediction")
            );
        }
    }

    #[test]
    fn directional_model_confidence_crossing_survives_execution_gate_rejection() {
        let config = directional_model_config();
        let mut snapshot = directional_model_snapshot("up");
        snapshot.up_book.best_bid = snapshot.up_book.best_ask;

        let decision = DeterministicBtcStrategy::evaluate(&config, &snapshot);

        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(decision.reject_reason, Some(BtcRejectReason::CrossedBook));
        assert!(matches!(
            decision.prediction,
            Some(BtcStrategyPrediction::DirectionalPrediction {
                outcome: BtcOutcome::Up,
                ..
            })
        ));
        assert!(decision.approved_intent.is_none());
    }

    #[test]
    fn directional_model_real_artifact_trades_without_live_chainlink_features() {
        for (action, expected_outcome, expected_action) in [
            ("up", BtcOutcome::Up, BtcDecisionAction::BuyUp),
            ("down", BtcOutcome::Down, BtcDecisionAction::BuyDown),
        ] {
            let config = directional_model_config();
            let mut snapshot = directional_model_snapshot(action);
            snapshot.chainlink_open_price = None;
            snapshot.chainlink_price = None;
            snapshot.chainlink_gap_bps = None;
            snapshot.chainlink_return_5s = None;
            snapshot.chainlink_return_15s = None;
            snapshot.chainlink_return_30s = None;
            snapshot.chainlink_path_efficiency_30s = None;
            snapshot.chainlink_path_tick_count_30s = None;
            snapshot.chainlink_realized_volatility_5s = None;
            snapshot.chainlink_realized_volatility_30s = None;
            snapshot.binance_return_1s = None;
            snapshot.binance_return_5s = None;
            snapshot.binance_return_30s = None;
            snapshot.realized_volatility = None;
            snapshot.binance_chainlink_basis_bps = None;
            snapshot.chainlink_age_ms = None;
            snapshot.source_skew_ms = None;
            snapshot.chainlink_quality_ok = false;
            snapshot.lineage.chainlink_open_tick_id = None;
            snapshot.lineage.chainlink_tick_id = None;
            snapshot.lineage.chainlink_open_source_timestamp = None;
            snapshot.lineage.chainlink_open_received_at = None;
            snapshot.lineage.chainlink_source_timestamp = None;
            snapshot.lineage.chainlink_received_at = None;
            snapshot.lineage.chainlink_open_ingest_sequence = None;
            snapshot.lineage.chainlink_ingest_sequence = None;

            let decision = DeterministicBtcStrategy::evaluate(&config, &snapshot);

            assert_eq!(decision.action, expected_action);
            assert_eq!(
                decision
                    .approved_intent
                    .as_ref()
                    .map(|intent| (intent.outcome, intent.process_id)),
                Some((expected_outcome, snapshot.process_id))
            );
            assert!(matches!(
                decision.prediction.as_ref(),
                Some(BtcStrategyPrediction::DirectionalPrediction { outcome, .. })
                    if *outcome == expected_outcome
            ));
            let intent = decision.approved_intent.as_ref().unwrap();
            let request = OrderRequest {
                client_order_id: Uuid::new_v4(),
                process_id: Some(snapshot.process_id),
                market_id: snapshot.market_id.clone(),
                token_id: intent.token_id.clone(),
                side: OrderSide::Buy,
                order_type: OrderType::Fok,
                price: intent.limit_price,
                size: intent.size,
                metadata: serde_json::json!({}),
            };
            let guard = BtcReferenceExecutionGuard::from_snapshot(
                &snapshot,
                &decision,
                intent,
                &request,
                &"a".repeat(64),
                snapshot.fee_rate.unwrap(),
                BtcExecutionFreshnessBounds {
                    max_reference_age_ms: config.max_reference_age_ms,
                    max_directional_feature_age_ms: config
                        .effective_max_directional_feature_age_ms()
                        .unwrap(),
                },
            )
            .unwrap();
            assert_eq!(
                guard.guard_version,
                BTC_DIRECTIONAL_MODEL_EXECUTION_GUARD_VERSION
            );
            assert!(guard.chainlink_open.is_none());
            assert!(guard.chainlink.is_none());
            assert!(guard.directional_model.is_some());
            assert!(guard.selected_book.is_some());

            let mut missing_model_snapshot = snapshot.clone();
            missing_model_snapshot.directional_model = None;
            assert!(BtcReferenceExecutionGuard::from_snapshot(
                &missing_model_snapshot,
                &decision,
                intent,
                &request,
                &"a".repeat(64),
                snapshot.fee_rate.unwrap(),
                BtcExecutionFreshnessBounds {
                    max_reference_age_ms: config.max_reference_age_ms,
                    max_directional_feature_age_ms: config
                        .effective_max_directional_feature_age_ms()
                        .unwrap(),
                },
            )
            .is_err());
        }
    }

    #[test]
    fn directional_model_rejects_a_feature_vector_with_a_mismatched_input_hash() {
        let config = directional_model_config();
        let mut snapshot = directional_model_snapshot("up");
        snapshot.directional_model.as_mut().unwrap().feature_values[0] += 1.0;

        let decision = DeterministicBtcStrategy::evaluate(&config, &snapshot);

        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(
            decision.reject_reason,
            Some(BtcRejectReason::InvalidFeatureValue)
        );
        assert!(decision.approved_intent.is_none());
    }

    #[test]
    fn directional_model_keeps_book_freshness_bound_aligned_with_reference_evidence() {
        let mut config = directional_model_config();
        config.validate().unwrap();
        config.max_book_age_ms += 1;

        assert!(config.validate().is_err());
    }

    #[test]
    fn directional_model_defaults_feature_freshness_to_its_candidate_cadence() {
        let config = directional_model_config();
        let mut within_cadence = directional_model_snapshot("up");
        within_cadence.observed_at += Duration::milliseconds(4_999);

        let accepted = DeterministicBtcStrategy::evaluate(&config, &within_cadence);
        assert_ne!(
            accepted.reject_reason,
            Some(BtcRejectReason::StaleDirectionalFeatures)
        );

        let mut beyond_cadence = directional_model_snapshot("up");
        beyond_cadence.observed_at += Duration::milliseconds(5_001);
        let rejected = DeterministicBtcStrategy::evaluate(&config, &beyond_cadence);
        assert_eq!(
            rejected.reject_reason,
            Some(BtcRejectReason::StaleDirectionalFeatures)
        );
    }

    #[test]
    fn directional_model_distinguishes_future_and_misaligned_features() {
        let config = directional_model_config();
        let mut future = directional_model_snapshot("up");
        future.observed_at -= Duration::milliseconds(1);
        assert_eq!(
            DeterministicBtcStrategy::evaluate(&config, &future).reject_reason,
            Some(BtcRejectReason::FutureDirectionalFeatures)
        );

        let mut misaligned = directional_model_snapshot("up");
        misaligned.directional_model.as_mut().unwrap().feature_as_of += Duration::milliseconds(1);
        assert_eq!(
            DeterministicBtcStrategy::evaluate(&config, &misaligned).reject_reason,
            Some(BtcRejectReason::InvalidDirectionalFeatureTimestamp)
        );
    }

    #[test]
    fn directional_feature_freshness_override_cannot_exceed_model_cadence() {
        let mut config = directional_model_config();
        config.max_directional_feature_age_ms = Some(5_001);
        assert!(config.validate().is_err());

        config.max_directional_feature_age_ms = Some(3_000);
        config.validate().unwrap();
        let mut snapshot = directional_model_snapshot("up");
        snapshot.observed_at += Duration::milliseconds(3_001);
        assert_eq!(
            DeterministicBtcStrategy::evaluate(&config, &snapshot).reject_reason,
            Some(BtcRejectReason::StaleDirectionalFeatures)
        );
    }

    #[test]
    fn dynamic_crypto_fee_matches_golden_formula() {
        assert_eq!(
            dynamic_crypto_taker_fee(dec!(10), dec!(0.25), dec!(0.40)),
            dec!(0.6000)
        );
    }

    #[test]
    fn same_snapshot_and_config_produce_identical_decision() {
        let config = BtcStrategyConfig::default();
        let snapshot = snapshot();
        assert_eq!(
            DeterministicBtcStrategy::evaluate(&config, &snapshot),
            DeterministicBtcStrategy::evaluate(&config, &snapshot)
        );
    }
}
