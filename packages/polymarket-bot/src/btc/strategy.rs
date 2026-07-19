use chrono::{DateTime, Utc};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::types::{BtcOutcome, FeedIntegrityStatus};

mod market_anchored;

pub const BTC_FEATURE_SCHEMA_VERSION: &str = "btc_5m_features_v2";
pub const BTC_STRATEGY_VERSION: &str = "btc_5m_chainlink_fair_value_v1";
pub const BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION: &str = "btc_5m_volatility_continuation_v1";
pub const BTC_MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION: &str =
    market_anchored::MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION;
pub const BTC_MARKET_ANCHORED_RESEARCH_PROFILE_ID: &str = market_anchored::PROFILE_ID;
pub const BTC_MARKET_ANCHORED_RESEARCH_PROFILE_SHA256: &str = market_anchored::PROFILE_SHA256;
pub const BTC_FEATURE_LINEAGE_VERSION: &str = "btc_5m_feature_lineage_v2";
pub const BTC_CHAINLINK_FAIR_VALUE_STRATEGY_FAMILY: &str = "btc_5m_chainlink_fair_value";
pub const BTC_VOLATILITY_CONTINUATION_STRATEGY_FAMILY: &str = "btc_5m_volatility_continuation";
pub const BTC_MARKET_ANCHORED_FAIR_VALUE_STRATEGY_FAMILY: &str =
    "btc_5m_market_anchored_fair_value";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BtcVolatilityContinuationConfig {
    pub min_seconds_to_close: i64,
    pub max_seconds_to_close: i64,
    pub min_realized_volatility_per_sqrt_second: Decimal,
    pub min_abs_chainlink_gap_bps: Decimal,
    pub min_executable_price: Decimal,
    pub max_executable_price: Decimal,
    pub max_logit_adjustment: Decimal,
}

impl Default for BtcVolatilityContinuationConfig {
    fn default() -> Self {
        Self {
            min_seconds_to_close: 45,
            max_seconds_to_close: 120,
            min_realized_volatility_per_sqrt_second: dec!(0.00002),
            min_abs_chainlink_gap_bps: dec!(3),
            min_executable_price: dec!(0.55),
            max_executable_price: dec!(0.80),
            max_logit_adjustment: dec!(0.35),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum BtcDecisionStrategyConfig {
    ChainlinkFairValue {},
    VolatilityContinuation {
        config: BtcVolatilityContinuationConfig,
    },
    MarketAnchoredFairValue {
        profile_id: String,
        profile_sha256: String,
    },
}

/// Immutable, process-owned parameters for the deterministic baseline.
///
/// Runtime configuration should be hashed and frozen when an experiment starts. The strategy
/// deliberately excludes Polymarket order-book imbalance from its probability estimator. Book
/// features are used only to determine executability and net edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BtcStrategyConfig {
    pub strategy_version: String,
    pub feature_schema_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_strategy: Option<BtcDecisionStrategyConfig>,
    pub target_size: Decimal,
    pub min_seconds_after_open: i64,
    pub min_seconds_before_close: i64,
    pub max_reference_age_ms: i64,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volatility_continuation: Option<BtcVolatilityContinuationConfig>,
}

impl Default for BtcStrategyConfig {
    fn default() -> Self {
        Self {
            strategy_version: BTC_STRATEGY_VERSION.to_string(),
            feature_schema_version: BTC_FEATURE_SCHEMA_VERSION.to_string(),
            decision_strategy: None,
            target_size: dec!(5),
            min_seconds_after_open: 15,
            min_seconds_before_close: 20,
            max_reference_age_ms: 2_000,
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
            volatility_continuation: None,
        }
    }
}

impl BtcStrategyConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        validate_config(self)
            .map_err(|_| anyhow::anyhow!("invalid BTC deterministic strategy configuration"))
    }

    pub fn attribution(&self) -> Option<BtcStrategyAttribution<'_>> {
        let family = match ResolvedBtcDecisionStrategy::resolve(self).ok()? {
            ResolvedBtcDecisionStrategy::ChainlinkFairValue => {
                BTC_CHAINLINK_FAIR_VALUE_STRATEGY_FAMILY
            }
            ResolvedBtcDecisionStrategy::VolatilityContinuation(_) => {
                BTC_VOLATILITY_CONTINUATION_STRATEGY_FAMILY
            }
            ResolvedBtcDecisionStrategy::MarketAnchoredFairValue(_) => {
                BTC_MARKET_ANCHORED_FAIR_VALUE_STRATEGY_FAMILY
            }
        };
        let (profile_id, profile_sha256) = match self.decision_strategy.as_ref() {
            Some(BtcDecisionStrategyConfig::MarketAnchoredFairValue {
                profile_id,
                profile_sha256,
            }) => (Some(profile_id.as_str()), Some(profile_sha256.as_str())),
            _ => (None, None),
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
    MissingBinancePrice,
    MissingBinanceReturns,
    MissingRealizedVolatility,
    MissingBasis,
    ChainlinkFeedUnhealthy,
    BinanceFeedUnhealthy,
    MissingLineage,
    FutureInputTimestamp,
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
    VolatilityRegimeBelowThreshold,
    ContinuationSignalUnconfirmed,
    MarketPriorOutsideBounds,
    ExistingExperimentEntry,
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
            Self::MissingBinancePrice => "missing_binance_price",
            Self::MissingBinanceReturns => "missing_binance_returns",
            Self::MissingRealizedVolatility => "missing_realized_volatility",
            Self::MissingBasis => "missing_basis",
            Self::ChainlinkFeedUnhealthy => "chainlink_feed_unhealthy",
            Self::BinanceFeedUnhealthy => "binance_feed_unhealthy",
            Self::MissingLineage => "missing_lineage",
            Self::FutureInputTimestamp => "future_input_timestamp",
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
            Self::VolatilityRegimeBelowThreshold => "volatility_regime_below_threshold",
            Self::ContinuationSignalUnconfirmed => "continuation_signal_unconfirmed",
            Self::MarketPriorOutsideBounds => "market_prior_outside_bounds",
            Self::ExistingExperimentEntry => "existing_experiment_entry",
            Self::RuntimeNotReady => "runtime_not_ready",
        }
    }
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BtcOutcomeScope {
    Both,
    Only(BtcOutcome),
}

#[derive(Debug, Clone, PartialEq)]
struct BtcStrategyEstimate {
    fair_value: FairValueEstimate,
    outcome_scope: BtcOutcomeScope,
    executable_price_bounds: Option<(Decimal, Decimal)>,
}

#[derive(Debug, Clone, Copy)]
enum ResolvedBtcDecisionStrategy<'a> {
    ChainlinkFairValue,
    VolatilityContinuation(&'a BtcVolatilityContinuationConfig),
    MarketAnchoredFairValue(&'static market_anchored::MarketAnchoredProfile),
}

impl<'a> ResolvedBtcDecisionStrategy<'a> {
    fn resolve(config: &'a BtcStrategyConfig) -> Result<Self, BtcRejectReason> {
        if let Some(selection) = config.decision_strategy.as_ref() {
            return match selection {
                BtcDecisionStrategyConfig::ChainlinkFairValue {}
                    if config.strategy_version == BTC_STRATEGY_VERSION
                        && config.volatility_continuation.is_none() =>
                {
                    Ok(Self::ChainlinkFairValue)
                }
                BtcDecisionStrategyConfig::VolatilityContinuation {
                    config: continuation,
                } if config.strategy_version == BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION
                    && config.volatility_continuation.is_none() =>
                {
                    Ok(Self::VolatilityContinuation(continuation))
                }
                BtcDecisionStrategyConfig::MarketAnchoredFairValue {
                    profile_id,
                    profile_sha256,
                } if config.strategy_version == BTC_MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION
                    && config.volatility_continuation.is_none() =>
                {
                    market_anchored::resolve_profile(
                        &market_anchored::MarketAnchoredProfileSelection::new(
                            profile_id,
                            profile_sha256,
                        ),
                    )
                    .map(Self::MarketAnchoredFairValue)
                }
                _ => Err(BtcRejectReason::InvalidConfiguration),
            };
        }
        match config.strategy_version.as_str() {
            BTC_STRATEGY_VERSION if config.volatility_continuation.is_none() => {
                Ok(Self::ChainlinkFairValue)
            }
            BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION => config
                .volatility_continuation
                .as_ref()
                .map(Self::VolatilityContinuation)
                .ok_or(BtcRejectReason::InvalidConfiguration),
            _ => Err(BtcRejectReason::InvalidConfiguration),
        }
    }

    fn estimate(
        self,
        config: &BtcStrategyConfig,
        snapshot: &BtcFeatureSnapshot,
    ) -> Result<BtcStrategyEstimate, BtcRejectReason> {
        match self {
            Self::ChainlinkFairValue => Ok(BtcStrategyEstimate {
                fair_value: estimate_chainlink_fair_value(config, snapshot)?,
                outcome_scope: BtcOutcomeScope::Both,
                executable_price_bounds: None,
            }),
            Self::VolatilityContinuation(continuation) => Ok(BtcStrategyEstimate {
                fair_value: estimate_market_anchored_continuation(config, continuation, snapshot)?,
                outcome_scope: BtcOutcomeScope::Only(continuation_signal_outcome(
                    continuation,
                    snapshot,
                )?),
                executable_price_bounds: Some((
                    continuation.min_executable_price,
                    continuation.max_executable_price,
                )),
            }),
            Self::MarketAnchoredFairValue(profile) => Ok(BtcStrategyEstimate {
                fair_value: market_anchored::estimate(config, snapshot, profile)?.fair_value,
                outcome_scope: BtcOutcomeScope::Both,
                executable_price_bounds: None,
            }),
        }
    }
}

pub struct DeterministicBtcStrategy;

impl DeterministicBtcStrategy {
    pub fn evaluate(config: &BtcStrategyConfig, snapshot: &BtcFeatureSnapshot) -> BtcDecision {
        let decision_id = deterministic_decision_id(config, snapshot);
        if let Err(reason) = validate_config(config) {
            return rejected(decision_id, snapshot, reason, None, None, None);
        }
        if let Err(reason) = validate_snapshot(config, snapshot) {
            return rejected(decision_id, snapshot, reason, None, None, None);
        }

        let strategy = match ResolvedBtcDecisionStrategy::resolve(config) {
            Ok(strategy) => strategy,
            Err(reason) => return rejected(decision_id, snapshot, reason, None, None, None),
        };
        let estimate = match strategy.estimate(config, snapshot) {
            Ok(value) => value,
            Err(reason) => return rejected(decision_id, snapshot, reason, None, None, None),
        };
        build_decision_from_estimate(config, snapshot, decision_id, estimate)
    }
}

fn build_decision_from_estimate(
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
    decision_id: Uuid,
    estimate: BtcStrategyEstimate,
) -> BtcDecision {
    let BtcStrategyEstimate {
        fair_value,
        outcome_scope,
        executable_price_bounds,
    } = estimate;
    let fee_rate = if snapshot.fees_enabled {
        snapshot.fee_rate.unwrap_or(Decimal::ZERO)
    } else {
        Decimal::ZERO
    };

    if let BtcOutcomeScope::Only(outcome) = outcome_scope {
        let book = match outcome {
            BtcOutcome::Up => &snapshot.up_book,
            BtcOutcome::Down => &snapshot.down_book,
        };
        let executable_price = book.executable_ask_vwap.unwrap_or_default();
        if executable_price_bounds.is_some_and(|(minimum, maximum)| {
            executable_price < minimum || executable_price > maximum
        }) {
            return rejected(
                decision_id,
                snapshot,
                BtcRejectReason::MarketPriorOutsideBounds,
                Some(fair_value),
                None,
                None,
            );
        }
        let conservative_probability = match outcome {
            BtcOutcome::Up => fair_value.up_lower_bound,
            BtcOutcome::Down => fair_value.down_lower_bound,
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
                return rejected(decision_id, snapshot, reason, Some(fair_value), None, None)
            }
        };
        return match outcome {
            BtcOutcome::Up => finish_selected_edge(
                config,
                snapshot,
                decision_id,
                fair_value,
                Some(selected.clone()),
                None,
                selected,
            ),
            BtcOutcome::Down => finish_selected_edge(
                config,
                snapshot,
                decision_id,
                fair_value,
                None,
                Some(selected.clone()),
                selected,
            ),
        };
    }

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

/// Polymarket's dynamic crypto taker fee: `contracts * rate * price * (1 - price)`.
pub fn dynamic_crypto_taker_fee(contracts: Decimal, fee_rate: Decimal, price: Decimal) -> Decimal {
    if contracts <= Decimal::ZERO
        || fee_rate <= Decimal::ZERO
        || price <= Decimal::ZERO
        || price >= Decimal::ONE
    {
        return Decimal::ZERO;
    }
    contracts * fee_rate * price * (Decimal::ONE - price)
}

pub fn estimate_fair_value(
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
) -> Result<FairValueEstimate, BtcRejectReason> {
    match ResolvedBtcDecisionStrategy::resolve(config)? {
        ResolvedBtcDecisionStrategy::ChainlinkFairValue => {
            estimate_chainlink_fair_value(config, snapshot)
        }
        ResolvedBtcDecisionStrategy::VolatilityContinuation(continuation) => {
            estimate_market_anchored_continuation(config, continuation, snapshot)
        }
        ResolvedBtcDecisionStrategy::MarketAnchoredFairValue(profile) => {
            Ok(market_anchored::estimate(config, snapshot, profile)?.fair_value)
        }
    }
}

fn estimate_chainlink_fair_value(
    config: &BtcStrategyConfig,
    snapshot: &BtcFeatureSnapshot,
) -> Result<FairValueEstimate, BtcRejectReason> {
    let open = positive(
        snapshot.chainlink_open_price,
        BtcRejectReason::MissingChainlinkOpen,
    )?;
    let chainlink = positive(
        snapshot.chainlink_price,
        BtcRejectReason::MissingChainlinkPrice,
    )?;
    positive(snapshot.binance_price, BtcRejectReason::MissingBinancePrice)?;
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
    .max(config.volatility_floor_per_sqrt_second);
    let basis_bps = snapshot
        .binance_chainlink_basis_bps
        .ok_or(BtcRejectReason::MissingBasis)?;

    let seconds_to_close = (snapshot.window_end - snapshot.observed_at).num_milliseconds();
    if seconds_to_close <= 0 {
        return Err(BtcRejectReason::OutsideEntryWindow);
    }
    let tau = seconds_to_close as f64 / 1000.0;
    let open_f64 = open.to_f64().ok_or(BtcRejectReason::InvalidFeatureValue)?;
    let chainlink_f64 = chainlink
        .to_f64()
        .ok_or(BtcRejectReason::InvalidFeatureValue)?;
    let chainlink_log_gap_f64 = (chainlink_f64 / open_f64).ln();
    if !chainlink_log_gap_f64.is_finite() {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    let chainlink_log_gap = decimal_from_f64(chainlink_log_gap_f64)?;
    let terminal_volatility = decimal_from_f64(
        volatility
            .to_f64()
            .ok_or(BtcRejectReason::InvalidFeatureValue)?
            * tau.sqrt(),
    )?;
    if terminal_volatility <= Decimal::ZERO {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }

    let raw_lead = config.basis_lead_weight * basis_bps / dec!(10000)
        + config.momentum_1s_weight * return_1s
        + config.momentum_5s_weight * return_5s
        + config.momentum_30s_weight * return_30s;
    let max_lead = terminal_volatility * config.max_lead_sigma_fraction;
    let lead_adjustment = raw_lead.clamp(-max_lead, max_lead);
    let z_score = (chainlink_log_gap + lead_adjustment) / terminal_volatility;
    let p_up = decimal_from_f64(normal_cdf(
        z_score
            .to_f64()
            .ok_or(BtcRejectReason::InvalidFeatureValue)?,
    ))?
    .clamp(
        config.probability_floor,
        Decimal::ONE - config.probability_floor,
    );

    let max_age_ms = snapshot
        .chainlink_age_ms
        .unwrap_or_default()
        .max(snapshot.binance_age_ms.unwrap_or_default())
        .max(0);
    let age_seconds = Decimal::from(max_age_ms) / dec!(1000);
    let uncertainty = (config.base_probability_uncertainty
        + config.basis_uncertainty_weight * (basis_bps / dec!(10000)).abs()
        + config.feed_age_uncertainty_per_second * age_seconds)
        .clamp(Decimal::ZERO, config.max_probability_uncertainty);
    let up_lower = (p_up - uncertainty).max(config.probability_floor);
    let up_upper = (p_up + uncertainty).min(Decimal::ONE - config.probability_floor);

    Ok(FairValueEstimate {
        up_probability: p_up,
        down_probability: Decimal::ONE - p_up,
        up_lower_bound: up_lower,
        up_upper_bound: up_upper,
        down_lower_bound: Decimal::ONE - up_upper,
        down_upper_bound: Decimal::ONE - up_lower,
        z_score,
        chainlink_log_gap,
        lead_adjustment,
        terminal_volatility,
        probability_uncertainty: uncertainty,
        market_up_prior: None,
        signed_logit_adjustment: None,
    })
}

fn estimate_market_anchored_continuation(
    config: &BtcStrategyConfig,
    continuation: &BtcVolatilityContinuationConfig,
    snapshot: &BtcFeatureSnapshot,
) -> Result<FairValueEstimate, BtcRejectReason> {
    let open = positive(
        snapshot.chainlink_open_price,
        BtcRejectReason::MissingChainlinkOpen,
    )?;
    let chainlink = positive(
        snapshot.chainlink_price,
        BtcRejectReason::MissingChainlinkPrice,
    )?;
    let volatility = positive(
        snapshot.realized_volatility,
        BtcRejectReason::MissingRealizedVolatility,
    )?
    .max(config.volatility_floor_per_sqrt_second);
    let basis_bps = snapshot
        .binance_chainlink_basis_bps
        .ok_or(BtcRejectReason::MissingBasis)?;
    let return_1s = snapshot
        .binance_return_1s
        .ok_or(BtcRejectReason::MissingBinanceReturns)?;
    let return_5s = snapshot
        .binance_return_5s
        .ok_or(BtcRejectReason::MissingBinanceReturns)?;
    let return_30s = snapshot
        .binance_return_30s
        .ok_or(BtcRejectReason::MissingBinanceReturns)?;
    let outcome = continuation_signal_outcome(continuation, snapshot)?;

    let seconds_to_close = (snapshot.window_end - snapshot.observed_at).num_milliseconds();
    if seconds_to_close <= 0 {
        return Err(BtcRejectReason::OutsideEntryWindow);
    }
    let tau = seconds_to_close as f64 / 1_000.0;
    let open_f64 = open.to_f64().ok_or(BtcRejectReason::InvalidFeatureValue)?;
    let chainlink_f64 = chainlink
        .to_f64()
        .ok_or(BtcRejectReason::InvalidFeatureValue)?;
    let chainlink_log_gap_f64 = (chainlink_f64 / open_f64).ln();
    if !chainlink_log_gap_f64.is_finite() {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    let chainlink_log_gap = decimal_from_f64(chainlink_log_gap_f64)?;
    let terminal_volatility = decimal_from_f64(
        volatility
            .to_f64()
            .ok_or(BtcRejectReason::InvalidFeatureValue)?
            * tau.sqrt(),
    )?;
    if terminal_volatility <= Decimal::ZERO {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }

    let up_bid = snapshot
        .up_book
        .best_bid
        .ok_or(BtcRejectReason::MissingUpBook)?;
    let up_ask = snapshot
        .up_book
        .best_ask
        .ok_or(BtcRejectReason::MissingUpBook)?;
    let down_bid = snapshot
        .down_book
        .best_bid
        .ok_or(BtcRejectReason::MissingDownBook)?;
    let down_ask = snapshot
        .down_book
        .best_ask
        .ok_or(BtcRejectReason::MissingDownBook)?;
    let up_mid = (up_bid + up_ask) / dec!(2);
    let down_mid = (down_bid + down_ask) / dec!(2);
    let midpoint_sum = up_mid + down_mid;
    if midpoint_sum <= Decimal::ZERO {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    let market_up_probability = (up_mid / midpoint_sum).clamp(
        config.probability_floor,
        Decimal::ONE - config.probability_floor,
    );

    let raw_lead = config.basis_lead_weight * basis_bps / dec!(10000)
        + config.momentum_1s_weight * return_1s
        + config.momentum_5s_weight * return_5s
        + config.momentum_30s_weight * return_30s;
    let max_lead = terminal_volatility * config.max_lead_sigma_fraction;
    let lead_adjustment = raw_lead.clamp(-max_lead, max_lead);
    let external_z = (chainlink_log_gap + lead_adjustment) / terminal_volatility;
    let strength = external_z.abs().min(Decimal::ONE);
    let signed_logit_adjustment = match outcome {
        BtcOutcome::Up => continuation.max_logit_adjustment * strength,
        BtcOutcome::Down => -continuation.max_logit_adjustment * strength,
    };
    let market_up_f64 = market_up_probability
        .to_f64()
        .ok_or(BtcRejectReason::InvalidFeatureValue)?;
    let logit = (market_up_f64 / (1.0 - market_up_f64)).ln();
    let adjusted_logit = logit
        + signed_logit_adjustment
            .to_f64()
            .ok_or(BtcRejectReason::InvalidFeatureValue)?;
    let p_up = decimal_from_f64(1.0 / (1.0 + (-adjusted_logit).exp()))?.clamp(
        config.probability_floor,
        Decimal::ONE - config.probability_floor,
    );

    let max_age_ms = snapshot
        .chainlink_age_ms
        .unwrap_or_default()
        .max(snapshot.binance_age_ms.unwrap_or_default())
        .max(0);
    let age_seconds = Decimal::from(max_age_ms) / dec!(1000);
    let uncertainty = (config.base_probability_uncertainty
        + config.basis_uncertainty_weight * (basis_bps / dec!(10000)).abs()
        + config.feed_age_uncertainty_per_second * age_seconds)
        .clamp(Decimal::ZERO, config.max_probability_uncertainty);
    let up_lower = (p_up - uncertainty).max(config.probability_floor);
    let up_upper = (p_up + uncertainty).min(Decimal::ONE - config.probability_floor);

    Ok(FairValueEstimate {
        up_probability: p_up,
        down_probability: Decimal::ONE - p_up,
        up_lower_bound: up_lower,
        up_upper_bound: up_upper,
        down_lower_bound: Decimal::ONE - up_upper,
        down_upper_bound: Decimal::ONE - up_lower,
        z_score: external_z,
        chainlink_log_gap,
        lead_adjustment,
        terminal_volatility,
        probability_uncertainty: uncertainty,
        market_up_prior: None,
        signed_logit_adjustment: None,
    })
}

fn continuation_signal_outcome(
    continuation: &BtcVolatilityContinuationConfig,
    snapshot: &BtcFeatureSnapshot,
) -> Result<BtcOutcome, BtcRejectReason> {
    let gap_bps = snapshot
        .chainlink_gap_bps
        .ok_or(BtcRejectReason::MissingChainlinkGap)?;
    let return_5s = snapshot
        .binance_return_5s
        .ok_or(BtcRejectReason::MissingBinanceReturns)?;
    let return_30s = snapshot
        .binance_return_30s
        .ok_or(BtcRejectReason::MissingBinanceReturns)?;
    if gap_bps.abs() < continuation.min_abs_chainlink_gap_bps {
        return Err(BtcRejectReason::ContinuationSignalUnconfirmed);
    }
    if gap_bps > Decimal::ZERO && return_5s > Decimal::ZERO && return_30s > Decimal::ZERO {
        Ok(BtcOutcome::Up)
    } else if gap_bps < Decimal::ZERO && return_5s < Decimal::ZERO && return_30s < Decimal::ZERO {
        Ok(BtcOutcome::Down)
    } else {
        Err(BtcRejectReason::ContinuationSignalUnconfirmed)
    }
}

fn validate_config(config: &BtcStrategyConfig) -> Result<(), BtcRejectReason> {
    let strategy_contract_valid = match ResolvedBtcDecisionStrategy::resolve(config) {
        Ok(ResolvedBtcDecisionStrategy::ChainlinkFairValue) => true,
        Ok(ResolvedBtcDecisionStrategy::VolatilityContinuation(continuation)) => {
            continuation.min_seconds_to_close > 0
                && continuation.max_seconds_to_close >= continuation.min_seconds_to_close
                && continuation.min_seconds_to_close >= config.min_seconds_before_close
                && continuation.max_seconds_to_close <= 300 - config.min_seconds_after_open
                && continuation.min_realized_volatility_per_sqrt_second > Decimal::ZERO
                && continuation.min_abs_chainlink_gap_bps > Decimal::ZERO
                && continuation.min_executable_price > Decimal::ZERO
                && continuation.max_executable_price < Decimal::ONE
                && continuation.min_executable_price < continuation.max_executable_price
                && continuation.min_executable_price >= config.min_entry_price
                && continuation.max_executable_price <= config.max_entry_price
                && continuation.max_logit_adjustment > Decimal::ZERO
                && continuation.max_logit_adjustment <= Decimal::ONE
        }
        Ok(ResolvedBtcDecisionStrategy::MarketAnchoredFairValue(profile)) => {
            market_anchored::validate_strategy_config(config, profile).is_ok()
        }
        Err(_) => false,
    };
    let valid = !config.strategy_version.trim().is_empty()
        && !config.feature_schema_version.trim().is_empty()
        && config.target_size > Decimal::ZERO
        && config.min_seconds_after_open >= 0
        && config.min_seconds_before_close > 0
        && config.max_reference_age_ms > 0
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
    let earliest_entry =
        snapshot.window_start + chrono::Duration::seconds(config.min_seconds_after_open);
    let latest_entry =
        snapshot.window_end - chrono::Duration::seconds(config.min_seconds_before_close);
    if snapshot.observed_at < earliest_entry || snapshot.observed_at >= latest_entry {
        return Err(BtcRejectReason::OutsideEntryWindow);
    }
    if let Ok(ResolvedBtcDecisionStrategy::VolatilityContinuation(continuation)) =
        ResolvedBtcDecisionStrategy::resolve(config)
    {
        let seconds_to_close = (snapshot.window_end - snapshot.observed_at).num_milliseconds();
        let min_ms = continuation.min_seconds_to_close * 1_000;
        let max_ms = continuation.max_seconds_to_close * 1_000;
        if seconds_to_close < min_ms || seconds_to_close > max_ms {
            return Err(BtcRejectReason::OutsideEntryWindow);
        }
        let realized_volatility = positive(
            snapshot.realized_volatility,
            BtcRejectReason::MissingRealizedVolatility,
        )?;
        if realized_volatility < continuation.min_realized_volatility_per_sqrt_second {
            return Err(BtcRejectReason::VolatilityRegimeBelowThreshold);
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
    positive(
        snapshot.chainlink_open_price,
        BtcRejectReason::MissingChainlinkOpen,
    )?;
    positive(
        snapshot.chainlink_price,
        BtcRejectReason::MissingChainlinkPrice,
    )?;
    if snapshot.chainlink_gap_bps.is_none() {
        return Err(BtcRejectReason::MissingChainlinkGap);
    }
    positive(snapshot.binance_price, BtcRejectReason::MissingBinancePrice)?;
    if snapshot.binance_return_1s.is_none()
        || snapshot.binance_return_5s.is_none()
        || snapshot.binance_return_30s.is_none()
    {
        return Err(BtcRejectReason::MissingBinanceReturns);
    }
    positive(
        snapshot.realized_volatility,
        BtcRejectReason::MissingRealizedVolatility,
    )?;
    if snapshot.binance_chainlink_basis_bps.is_none() {
        return Err(BtcRejectReason::MissingBasis);
    }
    if !snapshot.chainlink_quality_ok {
        return Err(BtcRejectReason::ChainlinkFeedUnhealthy);
    }
    if !snapshot.binance_quality_ok {
        return Err(BtcRejectReason::BinanceFeedUnhealthy);
    }
    validate_reference_lineage(snapshot)?;
    validate_age(
        snapshot.chainlink_age_ms,
        config.max_reference_age_ms,
        BtcRejectReason::StaleChainlinkFeed,
    )?;
    validate_age(
        snapshot.binance_age_ms,
        config.max_reference_age_ms,
        BtcRejectReason::StaleBinanceFeed,
    )?;
    let skew = snapshot
        .source_skew_ms
        .ok_or(BtcRejectReason::MissingLineage)?;
    if skew < 0 {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    if skew > config.max_source_skew_ms {
        return Err(BtcRejectReason::SourceTimestampSkew);
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
    validate_fee(config, snapshot)?;
    Ok(())
}

fn validate_reference_lineage(snapshot: &BtcFeatureSnapshot) -> Result<(), BtcRejectReason> {
    if snapshot.lineage.chainlink_open_tick_id.is_none()
        || snapshot.lineage.chainlink_tick_id.is_none()
        || snapshot.lineage.binance_tick_id.is_none()
        || snapshot.lineage.up_book_checkpoint_id.is_none()
        || snapshot.lineage.down_book_checkpoint_id.is_none()
        || snapshot.lineage.chainlink_ingest_sequence.is_none()
        || snapshot.lineage.binance_ingest_sequence.is_none()
        || snapshot.lineage.up_book_ingest_sequence.is_none()
        || snapshot.lineage.down_book_ingest_sequence.is_none()
    {
        return Err(BtcRejectReason::MissingLineage);
    }
    let required = [
        snapshot.lineage.chainlink_open_source_timestamp,
        snapshot.lineage.chainlink_source_timestamp,
        snapshot.lineage.chainlink_received_at,
        snapshot.lineage.binance_source_timestamp,
        snapshot.lineage.binance_received_at,
    ];
    if required.iter().any(Option::is_none) {
        return Err(BtcRejectReason::MissingLineage);
    }
    if required
        .into_iter()
        .flatten()
        .any(|at| at > snapshot.observed_at)
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
    }
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

fn positive(
    value: Option<Decimal>,
    missing_reason: BtcRejectReason,
) -> Result<Decimal, BtcRejectReason> {
    let value = value.ok_or(missing_reason)?;
    if value <= Decimal::ZERO {
        return Err(BtcRejectReason::InvalidFeatureValue);
    }
    Ok(value)
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

fn normal_cdf(value: f64) -> f64 {
    // Abramowitz-Stegun 26.2.17. Absolute error is below 7.5e-8, more than adequate for
    // tick-sized prediction-market decisions.
    let x = value.abs();
    let k = 1.0 / (1.0 + 0.231_641_9 * x);
    let polynomial = k
        * (0.319_381_530
            + k * (-0.356_563_782
                + k * (1.781_477_937 + k * (-1.821_255_978 + k * 1.330_274_429))));
    let tail = 0.398_942_280_401_432_7 * (-0.5 * x * x).exp() * polynomial;
    if value >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
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
    use chrono::{Duration, TimeZone};
    use sha2::{Digest, Sha256};

    use super::*;

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

    fn continuation_config() -> BtcStrategyConfig {
        BtcStrategyConfig {
            strategy_version: BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION.to_string(),
            volatility_continuation: Some(BtcVolatilityContinuationConfig::default()),
            ..BtcStrategyConfig::default()
        }
    }

    fn market_anchored_config() -> BtcStrategyConfig {
        BtcStrategyConfig {
            strategy_version: BTC_MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION.to_string(),
            decision_strategy: Some(BtcDecisionStrategyConfig::MarketAnchoredFairValue {
                profile_id: BTC_MARKET_ANCHORED_RESEARCH_PROFILE_ID.to_string(),
                profile_sha256: BTC_MARKET_ANCHORED_RESEARCH_PROFILE_SHA256.to_string(),
            }),
            ..BtcStrategyConfig::default()
        }
    }

    fn continuation_snapshot() -> BtcFeatureSnapshot {
        let mut snapshot = snapshot();
        snapshot.up_book = book(BtcOutcome::Up, "up", dec!(0.55), dec!(0.56));
        snapshot.down_book = book(BtcOutcome::Down, "down", dec!(0.44), dec!(0.45));
        snapshot
    }

    fn decision_sha256(decision: &BtcDecision) -> String {
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(decision).expect("decision must serialize"))
        )
    }

    #[test]
    fn chainlink_fair_value_decision_matches_golden_contract() {
        let decision =
            DeterministicBtcStrategy::evaluate(&BtcStrategyConfig::default(), &snapshot());

        assert_eq!(
            decision_sha256(&decision),
            "60bd4559e35f10b8465cbe142085506ee860796411672be5baaeec98827f1c23"
        );
    }

    #[test]
    fn volatility_continuation_decision_matches_golden_contract() {
        let decision =
            DeterministicBtcStrategy::evaluate(&continuation_config(), &continuation_snapshot());

        assert_eq!(
            decision_sha256(&decision),
            "8c41f5bbc71da61e54a1b3f5a9984ec5ce97c4387f16c01b1c9159c1c3fd5fd4"
        );
    }

    #[test]
    fn fair_probability_is_monotonic_in_chainlink_distance_to_open() {
        let config = BtcStrategyConfig::default();
        let mut lower = snapshot();
        lower.chainlink_price = Some(dec!(99950));
        lower.chainlink_gap_bps = Some(dec!(-5));
        let mut higher = snapshot();
        higher.chainlink_price = Some(dec!(100150));
        higher.chainlink_gap_bps = Some(dec!(15));

        let lower = estimate_fair_value(&config, &lower).unwrap();
        let higher = estimate_fair_value(&config, &higher).unwrap();
        assert!(higher.up_probability > lower.up_probability);
    }

    #[test]
    fn approved_intents_are_idempotent_per_snapshot_but_retryable_on_fresh_snapshots() {
        let config = BtcStrategyConfig::default();
        let first_snapshot = snapshot();
        let first = DeterministicBtcStrategy::evaluate(&config, &first_snapshot)
            .approved_intent
            .expect("baseline snapshot should produce an approved intent");
        let replay = DeterministicBtcStrategy::evaluate(&config, &first_snapshot)
            .approved_intent
            .expect("same snapshot should remain approved");
        assert_eq!(first.intent_id, replay.intent_id);

        let mut fresh_snapshot = first_snapshot;
        fresh_snapshot.snapshot_id = Uuid::new_v4();
        fresh_snapshot.observed_at += Duration::seconds(1);
        let retry = DeterministicBtcStrategy::evaluate(&config, &fresh_snapshot)
            .approved_intent
            .expect("fresh snapshot should remain eligible");
        assert_ne!(first.intent_id, retry.intent_id);
    }

    #[test]
    fn probabilities_and_conservative_bounds_are_complementary() {
        let estimate = estimate_fair_value(&BtcStrategyConfig::default(), &snapshot()).unwrap();
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
        assert!(estimate.up_upper_bound >= estimate.up_probability);
    }

    #[test]
    fn orderbook_imbalance_does_not_influence_fair_probability() {
        let config = BtcStrategyConfig::default();
        let baseline = snapshot();
        let mut changed = baseline.clone();
        changed.up_book.imbalance = Some(dec!(-0.99));
        changed.down_book.imbalance = Some(dec!(0.99));
        assert_eq!(
            estimate_fair_value(&config, &baseline).unwrap(),
            estimate_fair_value(&config, &changed).unwrap()
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
    fn stale_reference_feed_is_no_trade() {
        let mut snapshot = snapshot();
        snapshot.chainlink_age_ms = Some(2_001);
        let decision = DeterministicBtcStrategy::evaluate(&BtcStrategyConfig::default(), &snapshot);
        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(
            decision.reject_reason,
            Some(BtcRejectReason::StaleChainlinkFeed)
        );
    }

    #[test]
    fn missing_required_feature_is_no_trade() {
        let mut snapshot = snapshot();
        snapshot.chainlink_price = None;
        let decision = DeterministicBtcStrategy::evaluate(&BtcStrategyConfig::default(), &snapshot);
        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(
            decision.reject_reason,
            Some(BtcRejectReason::MissingChainlinkPrice)
        );
    }

    #[test]
    fn outside_entry_window_is_no_trade() {
        let mut snapshot = snapshot();
        snapshot.observed_at = snapshot.window_start + Duration::seconds(5);
        let decision = DeterministicBtcStrategy::evaluate(&BtcStrategyConfig::default(), &snapshot);
        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(
            decision.reject_reason,
            Some(BtcRejectReason::OutsideEntryWindow)
        );
    }

    #[test]
    fn missing_dynamic_fee_is_no_trade_when_fees_are_enabled() {
        let mut snapshot = snapshot();
        snapshot.fee_rate = None;
        let decision = DeterministicBtcStrategy::evaluate(&BtcStrategyConfig::default(), &snapshot);
        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(
            decision.reject_reason,
            Some(BtcRejectReason::MissingFeeRate)
        );
    }

    #[test]
    fn insufficient_depth_is_no_trade() {
        let mut snapshot = snapshot();
        snapshot.up_book.ask_depth = dec!(1);
        snapshot.down_book.ask_depth = dec!(1);
        let decision = DeterministicBtcStrategy::evaluate(&BtcStrategyConfig::default(), &snapshot);
        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(
            decision.reject_reason,
            Some(BtcRejectReason::InsufficientDepth)
        );
    }

    #[test]
    fn decision_structurally_approves_at_most_one_side() {
        let mut snapshot = snapshot();
        snapshot.up_book.best_ask = Some(dec!(0.20));
        snapshot.up_book.best_bid = Some(dec!(0.19));
        snapshot.up_book.executable_ask_vwap = Some(dec!(0.20));
        snapshot.up_book.marketable_limit_price = Some(dec!(0.20));
        snapshot.down_book.best_ask = Some(dec!(0.80));
        snapshot.down_book.executable_ask_vwap = Some(dec!(0.80));
        snapshot.down_book.marketable_limit_price = Some(dec!(0.80));
        let decision = DeterministicBtcStrategy::evaluate(&BtcStrategyConfig::default(), &snapshot);
        assert_eq!(decision.action, BtcDecisionAction::BuyUp);
        assert_eq!(
            decision
                .approved_intent
                .as_ref()
                .map(|intent| intent.outcome),
            Some(BtcOutcome::Up)
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

    #[test]
    fn legacy_strategy_serialization_omits_continuation_contract() {
        let value = serde_json::to_value(BtcStrategyConfig::default()).unwrap();
        assert!(value.get("volatility_continuation").is_none());
        assert!(value.get("decision_strategy").is_none());
    }

    #[test]
    fn explicit_chainlink_selection_preserves_legacy_decision() {
        let legacy = DeterministicBtcStrategy::evaluate(&BtcStrategyConfig::default(), &snapshot());
        let explicit_config = BtcStrategyConfig {
            decision_strategy: Some(BtcDecisionStrategyConfig::ChainlinkFairValue {}),
            ..BtcStrategyConfig::default()
        };
        explicit_config.validate().unwrap();
        let explicit = DeterministicBtcStrategy::evaluate(&explicit_config, &snapshot());

        assert_eq!(explicit, legacy);
        assert_eq!(
            decision_sha256(&explicit),
            "60bd4559e35f10b8465cbe142085506ee860796411672be5baaeec98827f1c23"
        );
    }

    #[test]
    fn explicit_continuation_selection_preserves_legacy_decision() {
        let legacy =
            DeterministicBtcStrategy::evaluate(&continuation_config(), &continuation_snapshot());
        let explicit_config = BtcStrategyConfig {
            strategy_version: BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION.to_string(),
            decision_strategy: Some(BtcDecisionStrategyConfig::VolatilityContinuation {
                config: BtcVolatilityContinuationConfig::default(),
            }),
            ..BtcStrategyConfig::default()
        };
        explicit_config.validate().unwrap();
        let explicit =
            DeterministicBtcStrategy::evaluate(&explicit_config, &continuation_snapshot());

        assert_eq!(explicit, legacy);
        assert_eq!(
            decision_sha256(&explicit),
            "8c41f5bbc71da61e54a1b3f5a9984ec5ce97c4387f16c01b1c9159c1c3fd5fd4"
        );
    }

    #[test]
    fn explicit_selection_rejects_conflicting_legacy_contract() {
        let conflict = BtcStrategyConfig {
            decision_strategy: Some(BtcDecisionStrategyConfig::ChainlinkFairValue {}),
            volatility_continuation: Some(BtcVolatilityContinuationConfig::default()),
            ..BtcStrategyConfig::default()
        };

        assert!(conflict.validate().is_err());
    }

    #[test]
    fn market_anchored_selection_uses_common_engine_for_both_outcomes() {
        let config = market_anchored_config();
        config.validate().unwrap();
        let mut snapshot = snapshot();
        snapshot.chainlink_price = Some(dec!(102000));
        snapshot.chainlink_gap_bps = Some(dec!(200));

        let decision = DeterministicBtcStrategy::evaluate(&config, &snapshot);
        let fair_value = decision
            .fair_value
            .as_ref()
            .expect("candidate must emit a fair-value estimate");

        assert!(fair_value.market_up_prior.is_some());
        assert_eq!(fair_value.signed_logit_adjustment, Some(dec!(0.40)));
        assert!(decision.up_edge.is_some());
        assert!(decision.down_edge.is_some());
        assert_eq!(decision.action, BtcDecisionAction::BuyUp);
        assert_eq!(
            decision
                .up_edge
                .as_ref()
                .map(|edge| edge.conservative_probability),
            Some(fair_value.up_lower_bound)
        );
        assert_eq!(
            decision
                .approved_intent
                .as_ref()
                .map(|intent| intent.strategy_version.as_str()),
            Some(BTC_MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION)
        );
    }

    #[test]
    fn market_anchored_decision_matches_golden_contract() {
        let mut snapshot = snapshot();
        snapshot.chainlink_price = Some(dec!(102000));
        snapshot.chainlink_gap_bps = Some(dec!(200));
        let decision = DeterministicBtcStrategy::evaluate(&market_anchored_config(), &snapshot);

        assert_eq!(
            decision_sha256(&decision),
            "0598c0882f3d5d4873646722d5d9b784362bdfbe96fc548bdec3e069e6ee3134"
        );
    }

    #[test]
    fn market_anchored_profile_identity_and_floor_fail_closed() {
        let mut wrong_hash = market_anchored_config();
        wrong_hash.decision_strategy = Some(BtcDecisionStrategyConfig::MarketAnchoredFairValue {
            profile_id: BTC_MARKET_ANCHORED_RESEARCH_PROFILE_ID.to_string(),
            profile_sha256: "0".repeat(64),
        });
        assert!(wrong_hash.validate().is_err());

        let mut wrong_floor = market_anchored_config();
        wrong_floor.probability_floor = dec!(0.02);
        assert!(wrong_floor.validate().is_err());
    }

    #[test]
    fn strategy_attribution_is_dynamic_and_profile_aware() {
        let chainlink_config = BtcStrategyConfig::default();
        let chainlink = chainlink_config.attribution().unwrap();
        assert_eq!(chainlink.family, BTC_CHAINLINK_FAIR_VALUE_STRATEGY_FAMILY);
        assert_eq!(chainlink.strategy_version, BTC_STRATEGY_VERSION);
        assert_eq!(chainlink.profile_id, None);

        let continuation_config = continuation_config();
        let continuation = continuation_config.attribution().unwrap();
        assert_eq!(
            continuation.family,
            BTC_VOLATILITY_CONTINUATION_STRATEGY_FAMILY
        );
        assert_eq!(
            continuation.strategy_version,
            BTC_VOLATILITY_CONTINUATION_STRATEGY_VERSION
        );
        assert_eq!(continuation.profile_id, None);

        let market_anchored_config = market_anchored_config();
        let market_anchored = market_anchored_config.attribution().unwrap();
        assert_eq!(
            market_anchored.family,
            BTC_MARKET_ANCHORED_FAIR_VALUE_STRATEGY_FAMILY
        );
        assert_eq!(
            market_anchored.strategy_version,
            BTC_MARKET_ANCHORED_RESEARCH_STRATEGY_VERSION
        );
        assert_eq!(
            market_anchored.profile_id,
            Some(BTC_MARKET_ANCHORED_RESEARCH_PROFILE_ID)
        );
        assert_eq!(
            market_anchored.profile_sha256,
            Some(BTC_MARKET_ANCHORED_RESEARCH_PROFILE_SHA256)
        );
    }

    #[test]
    fn volatility_continuation_uses_market_prior_and_approves_only_confirmed_side() {
        let config = continuation_config();
        let snapshot = continuation_snapshot();
        let market_up_mid = dec!(0.555);
        let estimate = estimate_fair_value(&config, &snapshot).unwrap();
        assert!(estimate.up_probability > market_up_mid);
        assert!(estimate.up_probability - market_up_mid < dec!(0.10));

        let decision = DeterministicBtcStrategy::evaluate(&config, &snapshot);
        assert_eq!(decision.action, BtcDecisionAction::BuyUp);
        assert_eq!(decision.down_edge, None);
        assert_eq!(
            decision.approved_intent.map(|intent| intent.outcome),
            Some(BtcOutcome::Up)
        );
    }

    #[test]
    fn volatility_continuation_rejects_low_volatility_regime() {
        let config = continuation_config();
        let mut snapshot = continuation_snapshot();
        snapshot.realized_volatility = Some(dec!(0.00001));
        let decision = DeterministicBtcStrategy::evaluate(&config, &snapshot);
        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(
            decision.reject_reason,
            Some(BtcRejectReason::VolatilityRegimeBelowThreshold)
        );
    }

    #[test]
    fn volatility_continuation_rejects_unconfirmed_momentum() {
        let config = continuation_config();
        let mut snapshot = continuation_snapshot();
        snapshot.binance_return_30s = Some(dec!(-0.00020));
        let decision = DeterministicBtcStrategy::evaluate(&config, &snapshot);
        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(
            decision.reject_reason,
            Some(BtcRejectReason::ContinuationSignalUnconfirmed)
        );
    }

    #[test]
    fn volatility_continuation_rejects_outside_fixed_entry_horizon() {
        let config = continuation_config();
        let mut snapshot = continuation_snapshot();
        snapshot.observed_at = snapshot.window_start + Duration::seconds(150);
        let decision = DeterministicBtcStrategy::evaluate(&config, &snapshot);
        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(
            decision.reject_reason,
            Some(BtcRejectReason::OutsideEntryWindow)
        );
    }

    #[test]
    fn volatility_continuation_rejects_underdog_price() {
        let config = continuation_config();
        let mut snapshot = continuation_snapshot();
        snapshot.up_book = book(BtcOutcome::Up, "up", dec!(0.44), dec!(0.45));
        snapshot.down_book = book(BtcOutcome::Down, "down", dec!(0.55), dec!(0.56));
        let decision = DeterministicBtcStrategy::evaluate(&config, &snapshot);
        assert_eq!(decision.action, BtcDecisionAction::NoTrade);
        assert_eq!(
            decision.reject_reason,
            Some(BtcRejectReason::MarketPriorOutsideBounds)
        );
    }
}
