use std::collections::{BTreeSet, HashMap};

use chrono::{DateTime, Duration, Utc};
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::{Decimal, RoundingStrategy};
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    config::WhaleConfig,
    execution::OrderPlan,
    idempotency::{deterministic_client_order_id, ClientOrderIdSeed},
    models::{
        CopyTradeBacktestResult, CopyTradeSignal, EffectiveCopyTradeProcessConfig, OrderRequest,
        OrderSide, OrderType, SignalCandidate, SignalStatus, SignalType, WalletPerformance,
        WalletScore, WalletScoreCalibrationSnapshot, WalletSegmentPerformance, WhaleTrade,
    },
    segments::{
        classify_trade_segment, normalize_gamma_segment_key, SegmentClassification,
        MRS_SEGMENT_V2_SCORE_VERSION,
    },
};

pub const COPY_SCORE_VERSION: &str = "whale_score_v1";
pub const CALIBRATION_VERSION: &str = "copy_trade_calibration_v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeConfig {
    pub enabled: bool,
    pub min_wallet_score: Decimal,
    pub min_wallet_trades: i32,
    pub min_wallet_realized_pnl_usd: Decimal,
    pub min_wallet_roi: Decimal,
    pub min_wallet_closed_positions: i32,
    pub min_trade_usd: Decimal,
    pub min_copy_size_usd: Decimal,
    pub max_copy_size_usd: Decimal,
    pub max_open_notional_usd: Decimal,
    pub copy_size_fraction: Decimal,
    pub max_follow_lag_secs: i64,
    pub max_price_slippage_bps: Decimal,
    pub min_book_depth_usd: Decimal,
    pub backtest_horizon_secs: i64,
    pub taker_fee_rate: Decimal,
    pub allow_sell_entries: bool,
    pub mrs_enabled: bool,
    pub mrs_enforce: bool,
    pub min_mrs_score: Decimal,
    pub mrs_percentile_floor: Decimal,
    pub mrs_score_version: String,
    pub segment_scoring_enabled: bool,
    pub segment_scoring_mode: String,
    pub segment_score_version: String,
    pub segment_classifier_version: String,
    pub min_segment_score: Decimal,
    pub segment_mrs_percentile_floor: Decimal,
    pub min_segment_confidence: Decimal,
    pub min_segment_closed_positions: i32,
    pub min_segment_win_rate: Decimal,
    pub reject_negative_segment_roi_sample_size: i32,
    pub hard_reject_segment_win_rate_below: Decimal,
    pub hard_reject_segment_sample_size: i32,
    pub unknown_segment_policy: String,
    pub segment_allowlist: Vec<String>,
    pub entry_safety: CopyTradeEntrySafetyConfig,
}

impl Default for CopyTradeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_wallet_score: dec!(70),
            min_wallet_trades: 3,
            min_wallet_realized_pnl_usd: dec!(100),
            min_wallet_roi: dec!(0.05),
            min_wallet_closed_positions: 3,
            min_trade_usd: dec!(1000),
            min_copy_size_usd: dec!(10),
            max_copy_size_usd: dec!(100),
            max_open_notional_usd: dec!(20),
            copy_size_fraction: dec!(0.02),
            max_follow_lag_secs: 300,
            max_price_slippage_bps: dec!(150),
            min_book_depth_usd: dec!(25),
            backtest_horizon_secs: 3600,
            taker_fee_rate: dec!(0.03),
            allow_sell_entries: false,
            mrs_enabled: true,
            mrs_enforce: false,
            min_mrs_score: dec!(80),
            mrs_percentile_floor: dec!(0.80),
            mrs_score_version: "mrs_v1".to_string(),
            segment_scoring_enabled: false,
            segment_scoring_mode: "shadow".to_string(),
            segment_score_version: "mrs_segment_v1".to_string(),
            segment_classifier_version: "segment_rules_v1".to_string(),
            min_segment_score: dec!(50),
            segment_mrs_percentile_floor: dec!(0.95),
            min_segment_confidence: dec!(0.10),
            min_segment_closed_positions: 5,
            min_segment_win_rate: dec!(0.52),
            reject_negative_segment_roi_sample_size: 5,
            hard_reject_segment_win_rate_below: dec!(0.40),
            hard_reject_segment_sample_size: 10,
            unknown_segment_policy: "neutral".to_string(),
            segment_allowlist: Vec::new(),
            entry_safety: CopyTradeEntrySafetyConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeEntrySafetyConfig {
    pub enabled: bool,
    pub min_time_to_expiry_secs: i64,
    pub require_two_sided_book: bool,
    pub max_spread_bps: Decimal,
    pub require_exit_depth: bool,
    pub exit_depth_size_fraction: Decimal,
    pub exit_depth_slippage_bps: Decimal,
    pub min_entry_price: Decimal,
    pub max_entry_price: Decimal,
}

impl Default for CopyTradeEntrySafetyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_time_to_expiry_secs: 0,
            require_two_sided_book: false,
            max_spread_bps: Decimal::ZERO,
            require_exit_depth: false,
            exit_depth_size_fraction: dec!(1.0),
            exit_depth_slippage_bps: dec!(150),
            min_entry_price: Decimal::ZERO,
            max_entry_price: dec!(1.0),
        }
    }
}

impl From<&WhaleConfig> for CopyTradeConfig {
    fn from(config: &WhaleConfig) -> Self {
        Self {
            enabled: config.copy_trade_enabled,
            min_wallet_score: config.min_wallet_score,
            min_wallet_trades: config.min_wallet_trades,
            min_wallet_realized_pnl_usd: config.min_wallet_realized_pnl_usd,
            min_wallet_roi: config.min_wallet_roi,
            min_wallet_closed_positions: config.min_wallet_closed_positions,
            min_trade_usd: config.min_trade_usd,
            min_copy_size_usd: config.min_copy_size_usd,
            max_copy_size_usd: config.max_copy_size_usd,
            max_open_notional_usd: Self::default().max_open_notional_usd,
            copy_size_fraction: config.copy_size_fraction,
            max_follow_lag_secs: config.max_follow_lag.as_secs() as i64,
            max_price_slippage_bps: config.max_price_slippage_bps,
            min_book_depth_usd: config.min_book_depth_usd,
            backtest_horizon_secs: config.backtest_horizon.as_secs() as i64,
            allow_sell_entries: config.copy_allow_sell_entries,
            ..Self::default()
        }
    }
}

impl From<&EffectiveCopyTradeProcessConfig> for CopyTradeConfig {
    fn from(config: &EffectiveCopyTradeProcessConfig) -> Self {
        Self {
            enabled: config.enabled,
            min_wallet_score: config.min_wallet_score,
            min_wallet_trades: config.min_wallet_trades,
            min_wallet_realized_pnl_usd: config.min_wallet_realized_pnl_usd,
            min_wallet_roi: config.min_wallet_roi,
            min_wallet_closed_positions: config.min_wallet_closed_positions,
            min_trade_usd: config.min_trade_usd,
            min_copy_size_usd: config.min_copy_size_usd,
            max_copy_size_usd: config.max_copy_size_usd,
            max_open_notional_usd: config.max_open_notional_usd,
            copy_size_fraction: config.copy_size_fraction,
            max_follow_lag_secs: config.max_follow_lag_secs,
            max_price_slippage_bps: config.max_price_slippage_bps,
            min_book_depth_usd: config.min_book_depth_usd,
            backtest_horizon_secs: config.backtest_horizon_secs,
            taker_fee_rate: config.taker_fee_rate,
            allow_sell_entries: config.allow_sell_entries,
            mrs_enabled: config.mrs_enabled,
            mrs_enforce: config.mrs_enforce,
            min_mrs_score: config.min_mrs_score,
            mrs_percentile_floor: config.mrs_percentile_floor,
            mrs_score_version: config.mrs_score_version.clone(),
            segment_scoring_enabled: config.segment_scoring_enabled,
            segment_scoring_mode: config.segment_scoring_mode.clone(),
            segment_score_version: config.segment_score_version.clone(),
            segment_classifier_version: config.segment_classifier_version.clone(),
            min_segment_score: config.min_segment_score,
            segment_mrs_percentile_floor: config.segment_mrs_percentile_floor,
            min_segment_confidence: config.min_segment_confidence,
            min_segment_closed_positions: config.min_segment_closed_positions,
            min_segment_win_rate: config.min_segment_win_rate,
            reject_negative_segment_roi_sample_size: config.reject_negative_segment_roi_sample_size,
            hard_reject_segment_win_rate_below: config.hard_reject_segment_win_rate_below,
            hard_reject_segment_sample_size: config.hard_reject_segment_sample_size,
            unknown_segment_policy: config.unknown_segment_policy.clone(),
            segment_allowlist: config.segment_allowlist.clone(),
            entry_safety: CopyTradeEntrySafetyConfig {
                enabled: config.entry_safety.enabled,
                min_time_to_expiry_secs: config.entry_safety.min_time_to_expiry_secs,
                require_two_sided_book: config.entry_safety.require_two_sided_book,
                max_spread_bps: config.entry_safety.max_spread_bps,
                require_exit_depth: config.entry_safety.require_exit_depth,
                exit_depth_size_fraction: config.entry_safety.exit_depth_size_fraction,
                exit_depth_slippage_bps: config.entry_safety.exit_depth_slippage_bps,
                min_entry_price: config.entry_safety.min_entry_price,
                max_entry_price: config.entry_safety.max_entry_price,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeWalletPerformance {
    pub proxy_wallet: String,
    pub realized_pnl_usd: Decimal,
    pub roi: Decimal,
    pub closed_positions: i32,
    pub wallet_score_diagnostic: Option<Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeMrsScore {
    pub score: Decimal,
    pub score_version: String,
    pub percentile: Option<Decimal>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeSegmentScore {
    pub segment_key: String,
    pub score: Decimal,
    pub score_version: String,
    pub classifier_version: String,
    pub confidence: Decimal,
    pub closed_positions: i32,
    pub winning_positions: i32,
    pub win_rate: Decimal,
    pub realized_pnl_usd: Decimal,
    pub total_bought_usd: Decimal,
    pub roi: Decimal,
    pub observed_trade_count: i32,
    pub observed_volume_usd: Decimal,
    pub percentile: Option<Decimal>,
    pub metadata: serde_json::Value,
}

impl CopyTradeWalletPerformance {
    pub fn rank_score(&self) -> Decimal {
        self.realized_pnl_usd * self.roi
    }
}

impl From<&WalletSegmentPerformance> for CopyTradeSegmentScore {
    fn from(performance: &WalletSegmentPerformance) -> Self {
        Self {
            segment_key: performance.segment_key.clone(),
            score: performance.score,
            score_version: performance.score_version.clone(),
            classifier_version: performance.classifier_version.clone(),
            confidence: performance.confidence,
            closed_positions: performance.closed_positions,
            winning_positions: performance.winning_positions,
            win_rate: performance.win_rate,
            realized_pnl_usd: performance.realized_pnl_usd,
            total_bought_usd: performance.total_bought_usd,
            roi: performance.roi,
            observed_trade_count: performance.observed_trade_count,
            observed_volume_usd: performance.observed_volume_usd,
            percentile: performance
                .metadata
                .get("segment_percentile")
                .and_then(|value| value.as_str())
                .and_then(|value| value.parse::<Decimal>().ok())
                .or_else(|| {
                    performance
                        .metadata
                        .get("segment_percentile")
                        .and_then(|value| value.as_f64())
                        .and_then(Decimal::from_f64)
                }),
            metadata: performance.metadata.clone(),
        }
    }
}

impl From<&WalletScore> for CopyTradeWalletPerformance {
    fn from(score: &WalletScore) -> Self {
        Self {
            proxy_wallet: score.proxy_wallet.clone(),
            realized_pnl_usd: score.realized_pnl,
            roi: score.roi,
            closed_positions: closed_positions_from_score(score),
            wallet_score_diagnostic: Some(score.score),
        }
    }
}

impl From<&WalletPerformance> for CopyTradeWalletPerformance {
    fn from(performance: &WalletPerformance) -> Self {
        Self {
            proxy_wallet: performance.proxy_wallet.clone(),
            realized_pnl_usd: performance.realized_pnl_usd,
            roi: performance.roi,
            closed_positions: performance.closed_positions,
            wallet_score_diagnostic: Some(performance.rank_score),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ObservedMarket {
    pub observed_price: Decimal,
    pub available_depth_usd: Decimal,
    pub observed_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct CopyTradeDecision {
    pub signal_candidate: SignalCandidate,
    pub copy_signal: CopyTradeSignal,
    pub order_plan: Option<OrderPlan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopyTradeBacktestSummary {
    pub backtest_id: Uuid,
    pub wallet_count: i32,
    pub signal_count: i32,
    pub trade_count: i32,
    pub gross_pnl_usd: Decimal,
    pub net_pnl_usd: Decimal,
    pub roi: Option<Decimal>,
    pub max_drawdown: Option<Decimal>,
    pub win_rate: Option<Decimal>,
    pub recommended_min_wallet_score: Decimal,
    pub threshold_trials: serde_json::Value,
}

pub fn evaluate_copy_trade(
    trade: &WhaleTrade,
    performance: Option<&CopyTradeWalletPerformance>,
    mrs_score: Option<&CopyTradeMrsScore>,
    segment_score: Option<&CopyTradeSegmentScore>,
    observed: ObservedMarket,
    config: &CopyTradeConfig,
    process_id: Option<Uuid>,
) -> CopyTradeDecision {
    evaluate_copy_trade_with_segment(
        trade,
        performance,
        mrs_score,
        segment_score,
        None,
        observed,
        config,
        process_id,
    )
}

pub fn evaluate_copy_trade_with_segment(
    trade: &WhaleTrade,
    performance: Option<&CopyTradeWalletPerformance>,
    mrs_score: Option<&CopyTradeMrsScore>,
    segment_score: Option<&CopyTradeSegmentScore>,
    segment_classification: Option<SegmentClassification>,
    observed: ObservedMarket,
    config: &CopyTradeConfig,
    process_id: Option<Uuid>,
) -> CopyTradeDecision {
    let signal_id = trade.trade_id;
    let rank_score = performance
        .map(CopyTradeWalletPerformance::rank_score)
        .unwrap_or(Decimal::ZERO);
    let token_id = non_empty(trade.asset.clone());
    let market_id = trade
        .market_id
        .clone()
        .or_else(|| trade.condition_id.clone())
        .or_else(|| trade.slug.clone());
    let copy_size_usd = bounded_copy_size(trade.cash_value, config);
    let side = trade.side.to_ascii_uppercase();
    let lag_secs = (observed.observed_at - trade.timestamp_utc)
        .num_seconds()
        .max(0);
    let slippage_bps = price_slippage_bps(trade.price, observed.observed_price);
    let mut reject_reason = None;
    let segment_classification =
        segment_classification.unwrap_or_else(|| classify_trade_segment(trade));
    let segment_filter_decision =
        evaluate_segment_filter(&segment_classification.segment_key, config);
    let segment_decision =
        evaluate_segment_gate(&segment_classification.segment_key, segment_score, config);

    if !config.enabled {
        reject_reason = Some("copy_trade_disabled");
    } else if token_id.is_none() {
        reject_reason = Some("missing_token_id");
    } else if !matches!(side.as_str(), "BUY" | "SELL") {
        reject_reason = Some("unsupported_trade_side");
    } else if side == "SELL" && !config.allow_sell_entries {
        reject_reason = Some("sell_entries_disabled");
    } else if segment_filter_decision.enforced_reject {
        reject_reason = Some(segment_filter_decision.reason);
    } else if config.mrs_enabled && config.mrs_enforce && mrs_score.is_none() {
        reject_reason = Some("missing_mrs_score");
    } else if config.mrs_enabled
        && config.mrs_enforce
        && mrs_score
            .map(|score| score.score < config.min_mrs_score)
            .unwrap_or(true)
    {
        reject_reason = Some("mrs_score_below_threshold");
    } else if performance.is_none() {
        reject_reason = Some("missing_wallet_performance");
    } else if performance
        .map(|performance| performance.realized_pnl_usd < config.min_wallet_realized_pnl_usd)
        .unwrap_or(true)
    {
        reject_reason = Some("pnl_below_threshold");
    } else if performance
        .map(|performance| performance.roi < config.min_wallet_roi)
        .unwrap_or(true)
    {
        reject_reason = Some("roi_below_threshold");
    } else if performance
        .map(|performance| performance.closed_positions < config.min_wallet_closed_positions)
        .unwrap_or(true)
    {
        reject_reason = Some("closed_positions_below_threshold");
    } else if trade.cash_value < config.min_trade_usd {
        reject_reason = Some("trade_size_below_threshold");
    } else if lag_secs > config.max_follow_lag_secs {
        reject_reason = Some("trade_too_old");
    } else if slippage_bps > config.max_price_slippage_bps {
        reject_reason = Some("price_moved_too_far");
    } else if observed.available_depth_usd < config.min_book_depth_usd {
        reject_reason = Some("insufficient_observed_depth");
    } else if copy_size_usd < config.min_copy_size_usd {
        reject_reason = Some("copy_size_below_minimum");
    } else if segment_decision.enforced_reject {
        reject_reason = Some(segment_decision.reason);
    }

    let detected = reject_reason.is_none();
    let reason = reject_reason.unwrap_or("qualified_profitable_whale_follow");
    let status = if detected { "detected" } else { "rejected" }.to_string();
    let metadata = serde_json::json!({
        "trade_cash_value": trade.cash_value,
        "wallet_score_basis": if config.mrs_enabled { "mrs_v1" } else { "wallet_rank_score" },
        "wallet_rank_score": rank_score,
        "wallet_realized_pnl_usd": performance.map(|performance| performance.realized_pnl_usd),
        "wallet_roi": performance.map(|performance| performance.roi),
        "wallet_closed_positions": performance.map(|performance| performance.closed_positions),
        "wallet_score_diagnostic": performance.and_then(|performance| performance.wallet_score_diagnostic),
        "mrs": {
            "enabled": config.mrs_enabled,
            "enforced": config.mrs_enforce,
            "min_score": config.min_mrs_score,
            "percentile_floor": config.mrs_percentile_floor,
            "score_version_config": config.mrs_score_version,
            "score": mrs_score.map(|score| score.score),
            "score_version": mrs_score.map(|score| score.score_version.as_str()),
            "percentile": mrs_score.and_then(|score| score.percentile),
            "passed": mrs_score
                .map(|score| score.score >= config.min_mrs_score)
                .unwrap_or(false),
            "metadata": mrs_score.map(|score| score.metadata.clone())
        },
        "segment": {
            "enabled": config.segment_scoring_enabled,
            "mode": config.segment_scoring_mode,
            "score_version_config": config.segment_score_version,
            "classifier_version_config": config.segment_classifier_version,
            "segment_key": segment_classification.segment_key,
            "classifier_version": segment_classification.classifier_version,
            "classifier_confidence": segment_classification.confidence,
            "matched_rule": segment_classification.matched_rule,
            "matched_terms": segment_classification.matched_terms,
            "source_fields": segment_classification.source_fields,
            "score": segment_score.map(|score| score.score),
            "score_version": segment_score.map(|score| score.score_version.as_str()),
            "confidence": segment_score.map(|score| score.confidence),
            "percentile": segment_score.and_then(|score| score.percentile),
            "closed_positions": segment_score.map(|score| score.closed_positions),
            "winning_positions": segment_score.map(|score| score.winning_positions),
            "win_rate": segment_score.map(|score| score.win_rate),
            "roi": segment_score.map(|score| score.roi),
            "realized_pnl_usd": segment_score.map(|score| score.realized_pnl_usd),
            "observed_trade_count": segment_score.map(|score| score.observed_trade_count),
            "observed_volume_usd": segment_score.map(|score| score.observed_volume_usd),
            "decision": segment_decision.decision,
            "reject_reason": segment_decision.reject_reason,
            "enforced_reject": segment_decision.enforced_reject,
            "thresholds": {
                "min_segment_score": config.min_segment_score,
                "segment_mrs_percentile_floor": config.segment_mrs_percentile_floor,
                "min_segment_confidence": config.min_segment_confidence,
                "min_closed_positions": config.min_segment_closed_positions,
                "min_win_rate": config.min_segment_win_rate,
                "reject_negative_roi_sample_size": config.reject_negative_segment_roi_sample_size,
                "hard_reject_win_rate_below": config.hard_reject_segment_win_rate_below,
                "hard_reject_sample_size": config.hard_reject_segment_sample_size,
                "unknown_segment_policy": config.unknown_segment_policy
            },
            "metadata": segment_score.map(|score| score.metadata.clone())
        },
        "segment_filter": {
            "enabled": !config.segment_allowlist.is_empty(),
            "allowed_segments": segment_filter_decision.allowed_segments,
            "segment_key": segment_classification.segment_key,
            "decision": segment_filter_decision.decision,
            "reject_reason": segment_filter_decision.reject_reason,
            "enforced_reject": segment_filter_decision.enforced_reject
        },
        "lag_secs": lag_secs,
        "slippage_bps": slippage_bps,
        "available_depth_usd": observed.available_depth_usd,
        "process_id": process_id,
        "config": config,
    });

    let copy_signal = CopyTradeSignal {
        signal_id,
        process_id,
        timestamp_utc: trade.timestamp_utc,
        proxy_wallet: trade.proxy_wallet.clone(),
        source_trade_id: trade.trade_id,
        market_id: market_id.clone(),
        token_id: token_id.clone(),
        side: side.clone(),
        whale_price: trade.price,
        observed_price: observed.observed_price,
        copy_size_usd,
        reason: reason.to_string(),
        status: status.clone(),
        metadata: metadata.clone(),
    };

    let signal_candidate = SignalCandidate {
        signal_id,
        process_id,
        signal_type: SignalType::WhaleFollow,
        market_id: market_id.clone().unwrap_or_else(|| "unknown".to_string()),
        expected_edge: Decimal::ZERO,
        threshold: config.min_wallet_realized_pnl_usd,
        size: copy_size_usd,
        status: if detected {
            SignalStatus::Detected
        } else {
            SignalStatus::Rejected
        },
        reject_reason: (!detected).then(|| reason.to_string()),
        worst_case_loss: Some(copy_size_usd),
        metadata: metadata.clone(),
    };

    let order_plan = if detected {
        token_id.map(|token_id| {
            let market_id = market_id.unwrap_or_else(|| "unknown".to_string());
            let order_side = if side == "SELL" {
                OrderSide::Sell
            } else {
                OrderSide::Buy
            };
            let limit_price = clob_tick_price(observed.observed_price, order_side);
            let client_order_id = deterministic_client_order_id(&ClientOrderIdSeed {
                strategy_version: "whale-follow-v1",
                process_id,
                source_id: signal_id,
                purpose: "whale_follow_entry",
                market_id: &market_id,
                token_id: &token_id,
                side: order_side,
                notional_key: &copy_size_usd.round_dp(4).normalize().to_string(),
            });
            OrderPlan {
                plan_id: signal_id,
                orders: vec![OrderRequest {
                    client_order_id,
                    process_id,
                    market_id,
                    token_id,
                    side: order_side,
                    order_type: OrderType::Fok,
                    price: limit_price,
                    size: shares_for_notional(copy_size_usd, limit_price, order_side),
                    signal_id: Some(signal_id),
                    metadata: serde_json::json!({"purpose": "whale_follow_entry"}),
                }],
            }
        })
    } else {
        None
    };

    CopyTradeDecision {
        signal_candidate,
        copy_signal,
        order_plan,
    }
}

#[derive(Debug, Clone)]
struct SegmentGateDecision {
    decision: &'static str,
    reject_reason: Option<&'static str>,
    reason: &'static str,
    enforced_reject: bool,
}

#[derive(Debug, Clone)]
struct SegmentFilterDecision {
    decision: &'static str,
    reject_reason: Option<&'static str>,
    reason: &'static str,
    enforced_reject: bool,
    allowed_segments: Vec<String>,
}

fn evaluate_segment_filter(segment_key: &str, config: &CopyTradeConfig) -> SegmentFilterDecision {
    let allowed_segments = normalized_segment_allowlist(&config.segment_allowlist);
    if allowed_segments.is_empty() {
        return SegmentFilterDecision {
            decision: "disabled",
            reject_reason: None,
            reason: "segment_filter_disabled",
            enforced_reject: false,
            allowed_segments,
        };
    }

    if allowed_segments
        .iter()
        .any(|allowed| allowed.eq_ignore_ascii_case(segment_key))
    {
        return SegmentFilterDecision {
            decision: "accepted",
            reject_reason: None,
            reason: "segment_filter_passed",
            enforced_reject: false,
            allowed_segments,
        };
    }

    SegmentFilterDecision {
        decision: "rejected",
        reject_reason: Some("segment_not_allowlisted"),
        reason: "segment_not_allowlisted",
        enforced_reject: true,
        allowed_segments,
    }
}

fn normalized_segment_allowlist(segments: &[String]) -> Vec<String> {
    segments
        .iter()
        .filter_map(|segment| {
            normalize_gamma_segment_key(segment).or_else(|| {
                let trimmed = segment.trim().to_ascii_lowercase();
                (!trimmed.is_empty()).then_some(trimmed)
            })
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn evaluate_segment_gate(
    segment_key: &str,
    score: Option<&CopyTradeSegmentScore>,
    config: &CopyTradeConfig,
) -> SegmentGateDecision {
    if !config.segment_scoring_enabled || config.segment_scoring_mode == "off" {
        return SegmentGateDecision {
            decision: "disabled",
            reject_reason: None,
            reason: "segment_scoring_disabled",
            enforced_reject: false,
        };
    }

    let reject_reason = segment_reject_reason(segment_key, score, config);
    let enforce = matches!(
        config.segment_scoring_mode.as_str(),
        "sim_enforce" | "live_enforce"
    );
    match reject_reason {
        Some(reason) if enforce => SegmentGateDecision {
            decision: "rejected",
            reject_reason: Some(reason),
            reason,
            enforced_reject: true,
        },
        Some(reason) => SegmentGateDecision {
            decision: "would_reject",
            reject_reason: Some(reason),
            reason,
            enforced_reject: false,
        },
        None if enforce => SegmentGateDecision {
            decision: "accepted",
            reject_reason: None,
            reason: "segment_gate_passed",
            enforced_reject: false,
        },
        None => SegmentGateDecision {
            decision: "would_accept",
            reject_reason: None,
            reason: "segment_gate_passed",
            enforced_reject: false,
        },
    }
}

fn segment_reject_reason(
    segment_key: &str,
    score: Option<&CopyTradeSegmentScore>,
    config: &CopyTradeConfig,
) -> Option<&'static str> {
    if segment_key == "other" && config.unknown_segment_policy == "reject" {
        return Some("unknown_segment");
    }
    let Some(score) = score else {
        return if matches!(
            config.segment_scoring_mode.as_str(),
            "sim_enforce" | "live_enforce"
        ) {
            Some("missing_segment_score")
        } else {
            None
        };
    };
    if score.confidence < config.min_segment_confidence {
        return Some("segment_confidence_below_threshold");
    }
    if config.segment_score_version == MRS_SEGMENT_V2_SCORE_VERSION {
        if score
            .percentile
            .map(|percentile| percentile < config.segment_mrs_percentile_floor)
            .unwrap_or(true)
        {
            return Some("segment_percentile_below_threshold");
        }
    }
    if score.closed_positions < config.min_segment_closed_positions {
        return None;
    }
    if score.closed_positions >= config.hard_reject_segment_sample_size
        && score.win_rate < config.hard_reject_segment_win_rate_below
    {
        return Some("segment_win_rate_hard_reject");
    }
    if score.closed_positions >= config.reject_negative_segment_roi_sample_size
        && score.roi < Decimal::ZERO
    {
        return Some("segment_negative_roi");
    }
    if score.win_rate < config.min_segment_win_rate {
        return Some("segment_win_rate_below_threshold");
    }
    if score.score < config.min_segment_score {
        return Some("segment_score_below_threshold");
    }
    None
}

pub fn run_copy_trade_backtest(
    trades: &[WhaleTrade],
    scores: &[WalletScore],
    config: &CopyTradeConfig,
) -> (CopyTradeBacktestResult, WalletScoreCalibrationSnapshot) {
    let backtest_id = Uuid::new_v4();
    let mut ordered_trades = trades.to_vec();
    ordered_trades.sort_by_key(|trade| trade.timestamp_utc);
    let performances: Vec<CopyTradeWalletPerformance> = scores
        .iter()
        .map(CopyTradeWalletPerformance::from)
        .collect();
    let performance_by_wallet: HashMap<&str, &CopyTradeWalletPerformance> = performances
        .iter()
        .map(|performance| (performance.proxy_wallet.as_str(), performance))
        .collect();

    let result =
        backtest_at_threshold(backtest_id, &ordered_trades, &performance_by_wallet, config);
    let trials = threshold_trials(&ordered_trades, &performance_by_wallet, config);
    let recommended_min_wallet_score =
        recommend_performance_threshold(&trials, config.min_wallet_realized_pnl_usd);
    let wallet_count = ordered_trades
        .iter()
        .map(|trade| trade.proxy_wallet.as_str())
        .collect::<BTreeSet<_>>()
        .len() as i32;
    let trade_count = ordered_trades.len() as i32;
    let range_start = ordered_trades.first().map(|trade| trade.timestamp_utc);
    let range_end = ordered_trades.last().map(|trade| trade.timestamp_utc);

    let calibration = WalletScoreCalibrationSnapshot {
        snapshot_id: Uuid::new_v4(),
        timestamp_utc: Utc::now(),
        score_version: COPY_SCORE_VERSION.to_string(),
        calibration_version: CALIBRATION_VERSION.to_string(),
        sample_start: range_start,
        sample_end: range_end,
        wallet_count,
        trade_count,
        feature_weights: serde_json::json!({
            "current_score_model": COPY_SCORE_VERSION,
            "note": "copy decisions use persisted wallet performance gates; wallet score is diagnostic only"
        }),
        thresholds: serde_json::json!({
            "current_min_wallet_realized_pnl_usd": config.min_wallet_realized_pnl_usd,
            "recommended_min_wallet_realized_pnl_usd": recommended_min_wallet_score,
            "min_wallet_roi": config.min_wallet_roi,
            "min_wallet_closed_positions": config.min_wallet_closed_positions,
            "min_trade_usd": config.min_trade_usd,
            "max_price_slippage_bps": config.max_price_slippage_bps
        }),
        metrics: serde_json::json!({
            "backtest_id": backtest_id,
            "threshold_trials": trials,
            "net_pnl_usd": result.net_pnl_usd,
            "win_rate": result.win_rate,
            "roi": result.roi
        }),
        metadata: serde_json::json!({
            "backtest_horizon_secs": config.backtest_horizon_secs,
            "taker_fee_rate": config.taker_fee_rate
        }),
    };

    (result, calibration)
}

fn backtest_at_threshold(
    backtest_id: Uuid,
    trades: &[WhaleTrade],
    performance_by_wallet: &HashMap<&str, &CopyTradeWalletPerformance>,
    config: &CopyTradeConfig,
) -> CopyTradeBacktestResult {
    let mut net_pnl = Vec::new();
    let mut gross_pnl = Vec::new();
    let mut invested = Decimal::ZERO;
    let mut signals = 0;
    let mut traded_wallets = BTreeSet::new();
    let range_start = trades.first().map(|trade| trade.timestamp_utc);
    let range_end = trades.last().map(|trade| trade.timestamp_utc);

    for trade in trades {
        let Some(performance) = performance_by_wallet
            .get(trade.proxy_wallet.as_str())
            .copied()
        else {
            continue;
        };
        let decision = evaluate_copy_trade(
            trade,
            Some(performance),
            None,
            None,
            ObservedMarket {
                observed_price: trade.price,
                available_depth_usd: trade.cash_value,
                observed_at: trade.timestamp_utc,
            },
            config,
            None,
        );
        if decision.order_plan.is_none() {
            continue;
        }
        let Some(exit_price) = future_price(trades, trade, config.backtest_horizon_secs) else {
            continue;
        };
        let (order_side, side_multiplier) = if trade.side.eq_ignore_ascii_case("SELL") {
            (OrderSide::Sell, dec!(-1))
        } else {
            (OrderSide::Buy, dec!(1))
        };
        let shares =
            shares_for_notional(decision.copy_signal.copy_size_usd, trade.price, order_side);
        let gross = (exit_price - trade.price) * shares * side_multiplier;
        let fees = decision.copy_signal.copy_size_usd * config.taker_fee_rate;
        gross_pnl.push(gross);
        net_pnl.push(gross - fees);
        invested += decision.copy_signal.copy_size_usd;
        signals += 1;
        traded_wallets.insert(trade.proxy_wallet.as_str());
    }

    let net_pnl_usd: Decimal = net_pnl.iter().copied().sum();
    let gross_pnl_usd: Decimal = gross_pnl.iter().copied().sum();
    let wins = net_pnl
        .iter()
        .filter(|value| **value > Decimal::ZERO)
        .count();
    let max_drawdown = max_drawdown(&net_pnl);

    CopyTradeBacktestResult {
        result_id: Uuid::new_v4(),
        backtest_id,
        timestamp_utc: Utc::now(),
        wallet_count: traded_wallets.len() as i32,
        signal_count: signals,
        trade_count: signals,
        gross_pnl_usd,
        net_pnl_usd,
        roi: (invested > Decimal::ZERO).then(|| net_pnl_usd / invested),
        max_drawdown: Some(max_drawdown),
        win_rate: (signals > 0)
            .then(|| Decimal::from_i64(wins as i64).unwrap_or_default() / Decimal::from(signals)),
        result_summary: serde_json::json!({
            "range_start": range_start,
            "range_end": range_end,
            "horizon_secs": config.backtest_horizon_secs,
            "fee_rate": config.taker_fee_rate,
            "min_wallet_realized_pnl_usd": config.min_wallet_realized_pnl_usd,
            "min_wallet_roi": config.min_wallet_roi,
            "min_wallet_closed_positions": config.min_wallet_closed_positions,
        }),
    }
}

fn threshold_trials(
    trades: &[WhaleTrade],
    performance_by_wallet: &HashMap<&str, &CopyTradeWalletPerformance>,
    config: &CopyTradeConfig,
) -> serde_json::Value {
    let mut trials = Vec::new();
    for threshold in [
        dec!(0),
        dec!(100),
        dec!(250),
        dec!(500),
        dec!(1000),
        dec!(2500),
    ] {
        let mut trial_config = config.clone();
        trial_config.min_wallet_realized_pnl_usd = threshold;
        let result =
            backtest_at_threshold(Uuid::new_v4(), trades, performance_by_wallet, &trial_config);
        trials.push(serde_json::json!({
            "min_wallet_realized_pnl_usd": threshold,
            "signals": result.signal_count,
            "net_pnl_usd": result.net_pnl_usd,
            "win_rate": result.win_rate,
            "roi": result.roi
        }));
    }
    serde_json::Value::Array(trials)
}

fn recommend_performance_threshold(trials: &serde_json::Value, fallback: Decimal) -> Decimal {
    let Some(items) = trials.as_array() else {
        return fallback;
    };
    items
        .iter()
        .filter_map(|item| {
            let threshold = item
                .get("min_wallet_realized_pnl_usd")?
                .as_str()?
                .parse::<Decimal>()
                .ok()?;
            let net = item.get("net_pnl_usd")?.as_str()?.parse::<Decimal>().ok()?;
            let signals = item.get("signals")?.as_i64().unwrap_or_default();
            (signals >= 5 && net > Decimal::ZERO).then_some((threshold, net))
        })
        .max_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(threshold, _)| threshold)
        .unwrap_or(fallback)
}

fn future_price(trades: &[WhaleTrade], entry: &WhaleTrade, horizon_secs: i64) -> Option<Decimal> {
    let target = entry.timestamp_utc + Duration::seconds(horizon_secs);
    trades
        .iter()
        .filter(|trade| trade.asset == entry.asset && trade.timestamp_utc >= target)
        .min_by_key(|trade| trade.timestamp_utc)
        .map(|trade| trade.price)
}

fn max_drawdown(pnl: &[Decimal]) -> Decimal {
    let mut equity = Decimal::ZERO;
    let mut peak = Decimal::ZERO;
    let mut drawdown = Decimal::ZERO;
    for value in pnl {
        equity += *value;
        peak = peak.max(equity);
        drawdown = drawdown.max(peak - equity);
    }
    drawdown
}

fn bounded_copy_size(trade_cash_value: Decimal, config: &CopyTradeConfig) -> Decimal {
    (trade_cash_value * config.copy_size_fraction)
        .max(config.min_copy_size_usd)
        .min(config.max_copy_size_usd)
}

fn shares_for_notional(notional: Decimal, price: Decimal, side: OrderSide) -> Decimal {
    if price <= Decimal::ZERO {
        return Decimal::ZERO;
    }
    let mut shares = (notional / price).round_dp_with_strategy(2, RoundingStrategy::ToZero);
    if side != OrderSide::Buy {
        return shares;
    }

    while shares > Decimal::ZERO {
        let maker_amount = shares * price;
        if maker_amount <= notional
            && maker_amount == maker_amount.round_dp_with_strategy(2, RoundingStrategy::ToZero)
        {
            return shares;
        }
        shares -= dec!(0.01);
    }
    Decimal::ZERO
}

fn clob_tick_price(price: Decimal, side: OrderSide) -> Decimal {
    let rounded = match side {
        OrderSide::Buy => price.round_dp_with_strategy(2, RoundingStrategy::ToPositiveInfinity),
        OrderSide::Sell => price.round_dp_with_strategy(2, RoundingStrategy::ToNegativeInfinity),
    };
    rounded.clamp(dec!(0.01), dec!(0.99))
}

fn price_slippage_bps(reference: Decimal, observed: Decimal) -> Decimal {
    if reference <= Decimal::ZERO {
        return Decimal::MAX;
    }
    ((observed - reference).abs() / reference) * dec!(10000)
}

fn non_empty(value: String) -> Option<String> {
    (!value.trim().is_empty()).then_some(value)
}

fn closed_positions_from_score(score: &WalletScore) -> i32 {
    score
        .metadata
        .get("closed_position_count")
        .and_then(|value| value.as_i64())
        .and_then(|value| i32::try_from(value).ok())
        .unwrap_or(score.resolved_markets)
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    use crate::{
        copytrade::{
            clob_tick_price, evaluate_copy_trade, run_copy_trade_backtest, CopyTradeConfig,
            CopyTradeMrsScore, CopyTradeSegmentScore, CopyTradeWalletPerformance, ObservedMarket,
        },
        models::{OrderSide, WalletScore, WhaleTrade},
        segments::MRS_SEGMENT_V2_SCORE_VERSION,
    };

    fn trade(wallet: &str, asset: &str, price: rust_decimal::Decimal, minutes: i64) -> WhaleTrade {
        WhaleTrade {
            trade_id: Uuid::new_v4(),
            proxy_wallet: wallet.to_string(),
            asset: asset.to_string(),
            condition_id: Some("condition".to_string()),
            market_id: Some("market".to_string()),
            side: "BUY".to_string(),
            outcome: Some("Yes".to_string()),
            price,
            size: dec!(10000),
            cash_value: price * dec!(10000),
            timestamp_utc: Utc::now() + chrono::Duration::minutes(minutes),
            title: None,
            slug: None,
            event_slug: None,
            transaction_hash: Some(format!("tx-{wallet}-{asset}-{minutes}")),
            raw_payload: serde_json::json!({}),
        }
    }

    fn score(wallet: &str, score: rust_decimal::Decimal) -> WalletScore {
        WalletScore {
            proxy_wallet: wallet.to_string(),
            score_version: "whale_score_v1".to_string(),
            resolved_markets: 10,
            total_trades: 20,
            total_volume: dec!(100000),
            realized_pnl: dec!(1000),
            roi: dec!(0.10),
            win_rate: dec!(0.60),
            avg_trade_size: dec!(5000),
            max_drawdown: dec!(100),
            score,
            metadata: serde_json::json!({
                "closed_position_count": 10
            }),
        }
    }

    fn performance(
        wallet: &str,
        realized_pnl_usd: rust_decimal::Decimal,
        roi: rust_decimal::Decimal,
        closed_positions: i32,
    ) -> CopyTradeWalletPerformance {
        CopyTradeWalletPerformance {
            proxy_wallet: wallet.to_string(),
            realized_pnl_usd,
            roi,
            closed_positions,
            wallet_score_diagnostic: Some(dec!(80)),
        }
    }

    fn segment_score(segment_key: &str, score: rust_decimal::Decimal) -> CopyTradeSegmentScore {
        CopyTradeSegmentScore {
            segment_key: segment_key.to_string(),
            score,
            score_version: "mrs_segment_v1".to_string(),
            classifier_version: "segment_rules_v1".to_string(),
            confidence: dec!(0.90),
            closed_positions: 12,
            winning_positions: 4,
            win_rate: dec!(0.3333),
            realized_pnl_usd: dec!(-100),
            total_bought_usd: dec!(1000),
            roi: dec!(-0.10),
            observed_trade_count: 20,
            observed_volume_usd: dec!(5000),
            percentile: None,
            metadata: serde_json::json!({}),
        }
    }

    #[test]
    fn evaluates_qualified_whale_trade_into_order_plan() {
        let config = CopyTradeConfig::default();
        let trade = trade("0xabc", "token", dec!(0.50), 0);
        let wallet_performance = performance("0xabc", dec!(1000), dec!(0.10), 10);
        let decision = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            None,
            None,
            ObservedMarket {
                observed_price: dec!(0.505),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc + chrono::Duration::seconds(10),
            },
            &config,
            None,
        );
        assert_eq!(decision.copy_signal.status, "detected");
        let order = &decision.order_plan.as_ref().unwrap().orders[0];
        assert_eq!(order.price, dec!(0.51));
    }

    #[test]
    fn generated_order_size_does_not_exceed_copy_notional_cap() {
        let mut config = CopyTradeConfig::default();
        config.min_copy_size_usd = dec!(2);
        config.max_copy_size_usd = dec!(2);
        let trade = trade("0xabc", "token", dec!(0.37), 0);
        let wallet_performance = performance("0xabc", dec!(1000), dec!(0.10), 10);
        let decision = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            None,
            None,
            ObservedMarket {
                observed_price: dec!(0.37),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc + chrono::Duration::seconds(10),
            },
            &config,
            None,
        );
        let order = &decision.order_plan.unwrap().orders[0];
        assert_eq!(order.size, dec!(5.00));
        assert!(order.price * order.size <= dec!(2));
        assert_eq!(
            (order.price * order.size).round_dp(2),
            order.price * order.size
        );
    }

    #[test]
    fn generated_buy_order_has_clob_compatible_collateral_precision() {
        let mut config = CopyTradeConfig::default();
        config.min_copy_size_usd = dec!(2);
        config.max_copy_size_usd = dec!(2);
        let trade = trade("0xabc", "token", dec!(0.15), 0);
        let wallet_performance = performance("0xabc", dec!(1000), dec!(0.10), 10);
        let decision = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            None,
            None,
            ObservedMarket {
                observed_price: dec!(0.15),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc + chrono::Duration::seconds(10),
            },
            &config,
            None,
        );
        let order = &decision.order_plan.unwrap().orders[0];
        assert_eq!(order.price, dec!(0.15));
        assert_eq!(order.size, dec!(13.20));
        assert_eq!(order.price * order.size, dec!(1.98));
    }

    #[test]
    fn generated_order_price_stays_inside_clob_price_bounds() {
        assert_eq!(clob_tick_price(dec!(0.999), OrderSide::Buy), dec!(0.99));
        assert_eq!(clob_tick_price(dec!(1.00), OrderSide::Buy), dec!(0.99));
        assert_eq!(clob_tick_price(dec!(0.001), OrderSide::Sell), dec!(0.01));
        assert_eq!(clob_tick_price(dec!(0.00), OrderSide::Sell), dec!(0.01));
    }

    #[test]
    fn rejects_missing_or_underperforming_wallet_performance() {
        let config = CopyTradeConfig::default();
        let trade = trade("0xabc", "token", dec!(0.50), 0);
        let wallet_performance = performance("0xabc", dec!(20), dec!(0.10), 10);
        let decision = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            None,
            None,
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );
        assert_eq!(decision.copy_signal.status, "rejected");
        assert_eq!(
            decision.signal_candidate.reject_reason.as_deref(),
            Some("pnl_below_threshold")
        );

        let missing = evaluate_copy_trade(
            &trade,
            None,
            None,
            None,
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );
        assert_eq!(
            missing.signal_candidate.reject_reason.as_deref(),
            Some("missing_wallet_performance")
        );
    }

    #[test]
    fn rejects_roi_and_closed_position_thresholds() {
        let config = CopyTradeConfig::default();
        let trade = trade("0xabc", "token", dec!(0.50), 0);

        let low_roi = evaluate_copy_trade(
            &trade,
            Some(&performance("0xabc", dec!(1000), dec!(0.01), 10)),
            None,
            None,
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );
        assert_eq!(
            low_roi.signal_candidate.reject_reason.as_deref(),
            Some("roi_below_threshold")
        );

        let low_closed_positions = evaluate_copy_trade(
            &trade,
            Some(&performance("0xabc", dec!(1000), dec!(0.10), 1)),
            None,
            None,
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );
        assert_eq!(
            low_closed_positions
                .signal_candidate
                .reject_reason
                .as_deref(),
            Some("closed_positions_below_threshold")
        );
    }

    #[test]
    fn mrs_gate_rejects_only_when_enforced() {
        let mut config = CopyTradeConfig::default();
        config.mrs_enforce = true;
        config.min_mrs_score = dec!(80);
        let trade = trade("0xabc", "token", dec!(0.50), 0);
        let wallet_performance = performance("0xabc", dec!(1000), dec!(0.10), 10);
        let low_mrs = CopyTradeMrsScore {
            score: dec!(79.9999),
            score_version: "mrs_v1".to_string(),
            percentile: Some(dec!(0.79)),
            metadata: serde_json::json!({}),
        };
        let high_mrs = CopyTradeMrsScore {
            score: dec!(80),
            score_version: "mrs_v1".to_string(),
            percentile: Some(dec!(0.80)),
            metadata: serde_json::json!({}),
        };

        let low = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            Some(&low_mrs),
            None,
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );
        assert_eq!(
            low.signal_candidate.reject_reason.as_deref(),
            Some("mrs_score_below_threshold")
        );

        let high = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            Some(&high_mrs),
            None,
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );
        assert_eq!(high.copy_signal.status, "detected");
        assert_eq!(
            high.copy_signal.metadata["mrs"]["score"],
            serde_json::to_value(high_mrs.score).unwrap()
        );
        assert_eq!(
            high.copy_signal.metadata["wallet_score_basis"],
            serde_json::json!("mrs_v1")
        );
        assert_eq!(
            high.copy_signal.metadata["wallet_rank_score"],
            serde_json::to_value(wallet_performance.rank_score()).unwrap()
        );
        assert_ne!(
            high.copy_signal.metadata["mrs"]["score"],
            high.copy_signal.metadata["wallet_rank_score"]
        );

        config.mrs_enforce = false;
        let observe_only = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            Some(&low_mrs),
            None,
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );
        assert_eq!(observe_only.copy_signal.status, "detected");
    }

    #[test]
    fn segment_gate_records_shadow_reject_without_blocking() {
        let mut config = CopyTradeConfig::default();
        config.segment_scoring_enabled = true;
        config.segment_scoring_mode = "shadow".to_string();
        let mut trade = trade("0xabc", "token", dec!(0.50), 0);
        trade.title = Some("Will Bitcoin hit $120k?".to_string());
        let wallet_performance = performance("0xabc", dec!(1000), dec!(0.10), 10);
        let segment = segment_score("crypto", dec!(25));

        let decision = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            None,
            Some(&segment),
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );

        assert_eq!(decision.copy_signal.status, "detected");
        assert_eq!(
            decision.copy_signal.metadata["segment"]["decision"],
            serde_json::json!("would_reject")
        );
        assert_eq!(
            decision.copy_signal.metadata["segment"]["reject_reason"],
            serde_json::json!("segment_win_rate_hard_reject")
        );
    }

    #[test]
    fn segment_gate_rejects_when_enforced() {
        let mut config = CopyTradeConfig::default();
        config.segment_scoring_enabled = true;
        config.segment_scoring_mode = "sim_enforce".to_string();
        let mut trade = trade("0xabc", "token", dec!(0.50), 0);
        trade.title = Some("Will Bitcoin hit $120k?".to_string());
        let wallet_performance = performance("0xabc", dec!(1000), dec!(0.10), 10);
        let segment = segment_score("crypto", dec!(25));

        let decision = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            None,
            Some(&segment),
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );

        assert_eq!(decision.copy_signal.status, "rejected");
        assert_eq!(
            decision.signal_candidate.reject_reason.as_deref(),
            Some("segment_win_rate_hard_reject")
        );
    }

    #[test]
    fn segment_v2_gate_rejects_missing_segment_score_when_enforced() {
        let mut config = CopyTradeConfig::default();
        config.segment_scoring_enabled = true;
        config.segment_scoring_mode = "sim_enforce".to_string();
        config.segment_score_version = MRS_SEGMENT_V2_SCORE_VERSION.to_string();
        config.mrs_enforce = true;
        config.min_mrs_score = dec!(0);
        let mut trade = trade("0xabc", "token", dec!(0.50), 0);
        trade.title = Some("Will Bitcoin go up?".to_string());
        let wallet_performance = performance("0xabc", dec!(1000), dec!(0.10), 10);
        let mrs = CopyTradeMrsScore {
            score: dec!(100),
            score_version: "mrs_v1".to_string(),
            percentile: Some(dec!(1)),
            metadata: serde_json::json!({}),
        };

        let decision = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            Some(&mrs),
            None,
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );

        assert_eq!(decision.copy_signal.status, "rejected");
        assert_eq!(
            decision.signal_candidate.reject_reason.as_deref(),
            Some("missing_segment_score")
        );
    }

    #[test]
    fn segment_v2_gate_requires_segment_percentile() {
        let mut config = CopyTradeConfig::default();
        config.segment_scoring_enabled = true;
        config.segment_scoring_mode = "sim_enforce".to_string();
        config.segment_score_version = MRS_SEGMENT_V2_SCORE_VERSION.to_string();
        config.segment_mrs_percentile_floor = dec!(0.95);
        let mut trade = trade("0xabc", "token", dec!(0.50), 0);
        trade.title = Some("Will Bitcoin go up?".to_string());
        let wallet_performance = performance("0xabc", dec!(1000), dec!(0.10), 10);
        let mut segment = segment_score("crypto", dec!(95));
        segment.score_version = MRS_SEGMENT_V2_SCORE_VERSION.to_string();
        segment.win_rate = dec!(0.80);
        segment.roi = dec!(0.20);
        segment.realized_pnl_usd = dec!(500);
        segment.percentile = Some(dec!(0.90));

        let low = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            None,
            Some(&segment),
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );
        assert_eq!(
            low.signal_candidate.reject_reason.as_deref(),
            Some("segment_percentile_below_threshold")
        );

        segment.percentile = Some(dec!(0.95));
        let high = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            None,
            Some(&segment),
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );
        assert_eq!(high.copy_signal.status, "detected");
    }

    #[test]
    fn segment_allowlist_rejects_non_matching_segments_before_mrs_gate() {
        let mut config = CopyTradeConfig::default();
        config.segment_allowlist = vec!["crypto.bitcoin.short_interval".to_string()];
        config.mrs_enforce = true;
        let mut trade = trade("0xabc", "token", dec!(0.50), 0);
        trade.title = Some("Will the NBA Finals go seven games?".to_string());
        let wallet_performance = performance("0xabc", dec!(1000), dec!(0.10), 10);

        let decision = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            None,
            None,
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );

        assert_eq!(decision.copy_signal.status, "rejected");
        assert_eq!(
            decision.signal_candidate.reject_reason.as_deref(),
            Some("segment_not_allowlisted")
        );
        assert_eq!(
            decision.copy_signal.metadata["segment_filter"]["decision"],
            serde_json::json!("rejected")
        );
    }

    #[test]
    fn segment_allowlist_accepts_normalized_btc_short_interval_slug() {
        let mut config = CopyTradeConfig::default();
        config.segment_allowlist = vec!["series.btc-up-or-down-5m".to_string()];
        let mut trade = trade("0xabc", "token", dec!(0.50), 0);
        trade.event_slug = Some("btc-updown-5m-1779894000".to_string());
        let wallet_performance = performance("0xabc", dec!(1000), dec!(0.10), 10);

        let decision = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
            None,
            None,
            ObservedMarket {
                observed_price: dec!(0.50),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc,
            },
            &config,
            None,
        );

        assert_eq!(decision.copy_signal.status, "detected");
        assert_eq!(
            decision.copy_signal.metadata["segment"]["segment_key"],
            serde_json::json!("crypto.bitcoin.short_interval")
        );
        assert_eq!(
            decision.copy_signal.metadata["segment_filter"]["allowed_segments"],
            serde_json::json!(["crypto.bitcoin.short_interval"])
        );
    }

    #[test]
    fn backtest_uses_future_prices_and_returns_calibration_snapshot() {
        let mut config = CopyTradeConfig::default();
        config.backtest_horizon_secs = 60;
        config.taker_fee_rate = dec!(0);
        let trades = vec![
            trade("0xabc", "token", dec!(0.40), 0),
            trade("0xabc", "token", dec!(0.60), 2),
        ];
        let scores = vec![score("0xabc", dec!(80))];
        let (result, calibration) = run_copy_trade_backtest(&trades, &scores, &config);
        assert_eq!(result.signal_count, 1);
        assert!(result.net_pnl_usd > dec!(0));
        assert_eq!(calibration.wallet_count, 1);
    }
}
