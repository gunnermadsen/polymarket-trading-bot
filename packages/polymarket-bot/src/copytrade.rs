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
        WalletScore, WalletScoreCalibrationSnapshot, WhaleTrade,
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
    pub copy_size_fraction: Decimal,
    pub max_follow_lag_secs: i64,
    pub max_price_slippage_bps: Decimal,
    pub min_book_depth_usd: Decimal,
    pub backtest_horizon_secs: i64,
    pub taker_fee_rate: Decimal,
    pub allow_sell_entries: bool,
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
            copy_size_fraction: dec!(0.02),
            max_follow_lag_secs: 300,
            max_price_slippage_bps: dec!(150),
            min_book_depth_usd: dec!(25),
            backtest_horizon_secs: 3600,
            taker_fee_rate: dec!(0.03),
            allow_sell_entries: false,
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
            copy_size_fraction: config.copy_size_fraction,
            max_follow_lag_secs: config.max_follow_lag_secs,
            max_price_slippage_bps: config.max_price_slippage_bps,
            min_book_depth_usd: config.min_book_depth_usd,
            backtest_horizon_secs: config.backtest_horizon_secs,
            taker_fee_rate: config.taker_fee_rate,
            allow_sell_entries: config.allow_sell_entries,
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

impl CopyTradeWalletPerformance {
    pub fn rank_score(&self) -> Decimal {
        self.realized_pnl_usd * self.roi
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

    if !config.enabled {
        reject_reason = Some("copy_trade_disabled");
    } else if token_id.is_none() {
        reject_reason = Some("missing_token_id");
    } else if !matches!(side.as_str(), "BUY" | "SELL") {
        reject_reason = Some("unsupported_trade_side");
    } else if side == "SELL" && !config.allow_sell_entries {
        reject_reason = Some("sell_entries_disabled");
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
    }

    let detected = reject_reason.is_none();
    let reason = reject_reason.unwrap_or("qualified_profitable_whale_follow");
    let status = if detected { "detected" } else { "rejected" }.to_string();
    let metadata = serde_json::json!({
        "trade_cash_value": trade.cash_value,
        "wallet_rank_score": rank_score,
        "wallet_realized_pnl_usd": performance.map(|performance| performance.realized_pnl_usd),
        "wallet_roi": performance.map(|performance| performance.roi),
        "wallet_closed_positions": performance.map(|performance| performance.closed_positions),
        "wallet_score_diagnostic": performance.and_then(|performance| performance.wallet_score_diagnostic),
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
        wallet_score: rank_score,
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
                    size: shares_for_notional(copy_size_usd, limit_price),
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
        let side_multiplier = if trade.side.eq_ignore_ascii_case("SELL") {
            dec!(-1)
        } else {
            dec!(1)
        };
        let shares = shares_for_notional(decision.copy_signal.copy_size_usd, trade.price);
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

fn shares_for_notional(notional: Decimal, price: Decimal) -> Decimal {
    if price <= Decimal::ZERO {
        return Decimal::ZERO;
    }
    (notional / price).round_dp_with_strategy(2, RoundingStrategy::ToZero)
}

fn clob_tick_price(price: Decimal, side: OrderSide) -> Decimal {
    match side {
        OrderSide::Buy => price.round_dp_with_strategy(2, RoundingStrategy::ToPositiveInfinity),
        OrderSide::Sell => price.round_dp_with_strategy(2, RoundingStrategy::ToNegativeInfinity),
    }
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
            evaluate_copy_trade, run_copy_trade_backtest, CopyTradeConfig,
            CopyTradeWalletPerformance, ObservedMarket,
        },
        models::{WalletScore, WhaleTrade},
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

    #[test]
    fn evaluates_qualified_whale_trade_into_order_plan() {
        let config = CopyTradeConfig::default();
        let trade = trade("0xabc", "token", dec!(0.50), 0);
        let wallet_performance = performance("0xabc", dec!(1000), dec!(0.10), 10);
        let decision = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
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
            ObservedMarket {
                observed_price: dec!(0.37),
                available_depth_usd: dec!(1000),
                observed_at: trade.timestamp_utc + chrono::Duration::seconds(10),
            },
            &config,
            None,
        );
        let order = &decision.order_plan.unwrap().orders[0];
        assert_eq!(order.size, dec!(5.40));
        assert!(order.price * order.size <= dec!(2));
    }

    #[test]
    fn rejects_missing_or_underperforming_wallet_performance() {
        let config = CopyTradeConfig::default();
        let trade = trade("0xabc", "token", dec!(0.50), 0);
        let wallet_performance = performance("0xabc", dec!(20), dec!(0.10), 10);
        let decision = evaluate_copy_trade(
            &trade,
            Some(&wallet_performance),
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
