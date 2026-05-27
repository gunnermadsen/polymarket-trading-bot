use anyhow::Result;
use chrono::{Duration, Utc};
use futures_util::{stream, StreamExt};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::time::Duration as StdDuration;
use tracing::warn;
use uuid::Uuid;

use crate::{
    clob::ClobClient,
    execution::{execute_order_plan, ExecutionVenue, OrderPlan},
    idempotency::{deterministic_client_order_id, ClientOrderIdSeed},
    models::{
        EffectiveStopLossExitRuleProcessConfig, EffectiveTakeProfitExitRuleProcessConfig,
        OrderRequest, OrderSide, OrderType,
    },
    store::{
        OrderbookSnapshot, Store, TakeProfitTradeExitCandidate, TradeMarkSourceFailure,
        WhaleLedTradeExitCandidate,
    },
};

#[derive(Debug, Clone)]
pub struct TradePnlConfig {
    pub exit_candidate_max_age: Duration,
    pub mark_token_refresh_limit: usize,
    pub mark_refresh_concurrency: usize,
    pub max_orderbook_mark_age: Duration,
    pub mark_failure_backoff: Duration,
    pub mark_fresh_max_age: Duration,
    pub mark_refresh_retry_attempts: usize,
    pub mark_refresh_retry_delay: StdDuration,
}

impl Default for TradePnlConfig {
    fn default() -> Self {
        Self {
            exit_candidate_max_age: Duration::seconds(900),
            mark_token_refresh_limit: 100,
            mark_refresh_concurrency: 8,
            max_orderbook_mark_age: Duration::minutes(15),
            mark_failure_backoff: Duration::minutes(5),
            mark_fresh_max_age: Duration::minutes(5),
            mark_refresh_retry_attempts: 2,
            mark_refresh_retry_delay: StdDuration::from_millis(250),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MarkOrderbookRefreshReport {
    pub snapshots_inserted: u64,
    pub failures: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TradePnlRefreshReport {
    pub positions_backfilled: u64,
    pub positions_reconciled: u64,
    pub exits_applied: u64,
    pub exit_orders_submitted: u64,
    pub exit_order_failures: u64,
    pub exit_fills_inserted: u64,
    pub closed_marks_cleared: u64,
    pub mark_orderbook_snapshots_inserted: u64,
    pub mark_orderbook_refresh_failures: u64,
    pub marks_written: u64,
    pub wallets_refreshed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TakeProfitExitExecutionReport {
    pub candidates_evaluated: u64,
    pub exits_applied: u64,
    pub exit_orders_submitted: u64,
    pub exit_order_failures: u64,
    pub exit_fills_inserted: u64,
    pub closed_marks_cleared: u64,
    pub mark_orderbook_snapshots_inserted: u64,
    pub mark_orderbook_refresh_failures: u64,
    pub marks_written: u64,
    pub positions_backfilled: u64,
    pub positions_reconciled: u64,
    pub wallets_refreshed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TakeProfitExitConfig {
    pub min_roi: Decimal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StopLossExitConfig {
    pub max_loss_roi: Decimal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TakeProfitPosition {
    pub position_id: Uuid,
    pub token_id: String,
    pub market_id: Option<String>,
    pub side: String,
    pub status: String,
    pub entry_price: Decimal,
    pub open_size: Decimal,
    pub entry_notional: Decimal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TakeProfitMark {
    pub mark_price: Decimal,
    pub mark_source: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TakeProfitExitDecision {
    pub position_id: Uuid,
    pub token_id: String,
    pub market_id: Option<String>,
    pub exit_type: &'static str,
    pub exit_price: Decimal,
    pub exit_size: Decimal,
    pub gross_pnl: Decimal,
    pub roi: Decimal,
    pub mark_source: String,
}

pub fn evaluate_take_profit_exit(
    position: &TakeProfitPosition,
    mark: &TakeProfitMark,
    config: &TakeProfitExitConfig,
) -> Option<TakeProfitExitDecision> {
    evaluate_roi_exit(
        position,
        mark,
        config.min_roi,
        "take_profit",
        |roi, threshold| threshold > Decimal::ZERO && roi >= threshold,
    )
}

pub fn evaluate_stop_loss_exit(
    position: &TakeProfitPosition,
    mark: &TakeProfitMark,
    config: &StopLossExitConfig,
) -> Option<TakeProfitExitDecision> {
    evaluate_roi_exit(
        position,
        mark,
        config.max_loss_roi,
        "stop_loss",
        |roi, threshold| threshold < Decimal::ZERO && roi <= threshold,
    )
}

fn evaluate_roi_exit(
    position: &TakeProfitPosition,
    mark: &TakeProfitMark,
    threshold_roi: Decimal,
    exit_type: &'static str,
    threshold_matches: impl Fn(Decimal, Decimal) -> bool,
) -> Option<TakeProfitExitDecision> {
    if !matches!(position.status.as_str(), "open" | "partially_closed") {
        return None;
    }
    if position.open_size <= Decimal::ZERO || position.entry_notional <= Decimal::ZERO {
        return None;
    }

    let gross_pnl = match position.side.as_str() {
        "buy" => position.open_size * (mark.mark_price - position.entry_price),
        "sell" => position.open_size * (position.entry_price - mark.mark_price),
        _ => return None,
    };
    let roi = gross_pnl / position.entry_notional;
    if !threshold_matches(roi, threshold_roi) {
        return None;
    }

    Some(TakeProfitExitDecision {
        position_id: position.position_id,
        token_id: position.token_id.clone(),
        market_id: position.market_id.clone(),
        exit_type,
        exit_price: mark.mark_price,
        exit_size: position.open_size,
        gross_pnl,
        roi,
        mark_source: mark.mark_source.clone(),
    })
}

pub async fn refresh_trade_pnl(
    store: &Store,
    venue: Option<&dyn ExecutionVenue>,
) -> Result<TradePnlRefreshReport> {
    refresh_trade_pnl_with_config(store, venue, None, &TradePnlConfig::default()).await
}

pub async fn refresh_trade_pnl_with_config(
    store: &Store,
    venue: Option<&dyn ExecutionVenue>,
    clob: Option<&ClobClient>,
    config: &TradePnlConfig,
) -> Result<TradePnlRefreshReport> {
    let positions_backfilled = store.backfill_trade_positions_from_copy_signals().await?;
    let positions_reconciled = store
        .reconcile_trade_positions_from_executable_exits()
        .await?;
    let exit_execution = execute_whale_led_trade_exits(store, venue, config).await?;
    let closed_marks_cleared = store.clear_closed_trade_position_unrealized_pnl().await?;
    let mark_orderbook_refresh = refresh_open_position_orderbook_marks(store, clob, config).await?;
    let marks_written = store.mark_open_trade_positions().await?;
    let wallets_refreshed = store.refresh_wallet_trade_performance().await?;
    Ok(TradePnlRefreshReport {
        positions_backfilled,
        positions_reconciled,
        exits_applied: exit_execution.exits_applied,
        exit_orders_submitted: exit_execution.exit_orders_submitted,
        exit_order_failures: exit_execution.exit_order_failures,
        exit_fills_inserted: exit_execution.exit_fills_inserted,
        closed_marks_cleared,
        mark_orderbook_snapshots_inserted: mark_orderbook_refresh.snapshots_inserted,
        mark_orderbook_refresh_failures: mark_orderbook_refresh.failures,
        marks_written,
        wallets_refreshed,
    })
}

pub async fn mark_trade_pnl_now(
    store: &Store,
    venue: Option<&dyn ExecutionVenue>,
) -> Result<TradePnlRefreshReport> {
    mark_trade_pnl_now_with_config(store, venue, None, &TradePnlConfig::default()).await
}

pub async fn mark_trade_pnl_now_with_config(
    store: &Store,
    venue: Option<&dyn ExecutionVenue>,
    clob: Option<&ClobClient>,
    config: &TradePnlConfig,
) -> Result<TradePnlRefreshReport> {
    let positions_reconciled = store
        .reconcile_trade_positions_from_executable_exits()
        .await?;
    let exit_execution = execute_whale_led_trade_exits(store, venue, config).await?;
    let closed_marks_cleared = store.clear_closed_trade_position_unrealized_pnl().await?;
    let mark_orderbook_refresh = refresh_open_position_orderbook_marks(store, clob, config).await?;
    let marks_written = store.mark_open_trade_positions().await?;
    let wallets_refreshed = store.refresh_wallet_trade_performance().await?;
    Ok(TradePnlRefreshReport {
        positions_backfilled: 0,
        positions_reconciled,
        exits_applied: exit_execution.exits_applied,
        exit_orders_submitted: exit_execution.exit_orders_submitted,
        exit_order_failures: exit_execution.exit_order_failures,
        exit_fills_inserted: exit_execution.exit_fills_inserted,
        closed_marks_cleared,
        mark_orderbook_snapshots_inserted: mark_orderbook_refresh.snapshots_inserted,
        mark_orderbook_refresh_failures: mark_orderbook_refresh.failures,
        marks_written,
        wallets_refreshed,
    })
}

pub async fn execute_take_profit_exits_for_process(
    store: &Store,
    venue: Option<&dyn ExecutionVenue>,
    clob: Option<&ClobClient>,
    process_id: Uuid,
    take_profit_config: &EffectiveTakeProfitExitRuleProcessConfig,
    pnl_config: &TradePnlConfig,
) -> Result<TakeProfitExitExecutionReport> {
    if !take_profit_config.take_profit_enabled {
        return Ok(TakeProfitExitExecutionReport::default());
    }
    let Some(venue) = venue else {
        return Ok(TakeProfitExitExecutionReport::default());
    };

    let positions_backfilled = store.backfill_trade_positions_from_copy_signals().await?;
    let positions_reconciled = store
        .reconcile_trade_positions_from_executable_exits()
        .await?;
    let mark_orderbook_refresh =
        refresh_open_position_orderbook_marks(store, clob, pnl_config).await?;
    let marks_written = store.mark_open_trade_positions().await?;

    let min_hold = Duration::seconds(take_profit_config.min_hold_secs.max(0));
    let require_fresh_mark = Duration::seconds(take_profit_config.require_fresh_mark_secs.max(0));
    let candidates = store
        .fetch_take_profit_trade_exit_candidates(
            process_id,
            take_profit_config.take_profit_roi,
            take_profit_config.exit_size_fraction,
            min_hold,
            require_fresh_mark,
            take_profit_config.max_exit_slippage_bps,
            100,
        )
        .await?;

    let mut report = TakeProfitExitExecutionReport {
        candidates_evaluated: candidates.len() as u64,
        positions_backfilled,
        positions_reconciled,
        mark_orderbook_snapshots_inserted: mark_orderbook_refresh.snapshots_inserted,
        mark_orderbook_refresh_failures: mark_orderbook_refresh.failures,
        marks_written,
        ..TakeProfitExitExecutionReport::default()
    };
    for candidate in candidates {
        let order_plan = OrderPlan {
            plan_id: Uuid::new_v4(),
            orders: vec![risk_control_order_request(&candidate)],
        };
        let execution = match execute_order_plan(venue, order_plan).await {
            Ok(execution) => execution,
            Err(error) => {
                report.exit_order_failures += 1;
                warn!(
                    error = %error,
                    position_id = %candidate.position_id,
                    trigger_roi = %candidate.trigger_roi,
                    take_profit_roi = %candidate.take_profit_roi,
                    "take-profit exit execution failed; continuing scheduler"
                );
                continue;
            }
        };
        report.exit_orders_submitted += execution.orders.len() as u64;
        report.exit_fills_inserted += execution.fills.len() as u64;
        store.persist_order_plan_report(&execution).await?;
        report.exits_applied += store
            .apply_take_profit_trade_exit(&candidate, &execution)
            .await?;
    }

    report.closed_marks_cleared = store.clear_closed_trade_position_unrealized_pnl().await?;
    report.wallets_refreshed = store.refresh_wallet_trade_performance().await?;
    Ok(report)
}

pub async fn execute_stop_loss_exits_for_process(
    store: &Store,
    venue: Option<&dyn ExecutionVenue>,
    clob: Option<&ClobClient>,
    process_id: Uuid,
    stop_loss_config: &EffectiveStopLossExitRuleProcessConfig,
    pnl_config: &TradePnlConfig,
) -> Result<TakeProfitExitExecutionReport> {
    if !stop_loss_config.stop_loss_enabled || stop_loss_config.stop_loss_roi >= Decimal::ZERO {
        return Ok(TakeProfitExitExecutionReport::default());
    }
    let Some(venue) = venue else {
        return Ok(TakeProfitExitExecutionReport::default());
    };

    let positions_backfilled = store.backfill_trade_positions_from_copy_signals().await?;
    let positions_reconciled = store
        .reconcile_trade_positions_from_executable_exits()
        .await?;
    let mark_orderbook_refresh =
        refresh_open_position_orderbook_marks(store, clob, pnl_config).await?;
    let marks_written = store.mark_open_trade_positions().await?;

    let min_hold = Duration::seconds(stop_loss_config.min_hold_secs.max(0));
    let require_fresh_mark = Duration::seconds(stop_loss_config.require_fresh_mark_secs.max(0));
    let candidates = store
        .fetch_stop_loss_trade_exit_candidates(
            process_id,
            stop_loss_config.stop_loss_roi,
            stop_loss_config.exit_size_fraction,
            min_hold,
            require_fresh_mark,
            stop_loss_config.max_exit_slippage_bps,
            100,
        )
        .await?;

    let mut report = TakeProfitExitExecutionReport {
        candidates_evaluated: candidates.len() as u64,
        positions_backfilled,
        positions_reconciled,
        mark_orderbook_snapshots_inserted: mark_orderbook_refresh.snapshots_inserted,
        mark_orderbook_refresh_failures: mark_orderbook_refresh.failures,
        marks_written,
        ..TakeProfitExitExecutionReport::default()
    };
    for candidate in candidates {
        let order_plan = OrderPlan {
            plan_id: Uuid::new_v4(),
            orders: vec![risk_control_order_request(&candidate)],
        };
        let execution = match execute_order_plan(venue, order_plan).await {
            Ok(execution) => execution,
            Err(error) => {
                report.exit_order_failures += 1;
                warn!(
                    error = %error,
                    position_id = %candidate.position_id,
                    trigger_roi = %candidate.trigger_roi,
                    stop_loss_roi = %candidate.threshold_roi,
                    "stop-loss exit execution failed; continuing scheduler"
                );
                continue;
            }
        };
        report.exit_orders_submitted += execution.orders.len() as u64;
        report.exit_fills_inserted += execution.fills.len() as u64;
        store.persist_order_plan_report(&execution).await?;
        report.exits_applied += store
            .apply_take_profit_trade_exit(&candidate, &execution)
            .await?;
    }

    report.closed_marks_cleared = store.clear_closed_trade_position_unrealized_pnl().await?;
    report.wallets_refreshed = store.refresh_wallet_trade_performance().await?;
    Ok(report)
}

async fn refresh_open_position_orderbook_marks(
    store: &Store,
    clob: Option<&ClobClient>,
    config: &TradePnlConfig,
) -> Result<MarkOrderbookRefreshReport> {
    let Some(clob) = clob else {
        return Ok(MarkOrderbookRefreshReport::default());
    };
    let tokens = store
        .open_trade_position_mark_tokens(
            config.mark_token_refresh_limit as i64,
            config.max_orderbook_mark_age,
            config.mark_failure_backoff,
        )
        .await?;
    if tokens.is_empty() {
        return Ok(MarkOrderbookRefreshReport::default());
    }

    let concurrency = config.mark_refresh_concurrency.max(1);
    let mut snapshots = stream::iter(tokens)
        .map(|token| async move {
            let token_id = token.token_id.clone();
            let result = async {
                let book = fetch_orderbook_with_retry(
                    clob,
                    &token_id,
                    config.mark_refresh_retry_attempts,
                    config.mark_refresh_retry_delay,
                )
                .await?;
                let snapshot = OrderbookSnapshot::from_local_book(
                    token.market_id.clone(),
                    token.token_id.clone(),
                    None,
                    &book,
                )?;
                store.insert_orderbook_snapshot(&snapshot).await?;
                store
                    .resolve_trade_mark_source_failure(
                        token.process_id,
                        None,
                        &token.token_id,
                        "mark_orderbook_refresh",
                    )
                    .await?;
                Ok::<u64, anyhow::Error>(1)
            }
            .await;
            (token, result)
        })
        .buffer_unordered(concurrency);

    let mut inserted = 0;
    let mut failed = 0;
    while let Some((token, result)) = snapshots.next().await {
        match result {
            Ok(count) => inserted += count,
            Err(error) => {
                failed += 1;
                store
                    .upsert_trade_mark_source_failure(&TradeMarkSourceFailure {
                        process_id: token.process_id,
                        position_id: None,
                        token_id: token.token_id,
                        market_id: token.market_id,
                        failure_source: "mark_orderbook_refresh".to_string(),
                        failure_reason: mark_source_failure_reason(&error).to_string(),
                        metadata: serde_json::json!({
                            "operation": "orderbook_mark_refresh",
                            "error": error.to_string(),
                            "error_chain": format!("{error:#}"),
                            "attempted_at": Utc::now()
                        }),
                    })
                    .await?;
            }
        }
    }
    if failed > 0 {
        warn!(
            failed,
            inserted, "some open position orderbook mark refreshes failed"
        );
    }
    Ok(MarkOrderbookRefreshReport {
        snapshots_inserted: inserted,
        failures: failed,
    })
}

async fn fetch_orderbook_with_retry(
    clob: &ClobClient,
    token_id: &str,
    retry_attempts: usize,
    retry_delay: StdDuration,
) -> Result<crate::orderbook::LocalOrderBook> {
    let attempts = retry_attempts.max(1);
    let mut last_error = None;
    for attempt in 1..=attempts {
        match clob.fetch_orderbook(token_id).await {
            Ok(book) => return Ok(book),
            Err(error) => {
                let should_retry =
                    mark_source_failure_reason(&error) == "clob_orderbook_request_failed";
                last_error = Some(error);
                if !should_retry || attempt == attempts {
                    break;
                }
                tokio::time::sleep(retry_delay).await;
            }
        }
    }
    Err(last_error.expect("orderbook fetch attempts are always at least one"))
}

fn mark_source_failure_reason(error: &anyhow::Error) -> &'static str {
    let chain = format!("{error:#}");
    if chain.contains("CLOB book response was not successful") {
        "clob_orderbook_unavailable"
    } else if chain.contains("failed to request CLOB book") {
        "clob_orderbook_request_failed"
    } else if chain.contains("failed to decode CLOB book") {
        "clob_orderbook_decode_failed"
    } else {
        "orderbook_mark_source_failed"
    }
}

#[derive(Debug, Default)]
struct ExitExecutionReport {
    exits_applied: u64,
    exit_orders_submitted: u64,
    exit_order_failures: u64,
    exit_fills_inserted: u64,
}

async fn execute_whale_led_trade_exits(
    store: &Store,
    venue: Option<&dyn ExecutionVenue>,
    config: &TradePnlConfig,
) -> Result<ExitExecutionReport> {
    let Some(venue) = venue else {
        return Ok(ExitExecutionReport::default());
    };
    let mut report = ExitExecutionReport::default();
    for candidate in store
        .fetch_whale_led_trade_exit_candidates(100, config.exit_candidate_max_age)
        .await?
    {
        let order_plan = OrderPlan {
            plan_id: Uuid::new_v4(),
            orders: vec![close_order_request(&candidate)],
        };
        let execution = match execute_order_plan(venue, order_plan).await {
            Ok(execution) => execution,
            Err(error) => {
                report.exit_order_failures += 1;
                warn!(
                    error = %error,
                    position_id = %candidate.position_id,
                    exit_source_trade_id = %candidate.exit_source_trade_id,
                    "whale-led exit execution failed; continuing trade PnL refresh"
                );
                continue;
            }
        };
        report.exit_orders_submitted += execution.orders.len() as u64;
        report.exit_fills_inserted += execution.fills.len() as u64;
        store.persist_order_plan_report(&execution).await?;
        report.exits_applied += store
            .apply_executable_trade_exit(&candidate, &execution)
            .await?;
    }
    Ok(report)
}

fn close_order_request(candidate: &WhaleLedTradeExitCandidate) -> OrderRequest {
    let side = if candidate.side == "buy" {
        OrderSide::Sell
    } else {
        OrderSide::Buy
    };
    let candidate_age_seconds = (Utc::now() - candidate.exit_timestamp).num_seconds().max(0);
    OrderRequest {
        client_order_id: deterministic_client_order_id(&ClientOrderIdSeed {
            strategy_version: "whale-follow-v1",
            process_id: candidate.process_id,
            source_id: candidate.position_id,
            purpose: "whale_led_exit",
            market_id: candidate.market_id.as_deref().unwrap_or("unknown"),
            token_id: &candidate.token_id,
            side,
            notional_key: &(candidate.reference_exit_price * candidate.exit_size)
                .round_dp(4)
                .normalize()
                .to_string(),
        }),
        process_id: candidate.process_id,
        market_id: candidate
            .market_id
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        token_id: candidate.token_id.clone(),
        side,
        order_type: OrderType::Fok,
        price: candidate.reference_exit_price,
        size: candidate.exit_size,
        signal_id: None,
        metadata: serde_json::json!({
            "purpose": "whale_led_exit",
            "position_id": candidate.position_id,
            "source_signal_id": candidate.source_signal_id,
            "process_id": candidate.process_id,
            "exit_source_trade_id": candidate.exit_source_trade_id,
            "reference_exit_price": candidate.reference_exit_price,
            "reference_exit_timestamp": candidate.exit_timestamp,
            "candidate_age_seconds": candidate_age_seconds,
            "reference_exit_size": candidate.exit_size
        }),
    }
}

fn risk_control_order_request(candidate: &TakeProfitTradeExitCandidate) -> OrderRequest {
    let side = if candidate.side == "buy" {
        OrderSide::Sell
    } else {
        OrderSide::Buy
    };
    OrderRequest {
        client_order_id: deterministic_client_order_id(&ClientOrderIdSeed {
            strategy_version: "whale-follow-v1",
            process_id: candidate.process_id,
            source_id: candidate.position_id,
            purpose: &candidate.exit_purpose,
            market_id: candidate.market_id.as_deref().unwrap_or("unknown"),
            token_id: &candidate.token_id,
            side,
            notional_key: &(candidate.order_limit_price * candidate.exit_size)
                .round_dp(4)
                .normalize()
                .to_string(),
        }),
        process_id: candidate.process_id,
        market_id: candidate
            .market_id
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        token_id: candidate.token_id.clone(),
        side,
        order_type: OrderType::Fok,
        price: candidate.order_limit_price,
        size: candidate.exit_size,
        signal_id: None,
        metadata: serde_json::json!({
            "execution_intent": "exit",
            "purpose": candidate.exit_purpose.as_str(),
            "position_id": candidate.position_id,
            "source_signal_id": candidate.source_signal_id,
            "process_id": candidate.process_id,
            "exit_source_trade_id": candidate.exit_source_trade_id,
            "reference_exit_price": candidate.reference_exit_price,
            "order_limit_price": candidate.order_limit_price,
            "reference_exit_timestamp": candidate.exit_timestamp,
            "latest_mark_timestamp": candidate.latest_mark_timestamp,
            "trigger_roi": candidate.trigger_roi,
            "threshold_roi": candidate.threshold_roi,
            "take_profit_roi": candidate.take_profit_roi,
            "stop_loss_roi": if candidate.exit_purpose == "stop_loss_exit" {
                Some(candidate.threshold_roi)
            } else {
                None
            },
            "max_exit_slippage_bps": candidate.max_exit_slippage_bps,
            "reference_exit_size": candidate.exit_size
        }),
    }
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use chrono::Utc;
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    use crate::{
        models::OrderSide,
        store::{TakeProfitTradeExitCandidate, WhaleLedTradeExitCandidate},
        trade_pnl::{
            close_order_request, evaluate_stop_loss_exit, evaluate_take_profit_exit,
            mark_source_failure_reason, risk_control_order_request, StopLossExitConfig,
            TakeProfitExitConfig, TakeProfitMark, TakeProfitPosition,
        },
    };

    fn take_profit_position(side: &str) -> TakeProfitPosition {
        TakeProfitPosition {
            position_id: Uuid::new_v4(),
            token_id: "token-1".to_string(),
            market_id: Some("market-1".to_string()),
            side: side.to_string(),
            status: "open".to_string(),
            entry_price: dec!(0.40),
            open_size: dec!(25),
            entry_notional: dec!(10),
        }
    }

    fn take_profit_candidate(side: &str) -> TakeProfitTradeExitCandidate {
        TakeProfitTradeExitCandidate {
            exit_purpose: "take_profit_exit".to_string(),
            process_id: Some(Uuid::new_v4()),
            position_id: Uuid::new_v4(),
            source_signal_id: Uuid::new_v4(),
            proxy_wallet: Some("0xabc".to_string()),
            market_id: Some("market-1".to_string()),
            token_id: "token-1".to_string(),
            side: side.to_string(),
            entry_price: dec!(0.40),
            entry_size: dec!(25),
            open_size: dec!(25),
            entry_fee: Decimal::ZERO,
            entry_notional: dec!(10),
            exit_source_trade_id: Uuid::new_v4(),
            exit_timestamp: Utc::now(),
            reference_exit_price: dec!(0.44),
            order_limit_price: dec!(0.4334),
            exit_size: dec!(25),
            trigger_roi: dec!(0.10),
            threshold_roi: dec!(0.10),
            take_profit_roi: dec!(0.10),
            latest_mark_timestamp: Utc::now(),
            max_exit_slippage_bps: dec!(150),
        }
    }

    #[test]
    fn close_order_uses_opposite_side_and_reference_metadata() {
        let position_id = Uuid::new_v4();
        let process_id = Uuid::new_v4();
        let source_signal_id = Uuid::new_v4();
        let exit_source_trade_id = Uuid::new_v4();
        let candidate = WhaleLedTradeExitCandidate {
            process_id: Some(process_id),
            position_id,
            source_signal_id,
            proxy_wallet: Some("0xabc".to_string()),
            market_id: Some("market-1".to_string()),
            token_id: "token-1".to_string(),
            side: "buy".to_string(),
            entry_price: dec!(0.40),
            entry_size: dec!(25),
            open_size: dec!(25),
            entry_fee: dec!(0.01),
            entry_notional: dec!(10),
            exit_source_trade_id,
            exit_timestamp: Utc::now(),
            reference_exit_price: dec!(0.50),
            exit_size: dec!(20),
        };

        let request = close_order_request(&candidate);

        assert_eq!(request.side, OrderSide::Sell);
        assert_eq!(request.signal_id, None);
        assert_eq!(request.process_id, Some(process_id));
        assert_eq!(request.price, dec!(0.50));
        assert_eq!(request.size, dec!(20));
        assert_eq!(request.metadata["purpose"], "whale_led_exit");
        assert_eq!(request.metadata["position_id"], position_id.to_string());
        assert_eq!(request.metadata["process_id"], process_id.to_string());
        assert_eq!(
            request.metadata["exit_source_trade_id"],
            exit_source_trade_id.to_string()
        );
    }

    #[test]
    fn take_profit_order_uses_exit_intent_and_position_size() {
        let candidate = take_profit_candidate("buy");

        let request = risk_control_order_request(&candidate);

        assert_eq!(request.side, OrderSide::Sell);
        assert_eq!(request.price, candidate.order_limit_price);
        assert_eq!(request.size, candidate.exit_size);
        assert_eq!(
            request.metadata["execution_intent"],
            serde_json::json!("exit")
        );
        assert_eq!(
            request.metadata["purpose"],
            serde_json::json!("take_profit_exit")
        );
        assert_eq!(
            request.metadata["position_id"],
            serde_json::json!(candidate.position_id)
        );
        assert_eq!(
            request.metadata["trigger_roi"],
            serde_json::json!(candidate.trigger_roi)
        );
    }

    #[test]
    fn stop_loss_order_uses_exit_intent_and_threshold_metadata() {
        let mut candidate = take_profit_candidate("buy");
        candidate.exit_purpose = "stop_loss_exit".to_string();
        candidate.trigger_roi = dec!(-0.10);
        candidate.threshold_roi = dec!(-0.10);
        candidate.take_profit_roi = dec!(-0.10);

        let request = risk_control_order_request(&candidate);

        assert_eq!(request.side, OrderSide::Sell);
        assert_eq!(
            request.metadata["execution_intent"],
            serde_json::json!("exit")
        );
        assert_eq!(
            request.metadata["purpose"],
            serde_json::json!("stop_loss_exit")
        );
        assert_eq!(
            request.metadata["threshold_roi"],
            serde_json::json!(dec!(-0.10))
        );
        assert_eq!(
            request.metadata["stop_loss_roi"],
            serde_json::json!(dec!(-0.10))
        );
    }

    #[test]
    fn mark_source_failure_reason_classifies_clob_errors() {
        assert_eq!(
            mark_source_failure_reason(&anyhow!("CLOB book response was not successful")),
            "clob_orderbook_unavailable"
        );
        assert_eq!(
            mark_source_failure_reason(&anyhow!("failed to request CLOB book for token abc")),
            "clob_orderbook_request_failed"
        );
        assert_eq!(
            mark_source_failure_reason(&anyhow!("failed to decode CLOB book")),
            "clob_orderbook_decode_failed"
        );
    }

    #[test]
    fn take_profit_exit_triggers_for_buy_at_configured_roi_threshold() {
        let position = take_profit_position("buy");
        let mark = TakeProfitMark {
            mark_price: dec!(0.44),
            mark_source: "clob_mid".to_string(),
        };
        let decision = evaluate_take_profit_exit(
            &position,
            &mark,
            &TakeProfitExitConfig {
                min_roi: dec!(0.10),
            },
        )
        .expect("buy position should hit take-profit threshold");

        assert_eq!(decision.position_id, position.position_id);
        assert_eq!(decision.token_id, position.token_id);
        assert_eq!(decision.market_id, position.market_id);
        assert_eq!(decision.exit_type, "take_profit");
        assert_eq!(decision.exit_price, dec!(0.44));
        assert_eq!(decision.exit_size, dec!(25));
        assert_eq!(decision.gross_pnl, dec!(1.00));
        assert_eq!(decision.roi, dec!(0.10));
        assert_eq!(decision.mark_source, "clob_mid");
    }

    #[test]
    fn take_profit_exit_holds_when_buy_roi_is_below_threshold() {
        let position = take_profit_position("buy");
        let mark = TakeProfitMark {
            mark_price: dec!(0.43),
            mark_source: "clob_mid".to_string(),
        };

        assert_eq!(
            evaluate_take_profit_exit(
                &position,
                &mark,
                &TakeProfitExitConfig {
                    min_roi: dec!(0.10)
                },
            ),
            None
        );
    }

    #[test]
    fn stop_loss_exit_triggers_for_buy_at_configured_loss_threshold() {
        let position = take_profit_position("buy");
        let mark = TakeProfitMark {
            mark_price: dec!(0.36),
            mark_source: "clob_mid".to_string(),
        };
        let decision = evaluate_stop_loss_exit(
            &position,
            &mark,
            &StopLossExitConfig {
                max_loss_roi: dec!(-0.10),
            },
        )
        .expect("buy position should hit stop-loss threshold");

        assert_eq!(decision.position_id, position.position_id);
        assert_eq!(decision.exit_type, "stop_loss");
        assert_eq!(decision.exit_price, dec!(0.36));
        assert_eq!(decision.gross_pnl, dec!(-1.00));
        assert_eq!(decision.roi, dec!(-0.10));
    }

    #[test]
    fn stop_loss_exit_holds_when_loss_is_above_threshold() {
        let position = take_profit_position("buy");
        let mark = TakeProfitMark {
            mark_price: dec!(0.37),
            mark_source: "clob_mid".to_string(),
        };

        assert_eq!(
            evaluate_stop_loss_exit(
                &position,
                &mark,
                &StopLossExitConfig {
                    max_loss_roi: dec!(-0.10)
                },
            ),
            None
        );
    }

    #[test]
    fn take_profit_exit_uses_inverse_mark_move_for_sell_positions() {
        let position = take_profit_position("sell");
        let mark = TakeProfitMark {
            mark_price: dec!(0.36),
            mark_source: "clob_mid".to_string(),
        };
        let decision = evaluate_take_profit_exit(
            &position,
            &mark,
            &TakeProfitExitConfig {
                min_roi: dec!(0.10),
            },
        )
        .expect("sell position should hit take-profit threshold when mark falls");

        assert_eq!(decision.exit_price, dec!(0.36));
        assert_eq!(decision.gross_pnl, dec!(1.00));
        assert_eq!(decision.roi, dec!(0.10));
    }

    #[test]
    fn stop_loss_exit_uses_inverse_mark_move_for_sell_positions() {
        let position = take_profit_position("sell");
        let mark = TakeProfitMark {
            mark_price: dec!(0.44),
            mark_source: "clob_mid".to_string(),
        };
        let decision = evaluate_stop_loss_exit(
            &position,
            &mark,
            &StopLossExitConfig {
                max_loss_roi: dec!(-0.10),
            },
        )
        .expect("sell position should hit stop-loss threshold when mark rises");

        assert_eq!(decision.exit_price, dec!(0.44));
        assert_eq!(decision.gross_pnl, dec!(-1.00));
        assert_eq!(decision.roi, dec!(-0.10));
    }

    #[test]
    fn take_profit_exit_ignores_positions_that_are_not_open() {
        let mut position = take_profit_position("buy");
        position.status = "closed".to_string();
        let mark = TakeProfitMark {
            mark_price: dec!(0.60),
            mark_source: "clob_mid".to_string(),
        };

        assert_eq!(
            evaluate_take_profit_exit(
                &position,
                &mark,
                &TakeProfitExitConfig {
                    min_roi: dec!(0.10)
                },
            ),
            None
        );
    }

    #[test]
    fn take_profit_exit_ignores_zero_open_size_and_disabled_threshold() {
        let mut position = take_profit_position("buy");
        let mark = TakeProfitMark {
            mark_price: dec!(0.60),
            mark_source: "clob_mid".to_string(),
        };

        position.open_size = Decimal::ZERO;
        assert_eq!(
            evaluate_take_profit_exit(
                &position,
                &mark,
                &TakeProfitExitConfig {
                    min_roi: dec!(0.10)
                },
            ),
            None
        );

        position.open_size = dec!(25);
        assert_eq!(
            evaluate_take_profit_exit(
                &position,
                &mark,
                &TakeProfitExitConfig {
                    min_roi: Decimal::ZERO
                },
            ),
            None
        );
    }
}
