use anyhow::Result;
use chrono::{Duration, Utc};
use futures_util::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

use crate::{
    clob::ClobClient,
    execution::{execute_order_plan, ExecutionVenue, OrderPlan},
    idempotency::{deterministic_client_order_id, ClientOrderIdSeed},
    models::{OrderRequest, OrderSide, OrderType},
    store::{OrderbookSnapshot, Store, WhaleLedTradeExitCandidate},
};

#[derive(Debug, Clone)]
pub struct TradePnlConfig {
    pub exit_candidate_max_age: Duration,
    pub mark_token_refresh_limit: usize,
    pub mark_refresh_concurrency: usize,
    pub max_orderbook_mark_age: Duration,
}

impl Default for TradePnlConfig {
    fn default() -> Self {
        Self {
            exit_candidate_max_age: Duration::seconds(900),
            mark_token_refresh_limit: 100,
            mark_refresh_concurrency: 8,
            max_orderbook_mark_age: Duration::minutes(15),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TradePnlRefreshReport {
    pub positions_backfilled: u64,
    pub positions_reconciled: u64,
    pub exits_applied: u64,
    pub exit_orders_submitted: u64,
    pub exit_fills_inserted: u64,
    pub closed_marks_cleared: u64,
    pub mark_orderbook_snapshots_inserted: u64,
    pub marks_written: u64,
    pub wallets_refreshed: u64,
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
    let mark_orderbook_snapshots_inserted =
        refresh_open_position_orderbook_marks(store, clob, config).await?;
    let marks_written = store.mark_open_trade_positions().await?;
    let wallets_refreshed = store.refresh_wallet_trade_performance().await?;
    Ok(TradePnlRefreshReport {
        positions_backfilled,
        positions_reconciled,
        exits_applied: exit_execution.exits_applied,
        exit_orders_submitted: exit_execution.exit_orders_submitted,
        exit_fills_inserted: exit_execution.exit_fills_inserted,
        closed_marks_cleared,
        mark_orderbook_snapshots_inserted,
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
    let mark_orderbook_snapshots_inserted =
        refresh_open_position_orderbook_marks(store, clob, config).await?;
    let marks_written = store.mark_open_trade_positions().await?;
    let wallets_refreshed = store.refresh_wallet_trade_performance().await?;
    Ok(TradePnlRefreshReport {
        positions_backfilled: 0,
        positions_reconciled,
        exits_applied: exit_execution.exits_applied,
        exit_orders_submitted: exit_execution.exit_orders_submitted,
        exit_fills_inserted: exit_execution.exit_fills_inserted,
        closed_marks_cleared,
        mark_orderbook_snapshots_inserted,
        marks_written,
        wallets_refreshed,
    })
}

async fn refresh_open_position_orderbook_marks(
    store: &Store,
    clob: Option<&ClobClient>,
    config: &TradePnlConfig,
) -> Result<u64> {
    let Some(clob) = clob else {
        return Ok(0);
    };
    let tokens = store
        .open_trade_position_mark_tokens(
            config.mark_token_refresh_limit as i64,
            config.max_orderbook_mark_age,
        )
        .await?;
    if tokens.is_empty() {
        return Ok(0);
    }

    let concurrency = config.mark_refresh_concurrency.max(1);
    let mut snapshots = stream::iter(tokens)
        .map(|token| async move {
            let token_id = token.token_id.clone();
            let book = clob.fetch_orderbook(&token_id).await?;
            let snapshot =
                OrderbookSnapshot::from_local_book(token.market_id, token.token_id, None, &book)?;
            store.insert_orderbook_snapshot(&snapshot).await?;
            Ok::<u64, anyhow::Error>(1)
        })
        .buffer_unordered(concurrency);

    let mut inserted = 0;
    let mut failed = 0;
    while let Some(result) = snapshots.next().await {
        match result {
            Ok(count) => inserted += count,
            Err(_) => failed += 1,
        }
    }
    if failed > 0 {
        warn!(
            failed,
            inserted, "some open position orderbook mark refreshes failed"
        );
    }
    Ok(inserted)
}

#[derive(Debug, Default)]
struct ExitExecutionReport {
    exits_applied: u64,
    exit_orders_submitted: u64,
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
        let execution = execute_order_plan(venue, order_plan).await?;
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

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    use crate::{
        models::OrderSide, store::WhaleLedTradeExitCandidate, trade_pnl::close_order_request,
    };

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
}
