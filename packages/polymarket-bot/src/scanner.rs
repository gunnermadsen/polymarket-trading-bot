use anyhow::{Context, Result};
use chrono::{Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use uuid::Uuid;

use crate::{
    clob::ClobClient,
    edge::{
        cheap_basket_edge, cheap_basket_worst_case_loss, compute_taker_fee, dynamic_threshold,
        leg_risk_reserve, Reserves, ThresholdState,
    },
    execution::{execute_order_plan, ExecutionVenue, OrderPlan},
    idempotency::{deterministic_client_order_id, ClientOrderIdSeed},
    models::{
        Market, OrderRequest, OrderSide, OrderType, OutcomeToken, SignalCandidate, SignalStatus,
        SignalType, TokenSide,
    },
    orderbook::{DepthWalk, LocalOrderBook},
    risk::{per_event_loss_cap, remaining_daily_loss_budget, RiskLimits, RiskState},
    store::{FunnelEvent, OrderbookSnapshot, PositionSnapshot, Store},
};

#[derive(Debug, Clone)]
pub struct ScannerConfig {
    pub target_size: Decimal,
    pub taker_fee_rate: Decimal,
    pub bootstrap_threshold: Decimal,
    pub max_quote_age: Duration,
    pub min_event_markets: usize,
}

#[derive(Debug, Default, Clone)]
pub struct ScannerCycleReport {
    pub events_considered: usize,
    pub tokens_persisted: usize,
    pub books_persisted: usize,
    pub signals_inserted: usize,
    pub positive_signals: usize,
    pub orders_inserted: usize,
    pub fills_inserted: usize,
    pub positions_upserted: usize,
    pub errors: usize,
}

pub async fn scan_markets_for_signal1(
    markets: &[Market],
    clob: &ClobClient,
    store: &Store,
    venue: &dyn ExecutionVenue,
    config: &ScannerConfig,
    risk_limits: &RiskLimits,
    risk_state: &RiskState,
) -> ScannerCycleReport {
    let mut report = ScannerCycleReport::default();
    for market in markets
        .iter()
        .filter(|m| m.active && !m.closed && !m.archived)
    {
        report.events_considered += 1;

        match scan_market_for_binary_cheap_basket(
            market,
            clob,
            store,
            venue,
            config,
            risk_limits,
            risk_state,
        )
        .await
        {
            Ok(market_report) => report.merge(market_report),
            Err(error) => {
                report.errors += 1;
                let _ = store
                    .insert_funnel_event(&FunnelEvent::new(
                        None,
                        Some(market.market_id.clone()),
                        "market_scan",
                        "error",
                        Some(error.to_string()),
                        serde_json::json!({}),
                    ))
                    .await;
            }
        }
    }

    report
}

async fn scan_market_for_binary_cheap_basket(
    market: &Market,
    clob: &ClobClient,
    store: &Store,
    venue: &dyn ExecutionVenue,
    config: &ScannerConfig,
    risk_limits: &RiskLimits,
    risk_state: &RiskState,
) -> Result<ScannerCycleReport> {
    let mut report = ScannerCycleReport::default();
    let mut legs = Vec::new();
    let now = Utc::now();

    let size = config.target_size * risk_state.fill_confidence_discount;

    let mut basket_tokens: Vec<&OutcomeToken> = market.outcome_tokens.iter().collect();
    basket_tokens.sort_by_key(|token| match token.side {
        TokenSide::Yes => 0,
        TokenSide::No => 1,
    });

    let has_yes = basket_tokens
        .iter()
        .any(|token| token.side == TokenSide::Yes);
    let has_no = basket_tokens
        .iter()
        .any(|token| token.side == TokenSide::No);
    if !has_yes || !has_no {
        insert_reject(
            store,
            market,
            "incomplete_binary_token_set",
            Decimal::ZERO,
            config.bootstrap_threshold,
            size,
            None,
        )
        .await?;
        report.signals_inserted += 1;
        return Ok(report);
    }

    for token in basket_tokens {
        store.upsert_outcome_token(token).await?;
        report.tokens_persisted += 1;
        let book = clob
            .fetch_orderbook(&token.token_id)
            .await
            .with_context(|| format!("failed to fetch book for {}", token.token_id))?;
        let snapshot = OrderbookSnapshot::from_local_book(
            Some(market.market_id.clone()),
            token.token_id.clone(),
            Some(token.tick_size),
            &book,
        )?;
        store.insert_orderbook_snapshot(&snapshot).await?;
        report.books_persisted += 1;

        let Some(walk) = book.depth_walk_buy(size, config.max_quote_age, now) else {
            insert_reject(
                store,
                market,
                "insufficient_fresh_ask_depth",
                Decimal::ZERO,
                config.bootstrap_threshold,
                size,
                None,
            )
            .await?;
            report.signals_inserted += 1;
            continue;
        };
        legs.push((market, token, book, walk));
    }

    if legs.len() != market.outcome_tokens.len() || legs.len() < config.min_event_markets {
        return Ok(report);
    }

    let buy_cost: Decimal = legs.iter().map(|(_, _, _, walk)| walk.total).sum();
    let fees: Decimal = legs
        .iter()
        .map(|(_, _, _, walk)| compute_taker_fee(&walk.fills, config.taker_fee_rate))
        .sum();
    let tick_size = legs
        .iter()
        .map(|(_, token, _, _)| token.tick_size)
        .min()
        .unwrap_or(dec!(0.01));
    let threshold = dynamic_threshold(&ThresholdState {
        bootstrap_threshold: config.bootstrap_threshold,
        realized_slippage: Vec::new(),
        tick_size,
    });
    let reserves = Reserves {
        threshold,
        carry_reserve: Decimal::ZERO,
        leg_risk_reserve: leg_risk_reserve(size, Decimal::ZERO),
        conversion_latency_reserve: Decimal::ZERO,
    };
    let edge = cheap_basket_edge(size, buy_cost, fees, reserves);
    let cap =
        per_event_loss_cap(risk_limits).min(remaining_daily_loss_budget(risk_limits, risk_state));
    let immediate_unwind = immediate_sell_revenue(&legs, size, config.max_quote_age, now);
    let worst_case = cheap_basket_worst_case_loss(buy_cost, fees, immediate_unwind, cap);

    let signal = SignalCandidate {
        signal_id: Uuid::new_v4(),
        signal_type: SignalType::CheapBasket,
        market_id: market.market_id.clone(),
        expected_edge: edge.net,
        threshold,
        size,
        status: if edge.net > Decimal::ZERO && worst_case.passes() {
            SignalStatus::Detected
        } else {
            SignalStatus::Rejected
        },
        reject_reason: if edge.net <= Decimal::ZERO {
            Some("edge_below_threshold".to_string())
        } else if !worst_case.passes() {
            Some("worst_case_loss_exceeds_cap".to_string())
        } else {
            None
        },
        worst_case_loss: Some(worst_case.amount),
        metadata: serde_json::json!({
            "event_id": market.event_id,
            "outcome_group_id": market.outcome_group_id,
            "gross": edge.gross,
            "fees": edge.fees,
            "reserves": edge.reserves,
            "buy_cost": buy_cost,
            "immediate_unwind_revenue": immediate_unwind,
            "leg_count": legs.len(),
            "basket_scope": "single_binary_market"
        }),
    };
    store.insert_signal(&signal).await?;
    store
        .insert_funnel_event(&FunnelEvent::new(
            Some(signal.signal_id),
            Some(signal.market_id.clone()),
            "edge_scan",
            if signal.status == SignalStatus::Detected {
                "pass"
            } else {
                "reject"
            },
            signal.reject_reason.clone(),
            signal.metadata.clone(),
        ))
        .await?;
    report.signals_inserted += 1;

    if signal.status == SignalStatus::Detected {
        report.positive_signals += 1;
        let order_plan = OrderPlan {
            plan_id: Uuid::new_v4(),
            orders: legs
                .iter()
                .map(|(market, token, _book, walk)| {
                    let price = walk
                        .fills
                        .last()
                        .map(|fill| fill.price)
                        .unwrap_or(Decimal::ZERO);
                    OrderRequest {
                        client_order_id: deterministic_client_order_id(&ClientOrderIdSeed {
                            strategy_version: "basket-v1",
                            source_id: signal.signal_id,
                            purpose: "signal_entry",
                            market_id: &market.market_id,
                            token_id: &token.token_id,
                            side: OrderSide::Buy,
                            notional_key: &(price * size).round_dp(4).normalize().to_string(),
                        }),
                        market_id: market.market_id.clone(),
                        token_id: token.token_id.clone(),
                        side: OrderSide::Buy,
                        order_type: OrderType::Fok,
                        price,
                        size,
                        signal_id: Some(signal.signal_id),
                        metadata: serde_json::json!({"purpose": "signal_entry"}),
                    }
                })
                .collect(),
        };
        let execution = execute_order_plan(venue, order_plan).await?;
        report.orders_inserted += execution.orders.len();
        report.fills_inserted += execution.fills.len();
        store.persist_order_plan_report(&execution).await?;
        for fill in &execution.fills {
            store
                .upsert_position(&PositionSnapshot {
                    position_id: None,
                    market_id: execution
                        .orders
                        .iter()
                        .find(|order| order.order_id == fill.order_id)
                        .map(|order| order.request.market_id.clone())
                        .unwrap_or_else(|| market.market_id.clone()),
                    token_id: fill.token_id.clone(),
                    underlying_key: market.underlying_key.clone(),
                    status: "open_sim".to_string(),
                    size: fill.size,
                    cost_basis: fill.price * fill.size + fill.fee,
                    worst_case_loss: worst_case.amount,
                    opened_at: None,
                    closed_at: None,
                    raw_payload: serde_json::to_value(fill)
                        .unwrap_or_else(|_| serde_json::json!({})),
                })
                .await?;
            report.positions_upserted += 1;
        }
    }

    Ok(report)
}

async fn insert_reject(
    store: &Store,
    market: &Market,
    reason: &str,
    edge: Decimal,
    threshold: Decimal,
    size: Decimal,
    worst_case_loss: Option<Decimal>,
) -> Result<()> {
    let signal = SignalCandidate {
        signal_id: Uuid::new_v4(),
        signal_type: SignalType::CheapBasket,
        market_id: market.market_id.clone(),
        expected_edge: edge,
        threshold,
        size,
        status: SignalStatus::Rejected,
        reject_reason: Some(reason.to_string()),
        worst_case_loss,
        metadata: serde_json::json!({
            "event_id": market.event_id,
            "outcome_group_id": market.outcome_group_id,
            "basket_scope": "single_binary_market"
        }),
    };
    store.insert_signal(&signal).await?;
    store
        .insert_funnel_event(&FunnelEvent::new(
            Some(signal.signal_id),
            Some(signal.market_id.clone()),
            "liquidity_filter",
            "reject",
            Some(reason.to_string()),
            signal.metadata,
        ))
        .await?;
    Ok(())
}

fn immediate_sell_revenue(
    legs: &[(&Market, &OutcomeToken, LocalOrderBook, DepthWalk)],
    size: Decimal,
    max_quote_age: Duration,
    now: chrono::DateTime<Utc>,
) -> Decimal {
    legs.iter()
        .filter_map(|(_, _, book, _)| book.depth_walk_sell(size, max_quote_age, now))
        .map(|walk| walk.total)
        .sum()
}

impl ScannerCycleReport {
    fn merge(&mut self, other: ScannerCycleReport) {
        self.events_considered += other.events_considered;
        self.tokens_persisted += other.tokens_persisted;
        self.books_persisted += other.books_persisted;
        self.signals_inserted += other.signals_inserted;
        self.positive_signals += other.positive_signals;
        self.orders_inserted += other.orders_inserted;
        self.fills_inserted += other.fills_inserted;
        self.positions_upserted += other.positions_upserted;
        self.errors += other.errors;
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use rust_decimal_macros::dec;

    use crate::orderbook::BookSide;

    use super::*;

    #[test]
    fn immediate_sell_revenue_walks_all_books() {
        let market = Market {
            event_id: "e1".to_string(),
            market_id: "m1".to_string(),
            outcome_group_id: Some("g1".to_string()),
            question: "q".to_string(),
            category: None,
            active: true,
            closed: false,
            archived: false,
            neg_risk: true,
            neg_risk_augmented: false,
            rules: None,
            end_date: None,
            underlying_key: "u".to_string(),
            resolution_score: 5,
            outcome_tokens: vec![],
            raw: serde_json::json!({}),
        };
        let token = OutcomeToken {
            market_id: "m1".to_string(),
            token_id: "t1".to_string(),
            outcome: "Yes".to_string(),
            side: TokenSide::Yes,
            condition_id: None,
            tick_size: dec!(0.01),
            neg_risk: true,
        };
        let now = Utc::now();
        let mut book = LocalOrderBook::default();
        book.upsert_level(BookSide::Bid, dec!(0.40), dec!(5), now);
        let walk = book
            .depth_walk_buy(dec!(0), Duration::seconds(45), now)
            .unwrap();
        let legs = vec![(&market, &token, book, walk)];
        assert_eq!(
            immediate_sell_revenue(&legs, dec!(5), Duration::seconds(45), now),
            dec!(2.00)
        );
    }
}
