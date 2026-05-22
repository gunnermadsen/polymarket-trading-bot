use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use rust_decimal::Decimal;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    clob::ClobClient,
    edge::compute_taker_fee,
    execution::{ExecutionVenue, LiveVenueStatus, ReconciliationReport},
    models::{
        ConversionRequest, ConversionResult, FillRecord, FillSource, OrderRecord, OrderRequest,
        OrderSide, OrderState, OrderType,
    },
    orderbook::{BookSide, FillQuote},
};

#[derive(Debug, Clone)]
pub struct SimVenue {
    state: Arc<Mutex<SimState>>,
    clob: Option<ClobClient>,
    taker_fee_rate: Decimal,
    fill_source: FillSource,
    order_prefix: &'static str,
    mode_name: &'static str,
}

#[derive(Debug, Default)]
struct SimState {
    orders: Vec<OrderRecord>,
    fills: Vec<FillRecord>,
}

struct SimExitExecution {
    fills: Vec<FillRecord>,
    metadata: serde_json::Value,
}

impl SimVenue {
    pub fn with_clob(clob: ClobClient, taker_fee_rate: Decimal) -> Self {
        Self {
            state: Arc::new(Mutex::new(SimState::default())),
            clob: Some(clob),
            taker_fee_rate,
            fill_source: FillSource::Sim,
            order_prefix: "sim",
            mode_name: "sim",
        }
    }

    pub fn paper_with_clob(clob: ClobClient, taker_fee_rate: Decimal) -> Self {
        Self {
            state: Arc::new(Mutex::new(SimState::default())),
            clob: Some(clob),
            taker_fee_rate,
            fill_source: FillSource::Paper,
            order_prefix: "paper",
            mode_name: "paper",
        }
    }

    async fn executable_exit_fills(
        &self,
        order_id: &str,
        request: &OrderRequest,
        now: chrono::DateTime<Utc>,
    ) -> Result<Option<SimExitExecution>> {
        if request
            .metadata
            .get("purpose")
            .and_then(|value| value.as_str())
            != Some("whale_led_exit")
        {
            return Ok(None);
        }
        let Some(clob) = &self.clob else {
            return Ok(Some(SimExitExecution {
                fills: vec![],
                metadata: serde_json::json!({
                    "reject_reason": "missing_orderbook",
                    "book_snapshot_timestamp": now,
                }),
            }));
        };
        let book = match clob.fetch_orderbook(&request.token_id).await {
            Ok(book) => book,
            Err(error) => {
                return Ok(Some(SimExitExecution {
                    fills: vec![],
                    metadata: serde_json::json!({
                        "reject_reason": "missing_orderbook",
                        "venue_error": error.to_string(),
                        "book_snapshot_timestamp": now,
                    }),
                }))
            }
        };
        let max_age = chrono::Duration::seconds(10);
        let book_side = match request.side {
            OrderSide::Buy => BookSide::Ask,
            OrderSide::Sell => BookSide::Bid,
        };
        let depth_summary = book.limit_depth_summary(book_side, request.price, max_age, now);
        let best_bid = book.best_bid();
        let best_ask = book.best_ask();
        let base_metadata = serde_json::json!({
            "best_bid": best_bid,
            "best_ask": best_ask,
            "requested_side": request.side,
            "requested_size": request.size,
            "requested_limit_price": request.price,
            "available_depth_at_limit": depth_summary.fillable_size,
            "depth_walk_fillable_size": depth_summary.fillable_size,
            "depth_walk_avg_price": depth_summary.avg_price,
            "book_snapshot_timestamp": now,
        });
        let walk = match request.side {
            OrderSide::Buy => book.depth_walk_buy_limit(request.size, request.price, max_age, now),
            OrderSide::Sell => {
                book.depth_walk_sell_limit(request.size, request.price, max_age, now)
            }
        };
        let Some(walk) = walk else {
            let reject_reason = if request.size <= Decimal::ZERO {
                "zero_or_invalid_size"
            } else if depth_summary.fillable_size <= Decimal::ZERO {
                "limit_price_not_crossable"
            } else {
                "insufficient_depth"
            };
            return Ok(Some(SimExitExecution {
                fills: vec![],
                metadata: merge_json(
                    base_metadata,
                    serde_json::json!({"reject_reason": reject_reason}),
                ),
            }));
        };
        let fee = compute_taker_fee(&walk.fills, self.taker_fee_rate);
        let total_notional = walk.total;
        let fills = walk
            .fills
            .into_iter()
            .enumerate()
            .map(|(idx, fill)| FillRecord {
                fill_id: deterministic_fill_id(self.fill_source, order_id, idx),
                process_id: request.process_id,
                order_id: order_id.to_string(),
                token_id: request.token_id.clone(),
                price: fill.price,
                size: fill.size,
                fee: allocate_level_fee(fill, fee, total_notional),
                source: self.fill_source,
                filled_at: now + chrono::Duration::microseconds(idx as i64),
            })
            .collect();
        Ok(Some(SimExitExecution {
            fills,
            metadata: merge_json(
                base_metadata,
                serde_json::json!({
                    "reject_reason": null,
                    "depth_walk_total": total_notional
                }),
            ),
        }))
    }
}

fn merge_json(mut left: serde_json::Value, right: serde_json::Value) -> serde_json::Value {
    if let (Some(left), Some(right)) = (left.as_object_mut(), right.as_object()) {
        for (key, value) in right {
            left.insert(key.clone(), value.clone());
        }
    }
    left
}

fn allocate_level_fee(fill: FillQuote, total_fee: Decimal, total_notional: Decimal) -> Decimal {
    if total_fee <= Decimal::ZERO || total_notional <= Decimal::ZERO {
        return Decimal::ZERO;
    }
    total_fee * (fill.price * fill.size) / total_notional
}

#[async_trait]
impl ExecutionVenue for SimVenue {
    async fn submit_order(&self, mut request: OrderRequest) -> Result<OrderRecord> {
        let now = Utc::now();
        let order_id = format!("{}-{}", self.order_prefix, request.client_order_id);
        {
            let state = self.state.lock().await;
            if let Some(order) = state.orders.iter().find(|order| order.order_id == order_id) {
                return Ok(order.clone());
            }
        }
        let exit_execution = self.executable_exit_fills(&order_id, &request, now).await?;
        let executable_fills = if let Some(exit_execution) = exit_execution {
            request.metadata = merge_json(request.metadata, exit_execution.metadata);
            exit_execution.fills
        } else {
            vec![FillRecord {
                fill_id: deterministic_fill_id(self.fill_source, &order_id, 0),
                process_id: request.process_id,
                order_id: order_id.clone(),
                token_id: request.token_id.clone(),
                price: request.price,
                size: request.size,
                fee: Decimal::ZERO,
                source: self.fill_source,
                filled_at: now,
            }]
        };
        let filled_size: Decimal = executable_fills.iter().map(|fill| fill.size).sum();
        let state = if filled_size >= request.size {
            OrderState::Filled
        } else if filled_size > Decimal::ZERO && request.order_type != OrderType::Fok {
            OrderState::PartiallyFilled
        } else {
            OrderState::Rejected
        };
        let order = OrderRecord {
            order_id: order_id.clone(),
            request: request.clone(),
            state,
            created_at: now,
            updated_at: now,
        };
        let mut state = self.state.lock().await;
        state.orders.push(order.clone());
        state.fills.extend(executable_fills);
        Ok(order)
    }

    async fn cancel_order(&self, order_id: &str) -> Result<OrderRecord> {
        let mut state = self.state.lock().await;
        if let Some(order) = state
            .orders
            .iter_mut()
            .find(|order| order.order_id == order_id)
        {
            order.state = OrderState::Cancelled;
            order.updated_at = Utc::now();
            return Ok(order.clone());
        }
        Ok(OrderRecord {
            order_id: order_id.to_string(),
            request: OrderRequest {
                client_order_id: Uuid::new_v4(),
                process_id: None,
                market_id: "unknown".to_string(),
                token_id: "unknown".to_string(),
                side: crate::models::OrderSide::Buy,
                order_type: crate::models::OrderType::Fok,
                price: Decimal::ZERO,
                size: Decimal::ZERO,
                signal_id: None,
                metadata: serde_json::json!({}),
            },
            state: OrderState::Unknown,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        })
    }

    async fn cancel_all(&self) -> Result<usize> {
        let mut state = self.state.lock().await;
        let mut count = 0;
        for order in &mut state.orders {
            if !is_terminal(order.state) {
                order.state = OrderState::Cancelled;
                count += 1;
            }
        }
        Ok(count)
    }

    async fn get_balances(&self) -> Result<Vec<(String, Decimal)>> {
        Ok(vec![("USDC".to_string(), Decimal::from(1_000_000))])
    }

    async fn get_open_orders(&self) -> Result<Vec<OrderRecord>> {
        let state = self.state.lock().await;
        Ok(state
            .orders
            .iter()
            .filter(|order| !is_terminal(order.state))
            .cloned()
            .collect())
    }

    async fn convert_negative_risk(&self, request: ConversionRequest) -> Result<ConversionResult> {
        Ok(ConversionResult {
            conversion_id: request.conversion_id,
            status: "sim_confirmed".to_string(),
            tx_hash: Some(format!("sim-conversion-{}", request.conversion_id)),
            latency_ms: 1000,
            gas_cost_usd: Decimal::ZERO,
        })
    }

    async fn split_ctf(&self, _market_id: &str, _size: Decimal) -> Result<ConversionResult> {
        Ok(ConversionResult {
            conversion_id: Uuid::new_v4(),
            status: "sim_split".to_string(),
            tx_hash: None,
            latency_ms: 0,
            gas_cost_usd: Decimal::ZERO,
        })
    }

    async fn merge_ctf(&self, _market_id: &str, _size: Decimal) -> Result<ConversionResult> {
        Ok(ConversionResult {
            conversion_id: Uuid::new_v4(),
            status: "sim_merge".to_string(),
            tx_hash: None,
            latency_ms: 0,
            gas_cost_usd: Decimal::ZERO,
        })
    }

    async fn reconcile(&self) -> Result<ReconciliationReport> {
        Ok(ReconciliationReport {
            open_orders: self.get_open_orders().await?.len(),
            balances_checked: true,
            mismatches_found: 0,
            unresolved_count: 0,
            checked_at: Utc::now(),
        })
    }

    async fn fills_for_order(&self, order_id: &str) -> Result<Vec<FillRecord>> {
        let state = self.state.lock().await;
        Ok(state
            .fills
            .iter()
            .filter(|fill| fill.order_id == order_id)
            .cloned()
            .collect())
    }

    async fn live_status(&self) -> Result<LiveVenueStatus> {
        Ok(LiveVenueStatus {
            mode: self.mode_name.to_string(),
            live_confirmed: false,
            order_submit_enabled: false,
            user_ws_enabled: false,
            user_ws_connected: false,
            last_user_ws_pong_age_secs: None,
            last_rest_reconcile_age_secs: None,
            idempotency_clean: true,
            unresolved_live_order_count: 0,
            max_order_notional_usd: Decimal::ZERO,
            max_open_notional_usd: Decimal::ZERO,
            entries_enabled: false,
            reason: Some(format!("{}_mode", self.mode_name)),
        })
    }

    async fn set_live_entries_enabled(
        &self,
        _enabled: bool,
        _reason: Option<String>,
    ) -> Result<LiveVenueStatus> {
        self.live_status().await
    }
}

impl Default for SimVenue {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(SimState::default())),
            clob: None,
            taker_fee_rate: Decimal::ZERO,
            fill_source: FillSource::Sim,
            order_prefix: "sim",
            mode_name: "sim",
        }
    }
}

fn deterministic_fill_id(source: FillSource, order_id: &str, fill_index: usize) -> Uuid {
    let source_name = match source {
        FillSource::Sim => "sim",
        FillSource::Paper => "paper",
        FillSource::Live => "live",
    };
    Uuid::new_v5(
        &Uuid::NAMESPACE_URL,
        format!("polymarket-bot:{source_name}:fill:{order_id}:{fill_index}").as_bytes(),
    )
}

fn is_terminal(state: OrderState) -> bool {
    matches!(
        state,
        OrderState::Filled | OrderState::Cancelled | OrderState::Rejected | OrderState::Expired
    )
}
