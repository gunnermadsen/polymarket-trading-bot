pub mod live;
pub mod sim;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::Serialize;

use crate::models::{ConversionRequest, ConversionResult, FillRecord, OrderRecord, OrderRequest};

#[derive(Debug, Clone, Serialize)]
pub struct ReconciliationReport {
    pub open_orders: usize,
    pub balances_checked: bool,
    pub mismatches_found: usize,
    pub unresolved_count: usize,
    pub checked_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveVenueStatus {
    pub mode: String,
    pub live_confirmed: bool,
    pub order_submit_enabled: bool,
    pub user_ws_enabled: bool,
    pub user_ws_connected: bool,
    pub last_user_ws_pong_age_secs: Option<i64>,
    pub last_rest_reconcile_age_secs: Option<i64>,
    pub idempotency_clean: bool,
    pub unresolved_live_order_count: usize,
    pub max_order_notional_usd: Decimal,
    pub max_open_notional_usd: Decimal,
    pub entries_enabled: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct OrderPlan {
    pub plan_id: uuid::Uuid,
    pub orders: Vec<OrderRequest>,
}

#[derive(Debug, Clone)]
pub struct OrderPlanReport {
    pub plan_id: uuid::Uuid,
    pub orders: Vec<OrderRecord>,
    pub fills: Vec<FillRecord>,
    pub reconciliation: ReconciliationReport,
}

pub async fn execute_order_plan<V: ExecutionVenue + ?Sized>(
    venue: &V,
    plan: OrderPlan,
) -> Result<OrderPlanReport> {
    let mut orders = Vec::with_capacity(plan.orders.len());
    let mut fills = Vec::new();

    for request in plan.orders {
        let order = venue.submit_order(request).await?;
        fills.extend(venue.fills_for_order(&order.order_id).await?);
        orders.push(order);
    }

    let reconciliation = venue.reconcile().await?;
    Ok(OrderPlanReport {
        plan_id: plan.plan_id,
        orders,
        fills,
        reconciliation,
    })
}

#[async_trait]
pub trait ExecutionVenue: Send + Sync {
    async fn submit_order(&self, request: OrderRequest) -> Result<OrderRecord>;
    async fn cancel_order(&self, order_id: &str) -> Result<OrderRecord>;
    async fn cancel_all(&self) -> Result<usize>;
    async fn get_balances(&self) -> Result<Vec<(String, Decimal)>>;
    async fn get_open_orders(&self) -> Result<Vec<OrderRecord>>;
    async fn convert_negative_risk(&self, request: ConversionRequest) -> Result<ConversionResult>;
    async fn split_ctf(&self, market_id: &str, size: Decimal) -> Result<ConversionResult>;
    async fn merge_ctf(&self, market_id: &str, size: Decimal) -> Result<ConversionResult>;
    async fn reconcile(&self) -> Result<ReconciliationReport>;
    async fn fills_for_order(&self, order_id: &str) -> Result<Vec<FillRecord>>;
    async fn live_status(&self) -> Result<LiveVenueStatus>;
    async fn set_live_entries_enabled(
        &self,
        enabled: bool,
        reason: Option<String>,
    ) -> Result<LiveVenueStatus>;
}

#[cfg(test)]
mod tests {
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    use crate::models::{OrderRequest, OrderSide, OrderType};

    use super::{execute_order_plan, sim::SimVenue, ExecutionVenue, OrderPlan};

    #[tokio::test]
    async fn order_plan_uses_same_venue_path_for_sim() {
        let venue = SimVenue::default();
        let plan = OrderPlan {
            plan_id: Uuid::new_v4(),
            orders: vec![OrderRequest {
                client_order_id: Uuid::new_v4(),
                process_id: None,
                market_id: "m1".to_string(),
                token_id: "t1".to_string(),
                side: OrderSide::Buy,
                order_type: OrderType::Fok,
                price: dec!(0.42),
                size: dec!(10),
                signal_id: None,
                metadata: serde_json::json!({}),
            }],
        };

        let report = execute_order_plan(&venue, plan).await.unwrap();
        assert_eq!(report.orders.len(), 1);
        assert_eq!(report.fills.len(), 1);
        assert!(report.reconciliation.balances_checked);
        assert_eq!(report.reconciliation.unresolved_count, 0);
    }

    #[tokio::test]
    async fn sim_venue_does_not_duplicate_fills_for_retried_order_id() {
        let venue = SimVenue::default();
        let request = OrderRequest {
            client_order_id: Uuid::new_v4(),
            process_id: Some(Uuid::new_v4()),
            market_id: "m1".to_string(),
            token_id: "t1".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.42),
            size: dec!(10),
            signal_id: None,
            metadata: serde_json::json!({}),
        };

        let first = execute_order_plan(
            &venue,
            OrderPlan {
                plan_id: Uuid::new_v4(),
                orders: vec![request.clone()],
            },
        )
        .await
        .unwrap();
        let second = execute_order_plan(
            &venue,
            OrderPlan {
                plan_id: Uuid::new_v4(),
                orders: vec![request],
            },
        )
        .await
        .unwrap();

        assert_eq!(first.orders[0].order_id, second.orders[0].order_id);
        assert_eq!(first.fills.len(), 1);
        assert_eq!(second.fills.len(), 1);
        assert_eq!(first.fills[0].fill_id, second.fills[0].fill_id);
        assert_eq!(
            venue
                .fills_for_order(&first.orders[0].order_id)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn sim_venue_replay_after_restart_uses_deterministic_fill_id() {
        let request = OrderRequest {
            client_order_id: Uuid::new_v4(),
            process_id: Some(Uuid::new_v4()),
            market_id: "m1".to_string(),
            token_id: "t1".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.42),
            size: dec!(10),
            signal_id: None,
            metadata: serde_json::json!({}),
        };

        let first = execute_order_plan(
            &SimVenue::default(),
            OrderPlan {
                plan_id: Uuid::new_v4(),
                orders: vec![request.clone()],
            },
        )
        .await
        .unwrap();
        let replay = execute_order_plan(
            &SimVenue::default(),
            OrderPlan {
                plan_id: Uuid::new_v4(),
                orders: vec![request],
            },
        )
        .await
        .unwrap();

        assert_eq!(first.orders[0].order_id, replay.orders[0].order_id);
        assert_eq!(first.fills.len(), 1);
        assert_eq!(replay.fills.len(), 1);
        assert_eq!(first.fills[0].fill_id, replay.fills[0].fill_id);
    }
}
