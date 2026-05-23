pub mod live;
pub mod sim;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::models::{
    ConversionRequest, ConversionResult, FillRecord, OrderRecord, OrderRequest, OrderSide,
    OrderType,
};

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

#[derive(Debug, Clone, Serialize)]
pub struct LiveIdentityDiagnostics {
    pub mode: String,
    pub clob_api_base_url: String,
    pub signer_address: Option<String>,
    pub configured_funder_address: Option<String>,
    pub configured_signature_type: Option<String>,
    pub resolved_signature_type: Option<String>,
    pub authenticated_client_address: Option<String>,
    pub credentials_present: bool,
    pub api_keys_readable: bool,
    pub api_keys_error: Option<String>,
    pub balance_allowance_readable: bool,
    pub balance_allowance_error: Option<String>,
    pub collateral_balance: Option<String>,
    pub open_orders_readable: bool,
    pub open_orders_error: Option<String>,
    pub open_orders_count: Option<usize>,
    pub checked_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveWalletAddressDiagnostics {
    pub mode: String,
    pub signer_address: Option<String>,
    pub configured_funder_address: Option<String>,
    pub configured_signature_type: Option<String>,
    pub resolved_signature_type: Option<String>,
    pub authenticated_client_address: Option<String>,
    pub derived_proxy_wallet_address: Option<String>,
    pub derived_safe_wallet_address: Option<String>,
    pub expected_order_maker_address: Option<String>,
    pub expected_order_signer_field: Option<String>,
    pub configured_funder_matches_signer: Option<bool>,
    pub configured_funder_matches_proxy_wallet: Option<bool>,
    pub configured_funder_matches_safe_wallet: Option<bool>,
    pub configured_funder_deployed_as_deposit_wallet: Option<bool>,
    pub configured_funder_deployed_as_deposit_wallet_error: Option<String>,
    pub relayer_base_url: Option<String>,
    pub relayer_deployment_check_url: Option<String>,
    pub signer_balances: Option<LiveWalletTokenBalances>,
    pub configured_funder_balances: Option<LiveWalletTokenBalances>,
    pub candidate_addresses: Vec<LiveWalletCandidateAddressDiagnostics>,
    pub verified_deposit_wallet_address: Option<String>,
    pub verified_deposit_wallet_candidates_count: usize,
    pub checked_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveWalletTokenBalances {
    pub address: String,
    pub pol_wei: Option<String>,
    pub pusd: Option<String>,
    pub usdc_e: Option<String>,
    pub native_usdc: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveWalletCandidateAddressDiagnostics {
    pub address: String,
    pub matches_signer: Option<bool>,
    pub matches_configured_funder: Option<bool>,
    pub matches_authenticated_client: Option<bool>,
    pub matches_proxy_wallet: Option<bool>,
    pub matches_safe_wallet: Option<bool>,
    pub deployed_as_deposit_wallet: Option<bool>,
    pub deployed_as_deposit_wallet_error: Option<String>,
    pub deposit_wallet_deployment_check_url: Option<String>,
    pub deployed_as_safe_wallet: Option<bool>,
    pub deployed_as_safe_wallet_error: Option<String>,
    pub safe_wallet_deployment_check_url: Option<String>,
    pub balances: Option<LiveWalletTokenBalances>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LiveOrderDryRunRequest {
    pub token_id: String,
    pub side: OrderSide,
    #[serde(default)]
    pub order_type: Option<OrderType>,
    pub price: Decimal,
    pub size: Decimal,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveOrderDryRunDiagnostics {
    pub mode: String,
    pub clob_api_base_url: String,
    pub signer_address: Option<String>,
    pub configured_funder_address: Option<String>,
    pub configured_signature_type: Option<String>,
    pub resolved_signature_type: Option<String>,
    pub authenticated_client_address: Option<String>,
    pub order_signer: Option<String>,
    pub order_maker: Option<String>,
    pub order_signature_type: Option<String>,
    pub order_signer_matches_authenticated_client: Option<bool>,
    pub order_signer_matches_configured_funder: Option<bool>,
    pub order_maker_matches_configured_funder: Option<bool>,
    pub owner_redacted: bool,
    pub signature_redacted: bool,
    pub signed_order: serde_json::Value,
    pub checked_at: DateTime<Utc>,
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
    async fn live_identity_diagnostics(&self) -> Result<LiveIdentityDiagnostics>;
    async fn live_wallet_address_diagnostics(
        &self,
        candidate_addresses: Vec<String>,
    ) -> Result<LiveWalletAddressDiagnostics>;
    async fn live_order_dry_run(
        &self,
        request: LiveOrderDryRunRequest,
    ) -> Result<LiveOrderDryRunDiagnostics>;
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
