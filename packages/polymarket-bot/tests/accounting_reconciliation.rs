use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{bail, Result};
use async_trait::async_trait;
use chrono::Utc;
use polymarket_bot::{
    account_reconcile::{AccountReconcileReport, AccountReconcileRequest},
    execution::{
        execute_order_plan, ExecutionVenue, LiveIdentityDiagnostics, LiveOrderDryRunDiagnostics,
        LiveOrderDryRunRequest, LivePoly1271FunderProbeRequest, LivePoly1271FunderProbeResponse,
        LiveVenueStatus, LiveWalletAddressDiagnostics, OrderPlan, ReconciliationReport,
    },
    fees::{dynamic_crypto_taker_fee, sealed_dynamic_fee_rate},
    models::{FillRecord, OrderRecord, OrderRequest, OrderSide, OrderState, OrderType},
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use uuid::Uuid;

#[derive(Default)]
struct RecoveringLiveVenue {
    reconciliation_attempts: AtomicUsize,
}

#[async_trait]
impl ExecutionVenue for RecoveringLiveVenue {
    fn preserve_liveness_on_post_order_reconcile_error(&self) -> bool {
        true
    }

    async fn submit_order(&self, request: OrderRequest) -> Result<OrderRecord> {
        let now = Utc::now();
        Ok(OrderRecord {
            order_id: format!("live-order-{}", request.client_order_id),
            request,
            state: OrderState::Submitted,
            created_at: now,
            updated_at: now,
        })
    }

    async fn cancel_order(&self, _order_id: &str) -> Result<OrderRecord> {
        bail!("not exercised")
    }

    async fn cancel_all(&self) -> Result<usize> {
        Ok(0)
    }

    async fn get_balances(&self) -> Result<Vec<(String, Decimal)>> {
        Ok(Vec::new())
    }

    async fn get_open_orders(&self) -> Result<Vec<OrderRecord>> {
        Ok(Vec::new())
    }

    async fn reconcile(&self) -> Result<ReconciliationReport> {
        let attempt = self.reconciliation_attempts.fetch_add(1, Ordering::SeqCst);
        if attempt == 0 {
            bail!("temporary account evidence outage");
        }
        Ok(ReconciliationReport {
            open_orders: 0,
            balances_checked: true,
            mismatches_found: 0,
            unresolved_count: 0,
            checked_at: Utc::now(),
        })
    }

    async fn fills_for_order(&self, _order_id: &str) -> Result<Vec<FillRecord>> {
        Ok(Vec::new())
    }

    async fn live_status(&self) -> Result<LiveVenueStatus> {
        bail!("not exercised")
    }

    async fn live_identity_diagnostics(&self) -> Result<LiveIdentityDiagnostics> {
        bail!("not exercised")
    }

    async fn live_wallet_address_diagnostics(
        &self,
        _candidate_addresses: Vec<String>,
    ) -> Result<LiveWalletAddressDiagnostics> {
        bail!("not exercised")
    }

    async fn live_order_dry_run(
        &self,
        _request: LiveOrderDryRunRequest,
    ) -> Result<LiveOrderDryRunDiagnostics> {
        bail!("not exercised")
    }

    async fn live_poly1271_funder_probe(
        &self,
        _request: LivePoly1271FunderProbeRequest,
    ) -> Result<LivePoly1271FunderProbeResponse> {
        bail!("not exercised")
    }

    async fn live_account_reconcile(
        &self,
        _request: AccountReconcileRequest,
    ) -> Result<AccountReconcileReport> {
        bail!("not exercised")
    }

    async fn set_live_entries_enabled(
        &self,
        _enabled: bool,
        _reason: Option<String>,
    ) -> Result<LiveVenueStatus> {
        bail!("not exercised")
    }
}

fn live_request() -> OrderRequest {
    OrderRequest {
        client_order_id: Uuid::new_v4(),
        process_id: Some(Uuid::new_v4()),
        market_id: "market".to_string(),
        token_id: "token".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Fok,
        price: dec!(0.40),
        size: dec!(10),
        metadata: serde_json::json!({"dynamic_fee_rate": "0.25"}),
    }
}

#[tokio::test]
async fn post_order_reconciliation_failure_is_nonfatal_and_next_attempt_recovers() {
    let venue = RecoveringLiveVenue::default();

    let degraded = execute_order_plan(
        &venue,
        OrderPlan {
            plan_id: Uuid::new_v4(),
            orders: vec![live_request()],
        },
    )
    .await
    .expect("recoverable reconciliation must not fail order execution");
    assert!(!degraded.reconciliation.balances_checked);
    assert_eq!(degraded.reconciliation.unresolved_count, 1);

    let recovered = execute_order_plan(
        &venue,
        OrderPlan {
            plan_id: Uuid::new_v4(),
            orders: vec![live_request()],
        },
    )
    .await
    .expect("a healthy retry must reconcile without operator intervention");
    assert!(recovered.reconciliation.balances_checked);
    assert_eq!(recovered.reconciliation.unresolved_count, 0);
    assert_eq!(venue.reconciliation_attempts.load(Ordering::SeqCst), 2);
}

#[test]
fn sealed_fee_evidence_produces_the_canonical_accounting_fee() {
    let request = live_request();
    let rate = sealed_dynamic_fee_rate(&request.metadata).unwrap();

    assert_eq!(rate, dec!(0.25));
    assert_eq!(
        dynamic_crypto_taker_fee(request.size, rate, request.price),
        dec!(0.60)
    );
}
