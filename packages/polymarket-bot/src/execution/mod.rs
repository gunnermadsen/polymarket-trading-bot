pub mod live;

use anyhow::{bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::account_reconcile::{AccountReconcileReport, AccountReconcileRequest};
use crate::models::{FillRecord, OrderRecord, OrderRequest, OrderSide, OrderState, OrderType};

pub const LIVE_EXECUTION_GATE_CLOSED_REASON: &str = "live_execution_gate_closed";
pub(crate) const LIVE_EXTERNAL_EVENT_CLOCK_SKEW: chrono::Duration = chrono::Duration::minutes(5);
pub(crate) const LIVE_FILL_RECONCILIATION_SKEW: chrono::Duration = chrono::Duration::hours(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveExecutionGateReason {
    OrderSubmissionDisabled,
    GlobalHalt,
    ManualEnableRequired,
    VenueReadiness,
    ProcessAccountingReadiness,
    PerOrderNotionalLimit,
    DailyLossLimit,
    OpenNotionalLimit,
    OpenPositionLimit,
    SettlementRedemptionUnproven,
    ReferenceFreshness,
    OrderbookReadiness,
    OrderbookFreshness,
    OrderbookMarketability,
}

impl LiveExecutionGateReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OrderSubmissionDisabled => "order_submission_disabled",
            Self::GlobalHalt => "global_halt",
            Self::ManualEnableRequired => "manual_enable_required",
            Self::VenueReadiness => "venue_readiness",
            Self::ProcessAccountingReadiness => "process_accounting_readiness",
            Self::PerOrderNotionalLimit => "per_order_notional_limit",
            Self::DailyLossLimit => "daily_loss_limit",
            Self::OpenNotionalLimit => "open_notional_limit",
            Self::OpenPositionLimit => "open_position_limit",
            Self::SettlementRedemptionUnproven => "settlement_redemption_unproven",
            Self::ReferenceFreshness => "reference_freshness",
            Self::OrderbookReadiness => "orderbook_readiness",
            Self::OrderbookFreshness => "orderbook_freshness",
            Self::OrderbookMarketability => "orderbook_marketability",
        }
    }
}

pub fn live_execution_gate_closed_order(
    mut request: OrderRequest,
    gate_reason: LiveExecutionGateReason,
) -> Result<OrderRecord> {
    let Some(metadata) = request.metadata.as_object_mut() else {
        bail!("live execution gate rejection requires object order metadata");
    };
    metadata.insert(
        "reject_reason".to_string(),
        serde_json::Value::String(LIVE_EXECUTION_GATE_CLOSED_REASON.to_string()),
    );
    metadata.insert(
        "live_execution_gate".to_string(),
        serde_json::json!({
            "gate_reason": gate_reason.as_str(),
            "post_attempted": false,
        }),
    );
    let now = Utc::now();
    Ok(OrderRecord {
        order_id: format!("live-pending-{}", request.client_order_id),
        request,
        state: OrderState::Rejected,
        created_at: now,
        updated_at: now,
    })
}

/// Process-owned safety check that runs after live order preparation and immediately before the
/// venue POST. Expected market/readiness denials return a bounded gate reason; corrupt evidence or
/// invariants remain hard errors.
pub(crate) mod live_pre_post_guard_sealed {
    pub trait Sealed {}
}

#[async_trait]
pub trait LivePrePostGuard: live_pre_post_guard_sealed::Sealed + Send + Sync {
    async fn validate_pre_post(
        &self,
        request: &OrderRequest,
    ) -> Result<Option<LiveExecutionGateReason>>;

    /// Observe the actual POST boundary after awaited preparation and authorization.
    /// Telemetry only; this never changes durable trading authorization.
    fn observe_post_attempt(&self, _request: &OrderRequest) {}
}

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
    pub geoblock_readable: bool,
    pub geoblock_blocked: Option<bool>,
    pub geoblock_country: Option<String>,
    pub geoblock_region: Option<String>,
    pub last_geoblock_check_age_secs: Option<i64>,
    pub order_submit_enabled: bool,
    pub user_ws_enabled: bool,
    pub user_ws_connected: bool,
    pub last_user_ws_pong_age_secs: Option<i64>,
    pub last_rest_reconcile_age_secs: Option<i64>,
    pub idempotency_clean: bool,
    pub unresolved_live_order_count: usize,
    pub process_accounting_proven: bool,
    pub process_accounting_status: String,
    pub max_order_notional_usd: Decimal,
    pub max_open_notional_usd: Decimal,
    pub entries_enabled: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveIdentityDiagnostics {
    pub mode: String,
    pub clob_api_base_url: String,
    pub geoblock_readable: bool,
    pub geoblock_blocked: Option<bool>,
    pub geoblock_country: Option<String>,
    pub geoblock_region: Option<String>,
    pub geoblock_error: Option<String>,
    pub signer_address: Option<String>,
    pub configured_funder_address: Option<String>,
    pub configured_signature_type: Option<String>,
    pub resolved_signature_type: Option<String>,
    pub authenticated_client_address: Option<String>,
    pub account_identity_valid: bool,
    pub account_identity_fingerprint_sha256: Option<String>,
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
    pub poly1271_authenticated_client_address: Option<String>,
    pub poly1271_api_keys_readable: bool,
    pub poly1271_api_keys_error: Option<String>,
    pub poly1271_balance_allowance_readable: bool,
    pub poly1271_balance_allowance_error: Option<String>,
    pub poly1271_collateral_balance: Option<String>,
    pub poly1271_open_orders_readable: bool,
    pub poly1271_open_orders_error: Option<String>,
    pub poly1271_open_orders_count: Option<usize>,
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

#[derive(Debug, Clone, Deserialize)]
pub struct LivePoly1271FunderProbeRequest {
    pub addresses: Vec<String>,
    pub token_id: String,
    pub side: OrderSide,
    #[serde(default)]
    pub order_type: Option<OrderType>,
    pub price: Decimal,
    pub size: Decimal,
}

#[derive(Debug, Clone, Serialize)]
pub struct LivePoly1271FunderProbeResponse {
    pub mode: String,
    pub clob_api_base_url: String,
    pub signer_address: Option<String>,
    pub token_id: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub price: Decimal,
    pub size: Decimal,
    pub candidates: Vec<LivePoly1271FunderProbeCandidate>,
    pub verified_funder_address: Option<String>,
    pub verified_funder_candidates_count: usize,
    pub checked_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LivePoly1271FunderProbeCandidate {
    pub address: String,
    pub address_valid: bool,
    pub derive_credentials_ok: bool,
    pub derive_credentials_error: Option<String>,
    pub authenticated_client_address: Option<String>,
    pub api_keys_readable: bool,
    pub api_keys_error: Option<String>,
    pub update_balance_allowance_ok: bool,
    pub update_balance_allowance_error: Option<String>,
    pub balance_allowance_readable: bool,
    pub balance_allowance_error: Option<String>,
    pub collateral_balance: Option<String>,
    pub open_orders_readable: bool,
    pub open_orders_error: Option<String>,
    pub open_orders_count: Option<usize>,
    pub signed_order_build_ok: bool,
    pub signed_order_error: Option<String>,
    pub signed_order_maker: Option<String>,
    pub signed_order_signer: Option<String>,
    pub signed_order_signature_type: Option<String>,
    pub maker_matches_candidate: Option<bool>,
    pub signer_matches_candidate: Option<bool>,
    pub signature_type_is_poly1271: Option<bool>,
    pub ready_for_live_canary: bool,
    pub signed_order: serde_json::Value,
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
        if !matches!(order.state, OrderState::Rejected | OrderState::Cancelled) {
            match venue.fills_for_order(&order.order_id).await {
                Ok(order_fills) => fills.extend(order_fills),
                Err(error) => {
                    warn!(
                        error = %error,
                        order_id = %order.order_id,
                        "deferred fill lookup failed; live reconciliation will backfill fills"
                    );
                }
            }
        }
        orders.push(order);
    }

    let reconciliation = match venue.reconcile().await {
        Ok(reconciliation) => reconciliation,
        Err(error) if venue.preserve_liveness_on_post_order_reconcile_error() => {
            warn!(
                error = %error,
                order_count = orders.len(),
                "post-order reconciliation deferred; preserving live trading process liveness"
            );
            ReconciliationReport {
                open_orders: 0,
                balances_checked: false,
                mismatches_found: 0,
                unresolved_count: 1,
                checked_at: Utc::now(),
            }
        }
        Err(error) => return Err(error),
    };
    Ok(OrderPlanReport {
        plan_id: plan.plan_id,
        orders,
        fills,
        reconciliation,
    })
}

#[async_trait]
pub trait ExecutionVenue: Send + Sync {
    fn preserve_liveness_on_post_order_reconcile_error(&self) -> bool {
        false
    }

    async fn find_existing_order(&self, _request: &OrderRequest) -> Result<Option<OrderRecord>> {
        Ok(None)
    }
    async fn submit_order(&self, request: OrderRequest) -> Result<OrderRecord>;
    async fn submit_order_with_pre_post_guard(
        &self,
        request: OrderRequest,
        guard: Option<std::sync::Arc<dyn LivePrePostGuard>>,
    ) -> Result<OrderRecord> {
        if guard.is_some() {
            bail!("execution venue does not support an adjacent live pre-POST guard");
        }
        self.submit_order(request).await
    }
    async fn cancel_order(&self, order_id: &str) -> Result<OrderRecord>;
    async fn cancel_all(&self) -> Result<usize>;
    async fn get_balances(&self) -> Result<Vec<(String, Decimal)>>;
    async fn get_open_orders(&self) -> Result<Vec<OrderRecord>>;
    async fn reconcile(&self) -> Result<ReconciliationReport>;
    async fn update_live_reconciliation_health(
        &self,
        _pending_settlement_count: usize,
        _error: Option<String>,
    ) -> Result<()> {
        Ok(())
    }
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
    async fn live_poly1271_funder_probe(
        &self,
        request: LivePoly1271FunderProbeRequest,
    ) -> Result<LivePoly1271FunderProbeResponse>;
    async fn live_account_reconcile(
        &self,
        request: AccountReconcileRequest,
    ) -> Result<AccountReconcileReport>;
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

    use crate::models::{OrderIntent, OrderRequest, OrderSide, OrderType};

    #[test]
    fn order_intent_is_derived_from_existing_metadata() {
        let mut request = OrderRequest {
            client_order_id: Uuid::new_v4(),
            process_id: None,
            market_id: "m1".to_string(),
            token_id: "t1".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.42),
            size: dec!(10),
            metadata: serde_json::json!({"purpose": "entry"}),
        };
        assert_eq!(request.intent(), OrderIntent::Entry);

        request.metadata = serde_json::json!({"purpose": "exit"});
        assert_eq!(request.intent(), OrderIntent::Exit);

        request.metadata = serde_json::json!({"execution_intent": "risk_reduction"});
        assert_eq!(request.intent(), OrderIntent::RiskReduction);

        request.metadata = serde_json::json!({"execution_intent": "unexpected", "purpose": "exit"});
        assert_eq!(request.intent(), OrderIntent::Exit);

        request.metadata = serde_json::json!({"execution_intent": "unexpected"});
        assert_eq!(request.intent(), OrderIntent::Entry);
    }
}
