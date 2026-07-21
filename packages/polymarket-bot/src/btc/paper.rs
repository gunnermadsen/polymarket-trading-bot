use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use anyhow::{bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

use super::{
    feeds::BookRegistry,
    strategy::dynamic_crypto_taker_fee,
    types::{FeedIntegrityStatus, OrderbookCheckpoint, OrderbookLevel},
};
use crate::{
    account_reconcile::{AccountReconcileReport, AccountReconcileRequest},
    execution::{
        ExecutionVenue, LiveIdentityDiagnostics, LiveOrderDryRunDiagnostics,
        LiveOrderDryRunRequest, LivePoly1271FunderProbeRequest, LivePoly1271FunderProbeResponse,
        LiveVenueStatus, LiveWalletAddressDiagnostics, LiveWalletCandidateAddressDiagnostics,
        ReconciliationReport,
    },
    models::{
        ConversionRequest, ConversionResult, FillRecord, FillSource, OrderRecord, OrderRequest,
        OrderSide, OrderState, OrderType,
    },
};

pub const PAPER_DYNAMIC_FEE_RATE_METADATA_KEY: &str = "dynamic_fee_rate";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaperVenueConfig {
    /// Delay from intent submission to the simulated venue arrival.
    pub arrival_latency: Duration,
    /// Fraction of displayed size considered executable. Must be in `(0, 1]`.
    pub visible_depth_haircut: Decimal,
    pub max_book_age: chrono::Duration,
    pub starting_collateral_usd: Decimal,
}

impl Default for PaperVenueConfig {
    fn default() -> Self {
        Self {
            arrival_latency: Duration::from_millis(250),
            visible_depth_haircut: dec!(0.50),
            max_book_age: chrono::Duration::seconds(2),
            starting_collateral_usd: dec!(1000),
        }
    }
}

impl PaperVenueConfig {
    pub fn validate(&self) -> Result<()> {
        if self.visible_depth_haircut <= Decimal::ZERO || self.visible_depth_haircut > Decimal::ONE
        {
            bail!("paper visible_depth_haircut must be in (0, 1]");
        }
        if self.max_book_age <= chrono::Duration::zero() {
            bail!("paper max_book_age must be positive");
        }
        if self.starting_collateral_usd < Decimal::ZERO {
            bail!("paper starting_collateral_usd cannot be negative");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaperPreviewConfig {
    pub scenario_key: String,
    pub arrival_latency: Duration,
    /// Fraction of displayed size considered executable. Must be in `(0, 1]`.
    pub visible_depth_haircut: Decimal,
}

impl PaperPreviewConfig {
    pub fn validate(&self) -> Result<()> {
        if self.scenario_key.trim().is_empty() {
            bail!("paper preview scenario_key must not be empty");
        }
        if self.visible_depth_haircut <= Decimal::ZERO || self.visible_depth_haircut > Decimal::ONE
        {
            bail!("paper preview visible_depth_haircut must be in (0, 1]");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperPreviewResult {
    pub scenario_key: String,
    pub state: OrderState,
    pub submitted_at: DateTime<Utc>,
    pub arrival_at: DateTime<Utc>,
    pub filled_size: Decimal,
    pub filled_notional: Decimal,
    pub fees: Decimal,
    pub average_fill_price: Option<Decimal>,
    pub reject_reason: Option<String>,
    pub execution_metadata: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperVenueStatus {
    pub starting_collateral_usd: Decimal,
    pub available_collateral_usd: Decimal,
    pub entry_debits_usd: Decimal,
    pub settlement_credits_usd: Decimal,
    pub order_count: usize,
    pub fill_count: usize,
    pub settlements_applied: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaperSettlementCreditResult {
    pub settlement_id: Uuid,
    pub newly_applied: bool,
    pub payout_usd: Decimal,
    pub collateral_before_usd: Decimal,
    pub collateral_after_usd: Decimal,
}

#[derive(Debug, Clone)]
pub struct PaperVenue {
    registry: Arc<RwLock<BookRegistry>>,
    config: PaperVenueConfig,
    /// Process-owned strategy cap applied to raw displayed ask depth at arrival.
    max_depth_participation: Decimal,
    state: Arc<Mutex<PaperState>>,
    /// Serializes arrival simulation so concurrent retries cannot both fill the same client id.
    submit_guard: Arc<Mutex<()>>,
}

#[derive(Debug, Default)]
struct PaperState {
    orders: HashMap<String, OrderRecord>,
    fills: Vec<FillRecord>,
    rehydrated_order_count: usize,
    rehydrated_fill_count: usize,
    collateral_usd: Decimal,
    entry_debits_usd: Decimal,
    settlement_credits_usd: Decimal,
    applied_settlements: HashSet<Uuid>,
}

#[derive(Debug)]
struct PaperExecution {
    state: OrderState,
    fills: Vec<FillRecord>,
    metadata: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct PaperLevelFill {
    price: Decimal,
    size: Decimal,
}

#[derive(Debug, Clone, Copy, Default)]
struct AskDepthAtLimit {
    displayed_size: Decimal,
    executable_size: Decimal,
    executable_notional: Decimal,
}

impl PaperVenue {
    pub fn new(
        registry: Arc<RwLock<BookRegistry>>,
        config: PaperVenueConfig,
        max_depth_participation: Decimal,
    ) -> Result<Self> {
        config.validate()?;
        if max_depth_participation <= Decimal::ZERO || max_depth_participation > Decimal::ONE {
            bail!("paper max_depth_participation must be in (0, 1]");
        }
        let starting_collateral_usd = config.starting_collateral_usd;
        Ok(Self {
            registry,
            config,
            max_depth_participation,
            state: Arc::new(Mutex::new(PaperState {
                collateral_usd: starting_collateral_usd,
                ..PaperState::default()
            })),
            submit_guard: Arc::new(Mutex::new(())),
        })
    }

    pub fn registry(&self) -> Arc<RwLock<BookRegistry>> {
        self.registry.clone()
    }

    pub async fn rehydrate_capital(
        &self,
        entry_debits_usd: Decimal,
        settlement_credits_usd: Decimal,
        order_count: usize,
        fill_count: usize,
        credited_settlement_ids: impl IntoIterator<Item = Uuid>,
    ) -> Result<()> {
        if entry_debits_usd < Decimal::ZERO || settlement_credits_usd < Decimal::ZERO {
            bail!("paper resume capital totals cannot be negative");
        }
        let _guard = self.submit_guard.lock().await;
        let mut state = self.state.lock().await;
        if !state.orders.is_empty()
            || !state.fills.is_empty()
            || state.rehydrated_order_count != 0
            || state.rehydrated_fill_count != 0
            || state.entry_debits_usd != Decimal::ZERO
            || state.settlement_credits_usd != Decimal::ZERO
            || !state.applied_settlements.is_empty()
        {
            bail!("paper venue can only rehydrate into a fresh runtime state");
        }
        let collateral_usd =
            self.config.starting_collateral_usd - entry_debits_usd + settlement_credits_usd;
        if collateral_usd < Decimal::ZERO {
            bail!("paper resume capital would produce negative collateral");
        }
        state.collateral_usd = collateral_usd;
        state.entry_debits_usd = entry_debits_usd;
        state.settlement_credits_usd = settlement_credits_usd;
        state.rehydrated_order_count = order_count;
        state.rehydrated_fill_count = fill_count;
        state.applied_settlements = credited_settlement_ids.into_iter().collect();
        Ok(())
    }

    /// Simulates an alternate arrival/depth scenario against the shared live book without
    /// acquiring the primary submit guard or mutating orders, fills, or collateral.
    pub async fn preview_order(
        &self,
        request: &OrderRequest,
        config: &PaperPreviewConfig,
    ) -> Result<PaperPreviewResult> {
        config.validate()?;
        let submitted_at = Utc::now();
        if !config.arrival_latency.is_zero() {
            tokio::time::sleep(config.arrival_latency).await;
        }
        let arrival_at = Utc::now();
        let available_collateral_usd = self.state.lock().await.collateral_usd;
        let order_id = format!(
            "paper-preview-{}-{}",
            config.scenario_key, request.client_order_id
        );
        let execution = self
            .execute_at_arrival(
                &order_id,
                request,
                submitted_at,
                arrival_at,
                available_collateral_usd,
                config.arrival_latency,
                config.visible_depth_haircut,
                self.config.max_book_age,
                false,
                Some(&config.scenario_key),
            )
            .await;
        let filled_size = execution
            .fills
            .iter()
            .map(|fill| fill.size)
            .sum::<Decimal>();
        let filled_notional = execution
            .fills
            .iter()
            .map(|fill| fill.price * fill.size)
            .sum::<Decimal>();
        let fees = execution.fills.iter().map(|fill| fill.fee).sum::<Decimal>();
        let average_fill_price =
            (filled_size > Decimal::ZERO).then(|| filled_notional / filled_size);
        let reject_reason = execution
            .metadata
            .get("reject_reason")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        Ok(PaperPreviewResult {
            scenario_key: config.scenario_key.clone(),
            state: execution.state,
            submitted_at,
            arrival_at,
            filled_size,
            filled_notional,
            fees,
            average_fill_price,
            reject_reason,
            execution_metadata: execution.metadata,
        })
    }

    /// Credits an official binary-market payout exactly once for this venue instance. The
    /// durable settlement ledger supplies the stable idempotency key.
    pub async fn apply_settlement_credit(
        &self,
        settlement_id: Uuid,
        payout_usd: Decimal,
    ) -> Result<PaperSettlementCreditResult> {
        if payout_usd < Decimal::ZERO {
            bail!("paper settlement payout cannot be negative");
        }
        let _guard = self.submit_guard.lock().await;
        let mut state = self.state.lock().await;
        let collateral_before_usd = state.collateral_usd;
        let newly_applied = state.applied_settlements.insert(settlement_id);
        if newly_applied {
            state.collateral_usd += payout_usd;
            state.settlement_credits_usd += payout_usd;
        }
        Ok(PaperSettlementCreditResult {
            settlement_id,
            newly_applied,
            payout_usd,
            collateral_before_usd,
            collateral_after_usd: state.collateral_usd,
        })
    }

    pub async fn status(&self) -> PaperVenueStatus {
        let state = self.state.lock().await;
        PaperVenueStatus {
            starting_collateral_usd: self.config.starting_collateral_usd,
            available_collateral_usd: state.collateral_usd,
            entry_debits_usd: state.entry_debits_usd,
            settlement_credits_usd: state.settlement_credits_usd,
            order_count: state.rehydrated_order_count + state.orders.len(),
            fill_count: state.rehydrated_fill_count + state.fills.len(),
            settlements_applied: state.applied_settlements.len(),
        }
    }

    async fn execute_at_arrival(
        &self,
        order_id: &str,
        request: &OrderRequest,
        submitted_at: DateTime<Utc>,
        arrival_at: DateTime<Utc>,
        available_collateral_usd: Decimal,
        arrival_latency: Duration,
        visible_depth_haircut: Decimal,
        max_book_age: chrono::Duration,
        enforce_collateral: bool,
        preview_scenario_key: Option<&str>,
    ) -> PaperExecution {
        let base = serde_json::json!({
            "paper_execution": {
                "submitted_at": submitted_at,
                "arrival_at": arrival_at,
                "configured_latency_ms": arrival_latency.as_millis(),
                "observed_submit_to_arrival_ms": (arrival_at - submitted_at).num_milliseconds(),
                "visible_depth_haircut": visible_depth_haircut,
                "max_depth_participation": self.max_depth_participation,
                "max_book_age_ms": max_book_age.num_milliseconds(),
                "collateral_enforced": enforce_collateral,
                "non_mutating_preview": !enforce_collateral,
                "preview_scenario_key": preview_scenario_key,
                "requested_side": request.side,
                "requested_order_type": request.order_type,
                "requested_limit_price": request.price,
                "requested_size": request.size,
            }
        });

        if request.side != OrderSide::Buy {
            return paper_reject(base, "paper_only_supports_buy_orders");
        }
        if request.order_type != OrderType::Fok {
            return paper_reject(base, "paper_only_supports_fok_orders");
        }
        if request.price <= Decimal::ZERO
            || request.price > Decimal::ONE
            || request.size <= Decimal::ZERO
        {
            return paper_reject(base, "invalid_order_price_or_size");
        }
        let fee_rate = match dynamic_fee_rate(&request.metadata) {
            Some(rate) if rate >= Decimal::ZERO && rate <= Decimal::ONE => rate,
            Some(_) => return paper_reject(base, "invalid_dynamic_fee_rate"),
            None => return paper_reject(base, "missing_dynamic_fee_rate"),
        };

        let checkpoint = {
            let registry = self.registry.read().await;
            registry.checkpoint(&request.token_id)
        };
        let Some(mut checkpoint) = checkpoint else {
            return paper_reject(base, "missing_arrival_orderbook");
        };
        checkpoint.checkpoint_id = deterministic_checkpoint_id(&checkpoint);
        if checkpoint.market_id != request.market_id {
            return paper_reject_with_checkpoint(base, "arrival_market_mismatch", &checkpoint);
        }
        if checkpoint.integrity_status != FeedIntegrityStatus::Ok {
            return paper_reject_with_checkpoint(
                base,
                "arrival_book_integrity_failure",
                &checkpoint,
            );
        }
        // Local receipt time is the causality boundary. Exchange source time independently guards
        // against delayed market data while retaining the existing clock-lead behavior.
        if checkpoint.received_at > arrival_at
            || checkpoint.source_timestamp - arrival_at > max_book_age
        {
            return paper_reject_with_checkpoint(base, "future_arrival_orderbook", &checkpoint);
        }
        let source_age = arrival_at - checkpoint.source_timestamp;
        let receive_age = arrival_at - checkpoint.received_at;
        if source_age > max_book_age || receive_age > max_book_age {
            return paper_reject_with_checkpoint(base, "stale_arrival_orderbook", &checkpoint);
        }

        let Some(depth) =
            available_ask_depth(&checkpoint.asks, request.price, visible_depth_haircut)
        else {
            return paper_reject_with_checkpoint(
                base,
                "invalid_arrival_orderbook_depth",
                &checkpoint,
            );
        };
        let Some(max_participating_size) = depth
            .displayed_size
            .checked_mul(self.max_depth_participation)
        else {
            return paper_reject_with_checkpoint(
                base,
                "invalid_arrival_orderbook_depth",
                &checkpoint,
            );
        };
        let requested_depth_participation_at_limit = (depth.displayed_size > Decimal::ZERO)
            .then(|| request.size.checked_div(depth.displayed_size))
            .flatten();
        let base = merge_json(
            base,
            serde_json::json!({
                "paper_execution": {
                    "displayed_size_at_limit": depth.displayed_size,
                    "max_participating_size_at_limit": max_participating_size,
                    "requested_depth_participation_at_limit": requested_depth_participation_at_limit,
                }
            }),
        );
        if depth.executable_size >= request.size && request.size > max_participating_size {
            return paper_reject_with_details(
                base,
                "arrival_depth_participation_exceeded",
                &checkpoint,
                depth.executable_size,
                depth.executable_notional,
                fee_rate,
                &[],
            );
        }

        let mut remaining = request.size;
        let mut walked = Vec::new();
        for level in &checkpoint.asks {
            if level.price > request.price || level.size <= Decimal::ZERO {
                continue;
            }
            let executable_size = level.size * visible_depth_haircut;
            let fill_size = remaining.min(executable_size);
            if fill_size <= Decimal::ZERO {
                continue;
            }
            walked.push(PaperLevelFill {
                price: level.price,
                size: fill_size,
            });
            remaining -= fill_size;
            if remaining <= Decimal::ZERO {
                break;
            }
        }
        if remaining > Decimal::ZERO {
            return paper_reject_with_details(
                base,
                "insufficient_arrival_depth",
                &checkpoint,
                depth.executable_size,
                depth.executable_notional,
                fee_rate,
                &walked,
            );
        }

        let fills = walked
            .iter()
            .enumerate()
            .map(|(index, fill)| FillRecord {
                fill_id: deterministic_fill_id(order_id, index),
                process_id: request.process_id,
                order_id: order_id.to_string(),
                token_id: request.token_id.clone(),
                price: fill.price,
                size: fill.size,
                fee: dynamic_crypto_taker_fee(fill.size, fee_rate, fill.price),
                source: FillSource::Paper,
                filled_at: arrival_at + chrono::Duration::microseconds(index as i64),
            })
            .collect::<Vec<_>>();
        let filled_size = fills.iter().map(|fill| fill.size).sum::<Decimal>();
        let filled_notional = fills
            .iter()
            .map(|fill| fill.price * fill.size)
            .sum::<Decimal>();
        let total_fee = fills.iter().map(|fill| fill.fee).sum::<Decimal>();
        let required_collateral_usd = filled_notional + total_fee;
        let base = merge_json(
            base,
            serde_json::json!({
                "paper_execution": {
                    "available_collateral_usd": available_collateral_usd,
                    "required_collateral_usd": required_collateral_usd,
                }
            }),
        );
        if enforce_collateral && required_collateral_usd > available_collateral_usd {
            return paper_reject_with_details(
                base,
                "insufficient_paper_collateral",
                &checkpoint,
                depth.executable_size,
                depth.executable_notional,
                fee_rate,
                &walked,
            );
        }
        let average_price = filled_notional / filled_size;
        PaperExecution {
            state: OrderState::Filled,
            fills,
            metadata: merge_json(
                base,
                serde_json::json!({
                    "reject_reason": null,
                    "paper_execution": {
                        "arrival_checkpoint_id": checkpoint.checkpoint_id,
                        "arrival_checkpoint": checkpoint,
                        "arrival_source_age_ms": source_age.num_milliseconds(),
                        "arrival_receive_age_ms": receive_age.num_milliseconds(),
                        "available_size_at_limit": depth.executable_size,
                        "available_notional_at_limit": depth.executable_notional,
                        "dynamic_fee_rate": fee_rate,
                        "walked_levels": walked,
                        "filled_size": filled_size,
                        "filled_depth_participation_at_limit": requested_depth_participation_at_limit,
                        "filled_notional": filled_notional,
                        "average_fill_price": average_price,
                        "total_fee": total_fee
                    }
                }),
            ),
        }
    }
}

#[async_trait]
impl ExecutionVenue for PaperVenue {
    async fn submit_order(&self, mut request: OrderRequest) -> Result<OrderRecord> {
        let _guard = self.submit_guard.lock().await;
        let order_id = format!("paper-{}", request.client_order_id);
        if let Some(existing) = self.state.lock().await.orders.get(&order_id).cloned() {
            return Ok(existing);
        }

        let submitted_at = Utc::now();
        if !self.config.arrival_latency.is_zero() {
            tokio::time::sleep(self.config.arrival_latency).await;
        }
        let arrival_at = Utc::now();
        let available_collateral_usd = self.state.lock().await.collateral_usd;
        let execution = self
            .execute_at_arrival(
                &order_id,
                &request,
                submitted_at,
                arrival_at,
                available_collateral_usd,
                self.config.arrival_latency,
                self.config.visible_depth_haircut,
                self.config.max_book_age,
                true,
                None,
            )
            .await;
        request.metadata = merge_json(request.metadata, execution.metadata);
        let order = OrderRecord {
            order_id: order_id.clone(),
            request,
            state: execution.state,
            created_at: submitted_at,
            updated_at: arrival_at,
        };
        let mut state = self.state.lock().await;
        if execution.state == OrderState::Filled {
            let debit = execution
                .fills
                .iter()
                .map(|fill| fill.price * fill.size + fill.fee)
                .sum::<Decimal>();
            state.collateral_usd -= debit;
            state.entry_debits_usd += debit;
        }
        state.orders.insert(order_id, order.clone());
        state.fills.extend(execution.fills);
        Ok(order)
    }

    async fn cancel_order(&self, order_id: &str) -> Result<OrderRecord> {
        let mut state = self.state.lock().await;
        if let Some(order) = state.orders.get_mut(order_id) {
            if !is_terminal(order.state) {
                order.state = OrderState::Cancelled;
                order.updated_at = Utc::now();
            }
            return Ok(order.clone());
        }
        let now = Utc::now();
        Ok(OrderRecord {
            order_id: order_id.to_string(),
            request: unknown_order_request(order_id),
            state: OrderState::Unknown,
            created_at: now,
            updated_at: now,
        })
    }

    async fn cancel_all(&self) -> Result<usize> {
        let mut state = self.state.lock().await;
        let mut cancelled = 0;
        for order in state.orders.values_mut() {
            if !is_terminal(order.state) {
                order.state = OrderState::Cancelled;
                order.updated_at = Utc::now();
                cancelled += 1;
            }
        }
        Ok(cancelled)
    }

    async fn get_balances(&self) -> Result<Vec<(String, Decimal)>> {
        Ok(vec![(
            "USDC".to_string(),
            self.state.lock().await.collateral_usd,
        )])
    }

    async fn get_open_orders(&self) -> Result<Vec<OrderRecord>> {
        Ok(self
            .state
            .lock()
            .await
            .orders
            .values()
            .filter(|order| !is_terminal(order.state))
            .cloned()
            .collect())
    }

    async fn convert_negative_risk(&self, request: ConversionRequest) -> Result<ConversionResult> {
        Ok(ConversionResult {
            conversion_id: request.conversion_id,
            status: "paper_noop".to_string(),
            tx_hash: None,
            latency_ms: 0,
            gas_cost_usd: Decimal::ZERO,
        })
    }

    async fn split_ctf(&self, market_id: &str, size: Decimal) -> Result<ConversionResult> {
        Ok(paper_conversion("split", market_id, size))
    }

    async fn merge_ctf(&self, market_id: &str, size: Decimal) -> Result<ConversionResult> {
        Ok(paper_conversion("merge", market_id, size))
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
        Ok(self
            .state
            .lock()
            .await
            .fills
            .iter()
            .filter(|fill| fill.order_id == order_id)
            .cloned()
            .collect())
    }

    async fn live_status(&self) -> Result<LiveVenueStatus> {
        Ok(LiveVenueStatus {
            mode: "paper".to_string(),
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
            reason: Some("paper_mode".to_string()),
        })
    }

    async fn live_identity_diagnostics(&self) -> Result<LiveIdentityDiagnostics> {
        Ok(LiveIdentityDiagnostics {
            mode: "paper".to_string(),
            clob_api_base_url: String::new(),
            signer_address: None,
            configured_funder_address: None,
            configured_signature_type: None,
            resolved_signature_type: None,
            authenticated_client_address: None,
            credentials_present: false,
            api_keys_readable: false,
            api_keys_error: Some("paper_mode".to_string()),
            balance_allowance_readable: false,
            balance_allowance_error: Some("paper_mode".to_string()),
            collateral_balance: None,
            open_orders_readable: false,
            open_orders_error: Some("paper_mode".to_string()),
            open_orders_count: None,
            checked_at: Utc::now(),
        })
    }

    async fn live_wallet_address_diagnostics(
        &self,
        candidate_addresses: Vec<String>,
    ) -> Result<LiveWalletAddressDiagnostics> {
        Ok(LiveWalletAddressDiagnostics {
            mode: "paper".to_string(),
            signer_address: None,
            configured_funder_address: None,
            configured_signature_type: None,
            resolved_signature_type: None,
            authenticated_client_address: None,
            derived_proxy_wallet_address: None,
            derived_safe_wallet_address: None,
            expected_order_maker_address: None,
            expected_order_signer_field: None,
            configured_funder_matches_signer: None,
            configured_funder_matches_proxy_wallet: None,
            configured_funder_matches_safe_wallet: None,
            configured_funder_deployed_as_deposit_wallet: None,
            configured_funder_deployed_as_deposit_wallet_error: Some("paper_mode".to_string()),
            relayer_base_url: None,
            relayer_deployment_check_url: None,
            signer_balances: None,
            configured_funder_balances: None,
            candidate_addresses: candidate_addresses
                .into_iter()
                .map(|address| LiveWalletCandidateAddressDiagnostics {
                    address,
                    matches_signer: None,
                    matches_configured_funder: None,
                    matches_authenticated_client: None,
                    matches_proxy_wallet: None,
                    matches_safe_wallet: None,
                    deployed_as_deposit_wallet: None,
                    deployed_as_deposit_wallet_error: Some("paper_mode".to_string()),
                    deposit_wallet_deployment_check_url: None,
                    deployed_as_safe_wallet: None,
                    deployed_as_safe_wallet_error: Some("paper_mode".to_string()),
                    safe_wallet_deployment_check_url: None,
                    balances: None,
                    poly1271_authenticated_client_address: None,
                    poly1271_api_keys_readable: false,
                    poly1271_api_keys_error: Some("paper_mode".to_string()),
                    poly1271_balance_allowance_readable: false,
                    poly1271_balance_allowance_error: Some("paper_mode".to_string()),
                    poly1271_collateral_balance: None,
                    poly1271_open_orders_readable: false,
                    poly1271_open_orders_error: Some("paper_mode".to_string()),
                    poly1271_open_orders_count: None,
                })
                .collect(),
            verified_deposit_wallet_address: None,
            verified_deposit_wallet_candidates_count: 0,
            checked_at: Utc::now(),
        })
    }

    async fn live_order_dry_run(
        &self,
        _request: LiveOrderDryRunRequest,
    ) -> Result<LiveOrderDryRunDiagnostics> {
        bail!("live order dry-run is unavailable in paper mode")
    }

    async fn live_poly1271_funder_probe(
        &self,
        _request: LivePoly1271FunderProbeRequest,
    ) -> Result<LivePoly1271FunderProbeResponse> {
        bail!("live funder probe is unavailable in paper mode")
    }

    async fn live_account_reconcile(
        &self,
        _request: AccountReconcileRequest,
    ) -> Result<AccountReconcileReport> {
        bail!("live account reconciliation is unavailable in paper mode")
    }

    async fn set_live_entries_enabled(
        &self,
        _enabled: bool,
        _reason: Option<String>,
    ) -> Result<LiveVenueStatus> {
        self.live_status().await
    }
}

fn paper_reject(base: serde_json::Value, reason: &str) -> PaperExecution {
    PaperExecution {
        state: OrderState::Rejected,
        fills: Vec::new(),
        metadata: merge_json(
            base,
            serde_json::json!({
                "reject_reason": reason,
                "paper_execution": { "reject_reason": reason }
            }),
        ),
    }
}

fn paper_reject_with_checkpoint(
    base: serde_json::Value,
    reason: &str,
    checkpoint: &OrderbookCheckpoint,
) -> PaperExecution {
    paper_reject_with_details(
        base,
        reason,
        checkpoint,
        Decimal::ZERO,
        Decimal::ZERO,
        Decimal::ZERO,
        &[],
    )
}

fn paper_reject_with_details(
    base: serde_json::Value,
    reason: &str,
    checkpoint: &OrderbookCheckpoint,
    available_size: Decimal,
    available_notional: Decimal,
    fee_rate: Decimal,
    walked: &[PaperLevelFill],
) -> PaperExecution {
    PaperExecution {
        state: OrderState::Rejected,
        fills: Vec::new(),
        metadata: merge_json(
            base,
            serde_json::json!({
                "reject_reason": reason,
                "paper_execution": {
                    "reject_reason": reason,
                    "arrival_checkpoint_id": checkpoint.checkpoint_id,
                    "arrival_checkpoint": checkpoint,
                    "available_size_at_limit": available_size,
                    "available_notional_at_limit": available_notional,
                    "dynamic_fee_rate": fee_rate,
                    "walked_levels_before_fok_reject": walked
                }
            }),
        ),
    }
}

fn available_ask_depth(
    asks: &[OrderbookLevel],
    limit_price: Decimal,
    haircut: Decimal,
) -> Option<AskDepthAtLimit> {
    asks.iter()
        .filter(|level| level.price <= limit_price && level.size > Decimal::ZERO)
        .try_fold(AskDepthAtLimit::default(), |depth, level| {
            let executable_size = level.size.checked_mul(haircut)?;
            let executable_notional = executable_size.checked_mul(level.price)?;
            Some(AskDepthAtLimit {
                displayed_size: depth.displayed_size.checked_add(level.size)?,
                executable_size: depth.executable_size.checked_add(executable_size)?,
                executable_notional: depth.executable_notional.checked_add(executable_notional)?,
            })
        })
}

fn dynamic_fee_rate(metadata: &serde_json::Value) -> Option<Decimal> {
    [
        metadata.get(PAPER_DYNAMIC_FEE_RATE_METADATA_KEY),
        metadata.get("taker_fee_rate"),
        metadata
            .get("execution")
            .and_then(|execution| execution.get(PAPER_DYNAMIC_FEE_RATE_METADATA_KEY)),
        metadata
            .get("execution")
            .and_then(|execution| execution.get("taker_fee_rate")),
    ]
    .into_iter()
    .flatten()
    .find_map(decimal_from_json)
}

fn decimal_from_json(value: &serde_json::Value) -> Option<Decimal> {
    value
        .as_str()
        .and_then(|text| Decimal::from_str(text).ok())
        .or_else(|| Decimal::from_str(&value.to_string()).ok())
}

fn deterministic_checkpoint_id(checkpoint: &OrderbookCheckpoint) -> Uuid {
    Uuid::new_v5(
        &Uuid::NAMESPACE_URL,
        format!(
            "polymarket-bot:paper:arrival-book:{}:{}:{}:{}:{}",
            checkpoint.connection_id,
            checkpoint.token_id,
            checkpoint.ingest_sequence,
            checkpoint.source_timestamp.timestamp_micros(),
            checkpoint.source_hash.as_deref().unwrap_or("-")
        )
        .as_bytes(),
    )
}

fn deterministic_fill_id(order_id: &str, fill_index: usize) -> Uuid {
    Uuid::new_v5(
        &Uuid::NAMESPACE_URL,
        format!("polymarket-bot:paper:fill:{order_id}:{fill_index}").as_bytes(),
    )
}

fn unknown_order_request(order_id: &str) -> OrderRequest {
    OrderRequest {
        client_order_id: Uuid::new_v5(
            &Uuid::NAMESPACE_URL,
            format!("polymarket-bot:paper:unknown-order:{order_id}").as_bytes(),
        ),
        process_id: None,
        market_id: "unknown".to_string(),
        token_id: "unknown".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Fok,
        price: Decimal::ZERO,
        size: Decimal::ZERO,
        signal_id: None,
        metadata: serde_json::json!({ "source": "paper_unknown_order" }),
    }
}

fn paper_conversion(kind: &str, market_id: &str, size: Decimal) -> ConversionResult {
    ConversionResult {
        conversion_id: Uuid::new_v5(
            &Uuid::NAMESPACE_URL,
            format!(
                "polymarket-bot:paper:{kind}:{market_id}:{}",
                size.normalize()
            )
            .as_bytes(),
        ),
        status: format!("paper_{kind}_noop"),
        tx_hash: None,
        latency_ms: 0,
        gas_cost_usd: Decimal::ZERO,
    }
}

fn merge_json(mut left: serde_json::Value, right: serde_json::Value) -> serde_json::Value {
    if let (Some(left), Some(right)) = (left.as_object_mut(), right.as_object()) {
        for (key, value) in right {
            if let (Some(existing), Some(incoming)) = (
                left.get_mut(key).and_then(|value| value.as_object_mut()),
                value.as_object(),
            ) {
                for (nested_key, nested_value) in incoming {
                    existing.insert(nested_key.clone(), nested_value.clone());
                }
            } else {
                left.insert(key.clone(), value.clone());
            }
        }
    }
    left
}

fn is_terminal(state: OrderState) -> bool {
    matches!(
        state,
        OrderState::Filled | OrderState::Cancelled | OrderState::Rejected | OrderState::Expired
    )
}

#[cfg(test)]
mod tests {
    use chrono::{Duration as ChronoDuration, TimeZone};

    use super::*;
    use crate::{
        btc::{
            feeds::ClobMessage,
            types::{BtcIntervalMarket, OrderbookLevel},
        },
        models::{OrderRequest, OrderSide, OrderState, OrderType},
    };

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, 12, 12, 3, 0).unwrap()
    }

    fn market() -> BtcIntervalMarket {
        BtcIntervalMarket {
            event_id: "event".to_string(),
            event_slug: "btc-updown-5m-1783857600".to_string(),
            series_slug: "btc-up-or-down-5m".to_string(),
            market_id: "market".to_string(),
            condition_id: "condition".to_string(),
            window_start: now() - ChronoDuration::minutes(3),
            window_end: now() + ChronoDuration::minutes(2),
            up_token_id: "up".to_string(),
            down_token_id: "down".to_string(),
            tick_size: dec!(0.01),
            minimum_order_size: Some(dec!(1)),
            resolution_source: "chainlink".to_string(),
            active: true,
            closed: false,
            accepting_orders: true,
            fees_enabled: true,
            fee_schedule: serde_json::json!({}),
            raw_payload: serde_json::json!({}),
        }
    }

    fn registry_with_book(
        at: DateTime<Utc>,
        asks: Vec<OrderbookLevel>,
    ) -> Arc<RwLock<BookRegistry>> {
        registry_with_book_times(at, at, asks)
    }

    fn registry_with_book_times(
        source_at: DateTime<Utc>,
        received_at: DateTime<Utc>,
        asks: Vec<OrderbookLevel>,
    ) -> Arc<RwLock<BookRegistry>> {
        let mut registry = BookRegistry::new(Uuid::from_u128(100));
        registry.register_market(&market());
        registry.apply(
            ClobMessage::Book {
                market_id: "market".to_string(),
                token_id: "up".to_string(),
                bids: vec![OrderbookLevel {
                    price: dec!(0.39),
                    size: dec!(100),
                }],
                asks,
                source_timestamp: source_at,
                source_hash: Some("book-hash".to_string()),
                raw_payload: serde_json::json!({}),
            },
            received_at,
        );
        Arc::new(RwLock::new(registry))
    }

    fn venue(registry: Arc<RwLock<BookRegistry>>, haircut: Decimal) -> PaperVenue {
        venue_with_participation_cap(registry, haircut, Decimal::ONE)
    }

    fn venue_with_participation_cap(
        registry: Arc<RwLock<BookRegistry>>,
        haircut: Decimal,
        max_depth_participation: Decimal,
    ) -> PaperVenue {
        PaperVenue::new(
            registry,
            PaperVenueConfig {
                arrival_latency: Duration::ZERO,
                visible_depth_haircut: haircut,
                max_book_age: chrono::Duration::hours(24),
                starting_collateral_usd: dec!(100),
            },
            max_depth_participation,
        )
        .unwrap()
    }

    fn request(size: Decimal, limit: Decimal) -> OrderRequest {
        OrderRequest {
            client_order_id: Uuid::from_u128(200),
            process_id: Some(Uuid::from_u128(201)),
            market_id: "market".to_string(),
            token_id: "up".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: limit,
            size,
            signal_id: Some(Uuid::from_u128(202)),
            metadata: serde_json::json!({
                "dynamic_fee_rate": "0.25"
            }),
        }
    }

    #[tokio::test]
    async fn fok_walks_recorded_arrival_depth_and_charges_dynamic_fee_per_level() {
        let venue = venue(
            registry_with_book(
                Utc::now(),
                vec![
                    OrderbookLevel {
                        price: dec!(0.40),
                        size: dec!(2),
                    },
                    OrderbookLevel {
                        price: dec!(0.42),
                        size: dec!(4),
                    },
                ],
            ),
            Decimal::ONE,
        );
        let order = venue
            .submit_order(request(dec!(5), dec!(0.42)))
            .await
            .unwrap();
        assert_eq!(order.state, OrderState::Filled);
        let fills = venue.fills_for_order(&order.order_id).await.unwrap();
        assert_eq!(fills.len(), 2);
        assert_eq!(fills[0].size, dec!(2));
        assert_eq!(fills[1].size, dec!(3));
        assert_eq!(fills[0].fee, dec!(0.12));
        assert_eq!(fills[1].fee, dec!(0.18270));
        assert!(
            order.request.metadata["paper_execution"]["arrival_checkpoint_id"]
                .as_str()
                .is_some()
        );
        assert_eq!(
            decimal_from_json(&order.request.metadata["paper_execution"]["average_fill_price"]),
            Some(dec!(0.412))
        );
    }

    #[tokio::test]
    async fn fok_rejects_without_partial_fills_when_arrival_depth_is_insufficient() {
        let venue = venue(
            registry_with_book(
                Utc::now(),
                vec![OrderbookLevel {
                    price: dec!(0.40),
                    size: dec!(2),
                }],
            ),
            Decimal::ONE,
        );
        let order = venue
            .submit_order(request(dec!(3), dec!(0.40)))
            .await
            .unwrap();
        assert_eq!(order.state, OrderState::Rejected);
        assert_eq!(
            order.request.metadata["reject_reason"],
            serde_json::json!("insufficient_arrival_depth")
        );
        assert!(venue
            .fills_for_order(&order.order_id)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn depth_haircut_is_applied_before_fok_decision() {
        let venue = venue(
            registry_with_book(
                Utc::now(),
                vec![OrderbookLevel {
                    price: dec!(0.40),
                    size: dec!(10),
                }],
            ),
            dec!(0.25),
        );
        let order = venue
            .submit_order(request(dec!(3), dec!(0.40)))
            .await
            .unwrap();
        assert_eq!(order.state, OrderState::Rejected);
        assert_eq!(
            decimal_from_json(
                &order.request.metadata["paper_execution"]["available_size_at_limit"]
            ),
            Some(dec!(2.50))
        );
    }

    #[tokio::test]
    async fn arrival_participation_excludes_depth_above_limit_and_rejects_atomically() {
        let venue = venue_with_participation_cap(
            registry_with_book(
                Utc::now(),
                vec![
                    OrderbookLevel {
                        price: dec!(0.40),
                        size: dec!(10),
                    },
                    OrderbookLevel {
                        price: dec!(0.41),
                        size: dec!(100),
                    },
                ],
            ),
            Decimal::ONE,
            dec!(0.25),
        );
        let before = venue.status().await;

        let order = venue
            .submit_order(request(dec!(3), dec!(0.40)))
            .await
            .unwrap();
        let after = venue.status().await;

        assert_eq!(order.state, OrderState::Rejected);
        assert_eq!(
            order.request.metadata["reject_reason"],
            serde_json::json!("arrival_depth_participation_exceeded")
        );
        assert_eq!(
            decimal_from_json(
                &order.request.metadata["paper_execution"]["displayed_size_at_limit"]
            ),
            Some(dec!(10))
        );
        assert_eq!(
            decimal_from_json(
                &order.request.metadata["paper_execution"]["available_size_at_limit"]
            ),
            Some(dec!(10))
        );
        assert_eq!(
            decimal_from_json(
                &order.request.metadata["paper_execution"]["max_participating_size_at_limit"]
            ),
            Some(dec!(2.5))
        );
        assert_eq!(
            decimal_from_json(
                &order.request.metadata["paper_execution"]
                    ["requested_depth_participation_at_limit"]
            ),
            Some(dec!(0.3))
        );
        assert!(order.request.metadata["paper_execution"]
            .get("filled_depth_participation_at_limit")
            .is_none());
        assert!(venue
            .fills_for_order(&order.order_id)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(after.fill_count, before.fill_count);
        assert_eq!(
            after.available_collateral_usd,
            before.available_collateral_usd
        );
        assert_eq!(after.entry_debits_usd, before.entry_debits_usd);
    }

    #[tokio::test]
    async fn arrival_participation_accepts_exact_cap_without_double_applying_haircut() {
        let venue = venue_with_participation_cap(
            registry_with_book(
                Utc::now(),
                vec![OrderbookLevel {
                    price: dec!(0.40),
                    size: dec!(20),
                }],
            ),
            dec!(0.50),
            dec!(0.25),
        );

        let order = venue
            .submit_order(request(dec!(5), dec!(0.40)))
            .await
            .unwrap();

        assert_eq!(order.state, OrderState::Filled);
        assert_eq!(
            decimal_from_json(
                &order.request.metadata["paper_execution"]["displayed_size_at_limit"]
            ),
            Some(dec!(20))
        );
        assert_eq!(
            decimal_from_json(
                &order.request.metadata["paper_execution"]["available_size_at_limit"]
            ),
            Some(dec!(10))
        );
        assert_eq!(
            decimal_from_json(
                &order.request.metadata["paper_execution"]["max_participating_size_at_limit"]
            ),
            Some(dec!(5))
        );
        assert_eq!(
            decimal_from_json(
                &order.request.metadata["paper_execution"]
                    ["requested_depth_participation_at_limit"]
            ),
            Some(dec!(0.25))
        );
        assert_eq!(
            decimal_from_json(
                &order.request.metadata["paper_execution"]["filled_depth_participation_at_limit"]
            ),
            Some(dec!(0.25))
        );
        assert_eq!(
            venue
                .fills_for_order(&order.order_id)
                .await
                .unwrap()
                .iter()
                .map(|fill| fill.size)
                .sum::<Decimal>(),
            dec!(5)
        );
    }

    #[tokio::test]
    async fn preview_uses_process_participation_cap_without_mutating_primary_state() {
        let venue = venue_with_participation_cap(
            registry_with_book(
                Utc::now(),
                vec![OrderbookLevel {
                    price: dec!(0.40),
                    size: dec!(10),
                }],
            ),
            Decimal::ONE,
            dec!(0.25),
        );
        let before = venue.status().await;

        let result = venue
            .preview_order(
                &request(dec!(3), dec!(0.40)),
                &PaperPreviewConfig {
                    scenario_key: "participation_cap".to_string(),
                    arrival_latency: Duration::ZERO,
                    visible_depth_haircut: Decimal::ONE,
                },
            )
            .await
            .unwrap();

        assert_eq!(result.state, OrderState::Rejected);
        assert_eq!(
            result.reject_reason.as_deref(),
            Some("arrival_depth_participation_exceeded")
        );
        assert_eq!(venue.status().await, before);
    }

    #[test]
    fn venue_validates_process_depth_participation_cap() {
        let registry = registry_with_book(Utc::now(), Vec::new());
        let config = PaperVenueConfig::default();
        for cap in [Decimal::ZERO, dec!(-0.01), dec!(1.01)] {
            let error = PaperVenue::new(registry.clone(), config.clone(), cap).unwrap_err();
            assert!(error
                .to_string()
                .contains("max_depth_participation must be in (0, 1]"));
        }
        assert!(PaperVenue::new(registry, config, Decimal::ONE).is_ok());
    }

    #[tokio::test]
    async fn retry_is_idempotent_and_fill_ids_are_deterministic() {
        let venue = venue(
            registry_with_book(
                Utc::now(),
                vec![OrderbookLevel {
                    price: dec!(0.40),
                    size: dec!(10),
                }],
            ),
            Decimal::ONE,
        );
        let request = request(dec!(3), dec!(0.40));
        let first = venue.submit_order(request.clone()).await.unwrap();
        let first_fills = venue.fills_for_order(&first.order_id).await.unwrap();
        let second = venue.submit_order(request).await.unwrap();
        let second_fills = venue.fills_for_order(&second.order_id).await.unwrap();
        assert_eq!(first.order_id, second.order_id);
        assert_eq!(first.state, second.state);
        assert_eq!(first_fills.len(), second_fills.len());
        assert_eq!(first_fills[0].fill_id, second_fills[0].fill_id);
        assert_eq!(first_fills.len(), 1);
        assert_eq!(
            first_fills[0].fill_id,
            deterministic_fill_id(&first.order_id, 0)
        );
        let balances = venue.get_balances().await.unwrap();
        assert_eq!(balances.len(), 1);
        assert!(balances[0].1 < dec!(100));
    }

    #[tokio::test]
    async fn fok_rejects_atomically_when_collateral_cannot_cover_notional_and_fee() {
        let registry = registry_with_book(
            Utc::now(),
            vec![OrderbookLevel {
                price: dec!(0.40),
                size: dec!(10),
            }],
        );
        let venue = PaperVenue::new(
            registry,
            PaperVenueConfig {
                arrival_latency: Duration::ZERO,
                visible_depth_haircut: Decimal::ONE,
                max_book_age: chrono::Duration::hours(24),
                starting_collateral_usd: dec!(1),
            },
            Decimal::ONE,
        )
        .unwrap();
        let order = venue
            .submit_order(request(dec!(3), dec!(0.40)))
            .await
            .unwrap();
        assert_eq!(order.state, OrderState::Rejected);
        assert_eq!(
            order.request.metadata["reject_reason"],
            serde_json::json!("insufficient_paper_collateral")
        );
        assert!(venue
            .fills_for_order(&order.order_id)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(venue.get_balances().await.unwrap()[0].1, dec!(1));
    }

    #[tokio::test]
    async fn missing_dynamic_fee_rate_fails_closed() {
        let venue = venue(
            registry_with_book(
                Utc::now(),
                vec![OrderbookLevel {
                    price: dec!(0.40),
                    size: dec!(10),
                }],
            ),
            Decimal::ONE,
        );
        let mut request = request(dec!(3), dec!(0.40));
        request.metadata = serde_json::json!({});
        let order = venue.submit_order(request).await.unwrap();
        assert_eq!(order.state, OrderState::Rejected);
        assert_eq!(
            order.request.metadata["reject_reason"],
            serde_json::json!("missing_dynamic_fee_rate")
        );
    }

    #[tokio::test]
    async fn stale_arrival_book_fails_closed() {
        let registry = registry_with_book(
            Utc::now() - ChronoDuration::seconds(5),
            vec![OrderbookLevel {
                price: dec!(0.40),
                size: dec!(10),
            }],
        );
        let venue = PaperVenue::new(
            registry,
            PaperVenueConfig {
                arrival_latency: Duration::ZERO,
                visible_depth_haircut: Decimal::ONE,
                max_book_age: chrono::Duration::seconds(2),
                starting_collateral_usd: dec!(100),
            },
            Decimal::ONE,
        )
        .unwrap();
        let order = venue
            .submit_order(request(dec!(3), dec!(0.40)))
            .await
            .unwrap();
        assert_eq!(order.state, OrderState::Rejected);
        assert_eq!(
            order.request.metadata["reject_reason"],
            serde_json::json!("stale_arrival_orderbook")
        );
    }

    #[tokio::test]
    async fn stale_source_with_fresh_receipt_fails_closed_without_accounting_mutation() {
        let now = Utc::now();
        let registry = registry_with_book_times(
            now - ChronoDuration::seconds(5),
            now - ChronoDuration::milliseconds(10),
            vec![OrderbookLevel {
                price: dec!(0.40),
                size: dec!(10),
            }],
        );
        let venue = PaperVenue::new(
            registry,
            PaperVenueConfig {
                arrival_latency: Duration::ZERO,
                visible_depth_haircut: Decimal::ONE,
                max_book_age: chrono::Duration::seconds(2),
                starting_collateral_usd: dec!(100),
            },
            Decimal::ONE,
        )
        .unwrap();
        let before = venue.status().await;

        let order = venue
            .submit_order(request(dec!(3), dec!(0.40)))
            .await
            .unwrap();
        let after = venue.status().await;

        assert_eq!(order.state, OrderState::Rejected);
        assert_eq!(
            order.request.metadata["reject_reason"],
            serde_json::json!("stale_arrival_orderbook")
        );
        assert_eq!(after.fill_count, before.fill_count);
        assert_eq!(
            after.available_collateral_usd,
            before.available_collateral_usd
        );
        assert_eq!(after.entry_debits_usd, before.entry_debits_usd);
    }

    #[tokio::test]
    async fn locally_received_book_allows_exchange_clock_lead() {
        let received_at = Utc::now() - ChronoDuration::milliseconds(10);
        let registry = registry_with_book_times(
            Utc::now() + ChronoDuration::seconds(5),
            received_at,
            vec![OrderbookLevel {
                price: dec!(0.40),
                size: dec!(10),
            }],
        );
        let order = venue(registry, Decimal::ONE)
            .submit_order(request(dec!(3), dec!(0.40)))
            .await
            .unwrap();
        assert_eq!(order.state, OrderState::Filled);
        assert!(
            order.request.metadata["paper_execution"]["arrival_source_age_ms"]
                .as_i64()
                .is_some_and(|age| age < 0)
        );
    }

    #[tokio::test]
    async fn source_timestamp_beyond_clock_lead_fails_closed() {
        let now = Utc::now();
        let registry = registry_with_book_times(
            now + ChronoDuration::seconds(5),
            now - ChronoDuration::milliseconds(10),
            vec![OrderbookLevel {
                price: dec!(0.40),
                size: dec!(10),
            }],
        );
        let venue = PaperVenue::new(
            registry,
            PaperVenueConfig {
                arrival_latency: Duration::ZERO,
                visible_depth_haircut: Decimal::ONE,
                max_book_age: chrono::Duration::seconds(2),
                starting_collateral_usd: dec!(100),
            },
            Decimal::ONE,
        )
        .unwrap();

        let order = venue
            .submit_order(request(dec!(3), dec!(0.40)))
            .await
            .unwrap();

        assert_eq!(order.state, OrderState::Rejected);
        assert_eq!(
            order.request.metadata["reject_reason"],
            serde_json::json!("future_arrival_orderbook")
        );
        assert_eq!(venue.status().await.fill_count, 0);
    }

    #[tokio::test]
    async fn locally_future_book_still_fails_closed() {
        let registry = registry_with_book_times(
            Utc::now(),
            Utc::now() + ChronoDuration::seconds(5),
            vec![OrderbookLevel {
                price: dec!(0.40),
                size: dec!(10),
            }],
        );
        let order = venue(registry, Decimal::ONE)
            .submit_order(request(dec!(3), dec!(0.40)))
            .await
            .unwrap();
        assert_eq!(order.state, OrderState::Rejected);
        assert_eq!(
            order.request.metadata["reject_reason"],
            serde_json::json!("future_arrival_orderbook")
        );
    }

    #[tokio::test]
    async fn delayed_arrival_preview_never_mutates_primary_paper_state() {
        let venue = venue(
            registry_with_book(
                Utc::now(),
                vec![OrderbookLevel {
                    price: dec!(0.40),
                    size: dec!(10),
                }],
            ),
            Decimal::ONE,
        );
        let before = venue.status().await;
        let result = venue
            .preview_order(
                &request(dec!(3), dec!(0.40)),
                &PaperPreviewConfig {
                    scenario_key: "latency_300ms_depth_65pct".to_string(),
                    arrival_latency: Duration::ZERO,
                    visible_depth_haircut: dec!(0.65),
                },
            )
            .await
            .unwrap();

        assert_eq!(result.state, OrderState::Filled);
        assert_eq!(result.filled_size, dec!(3));
        assert_eq!(
            result.execution_metadata["paper_execution"]["non_mutating_preview"],
            serde_json::json!(true)
        );
        assert_eq!(venue.status().await, before);
    }

    #[tokio::test]
    async fn resume_rehydrates_capital_counts_and_settlement_idempotency() {
        let venue = venue(registry_with_book(Utc::now(), Vec::new()), Decimal::ONE);
        let credited = Uuid::from_u128(776);
        venue
            .rehydrate_capital(dec!(12.50), dec!(8), 4, 3, [credited])
            .await
            .unwrap();

        let restored = venue.status().await;
        assert_eq!(restored.available_collateral_usd, dec!(95.50));
        assert_eq!(restored.entry_debits_usd, dec!(12.50));
        assert_eq!(restored.settlement_credits_usd, dec!(8));
        assert_eq!(restored.order_count, 4);
        assert_eq!(restored.fill_count, 3);
        assert_eq!(restored.settlements_applied, 1);

        let duplicate = venue
            .apply_settlement_credit(credited, dec!(8))
            .await
            .unwrap();
        assert!(!duplicate.newly_applied);
        assert_eq!(venue.status().await, restored);
        assert!(venue
            .rehydrate_capital(Decimal::ZERO, Decimal::ZERO, 0, 0, [])
            .await
            .is_err());
    }

    #[tokio::test]
    async fn official_settlement_credit_is_idempotent_and_recycles_collateral() {
        let venue = venue(
            registry_with_book(
                Utc::now(),
                vec![OrderbookLevel {
                    price: dec!(0.40),
                    size: dec!(10),
                }],
            ),
            Decimal::ONE,
        );
        let order = venue
            .submit_order(request(dec!(3), dec!(0.40)))
            .await
            .unwrap();
        assert_eq!(order.state, OrderState::Filled);
        let after_entry = venue.status().await;
        let settlement_id = Uuid::from_u128(777);

        let first = venue
            .apply_settlement_credit(settlement_id, dec!(3))
            .await
            .unwrap();
        let second = venue
            .apply_settlement_credit(settlement_id, dec!(3))
            .await
            .unwrap();
        let after_settlement = venue.status().await;

        assert!(first.newly_applied);
        assert!(!second.newly_applied);
        assert_eq!(first.payout_usd, dec!(3));
        assert_eq!(
            after_settlement.available_collateral_usd,
            after_entry.available_collateral_usd + dec!(3)
        );
        let net_pnl = dec!(3) - after_entry.entry_debits_usd;
        assert_eq!(
            after_settlement.available_collateral_usd,
            after_settlement.starting_collateral_usd + net_pnl
        );
        assert_eq!(after_settlement.settlement_credits_usd, dec!(3));
        assert_eq!(after_settlement.settlements_applied, 1);
    }

    #[tokio::test]
    async fn losing_official_settlement_credits_zero_and_keeps_entry_debit() {
        let venue = venue(
            registry_with_book(
                Utc::now(),
                vec![OrderbookLevel {
                    price: dec!(0.40),
                    size: dec!(10),
                }],
            ),
            Decimal::ONE,
        );
        let order = venue
            .submit_order(request(dec!(3), dec!(0.40)))
            .await
            .unwrap();
        assert_eq!(order.state, OrderState::Filled);
        let after_entry = venue.status().await;

        let credit = venue
            .apply_settlement_credit(Uuid::from_u128(779), Decimal::ZERO)
            .await
            .unwrap();
        let after_settlement = venue.status().await;

        assert!(credit.newly_applied);
        assert_eq!(credit.payout_usd, Decimal::ZERO);
        assert_eq!(
            after_settlement.available_collateral_usd,
            after_entry.available_collateral_usd
        );
        assert_eq!(after_settlement.settlement_credits_usd, Decimal::ZERO);
        assert!(after_settlement.entry_debits_usd > Decimal::ZERO);
        assert_eq!(after_settlement.settlements_applied, 1);
    }

    #[tokio::test]
    async fn official_settlement_credit_rejects_negative_payouts() {
        let venue = venue(registry_with_book(Utc::now(), Vec::new()), Decimal::ONE);
        let error = venue
            .apply_settlement_credit(Uuid::from_u128(778), dec!(-1))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("cannot be negative"));
        assert_eq!(venue.status().await.settlements_applied, 0);
    }
}
