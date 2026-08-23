use std::sync::Arc;

use anyhow::{bail, Result};
use async_trait::async_trait;
use chrono::{Duration, Utc};
use rust_decimal::Decimal;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

use crate::{
    account_reconcile::{AccountReconcileReport, AccountReconcileRequest},
    execution::{
        live_execution_gate_closed_order, ExecutionVenue, LiveExecutionGateReason,
        LiveIdentityDiagnostics, LiveOrderDryRunDiagnostics, LiveOrderDryRunRequest,
        LivePoly1271FunderProbeRequest, LivePoly1271FunderProbeResponse, LivePrePostGuard,
        LiveVenueStatus, LiveWalletAddressDiagnostics, ReconciliationReport,
    },
    models::{FillRecord, OrderRecord, OrderRequest, OrderSide},
};

use super::{
    execution_guard::{reference_execution_guard, BtcReferenceExecutionRejectReason},
    feeds::BookRegistry,
    types::{BookReadiness, FeedIntegrityStatus, OrderbookCheckpoint, OrderbookLevel},
};

/// Final, process-scoped safety boundary in front of a live execution venue.
///
/// Strategy selection, risk policy, accounting, and order construction remain on the shared BTC
/// process path. This adapter only revalidates immutable reference evidence and current executable
/// book liquidity immediately before the live venue receives an order.
#[derive(Clone)]
pub struct BtcLiveExecutionAdapter {
    delegate: Arc<dyn ExecutionVenue>,
    registry: Arc<RwLock<BookRegistry>>,
    expected_process_id: Uuid,
    max_reference_age: Duration,
    max_directional_feature_age: Option<Duration>,
    max_book_age: Duration,
    max_depth_participation: Decimal,
    require_exit_book: bool,
    submit_guard: Arc<Mutex<()>>,
}

impl crate::execution::live_pre_post_guard_sealed::Sealed for BtcLiveExecutionAdapter {}

#[async_trait]
impl LivePrePostGuard for BtcLiveExecutionAdapter {
    async fn validate_pre_post(
        &self,
        request: &OrderRequest,
    ) -> Result<Option<LiveExecutionGateReason>> {
        if let Some(reason) = self.validate_reference_execution(request)? {
            return Ok(Some(reason));
        }
        self.validate_current_market_pair(request).await
    }
}

impl BtcLiveExecutionAdapter {
    pub fn new(
        delegate: Arc<dyn ExecutionVenue>,
        registry: Arc<RwLock<BookRegistry>>,
        expected_process_id: Uuid,
        max_reference_age: Duration,
        max_directional_feature_age: Option<Duration>,
        max_book_age: Duration,
        max_depth_participation: Decimal,
        require_exit_book: bool,
    ) -> Result<Self> {
        if expected_process_id.is_nil() {
            bail!("BTC live execution expected_process_id must not be nil");
        }
        if max_reference_age <= Duration::zero() {
            bail!("BTC live execution max_reference_age must be positive");
        }
        if max_directional_feature_age.is_some_and(|max_age| max_age <= Duration::zero()) {
            bail!("BTC live execution max_directional_feature_age must be positive");
        }
        if max_book_age <= Duration::zero() {
            bail!("BTC live execution max_book_age must be positive");
        }
        if max_depth_participation <= Decimal::ZERO || max_depth_participation > Decimal::ONE {
            bail!("BTC live execution max_depth_participation must be in (0, 1]");
        }
        Ok(Self {
            delegate,
            registry,
            expected_process_id,
            max_reference_age,
            max_directional_feature_age,
            max_book_age,
            max_depth_participation,
            require_exit_book,
            submit_guard: Arc::new(Mutex::new(())),
        })
    }

    fn validate_reference_execution(
        &self,
        request: &OrderRequest,
    ) -> Result<Option<LiveExecutionGateReason>> {
        let checked_at = Utc::now();
        let guard = match reference_execution_guard(request) {
            Ok(guard) => guard,
            Err(reason) => return classify_reference_execution_rejection(reason),
        };
        match guard.validate_for_request(
            request,
            checked_at,
            self.expected_process_id,
            self.max_reference_age,
            self.max_directional_feature_age,
        ) {
            Ok(_) => Ok(None),
            Err(reason) => classify_reference_execution_rejection(reason),
        }
    }

    async fn validate_current_market_pair(
        &self,
        request: &OrderRequest,
    ) -> Result<Option<LiveExecutionGateReason>> {
        let (checked_at, connection_id, checkpoint, market_books) = {
            let registry = self.registry.read().await;
            let market_books = registry
                .book_readiness()
                .into_iter()
                .filter(|book| book.market_id == request.market_id)
                .collect::<Vec<_>>();
            let connection_id = registry.connection_id();
            let checkpoint = registry.checkpoint(&request.token_id);
            let checked_at = Utc::now();
            (checked_at, connection_id, checkpoint, market_books)
        };
        let Some(checkpoint) = checkpoint else {
            return Ok(Some(LiveExecutionGateReason::OrderbookReadiness));
        };
        if checkpoint.market_id != request.market_id || checkpoint.token_id != request.token_id {
            bail!("BTC live execution rejected: selected orderbook identity mismatch");
        }
        if checkpoint.integrity_status != FeedIntegrityStatus::Ok {
            if transient_book_unavailability(checkpoint.integrity_status) {
                return Ok(Some(LiveExecutionGateReason::OrderbookReadiness));
            }
            bail!(
                "BTC live execution rejected: selected orderbook integrity is {:?}",
                checkpoint.integrity_status
            );
        }
        if !book_timestamps_are_causal(
            checkpoint.source_timestamp,
            checkpoint.received_at,
            checked_at,
            self.max_book_age,
        ) {
            return Ok(Some(LiveExecutionGateReason::OrderbookFreshness));
        }
        if self.require_exit_book {
            if let Some(reason) = validate_market_pair_snapshot(
                request,
                connection_id,
                &market_books,
                &checkpoint,
                checked_at,
                self.max_book_age,
            )? {
                return Ok(Some(reason));
            }
        }
        validate_marketable_depth(&checkpoint, request, self.max_depth_participation)
    }
}

fn classify_reference_execution_rejection(
    reason: BtcReferenceExecutionRejectReason,
) -> Result<Option<LiveExecutionGateReason>> {
    if matches!(
        reason,
        BtcReferenceExecutionRejectReason::FutureEvidence
            | BtcReferenceExecutionRejectReason::StaleEvidence
    ) {
        return Ok(Some(LiveExecutionGateReason::ReferenceFreshness));
    }
    bail!("BTC live execution rejected: {}", reason.as_str())
}

fn transient_book_unavailability(status: FeedIntegrityStatus) -> bool {
    matches!(
        status,
        FeedIntegrityStatus::PreSnapshot | FeedIntegrityStatus::Stale
    )
}

fn book_timestamps_are_causal(
    source_timestamp: chrono::DateTime<Utc>,
    received_at: chrono::DateTime<Utc>,
    checked_at: chrono::DateTime<Utc>,
    max_book_age: Duration,
) -> bool {
    received_at <= checked_at
        && source_timestamp - checked_at <= max_book_age
        && received_at - source_timestamp <= max_book_age
}

fn validate_market_pair_snapshot(
    request: &OrderRequest,
    connection_id: Uuid,
    market_books: &[BookReadiness],
    selected_checkpoint: &OrderbookCheckpoint,
    checked_at: chrono::DateTime<Utc>,
    max_book_age: Duration,
) -> Result<Option<LiveExecutionGateReason>> {
    if connection_id.is_nil() {
        bail!("BTC live execution rejected: current orderbook connection epoch is nil");
    }
    if market_books.len() != 2 {
        return Ok(Some(LiveExecutionGateReason::OrderbookReadiness));
    }
    if market_books[0].token_id == market_books[1].token_id {
        bail!("BTC live execution rejected: current market orderbook pair identity mismatch");
    }

    for readiness in market_books {
        if readiness.market_id != request.market_id || readiness.token_id.trim().is_empty() {
            bail!("BTC live execution rejected: current market orderbook pair identity mismatch");
        }
        if readiness.connection_id != connection_id {
            bail!(
                "BTC live execution rejected: current market orderbook pair connection epoch mismatch"
            );
        }
        if !readiness.bootstrapped {
            return Ok(Some(LiveExecutionGateReason::OrderbookReadiness));
        }
        if readiness.integrity_status != FeedIntegrityStatus::Ok {
            if transient_book_unavailability(readiness.integrity_status) {
                return Ok(Some(LiveExecutionGateReason::OrderbookReadiness));
            }
            bail!(
                "BTC live execution rejected: current market orderbook pair integrity is {:?}",
                readiness.integrity_status
            );
        }
        let (Some(source_timestamp), Some(received_at)) =
            (readiness.source_timestamp, readiness.received_at)
        else {
            bail!(
                "BTC live execution rejected: current market orderbook pair is causally inconsistent"
            );
        };
        let (Some(best_bid), Some(best_ask)) = (readiness.best_bid, readiness.best_ask) else {
            return Ok(Some(LiveExecutionGateReason::OrderbookMarketability));
        };
        if best_bid >= best_ask {
            bail!("BTC live execution rejected: current market orderbook pair is crossed");
        }
        if !book_timestamps_are_causal(source_timestamp, received_at, checked_at, max_book_age) {
            return Ok(Some(LiveExecutionGateReason::OrderbookFreshness));
        }
    }

    let Some(selected_readiness) = market_books
        .iter()
        .find(|book| book.token_id == request.token_id)
    else {
        bail!("BTC live execution rejected: current market orderbook pair identity mismatch");
    };
    if selected_checkpoint.market_id != selected_readiness.market_id
        || selected_checkpoint.token_id != selected_readiness.token_id
        || selected_checkpoint.connection_id != connection_id
        || selected_checkpoint.integrity_status != selected_readiness.integrity_status
        || selected_readiness.source_timestamp != Some(selected_checkpoint.source_timestamp)
        || selected_readiness.received_at != Some(selected_checkpoint.received_at)
        || selected_readiness.best_bid != selected_checkpoint.best_bid
        || selected_readiness.best_ask != selected_checkpoint.best_ask
        || selected_checkpoint.ingest_sequence == 0
    {
        bail!(
            "BTC live execution rejected: current market orderbook pair is causally inconsistent"
        );
    }
    Ok(None)
}

fn validate_marketable_depth(
    checkpoint: &OrderbookCheckpoint,
    request: &OrderRequest,
    max_depth_participation: Decimal,
) -> Result<Option<LiveExecutionGateReason>> {
    let (levels, marketable): (&[OrderbookLevel], fn(Decimal, Decimal) -> bool) = match request.side
    {
        OrderSide::Buy => (&checkpoint.asks, |level_price, limit_price| {
            level_price <= limit_price
        }),
        OrderSide::Sell => (&checkpoint.bids, |level_price, limit_price| {
            level_price >= limit_price
        }),
    };
    let mut displayed_depth = Decimal::ZERO;
    for level in levels {
        if level.price <= Decimal::ZERO
            || level.price >= Decimal::ONE
            || level.size <= Decimal::ZERO
        {
            bail!("BTC live execution rejected: selected orderbook contains an invalid level");
        }
        if marketable(level.price, request.price) {
            displayed_depth = displayed_depth.checked_add(level.size).ok_or_else(|| {
                anyhow::anyhow!("BTC live execution rejected: selected orderbook depth overflow")
            })?;
        }
    }
    let permitted_depth = displayed_depth
        .checked_mul(max_depth_participation)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "BTC live execution rejected: selected orderbook participation overflow"
            )
        })?;
    if request.size > permitted_depth {
        return Ok(Some(LiveExecutionGateReason::OrderbookMarketability));
    }
    Ok(None)
}

#[async_trait]
impl ExecutionVenue for BtcLiveExecutionAdapter {
    fn preserve_liveness_on_post_order_reconcile_error(&self) -> bool {
        true
    }

    async fn find_existing_order(&self, request: &OrderRequest) -> Result<Option<OrderRecord>> {
        self.delegate.find_existing_order(request).await
    }

    async fn submit_order(&self, request: OrderRequest) -> Result<OrderRecord> {
        let _submit_guard = self.submit_guard.lock().await;
        if let Some(existing) = self.delegate.find_existing_order(&request).await? {
            return Ok(existing);
        }
        if let Some(reason) = self.validate_reference_execution(&request)? {
            return live_execution_gate_closed_order(request, reason);
        }
        if let Some(reason) = self.validate_current_market_pair(&request).await? {
            return live_execution_gate_closed_order(request, reason);
        }
        self.delegate
            .submit_order_with_pre_post_guard(request, Some(Arc::new(self.clone())))
            .await
    }

    async fn cancel_order(&self, order_id: &str) -> Result<OrderRecord> {
        self.delegate.cancel_order(order_id).await
    }

    async fn cancel_all(&self) -> Result<usize> {
        self.delegate.cancel_all().await
    }

    async fn get_balances(&self) -> Result<Vec<(String, Decimal)>> {
        self.delegate.get_balances().await
    }

    async fn get_open_orders(&self) -> Result<Vec<OrderRecord>> {
        self.delegate.get_open_orders().await
    }

    async fn reconcile(&self) -> Result<ReconciliationReport> {
        self.delegate.reconcile().await
    }

    async fn update_live_reconciliation_health(
        &self,
        pending_settlement_count: usize,
        error: Option<String>,
    ) -> Result<()> {
        self.delegate
            .update_live_reconciliation_health(pending_settlement_count, error)
            .await
    }

    async fn fills_for_order(&self, order_id: &str) -> Result<Vec<FillRecord>> {
        self.delegate.fills_for_order(order_id).await
    }

    async fn live_status(&self) -> Result<LiveVenueStatus> {
        self.delegate.live_status().await
    }

    async fn live_identity_diagnostics(&self) -> Result<LiveIdentityDiagnostics> {
        self.delegate.live_identity_diagnostics().await
    }

    async fn live_wallet_address_diagnostics(
        &self,
        candidate_addresses: Vec<String>,
    ) -> Result<LiveWalletAddressDiagnostics> {
        self.delegate
            .live_wallet_address_diagnostics(candidate_addresses)
            .await
    }

    async fn live_order_dry_run(
        &self,
        request: LiveOrderDryRunRequest,
    ) -> Result<LiveOrderDryRunDiagnostics> {
        self.delegate.live_order_dry_run(request).await
    }

    async fn live_poly1271_funder_probe(
        &self,
        request: LivePoly1271FunderProbeRequest,
    ) -> Result<LivePoly1271FunderProbeResponse> {
        self.delegate.live_poly1271_funder_probe(request).await
    }

    async fn live_account_reconcile(
        &self,
        request: AccountReconcileRequest,
    ) -> Result<AccountReconcileReport> {
        self.delegate.live_account_reconcile(request).await
    }

    async fn set_live_entries_enabled(
        &self,
        enabled: bool,
        reason: Option<String>,
    ) -> Result<LiveVenueStatus> {
        self.delegate
            .set_live_entries_enabled(enabled, reason)
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex as StdMutex,
    };

    use chrono::DateTime;
    use rust_decimal_macros::dec;
    use tokio::sync::Notify;

    use super::*;
    use crate::{
        btc::{
            execution_guard::{BtcReferenceExecutionGuard, BTC_REFERENCE_EXECUTION_GUARD_VERSION},
            feeds::ClobMessage,
            strategy::BTC_FEATURE_LINEAGE_VERSION,
            types::{BtcIntervalMarket, BtcOutcome},
        },
        models::{OrderState, OrderType},
    };

    const MAX_REFERENCE_AGE_MS: i64 = 30_000;

    struct FakeVenue {
        delegate_calls: AtomicUsize,
        post_calls: AtomicUsize,
        gate_reason: StdMutex<Option<LiveExecutionGateReason>>,
        existing_order: StdMutex<Option<OrderRecord>>,
        pause_before_pre_post: std::sync::atomic::AtomicBool,
        preparation_started: Notify,
        preparation_continue: Notify,
    }

    impl Default for FakeVenue {
        fn default() -> Self {
            Self {
                delegate_calls: AtomicUsize::new(0),
                post_calls: AtomicUsize::new(0),
                gate_reason: StdMutex::new(None),
                existing_order: StdMutex::new(None),
                pause_before_pre_post: std::sync::atomic::AtomicBool::new(false),
                preparation_started: Notify::new(),
                preparation_continue: Notify::new(),
            }
        }
    }

    impl FakeVenue {
        fn submit_calls(&self) -> usize {
            self.post_calls.load(Ordering::SeqCst)
        }

        fn delegate_calls(&self) -> usize {
            self.delegate_calls.load(Ordering::SeqCst)
        }

        fn close_gate(&self, reason: LiveExecutionGateReason) {
            *self.gate_reason.lock().unwrap() = Some(reason);
        }

        fn open_gate(&self) {
            *self.gate_reason.lock().unwrap() = None;
        }

        fn pause_before_pre_post(&self) {
            self.pause_before_pre_post.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn live_adapter_preserves_runtime_liveness_on_post_order_reconciliation_failure() {
        let fake = Arc::new(FakeVenue::default());
        let process_id = Uuid::new_v4();
        let now = Utc::now();
        let adapter = adapter(
            &fake,
            seeded_registry("market", "token", now, now, dec!(10)),
            process_id,
        );
        assert!(adapter.preserve_liveness_on_post_order_reconcile_error());
    }

    #[async_trait]
    impl ExecutionVenue for FakeVenue {
        async fn find_existing_order(&self, request: &OrderRequest) -> Result<Option<OrderRecord>> {
            Ok(self
                .existing_order
                .lock()
                .unwrap()
                .as_ref()
                .filter(|order| order.request.client_order_id == request.client_order_id)
                .cloned())
        }

        async fn submit_order(&self, request: OrderRequest) -> Result<OrderRecord> {
            self.delegate_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(reason) = *self.gate_reason.lock().unwrap() {
                let order = live_execution_gate_closed_order(request, reason)?;
                *self.existing_order.lock().unwrap() = Some(order.clone());
                return Ok(order);
            }
            self.post_calls.fetch_add(1, Ordering::SeqCst);
            let now = Utc::now();
            let order = OrderRecord {
                order_id: "delegated-order".to_string(),
                request,
                state: OrderState::Submitted,
                created_at: now,
                updated_at: now,
            };
            *self.existing_order.lock().unwrap() = Some(order.clone());
            Ok(order)
        }

        async fn submit_order_with_pre_post_guard(
            &self,
            request: OrderRequest,
            guard: Option<Arc<dyn LivePrePostGuard>>,
        ) -> Result<OrderRecord> {
            if self.pause_before_pre_post.load(Ordering::SeqCst) {
                self.preparation_started.notify_one();
                self.preparation_continue.notified().await;
            }
            if let Some(guard) = guard {
                if let Some(reason) = guard.validate_pre_post(&request).await? {
                    let order = live_execution_gate_closed_order(request, reason)?;
                    *self.existing_order.lock().unwrap() = Some(order.clone());
                    return Ok(order);
                }
            }
            self.submit_order(request).await
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
            bail!("not exercised")
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

    fn market(market_id: &str, token_id: &str) -> BtcIntervalMarket {
        let now = Utc::now();
        BtcIntervalMarket {
            event_id: format!("event-{market_id}"),
            event_slug: "btc-updown-5m-1783902600".to_string(),
            series_slug: "btc-up-or-down-5m".to_string(),
            market_id: market_id.to_string(),
            condition_id: format!("condition-{market_id}"),
            window_start: now - Duration::minutes(1),
            window_end: now + Duration::minutes(4),
            up_token_id: token_id.to_string(),
            down_token_id: format!("{token_id}-down"),
            tick_size: dec!(0.01),
            minimum_order_size: Some(dec!(1)),
            resolution_source: "https://data.chain.link/streams/btc-usd".to_string(),
            active: true,
            closed: false,
            accepting_orders: true,
            fees_enabled: true,
            fee_schedule: serde_json::json!({}),
            raw_payload: serde_json::json!({}),
        }
    }

    fn seeded_registry(
        market_id: &str,
        token_id: &str,
        source_timestamp: DateTime<Utc>,
        received_at: DateTime<Utc>,
        ask_size: Decimal,
    ) -> Arc<RwLock<BookRegistry>> {
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market(market_id, token_id));
        seed_book(
            &mut registry,
            market_id,
            token_id,
            source_timestamp,
            received_at,
            dec!(0.39),
            dec!(0.40),
            ask_size,
        );
        seed_book(
            &mut registry,
            market_id,
            &format!("{token_id}-down"),
            source_timestamp,
            received_at,
            dec!(0.59),
            dec!(0.60),
            dec!(10),
        );
        Arc::new(RwLock::new(registry))
    }

    #[allow(clippy::too_many_arguments)]
    fn seed_book(
        registry: &mut BookRegistry,
        market_id: &str,
        token_id: &str,
        source_timestamp: DateTime<Utc>,
        received_at: DateTime<Utc>,
        bid_price: Decimal,
        ask_price: Decimal,
        ask_size: Decimal,
    ) {
        registry.apply(
            ClobMessage::Book {
                market_id: market_id.to_string(),
                token_id: token_id.to_string(),
                bids: vec![OrderbookLevel {
                    price: bid_price,
                    size: dec!(10),
                }],
                asks: vec![OrderbookLevel {
                    price: ask_price,
                    size: ask_size,
                }],
                source_timestamp,
                source_hash: Some(format!("book-hash-{token_id}")),
            },
            received_at,
        );
    }

    fn guarded_request(
        feature_as_of: DateTime<Utc>,
        process_id: Uuid,
        market_id: &str,
        token_id: &str,
        size: Decimal,
    ) -> OrderRequest {
        let tick = |id: u128, age_ms: i64| {
            serde_json::json!({
                "tick_id": Uuid::from_u128(id),
                "source_timestamp": feature_as_of - Duration::milliseconds(age_ms),
                "received_at": feature_as_of - Duration::milliseconds(age_ms - 10),
                "ingest_sequence": id as u64,
            })
        };
        let client_order_id = Uuid::new_v4();
        let intent_id = Uuid::new_v4();
        let decision_id = Uuid::new_v4();
        let snapshot_id = Uuid::new_v4();
        let mut guard: BtcReferenceExecutionGuard = serde_json::from_value(serde_json::json!({
            "guard_version": BTC_REFERENCE_EXECUTION_GUARD_VERSION,
            "process_id": process_id,
            "intent_id": intent_id,
            "decision_id": decision_id,
            "decision_at": feature_as_of,
            "snapshot_id": snapshot_id,
            "feature_as_of": feature_as_of,
            "market_id": market_id,
            "token_id": token_id,
            "outcome": BtcOutcome::Up,
            "strategy_version": "strategy-v1",
            "feature_schema_version": "features-v1",
            "lineage_version": BTC_FEATURE_LINEAGE_VERSION,
            "feature_sha256": "a".repeat(64),
            "client_order_id": client_order_id,
            "side": OrderSide::Buy,
            "order_type": OrderType::Fok,
            "limit_price": dec!(0.40),
            "size": size,
            "signal_id": null,
            "dynamic_fee_rate": dec!(0.25),
            "chainlink_open": tick(5, 1_000),
            "chainlink": tick(6, 100),
            "binance": tick(7, 90),
            "directional_model": null,
            "selected_book": null,
            "max_reference_age_ms": MAX_REFERENCE_AGE_MS,
            "evidence_sha256": "",
        }))
        .unwrap();
        guard.reseal_for_test();
        let mut metadata = serde_json::json!({
            "execution_intent": "entry",
            "process_id": process_id,
            "intent_id": intent_id,
            "decision_id": decision_id,
            "feature_snapshot_id": snapshot_id,
            "strategy_version": "strategy-v1",
            "feature_schema_version": "features-v1",
            "outcome": "up",
            "dynamic_fee_rate": dec!(0.25),
        });
        guard.insert_into_metadata(&mut metadata).unwrap();
        OrderRequest {
            client_order_id,
            process_id: Some(process_id),
            market_id: market_id.to_string(),
            token_id: token_id.to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.40),
            size,
            metadata,
        }
    }

    fn adapter(
        fake: &Arc<FakeVenue>,
        registry: Arc<RwLock<BookRegistry>>,
        process_id: Uuid,
    ) -> BtcLiveExecutionAdapter {
        let delegate: Arc<dyn ExecutionVenue> = fake.clone();
        BtcLiveExecutionAdapter::new(
            delegate,
            registry,
            process_id,
            Duration::milliseconds(MAX_REFERENCE_AGE_MS),
            None,
            Duration::seconds(2),
            dec!(0.50),
            true,
        )
        .unwrap()
    }

    fn assert_gate_rejection(order: &OrderRecord, reason: LiveExecutionGateReason) {
        assert_eq!(order.state, OrderState::Rejected);
        assert_eq!(
            order.request.metadata["reject_reason"],
            crate::execution::LIVE_EXECUTION_GATE_CLOSED_REASON
        );
        assert_eq!(
            order.request.metadata["live_execution_gate"]["gate_reason"],
            reason.as_str()
        );
        assert_eq!(
            order.request.metadata["live_execution_gate"]["post_attempted"],
            false
        );
    }

    #[tokio::test]
    async fn missing_book_is_rejected_before_delegate_submission() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let venue = adapter(
            &fake,
            Arc::new(RwLock::new(BookRegistry::new(Uuid::new_v4()))),
            process_id,
        );

        let order = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();

        assert_gate_rejection(&order, LiveExecutionGateReason::OrderbookReadiness);
        assert_eq!(fake.submit_calls(), 0);
    }

    #[tokio::test]
    async fn locally_received_market_pair_allows_bounded_exchange_clock_lead() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let venue = adapter(
            &fake,
            seeded_registry(
                "market",
                "up",
                checked_at + Duration::seconds(1),
                checked_at - Duration::milliseconds(10),
                dec!(10),
            ),
            process_id,
        );

        let order = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();

        assert_eq!(order.state, OrderState::Submitted);
        assert_eq!(fake.delegate_calls(), 1);
        assert_eq!(fake.submit_calls(), 1);
    }

    #[tokio::test]
    async fn excessive_exchange_clock_lead_is_rejected_before_delegate_submission() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let venue = adapter(
            &fake,
            seeded_registry(
                "market",
                "up",
                checked_at + Duration::seconds(3),
                checked_at - Duration::milliseconds(10),
                dec!(10),
            ),
            process_id,
        );

        let order = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();

        assert_gate_rejection(&order, LiveExecutionGateReason::OrderbookFreshness);
        assert_eq!(fake.delegate_calls(), 0);
        assert_eq!(fake.submit_calls(), 0);
    }

    #[tokio::test]
    async fn locally_future_book_receipt_is_rejected_before_delegate_submission() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let venue = adapter(
            &fake,
            seeded_registry(
                "market",
                "up",
                checked_at - Duration::milliseconds(10),
                checked_at + Duration::seconds(3),
                dec!(10),
            ),
            process_id,
        );

        let order = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();

        assert_gate_rejection(&order, LiveExecutionGateReason::OrderbookFreshness);
        assert_eq!(fake.delegate_calls(), 0);
        assert_eq!(fake.submit_calls(), 0);
    }

    #[tokio::test]
    async fn unchanged_book_on_active_connection_is_delegated() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let stale_at = checked_at - Duration::seconds(10);
        let venue = adapter(
            &fake,
            seeded_registry("market", "up", stale_at, stale_at, dec!(10)),
            process_id,
        );

        let order = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();

        assert_eq!(order.state, OrderState::Submitted);
        assert_eq!(fake.submit_calls(), 1);
    }

    #[tokio::test]
    async fn missing_opposite_book_is_rejected_before_delegate_submission() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let mut registry = BookRegistry::new(Uuid::new_v4());
        registry.register_market(&market("market", "up"));
        seed_book(
            &mut registry,
            "market",
            "up",
            checked_at - Duration::milliseconds(100),
            checked_at,
            dec!(0.39),
            dec!(0.40),
            dec!(10),
        );
        let venue = adapter(&fake, Arc::new(RwLock::new(registry)), process_id);

        let order = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();

        assert_gate_rejection(&order, LiveExecutionGateReason::OrderbookReadiness);
        assert_eq!(fake.submit_calls(), 0);
    }

    #[tokio::test]
    async fn delayed_opposite_book_frame_is_rejected_before_delegate_submission() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let registry = seeded_registry(
            "market",
            "up",
            checked_at - Duration::milliseconds(100),
            checked_at,
            dec!(10),
        );
        let stale_at = checked_at - Duration::seconds(10);
        {
            let mut registry = registry.write().await;
            seed_book(
                &mut registry,
                "market",
                "up-down",
                stale_at,
                checked_at,
                dec!(0.59),
                dec!(0.60),
                dec!(10),
            );
        }
        let venue = adapter(&fake, registry, process_id);

        let order = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();

        assert_gate_rejection(&order, LiveExecutionGateReason::OrderbookFreshness);
        assert_eq!(fake.submit_calls(), 0);
    }

    #[tokio::test]
    async fn unchanged_opposite_book_receipt_remains_usable() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let registry = seeded_registry(
            "market",
            "up",
            checked_at - Duration::milliseconds(100),
            checked_at,
            dec!(10),
        );
        let stale_at = checked_at - Duration::seconds(10);
        {
            let mut registry = registry.write().await;
            seed_book(
                &mut registry,
                "market",
                "up-down",
                checked_at - Duration::milliseconds(100),
                stale_at,
                dec!(0.59),
                dec!(0.60),
                dec!(10),
            );
        }
        let venue = adapter(&fake, registry, process_id);

        let order = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();

        assert_eq!(order.state, OrderState::Submitted);
        assert_eq!(fake.submit_calls(), 1);
    }

    #[tokio::test]
    async fn quarantined_opposite_book_is_rejected_before_delegate_submission() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let registry = seeded_registry(
            "market",
            "up",
            checked_at - Duration::milliseconds(100),
            checked_at,
            dec!(10),
        );
        {
            let mut registry = registry.write().await;
            seed_book(
                &mut registry,
                "market",
                "up-down",
                checked_at - Duration::milliseconds(50),
                checked_at,
                dec!(0.61),
                dec!(0.60),
                dec!(10),
            );
        }
        let venue = adapter(&fake, registry, process_id);

        let error = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("integrity is CrossedBook"));
        assert_eq!(fake.submit_calls(), 0);
    }

    #[tokio::test]
    async fn mismatched_market_is_rejected_before_delegate_submission() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let venue = adapter(
            &fake,
            seeded_registry(
                "book-market",
                "up",
                checked_at - Duration::milliseconds(100),
                checked_at,
                dec!(10),
            ),
            process_id,
        );

        let error = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "request-market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("identity mismatch"));
        assert_eq!(fake.submit_calls(), 0);
    }

    #[tokio::test]
    async fn over_participation_depth_is_rejected_before_delegate_submission() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let venue = adapter(
            &fake,
            seeded_registry(
                "market",
                "up",
                checked_at - Duration::milliseconds(100),
                checked_at,
                dec!(3),
            ),
            process_id,
        );

        let order = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();

        assert_gate_rejection(&order, LiveExecutionGateReason::OrderbookMarketability);
        assert_eq!(fake.submit_calls(), 0);
    }

    #[tokio::test]
    async fn stale_reference_is_a_nonfatal_zero_post_gate_rejection() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let venue = adapter(
            &fake,
            seeded_registry(
                "market",
                "up",
                checked_at - Duration::milliseconds(100),
                checked_at,
                dec!(10),
            ),
            process_id,
        );

        let order = venue
            .submit_order(guarded_request(
                checked_at - Duration::seconds(31),
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();

        assert_gate_rejection(&order, LiveExecutionGateReason::ReferenceFreshness);
        assert_eq!(fake.delegate_calls(), 0);
        assert_eq!(fake.submit_calls(), 0);
    }

    #[tokio::test]
    async fn manual_gate_rejection_is_zero_post_and_a_later_intent_can_submit() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        fake.close_gate(LiveExecutionGateReason::ManualEnableRequired);
        let venue = adapter(
            &fake,
            seeded_registry(
                "market",
                "up",
                checked_at - Duration::milliseconds(100),
                checked_at,
                dec!(10),
            ),
            process_id,
        );

        let rejected = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();
        assert_gate_rejection(&rejected, LiveExecutionGateReason::ManualEnableRequired);
        assert_eq!(fake.delegate_calls(), 1);
        assert_eq!(fake.submit_calls(), 0);

        fake.open_gate();
        let submitted = venue
            .submit_order(guarded_request(
                Utc::now(),
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();

        assert_eq!(submitted.state, OrderState::Submitted);
        assert_eq!(fake.delegate_calls(), 2);
        assert_eq!(fake.submit_calls(), 1);
    }

    #[tokio::test]
    async fn risk_denial_from_delegate_is_a_nonfatal_zero_post_record() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        fake.close_gate(LiveExecutionGateReason::OpenNotionalLimit);
        let venue = adapter(
            &fake,
            seeded_registry(
                "market",
                "up",
                checked_at - Duration::milliseconds(100),
                checked_at,
                dec!(10),
            ),
            process_id,
        );

        let order = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();

        assert_gate_rejection(&order, LiveExecutionGateReason::OpenNotionalLimit);
        assert_eq!(fake.delegate_calls(), 1);
        assert_eq!(fake.submit_calls(), 0);
    }

    #[tokio::test]
    async fn duplicate_order_precedes_changed_gates_and_books_without_a_second_post() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let registry = seeded_registry(
            "market",
            "up",
            checked_at - Duration::milliseconds(100),
            checked_at,
            dec!(10),
        );
        let venue = adapter(&fake, registry.clone(), process_id);
        let request = guarded_request(checked_at, process_id, "market", "up", dec!(2));

        let first = venue.submit_order(request.clone()).await.unwrap();
        assert_eq!(first.state, OrderState::Submitted);
        assert_eq!(fake.delegate_calls(), 1);
        assert_eq!(fake.submit_calls(), 1);

        fake.close_gate(LiveExecutionGateReason::GlobalHalt);
        registry
            .write()
            .await
            .quarantine(FeedIntegrityStatus::Stale);
        let duplicate = venue.submit_order(request).await.unwrap();

        assert_eq!(duplicate.order_id, first.order_id);
        assert_eq!(duplicate.state, first.state);
        assert_eq!(
            duplicate.request.client_order_id,
            first.request.client_order_id
        );
        assert_eq!(duplicate.request.metadata, first.request.metadata);
        assert_eq!(fake.delegate_calls(), 1);
        assert_eq!(fake.submit_calls(), 1);
    }

    #[tokio::test]
    async fn book_quarantined_during_preparation_is_rejected_by_adjacent_guard_without_post() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        fake.pause_before_pre_post();
        let registry = seeded_registry(
            "market",
            "up",
            checked_at - Duration::milliseconds(100),
            checked_at,
            dec!(10),
        );
        let venue = adapter(&fake, registry.clone(), process_id);
        let request = guarded_request(checked_at, process_id, "market", "up", dec!(2));

        let submission = tokio::spawn(async move { venue.submit_order(request).await });
        fake.preparation_started.notified().await;
        registry
            .write()
            .await
            .quarantine(FeedIntegrityStatus::Stale);
        fake.preparation_continue.notify_one();
        let order = submission.await.unwrap().unwrap();

        assert_gate_rejection(&order, LiveExecutionGateReason::OrderbookReadiness);
        assert_eq!(fake.delegate_calls(), 0);
        assert_eq!(fake.submit_calls(), 0);
    }

    #[tokio::test]
    async fn valid_order_is_delegated_exactly_once() {
        let checked_at = Utc::now();
        let process_id = Uuid::new_v4();
        let fake = Arc::new(FakeVenue::default());
        let venue = adapter(
            &fake,
            seeded_registry(
                "market",
                "up",
                checked_at - Duration::milliseconds(100),
                checked_at,
                dec!(5),
            ),
            process_id,
        );

        let order = venue
            .submit_order(guarded_request(
                checked_at,
                process_id,
                "market",
                "up",
                dec!(2),
            ))
            .await
            .unwrap();

        assert_eq!(order.order_id, "delegated-order");
        assert_eq!(fake.submit_calls(), 1);
    }
}
