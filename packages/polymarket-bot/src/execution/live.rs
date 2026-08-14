use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
    sync::Arc,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use polymarket_client_sdk_v2::{
    auth::{state::Authenticated, Credentials, LocalSigner, Normal, Signer as _},
    clob::{
        types::{
            request::{
                BalanceAllowanceRequest, OrdersRequest, TradesRequest,
                UpdateBalanceAllowanceRequest,
            },
            response::{OpenOrderResponse, PostOrderResponse, TradeResponse},
            AssetType, OrderStatusType, OrderType as SdkOrderType, Side as SdkSide, SignatureType,
            TradeStatusType,
        },
        Client as SdkClient, Config as SdkConfig,
    },
    derive_proxy_wallet, derive_safe_wallet,
    error::{Kind as SdkErrorKind, Status as SdkStatus},
    types::{Address, Decimal as SdkDecimal, U256},
    POLYGON,
};
use rust_decimal::Decimal;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::{
    account_reconcile::{
        account_trade_from_live_event, reconcile_account_positions, AccountReconcileReport,
        AccountReconcileRequest,
    },
    config::LiveExecutionConfig,
    data_api::DataApiClient,
    execution::{
        live_execution_gate_closed_order, ExecutionVenue, LiveExecutionGateReason,
        LiveIdentityDiagnostics, LiveOrderDryRunDiagnostics, LiveOrderDryRunRequest,
        LivePoly1271FunderProbeCandidate, LivePoly1271FunderProbeRequest,
        LivePoly1271FunderProbeResponse, LivePrePostGuard, LiveVenueStatus,
        LiveWalletAddressDiagnostics, LiveWalletCandidateAddressDiagnostics,
        LiveWalletTokenBalances, ReconciliationReport,
    },
    idempotency::event_hash,
    models::{EffectiveProcessExecutionConfig, FillRecord, OrderRecord, OrderRequest},
    models::{FillSource, OrderSide, OrderState, OrderType},
    store::Store,
};

type AuthenticatedClient = SdkClient<Authenticated<Normal>>;

const DEFAULT_POLYGON_RPC_URL: &str = "https://polygon-bor-rpc.publicnode.com";
const DEFAULT_RELAYER_BASE_URL: &str = "https://relayer-v2.polymarket.com";
const PUSD_ADDRESS: &str = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB";
const CTF_EXCHANGE_V2_ADDRESS: &str = "0xE111180000d2663C0091e4f400237545B87B996B";
const NEG_RISK_CTF_EXCHANGE_V2_ADDRESS: &str = "0xe2222d279d744050d28e00520010520000310F59";
const USDC_E_ADDRESS: &str = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174";
const NATIVE_USDC_ADDRESS: &str = "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359";
const MAX_PROCESS_NONTERMINAL_ORDERS: usize = 256;
const CLOB_TERMINAL_CURSOR: &str = "LTE=";
const MAX_CLOB_RECONCILIATION_PAGES: usize = 32;
const MAX_CLOB_RECONCILIATION_ROWS: usize = 4_096;
const MAX_CLOB_RECONCILIATION_ORDER_IDS: usize = 8_192;
const MAX_CLOB_CURSOR_BYTES: usize = 256;
const CLOB_ORDER_ID_QUERY_CHUNK: usize = 500;
const FOK_FILL_RECONCILIATION_SKEW: chrono::Duration = chrono::Duration::hours(1);
const USER_WS_MAX_TRANSPORT_OPERATION_TIMEOUT: Duration = Duration::from_secs(10);
const USER_WS_RECONNECT_INITIAL_DELAY: Duration = Duration::from_secs(1);
const USER_WS_RECONNECT_MAX_DELAY: Duration = Duration::from_secs(30);

#[derive(Debug, Deserialize)]
struct RpcResponse {
    result: Option<String>,
    error: Option<Value>,
}

#[derive(Clone)]
pub struct LiveVenue {
    config: LiveExecutionConfig,
    clob_base_url: String,
    store: Option<Store>,
    data_api: Option<DataApiClient>,
    bound_process_id: Option<Uuid>,
    bound_account_ref: Option<String>,
    bound_execution: Option<EffectiveProcessExecutionConfig>,
    transport_state: Arc<Mutex<LiveTransportState>>,
    readiness_state: Arc<Mutex<LiveVenueState>>,
    global_entry_gate: Arc<Mutex<GlobalLiveEntryGate>>,
    submit_guard: Arc<Mutex<()>>,
}

#[derive(Debug, Clone)]
struct LiveTransportState {
    live_confirmed: bool,
    geoblock_readable: bool,
    geoblock_blocked: Option<bool>,
    geoblock_country: Option<String>,
    geoblock_region: Option<String>,
    geoblock_checked_at: Option<DateTime<Utc>>,
    user_ws_connected: bool,
    last_user_ws_pong_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
struct LiveVenueState {
    last_rest_reconcile_at: Option<DateTime<Utc>>,
    idempotency_clean: bool,
    unresolved_live_order_count: usize,
    manual_entries_enabled: bool,
    manual_entries_reason: Option<String>,
    process_accounting_proven: bool,
    process_accounting_status: String,
    credential_account_fingerprint_sha256: Option<String>,
    reconciled_safety_generation: Option<u64>,
}

#[derive(Debug, Clone)]
struct CanonicalLiveAccountIdentity {
    account_address: String,
    signer_address: String,
    signature_type: SignatureType,
    fingerprint_sha256: String,
}

#[derive(Debug, Clone)]
struct GlobalLiveEntryGate {
    halted: bool,
    reason: String,
    safety_generation: u64,
}

impl LiveTransportState {
    fn initial() -> Self {
        Self {
            live_confirmed: false,
            geoblock_readable: false,
            geoblock_blocked: None,
            geoblock_country: None,
            geoblock_region: None,
            geoblock_checked_at: None,
            user_ws_connected: false,
            last_user_ws_pong_at: None,
        }
    }
}

impl LiveVenueState {
    fn fail_closed() -> Self {
        Self {
            last_rest_reconcile_at: None,
            idempotency_clean: false,
            unresolved_live_order_count: 0,
            manual_entries_enabled: false,
            manual_entries_reason: Some("manual_enable_required".to_string()),
            process_accounting_proven: false,
            process_accounting_status: "unproven".to_string(),
            credential_account_fingerprint_sha256: None,
            reconciled_safety_generation: None,
        }
    }
}

impl GlobalLiveEntryGate {
    fn fail_closed() -> Self {
        Self {
            halted: true,
            reason: "global_enable_required".to_string(),
            safety_generation: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LiveVenueEvent {
    pub source: String,
    pub event_type: String,
    pub venue_event_id: Option<String>,
    pub venue_order_id: Option<String>,
    pub venue_trade_id: Option<String>,
    pub event_status: Option<String>,
    pub raw_payload: serde_json::Value,
}

impl LiveVenueEvent {
    pub fn hash(&self) -> String {
        event_hash(&json!({
            "source": self.source,
            "event_type": self.event_type,
            "venue_event_id": self.venue_event_id,
            "venue_order_id": self.venue_order_id,
            "venue_trade_id": self.venue_trade_id,
            "event_status": self.event_status,
            "raw_payload": self.raw_payload
        }))
    }
}

impl LiveVenue {
    pub fn new(
        config: LiveExecutionConfig,
        clob_base_url: String,
        store: Store,
        data_api: DataApiClient,
    ) -> Result<Self> {
        config.validate_for_live()?;
        let venue = Self {
            config,
            clob_base_url,
            store: Some(store),
            data_api: Some(data_api),
            bound_process_id: None,
            bound_account_ref: None,
            bound_execution: None,
            transport_state: Arc::new(Mutex::new(LiveTransportState::initial())),
            readiness_state: Arc::new(Mutex::new(LiveVenueState::fail_closed())),
            global_entry_gate: Arc::new(Mutex::new(GlobalLiveEntryGate::fail_closed())),
            submit_guard: Arc::new(Mutex::new(())),
        };
        venue.spawn_user_ws_task_if_enabled();
        Ok(venue)
    }

    #[cfg(test)]
    fn new_for_test(config: LiveExecutionConfig) -> Result<Self> {
        config.validate_for_live()?;
        Ok(Self {
            config,
            clob_base_url: "https://clob-v2.polymarket.com".to_string(),
            store: None,
            data_api: None,
            bound_process_id: None,
            bound_account_ref: None,
            bound_execution: None,
            transport_state: Arc::new(Mutex::new(LiveTransportState::initial())),
            readiness_state: Arc::new(Mutex::new(LiveVenueState::fail_closed())),
            global_entry_gate: Arc::new(Mutex::new(GlobalLiveEntryGate::fail_closed())),
            submit_guard: Arc::new(Mutex::new(())),
        })
    }

    /// Produces the process-owned execution adapter used by a managed trading process. Transport
    /// health and the authenticated account connection are shared, while readiness, reconciliation
    /// and the manual entry gate remain isolated to the process.
    pub fn bind_process(
        &self,
        process_id: Uuid,
        execution: &EffectiveProcessExecutionConfig,
    ) -> Result<Self> {
        if process_id.is_nil() {
            bail!("live execution process_id must not be nil");
        }
        if execution.mode != "live" {
            bail!("live execution venue requires execution.mode=live");
        }
        let account_ref = execution.account_ref.as_deref().unwrap_or("").trim();
        if account_ref.is_empty() {
            bail!("live execution account_ref must not be blank");
        }
        if account_ref.len() > 128 {
            bail!("live execution account_ref must not exceed 128 bytes");
        }
        Ok(Self {
            config: self.config.clone(),
            clob_base_url: self.clob_base_url.clone(),
            store: self.store.clone(),
            data_api: self.data_api.clone(),
            bound_process_id: Some(process_id),
            bound_account_ref: Some(account_ref.to_string()),
            bound_execution: Some(execution.clone()),
            transport_state: self.transport_state.clone(),
            readiness_state: Arc::new(Mutex::new(LiveVenueState::fail_closed())),
            global_entry_gate: self.global_entry_gate.clone(),
            submit_guard: self.submit_guard.clone(),
        })
    }

    pub fn bound_process_id(&self) -> Option<Uuid> {
        self.bound_process_id
    }

    pub fn bound_account_ref(&self) -> Option<&str> {
        self.bound_account_ref.as_deref()
    }

    fn bound_execution(&self) -> Result<&EffectiveProcessExecutionConfig> {
        self.bound_execution
            .as_ref()
            .context("live execution venue is not bound to a trading process")
    }

    fn order_submission_enabled(&self) -> bool {
        self.bound_execution.as_ref().is_some_and(|execution| {
            execution.mode == "live" && execution.execute_signals && execution.live_capital
        })
    }

    pub fn parse_user_event(raw_payload: serde_json::Value) -> LiveVenueEvent {
        let event_type = raw_payload
            .get("event_type")
            .or_else(|| raw_payload.get("type"))
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
            .to_ascii_lowercase();
        let venue_event_id = raw_payload
            .get("id")
            .and_then(|value| value.as_str())
            .map(str::to_string);
        let venue_order_id = raw_payload
            .get("order_id")
            .or_else(|| raw_payload.get("id").filter(|_| event_type == "order"))
            .or_else(|| raw_payload.get("taker_order_id"))
            .and_then(|value| value.as_str())
            .map(str::to_string);
        let venue_trade_id = raw_payload
            .get("trade_id")
            .or_else(|| raw_payload.get("id").filter(|_| event_type == "trade"))
            .and_then(|value| value.as_str())
            .map(str::to_string);
        let event_status = raw_payload
            .get("status")
            .or_else(|| raw_payload.get("type"))
            .and_then(|value| value.as_str())
            .map(str::to_string);
        LiveVenueEvent {
            source: "user_ws".to_string(),
            event_type,
            venue_event_id,
            venue_order_id,
            venue_trade_id,
            event_status,
            raw_payload,
        }
    }

    async fn persist_fill_from_live_event(store: &Store, event: &LiveVenueEvent) -> Result<bool> {
        let Some((order, fill)) = live_fill_record_from_event(store, event).await? else {
            return Ok(false);
        };

        store.insert_fill(&fill).await?;
        let cumulative_filled_size = store.order_filled_size(&order.order_id).await?;
        store
            .mark_order_fill_progress(
                &order.order_id,
                cumulative_filled_size,
                json!({
                    "source": "user_ws",
                    "event_type": event.event_type,
                    "venue_event_id": event.venue_event_id,
                    "venue_trade_id": event.venue_trade_id,
                    "cumulative_filled_size": cumulative_filled_size,
                    "raw_payload": event.raw_payload,
                }),
            )
            .await?;
        Ok(true)
    }

    async fn persist_order_update_from_live_event(
        store: &Store,
        event: &LiveVenueEvent,
    ) -> Result<bool> {
        if !matches!(event.event_type.as_str(), "order" | "cancellation")
            || !is_cancelled_order_status(event.event_status.as_deref())
        {
            return Ok(false);
        }
        let Some(order_id) = event
            .venue_order_id
            .as_deref()
            .or_else(|| json_str(&event.raw_payload, "id"))
        else {
            return Ok(false);
        };
        store
            .mark_order_cancelled(
                order_id,
                json!({
                    "source": "user_ws",
                    "event_type": event.event_type,
                    "event_status": event.event_status,
                    "raw_payload": event.raw_payload,
                }),
            )
            .await?;
        Ok(true)
    }

    async fn backfill_fills_from_live_events(&self) -> Result<usize> {
        let store = self.store()?;
        let mut processed = 0usize;
        for event in store.recent_live_trade_events(500).await? {
            if LiveVenue::persist_fill_from_live_event(&store, &event).await? {
                processed += 1;
            } else if let Some(account_address) = configured_account_address(&self.config)? {
                if let Some(account_trade) = account_trade_from_live_event(&account_address, &event)
                {
                    store.upsert_account_trade(&account_trade).await?;
                }
            }
        }
        for event in store.recent_live_order_events(500).await? {
            let _ = LiveVenue::persist_order_update_from_live_event(&store, &event).await?;
        }
        Ok(processed)
    }

    fn spawn_user_ws_task_if_enabled(&self) {
        if !self.config.user_ws_auth_available() {
            return;
        }
        let config = self.config.clone();
        let Some(store) = self.store.clone() else {
            return;
        };
        let data_api = self.data_api.clone();
        let transport_state = self.transport_state.clone();
        let submit_guard = self.submit_guard.clone();
        tokio::spawn(async move {
            let mut reconnect_delay = USER_WS_RECONNECT_INITIAL_DELAY;
            loop {
                let result = run_user_ws_once(
                    &config,
                    &store,
                    data_api.as_ref(),
                    &transport_state,
                    &submit_guard,
                )
                .await;
                let was_healthy = {
                    let mut state = transport_state.lock().await;
                    let was_healthy = state.last_user_ws_pong_at.is_some();
                    state.user_ws_connected = false;
                    state.last_user_ws_pong_at = None;
                    was_healthy
                };
                let retry_delay = if was_healthy {
                    reconnect_delay = USER_WS_RECONNECT_INITIAL_DELAY;
                    USER_WS_RECONNECT_INITIAL_DELAY
                } else {
                    let retry_delay = reconnect_delay;
                    reconnect_delay = next_user_ws_reconnect_delay(reconnect_delay);
                    retry_delay
                };
                if let Err(error) = result {
                    warn!(
                        error = %error,
                        retry_delay_ms = retry_delay.as_millis(),
                        "Polymarket live user websocket disconnected; reconnecting independently of trading process state"
                    );
                }
                tokio::time::sleep(retry_delay).await;
            }
        });
    }

    async fn authenticated_client(&self) -> Result<AuthenticatedClient> {
        if !self.config.submit_auth_available() {
            bail!("live submit auth is incomplete");
        }
        let api_key = self
            .config
            .clob_api_key
            .as_deref()
            .context("missing CLOB API key")?;
        let secret = self
            .config
            .clob_secret
            .clone()
            .context("missing CLOB secret")?;
        let passphrase = self
            .config
            .clob_passphrase
            .clone()
            .context("missing CLOB passphrase")?;
        let private_key = self
            .config
            .private_key
            .as_deref()
            .context("missing private key")?;
        let signature_type = parse_signature_type(self.config.signature_type.as_deref())?;
        let signer = LocalSigner::from_str(private_key)
            .context("failed to parse POLYMARKET_PRIVATE_KEY")?
            .with_chain_id(Some(POLYGON));
        let credentials = Credentials::new(
            Uuid::parse_str(api_key).context("POLYMARKET_CLOB_API_KEY must be a UUID")?,
            secret,
            passphrase,
        );
        let mut builder = SdkClient::new(&self.clob_base_url, SdkConfig::default())
            .context("failed to create Polymarket CLOB SDK client")?
            .authentication_builder(&signer)
            .credentials(credentials)
            .signature_type(signature_type);
        if signature_type != SignatureType::Eoa {
            let funder = self
                .config
                .funder_address
                .as_deref()
                .context("missing POLYMARKET_FUNDER_ADDRESS")?;
            builder = builder.funder(
                Address::from_str(funder).context("failed to parse POLYMARKET_FUNDER_ADDRESS")?,
            );
        }
        let client = builder
            .authenticate()
            .await
            .context("failed to authenticate Polymarket CLOB SDK client")?;
        Ok(client)
    }

    async fn refresh_geoblock(&self) -> Result<()> {
        #[cfg(test)]
        {
            // Unit tests explicitly seed bounded geoblock evidence when exercising checked
            // enable semantics. Production builds always perform the SDK read below.
            if self
                .transport_state
                .lock()
                .await
                .geoblock_checked_at
                .is_some()
            {
                return Ok(());
            }
        }
        let result = async {
            let client = SdkClient::new(&self.clob_base_url, SdkConfig::default())
                .context("failed to create Polymarket geoblock client")?;
            client
                .check_geoblock()
                .await
                .context("failed to read Polymarket geoblock status")
        }
        .await;
        let checked_at = Utc::now();
        match result {
            Ok(response) => {
                self.record_geoblock_state(
                    true,
                    Some(response.blocked),
                    bounded_geoblock_component(&response.country),
                    bounded_geoblock_component(&response.region),
                    checked_at,
                )
                .await;
                Ok(())
            }
            Err(error) => {
                self.record_geoblock_state(false, None, None, None, checked_at)
                    .await;
                Err(error)
            }
        }
    }

    async fn record_geoblock_state(
        &self,
        readable: bool,
        blocked: Option<bool>,
        country: Option<String>,
        region: Option<String>,
        checked_at: DateTime<Utc>,
    ) {
        {
            let mut state = self.transport_state.lock().await;
            state.geoblock_readable = readable;
            state.geoblock_blocked = blocked;
            state.geoblock_country = country;
            state.geoblock_region = region;
            state.geoblock_checked_at = Some(checked_at);
            state.live_confirmed = readable && blocked == Some(false);
        }
        if !readable {
            self.halt_global_entries("live_geoblock_unreadable").await;
        } else if blocked != Some(false) {
            self.halt_global_entries("live_geoblock_blocked").await;
        }
    }

    async fn halt_global_entries(&self, reason: &str) -> u64 {
        let mut gate = self.global_entry_gate.lock().await;
        record_global_entry_halt(&mut gate, reason)
    }

    async fn all_open_order_responses(
        &self,
        client: &AuthenticatedClient,
    ) -> Result<Vec<OpenOrderResponse>> {
        let request = OrdersRequest::builder().build();
        let mut cursor = None;
        let mut seen_cursors = HashSet::new();
        let mut orders = Vec::new();
        for _ in 0..MAX_CLOB_RECONCILIATION_PAGES {
            let page = client
                .orders(&request, cursor.clone())
                .await
                .context("Polymarket CLOB open orders fetch failed")?;
            validate_clob_page_metadata("open orders", page.data.len(), page.count, page.limit)?;
            checked_clob_row_count(orders.len(), page.data.len(), "open orders")?;
            orders.extend(page.data);
            let Some(next_cursor) = next_clob_reconciliation_cursor(
                &page.next_cursor,
                &mut seen_cursors,
                "open orders",
            )?
            else {
                return Ok(orders);
            };
            cursor = Some(next_cursor);
        }
        bail!(
            "Polymarket CLOB open orders exceed the bounded {}-page reconciliation window",
            MAX_CLOB_RECONCILIATION_PAGES
        )
    }

    async fn all_trade_responses(
        &self,
        client: &AuthenticatedClient,
        request: &TradesRequest,
    ) -> Result<Vec<TradeResponse>> {
        let mut cursor = None;
        let mut seen_cursors = HashSet::new();
        let mut trades = Vec::new();
        for _ in 0..MAX_CLOB_RECONCILIATION_PAGES {
            let page = client
                .trades(request, cursor.clone())
                .await
                .context("Polymarket CLOB trades fetch failed")?;
            validate_clob_page_metadata("trades", page.data.len(), page.count, page.limit)?;
            checked_clob_row_count(trades.len(), page.data.len(), "trades")?;
            trades.extend(page.data);
            let Some(next_cursor) =
                next_clob_reconciliation_cursor(&page.next_cursor, &mut seen_cursors, "trades")?
            else {
                return Ok(trades);
            };
            cursor = Some(next_cursor);
        }
        bail!(
            "Polymarket CLOB trades exceed the bounded {}-page reconciliation window",
            MAX_CLOB_RECONCILIATION_PAGES
        )
    }

    async fn bound_process_venue_order_ids<'a>(
        &self,
        store: &Store,
        process_id: Uuid,
        venue_order_ids: impl IntoIterator<Item = &'a str>,
    ) -> Result<HashMap<String, String>> {
        let mut unique_ids = Vec::new();
        let mut seen_ids = HashSet::new();
        for order_id in venue_order_ids {
            let order_id = order_id.trim();
            if order_id.is_empty() || order_id.len() > 256 {
                bail!("Polymarket CLOB reconciliation returned an invalid order identity");
            }
            if seen_ids.insert(order_id.to_string()) {
                unique_ids.push(order_id.to_string());
                if unique_ids.len() > MAX_CLOB_RECONCILIATION_ORDER_IDS {
                    bail!(
                        "Polymarket CLOB reconciliation exceeds the bounded {}-order identity window",
                        MAX_CLOB_RECONCILIATION_ORDER_IDS
                    );
                }
            }
        }

        let mut owned_orders = HashMap::new();
        for chunk in unique_ids.chunks(CLOB_ORDER_ID_QUERY_CHUNK) {
            for (venue_order_id, local_order_id) in store
                .process_order_ids_by_venue_order_ids(process_id, chunk)
                .await?
            {
                if owned_orders
                    .insert(venue_order_id.clone(), local_order_id.clone())
                    .is_some_and(|existing| existing != local_order_id)
                {
                    bail!(
                        "Polymarket CLOB venue order {} maps to conflicting local order identities",
                        venue_order_id
                    );
                }
            }
        }
        Ok(owned_orders)
    }

    async fn canonical_live_account_identity(&self) -> Result<CanonicalLiveAccountIdentity> {
        let identity = canonical_configured_account_identity(
            &self.config,
            self.bound_account_ref()
                .context("live account identity requires a process account_ref")?,
        )?;
        let authenticated_address = self
            .authenticated_client()
            .await?
            .address()
            .to_checksum(None)
            .to_ascii_lowercase();
        if authenticated_address != identity.signer_address {
            bail!("authenticated CLOB address does not match configured live signer identity");
        }
        Ok(identity)
    }

    async fn run_account_reconcile(
        &self,
        mut request: AccountReconcileRequest,
    ) -> Result<AccountReconcileReport> {
        let mut canonical_identity = None;
        if let Some(process_id) = self.bound_process_id {
            let account_ref = self
                .bound_account_ref
                .as_deref()
                .context("process-bound live venue is missing account_ref")?;
            if request
                .process_id
                .is_some_and(|requested| requested != process_id)
            {
                bail!(
                    "account reconciliation process_id must match bound process {}",
                    process_id
                );
            }
            if request
                .account_ref
                .as_deref()
                .is_some_and(|requested| requested.trim() != account_ref)
            {
                bail!(
                    "account reconciliation account_ref must match bound account_ref {}",
                    account_ref
                );
            }
            request.process_id = Some(process_id);
            request.account_ref = Some(account_ref.to_string());
            let identity = self.canonical_live_account_identity().await?;
            if request
                .account_address
                .as_deref()
                .is_some_and(|requested| !requested.eq_ignore_ascii_case(&identity.account_address))
            {
                bail!("account reconciliation address does not match canonical bound identity");
            }
            if request
                .credential_account_fingerprint_sha256
                .as_deref()
                .is_some_and(|requested| requested != identity.fingerprint_sha256)
            {
                bail!(
                    "account reconciliation credential/account fingerprint does not match bound identity"
                );
            }
            request.account_address = Some(identity.account_address.clone());
            request.credential_account_fingerprint_sha256 =
                Some(identity.fingerprint_sha256.clone());
            canonical_identity = Some(identity);
        } else if request.process_id.is_some()
            || request.account_ref.is_some()
            || request.credential_account_fingerprint_sha256.is_some()
        {
            bail!(
                "wallet-wide live account reconciliation rejects caller-supplied process scope; bind the venue to reconcile a process"
            );
        }
        let store = self.store()?;
        let data_api = self
            .data_api
            .as_ref()
            .context("Polymarket Data API client is not configured")?;
        if request.account_address.is_none() {
            request.account_address = configured_account_address(&self.config)?;
        }
        if request.account_address.is_none() {
            request.account_address = Some(
                canonical_identity
                    .as_ref()
                    .context("live account reconciliation has no canonical account identity")?
                    .account_address
                    .clone(),
            );
        }
        reconcile_account_positions(&store, data_api, request).await
    }

    async fn poly1271_candidate_client(&self, funder_address: &str) -> Result<AuthenticatedClient> {
        let private_key = self
            .config
            .private_key
            .as_deref()
            .context("missing private key")?;
        let signer = LocalSigner::from_str(private_key)
            .context("failed to parse POLYMARKET_PRIVATE_KEY")?
            .with_chain_id(Some(POLYGON));
        let funder = Address::from_str(funder_address)
            .context("failed to parse candidate funder address")?;
        SdkClient::new(&self.clob_base_url, SdkConfig::default())
            .context("failed to create Polymarket CLOB SDK client")?
            .authentication_builder(&signer)
            .signature_type(SignatureType::Poly1271)
            .funder(funder)
            .authenticate()
            .await
            .context("failed to authenticate candidate as POLY_1271")
    }

    async fn apply_poly1271_candidate_clob_diagnostics(
        &self,
        candidate: &mut LiveWalletCandidateAddressDiagnostics,
    ) {
        let client = match self.poly1271_candidate_client(&candidate.address).await {
            Ok(client) => client,
            Err(error) => {
                let message = error.to_string();
                candidate.poly1271_api_keys_error = Some(message.clone());
                candidate.poly1271_balance_allowance_error = Some(message.clone());
                candidate.poly1271_open_orders_error = Some(message);
                return;
            }
        };
        candidate.poly1271_authenticated_client_address = Some(client.address().to_checksum(None));

        match client.api_keys().await {
            Ok(_) => candidate.poly1271_api_keys_readable = true,
            Err(error) => candidate.poly1271_api_keys_error = Some(error.to_string()),
        }

        match client
            .balance_allowance(
                BalanceAllowanceRequest::builder()
                    .asset_type(AssetType::Collateral)
                    .signature_type(SignatureType::Poly1271)
                    .build(),
            )
            .await
        {
            Ok(balance) => {
                candidate.poly1271_balance_allowance_readable = true;
                match local_decimal(balance.balance) {
                    Ok(balance) => {
                        candidate.poly1271_collateral_balance = Some(balance.to_string())
                    }
                    Err(error) => {
                        candidate.poly1271_balance_allowance_error = Some(error.to_string())
                    }
                }
            }
            Err(error) => candidate.poly1271_balance_allowance_error = Some(error.to_string()),
        }

        match client.orders(&OrdersRequest::builder().build(), None).await {
            Ok(page) => {
                candidate.poly1271_open_orders_readable = true;
                candidate.poly1271_open_orders_count = Some(page.data.len());
            }
            Err(error) => candidate.poly1271_open_orders_error = Some(error.to_string()),
        }
    }

    async fn poly1271_funder_probe_candidate(
        &self,
        request: &LivePoly1271FunderProbeRequest,
        order_type: OrderType,
        address: &str,
    ) -> LivePoly1271FunderProbeCandidate {
        let mut candidate = LivePoly1271FunderProbeCandidate {
            address: address.to_string(),
            address_valid: is_address_like(address),
            derive_credentials_ok: false,
            derive_credentials_error: None,
            authenticated_client_address: None,
            api_keys_readable: false,
            api_keys_error: None,
            update_balance_allowance_ok: false,
            update_balance_allowance_error: None,
            balance_allowance_readable: false,
            balance_allowance_error: None,
            collateral_balance: None,
            open_orders_readable: false,
            open_orders_error: None,
            open_orders_count: None,
            signed_order_build_ok: false,
            signed_order_error: None,
            signed_order_maker: None,
            signed_order_signer: None,
            signed_order_signature_type: None,
            maker_matches_candidate: None,
            signer_matches_candidate: None,
            signature_type_is_poly1271: None,
            ready_for_live_canary: false,
            signed_order: json!(null),
        };

        if !candidate.address_valid {
            candidate.derive_credentials_error =
                Some("candidate address is not a valid 20-byte hex address".to_string());
            return candidate;
        }

        let client = match self.poly1271_candidate_client(address).await {
            Ok(client) => {
                candidate.derive_credentials_ok = true;
                candidate.authenticated_client_address = Some(client.address().to_checksum(None));
                client
            }
            Err(error) => {
                let message = error.to_string();
                candidate.derive_credentials_error = Some(message.clone());
                candidate.api_keys_error = Some(message.clone());
                candidate.balance_allowance_error = Some(message.clone());
                candidate.open_orders_error = Some(message);
                return candidate;
            }
        };

        match client.api_keys().await {
            Ok(_) => candidate.api_keys_readable = true,
            Err(error) => candidate.api_keys_error = Some(error.to_string()),
        }

        match client
            .update_balance_allowance(
                UpdateBalanceAllowanceRequest::builder()
                    .asset_type(AssetType::Collateral)
                    .signature_type(SignatureType::Poly1271)
                    .build(),
            )
            .await
        {
            Ok(_) => candidate.update_balance_allowance_ok = true,
            Err(error) => candidate.update_balance_allowance_error = Some(error.to_string()),
        }

        match client
            .balance_allowance(
                BalanceAllowanceRequest::builder()
                    .asset_type(AssetType::Collateral)
                    .signature_type(SignatureType::Poly1271)
                    .build(),
            )
            .await
        {
            Ok(balance) => {
                candidate.balance_allowance_readable = true;
                match local_decimal(balance.balance) {
                    Ok(balance) => candidate.collateral_balance = Some(balance.to_string()),
                    Err(error) => candidate.balance_allowance_error = Some(error.to_string()),
                }
            }
            Err(error) => candidate.balance_allowance_error = Some(error.to_string()),
        }

        match client.orders(&OrdersRequest::builder().build(), None).await {
            Ok(page) => {
                candidate.open_orders_readable = true;
                candidate.open_orders_count = Some(page.data.len());
            }
            Err(error) => candidate.open_orders_error = Some(error.to_string()),
        }

        let signed = match self
            .build_signed_poly1271_candidate_order(&client, address, request, order_type)
            .await
        {
            Ok(signed) => signed,
            Err(error) => {
                candidate.signed_order_error = Some(error.to_string());
                return candidate;
            }
        };

        let mut signed_order = serde_json::to_value(&signed)
            .unwrap_or_else(|error| json!({ "error": error.to_string() }));
        candidate.signed_order_maker = signed_order_address_field(&signed_order, "maker");
        candidate.signed_order_signer = signed_order_address_field(&signed_order, "signer");
        candidate.signed_order_signature_type = signed_order_signature_type(&signed_order);
        candidate.maker_matches_candidate =
            addresses_equal(candidate.signed_order_maker.as_deref(), Some(address));
        candidate.signer_matches_candidate =
            addresses_equal(candidate.signed_order_signer.as_deref(), Some(address));
        candidate.signature_type_is_poly1271 = candidate
            .signed_order_signature_type
            .as_deref()
            .map(is_poly1271_signature_type);
        redact_signed_order_secrets(&mut signed_order);
        candidate.signed_order = signed_order;
        candidate.signed_order_build_ok = true;
        candidate.ready_for_live_canary = candidate.derive_credentials_ok
            && candidate.api_keys_readable
            && candidate.update_balance_allowance_ok
            && candidate.balance_allowance_readable
            && candidate.open_orders_readable
            && candidate.signed_order_build_ok
            && decimal_string_positive(candidate.collateral_balance.as_deref())
            && candidate.maker_matches_candidate == Some(true)
            && candidate.signer_matches_candidate == Some(true)
            && candidate.signature_type_is_poly1271 == Some(true);

        candidate
    }

    async fn build_signed_poly1271_candidate_order(
        &self,
        client: &AuthenticatedClient,
        funder_address: &str,
        request: &LivePoly1271FunderProbeRequest,
        order_type: OrderType,
    ) -> Result<impl serde::Serialize> {
        let private_key = self
            .config
            .private_key
            .as_deref()
            .context("missing private key")?;
        let signer = LocalSigner::from_str(private_key)
            .context("failed to parse POLYMARKET_PRIVATE_KEY")?
            .with_chain_id(Some(POLYGON));
        let token_id =
            U256::from_str(&request.token_id).context("failed to parse CLOB token_id")?;
        let signable = client
            .limit_order()
            .token_id(token_id)
            .side(sdk_side(request.side))
            .price(sdk_decimal(request.price)?)
            .size(sdk_decimal(request.size)?)
            .order_type(sdk_order_type(order_type)?)
            .build()
            .await
            .with_context(|| {
                format!("failed to build POLY_1271 dry-run order for funder {funder_address}")
            })?;
        client.sign(&signer, signable).await.with_context(|| {
            format!("failed to sign POLY_1271 dry-run order for funder {funder_address}")
        })
    }

    fn store(&self) -> Result<Store> {
        self.store
            .clone()
            .context("live persistence store is not configured")
    }

    fn require_bound_process_id(&self) -> Result<Uuid> {
        self.bound_process_id.context(
            "live order execution requires a process-bound venue; call LiveVenue::bind_process",
        )
    }

    fn validate_request_process(&self, request: &OrderRequest) -> Result<Uuid> {
        let process_id = self.require_bound_process_id()?;
        if request.process_id != Some(process_id) {
            bail!(
                "live order process_id must match bound process {}; received {:?}",
                process_id,
                request.process_id
            );
        }
        Ok(process_id)
    }

    async fn bounded_nonterminal_orders(&self, process_id: Uuid) -> Result<Vec<OrderRecord>> {
        let store = self.store()?;
        let orders = store
            .live_process_nonterminal_orders(
                process_id,
                (MAX_PROCESS_NONTERMINAL_ORDERS + 1) as i64,
            )
            .await?;
        if orders.len() > MAX_PROCESS_NONTERMINAL_ORDERS {
            bail!(
                "live process {} exceeds the bounded {}-order reconciliation window",
                process_id,
                MAX_PROCESS_NONTERMINAL_ORDERS
            );
        }
        Ok(orders)
    }

    async fn enforce_submission_risk(
        &self,
        process_id: Uuid,
        request: &OrderRequest,
        ignored_client_order_id: Option<Uuid>,
    ) -> Result<Option<LiveExecutionGateReason>> {
        let identity = canonical_configured_account_identity(
            &self.config,
            self.bound_account_ref()
                .context("live risk checks require a process account_ref")?,
        )?;
        let (process_accounting_proven, reconciled_fingerprint) = {
            let state = self.readiness_state.lock().await;
            (
                state.process_accounting_proven,
                state.credential_account_fingerprint_sha256.clone(),
            )
        };
        if !process_accounting_proven {
            return Ok(Some(LiveExecutionGateReason::ProcessAccountingReadiness));
        }
        if reconciled_fingerprint.as_deref() != Some(identity.fingerprint_sha256.as_str()) {
            bail!("live process reconciliation identity does not match configured credentials");
        }

        let store = self.store()?;
        let exposure = store
            .conservative_live_process_exposure(process_id, ignored_client_order_id)
            .await?;
        if exposure.has_unredeemed_settlement {
            return Ok(Some(LiveExecutionGateReason::SettlementRedemptionUnproven));
        }
        let requested_exposure = Store::conservative_live_request_exposure(request)?;
        let resulting_exposure = exposure
            .total_exposure_usd
            .checked_add(requested_exposure.total_exposure_usd)
            .context("live requested capital exposure overflow")?;
        if let Some(reason) = live_capital_exposure_gate(
            resulting_exposure,
            self.bound_execution()?.max_daily_loss_usd,
            self.bound_execution()?.max_open_notional_usd,
        ) {
            return Ok(Some(reason));
        }

        if let Some(max_daily_loss_usd) = self.bound_execution()?.max_daily_loss_usd {
            let now = Utc::now();
            let day_start = now
                .date_naive()
                .and_hms_opt(0, 0, 0)
                .expect("midnight is a valid UTC time")
                .and_utc();
            let day_end = day_start + chrono::Duration::days(1);
            let recognized_net_pnl = store
                .recognized_live_process_net_pnl_for_utc_day(process_id, day_start, day_end, now)
                .await?;
            if recognized_net_pnl <= -max_daily_loss_usd {
                return Ok(Some(LiveExecutionGateReason::DailyLossLimit));
            }
        }

        let mut markets = exposure
            .exposed_market_ids
            .into_iter()
            .collect::<HashSet<_>>();
        markets.insert(request.market_id.clone());
        if self
            .bound_execution()?
            .max_open_positions
            .is_some_and(|maximum| markets.len() > maximum)
        {
            return Ok(Some(LiveExecutionGateReason::OpenPositionLimit));
        }
        Ok(None)
    }

    async fn current_entry_gate_reason(&self) -> Result<Option<LiveExecutionGateReason>> {
        if !self.order_submission_enabled() {
            return Ok(Some(LiveExecutionGateReason::OrderSubmissionDisabled));
        }
        if !self.config.submit_auth_available() {
            bail!("live submit auth is incomplete");
        }
        if self.live_status().await?.entries_enabled {
            return Ok(None);
        }
        if self.global_entry_gate.lock().await.halted {
            return Ok(Some(LiveExecutionGateReason::GlobalHalt));
        }
        let state = self.readiness_state.lock().await;
        if !state.manual_entries_enabled {
            return Ok(Some(LiveExecutionGateReason::ManualEnableRequired));
        }
        if !state.process_accounting_proven {
            return Ok(Some(LiveExecutionGateReason::ProcessAccountingReadiness));
        }
        Ok(Some(LiveExecutionGateReason::VenueReadiness))
    }

    async fn final_submission_gate_reason(
        &self,
        process_id: Uuid,
        request: &OrderRequest,
    ) -> Result<Option<LiveExecutionGateReason>> {
        if self.global_entry_gate.lock().await.halted {
            return Ok(Some(LiveExecutionGateReason::GlobalHalt));
        }
        if let Some(reason) = self.current_entry_gate_reason().await? {
            return Ok(Some(reason));
        }
        self.enforce_submission_risk(process_id, request, Some(request.client_order_id))
            .await
    }

    async fn mark_idempotency_dirty(&self) {
        self.readiness_state.lock().await.idempotency_clean = false;
    }
}

async fn run_user_ws_once(
    config: &LiveExecutionConfig,
    store: &Store,
    data_api: Option<&DataApiClient>,
    state: &Arc<Mutex<LiveTransportState>>,
    submit_guard: &Arc<Mutex<()>>,
) -> Result<()> {
    let subscription = user_ws_subscription_payload(config)?;
    {
        let mut state = state.lock().await;
        state.user_ws_connected = false;
        state.last_user_ws_pong_at = None;
    }
    let operation_timeout = config
        .user_ws_stale
        .min(USER_WS_MAX_TRANSPORT_OPERATION_TIMEOUT);
    let (mut ws, _) = tokio::time::timeout(operation_timeout, connect_async(&config.user_ws_url))
        .await
        .context("Polymarket user websocket connect timed out")?
        .context("failed to connect Polymarket user websocket")?;
    tokio::time::timeout(
        operation_timeout,
        ws.send(Message::Text(subscription.to_string().into())),
    )
    .await
    .context("Polymarket user websocket subscription send timed out")?
    .context("failed to subscribe Polymarket user websocket")?;

    let mut heartbeat = tokio::time::interval(Duration::from_secs(10));
    let mut awaiting_pong_since: Option<tokio::time::Instant> = None;
    loop {
        tokio::select! {
            _ = heartbeat.tick() => {
                if user_ws_heartbeat_ack_timed_out(
                    awaiting_pong_since.map(|sent_at| sent_at.elapsed()),
                    config.user_ws_stale,
                ) {
                    bail!("Polymarket user websocket heartbeat acknowledgement timed out");
                }
                tokio::time::timeout(
                    operation_timeout,
                    ws.send(Message::Text("PING".into())),
                )
                    .await
                    .context("Polymarket user websocket heartbeat send timed out")?
                    .context("failed to ping Polymarket user websocket")?;
                awaiting_pong_since.get_or_insert_with(tokio::time::Instant::now);
            }
            message = ws.next() => {
                let Some(message) = message else {
                    bail!("Polymarket user websocket ended without a close frame");
                };
                match message.context("failed to read Polymarket user websocket")? {
                    Message::Text(text) => {
                        let text = text.to_string();
                        let payload = match classify_user_ws_text(&text, &[])? {
                            UserWsText::HeartbeatPong => {
                                awaiting_pong_since = None;
                                let mut state = state.lock().await;
                                state.user_ws_connected = true;
                                state.last_user_ws_pong_at = Some(Utc::now());
                                continue;
                            }
                            UserWsText::HeartbeatPing => {
                                tokio::time::timeout(
                                    operation_timeout,
                                    ws.send(Message::Text("PONG".into())),
                                )
                                    .await
                                    .context("Polymarket user websocket text PONG send timed out")?
                                    .context("failed to pong Polymarket user websocket")?;
                                continue;
                            }
                            UserWsText::UserEvent(payload) => payload,
                        };
                        // Apply the websocket event atomically with respect to live submission.
                        // Connectivity and event delivery never mutate process authorization.
                        let _event_guard = submit_guard.lock().await;
                        let event = LiveVenue::parse_user_event(payload);
                        let inserted = store.insert_live_venue_event(&event).await?;
                        let bot_fill_persisted = match LiveVenue::persist_fill_from_live_event(&store, &event).await {
                            Ok(persisted) => persisted,
                            Err(error) => {
                                warn!(error = %error, "failed to persist Polymarket live user websocket fill event");
                                false
                            }
                        };
                        if !bot_fill_persisted {
                            if let Some(account_address) = configured_account_address(config)? {
                                if let Some(account_trade) = account_trade_from_live_event(&account_address, &event) {
                                    if let Err(error) = store.upsert_account_trade(&account_trade).await {
                                        warn!(error = %error, "failed to persist Polymarket account trade from user websocket event");
                                    }
                                    if let Some(data_api) = data_api.cloned() {
                                        let store = store.clone();
                                        let token_id = account_trade.token_id.clone();
                                        tokio::spawn(async move {
                                            tokio::time::sleep(Duration::from_secs(2)).await;
                                            let request = AccountReconcileRequest {
                                                account_address: Some(account_address),
                                                lookback_hours: Some(1),
                                                process_id: None,
                                                account_ref: None,
                                                credential_account_fingerprint_sha256: None,
                                                dry_run: false,
                                                token_id: Some(token_id),
                                                source: Some("user_ws".to_string()),
                                            };
                                            if let Err(error) = reconcile_account_positions(&store, &data_api, request).await {
                                                warn!(error = %error, "websocket-triggered account reconciliation failed");
                                            }
                                        });
                                    }
                                }
                            }
                        }
                        if let Err(error) = LiveVenue::persist_order_update_from_live_event(&store, &event).await {
                            warn!(error = %error, "failed to persist Polymarket live user websocket order event");
                        }
                        if inserted {
                            debug!(
                                event_type = %event.event_type,
                                venue_order_id = ?event.venue_order_id,
                                venue_trade_id = ?event.venue_trade_id,
                                "persisted Polymarket live user websocket event"
                            );
                        }
                        let mut state = state.lock().await;
                        state.user_ws_connected = true;
                    }
                    Message::Ping(payload) => {
                        tokio::time::timeout(operation_timeout, ws.send(Message::Pong(payload)))
                            .await
                            .context("Polymarket user websocket control PONG send timed out")?
                            .context("failed to answer websocket control ping")?;
                    }
                    Message::Pong(_) => {
                        awaiting_pong_since = None;
                        let mut state = state.lock().await;
                        state.user_ws_connected = true;
                        state.last_user_ws_pong_at = Some(Utc::now());
                    }
                    Message::Close(_) => bail!("Polymarket user websocket closed"),
                    Message::Binary(_) => bail!("Polymarket user websocket sent an unsupported binary frame"),
                    _ => {}
                }
            }
        }
    }
}

#[derive(Debug)]
enum UserWsText {
    HeartbeatPong,
    HeartbeatPing,
    UserEvent(Value),
}

fn user_ws_subscription_payload(config: &LiveExecutionConfig) -> Result<Value> {
    Ok(json!({
        "auth": {
            "apiKey": config.clob_api_key.as_deref().unwrap_or(""),
            "secret": config.clob_secret.as_deref().unwrap_or(""),
            "passphrase": config.clob_passphrase.as_deref().unwrap_or("")
        },
        "type": "user"
    }))
}

fn classify_user_ws_text(text: &str, subscribed_markets: &[String]) -> Result<UserWsText> {
    match text.trim() {
        "PONG" => return Ok(UserWsText::HeartbeatPong),
        "PING" => return Ok(UserWsText::HeartbeatPing),
        _ => {}
    }
    let payload: Value = serde_json::from_str(text)
        .context("failed to decode Polymarket user websocket text payload")?;
    let object = payload
        .as_object()
        .context("Polymarket user websocket payload must be an object")?;
    let explicit_error = object
        .get("error")
        .is_some_and(|value| !value.is_null() && value != &Value::Bool(false))
        || object
            .get("success")
            .and_then(Value::as_bool)
            .is_some_and(|success| !success)
        || object
            .get("status")
            .and_then(Value::as_str)
            .is_some_and(|status| {
                matches!(
                    status.trim().to_ascii_lowercase().as_str(),
                    "error" | "failed" | "rejected" | "unauthorized" | "forbidden"
                )
            });
    if explicit_error {
        bail!("Polymarket user websocket returned an authentication or subscription error");
    }

    let event_type = object
        .get("event_type")
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase())
        .context("Polymarket user websocket payload has no recognized event_type")?;
    if !matches!(event_type.as_str(), "order" | "trade") {
        bail!("Polymarket user websocket payload has unsupported event_type");
    }
    let event_id_present = object
        .get("id")
        .or_else(|| object.get("order_id"))
        .or_else(|| object.get("trade_id"))
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    let market = object
        .get("market")
        .or_else(|| object.get("condition_id"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .context("Polymarket user websocket event is missing its market identity")?;
    let asset_present = object
        .get("asset_id")
        .or_else(|| object.get("asset"))
        .or_else(|| object.get("token_id"))
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    if !event_id_present || !asset_present {
        bail!("Polymarket user websocket event is missing an immutable order/trade identity");
    }
    if !subscribed_markets.is_empty()
        && !subscribed_markets
            .iter()
            .any(|subscribed| subscribed == market)
    {
        bail!("Polymarket user websocket event does not match the configured subscription");
    }
    Ok(UserWsText::UserEvent(payload))
}

fn parse_signature_type(value: Option<&str>) -> Result<SignatureType> {
    match value.unwrap_or("0").trim().to_ascii_lowercase().as_str() {
        "0" | "eoa" => Ok(SignatureType::Eoa),
        "1" | "proxy" => Ok(SignatureType::Proxy),
        "2" | "gnosis" | "gnosis_safe" | "gnosissafe" => Ok(SignatureType::GnosisSafe),
        "3" | "poly1271" | "poly_1271" => Ok(SignatureType::Poly1271),
        other => bail!("unsupported POLYMARKET_SIGNATURE_TYPE={other}"),
    }
}

fn configured_account_address(config: &LiveExecutionConfig) -> Result<Option<String>> {
    if config.private_key.is_none() {
        return Ok(None);
    }
    Ok(Some(
        canonical_configured_account_identity(config, "")?.account_address,
    ))
}

fn canonical_configured_account_identity(
    config: &LiveExecutionConfig,
    account_ref: &str,
) -> Result<CanonicalLiveAccountIdentity> {
    let signature_type = parse_signature_type(config.signature_type.as_deref())?;
    let private_key = config
        .private_key
        .as_deref()
        .context("missing private key for canonical live account identity")?;
    let signer = LocalSigner::from_str(private_key)
        .context("failed to parse POLYMARKET_PRIVATE_KEY")?
        .with_chain_id(Some(POLYGON));
    let signer_address = signer.address().to_checksum(None).to_ascii_lowercase();
    let configured_funder = config
        .funder_address
        .as_deref()
        .filter(|address| !address.trim().is_empty())
        .map(|address| {
            Address::from_str(address)
                .context("failed to parse POLYMARKET_FUNDER_ADDRESS")
                .map(|address| address.to_checksum(None).to_ascii_lowercase())
        })
        .transpose()?;

    let account_address = match signature_type {
        SignatureType::Eoa => {
            if configured_funder
                .as_deref()
                .is_some_and(|funder| funder != signer_address)
            {
                bail!("EOA live identity has a configured funder different from the signer");
            }
            signer_address.clone()
        }
        SignatureType::Proxy => {
            let funder = configured_funder
                .as_deref()
                .context("proxy live identity requires POLYMARKET_FUNDER_ADDRESS")?;
            let expected = derive_proxy_wallet(signer.address(), POLYGON)
                .context("failed to derive signer proxy wallet")?
                .to_checksum(None)
                .to_ascii_lowercase();
            if funder != expected {
                bail!("configured proxy funder does not match the signer-derived proxy wallet");
            }
            funder.to_string()
        }
        SignatureType::GnosisSafe => {
            let funder = configured_funder
                .as_deref()
                .context("safe live identity requires POLYMARKET_FUNDER_ADDRESS")?;
            let expected = derive_safe_wallet(signer.address(), POLYGON)
                .context("failed to derive signer safe wallet")?
                .to_checksum(None)
                .to_ascii_lowercase();
            if funder != expected {
                bail!("configured safe funder does not match the signer-derived safe wallet");
            }
            funder.to_string()
        }
        SignatureType::Poly1271 => configured_funder
            .context("POLY_1271 live identity requires POLYMARKET_FUNDER_ADDRESS")?,
        _ => bail!("unsupported live signature type for canonical account identity"),
    };
    let fingerprint_sha256 = event_hash(&json!({
        "identity_version": "polymarket_live_account_v2",
        "chain_id": POLYGON,
        "account_ref": account_ref,
        "signature_type": signature_type as u8,
        "signer_address": &signer_address,
        "account_address": &account_address,
    }));
    Ok(CanonicalLiveAccountIdentity {
        account_address,
        signer_address,
        signature_type,
        fingerprint_sha256,
    })
}

fn sdk_decimal(value: Decimal) -> Result<SdkDecimal> {
    value
        .to_string()
        .parse::<SdkDecimal>()
        .context("failed to convert decimal to SDK decimal")
}

fn local_decimal(value: SdkDecimal) -> Result<Decimal> {
    value
        .to_string()
        .parse::<Decimal>()
        .context("failed to convert SDK decimal")
}

fn sdk_side(side: OrderSide) -> SdkSide {
    match side {
        OrderSide::Buy => SdkSide::Buy,
        OrderSide::Sell => SdkSide::Sell,
    }
}

fn local_side(side: SdkSide) -> OrderSide {
    match side {
        SdkSide::Sell => OrderSide::Sell,
        _ => OrderSide::Buy,
    }
}

fn sdk_order_type(order_type: OrderType) -> Result<SdkOrderType> {
    match order_type {
        OrderType::Fok => Ok(SdkOrderType::FOK),
        OrderType::Gtc => Ok(SdkOrderType::GTC),
        OrderType::Gtd => bail!("live GTD orders require explicit expiration and are not enabled"),
    }
}

fn local_order_type(order_type: SdkOrderType) -> OrderType {
    match order_type {
        SdkOrderType::GTC => OrderType::Gtc,
        SdkOrderType::GTD => OrderType::Gtd,
        _ => OrderType::Fok,
    }
}

fn order_state_from_status(status: &OrderStatusType, size_matched: Option<Decimal>) -> OrderState {
    match status {
        OrderStatusType::Matched => OrderState::Filled,
        OrderStatusType::Canceled => OrderState::Cancelled,
        OrderStatusType::Live | OrderStatusType::Unmatched | OrderStatusType::Delayed => {
            if size_matched.unwrap_or(Decimal::ZERO) > Decimal::ZERO {
                OrderState::PartiallyFilled
            } else {
                OrderState::Acknowledged
            }
        }
        OrderStatusType::Unknown(_) => OrderState::Unknown,
        _ => OrderState::Unknown,
    }
}

fn order_record_from_open_order(order: OpenOrderResponse) -> Result<OrderRecord> {
    let created_at = order.created_at;
    let price = local_decimal(order.price)?;
    let original_size = local_decimal(order.original_size)?;
    let size_matched = local_decimal(order.size_matched)?;
    let order_id = order.id.clone();
    let market_id = order.market.to_string();
    let token_id = order.asset_id.to_string();
    let side = local_side(order.side);
    let order_type = local_order_type(order.order_type.clone());
    let status = order.status.to_string();
    Ok(OrderRecord {
        order_id: order_id.clone(),
        request: OrderRequest {
            client_order_id: Uuid::new_v5(
                &Uuid::NAMESPACE_URL,
                format!("polymarket:venue-order:{order_id}").as_bytes(),
            ),
            process_id: None,
            market_id,
            token_id,
            side,
            order_type,
            price,
            size: original_size,
            metadata: json!({
                "source": "live_open_order",
                "venue_order_id": order_id,
                "venue_status": status,
                "outcome": order.outcome,
                "size_matched": size_matched
            }),
        },
        state: order_state_from_status(&order.status, Some(size_matched)),
        created_at,
        updated_at: Utc::now(),
    })
}

fn is_rest_fill_trade_status(status: &TradeStatusType) -> Result<bool> {
    match status {
        TradeStatusType::Matched | TradeStatusType::Mined | TradeStatusType::Confirmed => Ok(true),
        TradeStatusType::Retrying | TradeStatusType::Failed => Ok(false),
        TradeStatusType::Unknown(value) => {
            bail!("Polymarket CLOB trade has unknown status {value}")
        }
        _ => bail!("Polymarket CLOB trade has an unsupported status"),
    }
}

fn fill_record_from_trade_for_order(
    order: &OrderRecord,
    venue_order_id: &str,
    trade: &TradeResponse,
    checked_at: DateTime<Utc>,
) -> Result<FillRecord> {
    let process_id = order
        .request
        .process_id
        .context("live REST fill backfill requires process-owned order evidence")?;
    if venue_order_id.trim().is_empty() || venue_order_id.len() > 256 {
        bail!("live REST fill backfill has an invalid venue order identity");
    }
    if trade.id.trim().is_empty() || trade.id.trim() != trade.id || trade.id.len() > 256 {
        bail!("live REST fill backfill has an invalid venue trade identity");
    }
    if !is_rest_fill_trade_status(&trade.status)? {
        bail!("live REST fill backfill cannot persist a non-fill trade status");
    }
    if order.request.price <= Decimal::ZERO
        || order.request.price > Decimal::ONE
        || order.request.size <= Decimal::ZERO
    {
        bail!("live REST fill backfill has invalid local order economics");
    }

    let taker_match = trade.taker_order_id == venue_order_id;
    let maker_matches = trade
        .maker_orders
        .iter()
        .filter(|maker_order| maker_order.order_id == venue_order_id)
        .collect::<Vec<_>>();
    if usize::from(taker_match)
        .checked_add(maker_matches.len())
        .context("live REST fill role count overflow")?
        != 1
    {
        bail!(
            "live REST fill trade {} does not identify exactly one role for venue order {}",
            trade.id,
            venue_order_id
        );
    }

    let (asset_id, side, price, size, fee_rate_bps) = if taker_match {
        if !matches!(
            trade.trader_side,
            polymarket_client_sdk_v2::clob::types::TraderSide::Taker
        ) {
            bail!(
                "live REST fill trade {} conflicts with its authenticated taker role",
                trade.id
            );
        }
        (
            trade.asset_id,
            trade.side,
            trade.price,
            trade.size,
            trade.fee_rate_bps,
        )
    } else {
        if !matches!(
            trade.trader_side,
            polymarket_client_sdk_v2::clob::types::TraderSide::Maker
        ) {
            bail!(
                "live REST fill trade {} conflicts with its authenticated maker role",
                trade.id
            );
        }
        let maker_order = maker_matches[0];
        (
            maker_order.asset_id,
            maker_order.side,
            maker_order.price,
            maker_order.matched_amount,
            maker_order.fee_rate_bps,
        )
    };

    let token_id = asset_id.to_string();
    if token_id != order.request.token_id {
        bail!(
            "live REST fill trade {} token identity does not match local order {}",
            trade.id,
            order.order_id
        );
    }
    let side = match side {
        SdkSide::Buy => OrderSide::Buy,
        SdkSide::Sell => OrderSide::Sell,
        _ => bail!("live REST fill trade {} has an unknown side", trade.id),
    };
    if side != order.request.side {
        bail!(
            "live REST fill trade {} side does not match local order {}",
            trade.id,
            order.order_id
        );
    }

    let price = local_decimal(price)?;
    let size = local_decimal(size)?;
    let fee_rate_bps = local_decimal(fee_rate_bps)?;
    if price <= Decimal::ZERO || price > Decimal::ONE {
        bail!("live REST fill trade {} has an invalid price", trade.id);
    }
    if size <= Decimal::ZERO || size > order.request.size {
        bail!("live REST fill trade {} has an invalid size", trade.id);
    }
    if fee_rate_bps < Decimal::ZERO || fee_rate_bps > Decimal::from(10_000) {
        bail!("live REST fill trade {} has an invalid fee rate", trade.id);
    }
    let marketable = match order.request.side {
        OrderSide::Buy => price <= order.request.price,
        OrderSide::Sell => price >= order.request.price,
    };
    if !marketable {
        bail!(
            "live REST fill trade {} violates local order limit price",
            trade.id
        );
    }
    let earliest_match_time = order
        .created_at
        .checked_sub_signed(FOK_FILL_RECONCILIATION_SKEW)
        .context("live REST fill order window underflow")?;
    let latest_match_time = order
        .created_at
        .checked_add_signed(FOK_FILL_RECONCILIATION_SKEW)
        .context("live REST fill order window overflow")?;
    if trade.match_time > checked_at
        || trade.match_time < earliest_match_time
        || trade.match_time > latest_match_time
    {
        bail!(
            "live REST fill trade {} is outside the bounded order execution window",
            trade.id
        );
    }

    let fee = price
        .checked_mul(size)
        .context("live REST fill notional overflow")?
        .checked_mul(fee_rate_bps)
        .context("live REST fill fee overflow")?
        .checked_div(Decimal::from(10_000))
        .context("live REST fill fee division failed")?;
    Ok(FillRecord {
        fill_id: Uuid::new_v5(
            &Uuid::NAMESPACE_URL,
            format!("polymarket:trade:{}", trade.id).as_bytes(),
        ),
        process_id: Some(process_id),
        order_id: order.order_id.clone(),
        token_id,
        price,
        size,
        fee,
        source: FillSource::Live,
        filled_at: trade.match_time,
    })
}

fn rest_fill_backfill_plan(
    process_id: Uuid,
    local_nonterminal: &[OrderRecord],
    owned_orders: &HashMap<String, String>,
    trades: &[TradeResponse],
    checked_at: DateTime<Utc>,
) -> Result<Vec<FillRecord>> {
    if process_id.is_nil() {
        bail!("live REST fill backfill requires a non-nil process_id");
    }
    if local_nonterminal.len() > MAX_PROCESS_NONTERMINAL_ORDERS
        || owned_orders.len() > MAX_CLOB_RECONCILIATION_ORDER_IDS
        || trades.len() > MAX_CLOB_RECONCILIATION_ROWS
    {
        bail!("live REST fill backfill exceeds its bounded evidence window");
    }

    let mut local_orders = HashMap::with_capacity(local_nonterminal.len());
    for order in local_nonterminal {
        if order.request.process_id != Some(process_id) {
            bail!("live REST fill backfill crossed process order ownership");
        }
        if order.order_id.trim().is_empty() || order.order_id.len() > 256 {
            bail!("live REST fill backfill has an invalid local order identity");
        }
        if local_orders
            .insert(order.order_id.as_str(), order)
            .is_some()
        {
            bail!("live REST fill backfill has duplicate local order evidence");
        }
    }

    let mut seen_trade_ids = HashSet::with_capacity(trades.len());
    let mut fills = Vec::new();
    for trade in trades {
        if trade.id.trim().is_empty() || trade.id.trim() != trade.id || trade.id.len() > 256 {
            bail!("live REST fill backfill has an invalid venue trade identity");
        }
        if !seen_trade_ids.insert(trade.id.as_str()) {
            bail!(
                "live REST fill backfill has duplicate venue trade {}",
                trade.id
            );
        }
        if !is_rest_fill_trade_status(&trade.status)? {
            continue;
        }

        let mut candidates = Vec::new();
        if let Some(local_order_id) = owned_orders.get(&trade.taker_order_id) {
            if local_orders.contains_key(local_order_id.as_str()) {
                candidates.push((trade.taker_order_id.as_str(), local_order_id.as_str()));
            }
        }
        for maker_order in &trade.maker_orders {
            if let Some(local_order_id) = owned_orders.get(&maker_order.order_id) {
                if local_orders.contains_key(local_order_id.as_str()) {
                    candidates.push((maker_order.order_id.as_str(), local_order_id.as_str()));
                }
            }
        }
        if candidates.len() > 1 {
            bail!(
                "live REST fill trade {} maps to multiple nonterminal process orders",
                trade.id
            );
        }
        let Some((venue_order_id, local_order_id)) = candidates.first().copied() else {
            continue;
        };
        let order = local_orders
            .get(local_order_id)
            .context("live REST fill ownership points to missing local order")?;
        fills.push(fill_record_from_trade_for_order(
            order,
            venue_order_id,
            trade,
            checked_at,
        )?);
    }
    Ok(fills)
}

fn reconciliation_trade_window_start(
    local_nonterminal: &[OrderRecord],
    checked_at: DateTime<Utc>,
) -> Result<DateTime<Utc>> {
    if local_nonterminal.len() > MAX_PROCESS_NONTERMINAL_ORDERS {
        bail!("live reconciliation trade window exceeds its bounded local order evidence");
    }
    let oldest_local_created_at = local_nonterminal
        .iter()
        .map(|order| order.created_at)
        .min()
        .unwrap_or(checked_at);
    if oldest_local_created_at > checked_at {
        bail!("live reconciliation found a future-dated local order");
    }
    oldest_local_created_at
        .checked_sub_signed(FOK_FILL_RECONCILIATION_SKEW)
        .context("live reconciliation trade window underflow")
}

async fn persist_rest_fill_backfill(
    store: &Store,
    process_id: Uuid,
    local_nonterminal: &[OrderRecord],
    owned_orders: &HashMap<String, String>,
    trades: &[TradeResponse],
    checked_at: DateTime<Utc>,
) -> Result<Vec<FillRecord>> {
    let fills = rest_fill_backfill_plan(
        process_id,
        local_nonterminal,
        owned_orders,
        trades,
        checked_at,
    )?;
    let mut fill_ids_by_order: HashMap<String, Vec<Uuid>> = HashMap::new();
    for fill in &fills {
        store.insert_fill(fill).await?;
        fill_ids_by_order
            .entry(fill.order_id.clone())
            .or_default()
            .push(fill.fill_id);
    }

    let local_orders = local_nonterminal
        .iter()
        .map(|order| (order.order_id.as_str(), order))
        .collect::<HashMap<_, _>>();
    for (order_id, fill_ids) in fill_ids_by_order {
        let order = local_orders
            .get(order_id.as_str())
            .context("live REST fill progress points to missing local order")?;
        let cumulative_filled_size = store.order_filled_size(&order_id).await?;
        if cumulative_filled_size <= Decimal::ZERO || cumulative_filled_size > order.request.size {
            bail!(
                "live REST cumulative fill size is invalid for order {}",
                order_id
            );
        }
        let updated = store
            .mark_order_fill_progress(
                &order_id,
                cumulative_filled_size,
                json!({
                    "source": "rest_reconcile_backfill",
                    "cumulative_filled_size": cumulative_filled_size,
                    "fill_ids": fill_ids,
                }),
            )
            .await?
            .with_context(|| {
                format!("live REST fill progress could not find local order {order_id}")
            })?;
        let expected_state = if cumulative_filled_size >= order.request.size {
            OrderState::Filled
        } else {
            OrderState::PartiallyFilled
        };
        if updated.state != expected_state {
            bail!(
                "live REST fill progress did not persist expected state for order {}",
                order_id
            );
        }
    }
    Ok(fills)
}

async fn live_fill_record_from_event(
    store: &Store,
    event: &LiveVenueEvent,
) -> Result<Option<(OrderRecord, FillRecord)>> {
    if event.event_type != "trade" || !is_fill_trade_status(event.event_status.as_deref()) {
        return Ok(None);
    }

    let Some(trade_id) = event
        .venue_trade_id
        .as_deref()
        .or_else(|| json_str(&event.raw_payload, "id"))
    else {
        return Ok(None);
    };

    for candidate in live_event_order_id_candidates(&event.raw_payload) {
        let Some(order) = store.find_order_by_venue_order_id(&candidate).await? else {
            continue;
        };
        let details = live_event_fill_details_for_order(&event.raw_payload, &candidate)?;
        let price = details
            .price
            .unwrap_or(json_decimal(&event.raw_payload, "price")?);
        let size = details
            .size
            .unwrap_or(json_decimal(&event.raw_payload, "size")?);
        let fee_rate_bps = match details.fee_rate_bps {
            Some(value) => value,
            None if event.raw_payload.get("fee_rate_bps").is_some() => {
                json_decimal(&event.raw_payload, "fee_rate_bps")?
            }
            None => Decimal::ZERO,
        };
        let fee = price * size * fee_rate_bps / Decimal::from(10_000);
        let token_id = details
            .token_id
            .or_else(|| json_str(&event.raw_payload, "asset_id").map(str::to_string))
            .unwrap_or_else(|| order.request.token_id.clone());
        let filled_at = json_timestamp(&event.raw_payload, "match_time")
            .or_else(|| json_timestamp(&event.raw_payload, "timestamp"))
            .unwrap_or_else(Utc::now);
        let fill = FillRecord {
            fill_id: Uuid::new_v5(
                &Uuid::NAMESPACE_URL,
                format!("polymarket:trade:{trade_id}").as_bytes(),
            ),
            process_id: order.request.process_id,
            order_id: order.order_id.clone(),
            token_id,
            price,
            size,
            fee,
            source: FillSource::Live,
            filled_at,
        };
        return Ok(Some((order, fill)));
    }

    Ok(None)
}

fn is_fill_trade_status(status: Option<&str>) -> bool {
    matches!(
        status.map(|value| value.to_ascii_uppercase()),
        Some(value) if matches!(value.as_str(), "MATCHED" | "MINED" | "CONFIRMED")
    )
}

fn is_cancelled_order_status(status: Option<&str>) -> bool {
    matches!(
        status.map(|value| value.trim().to_ascii_uppercase()),
        Some(value) if matches!(value.as_str(), "CANCELLATION" | "CANCELED" | "CANCELLED")
    )
}

fn is_definitive_live_submit_error(error: &anyhow::Error) -> bool {
    for cause in error.chain() {
        if let Some(status) = cause.downcast_ref::<SdkStatus>() {
            return status.status_code.is_client_error()
                && !matches!(status.status_code.as_u16(), 408 | 409 | 425 | 429);
        }
        if let Some(error) = cause.downcast_ref::<polymarket_client_sdk_v2::error::Error>() {
            if matches!(
                error.kind(),
                SdkErrorKind::Validation | SdkErrorKind::Geoblock
            ) {
                return true;
            }
        }
    }
    false
}

fn live_event_order_id_candidates(payload: &Value) -> Vec<String> {
    let mut candidates = Vec::new();
    if let Some(order_id) = json_str(payload, "taker_order_id") {
        push_unique(&mut candidates, order_id);
    }
    if let Some(order_id) = json_str(payload, "order_id").or_else(|| json_str(payload, "id")) {
        push_unique(&mut candidates, order_id);
    }
    if let Some(maker_orders) = payload.get("maker_orders").and_then(Value::as_array) {
        for maker_order in maker_orders {
            if let Some(order_id) = json_str(maker_order, "order_id") {
                push_unique(&mut candidates, order_id);
            }
        }
    }
    candidates
}

#[derive(Debug, Default)]
struct LiveEventFillDetails {
    price: Option<Decimal>,
    size: Option<Decimal>,
    fee_rate_bps: Option<Decimal>,
    token_id: Option<String>,
}

fn live_event_fill_details_for_order(
    payload: &Value,
    order_id: &str,
) -> Result<LiveEventFillDetails> {
    let Some(maker_orders) = payload.get("maker_orders").and_then(Value::as_array) else {
        return Ok(LiveEventFillDetails::default());
    };
    for maker_order in maker_orders {
        if json_str(maker_order, "order_id") == Some(order_id) {
            return Ok(LiveEventFillDetails {
                price: Some(json_decimal(maker_order, "price")?),
                size: Some(json_decimal(maker_order, "matched_amount")?),
                fee_rate_bps: Some(json_decimal(maker_order, "fee_rate_bps")?),
                token_id: json_str(maker_order, "asset_id").map(str::to_string),
            });
        }
    }
    Ok(LiveEventFillDetails::default())
}

fn json_str<'a>(payload: &'a Value, field: &str) -> Option<&'a str> {
    payload.get(field).and_then(Value::as_str)
}

fn json_decimal(payload: &Value, field: &str) -> Result<Decimal> {
    let value = payload
        .get(field)
        .with_context(|| format!("missing decimal field {field}"))?;
    if let Some(text) = value.as_str() {
        return Decimal::from_str(text)
            .with_context(|| format!("failed to parse decimal field {field}={text}"));
    }
    if let Some(number) = value.as_f64() {
        return Decimal::from_str(&number.to_string())
            .with_context(|| format!("failed to parse decimal field {field}={number}"));
    }
    bail!("decimal field {field} is not a string or number")
}

fn json_timestamp(payload: &Value, field: &str) -> Option<DateTime<Utc>> {
    let raw = payload.get(field)?;
    let value = raw
        .as_i64()
        .or_else(|| raw.as_str().and_then(|text| text.parse::<i64>().ok()))?;
    let seconds = if value > 9_999_999_999 {
        value / 1000
    } else {
        value
    };
    DateTime::<Utc>::from_timestamp(seconds, 0)
}

fn push_unique(candidates: &mut Vec<String>, candidate: &str) {
    if !candidates.iter().any(|existing| existing == candidate) {
        candidates.push(candidate.to_string());
    }
}

fn post_order_response_payload(response: &PostOrderResponse) -> serde_json::Value {
    json!({
        "order_id": response.order_id,
        "status": response.status.to_string(),
        "success": response.success,
        "making_amount": response.making_amount.to_string(),
        "taking_amount": response.taking_amount.to_string(),
        "trade_ids": response.trade_ids.clone(),
        "transaction_hashes": response.transaction_hashes.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "error_msg": response.error_msg.clone(),
    })
}

fn normalized_address(value: Option<&str>) -> Option<String> {
    value
        .map(|address| address.trim().to_ascii_lowercase())
        .filter(|address| !address.is_empty())
}

fn addresses_equal(left: Option<&str>, right: Option<&str>) -> Option<bool> {
    Some(normalized_address(left)? == normalized_address(right)?)
}

fn signed_order_address_field(signed_order: &Value, field: &str) -> Option<String> {
    signed_order
        .get("order")
        .and_then(|order| order.get(field))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn signed_order_signature_type(signed_order: &Value) -> Option<String> {
    signed_order
        .get("order")
        .and_then(|order| order.get("signatureType"))
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| value.to_string())
        })
}

fn redact_signed_order_secrets(signed_order: &mut Value) {
    if let Some(owner) = signed_order.get_mut("owner") {
        *owner = json!("<redacted>");
    }
    if let Some(signature) = signed_order
        .get_mut("order")
        .and_then(|order| order.get_mut("signature"))
    {
        *signature = json!("<redacted>");
    }
}

fn is_poly1271_signature_type(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "3" | "poly1271" | "poly_1271"
    )
}

fn decimal_string_positive(value: Option<&str>) -> bool {
    value
        .and_then(|value| value.parse::<Decimal>().ok())
        .is_some_and(|value| value > Decimal::ZERO)
}

fn collateral_allowances_positive(allowances: &HashMap<Address, String>) -> bool {
    [CTF_EXCHANGE_V2_ADDRESS, NEG_RISK_CTF_EXCHANGE_V2_ADDRESS]
        .into_iter()
        .all(|required_exchange| {
            allowances.iter().any(|(exchange, allowance)| {
                exchange.to_string().eq_ignore_ascii_case(required_exchange)
                    && allowance
                        .trim()
                        .parse::<U256>()
                        .ok()
                        .is_some_and(|allowance| allowance > U256::ZERO)
            })
        })
}

fn is_address_like(value: &str) -> bool {
    let trimmed = value.trim();
    let Some(hex) = trimmed.strip_prefix("0x") else {
        return false;
    };
    hex.len() == 40 && hex.chars().all(|character| character.is_ascii_hexdigit())
}

fn push_unique_candidate_address(candidates: &mut Vec<String>, candidate: Option<&str>) {
    let Some(candidate) = candidate
        .map(str::trim)
        .filter(|candidate| is_address_like(candidate))
    else {
        return;
    };
    if candidates
        .iter()
        .any(|existing| addresses_equal(Some(existing), Some(candidate)) == Some(true))
    {
        return;
    }
    candidates.push(candidate.to_string());
}

async fn persist_pre_submit_gate_rejection(
    store: &Store,
    pending: OrderRecord,
    reason: LiveExecutionGateReason,
) -> Result<OrderRecord> {
    let mut rejected = live_execution_gate_closed_order(pending.request.clone(), reason)?;
    rejected.order_id = pending.order_id;
    rejected.created_at = pending.created_at;
    store.mark_order_pre_submit_rejected(&rejected).await
}

async fn persist_pre_submit_hard_failure(
    store: &Store,
    pending: &OrderRecord,
    error: &anyhow::Error,
) -> Result<()> {
    let mut message = format!("{error:#}");
    message.truncate(512);
    store
        .mark_order_submit_failed(
            pending.request.client_order_id,
            "local_pre_submit_failure",
            json!({
                "error": message,
                "post_attempted": false,
            }),
        )
        .await?;
    Ok(())
}

#[async_trait]
impl ExecutionVenue for LiveVenue {
    async fn find_existing_order(&self, request: &OrderRequest) -> Result<Option<OrderRecord>> {
        self.validate_request_process(request)?;
        let Some(store) = self.store.as_ref() else {
            if cfg!(test) {
                return Ok(None);
            }
            bail!("live persistence store is not configured");
        };
        if store
            .find_order_by_client_order_id(request.client_order_id)
            .await?
            .is_none()
        {
            return Ok(None);
        }
        let (existing, newly_created) = store.create_pending_order(request).await?;
        debug_assert!(!newly_created);
        Ok(Some(existing))
    }

    async fn submit_order(&self, request: OrderRequest) -> Result<OrderRecord> {
        self.validate_request_process(&request)?;
        bail!("process-bound live venue submission requires an adjacent pre-POST guard")
    }

    async fn submit_order_with_pre_post_guard(
        &self,
        request: OrderRequest,
        pre_post_guard: Option<Arc<dyn LivePrePostGuard>>,
    ) -> Result<OrderRecord> {
        let process_id = self.validate_request_process(&request)?;
        let pre_post_guard = pre_post_guard
            .context("process-bound live venue submission requires an adjacent pre-POST guard")?;
        let _submit_guard = self.submit_guard.lock().await;
        if let Some(existing) = self.find_existing_order(&request).await? {
            return Ok(existing);
        }
        let notional = request.price * request.size;
        if self
            .bound_execution()?
            .max_order_notional_usd
            .is_some_and(|maximum| notional > maximum)
        {
            return live_execution_gate_closed_order(
                request,
                LiveExecutionGateReason::PerOrderNotionalLimit,
            );
        }
        if !self.order_submission_enabled() {
            return live_execution_gate_closed_order(
                request,
                LiveExecutionGateReason::OrderSubmissionDisabled,
            );
        }
        let global_halted = {
            let global = self.global_entry_gate.lock().await;
            global.halted
        };
        if global_halted {
            return live_execution_gate_closed_order(request, LiveExecutionGateReason::GlobalHalt);
        }
        if let Some(reason) = self.current_entry_gate_reason().await? {
            return live_execution_gate_closed_order(request, reason);
        }
        let store = self.store()?;
        if let Some(reason) = self
            .enforce_submission_risk(process_id, &request, None)
            .await?
        {
            return live_execution_gate_closed_order(request, reason);
        }

        // Complete all local validation, CLOB metadata reads and signing before claiming a pending
        // order. The durable row is then written immediately before the only operation whose
        // outcome can be ambiguous: the venue POST.
        let private_key = self
            .config
            .private_key
            .as_deref()
            .context("missing private key")?;
        let signer = LocalSigner::from_str(private_key)
            .context("failed to parse POLYMARKET_PRIVATE_KEY")?
            .with_chain_id(Some(POLYGON));
        let client = self.authenticated_client().await?;
        let token_id =
            U256::from_str(&request.token_id).context("failed to parse CLOB token_id")?;
        let signable = client
            .limit_order()
            .token_id(token_id)
            .side(sdk_side(request.side))
            .price(sdk_decimal(request.price)?)
            .size(sdk_decimal(request.size)?)
            .order_type(sdk_order_type(request.order_type)?)
            .build()
            .await
            .context("failed to build Polymarket CLOB order")?;
        let signed = client
            .sign(&signer, signable)
            .await
            .context("failed to sign Polymarket CLOB order")?;
        let (pending_order, newly_created) = store.create_pending_order(&request).await?;
        if !newly_created {
            return Ok(pending_order);
        }
        let final_gate_reason = match self
            .final_submission_gate_reason(process_id, &request)
            .await
        {
            Ok(reason) => reason,
            Err(error) => {
                persist_pre_submit_hard_failure(&store, &pending_order, &error).await?;
                return Err(error);
            }
        };
        if let Some(reason) = final_gate_reason {
            return persist_pre_submit_gate_rejection(&store, pending_order, reason).await;
        }
        let admitted_safety_generation = {
            let global = self.global_entry_gate.lock().await;
            if global.halted || global.safety_generation == u64::MAX {
                return persist_pre_submit_gate_rejection(
                    &store,
                    pending_order,
                    LiveExecutionGateReason::GlobalHalt,
                )
                .await;
            }
            global.safety_generation
        };
        let guard_reason = match pre_post_guard.validate_pre_post(&request).await {
            Ok(reason) => reason,
            Err(error) => {
                persist_pre_submit_hard_failure(&store, &pending_order, &error).await?;
                return Err(error);
            }
        };
        if let Some(reason) = guard_reason {
            return persist_pre_submit_gate_rejection(&store, pending_order, reason).await;
        }
        // Revalidate the checked process authorization after every awaited check and immediately
        // before the venue POST. A safety halt changes the generation and converts this pending
        // order into a durable zero-POST rejection.
        let commit_reason = {
            let mut state = self.readiness_state.lock().await;
            let mut global = self.global_entry_gate.lock().await;
            commit_live_post_attempt(&mut global, &mut state, admitted_safety_generation)
        };
        if let Some(reason) = commit_reason {
            return persist_pre_submit_gate_rejection(&store, pending_order, reason).await;
        }
        let submit_result = client
            .post_order(signed)
            .await
            .context("Polymarket CLOB order submit failed");

        match submit_result {
            Ok(response) if response.success => {
                let raw_ack = post_order_response_payload(&response);
                store
                    .mark_order_submitted(request.client_order_id, &response.order_id, raw_ack)
                    .await
            }
            Ok(response) => {
                let raw = post_order_response_payload(&response);
                let failed_order = self
                    .store()?
                    .mark_order_submit_failed(request.client_order_id, "venue_rejected", raw)
                    .await?;
                let error_msg = response
                    .error_msg
                    .unwrap_or_else(|| "unknown rejection".to_string());
                warn!(
                    client_order_id = %request.client_order_id,
                    error = %error_msg,
                    "Polymarket CLOB definitively rejected order; preserving trading process liveness"
                );
                Ok(failed_order)
            }
            Err(error) => {
                let error_chain = format!("{error:#}");
                if is_definitive_live_submit_error(&error) {
                    let failed_order = store
                        .mark_order_submit_failed(
                            request.client_order_id,
                            "venue_rejected",
                            json!({
                                "error": error.to_string(),
                                "error_chain": error_chain
                            }),
                        )
                        .await?;
                    warn!(
                        client_order_id = %request.client_order_id,
                        error = %error_chain,
                        "Polymarket CLOB definitively rejected order; preserving trading process liveness"
                    );
                    return Ok(failed_order);
                } else {
                    store
                        .mark_order_submit_unknown(
                            request.client_order_id,
                            "ambiguous_submit_error",
                            json!({
                                "error": error.to_string(),
                                "error_chain": error_chain
                            }),
                        )
                        .await?;
                    self.mark_idempotency_dirty().await;
                }
                Err(error)
            }
        }
    }

    async fn cancel_order(&self, order_id: &str) -> Result<OrderRecord> {
        let store = self.store()?;
        if let Some(process_id) = self.bound_process_id {
            let order = store
                .find_order_by_venue_order_id(order_id)
                .await?
                .with_context(|| {
                    format!("live process {process_id} cannot cancel an unowned order {order_id}")
                })?;
            if order.request.process_id != Some(process_id) {
                bail!(
                    "live process {} cannot cancel order {} owned by {:?}",
                    process_id,
                    order_id,
                    order.request.process_id
                );
            }
            if order.order_id.starts_with("live-pending-") {
                bail!(
                    "live process {} cannot cancel pending order {} before venue identity reconciliation",
                    process_id,
                    order_id
                );
            }
        }
        store
            .mark_cancel_requested(order_id, json!({"requested_at": Utc::now()}))
            .await?;
        let client = self.authenticated_client().await?;
        let response = client
            .cancel_order(order_id)
            .await
            .context("Polymarket CLOB cancel_order failed")?;
        if response.canceled.iter().any(|id| id == order_id) {
            if let Some(order) = self
                .store()?
                .mark_order_cancelled(
                    order_id,
                    json!({
                        "canceled": &response.canceled,
                        "not_canceled": &response.not_canceled
                    }),
                )
                .await?
            {
                return Ok(order);
            }
        }
        if let Some(reason) = response.not_canceled.get(order_id) {
            bail!("Polymarket CLOB did not cancel order {order_id}: {reason}");
        }
        bail!("Polymarket CLOB did not confirm cancellation for order {order_id}")
    }

    async fn cancel_all(&self) -> Result<usize> {
        if let Some(process_id) = self.bound_process_id {
            let orders = self.bounded_nonterminal_orders(process_id).await?;
            let mut cancelled = 0usize;
            for order in orders {
                if order.order_id.starts_with("live-pending-") {
                    self.mark_idempotency_dirty().await;
                    bail!(
                        "live process {} has pending order {} without a reconciled venue identity",
                        process_id,
                        order.order_id
                    );
                }
                self.cancel_order(&order.order_id).await?;
                cancelled = cancelled.saturating_add(1);
            }
            return Ok(cancelled);
        }

        // The unbound venue is reserved for the explicitly wallet-wide administrative halt path.
        let store = self.store()?;
        let client = self.authenticated_client().await?;
        let response = client
            .cancel_all_orders()
            .await
            .context("Polymarket CLOB cancel_all failed")?;
        let raw = json!({
            "canceled": &response.canceled,
            "not_canceled": &response.not_canceled
        });
        for order_id in &response.canceled {
            let _ = store.mark_order_cancelled(order_id, raw.clone()).await?;
        }
        Ok(response.canceled.len())
    }

    async fn get_balances(&self) -> Result<Vec<(String, Decimal)>> {
        let client = self.authenticated_client().await?;
        let balance = client
            .balance_allowance(
                BalanceAllowanceRequest::builder()
                    .asset_type(AssetType::Collateral)
                    .signature_type(parse_signature_type(self.config.signature_type.as_deref())?)
                    .build(),
            )
            .await
            .context("Polymarket CLOB balance_allowance failed")?;
        Ok(vec![("USDC".to_string(), local_decimal(balance.balance)?)])
    }

    async fn get_open_orders(&self) -> Result<Vec<OrderRecord>> {
        let client = self.authenticated_client().await?;
        self.all_open_order_responses(&client)
            .await?
            .into_iter()
            .map(order_record_from_open_order)
            .collect::<Result<Vec<_>>>()
    }

    async fn reconcile(&self) -> Result<ReconciliationReport> {
        let checked_at = Utc::now();
        let reconciliation_safety_generation = {
            let global = self.global_entry_gate.lock().await;
            if global.safety_generation == u64::MAX {
                bail!("live safety generation is exhausted; reconciliation remains fail-closed");
            }
            global.safety_generation
        };
        let store = self.store()?;
        let reconcile_result = async {
            self.refresh_geoblock().await?;
            let geoblock = self.transport_state.lock().await.clone();
            if !geoblock.geoblock_readable || geoblock.geoblock_blocked != Some(false) {
                bail!("Polymarket live reconciliation requires an unblocked readable geoblock status");
            }
            let mut fills_backfilled = self
                .backfill_fills_from_live_events()
                .await
                .context("failed to backfill live user websocket fills")?;
            let open_orders = self.get_open_orders().await?;
            let mut local_nonterminal = match self.bound_process_id {
                Some(process_id) => self.bounded_nonterminal_orders(process_id).await?,
                None => Vec::new(),
            };
            let trades = if self.bound_process_id.is_some() {
                let trade_window_start =
                    reconciliation_trade_window_start(&local_nonterminal, checked_at)?;
                let client = self.authenticated_client().await?;
                let trades_request = TradesRequest::builder()
                    .after(trade_window_start.timestamp())
                    .before(checked_at.timestamp())
                    .build();
                self.all_trade_responses(&client, &trades_request).await?
            } else {
                Vec::new()
            };
            let balances = self.get_balances().await?;
            if balances.is_empty() {
                bail!("Polymarket CLOB balance reconciliation returned no assets");
            }
            let owned_orders = if let Some(process_id) = self.bound_process_id {
                let mut venue_order_ids = open_orders
                    .iter()
                    .map(|order| order.order_id.clone())
                    .collect::<Vec<_>>();
                for trade in &trades {
                    let candidate_count = trade
                        .maker_orders
                        .len()
                        .checked_add(1)
                        .context("Polymarket CLOB trade order identity count overflow")?;
                    let total_candidate_count = venue_order_ids
                        .len()
                        .checked_add(candidate_count)
                        .context("Polymarket CLOB reconciliation order identity overflow")?;
                    if total_candidate_count > MAX_CLOB_RECONCILIATION_ORDER_IDS {
                        bail!(
                            "Polymarket CLOB reconciliation exceeds the bounded {}-order identity window",
                            MAX_CLOB_RECONCILIATION_ORDER_IDS
                        );
                    }
                    venue_order_ids.push(trade.taker_order_id.clone());
                    venue_order_ids.extend(
                        trade
                            .maker_orders
                            .iter()
                            .map(|maker_order| maker_order.order_id.clone()),
                    );
                }
                self.bound_process_venue_order_ids(
                    &store,
                    process_id,
                    venue_order_ids.iter().map(String::as_str),
                )
                .await?
            } else {
                HashMap::new()
            };
            if let Some(process_id) = self.bound_process_id {
                let rest_fills = persist_rest_fill_backfill(
                    &store,
                    process_id,
                    &local_nonterminal,
                    &owned_orders,
                    &trades,
                    checked_at,
                )
                .await
                .context("failed to backfill authenticated CLOB REST fills")?;
                fills_backfilled = fills_backfilled
                    .checked_add(rest_fills.len())
                    .context("live reconciliation fill backfill count overflow")?;
                // Reconciliation readiness must be derived from the state after REST evidence is
                // durable and cumulative fill progress has terminalized fully filled orders.
                local_nonterminal = self.bounded_nonterminal_orders(process_id).await?;
            }
            // Position ownership is reconstructed from persisted live fills, so reconcile the
            // wallet only after authenticated REST evidence has been backfilled durably.
            let account_reconcile = self
                .run_account_reconcile(AccountReconcileRequest {
                    account_address: None,
                    lookback_hours: Some(1),
                    process_id: self.bound_process_id,
                    account_ref: self.bound_account_ref.clone(),
                    credential_account_fingerprint_sha256: None,
                    dry_run: false,
                    token_id: None,
                    source: Some("poll".to_string()),
                })
                .await
                .context("live account reconciliation polling backup failed")?;
            let owned_venue_order_ids = owned_orders
                .keys()
                .map(String::as_str)
                .collect::<HashSet<_>>();
            let owned_local_order_ids = owned_orders
                .values()
                .map(String::as_str)
                .collect::<HashSet<_>>();
            let foreign_venue_orders = if self.bound_process_id.is_some() {
                open_orders
                    .iter()
                    .filter(|order| !owned_venue_order_ids.contains(order.order_id.as_str()))
                    .count()
            } else {
                0
            };
            let foreign_wallet_trades = trades
                .iter()
                .filter(|trade| {
                    !owned_venue_order_ids.contains(trade.taker_order_id.as_str())
                        && !trade.maker_orders.iter().any(|maker_order| {
                            owned_venue_order_ids.contains(maker_order.order_id.as_str())
                        })
                })
                .count();
            let missing_local_orders = local_nonterminal
                .iter()
                .filter(|order| !owned_local_order_ids.contains(order.order_id.as_str()))
                .count();
            let unresolved = if self.bound_process_id.is_some() {
                local_nonterminal.len().max(open_orders.len())
            } else {
                open_orders.len()
            };
            let mismatches = missing_local_orders
                .saturating_add(foreign_venue_orders)
                .saturating_add(foreign_wallet_trades)
                .saturating_add(account_reconcile.mismatches.len())
                .saturating_add(account_reconcile.unmatched_trades as usize)
                .saturating_add(usize::from(
                    self.bound_process_id.is_some() && !account_reconcile.process_accounting_proven,
                ));
            if self.global_entry_gate.lock().await.safety_generation
                != reconciliation_safety_generation
            {
                bail!("live safety generation changed during reconciliation");
            }
            Ok::<_, anyhow::Error>((
                fills_backfilled,
                account_reconcile,
                open_orders,
                unresolved,
                mismatches,
                foreign_venue_orders,
                foreign_wallet_trades,
            ))
        }
        .await;

        let (
            fills_backfilled,
            account_reconcile,
            open_orders,
            unresolved,
            mismatches,
            foreign_venue_orders,
            foreign_wallet_trades,
        ) = match reconcile_result {
            Ok(result) => result,
            Err(error) => {
                if let Err(record_error) = store
                    .insert_live_reconciliation_run(
                        self.bound_process_id,
                        self.bound_account_ref.as_deref(),
                        "failed",
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        json!({
                            "process_id": self.bound_process_id,
                            "error": error.to_string(),
                            "checked_at": checked_at,
                        }),
                    )
                    .await
                {
                    warn!(
                        error = %record_error,
                        reconciliation_error = %error,
                        "failed to persist failed live reconciliation run"
                    );
                }
                return Err(error);
            }
        };

        let report = ReconciliationReport {
            open_orders: open_orders.len(),
            balances_checked: true,
            mismatches_found: mismatches,
            unresolved_count: unresolved,
            checked_at,
        };
        let idempotency_clean = report.unresolved_count == 0 && report.mismatches_found == 0;
        if let Err(error) = store
            .insert_live_reconciliation_run(
                self.bound_process_id,
                self.bound_account_ref.as_deref(),
                "completed",
                report.open_orders as i32,
                0,
                1,
                report.mismatches_found as i32,
                fills_backfilled as i32,
                report.unresolved_count as i32,
                serde_json::json!({
                    "process_id": self.bound_process_id,
                    "venue": report,
                    "account_reconcile": account_reconcile,
                    "process_accounting_proof": &account_reconcile.process_accounting_proof,
                    "credential_account_fingerprint_sha256": &account_reconcile.credential_account_fingerprint_sha256,
                    "foreign_wallet_open_orders": foreign_venue_orders,
                    "foreign_wallet_trades": foreign_wallet_trades,
                    "idempotency_clean": idempotency_clean,
                    "reconciliation_safety_generation": reconciliation_safety_generation,
                }),
            )
            .await
        {
            return Err(error);
        }
        {
            let mut state = self.readiness_state.lock().await;
            let global = self.global_entry_gate.lock().await;
            state.last_rest_reconcile_at = Some(checked_at);
            state.unresolved_live_order_count = unresolved;
            state.idempotency_clean = idempotency_clean;
            state.process_accounting_proven = account_reconcile.process_accounting_proven;
            state.process_accounting_status = account_reconcile.process_accounting_status.clone();
            state.credential_account_fingerprint_sha256 = account_reconcile
                .credential_account_fingerprint_sha256
                .clone();
            if self.bound_process_id.is_some()
                && idempotency_clean
                && global.safety_generation == reconciliation_safety_generation
            {
                state.reconciled_safety_generation = Some(reconciliation_safety_generation);
            } else {
                state.reconciled_safety_generation = None;
            }
            if self.bound_process_id.is_some()
                && idempotency_clean
                && global.safety_generation != reconciliation_safety_generation
            {
                state.idempotency_clean = false;
                bail!("live safety generation changed before reconciliation commit");
            }
        }
        Ok(report)
    }

    async fn fills_for_order(&self, order_id: &str) -> Result<Vec<FillRecord>> {
        let store = self.store()?;
        let client = self.authenticated_client().await?;
        let persisted_order = store
            .find_order_by_venue_order_id(order_id)
            .await?
            .with_context(|| format!("cannot reconcile fills for unknown live order {order_id}"))?;
        if let Some(process_id) = self.bound_process_id {
            if persisted_order.request.process_id != Some(process_id) {
                bail!(
                    "live process {} cannot reconcile fills for order {} owned by {:?}",
                    process_id,
                    order_id,
                    persisted_order.request.process_id
                );
            }
        }
        let order_process_id = persisted_order
            .request
            .process_id
            .context("cannot reconcile fills for a live order without process ownership")?;
        let trades_request = TradesRequest::builder()
            .asset_id(
                U256::from_str(&persisted_order.request.token_id)
                    .context("failed to parse persisted CLOB token_id")?,
            )
            .after((persisted_order.created_at - FOK_FILL_RECONCILIATION_SKEW).timestamp())
            .before((persisted_order.created_at + FOK_FILL_RECONCILIATION_SKEW).timestamp())
            .build();
        let trades = self.all_trade_responses(&client, &trades_request).await?;
        let owned_orders =
            HashMap::from([(order_id.to_string(), persisted_order.order_id.clone())]);
        persist_rest_fill_backfill(
            &store,
            order_process_id,
            std::slice::from_ref(&persisted_order),
            &owned_orders,
            &trades,
            Utc::now(),
        )
        .await
    }

    async fn live_status(&self) -> Result<LiveVenueStatus> {
        let transport = self.transport_state.lock().await;
        let state = self.readiness_state.lock().await;
        let global = self.global_entry_gate.lock().await;
        let now = Utc::now();
        let last_user_ws_pong_age_secs = stateful_age_seconds(transport.last_user_ws_pong_at, now);
        let last_geoblock_check_age_secs = stateful_age_seconds(transport.geoblock_checked_at, now);
        let last_rest_reconcile_age_secs = stateful_age_seconds(state.last_rest_reconcile_at, now);
        let rest_fresh = last_rest_reconcile_age_secs
            .map(|age| age <= self.config.stale_reconcile.as_secs() as i64)
            .unwrap_or(false);
        let geoblock_fresh = transport.geoblock_readable
            && transport.geoblock_blocked == Some(false)
            && last_geoblock_check_age_secs
                .is_some_and(|age| age <= self.config.stale_reconcile.as_secs() as i64);
        let live_confirmed = transport.live_confirmed && geoblock_fresh;
        let order_submit_enabled = self.order_submission_enabled();
        let entries_enabled = live_confirmed
            && order_submit_enabled
            && self.config.submit_auth_available()
            && !global.halted
            && state.manual_entries_enabled
            && state.process_accounting_proven
            && rest_fresh
            && state.idempotency_clean
            && state.unresolved_live_order_count == 0;
        let reason = if entries_enabled {
            None
        } else if !order_submit_enabled {
            Some("live_order_submit_disabled".to_string())
        } else if !self.config.submit_auth_available() {
            Some("live_submit_auth_missing".to_string())
        } else if !transport.geoblock_readable {
            Some("live_geoblock_status_unreadable".to_string())
        } else if transport.geoblock_blocked != Some(false) {
            Some("live_geoblock_blocked".to_string())
        } else if !geoblock_fresh {
            Some("live_geoblock_status_stale".to_string())
        } else if global.halted {
            Some(format!("live_global_halt:{}", global.reason))
        } else if !state.manual_entries_enabled {
            state
                .manual_entries_reason
                .clone()
                .or_else(|| Some("manual_enable_required".to_string()))
        } else if !state.process_accounting_proven {
            Some(format!(
                "live_process_accounting_not_proven:{}",
                state.process_accounting_status
            ))
        } else if !rest_fresh {
            Some("live_rest_reconcile_stale".to_string())
        } else if !state.idempotency_clean {
            Some("live_idempotency_not_clean".to_string())
        } else if state.unresolved_live_order_count > 0 {
            Some("live_unresolved_orders_present".to_string())
        } else {
            Some("live_not_ready".to_string())
        };
        Ok(LiveVenueStatus {
            mode: "live".to_string(),
            live_confirmed,
            geoblock_readable: transport.geoblock_readable,
            geoblock_blocked: transport.geoblock_blocked,
            geoblock_country: transport.geoblock_country.clone(),
            geoblock_region: transport.geoblock_region.clone(),
            last_geoblock_check_age_secs,
            order_submit_enabled,
            user_ws_enabled: self.config.user_ws_auth_available(),
            user_ws_connected: transport.user_ws_connected,
            last_user_ws_pong_age_secs,
            last_rest_reconcile_age_secs,
            idempotency_clean: state.idempotency_clean,
            unresolved_live_order_count: state.unresolved_live_order_count,
            process_accounting_proven: state.process_accounting_proven,
            process_accounting_status: state.process_accounting_status.clone(),
            max_order_notional_usd: self
                .bound_execution
                .as_ref()
                .and_then(|execution| execution.max_order_notional_usd)
                .unwrap_or(Decimal::ZERO),
            max_open_notional_usd: self
                .bound_execution
                .as_ref()
                .and_then(|execution| execution.max_open_notional_usd)
                .unwrap_or(Decimal::ZERO),
            entries_enabled,
            reason,
        })
    }

    async fn live_identity_diagnostics(&self) -> Result<LiveIdentityDiagnostics> {
        let geoblock_error = self
            .refresh_geoblock()
            .await
            .err()
            .map(|error| error.to_string().chars().take(256).collect::<String>());
        let geoblock = self.transport_state.lock().await.clone();
        let signature_type = parse_signature_type(self.config.signature_type.as_deref())?;
        let signer_address = self
            .config
            .private_key
            .as_deref()
            .map(|private_key| {
                LocalSigner::from_str(private_key)
                    .context("failed to parse POLYMARKET_PRIVATE_KEY")
                    .map(|signer| {
                        signer
                            .with_chain_id(Some(POLYGON))
                            .address()
                            .to_checksum(None)
                    })
            })
            .transpose()?;
        let credentials_present = self
            .config
            .clob_api_key
            .as_deref()
            .is_some_and(|value| !value.is_empty())
            && self
                .config
                .clob_secret
                .as_deref()
                .is_some_and(|value| !value.is_empty())
            && self
                .config
                .clob_passphrase
                .as_deref()
                .is_some_and(|value| !value.is_empty());

        let mut diagnostics = LiveIdentityDiagnostics {
            mode: "live".to_string(),
            clob_api_base_url: self.clob_base_url.clone(),
            geoblock_readable: geoblock.geoblock_readable,
            geoblock_blocked: geoblock.geoblock_blocked,
            geoblock_country: geoblock.geoblock_country,
            geoblock_region: geoblock.geoblock_region,
            geoblock_error,
            signer_address,
            configured_funder_address: self.config.funder_address.clone(),
            configured_signature_type: self.config.signature_type.clone(),
            resolved_signature_type: Some(format!("{signature_type:?}")),
            authenticated_client_address: None,
            account_identity_valid: false,
            account_identity_fingerprint_sha256: None,
            credentials_present,
            api_keys_readable: false,
            api_keys_error: None,
            balance_allowance_readable: false,
            balance_allowance_error: None,
            collateral_balance: None,
            open_orders_readable: false,
            open_orders_error: None,
            open_orders_count: None,
            checked_at: Utc::now(),
        };

        let client = match self.authenticated_client().await {
            Ok(client) => client,
            Err(error) => {
                let message = error.to_string();
                diagnostics.api_keys_error = Some(message.clone());
                diagnostics.balance_allowance_error = Some(message.clone());
                diagnostics.open_orders_error = Some(message);
                return Ok(diagnostics);
            }
        };
        let authenticated_client_address = client.address().to_checksum(None);
        diagnostics.authenticated_client_address = Some(authenticated_client_address.clone());
        match self.bound_account_ref().map_or_else(
            || {
                Err(anyhow::anyhow!(
                    "live identity diagnostics require a process account_ref"
                ))
            },
            |account_ref| canonical_configured_account_identity(&self.config, account_ref),
        ) {
            Ok(identity)
                if identity.signature_type == signature_type
                    && authenticated_client_address
                        .eq_ignore_ascii_case(&identity.signer_address) =>
            {
                // The SDK client authenticates as the private-key signer for every signature
                // type. The canonical trading account is the signer only for EOA; for proxy,
                // Safe, and POLY_1271 it is the validated maker/funder carried by `identity`.
                diagnostics.account_identity_valid = true;
                diagnostics.account_identity_fingerprint_sha256 = Some(identity.fingerprint_sha256);
            }
            Ok(_) => {
                let message =
                    "authenticated CLOB signer does not match canonical signature-aware identity"
                        .to_string();
                diagnostics.api_keys_error = Some(message.clone());
                diagnostics.balance_allowance_error = Some(message.clone());
                diagnostics.open_orders_error = Some(message);
                return Ok(diagnostics);
            }
            Err(error) => {
                let message = error.to_string();
                diagnostics.api_keys_error = Some(message.clone());
                diagnostics.balance_allowance_error = Some(message.clone());
                diagnostics.open_orders_error = Some(message);
                return Ok(diagnostics);
            }
        }

        match client.api_keys().await {
            Ok(_) => diagnostics.api_keys_readable = true,
            Err(error) => diagnostics.api_keys_error = Some(error.to_string()),
        }

        if let Err(error) = client
            .update_balance_allowance(
                UpdateBalanceAllowanceRequest::builder()
                    .asset_type(AssetType::Collateral)
                    .signature_type(signature_type)
                    .build(),
            )
            .await
        {
            diagnostics.balance_allowance_error = Some(error.to_string());
        }

        match client
            .balance_allowance(
                BalanceAllowanceRequest::builder()
                    .asset_type(AssetType::Collateral)
                    .signature_type(signature_type)
                    .build(),
            )
            .await
        {
            Ok(balance) => {
                diagnostics.balance_allowance_readable = true;
                diagnostics.collateral_balance = Some(local_decimal(balance.balance)?.to_string());
                if !collateral_allowances_positive(&balance.allowances) {
                    diagnostics.balance_allowance_error = Some(
                        "CLOB collateral allowances are missing, zero, or invalid".to_string(),
                    );
                }
            }
            Err(error) => diagnostics.balance_allowance_error = Some(error.to_string()),
        }

        match self.all_open_order_responses(&client).await {
            Ok(orders) => {
                diagnostics.open_orders_readable = true;
                diagnostics.open_orders_count = Some(orders.len());
            }
            Err(error) => diagnostics.open_orders_error = Some(error.to_string()),
        }

        Ok(diagnostics)
    }

    async fn live_wallet_address_diagnostics(
        &self,
        candidate_addresses: Vec<String>,
    ) -> Result<LiveWalletAddressDiagnostics> {
        let signature_type = parse_signature_type(self.config.signature_type.as_deref())?;
        let signer_address = signer_address_from_private_key(self.config.private_key.as_deref())?;
        let configured_funder_address = self.config.funder_address.clone();
        let signer: Option<Address> = signer_address
            .as_deref()
            .and_then(|address| Address::from_str(address).ok());
        let derived_proxy_wallet_address = signer
            .and_then(|address| derive_proxy_wallet(address, POLYGON))
            .map(|address| address.to_checksum(None));
        let derived_safe_wallet_address = signer
            .and_then(|address| derive_safe_wallet(address, POLYGON))
            .map(|address| address.to_checksum(None));

        let authenticated_client_address = match self.authenticated_client().await {
            Ok(client) => Some(client.address().to_checksum(None)),
            Err(_) => None,
        };

        let expected_order_maker_address = match signature_type {
            SignatureType::Eoa => signer_address.clone(),
            SignatureType::Proxy | SignatureType::GnosisSafe | SignatureType::Poly1271 => {
                configured_funder_address.clone()
            }
            _ => configured_funder_address.clone(),
        };
        let expected_order_signer_field = match signature_type {
            SignatureType::Poly1271 => configured_funder_address.clone(),
            _ => signer_address.clone(),
        };

        let relayer_base_url = std::env::var("POLYMARKET_RELAYER_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_RELAYER_BASE_URL.to_string());
        let (
            configured_funder_deployed_as_deposit_wallet,
            configured_funder_deployed_as_deposit_wallet_error,
            relayer_deployment_check_url,
        ) = check_candidate_relayer_deployment(
            &relayer_base_url,
            configured_funder_address.as_deref(),
            "WALLET",
        )
        .await;

        let rpc_url = std::env::var("POLYMARKET_POLYGON_RPC_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_POLYGON_RPC_URL.to_string());
        let signer_balances = match signer_address.as_deref() {
            Some(address) if is_address_like(address) => {
                Some(wallet_token_balances(&rpc_url, address).await)
            }
            _ => None,
        };
        let configured_funder_balances = match configured_funder_address.as_deref() {
            Some(address) if is_address_like(address) => {
                Some(wallet_token_balances(&rpc_url, address).await)
            }
            _ => None,
        };
        let mut candidate_addresses = wallet_candidate_address_diagnostics(
            candidate_addresses,
            configured_funder_address.as_deref(),
            signer_address.as_deref(),
            authenticated_client_address.as_deref(),
            derived_proxy_wallet_address.as_deref(),
            derived_safe_wallet_address.as_deref(),
            &relayer_base_url,
            &rpc_url,
        )
        .await;
        for candidate in &mut candidate_addresses {
            self.apply_poly1271_candidate_clob_diagnostics(candidate)
                .await;
        }
        let verified_deposit_wallet_addresses = candidate_addresses
            .iter()
            .filter(|candidate| candidate.deployed_as_deposit_wallet == Some(true))
            .map(|candidate| candidate.address.clone())
            .collect::<Vec<_>>();

        Ok(LiveWalletAddressDiagnostics {
            mode: "live".to_string(),
            signer_address: signer_address.clone(),
            configured_funder_address: configured_funder_address.clone(),
            configured_signature_type: self.config.signature_type.clone(),
            resolved_signature_type: Some(format!("{signature_type:?}")),
            authenticated_client_address,
            derived_proxy_wallet_address: derived_proxy_wallet_address.clone(),
            derived_safe_wallet_address: derived_safe_wallet_address.clone(),
            expected_order_maker_address,
            expected_order_signer_field,
            configured_funder_matches_signer: addresses_equal(
                configured_funder_address.as_deref(),
                signer_address.as_deref(),
            ),
            configured_funder_matches_proxy_wallet: addresses_equal(
                configured_funder_address.as_deref(),
                derived_proxy_wallet_address.as_deref(),
            ),
            configured_funder_matches_safe_wallet: addresses_equal(
                configured_funder_address.as_deref(),
                derived_safe_wallet_address.as_deref(),
            ),
            configured_funder_deployed_as_deposit_wallet,
            configured_funder_deployed_as_deposit_wallet_error,
            relayer_base_url: Some(relayer_base_url),
            relayer_deployment_check_url,
            signer_balances,
            configured_funder_balances,
            candidate_addresses,
            verified_deposit_wallet_address: if verified_deposit_wallet_addresses.len() == 1 {
                verified_deposit_wallet_addresses.first().cloned()
            } else {
                None
            },
            verified_deposit_wallet_candidates_count: verified_deposit_wallet_addresses.len(),
            checked_at: Utc::now(),
        })
    }

    async fn live_order_dry_run(
        &self,
        request: LiveOrderDryRunRequest,
    ) -> Result<LiveOrderDryRunDiagnostics> {
        let signature_type = parse_signature_type(self.config.signature_type.as_deref())?;
        let private_key = self
            .config
            .private_key
            .as_deref()
            .context("missing private key")?;
        let signer = LocalSigner::from_str(private_key)
            .context("failed to parse POLYMARKET_PRIVATE_KEY")?
            .with_chain_id(Some(POLYGON));
        let signer_address = signer.address().to_checksum(None);
        let client = self.authenticated_client().await?;
        let authenticated_client_address = client.address().to_checksum(None);
        let token_id =
            U256::from_str(&request.token_id).context("failed to parse CLOB token_id")?;
        let signable = client
            .limit_order()
            .token_id(token_id)
            .side(sdk_side(request.side))
            .price(sdk_decimal(request.price)?)
            .size(sdk_decimal(request.size)?)
            .order_type(sdk_order_type(
                request.order_type.unwrap_or(OrderType::Fok),
            )?)
            .build()
            .await
            .context("failed to build Polymarket CLOB dry-run order")?;
        let signed = client
            .sign(&signer, signable)
            .await
            .context("failed to sign Polymarket CLOB dry-run order")?;
        let mut signed_order =
            serde_json::to_value(&signed).context("failed to serialize dry-run signed order")?;

        let order_signer = signed_order
            .get("order")
            .and_then(|order| order.get("signer"))
            .and_then(|value| value.as_str())
            .map(str::to_string);
        let order_maker = signed_order
            .get("order")
            .and_then(|order| order.get("maker"))
            .and_then(|value| value.as_str())
            .map(str::to_string);
        let order_signature_type = signed_order
            .get("order")
            .and_then(|order| order.get("signatureType"))
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| value.to_string())
            });

        let mut owner_redacted = false;
        if let Some(owner) = signed_order.get_mut("owner") {
            *owner = json!("<redacted>");
            owner_redacted = true;
        }
        let mut signature_redacted = false;
        if let Some(signature) = signed_order
            .get_mut("order")
            .and_then(|order| order.get_mut("signature"))
        {
            *signature = json!("<redacted>");
            signature_redacted = true;
        }

        Ok(LiveOrderDryRunDiagnostics {
            mode: "live".to_string(),
            clob_api_base_url: self.clob_base_url.clone(),
            signer_address: Some(signer_address),
            configured_funder_address: self.config.funder_address.clone(),
            configured_signature_type: self.config.signature_type.clone(),
            resolved_signature_type: Some(format!("{signature_type:?}")),
            authenticated_client_address: Some(authenticated_client_address.clone()),
            order_signer: order_signer.clone(),
            order_maker: order_maker.clone(),
            order_signature_type,
            order_signer_matches_authenticated_client: addresses_equal(
                order_signer.as_deref(),
                Some(&authenticated_client_address),
            ),
            order_signer_matches_configured_funder: addresses_equal(
                order_signer.as_deref(),
                self.config.funder_address.as_deref(),
            ),
            order_maker_matches_configured_funder: addresses_equal(
                order_maker.as_deref(),
                self.config.funder_address.as_deref(),
            ),
            owner_redacted,
            signature_redacted,
            signed_order,
            checked_at: Utc::now(),
        })
    }

    async fn live_poly1271_funder_probe(
        &self,
        request: LivePoly1271FunderProbeRequest,
    ) -> Result<LivePoly1271FunderProbeResponse> {
        let signer_address = signer_address_from_private_key(self.config.private_key.as_deref())?;
        let order_type = request.order_type.unwrap_or(OrderType::Fok);
        let mut candidates = Vec::with_capacity(request.addresses.len());

        for address in &request.addresses {
            candidates.push(
                self.poly1271_funder_probe_candidate(&request, order_type, address.trim())
                    .await,
            );
        }

        let verified_funder_addresses = candidates
            .iter()
            .filter(|candidate| candidate.ready_for_live_canary)
            .map(|candidate| candidate.address.clone())
            .collect::<Vec<_>>();

        Ok(LivePoly1271FunderProbeResponse {
            mode: "live".to_string(),
            clob_api_base_url: self.clob_base_url.clone(),
            signer_address,
            token_id: request.token_id,
            side: request.side,
            order_type,
            price: request.price,
            size: request.size,
            candidates,
            verified_funder_address: if verified_funder_addresses.len() == 1 {
                verified_funder_addresses.first().cloned()
            } else {
                None
            },
            verified_funder_candidates_count: verified_funder_addresses.len(),
            checked_at: Utc::now(),
        })
    }

    async fn live_account_reconcile(
        &self,
        request: AccountReconcileRequest,
    ) -> Result<AccountReconcileReport> {
        self.run_account_reconcile(request).await
    }

    async fn set_live_entries_enabled(
        &self,
        enabled: bool,
        reason: Option<String>,
    ) -> Result<LiveVenueStatus> {
        if !enabled {
            if self.bound_process_id.is_none() {
                let reason = bounded_live_gate_reason(reason.as_deref(), "manual_live_halt");
                // Close first. A submit that has not passed the global check will observe the
                // halt, while one already admitted remains serialized behind the barrier below.
                // Cancellation of this wait cannot reopen the gate.
                {
                    let mut global = self.global_entry_gate.lock().await;
                    record_global_entry_halt(&mut global, &reason);
                }
                let mut state = self.readiness_state.lock().await;
                state.manual_entries_enabled = false;
                state.manual_entries_reason = Some(reason);
            } else {
                let reason = bounded_live_gate_reason(reason.as_deref(), "process_manual_disable");
                let mut state = self.readiness_state.lock().await;
                state.manual_entries_enabled = false;
                state.manual_entries_reason = Some(reason);
            }
            let _submit_barrier = self.submit_guard.lock().await;
            return self.live_status().await;
        }

        if self.bound_process_id.is_none() {
            bail!("wallet-wide live entry enable is forbidden; use a checked process-bound enable");
        }
        {
            let mut state = self.readiness_state.lock().await;
            state.manual_entries_enabled = false;
            state.manual_entries_reason = Some("checked_enable_revalidation".to_string());
            state.reconciled_safety_generation = None;
        }
        // Revoke the old grant before waiting for an in-flight submit, then hold the shared submit
        // barrier across the authoritative reconciliation and final compare-and-open commit.
        let expected_safety_generation = self
            .halt_global_entries("checked_enable_revalidation")
            .await;
        if expected_safety_generation == u64::MAX {
            bail!("live safety generation is exhausted; checked enable remains fail-closed");
        }
        let _submit_guard = self.submit_guard.lock().await;
        let reconciliation = self.reconcile().await?;
        if reconciliation.mismatches_found != 0 || reconciliation.unresolved_count != 0 {
            bail!("checked process-bound live enable requires a clean reconciliation");
        }

        let now = Utc::now();
        let transport = self.transport_state.lock().await;
        let mut state = self.readiness_state.lock().await;
        let mut global = self.global_entry_gate.lock().await;
        let rest_fresh = stateful_age_seconds(state.last_rest_reconcile_at, now)
            .is_some_and(|age| age <= self.config.stale_reconcile.as_secs() as i64);
        let geoblock_fresh = transport.geoblock_readable
            && transport.geoblock_blocked == Some(false)
            && stateful_age_seconds(transport.geoblock_checked_at, now)
                .is_some_and(|age| age <= self.config.stale_reconcile.as_secs() as i64);
        let configured_identity = canonical_configured_account_identity(
            &self.config,
            self.bound_account_ref()
                .context("checked live enable requires a process account_ref")?,
        )?;
        let identity_matches = state.credential_account_fingerprint_sha256.as_deref()
            == Some(configured_identity.fingerprint_sha256.as_str());
        if !self.order_submission_enabled()
            || !self.config.submit_auth_available()
            || !geoblock_fresh
            || !rest_fresh
            || !state.process_accounting_proven
            || !identity_matches
            || !state.idempotency_clean
            || state.unresolved_live_order_count != 0
            || !global.halted
            || global.safety_generation != expected_safety_generation
            || state.reconciled_safety_generation != Some(expected_safety_generation)
        {
            bail!(
                "checked process-bound live enable requires an unblocked egress, fresh reconciliation, proven accounting, matching identity, and clean idempotency"
            );
        }
        if !commit_checked_live_enable(&mut global, &mut state, expected_safety_generation) {
            bail!("live safety generation changed before checked enable commit");
        }
        drop(global);
        drop(state);
        drop(transport);
        self.live_status().await
    }
}

fn stateful_age_seconds(value: Option<DateTime<Utc>>, now: DateTime<Utc>) -> Option<i64> {
    value.and_then(|value| (value <= now).then(|| (now - value).num_seconds()))
}

fn bounded_live_gate_reason(reason: Option<&str>, fallback: &str) -> String {
    let reason = reason
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback);
    reason.chars().take(128).collect()
}

fn bounded_geoblock_component(value: &str) -> Option<String> {
    let bounded = value
        .trim()
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
        .take(16)
        .collect::<String>();
    (!bounded.is_empty()).then_some(bounded)
}

fn next_clob_reconciliation_cursor(
    next_cursor: &str,
    seen_cursors: &mut HashSet<String>,
    resource: &str,
) -> Result<Option<String>> {
    if next_cursor == CLOB_TERMINAL_CURSOR {
        return Ok(None);
    }
    if next_cursor.is_empty()
        || next_cursor.len() > MAX_CLOB_CURSOR_BYTES
        || next_cursor.trim() != next_cursor
        || !next_cursor.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '/' | '=' | '-' | '_')
        })
        || !seen_cursors.insert(next_cursor.to_string())
    {
        bail!("Polymarket CLOB {resource} returned an invalid pagination cursor");
    }
    Ok(Some(next_cursor.to_string()))
}

fn validate_clob_page_metadata(
    resource: &str,
    data_len: usize,
    count: u64,
    limit: u64,
) -> Result<()> {
    let data_len = u64::try_from(data_len).context("Polymarket CLOB page length overflow")?;
    if limit == 0
        || limit > MAX_CLOB_RECONCILIATION_ROWS as u64
        || count != data_len
        || count > limit
    {
        bail!("Polymarket CLOB {resource} returned inconsistent pagination metadata");
    }
    Ok(())
}

fn checked_clob_row_count(current: usize, page_len: usize, resource: &str) -> Result<usize> {
    let total = current
        .checked_add(page_len)
        .context("Polymarket CLOB reconciliation row count overflow")?;
    if total > MAX_CLOB_RECONCILIATION_ROWS {
        bail!(
            "Polymarket CLOB {resource} exceed the bounded {}-row reconciliation window",
            MAX_CLOB_RECONCILIATION_ROWS
        );
    }
    Ok(total)
}

fn next_user_ws_reconnect_delay(current: Duration) -> Duration {
    current
        .checked_mul(2)
        .unwrap_or(USER_WS_RECONNECT_MAX_DELAY)
        .min(USER_WS_RECONNECT_MAX_DELAY)
}

fn record_global_entry_halt(gate: &mut GlobalLiveEntryGate, reason: &str) -> u64 {
    gate.halted = true;
    if let Some(next_generation) = gate.safety_generation.checked_add(1) {
        gate.safety_generation = next_generation;
        gate.reason = bounded_live_gate_reason(Some(reason), "live_global_halt");
    } else {
        gate.reason = "safety_generation_exhausted".to_string();
    }
    gate.safety_generation
}

fn commit_checked_live_enable(
    gate: &mut GlobalLiveEntryGate,
    state: &mut LiveVenueState,
    expected_safety_generation: u64,
) -> bool {
    if expected_safety_generation == u64::MAX
        || !gate.halted
        || gate.safety_generation != expected_safety_generation
        || state.reconciled_safety_generation != Some(expected_safety_generation)
    {
        return false;
    }
    state.manual_entries_enabled = true;
    state.manual_entries_reason = None;
    gate.halted = false;
    gate.reason = "process_checked_enable".to_string();
    true
}

fn commit_live_post_attempt(
    gate: &mut GlobalLiveEntryGate,
    state: &mut LiveVenueState,
    admitted_safety_generation: u64,
) -> Option<LiveExecutionGateReason> {
    if admitted_safety_generation == u64::MAX
        || gate.halted
        || gate.safety_generation != admitted_safety_generation
        || state.reconciled_safety_generation != Some(admitted_safety_generation)
        || !state.manual_entries_enabled
    {
        return Some(LiveExecutionGateReason::GlobalHalt);
    }
    None
}

fn user_ws_heartbeat_ack_timed_out(
    awaiting_pong_elapsed: Option<Duration>,
    timeout: Duration,
) -> bool {
    awaiting_pong_elapsed.is_some_and(|elapsed| elapsed >= timeout)
}

fn live_capital_exposure_gate(
    resulting_exposure: Decimal,
    max_daily_loss_usd: Option<Decimal>,
    max_open_notional_usd: Option<Decimal>,
) -> Option<LiveExecutionGateReason> {
    if max_daily_loss_usd.is_some_and(|maximum| resulting_exposure > maximum) {
        Some(LiveExecutionGateReason::DailyLossLimit)
    } else if max_open_notional_usd.is_some_and(|maximum| resulting_exposure > maximum) {
        Some(LiveExecutionGateReason::OpenNotionalLimit)
    } else {
        None
    }
}

fn signer_address_from_private_key(private_key: Option<&str>) -> Result<Option<String>> {
    private_key
        .map(|private_key| {
            LocalSigner::from_str(private_key)
                .context("failed to parse POLYMARKET_PRIVATE_KEY")
                .map(|signer| {
                    signer
                        .with_chain_id(Some(POLYGON))
                        .address()
                        .to_checksum(None)
                })
        })
        .transpose()
}

async fn check_relayer_deployed(url: &str) -> Result<bool> {
    let value: Value = reqwest::Client::new()
        .get(url)
        .send()
        .await
        .with_context(|| format!("failed to request relayer deployment status from {url}"))?
        .error_for_status()
        .with_context(|| format!("relayer deployment status request failed for {url}"))?
        .json()
        .await
        .with_context(|| format!("failed to decode relayer deployment status from {url}"))?;

    if let Some(deployed) = value.as_bool() {
        return Ok(deployed);
    }
    if let Some(deployed) = value.get("deployed").and_then(Value::as_bool) {
        return Ok(deployed);
    }
    if let Some(deployed) = value.get("isDeployed").and_then(Value::as_bool) {
        return Ok(deployed);
    }
    if let Some(deployed) = value.get("result").and_then(Value::as_bool) {
        return Ok(deployed);
    }
    bail!("relayer deployment response did not contain a boolean deployment status: {value}");
}

async fn check_candidate_relayer_deployment(
    relayer_base_url: &str,
    address: Option<&str>,
    wallet_type: &str,
) -> (Option<bool>, Option<String>, Option<String>) {
    let Some(address) = address
        .map(str::trim)
        .filter(|address| is_address_like(address))
    else {
        return (None, None, None);
    };
    let url = format!(
        "{}/deployed?address={}&type={}",
        relayer_base_url.trim_end_matches('/'),
        address,
        wallet_type
    );
    match check_relayer_deployed(&url).await {
        Ok(deployed) => (Some(deployed), None, Some(url)),
        Err(error) => (None, Some(error.to_string()), Some(url)),
    }
}

async fn wallet_candidate_address_diagnostics(
    requested_candidates: Vec<String>,
    configured_funder_address: Option<&str>,
    signer_address: Option<&str>,
    authenticated_client_address: Option<&str>,
    derived_proxy_wallet_address: Option<&str>,
    derived_safe_wallet_address: Option<&str>,
    relayer_base_url: &str,
    rpc_url: &str,
) -> Vec<LiveWalletCandidateAddressDiagnostics> {
    let mut candidates = Vec::new();
    push_unique_candidate_address(&mut candidates, signer_address);
    push_unique_candidate_address(&mut candidates, configured_funder_address);
    push_unique_candidate_address(&mut candidates, authenticated_client_address);
    push_unique_candidate_address(&mut candidates, derived_proxy_wallet_address);
    push_unique_candidate_address(&mut candidates, derived_safe_wallet_address);
    for candidate in requested_candidates {
        push_unique_candidate_address(&mut candidates, Some(&candidate));
    }

    let mut diagnostics = Vec::with_capacity(candidates.len());
    for address in candidates {
        let (
            deployed_as_deposit_wallet,
            deployed_as_deposit_wallet_error,
            deposit_wallet_deployment_check_url,
        ) = check_candidate_relayer_deployment(relayer_base_url, Some(&address), "WALLET").await;
        let (
            deployed_as_safe_wallet,
            deployed_as_safe_wallet_error,
            safe_wallet_deployment_check_url,
        ) = check_candidate_relayer_deployment(relayer_base_url, Some(&address), "SAFE").await;
        let balances = Some(wallet_token_balances(rpc_url, &address).await);
        diagnostics.push(LiveWalletCandidateAddressDiagnostics {
            address: address.clone(),
            matches_signer: addresses_equal(Some(&address), signer_address),
            matches_configured_funder: addresses_equal(Some(&address), configured_funder_address),
            matches_authenticated_client: addresses_equal(
                Some(&address),
                authenticated_client_address,
            ),
            matches_proxy_wallet: addresses_equal(Some(&address), derived_proxy_wallet_address),
            matches_safe_wallet: addresses_equal(Some(&address), derived_safe_wallet_address),
            deployed_as_deposit_wallet,
            deployed_as_deposit_wallet_error,
            deposit_wallet_deployment_check_url,
            deployed_as_safe_wallet,
            deployed_as_safe_wallet_error,
            safe_wallet_deployment_check_url,
            balances,
            poly1271_authenticated_client_address: None,
            poly1271_api_keys_readable: false,
            poly1271_api_keys_error: None,
            poly1271_balance_allowance_readable: false,
            poly1271_balance_allowance_error: None,
            poly1271_collateral_balance: None,
            poly1271_open_orders_readable: false,
            poly1271_open_orders_error: None,
            poly1271_open_orders_count: None,
        });
    }
    diagnostics
}

async fn wallet_token_balances(rpc_url: &str, address: &str) -> LiveWalletTokenBalances {
    let mut balances = LiveWalletTokenBalances {
        address: address.to_string(),
        pol_wei: None,
        pusd: None,
        usdc_e: None,
        native_usdc: None,
        error: None,
    };

    match rpc_balance_snapshot(rpc_url, address).await {
        Ok(snapshot) => {
            balances.pol_wei = Some(snapshot.pol_wei);
            balances.pusd = Some(snapshot.pusd);
            balances.usdc_e = Some(snapshot.usdc_e);
            balances.native_usdc = Some(snapshot.native_usdc);
        }
        Err(error) => balances.error = Some(error.to_string()),
    }

    balances
}

struct WalletBalanceSnapshot {
    pol_wei: String,
    pusd: String,
    usdc_e: String,
    native_usdc: String,
}

async fn rpc_balance_snapshot(rpc_url: &str, address: &str) -> Result<WalletBalanceSnapshot> {
    Ok(WalletBalanceSnapshot {
        pol_wei: rpc_call_quantity(rpc_url, "eth_getBalance", json!([address, "latest"])).await?,
        pusd: erc20_balance_of(rpc_url, PUSD_ADDRESS, address).await?,
        usdc_e: erc20_balance_of(rpc_url, USDC_E_ADDRESS, address).await?,
        native_usdc: erc20_balance_of(rpc_url, NATIVE_USDC_ADDRESS, address).await?,
    })
}

async fn erc20_balance_of(
    rpc_url: &str,
    token_address: &str,
    owner_address: &str,
) -> Result<String> {
    let owner = owner_address
        .strip_prefix("0x")
        .unwrap_or(owner_address)
        .to_ascii_lowercase();
    let data = format!("0x70a08231{:0>64}", owner);
    rpc_call_quantity(
        rpc_url,
        "eth_call",
        json!([{"to": token_address, "data": data}, "latest"]),
    )
    .await
}

async fn rpc_call_quantity(rpc_url: &str, method: &str, params: Value) -> Result<String> {
    let response: RpcResponse = reqwest::Client::new()
        .post(rpc_url)
        .json(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }))
        .send()
        .await
        .with_context(|| format!("failed to request Polygon RPC method {method}"))?
        .error_for_status()
        .with_context(|| format!("Polygon RPC method {method} returned non-success status"))?
        .json()
        .await
        .with_context(|| format!("failed to decode Polygon RPC response for {method}"))?;

    if let Some(error) = response.error {
        bail!("Polygon RPC method {method} returned error: {error}");
    }

    let raw = response
        .result
        .with_context(|| format!("Polygon RPC method {method} did not return a result"))?;
    Ok(u256_hex_to_decimal_string(&raw)?)
}

fn u256_hex_to_decimal_string(value: &str) -> Result<String> {
    let trimmed = value.strip_prefix("0x").unwrap_or(value);
    if trimmed.is_empty() {
        return Ok("0".to_string());
    }
    let parsed = U256::from_str_radix(trimmed, 16)
        .with_context(|| format!("failed to parse U256 hex quantity {value}"))?;
    Ok(parsed.to_string())
}

#[cfg(test)]
mod tests {
    use rust_decimal_macros::dec;

    use crate::config::LiveExecutionConfig;

    use super::*;

    #[test]
    fn collateral_allowances_require_both_current_v2_exchanges_to_be_positive() {
        let exchange = CTF_EXCHANGE_V2_ADDRESS.parse::<Address>().unwrap();
        let neg_risk_exchange = NEG_RISK_CTF_EXCHANGE_V2_ADDRESS.parse::<Address>().unwrap();

        assert!(!collateral_allowances_positive(&HashMap::new()));
        assert!(!collateral_allowances_positive(&HashMap::from([(
            exchange,
            "0".to_string(),
        )])));
        assert!(!collateral_allowances_positive(&HashMap::from([(
            exchange,
            "invalid".to_string(),
        )])));
        assert!(!collateral_allowances_positive(&HashMap::from([
            (exchange, U256::MAX.to_string()),
            (neg_risk_exchange, "0".to_string()),
        ])));
        assert!(collateral_allowances_positive(&HashMap::from([
            (exchange, U256::MAX.to_string()),
            (neg_risk_exchange, "1".to_string()),
            (Address::ZERO, "0".to_string()),
        ])));
    }

    struct AlwaysReadyPrePostGuard;

    impl crate::execution::live_pre_post_guard_sealed::Sealed for AlwaysReadyPrePostGuard {}

    #[async_trait]
    impl LivePrePostGuard for AlwaysReadyPrePostGuard {
        async fn validate_pre_post(
            &self,
            _request: &OrderRequest,
        ) -> Result<Option<LiveExecutionGateReason>> {
            Ok(None)
        }
    }

    async fn submit_with_test_guard(
        venue: &LiveVenue,
        request: OrderRequest,
    ) -> Result<OrderRecord> {
        venue
            .submit_order_with_pre_post_guard(request, Some(Arc::new(AlwaysReadyPrePostGuard)))
            .await
    }

    fn live_config() -> LiveExecutionConfig {
        LiveExecutionConfig {
            user_ws_url: "wss://ws-subscriptions-clob.polymarket.com/ws/user".to_string(),
            clob_api_base_url: "https://clob-v2.polymarket.com".to_string(),
            user_ws_stale: std::time::Duration::from_secs(20),
            reconcile_interval: std::time::Duration::from_secs(30),
            stale_reconcile: std::time::Duration::from_secs(60),
            clob_api_key: Some("00000000-0000-0000-0000-000000000001".to_string()),
            clob_secret: Some("secret".to_string()),
            clob_passphrase: Some("pass".to_string()),
            private_key: Some(format!("0x{:064x}", 1)),
            funder_address: Some("0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf".to_string()),
            signature_type: Some("0".to_string()),
        }
    }

    fn live_execution() -> EffectiveProcessExecutionConfig {
        EffectiveProcessExecutionConfig {
            mode: "live".to_string(),
            execute_signals: true,
            live_capital: true,
            account_ref: Some("polymarket-test".to_string()),
            taker_fee_rate: dec!(0.03),
            max_order_notional_usd: Some(dec!(2)),
            max_open_notional_usd: Some(dec!(30)),
            max_open_positions: Some(6),
            max_daily_loss_usd: Some(dec!(10)),
            require_exit_book: Some(true),
        }
    }

    fn assert_live_gate_rejection(order: &OrderRecord, reason: LiveExecutionGateReason) {
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

    async fn seed_fresh_unblocked_geoblock(venue: &LiveVenue) {
        let mut transport = venue.transport_state.lock().await;
        transport.live_confirmed = true;
        transport.geoblock_readable = true;
        transport.geoblock_blocked = Some(false);
        transport.geoblock_country = Some("US".to_string());
        transport.geoblock_region = Some("NY".to_string());
        transport.geoblock_checked_at = Some(Utc::now());
    }

    fn rest_backfill_order(
        process_id: Uuid,
        order_id: &str,
        side: OrderSide,
        price: Decimal,
        size: Decimal,
        created_at: DateTime<Utc>,
    ) -> OrderRecord {
        OrderRecord {
            order_id: order_id.to_string(),
            request: OrderRequest {
                client_order_id: Uuid::new_v4(),
                process_id: Some(process_id),
                market_id: "market".to_string(),
                token_id: "1".to_string(),
                side,
                order_type: OrderType::Fok,
                price,
                size,
                metadata: json!({"execution_intent": "entry"}),
            },
            state: OrderState::Acknowledged,
            created_at,
            updated_at: created_at,
        }
    }

    fn taker_trade(
        trade_id: &str,
        order_id: &str,
        price: Decimal,
        size: Decimal,
        fee_rate_bps: Decimal,
        match_time: DateTime<Utc>,
    ) -> TradeResponse {
        TradeResponse::builder()
            .id(trade_id)
            .taker_order_id(order_id)
            .market(polymarket_client_sdk_v2::types::B256::ZERO)
            .asset_id(U256::from(1))
            .side(SdkSide::Buy)
            .size(size)
            .fee_rate_bps(fee_rate_bps)
            .price(price)
            .status(TradeStatusType::Matched)
            .match_time(match_time)
            .last_update(match_time)
            .outcome("YES")
            .bucket_index(0)
            .owner(Uuid::max())
            .maker_address(Address::ZERO)
            .maker_orders(Vec::new())
            .transaction_hash(polymarket_client_sdk_v2::types::B256::ZERO)
            .trader_side(polymarket_client_sdk_v2::clob::types::TraderSide::Taker)
            .build()
    }

    #[test]
    fn rest_fill_backfill_recovers_complete_owned_trade_after_hour_long_gap() {
        let process_id = Uuid::new_v4();
        let checked_at = Utc::now();
        let created_at = checked_at - chrono::Duration::minutes(90);
        let order = rest_backfill_order(
            process_id,
            "venue-order-1",
            OrderSide::Buy,
            dec!(0.50),
            dec!(2),
            created_at,
        );
        let trade = taker_trade(
            "trade-1",
            "venue-order-1",
            dec!(0.49),
            dec!(2),
            dec!(25),
            created_at + chrono::Duration::seconds(1),
        );
        let owned_orders =
            HashMap::from([("venue-order-1".to_string(), "venue-order-1".to_string())]);

        let window_start =
            reconciliation_trade_window_start(std::slice::from_ref(&order), checked_at).unwrap();
        assert_eq!(window_start, created_at - FOK_FILL_RECONCILIATION_SKEW);
        assert!(window_start < checked_at - chrono::Duration::hours(1));

        let fills = rest_fill_backfill_plan(
            process_id,
            std::slice::from_ref(&order),
            &owned_orders,
            std::slice::from_ref(&trade),
            checked_at,
        )
        .unwrap();
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].process_id, Some(process_id));
        assert_eq!(fills[0].order_id, order.order_id);
        assert_eq!(fills[0].token_id, "1");
        assert_eq!(fills[0].price, dec!(0.49));
        assert_eq!(fills[0].size, order.request.size);
        assert_eq!(fills[0].fee, dec!(0.00245));
        assert_eq!(
            fills[0].fill_id,
            Uuid::new_v5(&Uuid::NAMESPACE_URL, b"polymarket:trade:trade-1")
        );
    }

    #[test]
    fn rest_fill_backfill_uses_exact_maker_order_economics() {
        use polymarket_client_sdk_v2::clob::types::response::MakerOrder;

        let process_id = Uuid::new_v4();
        let checked_at = Utc::now();
        let order = rest_backfill_order(
            process_id,
            "maker-order-1",
            OrderSide::Sell,
            dec!(0.40),
            dec!(1),
            checked_at - chrono::Duration::seconds(2),
        );
        let trade = TradeResponse::builder()
            .id("maker-trade-1")
            .taker_order_id("foreign-taker")
            .market(polymarket_client_sdk_v2::types::B256::ZERO)
            .asset_id(U256::from(999))
            .side(SdkSide::Buy)
            .size(dec!(9))
            .fee_rate_bps(dec!(99))
            .price(dec!(0.90))
            .status(TradeStatusType::Confirmed)
            .match_time(checked_at - chrono::Duration::seconds(1))
            .last_update(checked_at)
            .outcome("YES")
            .bucket_index(0)
            .owner(Uuid::max())
            .maker_address(Address::ZERO)
            .maker_orders(vec![MakerOrder::builder()
                .order_id("maker-order-1")
                .owner(Uuid::max())
                .maker_address(Address::ZERO)
                .matched_amount(dec!(1))
                .price(dec!(0.42))
                .fee_rate_bps(dec!(7))
                .asset_id(U256::from(1))
                .outcome("YES")
                .side(SdkSide::Sell)
                .build()])
            .transaction_hash(polymarket_client_sdk_v2::types::B256::ZERO)
            .trader_side(polymarket_client_sdk_v2::clob::types::TraderSide::Maker)
            .build();
        let owned_orders =
            HashMap::from([("maker-order-1".to_string(), "maker-order-1".to_string())]);

        let fills = rest_fill_backfill_plan(
            process_id,
            std::slice::from_ref(&order),
            &owned_orders,
            std::slice::from_ref(&trade),
            checked_at,
        )
        .unwrap();

        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].token_id, "1");
        assert_eq!(fills[0].price, dec!(0.42));
        assert_eq!(fills[0].size, dec!(1));
        assert_eq!(fills[0].fee, dec!(0.000294));
    }

    #[tokio::test]
    async fn user_ws_health_is_diagnostic_and_rest_freshness_controls_backup_readiness() {
        let venue = LiveVenue::new_for_test(live_config())
            .unwrap()
            .bind_process(Uuid::new_v4(), &live_execution())
            .unwrap();
        seed_fresh_unblocked_geoblock(&venue).await;
        {
            let mut transport = venue.transport_state.lock().await;
            transport.user_ws_connected = true;
            transport.last_user_ws_pong_at = Some(Utc::now() + chrono::Duration::minutes(1));
        }
        {
            let mut state = venue.readiness_state.lock().await;
            state.last_rest_reconcile_at = Some(Utc::now());
            state.idempotency_clean = true;
            state.unresolved_live_order_count = 0;
            state.manual_entries_enabled = true;
            state.manual_entries_reason = None;
            state.process_accounting_proven = true;
            state.process_accounting_status = "proven".to_string();
        }
        {
            let mut global = venue.global_entry_gate.lock().await;
            global.halted = false;
            global.reason = "test_checked_enable".to_string();
        }

        let status = venue.live_status().await.unwrap();
        assert!(status.entries_enabled);
        assert_eq!(status.last_user_ws_pong_age_secs, None);
        assert_eq!(status.reason, None);

        {
            let mut transport = venue.transport_state.lock().await;
            transport.user_ws_connected = false;
            transport.last_user_ws_pong_at = None;
        }
        let status = venue.live_status().await.unwrap();
        assert!(status.entries_enabled);
        assert!(!status.user_ws_connected);

        venue.readiness_state.lock().await.last_rest_reconcile_at =
            Some(Utc::now() + chrono::Duration::minutes(1));
        let status = venue.live_status().await.unwrap();
        assert!(!status.entries_enabled);
        assert_eq!(status.last_rest_reconcile_age_secs, None);
        assert_eq!(status.reason.as_deref(), Some("live_rest_reconcile_stale"));
    }

    #[tokio::test]
    async fn clean_reconciliation_restores_readiness_without_manual_reenable() {
        let venue = LiveVenue::new_for_test(live_config())
            .unwrap()
            .bind_process(Uuid::new_v4(), &live_execution())
            .unwrap();
        seed_fresh_unblocked_geoblock(&venue).await;
        {
            let mut state = venue.readiness_state.lock().await;
            state.last_rest_reconcile_at = Some(Utc::now());
            state.idempotency_clean = false;
            state.unresolved_live_order_count = 1;
            state.manual_entries_enabled = true;
            state.manual_entries_reason = None;
            state.process_accounting_proven = true;
            state.process_accounting_status = "proven".to_string();
        }
        {
            let mut global = venue.global_entry_gate.lock().await;
            global.halted = false;
            global.reason = "process_checked_enable".to_string();
        }

        let status = venue.live_status().await.unwrap();
        assert!(!status.entries_enabled);
        assert_eq!(status.reason.as_deref(), Some("live_idempotency_not_clean"));

        {
            let mut state = venue.readiness_state.lock().await;
            state.last_rest_reconcile_at = Some(Utc::now());
            state.idempotency_clean = true;
            state.unresolved_live_order_count = 0;
        }
        let status = venue.live_status().await.unwrap();
        assert!(status.entries_enabled);
        assert_eq!(status.reason, None);
        assert!(venue.readiness_state.lock().await.manual_entries_enabled);
        assert!(!venue.global_entry_gate.lock().await.halted);
    }

    #[tokio::test]
    async fn unsafe_geoblock_refresh_revokes_the_shared_live_generation() {
        let root = LiveVenue::new_for_test(live_config()).unwrap();
        let venue = root
            .bind_process(Uuid::new_v4(), &live_execution())
            .unwrap();
        {
            let mut global = venue.global_entry_gate.lock().await;
            global.halted = false;
            global.reason = "active_test_grant".to_string();
            global.safety_generation = 7;
        }

        root.record_geoblock_state(true, Some(true), Some("XX".to_string()), None, Utc::now())
            .await;
        {
            let global = venue.global_entry_gate.lock().await;
            assert!(global.halted);
            assert_eq!(global.reason, "live_geoblock_blocked");
            assert_eq!(global.safety_generation, 8);
        }

        root.record_geoblock_state(false, None, None, None, Utc::now())
            .await;
        let global = venue.global_entry_gate.lock().await;
        assert!(global.halted);
        assert_eq!(global.reason, "live_geoblock_unreadable");
        assert_eq!(global.safety_generation, 9);
    }

    #[test]
    fn user_event_hash_is_stable() {
        let raw = json!({"event_type":"trade","id":"t1","status":"CONFIRMED"});
        let event = LiveVenue::parse_user_event(raw);
        assert_eq!(event.hash(), event.hash());
        assert_eq!(event.event_type, "trade");
        assert_eq!(event.venue_trade_id.as_deref(), Some("t1"));

        let cancellation = LiveVenue::parse_user_event(json!({
            "type": "CANCELLATION",
            "id": "order-1"
        }));
        assert_eq!(cancellation.event_type, "cancellation");
        assert!(is_cancelled_order_status(
            cancellation.event_status.as_deref()
        ));
        for status in ["CANCELED", "CANCELLED", " cancellation "] {
            assert!(is_cancelled_order_status(Some(status)));
        }
        assert!(!is_cancelled_order_status(Some("CANCELED_BY_SYSTEM")));
        assert!(!is_cancelled_order_status(Some("CANCEL")));
    }

    #[test]
    fn user_ws_omits_empty_market_filter_and_strictly_classifies_health() {
        let config = live_config();
        let subscription = user_ws_subscription_payload(&config).unwrap();
        assert!(subscription.get("markets").is_none());
        assert!(matches!(
            classify_user_ws_text("PONG", &[]).unwrap(),
            UserWsText::HeartbeatPong
        ));
        assert!(classify_user_ws_text("not-json", &[]).is_err());
        assert!(classify_user_ws_text(r#"{"status":"unauthorized"}"#, &[]).is_err());

        let event = json!({
            "event_type": "trade",
            "id": "trade-1",
            "market": "condition-1",
            "asset_id": "token-1",
            "status": "CONFIRMED"
        });
        assert!(matches!(
            classify_user_ws_text(&event.to_string(), &[]).unwrap(),
            UserWsText::UserEvent(_)
        ));
        assert!(
            classify_user_ws_text(&event.to_string(), &["different-condition".to_string()])
                .is_err()
        );
    }

    #[test]
    fn user_ws_heartbeat_requires_an_acknowledgement_within_the_bound() {
        let timeout = Duration::from_secs(20);
        assert!(!user_ws_heartbeat_ack_timed_out(None, timeout));
        assert!(!user_ws_heartbeat_ack_timed_out(
            Some(Duration::from_secs(19)),
            timeout
        ));
        assert!(user_ws_heartbeat_ack_timed_out(
            Some(Duration::from_secs(20)),
            timeout
        ));
    }

    #[test]
    fn user_ws_reconnect_backoff_is_bounded() {
        assert_eq!(
            next_user_ws_reconnect_delay(USER_WS_RECONNECT_INITIAL_DELAY),
            Duration::from_secs(2)
        );
        assert_eq!(
            next_user_ws_reconnect_delay(Duration::from_secs(16)),
            USER_WS_RECONNECT_MAX_DELAY
        );
        assert_eq!(
            next_user_ws_reconnect_delay(USER_WS_RECONNECT_MAX_DELAY),
            USER_WS_RECONNECT_MAX_DELAY
        );
    }

    #[test]
    fn clob_pagination_fails_closed_on_bad_cursors_metadata_and_overflow() {
        let mut seen = HashSet::new();
        assert_eq!(
            next_clob_reconciliation_cursor(CLOB_TERMINAL_CURSOR, &mut seen, "trades").unwrap(),
            None
        );
        assert_eq!(
            next_clob_reconciliation_cursor("YWJj", &mut seen, "trades").unwrap(),
            Some("YWJj".to_string())
        );
        assert!(next_clob_reconciliation_cursor("YWJj", &mut seen, "trades").is_err());
        assert!(next_clob_reconciliation_cursor("", &mut seen, "trades").is_err());
        assert!(next_clob_reconciliation_cursor(
            &"A".repeat(MAX_CLOB_CURSOR_BYTES + 1),
            &mut seen,
            "trades"
        )
        .is_err());

        assert!(validate_clob_page_metadata("trades", 2, 2, 100).is_ok());
        assert!(validate_clob_page_metadata("trades", 2, 1, 100).is_err());
        assert!(validate_clob_page_metadata("trades", 0, 0, 0).is_err());
        assert!(validate_clob_page_metadata(
            "trades",
            1,
            1,
            MAX_CLOB_RECONCILIATION_ROWS as u64 + 1
        )
        .is_err());
        assert!(checked_clob_row_count(MAX_CLOB_RECONCILIATION_ROWS, 1, "trades").is_err());
        assert!(checked_clob_row_count(usize::MAX, 1, "trades").is_err());
    }

    #[test]
    fn generation_compare_prevents_enable_and_post_races() {
        let mut gate = GlobalLiveEntryGate {
            halted: true,
            reason: "checked_enable_revalidation".to_string(),
            safety_generation: 11,
        };
        let mut state = LiveVenueState::fail_closed();
        state.reconciled_safety_generation = Some(11);
        record_global_entry_halt(&mut gate, "live_geoblock_unreadable");
        assert!(!commit_checked_live_enable(&mut gate, &mut state, 11));
        assert!(gate.halted);
        assert!(!state.manual_entries_enabled);

        gate.halted = false;
        gate.reason = "process_checked_enable".to_string();
        state.manual_entries_enabled = true;
        state.reconciled_safety_generation = Some(gate.safety_generation);
        let admitted_generation = gate.safety_generation;
        record_global_entry_halt(&mut gate, "live_geoblock_blocked");
        assert_eq!(
            commit_live_post_attempt(&mut gate, &mut state, admitted_generation),
            Some(LiveExecutionGateReason::GlobalHalt)
        );

        gate.halted = false;
        gate.reason = "process_checked_enable".to_string();
        state.manual_entries_enabled = true;
        state.reconciled_safety_generation = Some(gate.safety_generation);
        let admitted_generation = gate.safety_generation;
        assert_eq!(
            commit_live_post_attempt(&mut gate, &mut state, admitted_generation),
            None
        );
        assert!(!gate.halted);
        assert!(state.manual_entries_enabled);
        assert_eq!(
            state.reconciled_safety_generation,
            Some(admitted_generation)
        );
    }

    #[test]
    fn cumulative_capital_exposure_enforces_the_daily_hard_cap() {
        assert_eq!(
            live_capital_exposure_gate(dec!(10), Some(dec!(10)), Some(dec!(30))),
            None
        );
        assert_eq!(
            live_capital_exposure_gate(dec!(10.01), Some(dec!(10)), Some(dec!(30))),
            Some(LiveExecutionGateReason::DailyLossLimit)
        );
        assert_eq!(
            live_capital_exposure_gate(dec!(5.01), Some(dec!(10)), Some(dec!(5))),
            Some(LiveExecutionGateReason::OpenNotionalLimit)
        );
    }

    #[tokio::test]
    async fn unbound_admin_reconcile_cannot_supply_process_scope() {
        let venue = LiveVenue::new_for_test(live_config()).unwrap();
        for request in [
            AccountReconcileRequest {
                account_address: None,
                lookback_hours: Some(1),
                process_id: Some(Uuid::new_v4()),
                account_ref: None,
                credential_account_fingerprint_sha256: None,
                dry_run: true,
                token_id: None,
                source: Some("admin".to_string()),
            },
            AccountReconcileRequest {
                account_address: None,
                lookback_hours: Some(1),
                process_id: None,
                account_ref: Some("polymarket-test".to_string()),
                credential_account_fingerprint_sha256: None,
                dry_run: true,
                token_id: None,
                source: Some("admin".to_string()),
            },
            AccountReconcileRequest {
                account_address: None,
                lookback_hours: Some(1),
                process_id: None,
                account_ref: None,
                credential_account_fingerprint_sha256: Some("a".repeat(64)),
                dry_run: true,
                token_id: None,
                source: Some("admin".to_string()),
            },
        ] {
            let error = venue
                .run_account_reconcile(request)
                .await
                .unwrap_err()
                .to_string();
            assert!(error.contains("rejects caller-supplied process scope"));
        }
    }

    #[tokio::test]
    async fn live_status_blocks_entries_when_submit_disabled() {
        let venue = LiveVenue::new_for_test(live_config()).unwrap();
        let status = venue.live_status().await.unwrap();
        assert!(!status.entries_enabled);
        assert_eq!(status.reason.as_deref(), Some("live_order_submit_disabled"));
    }

    #[tokio::test]
    async fn disabled_entries_block_entry_and_metadata_labeled_exit_intents() {
        let process_id = uuid::Uuid::new_v4();
        let venue = LiveVenue::new_for_test(live_config())
            .unwrap()
            .bind_process(process_id, &live_execution())
            .unwrap();
        venue.global_entry_gate.lock().await.halted = false;
        let base = OrderRequest {
            client_order_id: uuid::Uuid::new_v4(),
            process_id: Some(process_id),
            market_id: "market".to_string(),
            token_id: "1".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.50),
            size: dec!(2),
            metadata: json!({"purpose": "entry"}),
        };

        let entry = submit_with_test_guard(&venue, base.clone()).await.unwrap();
        assert_live_gate_rejection(&entry, LiveExecutionGateReason::ManualEnableRequired);

        let mut exit = base;
        exit.metadata = json!({"purpose": "exit"});
        let exit = submit_with_test_guard(&venue, exit).await.unwrap();
        assert_live_gate_rejection(&exit, LiveExecutionGateReason::ManualEnableRequired);
    }

    #[tokio::test]
    async fn disabled_entries_block_non_exit_intents() {
        let process_id = uuid::Uuid::new_v4();
        let venue = LiveVenue::new_for_test(live_config())
            .unwrap()
            .bind_process(process_id, &live_execution())
            .unwrap();
        venue.global_entry_gate.lock().await.halted = false;
        let mut request = OrderRequest {
            client_order_id: uuid::Uuid::new_v4(),
            process_id: Some(process_id),
            market_id: "market".to_string(),
            token_id: "1".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.50),
            size: dec!(2),
            metadata: json!({"execution_intent": "risk_reduction"}),
        };

        for intent in ["risk_reduction", "admin_manual"] {
            request.client_order_id = uuid::Uuid::new_v4();
            request.metadata = json!({"execution_intent": intent});
            let order = submit_with_test_guard(&venue, request.clone())
                .await
                .unwrap();
            assert_live_gate_rejection(&order, LiveExecutionGateReason::ManualEnableRequired);
        }
    }

    #[tokio::test]
    async fn per_order_risk_denial_is_a_zero_post_nonfatal_record() {
        let process_id = Uuid::new_v4();
        let venue = LiveVenue::new_for_test(live_config())
            .unwrap()
            .bind_process(process_id, &live_execution())
            .unwrap();
        let request = OrderRequest {
            client_order_id: Uuid::new_v4(),
            process_id: Some(process_id),
            market_id: "market".to_string(),
            token_id: "1".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.75),
            size: dec!(3),
            metadata: json!({"execution_intent": "entry"}),
        };

        let order = submit_with_test_guard(&venue, request).await.unwrap();

        assert_live_gate_rejection(&order, LiveExecutionGateReason::PerOrderNotionalLimit);
    }

    #[tokio::test]
    async fn omitted_per_order_limit_does_not_create_a_risk_gate() {
        let process_id = Uuid::new_v4();
        let mut execution = live_execution();
        execution.max_order_notional_usd = None;
        let venue = LiveVenue::new_for_test(live_config())
            .unwrap()
            .bind_process(process_id, &execution)
            .unwrap();
        let request = OrderRequest {
            client_order_id: Uuid::new_v4(),
            process_id: Some(process_id),
            market_id: "market".to_string(),
            token_id: "1".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.75),
            size: dec!(3),
            metadata: json!({"execution_intent": "entry"}),
        };

        let order = submit_with_test_guard(&venue, request).await.unwrap();

        assert_live_gate_rejection(&order, LiveExecutionGateReason::GlobalHalt);
    }

    #[tokio::test]
    async fn process_bound_venue_rejects_missing_or_mismatched_process_identity() {
        let process_id = uuid::Uuid::new_v4();
        let venue = LiveVenue::new_for_test(live_config())
            .unwrap()
            .bind_process(process_id, &live_execution())
            .unwrap();
        let request = OrderRequest {
            client_order_id: uuid::Uuid::new_v4(),
            process_id: Some(uuid::Uuid::new_v4()),
            market_id: "market".to_string(),
            token_id: "1".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.50),
            size: dec!(2),
            metadata: json!({"purpose": "entry"}),
        };
        let error = venue.submit_order(request).await.unwrap_err().to_string();
        assert!(error.contains("must match bound process"));
    }

    #[tokio::test]
    async fn manual_enable_remains_fail_closed_before_first_successful_reconcile() {
        let venue = LiveVenue::new_for_test(live_config())
            .unwrap()
            .bind_process(uuid::Uuid::new_v4(), &live_execution())
            .unwrap();
        seed_fresh_unblocked_geoblock(&venue).await;

        let error = venue
            .set_live_entries_enabled(true, None)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("live persistence store is not configured"));
        let status = venue.live_status().await.unwrap();
        assert!(!status.entries_enabled);
        assert!(!status.idempotency_clean);
        assert_eq!(status.last_rest_reconcile_age_secs, None);
        assert_eq!(
            status.reason.as_deref(),
            Some("live_global_halt:checked_enable_revalidation")
        );
    }

    #[tokio::test]
    async fn wallet_wide_halt_closes_every_bound_submit_path() {
        let root = LiveVenue::new_for_test(live_config()).unwrap();
        let process_id = Uuid::new_v4();
        let bound = root.bind_process(process_id, &live_execution()).unwrap();
        let identity =
            canonical_configured_account_identity(&bound.config, "polymarket-test").unwrap();
        seed_fresh_unblocked_geoblock(&bound).await;
        {
            let mut transport = bound.transport_state.lock().await;
            transport.user_ws_connected = true;
            transport.last_user_ws_pong_at = Some(Utc::now());
        }
        {
            let mut state = bound.readiness_state.lock().await;
            state.last_rest_reconcile_at = Some(Utc::now());
            state.idempotency_clean = true;
            state.unresolved_live_order_count = 0;
            state.process_accounting_proven = true;
            state.process_accounting_status = "proven".to_string();
            state.credential_account_fingerprint_sha256 = Some(identity.fingerprint_sha256);
            state.reconciled_safety_generation = Some(0);
            state.manual_entries_enabled = true;
            state.manual_entries_reason = None;
        }
        {
            let mut global = bound.global_entry_gate.lock().await;
            global.halted = false;
            global.reason = "process_checked_enable".to_string();
        }
        assert!(bound.live_status().await.unwrap().entries_enabled);

        root.set_live_entries_enabled(false, Some("test_global_halt".to_string()))
            .await
            .unwrap();
        let status = bound.live_status().await.unwrap();
        assert!(!status.entries_enabled);
        assert_eq!(
            status.reason.as_deref(),
            Some("live_global_halt:test_global_halt")
        );

        let exit = OrderRequest {
            client_order_id: Uuid::new_v4(),
            process_id: Some(process_id),
            market_id: "market".to_string(),
            token_id: "1".to_string(),
            side: OrderSide::Sell,
            order_type: OrderType::Fok,
            price: dec!(0.50),
            size: dec!(1),
            metadata: json!({"purpose": "exit"}),
        };
        let order = submit_with_test_guard(&bound, exit).await.unwrap();
        assert_live_gate_rejection(&order, LiveExecutionGateReason::GlobalHalt);
    }

    #[tokio::test]
    async fn process_bound_live_submit_cannot_bypass_adjacent_guard() {
        let process_id = Uuid::new_v4();
        let venue = LiveVenue::new_for_test(live_config())
            .unwrap()
            .bind_process(process_id, &live_execution())
            .unwrap();
        let request = OrderRequest {
            client_order_id: Uuid::new_v4(),
            process_id: Some(process_id),
            market_id: "market".to_string(),
            token_id: "1".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.50),
            size: dec!(1),
            metadata: json!({"purpose": "entry"}),
        };

        let error = venue.submit_order(request).await.unwrap_err().to_string();

        assert!(error.contains("requires an adjacent pre-POST guard"));
    }

    #[tokio::test]
    async fn process_bound_live_submit_rejects_an_explicitly_missing_guard_before_io() {
        let process_id = Uuid::new_v4();
        let venue = LiveVenue::new_for_test(live_config())
            .unwrap()
            .bind_process(process_id, &live_execution())
            .unwrap();
        let request = OrderRequest {
            client_order_id: Uuid::new_v4(),
            process_id: Some(process_id),
            market_id: "market".to_string(),
            token_id: "1".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.50),
            size: dec!(1),
            metadata: json!({"purpose": "entry"}),
        };

        let error = venue
            .submit_order_with_pre_post_guard(request, None)
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("requires an adjacent pre-POST guard"));
    }

    #[tokio::test]
    async fn repeated_metadata_labeled_exit_attempts_remain_zero_post_while_halted() {
        let process_id = Uuid::new_v4();
        let venue = LiveVenue::new_for_test(live_config())
            .unwrap()
            .bind_process(process_id, &live_execution())
            .unwrap();
        let request = OrderRequest {
            client_order_id: Uuid::new_v4(),
            process_id: Some(process_id),
            market_id: "market".to_string(),
            token_id: "1".to_string(),
            side: OrderSide::Sell,
            order_type: OrderType::Fok,
            price: dec!(0.50),
            size: dec!(1),
            metadata: json!({"purpose": "exit"}),
        };

        for _ in 0..2 {
            let order = submit_with_test_guard(&venue, request.clone())
                .await
                .unwrap();
            assert_live_gate_rejection(&order, LiveExecutionGateReason::GlobalHalt);
        }
    }

    #[test]
    fn canonical_identity_is_signature_aware_and_fingerprint_is_bounded() {
        let eoa = canonical_configured_account_identity(&live_config(), "polymarket-test").unwrap();
        assert_eq!(eoa.account_address, eoa.signer_address);
        assert_eq!(eoa.fingerprint_sha256.len(), 64);

        let mut poly1271 = live_config();
        poly1271.signature_type = Some("3".to_string());
        poly1271.funder_address = Some("0x0000000000000000000000000000000000000002".to_string());
        let poly1271 = canonical_configured_account_identity(&poly1271, "polymarket-test").unwrap();
        assert_ne!(poly1271.account_address, poly1271.signer_address);
        assert_eq!(
            poly1271.account_address,
            "0x0000000000000000000000000000000000000002"
        );
        assert_eq!(poly1271.fingerprint_sha256.len(), 64);
    }

    #[test]
    fn canonical_identity_fingerprint_survives_api_key_rotation_but_not_account_rotation() {
        let config = live_config();
        let baseline = canonical_configured_account_identity(&config, "polymarket-test")
            .unwrap()
            .fingerprint_sha256;

        let mut rotated_api = config.clone();
        rotated_api.clob_api_key = Some("00000000-0000-0000-0000-000000000002".to_string());
        rotated_api.clob_secret = Some("rotated-secret".to_string());
        rotated_api.clob_passphrase = Some("rotated-passphrase".to_string());
        assert_eq!(
            canonical_configured_account_identity(&rotated_api, "polymarket-test")
                .unwrap()
                .fingerprint_sha256,
            baseline
        );

        assert_ne!(
            canonical_configured_account_identity(&config, "polymarket-other")
                .unwrap()
                .fingerprint_sha256,
            baseline
        );
    }

    #[test]
    fn definitive_client_rejections_are_safe_nonfatal_order_outcomes() {
        use polymarket_client_sdk_v2::error::{Error as SdkError, Method, StatusCode};

        let rejected = anyhow::Error::new(SdkError::status(
            StatusCode::BAD_REQUEST,
            Method::POST,
            "/order".to_string(),
            "maker address not allowed, please use the deposit wallet flow",
        ));
        assert!(is_definitive_live_submit_error(&rejected));

        let duplicate_or_ambiguous = anyhow::Error::new(SdkError::status(
            StatusCode::CONFLICT,
            Method::POST,
            "/order".to_string(),
            "conflict",
        ));
        assert!(!is_definitive_live_submit_error(&duplicate_or_ambiguous));
        assert!(!is_definitive_live_submit_error(&anyhow::anyhow!(
            "connection reset after write"
        )));
    }

    #[test]
    fn process_binding_rejects_nil_identity() {
        let venue = LiveVenue::new_for_test(live_config()).unwrap();
        assert!(venue.bind_process(Uuid::nil(), &live_execution()).is_err());
        let mut blank_account = live_execution();
        blank_account.account_ref = Some("   ".to_string());
        assert!(venue.bind_process(Uuid::new_v4(), &blank_account).is_err());

        let process_id = Uuid::new_v4();
        let mut padded_account = live_execution();
        padded_account.account_ref = Some("  polymarket-test  ".to_string());
        let bound = venue.bind_process(process_id, &padded_account).unwrap();
        assert_eq!(bound.bound_process_id(), Some(process_id));
        assert_eq!(bound.bound_account_ref(), Some("polymarket-test"));
    }

    #[test]
    fn live_config_requires_transport_endpoints() {
        let mut config = live_config();
        config.user_ws_url.clear();
        assert!(config.validate_for_live().is_err());

        let mut config = live_config();
        config.clob_api_base_url.clear();
        assert!(config.validate_for_live().is_err());
    }
}
