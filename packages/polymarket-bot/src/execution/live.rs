use std::{str::FromStr, sync::Arc, time::Duration};

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
        },
        Client as SdkClient, Config as SdkConfig,
    },
    derive_proxy_wallet, derive_safe_wallet,
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
    config::LiveExecutionConfig,
    execution::{
        ExecutionVenue, LiveIdentityDiagnostics, LiveOrderDryRunDiagnostics,
        LiveOrderDryRunRequest, LivePoly1271FunderProbeCandidate, LivePoly1271FunderProbeRequest,
        LivePoly1271FunderProbeResponse, LiveVenueStatus, LiveWalletAddressDiagnostics,
        LiveWalletCandidateAddressDiagnostics, LiveWalletTokenBalances, ReconciliationReport,
    },
    idempotency::{event_hash, order_request_notional_key},
    models::{ConversionRequest, ConversionResult, FillRecord, OrderRecord, OrderRequest},
    models::{FillSource, OrderSide, OrderState, OrderType},
    store::Store,
};

type AuthenticatedClient = SdkClient<Authenticated<Normal>>;

const DEFAULT_POLYGON_RPC_URL: &str = "https://polygon-bor-rpc.publicnode.com";
const DEFAULT_RELAYER_BASE_URL: &str = "https://relayer-v2.polymarket.com";
const PUSD_ADDRESS: &str = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB";
const USDC_E_ADDRESS: &str = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174";
const NATIVE_USDC_ADDRESS: &str = "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359";

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
    state: Arc<Mutex<LiveVenueState>>,
}

#[derive(Debug, Clone)]
struct LiveVenueState {
    live_confirmed: bool,
    user_ws_connected: bool,
    last_user_ws_pong_at: Option<DateTime<Utc>>,
    last_rest_reconcile_at: Option<DateTime<Utc>>,
    idempotency_clean: bool,
    unresolved_live_order_count: usize,
    manual_entries_enabled: bool,
    manual_entries_reason: Option<String>,
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
    pub fn new(config: LiveExecutionConfig, clob_base_url: String, store: Store) -> Result<Self> {
        config.validate_for_live()?;
        let venue = Self {
            config,
            clob_base_url,
            store: Some(store),
            state: Arc::new(Mutex::new(LiveVenueState {
                live_confirmed: true,
                user_ws_connected: false,
                last_user_ws_pong_at: None,
                last_rest_reconcile_at: None,
                idempotency_clean: true,
                unresolved_live_order_count: 0,
                manual_entries_enabled: false,
                manual_entries_reason: Some("manual_enable_required".to_string()),
            })),
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
            state: Arc::new(Mutex::new(LiveVenueState {
                live_confirmed: true,
                user_ws_connected: false,
                last_user_ws_pong_at: None,
                last_rest_reconcile_at: None,
                idempotency_clean: true,
                unresolved_live_order_count: 0,
                manual_entries_enabled: false,
                manual_entries_reason: Some("manual_enable_required".to_string()),
            })),
        })
    }

    pub fn user_ws_subscription(&self, markets: &[String]) -> serde_json::Value {
        json!({
            "auth": {
                "apiKey": self.config.clob_api_key.as_deref().unwrap_or(""),
                "secret": self.config.clob_secret.as_deref().unwrap_or(""),
                "passphrase": self.config.clob_passphrase.as_deref().unwrap_or("")
            },
            "markets": markets,
            "type": "user"
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
        let order_state = if fill.size >= order.request.size {
            OrderState::Filled
        } else {
            OrderState::PartiallyFilled
        };
        store
            .mark_order_filled(
                &order.order_id,
                order_state,
                json!({
                    "source": "user_ws",
                    "event_type": event.event_type,
                    "venue_event_id": event.venue_event_id,
                    "venue_trade_id": event.venue_trade_id,
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
            }
        }
        Ok(processed)
    }

    fn spawn_user_ws_task_if_enabled(&self) {
        if !self.config.user_ws_enabled || !self.config.user_ws_auth_available() {
            return;
        }
        let config = self.config.clone();
        let Some(store) = self.store.clone() else {
            return;
        };
        let state = self.state.clone();
        tokio::spawn(async move {
            loop {
                if let Err(error) = run_user_ws_once(&config, &store, &state).await {
                    warn!(error = %error, "Polymarket live user websocket disconnected");
                    let mut state = state.lock().await;
                    state.user_ws_connected = false;
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
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

    fn live_submit_unavailable(&self, request: &OrderRequest) -> anyhow::Error {
        anyhow::anyhow!(
            "live order submit is disabled or not ready; client_order_id={} notional={} purpose={}",
            request.client_order_id,
            order_request_notional_key(request),
            request
                .metadata
                .get("purpose")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown")
        )
    }
}

async fn run_user_ws_once(
    config: &LiveExecutionConfig,
    store: &Store,
    state: &Arc<Mutex<LiveVenueState>>,
) -> Result<()> {
    let (mut ws, _) = connect_async(&config.user_ws_url)
        .await
        .context("failed to connect Polymarket user websocket")?;
    ws.send(Message::Text(
        json!({
            "auth": {
                "apiKey": config.clob_api_key.as_deref().unwrap_or(""),
                "secret": config.clob_secret.as_deref().unwrap_or(""),
                "passphrase": config.clob_passphrase.as_deref().unwrap_or("")
            },
            "markets": &config.user_ws_markets,
            "type": "user"
        })
        .to_string()
        .into(),
    ))
    .await
    .context("failed to subscribe Polymarket user websocket")?;
    {
        let mut state = state.lock().await;
        state.user_ws_connected = true;
        state.last_user_ws_pong_at = Some(Utc::now());
    }

    let mut heartbeat = tokio::time::interval(Duration::from_secs(10));
    loop {
        tokio::select! {
            _ = heartbeat.tick() => {
                ws.send(Message::Ping(Vec::new().into()))
                    .await
                    .context("failed to ping Polymarket user websocket")?;
            }
            message = ws.next() => {
                let Some(message) = message else {
                    break;
                };
                match message.context("failed to read Polymarket user websocket")? {
                    Message::Text(text) => {
                        let text = text.to_string();
                        let payload: serde_json::Value = serde_json::from_str(&text)
                            .unwrap_or_else(|_| json!({ "event_type": "raw", "message": text }));
                        let event = LiveVenue::parse_user_event(payload);
                        let inserted = store.insert_live_venue_event(&event).await?;
                        if let Err(error) = LiveVenue::persist_fill_from_live_event(&store, &event).await {
                            warn!(error = %error, "failed to persist Polymarket live user websocket fill event");
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
                        state.last_user_ws_pong_at = Some(Utc::now());
                    }
                    Message::Ping(payload) => {
                        ws.send(Message::Pong(payload)).await.ok();
                        let mut state = state.lock().await;
                        state.last_user_ws_pong_at = Some(Utc::now());
                    }
                    Message::Pong(_) => {
                        let mut state = state.lock().await;
                        state.last_user_ws_pong_at = Some(Utc::now());
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
        }
    }
    Ok(())
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
            signal_id: None,
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

fn fill_record_from_trade(
    order_id: &str,
    process_id: Option<Uuid>,
    trade: TradeResponse,
) -> Result<FillRecord> {
    let price = local_decimal(trade.price)?;
    let size = local_decimal(trade.size)?;
    let fee_rate_bps = local_decimal(trade.fee_rate_bps)?;
    let fee = price * size * fee_rate_bps / Decimal::from(10_000);
    Ok(FillRecord {
        fill_id: Uuid::new_v5(
            &Uuid::NAMESPACE_URL,
            format!("polymarket:trade:{}", trade.id).as_bytes(),
        ),
        process_id,
        order_id: order_id.to_string(),
        token_id: trade.asset_id.to_string(),
        price,
        size,
        fee,
        source: FillSource::Live,
        filled_at: trade.match_time,
    })
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
        let price = json_decimal(&event.raw_payload, "price")?;
        let size = live_event_matched_size_for_order(&event.raw_payload, &candidate)?
            .unwrap_or(json_decimal(&event.raw_payload, "size")?);
        let fee_rate_bps =
            json_decimal(&event.raw_payload, "fee_rate_bps").unwrap_or(Decimal::ZERO);
        let fee = price * size * fee_rate_bps / Decimal::from(10_000);
        let token_id = live_event_asset_id_for_order(&event.raw_payload, &candidate)
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

fn live_event_matched_size_for_order(payload: &Value, order_id: &str) -> Result<Option<Decimal>> {
    let Some(maker_orders) = payload.get("maker_orders").and_then(Value::as_array) else {
        return Ok(None);
    };
    for maker_order in maker_orders {
        if json_str(maker_order, "order_id") == Some(order_id) {
            return json_decimal(maker_order, "matched_amount").map(Some);
        }
    }
    Ok(None)
}

fn live_event_asset_id_for_order(payload: &Value, order_id: &str) -> Option<String> {
    let maker_orders = payload.get("maker_orders").and_then(Value::as_array)?;
    for maker_order in maker_orders {
        if json_str(maker_order, "order_id") == Some(order_id) {
            return json_str(maker_order, "asset_id").map(str::to_string);
        }
    }
    None
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

#[async_trait]
impl ExecutionVenue for LiveVenue {
    async fn submit_order(&self, request: OrderRequest) -> Result<OrderRecord> {
        let notional = request.price * request.size;
        if notional > self.config.max_order_notional_usd {
            bail!(
                "live order notional {} exceeds cap {}",
                notional,
                self.config.max_order_notional_usd
            );
        }
        if !self.config.order_submit_enabled {
            return Err(self.live_submit_unavailable(&request));
        }
        if !self.live_status().await?.entries_enabled {
            return Err(self.live_submit_unavailable(&request));
        }
        let store = self.store()?;
        if let Some(existing) = self
            .store()?
            .find_order_by_client_order_id(request.client_order_id)
            .await?
        {
            if !matches!(existing.state, OrderState::Rejected | OrderState::Unknown) {
                return Ok(existing);
            }
        }

        store.create_pending_order(&request).await?;
        let private_key = self
            .config
            .private_key
            .as_deref()
            .context("missing private key")?;
        let signer = LocalSigner::from_str(private_key)
            .context("failed to parse POLYMARKET_PRIVATE_KEY")?
            .with_chain_id(Some(POLYGON));
        let client = self.authenticated_client().await?;
        let submit_result = async {
            let token_id =
                U256::from_str(&request.token_id).context("failed to parse CLOB token_id")?;
            let response = client
                .limit_order()
                .token_id(token_id)
                .side(sdk_side(request.side))
                .price(sdk_decimal(request.price)?)
                .size(sdk_decimal(request.size)?)
                .order_type(sdk_order_type(request.order_type)?)
                .build_sign_and_post(&signer)
                .await
                .context("Polymarket CLOB order submit failed")?;
            Ok::<PostOrderResponse, anyhow::Error>(response)
        }
        .await;

        match submit_result {
            Ok(response) if response.success => {
                let raw_ack = post_order_response_payload(&response);
                store
                    .mark_order_submitted(request.client_order_id, &response.order_id, raw_ack)
                    .await
            }
            Ok(response) => {
                let raw = post_order_response_payload(&response);
                let _ = self
                    .store()?
                    .mark_order_submit_failed(request.client_order_id, "venue_rejected", raw)
                    .await;
                bail!(
                    "Polymarket CLOB rejected order {}: {}",
                    request.client_order_id,
                    response
                        .error_msg
                        .unwrap_or_else(|| "unknown rejection".to_string())
                )
            }
            Err(error) => {
                let error_chain = format!("{error:#}");
                let _ = self
                    .store()?
                    .mark_order_submit_failed(
                        request.client_order_id,
                        "submit_error",
                        json!({
                            "error": error.to_string(),
                            "error_chain": error_chain
                        }),
                    )
                    .await;
                Err(error)
            }
        }
    }

    async fn cancel_order(&self, order_id: &str) -> Result<OrderRecord> {
        let store = self.store()?;
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
        let page = client
            .orders(&OrdersRequest::builder().build(), None)
            .await
            .context("Polymarket CLOB open orders fetch failed")?;
        page.data
            .into_iter()
            .map(order_record_from_open_order)
            .collect()
    }

    async fn convert_negative_risk(&self, _request: ConversionRequest) -> Result<ConversionResult> {
        bail!("live negative-risk conversion is not enabled for production canary")
    }

    async fn split_ctf(&self, _market_id: &str, _size: Decimal) -> Result<ConversionResult> {
        bail!("live CTF split is not enabled for production canary")
    }

    async fn merge_ctf(&self, _market_id: &str, _size: Decimal) -> Result<ConversionResult> {
        bail!("live CTF merge is not enabled for production canary")
    }

    async fn reconcile(&self) -> Result<ReconciliationReport> {
        let fills_backfilled = self.backfill_fills_from_live_events().await.unwrap_or_else(|error| {
            warn!(error = %error, "failed to backfill fills from Polymarket live user websocket events");
            0
        });
        let open_orders = self.get_open_orders().await.unwrap_or_default();
        let balances_checked = self.get_balances().await.is_ok();
        let unresolved = open_orders.len();
        {
            let mut state = self.state.lock().await;
            state.last_rest_reconcile_at = Some(Utc::now());
            state.unresolved_live_order_count = unresolved;
            state.idempotency_clean = true;
        }
        let report = ReconciliationReport {
            open_orders: unresolved,
            balances_checked,
            mismatches_found: 0,
            unresolved_count: unresolved,
            checked_at: Utc::now(),
        };
        self.store()?
            .insert_live_reconciliation_run(
                "completed",
                report.open_orders as i32,
                0,
                if report.balances_checked { 1 } else { 0 },
                report.mismatches_found as i32,
                fills_backfilled as i32,
                report.unresolved_count as i32,
                serde_json::to_value(&report)?,
            )
            .await?;
        Ok(report)
    }

    async fn fills_for_order(&self, order_id: &str) -> Result<Vec<FillRecord>> {
        let store = self.store()?;
        let client = self.authenticated_client().await?;
        let order_process_id = store
            .find_order_by_venue_order_id(order_id)
            .await?
            .and_then(|order| order.request.process_id);
        let page = client
            .trades(&TradesRequest::builder().build(), None)
            .await
            .context("Polymarket CLOB trades fetch failed")?;
        let mut fills = Vec::new();
        for trade in page
            .data
            .into_iter()
            .filter(|trade| trade.taker_order_id == order_id)
        {
            let fill = fill_record_from_trade(order_id, order_process_id, trade)?;
            store.insert_fill(&fill).await?;
            fills.push(fill);
        }
        Ok(fills)
    }

    async fn live_status(&self) -> Result<LiveVenueStatus> {
        let state = self.state.lock().await;
        let now = Utc::now();
        let last_user_ws_pong_age_secs = state
            .last_user_ws_pong_at
            .map(|value| (now - value).num_seconds());
        let last_rest_reconcile_age_secs = state
            .last_rest_reconcile_at
            .map(|value| (now - value).num_seconds());
        let user_ws_fresh = !self.config.user_ws_enabled
            || last_user_ws_pong_age_secs
                .map(|age| age <= self.config.user_ws_stale.as_secs() as i64)
                .unwrap_or(false);
        let rest_fresh = last_rest_reconcile_age_secs
            .map(|age| age <= self.config.stale_reconcile.as_secs() as i64)
            .unwrap_or(false);
        let entries_enabled = state.live_confirmed
            && self.config.order_submit_enabled
            && self.config.submit_auth_available()
            && state.manual_entries_enabled
            && user_ws_fresh
            && rest_fresh
            && state.idempotency_clean
            && state.unresolved_live_order_count == 0;
        let reason = if entries_enabled {
            None
        } else if !self.config.order_submit_enabled {
            Some("live_order_submit_disabled".to_string())
        } else if !self.config.submit_auth_available() {
            Some("live_submit_auth_missing".to_string())
        } else if !state.manual_entries_enabled {
            state
                .manual_entries_reason
                .clone()
                .or_else(|| Some("manual_enable_required".to_string()))
        } else if !user_ws_fresh {
            Some("live_user_ws_stale_or_disconnected".to_string())
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
            live_confirmed: state.live_confirmed,
            order_submit_enabled: self.config.order_submit_enabled,
            user_ws_enabled: self.config.user_ws_enabled,
            user_ws_connected: state.user_ws_connected,
            last_user_ws_pong_age_secs,
            last_rest_reconcile_age_secs,
            idempotency_clean: state.idempotency_clean,
            unresolved_live_order_count: state.unresolved_live_order_count,
            max_order_notional_usd: self.config.max_order_notional_usd,
            max_open_notional_usd: self.config.max_open_notional_usd,
            entries_enabled,
            reason,
        })
    }

    async fn live_identity_diagnostics(&self) -> Result<LiveIdentityDiagnostics> {
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
            signer_address,
            configured_funder_address: self.config.funder_address.clone(),
            configured_signature_type: self.config.signature_type.clone(),
            resolved_signature_type: Some(format!("{signature_type:?}")),
            authenticated_client_address: None,
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
        diagnostics.authenticated_client_address = Some(client.address().to_checksum(None));

        match client.api_keys().await {
            Ok(_) => diagnostics.api_keys_readable = true,
            Err(error) => diagnostics.api_keys_error = Some(error.to_string()),
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
            }
            Err(error) => diagnostics.balance_allowance_error = Some(error.to_string()),
        }

        match client.orders(&OrdersRequest::builder().build(), None).await {
            Ok(page) => {
                diagnostics.open_orders_readable = true;
                diagnostics.open_orders_count = Some(page.data.len());
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

    async fn set_live_entries_enabled(
        &self,
        enabled: bool,
        reason: Option<String>,
    ) -> Result<LiveVenueStatus> {
        let mut state = self.state.lock().await;
        state.manual_entries_enabled = enabled;
        state.manual_entries_reason = if enabled {
            None
        } else {
            Some(reason.unwrap_or_else(|| "manual_disable".to_string()))
        };
        drop(state);
        self.live_status().await
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

    fn live_config() -> LiveExecutionConfig {
        LiveExecutionConfig {
            order_submit_enabled: false,
            max_order_notional_usd: dec!(2),
            max_open_notional_usd: dec!(30),
            max_daily_loss_usd: dec!(10),
            max_open_positions: 6,
            require_exit_book: true,
            require_idempotency_clean: true,
            user_ws_enabled: true,
            user_ws_url: "wss://ws-subscriptions-clob.polymarket.com/ws/user".to_string(),
            user_ws_markets: Vec::new(),
            clob_api_base_url: "https://clob-v2.polymarket.com".to_string(),
            user_ws_stale: std::time::Duration::from_secs(20),
            reconcile_interval: std::time::Duration::from_secs(30),
            stale_reconcile: std::time::Duration::from_secs(60),
            clob_api_key: Some("key".to_string()),
            clob_secret: Some("secret".to_string()),
            clob_passphrase: Some("pass".to_string()),
            private_key: Some("private".to_string()),
            funder_address: Some("0xabc".to_string()),
            signature_type: Some("1".to_string()),
        }
    }

    #[test]
    fn user_event_hash_is_stable() {
        let raw = json!({"event_type":"trade","id":"t1","status":"CONFIRMED"});
        let event = LiveVenue::parse_user_event(raw);
        assert_eq!(event.hash(), event.hash());
        assert_eq!(event.event_type, "trade");
        assert_eq!(event.venue_trade_id.as_deref(), Some("t1"));
    }

    #[tokio::test]
    async fn live_status_blocks_entries_when_submit_disabled() {
        let venue = LiveVenue::new_for_test(live_config()).unwrap();
        let status = venue.live_status().await.unwrap();
        assert!(!status.entries_enabled);
        assert_eq!(status.reason.as_deref(), Some("live_order_submit_disabled"));
    }

    #[test]
    fn live_config_rejects_unsafe_caps() {
        let mut config = live_config();
        config.max_order_notional_usd = dec!(2.01);
        assert!(config.validate_for_live().is_err());
    }
}
