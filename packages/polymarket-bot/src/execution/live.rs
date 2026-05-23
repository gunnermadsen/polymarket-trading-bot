use std::{str::FromStr, sync::Arc, time::Duration};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures_util::{SinkExt, StreamExt};
use polymarket_client_sdk_v2::{
    auth::{state::Authenticated, Credentials, LocalSigner, Normal, Signer as _},
    clob::{
        types::{
            request::{BalanceAllowanceRequest, OrdersRequest, TradesRequest},
            response::{OpenOrderResponse, PostOrderResponse, TradeResponse},
            AssetType, OrderStatusType, OrderType as SdkOrderType, Side as SdkSide, SignatureType,
        },
        Client as SdkClient, Config as SdkConfig,
    },
    types::{Address, Decimal as SdkDecimal, U256},
    POLYGON,
};
use rust_decimal::Decimal;
use serde_json::json;
use tokio::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::{
    config::LiveExecutionConfig,
    execution::{ExecutionVenue, LiveIdentityDiagnostics, LiveVenueStatus, ReconciliationReport},
    idempotency::{event_hash, order_request_notional_key},
    models::{ConversionRequest, ConversionResult, FillRecord, OrderRecord, OrderRequest},
    models::{FillSource, OrderSide, OrderState, OrderType},
    store::Store,
};

type AuthenticatedClient = SdkClient<Authenticated<Normal>>;

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
                0,
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
