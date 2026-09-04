use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        atomic::{AtomicI64, AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::Duration,
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::{metadata::MetadataValue, transport::Channel, Request};
use uuid::Uuid;

use crate::btc::{
    BinanceOneSecondKline, BinanceOpenInterestPoint, BookRegistry, BtcIntervalMarket, BtcOutcome,
    BtcRepository, ChainlinkTwap60Point, PolygonOraclePoint, RealtimeState, ReferencePriceSource,
    ReferencePriceTick,
};

pub mod proto {
    tonic::include_proto!("capitonic.marketdata.v1");
}

use proto::{
    market_data_message::Message, market_data_stream_client::MarketDataStreamClient,
    MarketDataEvent, ProductSelector, SubscriptionCommand,
};

pub const CONTRACT_VERSION: u32 = 1;
pub const PRODUCT_MARKETS: &str = "polymarket_btc_five_minute_market_contracts";
pub const PRODUCT_BOOKS: &str = "polymarket_btc_five_minute_orderbooks";
pub const PRODUCT_RESOLUTIONS: &str = "polymarket_btc_five_minute_resolutions";
pub const PRODUCT_CHAINLINK: &str = "polymarket_rtds_chainlink_reference_price";
pub const PRODUCT_TWAP: &str = "polymarket_chainlink_btcusd_twap";
pub const PRODUCT_BINANCE_1S: &str = "binance_spot_btcusdt_one_second_ohlcv";
pub const PRODUCT_POLYGON_ORACLE: &str = "polygon_chainlink_btcusd_oracle";
pub const PRODUCT_BINANCE_OPEN_INTEREST: &str = "binance_futures_btcusdt_open_interest";

pub const DEFAULT_BTC_PRODUCTS: [&str; 8] = [
    PRODUCT_MARKETS,
    PRODUCT_BOOKS,
    PRODUCT_RESOLUTIONS,
    PRODUCT_CHAINLINK,
    PRODUCT_TWAP,
    PRODUCT_BINANCE_1S,
    PRODUCT_POLYGON_ORACLE,
    PRODUCT_BINANCE_OPEN_INTEREST,
];

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct SourceSelector {
    pub key: String,
    #[serde(default = "default_contract_version")]
    pub contract_version: u32,
    #[serde(default = "default_required")]
    pub required: bool,
    #[serde(default)]
    pub maximum_age_ms: Option<u64>,
    #[serde(default)]
    pub require_sequence_integrity: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SourceSelectorInput {
    Key(String),
    Config(SourceSelectorConfig),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceSelectorConfig {
    key: String,
    #[serde(default = "default_contract_version")]
    contract_version: u32,
    #[serde(default = "default_required")]
    required: bool,
    #[serde(default)]
    maximum_age_ms: Option<u64>,
    #[serde(default)]
    require_sequence_integrity: bool,
}

impl<'de> Deserialize<'de> for SourceSelector {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(match SourceSelectorInput::deserialize(deserializer)? {
            SourceSelectorInput::Key(key) => Self {
                key,
                contract_version: CONTRACT_VERSION,
                required: true,
                maximum_age_ms: None,
                require_sequence_integrity: true,
            },
            SourceSelectorInput::Config(config) => Self {
                key: config.key,
                contract_version: config.contract_version,
                required: config.required,
                maximum_age_ms: config.maximum_age_ms,
                require_sequence_integrity: config.require_sequence_integrity,
            },
        })
    }
}

const fn default_contract_version() -> u32 {
    CONTRACT_VERSION
}
const fn default_required() -> bool {
    true
}

impl SourceSelector {
    pub fn validate(&self) -> Result<()> {
        if !DEFAULT_BTC_PRODUCTS.contains(&self.key.as_str()) {
            bail!("unsupported BTC market-data source {}", self.key);
        }
        if self.contract_version != CONTRACT_VERSION {
            bail!("unsupported contract version for {}", self.key);
        }
        if self
            .maximum_age_ms
            .is_some_and(|age| age == 0 || age > 600_000)
        {
            bail!("source maximum_age_ms must be between 1 and 600000");
        }
        Ok(())
    }

    fn effective_maximum_age_ms(&self) -> u64 {
        self.maximum_age_ms.unwrap_or(match self.key.as_str() {
            PRODUCT_BOOKS | PRODUCT_CHAINLINK | PRODUCT_BINANCE_1S => 10_000,
            PRODUCT_BINANCE_OPEN_INTEREST => 360_000,
            PRODUCT_TWAP | PRODUCT_POLYGON_ORACLE => 120_000,
            PRODUCT_MARKETS | PRODUCT_RESOLUTIONS => 0,
            _ => 60_000,
        })
    }
}

pub fn legacy_default_sources() -> Vec<SourceSelector> {
    DEFAULT_BTC_PRODUCTS
        .iter()
        .map(|key| SourceSelector {
            key: (*key).to_owned(),
            contract_version: CONTRACT_VERSION,
            required: true,
            maximum_age_ms: None,
            require_sequence_integrity: true,
        })
        .collect()
}

fn validate_selectors(selectors: &[SourceSelector]) -> Result<()> {
    if selectors.is_empty() {
        bail!("BTC process sources must not be empty");
    }
    let mut seen = BTreeSet::new();
    for selector in selectors {
        selector.validate()?;
        if !seen.insert(selector.key.as_str()) {
            bail!("duplicate BTC source {}", selector.key);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
struct RouteResponse {
    routes: Vec<WorkerRoute>,
    unresolved: Vec<RouteRejection>,
}
#[derive(Debug, Clone, Deserialize)]
struct WorkerRoute {
    worker_id: String,
    endpoint: String,
    source_revision: String,
    products: Vec<RouteProduct>,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
struct RouteProduct {
    key: String,
    contract_version: u32,
}
#[derive(Debug, Clone, Deserialize)]
struct RouteRejection {
    product: RouteProduct,
    reason: String,
}

#[derive(Default)]
pub struct StreamMetrics {
    connections: AtomicI64,
    desired_connections: AtomicI64,
    reconnects: AtomicU64,
    events: AtomicU64,
    duplicates: AtomicU64,
    gaps: AtomicU64,
    decode_errors: AtomicU64,
    apply_latency_micros: AtomicU64,
    last_event_micros: AtomicI64,
    products: Mutex<BTreeMap<String, ProductStreamMetrics>>,
    product_ready: Mutex<BTreeMap<String, bool>>,
    route_failures: Mutex<BTreeMap<String, u64>>,
}

#[derive(Default)]
struct ProductStreamMetrics {
    events: u64,
    apply_latency_micros: u64,
    last_event_micros: i64,
}

struct ConnectionGauge<'a> {
    metrics: &'a StreamMetrics,
    products: Vec<String>,
}

impl Drop for ConnectionGauge<'_> {
    fn drop(&mut self) {
        self.metrics.connections.fetch_sub(1, Ordering::Relaxed);
        let mut ready = self
            .metrics
            .product_ready
            .lock()
            .expect("stream readiness lock");
        for product in &self.products {
            ready.insert(product.clone(), false);
        }
    }
}

fn route_failure_reason(error: &anyhow::Error) -> &'static str {
    let message = error.to_string();
    if message.contains(" is stale") || message.contains("not contiguous") {
        "required_product_stale"
    } else if message.contains("sequence gap") {
        "sequence_integrity"
    } else if message.contains("rejected market-data subscription")
        || message.contains("unauthenticated")
    {
        "subscription_rejected"
    } else if message.contains("closed market-data stream") {
        "worker_stream_closed"
    } else if message.contains("connect worker") || message.contains("transport error") {
        "worker_connect"
    } else {
        "stream_error"
    }
}

fn topology_signature(routes: &[WorkerRoute]) -> BTreeSet<String> {
    routes
        .iter()
        .flat_map(|route| {
            route.products.iter().map(|product| {
                format!(
                    "{}|{}|{}|{}|{}",
                    route.worker_id,
                    route.endpoint,
                    route.source_revision,
                    product.key,
                    product.contract_version
                )
            })
        })
        .collect()
}

static STREAM_METRICS: OnceLock<Mutex<Arc<StreamMetrics>>> = OnceLock::new();

pub fn install_metrics(metrics: Arc<StreamMetrics>) {
    let installed = STREAM_METRICS.get_or_init(|| Mutex::new(metrics.clone()));
    *installed.lock().expect("stream metrics lock") = metrics;
}

pub fn prometheus_metrics() -> String {
    STREAM_METRICS.get().map_or_else(String::new, |metrics| {
        metrics
            .lock()
            .expect("stream metrics lock")
            .render_prometheus()
    })
}

pub struct MarketDataStreamRuntime {
    master_url: String,
    token: String,
    consumer_id: String,
    repository: BtcRepository,
    state: Arc<RwLock<RealtimeState>>,
    books: Arc<RwLock<BookRegistry>>,
    market_contracts: Arc<Mutex<BTreeMap<DateTime<Utc>, BtcIntervalMarket>>>,
    sequence: Arc<Mutex<BTreeMap<String, (String, u64)>>>,
    metrics: Arc<StreamMetrics>,
}

impl MarketDataStreamRuntime {
    pub fn new(
        master_url: String,
        token: String,
        consumer_id: String,
        repository: BtcRepository,
        state: Arc<RwLock<RealtimeState>>,
        books: Arc<RwLock<BookRegistry>>,
        metrics: Arc<StreamMetrics>,
    ) -> Result<Self> {
        Ok(Self {
            master_url: master_url.trim_end_matches('/').to_owned(),
            token,
            consumer_id,
            repository,
            state,
            books,
            market_contracts: Arc::new(Mutex::new(BTreeMap::new())),
            sequence: Arc::new(Mutex::new(BTreeMap::new())),
            metrics,
        })
    }

    pub async fn run(
        self,
        mut selectors: watch::Receiver<Vec<SourceSelector>>,
        shutdown: CancellationToken,
    ) {
        let runtime = Arc::new(self);
        let mut delay = Duration::from_millis(250);
        loop {
            if shutdown.is_cancelled() {
                return;
            }
            let desired = selectors.borrow().clone();
            if let Err(error) = validate_selectors(&desired) {
                tracing::error!(%error, "invalid market-data selector union");
                return;
            }
            let route_shutdown = shutdown.child_token();
            let route_future = runtime.run_routes(&desired, route_shutdown.clone());
            tokio::pin!(route_future);
            let outcome = tokio::select! {
                outcome = &mut route_future => Some(outcome),
                changed = selectors.changed() => {
                    route_shutdown.cancel();
                    if changed.is_err() { return; }
                    None
                }
                _ = shutdown.cancelled() => { route_shutdown.cancel(); return; }
            };
            let Some(outcome) = outcome else {
                delay = Duration::from_millis(250);
                continue;
            };
            match outcome {
                Ok(()) if shutdown.is_cancelled() => return,
                Ok(()) => {
                    delay = Duration::from_millis(250);
                    continue;
                }
                Err(error) => {
                    tracing::warn!(%error, "market-data gRPC routes unavailable; preserving process intent")
                }
            }
            runtime.metrics.observe_route_failure("route_resolution");
            runtime.metrics.reconnects.fetch_add(1, Ordering::Relaxed);
            tokio::select! {
                _ = shutdown.cancelled() => return,
                changed = selectors.changed() => { if changed.is_err() { return; } },
                _ = tokio::time::sleep(delay) => {}
            }
            delay = (delay * 2).min(Duration::from_secs(15));
        }
    }

    async fn run_routes(
        self: &Arc<Self>,
        selectors: &[SourceSelector],
        shutdown: CancellationToken,
    ) -> Result<()> {
        let response = self.resolve_routes(selectors).await?;
        let signature = topology_signature(&response.routes);
        self.metrics.desired_connections.store(
            i64::try_from(response.routes.len()).unwrap_or(i64::MAX),
            Ordering::Relaxed,
        );
        let selector_by_key = Arc::new(
            selectors
                .iter()
                .cloned()
                .map(|selector| (selector.key.clone(), selector))
                .collect::<BTreeMap<_, _>>(),
        );
        let mut tasks = tokio::task::JoinSet::new();
        for route in response.routes {
            let runtime = self.clone();
            let route_shutdown = shutdown.clone();
            let route_selectors = selector_by_key.clone();
            tasks.spawn(async move {
                runtime
                    .supervise_route(route, route_selectors, route_shutdown)
                    .await
            });
        }
        let mut topology_tick = tokio::time::interval(Duration::from_secs(30));
        topology_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        topology_tick.tick().await;
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => {
                    tasks.abort_all();
                    while tasks.join_next().await.is_some() {}
                    return Ok(());
                }
                result = tasks.join_next() => {
                    tasks.abort_all();
                    while tasks.join_next().await.is_some() {}
                    match result {
                        Some(Ok(())) => bail!("ingester route supervisor stopped unexpectedly"),
                        Some(Err(error)) => return Err(anyhow::anyhow!(error)),
                        None => bail!("all ingester route supervisors stopped"),
                    }
                }
                _ = topology_tick.tick() => {
                    match self.resolve_routes(selectors).await {
                        Ok(current) if topology_signature(&current.routes) != signature => {
                            tracing::info!("market-data worker ownership changed; refreshing route topology");
                            tasks.abort_all();
                            while tasks.join_next().await.is_some() {}
                            return Ok(());
                        }
                        Ok(_) => {}
                        Err(error) => {
                            self.metrics.observe_route_failure("route_resolution");
                            tracing::warn!(%error, "market-data route refresh failed; preserving established worker streams");
                        }
                    }
                }
            }
        }
    }

    async fn resolve_routes(&self, selectors: &[SourceSelector]) -> Result<RouteResponse> {
        let products = selectors
            .iter()
            .map(|selector| RouteProduct {
                key: selector.key.clone(),
                contract_version: selector.contract_version,
            })
            .collect::<Vec<_>>();
        let response = reqwest::Client::new()
            .post(format!("{}/stream/routes", self.master_url))
            .bearer_auth(&self.token)
            .json(&serde_json::json!({"products": products}))
            .send()
            .await
            .context("resolve ingester stream routes")?
            .error_for_status()
            .context("ingester master rejected stream routes")?
            .json::<RouteResponse>()
            .await?;
        let required = selectors
            .iter()
            .filter(|selector| selector.required)
            .map(|selector| selector.key.as_str())
            .collect::<BTreeSet<_>>();
        let required_unresolved = response
            .unresolved
            .iter()
            .filter(|item| required.contains(item.product.key.as_str()))
            .collect::<Vec<_>>();
        if !required_unresolved.is_empty() {
            let detail = required_unresolved
                .into_iter()
                .map(|item| format!("{}:{}", item.product.key, item.reason))
                .collect::<Vec<_>>()
                .join(",");
            bail!("required market-data products unresolved: {detail}");
        }
        if response.routes.is_empty() {
            bail!("ingester master returned no stream routes");
        }
        Ok(response)
    }

    async fn supervise_route(
        &self,
        route: WorkerRoute,
        selectors: Arc<BTreeMap<String, SourceSelector>>,
        shutdown: CancellationToken,
    ) {
        let mut delay = Duration::from_millis(250);
        loop {
            if shutdown.is_cancelled() {
                return;
            }
            let attempt_started = tokio::time::Instant::now();
            match self
                .consume_route(route.clone(), selectors.clone(), shutdown.clone())
                .await
            {
                Ok(()) if shutdown.is_cancelled() => return,
                Ok(()) => {}
                Err(error) => {
                    let reason = route_failure_reason(&error);
                    self.metrics.observe_route_failure(reason);
                    self.metrics.reconnects.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(
                        worker_id = %route.worker_id,
                        products = route.products.len(),
                        reason,
                        %error,
                        "market-data worker route unavailable; preserving other worker streams"
                    );
                }
            }
            if attempt_started.elapsed() >= Duration::from_secs(30) {
                delay = Duration::from_millis(250);
            }
            tokio::select! {
                _ = shutdown.cancelled() => return,
                _ = tokio::time::sleep(delay) => {}
            }
            delay = (delay * 2).min(Duration::from_secs(15));
        }
    }

    async fn consume_route(
        &self,
        route: WorkerRoute,
        selectors: Arc<BTreeMap<String, SourceSelector>>,
        shutdown: CancellationToken,
    ) -> Result<()> {
        let channel = Channel::from_shared(route.endpoint.clone())?
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(30))
            .tcp_keepalive(Some(Duration::from_secs(15)))
            .connect()
            .await
            .with_context(|| format!("connect worker {} at {}", route.worker_id, route.endpoint))?;
        let mut client = MarketDataStreamClient::new(channel);
        let (commands, command_rx) = mpsc::channel(2);
        let mut request = Request::new(ReceiverStream::new(command_rx));
        request.metadata_mut().insert(
            "authorization",
            MetadataValue::try_from(format!("Bearer {}", self.token))?,
        );
        let mut stream = client.stream(request).await?.into_inner();
        let connected_at = tokio::time::Instant::now();
        commands
            .send(SubscriptionCommand {
                consumer_id: self.consumer_id.clone(),
                revision: 1,
                products: route
                    .products
                    .iter()
                    .map(|product| ProductSelector {
                        key: product.key.clone(),
                        contract_version: product.contract_version,
                        required: selectors
                            .get(&product.key)
                            .is_some_and(|selector| selector.required),
                        maximum_age_ms: selectors
                            .get(&product.key)
                            .map_or(0, SourceSelector::effective_maximum_age_ms),
                        require_sequence_integrity: selectors
                            .get(&product.key)
                            .is_some_and(|selector| selector.require_sequence_integrity),
                    })
                    .collect(),
            })
            .await?;
        {
            let route_products = route
                .products
                .iter()
                .map(|product| product.key.as_str())
                .collect::<BTreeSet<_>>();
            self.sequence
                .lock()
                .expect("stream sequence lock")
                .retain(|product, _| !route_products.contains(product.as_str()));
        }
        self.metrics.connections.fetch_add(1, Ordering::Relaxed);
        let _connection_gauge = ConnectionGauge {
            metrics: &self.metrics,
            products: route
                .products
                .iter()
                .map(|product| product.key.clone())
                .collect(),
        };
        tracing::info!(worker_id=%route.worker_id, source_revision=%route.source_revision, products=route.products.len(), "market-data gRPC route connected");
        loop {
            let next =
                tokio::select! { _ = shutdown.cancelled() => break, next = stream.next() => next };
            let Some(message) = next else {
                bail!("worker {} closed market-data stream", route.worker_id);
            };
            match message?.message {
                Some(Message::SubscriptionAck(ack)) if !ack.rejected.is_empty() => {
                    bail!("worker rejected market-data subscription")
                }
                Some(Message::Health(health)) => {
                    self.metrics
                        .set_product_ready(&health.product_key, health.ready);
                    let selector = selectors
                        .get(&health.product_key)
                        .context("worker reported health outside the selector set")?;
                    if selector.required
                        && !health.ready
                        && connected_at.elapsed() >= Duration::from_secs(30)
                    {
                        bail!(
                            "required market-data product {} is {}",
                            health.product_key,
                            health.detail
                        );
                    }
                }
                Some(Message::SubscriptionAck(_)) | None => {}
                Some(Message::Event(event)) => {
                    let selector = selectors
                        .get(&event.product_key)
                        .context("worker emitted a product outside the selector set")?;
                    let product_key = event.product_key.clone();
                    self.apply_event(event, selector).await?;
                    self.metrics.set_product_ready(&product_key, true);
                }
            }
        }
        Ok(())
    }

    async fn apply_event(&self, event: MarketDataEvent, selector: &SourceSelector) -> Result<()> {
        if event.contract_version != CONTRACT_VERSION || event.integrity != "ok" {
            bail!("invalid market-data envelope");
        }
        if let Some(maximum_age_ms) = selector.maximum_age_ms {
            let age_micros = Utc::now()
                .timestamp_micros()
                .saturating_sub(event.source_timestamp_micros);
            if age_micros > i64::try_from(maximum_age_ms.saturating_mul(1_000))? {
                if selector.required {
                    bail!(
                        "required market-data product {} is stale",
                        event.product_key
                    );
                }
                return Ok(());
            }
        }
        let is_connection_baseline = {
            let mut sequence = self.sequence.lock().expect("stream sequence lock");
            let mut is_connection_baseline = true;
            if let Some((epoch, last)) = sequence.get(&event.product_key) {
                if epoch == &event.publisher_epoch {
                    is_connection_baseline = false;
                    if event.sequence <= *last {
                        self.metrics.duplicates.fetch_add(1, Ordering::Relaxed);
                        return Ok(());
                    }
                    if selector.require_sequence_integrity
                        && event.sequence != last.saturating_add(1)
                    {
                        self.metrics.gaps.fetch_add(1, Ordering::Relaxed);
                        bail!("market-data sequence gap for {}", event.product_key);
                    }
                }
            }
            sequence.insert(
                event.product_key.clone(),
                (event.publisher_epoch.clone(), event.sequence),
            );
            is_connection_baseline
        };
        let result = self.apply_payload(&event, is_connection_baseline).await;
        if result.is_err() {
            self.metrics.decode_errors.fetch_add(1, Ordering::Relaxed);
        }
        result?;
        self.metrics.events.fetch_add(1, Ordering::Relaxed);
        let applied_at = Utc::now().timestamp_micros();
        self.metrics
            .last_event_micros
            .store(applied_at, Ordering::Relaxed);
        let latency = applied_at.saturating_sub(event.published_at_micros) as u64;
        self.metrics
            .apply_latency_micros
            .store(latency, Ordering::Relaxed);
        let mut products = self.metrics.products.lock().expect("stream metrics lock");
        let product = products.entry(event.product_key.clone()).or_default();
        product.events = product.events.saturating_add(1);
        product.apply_latency_micros = latency;
        product.last_event_micros = applied_at;
        Ok(())
    }

    async fn apply_payload(
        &self,
        event: &MarketDataEvent,
        is_connection_baseline: bool,
    ) -> Result<()> {
        match event.product_key.as_str() {
            PRODUCT_MARKETS => {
                let payload: MarketPayload = serde_json::from_slice(&event.payload_json)?;
                let market = payload.into_market();
                self.repository.upsert_interval_market(&market).await?;
                self.books.write().await.try_register_market(&market)?;
                self.market_contracts
                    .lock()
                    .expect("market contract catalog lock")
                    .insert(market.window_start, market);
            }
            PRODUCT_BOOKS => {
                let payload: BookPayload = serde_json::from_slice(&event.payload_json)?;
                // The streamed product is the ingester's canonical aligned
                // snapshot, so runtime freshness follows the envelope's
                // sample and publish clocks. The payload retains the original
                // provider-change clocks for persistence provenance.
                let sampled_at =
                    stream_timestamp(event.source_timestamp_micros, "orderbook source timestamp")?;
                let published_at =
                    stream_timestamp(event.published_at_micros, "orderbook publish timestamp")?;
                let outcome = match payload.outcome.as_str() {
                    "Up" | "up" => BtcOutcome::Up,
                    "Down" | "down" => BtcOutcome::Down,
                    _ => bail!("invalid book outcome"),
                };
                let bids = parse_levels(payload.bids)?;
                let asks = parse_levels(payload.asks)?;
                let epoch = Uuid::parse_str(&event.publisher_epoch)?;
                let mut books = self.books.write().await;
                books.apply_canonical_snapshot_for_market_identity(
                    epoch,
                    &payload.market.market_id,
                    &payload.market.condition_id,
                    &payload.market.up_token_id,
                    &payload.market.down_token_id,
                    &payload.token_id,
                    outcome,
                    payload.tick_size,
                    sampled_at,
                    published_at,
                    u64::try_from(payload.ingest_sequence)?,
                    payload.source_hash,
                    bids,
                    asks,
                )?;
                self.state.write().await.update_books(&books);
            }
            PRODUCT_CHAINLINK => {
                let payload: ReferencePayload = serde_json::from_slice(&event.payload_json)?;
                let tick = ReferencePriceTick {
                    tick_id: Uuid::new_v4(),
                    dedup_key: event.payload_sha256.clone(),
                    source: ReferencePriceSource::RtdsChainlink,
                    symbol: "BTCUSD".to_owned(),
                    price: payload.price,
                    source_timestamp: payload.source_timestamp,
                    envelope_timestamp: payload.provider_available_at,
                    received_at: payload.received_at,
                    connection_id: Uuid::parse_str(&event.publisher_epoch)?,
                    ingest_sequence: event.sequence,
                    source_event_id: Some(event.source_event_id.clone()),
                    raw_payload: serde_json::from_slice(&event.payload_json)?,
                };
                let mut state = self.state.write().await;
                state.directional_external.observe_rtds_chainlink(&tick)?;
                state.update_reference_price(tick);
            }
            PRODUCT_TWAP => {
                let payload: TwapPayload = serde_json::from_slice(&event.payload_json)?;
                if payload.window_seconds == 60 {
                    self.state
                        .write()
                        .await
                        .chainlink_twap_60
                        .observe(ChainlinkTwap60Point {
                            price: payload.price,
                            source_timestamp: payload.source_timestamp,
                            available_at: payload.published_at,
                        });
                }
            }
            PRODUCT_BINANCE_1S => {
                let raw_payload: serde_json::Value = serde_json::from_slice(&event.payload_json)?;
                let payload: KlinePayload = serde_json::from_value(raw_payload.clone())?;
                let reference = payload.reference_tick(event, raw_payload)?;
                let kline = payload.into_kline();
                let mut state = self.state.write().await;
                // The one-second close replaces the retired aggregate-trade
                // socket as the canonical direct Binance reference while the
                // existing model adapter continues to consume its unchanged
                // ReferencePriceTick contract.
                state.update_reference_price(reference);
                if let Err(error) = state
                    .binance_one_second_window
                    .observe_completed(kline.clone())
                {
                    if !is_connection_baseline {
                        return Err(error);
                    }
                    state.binance_one_second_window.clear();
                    state.binance_one_second_window.observe_completed(kline)?;
                }
            }
            PRODUCT_POLYGON_ORACLE => {
                let payload: PolygonOraclePayload = serde_json::from_slice(&event.payload_json)?;
                self.state.write().await.directional_external.merge_oracle(
                    vec![PolygonOraclePoint {
                        phase_id: u16::try_from(payload.phase_id)?,
                        round_id: u64::try_from(payload.aggregator_round_id)?,
                        source_timestamp: payload.source_timestamp,
                        block_timestamp: payload.block_timestamp,
                        available_at: payload.provider_available_at,
                        price: payload.price,
                    }],
                    payload.received_at,
                );
            }
            PRODUCT_BINANCE_OPEN_INTEREST => {
                let payload: OpenInterestPayload = serde_json::from_slice(&event.payload_json)?;
                self.state
                    .write()
                    .await
                    .directional_external
                    .merge_open_interest(
                        vec![BinanceOpenInterestPoint {
                            source_timestamp: payload.source_timestamp,
                            available_at: payload.received_at,
                            sum_open_interest: payload.sum_open_interest,
                            sum_open_interest_value: payload.sum_open_interest_value,
                        }],
                        payload.received_at,
                    );
            }
            PRODUCT_RESOLUTIONS => {
                let payload: ResolutionPayload = serde_json::from_slice(&event.payload_json)?;
                let source_timestamp = payload.source_timestamp.unwrap_or(payload.received_at);
                self.repository
                    .persist_official_market_resolution(
                        &payload.market_id,
                        &payload.winning_token_id,
                        &payload.winning_outcome,
                        source_timestamp,
                        &payload.resolution_source,
                        payload.received_at,
                        &serde_json::from_slice(&event.payload_json)?,
                    )
                    .await?;
                self.state
                    .write()
                    .await
                    .apply_market_resolution(&payload.market_id, &payload.winning_token_id);
            }
            other => bail!("unhandled market-data product {other}"),
        }
        self.reconcile_market_window(Utc::now()).await;
        Ok(())
    }

    async fn reconcile_market_window(&self, now: DateTime<Utc>) {
        let selection = {
            let mut contracts = self
                .market_contracts
                .lock()
                .expect("market contract catalog lock");
            select_market_window(&mut contracts, now)
        };
        let mut state = self.state.write().await;
        apply_market_window_selection(&mut state, selection);
    }
}

fn stream_timestamp(micros: i64, field: &'static str) -> Result<DateTime<Utc>> {
    DateTime::from_timestamp_micros(micros).with_context(|| format!("invalid {field}"))
}

#[derive(Debug, Clone, PartialEq)]
struct MarketWindowSelection {
    tradable: Option<BtcIntervalMarket>,
    display: Option<BtcIntervalMarket>,
}

fn select_market_window(
    contracts: &mut BTreeMap<DateTime<Utc>, BtcIntervalMarket>,
    now: DateTime<Utc>,
) -> MarketWindowSelection {
    let retention_start = now - chrono::Duration::minutes(10);
    contracts.retain(|_, market| market.window_end >= retention_start);
    MarketWindowSelection {
        tradable: contracts
            .values()
            .find(|market| market.is_trade_window(now))
            .cloned(),
        display: contracts
            .values()
            .find(|market| market.is_interval_window(now))
            .cloned(),
    }
}

fn apply_market_window_selection(state: &mut RealtimeState, selection: MarketWindowSelection) {
    state.display_market = selection.display;
    state.set_current_market(selection.tradable);
}

fn parse_levels(levels: Vec<[String; 2]>) -> Result<Vec<(Decimal, Decimal)>> {
    levels
        .into_iter()
        .map(|[price, size]| Ok((price.parse()?, size.parse()?)))
        .collect()
}

#[derive(Debug, Clone, Deserialize)]
struct MarketPayload {
    event_id: String,
    event_slug: String,
    series_slug: String,
    market_id: String,
    condition_id: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    up_token_id: String,
    down_token_id: String,
    tick_size: Decimal,
    minimum_order_size: Option<Decimal>,
    resolution_source: String,
    active: bool,
    closed: bool,
    accepting_orders: bool,
    fees_enabled: bool,
    fee_schedule: serde_json::Value,
    source_payload: serde_json::Value,
}
impl MarketPayload {
    fn into_market(self) -> BtcIntervalMarket {
        BtcIntervalMarket {
            event_id: self.event_id,
            event_slug: self.event_slug,
            series_slug: self.series_slug,
            market_id: self.market_id,
            condition_id: self.condition_id,
            window_start: self.window_start,
            window_end: self.window_end,
            up_token_id: self.up_token_id,
            down_token_id: self.down_token_id,
            tick_size: self.tick_size,
            minimum_order_size: self.minimum_order_size,
            resolution_source: self.resolution_source,
            active: self.active,
            closed: self.closed,
            accepting_orders: self.accepting_orders,
            fees_enabled: self.fees_enabled,
            fee_schedule: self.fee_schedule,
            raw_payload: self.source_payload,
        }
    }
}

#[derive(Debug, Deserialize)]
struct BookPayload {
    market: BookMarketPayload,
    token_id: String,
    outcome: String,
    tick_size: Decimal,
    #[serde(rename = "source_timestamp")]
    _provider_source_timestamp: DateTime<Utc>,
    #[serde(rename = "received_at")]
    _ingester_received_at: DateTime<Utc>,
    source_hash: Option<String>,
    ingest_sequence: i64,
    bids: Vec<[String; 2]>,
    asks: Vec<[String; 2]>,
}

#[derive(Debug, Deserialize)]
struct BookMarketPayload {
    market_id: String,
    condition_id: String,
    up_token_id: String,
    down_token_id: String,
}
#[derive(Debug, Deserialize)]
struct ReferencePayload {
    source_timestamp: DateTime<Utc>,
    price: Decimal,
    provider_available_at: Option<DateTime<Utc>>,
    received_at: DateTime<Utc>,
}
#[derive(Debug, Deserialize)]
struct TwapPayload {
    source_timestamp: DateTime<Utc>,
    published_at: DateTime<Utc>,
    window_seconds: i16,
    price: Decimal,
}
#[derive(Debug, Deserialize)]
struct ResolutionPayload {
    market_id: String,
    winning_token_id: String,
    winning_outcome: String,
    resolution_source: String,
    source_timestamp: Option<DateTime<Utc>>,
    received_at: DateTime<Utc>,
}
#[derive(Debug, Clone, Deserialize)]
struct KlinePayload {
    open_timestamp: DateTime<Utc>,
    close_timestamp: DateTime<Utc>,
    provider_available_at: Option<DateTime<Utc>>,
    received_at: DateTime<Utc>,
    open_price: Decimal,
    high_price: Decimal,
    low_price: Decimal,
    close_price: Decimal,
    base_volume: Decimal,
    quote_volume: Decimal,
    trade_count: i64,
    taker_buy_base_volume: Decimal,
    taker_buy_quote_volume: Decimal,
}
impl KlinePayload {
    fn reference_tick(
        &self,
        event: &MarketDataEvent,
        raw_payload: serde_json::Value,
    ) -> Result<ReferencePriceTick> {
        Ok(ReferencePriceTick {
            tick_id: Uuid::new_v4(),
            dedup_key: event.payload_sha256.clone(),
            source: ReferencePriceSource::DirectBinance,
            symbol: "BTCUSDT".to_owned(),
            price: self.close_price,
            source_timestamp: self.close_timestamp,
            envelope_timestamp: self.provider_available_at,
            received_at: self.received_at,
            connection_id: Uuid::parse_str(&event.publisher_epoch)?,
            ingest_sequence: event.sequence,
            source_event_id: Some(event.source_event_id.clone()),
            raw_payload,
        })
    }

    fn into_kline(self) -> BinanceOneSecondKline {
        BinanceOneSecondKline {
            open_timestamp: self.open_timestamp,
            close_timestamp: self.close_timestamp + chrono::Duration::milliseconds(1),
            open_price: self.open_price,
            high_price: self.high_price,
            low_price: self.low_price,
            close_price: self.close_price,
            base_volume: self.base_volume,
            quote_volume: self.quote_volume,
            trade_count: u64::try_from(self.trade_count).unwrap_or_default(),
            taker_buy_base_volume: self.taker_buy_base_volume,
            taker_buy_quote_volume: self.taker_buy_quote_volume,
            first_aggregate_trade_id: 0,
            last_aggregate_trade_id: 0,
            first_source_timestamp: self.open_timestamp,
            last_source_timestamp: self.provider_available_at.unwrap_or(self.close_timestamp),
            max_received_at: self.received_at,
            source_complete: true,
            synthetic: false,
        }
    }
}

#[derive(Debug, Deserialize)]
struct PolygonOraclePayload {
    phase_id: i32,
    aggregator_round_id: i64,
    source_timestamp: DateTime<Utc>,
    block_timestamp: DateTime<Utc>,
    provider_available_at: DateTime<Utc>,
    received_at: DateTime<Utc>,
    price: Decimal,
}

#[derive(Debug, Deserialize)]
struct OpenInterestPayload {
    source_timestamp: DateTime<Utc>,
    received_at: DateTime<Utc>,
    sum_open_interest: Decimal,
    sum_open_interest_value: Decimal,
}

impl StreamMetrics {
    fn set_product_ready(&self, product: &str, ready: bool) {
        self.product_ready
            .lock()
            .expect("stream readiness lock")
            .insert(product.to_owned(), ready);
    }

    fn observe_route_failure(&self, reason: &str) {
        let mut failures = self
            .route_failures
            .lock()
            .expect("stream route failure lock");
        *failures.entry(reason.to_owned()).or_insert(0) += 1;
    }

    pub fn render_prometheus(&self) -> String {
        let mut output = format!(
            "# HELP polymarket_market_data_grpc_connections Active direct worker streams.\n# TYPE polymarket_market_data_grpc_connections gauge\npolymarket_market_data_grpc_connections {}\n\
# HELP polymarket_market_data_grpc_desired_connections Worker streams required by the current selector union.\n# TYPE polymarket_market_data_grpc_desired_connections gauge\npolymarket_market_data_grpc_desired_connections {}\n\
# HELP polymarket_market_data_grpc_reconnects_total Direct worker stream reconnects.\n# TYPE polymarket_market_data_grpc_reconnects_total counter\npolymarket_market_data_grpc_reconnects_total {}\n\
# HELP polymarket_market_data_events_applied_total Canonical events applied to the shared trading runtime.\n# TYPE polymarket_market_data_events_applied_total counter\npolymarket_market_data_events_applied_total {}\n\
# HELP polymarket_market_data_duplicates_total Replayed canonical events ignored.\n# TYPE polymarket_market_data_duplicates_total counter\npolymarket_market_data_duplicates_total {}\n\
# HELP polymarket_market_data_sequence_gaps_total Stream sequence gaps detected.\n# TYPE polymarket_market_data_sequence_gaps_total counter\npolymarket_market_data_sequence_gaps_total {}\n\
# HELP polymarket_market_data_decode_errors_total Contract payload failures.\n# TYPE polymarket_market_data_decode_errors_total counter\npolymarket_market_data_decode_errors_total {}\n\
# HELP polymarket_market_data_publish_apply_latency_seconds Latest worker publish to bot apply latency.\n# TYPE polymarket_market_data_publish_apply_latency_seconds gauge\npolymarket_market_data_publish_apply_latency_seconds {}\n\
# HELP polymarket_market_data_last_event_timestamp_seconds Latest applied canonical event time.\n# TYPE polymarket_market_data_last_event_timestamp_seconds gauge\npolymarket_market_data_last_event_timestamp_seconds {}\n",
            self.connections.load(Ordering::Relaxed), self.desired_connections.load(Ordering::Relaxed), self.reconnects.load(Ordering::Relaxed),
            self.events.load(Ordering::Relaxed), self.duplicates.load(Ordering::Relaxed), self.gaps.load(Ordering::Relaxed),
            self.decode_errors.load(Ordering::Relaxed), self.apply_latency_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.last_event_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0)
        ;
        output.push_str("# HELP polymarket_market_data_product_events_applied_total Canonical events applied by selected product.\n# TYPE polymarket_market_data_product_events_applied_total counter\n");
        let products = self.products.lock().expect("stream metrics lock");
        for (key, metrics) in products.iter() {
            output.push_str(&format!(
                "polymarket_market_data_product_events_applied_total{{product=\"{key}\"}} {}\n",
                metrics.events
            ));
        }
        output.push_str("# HELP polymarket_market_data_product_publish_apply_latency_seconds Latest worker publish to bot apply latency by product.\n# TYPE polymarket_market_data_product_publish_apply_latency_seconds gauge\n");
        for (key, metrics) in products.iter() {
            output.push_str(&format!(
                "polymarket_market_data_product_publish_apply_latency_seconds{{product=\"{key}\"}} {}\n",
                metrics.apply_latency_micros as f64 / 1_000_000.0
            ));
        }
        output.push_str("# HELP polymarket_market_data_product_last_event_timestamp_seconds Latest applied canonical event time by product.\n# TYPE polymarket_market_data_product_last_event_timestamp_seconds gauge\n");
        for (key, metrics) in products.iter() {
            output.push_str(&format!(
                "polymarket_market_data_product_last_event_timestamp_seconds{{product=\"{key}\"}} {}\n",
                metrics.last_event_micros as f64 / 1_000_000.0
            ));
        }
        drop(products);
        output.push_str("# HELP polymarket_market_data_product_ready Required product readiness reported by its owning worker.\n# TYPE polymarket_market_data_product_ready gauge\n");
        for (key, ready) in self
            .product_ready
            .lock()
            .expect("stream readiness lock")
            .iter()
        {
            output.push_str(&format!(
                "polymarket_market_data_product_ready{{product=\"{key}\"}} {}\n",
                u8::from(*ready)
            ));
        }
        output.push_str("# HELP polymarket_market_data_route_failures_total Worker route failures by bounded reason.\n# TYPE polymarket_market_data_route_failures_total counter\n");
        for (reason, value) in self
            .route_failures
            .lock()
            .expect("stream route failure lock")
            .iter()
        {
            output.push_str(&format!(
                "polymarket_market_data_route_failures_total{{reason=\"{reason}\"}} {value}\n"
            ));
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use crate::grafana_live::{CountdownSnapshot, CountdownStatus, MarketPathPublicationState};

    use super::*;

    fn market(
        market_id: &str,
        window_start: DateTime<Utc>,
        active: bool,
        closed: bool,
        accepting_orders: bool,
    ) -> BtcIntervalMarket {
        MarketPayload {
            event_id: format!("event-{market_id}"),
            event_slug: format!("btc-updown-5m-{}", window_start.timestamp()),
            series_slug: "btc-up-or-down-5m".to_owned(),
            market_id: market_id.to_owned(),
            condition_id: format!("condition-{market_id}"),
            window_start,
            window_end: window_start + chrono::Duration::minutes(5),
            up_token_id: format!("up-{market_id}"),
            down_token_id: format!("down-{market_id}"),
            tick_size: Decimal::new(1, 2),
            minimum_order_size: None,
            resolution_source: "Chainlink BTC/USD".to_owned(),
            active,
            closed,
            accepting_orders,
            fees_enabled: false,
            fee_schedule: serde_json::json!({}),
            source_payload: serde_json::json!({}),
        }
        .into_market()
    }

    #[test]
    fn future_contract_preloads_without_replacing_the_current_runtime_window() {
        let current_start = Utc.with_ymd_and_hms(2026, 9, 3, 18, 0, 0).unwrap();
        let now = current_start + chrono::Duration::minutes(2);
        let current = market("current", current_start, true, false, true);
        let future = market(
            "future",
            current_start + chrono::Duration::minutes(5),
            true,
            false,
            true,
        );
        let mut contracts = BTreeMap::from([
            (current.window_start, current.clone()),
            (future.window_start, future.clone()),
        ]);

        let selection = select_market_window(&mut contracts, now);
        let mut state = RealtimeState::default();
        apply_market_window_selection(&mut state, selection);

        assert_eq!(contracts.get(&future.window_start), Some(&future));
        assert_eq!(state.current_market.as_ref(), Some(&current));
        assert_eq!(state.display_market.as_ref(), Some(&current));

        let rollover = select_market_window(
            &mut contracts,
            future.window_start + chrono::Duration::seconds(1),
        );
        assert_eq!(rollover.tradable.as_ref(), Some(&future));
        assert_eq!(rollover.display.as_ref(), Some(&future));
    }

    #[test]
    fn current_display_window_does_not_become_tradable_when_exchange_flags_reject_it() {
        let window_start = Utc.with_ymd_and_hms(2026, 9, 3, 18, 0, 0).unwrap();
        let current = market("closed", window_start, false, true, false);
        let mut contracts = BTreeMap::from([(current.window_start, current.clone())]);

        let selection =
            select_market_window(&mut contracts, window_start + chrono::Duration::minutes(2));

        assert_eq!(selection.display.as_ref(), Some(&current));
        assert!(selection.tradable.is_none());
    }

    #[test]
    fn selected_display_market_restores_live_countdown_and_market_path_snapshots() {
        let window_start = Utc.with_ymd_and_hms(2026, 9, 3, 18, 0, 0).unwrap();
        let now = window_start + chrono::Duration::minutes(2);
        let current = market("current", window_start, true, false, true);
        let mut contracts = BTreeMap::from([(current.window_start, current.clone())]);
        let selection = select_market_window(&mut contracts, now);
        let mut state = RealtimeState::default();
        apply_market_window_selection(&mut state, selection);

        let display = state.display_market.clone().expect("display market");
        let countdown = CountdownSnapshot::resolve(now, 1, vec![display.clone()]);
        assert_eq!(countdown.status, CountdownStatus::Active);
        assert_eq!(countdown.seconds_remaining, 180);

        let mut path = MarketPathPublicationState::default();
        let snapshot = path.observe(
            now,
            Some((
                display,
                vec![ChainlinkTwap60Point {
                    price: Decimal::new(81_000, 0),
                    source_timestamp: window_start,
                    available_at: window_start,
                }],
            )),
        );
        assert!(snapshot.is_some());
    }

    #[test]
    fn source_selectors_accept_the_concise_and_configured_contracts() {
        let concise: SourceSelector =
            serde_json::from_str(&format!("\"{PRODUCT_BOOKS}\"")).expect("concise selector");
        assert_eq!(concise.key, PRODUCT_BOOKS);
        assert!(concise.required);
        assert!(concise.require_sequence_integrity);
        assert_eq!(concise.effective_maximum_age_ms(), 10_000);

        let configured: SourceSelector = serde_json::from_value(serde_json::json!({
            "key": PRODUCT_TWAP,
            "contract_version": 1,
            "required": false,
            "maximum_age_ms": 45_000,
            "require_sequence_integrity": false
        }))
        .expect("configured selector");
        configured.validate().expect("valid selector");
        assert!(!configured.required);
        assert_eq!(configured.effective_maximum_age_ms(), 45_000);
    }

    #[test]
    fn selector_contract_rejects_unknown_products_and_invalid_freshness() {
        let unknown: SourceSelector =
            serde_json::from_str("\"unregistered_feed\"").expect("selector shape");
        assert!(unknown.validate().is_err());
        let invalid: SourceSelector = serde_json::from_value(serde_json::json!({
            "key": PRODUCT_BOOKS,
            "maximum_age_ms": 0
        }))
        .expect("selector shape");
        assert!(invalid.validate().is_err());

        let too_old: SourceSelector = serde_json::from_value(serde_json::json!({
            "key": PRODUCT_BINANCE_OPEN_INTEREST,
            "maximum_age_ms": 600_001
        }))
        .expect("selector shape");
        assert!(too_old.validate().is_err());

        let open_interest: SourceSelector =
            serde_json::from_str(&format!("\"{PRODUCT_BINANCE_OPEN_INTEREST}\""))
                .expect("open-interest selector");
        assert_eq!(open_interest.effective_maximum_age_ms(), 360_000);
    }

    #[test]
    fn orderbook_payload_accepts_the_ingester_compact_market_contract() {
        let payload: BookPayload = serde_json::from_value(serde_json::json!({
            "market": {
                "event_slug": "btc-updown-5m-1788370200",
                "market_id": "market-1",
                "condition_id": "condition-1",
                "window_start": "2026-09-03T17:30:00Z",
                "window_end": "2026-09-03T17:35:00Z",
                "up_token_id": "up-1",
                "down_token_id": "down-1",
                "tick_size": "0.01",
                "received_at": "2026-09-03T17:30:01Z"
            },
            "token_id": "up-1",
            "outcome": "Up",
            "tick_size": "0.01",
            "source_timestamp": "2026-09-03T17:30:01Z",
            "received_at": "2026-09-03T17:30:01.050Z",
            "source_hash": "sha256:test",
            "ingest_sequence": 1,
            "bids": [["0.49", "100"]],
            "asks": [["0.51", "100"]]
        }))
        .expect("ingester orderbook payload");

        assert_eq!(payload.market.market_id, "market-1");
        assert_eq!(payload.market.condition_id, "condition-1");
        assert_eq!(payload.tick_size, Decimal::new(1, 2));
    }

    #[test]
    fn canonical_stream_snapshot_accepts_validated_tick_transition_for_exact_market_identity() {
        let window_start = Utc.with_ymd_and_hms(2026, 9, 3, 19, 45, 0).unwrap();
        let market = market("rollover", window_start, true, false, true);
        let connection_id = Uuid::new_v4();
        let mut books = BookRegistry::new(connection_id);
        books
            .try_register_market(&market)
            .expect("Gamma market registration");
        let sampled_at = window_start + chrono::Duration::minutes(4);

        let applied = books
            .apply_canonical_snapshot_for_market_identity(
                connection_id,
                &market.market_id,
                &market.condition_id,
                &market.up_token_id,
                &market.down_token_id,
                &market.up_token_id,
                BtcOutcome::Up,
                Decimal::new(1, 3),
                sampled_at,
                sampled_at + chrono::Duration::milliseconds(2),
                42,
                Some("sha256:book".to_owned()),
                vec![(Decimal::new(499, 3), Decimal::new(10, 0))],
                vec![(Decimal::new(501, 3), Decimal::new(10, 0))],
            )
            .expect("validated canonical tick transition");

        assert!(applied.applied);
        assert_eq!(
            books
                .checkpoint(&market.up_token_id)
                .expect("canonical checkpoint")
                .tick_size,
            Decimal::new(1, 3)
        );

        let stale = books
            .apply_canonical_snapshot_for_market_identity(
                connection_id,
                &market.market_id,
                &market.condition_id,
                &market.up_token_id,
                &market.down_token_id,
                &market.up_token_id,
                BtcOutcome::Up,
                Decimal::new(1, 2),
                sampled_at - chrono::Duration::seconds(1),
                sampled_at + chrono::Duration::milliseconds(3),
                41,
                None,
                vec![],
                vec![],
            )
            .expect("stale canonical snapshot is ignored");
        assert_eq!(
            stale.integrity_status,
            crate::btc::FeedIntegrityStatus::OutOfOrder
        );
        assert_eq!(
            books
                .checkpoint(&market.up_token_id)
                .expect("newer tick remains authoritative")
                .tick_size,
            Decimal::new(1, 3)
        );

        let conflicting = books.apply_canonical_snapshot_for_market_identity(
            connection_id,
            "different-market",
            "different-condition",
            &market.up_token_id,
            "different-down-token",
            &market.up_token_id,
            BtcOutcome::Up,
            Decimal::new(1, 3),
            sampled_at,
            sampled_at + chrono::Duration::milliseconds(3),
            43,
            None,
            vec![],
            vec![],
        );
        assert!(conflicting.is_err());
        assert_eq!(
            books
                .checkpoint(&market.up_token_id)
                .expect("original identity remains")
                .market_id,
            market.market_id
        );
    }

    #[test]
    fn canonical_stream_timestamp_preserves_subsecond_sample_time() {
        let sampled_at = Utc.with_ymd_and_hms(2026, 9, 3, 18, 30, 1).unwrap()
            + chrono::Duration::microseconds(234_567);

        assert_eq!(
            stream_timestamp(sampled_at.timestamp_micros(), "sample").expect("timestamp"),
            sampled_at
        );
        assert!(stream_timestamp(i64::MAX, "sample").is_err());
    }

    #[test]
    fn one_second_kline_preserves_the_direct_binance_reference_contract() {
        let raw = serde_json::json!({
            "open_timestamp": "2026-09-03T17:30:00Z",
            "close_timestamp": "2026-09-03T17:30:00.999Z",
            "provider_available_at": "2026-09-03T17:30:01.001Z",
            "received_at": "2026-09-03T17:30:01.002Z",
            "open_price": "81000.00",
            "high_price": "81002.00",
            "low_price": "80999.00",
            "close_price": "81001.25",
            "base_volume": "1.5",
            "quote_volume": "121501.875",
            "trade_count": 12,
            "taker_buy_base_volume": "0.8",
            "taker_buy_quote_volume": "64801.0"
        });
        let payload: KlinePayload = serde_json::from_value(raw.clone()).expect("kline payload");
        let event = MarketDataEvent {
            product_key: PRODUCT_BINANCE_1S.to_owned(),
            contract_version: CONTRACT_VERSION,
            worker_id: "worker-1".to_owned(),
            publisher_epoch: "6675831f-15ea-45ed-a21f-397cd76bf1ec".to_owned(),
            sequence: 42,
            source_event_id: "1788456600000000".to_owned(),
            source_timestamp_micros: 0,
            provider_available_at_micros: 0,
            received_at_micros: 0,
            published_at_micros: 0,
            payload_sha256: "sha256:kline".to_owned(),
            integrity: "ok".to_owned(),
            persistence_healthy: true,
            payload_json: Vec::new(),
        };

        let reference = payload
            .reference_tick(&event, raw.clone())
            .expect("direct Binance reference");
        assert_eq!(reference.source, ReferencePriceSource::DirectBinance);
        assert_eq!(reference.symbol, "BTCUSDT");
        assert_eq!(reference.price, Decimal::new(8_100_125, 2));
        assert_eq!(
            reference.source_event_id.as_deref(),
            Some("1788456600000000")
        );
        assert_eq!(reference.raw_payload, raw);
    }

    #[test]
    fn route_topology_signature_is_stable_across_response_ordering() {
        let first = vec![
            WorkerRoute {
                worker_id: "worker-b".to_owned(),
                endpoint: "http://worker-b:8100".to_owned(),
                source_revision: "revision".to_owned(),
                products: vec![RouteProduct {
                    key: PRODUCT_BINANCE_1S.to_owned(),
                    contract_version: CONTRACT_VERSION,
                }],
            },
            WorkerRoute {
                worker_id: "worker-a".to_owned(),
                endpoint: "http://worker-a:8100".to_owned(),
                source_revision: "revision".to_owned(),
                products: vec![RouteProduct {
                    key: PRODUCT_BOOKS.to_owned(),
                    contract_version: CONTRACT_VERSION,
                }],
            },
        ];
        let mut reversed = first.clone();
        reversed.reverse();

        assert_eq!(topology_signature(&first), topology_signature(&reversed));
    }

    #[test]
    fn route_failure_metrics_use_bounded_operational_reasons() {
        assert_eq!(
            route_failure_reason(&anyhow::anyhow!(
                "required market-data product {PRODUCT_BINANCE_1S} is stale"
            )),
            "required_product_stale"
        );
        assert_eq!(
            route_failure_reason(&anyhow::anyhow!(
                "worker worker-1 closed market-data stream"
            )),
            "worker_stream_closed"
        );

        let metrics = StreamMetrics::default();
        metrics.desired_connections.store(4, Ordering::Relaxed);
        metrics.set_product_ready(PRODUCT_BINANCE_1S, false);
        metrics.observe_route_failure("required_product_stale");
        let rendered = metrics.render_prometheus();
        assert!(rendered.contains("polymarket_market_data_grpc_desired_connections 4"));
        assert!(rendered.contains(&format!(
            "polymarket_market_data_product_ready{{product=\"{PRODUCT_BINANCE_1S}\"}} 0"
        )));
        assert!(rendered.contains(
            "polymarket_market_data_route_failures_total{reason=\"required_product_stale\"} 1"
        ));
    }

    #[test]
    fn dropping_one_route_preserves_other_route_readiness() {
        let metrics = StreamMetrics::default();
        metrics.connections.store(2, Ordering::Relaxed);
        metrics.set_product_ready(PRODUCT_BINANCE_1S, true);
        metrics.set_product_ready(PRODUCT_BOOKS, true);

        {
            let _failed_route = ConnectionGauge {
                metrics: &metrics,
                products: vec![PRODUCT_BINANCE_1S.to_owned()],
            };
        }

        assert_eq!(metrics.connections.load(Ordering::Relaxed), 1);
        let readiness = metrics.product_ready.lock().expect("readiness lock");
        assert_eq!(readiness.get(PRODUCT_BINANCE_1S), Some(&false));
        assert_eq!(readiness.get(PRODUCT_BOOKS), Some(&true));
    }
}
