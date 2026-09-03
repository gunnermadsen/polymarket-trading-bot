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

#[derive(Debug, Deserialize)]
struct RouteResponse {
    routes: Vec<WorkerRoute>,
    unresolved: Vec<RouteRejection>,
}
#[derive(Debug, Deserialize)]
struct WorkerRoute {
    worker_id: String,
    endpoint: String,
    source_revision: String,
    products: Vec<RouteProduct>,
}
#[derive(Debug, Deserialize, Serialize)]
struct RouteProduct {
    key: String,
    contract_version: u32,
}
#[derive(Debug, Deserialize)]
struct RouteRejection {
    product: RouteProduct,
    reason: String,
}

#[derive(Default)]
pub struct StreamMetrics {
    connections: AtomicI64,
    reconnects: AtomicU64,
    events: AtomicU64,
    duplicates: AtomicU64,
    gaps: AtomicU64,
    decode_errors: AtomicU64,
    apply_latency_micros: AtomicU64,
    last_event_micros: AtomicI64,
    products: Mutex<BTreeMap<String, ProductStreamMetrics>>,
}

#[derive(Default)]
struct ProductStreamMetrics {
    events: u64,
    apply_latency_micros: u64,
    last_event_micros: i64,
}

struct ConnectionGauge<'a>(&'a AtomicI64);

impl Drop for ConnectionGauge<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
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
                Ok(()) => {}
                Err(error) => {
                    tracing::warn!(%error, "market-data gRPC routes unavailable; preserving process intent")
                }
            }
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
        let mut tasks = tokio::task::JoinSet::new();
        let selector_by_key = Arc::new(
            selectors
                .iter()
                .cloned()
                .map(|selector| (selector.key.clone(), selector))
                .collect::<BTreeMap<_, _>>(),
        );
        for route in response.routes {
            let runtime = self.clone();
            let route_shutdown = shutdown.clone();
            let route_selectors = selector_by_key.clone();
            tasks.spawn(async move {
                runtime
                    .consume_route(route, route_selectors, route_shutdown)
                    .await
            });
        }
        tokio::select! {
            _ = shutdown.cancelled() => { tasks.abort_all(); while tasks.join_next().await.is_some() {} Ok(()) },
            result = tasks.join_next() => {
                tasks.abort_all(); while tasks.join_next().await.is_some() {}
                match result { Some(Ok(result)) => result, Some(Err(error)) => Err(anyhow::anyhow!(error)), None => bail!("all ingester stream routes stopped") }
            }
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
        let _connection_gauge = ConnectionGauge(&self.metrics.connections);
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
                    self.apply_event(event, selector).await?;
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
                let now = Utc::now();
                let mut state = self.state.write().await;
                let replace = market.is_interval_window(now)
                    || state
                        .current_market
                        .as_ref()
                        .is_none_or(|current| market.window_start > current.window_start);
                if replace {
                    state.set_market(market.clone());
                }
                drop(state);
                self.books.write().await.try_register_market(&market)?;
            }
            PRODUCT_BOOKS => {
                let payload: BookPayload = serde_json::from_slice(&event.payload_json)?;
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
                    payload.source_timestamp,
                    payload.received_at,
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
                self.state.write().await.update_reference_price(tick);
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
        Ok(())
    }
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
    source_timestamp: DateTime<Utc>,
    received_at: DateTime<Utc>,
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
    pub fn render_prometheus(&self) -> String {
        let mut output = format!(
            "# HELP polymarket_market_data_grpc_connections Active direct worker streams.\n# TYPE polymarket_market_data_grpc_connections gauge\npolymarket_market_data_grpc_connections {}\n\
# HELP polymarket_market_data_grpc_reconnects_total Direct worker stream reconnects.\n# TYPE polymarket_market_data_grpc_reconnects_total counter\npolymarket_market_data_grpc_reconnects_total {}\n\
# HELP polymarket_market_data_events_applied_total Canonical events applied to the shared trading runtime.\n# TYPE polymarket_market_data_events_applied_total counter\npolymarket_market_data_events_applied_total {}\n\
# HELP polymarket_market_data_duplicates_total Replayed canonical events ignored.\n# TYPE polymarket_market_data_duplicates_total counter\npolymarket_market_data_duplicates_total {}\n\
# HELP polymarket_market_data_sequence_gaps_total Stream sequence gaps detected.\n# TYPE polymarket_market_data_sequence_gaps_total counter\npolymarket_market_data_sequence_gaps_total {}\n\
# HELP polymarket_market_data_decode_errors_total Contract payload failures.\n# TYPE polymarket_market_data_decode_errors_total counter\npolymarket_market_data_decode_errors_total {}\n\
# HELP polymarket_market_data_publish_apply_latency_seconds Latest worker publish to bot apply latency.\n# TYPE polymarket_market_data_publish_apply_latency_seconds gauge\npolymarket_market_data_publish_apply_latency_seconds {}\n\
# HELP polymarket_market_data_last_event_timestamp_seconds Latest applied canonical event time.\n# TYPE polymarket_market_data_last_event_timestamp_seconds gauge\npolymarket_market_data_last_event_timestamp_seconds {}\n",
            self.connections.load(Ordering::Relaxed), self.reconnects.load(Ordering::Relaxed),
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
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
