use std::{
    collections::{BTreeMap, BTreeSet},
    net::SocketAddr,
    pin::Pin,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures_util::{Stream, StreamExt};
use serde::Serialize;
use subtle::ConstantTimeEq;
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};
use tracing::{info, warn};
use uuid::Uuid;

pub mod proto {
    tonic::include_proto!("capitonic.marketdata.v1");
}

use proto::{
    market_data_message::Message, market_data_stream_server::MarketDataStream, MarketDataEvent,
    MarketDataMessage, ProductHealth, ProductRejection, ProductSelector, SubscriptionAck,
    SubscriptionCommand,
};

pub const CONTRACT_VERSION: u32 = 1;
const CHANNEL_CAPACITY: usize = 4096;
const CLIENT_CAPACITY: usize = 1024;
const LATENCY_BUCKET_MICROS: [u64; 13] = [
    100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000,
    1_000_000,
];

#[derive(Debug, Default)]
struct LatencyHistogram {
    buckets: [u64; LATENCY_BUCKET_MICROS.len()],
    count: u64,
    sum_micros: u128,
}

impl LatencyHistogram {
    fn observe(&mut self, duration: Duration) {
        let micros = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
        for (index, upper_bound) in LATENCY_BUCKET_MICROS.iter().enumerate() {
            if micros <= *upper_bound {
                self.buckets[index] = self.buckets[index].saturating_add(1);
            }
        }
        self.count = self.count.saturating_add(1);
        self.sum_micros = self.sum_micros.saturating_add(u128::from(micros));
    }
}

#[derive(Debug, Default)]
struct RealtimePipelineMetrics {
    websocket_frames: u64,
    websocket_bytes: u64,
    websocket_queue_depth: usize,
    websocket_queue_capacity: usize,
    websocket_queue_high_watermark: usize,
    websocket_queue_overflows: u64,
    publication_queue_depth: usize,
    publication_queue_capacity: usize,
    publication_queue_high_watermark: usize,
    publication_queue_overflows: u64,
    websocket_queue_delay: LatencyHistogram,
    websocket_interframe: LatencyHistogram,
    websocket_io_scheduling_delay: LatencyHistogram,
    clob_frame_parse: LatencyHistogram,
    clob_book_apply: LatencyHistogram,
    clob_sample_build: LatencyHistogram,
    publication_queue_delay: LatencyHistogram,
    publication: LatencyHistogram,
    clob_messages: u64,
    clob_changes: u64,
    clob_bid_levels: usize,
    clob_ask_levels: usize,
    clob_current_market_ready: bool,
    clob_successor_market_ready: bool,
    gamma_refresh_successes: u64,
    gamma_refresh_failures: u64,
    cached_contract_reconnects: u64,
}

#[derive(Clone)]
pub struct Publisher {
    worker_id: Arc<str>,
    epoch: Arc<str>,
    tx: broadcast::Sender<Arc<MarketDataEvent>>,
    latest: Arc<RwLock<BTreeMap<String, Arc<MarketDataEvent>>>>,
    sequences: Arc<Mutex<BTreeMap<String, u64>>>,
    metrics: Arc<StreamingMetrics>,
}

#[derive(Default)]
pub struct StreamingMetrics {
    published: Mutex<BTreeMap<String, u64>>,
    subscribers: Mutex<BTreeMap<String, i64>>,
    dropped: Mutex<BTreeMap<String, u64>>,
    last_published_micros: Mutex<BTreeMap<String, i64>>,
    source_reconnects: Mutex<BTreeMap<(String, String), u64>>,
    source_connection_attempts: Mutex<BTreeMap<(String, String), u64>>,
    source_connection_failures: Mutex<BTreeMap<(String, String, String), u64>>,
    source_active_endpoint: Mutex<BTreeMap<(String, String), i64>>,
    source_snapshot_requests: Mutex<BTreeMap<(String, String, String), u64>>,
    source_last_event_micros: Mutex<BTreeMap<String, i64>>,
    source_connection_ready: Mutex<BTreeMap<String, i64>>,
    source_stale_transitions: Mutex<BTreeMap<(String, String), u64>>,
    websocket_pings: Mutex<BTreeMap<String, u64>>,
    websocket_pongs: Mutex<BTreeMap<String, u64>>,
    websocket_pong_latency_micros: Mutex<BTreeMap<String, u64>>,
    persistence_latency_micros: Mutex<BTreeMap<String, u64>>,
    persistence_queue_depth: Mutex<BTreeMap<String, usize>>,
    persistence_queue_overflows: Mutex<BTreeMap<String, u64>>,
    realtime_pipeline: Mutex<BTreeMap<String, RealtimePipelineMetrics>>,
}

static PUBLISHER: OnceLock<Publisher> = OnceLock::new();

impl Publisher {
    pub fn install(worker_id: impl Into<Arc<str>>) -> Self {
        let (tx, _) = broadcast::channel(CHANNEL_CAPACITY);
        let publisher = Self {
            worker_id: worker_id.into(),
            epoch: Arc::from(Uuid::new_v4().to_string()),
            tx,
            latest: Arc::new(RwLock::new(BTreeMap::new())),
            sequences: Arc::new(Mutex::new(BTreeMap::new())),
            metrics: Arc::new(StreamingMetrics::default()),
        };
        let _ = PUBLISHER.set(publisher.clone());
        publisher
    }

    pub async fn publish<T: Serialize>(
        &self,
        product_key: &'static str,
        source_event_id: impl Into<String>,
        source_timestamp: DateTime<Utc>,
        provider_available_at: DateTime<Utc>,
        received_at: DateTime<Utc>,
        payload_sha256: impl Into<String>,
        persistence_healthy: bool,
        payload: &T,
    ) -> Result<()> {
        let payload_json = serde_json::to_vec(payload).context("serialize streaming payload")?;
        let sequence = {
            let mut sequences = self.sequences.lock().expect("sequence lock");
            let sequence = sequences.entry(product_key.to_owned()).or_insert(0);
            *sequence = sequence.saturating_add(1);
            *sequence
        };
        let published_at = Utc::now();
        let event = Arc::new(MarketDataEvent {
            product_key: product_key.to_owned(),
            contract_version: CONTRACT_VERSION,
            worker_id: self.worker_id.to_string(),
            publisher_epoch: self.epoch.to_string(),
            sequence,
            source_event_id: source_event_id.into(),
            source_timestamp_micros: source_timestamp.timestamp_micros(),
            provider_available_at_micros: provider_available_at.timestamp_micros(),
            received_at_micros: received_at.timestamp_micros(),
            published_at_micros: published_at.timestamp_micros(),
            payload_sha256: payload_sha256.into(),
            integrity: "ok".to_owned(),
            persistence_healthy,
            payload_json,
        });
        self.latest
            .write()
            .await
            .insert(product_key.to_owned(), event.clone());
        self.metrics
            .observe_publish(product_key, published_at.timestamp_micros());
        let _ = self.tx.send(event);
        Ok(())
    }

    pub async fn serve(
        self,
        bind: SocketAddr,
        token: Arc<str>,
        shutdown: CancellationToken,
    ) -> Result<()> {
        let worker_id = self.worker_id.clone();
        let service = StreamService { publisher: self };
        let expected = token.clone();
        let intercepted =
            proto::market_data_stream_server::MarketDataStreamServer::with_interceptor(
                service,
                move |request: Request<()>| {
                    let supplied = request
                        .metadata()
                        .get("authorization")
                        .and_then(|value| value.to_str().ok())
                        .and_then(|value| value.strip_prefix("Bearer "))
                        .unwrap_or_default();
                    if supplied.as_bytes().ct_eq(expected.as_bytes()).into() {
                        Ok(request)
                    } else {
                        Err(Status::unauthenticated(
                            "invalid ingester stream credential",
                        ))
                    }
                },
            );
        info!(%bind, worker_id=%worker_id, "ingester market-data gRPC server starting");
        tonic::transport::Server::builder()
            .http2_keepalive_interval(Some(Duration::from_secs(10)))
            .http2_keepalive_timeout(Some(Duration::from_secs(5)))
            .add_service(intercepted)
            .serve_with_shutdown(bind, shutdown.cancelled_owned())
            .await
            .context("market-data gRPC server failed")
    }

    pub fn render_metrics(&self) -> String {
        self.metrics.render()
    }
}

pub async fn publish<T: Serialize>(
    product_key: &'static str,
    source_event_id: impl Into<String>,
    source_timestamp: DateTime<Utc>,
    provider_available_at: DateTime<Utc>,
    received_at: DateTime<Utc>,
    payload_sha256: impl Into<String>,
    persistence_healthy: bool,
    payload: &T,
) {
    if let Some(publisher) = PUBLISHER.get() {
        if let Err(error) = publisher
            .publish(
                product_key,
                source_event_id,
                source_timestamp,
                provider_available_at,
                received_at,
                payload_sha256,
                persistence_healthy,
                payload,
            )
            .await
        {
            warn!(product_key, %error, "failed to encode canonical market-data event");
        }
    }
}

pub fn observe_source_reconnect(product_key: &str, reason: &str) {
    if let Some(publisher) = PUBLISHER.get() {
        let mut reconnects = publisher
            .metrics
            .source_reconnects
            .lock()
            .expect("metrics lock");
        *reconnects
            .entry((product_key.to_owned(), reason.to_owned()))
            .or_insert(0) += 1;
    }
}

pub fn observe_source_connection_attempt(product_key: &str, endpoint: &str) {
    if let Some(publisher) = PUBLISHER.get() {
        let mut attempts = publisher
            .metrics
            .source_connection_attempts
            .lock()
            .expect("metrics lock");
        *attempts
            .entry((product_key.to_owned(), endpoint.to_owned()))
            .or_insert(0) += 1;
    }
}

pub fn observe_source_connection_failure(product_key: &str, endpoint: &str, reason: &str) {
    if let Some(publisher) = PUBLISHER.get() {
        let mut failures = publisher
            .metrics
            .source_connection_failures
            .lock()
            .expect("metrics lock");
        *failures
            .entry((
                product_key.to_owned(),
                endpoint.to_owned(),
                reason.to_owned(),
            ))
            .or_insert(0) += 1;
    }
}

pub fn set_source_active_endpoint(product_key: &str, endpoint: &str) {
    if let Some(publisher) = PUBLISHER.get() {
        let mut endpoints = publisher
            .metrics
            .source_active_endpoint
            .lock()
            .expect("metrics lock");
        for ((product, _), value) in endpoints.iter_mut() {
            if product == product_key {
                *value = 0;
            }
        }
        endpoints.insert((product_key.to_owned(), endpoint.to_owned()), 1);
    }
}

pub fn observe_source_snapshot_request(product_key: &str, endpoint: &str, outcome: &str) {
    if let Some(publisher) = PUBLISHER.get() {
        let mut requests = publisher
            .metrics
            .source_snapshot_requests
            .lock()
            .expect("metrics lock");
        *requests
            .entry((
                product_key.to_owned(),
                endpoint.to_owned(),
                outcome.to_owned(),
            ))
            .or_insert(0) += 1;
    }
}

pub fn observe_source_event(product_key: &str, at: DateTime<Utc>) {
    if let Some(publisher) = PUBLISHER.get() {
        publisher
            .metrics
            .source_last_event_micros
            .lock()
            .expect("metrics lock")
            .insert(product_key.to_owned(), at.timestamp_micros());
    }
}

pub fn set_source_connection_ready(product_key: &str, ready: bool) {
    if let Some(publisher) = PUBLISHER.get() {
        publisher
            .metrics
            .source_connection_ready
            .lock()
            .expect("metrics lock")
            .insert(product_key.to_owned(), i64::from(ready));
    }
}

pub fn observe_source_stale_transition(product_key: &str, reason: &str) {
    if let Some(publisher) = PUBLISHER.get() {
        let mut transitions = publisher
            .metrics
            .source_stale_transitions
            .lock()
            .expect("metrics lock");
        *transitions
            .entry((product_key.to_owned(), reason.to_owned()))
            .or_insert(0) += 1;
    }
}

pub fn observe_websocket_ping(product_key: &str) {
    if let Some(publisher) = PUBLISHER.get() {
        let mut pings = publisher
            .metrics
            .websocket_pings
            .lock()
            .expect("metrics lock");
        *pings.entry(product_key.to_owned()).or_insert(0) += 1;
    }
}

pub fn observe_websocket_pong(product_key: &str, latency: Duration) {
    if let Some(publisher) = PUBLISHER.get() {
        let mut pongs = publisher
            .metrics
            .websocket_pongs
            .lock()
            .expect("metrics lock");
        *pongs.entry(product_key.to_owned()).or_insert(0) += 1;
        publisher
            .metrics
            .websocket_pong_latency_micros
            .lock()
            .expect("metrics lock")
            .insert(
                product_key.to_owned(),
                u64::try_from(latency.as_micros()).unwrap_or(u64::MAX),
            );
    }
}

pub fn observe_persistence(product_key: &str, latency: Duration) {
    if let Some(publisher) = PUBLISHER.get() {
        publisher
            .metrics
            .persistence_latency_micros
            .lock()
            .expect("metrics lock")
            .insert(
                product_key.to_owned(),
                u64::try_from(latency.as_micros()).unwrap_or(u64::MAX),
            );
    }
}

pub fn set_persistence_queue_depth(product_key: &str, depth: usize) {
    if let Some(publisher) = PUBLISHER.get() {
        publisher
            .metrics
            .persistence_queue_depth
            .lock()
            .expect("metrics lock")
            .insert(product_key.to_owned(), depth);
    }
}

pub fn observe_persistence_queue_overflow(product_key: &str) {
    if let Some(publisher) = PUBLISHER.get() {
        let mut overflows = publisher
            .metrics
            .persistence_queue_overflows
            .lock()
            .expect("metrics lock");
        *overflows.entry(product_key.to_owned()).or_insert(0) += 1;
    }
}

fn with_pipeline_metrics(product_key: &str, observe: impl FnOnce(&mut RealtimePipelineMetrics)) {
    if let Some(publisher) = PUBLISHER.get() {
        let mut metrics = publisher
            .metrics
            .realtime_pipeline
            .lock()
            .expect("metrics lock");
        observe(metrics.entry(product_key.to_owned()).or_default());
    }
}

pub fn observe_websocket_frame(product_key: &str, bytes: usize, interframe: Option<Duration>) {
    with_pipeline_metrics(product_key, |metrics| {
        metrics.websocket_frames = metrics.websocket_frames.saturating_add(1);
        metrics.websocket_bytes = metrics
            .websocket_bytes
            .saturating_add(u64::try_from(bytes).unwrap_or(u64::MAX));
        if let Some(interframe) = interframe {
            metrics.websocket_interframe.observe(interframe);
        }
    });
}

pub fn set_websocket_queue_depth(product_key: &str, depth: usize, capacity: usize) {
    with_pipeline_metrics(product_key, |metrics| {
        metrics.websocket_queue_depth = depth;
        metrics.websocket_queue_capacity = capacity;
        metrics.websocket_queue_high_watermark = metrics.websocket_queue_high_watermark.max(depth);
    });
}

pub fn observe_websocket_queue_overflow(product_key: &str) {
    with_pipeline_metrics(product_key, |metrics| {
        metrics.websocket_queue_overflows = metrics.websocket_queue_overflows.saturating_add(1);
    });
}

pub fn observe_websocket_queue_delay(product_key: &str, duration: Duration) {
    with_pipeline_metrics(product_key, |metrics| {
        metrics.websocket_queue_delay.observe(duration);
    });
}

pub fn observe_websocket_io_scheduling_delay(product_key: &str, duration: Duration) {
    with_pipeline_metrics(product_key, |metrics| {
        metrics.websocket_io_scheduling_delay.observe(duration);
    });
}

pub fn observe_clob_frame_processing(
    product_key: &str,
    parse: Duration,
    apply: Duration,
    sample_build: Duration,
    messages: usize,
    changes: usize,
    bid_levels: usize,
    ask_levels: usize,
) {
    with_pipeline_metrics(product_key, |metrics| {
        metrics.clob_frame_parse.observe(parse);
        metrics.clob_book_apply.observe(apply);
        metrics.clob_sample_build.observe(sample_build);
        metrics.clob_messages = metrics
            .clob_messages
            .saturating_add(u64::try_from(messages).unwrap_or(u64::MAX));
        metrics.clob_changes = metrics
            .clob_changes
            .saturating_add(u64::try_from(changes).unwrap_or(u64::MAX));
        metrics.clob_bid_levels = bid_levels;
        metrics.clob_ask_levels = ask_levels;
    });
}

pub fn set_clob_bootstrap_readiness(product_key: &str, current: bool, successor: bool) {
    with_pipeline_metrics(product_key, |metrics| {
        metrics.clob_current_market_ready = current;
        metrics.clob_successor_market_ready = successor;
    });
}

pub fn observe_gamma_refresh(product_key: &str, success: bool) {
    with_pipeline_metrics(product_key, |metrics| {
        if success {
            metrics.gamma_refresh_successes = metrics.gamma_refresh_successes.saturating_add(1);
        } else {
            metrics.gamma_refresh_failures = metrics.gamma_refresh_failures.saturating_add(1);
        }
    });
}

pub fn observe_cached_contract_reconnect(product_key: &str) {
    with_pipeline_metrics(product_key, |metrics| {
        metrics.cached_contract_reconnects = metrics.cached_contract_reconnects.saturating_add(1);
    });
}

pub fn set_publication_queue_depth(product_key: &str, depth: usize, capacity: usize) {
    with_pipeline_metrics(product_key, |metrics| {
        metrics.publication_queue_depth = depth;
        metrics.publication_queue_capacity = capacity;
        metrics.publication_queue_high_watermark =
            metrics.publication_queue_high_watermark.max(depth);
    });
}

pub fn observe_publication_queue_overflow(product_key: &str) {
    with_pipeline_metrics(product_key, |metrics| {
        metrics.publication_queue_overflows = metrics.publication_queue_overflows.saturating_add(1);
    });
}

pub fn observe_publication(product_key: &str, queue_delay: Duration, duration: Duration) {
    with_pipeline_metrics(product_key, |metrics| {
        metrics.publication_queue_delay.observe(queue_delay);
        metrics.publication.observe(duration);
    });
}

#[derive(Clone)]
struct StreamService {
    publisher: Publisher,
}

type ResponseStream = Pin<Box<dyn Stream<Item = Result<MarketDataMessage, Status>> + Send>>;

#[tonic::async_trait]
impl MarketDataStream for StreamService {
    type StreamStream = ResponseStream;

    async fn stream(
        &self,
        request: Request<tonic::Streaming<SubscriptionCommand>>,
    ) -> Result<Response<Self::StreamStream>, Status> {
        let mut commands = request.into_inner();
        let mut events = self.publisher.tx.subscribe();
        let (output, receiver) = mpsc::channel(CLIENT_CAPACITY);
        let publisher = self.publisher.clone();
        tokio::spawn(async move {
            let mut selected = BTreeSet::<String>::new();
            let mut selectors = BTreeMap::<String, ProductSelector>::new();
            let mut health_tick = tokio::time::interval(Duration::from_secs(5));
            health_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    command = commands.next() => {
                        let Some(command) = command else { break };
                        let command = match command { Ok(command) => command, Err(_) => break };
                        let rejected = command.products.iter().filter(|product| product.contract_version != CONTRACT_VERSION)
                            .map(|product| ProductRejection { product: Some(product.clone()), reason: "unsupported_contract_version".to_owned() }).collect::<Vec<_>>();
                        let accepted = command.products.iter().filter(|product| product.contract_version == CONTRACT_VERSION)
                            .cloned().collect::<Vec<_>>();
                        let accepted_keys = accepted.iter().map(|product| product.key.clone()).collect::<BTreeSet<_>>();
                        publisher.metrics.replace_subscriptions(&selected, &accepted_keys);
                        selected = accepted_keys;
                        selectors = accepted.iter().cloned().map(|selector| (selector.key.clone(), selector)).collect();
                        let ack = MarketDataMessage { message: Some(Message::SubscriptionAck(SubscriptionAck {
                            revision: command.revision,
                            accepted,
                            rejected,
                            acknowledged_at_micros: Utc::now().timestamp_micros(),
                        }))};
                        if output.send(Ok(ack)).await.is_err() { break; }
                        let snapshots = publisher.latest.read().await;
                        for key in &selected {
                            let health = ProductHealth {
                                product_key: key.clone(), contract_version: CONTRACT_VERSION,
                                worker_id: publisher.worker_id.to_string(), publisher_epoch: publisher.epoch.to_string(),
                                ready: snapshots.contains_key(key),
                                last_event_at_micros: snapshots.get(key).map_or(0, |event| event.published_at_micros),
                                detail: if snapshots.contains_key(key) { "snapshot_available" } else { "awaiting_first_event" }.to_owned(),
                            };
                            if output.send(Ok(MarketDataMessage { message: Some(Message::Health(health)) })).await.is_err() { return; }
                            if let Some(event) = snapshots.get(key) {
                                if output.send(Ok(MarketDataMessage { message: Some(Message::Event((**event).clone())) })).await.is_err() { return; }
                            }
                        }
                    }
                    _ = health_tick.tick(), if !selected.is_empty() => {
                        let snapshots = publisher.latest.read().await;
                        let now = Utc::now().timestamp_micros();
                        for key in &selected {
                            let snapshot = snapshots.get(key);
                            let maximum_age_ms = selectors.get(key).map_or(0, |selector| selector.maximum_age_ms);
                            let fresh = maximum_age_ms == 0 || snapshot.is_some_and(|event| {
                                now.saturating_sub(event.published_at_micros) <= i64::try_from(maximum_age_ms.saturating_mul(1_000)).unwrap_or(i64::MAX)
                            });
                            let health = ProductHealth {
                                product_key: key.clone(), contract_version: CONTRACT_VERSION,
                                worker_id: publisher.worker_id.to_string(), publisher_epoch: publisher.epoch.to_string(),
                                ready: fresh,
                                last_event_at_micros: snapshot.map_or(0, |event| event.published_at_micros),
                                detail: if fresh { "current" } else if snapshot.is_some() { "stale" } else { "awaiting_first_event" }.to_owned(),
                            };
                            if output.try_send(Ok(MarketDataMessage { message: Some(Message::Health(health)) })).is_err() { return; }
                        }
                    }
                    event = events.recv() => match event {
                        Ok(event) if selected.contains(&event.product_key) => {
                            let key = event.product_key.clone();
                            if output.try_send(Ok(MarketDataMessage { message: Some(Message::Event((*event).clone())) })).is_err() {
                                publisher.metrics.observe_drops(&key, 1);
                                break;
                            }
                        }
                        Ok(_) => {}
                        Err(broadcast::error::RecvError::Lagged(skipped)) => {
                            for key in &selected { publisher.metrics.observe_drops(key, skipped); }
                            break;
                        }
                        Err(broadcast::error::RecvError::Closed) => break,
                    }
                }
            }
            publisher
                .metrics
                .replace_subscriptions(&selected, &BTreeSet::new());
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(receiver))))
    }
}

impl StreamingMetrics {
    fn observe_publish(&self, key: &str, at: i64) {
        let mut published = self.published.lock().expect("metrics lock");
        *published.entry(key.to_owned()).or_insert(0) += 1;
        self.last_published_micros
            .lock()
            .expect("metrics lock")
            .insert(key.to_owned(), at);
    }

    fn replace_subscriptions(&self, old: &BTreeSet<String>, new: &BTreeSet<String>) {
        let mut subscriptions = self.subscribers.lock().expect("metrics lock");
        for key in old.difference(new) {
            *subscriptions.entry(key.clone()).or_insert(0) -= 1;
        }
        for key in new.difference(old) {
            *subscriptions.entry(key.clone()).or_insert(0) += 1;
        }
    }

    fn observe_drops(&self, key: &str, count: u64) {
        *self
            .dropped
            .lock()
            .expect("metrics lock")
            .entry(key.to_owned())
            .or_insert(0) += count;
    }

    fn render(&self) -> String {
        let published = self.published.lock().expect("metrics lock");
        let subscriptions = self.subscribers.lock().expect("metrics lock");
        let dropped = self.dropped.lock().expect("metrics lock");
        let last = self.last_published_micros.lock().expect("metrics lock");
        let reconnects = self.source_reconnects.lock().expect("metrics lock");
        let connection_attempts = self
            .source_connection_attempts
            .lock()
            .expect("metrics lock");
        let connection_failures = self
            .source_connection_failures
            .lock()
            .expect("metrics lock");
        let active_endpoint = self.source_active_endpoint.lock().expect("metrics lock");
        let snapshot_requests = self.source_snapshot_requests.lock().expect("metrics lock");
        let source_last_event = self.source_last_event_micros.lock().expect("metrics lock");
        let source_ready = self.source_connection_ready.lock().expect("metrics lock");
        let stale_transitions = self.source_stale_transitions.lock().expect("metrics lock");
        let pings = self.websocket_pings.lock().expect("metrics lock");
        let pongs = self.websocket_pongs.lock().expect("metrics lock");
        let pong_latency = self
            .websocket_pong_latency_micros
            .lock()
            .expect("metrics lock");
        let persistence_latency = self
            .persistence_latency_micros
            .lock()
            .expect("metrics lock");
        let queue_depth = self.persistence_queue_depth.lock().expect("metrics lock");
        let queue_overflows = self
            .persistence_queue_overflows
            .lock()
            .expect("metrics lock");
        let realtime_pipeline = self.realtime_pipeline.lock().expect("metrics lock");
        let mut out = String::from(
            "# HELP ingester_stream_published_total Canonical events published to the worker stream.\n# TYPE ingester_stream_published_total counter\n",
        );
        for (key, value) in published.iter() {
            out.push_str(&format!(
                "ingester_stream_published_total{{product=\"{key}\"}} {value}\n"
            ));
        }
        out.push_str("# HELP ingester_stream_subscribers Active consumer subscriptions after selector deduplication.\n# TYPE ingester_stream_subscribers gauge\n");
        for (key, value) in subscriptions.iter() {
            out.push_str(&format!(
                "ingester_stream_subscribers{{product=\"{key}\"}} {}\n",
                value.max(&0)
            ));
        }
        out.push_str("# HELP ingester_stream_dropped_total Events dropped for slow consumers.\n# TYPE ingester_stream_dropped_total counter\n");
        for (key, value) in dropped.iter() {
            out.push_str(&format!(
                "ingester_stream_dropped_total{{product=\"{key}\"}} {value}\n"
            ));
        }
        out.push_str("# HELP ingester_stream_last_published_timestamp_seconds Last canonical publish time.\n# TYPE ingester_stream_last_published_timestamp_seconds gauge\n");
        for (key, value) in last.iter() {
            out.push_str(&format!(
                "ingester_stream_last_published_timestamp_seconds{{product=\"{key}\"}} {}\n",
                *value as f64 / 1_000_000.0
            ));
        }
        out.push_str("# HELP ingester_source_reconnects_total Provider source reconnects by bounded reason code.\n# TYPE ingester_source_reconnects_total counter\n");
        for ((key, reason), value) in reconnects.iter() {
            out.push_str(&format!(
                "ingester_source_reconnects_total{{product=\"{key}\",reason=\"{reason}\"}} {value}\n"
            ));
        }
        out.push_str("# HELP ingester_source_connection_attempts_total Provider connection attempts by bounded endpoint label.\n# TYPE ingester_source_connection_attempts_total counter\n");
        for ((key, endpoint), value) in connection_attempts.iter() {
            out.push_str(&format!(
                "ingester_source_connection_attempts_total{{product=\"{key}\",endpoint=\"{endpoint}\"}} {value}\n"
            ));
        }
        out.push_str("# HELP ingester_source_connection_failures_total Provider connection failures by bounded endpoint and reason.\n# TYPE ingester_source_connection_failures_total counter\n");
        for ((key, endpoint, reason), value) in connection_failures.iter() {
            out.push_str(&format!(
                "ingester_source_connection_failures_total{{product=\"{key}\",endpoint=\"{endpoint}\",reason=\"{reason}\"}} {value}\n"
            ));
        }
        out.push_str("# HELP ingester_source_active_endpoint Currently connected bounded provider endpoint.\n# TYPE ingester_source_active_endpoint gauge\n");
        for ((key, endpoint), value) in active_endpoint.iter() {
            out.push_str(&format!(
                "ingester_source_active_endpoint{{product=\"{key}\",endpoint=\"{endpoint}\"}} {value}\n"
            ));
        }
        out.push_str("# HELP ingester_source_snapshot_requests_total Provider snapshot requests by bounded endpoint and outcome.\n# TYPE ingester_source_snapshot_requests_total counter\n");
        for ((key, endpoint, outcome), value) in snapshot_requests.iter() {
            out.push_str(&format!(
                "ingester_source_snapshot_requests_total{{product=\"{key}\",endpoint=\"{endpoint}\",outcome=\"{outcome}\"}} {value}\n"
            ));
        }
        out.push_str("# HELP ingester_source_last_event_timestamp_seconds Latest valid event received from the provider.\n# TYPE ingester_source_last_event_timestamp_seconds gauge\n");
        for (key, value) in source_last_event.iter() {
            out.push_str(&format!(
                "ingester_source_last_event_timestamp_seconds{{product=\"{key}\"}} {}\n",
                *value as f64 / 1_000_000.0
            ));
        }
        out.push_str("# HELP ingester_source_connection_ready Whether the provider connection and product subscription are established.\n# TYPE ingester_source_connection_ready gauge\n");
        for (key, value) in source_ready.iter() {
            out.push_str(&format!(
                "ingester_source_connection_ready{{product=\"{key}\"}} {value}\n"
            ));
        }
        out.push_str("# HELP ingester_source_stale_transitions_total Transitions into provider-source staleness by bounded reason.\n# TYPE ingester_source_stale_transitions_total counter\n");
        for ((key, reason), value) in stale_transitions.iter() {
            out.push_str(&format!(
                "ingester_source_stale_transitions_total{{product=\"{key}\",reason=\"{reason}\"}} {value}\n"
            ));
        }
        out.push_str("# HELP ingester_source_websocket_pings_total Provider websocket PING frames received.\n# TYPE ingester_source_websocket_pings_total counter\n");
        for (key, value) in pings.iter() {
            out.push_str(&format!(
                "ingester_source_websocket_pings_total{{product=\"{key}\"}} {value}\n"
            ));
        }
        out.push_str("# HELP ingester_source_websocket_pongs_total Provider websocket PONG frames sent.\n# TYPE ingester_source_websocket_pongs_total counter\n");
        for (key, value) in pongs.iter() {
            out.push_str(&format!(
                "ingester_source_websocket_pongs_total{{product=\"{key}\"}} {value}\n"
            ));
        }
        out.push_str("# HELP ingester_source_websocket_pong_latency_seconds Latest provider PING-to-PONG latency.\n# TYPE ingester_source_websocket_pong_latency_seconds gauge\n");
        for (key, value) in pong_latency.iter() {
            out.push_str(&format!(
                "ingester_source_websocket_pong_latency_seconds{{product=\"{key}\"}} {}\n",
                *value as f64 / 1_000_000.0
            ));
        }
        out.push_str("# HELP ingester_source_persistence_latency_seconds Latest realtime persistence batch latency.\n# TYPE ingester_source_persistence_latency_seconds gauge\n");
        for (key, value) in persistence_latency.iter() {
            out.push_str(&format!(
                "ingester_source_persistence_latency_seconds{{product=\"{key}\"}} {}\n",
                *value as f64 / 1_000_000.0
            ));
        }
        out.push_str("# HELP ingester_source_persistence_queue_depth Realtime events awaiting persistence.\n# TYPE ingester_source_persistence_queue_depth gauge\n");
        for (key, value) in queue_depth.iter() {
            out.push_str(&format!(
                "ingester_source_persistence_queue_depth{{product=\"{key}\"}} {value}\n"
            ));
        }
        out.push_str("# HELP ingester_source_persistence_queue_overflows_total Realtime persistence handoff overflows.\n# TYPE ingester_source_persistence_queue_overflows_total counter\n");
        for (key, value) in queue_overflows.iter() {
            out.push_str(&format!(
                "ingester_source_persistence_queue_overflows_total{{product=\"{key}\"}} {value}\n"
            ));
        }
        out.push_str("# HELP market_data_ingester_websocket_frames_total Websocket data frames received.\n# TYPE market_data_ingester_websocket_frames_total counter\n");
        out.push_str("# HELP market_data_ingester_websocket_bytes_total Websocket data-frame bytes received.\n# TYPE market_data_ingester_websocket_bytes_total counter\n");
        out.push_str("# HELP market_data_ingester_websocket_queue_depth Raw websocket frames awaiting processing.\n# TYPE market_data_ingester_websocket_queue_depth gauge\n");
        out.push_str("# HELP market_data_ingester_websocket_queue_capacity Raw websocket frame queue capacity.\n# TYPE market_data_ingester_websocket_queue_capacity gauge\n");
        out.push_str("# HELP market_data_ingester_websocket_queue_utilization_ratio Raw websocket queue utilization.\n# TYPE market_data_ingester_websocket_queue_utilization_ratio gauge\n");
        out.push_str("# HELP market_data_ingester_websocket_queue_high_watermark Highest raw websocket queue depth since worker start.\n# TYPE market_data_ingester_websocket_queue_high_watermark gauge\n");
        out.push_str("# HELP market_data_ingester_websocket_queue_overflow_total Raw websocket frame handoff overflows.\n# TYPE market_data_ingester_websocket_queue_overflow_total counter\n");
        out.push_str("# HELP market_data_ingester_publication_queue_depth Canonical samples awaiting direct-stream publication.\n# TYPE market_data_ingester_publication_queue_depth gauge\n");
        out.push_str("# HELP market_data_ingester_publication_queue_capacity Direct-stream publication queue capacity.\n# TYPE market_data_ingester_publication_queue_capacity gauge\n");
        out.push_str("# HELP market_data_ingester_publication_queue_utilization_ratio Direct-stream publication queue utilization.\n# TYPE market_data_ingester_publication_queue_utilization_ratio gauge\n");
        out.push_str("# HELP market_data_ingester_publication_queue_high_watermark Highest publication queue depth since worker start.\n# TYPE market_data_ingester_publication_queue_high_watermark gauge\n");
        out.push_str("# HELP market_data_ingester_publication_queue_overflow_total Direct-stream publication handoff overflows.\n# TYPE market_data_ingester_publication_queue_overflow_total counter\n");
        out.push_str("# HELP market_data_ingester_clob_messages_total Parsed CLOB messages.\n# TYPE market_data_ingester_clob_messages_total counter\n");
        out.push_str("# HELP market_data_ingester_clob_changes_total Parsed CLOB price changes.\n# TYPE market_data_ingester_clob_changes_total counter\n");
        out.push_str("# HELP market_data_ingester_clob_levels Current reconstructed CLOB levels by side.\n# TYPE market_data_ingester_clob_levels gauge\n");
        out.push_str("# HELP market_data_ingester_clob_bootstrap_ready Whether authoritative CLOB books are ready by bounded scope.\n# TYPE market_data_ingester_clob_bootstrap_ready gauge\n");
        out.push_str("# HELP market_data_ingester_gamma_refresh_total Gamma contract refresh attempts by result.\n# TYPE market_data_ingester_gamma_refresh_total counter\n");
        out.push_str("# HELP market_data_ingester_cached_contract_reconnect_total Websocket reconnects that reused the last verified contract set.\n# TYPE market_data_ingester_cached_contract_reconnect_total counter\n");
        for name in [
            "market_data_ingester_websocket_queue_delay_seconds",
            "market_data_ingester_websocket_interframe_seconds",
            "market_data_ingester_websocket_io_scheduling_delay_seconds",
            "market_data_ingester_clob_frame_parse_seconds",
            "market_data_ingester_clob_book_apply_seconds",
            "market_data_ingester_clob_sample_build_seconds",
            "market_data_ingester_publication_queue_delay_seconds",
            "market_data_ingester_publication_seconds",
        ] {
            out.push_str(&format!(
                "# HELP {name} Observed realtime pipeline latency.\n# TYPE {name} histogram\n"
            ));
        }
        for (key, metrics) in realtime_pipeline.iter() {
            let websocket_utilization = ratio(
                metrics.websocket_queue_depth,
                metrics.websocket_queue_capacity,
            );
            let publication_utilization = ratio(
                metrics.publication_queue_depth,
                metrics.publication_queue_capacity,
            );
            out.push_str(&format!(
                "market_data_ingester_websocket_frames_total{{product=\"{key}\"}} {}\nmarket_data_ingester_websocket_bytes_total{{product=\"{key}\"}} {}\nmarket_data_ingester_websocket_queue_depth{{product=\"{key}\"}} {}\nmarket_data_ingester_websocket_queue_capacity{{product=\"{key}\"}} {}\nmarket_data_ingester_websocket_queue_utilization_ratio{{product=\"{key}\"}} {websocket_utilization}\nmarket_data_ingester_websocket_queue_high_watermark{{product=\"{key}\"}} {}\nmarket_data_ingester_websocket_queue_overflow_total{{product=\"{key}\"}} {}\nmarket_data_ingester_publication_queue_depth{{product=\"{key}\"}} {}\nmarket_data_ingester_publication_queue_capacity{{product=\"{key}\"}} {}\nmarket_data_ingester_publication_queue_utilization_ratio{{product=\"{key}\"}} {publication_utilization}\nmarket_data_ingester_publication_queue_high_watermark{{product=\"{key}\"}} {}\nmarket_data_ingester_publication_queue_overflow_total{{product=\"{key}\"}} {}\nmarket_data_ingester_clob_messages_total{{product=\"{key}\"}} {}\nmarket_data_ingester_clob_changes_total{{product=\"{key}\"}} {}\nmarket_data_ingester_clob_levels{{product=\"{key}\",side=\"bid\"}} {}\nmarket_data_ingester_clob_levels{{product=\"{key}\",side=\"ask\"}} {}\n",
                metrics.websocket_frames,
                metrics.websocket_bytes,
                metrics.websocket_queue_depth,
                metrics.websocket_queue_capacity,
                metrics.websocket_queue_high_watermark,
                metrics.websocket_queue_overflows,
                metrics.publication_queue_depth,
                metrics.publication_queue_capacity,
                metrics.publication_queue_high_watermark,
                metrics.publication_queue_overflows,
                metrics.clob_messages,
                metrics.clob_changes,
                metrics.clob_bid_levels,
                metrics.clob_ask_levels,
            ));
            out.push_str(&format!(
                "market_data_ingester_clob_bootstrap_ready{{product=\"{key}\",scope=\"current\"}} {}\nmarket_data_ingester_clob_bootstrap_ready{{product=\"{key}\",scope=\"successor\"}} {}\n",
                i32::from(metrics.clob_current_market_ready),
                i32::from(metrics.clob_successor_market_ready),
            ));
            out.push_str(&format!(
                "market_data_ingester_gamma_refresh_total{{product=\"{key}\",result=\"success\"}} {}\nmarket_data_ingester_gamma_refresh_total{{product=\"{key}\",result=\"failure\"}} {}\nmarket_data_ingester_cached_contract_reconnect_total{{product=\"{key}\"}} {}\n",
                metrics.gamma_refresh_successes,
                metrics.gamma_refresh_failures,
                metrics.cached_contract_reconnects,
            ));
            render_histogram(
                &mut out,
                "market_data_ingester_websocket_queue_delay_seconds",
                key,
                &metrics.websocket_queue_delay,
            );
            render_histogram(
                &mut out,
                "market_data_ingester_websocket_interframe_seconds",
                key,
                &metrics.websocket_interframe,
            );
            render_histogram(
                &mut out,
                "market_data_ingester_websocket_io_scheduling_delay_seconds",
                key,
                &metrics.websocket_io_scheduling_delay,
            );
            render_histogram(
                &mut out,
                "market_data_ingester_clob_frame_parse_seconds",
                key,
                &metrics.clob_frame_parse,
            );
            render_histogram(
                &mut out,
                "market_data_ingester_clob_book_apply_seconds",
                key,
                &metrics.clob_book_apply,
            );
            render_histogram(
                &mut out,
                "market_data_ingester_clob_sample_build_seconds",
                key,
                &metrics.clob_sample_build,
            );
            render_histogram(
                &mut out,
                "market_data_ingester_publication_queue_delay_seconds",
                key,
                &metrics.publication_queue_delay,
            );
            render_histogram(
                &mut out,
                "market_data_ingester_publication_seconds",
                key,
                &metrics.publication,
            );
        }
        out
    }
}

fn ratio(depth: usize, capacity: usize) -> f64 {
    if capacity == 0 {
        0.0
    } else {
        depth as f64 / capacity as f64
    }
}

fn render_histogram(out: &mut String, name: &str, product: &str, histogram: &LatencyHistogram) {
    for (upper_bound, count) in LATENCY_BUCKET_MICROS.iter().zip(histogram.buckets.iter()) {
        out.push_str(&format!(
            "{name}_bucket{{product=\"{product}\",le=\"{}\"}} {count}\n",
            *upper_bound as f64 / 1_000_000.0
        ));
    }
    out.push_str(&format!(
        "{name}_bucket{{product=\"{product}\",le=\"+Inf\"}} {}\n{name}_sum{{product=\"{product}\"}} {}\n{name}_count{{product=\"{product}\"}} {}\n",
        histogram.count,
        histogram.sum_micros as f64 / 1_000_000.0,
        histogram.count,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_transport_and_persistence_health_metrics() {
        let metrics = StreamingMetrics::default();
        metrics
            .source_reconnects
            .lock()
            .expect("metrics lock")
            .insert(("product".to_owned(), "read_idle".to_owned()), 2);
        metrics
            .source_last_event_micros
            .lock()
            .expect("metrics lock")
            .insert("product".to_owned(), 1_500_000);
        metrics
            .source_connection_attempts
            .lock()
            .expect("metrics lock")
            .insert(("product".to_owned(), "primary".to_owned()), 4);
        metrics
            .source_connection_failures
            .lock()
            .expect("metrics lock")
            .insert(
                (
                    "product".to_owned(),
                    "primary".to_owned(),
                    "timeout".to_owned(),
                ),
                2,
            );
        metrics
            .source_active_endpoint
            .lock()
            .expect("metrics lock")
            .insert(("product".to_owned(), "alternate".to_owned()), 1);
        metrics
            .source_snapshot_requests
            .lock()
            .expect("metrics lock")
            .insert(
                (
                    "product".to_owned(),
                    "alternate".to_owned(),
                    "success".to_owned(),
                ),
                1,
            );
        metrics
            .source_connection_ready
            .lock()
            .expect("metrics lock")
            .insert("product".to_owned(), 1);
        metrics
            .source_stale_transitions
            .lock()
            .expect("metrics lock")
            .insert(("product".to_owned(), "topic_stale".to_owned()), 3);
        metrics
            .websocket_pong_latency_micros
            .lock()
            .expect("metrics lock")
            .insert("product".to_owned(), 1_500);
        metrics
            .persistence_queue_depth
            .lock()
            .expect("metrics lock")
            .insert("product".to_owned(), 7);
        let mut pipeline = RealtimePipelineMetrics {
            websocket_frames: 12,
            websocket_bytes: 4_096,
            websocket_queue_depth: 2,
            websocket_queue_capacity: 8,
            publication_queue_depth: 1,
            publication_queue_capacity: 4,
            clob_current_market_ready: true,
            ..Default::default()
        };
        pipeline
            .clob_frame_parse
            .observe(Duration::from_micros(500));
        metrics
            .realtime_pipeline
            .lock()
            .expect("metrics lock")
            .insert("product".to_owned(), pipeline);

        let rendered = metrics.render();
        assert!(rendered.contains(
            "ingester_source_reconnects_total{product=\"product\",reason=\"read_idle\"} 2"
        ));
        assert!(rendered
            .contains("ingester_source_last_event_timestamp_seconds{product=\"product\"} 1.5"));
        assert!(rendered.contains(
            "ingester_source_connection_attempts_total{product=\"product\",endpoint=\"primary\"} 4"
        ));
        assert!(rendered.contains(
            "ingester_source_connection_failures_total{product=\"product\",endpoint=\"primary\",reason=\"timeout\"} 2"
        ));
        assert!(rendered.contains(
            "ingester_source_active_endpoint{product=\"product\",endpoint=\"alternate\"} 1"
        ));
        assert!(rendered.contains(
            "ingester_source_snapshot_requests_total{product=\"product\",endpoint=\"alternate\",outcome=\"success\"} 1"
        ));
        assert!(rendered.contains("ingester_source_connection_ready{product=\"product\"} 1"));
        assert!(rendered.contains(
            "ingester_source_stale_transitions_total{product=\"product\",reason=\"topic_stale\"} 3"
        ));
        assert!(rendered.contains(
            "ingester_source_websocket_pong_latency_seconds{product=\"product\"} 0.0015"
        ));
        assert!(rendered.contains("ingester_source_persistence_queue_depth{product=\"product\"} 7"));
        assert!(rendered
            .contains("market_data_ingester_websocket_frames_total{product=\"product\"} 12"));
        assert!(rendered.contains(
            "market_data_ingester_websocket_queue_utilization_ratio{product=\"product\"} 0.25"
        ));
        assert!(rendered.contains(
            "market_data_ingester_clob_frame_parse_seconds_count{product=\"product\"} 1"
        ));
        assert!(rendered.contains(
            "market_data_ingester_clob_bootstrap_ready{product=\"product\",scope=\"current\"} 1"
        ));
    }
}
