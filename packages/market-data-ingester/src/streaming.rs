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
    websocket_pings: Mutex<BTreeMap<String, u64>>,
    websocket_pongs: Mutex<BTreeMap<String, u64>>,
    websocket_pong_latency_micros: Mutex<BTreeMap<String, u64>>,
    persistence_latency_micros: Mutex<BTreeMap<String, u64>>,
    persistence_queue_depth: Mutex<BTreeMap<String, usize>>,
    persistence_queue_overflows: Mutex<BTreeMap<String, u64>>,
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
        out
    }
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
            .websocket_pong_latency_micros
            .lock()
            .expect("metrics lock")
            .insert("product".to_owned(), 1_500);
        metrics
            .persistence_queue_depth
            .lock()
            .expect("metrics lock")
            .insert("product".to_owned(), 7);

        let rendered = metrics.render();
        assert!(rendered.contains(
            "ingester_source_reconnects_total{product=\"product\",reason=\"read_idle\"} 2"
        ));
        assert!(rendered.contains(
            "ingester_source_websocket_pong_latency_seconds{product=\"product\"} 0.0015"
        ));
        assert!(rendered.contains("ingester_source_persistence_queue_depth{product=\"product\"} 7"));
    }
}
