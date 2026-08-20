//! Continuous exact Polymarket RTDS Chainlink BTC/USD TWAP capture.

use std::{fmt::Write as _, str::FromStr, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use futures_util::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::{FromRow, PgPool, Postgres, Transaction};
use tokio::{sync::mpsc, task::JoinHandle, time::Instant};
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{protocol::WebSocketConfig, Message},
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    domain::{
        CaptureArtifact, IngesterProfile, IngesterStrategy, IngesterStrategyKey, StrategyError,
        StrategyErrorKind,
    },
    persistence::{
        ArtifactBatch, ArtifactRepository, GapRepository, NewCaptureArtifact, NewDataGap,
        ProfileRepository, StrategyDegradation, StrategyProgress,
    },
    runtime::{StrategyFactory, StrategyFactoryError},
};

const STRATEGY_KEY: IngesterStrategyKey = IngesterStrategyKey::PolymarketChainlinkBtcusdTwap;
const CONFIG_SCHEMA_VERSION: i32 = 1;
const CHECKPOINT_SCHEMA_VERSION: i32 = 1;
const DEFAULT_WEBSOCKET_URL: &str = "wss://ws-live-data.polymarket.com";
const SYMBOL: &str = "btc/usd";
const TOPIC_THIRTY: &str = "crypto_prices_twap_thirty";
const TOPIC_SIXTY: &str = "crypto_prices_twap_sixty";
const MAX_FRAME_BYTES: usize = 64 * 1024;
const EVENT_BUFFER: usize = 2_048;
const COMMAND_BUFFER: usize = 32;
const MAX_CLOCK_LEAD_MS: i64 = 5_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct PolymarketChainlinkBtcusdTwapConfig {
    pub websocket_url: String,
    pub symbol: String,
    pub connect_timeout_ms: u64,
    pub initial_stream_timeout_ms: u64,
    pub stream_stale_timeout_ms: u64,
    pub ping_interval_ms: u64,
    pub write_timeout_ms: u64,
    pub reconnect_initial_ms: u64,
    pub reconnect_max_ms: u64,
    pub artifact_window_seconds: i64,
}

impl Default for PolymarketChainlinkBtcusdTwapConfig {
    fn default() -> Self {
        Self {
            websocket_url: DEFAULT_WEBSOCKET_URL.to_owned(),
            symbol: SYMBOL.to_owned(),
            connect_timeout_ms: 10_000,
            initial_stream_timeout_ms: 20_000,
            stream_stale_timeout_ms: 30_000,
            ping_interval_ms: 5_000,
            write_timeout_ms: 5_000,
            reconnect_initial_ms: 250,
            reconnect_max_ms: 30_000,
            artifact_window_seconds: 3_600,
        }
    }
}

impl PolymarketChainlinkBtcusdTwapConfig {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let config = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::InvalidConfiguration(format!(
                "invalid Polymarket Chainlink TWAP config: {error}"
            ))
        })?;
        let invalid = |message: &str| StrategyFactoryError::InvalidConfiguration(message.into());
        if config.websocket_url != DEFAULT_WEBSOCKET_URL {
            return Err(invalid(
                "websocket_url must be the official Polymarket RTDS endpoint",
            ));
        }
        if config.symbol != SYMBOL {
            return Err(invalid("symbol must be btc/usd"));
        }
        if !(1_000..=30_000).contains(&config.connect_timeout_ms)
            || !(5_000..=60_000).contains(&config.initial_stream_timeout_ms)
            || !(10_000..=120_000).contains(&config.stream_stale_timeout_ms)
            || config.stream_stale_timeout_ms < config.initial_stream_timeout_ms
            || config.ping_interval_ms != 5_000
            || !(1_000..=15_000).contains(&config.write_timeout_ms)
            || !(100..=5_000).contains(&config.reconnect_initial_ms)
            || config.reconnect_max_ms < config.reconnect_initial_ms
            || config.reconnect_max_ms > 60_000
            || config.artifact_window_seconds < 60
            || config.artifact_window_seconds > 86_400
            || 86_400 % config.artifact_window_seconds != 0
        {
            return Err(invalid(
                "Polymarket Chainlink TWAP timing configuration is outside production bounds",
            ));
        }
        Ok(config)
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct TwapCheckpoint {
    thirty_source_timestamp_ms: Option<i64>,
    sixty_source_timestamp_ms: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RtdsEnvelope {
    topic: String,
    #[serde(rename = "type")]
    message_type: String,
    timestamp: i64,
    payload: RtdsPayload,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RtdsPayload {
    symbol: String,
    value: Value,
    full_accuracy_value: String,
    timestamp: i64,
    window_s: i16,
}

#[derive(Debug)]
struct TwapObservation {
    source_timestamp: DateTime<Utc>,
    published_at: DateTime<Utc>,
    received_at: DateTime<Utc>,
    window_seconds: i16,
    price: Decimal,
    full_accuracy_value: String,
    source_payload: Value,
    payload_sha256: String,
}

#[derive(Debug, FromRow)]
struct StoredTwap {
    published_at: DateTime<Utc>,
    twap_price: Decimal,
    full_accuracy_value: String,
    payload_sha256: String,
}

#[derive(Debug)]
enum IoEvent {
    Frame {
        bytes: Vec<u8>,
        received_at: DateTime<Utc>,
    },
    Failed(StrategyError),
}

struct SessionWorkers(Vec<JoinHandle<()>>);

impl Drop for SessionWorkers {
    fn drop(&mut self) {
        for handle in &self.0 {
            handle.abort();
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PolymarketChainlinkBtcusdTwapFactory;

impl StrategyFactory for PolymarketChainlinkBtcusdTwapFactory {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }
    fn config_schema_version(&self) -> i32 {
        CONFIG_SCHEMA_VERSION
    }
    fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError> {
        PolymarketChainlinkBtcusdTwapConfig::from_value(config).map(|_| ())
    }
    fn build(
        &self,
        profile: &IngesterProfile,
        pool: PgPool,
    ) -> Result<Box<dyn IngesterStrategy>, StrategyFactoryError> {
        let config = PolymarketChainlinkBtcusdTwapConfig::from_value(&profile.config)?;
        if profile.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(StrategyFactoryError::Construction(format!(
                "Polymarket Chainlink TWAP checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}"
            )));
        }
        let checkpoint: TwapCheckpoint = serde_json::from_value(profile.checkpoint.clone())
            .map_err(|error| {
                StrategyFactoryError::Construction(format!(
                    "invalid Polymarket Chainlink TWAP checkpoint: {error}"
                ))
            })?;
        validate_checkpoint(&checkpoint)?;
        let lease_owner = profile.lease_owner.clone().ok_or_else(|| {
            StrategyFactoryError::Construction("profile is missing its claimed lease owner".into())
        })?;
        let lease_token = profile.lease_token.ok_or_else(|| {
            StrategyFactoryError::Construction("profile is missing its claimed lease token".into())
        })?;
        let config_snapshot = serde_json::to_value(&config)
            .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?;
        Ok(Box::new(PolymarketChainlinkBtcusdTwapStrategy {
            config,
            config_snapshot,
            profile_generation: profile.desired_generation,
            lease_owner,
            lease_token,
            checkpoint,
            pool: pool.clone(),
            artifacts: ArtifactRepository::new(pool.clone()),
            gaps: GapRepository::new(pool.clone()),
            profiles: ProfileRepository::new(pool),
        }))
    }
}

struct PolymarketChainlinkBtcusdTwapStrategy {
    config: PolymarketChainlinkBtcusdTwapConfig,
    config_snapshot: Value,
    profile_generation: i64,
    lease_owner: String,
    lease_token: Uuid,
    checkpoint: TwapCheckpoint,
    pool: PgPool,
    artifacts: ArtifactRepository,
    gaps: GapRepository,
    profiles: ProfileRepository,
}

#[async_trait]
impl IngesterStrategy for PolymarketChainlinkBtcusdTwapStrategy {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    async fn run(&self, shutdown: CancellationToken) -> Result<(), StrategyError> {
        let mut checkpoint = self.checkpoint.clone();
        let mut backoff_ms = self.config.reconnect_initial_ms;
        loop {
            if shutdown.is_cancelled() {
                self.seal_open_artifact().await?;
                return Ok(());
            }
            let session_started = Utc::now();
            match self.capture_session(&mut checkpoint, &shutdown).await {
                Ok(()) => {
                    self.seal_open_artifact().await?;
                    return Ok(());
                }
                Err(error) if error.kind == StrategyErrorKind::LeaseLost => {
                    let _ = self.seal_open_artifact().await;
                    shutdown.cancelled().await;
                    return Ok(());
                }
                Err(error) => {
                    if checkpoint.thirty_source_timestamp_ms.is_some()
                        || checkpoint.sixty_source_timestamp_ms.is_some()
                    {
                        self.record_transport_gap(&checkpoint, session_started, Utc::now(), &error)
                            .await?;
                    }
                    let delay = tokio::time::sleep(Duration::from_millis(backoff_ms));
                    tokio::pin!(delay);
                    tokio::select! {
                        _ = shutdown.cancelled() => {
                            self.seal_open_artifact().await?;
                            return Ok(());
                        }
                        _ = &mut delay => {}
                    }
                    backoff_ms = backoff_ms
                        .saturating_mul(2)
                        .min(self.config.reconnect_max_ms);
                }
            }
        }
    }
}

impl PolymarketChainlinkBtcusdTwapStrategy {
    async fn capture_session(
        &self,
        checkpoint: &mut TwapCheckpoint,
        shutdown: &CancellationToken,
    ) -> Result<(), StrategyError> {
        let websocket_config = WebSocketConfig::default()
            .read_buffer_size(32 * 1024)
            .write_buffer_size(8 * 1024)
            .max_write_buffer_size(32 * 1024)
            .max_message_size(Some(MAX_FRAME_BYTES))
            .max_frame_size(Some(MAX_FRAME_BYTES));
        let websocket = tokio::time::timeout(
            Duration::from_millis(self.config.connect_timeout_ms),
            connect_async_with_config(&self.config.websocket_url, Some(websocket_config), true),
        )
        .await
        .map_err(|_| {
            source(
                "polymarket_rtds_connect_timeout",
                "timed out connecting to Polymarket RTDS",
            )
        })?
        .map_err(|error| source("polymarket_rtds_connect_failed", error.to_string()))?
        .0;
        let (mut sink, mut stream) = websocket.split();
        let subscription = json!({
            "action": "subscribe",
            "subscriptions": [
                {"topic": TOPIC_THIRTY, "type": "update", "filters": "{\"symbol\":\"btc/usd\"}"},
                {"topic": TOPIC_SIXTY, "type": "update", "filters": "{\"symbol\":\"btc/usd\"}"}
            ]
        })
        .to_string();
        send_message(
            &mut sink,
            Message::Text(subscription.into()),
            self.config.write_timeout_ms,
        )
        .await?;

        let (events_tx, mut events_rx) = mpsc::channel(EVENT_BUFFER);
        let (commands_tx, mut commands_rx) = mpsc::channel(COMMAND_BUFFER);
        let writer_events = events_tx.clone();
        let ping_every = Duration::from_millis(self.config.ping_interval_ms);
        let write_timeout = self.config.write_timeout_ms;
        let writer = tokio::spawn(async move {
            let mut ticker = tokio::time::interval_at(Instant::now() + ping_every, ping_every);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                let message = tokio::select! {
                    _ = ticker.tick() => Message::Text("PING".into()),
                    command = commands_rx.recv() => match command { Some(command) => command, None => return },
                };
                if let Err(error) = send_message(&mut sink, message, write_timeout).await {
                    let _ = writer_events.try_send(IoEvent::Failed(error));
                    return;
                }
            }
        });
        let reader_events = events_tx;
        let reader = tokio::spawn(async move {
            while let Some(frame) = stream.next().await {
                let received_at = Utc::now();
                let event = match frame {
                    Ok(Message::Text(text)) => {
                        let value = text.as_str().trim();
                        if value.is_empty() || value.eq_ignore_ascii_case("PONG") {
                            continue;
                        }
                        if value.eq_ignore_ascii_case("PING") {
                            if commands_tx.try_send(Message::Text("PONG".into())).is_err() {
                                IoEvent::Failed(source(
                                    "polymarket_rtds_command_backpressure",
                                    "RTDS writer command buffer is unavailable",
                                ))
                            } else {
                                continue;
                            }
                        } else {
                            IoEvent::Frame {
                                bytes: text.as_bytes().to_vec(),
                                received_at,
                            }
                        }
                    }
                    Ok(Message::Binary(bytes)) => IoEvent::Frame {
                        bytes: bytes.to_vec(),
                        received_at,
                    },
                    Ok(Message::Ping(payload)) => {
                        if commands_tx.try_send(Message::Pong(payload)).is_err() {
                            IoEvent::Failed(source(
                                "polymarket_rtds_command_backpressure",
                                "RTDS writer command buffer is unavailable",
                            ))
                        } else {
                            continue;
                        }
                    }
                    Ok(Message::Pong(_)) => continue,
                    Ok(Message::Close(frame)) => IoEvent::Failed(source(
                        "polymarket_rtds_closed",
                        format!("Polymarket RTDS websocket closed: {frame:?}"),
                    )),
                    Ok(_) => continue,
                    Err(error) => {
                        IoEvent::Failed(source("polymarket_rtds_read_failed", error.to_string()))
                    }
                };
                if reader_events.try_send(event).is_err() {
                    let _ = reader_events.try_send(IoEvent::Failed(source(
                        "polymarket_rtds_consumer_backpressure",
                        "RTDS persistence fell behind its bounded frame buffer",
                    )));
                    return;
                }
            }
            let _ = reader_events.try_send(IoEvent::Failed(source(
                "polymarket_rtds_eof",
                "Polymarket RTDS websocket ended",
            )));
        });
        let _workers = SessionWorkers(vec![reader, writer]);

        let connected_at = Instant::now();
        let mut thirty_seen_at = checkpoint
            .thirty_source_timestamp_ms
            .map(|_| Instant::now());
        let mut sixty_seen_at = checkpoint.sixty_source_timestamp_ms.map(|_| Instant::now());
        let mut freshness_tick = tokio::time::interval(Duration::from_secs(1));
        freshness_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => return Ok(()),
                _ = freshness_tick.tick() => {
                    let now = Instant::now();
                    let initial = Duration::from_millis(self.config.initial_stream_timeout_ms);
                    let stale = Duration::from_millis(self.config.stream_stale_timeout_ms);
                    if thirty_seen_at.map_or(now.duration_since(connected_at) > initial, |last| now.duration_since(last) > stale) {
                        return Err(source("polymarket_rtds_thirty_stale", "30-second TWAP stream did not produce a fresh update"));
                    }
                    if sixty_seen_at.map_or(now.duration_since(connected_at) > initial, |last| now.duration_since(last) > stale) {
                        return Err(source("polymarket_rtds_sixty_stale", "60-second TWAP stream did not produce a fresh update"));
                    }
                }
                event = events_rx.recv() => match event {
                    Some(IoEvent::Frame { bytes, received_at }) => {
                        let observation = decode_observation(&bytes, received_at)?;
                        self.persist_observation(checkpoint, &observation).await?;
                        if observation.window_seconds == 30 { thirty_seen_at = Some(Instant::now()); }
                        else { sixty_seen_at = Some(Instant::now()); }
                    }
                    Some(IoEvent::Failed(error)) => return Err(error),
                    None => return Err(source("polymarket_rtds_workers_stopped", "RTDS websocket workers stopped")),
                }
            }
        }
    }

    async fn persist_observation(
        &self,
        checkpoint: &mut TwapCheckpoint,
        observation: &TwapObservation,
    ) -> Result<(), StrategyError> {
        let artifact = self
            .ensure_artifact(observation.received_at, checkpoint)
            .await?;
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(db("polymarket_twap_begin"))?;
        let inserted = sqlx::query_scalar::<_, DateTime<Utc>>(
            r#"
          INSERT INTO market_data.polymarket_chainlink_btcusd_twap (
            source_timestamp, published_at, received_at, symbol, window_seconds,
            twap_price, full_accuracy_value, source_payload, payload_sha256,
            capture_artifact_id
          ) VALUES ($1,$2,$3,'btc/usd',$4,$5,$6,$7,$8,$9)
          ON CONFLICT (source_timestamp, symbol, window_seconds) DO NOTHING
          RETURNING source_timestamp
        "#,
        )
        .bind(observation.source_timestamp)
        .bind(observation.published_at)
        .bind(observation.received_at)
        .bind(observation.window_seconds)
        .bind(observation.price)
        .bind(&observation.full_accuracy_value)
        .bind(&observation.source_payload)
        .bind(&observation.payload_sha256)
        .bind(artifact.artifact_id)
        .fetch_optional(&mut *transaction)
        .await
        .map_err(db("polymarket_twap_insert"))?;
        if inserted.is_none() {
            let stored = sqlx::query_as::<_, StoredTwap>(
                r#"
              SELECT published_at, twap_price, full_accuracy_value, payload_sha256
              FROM market_data.polymarket_chainlink_btcusd_twap
              WHERE source_timestamp=$1 AND symbol='btc/usd' AND window_seconds=$2
            "#,
            )
            .bind(observation.source_timestamp)
            .bind(observation.window_seconds)
            .fetch_optional(&mut *transaction)
            .await
            .map_err(db("polymarket_twap_verify"))?
            .ok_or_else(|| {
                integrity(
                    "polymarket_twap_conflict_missing",
                    "conflicting TWAP fact was not visible",
                )
            })?;
            if stored.published_at != observation.published_at
                || stored.twap_price != observation.price
                || stored.full_accuracy_value != observation.full_accuracy_value
                || stored.payload_sha256 != observation.payload_sha256
            {
                return Err(integrity(
                    "polymarket_twap_immutable_conflict",
                    "RTDS replay changed an immutable TWAP fact",
                ));
            }
        } else {
            self.artifacts
                .record_batch_in(
                    &mut transaction,
                    artifact.artifact_id,
                    &ArtifactBatch {
                        inserted_record_count: 1,
                        minimum_source_timestamp: Some(observation.source_timestamp),
                        maximum_source_timestamp: Some(observation.source_timestamp),
                        minimum_received_at: Some(observation.received_at),
                        maximum_received_at: Some(observation.received_at),
                        start_cursor: Some(cursor(observation)),
                        end_cursor: Some(cursor(observation)),
                    },
                )
                .await
                .map_err(db("polymarket_twap_artifact_batch"))?
                .ok_or_else(|| {
                    integrity(
                        "polymarket_twap_artifact_closed",
                        "TWAP capture artifact is not open",
                    )
                })?;
        }
        let mut next = checkpoint.clone();
        if observation.window_seconds == 30 {
            next.thirty_source_timestamp_ms = Some(observation.source_timestamp.timestamp_millis());
        } else {
            next.sixty_source_timestamp_ms = Some(observation.source_timestamp.timestamp_millis());
        }
        if next.thirty_source_timestamp_ms.is_some() && next.sixty_source_timestamp_ms.is_some() {
            let source_watermark_ms = next
                .thirty_source_timestamp_ms
                .unwrap()
                .min(next.sixty_source_timestamp_ms.unwrap());
            let source_watermark =
                timestamp_ms(source_watermark_ms, "polymarket_twap_checkpoint_time")?;
            let advanced = self
                .profiles
                .record_progress_in(
                    &mut transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                    &StrategyProgress {
                        verified_record_count: 1,
                        checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
                        checkpoint: serde_json::to_value(&next)
                            .map_err(integrity_err("polymarket_twap_checkpoint_encode"))?,
                        last_source_event_at: Some(observation.source_timestamp),
                        last_provider_available_at: Some(observation.published_at),
                        source_watermark: Some(source_watermark),
                        availability_watermark: Some(observation.received_at),
                    },
                )
                .await
                .map_err(db("polymarket_twap_progress"))?;
            if !advanced {
                return Err(lease_lost("committing TWAP progress"));
            }
        } else {
            self.lock_lease(&mut transaction, "persisting the first TWAP dimension")
                .await?;
        }
        transaction
            .commit()
            .await
            .map_err(db("polymarket_twap_commit"))?;
        *checkpoint = next;
        Ok(())
    }

    async fn ensure_artifact(
        &self,
        received_at: DateTime<Utc>,
        checkpoint: &TwapCheckpoint,
    ) -> Result<CaptureArtifact, StrategyError> {
        if let Some(open) = self
            .artifacts
            .get_open(STRATEGY_KEY)
            .await
            .map_err(db("polymarket_twap_load_artifact"))?
        {
            if open.profile_generation > self.profile_generation {
                return Err(lease_lost("opening a TWAP artifact"));
            }
            if open.profile_generation == self.profile_generation
                && open.config_schema_version == CONFIG_SCHEMA_VERSION
                && open.config_snapshot == self.config_snapshot
                && received_at < open.capture_window_end
            {
                return Ok(open);
            }
            self.seal_artifact(&open, false).await?;
        }
        let seconds = received_at.timestamp();
        let start_seconds = seconds - seconds.rem_euclid(self.config.artifact_window_seconds);
        let window_start = Utc
            .timestamp_opt(start_seconds, 0)
            .single()
            .ok_or_else(|| integrity("polymarket_twap_artifact_time", "invalid artifact window"))?;
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(db("polymarket_twap_artifact_begin"))?;
        self.lock_lease(&mut transaction, "creating a TWAP artifact")
            .await?;
        let artifact = self
            .artifacts
            .create_in(
                &mut transaction,
                &NewCaptureArtifact {
                    strategy_key: STRATEGY_KEY,
                    profile_generation: self.profile_generation,
                    config_schema_version: CONFIG_SCHEMA_VERSION,
                    config_snapshot: self.config_snapshot.clone(),
                    capture_window_start: window_start,
                    capture_window_end: window_start
                        + chrono::Duration::seconds(self.config.artifact_window_seconds),
                    start_cursor: checkpoint_cursor(checkpoint),
                },
            )
            .await
            .map_err(db("polymarket_twap_artifact_create"))?;
        transaction
            .commit()
            .await
            .map_err(db("polymarket_twap_artifact_commit"))?;
        Ok(artifact)
    }

    async fn seal_open_artifact(&self) -> Result<(), StrategyError> {
        if let Some(open) = self
            .artifacts
            .get_open(STRATEGY_KEY)
            .await
            .map_err(db("polymarket_twap_load_artifact_seal"))?
        {
            if open.profile_generation > self.profile_generation {
                return Err(lease_lost("sealing a TWAP artifact"));
            }
            self.seal_artifact(&open, true).await
        } else {
            self.verify_owned_lease().await
        }
    }

    async fn seal_artifact(
        &self,
        artifact: &CaptureArtifact,
        draining: bool,
    ) -> Result<(), StrategyError> {
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(db("polymarket_twap_seal_begin"))?;
        if draining {
            self.lock_owned_lease(&mut transaction, "sealing a TWAP artifact")
                .await?;
        } else {
            self.lock_lease(&mut transaction, "sealing a TWAP artifact")
                .await?;
        }
        let hashes = sqlx::query_scalar::<_, String>(
            r#"
          SELECT payload_sha256 FROM market_data.polymarket_chainlink_btcusd_twap
          WHERE capture_artifact_id=$1 ORDER BY source_timestamp, window_seconds
        "#,
        )
        .bind(artifact.artifact_id)
        .fetch_all(&mut *transaction)
        .await
        .map_err(db("polymarket_twap_hash_artifact"))?;
        if i64::try_from(hashes.len()).ok() != Some(artifact.record_count) {
            return Err(integrity(
                "polymarket_twap_artifact_count",
                "TWAP artifact record count does not match durable facts",
            ));
        }
        let mut digest = Sha256::new();
        for hash in hashes {
            digest.update(hash.as_bytes());
        }
        self.artifacts
            .complete_in(
                &mut transaction,
                artifact.artifact_id,
                &hex_digest(digest.finalize()),
                artifact.end_cursor.as_deref(),
            )
            .await
            .map_err(db("polymarket_twap_artifact_complete"))?
            .ok_or_else(|| {
                integrity(
                    "polymarket_twap_artifact_not_open",
                    "TWAP artifact could not be completed",
                )
            })?;
        transaction
            .commit()
            .await
            .map_err(db("polymarket_twap_seal_commit"))
    }

    async fn record_transport_gap(
        &self,
        checkpoint: &TwapCheckpoint,
        started: DateTime<Utc>,
        ended: DateTime<Utc>,
        error: &StrategyError,
    ) -> Result<(), StrategyError> {
        let artifact_id = self
            .artifacts
            .get_open(STRATEGY_KEY)
            .await
            .map_err(db("polymarket_twap_gap_artifact"))?
            .map(|artifact| artifact.artifact_id);
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(db("polymarket_twap_gap_begin"))?;
        self.lock_lease(&mut transaction, "recording a TWAP transport gap")
            .await?;
        let detection = self
            .gaps
            .detect_in(
                &mut transaction,
                &NewDataGap {
                    strategy_key: STRATEGY_KEY,
                    detected_artifact_id: artifact_id,
                    gap_kind: "transport_discontinuity".into(),
                    reason_code: error.code.into(),
                    reason_message: Some(error.message.clone()),
                    source_time_start: Some(started),
                    source_time_end: Some(ended),
                    start_cursor: checkpoint_cursor(checkpoint),
                    end_cursor: Some(ended.timestamp_millis().to_string()),
                },
            )
            .await
            .map_err(integrity_err("polymarket_twap_gap_detect"))?;
        self.gaps
            .mark_unrecoverable_in(
                &mut transaction,
                detection.gap.gap_id,
                "provider_has_no_replay",
                Some("Polymarket RTDS provides no history or replay endpoint"),
            )
            .await
            .map_err(db("polymarket_twap_gap_unrecoverable"))?;
        let marked = self
            .profiles
            .mark_degraded_in(
                &mut transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &StrategyDegradation {
                    reason_code: "polymarket_twap_transport_gap".into(),
                    reason_message: error.message.clone(),
                },
            )
            .await
            .map_err(db("polymarket_twap_degraded"))?;
        if !marked {
            return Err(lease_lost("marking a TWAP transport gap"));
        }
        transaction
            .commit()
            .await
            .map_err(db("polymarket_twap_gap_commit"))
    }

    async fn lock_lease(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        action: &str,
    ) -> Result<(), StrategyError> {
        let locked = self
            .profiles
            .lock_current_lease_in(
                transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
            )
            .await
            .map_err(db("polymarket_twap_lock_lease"))?;
        if locked {
            Ok(())
        } else {
            Err(lease_lost(action))
        }
    }
    async fn lock_owned_lease(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        action: &str,
    ) -> Result<(), StrategyError> {
        let locked = self
            .profiles
            .lock_owned_lease_in(
                transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
            )
            .await
            .map_err(db("polymarket_twap_lock_owned_lease"))?;
        if locked {
            Ok(())
        } else {
            Err(lease_lost(action))
        }
    }
    async fn verify_owned_lease(&self) -> Result<(), StrategyError> {
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(db("polymarket_twap_owned_begin"))?;
        self.lock_owned_lease(&mut transaction, "draining TWAP capture")
            .await?;
        transaction
            .commit()
            .await
            .map_err(db("polymarket_twap_owned_commit"))
    }
}

fn decode_observation(
    bytes: &[u8],
    received_at: DateTime<Utc>,
) -> Result<TwapObservation, StrategyError> {
    if bytes.len() > MAX_FRAME_BYTES {
        return Err(integrity(
            "polymarket_twap_frame_too_large",
            "RTDS frame exceeded 64 KiB",
        ));
    }
    let envelope: RtdsEnvelope =
        serde_json::from_slice(bytes).map_err(integrity_err("polymarket_twap_decode"))?;
    let expected_window = match envelope.topic.as_str() {
        TOPIC_THIRTY => 30,
        TOPIC_SIXTY => 60,
        _ => {
            return Err(integrity(
                "polymarket_twap_topic",
                "RTDS sent an unsubscribed topic",
            ))
        }
    };
    if envelope.message_type != "update"
        || envelope.payload.symbol != SYMBOL
        || envelope.payload.window_s != expected_window
    {
        return Err(integrity(
            "polymarket_twap_identity",
            "RTDS TWAP message identity did not match its subscription",
        ));
    }
    if !envelope.payload.value.is_number() {
        return Err(integrity(
            "polymarket_twap_display_value",
            "RTDS display value was not numeric",
        ));
    }
    let raw = &envelope.payload.full_accuracy_value;
    let digits = raw.strip_prefix('-').unwrap_or(raw);
    if digits.is_empty() || digits.len() > 29 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(integrity(
            "polymarket_twap_exact_value",
            "RTDS full-accuracy value was not a bounded signed integer",
        ));
    }
    let integer = i128::from_str(raw).map_err(integrity_err("polymarket_twap_exact_value"))?;
    let price = Decimal::from_i128_with_scale(integer, 18);
    if price <= Decimal::ZERO {
        return Err(integrity(
            "polymarket_twap_nonpositive",
            "RTDS TWAP price must be positive",
        ));
    }
    let source_timestamp = timestamp_ms(envelope.payload.timestamp, "polymarket_twap_source_time")?;
    let published_at = timestamp_ms(envelope.timestamp, "polymarket_twap_publish_time")?;
    if published_at < source_timestamp
        || published_at > received_at + chrono::Duration::milliseconds(MAX_CLOCK_LEAD_MS)
    {
        return Err(integrity(
            "polymarket_twap_time_order",
            "RTDS timestamps are not causally ordered",
        ));
    }
    let source_payload =
        serde_json::from_slice::<Value>(bytes).map_err(integrity_err("polymarket_twap_payload"))?;
    if source_payload.to_string().len() > 4_096 {
        return Err(integrity(
            "polymarket_twap_payload_too_large",
            "RTDS TWAP payload exceeded 4 KiB",
        ));
    }
    let canonical = serde_json::to_vec(&json!({"topic": envelope.topic, "type": envelope.message_type, "timestamp": envelope.timestamp, "payload": envelope.payload}))
        .map_err(integrity_err("polymarket_twap_hash_encode"))?;
    Ok(TwapObservation {
        source_timestamp,
        published_at,
        received_at,
        window_seconds: expected_window,
        price,
        full_accuracy_value: raw.clone(),
        source_payload,
        payload_sha256: hex_digest(Sha256::digest(canonical)),
    })
}

fn validate_checkpoint(checkpoint: &TwapCheckpoint) -> Result<(), StrategyFactoryError> {
    let now = Utc::now()
        .timestamp_millis()
        .saturating_add(MAX_CLOCK_LEAD_MS);
    if checkpoint
        .thirty_source_timestamp_ms
        .is_some_and(|value| value <= 0 || value > now)
        || checkpoint
            .sixty_source_timestamp_ms
            .is_some_and(|value| value <= 0 || value > now)
    {
        return Err(StrategyFactoryError::Construction(
            "Polymarket Chainlink TWAP checkpoint contains an invalid timestamp".into(),
        ));
    }
    Ok(())
}

async fn send_message<S>(
    sink: &mut S,
    message: Message,
    timeout_ms: u64,
) -> Result<(), StrategyError>
where
    S: futures_util::Sink<Message> + Unpin,
    S::Error: std::fmt::Display,
{
    tokio::time::timeout(Duration::from_millis(timeout_ms), sink.send(message))
        .await
        .map_err(|_| {
            source(
                "polymarket_rtds_write_timeout",
                "timed out writing to Polymarket RTDS",
            )
        })?
        .map_err(|error| source("polymarket_rtds_write_failed", error.to_string()))
}

fn timestamp_ms(value: i64, code: &'static str) -> Result<DateTime<Utc>, StrategyError> {
    Utc.timestamp_millis_opt(value)
        .single()
        .ok_or_else(|| integrity(code, "timestamp is outside the supported range"))
}
fn cursor(observation: &TwapObservation) -> String {
    format!(
        "{}:{}",
        observation.window_seconds,
        observation.source_timestamp.timestamp_millis()
    )
}
fn checkpoint_cursor(checkpoint: &TwapCheckpoint) -> Option<String> {
    match (
        checkpoint.thirty_source_timestamp_ms,
        checkpoint.sixty_source_timestamp_ms,
    ) {
        (None, None) => None,
        (thirty, sixty) => Some(format!(
            "30:{};60:{}",
            thirty.map_or_else(|| "none".into(), |v| v.to_string()),
            sixty.map_or_else(|| "none".into(), |v| v.to_string())
        )),
    }
}
fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    let mut encoded = String::with_capacity(bytes.as_ref().len() * 2);
    for byte in bytes.as_ref() {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}
fn source(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::TransientSource, code, message)
}
fn db<E: std::fmt::Display>(code: &'static str) -> impl FnOnce(E) -> StrategyError {
    move |error| {
        StrategyError::new(
            StrategyErrorKind::TransientDatabase,
            code,
            error.to_string(),
        )
    }
}
fn integrity_err<E: std::fmt::Display>(code: &'static str) -> impl FnOnce(E) -> StrategyError {
    move |error| StrategyError::new(StrategyErrorKind::Integrity, code, error.to_string())
}
fn integrity(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::Integrity, code, message)
}
fn lease_lost(action: &str) -> StrategyError {
    StrategyError::new(
        StrategyErrorKind::LeaseLost,
        "polymarket_twap_lease_lost",
        format!("profile lease was lost while {action}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_each_exact_twap_window() {
        for (topic, window) in [(TOPIC_THIRTY, 30), (TOPIC_SIXTY, 60)] {
            let bytes = serde_json::to_vec(&json!({
                "topic": topic, "type": "update", "timestamp": 1_785_178_800_123_i64,
                "payload": {"symbol": "btc/usd", "value": 65000.5,
                    "full_accuracy_value": "65000500000000000000000",
                    "timestamp": 1_785_178_800_000_i64, "window_s": window}
            }))
            .unwrap();
            let received = Utc.timestamp_millis_opt(1_785_178_800_500).unwrap();
            let decoded = decode_observation(&bytes, received).unwrap();
            assert_eq!(decoded.window_seconds, window);
            assert_eq!(decoded.price, Decimal::from_str("65000.5").unwrap());
        }
    }

    #[test]
    fn rejects_topic_window_mismatch() {
        let bytes = serde_json::to_vec(&json!({
            "topic": TOPIC_THIRTY, "type": "update", "timestamp": 1_785_178_800_123_i64,
            "payload": {"symbol": "btc/usd", "value": 65000.5,
                "full_accuracy_value": "65000500000000000000000",
                "timestamp": 1_785_178_800_000_i64, "window_s": 60}
        }))
        .unwrap();
        assert!(
            decode_observation(&bytes, Utc.timestamp_millis_opt(1_785_178_800_500).unwrap())
                .is_err()
        );
    }
}
