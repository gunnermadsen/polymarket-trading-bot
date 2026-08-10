//! Continuous Chainlink Data Streams BTC/USD reference-price ingestion.
//!
//! Chainlink authenticates archive requests with HMAC. Decoding a v3 report
//! validates its envelope identity and timestamps, but does not verify the
//! report's DON signature quorum.

use std::{
    collections::{BTreeSet, HashSet},
    env,
    fmt::Write as _,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use chainlink_data_streams_report::report::{decode_full_report, v3::ReportDataV3, Report};
use chrono::{DateTime, TimeZone, Utc};
use hmac::{Hmac, Mac};
use reqwest::{Client, StatusCode, Url};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::{FromRow, PgPool, Postgres, QueryBuilder, Transaction};
use tokio::time::MissedTickBehavior;
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

pub const STRATEGY_KEY: IngesterStrategyKey = IngesterStrategyKey::ChainlinkBtcusdReferencePrice;
pub const CONFIG_SCHEMA_VERSION: i32 = 1;
pub const CHECKPOINT_SCHEMA_VERSION: i32 = 1;
pub const API_KEY_ENV: &str = "MARKET_DATA_INGESTER_CHAINLINK_DATA_STREAMS_API_KEY";
pub const API_SECRET_ENV: &str = "MARKET_DATA_INGESTER_CHAINLINK_DATA_STREAMS_API_SECRET";

const SOURCE: &str = "chainlink_data_streams";
const DEFAULT_REST_BASE_URL: &str = "https://api.dataengine.chain.link";
const BTCUSD_FEED_ID: &str = "0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8";
const MAX_HTTP_RESPONSE_BYTES: usize = 4 * 1_048_576;
const MAX_INSERT_ROWS: usize = 1_000;
const GAP_REPAIRS_PER_POLL: i64 = 8;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct ChainlinkBtcusdReferencePriceConfig {
    pub rest_base_url: String,
    pub feed_id: String,
    pub poll_interval_ms: u64,
    pub recent_window_seconds: u64,
    pub overlap_seconds: u64,
    pub page_limit: usize,
    pub max_pages_per_poll: usize,
    pub artifact_window_seconds: i64,
    pub request_timeout_seconds: u64,
    pub max_request_attempts: u8,
    pub retry_initial_delay_ms: u64,
    pub retry_max_delay_ms: u64,
}

impl Default for ChainlinkBtcusdReferencePriceConfig {
    fn default() -> Self {
        Self {
            rest_base_url: DEFAULT_REST_BASE_URL.to_owned(),
            feed_id: BTCUSD_FEED_ID.to_owned(),
            poll_interval_ms: 1_000,
            recent_window_seconds: 300,
            overlap_seconds: 5,
            page_limit: 100,
            max_pages_per_poll: 8,
            artifact_window_seconds: 3_600,
            request_timeout_seconds: 10,
            max_request_attempts: 3,
            retry_initial_delay_ms: 250,
            retry_max_delay_ms: 2_000,
        }
    }
}

impl ChainlinkBtcusdReferencePriceConfig {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let config = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            invalid_config(format!(
                "invalid Chainlink BTC/USD reference-price config: {error}"
            ))
        })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), StrategyFactoryError> {
        validate_https_origin(&self.rest_base_url)?;
        if self.feed_id != BTCUSD_FEED_ID {
            return Err(invalid_config(
                "feed_id must be the lowercase Chainlink BTC/USD v3 feed ID",
            ));
        }
        if !(250..=60_000).contains(&self.poll_interval_ms) {
            return Err(invalid_config(
                "poll_interval_ms must be between 250 and 60000",
            ));
        }
        if !(5..=86_400).contains(&self.recent_window_seconds) {
            return Err(invalid_config(
                "recent_window_seconds must be between 5 and 86400",
            ));
        }
        if self.overlap_seconds == 0 || self.overlap_seconds > self.recent_window_seconds {
            return Err(invalid_config(
                "overlap_seconds must be positive and no greater than recent_window_seconds",
            ));
        }
        if !(1..=100).contains(&self.page_limit) {
            return Err(invalid_config("page_limit must be between 1 and 100"));
        }
        if !(1..=64).contains(&self.max_pages_per_poll) {
            return Err(invalid_config(
                "max_pages_per_poll must be between 1 and 64",
            ));
        }
        if self.page_limit.saturating_mul(self.max_pages_per_poll) > 50_000 {
            return Err(invalid_config(
                "page_limit multiplied by max_pages_per_poll must not exceed 50000",
            ));
        }
        let safe_seconds = safe_pagination_second_budget(self.page_limit, self.max_pages_per_poll)
            .ok_or_else(|| {
                invalid_config(
                "pagination must reserve capacity for inclusive page overlap and boundary proof",
            )
            })?;
        let initial_window_seconds = self.recent_window_seconds.saturating_add(1);
        if safe_seconds < initial_window_seconds {
            return Err(invalid_config(format!(
                "pagination safely covers {safe_seconds} source seconds but recent_window_seconds requires at least {initial_window_seconds}"
            )));
        }
        let checkpoint_advance_seconds = self.overlap_seconds.saturating_add(2);
        if safe_seconds < checkpoint_advance_seconds {
            return Err(invalid_config(format!(
                "pagination safely covers {safe_seconds} source seconds but overlap_seconds requires at least {checkpoint_advance_seconds} to advance a checkpoint"
            )));
        }
        if !(60..=86_400).contains(&self.artifact_window_seconds) {
            return Err(invalid_config(
                "artifact_window_seconds must be between 60 and 86400",
            ));
        }
        if !(1..=30).contains(&self.request_timeout_seconds) {
            return Err(invalid_config(
                "request_timeout_seconds must be between 1 and 30",
            ));
        }
        if !(1..=5).contains(&self.max_request_attempts) {
            return Err(invalid_config(
                "max_request_attempts must be between 1 and 5",
            ));
        }
        if !(50..=5_000).contains(&self.retry_initial_delay_ms)
            || self.retry_max_delay_ms < self.retry_initial_delay_ms
            || self.retry_max_delay_ms > 30_000
        {
            return Err(invalid_config(
                "retry delays must be ordered between 50 and 30000 milliseconds",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
struct ReferencePriceCheckpoint {
    last_source_timestamp_seconds: Option<i64>,
}

impl ReferencePriceCheckpoint {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let checkpoint = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "invalid Chainlink reference-price checkpoint: {error}"
            ))
        })?;
        validate_checkpoint(&checkpoint)?;
        Ok(checkpoint)
    }
}

struct ChainlinkCredentials {
    api_key: String,
    api_secret: String,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ChainlinkBtcusdReferencePriceFactory;

impl StrategyFactory for ChainlinkBtcusdReferencePriceFactory {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    fn config_schema_version(&self) -> i32 {
        CONFIG_SCHEMA_VERSION
    }

    fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError> {
        ChainlinkBtcusdReferencePriceConfig::from_value(config).map(|_| ())
    }

    fn build(
        &self,
        profile: &IngesterProfile,
        pool: PgPool,
    ) -> Result<Box<dyn IngesterStrategy>, StrategyFactoryError> {
        if profile.strategy_key != STRATEGY_KEY {
            return Err(StrategyFactoryError::Construction(format!(
                "received profile for {}",
                profile.strategy_key
            )));
        }
        if profile.config_schema_version != CONFIG_SCHEMA_VERSION {
            return Err(StrategyFactoryError::Construction(format!(
                "Chainlink reference-price config schema must be {CONFIG_SCHEMA_VERSION}"
            )));
        }
        if profile.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(StrategyFactoryError::Construction(format!(
                "Chainlink reference-price checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}"
            )));
        }
        if profile.desired_generation <= 0 {
            return Err(StrategyFactoryError::Construction(
                "profile generation must be positive".to_owned(),
            ));
        }

        let config = ChainlinkBtcusdReferencePriceConfig::from_value(&profile.config)?;
        let config_snapshot = serde_json::to_value(&config).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "failed to encode effective Chainlink reference-price config: {error}"
            ))
        })?;
        let checkpoint = ReferencePriceCheckpoint::from_value(&profile.checkpoint)?;
        let credentials = credentials_from_environment()?;
        let lease_owner = profile.lease_owner.clone().ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease owner is missing".to_owned())
        })?;
        let lease_token = profile.lease_token.ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease token is missing".to_owned())
        })?;
        let client = build_chainlink_http_client(config.request_timeout_seconds)?;

        Ok(Box::new(ChainlinkBtcusdReferencePriceStrategy {
            config,
            config_snapshot,
            profile_generation: profile.desired_generation,
            lease_owner,
            lease_token,
            initial_checkpoint: checkpoint,
            credentials,
            client,
            pool: pool.clone(),
            artifacts: ArtifactRepository::new(pool.clone()),
            gaps: GapRepository::new(pool.clone()),
            profiles: ProfileRepository::new(pool),
        }))
    }
}

struct ChainlinkBtcusdReferencePriceStrategy {
    config: ChainlinkBtcusdReferencePriceConfig,
    config_snapshot: Value,
    profile_generation: i64,
    lease_owner: String,
    lease_token: Uuid,
    initial_checkpoint: ReferencePriceCheckpoint,
    credentials: ChainlinkCredentials,
    client: Client,
    pool: PgPool,
    artifacts: ArtifactRepository,
    gaps: GapRepository,
    profiles: ProfileRepository,
}

#[derive(Debug, Clone, PartialEq)]
struct ReferencePriceObservation {
    feed_id: String,
    source_timestamp: DateTime<Utc>,
    valid_from_timestamp: DateTime<Utc>,
    price: Decimal,
    bid: Decimal,
    ask: Decimal,
    provider_available_at: Option<DateTime<Utc>>,
    received_at: DateTime<Utc>,
    report_sha256: String,
    payload_sha256: String,
}

#[derive(Debug, FromRow)]
struct StoredReferencePrice {
    valid_from_timestamp: DateTime<Utc>,
    price: Decimal,
    bid: Decimal,
    ask: Decimal,
    provider_available_at: Option<DateTime<Utc>>,
    payload_sha256: String,
}

#[derive(Debug, FromRow)]
struct InsertedReferencePrice {
    source_timestamp: DateTime<Utc>,
    report_sha256: String,
}

#[async_trait]
impl IngesterStrategy for ChainlinkBtcusdReferencePriceStrategy {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    async fn run(&self, shutdown: CancellationToken) -> Result<(), StrategyError> {
        let mut checkpoint = self.initial_checkpoint.clone();
        let mut ticker = tokio::time::interval(Duration::from_millis(self.config.poll_interval_ms));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {
                    self.seal_open_artifact().await?;
                    return Ok(());
                }
                _ = ticker.tick() => {
                    match self.capture(&mut checkpoint).await {
                        Ok(()) => {}
                        Err(error) if error.kind == StrategyErrorKind::LeaseLost => {
                            if let Err(drain_error) = self.seal_open_artifact().await {
                                return Err(owned_drain_failure(error, drain_error));
                            }
                            shutdown.cancelled().await;
                            return Ok(());
                        }
                        Err(error) => return Err(error),
                    }
                }
            }
        }
    }
}

impl ChainlinkBtcusdReferencePriceStrategy {
    async fn capture(
        &self,
        checkpoint: &mut ReferencePriceCheckpoint,
    ) -> Result<(), StrategyError> {
        let observations = self.fetch_recent(checkpoint).await?;
        if observations.is_empty() {
            self.reconcile_gaps(checkpoint).await?;
            return Ok(());
        }

        let received_at = observations
            .iter()
            .map(|observation| observation.received_at)
            .max()
            .expect("nonempty observations have a receipt timestamp");
        let artifact = self.ensure_artifact(received_at, checkpoint).await?;
        let gaps = find_gaps(&observations, checkpoint.last_source_timestamp_seconds);
        let maximum_source_timestamp = observations
            .iter()
            .map(|observation| observation.source_timestamp)
            .max()
            .expect("nonempty observations have a maximum timestamp");

        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(database_error("chainlink_reference_begin_transaction"))?;
        let persisted = self
            .persist_observations(&mut transaction, artifact.artifact_id, &observations)
            .await?;

        for gap in gaps {
            self.gaps
                .detect_in(
                    &mut transaction,
                    &NewDataGap {
                        strategy_key: STRATEGY_KEY,
                        detected_artifact_id: Some(artifact.artifact_id),
                        gap_kind: "source_time_discontinuity".to_owned(),
                        reason_code: "chainlink_reference_second_missing".to_owned(),
                        reason_message: Some(
                            "Chainlink returned a non-contiguous one-second report series"
                                .to_owned(),
                        ),
                        source_time_start: Some(gap.start),
                        source_time_end: Some(gap.end),
                        start_cursor: Some(gap.start.timestamp().to_string()),
                        end_cursor: Some(gap.end.timestamp().to_string()),
                    },
                )
                .await
                .map_err(integrity_error("chainlink_reference_record_gap"))?;
            let marked = self
                .profiles
                .mark_degraded_in(
                    &mut transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                    &StrategyDegradation {
                        reason_code: "chainlink_reference_gap".to_owned(),
                        reason_message: "source returned a non-contiguous report series".to_owned(),
                    },
                )
                .await
                .map_err(database_error("chainlink_reference_mark_degraded"))?;
            if !marked {
                return Err(lease_lost("recording a Chainlink reference-price gap"));
            }
        }

        self.record_artifact_batch(&mut transaction, artifact.artifact_id, &persisted)
            .await?;

        let maximum_source_seconds = maximum_source_timestamp.timestamp();
        let next_checkpoint = ReferencePriceCheckpoint {
            last_source_timestamp_seconds: Some(
                checkpoint
                    .last_source_timestamp_seconds
                    .map_or(maximum_source_seconds, |current| {
                        current.max(maximum_source_seconds)
                    }),
            ),
        };
        let advanced = self
            .profiles
            .record_progress_in(
                &mut transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &StrategyProgress {
                    verified_record_count: persisted.verified,
                    checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
                    checkpoint: serde_json::to_value(&next_checkpoint)
                        .map_err(integrity_error("chainlink_reference_encode_checkpoint"))?,
                    last_source_event_at: Some(maximum_source_timestamp),
                    last_provider_available_at: None,
                    source_watermark: Some(maximum_source_timestamp),
                    availability_watermark: None,
                },
            )
            .await
            .map_err(database_error("chainlink_reference_record_progress"))?;
        if !advanced {
            return Err(lease_lost("committing Chainlink reference-price progress"));
        }
        transaction
            .commit()
            .await
            .map_err(database_error("chainlink_reference_commit_transaction"))?;
        *checkpoint = next_checkpoint;
        self.reconcile_gaps(checkpoint).await?;
        Ok(())
    }

    async fn fetch_recent(
        &self,
        checkpoint: &ReferencePriceCheckpoint,
    ) -> Result<Vec<ReferencePriceObservation>, StrategyError> {
        let range = plan_live_fetch_range(
            checkpoint.last_source_timestamp_seconds,
            Utc::now().timestamp(),
            &self.config,
        )?;
        self.fetch_range(range.start, range.end).await
    }

    async fn fetch_range(
        &self,
        requested_start: i64,
        requested_end: i64,
    ) -> Result<Vec<ReferencePriceObservation>, StrategyError> {
        if requested_start < 0 || requested_end < requested_start {
            return Err(integrity(
                "chainlink_reference_invalid_range",
                "requested Chainlink report range is invalid",
            ));
        }

        let mut page_start = requested_start;
        let mut observations = Vec::new();
        let mut seen_identities = BTreeSet::new();
        let mut pending_boundary = None;
        for _ in 0..self.config.max_pages_per_poll {
            let page = self.fetch_page(page_start).await?;
            match reconcile_reports_page(
                &mut observations,
                &mut seen_identities,
                page,
                page_start,
                requested_end,
                self.config.page_limit,
                pending_boundary.as_ref(),
            )? {
                PageDisposition::Complete => return Ok(observations),
                PageDisposition::Continue(boundary) => {
                    page_start = boundary.timestamp;
                    pending_boundary = Some(boundary);
                }
            }
        }
        Err(source(
            "chainlink_reference_pagination_budget_exhausted",
            "bounded Chainlink pagination ended before the final-second boundary was proven complete",
        ))
    }

    async fn fetch_page(
        &self,
        start_timestamp_seconds: i64,
    ) -> Result<Vec<ReferencePriceObservation>, StrategyError> {
        let mut attempt = 1_u8;
        loop {
            match self.fetch_page_once(start_timestamp_seconds).await {
                Ok(page) => return Ok(page),
                Err(error)
                    if error.kind == StrategyErrorKind::TransientSource
                        && attempt < self.config.max_request_attempts =>
                {
                    tokio::time::sleep(retry_delay(&self.config, attempt)).await;
                    attempt = attempt.saturating_add(1);
                }
                Err(error) => return Err(error),
            }
        }
    }

    async fn fetch_page_once(
        &self,
        start_timestamp_seconds: i64,
    ) -> Result<Vec<ReferencePriceObservation>, StrategyError> {
        let path = format!(
            "/api/v1/reports/page?feedID={}&startTimestamp={}&limit={}",
            self.config.feed_id, start_timestamp_seconds, self.config.page_limit
        );
        // Recheck the exact allowlisted origin at the request boundary, before
        // signing or attaching either credential header.
        let endpoint = credentialed_request_url(&self.config.rest_base_url, &path)?;
        let timestamp_ms = current_timestamp_millis()?;
        let signature = sign_request(&self.credentials, "GET", &path, timestamp_ms)?;
        let mut response = self
            .client
            .get(endpoint)
            .header("Authorization", self.credentials.api_key.trim())
            .header("X-Authorization-Timestamp", timestamp_ms.to_string())
            .header("X-Authorization-Signature-SHA256", signature)
            .send()
            .await
            .map_err(source_error("chainlink_reference_http_request"))?;
        let status = response.status();
        if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
            return Err(source(
                "chainlink_reference_http_retryable_status",
                format!("Chainlink reports endpoint returned {status}"),
            ));
        }
        if !status.is_success() {
            return Err(StrategyError::new(
                StrategyErrorKind::InvalidConfiguration,
                "chainlink_reference_http_rejected",
                format!("Chainlink reports endpoint returned {status}"),
            ));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_HTTP_RESPONSE_BYTES as u64)
        {
            return Err(integrity(
                "chainlink_reference_response_too_large",
                "Chainlink response exceeded the four-megabyte bound",
            ));
        }

        let mut body = Vec::with_capacity(
            response
                .content_length()
                .unwrap_or(0)
                .min(MAX_HTTP_RESPONSE_BYTES as u64) as usize,
        );
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(source_error("chainlink_reference_read_response"))?
        {
            if body.len().saturating_add(chunk.len()) > MAX_HTTP_RESPONSE_BYTES {
                return Err(integrity(
                    "chainlink_reference_response_too_large",
                    "Chainlink response exceeded the four-megabyte bound",
                ));
            }
            body.extend_from_slice(&chunk);
        }
        // Receipt time is meaningful only after the complete bounded body is durable in memory.
        let received_at = Utc::now();
        let payload: Value = serde_json::from_slice(&body)
            .map_err(integrity_error("chainlink_reference_decode_response"))?;
        let observations = decode_reports_page(payload, &self.config.feed_id, received_at)?;
        if observations.len() > self.config.page_limit {
            return Err(integrity(
                "chainlink_reference_response_limit_exceeded",
                "Chainlink returned more reports than requested",
            ));
        }
        if observations
            .iter()
            .any(|observation| observation.source_timestamp.timestamp() < start_timestamp_seconds)
        {
            return Err(integrity(
                "chainlink_reference_page_regressed",
                "Chainlink returned a report before the requested cursor",
            ));
        }
        Ok(observations)
    }
}

#[derive(Debug, Default)]
struct PersistedBatch {
    inserted: i64,
    verified: i64,
    minimum_inserted_source: Option<DateTime<Utc>>,
    maximum_inserted_source: Option<DateTime<Utc>>,
    minimum_inserted_received: Option<DateTime<Utc>>,
    maximum_inserted_received: Option<DateTime<Utc>>,
}

impl PersistedBatch {
    fn record_insert(&mut self, observation: &ReferencePriceObservation) {
        self.inserted += 1;
        self.minimum_inserted_source =
            minimum(self.minimum_inserted_source, observation.source_timestamp);
        self.maximum_inserted_source =
            maximum(self.maximum_inserted_source, observation.source_timestamp);
        self.minimum_inserted_received =
            minimum(self.minimum_inserted_received, observation.received_at);
        self.maximum_inserted_received =
            maximum(self.maximum_inserted_received, observation.received_at);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SourceGap {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourceSecondRange {
    start: i64,
    end: i64,
}

/// Each full inclusive page replays at least its final report on the next
/// page. Reserving one report per page for that overlap leaves this many
/// one-report source seconds, including room for a later timestamp to prove
/// the requested final second complete. Additional same-second identities
/// still fail closed through the runtime page budget.
fn safe_pagination_second_budget(page_limit: usize, max_pages: usize) -> Option<u64> {
    let new_seconds_per_page = page_limit.checked_sub(1)?;
    let seconds = new_seconds_per_page.checked_mul(max_pages)?;
    u64::try_from(seconds).ok().filter(|seconds| *seconds > 0)
}

fn plan_live_fetch_range(
    checkpoint_seconds: Option<i64>,
    request_end: i64,
    config: &ChainlinkBtcusdReferencePriceConfig,
) -> Result<SourceSecondRange, StrategyError> {
    let safe_seconds = safe_pagination_second_budget(config.page_limit, config.max_pages_per_poll)
        .ok_or_else(|| {
            integrity(
                "chainlink_reference_pagination_budget_invalid",
                "pagination cannot safely advance an inclusive source-time range",
            )
        })?;
    let recent_window = i64::try_from(config.recent_window_seconds).unwrap_or(i64::MAX);
    let recent_floor = request_end.saturating_sub(recent_window).max(0);
    let range = if let Some(checkpoint) = checkpoint_seconds {
        let overlap = i64::try_from(config.overlap_seconds).unwrap_or(i64::MAX);
        let safe_seconds = i64::try_from(safe_seconds).unwrap_or(i64::MAX);
        let forward_seconds = safe_seconds
            .checked_sub(overlap)
            .and_then(|remaining| remaining.checked_sub(1))
            .filter(|seconds| *seconds > 0)
            .ok_or_else(|| {
                integrity(
                    "chainlink_reference_pagination_budget_invalid",
                    "pagination cannot retain overlap while advancing the checkpoint",
                )
            })?;
        SourceSecondRange {
            start: checkpoint.saturating_sub(overlap).max(0),
            end: request_end.min(checkpoint.saturating_add(forward_seconds)),
        }
    } else {
        SourceSecondRange {
            start: recent_floor,
            end: request_end,
        }
    };
    if range.end < range.start {
        return Err(integrity(
            "chainlink_reference_live_range_invalid",
            "checkpoint overlap begins after the current source-time boundary",
        ));
    }
    Ok(range)
}

fn bounded_gap_repair_range(
    gap_start: i64,
    gap_end: i64,
    max_seconds: u64,
) -> Result<Option<SourceSecondRange>, StrategyError> {
    if gap_start < 0 || gap_end < gap_start || max_seconds == 0 {
        return Err(integrity(
            "chainlink_reference_gap_repair_range_invalid",
            "Chainlink gap repair range is invalid",
        ));
    }
    let seconds = gap_end
        .checked_sub(gap_start)
        .and_then(|difference| difference.checked_add(1))
        .ok_or_else(|| {
            integrity(
                "chainlink_reference_gap_range_overflow",
                "Chainlink gap range exceeded timestamp capacity",
            )
        })?;
    let seconds = u64::try_from(seconds).map_err(|_| {
        integrity(
            "chainlink_reference_gap_range_overflow",
            "Chainlink gap range exceeded timestamp capacity",
        )
    })?;
    Ok((seconds <= max_seconds).then_some(SourceSecondRange {
        start: gap_start,
        end: gap_end,
    }))
}

fn gap_second_count_is_complete(
    range: SourceSecondRange,
    distinct_source_seconds: i64,
) -> Result<bool, StrategyError> {
    let expected = range
        .end
        .checked_sub(range.start)
        .and_then(|difference| difference.checked_add(1))
        .ok_or_else(|| {
            integrity(
                "chainlink_reference_gap_range_overflow",
                "Chainlink gap range exceeded timestamp capacity",
            )
        })?;
    Ok(distinct_source_seconds == expected)
}

#[derive(Debug, Deserialize)]
struct ReportsPage {
    reports: Vec<Report>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingBoundary {
    timestamp: i64,
    report_sha256s: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PageDisposition {
    Complete,
    Continue(PendingBoundary),
}

/// Merges one inclusive reports page while proving that a full page did not
/// split distinct identities at its final second.
fn reconcile_reports_page(
    observations: &mut Vec<ReferencePriceObservation>,
    seen_identities: &mut BTreeSet<(i64, String)>,
    page: Vec<ReferencePriceObservation>,
    page_start: i64,
    requested_end: i64,
    page_limit: usize,
    pending_boundary: Option<&PendingBoundary>,
) -> Result<PageDisposition, StrategyError> {
    if page.len() > page_limit {
        return Err(integrity(
            "chainlink_reference_response_limit_exceeded",
            "Chainlink returned more reports than requested",
        ));
    }
    if page.is_empty() {
        return if pending_boundary.is_some() {
            Err(integrity(
                "chainlink_reference_boundary_overlap_missing",
                "Chainlink omitted the known final-second identities from an overlap page",
            ))
        } else {
            Ok(PageDisposition::Complete)
        };
    }

    let page_first = page
        .first()
        .expect("nonempty page has a first report")
        .source_timestamp
        .timestamp();
    let page_last = page
        .last()
        .expect("nonempty page has a final report")
        .source_timestamp
        .timestamp();
    if page_first < page_start {
        return Err(integrity(
            "chainlink_reference_page_regressed",
            "Chainlink report page began before its requested cursor",
        ));
    }

    if let Some(boundary) = pending_boundary {
        if boundary.timestamp != page_start {
            return Err(integrity(
                "chainlink_reference_boundary_cursor_mismatch",
                "Chainlink boundary proof did not match the inclusive request cursor",
            ));
        }
        let returned_boundary_identities = page
            .iter()
            .filter(|observation| observation.source_timestamp.timestamp() == boundary.timestamp)
            .map(|observation| observation.report_sha256.clone())
            .collect::<BTreeSet<_>>();
        if !boundary
            .report_sha256s
            .is_subset(&returned_boundary_identities)
        {
            return Err(integrity(
                "chainlink_reference_boundary_overlap_missing",
                "Chainlink overlap page omitted a previously observed final-second identity",
            ));
        }
    }

    let mut newly_observed = 0_usize;
    for observation in &page {
        let timestamp = observation.source_timestamp.timestamp();
        let identity = (timestamp, observation.report_sha256.clone());
        if seen_identities.insert(identity) {
            newly_observed += 1;
            if timestamp <= requested_end {
                observations.push(observation.clone());
            }
        }
    }

    if page.len() < page_limit || page_last > requested_end {
        return Ok(PageDisposition::Complete);
    }
    if pending_boundary.is_some_and(|boundary| boundary.timestamp == page_last)
        && newly_observed == 0
    {
        return Err(integrity(
            "chainlink_reference_boundary_saturated",
            "Chainlink repeated a full final-second page without proving the boundary complete",
        ));
    }

    let mut report_sha256s = page
        .iter()
        .filter(|observation| observation.source_timestamp.timestamp() == page_last)
        .map(|observation| observation.report_sha256.clone())
        .collect::<BTreeSet<_>>();
    if let Some(boundary) = pending_boundary.filter(|boundary| boundary.timestamp == page_last) {
        report_sha256s.extend(boundary.report_sha256s.iter().cloned());
    }
    Ok(PageDisposition::Continue(PendingBoundary {
        timestamp: page_last,
        report_sha256s,
    }))
}

fn decode_reports_page(
    payload: Value,
    expected_feed_id: &str,
    received_at: DateTime<Utc>,
) -> Result<Vec<ReferencePriceObservation>, StrategyError> {
    let page = serde_json::from_value::<ReportsPage>(payload)
        .map_err(integrity_error("chainlink_reference_invalid_page"))?;
    let mut observations = Vec::with_capacity(page.reports.len());
    let mut previous_timestamp = None;
    let mut identities = BTreeSet::new();
    for envelope in page.reports {
        let observation = decode_report(&envelope, expected_feed_id, received_at)?;
        let source_seconds = observation.source_timestamp.timestamp();
        if previous_timestamp.is_some_and(|previous| source_seconds < previous) {
            return Err(integrity(
                "chainlink_reference_nonmonotonic_page",
                "Chainlink report timestamps must be nondecreasing within a page",
            ));
        }
        if !identities.insert((source_seconds, observation.report_sha256.clone())) {
            return Err(integrity(
                "chainlink_reference_duplicate_report",
                "Chainlink page repeated the same report identity",
            ));
        }
        previous_timestamp = Some(source_seconds);
        observations.push(observation);
    }
    Ok(observations)
}

fn decode_report(
    envelope: &Report,
    expected_feed_id: &str,
    received_at: DateTime<Utc>,
) -> Result<ReferencePriceObservation, StrategyError> {
    let encoded_report = envelope
        .full_report
        .strip_prefix("0x")
        .unwrap_or(&envelope.full_report);
    let bytes = hex::decode(encoded_report).map_err(integrity_error(
        "chainlink_reference_full_report_invalid_hex",
    ))?;
    let (_, report_blob) = decode_full_report(&bytes)
        .map_err(integrity_error("chainlink_reference_full_report_decode"))?;
    let report = ReportDataV3::decode(&report_blob)
        .map_err(integrity_error("chainlink_reference_v3_decode"))?;

    let feed_id = report.feed_id.to_string().to_ascii_lowercase();
    let envelope_feed_id = envelope.feed_id.to_string().to_ascii_lowercase();
    if feed_id != expected_feed_id || envelope_feed_id != feed_id {
        return Err(integrity(
            "chainlink_reference_feed_mismatch",
            "Chainlink report feed ID did not match the configured BTC/USD feed",
        ));
    }
    let envelope_observation_timestamp =
        u32::try_from(envelope.observations_timestamp).map_err(|_| {
            integrity(
                "chainlink_reference_envelope_timestamp_overflow",
                "Chainlink envelope observation timestamp exceeded u32 capacity",
            )
        })?;
    let envelope_valid_from_timestamp =
        u32::try_from(envelope.valid_from_timestamp).map_err(|_| {
            integrity(
                "chainlink_reference_envelope_timestamp_overflow",
                "Chainlink envelope valid-from timestamp exceeded u32 capacity",
            )
        })?;
    if report.observations_timestamp != envelope_observation_timestamp
        || report.valid_from_timestamp != envelope_valid_from_timestamp
    {
        return Err(integrity(
            "chainlink_reference_envelope_timestamp_mismatch",
            "Chainlink envelope timestamps did not match its v3 report payload",
        ));
    }
    if report.valid_from_timestamp > report.observations_timestamp {
        return Err(integrity(
            "chainlink_reference_invalid_time_order",
            "Chainlink valid-from timestamp followed its observation timestamp",
        ));
    }

    let source_timestamp = timestamp_seconds(
        i64::from(report.observations_timestamp),
        "observation timestamp",
    )?;
    let valid_from_timestamp = timestamp_seconds(
        i64::from(report.valid_from_timestamp),
        "valid-from timestamp",
    )?;
    let price = scaled_decimal("benchmark price", &report.benchmark_price.to_string())?;
    let bid = scaled_decimal("bid", &report.bid.to_string())?;
    let ask = scaled_decimal("ask", &report.ask.to_string())?;
    if price <= Decimal::ZERO
        || bid <= Decimal::ZERO
        || ask <= Decimal::ZERO
        || bid > price
        || price > ask
    {
        return Err(integrity(
            "chainlink_reference_invalid_prices",
            "Chainlink report must have positive bid <= benchmark price <= ask",
        ));
    }

    let report_sha256 = hex_digest(Sha256::digest(&bytes));
    let payload_sha256 = canonical_payload_sha256(
        &feed_id,
        report.observations_timestamp,
        report.valid_from_timestamp,
        price,
        bid,
        ask,
        &report_sha256,
    );
    Ok(ReferencePriceObservation {
        feed_id,
        source_timestamp,
        valid_from_timestamp,
        price,
        bid,
        ask,
        // The reports-page envelope does not expose a publication timestamp.
        provider_available_at: None,
        received_at,
        report_sha256,
        payload_sha256,
    })
}

fn canonical_payload_sha256(
    feed_id: &str,
    source_timestamp_seconds: u32,
    valid_from_timestamp_seconds: u32,
    price: Decimal,
    bid: Decimal,
    ask: Decimal,
    report_sha256: &str,
) -> String {
    let canonical = json!({
        "ask": ask.normalize().to_string(),
        "bid": bid.normalize().to_string(),
        "feed_id": feed_id,
        "price": price.normalize().to_string(),
        "report_sha256": report_sha256,
        "source_timestamp_seconds": source_timestamp_seconds,
        "valid_from_timestamp_seconds": valid_from_timestamp_seconds,
    });
    let encoded = serde_json::to_vec(&canonical).expect("canonical report values serialize");
    hex_digest(Sha256::digest(encoded))
}

fn scaled_decimal(name: &str, unscaled: &str) -> Result<Decimal, StrategyError> {
    let value = unscaled.parse::<i128>().map_err(|error| {
        integrity(
            "chainlink_reference_decimal_overflow",
            format!("Chainlink {name} exceeded decimal capacity: {error}"),
        )
    })?;
    Ok(Decimal::from_i128_with_scale(value, 18))
}

fn find_gaps(
    observations: &[ReferencePriceObservation],
    checkpoint_seconds: Option<i64>,
) -> Vec<SourceGap> {
    let timestamps = observations
        .iter()
        .map(|observation| observation.source_timestamp.timestamp())
        .collect::<BTreeSet<_>>();
    let mut gaps = BTreeSet::new();
    if let Some(checkpoint) = checkpoint_seconds {
        if let Some(first_new) = timestamps
            .iter()
            .copied()
            .find(|timestamp| *timestamp > checkpoint)
        {
            let expected = checkpoint.saturating_add(1);
            if first_new > expected {
                if let (Ok(start), Ok(end)) = (
                    timestamp_seconds(expected, "gap start"),
                    timestamp_seconds(first_new.saturating_sub(1), "gap end"),
                ) {
                    gaps.insert(SourceGap { start, end });
                }
            }
        }
    }
    let mut previous: Option<i64> = None;
    for timestamp in timestamps {
        if let Some(previous_timestamp) = previous {
            let expected = previous_timestamp.saturating_add(1);
            if timestamp > expected {
                if let (Ok(start), Ok(end)) = (
                    timestamp_seconds(expected, "gap start"),
                    timestamp_seconds(timestamp.saturating_sub(1), "gap end"),
                ) {
                    gaps.insert(SourceGap { start, end });
                }
            }
        }
        previous = Some(timestamp);
    }
    gaps.into_iter().collect()
}

fn sign_request(
    credentials: &ChainlinkCredentials,
    method: &str,
    full_path: &str,
    timestamp_ms: u64,
) -> Result<String, StrategyError> {
    type HmacSha256 = Hmac<Sha256>;
    let body_hash = hex_digest(Sha256::digest([]));
    let message = format!(
        "{} {} {} {} {}",
        method.to_ascii_uppercase(),
        full_path,
        body_hash,
        credentials.api_key.trim(),
        timestamp_ms
    );
    let mut mac =
        HmacSha256::new_from_slice(credentials.api_secret.trim().as_bytes()).map_err(|_| {
            StrategyError::new(
                StrategyErrorKind::InvalidConfiguration,
                "chainlink_reference_invalid_hmac_secret",
                "Chainlink HMAC secret could not initialize SHA-256 authentication",
            )
        })?;
    mac.update(message.as_bytes());
    Ok(hex::encode(mac.finalize().into_bytes()))
}

fn credentials_from_environment() -> Result<ChainlinkCredentials, StrategyFactoryError> {
    let api_key = required_secret(API_KEY_ENV)?;
    let api_secret = required_secret(API_SECRET_ENV)?;
    Ok(ChainlinkCredentials {
        api_key,
        api_secret,
    })
}

fn required_secret(name: &'static str) -> Result<String, StrategyFactoryError> {
    let value = env::var(name).map_err(|_| {
        StrategyFactoryError::Construction(format!(
            "required Chainlink credential environment variable {name} is unavailable"
        ))
    })?;
    if value.trim().is_empty() {
        return Err(StrategyFactoryError::Construction(format!(
            "required Chainlink credential environment variable {name} is empty"
        )));
    }
    Ok(value)
}

fn validate_checkpoint(checkpoint: &ReferencePriceCheckpoint) -> Result<(), StrategyFactoryError> {
    if let Some(timestamp) = checkpoint.last_source_timestamp_seconds {
        if timestamp < 0 {
            return Err(StrategyFactoryError::Construction(
                "Chainlink reference-price checkpoint cannot be negative".to_owned(),
            ));
        }
        if timestamp > Utc::now().timestamp().saturating_add(300) {
            return Err(StrategyFactoryError::Construction(
                "Chainlink reference-price checkpoint is implausibly in the future".to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_https_origin(value: &str) -> Result<(), StrategyFactoryError> {
    parse_allowed_chainlink_origin(value)
        .map(|_| ())
        .map_err(invalid_config)
}

fn parse_allowed_chainlink_origin(value: &str) -> Result<Url, String> {
    let url = Url::parse(value).map_err(|error| format!("rest_base_url is invalid: {error}"))?;
    if url.scheme() != "https"
        || url.host_str() != Some("api.dataengine.chain.link")
        || url.port_or_known_default() != Some(443)
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.path() != "/"
    {
        return Err(
            "rest_base_url must be the exact https://api.dataengine.chain.link origin".to_owned(),
        );
    }
    Ok(url)
}

fn credentialed_request_url(base_url: &str, path: &str) -> Result<Url, StrategyError> {
    let base = parse_allowed_chainlink_origin(base_url).map_err(|message| {
        StrategyError::new(
            StrategyErrorKind::InvalidConfiguration,
            "chainlink_reference_origin_not_allowed",
            message,
        )
    })?;
    let endpoint = base.join(path).map_err(|error| {
        StrategyError::new(
            StrategyErrorKind::InvalidConfiguration,
            "chainlink_reference_request_url_invalid",
            format!("failed to construct the allowlisted Chainlink request URL: {error}"),
        )
    })?;
    if endpoint.scheme() != "https"
        || endpoint.host_str() != Some("api.dataengine.chain.link")
        || endpoint.port_or_known_default() != Some(443)
    {
        return Err(StrategyError::new(
            StrategyErrorKind::InvalidConfiguration,
            "chainlink_reference_origin_not_allowed",
            "credentialed Chainlink request escaped the allowlisted origin",
        ));
    }
    Ok(endpoint)
}

fn build_chainlink_http_client(
    request_timeout_seconds: u64,
) -> Result<Client, StrategyFactoryError> {
    Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(request_timeout_seconds))
        // The request carries custom HMAC credentials. Never allow reqwest to
        // replay those headers to a redirect target, even if the provider
        // response supplies a valid Location header.
        .redirect(chainlink_redirect_policy())
        .user_agent("capitonic-market-data-ingester/0.1")
        .build()
        .map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "failed to build Chainlink HTTP client: {error}"
            ))
        })
}

fn chainlink_redirect_policy() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::none()
}

fn retry_delay(config: &ChainlinkBtcusdReferencePriceConfig, failed_attempt: u8) -> Duration {
    let shift = u32::from(failed_attempt.saturating_sub(1)).min(16);
    let multiplier = 1_u64.checked_shl(shift).unwrap_or(u64::MAX);
    Duration::from_millis(
        config
            .retry_initial_delay_ms
            .saturating_mul(multiplier)
            .min(config.retry_max_delay_ms),
    )
}

fn current_timestamp_millis() -> Result<u64, StrategyError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            integrity(
                "chainlink_reference_system_time",
                format!("system time predates the Unix epoch: {error}"),
            )
        })?
        .as_millis()
        .try_into()
        .map_err(|_| {
            integrity(
                "chainlink_reference_system_time_overflow",
                "system timestamp exceeded u64 milliseconds",
            )
        })
}

fn timestamp_seconds(value: i64, name: &str) -> Result<DateTime<Utc>, StrategyError> {
    Utc.timestamp_opt(value, 0).single().ok_or_else(|| {
        integrity(
            "chainlink_reference_timestamp_out_of_range",
            format!("Chainlink {name} is outside the supported range"),
        )
    })
}

fn floor_period(
    timestamp: DateTime<Utc>,
    period_seconds: i64,
) -> Result<DateTime<Utc>, StrategyError> {
    let seconds = timestamp.timestamp();
    let floored = seconds - seconds.rem_euclid(period_seconds);
    timestamp_seconds(floored, "artifact window")
}

fn minimum(current: Option<DateTime<Utc>>, candidate: DateTime<Utc>) -> Option<DateTime<Utc>> {
    Some(current.map_or(candidate, |value| value.min(candidate)))
}

fn maximum(current: Option<DateTime<Utc>>, candidate: DateTime<Utc>) -> Option<DateTime<Utc>> {
    Some(current.map_or(candidate, |value| value.max(candidate)))
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in bytes.as_ref() {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn invalid_config(message: impl Into<String>) -> StrategyFactoryError {
    StrategyFactoryError::InvalidConfiguration(message.into())
}

fn source_error<E>(code: &'static str) -> impl FnOnce(E) -> StrategyError
where
    E: std::fmt::Display,
{
    move |error| source(code, error.to_string())
}

fn source(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::TransientSource, code, message)
}

fn database_error<E>(code: &'static str) -> impl FnOnce(E) -> StrategyError
where
    E: std::fmt::Display,
{
    move |error| {
        StrategyError::new(
            StrategyErrorKind::TransientDatabase,
            code,
            error.to_string(),
        )
    }
}

fn integrity_error<E>(code: &'static str) -> impl FnOnce(E) -> StrategyError
where
    E: std::fmt::Display,
{
    move |error| integrity(code, error.to_string())
}

fn integrity(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::Integrity, code, message)
}

fn lease_lost(action: &str) -> StrategyError {
    StrategyError::new(
        StrategyErrorKind::LeaseLost,
        "chainlink_reference_lease_lost",
        format!("profile lease was lost while {action}"),
    )
}

fn owned_drain_failure(original: StrategyError, drain: StrategyError) -> StrategyError {
    if drain.kind == StrategyErrorKind::LeaseLost {
        original
    } else {
        drain
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use chrono::Duration as ChronoDuration;
    use pretty_assertions::assert_eq;

    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/chainlink/reports_page_v3.json"
        ))
        .expect("valid checked-in Chainlink fixture")
    }

    fn page_observation(timestamp: i64, hash_byte: char) -> ReferencePriceObservation {
        ReferencePriceObservation {
            feed_id: BTCUSD_FEED_ID.to_owned(),
            source_timestamp: timestamp_seconds(timestamp, "test report").expect("valid time"),
            valid_from_timestamp: timestamp_seconds(timestamp, "test report").expect("valid time"),
            price: Decimal::ONE,
            bid: Decimal::ONE,
            ask: Decimal::ONE,
            provider_available_at: None,
            received_at: timestamp_seconds(timestamp, "test receipt").expect("valid time"),
            report_sha256: hash_byte.to_string().repeat(64),
            payload_sha256: hash_byte.to_string().repeat(64),
        }
    }

    #[test]
    fn default_config_is_narrow_valid_and_contains_no_credentials() {
        let config = ChainlinkBtcusdReferencePriceConfig::default();
        config.validate().expect("default config should be valid");
        let encoded = serde_json::to_value(config).expect("config serializes");
        assert_eq!(encoded["feed_id"], BTCUSD_FEED_ID);
        assert_eq!(encoded["page_limit"], 100);
        assert!(encoded
            .as_object()
            .expect("config is an object")
            .keys()
            .all(|key| !key.contains("key") && !key.contains("secret")));
    }

    #[test]
    fn unknown_config_fields_are_rejected() {
        let mut config = serde_json::to_value(ChainlinkBtcusdReferencePriceConfig::default())
            .expect("config serializes");
        config["unexpected"] = json!(true);
        assert!(ChainlinkBtcusdReferencePriceConfig::from_value(&config).is_err());
    }

    #[test]
    fn tiny_pagination_budget_is_rejected() {
        let cannot_cover_initial_window = ChainlinkBtcusdReferencePriceConfig {
            recent_window_seconds: 5,
            overlap_seconds: 1,
            page_limit: 2,
            max_pages_per_poll: 2,
            ..ChainlinkBtcusdReferencePriceConfig::default()
        };
        assert!(cannot_cover_initial_window.validate().is_err());

        let cannot_advance_after_overlap = ChainlinkBtcusdReferencePriceConfig {
            recent_window_seconds: 5,
            overlap_seconds: 5,
            page_limit: 7,
            max_pages_per_poll: 1,
            ..ChainlinkBtcusdReferencePriceConfig::default()
        };
        assert!(cannot_advance_after_overlap.validate().is_err());
    }

    #[test]
    fn provider_page_limit_is_enforced() {
        let config = ChainlinkBtcusdReferencePriceConfig {
            page_limit: 101,
            ..ChainlinkBtcusdReferencePriceConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn stale_checkpoint_advances_in_bounded_overlapping_prefixes() {
        let config = ChainlinkBtcusdReferencePriceConfig::default();
        let checkpoint = 100_000;
        let now = 1_000_000;
        let safe_seconds =
            safe_pagination_second_budget(config.page_limit, config.max_pages_per_poll)
                .expect("default pagination budget");
        let forward_seconds = i64::try_from(safe_seconds).expect("budget fits i64")
            - i64::try_from(config.overlap_seconds).expect("overlap fits i64")
            - 1;

        let first = plan_live_fetch_range(Some(checkpoint), now, &config)
            .expect("first bounded catch-up range");
        assert_eq!(first.start, checkpoint - 5);
        assert_eq!(first.end, checkpoint + forward_seconds);
        assert!(
            first.end
                < now
                    - i64::try_from(config.recent_window_seconds).expect("recent window fits i64"),
            "a stale durable checkpoint must not jump to the recent floor"
        );
        assert_eq!(
            first.end - first.start + 1,
            i64::try_from(safe_seconds).expect("budget fits i64")
        );

        let second = plan_live_fetch_range(Some(first.end), now, &config)
            .expect("second bounded catch-up range");
        assert_eq!(second.start, first.end - 5);
        assert_eq!(second.end, first.end + forward_seconds);
    }

    #[test]
    fn malicious_https_origin_is_rejected_before_a_credentialed_request() {
        let config = ChainlinkBtcusdReferencePriceConfig {
            rest_base_url: "https://api.dataengine.chain.link.attacker.example".to_owned(),
            ..ChainlinkBtcusdReferencePriceConfig::default()
        };
        assert!(config.validate().is_err());
        let error = credentialed_request_url(
            &config.rest_base_url,
            "/api/v1/reports/page?feedID=redacted&startTimestamp=1&limit=1",
        )
        .expect_err("malicious HTTPS origin must fail closed");
        assert_eq!(error.kind, StrategyErrorKind::InvalidConfiguration);
        assert_eq!(error.code, "chainlink_reference_origin_not_allowed");

        assert!(credentialed_request_url(
            DEFAULT_REST_BASE_URL,
            "/api/v1/reports/page?feedID=redacted&startTimestamp=1&limit=1",
        )
        .is_ok());
    }

    #[test]
    fn chainlink_http_client_uses_no_redirect_policy() {
        assert_eq!(format!("{:?}", chainlink_redirect_policy()), "Policy(None)");
        build_chainlink_http_client(1).expect("no-redirect client builds");
    }

    #[test]
    fn gap_repairs_are_bounded_and_partial_coverage_never_completes() {
        assert_eq!(
            bounded_gap_repair_range(100, 110, 11).expect("valid bounded gap"),
            Some(SourceSecondRange {
                start: 100,
                end: 110,
            })
        );
        assert_eq!(
            bounded_gap_repair_range(100, 110, 10).expect("valid oversized gap"),
            None
        );
        assert!(!gap_second_count_is_complete(
            SourceSecondRange {
                start: 100,
                end: 110,
            },
            4,
        )
        .expect("valid full-gap count"));
        assert!(gap_second_count_is_complete(
            SourceSecondRange {
                start: 100,
                end: 110,
            },
            11,
        )
        .expect("complete full-gap count"));
    }

    #[test]
    fn full_page_overlaps_until_multiple_final_second_identities_are_proven() {
        let mut observations = Vec::new();
        let mut seen = BTreeSet::new();
        let first = vec![
            page_observation(10, 'a'),
            page_observation(11, 'b'),
            page_observation(11, 'c'),
        ];
        let PageDisposition::Continue(boundary) =
            reconcile_reports_page(&mut observations, &mut seen, first, 10, 11, 3, None)
                .expect("full page requires boundary proof")
        else {
            panic!("full final-second page must not be accepted as complete");
        };
        assert_eq!(boundary.timestamp, 11);
        assert_eq!(boundary.report_sha256s.len(), 2);

        let proof = vec![
            page_observation(11, 'b'),
            page_observation(11, 'c'),
            page_observation(12, 'd'),
        ];
        assert_eq!(
            reconcile_reports_page(
                &mut observations,
                &mut seen,
                proof,
                boundary.timestamp,
                11,
                3,
                Some(&boundary),
            )
            .expect("later timestamp proves the inclusive boundary"),
            PageDisposition::Complete
        );
        assert_eq!(observations.len(), 3);
        assert_eq!(
            observations
                .iter()
                .filter(|observation| observation.source_timestamp.timestamp() == 11)
                .count(),
            2
        );
    }

    #[test]
    fn repeated_full_same_second_page_fails_closed_without_false_completion() {
        let mut observations = Vec::new();
        let mut seen = BTreeSet::new();
        let PageDisposition::Continue(first_boundary) = reconcile_reports_page(
            &mut observations,
            &mut seen,
            vec![
                page_observation(10, 'a'),
                page_observation(11, 'b'),
                page_observation(11, 'c'),
            ],
            10,
            20,
            3,
            None,
        )
        .expect("first full page requires proof") else {
            panic!("first page must remain unproven");
        };
        let saturated = vec![
            page_observation(11, 'b'),
            page_observation(11, 'c'),
            page_observation(11, 'd'),
        ];
        let PageDisposition::Continue(expanded_boundary) = reconcile_reports_page(
            &mut observations,
            &mut seen,
            saturated.clone(),
            first_boundary.timestamp,
            20,
            3,
            Some(&first_boundary),
        )
        .expect("new same-second identity remains unproven") else {
            panic!("saturated boundary must remain unproven");
        };
        assert_eq!(expanded_boundary.report_sha256s.len(), 3);

        let error = reconcile_reports_page(
            &mut observations,
            &mut seen,
            saturated,
            expanded_boundary.timestamp,
            20,
            3,
            Some(&expanded_boundary),
        )
        .expect_err("repeated saturated boundary must fail closed");
        assert_eq!(error.kind, StrategyErrorKind::Integrity);
        assert_eq!(error.code, "chainlink_reference_boundary_saturated");
    }

    #[test]
    fn signed_request_matches_the_chainlink_hmac_contract() {
        let credentials = ChainlinkCredentials {
            api_key: "test-key".to_owned(),
            api_secret: "test-secret".to_owned(),
        };
        assert_eq!(
            sign_request(
                &credentials,
                "GET",
                "/api/v1/reports/page?feedID=0x123&startTimestamp=1&limit=10",
                1_716_211_845_123,
            )
            .expect("signature"),
            "7a1c55e02d4b5d43bd45bfbdd6c95e519260289007ca9a1b4d785fb856102cfc"
        );
    }

    #[test]
    fn sanitized_v3_fixture_decodes_exact_prices_and_same_second_reports() {
        let received_at = Utc
            .with_ymd_and_hms(2025, 1, 1, 0, 0, 5)
            .single()
            .expect("valid time");
        let reports = decode_reports_page(fixture(), BTCUSD_FEED_ID, received_at)
            .expect("fixture should decode");
        assert_eq!(reports.len(), 3);
        assert_eq!(
            reports[0].source_timestamp,
            Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0)
                .single()
                .expect("valid time")
        );
        assert_eq!(reports[0].price, Decimal::new(67_123_456_789, 6));
        assert_eq!(reports[0].bid, Decimal::new(67_123, 0));
        assert_eq!(reports[0].ask, Decimal::new(67_124, 0));
        assert_eq!(reports[0].price.scale(), 18);
        assert_eq!(reports[0].bid.scale(), 18);
        assert_eq!(reports[0].ask.scale(), 18);
        assert_eq!(reports[0].received_at, received_at);
        assert_eq!(reports[0].provider_available_at, None);
        assert_eq!(reports[0].source_timestamp, reports[1].source_timestamp);
        assert_ne!(reports[0].report_sha256, reports[1].report_sha256);
        assert_ne!(reports[0].payload_sha256, reports[1].payload_sha256);
        assert!(reports
            .iter()
            .all(|report| report.report_sha256.len() == 64));
    }

    #[test]
    fn envelope_feed_and_timestamp_must_match_the_v3_payload() {
        let mut wrong_timestamp = fixture();
        wrong_timestamp["reports"][0]["observationsTimestamp"] = json!(1_735_689_601_u64);
        assert!(decode_reports_page(wrong_timestamp, BTCUSD_FEED_ID, Utc::now()).is_err());

        let mut wrong_feed = fixture();
        wrong_feed["reports"][0]["feedID"] =
            json!("0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b9");
        assert!(decode_reports_page(wrong_feed, BTCUSD_FEED_ID, Utc::now()).is_err());
    }

    #[test]
    fn same_second_reports_do_not_hide_a_precisely_bounded_gap() {
        let received_at = Utc::now();
        let reports = decode_reports_page(fixture(), BTCUSD_FEED_ID, received_at)
            .expect("fixture should decode");
        let start = reports[0].source_timestamp;
        assert_eq!(
            find_gaps(&reports, None),
            vec![SourceGap {
                start: start + ChronoDuration::seconds(1),
                end: start + ChronoDuration::seconds(1),
            }]
        );
    }

    #[test]
    fn canonical_hash_normalizes_decimal_scale() {
        let first = canonical_payload_sha256(
            BTCUSD_FEED_ID,
            1,
            0,
            "1.0".parse().expect("decimal"),
            "0.90".parse().expect("decimal"),
            "1.10".parse().expect("decimal"),
            &"a".repeat(64),
        );
        let second = canonical_payload_sha256(
            BTCUSD_FEED_ID,
            1,
            0,
            "1".parse().expect("decimal"),
            "0.9".parse().expect("decimal"),
            "1.1".parse().expect("decimal"),
            &"a".repeat(64),
        );
        assert_eq!(first, second);
    }

    #[test]
    fn checkpoint_rejects_negative_and_implausibly_future_timestamps() {
        for timestamp in [-1, Utc::now().timestamp().saturating_add(86_400)] {
            assert!(validate_checkpoint(&ReferencePriceCheckpoint {
                last_source_timestamp_seconds: Some(timestamp),
            })
            .is_err());
        }
    }

    #[test]
    fn retry_delay_is_bounded_exponential_backoff() {
        let config = ChainlinkBtcusdReferencePriceConfig::default();
        assert_eq!(retry_delay(&config, 1), Duration::from_millis(250));
        assert_eq!(retry_delay(&config, 2), Duration::from_millis(500));
        assert_eq!(retry_delay(&config, 5), Duration::from_millis(2_000));
    }

    #[test]
    fn owned_drain_preserves_true_lease_loss_and_surfaces_operational_failure() {
        let original = lease_lost("committing strict progress");
        let lost_during_drain = lease_lost("draining an artifact");
        let preserved = owned_drain_failure(original, lost_during_drain);
        assert_eq!(preserved.kind, StrategyErrorKind::LeaseLost);
        assert_eq!(
            preserved.message,
            "profile lease was lost while committing strict progress"
        );

        let original = lease_lost("committing strict progress");
        let database = StrategyError::new(
            StrategyErrorKind::TransientDatabase,
            "owned_drain_database_failure",
            "database unavailable while verifying owned drain",
        );
        let surfaced = owned_drain_failure(original, database);
        assert_eq!(surfaced.kind, StrategyErrorKind::TransientDatabase);
        assert_eq!(surfaced.code, "owned_drain_database_failure");
    }
}

impl ChainlinkBtcusdReferencePriceStrategy {
    async fn persist_observations(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        observations: &[ReferencePriceObservation],
    ) -> Result<PersistedBatch, StrategyError> {
        let mut batch = PersistedBatch::default();
        for chunk in observations.chunks(MAX_INSERT_ROWS) {
            let mut query = QueryBuilder::<Postgres>::new(
                "INSERT INTO market_data.chainlink_btcusd_reference_prices (\
                 source, feed_id, source_timestamp, valid_from_timestamp, \
                 price, bid, ask, provider_available_at, received_at, \
                 report_sha256, payload_sha256, strategy_key, capture_artifact_id) ",
            );
            query.push_values(chunk, |mut row, observation| {
                row.push_bind(SOURCE)
                    .push_bind(&observation.feed_id)
                    .push_bind(observation.source_timestamp)
                    .push_bind(observation.valid_from_timestamp)
                    .push_bind(observation.price)
                    .push_bind(observation.bid)
                    .push_bind(observation.ask)
                    .push_bind(observation.provider_available_at)
                    .push_bind(observation.received_at)
                    .push_bind(&observation.report_sha256)
                    .push_bind(&observation.payload_sha256)
                    .push_bind(STRATEGY_KEY.as_str())
                    .push_bind(artifact_id);
            });
            query.push(
                " ON CONFLICT (feed_id, source_timestamp, report_sha256) DO NOTHING \
                 RETURNING source_timestamp, report_sha256::text AS report_sha256",
            );
            let inserted = query
                .build_query_as::<InsertedReferencePrice>()
                .fetch_all(&mut **transaction)
                .await
                .map_err(database_error("chainlink_reference_insert_facts"))?;
            let inserted_identities = inserted
                .into_iter()
                .map(|row| (row.source_timestamp.timestamp(), row.report_sha256))
                .collect::<HashSet<_>>();

            for observation in chunk {
                let identity = (
                    observation.source_timestamp.timestamp(),
                    observation.report_sha256.clone(),
                );
                if inserted_identities.contains(&identity) {
                    batch.record_insert(observation);
                } else {
                    self.verify_replayed_observation(transaction, observation)
                        .await?;
                }
                batch.verified += 1;
            }
        }
        Ok(batch)
    }

    async fn verify_replayed_observation(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        observation: &ReferencePriceObservation,
    ) -> Result<(), StrategyError> {
        let stored = sqlx::query_as::<_, StoredReferencePrice>(
            r#"
            SELECT valid_from_timestamp, price, bid, ask,
                   provider_available_at,
                   payload_sha256::text AS payload_sha256
            FROM market_data.chainlink_btcusd_reference_prices
            WHERE feed_id = $1
              AND source_timestamp = $2
              AND report_sha256 = $3
            "#,
        )
        .bind(&observation.feed_id)
        .bind(observation.source_timestamp)
        .bind(&observation.report_sha256)
        .fetch_optional(&mut **transaction)
        .await
        .map_err(database_error("chainlink_reference_verify_replay"))?
        .ok_or_else(|| {
            integrity(
                "chainlink_reference_conflict_missing",
                "fact conflicted but no durable Chainlink row was visible",
            )
        })?;
        if stored.valid_from_timestamp != observation.valid_from_timestamp
            || stored.price != observation.price
            || stored.bid != observation.bid
            || stored.ask != observation.ask
            || stored.provider_available_at != observation.provider_available_at
            || stored.payload_sha256 != observation.payload_sha256
        {
            return Err(integrity(
                "chainlink_reference_immutable_conflict",
                format!(
                    "Chainlink report {} at {} changed factual values",
                    observation.report_sha256, observation.source_timestamp
                ),
            ));
        }
        Ok(())
    }

    async fn record_artifact_batch(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        persisted: &PersistedBatch,
    ) -> Result<(), StrategyError> {
        if persisted.inserted == 0 {
            return Ok(());
        }
        self.artifacts
            .record_batch_in(
                transaction,
                artifact_id,
                &ArtifactBatch {
                    inserted_record_count: persisted.inserted,
                    minimum_source_timestamp: persisted.minimum_inserted_source,
                    maximum_source_timestamp: persisted.maximum_inserted_source,
                    minimum_received_at: persisted.minimum_inserted_received,
                    maximum_received_at: persisted.maximum_inserted_received,
                    start_cursor: persisted
                        .minimum_inserted_source
                        .map(|timestamp| timestamp.timestamp().to_string()),
                    end_cursor: persisted
                        .maximum_inserted_source
                        .map(|timestamp| timestamp.timestamp().to_string()),
                },
            )
            .await
            .map_err(database_error("chainlink_reference_record_artifact_batch"))?
            .ok_or_else(|| {
                integrity(
                    "chainlink_reference_artifact_closed",
                    "Chainlink capture artifact is not open",
                )
            })?;
        Ok(())
    }

    async fn ensure_artifact(
        &self,
        received_at: DateTime<Utc>,
        checkpoint: &ReferencePriceCheckpoint,
    ) -> Result<CaptureArtifact, StrategyError> {
        if let Some(open) = self
            .artifacts
            .get_open(STRATEGY_KEY)
            .await
            .map_err(database_error("chainlink_reference_load_artifact"))?
        {
            if open.profile_generation > self.profile_generation {
                return Err(lease_lost("opening a Chainlink capture artifact"));
            }
            if open.profile_generation == self.profile_generation {
                if open.config_schema_version != CONFIG_SCHEMA_VERSION
                    || open.config_snapshot != self.config_snapshot
                {
                    return Err(integrity(
                        "chainlink_reference_artifact_config_mismatch",
                        "open artifact config differs within the same profile generation",
                    ));
                }
                if received_at < open.capture_window_end {
                    return Ok(open);
                }
            }
            self.seal_artifact(&open, false).await?;
        }

        let window_start = floor_period(received_at, self.config.artifact_window_seconds)?;
        let window_end =
            window_start + chrono::Duration::seconds(self.config.artifact_window_seconds);
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(database_error("chainlink_reference_begin_artifact_create"))?;
        self.lock_current_lease(&mut transaction, "creating a Chainlink capture artifact")
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
                    capture_window_end: window_end,
                    start_cursor: checkpoint
                        .last_source_timestamp_seconds
                        .map(|timestamp| timestamp.to_string()),
                },
            )
            .await
            .map_err(database_error("chainlink_reference_create_artifact"))?;
        transaction
            .commit()
            .await
            .map_err(database_error("chainlink_reference_commit_artifact_create"))?;
        Ok(artifact)
    }

    async fn seal_open_artifact(&self) -> Result<(), StrategyError> {
        if let Some(open) = self
            .artifacts
            .get_open(STRATEGY_KEY)
            .await
            .map_err(database_error("chainlink_reference_load_artifact_for_seal"))?
        {
            if open.profile_generation > self.profile_generation {
                return Err(lease_lost("draining a newer Chainlink capture artifact"));
            }
            self.seal_artifact(&open, true).await?;
        } else {
            self.verify_owned_lease().await?;
        }
        Ok(())
    }

    async fn seal_artifact(
        &self,
        artifact: &CaptureArtifact,
        allow_draining_generation: bool,
    ) -> Result<(), StrategyError> {
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(database_error("chainlink_reference_begin_artifact_seal"))?;
        if allow_draining_generation {
            self.lock_owned_lease(&mut transaction, "sealing a Chainlink capture artifact")
                .await?;
        } else {
            self.lock_current_lease(&mut transaction, "sealing a Chainlink capture artifact")
                .await?;
        }
        let content_sha256 = self
            .artifact_content_sha256(&mut transaction, artifact)
            .await?;
        self.artifacts
            .complete_in(
                &mut transaction,
                artifact.artifact_id,
                &content_sha256,
                artifact.end_cursor.as_deref(),
            )
            .await
            .map_err(database_error("chainlink_reference_complete_artifact"))?
            .ok_or_else(|| {
                integrity(
                    "chainlink_reference_artifact_not_open",
                    "Chainlink capture artifact could not be completed",
                )
            })?;
        transaction
            .commit()
            .await
            .map_err(database_error("chainlink_reference_commit_artifact_seal"))?;
        Ok(())
    }

    async fn artifact_content_sha256(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact: &CaptureArtifact,
    ) -> Result<String, StrategyError> {
        let hashes = sqlx::query_scalar::<_, String>(
            r#"
            SELECT payload_sha256::text
            FROM market_data.chainlink_btcusd_reference_prices
            WHERE capture_artifact_id = $1
            ORDER BY source_timestamp, feed_id, report_sha256
            "#,
        )
        .bind(artifact.artifact_id)
        .fetch_all(&mut **transaction)
        .await
        .map_err(database_error("chainlink_reference_hash_artifact"))?;
        if i64::try_from(hashes.len()).ok() != Some(artifact.record_count) {
            return Err(integrity(
                "chainlink_reference_artifact_count_mismatch",
                format!(
                    "artifact records {} differ from stored Chainlink facts {}",
                    artifact.record_count,
                    hashes.len()
                ),
            ));
        }
        let mut digest = Sha256::new();
        for hash in hashes {
            digest.update(hash.as_bytes());
        }
        Ok(hex_digest(digest.finalize()))
    }

    async fn lock_current_lease(
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
            .map_err(database_error("chainlink_reference_lock_current_lease"))?;
        if !locked {
            return Err(lease_lost(action));
        }
        Ok(())
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
            .map_err(database_error("chainlink_reference_lock_owned_lease"))?;
        if !locked {
            return Err(lease_lost(action));
        }
        Ok(())
    }

    async fn verify_owned_lease(&self) -> Result<(), StrategyError> {
        let mut transaction = self.pool.begin().await.map_err(database_error(
            "chainlink_reference_begin_owned_lease_check",
        ))?;
        self.lock_owned_lease(&mut transaction, "draining a superseded generation")
            .await?;
        transaction.commit().await.map_err(database_error(
            "chainlink_reference_commit_owned_lease_check",
        ))
    }
}

impl ChainlinkBtcusdReferencePriceStrategy {
    async fn reconcile_gaps(
        &self,
        checkpoint: &ReferencePriceCheckpoint,
    ) -> Result<(), StrategyError> {
        let repair_budget_seconds =
            safe_pagination_second_budget(self.config.page_limit, self.config.max_pages_per_poll)
                .ok_or_else(|| {
                integrity(
                    "chainlink_reference_pagination_budget_invalid",
                    "pagination cannot safely repair an inclusive source-time range",
                )
            })?;
        let unresolved = self
            .gaps
            .list_unresolved_limited(STRATEGY_KEY, GAP_REPAIRS_PER_POLL)
            .await
            .map_err(database_error("chainlink_reference_list_gaps"))?;
        for gap in unresolved {
            let (Some(start), Some(end)) = (gap.source_time_start, gap.source_time_end) else {
                continue;
            };
            let Some(repair_range) = bounded_gap_repair_range(
                start.timestamp(),
                end.timestamp(),
                repair_budget_seconds,
            )?
            else {
                self.terminalize_oversized_gap(gap.gap_id).await?;
                continue;
            };
            let mut complete = self.gap_range_is_complete(repair_range).await?;
            let mut repair_artifact_id = None;
            if !complete {
                if !self.begin_gap_repair(gap.gap_id).await? {
                    continue;
                }
                let observations = self
                    .fetch_range(repair_range.start, repair_range.end)
                    .await?;
                if !observations.is_empty() {
                    repair_artifact_id = Some(
                        self.persist_repair_observations(checkpoint, &observations)
                            .await?,
                    );
                }
                complete = self.gap_range_is_complete(repair_range).await?;
            } else if gap.repair_attempts == 0 && !self.begin_gap_repair(gap.gap_id).await? {
                continue;
            }
            if !complete {
                continue;
            }

            let artifact = match self
                .artifacts
                .get_open(STRATEGY_KEY)
                .await
                .map_err(database_error("chainlink_reference_load_repair_artifact"))?
            {
                Some(artifact) => artifact,
                None => self.ensure_artifact(Utc::now(), checkpoint).await?,
            };
            if artifact.profile_generation != self.profile_generation {
                return Err(lease_lost("completing a Chainlink gap repair artifact"));
            }
            if repair_artifact_id.is_some_and(|expected| expected != artifact.artifact_id) {
                return Err(integrity(
                    "chainlink_reference_repair_artifact_changed",
                    "open Chainlink capture artifact changed during gap repair",
                ));
            }
            self.seal_and_complete_gap(gap.gap_id, &artifact, repair_range)
                .await?;
        }
        Ok(())
    }

    async fn begin_gap_repair(&self, gap_id: Uuid) -> Result<bool, StrategyError> {
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(database_error("chainlink_reference_begin_gap_transaction"))?;
        self.lock_current_lease(&mut transaction, "beginning a Chainlink gap repair")
            .await?;
        let repairing = self
            .gaps
            .begin_repair_in(&mut transaction, gap_id)
            .await
            .map_err(database_error("chainlink_reference_begin_gap_repair"))?
            .is_some();
        transaction
            .commit()
            .await
            .map_err(database_error("chainlink_reference_commit_gap_begin"))?;
        Ok(repairing)
    }

    async fn terminalize_oversized_gap(&self, gap_id: Uuid) -> Result<(), StrategyError> {
        let mut transaction = self.pool.begin().await.map_err(database_error(
            "chainlink_reference_begin_oversized_gap_resolution",
        ))?;
        self.lock_current_lease(
            &mut transaction,
            "terminalizing an oversized legacy Chainlink gap",
        )
        .await?;
        self.gaps
            .mark_unrecoverable_in(
                &mut transaction,
                gap_id,
                "repair_range_exceeds_safe_pagination_budget",
                Some("immutable legacy gap exceeds the bounded Chainlink pagination repair window"),
            )
            .await
            .map_err(database_error(
                "chainlink_reference_terminalize_oversized_gap",
            ))?
            .ok_or_else(|| {
                integrity(
                    "chainlink_reference_oversized_gap_resolution_race",
                    "oversized Chainlink gap could not be terminalized",
                )
            })?;
        transaction.commit().await.map_err(database_error(
            "chainlink_reference_commit_oversized_gap_resolution",
        ))?;
        Ok(())
    }

    async fn persist_repair_observations(
        &self,
        checkpoint: &ReferencePriceCheckpoint,
        observations: &[ReferencePriceObservation],
    ) -> Result<Uuid, StrategyError> {
        let received_at = observations
            .iter()
            .map(|observation| observation.received_at)
            .max()
            .expect("nonempty repair batch has a receipt timestamp");
        let artifact = self.ensure_artifact(received_at, checkpoint).await?;
        let mut transaction = self.pool.begin().await.map_err(database_error(
            "chainlink_reference_repair_begin_transaction",
        ))?;
        let persisted = self
            .persist_observations(&mut transaction, artifact.artifact_id, observations)
            .await?;
        self.record_artifact_batch(&mut transaction, artifact.artifact_id, &persisted)
            .await?;
        let maximum_source_timestamp = observations
            .iter()
            .map(|observation| observation.source_timestamp)
            .max();
        let advanced = self
            .profiles
            .record_progress_in(
                &mut transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &StrategyProgress {
                    verified_record_count: persisted.verified,
                    checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
                    checkpoint: serde_json::to_value(checkpoint).map_err(integrity_error(
                        "chainlink_reference_encode_repair_checkpoint",
                    ))?,
                    last_source_event_at: maximum_source_timestamp,
                    last_provider_available_at: None,
                    source_watermark: maximum_source_timestamp,
                    availability_watermark: None,
                },
            )
            .await
            .map_err(database_error("chainlink_reference_record_repair_progress"))?;
        if !advanced {
            return Err(lease_lost("committing Chainlink gap-repair progress"));
        }
        transaction
            .commit()
            .await
            .map_err(database_error("chainlink_reference_commit_repair"))?;
        Ok(artifact.artifact_id)
    }

    async fn seal_and_complete_gap(
        &self,
        gap_id: Uuid,
        artifact: &CaptureArtifact,
        repair_range: SourceSecondRange,
    ) -> Result<(), StrategyError> {
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(database_error("chainlink_reference_begin_gap_completion"))?;
        self.lock_current_lease(&mut transaction, "completing a Chainlink gap repair")
            .await?;
        if !self
            .gap_range_is_complete_in(&mut transaction, repair_range)
            .await?
        {
            return Err(integrity(
                "chainlink_reference_gap_range_incomplete",
                "Chainlink gap range was incomplete at its transactional completion fence",
            ));
        }
        let content_sha256 = self
            .artifact_content_sha256(&mut transaction, artifact)
            .await?;
        self.artifacts
            .complete_in(
                &mut transaction,
                artifact.artifact_id,
                &content_sha256,
                artifact.end_cursor.as_deref(),
            )
            .await
            .map_err(database_error(
                "chainlink_reference_complete_repair_artifact",
            ))?
            .ok_or_else(|| {
                integrity(
                    "chainlink_reference_repair_artifact_not_open",
                    "Chainlink repair artifact could not be completed",
                )
            })?;
        self.gaps
            .mark_repaired_in(
                &mut transaction,
                gap_id,
                artifact.artifact_id,
                "provider_interval_recovered",
                Some("all one-second Chainlink source timestamps are now durably present"),
            )
            .await
            .map_err(database_error("chainlink_reference_complete_gap_repair"))?
            .ok_or_else(|| {
                integrity(
                    "chainlink_reference_gap_repair_race",
                    "Chainlink gap could not be marked repaired",
                )
            })?;
        transaction
            .commit()
            .await
            .map_err(database_error("chainlink_reference_commit_gap_completion"))?;
        Ok(())
    }

    async fn gap_range_is_complete(&self, range: SourceSecondRange) -> Result<bool, StrategyError> {
        let start = timestamp_seconds(range.start, "gap repair start")?;
        let end = timestamp_seconds(range.end, "gap repair end")?;
        let distinct_source_seconds = sqlx::query_scalar::<_, i64>(
            r#"
            SELECT count(DISTINCT source_timestamp)::bigint
            FROM market_data.chainlink_btcusd_reference_prices
            WHERE feed_id = $1
              AND source_timestamp >= $2
              AND source_timestamp <= $3
            "#,
        )
        .bind(&self.config.feed_id)
        .bind(start)
        .bind(end)
        .fetch_one(&self.pool)
        .await
        .map_err(database_error("chainlink_reference_verify_gap_range"))?;
        gap_second_count_is_complete(range, distinct_source_seconds)
    }

    async fn gap_range_is_complete_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        range: SourceSecondRange,
    ) -> Result<bool, StrategyError> {
        let start = timestamp_seconds(range.start, "gap repair start")?;
        let end = timestamp_seconds(range.end, "gap repair end")?;
        let distinct_source_seconds = sqlx::query_scalar::<_, i64>(
            r#"
            SELECT count(DISTINCT source_timestamp)::bigint
            FROM market_data.chainlink_btcusd_reference_prices
            WHERE feed_id = $1
              AND source_timestamp >= $2
              AND source_timestamp <= $3
            "#,
        )
        .bind(&self.config.feed_id)
        .bind(start)
        .bind(end)
        .fetch_one(&mut **transaction)
        .await
        .map_err(database_error(
            "chainlink_reference_verify_gap_range_transactional",
        ))?;
        gap_second_count_is_complete(range, distinct_source_seconds)
    }
}
