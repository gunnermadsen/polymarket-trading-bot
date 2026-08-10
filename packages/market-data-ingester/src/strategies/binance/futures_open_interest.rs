use std::{fmt::Write as _, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use reqwest::{Client, Url};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::{FromRow, PgPool, Postgres, Transaction};
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

const STRATEGY_KEY: IngesterStrategyKey = IngesterStrategyKey::BinanceFuturesBtcusdtOpenInterest;
const CONFIG_SCHEMA_VERSION: i32 = 1;
const CHECKPOINT_SCHEMA_VERSION: i32 = 1;
const PERIOD_SECONDS: i64 = 300;
const MAX_PROVIDER_HISTORY_PERIODS: u16 = 8_640;
const MAX_HTTP_RESPONSE_BYTES: usize = 1_048_576;
const GAP_REPAIRS_PER_POLL: i64 = 16;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BinanceFuturesOpenInterestConfig {
    pub symbol: String,
    pub rest_base_url: String,
    pub period: String,
    pub poll_interval_seconds: u64,
    pub request_limit: u16,
    pub startup_lookback_periods: u16,
    pub overlap_periods: u16,
    pub artifact_window_seconds: i64,
    pub request_timeout_seconds: u64,
}

impl Default for BinanceFuturesOpenInterestConfig {
    fn default() -> Self {
        Self {
            symbol: "BTCUSDT".to_owned(),
            rest_base_url: "https://fapi.binance.com".to_owned(),
            period: "5m".to_owned(),
            poll_interval_seconds: 60,
            request_limit: 500,
            startup_lookback_periods: 288,
            overlap_periods: 2,
            artifact_window_seconds: 86_400,
            request_timeout_seconds: 10,
        }
    }
}

impl BinanceFuturesOpenInterestConfig {
    fn validate(&self) -> Result<(), StrategyFactoryError> {
        if self.symbol != "BTCUSDT" {
            return Err(invalid_config("symbol must be BTCUSDT"));
        }
        if self.period != "5m" {
            return Err(invalid_config("period must be 5m"));
        }
        validate_https_base_url(&self.rest_base_url)?;
        if !(15..=300).contains(&self.poll_interval_seconds) {
            return Err(invalid_config(
                "poll_interval_seconds must be between 15 and 300",
            ));
        }
        if !(1..=500).contains(&self.request_limit) {
            return Err(invalid_config("request_limit must be between 1 and 500"));
        }
        if !(1..=MAX_PROVIDER_HISTORY_PERIODS).contains(&self.startup_lookback_periods) {
            return Err(invalid_config(
                "startup_lookback_periods must be between 1 and 8640",
            ));
        }
        if self.overlap_periods == 0 || self.overlap_periods > self.request_limit {
            return Err(invalid_config(
                "overlap_periods must be positive and no greater than request_limit",
            ));
        }
        if self.artifact_window_seconds < PERIOD_SECONDS
            || self.artifact_window_seconds > 86_400
            || self.artifact_window_seconds % PERIOD_SECONDS != 0
        {
            return Err(invalid_config(
                "artifact_window_seconds must be a 5-minute multiple between 300 and 86400",
            ));
        }
        if !(1..=30).contains(&self.request_timeout_seconds) {
            return Err(invalid_config(
                "request_timeout_seconds must be between 1 and 30",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct OpenInterestCheckpoint {
    last_source_timestamp_ms: Option<i64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProviderOpenInterest {
    symbol: String,
    sum_open_interest: String,
    sum_open_interest_value: String,
    #[serde(rename = "CMCCirculatingSupply", default)]
    cmc_circulating_supply: Option<String>,
    timestamp: i64,
}

#[derive(Debug, Clone)]
struct OpenInterestObservation {
    source_timestamp: DateTime<Utc>,
    received_at: DateTime<Utc>,
    sum_open_interest: Decimal,
    sum_open_interest_value: Decimal,
    cmc_circulating_supply: Option<Decimal>,
    source_payload: Value,
    payload_sha256: String,
}

#[derive(Debug, Clone, FromRow)]
struct StoredOpenInterest {
    sum_open_interest: Decimal,
    sum_open_interest_value: Decimal,
    cmc_circulating_supply: Option<Decimal>,
    payload_sha256: String,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BinanceFuturesOpenInterestFactory;

impl StrategyFactory for BinanceFuturesOpenInterestFactory {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    fn config_schema_version(&self) -> i32 {
        CONFIG_SCHEMA_VERSION
    }

    fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError> {
        decode_config(config).map(|_| ())
    }

    fn build(
        &self,
        profile: &IngesterProfile,
        pool: PgPool,
    ) -> Result<Box<dyn IngesterStrategy>, StrategyFactoryError> {
        let config = decode_config(&profile.config)?;
        if profile.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(StrategyFactoryError::Construction(format!(
                "open-interest checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}"
            )));
        }
        let checkpoint: OpenInterestCheckpoint = serde_json::from_value(profile.checkpoint.clone())
            .map_err(|error| {
                StrategyFactoryError::Construction(format!(
                    "invalid open-interest checkpoint: {error}"
                ))
            })?;
        validate_checkpoint(&checkpoint)?;
        let lease_owner = profile.lease_owner.clone().ok_or_else(|| {
            StrategyFactoryError::Construction("profile is missing its claimed lease owner".into())
        })?;
        let lease_token = profile.lease_token.ok_or_else(|| {
            StrategyFactoryError::Construction("profile is missing its claimed lease token".into())
        })?;
        let client = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_seconds))
            .user_agent("capitonic-market-data-ingester/0.1")
            .build()
            .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?;
        let config_snapshot = serde_json::to_value(&config).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "failed to encode effective open-interest config: {error}"
            ))
        })?;

        Ok(Box::new(BinanceFuturesOpenInterestStrategy {
            config,
            config_snapshot,
            profile_generation: profile.desired_generation,
            lease_owner,
            lease_token,
            checkpoint,
            client,
            pool: pool.clone(),
            artifacts: ArtifactRepository::new(pool.clone()),
            gaps: GapRepository::new(pool.clone()),
            profiles: ProfileRepository::new(pool),
        }))
    }
}

struct BinanceFuturesOpenInterestStrategy {
    config: BinanceFuturesOpenInterestConfig,
    config_snapshot: Value,
    profile_generation: i64,
    lease_owner: String,
    lease_token: Uuid,
    checkpoint: OpenInterestCheckpoint,
    client: Client,
    pool: PgPool,
    artifacts: ArtifactRepository,
    gaps: GapRepository,
    profiles: ProfileRepository,
}

#[async_trait]
impl IngesterStrategy for BinanceFuturesOpenInterestStrategy {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    async fn run(&self, shutdown: CancellationToken) -> Result<(), StrategyError> {
        let mut checkpoint = self.checkpoint.clone();
        let mut ticker =
            tokio::time::interval(Duration::from_secs(self.config.poll_interval_seconds));
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

impl BinanceFuturesOpenInterestStrategy {
    async fn capture(&self, checkpoint: &mut OpenInterestCheckpoint) -> Result<(), StrategyError> {
        let observations = self.fetch_observations(checkpoint).await?;
        if observations.is_empty() {
            return Ok(());
        }

        let received_at = observations
            .iter()
            .map(|observation| observation.received_at)
            .max()
            .expect("nonempty observations have a receipt timestamp");
        let artifact = self.ensure_artifact(received_at, checkpoint).await?;
        let gaps = find_gaps(&observations, checkpoint.last_source_timestamp_ms);
        let maximum_source_timestamp = observations
            .last()
            .expect("nonempty observations have a final timestamp")
            .source_timestamp;

        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(database_error("open_interest_begin_transaction"))?;
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
                        reason_code: "binance_open_interest_interval_missing".to_owned(),
                        reason_message: Some(
                            "Binance returned non-contiguous five-minute open-interest rows"
                                .to_owned(),
                        ),
                        source_time_start: Some(gap.start),
                        source_time_end: Some(gap.end),
                        start_cursor: Some(gap.start.timestamp_millis().to_string()),
                        end_cursor: Some(gap.end.timestamp_millis().to_string()),
                    },
                )
                .await
                .map_err(integrity_error("open_interest_record_gap"))?;
            let marked = self
                .profiles
                .mark_degraded_in(
                    &mut transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                    &StrategyDegradation {
                        reason_code: "binance_open_interest_gap".to_owned(),
                        reason_message: "source returned a non-contiguous open-interest series"
                            .to_owned(),
                    },
                )
                .await
                .map_err(database_error("open_interest_mark_degraded"))?;
            if !marked {
                return Err(lease_lost("recording an open-interest gap"));
            }
        }

        if persisted.inserted > 0 {
            self.artifacts
                .record_batch_in(
                    &mut transaction,
                    artifact.artifact_id,
                    &ArtifactBatch {
                        inserted_record_count: persisted.inserted,
                        minimum_source_timestamp: persisted.minimum_inserted_source,
                        maximum_source_timestamp: persisted.maximum_inserted_source,
                        minimum_received_at: persisted.minimum_inserted_received,
                        maximum_received_at: persisted.maximum_inserted_received,
                        start_cursor: persisted
                            .minimum_inserted_source
                            .map(|timestamp| timestamp.timestamp_millis().to_string()),
                        end_cursor: persisted
                            .maximum_inserted_source
                            .map(|timestamp| timestamp.timestamp_millis().to_string()),
                    },
                )
                .await
                .map_err(database_error("open_interest_record_artifact_batch"))?
                .ok_or_else(|| {
                    integrity(
                        "open_interest_artifact_closed",
                        "capture artifact is not open",
                    )
                })?;
        }

        let next_checkpoint = OpenInterestCheckpoint {
            last_source_timestamp_ms: Some(maximum_source_timestamp.timestamp_millis()),
        };
        let progress = StrategyProgress {
            verified_record_count: persisted.verified,
            checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
            checkpoint: serde_json::to_value(&next_checkpoint)
                .map_err(integrity_error("open_interest_encode_checkpoint"))?,
            last_source_event_at: Some(maximum_source_timestamp),
            last_provider_available_at: Some(received_at),
            source_watermark: Some(maximum_source_timestamp),
            availability_watermark: Some(received_at),
        };
        let advanced = self
            .profiles
            .record_progress_in(
                &mut transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &progress,
            )
            .await
            .map_err(database_error("open_interest_record_progress"))?;
        if !advanced {
            return Err(lease_lost("committing open-interest progress"));
        }
        transaction
            .commit()
            .await
            .map_err(database_error("open_interest_commit_transaction"))?;
        *checkpoint = next_checkpoint;
        self.reconcile_gaps(checkpoint).await?;
        Ok(())
    }

    async fn fetch_observations(
        &self,
        checkpoint: &OpenInterestCheckpoint,
    ) -> Result<Vec<OpenInterestObservation>, StrategyError> {
        let request_end = floor_period(Utc::now(), PERIOD_SECONDS)?;
        let provider_floor = request_end
            .timestamp_millis()
            .saturating_sub(i64::from(MAX_PROVIDER_HISTORY_PERIODS) * PERIOD_SECONDS * 1_000);
        let requested_start = checkpoint
            .last_source_timestamp_ms
            .map(|timestamp| {
                timestamp
                    .saturating_sub(i64::from(self.config.overlap_periods) * PERIOD_SECONDS * 1_000)
            })
            .unwrap_or_else(|| {
                request_end.timestamp_millis().saturating_sub(
                    i64::from(self.config.startup_lookback_periods) * PERIOD_SECONDS * 1_000,
                )
            })
            .max(provider_floor);
        let request_end_ms = request_end.timestamp_millis();
        self.fetch_range(requested_start, request_end_ms).await
    }

    async fn fetch_range(
        &self,
        requested_start: i64,
        requested_end: i64,
    ) -> Result<Vec<OpenInterestObservation>, StrategyError> {
        let mut page_start = requested_start;
        let mut observations = Vec::new();

        while page_start <= requested_end {
            let page = self.fetch_page(page_start, requested_end).await?;
            if page.is_empty() {
                break;
            }
            let page_len = page.len();
            let final_timestamp = page
                .last()
                .expect("nonempty provider page has a final timestamp")
                .source_timestamp
                .timestamp_millis();
            if page[0].source_timestamp.timestamp_millis() < page_start
                || final_timestamp > requested_end
            {
                return Err(integrity(
                    "open_interest_page_outside_range",
                    "provider page escaped the requested timestamp range",
                ));
            }
            if observations
                .last()
                .is_some_and(|previous: &OpenInterestObservation| {
                    previous.source_timestamp >= page[0].source_timestamp
                })
            {
                return Err(integrity(
                    "open_interest_page_overlap",
                    "provider pages overlapped or regressed",
                ));
            }
            observations.extend(page);
            if page_len < usize::from(self.config.request_limit) || final_timestamp >= requested_end
            {
                break;
            }
            page_start = final_timestamp.saturating_add(PERIOD_SECONDS * 1_000);
        }
        Ok(observations)
    }

    async fn fetch_page(
        &self,
        start_timestamp_ms: i64,
        end_timestamp_ms: i64,
    ) -> Result<Vec<OpenInterestObservation>, StrategyError> {
        let endpoint = format!(
            "{}/futures/data/openInterestHist",
            self.config.rest_base_url.trim_end_matches('/')
        );
        let mut response = self
            .client
            .get(endpoint)
            .query(&[
                ("symbol", self.config.symbol.as_str()),
                ("period", self.config.period.as_str()),
                ("startTime", &start_timestamp_ms.to_string()),
                ("endTime", &end_timestamp_ms.to_string()),
                ("limit", &self.config.request_limit.to_string()),
            ])
            .send()
            .await
            .map_err(source_error("open_interest_http_request"))?
            .error_for_status()
            .map_err(source_error("open_interest_http_status"))?;
        if response
            .content_length()
            .is_some_and(|length| length > MAX_HTTP_RESPONSE_BYTES as u64)
        {
            return Err(integrity(
                "open_interest_response_too_large",
                "provider response exceeded the one-megabyte bound",
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
            .map_err(source_error("open_interest_read_response"))?
        {
            if body.len().saturating_add(chunk.len()) > MAX_HTTP_RESPONSE_BYTES {
                return Err(integrity(
                    "open_interest_response_too_large",
                    "provider response exceeded the one-megabyte bound",
                ));
            }
            body.extend_from_slice(&chunk);
        }
        let received_at = Utc::now();
        let payload: Value = serde_json::from_slice(&body)
            .map_err(integrity_error("open_interest_decode_response"))?;
        let observations = decode_provider_page(payload, &self.config.symbol, received_at)?;
        if observations.len() > usize::from(self.config.request_limit) {
            return Err(integrity(
                "open_interest_response_limit_exceeded",
                "provider returned more rows than requested",
            ));
        }
        if observations.iter().any(|observation| {
            let timestamp = observation.source_timestamp.timestamp_millis();
            timestamp < start_timestamp_ms || timestamp > end_timestamp_ms
        }) {
            return Err(integrity(
                "open_interest_page_outside_range",
                "provider row escaped the requested timestamp range",
            ));
        }
        Ok(observations)
    }

    async fn persist_observations(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        observations: &[OpenInterestObservation],
    ) -> Result<PersistedBatch, StrategyError> {
        let mut batch = PersistedBatch::default();
        for observation in observations {
            let inserted = sqlx::query_scalar::<_, DateTime<Utc>>(
                r#"
                INSERT INTO market_data.binance_futures_btcusdt_open_interest (
                  source, symbol, source_timestamp, period_seconds,
                  sum_open_interest, sum_open_interest_value,
                  cmc_circulating_supply, provider_available_at, received_at,
                  source_payload, payload_sha256, capture_artifact_id
                ) VALUES (
                  'binance_usd_m_futures', $1, $2, $3, $4, $5, $6, $7, $7, $8, $9, $10
                )
                ON CONFLICT (source_timestamp, symbol, period_seconds) DO NOTHING
                RETURNING source_timestamp
                "#,
            )
            .bind(&self.config.symbol)
            .bind(observation.source_timestamp)
            .bind(PERIOD_SECONDS as i32)
            .bind(observation.sum_open_interest)
            .bind(observation.sum_open_interest_value)
            .bind(observation.cmc_circulating_supply)
            .bind(observation.received_at)
            .bind(&observation.source_payload)
            .bind(&observation.payload_sha256)
            .bind(artifact_id)
            .fetch_optional(&mut **transaction)
            .await
            .map_err(database_error("open_interest_insert_fact"))?;

            if inserted.is_some() {
                batch.record_insert(observation);
            } else {
                let stored = sqlx::query_as::<_, StoredOpenInterest>(
                    r#"
                    SELECT sum_open_interest, sum_open_interest_value,
                           cmc_circulating_supply, payload_sha256
                    FROM market_data.binance_futures_btcusdt_open_interest
                    WHERE source_timestamp = $1 AND symbol = $2 AND period_seconds = $3
                    "#,
                )
                .bind(observation.source_timestamp)
                .bind(&self.config.symbol)
                .bind(PERIOD_SECONDS as i32)
                .fetch_optional(&mut **transaction)
                .await
                .map_err(database_error("open_interest_verify_replay"))?
                .ok_or_else(|| {
                    integrity(
                        "open_interest_conflict_missing",
                        "fact conflicted but no durable row was visible",
                    )
                })?;
                if stored.sum_open_interest != observation.sum_open_interest
                    || stored.sum_open_interest_value != observation.sum_open_interest_value
                    || stored.cmc_circulating_supply != observation.cmc_circulating_supply
                    || stored.payload_sha256 != observation.payload_sha256
                {
                    return Err(integrity(
                        "open_interest_immutable_conflict",
                        format!(
                            "source timestamp {} changed factual values",
                            observation.source_timestamp
                        ),
                    ));
                }
            }
            batch.verified += 1;
        }
        Ok(batch)
    }

    async fn ensure_artifact(
        &self,
        received_at: DateTime<Utc>,
        checkpoint: &OpenInterestCheckpoint,
    ) -> Result<CaptureArtifact, StrategyError> {
        if let Some(open) = self
            .artifacts
            .get_open(STRATEGY_KEY)
            .await
            .map_err(database_error("open_interest_load_artifact"))?
        {
            if open.profile_generation > self.profile_generation {
                return Err(lease_lost("opening a capture artifact"));
            }
            if open.profile_generation == self.profile_generation {
                if open.config_schema_version != CONFIG_SCHEMA_VERSION
                    || open.config_snapshot != self.config_snapshot
                {
                    return Err(integrity(
                        "open_interest_artifact_config_mismatch",
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
            .map_err(database_error("open_interest_begin_artifact_create"))?;
        self.lock_lease(&mut transaction, "creating a capture artifact")
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
                        .last_source_timestamp_ms
                        .map(|timestamp| timestamp.to_string()),
                },
            )
            .await
            .map_err(database_error("open_interest_create_artifact"))?;
        transaction
            .commit()
            .await
            .map_err(database_error("open_interest_commit_artifact_create"))?;
        Ok(artifact)
    }

    async fn seal_open_artifact(&self) -> Result<(), StrategyError> {
        if let Some(open) = self
            .artifacts
            .get_open(STRATEGY_KEY)
            .await
            .map_err(database_error("open_interest_load_artifact_for_seal"))?
        {
            if open.profile_generation > self.profile_generation {
                return Err(lease_lost("draining a newer capture artifact"));
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
            .map_err(database_error("open_interest_begin_artifact_seal"))?;
        if allow_draining_generation {
            self.lock_owned_lease(&mut transaction, "sealing a capture artifact")
                .await?;
        } else {
            self.lock_lease(&mut transaction, "sealing a capture artifact")
                .await?;
        }
        let hashes = sqlx::query_scalar::<_, String>(
            r#"
            SELECT payload_sha256
            FROM market_data.binance_futures_btcusdt_open_interest
            WHERE capture_artifact_id = $1
            ORDER BY source_timestamp, symbol, period_seconds
            "#,
        )
        .bind(artifact.artifact_id)
        .fetch_all(&mut *transaction)
        .await
        .map_err(database_error("open_interest_hash_artifact"))?;
        if i64::try_from(hashes.len()).ok() != Some(artifact.record_count) {
            return Err(integrity(
                "open_interest_artifact_count_mismatch",
                format!(
                    "artifact records {} differ from stored facts {}",
                    artifact.record_count,
                    hashes.len()
                ),
            ));
        }
        let mut digest = Sha256::new();
        for hash in hashes {
            digest.update(hash.as_bytes());
        }
        let content_sha256 = hex_digest(digest.finalize());
        self.artifacts
            .complete_in(
                &mut transaction,
                artifact.artifact_id,
                &content_sha256,
                artifact.end_cursor.as_deref(),
            )
            .await
            .map_err(database_error("open_interest_complete_artifact"))?
            .ok_or_else(|| {
                integrity(
                    "open_interest_artifact_not_open",
                    "capture artifact could not be completed",
                )
            })?;
        transaction
            .commit()
            .await
            .map_err(database_error("open_interest_commit_artifact_seal"))?;
        Ok(())
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
            .map_err(database_error("open_interest_lock_lease"))?;
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
            .map_err(database_error("open_interest_lock_owned_lease"))?;
        if !locked {
            return Err(lease_lost(action));
        }
        Ok(())
    }

    async fn verify_owned_lease(&self) -> Result<(), StrategyError> {
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(database_error("open_interest_begin_owned_lease_check"))?;
        self.lock_owned_lease(&mut transaction, "draining a superseded generation")
            .await?;
        transaction
            .commit()
            .await
            .map_err(database_error("open_interest_commit_owned_lease_check"))
    }

    async fn reconcile_gaps(
        &self,
        checkpoint: &OpenInterestCheckpoint,
    ) -> Result<(), StrategyError> {
        let unresolved = self
            .gaps
            .list_unresolved_limited(STRATEGY_KEY, GAP_REPAIRS_PER_POLL)
            .await
            .map_err(database_error("open_interest_list_gaps"))?;
        if unresolved.is_empty() {
            return Ok(());
        }

        let provider_floor = Utc::now()
            - chrono::Duration::seconds(i64::from(MAX_PROVIDER_HISTORY_PERIODS) * PERIOD_SECONDS);
        for gap in unresolved {
            let (Some(start), Some(end)) = (gap.source_time_start, gap.source_time_end) else {
                continue;
            };
            let mut is_complete = self.gap_is_complete(start, end).await?;
            let mut repair_artifact_id = None;
            if !is_complete {
                if end < provider_floor {
                    self.expire_gap(gap.gap_id).await?;
                    continue;
                }
                if !self.begin_gap_repair(gap.gap_id).await? {
                    continue;
                }
                let observations = self
                    .fetch_range(start.timestamp_millis(), end.timestamp_millis())
                    .await?;
                if !observations.is_empty() {
                    repair_artifact_id = Some(
                        self.persist_repair_observations(checkpoint, &observations)
                            .await?,
                    );
                }
                is_complete = self.gap_is_complete(start, end).await?;
                if !is_complete {
                    continue;
                }
            } else if gap.repair_attempts == 0 && !self.begin_gap_repair(gap.gap_id).await? {
                continue;
            }

            let artifact = match self
                .artifacts
                .get_open(STRATEGY_KEY)
                .await
                .map_err(database_error("open_interest_load_repair_artifact"))?
            {
                Some(artifact) => artifact,
                None => self.ensure_artifact(Utc::now(), checkpoint).await?,
            };
            if artifact.profile_generation != self.profile_generation {
                return Err(lease_lost("completing a gap repair artifact"));
            }
            if repair_artifact_id.is_some_and(|expected| expected != artifact.artifact_id) {
                return Err(integrity(
                    "open_interest_repair_artifact_changed",
                    "open capture artifact changed during a gap repair",
                ));
            }
            self.seal_and_complete_gap(gap.gap_id, &artifact).await?;
        }
        Ok(())
    }

    async fn begin_gap_repair(&self, gap_id: Uuid) -> Result<bool, StrategyError> {
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(database_error("open_interest_begin_gap_transaction"))?;
        self.lock_lease(&mut transaction, "beginning a gap repair")
            .await?;
        let repairing = self
            .gaps
            .begin_repair_in(&mut transaction, gap_id)
            .await
            .map_err(database_error("open_interest_begin_gap_repair"))?
            .is_some();
        transaction
            .commit()
            .await
            .map_err(database_error("open_interest_commit_gap_begin"))?;
        Ok(repairing)
    }

    async fn seal_and_complete_gap(
        &self,
        gap_id: Uuid,
        artifact: &CaptureArtifact,
    ) -> Result<(), StrategyError> {
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(database_error("open_interest_begin_gap_completion"))?;
        self.lock_lease(&mut transaction, "completing a gap repair")
            .await?;
        let hashes = sqlx::query_scalar::<_, String>(
            r#"
            SELECT payload_sha256
            FROM market_data.binance_futures_btcusdt_open_interest
            WHERE capture_artifact_id = $1
            ORDER BY source_timestamp, symbol, period_seconds
            "#,
        )
        .bind(artifact.artifact_id)
        .fetch_all(&mut *transaction)
        .await
        .map_err(database_error("open_interest_hash_repair_artifact"))?;
        if i64::try_from(hashes.len()).ok() != Some(artifact.record_count) {
            return Err(integrity(
                "open_interest_repair_artifact_count_mismatch",
                format!(
                    "artifact records {} differ from stored facts {}",
                    artifact.record_count,
                    hashes.len()
                ),
            ));
        }
        let mut digest = Sha256::new();
        for hash in hashes {
            digest.update(hash.as_bytes());
        }
        let content_sha256 = hex_digest(digest.finalize());
        self.artifacts
            .complete_in(
                &mut transaction,
                artifact.artifact_id,
                &content_sha256,
                artifact.end_cursor.as_deref(),
            )
            .await
            .map_err(database_error("open_interest_complete_repair_artifact"))?
            .ok_or_else(|| {
                integrity(
                    "open_interest_repair_artifact_not_open",
                    "repair capture artifact could not be completed",
                )
            })?;
        self.gaps
            .mark_repaired_in(
                &mut transaction,
                gap_id,
                artifact.artifact_id,
                "provider_interval_recovered",
                Some("all five-minute source timestamps are now durably present"),
            )
            .await
            .map_err(database_error("open_interest_complete_gap_repair"))?
            .ok_or_else(|| {
                integrity(
                    "open_interest_gap_repair_race",
                    "open-interest gap could not be marked repaired",
                )
            })?;
        transaction
            .commit()
            .await
            .map_err(database_error("open_interest_commit_gap_completion"))?;
        Ok(())
    }

    async fn expire_gap(&self, gap_id: Uuid) -> Result<(), StrategyError> {
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(database_error("open_interest_begin_gap_expiry"))?;
        self.lock_lease(&mut transaction, "expiring a gap beyond provider retention")
            .await?;
        self.gaps
            .mark_unrecoverable_in(
                &mut transaction,
                gap_id,
                "provider_retention_elapsed",
                Some("Binance no longer exposes this interval through the one-month endpoint"),
            )
            .await
            .map_err(database_error("open_interest_expire_gap"))?
            .ok_or_else(|| {
                integrity(
                    "open_interest_gap_resolution_race",
                    "open-interest gap could not be marked unrecoverable",
                )
            })?;
        transaction
            .commit()
            .await
            .map_err(database_error("open_interest_commit_gap_expiry"))?;
        Ok(())
    }

    async fn persist_repair_observations(
        &self,
        checkpoint: &OpenInterestCheckpoint,
        observations: &[OpenInterestObservation],
    ) -> Result<Uuid, StrategyError> {
        let received_at = observations
            .iter()
            .map(|observation| observation.received_at)
            .max()
            .expect("nonempty repair batch has a receipt timestamp");
        let artifact = self.ensure_artifact(received_at, checkpoint).await?;
        let mut transaction = self
            .pool
            .begin()
            .await
            .map_err(database_error("open_interest_repair_begin_transaction"))?;
        let persisted = self
            .persist_observations(&mut transaction, artifact.artifact_id, observations)
            .await?;
        if persisted.inserted > 0 {
            self.artifacts
                .record_batch_in(
                    &mut transaction,
                    artifact.artifact_id,
                    &ArtifactBatch {
                        inserted_record_count: persisted.inserted,
                        minimum_source_timestamp: persisted.minimum_inserted_source,
                        maximum_source_timestamp: persisted.maximum_inserted_source,
                        minimum_received_at: persisted.minimum_inserted_received,
                        maximum_received_at: persisted.maximum_inserted_received,
                        start_cursor: persisted
                            .minimum_inserted_source
                            .map(|timestamp| timestamp.timestamp_millis().to_string()),
                        end_cursor: persisted
                            .maximum_inserted_source
                            .map(|timestamp| timestamp.timestamp_millis().to_string()),
                    },
                )
                .await
                .map_err(database_error("open_interest_record_repair_artifact"))?
                .ok_or_else(|| {
                    integrity(
                        "open_interest_repair_artifact_closed",
                        "repair capture artifact is not open",
                    )
                })?;
        }
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
                    checkpoint: serde_json::to_value(checkpoint)
                        .map_err(integrity_error("open_interest_encode_repair_checkpoint"))?,
                    last_source_event_at: maximum_source_timestamp,
                    last_provider_available_at: Some(received_at),
                    source_watermark: maximum_source_timestamp,
                    availability_watermark: Some(received_at),
                },
            )
            .await
            .map_err(database_error("open_interest_record_repair_progress"))?;
        if !advanced {
            return Err(lease_lost("committing open-interest repair progress"));
        }
        transaction
            .commit()
            .await
            .map_err(database_error("open_interest_commit_repair"))?;
        Ok(artifact.artifact_id)
    }

    async fn gap_is_complete(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<bool, StrategyError> {
        sqlx::query_scalar::<_, bool>(
            r#"
            SELECT NOT EXISTS (
              SELECT 1
              FROM generate_series($1::timestamptz, $2::timestamptz, INTERVAL '5 minutes') AS expected(source_timestamp)
              LEFT JOIN market_data.binance_futures_btcusdt_open_interest AS facts
                ON facts.source_timestamp = expected.source_timestamp
               AND facts.symbol = 'BTCUSDT'
               AND facts.period_seconds = 300
              WHERE facts.source_timestamp IS NULL
            )
            "#,
        )
        .bind(start)
        .bind(end)
        .fetch_one(&self.pool)
        .await
        .map_err(database_error("open_interest_verify_gap"))
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
    fn record_insert(&mut self, observation: &OpenInterestObservation) {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourceGap {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
}

fn find_gaps(
    observations: &[OpenInterestObservation],
    checkpoint_timestamp_ms: Option<i64>,
) -> Vec<SourceGap> {
    let mut gaps = Vec::new();
    if let Some(checkpoint_timestamp) = checkpoint_timestamp_ms {
        let expected_timestamp = checkpoint_timestamp.saturating_add(PERIOD_SECONDS * 1_000);
        if let Some(first_new) = observations
            .iter()
            .find(|observation| {
                observation.source_timestamp.timestamp_millis() > checkpoint_timestamp
            })
            .map(|observation| observation.source_timestamp)
        {
            if first_new.timestamp_millis() > expected_timestamp {
                let start = Utc
                    .timestamp_millis_opt(expected_timestamp)
                    .single()
                    .expect("aligned checkpoint timestamp remains representable");
                gaps.push(SourceGap {
                    start,
                    end: first_new - chrono::Duration::seconds(PERIOD_SECONDS),
                });
            }
        }
    }
    gaps.extend(observations.windows(2).filter_map(|pair| {
        let expected = pair[0].source_timestamp + chrono::Duration::seconds(PERIOD_SECONDS);
        (pair[1].source_timestamp > expected).then_some(SourceGap {
            start: expected,
            end: pair[1].source_timestamp - chrono::Duration::seconds(PERIOD_SECONDS),
        })
    }));
    gaps.sort_unstable_by_key(|gap| gap.start);
    gaps.dedup();
    gaps
}

fn decode_provider_page(
    payload: Value,
    expected_symbol: &str,
    received_at: DateTime<Utc>,
) -> Result<Vec<OpenInterestObservation>, StrategyError> {
    let rows = payload.as_array().ok_or_else(|| {
        integrity(
            "open_interest_invalid_payload",
            "provider response must be a JSON array",
        )
    })?;
    let mut observations = Vec::with_capacity(rows.len());
    let mut previous_timestamp = None;

    for source_payload in rows {
        let row: ProviderOpenInterest = serde_json::from_value(source_payload.clone())
            .map_err(integrity_error("open_interest_invalid_payload"))?;
        if row.symbol != expected_symbol {
            return Err(integrity(
                "open_interest_symbol_mismatch",
                format!("expected {expected_symbol}, received {}", row.symbol),
            ));
        }
        if row.timestamp.rem_euclid(PERIOD_SECONDS * 1_000) != 0 {
            return Err(integrity(
                "open_interest_unaligned_timestamp",
                format!("timestamp {} is not five-minute aligned", row.timestamp),
            ));
        }
        if previous_timestamp.is_some_and(|previous| row.timestamp <= previous) {
            return Err(integrity(
                "open_interest_nonmonotonic_page",
                "provider timestamps must be strictly increasing",
            ));
        }
        let source_timestamp = Utc
            .timestamp_millis_opt(row.timestamp)
            .single()
            .ok_or_else(|| {
                integrity(
                    "open_interest_timestamp_out_of_range",
                    format!("timestamp {} is outside the supported range", row.timestamp),
                )
            })?;
        let sum_open_interest =
            parse_nonnegative_decimal("sumOpenInterest", &row.sum_open_interest)?;
        let sum_open_interest_value =
            parse_nonnegative_decimal("sumOpenInterestValue", &row.sum_open_interest_value)?;
        let cmc_circulating_supply = row
            .cmc_circulating_supply
            .as_deref()
            .map(|value| parse_nonnegative_decimal("CMCCirculatingSupply", value))
            .transpose()?;
        let payload_sha256 = canonical_payload_sha256(
            &row.symbol,
            row.timestamp,
            sum_open_interest,
            sum_open_interest_value,
            cmc_circulating_supply,
        );
        observations.push(OpenInterestObservation {
            source_timestamp,
            received_at,
            sum_open_interest,
            sum_open_interest_value,
            cmc_circulating_supply,
            source_payload: source_payload.clone(),
            payload_sha256,
        });
        previous_timestamp = Some(row.timestamp);
    }
    Ok(observations)
}

fn canonical_payload_sha256(
    symbol: &str,
    timestamp: i64,
    open_interest: Decimal,
    open_interest_value: Decimal,
    circulating_supply: Option<Decimal>,
) -> String {
    let canonical = json!({
        "cmc_circulating_supply": circulating_supply.map(|value| value.normalize().to_string()),
        "sum_open_interest": open_interest.normalize().to_string(),
        "sum_open_interest_value": open_interest_value.normalize().to_string(),
        "symbol": symbol,
        "timestamp": timestamp,
    });
    let encoded = serde_json::to_vec(&canonical).expect("canonical JSON values serialize");
    hex_digest(Sha256::digest(encoded))
}

fn parse_nonnegative_decimal(name: &str, value: &str) -> Result<Decimal, StrategyError> {
    let parsed = value.parse::<Decimal>().map_err(|error| {
        integrity(
            "open_interest_invalid_decimal",
            format!("{name} is not a decimal: {error}"),
        )
    })?;
    if parsed < Decimal::ZERO {
        return Err(integrity(
            "open_interest_negative_value",
            format!("{name} must be nonnegative"),
        ));
    }
    Ok(parsed)
}

fn validate_checkpoint(checkpoint: &OpenInterestCheckpoint) -> Result<(), StrategyFactoryError> {
    if let Some(timestamp) = checkpoint.last_source_timestamp_ms {
        if timestamp < 0 || timestamp.rem_euclid(PERIOD_SECONDS * 1_000) != 0 {
            return Err(StrategyFactoryError::Construction(
                "open-interest checkpoint must be nonnegative and five-minute aligned".to_owned(),
            ));
        }
        let maximum_plausible = Utc::now()
            .timestamp_millis()
            .saturating_add(PERIOD_SECONDS * 1_000);
        if timestamp > maximum_plausible {
            return Err(StrategyFactoryError::Construction(
                "open-interest checkpoint is implausibly in the future".to_owned(),
            ));
        }
    }
    Ok(())
}

fn decode_config(config: &Value) -> Result<BinanceFuturesOpenInterestConfig, StrategyFactoryError> {
    let config: BinanceFuturesOpenInterestConfig = serde_json::from_value(config.clone())
        .map_err(|error| invalid_config(format!("failed to decode config: {error}")))?;
    config.validate()?;
    Ok(config)
}

fn validate_https_base_url(value: &str) -> Result<(), StrategyFactoryError> {
    let url = Url::parse(value)
        .map_err(|error| invalid_config(format!("rest_base_url is invalid: {error}")))?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid_config(
            "rest_base_url must be an HTTPS origin without credentials, query, or fragment",
        ));
    }
    Ok(())
}

fn floor_period(
    timestamp: DateTime<Utc>,
    period_seconds: i64,
) -> Result<DateTime<Utc>, StrategyError> {
    let seconds = timestamp.timestamp();
    let floored = seconds - seconds.rem_euclid(period_seconds);
    Utc.timestamp_opt(floored, 0).single().ok_or_else(|| {
        integrity(
            "open_interest_time_out_of_range",
            "failed to align timestamp to capture period",
        )
    })
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
    move |error| StrategyError::new(StrategyErrorKind::TransientSource, code, error.to_string())
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
    move |error| StrategyError::new(StrategyErrorKind::Integrity, code, error.to_string())
}

fn integrity(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::Integrity, code, message)
}

fn lease_lost(action: &str) -> StrategyError {
    StrategyError::new(
        StrategyErrorKind::LeaseLost,
        "open_interest_lease_lost",
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
mod tests {
    use chrono::Duration as ChronoDuration;
    use pretty_assertions::assert_eq;

    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/binance/futures_open_interest.json"
        ))
        .expect("valid checked-in fixture")
    }

    #[test]
    fn default_config_is_narrow_and_valid() {
        let config = BinanceFuturesOpenInterestConfig::default();
        config.validate().expect("default config should be valid");
        assert_eq!(config.artifact_window_seconds, 86_400);
        let encoded = serde_json::to_value(config).expect("config serializes");
        assert!(encoded.get("api_key").is_none());
    }

    #[test]
    fn unknown_config_fields_are_rejected() {
        let mut config = serde_json::to_value(BinanceFuturesOpenInterestConfig::default())
            .expect("config serializes");
        config["labels"] = json!(true);
        assert!(decode_config(&config).is_err());
    }

    #[test]
    fn captured_provider_fixture_decodes_exact_values() {
        let received_at = Utc
            .with_ymd_and_hms(2025, 1, 1, 0, 11, 0)
            .single()
            .expect("valid time");
        let rows =
            decode_provider_page(fixture(), "BTCUSDT", received_at).expect("fixture should decode");
        assert_eq!(rows.len(), 3);
        assert_eq!(
            rows[0].source_timestamp,
            Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0)
                .single()
                .expect("valid time")
        );
        assert_eq!(rows[0].sum_open_interest, Decimal::new(123456789, 3));
        assert_eq!(
            rows[0].sum_open_interest_value,
            Decimal::new(120001234567, 2)
        );
        assert_eq!(
            rows[0].cmc_circulating_supply,
            Some(Decimal::new(1987654321, 2))
        );
        assert_eq!(rows[0].received_at, received_at);
        assert_eq!(rows[0].payload_sha256.len(), 64);
    }

    #[test]
    fn provider_page_rejects_regression_and_wrong_symbol() {
        let mut payload = fixture();
        payload[1]["timestamp"] = payload[0]["timestamp"].clone();
        assert!(decode_provider_page(payload, "BTCUSDT", Utc::now()).is_err());

        let mut payload = fixture();
        payload[0]["symbol"] = json!("ETHUSDT");
        assert!(decode_provider_page(payload, "BTCUSDT", Utc::now()).is_err());
    }

    #[test]
    fn interval_gap_is_precisely_bounded() {
        let received_at = Utc::now();
        let first = Utc
            .with_ymd_and_hms(2025, 1, 1, 0, 0, 0)
            .single()
            .expect("valid time");
        let make = |source_timestamp| OpenInterestObservation {
            source_timestamp,
            received_at,
            sum_open_interest: Decimal::ONE,
            sum_open_interest_value: Decimal::ONE,
            cmc_circulating_supply: None,
            source_payload: json!({}),
            payload_sha256: "a".repeat(64),
        };
        let observations = vec![make(first), make(first + ChronoDuration::minutes(15))];
        assert_eq!(
            find_gaps(&observations, None),
            vec![SourceGap {
                start: first + ChronoDuration::minutes(5),
                end: first + ChronoDuration::minutes(10),
            }]
        );
    }

    #[test]
    fn canonical_hash_ignores_decimal_scale_spelling() {
        let first = canonical_payload_sha256(
            "BTCUSDT",
            1_735_689_600_000,
            "1.0".parse().expect("decimal"),
            "2.00".parse().expect("decimal"),
            None,
        );
        let second = canonical_payload_sha256(
            "BTCUSDT",
            1_735_689_600_000,
            "1".parse().expect("decimal"),
            "2".parse().expect("decimal"),
            None,
        );
        assert_eq!(first, second);
    }

    #[test]
    fn checkpoint_rejects_negative_unaligned_and_future_timestamps() {
        for timestamp in [
            -300_000,
            1,
            Utc::now().timestamp_millis().saturating_add(86_400_000),
        ] {
            assert!(validate_checkpoint(&OpenInterestCheckpoint {
                last_source_timestamp_ms: Some(timestamp),
            })
            .is_err());
        }
    }

    #[test]
    fn zero_open_interest_is_a_valid_fact() {
        let payload = json!([{
            "symbol": "BTCUSDT",
            "sumOpenInterest": "0",
            "sumOpenInterestValue": "0.0",
            "timestamp": 1735689600000_i64
        }]);
        let rows = decode_provider_page(payload, "BTCUSDT", Utc::now())
            .expect("zero is a factual nonnegative value");
        assert_eq!(rows[0].sum_open_interest, Decimal::ZERO);
        assert_eq!(rows[0].sum_open_interest_value, Decimal::ZERO);
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
