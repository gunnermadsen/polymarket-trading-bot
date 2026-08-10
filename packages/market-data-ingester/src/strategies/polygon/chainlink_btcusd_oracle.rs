use std::{
    collections::{BTreeMap, BTreeSet},
    str::FromStr,
    time::Duration,
};

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use reqwest::{Client, Url};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sha3::Keccak256;
use sqlx::{FromRow, PgPool, Postgres, QueryBuilder, Transaction};
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::info;
use uuid::Uuid;

use crate::{
    domain::{
        CaptureArtifact, DataGap, IngesterProfile, IngesterStrategy, IngesterStrategyKey,
        StrategyError, StrategyErrorKind,
    },
    persistence::{
        ArtifactBatch, ArtifactRepository, GapRepository, NewCaptureArtifact, NewDataGap,
        ProfileRepository, StrategyDegradation, StrategyProgress,
    },
    runtime::{StrategyFactory, StrategyFactoryError},
};

pub const STRATEGY_KEY: IngesterStrategyKey = IngesterStrategyKey::PolygonChainlinkBtcusdOracle;
pub const CONFIG_SCHEMA_VERSION: i32 = 1;
pub const CHECKPOINT_SCHEMA_VERSION: i32 = 1;

const SOURCE: &str = "chainlink_polygon_data_feed";
const POLYGON_CHAIN_ID: i64 = 137;
const DEFAULT_RPC_URL: &str = "https://polygon-bor-rpc.publicnode.com";
const DEFAULT_ARCHIVE_LOG_RPC_URL: &str = "https://tenderly.rpc.polygon.community";
const DEFAULT_FEED_PROXY_ADDRESS: &str = "0xc907e116054ad103354f2d350fd2514433d57f6f";
const MAX_SUPPORTED_DECIMALS: u32 = 18;
const MAX_AGGREGATOR_PHASES: u16 = 128;
const MAX_RPC_RESPONSE_BYTES: usize = 8 * 1_024 * 1_024;
const MAX_RPC_LOGS_PER_RESPONSE: usize = 20_000;
const BLOCK_HEADER_BATCH_SIZE: usize = 100;
const MAX_INSERT_ROWS: usize = 500;
const MAX_DATABASE_RANGE_ROWS: i64 = 20_001;
const GAP_REPAIRS_PER_POLL: i64 = 4;
const MAX_GAP_REPAIR_ATTEMPTS: i32 = 3;
const MAX_GAP_REPAIR_ROUNDS: u64 = 4_096;
const MAX_GAP_REPAIR_BLOCKS: u64 = 1_000_000;
const MAX_GAP_REPAIR_RPC_REQUESTS: u64 = 64;
const STARTUP_ANCHOR_WINDOW_BLOCKS: u64 = 2_048;
const MAX_STARTUP_ANCHOR_WINDOWS: usize = 64;

const APPROVED_RPC_URLS: [&str; 2] = [DEFAULT_RPC_URL, DEFAULT_ARCHIVE_LOG_RPC_URL];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct PolygonChainlinkBtcusdOracleConfig {
    pub rpc_url: String,
    pub archive_log_rpc_url: String,
    pub feed_proxy_address: String,
    pub poll_interval_seconds: u64,
    pub confirmation_depth: u64,
    pub maximum_block_range: u64,
    pub startup_lookback_blocks: u64,
    pub overlap_blocks: u64,
    pub artifact_window_seconds: i64,
    pub request_timeout_seconds: u64,
}

impl Default for PolygonChainlinkBtcusdOracleConfig {
    fn default() -> Self {
        Self {
            rpc_url: DEFAULT_RPC_URL.to_owned(),
            archive_log_rpc_url: DEFAULT_ARCHIVE_LOG_RPC_URL.to_owned(),
            feed_proxy_address: DEFAULT_FEED_PROXY_ADDRESS.to_owned(),
            poll_interval_seconds: 15,
            confirmation_depth: 128,
            maximum_block_range: 30_000,
            startup_lookback_blocks: 43_200,
            overlap_blocks: 256,
            artifact_window_seconds: 3_600,
            request_timeout_seconds: 30,
        }
    }
}

impl PolygonChainlinkBtcusdOracleConfig {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let mut config = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::InvalidConfiguration(format!(
                "invalid Polygon Chainlink BTC/USD oracle config: {error}"
            ))
        })?;
        config.feed_proxy_address = config.feed_proxy_address.to_ascii_lowercase();
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), StrategyFactoryError> {
        validate_public_rpc_url("rpc_url", &self.rpc_url)?;
        validate_public_rpc_url("archive_log_rpc_url", &self.archive_log_rpc_url)?;
        if self.rpc_url.trim_end_matches('/') == self.archive_log_rpc_url.trim_end_matches('/') {
            return Err(invalid_config(
                "rpc_url and archive_log_rpc_url must identify independent providers",
            ));
        }
        if self.feed_proxy_address != DEFAULT_FEED_PROXY_ADDRESS {
            return Err(invalid_config(
                "feed_proxy_address must be the Polygon Chainlink BTC/USD proxy",
            ));
        }
        validate_address(&self.feed_proxy_address)
            .map_err(|message| invalid_config(format!("feed_proxy_address {message}")))?;
        if !(5..=300).contains(&self.poll_interval_seconds) {
            return Err(invalid_config(
                "poll_interval_seconds must be between 5 and 300",
            ));
        }
        if !(1..=2_048).contains(&self.confirmation_depth) {
            return Err(invalid_config(
                "confirmation_depth must be between 1 and 2048",
            ));
        }
        if !(1..=30_000).contains(&self.maximum_block_range) {
            return Err(invalid_config(
                "maximum_block_range must be between 1 and 30000",
            ));
        }
        if !(1..=1_000_000).contains(&self.startup_lookback_blocks) {
            return Err(invalid_config(
                "startup_lookback_blocks must be between 1 and 1000000",
            ));
        }
        if self.overlap_blocks < self.confirmation_depth
            || self.overlap_blocks > self.startup_lookback_blocks
            || self.overlap_blocks > self.maximum_block_range
        {
            return Err(invalid_config(
                "overlap_blocks must be at least confirmation_depth and no greater than startup_lookback_blocks or maximum_block_range",
            ));
        }
        if !(60..=86_400).contains(&self.artifact_window_seconds) {
            return Err(invalid_config(
                "artifact_window_seconds must be between 60 and 86400",
            ));
        }
        if !(1..=60).contains(&self.request_timeout_seconds) {
            return Err(invalid_config(
                "request_timeout_seconds must be between 1 and 60",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
struct OracleCheckpoint {
    last_finalized_block: Option<u64>,
    last_finalized_block_hash: Option<String>,
}

impl OracleCheckpoint {
    fn from_value(value: &Value) -> Result<Self, StrategyFactoryError> {
        let checkpoint = serde_json::from_value::<Self>(value.clone()).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "invalid Polygon oracle checkpoint: {error}"
            ))
        })?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    fn validate(&self) -> Result<(), StrategyFactoryError> {
        match (
            self.last_finalized_block,
            self.last_finalized_block_hash.as_deref(),
        ) {
            (None, None) => Ok(()),
            (Some(_), Some(hash)) => {
                validate_hash(hash).map_err(|message| {
                    StrategyFactoryError::Construction(format!(
                        "invalid Polygon oracle checkpoint hash: {message}"
                    ))
                })?;
                if hash != hash.to_ascii_lowercase() {
                    return Err(StrategyFactoryError::Construction(
                        "Polygon oracle checkpoint hash must be lowercase".to_owned(),
                    ));
                }
                Ok(())
            }
            _ => Err(StrategyFactoryError::Construction(
                "Polygon oracle checkpoint block and hash must both be present or absent"
                    .to_owned(),
            )),
        }
    }

    fn to_value(&self) -> Result<Value, StrategyError> {
        serde_json::to_value(self).map_err(|error| {
            integrity_error("polygon_oracle_checkpoint_encode_failed", error.to_string())
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PolygonChainlinkBtcusdOracleFactory;

impl StrategyFactory for PolygonChainlinkBtcusdOracleFactory {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    fn config_schema_version(&self) -> i32 {
        CONFIG_SCHEMA_VERSION
    }

    fn validate_config(&self, config: &Value) -> Result<(), StrategyFactoryError> {
        PolygonChainlinkBtcusdOracleConfig::from_value(config).map(|_| ())
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
                "Polygon oracle config schema must be {CONFIG_SCHEMA_VERSION}"
            )));
        }
        if profile.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(StrategyFactoryError::Construction(format!(
                "Polygon oracle checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}"
            )));
        }
        if profile.desired_generation <= 0 {
            return Err(StrategyFactoryError::Construction(
                "Polygon oracle profile generation must be positive".to_owned(),
            ));
        }

        let config = PolygonChainlinkBtcusdOracleConfig::from_value(&profile.config)?;
        let config_snapshot = serde_json::to_value(&config).map_err(|error| {
            StrategyFactoryError::Construction(format!(
                "failed to encode effective Polygon oracle config: {error}"
            ))
        })?;
        let checkpoint = OracleCheckpoint::from_value(&profile.checkpoint)?;
        let lease_owner = profile.lease_owner.clone().ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease owner is missing".to_owned())
        })?;
        let lease_token = profile.lease_token.ok_or_else(|| {
            StrategyFactoryError::Construction("profile lease token is missing".to_owned())
        })?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(config.request_timeout_seconds))
            .user_agent("capitonic-market-data-ingester/0.1")
            .build()
            .map_err(|error| StrategyFactoryError::Construction(error.to_string()))?;

        Ok(Box::new(PolygonChainlinkBtcusdOracleStrategy {
            config,
            config_snapshot,
            profile_generation: profile.desired_generation,
            lease_owner,
            lease_token,
            checkpoint,
            client,
            pool,
        }))
    }
}

struct PolygonChainlinkBtcusdOracleStrategy {
    config: PolygonChainlinkBtcusdOracleConfig,
    config_snapshot: Value,
    profile_generation: i64,
    lease_owner: String,
    lease_token: Uuid,
    checkpoint: OracleCheckpoint,
    client: Client,
    pool: PgPool,
}

struct OracleRunState {
    checkpoint: OracleCheckpoint,
    artifact: Option<CaptureArtifact>,
    feed: Option<FeedMetadata>,
}

#[derive(Debug, Clone)]
struct FeedMetadata {
    decimals: u32,
    phase_count: u16,
    aggregators: BTreeMap<String, u16>,
}

#[derive(Debug, Clone, PartialEq)]
struct OracleRound {
    chain_id: i64,
    feed_proxy_address: String,
    aggregator_address: String,
    phase_id: i32,
    aggregator_round_id: i64,
    source_timestamp: DateTime<Utc>,
    block_timestamp: DateTime<Utc>,
    answer_raw: Decimal,
    price: Decimal,
    decimals: i32,
    block_number: i64,
    block_hash: String,
    transaction_hash: String,
    log_index: i32,
    provider_available_at: DateTime<Utc>,
    received_at: DateTime<Utc>,
    source_payload: Value,
    payload_sha256: String,
}

impl OracleRound {
    fn identity(&self) -> (String, i32) {
        (self.transaction_hash.clone(), self.log_index)
    }

    fn cursor(&self) -> String {
        format!(
            "block:{}:tx:{}:log:{}",
            self.block_number, self.transaction_hash, self.log_index
        )
    }

    fn factual_eq(&self, stored: &StoredOracleRound) -> bool {
        self.chain_id == stored.chain_id
            && self.feed_proxy_address == stored.feed_proxy_address
            && self.aggregator_address == stored.aggregator_address
            && self.phase_id == stored.phase_id
            && self.aggregator_round_id == stored.aggregator_round_id
            && self.source_timestamp == stored.source_timestamp
            && self.block_timestamp == stored.block_timestamp
            && self.answer_raw == stored.answer_raw
            && self.price == stored.price
            && self.decimals == stored.decimals
            && self.block_number == stored.block_number
            && self.block_hash == stored.block_hash
            && self.transaction_hash == stored.transaction_hash
            && self.log_index == stored.log_index
            && self.provider_available_at == stored.provider_available_at
            && self.source_payload == stored.source_payload
            && self.payload_sha256 == stored.payload_sha256
    }
}

#[derive(Debug, Clone, FromRow)]
struct StoredOracleRound {
    chain_id: i64,
    feed_proxy_address: String,
    aggregator_address: String,
    phase_id: i32,
    aggregator_round_id: i64,
    source_timestamp: DateTime<Utc>,
    block_timestamp: DateTime<Utc>,
    answer_raw: Decimal,
    price: Decimal,
    decimals: i32,
    block_number: i64,
    block_hash: String,
    transaction_hash: String,
    log_index: i32,
    provider_available_at: DateTime<Utc>,
    source_payload: Value,
    payload_sha256: String,
}

impl StoredOracleRound {
    fn identity(&self) -> (String, i32) {
        (self.transaction_hash.clone(), self.log_index)
    }
}

#[derive(Debug, Clone, FromRow)]
struct StoredRangeIdentity {
    transaction_hash: String,
    log_index: i32,
    block_number: i64,
    block_hash: String,
    payload_sha256: String,
}

#[derive(Debug, Clone, FromRow)]
struct RoundBoundary {
    phase_id: i32,
    aggregator_round_id: i64,
    source_timestamp: DateTime<Utc>,
    block_number: i64,
}

#[derive(Debug, Clone, FromRow)]
struct ArtifactChecksumRow {
    source_timestamp: DateTime<Utc>,
    block_number: i64,
    transaction_hash: String,
    log_index: i32,
    payload_sha256: String,
}

#[derive(Debug, Clone, FromRow)]
struct UnresolvedRoundGap {
    gap_id: Uuid,
    start_cursor: Option<String>,
    end_cursor: Option<String>,
}

#[derive(Debug, Clone, FromRow)]
struct RoundGapNeighbor {
    aggregator_round_id: i64,
    block_number: i64,
    block_hash: String,
}

#[derive(Debug, Clone)]
struct BlockHeader {
    number: u64,
    hash: String,
    timestamp: DateTime<Utc>,
}

#[derive(Debug)]
struct RpcResult {
    value: Value,
    received_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
struct RpcEnvelope {
    #[serde(default)]
    id: Option<Value>,
    result: Option<Value>,
    error: Option<RpcError>,
}

#[derive(Debug, Deserialize)]
struct RpcError {
    code: i64,
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RpcBlock {
    number: Option<String>,
    hash: Option<String>,
    timestamp: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RpcLog {
    address: String,
    topics: Vec<String>,
    data: String,
    block_number: String,
    block_hash: String,
    transaction_hash: String,
    log_index: String,
    #[serde(default)]
    block_timestamp: Option<String>,
    #[serde(default)]
    removed: bool,
}

#[derive(Debug)]
struct LogBatch {
    logs: Vec<RpcLog>,
    received_at: DateTime<Utc>,
}

#[derive(Debug)]
struct DecodedLogBatch {
    rounds: Vec<OracleRound>,
    received_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct GapSignal {
    gap_kind: &'static str,
    reason_code: &'static str,
    reason_message: String,
    source_time_start: Option<DateTime<Utc>>,
    source_time_end: Option<DateTime<Utc>>,
    start_cursor: String,
    end_cursor: String,
}

#[derive(Debug)]
struct ArtifactSeal {
    content_sha256: String,
    end_cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RoundGapRepair {
    phase_id: i32,
    start_round: i64,
    end_round: i64,
    from_block: u64,
    to_block: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoundGapOutcome {
    Repaired,
    Retry,
    Unrecoverable,
}

#[async_trait]
impl IngesterStrategy for PolygonChainlinkBtcusdOracleStrategy {
    fn key(&self) -> IngesterStrategyKey {
        STRATEGY_KEY
    }

    async fn run(&self, shutdown: CancellationToken) -> Result<(), StrategyError> {
        self.validate_durable_cursor().await?;
        let mut state = OracleRunState {
            checkpoint: self.checkpoint.clone(),
            artifact: None,
            feed: None,
        };
        let mut ticker =
            tokio::time::interval(Duration::from_secs(self.config.poll_interval_seconds));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {
                    self.seal_artifact(&mut state, true).await?;
                    return Ok(());
                }
                _ = ticker.tick() => {
                    match self.capture_finalized_head(&mut state).await {
                        Ok(()) => {}
                        Err(error) if error.kind == StrategyErrorKind::LeaseLost => {
                            return self.finish_owned_drain(&mut state).await;
                        }
                        Err(error) => return Err(error),
                    }
                }
            }
        }
    }
}

impl PolygonChainlinkBtcusdOracleStrategy {
    async fn validate_durable_cursor(&self) -> Result<(), StrategyError> {
        let durable_max = sqlx::query_scalar::<_, i64>(
            r#"
            SELECT block_number
            FROM market_data.polygon_chainlink_btcusd_oracle_rounds
            WHERE chain_id = 137 AND feed_proxy_address = $1
            ORDER BY block_number DESC, source_timestamp DESC, log_index DESC
            LIMIT 1
            "#,
        )
        .bind(&self.config.feed_proxy_address)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| database_error("polygon_oracle_durable_cursor_read_failed", error))?;
        validate_durable_cursor_state(durable_max, self.checkpoint.last_finalized_block)
    }

    async fn capture_finalized_head(
        &self,
        state: &mut OracleRunState,
    ) -> Result<(), StrategyError> {
        let head = self.block_number().await?;
        if head < self.config.confirmation_depth {
            return Ok(());
        }
        let finalized_head = head - self.config.confirmation_depth;
        let (finalized_header, finalized_header_received_at) =
            self.block_by_number_with_receipt(finalized_head).await?;

        if let (Some(checkpoint_block), Some(checkpoint_hash)) = (
            state.checkpoint.last_finalized_block,
            state.checkpoint.last_finalized_block_hash.as_deref(),
        ) {
            if finalized_head < checkpoint_block {
                let signal = GapSignal {
                    gap_kind: "chain_reorganization",
                    reason_code: "polygon_finalized_head_regressed",
                    reason_message: format!(
                        "finalized Polygon head regressed from {checkpoint_block} to {finalized_head}"
                    ),
                    source_time_start: None,
                    source_time_end: None,
                    start_cursor: format!("block:{checkpoint_block}:{checkpoint_hash}"),
                    end_cursor: format!("block:{finalized_head}:{}", finalized_header.hash),
                };
                return self.record_fatal_gap(signal).await;
            }
            let canonical = if checkpoint_block == finalized_head {
                finalized_header.clone()
            } else {
                self.block_by_number(checkpoint_block).await?
            };
            if canonical.hash != checkpoint_hash {
                let signal = GapSignal {
                    gap_kind: "chain_reorganization",
                    reason_code: "polygon_finalized_checkpoint_reorg",
                    reason_message: format!(
                        "finalized block {checkpoint_block} changed from {checkpoint_hash} to {}",
                        canonical.hash
                    ),
                    source_time_start: Some(canonical.timestamp),
                    source_time_end: Some(canonical.timestamp),
                    start_cursor: format!("block:{checkpoint_block}:{checkpoint_hash}"),
                    end_cursor: format!("block:{checkpoint_block}:{}", canonical.hash),
                };
                return self.record_fatal_gap(signal).await;
            }
        }

        let feed = self.load_feed_metadata(state.feed.as_ref()).await?;
        state.feed = Some(feed.clone());
        self.reconcile_round_gaps(state, &feed, finalized_head)
            .await?;
        self.seal_elapsed_artifact(state, Utc::now()).await?;
        if state.checkpoint.last_finalized_block == Some(finalized_head) {
            return Ok(());
        }
        let mut from_block = scan_start(
            state.checkpoint.last_finalized_block,
            finalized_head,
            self.config.overlap_blocks,
            self.config.startup_lookback_blocks,
        );
        let (mut startup_boundary, mut startup_proof_received_at) =
            if empty_scan_may_advance(&state.checkpoint) {
                (None, None)
            } else {
                let (boundary, received_at) =
                    self.startup_round_boundary(&feed, from_block).await?;
                (Some(boundary), Some(received_at))
            };

        while from_block <= finalized_head {
            let to_block = from_block
                .saturating_add(self.config.maximum_block_range.saturating_sub(1))
                .min(finalized_head);
            let log_batch = self.fetch_logs(&feed, from_block, to_block).await?;
            if let Some(proof_received_at) = startup_proof_received_at.as_mut() {
                *proof_received_at = (*proof_received_at).max(log_batch.received_at);
            }
            let (checkpoint_header, checkpoint_received_at) = if to_block == finalized_head {
                (finalized_header.clone(), finalized_header_received_at)
            } else {
                self.block_by_number_with_receipt(to_block).await?
            };
            let mut decoded = self
                .decode_log_batch(log_batch, &feed, from_block, to_block)
                .await?;
            decoded.received_at = decoded.received_at.max(checkpoint_received_at);
            if let Some(proof_received_at) = startup_proof_received_at.as_mut() {
                *proof_received_at = (*proof_received_at).max(decoded.received_at);
                decoded.received_at = *proof_received_at;
            }
            for round in &mut decoded.rounds {
                round.received_at = round.received_at.max(decoded.received_at);
            }
            if startup_boundary.is_some() && decoded.rounds.is_empty() {
                if to_block == u64::MAX {
                    break;
                }
                from_block = to_block + 1;
                continue;
            }
            self.verify_canonical_overlap(from_block, to_block, &decoded.rounds)
                .await?;
            let round_gaps = self
                .detect_round_gaps(from_block, &decoded.rounds, startup_boundary.as_ref())
                .await?;
            self.persist_scanned_chunk(state, decoded.rounds, round_gaps, checkpoint_header)
                .await?;
            startup_boundary = None;
            startup_proof_received_at = None;

            if to_block == u64::MAX {
                break;
            }
            from_block = to_block + 1;
        }
        if startup_boundary.is_some() {
            return Err(source_error_value(
                "polygon_oracle_startup_coverage_unverified",
                "entire initial Polygon oracle scan was empty; retrying without checkpoint advancement",
            ));
        }
        Ok(())
    }

    async fn reconcile_round_gaps(
        &self,
        state: &mut OracleRunState,
        feed: &FeedMetadata,
        finalized_head: u64,
    ) -> Result<(), StrategyError> {
        let gaps = sqlx::query_as::<_, UnresolvedRoundGap>(
            r#"
            SELECT gap_id, start_cursor, end_cursor
            FROM ingester.data_gaps
            WHERE strategy_key = $1
              AND reason_code = 'chainlink_aggregator_round_gap'
              AND status IN ('open', 'repairing')
            ORDER BY (status = 'open') DESC,
                     repair_started_at ASC NULLS FIRST,
                     detected_at,
                     gap_id
            LIMIT $2
            "#,
        )
        .bind(STRATEGY_KEY.as_str())
        .bind(GAP_REPAIRS_PER_POLL)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| database_error("polygon_oracle_gap_list_failed", error))?;

        for gap in gaps {
            let repair =
                parse_round_gap_cursors(gap.start_cursor.as_deref(), gap.end_cursor.as_deref())?;
            if !repair.is_resource_bounded(self.config.maximum_block_range) {
                self.mark_round_gap_unrecoverable(
                    gap.gap_id,
                    "repair_resource_bound_exceeded",
                    "round gap exceeds the configured bounded repair envelope",
                )
                .await?;
                continue;
            }
            if repair.to_block > finalized_head {
                continue;
            }

            if self.round_gap_is_complete(&repair).await? {
                let received_at = self.validate_round_gap_bracket(feed, &repair).await?;
                let repairing = self.begin_round_gap_repair(gap.gap_id).await?;
                self.finish_round_gap_repair(state, &repairing, &repair, Vec::new(), received_at)
                    .await?;
                continue;
            }

            // A repair attempt is counted only after a complete bounded archive query
            // succeeds. Transient transport failures therefore never terminalize a gap.
            let decoded = self.fetch_round_gap(feed, &repair).await?;
            let repairing = self.begin_round_gap_repair(gap.gap_id).await?;
            self.finish_round_gap_repair(
                state,
                &repairing,
                &repair,
                decoded.rounds,
                decoded.received_at,
            )
            .await?;
        }
        Ok(())
    }

    async fn fetch_round_gap(
        &self,
        feed: &FeedMetadata,
        repair: &RoundGapRepair,
    ) -> Result<DecodedLogBatch, StrategyError> {
        let bracket_received_at = self.validate_round_gap_bracket(feed, repair).await?;
        let aggregator = feed
            .aggregators
            .iter()
            .find_map(|(address, phase_id)| {
                (i32::from(*phase_id) == repair.phase_id).then(|| address.clone())
            })
            .ok_or_else(|| {
                integrity_error(
                    "polygon_oracle_repair_phase_missing",
                    format!("phase {} has no finalized aggregator", repair.phase_id),
                )
            })?;
        let addresses = vec![aggregator];
        let mut from_block = repair.from_block;
        let mut maximum_received_at = Some(bracket_received_at);
        let mut unique = BTreeMap::<(String, i32), OracleRound>::new();
        while from_block <= repair.to_block {
            let to_block = from_block
                .saturating_add(self.config.maximum_block_range.saturating_sub(1))
                .min(repair.to_block);
            let batch = self
                .fetch_logs_for_addresses(&addresses, from_block, to_block)
                .await?;
            let decoded = self
                .decode_log_batch(batch, feed, from_block, to_block)
                .await?;
            maximum_received_at = Some(
                maximum_received_at
                    .map(|current: DateTime<Utc>| current.max(decoded.received_at))
                    .unwrap_or(decoded.received_at),
            );
            for round in decoded.rounds.into_iter().filter(|round| {
                round.phase_id == repair.phase_id
                    && round.aggregator_round_id >= repair.start_round
                    && round.aggregator_round_id <= repair.end_round
            }) {
                let identity = round.identity();
                if let Some(existing) = unique.get(&identity) {
                    if existing != &round {
                        return Err(integrity_error(
                            "polygon_oracle_repair_duplicate_conflict",
                            format!(
                                "repair returned conflicting log {}:{}",
                                identity.0, identity.1
                            ),
                        ));
                    }
                } else {
                    unique.insert(identity, round);
                }
            }
            if unique.len() > MAX_GAP_REPAIR_ROUNDS as usize {
                return Err(integrity_error(
                    "polygon_oracle_repair_result_too_large",
                    "round-gap repair exceeded its result bound",
                ));
            }
            if to_block == u64::MAX {
                break;
            }
            from_block = to_block + 1;
        }
        let mut rounds = unique.into_values().collect::<Vec<_>>();
        let final_bracket_received_at = self.validate_round_gap_bracket(feed, repair).await?;
        let received_at = maximum_received_at
            .map(|received_at| received_at.max(final_bracket_received_at))
            .ok_or_else(|| {
                integrity_error(
                    "polygon_oracle_repair_receipt_missing",
                    "bounded round-gap repair issued no RPC request",
                )
            })?;
        for round in &mut rounds {
            round.received_at = round.received_at.max(received_at);
        }
        rounds.sort_by_key(|round| {
            (
                round.block_number,
                round.log_index,
                round.transaction_hash.clone(),
            )
        });
        Ok(DecodedLogBatch {
            rounds,
            received_at,
        })
    }

    async fn validate_round_gap_bracket(
        &self,
        feed: &FeedMetadata,
        repair: &RoundGapRepair,
    ) -> Result<DateTime<Utc>, StrategyError> {
        let previous_round = repair.start_round.checked_sub(1).ok_or_else(|| {
            integrity_error(
                "polygon_oracle_repair_neighbor_invalid",
                "round gap has no positive predecessor",
            )
        })?;
        let following_round = repair.end_round.checked_add(1).ok_or_else(|| {
            integrity_error(
                "polygon_oracle_repair_neighbor_invalid",
                "round gap successor overflowed",
            )
        })?;
        let neighbor_ids = vec![previous_round, following_round];
        let neighbors = sqlx::query_as::<_, RoundGapNeighbor>(
            r#"
            SELECT aggregator_round_id, block_number, block_hash
            FROM market_data.polygon_chainlink_btcusd_oracle_rounds
            WHERE chain_id = 137
              AND feed_proxy_address = $1
              AND phase_id = $2
              AND block_number BETWEEN $3 AND $4
              AND aggregator_round_id = ANY($5::bigint[])
            ORDER BY block_number, log_index, source_timestamp
            LIMIT 3
            "#,
        )
        .bind(&self.config.feed_proxy_address)
        .bind(repair.phase_id)
        .bind(i64::try_from(repair.from_block).map_err(|_| {
            integrity_error(
                "polygon_oracle_repair_block_overflow",
                "repair block overflow",
            )
        })?)
        .bind(i64::try_from(repair.to_block).map_err(|_| {
            integrity_error(
                "polygon_oracle_repair_block_overflow",
                "repair block overflow",
            )
        })?)
        .bind(&neighbor_ids)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| database_error("polygon_oracle_repair_neighbor_read_failed", error))?;
        if neighbors.len() > 2 {
            return Err(integrity_error(
                "polygon_oracle_repair_neighbor_ambiguous",
                "round-gap repair found more than two durable neighboring facts",
            ));
        }
        let mut by_round = BTreeMap::new();
        for neighbor in neighbors {
            if by_round
                .insert(neighbor.aggregator_round_id, neighbor)
                .is_some()
            {
                return Err(integrity_error(
                    "polygon_oracle_repair_neighbor_ambiguous",
                    "round-gap repair found duplicate durable neighboring rounds",
                ));
            }
        }
        let following = by_round.get(&following_round).ok_or_else(|| {
            integrity_error(
                "polygon_oracle_repair_neighbor_missing",
                "round-gap successor was not durable",
            )
        })?;
        if u64::try_from(following.block_number).ok() != Some(repair.to_block) {
            return Err(integrity_error(
                "polygon_oracle_repair_bracket_changed",
                "durable successor no longer matches the persisted repair bracket",
            ));
        }

        let (to_header, to_received_at) =
            self.block_by_number_with_receipt(repair.to_block).await?;
        if following.block_hash != to_header.hash {
            return Err(integrity_error(
                "polygon_oracle_repair_bracket_reorg",
                format!(
                    "canonical block hashes changed across repair bracket {}..={} (phase {}, rounds {}..={})",
                    repair.from_block,
                    repair.to_block,
                    repair.phase_id,
                    repair.start_round,
                    repair.end_round
                ),
            ));
        }
        let from_received_at = if let Some(previous) = by_round.get(&previous_round) {
            if u64::try_from(previous.block_number).ok() != Some(repair.from_block) {
                return Err(integrity_error(
                    "polygon_oracle_repair_bracket_changed",
                    "durable predecessor no longer matches the persisted repair bracket",
                ));
            }
            let (from_header, from_received_at) = if repair.from_block == repair.to_block {
                (to_header.clone(), to_received_at)
            } else {
                self.block_by_number_with_receipt(repair.from_block).await?
            };
            if previous.block_hash != from_header.hash {
                return Err(integrity_error(
                    "polygon_oracle_repair_bracket_reorg",
                    format!(
                        "canonical predecessor block changed across repair bracket {}..={} (phase {}, rounds {}..={})",
                        repair.from_block,
                        repair.to_block,
                        repair.phase_id,
                        repair.start_round,
                        repair.end_round
                    ),
                ));
            }
            from_received_at
        } else {
            let proof_scan_start = repair.from_block.checked_add(1).ok_or_else(|| {
                integrity_error(
                    "polygon_oracle_repair_anchor_block_overflow",
                    "startup proof block cannot be advanced for reconstruction",
                )
            })?;
            let (proof, proof_received_at) =
                self.startup_round_boundary(feed, proof_scan_start).await?;
            if proof.phase_id != repair.phase_id
                || proof.aggregator_round_id != previous_round
                || u64::try_from(proof.block_number).ok() != Some(repair.from_block)
            {
                return Err(integrity_error(
                    "polygon_oracle_repair_anchor_changed",
                    format!(
                        "historical startup proof no longer bounds phase {} round {} at block {}",
                        repair.phase_id, previous_round, repair.from_block
                    ),
                ));
            }
            proof_received_at
        };
        Ok(from_received_at.max(to_received_at))
    }

    async fn begin_round_gap_repair(&self, gap_id: Uuid) -> Result<DataGap, StrategyError> {
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polygon_oracle_gap_begin_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let gap = GapRepository::new(self.pool.clone())
            .begin_repair_in(&mut transaction, gap_id)
            .await
            .map_err(|error| database_error("polygon_oracle_gap_begin_failed", error))?
            .ok_or_else(|| {
                integrity_error(
                    "polygon_oracle_gap_begin_race",
                    format!("round gap {gap_id} was no longer unresolved"),
                )
            })?;
        transaction
            .commit()
            .await
            .map_err(|error| database_error("polygon_oracle_gap_begin_commit_failed", error))?;
        Ok(gap)
    }

    async fn finish_round_gap_repair(
        &self,
        state: &mut OracleRunState,
        gap: &DataGap,
        repair: &RoundGapRepair,
        rounds: Vec<OracleRound>,
        received_at: DateTime<Utc>,
    ) -> Result<(), StrategyError> {
        self.seal_artifact(state, false).await?;
        let mut artifact = self.create_repair_artifact(state, received_at, gap).await?;
        let mut transaction =
            self.pool.begin().await.map_err(|error| {
                database_error("polygon_oracle_repair_transaction_failed", error)
            })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let inserted_identities = self
            .persist_round_facts_in(&mut transaction, artifact.artifact_id, &rounds)
            .await?;
        let inserted = rounds
            .iter()
            .filter(|round| inserted_identities.contains(&round.identity()))
            .collect::<Vec<_>>();
        if !inserted.is_empty() {
            artifact = ArtifactRepository::new(self.pool.clone())
                .record_batch_in(
                    &mut transaction,
                    artifact.artifact_id,
                    &artifact_batch(&inserted),
                )
                .await
                .map_err(|error| {
                    database_error("polygon_oracle_repair_artifact_progress_failed", error)
                })?
                .ok_or_else(|| {
                    integrity_error(
                        "polygon_oracle_repair_artifact_not_open",
                        "repair artifact was not open while recording inserted facts",
                    )
                })?;
        }
        if !rounds.is_empty() {
            let maximum_source = rounds.iter().map(|round| round.source_timestamp).max();
            let maximum_available = rounds.iter().map(|round| round.provider_available_at).max();
            let progressed = ProfileRepository::new(self.pool.clone())
                .record_progress_in(
                    &mut transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                    &StrategyProgress {
                        verified_record_count: rounds.len() as i64,
                        checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
                        checkpoint: state.checkpoint.to_value()?,
                        last_source_event_at: maximum_source,
                        last_provider_available_at: maximum_available,
                        source_watermark: maximum_source,
                        availability_watermark: maximum_available,
                    },
                )
                .await
                .map_err(|error| database_error("polygon_oracle_repair_progress_failed", error))?;
            if !progressed {
                return Err(lease_lost_error());
            }
        }

        let complete = self
            .round_gap_is_complete_in(&mut transaction, repair)
            .await?;
        let seal = self.artifact_seal_in(&mut transaction, &artifact).await?;
        ArtifactRepository::new(self.pool.clone())
            .complete_in(
                &mut transaction,
                artifact.artifact_id,
                &seal.content_sha256,
                seal.end_cursor
                    .as_deref()
                    .or(artifact.end_cursor.as_deref()),
            )
            .await
            .map_err(|error| {
                database_error("polygon_oracle_repair_artifact_complete_failed", error)
            })?
            .ok_or_else(|| {
                integrity_error(
                    "polygon_oracle_repair_artifact_completion_race",
                    "repair artifact could not be completed",
                )
            })?;
        let gaps = GapRepository::new(self.pool.clone());
        match classify_round_gap_outcome(complete, gap.repair_attempts) {
            RoundGapOutcome::Repaired => {
                gaps.mark_repaired_in(
                    &mut transaction,
                    gap.gap_id,
                    artifact.artifact_id,
                    "canonical_rounds_recovered",
                    Some("all bounded Chainlink aggregator rounds are durably verified"),
                )
                .await
                .map_err(|error| database_error("polygon_oracle_gap_complete_failed", error))?
                .ok_or_else(|| {
                    integrity_error(
                        "polygon_oracle_gap_completion_race",
                        "round gap was not repairing during atomic completion",
                    )
                })?;
            }
            RoundGapOutcome::Unrecoverable => {
                let evidence = format!(
                    "{} successful complete finalized archive scans omitted phase {} rounds {}..={} within canonically verified block bracket {}..={}",
                    gap.repair_attempts,
                    repair.phase_id,
                    repair.start_round,
                    repair.end_round,
                    repair.from_block,
                    repair.to_block
                );
                gaps.mark_unrecoverable_in(
                    &mut transaction,
                    gap.gap_id,
                    "canonical_round_absent_after_bounded_retries",
                    Some(&evidence),
                )
                .await
                .map_err(|error| database_error("polygon_oracle_gap_terminal_failed", error))?
                .ok_or_else(|| {
                    integrity_error(
                        "polygon_oracle_gap_terminal_race",
                        "round gap was not unresolved during terminal classification",
                    )
                })?;
            }
            RoundGapOutcome::Retry => {}
        }
        self.refresh_profile_gap_health_in(&mut transaction).await?;
        transaction
            .commit()
            .await
            .map_err(|error| database_error("polygon_oracle_repair_commit_failed", error))?;
        state.artifact = None;
        Ok(())
    }

    async fn create_repair_artifact(
        &self,
        state: &mut OracleRunState,
        received_at: DateTime<Utc>,
        gap: &DataGap,
    ) -> Result<CaptureArtifact, StrategyError> {
        let (window_start, window_end) = self.artifact_window(received_at);
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polygon_oracle_repair_artifact_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let artifact = ArtifactRepository::new(self.pool.clone())
            .create_in(
                &mut transaction,
                &NewCaptureArtifact {
                    strategy_key: STRATEGY_KEY,
                    profile_generation: self.profile_generation,
                    config_schema_version: CONFIG_SCHEMA_VERSION,
                    config_snapshot: self.config_snapshot.clone(),
                    capture_window_start: window_start,
                    capture_window_end: window_end,
                    start_cursor: gap.start_cursor.clone(),
                },
            )
            .await
            .map_err(|error| {
                database_error("polygon_oracle_repair_artifact_create_failed", error)
            })?;
        transaction.commit().await.map_err(|error| {
            database_error("polygon_oracle_repair_artifact_create_commit_failed", error)
        })?;
        state.artifact = Some(artifact.clone());
        Ok(artifact)
    }

    async fn round_gap_is_complete(&self, repair: &RoundGapRepair) -> Result<bool, StrategyError> {
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polygon_oracle_gap_check_transaction_failed", error)
        })?;
        let complete = self
            .round_gap_is_complete_in(&mut transaction, repair)
            .await?;
        transaction
            .commit()
            .await
            .map_err(|error| database_error("polygon_oracle_gap_check_commit_failed", error))?;
        Ok(complete)
    }

    async fn round_gap_is_complete_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        repair: &RoundGapRepair,
    ) -> Result<bool, StrategyError> {
        let limit = i64::try_from(MAX_GAP_REPAIR_ROUNDS + 1).expect("repair limit fits bigint");
        let rows = sqlx::query_scalar::<_, i64>(
            r#"
            SELECT aggregator_round_id
            FROM market_data.polygon_chainlink_btcusd_oracle_rounds
            WHERE chain_id = 137
              AND feed_proxy_address = $1
              AND phase_id = $2
              AND block_number BETWEEN $3 AND $4
              AND aggregator_round_id BETWEEN $5 AND $6
            ORDER BY block_number, log_index, source_timestamp
            LIMIT $7
            "#,
        )
        .bind(&self.config.feed_proxy_address)
        .bind(repair.phase_id)
        .bind(i64::try_from(repair.from_block).map_err(|_| {
            integrity_error(
                "polygon_oracle_repair_block_overflow",
                "repair block overflow",
            )
        })?)
        .bind(i64::try_from(repair.to_block).map_err(|_| {
            integrity_error(
                "polygon_oracle_repair_block_overflow",
                "repair block overflow",
            )
        })?)
        .bind(repair.start_round)
        .bind(repair.end_round)
        .bind(limit)
        .fetch_all(&mut **transaction)
        .await
        .map_err(|error| database_error("polygon_oracle_gap_check_failed", error))?;
        round_gap_ids_complete(repair, &rows)
    }

    async fn mark_round_gap_unrecoverable(
        &self,
        gap_id: Uuid,
        resolution_code: &str,
        resolution_message: &str,
    ) -> Result<(), StrategyError> {
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polygon_oracle_gap_terminal_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        GapRepository::new(self.pool.clone())
            .mark_unrecoverable_in(
                &mut transaction,
                gap_id,
                resolution_code,
                Some(resolution_message),
            )
            .await
            .map_err(|error| database_error("polygon_oracle_gap_terminal_failed", error))?
            .ok_or_else(|| {
                integrity_error(
                    "polygon_oracle_gap_terminal_race",
                    "round gap was no longer unresolved",
                )
            })?;
        self.refresh_profile_gap_health_in(&mut transaction).await?;
        transaction
            .commit()
            .await
            .map_err(|error| database_error("polygon_oracle_gap_terminal_commit_failed", error))?;
        Ok(())
    }

    async fn refresh_profile_gap_health_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
    ) -> Result<(), StrategyError> {
        let updated = sqlx::query_scalar::<_, String>(
            r#"
            WITH gap_health AS (
              SELECT EXISTS (
                SELECT 1 FROM ingester.data_gaps
                WHERE strategy_key = $1 AND status IN ('open', 'repairing')
              ) AS has_unresolved_gap
            )
            UPDATE ingester.profiles
            SET observed_state = CASE
                  WHEN gap_health.has_unresolved_gap THEN 'degraded' ELSE 'running'
                END,
                health_status = CASE
                  WHEN gap_health.has_unresolved_gap THEN 'degraded' ELSE 'healthy'
                END,
                updated_at = now()
            FROM gap_health
            WHERE strategy_key = $1
              AND lease_owner = $2
              AND lease_token = $3
              AND lease_expires_at > now()
              AND desired_state = 'running'
              AND desired_generation = $4
              AND applied_generation = $4
            RETURNING strategy_key
            "#,
        )
        .bind(STRATEGY_KEY.as_str())
        .bind(&self.lease_owner)
        .bind(self.lease_token)
        .bind(self.profile_generation)
        .fetch_optional(&mut **transaction)
        .await
        .map_err(|error| database_error("polygon_oracle_gap_health_refresh_failed", error))?;
        if updated.is_none() {
            return Err(lease_lost_error());
        }
        Ok(())
    }

    async fn load_feed_metadata(
        &self,
        cached: Option<&FeedMetadata>,
    ) -> Result<FeedMetadata, StrategyError> {
        // Chainlink phase mappings are append-only. Reading the current mapping is
        // therefore a safe superset for the already-finalized log range, while
        // avoiding providers' optional historical-state retention. The repeated
        // phase/decimals reads fence an upgrade that becomes visible mid-snapshot.
        let phase_count = decode_phase_count(
            &self
                .eth_call_latest(&abi_calldata("phaseId()", &[]))
                .await?,
        )?;
        let decimals = decode_feed_decimals(
            &self
                .eth_call_latest(&abi_calldata("decimals()", &[]))
                .await?,
        )?;

        let mut aggregators = cached
            .map(|metadata| metadata.aggregators.clone())
            .unwrap_or_default();
        let first_phase = cached
            .map(|metadata| metadata.phase_count.saturating_add(1))
            .unwrap_or(1);
        for phase_id in first_phase..=phase_count {
            let value = self
                .eth_call_latest(&abi_calldata(
                    "phaseAggregators(uint16)",
                    &[encode_u16_word(phase_id)],
                ))
                .await?;
            let address = parse_abi_address(&value)?;
            if address == "0x0000000000000000000000000000000000000000" {
                return Err(source_error_value(
                    "polygon_oracle_phase_mapping_incomplete",
                    format!("latest proxy metadata omitted aggregator phase {phase_id}"),
                ));
            }
            if let Some(existing) = aggregators.insert(address.clone(), phase_id) {
                return Err(integrity_error(
                    "polygon_oracle_aggregator_phase_ambiguous",
                    format!(
                        "aggregator {address} was assigned to both phases {existing} and {phase_id}"
                    ),
                ));
            }
        }
        let confirmed_decimals = decode_feed_decimals(
            &self
                .eth_call_latest(&abi_calldata("decimals()", &[]))
                .await?,
        )?;
        let confirmed_phase_count = decode_phase_count(
            &self
                .eth_call_latest(&abi_calldata("phaseId()", &[]))
                .await?,
        )?;
        validate_latest_metadata_snapshot(
            decimals,
            phase_count,
            confirmed_decimals,
            confirmed_phase_count,
        )?;

        if let Some(cached) = cached {
            if cached.decimals != decimals {
                return Err(integrity_error(
                    "polygon_oracle_decimals_changed",
                    format!(
                        "feed decimals changed from {} to {decimals}",
                        cached.decimals
                    ),
                ));
            }
            if phase_count < cached.phase_count {
                return Err(integrity_error(
                    "polygon_oracle_phase_regressed",
                    format!(
                        "feed phase regressed from {} to {phase_count}",
                        cached.phase_count
                    ),
                ));
            }
        }
        validate_phase_mapping(phase_count, &aggregators)?;
        Ok(FeedMetadata {
            decimals,
            phase_count,
            aggregators,
        })
    }

    async fn block_number(&self) -> Result<u64, StrategyError> {
        let result = self.rpc("eth_blockNumber", json!([])).await?;
        let encoded = result.value.as_str().ok_or_else(|| {
            integrity_error(
                "polygon_oracle_block_number_invalid",
                "eth_blockNumber returned a non-string result",
            )
        })?;
        parse_quantity_u64(encoded)
    }

    async fn block_by_number(&self, number: u64) -> Result<BlockHeader, StrategyError> {
        self.block_by_number_with_receipt(number)
            .await
            .map(|(header, _)| header)
    }

    async fn block_by_number_with_receipt(
        &self,
        number: u64,
    ) -> Result<(BlockHeader, DateTime<Utc>), StrategyError> {
        self.block_by_number_at_with_receipt(&self.config.rpc_url, number)
            .await
    }

    async fn block_by_number_at_with_receipt(
        &self,
        rpc_url: &str,
        number: u64,
    ) -> Result<(BlockHeader, DateTime<Utc>), StrategyError> {
        let result = self
            .rpc_at(
                rpc_url,
                "eth_getBlockByNumber",
                json!([format_quantity(number), false]),
            )
            .await?;
        Ok((
            decode_block_header(result.value, Some(number))?,
            result.received_at,
        ))
    }

    async fn eth_call_latest(&self, data: &str) -> Result<String, StrategyError> {
        let result = self
            .rpc(
                "eth_call",
                json!([
                    {
                        "to": self.config.feed_proxy_address,
                        "data": data,
                    },
                    "latest"
                ]),
            )
            .await?;
        result.value.as_str().map(ToOwned::to_owned).ok_or_else(|| {
            integrity_error(
                "polygon_oracle_eth_call_invalid",
                "eth_call returned a non-string result",
            )
        })
    }

    async fn startup_round_boundary(
        &self,
        feed: &FeedMetadata,
        scan_start: u64,
    ) -> Result<(RoundBoundary, DateTime<Utc>), StrategyError> {
        let addresses = feed.aggregators.keys().cloned().collect::<Vec<_>>();
        let mut next_block_exclusive = scan_start;
        let mut proof_received_at = None;
        for _ in 0..MAX_STARTUP_ANCHOR_WINDOWS {
            let Some((from_block, to_block)) =
                prior_anchor_range(next_block_exclusive, self.config.maximum_block_range)
            else {
                break;
            };
            let batch = self
                .cross_checked_startup_logs(feed, &addresses, from_block, to_block)
                .await?;
            proof_received_at = Some(
                proof_received_at
                    .map(|received_at: DateTime<Utc>| received_at.max(batch.received_at))
                    .unwrap_or(batch.received_at),
            );
            if let Some(round) = batch.rounds.last() {
                return Ok((
                    RoundBoundary {
                        phase_id: round.phase_id,
                        aggregator_round_id: round.aggregator_round_id,
                        source_timestamp: round.source_timestamp,
                        block_number: round.block_number,
                    },
                    proof_received_at.expect("cross-checked anchor range has a receipt"),
                ));
            }
            if from_block == 0 {
                break;
            }
            next_block_exclusive = from_block;
        }
        Err(source_error_value(
            "polygon_oracle_startup_anchor_unavailable",
            format!(
                "no cross-provider canonical AnswerUpdated anchor was found in {} bounded windows before block {scan_start}; retrying without checkpoint advancement",
                MAX_STARTUP_ANCHOR_WINDOWS
            ),
        ))
    }

    async fn cross_checked_startup_logs(
        &self,
        feed: &FeedMetadata,
        addresses: &[String],
        from_block: u64,
        to_block: u64,
    ) -> Result<DecodedLogBatch, StrategyError> {
        let (archive, canonical) = tokio::try_join!(
            self.fetch_logs_for_addresses_at(
                &self.config.archive_log_rpc_url,
                addresses,
                from_block,
                to_block,
            ),
            self.fetch_logs_for_addresses_at(
                &self.config.rpc_url,
                addresses,
                from_block,
                to_block,
            )
        )?;
        let ((archive_end, archive_end_received_at), (canonical_end, canonical_end_received_at)) =
            tokio::try_join!(
                self.block_by_number_at_with_receipt(&self.config.archive_log_rpc_url, to_block,),
                self.block_by_number_with_receipt(to_block),
            )?;
        if archive_end.hash != canonical_end.hash
            || archive_end.timestamp != canonical_end.timestamp
        {
            return Err(source_error_value(
                "polygon_oracle_startup_anchor_fork_mismatch",
                format!(
                    "archive and canonical RPCs disagreed on startup proof endpoint block {to_block}"
                ),
            ));
        }
        let archive = self
            .decode_log_batch(archive, feed, from_block, to_block)
            .await?;
        let canonical = self
            .decode_log_batch(canonical, feed, from_block, to_block)
            .await?;
        cross_checked_startup_batch(
            archive,
            canonical,
            archive_end_received_at,
            canonical_end_received_at,
        )
    }

    async fn fetch_logs(
        &self,
        feed: &FeedMetadata,
        from_block: u64,
        to_block: u64,
    ) -> Result<LogBatch, StrategyError> {
        let addresses = feed.aggregators.keys().cloned().collect::<Vec<_>>();
        self.fetch_logs_for_addresses(&addresses, from_block, to_block)
            .await
    }

    async fn fetch_logs_for_addresses(
        &self,
        addresses: &[String],
        from_block: u64,
        to_block: u64,
    ) -> Result<LogBatch, StrategyError> {
        self.fetch_logs_for_addresses_at(
            &self.config.archive_log_rpc_url,
            addresses,
            from_block,
            to_block,
        )
        .await
    }

    async fn fetch_logs_for_addresses_at(
        &self,
        rpc_url: &str,
        addresses: &[String],
        from_block: u64,
        to_block: u64,
    ) -> Result<LogBatch, StrategyError> {
        if addresses.is_empty() || addresses.len() > usize::from(MAX_AGGREGATOR_PHASES) {
            return Err(integrity_error(
                "polygon_oracle_log_address_bound_invalid",
                "log query address count was outside its resource bound",
            ));
        }
        let result = self
            .rpc_at(
                rpc_url,
                "eth_getLogs",
                json!([{
                    "address": addresses,
                    "fromBlock": format_quantity(from_block),
                    "toBlock": format_quantity(to_block),
                    "topics": [event_topic("AnswerUpdated(int256,uint256,uint256)")],
                }]),
            )
            .await?;
        let logs = serde_json::from_value::<Vec<RpcLog>>(result.value).map_err(|error| {
            integrity_error("polygon_oracle_logs_decode_failed", error.to_string())
        })?;
        if logs.len() > MAX_RPC_LOGS_PER_RESPONSE {
            return Err(integrity_error(
                "polygon_oracle_log_result_too_large",
                format!(
                    "eth_getLogs returned {} rows, above the {MAX_RPC_LOGS_PER_RESPONSE} bound",
                    logs.len()
                ),
            ));
        }
        Ok(LogBatch {
            logs,
            received_at: result.received_at,
        })
    }

    async fn decode_log_batch(
        &self,
        batch: LogBatch,
        feed: &FeedMetadata,
        from_block: u64,
        to_block: u64,
    ) -> Result<DecodedLogBatch, StrategyError> {
        for log in &batch.logs {
            let block_number = parse_quantity_u64(&log.block_number)?;
            if block_number < from_block || block_number > to_block {
                return Err(integrity_error(
                    "polygon_oracle_log_outside_range",
                    format!(
                        "log block {block_number} escaped requested range {from_block}..={to_block}"
                    ),
                ));
            }
            if log.removed {
                let log_index = parse_quantity_u64(&log.log_index).unwrap_or_default();
                let signal = GapSignal {
                    gap_kind: "chain_reorganization",
                    reason_code: "polygon_removed_oracle_log",
                    reason_message: format!(
                        "Polygon RPC marked oracle log {}:{} in block {block_number} as removed",
                        log.transaction_hash, log_index
                    ),
                    source_time_start: None,
                    source_time_end: None,
                    start_cursor: format!(
                        "block:{block_number}:tx:{}:log:{log_index}",
                        log.transaction_hash.to_ascii_lowercase()
                    ),
                    end_cursor: format!(
                        "removed:block:{block_number}:tx:{}:log:{log_index}",
                        log.transaction_hash.to_ascii_lowercase()
                    ),
                };
                return self.record_fatal_gap(signal).await;
            }
        }

        let block_numbers = batch
            .logs
            .iter()
            .map(|log| parse_quantity_u64(&log.block_number))
            .collect::<Result<BTreeSet<_>, _>>()?
            .into_iter()
            .collect::<Vec<_>>();
        let (headers, header_received_at) = self.block_headers(&block_numbers).await?;
        let received_at = maximum_receipt(batch.received_at, header_received_at);
        let mut unique = BTreeMap::<(String, i32), OracleRound>::new();
        for log in batch.logs {
            let address = log.address.to_ascii_lowercase();
            let phase_id = feed.aggregators.get(&address).copied().ok_or_else(|| {
                integrity_error(
                    "polygon_oracle_unknown_aggregator",
                    format!("log came from unknown aggregator {address}"),
                )
            })?;
            let block_number = parse_quantity_u64(&log.block_number)?;
            let header = headers.get(&block_number).ok_or_else(|| {
                integrity_error(
                    "polygon_oracle_block_header_missing",
                    format!("block header {block_number} was absent"),
                )
            })?;
            let log_hash = normalized_hash(&log.block_hash)?;
            if header.hash != log_hash {
                let signal = GapSignal {
                    gap_kind: "chain_reorganization",
                    reason_code: "polygon_oracle_log_block_hash_mismatch",
                    reason_message: format!(
                        "oracle log block {block_number} used hash {log_hash}, canonical header used {}",
                        header.hash
                    ),
                    source_time_start: Some(header.timestamp),
                    source_time_end: Some(header.timestamp),
                    start_cursor: format!("block:{block_number}:{log_hash}"),
                    end_cursor: format!("block:{block_number}:{}", header.hash),
                };
                return self.record_fatal_gap(signal).await;
            }
            if let Some(provider_timestamp) = log.block_timestamp.as_deref() {
                let timestamp = quantity_timestamp(provider_timestamp)?;
                if timestamp != header.timestamp {
                    return Err(integrity_error(
                        "polygon_oracle_block_timestamp_mismatch",
                        format!(
                            "provider timestamp {timestamp} differed from canonical block timestamp {} for block {block_number}",
                            header.timestamp
                        ),
                    ));
                }
            }
            let round = decode_answer_updated(
                log,
                &self.config.feed_proxy_address,
                &address,
                phase_id,
                feed.decimals,
                header,
                received_at,
            )?;
            let identity = round.identity();
            if let Some(existing) = unique.get(&identity) {
                if existing != &round {
                    return Err(integrity_error(
                        "polygon_oracle_duplicate_log_conflict",
                        format!(
                            "RPC returned conflicting oracle payloads for {}:{}",
                            identity.0, identity.1
                        ),
                    ));
                }
            } else {
                unique.insert(identity, round);
            }
        }
        let mut rounds = unique.into_values().collect::<Vec<_>>();
        rounds.sort_by_key(|round| {
            (
                round.block_number,
                round.log_index,
                round.transaction_hash.clone(),
            )
        });
        Ok(DecodedLogBatch {
            rounds,
            received_at,
        })
    }

    async fn block_headers(
        &self,
        block_numbers: &[u64],
    ) -> Result<(BTreeMap<u64, BlockHeader>, Option<DateTime<Utc>>), StrategyError> {
        let mut headers = BTreeMap::new();
        let mut maximum_received_at = None;
        for numbers in block_numbers.chunks(BLOCK_HEADER_BATCH_SIZE) {
            let request = numbers
                .iter()
                .enumerate()
                .map(|(index, number)| {
                    json!({
                        "jsonrpc": "2.0",
                        "id": index + 1,
                        "method": "eth_getBlockByNumber",
                        "params": [format_quantity(*number), false],
                    })
                })
                .collect::<Vec<_>>();
            let (body, received_at) = self
                .post_json(
                    &self.config.rpc_url,
                    &Value::Array(request),
                    "block header batch",
                )
                .await?;
            maximum_received_at = Some(
                maximum_received_at
                    .map(|current: DateTime<Utc>| current.max(received_at))
                    .unwrap_or(received_at),
            );
            let envelopes = serde_json::from_slice::<Vec<RpcEnvelope>>(&body).map_err(|error| {
                integrity_error(
                    "polygon_oracle_block_batch_decode_failed",
                    error.to_string(),
                )
            })?;
            if envelopes.len() != numbers.len() {
                return Err(integrity_error(
                    "polygon_oracle_block_batch_incomplete",
                    "block header batch returned an incomplete response",
                ));
            }
            let requested = numbers.iter().copied().collect::<BTreeSet<_>>();
            for envelope in envelopes {
                let response_id =
                    envelope
                        .id
                        .as_ref()
                        .and_then(Value::as_u64)
                        .ok_or_else(|| {
                            integrity_error(
                                "polygon_oracle_block_batch_id_invalid",
                                "block header batch response omitted its numeric request ID",
                            )
                        })?;
                if response_id == 0 || response_id > numbers.len() as u64 {
                    return Err(integrity_error(
                        "polygon_oracle_block_batch_id_invalid",
                        format!("block header batch returned invalid request ID {response_id}"),
                    ));
                }
                if let Some(error) = envelope.error {
                    return Err(source_error_value(
                        "polygon_oracle_block_batch_rpc_error",
                        format!("JSON-RPC {}: {}", error.code, error.message),
                    ));
                }
                let value = envelope.result.ok_or_else(|| {
                    integrity_error(
                        "polygon_oracle_block_batch_result_missing",
                        "block header batch omitted a result",
                    )
                })?;
                let header = decode_block_header(value, None)?;
                if !requested.contains(&header.number) {
                    return Err(integrity_error(
                        "polygon_oracle_block_batch_unrequested",
                        format!("block batch returned unrequested block {}", header.number),
                    ));
                }
                if headers.insert(header.number, header).is_some() {
                    return Err(integrity_error(
                        "polygon_oracle_block_batch_duplicate",
                        "block header batch returned a duplicate block",
                    ));
                }
            }
        }
        if headers.len() != block_numbers.len() {
            return Err(integrity_error(
                "polygon_oracle_block_batch_missing",
                "block header batch omitted a requested block",
            ));
        }
        Ok((headers, maximum_received_at))
    }

    async fn rpc(&self, method: &str, params: Value) -> Result<RpcResult, StrategyError> {
        self.rpc_at(&self.config.rpc_url, method, params).await
    }

    async fn rpc_at(
        &self,
        rpc_url: &str,
        method: &str,
        params: Value,
    ) -> Result<RpcResult, StrategyError> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        });
        let (body, received_at) = self.post_json(rpc_url, &request, method).await?;
        let envelope = serde_json::from_slice::<RpcEnvelope>(&body).map_err(|error| {
            integrity_error("polygon_oracle_rpc_decode_failed", error.to_string())
        })?;
        if envelope.id.as_ref().and_then(Value::as_u64) != Some(1) {
            return Err(integrity_error(
                "polygon_oracle_rpc_id_invalid",
                format!("JSON-RPC method {method} returned an unexpected response ID"),
            ));
        }
        if let Some(error) = envelope.error {
            return Err(source_error_value(
                "polygon_oracle_rpc_error",
                format!(
                    "JSON-RPC method {method} failed with {}: {}",
                    error.code, error.message
                ),
            ));
        }
        let value = envelope.result.ok_or_else(|| {
            integrity_error(
                "polygon_oracle_rpc_result_missing",
                format!("JSON-RPC method {method} omitted its result"),
            )
        })?;
        Ok(RpcResult { value, received_at })
    }

    async fn post_json(
        &self,
        rpc_url: &str,
        request: &Value,
        operation: &str,
    ) -> Result<(Vec<u8>, DateTime<Utc>), StrategyError> {
        let mut response = self
            .client
            .post(rpc_url)
            .json(request)
            .send()
            .await
            .map_err(|error| source_error("polygon_oracle_rpc_request_failed", error))?;
        let status = response.status();
        if response
            .content_length()
            .is_some_and(|length| length > MAX_RPC_RESPONSE_BYTES as u64)
        {
            return Err(integrity_error(
                "polygon_oracle_rpc_response_too_large",
                format!("{operation} response exceeded the eight-megabyte bound"),
            ));
        }
        let mut body = Vec::with_capacity(
            response
                .content_length()
                .unwrap_or(0)
                .min(MAX_RPC_RESPONSE_BYTES as u64) as usize,
        );
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| source_error("polygon_oracle_rpc_body_failed", error))?
        {
            if body.len().saturating_add(chunk.len()) > MAX_RPC_RESPONSE_BYTES {
                return Err(integrity_error(
                    "polygon_oracle_rpc_response_too_large",
                    format!("{operation} response exceeded the eight-megabyte bound"),
                ));
            }
            body.extend_from_slice(&chunk);
        }
        // A receipt is factual only after the complete bounded body is locally available.
        let received_at = Utc::now();
        if !status.is_success() {
            let bounded = &body[..body.len().min(512)];
            return Err(source_error_value(
                "polygon_oracle_rpc_http_status",
                format!(
                    "{operation} returned HTTP {status}: {}",
                    String::from_utf8_lossy(bounded)
                ),
            ));
        }
        Ok((body, received_at))
    }

    async fn verify_canonical_overlap(
        &self,
        from_block: u64,
        to_block: u64,
        rounds: &[OracleRound],
    ) -> Result<(), StrategyError> {
        let from_block_i64 = i64::try_from(from_block).map_err(|_| {
            integrity_error(
                "polygon_oracle_block_number_overflow",
                "scan start block exceeds PostgreSQL bigint",
            )
        })?;
        let to_block_i64 = i64::try_from(to_block).map_err(|_| {
            integrity_error(
                "polygon_oracle_block_number_overflow",
                "scan end block exceeds PostgreSQL bigint",
            )
        })?;
        let stored = sqlx::query_as::<_, StoredRangeIdentity>(
            r#"
            SELECT transaction_hash, log_index, block_number, block_hash,
                   payload_sha256::text AS payload_sha256
            FROM market_data.polygon_chainlink_btcusd_oracle_rounds
            WHERE chain_id = 137
              AND feed_proxy_address = $1
              AND block_number BETWEEN $2 AND $3
            ORDER BY block_number, log_index, transaction_hash, source_timestamp
            LIMIT $4
            "#,
        )
        .bind(&self.config.feed_proxy_address)
        .bind(from_block_i64)
        .bind(to_block_i64)
        .bind(MAX_DATABASE_RANGE_ROWS)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| database_error("polygon_oracle_overlap_read_failed", error))?;
        if stored.len() >= MAX_DATABASE_RANGE_ROWS as usize {
            return Err(integrity_error(
                "polygon_oracle_overlap_result_too_large",
                "durable overlap exceeded its bounded result size",
            ));
        }

        let canonical = rounds
            .iter()
            .map(|round| (round.identity(), round))
            .collect::<BTreeMap<_, _>>();
        let mut durable = BTreeMap::new();
        for row in stored {
            let identity = (row.transaction_hash.clone(), row.log_index);
            if durable.insert(identity.clone(), row).is_some() {
                return Err(integrity_error(
                    "polygon_oracle_duplicate_logical_identity",
                    format!(
                        "multiple durable rows share oracle log {}:{}",
                        identity.0, identity.1
                    ),
                ));
            }
        }
        for (identity, row) in durable {
            let Some(round) = canonical.get(&identity) else {
                let signal = GapSignal {
                    gap_kind: "chain_reorganization",
                    reason_code: "polygon_durable_log_missing_from_canonical_overlap",
                    reason_message: format!(
                        "durable oracle log {}:{} in block {} was absent from the canonical overlap response",
                        identity.0, identity.1, row.block_number
                    ),
                    source_time_start: None,
                    source_time_end: None,
                    start_cursor: format!(
                        "block:{}:tx:{}:log:{}",
                        row.block_number, identity.0, identity.1
                    ),
                    end_cursor: format!(
                        "missing:block:{}:tx:{}:log:{}",
                        row.block_number, identity.0, identity.1
                    ),
                };
                return self.record_fatal_gap(signal).await;
            };
            if round.block_number != row.block_number
                || round.block_hash != row.block_hash
                || round.payload_sha256 != row.payload_sha256
            {
                let signal = GapSignal {
                    gap_kind: "chain_reorganization",
                    reason_code: "polygon_canonical_log_changed",
                    reason_message: format!(
                        "canonical oracle log {}:{} no longer matches its durable block or payload",
                        identity.0, identity.1
                    ),
                    source_time_start: Some(round.source_timestamp),
                    source_time_end: Some(round.source_timestamp),
                    start_cursor: format!(
                        "block:{}:{}:tx:{}:log:{}",
                        row.block_number, row.block_hash, identity.0, identity.1
                    ),
                    end_cursor: format!(
                        "block:{}:{}:tx:{}:log:{}",
                        round.block_number, round.block_hash, identity.0, identity.1
                    ),
                };
                return self.record_fatal_gap(signal).await;
            }
        }
        Ok(())
    }

    async fn detect_round_gaps(
        &self,
        from_block: u64,
        rounds: &[OracleRound],
        startup_boundary: Option<&RoundBoundary>,
    ) -> Result<Vec<GapSignal>, StrategyError> {
        if rounds.is_empty() {
            return Ok(Vec::new());
        }
        let from_block = i64::try_from(from_block).map_err(|_| {
            integrity_error(
                "polygon_oracle_block_number_overflow",
                "scan start block exceeds PostgreSQL bigint",
            )
        })?;
        let phases = rounds
            .iter()
            .map(|round| round.phase_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if phases.len() > usize::from(MAX_AGGREGATOR_PHASES) {
            return Err(integrity_error(
                "polygon_oracle_phase_boundary_bound_exceeded",
                "phase boundary lookup exceeded its resource bound",
            ));
        }
        let mut boundaries = sqlx::query_as::<_, RoundBoundary>(
            r#"
            SELECT boundary.phase_id, boundary.aggregator_round_id,
                   boundary.source_timestamp, boundary.block_number
            FROM unnest($2::integer[]) AS requested(phase_id)
            CROSS JOIN LATERAL (
              SELECT fact.phase_id, fact.aggregator_round_id,
                     fact.source_timestamp, fact.block_number
              FROM market_data.polygon_chainlink_btcusd_oracle_rounds AS fact
              WHERE fact.chain_id = 137
                AND fact.feed_proxy_address = $1
                AND fact.phase_id = requested.phase_id
                AND fact.block_number < $3
              ORDER BY fact.block_number DESC,
                       fact.log_index DESC,
                       fact.source_timestamp DESC
              LIMIT 1
            ) AS boundary
            ORDER BY boundary.phase_id
            "#,
        )
        .bind(&self.config.feed_proxy_address)
        .bind(&phases)
        .bind(from_block)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| database_error("polygon_oracle_round_boundary_read_failed", error))?
        .into_iter()
        .map(|boundary| (boundary.phase_id, boundary))
        .collect::<BTreeMap<_, _>>();
        if let Some(boundary) = startup_boundary {
            if boundary.block_number >= from_block {
                return Err(integrity_error(
                    "polygon_oracle_startup_anchor_block_invalid",
                    "historical startup round anchor did not precede the scanned block range",
                ));
            }
            if boundaries
                .insert(boundary.phase_id, boundary.clone())
                .is_some()
            {
                return Err(integrity_error(
                    "polygon_oracle_startup_anchor_conflict",
                    "durable and historical startup round boundaries overlapped",
                ));
            }
        }
        detect_round_gaps_from_boundaries(boundaries, rounds)
    }

    async fn persist_scanned_chunk(
        &self,
        state: &mut OracleRunState,
        rounds: Vec<OracleRound>,
        gaps: Vec<GapSignal>,
        checkpoint_header: BlockHeader,
    ) -> Result<(), StrategyError> {
        let next_checkpoint = OracleCheckpoint {
            last_finalized_block: Some(checkpoint_header.number),
            last_finalized_block_hash: Some(checkpoint_header.hash.clone()),
        };
        if rounds.is_empty() {
            if !gaps.is_empty() {
                return Err(integrity_error(
                    "polygon_oracle_empty_chunk_gap",
                    "an empty oracle scan unexpectedly produced round gaps",
                ));
            }
            if !empty_scan_may_advance(&state.checkpoint) {
                return Err(source_error_value(
                    "polygon_oracle_startup_coverage_unverified",
                    "initial Polygon oracle scan was empty and had no durable coverage anchor; retrying without checkpoint advancement",
                ));
            }
            self.advance_empty_checkpoint(&next_checkpoint).await?;
            state.checkpoint = next_checkpoint;
            return Ok(());
        }

        let received_at = rounds
            .iter()
            .map(|round| round.received_at)
            .max()
            .expect("nonempty oracle batch has a receipt");
        let artifact_id = self.ensure_artifact(state, received_at).await?;
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polygon_oracle_fact_transaction_begin_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let inserted_identities = self
            .persist_round_facts_in(&mut transaction, artifact_id, &rounds)
            .await?;

        let inserted = rounds
            .iter()
            .filter(|round| inserted_identities.contains(&round.identity()))
            .collect::<Vec<_>>();
        let mut artifact_after_commit = None;
        if !inserted.is_empty() {
            let artifact = ArtifactRepository::new(self.pool.clone())
                .record_batch_in(&mut transaction, artifact_id, &artifact_batch(&inserted))
                .await
                .map_err(|error| database_error("polygon_oracle_artifact_progress_failed", error))?
                .ok_or_else(|| {
                    integrity_error(
                        "polygon_oracle_artifact_not_open",
                        format!("artifact {artifact_id} was not open during fact insert"),
                    )
                })?;
            artifact_after_commit = Some(artifact);
        }

        for gap in gaps {
            self.persist_gap_in(&mut transaction, Some(artifact_id), &gap)
                .await?;
        }
        let last_source = rounds.iter().map(|round| round.source_timestamp).max();
        let last_available = rounds.iter().map(|round| round.provider_available_at).max();
        let progressed = ProfileRepository::new(self.pool.clone())
            .record_progress_in(
                &mut transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &StrategyProgress {
                    verified_record_count: rounds.len() as i64,
                    checkpoint_schema_version: CHECKPOINT_SCHEMA_VERSION,
                    checkpoint: next_checkpoint.to_value()?,
                    last_source_event_at: last_source,
                    last_provider_available_at: last_available,
                    source_watermark: last_source,
                    availability_watermark: last_available,
                },
            )
            .await
            .map_err(|error| database_error("polygon_oracle_progress_failed", error))?;
        if !progressed {
            return Err(lease_lost_error());
        }
        transaction.commit().await.map_err(|error| {
            database_error("polygon_oracle_fact_transaction_commit_failed", error)
        })?;
        if let Some(artifact) = artifact_after_commit {
            state.artifact = Some(artifact);
        }
        state.checkpoint = next_checkpoint;
        Ok(())
    }

    async fn persist_round_facts_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        rounds: &[OracleRound],
    ) -> Result<BTreeSet<(String, i32)>, StrategyError> {
        if rounds.is_empty() {
            return Ok(BTreeSet::new());
        }
        let identities = rounds
            .iter()
            .map(|round| {
                format!(
                    "{}:{}:{}:{}",
                    round.chain_id,
                    round.feed_proxy_address,
                    round.transaction_hash,
                    round.log_index
                )
            })
            .collect::<Vec<_>>();
        let transaction_hashes = rounds
            .iter()
            .map(|round| round.transaction_hash.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        sqlx::query(
            r#"
            SELECT pg_advisory_xact_lock(
              hashtextextended('polygon_chainlink_btcusd_oracle:' || identity, 0)
            )
            FROM unnest($1::text[]) AS identity
            ORDER BY identity
            "#,
        )
        .bind(&identities)
        .fetch_all(&mut **transaction)
        .await
        .map_err(|error| database_error("polygon_oracle_identity_lock_failed", error))?;

        let existing = self
            .load_existing_rounds(transaction, &transaction_hashes)
            .await?;
        let stored = index_stored_rounds(existing)?;
        let mut missing = Vec::new();
        for round in rounds {
            match stored.get(&round.identity()) {
                Some(existing) if !round.factual_eq(existing) => {
                    return Err(integrity_error(
                        "polygon_oracle_immutable_conflict",
                        format!(
                            "durable oracle log {}:{} conflicts with the canonical payload",
                            round.transaction_hash, round.log_index
                        ),
                    ));
                }
                Some(_) => {}
                None => missing.push(round),
            }
        }

        let mut inserted_identities = BTreeSet::new();
        for rows in missing.chunks(MAX_INSERT_ROWS) {
            let inserted = self
                .insert_missing_rounds(transaction, artifact_id, rows)
                .await?;
            for identity in inserted {
                if !inserted_identities.insert(identity.clone()) {
                    return Err(integrity_error(
                        "polygon_oracle_insert_returned_duplicate",
                        format!("insert returned duplicate {}:{}", identity.0, identity.1),
                    ));
                }
            }
        }

        let durable = self
            .load_existing_rounds(transaction, &transaction_hashes)
            .await?;
        let durable = index_stored_rounds(durable)?;
        for round in rounds {
            let stored = durable.get(&round.identity()).ok_or_else(|| {
                integrity_error(
                    "polygon_oracle_insert_missing",
                    format!(
                        "oracle log {}:{} was absent after insert",
                        round.transaction_hash, round.log_index
                    ),
                )
            })?;
            if !round.factual_eq(stored) {
                return Err(integrity_error(
                    "polygon_oracle_immutable_conflict",
                    format!(
                        "durable oracle log {}:{} conflicts after insert",
                        round.transaction_hash, round.log_index
                    ),
                ));
            }
        }
        Ok(inserted_identities)
    }

    async fn advance_empty_checkpoint(
        &self,
        checkpoint: &OracleCheckpoint,
    ) -> Result<(), StrategyError> {
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polygon_oracle_empty_scan_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let updated = sqlx::query_scalar::<_, String>(
            r#"
            WITH gap_health AS (
              SELECT EXISTS (
                SELECT 1 FROM ingester.data_gaps
                WHERE strategy_key = $1 AND status IN ('open', 'repairing')
              ) AS has_unresolved_gap
            )
            UPDATE ingester.profiles
            SET checkpoint_schema_version = $5,
                checkpoint = $6,
                observed_state = CASE
                  WHEN gap_health.has_unresolved_gap THEN 'degraded'
                  ELSE 'running'
                END,
                health_status = CASE
                  WHEN gap_health.has_unresolved_gap THEN 'degraded'
                  ELSE 'healthy'
                END,
                updated_at = now()
            FROM gap_health
            WHERE strategy_key = $1
              AND lease_owner = $2
              AND lease_token = $3
              AND lease_expires_at > now()
              AND desired_state = 'running'
              AND desired_generation = $4
              AND applied_generation = $4
            RETURNING strategy_key
            "#,
        )
        .bind(STRATEGY_KEY.as_str())
        .bind(&self.lease_owner)
        .bind(self.lease_token)
        .bind(self.profile_generation)
        .bind(CHECKPOINT_SCHEMA_VERSION)
        .bind(checkpoint.to_value()?)
        .fetch_optional(&mut *transaction)
        .await
        .map_err(|error| database_error("polygon_oracle_empty_scan_progress_failed", error))?;
        if updated.is_none() {
            return Err(lease_lost_error());
        }
        transaction
            .commit()
            .await
            .map_err(|error| database_error("polygon_oracle_empty_scan_commit_failed", error))?;
        Ok(())
    }

    async fn load_existing_rounds(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        transaction_hashes: &[String],
    ) -> Result<Vec<StoredOracleRound>, StrategyError> {
        sqlx::query_as::<_, StoredOracleRound>(
            r#"
            SELECT chain_id, feed_proxy_address, aggregator_address, phase_id,
                   aggregator_round_id, source_timestamp, block_timestamp,
                   answer_raw, price, decimals, block_number, block_hash,
                   transaction_hash, log_index, provider_available_at,
                   source_payload, payload_sha256::text AS payload_sha256
            FROM market_data.polygon_chainlink_btcusd_oracle_rounds
            WHERE chain_id = 137
              AND feed_proxy_address = $1
              AND transaction_hash = ANY($2::text[])
            ORDER BY transaction_hash, log_index, source_timestamp
            LIMIT $3
            "#,
        )
        .bind(&self.config.feed_proxy_address)
        .bind(transaction_hashes)
        .bind(MAX_DATABASE_RANGE_ROWS)
        .fetch_all(&mut **transaction)
        .await
        .map_err(|error| database_error("polygon_oracle_existing_round_read_failed", error))
        .and_then(|rows| {
            if rows.len() >= MAX_DATABASE_RANGE_ROWS as usize {
                Err(integrity_error(
                    "polygon_oracle_existing_result_too_large",
                    "existing oracle identity lookup exceeded its bounded result size",
                ))
            } else {
                Ok(rows)
            }
        })
    }

    async fn insert_missing_rounds(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Uuid,
        rounds: &[&OracleRound],
    ) -> Result<Vec<(String, i32)>, StrategyError> {
        if rounds.is_empty() {
            return Ok(Vec::new());
        }
        let mut query = QueryBuilder::<Postgres>::new(
            "INSERT INTO market_data.polygon_chainlink_btcusd_oracle_rounds (\
             source, chain_id, feed_proxy_address, aggregator_address, phase_id, \
             aggregator_round_id, source_timestamp, block_timestamp, answer_raw, price, \
             decimals, block_number, block_hash, transaction_hash, log_index, \
             provider_available_at, received_at, source_payload, payload_sha256, \
             strategy_key, capture_artifact_id) ",
        );
        query.push_values(rounds, |mut row, round| {
            row.push_bind(SOURCE)
                .push_bind(round.chain_id)
                .push_bind(&round.feed_proxy_address)
                .push_bind(&round.aggregator_address)
                .push_bind(round.phase_id)
                .push_bind(round.aggregator_round_id)
                .push_bind(round.source_timestamp)
                .push_bind(round.block_timestamp)
                .push_bind(round.answer_raw)
                .push_bind(round.price)
                .push_bind(round.decimals)
                .push_bind(round.block_number)
                .push_bind(&round.block_hash)
                .push_bind(&round.transaction_hash)
                .push_bind(round.log_index)
                .push_bind(round.provider_available_at)
                .push_bind(round.received_at)
                .push_bind(&round.source_payload)
                .push_bind(&round.payload_sha256)
                .push_bind(STRATEGY_KEY.as_str())
                .push_bind(artifact_id);
        });
        query.push(
            " ON CONFLICT (source_timestamp, chain_id, feed_proxy_address, transaction_hash, log_index) \
             DO NOTHING RETURNING transaction_hash, log_index",
        );
        let rows = query
            .build_query_as::<InsertedIdentity>()
            .fetch_all(&mut **transaction)
            .await
            .map_err(|error| database_error("polygon_oracle_insert_failed", error))?;
        Ok(rows
            .into_iter()
            .map(|row| (row.transaction_hash, row.log_index))
            .collect())
    }

    async fn persist_gap_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact_id: Option<Uuid>,
        gap: &GapSignal,
    ) -> Result<(), StrategyError> {
        GapRepository::new(self.pool.clone())
            .detect_in(
                transaction,
                &NewDataGap {
                    strategy_key: STRATEGY_KEY,
                    detected_artifact_id: artifact_id,
                    gap_kind: gap.gap_kind.to_owned(),
                    reason_code: gap.reason_code.to_owned(),
                    reason_message: Some(gap.reason_message.clone()),
                    source_time_start: gap.source_time_start,
                    source_time_end: gap.source_time_end,
                    start_cursor: Some(gap.start_cursor.clone()),
                    end_cursor: Some(gap.end_cursor.clone()),
                },
            )
            .await
            .map_err(|error| integrity_error("polygon_oracle_gap_record_failed", error))?;
        let marked = ProfileRepository::new(self.pool.clone())
            .mark_degraded_in(
                transaction,
                STRATEGY_KEY,
                &self.lease_owner,
                self.lease_token,
                self.profile_generation,
                &StrategyDegradation {
                    reason_code: gap.reason_code.to_owned(),
                    reason_message: gap.reason_message.clone(),
                },
            )
            .await
            .map_err(|error| database_error("polygon_oracle_mark_degraded_failed", error))?;
        if !marked {
            return Err(lease_lost_error());
        }
        Ok(())
    }

    async fn record_fatal_gap<T>(&self, gap: GapSignal) -> Result<T, StrategyError> {
        let mut transaction =
            self.pool.begin().await.map_err(|error| {
                database_error("polygon_oracle_reorg_transaction_failed", error)
            })?;
        self.assert_lease_in(&mut transaction, false).await?;
        self.persist_gap_in(&mut transaction, None, &gap).await?;
        transaction
            .commit()
            .await
            .map_err(|error| database_error("polygon_oracle_reorg_commit_failed", error))?;
        Err(integrity_error(gap.reason_code, gap.reason_message))
    }

    async fn assert_lease_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        allow_draining_generation: bool,
    ) -> Result<(), StrategyError> {
        let profiles = ProfileRepository::new(self.pool.clone());
        let current = if allow_draining_generation {
            profiles
                .lock_owned_lease_in(
                    transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                )
                .await
        } else {
            profiles
                .lock_current_lease_in(
                    transaction,
                    STRATEGY_KEY,
                    &self.lease_owner,
                    self.lease_token,
                    self.profile_generation,
                )
                .await
        }
        .map_err(|error| database_error("polygon_oracle_lease_check_failed", error))?;
        if !current {
            return Err(lease_lost_error());
        }
        Ok(())
    }

    fn artifact_window(&self, timestamp: DateTime<Utc>) -> (DateTime<Utc>, DateTime<Utc>) {
        let seconds = timestamp.timestamp();
        let start_seconds = seconds
            .div_euclid(self.config.artifact_window_seconds)
            .saturating_mul(self.config.artifact_window_seconds);
        let start = Utc
            .timestamp_opt(start_seconds, 0)
            .single()
            .expect("aligned artifact timestamp is representable");
        let end = Utc
            .timestamp_opt(
                start_seconds.saturating_add(self.config.artifact_window_seconds),
                0,
            )
            .single()
            .expect("bounded artifact timestamp is representable");
        (start, end)
    }

    async fn seal_elapsed_artifact(
        &self,
        state: &mut OracleRunState,
        now: DateTime<Utc>,
    ) -> Result<(), StrategyError> {
        if state.artifact.is_none() {
            state.artifact = ArtifactRepository::new(self.pool.clone())
                .get_open(STRATEGY_KEY)
                .await
                .map_err(|error| database_error("polygon_oracle_artifact_read_failed", error))?;
        }
        if let Some(artifact) = state.artifact.as_ref() {
            verify_artifact_generation(artifact.profile_generation, self.profile_generation)?;
            if artifact.profile_generation == self.profile_generation
                && (artifact.config_schema_version != CONFIG_SCHEMA_VERSION
                    || artifact.config_snapshot != self.config_snapshot)
            {
                return Err(integrity_error(
                    "polygon_oracle_artifact_config_conflict",
                    "open artifact has the current generation but different effective config",
                ));
            }
            if now >= artifact.capture_window_end {
                self.seal_artifact(state, false).await?;
            }
        }
        Ok(())
    }

    async fn ensure_artifact(
        &self,
        state: &mut OracleRunState,
        received_at: DateTime<Utc>,
    ) -> Result<Uuid, StrategyError> {
        let (window_start, window_end) = self.artifact_window(received_at);
        let current_matches = state.artifact.as_ref().is_some_and(|artifact| {
            artifact.profile_generation == self.profile_generation
                && artifact.config_schema_version == CONFIG_SCHEMA_VERSION
                && artifact.config_snapshot == self.config_snapshot
                && artifact.capture_window_start == window_start
                && artifact.capture_window_end == window_end
        });
        if current_matches {
            return Ok(state
                .artifact
                .as_ref()
                .expect("matching artifact exists")
                .artifact_id);
        }
        if state.artifact.is_some() {
            self.seal_artifact(state, false).await?;
        }

        let repository = ArtifactRepository::new(self.pool.clone());
        if let Some(open) = repository
            .get_open(STRATEGY_KEY)
            .await
            .map_err(|error| database_error("polygon_oracle_artifact_read_failed", error))?
        {
            verify_artifact_generation(open.profile_generation, self.profile_generation)?;
            if open.profile_generation == self.profile_generation
                && (open.config_schema_version != CONFIG_SCHEMA_VERSION
                    || open.config_snapshot != self.config_snapshot)
            {
                return Err(integrity_error(
                    "polygon_oracle_artifact_config_conflict",
                    "open artifact has the current generation but different effective config",
                ));
            }
            let reusable = open.profile_generation == self.profile_generation
                && open.config_schema_version == CONFIG_SCHEMA_VERSION
                && open.config_snapshot == self.config_snapshot
                && open.capture_window_start == window_start
                && open.capture_window_end == window_end;
            state.artifact = Some(open);
            if reusable {
                return Ok(state
                    .artifact
                    .as_ref()
                    .expect("reusable artifact exists")
                    .artifact_id);
            }
            self.seal_artifact(state, false).await?;
        }

        let mut transaction =
            self.pool.begin().await.map_err(|error| {
                database_error("polygon_oracle_artifact_transaction_failed", error)
            })?;
        self.assert_lease_in(&mut transaction, false).await?;
        let artifact = repository
            .create_in(
                &mut transaction,
                &NewCaptureArtifact {
                    strategy_key: STRATEGY_KEY,
                    profile_generation: self.profile_generation,
                    config_schema_version: CONFIG_SCHEMA_VERSION,
                    config_snapshot: self.config_snapshot.clone(),
                    capture_window_start: window_start,
                    capture_window_end: window_end,
                    start_cursor: state
                        .checkpoint
                        .last_finalized_block
                        .map(|block| format!("block:{block}")),
                },
            )
            .await
            .map_err(|error| database_error("polygon_oracle_artifact_create_failed", error))?;
        transaction.commit().await.map_err(|error| {
            database_error("polygon_oracle_artifact_create_commit_failed", error)
        })?;
        let artifact_id = artifact.artifact_id;
        state.artifact = Some(artifact);
        info!(
            strategy = %STRATEGY_KEY,
            %artifact_id,
            %window_start,
            %window_end,
            "opened Polygon oracle capture artifact"
        );
        Ok(artifact_id)
    }

    async fn seal_artifact(
        &self,
        state: &mut OracleRunState,
        allow_draining_generation: bool,
    ) -> Result<(), StrategyError> {
        if state.artifact.is_none() {
            state.artifact = ArtifactRepository::new(self.pool.clone())
                .get_open(STRATEGY_KEY)
                .await
                .map_err(|error| database_error("polygon_oracle_artifact_read_failed", error))?;
        }
        if let Some(artifact) = state.artifact.as_ref() {
            verify_artifact_generation(artifact.profile_generation, self.profile_generation)?;
        }
        let Some(artifact) = state.artifact.as_ref().cloned() else {
            if allow_draining_generation {
                let mut transaction = self.pool.begin().await.map_err(|error| {
                    database_error("polygon_oracle_owned_drain_transaction_failed", error)
                })?;
                self.assert_lease_in(&mut transaction, true).await?;
                transaction.commit().await.map_err(|error| {
                    database_error("polygon_oracle_owned_drain_commit_failed", error)
                })?;
            }
            return Ok(());
        };
        let mut transaction = self.pool.begin().await.map_err(|error| {
            database_error("polygon_oracle_artifact_seal_transaction_failed", error)
        })?;
        self.assert_lease_in(&mut transaction, allow_draining_generation)
            .await?;
        let artifact = ArtifactRepository::new(self.pool.clone())
            .get_in(&mut transaction, artifact.artifact_id)
            .await
            .map_err(|error| database_error("polygon_oracle_artifact_refresh_failed", error))?
            .ok_or_else(|| {
                integrity_error(
                    "polygon_oracle_artifact_missing",
                    "capture artifact disappeared before lease-fenced completion",
                )
            })?;
        let seal = self.artifact_seal_in(&mut transaction, &artifact).await?;
        let completed = ArtifactRepository::new(self.pool.clone())
            .complete_in(
                &mut transaction,
                artifact.artifact_id,
                &seal.content_sha256,
                seal.end_cursor
                    .as_deref()
                    .or(artifact.end_cursor.as_deref()),
            )
            .await
            .map_err(|error| database_error("polygon_oracle_artifact_complete_failed", error))?;
        if completed.is_none() {
            return Err(integrity_error(
                "polygon_oracle_artifact_not_open",
                format!(
                    "artifact {} was not open while sealing",
                    artifact.artifact_id
                ),
            ));
        }
        transaction
            .commit()
            .await
            .map_err(|error| database_error("polygon_oracle_artifact_seal_commit_failed", error))?;
        state.artifact = None;
        info!(
            strategy = %STRATEGY_KEY,
            artifact_id = %artifact.artifact_id,
            content_sha256 = %seal.content_sha256,
            "sealed Polygon oracle capture artifact"
        );
        Ok(())
    }

    async fn artifact_seal_in(
        &self,
        transaction: &mut Transaction<'_, Postgres>,
        artifact: &CaptureArtifact,
    ) -> Result<ArtifactSeal, StrategyError> {
        let rows = sqlx::query_as::<_, ArtifactChecksumRow>(
            r#"
            SELECT source_timestamp, block_number, transaction_hash, log_index,
                   payload_sha256::text AS payload_sha256
            FROM market_data.polygon_chainlink_btcusd_oracle_rounds
            WHERE capture_artifact_id = $1 AND strategy_key = $2
            ORDER BY source_timestamp, block_number, transaction_hash, log_index
            "#,
        )
        .bind(artifact.artifact_id)
        .bind(STRATEGY_KEY.as_str())
        .fetch_all(&mut **transaction)
        .await
        .map_err(|error| database_error("polygon_oracle_artifact_checksum_read_failed", error))?;
        artifact_seal_from_rows(artifact.artifact_id, artifact.record_count, &rows)
    }

    async fn finish_owned_drain(&self, state: &mut OracleRunState) -> Result<(), StrategyError> {
        self.seal_artifact(state, true).await?;
        info!(
            strategy = %STRATEGY_KEY,
            generation = self.profile_generation,
            "Polygon oracle strategy drained after a desired-state lease race"
        );
        Ok(())
    }
}

fn artifact_seal_from_rows(
    artifact_id: Uuid,
    expected_record_count: i64,
    rows: &[ArtifactChecksumRow],
) -> Result<ArtifactSeal, StrategyError> {
    if i64::try_from(rows.len()).ok() != Some(expected_record_count) {
        return Err(integrity_error(
            "polygon_oracle_artifact_count_mismatch",
            format!(
                "artifact {} records {} rows but owns {} facts",
                artifact_id,
                expected_record_count,
                rows.len()
            ),
        ));
    }
    let mut hasher = Sha256::new();
    for row in rows {
        hash_field(&mut hasher, &row.source_timestamp.to_rfc3339());
        hash_field(&mut hasher, &row.block_number.to_string());
        hash_field(&mut hasher, &row.transaction_hash);
        hash_field(&mut hasher, &row.log_index.to_string());
        hash_field(&mut hasher, &row.payload_sha256);
    }
    Ok(ArtifactSeal {
        content_sha256: hex_digest(hasher.finalize()),
        end_cursor: rows.last().map(|row| {
            format!(
                "block:{}:tx:{}:log:{}",
                row.block_number, row.transaction_hash, row.log_index
            )
        }),
    })
}

#[derive(Debug, FromRow)]
struct InsertedIdentity {
    transaction_hash: String,
    log_index: i32,
}

fn index_stored_rounds(
    rows: Vec<StoredOracleRound>,
) -> Result<BTreeMap<(String, i32), StoredOracleRound>, StrategyError> {
    let mut indexed = BTreeMap::new();
    for row in rows {
        let identity = row.identity();
        if indexed.insert(identity.clone(), row).is_some() {
            return Err(integrity_error(
                "polygon_oracle_duplicate_logical_identity",
                format!(
                    "multiple durable rows share oracle log {}:{}",
                    identity.0, identity.1
                ),
            ));
        }
    }
    Ok(indexed)
}

fn decode_answer_updated(
    log: RpcLog,
    feed_proxy_address: &str,
    expected_aggregator_address: &str,
    phase_id: u16,
    decimals: u32,
    block: &BlockHeader,
    received_at: DateTime<Utc>,
) -> Result<OracleRound, StrategyError> {
    if log.removed {
        return Err(integrity_error(
            "polygon_removed_oracle_log",
            "removed log reached the factual decoder",
        ));
    }
    let address = normalized_address(&log.address)?;
    if address != expected_aggregator_address {
        return Err(integrity_error(
            "polygon_oracle_aggregator_mismatch",
            "oracle log address did not match its selected phase aggregator",
        ));
    }
    let answer_topic = event_topic("AnswerUpdated(int256,uint256,uint256)");
    if log.topics.len() != 3 || log.topics[0].to_ascii_lowercase() != answer_topic {
        return Err(integrity_error(
            "polygon_oracle_topic_layout_invalid",
            "AnswerUpdated log had an unexpected topic layout",
        ));
    }
    let answer = parse_positive_i128_word(&log.topics[1])?;
    let aggregator_round_id = i64::try_from(parse_abi_u64(&log.topics[2])?).map_err(|_| {
        integrity_error(
            "polygon_oracle_round_id_overflow",
            "aggregator round ID exceeds PostgreSQL bigint",
        )
    })?;
    if aggregator_round_id <= 0 {
        return Err(integrity_error(
            "polygon_oracle_round_id_invalid",
            "aggregator round ID must be positive",
        ));
    }
    let source_seconds = i64::try_from(parse_first_abi_u64(&log.data)?).map_err(|_| {
        integrity_error(
            "polygon_oracle_source_timestamp_overflow",
            "oracle source timestamp exceeds signed seconds",
        )
    })?;
    let source_timestamp = Utc
        .timestamp_opt(source_seconds, 0)
        .single()
        .ok_or_else(|| {
            integrity_error(
                "polygon_oracle_source_timestamp_invalid",
                "oracle source timestamp is not representable",
            )
        })?;
    if source_timestamp > block.timestamp {
        return Err(integrity_error(
            "polygon_oracle_source_after_block",
            format!(
                "oracle source timestamp {source_timestamp} follows block timestamp {}",
                block.timestamp
            ),
        ));
    }
    let encoded_block = parse_quantity_u64(&log.block_number)?;
    if encoded_block != block.number {
        return Err(integrity_error(
            "polygon_oracle_log_block_mismatch",
            format!(
                "oracle log encoded block {encoded_block}, expected {}",
                block.number
            ),
        ));
    }
    let block_number = i64::try_from(block.number).map_err(|_| {
        integrity_error(
            "polygon_oracle_block_number_overflow",
            "block number exceeds PostgreSQL bigint",
        )
    })?;
    let log_index = i32::try_from(parse_quantity_u64(&log.log_index)?).map_err(|_| {
        integrity_error(
            "polygon_oracle_log_index_overflow",
            "log index exceeds PostgreSQL integer",
        )
    })?;
    let transaction_hash = normalized_hash(&log.transaction_hash)?;
    let block_hash = normalized_hash(&log.block_hash)?;
    if block_hash != block.hash {
        return Err(integrity_error(
            "polygon_oracle_log_block_hash_mismatch",
            "oracle log block hash did not match the canonical header",
        ));
    }
    let answer_raw = Decimal::from_str(&answer.to_string()).map_err(|error| {
        integrity_error(
            "polygon_oracle_answer_decimal_overflow",
            format!("oracle answer cannot be represented exactly: {error}"),
        )
    })?;
    let price = Decimal::from_i128_with_scale(answer, decimals);
    if price <= Decimal::ZERO {
        return Err(integrity_error(
            "polygon_oracle_price_invalid",
            "scaled BTC/USD price must be positive",
        ));
    }
    let normalized_data = format!("0x{}", normalized_word(&log.data)?);
    let normalized_topics = log
        .topics
        .iter()
        .map(|topic| normalized_hash(topic))
        .collect::<Result<Vec<_>, _>>()?;
    let source_payload = json!({
        "address": address,
        "topics": normalized_topics,
        "data": normalized_data,
        "blockNumber": format_quantity(block.number),
        "blockHash": block.hash,
        "transactionHash": transaction_hash,
        "logIndex": format_quantity(u64::try_from(log_index).expect("nonnegative log index")),
        "blockTimestamp": format_quantity(u64::try_from(block.timestamp.timestamp()).map_err(|_| {
            integrity_error(
                "polygon_oracle_block_timestamp_negative",
                "block timestamp was before the Unix epoch",
            )
        })?),
        "removed": false,
        "phaseId": phase_id,
    });
    let payload = serde_json::to_vec(&source_payload).map_err(|error| {
        integrity_error("polygon_oracle_payload_encode_failed", error.to_string())
    })?;
    if payload.len() > 32_768 {
        return Err(integrity_error(
            "polygon_oracle_payload_too_large",
            "canonical oracle payload exceeded 32768 bytes",
        ));
    }
    let payload_sha256 = hex_digest(Sha256::digest(&payload));

    Ok(OracleRound {
        chain_id: POLYGON_CHAIN_ID,
        feed_proxy_address: feed_proxy_address.to_owned(),
        aggregator_address: expected_aggregator_address.to_owned(),
        phase_id: i32::from(phase_id),
        aggregator_round_id,
        source_timestamp,
        block_timestamp: block.timestamp,
        answer_raw,
        price,
        decimals: i32::try_from(decimals).map_err(|_| {
            integrity_error("polygon_oracle_decimals_overflow", "decimals overflow")
        })?,
        block_number,
        block_hash,
        transaction_hash,
        log_index,
        provider_available_at: block.timestamp,
        received_at,
        source_payload,
        payload_sha256,
    })
}

fn decode_block_header(
    value: Value,
    expected_number: Option<u64>,
) -> Result<BlockHeader, StrategyError> {
    if value.is_null() {
        return Err(integrity_error(
            "polygon_oracle_block_missing",
            "Polygon RPC returned null for a finalized block",
        ));
    }
    let block = serde_json::from_value::<RpcBlock>(value).map_err(|error| {
        integrity_error("polygon_oracle_block_decode_failed", error.to_string())
    })?;
    let number = block
        .number
        .as_deref()
        .ok_or_else(|| {
            integrity_error(
                "polygon_oracle_pending_block",
                "Polygon RPC returned a pending block for a finalized query",
            )
        })
        .and_then(parse_quantity_u64)?;
    if expected_number.is_some_and(|expected| number != expected) {
        return Err(integrity_error(
            "polygon_oracle_block_number_mismatch",
            format!(
                "Polygon RPC returned block {number} when {} was requested",
                expected_number.expect("checked as some")
            ),
        ));
    }
    let hash = block.hash.as_deref().ok_or_else(|| {
        integrity_error(
            "polygon_oracle_pending_block",
            "Polygon RPC omitted the finalized block hash",
        )
    })?;
    Ok(BlockHeader {
        number,
        hash: normalized_hash(hash)?,
        timestamp: quantity_timestamp(&block.timestamp)?,
    })
}

fn abi_calldata(signature: &str, arguments: &[String]) -> String {
    let digest = Keccak256::digest(signature.as_bytes());
    let mut encoded = format!("0x{}", hex::encode(&digest[..4]));
    for argument in arguments {
        encoded.push_str(argument);
    }
    encoded
}

fn event_topic(signature: &str) -> String {
    format!("0x{}", hex::encode(Keccak256::digest(signature.as_bytes())))
}

fn encode_u16_word(value: u16) -> String {
    format!("{value:064x}")
}

fn decode_feed_decimals(value: &str) -> Result<u32, StrategyError> {
    let decimals = u32::try_from(parse_abi_u64(value)?)
        .map_err(|_| integrity_error("polygon_oracle_decimals_overflow", "decimals overflow"))?;
    if decimals > MAX_SUPPORTED_DECIMALS {
        return Err(integrity_error(
            "polygon_oracle_decimals_unsupported",
            format!("feed decimals {decimals} exceed supported precision"),
        ));
    }
    Ok(decimals)
}

fn decode_phase_count(value: &str) -> Result<u16, StrategyError> {
    let phase_count = u16::try_from(parse_abi_u64(value)?)
        .map_err(|_| integrity_error("polygon_oracle_phase_overflow", "phase ID overflow"))?;
    if phase_count == 0 || phase_count > MAX_AGGREGATOR_PHASES {
        return Err(integrity_error(
            "polygon_oracle_phase_count_invalid",
            format!("feed phase count {phase_count} is outside 1..={MAX_AGGREGATOR_PHASES}"),
        ));
    }
    Ok(phase_count)
}

fn validate_latest_metadata_snapshot(
    decimals: u32,
    phase_count: u16,
    confirmed_decimals: u32,
    confirmed_phase_count: u16,
) -> Result<(), StrategyError> {
    if decimals != confirmed_decimals || phase_count != confirmed_phase_count {
        return Err(source_error_value(
            "polygon_oracle_metadata_snapshot_changed",
            format!(
                "latest proxy metadata changed while reading it (decimals {decimals}->{confirmed_decimals}, phase {phase_count}->{confirmed_phase_count})"
            ),
        ));
    }
    Ok(())
}

fn validate_phase_mapping(
    phase_count: u16,
    aggregators: &BTreeMap<String, u16>,
) -> Result<(), StrategyError> {
    let mapped_phases = aggregators.values().copied().collect::<BTreeSet<_>>();
    let expected_phases = (1..=phase_count).collect::<BTreeSet<_>>();
    if mapped_phases != expected_phases || aggregators.len() != usize::from(phase_count) {
        return Err(integrity_error(
            "polygon_oracle_phase_mapping_incomplete",
            "latest proxy metadata did not map every declared phase exactly once",
        ));
    }
    Ok(())
}

fn parse_abi_address(value: &str) -> Result<String, StrategyError> {
    let word = normalized_word(value)?;
    if !word[..24].bytes().all(|byte| byte == b'0') {
        return Err(integrity_error(
            "polygon_oracle_abi_address_invalid",
            "ABI address had non-zero high bytes",
        ));
    }
    normalized_address(&format!("0x{}", &word[24..]))
}

fn parse_first_abi_u64(value: &str) -> Result<u64, StrategyError> {
    let encoded = value.strip_prefix("0x").unwrap_or(value);
    if encoded.len() < 64 {
        return Err(integrity_error(
            "polygon_oracle_abi_word_missing",
            "ABI data did not contain a complete word",
        ));
    }
    parse_abi_u64(&encoded[..64])
}

fn parse_abi_u64(value: &str) -> Result<u64, StrategyError> {
    let word = normalized_word(value)?;
    if !word[..48].bytes().all(|byte| byte == b'0') {
        return Err(integrity_error(
            "polygon_oracle_abi_u64_overflow",
            "ABI unsigned integer exceeded 64 bits",
        ));
    }
    u64::from_str_radix(&word[48..], 16)
        .map_err(|error| integrity_error("polygon_oracle_abi_u64_invalid", error.to_string()))
}

fn parse_positive_i128_word(value: &str) -> Result<i128, StrategyError> {
    let word = normalized_word(value)?;
    if !word[..32].bytes().all(|byte| byte == b'0') {
        return Err(integrity_error(
            "polygon_oracle_answer_out_of_range",
            "oracle answer was negative or exceeded 128 bits",
        ));
    }
    let raw = u128::from_str_radix(&word[32..], 16)
        .map_err(|error| integrity_error("polygon_oracle_answer_invalid", error.to_string()))?;
    let answer = i128::try_from(raw).map_err(|_| {
        integrity_error(
            "polygon_oracle_answer_out_of_range",
            "oracle answer exceeded signed 128-bit capacity",
        )
    })?;
    if answer <= 0 {
        return Err(integrity_error(
            "polygon_oracle_answer_nonpositive",
            "oracle answer must be positive",
        ));
    }
    Ok(answer)
}

fn normalized_word(value: &str) -> Result<String, StrategyError> {
    let word = value.strip_prefix("0x").unwrap_or(value);
    if word.len() != 64 || !word.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(integrity_error(
            "polygon_oracle_abi_word_invalid",
            "ABI value was not one 32-byte hexadecimal word",
        ));
    }
    Ok(word.to_ascii_lowercase())
}

fn parse_quantity_u64(value: &str) -> Result<u64, StrategyError> {
    let encoded = value.strip_prefix("0x").ok_or_else(|| {
        integrity_error(
            "polygon_oracle_quantity_invalid",
            "JSON-RPC quantity lacked its 0x prefix",
        )
    })?;
    if encoded.is_empty()
        || !encoded.bytes().all(|byte| byte.is_ascii_hexdigit())
        || (encoded.len() > 1 && encoded.starts_with('0'))
    {
        return Err(integrity_error(
            "polygon_oracle_quantity_invalid",
            "JSON-RPC quantity was not canonical hexadecimal",
        ));
    }
    u64::from_str_radix(encoded, 16)
        .map_err(|error| integrity_error("polygon_oracle_quantity_overflow", error.to_string()))
}

fn format_quantity(value: u64) -> String {
    format!("0x{value:x}")
}

fn quantity_timestamp(value: &str) -> Result<DateTime<Utc>, StrategyError> {
    let seconds = i64::try_from(parse_quantity_u64(value)?).map_err(|_| {
        integrity_error(
            "polygon_oracle_block_timestamp_overflow",
            "block timestamp exceeds signed seconds",
        )
    })?;
    Utc.timestamp_opt(seconds, 0).single().ok_or_else(|| {
        integrity_error(
            "polygon_oracle_block_timestamp_invalid",
            "block timestamp is not representable",
        )
    })
}

fn normalized_address(value: &str) -> Result<String, StrategyError> {
    validate_address(value)
        .map_err(|message| integrity_error("polygon_oracle_address_invalid", message))?;
    Ok(value.to_ascii_lowercase())
}

fn normalized_hash(value: &str) -> Result<String, StrategyError> {
    validate_hash(value)
        .map_err(|message| integrity_error("polygon_oracle_hash_invalid", message))?;
    Ok(value.to_ascii_lowercase())
}

fn validate_address(value: &str) -> Result<(), &'static str> {
    if value.len() != 42
        || !value.starts_with("0x")
        || !value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("must be a 20-byte 0x-prefixed hexadecimal value");
    }
    Ok(())
}

fn validate_hash(value: &str) -> Result<(), &'static str> {
    if value.len() != 66
        || !value.starts_with("0x")
        || !value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("must be a 32-byte 0x-prefixed hexadecimal value");
    }
    Ok(())
}

fn validate_public_rpc_url(name: &str, value: &str) -> Result<(), StrategyFactoryError> {
    let parsed =
        Url::parse(value).map_err(|error| invalid_config(format!("{name} is invalid: {error}")))?;
    if parsed.scheme() != "https"
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(invalid_config(format!(
            "{name} must be an approved credential-free HTTPS endpoint"
        )));
    }
    let normalized = value.trim_end_matches('/');
    if !APPROVED_RPC_URLS.contains(&normalized) {
        return Err(invalid_config(format!(
            "{name} is not an approved Polygon public RPC endpoint"
        )));
    }
    Ok(())
}

fn verify_artifact_generation(
    artifact_generation: i64,
    profile_generation: i64,
) -> Result<(), StrategyError> {
    if artifact_generation > profile_generation {
        return Err(lease_lost_error());
    }
    Ok(())
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(bytes)
}

fn invalid_config(message: impl Into<String>) -> StrategyFactoryError {
    StrategyFactoryError::InvalidConfiguration(format!(
        "Polygon Chainlink BTC/USD oracle config {}",
        message.into()
    ))
}

fn integrity_error(code: &'static str, message: impl ToString) -> StrategyError {
    StrategyError::new(StrategyErrorKind::Integrity, code, message.to_string())
}

fn source_error(code: &'static str, error: impl ToString) -> StrategyError {
    source_error_value(code, error.to_string())
}

fn source_error_value(code: &'static str, message: impl Into<String>) -> StrategyError {
    StrategyError::new(StrategyErrorKind::TransientSource, code, message.into())
}

fn database_error(code: &'static str, error: impl ToString) -> StrategyError {
    StrategyError::new(
        StrategyErrorKind::TransientDatabase,
        code,
        error.to_string(),
    )
}

fn lease_lost_error() -> StrategyError {
    StrategyError::new(
        StrategyErrorKind::LeaseLost,
        "polygon_oracle_lease_lost",
        "Polygon oracle profile lease is no longer current",
    )
}

fn artifact_batch(rounds: &[&OracleRound]) -> ArtifactBatch {
    ArtifactBatch {
        inserted_record_count: rounds.len() as i64,
        minimum_source_timestamp: rounds.iter().map(|round| round.source_timestamp).min(),
        maximum_source_timestamp: rounds.iter().map(|round| round.source_timestamp).max(),
        minimum_received_at: rounds.iter().map(|round| round.received_at).min(),
        maximum_received_at: rounds.iter().map(|round| round.received_at).max(),
        start_cursor: rounds.first().map(|round| round.cursor()),
        end_cursor: rounds.last().map(|round| round.cursor()),
    }
}

fn detect_round_gaps_from_boundaries(
    mut previous: BTreeMap<i32, RoundBoundary>,
    rounds: &[OracleRound],
) -> Result<Vec<GapSignal>, StrategyError> {
    let mut ordered = rounds.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|round| {
        (
            round.phase_id,
            round.block_number,
            round.log_index,
            round.transaction_hash.clone(),
        )
    });
    let mut gaps = Vec::new();
    for round in ordered {
        if let Some(boundary) = previous.get(&round.phase_id) {
            if round.aggregator_round_id <= boundary.aggregator_round_id {
                return Err(integrity_error(
                    "polygon_oracle_round_sequence_regressed",
                    format!(
                        "phase {} canonical event at block {} log {} carried round {} after round {}",
                        round.phase_id,
                        round.block_number,
                        round.log_index,
                        round.aggregator_round_id,
                        boundary.aggregator_round_id
                    ),
                ));
            }
            if round.aggregator_round_id > boundary.aggregator_round_id.saturating_add(1) {
                let missing_start = boundary.aggregator_round_id.saturating_add(1);
                let missing_end = round.aggregator_round_id.saturating_sub(1);
                let from_block = u64::try_from(boundary.block_number).map_err(|_| {
                    integrity_error(
                        "polygon_oracle_gap_block_invalid",
                        "round-gap predecessor had a negative block number",
                    )
                })?;
                let to_block = u64::try_from(round.block_number).map_err(|_| {
                    integrity_error(
                        "polygon_oracle_gap_block_invalid",
                        "round-gap successor had a negative block number",
                    )
                })?;
                gaps.push(GapSignal {
                    gap_kind: "source_cursor_discontinuity",
                    reason_code: "chainlink_aggregator_round_gap",
                    reason_message: format!(
                        "Chainlink phase {} advanced from round {} at block {from_block} to {} at block {to_block}",
                        round.phase_id,
                        boundary.aggregator_round_id,
                        round.aggregator_round_id
                    ),
                    source_time_start: Some(boundary.source_timestamp),
                    source_time_end: Some(round.source_timestamp),
                    start_cursor: round_gap_cursor(
                        round.phase_id,
                        missing_start,
                        from_block,
                    ),
                    end_cursor: round_gap_cursor(round.phase_id, missing_end, to_block),
                });
            }
        }
        previous.insert(
            round.phase_id,
            RoundBoundary {
                phase_id: round.phase_id,
                aggregator_round_id: round.aggregator_round_id,
                source_timestamp: round.source_timestamp,
                block_number: round.block_number,
            },
        );
    }
    Ok(gaps)
}

fn round_gap_cursor(phase_id: i32, round_id: i64, block_number: u64) -> String {
    format!("phase:{phase_id}:round:{round_id}:block:{block_number}")
}

fn parse_round_gap_cursors(
    start_cursor: Option<&str>,
    end_cursor: Option<&str>,
) -> Result<RoundGapRepair, StrategyError> {
    fn parse(value: Option<&str>) -> Result<(i32, i64, u64), StrategyError> {
        let value = value.ok_or_else(|| {
            integrity_error(
                "polygon_oracle_repair_cursor_missing",
                "round gap omitted a repair cursor",
            )
        })?;
        let fields = value.split(':').collect::<Vec<_>>();
        if fields.len() != 6 || fields[0] != "phase" || fields[2] != "round" || fields[4] != "block"
        {
            return Err(integrity_error(
                "polygon_oracle_repair_cursor_invalid",
                format!("round-gap cursor has an invalid shape: {value}"),
            ));
        }
        let phase_id = fields[1].parse::<i32>().map_err(|error| {
            integrity_error("polygon_oracle_repair_cursor_invalid", error.to_string())
        })?;
        let round_id = fields[3].parse::<i64>().map_err(|error| {
            integrity_error("polygon_oracle_repair_cursor_invalid", error.to_string())
        })?;
        let block_number = fields[5].parse::<u64>().map_err(|error| {
            integrity_error("polygon_oracle_repair_cursor_invalid", error.to_string())
        })?;
        if phase_id <= 0 || round_id <= 0 {
            return Err(integrity_error(
                "polygon_oracle_repair_cursor_invalid",
                "round-gap phase and round must be positive",
            ));
        }
        Ok((phase_id, round_id, block_number))
    }

    let (start_phase, start_round, from_block) = parse(start_cursor)?;
    let (end_phase, end_round, to_block) = parse(end_cursor)?;
    if start_phase != end_phase || start_round > end_round || from_block > to_block {
        return Err(integrity_error(
            "polygon_oracle_repair_cursor_invalid",
            "round-gap cursor bounds disagree or regress",
        ));
    }
    Ok(RoundGapRepair {
        phase_id: start_phase,
        start_round,
        end_round,
        from_block,
        to_block,
    })
}

impl RoundGapRepair {
    fn is_resource_bounded(&self, maximum_block_range: u64) -> bool {
        let Some(round_count) = self
            .end_round
            .checked_sub(self.start_round)
            .and_then(|difference| u64::try_from(difference).ok())
            .and_then(|difference| difference.checked_add(1))
        else {
            return false;
        };
        let Some(block_count) = self
            .to_block
            .checked_sub(self.from_block)
            .and_then(|difference| difference.checked_add(1))
        else {
            return false;
        };
        let request_count = block_count.saturating_add(maximum_block_range.saturating_sub(1))
            / maximum_block_range.max(1);
        round_count <= MAX_GAP_REPAIR_ROUNDS
            && block_count <= MAX_GAP_REPAIR_BLOCKS
            && request_count <= MAX_GAP_REPAIR_RPC_REQUESTS
    }
}

fn round_gap_ids_complete(
    repair: &RoundGapRepair,
    round_ids: &[i64],
) -> Result<bool, StrategyError> {
    if round_ids.len() > MAX_GAP_REPAIR_ROUNDS as usize {
        return Err(integrity_error(
            "polygon_oracle_repair_result_too_large",
            "durable round-gap verification exceeded its result bound",
        ));
    }
    let unique = round_ids.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != round_ids.len() {
        return Err(integrity_error(
            "polygon_oracle_duplicate_phase_round",
            "multiple durable facts share one Chainlink phase round",
        ));
    }
    let expected_count = repair
        .end_round
        .checked_sub(repair.start_round)
        .and_then(|difference| usize::try_from(difference).ok())
        .and_then(|difference| difference.checked_add(1))
        .ok_or_else(|| {
            integrity_error(
                "polygon_oracle_repair_round_bound_invalid",
                "round-gap bounds cannot be counted",
            )
        })?;
    Ok(unique.len() == expected_count
        && unique.first().copied() == Some(repair.start_round)
        && unique.last().copied() == Some(repair.end_round))
}

fn empty_scan_may_advance(checkpoint: &OracleCheckpoint) -> bool {
    checkpoint.last_finalized_block.is_some() && checkpoint.last_finalized_block_hash.is_some()
}

fn maximum_receipt(
    initial: DateTime<Utc>,
    receipts: impl IntoIterator<Item = DateTime<Utc>>,
) -> DateTime<Utc> {
    receipts
        .into_iter()
        .fold(initial, |maximum, receipt| maximum.max(receipt))
}

fn prior_anchor_range(next_block_exclusive: u64, maximum_block_range: u64) -> Option<(u64, u64)> {
    let to_block = next_block_exclusive.checked_sub(1)?;
    let block_count = maximum_block_range.clamp(1, STARTUP_ANCHOR_WINDOW_BLOCKS);
    Some((
        to_block.saturating_sub(block_count.saturating_sub(1)),
        to_block,
    ))
}

fn cross_checked_startup_batch(
    mut archive: DecodedLogBatch,
    mut canonical: DecodedLogBatch,
    archive_header_received_at: DateTime<Utc>,
    canonical_header_received_at: DateTime<Utc>,
) -> Result<DecodedLogBatch, StrategyError> {
    let received_at = maximum_receipt(
        archive.received_at,
        [
            canonical.received_at,
            archive_header_received_at,
            canonical_header_received_at,
        ],
    );
    for round in &mut archive.rounds {
        round.received_at = received_at;
    }
    for round in &mut canonical.rounds {
        round.received_at = received_at;
    }
    if archive.rounds != canonical.rounds {
        return Err(source_error_value(
            "polygon_oracle_startup_anchor_provider_mismatch",
            format!(
                "archive and canonical RPCs returned different AnswerUpdated facts ({} versus {})",
                archive.rounds.len(),
                canonical.rounds.len()
            ),
        ));
    }
    canonical.received_at = received_at;
    Ok(canonical)
}

fn classify_round_gap_outcome(complete: bool, repair_attempts: i32) -> RoundGapOutcome {
    if complete {
        RoundGapOutcome::Repaired
    } else if repair_attempts >= MAX_GAP_REPAIR_ATTEMPTS {
        RoundGapOutcome::Unrecoverable
    } else {
        RoundGapOutcome::Retry
    }
}

fn validate_durable_cursor_state(
    durable_max: Option<i64>,
    checkpoint_block: Option<u64>,
) -> Result<(), StrategyError> {
    match (durable_max, checkpoint_block) {
        (Some(database), Some(checkpoint))
            if u64::try_from(database)
                .ok()
                .is_some_and(|block| block <= checkpoint) =>
        {
            Ok(())
        }
        (None, None) => Ok(()),
        (None, Some(checkpoint)) => Err(integrity_error(
            "polygon_oracle_checkpoint_without_anchor",
            format!("checkpoint block {checkpoint} has no durable oracle fact coverage anchor"),
        )),
        (Some(database), checkpoint) => Err(integrity_error(
            "polygon_oracle_checkpoint_behind_facts",
            format!("durable oracle block {database} is ahead of checkpoint {checkpoint:?}"),
        )),
    }
}

fn scan_start(
    checkpoint_block: Option<u64>,
    finalized_head: u64,
    overlap_blocks: u64,
    startup_lookback_blocks: u64,
) -> u64 {
    checkpoint_block
        .map(|block| block.saturating_sub(overlap_blocks.saturating_sub(1)))
        .unwrap_or_else(|| finalized_head.saturating_sub(startup_lookback_blocks.saturating_sub(1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_round() -> OracleRound {
        let logs_envelope: RpcEnvelope = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/polygon/answer_updated_logs_v1.json"
        ))
        .unwrap();
        let logs: Vec<RpcLog> = serde_json::from_value(logs_envelope.result.unwrap()).unwrap();
        let block_envelope: RpcEnvelope = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/polygon/finalized_block_v1.json"
        ))
        .unwrap();
        let block = decode_block_header(block_envelope.result.unwrap(), Some(100)).unwrap();
        decode_answer_updated(
            logs.into_iter().next().unwrap(),
            DEFAULT_FEED_PROXY_ADDRESS,
            "0x1111111111111111111111111111111111111111",
            3,
            8,
            &block,
            Utc.with_ymd_and_hms(2026, 3, 21, 12, 0, 4).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn default_config_is_the_seed_contract() {
        let config = PolygonChainlinkBtcusdOracleConfig::default();
        assert_eq!(config.rpc_url, DEFAULT_RPC_URL);
        assert_eq!(config.archive_log_rpc_url, DEFAULT_ARCHIVE_LOG_RPC_URL);
        assert_eq!(config.feed_proxy_address, DEFAULT_FEED_PROXY_ADDRESS);
        assert_eq!(config.poll_interval_seconds, 15);
        assert_eq!(config.confirmation_depth, 128);
        assert_eq!(config.maximum_block_range, 30_000);
        assert_eq!(config.startup_lookback_blocks, 43_200);
        assert_eq!(config.overlap_blocks, 256);
        assert_eq!(config.artifact_window_seconds, 3_600);
        assert_eq!(config.request_timeout_seconds, 30);
        config.validate().unwrap();
    }

    #[test]
    fn config_is_typed_narrow_and_credential_free() {
        let mut value =
            serde_json::to_value(PolygonChainlinkBtcusdOracleConfig::default()).unwrap();
        value["unexpected"] = json!(true);
        assert!(PolygonChainlinkBtcusdOracleConfig::from_value(&value).is_err());

        let mut config = PolygonChainlinkBtcusdOracleConfig {
            rpc_url: "https://user:secret@polygon-bor-rpc.publicnode.com".to_owned(),
            ..PolygonChainlinkBtcusdOracleConfig::default()
        };
        assert!(config.validate().is_err());
        config.rpc_url = "https://example.com".to_owned();
        assert!(config.validate().is_err());
        let same_provider = PolygonChainlinkBtcusdOracleConfig {
            archive_log_rpc_url: DEFAULT_RPC_URL.to_owned(),
            ..PolygonChainlinkBtcusdOracleConfig::default()
        };
        assert!(same_provider.validate().is_err());
    }

    #[test]
    fn checkpoint_requires_a_paired_canonical_hash() {
        assert!(OracleCheckpoint::default().validate().is_ok());
        assert!(OracleCheckpoint {
            last_finalized_block: Some(100),
            last_finalized_block_hash: None,
        }
        .validate()
        .is_err());
        assert!(OracleCheckpoint {
            last_finalized_block: Some(100),
            last_finalized_block_hash: Some(format!("0x{}", "A".repeat(64))),
        }
        .validate()
        .is_err());
        assert!(OracleCheckpoint {
            last_finalized_block: Some(100),
            last_finalized_block_hash: Some(format!("0x{}", "a".repeat(64))),
        }
        .validate()
        .is_ok());
    }

    #[test]
    fn finalized_scan_rewinds_overlap_and_bounds_startup() {
        assert_eq!(scan_start(Some(1_000), 1_100, 256, 43_200), 745);
        assert_eq!(scan_start(Some(10), 20, 256, 43_200), 0);
        assert_eq!(scan_start(None, 50_000, 256, 43_200), 6_801);
        assert_eq!(scan_start(None, 100, 256, 43_200), 0);
    }

    #[test]
    fn computes_canonical_chainlink_selectors_and_topic() {
        assert_eq!(abi_calldata("decimals()", &[]), "0x313ce567");
        assert_eq!(abi_calldata("phaseId()", &[]), "0x58303b10");
        assert_eq!(
            event_topic("AnswerUpdated(int256,uint256,uint256)"),
            "0x0559884fd3a460db3073b7fc896cc77986f16e378210ded43186175bf646fc5f"
        );
    }

    #[test]
    fn latest_metadata_snapshot_is_strict_and_upgrade_races_fail_closed() {
        let word = |value: u64| format!("0x{value:064x}");
        assert_eq!(decode_feed_decimals(&word(8)).unwrap(), 8);
        assert_eq!(decode_phase_count(&word(3)).unwrap(), 3);
        assert!(decode_feed_decimals(&word(19)).is_err());
        assert!(decode_phase_count(&word(0)).is_err());
        assert!(decode_phase_count(&word(129)).is_err());
        validate_latest_metadata_snapshot(8, 3, 8, 3).unwrap();

        let phase_race = validate_latest_metadata_snapshot(8, 3, 8, 4).unwrap_err();
        assert_eq!(phase_race.code, "polygon_oracle_metadata_snapshot_changed");
        assert_eq!(phase_race.kind, StrategyErrorKind::TransientSource);
        let decimals_race = validate_latest_metadata_snapshot(8, 3, 18, 3).unwrap_err();
        assert_eq!(
            decimals_race.code,
            "polygon_oracle_metadata_snapshot_changed"
        );
    }

    #[test]
    fn latest_phase_mapping_is_an_append_only_superset_for_finalized_logs() {
        let aggregators = BTreeMap::from([
            ("0x1111111111111111111111111111111111111111".to_owned(), 1),
            ("0x2222222222222222222222222222222222222222".to_owned(), 2),
            ("0x3333333333333333333333333333333333333333".to_owned(), 3),
        ]);
        validate_phase_mapping(3, &aggregators).unwrap();
        assert!(aggregators.values().any(|phase_id| *phase_id == 2));

        let missing_finalized_phase = BTreeMap::from([
            ("0x1111111111111111111111111111111111111111".to_owned(), 1),
            ("0x3333333333333333333333333333333333333333".to_owned(), 3),
        ]);
        let error = validate_phase_mapping(3, &missing_finalized_phase).unwrap_err();
        assert_eq!(error.code, "polygon_oracle_phase_mapping_incomplete");
    }

    #[test]
    fn sanitized_fixture_decodes_exact_price_and_provenance() {
        let received_at = Utc.with_ymd_and_hms(2026, 3, 21, 12, 0, 4).unwrap();
        let round = fixture_round();

        assert_eq!(round.chain_id, 137);
        assert_eq!(round.phase_id, 3);
        assert_eq!(round.aggregator_round_id, 42);
        assert_eq!(round.answer_raw, Decimal::new(8_412_345_678_901, 0));
        assert_eq!(round.price, Decimal::new(8_412_345_678_901, 8));
        assert_eq!(round.source_timestamp.timestamp(), 1_774_094_401);
        assert_eq!(round.block_timestamp.timestamp(), 1_774_094_403);
        assert_eq!(round.provider_available_at, round.block_timestamp);
        assert_eq!(round.received_at, received_at);
        assert_eq!(round.block_number, 100);
        assert_eq!(round.log_index, 2);
        assert_eq!(round.payload_sha256.len(), 64);
        assert_eq!(round.source_payload["removed"], json!(false));
    }

    #[test]
    fn empty_startup_scan_cannot_create_an_unanchored_checkpoint() {
        assert!(!empty_scan_may_advance(&OracleCheckpoint::default()));
        let anchored = OracleCheckpoint {
            last_finalized_block: Some(100),
            last_finalized_block_hash: Some(format!("0x{}", "a".repeat(64))),
        };
        assert!(empty_scan_may_advance(&anchored));

        assert!(validate_durable_cursor_state(None, None).is_ok());
        let error = validate_durable_cursor_state(None, Some(100)).unwrap_err();
        assert_eq!(error.code, "polygon_oracle_checkpoint_without_anchor");
        assert!(validate_durable_cursor_state(Some(99), Some(100)).is_ok());
        let error = validate_durable_cursor_state(Some(101), Some(100)).unwrap_err();
        assert_eq!(error.code, "polygon_oracle_checkpoint_behind_facts");
    }

    #[test]
    fn cross_checked_log_anchor_detects_a_truncated_leading_round() {
        assert_eq!(prior_anchor_range(10_000, 30_000), Some((7_952, 9_999)));
        assert_eq!(prior_anchor_range(100, 1), Some((99, 99)));
        assert_eq!(prior_anchor_range(0, 30_000), None);

        let first_receipt = Utc.with_ymd_and_hms(2026, 3, 21, 12, 0, 1).unwrap();
        let second_receipt = Utc.with_ymd_and_hms(2026, 3, 21, 12, 0, 2).unwrap();
        let archive_header_receipt = Utc.with_ymd_and_hms(2026, 3, 21, 12, 0, 3).unwrap();
        let canonical_header_receipt = Utc.with_ymd_and_hms(2026, 3, 21, 12, 0, 4).unwrap();
        let anchor_round = fixture_round();
        let verified = cross_checked_startup_batch(
            DecodedLogBatch {
                rounds: vec![anchor_round.clone()],
                received_at: first_receipt,
            },
            DecodedLogBatch {
                rounds: vec![anchor_round],
                received_at: second_receipt,
            },
            archive_header_receipt,
            canonical_header_receipt,
        )
        .unwrap();
        assert_eq!(verified.received_at, canonical_header_receipt);
        assert_eq!(verified.rounds[0].received_at, canonical_header_receipt);

        let mut mismatch = fixture_round();
        mismatch.aggregator_round_id += 1;
        let error = cross_checked_startup_batch(
            DecodedLogBatch {
                rounds: vec![fixture_round()],
                received_at: first_receipt,
            },
            DecodedLogBatch {
                rounds: vec![mismatch],
                received_at: second_receipt,
            },
            archive_header_receipt,
            canonical_header_receipt,
        )
        .unwrap_err();
        assert_eq!(
            error.code,
            "polygon_oracle_startup_anchor_provider_mismatch"
        );

        let boundary = RoundBoundary {
            phase_id: 3,
            aggregator_round_id: 40,
            source_timestamp: Utc.with_ymd_and_hms(2026, 3, 21, 11, 59, 58).unwrap(),
            block_number: 98,
        };
        let mut first_returned = fixture_round();
        first_returned.aggregator_round_id = 42;
        let gaps = detect_round_gaps_from_boundaries(
            BTreeMap::from([(boundary.phase_id, boundary)]),
            &[first_returned],
        )
        .unwrap();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start_cursor, "phase:3:round:41:block:98");
        assert_eq!(gaps[0].end_cursor, "phase:3:round:41:block:100");
    }

    #[test]
    fn fact_receipt_uses_the_latest_full_body_rpc_receipt() {
        let log_receipt = Utc.with_ymd_and_hms(2026, 3, 21, 12, 0, 1).unwrap();
        let first_header = Utc.with_ymd_and_hms(2026, 3, 21, 12, 0, 3).unwrap();
        let last_header = Utc.with_ymd_and_hms(2026, 3, 21, 12, 0, 5).unwrap();
        assert_eq!(
            maximum_receipt(log_receipt, [first_header, last_header]),
            last_header
        );
        assert_eq!(maximum_receipt(last_header, [log_receipt]), last_header);
    }

    #[test]
    fn round_gap_repair_is_bounded_and_terminal_only_after_successful_attempts() {
        let repair = parse_round_gap_cursors(
            Some("phase:3:round:41:block:98"),
            Some("phase:3:round:43:block:102"),
        )
        .unwrap();
        assert_eq!(
            repair,
            RoundGapRepair {
                phase_id: 3,
                start_round: 41,
                end_round: 43,
                from_block: 98,
                to_block: 102,
            }
        );
        assert!(repair.is_resource_bounded(30_000));
        assert!(round_gap_ids_complete(&repair, &[41, 42, 43]).unwrap());
        assert!(!round_gap_ids_complete(&repair, &[41, 43]).unwrap());
        assert!(round_gap_ids_complete(&repair, &[41, 41, 43]).is_err());
        assert_eq!(
            classify_round_gap_outcome(true, 1),
            RoundGapOutcome::Repaired
        );
        assert_eq!(
            classify_round_gap_outcome(false, MAX_GAP_REPAIR_ATTEMPTS - 1),
            RoundGapOutcome::Retry
        );
        assert_eq!(
            classify_round_gap_outcome(false, MAX_GAP_REPAIR_ATTEMPTS),
            RoundGapOutcome::Unrecoverable
        );

        let oversized = RoundGapRepair {
            end_round: 41 + i64::try_from(MAX_GAP_REPAIR_ROUNDS).unwrap(),
            ..repair
        };
        assert!(!oversized.is_resource_bounded(30_000));
        assert!(parse_round_gap_cursors(
            Some("phase:3:round:43:block:102"),
            Some("phase:4:round:41:block:98"),
        )
        .is_err());
    }

    #[test]
    fn gap_detection_uses_canonical_event_order_and_persists_repair_brackets() {
        let mut previous = BTreeMap::new();
        previous.insert(
            3,
            RoundBoundary {
                phase_id: 3,
                aggregator_round_id: 40,
                source_timestamp: Utc.with_ymd_and_hms(2026, 3, 21, 11, 59, 58).unwrap(),
                block_number: 98,
            },
        );
        let mut round_41 = fixture_round();
        round_41.aggregator_round_id = 41;
        round_41.block_number = 99;
        round_41.log_index = 1;
        round_41.transaction_hash = format!("0x{}", "b".repeat(64));
        let round_42 = fixture_round();

        let gaps = detect_round_gaps_from_boundaries(
            previous.clone(),
            &[round_42.clone(), round_41.clone()],
        )
        .unwrap();
        assert!(gaps.is_empty());

        let gaps =
            detect_round_gaps_from_boundaries(previous, std::slice::from_ref(&round_42)).unwrap();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start_cursor, "phase:3:round:41:block:98");
        assert_eq!(gaps[0].end_cursor, "phase:3:round:41:block:100");

        let mut regressing = round_41;
        regressing.block_number = 101;
        regressing.log_index = 0;
        let error = detect_round_gaps_from_boundaries(BTreeMap::new(), &[round_42, regressing])
            .unwrap_err();
        assert_eq!(error.code, "polygon_oracle_round_sequence_regressed");
    }

    #[test]
    fn artifact_seal_checks_count_and_hashes_the_transaction_snapshot() {
        let artifact_id = Uuid::nil();
        let timestamp = Utc.with_ymd_and_hms(2026, 3, 21, 12, 0, 1).unwrap();
        let row = ArtifactChecksumRow {
            source_timestamp: timestamp,
            block_number: 100,
            transaction_hash: format!("0x{}", "c".repeat(64)),
            log_index: 2,
            payload_sha256: "d".repeat(64),
        };
        let first = artifact_seal_from_rows(artifact_id, 1, std::slice::from_ref(&row)).unwrap();
        let second = artifact_seal_from_rows(artifact_id, 1, &[row]).unwrap();
        assert_eq!(first.content_sha256, second.content_sha256);
        assert_eq!(first.content_sha256.len(), 64);
        assert_eq!(
            first.end_cursor.as_deref(),
            Some(format!("block:100:tx:0x{}:log:2", "c".repeat(64)).as_str())
        );
        let empty = artifact_seal_from_rows(artifact_id, 0, &[]).unwrap();
        assert_eq!(
            empty.content_sha256,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(empty.end_cursor, None);
        let error = artifact_seal_from_rows(artifact_id, 2, &[]).unwrap_err();
        assert_eq!(error.code, "polygon_oracle_artifact_count_mismatch");
    }

    #[test]
    fn removed_fixture_never_becomes_a_fact() {
        let log: RpcLog = serde_json::from_slice(include_bytes!(
            "../../../tests/fixtures/polygon/removed_answer_updated_log_v1.json"
        ))
        .unwrap();
        let block = BlockHeader {
            number: 100,
            hash: format!("0x{}", "a".repeat(64)),
            timestamp: Utc.with_ymd_and_hms(2026, 3, 21, 12, 0, 3).unwrap(),
        };
        let error = decode_answer_updated(
            log,
            DEFAULT_FEED_PROXY_ADDRESS,
            "0x1111111111111111111111111111111111111111",
            3,
            8,
            &block,
            Utc::now(),
        )
        .unwrap_err();
        assert_eq!(error.code, "polygon_removed_oracle_log");
    }

    #[test]
    fn rejects_malformed_quantities_answers_and_abi_values() {
        assert!(parse_quantity_u64("64").is_err());
        assert!(parse_quantity_u64("0x00").is_err());
        assert!(parse_quantity_u64("0x0").is_ok());
        assert!(parse_positive_i128_word(
            "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        )
        .is_err());
        assert!(parse_abi_u64(
            "0x0000000000000001000000000000000000000000000000000000000000000000"
        )
        .is_err());
    }
}
