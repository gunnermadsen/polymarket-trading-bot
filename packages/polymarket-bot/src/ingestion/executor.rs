use std::{fmt, path::PathBuf};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, NaiveDate, Utc};
use rust_decimal::Decimal;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::{fs, time::timeout};

use super::{
    binance_archive::{
        download_archive_with_cancellation, fetch_expected_checksum_with_cancellation,
        spawn_aggregate_trade_parser_with_control, spawn_one_second_kline_parser_with_control,
        ArchiveCancellation, ArchiveDownloadLimits, ArchiveParseSummary, BinanceArchiveKind,
        BinanceArchiveSpec, BINANCE_ARCHIVE_PROVIDER,
    },
    job::{
        ArtifactCompletion, ArtifactDisposition, ArtifactSpec, BackfillArtifactStatus,
        BackfillCheckpoint, BackfillFailureKind, BackfillJobSummary, BackfillProgress,
        BtcIntervalMarket, BtcOutcome, BtcReferenceFact, BtcReferenceFactType, ClaimedJob,
        IngesterKey, WorkerControl,
    },
    repository::IngestionRepository,
};

const MAX_SOURCE_BODY_BYTES: usize = 4 * 1024 * 1024;
const SOURCE_CHUNK_TIMEOUT_SECS: u64 = 30;
const MAX_UNCOMPRESSED_ARCHIVE_BYTES: u64 = 128 * 1024 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct IngestionExecutorConfig {
    pub gamma_base_url: String,
    pub clob_base_url: String,
    pub binance_archive_base_url: String,
    pub cache_directory: PathBuf,
    pub batch_rows: usize,
}

impl IngestionExecutorConfig {
    pub fn validate(&self) -> Result<()> {
        if self.gamma_base_url.trim().is_empty()
            || self.clob_base_url.trim().is_empty()
            || self.binance_archive_base_url.trim().is_empty()
        {
            bail!("backfill source base URLs must not be empty");
        }
        if !(1..=4_000).contains(&self.batch_rows) {
            bail!("POLYMARKET_BACKFILL_BATCH_ROWS must be between 1 and 4000");
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct IngestionExecutionError {
    pub kind: BackfillFailureKind,
    message: String,
}

impl IngestionExecutionError {
    fn transient(error: impl fmt::Display) -> Self {
        Self {
            kind: BackfillFailureKind::Transient,
            message: error.to_string(),
        }
    }

    fn permanent(error: impl fmt::Display) -> Self {
        Self {
            kind: BackfillFailureKind::Permanent,
            message: error.to_string(),
        }
    }
}

impl fmt::Display for IngestionExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for IngestionExecutionError {}

#[derive(Clone)]
pub struct IngestionExecutor {
    repository: IngestionRepository,
    client: reqwest::Client,
    config: IngestionExecutorConfig,
}

impl IngestionExecutor {
    pub fn new(
        repository: IngestionRepository,
        client: reqwest::Client,
        config: IngestionExecutorConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            repository,
            client,
            config,
        })
    }

    pub async fn execute(
        &self,
        claim: &ClaimedJob,
        cancellation: ArchiveCancellation,
    ) -> std::result::Result<BackfillJobSummary, IngestionExecutionError> {
        let ingester = claim
            .job
            .ingester()
            .map_err(IngestionExecutionError::permanent)?;
        let range_start = claim
            .job
            .range_start
            .ok_or_else(|| IngestionExecutionError::permanent("job is missing range_start"))?;
        let range_end = claim
            .job
            .range_end
            .ok_or_else(|| IngestionExecutionError::permanent("job is missing range_end"))?;
        let progress = serde_json::from_value::<BackfillProgress>(claim.job.progress.clone())
            .unwrap_or(BackfillProgress {
                expected_work_units: expected_units(ingester, range_start, range_end),
                ..BackfillProgress::default()
            });
        match ingester {
            IngesterKey::BtcFiveMinuteMarkets => {
                self.ingest_btc_markets(claim, range_start, range_end, progress, &cancellation)
                    .await
            }
            IngesterKey::BtcFiveMinuteResolutions => {
                self.ingest_btc_resolutions(claim, range_start, range_end, progress, &cancellation)
                    .await
            }
            IngesterKey::BinanceBtcusdtAggTrades => {
                self.ingest_binance(
                    claim,
                    range_start,
                    range_end,
                    progress,
                    BinanceArchiveKind::AggregateTrades,
                    cancellation,
                )
                .await
            }
            IngesterKey::BinanceBtcusdtOneSecondKlines => {
                self.ingest_binance(
                    claim,
                    range_start,
                    range_end,
                    progress,
                    BinanceArchiveKind::OneSecondKlines,
                    cancellation,
                )
                .await
            }
        }
    }

    async fn ingest_btc_markets(
        &self,
        claim: &ClaimedJob,
        range_start: DateTime<Utc>,
        range_end: DateTime<Utc>,
        mut progress: BackfillProgress,
        cancellation: &ArchiveCancellation,
    ) -> std::result::Result<BackfillJobSummary, IngestionExecutionError> {
        let mut summary = summary_from_progress(&progress);
        let mut window_start = checkpoint_window_start(claim, range_start, 300);
        while window_start < range_end {
            self.ensure_continue(claim, cancellation).await?;
            let slug = slug_for_window(window_start);
            let source_uri = format!(
                "{}/events/slug/{slug}",
                self.config.gamma_base_url.trim_end_matches('/')
            );
            progress.current_logical_key = Some(slug.clone());
            let prepared = self
                .repository
                .prepare_artifact(
                    claim,
                    &ArtifactSpec {
                        job_id: claim.job.job_id,
                        ingester: IngesterKey::BtcFiveMinuteMarkets,
                        logical_key: format!("gamma:event:{slug}"),
                        provider: "polymarket_gamma".to_string(),
                        source_uri: source_uri.clone(),
                        source_date: Some(window_start.date_naive()),
                        expected_checksum: None,
                        metadata: serde_json::json!({"slug": slug, "window_start": window_start}),
                    },
                )
                .await
                .map_err(IngestionExecutionError::transient)?;
            if prepared.disposition == ArtifactDisposition::AlreadyCompleted {
                observe_reused_artifact(&mut progress, &mut summary, &prepared.artifact);
                window_start += ChronoDuration::minutes(5);
                self.finish_work_unit(claim, &mut progress, window_start, None)
                    .await?;
                continue;
            }
            self.set_artifact_status(
                claim,
                prepared.artifact.artifact_id,
                BackfillArtifactStatus::Downloading,
            )
            .await?;
            let body = match self
                .fetch_bounded_json_bytes(&source_uri, cancellation)
                .await
            {
                Ok(Some(body)) => body,
                Ok(None) => {
                    let _ = self
                        .repository
                        .fail_artifact(claim, prepared.artifact.artifact_id, "Gamma returned 404")
                        .await;
                    increment_missing(&mut summary, "missing_gamma_market");
                    summary.artifacts_failed = summary.artifacts_failed.saturating_add(1);
                    window_start += ChronoDuration::minutes(5);
                    self.finish_work_unit(claim, &mut progress, window_start, None)
                        .await?;
                    continue;
                }
                Err(error) => {
                    let _ = self
                        .repository
                        .fail_artifact(claim, prepared.artifact.artifact_id, &error.to_string())
                        .await;
                    return Err(error);
                }
            };
            let payload_sha256 = sha256(&body);
            let value: Value = serde_json::from_slice(&body).map_err(|error| {
                IngestionExecutionError::permanent(format!(
                    "Gamma event {slug} was invalid JSON: {error}"
                ))
            })?;
            let market = parse_gamma_btc_interval_event(&value, window_start)
                .map_err(IngestionExecutionError::permanent)?;
            self.set_artifact_status(
                claim,
                prepared.artifact.artifact_id,
                BackfillArtifactStatus::Downloaded,
            )
            .await?;
            self.set_artifact_status(
                claim,
                prepared.artifact.artifact_id,
                BackfillArtifactStatus::Verified,
            )
            .await?;
            self.set_artifact_status(
                claim,
                prepared.artifact.artifact_id,
                BackfillArtifactStatus::Ingesting,
            )
            .await?;
            self.ensure_continue(claim, cancellation).await?;
            self.repository
                .upsert_btc_interval_market(claim, prepared.artifact.artifact_id, &market)
                .await
                .map_err(IngestionExecutionError::transient)?;

            let (opening_boundary, final_price) = extract_reference_values(&value);
            if let Some(opening_boundary) = opening_boundary {
                self.persist_reference_fact(
                    claim,
                    prepared.artifact.artifact_id,
                    &market.market_id,
                    BtcReferenceFactType::OpeningBoundary,
                    opening_boundary,
                    window_start,
                    &payload_sha256,
                    &value,
                )
                .await?;
            } else {
                increment_missing(&mut summary, "missing_opening_boundary");
            }
            if let Some(final_price) = final_price {
                self.persist_reference_fact(
                    claim,
                    prepared.artifact.artifact_id,
                    &market.market_id,
                    BtcReferenceFactType::FinalPrice,
                    final_price,
                    market.window_end,
                    &payload_sha256,
                    &value,
                )
                .await?;
            } else {
                increment_missing(&mut summary, "missing_final_price");
            }
            self.repository
                .complete_artifact(
                    claim,
                    prepared.artifact.artifact_id,
                    &ArtifactCompletion {
                        actual_checksum: payload_sha256,
                        compressed_bytes: u64::try_from(body.len()).unwrap_or(u64::MAX),
                        record_count: 1,
                        minimum_source_timestamp: Some(market.window_start),
                        maximum_source_timestamp: Some(market.window_end),
                        metadata: serde_json::json!({"market_id": market.market_id}),
                    },
                )
                .await
                .map_err(IngestionExecutionError::transient)?;
            progress.records_read = progress.records_read.saturating_add(1);
            progress.records_committed = progress.records_committed.saturating_add(1);
            summary.records_read = summary.records_read.saturating_add(1);
            summary.records_committed = summary.records_committed.saturating_add(1);
            summary.artifacts_completed = summary.artifacts_completed.saturating_add(1);
            window_start += ChronoDuration::minutes(5);
            self.finish_work_unit(claim, &mut progress, window_start, None)
                .await?;
        }
        summary.completed_work_units = progress.completed_work_units;
        Ok(summary)
    }

    async fn ingest_btc_resolutions(
        &self,
        claim: &ClaimedJob,
        range_start: DateTime<Utc>,
        range_end: DateTime<Utc>,
        mut progress: BackfillProgress,
        cancellation: &ArchiveCancellation,
    ) -> std::result::Result<BackfillJobSummary, IngestionExecutionError> {
        let mut summary = summary_from_progress(&progress);
        let candidates = self
            .repository
            .resolution_candidates(range_start, range_end)
            .await
            .map_err(IngestionExecutionError::transient)?;
        let expected = u64::try_from((range_end - range_start).num_seconds() / 300).unwrap_or(0);
        let candidate_count = u64::try_from(candidates.len()).unwrap_or(u64::MAX);
        if candidate_count < expected {
            summary.missing_by_reason.insert(
                "missing_market_definition".to_string(),
                expected.saturating_sub(candidate_count),
            );
        }
        for candidate in candidates {
            if candidate.market.window_start < checkpoint_window_start(claim, range_start, 300) {
                continue;
            }
            self.ensure_continue(claim, cancellation).await?;
            progress.current_logical_key = Some(candidate.market.condition_id.clone());
            if candidate.official_outcome.is_some() {
                let next = candidate.market.window_start + ChronoDuration::minutes(5);
                self.finish_work_unit(claim, &mut progress, next, None)
                    .await?;
                continue;
            }
            let source_uri = format!(
                "{}/markets/{}",
                self.config.clob_base_url.trim_end_matches('/'),
                candidate.market.condition_id
            );
            let prepared = self
                .repository
                .prepare_artifact(
                    claim,
                    &ArtifactSpec {
                        job_id: claim.job.job_id,
                        ingester: IngesterKey::BtcFiveMinuteResolutions,
                        logical_key: format!("clob:market:{}", candidate.market.condition_id),
                        provider: "polymarket_clob".to_string(),
                        source_uri: source_uri.clone(),
                        source_date: Some(candidate.market.window_start.date_naive()),
                        expected_checksum: None,
                        metadata: serde_json::json!({"market_id": candidate.market.market_id}),
                    },
                )
                .await
                .map_err(IngestionExecutionError::transient)?;
            if prepared.disposition == ArtifactDisposition::AlreadyCompleted {
                observe_reused_artifact(&mut progress, &mut summary, &prepared.artifact);
                let next = candidate.market.window_start + ChronoDuration::minutes(5);
                self.finish_work_unit(claim, &mut progress, next, None)
                    .await?;
                continue;
            }
            self.set_artifact_status(
                claim,
                prepared.artifact.artifact_id,
                BackfillArtifactStatus::Downloading,
            )
            .await?;
            let body = match self
                .fetch_bounded_json_bytes(&source_uri, cancellation)
                .await
            {
                Ok(Some(body)) => body,
                Ok(None) => {
                    let _ = self
                        .repository
                        .fail_artifact(claim, prepared.artifact.artifact_id, "CLOB returned 404")
                        .await;
                    increment_missing(&mut summary, "missing_clob_resolution");
                    summary.artifacts_failed = summary.artifacts_failed.saturating_add(1);
                    let next = candidate.market.window_start + ChronoDuration::minutes(5);
                    self.finish_work_unit(claim, &mut progress, next, None)
                        .await?;
                    continue;
                }
                Err(error) => {
                    let _ = self
                        .repository
                        .fail_artifact(claim, prepared.artifact.artifact_id, &error.to_string())
                        .await;
                    return Err(error);
                }
            };
            let payload_sha256 = sha256(&body);
            let value: Value = serde_json::from_slice(&body).map_err(|error| {
                IngestionExecutionError::permanent(format!(
                    "CLOB market {} was invalid JSON: {error}",
                    candidate.market.market_id
                ))
            })?;
            let observed_at = Utc::now();
            let resolution =
                parse_clob_rest_official_resolution(&value, &candidate.market, observed_at)
                    .map_err(IngestionExecutionError::permanent)?;
            let Some(resolution) = resolution else {
                let _ = self
                    .repository
                    .fail_artifact(
                        claim,
                        prepared.artifact.artifact_id,
                        "CLOB market was not officially resolved",
                    )
                    .await;
                increment_missing(&mut summary, "unresolved_clob_market");
                summary.artifacts_failed = summary.artifacts_failed.saturating_add(1);
                let next = candidate.market.window_start + ChronoDuration::minutes(5);
                self.finish_work_unit(claim, &mut progress, next, None)
                    .await?;
                continue;
            };
            self.set_artifact_status(
                claim,
                prepared.artifact.artifact_id,
                BackfillArtifactStatus::Ingesting,
            )
            .await?;
            self.ensure_continue(claim, cancellation).await?;
            self.repository
                .persist_official_market_resolution(
                    claim,
                    prepared.artifact.artifact_id,
                    &candidate.market,
                    &resolution.winning_token_id,
                    resolution.winning_outcome,
                    resolution.observed_at,
                    &value,
                )
                .await
                .map_err(IngestionExecutionError::transient)?;
            self.repository
                .complete_artifact(
                    claim,
                    prepared.artifact.artifact_id,
                    &ArtifactCompletion {
                        actual_checksum: payload_sha256,
                        compressed_bytes: u64::try_from(body.len()).unwrap_or(u64::MAX),
                        record_count: 1,
                        minimum_source_timestamp: None,
                        maximum_source_timestamp: None,
                        metadata: serde_json::json!({
                            "market_id": candidate.market.market_id,
                            "winning_token_id": resolution.winning_token_id,
                        }),
                    },
                )
                .await
                .map_err(IngestionExecutionError::transient)?;
            progress.records_read = progress.records_read.saturating_add(1);
            progress.records_committed = progress.records_committed.saturating_add(1);
            summary.records_read = summary.records_read.saturating_add(1);
            summary.records_committed = summary.records_committed.saturating_add(1);
            summary.artifacts_completed = summary.artifacts_completed.saturating_add(1);
            let next = candidate.market.window_start + ChronoDuration::minutes(5);
            self.finish_work_unit(claim, &mut progress, next, None)
                .await?;
        }
        summary.completed_work_units = progress.completed_work_units;
        Ok(summary)
    }

    async fn ingest_binance(
        &self,
        claim: &ClaimedJob,
        range_start: DateTime<Utc>,
        range_end: DateTime<Utc>,
        mut progress: BackfillProgress,
        kind: BinanceArchiveKind,
        cancellation: ArchiveCancellation,
    ) -> std::result::Result<BackfillJobSummary, IngestionExecutionError> {
        let mut summary = summary_from_progress(&progress);
        let mut date = checkpoint_date(claim).unwrap_or(range_start.date_naive());
        let end_date = range_end.date_naive();
        while date < end_date {
            self.ensure_continue(claim, &cancellation).await?;
            let spec = BinanceArchiveSpec::new(&self.config.binance_archive_base_url, kind, date);
            progress.current_logical_key = Some(spec.logical_key.clone());
            let expected_checksum =
                fetch_expected_checksum_with_cancellation(&self.client, &spec, &cancellation)
                    .await
                    .map_err(IngestionExecutionError::transient)?;
            let prepared = self
                .repository
                .prepare_artifact(
                    claim,
                    &ArtifactSpec {
                        job_id: claim.job.job_id,
                        ingester: claim
                            .job
                            .ingester()
                            .map_err(IngestionExecutionError::permanent)?,
                        logical_key: spec.logical_key.clone(),
                        provider: BINANCE_ARCHIVE_PROVIDER.to_string(),
                        source_uri: spec.source_uri.clone(),
                        source_date: Some(date),
                        expected_checksum: Some(expected_checksum.clone()),
                        metadata: serde_json::json!({"archive_file": spec.file_name}),
                    },
                )
                .await
                .map_err(IngestionExecutionError::transient)?;
            if prepared.disposition == ArtifactDisposition::AlreadyCompleted {
                observe_reused_artifact(&mut progress, &mut summary, &prepared.artifact);
                date += ChronoDuration::days(1);
                self.finish_work_unit(claim, &mut progress, range_end, Some(date))
                    .await?;
                continue;
            }
            self.set_artifact_status(
                claim,
                prepared.artifact.artifact_id,
                BackfillArtifactStatus::Downloading,
            )
            .await?;
            let archive = match download_archive_with_cancellation(
                &self.client,
                &spec,
                &expected_checksum,
                &self.config.cache_directory,
                &ArchiveDownloadLimits::default(),
                &cancellation,
            )
            .await
            {
                Ok(archive) => archive,
                Err(error) => {
                    let _ = self
                        .repository
                        .fail_artifact(claim, prepared.artifact.artifact_id, &error.to_string())
                        .await;
                    return Err(IngestionExecutionError::transient(error));
                }
            };
            if !archive.reused_cache {
                progress.bytes_downloaded = progress
                    .bytes_downloaded
                    .saturating_add(archive.compressed_bytes);
            }
            self.set_artifact_status(
                claim,
                prepared.artifact.artifact_id,
                BackfillArtifactStatus::Downloaded,
            )
            .await?;
            self.set_artifact_status(
                claim,
                prepared.artifact.artifact_id,
                BackfillArtifactStatus::Verified,
            )
            .await?;
            self.set_artifact_status(
                claim,
                prepared.artifact.artifact_id,
                BackfillArtifactStatus::Ingesting,
            )
            .await?;

            let parse_summary = match kind {
                BinanceArchiveKind::AggregateTrades => {
                    let (mut receiver, handle) = spawn_aggregate_trade_parser_with_control(
                        archive.path.clone(),
                        spec.clone(),
                        self.config.batch_rows,
                        MAX_UNCOMPRESSED_ARCHIVE_BYTES,
                        cancellation.clone(),
                    )
                    .map_err(IngestionExecutionError::permanent)?;
                    while let Some(batch) = receiver.recv().await {
                        self.ensure_continue(claim, &cancellation).await?;
                        let result = self
                            .repository
                            .insert_aggregate_trade_batch(
                                claim,
                                prepared.artifact.artifact_id,
                                &batch,
                            )
                            .await
                            .map_err(IngestionExecutionError::transient)?;
                        observe_batch(&mut progress, &mut summary, result);
                        self.update_batch_checkpoint(claim, &progress, date).await?;
                    }
                    handle
                        .await
                        .map_err(IngestionExecutionError::transient)?
                        .map_err(IngestionExecutionError::permanent)?
                }
                BinanceArchiveKind::OneSecondKlines => {
                    let (mut receiver, handle) = spawn_one_second_kline_parser_with_control(
                        archive.path.clone(),
                        spec.clone(),
                        self.config.batch_rows,
                        MAX_UNCOMPRESSED_ARCHIVE_BYTES,
                        cancellation.clone(),
                    )
                    .map_err(IngestionExecutionError::permanent)?;
                    while let Some(batch) = receiver.recv().await {
                        self.ensure_continue(claim, &cancellation).await?;
                        let result = self
                            .repository
                            .insert_one_second_kline_batch(
                                claim,
                                prepared.artifact.artifact_id,
                                &batch,
                            )
                            .await
                            .map_err(IngestionExecutionError::transient)?;
                        observe_batch(&mut progress, &mut summary, result);
                        self.update_batch_checkpoint(claim, &progress, date).await?;
                    }
                    handle
                        .await
                        .map_err(IngestionExecutionError::transient)?
                        .map_err(IngestionExecutionError::permanent)?
                }
            };
            validate_archive_coverage(kind, date, &parse_summary)
                .map_err(IngestionExecutionError::permanent)?;
            self.repository
                .complete_artifact(
                    claim,
                    prepared.artifact.artifact_id,
                    &ArtifactCompletion {
                        actual_checksum: archive.sha256,
                        compressed_bytes: archive.compressed_bytes,
                        record_count: parse_summary.records,
                        minimum_source_timestamp: parse_summary.minimum_timestamp,
                        maximum_source_timestamp: parse_summary.maximum_timestamp,
                        metadata: serde_json::json!({
                            "batches": parse_summary.batches,
                            "maximum_batch_records": parse_summary.maximum_batch_records,
                        }),
                    },
                )
                .await
                .map_err(IngestionExecutionError::transient)?;
            summary.artifacts_completed = summary.artifacts_completed.saturating_add(1);
            if let Err(error) = fs::remove_file(&archive.path).await {
                let _ = self
                    .repository
                    .append_event(
                        claim.job.job_id,
                        super::job::BackfillEventLevel::Warn,
                        "completed archive cache cleanup failed",
                        serde_json::json!({"path": archive.path, "error": error.to_string()}),
                    )
                    .await;
            }
            date += ChronoDuration::days(1);
            self.finish_work_unit(claim, &mut progress, range_end, Some(date))
                .await?;
        }
        summary.completed_work_units = progress.completed_work_units;
        Ok(summary)
    }

    async fn fetch_bounded_json_bytes(
        &self,
        source_uri: &str,
        cancellation: &ArchiveCancellation,
    ) -> std::result::Result<Option<Vec<u8>>, IngestionExecutionError> {
        self.ensure_not_cancelled(cancellation)?;
        let mut response = self
            .client
            .get(source_uri)
            .send()
            .await
            .map_err(IngestionExecutionError::transient)?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !response.status().is_success() {
            let message = format!("source {source_uri} returned HTTP {}", response.status());
            return if response.status().is_server_error()
                || response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS
            {
                Err(IngestionExecutionError::transient(message))
            } else {
                Err(IngestionExecutionError::permanent(message))
            };
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_SOURCE_BODY_BYTES as u64)
        {
            return Err(IngestionExecutionError::permanent(format!(
                "source {source_uri} exceeded its body limit"
            )));
        }
        let mut body = Vec::with_capacity(16 * 1024);
        loop {
            self.ensure_not_cancelled(cancellation)?;
            let chunk = timeout(
                std::time::Duration::from_secs(SOURCE_CHUNK_TIMEOUT_SECS),
                response.chunk(),
            )
            .await
            .map_err(|_| IngestionExecutionError::transient("source response stalled"))?
            .map_err(IngestionExecutionError::transient)?;
            let Some(chunk) = chunk else { break };
            if body.len().saturating_add(chunk.len()) > MAX_SOURCE_BODY_BYTES {
                return Err(IngestionExecutionError::permanent(format!(
                    "source {source_uri} exceeded its body limit"
                )));
            }
            body.extend_from_slice(&chunk);
        }
        Ok(Some(body))
    }

    async fn persist_reference_fact(
        &self,
        claim: &ClaimedJob,
        artifact_id: uuid::Uuid,
        market_id: &str,
        fact_type: BtcReferenceFactType,
        value: Decimal,
        source_effective_at: DateTime<Utc>,
        payload_sha256: &str,
        evidence: &Value,
    ) -> std::result::Result<(), IngestionExecutionError> {
        self.repository
            .persist_market_fact(
                claim,
                &BtcReferenceFact {
                    market_id: market_id.to_string(),
                    artifact_id,
                    fact_type,
                    value,
                    provider: "polymarket_gamma".to_string(),
                    source_effective_at,
                    fetched_at: Utc::now(),
                    payload_sha256: payload_sha256.to_string(),
                    evidence: evidence.clone(),
                },
            )
            .await
            .map_err(IngestionExecutionError::transient)?;
        Ok(())
    }

    async fn set_artifact_status(
        &self,
        claim: &ClaimedJob,
        artifact_id: uuid::Uuid,
        status: BackfillArtifactStatus,
    ) -> std::result::Result<(), IngestionExecutionError> {
        self.repository
            .set_artifact_status(claim, artifact_id, status, serde_json::json!({}))
            .await
            .map_err(IngestionExecutionError::transient)?;
        Ok(())
    }

    async fn ensure_continue(
        &self,
        claim: &ClaimedJob,
        cancellation: &ArchiveCancellation,
    ) -> std::result::Result<(), IngestionExecutionError> {
        self.ensure_not_cancelled(cancellation)?;
        match self
            .repository
            .is_cancel_requested(claim)
            .await
            .map_err(IngestionExecutionError::transient)?
        {
            WorkerControl::Continue => Ok(()),
            WorkerControl::CancelRequested => Err(IngestionExecutionError::transient(
                "backfill cancellation requested",
            )),
            WorkerControl::LeaseLost => Err(IngestionExecutionError::transient(
                "backfill worker lost its lease",
            )),
        }
    }

    fn ensure_not_cancelled(
        &self,
        cancellation: &ArchiveCancellation,
    ) -> std::result::Result<(), IngestionExecutionError> {
        if cancellation.is_cancelled() {
            Err(IngestionExecutionError::transient(
                "backfill execution was interrupted",
            ))
        } else {
            Ok(())
        }
    }

    async fn update_batch_checkpoint(
        &self,
        claim: &ClaimedJob,
        progress: &BackfillProgress,
        date: NaiveDate,
    ) -> std::result::Result<(), IngestionExecutionError> {
        self.repository
            .update_progress(
                claim,
                progress,
                &BackfillCheckpoint {
                    logical_key: progress.current_logical_key.clone(),
                    source_date: Some(date),
                    committed_record_ordinal: progress.records_read,
                    details: serde_json::json!({}),
                },
            )
            .await
            .map_err(IngestionExecutionError::transient)
    }

    async fn finish_work_unit(
        &self,
        claim: &ClaimedJob,
        progress: &mut BackfillProgress,
        next_window_start: DateTime<Utc>,
        next_source_date: Option<NaiveDate>,
    ) -> std::result::Result<(), IngestionExecutionError> {
        progress.completed_work_units = progress.completed_work_units.saturating_add(1);
        self.repository
            .update_progress(
                claim,
                progress,
                &BackfillCheckpoint {
                    logical_key: progress.current_logical_key.clone(),
                    source_date: next_source_date,
                    committed_record_ordinal: progress.records_read,
                    details: serde_json::json!({"next_window_start": next_window_start}),
                },
            )
            .await
            .map_err(IngestionExecutionError::transient)
    }
}

fn expected_units(
    ingester: IngesterKey,
    range_start: DateTime<Utc>,
    range_end: DateTime<Utc>,
) -> u64 {
    let divisor = if ingester.is_binance() { 86_400 } else { 300 };
    u64::try_from((range_end - range_start).num_seconds() / divisor).unwrap_or(0)
}

fn summary_from_progress(progress: &BackfillProgress) -> BackfillJobSummary {
    BackfillJobSummary {
        expected_work_units: progress.expected_work_units,
        completed_work_units: progress.completed_work_units,
        records_read: progress.records_read,
        records_committed: progress.records_committed,
        ..BackfillJobSummary::default()
    }
}

fn checkpoint_window_start(
    claim: &ClaimedJob,
    default: DateTime<Utc>,
    alignment_seconds: i64,
) -> DateTime<Utc> {
    let candidate = claim
        .job
        .checkpoint
        .pointer("/details/next_window_start")
        .and_then(Value::as_str)
        .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        .map(|value| value.with_timezone(&Utc));
    candidate
        .filter(|value| value.timestamp().rem_euclid(alignment_seconds) == 0)
        .unwrap_or(default)
}

fn checkpoint_date(claim: &ClaimedJob) -> Option<NaiveDate> {
    claim
        .job
        .checkpoint
        .get("source_date")
        .and_then(Value::as_str)
        .and_then(|value| NaiveDate::parse_from_str(value, "%Y-%m-%d").ok())
}

fn observe_batch(
    progress: &mut BackfillProgress,
    summary: &mut BackfillJobSummary,
    result: super::job::BatchWriteResult,
) {
    progress.records_read = progress.records_read.saturating_add(result.input_records);
    progress.records_committed = progress
        .records_committed
        .saturating_add(result.inserted_records);
    summary.records_read = summary.records_read.saturating_add(result.input_records);
    summary.records_committed = summary
        .records_committed
        .saturating_add(result.inserted_records);
    summary.duplicate_records = summary
        .duplicate_records
        .saturating_add(result.duplicate_records);
}

fn observe_reused_artifact(
    progress: &mut BackfillProgress,
    summary: &mut BackfillJobSummary,
    artifact: &super::job::BackfillArtifact,
) {
    let records = artifact
        .record_count
        .and_then(|count| u64::try_from(count).ok())
        .unwrap_or(0);
    progress.records_read = progress.records_read.saturating_add(records);
    progress.records_committed = progress.records_committed.saturating_add(records);
    summary.records_read = summary.records_read.saturating_add(records);
    summary.records_committed = summary.records_committed.saturating_add(records);
    summary.artifacts_completed = summary.artifacts_completed.saturating_add(1);
}

fn increment_missing(summary: &mut BackfillJobSummary, reason: &str) {
    let count = summary
        .missing_by_reason
        .entry(reason.to_string())
        .or_insert(0);
    *count = count.saturating_add(1);
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug)]
struct ClobOfficialResolution {
    winning_token_id: String,
    winning_outcome: BtcOutcome,
    observed_at: DateTime<Utc>,
}

fn slug_for_window(window_start: DateTime<Utc>) -> String {
    format!("btc-updown-5m-{}", window_start.timestamp())
}

fn parse_gamma_btc_interval_event(
    value: &Value,
    expected_window_start: DateTime<Utc>,
) -> Result<BtcIntervalMarket> {
    let event = value
        .as_object()
        .context("Gamma event response must be an object")?;
    let event_slug = required_string(event, &["slug"])?;
    let slug_epoch = event_slug
        .strip_prefix("btc-updown-5m-")
        .context("event slug is not a BTC Up/Down 5m slug")?
        .parse::<i64>()
        .context("BTC Up/Down 5m slug has an invalid epoch suffix")?;
    if slug_epoch.rem_euclid(300) != 0 {
        bail!("BTC Up/Down 5m slug is not five-minute aligned");
    }
    let slug_window_start = DateTime::from_timestamp(slug_epoch, 0)
        .context("BTC Up/Down 5m slug epoch is out of range")?;
    if slug_window_start != expected_window_start {
        bail!("Gamma event slug does not match the requested window");
    }
    let series_slug = string_field(event, &["seriesSlug", "series_slug"])
        .or_else(|| {
            event
                .get("series")?
                .as_array()?
                .iter()
                .find_map(|series| string_field(series.as_object()?, &["slug"]))
        })
        .context("Gamma event is missing a series slug")?;
    if series_slug != "btc-up-or-down-5m" {
        bail!("Gamma event belongs to unexpected series {series_slug}");
    }
    let window_start = datetime_field(event, &["eventStartTime", "startTime"])
        .context("Gamma event is missing eventStartTime")?;
    if window_start != slug_window_start {
        bail!("Gamma eventStartTime does not match its slug epoch");
    }
    let markets = event
        .get("markets")
        .and_then(Value::as_array)
        .context("Gamma event is missing its markets array")?;
    if markets.len() != 1 {
        bail!("BTC Up/Down 5m event must contain exactly one market");
    }
    let market = markets[0]
        .as_object()
        .context("Gamma event market must be an object")?;
    let window_end = datetime_field(market, &["endDate", "endDateIso"])
        .or_else(|| datetime_field(event, &["endDate"]))
        .context("Gamma event is missing the market end date")?;
    if window_end != window_start + ChronoDuration::minutes(5) {
        bail!("BTC Up/Down market does not have an exact five-minute window");
    }
    let resolution_source = string_field(market, &["resolutionSource", "resolution_source"])
        .or_else(|| string_field(event, &["resolutionSource", "resolution_source"]))
        .context("Gamma event is missing a resolution source")?;
    let normalized_source = resolution_source.trim().to_ascii_lowercase();
    if !(normalized_source.contains("chainlink") || normalized_source.contains("chain.link"))
        || !(normalized_source.contains("btc-usd")
            || normalized_source.contains("btc/usd")
            || (normalized_source.contains("btc") && normalized_source.contains("usd")))
    {
        bail!("BTC Up/Down market resolution source is not Chainlink BTC/USD");
    }
    let outcomes =
        string_array_field(market, &["outcomes"]).context("Gamma market is missing outcomes")?;
    let token_ids = string_array_field(
        market,
        &["clobTokenIds", "clob_token_ids", "tokenIds", "token_ids"],
    )
    .context("Gamma market is missing CLOB token IDs")?;
    if outcomes.len() != 2 || token_ids.len() != 2 {
        bail!("BTC Up/Down market must contain exactly two outcomes and token IDs");
    }
    let mut up_token_id = None;
    let mut down_token_id = None;
    for (outcome, token_id) in outcomes.iter().zip(token_ids) {
        match parse_outcome(outcome)? {
            BtcOutcome::Up => {
                if up_token_id.replace(token_id).is_some() {
                    bail!("BTC Up/Down market contains duplicate Up outcomes");
                }
            }
            BtcOutcome::Down => {
                if down_token_id.replace(token_id).is_some() {
                    bail!("BTC Up/Down market contains duplicate Down outcomes");
                }
            }
        }
    }
    let up_token_id = up_token_id.context("BTC Up/Down market is missing its Up token")?;
    let down_token_id = down_token_id.context("BTC Up/Down market is missing its Down token")?;
    if up_token_id == down_token_id {
        bail!("BTC Up/Down market token IDs must be distinct");
    }
    let tick_size = decimal_field(
        market,
        &[
            "orderPriceMinTickSize",
            "minimumTickSize",
            "tickSize",
            "tick_size",
        ],
    )
    .context("Gamma market is missing its minimum tick size")?;
    if tick_size <= Decimal::ZERO {
        bail!("Gamma market minimum tick size must be positive");
    }
    Ok(BtcIntervalMarket {
        event_id: required_string(event, &["id"])?,
        event_slug,
        series_slug,
        market_id: required_string(market, &["id"])?,
        condition_id: required_string(market, &["conditionId", "condition_id"])?,
        window_start,
        window_end,
        up_token_id,
        down_token_id,
        tick_size,
        minimum_order_size: decimal_field(market, &["orderMinSize", "minimumOrderSize"]),
        resolution_source,
        active: bool_field(market, &["active"])
            .or_else(|| bool_field(event, &["active"]))
            .unwrap_or(false),
        closed: bool_field(market, &["closed"])
            .or_else(|| bool_field(event, &["closed"]))
            .unwrap_or(false),
        accepting_orders: bool_field(market, &["acceptingOrders", "accepting_orders"])
            .unwrap_or(false),
        fees_enabled: bool_field(market, &["feesEnabled", "fees_enabled"])
            .or_else(|| bool_field(event, &["feesEnabled", "fees_enabled"]))
            .unwrap_or(false),
        fee_schedule: market
            .get("feeSchedule")
            .or_else(|| event.get("feeSchedule"))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({})),
        raw_payload: value.clone(),
    })
}

fn parse_clob_rest_official_resolution(
    value: &Value,
    market: &BtcIntervalMarket,
    observed_at: DateTime<Utc>,
) -> Result<Option<ClobOfficialResolution>> {
    let object = value
        .as_object()
        .context("CLOB market response must be an object")?;
    if required_string(object, &["condition_id", "conditionId"])? != market.condition_id {
        bail!("CLOB condition does not match the stored market");
    }
    let closed = bool_field(object, &["closed"]).context("CLOB market is missing closed status")?;
    let tokens = object
        .get("tokens")
        .and_then(Value::as_array)
        .context("CLOB market is missing its tokens array")?;
    if tokens.len() != 2 {
        bail!("CLOB BTC interval market must contain exactly two tokens");
    }
    let mut parsed = Vec::with_capacity(2);
    for token in tokens {
        let token = token.as_object().context("CLOB token must be an object")?;
        let token_id = required_string(token, &["token_id", "tokenId"])?;
        let outcome = parse_outcome(&required_string(token, &["outcome"])?)?;
        if token_id != market.token_id(outcome) {
            bail!("CLOB token does not match its stored outcome token");
        }
        parsed.push((
            token_id,
            outcome,
            bool_field(token, &["winner"]),
            decimal_field(token, &["price"]),
        ));
    }
    if parsed[0].1 == parsed[1].1 {
        bail!("CLOB market contains duplicate outcomes");
    }
    if !closed {
        return Ok(None);
    }
    let winners = parsed
        .iter()
        .filter(|(_, _, winner, _)| *winner == Some(true))
        .collect::<Vec<_>>();
    if winners.len() != 1 {
        bail!("closed CLOB market must expose exactly one winner");
    }
    for (token_id, _, winner, price) in &parsed {
        let winner = winner.context("closed CLOB token is missing winner status")?;
        let price = price.context("closed CLOB token is missing terminal price")?;
        let expected = if winner { Decimal::ONE } else { Decimal::ZERO };
        if price != expected {
            bail!("closed CLOB token {token_id} has a non-terminal price");
        }
    }
    Ok(Some(ClobOfficialResolution {
        winning_token_id: winners[0].0.clone(),
        winning_outcome: winners[0].1,
        observed_at,
    }))
}

fn parse_outcome(value: &str) -> Result<BtcOutcome> {
    match value.trim().to_ascii_lowercase().as_str() {
        "up" => Ok(BtcOutcome::Up),
        "down" => Ok(BtcOutcome::Down),
        other => bail!("unexpected BTC interval outcome {other}"),
    }
}

fn required_string(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Result<String> {
    string_field(object, keys).with_context(|| format!("missing required field {}", keys[0]))
}

fn string_field(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| match object.get(*key)? {
        Value::String(value) if !value.trim().is_empty() => Some(value.trim().to_string()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    })
}

fn string_array_field(
    object: &serde_json::Map<String, Value>,
    keys: &[&str],
) -> Option<Vec<String>> {
    let value = keys.iter().find_map(|key| object.get(*key))?;
    let values = match value {
        Value::Array(values) => values.clone(),
        Value::String(value) => serde_json::from_str(value).ok()?,
        _ => return None,
    };
    values
        .into_iter()
        .map(|value| match value {
            Value::String(value) if !value.trim().is_empty() => Some(value.trim().to_string()),
            Value::Number(value) => Some(value.to_string()),
            _ => None,
        })
        .collect()
}

fn decimal_field(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<Decimal> {
    match keys.iter().find_map(|key| object.get(*key))? {
        Value::String(value) => value.parse().ok(),
        Value::Number(value) => value.to_string().parse().ok(),
        _ => None,
    }
}

fn bool_field(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|key| object.get(*key)?.as_bool())
}

fn datetime_field(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(&string_field(object, keys)?)
        .ok()
        .map(|value| value.with_timezone(&Utc))
}

fn extract_reference_values(value: &Value) -> (Option<Decimal>, Option<Decimal>) {
    let opening = decimal_at_paths(
        value,
        &[
            &["priceToBeat"],
            &["eventMetadata", "priceToBeat"],
            &["markets", "0", "priceToBeat"],
            &["markets", "0", "eventMetadata", "priceToBeat"],
        ],
    );
    let final_price = decimal_at_paths(
        value,
        &[
            &["finalPrice"],
            &["settlementValue"],
            &["eventMetadata", "finalPrice"],
            &["eventMetadata", "settlementValue"],
            &["markets", "0", "finalPrice"],
            &["markets", "0", "settlementValue"],
            &["markets", "0", "eventMetadata", "finalPrice"],
        ],
    );
    (opening, final_price)
}

fn decimal_at_paths(value: &Value, paths: &[&[&str]]) -> Option<Decimal> {
    paths.iter().find_map(|path| {
        let mut current = value;
        for segment in *path {
            current = if let Ok(index) = segment.parse::<usize>() {
                current.as_array()?.get(index)?
            } else {
                current.get(*segment)?
            };
        }
        let decimal = match current {
            Value::String(value) => value.parse::<Decimal>().ok(),
            Value::Number(value) => value.to_string().parse::<Decimal>().ok(),
            _ => None,
        }?;
        (decimal > Decimal::ZERO).then_some(decimal)
    })
}

fn validate_archive_coverage(
    kind: BinanceArchiveKind,
    date: NaiveDate,
    summary: &ArchiveParseSummary,
) -> Result<()> {
    let start = date
        .and_hms_opt(0, 0, 0)
        .context("archive date was out of range")?
        .and_utc();
    let end = start + ChronoDuration::days(1);
    let minimum = summary
        .minimum_timestamp
        .context("archive contained no records")?;
    let maximum = summary
        .maximum_timestamp
        .context("archive contained no records")?;
    if minimum < start || maximum >= end {
        bail!("archive timestamps escaped their UTC source date");
    }
    if kind == BinanceArchiveKind::OneSecondKlines
        && (summary.records != 86_400
            || minimum != start
            || maximum != end - ChronoDuration::seconds(1))
    {
        bail!("one-second kline archive did not contain one complete UTC day");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::TimeZone;
    use rust_decimal_macros::dec;

    use super::*;

    #[test]
    fn extracts_only_named_positive_reference_values() {
        let value = serde_json::json!({
            "eventMetadata": {"priceToBeat": "60000.25"},
            "markets": [{"eventMetadata": {"finalPrice": 60001.5}}],
            "noise": 123,
        });
        assert_eq!(
            extract_reference_values(&value),
            (Some(dec!(60000.25)), Some(dec!(60001.5)))
        );
        assert_eq!(
            extract_reference_values(&serde_json::json!({"noise": 123})),
            (None, None)
        );
    }

    #[test]
    fn market_parser_maps_tokens_by_outcome_and_rejects_wrong_series() {
        let start = Utc.with_ymd_and_hms(2026, 7, 13, 0, 30, 0).unwrap();
        let mut event = serde_json::json!({
            "id": "event-1",
            "slug": "btc-updown-5m-1783902600",
            "seriesSlug": "btc-up-or-down-5m",
            "eventStartTime": "2026-07-13T00:30:00Z",
            "resolutionSource": "https://data.chain.link/streams/btc-usd",
            "markets": [{
                "id": "market-1",
                "conditionId": "0xcondition",
                "endDate": "2026-07-13T00:35:00Z",
                "outcomes": "[\"Down\",\"Up\"]",
                "clobTokenIds": "[\"down-token\",\"up-token\"]",
                "orderPriceMinTickSize": "0.01",
                "orderMinSize": 5
            }]
        });
        let parsed = parse_gamma_btc_interval_event(&event, start).unwrap();
        assert_eq!(parsed.up_token_id, "up-token");
        assert_eq!(parsed.down_token_id, "down-token");
        event["seriesSlug"] = serde_json::json!("not-the-contract-series");
        assert!(parse_gamma_btc_interval_event(&event, start).is_err());
    }

    #[test]
    fn resolution_parser_requires_the_stored_binary_winner() {
        let start = Utc.with_ymd_and_hms(2026, 7, 13, 0, 30, 0).unwrap();
        let event = serde_json::json!({
            "id": "event-1",
            "slug": "btc-updown-5m-1783902600",
            "seriesSlug": "btc-up-or-down-5m",
            "eventStartTime": "2026-07-13T00:30:00Z",
            "resolutionSource": "Chainlink BTC/USD",
            "markets": [{
                "id": "market-1",
                "conditionId": "0xcondition",
                "endDate": "2026-07-13T00:35:00Z",
                "outcomes": ["Up", "Down"],
                "clobTokenIds": ["up-token", "down-token"],
                "orderPriceMinTickSize": "0.01",
                "orderMinSize": 5
            }]
        });
        let market = parse_gamma_btc_interval_event(&event, start).unwrap();
        let resolution = serde_json::json!({
            "condition_id": "0xcondition",
            "closed": true,
            "tokens": [
                {"token_id": "up-token", "outcome": "Up", "winner": true, "price": "1"},
                {"token_id": "down-token", "outcome": "Down", "winner": false, "price": "0"}
            ]
        });
        let parsed = parse_clob_rest_official_resolution(
            &resolution,
            &market,
            start + ChronoDuration::minutes(6),
        )
        .unwrap()
        .unwrap();
        assert_eq!(parsed.winning_outcome, BtcOutcome::Up);
        assert_eq!(parsed.winning_token_id, "up-token");
    }

    #[test]
    fn one_second_readiness_requires_a_complete_day() {
        let date = NaiveDate::from_ymd_opt(2026, 7, 1).unwrap();
        let start = Utc.with_ymd_and_hms(2026, 7, 1, 0, 0, 0).unwrap();
        let complete = ArchiveParseSummary {
            records: 86_400,
            minimum_timestamp: Some(start),
            maximum_timestamp: Some(start + ChronoDuration::seconds(86_399)),
            ..ArchiveParseSummary::default()
        };
        assert!(
            validate_archive_coverage(BinanceArchiveKind::OneSecondKlines, date, &complete).is_ok()
        );
        let incomplete = ArchiveParseSummary {
            records: 86_399,
            ..complete
        };
        assert!(
            validate_archive_coverage(BinanceArchiveKind::OneSecondKlines, date, &incomplete)
                .is_err()
        );
    }

    #[test]
    fn executor_config_enforces_database_batch_bound() {
        let mut config = IngestionExecutorConfig {
            gamma_base_url: "https://gamma.example".to_string(),
            clob_base_url: "https://clob.example".to_string(),
            binance_archive_base_url: "https://archive.example".to_string(),
            cache_directory: PathBuf::from("/tmp/cache"),
            batch_rows: 4_000,
        };
        assert!(config.validate().is_ok());
        config.batch_rows = 4_001;
        assert!(config.validate().is_err());
    }

    #[test]
    fn summary_missing_counts_are_stable() {
        let mut summary = BackfillJobSummary {
            missing_by_reason: BTreeMap::new(),
            ..BackfillJobSummary::default()
        };
        increment_missing(&mut summary, "missing");
        increment_missing(&mut summary, "missing");
        assert_eq!(summary.missing_by_reason["missing"], 2);
    }
}
