use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Timelike, Utc};
use futures_util::{stream, StreamExt};
use md5::Md5;
use parquet::{
    data_type::Decimal as ParquetDecimal,
    file::{
        metadata::RowGroupMetaData,
        reader::{FileReader, SerializedFileReader},
    },
    record::{Field, Row},
    schema::types::Type as SchemaType,
};
use rust_decimal::Decimal;
use sha2::{Digest, Sha256};
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncWriteExt},
    sync::mpsc,
    task::JoinHandle,
    time::timeout,
};
use tracing::{info, warn};

use super::{
    binance_archive::{
        ArchiveCancellation, ArchiveDownloadLimits, ArchiveParseSummary, DownloadedArchive,
    },
    job::{BtcOrderbookArchiveEvent, BtcOrderbookMarketScope},
};

pub const PMXT_ARCHIVE_PROVIDER: &str = "pmxt_v2";
pub const DEFAULT_PMXT_ARCHIVE_URL: &str = "https://r2v2.pmxt.dev";
pub const PMXT_COVERAGE_START_EPOCH: i64 = 1_776_106_800; // 2026-04-13T19:00:00Z
const PMXT_MULTIPART_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const PMXT_DOWNLOAD_ATTEMPTS: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
struct PmxtEtag {
    digest: String,
    parts: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PmxtObjectIdentity {
    content_length: Option<u64>,
    etag: Option<PmxtEtag>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PmxtArchiveDigest {
    sha256: String,
    bytes: u64,
    single_md5: String,
    multipart_md5: String,
    multipart_parts: usize,
}

struct PmxtDigestAccumulator {
    sha256: Sha256,
    whole_md5: Md5,
    part_md5: Md5,
    part_bytes: usize,
    part_digests: Vec<[u8; 16]>,
    bytes: u64,
}

type PrefetchResult = Result<Option<DownloadedArchive>>;

pub struct PmxtArchivePrefetch {
    receiver: mpsc::Receiver<(String, PrefetchResult)>,
    ready: HashMap<String, PrefetchResult>,
    task: JoinHandle<()>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PmxtArchiveSpec {
    pub hour: DateTime<Utc>,
    pub file_name: String,
    pub source_uri: String,
    pub logical_key: String,
}

impl PmxtArchiveSpec {
    pub fn new(base_url: &str, hour: DateTime<Utc>) -> Result<Self> {
        if hour.timestamp_subsec_nanos() != 0 || hour.minute() != 0 || hour.second() != 0 {
            bail!("PMXT archive timestamp must be UTC-hour aligned");
        }
        let stamp = hour.format("%Y-%m-%dT%H");
        let file_name = format!("polymarket_orderbook_{stamp}.parquet");
        Ok(Self {
            hour,
            source_uri: format!("{}/{}", base_url.trim_end_matches('/'), file_name),
            logical_key: format!("pmxt:v2:polymarket_orderbook:{stamp}"),
            file_name,
        })
    }
}

impl PmxtDigestAccumulator {
    fn new() -> Self {
        Self {
            sha256: Sha256::new(),
            whole_md5: Md5::new(),
            part_md5: Md5::new(),
            part_bytes: 0,
            part_digests: Vec::new(),
            bytes: 0,
        }
    }

    fn update(&mut self, bytes: &[u8]) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_add(u64::try_from(bytes.len()).context("PMXT archive size overflow")?)
            .context("PMXT archive size overflow")?;
        self.sha256.update(bytes);
        self.whole_md5.update(bytes);
        let mut remaining = bytes;
        while !remaining.is_empty() {
            let available = PMXT_MULTIPART_CHUNK_BYTES.saturating_sub(self.part_bytes);
            let take = available.min(remaining.len());
            self.part_md5.update(&remaining[..take]);
            self.part_bytes += take;
            remaining = &remaining[take..];
            if self.part_bytes == PMXT_MULTIPART_CHUNK_BYTES {
                self.finish_part();
            }
        }
        Ok(())
    }

    fn finish_part(&mut self) {
        let digest: [u8; 16] = self.part_md5.finalize_reset().into();
        self.part_digests.push(digest);
        self.part_bytes = 0;
    }

    fn finish(mut self) -> PmxtArchiveDigest {
        if self.part_bytes > 0 || self.part_digests.is_empty() {
            self.finish_part();
        }
        let mut multipart = Md5::new();
        for digest in &self.part_digests {
            multipart.update(digest);
        }
        PmxtArchiveDigest {
            sha256: format!("{:x}", self.sha256.finalize()),
            bytes: self.bytes,
            single_md5: format!("{:x}", self.whole_md5.finalize()),
            multipart_md5: format!("{:x}", multipart.finalize()),
            multipart_parts: self.part_digests.len(),
        }
    }
}

impl PmxtObjectIdentity {
    fn from_response(response: &reqwest::Response) -> Result<Self> {
        let etag = response
            .headers()
            .get(reqwest::header::ETAG)
            .map(|value| value.to_str().context("PMXT ETag was not ASCII"))
            .transpose()?
            .map(parse_etag)
            .transpose()?;
        Ok(Self {
            content_length: response.content_length(),
            etag,
        })
    }

    fn validate(&self, digest: &PmxtArchiveDigest) -> Result<()> {
        if self
            .content_length
            .is_some_and(|expected| expected != digest.bytes)
        {
            bail!(
                "PMXT archive length mismatch: expected {}, received {}",
                self.content_length.unwrap_or_default(),
                digest.bytes
            );
        }
        if let Some(expected) = &self.etag {
            let matches = match expected.parts {
                Some(parts) => {
                    parts == digest.multipart_parts && expected.digest == digest.multipart_md5
                }
                None => expected.digest == digest.single_md5,
            };
            if !matches {
                bail!("PMXT archive ETag mismatch");
            }
        }
        Ok(())
    }
}

fn parse_etag(value: &str) -> Result<PmxtEtag> {
    let value = value.trim().trim_start_matches("W/").trim_matches('"');
    let (digest, parts) = value
        .split_once('-')
        .map_or((value, None), |(digest, parts)| {
            (digest, parts.parse::<usize>().ok())
        });
    if digest.len() != 32
        || !digest.bytes().all(|value| value.is_ascii_hexdigit())
        || value.contains('-') && parts.is_none()
    {
        bail!("PMXT ETag had an unsupported format");
    }
    Ok(PmxtEtag {
        digest: digest.to_ascii_lowercase(),
        parts,
    })
}

impl PmxtArchivePrefetch {
    pub async fn take(&mut self, spec: &PmxtArchiveSpec) -> Result<Option<DownloadedArchive>> {
        if let Some(result) = self.ready.remove(&spec.logical_key) {
            return result;
        }
        while let Some((logical_key, result)) = self.receiver.recv().await {
            if logical_key == spec.logical_key {
                return result;
            }
            self.ready.insert(logical_key, result);
        }
        bail!(
            "PMXT prefetch ended before archive {} became available",
            spec.logical_key
        )
    }
}

impl Drop for PmxtArchivePrefetch {
    fn drop(&mut self) {
        self.task.abort();
    }
}

pub fn spawn_archive_prefetch(
    client: reqwest::Client,
    specs: Vec<PmxtArchiveSpec>,
    cache_directory: PathBuf,
    limits: ArchiveDownloadLimits,
    cancellation: ArchiveCancellation,
    concurrency: usize,
    buffered_archives: usize,
) -> PmxtArchivePrefetch {
    let (sender, receiver) = mpsc::channel(buffered_archives.max(concurrency).max(1));
    let task = tokio::spawn(async move {
        let downloads = stream::iter(specs).map(|spec| {
            let client = client.clone();
            let cache_directory = cache_directory.clone();
            let limits = limits.clone();
            let cancellation = cancellation.clone();
            async move {
                let logical_key = spec.logical_key.clone();
                let started = Instant::now();
                let result =
                    download_archive(&client, &spec, &cache_directory, &limits, &cancellation)
                        .await;
                match &result {
                    Ok(Some(archive)) => info!(
                        logical_key,
                        elapsed_milliseconds = started.elapsed().as_millis(),
                        compressed_bytes = archive.compressed_bytes,
                        reused_cache = archive.reused_cache,
                        "PMXT archive prefetch completed"
                    ),
                    Ok(None) => warn!(
                        logical_key,
                        elapsed_milliseconds = started.elapsed().as_millis(),
                        "PMXT archive prefetch found no source object"
                    ),
                    Err(error) => warn!(
                        logical_key,
                        elapsed_milliseconds = started.elapsed().as_millis(),
                        error = %error,
                        "PMXT archive prefetch failed"
                    ),
                }
                (logical_key, result)
            }
        });
        let mut downloads = downloads.buffered(concurrency.max(1));
        while let Some(result) = downloads.next().await {
            if sender.send(result).await.is_err() {
                break;
            }
        }
    });
    PmxtArchivePrefetch {
        receiver,
        ready: HashMap::new(),
        task,
    }
}

pub async fn download_archive(
    client: &reqwest::Client,
    spec: &PmxtArchiveSpec,
    cache_directory: &Path,
    limits: &ArchiveDownloadLimits,
    cancellation: &ArchiveCancellation,
) -> Result<Option<DownloadedArchive>> {
    fs::create_dir_all(cache_directory)
        .await
        .with_context(|| format!("failed to create {}", cache_directory.display()))?;
    let final_path = cache_directory.join(&spec.file_name);
    if final_path.exists() {
        let Some(identity) = fetch_object_identity(client, spec, cancellation).await? else {
            fs::remove_file(&final_path)
                .await
                .with_context(|| format!("failed to remove {}", final_path.display()))?;
            return Ok(None);
        };
        let digest = hash_file(&final_path, limits.maximum_compressed_bytes, cancellation).await?;
        if identity.validate(&digest).is_ok() {
            return Ok(Some(DownloadedArchive {
                path: final_path,
                sha256: digest.sha256,
                compressed_bytes: digest.bytes,
                reused_cache: true,
            }));
        }
        fs::remove_file(&final_path)
            .await
            .with_context(|| format!("failed to remove invalid {}", final_path.display()))?;
    }

    let partial_path = cache_directory.join(format!("{}.part", spec.file_name));
    let mut last_error = None;
    for _ in 0..PMXT_DOWNLOAD_ATTEMPTS {
        match download_archive_once(client, spec, &partial_path, limits, cancellation).await {
            Ok(None) => return Ok(None),
            Ok(Some(digest)) => {
                fs::rename(&partial_path, &final_path)
                    .await
                    .context("failed to publish PMXT archive cache")?;
                return Ok(Some(DownloadedArchive {
                    path: final_path,
                    sha256: digest.sha256,
                    compressed_bytes: digest.bytes,
                    reused_cache: false,
                }));
            }
            Err(error) => {
                last_error = Some(error);
            }
        }
        if partial_path.exists() {
            let _ = fs::remove_file(&partial_path).await;
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("PMXT archive download failed")))
}

async fn fetch_object_identity(
    client: &reqwest::Client,
    spec: &PmxtArchiveSpec,
    cancellation: &ArchiveCancellation,
) -> Result<Option<PmxtObjectIdentity>> {
    if cancellation.is_cancelled() {
        bail!("archive operation was cancelled");
    }
    let response = client
        .head(&spec.source_uri)
        .send()
        .await
        .with_context(|| format!("failed to inspect {}", spec.source_uri))?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    let response = response
        .error_for_status()
        .with_context(|| format!("PMXT rejected {}", spec.source_uri))?;
    PmxtObjectIdentity::from_response(&response).map(Some)
}

async fn download_archive_once(
    client: &reqwest::Client,
    spec: &PmxtArchiveSpec,
    partial_path: &Path,
    limits: &ArchiveDownloadLimits,
    cancellation: &ArchiveCancellation,
) -> Result<Option<PmxtArchiveDigest>> {
    if cancellation.is_cancelled() {
        bail!("archive operation was cancelled");
    }
    let mut response = client
        .get(&spec.source_uri)
        .send()
        .await
        .with_context(|| format!("failed to request {}", spec.source_uri))?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    response = response
        .error_for_status()
        .with_context(|| format!("PMXT rejected {}", spec.source_uri))?;
    let identity = PmxtObjectIdentity::from_response(&response)?;
    if identity
        .content_length
        .is_some_and(|size| size > limits.maximum_compressed_bytes)
    {
        bail!("PMXT archive exceeded its compressed size limit");
    }
    let mut output = fs::File::create(partial_path)
        .await
        .with_context(|| format!("failed to create {}", partial_path.display()))?;
    let mut digest = PmxtDigestAccumulator::new();
    loop {
        if cancellation.is_cancelled() {
            bail!("archive operation was cancelled");
        }
        let chunk = timeout(limits.chunk_idle_timeout, response.chunk())
            .await
            .context("PMXT archive response stalled")??;
        let Some(chunk) = chunk else { break };
        digest.update(&chunk)?;
        if digest.bytes > limits.maximum_compressed_bytes {
            bail!("PMXT archive exceeded its compressed size limit");
        }
        output
            .write_all(&chunk)
            .await
            .context("failed to write PMXT archive cache")?;
    }
    let digest = digest.finish();
    identity.validate(&digest)?;
    output
        .sync_all()
        .await
        .context("failed to sync PMXT archive cache")?;
    drop(output);
    Ok(Some(digest))
}

pub fn spawn_parser(
    path: PathBuf,
    markets: Vec<BtcOrderbookMarketScope>,
    batch_rows: usize,
    cancellation: ArchiveCancellation,
) -> (
    mpsc::Receiver<Result<Vec<BtcOrderbookArchiveEvent>>>,
    JoinHandle<Result<ArchiveParseSummary>>,
) {
    spawn_parser_inner(path, markets, batch_rows, cancellation, false)
}

pub fn spawn_execution_parser(
    path: PathBuf,
    markets: Vec<BtcOrderbookMarketScope>,
    batch_rows: usize,
    cancellation: ArchiveCancellation,
) -> (
    mpsc::Receiver<Result<Vec<BtcOrderbookArchiveEvent>>>,
    JoinHandle<Result<ArchiveParseSummary>>,
) {
    spawn_parser_inner(path, markets, batch_rows, cancellation, true)
}

fn spawn_parser_inner(
    path: PathBuf,
    markets: Vec<BtcOrderbookMarketScope>,
    batch_rows: usize,
    cancellation: ArchiveCancellation,
    execution_projection: bool,
) -> (
    mpsc::Receiver<Result<Vec<BtcOrderbookArchiveEvent>>>,
    JoinHandle<Result<ArchiveParseSummary>>,
) {
    let (sender, receiver) = mpsc::channel(1);
    let handle = tokio::task::spawn_blocking(move || {
        let conditions = markets
            .iter()
            .map(|market| market.condition_id.as_str())
            .collect::<HashSet<_>>();
        let condition_bytes = conditions
            .iter()
            .map(|condition| condition.as_bytes().to_vec())
            .collect::<Vec<_>>();
        let assets = markets
            .iter()
            .flat_map(|market| [&market.up_token_id, &market.down_token_id])
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let file = File::open(&path)
            .with_context(|| format!("failed to open PMXT archive {}", path.display()))?;
        let reader =
            SerializedFileReader::new(file).context("failed to initialize PMXT Parquet reader")?;
        let projection = execution_projection
            .then(|| leading_column_projection(&reader, 10))
            .transpose()?;
        let mut summary = ArchiveParseSummary::default();
        let mut batch = Vec::with_capacity(batch_rows);
        let mut source_row_base = 0i64;

        for row_group_index in 0..reader.num_row_groups() {
            let metadata = reader.metadata().row_group(row_group_index);
            let row_group_records = metadata.num_rows();
            if row_group_may_contain_condition(metadata, &condition_bytes) {
                let row_group = reader.get_row_group(row_group_index).with_context(|| {
                    format!("failed to initialize PMXT Parquet row group {row_group_index}")
                })?;
                let rows = row_group
                    .get_row_iter(projection.clone())
                    .with_context(|| {
                        format!("failed to stream PMXT Parquet row group {row_group_index}")
                    })?;
                for (row_group_ordinal, row) in rows.enumerate() {
                    if cancellation.is_cancelled() {
                        bail!("archive operation was cancelled");
                    }
                    let row_group_ordinal = i64::try_from(row_group_ordinal)
                        .context("PMXT row-group ordinal overflow")?;
                    let source_row_number = source_row_base
                        .checked_add(row_group_ordinal)
                        .context("PMXT source row number overflow")?;
                    let row = row.map_err(|error| {
                        anyhow::anyhow!(
                            "failed to decode PMXT Parquet row group {row_group_index}, \
                             source row {source_row_number}: {error}"
                        )
                    })?;
                    let record = if execution_projection {
                        parse_execution_row(row, source_row_number, &conditions, &assets)?
                    } else {
                        parse_row(row, source_row_number, &conditions, &assets)?
                    };
                    let Some(record) = record else {
                        continue;
                    };
                    summary.records = summary.records.saturating_add(1);
                    summary.minimum_timestamp = Some(
                        summary
                            .minimum_timestamp
                            .map_or(record.source_timestamp, |value| {
                                value.min(record.source_timestamp)
                            }),
                    );
                    summary.maximum_timestamp = Some(
                        summary
                            .maximum_timestamp
                            .map_or(record.source_timestamp, |value| {
                                value.max(record.source_timestamp)
                            }),
                    );
                    batch.push(record);
                    if batch.len() == batch_rows {
                        summary.batches = summary.batches.saturating_add(1);
                        summary.maximum_batch_records =
                            summary.maximum_batch_records.max(batch.len());
                        sender
                            .blocking_send(Ok(std::mem::take(&mut batch)))
                            .context("PMXT parser consumer stopped")?;
                        batch = Vec::with_capacity(batch_rows);
                    }
                }
            }
            source_row_base = source_row_base
                .checked_add(row_group_records)
                .context("PMXT source row base overflow")?;
        }
        if !batch.is_empty() {
            summary.batches = summary.batches.saturating_add(1);
            summary.maximum_batch_records = summary.maximum_batch_records.max(batch.len());
            sender
                .blocking_send(Ok(batch))
                .context("PMXT parser consumer stopped")?;
        }
        Ok(summary)
    });
    (receiver, handle)
}

fn leading_column_projection(
    reader: &SerializedFileReader<File>,
    column_count: usize,
) -> Result<SchemaType> {
    let root = reader.metadata().file_metadata().schema();
    let fields = root.get_fields();
    if fields.len() < column_count {
        bail!(
            "PMXT schema exposed {} columns; expected at least {column_count}",
            fields.len()
        );
    }
    SchemaType::group_type_builder(root.name())
        .with_fields(fields[..column_count].to_vec())
        .build()
        .context("failed to build PMXT execution projection")
}

fn row_group_may_contain_condition(metadata: &RowGroupMetaData, condition_ids: &[Vec<u8>]) -> bool {
    let Some(statistics) = metadata.column(2).statistics() else {
        return true;
    };
    if !statistics.min_is_exact() || !statistics.max_is_exact() {
        return true;
    }
    let (Some(minimum), Some(maximum)) = (statistics.min_bytes_opt(), statistics.max_bytes_opt())
    else {
        return true;
    };
    condition_range_overlaps(condition_ids, minimum, maximum)
}

fn condition_range_overlaps(condition_ids: &[Vec<u8>], minimum: &[u8], maximum: &[u8]) -> bool {
    condition_ids
        .iter()
        .any(|condition| condition.as_slice() >= minimum && condition.as_slice() <= maximum)
}

fn parse_execution_row(
    row: Row,
    source_row_number: i64,
    conditions: &HashSet<&str>,
    assets: &HashSet<&str>,
) -> Result<Option<BtcOrderbookArchiveEvent>> {
    let columns = row.into_columns();
    if columns.len() != 10 {
        bail!("PMXT execution row did not have the projected 10-column schema");
    }
    let provider_received_at = timestamp_field(&columns[0].1, "timestamp_received")?;
    let source_timestamp = timestamp_field(&columns[1].1, "timestamp")?;
    let condition_id = bytes_field(&columns[2].1, "market")?;
    let event_type = string_field(&columns[3].1, "event_type")?;
    let asset_id = string_field(&columns[4].1, "asset_id")?;
    if !conditions.contains(condition_id.as_str()) || !assets.contains(asset_id.as_str()) {
        return Ok(None);
    }
    if matches!(event_type.as_str(), "last_trade_price" | "tick_size_change") {
        return Ok(None);
    }
    if !matches!(event_type.as_str(), "book" | "price_change") {
        bail!("PMXT row had unsupported event type {event_type}");
    }
    let bids = json_field(&columns[5].1, "bids")?;
    let asks = json_field(&columns[6].1, "asks")?;
    if event_type == "book"
        && (!bids.as_ref().is_some_and(serde_json::Value::is_array)
            || !asks.as_ref().is_some_and(serde_json::Value::is_array))
    {
        bail!("PMXT book event did not contain bid and ask arrays");
    }
    Ok(Some(BtcOrderbookArchiveEvent {
        source_row_number,
        provider_received_at,
        source_timestamp,
        condition_id,
        asset_id,
        event_type,
        bids,
        asks,
        price: decimal_field(&columns[7].1)?,
        size: decimal_field(&columns[8].1)?,
        side: optional_string_field(&columns[9].1, "side")?.map(|side| side.to_ascii_lowercase()),
        best_bid: None,
        best_ask: None,
        fee_rate_bps: None,
        transaction_hash: None,
        old_tick_size: None,
        new_tick_size: None,
    }))
}

fn parse_row(
    row: Row,
    source_row_number: i64,
    conditions: &HashSet<&str>,
    assets: &HashSet<&str>,
) -> Result<Option<BtcOrderbookArchiveEvent>> {
    let columns = row.into_columns();
    if columns.len() != 16 {
        bail!("PMXT row did not have the documented 16-column schema");
    }
    let provider_received_at = timestamp_field(&columns[0].1, "timestamp_received")?;
    let source_timestamp = timestamp_field(&columns[1].1, "timestamp")?;
    let condition_id = bytes_field(&columns[2].1, "market")?;
    let event_type = string_field(&columns[3].1, "event_type")?;
    let asset_id = string_field(&columns[4].1, "asset_id")?;
    if !conditions.contains(condition_id.as_str()) || !assets.contains(asset_id.as_str()) {
        return Ok(None);
    }
    if !matches!(
        event_type.as_str(),
        "book" | "price_change" | "last_trade_price" | "tick_size_change"
    ) {
        bail!("PMXT row had unsupported event type {event_type}");
    }
    let bids = json_field(&columns[5].1, "bids")?;
    let asks = json_field(&columns[6].1, "asks")?;
    if event_type == "book"
        && (!bids.as_ref().is_some_and(serde_json::Value::is_array)
            || !asks.as_ref().is_some_and(serde_json::Value::is_array))
    {
        bail!("PMXT book event did not contain bid and ask arrays");
    }
    Ok(Some(BtcOrderbookArchiveEvent {
        source_row_number,
        provider_received_at,
        source_timestamp,
        condition_id,
        asset_id,
        event_type,
        bids,
        asks,
        price: decimal_field(&columns[7].1)?,
        size: decimal_field(&columns[8].1)?,
        side: optional_string_field(&columns[9].1, "side")?.map(|side| side.to_ascii_lowercase()),
        best_bid: decimal_field(&columns[10].1)?,
        best_ask: decimal_field(&columns[11].1)?,
        fee_rate_bps: optional_u16_field(&columns[12].1)?.map(i32::from),
        transaction_hash: optional_string_field(&columns[13].1, "transaction_hash")?,
        old_tick_size: decimal_field(&columns[14].1)?,
        new_tick_size: decimal_field(&columns[15].1)?,
    }))
}

fn timestamp_field(field: &Field, name: &str) -> Result<DateTime<Utc>> {
    let Field::TimestampMillis(value) = field else {
        bail!("PMXT {name} was not timestamp[ms]");
    };
    DateTime::from_timestamp_millis(*value).context("PMXT timestamp was out of range")
}

fn bytes_field(field: &Field, name: &str) -> Result<String> {
    let Field::Bytes(value) = field else {
        bail!("PMXT {name} was not fixed binary");
    };
    String::from_utf8(value.data().to_vec()).with_context(|| format!("PMXT {name} was not ASCII"))
}

fn string_field(field: &Field, name: &str) -> Result<String> {
    let Field::Str(value) = field else {
        bail!("PMXT {name} was not a string");
    };
    Ok(value.clone())
}

fn optional_string_field(field: &Field, name: &str) -> Result<Option<String>> {
    match field {
        Field::Null => Ok(None),
        Field::Str(value) => Ok(Some(value.clone())),
        _ => bail!("PMXT {name} was neither null nor a string"),
    }
}

fn json_field(field: &Field, name: &str) -> Result<Option<serde_json::Value>> {
    optional_string_field(field, name)?
        .map(|value| {
            serde_json::from_str(&value).with_context(|| format!("PMXT {name} was invalid JSON"))
        })
        .transpose()
}

fn decimal_field(field: &Field) -> Result<Option<Decimal>> {
    match field {
        Field::Null => Ok(None),
        Field::Decimal(value) => Ok(Some(parquet_decimal(value)?)),
        _ => bail!("PMXT decimal column had an unexpected type"),
    }
}

fn parquet_decimal(value: &ParquetDecimal) -> Result<Decimal> {
    let bytes = value.data();
    if bytes.is_empty() || bytes.len() > 16 {
        bail!("PMXT decimal exceeded supported i128 width");
    }
    let fill = if bytes[0] & 0x80 == 0 { 0 } else { 0xff };
    let mut extended = [fill; 16];
    extended[16 - bytes.len()..].copy_from_slice(bytes);
    let scale = u32::try_from(value.scale()).context("PMXT decimal scale was negative")?;
    Ok(Decimal::from_i128_with_scale(
        i128::from_be_bytes(extended),
        scale,
    ))
}

fn optional_u16_field(field: &Field) -> Result<Option<u16>> {
    match field {
        Field::Null => Ok(None),
        Field::UShort(value) => Ok(Some(*value)),
        _ => bail!("PMXT fee_rate_bps was neither null nor uint16"),
    }
}

async fn hash_file(
    path: &Path,
    maximum_bytes: u64,
    cancellation: &ArchiveCancellation,
) -> Result<PmxtArchiveDigest> {
    let mut input = fs::File::open(path).await?;
    let mut digest = PmxtDigestAccumulator::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        if cancellation.is_cancelled() {
            bail!("archive operation was cancelled");
        }
        let read = input.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read])?;
        if digest.bytes > maximum_bytes {
            bail!("PMXT cached archive exceeded its compressed size limit");
        }
    }
    Ok(digest.finish())
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use parquet::data_type::ByteArray;

    use super::*;

    #[test]
    fn archive_identity_is_hourly_and_stable() {
        let hour = Utc.with_ymd_and_hms(2026, 4, 17, 12, 0, 0).unwrap();
        let spec = PmxtArchiveSpec::new(DEFAULT_PMXT_ARCHIVE_URL, hour).unwrap();
        assert_eq!(
            spec.source_uri,
            "https://r2v2.pmxt.dev/polymarket_orderbook_2026-04-17T12.parquet"
        );
        assert_eq!(
            spec.logical_key,
            "pmxt:v2:polymarket_orderbook:2026-04-17T12"
        );
    }

    #[test]
    fn parquet_decimals_preserve_signed_scale() {
        let value = ParquetDecimal::from_bytes(ByteArray::from(vec![0x01, 0x86, 0xa0]), 9, 4);
        assert_eq!(parquet_decimal(&value).unwrap(), Decimal::new(100_000, 4));
        let negative =
            ParquetDecimal::from_bytes(ByteArray::from(vec![0xff, 0xff, 0xff, 0x9c]), 9, 2);
        assert_eq!(parquet_decimal(&negative).unwrap(), Decimal::new(-100, 2));
    }

    #[test]
    fn condition_ranges_prune_only_disjoint_row_groups() {
        let conditions = vec![b"0x20".to_vec(), b"0x80".to_vec()];
        assert!(condition_range_overlaps(&conditions, b"0x10", b"0x20"));
        assert!(condition_range_overlaps(&conditions, b"0x70", b"0x90"));
        assert!(!condition_range_overlaps(&conditions, b"0x21", b"0x79"));
        assert!(!condition_range_overlaps(&conditions, b"0x81", b"0xff"));
    }

    #[test]
    fn multipart_etag_validation_matches_r2_layout() {
        let mut accumulator = PmxtDigestAccumulator::new();
        accumulator
            .update(&vec![b'a'; PMXT_MULTIPART_CHUNK_BYTES])
            .unwrap();
        accumulator.update(b"b").unwrap();
        let digest = accumulator.finish();
        assert_eq!(digest.bytes, 8_388_609);
        assert_eq!(digest.multipart_parts, 2);
        assert_eq!(digest.multipart_md5, "15c088024dc2b3017cad9ee6965f364a");
        assert_eq!(digest.single_md5, "6012c8a1ea54f0626ea128968f6583dd");
        let identity = PmxtObjectIdentity {
            content_length: Some(8_388_609),
            etag: Some(parse_etag("\"15c088024dc2b3017cad9ee6965f364a-2\"").unwrap()),
        };
        assert!(identity.validate(&digest).is_ok());
    }
}
