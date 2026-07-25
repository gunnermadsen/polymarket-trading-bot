use std::{
    fs::File,
    io::BufReader,
    path::{Component, Path, PathBuf},
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, NaiveDate, TimeZone, Utc};
use rust_decimal::Decimal;
use sha2::{Digest, Sha256};
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncWriteExt},
    sync::mpsc,
    task::JoinHandle,
    time::timeout,
};
use zip::ZipArchive;

use super::job::{BinanceAggregateTradeRecord, BinanceOneSecondKlineRecord};

pub const BINANCE_ARCHIVE_PROVIDER: &str = "binance_public_data";
pub const BINANCE_SYMBOL: &str = "BTCUSDT";
const MAX_CHECKSUM_BODY_BYTES: usize = 1_024;
const DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES: u64 = 128 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinanceArchiveKind {
    AggregateTrades,
    OneSecondKlines,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinanceArchiveSpec {
    pub kind: BinanceArchiveKind,
    pub date: NaiveDate,
    pub file_name: String,
    pub entry_name: String,
    pub source_uri: String,
    pub checksum_uri: String,
    pub logical_key: String,
}

impl BinanceArchiveSpec {
    pub fn new(base_url: &str, kind: BinanceArchiveKind, date: NaiveDate) -> Self {
        let base_url = base_url.trim_end_matches('/');
        let date_text = date.format("%Y-%m-%d");
        let (directory, stem) = match kind {
            BinanceArchiveKind::AggregateTrades => (
                format!("data/spot/daily/aggTrades/{BINANCE_SYMBOL}"),
                format!("{BINANCE_SYMBOL}-aggTrades-{date_text}"),
            ),
            BinanceArchiveKind::OneSecondKlines => (
                format!("data/spot/daily/klines/{BINANCE_SYMBOL}/1s"),
                format!("{BINANCE_SYMBOL}-1s-{date_text}"),
            ),
        };
        let file_name = format!("{stem}.zip");
        let source_uri = format!("{base_url}/{directory}/{file_name}");
        Self {
            kind,
            date,
            entry_name: format!("{stem}.csv"),
            checksum_uri: format!("{source_uri}.CHECKSUM"),
            logical_key: match kind {
                BinanceArchiveKind::AggregateTrades => {
                    format!("binance:{BINANCE_SYMBOL}:agg_trades:{date_text}")
                }
                BinanceArchiveKind::OneSecondKlines => {
                    format!("binance:{BINANCE_SYMBOL}:klines_1s:{date_text}")
                }
            },
            file_name,
            source_uri,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArchiveDownloadLimits {
    pub maximum_compressed_bytes: u64,
    pub chunk_idle_timeout: Duration,
}

impl Default for ArchiveDownloadLimits {
    fn default() -> Self {
        Self {
            maximum_compressed_bytes: 32 * 1024 * 1024 * 1024,
            chunk_idle_timeout: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ArchiveCancellation {
    cancelled: Arc<AtomicBool>,
}

impl ArchiveCancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            bail!("archive operation was cancelled");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DownloadedArchive {
    pub path: PathBuf,
    pub sha256: String,
    pub compressed_bytes: u64,
    pub reused_cache: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArchiveParseSummary {
    pub records: u64,
    pub batches: u64,
    pub maximum_batch_records: usize,
    pub minimum_timestamp: Option<DateTime<Utc>>,
    pub maximum_timestamp: Option<DateTime<Utc>>,
}

impl ArchiveParseSummary {
    fn observe(&mut self, timestamp: DateTime<Utc>) {
        self.records = self.records.saturating_add(1);
        self.minimum_timestamp = Some(
            self.minimum_timestamp
                .map_or(timestamp, |current| current.min(timestamp)),
        );
        self.maximum_timestamp = Some(
            self.maximum_timestamp
                .map_or(timestamp, |current| current.max(timestamp)),
        );
    }

    fn observe_batch(&mut self, records: usize) {
        self.batches = self.batches.saturating_add(1);
        self.maximum_batch_records = self.maximum_batch_records.max(records);
    }
}

pub async fn fetch_expected_checksum(
    client: &reqwest::Client,
    spec: &BinanceArchiveSpec,
) -> Result<String> {
    fetch_expected_checksum_with_cancellation(client, spec, &ArchiveCancellation::default()).await
}

pub async fn fetch_expected_checksum_with_cancellation(
    client: &reqwest::Client,
    spec: &BinanceArchiveSpec,
    cancellation: &ArchiveCancellation,
) -> Result<String> {
    cancellation.check()?;
    let mut response = client
        .get(&spec.checksum_uri)
        .send()
        .await
        .with_context(|| format!("failed to request {}", spec.checksum_uri))?
        .error_for_status()
        .with_context(|| format!("Binance rejected {}", spec.checksum_uri))?;
    if response
        .content_length()
        .is_some_and(|length| length > u64::try_from(MAX_CHECKSUM_BODY_BYTES).unwrap_or(u64::MAX))
    {
        bail!("Binance checksum response exceeded its size limit");
    }
    let mut bytes = Vec::with_capacity(128);
    loop {
        cancellation.check()?;
        let chunk = timeout(Duration::from_secs(30), response.chunk())
            .await
            .context("Binance checksum response stalled")??;
        let Some(chunk) = chunk else { break };
        if bytes.len().saturating_add(chunk.len()) > MAX_CHECKSUM_BODY_BYTES {
            bail!("Binance checksum response exceeded its size limit");
        }
        bytes.extend_from_slice(&chunk);
    }
    let body = std::str::from_utf8(&bytes).context("Binance checksum response was not UTF-8")?;
    parse_checksum_document(body, &spec.file_name)
}

pub fn parse_checksum_document(body: &str, expected_file_name: &str) -> Result<String> {
    let mut fields = body.split_whitespace();
    let checksum = fields.next().context("checksum document was empty")?;
    let file_name = fields
        .next()
        .context("checksum document did not include a filename")?
        .trim_start_matches('*');
    if fields.next().is_some() {
        bail!("checksum document contained unexpected fields");
    }
    if file_name != expected_file_name {
        bail!("checksum filename {file_name} did not match {expected_file_name}");
    }
    let checksum = checksum.to_ascii_lowercase();
    if checksum.len() != 64 || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("checksum document did not contain a valid SHA-256 digest");
    }
    Ok(checksum)
}

pub async fn download_archive(
    client: &reqwest::Client,
    spec: &BinanceArchiveSpec,
    expected_sha256: &str,
    cache_directory: &Path,
    limits: &ArchiveDownloadLimits,
) -> Result<DownloadedArchive> {
    download_archive_with_cancellation(
        client,
        spec,
        expected_sha256,
        cache_directory,
        limits,
        &ArchiveCancellation::default(),
    )
    .await
}

pub async fn download_archive_with_cancellation(
    client: &reqwest::Client,
    spec: &BinanceArchiveSpec,
    expected_sha256: &str,
    cache_directory: &Path,
    limits: &ArchiveDownloadLimits,
    cancellation: &ArchiveCancellation,
) -> Result<DownloadedArchive> {
    validate_sha256(expected_sha256)?;
    validate_download_limits(limits)?;
    cancellation.check()?;
    fs::create_dir_all(cache_directory)
        .await
        .with_context(|| format!("failed to create {}", cache_directory.display()))?;
    let final_path = cache_directory.join(&spec.file_name);
    if final_path.exists() {
        let (sha256, compressed_bytes) =
            hash_file(&final_path, limits.maximum_compressed_bytes, cancellation).await?;
        if sha256 == expected_sha256 {
            return Ok(DownloadedArchive {
                path: final_path,
                sha256,
                compressed_bytes,
                reused_cache: true,
            });
        }
        fs::remove_file(&final_path)
            .await
            .with_context(|| format!("failed to remove invalid cache {}", final_path.display()))?;
    }

    let partial_path = cache_directory.join(format!("{}.partial", spec.file_name));
    let _ = fs::remove_file(&partial_path).await;
    let result = download_archive_inner(
        client,
        spec,
        expected_sha256,
        &partial_path,
        limits,
        cancellation,
    )
    .await;
    match result {
        Ok((sha256, compressed_bytes)) => {
            fs::rename(&partial_path, &final_path)
                .await
                .with_context(|| {
                    format!(
                        "failed to atomically publish {} as {}",
                        partial_path.display(),
                        final_path.display()
                    )
                })?;
            Ok(DownloadedArchive {
                path: final_path,
                sha256,
                compressed_bytes,
                reused_cache: false,
            })
        }
        Err(error) => {
            let _ = fs::remove_file(&partial_path).await;
            Err(error)
        }
    }
}

async fn download_archive_inner(
    client: &reqwest::Client,
    spec: &BinanceArchiveSpec,
    expected_sha256: &str,
    partial_path: &Path,
    limits: &ArchiveDownloadLimits,
    cancellation: &ArchiveCancellation,
) -> Result<(String, u64)> {
    cancellation.check()?;
    let mut response = client
        .get(&spec.source_uri)
        .send()
        .await
        .with_context(|| format!("failed to request {}", spec.source_uri))?
        .error_for_status()
        .with_context(|| format!("Binance rejected {}", spec.source_uri))?;
    if response
        .content_length()
        .is_some_and(|length| length > limits.maximum_compressed_bytes)
    {
        bail!("Binance archive exceeded the configured compressed size limit");
    }
    let mut file = fs::File::create(partial_path)
        .await
        .with_context(|| format!("failed to create {}", partial_path.display()))?;
    let mut hasher = Sha256::new();
    let mut bytes = 0u64;
    loop {
        cancellation.check()?;
        let chunk = timeout(limits.chunk_idle_timeout, response.chunk())
            .await
            .context("Binance archive download stalled")??;
        let Some(chunk) = chunk else { break };
        bytes = bytes
            .checked_add(u64::try_from(chunk.len()).context("archive chunk length overflow")?)
            .context("archive byte count overflow")?;
        if bytes > limits.maximum_compressed_bytes {
            bail!("Binance archive exceeded the configured compressed size limit");
        }
        file.write_all(&chunk)
            .await
            .context("failed to write Binance archive chunk")?;
        hasher.update(&chunk);
    }
    file.flush()
        .await
        .context("failed to flush Binance archive")?;
    file.sync_all()
        .await
        .context("failed to fsync Binance archive")?;
    let sha256 = format!("{:x}", hasher.finalize());
    if sha256 != expected_sha256 {
        bail!("Binance archive checksum mismatch: expected {expected_sha256}, received {sha256}");
    }
    Ok((sha256, bytes))
}

async fn hash_file(
    path: &Path,
    maximum_bytes: u64,
    cancellation: &ArchiveCancellation,
) -> Result<(String, u64)> {
    let mut file = fs::File::open(path)
        .await
        .with_context(|| format!("failed to open {}", path.display()))?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut hasher = Sha256::new();
    let mut bytes = 0u64;
    loop {
        cancellation.check()?;
        let read = file
            .read(&mut buffer)
            .await
            .context("failed to hash cached archive")?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(u64::try_from(read).context("archive read length overflow")?)
            .context("archive byte count overflow")?;
        if bytes > maximum_bytes {
            bail!("cached archive exceeded the configured compressed size limit");
        }
        hasher.update(&buffer[..read]);
    }
    Ok((format!("{:x}", hasher.finalize()), bytes))
}

pub fn spawn_aggregate_trade_parser(
    path: PathBuf,
    spec: BinanceArchiveSpec,
    batch_records: usize,
) -> Result<(
    mpsc::Receiver<Vec<BinanceAggregateTradeRecord>>,
    JoinHandle<Result<ArchiveParseSummary>>,
)> {
    spawn_aggregate_trade_parser_with_control(
        path,
        spec,
        batch_records,
        DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES,
        ArchiveCancellation::default(),
    )
}

pub fn spawn_aggregate_trade_parser_with_control(
    path: PathBuf,
    spec: BinanceArchiveSpec,
    batch_records: usize,
    maximum_uncompressed_bytes: u64,
    cancellation: ArchiveCancellation,
) -> Result<(
    mpsc::Receiver<Vec<BinanceAggregateTradeRecord>>,
    JoinHandle<Result<ArchiveParseSummary>>,
)> {
    validate_batch_records(batch_records)?;
    validate_uncompressed_limit(maximum_uncompressed_bytes)?;
    let (sender, receiver) = mpsc::channel(1);
    let handle = tokio::task::spawn_blocking(move || {
        parse_aggregate_trade_archive(
            &path,
            &spec,
            batch_records,
            maximum_uncompressed_bytes,
            &cancellation,
            sender,
        )
    });
    Ok((receiver, handle))
}

pub fn spawn_one_second_kline_parser(
    path: PathBuf,
    spec: BinanceArchiveSpec,
    batch_records: usize,
) -> Result<(
    mpsc::Receiver<Vec<BinanceOneSecondKlineRecord>>,
    JoinHandle<Result<ArchiveParseSummary>>,
)> {
    spawn_one_second_kline_parser_with_control(
        path,
        spec,
        batch_records,
        DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES,
        ArchiveCancellation::default(),
    )
}

pub fn spawn_one_second_kline_parser_with_control(
    path: PathBuf,
    spec: BinanceArchiveSpec,
    batch_records: usize,
    maximum_uncompressed_bytes: u64,
    cancellation: ArchiveCancellation,
) -> Result<(
    mpsc::Receiver<Vec<BinanceOneSecondKlineRecord>>,
    JoinHandle<Result<ArchiveParseSummary>>,
)> {
    validate_batch_records(batch_records)?;
    validate_uncompressed_limit(maximum_uncompressed_bytes)?;
    let (sender, receiver) = mpsc::channel(1);
    let handle = tokio::task::spawn_blocking(move || {
        parse_one_second_kline_archive(
            &path,
            &spec,
            batch_records,
            maximum_uncompressed_bytes,
            &cancellation,
            sender,
        )
    });
    Ok((receiver, handle))
}

fn parse_aggregate_trade_archive(
    path: &Path,
    spec: &BinanceArchiveSpec,
    batch_records: usize,
    maximum_uncompressed_bytes: u64,
    cancellation: &ArchiveCancellation,
    sender: mpsc::Sender<Vec<BinanceAggregateTradeRecord>>,
) -> Result<ArchiveParseSummary> {
    let mut archive = open_validated_zip(path, &spec.entry_name, maximum_uncompressed_bytes)?;
    let entry = archive
        .by_index(0)
        .context("failed to open Binance ZIP entry")?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(BufReader::new(entry));
    let mut batch = Vec::with_capacity(batch_records);
    let mut summary = ArchiveParseSummary::default();
    let mut previous_id = None;
    for (ordinal, record) in reader.byte_records().enumerate() {
        cancellation.check()?;
        let record =
            record.with_context(|| format!("invalid aggregate-trade CSV row {}", ordinal + 1))?;
        if ordinal == 0 && is_header(&record, &[b"agg_trade_id", b"aggregate_trade_id", b"a"]) {
            continue;
        }
        require_columns(&record, 8, "aggregate trade", ordinal)?;
        let row = BinanceAggregateTradeRecord {
            symbol: BINANCE_SYMBOL.to_string(),
            aggregate_trade_id: parse_i64(&record, 0, "aggregate_trade_id")?,
            price: parse_decimal(&record, 1, "price")?,
            quantity: parse_decimal(&record, 2, "quantity")?,
            first_trade_id: parse_i64(&record, 3, "first_trade_id")?,
            last_trade_id: parse_i64(&record, 4, "last_trade_id")?,
            trade_timestamp: parse_epoch(field(&record, 5, "trade_timestamp")?)?,
            buyer_maker: parse_bool(&record, 6, "buyer_maker")?,
            best_match: parse_bool(&record, 7, "best_match")?,
        };
        if row.price <= Decimal::ZERO || row.quantity <= Decimal::ZERO {
            bail!("aggregate trade price and quantity must be positive");
        }
        if row.aggregate_trade_id < 0
            || row.first_trade_id < 0
            || row.last_trade_id < row.first_trade_id
        {
            bail!("aggregate trade identifiers were invalid");
        }
        if previous_id.is_some_and(|previous| row.aggregate_trade_id <= previous) {
            bail!("aggregate trade identifiers were not strictly increasing");
        }
        require_archive_date(row.trade_timestamp, spec.date)?;
        previous_id = Some(row.aggregate_trade_id);
        summary.observe(row.trade_timestamp);
        batch.push(row);
        if batch.len() == batch_records {
            send_batch(&sender, &mut batch, &mut summary)?;
        }
    }
    if !batch.is_empty() {
        send_batch(&sender, &mut batch, &mut summary)?;
    }
    Ok(summary)
}

fn parse_one_second_kline_archive(
    path: &Path,
    spec: &BinanceArchiveSpec,
    batch_records: usize,
    maximum_uncompressed_bytes: u64,
    cancellation: &ArchiveCancellation,
    sender: mpsc::Sender<Vec<BinanceOneSecondKlineRecord>>,
) -> Result<ArchiveParseSummary> {
    let mut archive = open_validated_zip(path, &spec.entry_name, maximum_uncompressed_bytes)?;
    let entry = archive
        .by_index(0)
        .context("failed to open Binance ZIP entry")?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(BufReader::new(entry));
    let mut batch = Vec::with_capacity(batch_records);
    let mut summary = ArchiveParseSummary::default();
    let mut previous_open = None;
    for (ordinal, record) in reader.byte_records().enumerate() {
        cancellation.check()?;
        let record = record.with_context(|| format!("invalid kline CSV row {}", ordinal + 1))?;
        if ordinal == 0 && is_header(&record, &[b"open_time"]) {
            continue;
        }
        require_columns(&record, 12, "one-second kline", ordinal)?;
        let row = BinanceOneSecondKlineRecord {
            symbol: BINANCE_SYMBOL.to_string(),
            open_timestamp: parse_epoch(field(&record, 0, "open_timestamp")?)?,
            open_price: parse_decimal(&record, 1, "open_price")?,
            high_price: parse_decimal(&record, 2, "high_price")?,
            low_price: parse_decimal(&record, 3, "low_price")?,
            close_price: parse_decimal(&record, 4, "close_price")?,
            base_volume: parse_decimal(&record, 5, "base_volume")?,
            close_timestamp: parse_epoch(field(&record, 6, "close_timestamp")?)?,
            quote_volume: parse_decimal(&record, 7, "quote_volume")?,
            trade_count: parse_i64(&record, 8, "trade_count")?,
            taker_buy_base_volume: parse_decimal(&record, 9, "taker_buy_base_volume")?,
            taker_buy_quote_volume: parse_decimal(&record, 10, "taker_buy_quote_volume")?,
        };
        Decimal::from_str(field(&record, 11, "ignore")?)
            .context("CSV ignore was not a valid decimal")?;
        validate_kline(&row)?;
        require_archive_date(row.open_timestamp, spec.date)?;
        if previous_open.is_some_and(|previous| row.open_timestamp <= previous) {
            bail!("kline open timestamps were not strictly increasing");
        }
        previous_open = Some(row.open_timestamp);
        summary.observe(row.open_timestamp);
        batch.push(row);
        if batch.len() == batch_records {
            send_batch(&sender, &mut batch, &mut summary)?;
        }
    }
    if !batch.is_empty() {
        send_batch(&sender, &mut batch, &mut summary)?;
    }
    Ok(summary)
}

fn open_validated_zip(
    path: &Path,
    expected_entry_name: &str,
    maximum_uncompressed_bytes: u64,
) -> Result<ZipArchive<File>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut archive = ZipArchive::new(file).context("failed to decode Binance ZIP archive")?;
    let entries = (0..archive.len())
        .filter_map(|index| {
            archive
                .by_index(index)
                .ok()
                .and_then(|entry| (!entry.is_dir()).then_some((index, entry.name().to_string())))
        })
        .collect::<Vec<_>>();
    if entries.len() != 1 {
        bail!("Binance ZIP archive must contain exactly one file");
    }
    let (entry_index, entry_name) = &entries[0];
    validate_archive_entry_name(entry_name, expected_entry_name)?;
    if *entry_index != 0 {
        bail!("Binance ZIP data entry must be the first archive entry");
    }
    let entry = archive
        .by_index(*entry_index)
        .context("failed to inspect Binance ZIP data entry")?;
    if entry.size() > maximum_uncompressed_bytes {
        bail!("Binance ZIP entry exceeded the configured uncompressed size limit");
    }
    if !matches!(
        entry.compression(),
        zip::CompressionMethod::Stored | zip::CompressionMethod::Deflated
    ) {
        bail!("Binance ZIP entry used an unsupported compression method");
    }
    drop(entry);
    Ok(archive)
}

fn validate_archive_entry_name(entry_name: &str, expected_entry_name: &str) -> Result<()> {
    let path = Path::new(entry_name);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("Binance ZIP entry used an unsafe path");
    }
    if entry_name != expected_entry_name {
        bail!("Binance ZIP entry {entry_name} did not match {expected_entry_name}");
    }
    Ok(())
}

fn validate_kline(row: &BinanceOneSecondKlineRecord) -> Result<()> {
    if row.open_price <= Decimal::ZERO
        || row.high_price <= Decimal::ZERO
        || row.low_price <= Decimal::ZERO
        || row.close_price <= Decimal::ZERO
        || row.high_price < row.open_price
        || row.high_price < row.low_price
        || row.high_price < row.close_price
        || row.low_price > row.open_price
        || row.low_price > row.close_price
    {
        bail!("one-second kline OHLC values were invalid");
    }
    if row.base_volume < Decimal::ZERO
        || row.quote_volume < Decimal::ZERO
        || row.trade_count < 0
        || row.taker_buy_base_volume < Decimal::ZERO
        || row.taker_buy_quote_volume < Decimal::ZERO
        || row.taker_buy_base_volume > row.base_volume
        || row.taker_buy_quote_volume > row.quote_volume
    {
        bail!("one-second kline volume values were invalid");
    }
    if row.close_timestamp < row.open_timestamp
        || row.close_timestamp >= row.open_timestamp + ChronoDuration::seconds(1)
    {
        bail!("one-second kline close timestamp was outside its interval");
    }
    Ok(())
}

fn send_batch<T>(
    sender: &mpsc::Sender<Vec<T>>,
    batch: &mut Vec<T>,
    summary: &mut ArchiveParseSummary,
) -> Result<()> {
    let replacement = Vec::with_capacity(batch.capacity());
    let ready = std::mem::replace(batch, replacement);
    summary.observe_batch(ready.len());
    sender
        .blocking_send(ready)
        .map_err(|_| anyhow::anyhow!("archive consumer stopped before parsing completed"))
}

fn validate_batch_records(batch_records: usize) -> Result<()> {
    if !(1..=25_000).contains(&batch_records) {
        bail!("archive batch size must be between 1 and 25000 records");
    }
    Ok(())
}

fn validate_uncompressed_limit(maximum_uncompressed_bytes: u64) -> Result<()> {
    if maximum_uncompressed_bytes == 0 {
        bail!("archive uncompressed size limit must be positive");
    }
    Ok(())
}

fn validate_download_limits(limits: &ArchiveDownloadLimits) -> Result<()> {
    if limits.maximum_compressed_bytes == 0 {
        bail!("archive compressed size limit must be positive");
    }
    if limits.chunk_idle_timeout.is_zero() {
        bail!("archive chunk idle timeout must be positive");
    }
    Ok(())
}

fn is_header(record: &csv::ByteRecord, accepted_names: &[&[u8]]) -> bool {
    record.get(0).is_some_and(|field| {
        accepted_names
            .iter()
            .any(|accepted| field.eq_ignore_ascii_case(accepted))
    })
}

fn require_columns(
    record: &csv::ByteRecord,
    expected: usize,
    kind: &str,
    ordinal: usize,
) -> Result<()> {
    if record.len() != expected {
        bail!(
            "{kind} CSV row {} had {} columns, expected {expected}",
            ordinal + 1,
            record.len()
        );
    }
    Ok(())
}

fn field<'a>(record: &'a csv::ByteRecord, index: usize, name: &str) -> Result<&'a str> {
    let raw = record
        .get(index)
        .with_context(|| format!("CSV row was missing {name}"))?;
    std::str::from_utf8(raw).with_context(|| format!("CSV {name} was not UTF-8"))
}

fn parse_i64(record: &csv::ByteRecord, index: usize, name: &str) -> Result<i64> {
    field(record, index, name)?
        .parse()
        .with_context(|| format!("CSV {name} was not a valid integer"))
}

fn parse_decimal(record: &csv::ByteRecord, index: usize, name: &str) -> Result<Decimal> {
    Decimal::from_str(field(record, index, name)?)
        .with_context(|| format!("CSV {name} was not a valid decimal"))
}

fn parse_bool(record: &csv::ByteRecord, index: usize, name: &str) -> Result<bool> {
    let value = field(record, index, name)?;
    if value.eq_ignore_ascii_case("true") {
        Ok(true)
    } else if value.eq_ignore_ascii_case("false") {
        Ok(false)
    } else {
        bail!("CSV {name} was not a boolean")
    }
}

pub fn parse_epoch(raw: &str) -> Result<DateTime<Utc>> {
    let raw = raw
        .parse::<i64>()
        .context("timestamp was not a valid integer")?;
    let (seconds, nanos) = if raw >= 1_000_000_000_000_000 {
        (
            raw.div_euclid(1_000_000),
            u32::try_from(raw.rem_euclid(1_000_000) * 1_000)
                .context("microsecond timestamp remainder overflow")?,
        )
    } else if raw >= 1_000_000_000_000 {
        (
            raw.div_euclid(1_000),
            u32::try_from(raw.rem_euclid(1_000) * 1_000_000)
                .context("millisecond timestamp remainder overflow")?,
        )
    } else {
        (raw, 0)
    };
    Utc.timestamp_opt(seconds, nanos)
        .single()
        .context("timestamp was outside the supported UTC range")
}

fn require_archive_date(timestamp: DateTime<Utc>, date: NaiveDate) -> Result<()> {
    let start = Utc.from_utc_datetime(
        &date
            .and_hms_opt(0, 0, 0)
            .context("archive date was outside the supported UTC range")?,
    );
    let end = start + ChronoDuration::days(1);
    if timestamp < start || timestamp >= end {
        bail!("record timestamp {timestamp} was outside archive date {date}");
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("expected checksum was not a valid SHA-256 digest");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use chrono::Datelike;
    use tempfile::TempDir;
    use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

    use super::*;

    fn date() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 7, 7).unwrap()
    }

    fn write_zip(directory: &TempDir, entry_name: &str, csv: &str) -> PathBuf {
        let path = directory.path().join("fixture.zip");
        let file = File::create(&path).unwrap();
        let mut zip = ZipWriter::new(file);
        zip.start_file(
            entry_name,
            SimpleFileOptions::default().compression_method(CompressionMethod::Deflated),
        )
        .unwrap();
        zip.write_all(csv.as_bytes()).unwrap();
        zip.finish().unwrap();
        path
    }

    fn micros(second: u32, trailing_micros: u32) -> i64 {
        let timestamp = Utc
            .with_ymd_and_hms(2026, 7, 7, 0, 0, second)
            .single()
            .unwrap();
        timestamp.timestamp_micros() + i64::from(trailing_micros)
    }

    #[test]
    fn archive_urls_and_checksum_contract_are_stable() {
        let spec = BinanceArchiveSpec::new(
            "https://data.binance.vision/",
            BinanceArchiveKind::AggregateTrades,
            date(),
        );
        assert_eq!(spec.file_name, "BTCUSDT-aggTrades-2026-07-07.zip");
        assert!(spec.source_uri.ends_with(&spec.file_name));
        let checksum = "a".repeat(64);
        assert_eq!(
            parse_checksum_document(&format!("{checksum}  {}", spec.file_name), &spec.file_name)
                .unwrap(),
            checksum
        );
        assert!(parse_checksum_document(
            &format!("{}  wrong.zip", "a".repeat(64)),
            &spec.file_name
        )
        .is_err());
    }

    #[test]
    fn checksum_parser_rejects_malformed_or_ambiguous_documents() {
        let expected = "BTCUSDT-aggTrades-2026-07-07.zip";
        assert!(parse_checksum_document("", expected).is_err());
        assert!(parse_checksum_document(
            &format!("{}  {expected} extra", "a".repeat(64)),
            expected
        )
        .is_err());
        assert!(
            parse_checksum_document(&format!("{}  {expected}", "z".repeat(64)), expected).is_err()
        );
    }

    #[test]
    fn timestamp_parser_supports_milliseconds_and_microseconds() {
        let timestamp = Utc.with_ymd_and_hms(2026, 7, 7, 1, 2, 3).single().unwrap();
        assert_eq!(
            parse_epoch(&timestamp.timestamp_millis().to_string()).unwrap(),
            timestamp
        );
        assert_eq!(
            parse_epoch(&(timestamp.timestamp_micros() + 456).to_string()).unwrap(),
            timestamp + ChronoDuration::microseconds(456)
        );
    }

    #[tokio::test]
    async fn aggregate_trade_parser_streams_strictly_bounded_batches() {
        let directory = TempDir::new().unwrap();
        let spec = BinanceArchiveSpec::new(
            "https://example.test",
            BinanceArchiveKind::AggregateTrades,
            date(),
        );
        let mut csv = String::from("aggregate_trade_id,price,quantity,first_trade_id,last_trade_id,trade_timestamp,buyer_maker,best_match\n");
        for id in 0..10_003i64 {
            csv.push_str(&format!(
                "{id},60000.1,0.001,{id},{id},{},false,true\n",
                micros(
                    u32::try_from(id % 60).unwrap(),
                    u32::try_from(id / 60).unwrap()
                )
            ));
        }
        let path = write_zip(&directory, &spec.entry_name, &csv);
        let (mut receiver, handle) = spawn_aggregate_trade_parser(path, spec, 2_000).unwrap();
        let mut rows = 0usize;
        let mut maximum_batch = 0usize;
        while let Some(batch) = receiver.recv().await {
            rows += batch.len();
            maximum_batch = maximum_batch.max(batch.len());
        }
        let summary = handle.await.unwrap().unwrap();
        assert_eq!(rows, 10_003);
        assert_eq!(summary.records, 10_003);
        assert_eq!(summary.batches, 6);
        assert_eq!(maximum_batch, 2_000);
        assert_eq!(summary.maximum_batch_records, 2_000);
    }

    #[tokio::test]
    async fn aggregate_trade_parser_accepts_official_header_and_boolean_casing() {
        let directory = TempDir::new().unwrap();
        let spec = BinanceArchiveSpec::new(
            "https://example.test",
            BinanceArchiveKind::AggregateTrades,
            date(),
        );
        let csv = format!(
            "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match\n1,60000,0.01,1,1,{},False,True\n",
            micros(0, 0)
        );
        let path = write_zip(&directory, &spec.entry_name, &csv);
        let (mut receiver, handle) = spawn_aggregate_trade_parser(path, spec, 10).unwrap();
        assert_eq!(receiver.recv().await.unwrap().len(), 1);
        assert!(receiver.recv().await.is_none());
        assert_eq!(handle.await.unwrap().unwrap().records, 1);
    }

    #[tokio::test]
    async fn kline_parser_accepts_microseconds_and_rejects_invalid_ohlc() {
        let directory = TempDir::new().unwrap();
        let spec = BinanceArchiveSpec::new(
            "https://example.test",
            BinanceArchiveKind::OneSecondKlines,
            date(),
        );
        let open = micros(0, 0);
        let close = open + 999_999;
        let path = write_zip(
            &directory,
            &spec.entry_name,
            &format!("{open},60000,60002,59999,60001,2,{close},120000,3,1,60000,0\n"),
        );
        let (mut receiver, handle) =
            spawn_one_second_kline_parser(path, spec.clone(), 100).unwrap();
        let rows = receiver.recv().await.unwrap();
        assert_eq!(rows.len(), 1);
        assert!(receiver.recv().await.is_none());
        assert_eq!(handle.await.unwrap().unwrap().records, 1);

        let invalid = write_zip(
            &directory,
            &spec.entry_name,
            &format!("{open},60000,59998,59999,60001,2,{close},120000,3,1,60000,0\n"),
        );
        let (mut receiver, handle) = spawn_one_second_kline_parser(invalid, spec, 100).unwrap();
        assert!(receiver.recv().await.is_none());
        assert!(handle.await.unwrap().is_err());
    }

    #[test]
    fn zip_entry_paths_are_exact_and_safe() {
        assert!(validate_archive_entry_name("../data.csv", "data.csv").is_err());
        assert!(validate_archive_entry_name("nested/data.csv", "data.csv").is_err());
        assert!(validate_archive_entry_name("data.csv", "data.csv").is_ok());
    }

    #[tokio::test]
    async fn parser_rejects_zip_entries_over_the_declared_limit() {
        let directory = TempDir::new().unwrap();
        let spec = BinanceArchiveSpec::new(
            "https://example.test",
            BinanceArchiveKind::AggregateTrades,
            date(),
        );
        let csv = format!("1,60000,0.01,1,1,{},false,true\n", micros(0, 0));
        let path = write_zip(&directory, &spec.entry_name, &csv);
        let (mut receiver, handle) = spawn_aggregate_trade_parser_with_control(
            path,
            spec,
            10,
            1,
            ArchiveCancellation::default(),
        )
        .unwrap();
        assert!(receiver.recv().await.is_none());
        assert!(handle.await.unwrap().is_err());
    }

    #[tokio::test]
    async fn parser_observes_cancellation_before_emitting_rows() {
        let directory = TempDir::new().unwrap();
        let spec = BinanceArchiveSpec::new(
            "https://example.test",
            BinanceArchiveKind::AggregateTrades,
            date(),
        );
        let csv = format!("1,60000,0.01,1,1,{},false,true\n", micros(0, 0));
        let path = write_zip(&directory, &spec.entry_name, &csv);
        let cancellation = ArchiveCancellation::default();
        cancellation.cancel();
        let (mut receiver, handle) = spawn_aggregate_trade_parser_with_control(
            path,
            spec,
            10,
            DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES,
            cancellation,
        )
        .unwrap();
        assert!(receiver.recv().await.is_none());
        assert!(handle.await.unwrap().is_err());
    }

    #[test]
    fn source_fixture_date_is_utc() {
        assert_eq!(date().year(), 2026);
    }
}
