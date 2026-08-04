use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::{Field, Row},
};
use rust_decimal::Decimal;
use sha2::{Digest, Sha256};
use tokio::{fs, io::AsyncWriteExt, sync::mpsc, task::JoinHandle, time::timeout};
use uuid::Uuid;

use super::{
    binance_archive::ArchiveCancellation,
    cryptohft_binance_l2::{
        BinanceL2DaySummary, BinanceSpotL2RangeReplay, BinanceSpotL2ReplayEvent,
        BinanceSpotL2ReplayLevel, BinanceSpotL2ReplaySide, CryptoHftBinanceL2Config,
    },
    job::BinanceL2OneSecondFeature,
};

pub const HUGGINGFACE_ARCHIVE_PROVIDER: &str = "huggingface";
pub const DEFAULT_HUGGINGFACE_GOOODDY_BASE_URL: &str =
    "https://huggingface.co/datasets/Goooddy/crypto-lob-stream/resolve/main";
pub const HUGGINGFACE_GOOODDY_MATERIALIZATION_CONTRACT: &str =
    "huggingface-goooddy-binance-spot-btcusdt-l2-features-v1";
pub const HUGGINGFACE_GOOODDY_STRATEGY: &str = "huggingface_goooddy";

const SYMBOL: &str = "BTCUSDT";
const EXCHANGE: &str = "binance";
const MAXIMUM_SOURCE_OBJECT_BYTES: u64 = 512 * 1024 * 1024;
const GOOODDY_SOURCE_IDENTITY_CONTRACT: &str = "resolved-object-sha256-v1";

const DEPTH_COLUMNS: [&str; 8] = [
    "timestamp_ms",
    "asset",
    "side",
    "price",
    "quantity",
    "first_update_id",
    "last_update_id",
    "exchange",
];
const SNAPSHOT_COLUMNS: [&str; 7] = [
    "timestamp_ms",
    "asset",
    "side",
    "price",
    "quantity",
    "last_update_id",
    "exchange",
];

#[derive(Debug, Clone)]
pub struct HuggingFaceBinanceL2Config {
    pub base_url: String,
    pub storage: CryptoHftBinanceL2Config,
}

impl HuggingFaceBinanceL2Config {
    pub fn validate(&self) -> Result<()> {
        if self.base_url.trim().is_empty() || !self.base_url.starts_with("https://") {
            bail!("Hugging Face Binance L2 base URL must be a non-empty HTTPS URL");
        }
        self.storage.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoooddySourceObjectSpec {
    pub remote_path: &'static str,
    pub expected_sha256: &'static str,
    pub expected_bytes: u64,
}

impl GoooddySourceObjectSpec {
    pub fn source_uri(&self, config: &HuggingFaceBinanceL2Config) -> String {
        format!(
            "{}/{}?download=true",
            config.base_url.trim_end_matches('/'),
            self.remote_path
        )
    }

    pub fn archive_path(&self, config: &HuggingFaceBinanceL2Config) -> PathBuf {
        config
            .storage
            .archive_root
            .join("huggingface")
            .join("Goooddy")
            .join("crypto-lob-stream")
            .join(self.remote_path)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoooddyMonthSpec {
    pub target_start: DateTime<Utc>,
    pub target_end: DateTime<Utc>,
    pub source_month: &'static str,
    pub depth: GoooddySourceObjectSpec,
    pub snapshots: GoooddySourceObjectSpec,
}

impl GoooddyMonthSpec {
    pub fn logical_key(&self) -> String {
        format!(
            "huggingface:Goooddy:binance-spot:BTCUSDT:l2-month:{}:{}:{}",
            HUGGINGFACE_GOOODDY_MATERIALIZATION_CONTRACT,
            GOOODDY_SOURCE_IDENTITY_CONTRACT,
            self.source_month
        )
    }

    pub fn combined_checksum(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.depth.expected_sha256.as_bytes());
        hasher.update(b"\n");
        hasher.update(self.snapshots.expected_sha256.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

pub fn goooddy_month_spec(
    target_start: DateTime<Utc>,
    target_end: DateTime<Utc>,
) -> Result<GoooddyMonthSpec> {
    let june_start = utc_midnight(2026, 6, 3)?;
    let july_start = utc_midnight(2026, 7, 1)?;
    let august_start = utc_midnight(2026, 8, 1)?;
    match (target_start, target_end) {
        (start, end) if start == june_start && end == july_start => Ok(GoooddyMonthSpec {
            target_start,
            target_end,
            source_month: "2026-06",
            depth: GoooddySourceObjectSpec {
                remote_path: "depth/binance/BTCUSDT/2026-06.parquet",
                expected_sha256:
                    "6a8ca39bef7b8ec05329bb7c13a2f674a9bb6fdd60191e728d892bee17c45450",
                expected_bytes: 385_932_069,
            },
            snapshots: GoooddySourceObjectSpec {
                remote_path: "snapshots/binance/BTCUSDT/2026-06.parquet",
                expected_sha256:
                    "6b4579b4fb8a06fe46c6f11a824c42490fb4b7833ecb8e1beb53b014969e6cdb",
                expected_bytes: 299_002,
            },
        }),
        (start, end) if start == july_start && end == august_start => Ok(GoooddyMonthSpec {
            target_start,
            target_end,
            source_month: "2026-07",
            depth: GoooddySourceObjectSpec {
                remote_path: "depth/binance/BTCUSDT/2026-07.parquet",
                expected_sha256:
                    "27c87c42dbc420cbb80b0c82b9bfe29e3998ef3fdf7bf06106fe76231b369504",
                expected_bytes: 317_605_448,
            },
            snapshots: GoooddySourceObjectSpec {
                remote_path: "snapshots/binance/BTCUSDT/2026-07.parquet",
                expected_sha256:
                    "6c6e474ff14fba5f53b6f9570c6101fe841ebb75908d9809aa8dbefdeb6b5948",
                expected_bytes: 350_331,
            },
        }),
        _ => bail!(
            "Goooddy Binance spot L2 requests must be exactly [2026-06-03, 2026-07-01) or [2026-07-01, 2026-08-01)"
        ),
    }
}

fn utc_midnight(year: i32, month: u32, day: u32) -> Result<DateTime<Utc>> {
    Ok(NaiveDate::from_ymd_opt(year, month, day)
        .context("invalid pinned Hugging Face source date")?
        .and_hms_opt(0, 0, 0)
        .context("invalid pinned Hugging Face source midnight")?
        .and_utc())
}

#[derive(Debug, Clone)]
pub struct HuggingFaceArchiveManifest {
    pub source_uri: String,
    pub archive_path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub reused_archive: bool,
}

pub async fn download_goooddy_object(
    client: &reqwest::Client,
    config: &HuggingFaceBinanceL2Config,
    spec: &GoooddySourceObjectSpec,
    cancellation: &ArchiveCancellation,
) -> Result<HuggingFaceArchiveManifest> {
    config.validate()?;
    check_cancelled(cancellation)?;
    if spec.expected_bytes == 0 || spec.expected_bytes > MAXIMUM_SOURCE_OBJECT_BYTES {
        bail!("pinned Hugging Face source object size escaped its safety bound");
    }
    let archive_path = spec.archive_path(config);
    let source_uri = spec.source_uri(config);
    if fs::try_exists(&archive_path).await? {
        let (sha256, bytes) = hash_parquet(&archive_path, cancellation).await?;
        if sha256 != spec.expected_sha256 || bytes != spec.expected_bytes {
            bail!(
                "cached Hugging Face object {} did not match its pinned immutable identity",
                archive_path.display()
            );
        }
        return Ok(HuggingFaceArchiveManifest {
            source_uri,
            archive_path,
            sha256,
            bytes,
            reused_archive: true,
        });
    }

    let parent = archive_path
        .parent()
        .context("Hugging Face archive path had no parent")?;
    fs::create_dir_all(parent).await?;
    let partial_path = parent.join(format!(
        ".{}.{}.part",
        archive_path
            .file_name()
            .and_then(|value| value.to_str())
            .context("Hugging Face archive filename was not UTF-8")?,
        Uuid::new_v4()
    ));
    let result = async {
        let request = client.get(&source_uri).send();
        let mut response = timeout(config.storage.download_chunk_idle_timeout, request)
            .await
            .context("Hugging Face source request timed out")??
            .error_for_status()
            .with_context(|| format!("Hugging Face rejected {source_uri}"))?;
        if response
            .content_length()
            .is_some_and(|length| length != spec.expected_bytes)
        {
            bail!("Hugging Face Content-Length did not match the pinned object size");
        }
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial_path)
            .await?;
        let mut hasher = Sha256::new();
        let mut bytes = 0u64;
        while let Some(chunk) =
            timeout(config.storage.download_chunk_idle_timeout, response.chunk())
                .await
                .context("Hugging Face source download stalled")??
        {
            check_cancelled(cancellation)?;
            bytes = bytes
                .checked_add(u64::try_from(chunk.len()).context("source chunk size overflow")?)
                .context("Hugging Face source size overflow")?;
            if bytes > spec.expected_bytes {
                bail!("Hugging Face source exceeded its pinned object size");
            }
            hasher.update(&chunk);
            output.write_all(&chunk).await?;
        }
        output.sync_all().await?;
        drop(output);
        let sha256 = format!("{:x}", hasher.finalize());
        if bytes != spec.expected_bytes || sha256 != spec.expected_sha256 {
            bail!("downloaded Hugging Face source object failed its pinned SHA-256 identity");
        }
        validate_parquet_magic(&partial_path).await?;
        match fs::hard_link(&partial_path, &archive_path).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let (winner_sha, winner_bytes) = hash_parquet(&archive_path, cancellation).await?;
                if winner_sha != sha256 || winner_bytes != bytes {
                    bail!("concurrent Hugging Face archive finalization produced a conflict");
                }
            }
            Err(error) => return Err(error).context("failed to finalize Hugging Face archive"),
        }
        let mut permissions = fs::metadata(&archive_path).await?.permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&archive_path, permissions).await?;
        Ok(HuggingFaceArchiveManifest {
            source_uri,
            archive_path: archive_path.clone(),
            sha256,
            bytes,
            reused_archive: false,
        })
    }
    .await;
    let _ = fs::remove_file(&partial_path).await;
    result
}

async fn hash_parquet(path: &Path, cancellation: &ArchiveCancellation) -> Result<(String, u64)> {
    let path = path.to_path_buf();
    let cancellation = cancellation.clone();
    tokio::task::spawn_blocking(move || {
        check_cancelled(&cancellation)?;
        let mut input = File::open(&path)?;
        let metadata = input.metadata()?;
        if metadata.len() == 0 || metadata.len() > MAXIMUM_SOURCE_OBJECT_BYTES {
            bail!("cached Hugging Face object had an invalid size");
        }
        validate_parquet_magic_blocking(&mut input)?;
        input.seek(SeekFrom::Start(0))?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            check_cancelled(&cancellation)?;
            let read = input.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok((format!("{:x}", hasher.finalize()), metadata.len()))
    })
    .await
    .context("Hugging Face archive hash task failed")?
}

async fn validate_parquet_magic(path: &Path) -> Result<()> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let mut input = File::open(path)?;
        validate_parquet_magic_blocking(&mut input)
    })
    .await
    .context("Hugging Face Parquet validation task failed")?
}

fn validate_parquet_magic_blocking(input: &mut File) -> Result<()> {
    let length = input.metadata()?.len();
    if length < 8 {
        bail!("Hugging Face source object was too short to be Parquet");
    }
    let mut magic = [0u8; 4];
    input.read_exact(&mut magic)?;
    if &magic != b"PAR1" {
        bail!("Hugging Face source object did not start with Parquet magic");
    }
    input.seek(SeekFrom::End(-4))?;
    input.read_exact(&mut magic)?;
    if &magic != b"PAR1" {
        bail!("Hugging Face source object did not end with Parquet magic");
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct GoooddyParseRequest {
    pub target_start: DateTime<Utc>,
    pub target_end: DateTime<Utc>,
    pub depth_archive: HuggingFaceArchiveManifest,
    pub snapshot_archive: HuggingFaceArchiveManifest,
    pub output_batch_rows: usize,
    pub cancellation: ArchiveCancellation,
}

pub fn spawn_goooddy_parser(
    config: HuggingFaceBinanceL2Config,
    request: GoooddyParseRequest,
) -> (
    mpsc::Receiver<Vec<BinanceL2OneSecondFeature>>,
    JoinHandle<Result<BinanceL2DaySummary>>,
) {
    let (sender, receiver) = mpsc::channel(1);
    let handle = tokio::spawn(async move {
        config.validate()?;
        let cancellation = request.cancellation.clone();
        tokio::task::spawn_blocking(move || {
            replay_goooddy_month(&config, request, sender, cancellation)
        })
        .await
        .context("Goooddy Binance L2 replay task failed")?
    });
    (receiver, handle)
}

fn replay_goooddy_month(
    config: &HuggingFaceBinanceL2Config,
    request: GoooddyParseRequest,
    sender: mpsc::Sender<Vec<BinanceL2OneSecondFeature>>,
    cancellation: ArchiveCancellation,
) -> Result<BinanceL2DaySummary> {
    if request.target_end <= request.target_start {
        bail!("Goooddy replay target range was empty");
    }
    let mut replay = BinanceSpotL2RangeReplay::new(
        &config.storage,
        request.target_start,
        request.target_end,
        request.output_batch_rows,
        sender,
        cancellation.clone(),
    )?;
    let (snapshots, snapshot_rows) =
        read_snapshot_events(&request.snapshot_archive.archive_path, &cancellation)?;
    let mut snapshot_index = 0usize;
    let mut raw_rows = snapshot_rows;
    let mut last_update_key: Option<(i64, i64)> = None;
    stream_depth_events(&request.depth_archive.archive_path, &cancellation, |item| {
        let DepthStreamItem::Event(event, event_rows) = item else {
            replay.mark_source_gap();
            return Ok(());
        };
        let update_key = (
            event.event_time_ms,
            event
                .final_update_id
                .context("Goooddy update did not have a final update id")?,
        );
        if last_update_key.is_some_and(|previous| update_key < previous) {
            bail!("Goooddy depth events were not monotonic");
        }
        last_update_key = Some(update_key);
        while let Some(snapshot) = snapshots.get(snapshot_index) {
            let snapshot_key = (
                snapshot.event_time_ms,
                snapshot
                    .last_update_id
                    .context("Goooddy snapshot did not have a last update id")?,
            );
            if snapshot_key > update_key {
                break;
            }
            replay.process(snapshot.clone())?;
            snapshot_index += 1;
        }
        replay.process(event)?;
        raw_rows = raw_rows.saturating_add(event_rows);
        Ok(())
    })?;
    for snapshot in snapshots.into_iter().skip(snapshot_index) {
        replay.process(snapshot)?;
    }
    let mut summary = replay.finish()?;
    summary.raw_rows = raw_rows;
    Ok(summary)
}

fn read_snapshot_events(
    path: &Path,
    cancellation: &ArchiveCancellation,
) -> Result<(Vec<BinanceSpotL2ReplayEvent>, u64)> {
    let reader = open_validated_reader(path, &SNAPSHOT_COLUMNS)?;
    let mut events = Vec::new();
    let mut pending: Option<BinanceSpotL2ReplayEvent> = None;
    let mut raw_rows = 0u64;
    for row_group_index in 0..reader.num_row_groups() {
        check_cancelled(cancellation)?;
        let row_group = reader.get_row_group(row_group_index).with_context(|| {
            format!("failed to open Goooddy snapshot row group {row_group_index}")
        })?;
        let rows = row_group.get_row_iter(None).with_context(|| {
            format!("failed to stream Goooddy snapshot row group {row_group_index}")
        })?;
        for (row_ordinal, row) in rows.enumerate() {
            check_cancelled(cancellation)?;
            let row = row.with_context(|| {
                format!("failed to decode Goooddy snapshot row group {row_group_index}, row {row_ordinal}")
            })?;
            let (key, level) = parse_snapshot_row(row)?;
            raw_rows = raw_rows.saturating_add(1);
            push_grouped_event(&mut events, &mut pending, key, level)?;
        }
    }
    if let Some(event) = pending {
        events.push(event);
    }
    for pair in events.windows(2) {
        if replay_event_key(&pair[1]) < replay_event_key(&pair[0]) {
            bail!("Goooddy snapshots were not monotonic");
        }
    }
    Ok((events, raw_rows))
}

enum DepthStreamItem {
    Event(BinanceSpotL2ReplayEvent, u64),
    Gap,
}

fn stream_depth_events(
    path: &Path,
    cancellation: &ArchiveCancellation,
    mut consumer: impl FnMut(DepthStreamItem) -> Result<()>,
) -> Result<()> {
    let reader = open_validated_reader(path, &DEPTH_COLUMNS)?;
    let mut pending: Option<BinanceSpotL2ReplayEvent> = None;
    let mut pending_rows = 0u64;
    for row_group_index in 0..reader.num_row_groups() {
        check_cancelled(cancellation)?;
        let row_group = reader
            .get_row_group(row_group_index)
            .with_context(|| format!("failed to open Goooddy depth row group {row_group_index}"))?;
        let rows = match row_group.get_row_iter(None) {
            Ok(rows) => rows,
            Err(_) => {
                pending = None;
                pending_rows = 0;
                consumer(DepthStreamItem::Gap)?;
                continue;
            }
        };
        let mut row_group_failed = false;
        for (_row_ordinal, row) in rows.enumerate() {
            check_cancelled(cancellation)?;
            let row = match row {
                Ok(row) => row,
                Err(_) => {
                    pending = None;
                    pending_rows = 0;
                    consumer(DepthStreamItem::Gap)?;
                    row_group_failed = true;
                    break;
                }
            };
            let (key, level) = parse_depth_row(row)?;
            match pending.as_mut() {
                Some(event) if same_replay_event(event, &key) => {
                    push_level(event, level)?;
                    pending_rows = pending_rows.saturating_add(1);
                }
                Some(_) => {
                    let complete = pending
                        .take()
                        .context("pending Goooddy event disappeared")?;
                    consumer(DepthStreamItem::Event(complete, pending_rows))?;
                    pending = Some(event_with_level(key, level));
                    pending_rows = 1;
                }
                None => {
                    pending = Some(event_with_level(key, level));
                    pending_rows = 1;
                }
            }
        }
        if row_group_failed {
            continue;
        }
    }
    if let Some(event) = pending {
        consumer(DepthStreamItem::Event(event, pending_rows))?;
    }
    Ok(())
}

fn open_validated_reader(
    path: &Path,
    expected_columns: &[&str],
) -> Result<SerializedFileReader<File>> {
    let file = File::open(path).with_context(|| {
        format!(
            "failed to open Hugging Face Parquet file {}",
            path.display()
        )
    })?;
    let reader = SerializedFileReader::new(file)?;
    let fields = reader.metadata().file_metadata().schema().get_fields();
    if fields.len() != expected_columns.len() {
        bail!("Hugging Face Parquet schema had an unexpected column count");
    }
    for (index, expected) in expected_columns.iter().enumerate() {
        if fields[index].name() != *expected {
            bail!(
                "Hugging Face Parquet column {index} was {}; expected {expected}",
                fields[index].name()
            );
        }
    }
    Ok(reader)
}

fn parse_snapshot_row(row: Row) -> Result<(BinanceSpotL2ReplayEvent, BinanceSpotL2ReplayLevel)> {
    let columns = row.into_columns();
    if columns.len() != SNAPSHOT_COLUMNS.len() {
        bail!("Goooddy snapshot row did not match its validated schema");
    }
    validate_identity(&columns[1].1, &columns[6].1)?;
    let event_time_ms = required_long(&columns[0].1, "timestamp_ms")?;
    let last_update_id = required_long(&columns[5].1, "last_update_id")?;
    let level = parse_level(&columns[2].1, &columns[3].1, &columns[4].1, true)?;
    Ok((
        BinanceSpotL2ReplayEvent {
            event_time_ms,
            first_update_id: None,
            final_update_id: None,
            last_update_id: Some(last_update_id),
            levels: Vec::new(),
        },
        level,
    ))
}

fn parse_depth_row(row: Row) -> Result<(BinanceSpotL2ReplayEvent, BinanceSpotL2ReplayLevel)> {
    let columns = row.into_columns();
    if columns.len() != DEPTH_COLUMNS.len() {
        bail!("Goooddy depth row did not match its validated schema");
    }
    validate_identity(&columns[1].1, &columns[7].1)?;
    let event_time_ms = required_long(&columns[0].1, "timestamp_ms")?;
    let first_update_id = required_long(&columns[5].1, "first_update_id")?;
    let final_update_id = required_long(&columns[6].1, "last_update_id")?;
    if first_update_id < 0 || final_update_id < first_update_id {
        bail!("Goooddy update sequence was invalid");
    }
    let level = parse_level(&columns[2].1, &columns[3].1, &columns[4].1, false)?;
    Ok((
        BinanceSpotL2ReplayEvent {
            event_time_ms,
            first_update_id: Some(first_update_id),
            final_update_id: Some(final_update_id),
            last_update_id: None,
            levels: Vec::new(),
        },
        level,
    ))
}

fn validate_identity(asset: &Field, exchange: &Field) -> Result<()> {
    if required_string(asset, "asset")? != SYMBOL
        || required_string(exchange, "exchange")? != EXCHANGE
    {
        bail!("Goooddy archive identity was not Binance spot BTCUSDT");
    }
    Ok(())
}

fn parse_level(
    side: &Field,
    price: &Field,
    quantity: &Field,
    require_positive_quantity: bool,
) -> Result<BinanceSpotL2ReplayLevel> {
    let side = match required_string(side, "side")? {
        "bid" => BinanceSpotL2ReplaySide::Bid,
        "ask" => BinanceSpotL2ReplaySide::Ask,
        value => bail!("Goooddy archive contained unsupported side {value}"),
    };
    let price = required_decimal_double(price, "price")?;
    let quantity = required_decimal_double(quantity, "quantity")?;
    if price <= Decimal::ZERO
        || quantity < Decimal::ZERO
        || (require_positive_quantity && quantity == Decimal::ZERO)
    {
        bail!("Goooddy archive contained an invalid price level");
    }
    Ok(BinanceSpotL2ReplayLevel {
        side,
        price,
        quantity,
    })
}

fn required_long(field: &Field, name: &str) -> Result<i64> {
    let Field::Long(value) = field else {
        bail!("Goooddy {name} was not a required int64");
    };
    Ok(*value)
}

fn required_string<'a>(field: &'a Field, name: &str) -> Result<&'a str> {
    let Field::Str(value) = field else {
        bail!("Goooddy {name} was not a required string");
    };
    Ok(value)
}

fn required_decimal_double(field: &Field, name: &str) -> Result<Decimal> {
    let Field::Double(value) = field else {
        bail!("Goooddy {name} was not a required double");
    };
    if !value.is_finite() {
        bail!("Goooddy {name} was not finite");
    }
    Decimal::from_str(&value.to_string()).with_context(|| format!("Goooddy {name} was invalid"))
}

fn push_grouped_event(
    events: &mut Vec<BinanceSpotL2ReplayEvent>,
    pending: &mut Option<BinanceSpotL2ReplayEvent>,
    key: BinanceSpotL2ReplayEvent,
    level: BinanceSpotL2ReplayLevel,
) -> Result<()> {
    match pending.as_mut() {
        Some(event) if same_replay_event(event, &key) => push_level(event, level),
        Some(_) => {
            events.push(
                pending
                    .take()
                    .context("pending Goooddy event disappeared")?,
            );
            *pending = Some(event_with_level(key, level));
            Ok(())
        }
        None => {
            *pending = Some(event_with_level(key, level));
            Ok(())
        }
    }
}

fn event_with_level(
    mut event: BinanceSpotL2ReplayEvent,
    level: BinanceSpotL2ReplayLevel,
) -> BinanceSpotL2ReplayEvent {
    event.levels.push(level);
    event
}

fn push_level(event: &mut BinanceSpotL2ReplayEvent, level: BinanceSpotL2ReplayLevel) -> Result<()> {
    if event.levels.len() >= 10_000 {
        bail!("Goooddy logical event exceeded the price-level safety bound");
    }
    event.levels.push(level);
    Ok(())
}

fn same_replay_event(left: &BinanceSpotL2ReplayEvent, right: &BinanceSpotL2ReplayEvent) -> bool {
    left.event_time_ms == right.event_time_ms
        && left.first_update_id == right.first_update_id
        && left.final_update_id == right.final_update_id
        && left.last_update_id == right.last_update_id
}

fn replay_event_key(event: &BinanceSpotL2ReplayEvent) -> (i64, i64) {
    (
        event.event_time_ms,
        event
            .last_update_id
            .or(event.final_update_id)
            .unwrap_or(i64::MIN),
    )
}

fn check_cancelled(cancellation: &ArchiveCancellation) -> Result<()> {
    if cancellation.is_cancelled() {
        bail!("Hugging Face Binance L2 operation cancelled");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_months_have_immutable_distinct_identities() {
        let june = goooddy_month_spec(
            utc_midnight(2026, 6, 3).unwrap(),
            utc_midnight(2026, 7, 1).unwrap(),
        )
        .unwrap();
        let july = goooddy_month_spec(
            utc_midnight(2026, 7, 1).unwrap(),
            utc_midnight(2026, 8, 1).unwrap(),
        )
        .unwrap();

        assert_ne!(june.logical_key(), july.logical_key());
        assert_ne!(june.combined_checksum(), july.combined_checksum());
        assert_eq!(june.depth.expected_bytes, 385_932_069);
        assert_eq!(july.depth.expected_bytes, 317_605_448);
    }

    #[test]
    fn strategy_refuses_unpinned_ranges() {
        assert!(goooddy_month_spec(
            utc_midnight(2026, 6, 4).unwrap(),
            utc_midnight(2026, 7, 1).unwrap(),
        )
        .is_err());
    }

    #[tokio::test]
    #[ignore = "requires downloaded immutable Goooddy Parquet fixtures"]
    async fn downloaded_months_decode_and_replay() {
        let Ok(root) = std::env::var("POLYMARKET_HUGGINGFACE_TEST_ARCHIVE_ROOT") else {
            return;
        };
        let root = PathBuf::from(root);
        let config = HuggingFaceBinanceL2Config {
            base_url: DEFAULT_HUGGINGFACE_GOOODDY_BASE_URL.to_string(),
            storage: CryptoHftBinanceL2Config::new(root, PathBuf::from("/tmp/goooddy-l2-test")),
        };
        for (start, end) in [((2026, 6, 3), (2026, 7, 1)), ((2026, 7, 1), (2026, 8, 1))] {
            let spec = goooddy_month_spec(
                utc_midnight(start.0, start.1, start.2).unwrap(),
                utc_midnight(end.0, end.1, end.2).unwrap(),
            )
            .unwrap();
            let (mut receiver, parser) = spawn_goooddy_parser(
                config.clone(),
                GoooddyParseRequest {
                    target_start: spec.target_start,
                    target_end: spec.target_end,
                    depth_archive: HuggingFaceArchiveManifest {
                        source_uri: spec.depth.source_uri(&config),
                        archive_path: spec.depth.archive_path(&config),
                        sha256: spec.depth.expected_sha256.to_string(),
                        bytes: spec.depth.expected_bytes,
                        reused_archive: true,
                    },
                    snapshot_archive: HuggingFaceArchiveManifest {
                        source_uri: spec.snapshots.source_uri(&config),
                        archive_path: spec.snapshots.archive_path(&config),
                        sha256: spec.snapshots.expected_sha256.to_string(),
                        bytes: spec.snapshots.expected_bytes,
                        reused_archive: true,
                    },
                    output_batch_rows: 1_000,
                    cancellation: ArchiveCancellation::default(),
                },
            );
            let mut rows = 0u64;
            while let Some(batch) = receiver.recv().await {
                rows = rows.saturating_add(u64::try_from(batch.len()).unwrap());
            }
            let summary = parser.await.unwrap().unwrap();
            eprintln!("{}: {summary:?}", spec.source_month);
            assert_eq!(rows, summary.emitted_feature_rows);
            assert!(summary.snapshot_events > 0);
        }
    }
}
