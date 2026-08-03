use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    ffi::CString,
    fs::{self as std_fs, File, OpenOptions},
    io::{self, BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    str::FromStr,
    time::{Duration, Instant as StdInstant, SystemTime},
};

#[cfg(unix)]
use std::os::unix::{ffi::OsStrExt, fs::MetadataExt};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, NaiveDate, TimeDelta, Utc};
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::{Field, Row},
};
use rust_decimal::{Decimal, RoundingStrategy};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncWriteExt},
    sync::mpsc,
    task::JoinHandle,
    time::{sleep, timeout},
};
use uuid::Uuid;

use super::{
    binance_archive::ArchiveCancellation, job::BinanceL2OneSecondFeature,
    repository::IngestionRepository,
};

pub const CRYPTOHFT_ARCHIVE_PROVIDER: &str = "cryptohftdata";
pub const DEFAULT_CRYPTOHFT_BASE_URL: &str = "https://api.cryptohftdata.com";
pub const CRYPTOHFT_SYMBOL: &str = "BTCUSDT";
pub const DEFAULT_AVAILABILITY_OFFSET_MS: u64 = 100;
pub const DEFAULT_MAX_STALE_MS: u64 = 1_000;
pub const DEFAULT_MINIMUM_FREE_FRACTION: f64 = 0.25;
pub const DEFAULT_MINIMUM_FREE_BYTES: u64 = 64 * 1024 * 1024 * 1024;
pub const DEFAULT_MINIMUM_WRITE_BYTES_PER_SECOND: u64 = 20 * 1024 * 1024;
pub const DEFAULT_REQUEST_MINIMUM_INTERVAL: Duration = Duration::from_millis(1_100);
pub const DEFAULT_MAXIMUM_COMPRESSED_BYTES: u64 = 1024 * 1024 * 1024;
pub const DEFAULT_MAXIMUM_DECODED_BYTES: u64 = 8 * 1024 * 1024 * 1024;
pub const DEFAULT_DOWNLOAD_CHUNK_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
pub const BINANCE_L2_FEATURE_SCHEMA_VERSION: &str = "binance-btcusdt-l2-one-second-features-v1";
pub const BINANCE_L2_MATERIALIZATION_CONTRACT: &str =
    "cryptohft-binance-futures-btcusdt-l2-features-v1";
pub const CRYPTOHFT_EARLIEST_CONTEXT_HOUR_EPOCH: i64 = 1_776_121_200;
pub const MAX_CONTEXT_LOOKBACK_HOURS: usize = 2_617;
pub const CRYPTOHFT_AUDITED_ANCHOR_REMOTE_FILE: &str =
    "binance_futures/2026-04-13/23/BTCUSDT_orderbook.parquet.zst";
pub const CRYPTOHFT_AUDITED_ANCHOR_SHA256: &str =
    "9e5557a44fe0c414feb353ad64f22ea232bd5e895f01f32f17fb1bd8d36e5174";
pub const CRYPTOHFT_AUDITED_SNAPSHOT_EVENT_MILLIS: i64 = 1_776_121_519_649;
pub const CRYPTOHFT_AUDITED_SNAPSHOT_LAST_UPDATE_ID: i64 = 10_318_083_192_958;

const WRITE_PROBE_BYTES: usize = 16 * 1024 * 1024;
const IO_BUFFER_BYTES: usize = 1024 * 1024;
const MAX_CONFIGURED_DOWNLOAD_WORKERS: u64 = 6;
const STALE_WORK_FILE_AGE: Duration = Duration::from_secs(6 * 60 * 60);
const MAX_LOGICAL_EVENT_LEVELS: usize = 10_000;
const MAX_BOOK_LEVELS_PER_SIDE: usize = 100_000;
const MIN_FULL_DEPTH_SNAPSHOT_LEVELS: usize = 1_000;
const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
const STORAGE_SENTINEL_FILE: &str = ".cryptohft-l2-storage";
const STORAGE_SENTINEL_CONTENT: &str = "cryptohft-btcusdt-l2-archive-v1\n";

#[derive(Debug, Clone)]
pub struct CryptoHftBinanceL2Config {
    pub base_url: String,
    pub archive_root: PathBuf,
    pub temporary_directory: PathBuf,
    pub availability_offset_ms: u64,
    pub max_stale_ms: u64,
    pub minimum_free_fraction: f64,
    pub minimum_free_bytes: u64,
    pub minimum_write_bytes_per_second: u64,
    pub request_minimum_interval: Duration,
    pub maximum_compressed_bytes: u64,
    pub maximum_decoded_bytes: u64,
    pub download_chunk_idle_timeout: Duration,
}

impl CryptoHftBinanceL2Config {
    pub fn new(archive_root: PathBuf, temporary_directory: PathBuf) -> Self {
        Self {
            base_url: DEFAULT_CRYPTOHFT_BASE_URL.to_owned(),
            archive_root,
            temporary_directory,
            availability_offset_ms: DEFAULT_AVAILABILITY_OFFSET_MS,
            max_stale_ms: DEFAULT_MAX_STALE_MS,
            minimum_free_fraction: DEFAULT_MINIMUM_FREE_FRACTION,
            minimum_free_bytes: DEFAULT_MINIMUM_FREE_BYTES,
            minimum_write_bytes_per_second: DEFAULT_MINIMUM_WRITE_BYTES_PER_SECOND,
            request_minimum_interval: DEFAULT_REQUEST_MINIMUM_INTERVAL,
            maximum_compressed_bytes: DEFAULT_MAXIMUM_COMPRESSED_BYTES,
            maximum_decoded_bytes: DEFAULT_MAXIMUM_DECODED_BYTES,
            download_chunk_idle_timeout: DEFAULT_DOWNLOAD_CHUNK_IDLE_TIMEOUT,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !self.archive_root.is_absolute() {
            bail!("CryptoHFT archive root must be absolute");
        }
        if !self.temporary_directory.is_absolute() {
            bail!("CryptoHFT temporary directory must be absolute");
        }
        if !self.base_url.starts_with("https://") {
            bail!("CryptoHFT base URL must use HTTPS");
        }
        if self.base_url.contains('?') || self.base_url.contains('#') {
            bail!("CryptoHFT base URL must not contain a query or fragment");
        }
        if !self.minimum_free_fraction.is_finite()
            || !(DEFAULT_MINIMUM_FREE_FRACTION..1.0).contains(&self.minimum_free_fraction)
        {
            bail!("CryptoHFT minimum free fraction must be in [0.25, 1.0)");
        }
        if self.minimum_free_bytes == 0
            || self.minimum_write_bytes_per_second == 0
            || self.maximum_compressed_bytes == 0
            || self.maximum_decoded_bytes == 0
            || self.download_chunk_idle_timeout.is_zero()
        {
            bail!("CryptoHFT storage and download limits must be positive");
        }
        if self.maximum_decoded_bytes < self.maximum_compressed_bytes {
            bail!("CryptoHFT decoded limit must not be smaller than compressed limit");
        }
        if self.availability_offset_ms != DEFAULT_AVAILABILITY_OFFSET_MS
            || self.max_stale_ms != DEFAULT_MAX_STALE_MS
        {
            bail!("CryptoHFT causal feature contract parameters are immutable");
        }
        if self.request_minimum_interval != DEFAULT_REQUEST_MINIMUM_INTERVAL {
            bail!("CryptoHFT request interval is part of the immutable source contract");
        }
        Ok(())
    }
}

pub fn day_artifact_logical_key(date: NaiveDate) -> String {
    format!(
        "cryptohftdata:binance-futures:{CRYPTOHFT_SYMBOL}:l2-day:{BINANCE_L2_MATERIALIZATION_CONTRACT}:{date}"
    )
}

pub fn validate_representative_day_quality(metadata: &serde_json::Value) -> Result<()> {
    if metadata
        .get("materialization_contract")
        .and_then(serde_json::Value::as_str)
        != Some(BINANCE_L2_MATERIALIZATION_CONTRACT)
        || metadata
            .get("source_objects")
            .and_then(serde_json::Value::as_u64)
            != Some(24)
    {
        bail!("Binance L2 representative artifact contract metadata was invalid");
    }
    let manifest = metadata
        .get("hourly_manifest")
        .and_then(serde_json::Value::as_object)
        .context("Binance L2 representative artifact omitted its hourly manifest")?;
    let context_objects = manifest
        .get("context")
        .and_then(serde_json::Value::as_array)
        .filter(|objects| !objects.is_empty() && objects.len() <= MAX_CONTEXT_LOOKBACK_HOURS)
        .context("Binance L2 representative manifest omitted its bounded context objects")?;
    let target_objects = manifest
        .get("target")
        .and_then(serde_json::Value::as_array)
        .filter(|objects| objects.len() == 24)
        .context("Binance L2 representative manifest did not contain 24 target objects")?;
    let mut context_has_validated_snapshot = false;
    let mut representative_has_pinned_anchor = false;
    for (is_context, object) in context_objects
        .iter()
        .map(|object| (true, object))
        .chain(target_objects.iter().map(|object| (false, object)))
    {
        let object = object
            .as_object()
            .context("Binance L2 representative manifest object was invalid")?;
        let checksum = object
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if object.get("provider").and_then(serde_json::Value::as_str)
            != Some(CRYPTOHFT_ARCHIVE_PROVIDER)
            || checksum.len() != 64
            || !checksum
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            || object
                .get("compressed_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default()
                == 0
            || object
                .get("raw_rows")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default()
                == 0
            || object
                .get("update_events")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_default()
                == 0
        {
            bail!("Binance L2 representative manifest object failed integrity validation");
        }
        let snapshots = object
            .get("snapshot_events")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default();
        let validated_snapshots = object
            .get("validated_snapshot_events")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default();
        if validated_snapshots > snapshots {
            bail!("Binance L2 representative manifest snapshot metrics were invalid");
        }
        context_has_validated_snapshot |= is_context && validated_snapshots > 0;
        representative_has_pinned_anchor |= is_context
            && object
                .get("remote_file")
                .and_then(serde_json::Value::as_str)
                == Some(CRYPTOHFT_AUDITED_ANCHOR_REMOTE_FILE)
            && checksum == CRYPTOHFT_AUDITED_ANCHOR_SHA256
            && object
                .get("audited_anchor_snapshot_verified")
                .and_then(serde_json::Value::as_bool)
                == Some(true);
    }
    if !context_has_validated_snapshot {
        bail!("Binance L2 representative context omitted a validated full-depth snapshot");
    }
    if !representative_has_pinned_anchor {
        bail!("Binance L2 representative context omitted the pinned audited anchor");
    }
    if !context_objects.iter().any(|object| {
        object
            .get("snapshot_events")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|count| count > 0)
    }) {
        bail!("Binance L2 representative context did not contain a snapshot");
    }

    let qualified_seconds = metadata
        .get("qualified_seconds")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default();
    let unavailable_seconds = metadata
        .get("unavailable_seconds")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let sequence_gaps = metadata
        .get("sequence_gaps")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let invalid_book_events = metadata
        .get("invalid_book_events")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let snapshots = metadata
        .get("snapshots")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default();
    let no_snapshot_bootstrap = metadata
        .get("no_snapshot_bootstrap")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    if no_snapshot_bootstrap
        || snapshots == 0
        || sequence_gaps == u64::MAX
        || invalid_book_events != 0
        || qualified_seconds == 0
        || qualified_seconds.checked_add(unavailable_seconds) != Some(86_400)
    {
        bail!(
            "Binance L2 representative source audit was not causally classified: qualified={qualified_seconds}, unavailable={unavailable_seconds}, snapshots={snapshots}, sequence_gaps={sequence_gaps}, invalid_book_events={invalid_book_events}, no_snapshot_bootstrap={no_snapshot_bootstrap}"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArchiveStoragePreflight {
    pub canonical_archive_root: PathBuf,
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub available_fraction: f64,
    pub measured_write_bytes_per_second: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CryptoHftHourlySpec {
    pub date: NaiveDate,
    pub hour: u8,
    pub remote_file: String,
    pub source_uri: String,
    pub logical_key: String,
    pub relative_path: PathBuf,
    pub archive_path: PathBuf,
}

impl CryptoHftHourlySpec {
    pub fn new(config: &CryptoHftBinanceL2Config, date: NaiveDate, hour: u8) -> Result<Self> {
        config.validate()?;
        if hour > 23 {
            bail!("CryptoHFT archive hour must be between 0 and 23");
        }
        let date_text = date.format("%Y-%m-%d");
        let remote_file = format!(
            "binance_futures/{date_text}/{hour:02}/{CRYPTOHFT_SYMBOL}_orderbook.parquet.zst"
        );
        let relative_path = PathBuf::from(&remote_file);
        Ok(Self {
            date,
            hour,
            source_uri: format!(
                "{}/download?file={remote_file}",
                config.base_url.trim_end_matches('/')
            ),
            logical_key: format!(
                "cryptohftdata:binance-futures:{CRYPTOHFT_SYMBOL}:l2:{date_text}T{hour:02}"
            ),
            archive_path: config.archive_root.join(&relative_path),
            remote_file,
            relative_path,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HourlyArchiveManifest {
    pub provider: String,
    pub source_uri: String,
    pub logical_key: String,
    pub remote_file: String,
    pub archive_path: PathBuf,
    pub sha256: String,
    pub compressed_bytes: u64,
    pub raw_rows: u64,
    pub snapshot_events: u64,
    pub validated_snapshot_events: u64,
    pub audited_anchor_snapshot_verified: bool,
    pub update_events: u64,
    pub minimum_provider_received_at: DateTime<Utc>,
    pub maximum_provider_received_at: DateTime<Utc>,
    pub receipt_rows_outside_exact_hour: u64,
    pub downloaded_at: DateTime<Utc>,
    pub reused_archive: bool,
    pub vendor_checksum: Option<String>,
    pub integrity_basis: String,
}

impl HourlyArchiveManifest {
    pub fn lineage_metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "provider": self.provider,
            "remote_file": self.remote_file,
            "archive_path": self.archive_path,
            "sha256": self.sha256,
            "compressed_bytes": self.compressed_bytes,
            "raw_rows": self.raw_rows,
            "snapshot_events": self.snapshot_events,
            "validated_snapshot_events": self.validated_snapshot_events,
            "audited_anchor_snapshot_verified": self.audited_anchor_snapshot_verified,
            "update_events": self.update_events,
            "minimum_provider_received_at": self.minimum_provider_received_at,
            "maximum_provider_received_at": self.maximum_provider_received_at,
            "receipt_rows_outside_exact_hour": self.receipt_rows_outside_exact_hour,
            "vendor_checksum": self.vendor_checksum,
            "integrity_basis": self.integrity_basis,
            "reused_archive": self.reused_archive,
        })
    }
}

impl CryptoHftBinanceL2Config {
    pub async fn preflight(&self) -> Result<ArchiveStoragePreflight> {
        self.validate()?;
        let config = self.clone();
        tokio::task::spawn_blocking(move || preflight_blocking(&config))
            .await
            .context("CryptoHFT storage preflight task failed")?
    }
}

fn preflight_blocking(config: &CryptoHftBinanceL2Config) -> Result<ArchiveStoragePreflight> {
    if !config.archive_root.is_dir() {
        bail!(
            "CryptoHFT archive root {} must already exist; refusing to create a path when the SSD may be unmounted",
            config.archive_root.display()
        );
    }
    let sentinel_path = config.archive_root.join(STORAGE_SENTINEL_FILE);
    let sentinel = std_fs::read_to_string(&sentinel_path).with_context(|| {
        format!(
            "CryptoHFT SSD sentinel {} is missing; refusing to write when mount identity is unproven",
            sentinel_path.display()
        )
    })?;
    if sentinel != STORAGE_SENTINEL_CONTENT {
        bail!(
            "CryptoHFT SSD sentinel {} did not match the storage contract",
            sentinel_path.display()
        );
    }
    let canonical_archive_root = config
        .archive_root
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", config.archive_root.display()))?;
    if canonical_archive_root == Path::new("/") {
        bail!("CryptoHFT archive root must not be the filesystem root");
    }

    #[cfg(unix)]
    {
        let archive_device = std_fs::metadata(&canonical_archive_root)?.dev();
        let root_device = std_fs::metadata("/")?.dev();
        if archive_device == root_device {
            bail!(
                "CryptoHFT archive root {} is not on a separately mounted filesystem",
                canonical_archive_root.display()
            );
        }
    }

    let (total_bytes, available_bytes) = filesystem_capacity(&canonical_archive_root)?;
    if total_bytes == 0 {
        bail!("CryptoHFT archive filesystem reported zero capacity");
    }
    let available_fraction = available_bytes as f64 / total_bytes as f64;
    validate_archive_headroom(config, total_bytes, available_bytes)?;

    let measured_write_bytes_per_second = measure_synchronous_write(&canonical_archive_root)?;
    if measured_write_bytes_per_second < config.minimum_write_bytes_per_second {
        bail!(
            "CryptoHFT SSD write probe measured {} bytes/s; {} bytes/s required",
            measured_write_bytes_per_second,
            config.minimum_write_bytes_per_second
        );
    }
    std_fs::create_dir_all(&config.temporary_directory).with_context(|| {
        format!(
            "failed to create CryptoHFT temporary directory {}",
            config.temporary_directory.display()
        )
    })?;
    cleanup_stale_work_files(&config.temporary_directory, true)?;

    Ok(ArchiveStoragePreflight {
        canonical_archive_root,
        total_bytes,
        available_bytes,
        available_fraction,
        measured_write_bytes_per_second,
    })
}

fn validate_archive_headroom(
    config: &CryptoHftBinanceL2Config,
    total_bytes: u64,
    available_bytes: u64,
) -> Result<()> {
    let in_flight_reservation = config
        .maximum_compressed_bytes
        .checked_mul(MAX_CONFIGURED_DOWNLOAD_WORKERS)
        .context("CryptoHFT in-flight storage reservation overflow")?;
    let available_after_reservation = available_bytes.saturating_sub(in_flight_reservation);
    if available_after_reservation < config.minimum_free_bytes {
        bail!(
            "CryptoHFT archive filesystem has {available_bytes} available bytes; {} bytes plus {in_flight_reservation} bytes of in-flight headroom required",
            config.minimum_free_bytes
        );
    }
    let fraction_after_reservation = available_after_reservation as f64 / total_bytes as f64;
    if fraction_after_reservation < config.minimum_free_fraction {
        bail!(
            "CryptoHFT archive filesystem would be {:.2}% free after bounded in-flight downloads; {:.2}% required",
            fraction_after_reservation * 100.0,
            config.minimum_free_fraction * 100.0
        );
    }
    Ok(())
}

async fn ensure_archive_headroom(config: &CryptoHftBinanceL2Config) -> Result<()> {
    let config = config.clone();
    tokio::task::spawn_blocking(move || {
        let (total_bytes, available_bytes) = filesystem_capacity(&config.archive_root)?;
        if total_bytes == 0 {
            bail!("CryptoHFT archive filesystem reported zero capacity");
        }
        validate_archive_headroom(&config, total_bytes, available_bytes)
    })
    .await
    .context("CryptoHFT archive headroom task failed")?
}

fn cleanup_stale_work_files(directory: &Path, include_parquet: bool) -> Result<()> {
    let now = SystemTime::now();
    for entry in std_fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if !metadata.is_file() {
            continue;
        }
        let path = entry.path();
        let stale = metadata
            .modified()
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age >= STALE_WORK_FILE_AGE);
        let is_partial = path
            .extension()
            .is_some_and(|extension| extension == "part");
        let is_parquet = include_parquet
            && path
                .extension()
                .is_some_and(|extension| extension == "parquet");
        if stale && (is_partial || is_parquet) {
            std_fs::remove_file(&path)
                .with_context(|| format!("failed to remove stale work file {}", path.display()))?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn filesystem_capacity(path: &Path) -> Result<(u64, u64)> {
    let bytes = path.as_os_str().as_bytes();
    let path = CString::new(bytes).context("CryptoHFT archive path contained a NUL byte")?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `path` is a live NUL-terminated string and `stats` points to writable storage.
    let result = unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) };
    if result != 0 {
        return Err(io::Error::last_os_error())
            .context("statvfs failed for CryptoHFT archive root");
    }
    // SAFETY: a successful statvfs call initialized the entire structure.
    let stats = unsafe { stats.assume_init() };
    let fragment_size = stats.f_frsize as u64;
    Ok((
        (stats.f_blocks as u64).saturating_mul(fragment_size),
        (stats.f_bavail as u64).saturating_mul(fragment_size),
    ))
}

#[cfg(not(unix))]
fn filesystem_capacity(_path: &Path) -> Result<(u64, u64)> {
    bail!("CryptoHFT archive capacity preflight is unsupported on this platform")
}

fn measure_synchronous_write(root: &Path) -> Result<u64> {
    let probe_path = root.join(format!(".cryptohft-write-probe-{}.part", Uuid::new_v4()));
    let _cleanup = RemoveOnDrop(probe_path.clone());
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe_path)
        .with_context(|| format!("CryptoHFT archive root {} is not writable", root.display()))?;
    let block = vec![0x5au8; IO_BUFFER_BYTES];
    let started = StdInstant::now();
    for _ in 0..(WRITE_PROBE_BYTES / IO_BUFFER_BYTES) {
        output.write_all(&block)?;
    }
    output.sync_all()?;
    let elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
    drop(output);
    let bytes_per_second = (WRITE_PROBE_BYTES as f64 / elapsed).floor();
    Ok(bytes_per_second.min(u64::MAX as f64) as u64)
}

struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std_fs::remove_file(&self.0);
    }
}

pub async fn download_hour(
    client: &reqwest::Client,
    repository: &IngestionRepository,
    config: &CryptoHftBinanceL2Config,
    spec: &CryptoHftHourlySpec,
    cancellation: &ArchiveCancellation,
) -> Result<HourlyArchiveManifest> {
    config.validate()?;
    check_cancelled(cancellation)?;
    validate_spec_paths(config, spec)?;

    if fs::try_exists(&spec.archive_path).await? {
        match reuse_hour(config, spec, cancellation).await? {
            CachedHourReuse::Reused(manifest) => return Ok(manifest),
            CachedHourReuse::ProvenCorruption(error) => {
                check_cancelled(cancellation)?;
                quarantine_cached_hour(spec, &error)
                    .await
                    .with_context(|| {
                        format!("failed to quarantine invalid CryptoHFT cache entry: {error:#}")
                    })?;
            }
        }
    }

    let parent = spec
        .archive_path
        .parent()
        .context("CryptoHFT archive path had no parent")?;
    fs::create_dir_all(parent)
        .await
        .with_context(|| format!("failed to create {}", parent.display()))?;
    let cleanup_parent = parent.to_path_buf();
    tokio::task::spawn_blocking(move || cleanup_stale_work_files(&cleanup_parent, false))
        .await
        .context("CryptoHFT stale-partial cleanup task failed")??;
    ensure_archive_headroom(config).await?;
    wait_for_request_slot(repository, cancellation).await?;

    let request = client.get(&spec.source_uri).send();
    let mut response = timeout(config.download_chunk_idle_timeout, request)
        .await
        .context("CryptoHFT archive request timed out")?
        .with_context(|| format!("failed to request {}", spec.source_uri))?
        .error_for_status()
        .with_context(|| format!("CryptoHFT rejected {}", spec.source_uri))?;
    let expected_content_length = response.content_length();
    if expected_content_length.is_some_and(|bytes| bytes > config.maximum_compressed_bytes) {
        bail!("CryptoHFT archive exceeded the compressed size limit");
    }

    let partial_path = parent.join(format!(
        ".{}.{}.part",
        spec.archive_path
            .file_name()
            .and_then(|name| name.to_str())
            .context("CryptoHFT archive filename was not UTF-8")?,
        Uuid::new_v4()
    ));
    let _partial_cleanup = RemoveOnDrop(partial_path.clone());
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial_path)
        .await
        .with_context(|| format!("failed to create {}", partial_path.display()))?;
    let mut hasher = Sha256::new();
    let mut compressed_bytes = 0u64;
    let mut leading_bytes = Vec::with_capacity(4);
    loop {
        check_cancelled(cancellation)?;
        let chunk = timeout(config.download_chunk_idle_timeout, response.chunk())
            .await
            .context("CryptoHFT archive download stalled")??;
        let Some(chunk) = chunk else { break };
        compressed_bytes = compressed_bytes
            .checked_add(u64::try_from(chunk.len()).context("archive chunk length overflow")?)
            .context("CryptoHFT archive length overflow")?;
        if compressed_bytes > config.maximum_compressed_bytes {
            bail!("CryptoHFT archive exceeded the compressed size limit");
        }
        if leading_bytes.len() < 4 {
            let required = 4usize.saturating_sub(leading_bytes.len());
            leading_bytes.extend_from_slice(&chunk[..required.min(chunk.len())]);
        }
        hasher.update(&chunk);
        output.write_all(&chunk).await?;
    }
    if compressed_bytes == 0 {
        bail!("CryptoHFT returned an empty archive");
    }
    if leading_bytes != [0x28, 0xb5, 0x2f, 0xfd] {
        bail!("CryptoHFT response did not have a Zstandard frame header");
    }
    validate_downloaded_content_length(expected_content_length, compressed_bytes)?;
    output.sync_all().await?;
    drop(output);
    check_cancelled(cancellation)?;
    let payload = validate_archive_payload(config, spec, &partial_path, cancellation).await?;

    let sha256 = format!("{:x}", hasher.finalize());
    if spec.remote_file == CRYPTOHFT_AUDITED_ANCHOR_REMOTE_FILE
        && sha256 != CRYPTOHFT_AUDITED_ANCHOR_SHA256
    {
        bail!("CryptoHFT audited anchor archive checksum did not match the pinned source object");
    }
    let installed = match fs::hard_link(&partial_path, &spec.archive_path).await {
        Ok(()) => true,
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => false,
        Err(error) => return Err(error).context("failed to atomically finalize CryptoHFT archive"),
    };
    if !installed {
        let winner = hash_file_bounded(
            &spec.archive_path,
            config.maximum_compressed_bytes,
            cancellation,
        )
        .await?;
        if winner.sha256 != sha256 || winner.bytes != compressed_bytes {
            bail!("concurrent CryptoHFT archive finalization produced a different object");
        }
    }
    make_read_only(&spec.archive_path).await?;
    sync_parent_directory(parent)?;

    let manifest = HourlyArchiveManifest {
        provider: CRYPTOHFT_ARCHIVE_PROVIDER.to_owned(),
        source_uri: spec.source_uri.clone(),
        logical_key: spec.logical_key.clone(),
        remote_file: spec.remote_file.clone(),
        archive_path: spec.archive_path.clone(),
        sha256,
        compressed_bytes,
        raw_rows: payload.raw_rows,
        snapshot_events: payload.snapshot_events,
        validated_snapshot_events: payload.validated_snapshot_events,
        audited_anchor_snapshot_verified: payload.audited_anchor_snapshot_verified,
        update_events: payload.update_events,
        minimum_provider_received_at: payload.minimum_provider_received_at,
        maximum_provider_received_at: payload.maximum_provider_received_at,
        receipt_rows_outside_exact_hour: payload.receipt_rows_outside_exact_hour,
        downloaded_at: Utc::now(),
        reused_archive: !installed,
        vendor_checksum: None,
        integrity_basis: "locally_computed_sha256_no_vendor_checksum".to_owned(),
    };
    persist_manifest(spec, &manifest).await?;
    Ok(manifest)
}

fn validate_downloaded_content_length(expected: Option<u64>, actual: u64) -> Result<()> {
    if expected.is_some_and(|expected| expected != actual) {
        bail!("CryptoHFT response length did not match Content-Length");
    }
    Ok(())
}

async fn quarantine_cached_hour(spec: &CryptoHftHourlySpec, cause: &anyhow::Error) -> Result<()> {
    let parent = spec
        .archive_path
        .parent()
        .context("CryptoHFT archive path had no parent")?;
    let token = Uuid::new_v4();
    let archive_name = spec
        .archive_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("CryptoHFT archive filename was not UTF-8")?;
    let quarantined_archive = parent.join(format!(".{archive_name}.quarantine.{token}"));
    if fs::try_exists(&spec.archive_path).await? {
        match fs::rename(&spec.archive_path, &quarantined_archive).await {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("failed to quarantine CryptoHFT archive"),
        }
    }
    let local_manifest = manifest_path(spec)?;
    if fs::try_exists(&local_manifest).await? {
        let manifest_name = local_manifest
            .file_name()
            .and_then(|name| name.to_str())
            .context("CryptoHFT manifest filename was not UTF-8")?;
        let quarantined_manifest = parent.join(format!(".{manifest_name}.quarantine.{token}"));
        match fs::rename(&local_manifest, quarantined_manifest).await {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("failed to quarantine CryptoHFT manifest"),
        }
    }
    let reason_path = parent.join(format!(".{archive_name}.quarantine.{token}.reason.json"));
    let reason = serde_json::to_vec_pretty(&serde_json::json!({
        "provider": CRYPTOHFT_ARCHIVE_PROVIDER,
        "remote_file": spec.remote_file,
        "quarantined_at": Utc::now(),
        "error": cause.to_string(),
    }))?;
    fs::write(&reason_path, reason).await?;
    make_read_only(&reason_path).await?;
    sync_parent_directory(parent)?;
    Ok(())
}

#[derive(Debug)]
enum CachedHourReuse {
    Reused(HourlyArchiveManifest),
    ProvenCorruption(anyhow::Error),
}

async fn reuse_hour(
    config: &CryptoHftBinanceL2Config,
    spec: &CryptoHftHourlySpec,
    cancellation: &ArchiveCancellation,
) -> Result<CachedHourReuse> {
    let archive_metadata = fs::metadata(&spec.archive_path).await?;
    if !archive_metadata.is_file()
        || archive_metadata.len() == 0
        || archive_metadata.len() > config.maximum_compressed_bytes
    {
        return Ok(CachedHourReuse::ProvenCorruption(anyhow::anyhow!(
            "CryptoHFT cached archive had an invalid file type or compressed size"
        )));
    }
    let digest = hash_file_bounded(
        &spec.archive_path,
        config.maximum_compressed_bytes,
        cancellation,
    )
    .await?;
    let manifest_path = manifest_path(spec)?;
    let mut manifest = if fs::try_exists(&manifest_path).await? {
        let manifest_metadata = fs::metadata(&manifest_path).await?;
        if !manifest_metadata.is_file()
            || manifest_metadata.len() == 0
            || manifest_metadata.len() > MAX_MANIFEST_BYTES
        {
            return Ok(CachedHourReuse::ProvenCorruption(anyhow::anyhow!(
                "CryptoHFT local manifest had an invalid file type or size"
            )));
        }
        let bytes = fs::read(&manifest_path).await?;
        let manifest: HourlyArchiveManifest = match serde_json::from_slice(&bytes) {
            Ok(manifest) => manifest,
            Err(error) => {
                return Ok(CachedHourReuse::ProvenCorruption(
                    anyhow::Error::new(error).context("CryptoHFT local manifest was invalid JSON"),
                ));
            }
        };
        if let Some(error) = cached_manifest_corruption(spec, &digest, &manifest) {
            return Ok(CachedHourReuse::ProvenCorruption(error));
        }
        make_read_only(&manifest_path).await?;
        manifest
    } else {
        let payload =
            validate_archive_payload(config, spec, &spec.archive_path, cancellation).await?;
        let recovered = HourlyArchiveManifest {
            provider: CRYPTOHFT_ARCHIVE_PROVIDER.to_owned(),
            source_uri: spec.source_uri.clone(),
            logical_key: spec.logical_key.clone(),
            remote_file: spec.remote_file.clone(),
            archive_path: spec.archive_path.clone(),
            sha256: digest.sha256,
            compressed_bytes: digest.bytes,
            raw_rows: payload.raw_rows,
            snapshot_events: payload.snapshot_events,
            validated_snapshot_events: payload.validated_snapshot_events,
            audited_anchor_snapshot_verified: payload.audited_anchor_snapshot_verified,
            update_events: payload.update_events,
            minimum_provider_received_at: payload.minimum_provider_received_at,
            maximum_provider_received_at: payload.maximum_provider_received_at,
            receipt_rows_outside_exact_hour: payload.receipt_rows_outside_exact_hour,
            downloaded_at: Utc::now(),
            reused_archive: false,
            vendor_checksum: None,
            integrity_basis: "locally_computed_sha256_no_vendor_checksum".to_owned(),
        };
        persist_manifest(spec, &recovered).await?;
        recovered
    };
    make_read_only(&spec.archive_path).await?;
    manifest.reused_archive = true;
    Ok(CachedHourReuse::Reused(manifest))
}

fn cached_manifest_corruption(
    spec: &CryptoHftHourlySpec,
    digest: &FileDigest,
    manifest: &HourlyArchiveManifest,
) -> Option<anyhow::Error> {
    let invalid = manifest.provider != CRYPTOHFT_ARCHIVE_PROVIDER
        || manifest.source_uri != spec.source_uri
        || manifest.logical_key != spec.logical_key
        || manifest.remote_file != spec.remote_file
        || manifest.archive_path != spec.archive_path
        || manifest.sha256 != digest.sha256
        || manifest.compressed_bytes != digest.bytes
        || manifest.raw_rows == 0
        || manifest.update_events == 0
        || manifest.validated_snapshot_events > manifest.snapshot_events
        || (spec.remote_file == CRYPTOHFT_AUDITED_ANCHOR_REMOTE_FILE
            && (manifest.sha256 != CRYPTOHFT_AUDITED_ANCHOR_SHA256
                || !manifest.audited_anchor_snapshot_verified))
        || (spec.remote_file != CRYPTOHFT_AUDITED_ANCHOR_REMOTE_FILE
            && manifest.audited_anchor_snapshot_verified)
        || manifest.minimum_provider_received_at > manifest.maximum_provider_received_at
        || manifest
            .receipt_rows_outside_exact_hour
            .saturating_mul(1_000)
            > manifest.raw_rows
        || manifest.vendor_checksum.is_some()
        || manifest.integrity_basis != "locally_computed_sha256_no_vendor_checksum";
    invalid.then(|| anyhow::anyhow!("CryptoHFT immutable archive did not match its local manifest"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileDigest {
    sha256: String,
    bytes: u64,
}

async fn hash_file_bounded(
    path: &Path,
    maximum_bytes: u64,
    cancellation: &ArchiveCancellation,
) -> Result<FileDigest> {
    let mut input = fs::File::open(path).await?;
    let mut buffer = vec![0u8; IO_BUFFER_BYTES];
    let mut hasher = Sha256::new();
    let mut bytes = 0u64;
    loop {
        check_cancelled(cancellation)?;
        let read = input.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(u64::try_from(read).context("archive read length overflow")?)
            .context("CryptoHFT archive length overflow")?;
        if bytes > maximum_bytes {
            bail!("CryptoHFT cached archive exceeded the compressed size limit");
        }
        hasher.update(&buffer[..read]);
    }
    if bytes == 0 {
        bail!("CryptoHFT cached archive was empty");
    }
    Ok(FileDigest {
        sha256: format!("{:x}", hasher.finalize()),
        bytes,
    })
}

fn validate_spec_paths(
    config: &CryptoHftBinanceL2Config,
    spec: &CryptoHftHourlySpec,
) -> Result<()> {
    if spec.archive_path != config.archive_root.join(&spec.relative_path)
        || spec.relative_path.is_absolute()
        || spec
            .relative_path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        bail!("CryptoHFT archive spec escaped the configured archive root");
    }
    Ok(())
}

fn check_cancelled(cancellation: &ArchiveCancellation) -> Result<()> {
    if cancellation.is_cancelled() {
        bail!("CryptoHFT archive operation was cancelled");
    }
    Ok(())
}

async fn wait_for_request_slot(
    repository: &IngestionRepository,
    cancellation: &ArchiveCancellation,
) -> Result<()> {
    let scheduled_at = repository.reserve_cryptohft_request_slot().await?;
    loop {
        check_cancelled(cancellation)?;
        let Ok(wait) = (scheduled_at - Utc::now()).to_std() else {
            return Ok(());
        };
        if wait.is_zero() {
            return Ok(());
        }
        sleep(wait.min(Duration::from_millis(250))).await;
    }
}

fn manifest_path(spec: &CryptoHftHourlySpec) -> Result<PathBuf> {
    let file_name = spec
        .archive_path
        .file_name()
        .and_then(|value| value.to_str())
        .context("CryptoHFT archive filename was not UTF-8")?;
    Ok(spec
        .archive_path
        .with_file_name(format!("{file_name}.manifest.json")))
}

async fn persist_manifest(
    spec: &CryptoHftHourlySpec,
    manifest: &HourlyArchiveManifest,
) -> Result<()> {
    let final_path = manifest_path(spec)?;
    if fs::try_exists(&final_path).await? {
        validate_persisted_manifest(&final_path, manifest).await?;
        make_read_only(&final_path).await?;
        return Ok(());
    }
    let parent = final_path
        .parent()
        .context("CryptoHFT manifest path had no parent")?;
    let file_name = final_path
        .file_name()
        .and_then(|value| value.to_str())
        .context("CryptoHFT manifest filename was not UTF-8")?;
    let partial_path = parent.join(format!(".{file_name}.{}.part", Uuid::new_v4()));
    let _cleanup = RemoveOnDrop(partial_path.clone());
    let bytes = serde_json::to_vec_pretty(manifest)?;
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial_path)
        .await?;
    output.write_all(&bytes).await?;
    output.sync_all().await?;
    drop(output);
    match fs::hard_link(&partial_path, &final_path).await {
        Ok(()) => {
            make_read_only(&final_path).await?;
            sync_parent_directory(parent)?;
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            validate_persisted_manifest(&final_path, manifest).await?;
            make_read_only(&final_path).await?;
        }
        Err(error) => return Err(error).context("failed to atomically finalize local manifest"),
    }
    Ok(())
}

async fn validate_persisted_manifest(path: &Path, expected: &HourlyArchiveManifest) -> Result<()> {
    let bytes = fs::read(path).await?;
    let actual: HourlyArchiveManifest =
        serde_json::from_slice(&bytes).context("CryptoHFT local manifest was invalid JSON")?;
    if actual.provider != expected.provider
        || actual.source_uri != expected.source_uri
        || actual.logical_key != expected.logical_key
        || actual.remote_file != expected.remote_file
        || actual.archive_path != expected.archive_path
        || actual.sha256 != expected.sha256
        || actual.compressed_bytes != expected.compressed_bytes
        || actual.raw_rows != expected.raw_rows
        || actual.snapshot_events != expected.snapshot_events
        || actual.validated_snapshot_events != expected.validated_snapshot_events
        || actual.audited_anchor_snapshot_verified != expected.audited_anchor_snapshot_verified
        || actual.update_events != expected.update_events
        || actual.minimum_provider_received_at != expected.minimum_provider_received_at
        || actual.maximum_provider_received_at != expected.maximum_provider_received_at
        || actual.receipt_rows_outside_exact_hour != expected.receipt_rows_outside_exact_hour
        || actual.vendor_checksum != expected.vendor_checksum
        || actual.integrity_basis != expected.integrity_basis
    {
        bail!("CryptoHFT local manifests disagreed for an immutable archive");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadOnlyDisposition {
    AlreadyReadOnly,
    Updated,
}

async fn make_read_only(path: &Path) -> Result<ReadOnlyDisposition> {
    let mut permissions = fs::metadata(path).await?.permissions();
    if permissions.readonly() {
        return Ok(ReadOnlyDisposition::AlreadyReadOnly);
    }
    permissions.set_readonly(true);
    match fs::set_permissions(path, permissions).await {
        Ok(()) => Ok(ReadOnlyDisposition::Updated),
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {
            let current = fs::metadata(path).await.with_context(|| {
                format!(
                    "failed to verify permissions for immutable CryptoHFT file {}",
                    path.display()
                )
            })?;
            read_only_postcondition(path, error, &current.permissions())
        }
        Err(error) => Err(error)
            .with_context(|| format!("failed to make CryptoHFT file read-only {}", path.display())),
    }
}

fn read_only_postcondition(
    path: &Path,
    permission_error: io::Error,
    current_permissions: &std_fs::Permissions,
) -> Result<ReadOnlyDisposition> {
    if current_permissions.readonly() {
        Ok(ReadOnlyDisposition::AlreadyReadOnly)
    } else {
        Err(permission_error)
            .with_context(|| format!("failed to make CryptoHFT file read-only {}", path.display()))
    }
}

fn sync_parent_directory(parent: &Path) -> Result<()> {
    std_fs::File::open(parent)?.sync_all()?;
    Ok(())
}

#[derive(Debug)]
pub struct TemporaryParquetFile {
    path: PathBuf,
    pub decoded_bytes: u64,
}

impl TemporaryParquetFile {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryParquetFile {
    fn drop(&mut self) {
        let _ = std_fs::remove_file(&self.path);
    }
}

pub async fn decompress_hour_to_temporary_parquet(
    config: &CryptoHftBinanceL2Config,
    manifest: &HourlyArchiveManifest,
    cancellation: &ArchiveCancellation,
) -> Result<TemporaryParquetFile> {
    config.validate()?;
    check_cancelled(cancellation)?;
    if !manifest.archive_path.starts_with(&config.archive_root) {
        bail!("CryptoHFT manifest archive was outside the configured archive root");
    }
    let config = config.clone();
    let manifest = manifest.clone();
    let cancellation = cancellation.clone();
    tokio::task::spawn_blocking(move || decompress_hour_blocking(&config, &manifest, &cancellation))
        .await
        .context("CryptoHFT decompression task failed")?
}

async fn validate_archive_payload(
    config: &CryptoHftBinanceL2Config,
    spec: &CryptoHftHourlySpec,
    archive_path: &Path,
    cancellation: &ArchiveCancellation,
) -> Result<HourlyPayloadValidation> {
    let provisional_timestamp = Utc::now();
    let provisional = HourlyArchiveManifest {
        provider: CRYPTOHFT_ARCHIVE_PROVIDER.to_owned(),
        source_uri: spec.source_uri.clone(),
        logical_key: spec.logical_key.clone(),
        remote_file: spec.remote_file.clone(),
        archive_path: archive_path.to_path_buf(),
        sha256: String::new(),
        compressed_bytes: 0,
        raw_rows: 0,
        snapshot_events: 0,
        validated_snapshot_events: 0,
        audited_anchor_snapshot_verified: false,
        update_events: 0,
        minimum_provider_received_at: provisional_timestamp,
        maximum_provider_received_at: provisional_timestamp,
        receipt_rows_outside_exact_hour: 0,
        downloaded_at: provisional_timestamp,
        reused_archive: false,
        vendor_checksum: None,
        integrity_basis: "pre_finalization_payload_validation".to_owned(),
    };
    let parquet = decompress_hour_to_temporary_parquet(config, &provisional, cancellation).await?;
    let expected_date = spec.date;
    let expected_hour = spec.hour;
    let remote_file = spec.remote_file.clone();
    let cancellation = cancellation.clone();
    tokio::task::spawn_blocking(move || {
        validate_hourly_parquet_payload(parquet.path(), expected_date, expected_hour, &cancellation)
            .with_context(|| {
                format!("CryptoHFT source object {remote_file} failed payload validation")
            })
    })
    .await
    .context("CryptoHFT payload-validation task failed")?
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HourlyPayloadValidation {
    raw_rows: u64,
    snapshot_events: u64,
    validated_snapshot_events: u64,
    audited_anchor_snapshot_verified: bool,
    update_events: u64,
    minimum_provider_received_at: DateTime<Utc>,
    maximum_provider_received_at: DateTime<Utc>,
    receipt_rows_outside_exact_hour: u64,
}

fn validate_hourly_parquet_payload(
    path: &Path,
    expected_date: NaiveDate,
    expected_hour: u8,
    cancellation: &ArchiveCancellation,
) -> Result<HourlyPayloadValidation> {
    let expected_start = expected_date
        .and_hms_opt(u32::from(expected_hour), 0, 0)
        .context("CryptoHFT archive hour could not form a UTC timestamp")?
        .and_utc();
    let expected_end = expected_start + TimeDelta::hours(1);
    let is_audited_anchor_hour =
        expected_start.timestamp() == CRYPTOHFT_EARLIEST_CONTEXT_HOUR_EPOCH;
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)
        .context("failed to read validated CryptoHFT Parquet metadata")?;
    let has_order_count = validate_cryptohft_schema(&reader)?;
    let expected_rows = reader.metadata().file_metadata().num_rows();
    if expected_rows <= 0 {
        bail!("CryptoHFT Parquet archive contained no rows");
    }
    let tolerated_start = expected_start - TimeDelta::seconds(1);
    let tolerated_end = expected_end + TimeDelta::seconds(1);
    let mut decoded_rows = 0u64;
    let mut snapshot_events = 0u64;
    let mut validated_snapshot_events = 0u64;
    let mut audited_anchor_snapshot_verified = false;
    let mut update_events = 0u64;
    let mut outside_exact_hour = 0u64;
    let mut inside_exact_hour = 0u64;
    let mut minimum_received_at = None;
    let mut maximum_received_at = None;
    let mut last_event_key: Option<EventKey> = None;
    let mut pending_snapshot: Option<LogicalEvent> = None;
    for row_group_index in 0..reader.num_row_groups() {
        check_cancelled(cancellation)?;
        let row_group = reader.get_row_group(row_group_index)?;
        let rows = row_group.get_row_iter(None)?;
        for row in rows {
            check_cancelled(cancellation)?;
            let (key, level) = parse_cryptohft_row(row?, has_order_count)?;
            let received_at = timestamp_from_nanoseconds(key.received_time_ns)?;
            let event_at = DateTime::from_timestamp_millis(key.event_time_ms)
                .context("CryptoHFT event_time was outside the supported UTC range")?;
            validate_hourly_row_causal_time(received_at, event_at, tolerated_start, tolerated_end)?;
            minimum_received_at = Some(
                minimum_received_at
                    .map_or(received_at, |value: DateTime<Utc>| value.min(received_at)),
            );
            maximum_received_at = Some(
                maximum_received_at
                    .map_or(received_at, |value: DateTime<Utc>| value.max(received_at)),
            );
            if received_at < expected_start || received_at >= expected_end {
                outside_exact_hour = outside_exact_hour.saturating_add(1);
            } else {
                inside_exact_hour = inside_exact_hour.saturating_add(1);
            }
            match last_event_key.as_mut() {
                Some(previous) if previous.same_logical_event(&key) => {
                    previous.received_time_ns = previous.received_time_ns.max(key.received_time_ns);
                    if let Some(snapshot) = pending_snapshot.as_mut() {
                        snapshot.key.received_time_ns = previous.received_time_ns;
                        snapshot.push_level(level)?;
                    }
                }
                Some(_) => {
                    finish_hourly_validation_event(
                        last_event_key
                            .take()
                            .context("CryptoHFT validation event key disappeared")?,
                        pending_snapshot.take(),
                        &mut snapshot_events,
                        &mut validated_snapshot_events,
                        &mut audited_anchor_snapshot_verified,
                        &mut update_events,
                    )?;
                    pending_snapshot = matches!(key.event_type, EventType::Snapshot)
                        .then(|| LogicalEvent::new(key.clone(), level));
                    last_event_key = Some(key);
                }
                None => {
                    pending_snapshot = matches!(key.event_type, EventType::Snapshot)
                        .then(|| LogicalEvent::new(key.clone(), level));
                    last_event_key = Some(key);
                }
            }
            decoded_rows = decoded_rows
                .checked_add(1)
                .context("CryptoHFT validation row count overflow")?;
        }
    }
    if let Some(key) = last_event_key.take() {
        finish_hourly_validation_event(
            key,
            pending_snapshot.take(),
            &mut snapshot_events,
            &mut validated_snapshot_events,
            &mut audited_anchor_snapshot_verified,
            &mut update_events,
        )?;
    }
    if i64::try_from(decoded_rows).context("CryptoHFT row count exceeded i64")? != expected_rows {
        bail!("CryptoHFT decoded {decoded_rows} Parquet rows; metadata declared {expected_rows}");
    }
    validate_hourly_receipt_distribution(decoded_rows, inside_exact_hour, outside_exact_hour)?;
    if update_events == 0 {
        bail!("CryptoHFT archive contained no logical depth updates");
    }
    if is_audited_anchor_hour && !audited_anchor_snapshot_verified {
        bail!("CryptoHFT audited anchor archive omitted its pinned full-depth snapshot");
    }
    if !is_audited_anchor_hour {
        audited_anchor_snapshot_verified = false;
    }
    Ok(HourlyPayloadValidation {
        raw_rows: decoded_rows,
        snapshot_events,
        validated_snapshot_events,
        audited_anchor_snapshot_verified,
        update_events,
        minimum_provider_received_at: minimum_received_at
            .context("CryptoHFT archive receipt minimum was unavailable")?,
        maximum_provider_received_at: maximum_received_at
            .context("CryptoHFT archive receipt maximum was unavailable")?,
        receipt_rows_outside_exact_hour: outside_exact_hour,
    })
}

fn validate_hourly_row_causal_time(
    received_at: DateTime<Utc>,
    event_at: DateTime<Utc>,
    tolerated_start: DateTime<Utc>,
    tolerated_end: DateTime<Utc>,
) -> Result<()> {
    let causal_at = received_at.max(event_at);
    if causal_at < tolerated_start || causal_at >= tolerated_end {
        bail!(
            "CryptoHFT archive row causal time {causal_at} from receipt time {received_at} and event time {event_at} escaped tolerated hour [{tolerated_start}, {tolerated_end})"
        );
    }
    Ok(())
}

fn validate_hourly_receipt_distribution(
    decoded_rows: u64,
    inside_exact_hour: u64,
    outside_exact_hour: u64,
) -> Result<()> {
    if inside_exact_hour.checked_add(outside_exact_hour) != Some(decoded_rows) {
        bail!("CryptoHFT receipt-hour classification did not match the decoded row count");
    }
    if inside_exact_hour == 0 {
        bail!("CryptoHFT archive had no receipt rows inside the requested UTC hour");
    }
    if outside_exact_hour.saturating_mul(1_000) > decoded_rows {
        bail!("fewer than 99.9% of CryptoHFT receipt rows belonged to the requested UTC hour");
    }
    Ok(())
}

fn finish_hourly_validation_event(
    key: EventKey,
    snapshot: Option<LogicalEvent>,
    snapshot_events: &mut u64,
    validated_snapshot_events: &mut u64,
    audited_anchor_snapshot_verified: &mut bool,
    update_events: &mut u64,
) -> Result<()> {
    match key.event_type {
        EventType::Snapshot => {
            *snapshot_events = snapshot_events.saturating_add(1);
            let snapshot = snapshot.context("CryptoHFT snapshot validation state was missing")?;
            if is_valid_full_depth_snapshot(&snapshot)? {
                *validated_snapshot_events = validated_snapshot_events.saturating_add(1);
                if snapshot.key.event_time_ms == CRYPTOHFT_AUDITED_SNAPSHOT_EVENT_MILLIS
                    && snapshot.key.last_update_id
                        == Some(CRYPTOHFT_AUDITED_SNAPSHOT_LAST_UPDATE_ID)
                {
                    *audited_anchor_snapshot_verified = true;
                }
            }
        }
        EventType::Update => {
            if snapshot.is_some() {
                bail!("CryptoHFT update unexpectedly carried snapshot validation state");
            }
            *update_events = update_events.saturating_add(1);
        }
    }
    Ok(())
}

fn is_valid_full_depth_snapshot(event: &LogicalEvent) -> Result<bool> {
    let Some(sequence) = event.key.last_update_id else {
        return Ok(false);
    };
    if sequence < 0
        || event.key.first_update_id.is_some()
        || event.key.previous_final_update_id.is_some()
        || event
            .key
            .final_update_id
            .is_some_and(|final_id| final_id != sequence)
        || event.levels.len() < MIN_FULL_DEPTH_SNAPSHOT_LEVELS
    {
        return Ok(false);
    }
    let mut bids = BTreeMap::new();
    let mut asks = BTreeMap::new();
    for level in event.levels.values() {
        apply_absolute_level(
            match level.side {
                BookSide::Bid => &mut bids,
                BookSide::Ask => &mut asks,
            },
            level,
        )?;
    }
    if bids.len() > MAX_BOOK_LEVELS_PER_SIDE
        || asks.len() > MAX_BOOK_LEVELS_PER_SIDE
        || bids.len() < 20
        || asks.len() < 20
    {
        return Ok(false);
    }
    let best_bid = bids.last_key_value();
    let best_ask = asks.first_key_value();
    Ok(matches!(
        (best_bid, best_ask),
        (Some((bid_price, bid_quantity)), Some((ask_price, ask_quantity)))
            if bid_price < ask_price
                && *bid_quantity > Decimal::ZERO
                && *ask_quantity > Decimal::ZERO
    ))
}

#[derive(Debug, Clone)]
pub struct CryptoHftDayParseRequest {
    pub target_date: NaiveDate,
    pub context_archives: Vec<HourlyArchiveManifest>,
    pub target_archives: Vec<HourlyArchiveManifest>,
    pub output_batch_rows: usize,
    pub cancellation: ArchiveCancellation,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct BinanceL2DaySummary {
    pub raw_rows: u64,
    pub logical_events: u64,
    pub snapshot_events: u64,
    pub update_events: u64,
    pub updates_before_snapshot: u64,
    pub sequence_gaps: u64,
    pub invalid_book_events: u64,
    pub emitted_feature_rows: u64,
    pub unavailable_seconds: u64,
    pub batches: u64,
    pub maximum_batch_rows: u64,
    pub minimum_source_timestamp: Option<DateTime<Utc>>,
    pub maximum_source_timestamp: Option<DateTime<Utc>>,
    pub no_snapshot_bootstrap: bool,
}

pub fn spawn_day_parser(
    config: CryptoHftBinanceL2Config,
    request: CryptoHftDayParseRequest,
) -> (
    mpsc::Receiver<Vec<BinanceL2OneSecondFeature>>,
    JoinHandle<Result<BinanceL2DaySummary>>,
) {
    let (sender, receiver) = mpsc::channel(1);
    let handle = tokio::spawn(async move {
        config.validate()?;
        validate_day_parse_request(&request)?;
        let cancellation = request.cancellation.clone();
        let mut archives = BTreeMap::new();
        for manifest in request
            .context_archives
            .into_iter()
            .chain(request.target_archives)
        {
            if let Some(existing) = archives.insert(manifest.remote_file.clone(), manifest.clone())
            {
                if existing.sha256 != manifest.sha256
                    || existing.compressed_bytes != manifest.compressed_bytes
                {
                    bail!("CryptoHFT day replay received conflicting duplicate archives");
                }
            }
        }
        let mut replay = DayReplay::new(
            &config,
            request.target_date,
            request.output_batch_rows,
            sender,
            cancellation.clone(),
        )?;
        for manifest in archives.into_values() {
            check_cancelled(&cancellation)?;
            let parquet =
                decompress_hour_to_temporary_parquet(&config, &manifest, &cancellation).await?;
            replay = tokio::task::spawn_blocking(move || {
                replay.parse_parquet(parquet.path())?;
                Ok::<_, anyhow::Error>(replay)
            })
            .await
            .context("CryptoHFT Parquet replay task failed")??;
        }
        tokio::task::spawn_blocking(move || replay.finish())
            .await
            .context("CryptoHFT replay finalization task failed")?
    });
    (receiver, handle)
}

fn validate_day_parse_request(request: &CryptoHftDayParseRequest) -> Result<()> {
    if !(1..=1_000).contains(&request.output_batch_rows) {
        bail!("CryptoHFT feature output batches must contain between 1 and 1000 rows");
    }
    if request.target_archives.len() != 24 {
        bail!("CryptoHFT day replay requires exactly 24 target archives");
    }
    if request.context_archives.is_empty()
        || request.context_archives.len() > MAX_CONTEXT_LOOKBACK_HOURS
    {
        bail!(
            "CryptoHFT day replay requires between 1 and {MAX_CONTEXT_LOOKBACK_HOURS} context hours"
        );
    }
    let target_start = request
        .target_date
        .and_hms_opt(0, 0, 0)
        .context("CryptoHFT target date could not form UTC midnight")?
        .and_utc();
    let mut hours = BTreeSet::new();
    for manifest in &request.target_archives {
        let hour = manifest_utc_hour(manifest)?;
        if hour.date_naive() != request.target_date || !hours.insert(hour) {
            bail!("CryptoHFT target archive hours must be unique and cover 00 through 23");
        }
    }
    if hours.len() != 24 {
        bail!("CryptoHFT target archives did not cover a complete UTC day");
    }
    let context_hours = request
        .context_archives
        .iter()
        .map(manifest_utc_hour)
        .collect::<Result<BTreeSet<_>>>()?;
    if context_hours.len() != request.context_archives.len() {
        bail!("CryptoHFT context archive hours must be unique");
    }
    let expected_latest_context = target_start - TimeDelta::hours(1);
    if context_hours.last().copied() != Some(expected_latest_context) {
        bail!("CryptoHFT context must end at the hour immediately before the target day");
    }
    let first_context = context_hours
        .first()
        .copied()
        .context("CryptoHFT context archive set was empty")?;
    let earliest_context =
        DateTime::<Utc>::from_timestamp(CRYPTOHFT_EARLIEST_CONTEXT_HOUR_EPOCH, 0)
            .context("CryptoHFT audited context boundary was invalid")?;
    if first_context < earliest_context {
        bail!("CryptoHFT context preceded the audited 2026-04-13T23:00:00Z bootstrap boundary");
    }
    if first_context == earliest_context {
        let anchor = request
            .context_archives
            .iter()
            .find(|manifest| manifest.remote_file == CRYPTOHFT_AUDITED_ANCHOR_REMOTE_FILE)
            .context("CryptoHFT context omitted the audited anchor archive")?;
        if anchor.sha256 != CRYPTOHFT_AUDITED_ANCHOR_SHA256
            || !anchor.audited_anchor_snapshot_verified
        {
            bail!("CryptoHFT context did not match the pinned audited anchor");
        }
    }
    let expected_context_hours = usize::try_from(
        expected_latest_context
            .signed_duration_since(first_context)
            .num_hours()
            + 1,
    )
    .context("CryptoHFT context-hour count overflow")?;
    if expected_context_hours != context_hours.len() {
        bail!("CryptoHFT context archives were not a contiguous hourly chain");
    }
    if !request
        .context_archives
        .iter()
        .any(|manifest| manifest.validated_snapshot_events > 0)
    {
        bail!("CryptoHFT context did not contain a structurally validated full-depth snapshot bootstrap");
    }
    Ok(())
}

fn manifest_utc_hour(manifest: &HourlyArchiveManifest) -> Result<DateTime<Utc>> {
    let suffix = format!("/{CRYPTOHFT_SYMBOL}_orderbook.parquet.zst");
    let relative = manifest
        .remote_file
        .strip_prefix("binance_futures/")
        .and_then(|value| value.strip_suffix(&suffix))
        .context("CryptoHFT archive name did not match the source contract")?;
    let (date, hour) = relative
        .split_once('/')
        .context("CryptoHFT archive name omitted its UTC hour")?;
    let date = NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .context("CryptoHFT archive name contained an invalid UTC date")?;
    let hour = hour
        .parse::<u32>()
        .context("CryptoHFT archive name contained an invalid UTC hour")?;
    date.and_hms_opt(hour, 0, 0)
        .context("CryptoHFT archive name contained an out-of-range UTC hour")
        .map(|timestamp| timestamp.and_utc())
}

const CRYPTOHFT_REQUIRED_COLUMNS: [&str; 13] = [
    "received_time",
    "event_time",
    "transaction_time",
    "symbol",
    "event_type",
    "first_update_id",
    "final_update_id",
    "prev_final_update_id",
    "last_update_id",
    "side",
    "price",
    "quantity",
    "order_count",
];
const ROLLING_HORIZONS_SECONDS: [i64; 5] = [1, 5, 15, 30, 60];
const TARGET_SECONDS_PER_DAY: u64 = 86_400;
const PERSISTED_DECIMAL_SCALE: u32 = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
struct EventKey {
    /// Maximum provider receipt time across every price-level row in this logical event.
    received_time_ns: i64,
    event_time_ms: i64,
    transaction_time_ms: Option<i64>,
    event_type: EventType,
    first_update_id: Option<i64>,
    final_update_id: Option<i64>,
    previous_final_update_id: Option<i64>,
    last_update_id: Option<i64>,
}

impl EventKey {
    fn same_logical_event(&self, other: &Self) -> bool {
        self.event_time_ms == other.event_time_ms
            && self.transaction_time_ms == other.transaction_time_ms
            && self.event_type == other.event_type
            && self.first_update_id == other.first_update_id
            && self.final_update_id == other.final_update_id
            && self.previous_final_update_id == other.previous_final_update_id
            && self.last_update_id == other.last_update_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventType {
    Snapshot,
    Update,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum BookSide {
    Bid,
    Ask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UpdateDisposition {
    Apply,
    IgnoreStale,
    Gap,
}

#[derive(Debug, Clone, PartialEq)]
struct PriceLevel {
    side: BookSide,
    price: Decimal,
    quantity: Decimal,
}

#[derive(Debug, Clone, PartialEq)]
struct LogicalEvent {
    key: EventKey,
    levels: BTreeMap<(BookSide, Decimal), PriceLevel>,
}

impl LogicalEvent {
    fn new(key: EventKey, level: PriceLevel) -> Self {
        let mut event = Self {
            key,
            levels: BTreeMap::new(),
        };
        event
            .push_level(level)
            .expect("a new logical event must accept its first valid level");
        event
    }

    fn push_level(&mut self, level: PriceLevel) -> Result<()> {
        let key = (level.side, level.price);
        if let Some(existing) = self.levels.get(&key) {
            if existing.quantity != level.quantity {
                bail!("CryptoHFT logical event repeated a price level with conflicting quantity");
            }
            return Ok(());
        }
        if self.levels.len() >= MAX_LOGICAL_EVENT_LEVELS {
            bail!("CryptoHFT logical event exceeded the price-level safety bound");
        }
        self.levels.insert(key, level);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct BaseSecondState {
    second_start: DateTime<Utc>,
    source_event_timestamp: DateTime<Utc>,
    provider_received_at: DateTime<Utc>,
    available_at: DateTime<Utc>,
    source_update_id: i64,
    midpoint: Decimal,
    microprice: Decimal,
    spread_bps: Decimal,
    bid_depth_5: Decimal,
    ask_depth_5: Decimal,
    imbalance_5: Decimal,
    bid_depth_10: Decimal,
    ask_depth_10: Decimal,
    imbalance_10: Decimal,
    bid_depth_20: Decimal,
    ask_depth_20: Decimal,
    imbalance_20: Decimal,
    bid_depth_slope_20: Decimal,
    ask_depth_slope_20: Decimal,
    bid_depth_concentration_20: Decimal,
    ask_depth_concentration_20: Decimal,
    bid_quote_replenishment_1s: Decimal,
    ask_quote_replenishment_1s: Decimal,
    bid_quote_churn_1s: Decimal,
    ask_quote_churn_1s: Decimal,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct QuoteFlow {
    bid_replenishment: Decimal,
    ask_replenishment: Decimal,
    bid_churn: Decimal,
    ask_churn: Decimal,
}

#[derive(Debug, Clone)]
struct PendingSecond {
    second_start: DateTime<Utc>,
    latest_state: Option<BaseSecondState>,
}

struct DayReplay {
    target_start: DateTime<Utc>,
    target_end: DateTime<Utc>,
    availability_offset: TimeDelta,
    max_stale: TimeDelta,
    output_batch_rows: usize,
    sender: mpsc::Sender<Vec<BinanceL2OneSecondFeature>>,
    cancellation: ArchiveCancellation,
    bids: BTreeMap<Decimal, Decimal>,
    asks: BTreeMap<Decimal, Decimal>,
    book_ready: bool,
    saw_snapshot: bool,
    last_sequence: Option<i64>,
    awaiting_snapshot_bridge: bool,
    pending_event: Option<LogicalEvent>,
    last_available_at: Option<DateTime<Utc>>,
    pending_second: Option<PendingSecond>,
    current_flow_second: Option<DateTime<Utc>>,
    current_flow: QuoteFlow,
    rolling: VecDeque<BaseSecondState>,
    output_batch: Vec<BinanceL2OneSecondFeature>,
    summary: BinanceL2DaySummary,
}

impl DayReplay {
    fn new(
        config: &CryptoHftBinanceL2Config,
        target_date: NaiveDate,
        output_batch_rows: usize,
        sender: mpsc::Sender<Vec<BinanceL2OneSecondFeature>>,
        cancellation: ArchiveCancellation,
    ) -> Result<Self> {
        if !(1..=1_000).contains(&output_batch_rows) {
            bail!("CryptoHFT feature output batches must contain between 1 and 1000 rows");
        }
        let availability_offset_ms = i64::try_from(config.availability_offset_ms)
            .context("CryptoHFT availability offset exceeded i64")?;
        let max_stale_ms = i64::try_from(config.max_stale_ms)
            .context("CryptoHFT maximum stale duration exceeded i64")?;
        let target_start = target_date
            .and_hms_opt(0, 0, 0)
            .context("CryptoHFT target date could not form midnight")?
            .and_utc();
        let target_end = target_start
            .checked_add_signed(TimeDelta::days(1))
            .context("CryptoHFT target date overflow")?;
        Ok(Self {
            target_start,
            target_end,
            availability_offset: TimeDelta::milliseconds(availability_offset_ms),
            max_stale: TimeDelta::milliseconds(max_stale_ms),
            output_batch_rows,
            sender,
            cancellation,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            book_ready: false,
            saw_snapshot: false,
            last_sequence: None,
            awaiting_snapshot_bridge: false,
            pending_event: None,
            last_available_at: None,
            pending_second: None,
            current_flow_second: None,
            current_flow: QuoteFlow::default(),
            rolling: VecDeque::with_capacity(61),
            output_batch: Vec::with_capacity(output_batch_rows),
            summary: BinanceL2DaySummary::default(),
        })
    }

    fn parse_parquet(&mut self, path: &Path) -> Result<()> {
        check_cancelled(&self.cancellation)?;
        let file = File::open(path)
            .with_context(|| format!("failed to open CryptoHFT Parquet file {}", path.display()))?;
        let reader = SerializedFileReader::new(file).with_context(|| {
            format!(
                "failed to initialize CryptoHFT Parquet file {}",
                path.display()
            )
        })?;
        let has_order_count = validate_cryptohft_schema(&reader)?;
        for row_group_index in 0..reader.num_row_groups() {
            check_cancelled(&self.cancellation)?;
            let row_group = reader.get_row_group(row_group_index).with_context(|| {
                format!("failed to initialize CryptoHFT Parquet row group {row_group_index}")
            })?;
            let rows = row_group.get_row_iter(None).with_context(|| {
                format!("failed to stream CryptoHFT Parquet row group {row_group_index}")
            })?;
            for (row_ordinal, row) in rows.enumerate() {
                check_cancelled(&self.cancellation)?;
                let row = row.map_err(|error| {
                    anyhow::anyhow!(
                        "failed to decode CryptoHFT Parquet row group {row_group_index}, row {row_ordinal}: {error}"
                    )
                })?;
                let (key, level) = parse_cryptohft_row(row, has_order_count)?;
                self.summary.raw_rows = self.summary.raw_rows.saturating_add(1);
                match self.pending_event.as_mut() {
                    Some(pending) if pending.key.same_logical_event(&key) => {
                        pending.key.received_time_ns =
                            pending.key.received_time_ns.max(key.received_time_ns);
                        pending.push_level(level)?;
                    }
                    Some(_) => {
                        let complete = self
                            .pending_event
                            .take()
                            .context("CryptoHFT pending event disappeared")?;
                        self.process_event(complete)?;
                        self.pending_event = Some(LogicalEvent::new(key, level));
                    }
                    None => {
                        self.pending_event = Some(LogicalEvent::new(key, level));
                    }
                }
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<BinanceL2DaySummary> {
        check_cancelled(&self.cancellation)?;
        if let Some(event) = self.pending_event.take() {
            self.process_event(event)?;
        }
        self.finalize_pending_second()?;
        self.flush_batch()?;
        self.summary.no_snapshot_bootstrap = !self.saw_snapshot;
        self.summary.unavailable_seconds = TARGET_SECONDS_PER_DAY.saturating_sub(
            self.summary
                .emitted_feature_rows
                .min(TARGET_SECONDS_PER_DAY),
        );
        Ok(self.summary)
    }

    fn process_event(&mut self, event: LogicalEvent) -> Result<()> {
        check_cancelled(&self.cancellation)?;
        self.summary.logical_events = self.summary.logical_events.saturating_add(1);
        let source_event_timestamp = DateTime::from_timestamp_millis(event.key.event_time_ms)
            .context("CryptoHFT event_time was outside the supported UTC range")?;
        let provider_received_at = timestamp_from_nanoseconds(event.key.received_time_ns)?;
        self.summary.minimum_source_timestamp = Some(
            self.summary
                .minimum_source_timestamp
                .map_or(source_event_timestamp, |value| {
                    value.min(source_event_timestamp)
                }),
        );
        self.summary.maximum_source_timestamp = Some(
            self.summary
                .maximum_source_timestamp
                .map_or(source_event_timestamp, |value| {
                    value.max(source_event_timestamp)
                }),
        );
        let reported_available_at = provider_received_at
            .max(source_event_timestamp)
            .checked_add_signed(self.availability_offset)
            .context("CryptoHFT available_at overflow")?;
        let available_at = self
            .last_available_at
            .map_or(reported_available_at, |previous| {
                previous.max(reported_available_at)
            });
        self.last_available_at = Some(available_at);
        self.advance_flow_second(floor_utc_second(available_at))?;

        match event.key.event_type {
            EventType::Snapshot => {
                self.summary.snapshot_events = self.summary.snapshot_events.saturating_add(1);
                self.apply_snapshot(&event)?;
            }
            EventType::Update => {
                self.summary.update_events = self.summary.update_events.saturating_add(1);
                if !self.book_ready {
                    self.summary.updates_before_snapshot =
                        self.summary.updates_before_snapshot.saturating_add(1);
                    self.observe_unavailable(available_at)?;
                    return Ok(());
                }
                match self.update_disposition(&event.key)? {
                    UpdateDisposition::Apply => self.apply_update(&event)?,
                    UpdateDisposition::IgnoreStale => return Ok(()),
                    UpdateDisposition::Gap => {
                        self.summary.sequence_gaps = self.summary.sequence_gaps.saturating_add(1);
                        self.invalidate_book();
                        self.observe_unavailable(available_at)?;
                        return Ok(());
                    }
                }
            }
        }

        if !self.book_ready {
            self.observe_unavailable(available_at)?;
            return Ok(());
        }
        let source_update_id = self
            .last_sequence
            .context("CryptoHFT ready book did not have a sequence")?;
        let state = match self.build_base_state(
            source_event_timestamp,
            provider_received_at,
            available_at,
            source_update_id,
        )? {
            Some(state) => state,
            None => {
                self.summary.invalid_book_events =
                    self.summary.invalid_book_events.saturating_add(1);
                if matches!(event.key.event_type, EventType::Snapshot) {
                    self.invalidate_book();
                }
                self.observe_unavailable(available_at)?;
                return Ok(());
            }
        };
        if matches!(event.key.event_type, EventType::Snapshot) {
            self.saw_snapshot = true;
        }
        let stale = available_at.signed_duration_since(source_event_timestamp) > self.max_stale;
        if stale {
            self.observe_unavailable(available_at)?;
            return Ok(());
        }
        self.observe_state(state)?;
        Ok(())
    }

    fn apply_snapshot(&mut self, event: &LogicalEvent) -> Result<()> {
        let sequence = event
            .key
            .last_update_id
            .context("CryptoHFT snapshot omitted last_update_id")?;
        if sequence < 0
            || event.key.first_update_id.is_some()
            || event.key.previous_final_update_id.is_some()
            || event
                .key
                .final_update_id
                .is_some_and(|final_id| final_id != sequence)
        {
            bail!("CryptoHFT snapshot sequence columns were invalid");
        }
        self.bids.clear();
        self.asks.clear();
        self.rolling.clear();
        self.current_flow = QuoteFlow::default();
        for level in event.levels.values() {
            apply_absolute_level(
                match level.side {
                    BookSide::Bid => &mut self.bids,
                    BookSide::Ask => &mut self.asks,
                },
                level,
            )?;
        }
        self.require_bounded_book()?;
        self.last_sequence = Some(sequence);
        self.awaiting_snapshot_bridge = true;
        self.book_ready = true;
        Ok(())
    }

    fn update_disposition(&self, key: &EventKey) -> Result<UpdateDisposition> {
        if key.last_update_id.is_some() {
            bail!("CryptoHFT update unexpectedly contained last_update_id");
        }
        let first = key
            .first_update_id
            .context("CryptoHFT update omitted first_update_id")?;
        let final_id = key
            .final_update_id
            .context("CryptoHFT update omitted final_update_id")?;
        let previous = key
            .previous_final_update_id
            .context("CryptoHFT update omitted prev_final_update_id")?;
        if first < 0 || final_id < first || previous < 0 {
            bail!("CryptoHFT update sequence columns were invalid");
        }
        let current = self
            .last_sequence
            .context("CryptoHFT ready book did not have a sequence")?;
        if self.awaiting_snapshot_bridge {
            if final_id < current {
                return Ok(UpdateDisposition::IgnoreStale);
            }
            if first <= current && current <= final_id {
                return Ok(UpdateDisposition::Apply);
            }
            return Ok(UpdateDisposition::Gap);
        }
        if final_id <= current {
            return Ok(UpdateDisposition::IgnoreStale);
        }
        if previous == current {
            Ok(UpdateDisposition::Apply)
        } else {
            Ok(UpdateDisposition::Gap)
        }
    }

    fn apply_update(&mut self, event: &LogicalEvent) -> Result<()> {
        for level in event.levels.values() {
            let map = match level.side {
                BookSide::Bid => &mut self.bids,
                BookSide::Ask => &mut self.asks,
            };
            let previous = map.get(&level.price).copied().unwrap_or(Decimal::ZERO);
            apply_absolute_level(map, level)?;
            let quantity_delta = level.quantity - previous;
            let quote_delta = level
                .price
                .checked_mul(quantity_delta.abs())
                .context("CryptoHFT quote-flow multiplication overflow")?;
            match (level.side, quantity_delta.is_sign_positive()) {
                (BookSide::Bid, true) => {
                    self.current_flow.bid_replenishment = self
                        .current_flow
                        .bid_replenishment
                        .checked_add(quote_delta)
                        .context("CryptoHFT bid replenishment overflow")?;
                }
                (BookSide::Ask, true) => {
                    self.current_flow.ask_replenishment = self
                        .current_flow
                        .ask_replenishment
                        .checked_add(quote_delta)
                        .context("CryptoHFT ask replenishment overflow")?;
                }
                (BookSide::Bid, false) => {
                    self.current_flow.bid_churn = self
                        .current_flow
                        .bid_churn
                        .checked_add(quote_delta)
                        .context("CryptoHFT bid churn overflow")?;
                }
                (BookSide::Ask, false) => {
                    self.current_flow.ask_churn = self
                        .current_flow
                        .ask_churn
                        .checked_add(quote_delta)
                        .context("CryptoHFT ask churn overflow")?;
                }
            }
        }
        self.require_bounded_book()?;
        self.last_sequence = event.key.final_update_id;
        self.awaiting_snapshot_bridge = false;
        Ok(())
    }

    fn require_bounded_book(&self) -> Result<()> {
        if self.bids.len() > MAX_BOOK_LEVELS_PER_SIDE || self.asks.len() > MAX_BOOK_LEVELS_PER_SIDE
        {
            bail!("CryptoHFT reconstructed book exceeded the memory safety bound");
        }
        Ok(())
    }

    fn invalidate_book(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.book_ready = false;
        self.last_sequence = None;
        self.awaiting_snapshot_bridge = false;
        self.rolling.clear();
        self.current_flow = QuoteFlow::default();
    }

    fn advance_flow_second(&mut self, second_start: DateTime<Utc>) -> Result<()> {
        if let Some(current) = self.current_flow_second {
            if second_start < current {
                bail!("CryptoHFT events were not monotonic by causal availability time");
            }
            if second_start > current {
                self.current_flow = QuoteFlow::default();
            }
        }
        self.current_flow_second = Some(second_start);
        Ok(())
    }

    fn observe_state(&mut self, state: BaseSecondState) -> Result<()> {
        self.advance_pending_second(state.second_start)?;
        let pending = self
            .pending_second
            .as_mut()
            .context("CryptoHFT pending second was not initialized")?;
        pending.latest_state = Some(state);
        Ok(())
    }

    fn observe_unavailable(&mut self, available_at: DateTime<Utc>) -> Result<()> {
        let second_start = floor_utc_second(available_at);
        self.advance_pending_second(second_start)?;
        let pending = self
            .pending_second
            .as_mut()
            .context("CryptoHFT pending second was not initialized")?;
        pending.latest_state = None;
        Ok(())
    }

    fn advance_pending_second(&mut self, second_start: DateTime<Utc>) -> Result<()> {
        if let Some(current) = self.pending_second.as_ref() {
            if second_start < current.second_start {
                bail!("CryptoHFT feature seconds were not monotonic");
            }
            if second_start == current.second_start {
                return Ok(());
            }
            self.finalize_pending_second()?;
        }
        self.pending_second = Some(PendingSecond {
            second_start,
            latest_state: None,
        });
        Ok(())
    }

    fn finalize_pending_second(&mut self) -> Result<()> {
        let Some(pending) = self.pending_second.take() else {
            return Ok(());
        };
        let Some(state) = pending.latest_state else {
            self.rolling.clear();
            return Ok(());
        };
        if self.rolling.back().is_some_and(|previous| {
            previous
                .second_start
                .checked_add_signed(TimeDelta::seconds(1))
                != Some(state.second_start)
        }) {
            self.rolling.clear();
        }
        let feature = self.qualified_feature(&state)?;
        self.rolling.push_back(state);
        while self.rolling.len() > 61 {
            self.rolling.pop_front();
        }
        if let Some(feature) = feature {
            self.output_batch.push(feature);
            self.summary.emitted_feature_rows = self.summary.emitted_feature_rows.saturating_add(1);
            if self.output_batch.len() == self.output_batch_rows {
                self.flush_batch()?;
            }
        }
        Ok(())
    }

    fn qualified_feature(
        &self,
        current: &BaseSecondState,
    ) -> Result<Option<BinanceL2OneSecondFeature>> {
        if current.second_start < self.target_start || current.second_start >= self.target_end {
            return Ok(None);
        }
        let mut prior = Vec::with_capacity(ROLLING_HORIZONS_SECONDS.len());
        for horizon in ROLLING_HORIZONS_SECONDS {
            let expected = current
                .second_start
                .checked_sub_signed(TimeDelta::seconds(horizon))
                .context("CryptoHFT rolling horizon underflow")?;
            let Some(state) = self
                .rolling
                .iter()
                .rev()
                .find(|candidate| candidate.second_start == expected)
            else {
                return Ok(None);
            };
            prior.push(state);
        }
        let changes = prior
            .iter()
            .map(|state| rolling_changes(current, state))
            .collect::<Result<Vec<_>>>()?;
        let feature = BinanceL2OneSecondFeature {
            symbol: CRYPTOHFT_SYMBOL.to_owned(),
            second_start: current.second_start,
            source_event_timestamp: current.source_event_timestamp,
            provider_received_at: current.provider_received_at,
            available_at: current.available_at,
            source_update_id: current.source_update_id,
            feature_schema_version: BINANCE_L2_FEATURE_SCHEMA_VERSION.to_owned(),
            quality_status: "qualified".to_owned(),
            midpoint: current.midpoint,
            microprice: current.microprice,
            spread_bps: current.spread_bps,
            bid_depth_5: current.bid_depth_5,
            ask_depth_5: current.ask_depth_5,
            imbalance_5: current.imbalance_5,
            bid_depth_10: current.bid_depth_10,
            ask_depth_10: current.ask_depth_10,
            imbalance_10: current.imbalance_10,
            bid_depth_20: current.bid_depth_20,
            ask_depth_20: current.ask_depth_20,
            imbalance_20: current.imbalance_20,
            bid_depth_slope_20: current.bid_depth_slope_20,
            ask_depth_slope_20: current.ask_depth_slope_20,
            bid_depth_concentration_20: current.bid_depth_concentration_20,
            ask_depth_concentration_20: current.ask_depth_concentration_20,
            bid_quote_replenishment_1s: current.bid_quote_replenishment_1s,
            ask_quote_replenishment_1s: current.ask_quote_replenishment_1s,
            bid_quote_churn_1s: current.bid_quote_churn_1s,
            ask_quote_churn_1s: current.ask_quote_churn_1s,
            midpoint_change_bps_1s: changes[0].0,
            spread_bps_delta_1s: changes[0].1,
            depth_20_change_bps_1s: changes[0].2,
            imbalance_20_delta_1s: changes[0].3,
            midpoint_change_bps_5s: changes[1].0,
            spread_bps_delta_5s: changes[1].1,
            depth_20_change_bps_5s: changes[1].2,
            imbalance_20_delta_5s: changes[1].3,
            midpoint_change_bps_15s: changes[2].0,
            spread_bps_delta_15s: changes[2].1,
            depth_20_change_bps_15s: changes[2].2,
            imbalance_20_delta_15s: changes[2].3,
            midpoint_change_bps_30s: changes[3].0,
            spread_bps_delta_30s: changes[3].1,
            depth_20_change_bps_30s: changes[3].2,
            imbalance_20_delta_30s: changes[3].3,
            midpoint_change_bps_60s: changes[4].0,
            spread_bps_delta_60s: changes[4].1,
            depth_20_change_bps_60s: changes[4].2,
            imbalance_20_delta_60s: changes[4].3,
        };
        Ok(Some(quantize_feature(feature)))
    }

    fn build_base_state(
        &self,
        source_event_timestamp: DateTime<Utc>,
        provider_received_at: DateTime<Utc>,
        available_at: DateTime<Utc>,
        source_update_id: i64,
    ) -> Result<Option<BaseSecondState>> {
        let bids = self
            .bids
            .iter()
            .rev()
            .take(20)
            .map(|(price, quantity)| (*price, *quantity))
            .collect::<Vec<_>>();
        let asks = self
            .asks
            .iter()
            .take(20)
            .map(|(price, quantity)| (*price, *quantity))
            .collect::<Vec<_>>();
        if bids.len() < 20 || asks.len() < 20 {
            return Ok(None);
        }
        let best_bid = bids[0];
        let best_ask = asks[0];
        if best_bid.0 >= best_ask.0 || best_bid.1 <= Decimal::ZERO || best_ask.1 <= Decimal::ZERO {
            return Ok(None);
        }
        let midpoint = checked_div(
            checked_add(best_bid.0, best_ask.0, "midpoint addition")?,
            Decimal::from(2u32),
            "midpoint division",
        )?;
        let top_quantity = checked_add(best_bid.1, best_ask.1, "top quantity addition")?;
        let microprice = checked_div(
            checked_add(
                checked_mul(best_ask.0, best_bid.1, "microprice bid term")?,
                checked_mul(best_bid.0, best_ask.1, "microprice ask term")?,
                "microprice numerator",
            )?,
            top_quantity,
            "microprice division",
        )?;
        let spread_bps = basis_points_delta(best_ask.0, best_bid.0, midpoint)?;
        let (bid_depth_5, bid_depth_10, bid_depth_20) = tier_depths(&bids)?;
        let (ask_depth_5, ask_depth_10, ask_depth_20) = tier_depths(&asks)?;
        let imbalance_5 = imbalance(bid_depth_5, ask_depth_5)?;
        let imbalance_10 = imbalance(bid_depth_10, ask_depth_10)?;
        let imbalance_20 = imbalance(bid_depth_20, ask_depth_20)?;
        let bid_depth_slope_20 = checked_div(
            basis_points_delta(best_bid.0, bids[19].0, best_bid.0)?,
            bid_depth_20,
            "bid depth slope",
        )?;
        let ask_depth_slope_20 = checked_div(
            basis_points_delta(asks[19].0, best_ask.0, best_ask.0)?,
            ask_depth_20,
            "ask depth slope",
        )?;
        Ok(Some(BaseSecondState {
            second_start: floor_utc_second(available_at),
            source_event_timestamp,
            provider_received_at,
            available_at,
            source_update_id,
            midpoint,
            microprice,
            spread_bps,
            bid_depth_5,
            ask_depth_5,
            imbalance_5,
            bid_depth_10,
            ask_depth_10,
            imbalance_10,
            bid_depth_20,
            ask_depth_20,
            imbalance_20,
            bid_depth_slope_20,
            ask_depth_slope_20,
            bid_depth_concentration_20: checked_div(
                bid_depth_5,
                bid_depth_20,
                "bid depth concentration",
            )?,
            ask_depth_concentration_20: checked_div(
                ask_depth_5,
                ask_depth_20,
                "ask depth concentration",
            )?,
            bid_quote_replenishment_1s: self.current_flow.bid_replenishment,
            ask_quote_replenishment_1s: self.current_flow.ask_replenishment,
            bid_quote_churn_1s: self.current_flow.bid_churn,
            ask_quote_churn_1s: self.current_flow.ask_churn,
        }))
    }

    fn flush_batch(&mut self) -> Result<()> {
        if self.output_batch.is_empty() {
            return Ok(());
        }
        self.summary.batches = self.summary.batches.saturating_add(1);
        self.summary.maximum_batch_rows = self
            .summary
            .maximum_batch_rows
            .max(u64::try_from(self.output_batch.len()).context("feature batch length overflow")?);
        self.sender
            .blocking_send(std::mem::replace(
                &mut self.output_batch,
                Vec::with_capacity(self.output_batch_rows),
            ))
            .context("CryptoHFT feature consumer stopped")?;
        Ok(())
    }
}

fn validate_cryptohft_schema(reader: &SerializedFileReader<File>) -> Result<bool> {
    let fields = reader.metadata().file_metadata().schema().get_fields();
    if fields.len() != 12 && fields.len() != 13 {
        bail!(
            "CryptoHFT Parquet schema exposed {} columns; expected 12 or 13",
            fields.len()
        );
    }
    for (index, field) in fields.iter().enumerate() {
        if field.name() != CRYPTOHFT_REQUIRED_COLUMNS[index] {
            bail!(
                "CryptoHFT Parquet column {index} was {}; expected {}",
                field.name(),
                CRYPTOHFT_REQUIRED_COLUMNS[index]
            );
        }
    }
    Ok(fields.len() == 13)
}

fn parse_cryptohft_row(row: Row, has_order_count: bool) -> Result<(EventKey, PriceLevel)> {
    let columns = row.into_columns();
    let expected = if has_order_count { 13 } else { 12 };
    if columns.len() != expected {
        bail!("CryptoHFT Parquet row did not match its validated schema");
    }
    let received_time_ns = required_long(&columns[0].1, "received_time")?;
    let event_time_ms = required_long(&columns[1].1, "event_time")?;
    let transaction_time_ms = optional_long(&columns[2].1, "transaction_time")?;
    let symbol = required_string(&columns[3].1, "symbol")?;
    if symbol != CRYPTOHFT_SYMBOL {
        bail!("CryptoHFT archive contained symbol {symbol}; expected {CRYPTOHFT_SYMBOL}");
    }
    let event_type = match required_string(&columns[4].1, "event_type")? {
        "snapshot" => EventType::Snapshot,
        "update" => EventType::Update,
        value => bail!("CryptoHFT archive contained unsupported event_type {value}"),
    };
    let first_update_id = optional_long(&columns[5].1, "first_update_id")?;
    let final_update_id = optional_long(&columns[6].1, "final_update_id")?;
    let previous_final_update_id = optional_long(&columns[7].1, "prev_final_update_id")?;
    let last_update_id = optional_long(&columns[8].1, "last_update_id")?;
    let side = match required_string(&columns[9].1, "side")? {
        "bid" => BookSide::Bid,
        "ask" => BookSide::Ask,
        value => bail!("CryptoHFT archive contained unsupported side {value}"),
    };
    let price = parse_positive_decimal(required_string(&columns[10].1, "price")?, "price")?;
    let quantity =
        parse_nonnegative_decimal(required_string(&columns[11].1, "quantity")?, "quantity")?;
    if has_order_count {
        let order_count = optional_long(&columns[12].1, "order_count")?;
        if order_count.is_some_and(|value| value < 0) {
            bail!("CryptoHFT order_count was negative");
        }
    }
    Ok((
        EventKey {
            received_time_ns,
            event_time_ms,
            transaction_time_ms,
            event_type,
            first_update_id,
            final_update_id,
            previous_final_update_id,
            last_update_id,
        },
        PriceLevel {
            side,
            price,
            quantity,
        },
    ))
}

fn required_long(field: &Field, name: &str) -> Result<i64> {
    let Field::Long(value) = field else {
        bail!("CryptoHFT {name} was not a required int64");
    };
    Ok(*value)
}

fn optional_long(field: &Field, name: &str) -> Result<Option<i64>> {
    match field {
        Field::Null => Ok(None),
        Field::Long(value) => Ok(Some(*value)),
        _ => bail!("CryptoHFT {name} was neither null nor int64"),
    }
}

fn required_string<'a>(field: &'a Field, name: &str) -> Result<&'a str> {
    let Field::Str(value) = field else {
        bail!("CryptoHFT {name} was not a required string");
    };
    Ok(value)
}

fn parse_positive_decimal(value: &str, name: &str) -> Result<Decimal> {
    let value =
        Decimal::from_str(value).with_context(|| format!("CryptoHFT {name} was invalid"))?;
    if value <= Decimal::ZERO {
        bail!("CryptoHFT {name} was not positive");
    }
    Ok(value)
}

fn parse_nonnegative_decimal(value: &str, name: &str) -> Result<Decimal> {
    let value =
        Decimal::from_str(value).with_context(|| format!("CryptoHFT {name} was invalid"))?;
    if value < Decimal::ZERO {
        bail!("CryptoHFT {name} was negative");
    }
    Ok(value)
}

fn timestamp_from_nanoseconds(value: i64) -> Result<DateTime<Utc>> {
    let seconds = value.div_euclid(1_000_000_000);
    let nanoseconds = u32::try_from(value.rem_euclid(1_000_000_000))
        .context("CryptoHFT received_time nanoseconds overflow")?;
    DateTime::from_timestamp(seconds, nanoseconds)
        .context("CryptoHFT received_time was outside the supported UTC range")
}

fn floor_utc_second(value: DateTime<Utc>) -> DateTime<Utc> {
    DateTime::from_timestamp(value.timestamp(), 0)
        .expect("a valid DateTime must have a valid whole-second representation")
}

fn apply_absolute_level(book: &mut BTreeMap<Decimal, Decimal>, level: &PriceLevel) -> Result<()> {
    if level.price <= Decimal::ZERO || level.quantity < Decimal::ZERO {
        bail!("CryptoHFT book level had an invalid price or quantity");
    }
    if level.quantity == Decimal::ZERO {
        book.remove(&level.price);
    } else {
        book.insert(level.price, level.quantity);
    }
    Ok(())
}

fn tier_depths(levels: &[(Decimal, Decimal)]) -> Result<(Decimal, Decimal, Decimal)> {
    let mut depth_5 = Decimal::ZERO;
    let mut depth_10 = Decimal::ZERO;
    let mut depth_20 = Decimal::ZERO;
    for (index, (_, quantity)) in levels.iter().take(20).enumerate() {
        depth_20 = checked_add(depth_20, *quantity, "depth-20 addition")?;
        if index < 10 {
            depth_10 = checked_add(depth_10, *quantity, "depth-10 addition")?;
        }
        if index < 5 {
            depth_5 = checked_add(depth_5, *quantity, "depth-5 addition")?;
        }
    }
    if depth_5 <= Decimal::ZERO || depth_10 <= Decimal::ZERO || depth_20 <= Decimal::ZERO {
        bail!("CryptoHFT book depth was not positive");
    }
    Ok((depth_5, depth_10, depth_20))
}

fn imbalance(bid: Decimal, ask: Decimal) -> Result<Decimal> {
    checked_div(
        bid.checked_sub(ask)
            .context("CryptoHFT imbalance subtraction overflow")?,
        checked_add(bid, ask, "imbalance denominator")?,
        "imbalance division",
    )
}

fn rolling_changes(
    current: &BaseSecondState,
    previous: &BaseSecondState,
) -> Result<(Decimal, Decimal, Decimal, Decimal)> {
    let current_depth = checked_add(
        current.bid_depth_20,
        current.ask_depth_20,
        "current rolling depth",
    )?;
    let previous_depth = checked_add(
        previous.bid_depth_20,
        previous.ask_depth_20,
        "previous rolling depth",
    )?;
    Ok((
        relative_change_bps(current.midpoint, previous.midpoint)?,
        current
            .spread_bps
            .checked_sub(previous.spread_bps)
            .context("CryptoHFT spread delta overflow")?,
        relative_change_bps(current_depth, previous_depth)?,
        current
            .imbalance_20
            .checked_sub(previous.imbalance_20)
            .context("CryptoHFT imbalance delta overflow")?,
    ))
}

fn relative_change_bps(current: Decimal, previous: Decimal) -> Result<Decimal> {
    basis_points_delta(current, previous, previous)
}

fn basis_points_delta(high: Decimal, low: Decimal, denominator: Decimal) -> Result<Decimal> {
    let difference = high
        .checked_sub(low)
        .context("CryptoHFT basis-point subtraction overflow")?;
    checked_div(
        checked_mul(difference, Decimal::from(10_000u32), "basis-point scaling")?,
        denominator,
        "basis-point division",
    )
}

fn checked_add(left: Decimal, right: Decimal, operation: &str) -> Result<Decimal> {
    left.checked_add(right)
        .with_context(|| format!("CryptoHFT {operation} overflow"))
}

fn checked_mul(left: Decimal, right: Decimal, operation: &str) -> Result<Decimal> {
    left.checked_mul(right)
        .with_context(|| format!("CryptoHFT {operation} overflow"))
}

fn checked_div(numerator: Decimal, denominator: Decimal, operation: &str) -> Result<Decimal> {
    if denominator == Decimal::ZERO {
        bail!("CryptoHFT {operation} divided by zero");
    }
    numerator
        .checked_div(denominator)
        .with_context(|| format!("CryptoHFT {operation} overflow"))
}

fn quantize_feature(mut feature: BinanceL2OneSecondFeature) -> BinanceL2OneSecondFeature {
    macro_rules! quantize {
        ($($field:ident),+ $(,)?) => {
            $(
                feature.$field = feature.$field.round_dp_with_strategy(
                    PERSISTED_DECIMAL_SCALE,
                    RoundingStrategy::MidpointAwayFromZero,
                );
            )+
        };
    }
    quantize!(
        midpoint,
        microprice,
        spread_bps,
        bid_depth_5,
        ask_depth_5,
        imbalance_5,
        bid_depth_10,
        ask_depth_10,
        imbalance_10,
        bid_depth_20,
        ask_depth_20,
        imbalance_20,
        bid_depth_slope_20,
        ask_depth_slope_20,
        bid_depth_concentration_20,
        ask_depth_concentration_20,
        bid_quote_replenishment_1s,
        ask_quote_replenishment_1s,
        bid_quote_churn_1s,
        ask_quote_churn_1s,
        midpoint_change_bps_1s,
        spread_bps_delta_1s,
        depth_20_change_bps_1s,
        imbalance_20_delta_1s,
        midpoint_change_bps_5s,
        spread_bps_delta_5s,
        depth_20_change_bps_5s,
        imbalance_20_delta_5s,
        midpoint_change_bps_15s,
        spread_bps_delta_15s,
        depth_20_change_bps_15s,
        imbalance_20_delta_15s,
        midpoint_change_bps_30s,
        spread_bps_delta_30s,
        depth_20_change_bps_30s,
        imbalance_20_delta_30s,
        midpoint_change_bps_60s,
        spread_bps_delta_60s,
        depth_20_change_bps_60s,
        imbalance_20_delta_60s,
    );
    feature
}

fn decompress_hour_blocking(
    config: &CryptoHftBinanceL2Config,
    manifest: &HourlyArchiveManifest,
    cancellation: &ArchiveCancellation,
) -> Result<TemporaryParquetFile> {
    check_cancelled(cancellation)?;
    std_fs::create_dir_all(&config.temporary_directory)?;
    let stem = Path::new(&manifest.remote_file)
        .file_name()
        .and_then(|value| value.to_str())
        .context("CryptoHFT remote filename was not UTF-8")?
        .trim_end_matches(".zst");
    let partial_path = config
        .temporary_directory
        .join(format!(".{stem}.{}.part", Uuid::new_v4()));
    let final_path = partial_path.with_extension("parquet");
    let _cleanup = RemoveOnDrop(partial_path.clone());

    let input = std_fs::File::open(&manifest.archive_path)?;
    let mut decoder =
        zstd::stream::read::Decoder::new(BufReader::with_capacity(IO_BUFFER_BYTES, input))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial_path)?;
    let mut buffer = vec![0u8; IO_BUFFER_BYTES];
    let mut decoded_bytes = 0u64;
    loop {
        check_cancelled(cancellation)?;
        let read = decoder.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        decoded_bytes = decoded_bytes
            .checked_add(u64::try_from(read).context("decoded chunk length overflow")?)
            .context("CryptoHFT decoded archive length overflow")?;
        if decoded_bytes > config.maximum_decoded_bytes {
            bail!("CryptoHFT archive exceeded the decoded size limit");
        }
        output.write_all(&buffer[..read])?;
    }
    output.sync_all()?;
    drop(output);
    if decoded_bytes < 8 {
        bail!("CryptoHFT decoded Parquet file was too short");
    }
    validate_parquet_magic(&partial_path)?;
    check_cancelled(cancellation)?;
    std_fs::rename(&partial_path, &final_path)?;
    Ok(TemporaryParquetFile {
        path: final_path,
        decoded_bytes,
    })
}

fn validate_parquet_magic(path: &Path) -> Result<()> {
    let mut file = std_fs::File::open(path)?;
    let mut leading = [0u8; 4];
    let mut trailing = [0u8; 4];
    file.read_exact(&mut leading)?;
    file.seek(SeekFrom::End(-4))?;
    file.read_exact(&mut trailing)?;
    if leading != *b"PAR1" || trailing != *b"PAR1" {
        bail!("CryptoHFT decoded object was not a complete Parquet file");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(root: &Path) -> CryptoHftBinanceL2Config {
        CryptoHftBinanceL2Config::new(root.join("archive"), root.join("working"))
    }

    fn hourly_manifest(
        config: &CryptoHftBinanceL2Config,
        date: NaiveDate,
        hour: u8,
        sha256: String,
        compressed_bytes: u64,
        snapshot_events: u64,
        validated_snapshot_events: u64,
        audited_anchor_snapshot_verified: bool,
    ) -> HourlyArchiveManifest {
        let spec = CryptoHftHourlySpec::new(config, date, hour).unwrap();
        let start = date.and_hms_opt(u32::from(hour), 0, 0).unwrap().and_utc();
        HourlyArchiveManifest {
            provider: CRYPTOHFT_ARCHIVE_PROVIDER.to_owned(),
            source_uri: spec.source_uri,
            logical_key: spec.logical_key,
            remote_file: spec.remote_file,
            archive_path: spec.archive_path,
            sha256,
            compressed_bytes,
            raw_rows: 1,
            snapshot_events,
            validated_snapshot_events,
            audited_anchor_snapshot_verified,
            update_events: 1,
            minimum_provider_received_at: start,
            maximum_provider_received_at: start + TimeDelta::minutes(59),
            receipt_rows_outside_exact_hour: 0,
            downloaded_at: start,
            reused_archive: false,
            vendor_checksum: None,
            integrity_basis: "locally_computed_sha256_no_vendor_checksum".to_owned(),
        }
    }

    #[test]
    fn config_requires_absolute_storage_paths_and_safe_limits() {
        let temporary = tempfile::tempdir().unwrap();
        let config = test_config(temporary.path());
        assert!(config.validate().is_ok());

        let mut relative = config.clone();
        relative.archive_root = PathBuf::from("relative/archive");
        assert!(relative.validate().is_err());

        let mut low_headroom = config.clone();
        low_headroom.minimum_free_fraction = 0.249;
        assert!(low_headroom.validate().is_err());

        let mut unsafe_rate = config;
        unsafe_rate.request_minimum_interval = Duration::from_millis(999);
        assert!(unsafe_rate.validate().is_err());
    }

    #[test]
    fn downloaded_content_length_uses_the_initial_response_size() {
        assert!(validate_downloaded_content_length(Some(24_232_760), 24_232_760).is_ok());
        assert!(validate_downloaded_content_length(None, 24_232_760).is_ok());
        assert!(validate_downloaded_content_length(Some(0), 24_232_760).is_err());
    }

    #[test]
    fn hourly_validation_uses_effective_causal_time_for_partition_membership() {
        let expected_start = NaiveDate::from_ymd_opt(2026, 6, 30)
            .unwrap()
            .and_hms_opt(9, 0, 0)
            .unwrap()
            .and_utc();
        let tolerated_start = expected_start - TimeDelta::seconds(1);
        let tolerated_end = expected_start + TimeDelta::hours(1) + TimeDelta::seconds(1);
        let early_receipt = DateTime::parse_from_rfc3339("2026-06-30T08:51:13.791143607Z")
            .unwrap()
            .with_timezone(&Utc);
        let in_hour_event = DateTime::parse_from_rfc3339("2026-06-30T09:00:16.651Z")
            .unwrap()
            .with_timezone(&Utc);

        assert!(validate_hourly_row_causal_time(
            early_receipt,
            in_hour_event,
            tolerated_start,
            tolerated_end,
        )
        .is_ok());

        let wrong_hour = validate_hourly_row_causal_time(
            early_receipt,
            early_receipt,
            tolerated_start,
            tolerated_end,
        )
        .unwrap_err()
        .to_string();
        assert!(wrong_hour.contains("causal time"));
        assert!(wrong_hour.contains("receipt time"));
        assert!(wrong_hour.contains("event time"));

        let tolerated_tail = expected_start + TimeDelta::hours(1) + TimeDelta::milliseconds(999);
        assert!(validate_hourly_row_causal_time(
            tolerated_tail,
            in_hour_event,
            tolerated_start,
            tolerated_end,
        )
        .is_ok());

        let late_receipt = expected_start + TimeDelta::hours(1) + TimeDelta::seconds(1);
        assert!(validate_hourly_row_causal_time(
            late_receipt,
            in_hour_event,
            tolerated_start,
            tolerated_end,
        )
        .is_err());
    }

    #[test]
    fn hourly_validation_classifies_sparse_boundaries_as_unavailable_coverage() {
        let expected_start = NaiveDate::from_ymd_opt(2026, 5, 18)
            .unwrap()
            .and_hms_opt(10, 0, 0)
            .unwrap()
            .and_utc();
        let tolerated_start = expected_start - TimeDelta::seconds(1);
        let tolerated_end = expected_start + TimeDelta::hours(1) + TimeDelta::seconds(1);

        for observed_at in [
            expected_start + TimeDelta::minutes(5),
            expected_start + TimeDelta::minutes(49),
        ] {
            assert!(validate_hourly_row_causal_time(
                observed_at,
                observed_at,
                tolerated_start,
                tolerated_end,
            )
            .is_ok());
        }
        assert!(validate_hourly_receipt_distribution(2, 2, 0).is_ok());
    }

    #[test]
    fn hourly_validation_retains_receipt_spill_and_in_hour_evidence_gates() {
        assert!(validate_hourly_receipt_distribution(1_000, 999, 1).is_ok());
        assert!(validate_hourly_receipt_distribution(1_000, 998, 2).is_err());
        assert!(validate_hourly_receipt_distribution(1, 0, 1).is_err());
        assert!(validate_hourly_receipt_distribution(4_955_548, 4_954_548, 1_000).is_ok());
        assert!(validate_hourly_receipt_distribution(1_000, 998, 1).is_err());
    }

    #[tokio::test]
    async fn making_an_immutable_file_read_only_is_idempotent() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("immutable-archive");
        std_fs::write(&path, b"archive").unwrap();

        assert_eq!(
            make_read_only(&path).await.unwrap(),
            ReadOnlyDisposition::Updated
        );
        assert_eq!(
            make_read_only(&path).await.unwrap(),
            ReadOnlyDisposition::AlreadyReadOnly
        );
    }

    #[test]
    fn permission_denied_requires_a_proven_read_only_postcondition() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("immutable-archive");
        std_fs::write(&path, b"archive").unwrap();
        let writable = std_fs::metadata(&path).unwrap().permissions();
        let mut read_only = writable.clone();
        read_only.set_readonly(true);

        assert_eq!(
            read_only_postcondition(
                &path,
                io::Error::from(io::ErrorKind::PermissionDenied),
                &read_only,
            )
            .unwrap(),
            ReadOnlyDisposition::AlreadyReadOnly
        );
        let error = read_only_postcondition(
            &path,
            io::Error::from(io::ErrorKind::PermissionDenied),
            &writable,
        )
        .unwrap_err();
        assert!(error.to_string().contains(path.to_str().unwrap()));
    }

    #[test]
    fn hourly_spec_matches_cryptohft_object_identity() {
        let temporary = tempfile::tempdir().unwrap();
        let config = test_config(temporary.path());
        let date = NaiveDate::from_ymd_opt(2026, 3, 21).unwrap();
        let spec = CryptoHftHourlySpec::new(&config, date, 7).unwrap();
        assert_eq!(
            spec.remote_file,
            "binance_futures/2026-03-21/07/BTCUSDT_orderbook.parquet.zst"
        );
        assert_eq!(
            spec.source_uri,
            "https://api.cryptohftdata.com/download?file=binance_futures/2026-03-21/07/BTCUSDT_orderbook.parquet.zst"
        );
        assert_eq!(
            spec.logical_key,
            "cryptohftdata:binance-futures:BTCUSDT:l2:2026-03-21T07"
        );
        assert_eq!(
            spec.archive_path,
            config.archive_root.join(spec.remote_file)
        );
        assert!(CryptoHftHourlySpec::new(&config, date, 24).is_err());
    }

    #[tokio::test]
    async fn matching_cached_manifest_reuses_only_after_compressed_sha_verification() {
        let temporary = tempfile::tempdir().unwrap();
        let config = test_config(temporary.path());
        let date = NaiveDate::from_ymd_opt(2026, 4, 14).unwrap();
        let spec = CryptoHftHourlySpec::new(&config, date, 0).unwrap();
        std_fs::create_dir_all(spec.archive_path.parent().unwrap()).unwrap();
        let compressed = b"intentionally-not-a-zstd-frame";
        std_fs::write(&spec.archive_path, compressed).unwrap();
        let sha256 = format!("{:x}", Sha256::digest(compressed));
        let manifest = hourly_manifest(
            &config,
            date,
            0,
            sha256,
            u64::try_from(compressed.len()).unwrap(),
            0,
            0,
            false,
        );
        let local_manifest_path = manifest_path(&spec).unwrap();
        std_fs::write(
            &local_manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        assert_eq!(
            make_read_only(&spec.archive_path).await.unwrap(),
            ReadOnlyDisposition::Updated
        );
        assert_eq!(
            make_read_only(&local_manifest_path).await.unwrap(),
            ReadOnlyDisposition::Updated
        );

        let reused = reuse_hour(&config, &spec, &ArchiveCancellation::default())
            .await
            .unwrap();

        match reused {
            CachedHourReuse::Reused(reused) => {
                assert!(reused.reused_archive);
                assert_eq!(reused.sha256, manifest.sha256);
            }
            CachedHourReuse::ProvenCorruption(error) => {
                panic!("matching cache was classified as corrupt: {error:#}")
            }
        }
        assert_eq!(
            make_read_only(&spec.archive_path).await.unwrap(),
            ReadOnlyDisposition::AlreadyReadOnly
        );
        assert_eq!(
            make_read_only(&local_manifest_path).await.unwrap(),
            ReadOnlyDisposition::AlreadyReadOnly
        );
    }

    #[tokio::test]
    async fn cached_manifest_sha_mismatch_is_proven_corruption() {
        let temporary = tempfile::tempdir().unwrap();
        let config = test_config(temporary.path());
        let date = NaiveDate::from_ymd_opt(2026, 4, 14).unwrap();
        let spec = CryptoHftHourlySpec::new(&config, date, 0).unwrap();
        std_fs::create_dir_all(spec.archive_path.parent().unwrap()).unwrap();
        let compressed = b"cached-object";
        std_fs::write(&spec.archive_path, compressed).unwrap();
        let manifest = hourly_manifest(
            &config,
            date,
            0,
            "0".repeat(64),
            u64::try_from(compressed.len()).unwrap(),
            0,
            0,
            false,
        );
        std_fs::write(
            manifest_path(&spec).unwrap(),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let reused = reuse_hour(&config, &spec, &ArchiveCancellation::default())
            .await
            .unwrap();

        assert!(matches!(reused, CachedHourReuse::ProvenCorruption(_)));
        assert!(spec.archive_path.is_file());
    }

    #[test]
    fn day_parse_context_is_bounded_by_the_pinned_validated_anchor() {
        let temporary = tempfile::tempdir().unwrap();
        let config = test_config(temporary.path());
        let target_date = NaiveDate::from_ymd_opt(2026, 4, 14).unwrap();
        let target_archives = (0..24)
            .map(|hour| hourly_manifest(&config, target_date, hour, "a".repeat(64), 1, 0, 0, false))
            .collect::<Vec<_>>();
        let anchor = hourly_manifest(
            &config,
            NaiveDate::from_ymd_opt(2026, 4, 13).unwrap(),
            23,
            CRYPTOHFT_AUDITED_ANCHOR_SHA256.to_owned(),
            1,
            1,
            1,
            true,
        );
        let request = CryptoHftDayParseRequest {
            target_date,
            context_archives: vec![anchor],
            target_archives,
            output_batch_rows: 1_000,
            cancellation: ArchiveCancellation::default(),
        };
        assert!(validate_day_parse_request(&request).is_ok());

        let mut unvalidated = request.clone();
        unvalidated.context_archives[0].validated_snapshot_events = 0;
        assert!(validate_day_parse_request(&unvalidated).is_err());

        let mut wrong_checksum = request.clone();
        wrong_checksum.context_archives[0].sha256 = "0".repeat(64);
        assert!(validate_day_parse_request(&wrong_checksum).is_err());

        let mut before_anchor = request;
        before_anchor.context_archives.insert(
            0,
            hourly_manifest(
                &config,
                NaiveDate::from_ymd_opt(2026, 4, 13).unwrap(),
                22,
                "b".repeat(64),
                1,
                0,
                0,
                false,
            ),
        );
        assert!(validate_day_parse_request(&before_anchor).is_err());
    }

    #[test]
    fn maximum_context_chain_reaches_the_last_approved_target_day() {
        let anchor =
            DateTime::<Utc>::from_timestamp(CRYPTOHFT_EARLIEST_CONTEXT_HOUR_EPOCH, 0).unwrap();
        let last_context = NaiveDate::from_ymd_opt(2026, 8, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            - TimeDelta::hours(1);
        let inclusive_hours =
            usize::try_from(last_context.signed_duration_since(anchor).num_hours() + 1).unwrap();

        assert_eq!(inclusive_hours, MAX_CONTEXT_LOOKBACK_HOURS);
        assert_eq!(inclusive_hours, 2_617);
    }

    fn replay_for_test(
        target_date: NaiveDate,
    ) -> (DayReplay, mpsc::Receiver<Vec<BinanceL2OneSecondFeature>>) {
        let temporary = tempfile::tempdir().unwrap();
        let config = test_config(temporary.path());
        let (sender, receiver) = mpsc::channel(4);
        let replay = DayReplay::new(
            &config,
            target_date,
            1_000,
            sender,
            ArchiveCancellation::default(),
        )
        .unwrap();
        (replay, receiver)
    }

    fn event_key(
        timestamp: DateTime<Utc>,
        event_type: EventType,
        first_update_id: Option<i64>,
        final_update_id: Option<i64>,
        previous_final_update_id: Option<i64>,
        last_update_id: Option<i64>,
    ) -> EventKey {
        EventKey {
            received_time_ns: timestamp.timestamp_nanos_opt().unwrap(),
            event_time_ms: timestamp.timestamp_millis(),
            transaction_time_ms: Some(timestamp.timestamp_millis()),
            event_type,
            first_update_id,
            final_update_id,
            previous_final_update_id,
            last_update_id,
        }
    }

    fn snapshot(timestamp: DateTime<Utc>, sequence: i64) -> LogicalEvent {
        let key = event_key(
            timestamp,
            EventType::Snapshot,
            None,
            Some(sequence),
            None,
            Some(sequence),
        );
        let mut event: Option<LogicalEvent> = None;
        for index in 0..20 {
            for level in [
                PriceLevel {
                    side: BookSide::Bid,
                    price: Decimal::from(100 - index),
                    quantity: Decimal::ONE,
                },
                PriceLevel {
                    side: BookSide::Ask,
                    price: Decimal::from(101 + index),
                    quantity: Decimal::ONE,
                },
            ] {
                match event.as_mut() {
                    Some(event) => event.push_level(level).unwrap(),
                    None => event = Some(LogicalEvent::new(key.clone(), level)),
                }
            }
        }
        event.unwrap()
    }

    fn full_depth_snapshot(timestamp: DateTime<Utc>, sequence: i64) -> LogicalEvent {
        let key = event_key(
            timestamp,
            EventType::Snapshot,
            None,
            Some(sequence),
            None,
            Some(sequence),
        );
        let mut event: Option<LogicalEvent> = None;
        for index in 0..500 {
            for level in [
                PriceLevel {
                    side: BookSide::Bid,
                    price: Decimal::from(50_000 - index),
                    quantity: Decimal::ONE,
                },
                PriceLevel {
                    side: BookSide::Ask,
                    price: Decimal::from(50_001 + index),
                    quantity: Decimal::ONE,
                },
            ] {
                match event.as_mut() {
                    Some(event) => event.push_level(level).unwrap(),
                    None => event = Some(LogicalEvent::new(key.clone(), level)),
                }
            }
        }
        event.unwrap()
    }

    fn update(
        timestamp: DateTime<Utc>,
        sequence: i64,
        previous_sequence: i64,
        quantity: i64,
    ) -> LogicalEvent {
        LogicalEvent::new(
            event_key(
                timestamp,
                EventType::Update,
                Some(sequence),
                Some(sequence),
                Some(previous_sequence),
                None,
            ),
            PriceLevel {
                side: BookSide::Bid,
                price: Decimal::from(100),
                quantity: Decimal::from(quantity),
            },
        )
    }

    fn update_range(
        timestamp: DateTime<Utc>,
        first_sequence: i64,
        final_sequence: i64,
        previous_sequence: i64,
        quantity: i64,
    ) -> LogicalEvent {
        LogicalEvent::new(
            event_key(
                timestamp,
                EventType::Update,
                Some(first_sequence),
                Some(final_sequence),
                Some(previous_sequence),
                None,
            ),
            PriceLevel {
                side: BookSide::Bid,
                price: Decimal::from(100),
                quantity: Decimal::from(quantity),
            },
        )
    }

    #[test]
    fn updates_never_bootstrap_without_a_snapshot() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 21).unwrap();
        let start = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        let (mut replay, mut receiver) = replay_for_test(date);
        replay.process_event(update(start, 101, 100, 2)).unwrap();
        let summary = replay.finish().unwrap();

        assert_eq!(summary.updates_before_snapshot, 1);
        assert_eq!(summary.emitted_feature_rows, 0);
        assert!(summary.no_snapshot_bootstrap);
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn sequence_gap_invalidates_until_a_new_snapshot() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 21).unwrap();
        let start = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        let (mut replay, _receiver) = replay_for_test(date);
        replay.process_event(snapshot(start, 100)).unwrap();
        replay
            .process_event(update(start + TimeDelta::milliseconds(500), 100, 99, 9))
            .unwrap();
        assert!(replay.book_ready);
        assert_eq!(replay.summary.sequence_gaps, 0);
        replay
            .process_event(update_range(start + TimeDelta::seconds(1), 100, 101, 99, 2))
            .unwrap();
        replay
            .process_event(update(start + TimeDelta::seconds(2), 105, 104, 3))
            .unwrap();
        assert!(!replay.book_ready);
        assert!(replay.bids.is_empty());
        replay
            .process_event(update(start + TimeDelta::seconds(3), 106, 105, 4))
            .unwrap();
        assert_eq!(replay.summary.sequence_gaps, 1);
        assert_eq!(replay.summary.updates_before_snapshot, 2);

        replay
            .process_event(snapshot(start + TimeDelta::seconds(4), 200))
            .unwrap();
        assert!(replay.book_ready);
        assert_eq!(replay.last_sequence, Some(200));
    }

    #[test]
    fn availability_offset_controls_the_causal_second() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 21).unwrap();
        let start = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        let event_time = start + TimeDelta::milliseconds(950);
        let (mut replay, _receiver) = replay_for_test(date);
        replay.process_event(snapshot(event_time, 100)).unwrap();

        let pending = replay.pending_second.as_ref().unwrap();
        let state = pending.latest_state.as_ref().unwrap();
        assert_eq!(pending.second_start, start + TimeDelta::seconds(1));
        assert_eq!(state.available_at, start + TimeDelta::milliseconds(1_050));
        assert!(state.available_at > state.source_event_timestamp);
    }

    #[test]
    fn early_provider_receipt_never_moves_availability_before_event_time() {
        let date = NaiveDate::from_ymd_opt(2026, 6, 30).unwrap();
        let event_at = date.and_hms_milli_opt(9, 0, 16, 651).unwrap().and_utc();
        let received_at = DateTime::parse_from_rfc3339("2026-06-30T08:51:13.791143607Z")
            .unwrap()
            .with_timezone(&Utc);
        let (mut replay, _receiver) = replay_for_test(date);
        let mut event = snapshot(event_at, 100);
        event.key.received_time_ns = received_at.timestamp_nanos_opt().unwrap();

        replay.process_event(event).unwrap();

        let state = replay
            .pending_second
            .as_ref()
            .unwrap()
            .latest_state
            .as_ref()
            .unwrap();
        assert_eq!(state.provider_received_at, received_at);
        assert_eq!(state.source_event_timestamp, event_at);
        assert_eq!(state.available_at, event_at + TimeDelta::milliseconds(100));
    }

    #[test]
    fn continuous_sixty_second_history_emits_only_qualified_rows() {
        let date = NaiveDate::from_ymd_opt(2026, 3, 21).unwrap();
        let start = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        let (mut replay, mut receiver) = replay_for_test(date);
        replay.process_event(snapshot(start, 100)).unwrap();
        for second in 1..=61 {
            let event = if second == 1 {
                update_range(start + TimeDelta::seconds(second), 100, 101, 99, 2)
            } else {
                update(
                    start + TimeDelta::seconds(second),
                    100 + second,
                    99 + second,
                    if second % 2 == 0 { 1 } else { 2 },
                )
            };
            replay.process_event(event).unwrap();
        }

        assert_eq!(replay.summary.emitted_feature_rows, 1);
        assert_eq!(
            replay.output_batch[0].second_start,
            start + TimeDelta::seconds(60)
        );
        assert_eq!(replay.output_batch[0].quality_status, "qualified");
        assert!(replay.output_batch[0].microprice.scale() <= PERSISTED_DECIMAL_SCALE);
        assert!(replay.output_batch[0].spread_bps.scale() <= PERSISTED_DECIMAL_SCALE);
        assert_eq!(
            replay.output_batch[0].bid_quote_churn_1s,
            Decimal::from(100)
        );
        let summary = replay.finish().unwrap();
        assert_eq!(summary.emitted_feature_rows, 2);
        let batch = receiver.try_recv().unwrap();
        assert_eq!(batch.len(), 2);
        assert!(batch
            .windows(2)
            .all(|rows| rows[0].second_start < rows[1].second_start));
    }

    #[test]
    fn multi_minute_source_gap_emits_no_rows_and_resets_rolling_history() {
        let date = NaiveDate::from_ymd_opt(2026, 5, 18).unwrap();
        let start = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        let (mut replay, _receiver) = replay_for_test(date);
        replay.process_event(snapshot(start, 100)).unwrap();
        for second in 1..=61 {
            let event = if second == 1 {
                update_range(start + TimeDelta::seconds(second), 100, 101, 99, 2)
            } else {
                update(
                    start + TimeDelta::seconds(second),
                    100 + second,
                    99 + second,
                    2,
                )
            };
            replay.process_event(event).unwrap();
        }
        assert_eq!(replay.summary.emitted_feature_rows, 1);

        replay
            .process_event(update(start + TimeDelta::minutes(10), 162, 161, 3))
            .unwrap();
        replay
            .process_event(update(
                start + TimeDelta::minutes(10) + TimeDelta::seconds(1),
                163,
                162,
                4,
            ))
            .unwrap();

        assert_eq!(replay.summary.sequence_gaps, 0);
        assert_eq!(replay.summary.emitted_feature_rows, 2);
        assert_eq!(replay.output_batch.len(), 2);
        assert_eq!(replay.rolling.len(), 1);
        assert_eq!(
            replay.rolling.back().unwrap().second_start,
            start + TimeDelta::minutes(10)
        );
    }

    #[test]
    fn futures_updates_bridge_snapshot_then_follow_previous_final_id() {
        let date = NaiveDate::from_ymd_opt(2026, 4, 14).unwrap();
        let start = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        let (mut replay, _receiver) = replay_for_test(date);
        replay.process_event(snapshot(start, 100)).unwrap();

        replay
            .process_event(update_range(
                start + TimeDelta::milliseconds(100),
                98,
                105,
                90,
                2,
            ))
            .unwrap();
        assert_eq!(replay.last_sequence, Some(105));
        assert!(!replay.awaiting_snapshot_bridge);

        replay
            .process_event(update_range(
                start + TimeDelta::milliseconds(200),
                110,
                120,
                105,
                3,
            ))
            .unwrap();
        assert_eq!(replay.last_sequence, Some(120));
        assert_eq!(replay.summary.sequence_gaps, 0);
    }

    #[test]
    fn logical_event_identity_ignores_per_level_receive_jitter() {
        let timestamp = NaiveDate::from_ymd_opt(2026, 4, 14)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        let first = event_key(
            timestamp,
            EventType::Update,
            Some(10),
            Some(12),
            Some(9),
            None,
        );
        let mut second = first.clone();
        second.received_time_ns -= 50_000;
        assert!(first.same_logical_event(&second));
        assert_ne!(first, second);
    }

    #[test]
    fn logical_event_deduplicates_an_identical_side_and_price_level() {
        let timestamp = NaiveDate::from_ymd_opt(2026, 4, 14)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        let mut event = update(timestamp, 101, 100, 2);
        let duplicate = event.levels.values().next().unwrap().clone();

        event.push_level(duplicate).unwrap();

        assert_eq!(event.levels.len(), 1);
        assert_eq!(
            event.levels.values().next().unwrap().quantity,
            Decimal::from(2)
        );
    }

    #[test]
    fn logical_event_rejects_a_conflicting_quantity_for_the_same_level() {
        let timestamp = NaiveDate::from_ymd_opt(2026, 4, 14)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        let mut event = update(timestamp, 101, 100, 2);

        let error = event
            .push_level(PriceLevel {
                side: BookSide::Bid,
                price: Decimal::from(100),
                quantity: Decimal::from(3),
            })
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("repeated a price level with conflicting quantity"));
        assert_eq!(event.levels.len(), 1);
        assert_eq!(
            event.levels.values().next().unwrap().quantity,
            Decimal::from(2)
        );
    }

    #[test]
    fn source_style_snapshot_accepts_matching_final_and_last_ids_only() {
        let date = NaiveDate::from_ymd_opt(2026, 4, 14).unwrap();
        let start = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        let (mut accepted, _receiver) = replay_for_test(date);

        accepted.process_event(snapshot(start, 42)).unwrap();

        assert!(accepted.book_ready);
        assert!(accepted.saw_snapshot);
        assert_eq!(accepted.last_sequence, Some(42));

        let (mut rejected, _receiver) = replay_for_test(date);
        let mut mismatched = snapshot(start, 42);
        mismatched.key.final_update_id = Some(43);
        let error = rejected.process_event(mismatched).unwrap_err();

        assert!(error
            .to_string()
            .contains("snapshot sequence columns were invalid"));
        assert!(!rejected.book_ready);
        assert!(!rejected.saw_snapshot);
    }

    #[test]
    fn april_14_archive_update_bridges_the_observed_snapshot_sequence() {
        let date = NaiveDate::from_ymd_opt(2026, 4, 14).unwrap();
        let snapshot_time = NaiveDate::from_ymd_opt(2026, 4, 13)
            .unwrap()
            .and_hms_milli_opt(23, 5, 19, 649)
            .unwrap()
            .and_utc();
        let (mut replay, _receiver) = replay_for_test(date);
        replay
            .process_event(snapshot(snapshot_time, 10_318_083_192_958))
            .unwrap();

        replay
            .process_event(update_range(
                snapshot_time + TimeDelta::milliseconds(1),
                10_318_083_192_710,
                10_318_083_197_287,
                10_318_083_192_641,
                2,
            ))
            .unwrap();

        assert_eq!(replay.last_sequence, Some(10_318_083_197_287));
        assert!(!replay.awaiting_snapshot_bridge);
        assert_eq!(replay.summary.sequence_gaps, 0);
    }

    #[test]
    fn bridge_ending_at_the_snapshot_sequence_enters_steady_state() {
        let date = NaiveDate::from_ymd_opt(2026, 4, 14).unwrap();
        let start = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        let (mut replay, _receiver) = replay_for_test(date);
        replay.process_event(snapshot(start, 100)).unwrap();

        replay
            .process_event(update_range(
                start + TimeDelta::milliseconds(100),
                98,
                100,
                97,
                2,
            ))
            .unwrap();

        assert_eq!(replay.last_sequence, Some(100));
        assert!(!replay.awaiting_snapshot_bridge);

        replay
            .process_event(update_range(
                start + TimeDelta::milliseconds(200),
                105,
                110,
                100,
                3,
            ))
            .unwrap();
        assert_eq!(replay.last_sequence, Some(110));
        assert_eq!(replay.summary.sequence_gaps, 0);
    }

    #[test]
    fn regressing_provider_receipt_time_clamps_available_at_monotonically() {
        let date = NaiveDate::from_ymd_opt(2026, 4, 14).unwrap();
        let start = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        let (mut replay, _receiver) = replay_for_test(date);
        let mut initial = snapshot(start, 100);
        initial.key.received_time_ns = (start + TimeDelta::milliseconds(500))
            .timestamp_nanos_opt()
            .unwrap();
        replay.process_event(initial).unwrap();
        let initial_available_at = replay.last_available_at.unwrap();

        let mut regressed = update_range(start, 98, 105, 97, 2);
        regressed.key.received_time_ns = (start + TimeDelta::milliseconds(100))
            .timestamp_nanos_opt()
            .unwrap();
        replay.process_event(regressed).unwrap();

        assert_eq!(initial_available_at, start + TimeDelta::milliseconds(600));
        assert_eq!(replay.last_available_at, Some(initial_available_at));
        assert_eq!(
            replay
                .pending_second
                .as_ref()
                .unwrap()
                .latest_state
                .as_ref()
                .unwrap()
                .available_at,
            initial_available_at
        );
    }

    #[test]
    fn shallow_and_crossed_snapshots_never_count_as_valid_bootstraps() {
        let date = NaiveDate::from_ymd_opt(2026, 4, 14).unwrap();
        let start = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        let mut shallow = snapshot(start, 100);
        while shallow.levels.len() > 38 {
            let key = *shallow.levels.keys().next_back().unwrap();
            shallow.levels.remove(&key);
        }
        let mut crossed = snapshot(start, 200);
        crossed
            .push_level(PriceLevel {
                side: BookSide::Ask,
                price: Decimal::from(100),
                quantity: Decimal::ONE,
            })
            .unwrap();

        for invalid in [shallow, crossed] {
            let (mut replay, mut receiver) = replay_for_test(date);
            replay.process_event(invalid).unwrap();

            assert!(!replay.book_ready);
            assert!(!replay.saw_snapshot);
            assert!(replay.bids.is_empty());
            assert!(replay.asks.is_empty());
            assert_eq!(replay.summary.invalid_book_events, 1);

            let summary = replay.finish().unwrap();
            assert!(summary.no_snapshot_bootstrap);
            assert_eq!(summary.emitted_feature_rows, 0);
            assert!(receiver.try_recv().is_err());
        }
    }

    #[test]
    fn hourly_lineage_counts_only_structurally_valid_full_depth_snapshots() {
        let timestamp = NaiveDate::from_ymd_opt(2026, 4, 13)
            .unwrap()
            .and_hms_milli_opt(23, 5, 19, 649)
            .unwrap()
            .and_utc();
        let valid = full_depth_snapshot(timestamp, CRYPTOHFT_AUDITED_SNAPSHOT_LAST_UPDATE_ID);
        assert_eq!(valid.levels.len(), MIN_FULL_DEPTH_SNAPSHOT_LEVELS);
        assert!(is_valid_full_depth_snapshot(&valid).unwrap());

        let mut shallow = valid.clone();
        shallow.levels.pop_last();
        assert!(!is_valid_full_depth_snapshot(&shallow).unwrap());

        let mut crossed = valid;
        crossed
            .push_level(PriceLevel {
                side: BookSide::Ask,
                price: Decimal::from(50_000),
                quantity: Decimal::ONE,
            })
            .unwrap();
        assert!(!is_valid_full_depth_snapshot(&crossed).unwrap());
    }

    #[test]
    fn representative_gate_accepts_classified_gaps_and_requires_complete_lineage() {
        let manifest_object =
            |remote_file: &str,
             sha256: String,
             snapshot_events: u64,
             validated_snapshot_events: u64,
             audited_anchor_snapshot_verified: bool| {
                serde_json::json!({
                    "provider": CRYPTOHFT_ARCHIVE_PROVIDER,
                    "remote_file": remote_file,
                    "sha256": sha256,
                    "compressed_bytes": 1,
                    "raw_rows": 1,
                    "snapshot_events": snapshot_events,
                    "validated_snapshot_events": validated_snapshot_events,
                    "audited_anchor_snapshot_verified": audited_anchor_snapshot_verified,
                    "update_events": 1,
                })
            };
        let mut metadata = serde_json::json!({
            "materialization_contract": BINANCE_L2_MATERIALIZATION_CONTRACT,
            "source_objects": 24,
            "hourly_manifest": {
                "context": [manifest_object(
                    CRYPTOHFT_AUDITED_ANCHOR_REMOTE_FILE,
                    CRYPTOHFT_AUDITED_ANCHOR_SHA256.to_owned(),
                    1,
                    1,
                    true,
                )],
                "target": (0..24).map(|hour| manifest_object(
                    &format!("binance_futures/2026-04-14/{hour:02}/BTCUSDT_orderbook.parquet.zst"),
                    "a".repeat(64),
                    0,
                    0,
                    false,
                )).collect::<Vec<_>>(),
            },
            "qualified_seconds": 40_239,
            "unavailable_seconds": 46_161,
            "snapshots": 1,
            "sequence_gaps": 1,
            "invalid_book_events": 0,
            "no_snapshot_bootstrap": false,
        });
        assert!(validate_representative_day_quality(&metadata).is_ok());

        let mut missing_gap_metric = metadata.clone();
        missing_gap_metric
            .as_object_mut()
            .unwrap()
            .remove("sequence_gaps");
        assert!(validate_representative_day_quality(&missing_gap_metric).is_err());

        metadata["qualified_seconds"] = serde_json::json!(0);
        metadata["unavailable_seconds"] = serde_json::json!(86_400);
        assert!(validate_representative_day_quality(&metadata).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "requires the audited CryptoHFT April 13/14 Parquet fixtures"]
    async fn audited_april_14_boundary_replays_without_a_sequence_gap() {
        let Some(fixture_root) =
            std::env::var_os("CRYPTOHFT_APRIL14_FIXTURE_DIR").map(PathBuf::from)
        else {
            eprintln!("skipping audited fixture replay: CRYPTOHFT_APRIL14_FIXTURE_DIR is not set");
            return;
        };
        let paths = [
            fixture_root.join("BTCUSDT_orderbook_2026-04-13_23.parquet"),
            fixture_root.join("BTCUSDT_orderbook_2026-04-14_00.parquet"),
            fixture_root.join("BTCUSDT_orderbook_2026-04-14_01.parquet"),
        ];
        assert!(paths.iter().all(|path| path.is_file()));
        let validation_paths = paths.clone();
        tokio::task::spawn_blocking(move || {
            let cancellation = ArchiveCancellation::default();
            validate_hourly_parquet_payload(
                &validation_paths[0],
                NaiveDate::from_ymd_opt(2026, 4, 13).unwrap(),
                23,
                &cancellation,
            )?;
            validate_hourly_parquet_payload(
                &validation_paths[1],
                NaiveDate::from_ymd_opt(2026, 4, 14).unwrap(),
                0,
                &cancellation,
            )?;
            validate_hourly_parquet_payload(
                &validation_paths[2],
                NaiveDate::from_ymd_opt(2026, 4, 14).unwrap(),
                1,
                &cancellation,
            )?;
            Ok::<_, anyhow::Error>(())
        })
        .await
        .unwrap()
        .unwrap();

        let temporary = tempfile::tempdir().unwrap();
        let config = test_config(temporary.path());
        let target_date = NaiveDate::from_ymd_opt(2026, 4, 14).unwrap();
        let target_start = target_date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        let target_end = target_start + TimeDelta::days(1);
        let (sender, mut receiver) = mpsc::channel(4);
        let parser = tokio::task::spawn_blocking(move || {
            let mut replay = DayReplay::new(
                &config,
                target_date,
                1_000,
                sender,
                ArchiveCancellation::default(),
            )?;
            for path in paths {
                replay.parse_parquet(&path)?;
            }
            replay.finish()
        });

        let mut emitted = 0u64;
        while let Some(batch) = receiver.recv().await {
            assert!(batch.iter().all(|row| {
                row.second_start >= target_start
                    && row.second_start < target_end
                    && row.available_at >= row.second_start
                    && row.available_at < row.second_start + TimeDelta::seconds(1)
            }));
            emitted = emitted.saturating_add(u64::try_from(batch.len()).unwrap());
        }
        let summary = parser.await.unwrap().unwrap();
        assert_eq!(summary.snapshot_events, 1);
        assert_eq!(summary.sequence_gaps, 0);
        assert!(!summary.no_snapshot_bootstrap);
        assert!(summary.emitted_feature_rows > 6_000);
        assert_eq!(emitted, summary.emitted_feature_rows);
    }
}
