use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Days, NaiveDate, Timelike, Utc};
use futures_util::{stream, StreamExt};
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::RowAccessor,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    fs,
    io::AsyncWriteExt,
    sync::{watch, Mutex},
    time::{sleep, Instant},
};
use tracing::{info, warn};
use uuid::Uuid;

pub const BASE_URL: &str = "https://api.cryptohftdata.com";
pub const EXCHANGE: &str = "kraken_spot";
pub const SYMBOL: &str = "BTC_USD";
pub const DATA_SYMBOL: &str = "BTC/USD";
pub const ARCHIVE_ROOT: &str = "/var/lib/kraken-data/spot-l2/cryptohftdata";
pub const END_EXCLUSIVE: &str = "2026-08-31";
pub const START_TIERS: [&str; 3] = ["2026-04-01", "2026-05-01", "2026-06-01"];
pub const REQUEST_INTERVAL: Duration = Duration::from_millis(1_100);
pub const DOWNLOAD_CONCURRENCY: usize = 4;
const MAXIMUM_HOUR_ATTEMPTS: u32 = 3;
const MAXIMUM_COMPRESSED_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAXIMUM_DECODED_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const REQUIRED_COLUMNS: [&str; 13] = [
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArchiveManifest {
    pub provider: String,
    pub exchange: String,
    pub symbol: String,
    pub source_uri: String,
    pub source_date: NaiveDate,
    pub source_hour: u8,
    pub sha256: String,
    pub compressed_bytes: u64,
    pub decoded_bytes: u64,
    pub parquet_rows: u64,
    pub archived_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkippedHourManifest {
    pub provider: String,
    pub exchange: String,
    pub symbol: String,
    pub source_uri: String,
    pub source_date: NaiveDate,
    pub source_hour: u8,
    pub http_status: Option<u16>,
    pub attempts: u32,
    pub error: String,
    pub confirmed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HourSpec {
    pub date: NaiveDate,
    pub hour: u8,
    pub remote_file: String,
    pub source_uri: String,
    pub final_path: PathBuf,
    pub manifest_path: PathBuf,
}

impl HourSpec {
    pub fn new(root: &Path, date: NaiveDate, hour: u8) -> Result<Self> {
        if !root.is_absolute() {
            bail!("Kraken Spot L2 archive root must be absolute");
        }
        if hour > 23 {
            bail!("Kraken Spot L2 archive hour must be between 0 and 23");
        }
        let date_text = date.format("%Y-%m-%d");
        let remote_file = format!("{EXCHANGE}/{date_text}/{hour:02}/{SYMBOL}_orderbook.parquet");
        let final_path = root
            .join(EXCHANGE)
            .join(date_text.to_string())
            .join(format!("{hour:02}"))
            .join(format!("{SYMBOL}_orderbook.parquet"));
        let manifest_path =
            final_path.with_file_name(format!("{}_orderbook.parquet.manifest.json", SYMBOL));
        Ok(Self {
            date,
            hour,
            source_uri: format!("{BASE_URL}/download?file={remote_file}"),
            remote_file,
            final_path,
            manifest_path,
        })
    }
}

#[derive(Clone)]
pub struct KrakenSpotL2ArchiveWorker {
    client: reqwest::Client,
    root: PathBuf,
    next_request_at: Arc<Mutex<Instant>>,
}

impl KrakenSpotL2ArchiveWorker {
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(15 * 60))
            .user_agent("polymarket-kraken-spot-l2-archive/1")
            .build()
            .context("failed to build Kraken Spot L2 archive client")?;
        Ok(Self {
            client,
            root: PathBuf::from(ARCHIVE_ROOT),
            next_request_at: Arc::new(Mutex::new(Instant::now())),
        })
    }

    #[cfg(test)]
    pub fn with_root(root: PathBuf) -> Result<Self> {
        let mut worker = Self::new()?;
        worker.root = root;
        Ok(worker)
    }

    pub async fn run(&self, mut shutdown: watch::Receiver<bool>) -> Result<()> {
        fs::create_dir_all(&self.root)
            .await
            .with_context(|| format!("failed to create {}", self.root.display()))?;
        let cleanup_root = self.root.clone();
        let removed_partials =
            tokio::task::spawn_blocking(move || cleanup_partial_files(&cleanup_root))
                .await
                .context("Kraken Spot L2 partial-file cleanup panicked")??;
        if removed_partials > 0 {
            info!(
                removed_partials,
                "removed interrupted Kraken Spot L2 work files"
            );
        }
        let start = self.select_start_tier(&mut shutdown).await?;
        let end = parse_date(END_EXCLUSIVE)?;
        info!(%start, %end, root = %self.root.display(), "Kraken Spot L2 archive backfill started");

        let mut instant = start.and_hms_opt(0, 0, 0).unwrap().and_utc();
        let end_instant = end.and_hms_opt(0, 0, 0).unwrap().and_utc();
        let mut specs = Vec::with_capacity(usize::try_from(expected_hours(start)?)?);
        while instant < end_instant {
            specs.push(HourSpec::new(
                &self.root,
                instant.date_naive(),
                u8::try_from(instant.hour()).context("invalid UTC hour")?,
            )?);
            instant += chrono::Duration::hours(1);
        }

        let worker = self.clone();
        let operations = stream::iter(specs)
            .map(move |spec| {
                let worker = worker.clone();
                let shutdown = shutdown.clone();
                async move { worker.archive_with_retry(spec, shutdown).await }
            })
            .buffer_unordered(DOWNLOAD_CONCURRENCY);
        tokio::pin!(operations);
        while let Some(result) = operations.next().await {
            if !result? {
                info!("Kraken Spot L2 archive worker stopped cleanly");
                return Ok(());
            }
        }
        info!(%start, %end, "Kraken Spot L2 archive backfill completed");
        Ok(())
    }

    async fn archive_with_retry(
        &self,
        spec: HourSpec,
        mut shutdown: watch::Receiver<bool>,
    ) -> Result<bool> {
        let mut attempt = 0u32;
        loop {
            if *shutdown.borrow() {
                return Ok(false);
            }
            match self.archive_hour(&spec).await {
                Ok(reused) => {
                    info!(date = %spec.date, hour = spec.hour, reused, "Kraken Spot L2 hour archived");
                    return Ok(true);
                }
                Err(error) => {
                    attempt = attempt.saturating_add(1);
                    if attempt >= MAXIMUM_HOUR_ATTEMPTS {
                        persist_skipped_hour(&spec, attempt, &error).await?;
                        warn!(date = %spec.date, hour = spec.hour, attempt, error = %error, "Kraken Spot L2 hour skipped after bounded retries; continuing backfill");
                        return Ok(true);
                    }
                    let delay = Duration::from_secs(5 * u64::from(attempt.min(12)));
                    warn!(date = %spec.date, hour = spec.hour, attempt, error = %error, retry_seconds = delay.as_secs(), "Kraken Spot L2 hour failed; retrying");
                    tokio::select! {
                        _ = sleep(delay) => {}
                        changed = shutdown.changed() => {
                            if changed.is_err() || *shutdown.borrow() {
                                return Ok(false);
                            }
                        }
                    }
                }
            }
        }
    }

    async fn select_start_tier(&self, shutdown: &mut watch::Receiver<bool>) -> Result<NaiveDate> {
        for raw in START_TIERS {
            if *shutdown.borrow() {
                bail!("shutdown requested before Kraken Spot L2 tier selection completed");
            }
            let date = parse_date(raw)?;
            let spec = HourSpec::new(&self.root, date, 0)?;
            match self.archive_hour(&spec).await {
                Ok(_) => {
                    info!(%date, "selected earliest available Kraken Spot L2 backfill tier");
                    return Ok(date);
                }
                Err(error) if is_unavailable(&error) => {
                    warn!(%date, error = %error, "Kraken Spot L2 tier unavailable");
                }
                Err(error) => return Err(error),
            }
            sleep(REQUEST_INTERVAL).await;
        }
        bail!("CryptoHFTData did not expose Kraken Spot BTC/USD L2 at any approved start tier")
    }

    async fn archive_hour(&self, spec: &HourSpec) -> Result<bool> {
        if fs::try_exists(&spec.final_path).await? && fs::try_exists(&spec.manifest_path).await? {
            validate_existing(spec).await?;
            return Ok(true);
        }
        if let Some(parent) = spec.final_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        let compressed_path = spec
            .final_path
            .with_file_name(format!(".{}.zst.part", Uuid::new_v4()));
        let decoded_path = spec
            .final_path
            .with_file_name(format!(".{}.parquet.part", Uuid::new_v4()));
        let result = self
            .download_decode_publish(spec, &compressed_path, &decoded_path)
            .await;
        if result.is_err() {
            let _ = fs::remove_file(&compressed_path).await;
            let _ = fs::remove_file(&decoded_path).await;
        }
        result.map(|_| false)
    }

    async fn download_decode_publish(
        &self,
        spec: &HourSpec,
        compressed_path: &Path,
        decoded_path: &Path,
    ) -> Result<()> {
        self.wait_for_request_slot().await;
        let response = self.client.get(&spec.source_uri).send().await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            bail!("source_unavailable: {} returned 404", spec.remote_file);
        }
        if !response.status().is_success() {
            bail!("{} returned HTTP {}", spec.remote_file, response.status());
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAXIMUM_COMPRESSED_BYTES)
        {
            bail!("{} exceeds the compressed size limit", spec.remote_file);
        }
        let mut output = fs::File::create(compressed_path).await?;
        let mut response = response;
        let mut compressed_bytes = 0u64;
        while let Some(chunk) = response.chunk().await? {
            compressed_bytes = compressed_bytes.saturating_add(chunk.len() as u64);
            if compressed_bytes > MAXIMUM_COMPRESSED_BYTES {
                bail!("{} exceeded the compressed size limit", spec.remote_file);
            }
            output.write_all(&chunk).await?;
        }
        output.flush().await?;
        output.sync_all().await?;
        drop(output);
        if compressed_bytes == 0 {
            bail!("{} returned an empty body", spec.remote_file);
        }

        let compressed = compressed_path.to_owned();
        let decoded = decoded_path.to_owned();
        let (sha256, decoded_bytes, parquet_rows) =
            tokio::task::spawn_blocking(move || decode_and_validate(&compressed, &decoded))
                .await
                .context("Kraken Spot L2 decode task panicked")??;

        fs::rename(decoded_path, &spec.final_path).await?;
        let manifest = ArchiveManifest {
            provider: "cryptohftdata".to_string(),
            exchange: EXCHANGE.to_string(),
            symbol: SYMBOL.to_string(),
            source_uri: spec.source_uri.clone(),
            source_date: spec.date,
            source_hour: spec.hour,
            sha256,
            compressed_bytes,
            decoded_bytes,
            parquet_rows,
            archived_at: Utc::now(),
        };
        persist_manifest(&spec.manifest_path, &manifest).await?;
        let unavailable_path = source_unavailable_path(spec)?;
        if fs::try_exists(&unavailable_path).await? {
            fs::remove_file(unavailable_path).await?;
        }
        fs::remove_file(compressed_path).await?;
        Ok(())
    }

    async fn wait_for_request_slot(&self) {
        let mut next = self.next_request_at.lock().await;
        let now = Instant::now();
        if *next > now {
            sleep(*next - now).await;
        }
        *next = Instant::now() + REQUEST_INTERVAL;
    }
}

fn decode_and_validate(compressed: &Path, decoded: &Path) -> Result<(String, u64, u64)> {
    let input = BufReader::new(File::open(compressed)?);
    let mut decoder = zstd::stream::read::Decoder::new(input)?;
    let output = File::create(decoded)?;
    let mut limited = decoder.by_ref().take(MAXIMUM_DECODED_BYTES + 1);
    let mut output = BufWriter::new(output);
    let decoded_bytes = io::copy(&mut limited, &mut output)?;
    output.flush()?;
    output.get_ref().sync_all()?;
    if decoded_bytes == 0 || decoded_bytes > MAXIMUM_DECODED_BYTES {
        bail!("decoded Kraken Spot L2 object violated the size contract");
    }

    let file = File::open(decoded)?;
    let reader = SerializedFileReader::new(file)?;
    let fields = reader.metadata().file_metadata().schema().get_fields();
    if fields.len() != 12 && fields.len() != 13 {
        bail!("Kraken Spot L2 Parquet exposed {} columns", fields.len());
    }
    for (index, field) in fields.iter().enumerate() {
        if field.name() != REQUIRED_COLUMNS[index] {
            bail!("Kraken Spot L2 Parquet column {index} was {}", field.name());
        }
    }
    let parquet_rows = reader.metadata().file_metadata().num_rows();
    if parquet_rows <= 0 {
        bail!("Kraken Spot L2 Parquet contained no rows");
    }
    let mut rows = reader.get_row_iter(None)?;
    let first = rows
        .next()
        .transpose()?
        .context("Kraken Spot L2 Parquet contained no readable rows")?;
    if first.get_string(3)? != DATA_SYMBOL {
        bail!(
            "Kraken Spot L2 Parquet contained unexpected symbol {}",
            first.get_string(3)?
        );
    }
    if !matches!(first.get_string(4)?.as_str(), "snapshot" | "update") {
        bail!("Kraken Spot L2 Parquet contained an unsupported event type");
    }
    let mut hash = Sha256::new();
    let mut decoded_input = BufReader::new(File::open(decoded)?);
    io::copy(&mut decoded_input, &mut HashWriter(&mut hash))?;
    Ok((
        hex::encode(hash.finalize()),
        decoded_bytes,
        parquet_rows as u64,
    ))
}

struct HashWriter<'a>(&'a mut Sha256);

impl Write for HashWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0.update(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

async fn validate_existing(spec: &HourSpec) -> Result<()> {
    let bytes = fs::read(&spec.manifest_path).await?;
    let manifest: ArchiveManifest = serde_json::from_slice(&bytes)?;
    if manifest.exchange != EXCHANGE
        || manifest.symbol != SYMBOL
        || manifest.source_date != spec.date
        || manifest.source_hour != spec.hour
        || manifest.source_uri != spec.source_uri
    {
        bail!("existing Kraken Spot L2 manifest did not match its immutable source identity");
    }
    let metadata = fs::metadata(&spec.final_path).await?;
    if metadata.len() != manifest.decoded_bytes || manifest.parquet_rows == 0 {
        bail!("existing Kraken Spot L2 archive did not match its manifest");
    }
    Ok(())
}

async fn persist_manifest(path: &Path, manifest: &ArchiveManifest) -> Result<()> {
    let parent = path.parent().context("manifest path had no parent")?;
    let temporary = parent.join(format!(".{}.manifest.part", Uuid::new_v4()));
    let bytes = serde_json::to_vec_pretty(manifest)?;
    let mut file = fs::File::create(&temporary).await?;
    file.write_all(&bytes).await?;
    file.flush().await?;
    file.sync_all().await?;
    drop(file);
    fs::rename(temporary, path).await?;
    Ok(())
}

async fn persist_skipped_hour(spec: &HourSpec, attempts: u32, error: &anyhow::Error) -> Result<()> {
    let path = source_unavailable_path(spec)?;
    let parent = path.parent().context("unavailable marker had no parent")?;
    fs::create_dir_all(parent).await?;
    let temporary = parent.join(format!(".{}.unavailable.part", Uuid::new_v4()));
    let marker = SkippedHourManifest {
        provider: "cryptohftdata".to_string(),
        exchange: EXCHANGE.to_string(),
        symbol: SYMBOL.to_string(),
        source_uri: spec.source_uri.clone(),
        source_date: spec.date,
        source_hour: spec.hour,
        http_status: is_unavailable(error).then_some(404),
        attempts,
        error: error.to_string(),
        confirmed_at: Utc::now(),
    };
    let bytes = serde_json::to_vec_pretty(&marker)?;
    let mut file = fs::File::create(&temporary).await?;
    file.write_all(&bytes).await?;
    file.flush().await?;
    file.sync_all().await?;
    drop(file);
    fs::rename(temporary, path).await?;
    Ok(())
}

fn source_unavailable_path(spec: &HourSpec) -> Result<PathBuf> {
    let file_name = spec
        .final_path
        .file_name()
        .and_then(|value| value.to_str())
        .context("Kraken Spot L2 filename was not UTF-8")?;
    Ok(spec
        .final_path
        .with_file_name(format!("{file_name}.unavailable.json")))
}

fn parse_date(raw: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(raw, "%Y-%m-%d").with_context(|| format!("invalid date {raw}"))
}

fn is_unavailable(error: &anyhow::Error) -> bool {
    error.to_string().starts_with("source_unavailable:")
}

fn cleanup_partial_files(root: &Path) -> Result<u64> {
    if !root.exists() {
        return Ok(0);
    }
    let mut removed = 0u64;
    let mut directories = vec![root.to_owned()];
    while let Some(directory) = directories.pop() {
        for entry in std::fs::read_dir(&directory)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                directories.push(entry.path());
                continue;
            }
            if file_type.is_file()
                && entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with('.') && name.ends_with(".part"))
            {
                std::fs::remove_file(entry.path())?;
                removed = removed.saturating_add(1);
            }
        }
    }
    Ok(removed)
}

pub fn expected_hours(start: NaiveDate) -> Result<u64> {
    let end = parse_date(END_EXCLUSIVE)?;
    let days = end.signed_duration_since(start).num_days();
    if days <= 0 {
        bail!("Kraken Spot L2 range must contain at least one day");
    }
    Ok(u64::try_from(days)? * 24)
}

pub fn latest_required_date() -> Result<NaiveDate> {
    parse_date(END_EXCLUSIVE)?
        .checked_sub_days(Days::new(1))
        .context("Kraken Spot L2 end date underflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immutable_hour_identity_is_kraken_spot_btcusd() {
        let date = NaiveDate::from_ymd_opt(2026, 4, 1).unwrap();
        let spec = HourSpec::new(Path::new("/archive"), date, 7).unwrap();
        assert_eq!(
            spec.remote_file,
            "kraken_spot/2026-04-01/07/BTC_USD_orderbook.parquet"
        );
        assert_eq!(spec.source_uri, "https://api.cryptohftdata.com/download?file=kraken_spot/2026-04-01/07/BTC_USD_orderbook.parquet");
        assert_eq!(
            spec.final_path,
            PathBuf::from("/archive/kraken_spot/2026-04-01/07/BTC_USD_orderbook.parquet")
        );
    }

    #[test]
    fn approved_tiers_and_end_cover_the_requested_range() {
        assert_eq!(START_TIERS, ["2026-04-01", "2026-05-01", "2026-06-01"]);
        assert_eq!(
            latest_required_date().unwrap(),
            NaiveDate::from_ymd_opt(2026, 8, 30).unwrap()
        );
        assert_eq!(
            expected_hours(NaiveDate::from_ymd_opt(2026, 4, 1).unwrap()).unwrap(),
            3_648
        );
        assert_eq!(
            expected_hours(NaiveDate::from_ymd_opt(2026, 5, 1).unwrap()).unwrap(),
            2_928
        );
        assert_eq!(
            expected_hours(NaiveDate::from_ymd_opt(2026, 6, 1).unwrap()).unwrap(),
            2_184
        );
    }

    #[test]
    fn restart_cleanup_removes_only_atomic_work_files() {
        let directory = tempfile::tempdir().unwrap();
        let nested = directory.path().join("kraken_spot/2026-04-01/00");
        std::fs::create_dir_all(&nested).unwrap();
        let partial = nested.join(".interrupted.zst.part");
        let parquet = nested.join("BTC_USD_orderbook.parquet");
        std::fs::write(&partial, b"partial").unwrap();
        std::fs::write(&parquet, b"parquet").unwrap();

        assert_eq!(cleanup_partial_files(directory.path()).unwrap(), 1);
        assert!(!partial.exists());
        assert!(parquet.exists());
    }

    #[test]
    fn unavailable_marker_path_is_hour_specific() {
        let date = NaiveDate::from_ymd_opt(2026, 7, 9).unwrap();
        let spec = HourSpec::new(Path::new("/archive"), date, 21).unwrap();
        assert_eq!(
            source_unavailable_path(&spec).unwrap(),
            PathBuf::from(
                "/archive/kraken_spot/2026-07-09/21/BTC_USD_orderbook.parquet.unavailable.json"
            )
        );
        assert_eq!(MAXIMUM_HOUR_ATTEMPTS, 3);
    }
}
