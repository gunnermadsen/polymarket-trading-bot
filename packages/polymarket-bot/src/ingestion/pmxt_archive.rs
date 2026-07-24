use std::{
    collections::HashSet,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Timelike, Utc};
use parquet::{
    data_type::Decimal as ParquetDecimal,
    file::reader::{FileReader, SerializedFileReader},
    record::{Field, Row},
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

use super::{
    binance_archive::{
        ArchiveCancellation, ArchiveDownloadLimits, ArchiveParseSummary, DownloadedArchive,
    },
    job::{BtcOrderbookArchiveEvent, BtcOrderbookMarketScope},
};

pub const PMXT_ARCHIVE_PROVIDER: &str = "pmxt_v2";
pub const DEFAULT_PMXT_ARCHIVE_URL: &str = "https://r2v2.pmxt.dev";
pub const PMXT_COVERAGE_START_EPOCH: i64 = 1_776_106_800; // 2026-04-13T19:00:00Z

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
        let (sha256, compressed_bytes) =
            hash_file(&final_path, limits.maximum_compressed_bytes, cancellation).await?;
        return Ok(Some(DownloadedArchive {
            path: final_path,
            sha256,
            compressed_bytes,
            reused_cache: true,
        }));
    }

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
    if response
        .content_length()
        .is_some_and(|size| size > limits.maximum_compressed_bytes)
    {
        bail!("PMXT archive exceeded its compressed size limit");
    }

    let partial_path = cache_directory.join(format!("{}.part", spec.file_name));
    let mut output = fs::File::create(&partial_path)
        .await
        .with_context(|| format!("failed to create {}", partial_path.display()))?;
    let mut digest = Sha256::new();
    let mut compressed_bytes = 0u64;
    loop {
        if cancellation.is_cancelled() {
            let _ = fs::remove_file(&partial_path).await;
            bail!("archive operation was cancelled");
        }
        let chunk = timeout(limits.chunk_idle_timeout, response.chunk())
            .await
            .context("PMXT archive response stalled")??;
        let Some(chunk) = chunk else { break };
        compressed_bytes = compressed_bytes
            .saturating_add(u64::try_from(chunk.len()).context("PMXT archive size overflow")?);
        if compressed_bytes > limits.maximum_compressed_bytes {
            let _ = fs::remove_file(&partial_path).await;
            bail!("PMXT archive exceeded its compressed size limit");
        }
        digest.update(&chunk);
        output
            .write_all(&chunk)
            .await
            .context("failed to write PMXT archive cache")?;
    }
    output
        .sync_all()
        .await
        .context("failed to sync PMXT archive cache")?;
    drop(output);
    fs::rename(&partial_path, &final_path)
        .await
        .context("failed to publish PMXT archive cache")?;
    Ok(Some(DownloadedArchive {
        path: final_path,
        sha256: format!("{:x}", digest.finalize()),
        compressed_bytes,
        reused_cache: false,
    }))
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
    let (sender, receiver) = mpsc::channel(1);
    let handle = tokio::task::spawn_blocking(move || {
        let conditions = markets
            .iter()
            .map(|market| market.condition_id.as_str())
            .collect::<HashSet<_>>();
        let assets = markets
            .iter()
            .flat_map(|market| [&market.up_token_id, &market.down_token_id])
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let file = File::open(&path)
            .with_context(|| format!("failed to open PMXT archive {}", path.display()))?;
        let reader =
            SerializedFileReader::new(file).context("failed to initialize PMXT Parquet reader")?;
        let rows = reader
            .get_row_iter(None)
            .context("failed to stream PMXT Parquet rows")?;
        let mut summary = ArchiveParseSummary::default();
        let mut batch = Vec::with_capacity(batch_rows);

        for (ordinal, row) in rows.enumerate() {
            if cancellation.is_cancelled() {
                bail!("archive operation was cancelled");
            }
            let row = row.context("failed to decode PMXT Parquet row")?;
            let source_row_number =
                i64::try_from(ordinal).context("PMXT source row number overflow")?;
            let Some(record) = parse_row(row, source_row_number, &conditions, &assets)? else {
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
                summary.maximum_batch_records = summary.maximum_batch_records.max(batch.len());
                sender
                    .blocking_send(Ok(std::mem::take(&mut batch)))
                    .context("PMXT parser consumer stopped")?;
                batch = Vec::with_capacity(batch_rows);
            }
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
) -> Result<(String, u64)> {
    let mut input = fs::File::open(path).await?;
    let mut digest = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        if cancellation.is_cancelled() {
            bail!("archive operation was cancelled");
        }
        let read = input.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        bytes = bytes.saturating_add(u64::try_from(read)?);
        if bytes > maximum_bytes {
            bail!("PMXT cached archive exceeded its compressed size limit");
        }
        digest.update(&buffer[..read]);
    }
    Ok((format!("{:x}", digest.finalize()), bytes))
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
}
