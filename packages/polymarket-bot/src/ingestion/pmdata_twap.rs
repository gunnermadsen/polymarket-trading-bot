use std::{
    fs::File as StdFile,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use arrow_array::{Array, LargeStringArray, StringArray, TimestampMicrosecondArray};
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rust_decimal::Decimal;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use super::{binance_archive::ArchiveCancellation, job::PmdataChainlinkBtcusdTwapRecord};

pub const PMDATA_TWAP_PROVIDER: &str = "pmdata_chainlink_streams_twap";
pub const DEFAULT_PMDATA_BASE_URL: &str = "https://api.pmdata.dev";
const MAX_ARCHIVE_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct PmdataTwapConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub archive_root: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PmdataTwapWindow {
    Seconds30,
    Seconds60,
}

impl PmdataTwapWindow {
    pub const fn seconds(self) -> i16 {
        match self {
            Self::Seconds30 => 30,
            Self::Seconds60 => 60,
        }
    }

    pub const fn data_type(self) -> &'static str {
        match self {
            Self::Seconds30 => "streams_twap30s",
            Self::Seconds60 => "streams_twap60s",
        }
    }
}

#[derive(Debug)]
pub struct PmdataArchive {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug)]
pub struct PmdataParsedDay {
    pub records: Vec<PmdataChainlinkBtcusdTwapRecord>,
    pub minimum_timestamp: DateTime<Utc>,
    pub maximum_timestamp: DateTime<Utc>,
}

impl PmdataTwapConfig {
    pub fn validate(&self) -> Result<()> {
        if self.base_url.trim().is_empty() {
            bail!("POLYMARKET_PMDATA_BASE_URL must not be empty");
        }
        if self.archive_root.as_os_str().is_empty() {
            bail!("POLYMARKET_PMDATA_ARCHIVE_ROOT must not be empty");
        }
        if self
            .api_key
            .as_deref()
            .is_some_and(|key| key.trim().is_empty())
        {
            bail!("POLYMARKET_PMDATA_API_KEY must be non-empty when configured");
        }
        Ok(())
    }

    pub fn source_uri(&self, date: NaiveDate, window: PmdataTwapWindow) -> String {
        let data_type = window.data_type();
        format!(
            "{}/chainlink/BTCUSD/{data_type}/BTCUSD_{data_type}_{date}.parquet",
            self.base_url.trim_end_matches('/')
        )
    }

    pub fn logical_key(&self, date: NaiveDate, window: PmdataTwapWindow) -> String {
        format!("chainlink:BTCUSD:{}:{date}", window.data_type())
    }

    pub fn archive_path(&self, date: NaiveDate, window: PmdataTwapWindow) -> PathBuf {
        let data_type = window.data_type();
        self.archive_root
            .join("chainlink")
            .join("BTCUSD")
            .join(data_type)
            .join(format!("{:04}", date.year()))
            .join(format!("{:02}", date.month()))
            .join(format!("BTCUSD_{data_type}_{date}.parquet"))
    }

    pub async fn ensure_archive(
        &self,
        client: &reqwest::Client,
        date: NaiveDate,
        window: PmdataTwapWindow,
        expected_sha256: Option<&str>,
        cancellation: &ArchiveCancellation,
    ) -> Result<PmdataArchive> {
        self.validate()?;
        let final_path = self.archive_path(date, window);
        if final_path.is_file() {
            return verify_archive(&final_path, expected_sha256).await;
        }
        let api_key = self
            .api_key
            .as_deref()
            .context("POLYMARKET_PMDATA_API_KEY is not configured for this worker")?;
        if cancellation.is_cancelled() {
            bail!("archive operation was cancelled");
        }

        let parent = final_path
            .parent()
            .context("PMData archive path has no parent")?;
        tokio::fs::create_dir_all(parent).await.with_context(|| {
            format!(
                "failed to create PMData archive directory {}",
                parent.display()
            )
        })?;
        let source_uri = self.source_uri(date, window);
        let response = client
            .get(&source_uri)
            .header("api_key", api_key.trim())
            .send()
            .await
            .with_context(|| format!("failed to request PMData archive {source_uri}"))?
            .error_for_status()
            .with_context(|| format!("PMData rejected archive request {source_uri}"))?;
        if let Some(length) = response.content_length() {
            if length > MAX_ARCHIVE_BYTES as u64 {
                bail!("PMData archive exceeds {MAX_ARCHIVE_BYTES} bytes");
            }
        }
        let body = response
            .bytes()
            .await
            .context("failed to read PMData archive response")?;
        if body.len() > MAX_ARCHIVE_BYTES {
            bail!("PMData archive exceeds {MAX_ARCHIVE_BYTES} bytes");
        }
        validate_parquet_magic(&body)?;
        let sha256 = format!("{:x}", Sha256::digest(&body));
        if expected_sha256.is_some_and(|expected| expected != sha256) {
            bail!("downloaded PMData archive checksum conflicts with completed artifact");
        }

        let partial_path = final_path.with_extension(format!("parquet.part.{}", Uuid::new_v4()));
        let write_result = async {
            let mut file = tokio::fs::File::create(&partial_path)
                .await
                .with_context(|| format!("failed to create {}", partial_path.display()))?;
            file.write_all(&body).await?;
            file.sync_all().await?;
            drop(file);
            tokio::fs::rename(&partial_path, &final_path)
                .await
                .with_context(|| {
                    format!("failed to publish PMData archive {}", final_path.display())
                })?;
            Result::<()>::Ok(())
        }
        .await;
        if write_result.is_err() {
            let _ = tokio::fs::remove_file(&partial_path).await;
        }
        write_result?;
        Ok(PmdataArchive {
            path: final_path,
            sha256,
            bytes: body.len() as u64,
        })
    }
}

pub async fn parse_archive(
    path: PathBuf,
    date: NaiveDate,
    window: PmdataTwapWindow,
    batch_rows: usize,
) -> Result<PmdataParsedDay> {
    tokio::task::spawn_blocking(move || parse_archive_blocking(&path, date, window, batch_rows))
        .await
        .context("PMData Parquet parser task failed")?
}

fn parse_archive_blocking(
    path: &Path,
    date: NaiveDate,
    window: PmdataTwapWindow,
    batch_rows: usize,
) -> Result<PmdataParsedDay> {
    let file = StdFile::open(path)
        .with_context(|| format!("failed to open PMData archive {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("invalid PMData Parquet metadata")?
        .with_batch_size(batch_rows)
        .build()
        .context("failed to build PMData Parquet reader")?;
    let day_start = date
        .and_hms_opt(0, 0, 0)
        .context("invalid PMData source date")?
        .and_utc();
    let day_end = day_start + chrono::Duration::days(1);
    let mut records = Vec::with_capacity(86_400);
    let mut previous_timestamp = None;

    for batch in reader {
        let batch = batch.context("failed to decode PMData Parquet batch")?;
        let schema = batch.schema();
        let observation = timestamp_column(&batch, &schema, "observationsTimestamp")?;
        let received = timestamp_column(&batch, &schema, "receiveMicrosecondTimestamp")?;
        let valid_from = timestamp_column(&batch, &schema, "validFromTimestamp")?;
        let expires_at = timestamp_column(&batch, &schema, "expiresAt")?;
        let price = string_column(&batch, &schema, "price")?;
        let bid = string_column(&batch, &schema, "bid")?;
        let ask = string_column(&batch, &schema, "ask")?;
        let version = string_column(&batch, &schema, "version")?;

        for row in 0..batch.num_rows() {
            let source_timestamp = timestamp_value(observation, row, "observationsTimestamp")?;
            let provider_received_at =
                timestamp_value(received, row, "receiveMicrosecondTimestamp")?;
            let valid_from_timestamp = timestamp_value(valid_from, row, "validFromTimestamp")?;
            let expires_at = timestamp_value(expires_at, row, "expiresAt")?;
            if source_timestamp < day_start || source_timestamp >= day_end {
                bail!("PMData observation timestamp falls outside source date {date}");
            }
            if previous_timestamp.is_some_and(|previous| previous >= source_timestamp) {
                bail!("PMData observation timestamps are not strictly increasing");
            }
            if valid_from_timestamp > source_timestamp || expires_at <= source_timestamp {
                bail!("PMData report validity interval does not contain its observation");
            }
            let full_accuracy_value = price.value(row)?.to_string();
            let unscaled = full_accuracy_value
                .parse::<i128>()
                .with_context(|| format!("invalid PMData scaled price {full_accuracy_value}"))?;
            let twap_price = Decimal::from_i128_with_scale(unscaled, 18);
            if twap_price <= Decimal::ZERO {
                bail!("PMData TWAP price must be positive");
            }
            if bid.value(row)? != "none" || ask.value(row)? != "none" {
                bail!("PMData TWAP bid and ask columns must contain the literal none");
            }
            let report_version = version.value(row)?.trim().to_string();
            if report_version.is_empty() {
                bail!("PMData report version must not be empty");
            }
            let archive_row_number =
                i64::try_from(records.len()).context("PMData archive row count exceeds bigint")?;
            records.push(PmdataChainlinkBtcusdTwapRecord {
                source_timestamp,
                provider_received_at,
                valid_from_timestamp,
                expires_at,
                window_seconds: window.seconds(),
                twap_price,
                full_accuracy_value,
                report_version,
                source_date: date,
                archive_row_number,
            });
            previous_timestamp = Some(source_timestamp);
        }
    }

    let minimum_timestamp = records
        .first()
        .map(|record| record.source_timestamp)
        .context("PMData archive contained no records")?;
    let maximum_timestamp = records
        .last()
        .map(|record| record.source_timestamp)
        .context("PMData archive contained no records")?;
    Ok(PmdataParsedDay {
        records,
        minimum_timestamp,
        maximum_timestamp,
    })
}

async fn verify_archive(path: &Path, expected_sha256: Option<&str>) -> Result<PmdataArchive> {
    let body = tokio::fs::read(path)
        .await
        .with_context(|| format!("failed to read existing PMData archive {}", path.display()))?;
    validate_parquet_magic(&body)?;
    let sha256 = format!("{:x}", Sha256::digest(&body));
    if expected_sha256.is_some_and(|expected| expected != sha256) {
        bail!("existing PMData archive checksum conflicts with completed artifact");
    }
    Ok(PmdataArchive {
        path: path.to_path_buf(),
        sha256,
        bytes: body.len() as u64,
    })
}

fn validate_parquet_magic(body: &[u8]) -> Result<()> {
    if body.len() < 8 || &body[..4] != b"PAR1" || &body[body.len() - 4..] != b"PAR1" {
        bail!("PMData response is not a complete Parquet file");
    }
    Ok(())
}

fn timestamp_column<'a>(
    batch: &'a arrow_array::RecordBatch,
    schema: &arrow_schema::Schema,
    name: &str,
) -> Result<&'a TimestampMicrosecondArray> {
    let index = schema
        .index_of(name)
        .with_context(|| format!("PMData Parquet is missing {name}"))?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .with_context(|| format!("PMData column {name} is not timestamp[us]"))
}

enum StringColumn<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
}

impl StringColumn<'_> {
    fn value(&self, row: usize) -> Result<&str> {
        match self {
            Self::Utf8(values) if !values.is_null(row) => Ok(values.value(row)),
            Self::LargeUtf8(values) if !values.is_null(row) => Ok(values.value(row)),
            _ => bail!("PMData string column contains null"),
        }
    }
}

fn string_column<'a>(
    batch: &'a arrow_array::RecordBatch,
    schema: &arrow_schema::Schema,
    name: &str,
) -> Result<StringColumn<'a>> {
    let index = schema
        .index_of(name)
        .with_context(|| format!("PMData Parquet is missing {name}"))?;
    let column = batch.column(index);
    if let Some(values) = column.as_any().downcast_ref::<StringArray>() {
        return Ok(StringColumn::Utf8(values));
    }
    if let Some(values) = column.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(StringColumn::LargeUtf8(values));
    }
    bail!("PMData column {name} is not a string")
}

fn timestamp_value(
    values: &TimestampMicrosecondArray,
    row: usize,
    name: &str,
) -> Result<DateTime<Utc>> {
    if values.is_null(row) {
        bail!("PMData column {name} contains null");
    }
    DateTime::from_timestamp_micros(values.value(row))
        .with_context(|| format!("PMData column {name} contains an invalid timestamp"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use parquet::arrow::ArrowWriter;

    use super::*;

    #[test]
    fn archive_path_is_stable_and_partitioned() {
        let config = PmdataTwapConfig {
            base_url: DEFAULT_PMDATA_BASE_URL.to_string(),
            api_key: None,
            archive_root: PathBuf::from("/archive"),
        };
        let date = NaiveDate::from_ymd_opt(2026, 8, 1).unwrap();
        assert_eq!(
            config.archive_path(date, PmdataTwapWindow::Seconds60),
            PathBuf::from(
                "/archive/chainlink/BTCUSD/streams_twap60s/2026/08/BTCUSD_streams_twap60s_2026-08-01.parquet"
            )
        );
    }

    #[test]
    fn source_uri_matches_pmdata_contract() {
        let config = PmdataTwapConfig {
            base_url: DEFAULT_PMDATA_BASE_URL.to_string(),
            api_key: None,
            archive_root: PathBuf::from("/archive"),
        };
        let date = NaiveDate::from_ymd_opt(2026, 8, 1).unwrap();
        assert_eq!(
            config.source_uri(date, PmdataTwapWindow::Seconds30),
            "https://api.pmdata.dev/chainlink/BTCUSD/streams_twap30s/BTCUSD_streams_twap30s_2026-08-01.parquet"
        );
    }

    #[tokio::test]
    async fn parser_preserves_scaled_price_and_microsecond_timestamps() {
        let temporary = tempfile::tempdir().unwrap();
        let archive = temporary.path().join("sample.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "observationsTimestamp",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "receiveMicrosecondTimestamp",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("price", DataType::LargeUtf8, false),
            Field::new("bid", DataType::LargeUtf8, false),
            Field::new("ask", DataType::LargeUtf8, false),
            Field::new(
                "validFromTimestamp",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "expiresAt",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("version", DataType::LargeUtf8, false),
        ]));
        let source_micros = 1_785_543_118_000_000i64;
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(TimestampMicrosecondArray::from(vec![source_micros])),
                Arc::new(TimestampMicrosecondArray::from(vec![
                    source_micros + 1_014_022,
                ])),
                Arc::new(LargeStringArray::from(vec!["63018805980689359437824"])),
                Arc::new(LargeStringArray::from(vec!["none"])),
                Arc::new(LargeStringArray::from(vec!["none"])),
                Arc::new(TimestampMicrosecondArray::from(vec![source_micros])),
                Arc::new(TimestampMicrosecondArray::from(vec![
                    source_micros + 2_592_000_000_000,
                ])),
                Arc::new(LargeStringArray::from(vec!["V2"])),
            ],
        )
        .unwrap();
        let file = StdFile::create(&archive).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let parsed = parse_archive(
            archive,
            NaiveDate::from_ymd_opt(2026, 8, 1).unwrap(),
            PmdataTwapWindow::Seconds60,
            4_000,
        )
        .await
        .unwrap();
        assert_eq!(parsed.records.len(), 1);
        assert_eq!(parsed.records[0].window_seconds, 60);
        assert_eq!(
            parsed.records[0].full_accuracy_value,
            "63018805980689359437824"
        );
        assert_eq!(
            parsed.records[0].twap_price,
            Decimal::from_i128_with_scale(63_018_805_980_689_359_437_824, 18)
        );
        assert_eq!(
            parsed.records[0].source_timestamp.timestamp_micros(),
            source_micros
        );
    }
}
