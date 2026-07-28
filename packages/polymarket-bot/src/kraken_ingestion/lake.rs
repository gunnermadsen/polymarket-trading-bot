use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use arrow_array::{Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use sha2::{Digest, Sha256};
use tokio::task;

use super::job::{
    KrakenBackfillJob, KrakenDataset, NormalizedRows, PublishedLakeObject, KRAKEN_PROVIDER,
};

#[derive(Debug, Clone)]
pub struct KrakenDataLake {
    root: PathBuf,
}

impl KrakenDataLake {
    pub fn new(root: PathBuf) -> Result<Self> {
        if !root.is_absolute() {
            bail!("Kraken data lake root must be an absolute path");
        }
        Ok(Self { root })
    }

    pub async fn publish(
        &self,
        job: &KrakenBackfillJob,
        rows: &NormalizedRows,
    ) -> Result<PublishedLakeObject> {
        let root = self.root.clone();
        let job = job.clone();
        let records = rows.lake_records();
        task::spawn_blocking(move || publish_blocking(&root, &job, records))
            .await
            .context("Kraken Parquet publication task failed")?
    }
}

fn publish_blocking(
    root: &Path,
    job: &KrakenBackfillJob,
    records: Vec<super::job::LakeRecord>,
) -> Result<PublishedLakeObject> {
    let dataset = job.dataset()?;
    let start = job.range_start.format("%Y%m%dT%H%M%SZ");
    let end = job.range_end.format("%Y%m%dT%H%M%SZ");
    let partition = PathBuf::from(format!(
        "provider={KRAKEN_PROVIDER}/dataset={}/symbol={}/interval_seconds={}/year={}/month={:02}",
        dataset,
        job.symbol,
        job.interval_seconds,
        job.range_start.format("%Y"),
        job.range_start.format("%m")
    ));
    let staging_dir = root.join(".staging");
    let partition_dir = root.join(&partition);
    fs::create_dir_all(&staging_dir).context("failed to create Kraken staging directory")?;
    fs::create_dir_all(&partition_dir).context("failed to create Kraken partition directory")?;

    let staging_path = staging_dir.join(format!("{}.parquet.tmp", job.job_id));
    write_parquet(&staging_path, dataset, &job, &records)?;
    let (sha256, byte_size) = hash_file(&staging_path)?;
    let file_name = format!("{start}_{end}_{sha256}.parquet");
    let relative_path = partition.join(file_name);
    let final_path = root.join(&relative_path);

    if final_path.exists() {
        let (existing_hash, existing_size) = hash_file(&final_path)?;
        if existing_hash != sha256 || existing_size != byte_size {
            bail!("existing Kraken lake object did not match its content address");
        }
        fs::remove_file(&staging_path).context("failed to remove duplicate staging object")?;
    } else {
        fs::rename(&staging_path, &final_path)
            .context("failed to atomically publish Kraken lake object")?;
    }

    Ok(PublishedLakeObject {
        relative_path: relative_path
            .to_str()
            .context("Kraken lake path was not valid UTF-8")?
            .to_string(),
        sha256,
        byte_size,
        row_count: i64::try_from(records.len()).context("Kraken row count exceeded i64")?,
    })
}

fn write_parquet(
    path: &Path,
    dataset: KrakenDataset,
    job: &KrakenBackfillJob,
    records: &[super::job::LakeRecord],
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("observed_at_ms", DataType::Int64, false),
        Field::new("dataset", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("interval_seconds", DataType::Int32, false),
        Field::new("payload_json", DataType::Utf8, false),
    ]));
    let observed_at = records
        .iter()
        .map(|record| record.observed_at.timestamp_millis())
        .collect::<Vec<_>>();
    let datasets = vec![dataset.as_str(); records.len()];
    let symbols = vec![job.symbol.as_str(); records.len()];
    let intervals = vec![job.interval_seconds; records.len()];
    let payloads = records
        .iter()
        .map(|record| serde_json::to_string(&record.payload))
        .collect::<serde_json::Result<Vec<_>>>()
        .context("failed to serialize Kraken lake payloads")?;
    let payload_refs = payloads.iter().map(String::as_str).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(observed_at)),
            Arc::new(StringArray::from(datasets)),
            Arc::new(StringArray::from(symbols)),
            Arc::new(Int32Array::from(intervals)),
            Arc::new(StringArray::from(payload_refs)),
        ],
    )
    .context("failed to build Kraken Arrow record batch")?;
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(6).context("invalid Kraken Parquet compression level")?,
        ))
        .build();
    let file = File::create(path).context("failed to create Kraken staging Parquet file")?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(properties))
        .context("failed to open Parquet writer")?;
    writer
        .write(&batch)
        .context("failed to write Kraken Parquet batch")?;
    writer
        .close()
        .context("failed to close Kraken Parquet file")?;
    Ok(())
}

fn hash_file(path: &Path) -> Result<(String, i64)> {
    let bytes = fs::read(path).context("failed to read Kraken lake object for hashing")?;
    let size = i64::try_from(bytes.len()).context("Kraken lake object exceeded i64 bytes")?;
    Ok((format!("{:x}", Sha256::digest(bytes)), size))
}
