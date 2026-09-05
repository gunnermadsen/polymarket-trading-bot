use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use chrono::{DateTime, Utc};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::{
        properties::WriterProperties,
        reader::{FileReader, SerializedFileReader},
    },
};
use sha2::{Digest, Sha256};
use sqlx::FromRow;
use tokio::{fs, io::AsyncReadExt, sync::mpsc};
use uuid::Uuid;

use crate::domain::{DrainContext, DrainExecutionError};

#[derive(Debug, Clone, FromRow)]
pub struct Chunk {
    pub chunk_schema: String,
    pub chunk_name: String,
    pub range_start: DateTime<Utc>,
    pub range_end: DateTime<Utc>,
}

#[derive(Debug, Clone, FromRow)]
pub struct Publication {
    pub object_id: Uuid,
    pub row_count: i64,
    pub relative_path: String,
    pub sha256: String,
    pub byte_size: i64,
    pub status: String,
}

pub async fn existing_publication(
    context: &DrainContext,
    strategy_key: &str,
    chunk: &Chunk,
) -> Result<Option<Publication>, DrainExecutionError> {
    let value = sqlx::query_as::<_, Publication>(
        "SELECT object_id,row_count,relative_path,sha256::text,byte_size,status \
         FROM ingester.drain_objects WHERE strategy_key=$1 AND source_chunk_schema=$2 \
         AND source_chunk_name=$3 AND status IN ('published','removed')",
    )
    .bind(strategy_key)
    .bind(&chunk.chunk_schema)
    .bind(&chunk.chunk_name)
    .fetch_optional(&context.pool)
    .await
    .map_err(db_error)?;
    if value.as_ref().is_some_and(|item| item.status == "removed") {
        return Err(invalid(
            "drain_chunk_already_removed",
            "removed drain object still has a source chunk",
        ));
    }
    Ok(value)
}

pub async fn create_object(
    context: &DrainContext,
    strategy_key: &str,
    relation: &str,
    chunk: &Chunk,
) -> Result<Uuid, DrainExecutionError> {
    sqlx::query_scalar(
        "INSERT INTO ingester.drain_objects(job_id,strategy_key,source_relation,\
         source_chunk_schema,source_chunk_name,source_start,source_end) \
         VALUES($1,$2,$3,$4,$5,$6,$7) ON CONFLICT(strategy_key,source_chunk_schema,\
         source_chunk_name) DO UPDATE SET job_id=EXCLUDED.job_id,updated_at=clock_timestamp() \
         WHERE ingester.drain_objects.status='staging' RETURNING object_id",
    )
    .bind(context.job_id)
    .bind(strategy_key)
    .bind(relation)
    .bind(&chunk.chunk_schema)
    .bind(&chunk.chunk_name)
    .bind(chunk.range_start)
    .bind(chunk.range_end)
    .fetch_one(&context.pool)
    .await
    .map_err(db_error)
}

pub fn start_writer(
    staging: PathBuf,
    schema: Arc<Schema>,
) -> (
    mpsc::Sender<RecordBatch>,
    tokio::task::JoinHandle<Result<(), String>>,
) {
    let (sender, mut receiver) = mpsc::channel::<RecordBatch>(2);
    let writer = tokio::task::spawn_blocking(move || -> Result<(), String> {
        let file = File::create(&staging).map_err(|error| error.to_string())?;
        let properties = WriterProperties::builder()
            .set_compression(Compression::ZSTD(
                ZstdLevel::try_new(6).map_err(|error| error.to_string())?,
            ))
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(properties))
            .map_err(|error| error.to_string())?;
        while let Some(batch) = receiver.blocking_recv() {
            writer.write(&batch).map_err(|error| error.to_string())?;
        }
        let mut file = writer.into_inner().map_err(|error| error.to_string())?;
        use std::io::Write;
        file.flush().map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        Ok(())
    });
    (sender, writer)
}

pub async fn finish_writer(
    sender: mpsc::Sender<RecordBatch>,
    writer: tokio::task::JoinHandle<Result<(), String>>,
) -> Result<(), DrainExecutionError> {
    drop(sender);
    writer
        .await
        .map_err(|error| io_error(error.to_string()))?
        .map_err(io_error)
}

pub async fn publish_file(
    staging: &Path,
    root: &Path,
    relative: &str,
    expected_rows: i64,
) -> Result<(String, i64), DrainExecutionError> {
    let (sha256, byte_size) = hash_file(staging).await?;
    verify_parquet(staging, expected_rows).await?;
    let final_path = root.join(relative);
    let directory = final_path.parent().ok_or_else(|| {
        invalid(
            "drain_path_invalid",
            "published drain object has no parent directory",
        )
    })?;
    fs::create_dir_all(directory).await.map_err(io_error)?;
    if final_path.exists() {
        verify_hash(&final_path, &sha256, byte_size).await?;
        fs::remove_file(staging).await.map_err(io_error)?;
    } else {
        fs::rename(staging, &final_path).await.map_err(io_error)?;
        sync_directory(directory.to_owned()).await?;
    }
    Ok((sha256, byte_size))
}

pub async fn verify_existing(
    root: &Path,
    publication: &Publication,
) -> Result<(), DrainExecutionError> {
    let path = root.join(&publication.relative_path);
    verify_hash(&path, &publication.sha256, publication.byte_size).await?;
    verify_parquet(&path, publication.row_count).await
}

async fn hash_file(path: &Path) -> Result<(String, i64), DrainExecutionError> {
    let mut file = fs::File::open(path).await.map_err(io_error)?;
    let mut hash = Sha256::new();
    let mut bytes = 0i64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).await.map_err(io_error)?;
        if count == 0 {
            break;
        }
        bytes += count as i64;
        hash.update(&buffer[..count]);
    }
    Ok((format!("{:x}", hash.finalize()), bytes))
}

async fn verify_hash(path: &Path, expected: &str, size: i64) -> Result<(), DrainExecutionError> {
    let (actual, actual_size) = hash_file(path).await?;
    if actual != expected || actual_size != size {
        return Err(invalid(
            "drain_object_conflict",
            "existing Parquet object hash or size differs",
        ));
    }
    Ok(())
}

async fn verify_parquet(path: &Path, expected: i64) -> Result<(), DrainExecutionError> {
    let path = path.to_owned();
    let rows = tokio::task::spawn_blocking(move || {
        SerializedFileReader::new(File::open(path).map_err(|error| error.to_string())?)
            .map(|reader| reader.metadata().file_metadata().num_rows())
            .map_err(|error| error.to_string())
    })
    .await
    .map_err(|error| io_error(error.to_string()))?
    .map_err(io_error)?;
    if rows != expected {
        return Err(invalid(
            "drain_row_count_mismatch",
            format!("Parquet has {rows} rows, expected {expected}"),
        ));
    }
    Ok(())
}

async fn sync_directory(path: PathBuf) -> Result<(), DrainExecutionError> {
    tokio::task::spawn_blocking(move || File::open(path)?.sync_all())
        .await
        .map_err(|error| io_error(error.to_string()))?
        .map_err(io_error)
}

pub fn invalid(code: &'static str, message: impl Into<String>) -> DrainExecutionError {
    DrainExecutionError::new(code, message, false)
}

pub fn io_error(error: impl std::fmt::Display) -> DrainExecutionError {
    DrainExecutionError::new("drain_io_failed", error.to_string(), true)
}

pub fn db_error(error: impl std::fmt::Display) -> DrainExecutionError {
    DrainExecutionError::new("drain_database_failed", error.to_string(), true)
}
