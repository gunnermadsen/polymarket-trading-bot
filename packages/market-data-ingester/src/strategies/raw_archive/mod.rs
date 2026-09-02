use std::{
    path::{Component, Path, PathBuf},
    time::Duration,
};

use chrono::{DateTime, Utc};
use reqwest::{header::RANGE, Client, StatusCode};
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncWriteExt},
    time::timeout,
};

use crate::{
    domain::{
        BackfillContext, BackfillExecutionError, BackfillFailureKind, BackfillOutcome,
        BackfillShard,
    },
    strategies::backfill_support,
};

const DEFAULT_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_SOURCE_RECONNECTS: usize = 5;

#[derive(Clone, Debug)]
pub struct RawObject {
    pub logical_key: String,
    pub provider: &'static str,
    pub source_uri: String,
    pub relative_path: PathBuf,
    pub media_type: &'static str,
    pub minimum: DateTime<Utc>,
    pub maximum: DateTime<Utc>,
}

pub fn root() -> Result<PathBuf, BackfillExecutionError> {
    let path = PathBuf::from(
        std::env::var("INGESTER_WEATHER_RAW_ROOT")
            .unwrap_or_else(|_| "/var/lib/weather/raw".into()),
    );
    if !path.is_absolute() {
        return Err(BackfillExecutionError::invalid(
            "raw_root_invalid",
            "raw data-lake root must be absolute",
        ));
    }
    Ok(path)
}

fn safe_relative(path: &Path) -> bool {
    !path.is_absolute()
        && path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

pub async fn store(
    context: &BackfillContext,
    strategy_key: &str,
    client: &Client,
    object: &RawObject,
) -> Result<BackfillOutcome, BackfillExecutionError> {
    if !safe_relative(&object.relative_path) {
        return Err(BackfillExecutionError::invalid(
            "raw_path_invalid",
            "raw artifact path was not safe",
        ));
    }
    if let Some(outcome) =
        backfill_support::completed_outcome(context, strategy_key, &object.logical_key).await?
    {
        return Ok(outcome);
    }
    let final_path = root()?.join(&object.relative_path);
    let parent = final_path.parent().ok_or_else(|| {
        BackfillExecutionError::invalid("raw_path_invalid", "raw artifact had no parent")
    })?;
    fs::create_dir_all(parent).await.map_err(io_error)?;
    let partial = parent.join(format!(
        ".{}.{}.partial",
        final_path
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or("artifact"),
        context.lease_token
    ));
    let _ = fs::remove_file(&partial).await;

    let artifact_id = backfill_support::create_artifact(
        context,
        strategy_key,
        strategy_key,
        &object.logical_key,
        object.provider,
        &object.source_uri,
        final_path.to_string_lossy().as_ref(),
    )
    .await?;

    let result = download(context, client, object, &partial).await;
    let (checksum, bytes) = match result {
        Ok(value) => value,
        Err(error) => {
            let _ = fs::remove_file(&partial).await;
            return Err(error);
        }
    };
    if final_path.exists() {
        let (existing_hash, existing_bytes) = hash_file(&final_path).await?;
        if existing_hash != checksum || existing_bytes != bytes {
            let _ = fs::remove_file(&partial).await;
            return Err(BackfillExecutionError::new(
                BackfillFailureKind::Integrity,
                "raw_artifact_conflict",
                format!(
                    "{} already exists with different bytes",
                    final_path.display()
                ),
            ));
        }
        fs::remove_file(&partial).await.map_err(io_error)?;
    } else {
        fs::rename(&partial, &final_path).await.map_err(io_error)?;
    }

    let mut tx = context
        .pool
        .begin()
        .await
        .map_err(backfill_support::database_error)?;
    backfill_support::complete_artifact(&mut tx, context, backfill_support::ArtifactCompletion {
        artifact_id, checksum: &checksum, byte_size: bytes, record_count: 1,
        minimum: Some(object.minimum), maximum: Some(object.maximum),
        metadata: json!({"media_type":object.media_type,"raw_source":true,"durable_target":final_path}),
    }).await?;
    tx.commit()
        .await
        .map_err(backfill_support::database_error)?;
    Ok(BackfillOutcome {
        records_verified: 1,
        verified_coverage: json!({"minimum_source_timestamp":object.minimum,"maximum_source_timestamp":object.maximum,"records_verified":1}),
        summary: json!({"provider":object.provider,"media_type":object.media_type,"byte_size":bytes,"sha256":checksum,"durable_target":final_path}),
    })
}

async fn download(
    context: &BackfillContext,
    client: &Client,
    object: &RawObject,
    path: &Path,
) -> Result<(String, u64), BackfillExecutionError> {
    let mut file = fs::File::create(path).await.map_err(io_error)?;
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    let mut reconnects = 0usize;
    'request: loop {
        let mut request = client.get(&object.source_uri);
        if bytes > 0 {
            request = request.header(RANGE, format!("bytes={bytes}-"));
        }
        let mut response = request
            .send()
            .await
            .map_err(source_error)?
            .error_for_status()
            .map_err(source_error)?;
        if bytes > 0 && response.status() != StatusCode::PARTIAL_CONTENT {
            return Err(source_error(
                "raw source did not honor the requested resume offset",
            ));
        }
        if response
            .content_length()
            .and_then(|remaining| bytes.checked_add(remaining))
            .is_some_and(|total| total > DEFAULT_MAX_BYTES)
        {
            return Err(source_error("raw artifact exceeded four GiB limit"));
        }
        loop {
            let next = tokio::select! {
                _ = context.shutdown.cancelled() => return Err(BackfillExecutionError::new(BackfillFailureKind::LeaseLost,"raw_download_cancelled","raw artifact download was cancelled")),
                result = timeout(Duration::from_secs(60), response.chunk()) => result,
            };
            match next {
                Ok(Ok(Some(chunk))) => {
                    bytes = bytes
                        .checked_add(chunk.len() as u64)
                        .ok_or_else(|| source_error("raw artifact size overflow"))?;
                    if bytes > DEFAULT_MAX_BYTES {
                        return Err(source_error("raw artifact exceeded four GiB limit"));
                    }
                    file.write_all(&chunk).await.map_err(io_error)?;
                    hash.update(&chunk);
                }
                Ok(Ok(None)) => break 'request,
                Ok(Err(_)) | Err(_) if reconnects < MAX_SOURCE_RECONNECTS => {
                    reconnects += 1;
                    continue 'request;
                }
                Ok(Err(error)) => return Err(source_error(error)),
                Err(_) => return Err(source_error("raw artifact download stalled")),
            }
        }
    }
    if bytes == 0 {
        return Err(source_error("raw artifact was empty"));
    }
    file.flush().await.map_err(io_error)?;
    file.sync_all().await.map_err(io_error)?;
    drop(file);
    validate_signature(path, object.media_type).await?;
    Ok((format!("{:x}", hash.finalize()), bytes))
}

async fn validate_signature(path: &Path, media_type: &str) -> Result<(), BackfillExecutionError> {
    let mut file = fs::File::open(path).await.map_err(io_error)?;
    let mut prefix = [0u8; 64];
    let count = file.read(&mut prefix).await.map_err(io_error)?;
    let bytes = &prefix[..count];
    let valid = match media_type {
        "application/x-netcdf" => {
            bytes.starts_with(b"CDF") || bytes.starts_with(b"\x89HDF\r\n\x1a\n")
        }
        "application/x-grib2" => bytes.starts_with(b"GRIB"),
        "application/vnd.apache.parquet" => bytes.starts_with(b"PAR1"),
        "application/json" => bytes
            .iter()
            .copied()
            .find(|v| !v.is_ascii_whitespace())
            .is_some_and(|v| matches!(v, b'[' | b'{')),
        "text/csv" => std::str::from_utf8(bytes).is_ok_and(|v| v.contains(',')),
        _ => false,
    };
    if !valid {
        return Err(BackfillExecutionError::new(
            BackfillFailureKind::Integrity,
            "raw_format_invalid",
            format!("downloaded artifact did not match {media_type}"),
        ));
    }
    Ok(())
}

async fn hash_file(path: &Path) -> Result<(String, u64), BackfillExecutionError> {
    let mut file = fs::File::open(path).await.map_err(io_error)?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    loop {
        let count = file.read(&mut buffer).await.map_err(io_error)?;
        if count == 0 {
            break;
        }
        bytes += count as u64;
        hash.update(&buffer[..count]);
    }
    Ok((format!("{:x}", hash.finalize()), bytes))
}

pub fn client() -> Result<Client, BackfillExecutionError> {
    Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .user_agent("capitonic-ingester-worker/1")
        .build()
        .map_err(|e| {
            BackfillExecutionError::new(
                BackfillFailureKind::Integrity,
                "raw_http_client",
                e.to_string(),
            )
        })
}

pub fn combine(shard: &BackfillShard, outcomes: Vec<BackfillOutcome>) -> BackfillOutcome {
    let records = outcomes.iter().map(|v| v.records_verified).sum::<i64>();
    BackfillOutcome {
        records_verified: records,
        verified_coverage: json!({"requested_start":shard.range_start,"requested_end":shard.range_end,"records_verified":records}),
        summary: json!({"raw_artifacts_verified":records}),
    }
}

fn source_error(error: impl std::fmt::Display) -> BackfillExecutionError {
    BackfillExecutionError::new(
        BackfillFailureKind::TransientSource,
        "raw_source",
        error.to_string(),
    )
}
fn io_error(error: impl std::fmt::Display) -> BackfillExecutionError {
    BackfillExecutionError::new(
        BackfillFailureKind::Integrity,
        "raw_storage",
        error.to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_path_traversal() {
        assert!(!safe_relative(Path::new("../escape")));
        assert!(safe_relative(Path::new("noaa/day/file.nc")));
    }
}
