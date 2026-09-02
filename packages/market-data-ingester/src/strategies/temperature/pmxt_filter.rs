use std::{collections::HashSet, fs::File, path::Path};

use arrow_array::{
    Array, BinaryArray, FixedSizeBinaryArray, LargeBinaryArray, LargeStringArray, RecordBatch,
    StringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
};
use arrow_select::filter::filter_record_batch;
use chrono::{DateTime, Utc};
use parquet::{
    arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter},
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use sha2::{Digest, Sha256};

use crate::domain::{BackfillExecutionError, BackfillFailureKind};

pub struct FilterSummary {
    pub records: i64,
    pub minimum: Option<DateTime<Utc>>,
    pub maximum: Option<DateTime<Utc>>,
    pub checksum: String,
    pub bytes: u64,
}

pub fn filter_archive(
    source: &Path,
    output: &Path,
    condition_ids: HashSet<String>,
    token_ids: HashSet<String>,
    range_start: DateTime<Utc>,
    range_end: DateTime<Utc>,
) -> Result<FilterSummary, BackfillExecutionError> {
    let input = File::open(source).map_err(integrity)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(input).map_err(integrity)?;
    let schema = builder.schema().clone();
    let market_index = schema.index_of("market").map_err(integrity)?;
    let asset_index = schema.index_of("asset_id").map_err(integrity)?;
    let event_index = schema.index_of("event_type").map_err(integrity)?;
    let timestamp_index = schema.index_of("timestamp").map_err(integrity)?;
    let reader = builder.with_batch_size(8_192).build().map_err(integrity)?;

    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).map_err(integrity)?))
        .build();
    let destination = File::create(output).map_err(integrity)?;
    let mut writer =
        ArrowWriter::try_new(destination, schema, Some(properties)).map_err(integrity)?;
    let mut records = 0i64;
    let mut minimum = None;
    let mut maximum = None;

    for batch in reader {
        let batch = batch.map_err(integrity)?;
        let mask = (0..batch.num_rows())
            .map(|row| {
                let condition = text_value(batch.column(market_index).as_ref(), row)?;
                let token = text_value(batch.column(asset_index).as_ref(), row)?;
                let event = text_value(batch.column(event_index).as_ref(), row)?;
                let timestamp = timestamp_value(batch.column(timestamp_index).as_ref(), row)?;
                Ok(condition_ids.contains(condition)
                    && token_ids.contains(token)
                    && matches!(event, "book" | "price_change")
                    && timestamp >= range_start
                    && timestamp < range_end)
            })
            .collect::<Result<Vec<_>, BackfillExecutionError>>()?;
        let mask = arrow_array::BooleanArray::from(mask);
        let filtered = filter_record_batch(&batch, &mask).map_err(integrity)?;
        if filtered.num_rows() == 0 {
            continue;
        }
        update_bounds(&filtered, timestamp_index, &mut minimum, &mut maximum)?;
        records = records
            .checked_add(i64::try_from(filtered.num_rows()).map_err(integrity)?)
            .ok_or_else(|| integrity("filtered PMXT row count overflow"))?;
        writer.write(&filtered).map_err(integrity)?;
    }
    writer.close().map_err(integrity)?;
    let (checksum, bytes) = hash_file(output)?;
    Ok(FilterSummary {
        records,
        minimum,
        maximum,
        checksum,
        bytes,
    })
}

fn update_bounds(
    batch: &RecordBatch,
    index: usize,
    minimum: &mut Option<DateTime<Utc>>,
    maximum: &mut Option<DateTime<Utc>>,
) -> Result<(), BackfillExecutionError> {
    for row in 0..batch.num_rows() {
        let timestamp = timestamp_value(batch.column(index).as_ref(), row)?;
        *minimum = Some(minimum.map_or(timestamp, |value| value.min(timestamp)));
        *maximum = Some(maximum.map_or(timestamp, |value| value.max(timestamp)));
    }
    Ok(())
}

fn text_value(array: &dyn Array, row: usize) -> Result<&str, BackfillExecutionError> {
    if array.is_null(row) {
        return Err(integrity("PMXT identity column contained null"));
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<BinaryArray>() {
        return std::str::from_utf8(values.value(row)).map_err(integrity);
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        return std::str::from_utf8(values.value(row)).map_err(integrity);
    }
    if let Some(values) = array.as_any().downcast_ref::<FixedSizeBinaryArray>() {
        return std::str::from_utf8(values.value(row)).map_err(integrity);
    }
    Err(integrity("PMXT identity column had an unsupported type"))
}

fn timestamp_value(array: &dyn Array, row: usize) -> Result<DateTime<Utc>, BackfillExecutionError> {
    if array.is_null(row) {
        return Err(integrity("PMXT timestamp column contained null"));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMillisecondArray>() {
        return DateTime::from_timestamp_millis(values.value(row))
            .ok_or_else(|| integrity("PMXT timestamp was out of range"));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return DateTime::from_timestamp_micros(values.value(row))
            .ok_or_else(|| integrity("PMXT timestamp was out of range"));
    }
    Err(integrity("PMXT timestamp column had an unsupported type"))
}

fn hash_file(path: &Path) -> Result<(String, u64), BackfillExecutionError> {
    use std::io::Read;
    let mut file = File::open(path).map_err(integrity)?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    loop {
        let count = file.read(&mut buffer).map_err(integrity)?;
        if count == 0 {
            break;
        }
        bytes = bytes
            .checked_add(count as u64)
            .ok_or_else(|| integrity("filtered PMXT byte count overflow"))?;
        hash.update(&buffer[..count]);
    }
    Ok((format!("{:x}", hash.finalize()), bytes))
}

fn integrity(error: impl std::fmt::Display) -> BackfillExecutionError {
    BackfillExecutionError::new(
        BackfillFailureKind::Integrity,
        "pmxt_temperature_filter",
        error.to_string(),
    )
}
