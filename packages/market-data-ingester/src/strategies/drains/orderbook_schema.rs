use std::sync::Arc;

use arrow_array::{
    ArrayRef, Decimal128Array, Int32Array, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde_json::Value;
use sqlx::FromRow;
use uuid::Uuid;

use crate::domain::DrainExecutionError;

use super::common::io_error;

#[derive(Debug, FromRow)]
pub struct OrderbookRow {
    pub sampled_at: DateTime<Utc>,
    pub source_timestamp: DateTime<Utc>,
    pub provider_available_at: DateTime<Utc>,
    pub received_at: DateTime<Utc>,
    pub source: String,
    pub market_id: String,
    pub condition_id: String,
    pub event_slug: String,
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    pub token_id: String,
    pub outcome: String,
    pub connection_epoch: Uuid,
    pub ingest_sequence: i64,
    pub tick_size: Decimal,
    pub best_bid: Option<Decimal>,
    pub best_ask: Option<Decimal>,
    pub bid_depth: i32,
    pub ask_depth: i32,
    pub bids: Value,
    pub asks: Value,
    pub source_hash: Option<String>,
    pub book_sha256: String,
    pub sampling_policy: Value,
    pub sampling_policy_sha256: String,
    pub payload_sha256: String,
    pub strategy_key: String,
    pub capture_artifact_id: Uuid,
    pub ingested_at: DateTime<Utc>,
}

pub fn schema() -> Arc<Schema> {
    let timestamp = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    Arc::new(Schema::new(vec![
        Field::new("sampled_at", timestamp.clone(), false),
        Field::new("source_timestamp", timestamp.clone(), false),
        Field::new("provider_available_at", timestamp.clone(), false),
        Field::new("received_at", timestamp.clone(), false),
        Field::new("source", DataType::Utf8, false),
        Field::new("market_id", DataType::Utf8, false),
        Field::new("condition_id", DataType::Utf8, false),
        Field::new("event_slug", DataType::Utf8, false),
        Field::new("window_start", timestamp.clone(), false),
        Field::new("window_end", timestamp.clone(), false),
        Field::new("token_id", DataType::Utf8, false),
        Field::new("outcome", DataType::Utf8, false),
        Field::new("connection_epoch", DataType::Utf8, false),
        Field::new("ingest_sequence", DataType::Int64, false),
        Field::new("tick_size", DataType::Decimal128(18, 8), false),
        Field::new("best_bid", DataType::Decimal128(18, 8), true),
        Field::new("best_ask", DataType::Decimal128(18, 8), true),
        Field::new("bid_depth", DataType::Int32, false),
        Field::new("ask_depth", DataType::Int32, false),
        Field::new("bids", DataType::Utf8, false),
        Field::new("asks", DataType::Utf8, false),
        Field::new("source_hash", DataType::Utf8, true),
        Field::new("book_sha256", DataType::Utf8, false),
        Field::new("sampling_policy", DataType::Utf8, false),
        Field::new("sampling_policy_sha256", DataType::Utf8, false),
        Field::new("payload_sha256", DataType::Utf8, false),
        Field::new("strategy_key", DataType::Utf8, false),
        Field::new("capture_artifact_id", DataType::Utf8, false),
        Field::new("ingested_at", timestamp, false),
    ]))
}

fn timestamp(values: impl Iterator<Item = DateTime<Utc>>) -> ArrayRef {
    Arc::new(
        TimestampMicrosecondArray::from_iter_values(values.map(|value| value.timestamp_micros()))
            .with_timezone("UTC"),
    )
}

fn decimal(mut value: Decimal) -> i128 {
    value.rescale(8);
    value.mantissa()
}

fn optional_decimal(
    values: impl Iterator<Item = Option<Decimal>>,
) -> Result<ArrayRef, DrainExecutionError> {
    Ok(Arc::new(
        Decimal128Array::from(values.map(|value| value.map(decimal)).collect::<Vec<_>>())
            .with_precision_and_scale(18, 8)
            .map_err(|error| io_error(error.to_string()))?,
    ))
}

pub fn to_batch(rows: Vec<OrderbookRow>) -> Result<RecordBatch, DrainExecutionError> {
    let arrays: Vec<ArrayRef> = vec![
        timestamp(rows.iter().map(|row| row.sampled_at)),
        timestamp(rows.iter().map(|row| row.source_timestamp)),
        timestamp(rows.iter().map(|row| row.provider_available_at)),
        timestamp(rows.iter().map(|row| row.received_at)),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.source.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.market_id.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.condition_id.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.event_slug.as_str()),
        )),
        timestamp(rows.iter().map(|row| row.window_start)),
        timestamp(rows.iter().map(|row| row.window_end)),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.token_id.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.outcome.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.connection_epoch.to_string()),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.ingest_sequence),
        )),
        Arc::new(
            Decimal128Array::from_iter_values(rows.iter().map(|row| decimal(row.tick_size)))
                .with_precision_and_scale(18, 8)
                .map_err(|error| io_error(error.to_string()))?,
        ),
        optional_decimal(rows.iter().map(|row| row.best_bid))?,
        optional_decimal(rows.iter().map(|row| row.best_ask))?,
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|row| row.bid_depth),
        )),
        Arc::new(Int32Array::from_iter_values(
            rows.iter().map(|row| row.ask_depth),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.bids.to_string()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.asks.to_string()),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| row.source_hash.as_deref())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.book_sha256.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.sampling_policy.to_string()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.sampling_policy_sha256.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.payload_sha256.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.strategy_key.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.capture_artifact_id.to_string()),
        )),
        timestamp(rows.iter().map(|row| row.ingested_at)),
    ];
    RecordBatch::try_new(schema(), arrays).map_err(|error| io_error(error.to_string()))
}
