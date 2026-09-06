use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Decimal128Array, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use sqlx::FromRow;
use uuid::Uuid;

use crate::domain::DrainExecutionError;

use super::common::io_error;

#[derive(Debug, FromRow)]
pub struct TradeRow {
    pub source: String,
    pub symbol: String,
    pub aggregate_trade_id: i64,
    pub trade_timestamp: DateTime<Utc>,
    pub provider_available_at: Option<DateTime<Utc>>,
    pub received_at: DateTime<Utc>,
    pub price: Decimal,
    pub quantity: Decimal,
    pub first_trade_id: i64,
    pub last_trade_id: i64,
    pub buyer_maker: bool,
    pub best_match: bool,
    pub payload_sha256: String,
    pub strategy_key: String,
    pub capture_artifact_id: Uuid,
    pub ingested_at: DateTime<Utc>,
}

pub fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("source", DataType::Utf8, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("aggregate_trade_id", DataType::Int64, false),
        Field::new(
            "trade_timestamp",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new(
            "provider_available_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new(
            "received_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("price", DataType::Decimal128(30, 10), false),
        Field::new("quantity", DataType::Decimal128(30, 10), false),
        Field::new("first_trade_id", DataType::Int64, false),
        Field::new("last_trade_id", DataType::Int64, false),
        Field::new("buyer_maker", DataType::Boolean, false),
        Field::new("best_match", DataType::Boolean, false),
        Field::new("payload_sha256", DataType::Utf8, false),
        Field::new("strategy_key", DataType::Utf8, false),
        Field::new("capture_artifact_id", DataType::Utf8, false),
        Field::new(
            "ingested_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
    ]))
}

fn decimal(mut value: Decimal) -> i128 {
    value.rescale(10);
    value.mantissa()
}

pub fn to_batch(rows: Vec<TradeRow>) -> Result<RecordBatch, DrainExecutionError> {
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.source.as_str()),
        )),
        Arc::new(StringArray::from_iter_values(
            rows.iter().map(|row| row.symbol.as_str()),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.aggregate_trade_id),
        )),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter()
                    .map(|row| row.trade_timestamp.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from(
                rows.iter()
                    .map(|row| {
                        row.provider_available_at
                            .map(|value| value.timestamp_micros())
                    })
                    .collect::<Vec<_>>(),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|row| row.received_at.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            Decimal128Array::from_iter_values(rows.iter().map(|row| decimal(row.price)))
                .with_precision_and_scale(30, 10)
                .map_err(|error| io_error(error.to_string()))?,
        ),
        Arc::new(
            Decimal128Array::from_iter_values(rows.iter().map(|row| decimal(row.quantity)))
                .with_precision_and_scale(30, 10)
                .map_err(|error| io_error(error.to_string()))?,
        ),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.first_trade_id),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|row| row.last_trade_id),
        )),
        Arc::new(BooleanArray::from_iter(
            rows.iter().map(|row| Some(row.buyer_maker)),
        )),
        Arc::new(BooleanArray::from_iter(
            rows.iter().map(|row| Some(row.best_match)),
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
        Arc::new(
            TimestampMicrosecondArray::from_iter_values(
                rows.iter().map(|row| row.ingested_at.timestamp_micros()),
            )
            .with_timezone("UTC"),
        ),
    ];
    RecordBatch::try_new(schema(), arrays).map_err(|error| io_error(error.to_string()))
}
