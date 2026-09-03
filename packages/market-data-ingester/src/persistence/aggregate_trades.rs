use chrono::{DateTime, Utc};
use sqlx::{Postgres, QueryBuilder, Transaction};
use uuid::Uuid;

use crate::domain::BinanceAggregateTradeRecord;

pub struct BinanceAggregateTradeWrite<'a> {
    pub record: &'a BinanceAggregateTradeRecord,
    pub provider_available_at: Option<DateTime<Utc>>,
    pub received_at: DateTime<Utc>,
    pub capture_artifact_id: Uuid,
}

pub async fn insert_binance_aggregate_trades(
    transaction: &mut Transaction<'_, Postgres>,
    strategy_key: &str,
    writes: &[BinanceAggregateTradeWrite<'_>],
) -> Result<Vec<i64>, sqlx::Error> {
    if writes.is_empty() {
        return Ok(Vec::new());
    }
    let mut query = QueryBuilder::<Postgres>::new(
        "INSERT INTO market_data.binance_spot_btcusdt_aggregate_trades (source,symbol,aggregate_trade_id,trade_timestamp,provider_available_at,received_at,price,quantity,first_trade_id,last_trade_id,buyer_maker,best_match,payload_sha256,strategy_key,capture_artifact_id) ",
    );
    query.push_values(writes, |mut row, write| {
        let record = write.record;
        row.push_bind("binance_spot")
            .push_bind(&record.symbol)
            .push_bind(record.aggregate_trade_id)
            .push_bind(record.trade_timestamp)
            .push_bind(write.provider_available_at)
            .push_bind(write.received_at)
            .push_bind(record.price)
            .push_bind(record.quantity)
            .push_bind(record.first_trade_id)
            .push_bind(record.last_trade_id)
            .push_bind(record.buyer_maker)
            .push_bind(record.best_match)
            .push_bind(&record.payload_sha256)
            .push_bind(strategy_key)
            .push_bind(write.capture_artifact_id);
    });
    query.push(
        " ON CONFLICT (symbol,trade_timestamp,aggregate_trade_id) DO NOTHING RETURNING aggregate_trade_id",
    );
    query
        .build_query_scalar::<i64>()
        .fetch_all(&mut **transaction)
        .await
}
