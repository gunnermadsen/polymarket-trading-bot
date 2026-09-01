use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use sqlx::{Postgres, QueryBuilder, Row, Transaction};

use crate::domain::{
    BackfillContext, BackfillExecutionError, BackfillFailureKind, BackfillOutcome, BackfillRequest,
    BackfillShard, BackfillWorkerStrategy, StrategyCapability, StrategyDescriptor,
    ValidatedBackfillRequest,
};

const STRATEGY_KEY: &str = "binance_spot_btcusdt_aggregate_trades";
const SYMBOL: &str = "BTCUSDT";
const SOURCE: &str = "binance_spot";
const REST_URL: &str = "https://data-api.binance.vision/api/v3/aggTrades";
const PAGE_LIMIT: usize = 1_000;
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

pub struct BinanceSpotAggregateTradesBackfill {
    descriptor: StrategyDescriptor,
    client: Client,
}

impl BinanceSpotAggregateTradesBackfill {
    pub fn new() -> Result<Self, BackfillExecutionError> {
        let descriptor = StrategyDescriptor {
            strategy_key: Arc::from(STRATEGY_KEY),
            name: Arc::from("Binance Spot BTCUSDT aggregate trades"),
            description: Arc::from("Collects historical Binance BTCUSDT aggregate trades"),
            capabilities: vec![StrategyCapability::Realtime, StrategyCapability::Backfill],
            strategy_contract_version: 1,
            request_schema_version: Some(1),
            shardable: true,
            maximum_shards: 10_080,
        };
        descriptor.validate()?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .user_agent("capitonic-ingester-worker/1")
            .build()
            .map_err(|error| {
                BackfillExecutionError::new(
                    BackfillFailureKind::Integrity,
                    "binance_backfill_client",
                    error.to_string(),
                )
            })?;
        Ok(Self { descriptor, client })
    }
}

#[async_trait]
impl BackfillWorkerStrategy for BinanceSpotAggregateTradesBackfill {
    fn descriptor(&self) -> &StrategyDescriptor {
        &self.descriptor
    }

    fn validate_request(
        &self,
        request: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        if request.strategy_key != STRATEGY_KEY {
            return Err(BackfillExecutionError::invalid(
                "strategy_key_mismatch",
                "request strategy key does not match Binance aggregate trades",
            ));
        }
        if request.range.end <= request.range.start || request.range.end > Utc::now() {
            return Err(BackfillExecutionError::invalid(
                "range_invalid",
                "range must be increasing and may not end in the future",
            ));
        }
        if !request
            .parameters
            .as_object()
            .is_some_and(|parameters| parameters.is_empty())
        {
            return Err(BackfillExecutionError::invalid(
                "parameters_invalid",
                "Binance BTCUSDT aggregate-trade backfills accept no parameters",
            ));
        }
        request.execution.validate()?;
        Ok(ValidatedBackfillRequest {
            strategy_key: self.descriptor.strategy_key.clone(),
            strategy_contract_version: self.descriptor.strategy_contract_version,
            request_schema_version: self.descriptor.request_schema_version.unwrap_or(1),
            range_start: request.range.start,
            range_end: request.range.end,
            parameters: request.parameters.clone(),
            execution: request.execution.clone(),
        })
    }

    fn plan_shards(
        &self,
        request: &ValidatedBackfillRequest,
    ) -> Result<Vec<BackfillShard>, BackfillExecutionError> {
        let mut start = request.range_start;
        let mut shards = Vec::new();
        while start < request.range_end {
            let end = (start + chrono::Duration::minutes(1)).min(request.range_end);
            shards.push(BackfillShard {
                shard_key: format!("{}-{}", start.timestamp_millis(), end.timestamp_millis()),
                range_start: start,
                range_end: end,
                parameters: json!({}),
            });
            if shards.len() > self.descriptor.maximum_shards {
                return Err(BackfillExecutionError::invalid(
                    "too_many_shards",
                    format!(
                        "request exceeds {} one-minute shards",
                        self.descriptor.maximum_shards
                    ),
                ));
            }
            start = end;
        }
        Ok(shards)
    }

    async fn execute_backfill(
        &self,
        context: BackfillContext,
        shard: BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        let trades = self.fetch_shard(&context, &shard).await?;
        let records_verified = persist(&context, &trades).await?;
        Ok(BackfillOutcome {
            records_verified,
            verified_coverage: json!({
                "minimum_source_time": trades.first().map(|trade| trade.trade_timestamp),
                "maximum_source_time": trades.last().map(|trade| trade.trade_timestamp),
                "records_verified": records_verified,
                "requested_start": shard.range_start,
                "requested_end": shard.range_end,
            }),
            summary: json!({
                "provider": SOURCE,
                "symbol": SYMBOL,
                "records_verified": records_verified,
            }),
        })
    }
}

impl BinanceSpotAggregateTradesBackfill {
    async fn fetch_shard(
        &self,
        context: &BackfillContext,
        shard: &BackfillShard,
    ) -> Result<Vec<Trade>, BackfillExecutionError> {
        let mut all = Vec::new();
        let mut from_id: Option<i64> = None;
        loop {
            let mut request = self.client.get(REST_URL).query(&[
                ("symbol", SYMBOL.to_owned()),
                ("limit", PAGE_LIMIT.to_string()),
            ]);
            request = if let Some(id) = from_id {
                request.query(&[("fromId", id.to_string())])
            } else {
                request.query(&[
                    (
                        "startTime",
                        shard.range_start.timestamp_millis().to_string(),
                    ),
                    (
                        "endTime",
                        (shard.range_end.timestamp_millis() - 1).to_string(),
                    ),
                ])
            };
            let response = tokio::select! {
                _ = context.shutdown.cancelled() => return Err(BackfillExecutionError::new(
                    BackfillFailureKind::Cancelled, "worker_shutdown", "worker shutdown interrupted source request")),
                response = request.send() => response,
            }
            .map_err(|error| source_error("binance_backfill_request", error.to_string()))?;
            if response.status().as_u16() == 429 {
                return Err(BackfillExecutionError::new(
                    BackfillFailureKind::RateLimited,
                    "binance_backfill_rate_limited",
                    "Binance rate limited the historical request",
                ));
            }
            if !response.status().is_success() {
                return Err(source_error(
                    "binance_backfill_status",
                    format!("Binance returned HTTP {}", response.status()),
                ));
            }
            if response
                .content_length()
                .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
            {
                return Err(source_error(
                    "binance_backfill_body_large",
                    "Binance response exceeded the body limit",
                ));
            }
            let bytes = response
                .bytes()
                .await
                .map_err(|error| source_error("binance_backfill_body", error.to_string()))?;
            if bytes.len() > MAX_RESPONSE_BYTES {
                return Err(source_error(
                    "binance_backfill_body_large",
                    "Binance response exceeded the body limit",
                ));
            }
            let wire: Vec<WireTrade> = serde_json::from_slice(&bytes)
                .map_err(|error| source_error("binance_backfill_decode", error.to_string()))?;
            let page_len = wire.len();
            if page_len == 0 {
                break;
            }
            let mut page = Vec::with_capacity(page_len);
            for row in wire {
                let trade = Trade::try_from(row)?;
                if trade.trade_timestamp >= shard.range_end {
                    break;
                }
                if trade.trade_timestamp >= shard.range_start {
                    page.push(trade);
                }
            }
            let last_id = page.last().map(|trade| trade.aggregate_trade_id);
            all.extend(page);
            if page_len < PAGE_LIMIT || last_id.is_none() {
                break;
            }
            from_id = last_id.and_then(|id| id.checked_add(1));
            if from_id.is_none() {
                return Err(BackfillExecutionError::new(
                    BackfillFailureKind::Integrity,
                    "binance_backfill_id_overflow",
                    "aggregate trade identifier overflowed",
                ));
            }
        }
        for pair in all.windows(2) {
            if pair[1].aggregate_trade_id <= pair[0].aggregate_trade_id {
                return Err(BackfillExecutionError::new(
                    BackfillFailureKind::Integrity,
                    "binance_backfill_unordered",
                    "Binance aggregate trades were not strictly ordered",
                ));
            }
        }
        Ok(all)
    }
}

#[derive(Debug, Deserialize)]
struct WireTrade {
    #[serde(rename = "a")]
    aggregate_trade_id: i64,
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "q")]
    quantity: String,
    #[serde(rename = "f")]
    first_trade_id: i64,
    #[serde(rename = "l")]
    last_trade_id: i64,
    #[serde(rename = "T")]
    trade_time_ms: i64,
    #[serde(rename = "m")]
    buyer_maker: bool,
    #[serde(rename = "M")]
    best_match: bool,
}

struct Trade {
    aggregate_trade_id: i64,
    trade_timestamp: DateTime<Utc>,
    price: Decimal,
    quantity: Decimal,
    first_trade_id: i64,
    last_trade_id: i64,
    buyer_maker: bool,
    best_match: bool,
    payload_sha256: String,
}

impl TryFrom<WireTrade> for Trade {
    type Error = BackfillExecutionError;

    fn try_from(value: WireTrade) -> Result<Self, Self::Error> {
        let trade_timestamp = DateTime::from_timestamp_millis(value.trade_time_ms)
            .ok_or_else(|| source_error("binance_backfill_timestamp", "invalid trade timestamp"))?;
        let price = value
            .price
            .parse::<Decimal>()
            .map_err(|error| source_error("binance_backfill_price", error.to_string()))?;
        let quantity = value
            .quantity
            .parse::<Decimal>()
            .map_err(|error| source_error("binance_backfill_quantity", error.to_string()))?;
        if value.aggregate_trade_id < 0
            || value.first_trade_id < 0
            || value.last_trade_id < value.first_trade_id
            || price <= Decimal::ZERO
            || quantity <= Decimal::ZERO
        {
            return Err(source_error(
                "binance_backfill_values",
                "invalid aggregate-trade values",
            ));
        }
        let canonical = format!(
            "v1|source={SOURCE}|symbol={SYMBOL}|aggregate_trade_id={}|trade_timestamp_ms={}|price={}|quantity={}|first_trade_id={}|last_trade_id={}|buyer_maker={}|best_match={}",
            value.aggregate_trade_id, value.trade_time_ms, price.normalize(), quantity.normalize(),
            value.first_trade_id, value.last_trade_id, value.buyer_maker, value.best_match,
        );
        Ok(Self {
            aggregate_trade_id: value.aggregate_trade_id,
            trade_timestamp,
            price,
            quantity,
            first_trade_id: value.first_trade_id,
            last_trade_id: value.last_trade_id,
            buyer_maker: value.buyer_maker,
            best_match: value.best_match,
            payload_sha256: format!("{:x}", Sha256::digest(canonical.as_bytes())),
        })
    }
}

async fn persist(
    context: &BackfillContext,
    trades: &[Trade],
) -> Result<i64, BackfillExecutionError> {
    let mut tx = context.pool.begin().await.map_err(database_error)?;
    require_lease(&mut tx, context).await?;
    let received_at = Utc::now();
    let minimum_source_timestamp = trades.first().map(|trade| trade.trade_timestamp);
    let maximum_source_timestamp = trades.last().map(|trade| trade.trade_timestamp);
    let mut content = Sha256::new();
    for trade in trades {
        content.update(trade.payload_sha256.as_bytes());
    }
    let content_sha256 = format!("{:x}", content.finalize());
    let record_count = i64::try_from(trades.len()).map_err(|_| {
        BackfillExecutionError::new(
            BackfillFailureKind::Integrity,
            "binance_backfill_count",
            "record count overflowed",
        )
    })?;
    let capture_artifact_id: uuid::Uuid = sqlx::query_scalar(
        r#"
        WITH existing AS (
          SELECT artifact.artifact_id
          FROM ingester.capture_artifacts artifact
          JOIN ingester.backfill_jobs job ON job.job_id=$1
          WHERE artifact.strategy_key=$7
            AND artifact.capture_window_start=job.range_start
            AND artifact.capture_window_end=job.range_end
            AND artifact.status='completed'
          ORDER BY artifact.created_at,artifact.artifact_id
          LIMIT 1
        ), inserted AS (
          INSERT INTO ingester.capture_artifacts (
            strategy_key,profile_generation,config_schema_version,config_sha256,
            config_snapshot,capture_window_start,capture_window_end,
            minimum_source_timestamp,maximum_source_timestamp,
            minimum_received_at,maximum_received_at,record_count,content_sha256,
            status,created_at,updated_at,completed_at
          )
          SELECT profile.strategy_key,profile.desired_generation,profile.config_schema_version,
            encode(digest(profile.config::text,'sha256'),'hex'),profile.config,
            job.range_start,job.range_end,$2,$3,$4,$4,$5,$6,'completed',now(),now(),now()
          FROM ingester.profiles profile
          JOIN ingester.backfill_jobs job ON job.job_id=$1
          WHERE profile.strategy_key=$7 AND NOT EXISTS (SELECT 1 FROM existing)
          RETURNING artifact_id
        )
        SELECT artifact_id FROM inserted
        UNION ALL
        SELECT artifact_id FROM existing
        LIMIT 1
        "#,
    )
    .bind(context.job_id)
    .bind(minimum_source_timestamp)
    .bind(maximum_source_timestamp)
    .bind(if trades.is_empty() {
        None
    } else {
        Some(received_at)
    })
    .bind(record_count)
    .bind(&content_sha256)
    .bind(STRATEGY_KEY)
    .fetch_one(&mut *tx)
    .await
    .map_err(database_error)?;
    for chunk in trades.chunks(1_000) {
        let mut builder = QueryBuilder::<Postgres>::new(
            "INSERT INTO market_data.binance_spot_btcusdt_aggregate_trades (source,symbol,aggregate_trade_id,trade_timestamp,provider_available_at,received_at,price,quantity,first_trade_id,last_trade_id,buyer_maker,best_match,payload_sha256,strategy_key,capture_artifact_id) ",
        );
        builder.push_values(chunk, |mut row, trade| {
            row.push_bind(SOURCE)
                .push_bind(SYMBOL)
                .push_bind(trade.aggregate_trade_id)
                .push_bind(trade.trade_timestamp)
                .push_bind(Option::<DateTime<Utc>>::None)
                .push_bind(received_at)
                .push_bind(trade.price)
                .push_bind(trade.quantity)
                .push_bind(trade.first_trade_id)
                .push_bind(trade.last_trade_id)
                .push_bind(trade.buyer_maker)
                .push_bind(trade.best_match)
                .push_bind(&trade.payload_sha256)
                .push_bind(STRATEGY_KEY)
                .push_bind(capture_artifact_id);
        });
        builder.push(" ON CONFLICT (symbol,trade_timestamp,aggregate_trade_id) DO NOTHING");
        builder
            .build()
            .execute(&mut *tx)
            .await
            .map_err(database_error)?;
    }
    let ids: Vec<i64> = trades
        .iter()
        .map(|trade| trade.aggregate_trade_id)
        .collect();
    let durable = sqlx::query(
        "SELECT aggregate_trade_id,payload_sha256::text FROM market_data.binance_spot_btcusdt_aggregate_trades WHERE symbol='BTCUSDT' AND aggregate_trade_id=ANY($1::bigint[])",
    )
    .bind(&ids).fetch_all(&mut *tx).await.map_err(database_error)?;
    if durable.len() != trades.len() {
        return Err(BackfillExecutionError::new(
            BackfillFailureKind::Integrity,
            "binance_backfill_missing",
            "durable aggregate-trade count did not match the provider page",
        ));
    }
    let expected: std::collections::BTreeMap<_, _> = trades
        .iter()
        .map(|trade| (trade.aggregate_trade_id, trade.payload_sha256.as_str()))
        .collect();
    for row in durable {
        let id: i64 = row.get("aggregate_trade_id");
        let hash: String = row.get("payload_sha256");
        if expected.get(&id).copied() != Some(hash.as_str()) {
            return Err(BackfillExecutionError::new(
                BackfillFailureKind::Integrity,
                "binance_backfill_conflict",
                format!("durable aggregate trade {id} conflicts with provider data"),
            ));
        }
    }
    sqlx::query(
        r#"
        INSERT INTO ingester.backfill_artifacts (
          job_id,strategy_key,logical_key,provider,source_uri,checksum,record_count,
          minimum_source_timestamp,maximum_source_timestamp,durable_target,status,
          metadata,completed_at
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'completed',$11,now())
        ON CONFLICT (strategy_key,logical_key) DO UPDATE SET
          checksum=EXCLUDED.checksum,record_count=EXCLUDED.record_count,
          minimum_source_timestamp=EXCLUDED.minimum_source_timestamp,
          maximum_source_timestamp=EXCLUDED.maximum_source_timestamp,
          metadata=EXCLUDED.metadata,completed_at=EXCLUDED.completed_at,status='completed'
        "#,
    )
    .bind(context.job_id)
    .bind(STRATEGY_KEY)
    .bind(context.job_id.to_string())
    .bind(SOURCE)
    .bind(REST_URL)
    .bind(&content_sha256)
    .bind(record_count)
    .bind(minimum_source_timestamp)
    .bind(maximum_source_timestamp)
    .bind("market_data.binance_spot_btcusdt_aggregate_trades")
    .bind(json!({"capture_artifact_id": capture_artifact_id}))
    .execute(&mut *tx)
    .await
    .map_err(database_error)?;
    require_lease(&mut tx, context).await?;
    tx.commit().await.map_err(database_error)?;
    Ok(record_count)
}

async fn require_lease(
    tx: &mut Transaction<'_, Postgres>,
    context: &BackfillContext,
) -> Result<(), BackfillExecutionError> {
    let valid: bool = sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM ingester.backfill_jobs WHERE job_id=$1 AND assigned_worker_id=$2 AND lease_token=$3 AND status='running' AND lease_expires_at>now())",
    )
    .bind(context.job_id).bind(context.worker_id.as_ref()).bind(context.lease_token)
    .fetch_one(&mut **tx).await.map_err(database_error)?;
    if !valid {
        return Err(BackfillExecutionError::new(
            BackfillFailureKind::LeaseLost,
            "lease_lost",
            "backfill job lease was lost",
        ));
    }
    Ok(())
}

fn source_error(code: &'static str, message: impl Into<String>) -> BackfillExecutionError {
    BackfillExecutionError::new(BackfillFailureKind::TransientSource, code, message)
}

fn database_error(error: sqlx::Error) -> BackfillExecutionError {
    BackfillExecutionError::new(
        BackfillFailureKind::TransientDatabase,
        "database_error",
        error.to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_plan_is_deterministic_and_minute_bounded() {
        let strategy = BinanceSpotAggregateTradesBackfill::new().unwrap();
        let start = "2026-08-01T00:15:00Z".parse().unwrap();
        let request = ValidatedBackfillRequest {
            strategy_key: Arc::from(STRATEGY_KEY),
            strategy_contract_version: 1,
            request_schema_version: 1,
            range_start: start,
            range_end: start + chrono::Duration::hours(3),
            parameters: json!({}),
            execution: Default::default(),
        };
        let first = strategy.plan_shards(&request).unwrap();
        let second = strategy.plan_shards(&request).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 180);
        assert!(first
            .iter()
            .all(|shard| shard.range_end - shard.range_start <= chrono::Duration::minutes(1)));
    }
}
