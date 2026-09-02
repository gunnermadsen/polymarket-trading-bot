use std::{collections::BTreeSet, path::PathBuf, sync::Arc, time::Duration};

use anyhow::{bail, Context as _, Result as AnyResult};
use async_trait::async_trait;
use chrono::{DateTime, Duration as ChronoDuration, Timelike, Utc};
use reqwest::Client;
use rust_decimal::Decimal;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::{Postgres, QueryBuilder, Row, Transaction};
use uuid::Uuid;

use crate::domain::{
    BackfillContext, BackfillExecutionError, BackfillFailureKind, BackfillOutcome, BackfillRequest,
    BackfillShard, BackfillWorkerStrategy, StrategyCapability, StrategyDescriptor,
    ValidatedBackfillRequest,
};

use super::{
    backfill_types::{
        ArchiveCancellation, ArchiveDownloadLimits, BtcExecutionSnapshot, BtcIntervalMarket,
        BtcOrderbookArchiveEvent, BtcOrderbookMarketScope, BtcOutcome,
    },
    execution_snapshots::{
        ExecutionSnapshotReconstructor, EXECUTION_SNAPSHOTS_PER_MARKET,
        EXECUTION_SNAPSHOT_SCHEMA_VERSION,
    },
    pmxt_archive::{
        download_archive, spawn_execution_parser, spawn_parser, PmxtArchiveSpec,
        DEFAULT_PMXT_ARCHIVE_URL, PMXT_ARCHIVE_PROVIDER, PMXT_COVERAGE_START_EPOCH,
    },
};

pub const MARKET_CONTRACTS_BACKFILL_KEY: &str =
    "polymarket_btc_five_minute_market_contracts_backfill";
pub const RESOLUTIONS_BACKFILL_KEY: &str = "polymarket_btc_five_minute_resolutions_backfill";
pub const ORDERBOOK_EVENTS_BACKFILL_KEY: &str =
    "polymarket_btc_five_minute_orderbook_events_backfill";
pub const EXECUTION_SNAPSHOTS_BACKFILL_KEY: &str =
    "polymarket_btc_five_minute_execution_snapshots_backfill";

const GAMMA_URL: &str = "https://gamma-api.polymarket.com";
const CLOB_URL: &str = "https://clob.polymarket.com";
const MAX_JSON_BYTES: usize = 2 * 1024 * 1024;
const MAX_SHARDS: usize = 10_080;
const DATABASE_BATCH_ROWS: usize = 1_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    MarketContracts,
    Resolutions,
    OrderbookEvents,
    ExecutionSnapshots,
}

pub struct PolymarketBtcBackfill {
    kind: Kind,
    descriptor: StrategyDescriptor,
    client: Client,
    gamma_url: Arc<str>,
    clob_url: Arc<str>,
    pmxt_url: Arc<str>,
    cache_directory: PathBuf,
}

impl PolymarketBtcBackfill {
    pub fn market_contracts() -> Result<Self, BackfillExecutionError> {
        Self::new(Kind::MarketContracts)
    }

    pub fn resolutions() -> Result<Self, BackfillExecutionError> {
        Self::new(Kind::Resolutions)
    }

    pub fn orderbook_events() -> Result<Self, BackfillExecutionError> {
        Self::new(Kind::OrderbookEvents)
    }

    pub fn execution_snapshots() -> Result<Self, BackfillExecutionError> {
        Self::new(Kind::ExecutionSnapshots)
    }

    fn new(kind: Kind) -> Result<Self, BackfillExecutionError> {
        let (strategy_key, name, description) = match kind {
            Kind::MarketContracts => (
                MARKET_CONTRACTS_BACKFILL_KEY,
                "Polymarket BTC five-minute market contracts backfill",
                "Collects historical BTC five-minute market contracts from Gamma",
            ),
            Kind::Resolutions => (
                RESOLUTIONS_BACKFILL_KEY,
                "Polymarket BTC five-minute resolutions backfill",
                "Collects official historical BTC five-minute resolutions from CLOB",
            ),
            Kind::OrderbookEvents => (
                ORDERBOOK_EVENTS_BACKFILL_KEY,
                "Polymarket BTC five-minute orderbook events backfill",
                "Collects historical BTC five-minute raw orderbook events from PMXT",
            ),
            Kind::ExecutionSnapshots => (
                EXECUTION_SNAPSHOTS_BACKFILL_KEY,
                "Polymarket BTC five-minute execution snapshots backfill",
                "Reconstructs historical BTC five-minute execution snapshots from PMXT",
            ),
        };
        let descriptor = StrategyDescriptor {
            strategy_key: Arc::from(strategy_key),
            name: Arc::from(name),
            description: Arc::from(description),
            capabilities: vec![StrategyCapability::Backfill],
            strategy_contract_version: 1,
            request_schema_version: Some(1),
            shardable: true,
            maximum_shards: MAX_SHARDS,
        };
        descriptor.validate()?;
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(120))
            .user_agent("capitonic-ingester-worker/1")
            .build()
            .map_err(|error| integrity("polymarket_backfill_client", error.to_string()))?;
        Ok(Self {
            kind,
            descriptor,
            client,
            gamma_url: env_url("POLYMARKET_GAMMA_BASE_URL", GAMMA_URL),
            clob_url: env_url("POLYMARKET_CLOB_BASE_URL", CLOB_URL),
            pmxt_url: env_url("POLYMARKET_PMXT_ARCHIVE_BASE_URL", DEFAULT_PMXT_ARCHIVE_URL),
            cache_directory: std::env::var("INGESTER_PMXT_CACHE_DIRECTORY")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/tmp/ingester-pmxt-cache")),
        })
    }

    fn key(&self) -> &'static str {
        match self.kind {
            Kind::MarketContracts => MARKET_CONTRACTS_BACKFILL_KEY,
            Kind::Resolutions => RESOLUTIONS_BACKFILL_KEY,
            Kind::OrderbookEvents => ORDERBOOK_EVENTS_BACKFILL_KEY,
            Kind::ExecutionSnapshots => EXECUTION_SNAPSHOTS_BACKFILL_KEY,
        }
    }
}
fn outcome(records: i64, shard: &BackfillShard, summary: Value) -> BackfillOutcome {
    BackfillOutcome {
        records_verified: records,
        verified_coverage: json!({
            "requested_start": shard.range_start,
            "requested_end": shard.range_end,
            "records_verified": records,
        }),
        summary,
    }
}

async fn completed_outcome(
    context: &BackfillContext,
    strategy_key: &str,
    logical_key: &str,
) -> Result<Option<BackfillOutcome>, BackfillExecutionError> {
    let row = sqlx::query(
        "SELECT record_count,minimum_source_timestamp,maximum_source_timestamp,metadata FROM ingester.backfill_artifacts WHERE strategy_key=$1 AND logical_key=$2 AND status='completed'",
    )
    .bind(strategy_key)
    .bind(logical_key)
    .fetch_optional(&context.pool)
    .await
    .map_err(database_error)?;
    Ok(row.map(|row| {
        let records: i64 = row.try_get::<Option<i64>, _>("record_count").ok().flatten().unwrap_or(0);
        BackfillOutcome {
            records_verified: records,
            verified_coverage: json!({
                "minimum_source_timestamp": row.try_get::<Option<DateTime<Utc>>, _>("minimum_source_timestamp").ok().flatten(),
                "maximum_source_timestamp": row.try_get::<Option<DateTime<Utc>>, _>("maximum_source_timestamp").ok().flatten(),
                "records_verified": records,
                "reused_artifact": true,
            }),
            summary: row.try_get::<Value, _>("metadata").unwrap_or_else(|_| json!({})),
        }
    }))
}

async fn completed_source_outcome(
    context: &BackfillContext,
    provider: &str,
    logical_key: &str,
    durable_target: &str,
) -> Result<Option<BackfillOutcome>, BackfillExecutionError> {
    let row = sqlx::query(
        "SELECT record_count,minimum_source_timestamp,maximum_source_timestamp,metadata,strategy_key FROM ingester.backfill_artifacts WHERE provider=$1 AND logical_key=$2 AND durable_target=$3 AND status='completed' ORDER BY completed_at DESC NULLS LAST LIMIT 1",
    )
    .bind(provider)
    .bind(logical_key)
    .bind(durable_target)
    .fetch_optional(&context.pool)
    .await
    .map_err(database_error)?;
    Ok(row.map(|row| {
        let records = row
            .try_get::<Option<i64>, _>("record_count")
            .ok()
            .flatten()
            .unwrap_or(0);
        BackfillOutcome {
            records_verified: records,
            verified_coverage: json!({
                "minimum_source_timestamp": row.try_get::<Option<DateTime<Utc>>, _>("minimum_source_timestamp").ok().flatten(),
                "maximum_source_timestamp": row.try_get::<Option<DateTime<Utc>>, _>("maximum_source_timestamp").ok().flatten(),
                "records_verified": records,
                "reused_artifact": true,
            }),
            summary: json!({
                "provider": provider,
                "reused_artifact": true,
                "source_strategy_key": row.try_get::<String, _>("strategy_key").ok(),
                "source_metadata": row.try_get::<Value, _>("metadata").unwrap_or_else(|_| json!({})),
            }),
        }
    }))
}

async fn create_artifact(
    context: &BackfillContext,
    strategy_key: &str,
    logical_key: &str,
    provider: &str,
    source_uri: &str,
    durable_target: &str,
) -> Result<Uuid, BackfillExecutionError> {
    let mut tx = context.pool.begin().await.map_err(database_error)?;
    require_lease(&mut tx, context).await?;
    let artifact_id = sqlx::query_scalar(
        r#"
        INSERT INTO ingester.backfill_artifacts (
          job_id,strategy_key,ingester_key,logical_key,provider,source_uri,
          durable_target,status,metadata
        ) VALUES ($1,$2,$2,$3,$4,$5,$6,'ingesting','{}'::jsonb)
        ON CONFLICT (strategy_key,logical_key) DO UPDATE SET
          job_id=EXCLUDED.job_id,status='ingesting',updated_at=now()
        WHERE ingester.backfill_artifacts.status <> 'completed'
        RETURNING artifact_id
        "#,
    )
    .bind(context.job_id)
    .bind(strategy_key)
    .bind(logical_key)
    .bind(provider)
    .bind(source_uri)
    .bind(durable_target)
    .fetch_one(&mut *tx)
    .await
    .map_err(database_error)?;
    tx.commit().await.map_err(database_error)?;
    Ok(artifact_id)
}

async fn complete_artifact(
    context: &BackfillContext,
    artifact_id: Uuid,
    records: i64,
    checksum: &str,
    shard: &BackfillShard,
    metadata: Value,
) -> Result<(), BackfillExecutionError> {
    let mut tx = context.pool.begin().await.map_err(database_error)?;
    require_lease(&mut tx, context).await?;
    let updated = sqlx::query(
        r#"
        UPDATE ingester.backfill_artifacts SET
          checksum=$2,actual_checksum=$2,record_count=$3,
          minimum_source_timestamp=$4,maximum_source_timestamp=$5,
          status='completed',metadata=$6,completed_at=now(),updated_at=now()
        WHERE artifact_id=$1 AND job_id=$7 AND status <> 'completed'
        "#,
    )
    .bind(artifact_id)
    .bind(checksum)
    .bind(records)
    .bind(shard.range_start)
    .bind(shard.range_end)
    .bind(metadata)
    .bind(context.job_id)
    .execute(&mut *tx)
    .await
    .map_err(database_error)?;
    if updated.rows_affected() != 1 {
        return Err(integrity(
            "artifact_completion_conflict",
            "backfill artifact was not writable by the active job",
        ));
    }
    require_lease(&mut tx, context).await?;
    tx.commit().await.map_err(database_error)
}

async fn require_lease(
    tx: &mut Transaction<'_, Postgres>,
    context: &BackfillContext,
) -> Result<(), BackfillExecutionError> {
    let valid: bool = sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM ingester.backfill_jobs WHERE job_id=$1 AND assigned_worker_id=$2 AND lease_token=$3 AND status='running' AND lease_expires_at>now())",
    )
    .bind(context.job_id)
    .bind(context.worker_id.as_ref())
    .bind(context.lease_token)
    .fetch_one(&mut **tx)
    .await
    .map_err(database_error)?;
    if !valid {
        return Err(BackfillExecutionError::new(
            BackfillFailureKind::LeaseLost,
            "lease_lost",
            "backfill job lease was lost",
        ));
    }
    Ok(())
}

fn ensure_running(context: &BackfillContext) -> Result<(), BackfillExecutionError> {
    if context.shutdown.is_cancelled() {
        Err(cancelled())
    } else {
        Ok(())
    }
}

fn cancellation_for(context: &BackfillContext) -> ArchiveCancellation {
    let cancellation = ArchiveCancellation::default();
    let mirrored = cancellation.clone();
    let shutdown = context.shutdown.clone();
    tokio::spawn(async move {
        shutdown.cancelled().await;
        mirrored.cancel();
    });
    cancellation
}

async fn persist_market(
    context: &BackfillContext,
    market: &BtcIntervalMarket,
) -> Result<(), BackfillExecutionError> {
    let mut tx = context.pool.begin().await.map_err(database_error)?;
    require_lease(&mut tx, context).await?;
    let minimum_order_size = market.minimum_order_size.unwrap_or(Decimal::ZERO);
    let (validation_status, validation_errors) = if market.minimum_order_size.is_some() {
        ("valid", json!([]))
    } else {
        ("ineligible", json!(["missing_minimum_order_size"]))
    };
    let question = market
        .raw_payload
        .pointer("/markets/0/question")
        .and_then(Value::as_str)
        .or_else(|| market.raw_payload.get("title").and_then(Value::as_str))
        .unwrap_or(&market.event_slug);
    let now = Utc::now();
    let fee_rate = decimal_json_field(&market.fee_schedule, &["rate"]);
    let fee_exponent = integer_json_field(&market.fee_schedule, &["exponent"]);
    let fee_taker_only = bool_json_field(&market.fee_schedule, &["takerOnly", "taker_only"]);
    let result = sqlx::query(
        r#"
        INSERT INTO polymarket.btc_interval_markets (
          market_id,event_id,event_slug,question,series_slug,window_start,window_end,
          condition_id,up_token_id,down_token_id,resolution_source,accepting_orders,
          active,closed,min_tick_size,min_order_size,fee_rate,fee_exponent,fee_taker_only,
          validation_status,validation_errors,discovered_at,last_refreshed_at,raw_payload
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24)
        ON CONFLICT (market_id) DO UPDATE SET
          question=EXCLUDED.question,accepting_orders=EXCLUDED.accepting_orders,
          active=EXCLUDED.active,closed=EXCLUDED.closed,min_tick_size=EXCLUDED.min_tick_size,
          min_order_size=EXCLUDED.min_order_size,fee_rate=EXCLUDED.fee_rate,
          fee_exponent=EXCLUDED.fee_exponent,fee_taker_only=EXCLUDED.fee_taker_only,
          validation_status=EXCLUDED.validation_status,
          validation_errors=EXCLUDED.validation_errors,last_refreshed_at=EXCLUDED.last_refreshed_at,
          raw_payload=EXCLUDED.raw_payload,updated_at=now()
        WHERE polymarket.btc_interval_markets.event_id=EXCLUDED.event_id
          AND polymarket.btc_interval_markets.event_slug=EXCLUDED.event_slug
          AND polymarket.btc_interval_markets.series_slug=EXCLUDED.series_slug
          AND polymarket.btc_interval_markets.window_start=EXCLUDED.window_start
          AND polymarket.btc_interval_markets.window_end=EXCLUDED.window_end
          AND polymarket.btc_interval_markets.condition_id=EXCLUDED.condition_id
          AND polymarket.btc_interval_markets.up_token_id=EXCLUDED.up_token_id
          AND polymarket.btc_interval_markets.down_token_id=EXCLUDED.down_token_id
          AND polymarket.btc_interval_markets.resolution_source=EXCLUDED.resolution_source
        "#,
    )
    .bind(&market.market_id)
    .bind(&market.event_id)
    .bind(&market.event_slug)
    .bind(question)
    .bind(&market.series_slug)
    .bind(market.window_start)
    .bind(market.window_end)
    .bind(&market.condition_id)
    .bind(&market.up_token_id)
    .bind(&market.down_token_id)
    .bind(&market.resolution_source)
    .bind(market.accepting_orders)
    .bind(market.active)
    .bind(market.closed)
    .bind(market.tick_size)
    .bind(minimum_order_size)
    .bind(fee_rate)
    .bind(fee_exponent)
    .bind(fee_taker_only)
    .bind(validation_status)
    .bind(validation_errors)
    .bind(now)
    .bind(now)
    .bind(&market.raw_payload)
    .execute(&mut *tx)
    .await
    .map_err(database_error)?;
    if result.rows_affected() != 1 {
        return Err(integrity(
            "market_identity_conflict",
            format!(
                "immutable BTC market identity conflict for {}",
                market.market_id
            ),
        ));
    }
    require_lease(&mut tx, context).await?;
    tx.commit().await.map_err(database_error)
}

struct ReferenceFact<'a> {
    artifact_id: Uuid,
    market_id: &'a str,
    fact_type: &'a str,
    value: Decimal,
    source_effective_at: DateTime<Utc>,
    payload_sha256: &'a str,
    evidence: &'a Value,
}

async fn persist_reference_fact(
    context: &BackfillContext,
    fact: ReferenceFact<'_>,
) -> Result<(), BackfillExecutionError> {
    let mut tx = context.pool.begin().await.map_err(database_error)?;
    require_lease(&mut tx, context).await?;
    let inserted = sqlx::query(
        r#"
        INSERT INTO polymarket.btc_market_reference_facts (
          fact_id,market_id,artifact_id,fact_type,value,provider,source_effective_at,
          fetched_at,payload_sha256,evidence
        ) VALUES (gen_random_uuid(),$1,$2,$3,$4,'polymarket_gamma',$5,now(),$6,$7)
        ON CONFLICT (market_id,fact_type,provider) DO NOTHING
        "#,
    )
    .bind(fact.market_id)
    .bind(fact.artifact_id)
    .bind(fact.fact_type)
    .bind(fact.value)
    .bind(fact.source_effective_at)
    .bind(fact.payload_sha256)
    .bind(fact.evidence)
    .execute(&mut *tx)
    .await
    .map_err(database_error)?
    .rows_affected();
    if inserted == 0 {
        let row = sqlx::query(
            "SELECT value,source_effective_at,payload_sha256 FROM polymarket.btc_market_reference_facts WHERE market_id=$1 AND fact_type=$2 AND provider='polymarket_gamma'",
        )
        .bind(fact.market_id)
        .bind(fact.fact_type)
        .fetch_one(&mut *tx)
        .await
        .map_err(database_error)?;
        let stored_value: Decimal = row.try_get("value").map_err(database_error)?;
        let stored_at: DateTime<Utc> =
            row.try_get("source_effective_at").map_err(database_error)?;
        let stored_hash: String = row.try_get("payload_sha256").map_err(database_error)?;
        if stored_value != fact.value
            || stored_at != fact.source_effective_at
            || stored_hash != fact.payload_sha256
        {
            return Err(integrity(
                "market_reference_conflict",
                format!(
                    "immutable BTC {} fact conflict for {}",
                    fact.fact_type, fact.market_id
                ),
            ));
        }
    }
    require_lease(&mut tx, context).await?;
    tx.commit().await.map_err(database_error)
}

fn decimal_json_field(value: &Value, keys: &[&str]) -> Option<Decimal> {
    keys.iter().find_map(|key| match value.get(*key)? {
        Value::String(value) => value.parse().ok(),
        Value::Number(value) => value.to_string().parse().ok(),
        _ => None,
    })
}

fn integer_json_field(value: &Value, keys: &[&str]) -> Option<i32> {
    keys.iter()
        .find_map(|key| value.get(*key)?.as_i64())
        .and_then(|value| i32::try_from(value).ok())
}

fn bool_json_field(value: &Value, keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|key| value.get(*key)?.as_bool())
}

struct MarketCandidate {
    market: BtcIntervalMarket,
    official_outcome: Option<String>,
}

async fn load_markets(
    context: &BackfillContext,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<Vec<MarketCandidate>, BackfillExecutionError> {
    let rows = sqlx::query(
        r#"
        SELECT event_id,event_slug,series_slug,market_id,condition_id,window_start,window_end,
          up_token_id,down_token_id,min_tick_size,min_order_size,resolution_source,
          accepting_orders,active,closed,raw_payload,official_outcome
        FROM polymarket.btc_interval_markets
        WHERE window_start >= $1 AND window_start < $2
        ORDER BY window_start,market_id
        "#,
    )
    .bind(start)
    .bind(end)
    .fetch_all(&context.pool)
    .await
    .map_err(database_error)?;
    rows.into_iter()
        .map(|row| {
            Ok(MarketCandidate {
                market: BtcIntervalMarket {
                    event_id: row.try_get("event_id").map_err(database_error)?,
                    event_slug: row.try_get("event_slug").map_err(database_error)?,
                    series_slug: row.try_get("series_slug").map_err(database_error)?,
                    market_id: row.try_get("market_id").map_err(database_error)?,
                    condition_id: row.try_get("condition_id").map_err(database_error)?,
                    window_start: row.try_get("window_start").map_err(database_error)?,
                    window_end: row.try_get("window_end").map_err(database_error)?,
                    up_token_id: row.try_get("up_token_id").map_err(database_error)?,
                    down_token_id: row.try_get("down_token_id").map_err(database_error)?,
                    tick_size: row.try_get("min_tick_size").map_err(database_error)?,
                    minimum_order_size: row.try_get("min_order_size").map_err(database_error)?,
                    resolution_source: row.try_get("resolution_source").map_err(database_error)?,
                    accepting_orders: row.try_get("accepting_orders").map_err(database_error)?,
                    active: row.try_get("active").map_err(database_error)?,
                    closed: row.try_get("closed").map_err(database_error)?,
                    fees_enabled: false,
                    fee_schedule: json!({}),
                    raw_payload: row.try_get("raw_payload").map_err(database_error)?,
                },
                official_outcome: row.try_get("official_outcome").map_err(database_error)?,
            })
        })
        .collect()
}

struct ClobOfficialResolution {
    winning_token_id: String,
    winning_outcome: BtcOutcome,
    observed_at: DateTime<Utc>,
}

async fn persist_resolution(
    context: &BackfillContext,
    market: &BtcIntervalMarket,
    resolution: &ClobOfficialResolution,
    payload: &Value,
) -> Result<(), BackfillExecutionError> {
    let mut tx = context.pool.begin().await.map_err(database_error)?;
    require_lease(&mut tx, context).await?;
    let row = sqlx::query(
        "SELECT official_outcome,official_winning_token_id FROM polymarket.btc_interval_markets WHERE market_id=$1 AND condition_id=$2 FOR UPDATE",
    )
    .bind(&market.market_id)
    .bind(&market.condition_id)
    .fetch_one(&mut *tx)
    .await
    .map_err(database_error)?;
    let existing_outcome: Option<String> =
        row.try_get("official_outcome").map_err(database_error)?;
    let existing_winner: Option<String> = row
        .try_get("official_winning_token_id")
        .map_err(database_error)?;
    if existing_outcome.is_some()
        && (existing_outcome.as_deref() != Some(resolution.winning_outcome.as_str())
            || existing_winner.as_deref() != Some(&resolution.winning_token_id))
    {
        return Err(integrity(
            "resolution_identity_conflict",
            format!(
                "immutable official resolution conflict for {}",
                market.market_id
            ),
        ));
    }
    sqlx::query(
        r#"
        UPDATE polymarket.btc_interval_markets SET
          official_outcome=COALESCE(official_outcome,$2),
          official_resolved_at=COALESCE(official_resolved_at,$3),
          official_winning_token_id=COALESCE(official_winning_token_id,$4),
          official_resolution_source=COALESCE(official_resolution_source,'clob_rest_reconciliation'),
          official_resolution_received_at=COALESCE(official_resolution_received_at,$3),
          official_resolution_payload=COALESCE(official_resolution_payload,$5),
          resolved_outcome=COALESCE(resolved_outcome,$2),updated_at=now()
        WHERE market_id=$1
        "#,
    )
    .bind(&market.market_id)
    .bind(resolution.winning_outcome.as_str())
    .bind(resolution.observed_at)
    .bind(&resolution.winning_token_id)
    .bind(payload)
    .execute(&mut *tx)
    .await
    .map_err(database_error)?;
    require_lease(&mut tx, context).await?;
    tx.commit().await.map_err(database_error)
}

async fn load_scope(
    context: &BackfillContext,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    include_end: bool,
) -> Result<Vec<BtcOrderbookMarketScope>, BackfillExecutionError> {
    let comparison = if include_end { "<=" } else { "<" };
    let query = format!(
        "SELECT market_id,condition_id,up_token_id,down_token_id,window_start,window_end FROM polymarket.btc_interval_markets WHERE validation_status='valid' AND window_start >= $1 AND window_start {comparison} $2 ORDER BY window_start,market_id"
    );
    let rows = sqlx::query(&query)
        .bind(start)
        .bind(end)
        .fetch_all(&context.pool)
        .await
        .map_err(database_error)?;
    rows.into_iter()
        .map(|row| {
            Ok(BtcOrderbookMarketScope {
                market_id: row.try_get("market_id").map_err(database_error)?,
                condition_id: row.try_get("condition_id").map_err(database_error)?,
                up_token_id: row.try_get("up_token_id").map_err(database_error)?,
                down_token_id: row.try_get("down_token_id").map_err(database_error)?,
                window_start: row.try_get("window_start").map_err(database_error)?,
                window_end: row.try_get("window_end").map_err(database_error)?,
            })
        })
        .collect()
}

async fn persist_orderbook_events(
    context: &BackfillContext,
    artifact_id: Uuid,
    records: &[BtcOrderbookArchiveEvent],
) -> Result<i64, BackfillExecutionError> {
    if records.is_empty() {
        return Ok(0);
    }
    let mut tx = context.pool.begin().await.map_err(database_error)?;
    require_lease(&mut tx, context).await?;
    let mut inserted = 0u64;
    for chunk in records.chunks(DATABASE_BATCH_ROWS) {
        let mut query = QueryBuilder::<Postgres>::new(
            "INSERT INTO polymarket.btc_orderbook_archive_events (artifact_id,source_row_number,provider_received_at,source_timestamp,condition_id,asset_id,event_type,bids,asks,price,size,side,best_bid,best_ask,fee_rate_bps,transaction_hash,old_tick_size,new_tick_size) ",
        );
        query.push_values(chunk, |mut row, record| {
            row.push_bind(artifact_id)
                .push_bind(record.source_row_number)
                .push_bind(record.provider_received_at)
                .push_bind(record.source_timestamp)
                .push_bind(&record.condition_id)
                .push_bind(&record.asset_id)
                .push_bind(&record.event_type)
                .push_bind(&record.bids)
                .push_bind(&record.asks)
                .push_bind(record.price)
                .push_bind(record.size)
                .push_bind(&record.side)
                .push_bind(record.best_bid)
                .push_bind(record.best_ask)
                .push_bind(record.fee_rate_bps)
                .push_bind(&record.transaction_hash)
                .push_bind(record.old_tick_size)
                .push_bind(record.new_tick_size);
        });
        query.push(" ON CONFLICT (artifact_id,source_row_number,provider_received_at) DO NOTHING");
        inserted = inserted.saturating_add(
            query
                .build()
                .execute(&mut *tx)
                .await
                .map_err(database_error)?
                .rows_affected(),
        );
    }
    require_lease(&mut tx, context).await?;
    tx.commit().await.map_err(database_error)?;
    i64::try_from(inserted)
        .map_err(|_| integrity("record_count_overflow", "record count overflowed"))
}

async fn persist_execution_snapshots(
    context: &BackfillContext,
    artifact_id: Uuid,
    records: &[BtcExecutionSnapshot],
) -> Result<i64, BackfillExecutionError> {
    let mut tx = context.pool.begin().await.map_err(database_error)?;
    require_lease(&mut tx, context).await?;
    let mut inserted = 0u64;
    for chunk in records.chunks(250) {
        let mut query = QueryBuilder::<Postgres>::new(
            "INSERT INTO polymarket.btc_market_capacity_execution_snapshots (market_id,sampled_at,artifact_id,schema_version,up_source_row_number,up_source_timestamp,up_provider_received_at,up_best_bid,up_best_ask,up_best_bid_size,up_best_ask_size,up_bid_depth,up_ask_depth,up_ask_vwap_1,up_ask_vwap_5,up_ask_vwap_10,up_ask_vwap_15,up_ask_vwap_20,up_ask_vwap_25,up_ask_vwap_30,up_ask_vwap_40,up_ask_vwap_50,up_ask_vwap_75,up_ask_vwap_100,up_ask_vwap_125,up_ask_vwap_150,up_ask_vwap_175,up_ask_vwap_200,up_imbalance,down_source_row_number,down_source_timestamp,down_provider_received_at,down_best_bid,down_best_ask,down_best_bid_size,down_best_ask_size,down_bid_depth,down_ask_depth,down_ask_vwap_1,down_ask_vwap_5,down_ask_vwap_10,down_ask_vwap_15,down_ask_vwap_20,down_ask_vwap_25,down_ask_vwap_30,down_ask_vwap_40,down_ask_vwap_50,down_ask_vwap_75,down_ask_vwap_100,down_ask_vwap_125,down_ask_vwap_150,down_ask_vwap_175,down_ask_vwap_200,down_imbalance,quality_flags) ",
        );
        query.push_values(chunk, |mut row, record| {
            row.push_bind(&record.market_id)
                .push_bind(record.sampled_at)
                .push_bind(artifact_id)
                .push_bind(EXECUTION_SNAPSHOT_SCHEMA_VERSION)
                .push_bind(record.up_source_row_number)
                .push_bind(record.up_source_timestamp)
                .push_bind(record.up_provider_received_at)
                .push_bind(record.up_best_bid)
                .push_bind(record.up_best_ask)
                .push_bind(record.up_best_bid_size)
                .push_bind(record.up_best_ask_size)
                .push_bind(record.up_bid_depth)
                .push_bind(record.up_ask_depth)
                .push_bind(record.up_ask_vwap_1)
                .push_bind(record.up_ask_vwap_5)
                .push_bind(record.up_ask_vwap_10)
                .push_bind(record.up_ask_vwap_15)
                .push_bind(record.up_ask_vwap_20)
                .push_bind(record.up_ask_vwap_25)
                .push_bind(record.up_ask_vwap_30)
                .push_bind(record.up_ask_vwap_40)
                .push_bind(record.up_ask_vwap_50)
                .push_bind(record.up_ask_vwap_75)
                .push_bind(record.up_ask_vwap_100)
                .push_bind(record.up_ask_vwap_125)
                .push_bind(record.up_ask_vwap_150)
                .push_bind(record.up_ask_vwap_175)
                .push_bind(record.up_ask_vwap_200)
                .push_bind(record.up_imbalance)
                .push_bind(record.down_source_row_number)
                .push_bind(record.down_source_timestamp)
                .push_bind(record.down_provider_received_at)
                .push_bind(record.down_best_bid)
                .push_bind(record.down_best_ask)
                .push_bind(record.down_best_bid_size)
                .push_bind(record.down_best_ask_size)
                .push_bind(record.down_bid_depth)
                .push_bind(record.down_ask_depth)
                .push_bind(record.down_ask_vwap_1)
                .push_bind(record.down_ask_vwap_5)
                .push_bind(record.down_ask_vwap_10)
                .push_bind(record.down_ask_vwap_15)
                .push_bind(record.down_ask_vwap_20)
                .push_bind(record.down_ask_vwap_25)
                .push_bind(record.down_ask_vwap_30)
                .push_bind(record.down_ask_vwap_40)
                .push_bind(record.down_ask_vwap_50)
                .push_bind(record.down_ask_vwap_75)
                .push_bind(record.down_ask_vwap_100)
                .push_bind(record.down_ask_vwap_125)
                .push_bind(record.down_ask_vwap_150)
                .push_bind(record.down_ask_vwap_175)
                .push_bind(record.down_ask_vwap_200)
                .push_bind(record.down_imbalance)
                .push_bind(record.quality_flags);
        });
        query.push(" ON CONFLICT (market_id,sampled_at) DO NOTHING");
        inserted = inserted.saturating_add(
            query
                .build()
                .execute(&mut *tx)
                .await
                .map_err(database_error)?
                .rows_affected(),
        );
    }
    require_lease(&mut tx, context).await?;
    tx.commit().await.map_err(database_error)?;
    let market_ids = records
        .iter()
        .map(|row| row.market_id.as_str())
        .collect::<BTreeSet<_>>();
    let minimum_sample = records.first().map(|row| row.sampled_at);
    let maximum_sample = records.last().map(|row| row.sampled_at);
    let verified: i64 = sqlx::query_scalar(
        "SELECT count(*)::bigint FROM polymarket.btc_market_capacity_execution_snapshots WHERE market_id=ANY($1::text[]) AND sampled_at >= $2 AND sampled_at <= $3",
    )
    .bind(market_ids.into_iter().collect::<Vec<_>>())
    .bind(minimum_sample)
    .bind(maximum_sample)
    .fetch_one(&context.pool)
    .await
    .map_err(database_error)?;
    if verified != i64::try_from(records.len()).unwrap_or(i64::MAX) {
        return Err(integrity(
            "execution_snapshot_verification_failed",
            format!(
                "expected {} durable snapshots, found {verified}",
                records.len()
            ),
        ));
    }
    Ok(verified)
}

fn invalid_source(code: &'static str, message: impl Into<String>) -> BackfillExecutionError {
    BackfillExecutionError::new(BackfillFailureKind::TransientSource, code, message)
}

fn database_error(error: sqlx::Error) -> BackfillExecutionError {
    BackfillExecutionError::new(
        BackfillFailureKind::TransientDatabase,
        "database_error",
        error.to_string(),
    )
}

fn integrity(code: &'static str, message: impl Into<String>) -> BackfillExecutionError {
    BackfillExecutionError::new(BackfillFailureKind::Integrity, code, message)
}

fn cancelled() -> BackfillExecutionError {
    BackfillExecutionError::new(
        BackfillFailureKind::Cancelled,
        "worker_shutdown",
        "worker shutdown interrupted the backfill",
    )
}
fn slug_for_window(window_start: DateTime<Utc>) -> String {
    format!("btc-updown-5m-{}", window_start.timestamp())
}

pub(super) fn parse_gamma_btc_interval_event(
    value: &Value,
    expected_window_start: DateTime<Utc>,
) -> AnyResult<BtcIntervalMarket> {
    let event = value
        .as_object()
        .context("Gamma event response must be an object")?;
    let event_slug = required_string(event, &["slug"])?;
    let slug_epoch = event_slug
        .strip_prefix("btc-updown-5m-")
        .context("event slug is not a BTC Up/Down 5m slug")?
        .parse::<i64>()
        .context("BTC Up/Down 5m slug has an invalid epoch suffix")?;
    if slug_epoch.rem_euclid(300) != 0 {
        bail!("BTC Up/Down 5m slug is not five-minute aligned");
    }
    let slug_window_start = DateTime::from_timestamp(slug_epoch, 0)
        .context("BTC Up/Down 5m slug epoch is out of range")?;
    if slug_window_start != expected_window_start {
        bail!("Gamma event slug does not match the requested window");
    }
    let series_slug = string_field(event, &["seriesSlug", "series_slug"])
        .or_else(|| {
            event
                .get("series")?
                .as_array()?
                .iter()
                .find_map(|series| string_field(series.as_object()?, &["slug"]))
        })
        .context("Gamma event is missing a series slug")?;
    if series_slug != "btc-up-or-down-5m" {
        bail!("Gamma event belongs to unexpected series {series_slug}");
    }
    let window_start = datetime_field(event, &["eventStartTime", "startTime"])
        .context("Gamma event is missing eventStartTime")?;
    if window_start != slug_window_start {
        bail!("Gamma eventStartTime does not match its slug epoch");
    }
    let markets = event
        .get("markets")
        .and_then(Value::as_array)
        .context("Gamma event is missing its markets array")?;
    if markets.len() != 1 {
        bail!("BTC Up/Down 5m event must contain exactly one market");
    }
    let market = markets[0]
        .as_object()
        .context("Gamma event market must be an object")?;
    let window_end = datetime_field(market, &["endDate", "endDateIso"])
        .or_else(|| datetime_field(event, &["endDate"]))
        .context("Gamma event is missing the market end date")?;
    if window_end != window_start + ChronoDuration::minutes(5) {
        bail!("BTC Up/Down market does not have an exact five-minute window");
    }
    let resolution_source = string_field(market, &["resolutionSource", "resolution_source"])
        .or_else(|| string_field(event, &["resolutionSource", "resolution_source"]))
        .context("Gamma event is missing a resolution source")?;
    let normalized_source = resolution_source.trim().to_ascii_lowercase();
    if !(normalized_source.contains("chainlink") || normalized_source.contains("chain.link"))
        || !(normalized_source.contains("btc-usd")
            || normalized_source.contains("btc/usd")
            || (normalized_source.contains("btc") && normalized_source.contains("usd")))
    {
        bail!("BTC Up/Down market resolution source is not Chainlink BTC/USD");
    }
    let outcomes =
        string_array_field(market, &["outcomes"]).context("Gamma market is missing outcomes")?;
    let token_ids = string_array_field(
        market,
        &["clobTokenIds", "clob_token_ids", "tokenIds", "token_ids"],
    )
    .context("Gamma market is missing CLOB token IDs")?;
    if outcomes.len() != 2 || token_ids.len() != 2 {
        bail!("BTC Up/Down market must contain exactly two outcomes and token IDs");
    }
    let mut up_token_id = None;
    let mut down_token_id = None;
    for (outcome, token_id) in outcomes.iter().zip(token_ids) {
        match parse_outcome(outcome)? {
            BtcOutcome::Up => {
                if up_token_id.replace(token_id).is_some() {
                    bail!("BTC Up/Down market contains duplicate Up outcomes");
                }
            }
            BtcOutcome::Down => {
                if down_token_id.replace(token_id).is_some() {
                    bail!("BTC Up/Down market contains duplicate Down outcomes");
                }
            }
        }
    }
    let up_token_id = up_token_id.context("BTC Up/Down market is missing its Up token")?;
    let down_token_id = down_token_id.context("BTC Up/Down market is missing its Down token")?;
    if up_token_id == down_token_id {
        bail!("BTC Up/Down market token IDs must be distinct");
    }
    let tick_size = decimal_field(
        market,
        &[
            "orderPriceMinTickSize",
            "minimumTickSize",
            "tickSize",
            "tick_size",
        ],
    )
    .context("Gamma market is missing its minimum tick size")?;
    if tick_size <= Decimal::ZERO {
        bail!("Gamma market minimum tick size must be positive");
    }
    Ok(BtcIntervalMarket {
        event_id: required_string(event, &["id"])?,
        event_slug,
        series_slug,
        market_id: required_string(market, &["id"])?,
        condition_id: required_string(market, &["conditionId", "condition_id"])?,
        window_start,
        window_end,
        up_token_id,
        down_token_id,
        tick_size,
        minimum_order_size: decimal_field(market, &["orderMinSize", "minimumOrderSize"]),
        resolution_source,
        active: bool_field(market, &["active"])
            .or_else(|| bool_field(event, &["active"]))
            .unwrap_or(false),
        closed: bool_field(market, &["closed"])
            .or_else(|| bool_field(event, &["closed"]))
            .unwrap_or(false),
        accepting_orders: bool_field(market, &["acceptingOrders", "accepting_orders"])
            .unwrap_or(false),
        fees_enabled: bool_field(market, &["feesEnabled", "fees_enabled"])
            .or_else(|| bool_field(event, &["feesEnabled", "fees_enabled"]))
            .unwrap_or(false),
        fee_schedule: market
            .get("feeSchedule")
            .or_else(|| event.get("feeSchedule"))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({})),
        raw_payload: value.clone(),
    })
}

fn parse_clob_rest_official_resolution(
    value: &Value,
    market: &BtcIntervalMarket,
    observed_at: DateTime<Utc>,
) -> AnyResult<Option<ClobOfficialResolution>> {
    let object = value
        .as_object()
        .context("CLOB market response must be an object")?;
    if required_string(object, &["condition_id", "conditionId"])? != market.condition_id {
        bail!("CLOB condition does not match the stored market");
    }
    let closed = bool_field(object, &["closed"]).context("CLOB market is missing closed status")?;
    let tokens = object
        .get("tokens")
        .and_then(Value::as_array)
        .context("CLOB market is missing its tokens array")?;
    if tokens.len() != 2 {
        bail!("CLOB BTC interval market must contain exactly two tokens");
    }
    let mut parsed = Vec::with_capacity(2);
    for token in tokens {
        let token = token.as_object().context("CLOB token must be an object")?;
        let token_id = required_string(token, &["token_id", "tokenId"])?;
        let outcome = parse_outcome(&required_string(token, &["outcome"])?)?;
        if token_id != market.token_id(outcome) {
            bail!("CLOB token does not match its stored outcome token");
        }
        parsed.push((
            token_id,
            outcome,
            bool_field(token, &["winner"]),
            decimal_field(token, &["price"]),
        ));
    }
    if parsed[0].1 == parsed[1].1 {
        bail!("CLOB market contains duplicate outcomes");
    }
    if !closed {
        return Ok(None);
    }
    let winners = parsed
        .iter()
        .filter(|(_, _, winner, _)| *winner == Some(true))
        .collect::<Vec<_>>();
    if winners.len() != 1 {
        bail!("closed CLOB market must expose exactly one winner");
    }
    for (token_id, _, winner, price) in &parsed {
        let winner = winner.context("closed CLOB token is missing winner status")?;
        let price = price.context("closed CLOB token is missing terminal price")?;
        let expected = if winner { Decimal::ONE } else { Decimal::ZERO };
        if price != expected {
            bail!("closed CLOB token {token_id} has a non-terminal price");
        }
    }
    Ok(Some(ClobOfficialResolution {
        winning_token_id: winners[0].0.clone(),
        winning_outcome: winners[0].1,
        observed_at,
    }))
}

fn parse_outcome(value: &str) -> AnyResult<BtcOutcome> {
    match value.trim().to_ascii_lowercase().as_str() {
        "up" => Ok(BtcOutcome::Up),
        "down" => Ok(BtcOutcome::Down),
        other => bail!("unexpected BTC interval outcome {other}"),
    }
}

fn required_string(object: &serde_json::Map<String, Value>, keys: &[&str]) -> AnyResult<String> {
    string_field(object, keys).with_context(|| format!("missing required field {}", keys[0]))
}

fn string_field(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| match object.get(*key)? {
        Value::String(value) if !value.trim().is_empty() => Some(value.trim().to_string()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    })
}

fn string_array_field(
    object: &serde_json::Map<String, Value>,
    keys: &[&str],
) -> Option<Vec<String>> {
    let value = keys.iter().find_map(|key| object.get(*key))?;
    let values = match value {
        Value::Array(values) => values.clone(),
        Value::String(value) => serde_json::from_str(value).ok()?,
        _ => return None,
    };
    values
        .into_iter()
        .map(|value| match value {
            Value::String(value) if !value.trim().is_empty() => Some(value.trim().to_string()),
            Value::Number(value) => Some(value.to_string()),
            _ => None,
        })
        .collect()
}

fn decimal_field(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<Decimal> {
    match keys.iter().find_map(|key| object.get(*key))? {
        Value::String(value) => value.parse().ok(),
        Value::Number(value) => value.to_string().parse().ok(),
        _ => None,
    }
}

fn bool_field(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|key| object.get(*key)?.as_bool())
}

fn datetime_field(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(&string_field(object, keys)?)
        .ok()
        .map(|value| value.with_timezone(&Utc))
}

fn extract_reference_values(value: &Value) -> (Option<Decimal>, Option<Decimal>) {
    let opening = decimal_at_paths(
        value,
        &[
            &["priceToBeat"],
            &["eventMetadata", "priceToBeat"],
            &["markets", "0", "priceToBeat"],
            &["markets", "0", "eventMetadata", "priceToBeat"],
        ],
    );
    let final_price = decimal_at_paths(
        value,
        &[
            &["finalPrice"],
            &["settlementValue"],
            &["eventMetadata", "finalPrice"],
            &["eventMetadata", "settlementValue"],
            &["markets", "0", "finalPrice"],
            &["markets", "0", "settlementValue"],
            &["markets", "0", "eventMetadata", "finalPrice"],
        ],
    );
    (opening, final_price)
}

fn decimal_at_paths(value: &Value, paths: &[&[&str]]) -> Option<Decimal> {
    paths.iter().find_map(|path| {
        let mut current = value;
        for segment in *path {
            current = if let Ok(index) = segment.parse::<usize>() {
                current.as_array()?.get(index)?
            } else {
                current.get(*segment)?
            };
        }
        let decimal = match current {
            Value::String(value) => value.parse::<Decimal>().ok(),
            Value::Number(value) => value.to_string().parse::<Decimal>().ok(),
            _ => None,
        }?;
        (decimal > Decimal::ZERO).then_some(decimal)
    })
}

#[async_trait]
impl BackfillWorkerStrategy for PolymarketBtcBackfill {
    fn descriptor(&self) -> &StrategyDescriptor {
        &self.descriptor
    }

    fn validate_request(
        &self,
        request: &BackfillRequest,
    ) -> Result<ValidatedBackfillRequest, BackfillExecutionError> {
        if request.strategy_key != self.key() {
            return Err(BackfillExecutionError::invalid(
                "strategy_key_mismatch",
                "request strategy key does not match the Polymarket backfill strategy",
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
                "Polymarket BTC backfills accept no parameters",
            ));
        }
        if matches!(self.kind, Kind::OrderbookEvents | Kind::ExecutionSnapshots)
            && (!hour_aligned(request.range.start) || !hour_aligned(request.range.end))
        {
            return Err(BackfillExecutionError::invalid(
                "range_alignment_invalid",
                "PMXT backfill ranges must be aligned to complete UTC hours",
            ));
        }
        if matches!(self.kind, Kind::OrderbookEvents | Kind::ExecutionSnapshots)
            && request.range.start.timestamp() < PMXT_COVERAGE_START_EPOCH
        {
            return Err(BackfillExecutionError::invalid(
                "range_before_source_coverage",
                "PMXT v2 coverage begins at 2026-04-13T19:00:00Z",
            ));
        }
        if self.kind == Kind::ExecutionSnapshots
            && request.range.start.timestamp() < PMXT_COVERAGE_START_EPOCH + 3_600
        {
            return Err(BackfillExecutionError::invalid(
                "range_before_seed_coverage",
                "execution-snapshot reconstruction requires the preceding PMXT seed hour",
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
        let mut shards = Vec::new();
        let mut start = request.range_start;
        while start < request.range_end {
            let end = (start + ChronoDuration::hours(1)).min(request.range_end);
            shards.push(BackfillShard {
                shard_key: format!("{}-{}", start.timestamp(), end.timestamp()),
                range_start: start,
                range_end: end,
                parameters: json!({}),
            });
            if shards.len() > self.descriptor.maximum_shards {
                return Err(BackfillExecutionError::invalid(
                    "too_many_shards",
                    format!(
                        "request exceeds {} hourly shards",
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
        match self.kind {
            Kind::MarketContracts => self.execute_market_contracts(&context, &shard).await,
            Kind::Resolutions => self.execute_resolutions(&context, &shard).await,
            Kind::OrderbookEvents => self.execute_orderbook_events(&context, &shard).await,
            Kind::ExecutionSnapshots => self.execute_execution_snapshots(&context, &shard).await,
        }
    }
}

fn env_url(name: &str, default: &'static str) -> Arc<str> {
    Arc::from(
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| default.to_owned()),
    )
}

fn hour_aligned(value: DateTime<Utc>) -> bool {
    value.minute() == 0 && value.second() == 0 && value.timestamp_subsec_nanos() == 0
}

impl PolymarketBtcBackfill {
    async fn execute_market_contracts(
        &self,
        context: &BackfillContext,
        shard: &BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        let logical_key = format!("gamma:btc5m:{}", shard.shard_key);
        if let Some(outcome) = completed_outcome(context, self.key(), &logical_key).await? {
            return Ok(outcome);
        }
        let artifact_id = create_artifact(
            context,
            self.key(),
            &logical_key,
            "polymarket_gamma",
            &format!("{}/events/slug/btc-updown-5m-*", self.gamma_url),
            "polymarket.btc_interval_markets",
        )
        .await?;
        let mut cursor = shard.range_start;
        let mut records = 0i64;
        let mut missing = 0i64;
        let mut digest = Sha256::new();
        while cursor < shard.range_end {
            ensure_running(context)?;
            let slug = slug_for_window(cursor);
            let uri = format!(
                "{}/events/slug/{slug}",
                self.gamma_url.trim_end_matches('/')
            );
            let Some((body, value)) = self.fetch_json(context, &uri).await? else {
                missing += 1;
                cursor += ChronoDuration::minutes(5);
                continue;
            };
            let market = parse_gamma_btc_interval_event(&value, cursor).map_err(|error| {
                integrity("polymarket_gamma_contract_invalid", error.to_string())
            })?;
            persist_market(context, &market).await?;
            let payload_sha256 = format!("{:x}", Sha256::digest(&body));
            let (opening_boundary, final_price) = extract_reference_values(&value);
            if let Some(reference_value) = opening_boundary {
                persist_reference_fact(
                    context,
                    ReferenceFact {
                        artifact_id,
                        market_id: &market.market_id,
                        fact_type: "opening_boundary",
                        value: reference_value,
                        source_effective_at: market.window_start,
                        payload_sha256: &payload_sha256,
                        evidence: &value,
                    },
                )
                .await?;
            }
            if let Some(reference_value) = final_price {
                persist_reference_fact(
                    context,
                    ReferenceFact {
                        artifact_id,
                        market_id: &market.market_id,
                        fact_type: "final_price",
                        value: reference_value,
                        source_effective_at: market.window_end,
                        payload_sha256: &payload_sha256,
                        evidence: &value,
                    },
                )
                .await?;
            }
            digest.update(&body);
            records += 1;
            cursor += ChronoDuration::minutes(5);
        }
        let checksum = format!("{:x}", digest.finalize());
        complete_artifact(
            context,
            artifact_id,
            records,
            &checksum,
            shard,
            json!({"missing_gamma_markets": missing}),
        )
        .await?;
        Ok(outcome(
            records,
            shard,
            json!({"provider":"polymarket_gamma","records_verified":records,"missing_gamma_markets":missing}),
        ))
    }

    async fn execute_resolutions(
        &self,
        context: &BackfillContext,
        shard: &BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        let logical_key = format!("clob:btc5m-resolutions:{}", shard.shard_key);
        if let Some(outcome) = completed_outcome(context, self.key(), &logical_key).await? {
            return Ok(outcome);
        }
        let artifact_id = create_artifact(
            context,
            self.key(),
            &logical_key,
            "polymarket_clob",
            &format!("{}/markets/*", self.clob_url),
            "polymarket.btc_interval_markets",
        )
        .await?;
        let markets = load_markets(context, shard.range_start, shard.range_end).await?;
        let mut records = 0i64;
        let mut unresolved = 0i64;
        let mut digest = Sha256::new();
        for market in &markets {
            ensure_running(context)?;
            if market.official_outcome.is_some() {
                records += 1;
                continue;
            }
            let uri = format!(
                "{}/markets/{}",
                self.clob_url.trim_end_matches('/'),
                market.market.condition_id
            );
            let Some((body, value)) = self.fetch_json(context, &uri).await? else {
                unresolved += 1;
                continue;
            };
            let resolution =
                parse_clob_rest_official_resolution(&value, &market.market, Utc::now()).map_err(
                    |error| integrity("polymarket_clob_resolution_invalid", error.to_string()),
                )?;
            if let Some(resolution) = resolution {
                persist_resolution(context, &market.market, &resolution, &value).await?;
                records += 1;
                digest.update(&body);
            } else {
                unresolved += 1;
            }
        }
        let checksum = format!("{:x}", digest.finalize());
        complete_artifact(
            context,
            artifact_id,
            records,
            &checksum,
            shard,
            json!({"unresolved_markets":unresolved,"candidate_markets":markets.len()}),
        )
        .await?;
        Ok(outcome(
            records,
            shard,
            json!({"provider":"polymarket_clob","records_verified":records,"unresolved_markets":unresolved}),
        ))
    }

    async fn execute_orderbook_events(
        &self,
        context: &BackfillContext,
        shard: &BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        let hour = shard.range_start;
        let spec = PmxtArchiveSpec::new(&self.pmxt_url, hour)
            .map_err(|error| invalid_source("pmxt_archive_spec", error.to_string()))?;
        if let Some(outcome) = completed_outcome(context, self.key(), &spec.logical_key).await? {
            return Ok(outcome);
        }
        if let Some(outcome) = completed_source_outcome(
            context,
            PMXT_ARCHIVE_PROVIDER,
            &spec.logical_key,
            "polymarket.btc_orderbook_archive_events",
        )
        .await?
        {
            return Ok(outcome);
        }
        let scope = load_scope(context, hour, shard.range_end, true).await?;
        if scope.is_empty() {
            return Err(integrity(
                "pmxt_market_scope_empty",
                "no valid BTC five-minute market identities exist; run the market-contract backfill first",
            ));
        }
        let artifact_id = create_artifact(
            context,
            self.key(),
            &spec.logical_key,
            PMXT_ARCHIVE_PROVIDER,
            &spec.source_uri,
            "polymarket.btc_orderbook_archive_events",
        )
        .await?;
        let cancellation = cancellation_for(context);
        let archive = download_archive(
            &self.client,
            &spec,
            &self.cache_directory,
            &ArchiveDownloadLimits {
                maximum_compressed_bytes: 2 * 1024 * 1024 * 1024,
                chunk_idle_timeout: Duration::from_secs(60),
            },
            &cancellation,
        )
        .await
        .map_err(|error| invalid_source("pmxt_archive_download", error.to_string()))?
        .ok_or_else(|| invalid_source("pmxt_archive_missing", "PMXT archive object was absent"))?;
        let (mut receiver, parser) = spawn_parser(
            archive.path.clone(),
            scope,
            DATABASE_BATCH_ROWS,
            cancellation,
        );
        let mut records = 0i64;
        while let Some(batch) = receiver.recv().await {
            let batch =
                batch.map_err(|error| integrity("pmxt_archive_parse", error.to_string()))?;
            records += persist_orderbook_events(context, artifact_id, &batch).await?;
        }
        let parsed = parser
            .await
            .map_err(|error| integrity("pmxt_parser_join", error.to_string()))?
            .map_err(|error| integrity("pmxt_archive_parse", error.to_string()))?;
        complete_artifact(
            context,
            artifact_id,
            records,
            &archive.sha256,
            shard,
            json!({"source_records":parsed.records,"compressed_bytes":archive.compressed_bytes}),
        )
        .await?;
        Ok(outcome(
            records,
            shard,
            json!({"provider":PMXT_ARCHIVE_PROVIDER,"records_verified":records}),
        ))
    }

    async fn execute_execution_snapshots(
        &self,
        context: &BackfillContext,
        shard: &BackfillShard,
    ) -> Result<BackfillOutcome, BackfillExecutionError> {
        let logical_key = format!(
            "pmxt:v2:btc5m_capacity_execution_snapshots:v2:1-240s:{}",
            shard.range_start.format("%Y-%m-%dT%H")
        );
        if let Some(outcome) = completed_outcome(context, self.key(), &logical_key).await? {
            return Ok(outcome);
        }
        let scope = load_scope(context, shard.range_start, shard.range_end, false).await?;
        if scope.len() != 12 {
            return Err(integrity(
                "execution_market_scope_invalid",
                format!(
                    "expected 12 valid BTC five-minute markets, found {}",
                    scope.len()
                ),
            ));
        }
        let artifact_id = create_artifact(
            context,
            self.key(),
            &logical_key,
            "pmxt_v2_capacity_execution_snapshots_v2",
            &format!("{}#btc5m-capacity-1-240s", self.pmxt_url),
            "polymarket.btc_market_capacity_execution_snapshots",
        )
        .await?;
        let mut reconstructor = ExecutionSnapshotReconstructor::new(scope.clone())
            .map_err(|error| integrity("execution_reconstructor", error.to_string()))?;
        let cancellation = cancellation_for(context);
        let mut source_records = 0u64;
        let mut checksum = Sha256::new();
        for source_hour in [
            shard.range_start - ChronoDuration::hours(1),
            shard.range_start,
        ] {
            let spec = PmxtArchiveSpec::new(&self.pmxt_url, source_hour)
                .map_err(|error| invalid_source("pmxt_archive_spec", error.to_string()))?;
            let archive = download_archive(
                &self.client,
                &spec,
                &self.cache_directory,
                &ArchiveDownloadLimits {
                    maximum_compressed_bytes: 2 * 1024 * 1024 * 1024,
                    chunk_idle_timeout: Duration::from_secs(60),
                },
                &cancellation,
            )
            .await
            .map_err(|error| invalid_source("pmxt_archive_download", error.to_string()))?
            .ok_or_else(|| {
                invalid_source(
                    "pmxt_archive_missing",
                    "PMXT seed or current archive was absent",
                )
            })?;
            checksum.update(archive.sha256.as_bytes());
            let (mut receiver, parser) = spawn_execution_parser(
                archive.path,
                scope.clone(),
                DATABASE_BATCH_ROWS,
                cancellation.clone(),
            );
            while let Some(batch) = receiver.recv().await {
                let batch =
                    batch.map_err(|error| integrity("pmxt_execution_parse", error.to_string()))?;
                for event in &batch {
                    reconstructor
                        .apply(event, &mut Vec::new())
                        .map_err(|error| {
                            integrity("execution_reconstruction", error.to_string())
                        })?;
                }
            }
            let parsed = parser
                .await
                .map_err(|error| integrity("pmxt_parser_join", error.to_string()))?
                .map_err(|error| integrity("pmxt_execution_parse", error.to_string()))?;
            source_records = source_records.saturating_add(parsed.records);
        }
        let mut snapshots = Vec::with_capacity(scope.len() * EXECUTION_SNAPSHOTS_PER_MARKET);
        reconstructor.finish_before(shard.range_end, &mut snapshots);
        snapshots
            .retain(|row| row.up_source_timestamp.is_some() || row.down_source_timestamp.is_some());
        let records = persist_execution_snapshots(context, artifact_id, &snapshots).await?;
        let checksum = format!("{:x}", checksum.finalize());
        complete_artifact(
            context,
            artifact_id,
            records,
            &checksum,
            shard,
            json!({"source_events_consumed":source_records,"schema_version":EXECUTION_SNAPSHOT_SCHEMA_VERSION}),
        )
        .await?;
        Ok(outcome(
            records,
            shard,
            json!({"provider":"pmxt_v2_capacity_execution_snapshots_v2","records_verified":records,"source_events_consumed":source_records}),
        ))
    }

    async fn fetch_json(
        &self,
        context: &BackfillContext,
        uri: &str,
    ) -> Result<Option<(Vec<u8>, Value)>, BackfillExecutionError> {
        let response = tokio::select! {
            _ = context.shutdown.cancelled() => return Err(cancelled()),
            response = self.client.get(uri).send() => response,
        }
        .map_err(|error| invalid_source("polymarket_http_request", error.to_string()))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if response.status().as_u16() == 429 {
            return Err(BackfillExecutionError::new(
                BackfillFailureKind::RateLimited,
                "polymarket_rate_limited",
                "Polymarket rate limited the historical request",
            ));
        }
        let response = response
            .error_for_status()
            .map_err(|error| invalid_source("polymarket_http_status", error.to_string()))?;
        if response
            .content_length()
            .is_some_and(|length| length > MAX_JSON_BYTES as u64)
        {
            return Err(invalid_source(
                "polymarket_body_large",
                "response exceeded the body limit",
            ));
        }
        let body = response
            .bytes()
            .await
            .map_err(|error| invalid_source("polymarket_http_body", error.to_string()))?;
        if body.len() > MAX_JSON_BYTES {
            return Err(invalid_source(
                "polymarket_body_large",
                "response exceeded the body limit",
            ));
        }
        let value = serde_json::from_slice(&body)
            .map_err(|error| integrity("polymarket_json_invalid", error.to_string()))?;
        Ok(Some((body.to_vec(), value)))
    }
}
