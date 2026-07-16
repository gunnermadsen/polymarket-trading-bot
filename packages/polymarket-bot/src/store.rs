use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use sqlx::{postgres::PgPoolOptions, FromRow, PgPool};
use uuid::Uuid;

use crate::{
    config::PostgresConfig,
    events::ServiceEvent,
    execution::live::LiveVenueEvent,
    execution::OrderPlanReport,
    models::{
        ConversionRequest, ConversionResult, DataApiClosedPosition, FillRecord,
        GammaMarketMetadata, Market, OrderRecord, OrderRequest, OrderState, OutcomeToken,
        SignalCandidate, TradingProcess, TradingProcessConfig, WalletPerformance, WalletScore,
        WalletScoreRefreshJob, WalletScoreRefreshStatus, WalletSegmentPerformance,
        WalletTradeTaxonomyCandidate, WalletTradeTaxonomyUpdate, WhaleTrade,
    },
    orderbook::LocalOrderBook,
    segments::{
        classify_gamma_taxonomy_segment, normalize_gamma_segment_key, score_wallet_segment,
        SegmentClassification, WalletSegmentPerformanceInput, GAMMA_SEGMENT_CLASSIFIER_VERSION,
        MRS_SEGMENT_V2_SCORE_VERSION,
    },
    taxonomy::{cache_key, taxonomy_update_from_metadata, GAMMA_TAXONOMY_VERSION},
    wallets::{score_mrs, MrsScoreInput, MRS_SCORE_VERSION},
};

#[derive(Clone)]
pub struct Store {
    pool: PgPool,
}

const HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL: &str = r#"
UPDATE polymarket.trading_processes
SET heartbeat_at = now(),
    updated_at = now()
WHERE process_id = $1
  AND enabled = true
  AND status IN ('starting', 'running', 'stopping')
"#;

#[derive(Debug, FromRow)]
struct OrderDbRow {
    order_id: String,
    state: String,
    raw_payload: serde_json::Value,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, FromRow)]
struct TradingProcessRow {
    process_id: Uuid,
    name: String,
    process_type: String,
    process_scope: String,
    process_key: Option<String>,
    status: String,
    enabled: bool,
    config: serde_json::Value,
    metadata: serde_json::Value,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    started_at: Option<DateTime<Utc>>,
    stopped_at: Option<DateTime<Utc>>,
    last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountTrade {
    pub account_trade_id: Uuid,
    pub account_address: String,
    pub token_id: String,
    pub market_id: Option<String>,
    pub side: String,
    pub price: Decimal,
    pub size: Decimal,
    pub notional: Decimal,
    pub timestamp_utc: DateTime<Utc>,
    pub transaction_hash: Option<String>,
    pub venue_order_id: Option<String>,
    pub venue_trade_id: Option<String>,
    pub source: String,
    pub linked_order_id: Option<String>,
    pub raw_payload: serde_json::Value,
}

impl AccountTrade {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        account_address: &str,
        token_id: &str,
        market_id: Option<String>,
        side: &str,
        price: Decimal,
        size: Decimal,
        timestamp_utc: DateTime<Utc>,
        transaction_hash: Option<String>,
        venue_order_id: Option<String>,
        venue_trade_id: Option<String>,
        source: &str,
        raw_payload: serde_json::Value,
    ) -> Self {
        let account_address = account_address.to_ascii_lowercase();
        let side = side.to_ascii_lowercase();
        let identity = format!(
            "{}|{}|{}|{}|{}|{}|{}|{}",
            account_address,
            token_id,
            side,
            price.normalize(),
            size.normalize(),
            timestamp_utc.timestamp_millis(),
            transaction_hash.as_deref().unwrap_or(""),
            venue_trade_id.as_deref().unwrap_or("")
        );
        Self {
            account_trade_id: Uuid::new_v5(&Uuid::NAMESPACE_URL, identity.as_bytes()),
            account_address,
            token_id: token_id.to_string(),
            market_id,
            side,
            price,
            size,
            notional: price * size,
            timestamp_utc,
            transaction_hash,
            venue_order_id,
            venue_trade_id,
            source: source.to_string(),
            linked_order_id: None,
            raw_payload,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountPositionSnapshot {
    pub snapshot_id: Uuid,
    pub account_address: String,
    pub token_id: String,
    pub market_id: Option<String>,
    pub size: Decimal,
    pub avg_price: Option<Decimal>,
    pub current_price: Option<Decimal>,
    pub current_value: Option<Decimal>,
    pub cash_pnl: Option<Decimal>,
    pub percent_pnl: Option<Decimal>,
    pub snapshot_at: DateTime<Utc>,
    pub source: String,
    pub raw_payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingProcessResetReport {
    pub process_id: Uuid,
    pub process_name: String,
    pub orders_deleted: u64,
    pub fills_deleted: u64,
    pub signal_candidates_deleted: u64,
    pub process_events_deleted: u64,
    pub backfill_job_events_deleted: u64,
    pub backfill_jobs_deleted: u64,
    pub process_stopped: bool,
}

#[derive(Debug, Clone)]
pub struct OrderbookSnapshot {
    pub snapshot_id: Uuid,
    pub timestamp_utc: DateTime<Utc>,
    pub market_id: Option<String>,
    pub token_id: String,
    pub best_bid: Option<Decimal>,
    pub best_ask: Option<Decimal>,
    pub tick_size: Option<Decimal>,
    pub stale_level_count: i32,
    pub fresh_depth_bid: Option<Decimal>,
    pub fresh_depth_ask: Option<Decimal>,
    pub book: serde_json::Value,
}

impl OrderbookSnapshot {
    pub fn from_local_book(
        market_id: Option<String>,
        token_id: impl Into<String>,
        tick_size: Option<Decimal>,
        book: &LocalOrderBook,
    ) -> Result<Self> {
        Ok(Self {
            snapshot_id: Uuid::new_v4(),
            timestamp_utc: Utc::now(),
            market_id,
            token_id: token_id.into(),
            best_bid: book.best_bid(),
            best_ask: book.best_ask(),
            tick_size,
            stale_level_count: 0,
            fresh_depth_bid: None,
            fresh_depth_ask: None,
            book: serde_json::json!({
                "best_bid": book.best_bid(),
                "best_ask": book.best_ask(),
                "mid": book.mid(),
            }),
        })
    }
}

#[derive(Debug, Clone)]
pub struct PositionSnapshot {
    pub position_id: Option<Uuid>,
    pub market_id: String,
    pub token_id: String,
    pub underlying_key: String,
    pub status: String,
    pub size: Decimal,
    pub cost_basis: Decimal,
    pub worst_case_loss: Decimal,
    pub opened_at: Option<DateTime<Utc>>,
    pub closed_at: Option<DateTime<Utc>>,
    pub raw_payload: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ConversionRecord {
    pub timestamp_utc: DateTime<Utc>,
    pub conversion_id: Uuid,
    pub market_id: String,
    pub no_token_id: String,
    pub size: Decimal,
    pub status: String,
    pub tx_hash: Option<String>,
    pub latency_ms: i64,
    pub gas_cost_usd: Decimal,
    pub raw_payload: serde_json::Value,
}

impl ConversionRecord {
    pub fn from_request_result(
        request: &ConversionRequest,
        result: &ConversionResult,
    ) -> Result<Self> {
        Ok(Self {
            timestamp_utc: Utc::now(),
            conversion_id: result.conversion_id,
            market_id: request.market_id.clone(),
            no_token_id: request.no_token_id.clone(),
            size: request.size,
            status: result.status.clone(),
            tx_hash: result.tx_hash.clone(),
            latency_ms: result.latency_ms,
            gas_cost_usd: result.gas_cost_usd,
            raw_payload: serde_json::json!({
                "request": request,
                "result": result,
            }),
        })
    }
}

#[derive(Debug, Clone)]
pub struct FunnelEvent {
    pub event_id: Uuid,
    pub timestamp_utc: DateTime<Utc>,
    pub signal_id: Option<Uuid>,
    pub market_id: Option<String>,
    pub stage: String,
    pub status: String,
    pub reason: Option<String>,
    pub metadata: serde_json::Value,
}

impl FunnelEvent {
    pub fn new(
        signal_id: Option<Uuid>,
        market_id: Option<String>,
        stage: impl Into<String>,
        status: impl Into<String>,
        reason: Option<String>,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            event_id: Uuid::new_v4(),
            timestamp_utc: Utc::now(),
            signal_id,
            market_id,
            stage: stage.into(),
            status: status.into(),
            reason,
            metadata,
        }
    }
}

impl Store {
    pub async fn connect(config: &PostgresConfig) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&config.database_url())
            .await
            .context("failed to connect to Postgres")?;
        Ok(Self { pool })
    }

    pub async fn create_trading_process(
        &self,
        name: &str,
        process_type: &str,
        process_scope: &str,
        process_key: Option<&str>,
        enabled: bool,
        config: TradingProcessConfig,
        metadata: serde_json::Value,
    ) -> Result<TradingProcess> {
        let row = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            INSERT INTO polymarket.trading_processes (
              process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at
            )
            VALUES (gen_random_uuid(),$1,$2,$3,$4,'created',$5,$6,$7,now(),now())
            RETURNING process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            "#,
        )
        .bind(name)
        .bind(process_type)
        .bind(process_scope)
        .bind(process_key)
        .bind(enabled)
        .bind(serde_json::to_value(config)?)
        .bind(metadata)
        .fetch_one(&self.pool)
        .await
        .context("failed to create trading process")?;
        trading_process_from_row(row)
    }

    pub async fn create_trading_process_with_status(
        &self,
        name: &str,
        process_type: &str,
        process_scope: &str,
        process_key: Option<&str>,
        status: &str,
        enabled: bool,
        config: TradingProcessConfig,
        metadata: serde_json::Value,
    ) -> Result<TradingProcess> {
        let row = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            INSERT INTO polymarket.trading_processes (
              process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at
            )
            VALUES (
              gen_random_uuid(),$1,$2,$3,$4,$5,$6,$7,$8,now(),now(),
              now(),
              CASE WHEN $5 IN ('stopped', 'failed', 'expired', 'completed') THEN now() ELSE NULL END
            )
            RETURNING process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            "#,
        )
        .bind(name)
        .bind(process_type)
        .bind(process_scope)
        .bind(process_key)
        .bind(status)
        .bind(enabled)
        .bind(serde_json::to_value(config)?)
        .bind(metadata)
        .fetch_one(&self.pool)
        .await
        .context("failed to create trading process with status")?;
        trading_process_from_row(row)
    }

    pub async fn update_trading_process_status(
        &self,
        process_id: Uuid,
        status: &str,
        enabled: bool,
        error: Option<&str>,
    ) -> Result<Option<TradingProcess>> {
        let row = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            UPDATE polymarket.trading_processes
            SET status = $2,
                enabled = $3,
                started_at = CASE
                  WHEN $2 = 'starting' THEN now()
                  WHEN $2 = 'running' AND status NOT IN ('starting', 'running') THEN now()
                  ELSE started_at
                END,
                heartbeat_at = CASE
                  WHEN $2 IN ('starting', 'running') THEN NULL
                  ELSE heartbeat_at
                END,
                stopped_at = CASE
                  WHEN $2 IN ('stopped', 'failed', 'expired', 'completed') OR $3 = false THEN now()
                  WHEN $2 IN ('starting', 'running', 'stopping') THEN NULL
                  ELSE stopped_at
                END,
                stop_reason = CASE
                  WHEN $2 IN ('starting', 'running') THEN NULL
                  WHEN $2 IN ('stopped', 'failed', 'expired', 'completed') THEN $4
                  ELSE stop_reason
                END,
                last_error = $4,
                updated_at = now()
            WHERE process_id = $1
            RETURNING process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            "#,
        )
        .bind(process_id)
        .bind(status)
        .bind(enabled)
        .bind(error)
        .fetch_optional(&self.pool)
        .await
        .context("failed to update trading process status")?;
        row.map(trading_process_from_row).transpose()
    }

    /// Records control-plane liveness only while the process definition still
    /// claims an active lifecycle. A stale manager cannot revive or make a
    /// stopped/disabled process appear healthy.
    pub async fn heartbeat_active_trading_process(&self, process_id: Uuid) -> Result<bool> {
        let result = sqlx::query(HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL)
            .bind(process_id)
            .execute(&self.pool)
            .await
            .context("failed to heartbeat active trading process")?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn record_trading_process_event(
        &self,
        process_id: Uuid,
        level: &str,
        event_type: &str,
        message: Option<&str>,
        metadata: serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.trading_process_events (
              event_id, process_id, timestamp_utc, level, event_type, message, metadata, created_at
            )
            VALUES (gen_random_uuid(), $1, now(), $2, $3, $4, $5, now())
            "#,
        )
        .bind(process_id)
        .bind(level)
        .bind(event_type)
        .bind(message)
        .bind(metadata)
        .execute(&self.pool)
        .await
        .context("failed to record trading process event")?;
        Ok(())
    }

    pub async fn upsert_trading_process_by_key(
        &self,
        name: &str,
        process_type: &str,
        process_scope: &str,
        process_key: &str,
        enabled: bool,
        status: &str,
        config: TradingProcessConfig,
        metadata: serde_json::Value,
    ) -> Result<TradingProcess> {
        let config_value = serde_json::to_value(config)?;
        let row = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            WITH updated AS (
              UPDATE polymarket.trading_processes
              SET name = $1,
                  status = $5,
                  enabled = $6,
                  config = $7,
                  metadata = $8,
                  started_at = CASE
                    WHEN $6 = true AND $5 = 'running' THEN COALESCE(started_at, now())
                    ELSE started_at
                  END,
                  stopped_at = CASE
                    WHEN $5 IN ('stopped', 'failed', 'expired', 'completed')
                      THEN COALESCE(stopped_at, now())
                    WHEN $6 = true THEN NULL
                    ELSE stopped_at
                  END,
                  last_error = CASE
                    WHEN $5 NOT IN ('failed', 'error') THEN NULL
                    ELSE last_error
                  END,
                  updated_at = now()
              WHERE process_type = $2
                AND process_scope = $3
                AND process_key = $4
              RETURNING process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
                created_at, updated_at, started_at, stopped_at, last_error
            ),
            inserted AS (
              INSERT INTO polymarket.trading_processes (
                process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
                created_at, updated_at, started_at, stopped_at, last_error
              )
              SELECT
                gen_random_uuid(), $1, $2, $3, $4, $5, $6, $7, $8,
                now(), now(),
                CASE WHEN $6 = true AND $5 = 'running' THEN now() ELSE NULL END,
                CASE WHEN $5 IN ('stopped', 'failed', 'expired', 'completed') THEN now() ELSE NULL END,
                NULL
              WHERE NOT EXISTS (SELECT 1 FROM updated)
              RETURNING process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
                created_at, updated_at, started_at, stopped_at, last_error
            )
            SELECT * FROM updated
            UNION ALL
            SELECT * FROM inserted
            "#,
        )
        .bind(name)
        .bind(process_type)
        .bind(process_scope)
        .bind(process_key)
        .bind(status)
        .bind(enabled)
        .bind(config_value)
        .bind(metadata)
        .fetch_one(&self.pool)
        .await
        .context("failed to upsert trading process by key")?;
        trading_process_from_row(row)
    }

    pub async fn list_trading_processes(&self, limit: i64) -> Result<Vec<TradingProcess>> {
        let rows = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            SELECT process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            FROM polymarket.trading_processes
            ORDER BY updated_at DESC
            LIMIT $1
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .context("failed to list trading processes")?;
        rows.into_iter().map(trading_process_from_row).collect()
    }

    pub async fn get_trading_process(&self, process_id: Uuid) -> Result<Option<TradingProcess>> {
        let row = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            SELECT process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            FROM polymarket.trading_processes
            WHERE process_id = $1
            "#,
        )
        .bind(process_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to get trading process")?;
        row.map(trading_process_from_row).transpose()
    }

    pub async fn get_trading_process_by_key(
        &self,
        process_type: &str,
        process_scope: &str,
        process_key: &str,
    ) -> Result<Option<TradingProcess>> {
        let row = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            SELECT process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            FROM polymarket.trading_processes
            WHERE process_type = $1
              AND process_scope = $2
              AND process_key = $3
            "#,
        )
        .bind(process_type)
        .bind(process_scope)
        .bind(process_key)
        .fetch_optional(&self.pool)
        .await
        .context("failed to get trading process by key")?;
        row.map(trading_process_from_row).transpose()
    }

    pub async fn trading_process_status(
        &self,
        process_id: Uuid,
    ) -> Result<Option<serde_json::Value>> {
        let status = sqlx::query_scalar::<_, serde_json::Value>(
            r#"
            WITH process AS (
              SELECT process_id, name, process_type, process_scope, process_key, status, enabled,
                started_at, stopped_at, heartbeat_at, last_error, created_at, updated_at
              FROM polymarket.trading_processes
              WHERE process_id = $1
            ),
            signals AS (
              SELECT count(*)::bigint AS total, max(timestamp_utc) AS last_signal_at
              FROM polymarket.signal_candidates
              WHERE process_id = $1
            ),
            orders AS (
              SELECT count(*)::bigint AS total,
                count(*) FILTER (WHERE state = 'accepted')::bigint AS accepted,
                count(*) FILTER (WHERE state = 'rejected')::bigint AS rejected,
                count(*) FILTER (WHERE state = 'simulated')::bigint AS simulated,
                max(created_at) AS last_order_at
              FROM polymarket.orders
              WHERE process_id = $1
            ),
            order_states AS (
              SELECT COALESCE(jsonb_object_agg(state, total), '{}'::jsonb) AS counts
              FROM (
                SELECT state, count(*)::bigint AS total
                FROM polymarket.orders
                WHERE process_id = $1
                GROUP BY state
              ) states
            ),
            fills AS (
              SELECT count(*)::bigint AS total, max(timestamp_utc) AS last_fill_at
              FROM polymarket.fills
              WHERE process_id = $1
            ),
            jobs AS (
              SELECT job_id, status, requested_at, started_at, completed_at, error
              FROM polymarket.backfill_jobs
              WHERE request->>'process_id' = $1::text
              ORDER BY requested_at DESC
              LIMIT 1
            )
            SELECT jsonb_build_object(
              'process', to_jsonb(process),
              'signals', jsonb_build_object(
                'total', signals.total,
                'last_signal_at', signals.last_signal_at
              ),
              'orders', jsonb_build_object(
                'total', orders.total,
                'accepted', orders.accepted,
                'rejected', orders.rejected,
                'simulated', orders.simulated,
                'by_state', order_states.counts,
                'last_order_at', orders.last_order_at
              ),
              'fills', jsonb_build_object(
                'total', fills.total,
                'last_fill_at', fills.last_fill_at
              ),
              'last_job', CASE
                WHEN jobs.job_id IS NULL THEN NULL
                ELSE jsonb_build_object(
                  'job_id', jobs.job_id,
                  'status', jobs.status,
                  'requested_at', jobs.requested_at,
                  'started_at', jobs.started_at,
                  'completed_at', jobs.completed_at,
                  'error', jobs.error
                )
              END
            )
            FROM process
            CROSS JOIN signals
            CROSS JOIN orders
            CROSS JOIN order_states
            CROSS JOIN fills
            LEFT JOIN jobs ON true
            "#,
        )
        .bind(process_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to load trading process status")?;
        Ok(status)
    }

    pub async fn update_trading_process(
        &self,
        process_id: Uuid,
        name: Option<&str>,
        process_type: Option<&str>,
        process_scope: Option<&str>,
        process_key: Option<Option<&str>>,
        enabled: Option<bool>,
        status: Option<&str>,
        config: Option<TradingProcessConfig>,
        metadata: Option<serde_json::Value>,
    ) -> Result<Option<TradingProcess>> {
        let config_value = config.map(serde_json::to_value).transpose()?;
        let row = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            UPDATE polymarket.trading_processes
            SET name = COALESCE($2, name),
                process_type = COALESCE($3, process_type),
                process_scope = COALESCE($4, process_scope),
                process_key = CASE WHEN $5::boolean THEN $6 ELSE process_key END,
                enabled = COALESCE($7, enabled),
                status = COALESCE($8, status),
                config = COALESCE($9, config),
                metadata = COALESCE($10, metadata),
                updated_at = now()
            WHERE process_id = $1
            RETURNING process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            "#,
        )
        .bind(process_id)
        .bind(name)
        .bind(process_type)
        .bind(process_scope)
        .bind(process_key.is_some())
        .bind(process_key.flatten())
        .bind(enabled)
        .bind(status)
        .bind(config_value)
        .bind(metadata)
        .fetch_optional(&self.pool)
        .await
        .context("failed to update trading process")?;
        row.map(trading_process_from_row).transpose()
    }

    pub async fn start_trading_process(&self, process_id: Uuid) -> Result<Option<TradingProcess>> {
        let row = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            UPDATE polymarket.trading_processes
            SET status = 'running',
                enabled = true,
                started_at = now(),
                heartbeat_at = NULL,
                stopped_at = NULL,
                stop_reason = NULL,
                last_error = NULL,
                updated_at = now()
            WHERE process_id = $1
            RETURNING process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            "#,
        )
        .bind(process_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to start trading process")?;
        row.map(trading_process_from_row).transpose()
    }

    pub async fn stop_trading_process(&self, process_id: Uuid) -> Result<Option<TradingProcess>> {
        let row = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            UPDATE polymarket.trading_processes
            SET status = 'stopped',
                enabled = false,
                stopped_at = now(),
                updated_at = now()
            WHERE process_id = $1
            RETURNING process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            "#,
        )
        .bind(process_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to stop trading process")?;
        row.map(trading_process_from_row).transpose()
    }

    pub async fn reset_trading_process_data(
        &self,
        process_id: Uuid,
    ) -> Result<Option<TradingProcessResetReport>> {
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin trading process reset transaction")?;
        let process_name = sqlx::query_scalar::<_, String>(
            r#"
            UPDATE polymarket.trading_processes
            SET status = 'stopped',
                enabled = false,
                stopped_at = now(),
                updated_at = now()
            WHERE process_id = $1
            RETURNING name
            "#,
        )
        .bind(process_id)
        .fetch_optional(&mut *tx)
        .await
        .context("failed to stop trading process for reset")?;
        let Some(process_name) = process_name else {
            tx.rollback()
                .await
                .context("failed to roll back missing process reset")?;
            return Ok(None);
        };

        let backfill_job_events_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.backfill_job_events e
            WHERE EXISTS (
              SELECT 1
              FROM polymarket.backfill_jobs j
              WHERE j.job_id = e.job_id
                AND j.request->>'process_id' = $1::text
            )
            "#,
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to delete reset backfill job events")?
        .rows_affected();
        let fills_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.fills f
            WHERE f.process_id = $1
               OR EXISTS (
                 SELECT 1
                 FROM polymarket.orders o
                 WHERE o.process_id = $1
                   AND o.order_id = f.order_id
               )
            "#,
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to delete reset fills")?
        .rows_affected();
        let orders_deleted = sqlx::query("DELETE FROM polymarket.orders WHERE process_id = $1")
            .bind(process_id)
            .execute(&mut *tx)
            .await
            .context("failed to delete reset orders")?
            .rows_affected();
        let signal_candidates_deleted =
            sqlx::query("DELETE FROM polymarket.signal_candidates WHERE process_id = $1")
                .bind(process_id)
                .execute(&mut *tx)
                .await
                .context("failed to delete reset signal candidates")?
                .rows_affected();
        let process_events_deleted =
            sqlx::query("DELETE FROM polymarket.trading_process_events WHERE process_id = $1")
                .bind(process_id)
                .execute(&mut *tx)
                .await
                .context("failed to delete reset process events")?
                .rows_affected();
        let backfill_jobs_deleted = sqlx::query(
            "DELETE FROM polymarket.backfill_jobs WHERE request->>'process_id' = $1::text",
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to delete reset backfill jobs")?
        .rows_affected();

        tx.commit()
            .await
            .context("failed to commit trading process reset")?;
        Ok(Some(TradingProcessResetReport {
            process_id,
            process_name,
            orders_deleted,
            fills_deleted,
            signal_candidates_deleted,
            process_events_deleted,
            backfill_job_events_deleted,
            backfill_jobs_deleted,
            process_stopped: true,
        }))
    }

    pub async fn insert_market(&self, market: &Market) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.markets (
              event_id, market_id, outcome_group_id, question, category, active, closed, archived,
              neg_risk, neg_risk_augmented, rules, end_date, underlying_key,
              resolution_score, raw_payload, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,now())
            ON CONFLICT (market_id) DO UPDATE SET
              event_id = EXCLUDED.event_id,
              outcome_group_id = EXCLUDED.outcome_group_id,
              question = EXCLUDED.question,
              category = EXCLUDED.category,
              active = EXCLUDED.active,
              closed = EXCLUDED.closed,
              archived = EXCLUDED.archived,
              neg_risk = EXCLUDED.neg_risk,
              neg_risk_augmented = EXCLUDED.neg_risk_augmented,
              rules = EXCLUDED.rules,
              end_date = EXCLUDED.end_date,
              underlying_key = EXCLUDED.underlying_key,
              resolution_score = EXCLUDED.resolution_score,
              raw_payload = EXCLUDED.raw_payload,
              updated_at = now()
            "#,
        )
        .bind(&market.event_id)
        .bind(&market.market_id)
        .bind(&market.outcome_group_id)
        .bind(&market.question)
        .bind(&market.category)
        .bind(market.active)
        .bind(market.closed)
        .bind(market.archived)
        .bind(market.neg_risk)
        .bind(market.neg_risk_augmented)
        .bind(&market.rules)
        .bind(market.end_date)
        .bind(&market.underlying_key)
        .bind(market.resolution_score)
        .bind(&market.raw)
        .execute(&self.pool)
        .await
        .context("failed to upsert polymarket market")?;
        for token in &market.outcome_tokens {
            self.upsert_outcome_token(token).await?;
        }
        Ok(())
    }

    pub async fn insert_signal(&self, signal: &SignalCandidate) -> Result<()> {
        self.insert_signal_with_worst_case_loss(signal, signal.worst_case_loss)
            .await
    }

    pub async fn insert_signal_with_worst_case_loss(
        &self,
        signal: &SignalCandidate,
        worst_case_loss: Option<Decimal>,
    ) -> Result<()> {
        let signal_type = serialized_name(&signal.signal_type)?;
        let status = serialized_name(&signal.status)?;
        sqlx::query(
            r#"
            INSERT INTO polymarket.signal_candidates (
              signal_id, process_id, timestamp_utc, signal_type, market_id, expected_edge,
              threshold, size, status, reject_reason, worst_case_loss, metadata
            )
            VALUES ($1,$2,now(),$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (signal_id, timestamp_utc) DO NOTHING
            "#,
        )
        .bind(signal.signal_id)
        .bind(signal.process_id)
        .bind(signal_type)
        .bind(&signal.market_id)
        .bind(signal.expected_edge)
        .bind(signal.threshold)
        .bind(signal.size)
        .bind(status)
        .bind(&signal.reject_reason)
        .bind(worst_case_loss)
        .bind(&signal.metadata)
        .execute(&self.pool)
        .await
        .context("failed to insert signal candidate")?;
        Ok(())
    }

    pub async fn insert_order(&self, order: &OrderRecord) -> Result<()> {
        let side = serialized_name(&order.request.side)?;
        let order_type = serialized_name(&order.request.order_type)?;
        let state = serialized_name(&order.state)?;
        sqlx::query(
            r#"
            INSERT INTO polymarket.orders (
              order_id, client_order_id, process_id, created_at, updated_at, market_id, token_id,
              side, order_type, price, size, state, raw_payload
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (client_order_id) DO UPDATE SET
              order_id = EXCLUDED.order_id,
              process_id = EXCLUDED.process_id,
              market_id = EXCLUDED.market_id,
              token_id = EXCLUDED.token_id,
              side = EXCLUDED.side,
              order_type = EXCLUDED.order_type,
              price = EXCLUDED.price,
              size = EXCLUDED.size,
              updated_at = EXCLUDED.updated_at,
              state = EXCLUDED.state,
              raw_payload = EXCLUDED.raw_payload
            "#,
        )
        .bind(&order.order_id)
        .bind(order.request.client_order_id)
        .bind(order.request.process_id)
        .bind(order.created_at)
        .bind(order.updated_at)
        .bind(&order.request.market_id)
        .bind(&order.request.token_id)
        .bind(side)
        .bind(order_type)
        .bind(order.request.price)
        .bind(order.request.size)
        .bind(state)
        .bind(serde_json::to_value(order)?)
        .execute(&self.pool)
        .await
        .context("failed to upsert order")?;
        Ok(())
    }

    pub async fn create_pending_order(&self, request: &OrderRequest) -> Result<OrderRecord> {
        let now = Utc::now();
        let order = OrderRecord {
            order_id: format!("live-pending-{}", request.client_order_id),
            request: request.clone(),
            state: OrderState::Submitted,
            created_at: now,
            updated_at: now,
        };
        self.insert_order(&order).await?;
        Ok(order)
    }

    pub async fn find_order_by_client_order_id(
        &self,
        client_order_id: Uuid,
    ) -> Result<Option<OrderRecord>> {
        let row = sqlx::query_as::<_, OrderDbRow>(
            r#"
            SELECT order_id, state, raw_payload, created_at, updated_at
            FROM polymarket.orders
            WHERE client_order_id = $1
            "#,
        )
        .bind(client_order_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to find order by client_order_id")?;
        row.map(order_from_db_row).transpose()
    }

    pub async fn find_order_by_venue_order_id(
        &self,
        venue_order_id: &str,
    ) -> Result<Option<OrderRecord>> {
        let row = sqlx::query_as::<_, OrderDbRow>(
            r#"
            SELECT order_id, state, raw_payload, created_at, updated_at
            FROM polymarket.orders
            WHERE venue_order_id = $1 OR order_id = $1
            ORDER BY updated_at DESC
            LIMIT 1
            "#,
        )
        .bind(venue_order_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to find order by venue_order_id")?;
        row.map(order_from_db_row).transpose()
    }

    pub async fn mark_order_submitted(
        &self,
        client_order_id: Uuid,
        venue_order_id: &str,
        raw_ack: serde_json::Value,
    ) -> Result<OrderRecord> {
        self.mark_order_by_client_id(
            client_order_id,
            Some(venue_order_id),
            OrderState::Acknowledged,
            "accepted",
            raw_ack,
        )
        .await
    }

    pub async fn mark_order_submit_failed(
        &self,
        client_order_id: Uuid,
        reason: &str,
        raw_error: serde_json::Value,
    ) -> Result<OrderRecord> {
        self.mark_order_by_client_id(
            client_order_id,
            None,
            OrderState::Rejected,
            reason,
            raw_error,
        )
        .await
    }

    pub async fn mark_cancel_requested(
        &self,
        order_id: &str,
        raw_request: serde_json::Value,
    ) -> Result<Option<OrderRecord>> {
        self.mark_order_by_order_id(
            order_id,
            OrderState::CancelRequested,
            "cancel_requested",
            raw_request,
        )
        .await
    }

    pub async fn mark_order_cancelled(
        &self,
        order_id: &str,
        raw_event: serde_json::Value,
    ) -> Result<Option<OrderRecord>> {
        self.mark_order_by_order_id(order_id, OrderState::Cancelled, "cancelled", raw_event)
            .await
    }

    pub async fn mark_order_filled(
        &self,
        order_id: &str,
        state: OrderState,
        raw_event: serde_json::Value,
    ) -> Result<Option<OrderRecord>> {
        let status = match state {
            OrderState::Filled => "filled",
            OrderState::PartiallyFilled => "partially_filled",
            _ => "filled",
        };
        self.mark_order_by_order_id(order_id, state, status, raw_event)
            .await
    }

    async fn mark_order_by_client_id(
        &self,
        client_order_id: Uuid,
        venue_order_id: Option<&str>,
        state: OrderState,
        reconciliation_status: &str,
        raw_event: serde_json::Value,
    ) -> Result<OrderRecord> {
        let state_name = serialized_name(&state)?;
        let row = sqlx::query_as::<_, OrderDbRow>(
            r#"
            UPDATE polymarket.orders
            SET order_id = COALESCE($2, order_id),
                venue_order_id = COALESCE($2, venue_order_id),
                venue_status = $3,
                state = $4,
                submitted_at = COALESCE(submitted_at, now()),
                accepted_at = CASE WHEN $4 = 'acknowledged' THEN COALESCE(accepted_at, now()) ELSE accepted_at END,
                last_venue_update_at = now(),
                reconciliation_status = $5,
                raw_payload = jsonb_set(raw_payload, '{venue}', $6::jsonb, true),
                updated_at = now()
            WHERE client_order_id = $1
            RETURNING order_id, state, raw_payload, created_at, updated_at
            "#,
        )
        .bind(client_order_id)
        .bind(venue_order_id)
        .bind(reconciliation_status)
        .bind(state_name)
        .bind(reconciliation_status)
        .bind(raw_event)
        .fetch_one(&self.pool)
        .await
        .context("failed to mark order by client_order_id")?;
        order_from_db_row(row)
    }

    async fn mark_order_by_order_id(
        &self,
        order_id: &str,
        state: OrderState,
        reconciliation_status: &str,
        raw_event: serde_json::Value,
    ) -> Result<Option<OrderRecord>> {
        let state_name = serialized_name(&state)?;
        let row = sqlx::query_as::<_, OrderDbRow>(
            r#"
            UPDATE polymarket.orders
            SET state = $2,
                venue_status = $3,
                last_venue_update_at = now(),
                reconciliation_status = $3,
                raw_payload = jsonb_set(raw_payload, '{venue}', $4::jsonb, true),
                updated_at = now()
            WHERE order_id = $1 OR venue_order_id = $1
            RETURNING order_id, state, raw_payload, created_at, updated_at
            "#,
        )
        .bind(order_id)
        .bind(state_name)
        .bind(reconciliation_status)
        .bind(raw_event)
        .fetch_optional(&self.pool)
        .await
        .context("failed to mark order by order_id")?;
        row.map(order_from_db_row).transpose()
    }

    pub async fn insert_fill(&self, fill: &FillRecord) -> Result<()> {
        let source = serialized_name(&fill.source)?;
        sqlx::query(
            r#"
            INSERT INTO polymarket.fills (
              fill_id, process_id, order_id, token_id, timestamp_utc, price, size, fee, source, raw_payload
            )
            SELECT $1,$2,$3,$4,$5,$6,$7,$8,$9,$10
            WHERE NOT EXISTS (
              SELECT 1
              FROM polymarket.fills
              WHERE fill_id = $1
            )
            ON CONFLICT (fill_id, timestamp_utc) DO NOTHING
            "#,
        )
        .bind(fill.fill_id)
        .bind(fill.process_id)
        .bind(&fill.order_id)
        .bind(&fill.token_id)
        .bind(fill.filled_at)
        .bind(fill.price)
        .bind(fill.size)
        .bind(fill.fee)
        .bind(source)
        .bind(serde_json::to_value(fill)?)
        .execute(&self.pool)
        .await
        .context("failed to insert fill")?;
        Ok(())
    }

    pub async fn insert_live_venue_event(&self, event: &LiveVenueEvent) -> Result<bool> {
        let inserted = sqlx::query_scalar::<_, bool>(
            r#"
            INSERT INTO polymarket.live_venue_events (
              event_id, source, event_type, venue_event_id, venue_order_id,
              venue_trade_id, market_id, token_id, event_status, event_timestamp,
              event_hash, raw_payload
            )
            VALUES (
              gen_random_uuid(),$1,$2,$3,$4,$5,
              $8->>'market',
              $8->>'asset_id',
              $6,
              CASE
                WHEN ($8->>'match_time') ~ '^[0-9]+$'
                  THEN to_timestamp(($8->>'match_time')::double precision)
                WHEN ($8->>'timestamp') ~ '^[0-9]+$'
                  THEN to_timestamp(
                    CASE
                      WHEN ($8->>'timestamp')::double precision > 9999999999
                        THEN ($8->>'timestamp')::double precision / 1000
                      ELSE ($8->>'timestamp')::double precision
                    END
                  )
                ELSE NULL
              END,
              $7,$8
            )
            ON CONFLICT (event_hash) DO NOTHING
            RETURNING true
            "#,
        )
        .bind(&event.source)
        .bind(&event.event_type)
        .bind(&event.venue_event_id)
        .bind(&event.venue_order_id)
        .bind(&event.venue_trade_id)
        .bind(&event.event_status)
        .bind(event.hash())
        .bind(&event.raw_payload)
        .fetch_optional(&self.pool)
        .await
        .context("failed to insert live venue event")?
        .unwrap_or(false);
        Ok(inserted)
    }

    pub async fn recent_live_trade_events(&self, limit: i64) -> Result<Vec<LiveVenueEvent>> {
        self.recent_live_events_by_type("trade", limit).await
    }

    pub async fn recent_live_order_events(&self, limit: i64) -> Result<Vec<LiveVenueEvent>> {
        self.recent_live_events_by_type("order", limit).await
    }

    async fn recent_live_events_by_type(
        &self,
        event_type: &str,
        limit: i64,
    ) -> Result<Vec<LiveVenueEvent>> {
        #[derive(sqlx::FromRow)]
        struct Row {
            source: String,
            event_type: String,
            venue_event_id: Option<String>,
            venue_order_id: Option<String>,
            venue_trade_id: Option<String>,
            event_status: Option<String>,
            raw_payload: serde_json::Value,
        }

        let rows = sqlx::query_as::<_, Row>(
            r#"
            SELECT source, event_type, venue_event_id, venue_order_id,
                   venue_trade_id, event_status, raw_payload
            FROM polymarket.live_venue_events
            WHERE source = 'user_ws'
              AND event_type = $1
            ORDER BY created_at DESC
            LIMIT $2
            "#,
        )
        .bind(event_type)
        .bind(limit.clamp(1, 1000))
        .fetch_all(&self.pool)
        .await
        .context("failed to list recent live venue events")?;

        Ok(rows
            .into_iter()
            .map(|row| LiveVenueEvent {
                source: row.source,
                event_type: row.event_type,
                venue_event_id: row.venue_event_id,
                venue_order_id: row.venue_order_id,
                venue_trade_id: row.venue_trade_id,
                event_status: row.event_status,
                raw_payload: row.raw_payload,
            })
            .collect())
    }

    pub async fn insert_live_reconciliation_run(
        &self,
        status: &str,
        open_orders_seen: i32,
        fills_seen: i32,
        balances_seen: i32,
        mismatches_found: i32,
        mismatches_repaired: i32,
        unresolved_count: i32,
        raw_summary: serde_json::Value,
    ) -> Result<Uuid> {
        let run_id = sqlx::query_scalar::<_, Uuid>(
            r#"
            INSERT INTO polymarket.live_reconciliation_runs (
              run_id, started_at, completed_at, status, open_orders_seen,
              fills_seen, balances_seen, mismatches_found, mismatches_repaired,
              unresolved_count, raw_summary
            )
            VALUES (gen_random_uuid(),now(),now(),$1,$2,$3,$4,$5,$6,$7,$8)
            RETURNING run_id
            "#,
        )
        .bind(status)
        .bind(open_orders_seen)
        .bind(fills_seen)
        .bind(balances_seen)
        .bind(mismatches_found)
        .bind(mismatches_repaired)
        .bind(unresolved_count)
        .bind(raw_summary)
        .fetch_one(&self.pool)
        .await
        .context("failed to insert live reconciliation run")?;
        Ok(run_id)
    }

    pub async fn upsert_account_trade(&self, trade: &AccountTrade) -> Result<bool> {
        let inserted = sqlx::query_scalar::<_, bool>(
            r#"
            INSERT INTO polymarket.account_trades (
              account_trade_id, account_address, token_id, market_id, side, price, size,
              notional, timestamp_utc, transaction_hash, venue_order_id, venue_trade_id,
              source, linked_order_id, raw_payload
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
            ON CONFLICT (account_trade_id) DO UPDATE SET
              source = EXCLUDED.source,
              linked_order_id = COALESCE(polymarket.account_trades.linked_order_id, EXCLUDED.linked_order_id),
              raw_payload = polymarket.account_trades.raw_payload || EXCLUDED.raw_payload,
              updated_at = now()
            RETURNING (xmax = 0) AS inserted
            "#,
        )
        .bind(trade.account_trade_id)
        .bind(&trade.account_address)
        .bind(&trade.token_id)
        .bind(&trade.market_id)
        .bind(&trade.side)
        .bind(trade.price)
        .bind(trade.size)
        .bind(trade.notional)
        .bind(trade.timestamp_utc)
        .bind(&trade.transaction_hash)
        .bind(&trade.venue_order_id)
        .bind(&trade.venue_trade_id)
        .bind(&trade.source)
        .bind(&trade.linked_order_id)
        .bind(&trade.raw_payload)
        .fetch_one(&self.pool)
        .await
        .context("failed to upsert account trade")?;
        Ok(inserted)
    }

    pub async fn insert_account_position_snapshot(
        &self,
        snapshot: &AccountPositionSnapshot,
    ) -> Result<bool> {
        let inserted = sqlx::query(
            r#"
            INSERT INTO polymarket.account_position_snapshots (
              snapshot_id, account_address, token_id, market_id, size, avg_price,
              current_price, current_value, cash_pnl, percent_pnl, snapshot_at, source, raw_payload
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (snapshot_id) DO NOTHING
            "#,
        )
        .bind(snapshot.snapshot_id)
        .bind(&snapshot.account_address)
        .bind(&snapshot.token_id)
        .bind(&snapshot.market_id)
        .bind(snapshot.size)
        .bind(snapshot.avg_price)
        .bind(snapshot.current_price)
        .bind(snapshot.current_value)
        .bind(snapshot.cash_pnl)
        .bind(snapshot.percent_pnl)
        .bind(snapshot.snapshot_at)
        .bind(&snapshot.source)
        .bind(&snapshot.raw_payload)
        .execute(&self.pool)
        .await
        .context("failed to insert account position snapshot")?
        .rows_affected()
            > 0;
        Ok(inserted)
    }

    pub async fn insert_account_reconciliation_run(
        &self,
        report: &crate::account_reconcile::AccountReconcileReport,
    ) -> Result<Uuid> {
        let run_id = sqlx::query_scalar::<_, Uuid>(
            r#"
            INSERT INTO polymarket.account_reconciliation_runs (
              run_id, account_address, source, dry_run, token_id, lookback_hours,
              started_at, completed_at, status, activities_fetched, account_trades_inserted,
              position_snapshots_inserted, exits_detected, exits_applied,
              mismatches_found, unmatched_trades, raw_summary
            )
            VALUES (
              gen_random_uuid(),$1,$2,$3,$4,$5,now(),now(),$6,$7,$8,$9,$10,$11,$12,$13,$14
            )
            RETURNING run_id
            "#,
        )
        .bind(&report.account_address)
        .bind(&report.source)
        .bind(report.dry_run)
        .bind(&report.token_id)
        .bind(report.lookback_hours as i32)
        .bind(if report.dry_run {
            "dry_run"
        } else {
            "completed"
        })
        .bind(report.activities_fetched as i32)
        .bind(report.account_trades_inserted as i32)
        .bind(report.position_snapshots_inserted as i32)
        .bind(report.exits_detected as i32)
        .bind(report.exits_applied as i32)
        .bind(report.mismatches.len() as i32)
        .bind(report.unmatched_trades as i32)
        .bind(serde_json::to_value(report)?)
        .fetch_one(&self.pool)
        .await
        .context("failed to insert account reconciliation run")?;
        Ok(run_id)
    }

    pub async fn upsert_outcome_token(&self, token: &OutcomeToken) -> Result<()> {
        let side = serialized_name(&token.side)?;
        sqlx::query(
            r#"
            INSERT INTO polymarket.outcome_tokens (
              token_id, market_id, outcome, side, condition_id, tick_size, neg_risk, raw_payload, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,now())
            ON CONFLICT (token_id) DO UPDATE SET
              market_id = EXCLUDED.market_id,
              outcome = EXCLUDED.outcome,
              side = EXCLUDED.side,
              condition_id = EXCLUDED.condition_id,
              tick_size = EXCLUDED.tick_size,
              neg_risk = EXCLUDED.neg_risk,
              raw_payload = EXCLUDED.raw_payload,
              updated_at = now()
            "#,
        )
        .bind(&token.token_id)
        .bind(&token.market_id)
        .bind(&token.outcome)
        .bind(side)
        .bind(&token.condition_id)
        .bind(token.tick_size)
        .bind(token.neg_risk)
        .bind(serde_json::to_value(token)?)
        .execute(&self.pool)
        .await
        .context("failed to upsert outcome token")?;
        Ok(())
    }

    pub async fn fetch_market_end_date_for_entry(
        &self,
        market_id: &str,
        token_id: &str,
    ) -> Result<Option<DateTime<Utc>>> {
        let row = sqlx::query_scalar::<_, DateTime<Utc>>(
            r#"
            SELECT m.end_date
            FROM polymarket.markets m
            WHERE m.market_id = $1
              AND m.end_date IS NOT NULL
            UNION ALL
            SELECT m.end_date
            FROM polymarket.outcome_tokens ot
            JOIN polymarket.markets m ON m.market_id = ot.market_id
            WHERE ot.token_id = $2
              AND m.end_date IS NOT NULL
            ORDER BY 1 ASC
            LIMIT 1
            "#,
        )
        .bind(market_id)
        .bind(token_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch market end date for entry safety")?;
        Ok(row)
    }

    pub async fn insert_orderbook_snapshot(&self, snapshot: &OrderbookSnapshot) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.orderbook_snapshots (
              snapshot_id, timestamp_utc, market_id, token_id, best_bid, best_ask,
              tick_size, stale_level_count, fresh_depth_bid, fresh_depth_ask, book
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (snapshot_id, timestamp_utc) DO UPDATE SET
              market_id = EXCLUDED.market_id,
              token_id = EXCLUDED.token_id,
              best_bid = EXCLUDED.best_bid,
              best_ask = EXCLUDED.best_ask,
              tick_size = EXCLUDED.tick_size,
              stale_level_count = EXCLUDED.stale_level_count,
              fresh_depth_bid = EXCLUDED.fresh_depth_bid,
              fresh_depth_ask = EXCLUDED.fresh_depth_ask,
              book = EXCLUDED.book
            "#,
        )
        .bind(snapshot.snapshot_id)
        .bind(snapshot.timestamp_utc)
        .bind(&snapshot.market_id)
        .bind(&snapshot.token_id)
        .bind(snapshot.best_bid)
        .bind(snapshot.best_ask)
        .bind(snapshot.tick_size)
        .bind(snapshot.stale_level_count)
        .bind(snapshot.fresh_depth_bid)
        .bind(snapshot.fresh_depth_ask)
        .bind(&snapshot.book)
        .execute(&self.pool)
        .await
        .context("failed to upsert orderbook snapshot")?;
        Ok(())
    }

    pub async fn upsert_position(&self, position: &PositionSnapshot) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.positions (
              position_id, market_id, token_id, underlying_key, status, size,
              cost_basis, worst_case_loss, opened_at, closed_at, raw_payload, updated_at
            )
            VALUES (COALESCE($1, gen_random_uuid()),$2,$3,$4,$5,$6,$7,$8,COALESCE($9, now()),$10,$11,now())
            ON CONFLICT (market_id, token_id) DO UPDATE SET
              underlying_key = EXCLUDED.underlying_key,
              status = EXCLUDED.status,
              size = EXCLUDED.size,
              cost_basis = EXCLUDED.cost_basis,
              worst_case_loss = EXCLUDED.worst_case_loss,
              closed_at = EXCLUDED.closed_at,
              raw_payload = EXCLUDED.raw_payload,
              updated_at = now()
            "#,
        )
        .bind(position.position_id)
        .bind(&position.market_id)
        .bind(&position.token_id)
        .bind(&position.underlying_key)
        .bind(&position.status)
        .bind(position.size)
        .bind(position.cost_basis)
        .bind(position.worst_case_loss)
        .bind(position.opened_at)
        .bind(position.closed_at)
        .bind(&position.raw_payload)
        .execute(&self.pool)
        .await
        .context("failed to upsert position")?;
        Ok(())
    }

    pub async fn insert_conversion_result(
        &self,
        request: &ConversionRequest,
        result: &ConversionResult,
    ) -> Result<()> {
        let record = ConversionRecord::from_request_result(request, result)?;
        self.upsert_conversion(&record).await
    }

    pub async fn upsert_conversion(&self, conversion: &ConversionRecord) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.conversions (
              conversion_id, timestamp_utc, market_id, no_token_id, size, status,
              tx_hash, latency_ms, gas_cost_usd, raw_payload, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,now())
            ON CONFLICT (conversion_id, timestamp_utc) DO UPDATE SET
              status = EXCLUDED.status,
              tx_hash = EXCLUDED.tx_hash,
              latency_ms = EXCLUDED.latency_ms,
              gas_cost_usd = EXCLUDED.gas_cost_usd,
              raw_payload = EXCLUDED.raw_payload,
              updated_at = now()
            "#,
        )
        .bind(conversion.conversion_id)
        .bind(conversion.timestamp_utc)
        .bind(&conversion.market_id)
        .bind(&conversion.no_token_id)
        .bind(conversion.size)
        .bind(&conversion.status)
        .bind(&conversion.tx_hash)
        .bind(conversion.latency_ms)
        .bind(conversion.gas_cost_usd)
        .bind(&conversion.raw_payload)
        .execute(&self.pool)
        .await
        .context("failed to upsert conversion")?;
        Ok(())
    }

    pub async fn insert_funnel_event(&self, event: &FunnelEvent) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.funnel_events (
              event_id, timestamp_utc, signal_id, market_id, stage, status, reason, metadata
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            ON CONFLICT (event_id, timestamp_utc) DO NOTHING
            "#,
        )
        .bind(event.event_id)
        .bind(event.timestamp_utc)
        .bind(event.signal_id)
        .bind(&event.market_id)
        .bind(&event.stage)
        .bind(&event.status)
        .bind(&event.reason)
        .bind(&event.metadata)
        .execute(&self.pool)
        .await
        .context("failed to insert funnel event")?;
        Ok(())
    }

    pub async fn persist_order_plan_report(&self, report: &OrderPlanReport) -> Result<()> {
        for order in &report.orders {
            self.insert_order(order).await?;
        }
        for fill in &report.fills {
            self.insert_fill(fill).await?;
        }
        Ok(())
    }

    pub async fn insert_service_event(&self, event: &ServiceEvent) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.risk_events (
              event_id, timestamp_utc, event_type, severity, message, metadata
            )
            VALUES (gen_random_uuid(),$1,$2,'info',$3,$4)
            "#,
        )
        .bind(event.emitted_at)
        .bind(&event.event_type)
        .bind(event.event_type.clone())
        .bind(&event.payload)
        .execute(&self.pool)
        .await
        .context("failed to insert service event")?;
        Ok(())
    }

    pub async fn record_daily_metric(&self, metric: &str, value: serde_json::Value) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.daily_metrics (metric_date, metric_name, value, updated_at)
            VALUES (CURRENT_DATE,$1,$2,now())
            ON CONFLICT (metric_date, metric_name) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
            "#,
        )
        .bind(metric)
        .bind(value)
        .execute(&self.pool)
        .await
        .context("failed to upsert daily metric")?;
        Ok(())
    }

    pub async fn healthcheck(&self) -> Result<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .context("Postgres healthcheck failed")?;
        Ok(())
    }

    pub async fn ensure_wallet_address(
        &self,
        proxy_wallet: &str,
        seen_at: DateTime<Utc>,
        metadata: serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.wallets (
              proxy_wallet, first_seen_at, last_seen_at, raw_payload, updated_at
            )
            VALUES (lower($1),$2,$2,$3,now())
            ON CONFLICT (proxy_wallet) DO UPDATE SET
              first_seen_at = LEAST(COALESCE(polymarket.wallets.first_seen_at, EXCLUDED.first_seen_at), EXCLUDED.first_seen_at),
              last_seen_at = GREATEST(COALESCE(polymarket.wallets.last_seen_at, EXCLUDED.last_seen_at), EXCLUDED.last_seen_at),
              raw_payload = CASE
                WHEN EXCLUDED.raw_payload = '{}'::jsonb THEN polymarket.wallets.raw_payload
                ELSE EXCLUDED.raw_payload
              END,
              updated_at = now()
            "#,
        )
        .bind(proxy_wallet)
        .bind(seen_at)
        .bind(metadata)
        .execute(&self.pool)
        .await
        .context("failed to ensure wallet address")?;
        Ok(())
    }

    pub async fn record_wallet_observed_trade(&self, trade: &WhaleTrade) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE polymarket.wallets
            SET
              first_seen_at = LEAST(COALESCE(first_seen_at, $2), $2),
              last_seen_at = GREATEST(COALESCE(last_seen_at, $2), $2),
              total_observed_volume = total_observed_volume + $3,
              total_observed_trades = polymarket.wallets.total_observed_trades + 1,
              raw_payload = $4,
              updated_at = now()
            WHERE proxy_wallet = $1
            "#,
        )
        .bind(&trade.proxy_wallet)
        .bind(trade.timestamp_utc)
        .bind(trade.cash_value)
        .bind(&trade.raw_payload)
        .execute(&self.pool)
        .await
        .context("failed to record wallet observed trade")?;
        Ok(())
    }

    pub async fn upsert_gamma_market_metadata(&self, metadata: &GammaMarketMetadata) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.gamma_market_metadata (
              cache_key, lookup_type, lookup_slug, event_slug, market_slug,
              gamma_event_id, gamma_market_id, category, series_slug, tag_slugs,
              sport_key, taxonomy_segment, taxonomy_source, taxonomy_confidence,
              taxonomy_version, raw_payload, fetched_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,now())
            ON CONFLICT (cache_key) DO UPDATE SET
              lookup_type = EXCLUDED.lookup_type,
              lookup_slug = EXCLUDED.lookup_slug,
              event_slug = EXCLUDED.event_slug,
              market_slug = EXCLUDED.market_slug,
              gamma_event_id = EXCLUDED.gamma_event_id,
              gamma_market_id = EXCLUDED.gamma_market_id,
              category = EXCLUDED.category,
              series_slug = EXCLUDED.series_slug,
              tag_slugs = EXCLUDED.tag_slugs,
              sport_key = EXCLUDED.sport_key,
              taxonomy_segment = EXCLUDED.taxonomy_segment,
              taxonomy_source = EXCLUDED.taxonomy_source,
              taxonomy_confidence = EXCLUDED.taxonomy_confidence,
              taxonomy_version = EXCLUDED.taxonomy_version,
              raw_payload = EXCLUDED.raw_payload,
              fetched_at = EXCLUDED.fetched_at,
              updated_at = now()
            "#,
        )
        .bind(&metadata.cache_key)
        .bind(&metadata.lookup_type)
        .bind(&metadata.lookup_slug)
        .bind(&metadata.event_slug)
        .bind(&metadata.market_slug)
        .bind(&metadata.gamma_event_id)
        .bind(&metadata.gamma_market_id)
        .bind(&metadata.category)
        .bind(&metadata.series_slug)
        .bind(&metadata.tag_slugs)
        .bind(&metadata.sport_key)
        .bind(&metadata.taxonomy_segment)
        .bind(&metadata.taxonomy_source)
        .bind(metadata.taxonomy_confidence)
        .bind(&metadata.taxonomy_version)
        .bind(&metadata.raw_payload)
        .bind(metadata.fetched_at)
        .execute(&self.pool)
        .await
        .context("failed to upsert Gamma market metadata")?;
        Ok(())
    }

    pub async fn fetch_gamma_market_metadata_by_lookup(
        &self,
        lookup_type: &str,
        lookup_slug: &str,
    ) -> Result<Option<GammaMarketMetadata>> {
        let row = sqlx::query_as::<_, GammaMarketMetadataRow>(
            r#"
            SELECT cache_key, lookup_type, lookup_slug, event_slug, market_slug,
              gamma_event_id, gamma_market_id, category, series_slug, tag_slugs,
              sport_key, taxonomy_segment, taxonomy_source, taxonomy_confidence,
              taxonomy_version, raw_payload, fetched_at
            FROM polymarket.gamma_market_metadata
            WHERE cache_key = $1
            LIMIT 1
            "#,
        )
        .bind(cache_key(lookup_type, lookup_slug))
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch Gamma market metadata")?;
        Ok(row.map(Into::into))
    }

    pub async fn fetch_wallet_trade_gamma_segment_classification(
        &self,
        trade_id: Uuid,
    ) -> Result<Option<SegmentClassification>> {
        let row = sqlx::query_as::<_, WalletTradeGammaSegmentRow>(
            r#"
            SELECT lower(proxy_wallet) AS proxy_wallet, taxonomy_segment, taxonomy_source,
              COALESCE(taxonomy_confidence, 0)::numeric AS taxonomy_confidence,
              cash_value, timestamp_utc, trade_id,
              COALESCE(taxonomy_metadata, '{}'::jsonb) AS taxonomy_metadata
            FROM polymarket.wallet_trades
            WHERE trade_id = $1
              AND taxonomy_version = $2
              AND taxonomy_source = 'gamma'
              AND taxonomy_segment IS NOT NULL
            LIMIT 1
            "#,
        )
        .bind(trade_id)
        .bind(GAMMA_TAXONOMY_VERSION)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch wallet trade Gamma segment classification")?;

        Ok(row.and_then(|row| {
            classify_gamma_taxonomy_segment(
                &row.taxonomy_segment,
                row.taxonomy_confidence,
                serde_json::json!({
                    "trade_id": row.trade_id,
                    "taxonomy_segment": row.taxonomy_segment,
                    "taxonomy_source": row.taxonomy_source,
                    "taxonomy_metadata": row.taxonomy_metadata
                }),
            )
        }))
    }

    async fn fetch_gamma_segment_lookup_map(&self) -> Result<HashMap<String, String>> {
        let rows = sqlx::query_as::<_, GammaSegmentLookupRow>(
            r#"
            SELECT lookup_slug, event_slug, market_slug, taxonomy_segment, COALESCE(taxonomy_confidence, 0)::numeric AS taxonomy_confidence
            FROM polymarket.gamma_market_metadata
            WHERE taxonomy_version = $1
              AND taxonomy_source = 'gamma'
              AND taxonomy_segment IS NOT NULL
            "#,
        )
        .bind(GAMMA_TAXONOMY_VERSION)
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch Gamma segment lookup map")?;
        let mut map = HashMap::new();
        for row in rows {
            let Some(classification) = classify_gamma_taxonomy_segment(
                &row.taxonomy_segment,
                row.taxonomy_confidence,
                serde_json::json!({
                    "lookup_slug": row.lookup_slug,
                    "event_slug": row.event_slug,
                    "market_slug": row.market_slug
                }),
            ) else {
                continue;
            };
            for key in [Some(row.lookup_slug), row.event_slug, row.market_slug]
                .into_iter()
                .flatten()
                .filter(|value| !value.is_empty())
            {
                map.insert(key.to_ascii_lowercase(), classification.segment_key.clone());
            }
        }
        Ok(map)
    }

    pub async fn fetch_unresolved_wallet_trade_taxonomy_candidates(
        &self,
        limit: i64,
    ) -> Result<Vec<WalletTradeTaxonomyCandidate>> {
        let rows = sqlx::query_as::<_, WalletTradeTaxonomyCandidateRow>(
            r#"
            SELECT trade_id, title, slug, event_slug, market_id, condition_id, asset, raw_payload
            FROM polymarket.wallet_trades
            WHERE taxonomy_version IS NULL
              AND (event_slug IS NOT NULL OR slug IS NOT NULL)
            ORDER BY timestamp_utc DESC
            LIMIT $1
            "#,
        )
        .bind(limit.clamp(1, 10_000))
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch unresolved wallet trade taxonomy candidates")?;
        Ok(rows.into_iter().map(Into::into).collect())
    }

    pub async fn update_wallet_trade_taxonomy(
        &self,
        update: &WalletTradeTaxonomyUpdate,
    ) -> Result<u64> {
        let result = sqlx::query(
            r#"
            UPDATE polymarket.wallet_trades
            SET
              taxonomy_segment = $2,
              taxonomy_source = $3,
              taxonomy_confidence = $4,
              taxonomy_version = $5,
              taxonomy_fetched_at = $6,
              taxonomy_metadata = $7
            WHERE trade_id = $1
            "#,
        )
        .bind(update.trade_id)
        .bind(&update.taxonomy_segment)
        .bind(&update.taxonomy_source)
        .bind(update.taxonomy_confidence)
        .bind(&update.taxonomy_version)
        .bind(update.taxonomy_fetched_at)
        .bind(&update.taxonomy_metadata)
        .execute(&self.pool)
        .await
        .context("failed to update wallet trade taxonomy")?;
        Ok(result.rows_affected())
    }

    pub async fn apply_cached_taxonomy_to_trade(&self, trade: &WhaleTrade) -> Result<bool> {
        let candidate = WalletTradeTaxonomyCandidate {
            trade_id: trade.trade_id,
            title: trade.title.clone(),
            slug: trade.slug.clone(),
            event_slug: trade.event_slug.clone(),
            market_id: trade.market_id.clone(),
            condition_id: trade.condition_id.clone(),
            asset: trade.asset.clone(),
            raw_payload: trade.raw_payload.clone(),
        };
        let mut metadata = if let Some(event_slug) = trade.event_slug.as_deref() {
            self.fetch_gamma_market_metadata_by_lookup("event_slug", event_slug)
                .await?
        } else {
            None
        };
        if metadata.is_none() {
            if let Some(slug) = trade.slug.as_deref() {
                metadata = self
                    .fetch_gamma_market_metadata_by_lookup("market_slug", slug)
                    .await?;
            }
        }
        let Some(update) = metadata
            .as_ref()
            .and_then(|metadata| taxonomy_update_from_metadata(&candidate, metadata))
        else {
            return Ok(false);
        };
        Ok(self.update_wallet_trade_taxonomy(&update).await? > 0)
    }

    pub async fn wallet_trade_taxonomy_status(&self) -> Result<serde_json::Value> {
        let value = sqlx::query_scalar::<_, serde_json::Value>(
            r#"
            WITH totals AS (
              SELECT
                count(*)::integer AS total_trades,
                count(*) FILTER (WHERE taxonomy_version = $1)::integer AS labeled_trades,
                count(*) FILTER (WHERE taxonomy_source = 'gamma')::integer AS gamma_labeled_trades,
                count(*) FILTER (WHERE taxonomy_source = 'keyword_fallback')::integer AS fallback_labeled_trades,
                count(*) FILTER (WHERE taxonomy_segment = 'other')::integer AS other_labeled_trades,
                count(*) FILTER (WHERE taxonomy_version IS NULL)::integer AS unresolved_trades
              FROM polymarket.wallet_trades
            ),
            cache AS (
              SELECT
                count(*)::integer AS cached_metadata,
                count(*) FILTER (WHERE lookup_type = 'event_slug')::integer AS cached_events,
                count(*) FILTER (WHERE lookup_type = 'market_slug')::integer AS cached_markets
              FROM polymarket.gamma_market_metadata
            ),
            top_segments AS (
              SELECT taxonomy_segment, taxonomy_source, count(*)::integer AS trades
              FROM polymarket.wallet_trades
              WHERE taxonomy_version = $1
              GROUP BY taxonomy_segment, taxonomy_source
              ORDER BY trades DESC, taxonomy_segment
              LIMIT 25
            )
            SELECT jsonb_build_object(
              'taxonomy_version', $1,
              'totals', to_jsonb(totals),
              'cache', to_jsonb(cache),
              'top_segments', COALESCE((SELECT jsonb_agg(to_jsonb(top_segments)) FROM top_segments), '[]'::jsonb),
              'updated_at', now()
            )
            FROM totals, cache
            "#,
        )
        .bind(GAMMA_TAXONOMY_VERSION)
        .fetch_one(&self.pool)
        .await
        .context("failed to build wallet trade taxonomy status")?;
        Ok(value)
    }

    pub async fn fetch_wallet_observed_trades(
        &self,
        proxy_wallet: &str,
        since: DateTime<Utc>,
    ) -> Result<Vec<WhaleTrade>> {
        let rows = sqlx::query_as::<_, WhaleTradeRow>(
            r#"
            SELECT trade_id, proxy_wallet, asset, condition_id, market_id, side, outcome,
              price, size, cash_value, timestamp_utc, title, slug, event_slug,
              transaction_hash, raw_payload
            FROM polymarket.wallet_trades
            WHERE lower(proxy_wallet) = lower($1)
              AND timestamp_utc >= $2
            ORDER BY timestamp_utc DESC
            "#,
        )
        .bind(proxy_wallet)
        .bind(since)
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch wallet observed trades")?;
        Ok(rows.into_iter().map(Into::into).collect())
    }

    #[allow(dead_code)]
    async fn fetch_wallet_observed_trades_for_wallets(
        &self,
        proxy_wallets: &[String],
        since: DateTime<Utc>,
    ) -> Result<HashMap<String, Vec<WhaleTrade>>> {
        if proxy_wallets.is_empty() {
            return Ok(HashMap::new());
        }
        let wallet_keys = proxy_wallets
            .iter()
            .map(|wallet| wallet.to_ascii_lowercase())
            .collect::<Vec<_>>();
        let rows = sqlx::query_as::<_, WhaleTradeRow>(
            r#"
            SELECT trade_id, proxy_wallet, asset, condition_id, market_id, side, outcome,
              price, size, cash_value, timestamp_utc, title, slug, event_slug,
              transaction_hash, raw_payload
            FROM polymarket.wallet_trades
            WHERE lower(proxy_wallet) = ANY($1)
              AND timestamp_utc >= $2
            ORDER BY proxy_wallet, timestamp_utc DESC
            "#,
        )
        .bind(&wallet_keys)
        .bind(since)
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch observed trades for wallet segment recompute batch")?;

        let mut by_wallet = HashMap::<String, Vec<WhaleTrade>>::new();
        for row in rows {
            let trade: WhaleTrade = row.into();
            by_wallet
                .entry(trade.proxy_wallet.to_ascii_lowercase())
                .or_default()
                .push(trade);
        }
        Ok(by_wallet)
    }

    async fn fetch_wallet_observed_gamma_segment_inputs(
        &self,
        proxy_wallets: &[String],
        since: DateTime<Utc>,
    ) -> Result<HashMap<String, HashMap<String, WalletSegmentPerformanceInput>>> {
        if proxy_wallets.is_empty() {
            return Ok(HashMap::new());
        }
        let wallet_keys = proxy_wallets
            .iter()
            .map(|wallet| wallet.to_ascii_lowercase())
            .collect::<Vec<_>>();
        let rows = sqlx::query_as::<_, WalletTradeGammaSegmentRow>(
            r#"
            SELECT lower(proxy_wallet) AS proxy_wallet, taxonomy_segment, taxonomy_source,
              COALESCE(taxonomy_confidence, 0)::numeric AS taxonomy_confidence,
              cash_value, timestamp_utc, trade_id,
              COALESCE(taxonomy_metadata, '{}'::jsonb) AS taxonomy_metadata
            FROM polymarket.wallet_trades
            WHERE lower(proxy_wallet) = ANY($1)
              AND timestamp_utc >= $2
              AND taxonomy_version = $3
              AND taxonomy_source = 'gamma'
              AND taxonomy_segment IS NOT NULL
            ORDER BY proxy_wallet, timestamp_utc DESC
            "#,
        )
        .bind(&wallet_keys)
        .bind(since)
        .bind(GAMMA_TAXONOMY_VERSION)
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch Gamma-labeled observed trades for segment v2 recompute")?;

        let mut by_wallet =
            HashMap::<String, HashMap<String, WalletSegmentPerformanceInput>>::new();
        for row in rows {
            let Some(classification) = classify_gamma_taxonomy_segment(
                &row.taxonomy_segment,
                row.taxonomy_confidence,
                serde_json::json!({
                    "trade_id": row.trade_id,
                    "taxonomy_segment": row.taxonomy_segment,
                    "taxonomy_metadata": row.taxonomy_metadata
                }),
            ) else {
                continue;
            };
            let wallet = row.proxy_wallet.to_ascii_lowercase();
            let entry = by_wallet
                .entry(wallet.clone())
                .or_default()
                .entry(classification.segment_key.clone())
                .or_insert_with(|| WalletSegmentPerformanceInput {
                    proxy_wallet: wallet,
                    segment_key: classification.segment_key.clone(),
                    classifier_version: classification.classifier_version.clone(),
                    ..WalletSegmentPerformanceInput::default()
                });
            entry.observed_trade_count = entry.observed_trade_count.saturating_add(1);
            entry.observed_volume_usd += row.cash_value;
            entry.sample_start = Some(
                entry
                    .sample_start
                    .map_or(row.timestamp_utc, |value| value.min(row.timestamp_utc)),
            );
            entry.sample_end = Some(
                entry
                    .sample_end
                    .map_or(row.timestamp_utc, |value| value.max(row.timestamp_utc)),
            );
        }
        Ok(by_wallet)
    }

    pub async fn fetch_wallet_observed_trade_stats(
        &self,
        proxy_wallet: &str,
        since: DateTime<Utc>,
    ) -> Result<WalletObservedTradeStats> {
        let row = sqlx::query_as::<_, WalletObservedTradeStatsRow>(
            r#"
            SELECT
              count(*)::integer AS observed_trade_count,
              COALESCE(sum(cash_value), 0)::numeric AS observed_volume_usd,
              count(DISTINCT COALESCE(condition_id, market_id, slug, asset))::integer AS observed_market_count,
              COALESCE(avg(cash_value), 0)::numeric AS avg_trade_size,
              min(timestamp_utc) AS sample_start,
              max(timestamp_utc) AS sample_end
            FROM polymarket.wallet_trades
            WHERE proxy_wallet = $1
              AND timestamp_utc >= $2
            "#,
        )
        .bind(proxy_wallet)
        .bind(since)
        .fetch_one(&self.pool)
        .await
        .context("failed to fetch wallet observed trade stats")?;
        Ok(row.into())
    }

    pub async fn fetch_wallet_score_by_version(
        &self,
        proxy_wallet: &str,
        score_version: &str,
    ) -> Result<Option<WalletScore>> {
        let row = sqlx::query_as::<_, WalletScoreRow>(
            r#"
            SELECT proxy_wallet, score_version, resolved_markets, total_trades, total_volume,
              realized_pnl, roi, win_rate, avg_trade_size, max_drawdown, score, metadata
            FROM polymarket.wallet_scores
            WHERE proxy_wallet = $1
              AND score_version = $2
            ORDER BY scored_at DESC
            LIMIT 1
            "#,
        )
        .bind(proxy_wallet)
        .bind(score_version)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch wallet score by version")?;
        Ok(row.map(Into::into))
    }

    pub async fn enqueue_wallet_score_refresh(
        &self,
        proxy_wallet: &str,
        score_version: &str,
        segment_score_version: &str,
        refresh_reason: &str,
        last_seen_trade_id: Option<Uuid>,
        metadata: serde_json::Value,
    ) -> Result<WalletScoreRefreshJob> {
        let wallet = proxy_wallet.trim().to_ascii_lowercase();
        if wallet.is_empty() {
            bail!("proxy wallet is required for wallet score refresh");
        }
        self.ensure_wallet_address(&wallet, Utc::now(), serde_json::json!({}))
            .await?;
        let row = sqlx::query_as::<_, WalletScoreRefreshJobRow>(
            r#"
            INSERT INTO polymarket.wallet_score_refresh_jobs (
              proxy_wallet, score_version, segment_score_version, status, refresh_reason,
              last_seen_trade_id, requested_at, available_at, last_error, request_metadata,
              result_metadata, updated_at
            )
            VALUES (lower($1), $2, $3, 'queued', $4, $5, now(), now(), NULL, $6, '{}'::jsonb, now())
            ON CONFLICT (proxy_wallet, score_version, segment_score_version) DO UPDATE SET
              status = 'queued',
              refresh_reason = EXCLUDED.refresh_reason,
              last_seen_trade_id = COALESCE(EXCLUDED.last_seen_trade_id, polymarket.wallet_score_refresh_jobs.last_seen_trade_id),
              requested_at = now(),
              available_at = now(),
              started_at = NULL,
              completed_at = NULL,
              last_error = NULL,
              request_metadata = polymarket.wallet_score_refresh_jobs.request_metadata || EXCLUDED.request_metadata,
              updated_at = now()
            RETURNING queue_id, proxy_wallet, score_version, status, refresh_reason,
              requested_at, available_at, started_at, completed_at, attempt_count,
              max_attempts, last_error, request_metadata, result_metadata
            "#,
        )
        .bind(&wallet)
        .bind(score_version)
        .bind(segment_score_version)
        .bind(refresh_reason)
        .bind(last_seen_trade_id)
        .bind(metadata)
        .fetch_one(&self.pool)
        .await
        .context("failed to enqueue wallet score refresh")?;
        row.try_into()
    }

    pub async fn claim_wallet_score_refresh_jobs(
        &self,
        limit: i64,
    ) -> Result<Vec<WalletScoreRefreshJob>> {
        let rows = sqlx::query_as::<_, WalletScoreRefreshJobRow>(
            r#"
            WITH claimed AS (
              SELECT queue_id
              FROM polymarket.wallet_score_refresh_jobs
              WHERE status IN ('queued', 'failed')
                AND available_at <= now()
                AND attempt_count < max_attempts
              ORDER BY available_at ASC, requested_at ASC
              LIMIT $1
              FOR UPDATE SKIP LOCKED
            )
            UPDATE polymarket.wallet_score_refresh_jobs q
            SET status = 'running',
                started_at = now(),
                completed_at = NULL,
                attempt_count = q.attempt_count + 1,
                last_error = NULL,
                updated_at = now()
            FROM claimed
            WHERE q.queue_id = claimed.queue_id
            RETURNING q.queue_id, q.proxy_wallet, q.score_version, q.status, q.refresh_reason,
              q.requested_at, q.available_at, q.started_at, q.completed_at, q.attempt_count,
              q.max_attempts, q.last_error, q.request_metadata, q.result_metadata
            "#,
        )
        .bind(limit.clamp(1, 500))
        .fetch_all(&self.pool)
        .await
        .context("failed to claim wallet score refresh jobs")?;
        rows.into_iter().map(TryInto::try_into).collect()
    }

    pub async fn complete_wallet_score_refresh_job(
        &self,
        queue_id: Uuid,
        result_metadata: serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE polymarket.wallet_score_refresh_jobs
            SET status = 'completed',
                completed_at = now(),
                last_error = NULL,
                result_metadata = $2,
                updated_at = now()
            WHERE queue_id = $1
            "#,
        )
        .bind(queue_id)
        .bind(result_metadata)
        .execute(&self.pool)
        .await
        .context("failed to complete wallet score refresh job")?;
        Ok(())
    }

    pub async fn fail_wallet_score_refresh_job(&self, queue_id: Uuid, error: &str) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE polymarket.wallet_score_refresh_jobs
            SET status = CASE WHEN attempt_count >= max_attempts THEN 'failed' ELSE 'queued' END,
                available_at = now() + (LEAST(attempt_count, 10) * interval '60 seconds'),
                last_error = $2,
                updated_at = now()
            WHERE queue_id = $1
            "#,
        )
        .bind(queue_id)
        .bind(error)
        .execute(&self.pool)
        .await
        .context("failed to fail wallet score refresh job")?;
        Ok(())
    }

    pub async fn list_wallet_score_refresh_jobs(
        &self,
        status: Option<&str>,
        limit: i64,
    ) -> Result<Vec<WalletScoreRefreshJob>> {
        let rows = sqlx::query_as::<_, WalletScoreRefreshJobRow>(
            r#"
            SELECT queue_id, proxy_wallet, score_version, status, refresh_reason,
              requested_at, available_at, started_at, completed_at, attempt_count,
              max_attempts, last_error, request_metadata, result_metadata
            FROM polymarket.wallet_score_refresh_jobs
            WHERE ($1::text IS NULL OR status = $1)
            ORDER BY updated_at DESC
            LIMIT $2
            "#,
        )
        .bind(status)
        .bind(limit.clamp(1, 500))
        .fetch_all(&self.pool)
        .await
        .context("failed to list wallet score refresh jobs")?;
        rows.into_iter().map(TryInto::try_into).collect()
    }

    pub async fn fetch_top_mrs_scores(&self, limit: i64) -> Result<Vec<WalletScore>> {
        let rows = sqlx::query_as::<_, WalletScoreRow>(
            r#"
            SELECT proxy_wallet, score_version, resolved_markets, total_trades, total_volume,
              realized_pnl, roi, win_rate, avg_trade_size, max_drawdown, score, metadata
            FROM polymarket.wallet_scores
            WHERE score_version = $1
            ORDER BY score DESC, realized_pnl DESC, roi DESC, proxy_wallet
            LIMIT $2
            "#,
        )
        .bind(MRS_SCORE_VERSION)
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch top MRS scores")?;
        Ok(rows.into_iter().map(Into::into).collect())
    }

    pub async fn recompute_mrs_scores_from_existing(
        &self,
        since: DateTime<Utc>,
        limit: i64,
    ) -> Result<u64> {
        let rows = sqlx::query_as::<_, WalletPerformanceRow>(
            r#"
            SELECT proxy_wallet, sample_updated_at, realized_pnl_usd, total_bought_usd,
              roi, closed_positions, winning_positions, win_rate, rank_score,
              raw_payload, metadata
            FROM polymarket.wallet_performance
            ORDER BY rank_score DESC, realized_pnl_usd DESC, roi DESC, proxy_wallet
            LIMIT $1
            "#,
        )
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch wallet performance rows for MRS recompute")?;

        let mut updated = 0u64;
        for row in rows {
            let performance: WalletPerformance = row.into();
            let stats = self
                .fetch_wallet_observed_trade_stats(&performance.proxy_wallet, since)
                .await?;
            let mut input = MrsScoreInput {
                proxy_wallet: performance.proxy_wallet.clone(),
                realized_pnl_usd: performance.realized_pnl_usd,
                total_bought_usd: performance.total_bought_usd,
                roi: performance.roi,
                closed_positions: performance.closed_positions,
                winning_positions: performance.winning_positions,
                win_rate: performance.win_rate,
                observed_trade_count: stats.observed_trade_count,
                observed_volume_usd: stats.observed_volume_usd,
                observed_market_count: stats.observed_market_count,
                avg_trade_size: stats.avg_trade_size,
            };
            if input.observed_volume_usd <= Decimal::ZERO {
                input.observed_volume_usd = performance.total_bought_usd;
            }
            if input.observed_market_count <= 0 {
                input.observed_market_count = performance.closed_positions;
            }
            let mut score = score_mrs(input).into_wallet_score();
            score.metadata = merge_json(
                score.metadata,
                serde_json::json!({
                    "source": "existing_wallet_performance_recompute",
                    "sample_start": stats.sample_start,
                    "sample_end": stats.sample_end
                }),
            );
            self.upsert_wallet_score(&score).await?;
            updated = updated.saturating_add(1);
        }
        Ok(updated)
    }

    pub async fn recompute_wallet_segment_scores_from_existing(
        &self,
        since: DateTime<Utc>,
        limit: i64,
    ) -> Result<u64> {
        self.recompute_wallet_segment_scores_from_gamma_taxonomy(since, limit)
            .await
    }

    pub async fn recompute_wallet_segment_scores_from_gamma_taxonomy(
        &self,
        since: DateTime<Utc>,
        limit: i64,
    ) -> Result<u64> {
        let performances = self
            .compute_wallet_segment_performance_from_gamma_taxonomy(since, limit)
            .await?;
        let mut updated = 0u64;
        for performance in performances {
            self.upsert_wallet_segment_performance(&performance).await?;
            updated = updated.saturating_add(1);
        }
        Ok(updated)
    }

    pub async fn recompute_wallet_segment_v2_scores_from_existing(
        &self,
        since: DateTime<Utc>,
        limit: i64,
    ) -> Result<u64> {
        let gamma_lookup = self.fetch_gamma_segment_lookup_map().await?;
        let mut updated = 0u64;
        let requested = limit.max(1);
        let batch_size = 50i64;
        let mut offset = 0i64;
        while offset < requested {
            let rows = sqlx::query_as::<_, WalletPerformanceRow>(
                r#"
                SELECT proxy_wallet, sample_updated_at, realized_pnl_usd, total_bought_usd,
                  roi, closed_positions, winning_positions, win_rate, rank_score,
                  raw_payload, metadata
                FROM polymarket.wallet_performance
                ORDER BY rank_score DESC, realized_pnl_usd DESC, roi DESC, proxy_wallet
                LIMIT $1 OFFSET $2
                "#,
            )
            .bind(batch_size.min(requested - offset))
            .bind(offset)
            .fetch_all(&self.pool)
            .await
            .context("failed to fetch wallet performance rows for segment v2 recompute")?;
            if rows.is_empty() {
                break;
            }

            let wallet_keys = rows
                .iter()
                .map(|row| row.proxy_wallet.clone())
                .collect::<Vec<_>>();
            let observed_by_wallet = self
                .fetch_wallet_observed_gamma_segment_inputs(&wallet_keys, since)
                .await?;

            for row in rows {
                let wallet = row.proxy_wallet.to_ascii_lowercase();
                let mut by_segment = observed_by_wallet.get(&wallet).cloned().unwrap_or_default();
                for position in closed_positions_from_raw_payload(&row.raw_payload)? {
                    let Some(segment_key) =
                        gamma_segment_for_closed_position(&position, &gamma_lookup)
                    else {
                        continue;
                    };
                    let entry = by_segment.entry(segment_key.clone()).or_insert_with(|| {
                        WalletSegmentPerformanceInput {
                            proxy_wallet: wallet.clone(),
                            segment_key,
                            classifier_version: GAMMA_SEGMENT_CLASSIFIER_VERSION.to_string(),
                            ..WalletSegmentPerformanceInput::default()
                        }
                    });
                    let realized_pnl = position.realized_pnl.unwrap_or(Decimal::ZERO);
                    entry.realized_pnl_usd += realized_pnl;
                    entry.total_bought_usd += position.total_bought.unwrap_or(Decimal::ZERO);
                    entry.closed_positions = entry.closed_positions.saturating_add(1);
                    if realized_pnl > Decimal::ZERO {
                        entry.winning_positions = entry.winning_positions.saturating_add(1);
                    }
                    if let Some(timestamp) = position.timestamp.and_then(timestamp_from_secs) {
                        entry.sample_start = Some(
                            entry
                                .sample_start
                                .map_or(timestamp, |value| value.min(timestamp)),
                        );
                        entry.sample_end = Some(
                            entry
                                .sample_end
                                .map_or(timestamp, |value| value.max(timestamp)),
                        );
                    }
                }

                for input in by_segment.into_values() {
                    let mut performance =
                        score_wallet_segment(input).into_wallet_segment_performance();
                    performance.score_version = MRS_SEGMENT_V2_SCORE_VERSION.to_string();
                    performance.classifier_version = GAMMA_SEGMENT_CLASSIFIER_VERSION.to_string();
                    performance.metadata = merge_json(
                        performance.metadata,
                        serde_json::json!({
                            "source": "gamma_taxonomy_segment_v2_recompute",
                            "score_basis": MRS_SEGMENT_V2_SCORE_VERSION,
                            "score_version": MRS_SEGMENT_V2_SCORE_VERSION,
                            "classifier_version": GAMMA_SEGMENT_CLASSIFIER_VERSION,
                            "sample_updated_at": row.sample_updated_at,
                            "wallet_performance": {
                                "closed_positions": row.closed_positions,
                                "winning_positions": row.winning_positions,
                                "realized_pnl_usd": row.realized_pnl_usd,
                                "roi": row.roi,
                                "win_rate": row.win_rate
                            }
                        }),
                    );
                    self.upsert_wallet_segment_performance(&performance).await?;
                    updated = updated.saturating_add(1);
                }
            }
            offset += batch_size;
        }
        self.refresh_wallet_segment_v2_percentiles().await?;
        Ok(updated)
    }

    pub async fn recompute_wallet_segment_v2_scores_for_wallets(
        &self,
        proxy_wallets: &[String],
        since: DateTime<Utc>,
        refresh_percentiles: bool,
    ) -> Result<u64> {
        let wallet_keys = proxy_wallets
            .iter()
            .map(|wallet| wallet.trim().to_ascii_lowercase())
            .filter(|wallet| !wallet.is_empty())
            .collect::<Vec<_>>();
        if wallet_keys.is_empty() {
            return Ok(0);
        }

        let gamma_lookup = self.fetch_gamma_segment_lookup_map().await?;
        let mut updated = 0u64;
        for wallet_chunk in wallet_keys.chunks(10) {
            let rows = sqlx::query_as::<_, WalletPerformanceRow>(
                r#"
                SELECT proxy_wallet, sample_updated_at, realized_pnl_usd, total_bought_usd,
                  roi, closed_positions, winning_positions, win_rate, rank_score,
                  raw_payload, metadata
                FROM polymarket.wallet_performance
                WHERE lower(proxy_wallet) = ANY($1)
                ORDER BY proxy_wallet
                "#,
            )
            .bind(wallet_chunk)
            .fetch_all(&self.pool)
            .await
            .context("failed to fetch wallet performance rows for targeted segment v2 recompute")?;
            if rows.is_empty() {
                continue;
            }

            let chunk_wallets = rows
                .iter()
                .map(|row| row.proxy_wallet.clone())
                .collect::<Vec<_>>();
            let observed_by_wallet = self
                .fetch_wallet_observed_gamma_segment_inputs(&chunk_wallets, since)
                .await?;

            for row in rows {
                let wallet = row.proxy_wallet.to_ascii_lowercase();
                let mut by_segment = observed_by_wallet.get(&wallet).cloned().unwrap_or_default();
                for position in closed_positions_from_raw_payload(&row.raw_payload)? {
                    let Some(segment_key) =
                        gamma_segment_for_closed_position(&position, &gamma_lookup)
                    else {
                        continue;
                    };
                    let entry = by_segment.entry(segment_key.clone()).or_insert_with(|| {
                        WalletSegmentPerformanceInput {
                            proxy_wallet: wallet.clone(),
                            segment_key,
                            classifier_version: GAMMA_SEGMENT_CLASSIFIER_VERSION.to_string(),
                            ..WalletSegmentPerformanceInput::default()
                        }
                    });
                    let realized_pnl = position.realized_pnl.unwrap_or(Decimal::ZERO);
                    entry.realized_pnl_usd += realized_pnl;
                    entry.total_bought_usd += position.total_bought.unwrap_or(Decimal::ZERO);
                    entry.closed_positions = entry.closed_positions.saturating_add(1);
                    if realized_pnl > Decimal::ZERO {
                        entry.winning_positions = entry.winning_positions.saturating_add(1);
                    }
                    if let Some(timestamp) = position.timestamp.and_then(timestamp_from_secs) {
                        entry.sample_start = Some(
                            entry
                                .sample_start
                                .map_or(timestamp, |value| value.min(timestamp)),
                        );
                        entry.sample_end = Some(
                            entry
                                .sample_end
                                .map_or(timestamp, |value| value.max(timestamp)),
                        );
                    }
                }

                for input in by_segment.into_values() {
                    let mut performance =
                        score_wallet_segment(input).into_wallet_segment_performance();
                    performance.score_version = MRS_SEGMENT_V2_SCORE_VERSION.to_string();
                    performance.classifier_version = GAMMA_SEGMENT_CLASSIFIER_VERSION.to_string();
                    performance.metadata = merge_json(
                        performance.metadata,
                        serde_json::json!({
                            "source": "targeted_gamma_taxonomy_segment_v2_recompute",
                            "score_basis": MRS_SEGMENT_V2_SCORE_VERSION,
                            "score_version": MRS_SEGMENT_V2_SCORE_VERSION,
                            "classifier_version": GAMMA_SEGMENT_CLASSIFIER_VERSION,
                            "sample_updated_at": row.sample_updated_at,
                            "wallet_performance": {
                                "closed_positions": row.closed_positions,
                                "winning_positions": row.winning_positions,
                                "realized_pnl_usd": row.realized_pnl_usd,
                                "roi": row.roi,
                                "win_rate": row.win_rate
                            }
                        }),
                    );
                    self.upsert_wallet_segment_performance(&performance).await?;
                    updated = updated.saturating_add(1);
                }
            }
        }

        if refresh_percentiles && updated > 0 {
            self.refresh_wallet_segment_v2_percentiles().await?;
        }
        Ok(updated)
    }

    pub async fn compute_wallet_segment_performance_from_gamma_taxonomy(
        &self,
        since: DateTime<Utc>,
        limit: i64,
    ) -> Result<Vec<WalletSegmentPerformance>> {
        let rows = sqlx::query_as::<_, GammaWalletSegmentPerformanceInputRow>(
            r#"
            WITH eligible_trades AS (
              SELECT
                lower(proxy_wallet) AS proxy_wallet,
                taxonomy_segment AS segment_key,
                taxonomy_source,
                taxonomy_confidence,
                side,
                cash_value,
                condition_id,
                market_id,
                slug,
                asset,
                timestamp_utc
              FROM polymarket.wallet_trades
              WHERE timestamp_utc >= $1
                AND taxonomy_version = $2
                AND taxonomy_source = 'gamma'
                AND taxonomy_segment IS NOT NULL
                AND taxonomy_segment <> ''
            ),
            ranked_wallets AS (
              SELECT
                proxy_wallet,
                count(*)::integer AS observed_trade_count,
                COALESCE(sum(cash_value), 0)::numeric AS observed_volume_usd,
                max(timestamp_utc) AS latest_trade_at
              FROM eligible_trades
              GROUP BY proxy_wallet
              ORDER BY observed_volume_usd DESC, observed_trade_count DESC, latest_trade_at DESC, proxy_wallet
              LIMIT $3
            )
            SELECT
              trades.proxy_wallet,
              trades.segment_key,
              count(*)::integer AS observed_trade_count,
              COALESCE(sum(trades.cash_value), 0)::numeric AS observed_volume_usd,
              COALESCE(sum(trades.cash_value) FILTER (WHERE upper(trades.side) = 'BUY'), 0)::numeric AS buy_volume_usd,
              COALESCE(sum(trades.cash_value) FILTER (WHERE upper(trades.side) = 'SELL'), 0)::numeric AS sell_volume_usd,
              count(DISTINCT COALESCE(trades.condition_id, trades.market_id, trades.slug, trades.asset))::integer AS observed_market_count,
              COALESCE(avg(trades.taxonomy_confidence), 0)::numeric AS avg_taxonomy_confidence,
              min(trades.timestamp_utc) AS sample_start,
              max(trades.timestamp_utc) AS sample_end,
              jsonb_build_object(
                'source', 'wallet_trades_gamma_taxonomy',
                'taxonomy_version', $2,
                'taxonomy_sources', COALESCE(
                  jsonb_agg(DISTINCT trades.taxonomy_source) FILTER (WHERE trades.taxonomy_source IS NOT NULL),
                  '[]'::jsonb
                )
              ) AS taxonomy_metadata
            FROM eligible_trades trades
            JOIN ranked_wallets wallets ON wallets.proxy_wallet = trades.proxy_wallet
            GROUP BY trades.proxy_wallet, trades.segment_key
            ORDER BY observed_volume_usd DESC, observed_trade_count DESC, trades.proxy_wallet, trades.segment_key
            "#,
        )
        .bind(since)
        .bind(GAMMA_TAXONOMY_VERSION)
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await
        .context("failed to compute wallet segment performance from Gamma taxonomy")?;

        Ok(rows
            .into_iter()
            .map(|row| row.into_wallet_segment_performance())
            .collect())
    }

    pub async fn upsert_wallet_segment_performance(
        &self,
        performance: &WalletSegmentPerformance,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.wallet_segment_performance (
              proxy_wallet, segment_key, score_version, classifier_version, score, confidence,
              closed_positions, winning_positions, losing_positions, win_rate,
              realized_pnl_usd, total_bought_usd, roi, observed_trade_count,
              observed_volume_usd, sample_start, sample_end, metadata, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,now())
            ON CONFLICT (proxy_wallet, segment_key, score_version) DO UPDATE SET
              classifier_version = EXCLUDED.classifier_version,
              score = EXCLUDED.score,
              confidence = EXCLUDED.confidence,
              closed_positions = EXCLUDED.closed_positions,
              winning_positions = EXCLUDED.winning_positions,
              losing_positions = EXCLUDED.losing_positions,
              win_rate = EXCLUDED.win_rate,
              realized_pnl_usd = EXCLUDED.realized_pnl_usd,
              total_bought_usd = EXCLUDED.total_bought_usd,
              roi = EXCLUDED.roi,
              observed_trade_count = EXCLUDED.observed_trade_count,
              observed_volume_usd = EXCLUDED.observed_volume_usd,
              sample_start = EXCLUDED.sample_start,
              sample_end = EXCLUDED.sample_end,
              metadata = EXCLUDED.metadata,
              updated_at = now()
            "#,
        )
        .bind(&performance.proxy_wallet)
        .bind(&performance.segment_key)
        .bind(&performance.score_version)
        .bind(&performance.classifier_version)
        .bind(performance.score)
        .bind(performance.confidence)
        .bind(performance.closed_positions)
        .bind(performance.winning_positions)
        .bind(performance.losing_positions)
        .bind(performance.win_rate)
        .bind(performance.realized_pnl_usd)
        .bind(performance.total_bought_usd)
        .bind(performance.roi)
        .bind(performance.observed_trade_count)
        .bind(performance.observed_volume_usd)
        .bind(performance.sample_start)
        .bind(performance.sample_end)
        .bind(&performance.metadata)
        .execute(&self.pool)
        .await
        .context("failed to upsert wallet segment performance")?;
        Ok(())
    }

    pub async fn fetch_wallet_segment_performance(
        &self,
        proxy_wallet: &str,
        segment_key: &str,
        score_version: &str,
    ) -> Result<Option<WalletSegmentPerformance>> {
        let row = sqlx::query_as::<_, WalletSegmentPerformanceRow>(
            r#"
            SELECT proxy_wallet, segment_key, score_version, classifier_version, score, confidence,
              closed_positions, winning_positions, losing_positions, win_rate,
              realized_pnl_usd, total_bought_usd, roi, observed_trade_count,
              observed_volume_usd, sample_start, sample_end, metadata
            FROM polymarket.wallet_segment_performance
            WHERE lower(proxy_wallet) = lower($1)
              AND segment_key = $2
              AND score_version = $3
            LIMIT 1
            "#,
        )
        .bind(proxy_wallet)
        .bind(segment_key)
        .bind(score_version)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch wallet segment performance")?;
        Ok(row.map(Into::into))
    }

    pub async fn refresh_wallet_segment_v2_percentiles(&self) -> Result<u64> {
        let result = sqlx::query(
            r#"
            WITH ranked AS (
              SELECT
                proxy_wallet,
                segment_key,
                score_version,
                cume_dist() OVER (
                  PARTITION BY segment_key, score_version
                  ORDER BY score ASC, realized_pnl_usd ASC, proxy_wallet ASC
                ) AS segment_percentile,
                row_number() OVER (
                  PARTITION BY segment_key, score_version
                  ORDER BY score DESC, realized_pnl_usd DESC, proxy_wallet ASC
                ) AS segment_rank,
                count(*) OVER (PARTITION BY segment_key, score_version) AS segment_wallet_count
              FROM polymarket.wallet_segment_performance
              WHERE score_version = $1
            )
            UPDATE polymarket.wallet_segment_performance wsp
            SET metadata = COALESCE(wsp.metadata, '{}'::jsonb)
                || jsonb_build_object(
                  'segment_percentile', round(r.segment_percentile::numeric, 6)::text,
                  'segment_rank', r.segment_rank,
                  'segment_wallet_count', r.segment_wallet_count
                ),
                updated_at = now()
            FROM ranked r
            WHERE wsp.proxy_wallet = r.proxy_wallet
              AND wsp.segment_key = r.segment_key
              AND wsp.score_version = r.score_version
            "#,
        )
        .bind(MRS_SEGMENT_V2_SCORE_VERSION)
        .execute(&self.pool)
        .await
        .context("failed to refresh wallet segment v2 percentiles")?;
        Ok(result.rows_affected())
    }

    pub async fn wallet_segment_summary(&self, limit: i64) -> Result<serde_json::Value> {
        self.wallet_segment_summary_by_version(MRS_SEGMENT_V2_SCORE_VERSION, limit)
            .await
    }

    pub async fn wallet_segment_summary_by_version(
        &self,
        score_version: &str,
        limit: i64,
    ) -> Result<serde_json::Value> {
        let value = sqlx::query_scalar::<_, serde_json::Value>(
            r#"
            WITH segment_status AS (
              SELECT
                count(*)::integer AS scored_rows,
                count(DISTINCT proxy_wallet)::integer AS wallets_scored,
                count(DISTINCT segment_key)::integer AS segments_scored,
                max(updated_at) AS last_updated_at
              FROM polymarket.wallet_segment_performance
              WHERE score_version = $1
            ),
            taxonomy_status AS (
              SELECT
                count(*) FILTER (WHERE taxonomy_version = $3)::integer AS labeled_trades,
                count(DISTINCT lower(proxy_wallet)) FILTER (WHERE taxonomy_version = $3)::integer AS wallets_with_labeled_trades,
                count(DISTINCT taxonomy_segment) FILTER (
                  WHERE taxonomy_version = $3
                    AND taxonomy_segment IS NOT NULL
                    AND taxonomy_segment <> ''
                )::integer AS taxonomy_segments
              FROM polymarket.wallet_trades
            ),
            by_segment AS (
              SELECT
                segment_key,
                count(*)::integer AS wallets_scored,
                count(*) FILTER (WHERE closed_positions > 0)::integer AS wallets_with_closed_positions,
                COALESCE(avg(score), 0) AS avg_score,
                COALESCE(avg(win_rate), 0) AS avg_win_rate,
                COALESCE(avg(roi), 0) AS avg_roi,
                COALESCE(sum(observed_trade_count), 0)::integer AS observed_trade_count,
                COALESCE(sum(observed_volume_usd), 0) AS observed_volume_usd
              FROM polymarket.wallet_segment_performance
              WHERE score_version = $1
              GROUP BY segment_key
            ),
            top_wallets AS (
              SELECT
                proxy_wallet,
                segment_key,
                score,
                confidence,
                closed_positions,
                win_rate,
                roi,
                realized_pnl_usd,
                observed_trade_count,
                updated_at
              FROM polymarket.wallet_segment_performance
              WHERE score_version = $1
              ORDER BY score DESC, realized_pnl_usd DESC, proxy_wallet
              LIMIT $2
            )
            SELECT jsonb_build_object(
              'score_version', $1,
              'taxonomy_version', $3,
              'status', to_jsonb(segment_status),
              'taxonomy_status', to_jsonb(taxonomy_status),
              'segments', COALESCE((SELECT jsonb_agg(to_jsonb(by_segment) ORDER BY segment_key) FROM by_segment), '[]'::jsonb),
              'top_wallets', COALESCE((SELECT jsonb_agg(to_jsonb(top_wallets) ORDER BY score DESC, realized_pnl_usd DESC) FROM top_wallets), '[]'::jsonb),
              'updated_at', now()
            )
            FROM segment_status
            CROSS JOIN taxonomy_status
            "#,
        )
        .bind(score_version)
        .bind(limit.max(1))
        .bind(GAMMA_TAXONOMY_VERSION)
        .fetch_one(&self.pool)
        .await
        .context("failed to fetch wallet segment summary")?;
        Ok(value)
    }

    pub async fn wallet_segment_status(
        &self,
        score_version: &str,
        limit: i64,
    ) -> Result<serde_json::Value> {
        self.wallet_segment_summary_by_version(score_version, limit)
            .await
    }

    pub async fn upsert_wallet_score(&self, score: &WalletScore) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.wallet_scores (
              score_id, proxy_wallet, score_version, scored_at, resolved_markets,
              total_trades, total_volume, realized_pnl, roi, win_rate,
              avg_trade_size, max_drawdown, score, metadata
            )
            VALUES (gen_random_uuid(),$1,$2,now(),$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (proxy_wallet, score_version) DO UPDATE SET
              scored_at = now(),
              resolved_markets = EXCLUDED.resolved_markets,
              total_trades = EXCLUDED.total_trades,
              total_volume = EXCLUDED.total_volume,
              realized_pnl = EXCLUDED.realized_pnl,
              roi = EXCLUDED.roi,
              win_rate = EXCLUDED.win_rate,
              avg_trade_size = EXCLUDED.avg_trade_size,
              max_drawdown = EXCLUDED.max_drawdown,
              score = EXCLUDED.score,
              metadata = EXCLUDED.metadata
            "#,
        )
        .bind(&score.proxy_wallet)
        .bind(&score.score_version)
        .bind(score.resolved_markets)
        .bind(score.total_trades)
        .bind(score.total_volume)
        .bind(score.realized_pnl)
        .bind(score.roi)
        .bind(score.win_rate)
        .bind(score.avg_trade_size)
        .bind(score.max_drawdown)
        .bind(score.score)
        .bind(&score.metadata)
        .execute(&self.pool)
        .await
        .context("failed to upsert wallet score")?;
        Ok(())
    }

    pub async fn fetch_latest_wallet_scores(&self) -> Result<Vec<WalletScore>> {
        let rows = sqlx::query_as::<_, WalletScoreRow>(
            r#"
            SELECT DISTINCT ON (proxy_wallet)
              proxy_wallet, score_version, resolved_markets, total_trades, total_volume,
              realized_pnl, roi, win_rate, avg_trade_size, max_drawdown, score, metadata
            FROM polymarket.wallet_scores
            ORDER BY proxy_wallet, scored_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch latest wallet scores")?;
        Ok(rows.into_iter().map(Into::into).collect())
    }

    pub async fn fetch_wallet_score(&self, proxy_wallet: &str) -> Result<Option<WalletScore>> {
        let row = sqlx::query_as::<_, WalletScoreRow>(
            r#"
            SELECT proxy_wallet, score_version, resolved_markets, total_trades, total_volume,
              realized_pnl, roi, win_rate, avg_trade_size, max_drawdown, score, metadata
            FROM polymarket.wallet_scores
            WHERE proxy_wallet = $1
            ORDER BY scored_at DESC
            LIMIT 1
            "#,
        )
        .bind(proxy_wallet)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch wallet score")?;
        Ok(row.map(Into::into))
    }

    pub async fn upsert_wallet_performance(&self, performance: &WalletPerformance) -> Result<()> {
        let rank_score = performance.realized_pnl_usd * performance.roi;
        sqlx::query(
            r#"
            INSERT INTO polymarket.wallet_performance (
              proxy_wallet, sample_updated_at, realized_pnl_usd, total_bought_usd,
              roi, closed_positions, winning_positions, win_rate, rank_score,
              raw_payload, metadata, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,now())
            ON CONFLICT (proxy_wallet) DO UPDATE SET
              sample_updated_at = EXCLUDED.sample_updated_at,
              realized_pnl_usd = EXCLUDED.realized_pnl_usd,
              total_bought_usd = EXCLUDED.total_bought_usd,
              roi = EXCLUDED.roi,
              closed_positions = EXCLUDED.closed_positions,
              winning_positions = EXCLUDED.winning_positions,
              win_rate = EXCLUDED.win_rate,
              rank_score = EXCLUDED.rank_score,
              raw_payload = EXCLUDED.raw_payload,
              metadata = EXCLUDED.metadata,
              updated_at = now()
            "#,
        )
        .bind(&performance.proxy_wallet)
        .bind(performance.sample_updated_at)
        .bind(performance.realized_pnl_usd)
        .bind(performance.total_bought_usd)
        .bind(performance.roi)
        .bind(performance.closed_positions)
        .bind(performance.winning_positions)
        .bind(performance.win_rate)
        .bind(rank_score)
        .bind(&performance.raw_payload)
        .bind(&performance.metadata)
        .execute(&self.pool)
        .await
        .context("failed to upsert wallet performance")?;
        Ok(())
    }

    pub async fn fetch_latest_wallet_performance(
        &self,
        proxy_wallet: &str,
    ) -> Result<Option<WalletPerformance>> {
        let row = sqlx::query_as::<_, WalletPerformanceRow>(
            r#"
            SELECT proxy_wallet, sample_updated_at, realized_pnl_usd, total_bought_usd,
              roi, closed_positions, winning_positions, win_rate, rank_score,
              raw_payload, metadata
            FROM polymarket.wallet_performance
            WHERE proxy_wallet = $1
            "#,
        )
        .bind(proxy_wallet)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch latest wallet performance")?;
        Ok(row.map(Into::into))
    }
}

fn merge_json(mut left: serde_json::Value, right: serde_json::Value) -> serde_json::Value {
    if let (Some(left), Some(right)) = (left.as_object_mut(), right.as_object()) {
        for (key, value) in right {
            left.insert(key.clone(), value.clone());
        }
    }
    left
}

fn closed_positions_from_raw_payload(
    raw_payload: &serde_json::Value,
) -> Result<Vec<DataApiClosedPosition>> {
    if raw_payload.is_array() {
        serde_json::from_value(raw_payload.clone())
            .context("failed to deserialize wallet performance closed-position payload")
    } else {
        Ok(Vec::new())
    }
}

#[derive(sqlx::FromRow)]
struct WhaleTradeRow {
    trade_id: Uuid,
    proxy_wallet: String,
    asset: String,
    condition_id: Option<String>,
    market_id: Option<String>,
    side: String,
    outcome: Option<String>,
    price: Decimal,
    size: Decimal,
    cash_value: Decimal,
    timestamp_utc: DateTime<Utc>,
    title: Option<String>,
    slug: Option<String>,
    event_slug: Option<String>,
    transaction_hash: Option<String>,
    raw_payload: serde_json::Value,
}

#[derive(sqlx::FromRow)]
struct WalletTradeGammaSegmentRow {
    proxy_wallet: String,
    taxonomy_segment: String,
    taxonomy_source: String,
    taxonomy_confidence: Decimal,
    cash_value: Decimal,
    timestamp_utc: DateTime<Utc>,
    trade_id: Uuid,
    taxonomy_metadata: serde_json::Value,
}

#[derive(sqlx::FromRow)]
struct GammaSegmentLookupRow {
    lookup_slug: String,
    event_slug: Option<String>,
    market_slug: Option<String>,
    taxonomy_segment: String,
    taxonomy_confidence: Decimal,
}

#[derive(sqlx::FromRow)]
struct GammaMarketMetadataRow {
    cache_key: String,
    lookup_type: String,
    lookup_slug: String,
    event_slug: Option<String>,
    market_slug: Option<String>,
    gamma_event_id: Option<String>,
    gamma_market_id: Option<String>,
    category: Option<String>,
    series_slug: Option<String>,
    tag_slugs: Vec<String>,
    sport_key: Option<String>,
    taxonomy_segment: Option<String>,
    taxonomy_source: String,
    taxonomy_confidence: Decimal,
    taxonomy_version: String,
    raw_payload: serde_json::Value,
    fetched_at: DateTime<Utc>,
}

impl From<GammaMarketMetadataRow> for GammaMarketMetadata {
    fn from(row: GammaMarketMetadataRow) -> Self {
        Self {
            cache_key: row.cache_key,
            lookup_type: row.lookup_type,
            lookup_slug: row.lookup_slug,
            event_slug: row.event_slug,
            market_slug: row.market_slug,
            gamma_event_id: row.gamma_event_id,
            gamma_market_id: row.gamma_market_id,
            category: row.category,
            series_slug: row.series_slug,
            tag_slugs: row.tag_slugs,
            sport_key: row.sport_key,
            taxonomy_segment: row.taxonomy_segment,
            taxonomy_source: row.taxonomy_source,
            taxonomy_confidence: row.taxonomy_confidence,
            taxonomy_version: row.taxonomy_version,
            raw_payload: row.raw_payload,
            fetched_at: row.fetched_at,
        }
    }
}

#[derive(sqlx::FromRow)]
struct WalletTradeTaxonomyCandidateRow {
    trade_id: Uuid,
    title: Option<String>,
    slug: Option<String>,
    event_slug: Option<String>,
    market_id: Option<String>,
    condition_id: Option<String>,
    asset: String,
    raw_payload: serde_json::Value,
}

impl From<WalletTradeTaxonomyCandidateRow> for WalletTradeTaxonomyCandidate {
    fn from(row: WalletTradeTaxonomyCandidateRow) -> Self {
        Self {
            trade_id: row.trade_id,
            title: row.title,
            slug: row.slug,
            event_slug: row.event_slug,
            market_id: row.market_id,
            condition_id: row.condition_id,
            asset: row.asset,
            raw_payload: row.raw_payload,
        }
    }
}

impl From<WhaleTradeRow> for WhaleTrade {
    fn from(row: WhaleTradeRow) -> Self {
        Self {
            trade_id: row.trade_id,
            proxy_wallet: row.proxy_wallet,
            asset: row.asset,
            condition_id: row.condition_id,
            market_id: row.market_id,
            side: row.side,
            outcome: row.outcome,
            price: row.price,
            size: row.size,
            cash_value: row.cash_value,
            timestamp_utc: row.timestamp_utc,
            title: row.title,
            slug: row.slug,
            event_slug: row.event_slug,
            transaction_hash: row.transaction_hash,
            raw_payload: row.raw_payload,
        }
    }
}

#[derive(sqlx::FromRow)]
struct WalletScoreRow {
    proxy_wallet: String,
    score_version: String,
    resolved_markets: i32,
    total_trades: i32,
    total_volume: Decimal,
    realized_pnl: Decimal,
    roi: Decimal,
    win_rate: Decimal,
    avg_trade_size: Decimal,
    max_drawdown: Decimal,
    score: Decimal,
    metadata: serde_json::Value,
}

#[derive(sqlx::FromRow)]
struct WalletScoreRefreshJobRow {
    queue_id: Uuid,
    proxy_wallet: String,
    score_version: String,
    status: String,
    refresh_reason: String,
    requested_at: DateTime<Utc>,
    available_at: DateTime<Utc>,
    started_at: Option<DateTime<Utc>>,
    completed_at: Option<DateTime<Utc>>,
    attempt_count: i32,
    max_attempts: i32,
    last_error: Option<String>,
    request_metadata: serde_json::Value,
    result_metadata: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct WalletObservedTradeStats {
    pub observed_trade_count: i32,
    pub observed_volume_usd: Decimal,
    pub observed_market_count: i32,
    pub avg_trade_size: Decimal,
    pub sample_start: Option<DateTime<Utc>>,
    pub sample_end: Option<DateTime<Utc>>,
}

#[derive(Debug, FromRow)]
struct WalletObservedTradeStatsRow {
    observed_trade_count: i32,
    observed_volume_usd: Decimal,
    observed_market_count: i32,
    avg_trade_size: Decimal,
    sample_start: Option<DateTime<Utc>>,
    sample_end: Option<DateTime<Utc>>,
}

impl From<WalletObservedTradeStatsRow> for WalletObservedTradeStats {
    fn from(row: WalletObservedTradeStatsRow) -> Self {
        Self {
            observed_trade_count: row.observed_trade_count,
            observed_volume_usd: row.observed_volume_usd,
            observed_market_count: row.observed_market_count,
            avg_trade_size: row.avg_trade_size,
            sample_start: row.sample_start,
            sample_end: row.sample_end,
        }
    }
}

impl From<WalletScoreRow> for WalletScore {
    fn from(row: WalletScoreRow) -> Self {
        Self {
            proxy_wallet: row.proxy_wallet,
            score_version: row.score_version,
            resolved_markets: row.resolved_markets,
            total_trades: row.total_trades,
            total_volume: row.total_volume,
            realized_pnl: row.realized_pnl,
            roi: row.roi,
            win_rate: row.win_rate,
            avg_trade_size: row.avg_trade_size,
            max_drawdown: row.max_drawdown,
            score: row.score,
            metadata: row.metadata,
        }
    }
}

impl TryFrom<WalletScoreRefreshJobRow> for WalletScoreRefreshJob {
    type Error = anyhow::Error;

    fn try_from(row: WalletScoreRefreshJobRow) -> Result<Self> {
        let status = match row.status.as_str() {
            "queued" => WalletScoreRefreshStatus::Queued,
            "running" => WalletScoreRefreshStatus::Running,
            "completed" => WalletScoreRefreshStatus::Completed,
            "failed" => WalletScoreRefreshStatus::Failed,
            "cancelled" => WalletScoreRefreshStatus::Cancelled,
            other => bail!("unknown wallet score refresh status {other}"),
        };
        Ok(Self {
            queue_id: row.queue_id,
            proxy_wallet: row.proxy_wallet,
            score_version: row.score_version,
            status,
            refresh_reason: row.refresh_reason,
            requested_at: row.requested_at,
            available_at: row.available_at,
            started_at: row.started_at,
            completed_at: row.completed_at,
            attempt_count: row.attempt_count,
            max_attempts: row.max_attempts,
            last_error: row.last_error,
            request_metadata: row.request_metadata,
            result_metadata: row.result_metadata,
        })
    }
}

#[derive(sqlx::FromRow)]
struct WalletPerformanceRow {
    proxy_wallet: String,
    sample_updated_at: DateTime<Utc>,
    realized_pnl_usd: Decimal,
    total_bought_usd: Decimal,
    roi: Decimal,
    closed_positions: i32,
    winning_positions: i32,
    win_rate: Decimal,
    rank_score: Decimal,
    raw_payload: serde_json::Value,
    metadata: serde_json::Value,
}

impl From<WalletPerformanceRow> for WalletPerformance {
    fn from(row: WalletPerformanceRow) -> Self {
        Self {
            proxy_wallet: row.proxy_wallet,
            sample_updated_at: row.sample_updated_at,
            realized_pnl_usd: row.realized_pnl_usd,
            total_bought_usd: row.total_bought_usd,
            roi: row.roi,
            closed_positions: row.closed_positions,
            winning_positions: row.winning_positions,
            win_rate: row.win_rate,
            rank_score: row.rank_score,
            raw_payload: row.raw_payload,
            metadata: row.metadata,
        }
    }
}

#[derive(sqlx::FromRow)]
struct WalletSegmentPerformanceRow {
    proxy_wallet: String,
    segment_key: String,
    score_version: String,
    classifier_version: String,
    score: Decimal,
    confidence: Decimal,
    closed_positions: i32,
    winning_positions: i32,
    losing_positions: i32,
    win_rate: Decimal,
    realized_pnl_usd: Decimal,
    total_bought_usd: Decimal,
    roi: Decimal,
    observed_trade_count: i32,
    observed_volume_usd: Decimal,
    sample_start: Option<DateTime<Utc>>,
    sample_end: Option<DateTime<Utc>>,
    metadata: serde_json::Value,
}

#[derive(sqlx::FromRow)]
struct GammaWalletSegmentPerformanceInputRow {
    proxy_wallet: String,
    segment_key: String,
    observed_trade_count: i32,
    observed_volume_usd: Decimal,
    buy_volume_usd: Decimal,
    sell_volume_usd: Decimal,
    observed_market_count: i32,
    avg_taxonomy_confidence: Decimal,
    sample_start: Option<DateTime<Utc>>,
    sample_end: Option<DateTime<Utc>>,
    taxonomy_metadata: serde_json::Value,
}

impl GammaWalletSegmentPerformanceInputRow {
    fn into_wallet_segment_performance(self) -> WalletSegmentPerformance {
        let taxonomy_confidence = self
            .avg_taxonomy_confidence
            .max(Decimal::ZERO)
            .min(Decimal::ONE);
        let taxonomy_segment = self.segment_key;
        let classification = classify_gamma_taxonomy_segment(
            &taxonomy_segment,
            taxonomy_confidence,
            self.taxonomy_metadata.clone(),
        );
        let segment_key = classification
            .as_ref()
            .map(|classification| classification.segment_key.clone())
            .or_else(|| normalize_gamma_segment_key(&taxonomy_segment))
            .unwrap_or_else(|| taxonomy_segment.trim().to_ascii_lowercase());
        let input = WalletSegmentPerformanceInput {
            proxy_wallet: self.proxy_wallet,
            segment_key,
            classifier_version: GAMMA_SEGMENT_CLASSIFIER_VERSION.to_string(),
            closed_positions: 0,
            winning_positions: 0,
            realized_pnl_usd: Decimal::ZERO,
            total_bought_usd: self.buy_volume_usd,
            observed_trade_count: self.observed_trade_count,
            observed_volume_usd: self.observed_volume_usd,
            sample_start: self.sample_start,
            sample_end: self.sample_end,
        };
        let mut performance = score_wallet_segment(input).into_wallet_segment_performance();
        performance.score_version = MRS_SEGMENT_V2_SCORE_VERSION.to_string();
        performance.confidence = ((performance.confidence * dec!(0.70))
            + (taxonomy_confidence * dec!(0.30)))
        .min(Decimal::ONE)
        .round_dp(4);
        performance.metadata = merge_json(
            performance.metadata,
            serde_json::json!({
                "source": "wallet_trades_gamma_taxonomy",
                "score_basis": MRS_SEGMENT_V2_SCORE_VERSION,
                "score_version": MRS_SEGMENT_V2_SCORE_VERSION,
                "classifier_version": GAMMA_SEGMENT_CLASSIFIER_VERSION,
                "raw_taxonomy_segment": taxonomy_segment,
                "taxonomy_confidence": taxonomy_confidence,
                "observed_market_count": self.observed_market_count,
                "buy_volume_usd": self.buy_volume_usd,
                "sell_volume_usd": self.sell_volume_usd,
                "closed_positions_basis": "not_inferred_from_wallet_trades",
                "taxonomy": self.taxonomy_metadata
            }),
        );
        performance
    }
}

impl From<WalletSegmentPerformanceRow> for WalletSegmentPerformance {
    fn from(row: WalletSegmentPerformanceRow) -> Self {
        Self {
            proxy_wallet: row.proxy_wallet,
            segment_key: row.segment_key,
            score_version: row.score_version,
            classifier_version: row.classifier_version,
            score: row.score,
            confidence: row.confidence,
            closed_positions: row.closed_positions,
            winning_positions: row.winning_positions,
            losing_positions: row.losing_positions,
            win_rate: row.win_rate,
            realized_pnl_usd: row.realized_pnl_usd,
            total_bought_usd: row.total_bought_usd,
            roi: row.roi,
            observed_trade_count: row.observed_trade_count,
            observed_volume_usd: row.observed_volume_usd,
            sample_start: row.sample_start,
            sample_end: row.sample_end,
            metadata: row.metadata,
        }
    }
}

fn trading_process_from_row(row: TradingProcessRow) -> Result<TradingProcess> {
    let config = serde_json::from_value(row.config.clone())
        .context("failed to deserialize trading process config")?;
    Ok(TradingProcess {
        process_id: row.process_id,
        name: row.name,
        process_type: row.process_type,
        process_scope: row.process_scope,
        process_key: row.process_key,
        status: row.status,
        enabled: row.enabled,
        config,
        metadata: row.metadata,
        created_at: row.created_at,
        updated_at: row.updated_at,
        started_at: row.started_at,
        stopped_at: row.stopped_at,
        last_error: row.last_error,
    })
}

fn serialized_name<T: Serialize>(value: &T) -> Result<String> {
    match serde_json::to_value(value).context("failed to serialize enum value")? {
        serde_json::Value::String(value) => Ok(value),
        other => bail!("expected enum to serialize as string, got {other}"),
    }
}

fn order_from_db_row(row: OrderDbRow) -> Result<OrderRecord> {
    let mut order: OrderRecord = serde_json::from_value(row.raw_payload.clone())
        .context("failed to deserialize persisted order raw_payload")?;
    order.order_id = row.order_id;
    order.state = serde_json::from_value(serde_json::Value::String(row.state))
        .context("failed to deserialize persisted order state")?;
    order.created_at = row.created_at;
    order.updated_at = row.updated_at;
    Ok(order)
}

fn gamma_segment_for_closed_position(
    position: &DataApiClosedPosition,
    lookup: &HashMap<String, String>,
) -> Option<String> {
    [
        position.slug.as_deref(),
        position.event_slug.as_deref(),
        position.condition_id.as_deref(),
        position.asset.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find_map(|key| {
        let key = key.to_ascii_lowercase();
        lookup
            .get(&key)
            .or_else(|| lookup.get(&format!("event:{key}")))
            .or_else(|| lookup.get(&format!("market:{key}")))
            .cloned()
    })
}

fn timestamp_from_secs(timestamp: i64) -> Option<DateTime<Utc>> {
    Utc.timestamp_opt(timestamp, 0).single()
}

#[cfg(test)]
mod tests {
    use crate::store::HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL;

    #[test]
    fn manager_heartbeat_cannot_revive_inactive_processes() {
        assert!(HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL.contains("enabled = true"));
        assert!(HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL
            .contains("status IN ('starting', 'running', 'stopping')"));
        assert!(!HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL.contains("SET status"));
        assert!(!HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL.contains("SET enabled"));
    }
}
