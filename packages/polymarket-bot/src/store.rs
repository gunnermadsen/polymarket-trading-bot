use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::{Decimal, RoundingStrategy};
use serde::{Deserialize, Serialize};
use sqlx::{postgres::PgPoolOptions, FromRow, PgPool};
use uuid::Uuid;

use crate::{
    config::PostgresConfig,
    events::ServiceEvent,
    execution::live::LiveVenueEvent,
    execution::OrderPlanReport,
    models::{
        FillRecord, OrderRecord, OrderRequest, OrderState, TradingProcess, TradingProcessConfig,
    },
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

const RECORD_IDEMPOTENT_TRADING_PROCESS_EVENT_SQL: &str = r#"
INSERT INTO polymarket.trading_process_events (
  event_id, process_id, timestamp_utc, level, event_type, message, metadata, created_at
)
VALUES ($1, $2, $3, $4, $5, $6, $7, now())
ON CONFLICT (event_id, timestamp_utc) DO NOTHING
"#;

const INSERT_ORDER_SQL: &str = r#"
INSERT INTO polymarket.orders (
  order_id, client_order_id, process_id, created_at, updated_at, market_id, token_id,
  side, order_type, price, size, state, raw_payload
)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
ON CONFLICT (client_order_id) DO NOTHING
"#;

const INSERT_FILL_IDENTITY_SQL: &str = r#"
INSERT INTO polymarket.fill_identities (
  fill_id, process_id, timestamp_utc
)
VALUES ($1,$2,$3)
ON CONFLICT (fill_id) DO NOTHING
"#;

const SELECT_FILL_IDENTITY_SQL: &str = r#"
SELECT process_id, timestamp_utc
FROM polymarket.fill_identities
WHERE fill_id = $1
"#;

const INSERT_FILL_SQL: &str = r#"
INSERT INTO polymarket.fills (
  fill_id, process_id, order_id, token_id, timestamp_utc, price, size, fee, source, raw_payload
)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
ON CONFLICT (fill_id, timestamp_utc) DO NOTHING
"#;

const SELECT_FILL_SQL: &str = r#"
SELECT process_id, order_id, token_id, timestamp_utc, price, size, fee, source
FROM polymarket.fills
WHERE fill_id = $1
  AND timestamp_utc = $2
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
struct FillDbRow {
    process_id: Option<Uuid>,
    order_id: String,
    token_id: String,
    timestamp_utc: DateTime<Utc>,
    price: Decimal,
    size: Decimal,
    fee: Decimal,
    source: String,
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

impl Store {
    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn connect(config: &PostgresConfig) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&config.database_url())
            .await
            .context("failed to connect to Postgres")?;
        Ok(Self::from_pool(pool))
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

    pub async fn record_trading_process_event_idempotent(
        &self,
        event_id: Uuid,
        timestamp_utc: DateTime<Utc>,
        process_id: Uuid,
        level: &str,
        event_type: &str,
        message: Option<&str>,
        metadata: serde_json::Value,
    ) -> Result<bool> {
        let result = sqlx::query(RECORD_IDEMPOTENT_TRADING_PROCESS_EVENT_SQL)
            .bind(event_id)
            .bind(process_id)
            .bind(timestamp_utc)
            .bind(level)
            .bind(event_type)
            .bind(message)
            .bind(metadata)
            .execute(&self.pool)
            .await
            .context("failed to record idempotent trading process event")?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn upsert_btc_realtime_paper_process_by_key(
        &self,
        name: &str,
        process_key: &str,
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
                  status = $3,
                  enabled = false,
                  config = $4,
                  metadata = $5,
                  stopped_at = CASE
                    WHEN $3 IN ('stopped', 'failed', 'expired', 'completed')
                      THEN COALESCE(stopped_at, now())
                    ELSE stopped_at
                  END,
                  last_error = CASE
                    WHEN $3 NOT IN ('failed', 'error') THEN NULL
                    ELSE last_error
                  END,
                  updated_at = now()
              WHERE process_type = 'btc_5m'
                AND process_scope = 'realtime_paper'
                AND process_key = $2
              RETURNING process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
                created_at, updated_at, started_at, stopped_at, last_error
            ),
            inserted AS (
              INSERT INTO polymarket.trading_processes (
                process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
                created_at, updated_at, started_at, stopped_at, last_error
              )
              SELECT
                gen_random_uuid(), $1, 'btc_5m', 'realtime_paper', $2, $3, false, $4, $5,
                now(), now(),
                NULL,
                CASE WHEN $3 IN ('stopped', 'failed', 'expired', 'completed') THEN now() ELSE NULL END,
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
        .bind(process_key)
        .bind(status)
        .bind(config_value)
        .bind(metadata)
        .fetch_one(&self.pool)
        .await
        .context("failed to upsert BTC realtime-paper process by key")?;
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

    pub async fn get_btc_realtime_paper_process_by_key(
        &self,
        process_key: &str,
    ) -> Result<Option<TradingProcess>> {
        let row = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            SELECT process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            FROM polymarket.trading_processes
            WHERE process_type = 'btc_5m'
              AND process_scope = 'realtime_paper'
              AND process_key = $1
            "#,
        )
        .bind(process_key)
        .fetch_optional(&self.pool)
        .await
        .context("failed to get BTC realtime-paper process by key")?;
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

    pub async fn update_btc_realtime_paper_process_definition(
        &self,
        process_id: Uuid,
        name: Option<&str>,
        config: Option<TradingProcessConfig>,
        metadata: Option<serde_json::Value>,
    ) -> Result<Option<TradingProcess>> {
        let config_value = config.map(serde_json::to_value).transpose()?;
        let row = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            UPDATE polymarket.trading_processes
            SET name = COALESCE($2, name),
                config = COALESCE($3, config),
                metadata = COALESCE($4, metadata),
                updated_at = now()
            WHERE process_id = $1
              AND process_type = 'btc_5m'
              AND process_scope = 'realtime_paper'
            RETURNING process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            "#,
        )
        .bind(process_id)
        .bind(name)
        .bind(config_value)
        .bind(metadata)
        .fetch_optional(&self.pool)
        .await
        .context("failed to update BTC realtime-paper process definition")?;
        row.map(trading_process_from_row).transpose()
    }

    pub async fn insert_order(&self, order: &OrderRecord) -> Result<()> {
        if self.try_insert_order(order).await? {
            return Ok(());
        }
        let existing = self
            .find_order_by_client_order_id(order.request.client_order_id)
            .await?
            .context("conflicting order disappeared during identity verification")?;
        if !order_request_result_matches(&existing.request, &order.request)
            || existing.order_id != order.order_id
            || existing.state != order.state
        {
            bail!(
                "client_order_id {} collides with immutable order identity, result, or reference execution evidence",
                order.request.client_order_id
            );
        }
        Ok(())
    }

    async fn try_insert_order(&self, order: &OrderRecord) -> Result<bool> {
        let side = serialized_name(&order.request.side)?;
        let order_type = serialized_name(&order.request.order_type)?;
        let state = serialized_name(&order.state)?;
        let result = sqlx::query(INSERT_ORDER_SQL)
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
        Ok(result.rows_affected() == 1)
    }

    pub async fn create_pending_order(
        &self,
        request: &OrderRequest,
    ) -> Result<(OrderRecord, bool)> {
        let now = Utc::now();
        let order = OrderRecord {
            order_id: format!("live-pending-{}", request.client_order_id),
            request: request.clone(),
            state: OrderState::Submitted,
            created_at: now,
            updated_at: now,
        };
        if self.try_insert_order(&order).await? {
            return Ok((order, true));
        }
        let existing = self
            .find_order_by_client_order_id(request.client_order_id)
            .await?
            .context("pending order disappeared during identity verification")?;
        if !order_request_identity_matches(&existing.request, request) {
            bail!(
                "client_order_id {} collides with immutable order identity or reference execution evidence",
                request.client_order_id
            );
        }
        Ok((existing, false))
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
        let fill = canonical_fill_for_storage(fill)?;
        let process_id = required_fill_process_id(&fill)?;
        let mut transaction = self
            .pool
            .begin()
            .await
            .context("failed to begin fill identity transaction")?;
        // The Timescale hypertable key includes timestamp_utc. Claim the deterministic fill_id in
        // a small canonical table first so different timestamps cannot become separate fills.
        let identity = sqlx::query(INSERT_FILL_IDENTITY_SQL)
            .bind(fill.fill_id)
            .bind(process_id)
            .bind(fill.filled_at)
            .execute(&mut *transaction)
            .await
            .context("failed to claim deterministic fill identity")?;
        if identity.rows_affected() == 0 {
            let (existing_process_id, existing_timestamp) =
                sqlx::query_as::<_, (Option<Uuid>, DateTime<Utc>)>(SELECT_FILL_IDENTITY_SQL)
                    .bind(fill.fill_id)
                    .fetch_optional(&mut *transaction)
                    .await
                    .context("failed to verify canonical fill identity")?
                    .context("canonical fill identity disappeared during verification")?;
            if existing_process_id != Some(process_id) || existing_timestamp != fill.filled_at {
                transaction
                    .rollback()
                    .await
                    .context("failed to release conflicting fill identity transaction")?;
                bail!(
                    "fill_id {} collides with a different process or execution timestamp",
                    fill.fill_id
                );
            }
            let durable = sqlx::query_as::<_, FillDbRow>(SELECT_FILL_SQL)
                .bind(fill.fill_id)
                .bind(fill.filled_at)
                .fetch_optional(&mut *transaction)
                .await
                .context("failed to verify fill for canonical identity")?
                .context("canonical fill identity has no durable fill row")?;
            let durable = fill_from_db_row(fill.fill_id, durable)?;
            if !fill_record_matches(&durable, &fill) {
                transaction
                    .rollback()
                    .await
                    .context("failed to release conflicting durable fill transaction")?;
                bail!(
                    "fill_id {} has a canonical identity but conflicting durable fill",
                    fill.fill_id
                );
            }
            transaction
                .commit()
                .await
                .context("failed to commit idempotent fill verification")?;
            return Ok(());
        }
        let source = serialized_name(&fill.source)?;
        let raw_payload = serde_json::to_value(&fill)?;
        let result = sqlx::query(INSERT_FILL_SQL)
            .bind(fill.fill_id)
            .bind(process_id)
            .bind(&fill.order_id)
            .bind(&fill.token_id)
            .bind(fill.filled_at)
            .bind(fill.price)
            .bind(fill.size)
            .bind(fill.fee)
            .bind(source)
            .bind(raw_payload)
            .execute(&mut *transaction)
            .await
            .context("failed to insert fill")?;
        if result.rows_affected() == 0 {
            transaction
                .rollback()
                .await
                .context("failed to release noncanonical fill transaction")?;
            bail!(
                "newly claimed fill_id {} already has a durable fill row",
                fill.fill_id
            );
        }
        transaction
            .commit()
            .await
            .context("failed to commit canonical fill")?;
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

    pub async fn healthcheck(&self) -> Result<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .context("Postgres healthcheck failed")?;
        Ok(())
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

fn required_fill_process_id(fill: &FillRecord) -> Result<Uuid> {
    fill.process_id.with_context(|| {
        format!(
            "fill {} is missing canonical process_id ownership",
            fill.fill_id
        )
    })
}

fn canonical_fill_for_storage(fill: &FillRecord) -> Result<FillRecord> {
    let mut canonical = fill.clone();
    canonical.filled_at = DateTime::<Utc>::from_timestamp_micros(fill.filled_at.timestamp_micros())
        .context("fill timestamp is outside the PostgreSQL timestamptz range")?;
    canonical.price = fill
        .price
        .round_dp_with_strategy(8, RoundingStrategy::MidpointAwayFromZero);
    canonical.size = fill
        .size
        .round_dp_with_strategy(10, RoundingStrategy::MidpointAwayFromZero);
    canonical.fee = fill
        .fee
        .round_dp_with_strategy(10, RoundingStrategy::MidpointAwayFromZero);
    Ok(canonical)
}

fn fill_from_db_row(fill_id: Uuid, row: FillDbRow) -> Result<FillRecord> {
    let source = serde_json::from_value(serde_json::Value::String(row.source))
        .context("failed to deserialize persisted fill source")?;
    Ok(FillRecord {
        fill_id,
        process_id: row.process_id,
        order_id: row.order_id,
        token_id: row.token_id,
        price: row.price,
        size: row.size,
        fee: row.fee,
        source,
        filled_at: row.timestamp_utc,
    })
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

fn order_request_identity_matches(existing: &OrderRequest, incoming: &OrderRequest) -> bool {
    existing.client_order_id == incoming.client_order_id
        && existing.process_id == incoming.process_id
        && existing.market_id == incoming.market_id
        && existing.token_id == incoming.token_id
        && existing.side == incoming.side
        && existing.order_type == incoming.order_type
        && existing.price == incoming.price
        && existing.size == incoming.size
        && existing.signal_id == incoming.signal_id
        && immutable_order_metadata_matches(&existing.metadata, &incoming.metadata)
}

fn order_request_result_matches(existing: &OrderRequest, incoming: &OrderRequest) -> bool {
    existing.client_order_id == incoming.client_order_id
        && existing.process_id == incoming.process_id
        && existing.market_id == incoming.market_id
        && existing.token_id == incoming.token_id
        && existing.side == incoming.side
        && existing.order_type == incoming.order_type
        && existing.price == incoming.price
        && existing.size == incoming.size
        && existing.signal_id == incoming.signal_id
        && existing.metadata == incoming.metadata
}

fn fill_record_matches(existing: &FillRecord, incoming: &FillRecord) -> bool {
    existing.fill_id == incoming.fill_id
        && existing.process_id == incoming.process_id
        && existing.order_id == incoming.order_id
        && existing.token_id == incoming.token_id
        && existing.price == incoming.price
        && existing.size == incoming.size
        && existing.fee == incoming.fee
        && existing.source == incoming.source
        && existing.filled_at == incoming.filled_at
}

fn immutable_order_metadata_matches(
    existing: &serde_json::Value,
    incoming: &serde_json::Value,
) -> bool {
    let (Some(existing), Some(incoming)) = (existing.as_object(), incoming.as_object()) else {
        return existing == incoming;
    };
    let existing_len = existing
        .keys()
        .filter(|key| !execution_generated_metadata_key(key))
        .count();
    let incoming_len = incoming
        .keys()
        .filter(|key| !execution_generated_metadata_key(key))
        .count();
    existing_len == incoming_len
        && existing.iter().all(|(key, value)| {
            execution_generated_metadata_key(key) || incoming.get(key) == Some(value)
        })
}

fn execution_generated_metadata_key(key: &str) -> bool {
    matches!(
        key,
        "paper_execution"
            | "reject_reason"
            | "reference_execution_submit"
            | "reference_execution_arrival"
    )
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    use crate::{
        models::{FillRecord, FillSource, OrderRequest, OrderSide, OrderType},
        store::{
            canonical_fill_for_storage, fill_record_matches, order_request_identity_matches,
            order_request_result_matches, required_fill_process_id,
            HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL, INSERT_FILL_IDENTITY_SQL, INSERT_FILL_SQL,
            INSERT_ORDER_SQL, RECORD_IDEMPOTENT_TRADING_PROCESS_EVENT_SQL,
            SELECT_FILL_IDENTITY_SQL, SELECT_FILL_SQL,
        },
    };

    #[test]
    fn manager_heartbeat_cannot_revive_inactive_processes() {
        assert!(HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL.contains("enabled = true"));
        assert!(HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL
            .contains("status IN ('starting', 'running', 'stopping')"));
        assert!(!HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL.contains("SET status"));
        assert!(!HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL.contains("SET enabled"));
    }

    #[test]
    fn idempotent_process_events_use_the_table_composite_primary_key() {
        assert!(RECORD_IDEMPOTENT_TRADING_PROCESS_EVENT_SQL
            .contains("ON CONFLICT (event_id, timestamp_utc) DO NOTHING"));
        assert!(RECORD_IDEMPOTENT_TRADING_PROCESS_EVENT_SQL
            .contains("VALUES ($1, $2, $3, $4, $5, $6, $7, now())"));
    }

    #[test]
    fn order_insert_never_mutates_an_existing_client_order() {
        assert!(
            INSERT_ORDER_SQL.contains("ON CONFLICT (client_order_id) DO NOTHING"),
            "duplicate identity must be verified in application code without rewriting the row"
        );
        assert!(!INSERT_ORDER_SQL.contains("DO UPDATE"));
        assert!(!INSERT_ORDER_SQL.contains("EXCLUDED."));
    }

    #[test]
    fn order_identity_ignores_only_execution_generated_metadata() {
        let request = OrderRequest {
            client_order_id: Uuid::from_u128(1),
            process_id: Some(Uuid::from_u128(2)),
            market_id: "market".to_string(),
            token_id: "token".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.40),
            size: dec!(2),
            signal_id: Some(Uuid::from_u128(3)),
            metadata: serde_json::json!({
                "execution_intent": "entry",
                "reference_execution_guard": { "evidence_sha256": "evidence" },
            }),
        };
        let mut observed = request.clone();
        observed.metadata["paper_execution"] = serde_json::json!({ "arrival_at": "later" });
        observed.metadata["reference_execution_submit"] =
            serde_json::json!({ "status": "accepted" });
        assert!(order_request_identity_matches(&observed, &request));
        assert!(!order_request_result_matches(&observed, &request));

        let mut changed_guard = request.clone();
        changed_guard.metadata["reference_execution_guard"]["evidence_sha256"] =
            serde_json::json!("changed");
        assert!(!order_request_identity_matches(&changed_guard, &request));

        let mut changed_intent = request.clone();
        changed_intent.metadata["execution_intent"] = serde_json::json!("exit");
        assert!(!order_request_identity_matches(&changed_intent, &request));
    }

    #[test]
    fn duplicate_fill_requires_exact_process_order_token_and_economics() {
        let fill = FillRecord {
            fill_id: Uuid::from_u128(10),
            process_id: Some(Uuid::from_u128(11)),
            order_id: "order".to_string(),
            token_id: "token".to_string(),
            price: dec!(0.40),
            size: dec!(2),
            fee: dec!(0.12),
            source: FillSource::Paper,
            filled_at: Utc::now(),
        };
        assert!(fill_record_matches(&fill, &fill));

        let mut changed = fill.clone();
        changed.price = dec!(0.41);
        assert!(!fill_record_matches(&fill, &changed));

        changed = fill.clone();
        changed.order_id = "other-order".to_string();
        assert!(!fill_record_matches(&fill, &changed));

        changed = fill.clone();
        changed.token_id = "other-token".to_string();
        assert!(!fill_record_matches(&fill, &changed));

        changed = fill.clone();
        changed.size = dec!(3);
        assert!(!fill_record_matches(&fill, &changed));

        changed = fill.clone();
        changed.fee = dec!(0.13);
        assert!(!fill_record_matches(&fill, &changed));

        changed = fill.clone();
        changed.source = FillSource::Live;
        assert!(!fill_record_matches(&fill, &changed));

        changed = fill.clone();
        changed.process_id = Some(Uuid::from_u128(12));
        assert!(!fill_record_matches(&fill, &changed));

        changed = fill.clone();
        changed.filled_at += chrono::Duration::microseconds(1);
        assert!(!fill_record_matches(&fill, &changed));

        assert_eq!(
            required_fill_process_id(&fill).unwrap(),
            Uuid::from_u128(11)
        );
        let mut unowned = fill;
        unowned.process_id = None;
        assert!(required_fill_process_id(&unowned)
            .unwrap_err()
            .to_string()
            .contains("missing canonical process_id"));
    }

    #[test]
    fn fill_insert_claims_a_global_identity_before_the_timescale_row() {
        assert!(INSERT_FILL_IDENTITY_SQL.contains("ON CONFLICT (fill_id) DO NOTHING"));
        assert!(SELECT_FILL_IDENTITY_SQL.contains("WHERE fill_id = $1"));
        assert!(SELECT_FILL_IDENTITY_SQL.contains("process_id, timestamp_utc"));
        assert!(INSERT_FILL_SQL.contains("ON CONFLICT (fill_id, timestamp_utc) DO NOTHING"));
        assert!(SELECT_FILL_SQL.contains("WHERE fill_id = $1"));
        assert!(SELECT_FILL_SQL.contains("timestamp_utc = $2"));
        assert!(!SELECT_FILL_SQL.contains("raw_payload"));
    }

    #[test]
    fn fill_identity_uses_the_exact_postgres_storage_representation() {
        let fill = FillRecord {
            fill_id: Uuid::from_u128(20),
            process_id: Some(Uuid::from_u128(21)),
            order_id: "order".to_string(),
            token_id: "token".to_string(),
            price: dec!(0.123456785),
            size: dec!(2.12345678905),
            fee: dec!(0.00000000005),
            source: FillSource::Paper,
            filled_at: chrono::DateTime::from_timestamp(1_700_000_000, 123_456_789).unwrap(),
        };

        let canonical = canonical_fill_for_storage(&fill).unwrap();

        assert_eq!(canonical.filled_at.timestamp_subsec_nanos(), 123_456_000);
        assert_eq!(canonical.price, dec!(0.12345679));
        assert_eq!(canonical.size, dec!(2.1234567891));
        assert_eq!(canonical.fee, dec!(0.0000000001));
        assert_eq!(
            canonical_fill_for_storage(&canonical).unwrap().filled_at,
            canonical.filled_at
        );
    }
}
