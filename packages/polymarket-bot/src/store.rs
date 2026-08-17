use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::{Decimal, RoundingStrategy};
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use crate::{
    events::ServiceEvent,
    execution::{
        live::LiveVenueEvent, OrderPlanReport, LIVE_EXTERNAL_EVENT_CLOCK_SKEW,
        LIVE_FILL_RECONCILIATION_SKEW,
    },
    fees::{dynamic_crypto_taker_fee, sealed_dynamic_fee_rate},
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

const MAX_LIVE_DAILY_SETTLEMENTS: i64 = 256;
const MAX_LIVE_DAILY_SETTLEMENT_FILLS: usize = 2_048;
const MAX_LIVE_PROCESS_EXPOSURE_ORDERS: usize = 1_000;
const MAX_LIVE_PROCESS_EXPOSURE_FILLS: usize = 2_048;
const MAX_LIVE_EXPOSURE_IDENTITY_BYTES: usize = 512;
const MAX_LIVE_EXPOSURE_ORDER_PAYLOAD_BYTES: i64 = 65_536;

const SELECT_LIVE_PROCESS_EXPOSURE_ORDERS_SQL: &str = r#"
SELECT order_id, client_order_id, process_id, market_id, token_id, side, order_type,
  price, size, state, raw_payload, octet_length(raw_payload::text)::bigint AS raw_payload_bytes,
  created_at, updated_at
FROM polymarket.orders
WHERE process_id = $1
  AND state NOT IN ('filled', 'cancelled', 'rejected', 'expired')
ORDER BY created_at, order_id
LIMIT $2
"#;

const SELECT_LIVE_PROCESS_EXPOSURE_FILLS_SQL: &str = r#"
SELECT f.fill_id,
  f.process_id AS fill_process_id,
  f.order_id AS fill_order_id,
  f.token_id AS fill_token_id,
  f.timestamp_utc AS filled_at,
  f.price AS fill_price,
  f.size AS fill_size,
  f.fee AS fill_fee,
  f.source AS fill_source,
  i.process_id AS identity_process_id,
  i.timestamp_utc AS identity_timestamp_utc,
  o.process_id AS order_process_id,
  o.client_order_id AS order_client_order_id,
  o.market_id AS order_market_id,
  o.token_id AS order_token_id,
  o.side AS order_side,
  o.order_type AS order_type,
  o.price AS order_price,
  o.size AS order_size,
  o.state AS order_state,
  o.raw_payload AS order_raw_payload,
  octet_length(o.raw_payload::text)::bigint AS order_raw_payload_bytes,
  o.created_at AS order_created_at,
  o.updated_at AS order_updated_at
FROM polymarket.fills f
LEFT JOIN polymarket.fill_identities i
  ON i.fill_id = f.fill_id
 AND i.timestamp_utc = f.timestamp_utc
LEFT JOIN polymarket.orders o
  ON o.order_id = f.order_id
WHERE f.process_id = $1
  AND NOT EXISTS (
    SELECT 1
    FROM polymarket.btc_paper_settlement_ledger settlement
    WHERE settlement.process_id = $1
      AND settlement.order_id = f.order_id
      AND settlement.execution_mode = 'live'
      AND settlement.credit_status = 'credited'
  )
ORDER BY f.timestamp_utc, f.fill_id
LIMIT $2
"#;

const SELECT_LIVE_PROCESS_CROSS_OWNED_FILL_EXISTS_SQL: &str = r#"
SELECT EXISTS (
  SELECT 1
  FROM polymarket.orders o
  JOIN polymarket.fills f ON f.order_id = o.order_id
  WHERE o.process_id = $1
    AND f.process_id IS DISTINCT FROM $1
  LIMIT 1
)
"#;

const SELECT_LIVE_PROCESS_UNPROVEN_FILLED_ORDER_EXISTS_SQL: &str = r#"
SELECT EXISTS (
  SELECT 1
  FROM polymarket.orders o
  WHERE o.process_id = $1
    AND o.state IN ('filled', 'partially_filled')
    AND NOT EXISTS (
      SELECT 1
      FROM polymarket.fills f
      WHERE f.order_id = o.order_id
        AND f.process_id = $1
        AND f.source = 'live'
    )
  LIMIT 1
)
"#;

const SELECT_PENDING_LIVE_PROCESS_SETTLEMENT_EXISTS_SQL: &str = r#"
SELECT EXISTS (
  SELECT 1
  FROM polymarket.btc_paper_settlement_ledger
  WHERE process_id = $1
    AND execution_mode = 'live'
    AND credit_status = 'pending'
  LIMIT 1
)
"#;

const SELECT_CREDITED_LIVE_PROCESS_SETTLEMENTS_SQL: &str = r#"
SELECT settlement_id, run_id, process_id, order_id, market_id, token_id, fill_ids,
  official_outcome, official_winning_token_id,
  official_resolution_received_at, official_resolution_source,
  filled_size, entry_notional, entry_fees, payout, net_pnl,
  credited_at, credit_attempts, credit_evidence, created_at, updated_at
FROM polymarket.btc_paper_settlement_ledger
WHERE process_id = $1
  AND execution_mode = 'live'
  AND credit_status = 'credited'
  AND credited_at >= $2
  AND credited_at < $3
ORDER BY credited_at DESC, order_id, settlement_id
LIMIT $4
"#;

const SELECT_LIVE_SETTLEMENT_FILLS_SQL: &str = r#"
SELECT i.fill_id, i.process_id AS identity_process_id,
  f.process_id AS fill_process_id, f.order_id, f.token_id,
  f.price, f.size, f.fee, f.source
FROM polymarket.fill_identities i
JOIN polymarket.fills f
  ON f.fill_id = i.fill_id
 AND f.timestamp_utc = i.timestamp_utc
WHERE i.fill_id = ANY($1::uuid[])
ORDER BY i.fill_id
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
struct LiveDailySettlementRow {
    settlement_id: Uuid,
    run_id: Uuid,
    process_id: Uuid,
    order_id: String,
    market_id: String,
    token_id: String,
    fill_ids: serde_json::Value,
    official_outcome: String,
    official_winning_token_id: String,
    official_resolution_received_at: DateTime<Utc>,
    official_resolution_source: String,
    filled_size: Decimal,
    entry_notional: Decimal,
    entry_fees: Decimal,
    payout: Decimal,
    net_pnl: Decimal,
    credited_at: Option<DateTime<Utc>>,
    credit_attempts: i64,
    credit_evidence: serde_json::Value,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, FromRow)]
struct LiveSettlementFillRow {
    fill_id: Uuid,
    identity_process_id: Uuid,
    fill_process_id: Option<Uuid>,
    order_id: String,
    token_id: String,
    price: Decimal,
    size: Decimal,
    fee: Decimal,
    source: String,
}

#[derive(Debug, FromRow)]
struct LiveDailyAccountExitRow {
    account_trade_id: Uuid,
    process_id: Uuid,
    order_id: String,
    token_id: String,
    exit_size: Decimal,
    net_proceeds: Decimal,
    entry_filled_size: Decimal,
    entry_notional: Decimal,
    entry_fees: Decimal,
}

#[derive(Debug, FromRow)]
struct LiveExposureOrderRow {
    order_id: String,
    client_order_id: Uuid,
    process_id: Option<Uuid>,
    market_id: String,
    token_id: String,
    side: String,
    order_type: String,
    price: Decimal,
    size: Decimal,
    state: String,
    raw_payload: serde_json::Value,
    raw_payload_bytes: i64,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, FromRow)]
struct LiveExposureFillRow {
    fill_id: Uuid,
    fill_process_id: Option<Uuid>,
    fill_order_id: String,
    fill_token_id: String,
    filled_at: DateTime<Utc>,
    fill_price: Decimal,
    fill_size: Decimal,
    fill_fee: Decimal,
    fill_source: String,
    identity_process_id: Option<Uuid>,
    identity_timestamp_utc: Option<DateTime<Utc>>,
    order_process_id: Option<Uuid>,
    order_client_order_id: Option<Uuid>,
    order_market_id: Option<String>,
    order_token_id: Option<String>,
    order_side: Option<String>,
    order_type: Option<String>,
    order_price: Option<Decimal>,
    order_size: Option<Decimal>,
    order_state: Option<String>,
    order_raw_payload: Option<serde_json::Value>,
    order_raw_payload_bytes: Option<i64>,
    order_created_at: Option<DateTime<Utc>>,
    order_updated_at: Option<DateTime<Utc>>,
}

/// Conservative, process-owned capital exposure used by the live submission gate.
///
/// Filled BUY exposure is intentionally cumulative until an exact, credited live redemption is
/// persisted. A SELL or market resolution alone does not release it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LiveProcessExposureSnapshot {
    pub process_id: Uuid,
    pub has_unredeemed_settlement: bool,
    pub pending_order_count: usize,
    pub buy_fill_count: usize,
    pub pending_requested_notional_usd: Decimal,
    pub pending_requested_fees_usd: Decimal,
    pub filled_buy_notional_usd: Decimal,
    pub filled_buy_fees_usd: Decimal,
    pub total_exposure_usd: Decimal,
    pub exposed_market_ids: Vec<String>,
}

impl LiveProcessExposureSnapshot {
    pub fn exposed_market_count(&self) -> usize {
        self.exposed_market_ids.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LiveRequestedExposure {
    pub requested_notional_usd: Decimal,
    pub requested_fees_usd: Decimal,
    pub total_exposure_usd: Decimal,
}

#[derive(Debug, Clone)]
pub struct LiveRedemptionEvidence {
    pub account_address: String,
    pub condition_id: String,
    pub token_id: String,
    pub redeemed_size: Decimal,
    pub payout_usd: Decimal,
    pub redeemed_at: DateTime<Utc>,
    pub transaction_hash: String,
    pub raw_payload: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveRedemptionRecognition {
    pub matched: bool,
    pub applied: bool,
    pub redeemed_size: Decimal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveAccountExitRecognition {
    pub matched: bool,
    pub applied: bool,
    pub order_id: Option<String>,
    pub exited_size: Decimal,
}

#[derive(Debug, FromRow)]
struct LiveAccountExitCandidateRow {
    process_id: Uuid,
    order_id: String,
    filled_size: Decimal,
    entry_notional: Decimal,
    entry_fees: Decimal,
    order_payload: serde_json::Value,
}

#[derive(Debug, FromRow)]
struct LiveRedemptionRecognitionRow {
    owner_count: i64,
    owner_filled_size: Decimal,
    candidate_count: i64,
    transaction_consistent: bool,
    settlement_id: Option<Uuid>,
    settlement_process_id: Option<Uuid>,
    settlement_order_id: Option<String>,
    settlement_token_id: Option<String>,
    settlement_filled_size: Option<Decimal>,
    settlement_payout: Option<Decimal>,
    credit_status: Option<String>,
    credit_evidence: Option<serde_json::Value>,
    applied: bool,
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

#[derive(Debug, Clone, PartialEq, FromRow)]
pub(crate) struct AccountLiveFillEvidence {
    pub transaction_hash: String,
    pub order_id: String,
    pub token_id: String,
    pub side: String,
    pub price: Decimal,
    pub size: Decimal,
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
            || (existing.state != order.state
                && !durable_fill_state_supersedes_report(existing.state, order.state))
        {
            bail!(
                "client_order_id {} collides with immutable order identity, result, or reference execution evidence",
                order.request.client_order_id
            );
        }
        Ok(())
    }

    pub async fn order_venue_reject_reason(
        &self,
        process_id: Uuid,
        client_order_id: Uuid,
    ) -> Result<Option<String>> {
        sqlx::query_scalar::<_, Option<String>>(
            r#"
            SELECT NULLIF(raw_payload #>> '{venue,reject_reason}', '')
            FROM polymarket.orders
            WHERE process_id = $1
              AND client_order_id = $2
            LIMIT 1
            "#,
        )
        .bind(process_id)
        .bind(client_order_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to load durable venue order rejection reason")
        .map(Option::flatten)
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

    /// Returns a bounded snapshot of orders that can still represent live venue exposure for one
    /// trading process. Callers deliberately request one row beyond their operational bound so an
    /// unexpectedly large result can fail closed instead of silently undercounting risk.
    pub async fn live_process_nonterminal_orders(
        &self,
        process_id: Uuid,
        limit: i64,
    ) -> Result<Vec<OrderRecord>> {
        let rows = sqlx::query_as::<_, OrderDbRow>(
            r#"
            SELECT order_id, state, raw_payload, created_at, updated_at
            FROM polymarket.orders
            WHERE process_id = $1
              AND state IN (
                'created', 'submitted', 'acknowledged', 'partially_filled',
                'cancel_requested', 'unknown'
              )
            ORDER BY created_at, order_id
            LIMIT $2
            "#,
        )
        .bind(process_id)
        .bind(limit.clamp(1, 1_001))
        .fetch_all(&self.pool)
        .await
        .context("failed to load bounded live process nonterminal orders")?;
        rows.into_iter().map(order_from_db_row).collect()
    }

    /// Returns unresolved orders for every live trading-process sleeve configured for one account.
    /// Process status is deliberately not part of ownership: stopped sleeves retain their orders,
    /// fills and positions until their financial state is terminal.
    pub async fn live_account_nonterminal_orders(
        &self,
        account_ref: &str,
        limit: i64,
    ) -> Result<Vec<OrderRecord>> {
        let account_ref = account_ref.trim();
        if account_ref.is_empty() || account_ref.len() > 128 {
            bail!("live account order lookup requires a bounded account_ref");
        }
        let rows = sqlx::query_as::<_, OrderDbRow>(
            r#"
            SELECT orders.order_id, orders.state, orders.raw_payload,
                   orders.created_at, orders.updated_at
            FROM polymarket.orders orders
            JOIN polymarket.trading_processes process
              ON process.process_id = orders.process_id
            WHERE process.config #>> '{execution,mode}' = 'live'
              AND lower(btrim(process.config #>> '{execution,account_ref}')) = lower($1)
              AND orders.state IN (
                'created', 'submitted', 'acknowledged', 'partially_filled',
                'cancel_requested', 'unknown'
              )
            ORDER BY orders.created_at, orders.order_id
            LIMIT $2
            "#,
        )
        .bind(account_ref)
        .bind(limit.clamp(1, 4_097))
        .fetch_all(&self.pool)
        .await
        .context("failed to load bounded live account nonterminal orders")?;
        rows.into_iter().map(order_from_db_row).collect()
    }

    /// Replays all bounded local evidence that can consume live capital for exactly one process.
    ///
    /// Pending/unknown/nonterminal orders reserve their full requested notional and deterministic
    /// dynamic fee. Every historical live BUY fill retains its actual notional and fee until the
    /// existing settlement ledger contains exact credited exchange-redemption evidence. This does
    /// not infer release from a SELL or market resolution alone.
    pub async fn conservative_live_process_exposure(
        &self,
        process_id: Uuid,
        ignored_client_order_id: Option<Uuid>,
    ) -> Result<LiveProcessExposureSnapshot> {
        if process_id.is_nil() {
            bail!("live exposure evidence requires a non-nil process_id");
        }
        let order_limit = (MAX_LIVE_PROCESS_EXPOSURE_ORDERS + 1) as i64;
        let fill_limit = (MAX_LIVE_PROCESS_EXPOSURE_FILLS + 1) as i64;

        // Read reservations before fills. The live writer persists a fill before advancing its
        // order state, so this ordering cannot observe neither side of a concurrent transition:
        // either the full nonterminal reservation is present, or the subsequent fill replay is.
        let orders =
            sqlx::query_as::<_, LiveExposureOrderRow>(SELECT_LIVE_PROCESS_EXPOSURE_ORDERS_SQL)
                .bind(process_id)
                .bind(order_limit)
                .fetch_all(&self.pool)
                .await
                .context("failed to load bounded live process exposure orders")?;
        let fills =
            sqlx::query_as::<_, LiveExposureFillRow>(SELECT_LIVE_PROCESS_EXPOSURE_FILLS_SQL)
                .bind(process_id)
                .bind(fill_limit)
                .fetch_all(&self.pool)
                .await
                .context("failed to load bounded live process exposure fills")?;
        let cross_owned_fill =
            sqlx::query_scalar::<_, bool>(SELECT_LIVE_PROCESS_CROSS_OWNED_FILL_EXISTS_SQL)
                .bind(process_id)
                .fetch_one(&self.pool)
                .await
                .context("failed to inspect cross-owned live process fills")?;
        let unproven_filled_order =
            sqlx::query_scalar::<_, bool>(SELECT_LIVE_PROCESS_UNPROVEN_FILLED_ORDER_EXISTS_SQL)
                .bind(process_id)
                .fetch_one(&self.pool)
                .await
                .context("failed to inspect unproven live process fill states")?;
        let has_unredeemed_settlement =
            sqlx::query_scalar::<_, bool>(SELECT_PENDING_LIVE_PROCESS_SETTLEMENT_EXISTS_SQL)
                .bind(process_id)
                .fetch_one(&self.pool)
                .await
                .context("failed to inspect pending live process redemption evidence")?;
        let as_of = sqlx::query_scalar::<_, DateTime<Utc>>("SELECT clock_timestamp()")
            .fetch_one(&self.pool)
            .await
            .context("failed to read database clock for live process exposure")?;

        if orders.len() > MAX_LIVE_PROCESS_EXPOSURE_ORDERS {
            bail!(
                "live process {} exceeds the bounded {}-nonterminal-order exposure window",
                process_id,
                MAX_LIVE_PROCESS_EXPOSURE_ORDERS
            );
        }
        if fills.len() > MAX_LIVE_PROCESS_EXPOSURE_FILLS {
            bail!(
                "live process {} exceeds the bounded {}-fill exposure window",
                process_id,
                MAX_LIVE_PROCESS_EXPOSURE_FILLS
            );
        }
        if cross_owned_fill {
            bail!("live process exposure contains cross-process fill ownership");
        }
        if unproven_filled_order {
            bail!("live process has a filled state without exact persisted live fill evidence");
        }

        build_live_process_exposure_snapshot(
            process_id,
            ignored_client_order_id,
            as_of,
            has_unredeemed_settlement,
            &orders,
            &fills,
        )
    }

    /// Computes the same deterministic full reservation for an adjacent incoming request that the
    /// durable exposure replay applies to persisted nonterminal orders.
    pub fn conservative_live_request_exposure(
        request: &OrderRequest,
    ) -> Result<LiveRequestedExposure> {
        let (requested_notional_usd, requested_fees_usd) = live_requested_exposure(request)?;
        let total_exposure_usd = checked_live_exposure_add(
            requested_notional_usd,
            requested_fees_usd,
            "incoming requested exposure",
        )?;
        Ok(LiveRequestedExposure {
            requested_notional_usd,
            requested_fees_usd,
            total_exposure_usd,
        })
    }

    /// Returns recognized live settlement PnL for exactly one process and UTC accounting day.
    /// The evidence window is deliberately bounded and independently replays the persisted live
    /// fills. Any pending, malformed, cross-process, or internally inconsistent evidence fails
    /// closed instead of understating the daily loss used by the submit gate.
    pub async fn recognized_live_process_net_pnl_for_utc_day(
        &self,
        process_id: Uuid,
        day_start: DateTime<Utc>,
        day_end: DateTime<Utc>,
        as_of: DateTime<Utc>,
    ) -> Result<Decimal> {
        if process_id.is_nil() {
            bail!("live daily loss evidence requires a non-nil process_id");
        }
        if day_start >= day_end || as_of < day_start || as_of >= day_end {
            bail!("live daily loss evidence requires a valid current UTC-day interval");
        }
        if day_start.time() != chrono::NaiveTime::MIN
            || day_end != day_start + chrono::Duration::days(1)
        {
            bail!("live daily loss evidence interval must be exactly one UTC day");
        }

        let pending =
            sqlx::query_scalar::<_, bool>(SELECT_PENDING_LIVE_PROCESS_SETTLEMENT_EXISTS_SQL)
                .bind(process_id)
                .fetch_one(&self.pool)
                .await
                .context("failed to inspect pending live process settlements")?;
        if pending {
            bail!(
                "live process {} has pending settlement evidence; daily loss is ambiguous",
                process_id
            );
        }

        let settlements = sqlx::query_as::<_, LiveDailySettlementRow>(
            SELECT_CREDITED_LIVE_PROCESS_SETTLEMENTS_SQL,
        )
        .bind(process_id)
        .bind(day_start)
        .bind(day_end)
        .bind(MAX_LIVE_DAILY_SETTLEMENTS + 1)
        .fetch_all(&self.pool)
        .await
        .context("failed to load credited live process settlements for UTC day")?;
        let validation_as_of = sqlx::query_scalar::<_, DateTime<Utc>>("SELECT clock_timestamp()")
            .fetch_one(&self.pool)
            .await
            .context("failed to read database clock for live daily loss evidence")?;
        if settlements.len() > MAX_LIVE_DAILY_SETTLEMENTS as usize {
            bail!(
                "live process {} exceeds the bounded {}-settlement daily loss window",
                process_id,
                MAX_LIVE_DAILY_SETTLEMENTS
            );
        }

        let mut settlement_ids = HashSet::with_capacity(settlements.len());
        let mut order_ids = HashSet::with_capacity(settlements.len());
        let mut all_fill_ids = Vec::new();
        let mut settlement_fill_ids = HashMap::with_capacity(settlements.len());
        for settlement in &settlements {
            if settlement.process_id != process_id {
                bail!("live daily loss evidence crossed process ownership");
            }
            if !settlement_ids.insert(settlement.settlement_id) {
                bail!("live daily loss evidence contains a duplicate settlement identity");
            }
            if !order_ids.insert(settlement.order_id.clone()) {
                bail!("live daily loss evidence contains duplicate settlement order ownership");
            }
            let fill_ids = parse_bounded_live_settlement_fill_ids(&settlement.fill_ids)?;
            if all_fill_ids.len().saturating_add(fill_ids.len()) > MAX_LIVE_DAILY_SETTLEMENT_FILLS {
                bail!(
                    "live process {} exceeds the bounded {}-fill daily loss window",
                    process_id,
                    MAX_LIVE_DAILY_SETTLEMENT_FILLS
                );
            }
            all_fill_ids.extend(fill_ids.iter().copied());
            settlement_fill_ids.insert(settlement.settlement_id, fill_ids);
            validate_live_daily_settlement(settlement, day_start, day_end, validation_as_of)?;
        }

        let unique_fill_ids = all_fill_ids.iter().copied().collect::<HashSet<_>>();
        if unique_fill_ids.len() != all_fill_ids.len() {
            bail!("live daily loss evidence reuses a fill across settlements");
        }
        let fill_rows = if all_fill_ids.is_empty() {
            Vec::new()
        } else {
            sqlx::query_as::<_, LiveSettlementFillRow>(SELECT_LIVE_SETTLEMENT_FILLS_SQL)
                .bind(&all_fill_ids)
                .fetch_all(&self.pool)
                .await
                .context("failed to load exact live settlement fill lineage")?
        };
        if fill_rows.len() != all_fill_ids.len() {
            bail!("live daily loss evidence has missing or duplicate persisted fill lineage");
        }
        let fills_by_id = fill_rows
            .into_iter()
            .map(|fill| (fill.fill_id, fill))
            .collect::<HashMap<_, _>>();
        if fills_by_id.len() != all_fill_ids.len() {
            bail!("live daily loss evidence has ambiguous persisted fill identities");
        }

        let mut net_pnl = Decimal::ZERO;
        for settlement in &settlements {
            let fill_ids = settlement_fill_ids
                .get(&settlement.settlement_id)
                .context("live settlement fill identity map is incomplete")?;
            let mut filled_size = Decimal::ZERO;
            let mut entry_notional = Decimal::ZERO;
            let mut entry_fees = Decimal::ZERO;
            for fill_id in fill_ids {
                let fill = fills_by_id
                    .get(fill_id)
                    .context("live settlement references an unknown persisted fill")?;
                if fill.identity_process_id != process_id
                    || fill.fill_process_id != Some(process_id)
                    || fill.order_id != settlement.order_id
                    || fill.token_id != settlement.token_id
                    || fill.source != "live"
                {
                    bail!(
                        "live settlement fill lineage does not match exact process/order ownership"
                    );
                }
                if fill.price <= Decimal::ZERO
                    || fill.size <= Decimal::ZERO
                    || fill.fee < Decimal::ZERO
                {
                    bail!("live settlement fill lineage contains invalid economics");
                }
                filled_size += fill.size;
                entry_notional += fill.price * fill.size;
                entry_fees += fill.fee;
            }
            if filled_size != settlement.filled_size
                || entry_notional != settlement.entry_notional
                || entry_fees != settlement.entry_fees
            {
                bail!("live settlement economics do not replay from exact persisted fills");
            }
            net_pnl += settlement.net_pnl;
        }

        let account_exits = sqlx::query_as::<_, LiveDailyAccountExitRow>(
            r#"
            SELECT
              account_exit.account_trade_id,
              orders.process_id,
              orders.order_id,
              orders.token_id,
              account_exit.applied_exit_size AS exit_size,
              (account_exit.raw_payload #>> '{reconciliation,net_proceeds}')::numeric
                AS net_proceeds,
              round(SUM(fill.size), 10)::numeric AS entry_filled_size,
              round(SUM(fill.price * fill.size), 10)::numeric AS entry_notional,
              round(SUM(fill.fee), 10)::numeric AS entry_fees
            FROM polymarket.account_trades account_exit
            JOIN polymarket.orders orders ON orders.order_id = account_exit.linked_order_id
            JOIN polymarket.fills fill
              ON fill.process_id = orders.process_id
             AND fill.order_id = orders.order_id
             AND fill.source = 'live'
            WHERE orders.process_id = $1
              AND account_exit.side = 'sell'
              AND account_exit.applied_exit_size = account_exit.size
              AND account_exit.timestamp_utc >= $2
              AND account_exit.timestamp_utc < $3
            GROUP BY account_exit.account_trade_id, orders.process_id, orders.order_id,
              orders.token_id, account_exit.applied_exit_size,
              account_exit.raw_payload #>> '{reconciliation,net_proceeds}'
            ORDER BY account_exit.timestamp_utc, account_exit.account_trade_id
            LIMIT $4
            "#,
        )
        .bind(process_id)
        .bind(day_start)
        .bind(day_end)
        .bind(MAX_LIVE_DAILY_SETTLEMENTS + 1)
        .fetch_all(&self.pool)
        .await
        .context("failed to load recognized manual live exits for UTC day")?;
        if account_exits.len() > MAX_LIVE_DAILY_SETTLEMENTS as usize {
            bail!(
                "live process {} exceeds the bounded {}-manual-exit daily loss window",
                process_id,
                MAX_LIVE_DAILY_SETTLEMENTS
            );
        }
        let mut account_trade_ids = HashSet::with_capacity(account_exits.len());
        for account_exit in account_exits {
            if account_exit.process_id != process_id
                || account_exit.order_id.trim().is_empty()
                || account_exit.token_id.trim().is_empty()
                || account_exit.exit_size <= Decimal::ZERO
                || account_exit.exit_size != account_exit.entry_filled_size
                || account_exit.net_proceeds <= Decimal::ZERO
                || account_exit.entry_notional <= Decimal::ZERO
                || account_exit.entry_fees < Decimal::ZERO
                || !account_trade_ids.insert(account_exit.account_trade_id)
            {
                bail!("manual live exit daily PnL evidence is invalid or ambiguous");
            }
            net_pnl +=
                account_exit.net_proceeds - account_exit.entry_notional - account_exit.entry_fees;
        }
        Ok(net_pnl)
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

    pub async fn find_fill_order_id(&self, fill_id: Uuid) -> Result<Option<String>> {
        sqlx::query_scalar::<_, String>(
            r#"
            SELECT fill.order_id
            FROM polymarket.fills fill
            JOIN polymarket.fill_identities identity
              ON identity.fill_id = fill.fill_id
             AND identity.timestamp_utc = fill.timestamp_utc
            WHERE fill.fill_id = $1
            LIMIT 1
            "#,
        )
        .bind(fill_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to resolve canonical fill order identity")
    }

    /// Resolves explicit venue order identities to orders owned by one trading process. The input
    /// is intentionally bounded to the maximum account reconciliation page, and the lookup uses
    /// the unique partial venue-order index instead of token or market inference.
    pub async fn process_order_ids_by_venue_order_ids(
        &self,
        process_id: Uuid,
        venue_order_ids: &[String],
    ) -> Result<HashMap<String, String>> {
        const MAX_RECONCILIATION_ORDER_IDS: usize = 500;

        if process_id.is_nil() {
            bail!("process order ownership lookup requires a non-nil process_id");
        }
        if venue_order_ids.len() > MAX_RECONCILIATION_ORDER_IDS {
            bail!(
                "process order ownership lookup exceeds the bounded {}-identity window",
                MAX_RECONCILIATION_ORDER_IDS
            );
        }
        if venue_order_ids.is_empty() {
            return Ok(HashMap::new());
        }

        #[derive(FromRow)]
        struct OwnedOrderRow {
            venue_order_id: String,
            order_id: String,
        }

        let rows = sqlx::query_as::<_, OwnedOrderRow>(
            r#"
            SELECT venue_order_id, order_id
            FROM polymarket.orders
            WHERE process_id = $1
              AND venue_order_id = ANY($2::text[])
            LIMIT 500
            "#,
        )
        .bind(process_id)
        .bind(venue_order_ids)
        .fetch_all(&self.pool)
        .await
        .context("failed to resolve process-owned venue order identities")?;

        Ok(rows
            .into_iter()
            .map(|row| (row.venue_order_id, row.order_id))
            .collect())
    }

    /// Resolves exact venue identities across all process sleeves assigned to one live account.
    pub async fn account_order_ids_by_venue_order_ids(
        &self,
        account_ref: &str,
        venue_order_ids: &[String],
    ) -> Result<HashMap<String, String>> {
        const MAX_RECONCILIATION_ORDER_IDS: usize = 500;
        let account_ref = account_ref.trim();
        if account_ref.is_empty() || account_ref.len() > 128 {
            bail!("account order ownership lookup requires a bounded account_ref");
        }
        if venue_order_ids.len() > MAX_RECONCILIATION_ORDER_IDS {
            bail!(
                "account order ownership lookup exceeds the bounded {}-identity window",
                MAX_RECONCILIATION_ORDER_IDS
            );
        }
        if venue_order_ids.is_empty() {
            return Ok(HashMap::new());
        }

        #[derive(FromRow)]
        struct OwnedOrderRow {
            venue_order_id: String,
            order_id: String,
        }

        let rows = sqlx::query_as::<_, OwnedOrderRow>(
            r#"
            SELECT orders.venue_order_id, orders.order_id
            FROM polymarket.orders orders
            JOIN polymarket.trading_processes process
              ON process.process_id = orders.process_id
            WHERE process.config #>> '{execution,mode}' = 'live'
              AND lower(btrim(process.config #>> '{execution,account_ref}')) = lower($1)
              AND orders.venue_order_id = ANY($2::text[])
            LIMIT 500
            "#,
        )
        .bind(account_ref)
        .bind(venue_order_ids)
        .fetch_all(&self.pool)
        .await
        .context("failed to resolve account-owned venue order identities")?;

        Ok(rows
            .into_iter()
            .map(|row| (row.venue_order_id, row.order_id))
            .collect())
    }

    /// Resolves transaction identities only through already-persisted live fills owned by one
    /// configured account. The fill timestamp bound preserves hypertable pruning and prevents an
    /// account reconciliation from searching historical execution data without limit.
    pub(crate) async fn account_live_fill_evidence_by_transaction_hashes(
        &self,
        account_ref: &str,
        transaction_hashes: &[String],
        window_start: DateTime<Utc>,
        window_end: DateTime<Utc>,
    ) -> Result<Vec<AccountLiveFillEvidence>> {
        const MAX_RECONCILIATION_TRANSACTION_HASHES: usize = 500;
        const MAX_RECONCILIATION_FILL_EVIDENCE: usize = 4_000;
        let account_ref = account_ref.trim();
        if account_ref.is_empty() || account_ref.len() > 128 {
            bail!("account fill ownership lookup requires a bounded account_ref");
        }
        if transaction_hashes.len() > MAX_RECONCILIATION_TRANSACTION_HASHES {
            bail!(
                "account fill ownership lookup exceeds the bounded {}-identity window",
                MAX_RECONCILIATION_TRANSACTION_HASHES
            );
        }
        if transaction_hashes.is_empty() {
            return Ok(Vec::new());
        }
        if window_start > window_end {
            bail!("account fill ownership lookup has an invalid time window");
        }

        let rows = sqlx::query_as::<_, AccountLiveFillEvidence>(
            r#"
            SELECT
              lower(btrim(orders.raw_payload #>> '{venue,raw_payload,transaction_hash}')) AS transaction_hash,
              fills.order_id,
              fills.token_id,
              lower(orders.side) AS side,
              fills.price,
              fills.size
            FROM polymarket.fills fills
            JOIN polymarket.orders orders
              ON orders.order_id = fills.order_id
             AND orders.process_id = fills.process_id
            JOIN polymarket.trading_processes process
              ON process.process_id = fills.process_id
            WHERE fills.source = 'live'
              AND fills.timestamp_utc >= $3
              AND fills.timestamp_utc <= $4
              AND process.config #>> '{execution,mode}' = 'live'
              AND lower(btrim(process.config #>> '{execution,account_ref}')) = lower($1)
              AND lower(btrim(orders.raw_payload #>> '{venue,raw_payload,transaction_hash}')) = ANY($2::text[])
            ORDER BY fills.timestamp_utc, fills.fill_id
            LIMIT 4001
            "#,
        )
        .bind(account_ref)
        .bind(transaction_hashes)
        .bind(window_start)
        .bind(window_end)
        .fetch_all(&self.pool)
        .await
        .context("failed to resolve account-owned live fill transaction identities")?;
        if rows.len() > MAX_RECONCILIATION_FILL_EVIDENCE {
            bail!(
                "account fill ownership lookup exceeds the bounded {}-fill evidence window",
                MAX_RECONCILIATION_FILL_EVIDENCE
            );
        }
        Ok(rows)
    }

    pub async fn oldest_unrecognized_live_fill_at(
        &self,
        process_id: Uuid,
    ) -> Result<Option<DateTime<Utc>>> {
        if process_id.is_nil() {
            bail!("live redemption discovery requires a non-nil process_id");
        }
        sqlx::query_scalar::<_, Option<DateTime<Utc>>>(
            r#"
            SELECT MIN(fill.timestamp_utc)
            FROM polymarket.fills fill
            JOIN polymarket.orders orders ON orders.order_id = fill.order_id
            WHERE fill.process_id = $1
              AND orders.process_id = $1
              AND fill.source = 'live'
              AND orders.side = 'buy'
              AND NOT EXISTS (
                SELECT 1
                FROM polymarket.btc_paper_settlement_ledger settlement
                WHERE settlement.process_id = $1
                  AND settlement.order_id = fill.order_id
                  AND settlement.execution_mode = 'live'
                  AND settlement.credit_status = 'credited'
              )
              AND NOT EXISTS (
                SELECT 1
                FROM polymarket.account_trades account_exit
                WHERE account_exit.linked_order_id = orders.order_id
                  AND account_exit.side = 'sell'
                  AND account_exit.applied_exit_size > 0
              )
            "#,
        )
        .bind(process_id)
        .fetch_one(&self.pool)
        .await
        .context("failed to load oldest unrecognized live fill")
    }

    /// Returns bounded process-owned resolved live fill sizes that are not covered by an exact
    /// credited settlement. These rows prevent realized PnL proof until settlement recovery
    /// completes, regardless of which runtime generation originally owned the order.
    pub async fn unsettled_resolved_live_position_sizes(
        &self,
        process_id: Uuid,
    ) -> Result<Vec<(String, Decimal)>> {
        const MAX_UNSETTLED_RESOLVED_ORDERS: usize = 4_000;
        if process_id.is_nil() {
            bail!("resolved live settlement proof requires a non-nil process_id");
        }
        let rows = sqlx::query_as::<_, (String, Decimal)>(
            r#"
            SELECT orders.token_id, round(SUM(fill.size), 10)::numeric AS filled_size
            FROM polymarket.orders orders
            JOIN polymarket.fills fill
              ON fill.order_id = orders.order_id
             AND fill.process_id = orders.process_id
             AND fill.source = 'live'
            JOIN polymarket.btc_interval_markets market ON market.market_id = orders.market_id
            JOIN polymarket.btc_official_resolution_watches watch
              ON watch.market_id = market.market_id
             AND watch.status IN ('resolved', 'resolved_late')
             AND watch.resolution_received_at = market.official_resolution_received_at
             AND watch.resolution_source = market.official_resolution_source
            WHERE orders.process_id = $1
              AND market.official_outcome IS NOT NULL
              AND market.official_winning_token_id IS NOT NULL
              AND NOT EXISTS (
                SELECT 1
                FROM polymarket.btc_paper_settlement_ledger settlement
                WHERE settlement.process_id = $1
                  AND settlement.order_id = orders.order_id
                  AND settlement.token_id = orders.token_id
                  AND settlement.execution_mode = 'live'
                  AND settlement.credit_status = 'credited'
                  AND settlement.filled_size = (
                    SELECT round(SUM(expected_fill.size), 10)::numeric
                    FROM polymarket.fills expected_fill
                    WHERE expected_fill.process_id = $1
                      AND expected_fill.order_id = orders.order_id
                      AND expected_fill.source = 'live'
                  )
                  AND jsonb_array_length(settlement.fill_ids) = (
                    SELECT COUNT(*)::integer
                    FROM polymarket.fills expected_fill
                    WHERE expected_fill.process_id = $1
                      AND expected_fill.order_id = orders.order_id
                      AND expected_fill.source = 'live'
                  )
                  AND NOT EXISTS (
                    SELECT 1
                    FROM polymarket.fills expected_fill
                    WHERE expected_fill.process_id = $1
                      AND expected_fill.order_id = orders.order_id
                      AND expected_fill.source = 'live'
                      AND NOT settlement.fill_ids @>
                        jsonb_build_array(to_jsonb(expected_fill.fill_id))
                  )
              )
              AND NOT EXISTS (
                SELECT 1
                FROM polymarket.account_trades account_exit
                WHERE account_exit.linked_order_id = orders.order_id
                  AND account_exit.side = 'sell'
                  AND account_exit.applied_exit_size > 0
              )
            GROUP BY orders.order_id, orders.token_id
            ORDER BY orders.order_id
            LIMIT 4001
            "#,
        )
        .bind(process_id)
        .fetch_all(&self.pool)
        .await
        .context("failed to prove resolved live fill settlement coverage")?;
        if rows.len() > MAX_UNSETTLED_RESOLVED_ORDERS {
            bail!(
                "resolved live settlement proof exceeds the bounded {}-order window",
                MAX_UNSETTLED_RESOLVED_ORDERS
            );
        }
        Ok(rows)
    }

    /// Reconciles one exact, full manual wallet exit to a unique account-owned live entry.
    /// Partial or ambiguous exits remain unmatched so custody and realized PnL stay fail-closed.
    pub async fn recognize_account_live_exit(
        &self,
        account_ref: &str,
        trade: &AccountTrade,
        apply: bool,
    ) -> Result<LiveAccountExitRecognition> {
        let exit_fee_tolerance = Decimal::new(1, 5);
        let account_ref = account_ref.trim();
        if account_ref.is_empty() || account_ref.len() > 128 {
            bail!("manual live exit recognition requires a bounded account_ref");
        }
        if trade.side != "sell"
            || trade.account_address.trim().is_empty()
            || trade.token_id.trim().is_empty()
            || trade.price <= Decimal::ZERO
            || trade.price >= Decimal::ONE
            || trade.size <= Decimal::ZERO
            || trade.transaction_hash.as_deref().is_none_or(str::is_empty)
        {
            bail!("manual live exit recognition requires complete sell evidence");
        }
        let net_proceeds = trade
            .raw_payload
            .get("usdcSize")
            .and_then(crate::fees::decimal_from_json)
            .context("manual live exit is missing Data API net proceeds")?;
        if net_proceeds <= Decimal::ZERO || net_proceeds > trade.notional {
            bail!("manual live exit has invalid Data API net proceeds");
        }

        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin manual live exit reconciliation")?;
        sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
            .bind(format!("live-account-exit:{}", trade.account_trade_id))
            .execute(&mut *tx)
            .await
            .context("failed to lock manual live exit reconciliation")?;

        if let Some((order_id, applied_size)) = sqlx::query_as::<_, (Option<String>, Decimal)>(
            r#"
                SELECT linked_order_id, applied_exit_size
                FROM polymarket.account_trades
                WHERE account_trade_id = $1
                "#,
        )
        .bind(trade.account_trade_id)
        .fetch_optional(&mut *tx)
        .await
        .context("failed to inspect existing manual live exit recognition")?
        {
            if order_id.is_some() && applied_size == trade.size {
                tx.commit()
                    .await
                    .context("failed to commit idempotent manual live exit reconciliation")?;
                return Ok(LiveAccountExitRecognition {
                    matched: true,
                    applied: false,
                    order_id,
                    exited_size: trade.size,
                });
            }
        }

        let candidates = sqlx::query_as::<_, LiveAccountExitCandidateRow>(
            r#"
            WITH account_processes AS (
              SELECT process_id
              FROM polymarket.trading_processes
              WHERE config #>> '{execution,mode}' = 'live'
                AND lower(btrim(config #>> '{execution,account_ref}')) = lower($1)
            ), entries AS (
              SELECT
                orders.process_id,
                orders.order_id,
                round(SUM(fill.size), 10)::numeric AS filled_size,
                round(SUM(fill.price * fill.size), 10)::numeric AS entry_notional,
                round(SUM(fill.fee), 10)::numeric AS entry_fees,
                orders.raw_payload AS order_payload,
                COALESCE((
                  SELECT SUM(existing_exit.applied_exit_size)
                  FROM polymarket.account_trades existing_exit
                  WHERE existing_exit.linked_order_id = orders.order_id
                    AND existing_exit.side = 'sell'
                ), 0)::numeric AS already_exited_size
              FROM polymarket.orders orders
              JOIN account_processes process ON process.process_id = orders.process_id
              JOIN polymarket.fills fill
                ON fill.process_id = orders.process_id
               AND fill.order_id = orders.order_id
               AND fill.source = 'live'
              WHERE orders.side = 'buy'
                AND orders.token_id = $2
                AND fill.timestamp_utc <= $3
                AND NOT EXISTS (
                  SELECT 1
                  FROM polymarket.btc_paper_settlement_ledger settlement
                  WHERE settlement.process_id = orders.process_id
                    AND settlement.order_id = orders.order_id
                    AND settlement.execution_mode = 'live'
                    AND settlement.credit_status = 'credited'
                )
              GROUP BY orders.process_id, orders.order_id, orders.raw_payload
            )
            SELECT process_id, order_id, filled_size, entry_notional, entry_fees, order_payload
            FROM entries
            WHERE filled_size - already_exited_size = $4
            ORDER BY order_id
            LIMIT 2
            "#,
        )
        .bind(account_ref)
        .bind(&trade.token_id)
        .bind(trade.timestamp_utc + LIVE_EXTERNAL_EVENT_CLOCK_SKEW)
        .bind(trade.size)
        .fetch_all(&mut *tx)
        .await
        .context("failed to resolve manual live exit ownership")?;
        if candidates.len() != 1 {
            tx.commit()
                .await
                .context("failed to commit unmatched manual live exit reconciliation")?;
            return Ok(LiveAccountExitRecognition {
                matched: false,
                applied: false,
                order_id: None,
                exited_size: Decimal::ZERO,
            });
        }
        let candidate = &candidates[0];
        if candidate.filled_size != trade.size {
            bail!("manual live exit does not exactly close its unique entry");
        }
        let fee_rate = sealed_dynamic_fee_rate(
            candidate
                .order_payload
                .get("request")
                .and_then(|request| request.get("metadata"))
                .context("manual live exit entry is missing sealed metadata")?,
        )?;
        let (expected_exit_fee, observed_exit_fee, net_pnl) = manual_live_exit_economics(
            trade.size,
            trade.price,
            fee_rate,
            net_proceeds,
            candidate.entry_notional,
            candidate.entry_fees,
            exit_fee_tolerance,
        )?;
        if apply {
            sqlx::query(
                r#"
                DELETE FROM polymarket.btc_paper_settlement_ledger
                WHERE process_id = $1
                  AND order_id = $2
                  AND execution_mode = 'live'
                  AND credit_status = 'pending'
                  AND official_resolution_received_at > $3
                "#,
            )
            .bind(candidate.process_id)
            .bind(&candidate.order_id)
            .bind(trade.timestamp_utc)
            .execute(&mut *tx)
            .await
            .context("failed to remove superseded pending live settlement")?;
            let updated = sqlx::query(
                r#"
                UPDATE polymarket.account_trades
                SET linked_order_id = $2,
                    applied_exit_size = size,
                    raw_payload = raw_payload || jsonb_build_object(
                      'reconciliation', jsonb_build_object(
                        'kind', 'manual_live_full_exit',
                        'process_id', $3::text,
                        'order_id', $2,
                        'net_proceeds', $4::text,
                        'observed_exit_fee', $5::text,
                        'expected_exit_fee', $6::text,
                        'entry_notional', $7::text,
                        'entry_fees', $8::text,
                        'net_pnl', $9::text
                      )
                    ),
                    updated_at = now()
                WHERE account_trade_id = $1
                  AND linked_order_id IS NULL
                  AND applied_exit_size = 0
                "#,
            )
            .bind(trade.account_trade_id)
            .bind(&candidate.order_id)
            .bind(candidate.process_id)
            .bind(net_proceeds)
            .bind(observed_exit_fee)
            .bind(expected_exit_fee)
            .bind(candidate.entry_notional)
            .bind(candidate.entry_fees)
            .bind(net_pnl)
            .execute(&mut *tx)
            .await
            .context("failed to persist manual live exit recognition")?;
            if updated.rows_affected() != 1 {
                bail!("manual live exit recognition lost its atomic ownership claim");
            }
        }
        tx.commit()
            .await
            .context("failed to commit manual live exit reconciliation")?;
        Ok(LiveAccountExitRecognition {
            matched: true,
            applied: apply,
            order_id: Some(candidate.order_id.clone()),
            exited_size: trade.size,
        })
    }

    pub async fn recognize_process_live_redemption(
        &self,
        process_id: Uuid,
        account_ref: &str,
        evidence: &LiveRedemptionEvidence,
        apply: bool,
    ) -> Result<LiveRedemptionRecognition> {
        if process_id.is_nil() {
            bail!("live redemption recognition requires a non-nil process_id");
        }
        let account_ref = account_ref.trim();
        if account_ref.is_empty() || account_ref.len() > 128 {
            bail!("live redemption recognition requires a bounded account_ref");
        }
        if evidence.account_address.trim().is_empty()
            || evidence.condition_id.trim().is_empty()
            || evidence.token_id.trim().is_empty()
            || evidence.transaction_hash.trim().is_empty()
            || evidence.redeemed_size <= Decimal::ZERO
            || (evidence.payout_usd != Decimal::ZERO
                && evidence.payout_usd != evidence.redeemed_size)
            || !evidence.raw_payload.is_object()
        {
            bail!("live redemption recognition requires complete exact payout evidence");
        }
        let credit_evidence = serde_json::json!({
            "proof_type": "polymarket_data_api_redeem",
            "account_address": evidence.account_address,
            "condition_id": evidence.condition_id,
            "token_id": evidence.token_id,
            "redeemed_size": evidence.redeemed_size,
            "payout_usd": evidence.payout_usd,
            "redeemed_at": evidence.redeemed_at,
            "redemption_transaction_hash": evidence.transaction_hash,
            "activity": evidence.raw_payload,
        });
        let row = sqlx::query_as::<_, LiveRedemptionRecognitionRow>(
            r#"
            WITH lock AS (
              SELECT pg_advisory_xact_lock(
                hashtextextended('live-redemption:' || $8, 0)
              )
            ), owners AS (
              SELECT
                COUNT(DISTINCT orders.process_id)::bigint AS owner_count,
                COALESCE(round(SUM(fill.size), 10), 0)::numeric(30,10) AS owner_filled_size
              FROM polymarket.fills fill
              JOIN polymarket.orders orders ON orders.order_id = fill.order_id
              JOIN polymarket.trading_processes process
                ON process.process_id = orders.process_id
              JOIN polymarket.btc_interval_markets market
                ON market.market_id = orders.market_id
              CROSS JOIN lock
              WHERE fill.source = 'live'
                AND orders.side = 'buy'
                AND fill.token_id = $3
                AND market.condition_id = $2
                AND process.config #>> '{execution,mode}' = 'live'
                AND lower(btrim(process.config #>> '{execution,account_ref}')) = lower($9)
            ), candidate AS (
              SELECT
                orders.process_id,
                COALESCE(
                  NULLIF(orders.raw_payload #>> '{request,metadata,run_id}', ''),
                  NULLIF(orders.raw_payload #>> '{request,metadata,experiment_id}', '')
                )::uuid AS run_id,
                orders.order_id,
                orders.market_id,
                orders.token_id,
                jsonb_agg(to_jsonb(fill.fill_id) ORDER BY fill.timestamp_utc, fill.fill_id)
                  AS fill_ids,
                round(SUM(fill.size), 10)::numeric(30,10) AS filled_size,
                round(SUM(fill.price * fill.size), 10)::numeric(30,10) AS entry_notional,
                round(SUM(fill.fee), 10)::numeric(30,10) AS entry_fees,
                market.official_outcome,
                market.official_winning_token_id,
                market.official_resolution_received_at,
                market.official_resolution_source
              FROM polymarket.orders orders
              JOIN polymarket.trading_processes process
                ON process.process_id = orders.process_id
              JOIN polymarket.fills fill
                ON fill.order_id = orders.order_id
               AND fill.process_id = orders.process_id
               AND fill.source = 'live'
              JOIN polymarket.btc_interval_markets market
                ON market.market_id = orders.market_id
              JOIN polymarket.btc_official_resolution_watches watch
                ON watch.market_id = market.market_id
               AND watch.status IN ('resolved', 'resolved_late')
               AND watch.resolution_received_at = market.official_resolution_received_at
               AND watch.resolution_source = market.official_resolution_source
              WHERE orders.process_id = $1
                AND process.config #>> '{execution,mode}' = 'live'
                AND lower(btrim(process.config #>> '{execution,account_ref}')) = lower($9)
                AND orders.side = 'buy'
                AND orders.token_id = $3
                AND market.condition_id = $2
                AND (
                  ($10 = $4 AND market.official_winning_token_id = $3)
                  OR ($10 = 0 AND market.official_winning_token_id <> $3)
                )
                AND market.official_outcome IN ('up', 'down')
                AND market.resolution_source_timestamp <= $5
              GROUP BY
                orders.process_id, orders.order_id, orders.market_id, orders.token_id,
                market.official_outcome, market.official_winning_token_id,
                market.official_resolution_received_at, market.official_resolution_source
              HAVING MAX(fill.timestamp_utc) <= $5
                AND round(SUM(fill.size), 10) = $4
            ), candidate_state AS (
              SELECT COUNT(*)::bigint AS candidate_count FROM candidate
            ), transaction_state AS (
              SELECT
                COUNT(*)::bigint AS transaction_count,
                COALESCE(
                  BOOL_AND(
                    settlement.token_id = $3
                    AND settlement.payout = $10
                  ),
                  true
                )
                AND COALESCE(SUM(settlement.payout), 0) <= $10 AS transaction_consistent
              FROM polymarket.btc_paper_settlement_ledger settlement
              WHERE settlement.execution_mode = 'live'
                AND settlement.credit_status = 'credited'
                AND settlement.credit_evidence ->> 'redemption_transaction_hash' = $8
            ), updated AS (
              UPDATE polymarket.btc_paper_settlement_ledger settlement
              SET credit_status = 'credited',
                  credited_at = now(),
                  credit_attempts = credit_attempts + 1,
                  credit_evidence = $6,
                  updated_at = now()
              FROM candidate
              CROSS JOIN owners
              CROSS JOIN candidate_state
              CROSS JOIN transaction_state
              WHERE $7
                AND owners.owner_count >= 1
                AND owners.owner_filled_size = $4
                AND candidate_state.candidate_count = 1
                AND transaction_state.transaction_consistent
                AND settlement.process_id = candidate.process_id
                AND settlement.run_id = candidate.run_id
                AND settlement.order_id = candidate.order_id
                AND settlement.execution_mode = 'live'
                AND settlement.credit_status = 'pending'
              RETURNING
                settlement.settlement_id, settlement.process_id, settlement.order_id,
                settlement.token_id, settlement.filled_size, settlement.payout,
                settlement.credit_status, settlement.credit_evidence
            ), inserted AS (
              INSERT INTO polymarket.btc_paper_settlement_ledger (
                process_id, run_id, execution_mode, order_id, market_id, token_id, fill_ids,
                official_outcome, official_winning_token_id,
                official_resolution_received_at, official_resolution_source,
                filled_size, entry_notional, entry_fees, payout, net_pnl,
                credit_status, credited_at, credit_attempts, credit_evidence
              )
              SELECT
                candidate.process_id, candidate.run_id, 'live', candidate.order_id,
                candidate.market_id, candidate.token_id, candidate.fill_ids,
                candidate.official_outcome, candidate.official_winning_token_id,
                candidate.official_resolution_received_at,
                candidate.official_resolution_source,
                candidate.filled_size, candidate.entry_notional, candidate.entry_fees,
                $10,
                ($10 - candidate.entry_notional - candidate.entry_fees)::numeric(30,10),
                'credited', now(), 1, $6
              FROM candidate
              CROSS JOIN owners
              CROSS JOIN candidate_state
              CROSS JOIN transaction_state
              WHERE $7
                AND owners.owner_count >= 1
                AND owners.owner_filled_size = $4
                AND candidate_state.candidate_count = 1
                AND transaction_state.transaction_consistent
                AND NOT EXISTS (
                  SELECT 1
                  FROM polymarket.btc_paper_settlement_ledger existing
                  WHERE existing.run_id = candidate.run_id
                    AND existing.order_id = candidate.order_id
                )
              ON CONFLICT (run_id, order_id) DO NOTHING
              RETURNING
                settlement_id, process_id, order_id, token_id, filled_size, payout,
                credit_status, credit_evidence
            ), selected AS (
              SELECT
                updated.settlement_id, updated.process_id, updated.order_id,
                updated.token_id, updated.filled_size, updated.payout,
                updated.credit_status, updated.credit_evidence, true AS applied
              FROM updated
              UNION ALL
              SELECT
                inserted.settlement_id, inserted.process_id, inserted.order_id,
                inserted.token_id, inserted.filled_size, inserted.payout,
                inserted.credit_status, inserted.credit_evidence, true AS applied
              FROM inserted
              UNION ALL
              SELECT
                settlement.settlement_id, settlement.process_id, settlement.order_id,
                settlement.token_id, settlement.filled_size, settlement.payout,
                settlement.credit_status, settlement.credit_evidence, false AS applied
              FROM polymarket.btc_paper_settlement_ledger settlement
              JOIN candidate
                ON candidate.process_id = settlement.process_id
               AND candidate.run_id = settlement.run_id
               AND candidate.order_id = settlement.order_id
              CROSS JOIN candidate_state
              WHERE candidate_state.candidate_count = 1
                AND settlement.execution_mode = 'live'
                AND NOT EXISTS (SELECT 1 FROM updated)
                AND NOT EXISTS (SELECT 1 FROM inserted)
              LIMIT 1
            )
            SELECT
              owners.owner_count,
              owners.owner_filled_size,
              candidate_state.candidate_count,
              transaction_state.transaction_count,
              transaction_state.transaction_consistent,
              selected.settlement_id,
              selected.process_id AS settlement_process_id,
              selected.order_id AS settlement_order_id,
              selected.token_id AS settlement_token_id,
              selected.filled_size AS settlement_filled_size,
              selected.payout AS settlement_payout,
              selected.credit_status,
              selected.credit_evidence,
              COALESCE(selected.applied, false) AS applied
            FROM owners
            CROSS JOIN candidate_state
            CROSS JOIN transaction_state
            LEFT JOIN selected ON true
            "#,
        )
        .bind(process_id)
        .bind(&evidence.condition_id)
        .bind(&evidence.token_id)
        .bind(evidence.redeemed_size)
        .bind(evidence.redeemed_at)
        .bind(&credit_evidence)
        .bind(apply)
        .bind(&evidence.transaction_hash)
        .bind(account_ref)
        .bind(evidence.payout_usd)
        .fetch_one(&self.pool)
        .await
        .context("failed to reconcile process-owned live redemption")?;

        if row.candidate_count > 1 {
            bail!("live redemption matches more than one process-owned order");
        }
        if !row.transaction_consistent {
            bail!("live redemption transaction conflicts with credited settlement evidence");
        }
        if row.owner_count > 0 && row.owner_filled_size != evidence.redeemed_size {
            bail!("live redemption size does not match aggregate sleeve ownership");
        }
        if row.owner_count == 0 || row.candidate_count == 0 {
            return Ok(LiveRedemptionRecognition {
                matched: false,
                applied: false,
                redeemed_size: Decimal::ZERO,
            });
        }
        if !apply {
            return Ok(LiveRedemptionRecognition {
                matched: true,
                applied: false,
                redeemed_size: evidence.redeemed_size,
            });
        }
        let settlement_id = row
            .settlement_id
            .context("matched live redemption is missing its settlement identity")?;
        if row.settlement_process_id != Some(process_id)
            || row.settlement_token_id.as_deref() != Some(evidence.token_id.as_str())
            || row.settlement_filled_size != Some(evidence.redeemed_size)
            || row.settlement_payout != Some(evidence.payout_usd)
            || row.credit_status.as_deref() != Some("credited")
            || row.settlement_order_id.as_deref().is_none_or(str::is_empty)
        {
            bail!("credited live redemption conflicts with exact settlement lineage");
        }
        let persisted_transaction_hash = row
            .credit_evidence
            .as_ref()
            .and_then(|value| value.get("redemption_transaction_hash"))
            .and_then(serde_json::Value::as_str);
        let persisted_proof_type = row
            .credit_evidence
            .as_ref()
            .and_then(|value| value.get("proof_type"))
            .and_then(serde_json::Value::as_str);
        if evidence.payout_usd == Decimal::ZERO
            && persisted_transaction_hash.is_none()
            && persisted_proof_type == Some("btc_official_zero_payout_loss")
        {
            return Ok(LiveRedemptionRecognition {
                matched: true,
                applied: false,
                redeemed_size: evidence.redeemed_size,
            });
        }
        if persisted_transaction_hash != Some(evidence.transaction_hash.as_str()) {
            bail!(
                "credited live redemption {} has conflicting transaction evidence",
                settlement_id
            );
        }
        Ok(LiveRedemptionRecognition {
            matched: true,
            applied: row.applied,
            redeemed_size: row
                .settlement_filled_size
                .context("credited live redemption is missing sleeve size")?,
        })
    }

    /// Reconstructs current outcome-token ownership from exact live fills owned by one process.
    /// The query does not infer ownership from a token or market match: both the fill and its
    /// persisted order must carry the requested process identity.
    pub async fn live_process_position_sizes(
        &self,
        process_id: Uuid,
    ) -> Result<HashMap<String, Decimal>> {
        const MAX_PROCESS_POSITION_TOKENS: usize = 4_000;
        if process_id.is_nil() {
            bail!("live process position evidence requires a non-nil process_id");
        }

        #[derive(FromRow)]
        struct PositionSizeRow {
            token_id: String,
            size: Decimal,
        }

        let cross_owned = sqlx::query_scalar::<_, bool>(
            r#"
            SELECT EXISTS (
              SELECT 1
              FROM polymarket.fills fill
              JOIN polymarket.orders orders ON orders.order_id = fill.order_id
              WHERE fill.process_id = $1
                AND fill.source = 'live'
                AND (
                  orders.process_id IS DISTINCT FROM $1
                  OR orders.token_id IS DISTINCT FROM fill.token_id
                )
              LIMIT 1
            )
            "#,
        )
        .bind(process_id)
        .fetch_one(&self.pool)
        .await
        .context("failed to inspect live process position ownership")?;
        if cross_owned {
            bail!("live process position evidence contains cross-owned fill data");
        }

        let rows = sqlx::query_as::<_, PositionSizeRow>(
            r#"
            WITH filled AS (
              SELECT fill.token_id,
                SUM(
                  CASE orders.side
                    WHEN 'buy' THEN fill.size
                    WHEN 'sell' THEN -fill.size
                  END
                )::numeric AS size
              FROM polymarket.fills fill
              JOIN polymarket.orders orders ON orders.order_id = fill.order_id
              WHERE fill.process_id = $1
                AND orders.process_id = $1
                AND fill.source = 'live'
              GROUP BY fill.token_id
            ), redeemed AS (
              SELECT settlement.token_id,
                SUM(settlement.filled_size)::numeric AS size
              FROM polymarket.btc_paper_settlement_ledger settlement
              WHERE settlement.process_id = $1
                AND settlement.execution_mode = 'live'
                AND settlement.credit_status = 'credited'
              GROUP BY settlement.token_id
            ), exited AS (
              SELECT orders.token_id,
                SUM(account_exit.applied_exit_size)::numeric AS size
              FROM polymarket.account_trades account_exit
              JOIN polymarket.orders orders ON orders.order_id = account_exit.linked_order_id
              WHERE orders.process_id = $1
                AND account_exit.side = 'sell'
                AND account_exit.applied_exit_size > 0
              GROUP BY orders.token_id
            ), tokens AS (
              SELECT token_id FROM filled
              UNION
              SELECT token_id FROM redeemed
              UNION
              SELECT token_id FROM exited
            )
            SELECT tokens.token_id,
              (COALESCE(filled.size, 0) - COALESCE(redeemed.size, 0)
                - COALESCE(exited.size, 0))::numeric AS size
            FROM tokens
            LEFT JOIN filled USING (token_id)
            LEFT JOIN redeemed USING (token_id)
            LEFT JOIN exited USING (token_id)
            WHERE COALESCE(filled.size, 0) - COALESCE(redeemed.size, 0)
              - COALESCE(exited.size, 0) <> 0
            ORDER BY tokens.token_id
            LIMIT 4001
            "#,
        )
        .bind(process_id)
        .fetch_all(&self.pool)
        .await
        .context("failed to reconstruct live process position sizes")?;
        if rows.len() > MAX_PROCESS_POSITION_TOKENS {
            bail!(
                "live process position evidence exceeds the bounded {}-token window",
                MAX_PROCESS_POSITION_TOKENS
            );
        }

        let mut positions = HashMap::with_capacity(rows.len());
        for row in rows {
            if row.token_id.trim().is_empty() || row.size <= Decimal::ZERO {
                bail!("live process position evidence contains an invalid net position");
            }
            if positions.insert(row.token_id, row.size).is_some() {
                bail!("live process position evidence contains a duplicate token identity");
            }
        }
        Ok(positions)
    }

    /// Reconstructs the wallet position expected from every process sleeve assigned to one live
    /// account. The aggregation uses immutable order/fill process ownership and includes stopped
    /// processes with unresolved financial state.
    pub async fn live_account_position_sizes(
        &self,
        account_ref: &str,
    ) -> Result<HashMap<String, Decimal>> {
        const MAX_ACCOUNT_POSITION_TOKENS: usize = 4_000;
        let account_ref = account_ref.trim();
        if account_ref.is_empty() || account_ref.len() > 128 {
            bail!("live account position evidence requires a bounded account_ref");
        }

        #[derive(FromRow)]
        struct PositionSizeRow {
            token_id: String,
            size: Decimal,
        }

        let rows = sqlx::query_as::<_, PositionSizeRow>(
            r#"
            WITH account_processes AS (
              SELECT process_id
              FROM polymarket.trading_processes
              WHERE config #>> '{execution,mode}' = 'live'
                AND lower(btrim(config #>> '{execution,account_ref}')) = lower($1)
            ), filled AS (
              SELECT fill.token_id,
                SUM(
                  CASE orders.side
                    WHEN 'buy' THEN fill.size
                    WHEN 'sell' THEN -fill.size
                  END
                )::numeric AS size
              FROM polymarket.fills fill
              JOIN polymarket.orders orders
                ON orders.order_id = fill.order_id
               AND orders.process_id = fill.process_id
              JOIN account_processes process
                ON process.process_id = fill.process_id
              WHERE fill.source = 'live'
              GROUP BY fill.token_id
            ), redeemed AS (
              SELECT settlement.token_id,
                SUM(settlement.filled_size)::numeric AS size
              FROM polymarket.btc_paper_settlement_ledger settlement
              JOIN account_processes process
                ON process.process_id = settlement.process_id
              WHERE settlement.execution_mode = 'live'
                AND settlement.credit_status = 'credited'
              GROUP BY settlement.token_id
            ), exited AS (
              SELECT orders.token_id,
                SUM(account_exit.applied_exit_size)::numeric AS size
              FROM polymarket.account_trades account_exit
              JOIN polymarket.orders orders ON orders.order_id = account_exit.linked_order_id
              JOIN account_processes process ON process.process_id = orders.process_id
              WHERE account_exit.side = 'sell'
                AND account_exit.applied_exit_size > 0
              GROUP BY orders.token_id
            ), tokens AS (
              SELECT token_id FROM filled
              UNION
              SELECT token_id FROM redeemed
              UNION
              SELECT token_id FROM exited
            )
            SELECT tokens.token_id,
              (COALESCE(filled.size, 0) - COALESCE(redeemed.size, 0)
                - COALESCE(exited.size, 0))::numeric AS size
            FROM tokens
            LEFT JOIN filled USING (token_id)
            LEFT JOIN redeemed USING (token_id)
            LEFT JOIN exited USING (token_id)
            WHERE COALESCE(filled.size, 0) - COALESCE(redeemed.size, 0)
              - COALESCE(exited.size, 0) <> 0
            ORDER BY tokens.token_id
            LIMIT 4001
            "#,
        )
        .bind(account_ref)
        .fetch_all(&self.pool)
        .await
        .context("failed to reconstruct live account position sizes")?;
        if rows.len() > MAX_ACCOUNT_POSITION_TOKENS {
            bail!(
                "live account position evidence exceeds the bounded {}-token window",
                MAX_ACCOUNT_POSITION_TOKENS
            );
        }

        let mut positions = HashMap::with_capacity(rows.len());
        for row in rows {
            if row.token_id.trim().is_empty() || row.size <= Decimal::ZERO {
                bail!("live account position evidence contains an invalid net position");
            }
            if positions.insert(row.token_id, row.size).is_some() {
                bail!("live account position evidence contains a duplicate token identity");
            }
        }
        Ok(positions)
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

    /// Durably closes a pending live submission that failed a final local gate before any venue
    /// POST. The complete rejected record is persisted so idempotent retries return the exact
    /// bounded gate evidence instead of an orphaned pending order.
    pub async fn mark_order_pre_submit_rejected(
        &self,
        rejected: &OrderRecord,
    ) -> Result<OrderRecord> {
        if rejected.state != OrderState::Rejected {
            bail!("pre-submit rejection must have rejected order state");
        }
        let row = sqlx::query_as::<_, OrderDbRow>(
            r#"
            UPDATE polymarket.orders
            SET state = 'rejected',
                venue_status = 'live_execution_gate_closed',
                last_venue_update_at = now(),
                reconciliation_status = 'live_execution_gate_closed',
                raw_payload = $2,
                updated_at = now()
            WHERE client_order_id = $1
              AND state = 'submitted'
              AND venue_order_id IS NULL
            RETURNING order_id, state, raw_payload, created_at, updated_at
            "#,
        )
        .bind(rejected.request.client_order_id)
        .bind(serde_json::to_value(rejected)?)
        .fetch_optional(&self.pool)
        .await
        .context("failed to mark live order pre-submit rejection")?
        .with_context(|| {
            format!(
                "pending live order {} changed before pre-submit rejection",
                rejected.request.client_order_id
            )
        })?;
        order_from_db_row(row)
    }

    pub async fn mark_order_submit_unknown(
        &self,
        client_order_id: Uuid,
        reason: &str,
        raw_error: serde_json::Value,
    ) -> Result<OrderRecord> {
        self.mark_order_by_client_id(
            client_order_id,
            None,
            OrderState::Unknown,
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

    pub async fn order_filled_size(&self, order_id: &str) -> Result<Decimal> {
        sqlx::query_scalar::<_, Decimal>(
            r#"
            SELECT COALESCE(SUM(size), 0)::numeric
            FROM polymarket.fills
            WHERE order_id = $1
            "#,
        )
        .bind(order_id)
        .fetch_one(&self.pool)
        .await
        .context("failed to calculate cumulative order fill size")
    }

    /// Applies cumulative fill progress without permitting a late partial-fill event to downgrade
    /// a terminal order. Complete venue fill evidence is authoritative and may repair any earlier
    /// local terminal classification.
    pub async fn mark_order_fill_progress(
        &self,
        order_id: &str,
        cumulative_filled_size: Decimal,
        raw_event: serde_json::Value,
    ) -> Result<Option<OrderRecord>> {
        let row = sqlx::query_as::<_, OrderDbRow>(
            r#"
            UPDATE polymarket.orders
            SET state = CASE
                  WHEN $2 >= size THEN 'filled'
                  WHEN $2 > 0
                    AND state NOT IN ('filled', 'cancelled', 'rejected', 'expired')
                    THEN 'partially_filled'
                  ELSE state
                END,
                venue_status = CASE
                  WHEN $2 >= size THEN 'filled'
                  WHEN $2 > 0 THEN 'partially_filled'
                  ELSE venue_status
                END,
                last_venue_update_at = now(),
                reconciliation_status = CASE
                  WHEN $2 >= size THEN 'filled'
                  WHEN $2 > 0 THEN 'partially_filled'
                  ELSE reconciliation_status
                END,
                raw_payload = jsonb_set(raw_payload, '{venue}', $3::jsonb, true),
                updated_at = now()
            WHERE order_id = $1 OR venue_order_id = $1
            RETURNING order_id, state, raw_payload, created_at, updated_at
            "#,
        )
        .bind(order_id)
        .bind(cumulative_filled_size)
        .bind(raw_event)
        .fetch_optional(&self.pool)
        .await
        .context("failed to apply cumulative order fill progress")?;
        row.map(order_from_db_row).transpose()
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
            SET state = CASE
                  WHEN state = 'filled' THEN state
                  WHEN state IN ('cancelled', 'rejected', 'expired') AND $2 <> 'filled' THEN state
                  ELSE $2
                END,
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
        process_id: Option<Uuid>,
        account_ref: Option<&str>,
        status: &str,
        open_orders_seen: i32,
        fills_seen: i32,
        balances_seen: i32,
        mismatches_found: i32,
        mismatches_repaired: i32,
        unresolved_count: i32,
        raw_summary: serde_json::Value,
    ) -> Result<Uuid> {
        if process_id.is_some() != account_ref.is_some() {
            bail!("live reconciliation process_id and account_ref must be provided together");
        }
        let account_ref = account_ref.map(str::trim);
        if account_ref.is_some_and(|account_ref| account_ref.is_empty() || account_ref.len() > 128)
        {
            bail!("live reconciliation account_ref must contain between 1 and 128 bytes");
        }
        let run_id = sqlx::query_scalar::<_, Uuid>(
            r#"
            INSERT INTO polymarket.live_reconciliation_runs (
              run_id, process_id, account_ref, started_at, completed_at, status, open_orders_seen,
              fills_seen, balances_seen, mismatches_found, mismatches_repaired,
              unresolved_count, raw_summary
            )
            VALUES (gen_random_uuid(),$1,$2,now(),now(),$3,$4,$5,$6,$7,$8,$9,$10)
            RETURNING run_id
            "#,
        )
        .bind(process_id)
        .bind(account_ref)
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
            WHERE polymarket.account_trades.linked_order_id IS NULL
               OR EXCLUDED.linked_order_id IS NULL
               OR polymarket.account_trades.linked_order_id = EXCLUDED.linked_order_id
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
        .fetch_optional(&self.pool)
        .await
        .context("failed to upsert account trade")?
        .with_context(|| {
            format!(
                "account trade {} conflicts with an existing persisted order link",
                trade.account_trade_id
            )
        })?;
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
              run_id, process_id, account_address, source, dry_run, token_id, lookback_hours,
              started_at, completed_at, status, activities_fetched, account_trades_inserted,
              position_snapshots_inserted, exits_detected, exits_applied,
              mismatches_found, unmatched_trades, raw_summary
            )
            VALUES (
              gen_random_uuid(),$1,$2,$3,$4,$5,$6,now(),now(),$7,$8,$9,$10,$11,$12,$13,$14,$15
            )
            RETURNING run_id
            "#,
        )
        .bind(report.process_id)
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
        .bind(serde_json::json!({
            "report": report,
            "process_accounting_proof": &report.process_accounting_proof,
        }))
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

fn build_live_process_exposure_snapshot(
    process_id: Uuid,
    ignored_client_order_id: Option<Uuid>,
    as_of: DateTime<Utc>,
    has_unredeemed_settlement: bool,
    orders: &[LiveExposureOrderRow],
    fills: &[LiveExposureFillRow],
) -> Result<LiveProcessExposureSnapshot> {
    let mut pending_order_count = 0usize;
    let mut pending_requested_notional_usd = Decimal::ZERO;
    let mut pending_requested_fees_usd = Decimal::ZERO;
    let mut exposed_market_ids = HashSet::new();

    for row in orders {
        let order = validate_live_exposure_order(row, process_id, as_of)?;
        if Some(order.request.client_order_id) == ignored_client_order_id {
            continue;
        }
        let (notional, fee) = live_requested_exposure(&order.request)?;
        pending_requested_notional_usd = checked_live_exposure_add(
            pending_requested_notional_usd,
            notional,
            "pending requested notional",
        )?;
        pending_requested_fees_usd =
            checked_live_exposure_add(pending_requested_fees_usd, fee, "pending requested fees")?;
        pending_order_count = pending_order_count
            .checked_add(1)
            .context("live pending order count overflow")?;
        exposed_market_ids.insert(order.request.market_id);
    }

    let mut buy_fill_count = 0usize;
    let mut filled_buy_notional_usd = Decimal::ZERO;
    let mut filled_buy_fees_usd = Decimal::ZERO;
    let mut fill_ids = HashSet::with_capacity(fills.len());
    let mut filled_sizes_by_order: HashMap<&str, (Decimal, Decimal)> = HashMap::new();
    for row in fills {
        let order = validate_live_exposure_fill(row, process_id, as_of)?;
        if !fill_ids.insert(row.fill_id) {
            bail!("live process exposure contains a duplicate fill identity");
        }
        let filled = filled_sizes_by_order
            .entry(row.fill_order_id.as_str())
            .or_insert((Decimal::ZERO, order.request.size));
        if filled.1 != order.request.size {
            bail!("live process fill evidence has conflicting order sizes");
        }
        filled.0 = checked_live_exposure_add(filled.0, row.fill_size, "filled order size")?;
        if filled.0 > filled.1 {
            bail!("live process cumulative fill size exceeds its requested order size");
        }

        if order.request.side == crate::models::OrderSide::Buy {
            let notional =
                checked_live_exposure_mul(row.fill_price, row.fill_size, "filled BUY notional")?;
            filled_buy_notional_usd = checked_live_exposure_add(
                filled_buy_notional_usd,
                notional,
                "filled BUY notional",
            )?;
            filled_buy_fees_usd =
                checked_live_exposure_add(filled_buy_fees_usd, row.fill_fee, "filled BUY fees")?;
            buy_fill_count = buy_fill_count
                .checked_add(1)
                .context("live BUY fill count overflow")?;
            exposed_market_ids.insert(order.request.market_id);
        }
    }

    let pending_exposure = checked_live_exposure_add(
        pending_requested_notional_usd,
        pending_requested_fees_usd,
        "pending exposure",
    )?;
    let filled_exposure = checked_live_exposure_add(
        filled_buy_notional_usd,
        filled_buy_fees_usd,
        "filled BUY exposure",
    )?;
    let total_exposure_usd =
        checked_live_exposure_add(pending_exposure, filled_exposure, "total live exposure")?;
    let mut exposed_market_ids = exposed_market_ids.into_iter().collect::<Vec<_>>();
    exposed_market_ids.sort_unstable();

    Ok(LiveProcessExposureSnapshot {
        process_id,
        has_unredeemed_settlement,
        pending_order_count,
        buy_fill_count,
        pending_requested_notional_usd,
        pending_requested_fees_usd,
        filled_buy_notional_usd,
        filled_buy_fees_usd,
        total_exposure_usd,
        exposed_market_ids,
    })
}

fn validate_live_exposure_order(
    row: &LiveExposureOrderRow,
    process_id: Uuid,
    as_of: DateTime<Utc>,
) -> Result<OrderRecord> {
    if row.process_id != Some(process_id) {
        bail!("live order exposure crossed process ownership");
    }
    validate_live_exposure_identity(&row.order_id, "order_id")?;
    validate_live_exposure_identity(&row.market_id, "market_id")?;
    validate_live_exposure_identity(&row.token_id, "token_id")?;
    if row.raw_payload_bytes <= 0 || row.raw_payload_bytes > MAX_LIVE_EXPOSURE_ORDER_PAYLOAD_BYTES {
        bail!("live order exposure payload exceeds its bounded evidence window");
    }
    if row.created_at > row.updated_at || row.updated_at > as_of {
        bail!("live order exposure has inconsistent persistence timestamps");
    }
    let order = order_from_db_row(OrderDbRow {
        order_id: row.order_id.clone(),
        state: row.state.clone(),
        raw_payload: row.raw_payload.clone(),
        created_at: row.created_at,
        updated_at: row.updated_at,
    })?;
    if !matches!(
        order.state,
        OrderState::Created
            | OrderState::Submitted
            | OrderState::Acknowledged
            | OrderState::PartiallyFilled
            | OrderState::CancelRequested
            | OrderState::Unknown
    ) {
        bail!("live order exposure query returned a terminal order");
    }
    validate_live_exposure_order_identity(
        &order,
        process_id,
        row.client_order_id,
        &row.market_id,
        &row.token_id,
        &row.side,
        &row.order_type,
        row.price,
        row.size,
    )?;
    live_requested_exposure(&order.request)?;
    Ok(order)
}

fn validate_live_exposure_fill(
    row: &LiveExposureFillRow,
    process_id: Uuid,
    as_of: DateTime<Utc>,
) -> Result<OrderRecord> {
    if row.fill_process_id != Some(process_id)
        || row.identity_process_id != Some(process_id)
        || row.order_process_id != Some(process_id)
        || row.identity_timestamp_utc != Some(row.filled_at)
    {
        bail!("live fill exposure crossed or lacks exact process identity lineage");
    }
    if row.fill_source != "live" {
        bail!("live process exposure contains a non-live fill source");
    }
    validate_live_exposure_identity(&row.fill_order_id, "fill order_id")?;
    validate_live_exposure_identity(&row.fill_token_id, "fill token_id")?;
    if row.fill_price <= Decimal::ZERO
        || row.fill_price > Decimal::ONE
        || row.fill_size <= Decimal::ZERO
        || row.fill_fee < Decimal::ZERO
    {
        bail!("live fill exposure contains invalid timestamps or economics");
    }

    let order_client_order_id = row
        .order_client_order_id
        .context("live fill exposure has no persisted order client identity")?;
    let order_market_id = row
        .order_market_id
        .as_deref()
        .context("live fill exposure has no persisted order market identity")?;
    let order_token_id = row
        .order_token_id
        .as_deref()
        .context("live fill exposure has no persisted order token identity")?;
    let order_side = row
        .order_side
        .as_deref()
        .context("live fill exposure has no persisted order side")?;
    let order_type = row
        .order_type
        .as_deref()
        .context("live fill exposure has no persisted order type")?;
    let order_price = row
        .order_price
        .context("live fill exposure has no persisted order price")?;
    let order_size = row
        .order_size
        .context("live fill exposure has no persisted order size")?;
    let order_state = row
        .order_state
        .as_deref()
        .context("live fill exposure has no persisted order state")?;
    let order_raw_payload = row
        .order_raw_payload
        .as_ref()
        .context("live fill exposure has no persisted order payload")?;
    let order_raw_payload_bytes = row
        .order_raw_payload_bytes
        .context("live fill exposure has no bounded order payload size")?;
    let order_created_at = row
        .order_created_at
        .context("live fill exposure has no persisted order creation time")?;
    let order_updated_at = row
        .order_updated_at
        .context("live fill exposure has no persisted order update time")?;
    validate_live_exposure_identity(order_market_id, "fill order market_id")?;
    validate_live_exposure_identity(order_token_id, "fill order token_id")?;
    let earliest_fill_at = order_created_at
        .checked_sub_signed(LIVE_FILL_RECONCILIATION_SKEW)
        .context("live fill exposure order window underflow")?;
    let latest_fill_at = order_created_at
        .checked_add_signed(LIVE_FILL_RECONCILIATION_SKEW)
        .context("live fill exposure order window overflow")?;
    let earliest_order_update_at = order_created_at
        .checked_sub_signed(LIVE_EXTERNAL_EVENT_CLOCK_SKEW)
        .context("live fill exposure order update window underflow")?;
    let latest_order_update_at = as_of
        .checked_add_signed(LIVE_EXTERNAL_EVENT_CLOCK_SKEW)
        .context("live fill exposure database observation window overflow")?;
    if order_raw_payload_bytes <= 0
        || order_raw_payload_bytes > MAX_LIVE_EXPOSURE_ORDER_PAYLOAD_BYTES
        || order_updated_at < earliest_order_update_at
        || order_updated_at > latest_order_update_at
        || row.filled_at < earliest_fill_at
        || row.filled_at > latest_fill_at
    {
        bail!(
            "live fill exposure has inconsistent bounded order evidence: fill_id={}, order_id={}, fill_at={}, order_created_at={}, order_updated_at={}, database_as_of={}, fill_order_delta_ms={}, allowed_fill_order_skew_ms={}, order_update_creation_delta_ms={}, order_update_observation_delta_ms={}, allowed_external_clock_skew_ms={}",
            row.fill_id,
            row.fill_order_id,
            row.filled_at,
            order_created_at,
            order_updated_at,
            as_of,
            row.filled_at
                .signed_duration_since(order_created_at)
                .num_milliseconds(),
            LIVE_FILL_RECONCILIATION_SKEW.num_milliseconds(),
            order_updated_at
                .signed_duration_since(order_created_at)
                .num_milliseconds(),
            order_updated_at.signed_duration_since(as_of).num_milliseconds(),
            LIVE_EXTERNAL_EVENT_CLOCK_SKEW.num_milliseconds(),
        );
    }
    if row.fill_token_id != order_token_id {
        bail!("live fill exposure token does not match its persisted order");
    }

    let order = order_from_db_row(OrderDbRow {
        order_id: row.fill_order_id.clone(),
        state: order_state.to_string(),
        raw_payload: order_raw_payload.clone(),
        created_at: order_created_at,
        updated_at: order_updated_at,
    })?;
    validate_live_exposure_order_identity(
        &order,
        process_id,
        order_client_order_id,
        order_market_id,
        order_token_id,
        order_side,
        order_type,
        order_price,
        order_size,
    )?;
    live_requested_exposure(&order.request)?;
    Ok(order)
}

#[allow(clippy::too_many_arguments)]
fn validate_live_exposure_order_identity(
    order: &OrderRecord,
    process_id: Uuid,
    client_order_id: Uuid,
    market_id: &str,
    token_id: &str,
    side: &str,
    order_type: &str,
    price: Decimal,
    size: Decimal,
) -> Result<()> {
    if order.request.process_id != Some(process_id)
        || order.request.client_order_id != client_order_id
        || order.request.market_id != market_id
        || order.request.token_id != token_id
        || serialized_name(&order.request.side)? != side
        || serialized_name(&order.request.order_type)? != order_type
        || order.request.price != price
        || order.request.size != size
        || !order.request.metadata.is_object()
    {
        bail!("live exposure order payload does not match canonical database identity");
    }
    Ok(())
}

fn live_requested_exposure(request: &OrderRequest) -> Result<(Decimal, Decimal)> {
    if request.price <= Decimal::ZERO
        || request.price > Decimal::ONE
        || request.size <= Decimal::ZERO
    {
        bail!("live requested exposure contains invalid price or size");
    }
    let fee_rate = sealed_dynamic_fee_rate(&request.metadata)
        .context("live requested exposure has invalid dynamic_fee_rate")?;
    let notional = checked_live_exposure_mul(request.price, request.size, "requested notional")?;
    let fee = dynamic_crypto_taker_fee(request.size, fee_rate, request.price);
    Ok((notional, fee))
}

fn checked_live_exposure_add(left: Decimal, right: Decimal, field: &str) -> Result<Decimal> {
    left.checked_add(right)
        .with_context(|| format!("live {field} overflow"))
}

fn checked_live_exposure_mul(left: Decimal, right: Decimal, field: &str) -> Result<Decimal> {
    left.checked_mul(right)
        .with_context(|| format!("live {field} overflow"))
}

fn validate_live_exposure_identity(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > MAX_LIVE_EXPOSURE_IDENTITY_BYTES {
        bail!("live exposure {field} must be non-empty and bounded");
    }
    Ok(())
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
        && existing.metadata == incoming.metadata
}

fn durable_fill_state_supersedes_report(existing: OrderState, reported: OrderState) -> bool {
    matches!(
        (existing, reported),
        (
            OrderState::PartiallyFilled,
            OrderState::Submitted | OrderState::Acknowledged
        ) | (
            OrderState::Filled,
            OrderState::Submitted | OrderState::Acknowledged | OrderState::PartiallyFilled
        )
    )
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
            | "live_execution_gate"
            | "reference_execution_submit"
            | "reference_execution_arrival"
    )
}

fn parse_bounded_live_settlement_fill_ids(value: &serde_json::Value) -> Result<Vec<Uuid>> {
    let values = value
        .as_array()
        .context("live settlement fill_ids must be an array")?;
    if values.is_empty() || values.len() > MAX_LIVE_DAILY_SETTLEMENT_FILLS {
        bail!("live settlement fill_ids must be non-empty and bounded");
    }
    let fill_ids = values
        .iter()
        .map(|value| {
            let value = value
                .as_str()
                .context("live settlement fill identity must be a UUID string")?;
            Uuid::parse_str(value).context("live settlement fill identity must be a UUID")
        })
        .collect::<Result<Vec<_>>>()?;
    if fill_ids.iter().copied().collect::<HashSet<_>>().len() != fill_ids.len() {
        bail!("live settlement contains duplicate fill identities");
    }
    Ok(fill_ids)
}

fn validate_live_daily_settlement(
    settlement: &LiveDailySettlementRow,
    day_start: DateTime<Utc>,
    day_end: DateTime<Utc>,
    as_of: DateTime<Utc>,
) -> Result<()> {
    let credited_at = settlement
        .credited_at
        .context("credited live settlement is missing credited_at")?;
    if credited_at < day_start || credited_at >= day_end || credited_at > as_of {
        bail!("credited live settlement is outside the proven UTC-day observation window");
    }
    let latest_resolution_receipt = credited_at
        .checked_add_signed(LIVE_EXTERNAL_EVENT_CLOCK_SKEW)
        .context("credited live settlement resolution clock window overflow")?;
    if settlement.official_resolution_received_at > latest_resolution_receipt
        || settlement.created_at > credited_at
        || settlement.updated_at < settlement.created_at
        || settlement.updated_at > as_of
        || settlement.credit_attempts < 1
    {
        bail!(
            "credited live settlement has inconsistent accounting timestamps or attempts: settlement_id={}, resolution_received_at={}, created_at={}, updated_at={}, credited_at={}, database_as_of={}, resolution_credit_delta_ms={}, allowed_external_clock_skew_ms={}, credit_attempts={}",
            settlement.settlement_id,
            settlement.official_resolution_received_at,
            settlement.created_at,
            settlement.updated_at,
            credited_at,
            as_of,
            settlement
                .official_resolution_received_at
                .signed_duration_since(credited_at)
                .num_milliseconds(),
            LIVE_EXTERNAL_EVENT_CLOCK_SKEW.num_milliseconds(),
            settlement.credit_attempts,
        );
    }
    if settlement.order_id.trim().is_empty()
        || settlement.market_id.trim().is_empty()
        || settlement.token_id.trim().is_empty()
        || settlement.official_winning_token_id.trim().is_empty()
        || !matches!(settlement.official_outcome.as_str(), "up" | "down")
        || !matches!(
            settlement.official_resolution_source.as_str(),
            "clob_websocket" | "clob_rest_reconciliation" | "gamma_rest_reconciliation"
        )
    {
        bail!("credited live settlement has invalid official-resolution identity");
    }
    let expected_payout = if settlement.token_id == settlement.official_winning_token_id {
        settlement.filled_size
    } else {
        Decimal::ZERO
    };
    if settlement.filled_size <= Decimal::ZERO
        || settlement.entry_notional < Decimal::ZERO
        || settlement.entry_fees < Decimal::ZERO
        || settlement.payout != expected_payout
        || settlement.net_pnl
            != settlement.payout - settlement.entry_notional - settlement.entry_fees
    {
        bail!("credited live settlement has inconsistent economics");
    }

    let evidence = settlement
        .credit_evidence
        .as_object()
        .context("credited live settlement evidence must be an object")?;
    if evidence
        .get("proof_type")
        .and_then(serde_json::Value::as_str)
        == Some("btc_official_zero_payout_loss")
    {
        if settlement.payout != Decimal::ZERO
            || settlement.token_id == settlement.official_winning_token_id
        {
            bail!("official zero-payout proof requires a losing live settlement");
        }
        require_live_evidence_string(
            evidence,
            "evidence_version",
            "btc_live_zero_payout_settlement_v1",
        )?;
        require_live_evidence_string(evidence, "recognition_kind", "official_resolution")?;
        require_live_evidence_string(evidence, "execution_mode", "live")?;
        if evidence
            .get("exchange_cash_credit_applied")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        {
            bail!("official zero-payout proof must not claim an exchange cash credit");
        }
        require_live_evidence_uuid(evidence, "settlement_id", settlement.settlement_id)?;
        require_live_evidence_uuid(evidence, "process_id", settlement.process_id)?;
        require_live_evidence_uuid(evidence, "run_id", settlement.run_id)?;
        require_live_evidence_string(evidence, "order_id", &settlement.order_id)?;
        require_live_evidence_string(evidence, "market_id", &settlement.market_id)?;
        require_live_evidence_string(evidence, "token_id", &settlement.token_id)?;
        if evidence.get("fill_ids") != Some(&settlement.fill_ids) {
            bail!("official zero-payout proof fill lineage does not match");
        }
        require_live_evidence_string(evidence, "official_outcome", &settlement.official_outcome)?;
        require_live_evidence_string(
            evidence,
            "official_winning_token_id",
            &settlement.official_winning_token_id,
        )?;
        require_live_evidence_string(
            evidence,
            "official_resolution_source",
            &settlement.official_resolution_source,
        )?;
        require_live_evidence_datetime(
            evidence,
            "official_resolution_received_at",
            settlement.official_resolution_received_at,
        )?;
        require_live_evidence_decimal(evidence, "filled_size", settlement.filled_size)?;
        require_live_evidence_decimal(evidence, "entry_notional", settlement.entry_notional)?;
        require_live_evidence_decimal(evidence, "entry_fees", settlement.entry_fees)?;
        require_live_evidence_decimal(evidence, "payout", settlement.payout)?;
        require_live_evidence_decimal(evidence, "net_pnl", settlement.net_pnl)?;
        if !evidence
            .get("recognized_by_config_hash")
            .and_then(serde_json::Value::as_str)
            .is_some_and(is_sha256_hex)
        {
            bail!("official zero-payout proof is missing its configuration fingerprint");
        }
        return Ok(());
    }
    if evidence
        .get("proof_type")
        .and_then(serde_json::Value::as_str)
        == Some("polymarket_data_api_redeem")
    {
        require_live_evidence_string(evidence, "token_id", &settlement.token_id)?;
        require_live_evidence_decimal(evidence, "redeemed_size", settlement.filled_size)?;
        require_live_evidence_decimal(evidence, "payout_usd", settlement.payout)?;
        let account_address = evidence
            .get("account_address")
            .and_then(serde_json::Value::as_str)
            .context("credited live redemption is missing its account address")?;
        if account_address.len() != 42
            || !account_address.starts_with("0x")
            || !account_address[2..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            bail!("credited live redemption has an invalid account address");
        }
        let condition_id = evidence
            .get("condition_id")
            .and_then(serde_json::Value::as_str)
            .context("credited live redemption is missing its condition identity")?;
        let transaction_hash = evidence
            .get("redemption_transaction_hash")
            .and_then(serde_json::Value::as_str)
            .context("credited live redemption is missing its transaction identity")?;
        if !is_prefixed_sha256_hex(condition_id) || !is_prefixed_sha256_hex(transaction_hash) {
            bail!("credited live redemption has invalid exchange identity evidence");
        }
        let redeemed_at = evidence
            .get("redeemed_at")
            .and_then(serde_json::Value::as_str)
            .context("credited live redemption is missing its exchange timestamp")?
            .parse::<DateTime<Utc>>()
            .context("credited live redemption exchange timestamp is invalid")?;
        if redeemed_at
            > credited_at
                .checked_add_signed(LIVE_EXTERNAL_EVENT_CLOCK_SKEW)
                .context("credited live redemption clock window overflow")?
        {
            bail!("credited live redemption was recorded before its exchange evidence");
        }
        if !evidence
            .get("activity")
            .is_some_and(serde_json::Value::is_object)
        {
            bail!("credited live redemption is missing its raw exchange activity");
        }
        return Ok(());
    }
    require_live_evidence_string(
        evidence,
        "evidence_version",
        "btc_live_settlement_recognition_v1",
    )?;
    require_live_evidence_string(evidence, "recognition_kind", "internal_accounting")?;
    require_live_evidence_string(evidence, "execution_mode", "live")?;
    if evidence
        .get("exchange_cash_credit_applied")
        .and_then(serde_json::Value::as_bool)
        != Some(false)
    {
        bail!("credited live settlement must not claim an exchange cash credit");
    }
    require_live_evidence_uuid(evidence, "settlement_id", settlement.settlement_id)?;
    require_live_evidence_uuid(evidence, "process_id", settlement.process_id)?;
    require_live_evidence_uuid(evidence, "run_id", settlement.run_id)?;
    require_live_evidence_string(evidence, "order_id", &settlement.order_id)?;
    require_live_evidence_string(evidence, "market_id", &settlement.market_id)?;
    require_live_evidence_string(evidence, "token_id", &settlement.token_id)?;
    require_live_evidence_string(evidence, "official_outcome", &settlement.official_outcome)?;
    require_live_evidence_string(
        evidence,
        "official_winning_token_id",
        &settlement.official_winning_token_id,
    )?;
    require_live_evidence_string(
        evidence,
        "official_resolution_source",
        &settlement.official_resolution_source,
    )?;
    require_live_evidence_datetime(
        evidence,
        "official_resolution_received_at",
        settlement.official_resolution_received_at,
    )?;
    require_live_evidence_decimal(evidence, "filled_size", settlement.filled_size)?;
    require_live_evidence_decimal(evidence, "entry_notional", settlement.entry_notional)?;
    require_live_evidence_decimal(evidence, "entry_fees", settlement.entry_fees)?;
    require_live_evidence_decimal(evidence, "payout", settlement.payout)?;
    require_live_evidence_decimal(evidence, "net_pnl", settlement.net_pnl)?;
    if !evidence
        .get("recognized_by_config_hash")
        .and_then(serde_json::Value::as_str)
        .is_some_and(is_sha256_hex)
    {
        bail!("credited live settlement is missing its bounded configuration fingerprint");
    }
    let venue = evidence
        .get("venue_reconciliation")
        .and_then(serde_json::Value::as_object)
        .context("credited live settlement is missing venue reconciliation evidence")?;
    if venue
        .get("balances_checked")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
        || venue
            .get("mismatches_found")
            .and_then(serde_json::Value::as_u64)
            != Some(0)
        || venue
            .get("unresolved_count")
            .and_then(serde_json::Value::as_u64)
            != Some(0)
    {
        bail!("credited live settlement was not recognized from a clean venue reconciliation");
    }
    Ok(())
}

fn require_live_evidence_string(
    evidence: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    expected: &str,
) -> Result<()> {
    if evidence.get(field).and_then(serde_json::Value::as_str) != Some(expected) {
        bail!("credited live settlement evidence field {field} does not match");
    }
    Ok(())
}

fn require_live_evidence_uuid(
    evidence: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    expected: Uuid,
) -> Result<()> {
    let actual = evidence
        .get(field)
        .and_then(serde_json::Value::as_str)
        .context("credited live settlement evidence UUID is missing")?;
    if Uuid::parse_str(actual).context("credited live settlement evidence UUID is invalid")?
        != expected
    {
        bail!("credited live settlement evidence UUID field {field} does not match");
    }
    Ok(())
}

fn require_live_evidence_datetime(
    evidence: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    expected: DateTime<Utc>,
) -> Result<()> {
    let actual = evidence
        .get(field)
        .and_then(serde_json::Value::as_str)
        .context("credited live settlement evidence timestamp is missing")?;
    let actual = DateTime::parse_from_rfc3339(actual)
        .context("credited live settlement evidence timestamp is invalid")?
        .with_timezone(&Utc);
    if actual != expected {
        bail!("credited live settlement evidence timestamp field {field} does not match");
    }
    Ok(())
}

fn require_live_evidence_decimal(
    evidence: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    expected: Decimal,
) -> Result<()> {
    let value = evidence
        .get(field)
        .context("credited live settlement evidence decimal is missing")?;
    let raw = value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string());
    let actual = raw
        .parse::<Decimal>()
        .context("credited live settlement evidence decimal is invalid")?;
    if actual != expected {
        bail!("credited live settlement evidence decimal field {field} does not match");
    }
    Ok(())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_prefixed_sha256_hex(value: &str) -> bool {
    value.len() == 66
        && value.starts_with("0x")
        && value[2..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[allow(clippy::too_many_arguments)]
fn manual_live_exit_economics(
    exit_size: Decimal,
    exit_price: Decimal,
    fee_rate: Decimal,
    net_proceeds: Decimal,
    entry_notional: Decimal,
    entry_fees: Decimal,
    fee_tolerance: Decimal,
) -> Result<(Decimal, Decimal, Decimal)> {
    let gross_proceeds = exit_size * exit_price;
    if exit_size <= Decimal::ZERO
        || exit_price <= Decimal::ZERO
        || exit_price >= Decimal::ONE
        || net_proceeds <= Decimal::ZERO
        || net_proceeds > gross_proceeds
        || entry_notional <= Decimal::ZERO
        || entry_fees < Decimal::ZERO
        || fee_tolerance < Decimal::ZERO
    {
        bail!("manual live exit economics are invalid");
    }
    let expected_exit_fee = dynamic_crypto_taker_fee(exit_size, fee_rate, exit_price);
    let observed_exit_fee = gross_proceeds - net_proceeds;
    if (expected_exit_fee - observed_exit_fee).abs() > fee_tolerance {
        bail!("manual live exit net proceeds do not match centralized fee economics");
    }
    Ok((
        expected_exit_fee,
        observed_exit_fee,
        net_proceeds - entry_notional - entry_fees,
    ))
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, Utc};
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    use crate::{
        models::{
            FillRecord, FillSource, OrderRecord, OrderRequest, OrderSide, OrderState, OrderType,
        },
        store::{
            build_live_process_exposure_snapshot, canonical_fill_for_storage,
            durable_fill_state_supersedes_report, fill_record_matches, live_requested_exposure,
            manual_live_exit_economics, order_request_identity_matches,
            order_request_result_matches, required_fill_process_id, validate_live_daily_settlement,
            validate_live_exposure_fill, LiveDailySettlementRow, LiveExposureFillRow,
            LiveExposureOrderRow, HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL, INSERT_FILL_IDENTITY_SQL,
            INSERT_FILL_SQL, INSERT_ORDER_SQL, RECORD_IDEMPOTENT_TRADING_PROCESS_EVENT_SQL,
            SELECT_CREDITED_LIVE_PROCESS_SETTLEMENTS_SQL, SELECT_FILL_IDENTITY_SQL,
            SELECT_FILL_SQL, SELECT_LIVE_PROCESS_CROSS_OWNED_FILL_EXISTS_SQL,
            SELECT_LIVE_PROCESS_EXPOSURE_FILLS_SQL, SELECT_LIVE_PROCESS_EXPOSURE_ORDERS_SQL,
            SELECT_LIVE_PROCESS_UNPROVEN_FILLED_ORDER_EXISTS_SQL, SELECT_LIVE_SETTLEMENT_FILLS_SQL,
            SELECT_PENDING_LIVE_PROCESS_SETTLEMENT_EXISTS_SQL,
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
    fn manual_live_exit_uses_centralized_fee_math_and_actual_net_proceeds() {
        let (expected_fee, observed_fee, net_pnl) = manual_live_exit_economics(
            dec!(5),
            dec!(0.999),
            dec!(0.07),
            dec!(4.99466),
            dec!(4.75),
            dec!(0.016625),
            dec!(0.00001),
        )
        .unwrap();

        assert_eq!(expected_fee, dec!(0.00034965));
        assert_eq!(observed_fee, dec!(0.00034));
        assert_eq!(net_pnl, dec!(0.228035));
        assert!(manual_live_exit_economics(
            dec!(5),
            dec!(0.999),
            dec!(0.07),
            dec!(4.99),
            dec!(4.75),
            dec!(0.016625),
            dec!(0.00001),
        )
        .is_err());
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
    fn durable_fill_progress_supersedes_a_stale_execution_report() {
        assert!(durable_fill_state_supersedes_report(
            OrderState::PartiallyFilled,
            OrderState::Acknowledged
        ));
        assert!(durable_fill_state_supersedes_report(
            OrderState::Filled,
            OrderState::Acknowledged
        ));
        assert!(durable_fill_state_supersedes_report(
            OrderState::Filled,
            OrderState::PartiallyFilled
        ));

        assert!(!durable_fill_state_supersedes_report(
            OrderState::Acknowledged,
            OrderState::Filled
        ));
        assert!(!durable_fill_state_supersedes_report(
            OrderState::Filled,
            OrderState::Rejected
        ));
        assert!(!durable_fill_state_supersedes_report(
            OrderState::Cancelled,
            OrderState::Acknowledged
        ));
    }

    #[test]
    fn official_zero_payout_loss_is_valid_daily_live_pnl_evidence() {
        let credited_at = Utc::now();
        let settlement_id = Uuid::from_u128(31);
        let process_id = Uuid::from_u128(32);
        let run_id = Uuid::from_u128(33);
        let fill_id = Uuid::from_u128(34);
        let resolution_at = credited_at - Duration::seconds(5);
        let fill_ids = serde_json::json!([fill_id]);
        let evidence = serde_json::json!({
            "proof_type": "btc_official_zero_payout_loss",
            "evidence_version": "btc_live_zero_payout_settlement_v1",
            "recognition_kind": "official_resolution",
            "execution_mode": "live",
            "exchange_cash_credit_applied": false,
            "settlement_id": settlement_id,
            "process_id": process_id,
            "run_id": run_id,
            "order_id": "order",
            "market_id": "market",
            "token_id": "down-token",
            "fill_ids": fill_ids,
            "official_outcome": "up",
            "official_winning_token_id": "up-token",
            "official_resolution_received_at": resolution_at,
            "official_resolution_source": "clob_websocket",
            "filled_size": "5",
            "entry_notional": "4.2",
            "entry_fees": "0",
            "payout": "0",
            "net_pnl": "-4.2",
            "recognized_by_config_hash": "a".repeat(64),
        });
        let mut settlement = LiveDailySettlementRow {
            settlement_id,
            run_id,
            process_id,
            order_id: "order".to_string(),
            market_id: "market".to_string(),
            token_id: "down-token".to_string(),
            fill_ids,
            official_outcome: "up".to_string(),
            official_winning_token_id: "up-token".to_string(),
            official_resolution_received_at: resolution_at,
            official_resolution_source: "clob_websocket".to_string(),
            filled_size: dec!(5),
            entry_notional: dec!(4.2),
            entry_fees: Decimal::ZERO,
            payout: Decimal::ZERO,
            net_pnl: dec!(-4.2),
            credited_at: Some(credited_at),
            credit_attempts: 1,
            credit_evidence: evidence,
            created_at: resolution_at,
            updated_at: credited_at,
        };

        validate_live_daily_settlement(
            &settlement,
            credited_at - Duration::hours(1),
            credited_at + Duration::hours(1),
            credited_at,
        )
        .unwrap();

        settlement.official_resolution_received_at =
            credited_at + crate::execution::LIVE_EXTERNAL_EVENT_CLOCK_SKEW;
        settlement.credit_evidence["official_resolution_received_at"] =
            serde_json::json!(settlement.official_resolution_received_at);
        validate_live_daily_settlement(
            &settlement,
            credited_at - Duration::hours(1),
            credited_at + Duration::hours(1),
            credited_at + Duration::seconds(1),
        )
        .unwrap();

        settlement.official_resolution_received_at += Duration::microseconds(1);
        settlement.credit_evidence["official_resolution_received_at"] =
            serde_json::json!(settlement.official_resolution_received_at);
        assert!(validate_live_daily_settlement(
            &settlement,
            credited_at - Duration::hours(1),
            credited_at + Duration::hours(1),
            credited_at + Duration::seconds(1),
        )
        .is_err());
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
    fn legacy_signal_null_order_matches_post_cutover_deterministic_retry() {
        let incoming = OrderRequest {
            client_order_id: Uuid::from_u128(20),
            process_id: Some(Uuid::from_u128(21)),
            market_id: "market".to_string(),
            token_id: "token".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.40),
            size: dec!(2),
            metadata: serde_json::json!({
                "execution_intent": "entry",
                "reference_execution_guard": {
                    "guard_version": "btc_reference_execution_guard_v1",
                    "signal_id": null,
                    "evidence_sha256": "evidence",
                },
            }),
        };
        let mut persisted_payload = serde_json::to_value(&incoming).unwrap();
        persisted_payload["signal_id"] = serde_json::Value::Null;
        let persisted: OrderRequest = serde_json::from_value(persisted_payload).unwrap();

        assert_eq!(persisted.client_order_id, incoming.client_order_id);
        assert!(
            serde_json::to_value(&persisted)
                .unwrap()
                .get("signal_id")
                .is_none(),
            "retired top-level request identity must not be serialized again"
        );
        assert_eq!(persisted.metadata, incoming.metadata);
        assert!(order_request_identity_matches(&persisted, &incoming));
        assert!(order_request_result_matches(&persisted, &incoming));
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

    #[test]
    fn live_daily_loss_queries_are_process_scoped_bounded_and_index_aligned() {
        assert!(SELECT_PENDING_LIVE_PROCESS_SETTLEMENT_EXISTS_SQL.contains("process_id = $1"));
        assert!(
            SELECT_PENDING_LIVE_PROCESS_SETTLEMENT_EXISTS_SQL.contains("execution_mode = 'live'")
        );
        assert!(
            SELECT_PENDING_LIVE_PROCESS_SETTLEMENT_EXISTS_SQL.contains("credit_status = 'pending'")
        );
        assert!(SELECT_CREDITED_LIVE_PROCESS_SETTLEMENTS_SQL.contains("process_id = $1"));
        assert!(SELECT_CREDITED_LIVE_PROCESS_SETTLEMENTS_SQL.contains("execution_mode = 'live'"));
        assert!(SELECT_CREDITED_LIVE_PROCESS_SETTLEMENTS_SQL.contains("credit_status = 'credited'"));
        assert!(SELECT_CREDITED_LIVE_PROCESS_SETTLEMENTS_SQL.contains("credited_at >= $2"));
        assert!(SELECT_CREDITED_LIVE_PROCESS_SETTLEMENTS_SQL.contains("credited_at < $3"));
        assert!(SELECT_CREDITED_LIVE_PROCESS_SETTLEMENTS_SQL.contains("LIMIT $4"));
        assert!(SELECT_LIVE_SETTLEMENT_FILLS_SQL.contains("i.fill_id = ANY($1::uuid[])"));
        assert!(SELECT_LIVE_SETTLEMENT_FILLS_SQL.contains("f.timestamp_utc = i.timestamp_utc"));
    }

    #[test]
    fn live_exposure_queries_are_process_scoped_bounded_and_release_only_credited_redemption() {
        assert!(SELECT_LIVE_PROCESS_EXPOSURE_ORDERS_SQL.contains("process_id = $1"));
        assert!(SELECT_LIVE_PROCESS_EXPOSURE_ORDERS_SQL.contains("LIMIT $2"));
        assert!(SELECT_LIVE_PROCESS_EXPOSURE_ORDERS_SQL
            .contains("state NOT IN ('filled', 'cancelled', 'rejected', 'expired')"));
        assert!(SELECT_LIVE_PROCESS_EXPOSURE_FILLS_SQL.contains("f.process_id = $1"));
        assert!(SELECT_LIVE_PROCESS_EXPOSURE_FILLS_SQL.contains("LIMIT $2"));
        assert!(SELECT_LIVE_PROCESS_EXPOSURE_FILLS_SQL.contains("polymarket.fill_identities"));
        assert!(SELECT_LIVE_PROCESS_CROSS_OWNED_FILL_EXISTS_SQL.contains("o.process_id = $1"));
        assert!(SELECT_LIVE_PROCESS_CROSS_OWNED_FILL_EXISTS_SQL
            .contains("f.process_id IS DISTINCT FROM $1"));
        assert!(SELECT_LIVE_PROCESS_UNPROVEN_FILLED_ORDER_EXISTS_SQL
            .contains("o.state IN ('filled', 'partially_filled')"));
        assert!(
            SELECT_LIVE_PROCESS_EXPOSURE_FILLS_SQL.contains("settlement.execution_mode = 'live'")
        );
        assert!(SELECT_LIVE_PROCESS_EXPOSURE_FILLS_SQL
            .contains("settlement.credit_status = 'credited'"));
        assert!(!SELECT_LIVE_PROCESS_EXPOSURE_FILLS_SQL.contains("resolution"));
        assert!(!SELECT_LIVE_PROCESS_EXPOSURE_FILLS_SQL.contains("sell"));
    }

    #[test]
    fn requested_live_exposure_includes_full_notional_and_dynamic_fee() {
        let row = live_exposure_order_row(Uuid::from_u128(100), OrderState::Submitted, Utc::now());
        let order: OrderRecord = serde_json::from_value(row.raw_payload).unwrap();
        let (notional, fee) = live_requested_exposure(&order.request).unwrap();
        assert_eq!(notional, dec!(0.80));
        assert_eq!(fee, dec!(0.12));

        let mut malformed = order.request;
        malformed.metadata["dynamic_fee_rate"] = serde_json::json!("1.01");
        assert!(live_requested_exposure(&malformed).is_err());
    }

    #[test]
    fn conservative_live_exposure_counts_pending_plus_buy_fills_without_netting() {
        let process_id = Uuid::from_u128(200);
        let at = Utc::now();
        let pending = live_exposure_order_row(process_id, OrderState::PartiallyFilled, at);
        let fill = live_exposure_fill_row(&pending, process_id, at, dec!(1), dec!(0.01));
        let snapshot = build_live_process_exposure_snapshot(
            process_id,
            None,
            at + chrono::Duration::seconds(1),
            false,
            &[pending],
            &[fill],
        )
        .unwrap();

        assert_eq!(snapshot.pending_order_count, 1);
        assert_eq!(snapshot.buy_fill_count, 1);
        assert_eq!(snapshot.pending_requested_notional_usd, dec!(0.80));
        assert_eq!(snapshot.pending_requested_fees_usd, dec!(0.12));
        assert_eq!(snapshot.filled_buy_notional_usd, dec!(0.40));
        assert_eq!(snapshot.filled_buy_fees_usd, dec!(0.01));
        assert_eq!(snapshot.total_exposure_usd, dec!(1.33));
        assert_eq!(snapshot.exposed_market_ids, vec!["market".to_string()]);
    }

    #[test]
    fn live_fill_exposure_accepts_exchange_second_precision_before_local_order_creation() {
        let process_id = Uuid::from_u128(201);
        let fill_at = "2026-08-14T14:03:16Z"
            .parse::<chrono::DateTime<Utc>>()
            .unwrap();
        let order_created_at = fill_at + Duration::microseconds(406_916);
        let mut order = live_exposure_order_row(
            process_id,
            OrderState::Filled,
            order_created_at + Duration::seconds(1),
        );
        order.created_at = order_created_at;
        order.updated_at = fill_at;
        let fill = live_exposure_fill_row(&order, process_id, fill_at, dec!(2), Decimal::ZERO);

        validate_live_exposure_fill(&fill, process_id, order_created_at + Duration::seconds(1))
            .unwrap();
    }

    #[test]
    fn live_fill_exposure_rejects_order_update_outside_external_clock_bound() {
        let process_id = Uuid::from_u128(203);
        let at = Utc::now();
        let mut order = live_exposure_order_row(process_id, OrderState::Filled, at);
        order.updated_at = order.created_at
            - crate::execution::LIVE_EXTERNAL_EVENT_CLOCK_SKEW
            - Duration::microseconds(1);
        let fill =
            live_exposure_fill_row(&order, process_id, order.created_at, dec!(2), Decimal::ZERO);

        let error = validate_live_exposure_fill(&fill, process_id, at).unwrap_err();
        assert!(error.to_string().contains("order_update_creation_delta_ms"));
        assert!(error.to_string().contains("allowed_external_clock_skew_ms"));
    }

    #[test]
    fn live_fill_exposure_rejects_timestamp_outside_existing_reconciliation_window() {
        let process_id = Uuid::from_u128(202);
        let at = Utc::now();
        let order = live_exposure_order_row(process_id, OrderState::Filled, at);
        let fill = live_exposure_fill_row(
            &order,
            process_id,
            order.created_at
                - crate::execution::LIVE_FILL_RECONCILIATION_SKEW
                - Duration::microseconds(1),
            dec!(2),
            Decimal::ZERO,
        );

        let error =
            validate_live_exposure_fill(&fill, process_id, order.updated_at + Duration::seconds(1))
                .unwrap_err();
        assert!(error.to_string().contains("fill_order_delta_ms"));
        assert!(error.to_string().contains("allowed_fill_order_skew_ms"));
    }

    #[test]
    fn filled_buy_exposure_remains_when_order_is_terminal_and_settlement_is_external() {
        let process_id = Uuid::from_u128(300);
        let at = Utc::now();
        let filled_order = live_exposure_order_row(process_id, OrderState::Filled, at);
        let fill = live_exposure_fill_row(&filled_order, process_id, at, dec!(2), dec!(0.02));
        let snapshot = build_live_process_exposure_snapshot(
            process_id,
            None,
            at + chrono::Duration::seconds(1),
            true,
            &[],
            &[fill],
        )
        .unwrap();

        assert_eq!(snapshot.pending_order_count, 0);
        assert_eq!(snapshot.buy_fill_count, 1);
        assert!(snapshot.has_unredeemed_settlement);
        assert_eq!(snapshot.total_exposure_usd, dec!(0.82));
        assert_eq!(snapshot.exposed_market_count(), 1);
    }

    #[test]
    fn live_exposure_excludes_only_the_adjacent_pending_request_and_fails_cross_process() {
        let process_id = Uuid::from_u128(400);
        let at = Utc::now();
        let pending = live_exposure_order_row(process_id, OrderState::Submitted, at);
        let snapshot = build_live_process_exposure_snapshot(
            process_id,
            Some(pending.client_order_id),
            at + chrono::Duration::seconds(1),
            false,
            std::slice::from_ref(&pending),
            &[],
        )
        .unwrap();
        assert_eq!(snapshot.total_exposure_usd, Decimal::ZERO);

        let mut fill = live_exposure_fill_row(&pending, process_id, at, dec!(1), Decimal::ZERO);
        fill.identity_process_id = Some(Uuid::from_u128(401));
        let error = build_live_process_exposure_snapshot(
            process_id,
            None,
            at + chrono::Duration::seconds(1),
            false,
            &[],
            &[fill],
        )
        .unwrap_err();
        assert!(error.to_string().contains("process identity lineage"));
    }

    fn live_exposure_order_row(
        process_id: Uuid,
        state: OrderState,
        at: chrono::DateTime<Utc>,
    ) -> LiveExposureOrderRow {
        let request = OrderRequest {
            client_order_id: Uuid::from_u128(500),
            process_id: Some(process_id),
            market_id: "market".to_string(),
            token_id: "token".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Fok,
            price: dec!(0.40),
            size: dec!(2),
            metadata: serde_json::json!({
                "execution_intent": "entry",
                "dynamic_fee_rate": dec!(0.25),
            }),
        };
        let order = OrderRecord {
            order_id: "venue-order".to_string(),
            request,
            state,
            created_at: at - chrono::Duration::seconds(1),
            updated_at: at,
        };
        let raw_payload = serde_json::to_value(&order).unwrap();
        LiveExposureOrderRow {
            order_id: order.order_id,
            client_order_id: order.request.client_order_id,
            process_id: order.request.process_id,
            market_id: order.request.market_id,
            token_id: order.request.token_id,
            side: "buy".to_string(),
            order_type: "fok".to_string(),
            price: order.request.price,
            size: order.request.size,
            state: match state {
                OrderState::Created => "created",
                OrderState::Submitted => "submitted",
                OrderState::Acknowledged => "acknowledged",
                OrderState::PartiallyFilled => "partially_filled",
                OrderState::Filled => "filled",
                OrderState::CancelRequested => "cancel_requested",
                OrderState::Cancelled => "cancelled",
                OrderState::Rejected => "rejected",
                OrderState::Expired => "expired",
                OrderState::Unknown => "unknown",
            }
            .to_string(),
            raw_payload_bytes: raw_payload.to_string().len() as i64,
            raw_payload,
            created_at: order.created_at,
            updated_at: order.updated_at,
        }
    }

    fn live_exposure_fill_row(
        order: &LiveExposureOrderRow,
        process_id: Uuid,
        at: chrono::DateTime<Utc>,
        size: Decimal,
        fee: Decimal,
    ) -> LiveExposureFillRow {
        LiveExposureFillRow {
            fill_id: Uuid::from_u128(600),
            fill_process_id: Some(process_id),
            fill_order_id: order.order_id.clone(),
            fill_token_id: order.token_id.clone(),
            filled_at: at,
            fill_price: dec!(0.40),
            fill_size: size,
            fill_fee: fee,
            fill_source: "live".to_string(),
            identity_process_id: Some(process_id),
            identity_timestamp_utc: Some(at),
            order_process_id: Some(process_id),
            order_client_order_id: Some(order.client_order_id),
            order_market_id: Some(order.market_id.clone()),
            order_token_id: Some(order.token_id.clone()),
            order_side: Some(order.side.clone()),
            order_type: Some(order.order_type.clone()),
            order_price: Some(order.price),
            order_size: Some(order.size),
            order_state: Some(order.state.clone()),
            order_raw_payload: Some(order.raw_payload.clone()),
            order_raw_payload_bytes: Some(order.raw_payload_bytes),
            order_created_at: Some(order.created_at),
            order_updated_at: Some(order.updated_at),
        }
    }
}
