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
    copytrade::{expectancy_price_bucket, CopyTradeExpectancyInput},
    events::ServiceEvent,
    execution::live::LiveVenueEvent,
    execution::OrderPlanReport,
    models::{
        BackfillJob, BackfillJobStatus, BacktestRun, ConversionRequest, ConversionResult,
        CopyTradeBacktestResult, CopyTradeBacktestRun, CopyTradeSignal, DataApiClosedPosition,
        EffectiveExpectancyFlowProcessConfig, ExpectancyFlowCell, ExpectancyFlowRecomputeReport,
        ExpectancyFlowWalletCell, FillRecord, GammaMarketMetadata, Market, OrderRecord,
        OrderRequest, OrderState, OutcomeToken, SignalCandidate, TradingProcess,
        TradingProcessConfig, WalletPerformance, WalletScore, WalletScoreCalibrationSnapshot,
        WalletScoreRefreshJob, WalletScoreRefreshStatus, WalletSegmentPerformance,
        WalletTradeTaxonomyCandidate, WalletTradeTaxonomyUpdate, WhalePollCheckpoint, WhaleTrade,
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

const MARK_OPEN_TRADE_POSITIONS_SQL: &str = r#"
WITH open_positions AS MATERIALIZED (
  SELECT
    p.position_id,
    p.process_id,
    p.source_signal_id,
    p.token_id,
    p.side,
    p.entry_price,
    p.open_size,
    p.entry_notional,
    p.entry_timestamp,
    p.realized_pnl,
    p.latest_mark_timestamp
  FROM polymarket.trade_positions p
  WHERE p.status IN ('open', 'partially_closed')
    AND p.open_size > 0
    AND ($1::uuid IS NULL OR p.process_id IS NOT DISTINCT FROM $1::uuid)
),
open_tokens AS MATERIALIZED (
  SELECT DISTINCT token_id
  FROM open_positions
),
latest_orderbook AS MATERIALIZED (
  SELECT
    t.token_id,
    b.timestamp_utc AS source_timestamp,
    CASE
      WHEN b.best_bid IS NOT NULL AND b.best_ask IS NOT NULL THEN (b.best_bid + b.best_ask) / 2
      WHEN b.best_bid IS NOT NULL THEN b.best_bid
      ELSE b.best_ask
    END AS mark_price,
    CASE
      WHEN b.best_bid IS NOT NULL AND b.best_ask IS NOT NULL THEN 'clob_mid'
      WHEN b.best_bid IS NOT NULL THEN 'clob_bid'
      ELSE 'clob_ask'
    END AS mark_source,
    jsonb_build_object(
      'source_timestamp', b.timestamp_utc,
      'best_bid', b.best_bid,
      'best_ask', b.best_ask,
      'mark_basis', 'orderbook_snapshot'
    ) AS metadata
  FROM open_tokens t
  JOIN LATERAL (
    SELECT timestamp_utc, best_bid, best_ask
    FROM polymarket.orderbook_snapshots b
    WHERE b.token_id = t.token_id
      AND b.timestamp_utc >= now() - interval '15 minutes'
      AND (b.best_bid IS NOT NULL OR b.best_ask IS NOT NULL)
    ORDER BY b.timestamp_utc DESC
    LIMIT 1
  ) b ON true
),
latest_wallet_trade AS MATERIALIZED (
  SELECT
    t.token_id,
    wt.timestamp_utc AS source_timestamp,
    wt.price AS mark_price,
    'data_api_trade' AS mark_source,
    jsonb_build_object(
      'source_timestamp', wt.timestamp_utc,
      'mark_basis', 'wallet_trade_fallback',
      'wallet_trade_id', wt.trade_id
    ) AS metadata
  FROM open_tokens t
  JOIN LATERAL (
    SELECT trade_id, timestamp_utc, price
    FROM polymarket.wallet_trades wt
    WHERE wt.asset = t.token_id
    ORDER BY wt.timestamp_utc DESC
    LIMIT 1
  ) wt ON true
),
marks AS (
  SELECT
    p.position_id,
    p.process_id,
    p.source_signal_id,
    p.side,
    p.entry_price,
    p.open_size,
    p.entry_notional,
    COALESCE(orderbook.mark_price, wallet.mark_price) AS mark_price,
    COALESCE(orderbook.source_timestamp, wallet.source_timestamp) AS source_timestamp,
    COALESCE(orderbook.mark_source, wallet.mark_source) AS mark_source,
    COALESCE(orderbook.metadata, wallet.metadata) AS metadata,
    GREATEST(
      0,
      floor(
        extract(epoch from (now() - COALESCE(orderbook.source_timestamp, wallet.source_timestamp)))
          * 1000
      )
    )::bigint AS mark_age_ms
  FROM open_positions p
  LEFT JOIN latest_orderbook orderbook
    ON orderbook.token_id = p.token_id
   AND orderbook.source_timestamp >= p.entry_timestamp
  LEFT JOIN latest_wallet_trade wallet
    ON wallet.token_id = p.token_id
   AND wallet.source_timestamp >= p.entry_timestamp
  WHERE COALESCE(orderbook.source_timestamp, wallet.source_timestamp) IS NOT NULL
    AND (
      p.latest_mark_timestamp IS NULL
      OR COALESCE(orderbook.source_timestamp, wallet.source_timestamp) > p.latest_mark_timestamp
    )
),
prepared AS (
  SELECT
    *,
    CASE
      WHEN side = 'buy' THEN open_size * (mark_price - entry_price)
      ELSE open_size * (entry_price - mark_price)
    END AS gross_unrealized_pnl
  FROM marks
),
inserted AS (
  INSERT INTO polymarket.trade_marks (
    position_id, process_id, source_signal_id, timestamp_utc, mark_price, mark_source,
    mark_age_ms, gross_unrealized_pnl, net_unrealized_pnl, roi, metadata
  )
  SELECT
    position_id,
    process_id,
    source_signal_id,
    source_timestamp,
    mark_price,
    mark_source,
    mark_age_ms,
    gross_unrealized_pnl,
    gross_unrealized_pnl,
    CASE WHEN entry_notional > 0 THEN gross_unrealized_pnl / entry_notional ELSE 0 END,
    metadata || jsonb_build_object('written_at', now())
  FROM prepared
  RETURNING position_id, mark_price, net_unrealized_pnl, timestamp_utc
)
UPDATE polymarket.trade_positions p
SET
  latest_mark_price = i.mark_price,
  latest_mark_timestamp = i.timestamp_utc,
  unrealized_pnl = i.net_unrealized_pnl,
  roi = CASE
    WHEN p.entry_notional > 0 THEN (p.realized_pnl + i.net_unrealized_pnl) / p.entry_notional
    ELSE 0
  END,
  updated_at = now()
FROM inserted i
WHERE p.position_id = i.position_id
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

#[derive(Debug, FromRow)]
struct BacktestRunRow {
    backtest_run_id: Uuid,
    status: String,
    range_start: DateTime<Utc>,
    range_end: DateTime<Utc>,
    warmup_start: DateTime<Utc>,
    lookback_days: i32,
    warmup_days: i32,
    source_process_ids: Vec<Uuid>,
    backtest_process_ids: Vec<Uuid>,
    request: serde_json::Value,
    summary: serde_json::Value,
    error: Option<String>,
    started_at: Option<DateTime<Utc>>,
    completed_at: Option<DateTime<Utc>>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, FromRow)]
pub struct WhaleLedTradeExitCandidate {
    pub process_id: Option<Uuid>,
    pub position_id: Uuid,
    pub source_signal_id: Uuid,
    pub proxy_wallet: Option<String>,
    pub market_id: Option<String>,
    pub token_id: String,
    pub side: String,
    pub entry_price: Decimal,
    pub entry_size: Decimal,
    pub open_size: Decimal,
    pub entry_fee: Decimal,
    pub entry_notional: Decimal,
    pub exit_source_trade_id: Uuid,
    pub exit_timestamp: DateTime<Utc>,
    pub reference_exit_price: Decimal,
    pub exit_size: Decimal,
}

#[derive(Debug, Clone, FromRow)]
pub struct TakeProfitTradeExitCandidate {
    pub exit_purpose: String,
    pub process_id: Option<Uuid>,
    pub position_id: Uuid,
    pub source_signal_id: Uuid,
    pub proxy_wallet: Option<String>,
    pub market_id: Option<String>,
    pub token_id: String,
    pub side: String,
    pub entry_price: Decimal,
    pub entry_size: Decimal,
    pub open_size: Decimal,
    pub entry_fee: Decimal,
    pub entry_notional: Decimal,
    pub exit_source_trade_id: Uuid,
    pub exit_timestamp: DateTime<Utc>,
    pub reference_exit_price: Decimal,
    pub order_limit_price: Decimal,
    pub marketable_exit_price: Option<Decimal>,
    pub best_bid: Option<Decimal>,
    pub best_ask: Option<Decimal>,
    pub exit_size: Decimal,
    pub trigger_roi: Decimal,
    pub threshold_roi: Decimal,
    pub take_profit_roi: Decimal,
    pub latest_mark_timestamp: DateTime<Utc>,
    pub max_exit_slippage_bps: Decimal,
    pub exit_pricing_mode: String,
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

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct AccountPositionMismatch {
    pub token_id: String,
    pub db_open_size: Decimal,
    pub account_size: Decimal,
    pub delta_size: Decimal,
    pub mismatch_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ManualExitReport {
    pub exits_detected: u64,
    pub exits_applied: u64,
    pub exit_size_applied: Decimal,
    pub unmatched_trades: u64,
}

#[derive(Debug, Clone, FromRow)]
struct AccountTradeExitRow {
    account_trade_id: Uuid,
    account_address: String,
    token_id: String,
    side: String,
    price: Decimal,
    size: Decimal,
    applied_exit_size: Decimal,
    timestamp_utc: DateTime<Utc>,
    transaction_hash: Option<String>,
    venue_order_id: Option<String>,
    venue_trade_id: Option<String>,
    source: String,
    raw_payload: serde_json::Value,
}

#[derive(Debug, Clone, FromRow)]
struct AccountExitPositionRow {
    process_id: Option<Uuid>,
    position_id: Uuid,
    source_signal_id: Uuid,
    side: String,
    entry_price: Decimal,
    entry_size: Decimal,
    open_size: Decimal,
    entry_fee: Decimal,
    entry_notional: Decimal,
}

#[derive(Debug, Clone, FromRow)]
struct AccountAdjustmentPositionRow {
    process_id: Option<Uuid>,
    position_id: Uuid,
    source_signal_id: Uuid,
    side: String,
    entry_price: Decimal,
    entry_size: Decimal,
    open_size: Decimal,
    entry_fee: Decimal,
    entry_notional: Decimal,
    snapshot_at: DateTime<Utc>,
    snapshot_size: Decimal,
    exit_price: Decimal,
    price_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingProcessResetReport {
    pub process_id: Uuid,
    pub process_name: String,
    pub orders_deleted: u64,
    pub fills_deleted: u64,
    pub signal_candidates_deleted: u64,
    pub copy_trade_signals_deleted: u64,
    pub trade_marks_deleted: u64,
    pub trade_exits_deleted: u64,
    pub trade_positions_deleted: u64,
    pub wallet_performance_deleted: u64,
    pub expectancy_flow_cells_deleted: u64,
    pub expectancy_flow_wallet_cells_deleted: u64,
    pub process_events_deleted: u64,
    pub copy_trade_backtest_results_deleted: u64,
    pub copy_trade_backtest_runs_deleted: u64,
    pub copy_trade_backtests_deleted: u64,
    pub backfill_job_events_deleted: u64,
    pub backfill_jobs_deleted: u64,
    pub whale_poll_checkpoints_deleted: u64,
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

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct OpenMarkToken {
    pub process_id: Option<Uuid>,
    pub token_id: String,
    pub market_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TradeMarkSourceFailure {
    pub process_id: Option<Uuid>,
    pub position_id: Option<Uuid>,
    pub token_id: String,
    pub market_id: Option<String>,
    pub failure_source: String,
    pub failure_reason: String,
    pub metadata: serde_json::Value,
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

    pub async fn ensure_default_trading_process(
        &self,
        config: TradingProcessConfig,
    ) -> Result<TradingProcess> {
        let process_id = Uuid::new_v5(
            &Uuid::NAMESPACE_URL,
            b"polymarket-bot/default-env-copy-trade-process",
        );
        let config_value = serde_json::to_value(config)?;
        let row = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            INSERT INTO polymarket.trading_processes (
              process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at
            )
            VALUES (
              $1, 'default-env-copy-trade', 'copy_trade', 'default', 'default-env-copy-trade', 'running', true, $2, $3,
              now(), now(), now()
            )
            ON CONFLICT (process_id) DO UPDATE SET
              process_scope = 'default',
              process_key = 'default-env-copy-trade',
              updated_at = now()
            RETURNING process_id, name, process_type, process_scope, process_key, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            "#,
        )
        .bind(process_id)
        .bind(config_value)
        .bind(serde_json::json!({"source": "env_default"}))
        .fetch_one(&self.pool)
        .await
        .context("failed to ensure default trading process")?;
        trading_process_from_row(row)
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

    pub async fn process_open_notional(&self, process_id: Uuid) -> Result<Decimal> {
        let notional = sqlx::query_scalar::<_, Decimal>(
            r#"
            SELECT COALESCE(sum(open_size * entry_price), 0)
            FROM polymarket.trade_positions
            WHERE process_id = $1
              AND status IN ('open', 'partially_closed')
              AND open_size > 0
            "#,
        )
        .bind(process_id)
        .fetch_one(&self.pool)
        .await
        .context("failed to fetch process open notional")?;
        Ok(notional)
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
              FROM polymarket.copy_trade_signals
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
            positions AS (
              SELECT count(*)::bigint AS total,
                count(*) FILTER (WHERE status IN ('open', 'partially_closed'))::bigint AS open,
                count(*) FILTER (WHERE status IN ('closed', 'resolved'))::bigint AS closed,
                count(*) FILTER (WHERE status IN ('closed', 'resolved') AND realized_pnl > 0)::bigint AS profitable_closed,
                count(*) FILTER (WHERE status IN ('open', 'partially_closed') AND unrealized_pnl > 0)::bigint AS profitable_open,
                COALESCE(sum(entry_notional), 0) AS entry_notional,
                COALESCE(sum(realized_pnl), 0) AS realized_pnl,
                COALESCE(sum(unrealized_pnl) FILTER (WHERE status IN ('open', 'partially_closed')), 0) AS unrealized_pnl,
                max(updated_at) AS last_position_update_at
              FROM polymarket.trade_positions
              WHERE process_id = $1
            ),
            marks AS (
              SELECT count(*)::bigint AS total, max(timestamp_utc) AS last_mark_at
              FROM polymarket.trade_marks
              WHERE process_id = $1
            ),
            exits AS (
              SELECT count(*)::bigint AS total, max(timestamp_utc) AS last_exit_at
              FROM polymarket.trade_exits
              WHERE process_id = $1
            ),
            jobs AS (
              SELECT job_id, status, requested_at, started_at, completed_at, error
              FROM polymarket.backfill_jobs
              WHERE request->>'process_id' = $1::text
              ORDER BY requested_at DESC
              LIMIT 1
            ),
            poll_checkpoint AS (
              SELECT checkpoint_name, last_polled_at, next_cursor, last_trade_timestamp_utc,
                last_trade_id, pages_seen, trades_seen, state
              FROM polymarket.whale_poll_checkpoints
              WHERE checkpoint_name = $1::text
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
              'positions', jsonb_build_object(
                'total', positions.total,
                'open', positions.open,
                'closed', positions.closed,
                'profitable_closed', positions.profitable_closed,
                'profitable_open', positions.profitable_open,
                'entry_notional', positions.entry_notional,
                'realized_pnl', positions.realized_pnl,
                'unrealized_pnl', positions.unrealized_pnl,
                'total_pnl', positions.realized_pnl + positions.unrealized_pnl,
                'roi', CASE
                  WHEN positions.entry_notional > 0
                  THEN (positions.realized_pnl + positions.unrealized_pnl) / positions.entry_notional
                  ELSE 0
                END,
                'last_position_update_at', positions.last_position_update_at
              ),
              'marks', jsonb_build_object(
                'total', marks.total,
                'last_mark_at', marks.last_mark_at
              ),
              'exits', jsonb_build_object(
                'total', exits.total,
                'last_exit_at', exits.last_exit_at
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
              END,
              'last_poll', CASE
                WHEN poll_checkpoint.checkpoint_name IS NULL THEN NULL
                ELSE jsonb_build_object(
                  'last_polled_at', poll_checkpoint.last_polled_at,
                  'next_cursor', poll_checkpoint.next_cursor,
                  'last_trade_timestamp_utc', poll_checkpoint.last_trade_timestamp_utc,
                  'last_trade_id', poll_checkpoint.last_trade_id,
                  'pages_seen', poll_checkpoint.pages_seen,
                  'trades_seen', poll_checkpoint.trades_seen,
                  'state', poll_checkpoint.state
                )
              END
            )
            FROM process
            CROSS JOIN signals
            CROSS JOIN orders
            CROSS JOIN order_states
            CROSS JOIN fills
            CROSS JOIN positions
            CROSS JOIN marks
            CROSS JOIN exits
            LEFT JOIN jobs ON true
            LEFT JOIN poll_checkpoint ON true
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

        sqlx::query("DROP TABLE IF EXISTS pg_temp.reset_process_positions")
            .execute(&mut *tx)
            .await
            .context("failed to drop reset positions temp table")?;
        sqlx::query("DROP TABLE IF EXISTS pg_temp.reset_process_signals")
            .execute(&mut *tx)
            .await
            .context("failed to drop reset signals temp table")?;
        sqlx::query("DROP TABLE IF EXISTS pg_temp.reset_process_orders")
            .execute(&mut *tx)
            .await
            .context("failed to drop reset orders temp table")?;
        sqlx::query("DROP TABLE IF EXISTS pg_temp.reset_process_jobs")
            .execute(&mut *tx)
            .await
            .context("failed to drop reset jobs temp table")?;
        sqlx::query("DROP TABLE IF EXISTS pg_temp.reset_process_backtests")
            .execute(&mut *tx)
            .await
            .context("failed to drop reset backtests temp table")?;
        sqlx::query(
            r#"
            CREATE TEMP TABLE reset_process_signals ON COMMIT DROP AS
            SELECT signal_id
            FROM polymarket.signal_candidates
            WHERE process_id = $1
            UNION
            SELECT signal_id
            FROM polymarket.copy_trade_signals
            WHERE process_id = $1
            UNION
            SELECT source_signal_id
            FROM polymarket.trade_positions
            WHERE process_id IS NULL
              AND execution_mode = 'sim'
            "#,
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to stage reset signals")?;
        sqlx::query(
            r#"
            CREATE TEMP TABLE reset_process_positions ON COMMIT DROP AS
            SELECT position_id
            FROM polymarket.trade_positions
            WHERE process_id = $1
               OR (
                 process_id IS NULL
                 AND execution_mode = 'sim'
               )
               OR (
                 process_id IS NULL
                 AND source_signal_id IN (SELECT signal_id FROM reset_process_signals)
               )
            "#,
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to stage reset positions")?;
        sqlx::query(
            r#"
            CREATE TEMP TABLE reset_process_orders ON COMMIT DROP AS
            SELECT DISTINCT o.order_id
            FROM polymarket.orders o
            WHERE o.process_id = $1
               OR o.raw_payload #>> '{request,process_id}' = $1::text
               OR o.raw_payload #>> '{request,metadata,process_id}' = $1::text
               OR (
                 o.raw_payload #>> '{request,signal_id}' ~* '^[0-9a-f-]{36}$'
                 AND (o.raw_payload #>> '{request,signal_id}')::uuid IN (
                   SELECT signal_id FROM reset_process_signals
                 )
               )
               OR EXISTS (
                 SELECT 1
                 FROM reset_process_positions p
                 WHERE o.raw_payload #>> '{request,metadata,position_id}' = p.position_id::text
               )
               OR EXISTS (
                 SELECT 1
                 FROM polymarket.trade_exits te
                 JOIN reset_process_positions p ON p.position_id = te.position_id
                 WHERE te.metadata #>> '{close_order_id}' = o.order_id
               )
            "#,
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to stage reset orders")?;
        sqlx::query(
            r#"
            CREATE TEMP TABLE reset_process_jobs ON COMMIT DROP AS
            SELECT job_id
            FROM polymarket.backfill_jobs
            WHERE request ->> 'process_id' = $1::text
            "#,
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to stage reset jobs")?;
        sqlx::query(
            r#"
            CREATE TEMP TABLE reset_process_backtests ON COMMIT DROP AS
            SELECT backtest_id
            FROM polymarket.copy_trade_backtest_runs
            WHERE job_id IN (SELECT job_id FROM reset_process_jobs)
               OR config #>> '{request,process_id}' = $1::text
               OR config ->> 'process_id' = $1::text
            "#,
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to stage reset backtests")?;

        let copy_trade_backtest_results_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.copy_trade_backtest_results r
            WHERE EXISTS (
              SELECT 1 FROM reset_process_backtests b WHERE b.backtest_id = r.backtest_id
            )
            "#,
        )
        .execute(&mut *tx)
        .await
        .context("failed to delete reset copy trade backtest results")?
        .rows_affected();
        let copy_trade_backtest_runs_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.copy_trade_backtest_runs r
            WHERE EXISTS (
              SELECT 1 FROM reset_process_backtests b WHERE b.backtest_id = r.backtest_id
            )
            "#,
        )
        .execute(&mut *tx)
        .await
        .context("failed to delete reset copy trade backtest runs")?
        .rows_affected();
        let copy_trade_backtests_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.copy_trade_backtests b
            WHERE b.job_id IN (SELECT job_id FROM reset_process_jobs)
               OR b.config #>> '{request,process_id}' = $1::text
               OR b.config ->> 'process_id' = $1::text
            "#,
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to delete reset copy trade backtests")?
        .rows_affected();
        let backfill_job_events_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.backfill_job_events e
            WHERE EXISTS (
              SELECT 1 FROM reset_process_jobs j WHERE j.job_id = e.job_id
            )
            "#,
        )
        .execute(&mut *tx)
        .await
        .context("failed to delete reset backfill job events")?
        .rows_affected();

        let trade_marks_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.trade_marks tm
            WHERE tm.process_id = $1
               OR EXISTS (
                 SELECT 1 FROM reset_process_positions p WHERE p.position_id = tm.position_id
               )
            "#,
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to delete reset trade marks")?
        .rows_affected();
        let trade_exits_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.trade_exits te
            WHERE te.process_id = $1
               OR EXISTS (
                 SELECT 1 FROM reset_process_positions p WHERE p.position_id = te.position_id
               )
               OR te.metadata #>> '{process_id}' = $1::text
            "#,
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to delete reset trade exits")?
        .rows_affected();
        let fills_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.fills f
            WHERE f.process_id = $1
               OR f.raw_payload ->> 'process_id' = $1::text
               OR EXISTS (
                 SELECT 1 FROM reset_process_orders o WHERE o.order_id = f.order_id
               )
               OR EXISTS (
                 SELECT 1 FROM reset_process_orders o WHERE o.order_id = f.raw_payload ->> 'order_id'
               )
            "#,
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to delete reset fills")?
        .rows_affected();
        let orders_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.orders o
            WHERE EXISTS (
              SELECT 1 FROM reset_process_orders staged WHERE staged.order_id = o.order_id
            )
            "#,
        )
        .execute(&mut *tx)
        .await
        .context("failed to delete reset orders")?
        .rows_affected();
        let trade_positions_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.trade_positions p
            WHERE EXISTS (
              SELECT 1 FROM reset_process_positions staged WHERE staged.position_id = p.position_id
            )
            "#,
        )
        .execute(&mut *tx)
        .await
        .context("failed to delete reset trade positions")?
        .rows_affected();
        let copy_trade_signals_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.copy_trade_signals
            WHERE process_id = $1
               OR (
                 process_id IS NULL
                 AND signal_id IN (SELECT signal_id FROM reset_process_signals)
               )
            "#,
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to delete reset copy trade signals")?
        .rows_affected();
        let signal_candidates_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.signal_candidates
            WHERE process_id = $1
               OR (
                 process_id IS NULL
                 AND signal_id IN (SELECT signal_id FROM reset_process_signals)
               )
            "#,
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to delete reset signal candidates")?
        .rows_affected();
        let wallet_performance_deleted =
            sqlx::query("DELETE FROM polymarket.wallet_trade_performance WHERE process_id = $1")
                .bind(process_id)
                .execute(&mut *tx)
                .await
                .context("failed to delete reset wallet performance")?
                .rows_affected();
        let expectancy_flow_wallet_cells_deleted = sqlx::query(
            "DELETE FROM polymarket.expectancy_flow_wallet_cells WHERE process_id = $1",
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to delete reset expectancy flow wallet cells")?
        .rows_affected();
        let expectancy_flow_cells_deleted =
            sqlx::query("DELETE FROM polymarket.expectancy_flow_cells WHERE process_id = $1")
                .bind(process_id)
                .execute(&mut *tx)
                .await
                .context("failed to delete reset expectancy flow cells")?
                .rows_affected();
        let process_events_deleted =
            sqlx::query("DELETE FROM polymarket.trading_process_events WHERE process_id = $1")
                .bind(process_id)
                .execute(&mut *tx)
                .await
                .context("failed to delete reset process events")?
                .rows_affected();
        let backfill_jobs_deleted = sqlx::query(
            r#"
            DELETE FROM polymarket.backfill_jobs j
            WHERE EXISTS (
              SELECT 1 FROM reset_process_jobs staged WHERE staged.job_id = j.job_id
            )
            "#,
        )
        .execute(&mut *tx)
        .await
        .context("failed to delete reset backfill jobs")?
        .rows_affected();
        let whale_poll_checkpoints_deleted = sqlx::query(
            "DELETE FROM polymarket.whale_poll_checkpoints WHERE checkpoint_name = $1::text",
        )
        .bind(process_id)
        .execute(&mut *tx)
        .await
        .context("failed to delete reset whale poll checkpoint")?
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
            copy_trade_signals_deleted,
            trade_marks_deleted,
            trade_exits_deleted,
            trade_positions_deleted,
            wallet_performance_deleted,
            expectancy_flow_cells_deleted,
            expectancy_flow_wallet_cells_deleted,
            process_events_deleted,
            copy_trade_backtest_results_deleted,
            copy_trade_backtest_runs_deleted,
            copy_trade_backtests_deleted,
            backfill_job_events_deleted,
            backfill_jobs_deleted,
            whale_poll_checkpoints_deleted,
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

    pub async fn open_live_trade_position_tokens(
        &self,
        token_id: Option<&str>,
    ) -> Result<Vec<String>> {
        let rows = sqlx::query_scalar::<_, String>(
            r#"
            SELECT DISTINCT token_id
            FROM polymarket.trade_positions
            WHERE is_live_capital = true
              AND status IN ('open', 'partially_closed')
              AND open_size > 0
              AND ($1::text IS NULL OR token_id = $1)
            ORDER BY token_id
            "#,
        )
        .bind(token_id)
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch open live trade position tokens")?;
        Ok(rows)
    }

    pub async fn preview_manual_account_exits(
        &self,
        account_address: &str,
        token_id: Option<&str>,
    ) -> Result<ManualExitReport> {
        self.reconcile_manual_account_exits(account_address, token_id, "manual_ui_exit", true)
            .await
    }

    pub async fn apply_manual_account_exits(
        &self,
        account_address: &str,
        token_id: Option<&str>,
        exit_type: &str,
    ) -> Result<ManualExitReport> {
        self.reconcile_manual_account_exits(account_address, token_id, exit_type, false)
            .await
    }

    async fn reconcile_manual_account_exits(
        &self,
        account_address: &str,
        token_id: Option<&str>,
        exit_type: &str,
        dry_run: bool,
    ) -> Result<ManualExitReport> {
        let trades = self
            .unapplied_account_exit_trades(account_address, token_id)
            .await?;
        let mut report = ManualExitReport::default();
        for trade in trades {
            let mut remaining = (trade.size - trade.applied_exit_size).max(Decimal::ZERO);
            if remaining <= Decimal::ZERO {
                continue;
            }
            let positions = self.open_positions_for_account_exit(&trade).await?;
            if positions.is_empty() {
                report.unmatched_trades += 1;
                continue;
            }
            for position in positions {
                if remaining <= Decimal::ZERO {
                    break;
                }
                let exit_size = remaining.min(position.open_size);
                if exit_size <= Decimal::ZERO {
                    continue;
                }
                report.exits_detected += 1;
                report.exit_size_applied += exit_size;
                if !dry_run {
                    let applied = self
                        .apply_manual_account_exit(&trade, &position, exit_size, exit_type)
                        .await?;
                    if applied {
                        report.exits_applied += 1;
                        remaining -= exit_size;
                    }
                }
            }
            if remaining > Decimal::ZERO {
                report.unmatched_trades += 1;
            }
        }
        Ok(report)
    }

    async fn unapplied_account_exit_trades(
        &self,
        account_address: &str,
        token_id: Option<&str>,
    ) -> Result<Vec<AccountTradeExitRow>> {
        let rows = sqlx::query_as::<_, AccountTradeExitRow>(
            r#"
            SELECT account_trade_id, account_address, token_id, side, price, size,
              applied_exit_size, timestamp_utc, transaction_hash, venue_order_id,
              venue_trade_id, source, raw_payload
            FROM polymarket.account_trades
            WHERE account_address = $1
              AND applied_exit_size < size
              AND source IN ('data_api', 'manual_backfill', 'poll')
              AND ($2::text IS NULL OR token_id = $2)
            ORDER BY timestamp_utc ASC, created_at ASC
            LIMIT 1000
            "#,
        )
        .bind(account_address)
        .bind(token_id)
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch unapplied account exit trades")?;
        Ok(rows)
    }

    async fn open_positions_for_account_exit(
        &self,
        trade: &AccountTradeExitRow,
    ) -> Result<Vec<AccountExitPositionRow>> {
        let rows = sqlx::query_as::<_, AccountExitPositionRow>(
            r#"
            SELECT process_id, position_id, source_signal_id, side, entry_price, entry_size,
              open_size, entry_fee, entry_notional
            FROM polymarket.trade_positions
            WHERE is_live_capital = true
              AND token_id = $1
              AND status IN ('open', 'partially_closed')
              AND open_size > 0
              AND entry_timestamp <= $2
              AND (
                (side = 'buy' AND $3 = 'sell')
                OR (side = 'sell' AND $3 = 'buy')
              )
              AND NOT EXISTS (
                SELECT 1
                FROM polymarket.trade_exits e
                WHERE e.position_id = polymarket.trade_positions.position_id
                  AND e.exit_source_trade_id = $4
                  AND e.is_synthetic = false
              )
            ORDER BY entry_timestamp ASC, created_at ASC
            "#,
        )
        .bind(&trade.token_id)
        .bind(trade.timestamp_utc)
        .bind(&trade.side)
        .bind(trade.account_trade_id)
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch open positions for account exit")?;
        Ok(rows)
    }

    async fn apply_manual_account_exit(
        &self,
        trade: &AccountTradeExitRow,
        position: &AccountExitPositionRow,
        exit_size: Decimal,
        exit_type: &str,
    ) -> Result<bool> {
        let exit_notional = trade.price * exit_size;
        let entry_value = position.entry_price * exit_size;
        let gross_pnl = if position.side == "buy" {
            exit_notional - entry_value
        } else {
            entry_value - exit_notional
        };
        let allocated_entry_fee = if position.entry_size > Decimal::ZERO {
            position.entry_fee * (exit_size / position.entry_size)
        } else {
            Decimal::ZERO
        };
        let net_pnl = gross_pnl - allocated_entry_fee;
        let metadata = serde_json::json!({
            "source": "manual_account_reconciliation",
            "manual_exit_kind": exit_type,
            "account_trade_id": trade.account_trade_id,
            "account_address": trade.account_address,
            "account_trade_source": trade.source,
            "transaction_hash": trade.transaction_hash,
            "venue_order_id": trade.venue_order_id,
            "venue_trade_id": trade.venue_trade_id,
            "allocated_entry_fee": allocated_entry_fee,
            "raw_account_trade": trade.raw_payload
        });
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin manual account exit transaction")?;
        let inserted = sqlx::query(
            r#"
            INSERT INTO polymarket.trade_exits (
              position_id, process_id, source_signal_id, timestamp_utc, exit_type, is_synthetic,
              exit_trigger_wallet, exit_source_trade_id, exit_price, exit_size,
              exit_notional, exit_fee, slippage_cost, gross_pnl, net_pnl, roi, metadata
            )
            VALUES (
              $1,$2,$3,$4,$5,false,$6,$7,$8,$9,$10,0,0,$11,$12,
              CASE WHEN $13 > 0 THEN $12 / $13 ELSE 0 END,
              $14
            )
            ON CONFLICT (position_id, exit_source_trade_id) DO NOTHING
            "#,
        )
        .bind(position.position_id)
        .bind(position.process_id)
        .bind(position.source_signal_id)
        .bind(trade.timestamp_utc)
        .bind(exit_type)
        .bind(&trade.account_address)
        .bind(trade.account_trade_id)
        .bind(trade.price)
        .bind(exit_size)
        .bind(exit_notional)
        .bind(gross_pnl)
        .bind(net_pnl)
        .bind(position.entry_notional)
        .bind(metadata)
        .execute(&mut *tx)
        .await
        .context("failed to insert manual account trade exit")?;
        if inserted.rows_affected() == 0 {
            tx.commit()
                .await
                .context("failed to commit duplicate manual account exit transaction")?;
            return Ok(false);
        }

        sqlx::query(
            r#"
            UPDATE polymarket.trade_positions p
            SET
              open_size = GREATEST(0, p.open_size - $2),
              realized_pnl = p.realized_pnl + $3,
              status = CASE
                WHEN GREATEST(0, p.open_size - $2) <= 0.000000001 THEN 'closed'
                ELSE 'partially_closed'
              END,
              unrealized_pnl = CASE
                WHEN GREATEST(0, p.open_size - $2) <= 0.000000001 THEN 0
                WHEN p.open_size > 0 THEN p.unrealized_pnl * (GREATEST(0, p.open_size - $2) / p.open_size)
                ELSE 0
              END,
              roi = CASE
                WHEN p.entry_notional > 0 THEN (
                  p.realized_pnl + $3 + CASE
                    WHEN GREATEST(0, p.open_size - $2) <= 0.000000001 THEN 0
                    WHEN p.open_size > 0 THEN p.unrealized_pnl * (GREATEST(0, p.open_size - $2) / p.open_size)
                    ELSE 0
                  END
                ) / p.entry_notional
                ELSE 0
              END,
              metadata = p.metadata || jsonb_build_object(
                'last_account_reconcile_exit', $4::jsonb
              ),
              updated_at = now()
            WHERE p.position_id = $1
            "#,
        )
        .bind(position.position_id)
        .bind(exit_size)
        .bind(net_pnl)
        .bind(serde_json::json!({
            "account_trade_id": trade.account_trade_id,
            "exit_type": exit_type,
            "exit_size": exit_size,
            "timestamp_utc": trade.timestamp_utc
        }))
        .execute(&mut *tx)
        .await
        .context("failed to update trade position from manual account exit")?;

        sqlx::query(
            r#"
            UPDATE polymarket.account_trades
            SET applied_exit_size = LEAST(size, applied_exit_size + $2),
                updated_at = now()
            WHERE account_trade_id = $1
            "#,
        )
        .bind(trade.account_trade_id)
        .bind(exit_size)
        .execute(&mut *tx)
        .await
        .context("failed to mark account trade applied size")?;

        tx.commit()
            .await
            .context("failed to commit manual account exit transaction")?;
        Ok(true)
    }

    pub async fn apply_account_position_mismatch_adjustments(
        &self,
        account_address: &str,
        token_id: Option<&str>,
        exit_type: &str,
    ) -> Result<ManualExitReport> {
        let mismatches = self
            .account_position_mismatches(account_address, token_id)
            .await?;
        let mut report = ManualExitReport::default();
        for mismatch in mismatches {
            if mismatch.mismatch_type != "account_less_than_db" {
                continue;
            }
            let mut remaining = (mismatch.db_open_size - mismatch.account_size).max(Decimal::ZERO);
            if remaining <= Decimal::ZERO {
                continue;
            }
            let positions = self
                .open_positions_for_account_adjustment(account_address, &mismatch.token_id)
                .await?;
            if positions.is_empty() {
                report.unmatched_trades += 1;
                continue;
            }
            for position in positions {
                if remaining <= Decimal::ZERO {
                    break;
                }
                let exit_size = remaining.min(position.open_size);
                if exit_size <= Decimal::ZERO {
                    continue;
                }
                report.exits_detected += 1;
                report.exit_size_applied += exit_size;
                let applied = self
                    .apply_account_position_mismatch_adjustment(
                        account_address,
                        &mismatch,
                        &position,
                        exit_size,
                        exit_type,
                    )
                    .await?;
                if applied {
                    report.exits_applied += 1;
                    remaining -= exit_size;
                }
            }
            if remaining > Decimal::ZERO {
                report.unmatched_trades += 1;
            }
        }
        Ok(report)
    }

    async fn open_positions_for_account_adjustment(
        &self,
        account_address: &str,
        token_id: &str,
    ) -> Result<Vec<AccountAdjustmentPositionRow>> {
        let rows = sqlx::query_as::<_, AccountAdjustmentPositionRow>(
            r#"
            WITH latest_snapshot AS (
              SELECT token_id, size, avg_price, current_price, snapshot_at
              FROM polymarket.account_position_snapshots
              WHERE account_address = $1
                AND token_id = $2
              ORDER BY snapshot_at DESC
              LIMIT 1
            )
            SELECT p.process_id, p.position_id, p.source_signal_id, p.side, p.entry_price,
              p.entry_size, p.open_size, p.entry_fee, p.entry_notional,
              COALESCE(ls.snapshot_at, now()) AS snapshot_at,
              COALESCE(ls.size, 0) AS snapshot_size,
              COALESCE(ls.current_price, ls.avg_price, p.entry_price) AS exit_price,
              CASE
                WHEN ls.current_price IS NOT NULL THEN 'account_snapshot_current_price'
                WHEN ls.avg_price IS NOT NULL THEN 'account_snapshot_avg_price'
                ELSE 'entry_price_fallback'
              END AS price_source
            FROM polymarket.trade_positions p
            LEFT JOIN latest_snapshot ls ON ls.token_id = p.token_id
            WHERE p.is_live_capital = true
              AND p.token_id = $2
              AND p.status IN ('open', 'partially_closed')
              AND p.open_size > 0
            ORDER BY p.entry_timestamp ASC, p.created_at ASC
            "#,
        )
        .bind(account_address)
        .bind(token_id)
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch open positions for account position adjustment")?;
        Ok(rows)
    }

    async fn apply_account_position_mismatch_adjustment(
        &self,
        account_address: &str,
        mismatch: &AccountPositionMismatch,
        position: &AccountAdjustmentPositionRow,
        exit_size: Decimal,
        exit_type: &str,
    ) -> Result<bool> {
        let exit_notional = position.exit_price * exit_size;
        let entry_value = position.entry_price * exit_size;
        let gross_pnl = if position.side == "buy" {
            exit_notional - entry_value
        } else {
            entry_value - exit_notional
        };
        let allocated_entry_fee = if position.entry_size > Decimal::ZERO {
            position.entry_fee * (exit_size / position.entry_size)
        } else {
            Decimal::ZERO
        };
        let net_pnl = gross_pnl - allocated_entry_fee;
        let metadata = serde_json::json!({
            "source": "account_position_snapshot_reconciliation",
            "manual_exit_kind": exit_type,
            "account_address": account_address,
            "token_id": mismatch.token_id,
            "db_open_size_before": mismatch.db_open_size,
            "account_size": mismatch.account_size,
            "delta_size": mismatch.delta_size,
            "snapshot_at": position.snapshot_at,
            "snapshot_size": position.snapshot_size,
            "price_source": position.price_source,
            "allocated_entry_fee": allocated_entry_fee,
            "note": "Synthetic adjustment used when account position snapshot is lower than DB open size and no matching account sell trade remains unapplied."
        });
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin account position adjustment transaction")?;
        let inserted = sqlx::query(
            r#"
            INSERT INTO polymarket.trade_exits (
              position_id, process_id, source_signal_id, timestamp_utc, exit_type, is_synthetic,
              exit_trigger_wallet, exit_source_trade_id, exit_price, exit_size,
              exit_notional, exit_fee, slippage_cost, gross_pnl, net_pnl, roi, metadata
            )
            VALUES (
              $1,$2,$3,$4,$5,true,$6,NULL,$7,$8,$9,0,0,$10,$11,
              CASE WHEN $12 > 0 THEN $11 / $12 ELSE 0 END,
              $13
            )
            "#,
        )
        .bind(position.position_id)
        .bind(position.process_id)
        .bind(position.source_signal_id)
        .bind(position.snapshot_at)
        .bind(exit_type)
        .bind(account_address)
        .bind(position.exit_price)
        .bind(exit_size)
        .bind(exit_notional)
        .bind(gross_pnl)
        .bind(net_pnl)
        .bind(position.entry_notional)
        .bind(metadata)
        .execute(&mut *tx)
        .await
        .context("failed to insert account position adjustment exit")?;
        if inserted.rows_affected() == 0 {
            tx.commit()
                .await
                .context("failed to commit duplicate account position adjustment transaction")?;
            return Ok(false);
        }

        sqlx::query(
            r#"
            UPDATE polymarket.trade_positions p
            SET
              open_size = GREATEST(0, p.open_size - $2),
              realized_pnl = p.realized_pnl + $3,
              status = CASE
                WHEN GREATEST(0, p.open_size - $2) <= 0.000000001 THEN 'closed'
                ELSE 'partially_closed'
              END,
              unrealized_pnl = CASE
                WHEN GREATEST(0, p.open_size - $2) <= 0.000000001 THEN 0
                WHEN p.open_size > 0 THEN p.unrealized_pnl * (GREATEST(0, p.open_size - $2) / p.open_size)
                ELSE 0
              END,
              roi = CASE
                WHEN p.entry_notional > 0 THEN (
                  p.realized_pnl + $3 + CASE
                    WHEN GREATEST(0, p.open_size - $2) <= 0.000000001 THEN 0
                    WHEN p.open_size > 0 THEN p.unrealized_pnl * (GREATEST(0, p.open_size - $2) / p.open_size)
                    ELSE 0
                  END
                ) / p.entry_notional
                ELSE 0
              END,
              metadata = p.metadata || jsonb_build_object(
                'last_account_position_adjustment', $4::jsonb
              ),
              updated_at = now()
            WHERE p.position_id = $1
            "#,
        )
        .bind(position.position_id)
        .bind(exit_size)
        .bind(net_pnl)
        .bind(serde_json::json!({
            "exit_type": exit_type,
            "exit_size": exit_size,
            "account_address": account_address,
            "snapshot_at": position.snapshot_at
        }))
        .execute(&mut *tx)
        .await
        .context("failed to update trade position from account position adjustment")?;

        tx.commit()
            .await
            .context("failed to commit account position adjustment transaction")?;
        Ok(true)
    }

    pub async fn preview_account_position_mismatches(
        &self,
        account_address: &str,
        snapshots: &[AccountPositionSnapshot],
        token_id: Option<&str>,
    ) -> Result<Vec<AccountPositionMismatch>> {
        let open = self.live_open_position_sizes(token_id).await?;
        let mut mismatches = Vec::new();
        for (token, db_open_size) in open {
            let account_size = snapshots
                .iter()
                .find(|snapshot| snapshot.token_id == token)
                .map(|snapshot| snapshot.size)
                .unwrap_or(Decimal::ZERO);
            push_position_mismatch(&mut mismatches, token, db_open_size, account_size);
        }
        if mismatches.is_empty() {
            debug_assert!(!account_address.is_empty() || snapshots.is_empty());
        }
        Ok(mismatches)
    }

    pub async fn open_trade_position_notional_for_process_scope(
        &self,
        process_id: Uuid,
        token_id: Option<&str>,
        market_id: Option<&str>,
    ) -> Result<Decimal> {
        let notional = sqlx::query_scalar::<_, Decimal>(
            r#"
            SELECT COALESCE(sum(open_size * entry_price), 0)
            FROM polymarket.trade_positions
            WHERE process_id = $1
              AND status IN ('open', 'partially_closed')
              AND open_size > 0
              AND ($2::text IS NULL OR token_id = $2)
              AND ($3::text IS NULL OR market_id = $3)
            "#,
        )
        .bind(process_id)
        .bind(token_id)
        .bind(market_id)
        .fetch_one(&self.pool)
        .await
        .context("failed to fetch open trade position notional for process scope")?;
        Ok(notional)
    }

    pub async fn account_position_mismatches(
        &self,
        account_address: &str,
        token_id: Option<&str>,
    ) -> Result<Vec<AccountPositionMismatch>> {
        let rows = sqlx::query_as::<_, AccountPositionMismatch>(
            r#"
            WITH db_open AS (
              SELECT token_id, COALESCE(sum(open_size), 0) AS db_open_size
              FROM polymarket.trade_positions
              WHERE is_live_capital = true
                AND status IN ('open', 'partially_closed')
                AND open_size > 0
                AND ($2::text IS NULL OR token_id = $2)
              GROUP BY token_id
            ),
            latest_snapshots AS (
              SELECT DISTINCT ON (token_id)
                token_id,
                size AS account_size
              FROM polymarket.account_position_snapshots
              WHERE account_address = $1
                AND ($2::text IS NULL OR token_id = $2)
              ORDER BY token_id, snapshot_at DESC
            ),
            compared AS (
              SELECT
                COALESCE(db_open.token_id, latest_snapshots.token_id) AS token_id,
                COALESCE(db_open.db_open_size, 0) AS db_open_size,
                COALESCE(latest_snapshots.account_size, 0) AS account_size
              FROM db_open
              FULL OUTER JOIN latest_snapshots USING (token_id)
            )
            SELECT
              token_id,
              db_open_size,
              account_size,
              account_size - db_open_size AS delta_size,
              CASE
                WHEN account_size < db_open_size THEN 'account_less_than_db'
                ELSE 'account_greater_than_db'
              END AS mismatch_type
            FROM compared
            WHERE abs(account_size - db_open_size) > 0.000000001
            ORDER BY abs(account_size - db_open_size) DESC
            "#,
        )
        .bind(account_address)
        .bind(token_id)
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch account position mismatches")?;
        Ok(rows)
    }

    async fn live_open_position_sizes(
        &self,
        token_id: Option<&str>,
    ) -> Result<Vec<(String, Decimal)>> {
        #[derive(FromRow)]
        struct Row {
            token_id: String,
            db_open_size: Decimal,
        }
        let rows = sqlx::query_as::<_, Row>(
            r#"
            SELECT token_id, COALESCE(sum(open_size), 0) AS db_open_size
            FROM polymarket.trade_positions
            WHERE is_live_capital = true
              AND status IN ('open', 'partially_closed')
              AND open_size > 0
              AND ($1::text IS NULL OR token_id = $1)
            GROUP BY token_id
            "#,
        )
        .bind(token_id)
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch live open position sizes")?;
        Ok(rows
            .into_iter()
            .map(|row| (row.token_id, row.db_open_size))
            .collect())
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

    pub async fn upsert_trade_mark_source_failure(
        &self,
        failure: &TradeMarkSourceFailure,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.trade_mark_source_failures (
              process_id, position_id, token_id, market_id, failure_source, failure_reason, metadata
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            ON CONFLICT (
              COALESCE(process_id, '00000000-0000-0000-0000-000000000000'::uuid),
              COALESCE(position_id, '00000000-0000-0000-0000-000000000000'::uuid),
              token_id,
              failure_source
            ) DO UPDATE SET
              market_id = COALESCE(EXCLUDED.market_id, polymarket.trade_mark_source_failures.market_id),
              failure_reason = EXCLUDED.failure_reason,
              failure_count = polymarket.trade_mark_source_failures.failure_count + 1,
              last_failed_at = now(),
              resolved_at = NULL,
              metadata = EXCLUDED.metadata
            "#,
        )
        .bind(failure.process_id)
        .bind(failure.position_id)
        .bind(&failure.token_id)
        .bind(&failure.market_id)
        .bind(&failure.failure_source)
        .bind(&failure.failure_reason)
        .bind(&failure.metadata)
        .execute(&self.pool)
        .await
        .context("failed to upsert trade mark source failure")?;
        Ok(())
    }

    pub async fn resolve_trade_mark_source_failure(
        &self,
        process_id: Option<Uuid>,
        position_id: Option<Uuid>,
        token_id: &str,
        failure_source: &str,
    ) -> Result<u64> {
        let result = sqlx::query(
            r#"
            UPDATE polymarket.trade_mark_source_failures
            SET resolved_at = now(),
                metadata = metadata || jsonb_build_object('resolved_reason', 'mark_source_available')
            WHERE process_id IS NOT DISTINCT FROM $1
              AND position_id IS NOT DISTINCT FROM $2
              AND token_id = $3
              AND failure_source = $4
              AND resolved_at IS NULL
            "#,
        )
        .bind(process_id)
        .bind(position_id)
        .bind(token_id)
        .bind(failure_source)
        .execute(&self.pool)
        .await
        .context("failed to resolve trade mark source failure")?;
        Ok(result.rows_affected())
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

    pub async fn backfill_trade_positions_from_copy_signals(&self) -> Result<u64> {
        let result = sqlx::query(
            r#"
            WITH filled_entries AS (
              SELECT
                s.signal_id,
                s.process_id,
                s.timestamp_utc AS signal_timestamp,
                s.proxy_wallet,
                s.source_trade_id,
                s.market_id,
                COALESCE(s.token_id, o.token_id, f.token_id) AS token_id,
                CASE WHEN upper(s.side) = 'SELL' THEN 'sell' ELSE 'buy' END AS side,
                CASE
                  WHEN f.source = 'live' THEN 'live'
                  WHEN f.source = 'paper' THEN 'paper'
                  ELSE 'sim'
                END AS execution_mode,
                CASE
                  WHEN f.source = 'live' THEN 'polymarket_clob'
                  WHEN f.source = 'paper' THEN 'paper'
                  ELSE 'sim'
                END AS venue,
                min(f.timestamp_utc) AS entry_timestamp,
                sum(f.size) AS entry_size,
                sum(f.price * f.size) / NULLIF(sum(f.size), 0) AS entry_price,
                sum(f.price * f.size) AS entry_notional,
                sum(f.fee) AS entry_fee,
                jsonb_build_object(
                  'copy_signal', to_jsonb(s),
                  'orders', jsonb_agg(DISTINCT to_jsonb(o)),
                  'fills', jsonb_agg(DISTINCT to_jsonb(f))
                ) AS metadata
              FROM polymarket.copy_trade_signals s
              JOIN polymarket.orders o
                ON (o.raw_payload #>> '{request,signal_id}')::uuid = s.signal_id
               AND o.process_id IS NOT DISTINCT FROM s.process_id
              JOIN polymarket.fills f
                ON f.order_id = o.order_id
               AND f.process_id IS NOT DISTINCT FROM s.process_id
              WHERE s.status = 'filled'
                AND NOT EXISTS (
                  SELECT 1
                  FROM polymarket.trade_positions p
                  WHERE p.source_signal_table = 'polymarket.copy_trade_signals'
                    AND p.source_signal_id = s.signal_id
                    AND p.token_id = COALESCE(s.token_id, o.token_id, f.token_id)
                    AND p.process_id IS NOT DISTINCT FROM s.process_id
                )
              GROUP BY
                s.signal_id, s.process_id, s.timestamp_utc, s.proxy_wallet, s.source_trade_id,
                s.market_id, COALESCE(s.token_id, o.token_id, f.token_id),
                CASE WHEN upper(s.side) = 'SELL' THEN 'sell' ELSE 'buy' END,
                CASE
                  WHEN f.source = 'live' THEN 'live'
                  WHEN f.source = 'paper' THEN 'paper'
                  ELSE 'sim'
                END,
                CASE
                  WHEN f.source = 'live' THEN 'polymarket_clob'
                  WHEN f.source = 'paper' THEN 'paper'
                  ELSE 'sim'
                END
            )
            INSERT INTO polymarket.trade_positions (
              process_id, source_signal_table, source_signal_id, source_trade_id, signal_source,
              strategy_name, strategy_version, strategy_config_hash, execution_mode, venue,
              is_live_capital, proxy_wallet, market_id, token_id, side, entry_price,
              entry_size, open_size, entry_notional, entry_fee, entry_timestamp,
              follow_lag_seconds, status, metadata
            )
            SELECT
              process_id,
              'polymarket.copy_trade_signals',
              signal_id,
              source_trade_id,
              'whale_follow',
              'whale_follow_v1',
              'whale_performance_v1',
              NULL,
              execution_mode,
              venue,
              execution_mode = 'live',
              proxy_wallet,
              market_id,
              token_id,
              side,
              entry_price,
              entry_size,
              entry_size,
              entry_notional,
              entry_fee,
              entry_timestamp,
              GREATEST(0, floor(extract(epoch from (entry_timestamp - signal_timestamp))))::integer,
              'open',
              metadata
            FROM filled_entries
            WHERE token_id IS NOT NULL
              AND entry_size > 0
              AND entry_price IS NOT NULL
            ON CONFLICT DO NOTHING
            "#,
        )
        .execute(&self.pool)
        .await
        .context("failed to backfill trade positions from filled copy signals")?;
        Ok(result.rows_affected())
    }

    pub async fn fetch_whale_led_trade_exit_candidates(
        &self,
        limit: i64,
        max_candidate_age: chrono::Duration,
    ) -> Result<Vec<WhaleLedTradeExitCandidate>> {
        let rows = sqlx::query_as::<_, WhaleLedTradeExitCandidate>(
            r#"
            SELECT
              p.process_id,
              p.position_id,
              p.source_signal_id,
              p.proxy_wallet,
              p.market_id,
              p.token_id,
              p.side,
              p.entry_price,
              p.entry_size,
              p.open_size,
              p.entry_fee,
              p.entry_notional,
              wt.trade_id AS exit_source_trade_id,
              wt.timestamp_utc AS exit_timestamp,
              wt.price AS reference_exit_price,
              LEAST(
                p.open_size,
                CASE
                  WHEN p.entry_notional > 0 THEN p.open_size * LEAST(1, wt.cash_value / p.entry_notional)
                  ELSE p.open_size
                END
              ) AS exit_size
            FROM polymarket.wallet_trades wt
            JOIN polymarket.trade_positions p
              ON p.proxy_wallet = wt.proxy_wallet
             AND p.token_id = wt.asset
             AND wt.timestamp_utc > p.entry_timestamp
             AND p.status IN ('open', 'partially_closed')
             AND p.open_size > 0
             AND (
               (p.side = 'buy' AND upper(wt.side) = 'SELL')
               OR (p.side = 'sell' AND upper(wt.side) = 'BUY')
             )
            WHERE wt.timestamp_utc >= now() - ($2 * interval '1 second')
              AND NOT EXISTS (
                SELECT 1
                FROM polymarket.trade_exits te
                WHERE te.position_id = p.position_id
                  AND te.is_synthetic = false
                  AND (
                    te.exit_source_trade_id = wt.trade_id
                    OR te.metadata #>> '{reference_exit_source_trade_id}' = wt.trade_id::text
                  )
              )
              AND NOT EXISTS (
                SELECT 1
                FROM polymarket.orders o
                WHERE o.raw_payload #>> '{request,metadata,purpose}' = 'whale_led_exit'
                  AND o.raw_payload #>> '{request,metadata,position_id}' = p.position_id::text
                  AND o.raw_payload #>> '{request,metadata,exit_source_trade_id}' = wt.trade_id::text
                  AND o.created_at > now() - interval '30 seconds'
              )
            ORDER BY
              wt.timestamp_utc DESC,
              p.entry_timestamp DESC,
              p.entry_notional ASC
            LIMIT $1
            "#,
        )
        .bind(limit)
        .bind(max_candidate_age.num_seconds().max(0))
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch whale-led trade exit candidates")?;
        Ok(rows
            .into_iter()
            .filter(|row| row.exit_size > Decimal::ZERO)
            .collect())
    }

    pub async fn fetch_take_profit_trade_exit_candidates(
        &self,
        process_id: Uuid,
        take_profit_roi: Decimal,
        exit_size_fraction: Decimal,
        min_hold: chrono::Duration,
        require_fresh_mark: chrono::Duration,
        max_exit_slippage_bps: Decimal,
        exit_pricing_mode: &str,
        limit: i64,
    ) -> Result<Vec<TakeProfitTradeExitCandidate>> {
        self.fetch_risk_control_trade_exit_candidates(
            process_id,
            "take_profit_exit",
            take_profit_roi,
            exit_size_fraction,
            min_hold,
            require_fresh_mark,
            max_exit_slippage_bps,
            exit_pricing_mode,
            limit,
            Utc::now(),
        )
        .await
    }

    pub async fn fetch_take_profit_trade_exit_candidates_as_of(
        &self,
        process_id: Uuid,
        take_profit_roi: Decimal,
        exit_size_fraction: Decimal,
        min_hold: chrono::Duration,
        require_fresh_mark: chrono::Duration,
        max_exit_slippage_bps: Decimal,
        exit_pricing_mode: &str,
        limit: i64,
        as_of: DateTime<Utc>,
    ) -> Result<Vec<TakeProfitTradeExitCandidate>> {
        self.fetch_risk_control_trade_exit_candidates(
            process_id,
            "take_profit_exit",
            take_profit_roi,
            exit_size_fraction,
            min_hold,
            require_fresh_mark,
            max_exit_slippage_bps,
            exit_pricing_mode,
            limit,
            as_of,
        )
        .await
    }

    pub async fn fetch_stop_loss_trade_exit_candidates(
        &self,
        process_id: Uuid,
        stop_loss_roi: Decimal,
        exit_size_fraction: Decimal,
        min_hold: chrono::Duration,
        require_fresh_mark: chrono::Duration,
        max_exit_slippage_bps: Decimal,
        exit_pricing_mode: &str,
        limit: i64,
    ) -> Result<Vec<TakeProfitTradeExitCandidate>> {
        self.fetch_risk_control_trade_exit_candidates(
            process_id,
            "stop_loss_exit",
            stop_loss_roi,
            exit_size_fraction,
            min_hold,
            require_fresh_mark,
            max_exit_slippage_bps,
            exit_pricing_mode,
            limit,
            Utc::now(),
        )
        .await
    }

    pub async fn fetch_stop_loss_trade_exit_candidates_as_of(
        &self,
        process_id: Uuid,
        stop_loss_roi: Decimal,
        exit_size_fraction: Decimal,
        min_hold: chrono::Duration,
        require_fresh_mark: chrono::Duration,
        max_exit_slippage_bps: Decimal,
        exit_pricing_mode: &str,
        limit: i64,
        as_of: DateTime<Utc>,
    ) -> Result<Vec<TakeProfitTradeExitCandidate>> {
        self.fetch_risk_control_trade_exit_candidates(
            process_id,
            "stop_loss_exit",
            stop_loss_roi,
            exit_size_fraction,
            min_hold,
            require_fresh_mark,
            max_exit_slippage_bps,
            exit_pricing_mode,
            limit,
            as_of,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn fetch_risk_control_trade_exit_candidates(
        &self,
        process_id: Uuid,
        exit_purpose: &str,
        threshold_roi: Decimal,
        exit_size_fraction: Decimal,
        min_hold: chrono::Duration,
        require_fresh_mark: chrono::Duration,
        max_exit_slippage_bps: Decimal,
        exit_pricing_mode: &str,
        limit: i64,
        as_of: DateTime<Utc>,
    ) -> Result<Vec<TakeProfitTradeExitCandidate>> {
        let rows = sqlx::query_as::<_, TakeProfitTradeExitCandidate>(
            r#"
            WITH prepared AS MATERIALIZED (
              SELECT
                $8::text AS exit_purpose,
                p.process_id,
                p.position_id,
                p.source_signal_id,
                p.proxy_wallet,
                p.market_id,
                p.token_id,
                p.side,
                p.entry_price,
                p.entry_size,
                p.open_size,
                p.entry_fee,
                p.entry_notional,
                p.entry_timestamp,
                (
                  substr(md5(p.position_id::text || '|' || $8::text || '|' || $2::text), 1, 8) || '-' ||
                  substr(md5(p.position_id::text || '|' || $8::text || '|' || $2::text), 9, 4) || '-' ||
                  substr(md5(p.position_id::text || '|' || $8::text || '|' || $2::text), 13, 4) || '-' ||
                  substr(md5(p.position_id::text || '|' || $8::text || '|' || $2::text), 17, 4) || '-' ||
                  substr(md5(p.position_id::text || '|' || $8::text || '|' || $2::text), 21, 12)
                )::uuid AS exit_source_trade_id,
                $9::timestamptz AS exit_timestamp,
                p.latest_mark_price AS reference_exit_price,
                CASE
                  WHEN p.side = 'buy' THEN GREATEST(
                    0::numeric,
                    p.latest_mark_price * (1 - ($6::numeric / 10000))
                  )
                  ELSE LEAST(
                    1::numeric,
                    p.latest_mark_price * (1 + ($6::numeric / 10000))
                  )
                END AS order_limit_price,
                CASE
                  WHEN $10::text = 'marketable_limit'
                    AND p.side = 'buy'
                    AND ob.best_bid IS NOT NULL
                    AND ob.best_bid >= GREATEST(0::numeric, p.latest_mark_price * (1 - ($6::numeric / 10000)))
                    THEN ob.best_bid
                  WHEN $10::text = 'marketable_limit'
                    AND p.side <> 'buy'
                    AND ob.best_ask IS NOT NULL
                    AND ob.best_ask <= LEAST(1::numeric, p.latest_mark_price * (1 + ($6::numeric / 10000)))
                    THEN ob.best_ask
                  ELSE NULL
                END AS marketable_exit_price,
                ob.best_bid,
                ob.best_ask,
                LEAST(
                  p.open_size,
                  p.open_size * LEAST(1::numeric, GREATEST(0::numeric, $3::numeric))
                ) AS exit_size,
                CASE
                  WHEN p.side = 'buy' THEN
                    (p.open_size * (p.latest_mark_price - p.entry_price))
                      / NULLIF(p.entry_price * p.open_size, 0)
                  ELSE
                    (p.open_size * (p.entry_price - p.latest_mark_price))
                      / NULLIF(p.entry_price * p.open_size, 0)
                END AS trigger_roi,
                $2::numeric AS threshold_roi,
                $2::numeric AS take_profit_roi,
                p.latest_mark_timestamp,
                $6::numeric AS max_exit_slippage_bps,
                $10::text AS exit_pricing_mode
              FROM polymarket.trade_positions p
              LEFT JOIN LATERAL (
                SELECT best_bid, best_ask, timestamp_utc
                FROM polymarket.orderbook_snapshots ob
                WHERE ob.token_id = p.token_id
                  AND ob.timestamp_utc >= $9::timestamptz - ($5::bigint * interval '1 millisecond')
                  AND ob.timestamp_utc <= $9::timestamptz
                ORDER BY ob.timestamp_utc DESC
                LIMIT 1
              ) ob ON true
              WHERE p.process_id = $1
                AND p.status IN ('open', 'partially_closed')
                AND p.open_size > 0
                AND p.entry_price > 0
                AND p.latest_mark_price IS NOT NULL
                AND p.latest_mark_timestamp IS NOT NULL
                AND p.latest_mark_timestamp >= $9::timestamptz - ($5::bigint * interval '1 millisecond')
                AND p.latest_mark_timestamp <= $9::timestamptz
                AND p.entry_timestamp <= $9::timestamptz - ($4::bigint * interval '1 millisecond')
            )
            SELECT
              exit_purpose,
              process_id,
              position_id,
              source_signal_id,
              proxy_wallet,
              market_id,
              token_id,
              side,
              entry_price,
              entry_size,
              open_size,
              entry_fee,
              entry_notional,
              exit_source_trade_id,
              exit_timestamp,
              reference_exit_price,
              order_limit_price,
              marketable_exit_price,
              best_bid,
              best_ask,
              exit_size,
              trigger_roi,
              threshold_roi,
              take_profit_roi,
              latest_mark_timestamp,
              max_exit_slippage_bps,
              exit_pricing_mode
            FROM prepared p
            WHERE (
                ($8::text = 'take_profit_exit' AND p.trigger_roi >= $2)
                OR ($8::text = 'stop_loss_exit' AND p.trigger_roi <= $2)
              )
              AND p.exit_size > 0
              AND NOT EXISTS (
                SELECT 1
                FROM polymarket.trade_exits te
                WHERE te.position_id = p.position_id
                  AND te.is_synthetic = false
                  AND (
                    te.exit_source_trade_id = p.exit_source_trade_id
                    OR te.metadata #>> '{purpose}' IN ('take_profit_exit', 'stop_loss_exit')
                    OR te.metadata #>> '{source}' IN ('take_profit_exit', 'stop_loss_exit')
                  )
              )
              AND NOT EXISTS (
                SELECT 1
                FROM polymarket.orders o
                WHERE o.raw_payload #>> '{request,metadata,purpose}' IN ('take_profit_exit', 'stop_loss_exit')
                  AND o.raw_payload #>> '{request,metadata,position_id}' = p.position_id::text
                  AND COALESCE(
                    (o.raw_payload #>> '{request,metadata,reference_exit_timestamp}')::timestamptz,
                    o.created_at
                  ) > $9::timestamptz - interval '30 seconds'
              )
            ORDER BY
              CASE WHEN $8::text = 'stop_loss_exit' THEN p.trigger_roi END ASC,
              CASE WHEN $8::text = 'take_profit_exit' THEN p.trigger_roi END DESC,
              p.entry_timestamp ASC
            LIMIT $7
            "#,
        )
        .bind(process_id)
        .bind(threshold_roi)
        .bind(exit_size_fraction)
        .bind(min_hold.num_milliseconds().max(0))
        .bind(require_fresh_mark.num_milliseconds().max(0))
        .bind(max_exit_slippage_bps.max(Decimal::ZERO))
        .bind(limit)
        .bind(exit_purpose)
        .bind(as_of)
        .bind(exit_pricing_mode)
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch risk-control trade exit candidates")?;
        Ok(rows
            .into_iter()
            .filter(|row| row.exit_size > Decimal::ZERO)
            .collect())
    }

    pub async fn reconcile_trade_positions_from_executable_exits(&self) -> Result<u64> {
        let result = sqlx::query(
            r#"
            WITH executable AS (
              SELECT
                position_id,
                COALESCE(sum(exit_size), 0) AS exit_size,
                COALESCE(sum(net_pnl), 0) AS realized_pnl
              FROM polymarket.trade_exits
              WHERE is_synthetic = false
              GROUP BY position_id
            ),
            synthetic_positions AS (
              SELECT DISTINCT position_id
              FROM polymarket.trade_exits
              WHERE is_synthetic = true
            ),
            prepared AS (
              SELECT
                p.position_id,
                GREATEST(0, p.entry_size - COALESCE(e.exit_size, 0)) AS executable_open_size,
                COALESCE(e.realized_pnl, 0) AS executable_realized_pnl
              FROM polymarket.trade_positions p
              JOIN synthetic_positions sp ON sp.position_id = p.position_id
              LEFT JOIN executable e ON e.position_id = p.position_id
            )
            UPDATE polymarket.trade_positions p
            SET
              open_size = prepared.executable_open_size,
              realized_pnl = prepared.executable_realized_pnl,
              status = CASE
                WHEN prepared.executable_open_size <= 0.000000001 THEN 'closed'
                WHEN prepared.executable_open_size < p.entry_size THEN 'partially_closed'
                ELSE 'open'
              END,
              unrealized_pnl = 0,
              roi = CASE
                WHEN p.entry_notional > 0 THEN prepared.executable_realized_pnl / p.entry_notional
                ELSE 0
              END,
              metadata = p.metadata || jsonb_build_object(
                'realized_pnl_basis', 'executable_exits_only',
                'synthetic_exits_preserved_as_reference', true
              ),
              updated_at = now()
            FROM prepared
            WHERE p.position_id = prepared.position_id
              AND (
                p.open_size <> prepared.executable_open_size
                OR p.realized_pnl <> prepared.executable_realized_pnl
                OR (prepared.executable_open_size > 0 AND p.status IN ('closed', 'resolved'))
              )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("failed to reconcile positions from executable exits")?;
        Ok(result.rows_affected())
    }

    pub async fn apply_executable_trade_exit(
        &self,
        candidate: &WhaleLedTradeExitCandidate,
        report: &OrderPlanReport,
    ) -> Result<u64> {
        let applied_fills = cap_fills_to_size(&report.fills, candidate.open_size);
        let filled_size: Decimal = applied_fills.iter().map(|fill| fill.size).sum();
        if filled_size <= Decimal::ZERO {
            return Ok(0);
        }
        let exit_fee: Decimal = applied_fills.iter().map(|fill| fill.fee).sum();
        let exit_notional: Decimal = applied_fills
            .iter()
            .map(|fill| fill.price * fill.size)
            .sum();
        let exit_price = exit_notional / filled_size;
        let entry_value = candidate.entry_price * filled_size;
        let gross_pnl = if candidate.side == "buy" {
            exit_notional - entry_value
        } else {
            entry_value - exit_notional
        };
        let allocated_entry_fee = if candidate.entry_size > Decimal::ZERO {
            candidate.entry_fee * (filled_size / candidate.entry_size)
        } else {
            Decimal::ZERO
        };
        let net_pnl = gross_pnl - allocated_entry_fee - exit_fee;
        let reference_notional = candidate.reference_exit_price * filled_size;
        let slippage_cost = if candidate.side == "buy" {
            (reference_notional - exit_notional).max(Decimal::ZERO)
        } else {
            (exit_notional - reference_notional).max(Decimal::ZERO)
        };
        let close_order_id = report.orders.first().map(|order| order.order_id.clone());
        let fill_payloads: Vec<_> = applied_fills
            .iter()
            .map(|fill| serde_json::to_value(fill))
            .collect::<std::result::Result<_, _>>()?;
        let metadata = serde_json::json!({
            "source": "executable_whale_led_exit",
            "close_order_id": close_order_id,
            "reference_exit_source_trade_id": candidate.exit_source_trade_id,
            "reference_exit_price": candidate.reference_exit_price,
            "reference_exit_notional": reference_notional,
            "allocated_entry_fee": allocated_entry_fee,
            "execution_source": applied_fills
                .first()
                .map(|fill| serialized_name(&fill.source).unwrap_or_else(|_| "unknown".to_string()))
                .unwrap_or_else(|| "unknown".to_string()),
            "applied_fill_size": filled_size,
            "fills": fill_payloads
        });

        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin executable exit transaction")?;
        let duplicate_exists = sqlx::query_scalar::<_, bool>(
            r#"
            SELECT EXISTS (
              SELECT 1
              FROM polymarket.trade_exits
              WHERE position_id = $1
                AND is_synthetic = false
                AND (
                  exit_source_trade_id = $2
                  OR ($3::text IS NOT NULL AND metadata #>> '{close_order_id}' = $3)
                  OR metadata #>> '{reference_exit_source_trade_id}' = $2::text
                )
            )
            "#,
        )
        .bind(candidate.position_id)
        .bind(candidate.exit_source_trade_id)
        .bind(close_order_id.as_deref())
        .fetch_one(&mut *tx)
        .await
        .context("failed to check duplicate executable trade exit")?;
        if duplicate_exists {
            tx.commit()
                .await
                .context("failed to commit duplicate executable exit transaction")?;
            return Ok(0);
        }

        let inserted = sqlx::query(
            r#"
            INSERT INTO polymarket.trade_exits (
              position_id, process_id, source_signal_id, timestamp_utc, exit_type, is_synthetic,
              exit_trigger_wallet, exit_source_trade_id, exit_price, exit_size,
              exit_notional, exit_fee, slippage_cost, gross_pnl, net_pnl, roi, metadata
            )
            VALUES (
              $1,$2,$3,$4,$5,false,$6,$7,$8,$9,$10,$11,$12,$13,$14,
              CASE WHEN $15 > 0 THEN $14 / $15 ELSE 0 END,
              $16
            )
            ON CONFLICT (position_id, exit_source_trade_id) DO NOTHING
            "#,
        )
        .bind(candidate.position_id)
        .bind(candidate.process_id)
        .bind(candidate.source_signal_id)
        .bind(candidate.exit_timestamp)
        .bind(if filled_size < candidate.open_size {
            "whale_reduce"
        } else {
            "whale_exit"
        })
        .bind(candidate.proxy_wallet.as_deref())
        .bind(candidate.exit_source_trade_id)
        .bind(exit_price)
        .bind(filled_size)
        .bind(exit_notional)
        .bind(exit_fee)
        .bind(slippage_cost)
        .bind(gross_pnl)
        .bind(net_pnl)
        .bind(candidate.entry_notional)
        .bind(metadata)
        .execute(&mut *tx)
        .await
        .context("failed to insert executable trade exit")?;

        if inserted.rows_affected() == 0 {
            tx.commit()
                .await
                .context("failed to commit duplicate executable exit transaction")?;
            return Ok(0);
        }

        sqlx::query(
            r#"
            UPDATE polymarket.trade_positions p
            SET
              open_size = GREATEST(0, p.open_size - $2),
              realized_pnl = p.realized_pnl + $3,
              status = CASE
                WHEN GREATEST(0, p.open_size - $2) <= 0.000000001 THEN 'closed'
                ELSE 'partially_closed'
              END,
              unrealized_pnl = CASE
                WHEN GREATEST(0, p.open_size - $2) <= 0.000000001 THEN 0
                WHEN p.open_size > 0 THEN p.unrealized_pnl * (GREATEST(0, p.open_size - $2) / p.open_size)
                ELSE 0
              END,
              roi = CASE
                WHEN p.entry_notional > 0 THEN (
                  p.realized_pnl + $3 + CASE
                    WHEN GREATEST(0, p.open_size - $2) <= 0.000000001 THEN 0
                    WHEN p.open_size > 0 THEN p.unrealized_pnl * (GREATEST(0, p.open_size - $2) / p.open_size)
                    ELSE 0
                  END
                ) / p.entry_notional
                ELSE 0
              END,
              updated_at = now()
            WHERE p.position_id = $1
            "#,
        )
        .bind(candidate.position_id)
        .bind(filled_size)
        .bind(net_pnl)
        .execute(&mut *tx)
        .await
        .context("failed to update trade position from executable exit")?;

        tx.commit()
            .await
            .context("failed to commit executable exit transaction")?;
        Ok(1)
    }

    pub async fn apply_take_profit_trade_exit(
        &self,
        candidate: &TakeProfitTradeExitCandidate,
        report: &OrderPlanReport,
    ) -> Result<u64> {
        let close_order_id = report.orders.first().map(|order| order.order_id.clone());

        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin risk-control exit transaction")?;
        let duplicate_exists = sqlx::query_scalar::<_, bool>(
            r#"
            SELECT EXISTS (
              SELECT 1
              FROM polymarket.trade_exits
              WHERE position_id = $1
                AND is_synthetic = false
                AND (
                  exit_source_trade_id = $2
                  OR ($3::text IS NOT NULL AND metadata #>> '{close_order_id}' = $3)
                  OR metadata #>> '{purpose}' IN ('take_profit_exit', 'stop_loss_exit')
                  OR metadata #>> '{source}' IN ('take_profit_exit', 'stop_loss_exit')
                )
            )
            "#,
        )
        .bind(candidate.position_id)
        .bind(candidate.exit_source_trade_id)
        .bind(close_order_id.as_deref())
        .fetch_one(&mut *tx)
        .await
        .context("failed to check duplicate risk-control trade exit")?;
        if duplicate_exists {
            tx.commit()
                .await
                .context("failed to commit duplicate risk-control exit transaction")?;
            return Ok(0);
        }

        let Some(current_open_size) = sqlx::query_scalar::<_, Decimal>(
            r#"
            SELECT open_size
            FROM polymarket.trade_positions
            WHERE position_id = $1
              AND status IN ('open', 'partially_closed')
              AND open_size > 0
            FOR UPDATE
            "#,
        )
        .bind(candidate.position_id)
        .fetch_optional(&mut *tx)
        .await
        .context("failed to lock risk-control trade position")?
        else {
            tx.commit()
                .await
                .context("failed to commit missing risk-control position transaction")?;
            return Ok(0);
        };

        let applied_fills = cap_fills_to_size(&report.fills, current_open_size);
        let filled_size: Decimal = applied_fills.iter().map(|fill| fill.size).sum();
        if filled_size <= Decimal::ZERO {
            tx.commit()
                .await
                .context("failed to commit empty risk-control exit transaction")?;
            return Ok(0);
        }
        let exit_fee: Decimal = applied_fills.iter().map(|fill| fill.fee).sum();
        let exit_notional: Decimal = applied_fills
            .iter()
            .map(|fill| fill.price * fill.size)
            .sum();
        let exit_price = exit_notional / filled_size;
        let entry_value = candidate.entry_price * filled_size;
        let gross_pnl = if candidate.side == "buy" {
            exit_notional - entry_value
        } else {
            entry_value - exit_notional
        };
        let allocated_entry_fee = if candidate.entry_size > Decimal::ZERO {
            candidate.entry_fee * (filled_size / candidate.entry_size)
        } else {
            Decimal::ZERO
        };
        let net_pnl = gross_pnl - allocated_entry_fee - exit_fee;
        let reference_notional = candidate.reference_exit_price * filled_size;
        let slippage_cost = if candidate.side == "buy" {
            (reference_notional - exit_notional).max(Decimal::ZERO)
        } else {
            (exit_notional - reference_notional).max(Decimal::ZERO)
        };
        let fill_payloads: Vec<_> = applied_fills
            .iter()
            .map(|fill| serde_json::to_value(fill))
            .collect::<std::result::Result<_, _>>()?;
        let exit_timestamp = applied_fills
            .iter()
            .map(|fill| fill.filled_at)
            .min()
            .unwrap_or(candidate.exit_timestamp);
        let metadata = serde_json::json!({
            "source": candidate.exit_purpose.as_str(),
            "purpose": candidate.exit_purpose.as_str(),
            "close_order_id": close_order_id,
            "reference_exit_price": candidate.reference_exit_price,
            "order_limit_price": candidate.order_limit_price,
            "reference_exit_notional": reference_notional,
            "trigger_roi": candidate.trigger_roi,
            "threshold_roi": candidate.threshold_roi,
            "take_profit_roi": candidate.take_profit_roi,
            "stop_loss_roi": if candidate.exit_purpose == "stop_loss_exit" {
                Some(candidate.threshold_roi)
            } else {
                None
            },
            "latest_mark_timestamp": candidate.latest_mark_timestamp,
            "max_exit_slippage_bps": candidate.max_exit_slippage_bps,
            "allocated_entry_fee": allocated_entry_fee,
            "execution_source": applied_fills
                .first()
                .map(|fill| serialized_name(&fill.source).unwrap_or_else(|_| "unknown".to_string()))
                .unwrap_or_else(|| "unknown".to_string()),
            "applied_fill_size": filled_size,
            "position_open_size_at_lock": current_open_size,
            "fills": fill_payloads
        });

        let inserted = sqlx::query(
            r#"
            INSERT INTO polymarket.trade_exits (
              position_id, process_id, source_signal_id, timestamp_utc, exit_type, is_synthetic,
              exit_trigger_wallet, exit_source_trade_id, exit_price, exit_size,
              exit_notional, exit_fee, slippage_cost, gross_pnl, net_pnl, roi, metadata
            )
            VALUES (
              $1,$2,$3,$4,'risk_control',false,$5,$6,$7,$8,$9,$10,$11,$12,$13,
              CASE WHEN $14 > 0 THEN $13 / $14 ELSE 0 END,
              $15
            )
            ON CONFLICT (position_id, exit_source_trade_id) DO NOTHING
            "#,
        )
        .bind(candidate.position_id)
        .bind(candidate.process_id)
        .bind(candidate.source_signal_id)
        .bind(exit_timestamp)
        .bind(candidate.proxy_wallet.as_deref())
        .bind(candidate.exit_source_trade_id)
        .bind(exit_price)
        .bind(filled_size)
        .bind(exit_notional)
        .bind(exit_fee)
        .bind(slippage_cost)
        .bind(gross_pnl)
        .bind(net_pnl)
        .bind(candidate.entry_notional)
        .bind(metadata)
        .execute(&mut *tx)
        .await
        .context("failed to insert risk-control trade exit")?;

        if inserted.rows_affected() == 0 {
            tx.commit()
                .await
                .context("failed to commit duplicate risk-control exit transaction")?;
            return Ok(0);
        }

        sqlx::query(
            r#"
            UPDATE polymarket.trade_positions p
            SET
              open_size = GREATEST(0, p.open_size - $2),
              realized_pnl = p.realized_pnl + $3,
              status = CASE
                WHEN GREATEST(0, p.open_size - $2) <= 0.000000001 THEN 'closed'
                ELSE 'partially_closed'
              END,
              unrealized_pnl = CASE
                WHEN GREATEST(0, p.open_size - $2) <= 0.000000001 THEN 0
                WHEN p.open_size > 0 THEN p.unrealized_pnl * (GREATEST(0, p.open_size - $2) / p.open_size)
                ELSE 0
              END,
              roi = CASE
                WHEN p.entry_notional > 0 THEN (
                  p.realized_pnl + $3 + CASE
                    WHEN GREATEST(0, p.open_size - $2) <= 0.000000001 THEN 0
                    WHEN p.open_size > 0 THEN p.unrealized_pnl * (GREATEST(0, p.open_size - $2) / p.open_size)
                    ELSE 0
                  END
                ) / p.entry_notional
                ELSE 0
              END,
              updated_at = now()
            WHERE p.position_id = $1
            "#,
        )
        .bind(candidate.position_id)
        .bind(filled_size)
        .bind(net_pnl)
        .execute(&mut *tx)
        .await
        .context("failed to update trade position from risk-control exit")?;

        tx.commit()
            .await
            .context("failed to commit risk-control exit transaction")?;
        Ok(1)
    }

    pub async fn apply_whale_led_trade_exits(&self) -> Result<u64> {
        let result = sqlx::query(
            r#"
            WITH candidates AS (
              SELECT
                p.position_id,
                p.source_signal_id,
                p.proxy_wallet,
                p.side,
                p.entry_price,
                p.entry_size,
                p.open_size,
                p.entry_fee,
                p.entry_notional,
                t.trade_id AS exit_source_trade_id,
                t.timestamp_utc,
                t.price AS exit_price,
                LEAST(
                  p.open_size,
                  CASE
                    WHEN p.entry_notional > 0 THEN p.open_size * LEAST(1, t.cash_value / p.entry_notional)
                    ELSE p.open_size
                  END
                ) AS exit_size
              FROM polymarket.trade_positions p
              JOIN LATERAL (
                SELECT wt.*
                FROM polymarket.wallet_trades wt
                WHERE wt.proxy_wallet = p.proxy_wallet
                  AND wt.asset = p.token_id
                  AND wt.timestamp_utc > p.entry_timestamp
                  AND (
                    (p.side = 'buy' AND upper(wt.side) = 'SELL')
                    OR (p.side = 'sell' AND upper(wt.side) = 'BUY')
                  )
                  AND NOT EXISTS (
                    SELECT 1
                    FROM polymarket.trade_exits te
                    WHERE te.position_id = p.position_id
                      AND te.exit_source_trade_id = wt.trade_id
                  )
                ORDER BY wt.timestamp_utc ASC
                LIMIT 1
              ) t ON true
              WHERE p.status IN ('open', 'partially_closed')
                AND p.open_size > 0
                AND (
                  p.latest_mark_timestamp IS NULL
                  OR latest.timestamp_utc > p.latest_mark_timestamp
                )
            ),
            prepared AS (
              SELECT
                *,
                exit_price * exit_size AS exit_notional,
                CASE
                  WHEN side = 'buy' THEN exit_size * (exit_price - entry_price)
                  ELSE exit_size * (entry_price - exit_price)
                END AS gross_pnl,
                CASE
                  WHEN entry_size > 0 THEN entry_fee * (exit_size / entry_size)
                  ELSE 0
                END AS allocated_entry_fee
              FROM candidates
              WHERE exit_size > 0
            ),
            inserted AS (
              INSERT INTO polymarket.trade_exits (
                position_id, source_signal_id, timestamp_utc, exit_type, is_synthetic,
                exit_trigger_wallet, exit_source_trade_id, exit_price, exit_size,
                exit_notional, exit_fee, slippage_cost, gross_pnl, net_pnl, roi, metadata
              )
              SELECT
                position_id,
                source_signal_id,
                timestamp_utc,
                CASE WHEN exit_size < open_size THEN 'whale_reduce' ELSE 'whale_exit' END,
                true,
                proxy_wallet,
                exit_source_trade_id,
                exit_price,
                exit_size,
                exit_notional,
                0,
                0,
                gross_pnl,
                gross_pnl - allocated_entry_fee,
                CASE WHEN entry_notional > 0 THEN (gross_pnl - allocated_entry_fee) / entry_notional ELSE 0 END,
                jsonb_build_object('source', 'whale_led_exit_detector')
              FROM prepared
              ON CONFLICT (position_id, exit_source_trade_id) DO NOTHING
              RETURNING position_id, exit_size, net_pnl, timestamp_utc
            )
            UPDATE polymarket.trade_positions p
            SET
              open_size = GREATEST(0, p.open_size - i.exit_size),
              realized_pnl = p.realized_pnl + i.net_pnl,
              status = CASE
                WHEN GREATEST(0, p.open_size - i.exit_size) <= 0.000000001 THEN 'closed'
                ELSE 'partially_closed'
              END,
              unrealized_pnl = CASE
                WHEN GREATEST(0, p.open_size - i.exit_size) <= 0.000000001 THEN 0
                WHEN p.open_size > 0 THEN p.unrealized_pnl * (GREATEST(0, p.open_size - i.exit_size) / p.open_size)
                ELSE 0
              END,
              roi = CASE
                WHEN p.entry_notional > 0 THEN (
                  p.realized_pnl + i.net_pnl + CASE
                    WHEN GREATEST(0, p.open_size - i.exit_size) <= 0.000000001 THEN 0
                    WHEN p.open_size > 0 THEN p.unrealized_pnl * (GREATEST(0, p.open_size - i.exit_size) / p.open_size)
                    ELSE 0
                  END
                ) / p.entry_notional
                ELSE 0
              END,
              updated_at = now()
            FROM inserted i
            WHERE p.position_id = i.position_id
            "#,
        )
        .execute(&self.pool)
        .await
        .context("failed to apply whale-led trade exits")?;
        Ok(result.rows_affected())
    }

    pub async fn apply_backtest_whale_led_trade_exits_for_process_until(
        &self,
        process_id: Uuid,
        as_of: DateTime<Utc>,
    ) -> Result<u64> {
        let result = sqlx::query(
            r#"
            WITH candidates AS (
              SELECT
                p.process_id,
                p.position_id,
                p.source_signal_id,
                p.proxy_wallet,
                p.side,
                p.entry_price,
                p.entry_size,
                p.open_size,
                p.entry_fee,
                p.entry_notional,
                t.trade_id AS exit_source_trade_id,
                t.timestamp_utc,
                t.price AS exit_price,
                LEAST(
                  p.open_size,
                  CASE
                    WHEN p.entry_notional > 0 THEN p.open_size * LEAST(1, t.cash_value / p.entry_notional)
                    ELSE p.open_size
                  END
                ) AS exit_size
              FROM polymarket.trade_positions p
              JOIN LATERAL (
                SELECT wt.*
                FROM polymarket.wallet_trades wt
                WHERE wt.proxy_wallet = p.proxy_wallet
                  AND wt.asset = p.token_id
                  AND wt.timestamp_utc > p.entry_timestamp
                  AND wt.timestamp_utc <= $2
                  AND (
                    (p.side = 'buy' AND upper(wt.side) = 'SELL')
                    OR (p.side = 'sell' AND upper(wt.side) = 'BUY')
                  )
                  AND NOT EXISTS (
                    SELECT 1
                    FROM polymarket.trade_exits te
                    WHERE te.position_id = p.position_id
                      AND te.exit_source_trade_id = wt.trade_id
                  )
                ORDER BY wt.timestamp_utc ASC, wt.trade_id ASC
                LIMIT 1
              ) t ON true
              WHERE p.process_id = $1
                AND p.status IN ('open', 'partially_closed')
                AND p.open_size > 0
            ),
            prepared AS (
              SELECT
                *,
                exit_price * exit_size AS exit_notional,
                CASE
                  WHEN side = 'buy' THEN exit_size * (exit_price - entry_price)
                  ELSE exit_size * (entry_price - exit_price)
                END AS gross_pnl,
                CASE
                  WHEN entry_size > 0 THEN entry_fee * (exit_size / entry_size)
                  ELSE 0
                END AS allocated_entry_fee
              FROM candidates
              WHERE exit_size > 0
            ),
            inserted AS (
              INSERT INTO polymarket.trade_exits (
                position_id, process_id, source_signal_id, timestamp_utc, exit_type, is_synthetic,
                exit_trigger_wallet, exit_source_trade_id, exit_price, exit_size,
                exit_notional, exit_fee, slippage_cost, gross_pnl, net_pnl, roi, metadata
              )
              SELECT
                position_id,
                process_id,
                source_signal_id,
                timestamp_utc,
                CASE WHEN exit_size < open_size THEN 'whale_reduce' ELSE 'whale_exit' END,
                true,
                proxy_wallet,
                exit_source_trade_id,
                exit_price,
                exit_size,
                exit_notional,
                0,
                0,
                gross_pnl,
                gross_pnl - allocated_entry_fee,
                CASE WHEN entry_notional > 0 THEN (gross_pnl - allocated_entry_fee) / entry_notional ELSE 0 END,
                jsonb_build_object('source', 'backtest_whale_led_exit', 'as_of', $2)
              FROM prepared
              ON CONFLICT (position_id, exit_source_trade_id) DO NOTHING
              RETURNING position_id, exit_size, net_pnl
            )
            UPDATE polymarket.trade_positions p
            SET
              open_size = GREATEST(0, p.open_size - i.exit_size),
              realized_pnl = p.realized_pnl + i.net_pnl,
              status = CASE
                WHEN GREATEST(0, p.open_size - i.exit_size) <= 0.000000001 THEN 'closed'
                ELSE 'partially_closed'
              END,
              unrealized_pnl = CASE
                WHEN GREATEST(0, p.open_size - i.exit_size) <= 0.000000001 THEN 0
                WHEN p.open_size > 0 THEN p.unrealized_pnl * (GREATEST(0, p.open_size - i.exit_size) / p.open_size)
                ELSE 0
              END,
              roi = CASE
                WHEN p.entry_notional > 0 THEN (
                  p.realized_pnl + i.net_pnl + CASE
                    WHEN GREATEST(0, p.open_size - i.exit_size) <= 0.000000001 THEN 0
                    WHEN p.open_size > 0 THEN p.unrealized_pnl * (GREATEST(0, p.open_size - i.exit_size) / p.open_size)
                    ELSE 0
                  END
                ) / p.entry_notional
                ELSE 0
              END,
              updated_at = now()
            FROM inserted i
            WHERE p.position_id = i.position_id
            "#,
        )
        .bind(process_id)
        .bind(as_of)
        .execute(&self.pool)
        .await
        .context("failed to apply backtest whale-led trade exits")?;
        Ok(result.rows_affected())
    }

    pub async fn mark_open_trade_positions_for_process_as_of(
        &self,
        process_id: Uuid,
        as_of: DateTime<Utc>,
    ) -> Result<u64> {
        let result = sqlx::query(
            r#"
            WITH open_positions AS MATERIALIZED (
              SELECT position_id, process_id, source_signal_id, token_id, side, entry_price,
                open_size, entry_notional, entry_timestamp, realized_pnl, latest_mark_timestamp
              FROM polymarket.trade_positions
              WHERE process_id = $1
                AND status IN ('open', 'partially_closed')
                AND open_size > 0
            ),
            marks AS (
              SELECT
                p.position_id,
                p.process_id,
                p.source_signal_id,
                p.side,
                p.entry_price,
                p.open_size,
                p.entry_notional,
                wt.price AS mark_price,
                wt.timestamp_utc AS source_timestamp,
                wt.trade_id AS wallet_trade_id
              FROM open_positions p
              JOIN LATERAL (
                SELECT trade_id, timestamp_utc, price
                FROM polymarket.wallet_trades wt
                WHERE wt.asset = p.token_id
                  AND wt.timestamp_utc >= p.entry_timestamp
                  AND wt.timestamp_utc <= $2
                ORDER BY wt.timestamp_utc DESC, wt.trade_id DESC
                LIMIT 1
              ) wt ON true
              WHERE p.latest_mark_timestamp IS NULL OR wt.timestamp_utc > p.latest_mark_timestamp
            ),
            prepared AS (
              SELECT
                *,
                CASE
                  WHEN side = 'buy' THEN open_size * (mark_price - entry_price)
                  ELSE open_size * (entry_price - mark_price)
                END AS gross_unrealized_pnl
              FROM marks
            ),
            inserted AS (
              INSERT INTO polymarket.trade_marks (
                position_id, process_id, source_signal_id, timestamp_utc, mark_price, mark_source,
                mark_age_ms, gross_unrealized_pnl, net_unrealized_pnl, roi, metadata
              )
              SELECT
                position_id,
                process_id,
                source_signal_id,
                source_timestamp,
                mark_price,
                'data_api_trade',
                GREATEST(0, floor(extract(epoch from ($2 - source_timestamp)) * 1000))::bigint,
                gross_unrealized_pnl,
                gross_unrealized_pnl,
                CASE WHEN entry_notional > 0 THEN gross_unrealized_pnl / entry_notional ELSE 0 END,
                jsonb_build_object(
                  'source', 'backtest_mark',
                  'as_of', $2,
                  'wallet_trade_id', wallet_trade_id
                )
              FROM prepared
              RETURNING position_id, mark_price, net_unrealized_pnl, timestamp_utc
            )
            UPDATE polymarket.trade_positions p
            SET latest_mark_price = i.mark_price,
                latest_mark_timestamp = i.timestamp_utc,
                unrealized_pnl = i.net_unrealized_pnl,
                roi = CASE
                  WHEN p.entry_notional > 0 THEN (p.realized_pnl + i.net_unrealized_pnl) / p.entry_notional
                  ELSE 0
                END,
                updated_at = now()
            FROM inserted i
            WHERE p.position_id = i.position_id
            "#,
        )
        .bind(process_id)
        .bind(as_of)
        .execute(&self.pool)
        .await
        .context("failed to mark backtest open trade positions")?;
        Ok(result.rows_affected())
    }

    pub async fn clear_closed_trade_position_unrealized_pnl(&self) -> Result<u64> {
        let result = sqlx::query(
            r#"
            UPDATE polymarket.trade_positions
            SET
              unrealized_pnl = 0,
              roi = CASE
                WHEN entry_notional > 0 THEN realized_pnl / entry_notional
                ELSE 0
              END,
              updated_at = now()
            WHERE status IN ('closed', 'resolved')
              AND unrealized_pnl <> 0
            "#,
        )
        .execute(&self.pool)
        .await
        .context("failed to clear closed trade position unrealized PnL")?;
        Ok(result.rows_affected())
    }

    pub async fn mark_open_trade_positions(&self) -> Result<u64> {
        let result = sqlx::query(MARK_OPEN_TRADE_POSITIONS_SQL)
            .bind(Option::<Uuid>::None)
            .execute(&self.pool)
            .await
            .context("failed to mark open trade positions")?;
        Ok(result.rows_affected())
    }

    pub async fn mark_open_trade_positions_for_process(&self, process_id: Uuid) -> Result<u64> {
        let result = sqlx::query(MARK_OPEN_TRADE_POSITIONS_SQL)
            .bind(Some(process_id))
            .execute(&self.pool)
            .await
            .context("failed to mark open trade positions for process")?;
        Ok(result.rows_affected())
    }

    pub async fn open_trade_position_mark_tokens(
        &self,
        limit: i64,
        max_mark_age: chrono::Duration,
        failure_backoff: chrono::Duration,
    ) -> Result<Vec<OpenMarkToken>> {
        self.open_trade_position_mark_tokens_for_process(
            None,
            limit,
            max_mark_age,
            failure_backoff,
            true,
        )
        .await
    }

    pub async fn open_trade_position_mark_tokens_for_process(
        &self,
        process_id: Option<Uuid>,
        limit: i64,
        max_mark_age: chrono::Duration,
        failure_backoff: chrono::Duration,
        stale_only: bool,
    ) -> Result<Vec<OpenMarkToken>> {
        let max_mark_age_ms = max_mark_age.num_milliseconds().max(0);
        let failure_backoff_ms = failure_backoff.num_milliseconds().max(0);
        let rows = sqlx::query_as::<_, OpenMarkToken>(
            r#"
            WITH stale_tokens AS MATERIALIZED (
              SELECT
                p.process_id,
                p.token_id,
                min(p.market_id) FILTER (WHERE p.market_id IS NOT NULL) AS market_id,
                min(COALESCE(p.latest_mark_timestamp, p.entry_timestamp)) AS oldest_mark_at
              FROM polymarket.trade_positions p
              WHERE p.status IN ('open', 'partially_closed')
                AND p.open_size > 0
                AND ($4::uuid IS NULL OR p.process_id IS NOT DISTINCT FROM $4::uuid)
                AND (
                  $5::boolean = false
                  OR
                  p.latest_mark_timestamp IS NULL
                  OR p.latest_mark_timestamp < now() - ($1::bigint * interval '1 millisecond')
                )
              GROUP BY p.process_id, p.token_id
            )
            SELECT process_id, token_id, market_id
            FROM stale_tokens t
            WHERE NOT EXISTS (
              SELECT 1
              FROM polymarket.orderbook_snapshots b
              WHERE b.token_id = t.token_id
                AND b.timestamp_utc >= now() - ($1::bigint * interval '1 millisecond')
                AND (b.best_bid IS NOT NULL OR b.best_ask IS NOT NULL)
              LIMIT 1
            )
              AND NOT EXISTS (
                SELECT 1
                FROM polymarket.trade_mark_source_failures f
                WHERE f.process_id IS NOT DISTINCT FROM t.process_id
                  AND f.token_id = t.token_id
                  AND f.failure_source = 'mark_orderbook_refresh'
                  AND f.resolved_at IS NULL
                  AND f.last_failed_at >= now() - ($2::bigint * interval '1 millisecond')
                LIMIT 1
              )
            ORDER BY oldest_mark_at ASC
            LIMIT $3
            "#,
        )
        .bind(max_mark_age_ms)
        .bind(failure_backoff_ms)
        .bind(limit.max(0))
        .bind(process_id)
        .bind(stale_only)
        .fetch_all(&self.pool)
        .await
        .context("failed to list open trade position tokens needing marks")?;
        Ok(rows)
    }

    pub async fn refresh_wallet_trade_performance(&self) -> Result<u64> {
        let result = sqlx::query(
            r#"
            WITH wallet_stats AS (
              SELECT
                proxy_wallet,
                strategy_name,
                execution_mode,
                venue,
                bool_or(is_live_capital) AS is_live_capital,
                count(*) FILTER (WHERE status IN ('open', 'partially_closed'))::integer AS open_positions,
                count(*) FILTER (WHERE status IN ('closed', 'resolved'))::integer AS closed_positions,
                count(*) FILTER (WHERE status IN ('closed', 'resolved') AND realized_pnl > 0)::integer AS winning_positions,
                count(*) FILTER (WHERE status IN ('closed', 'resolved') AND realized_pnl < 0)::integer AS losing_positions,
                COALESCE(sum(realized_pnl) FILTER (WHERE status IN ('closed', 'resolved')), 0) AS net_pnl,
                COALESCE(sum(unrealized_pnl) FILTER (WHERE status IN ('open', 'partially_closed')), 0) AS unrealized_pnl,
                COALESCE(sum(entry_notional), 0) AS total_notional,
                COALESCE(sum(realized_pnl) FILTER (WHERE status IN ('closed', 'resolved') AND realized_pnl > 0), 0) AS gross_wins,
                abs(COALESCE(sum(realized_pnl) FILTER (WHERE status IN ('closed', 'resolved') AND realized_pnl < 0), 0)) AS gross_losses
              FROM polymarket.trade_positions
              WHERE proxy_wallet IS NOT NULL
              GROUP BY proxy_wallet, strategy_name, execution_mode, venue
            )
            INSERT INTO polymarket.wallet_trade_performance (
              proxy_wallet, strategy_name, execution_mode, venue, is_live_capital,
              open_positions, closed_positions, winning_positions, losing_positions,
              gross_pnl, net_pnl, unrealized_pnl, total_notional, roi, win_rate,
              profit_factor, avg_win, avg_loss, max_drawdown, last_updated_at, metadata
            )
            SELECT
              proxy_wallet,
              strategy_name,
              execution_mode,
              venue,
              is_live_capital,
              open_positions,
              closed_positions,
              winning_positions,
              losing_positions,
              net_pnl,
              net_pnl,
              unrealized_pnl,
              total_notional,
              CASE WHEN total_notional > 0 THEN (net_pnl + unrealized_pnl) / total_notional ELSE 0 END,
              CASE WHEN closed_positions > 0 THEN winning_positions::numeric / closed_positions ELSE 0 END,
              CASE
                WHEN gross_losses > 0 THEN gross_wins / gross_losses
                WHEN gross_wins > 0 THEN gross_wins
                ELSE 0
              END,
              CASE WHEN winning_positions > 0 THEN gross_wins / winning_positions ELSE 0 END,
              CASE WHEN losing_positions > 0 THEN -gross_losses / losing_positions ELSE 0 END,
              0,
              now(),
              jsonb_build_object('source', 'trade_lifecycle')
            FROM wallet_stats
            ON CONFLICT (proxy_wallet, strategy_name, execution_mode, venue) DO UPDATE SET
              is_live_capital = EXCLUDED.is_live_capital,
              open_positions = EXCLUDED.open_positions,
              closed_positions = EXCLUDED.closed_positions,
              winning_positions = EXCLUDED.winning_positions,
              losing_positions = EXCLUDED.losing_positions,
              gross_pnl = EXCLUDED.gross_pnl,
              net_pnl = EXCLUDED.net_pnl,
              unrealized_pnl = EXCLUDED.unrealized_pnl,
              total_notional = EXCLUDED.total_notional,
              roi = EXCLUDED.roi,
              win_rate = EXCLUDED.win_rate,
              profit_factor = EXCLUDED.profit_factor,
              avg_win = EXCLUDED.avg_win,
              avg_loss = EXCLUDED.avg_loss,
              max_drawdown = EXCLUDED.max_drawdown,
              last_updated_at = now(),
              metadata = polymarket.wallet_trade_performance.metadata || EXCLUDED.metadata
            "#,
        )
        .execute(&self.pool)
        .await
        .context("failed to refresh wallet trade performance")?;
        Ok(result.rows_affected())
    }

    pub async fn trade_pnl_summary(
        &self,
        mark_fresh_max_age: chrono::Duration,
    ) -> Result<serde_json::Value> {
        let mark_fresh_max_age_ms = mark_fresh_max_age.num_milliseconds().max(0);
        let value = sqlx::query_scalar::<_, serde_json::Value>(
            r#"
            WITH position_stats AS (
              SELECT
                COALESCE(process_id::text, 'unbound') AS process_key,
                count(*) AS positions,
                count(*) FILTER (WHERE status IN ('open', 'partially_closed')) AS open_positions,
                count(*) FILTER (WHERE status IN ('closed', 'resolved')) AS closed_positions,
                COALESCE(sum(realized_pnl), 0) AS realized_pnl,
                COALESCE(sum(unrealized_pnl) FILTER (WHERE status IN ('open', 'partially_closed')), 0) AS unrealized_pnl,
                COALESCE(sum(entry_notional), 0) AS notional,
                count(*) FILTER (WHERE status IN ('closed', 'resolved') AND realized_pnl > 0) AS winning_closed_positions,
                count(*) FILTER (WHERE status IN ('closed', 'resolved') AND realized_pnl < 0) AS losing_closed_positions
              FROM polymarket.trade_positions
              GROUP BY COALESCE(process_id::text, 'unbound')
            ),
            total_stats AS (
              SELECT
                count(*) AS positions,
                count(*) FILTER (WHERE status IN ('open', 'partially_closed')) AS open_positions,
                count(*) FILTER (WHERE status IN ('closed', 'resolved')) AS closed_positions,
                COALESCE(sum(realized_pnl), 0) AS realized_pnl,
                COALESCE(sum(unrealized_pnl) FILTER (WHERE status IN ('open', 'partially_closed')), 0) AS unrealized_pnl,
                COALESCE(sum(entry_notional), 0) AS notional,
                count(*) FILTER (WHERE status IN ('closed', 'resolved') AND realized_pnl > 0) AS winning_closed_positions,
                count(*) FILTER (WHERE status IN ('closed', 'resolved') AND realized_pnl < 0) AS losing_closed_positions
              FROM polymarket.trade_positions
            ),
            open_positions AS MATERIALIZED (
              SELECT
                p.position_id,
                p.process_id,
                p.token_id,
                p.entry_notional,
                p.unrealized_pnl,
                p.latest_mark_timestamp,
                EXISTS (
                  SELECT 1
                  FROM polymarket.trade_mark_source_failures f
                  WHERE f.process_id IS NOT DISTINCT FROM p.process_id
                    AND f.token_id = p.token_id
                    AND f.failure_source = 'mark_orderbook_refresh'
                    AND f.resolved_at IS NULL
                  LIMIT 1
                ) AS has_unresolved_mark_failure
              FROM polymarket.trade_positions p
              WHERE p.status IN ('open', 'partially_closed')
                AND p.open_size > 0
            ),
            mark_position_states AS (
              SELECT
                *,
                CASE
                  WHEN has_unresolved_mark_failure THEN 'unavailable'
                  WHEN latest_mark_timestamp IS NULL THEN 'missing'
                  WHEN latest_mark_timestamp < now() - ($1::bigint * interval '1 millisecond') THEN 'stale'
                  ELSE 'fresh'
                END AS mark_state,
                CASE
                  WHEN latest_mark_timestamp IS NULL THEN NULL
                  ELSE GREATEST(0, floor(extract(epoch from (now() - latest_mark_timestamp))))::bigint
                END AS mark_age_secs
              FROM open_positions
            ),
            mark_totals AS (
              SELECT
                count(*) AS open_positions,
                count(*) FILTER (WHERE mark_state = 'fresh') AS fresh_mark_positions,
                count(*) FILTER (WHERE mark_state = 'stale') AS stale_mark_positions,
                count(*) FILTER (WHERE mark_state = 'missing') AS missing_mark_positions,
                count(*) FILTER (WHERE mark_state = 'unavailable') AS unavailable_mark_positions,
                max(mark_age_secs) AS oldest_mark_age_secs,
                COALESCE(sum(entry_notional) FILTER (WHERE mark_state <> 'fresh'), 0) AS degraded_notional,
                COALESCE(sum(unrealized_pnl) FILTER (WHERE mark_state <> 'fresh'), 0) AS degraded_unrealized_pnl
              FROM mark_position_states
            ),
            mark_failures AS (
              SELECT
                count(*) AS unresolved_mark_failures,
                COALESCE(sum(failure_count), 0) AS unresolved_mark_failure_events,
                max(last_failed_at) AS last_mark_failure_at
              FROM polymarket.trade_mark_source_failures
              WHERE failure_source = 'mark_orderbook_refresh'
                AND resolved_at IS NULL
                AND EXISTS (
                  SELECT 1
                  FROM open_positions p
                  WHERE p.process_id IS NOT DISTINCT FROM polymarket.trade_mark_source_failures.process_id
                    AND p.token_id = polymarket.trade_mark_source_failures.token_id
                )
            ),
            mark_state_counts AS (
              SELECT
                mark_state,
                count(*) AS positions,
                COALESCE(sum(entry_notional), 0) AS notional,
                COALESCE(sum(unrealized_pnl), 0) AS unrealized_pnl
              FROM mark_position_states
              GROUP BY mark_state
            )
            SELECT jsonb_build_object(
              'positions', total_stats.positions,
              'open_positions', total_stats.open_positions,
              'closed_positions', total_stats.closed_positions,
              'realized_pnl', total_stats.realized_pnl,
              'unrealized_pnl', total_stats.unrealized_pnl,
              'total_pnl', total_stats.realized_pnl + total_stats.unrealized_pnl,
              'notional', total_stats.notional,
              'roi', CASE WHEN total_stats.notional > 0
                THEN (total_stats.realized_pnl + total_stats.unrealized_pnl) / total_stats.notional
                ELSE 0
              END,
              'winning_closed_positions', total_stats.winning_closed_positions,
              'losing_closed_positions', total_stats.losing_closed_positions,
              'reports_by_process_id', COALESCE((
                SELECT jsonb_object_agg(
                  process_key,
                  jsonb_build_object(
                    'positions', positions,
                    'open_positions', open_positions,
                    'closed_positions', closed_positions,
                    'realized_pnl', realized_pnl,
                    'unrealized_pnl', unrealized_pnl,
                    'total_pnl', realized_pnl + unrealized_pnl,
                    'notional', notional,
                    'roi', CASE WHEN notional > 0
                      THEN (realized_pnl + unrealized_pnl) / notional
                      ELSE 0
                    END,
                    'winning_closed_positions', winning_closed_positions,
                    'losing_closed_positions', losing_closed_positions,
                    'updated_at', now()
                  )
                  ORDER BY process_key
                )
                FROM position_stats
              ), '{}'::jsonb),
              'mark_readiness', jsonb_build_object(
                'status', CASE
                  WHEN mark_totals.open_positions = 0 THEN 'reliable'
                  WHEN mark_totals.fresh_mark_positions = mark_totals.open_positions THEN 'reliable'
                  WHEN mark_totals.fresh_mark_positions = 0 THEN 'unavailable'
                  ELSE 'degraded'
                END,
                'max_fresh_age_secs', ($1::bigint / 1000),
                'open_positions', mark_totals.open_positions,
                'fresh_mark_positions', mark_totals.fresh_mark_positions,
                'stale_mark_positions', mark_totals.stale_mark_positions,
                'missing_mark_positions', mark_totals.missing_mark_positions,
                'unavailable_mark_positions', mark_totals.unavailable_mark_positions,
                'oldest_mark_age_secs', mark_totals.oldest_mark_age_secs,
                'degraded_notional', mark_totals.degraded_notional,
                'degraded_unrealized_pnl', mark_totals.degraded_unrealized_pnl,
                'unresolved_mark_failures', mark_failures.unresolved_mark_failures,
                'unresolved_mark_failure_events', mark_failures.unresolved_mark_failure_events,
                'last_mark_failure_at', mark_failures.last_mark_failure_at,
                'states', COALESCE((
                  SELECT jsonb_object_agg(
                    mark_state,
                    jsonb_build_object(
                      'positions', positions,
                      'notional', notional,
                      'unrealized_pnl', unrealized_pnl
                    )
                    ORDER BY mark_state
                  )
                  FROM mark_state_counts
                ), '{}'::jsonb)
              ),
              'updated_at', now()
            )
            FROM total_stats
            CROSS JOIN mark_totals
            CROSS JOIN mark_failures
            "#,
        )
        .bind(mark_fresh_max_age_ms)
        .fetch_one(&self.pool)
        .await
        .context("failed to fetch trade PnL summary")?;
        Ok(value)
    }

    pub async fn wallet_trade_performance_rows(&self, limit: i64) -> Result<serde_json::Value> {
        let value = sqlx::query_scalar::<_, serde_json::Value>(
            r#"
            SELECT COALESCE(jsonb_agg(to_jsonb(rows) ORDER BY rows.net_pnl DESC), '[]'::jsonb)
            FROM (
              SELECT *
              FROM polymarket.wallet_trade_performance
              ORDER BY net_pnl DESC, unrealized_pnl DESC
              LIMIT $1
            ) rows
            "#,
        )
        .bind(limit)
        .fetch_one(&self.pool)
        .await
        .context("failed to fetch wallet trade performance rows")?;
        Ok(value)
    }

    pub async fn open_trade_position_rows(&self, limit: i64) -> Result<serde_json::Value> {
        let value = sqlx::query_scalar::<_, serde_json::Value>(
            r#"
            SELECT COALESCE(jsonb_agg(to_jsonb(rows) ORDER BY rows.updated_at DESC), '[]'::jsonb)
            FROM (
              SELECT *
              FROM polymarket.trade_positions
              WHERE status IN ('open', 'partially_closed')
              ORDER BY updated_at DESC
              LIMIT $1
            ) rows
            "#,
        )
        .bind(limit)
        .fetch_one(&self.pool)
        .await
        .context("failed to fetch open trade positions")?;
        Ok(value)
    }

    pub async fn trade_pnl_mark_health(
        &self,
        process_id: Option<Uuid>,
        limit: i64,
    ) -> Result<serde_json::Value> {
        self.trade_pnl_mark_health_with_freshness(process_id, limit, chrono::Duration::minutes(5))
            .await
    }

    pub async fn trade_pnl_mark_health_with_freshness(
        &self,
        process_id: Option<Uuid>,
        limit: i64,
        mark_fresh_max_age: chrono::Duration,
    ) -> Result<serde_json::Value> {
        let mark_fresh_max_age_ms = mark_fresh_max_age.num_milliseconds().max(0);
        let value = sqlx::query_scalar::<_, serde_json::Value>(
            r#"
            WITH open_positions AS MATERIALIZED (
              SELECT *
              FROM polymarket.trade_positions
              WHERE status IN ('open', 'partially_closed')
                AND open_size > 0
                AND ($1::uuid IS NULL OR process_id = $1)
            ),
            coverage AS (
              SELECT
                CASE WHEN latest_mark_timestamp IS NULL THEN 'unmarked' ELSE 'marked' END AS state,
                count(*) AS positions,
                COALESCE(sum(entry_notional), 0) AS notional,
                COALESCE(sum(unrealized_pnl), 0) AS unrealized_pnl
              FROM open_positions
              GROUP BY 1
            ),
            coverage_by_process AS (
              SELECT
                process_id,
                CASE WHEN latest_mark_timestamp IS NULL THEN 'unmarked' ELSE 'marked' END AS state,
                count(*) AS positions,
                COALESCE(sum(entry_notional), 0) AS notional,
                COALESCE(sum(unrealized_pnl), 0) AS unrealized_pnl
              FROM open_positions
              GROUP BY 1, 2
            ),
            stale AS (
              SELECT
                CASE
                  WHEN latest_mark_timestamp IS NULL THEN 'unmarked'
                  WHEN latest_mark_timestamp >= now() - ($2::bigint * interval '1 millisecond') THEN 'fresh'
                  ELSE 'stale'
                END AS bucket,
                count(*) AS positions,
                COALESCE(sum(entry_notional), 0) AS notional
              FROM open_positions
              GROUP BY 1
            ),
            unmarked AS MATERIALIZED (
              SELECT
                p.position_id,
                p.process_id,
                p.market_id,
                p.token_id,
                p.entry_timestamp,
                p.entry_notional,
                EXISTS (
                  SELECT 1
                  FROM polymarket.orderbook_snapshots o
                  WHERE o.token_id = p.token_id
                    AND o.timestamp_utc >= greatest(now() - ($2::bigint * interval '1 millisecond'), p.entry_timestamp)
                    AND (o.best_bid IS NOT NULL OR o.best_ask IS NOT NULL)
                  LIMIT 1
                ) AS has_usable_orderbook,
                EXISTS (
                  SELECT 1
                  FROM polymarket.wallet_trades wt
                  WHERE wt.asset = p.token_id
                    AND wt.timestamp_utc >= p.entry_timestamp
                  LIMIT 1
                ) AS has_post_entry_wallet_trade
              FROM open_positions p
              WHERE p.latest_mark_timestamp IS NULL
            ),
            unmarked_availability AS (
              SELECT
                has_usable_orderbook,
                has_post_entry_wallet_trade,
                count(*) AS positions,
                COALESCE(sum(entry_notional), 0) AS notional
              FROM unmarked
              GROUP BY 1, 2
            ),
            failure_reasons AS (
              SELECT
                failure_source,
                failure_reason,
                count(*) AS tokens,
                COALESCE(sum(failure_count), 0) AS failures,
                max(last_failed_at) AS last_failed_at
              FROM polymarket.trade_mark_source_failures
              WHERE resolved_at IS NULL
                AND ($1::uuid IS NULL OR process_id = $1)
              GROUP BY 1, 2
            ),
            oldest_unmarked AS (
              SELECT
                u.position_id,
                u.process_id,
                u.market_id,
                u.token_id,
                u.entry_timestamp,
                u.entry_notional,
                u.has_usable_orderbook,
                u.has_post_entry_wallet_trade,
                f.failure_source,
                f.failure_reason,
                f.failure_count,
                f.last_failed_at
              FROM unmarked u
              LEFT JOIN LATERAL (
                SELECT failure_source, failure_reason, failure_count, last_failed_at
                FROM polymarket.trade_mark_source_failures f
                WHERE f.token_id = u.token_id
                  AND f.process_id IS NOT DISTINCT FROM u.process_id
                  AND f.resolved_at IS NULL
                ORDER BY f.last_failed_at DESC
                LIMIT 1
              ) f ON true
              ORDER BY u.entry_timestamp ASC
              LIMIT $3
            )
            SELECT jsonb_build_object(
              'process_id', $1::uuid,
              'max_fresh_age_secs', ($2::bigint / 1000),
              'coverage', COALESCE((SELECT jsonb_agg(to_jsonb(coverage) ORDER BY state) FROM coverage), '[]'::jsonb),
              'coverage_by_process', COALESCE((SELECT jsonb_agg(to_jsonb(coverage_by_process) ORDER BY process_id, state) FROM coverage_by_process), '[]'::jsonb),
              'stale', COALESCE((SELECT jsonb_agg(to_jsonb(stale) ORDER BY bucket) FROM stale), '[]'::jsonb),
              'unmarked_availability', COALESCE((SELECT jsonb_agg(to_jsonb(unmarked_availability) ORDER BY has_usable_orderbook, has_post_entry_wallet_trade) FROM unmarked_availability), '[]'::jsonb),
              'failure_reasons', COALESCE((SELECT jsonb_agg(to_jsonb(failure_reasons) ORDER BY failures DESC, last_failed_at DESC) FROM failure_reasons), '[]'::jsonb),
              'oldest_unmarked', COALESCE((SELECT jsonb_agg(to_jsonb(oldest_unmarked) ORDER BY entry_timestamp ASC) FROM oldest_unmarked), '[]'::jsonb),
              'updated_at', now()
            )
            "#,
        )
        .bind(process_id)
        .bind(mark_fresh_max_age_ms)
        .bind(limit)
        .fetch_one(&self.pool)
        .await
        .context("failed to fetch trade PnL mark health")?;
        Ok(value)
    }

    pub async fn recent_trade_exit_rows(&self, limit: i64) -> Result<serde_json::Value> {
        let value = sqlx::query_scalar::<_, serde_json::Value>(
            r#"
            SELECT COALESCE(jsonb_agg(to_jsonb(rows) ORDER BY rows.timestamp_utc DESC), '[]'::jsonb)
            FROM (
              SELECT e.*, p.proxy_wallet, p.strategy_name, p.execution_mode, p.venue, p.side, p.entry_price
              FROM polymarket.trade_exits e
              JOIN polymarket.trade_positions p ON p.position_id = e.position_id
              ORDER BY e.timestamp_utc DESC
              LIMIT $1
            ) rows
            "#,
        )
        .bind(limit)
        .fetch_one(&self.pool)
        .await
        .context("failed to fetch recent trade exits")?;
        Ok(value)
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

    pub async fn create_backfill_job(
        &self,
        job_id: Uuid,
        lookback_days: i32,
        min_trade_usd: Decimal,
        request: serde_json::Value,
    ) -> Result<BackfillJob> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.backfill_jobs (
              job_id, job_type, status, requested_at, lookback_days, min_trade_usd,
              request, summary
            )
            VALUES ($1,'whales','queued',now(),$2,$3,$4,'{}'::jsonb)
            "#,
        )
        .bind(job_id)
        .bind(lookback_days)
        .bind(min_trade_usd)
        .bind(&request)
        .execute(&self.pool)
        .await
        .context("failed to create backfill job")?;
        self.get_backfill_job(job_id).await
    }

    pub async fn get_backfill_job(&self, job_id: Uuid) -> Result<BackfillJob> {
        let row = sqlx::query_as::<_, BackfillJobRow>(
            r#"
            SELECT job_id, job_type, status, requested_at, started_at, completed_at,
              cancel_requested_at, lookback_days, min_trade_usd, request, summary, error
            FROM polymarket.backfill_jobs
            WHERE job_id = $1
            "#,
        )
        .bind(job_id)
        .fetch_one(&self.pool)
        .await
        .context("failed to fetch backfill job")?;
        row.try_into()
    }

    pub async fn list_backfill_jobs(&self, limit: i64) -> Result<Vec<BackfillJob>> {
        let rows = sqlx::query_as::<_, BackfillJobRow>(
            r#"
            SELECT job_id, job_type, status, requested_at, started_at, completed_at,
              cancel_requested_at, lookback_days, min_trade_usd, request, summary, error
            FROM polymarket.backfill_jobs
            ORDER BY requested_at DESC
            LIMIT $1
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .context("failed to list backfill jobs")?;
        rows.into_iter().map(TryInto::try_into).collect()
    }

    pub async fn mark_backfill_job_running(&self, job_id: Uuid) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE polymarket.backfill_jobs
            SET status='running', started_at=COALESCE(started_at, now()), updated_at=now()
            WHERE job_id=$1
            "#,
        )
        .bind(job_id)
        .execute(&self.pool)
        .await
        .context("failed to mark backfill job running")?;
        Ok(())
    }

    pub async fn complete_backfill_job(
        &self,
        job_id: Uuid,
        status: BackfillJobStatus,
        summary: serde_json::Value,
        error: Option<String>,
    ) -> Result<()> {
        let status = serialized_name(&status)?;
        sqlx::query(
            r#"
            UPDATE polymarket.backfill_jobs
            SET status=$2, completed_at=now(), summary=$3, error=$4, updated_at=now()
            WHERE job_id=$1
            "#,
        )
        .bind(job_id)
        .bind(status)
        .bind(summary)
        .bind(error)
        .execute(&self.pool)
        .await
        .context("failed to complete backfill job")?;
        Ok(())
    }

    pub async fn request_backfill_cancel(&self, job_id: Uuid) -> Result<BackfillJob> {
        sqlx::query(
            r#"
            UPDATE polymarket.backfill_jobs
            SET status='cancel_requested', cancel_requested_at=now(), updated_at=now()
            WHERE job_id=$1 AND status IN ('queued','running')
            "#,
        )
        .bind(job_id)
        .execute(&self.pool)
        .await
        .context("failed to request backfill cancellation")?;
        self.get_backfill_job(job_id).await
    }

    pub async fn insert_backfill_event(
        &self,
        job_id: Uuid,
        level: &str,
        message: &str,
        metadata: serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.backfill_job_events (
              event_id, job_id, timestamp_utc, level, message, metadata
            )
            VALUES (gen_random_uuid(),$1,now(),$2,$3,$4)
            "#,
        )
        .bind(job_id)
        .bind(level)
        .bind(message)
        .bind(metadata)
        .execute(&self.pool)
        .await
        .context("failed to insert backfill event")?;
        Ok(())
    }

    pub async fn upsert_whale_trade(&self, trade: &WhaleTrade) -> Result<bool> {
        let result = sqlx::query(
            r#"
            INSERT INTO polymarket.wallet_trades (
              trade_id, proxy_wallet, asset, condition_id, market_id, side, outcome,
              price, size, cash_value, timestamp_utc, title, slug, event_slug,
              transaction_hash, raw_payload
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
            ON CONFLICT ON CONSTRAINT uq_polymarket_wallet_trades_identity DO NOTHING
            "#,
        )
        .bind(trade.trade_id)
        .bind(&trade.proxy_wallet)
        .bind(&trade.asset)
        .bind(&trade.condition_id)
        .bind(&trade.market_id)
        .bind(&trade.side)
        .bind(&trade.outcome)
        .bind(trade.price)
        .bind(trade.size)
        .bind(trade.cash_value)
        .bind(trade.timestamp_utc)
        .bind(&trade.title)
        .bind(&trade.slug)
        .bind(&trade.event_slug)
        .bind(&trade.transaction_hash)
        .bind(&trade.raw_payload)
        .execute(&self.pool)
        .await
        .context("failed to upsert whale trade")?;
        Ok(result.rows_affected() > 0)
    }

    pub async fn fetch_whale_trade_by_identity(
        &self,
        trade: &WhaleTrade,
    ) -> Result<Option<WhaleTrade>> {
        let row = sqlx::query_as::<_, WhaleTradeRow>(
            r#"
            SELECT trade_id, proxy_wallet, asset, condition_id, market_id, side, outcome,
              price, size, cash_value, timestamp_utc, title, slug, event_slug,
              transaction_hash, raw_payload
            FROM polymarket.wallet_trades
            WHERE transaction_hash IS NOT DISTINCT FROM $1
              AND proxy_wallet = $2
              AND asset = $3
              AND side = $4
              AND price = $5
              AND size = $6
              AND timestamp_utc = $7
            ORDER BY created_at ASC
            LIMIT 1
            "#,
        )
        .bind(&trade.transaction_hash)
        .bind(&trade.proxy_wallet)
        .bind(&trade.asset)
        .bind(&trade.side)
        .bind(trade.price)
        .bind(trade.size)
        .bind(trade.timestamp_utc)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch whale trade by identity")?;
        Ok(row.map(Into::into))
    }

    pub async fn fetch_whale_trade_by_copy_identity(
        &self,
        trade: &WhaleTrade,
    ) -> Result<Option<WhaleTrade>> {
        let row = sqlx::query_as::<_, WhaleTradeRow>(
            r#"
            SELECT trade_id, proxy_wallet, asset, condition_id, market_id, side, outcome,
              price, size, cash_value, timestamp_utc, title, slug, event_slug,
              transaction_hash, raw_payload
            FROM polymarket.wallet_trades
            WHERE proxy_wallet = $1
              AND asset = $2
              AND side = $3
              AND price = $4
              AND timestamp_utc = $5
            ORDER BY created_at ASC
            LIMIT 1
            "#,
        )
        .bind(&trade.proxy_wallet)
        .bind(&trade.asset)
        .bind(&trade.side)
        .bind(trade.price)
        .bind(trade.timestamp_utc)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch whale trade by copy identity")?;
        Ok(row.map(Into::into))
    }

    pub async fn ensure_whale_wallet(&self, trade: &WhaleTrade) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.wallets (
              proxy_wallet, first_seen_at, last_seen_at, raw_payload, updated_at
            )
            VALUES ($1,$2,$2,$3,now())
            ON CONFLICT (proxy_wallet) DO UPDATE SET
              first_seen_at = LEAST(polymarket.wallets.first_seen_at, EXCLUDED.first_seen_at),
              last_seen_at = GREATEST(polymarket.wallets.last_seen_at, EXCLUDED.last_seen_at),
              raw_payload = EXCLUDED.raw_payload,
              updated_at = now()
            "#,
        )
        .bind(&trade.proxy_wallet)
        .bind(trade.timestamp_utc)
        .bind(&trade.raw_payload)
        .execute(&self.pool)
        .await
        .context("failed to ensure whale wallet")?;
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

    pub async fn fetch_recent_whale_trades(&self, since: DateTime<Utc>) -> Result<Vec<WhaleTrade>> {
        let rows = sqlx::query_as::<_, WhaleTradeRow>(
            r#"
            SELECT trade_id, proxy_wallet, asset, condition_id, market_id, side, outcome,
              price, size, cash_value, timestamp_utc, title, slug, event_slug,
              transaction_hash, raw_payload
            FROM polymarket.wallet_trades
            WHERE timestamp_utc >= $1
            "#,
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch recent whale trades")?;
        Ok(rows.into_iter().map(Into::into).collect())
    }

    pub async fn fetch_whale_trades_for_replay(
        &self,
        range_start: DateTime<Utc>,
        range_end: DateTime<Utc>,
        min_trade_usd: Decimal,
        limit: i64,
    ) -> Result<Vec<WhaleTrade>> {
        let rows = sqlx::query_as::<_, WhaleTradeRow>(
            r#"
            SELECT trade_id, proxy_wallet, asset, condition_id, market_id, side, outcome,
              price, size, cash_value, timestamp_utc, title, slug, event_slug,
              transaction_hash, raw_payload
            FROM polymarket.wallet_trades
            WHERE timestamp_utc >= $1
              AND timestamp_utc <= $2
              AND cash_value >= $3
            ORDER BY timestamp_utc ASC, trade_id ASC
            LIMIT $4
            "#,
        )
        .bind(range_start)
        .bind(range_end)
        .bind(min_trade_usd)
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch whale trades for replay")?;
        Ok(rows.into_iter().map(Into::into).collect())
    }

    pub async fn fetch_recent_whale_wallets(&self, since: DateTime<Utc>) -> Result<Vec<String>> {
        let rows = sqlx::query_scalar::<_, String>(
            r#"
            SELECT DISTINCT proxy_wallet
            FROM polymarket.wallet_trades
            WHERE timestamp_utc >= $1
            ORDER BY proxy_wallet
            "#,
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await
        .context("failed to fetch recent whale wallets")?;
        Ok(rows)
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

    pub async fn upsert_expectancy_flow_cell(&self, cell: &ExpectancyFlowCell) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.expectancy_flow_cells (
              process_id, score_version, cell_key, dimensions, horizon_secs, lookback_days,
              sample_count, winning_count, losing_count, observed_volume_usd, realized_pnl_usd,
              mean_price_delta, mean_return, win_rate, expectancy, confidence,
              sample_start, sample_end, metadata, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,now())
            ON CONFLICT (process_id, score_version, cell_key) DO UPDATE SET
              dimensions = EXCLUDED.dimensions,
              horizon_secs = EXCLUDED.horizon_secs,
              lookback_days = EXCLUDED.lookback_days,
              sample_count = EXCLUDED.sample_count,
              winning_count = EXCLUDED.winning_count,
              losing_count = EXCLUDED.losing_count,
              observed_volume_usd = EXCLUDED.observed_volume_usd,
              realized_pnl_usd = EXCLUDED.realized_pnl_usd,
              mean_price_delta = EXCLUDED.mean_price_delta,
              mean_return = EXCLUDED.mean_return,
              win_rate = EXCLUDED.win_rate,
              expectancy = EXCLUDED.expectancy,
              confidence = EXCLUDED.confidence,
              sample_start = EXCLUDED.sample_start,
              sample_end = EXCLUDED.sample_end,
              metadata = EXCLUDED.metadata,
              updated_at = now()
            "#,
        )
        .bind(cell.process_id)
        .bind(&cell.score_version)
        .bind(&cell.cell_key)
        .bind(&cell.dimensions)
        .bind(cell.horizon_secs)
        .bind(cell.lookback_days)
        .bind(cell.sample_count)
        .bind(cell.winning_count)
        .bind(cell.losing_count)
        .bind(cell.observed_volume_usd)
        .bind(cell.realized_pnl_usd)
        .bind(cell.mean_price_delta)
        .bind(cell.mean_return)
        .bind(cell.win_rate)
        .bind(cell.expectancy)
        .bind(cell.confidence)
        .bind(cell.sample_start)
        .bind(cell.sample_end)
        .bind(&cell.metadata)
        .execute(&self.pool)
        .await
        .context("failed to upsert expectancy flow cell")?;
        Ok(())
    }

    pub async fn upsert_expectancy_flow_wallet_cell(
        &self,
        cell: &ExpectancyFlowWalletCell,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.expectancy_flow_wallet_cells (
              process_id, score_version, proxy_wallet, cell_key, dimensions,
              horizon_secs, lookback_days, sample_count, winning_count, losing_count,
              observed_volume_usd, realized_pnl_usd, mean_price_delta, mean_return,
              win_rate, expectancy, confidence, sample_start, sample_end, metadata, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,now())
            ON CONFLICT (process_id, score_version, proxy_wallet, cell_key) DO UPDATE SET
              dimensions = EXCLUDED.dimensions,
              horizon_secs = EXCLUDED.horizon_secs,
              lookback_days = EXCLUDED.lookback_days,
              sample_count = EXCLUDED.sample_count,
              winning_count = EXCLUDED.winning_count,
              losing_count = EXCLUDED.losing_count,
              observed_volume_usd = EXCLUDED.observed_volume_usd,
              realized_pnl_usd = EXCLUDED.realized_pnl_usd,
              mean_price_delta = EXCLUDED.mean_price_delta,
              mean_return = EXCLUDED.mean_return,
              win_rate = EXCLUDED.win_rate,
              expectancy = EXCLUDED.expectancy,
              confidence = EXCLUDED.confidence,
              sample_start = EXCLUDED.sample_start,
              sample_end = EXCLUDED.sample_end,
              metadata = EXCLUDED.metadata,
              updated_at = now()
            "#,
        )
        .bind(cell.process_id)
        .bind(&cell.score_version)
        .bind(&cell.proxy_wallet)
        .bind(&cell.cell_key)
        .bind(&cell.dimensions)
        .bind(cell.horizon_secs)
        .bind(cell.lookback_days)
        .bind(cell.sample_count)
        .bind(cell.winning_count)
        .bind(cell.losing_count)
        .bind(cell.observed_volume_usd)
        .bind(cell.realized_pnl_usd)
        .bind(cell.mean_price_delta)
        .bind(cell.mean_return)
        .bind(cell.win_rate)
        .bind(cell.expectancy)
        .bind(cell.confidence)
        .bind(cell.sample_start)
        .bind(cell.sample_end)
        .bind(&cell.metadata)
        .execute(&self.pool)
        .await
        .context("failed to upsert expectancy flow wallet cell")?;
        Ok(())
    }

    pub async fn fetch_expectancy_input(
        &self,
        process_id: Uuid,
        score_version: &str,
        proxy_wallet: &str,
        segment_key: &str,
        side: &str,
        price: Decimal,
        horizon_secs: i64,
        min_trade_usd: Decimal,
    ) -> Result<CopyTradeExpectancyInput> {
        let side = side.trim().to_ascii_lowercase();
        let price_bucket = expectancy_price_bucket(price);
        let bucket_bounds = expectancy_bucket_bounds(price_bucket);
        let cell = self
            .fetch_latest_expectancy_flow_cell_by_dimensions(
                process_id,
                score_version,
                segment_key,
                &side,
                bucket_bounds.0,
                horizon_secs,
                min_trade_usd,
            )
            .await?;
        let wallet_cell = self
            .fetch_latest_expectancy_flow_wallet_cell_by_dimensions(
                process_id,
                score_version,
                proxy_wallet,
                segment_key,
                &side,
                bucket_bounds.0,
                horizon_secs,
                min_trade_usd,
            )
            .await?;
        Ok(CopyTradeExpectancyInput { cell, wallet_cell })
    }

    pub async fn fetch_latest_expectancy_flow_cell_by_dimensions(
        &self,
        process_id: Uuid,
        score_version: &str,
        segment_key: &str,
        side: &str,
        price_bucket_min: Decimal,
        horizon_secs: i64,
        _min_trade_usd: Decimal,
    ) -> Result<Option<ExpectancyFlowCell>> {
        let row = sqlx::query_as::<_, ExpectancyFlowCellRow>(
            r#"
            SELECT process_id, score_version, cell_key, dimensions, horizon_secs, lookback_days,
              sample_count, winning_count, losing_count, observed_volume_usd, realized_pnl_usd,
              mean_price_delta, mean_return, win_rate, expectancy, confidence,
              sample_start, sample_end, metadata, updated_at
            FROM polymarket.expectancy_flow_cells
            WHERE process_id = $1
              AND score_version = $2
              AND dimensions #>> '{segment_key}' = $3
              AND dimensions #>> '{side}' = $4
              AND (dimensions #>> '{price_bucket_min}')::numeric = $5
              AND horizon_secs = $6
            ORDER BY updated_at DESC, confidence DESC, expectancy DESC
            LIMIT 1
            "#,
        )
        .bind(process_id)
        .bind(score_version)
        .bind(segment_key)
        .bind(side)
        .bind(price_bucket_min)
        .bind(horizon_secs)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch latest expectancy flow cell by dimensions")?;
        Ok(row.map(Into::into))
    }

    pub async fn fetch_latest_expectancy_flow_wallet_cell_by_dimensions(
        &self,
        process_id: Uuid,
        score_version: &str,
        proxy_wallet: &str,
        segment_key: &str,
        side: &str,
        price_bucket_min: Decimal,
        horizon_secs: i64,
        _min_trade_usd: Decimal,
    ) -> Result<Option<ExpectancyFlowWalletCell>> {
        let row = sqlx::query_as::<_, ExpectancyFlowWalletCellRow>(
            r#"
            SELECT process_id, score_version, proxy_wallet, cell_key, dimensions,
              horizon_secs, lookback_days, sample_count, winning_count, losing_count,
              observed_volume_usd, realized_pnl_usd, mean_price_delta, mean_return,
              win_rate, expectancy, confidence, sample_start, sample_end, metadata, updated_at
            FROM polymarket.expectancy_flow_wallet_cells
            WHERE process_id = $1
              AND score_version = $2
              AND lower(proxy_wallet) = lower($3)
              AND dimensions #>> '{segment_key}' = $4
              AND dimensions #>> '{side}' = $5
              AND (dimensions #>> '{price_bucket_min}')::numeric = $6
              AND horizon_secs = $7
            ORDER BY updated_at DESC, confidence DESC, expectancy DESC
            LIMIT 1
            "#,
        )
        .bind(process_id)
        .bind(score_version)
        .bind(proxy_wallet)
        .bind(segment_key)
        .bind(side)
        .bind(price_bucket_min)
        .bind(horizon_secs)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch latest expectancy flow wallet cell by dimensions")?;
        Ok(row.map(Into::into))
    }

    pub async fn fetch_expectancy_flow_cell(
        &self,
        process_id: Uuid,
        score_version: &str,
        cell_key: &str,
    ) -> Result<Option<ExpectancyFlowCell>> {
        let row = sqlx::query_as::<_, ExpectancyFlowCellRow>(
            r#"
            SELECT process_id, score_version, cell_key, dimensions, horizon_secs, lookback_days,
              sample_count, winning_count, losing_count, observed_volume_usd, realized_pnl_usd,
              mean_price_delta, mean_return, win_rate, expectancy, confidence,
              sample_start, sample_end, metadata, updated_at
            FROM polymarket.expectancy_flow_cells
            WHERE process_id = $1 AND score_version = $2 AND cell_key = $3
            LIMIT 1
            "#,
        )
        .bind(process_id)
        .bind(score_version)
        .bind(cell_key)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch expectancy flow cell")?;
        Ok(row.map(Into::into))
    }

    pub async fn list_expectancy_flow_cells(
        &self,
        process_id: Uuid,
        score_version: &str,
        limit: i64,
    ) -> Result<Vec<ExpectancyFlowCell>> {
        let rows = sqlx::query_as::<_, ExpectancyFlowCellRow>(
            r#"
            SELECT process_id, score_version, cell_key, dimensions, horizon_secs, lookback_days,
              sample_count, winning_count, losing_count, observed_volume_usd, realized_pnl_usd,
              mean_price_delta, mean_return, win_rate, expectancy, confidence,
              sample_start, sample_end, metadata, updated_at
            FROM polymarket.expectancy_flow_cells
            WHERE process_id = $1 AND score_version = $2
            ORDER BY confidence DESC, expectancy DESC, sample_count DESC, cell_key
            LIMIT $3
            "#,
        )
        .bind(process_id)
        .bind(score_version)
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await
        .context("failed to list expectancy flow cells")?;
        Ok(rows.into_iter().map(Into::into).collect())
    }

    pub async fn fetch_expectancy_flow_wallet_cell(
        &self,
        process_id: Uuid,
        score_version: &str,
        proxy_wallet: &str,
        cell_key: &str,
    ) -> Result<Option<ExpectancyFlowWalletCell>> {
        let row = sqlx::query_as::<_, ExpectancyFlowWalletCellRow>(
            r#"
            SELECT process_id, score_version, proxy_wallet, cell_key, dimensions,
              horizon_secs, lookback_days, sample_count, winning_count, losing_count,
              observed_volume_usd, realized_pnl_usd, mean_price_delta, mean_return,
              win_rate, expectancy, confidence, sample_start, sample_end, metadata, updated_at
            FROM polymarket.expectancy_flow_wallet_cells
            WHERE process_id = $1
              AND score_version = $2
              AND lower(proxy_wallet) = lower($3)
              AND cell_key = $4
            LIMIT 1
            "#,
        )
        .bind(process_id)
        .bind(score_version)
        .bind(proxy_wallet)
        .bind(cell_key)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch expectancy flow wallet cell")?;
        Ok(row.map(Into::into))
    }

    pub async fn list_expectancy_flow_wallet_cells(
        &self,
        process_id: Uuid,
        score_version: &str,
        proxy_wallet: &str,
        limit: i64,
    ) -> Result<Vec<ExpectancyFlowWalletCell>> {
        let rows = sqlx::query_as::<_, ExpectancyFlowWalletCellRow>(
            r#"
            SELECT process_id, score_version, proxy_wallet, cell_key, dimensions,
              horizon_secs, lookback_days, sample_count, winning_count, losing_count,
              observed_volume_usd, realized_pnl_usd, mean_price_delta, mean_return,
              win_rate, expectancy, confidence, sample_start, sample_end, metadata, updated_at
            FROM polymarket.expectancy_flow_wallet_cells
            WHERE process_id = $1
              AND score_version = $2
              AND lower(proxy_wallet) = lower($3)
            ORDER BY confidence DESC, expectancy DESC, sample_count DESC, cell_key
            LIMIT $4
            "#,
        )
        .bind(process_id)
        .bind(score_version)
        .bind(proxy_wallet)
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await
        .context("failed to list expectancy flow wallet cells")?;
        Ok(rows.into_iter().map(Into::into).collect())
    }

    pub async fn recompute_expectancy_flow_cells(
        &self,
        process_id: Uuid,
        config: &EffectiveExpectancyFlowProcessConfig,
    ) -> Result<ExpectancyFlowRecomputeReport> {
        let cells_recomputed = self
            .recompute_expectancy_flow_global_cells(process_id, config)
            .await?;
        let wallet_cells_recomputed = if config.include_wallet_cells {
            self.recompute_expectancy_flow_wallet_cells(process_id, config)
                .await?
        } else {
            sqlx::query(
                "DELETE FROM polymarket.expectancy_flow_wallet_cells WHERE process_id = $1 AND score_version = $2",
            )
            .bind(process_id)
            .bind(&config.score_version)
            .execute(&self.pool)
            .await
            .context("failed to clear disabled expectancy flow wallet cells")?
            .rows_affected()
        };
        Ok(ExpectancyFlowRecomputeReport {
            process_id,
            score_version: config.score_version.clone(),
            cells_recomputed,
            wallet_cells_recomputed,
        })
    }

    async fn recompute_expectancy_flow_global_cells(
        &self,
        process_id: Uuid,
        config: &EffectiveExpectancyFlowProcessConfig,
    ) -> Result<u64> {
        let rows = sqlx::query_scalar::<_, i64>(
            r#"
            WITH deleted AS (
              DELETE FROM polymarket.expectancy_flow_cells
              WHERE process_id = $1 AND score_version = $2
            ),
            source_trades AS MATERIALIZED (
              SELECT
                wt.trade_id,
                wt.proxy_wallet AS proxy_wallet,
                wt.asset,
                wt.side,
                wt.price,
                wt.size,
                wt.cash_value,
                wt.timestamp_utc,
                CASE WHEN $8::boolean THEN COALESCE(wt.market_id, wt.condition_id, wt.asset) ELSE 'all' END AS market_key,
                CASE WHEN $9::boolean THEN COALESCE(NULLIF(wt.taxonomy_segment, ''), 'unknown') ELSE 'all' END AS segment_key,
                (floor((wt.price * 10000) / GREATEST($6::numeric, 1)) * GREATEST($6::numeric, 1))::integer AS price_bucket_bps
              FROM polymarket.wallet_trades wt
              WHERE wt.timestamp_utc >= now() - ($4::integer * interval '1 day')
                AND wt.price > 0
                AND wt.cash_value > 0
                AND wt.size > 0
                AND (
                  $9::boolean = false
                  OR (
                    wt.taxonomy_version = $12
                    AND wt.taxonomy_source = 'gamma'
                    AND wt.taxonomy_segment IS NOT NULL
                    AND wt.taxonomy_segment <> ''
                  )
                )
            ),
            observations AS MATERIALIZED (
              SELECT
                st.*,
                mark.timestamp_utc AS mark_timestamp,
                mark.mark_price
              FROM source_trades st
              JOIN LATERAL (
                SELECT
                  ob.timestamp_utc,
                  CASE
                    WHEN ob.best_bid IS NOT NULL AND ob.best_ask IS NOT NULL THEN (ob.best_bid + ob.best_ask) / 2
                    WHEN ob.best_bid IS NOT NULL THEN ob.best_bid
                    ELSE ob.best_ask
                  END AS mark_price
                FROM polymarket.orderbook_snapshots ob
                WHERE ob.token_id = st.asset
                  AND ob.timestamp_utc >= st.timestamp_utc + ($5::bigint * interval '1 second')
                  AND ob.timestamp_utc <= st.timestamp_utc + (($5::bigint + $7::bigint) * interval '1 second')
                  AND (ob.best_bid IS NOT NULL OR ob.best_ask IS NOT NULL)
                ORDER BY ob.timestamp_utc ASC
                LIMIT 1
              ) mark ON true
            ),
            prepared AS MATERIALIZED (
              SELECT
                *,
                CASE
                  WHEN upper(side) = 'SELL' THEN price - mark_price
                  ELSE mark_price - price
                END AS price_delta,
                CASE
                  WHEN upper(side) = 'SELL' THEN (price - mark_price) * size
                  ELSE (mark_price - price) * size
                END AS pnl_usd,
                CASE
                  WHEN cash_value > 0 THEN
                    CASE
                      WHEN upper(side) = 'SELL' THEN ((price - mark_price) * size) / cash_value
                      ELSE ((mark_price - price) * size) / cash_value
                    END
                  ELSE 0
                END AS return_on_notional
              FROM observations
              WHERE mark_price >= 0 AND mark_price <= 1
            ),
            grouped AS MATERIALIZED (
              SELECT
                lower(COALESCE(NULLIF(side, ''), 'unknown')) AS side_key,
                market_key,
                segment_key,
                price_bucket_bps,
                count(*)::integer AS sample_count,
                count(*) FILTER (WHERE pnl_usd > 0)::integer AS winning_count,
                count(*) FILTER (WHERE pnl_usd < 0)::integer AS losing_count,
                COALESCE(sum(cash_value), 0)::numeric AS observed_volume_usd,
                COALESCE(sum(pnl_usd), 0)::numeric AS realized_pnl_usd,
                COALESCE(avg(price_delta), 0)::numeric AS mean_price_delta,
                COALESCE(avg(return_on_notional), 0)::numeric AS mean_return,
                COALESCE(avg(return_on_notional), 0)::numeric AS expectancy,
                min(timestamp_utc) AS sample_start,
                max(timestamp_utc) AS sample_end
              FROM prepared
              GROUP BY lower(COALESCE(NULLIF(side, ''), 'unknown')), market_key, segment_key, price_bucket_bps
              HAVING count(*) >= GREATEST($10::integer, 1)
              ORDER BY expectancy DESC, sample_count DESC
              LIMIT $11
            ),
            inserted AS (
              INSERT INTO polymarket.expectancy_flow_cells (
                process_id, score_version, cell_key, dimensions, horizon_secs, lookback_days,
                sample_count, winning_count, losing_count, observed_volume_usd, realized_pnl_usd,
                mean_price_delta, mean_return, win_rate, expectancy, confidence,
                sample_start, sample_end, metadata, updated_at
              )
              SELECT
                $1,
                $2,
                md5(concat_ws('|', $2, side_key, market_key, segment_key, price_bucket_bps::text)),
                jsonb_build_object(
                  'side', side_key,
                  'market_key', market_key,
                  'segment_key', segment_key,
                  'price_bucket_bps', price_bucket_bps,
                  'price_bucket_min', price_bucket_bps::numeric / 10000,
                  'price_bucket_max', (price_bucket_bps + GREATEST($6::integer, 1))::numeric / 10000
                ),
                $5,
                $4,
                sample_count,
                winning_count,
                losing_count,
                observed_volume_usd,
                realized_pnl_usd,
                mean_price_delta,
                mean_return,
                CASE WHEN sample_count > 0 THEN winning_count::numeric / sample_count ELSE 0 END,
                expectancy,
                LEAST(1::numeric, sample_count::numeric / GREATEST(($10::numeric * 3), 1)),
                sample_start,
                sample_end,
                jsonb_build_object(
                  'source', 'wallet_trades_orderbook_snapshots',
                  'enabled', $3,
                  'max_snapshot_lag_secs', $7,
                  'include_market_dimension', $8,
                  'include_taxonomy_segment', $9,
                  'recomputed_at', now()
                ),
                now()
              FROM grouped
              RETURNING 1
            )
            SELECT count(*)::bigint FROM inserted
            "#,
        )
        .bind(process_id)
        .bind(&config.score_version)
        .bind(config.enabled)
        .bind(config.recompute_lookback_days.max(0))
        .bind(config.horizon_secs.max(1))
        .bind(config.price_bucket_bps.max(1))
        .bind(config.max_snapshot_lag_secs.max(1))
        .bind(config.include_market_dimension)
        .bind(config.include_taxonomy_segment)
        .bind(config.min_trades_per_cell.max(1))
        .bind(config.max_cells.max(1))
        .bind(GAMMA_TAXONOMY_VERSION)
        .fetch_one(&self.pool)
        .await
        .context("failed to recompute expectancy flow cells")?;
        Ok(rows.max(0) as u64)
    }

    async fn recompute_expectancy_flow_wallet_cells(
        &self,
        process_id: Uuid,
        config: &EffectiveExpectancyFlowProcessConfig,
    ) -> Result<u64> {
        let rows = sqlx::query_scalar::<_, i64>(
            r#"
            WITH deleted AS (
              DELETE FROM polymarket.expectancy_flow_wallet_cells
              WHERE process_id = $1 AND score_version = $2
            ),
            source_trades AS MATERIALIZED (
              SELECT
                wt.trade_id,
                wt.proxy_wallet AS proxy_wallet,
                wt.asset,
                wt.side,
                wt.price,
                wt.size,
                wt.cash_value,
                wt.timestamp_utc,
                CASE WHEN $8::boolean THEN COALESCE(wt.market_id, wt.condition_id, wt.asset) ELSE 'all' END AS market_key,
                CASE WHEN $9::boolean THEN COALESCE(NULLIF(wt.taxonomy_segment, ''), 'unknown') ELSE 'all' END AS segment_key,
                (floor((wt.price * 10000) / GREATEST($6::numeric, 1)) * GREATEST($6::numeric, 1))::integer AS price_bucket_bps
              FROM polymarket.wallet_trades wt
              WHERE wt.timestamp_utc >= now() - ($4::integer * interval '1 day')
                AND wt.price > 0
                AND wt.cash_value > 0
                AND wt.size > 0
                AND (
                  $9::boolean = false
                  OR (
                    wt.taxonomy_version = $12
                    AND wt.taxonomy_source = 'gamma'
                    AND wt.taxonomy_segment IS NOT NULL
                    AND wt.taxonomy_segment <> ''
                  )
                )
            ),
            observations AS MATERIALIZED (
              SELECT
                st.*,
                mark.timestamp_utc AS mark_timestamp,
                mark.mark_price
              FROM source_trades st
              JOIN LATERAL (
                SELECT
                  ob.timestamp_utc,
                  CASE
                    WHEN ob.best_bid IS NOT NULL AND ob.best_ask IS NOT NULL THEN (ob.best_bid + ob.best_ask) / 2
                    WHEN ob.best_bid IS NOT NULL THEN ob.best_bid
                    ELSE ob.best_ask
                  END AS mark_price
                FROM polymarket.orderbook_snapshots ob
                WHERE ob.token_id = st.asset
                  AND ob.timestamp_utc >= st.timestamp_utc + ($5::bigint * interval '1 second')
                  AND ob.timestamp_utc <= st.timestamp_utc + (($5::bigint + $7::bigint) * interval '1 second')
                  AND (ob.best_bid IS NOT NULL OR ob.best_ask IS NOT NULL)
                ORDER BY ob.timestamp_utc ASC
                LIMIT 1
              ) mark ON true
            ),
            prepared AS MATERIALIZED (
              SELECT
                *,
                CASE
                  WHEN upper(side) = 'SELL' THEN price - mark_price
                  ELSE mark_price - price
                END AS price_delta,
                CASE
                  WHEN upper(side) = 'SELL' THEN (price - mark_price) * size
                  ELSE (mark_price - price) * size
                END AS pnl_usd,
                CASE
                  WHEN cash_value > 0 THEN
                    CASE
                      WHEN upper(side) = 'SELL' THEN ((price - mark_price) * size) / cash_value
                      ELSE ((mark_price - price) * size) / cash_value
                    END
                  ELSE 0
                END AS return_on_notional
              FROM observations
              WHERE mark_price >= 0 AND mark_price <= 1
            ),
            grouped AS MATERIALIZED (
              SELECT
                proxy_wallet,
                lower(COALESCE(NULLIF(side, ''), 'unknown')) AS side_key,
                market_key,
                segment_key,
                price_bucket_bps,
                count(*)::integer AS sample_count,
                count(*) FILTER (WHERE pnl_usd > 0)::integer AS winning_count,
                count(*) FILTER (WHERE pnl_usd < 0)::integer AS losing_count,
                COALESCE(sum(cash_value), 0)::numeric AS observed_volume_usd,
                COALESCE(sum(pnl_usd), 0)::numeric AS realized_pnl_usd,
                COALESCE(avg(price_delta), 0)::numeric AS mean_price_delta,
                COALESCE(avg(return_on_notional), 0)::numeric AS mean_return,
                COALESCE(avg(return_on_notional), 0)::numeric AS expectancy,
                min(timestamp_utc) AS sample_start,
                max(timestamp_utc) AS sample_end
              FROM prepared
              GROUP BY proxy_wallet, lower(COALESCE(NULLIF(side, ''), 'unknown')), market_key, segment_key, price_bucket_bps
              HAVING count(*) >= GREATEST($10::integer, 1)
              ORDER BY expectancy DESC, sample_count DESC
              LIMIT $11
            ),
            inserted AS (
              INSERT INTO polymarket.expectancy_flow_wallet_cells (
                process_id, score_version, proxy_wallet, cell_key, dimensions, horizon_secs,
                lookback_days, sample_count, winning_count, losing_count, observed_volume_usd,
                realized_pnl_usd, mean_price_delta, mean_return, win_rate, expectancy,
                confidence, sample_start, sample_end, metadata, updated_at
              )
              SELECT
                $1,
                $2,
                proxy_wallet,
                md5(concat_ws('|', $2, proxy_wallet, side_key, market_key, segment_key, price_bucket_bps::text)),
                jsonb_build_object(
                  'side', side_key,
                  'market_key', market_key,
                  'segment_key', segment_key,
                  'price_bucket_bps', price_bucket_bps,
                  'price_bucket_min', price_bucket_bps::numeric / 10000,
                  'price_bucket_max', (price_bucket_bps + GREATEST($6::integer, 1))::numeric / 10000
                ),
                $5,
                $4,
                sample_count,
                winning_count,
                losing_count,
                observed_volume_usd,
                realized_pnl_usd,
                mean_price_delta,
                mean_return,
                CASE WHEN sample_count > 0 THEN winning_count::numeric / sample_count ELSE 0 END,
                expectancy,
                LEAST(1::numeric, sample_count::numeric / GREATEST(($10::numeric * 3), 1)),
                sample_start,
                sample_end,
                jsonb_build_object(
                  'source', 'wallet_trades_orderbook_snapshots',
                  'enabled', $3,
                  'max_snapshot_lag_secs', $7,
                  'include_market_dimension', $8,
                  'include_taxonomy_segment', $9,
                  'recomputed_at', now()
                ),
                now()
              FROM grouped
              RETURNING 1
            )
            SELECT count(*)::bigint FROM inserted
            "#,
        )
        .bind(process_id)
        .bind(&config.score_version)
        .bind(config.enabled)
        .bind(config.recompute_lookback_days.max(0))
        .bind(config.horizon_secs.max(1))
        .bind(config.price_bucket_bps.max(1))
        .bind(config.max_snapshot_lag_secs.max(1))
        .bind(config.include_market_dimension)
        .bind(config.include_taxonomy_segment)
        .bind(config.min_trades_per_cell.max(1))
        .bind(config.max_cells.max(1))
        .bind(GAMMA_TAXONOMY_VERSION)
        .fetch_one(&self.pool)
        .await
        .context("failed to recompute expectancy flow wallet cells")?;
        Ok(rows.max(0) as u64)
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

    pub async fn insert_wallet_score_calibration_snapshot(
        &self,
        snapshot: &WalletScoreCalibrationSnapshot,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.wallet_score_calibration_snapshots (
              snapshot_id, timestamp_utc, score_version, calibration_version,
              sample_start, sample_end, wallet_count, trade_count,
              feature_weights, thresholds, metrics, metadata
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (snapshot_id, timestamp_utc) DO UPDATE SET
              score_version = EXCLUDED.score_version,
              calibration_version = EXCLUDED.calibration_version,
              sample_start = EXCLUDED.sample_start,
              sample_end = EXCLUDED.sample_end,
              wallet_count = EXCLUDED.wallet_count,
              trade_count = EXCLUDED.trade_count,
              feature_weights = EXCLUDED.feature_weights,
              thresholds = EXCLUDED.thresholds,
              metrics = EXCLUDED.metrics,
              metadata = EXCLUDED.metadata
            "#,
        )
        .bind(snapshot.snapshot_id)
        .bind(snapshot.timestamp_utc)
        .bind(&snapshot.score_version)
        .bind(&snapshot.calibration_version)
        .bind(snapshot.sample_start)
        .bind(snapshot.sample_end)
        .bind(snapshot.wallet_count)
        .bind(snapshot.trade_count)
        .bind(&snapshot.feature_weights)
        .bind(&snapshot.thresholds)
        .bind(&snapshot.metrics)
        .bind(&snapshot.metadata)
        .execute(&self.pool)
        .await
        .context("failed to insert wallet score calibration snapshot")?;
        Ok(())
    }

    pub async fn insert_copy_trade_signal(&self, signal: &CopyTradeSignal) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.copy_trade_signals (
              signal_id, process_id, timestamp_utc, proxy_wallet, source_trade_id,
              market_id, token_id, side, whale_price, observed_price,
              copy_size_usd, reason, status, metadata
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
            ON CONFLICT (signal_id, timestamp_utc) DO UPDATE SET
              process_id = EXCLUDED.process_id,
              proxy_wallet = EXCLUDED.proxy_wallet,
              source_trade_id = EXCLUDED.source_trade_id,
              market_id = EXCLUDED.market_id,
              token_id = EXCLUDED.token_id,
              side = EXCLUDED.side,
              whale_price = EXCLUDED.whale_price,
              observed_price = EXCLUDED.observed_price,
              copy_size_usd = EXCLUDED.copy_size_usd,
              reason = EXCLUDED.reason,
              status = EXCLUDED.status,
              metadata = EXCLUDED.metadata
            "#,
        )
        .bind(signal.signal_id)
        .bind(signal.process_id)
        .bind(signal.timestamp_utc)
        .bind(&signal.proxy_wallet)
        .bind(signal.source_trade_id)
        .bind(&signal.market_id)
        .bind(&signal.token_id)
        .bind(&signal.side)
        .bind(signal.whale_price)
        .bind(signal.observed_price)
        .bind(signal.copy_size_usd)
        .bind(&signal.reason)
        .bind(&signal.status)
        .bind(&signal.metadata)
        .execute(&self.pool)
        .await
        .context("failed to insert copy-trade signal")?;
        Ok(())
    }

    pub async fn update_copy_trade_signal_status(
        &self,
        signal_id: Uuid,
        timestamp_utc: DateTime<Utc>,
        status: &str,
        metadata: serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE polymarket.copy_trade_signals
            SET status = $3, metadata = metadata || $4
            WHERE signal_id = $1 AND timestamp_utc = $2
            "#,
        )
        .bind(signal_id)
        .bind(timestamp_utc)
        .bind(status)
        .bind(metadata)
        .execute(&self.pool)
        .await
        .context("failed to update copy-trade signal status")?;
        Ok(())
    }

    pub async fn update_copy_trade_signal_status_by_id(
        &self,
        signal_id: Uuid,
        status: &str,
        metadata: serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE polymarket.copy_trade_signals
            SET status = $2, metadata = metadata || $3
            WHERE signal_id = $1
            "#,
        )
        .bind(signal_id)
        .bind(status)
        .bind(metadata)
        .execute(&self.pool)
        .await
        .context("failed to update copy-trade signal status by id")?;
        Ok(())
    }

    pub async fn upsert_copy_trade_backtest_run(&self, run: &CopyTradeBacktestRun) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.copy_trade_backtest_runs (
              backtest_id, job_id, status, score_version, strategy_name,
              range_start, range_end, config, started_at, completed_at, error, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,now())
            ON CONFLICT (backtest_id) DO UPDATE SET
              job_id = EXCLUDED.job_id,
              status = EXCLUDED.status,
              score_version = EXCLUDED.score_version,
              strategy_name = EXCLUDED.strategy_name,
              range_start = EXCLUDED.range_start,
              range_end = EXCLUDED.range_end,
              config = EXCLUDED.config,
              started_at = EXCLUDED.started_at,
              completed_at = EXCLUDED.completed_at,
              error = EXCLUDED.error,
              updated_at = now()
            "#,
        )
        .bind(run.backtest_id)
        .bind(run.job_id)
        .bind(&run.status)
        .bind(&run.score_version)
        .bind(&run.strategy_name)
        .bind(run.range_start)
        .bind(run.range_end)
        .bind(&run.config)
        .bind(run.started_at)
        .bind(run.completed_at)
        .bind(&run.error)
        .execute(&self.pool)
        .await
        .context("failed to upsert copy-trade backtest run")?;
        Ok(())
    }

    pub async fn create_backtest_run(
        &self,
        range_start: DateTime<Utc>,
        range_end: DateTime<Utc>,
        warmup_start: DateTime<Utc>,
        lookback_days: i32,
        warmup_days: i32,
        source_process_ids: &[Uuid],
        request: serde_json::Value,
    ) -> Result<BacktestRun> {
        let row = sqlx::query_as::<_, BacktestRunRow>(
            r#"
            INSERT INTO polymarket.backtest_runs (
              backtest_run_id, status, range_start, range_end, warmup_start,
              lookback_days, warmup_days, source_process_ids, request, created_at, updated_at
            )
            VALUES (gen_random_uuid(), 'queued', $1, $2, $3, $4, $5, $6, $7, now(), now())
            RETURNING backtest_run_id, status, range_start, range_end, warmup_start,
              lookback_days, warmup_days, source_process_ids, backtest_process_ids,
              request, summary, error, started_at, completed_at, created_at, updated_at
            "#,
        )
        .bind(range_start)
        .bind(range_end)
        .bind(warmup_start)
        .bind(lookback_days)
        .bind(warmup_days)
        .bind(source_process_ids)
        .bind(request)
        .fetch_one(&self.pool)
        .await
        .context("failed to create backtest run")?;
        Ok(row.into())
    }

    pub async fn add_backtest_process_to_run(
        &self,
        backtest_run_id: Uuid,
        process_id: Uuid,
    ) -> Result<BacktestRun> {
        let row = sqlx::query_as::<_, BacktestRunRow>(
            r#"
            UPDATE polymarket.backtest_runs
            SET backtest_process_ids = array_append(backtest_process_ids, $2),
                updated_at = now()
            WHERE backtest_run_id = $1
              AND NOT ($2 = ANY(backtest_process_ids))
            RETURNING backtest_run_id, status, range_start, range_end, warmup_start,
              lookback_days, warmup_days, source_process_ids, backtest_process_ids,
              request, summary, error, started_at, completed_at, created_at, updated_at
            "#,
        )
        .bind(backtest_run_id)
        .bind(process_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to append backtest process id")?;
        if let Some(row) = row {
            return Ok(row.into());
        }
        self.get_backtest_run(backtest_run_id)
            .await?
            .context("backtest run not found after process append")
    }

    pub async fn get_backtest_run(&self, backtest_run_id: Uuid) -> Result<Option<BacktestRun>> {
        let row = sqlx::query_as::<_, BacktestRunRow>(
            r#"
            SELECT backtest_run_id, status, range_start, range_end, warmup_start,
              lookback_days, warmup_days, source_process_ids, backtest_process_ids,
              request, summary, error, started_at, completed_at, created_at, updated_at
            FROM polymarket.backtest_runs
            WHERE backtest_run_id = $1
            "#,
        )
        .bind(backtest_run_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch backtest run")?;
        Ok(row.map(Into::into))
    }

    pub async fn list_backtest_runs(&self, limit: i64) -> Result<Vec<BacktestRun>> {
        let rows = sqlx::query_as::<_, BacktestRunRow>(
            r#"
            SELECT backtest_run_id, status, range_start, range_end, warmup_start,
              lookback_days, warmup_days, source_process_ids, backtest_process_ids,
              request, summary, error, started_at, completed_at, created_at, updated_at
            FROM polymarket.backtest_runs
            ORDER BY created_at DESC
            LIMIT $1
            "#,
        )
        .bind(limit.max(1))
        .fetch_all(&self.pool)
        .await
        .context("failed to list backtest runs")?;
        Ok(rows.into_iter().map(Into::into).collect())
    }

    pub async fn mark_backtest_run_status(
        &self,
        backtest_run_id: Uuid,
        status: &str,
        summary: serde_json::Value,
        error: Option<&str>,
    ) -> Result<Option<BacktestRun>> {
        let row = sqlx::query_as::<_, BacktestRunRow>(
            r#"
            UPDATE polymarket.backtest_runs
            SET status = $2,
                summary = summary || $3,
                error = $4,
                started_at = CASE WHEN $2 = 'running' THEN COALESCE(started_at, now()) ELSE started_at END,
                completed_at = CASE WHEN $2 IN ('completed', 'failed', 'cancelled') THEN now() ELSE completed_at END,
                updated_at = now()
            WHERE backtest_run_id = $1
            RETURNING backtest_run_id, status, range_start, range_end, warmup_start,
              lookback_days, warmup_days, source_process_ids, backtest_process_ids,
              request, summary, error, started_at, completed_at, created_at, updated_at
            "#,
        )
        .bind(backtest_run_id)
        .bind(status)
        .bind(summary)
        .bind(error)
        .fetch_optional(&self.pool)
        .await
        .context("failed to mark backtest run status")?;
        Ok(row.map(Into::into))
    }

    pub async fn complete_copy_trade_backtest_run(
        &self,
        backtest_id: Uuid,
        status: &str,
        completed_at: DateTime<Utc>,
        error: Option<String>,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE polymarket.copy_trade_backtest_runs
            SET status = $2, completed_at = $3, error = $4, updated_at = now()
            WHERE backtest_id = $1
            "#,
        )
        .bind(backtest_id)
        .bind(status)
        .bind(completed_at)
        .bind(error)
        .execute(&self.pool)
        .await
        .context("failed to complete copy-trade backtest run")?;
        Ok(())
    }

    pub async fn insert_copy_trade_backtest_result(
        &self,
        result: &CopyTradeBacktestResult,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.copy_trade_backtest_results (
              result_id, backtest_id, timestamp_utc, wallet_count, signal_count,
              trade_count, gross_pnl_usd, net_pnl_usd, roi, max_drawdown,
              win_rate, result_summary
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (result_id, timestamp_utc) DO UPDATE SET
              backtest_id = EXCLUDED.backtest_id,
              wallet_count = EXCLUDED.wallet_count,
              signal_count = EXCLUDED.signal_count,
              trade_count = EXCLUDED.trade_count,
              gross_pnl_usd = EXCLUDED.gross_pnl_usd,
              net_pnl_usd = EXCLUDED.net_pnl_usd,
              roi = EXCLUDED.roi,
              max_drawdown = EXCLUDED.max_drawdown,
              win_rate = EXCLUDED.win_rate,
              result_summary = EXCLUDED.result_summary
            "#,
        )
        .bind(result.result_id)
        .bind(result.backtest_id)
        .bind(result.timestamp_utc)
        .bind(result.wallet_count)
        .bind(result.signal_count)
        .bind(result.trade_count)
        .bind(result.gross_pnl_usd)
        .bind(result.net_pnl_usd)
        .bind(result.roi)
        .bind(result.max_drawdown)
        .bind(result.win_rate)
        .bind(&result.result_summary)
        .execute(&self.pool)
        .await
        .context("failed to insert copy-trade backtest result")?;
        Ok(())
    }

    pub async fn upsert_whale_poll_checkpoint(
        &self,
        checkpoint: &WhalePollCheckpoint,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.whale_poll_checkpoints (
              checkpoint_name, last_polled_at, next_cursor, last_trade_timestamp_utc,
              last_trade_id, pages_seen, trades_seen, state, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,now())
            ON CONFLICT (checkpoint_name) DO UPDATE SET
              last_polled_at = EXCLUDED.last_polled_at,
              next_cursor = EXCLUDED.next_cursor,
              last_trade_timestamp_utc = EXCLUDED.last_trade_timestamp_utc,
              last_trade_id = EXCLUDED.last_trade_id,
              pages_seen = EXCLUDED.pages_seen,
              trades_seen = EXCLUDED.trades_seen,
              state = EXCLUDED.state,
              updated_at = now()
            "#,
        )
        .bind(&checkpoint.checkpoint_name)
        .bind(checkpoint.last_polled_at)
        .bind(&checkpoint.next_cursor)
        .bind(checkpoint.last_trade_timestamp_utc)
        .bind(checkpoint.last_trade_id)
        .bind(checkpoint.pages_seen)
        .bind(checkpoint.trades_seen)
        .bind(&checkpoint.state)
        .execute(&self.pool)
        .await
        .context("failed to upsert whale poll checkpoint")?;
        Ok(())
    }

    pub async fn get_whale_poll_checkpoint(
        &self,
        checkpoint_name: &str,
    ) -> Result<Option<WhalePollCheckpoint>> {
        let row = sqlx::query_as::<_, WhalePollCheckpointRow>(
            r#"
            SELECT checkpoint_name, last_polled_at, next_cursor, last_trade_timestamp_utc,
              last_trade_id, pages_seen, trades_seen, state
            FROM polymarket.whale_poll_checkpoints
            WHERE checkpoint_name = $1
            "#,
        )
        .bind(checkpoint_name)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch whale poll checkpoint")?;
        Ok(row.map(Into::into))
    }
}

fn cap_fills_to_size(fills: &[FillRecord], max_size: Decimal) -> Vec<FillRecord> {
    if max_size <= Decimal::ZERO {
        return Vec::new();
    }
    let mut remaining = max_size;
    let mut capped = Vec::new();
    for fill in fills {
        if remaining <= Decimal::ZERO {
            break;
        }
        let take = fill.size.min(remaining);
        if take <= Decimal::ZERO {
            continue;
        }
        let mut capped_fill = fill.clone();
        if take < fill.size && fill.size > Decimal::ZERO {
            let ratio = take / fill.size;
            capped_fill.size = take;
            capped_fill.fee = fill.fee * ratio;
        }
        remaining -= take;
        capped.push(capped_fill);
    }
    capped
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
struct BackfillJobRow {
    job_id: Uuid,
    job_type: String,
    status: String,
    requested_at: DateTime<Utc>,
    started_at: Option<DateTime<Utc>>,
    completed_at: Option<DateTime<Utc>>,
    cancel_requested_at: Option<DateTime<Utc>>,
    lookback_days: i32,
    min_trade_usd: Decimal,
    request: serde_json::Value,
    summary: serde_json::Value,
    error: Option<String>,
}

impl TryFrom<BackfillJobRow> for BackfillJob {
    type Error = anyhow::Error;

    fn try_from(row: BackfillJobRow) -> Result<Self> {
        let status = match row.status.as_str() {
            "queued" => BackfillJobStatus::Queued,
            "running" => BackfillJobStatus::Running,
            "cancel_requested" => BackfillJobStatus::CancelRequested,
            "completed" => BackfillJobStatus::Completed,
            "failed" => BackfillJobStatus::Failed,
            "cancelled" => BackfillJobStatus::Cancelled,
            other => bail!("unknown backfill job status {other}"),
        };
        Ok(Self {
            job_id: row.job_id,
            job_type: row.job_type,
            status,
            requested_at: row.requested_at,
            started_at: row.started_at,
            completed_at: row.completed_at,
            cancel_requested_at: row.cancel_requested_at,
            lookback_days: row.lookback_days,
            min_trade_usd: row.min_trade_usd,
            request: row.request,
            summary: row.summary,
            error: row.error,
        })
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
struct ExpectancyFlowCellRow {
    process_id: Uuid,
    score_version: String,
    cell_key: String,
    dimensions: serde_json::Value,
    horizon_secs: i64,
    lookback_days: i32,
    sample_count: i32,
    winning_count: i32,
    losing_count: i32,
    observed_volume_usd: Decimal,
    realized_pnl_usd: Decimal,
    mean_price_delta: Decimal,
    mean_return: Decimal,
    win_rate: Decimal,
    expectancy: Decimal,
    confidence: Decimal,
    sample_start: Option<DateTime<Utc>>,
    sample_end: Option<DateTime<Utc>>,
    metadata: serde_json::Value,
    updated_at: DateTime<Utc>,
}

fn expectancy_bucket_bounds(price_bucket: &str) -> (Decimal, Decimal) {
    match price_bucket {
        "<20c" => (Decimal::ZERO, dec!(0.20)),
        "20-40c" => (dec!(0.20), dec!(0.40)),
        "40-60c" => (dec!(0.40), dec!(0.60)),
        "60-80c" => (dec!(0.60), dec!(0.80)),
        _ => (dec!(0.80), dec!(1.00)),
    }
}

impl From<ExpectancyFlowCellRow> for ExpectancyFlowCell {
    fn from(row: ExpectancyFlowCellRow) -> Self {
        Self {
            process_id: row.process_id,
            score_version: row.score_version,
            cell_key: row.cell_key,
            dimensions: row.dimensions,
            horizon_secs: row.horizon_secs,
            lookback_days: row.lookback_days,
            sample_count: row.sample_count,
            winning_count: row.winning_count,
            losing_count: row.losing_count,
            observed_volume_usd: row.observed_volume_usd,
            realized_pnl_usd: row.realized_pnl_usd,
            mean_price_delta: row.mean_price_delta,
            mean_return: row.mean_return,
            win_rate: row.win_rate,
            expectancy: row.expectancy,
            confidence: row.confidence,
            sample_start: row.sample_start,
            sample_end: row.sample_end,
            metadata: row.metadata,
            updated_at: row.updated_at,
        }
    }
}

#[derive(sqlx::FromRow)]
struct ExpectancyFlowWalletCellRow {
    process_id: Uuid,
    score_version: String,
    proxy_wallet: String,
    cell_key: String,
    dimensions: serde_json::Value,
    horizon_secs: i64,
    lookback_days: i32,
    sample_count: i32,
    winning_count: i32,
    losing_count: i32,
    observed_volume_usd: Decimal,
    realized_pnl_usd: Decimal,
    mean_price_delta: Decimal,
    mean_return: Decimal,
    win_rate: Decimal,
    expectancy: Decimal,
    confidence: Decimal,
    sample_start: Option<DateTime<Utc>>,
    sample_end: Option<DateTime<Utc>>,
    metadata: serde_json::Value,
    updated_at: DateTime<Utc>,
}

impl From<ExpectancyFlowWalletCellRow> for ExpectancyFlowWalletCell {
    fn from(row: ExpectancyFlowWalletCellRow) -> Self {
        Self {
            process_id: row.process_id,
            score_version: row.score_version,
            proxy_wallet: row.proxy_wallet,
            cell_key: row.cell_key,
            dimensions: row.dimensions,
            horizon_secs: row.horizon_secs,
            lookback_days: row.lookback_days,
            sample_count: row.sample_count,
            winning_count: row.winning_count,
            losing_count: row.losing_count,
            observed_volume_usd: row.observed_volume_usd,
            realized_pnl_usd: row.realized_pnl_usd,
            mean_price_delta: row.mean_price_delta,
            mean_return: row.mean_return,
            win_rate: row.win_rate,
            expectancy: row.expectancy,
            confidence: row.confidence,
            sample_start: row.sample_start,
            sample_end: row.sample_end,
            metadata: row.metadata,
            updated_at: row.updated_at,
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

#[derive(sqlx::FromRow)]
struct WhalePollCheckpointRow {
    checkpoint_name: String,
    last_polled_at: Option<DateTime<Utc>>,
    next_cursor: Option<String>,
    last_trade_timestamp_utc: Option<DateTime<Utc>>,
    last_trade_id: Option<Uuid>,
    pages_seen: i64,
    trades_seen: i64,
    state: serde_json::Value,
}

impl From<WhalePollCheckpointRow> for WhalePollCheckpoint {
    fn from(row: WhalePollCheckpointRow) -> Self {
        Self {
            checkpoint_name: row.checkpoint_name,
            last_polled_at: row.last_polled_at,
            next_cursor: row.next_cursor,
            last_trade_timestamp_utc: row.last_trade_timestamp_utc,
            last_trade_id: row.last_trade_id,
            pages_seen: row.pages_seen,
            trades_seen: row.trades_seen,
            state: row.state,
        }
    }
}

impl From<BacktestRunRow> for BacktestRun {
    fn from(row: BacktestRunRow) -> Self {
        Self {
            backtest_run_id: row.backtest_run_id,
            status: row.status,
            range_start: row.range_start,
            range_end: row.range_end,
            warmup_start: row.warmup_start,
            lookback_days: row.lookback_days,
            warmup_days: row.warmup_days,
            source_process_ids: row.source_process_ids,
            backtest_process_ids: row.backtest_process_ids,
            request: row.request,
            summary: row.summary,
            error: row.error,
            started_at: row.started_at,
            completed_at: row.completed_at,
            created_at: row.created_at,
            updated_at: row.updated_at,
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

fn push_position_mismatch(
    mismatches: &mut Vec<AccountPositionMismatch>,
    token_id: String,
    db_open_size: Decimal,
    account_size: Decimal,
) {
    let delta_size = account_size - db_open_size;
    if delta_size.abs() <= dec!(0.000000001) {
        return;
    }
    let mismatch_type = if account_size < db_open_size {
        "account_less_than_db"
    } else {
        "account_greater_than_db"
    };
    mismatches.push(AccountPositionMismatch {
        token_id,
        db_open_size,
        account_size,
        delta_size,
        mismatch_type: mismatch_type.to_string(),
    });
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
    use chrono::Utc;
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    use crate::{
        models::{FillRecord, FillSource},
        store::{
            cap_fills_to_size, HEARTBEAT_ACTIVE_TRADING_PROCESS_SQL, MARK_OPEN_TRADE_POSITIONS_SQL,
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
    fn cap_fills_to_size_prorates_the_terminal_fill() {
        let fills = vec![
            FillRecord {
                fill_id: Uuid::new_v4(),
                process_id: Some(Uuid::new_v4()),
                order_id: "sim-1".to_string(),
                token_id: "token-1".to_string(),
                price: dec!(0.50),
                size: dec!(3),
                fee: dec!(0.03),
                source: FillSource::Sim,
                filled_at: Utc::now(),
            },
            FillRecord {
                fill_id: Uuid::new_v4(),
                process_id: Some(Uuid::new_v4()),
                order_id: "sim-1".to_string(),
                token_id: "token-1".to_string(),
                price: dec!(0.60),
                size: dec!(4),
                fee: dec!(0.04),
                source: FillSource::Sim,
                filled_at: Utc::now(),
            },
        ];

        let capped = cap_fills_to_size(&fills, dec!(5));

        assert_eq!(capped.len(), 2);
        assert_eq!(capped[0].size, dec!(3));
        assert_eq!(capped[0].fee, dec!(0.03));
        assert_eq!(capped[1].size, dec!(2));
        assert_eq!(capped[1].fee, dec!(0.02));
    }

    #[test]
    fn mark_open_trade_positions_uses_orderbook_before_wallet_trade() {
        assert!(MARK_OPEN_TRADE_POSITIONS_SQL.contains("latest_orderbook"));
        assert!(MARK_OPEN_TRADE_POSITIONS_SQL.contains("polymarket.orderbook_snapshots"));
        assert!(MARK_OPEN_TRADE_POSITIONS_SQL.contains("latest_wallet_trade"));
        assert!(MARK_OPEN_TRADE_POSITIONS_SQL.contains("polymarket.wallet_trades"));
        assert!(MARK_OPEN_TRADE_POSITIONS_SQL
            .contains("COALESCE(orderbook.mark_price, wallet.mark_price)"));
        assert!(MARK_OPEN_TRADE_POSITIONS_SQL.contains("'clob_mid'"));
        assert!(MARK_OPEN_TRADE_POSITIONS_SQL.contains("'data_api_trade'"));
    }

    #[test]
    fn mark_open_trade_positions_records_source_timestamp_as_position_mark_timestamp() {
        assert!(MARK_OPEN_TRADE_POSITIONS_SQL.contains("source_timestamp"));
        assert!(MARK_OPEN_TRADE_POSITIONS_SQL.contains("latest_mark_timestamp = i.timestamp_utc"));
        assert!(MARK_OPEN_TRADE_POSITIONS_SQL.contains(
            "COALESCE(orderbook.source_timestamp, wallet.source_timestamp) > p.latest_mark_timestamp"
        ));
        assert!(MARK_OPEN_TRADE_POSITIONS_SQL
            .contains("b.timestamp_utc >= now() - interval '15 minutes'"));
    }
}
