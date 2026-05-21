use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::Serialize;
use sqlx::{postgres::PgPoolOptions, FromRow, PgPool};
use uuid::Uuid;

use crate::{
    config::PostgresConfig,
    events::ServiceEvent,
    execution::live::LiveVenueEvent,
    execution::OrderPlanReport,
    models::{
        BackfillJob, BackfillJobStatus, ConversionRequest, ConversionResult,
        CopyTradeBacktestResult, CopyTradeBacktestRun, CopyTradeSignal, FillRecord, Market,
        OrderRecord, OrderRequest, OrderState, OutcomeToken, SignalCandidate, TradingProcess,
        TradingProcessConfig, WalletPerformance, WalletScore, WalletScoreCalibrationSnapshot,
        WhalePollCheckpoint, WhaleTrade,
    },
    orderbook::LocalOrderBook,
};

#[derive(Clone)]
pub struct Store {
    pool: PgPool,
}

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

#[derive(Debug, Clone, FromRow)]
pub struct WhaleLedTradeExitCandidate {
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
              process_id, name, process_type, status, enabled, config, metadata,
              created_at, updated_at, started_at
            )
            VALUES (
              $1, 'default-env-copy-trade', 'copy_trade', 'running', true, $2, $3,
              now(), now(), now()
            )
            ON CONFLICT (process_id) DO UPDATE SET
              updated_at = now()
            RETURNING process_id, name, process_type, status, enabled, config, metadata,
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
        enabled: bool,
        config: TradingProcessConfig,
        metadata: serde_json::Value,
    ) -> Result<TradingProcess> {
        let row = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            INSERT INTO polymarket.trading_processes (
              process_id, name, process_type, status, enabled, config, metadata,
              created_at, updated_at
            )
            VALUES (gen_random_uuid(),$1,$2,'created',$3,$4,$5,now(),now())
            RETURNING process_id, name, process_type, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            "#,
        )
        .bind(name)
        .bind(process_type)
        .bind(enabled)
        .bind(serde_json::to_value(config)?)
        .bind(metadata)
        .fetch_one(&self.pool)
        .await
        .context("failed to create trading process")?;
        trading_process_from_row(row)
    }

    pub async fn list_trading_processes(&self, limit: i64) -> Result<Vec<TradingProcess>> {
        let rows = sqlx::query_as::<_, TradingProcessRow>(
            r#"
            SELECT process_id, name, process_type, status, enabled, config, metadata,
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
            SELECT process_id, name, process_type, status, enabled, config, metadata,
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

    pub async fn update_trading_process(
        &self,
        process_id: Uuid,
        name: Option<&str>,
        process_type: Option<&str>,
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
                enabled = COALESCE($4, enabled),
                status = COALESCE($5, status),
                config = COALESCE($6, config),
                metadata = COALESCE($7, metadata),
                updated_at = now()
            WHERE process_id = $1
            RETURNING process_id, name, process_type, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            "#,
        )
        .bind(process_id)
        .bind(name)
        .bind(process_type)
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
                started_at = COALESCE(started_at, now()),
                stopped_at = NULL,
                last_error = NULL,
                updated_at = now()
            WHERE process_id = $1
            RETURNING process_id, name, process_type, status, enabled, config, metadata,
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
            RETURNING process_id, name, process_type, status, enabled, config, metadata,
              created_at, updated_at, started_at, stopped_at, last_error
            "#,
        )
        .bind(process_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to stop trading process")?;
        row.map(trading_process_from_row).transpose()
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
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
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
              venue_trade_id, event_status, event_hash, raw_payload
            )
            VALUES (gen_random_uuid(),$1,$2,$3,$4,$5,$6,$7,$8)
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
              JOIN polymarket.fills f
                ON f.order_id = o.order_id
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
        let filled_size: Decimal = report.fills.iter().map(|fill| fill.size).sum();
        if filled_size <= Decimal::ZERO {
            return Ok(0);
        }
        let exit_fee: Decimal = report.fills.iter().map(|fill| fill.fee).sum();
        let exit_notional: Decimal = report.fills.iter().map(|fill| fill.price * fill.size).sum();
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
        let fill_payloads: Vec<_> = report
            .fills
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
            "execution_source": report
                .fills
                .first()
                .map(|fill| serialized_name(&fill.source).unwrap_or_else(|_| "unknown".to_string()))
                .unwrap_or_else(|| "unknown".to_string()),
            "fills": fill_payloads
        });

        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin executable exit transaction")?;
        let inserted = sqlx::query(
            r#"
            INSERT INTO polymarket.trade_exits (
              position_id, source_signal_id, timestamp_utc, exit_type, is_synthetic,
              exit_trigger_wallet, exit_source_trade_id, exit_price, exit_size,
              exit_notional, exit_fee, slippage_cost, gross_pnl, net_pnl, roi, metadata
            )
            VALUES (
              $1,$2,$3,$4,false,$5,$6,$7,$8,$9,$10,$11,$12,$13,
              CASE WHEN $14 > 0 THEN $13 / $14 ELSE 0 END,
              $15
            )
            ON CONFLICT (position_id, exit_source_trade_id) DO NOTHING
            "#,
        )
        .bind(candidate.position_id)
        .bind(candidate.source_signal_id)
        .bind(candidate.exit_timestamp)
        .bind(if filled_size < candidate.open_size {
            "whale_reduce"
        } else {
            "whale_exit"
        })
        .bind(candidate.proxy_wallet.as_deref())
        .bind(Option::<Uuid>::None)
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
        let result = sqlx::query(
            r#"
            WITH marks AS (
              SELECT
                p.position_id,
                p.source_signal_id,
                p.side,
                p.entry_price,
                p.open_size,
                p.entry_notional,
                latest.price AS mark_price,
                latest.timestamp_utc AS source_timestamp,
                GREATEST(0, floor(extract(epoch from (now() - latest.timestamp_utc)) * 1000))::bigint AS mark_age_ms
              FROM polymarket.trade_positions p
              JOIN LATERAL (
                SELECT wt.price, wt.timestamp_utc
                FROM polymarket.wallet_trades wt
                WHERE wt.asset = p.token_id
                  AND wt.timestamp_utc >= p.entry_timestamp
                ORDER BY wt.timestamp_utc DESC
                LIMIT 1
              ) latest ON true
              WHERE p.status IN ('open', 'partially_closed')
                AND p.open_size > 0
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
                position_id, source_signal_id, timestamp_utc, mark_price, mark_source,
                mark_age_ms, gross_unrealized_pnl, net_unrealized_pnl, roi, metadata
              )
              SELECT
                position_id,
                source_signal_id,
                now(),
                mark_price,
                'data_api_trade',
                mark_age_ms,
                gross_unrealized_pnl,
                gross_unrealized_pnl,
                CASE WHEN entry_notional > 0 THEN gross_unrealized_pnl / entry_notional ELSE 0 END,
                jsonb_build_object('source_timestamp', source_timestamp)
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
            "#,
        )
        .execute(&self.pool)
        .await
        .context("failed to mark open trade positions")?;
        Ok(result.rows_affected())
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

    pub async fn trade_pnl_summary(&self) -> Result<serde_json::Value> {
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
              'updated_at', now()
            )
            FROM total_stats
            "#,
        )
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
              signal_id, process_id, timestamp_utc, proxy_wallet, wallet_score, source_trade_id,
              market_id, token_id, side, whale_price, observed_price,
              copy_size_usd, reason, status, metadata
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
            ON CONFLICT (signal_id, timestamp_utc) DO UPDATE SET
              process_id = EXCLUDED.process_id,
              proxy_wallet = EXCLUDED.proxy_wallet,
              wallet_score = EXCLUDED.wallet_score,
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
        .bind(signal.wallet_score)
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

fn trading_process_from_row(row: TradingProcessRow) -> Result<TradingProcess> {
    let config =
        serde_json::from_value(row.config.clone()).unwrap_or_else(|_| TradingProcessConfig {
            raw: row.config.clone(),
            ..TradingProcessConfig::default()
        });
    Ok(TradingProcess {
        process_id: row.process_id,
        name: row.name,
        process_type: row.process_type,
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
