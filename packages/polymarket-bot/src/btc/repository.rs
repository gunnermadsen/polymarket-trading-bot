use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{postgres::PgPoolOptions, FromRow, PgPool, Postgres, Transaction};
use uuid::Uuid;

use crate::config::PostgresConfig;

use super::{
    admission::{
        DailyRealizedPnlCredit, DailyRealizedPnlHighWaterMarkState, LossRegimeCandidate,
        ShadowPredictiveRegimeCandidate, ShadowPredictiveRegimeEvaluation,
        ShadowPredictiveRegimeState, UnsettledEntryExposure,
    },
    strategy::{
        BtcDecision, BtcDecisionAction, BtcFeatureSnapshot, BtcStrategyPrediction,
        FairValueEstimate,
    },
    types::{
        BtcIntervalMarket, BtcOutcome, FeedIntegrityStatus, MarketFeedEvent, OrderbookCheckpoint,
        OrderbookLevel, ReferencePriceSource, ReferencePriceTick,
    },
};

#[derive(Debug, Clone)]
pub struct BtcRepository {
    pool: PgPool,
}

#[derive(Debug, Default, PartialEq)]
struct BtcDecisionEdgeProjection<'a> {
    token_id: Option<&'a str>,
    fair_probability: Option<Decimal>,
    executable_price: Option<Decimal>,
    gross_edge_per_share: Option<Decimal>,
    fee_per_share: Option<Decimal>,
    reserve_per_share: Option<Decimal>,
    net_edge_per_share: Option<Decimal>,
    size: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedSession {
    pub connection_id: Uuid,
    pub feed_name: String,
    pub endpoint: String,
    pub reconnect_ordinal: i32,
    pub started_at: DateTime<Utc>,
    pub connected_at: Option<DateTime<Utc>>,
    pub disconnected_at: Option<DateTime<Utc>>,
    pub messages_received: i64,
    pub messages_persisted: i64,
    pub decode_errors: i64,
    pub integrity_gaps: i64,
    pub dropped_messages: i64,
    pub disconnect_reason: Option<String>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BtcMarketLabel {
    pub market_id: String,
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    pub open_price: Decimal,
    pub close_price: Decimal,
    pub outcome: BtcOutcome,
    pub label_source: String,
    pub label_version: String,
    pub source_open_timestamp: DateTime<Utc>,
    pub source_close_timestamp: DateTime<Utc>,
    pub label_available_at: DateTime<Utc>,
    pub evidence: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct BtcOfficialResolutionWatch {
    pub market: BtcIntervalMarket,
    pub status: String,
    pub deadline_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BtcOfficialResolutionSubscriptionAck {
    pub subscribed: u64,
    pub already_terminal: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedOfficialResolution {
    pub market_id: String,
    pub winning_token_id: String,
    pub newly_recorded: bool,
}

#[derive(Debug, Clone)]
pub struct BtcPointInTimeInputs {
    pub chainlink_open: Option<ReferencePriceTick>,
    pub chainlink_current: Option<ReferencePriceTick>,
    /// Ascending event-time order, restricted to information received by `as_of`.
    pub chainlink_history: Vec<ReferencePriceTick>,
    /// Ascending event-time order, restricted to information received by `as_of`.
    pub binance_history: Vec<ReferencePriceTick>,
    pub up_book: Option<OrderbookCheckpoint>,
    pub down_book: Option<OrderbookCheckpoint>,
    pub fee_rate: Option<Decimal>,
    pub fee_observed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, FromRow)]
pub struct BtcPaperExperimentStatus {
    pub experiment_id: Uuid,
    pub name: String,
    pub status: String,
    pub started_at: Option<DateTime<Utc>>,
    pub stopped_at: Option<DateTime<Utc>>,
    pub markets_observed: i64,
    pub snapshots_recorded: i64,
    pub decisions_recorded: i64,
    pub trades_entered: i64,
    pub trades_resolved: i64,
    pub gross_pnl: Decimal,
    pub fees_paid: Decimal,
    pub net_pnl: Decimal,
    pub summary: serde_json::Value,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BtcPaperVenueResumeState {
    pub entry_debits_usd: Decimal,
    pub settlement_credits_usd: Decimal,
    pub order_count: usize,
    pub fill_count: usize,
    pub credited_settlement_ids: Vec<Uuid>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, FromRow)]
pub struct BtcExperimentSettlementSummary {
    pub trades_entered: i64,
    pub trades_resolved: i64,
    pub gross_pnl: Decimal,
    pub fees_paid: Decimal,
    pub net_pnl: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct BtcPaperSettlementRecord {
    pub settlement_id: Uuid,
    pub experiment_id: Uuid,
    pub process_id: Uuid,
    pub order_id: String,
    pub market_id: String,
    pub token_id: String,
    pub fill_ids: serde_json::Value,
    pub official_outcome: String,
    pub official_winning_token_id: String,
    pub official_resolution_received_at: DateTime<Utc>,
    pub official_resolution_source: String,
    pub filled_size: Decimal,
    pub entry_notional: Decimal,
    pub entry_fees: Decimal,
    pub payout: Decimal,
    pub net_pnl: Decimal,
    pub credit_status: String,
    pub credited_at: Option<DateTime<Utc>>,
    pub credit_attempts: i64,
    pub credit_evidence: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, FromRow)]
pub struct BtcPaperSettlementLedgerSummary {
    pub settlements_discovered: i64,
    pub settlements_credited: i64,
    pub settlements_pending: i64,
    pub entry_notional: Decimal,
    pub entry_fees: Decimal,
    pub payout_credited: Decimal,
    pub net_pnl: Decimal,
    pub last_official_resolution_received_at: Option<DateTime<Utc>>,
    pub last_credited_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, FromRow)]
struct ReferenceTickRow {
    tick_id: Uuid,
    source_timestamp: DateTime<Utc>,
    received_at: DateTime<Utc>,
    source: String,
    symbol: String,
    price: Decimal,
    envelope_timestamp: Option<DateTime<Utc>>,
    connection_id: Uuid,
    ingest_sequence: i64,
    source_event_id: Option<String>,
    dedup_key: String,
    raw_payload: serde_json::Value,
}

#[derive(Debug, Clone, FromRow)]
struct MarketOpenReferenceRow {
    market_id: String,
    tick_id: Uuid,
    source_timestamp: DateTime<Utc>,
    received_at: DateTime<Utc>,
    source: String,
    symbol: String,
    price: Decimal,
    envelope_timestamp: Option<DateTime<Utc>>,
    connection_id: Uuid,
    ingest_sequence: i64,
    source_event_id: Option<String>,
    dedup_key: String,
    raw_payload: serde_json::Value,
}

#[derive(Debug, Clone, FromRow)]
struct MarketBoundaryRow {
    market_id: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    up_token_id: String,
    down_token_id: String,
    reference_price: Option<Decimal>,
    reference_source_timestamp: Option<DateTime<Utc>>,
    resolution_price: Option<Decimal>,
    resolution_source_timestamp: Option<DateTime<Utc>>,
    official_outcome: Option<String>,
    official_resolved_at: Option<DateTime<Utc>>,
    official_winning_token_id: Option<String>,
}

#[derive(Debug, Clone, FromRow)]
struct ResolutionWatchMarketRow {
    event_id: String,
    event_slug: String,
    series_slug: String,
    market_id: String,
    condition_id: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    up_token_id: String,
    down_token_id: String,
    min_tick_size: Decimal,
    min_order_size: Decimal,
    resolution_source: String,
    accepting_orders: bool,
    active: bool,
    closed: bool,
    fee_rate: Option<Decimal>,
    fee_exponent: Option<i32>,
    fee_taker_only: Option<bool>,
    raw_payload: serde_json::Value,
    watch_status: String,
    deadline_at: DateTime<Utc>,
}

#[derive(Debug, Clone, FromRow)]
struct StoredMarketLabelRow {
    market_id: String,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    open_price: Decimal,
    close_price: Decimal,
    outcome: String,
    label_source: String,
    label_version: String,
    source_open_timestamp: DateTime<Utc>,
    source_close_timestamp: DateTime<Utc>,
    label_available_at: DateTime<Utc>,
    evidence: serde_json::Value,
}

#[derive(Debug, Clone, FromRow)]
struct LossRegimeCandidateRow {
    market_id: String,
    decision_id: Uuid,
    decision_outcome: String,
    resolved_outcome: String,
    decision_at: DateTime<Utc>,
    label_available_at: DateTime<Utc>,
}

#[derive(Debug, Clone, FromRow)]
struct ShadowPredictiveRegimeCandidateRow {
    market_id: String,
    decision_id: Uuid,
    decision_outcome: String,
    resolved_outcome: String,
    selected_point_probability: Decimal,
    decision_at: DateTime<Utc>,
    label_available_at: DateTime<Utc>,
}

#[derive(Debug, Clone, FromRow)]
struct PersistedShadowPredictiveRegimeEvaluationRow {
    decision_at: DateTime<Utc>,
    evaluation: serde_json::Value,
}

#[derive(Debug, Clone, FromRow)]
struct DailyHighWaterMarkEvidenceRow {
    evidence_kind: String,
    settlement_id: Option<Uuid>,
    order_id: String,
    occurred_at: DateTime<Utc>,
    amount_usd: Decimal,
    fill_ids: Vec<Uuid>,
}

const DAILY_HIGH_WATER_MARK_EVIDENCE_SQL: &str = r#"
WITH causal_credited_rows AS (
  SELECT
    l.process_id,
    l.order_id,
    l.settlement_id,
    l.credited_at,
    l.filled_size,
    l.entry_notional,
    l.entry_fees,
    l.payout,
    l.net_pnl
  FROM polymarket.btc_paper_settlement_ledger l
  WHERE l.process_id = $1
    AND l.credit_status = 'credited'
    AND l.credited_at IS NOT NULL
    AND l.credited_at <= $2
), conflicting_credited_orders AS (
  SELECT
    process_id,
    order_id,
    min(credited_at) AS first_credited_at,
    count(DISTINCT (
      filled_size,
      entry_notional,
      entry_fees,
      payout,
      net_pnl
    ))::numeric AS distinct_economics_count
  FROM causal_credited_rows
  GROUP BY process_id, order_id
  HAVING count(DISTINCT (
    filled_size,
    entry_notional,
    entry_fees,
    payout,
    net_pnl
  )) > 1
), canonical_credited_orders AS (
  SELECT DISTINCT ON (l.process_id, l.order_id)
    l.process_id,
    l.order_id,
    l.settlement_id,
    l.credited_at,
    l.net_pnl
  FROM causal_credited_rows l
  ORDER BY l.process_id, l.order_id, l.credited_at, l.settlement_id
), daily_credits AS (
  SELECT process_id, order_id, settlement_id, credited_at, net_pnl
  FROM canonical_credited_orders
  WHERE credited_at >= $3
    AND credited_at < $4
), process_entry_orders AS (
  SELECT DISTINCT o.process_id, o.order_id
  FROM polymarket.orders o
  WHERE o.process_id = $1
    AND o.created_at <= $2
    AND o.raw_payload #>> '{request,metadata,execution_intent}' = 'entry'
), process_paper_fills AS (
  SELECT
    o.process_id,
    o.order_id,
    max(f.timestamp_utc) AS last_filled_at,
    sum(f.price * f.size + f.fee)::numeric AS entry_debit_usd,
    array_agg(f.fill_id ORDER BY f.timestamp_utc, f.fill_id) AS fill_ids
  FROM process_entry_orders o
  JOIN polymarket.fills f
    ON f.process_id = o.process_id
   AND f.order_id = o.order_id
  WHERE f.source = 'paper'
    AND f.timestamp_utc <= $2
  GROUP BY o.process_id, o.order_id
), unsettled_orders AS (
  SELECT f.process_id, f.order_id, f.last_filled_at, f.entry_debit_usd, f.fill_ids
  FROM process_paper_fills f
  LEFT JOIN canonical_credited_orders c
    ON c.process_id = f.process_id
   AND c.order_id = f.order_id
  WHERE c.order_id IS NULL
)
SELECT
  'credited'::text AS evidence_kind,
  settlement_id,
  order_id,
  credited_at AS occurred_at,
  net_pnl::numeric AS amount_usd,
  ARRAY[]::uuid[] AS fill_ids
FROM daily_credits
UNION ALL
SELECT
  'unsettled'::text AS evidence_kind,
  NULL::uuid AS settlement_id,
  order_id,
  last_filled_at AS occurred_at,
  entry_debit_usd AS amount_usd,
  fill_ids
FROM unsettled_orders
UNION ALL
SELECT
  'conflicting_credit'::text AS evidence_kind,
  NULL::uuid AS settlement_id,
  order_id,
  first_credited_at AS occurred_at,
  distinct_economics_count AS amount_usd,
  ARRAY[]::uuid[] AS fill_ids
FROM conflicting_credited_orders
ORDER BY evidence_kind, occurred_at, order_id, settlement_id
"#;

const PROCESS_HAS_ENTRY_SQL: &str = r#"
SELECT EXISTS (
  SELECT 1
  FROM polymarket.btc_strategy_decisions
  WHERE process_id = $1
    AND market_id = $2
    AND action = 'buy'
    AND status IN ('approved','submitted','filled')
)
"#;

// One look-ahead row lets replay distinguish an exact 10,000-candidate history from truncation.
const MAX_SHADOW_PREDICTIVE_REGIME_HISTORY_CANDIDATES: u32 = 10_001;

const LATEST_SHADOW_PREDICTIVE_REGIME_STATE_SQL: &str = r#"
SELECT
  d.decision_at,
  d.metadata #> '{entry_admission,shadow_predictive_regime_circuit_breaker}' AS evaluation
FROM polymarket.btc_strategy_decisions d
WHERE d.process_id = $1
  AND ($2::text IS NULL OR d.metadata #>> '{entry_admission,shadow_predictive_regime_circuit_breaker,state,config_hash}' = $2)
  AND d.decision_at <= $3
  AND d.action = 'buy'
  AND d.metadata #>> '{entry_admission,shadow_predictive_regime_circuit_breaker,state_checkpoint_eligible}' = 'true'
  AND jsonb_typeof(
    d.metadata #> '{entry_admission,shadow_predictive_regime_circuit_breaker}'
  ) = 'object'
ORDER BY d.decision_at DESC, d.decision_id DESC
LIMIT 1
"#;

const SHADOW_PREDICTIVE_REGIME_CANDIDATES_SQL: &str = r#"
WITH canonical AS MATERIALIZED (
  SELECT DISTINCT ON (d.market_id)
    d.market_id,
    d.decision_id,
    d.snapshot_id,
    d.outcome AS decision_outcome,
    l.outcome AS resolved_outcome,
    d.decision_at,
    l.label_available_at
  FROM polymarket.btc_strategy_decisions d
  JOIN polymarket.btc_market_labels l
    ON l.market_id = d.market_id
  WHERE d.process_id = $1
    AND ($2::text IS NULL OR d.config_hash = $2)
    AND d.action = 'buy'
    AND d.outcome IN ('up', 'down')
    AND d.decision_at < l.label_available_at
    AND l.label_available_at <= $3
  ORDER BY d.market_id, d.decision_at, d.decision_id
), scored AS (
  SELECT
    c.market_id,
    c.decision_id,
    c.decision_outcome,
    c.resolved_outcome,
    CASE c.decision_outcome
      WHEN 'up' THEN f.fair_up_probability
      WHEN 'down' THEN 1 - f.fair_up_probability
    END AS selected_point_probability,
    c.decision_at,
    c.label_available_at
  FROM canonical c
  CROSS JOIN LATERAL (
    SELECT f.fair_up_probability
    FROM polymarket.btc_feature_snapshots f
    WHERE f.snapshot_id = c.snapshot_id
      AND f.feature_as_of = c.decision_at
      AND f.fair_up_probability BETWEEN 0 AND 1
    OFFSET 0
  ) f
), bounded AS (
  SELECT
    market_id,
    decision_id,
    decision_outcome,
    resolved_outcome,
    selected_point_probability,
    decision_at,
    label_available_at
  FROM scored
  ORDER BY label_available_at DESC, decision_at DESC, decision_id DESC, market_id DESC
  LIMIT $4
)
SELECT
  market_id,
  decision_id,
  decision_outcome,
  resolved_outcome,
  selected_point_probability,
  decision_at,
  label_available_at
FROM bounded
ORDER BY label_available_at, decision_at, decision_id, market_id
"#;

#[derive(Debug, Clone, FromRow)]
struct CheckpointRow {
    checkpoint_id: Uuid,
    source_timestamp: DateTime<Utc>,
    received_at: DateTime<Utc>,
    connection_id: Uuid,
    ingest_sequence: i64,
    market_id: String,
    token_id: String,
    best_bid: Option<Decimal>,
    best_ask: Option<Decimal>,
    tick_size: Decimal,
    book: serde_json::Value,
    source_hash: Option<String>,
    integrity_status: String,
}

fn validate_official_resolution_subscription_ack(
    market_ids: &[String],
    subscribed_market_ids: &[String],
    watch_states: &[(String, String)],
) -> Result<BtcOfficialResolutionSubscriptionAck> {
    let requested = market_ids.iter().cloned().collect::<HashSet<_>>();
    if requested.len() != market_ids.len() {
        bail!("BTC official-resolution subscription contains duplicate market ids");
    }
    let subscribed = subscribed_market_ids
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    if subscribed.len() != subscribed_market_ids.len()
        || subscribed
            .iter()
            .any(|market_id| !requested.contains(market_id))
    {
        bail!("BTC official-resolution subscription returned invalid updated market ids");
    }
    let states = watch_states.iter().cloned().collect::<HashMap<_, _>>();
    if states.len() != watch_states.len() {
        bail!("BTC official-resolution subscription returned duplicate watch rows");
    }

    let missing = requested
        .iter()
        .filter(|market_id| !states.contains_key(*market_id))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "BTC official-resolution subscription is missing durable watches for: {}",
            missing.join(",")
        );
    }

    let mut already_terminal = 0u64;
    let mut invalid = Vec::new();
    for market_id in market_ids {
        let status = states
            .get(market_id)
            .expect("every requested BTC resolution watch was verified above");
        if subscribed.contains(market_id) {
            if status != "pending" {
                invalid.push(format!("{market_id}={status}:updated"));
            }
        } else if matches!(status.as_str(), "resolved" | "resolved_late") {
            already_terminal = already_terminal.saturating_add(1);
        } else {
            invalid.push(format!("{market_id}={status}:not_updated"));
        }
    }
    if !invalid.is_empty() {
        bail!(
            "BTC official-resolution subscription has invalid durable watch states: {}",
            invalid.join(",")
        );
    }

    Ok(BtcOfficialResolutionSubscriptionAck {
        subscribed: subscribed.len() as u64,
        already_terminal,
    })
}

impl BtcRepository {
    pub async fn connect(config: &PostgresConfig) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(4)
            .connect(&config.database_url())
            .await
            .context("failed to connect BTC realtime repository to Postgres")?;
        Ok(Self { pool })
    }

    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn healthcheck(&self) -> Result<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .context("BTC realtime repository healthcheck failed")?;
        Ok(())
    }

    pub async fn upsert_interval_market(&self, market: &BtcIntervalMarket) -> Result<()> {
        let minimum_order_size = market.minimum_order_size.unwrap_or(Decimal::ZERO);
        let (validation_status, validation_errors) = if market.minimum_order_size.is_some() {
            ("valid", serde_json::json!([]))
        } else {
            (
                "ineligible",
                serde_json::json!(["missing_minimum_order_size"]),
            )
        };
        let fee_rate = decimal_json_field(&market.fee_schedule, &["rate"]);
        let fee_exponent = integer_json_field(&market.fee_schedule, &["exponent"]);
        let fee_taker_only = bool_json_field(&market.fee_schedule, &["takerOnly", "taker_only"]);
        let question = market
            .raw_payload
            .pointer("/markets/0/question")
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                market
                    .raw_payload
                    .get("title")
                    .and_then(serde_json::Value::as_str)
            })
            .unwrap_or(&market.event_slug);
        let now = Utc::now();
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin BTC interval market upsert transaction")?;
        sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
            .bind(&market.event_slug)
            .execute(&mut *tx)
            .await
            .context("failed to lock BTC interval market identity")?;
        let result = sqlx::query(
            r#"
            INSERT INTO polymarket.btc_interval_markets (
              market_id, event_id, event_slug, question, series_slug, window_start, window_end,
              condition_id, up_token_id, down_token_id, resolution_source, accepting_orders,
              active, closed, min_tick_size, min_order_size, fee_rate, fee_exponent,
              fee_taker_only, validation_status, validation_errors, discovered_at,
              last_refreshed_at, raw_payload
            )
            VALUES (
              $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,
              $22,$23,$24
            )
            ON CONFLICT (market_id) DO UPDATE SET
              question = EXCLUDED.question,
              accepting_orders = EXCLUDED.accepting_orders,
              active = EXCLUDED.active,
              closed = EXCLUDED.closed,
              min_tick_size = EXCLUDED.min_tick_size,
              min_order_size = EXCLUDED.min_order_size,
              fee_rate = EXCLUDED.fee_rate,
              fee_exponent = EXCLUDED.fee_exponent,
              fee_taker_only = EXCLUDED.fee_taker_only,
              validation_status = EXCLUDED.validation_status,
              validation_errors = EXCLUDED.validation_errors,
              last_refreshed_at = EXCLUDED.last_refreshed_at,
              raw_payload = EXCLUDED.raw_payload,
              updated_at = now()
            WHERE polymarket.btc_interval_markets.event_id = EXCLUDED.event_id
              AND polymarket.btc_interval_markets.event_slug = EXCLUDED.event_slug
              AND polymarket.btc_interval_markets.series_slug = EXCLUDED.series_slug
              AND polymarket.btc_interval_markets.window_start = EXCLUDED.window_start
              AND polymarket.btc_interval_markets.window_end = EXCLUDED.window_end
              AND polymarket.btc_interval_markets.condition_id = EXCLUDED.condition_id
              AND polymarket.btc_interval_markets.up_token_id = EXCLUDED.up_token_id
              AND polymarket.btc_interval_markets.down_token_id = EXCLUDED.down_token_id
              AND polymarket.btc_interval_markets.resolution_source = EXCLUDED.resolution_source
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
        .context("failed to upsert BTC interval market")?;
        if result.rows_affected() != 1 {
            bail!(
                "immutable BTC market identity conflict for market {}",
                market.market_id
            );
        }
        tx.commit()
            .await
            .context("failed to commit BTC interval market upsert")?;
        Ok(())
    }

    /// Rehydrates unresolved obligations on startup. Recent markets are retained normally; an
    /// older market with a paper fill is also retained so a restart cannot hide unsettled risk.
    pub async fn seed_recent_official_resolution_watches(
        &self,
        now: DateTime<Utc>,
        retention: Duration,
    ) -> Result<u64> {
        let earliest_window_end = now - retention;
        let latest_window_start = now + Duration::minutes(5);
        let result = sqlx::query(
            r#"
            INSERT INTO polymarket.btc_official_resolution_watches (
              market_id, status, watch_started_at, deadline_at
            )
            SELECT
              m.market_id,
              'pending',
              LEAST($1, m.window_end),
              m.window_end + ($4::double precision * interval '1 second')
            FROM polymarket.btc_interval_markets m
            WHERE m.validation_status = 'valid'
              AND m.official_outcome IS NULL
              AND m.window_start <= $3
              AND (
                m.window_end > $2
                OR EXISTS (
                  SELECT 1
                  FROM polymarket.orders o
                  JOIN polymarket.fills f ON f.order_id = o.order_id
                  WHERE o.market_id = m.market_id
                    AND f.source = 'paper'
                )
              )
            ON CONFLICT (market_id) DO NOTHING
            "#,
        )
        .bind(now)
        .bind(earliest_window_end)
        .bind(latest_window_start)
        .bind(retention.num_seconds())
        .execute(&self.pool)
        .await
        .context("failed to seed durable BTC official-resolution watches")?;
        Ok(result.rows_affected())
    }

    pub async fn register_official_resolution_watch(
        &self,
        market: &BtcIntervalMarket,
        now: DateTime<Utc>,
        retention: Duration,
    ) -> Result<()> {
        let watch_started_at = now.min(market.window_end);
        let deadline_at = market.window_end + retention;
        sqlx::query(
            r#"
            INSERT INTO polymarket.btc_official_resolution_watches (
              market_id, status, watch_started_at, deadline_at
            )
            SELECT market_id, 'pending', $2, $3
            FROM polymarket.btc_interval_markets
            WHERE market_id = $1 AND official_outcome IS NULL
            ON CONFLICT (market_id) DO NOTHING
            "#,
        )
        .bind(&market.market_id)
        .bind(watch_started_at)
        .bind(deadline_at)
        .execute(&self.pool)
        .await
        .context("failed to register BTC official-resolution watch")?;
        Ok(())
    }

    /// Includes expired unresolved watches so restart reconciliation gets one final chance before
    /// the runtime fails visibly. Only pending rows are sent to the websocket subscriber.
    pub async fn load_unsettled_official_resolution_watches(
        &self,
    ) -> Result<Vec<BtcOfficialResolutionWatch>> {
        sqlx::query_as::<_, ResolutionWatchMarketRow>(
            r#"
            SELECT
              m.event_id, m.event_slug, m.series_slug, m.market_id, m.condition_id,
              m.window_start, m.window_end, m.up_token_id, m.down_token_id,
              m.min_tick_size, m.min_order_size, m.resolution_source,
              m.accepting_orders, m.active, m.closed, m.fee_rate, m.fee_exponent,
              m.fee_taker_only, m.raw_payload, w.status AS watch_status, w.deadline_at
            FROM polymarket.btc_official_resolution_watches w
            JOIN polymarket.btc_interval_markets m ON m.market_id = w.market_id
            WHERE w.status IN ('pending','expired')
              AND m.official_outcome IS NULL
            ORDER BY m.window_start, m.market_id
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .context("failed to load unsettled BTC official-resolution watches")?
        .into_iter()
        .map(official_resolution_watch_from_row)
        .collect()
    }

    pub async fn mark_official_resolution_watches_subscribed(
        &self,
        market_ids: &[String],
        connection_id: Uuid,
        subscribed_at: DateTime<Utc>,
    ) -> Result<BtcOfficialResolutionSubscriptionAck> {
        if market_ids.is_empty() {
            return Ok(BtcOfficialResolutionSubscriptionAck::default());
        }
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin BTC official-resolution subscription transaction")?;
        let subscribed_market_ids = sqlx::query_scalar::<_, String>(
            r#"
            UPDATE polymarket.btc_official_resolution_watches
            SET last_subscribed_at = $2,
                last_subscription_connection_id = $3,
                subscription_count = subscription_count + 1,
                updated_at = now()
            WHERE market_id = ANY($1) AND status = 'pending'
            RETURNING market_id
            "#,
        )
        .bind(market_ids)
        .bind(subscribed_at)
        .bind(connection_id)
        .fetch_all(&mut *tx)
        .await
        .context("failed to acknowledge BTC official-resolution subscriptions")?;
        let watch_states = sqlx::query_as::<_, (String, String)>(
            r#"
            SELECT market_id, status
            FROM polymarket.btc_official_resolution_watches
            WHERE market_id = ANY($1)
            ORDER BY market_id
            "#,
        )
        .bind(market_ids)
        .fetch_all(&mut *tx)
        .await
        .context("failed to verify BTC official-resolution subscription states")?;
        let acknowledgement = validate_official_resolution_subscription_ack(
            market_ids,
            &subscribed_market_ids,
            &watch_states,
        )?;
        tx.commit()
            .await
            .context("failed to commit BTC official-resolution subscriptions")?;
        Ok(acknowledgement)
    }

    pub async fn mark_official_resolution_watch_checked(
        &self,
        market_id: &str,
        checked_at: DateTime<Utc>,
        error: Option<&str>,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE polymarket.btc_official_resolution_watches
            SET last_checked_at = $2,
                last_error = $3,
                updated_at = now()
            WHERE market_id = $1
            "#,
        )
        .bind(market_id)
        .bind(checked_at)
        .bind(error)
        .execute(&self.pool)
        .await
        .context("failed to update BTC official-resolution reconciliation status")?;
        Ok(())
    }

    pub async fn expire_overdue_official_resolution_watches(
        &self,
        now: DateTime<Utc>,
    ) -> Result<Vec<String>> {
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin BTC official-resolution expiry transaction")?;
        let market_ids = sqlx::query_scalar::<_, String>(
            r#"
            SELECT w.market_id
            FROM polymarket.btc_interval_markets m
            JOIN polymarket.btc_official_resolution_watches w ON w.market_id = m.market_id
            WHERE w.status = 'pending'
              AND w.deadline_at <= $1
              AND m.official_outcome IS NULL
            ORDER BY w.market_id
            FOR UPDATE OF m, w
            "#,
        )
        .bind(now)
        .fetch_all(&mut *tx)
        .await
        .context("failed to lock overdue BTC official-resolution watches")?;
        if !market_ids.is_empty() {
            sqlx::query(
                r#"
                UPDATE polymarket.btc_official_resolution_watches
                SET status = 'expired', expired_at = $2, updated_at = now()
                WHERE market_id = ANY($1) AND status = 'pending'
                "#,
            )
            .bind(&market_ids)
            .bind(now)
            .execute(&mut *tx)
            .await
            .context("failed to expire overdue BTC official-resolution watches")?;
        }
        tx.commit()
            .await
            .context("failed to commit BTC official-resolution expiry transaction")?;
        Ok(market_ids)
    }

    pub async fn insert_reference_tick(&self, tick: &ReferencePriceTick) -> Result<bool> {
        let clock_skew_ms = (tick.received_at - tick.source_timestamp).num_milliseconds();
        let integrity_status = reference_tick_integrity_status(tick);
        let result = sqlx::query(
            r#"
            INSERT INTO polymarket.reference_price_ticks (
              tick_id, source_timestamp, received_at, source, symbol, price, envelope_timestamp,
              connection_id, ingest_sequence, source_event_id, dedup_key, clock_skew_ms,
              integrity_status, raw_payload
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
            ON CONFLICT DO NOTHING
            "#,
        )
        .bind(tick.tick_id)
        .bind(tick.source_timestamp)
        .bind(tick.received_at)
        .bind(tick.source.as_str())
        .bind(&tick.symbol)
        .bind(tick.price)
        .bind(tick.envelope_timestamp)
        .bind(tick.connection_id)
        .bind(sequence_i64(tick.ingest_sequence))
        .bind(&tick.source_event_id)
        .bind(&tick.dedup_key)
        .bind(clock_skew_ms)
        .bind(integrity_status)
        .bind(&tick.raw_payload)
        .execute(&self.pool)
        .await
        .context("failed to insert BTC reference price tick")?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn insert_feed_event(&self, event: &MarketFeedEvent) -> Result<bool> {
        let result = sqlx::query(
            r#"
            INSERT INTO polymarket.market_feed_events (
              feed_event_id, source_timestamp, received_at, connection_id, ingest_sequence,
              market_id, token_id, event_type, source_hash, applied, integrity_status, raw_payload
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (feed_event_id, source_timestamp) DO NOTHING
            "#,
        )
        .bind(event.event_id)
        .bind(event.source_timestamp)
        .bind(event.received_at)
        .bind(event.connection_id)
        .bind(sequence_i64(event.ingest_sequence))
        .bind(empty_to_none(&event.market_id))
        .bind(&event.token_id)
        .bind(serde_name(&event.event_type)?)
        .bind(&event.source_hash)
        .bind(event.applied)
        .bind(serde_name(&event.integrity_status)?)
        .bind(&event.raw_payload)
        .execute(&self.pool)
        .await
        .context("failed to insert BTC market feed event")?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn insert_orderbook_checkpoint(
        &self,
        checkpoint: &OrderbookCheckpoint,
        bootstrap_source: &str,
    ) -> Result<bool> {
        let depth_bid: Decimal = checkpoint.bids.iter().map(|level| level.size).sum();
        let depth_ask: Decimal = checkpoint.asks.iter().map(|level| level.size).sum();
        let spread = checkpoint
            .best_bid
            .zip(checkpoint.best_ask)
            .map(|(bid, ask)| ask - bid);
        let book = serde_json::json!({
            "bids": checkpoint.bids,
            "asks": checkpoint.asks,
        });
        let result = sqlx::query(
            r#"
            INSERT INTO polymarket.orderbook_checkpoints (
              checkpoint_id, source_timestamp, received_at, connection_id, ingest_sequence,
              market_id, token_id, best_bid, best_ask, spread, tick_size, depth_bid, depth_ask,
              book, source_hash, bootstrap_source, integrity_status
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
            ON CONFLICT (checkpoint_id, source_timestamp) DO NOTHING
            "#,
        )
        .bind(checkpoint.checkpoint_id)
        .bind(checkpoint.source_timestamp)
        .bind(checkpoint.received_at)
        .bind(checkpoint.connection_id)
        .bind(sequence_i64(checkpoint.ingest_sequence))
        .bind(&checkpoint.market_id)
        .bind(&checkpoint.token_id)
        .bind(checkpoint.best_bid)
        .bind(checkpoint.best_ask)
        .bind(spread)
        .bind(checkpoint.tick_size)
        .bind(depth_bid)
        .bind(depth_ask)
        .bind(book)
        .bind(&checkpoint.source_hash)
        .bind(bootstrap_source)
        .bind(serde_name(&checkpoint.integrity_status)?)
        .execute(&self.pool)
        .await
        .context("failed to insert BTC orderbook checkpoint")?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn start_feed_session(&self, session: &FeedSession) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO polymarket.feed_sessions (
              connection_id, feed_name, endpoint, reconnect_ordinal, started_at, connected_at,
              messages_received, messages_persisted, decode_errors, integrity_gaps,
              dropped_messages, metadata
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (connection_id) DO NOTHING
            "#,
        )
        .bind(session.connection_id)
        .bind(&session.feed_name)
        .bind(&session.endpoint)
        .bind(session.reconnect_ordinal)
        .bind(session.started_at)
        .bind(session.connected_at)
        .bind(session.messages_received)
        .bind(session.messages_persisted)
        .bind(session.decode_errors)
        .bind(session.integrity_gaps)
        .bind(session.dropped_messages)
        .bind(&session.metadata)
        .execute(&self.pool)
        .await
        .context("failed to start BTC feed session")?;
        Ok(())
    }

    pub async fn finish_feed_session(&self, session: &FeedSession) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE polymarket.feed_sessions
            SET connected_at = COALESCE($2, connected_at),
                disconnected_at = $3,
                messages_received = $4,
                messages_persisted = $5,
                decode_errors = $6,
                integrity_gaps = $7,
                dropped_messages = $8,
                disconnect_reason = $9,
                metadata = $10,
                updated_at = now()
            WHERE connection_id = $1
            "#,
        )
        .bind(session.connection_id)
        .bind(session.connected_at)
        .bind(session.disconnected_at)
        .bind(session.messages_received)
        .bind(session.messages_persisted)
        .bind(session.decode_errors)
        .bind(session.integrity_gaps)
        .bind(session.dropped_messages)
        .bind(&session.disconnect_reason)
        .bind(&session.metadata)
        .execute(&self.pool)
        .await
        .context("failed to finish BTC feed session")?;
        Ok(())
    }

    /// Persists the first eligible Chainlink tick for a market exactly once. Replays of the same
    /// boundary are idempotent; a different value is an integrity conflict and is never silently
    /// substituted.
    pub async fn persist_immutable_market_open(
        &self,
        market_id: &str,
        tick: &ReferencePriceTick,
        max_delay: Duration,
    ) -> Result<()> {
        require_healthy_chainlink_tick(tick, "opening")?;
        self.insert_reference_tick(tick).await?;
        self.require_durable_reference_tick(tick).await?;

        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin BTC opening-reference transaction")?;
        let market = load_market_boundary_for_update(&mut tx, market_id).await?;
        require_tick_in_boundary_window(
            tick,
            market.window_start,
            market.window_start + max_delay,
            "opening",
        )?;

        match (market.reference_price, market.reference_source_timestamp) {
            (None, None) => {
                sqlx::query(
                    r#"
                    UPDATE polymarket.btc_interval_markets
                    SET reference_price = $2,
                        reference_source_timestamp = $3,
                        updated_at = now()
                    WHERE market_id = $1
                    "#,
                )
                .bind(market_id)
                .bind(tick.price)
                .bind(tick.source_timestamp)
                .execute(&mut *tx)
                .await
                .context("failed to persist immutable BTC opening reference")?;
            }
            (Some(price), Some(timestamp))
                if price == tick.price && timestamp == tick.source_timestamp => {}
            (price, timestamp) => {
                bail!(
                    "immutable BTC opening-reference conflict for market {market_id}: stored ({price:?}, {timestamp:?}), candidate ({}, {})",
                    tick.price,
                    tick.source_timestamp
                );
            }
        }

        tx.commit()
            .await
            .context("failed to commit BTC opening-reference transaction")?;
        Ok(())
    }

    pub async fn load_market_open_references(
        &self,
        markets: &[BtcIntervalMarket],
    ) -> Result<Vec<(String, ReferencePriceTick)>> {
        if markets.is_empty() {
            return Ok(Vec::new());
        }
        let market_ids = markets
            .iter()
            .map(|market| market.market_id.clone())
            .collect::<Vec<_>>();
        sqlx::query_as::<_, MarketOpenReferenceRow>(
            r#"
            SELECT m.market_id, t.tick_id, t.source_timestamp, t.received_at, t.source,
              t.symbol, t.price, t.envelope_timestamp, t.connection_id,
              t.ingest_sequence, t.source_event_id, t.dedup_key, t.raw_payload
            FROM polymarket.btc_interval_markets m
            JOIN LATERAL (
              SELECT tick_id, source_timestamp, received_at, source, symbol, price,
                envelope_timestamp, connection_id, ingest_sequence, source_event_id,
                dedup_key, raw_payload
              FROM polymarket.reference_price_ticks
              WHERE source = 'rtds_chainlink'
                AND source_timestamp = m.reference_source_timestamp
                AND price = m.reference_price
              ORDER BY received_at ASC, ingest_sequence ASC
              LIMIT 1
            ) t ON true
            WHERE m.market_id = ANY($1)
              AND m.reference_price IS NOT NULL
              AND m.reference_source_timestamp IS NOT NULL
            "#,
        )
        .bind(&market_ids)
        .fetch_all(&self.pool)
        .await
        .context("failed to load durable BTC opening references")?
        .into_iter()
        .map(market_open_reference_from_row)
        .collect()
    }

    /// Loads the earliest healthy Chainlink tick in every discovered market's close-boundary
    /// acceptance window. This reconstructs pending labels after a process restart.
    pub async fn load_market_close_references(
        &self,
        markets: &[BtcIntervalMarket],
        max_delay: Duration,
    ) -> Result<Vec<(String, ReferencePriceTick)>> {
        if markets.is_empty() {
            return Ok(Vec::new());
        }
        let market_ids = markets
            .iter()
            .map(|market| market.market_id.clone())
            .collect::<Vec<_>>();
        let max_delay_ms = max_delay.num_milliseconds();
        if max_delay_ms < 0 {
            bail!("BTC close-boundary delay must not be negative");
        }
        sqlx::query_as::<_, MarketOpenReferenceRow>(
            r#"
            SELECT m.market_id, t.tick_id, t.source_timestamp, t.received_at, t.source,
              t.symbol, t.price, t.envelope_timestamp, t.connection_id,
              t.ingest_sequence, t.source_event_id, t.dedup_key, t.raw_payload
            FROM polymarket.btc_interval_markets m
            JOIN LATERAL (
              SELECT tick_id, source_timestamp, received_at, source, symbol, price,
                envelope_timestamp, connection_id, ingest_sequence, source_event_id,
                dedup_key, raw_payload
              FROM polymarket.reference_price_ticks
              WHERE source = 'rtds_chainlink'
                AND integrity_status = 'ok'
                AND source_timestamp >= m.window_end
                AND source_timestamp <= m.window_end + ($2::bigint * interval '1 millisecond')
              ORDER BY source_timestamp ASC, received_at ASC, ingest_sequence ASC, tick_id ASC
              LIMIT 1
            ) t ON true
            WHERE m.market_id = ANY($1)
            "#,
        )
        .bind(&market_ids)
        .bind(max_delay_ms)
        .fetch_all(&self.pool)
        .await
        .context("failed to load durable BTC closing references")?
        .into_iter()
        .map(market_open_reference_from_row)
        .collect()
    }

    pub async fn load_market_labels(
        &self,
        markets: &[BtcIntervalMarket],
    ) -> Result<Vec<BtcMarketLabel>> {
        if markets.is_empty() {
            return Ok(Vec::new());
        }
        let market_ids = markets
            .iter()
            .map(|market| market.market_id.clone())
            .collect::<Vec<_>>();
        sqlx::query_as::<_, StoredMarketLabelRow>(
            r#"
            SELECT market_id, window_start, window_end, open_price, close_price, outcome,
              label_source, label_version, source_open_timestamp, source_close_timestamp,
              label_available_at, evidence
            FROM polymarket.btc_market_labels
            WHERE market_id = ANY($1)
            "#,
        )
        .bind(&market_ids)
        .fetch_all(&self.pool)
        .await
        .context("failed to load durable BTC market labels")?
        .into_iter()
        .map(market_label_from_row)
        .collect()
    }

    /// Persists an eligible close tick independently of label creation so a restart can recover a
    /// close observed before its opening reference was acknowledged.
    pub async fn persist_boundary_close_tick(
        &self,
        market_id: &str,
        tick: &ReferencePriceTick,
        max_delay: Duration,
    ) -> Result<()> {
        require_healthy_chainlink_tick(tick, "closing")?;
        let market = sqlx::query_as::<_, MarketBoundaryRow>(
            r#"
            SELECT market_id, window_start, window_end, up_token_id, down_token_id,
              reference_price, reference_source_timestamp, resolution_price,
              resolution_source_timestamp, official_outcome, official_resolved_at,
              official_winning_token_id
            FROM polymarket.btc_interval_markets
            WHERE market_id = $1
            "#,
        )
        .bind(market_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to load BTC market for closing-reference persistence")?
        .with_context(|| format!("BTC market {market_id} does not exist"))?;
        require_tick_in_boundary_window(
            tick,
            market.window_end,
            market.window_end + max_delay,
            "closing",
        )?;
        self.insert_reference_tick(tick).await?;
        self.require_durable_reference_tick(tick).await?;

        let earliest = sqlx::query_as::<_, ReferenceTickRow>(
            r#"
            SELECT tick_id, source_timestamp, received_at, source, symbol, price,
              envelope_timestamp, connection_id, ingest_sequence, source_event_id,
              dedup_key, raw_payload
            FROM polymarket.reference_price_ticks
            WHERE source = 'rtds_chainlink'
              AND integrity_status = 'ok'
              AND source_timestamp >= $1
              AND source_timestamp <= $2
            ORDER BY source_timestamp ASC, received_at ASC, ingest_sequence ASC, tick_id ASC
            LIMIT 1
            "#,
        )
        .bind(market.window_end)
        .bind(market.window_end + max_delay)
        .fetch_optional(&self.pool)
        .await
        .context("failed to confirm earliest durable BTC closing reference")?
        .map(reference_tick_from_row)
        .transpose()?
        .context("persisted BTC close-boundary tick was not recoverable")?;
        if !same_boundary_tick(&earliest, tick) {
            bail!(
                "BTC close-boundary candidate for market {market_id} is not the earliest durable eligible tick"
            );
        }
        Ok(())
    }

    async fn require_durable_reference_tick(&self, tick: &ReferencePriceTick) -> Result<()> {
        let stored = sqlx::query_as::<_, ReferenceTickRow>(
            r#"
            SELECT tick_id, source_timestamp, received_at, source, symbol, price,
              envelope_timestamp, connection_id, ingest_sequence, source_event_id,
              dedup_key, raw_payload
            FROM polymarket.reference_price_ticks
            WHERE source = $1
              AND dedup_key = $2
              AND source_timestamp = $3
            LIMIT 1
            "#,
        )
        .bind(tick.source.as_str())
        .bind(&tick.dedup_key)
        .bind(tick.source_timestamp)
        .fetch_optional(&self.pool)
        .await
        .context("failed to verify durable BTC boundary tick")?
        .map(reference_tick_from_row)
        .transpose()?
        .context("BTC boundary tick was not durable after insert acknowledgment")?;
        if !same_boundary_tick(&stored, tick) {
            bail!("immutable BTC boundary tick conflicts with its durable deduplication key");
        }
        Ok(())
    }

    /// Inserts a local Chainlink-derived label once. Existing local label fields are immutable;
    /// only the separately stored official exchange outcome may be populated later.
    pub async fn persist_immutable_market_label(
        &self,
        label: &BtcMarketLabel,
        close_tick: &ReferencePriceTick,
        max_delay: Duration,
    ) -> Result<()> {
        require_healthy_chainlink_tick(close_tick, "closing")?;
        if label.close_price != close_tick.price
            || label.source_close_timestamp != close_tick.source_timestamp
        {
            bail!(
                "BTC label close evidence does not match its close-boundary tick for market {}",
                label.market_id
            );
        }
        self.persist_boundary_close_tick(&label.market_id, close_tick, max_delay)
            .await?;
        let outcome = outcome_name(label.outcome);
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin BTC label transaction")?;
        let market = load_market_boundary_for_update(&mut tx, &label.market_id).await?;
        require_label_matches_market(label, &market, max_delay)?;

        let stored = sqlx::query_as::<_, StoredMarketLabelRow>(
            r#"
            SELECT market_id, window_start, window_end, open_price, close_price, outcome,
              label_source, label_version, source_open_timestamp, source_close_timestamp,
              label_available_at, evidence
            FROM polymarket.btc_market_labels
            WHERE market_id = $1
            FOR UPDATE
            "#,
        )
        .bind(&label.market_id)
        .fetch_optional(&mut *tx)
        .await
        .context("failed to inspect immutable BTC market label")?;
        if let Some(stored) = stored {
            let stored = market_label_from_row(stored)?;
            if !same_immutable_market_label(&stored, label) {
                bail!(
                    "immutable BTC market-label conflict for market {}",
                    label.market_id
                );
            }
        } else {
            sqlx::query(
                r#"
                INSERT INTO polymarket.btc_market_labels (
                  market_id, window_start, window_end, open_price, close_price, outcome,
                  label_source, label_version, source_open_timestamp, source_close_timestamp,
                  label_available_at, official_outcome, official_resolved_at, evidence
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                "#,
            )
            .bind(&label.market_id)
            .bind(label.window_start)
            .bind(label.window_end)
            .bind(label.open_price)
            .bind(label.close_price)
            .bind(outcome)
            .bind(&label.label_source)
            .bind(&label.label_version)
            .bind(label.source_open_timestamp)
            .bind(label.source_close_timestamp)
            .bind(label.label_available_at)
            .bind(&market.official_outcome)
            .bind(market.official_resolved_at)
            .bind(&label.evidence)
            .execute(&mut *tx)
            .await
            .context("failed to insert immutable BTC market label")?;
        }

        match (
            market.resolution_price,
            market.resolution_source_timestamp,
        ) {
            (None, None) => {}
            (Some(price), Some(timestamp))
                if price == label.close_price && timestamp == label.source_close_timestamp => {}
            (price, timestamp) => bail!(
                "immutable BTC label projection conflict for market {}: stored ({price:?}, {timestamp:?}), candidate ({}, {})",
                label.market_id,
                label.close_price,
                label.source_close_timestamp
            ),
        }
        sqlx::query(
            r#"
            UPDATE polymarket.btc_interval_markets
            SET resolution_price = $2,
                resolution_source_timestamp = $3,
                updated_at = now()
            WHERE market_id = $1
            "#,
        )
        .bind(&label.market_id)
        .bind(label.close_price)
        .bind(label.source_close_timestamp)
        .execute(&mut *tx)
        .await
        .context("failed to project immutable BTC interval label")?;
        tx.commit()
            .await
            .context("failed to commit BTC label transaction")?;
        Ok(())
    }

    /// Persists the exchange's official winner independently of the local Chainlink label. The
    /// official fact may arrive before or after label creation and is immutable once observed.
    pub async fn persist_official_market_resolution(
        &self,
        market_or_condition_id: &str,
        winning_token_id: &str,
        winning_outcome: &str,
        source_timestamp: DateTime<Utc>,
        resolution_source: &str,
        received_at: DateTime<Utc>,
        payload: &serde_json::Value,
    ) -> Result<PersistedOfficialResolution> {
        if !matches!(
            resolution_source,
            "clob_websocket" | "clob_rest_reconciliation"
        ) {
            bail!("unsupported official BTC resolution source {resolution_source}");
        }
        if !payload.is_object() {
            bail!("official BTC resolution evidence must be a JSON object");
        }
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin official BTC resolution transaction")?;
        let mut markets = sqlx::query_as::<_, MarketBoundaryRow>(
            r#"
            SELECT market_id, window_start, window_end, up_token_id, down_token_id,
              reference_price, reference_source_timestamp, resolution_price,
              resolution_source_timestamp, official_outcome, official_resolved_at,
              official_winning_token_id
            FROM polymarket.btc_interval_markets
            WHERE market_id = $1 OR condition_id = $1
            ORDER BY market_id
            FOR UPDATE
            "#,
        )
        .bind(market_or_condition_id)
        .fetch_all(&mut *tx)
        .await
        .context("failed to locate officially resolved BTC market")?;
        if markets.len() != 1 {
            bail!(
                "official BTC resolution identity {} matched {} markets",
                market_or_condition_id,
                markets.len()
            );
        }
        let market = markets.pop().expect("one exact official market match");
        if source_timestamp < market.window_end || received_at < market.window_end {
            bail!("official BTC resolution predates market close");
        }
        let official = official_outcome_for_winner(
            &market.up_token_id,
            &market.down_token_id,
            winning_token_id,
            winning_outcome,
        )?;
        let official_name = outcome_name(official);
        let newly_recorded = market.official_outcome.is_none();
        if (market.official_outcome.is_some()
            || market.official_resolved_at.is_some()
            || market.official_winning_token_id.is_some())
            && (market.official_outcome.as_deref() != Some(official_name)
                || market.official_winning_token_id.as_deref() != Some(winning_token_id))
        {
            bail!(
                "immutable official BTC resolution conflict for market {}",
                market.market_id
            );
        }

        sqlx::query(
            r#"
            UPDATE polymarket.btc_interval_markets
            SET official_outcome = COALESCE(official_outcome, $2),
                official_resolved_at = COALESCE(official_resolved_at, $3),
                official_winning_token_id = COALESCE(official_winning_token_id, $4),
                official_resolution_source = COALESCE(official_resolution_source, $5),
                official_resolution_received_at = COALESCE(official_resolution_received_at, $6),
                official_resolution_payload = COALESCE(official_resolution_payload, $7),
                resolved_outcome = $2,
                updated_at = now()
            WHERE market_id = $1
            "#,
        )
        .bind(&market.market_id)
        .bind(official_name)
        .bind(source_timestamp)
        .bind(winning_token_id)
        .bind(resolution_source)
        .bind(received_at)
        .bind(payload)
        .execute(&mut *tx)
        .await
        .context("failed to persist official BTC market resolution")?;
        let label_conflict = sqlx::query_scalar::<_, bool>(
            r#"
            SELECT EXISTS (
              SELECT 1
              FROM polymarket.btc_market_labels
              WHERE market_id = $1
                AND official_outcome IS NOT NULL
                AND official_outcome <> $2
            )
            "#,
        )
        .bind(&market.market_id)
        .bind(official_name)
        .fetch_one(&mut *tx)
        .await
        .context("failed to validate official BTC label projection")?;
        if label_conflict {
            bail!(
                "immutable official BTC label conflict for market {}",
                market.market_id
            );
        }
        sqlx::query(
            r#"
            UPDATE polymarket.btc_market_labels
            SET official_outcome = COALESCE(official_outcome, $2),
                official_resolved_at = COALESCE(official_resolved_at, $3),
                updated_at = now()
            WHERE market_id = $1
              AND (official_outcome IS NULL OR official_outcome = $2)
            "#,
        )
        .bind(&market.market_id)
        .bind(official_name)
        .bind(source_timestamp)
        .execute(&mut *tx)
        .await
        .context("failed to project official BTC resolution onto its local label")?;
        sqlx::query(
            r#"
            UPDATE polymarket.btc_official_resolution_watches
            SET status = CASE
                  WHEN status IN ('resolved','resolved_late') THEN status
                  WHEN status = 'expired' OR $2 > deadline_at THEN 'resolved_late'
                  ELSE 'resolved'
                END,
                resolution_received_at = COALESCE(resolution_received_at, $2),
                resolution_source = COALESCE(resolution_source, $3),
                expired_at = CASE
                  WHEN status = 'resolved' THEN expired_at
                  WHEN status = 'resolved_late' THEN COALESCE(expired_at, deadline_at)
                  WHEN status = 'expired' OR $2 > deadline_at
                    THEN COALESCE(expired_at, deadline_at)
                  ELSE expired_at
                END,
                last_checked_at = COALESCE(last_checked_at, $2),
                last_error = NULL,
                updated_at = now()
            WHERE market_id = $1
            "#,
        )
        .bind(&market.market_id)
        .bind(received_at)
        .bind(resolution_source)
        .execute(&mut *tx)
        .await
        .context("failed to resolve durable BTC official-resolution watch")?;
        tx.commit()
            .await
            .context("failed to commit official BTC resolution transaction")?;
        Ok(PersistedOfficialResolution {
            market_id: market.market_id,
            winning_token_id: winning_token_id.to_string(),
            newly_recorded,
        })
    }

    /// Loads only information whose exchange event time and local receive time are no later than
    /// `as_of`. This is the experiment runner's no-lookahead boundary.
    pub async fn load_point_in_time_inputs(
        &self,
        market: &BtcIntervalMarket,
        as_of: DateTime<Utc>,
        max_chainlink_open_delay: chrono::Duration,
        max_reference_age: chrono::Duration,
        max_book_age: chrono::Duration,
        clob_connection_id: Option<Uuid>,
    ) -> Result<BtcPointInTimeInputs> {
        if max_chainlink_open_delay <= Duration::zero()
            || max_reference_age <= Duration::zero()
            || max_book_age <= Duration::zero()
        {
            bail!("point-in-time input freshness bounds must be positive");
        }
        let reference_fresh_since = as_of
            .checked_sub_signed(max_reference_age)
            .context("point-in-time reference freshness bound is outside the timestamp range")?;
        let book_fresh_since = as_of
            .checked_sub_signed(max_book_age)
            .context("point-in-time book freshness bound is outside the timestamp range")?;
        let latest_valid_open = market
            .window_start
            .checked_add_signed(max_chainlink_open_delay)
            .context("Chainlink opening window exceeds the supported timestamp range")?;
        let chainlink_open = sqlx::query_as::<_, ReferenceTickRow>(
            r#"
            SELECT tick_id, source_timestamp, received_at, source, symbol, price,
              envelope_timestamp, connection_id, ingest_sequence, source_event_id,
              dedup_key, raw_payload
            FROM polymarket.reference_price_ticks
            WHERE source = 'rtds_chainlink'
              AND symbol = 'BTCUSD'
              AND integrity_status = 'ok'
              AND source_timestamp >= $1
              AND source_timestamp <= $2
              AND received_at <= $3
            ORDER BY source_timestamp ASC, received_at ASC, ingest_sequence ASC, tick_id ASC
            LIMIT 1
            "#,
        )
        .bind(market.window_start)
        .bind(latest_valid_open.min(as_of))
        .bind(as_of)
        .fetch_optional(&self.pool)
        .await
        .context("failed to load point-in-time BTC Chainlink open")?
        .map(reference_tick_from_row)
        .transpose()?;
        let chainlink_current = sqlx::query_as::<_, ReferenceTickRow>(
            r#"
            SELECT tick_id, source_timestamp, received_at, source, symbol, price,
              envelope_timestamp, connection_id, ingest_sequence, source_event_id,
              dedup_key, raw_payload
            FROM polymarket.reference_price_ticks
            WHERE source = 'rtds_chainlink'
              AND symbol = 'BTCUSD'
              AND integrity_status = 'ok'
              AND source_timestamp >= $1
              AND source_timestamp <= $2
              AND received_at >= $1
              AND received_at <= $2
            ORDER BY source_timestamp DESC, received_at DESC, ingest_sequence DESC, tick_id DESC
            LIMIT 1
            "#,
        )
        .bind(reference_fresh_since)
        .bind(as_of)
        .fetch_optional(&self.pool)
        .await
        .context("failed to load point-in-time BTC Chainlink tick")?
        .map(reference_tick_from_row)
        .transpose()?;
        let mut chainlink_history = sqlx::query_as::<_, ReferenceTickRow>(
            r#"
            SELECT tick_id, source_timestamp, received_at, source, symbol, price,
              envelope_timestamp, connection_id, ingest_sequence, source_event_id,
              dedup_key, raw_payload
            FROM polymarket.reference_price_ticks
            WHERE source = 'rtds_chainlink'
              AND symbol = 'BTCUSD'
              AND integrity_status = 'ok'
              AND source_timestamp >= $1
              AND source_timestamp <= $2
              AND received_at <= $2
            ORDER BY source_timestamp DESC, received_at DESC, ingest_sequence DESC, tick_id DESC
            LIMIT 2000
            "#,
        )
        .bind(as_of - chrono::Duration::seconds(35))
        .bind(as_of)
        .fetch_all(&self.pool)
        .await
        .context("failed to load point-in-time Chainlink history")?
        .into_iter()
        .map(reference_tick_from_row)
        .collect::<Result<Vec<_>>>()?;
        chainlink_history.reverse();
        truncate_history_at_current(&mut chainlink_history, chainlink_current.as_ref());
        let mut binance_history = sqlx::query_as::<_, ReferenceTickRow>(
            r#"
            SELECT tick_id, source_timestamp, received_at, source, symbol, price,
              envelope_timestamp, connection_id, ingest_sequence, source_event_id,
              dedup_key, raw_payload
            FROM polymarket.reference_price_ticks
            WHERE source = 'direct_binance'
              AND symbol = 'BTCUSD'
              AND integrity_status = 'ok'
              AND source_timestamp >= $1
              AND source_timestamp <= $2
              AND received_at <= $2
            ORDER BY source_timestamp DESC, received_at DESC, ingest_sequence DESC, tick_id DESC
            LIMIT 20000
            "#,
        )
        .bind(as_of - chrono::Duration::seconds(35))
        .bind(as_of)
        .fetch_all(&self.pool)
        .await
        .context("failed to load point-in-time Binance history")?
        .into_iter()
        .map(reference_tick_from_row)
        .collect::<Result<Vec<_>>>()?;
        binance_history.reverse();
        let (up_book, down_book) = match clob_connection_id {
            Some(connection_id) => tokio::try_join!(
                self.load_checkpoint_as_of(
                    &market.up_token_id,
                    connection_id,
                    book_fresh_since,
                    as_of,
                ),
                self.load_checkpoint_as_of(
                    &market.down_token_id,
                    connection_id,
                    book_fresh_since,
                    as_of,
                ),
            )?,
            None => (None, None),
        };
        let fee = sqlx::query_as::<_, (Option<Decimal>, DateTime<Utc>)>(
            r#"
            SELECT fee_rate, last_refreshed_at
            FROM polymarket.btc_interval_markets
            WHERE market_id = $1
              AND last_refreshed_at <= $2
            "#,
        )
        .bind(&market.market_id)
        .bind(as_of)
        .fetch_optional(&self.pool)
        .await
        .context("failed to load BTC market fee schedule")?;

        Ok(BtcPointInTimeInputs {
            chainlink_open,
            chainlink_current,
            chainlink_history,
            binance_history,
            up_book,
            down_book,
            fee_rate: fee.as_ref().and_then(|(rate, _)| *rate),
            fee_observed_at: fee.map(|(_, observed_at)| observed_at),
        })
    }

    async fn load_checkpoint_as_of(
        &self,
        token_id: &str,
        connection_id: Uuid,
        fresh_since: DateTime<Utc>,
        as_of: DateTime<Utc>,
    ) -> Result<Option<OrderbookCheckpoint>> {
        sqlx::query_as::<_, CheckpointRow>(
            r#"
            SELECT checkpoint_id, source_timestamp, received_at, connection_id,
              ingest_sequence, market_id, token_id, best_bid, best_ask, tick_size,
              book, source_hash, integrity_status
            FROM polymarket.orderbook_checkpoints
            WHERE token_id = $1
              AND source_timestamp >= $2
              AND source_timestamp <= $3
              AND received_at >= $2
              AND received_at <= $3
              AND connection_id = $4
              AND integrity_status = 'ok'
            ORDER BY source_timestamp DESC, received_at DESC, ingest_sequence DESC,
              checkpoint_id DESC
            LIMIT 1
            "#,
        )
        .bind(token_id)
        .bind(fresh_since)
        .bind(as_of)
        .bind(connection_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to load point-in-time BTC orderbook checkpoint")?
        .map(checkpoint_from_row)
        .transpose()
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn ensure_paper_experiment(
        &self,
        experiment_id: Uuid,
        name: &str,
        process_id: Uuid,
        strategy_version: &str,
        feature_schema_version: &str,
        config_hash: &str,
        config: &serde_json::Value,
    ) -> Result<()> {
        let result = sqlx::query(
            r#"
            INSERT INTO polymarket.btc_paper_experiments (
              experiment_id, name, status, strategy_version, feature_schema_version,
              config_hash, config, process_id, started_at
            )
            VALUES ($1,$2,'running',$3,$4,$5,$6,$7,now())
            ON CONFLICT (experiment_id) DO NOTHING
            "#,
        )
        .bind(experiment_id)
        .bind(name)
        .bind(strategy_version)
        .bind(feature_schema_version)
        .bind(config_hash)
        .bind(config)
        .bind(process_id)
        .execute(&self.pool)
        .await
        .context("failed to ensure BTC paper experiment")?;
        if result.rows_affected() != 1 {
            anyhow::bail!(
                "BTC experiment identity already exists; new explicit cohort starts require a new experiment key"
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn verify_resumable_paper_experiment(
        &self,
        experiment_id: Uuid,
        name: &str,
        process_id: Uuid,
        strategy_version: &str,
        feature_schema_version: &str,
        config_hash: &str,
        config: &serde_json::Value,
    ) -> Result<()> {
        let matches = sqlx::query_scalar::<_, bool>(
            r#"
            SELECT EXISTS (
              SELECT 1
              FROM polymarket.btc_paper_experiments
              WHERE experiment_id = $1
                AND name = $2
                AND process_id = $3
                AND strategy_version = $4
                AND feature_schema_version = $5
                AND config_hash = $6
                AND config = $7
                AND status = 'running'
                AND stopped_at IS NULL
            )
            "#,
        )
        .bind(experiment_id)
        .bind(name)
        .bind(process_id)
        .bind(strategy_version)
        .bind(feature_schema_version)
        .bind(config_hash)
        .bind(config)
        .fetch_one(&self.pool)
        .await
        .context("failed to verify resumable BTC paper experiment")?;
        if !matches {
            anyhow::bail!(
                "BTC paper experiment cannot resume because its running identity or frozen config no longer matches"
            );
        }
        Ok(())
    }

    pub async fn paper_venue_resume_state(
        &self,
        experiment_id: Uuid,
    ) -> Result<BtcPaperVenueResumeState> {
        let (entry_debits, settlement_credits, order_count, fill_count, settlement_ids) =
            sqlx::query_as::<_, (Decimal, Decimal, i64, i64, Vec<Uuid>)>(
                r#"
                WITH experiment_orders AS (
                  SELECT o.order_id
                  FROM polymarket.btc_paper_experiments e
                  JOIN polymarket.orders o
                    ON o.process_id = e.process_id
                   AND o.raw_payload #>> '{request,metadata,experiment_id}' = e.experiment_id::text
                  WHERE e.experiment_id = $1
                ), fill_totals AS (
                  SELECT
                    COALESCE(SUM(f.price * f.size + f.fee), 0)::numeric AS entry_debits,
                    COUNT(f.fill_id)::bigint AS fill_count
                  FROM experiment_orders o
                  LEFT JOIN polymarket.fills f
                    ON f.order_id = o.order_id
                   AND f.source = 'paper'
                ), order_totals AS (
                  SELECT COUNT(*)::bigint AS order_count FROM experiment_orders
                ), settlement_totals AS (
                  SELECT
                    COALESCE(SUM(payout) FILTER (WHERE credit_status = 'credited'), 0)::numeric
                      AS settlement_credits,
                    COALESCE(
                      ARRAY_AGG(settlement_id ORDER BY settlement_id)
                        FILTER (WHERE credit_status = 'credited'),
                      ARRAY[]::uuid[]
                    ) AS settlement_ids
                  FROM polymarket.btc_paper_settlement_ledger
                  WHERE experiment_id = $1
                )
                SELECT f.entry_debits, s.settlement_credits,
                       o.order_count, f.fill_count, s.settlement_ids
                FROM fill_totals f CROSS JOIN order_totals o CROSS JOIN settlement_totals s
                "#,
            )
            .bind(experiment_id)
            .fetch_one(&self.pool)
            .await
            .context("failed to load BTC paper venue resume state")?;
        Ok(BtcPaperVenueResumeState {
            entry_debits_usd: entry_debits,
            settlement_credits_usd: settlement_credits,
            order_count: usize::try_from(order_count)
                .context("BTC paper resume order count exceeded usize")?,
            fill_count: usize::try_from(fill_count)
                .context("BTC paper resume fill count exceeded usize")?,
            credited_settlement_ids: settlement_ids,
        })
    }

    pub async fn mark_paper_experiment_terminal(
        &self,
        experiment_id: Uuid,
        status: &str,
        reason: &str,
    ) -> Result<()> {
        if !matches!(status, "stopped" | "failed") || reason.trim().is_empty() {
            anyhow::bail!("invalid BTC experiment terminal status or reason");
        }
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin BTC experiment terminal transaction")?;
        sqlx::query(
            r#"
            UPDATE polymarket.btc_paper_experiments
            SET status = $2, stopped_at = now(), stop_reason = $3, updated_at = now()
            WHERE experiment_id = $1 AND status = 'running'
            "#,
        )
        .bind(experiment_id)
        .bind(status)
        .bind(reason)
        .execute(&mut *tx)
        .await
        .context("failed to mark BTC paper experiment terminal")?;
        sqlx::query(
            r#"
            UPDATE polymarket.trading_processes
            SET status = $2,
                enabled = false,
                stopped_at = now(),
                last_error = CASE WHEN $2 = 'failed' THEN $3 ELSE last_error END,
                updated_at = now()
            WHERE process_id = (
              SELECT process_id
              FROM polymarket.btc_paper_experiments
              WHERE experiment_id = $1
            )
              AND status = 'running'
            "#,
        )
        .bind(experiment_id)
        .bind(status)
        .bind(reason)
        .execute(&mut *tx)
        .await
        .context("failed to mark BTC trading process terminal")?;
        tx.commit()
            .await
            .context("failed to commit BTC experiment terminal transaction")?;
        Ok(())
    }

    pub async fn process_has_entry(&self, process_id: Uuid, market_id: &str) -> Result<bool> {
        sqlx::query_scalar::<_, bool>(PROCESS_HAS_ENTRY_SQL)
            .bind(process_id)
            .bind(market_id)
            .fetch_one(&self.pool)
            .await
            .context("failed to check existing BTC process entry")
    }

    pub async fn load_resolved_loss_regime_candidates(
        &self,
        process_id: Uuid,
        config_hash: &str,
        as_of: DateTime<Utc>,
    ) -> Result<Vec<LossRegimeCandidate>> {
        let rows = sqlx::query_as::<_, LossRegimeCandidateRow>(
            r#"
            WITH canonical AS (
              SELECT DISTINCT ON (d.market_id)
                d.market_id,
                d.decision_id,
                d.outcome AS decision_outcome,
                l.outcome AS resolved_outcome,
                d.decision_at,
                l.label_available_at
              FROM polymarket.btc_strategy_decisions d
              JOIN polymarket.btc_market_labels l
                ON l.market_id = d.market_id
              WHERE d.process_id = $1
                AND d.config_hash = $2
                AND d.action = 'buy'
                AND d.outcome IS NOT NULL
                AND d.decision_at < l.label_available_at
                AND l.label_available_at <= $3
              ORDER BY d.market_id, d.decision_at, d.decision_id
            )
            SELECT market_id, decision_id, decision_outcome, resolved_outcome,
              decision_at, label_available_at
            FROM canonical
            ORDER BY label_available_at, decision_at, decision_id, market_id
            "#,
        )
        .bind(process_id)
        .bind(config_hash)
        .bind(as_of)
        .fetch_all(&self.pool)
        .await
        .context("failed to load resolved loss-regime candidates")?;

        rows.into_iter()
            .map(|row| {
                Ok(LossRegimeCandidate {
                    market_id: row.market_id,
                    decision_id: row.decision_id,
                    decision_outcome: parse_outcome_name(&row.decision_outcome)?,
                    resolved_outcome: parse_outcome_name(&row.resolved_outcome)?,
                    decision_at: row.decision_at,
                    label_available_at: row.label_available_at,
                })
            })
            .collect()
    }

    /// Loads the newest bounded slice of causally resolved entry predictions for the shadow
    /// predictive-regime evaluator. The first buy decision for each process-owned market is the
    /// canonical candidate, independent of downstream order or fill status.
    pub async fn load_shadow_predictive_regime_candidates(
        &self,
        process_id: Uuid,
        config_hash: Option<&str>,
        as_of: DateTime<Utc>,
        max_candidates: u32,
    ) -> Result<Vec<ShadowPredictiveRegimeCandidate>> {
        let limit = shadow_predictive_regime_history_limit(max_candidates)?;
        let rows = sqlx::query_as::<_, ShadowPredictiveRegimeCandidateRow>(
            SHADOW_PREDICTIVE_REGIME_CANDIDATES_SQL,
        )
        .bind(process_id)
        .bind(config_hash)
        .bind(as_of)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .context("failed to load shadow predictive-regime candidates")?;

        rows.into_iter()
            .map(shadow_predictive_regime_candidate_from_row)
            .collect()
    }

    /// Loads the newest causally persisted shadow predictive-regime state from this process's
    /// decision evidence. The optional hash is the breaker's state-config hash, not the enclosing
    /// process-config hash. Execution status and experiment identity do not own this state.
    pub async fn load_latest_shadow_predictive_regime_state(
        &self,
        process_id: Uuid,
        breaker_state_config_hash: Option<&str>,
        as_of: DateTime<Utc>,
    ) -> Result<Option<ShadowPredictiveRegimeState>> {
        if process_id.is_nil() {
            bail!("shadow predictive-regime state process_id cannot be nil");
        }
        if breaker_state_config_hash.is_some_and(|config_hash| config_hash.trim().is_empty()) {
            bail!("shadow predictive-regime state config hash cannot be empty");
        }
        let row = sqlx::query_as::<_, PersistedShadowPredictiveRegimeEvaluationRow>(
            LATEST_SHADOW_PREDICTIVE_REGIME_STATE_SQL,
        )
        .bind(process_id)
        .bind(breaker_state_config_hash)
        .bind(as_of)
        .fetch_optional(&self.pool)
        .await
        .context("failed to load latest shadow predictive-regime state")?;

        row.map(|row| {
            shadow_predictive_regime_state_from_persisted_evaluation(
                process_id,
                breaker_state_config_hash,
                row,
            )
        })
        .transpose()
    }

    /// Reconstructs the causal UTC-day realized-PnL watermark and every still-unsettled paper
    /// entry for one stable trading process. This state is deliberately process-owned: it carries
    /// across immutable experiment runs and must never be filtered by experiment identity.
    pub async fn load_daily_realized_pnl_high_water_mark_state(
        &self,
        process_id: Uuid,
        as_of: DateTime<Utc>,
    ) -> Result<DailyRealizedPnlHighWaterMarkState> {
        let period_start_utc = as_of
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .context("failed to derive UTC high-water-mark query period")?
            .and_utc();
        let period_end_utc = period_start_utc + Duration::days(1);
        let rows =
            sqlx::query_as::<_, DailyHighWaterMarkEvidenceRow>(DAILY_HIGH_WATER_MARK_EVIDENCE_SQL)
                .bind(process_id)
                .bind(as_of)
                .bind(period_start_utc)
                .bind(period_end_utc)
                .fetch_all(&self.pool)
                .await
                .context("failed to load process-owned daily high-water-mark evidence")?;

        daily_high_water_mark_state_from_rows(process_id, as_of, rows)
    }

    pub async fn insert_feature_snapshot(
        &self,
        snapshot: &BtcFeatureSnapshot,
        fair_value: Option<&FairValueEstimate>,
        feature_hash: &str,
        readiness_status: &str,
        quality_flags: &serde_json::Value,
    ) -> Result<bool> {
        let binance_return_1s_bps =
            snapshot.binance_return_1s.unwrap_or_default() * Decimal::from(10_000);
        let binance_return_5s_bps =
            snapshot.binance_return_5s.unwrap_or_default() * Decimal::from(10_000);
        let binance_return_30s_bps =
            snapshot.binance_return_30s.unwrap_or_default() * Decimal::from(10_000);
        let realized_vol_bps =
            snapshot.realized_volatility.unwrap_or_default() * Decimal::from(10_000);
        let seconds_to_close =
            Decimal::from((snapshot.window_end - snapshot.observed_at).num_milliseconds())
                / Decimal::from(1_000);
        let inserted = sqlx::query(
            r#"
            INSERT INTO polymarket.btc_feature_snapshots (
              snapshot_id, feature_as_of, received_at, market_id, window_start, window_end,
              feature_schema_version, feature_hash, chainlink_price, chainlink_open_price,
              binance_price, seconds_to_close, chainlink_gap_bps, binance_return_1s_bps,
              binance_return_5s_bps, binance_return_30s_bps, realized_vol_30s_bps,
              basis_bps, up_best_bid, up_best_ask, down_best_bid, down_best_ask,
              up_depth_ask, down_depth_ask, up_imbalance, down_imbalance, chainlink_age_ms,
              binance_age_ms, book_age_ms, source_skew_ms, fair_up_probability,
              fair_up_lower, fair_up_upper, deterministic_logit, readiness_status,
              quality_flags, features, lineage
            )
            VALUES (
              $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,
              $21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34,$35,$36,$37,$38
            )
            ON CONFLICT (snapshot_id, feature_as_of) DO NOTHING
            "#,
        )
        .bind(snapshot.snapshot_id)
        .bind(snapshot.observed_at)
        .bind(snapshot.observed_at)
        .bind(&snapshot.market_id)
        .bind(snapshot.window_start)
        .bind(snapshot.window_end)
        .bind(&snapshot.feature_schema_version)
        .bind(feature_hash)
        .bind(snapshot.chainlink_price.unwrap_or_default())
        .bind(snapshot.chainlink_open_price.unwrap_or_default())
        .bind(snapshot.binance_price.unwrap_or_default())
        .bind(seconds_to_close)
        .bind(snapshot.chainlink_gap_bps.unwrap_or_default())
        .bind(binance_return_1s_bps)
        .bind(binance_return_5s_bps)
        .bind(binance_return_30s_bps)
        .bind(realized_vol_bps)
        .bind(snapshot.binance_chainlink_basis_bps.unwrap_or_default())
        .bind(snapshot.up_book.best_bid)
        .bind(snapshot.up_book.best_ask)
        .bind(snapshot.down_book.best_bid)
        .bind(snapshot.down_book.best_ask)
        .bind(snapshot.up_book.ask_depth)
        .bind(snapshot.down_book.ask_depth)
        .bind(snapshot.up_book.imbalance)
        .bind(snapshot.down_book.imbalance)
        .bind(snapshot.chainlink_age_ms.unwrap_or_default())
        .bind(snapshot.binance_age_ms.unwrap_or_default())
        .bind(
            snapshot
                .up_book
                .age_ms
                .unwrap_or_default()
                .max(snapshot.down_book.age_ms.unwrap_or_default()),
        )
        .bind(snapshot.source_skew_ms.unwrap_or_default())
        .bind(fair_value.map(|value| value.up_probability))
        .bind(fair_value.map(|value| value.up_lower_bound))
        .bind(fair_value.map(|value| value.up_upper_bound))
        .bind(fair_value.map(|value| value.z_score))
        .bind(readiness_status)
        .bind(quality_flags)
        .bind(serde_json::to_value(snapshot)?)
        .bind(serde_json::to_value(&snapshot.lineage)?)
        .execute(&self.pool)
        .await
        .context("failed to insert BTC feature snapshot")?;
        Ok(inserted.rows_affected() == 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn insert_strategy_decision(
        &self,
        experiment_id: Uuid,
        config_hash: &str,
        market_id: &str,
        strategy_version: &str,
        decision: &BtcDecision,
        entry_admission_evidence: Option<&serde_json::Value>,
        order_plan_id: Option<Uuid>,
        status: &str,
    ) -> Result<bool> {
        let edge = decision_edge_projection(decision)?;
        let (action, outcome) = match decision.action {
            BtcDecisionAction::BuyUp => ("buy", Some("up")),
            BtcDecisionAction::BuyDown => ("buy", Some("down")),
            BtcDecisionAction::NoTrade => ("no_trade", None),
        };
        let mut metadata = serde_json::to_value(decision)?;
        if let Some(entry_admission_evidence) = entry_admission_evidence {
            metadata
                .as_object_mut()
                .context("serialized BTC decision metadata must be an object")?
                .insert(
                    "entry_admission".to_string(),
                    entry_admission_evidence.clone(),
                );
        }
        let inserted = sqlx::query(
            r#"
            INSERT INTO polymarket.btc_strategy_decisions (
              decision_id, decision_at, experiment_id, process_id, market_id, snapshot_id,
              strategy_version, config_hash, action, outcome, token_id, fair_probability,
              executable_price, gross_edge_per_share, fee_per_share, reserve_per_share,
              net_edge_per_share, size, status, reject_reason, order_plan_id, execution_mode,
              metadata
            )
            VALUES (
              $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,
              $21,'paper',$22
            )
            ON CONFLICT (decision_id, decision_at) DO NOTHING
            "#,
        )
        .bind(decision.decision_id)
        .bind(decision.evaluated_at)
        .bind(experiment_id)
        .bind(decision.process_id)
        .bind(market_id)
        .bind(decision.feature_snapshot_id)
        .bind(strategy_version)
        .bind(config_hash)
        .bind(action)
        .bind(outcome)
        .bind(edge.token_id)
        .bind(edge.fair_probability)
        .bind(edge.executable_price)
        .bind(edge.gross_edge_per_share)
        .bind(edge.fee_per_share)
        .bind(edge.reserve_per_share)
        .bind(edge.net_edge_per_share)
        .bind(edge.size)
        .bind(status)
        .bind(decision.reject_reason.map(|reason| reason.as_str()))
        .bind(order_plan_id)
        .bind(metadata)
        .execute(&self.pool)
        .await
        .context("failed to insert BTC strategy decision")?;
        Ok(inserted.rows_affected() == 1)
    }

    pub async fn update_strategy_decision_execution(
        &self,
        decision_id: Uuid,
        decision_at: DateTime<Utc>,
        status: &str,
        reject_reason: Option<&str>,
        metadata: serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE polymarket.btc_strategy_decisions
            SET status = $3,
                reject_reason = COALESCE($4, reject_reason),
                metadata = metadata || $5
            WHERE decision_id = $1 AND decision_at = $2
            "#,
        )
        .bind(decision_id)
        .bind(decision_at)
        .bind(status)
        .bind(reject_reason)
        .bind(metadata)
        .execute(&self.pool)
        .await
        .context("failed to update BTC decision execution")?;
        Ok(())
    }

    pub async fn update_paper_capital_runtime_summary<T: Serialize + ?Sized>(
        &self,
        experiment_id: Uuid,
        status: &T,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE polymarket.btc_paper_experiments
            SET summary = jsonb_set(
                  COALESCE(summary, '{}'::jsonb),
                  '{paper_capital}',
                  $2::jsonb,
                  true
                ),
                updated_at = now()
            WHERE experiment_id = $1
            "#,
        )
        .bind(experiment_id)
        .bind(serde_json::to_value(status)?)
        .execute(&self.pool)
        .await
        .context("failed to update durable paper-capital runtime summary")?;
        Ok(())
    }

    pub async fn increment_experiment_counts(
        &self,
        experiment_id: Uuid,
        snapshots: i64,
        decisions: i64,
        trades: i64,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE polymarket.btc_paper_experiments
            SET markets_observed = (
                  SELECT count(DISTINCT market_id)::bigint
                  FROM polymarket.btc_strategy_decisions
                  WHERE experiment_id = $1
                ),
                snapshots_recorded = snapshots_recorded + $2,
                decisions_recorded = decisions_recorded + $3,
                trades_entered = trades_entered + $4,
                updated_at = now()
            WHERE experiment_id = $1
            "#,
        )
        .bind(experiment_id)
        .bind(snapshots)
        .bind(decisions)
        .bind(trades)
        .execute(&self.pool)
        .await
        .context("failed to update BTC experiment counters")?;
        sqlx::query(
            r#"
            UPDATE polymarket.trading_processes
            SET heartbeat_at = now(), updated_at = now()
            WHERE process_id = (
              SELECT process_id
              FROM polymarket.btc_paper_experiments
              WHERE experiment_id = $1
            )
            "#,
        )
        .bind(experiment_id)
        .execute(&self.pool)
        .await
        .context("failed to heartbeat BTC trading process")?;
        Ok(())
    }

    pub async fn paper_experiment_status(
        &self,
        experiment_id: Uuid,
    ) -> Result<Option<BtcPaperExperimentStatus>> {
        sqlx::query_as::<_, BtcPaperExperimentStatus>(
            r#"
            SELECT experiment_id, name, status, started_at, stopped_at, markets_observed,
              snapshots_recorded, decisions_recorded, trades_entered, trades_resolved,
              gross_pnl, fees_paid, net_pnl, summary, updated_at
            FROM polymarket.btc_paper_experiments
            WHERE experiment_id = $1
            "#,
        )
        .bind(experiment_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to fetch BTC paper experiment status")
    }

    /// Materializes every newly eligible official settlement into a durable, idempotent ledger
    /// and returns credits still awaiting application to the in-memory paper venue. Eligibility
    /// requires the immutable official market fact and its matching durable resolution watch.
    pub async fn discover_pending_paper_settlements(
        &self,
        experiment_id: Uuid,
    ) -> Result<Vec<BtcPaperSettlementRecord>> {
        sqlx::query(
            r#"
            WITH entered AS (
              SELECT
                e.experiment_id,
                e.process_id,
                o.order_id,
                o.market_id,
                o.token_id,
                jsonb_agg(to_jsonb(f.fill_id) ORDER BY f.timestamp_utc, f.fill_id) AS fill_ids,
                round(sum(f.size), 10)::numeric(30,10) AS filled_size,
                round(sum(f.price * f.size), 10)::numeric(30,10) AS entry_notional,
                round(sum(f.fee), 10)::numeric(30,10) AS entry_fees
              FROM polymarket.btc_paper_experiments e
              JOIN polymarket.orders o
                ON o.process_id = e.process_id
               AND o.raw_payload #>> '{request,metadata,experiment_id}' = e.experiment_id::text
              JOIN polymarket.fills f
                ON f.order_id = o.order_id
               AND f.source = 'paper'
               AND f.process_id = e.process_id
              WHERE e.experiment_id = $1
              GROUP BY e.experiment_id, e.process_id, o.order_id, o.market_id, o.token_id
            ),
            eligible AS (
              SELECT
                e.*,
                m.official_outcome,
                m.official_winning_token_id,
                m.official_resolution_received_at,
                m.official_resolution_source,
                CASE
                  WHEN e.token_id = m.official_winning_token_id THEN e.filled_size
                  ELSE 0::numeric(30,10)
                END::numeric(30,10) AS payout
              FROM entered e
              JOIN polymarket.btc_interval_markets m ON m.market_id = e.market_id
              JOIN polymarket.btc_official_resolution_watches w ON w.market_id = m.market_id
              WHERE m.official_outcome IS NOT NULL
                AND m.official_winning_token_id IS NOT NULL
                AND m.official_resolution_received_at IS NOT NULL
                AND m.official_resolution_source IS NOT NULL
                AND (
                  (m.official_outcome = 'up' AND m.official_winning_token_id = m.up_token_id)
                  OR
                  (m.official_outcome = 'down' AND m.official_winning_token_id = m.down_token_id)
                )
                AND w.status IN ('resolved', 'resolved_late')
                AND w.resolution_received_at = m.official_resolution_received_at
                AND w.resolution_source = m.official_resolution_source
            )
            INSERT INTO polymarket.btc_paper_settlement_ledger (
              experiment_id, process_id, order_id, market_id, token_id, fill_ids,
              official_outcome, official_winning_token_id,
              official_resolution_received_at, official_resolution_source,
              filled_size, entry_notional, entry_fees, payout, net_pnl
            )
            SELECT
              experiment_id, process_id, order_id, market_id, token_id, fill_ids,
              official_outcome, official_winning_token_id,
              official_resolution_received_at, official_resolution_source,
              filled_size, entry_notional, entry_fees, payout,
              (payout - entry_notional - entry_fees)::numeric(30,10)
            FROM eligible
            ON CONFLICT (experiment_id, order_id) DO NOTHING
            "#,
        )
        .bind(experiment_id)
        .execute(&self.pool)
        .await
        .context("failed to discover durable BTC paper settlements")?;

        let records = sqlx::query_as::<_, BtcPaperSettlementRecord>(
            r#"
            SELECT settlement_id, experiment_id, process_id, order_id, market_id, token_id,
              fill_ids,
              official_outcome, official_winning_token_id,
              official_resolution_received_at, official_resolution_source,
              filled_size, entry_notional, entry_fees, payout, net_pnl,
              credit_status, credited_at, credit_attempts, credit_evidence,
              created_at, updated_at
            FROM polymarket.btc_paper_settlement_ledger
            WHERE experiment_id = $1 AND credit_status = 'pending'
            ORDER BY official_resolution_received_at, order_id, settlement_id
            "#,
        )
        .bind(experiment_id)
        .fetch_all(&self.pool)
        .await
        .context("failed to load pending BTC paper settlement credits")?;
        for record in &records {
            validate_paper_settlement_record(record)?;
        }
        Ok(records)
    }

    pub async fn mark_paper_settlement_credited(
        &self,
        experiment_id: Uuid,
        settlement_id: Uuid,
        credit_evidence: &serde_json::Value,
    ) -> Result<bool> {
        if !credit_evidence.is_object() {
            bail!("paper settlement credit evidence must be a JSON object");
        }
        let result = sqlx::query(
            r#"
            UPDATE polymarket.btc_paper_settlement_ledger
            SET credit_status = 'credited',
                credited_at = now(),
                credit_attempts = credit_attempts + 1,
                credit_evidence = $3,
                updated_at = now()
            WHERE experiment_id = $1
              AND settlement_id = $2
              AND credit_status = 'pending'
            "#,
        )
        .bind(experiment_id)
        .bind(settlement_id)
        .bind(credit_evidence)
        .execute(&self.pool)
        .await
        .context("failed to mark BTC paper settlement credited")?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn list_paper_settlements(
        &self,
        experiment_id: Uuid,
    ) -> Result<Vec<BtcPaperSettlementRecord>> {
        sqlx::query_as::<_, BtcPaperSettlementRecord>(
            r#"
            SELECT settlement_id, experiment_id, process_id, order_id, market_id, token_id,
              fill_ids,
              official_outcome, official_winning_token_id,
              official_resolution_received_at, official_resolution_source,
              filled_size, entry_notional, entry_fees, payout, net_pnl,
              credit_status, credited_at, credit_attempts, credit_evidence,
              created_at, updated_at
            FROM polymarket.btc_paper_settlement_ledger
            WHERE experiment_id = $1
            ORDER BY official_resolution_received_at, order_id, settlement_id
            "#,
        )
        .bind(experiment_id)
        .fetch_all(&self.pool)
        .await
        .context("failed to query BTC paper settlement evidence")
    }

    pub async fn paper_settlement_ledger_summary(
        &self,
        experiment_id: Uuid,
    ) -> Result<BtcPaperSettlementLedgerSummary> {
        sqlx::query_as::<_, BtcPaperSettlementLedgerSummary>(
            r#"
            SELECT
              count(*)::bigint AS settlements_discovered,
              count(*) FILTER (WHERE credit_status = 'credited')::bigint
                AS settlements_credited,
              count(*) FILTER (WHERE credit_status = 'pending')::bigint
                AS settlements_pending,
              COALESCE(sum(entry_notional) FILTER (WHERE credit_status = 'credited'), 0)::numeric
                AS entry_notional,
              COALESCE(sum(entry_fees) FILTER (WHERE credit_status = 'credited'), 0)::numeric
                AS entry_fees,
              COALESCE(sum(payout) FILTER (WHERE credit_status = 'credited'), 0)::numeric
                AS payout_credited,
              COALESCE(sum(net_pnl) FILTER (WHERE credit_status = 'credited'), 0)::numeric
                AS net_pnl,
              max(official_resolution_received_at) AS last_official_resolution_received_at,
              max(credited_at) AS last_credited_at
            FROM polymarket.btc_paper_settlement_ledger
            WHERE experiment_id = $1
            "#,
        )
        .bind(experiment_id)
        .fetch_one(&self.pool)
        .await
        .context("failed to summarize BTC paper settlement ledger")
    }

    /// Recomputes paper results from immutable fills and official Polymarket outcomes. A local
    /// Chainlink label alone is never authoritative for P&L. Re-running this method is idempotent
    /// because counters and PnL are assigned from the aggregate rather than incremented.
    pub async fn refresh_paper_experiment_settlement(
        &self,
        experiment_id: Uuid,
    ) -> Result<BtcExperimentSettlementSummary> {
        let summary = sqlx::query_as::<_, BtcExperimentSettlementSummary>(
            r#"
            WITH entered AS (
              SELECT
                o.order_id,
                o.market_id,
                o.token_id,
                sum(f.size)::numeric AS filled_size,
                sum(f.price * f.size)::numeric AS entry_cost,
                sum(f.fee)::numeric AS fees
              FROM polymarket.orders o
              JOIN polymarket.fills f ON f.order_id = o.order_id
              WHERE o.raw_payload #>> '{request,metadata,experiment_id}' = $1::text
                AND f.source = 'paper'
              GROUP BY o.order_id, o.market_id, o.token_id
            ),
            resolved AS (
              SELECT
                e.*,
                CASE
                  WHEN (m.official_outcome = 'up' AND e.token_id = m.up_token_id)
                    OR (m.official_outcome = 'down' AND e.token_id = m.down_token_id)
                  THEN e.filled_size
                  ELSE 0::numeric
                END AS payout
              FROM entered e
              JOIN polymarket.btc_interval_markets m ON m.market_id = e.market_id
              JOIN polymarket.btc_official_resolution_watches w ON w.market_id = m.market_id
              WHERE m.official_outcome IS NOT NULL
                AND m.official_winning_token_id IS NOT NULL
                AND m.official_resolution_received_at IS NOT NULL
                AND m.official_resolution_source IS NOT NULL
                AND (
                  (m.official_outcome = 'up' AND m.official_winning_token_id = m.up_token_id)
                  OR
                  (m.official_outcome = 'down' AND m.official_winning_token_id = m.down_token_id)
                )
                AND w.status IN ('resolved', 'resolved_late')
                AND w.resolution_received_at = m.official_resolution_received_at
                AND w.resolution_source = m.official_resolution_source
            )
            SELECT
              (SELECT count(*)::bigint FROM entered) AS trades_entered,
              count(*)::bigint AS trades_resolved,
              COALESCE(sum(payout - entry_cost), 0)::numeric AS gross_pnl,
              COALESCE(sum(fees), 0)::numeric AS fees_paid,
              COALESCE(sum(payout - entry_cost - fees), 0)::numeric AS net_pnl
            FROM resolved
            "#,
        )
        .bind(experiment_id)
        .fetch_one(&self.pool)
        .await
        .context("failed to aggregate BTC paper experiment settlement")?;
        sqlx::query(
            r#"
            UPDATE polymarket.btc_paper_experiments
            SET trades_entered = $2,
                trades_resolved = $3,
                gross_pnl = $4,
                fees_paid = $5,
                net_pnl = $6,
                updated_at = now()
            WHERE experiment_id = $1
            "#,
        )
        .bind(experiment_id)
        .bind(summary.trades_entered)
        .bind(summary.trades_resolved)
        .bind(summary.gross_pnl)
        .bind(summary.fees_paid)
        .bind(summary.net_pnl)
        .execute(&self.pool)
        .await
        .context("failed to refresh BTC paper experiment settlement")?;
        Ok(summary)
    }

    /// Repairs settlement projections for every experiment that actually filled this market,
    /// including stopped cohorts. Running it after every idempotent official replay closes the
    /// crash window between the official commit and projection refresh.
    pub async fn refresh_paper_experiments_for_market(&self, market_id: &str) -> Result<u64> {
        let experiment_ids = sqlx::query_scalar::<_, Uuid>(
            r#"
            SELECT DISTINCT e.experiment_id
            FROM polymarket.btc_paper_experiments e
            JOIN polymarket.orders o
              ON o.raw_payload #>> '{request,metadata,experiment_id}' = e.experiment_id::text
            WHERE o.market_id = $1
              AND EXISTS (
                SELECT 1
                FROM polymarket.fills f
                WHERE f.order_id = o.order_id AND f.source = 'paper'
              )
            ORDER BY e.experiment_id
            "#,
        )
        .bind(market_id)
        .fetch_all(&self.pool)
        .await
        .context("failed to locate BTC experiments affected by official resolution")?;
        for experiment_id in &experiment_ids {
            self.refresh_paper_experiment_settlement(*experiment_id)
                .await
                .with_context(|| {
                    format!("failed to refresh BTC paper settlement for experiment {experiment_id}")
                })?;
        }
        Ok(experiment_ids.len() as u64)
    }
}

async fn load_market_boundary_for_update(
    tx: &mut Transaction<'_, Postgres>,
    market_id: &str,
) -> Result<MarketBoundaryRow> {
    sqlx::query_as::<_, MarketBoundaryRow>(
        r#"
        SELECT market_id, window_start, window_end, up_token_id, down_token_id,
          reference_price, reference_source_timestamp, resolution_price,
          resolution_source_timestamp, official_outcome, official_resolved_at,
          official_winning_token_id
        FROM polymarket.btc_interval_markets
        WHERE market_id = $1
        FOR UPDATE
        "#,
    )
    .bind(market_id)
    .fetch_optional(&mut **tx)
    .await
    .context("failed to lock BTC market boundary state")?
    .with_context(|| format!("BTC market {market_id} does not exist"))
}

fn validate_paper_settlement_record(record: &BtcPaperSettlementRecord) -> Result<()> {
    if !matches!(record.official_outcome.as_str(), "up" | "down") {
        bail!("paper settlement has a nonofficial outcome");
    }
    if !matches!(
        record.official_resolution_source.as_str(),
        "clob_websocket" | "clob_rest_reconciliation"
    ) {
        bail!("paper settlement has an unsupported official-resolution source");
    }
    if record.official_winning_token_id.trim().is_empty() {
        bail!("paper settlement is missing its official winning token");
    }
    if record
        .fill_ids
        .as_array()
        .is_none_or(|fill_ids| fill_ids.is_empty())
    {
        bail!("paper settlement must attribute at least one fill");
    }
    if record.filled_size <= Decimal::ZERO
        || record.entry_notional < Decimal::ZERO
        || record.entry_fees < Decimal::ZERO
        || record.payout < Decimal::ZERO
    {
        bail!("paper settlement contains invalid amounts");
    }
    let expected_payout = if record.token_id == record.official_winning_token_id {
        record.filled_size
    } else {
        Decimal::ZERO
    };
    if record.payout != expected_payout {
        bail!("paper settlement payout conflicts with the official winning token");
    }
    if record.net_pnl != record.payout - record.entry_notional - record.entry_fees {
        bail!("paper settlement net PnL conflicts with its payout and entry costs");
    }
    Ok(())
}

fn reference_tick_integrity_status(tick: &ReferencePriceTick) -> &'static str {
    let clock_skew_ms = (tick.received_at - tick.source_timestamp).num_milliseconds();
    if clock_skew_ms < -2_000 {
        "future"
    } else if clock_skew_ms > 10_000 {
        "stale"
    } else {
        "ok"
    }
}

fn require_healthy_chainlink_tick(tick: &ReferencePriceTick, boundary: &str) -> Result<()> {
    if tick.source != ReferencePriceSource::RtdsChainlink {
        bail!("BTC {boundary} boundary requires an RTDS Chainlink tick");
    }
    let integrity = reference_tick_integrity_status(tick);
    if integrity != "ok" {
        bail!("BTC {boundary} boundary tick has {integrity} clock integrity");
    }
    Ok(())
}

fn require_tick_in_boundary_window(
    tick: &ReferencePriceTick,
    boundary_at: DateTime<Utc>,
    latest_at: DateTime<Utc>,
    boundary: &str,
) -> Result<()> {
    if tick.source_timestamp < boundary_at || tick.source_timestamp > latest_at {
        bail!(
            "BTC {boundary} boundary tick at {} is outside [{boundary_at}, {latest_at}]",
            tick.source_timestamp
        );
    }
    Ok(())
}

fn require_label_matches_market(
    label: &BtcMarketLabel,
    market: &MarketBoundaryRow,
    max_delay: Duration,
) -> Result<()> {
    if label.market_id != market.market_id
        || label.window_start != market.window_start
        || label.window_end != market.window_end
    {
        bail!(
            "BTC label market/window identity conflicts with market {}",
            market.market_id
        );
    }
    if label.label_source != "rtds_chainlink" {
        bail!("BTC label {} has unsupported source", label.market_id);
    }
    match (
        market.reference_price,
        market.reference_source_timestamp,
    ) {
        (Some(price), Some(timestamp))
            if price == label.open_price && timestamp == label.source_open_timestamp => {}
        (price, timestamp) => bail!(
            "BTC label opening evidence conflicts with market {}: stored ({price:?}, {timestamp:?}), label ({}, {})",
            market.market_id,
            label.open_price,
            label.source_open_timestamp
        ),
    }
    if label.source_close_timestamp < market.window_end
        || label.source_close_timestamp > market.window_end + max_delay
    {
        bail!(
            "BTC label close timestamp is outside the eligible boundary window for market {}",
            market.market_id
        );
    }
    let expected = if label.close_price >= label.open_price {
        BtcOutcome::Up
    } else {
        BtcOutcome::Down
    };
    if label.outcome != expected {
        bail!(
            "BTC label outcome conflicts with boundary prices for market {}",
            market.market_id
        );
    }
    Ok(())
}

fn official_outcome_for_winner(
    up_token_id: &str,
    down_token_id: &str,
    winning_token_id: &str,
    winning_outcome: &str,
) -> Result<BtcOutcome> {
    let token_outcome = if winning_token_id == up_token_id {
        BtcOutcome::Up
    } else if winning_token_id == down_token_id {
        BtcOutcome::Down
    } else {
        bail!("official BTC resolution names an unknown winning token");
    };
    let declared = match winning_outcome.trim().to_ascii_lowercase().as_str() {
        "up" => BtcOutcome::Up,
        "down" => BtcOutcome::Down,
        value => bail!("official BTC resolution names unsupported outcome {value}"),
    };
    if declared != token_outcome {
        bail!("official BTC winner token and declared outcome disagree");
    }
    Ok(token_outcome)
}

fn same_boundary_tick(left: &ReferencePriceTick, right: &ReferencePriceTick) -> bool {
    left.tick_id == right.tick_id
        && left.dedup_key == right.dedup_key
        && left.source == right.source
        && left.symbol == right.symbol
        && left.source_timestamp == right.source_timestamp
        && left.price == right.price
}

fn same_immutable_market_label(left: &BtcMarketLabel, right: &BtcMarketLabel) -> bool {
    left.market_id == right.market_id
        && left.window_start == right.window_start
        && left.window_end == right.window_end
        && left.open_price == right.open_price
        && left.close_price == right.close_price
        && left.outcome == right.outcome
        && left.label_source == right.label_source
        && left.label_version == right.label_version
        && left.source_open_timestamp == right.source_open_timestamp
        && left.source_close_timestamp == right.source_close_timestamp
}

fn truncate_history_at_current(
    history: &mut Vec<ReferencePriceTick>,
    current: Option<&ReferencePriceTick>,
) {
    let Some(current) = current else {
        history.clear();
        return;
    };
    let Some(current_index) = history
        .iter()
        .rposition(|tick| tick.tick_id == current.tick_id)
    else {
        history.clear();
        return;
    };
    history.truncate(current_index + 1);
}

fn reference_tick_from_row(row: ReferenceTickRow) -> Result<ReferencePriceTick> {
    let source = match row.source.as_str() {
        "direct_binance" => ReferencePriceSource::DirectBinance,
        "rtds_binance" => ReferencePriceSource::RtdsBinance,
        "rtds_chainlink" => ReferencePriceSource::RtdsChainlink,
        source => anyhow::bail!("unsupported stored BTC reference source {source}"),
    };
    Ok(ReferencePriceTick {
        tick_id: row.tick_id,
        dedup_key: row.dedup_key,
        source,
        symbol: row.symbol,
        price: row.price,
        source_timestamp: row.source_timestamp,
        envelope_timestamp: row.envelope_timestamp,
        received_at: row.received_at,
        connection_id: row.connection_id,
        ingest_sequence: u64::try_from(row.ingest_sequence).unwrap_or_default(),
        source_event_id: row.source_event_id,
        raw_payload: row.raw_payload,
    })
}

fn market_label_from_row(row: StoredMarketLabelRow) -> Result<BtcMarketLabel> {
    Ok(BtcMarketLabel {
        market_id: row.market_id,
        window_start: row.window_start,
        window_end: row.window_end,
        open_price: row.open_price,
        close_price: row.close_price,
        outcome: parse_outcome_name(&row.outcome)?,
        label_source: row.label_source,
        label_version: row.label_version,
        source_open_timestamp: row.source_open_timestamp,
        source_close_timestamp: row.source_close_timestamp,
        label_available_at: row.label_available_at,
        evidence: row.evidence,
    })
}

fn official_resolution_watch_from_row(
    row: ResolutionWatchMarketRow,
) -> Result<BtcOfficialResolutionWatch> {
    let mut fee_schedule = serde_json::Map::new();
    if let Some(rate) = row.fee_rate {
        fee_schedule.insert(
            "rate".to_string(),
            serde_json::Value::String(rate.normalize().to_string()),
        );
    }
    if let Some(exponent) = row.fee_exponent {
        fee_schedule.insert("exponent".to_string(), serde_json::json!(exponent));
    }
    if let Some(taker_only) = row.fee_taker_only {
        fee_schedule.insert("takerOnly".to_string(), serde_json::json!(taker_only));
    }
    Ok(BtcOfficialResolutionWatch {
        status: row.watch_status,
        deadline_at: row.deadline_at,
        market: BtcIntervalMarket {
            event_id: row.event_id,
            event_slug: row.event_slug,
            series_slug: row.series_slug,
            market_id: row.market_id,
            condition_id: row.condition_id,
            window_start: row.window_start,
            window_end: row.window_end,
            up_token_id: row.up_token_id,
            down_token_id: row.down_token_id,
            tick_size: row.min_tick_size,
            minimum_order_size: Some(row.min_order_size),
            resolution_source: row.resolution_source,
            active: row.active,
            closed: row.closed,
            accepting_orders: row.accepting_orders,
            fees_enabled: row.fee_rate.is_some(),
            fee_schedule: serde_json::Value::Object(fee_schedule),
            raw_payload: row.raw_payload,
        },
    })
}

fn market_open_reference_from_row(
    row: MarketOpenReferenceRow,
) -> Result<(String, ReferencePriceTick)> {
    let market_id = row.market_id;
    let tick = reference_tick_from_row(ReferenceTickRow {
        tick_id: row.tick_id,
        source_timestamp: row.source_timestamp,
        received_at: row.received_at,
        source: row.source,
        symbol: row.symbol,
        price: row.price,
        envelope_timestamp: row.envelope_timestamp,
        connection_id: row.connection_id,
        ingest_sequence: row.ingest_sequence,
        source_event_id: row.source_event_id,
        dedup_key: row.dedup_key,
        raw_payload: row.raw_payload,
    })?;
    Ok((market_id, tick))
}

fn checkpoint_from_row(row: CheckpointRow) -> Result<OrderbookCheckpoint> {
    let bids = serde_json::from_value::<Vec<OrderbookLevel>>(
        row.book.get("bids").cloned().unwrap_or_default(),
    )
    .context("failed to decode stored BTC bid levels")?;
    let asks = serde_json::from_value::<Vec<OrderbookLevel>>(
        row.book.get("asks").cloned().unwrap_or_default(),
    )
    .context("failed to decode stored BTC ask levels")?;
    let integrity_status = match row.integrity_status.as_str() {
        "ok" => FeedIntegrityStatus::Ok,
        "pre_snapshot" => FeedIntegrityStatus::PreSnapshot,
        "stale" => FeedIntegrityStatus::Stale,
        "out_of_order" => FeedIntegrityStatus::OutOfOrder,
        "decode_error" => FeedIntegrityStatus::DecodeError,
        "crossed_book" => FeedIntegrityStatus::CrossedBook,
        "top_of_book_mismatch" => FeedIntegrityStatus::TopOfBookMismatch,
        "unknown_token" => FeedIntegrityStatus::UnknownToken,
        "market_mismatch" => FeedIntegrityStatus::MarketMismatch,
        value => anyhow::bail!("unsupported stored BTC book integrity status {value}"),
    };
    Ok(OrderbookCheckpoint {
        checkpoint_id: row.checkpoint_id,
        market_id: row.market_id,
        token_id: row.token_id,
        source_timestamp: row.source_timestamp,
        received_at: row.received_at,
        connection_id: row.connection_id,
        ingest_sequence: u64::try_from(row.ingest_sequence).unwrap_or_default(),
        source_hash: row.source_hash,
        tick_size: row.tick_size,
        best_bid: row.best_bid,
        best_ask: row.best_ask,
        bids,
        asks,
        integrity_status,
    })
}

fn decision_edge_projection(decision: &BtcDecision) -> Result<BtcDecisionEdgeProjection<'_>> {
    let (expected_outcome, edge) = match decision.action {
        BtcDecisionAction::BuyUp => (Some(BtcOutcome::Up), decision.up_edge.as_ref()),
        BtcDecisionAction::BuyDown => (Some(BtcOutcome::Down), decision.down_edge.as_ref()),
        BtcDecisionAction::NoTrade => (None, None),
    };
    let Some(edge) = edge else {
        if expected_outcome.is_some() {
            bail!("approved BTC decision is missing its selected outcome edge");
        }
        return Ok(BtcDecisionEdgeProjection::default());
    };
    if edge.size <= Decimal::ZERO || Some(edge.outcome) != expected_outcome {
        bail!("approved BTC decision has inconsistent selected outcome edge");
    }

    match decision.prediction.as_ref() {
        None => Ok(BtcDecisionEdgeProjection {
            token_id: Some(edge.token_id.as_str()),
            fair_probability: Some(edge.conservative_probability),
            executable_price: Some(edge.executable_price),
            gross_edge_per_share: Some(edge.gross_edge / edge.size),
            fee_per_share: Some(edge.taker_fee / edge.size),
            reserve_per_share: Some(
                (edge.spread_reserve + edge.slippage_reserve + edge.latency_reserve) / edge.size,
            ),
            net_edge_per_share: Some(edge.net_edge_per_share),
            size: edge.size,
        }),
        Some(BtcStrategyPrediction::NoPrediction { .. }) => {
            bail!("approved BTC decision cannot carry a no-prediction result")
        }
        Some(BtcStrategyPrediction::DirectionalPrediction {
            outcome,
            probability,
            conservative_probability,
            minimum_conservative_probability,
            executable_price,
            direct_taker_fee_per_share,
            direct_net_edge_per_share,
            ..
        }) => {
            let intent = decision
                .approved_intent
                .as_ref()
                .context("directional BTC decision is missing its approved intent")?;
            let executable_price =
                executable_price.context("directional BTC prediction is missing its price")?;
            let fee_per_share = direct_taker_fee_per_share
                .context("directional BTC prediction is missing its fee")?;
            let net_edge_per_share = direct_net_edge_per_share
                .context("directional BTC prediction is missing its direct edge")?;
            let gross_edge_per_share = *probability - executable_price;
            if *outcome != edge.outcome
                || intent.outcome != edge.outcome
                || intent.token_id != edge.token_id
                || intent.size != edge.size
                || executable_price != edge.executable_price
                || fee_per_share < Decimal::ZERO
                || *conservative_probability < *minimum_conservative_probability
                || net_edge_per_share <= Decimal::ZERO
                || gross_edge_per_share - fee_per_share != net_edge_per_share
                || intent.expected_net_edge_per_share != net_edge_per_share
                || intent.expected_net_edge != net_edge_per_share * edge.size
            {
                bail!("directional BTC decision has inconsistent prediction edge attribution");
            }
            Ok(BtcDecisionEdgeProjection {
                token_id: Some(edge.token_id.as_str()),
                fair_probability: Some(*probability),
                executable_price: Some(executable_price),
                gross_edge_per_share: Some(gross_edge_per_share),
                fee_per_share: Some(fee_per_share),
                reserve_per_share: Some(Decimal::ZERO),
                net_edge_per_share: Some(net_edge_per_share),
                size: edge.size,
            })
        }
    }
}

fn sequence_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn empty_to_none(value: &str) -> Option<&str> {
    (!value.trim().is_empty()).then_some(value)
}

fn serde_name<T: Serialize>(value: &T) -> Result<String> {
    serde_json::to_value(value)?
        .as_str()
        .map(ToString::to_string)
        .context("enum did not serialize to a string")
}

fn outcome_name(outcome: BtcOutcome) -> &'static str {
    match outcome {
        BtcOutcome::Up => "up",
        BtcOutcome::Down => "down",
    }
}

fn parse_outcome_name(value: &str) -> Result<BtcOutcome> {
    match value {
        "up" => Ok(BtcOutcome::Up),
        "down" => Ok(BtcOutcome::Down),
        value => bail!("unsupported stored BTC outcome {value}"),
    }
}

fn shadow_predictive_regime_history_limit(max_candidates: u32) -> Result<i64> {
    if !(1..=MAX_SHADOW_PREDICTIVE_REGIME_HISTORY_CANDIDATES).contains(&max_candidates) {
        bail!(
            "shadow predictive-regime history limit must be between 1 and {} candidates",
            MAX_SHADOW_PREDICTIVE_REGIME_HISTORY_CANDIDATES
        );
    }
    Ok(i64::from(max_candidates))
}

fn shadow_predictive_regime_candidate_from_row(
    row: ShadowPredictiveRegimeCandidateRow,
) -> Result<ShadowPredictiveRegimeCandidate> {
    if row.decision_at >= row.label_available_at {
        bail!("shadow predictive-regime candidate has a non-causal label");
    }
    let candidate = ShadowPredictiveRegimeCandidate {
        market_id: row.market_id,
        decision_id: row.decision_id,
        decision_outcome: parse_outcome_name(&row.decision_outcome)?,
        resolved_outcome: parse_outcome_name(&row.resolved_outcome)?,
        selected_point_probability: row.selected_point_probability,
        decision_at: row.decision_at,
        label_available_at: row.label_available_at,
    };
    candidate.validate()?;
    Ok(candidate)
}

fn shadow_predictive_regime_state_from_persisted_evaluation(
    process_id: Uuid,
    breaker_state_config_hash: Option<&str>,
    row: PersistedShadowPredictiveRegimeEvaluationRow,
) -> Result<ShadowPredictiveRegimeState> {
    let evaluation = serde_json::from_value::<ShadowPredictiveRegimeEvaluation>(row.evaluation)
        .context("failed to deserialize persisted shadow predictive-regime evaluation")?;
    let state = &evaluation.state;

    if evaluation.process_id != process_id || state.process_id != process_id {
        bail!("persisted shadow predictive-regime state is owned by another process");
    }
    if evaluation.config_hash != state.config_hash {
        bail!("persisted shadow predictive-regime evaluation and state config hashes disagree");
    }
    if let Some(expected) = breaker_state_config_hash {
        if evaluation.config_hash != expected {
            bail!("persisted shadow predictive-regime state config hash does not match");
        }
    }
    if evaluation.as_of != row.decision_at {
        bail!("persisted shadow predictive-regime evaluation is not aligned to its decision");
    }
    if evaluation.degraded != state.degraded
        || evaluation.would_defer != state.degraded
        || evaluation.consecutive_degradation_markets != state.consecutive_degradation_markets
        || evaluation.consecutive_recovery_markets != state.consecutive_recovery_markets
        || evaluation.resolved_markets_observed != state.resolved_markets_observed
        || evaluation.state_as_of_market_id != state.state_as_of_market_id
        || evaluation.state_as_of_decision_id != state.state_as_of_decision_id
        || evaluation.state_as_of_label_available_at != state.state_as_of_label_available_at
    {
        bail!("persisted shadow predictive-regime evaluation and state evidence disagree");
    }
    #[derive(Serialize)]
    struct PersistedStateEvidence<'a> {
        schema_version: &'a str,
        config_hash: String,
        state: &'a ShadowPredictiveRegimeState,
    }
    let expected_state_evidence_sha256 = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&PersistedStateEvidence {
            schema_version: &evaluation.schema_version,
            config_hash: state.config_hash.clone(),
            state,
        })?)
    );
    if !evaluation.shadow_only
        || !evaluation.state_checkpoint_eligible
        || evaluation.state_evidence_sha256 != expected_state_evidence_sha256
        || evaluation.evaluation_evidence_sha256.trim().is_empty()
    {
        bail!("persisted shadow predictive-regime evaluation evidence is incomplete");
    }
    if state.resolved_markets_observed
        < u64::try_from(state.rolling_candidates.len()).unwrap_or(u64::MAX)
    {
        bail!("persisted shadow predictive-regime state observation count is inconsistent");
    }

    let mut market_ids = HashSet::new();
    let mut decision_ids = HashSet::new();
    let mut previous_ordering_key = None;
    for candidate in &state.rolling_candidates {
        candidate.validate()?;
        if candidate.label_available_at > evaluation.as_of {
            bail!("persisted shadow predictive-regime state contains future evidence");
        }
        if !market_ids.insert(candidate.market_id.as_str())
            || !decision_ids.insert(candidate.decision_id)
        {
            bail!("persisted shadow predictive-regime state contains duplicate evidence");
        }
        let ordering_key = (
            candidate.label_available_at,
            candidate.decision_at,
            candidate.decision_id,
            candidate.market_id.as_str(),
        );
        if previous_ordering_key.is_some_and(|previous| ordering_key <= previous) {
            bail!("persisted shadow predictive-regime state is not in causal order");
        }
        previous_ordering_key = Some(ordering_key);
    }

    match state.rolling_candidates.last() {
        Some(last)
            if state.state_as_of_market_id.as_deref() == Some(last.market_id.as_str())
                && state.state_as_of_decision_id == Some(last.decision_id)
                && state.state_as_of_decision_at == Some(last.decision_at)
                && state.state_as_of_label_available_at == Some(last.label_available_at) => {}
        Some(_) => {
            bail!("persisted shadow predictive-regime state cursor does not match its evidence")
        }
        None if state.resolved_markets_observed == 0
            && state.state_as_of_market_id.is_none()
            && state.state_as_of_decision_id.is_none()
            && state.state_as_of_decision_at.is_none()
            && state.state_as_of_label_available_at.is_none()
            && !state.degraded
            && state.consecutive_degradation_markets == 0
            && state.consecutive_recovery_markets == 0 => {}
        None => bail!("persisted shadow predictive-regime state is missing its evidence"),
    }

    Ok(evaluation.state)
}

fn decimal_json_field(value: &serde_json::Value, keys: &[&str]) -> Option<Decimal> {
    keys.iter().find_map(|key| match value.get(*key)? {
        serde_json::Value::String(value) => Decimal::from_str(value).ok(),
        serde_json::Value::Number(value) => Decimal::from_str(&value.to_string()).ok(),
        _ => None,
    })
}

fn integer_json_field(value: &serde_json::Value, keys: &[&str]) -> Option<i32> {
    keys.iter().find_map(|key| match value.get(*key)? {
        serde_json::Value::String(value) => value.parse().ok(),
        serde_json::Value::Number(value) => value.as_i64().and_then(|value| value.try_into().ok()),
        _ => None,
    })
}

fn bool_json_field(value: &serde_json::Value, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(serde_json::Value::as_bool))
}

fn daily_high_water_mark_state_from_rows(
    process_id: Uuid,
    as_of: DateTime<Utc>,
    rows: Vec<DailyHighWaterMarkEvidenceRow>,
) -> Result<DailyRealizedPnlHighWaterMarkState> {
    let mut credits = Vec::new();
    let mut unsettled_exposures = Vec::new();
    for row in rows {
        match row.evidence_kind.as_str() {
            "credited" => {
                let settlement_id = row
                    .settlement_id
                    .context("credited high-water-mark evidence is missing settlement id")?;
                if !row.fill_ids.is_empty() {
                    bail!("credited high-water-mark evidence unexpectedly contains fill ids");
                }
                credits.push(DailyRealizedPnlCredit {
                    settlement_id,
                    order_id: row.order_id,
                    credited_at: row.occurred_at,
                    net_pnl_usd: row.amount_usd,
                });
            }
            "unsettled" => {
                if row.settlement_id.is_some() {
                    bail!("unsettled high-water-mark evidence unexpectedly has settlement id");
                }
                unsettled_exposures.push(UnsettledEntryExposure {
                    order_id: row.order_id,
                    fill_ids: row.fill_ids,
                    entry_debit_usd: row.amount_usd,
                });
            }
            "conflicting_credit" => bail!(
                "process-owned high-water-mark evidence has {} conflicting credited economics for order {}",
                row.amount_usd,
                row.order_id
            ),
            kind => bail!("unsupported high-water-mark evidence kind {kind}"),
        }
    }
    DailyRealizedPnlHighWaterMarkState::from_evidence(
        process_id,
        as_of,
        credits,
        unsettled_exposures,
    )
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;
    use rust_decimal_macros::dec;

    use super::*;
    use crate::btc::types::{FeedIntegrityStatus, MarketFeedEventType};
    use crate::btc::{
        admission::{
            ShadowPredictiveRegimeCircuitBreakerConfig,
            SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE,
            SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION,
        },
        strategy::{ApprovedIntent, OutcomeEdge},
    };

    fn history_tick(id: u128, at: DateTime<Utc>) -> ReferencePriceTick {
        ReferencePriceTick {
            tick_id: Uuid::from_u128(id),
            dedup_key: id.to_string(),
            source: ReferencePriceSource::RtdsChainlink,
            symbol: "BTCUSD".to_string(),
            price: dec!(70_000),
            source_timestamp: at + Duration::milliseconds(id as i64),
            envelope_timestamp: None,
            received_at: at + Duration::milliseconds(id as i64),
            connection_id: Uuid::from_u128(100),
            ingest_sequence: id as u64,
            source_event_id: None,
            raw_payload: serde_json::json!({}),
        }
    }

    #[test]
    fn history_endpoint_is_truncated_to_the_fresh_canonical_tick() {
        let at = Utc.with_ymd_and_hms(2026, 7, 21, 12, 0, 0).unwrap();
        let mut history = vec![
            history_tick(1, at),
            history_tick(2, at),
            history_tick(3, at),
        ];
        let current = history[1].clone();

        truncate_history_at_current(&mut history, Some(&current));

        assert_eq!(history.len(), 2);
        assert_eq!(
            history.last().map(|tick| tick.tick_id),
            Some(current.tick_id)
        );

        truncate_history_at_current(&mut history, Some(&history_tick(99, at)));
        assert!(history.is_empty());
    }

    fn approved_decision(
        outcome: BtcOutcome,
        prediction: Option<BtcStrategyPrediction>,
        intent_net_edge_per_share: Decimal,
    ) -> BtcDecision {
        let at = Utc::now();
        let edge = OutcomeEdge {
            outcome,
            token_id: format!("{}-token", outcome_name(outcome)),
            conservative_probability: dec!(0.78),
            executable_price: dec!(0.72),
            marketable_limit_price: dec!(0.73),
            size: dec!(5),
            gross_edge: dec!(0.30),
            taker_fee: dec!(0.05),
            spread_reserve: dec!(0.02),
            slippage_reserve: dec!(0.01),
            latency_reserve: dec!(0.01),
            net_edge: dec!(0.21),
            net_edge_per_share: dec!(0.042),
        };
        let intent = ApprovedIntent {
            intent_id: Uuid::from_u128(10),
            process_id: Uuid::from_u128(11),
            feature_snapshot_id: Uuid::from_u128(12),
            market_id: "market".to_string(),
            window_start: at,
            outcome,
            token_id: edge.token_id.clone(),
            limit_price: edge.marketable_limit_price,
            size: edge.size,
            expected_net_edge: intent_net_edge_per_share * edge.size,
            expected_net_edge_per_share: intent_net_edge_per_share,
            strategy_version: "strategy".to_string(),
            feature_schema_version: "features".to_string(),
        };
        BtcDecision {
            decision_id: Uuid::from_u128(13),
            process_id: intent.process_id,
            feature_snapshot_id: intent.feature_snapshot_id,
            evaluated_at: at,
            action: match outcome {
                BtcOutcome::Up => BtcDecisionAction::BuyUp,
                BtcOutcome::Down => BtcDecisionAction::BuyDown,
            },
            reject_reason: None,
            fair_value: None,
            up_edge: (outcome == BtcOutcome::Up).then(|| edge.clone()),
            down_edge: (outcome == BtcOutcome::Down).then_some(edge),
            approved_intent: Some(intent),
            prediction,
        }
    }

    #[test]
    fn high_water_mark_query_is_process_owned_and_canonicalizes_orders() {
        let normalized = DAILY_HIGH_WATER_MARK_EVIDENCE_SQL.to_ascii_lowercase();
        assert!(!normalized.contains("experiment_id"));
        assert!(normalized.contains("distinct on (l.process_id, l.order_id)"));
        assert!(normalized.contains("conflicting_credited_orders"));
        assert!(normalized.contains("count(distinct ("));
        assert!(normalized.contains("filled_size,"));
        assert!(normalized.contains("entry_notional,"));
        assert!(normalized.contains("entry_fees,"));
        assert!(normalized.contains("payout,"));
        assert!(normalized.contains("net_pnl"));
        assert!(normalized.contains("'conflicting_credit'::text"));
        assert!(normalized.contains("l.process_id = $1"));
        assert!(normalized.contains("o.process_id = $1"));
        assert!(normalized.contains("f.process_id = o.process_id"));
        assert!(normalized.contains("c.process_id = f.process_id"));
        assert!(normalized.contains("l.credited_at <= $2"));
        assert!(normalized.contains("f.timestamp_utc <= $2"));
    }

    #[test]
    fn existing_entry_guard_is_process_owned() {
        let normalized = PROCESS_HAS_ENTRY_SQL.to_ascii_lowercase();
        assert!(normalized.contains("where process_id = $1"));
        assert!(!normalized.contains("experiment_id"));
    }

    #[test]
    fn shadow_predictive_regime_query_is_process_owned_causal_canonical_and_bounded() {
        let normalized = SHADOW_PREDICTIVE_REGIME_CANDIDATES_SQL.to_ascii_lowercase();

        assert!(normalized.contains("canonical as materialized"));
        assert!(normalized.contains("distinct on (d.market_id)"));
        assert!(normalized.contains("d.process_id = $1"));
        assert!(normalized.contains("($2::text is null or d.config_hash = $2)"));
        assert!(normalized.contains("d.action = 'buy'"));
        assert!(normalized.contains("d.decision_at < l.label_available_at"));
        assert!(normalized.contains("l.label_available_at <= $3"));
        assert!(normalized.contains("f.snapshot_id = c.snapshot_id"));
        assert!(normalized.contains("f.feature_as_of = c.decision_at"));
        assert!(normalized.contains("cross join lateral"));
        assert!(normalized.contains("offset 0"));
        assert!(normalized.contains("when 'up' then f.fair_up_probability"));
        assert!(normalized.contains("when 'down' then 1 - f.fair_up_probability"));
        assert!(normalized.contains(
            "order by label_available_at desc, decision_at desc, decision_id desc, market_id desc"
        ));
        assert!(normalized.contains("limit $4"));
        assert!(normalized.contains(
            "from bounded\norder by label_available_at, decision_at, decision_id, market_id"
        ));
        assert!(!normalized.contains("experiment_id"));
        assert!(!normalized.contains("status"));
        assert!(!normalized.contains("polymarket.fills"));
        assert!(!normalized.contains("d.fair_probability"));
    }

    #[test]
    fn latest_shadow_predictive_regime_state_query_is_process_owned_causal_and_status_agnostic() {
        let normalized = LATEST_SHADOW_PREDICTIVE_REGIME_STATE_SQL.to_ascii_lowercase();

        assert!(normalized.contains("d.process_id = $1"));
        assert!(normalized
            .contains("shadow_predictive_regime_circuit_breaker,state,config_hash}' = $2"));
        assert!(normalized.contains("d.decision_at <= $3"));
        assert!(normalized.contains("d.action = 'buy'"));
        assert!(normalized.contains("state_checkpoint_eligible}' = 'true'"));
        assert!(normalized.contains("order by d.decision_at desc, d.decision_id desc"));
        assert!(normalized.contains("limit 1"));
        assert!(!normalized.contains("experiment_id"));
        assert!(!normalized.contains("d.config_hash"));
        assert!(!normalized.contains("status"));
        assert!(!normalized.contains("polymarket.fills"));
    }

    #[test]
    fn persisted_shadow_predictive_regime_state_round_trips_typed_evidence() {
        let process_id = Uuid::from_u128(450);
        let decision_at = Utc
            .with_ymd_and_hms(2026, 7, 20, 12, 0, 0)
            .single()
            .unwrap();
        let config = ShadowPredictiveRegimeCircuitBreakerConfig {
            schema_version: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION.to_string(),
            mode: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE.to_string(),
            rolling_resolved_market_window: 20,
            minimum_resolved_markets: 20,
            degradation_brier_score_threshold: dec!(0.23),
            degradation_overconfidence_gap_threshold: dec!(0.12),
            degradation_confirmation_markets: 2,
            recovery_brier_score_threshold: dec!(0.21),
            recovery_overconfidence_gap_threshold: dec!(0.05),
            recovery_confirmation_markets: 2,
        };
        let state = ShadowPredictiveRegimeState::new(process_id, &config).unwrap();
        let mut evaluation = state.evaluate(&config, decision_at).unwrap();
        evaluation.state_checkpoint_eligible = true;
        let config_hash = config.config_hash().unwrap();

        let restored = shadow_predictive_regime_state_from_persisted_evaluation(
            process_id,
            Some(&config_hash),
            PersistedShadowPredictiveRegimeEvaluationRow {
                decision_at,
                evaluation: serde_json::to_value(evaluation).unwrap(),
            },
        )
        .unwrap();

        assert_eq!(restored, state);
    }

    #[test]
    fn persisted_shadow_predictive_regime_state_rejects_identity_and_time_mismatches() {
        let process_id = Uuid::from_u128(451);
        let decision_at = Utc
            .with_ymd_and_hms(2026, 7, 20, 12, 0, 0)
            .single()
            .unwrap();
        let config = ShadowPredictiveRegimeCircuitBreakerConfig {
            schema_version: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION.to_string(),
            mode: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE.to_string(),
            rolling_resolved_market_window: 20,
            minimum_resolved_markets: 20,
            degradation_brier_score_threshold: dec!(0.23),
            degradation_overconfidence_gap_threshold: dec!(0.12),
            degradation_confirmation_markets: 2,
            recovery_brier_score_threshold: dec!(0.21),
            recovery_overconfidence_gap_threshold: dec!(0.05),
            recovery_confirmation_markets: 2,
        };
        let state = ShadowPredictiveRegimeState::new(process_id, &config).unwrap();
        let mut evaluation = state.evaluate(&config, decision_at).unwrap();
        evaluation.state_checkpoint_eligible = true;
        let evidence = serde_json::to_value(&evaluation).unwrap();
        let mut tampered_hash = evaluation.clone();
        tampered_hash.state_evidence_sha256 = "0".repeat(64);
        let mut ineligible_checkpoint = evaluation.clone();
        ineligible_checkpoint.state_checkpoint_eligible = false;

        assert!(shadow_predictive_regime_state_from_persisted_evaluation(
            Uuid::from_u128(999),
            None,
            PersistedShadowPredictiveRegimeEvaluationRow {
                decision_at,
                evaluation: evidence.clone(),
            },
        )
        .is_err());
        assert!(shadow_predictive_regime_state_from_persisted_evaluation(
            process_id,
            Some(&"f".repeat(64)),
            PersistedShadowPredictiveRegimeEvaluationRow {
                decision_at,
                evaluation: evidence.clone(),
            },
        )
        .is_err());
        assert!(shadow_predictive_regime_state_from_persisted_evaluation(
            process_id,
            None,
            PersistedShadowPredictiveRegimeEvaluationRow {
                decision_at: decision_at + Duration::seconds(1),
                evaluation: evidence,
            },
        )
        .is_err());
        assert!(shadow_predictive_regime_state_from_persisted_evaluation(
            process_id,
            None,
            PersistedShadowPredictiveRegimeEvaluationRow {
                decision_at,
                evaluation: serde_json::to_value(tampered_hash).unwrap(),
            },
        )
        .is_err());
        assert!(shadow_predictive_regime_state_from_persisted_evaluation(
            process_id,
            None,
            PersistedShadowPredictiveRegimeEvaluationRow {
                decision_at,
                evaluation: serde_json::to_value(ineligible_checkpoint).unwrap(),
            },
        )
        .is_err());
    }

    #[test]
    fn shadow_predictive_regime_history_limit_is_strictly_bounded() {
        assert!(shadow_predictive_regime_history_limit(0).is_err());
        assert_eq!(shadow_predictive_regime_history_limit(1).unwrap(), 1);
        assert_eq!(
            shadow_predictive_regime_history_limit(MAX_SHADOW_PREDICTIVE_REGIME_HISTORY_CANDIDATES)
                .unwrap(),
            i64::from(MAX_SHADOW_PREDICTIVE_REGIME_HISTORY_CANDIDATES)
        );
        assert!(shadow_predictive_regime_history_limit(
            MAX_SHADOW_PREDICTIVE_REGIME_HISTORY_CANDIDATES + 1
        )
        .is_err());
    }

    #[test]
    fn shadow_predictive_regime_row_decodes_selected_direction_probability() {
        let label_available_at = Utc
            .with_ymd_and_hms(2026, 7, 20, 12, 0, 0)
            .single()
            .unwrap();
        let candidate =
            shadow_predictive_regime_candidate_from_row(ShadowPredictiveRegimeCandidateRow {
                market_id: "btc-updown-5m-1".to_string(),
                decision_id: Uuid::from_u128(401),
                decision_outcome: "down".to_string(),
                resolved_outcome: "up".to_string(),
                selected_point_probability: dec!(0.27),
                decision_at: label_available_at - Duration::minutes(4),
                label_available_at,
            })
            .unwrap();

        assert_eq!(candidate.market_id, "btc-updown-5m-1");
        assert_eq!(candidate.decision_id, Uuid::from_u128(401));
        assert_eq!(candidate.decision_outcome, BtcOutcome::Down);
        assert_eq!(candidate.resolved_outcome, BtcOutcome::Up);
        assert_eq!(candidate.selected_point_probability, dec!(0.27));
        assert_eq!(candidate.label_available_at, label_available_at);
    }

    #[test]
    fn shadow_predictive_regime_row_fails_closed_on_malformed_evidence() {
        let label_available_at = Utc
            .with_ymd_and_hms(2026, 7, 20, 12, 0, 0)
            .single()
            .unwrap();
        let row = |decision_outcome: &str,
                   selected_point_probability: Decimal,
                   decision_at: DateTime<Utc>| {
            ShadowPredictiveRegimeCandidateRow {
                market_id: "btc-updown-5m-1".to_string(),
                decision_id: Uuid::from_u128(402),
                decision_outcome: decision_outcome.to_string(),
                resolved_outcome: "down".to_string(),
                selected_point_probability,
                decision_at,
                label_available_at,
            }
        };

        assert!(shadow_predictive_regime_candidate_from_row(row(
            "up",
            dec!(1.01),
            label_available_at - Duration::minutes(1),
        ))
        .is_err());
        assert!(shadow_predictive_regime_candidate_from_row(row(
            "up",
            dec!(0.60),
            label_available_at,
        ))
        .is_err());
        assert!(shadow_predictive_regime_candidate_from_row(row(
            "sideways",
            dec!(0.60),
            label_available_at - Duration::minutes(1),
        ))
        .is_err());
    }

    #[test]
    fn high_water_mark_rows_decode_into_credited_and_unsettled_evidence() {
        let as_of = Utc
            .with_ymd_and_hms(2026, 7, 20, 12, 0, 0)
            .single()
            .unwrap();
        let state = daily_high_water_mark_state_from_rows(
            Uuid::from_u128(100),
            as_of,
            vec![
                DailyHighWaterMarkEvidenceRow {
                    evidence_kind: "unsettled".to_string(),
                    settlement_id: None,
                    order_id: "open-order".to_string(),
                    occurred_at: as_of - Duration::minutes(1),
                    amount_usd: dec!(2.25),
                    fill_ids: vec![Uuid::from_u128(4), Uuid::from_u128(5)],
                },
                DailyHighWaterMarkEvidenceRow {
                    evidence_kind: "credited".to_string(),
                    settlement_id: Some(Uuid::from_u128(1)),
                    order_id: "closed-order".to_string(),
                    occurred_at: as_of - Duration::minutes(2),
                    amount_usd: dec!(6),
                    fill_ids: Vec::new(),
                },
            ],
        )
        .unwrap();

        assert_eq!(state.process_id, Uuid::from_u128(100));
        assert_eq!(state.daily_realized_pnl_usd, dec!(6));
        assert_eq!(state.high_water_mark_usd, dec!(6));
        assert_eq!(state.unsettled_order_count, 1);
        assert_eq!(state.unsettled_entry_debit_usd, dec!(2.25));
    }

    #[test]
    fn high_water_mark_rows_fail_closed_on_malformed_classification() {
        let as_of = Utc
            .with_ymd_and_hms(2026, 7, 20, 12, 0, 0)
            .single()
            .unwrap();
        let malformed = DailyHighWaterMarkEvidenceRow {
            evidence_kind: "unsettled".to_string(),
            settlement_id: Some(Uuid::from_u128(1)),
            order_id: "open-order".to_string(),
            occurred_at: as_of,
            amount_usd: dec!(1),
            fill_ids: vec![Uuid::from_u128(2)],
        };
        assert!(daily_high_water_mark_state_from_rows(
            Uuid::from_u128(100),
            as_of,
            vec![malformed],
        )
        .is_err());
    }

    #[test]
    fn high_water_mark_rows_fail_closed_on_conflicting_credited_economics() {
        let as_of = Utc
            .with_ymd_and_hms(2026, 7, 20, 12, 0, 0)
            .single()
            .unwrap();
        let conflict = DailyHighWaterMarkEvidenceRow {
            evidence_kind: "conflicting_credit".to_string(),
            settlement_id: None,
            order_id: "duplicated-order".to_string(),
            occurred_at: as_of - Duration::minutes(1),
            amount_usd: dec!(2),
            fill_ids: Vec::new(),
        };

        let error =
            daily_high_water_mark_state_from_rows(Uuid::from_u128(100), as_of, vec![conflict])
                .unwrap_err();
        assert!(error.to_string().contains("conflicting credited economics"));
        assert!(error.to_string().contains("duplicated-order"));
    }

    #[test]
    fn decision_edge_projection_preserves_legacy_reserve_contract() {
        let decision = approved_decision(BtcOutcome::Up, None, dec!(0.042));

        let projection = decision_edge_projection(&decision).unwrap();

        assert_eq!(projection.fair_probability, Some(dec!(0.78)));
        assert_eq!(projection.gross_edge_per_share, Some(dec!(0.06)));
        assert_eq!(projection.fee_per_share, Some(dec!(0.01)));
        assert_eq!(projection.reserve_per_share, Some(dec!(0.008)));
        assert_eq!(projection.net_edge_per_share, Some(dec!(0.042)));
    }

    #[test]
    fn decision_edge_projection_uses_directional_approval_contract_symmetrically() {
        for outcome in [BtcOutcome::Up, BtcOutcome::Down] {
            let prediction = BtcStrategyPrediction::DirectionalPrediction {
                outcome,
                probability: dec!(0.82),
                conservative_probability: dec!(0.78),
                minimum_conservative_probability: dec!(0.75),
                probability_uncertainty: dec!(0.04),
                executable_price: Some(dec!(0.72)),
                direct_taker_fee_per_share: Some(dec!(0.01)),
                direct_net_edge_per_share: Some(dec!(0.09)),
            };
            let decision = approved_decision(outcome, Some(prediction), dec!(0.09));

            let projection = decision_edge_projection(&decision).unwrap();

            assert_eq!(projection.fair_probability, Some(dec!(0.82)));
            assert_eq!(projection.gross_edge_per_share, Some(dec!(0.10)));
            assert_eq!(projection.fee_per_share, Some(dec!(0.01)));
            assert_eq!(projection.reserve_per_share, Some(Decimal::ZERO));
            assert_eq!(projection.net_edge_per_share, Some(dec!(0.09)));
            assert_eq!(
                projection.gross_edge_per_share.unwrap()
                    - projection.fee_per_share.unwrap()
                    - projection.reserve_per_share.unwrap(),
                projection.net_edge_per_share.unwrap()
            );
        }
    }

    #[test]
    fn resolution_subscription_ack_accepts_concurrent_terminal_transition() {
        let requested = vec![
            "market-a".to_string(),
            "market-b".to_string(),
            "market-c".to_string(),
        ];
        let subscribed = vec!["market-a".to_string(), "market-c".to_string()];
        let states = vec![
            ("market-a".to_string(), "pending".to_string()),
            ("market-b".to_string(), "resolved".to_string()),
            ("market-c".to_string(), "pending".to_string()),
        ];

        let acknowledgement =
            validate_official_resolution_subscription_ack(&requested, &subscribed, &states)
                .unwrap();

        assert_eq!(acknowledgement.subscribed, 2);
        assert_eq!(acknowledgement.already_terminal, 1);
    }

    #[test]
    fn resolution_subscription_ack_rejects_missing_durable_watch() {
        let requested = vec!["market-a".to_string(), "market-b".to_string()];
        let subscribed = vec!["market-a".to_string()];
        let states = vec![("market-a".to_string(), "pending".to_string())];

        let error = validate_official_resolution_subscription_ack(&requested, &subscribed, &states)
            .unwrap_err();

        assert!(error
            .to_string()
            .contains("missing durable watches for: market-b"));
    }

    #[test]
    fn resolution_subscription_ack_rejects_unacknowledged_pending_or_expired_watch() {
        for status in ["pending", "expired"] {
            let requested = vec!["market-a".to_string()];
            let states = vec![("market-a".to_string(), status.to_string())];

            let error = validate_official_resolution_subscription_ack(&requested, &[], &states)
                .unwrap_err();

            assert!(error
                .to_string()
                .contains(&format!("market-a={status}:not_updated")));
        }
    }

    #[test]
    fn serializes_database_enum_names() {
        assert_eq!(
            serde_name(&MarketFeedEventType::PriceChange).unwrap(),
            "price_change"
        );
        assert_eq!(
            serde_name(&FeedIntegrityStatus::PreSnapshot).unwrap(),
            "pre_snapshot"
        );
        assert_eq!(
            serde_name(&FeedIntegrityStatus::TopOfBookMismatch).unwrap(),
            "top_of_book_mismatch"
        );
        assert_eq!(outcome_name(BtcOutcome::Down), "down");
    }

    #[test]
    fn parses_fee_schedule_without_float_round_trip() {
        let value = serde_json::json!({
            "rate": "0.0200",
            "exponent": "2",
            "takerOnly": true
        });
        assert_eq!(decimal_json_field(&value, &["rate"]), Some(dec!(0.0200)));
        assert_eq!(integer_json_field(&value, &["exponent"]), Some(2));
        assert_eq!(bool_json_field(&value, &["takerOnly"]), Some(true));
    }

    #[test]
    fn saturates_ingest_sequence_for_postgres_bigint() {
        assert_eq!(sequence_i64(42), 42);
        assert_eq!(sequence_i64(u64::MAX), i64::MAX);
    }

    #[test]
    fn boundary_tick_identity_ignores_process_local_receipt_metadata() {
        let at = Utc::now();
        let mut left = ReferencePriceTick {
            tick_id: Uuid::new_v4(),
            dedup_key: "rtds_chainlink:BTCUSD:event".to_string(),
            source: ReferencePriceSource::RtdsChainlink,
            symbol: "BTCUSD".to_string(),
            price: dec!(64000),
            source_timestamp: at,
            envelope_timestamp: Some(at),
            received_at: at + Duration::milliseconds(10),
            connection_id: Uuid::new_v4(),
            ingest_sequence: 1,
            source_event_id: None,
            raw_payload: serde_json::json!({}),
        };
        let mut right = left.clone();
        right.received_at = at + Duration::milliseconds(25);
        right.connection_id = Uuid::new_v4();
        right.ingest_sequence = 9;
        assert!(same_boundary_tick(&left, &right));

        left.price = dec!(64001);
        assert!(!same_boundary_tick(&left, &right));
    }

    #[test]
    fn official_winner_requires_token_and_outcome_agreement() {
        assert_eq!(
            official_outcome_for_winner("up-token", "down-token", "up-token", "Up").unwrap(),
            BtcOutcome::Up
        );
        assert!(official_outcome_for_winner("up-token", "down-token", "up-token", "Down").is_err());
        assert!(
            official_outcome_for_winner("up-token", "down-token", "unknown-token", "Up").is_err()
        );
    }

    fn settlement_record() -> BtcPaperSettlementRecord {
        let at = Utc::now();
        BtcPaperSettlementRecord {
            settlement_id: Uuid::from_u128(1),
            experiment_id: Uuid::from_u128(2),
            process_id: Uuid::from_u128(3),
            order_id: "order".to_string(),
            market_id: "market".to_string(),
            token_id: "up-token".to_string(),
            fill_ids: serde_json::json!([Uuid::from_u128(4)]),
            official_outcome: "up".to_string(),
            official_winning_token_id: "up-token".to_string(),
            official_resolution_received_at: at,
            official_resolution_source: "clob_websocket".to_string(),
            filled_size: dec!(5),
            entry_notional: dec!(2),
            entry_fees: dec!(0.1),
            payout: dec!(5),
            net_pnl: dec!(2.9),
            credit_status: "pending".to_string(),
            credited_at: None,
            credit_attempts: 0,
            credit_evidence: serde_json::json!({}),
            created_at: at,
            updated_at: at,
        }
    }

    #[test]
    fn paper_settlement_contract_requires_official_provenance_fill_attribution_and_binary_payout() {
        let winner = settlement_record();
        validate_paper_settlement_record(&winner).unwrap();

        let mut loser = winner.clone();
        loser.token_id = "down-token".to_string();
        loser.payout = Decimal::ZERO;
        loser.net_pnl = dec!(-2.1);
        validate_paper_settlement_record(&loser).unwrap();

        let mut unsupported = winner.clone();
        unsupported.official_resolution_source = "local_chainlink_label".to_string();
        assert!(validate_paper_settlement_record(&unsupported).is_err());

        let mut missing_fill_attribution = winner.clone();
        missing_fill_attribution.fill_ids = serde_json::json!([]);
        assert!(validate_paper_settlement_record(&missing_fill_attribution).is_err());

        let mut negative = winner.clone();
        negative.payout = dec!(-1);
        assert!(validate_paper_settlement_record(&negative).is_err());

        let mut wrong_binary_payout = winner;
        wrong_binary_payout.payout = dec!(4.99);
        assert!(validate_paper_settlement_record(&wrong_binary_payout).is_err());

        let mut inconsistent_net_pnl = settlement_record();
        inconsistent_net_pnl.net_pnl = Decimal::ZERO;
        assert!(validate_paper_settlement_record(&inconsistent_net_pnl).is_err());
    }

    #[test]
    fn local_market_label_comparison_is_immutable() {
        let at = Utc::now();
        let label = BtcMarketLabel {
            market_id: "market".to_string(),
            window_start: at,
            window_end: at + Duration::minutes(5),
            open_price: dec!(100),
            close_price: dec!(101),
            outcome: BtcOutcome::Up,
            label_source: "rtds_chainlink".to_string(),
            label_version: "v1".to_string(),
            source_open_timestamp: at,
            source_close_timestamp: at + Duration::minutes(5),
            label_available_at: at + Duration::minutes(5),
            evidence: serde_json::json!({"open_tick_id": "open", "close_tick_id": "close"}),
        };
        assert!(same_immutable_market_label(&label, &label));
        let mut process_local_copy = label.clone();
        process_local_copy.label_available_at += Duration::milliseconds(25);
        process_local_copy.evidence = serde_json::json!({
            "open_tick_id": "open",
            "close_tick_id": "close",
            "open_received_at": "process-a",
            "close_received_at": "process-b"
        });
        assert!(same_immutable_market_label(&label, &process_local_copy));
        let mut changed = label.clone();
        changed.close_price = dec!(102);
        assert!(!same_immutable_market_label(&label, &changed));
    }
}
