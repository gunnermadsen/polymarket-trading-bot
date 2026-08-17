use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{FromRow, PgPool, Postgres, Transaction};
use uuid::Uuid;

use crate::models::{OrderRequest, OrderSide};

use super::{
    admission::{
        AdmissionDisposition, DailyRealizedPnlCredit, DailyRealizedPnlHighWaterMarkState,
        LossRegimeCandidate, ShadowPredictiveRegimeCandidate, ShadowPredictiveRegimeEvaluation,
        ShadowPredictiveRegimeState, UnsettledEntryExposure,
    },
    directional_model::BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION,
    execution_guard::reference_execution_guard,
    execution_lifecycle::BtcExecutionMode,
    predictive_regime_v2::{
        ShadowPredictiveRegimeV2Candidate, ShadowPredictiveRegimeV2CandidateSource,
        ShadowPredictiveRegimeV2Evaluation, ShadowPredictiveRegimeV2State,
        SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_V2_SCHEMA_VERSION,
    },
    strategy::{
        BtcDecision, BtcDecisionAction, BtcDirectionalModelEntryPolicy, BtcFeatureSnapshot,
        BtcStrategyPrediction, FairValueEstimate,
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

const LOAD_DIRECTIONAL_MODEL_BINANCE_TICK_SQL: &str = r#"
    SELECT tick_id, source_timestamp, received_at, source, symbol, price,
      envelope_timestamp, connection_id, ingest_sequence, source_event_id,
      dedup_key, raw_payload
    FROM polymarket.reference_price_ticks
    WHERE source = 'direct_binance'
      AND symbol = 'BTCUSD'
      AND integrity_status = 'ok'
      AND source_timestamp >= $1
      AND source_timestamp <= $2
      AND received_at >= $1
      AND received_at <= $2
    ORDER BY source_timestamp DESC, received_at DESC, ingest_sequence DESC, tick_id DESC
    LIMIT 1
    "#;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, FromRow)]
pub struct BtcRunManifest {
    pub process_id: Uuid,
    pub run_id: Uuid,
    pub run_key: String,
    pub config_hash: String,
    pub frozen_process_config: serde_json::Value,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BtcPaperVenueResumeState {
    pub entry_debits_usd: Decimal,
    pub settlement_credits_usd: Decimal,
    pub order_count: usize,
    pub fill_count: usize,
    pub credited_settlement_ids: Vec<Uuid>,
}

fn ensure_unambiguous_order_run_identity(
    process_id: Uuid,
    run_id: Uuid,
    has_identity_conflict: bool,
) -> Result<()> {
    if has_identity_conflict {
        bail!(
            "BTC process {process_id} run {run_id} has an order with conflicting current and legacy run identity"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct BtcSettlementRecord {
    pub process_id: Uuid,
    pub run_id: Uuid,
    pub settlement_id: Uuid,
    pub execution_mode: String,
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

/// Compatibility alias for callers that only operate the paper venue.
pub type BtcPaperSettlementRecord = BtcSettlementRecord;

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
struct ShadowPredictiveRegimeV2CandidateRow {
    market_id: String,
    order_id: String,
    order_client_order_id: Uuid,
    order_token_id: String,
    order_side: String,
    order_type: String,
    order_price: Decimal,
    order_size: Decimal,
    order_request: serde_json::Value,
    fill_id: Uuid,
    fill_at: DateTime<Utc>,
    decision_id: Option<Uuid>,
    decision_snapshot_id: Option<Uuid>,
    decision_action: Option<String>,
    decision_outcome: Option<String>,
    decision_fair_probability: Option<Decimal>,
    decision_prediction_status: Option<String>,
    decision_at: Option<DateTime<Utc>>,
    resolved_outcome: String,
    label_available_at: DateTime<Utc>,
    feature_snapshot_id: Option<Uuid>,
    feature_market_id: Option<String>,
    feature_as_of: Option<DateTime<Utc>>,
    feature_schema_version: Option<String>,
    feature_hash: Option<String>,
    fair_up_probability: Option<Decimal>,
    fair_up_lower: Option<Decimal>,
    fair_up_upper: Option<Decimal>,
}

#[derive(Debug, Clone, FromRow)]
struct PersistedShadowPredictiveRegimeV2EvaluationRow {
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
    l.execution_mode,
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
      execution_mode,
      filled_size,
      entry_notional,
      entry_fees,
      payout,
      net_pnl
    ))::numeric AS distinct_economics_count
  FROM causal_credited_rows
  GROUP BY process_id, order_id
  HAVING count(DISTINCT (
    execution_mode,
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
    l.execution_mode,
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
), process_execution_fills AS (
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
  WHERE f.source IN ('paper', 'live')
    AND f.timestamp_utc <= $2
  GROUP BY o.process_id, o.order_id
), unsettled_orders AS (
  SELECT f.process_id, f.order_id, f.last_filled_at, f.entry_debit_usd, f.fill_ids
  FROM process_execution_fills f
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

const RUN_MANIFEST_EXISTS_SQL: &str = r#"
SELECT EXISTS (
  SELECT 1
  FROM polymarket.trading_process_events
  WHERE event_type = 'btc_run_manifest'
    AND (
      event_id = $1
      OR metadata #>> '{run_id}' = $1::text
      OR metadata #>> '{run_key}' = $2
    )
)
"#;

const LOAD_RUN_MANIFEST_SQL: &str = r#"
SELECT
  process_id,
  event_id AS run_id,
  metadata #>> '{run_key}' AS run_key,
  metadata #>> '{config_hash}' AS config_hash,
  metadata #> '{frozen_process_config}' AS frozen_process_config
FROM polymarket.trading_process_events
WHERE process_id = $1
  AND event_type = 'btc_run_manifest'
  AND event_id = $2
  AND metadata #>> '{run_id}' = $2::text
  AND metadata #>> '{run_key}' = $3
ORDER BY timestamp_utc, created_at
LIMIT 2
"#;

const CLAIM_RUN_MANIFEST_LOCK_SQL: &str = r#"
SELECT pg_advisory_xact_lock(
  hashtextextended('polymarket.btc_run_manifest.start', 0)
)
"#;

const INSERT_RUN_MANIFEST_SQL: &str = r#"
INSERT INTO polymarket.trading_process_events (
  process_id, event_id, timestamp_utc, level, event_type, message, metadata, created_at
)
VALUES (
  $1, $2, now(), 'info', 'btc_run_manifest',
  'Immutable BTC execution run manifest', $3, now()
)
"#;

const PAPER_VENUE_RESUME_STATE_SQL: &str = r#"
WITH order_identity AS MATERIALIZED (
  SELECT
    o.order_id,
    o.raw_payload #>> '{request,metadata,run_id}' AS metadata_run_id,
    o.raw_payload #>> '{request,metadata,experiment_id}'
      AS legacy_experiment_id
  FROM polymarket.orders o
  WHERE o.process_id = $1
    AND (
      o.raw_payload #>> '{request,metadata,run_id}' = $2::text
      OR o.raw_payload #>> '{request,metadata,experiment_id}' = $2::text
    )
), identity_state AS (
  SELECT COALESCE(
    BOOL_OR(
      metadata_run_id IS NOT NULL
      AND legacy_experiment_id IS NOT NULL
      AND metadata_run_id <> legacy_experiment_id
    ),
    false
  ) AS has_identity_conflict
  FROM order_identity
), run_orders AS (
  SELECT order_id
  FROM order_identity
  WHERE NOT COALESCE(metadata_run_id <> legacy_experiment_id, false)
), fill_totals AS (
  SELECT
    COALESCE(SUM(f.price * f.size + f.fee), 0)::numeric AS entry_debits,
    COUNT(f.fill_id)::bigint AS fill_count
  FROM run_orders o
  LEFT JOIN polymarket.fills f
    ON f.process_id = $1
   AND f.order_id = o.order_id
   AND f.source = 'paper'
), order_totals AS (
  SELECT COUNT(*)::bigint AS order_count
  FROM run_orders
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
  WHERE process_id = $1
    AND run_id = $2
    AND execution_mode = 'paper'
)
SELECT f.entry_debits, s.settlement_credits,
       o.order_count, f.fill_count, s.settlement_ids,
       i.has_identity_conflict
FROM fill_totals f
CROSS JOIN order_totals o
CROSS JOIN settlement_totals s
CROSS JOIN identity_state i
"#;

const INSERT_STRATEGY_DECISION_SQL: &str = r#"
INSERT INTO polymarket.btc_strategy_decisions (
  process_id, run_id, decision_id, decision_at, market_id, snapshot_id,
  strategy_version, config_hash, action, outcome, token_id, fair_probability,
  executable_price, gross_edge_per_share, fee_per_share, reserve_per_share,
  net_edge_per_share, size, status, reject_reason, order_plan_id, execution_mode,
  metadata
)
VALUES (
  $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,
  $21,$22,$23
)
ON CONFLICT (decision_id, decision_at) DO NOTHING
"#;

const UPDATE_STRATEGY_DECISION_EXECUTION_SQL: &str = r#"
UPDATE polymarket.btc_strategy_decisions
SET status = $5,
    reject_reason = COALESCE($6, reject_reason),
    metadata = metadata || $7
WHERE process_id = $1
  AND run_id = $2
  AND decision_id = $3
  AND decision_at = $4
"#;

const AUTHORIZE_PENDING_STRATEGY_DECISION_SQL: &str = r#"
UPDATE polymarket.btc_strategy_decisions
SET status = 'approved'
WHERE process_id = $1
  AND run_id = $2
  AND decision_id = $3
  AND decision_at = $4
  AND status = 'execution_pending'
"#;

const DISCOVER_PENDING_SETTLEMENTS_SQL: &str = r#"
WITH order_identity AS MATERIALIZED (
  SELECT
    o.process_id,
    o.order_id,
    o.market_id,
    o.token_id,
    COALESCE(
      NULLIF(o.raw_payload #>> '{request,metadata,run_id}', ''),
      NULLIF(o.raw_payload #>> '{request,metadata,experiment_id}', '')
    )::uuid AS order_run_id,
    o.raw_payload #>> '{request,metadata,run_id}' AS metadata_run_id,
    o.raw_payload #>> '{request,metadata,experiment_id}'
      AS legacy_experiment_id
  FROM polymarket.orders o
  WHERE o.process_id = $1
    AND (
      ($3 = 'live' AND COALESCE(
        NULLIF(o.raw_payload #>> '{request,metadata,run_id}', ''),
        NULLIF(o.raw_payload #>> '{request,metadata,experiment_id}', '')
      ) IS NOT NULL)
      OR ($3 <> 'live' AND (
        o.raw_payload #>> '{request,metadata,run_id}' = $2::text
        OR o.raw_payload #>> '{request,metadata,experiment_id}' = $2::text
      ))
    )
), identity_state AS (
  SELECT COALESCE(
    BOOL_OR(
      metadata_run_id IS NOT NULL
      AND legacy_experiment_id IS NOT NULL
      AND metadata_run_id <> legacy_experiment_id
    ),
    false
  ) AS has_identity_conflict
  FROM order_identity
), entered AS (
  SELECT
    o.process_id,
    CASE WHEN $3 = 'live' THEN o.order_run_id ELSE $2::uuid END AS run_id,
    o.order_id,
    o.market_id,
    o.token_id,
    jsonb_agg(to_jsonb(f.fill_id) ORDER BY f.timestamp_utc, f.fill_id) AS fill_ids,
    round(sum(f.size), 10)::numeric(30,10) AS filled_size,
    round(sum(f.price * f.size), 10)::numeric(30,10) AS entry_notional,
    round(sum(f.fee), 10)::numeric(30,10) AS entry_fees
  FROM order_identity o
  JOIN polymarket.fills f
    ON f.process_id = $1
   AND f.order_id = o.order_id
   AND f.source = $3
  WHERE NOT COALESCE(
    o.metadata_run_id <> o.legacy_experiment_id,
    false
  )
  GROUP BY o.process_id, o.order_run_id, o.order_id, o.market_id, o.token_id
), eligible AS (
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
), inserted AS (
  INSERT INTO polymarket.btc_paper_settlement_ledger (
    process_id, run_id, execution_mode, order_id, market_id, token_id, fill_ids,
    official_outcome, official_winning_token_id,
    official_resolution_received_at, official_resolution_source,
    filled_size, entry_notional, entry_fees, payout, net_pnl
  )
  SELECT
    process_id, run_id, $3, order_id, market_id, token_id, fill_ids,
    official_outcome, official_winning_token_id,
    official_resolution_received_at, official_resolution_source,
    filled_size, entry_notional, entry_fees, payout,
    (payout - entry_notional - entry_fees)::numeric(30,10)
  FROM eligible
  WHERE NOT (SELECT has_identity_conflict FROM identity_state)
  ON CONFLICT (run_id, order_id) DO NOTHING
  RETURNING 1 AS inserted
)
SELECT
  (SELECT has_identity_conflict FROM identity_state) AS has_identity_conflict,
  (SELECT COUNT(*)::bigint FROM inserted) AS inserted_count
"#;

const LOAD_PENDING_SETTLEMENTS_SQL: &str = r#"
SELECT settlement_id, run_id, process_id, execution_mode, order_id, market_id, token_id,
  fill_ids,
  official_outcome, official_winning_token_id,
  official_resolution_received_at, official_resolution_source,
  filled_size, entry_notional, entry_fees, payout, net_pnl,
  credit_status, credited_at, credit_attempts, credit_evidence,
  created_at, updated_at
FROM polymarket.btc_paper_settlement_ledger
WHERE process_id = $1
  AND execution_mode = $3
  AND ($3 = 'live' OR run_id = $2)
  AND credit_status = 'pending'
ORDER BY official_resolution_received_at, order_id, settlement_id
"#;

const MARK_SETTLEMENT_RECOGNIZED_SQL: &str = r#"
UPDATE polymarket.btc_paper_settlement_ledger
SET credit_status = 'credited',
    credited_at = now(),
    credit_attempts = credit_attempts + 1,
    credit_evidence = $5,
    updated_at = now()
WHERE process_id = $1
  AND run_id = $2
  AND settlement_id = $3
  AND execution_mode = $4
  AND credit_status = 'pending'
"#;

const RECOGNIZE_LIVE_ZERO_PAYOUT_SETTLEMENT_SQL: &str = r#"
UPDATE polymarket.btc_paper_settlement_ledger settlement
SET credit_status = 'credited',
    credited_at = now(),
    credit_attempts = credit_attempts + 1,
    credit_evidence = $4,
    updated_at = now()
FROM polymarket.btc_interval_markets market
JOIN polymarket.btc_official_resolution_watches watch
  ON watch.market_id = market.market_id
 AND watch.status IN ('resolved', 'resolved_late')
 AND watch.resolution_received_at = market.official_resolution_received_at
 AND watch.resolution_source = market.official_resolution_source
WHERE settlement.process_id = $1
  AND settlement.run_id = $2
  AND settlement.settlement_id = $3
  AND settlement.execution_mode = 'live'
  AND settlement.credit_status = 'pending'
  AND settlement.payout = 0
  AND settlement.token_id <> settlement.official_winning_token_id
  AND market.market_id = settlement.market_id
  AND market.official_outcome = settlement.official_outcome
  AND market.official_winning_token_id = settlement.official_winning_token_id
  AND market.official_resolution_received_at = settlement.official_resolution_received_at
  AND market.official_resolution_source = settlement.official_resolution_source
"#;

const INSERT_ORDERBOOK_CHECKPOINT_PAIR_SQL: &str = r#"
INSERT INTO polymarket.orderbook_checkpoints (
  checkpoint_id, source_timestamp, received_at, connection_id, ingest_sequence,
  market_id, token_id, best_bid, best_ask, spread, tick_size, depth_bid, depth_ask,
  book, source_hash, bootstrap_source, integrity_status
)
VALUES
  ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17),
  ($18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34)
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

// One look-ahead row lets replay distinguish an exact 10,000-candidate history from truncation.
const MAX_SHADOW_PREDICTIVE_REGIME_V2_HISTORY_CANDIDATES: u32 = 10_001;

const LATEST_SHADOW_PREDICTIVE_REGIME_V2_STATE_SQL: &str = r#"
SELECT
  d.decision_at,
  d.metadata #> '{entry_admission,shadow_predictive_regime_circuit_breaker}' AS evaluation
FROM polymarket.btc_strategy_decisions d
WHERE d.process_id = $1
  AND d.metadata #>> '{entry_admission,shadow_predictive_regime_circuit_breaker,schema_version}' = 'shadow_predictive_regime_circuit_breaker_v2'
  AND d.metadata #>> '{entry_admission,shadow_predictive_regime_circuit_breaker,mode}' = 'shadow'
  AND ($2::text IS NULL OR d.metadata #>> '{entry_admission,shadow_predictive_regime_circuit_breaker,state,config_hash}' = $2)
  AND d.decision_at <= $3
  AND d.action = 'buy'
  AND d.metadata #>> '{entry_admission,shadow_predictive_regime_circuit_breaker,state_checkpoint_eligible}' = 'true'
  AND COALESCE(
    d.metadata #>> '{entry_admission,shadow_predictive_regime_circuit_breaker,refresh_pending}',
    'false'
  ) = 'false'
  AND COALESCE(
    d.metadata #> '{entry_admission,shadow_predictive_regime_circuit_breaker,telemetry_error}',
    'null'::jsonb
  ) = 'null'::jsonb
  AND jsonb_typeof(
    d.metadata #> '{entry_admission,shadow_predictive_regime_circuit_breaker}'
  ) = 'object'
ORDER BY d.decision_at DESC, d.decision_id DESC
LIMIT 1
"#;

const SHADOW_PREDICTIVE_REGIME_V2_CANDIDATES_SQL: &str = r#"
WITH process_boundary AS MATERIALIZED (
  SELECT created_at
  FROM polymarket.trading_processes
  WHERE process_id = $1
), first_process_fills AS MATERIALIZED (
  SELECT DISTINCT ON (f.order_id)
    f.order_id,
    f.fill_id,
    f.timestamp_utc AS fill_at
  FROM polymarket.fills f
  CROSS JOIN process_boundary boundary
  WHERE f.process_id = $1
    AND f.source = 'paper'
    AND f.size > 0
    AND f.timestamp_utc >= boundary.created_at
    AND f.timestamp_utc <= $2
  ORDER BY f.order_id, f.timestamp_utc, f.fill_id
), first_market_exposures AS MATERIALIZED (
  SELECT DISTINCT ON (o.market_id)
    o.market_id,
    o.order_id,
    o.client_order_id AS order_client_order_id,
    o.token_id AS order_token_id,
    o.side AS order_side,
    o.order_type,
    o.price AS order_price,
    o.size AS order_size,
    COALESCE(o.raw_payload #> '{request}', 'null'::jsonb) AS order_request,
    fill.fill_id,
    fill.fill_at,
    boundary.created_at AS process_created_at
  FROM first_process_fills fill
  JOIN polymarket.orders o
    ON o.process_id = $1
   AND o.order_id = fill.order_id
  CROSS JOIN process_boundary boundary
  WHERE o.created_at >= boundary.created_at
    AND o.created_at <= fill.fill_at
    AND o.side = 'buy'
    AND o.order_type = 'fok'
    AND o.raw_payload #>> '{request,metadata,execution_intent}' = 'entry'
    AND o.raw_payload #>> '{request,metadata,paper_execution,non_mutating_preview}' = 'false'
  ORDER BY o.market_id, fill.fill_at, fill.fill_id, o.order_id
), resolved_exposures AS MATERIALIZED (
  SELECT
    exposure.*,
    label.outcome AS resolved_outcome,
    label.label_available_at
  FROM first_market_exposures exposure
  JOIN polymarket.btc_market_labels label
    ON label.market_id = exposure.market_id
   AND label.label_available_at <= $2
), linked_evidence AS (
  SELECT
    exposure.market_id,
    exposure.order_id,
    exposure.order_client_order_id,
    exposure.order_token_id,
    exposure.order_side,
    exposure.order_type,
    exposure.order_price,
    exposure.order_size,
    exposure.order_request,
    exposure.fill_id,
    exposure.fill_at,
    decision.decision_id,
    decision.snapshot_id AS decision_snapshot_id,
    decision.action AS decision_action,
    decision.outcome AS decision_outcome,
    decision.fair_probability AS decision_fair_probability,
    decision.metadata #>> '{prediction,status}' AS decision_prediction_status,
    decision.decision_at,
    exposure.resolved_outcome,
    exposure.label_available_at,
    feature.snapshot_id AS feature_snapshot_id,
    feature.market_id AS feature_market_id,
    feature.feature_as_of,
    feature.feature_schema_version,
    feature.feature_hash,
    feature.fair_up_probability,
    feature.fair_up_lower,
    feature.fair_up_upper
  FROM resolved_exposures exposure
  LEFT JOIN polymarket.btc_strategy_decisions decision
    ON decision.process_id = $1
   AND decision.market_id = exposure.market_id
   AND decision.decision_id = CASE
     WHEN exposure.order_request #>> '{metadata,reference_execution_guard,decision_id}'
       ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
     THEN (exposure.order_request #>> '{metadata,reference_execution_guard,decision_id}')::uuid
     ELSE NULL::uuid
   END
   AND decision.action = 'buy'
   AND decision.outcome IN ('up', 'down')
   AND decision.fair_probability BETWEEN 0 AND 1
   AND decision.execution_mode = 'paper'
   AND decision.decision_at <= exposure.fill_at
   AND exposure.fill_at < exposure.label_available_at
  LEFT JOIN LATERAL (
    SELECT
      feature.snapshot_id,
      feature.market_id,
      feature.feature_as_of,
      feature.feature_schema_version,
      feature.feature_hash,
      feature.fair_up_probability,
      feature.fair_up_lower,
      feature.fair_up_upper
    FROM polymarket.btc_feature_snapshots feature
    WHERE feature.snapshot_id = decision.snapshot_id
      AND feature.market_id = exposure.market_id
      AND feature.features #>> '{process_id}' = $1::text
      AND feature.feature_as_of = decision.decision_at
      AND feature.feature_as_of >= exposure.process_created_at
      AND feature.feature_as_of <= $2
      AND feature.fair_up_probability BETWEEN 0 AND 1
    OFFSET 0
  ) feature ON true
), bounded AS (
  SELECT *
  FROM linked_evidence
  ORDER BY label_available_at DESC, fill_at DESC, fill_id DESC, market_id DESC
  LIMIT $3
)
SELECT *
FROM bounded
ORDER BY label_available_at, fill_at, fill_id, market_id
"#;

#[derive(Debug, Clone, FromRow)]
struct CheckpointRow {
    checkpoint_id: Uuid,
    source_timestamp: DateTime<Utc>,
    received_at: DateTime<Utc>,
    observed_at: DateTime<Utc>,
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

    /// Durably records both outcome books from one ready CLOB connection before publication.
    /// Validation happens before execution, and PostgreSQL applies the fixed two-row insert as
    /// one atomic statement, so a failure cannot commit only one outcome checkpoint.
    pub async fn insert_orderbook_checkpoint_pair(
        &self,
        checkpoints: &[OrderbookCheckpoint],
        bootstrap_source: &str,
        publication_boundary: DateTime<Utc>,
    ) -> Result<()> {
        let (first, second) =
            validate_orderbook_checkpoint_pair(checkpoints, publication_boundary)?;
        let first_depth_bid: Decimal = first.bids.iter().map(|level| level.size).sum();
        let first_depth_ask: Decimal = first.asks.iter().map(|level| level.size).sum();
        let first_spread = first
            .best_bid
            .zip(first.best_ask)
            .map(|(bid, ask)| ask - bid);
        let first_book = serde_json::json!({
            "bids": &first.bids,
            "asks": &first.asks,
        });
        let second_depth_bid: Decimal = second.bids.iter().map(|level| level.size).sum();
        let second_depth_ask: Decimal = second.asks.iter().map(|level| level.size).sum();
        let second_spread = second
            .best_bid
            .zip(second.best_ask)
            .map(|(bid, ask)| ask - bid);
        let second_book = serde_json::json!({
            "bids": &second.bids,
            "asks": &second.asks,
        });
        let integrity_status = serde_name(&FeedIntegrityStatus::Ok)?;

        let result = sqlx::query(INSERT_ORDERBOOK_CHECKPOINT_PAIR_SQL)
            .bind(first.checkpoint_id)
            .bind(first.source_timestamp)
            .bind(first.received_at)
            .bind(first.connection_id)
            .bind(sequence_i64(first.ingest_sequence))
            .bind(&first.market_id)
            .bind(&first.token_id)
            .bind(first.best_bid)
            .bind(first.best_ask)
            .bind(first_spread)
            .bind(first.tick_size)
            .bind(first_depth_bid)
            .bind(first_depth_ask)
            .bind(first_book)
            .bind(&first.source_hash)
            .bind(bootstrap_source)
            .bind(&integrity_status)
            .bind(second.checkpoint_id)
            .bind(second.source_timestamp)
            .bind(second.received_at)
            .bind(second.connection_id)
            .bind(sequence_i64(second.ingest_sequence))
            .bind(&second.market_id)
            .bind(&second.token_id)
            .bind(second.best_bid)
            .bind(second.best_ask)
            .bind(second_spread)
            .bind(second.tick_size)
            .bind(second_depth_bid)
            .bind(second_depth_ask)
            .bind(second_book)
            .bind(&second.source_hash)
            .bind(bootstrap_source)
            .bind(&integrity_status)
            .execute(&self.pool)
            .await
            .context("failed to insert BTC orderbook checkpoint pair")?;
        if result.rows_affected() != 2 {
            bail!(
                "BTC orderbook checkpoint-pair insert affected {} rows",
                result.rows_affected()
            );
        }
        Ok(())
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
        if !is_supported_official_resolution_source(resolution_source) {
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
    /// `as_of`. This is the execution runner's no-lookahead boundary.
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
        let fee = self.load_market_fee_as_of(&market.market_id, as_of).await?;

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

    /// Loads the market's immutable official Chainlink opening reference without rebuilding
    /// reference histories. The receipt-time and configured boundary-window predicates preserve
    /// the caller's point-in-time boundary.
    pub async fn load_market_opening_reference(
        &self,
        market: &BtcIntervalMarket,
        feature_as_of: DateTime<Utc>,
        max_chainlink_open_delay: chrono::Duration,
    ) -> Result<Option<ReferencePriceTick>> {
        if max_chainlink_open_delay <= Duration::zero() {
            bail!("Chainlink opening-reference delay must be positive");
        }
        let latest_valid_open = market
            .window_start
            .checked_add_signed(max_chainlink_open_delay)
            .context("Chainlink opening window exceeds the timestamp range")?;
        sqlx::query_as::<_, ReferenceTickRow>(
            r#"
            SELECT t.tick_id, t.source_timestamp, t.received_at, t.source, t.symbol, t.price,
              t.envelope_timestamp, t.connection_id, t.ingest_sequence, t.source_event_id,
              t.dedup_key, t.raw_payload
            FROM polymarket.btc_interval_markets m
            JOIN LATERAL (
              SELECT tick_id, source_timestamp, received_at, source, symbol, price,
                envelope_timestamp, connection_id, ingest_sequence, source_event_id,
                dedup_key, raw_payload
              FROM polymarket.reference_price_ticks
              WHERE source = 'rtds_chainlink'
                AND symbol = 'BTCUSD'
                AND integrity_status = 'ok'
                AND source_timestamp = m.reference_source_timestamp
                AND price = m.reference_price
                AND received_at <= $2
              ORDER BY received_at ASC, ingest_sequence ASC, tick_id ASC
              LIMIT 1
            ) t ON true
            WHERE m.market_id = $1
              AND m.reference_price IS NOT NULL
              AND m.reference_source_timestamp >= $3
              AND m.reference_source_timestamp <= $4
            "#,
        )
        .bind(&market.market_id)
        .bind(feature_as_of)
        .bind(market.window_start)
        .bind(latest_valid_open)
        .fetch_optional(&self.pool)
        .await
        .context("failed to load market Chainlink opening reference")?
        .map(reference_tick_from_row)
        .transpose()
    }

    /// Hydrates the bounded RTDS midpoint history used to construct closed Chainlink candles.
    /// Duplicate reconnect deliveries are collapsed by source timestamp, retaining the first
    /// causally received value exactly as the historical training source does.
    pub(crate) async fn load_directional_external_chainlink_mid_history(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<ReferencePriceTick>> {
        if start >= end || (end - start) > chrono::Duration::hours(2) {
            bail!("directional Chainlink midpoint bootstrap range is invalid");
        }
        let rows = sqlx::query_as::<_, ReferenceTickRow>(
            r#"
            SELECT tick_id, source_timestamp, received_at, source, symbol, price,
              envelope_timestamp, connection_id, ingest_sequence, source_event_id,
              dedup_key, raw_payload
            FROM (
              SELECT DISTINCT ON (source_timestamp)
                tick_id, source_timestamp, received_at, source, symbol, price,
                envelope_timestamp, connection_id, ingest_sequence, source_event_id,
                dedup_key, raw_payload
              FROM polymarket.reference_price_ticks
              WHERE source = 'rtds_chainlink'
                AND symbol = 'BTCUSD'
                AND integrity_status = 'ok'
                AND source_timestamp >= $1
                AND source_timestamp <= $2
                AND received_at <= $2
              ORDER BY source_timestamp ASC, received_at ASC, ingest_sequence ASC, tick_id ASC
            ) history
            ORDER BY source_timestamp ASC
            LIMIT 5000
            "#,
        )
        // SQLx executes persistent(false) through the unnamed statement slot across a
        // Parse/Sync and Bind/Execute boundary. PgBouncer must retain session affinity
        // for this query; transaction pooling can move Bind to a different backend.
        .persistent(false)
        .bind(start)
        .bind(end)
        .fetch_all(&self.pool)
        .await
        .context("failed to hydrate directional Chainlink midpoint history")?;
        rows.into_iter().map(reference_tick_from_row).collect()
    }

    /// Loads only the immutable execution evidence needed after native directional-model
    /// inference. Model features come from the runtime's bounded one-second Binance window, so
    /// this path deliberately avoids rebuilding unused reference histories from Postgres.
    pub(crate) async fn load_directional_model_execution_inputs(
        &self,
        market: &BtcIntervalMarket,
        as_of: DateTime<Utc>,
        max_reference_age: chrono::Duration,
        max_book_age: chrono::Duration,
        clob_connection_id: Option<Uuid>,
    ) -> Result<BtcPointInTimeInputs> {
        if max_reference_age <= Duration::zero() || max_book_age <= Duration::zero() {
            bail!("directional-model execution input freshness bounds must be positive");
        }
        let reference_fresh_since = as_of.checked_sub_signed(max_reference_age).context(
            "directional-model reference freshness bound is outside the timestamp range",
        )?;
        let book_fresh_since = as_of
            .checked_sub_signed(max_book_age)
            .context("directional-model book freshness bound is outside the timestamp range")?;

        let (binance_current, up_book, down_book, fee) =
            if let Some(connection_id) = clob_connection_id {
                let (binance, up, down, fee) = tokio::try_join!(
                    self.load_directional_model_binance_tick(reference_fresh_since, as_of),
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
                    self.load_market_fee_as_of(&market.market_id, as_of),
                )?;
                (binance, up, down, fee)
            } else {
                let (binance, fee) = tokio::try_join!(
                    self.load_directional_model_binance_tick(reference_fresh_since, as_of),
                    self.load_market_fee_as_of(&market.market_id, as_of),
                )?;
                (binance, None, None, fee)
            };

        Ok(BtcPointInTimeInputs {
            chainlink_open: None,
            chainlink_current: None,
            chainlink_history: Vec::new(),
            binance_history: binance_current.into_iter().collect(),
            up_book,
            down_book,
            fee_rate: fee.as_ref().and_then(|(rate, _)| *rate),
            fee_observed_at: fee.map(|(_, observed_at)| observed_at),
        })
    }

    async fn load_directional_model_binance_tick(
        &self,
        fresh_since: DateTime<Utc>,
        as_of: DateTime<Utc>,
    ) -> Result<Option<ReferencePriceTick>> {
        sqlx::query_as::<_, ReferenceTickRow>(LOAD_DIRECTIONAL_MODEL_BINANCE_TICK_SQL)
            .bind(fresh_since)
            .bind(as_of)
            .fetch_optional(&self.pool)
            .await
            .context("failed to load directional-model point-in-time Binance tick")?
            .map(reference_tick_from_row)
            .transpose()
    }

    async fn load_market_fee_as_of(
        &self,
        market_id: &str,
        as_of: DateTime<Utc>,
    ) -> Result<Option<(Option<Decimal>, DateTime<Utc>)>> {
        sqlx::query_as::<_, (Option<Decimal>, DateTime<Utc>)>(
            r#"
            SELECT fee_rate, last_refreshed_at
            FROM polymarket.btc_interval_markets
            WHERE market_id = $1
              AND last_refreshed_at <= $2
            "#,
        )
        .bind(market_id)
        .bind(as_of)
        .fetch_optional(&self.pool)
        .await
        .context("failed to load BTC market fee schedule")
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
            SELECT checkpoint_id, source_timestamp, received_at, persisted_at AS observed_at,
              connection_id,
              ingest_sequence, market_id, token_id, best_bid, best_ask, tick_size,
              book, source_hash, integrity_status
            FROM polymarket.orderbook_checkpoints
            WHERE token_id = $1
              AND source_timestamp >= $3 - INTERVAL '1 hour'
              AND source_timestamp <= $3
              AND received_at <= $3
              AND persisted_at >= $2
              AND persisted_at <= $3
              AND connection_id = $4
              AND integrity_status = 'ok'
            ORDER BY persisted_at DESC, source_timestamp DESC, received_at DESC, ingest_sequence DESC,
              checkpoint_id DESC
            LIMIT 1
            "#,
        )
        // A cached PostgreSQL generic plan expands this Timescale hypertable across every
        // compressed and uncompressed chunk before runtime exclusion. Keep this statement
        // custom-planned so the timestamp bounds prune chunks before relation locks are taken.
        // SQLx implements this with an unnamed statement across a protocol synchronization
        // boundary, so PgBouncer must use session pooling unless this query is redesigned.
        .persistent(false)
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

    pub async fn run_manifest_exists(&self, run_id: Uuid, run_key: &str) -> Result<bool> {
        if run_key.trim().is_empty() {
            bail!("BTC run key must not be empty");
        }
        sqlx::query_scalar::<_, bool>(RUN_MANIFEST_EXISTS_SQL)
            .bind(run_id)
            .bind(run_key)
            .fetch_one(&self.pool)
            .await
            .context("failed to check global BTC run-manifest identity")
    }

    pub async fn load_run_manifest(
        &self,
        process_id: Uuid,
        run_id: Uuid,
        run_key: &str,
    ) -> Result<Option<BtcRunManifest>> {
        if run_key.trim().is_empty() {
            bail!("BTC run key must not be empty");
        }
        let manifests = sqlx::query_as::<_, BtcRunManifest>(LOAD_RUN_MANIFEST_SQL)
            .bind(process_id)
            .bind(run_id)
            .bind(run_key)
            .fetch_all(&self.pool)
            .await
            .context("failed to load immutable BTC run manifest")?;
        match manifests.len() {
            0 => Ok(None),
            1 => Ok(manifests.into_iter().next()),
            count => bail!(
                "BTC run manifest identity matched {count} immutable events for process {process_id}"
            ),
        }
    }

    pub async fn run_config_hash(&self, process_id: Uuid, run_id: Uuid) -> Result<String> {
        let hashes = sqlx::query_scalar::<_, String>(
            r#"
            SELECT metadata #>> '{config_hash}'
            FROM polymarket.trading_process_events
            WHERE process_id = $1
              AND event_type = 'btc_run_manifest'
              AND event_id = $2
              AND metadata #>> '{run_id}' = $2::text
            ORDER BY timestamp_utc, created_at
            LIMIT 2
            "#,
        )
        .bind(process_id)
        .bind(run_id)
        .fetch_all(&self.pool)
        .await
        .context("failed to load immutable BTC run config hash")?;
        match hashes.as_slice() {
            [config_hash]
                if config_hash.len() == 64
                    && config_hash.bytes().all(|byte| byte.is_ascii_hexdigit()) =>
            {
                Ok(config_hash.clone())
            }
            [] => bail!("BTC process {process_id} run {run_id} has no immutable run manifest"),
            [_] => bail!("BTC process {process_id} run {run_id} has an invalid config hash"),
            _ => bail!("BTC process {process_id} run {run_id} has duplicate run manifests"),
        }
    }

    pub async fn verify_run_manifest(
        &self,
        process_id: Uuid,
        run_id: Uuid,
        run_key: &str,
        config_hash: &str,
        frozen_process_config: &serde_json::Value,
    ) -> Result<()> {
        validate_run_manifest(run_key, config_hash, frozen_process_config)?;
        let manifest = self
            .load_run_manifest(process_id, run_id, run_key)
            .await?
            .with_context(|| {
                format!("BTC run manifest {run_id} is missing for canonical process {process_id}")
            })?;
        if manifest.config_hash != config_hash
            || manifest.frozen_process_config != *frozen_process_config
        {
            bail!(
                "BTC run manifest cannot resume because its immutable frozen configuration no longer matches"
            );
        }
        Ok(())
    }

    pub async fn claim_run_manifest(
        &self,
        process_id: Uuid,
        run_id: Uuid,
        run_key: &str,
        config_hash: &str,
        frozen_process_config: &serde_json::Value,
    ) -> Result<()> {
        validate_run_manifest(run_key, config_hash, frozen_process_config)?;
        let mut tx = self
            .pool
            .begin()
            .await
            .context("failed to begin BTC run-manifest claim transaction")?;
        sqlx::query(CLAIM_RUN_MANIFEST_LOCK_SQL)
            .execute(&mut *tx)
            .await
            .context("failed to acquire global BTC run-start lock")?;
        let duplicate = sqlx::query_scalar::<_, bool>(RUN_MANIFEST_EXISTS_SQL)
            .bind(run_id)
            .bind(run_key)
            .fetch_one(&mut *tx)
            .await
            .context("failed to check BTC run-manifest identity under the start lock")?;
        if duplicate {
            bail!(
                "BTC run identity {run_id}/{run_key} already exists; every explicit start requires a globally unique id and key"
            );
        }
        let metadata = serde_json::json!({
            "run_id": run_id,
            "run_key": run_key,
            "config_hash": config_hash,
            "frozen_process_config": frozen_process_config,
        });
        let inserted = sqlx::query(INSERT_RUN_MANIFEST_SQL)
            .bind(process_id)
            .bind(run_id)
            .bind(metadata)
            .execute(&mut *tx)
            .await
            .context("failed to persist immutable BTC run manifest")?;
        if inserted.rows_affected() != 1 {
            bail!("immutable BTC run-manifest insert did not write exactly one event");
        }
        tx.commit()
            .await
            .context("failed to commit BTC run-manifest claim transaction")?;
        Ok(())
    }

    pub async fn paper_venue_resume_state(
        &self,
        process_id: Uuid,
        run_id: Uuid,
    ) -> Result<BtcPaperVenueResumeState> {
        let (
            entry_debits,
            settlement_credits,
            order_count,
            fill_count,
            settlement_ids,
            has_identity_conflict,
        ) = sqlx::query_as::<_, (Decimal, Decimal, i64, i64, Vec<Uuid>, bool)>(
            PAPER_VENUE_RESUME_STATE_SQL,
        )
        .bind(process_id)
        .bind(run_id)
        .fetch_one(&self.pool)
        .await
        .context("failed to load BTC paper venue resume state")?;
        ensure_unambiguous_order_run_identity(process_id, run_id, has_identity_conflict)?;
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
    /// process-config hash. Execution status and run identity do not own this state.
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

    /// Loads a bounded, causal history of the first actual paper exposure in each market.
    /// Exposure ownership is exclusively the stable trading process; run identity and
    /// decision status never participate in candidate selection.
    pub async fn load_shadow_predictive_regime_v2_candidates(
        &self,
        process_id: Uuid,
        as_of: DateTime<Utc>,
        max_candidates: u32,
        expected_max_reference_age: Duration,
        expected_max_directional_feature_age: Option<Duration>,
    ) -> Result<Vec<ShadowPredictiveRegimeV2Candidate>> {
        if process_id.is_nil() {
            bail!("shadow predictive-regime v2 candidate process_id cannot be nil");
        }
        if expected_max_reference_age <= Duration::zero() {
            bail!("shadow predictive-regime v2 expected reference age must be positive");
        }
        let limit = shadow_predictive_regime_v2_history_limit(max_candidates)?;
        let rows = sqlx::query_as::<_, ShadowPredictiveRegimeV2CandidateRow>(
            SHADOW_PREDICTIVE_REGIME_V2_CANDIDATES_SQL,
        )
        .bind(process_id)
        .bind(as_of)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .context("failed to load shadow predictive-regime v2 exposure candidates")?;

        rows.into_iter()
            .map(|row| {
                shadow_predictive_regime_v2_candidate_from_row(
                    process_id,
                    expected_max_reference_age,
                    expected_max_directional_feature_age,
                    row,
                )
            })
            .collect()
    }

    /// Loads the newest causally persisted V2 checkpoint owned by this trading process. V1
    /// evaluations are excluded by the durable schema discriminator before deserialization.
    pub async fn load_latest_shadow_predictive_regime_v2_state(
        &self,
        process_id: Uuid,
        breaker_state_config_hash: Option<&str>,
        as_of: DateTime<Utc>,
    ) -> Result<Option<ShadowPredictiveRegimeV2State>> {
        if process_id.is_nil() {
            bail!("shadow predictive-regime v2 state process_id cannot be nil");
        }
        if breaker_state_config_hash.is_some_and(|config_hash| config_hash.trim().is_empty()) {
            bail!("shadow predictive-regime v2 state config hash cannot be empty");
        }
        let row = sqlx::query_as::<_, PersistedShadowPredictiveRegimeV2EvaluationRow>(
            LATEST_SHADOW_PREDICTIVE_REGIME_V2_STATE_SQL,
        )
        .bind(process_id)
        .bind(breaker_state_config_hash)
        .bind(as_of)
        .fetch_optional(&self.pool)
        .await
        .context("failed to load latest shadow predictive-regime v2 state")?;

        row.map(|row| {
            shadow_predictive_regime_v2_state_from_persisted_evaluation(
                process_id,
                breaker_state_config_hash,
                row,
            )
        })
        .transpose()
    }

    /// Reconstructs the causal UTC-day realized-PnL watermark and every still-unsettled paper or
    /// live entry for one stable trading process. This state is deliberately process-owned: it
    /// carries across immutable execution runs and must never be filtered by run identity.
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
        process_id: Uuid,
        run_id: Uuid,
        config_hash: &str,
        market_id: &str,
        strategy_version: &str,
        decision: &BtcDecision,
        entry_admission_evidence: Option<&serde_json::Value>,
        order_plan_id: Option<Uuid>,
        execution_mode: BtcExecutionMode,
        status: &str,
    ) -> Result<bool> {
        if decision.process_id != process_id {
            bail!("BTC strategy decision process ownership does not match its repository scope");
        }
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
        let inserted = sqlx::query(INSERT_STRATEGY_DECISION_SQL)
            .bind(process_id)
            .bind(run_id)
            .bind(decision.decision_id)
            .bind(decision.evaluated_at)
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
            .bind(execution_mode.as_str())
            .bind(metadata)
            .execute(&self.pool)
            .await
            .context("failed to insert BTC strategy decision")?;
        Ok(inserted.rows_affected() == 1)
    }

    pub async fn update_strategy_decision_execution(
        &self,
        process_id: Uuid,
        run_id: Uuid,
        decision_id: Uuid,
        decision_at: DateTime<Utc>,
        status: &str,
        reject_reason: Option<&str>,
        metadata: serde_json::Value,
    ) -> Result<()> {
        sqlx::query(UPDATE_STRATEGY_DECISION_EXECUTION_SQL)
            .bind(process_id)
            .bind(run_id)
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

    pub async fn authorize_pending_strategy_decision(
        &self,
        process_id: Uuid,
        run_id: Uuid,
        decision_id: Uuid,
        decision_at: DateTime<Utc>,
    ) -> Result<()> {
        let result = sqlx::query(AUTHORIZE_PENDING_STRATEGY_DECISION_SQL)
            .bind(process_id)
            .bind(run_id)
            .bind(decision_id)
            .bind(decision_at)
            .execute(&self.pool)
            .await
            .context("failed to authorize pending BTC strategy decision")?;
        if result.rows_affected() != 1 {
            bail!("pending BTC strategy decision authorization did not update exactly one row");
        }
        Ok(())
    }

    /// Materializes every newly eligible official settlement into a durable, idempotent ledger
    /// and returns records still awaiting venue-specific recognition. Fill selection is exact for
    /// the requested execution source. Eligibility requires the immutable official market fact
    /// and its matching durable resolution watch.
    pub async fn discover_pending_settlements(
        &self,
        process_id: Uuid,
        run_id: Uuid,
        execution_mode: BtcExecutionMode,
    ) -> Result<Vec<BtcSettlementRecord>> {
        let (has_identity_conflict, _inserted_count) =
            sqlx::query_as::<_, (bool, i64)>(DISCOVER_PENDING_SETTLEMENTS_SQL)
                .bind(process_id)
                .bind(run_id)
                .bind(execution_mode.as_str())
                .fetch_one(&self.pool)
                .await
                .with_context(|| {
                    format!(
                        "failed to discover durable BTC {} settlements",
                        execution_mode.as_str()
                    )
                })?;
        ensure_unambiguous_order_run_identity(process_id, run_id, has_identity_conflict)?;

        let records = sqlx::query_as::<_, BtcSettlementRecord>(LOAD_PENDING_SETTLEMENTS_SQL)
            .bind(process_id)
            .bind(run_id)
            .bind(execution_mode.as_str())
            .fetch_all(&self.pool)
            .await
            .with_context(|| {
                format!(
                    "failed to load pending BTC {} settlements",
                    execution_mode.as_str()
                )
            })?;
        for record in &records {
            validate_settlement_record(record, execution_mode)?;
        }
        Ok(records)
    }

    /// Compatibility wrapper for the existing paper lifecycle.
    pub async fn discover_pending_paper_settlements(
        &self,
        process_id: Uuid,
        run_id: Uuid,
    ) -> Result<Vec<BtcPaperSettlementRecord>> {
        self.discover_pending_settlements(process_id, run_id, BtcExecutionMode::Paper)
            .await
    }

    pub async fn recognize_live_zero_payout_settlement(
        &self,
        process_id: Uuid,
        run_id: Uuid,
        settlement: &BtcSettlementRecord,
        config_hash: &str,
    ) -> Result<bool> {
        if settlement.process_id != process_id || settlement.run_id != run_id {
            bail!("live zero-payout settlement ownership does not match its process run");
        }
        let evidence = live_zero_payout_settlement_evidence(settlement, config_hash)?;
        let result = sqlx::query(RECOGNIZE_LIVE_ZERO_PAYOUT_SETTLEMENT_SQL)
            .bind(process_id)
            .bind(run_id)
            .bind(settlement.settlement_id)
            .bind(&evidence)
            .execute(&self.pool)
            .await
            .context("failed to recognize exact live zero-payout settlement")?;
        if result.rows_affected() == 1 {
            return Ok(true);
        }

        let existing = sqlx::query_as::<_, (String, serde_json::Value)>(
            r#"
            SELECT credit_status, credit_evidence
            FROM polymarket.btc_paper_settlement_ledger
            WHERE process_id = $1
              AND run_id = $2
              AND settlement_id = $3
              AND execution_mode = 'live'
            "#,
        )
        .bind(process_id)
        .bind(run_id)
        .bind(settlement.settlement_id)
        .fetch_optional(&self.pool)
        .await
        .context("failed to verify idempotent live zero-payout settlement recognition")?;
        match existing {
            Some((status, existing_evidence))
                if status == "credited" && existing_evidence == evidence =>
            {
                Ok(false)
            }
            Some(_) => {
                bail!("live zero-payout settlement recognition conflicts with durable state")
            }
            None => bail!("live zero-payout settlement disappeared during recognition"),
        }
    }

    pub async fn mark_settlement_recognized(
        &self,
        process_id: Uuid,
        run_id: Uuid,
        settlement_id: Uuid,
        execution_mode: BtcExecutionMode,
        recognition_evidence: &serde_json::Value,
    ) -> Result<bool> {
        validate_settlement_recognition_mode(execution_mode)?;
        if !recognition_evidence.is_object() {
            bail!(
                "{} settlement recognition evidence must be a JSON object",
                execution_mode.as_str()
            );
        }
        let result = sqlx::query(MARK_SETTLEMENT_RECOGNIZED_SQL)
            .bind(process_id)
            .bind(run_id)
            .bind(settlement_id)
            .bind(execution_mode.as_str())
            .bind(recognition_evidence)
            .execute(&self.pool)
            .await
            .with_context(|| {
                format!(
                    "failed to mark BTC {} settlement recognized",
                    execution_mode.as_str()
                )
            })?;
        Ok(result.rows_affected() == 1)
    }

    /// Compatibility wrapper for the existing paper lifecycle.
    pub async fn mark_paper_settlement_credited(
        &self,
        process_id: Uuid,
        run_id: Uuid,
        settlement_id: Uuid,
        credit_evidence: &serde_json::Value,
    ) -> Result<bool> {
        self.mark_settlement_recognized(
            process_id,
            run_id,
            settlement_id,
            BtcExecutionMode::Paper,
            credit_evidence,
        )
        .await
    }
}

fn validate_settlement_recognition_mode(execution_mode: BtcExecutionMode) -> Result<()> {
    if execution_mode == BtcExecutionMode::Live {
        bail!(
            "live settlement recognition requires exact exchange redemption proof and is not implemented"
        );
    }
    Ok(())
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

fn validate_run_manifest(
    run_key: &str,
    config_hash: &str,
    frozen_process_config: &serde_json::Value,
) -> Result<()> {
    if run_key.trim().is_empty() || config_hash.trim().is_empty() {
        bail!("BTC run key and config hash must not be empty");
    }
    if !frozen_process_config.is_object() {
        bail!("BTC run manifest frozen process config must be a JSON object");
    }
    Ok(())
}

fn validate_settlement_record(
    record: &BtcSettlementRecord,
    expected_mode: BtcExecutionMode,
) -> Result<()> {
    if record.execution_mode != expected_mode.as_str() {
        bail!("BTC settlement execution mode conflicts with the requested source");
    }
    if !matches!(record.official_outcome.as_str(), "up" | "down") {
        bail!("BTC settlement has a nonofficial outcome");
    }
    if !is_supported_official_resolution_source(&record.official_resolution_source) {
        bail!("BTC settlement has an unsupported official-resolution source");
    }
    if record.official_winning_token_id.trim().is_empty() {
        bail!("BTC settlement is missing its official winning token");
    }
    if record
        .fill_ids
        .as_array()
        .is_none_or(|fill_ids| fill_ids.is_empty())
    {
        bail!("BTC settlement must attribute at least one fill");
    }
    if record.filled_size <= Decimal::ZERO
        || record.entry_notional < Decimal::ZERO
        || record.entry_fees < Decimal::ZERO
        || record.payout < Decimal::ZERO
    {
        bail!("BTC settlement contains invalid amounts");
    }
    let expected_payout = if record.token_id == record.official_winning_token_id {
        record.filled_size
    } else {
        Decimal::ZERO
    };
    if record.payout != expected_payout {
        bail!("BTC settlement payout conflicts with the official winning token");
    }
    if record.net_pnl != record.payout - record.entry_notional - record.entry_fees {
        bail!("BTC settlement net PnL conflicts with its payout and entry costs");
    }
    Ok(())
}

fn live_zero_payout_settlement_evidence(
    record: &BtcSettlementRecord,
    config_hash: &str,
) -> Result<serde_json::Value> {
    validate_settlement_record(record, BtcExecutionMode::Live)?;
    if record.payout != Decimal::ZERO || record.token_id == record.official_winning_token_id {
        bail!("live zero-payout recognition requires an officially losing settlement");
    }
    if config_hash.len() != 64 || !config_hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("live zero-payout recognition requires a SHA-256 config hash");
    }
    Ok(serde_json::json!({
        "proof_type": "btc_official_zero_payout_loss",
        "evidence_version": "btc_live_zero_payout_settlement_v1",
        "recognition_kind": "official_resolution",
        "execution_mode": "live",
        "exchange_cash_credit_applied": false,
        "settlement_id": record.settlement_id,
        "process_id": record.process_id,
        "run_id": record.run_id,
        "order_id": record.order_id,
        "market_id": record.market_id,
        "token_id": record.token_id,
        "fill_ids": record.fill_ids,
        "official_outcome": record.official_outcome,
        "official_winning_token_id": record.official_winning_token_id,
        "official_resolution_received_at": record.official_resolution_received_at,
        "official_resolution_source": record.official_resolution_source,
        "filled_size": record.filled_size,
        "entry_notional": record.entry_notional,
        "entry_fees": record.entry_fees,
        "payout": record.payout,
        "net_pnl": record.net_pnl,
        "recognized_by_config_hash": config_hash,
    }))
}

fn is_supported_official_resolution_source(source: &str) -> bool {
    matches!(
        source,
        "clob_websocket" | "clob_rest_reconciliation" | "gamma_rest_reconciliation"
    )
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
        observed_at: row.observed_at,
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
            entry_policy,
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
                || (*entry_policy == BtcDirectionalModelEntryPolicy::RequirePositiveDirectEdge
                    && net_edge_per_share <= Decimal::ZERO)
                || (*entry_policy == BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction
                    && intent.strategy_version != BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION)
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

fn validate_orderbook_checkpoint_pair(
    checkpoints: &[OrderbookCheckpoint],
    publication_boundary: DateTime<Utc>,
) -> Result<(&OrderbookCheckpoint, &OrderbookCheckpoint)> {
    if checkpoints.len() != 2 {
        bail!(
            "BTC orderbook checkpoint pair requires exactly two checkpoints, received {}",
            checkpoints.len()
        );
    }
    let first = &checkpoints[0];
    let second = &checkpoints[1];
    if first.connection_id != second.connection_id {
        bail!("BTC orderbook checkpoint pair spans multiple CLOB connections");
    }
    if first.market_id != second.market_id {
        bail!("BTC orderbook checkpoint pair spans multiple markets");
    }
    if first.token_id == second.token_id {
        bail!("BTC orderbook checkpoint pair must contain distinct outcome tokens");
    }
    if first.integrity_status != FeedIntegrityStatus::Ok
        || second.integrity_status != FeedIntegrityStatus::Ok
    {
        bail!("BTC orderbook checkpoint pair contains a non-healthy book");
    }
    for checkpoint in [first, second] {
        if checkpoint.source_timestamp > publication_boundary {
            bail!("BTC orderbook checkpoint source timestamp exceeds the publication boundary");
        }
        if checkpoint.received_at > publication_boundary {
            bail!("BTC orderbook checkpoint receipt timestamp exceeds the publication boundary");
        }
    }
    Ok((first, second))
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

fn shadow_predictive_regime_v2_history_limit(max_candidates: u32) -> Result<i64> {
    if !(1..=MAX_SHADOW_PREDICTIVE_REGIME_V2_HISTORY_CANDIDATES).contains(&max_candidates) {
        bail!(
            "shadow predictive-regime v2 history limit must be between 1 and {} candidates",
            MAX_SHADOW_PREDICTIVE_REGIME_V2_HISTORY_CANDIDATES
        );
    }
    Ok(i64::from(max_candidates))
}

fn shadow_predictive_regime_v2_selected_point_probability(
    decision_outcome: BtcOutcome,
    decision_fair_probability: Decimal,
    prediction_status: Option<&str>,
    feature_up_probability: Decimal,
    feature_up_lower: Decimal,
    feature_up_upper: Decimal,
) -> Result<Decimal> {
    let probabilities = [
        decision_fair_probability,
        feature_up_probability,
        feature_up_lower,
        feature_up_upper,
    ];
    if probabilities
        .iter()
        .any(|value| *value < Decimal::ZERO || *value > Decimal::ONE)
        || feature_up_lower > feature_up_probability
        || feature_up_probability > feature_up_upper
    {
        bail!("shadow predictive-regime v2 feature fair-value interval is inconsistent");
    }

    let (selected_point_probability, selected_conservative_probability) = match decision_outcome {
        BtcOutcome::Up => (feature_up_probability, feature_up_lower),
        BtcOutcome::Down => (
            Decimal::ONE - feature_up_probability,
            Decimal::ONE - feature_up_upper,
        ),
    };
    let projected_decision_probability = match prediction_status {
        None => selected_conservative_probability,
        Some("directional_prediction") => selected_point_probability,
        Some(status) => bail!(
            "shadow predictive-regime v2 buy decision has unsupported prediction status {status}"
        ),
    };
    // The persisted probability columns use numeric(18,10). Down probabilities are stored from
    // the raw complement, while replay derives the complement from the stored Up value; midpoint
    // rounding can therefore differ by one storage quantum without changing the evidence.
    let projection_tolerance = match decision_outcome {
        BtcOutcome::Up => Decimal::ZERO,
        BtcOutcome::Down => Decimal::new(1, 10),
    };
    if (decision_fair_probability - projected_decision_probability).abs() > projection_tolerance {
        bail!(
            "shadow predictive-regime v2 persisted decision probability projection is inconsistent"
        );
    }
    Ok(selected_point_probability)
}

fn shadow_predictive_regime_v2_candidate_from_row(
    process_id: Uuid,
    expected_max_reference_age: Duration,
    expected_max_directional_feature_age: Option<Duration>,
    row: ShadowPredictiveRegimeV2CandidateRow,
) -> Result<ShadowPredictiveRegimeV2Candidate> {
    if process_id.is_nil() {
        bail!("shadow predictive-regime v2 candidate process_id cannot be nil");
    }
    let request = serde_json::from_value::<OrderRequest>(row.order_request)
        .context("failed to deserialize shadow predictive-regime v2 exposure order request")?;
    if request.process_id != Some(process_id)
        || request.client_order_id != row.order_client_order_id
        || request.market_id != row.market_id
        || request.token_id != row.order_token_id
        || serde_name(&request.side)? != row.order_side
        || serde_name(&request.order_type)? != row.order_type
        || request.price != row.order_price
        || request.size != row.order_size
        || request.side != OrderSide::Buy
    {
        bail!("shadow predictive-regime v2 order request does not match persisted exposure");
    }
    if request
        .metadata
        .get("execution_intent")
        .and_then(serde_json::Value::as_str)
        != Some("entry")
        || request
            .metadata
            .pointer("/paper_execution/non_mutating_preview")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
    {
        bail!("shadow predictive-regime v2 exposure is not an actual paper entry");
    }

    let guard = reference_execution_guard(&request)
        .map_err(anyhow::Error::new)
        .context("shadow predictive-regime v2 exposure has no valid execution guard")?;
    guard
        .validate_for_request(
            &request,
            row.fill_at,
            process_id,
            expected_max_reference_age,
            expected_max_directional_feature_age,
        )
        .map_err(anyhow::Error::new)
        .context("shadow predictive-regime v2 exposure execution guard failed validation")?;

    let decision_id = row
        .decision_id
        .context("shadow predictive-regime v2 exposure has no exact guarded decision")?;
    let snapshot_id = row
        .decision_snapshot_id
        .context("shadow predictive-regime v2 exposure decision has no snapshot")?;
    let decision_at = row
        .decision_at
        .context("shadow predictive-regime v2 exposure decision has no timestamp")?;
    let decision_action = row
        .decision_action
        .as_deref()
        .context("shadow predictive-regime v2 exposure decision has no action")?;
    let decision_outcome = parse_outcome_name(
        row.decision_outcome
            .as_deref()
            .context("shadow predictive-regime v2 exposure decision has no outcome")?,
    )?;
    let decision_fair_probability = row
        .decision_fair_probability
        .context("shadow predictive-regime v2 exposure decision has no fair probability")?;
    let resolved_outcome = parse_outcome_name(&row.resolved_outcome)?;
    let feature_snapshot_id = row
        .feature_snapshot_id
        .context("shadow predictive-regime v2 exposure has no exact feature snapshot")?;
    let feature_market_id = row
        .feature_market_id
        .as_deref()
        .context("shadow predictive-regime v2 exposure feature has no market")?;
    let feature_as_of = row
        .feature_as_of
        .context("shadow predictive-regime v2 exposure feature has no timestamp")?;
    let feature_schema_version = row
        .feature_schema_version
        .as_deref()
        .context("shadow predictive-regime v2 exposure feature has no schema version")?;
    let feature_hash = row
        .feature_hash
        .as_deref()
        .context("shadow predictive-regime v2 exposure feature has no hash")?;
    let fair_up_probability = row
        .fair_up_probability
        .context("shadow predictive-regime v2 exposure feature has no fair probability")?;
    let fair_up_lower = row
        .fair_up_lower
        .context("shadow predictive-regime v2 exposure feature has no lower probability bound")?;
    let fair_up_upper = row
        .fair_up_upper
        .context("shadow predictive-regime v2 exposure feature has no upper probability bound")?;
    let order_prediction_status = request
        .metadata
        .pointer("/prediction/status")
        .and_then(serde_json::Value::as_str);

    if decision_action != "buy"
        || guard.decision_id != decision_id
        || guard.snapshot_id != snapshot_id
        || feature_snapshot_id != snapshot_id
        || guard.market_id != row.market_id
        || feature_market_id != row.market_id
        || guard.outcome != decision_outcome
        || guard.decision_at.timestamp_micros() != decision_at.timestamp_micros()
        || guard.feature_as_of.timestamp_micros() != feature_as_of.timestamp_micros()
        || guard.feature_schema_version != feature_schema_version
        || guard.feature_sha256 != feature_hash
        || order_prediction_status != row.decision_prediction_status.as_deref()
    {
        bail!("shadow predictive-regime v2 guarded evidence identity is inconsistent");
    }

    let selected_point_probability = shadow_predictive_regime_v2_selected_point_probability(
        decision_outcome,
        decision_fair_probability,
        row.decision_prediction_status.as_deref(),
        fair_up_probability,
        fair_up_lower,
        fair_up_upper,
    )?;
    let candidate = ShadowPredictiveRegimeV2Candidate {
        market_id: row.market_id,
        decision_id,
        snapshot_id,
        order_id: row.order_id,
        fill_id: row.fill_id,
        source: ShadowPredictiveRegimeV2CandidateSource::ActualPaperFill,
        decision_outcome,
        resolved_outcome,
        selected_point_probability,
        decision_at,
        fill_at: row.fill_at,
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
    // PostgreSQL timestamptz round-trips at microsecond precision, while the JSON evaluation
    // retains Chrono's nanosecond precision. Compare only the precision the durable row can
    // represent so a valid checkpoint remains hydratable after restart.
    if evaluation.as_of.timestamp_micros() != row.decision_at.timestamp_micros() {
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

fn shadow_predictive_regime_v2_state_from_persisted_evaluation(
    process_id: Uuid,
    breaker_state_config_hash: Option<&str>,
    row: PersistedShadowPredictiveRegimeV2EvaluationRow,
) -> Result<ShadowPredictiveRegimeV2State> {
    let evaluation =
        serde_json::from_value::<ShadowPredictiveRegimeV2Evaluation>(row.evaluation)
            .context("failed to deserialize persisted shadow predictive-regime v2 evaluation")?;
    let state = &evaluation.state;

    if evaluation.schema_version != SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_V2_SCHEMA_VERSION
        || evaluation.mode != "shadow"
    {
        bail!("persisted shadow predictive-regime v2 evaluation has the wrong contract");
    }
    if evaluation.process_id != process_id || state.process_id != process_id {
        bail!("persisted shadow predictive-regime v2 state is owned by another process");
    }
    if evaluation.config_hash != state.config_hash {
        bail!("persisted shadow predictive-regime v2 evaluation and state config hashes disagree");
    }
    if state.config_hash.len() != 64
        || !state
            .config_hash
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("persisted shadow predictive-regime v2 state config hash is invalid");
    }
    if let Some(expected) = breaker_state_config_hash {
        if evaluation.config_hash != expected {
            bail!("persisted shadow predictive-regime v2 state config hash does not match");
        }
    }
    if evaluation.as_of.timestamp_micros() != row.decision_at.timestamp_micros() {
        bail!("persisted shadow predictive-regime v2 evaluation is not aligned to its decision");
    }
    if evaluation.degraded != state.degraded
        || evaluation.would_defer != state.degraded
        || evaluation.consecutive_degradation_markets != state.consecutive_degradation_markets
        || evaluation.consecutive_recovery_markets != state.consecutive_recovery_markets
        || evaluation.resolved_markets_observed != state.resolved_markets_observed
        || evaluation.state_as_of_market_id != state.state_as_of_market_id
        || evaluation.state_as_of_decision_id != state.state_as_of_decision_id
        || evaluation.state_as_of_fill_id != state.state_as_of_fill_id
        || evaluation.state_as_of_label_available_at != state.state_as_of_label_available_at
    {
        bail!("persisted shadow predictive-regime v2 evaluation and state evidence disagree");
    }

    #[derive(Serialize)]
    struct PersistedV2StateEvidence<'a> {
        schema_version: &'a str,
        config_hash: String,
        state: &'a ShadowPredictiveRegimeV2State,
    }
    let expected_state_evidence_sha256 = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&PersistedV2StateEvidence {
            schema_version: &evaluation.schema_version,
            config_hash: state.config_hash.clone(),
            state,
        })?)
    );
    #[derive(Serialize)]
    struct PersistedV2EvaluationEvidence<'a> {
        schema_version: &'a str,
        config_hash: &'a str,
        mode: &'a str,
        process_id: Uuid,
        as_of: DateTime<Utc>,
        state_evidence_sha256: &'a str,
        evidence_fresh: bool,
        degraded: bool,
        would_defer: bool,
        degradation_condition_met: bool,
        recovery_condition_met: bool,
        reason: &'a str,
    }
    let expected_evaluation_evidence_sha256 = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&PersistedV2EvaluationEvidence {
            schema_version: &evaluation.schema_version,
            config_hash: &evaluation.config_hash,
            mode: &evaluation.mode,
            process_id: evaluation.process_id,
            as_of: evaluation.as_of,
            state_evidence_sha256: &evaluation.state_evidence_sha256,
            evidence_fresh: evaluation.evidence_fresh,
            degraded: evaluation.degraded,
            would_defer: evaluation.would_defer,
            degradation_condition_met: evaluation.degradation_condition_met,
            recovery_condition_met: evaluation.recovery_condition_met,
            reason: &evaluation.reason,
        })?)
    );
    if !evaluation.shadow_only
        || !evaluation.state_checkpoint_eligible
        || evaluation.disposition != AdmissionDisposition::Allow
        || evaluation.telemetry_error.is_some()
        || evaluation.refresh_pending
        || evaluation.state_evidence_sha256 != expected_state_evidence_sha256
        || evaluation.evaluation_evidence_sha256 != expected_evaluation_evidence_sha256
    {
        bail!("persisted shadow predictive-regime v2 evaluation evidence is incomplete");
    }

    if state.resolved_markets_observed
        < u64::try_from(state.slow_candidates.len()).unwrap_or(u64::MAX)
    {
        bail!("persisted shadow predictive-regime v2 observation count is inconsistent");
    }
    let mut market_ids = HashSet::new();
    let mut decision_ids = HashSet::new();
    let mut order_ids = HashSet::new();
    let mut fill_ids = HashSet::new();
    let mut previous_ordering_key = None;
    let mut slow_probability_sum = Decimal::ZERO;
    let mut slow_win_count = 0_u32;
    let mut slow_brier_sum = Decimal::ZERO;
    for candidate in &state.slow_candidates {
        candidate.validate()?;
        if candidate.label_available_at > evaluation.as_of {
            bail!("persisted shadow predictive-regime v2 state contains future evidence");
        }
        if !market_ids.insert(candidate.market_id.as_str())
            || !decision_ids.insert(candidate.decision_id)
            || !order_ids.insert(candidate.order_id.as_str())
            || !fill_ids.insert(candidate.fill_id)
        {
            bail!("persisted shadow predictive-regime v2 state contains duplicate evidence");
        }
        let ordering_key = (
            candidate.label_available_at,
            candidate.fill_at,
            candidate.fill_id,
            candidate.market_id.as_str(),
        );
        if previous_ordering_key.is_some_and(|previous| ordering_key <= previous) {
            bail!("persisted shadow predictive-regime v2 state is not in causal order");
        }
        previous_ordering_key = Some(ordering_key);
        slow_probability_sum += candidate.selected_point_probability;
        slow_win_count = slow_win_count.saturating_add(u32::from(candidate.won()));
        slow_brier_sum += shadow_predictive_regime_v2_brier_score(candidate);
    }
    let fresh_count = usize::try_from(state.fresh_evidence_count)
        .context("persisted shadow predictive-regime v2 fresh count does not fit in memory")?;
    if fresh_count > state.slow_candidates.len()
        || (!state.slow_candidates.is_empty() && fresh_count == 0)
    {
        bail!("persisted shadow predictive-regime v2 fresh evidence count is inconsistent");
    }
    let mut fast_brier_sum = Decimal::ZERO;
    let mut fast_confidence_weighted_miss_sum = Decimal::ZERO;
    for candidate in state.slow_candidates.iter().rev().take(fresh_count) {
        fast_brier_sum += shadow_predictive_regime_v2_brier_score(candidate);
        if !candidate.won() {
            fast_confidence_weighted_miss_sum += candidate.selected_point_probability;
        }
    }
    if state.slow_probability_sum != slow_probability_sum
        || state.slow_win_count != slow_win_count
        || state.slow_brier_sum != slow_brier_sum
        || state.fast_brier_sum != fast_brier_sum
        || state.fast_confidence_weighted_miss_sum != fast_confidence_weighted_miss_sum
    {
        bail!("persisted shadow predictive-regime v2 rolling aggregates are inconsistent");
    }

    match state.slow_candidates.back() {
        Some(last)
            if state.state_as_of_market_id.as_deref() == Some(last.market_id.as_str())
                && state.state_as_of_decision_id == Some(last.decision_id)
                && state.state_as_of_fill_id == Some(last.fill_id)
                && state.state_as_of_decision_at == Some(last.decision_at)
                && state.state_as_of_label_available_at == Some(last.label_available_at) => {}
        Some(_) => {
            bail!("persisted shadow predictive-regime v2 state cursor does not match its evidence")
        }
        None if state.resolved_markets_observed == 0
            && state.fresh_evidence_count == 0
            && state.slow_probability_sum == Decimal::ZERO
            && state.slow_win_count == 0
            && state.slow_brier_sum == Decimal::ZERO
            && state.fast_brier_sum == Decimal::ZERO
            && state.fast_confidence_weighted_miss_sum == Decimal::ZERO
            && state.state_as_of_market_id.is_none()
            && state.state_as_of_decision_id.is_none()
            && state.state_as_of_fill_id.is_none()
            && state.state_as_of_decision_at.is_none()
            && state.state_as_of_label_available_at.is_none()
            && !state.degraded
            && state.consecutive_degradation_markets == 0
            && state.consecutive_recovery_markets == 0 => {}
        None => bail!("persisted shadow predictive-regime v2 state is missing its evidence"),
    }

    let metrics_present = evaluation.slow_resolved_market_count.is_some();
    if [
        evaluation.fast_resolved_market_count.is_some(),
        evaluation.rolling_wins.is_some(),
        evaluation.mean_selected_point_probability.is_some(),
        evaluation.empirical_accuracy.is_some(),
        evaluation.brier_score.is_some(),
        evaluation.slow_brier_score.is_some(),
        evaluation.fast_brier_score.is_some(),
        evaluation.fast_minus_slow_brier.is_some(),
        evaluation.fast_confidence_weighted_miss.is_some(),
    ]
    .into_iter()
    .any(|present| present != metrics_present)
        || evaluation.sample_ready != (metrics_present && evaluation.evidence_fresh)
    {
        bail!("persisted shadow predictive-regime v2 metric evidence is inconsistent");
    }
    if metrics_present {
        let slow_count = u32::try_from(state.slow_candidates.len())
            .context("persisted shadow predictive-regime v2 slow count does not fit in u32")?;
        let fast_count = state.fresh_evidence_count;
        if slow_count == 0 || fast_count == 0 {
            bail!("persisted shadow predictive-regime v2 metric window is empty");
        }
        let slow_denominator = Decimal::from(slow_count);
        let fast_denominator = Decimal::from(fast_count);
        let slow_brier_score = state.slow_brier_sum / slow_denominator;
        let fast_brier_score = state.fast_brier_sum / fast_denominator;
        if evaluation.slow_resolved_market_count != Some(slow_count)
            || evaluation.fast_resolved_market_count != Some(fast_count)
            || evaluation.rolling_wins != Some(state.slow_win_count)
            || evaluation.mean_selected_point_probability
                != Some(state.slow_probability_sum / slow_denominator)
            || evaluation.empirical_accuracy
                != Some(Decimal::from(state.slow_win_count) / slow_denominator)
            || evaluation.brier_score != Some(slow_brier_score)
            || evaluation.slow_brier_score != Some(slow_brier_score)
            || evaluation.fast_brier_score != Some(fast_brier_score)
            || evaluation.fast_minus_slow_brier != Some(fast_brier_score - slow_brier_score)
            || evaluation.fast_confidence_weighted_miss
                != Some(state.fast_confidence_weighted_miss_sum / fast_denominator)
        {
            bail!("persisted shadow predictive-regime v2 metric values are inconsistent");
        }
    }

    Ok(evaluation.state)
}

fn shadow_predictive_regime_v2_brier_score(
    candidate: &ShadowPredictiveRegimeV2Candidate,
) -> Decimal {
    let observed = if candidate.won() {
        Decimal::ONE
    } else {
        Decimal::ZERO
    };
    let error = candidate.selected_point_probability - observed;
    error * error
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
            ShadowPredictiveRegimeCandidate, ShadowPredictiveRegimeCircuitBreakerConfig,
            SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_MODE,
            SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION,
        },
        execution_guard::{
            BtcReferenceExecutionGuard, BtcReferenceTickEvidence,
            BTC_REFERENCE_EXECUTION_GUARD_VERSION,
        },
        predictive_regime_v2::{
            ShadowPredictiveRegimeCircuitBreakerV2Config,
            SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_V2_SCHEMA_VERSION,
        },
        strategy::{
            ApprovedIntent, OutcomeEdge, BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_LINEAGE_VERSION,
        },
    };
    use crate::models::OrderType;

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

    fn predictive_regime_v2_config() -> ShadowPredictiveRegimeCircuitBreakerV2Config {
        ShadowPredictiveRegimeCircuitBreakerV2Config {
            schema_version: SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_V2_SCHEMA_VERSION.to_string(),
            mode: "shadow".to_string(),
            fast_resolved_market_window: 4,
            slow_resolved_market_window: 20,
            minimum_resolved_markets: 20,
            max_evidence_gap_seconds: 900,
            degradation_fast_brier_score_threshold: dec!(0.27),
            degradation_fast_minus_slow_threshold: dec!(0.02),
            degradation_slow_brier_score_threshold: dec!(0.25),
            degradation_confirmation_markets: 2,
            recovery_fast_brier_score_threshold: dec!(0.25),
            recovery_fast_minus_slow_ceiling: Decimal::ZERO,
            recovery_confirmation_markets: 2,
        }
    }

    fn predictive_regime_v2_candidate(
        index: i64,
        probability: Decimal,
        won: bool,
    ) -> ShadowPredictiveRegimeV2Candidate {
        let label_available_at = Utc.with_ymd_and_hms(2026, 7, 22, 0, 0, 1).single().unwrap()
            + Duration::minutes(index * 5);
        ShadowPredictiveRegimeV2Candidate {
            market_id: format!("v2-market-{index}"),
            decision_id: Uuid::from_u128(u128::try_from(index + 1).unwrap()),
            snapshot_id: Uuid::from_u128(u128::try_from(index + 10_001).unwrap()),
            order_id: format!("v2-order-{index}"),
            fill_id: Uuid::from_u128(u128::try_from(index + 20_001).unwrap()),
            source: ShadowPredictiveRegimeV2CandidateSource::ActualPaperFill,
            decision_outcome: BtcOutcome::Up,
            resolved_outcome: if won {
                BtcOutcome::Up
            } else {
                BtcOutcome::Down
            },
            selected_point_probability: probability,
            decision_at: label_available_at - Duration::minutes(4),
            fill_at: label_available_at - Duration::minutes(3),
            label_available_at,
        }
    }

    fn predictive_regime_v2_candidate_row(
        process_id: Uuid,
    ) -> ShadowPredictiveRegimeV2CandidateRow {
        predictive_regime_v2_candidate_row_for_outcome(process_id, BtcOutcome::Up)
    }

    fn predictive_regime_v2_candidate_row_for_outcome(
        process_id: Uuid,
        outcome: BtcOutcome,
    ) -> ShadowPredictiveRegimeV2CandidateRow {
        let fill_at = Utc
            .with_ymd_and_hms(2026, 7, 22, 12, 0, 0)
            .single()
            .unwrap();
        let decision_at = fill_at - Duration::seconds(1) + Duration::nanoseconds(999);
        let feature_as_of = decision_at - Duration::seconds(1);
        let client_order_id = Uuid::from_u128(7_001);
        let decision_id = Uuid::from_u128(7_002);
        let snapshot_id = Uuid::from_u128(7_003);
        let intent_id = Uuid::from_u128(7_004);
        let (token_id, outcome_name, resolved_outcome, decision_fair_probability) = match outcome {
            BtcOutcome::Up => ("v2-up-token", "up", "down", dec!(0.58)),
            BtcOutcome::Down => ("v2-down-token", "down", "up", dec!(0.28)),
        };
        let tick = |id| BtcReferenceTickEvidence {
            tick_id: Uuid::from_u128(id),
            source_timestamp: feature_as_of - Duration::seconds(1),
            received_at: feature_as_of - Duration::seconds(1),
            ingest_sequence: u64::try_from(id).unwrap(),
        };
        let mut guard: BtcReferenceExecutionGuard = serde_json::from_value(serde_json::json!({
            "guard_version": BTC_REFERENCE_EXECUTION_GUARD_VERSION,
            "process_id": process_id,
            "intent_id": intent_id,
            "decision_id": decision_id,
            "decision_at": decision_at,
            "snapshot_id": snapshot_id,
            "feature_as_of": feature_as_of,
            "market_id": "v2-exposure-market",
            "token_id": token_id,
            "outcome": outcome,
            "strategy_version": "strategy-v2-test",
            "feature_schema_version": "feature-v2-test",
            "lineage_version": BTC_CHAINLINK_PATH_CONDITIONED_FEATURE_LINEAGE_VERSION,
            "feature_sha256": "a".repeat(64),
            "client_order_id": client_order_id,
            "side": OrderSide::Buy,
            "order_type": OrderType::Fok,
            "limit_price": dec!(0.61),
            "size": dec!(5),
            "signal_id": null,
            "dynamic_fee_rate": dec!(0.01),
            "chainlink_open": tick(7_101),
            "chainlink": tick(7_102),
            "binance": tick(7_103),
            "max_reference_age_ms": 60_000,
            "evidence_sha256": "",
        }))
        .unwrap();
        guard.reseal_for_test();
        let mut request = OrderRequest {
            client_order_id,
            process_id: Some(process_id),
            market_id: guard.market_id.clone(),
            token_id: guard.token_id.clone(),
            side: guard.side,
            order_type: guard.order_type,
            price: guard.limit_price,
            size: guard.size,
            metadata: serde_json::json!({
                "process_id": process_id,
                "decision_id": decision_id,
                "feature_snapshot_id": snapshot_id,
                "strategy_version": guard.strategy_version,
                "feature_schema_version": guard.feature_schema_version,
                "outcome": guard.outcome,
                "dynamic_fee_rate": guard.dynamic_fee_rate,
                "execution_intent": "entry",
                "paper_execution": { "non_mutating_preview": false }
            }),
        };
        guard.insert_into_metadata(&mut request.metadata).unwrap();

        ShadowPredictiveRegimeV2CandidateRow {
            market_id: request.market_id.clone(),
            order_id: "paper-v2-exposure-order".to_string(),
            order_client_order_id: request.client_order_id,
            order_token_id: request.token_id.clone(),
            order_side: "buy".to_string(),
            order_type: "fok".to_string(),
            order_price: request.price,
            order_size: request.size,
            order_request: serde_json::to_value(request).unwrap(),
            fill_id: Uuid::from_u128(7_005),
            fill_at,
            decision_id: Some(decision_id),
            decision_snapshot_id: Some(snapshot_id),
            decision_action: Some("buy".to_string()),
            decision_outcome: Some(outcome_name.to_string()),
            decision_fair_probability: Some(decision_fair_probability),
            decision_prediction_status: None,
            decision_at: Some(
                DateTime::<Utc>::from_timestamp_micros(decision_at.timestamp_micros()).unwrap(),
            ),
            resolved_outcome: resolved_outcome.to_string(),
            label_available_at: fill_at + Duration::minutes(4),
            feature_snapshot_id: Some(snapshot_id),
            feature_market_id: Some("v2-exposure-market".to_string()),
            feature_as_of: Some(
                DateTime::<Utc>::from_timestamp_micros(feature_as_of.timestamp_micros()).unwrap(),
            ),
            feature_schema_version: Some(guard.feature_schema_version.clone()),
            feature_hash: Some(guard.feature_sha256.clone()),
            fair_up_probability: Some(dec!(0.65)),
            fair_up_lower: Some(dec!(0.58)),
            fair_up_upper: Some(dec!(0.72)),
        }
    }

    fn healthy_checkpoint(
        id: u128,
        market_id: &str,
        token_id: &str,
        connection_id: Uuid,
        at: DateTime<Utc>,
    ) -> OrderbookCheckpoint {
        OrderbookCheckpoint {
            checkpoint_id: Uuid::from_u128(id),
            market_id: market_id.to_string(),
            token_id: token_id.to_string(),
            source_timestamp: at,
            received_at: at + Duration::milliseconds(10),
            observed_at: at + Duration::milliseconds(10),
            connection_id,
            ingest_sequence: id as u64,
            source_hash: Some(format!("hash-{id}")),
            tick_size: dec!(0.01),
            best_bid: Some(dec!(0.49)),
            best_ask: Some(dec!(0.51)),
            bids: vec![OrderbookLevel {
                price: dec!(0.49),
                size: dec!(10),
            }],
            asks: vec![OrderbookLevel {
                price: dec!(0.51),
                size: dec!(12),
            }],
            integrity_status: FeedIntegrityStatus::Ok,
        }
    }

    fn healthy_checkpoint_pair(at: DateTime<Utc>) -> Vec<OrderbookCheckpoint> {
        let connection_id = Uuid::from_u128(700);
        vec![
            healthy_checkpoint(701, "market", "up", connection_id, at),
            healthy_checkpoint(702, "market", "down", connection_id, at),
        ]
    }

    #[test]
    fn checkpoint_pair_validation_accepts_one_healthy_epoch_and_market() {
        let at = Utc.with_ymd_and_hms(2026, 7, 21, 12, 0, 0).unwrap();
        let checkpoints = healthy_checkpoint_pair(at);

        let (first, second) =
            validate_orderbook_checkpoint_pair(&checkpoints, at + Duration::milliseconds(10))
                .unwrap();

        assert_eq!(first.token_id, "up");
        assert_eq!(second.token_id, "down");
    }

    #[test]
    fn checkpoint_pair_validation_requires_exactly_two_books() {
        let at = Utc.with_ymd_and_hms(2026, 7, 21, 12, 0, 0).unwrap();
        let checkpoints = healthy_checkpoint_pair(at);

        assert!(
            validate_orderbook_checkpoint_pair(&checkpoints[..1], at + Duration::seconds(1))
                .is_err()
        );
        let mut three = checkpoints;
        three.push(healthy_checkpoint(
            703,
            "market",
            "third",
            Uuid::from_u128(700),
            at,
        ));
        assert!(validate_orderbook_checkpoint_pair(&three, at + Duration::seconds(1)).is_err());
    }

    #[test]
    fn checkpoint_pair_validation_rejects_mixed_identity_or_unhealthy_books() {
        let at = Utc.with_ymd_and_hms(2026, 7, 21, 12, 0, 0).unwrap();
        let boundary = at + Duration::seconds(1);

        let mut mixed_connections = healthy_checkpoint_pair(at);
        mixed_connections[1].connection_id = Uuid::from_u128(999);
        assert!(validate_orderbook_checkpoint_pair(&mixed_connections, boundary).is_err());

        let mut mixed_markets = healthy_checkpoint_pair(at);
        mixed_markets[1].market_id = "other-market".to_string();
        assert!(validate_orderbook_checkpoint_pair(&mixed_markets, boundary).is_err());

        let mut duplicate_tokens = healthy_checkpoint_pair(at);
        duplicate_tokens[1].token_id = duplicate_tokens[0].token_id.clone();
        assert!(validate_orderbook_checkpoint_pair(&duplicate_tokens, boundary).is_err());

        let mut unhealthy = healthy_checkpoint_pair(at);
        unhealthy[1].integrity_status = FeedIntegrityStatus::Stale;
        assert!(validate_orderbook_checkpoint_pair(&unhealthy, boundary).is_err());
    }

    #[test]
    fn checkpoint_pair_validation_rejects_data_newer_than_publication() {
        let at = Utc.with_ymd_and_hms(2026, 7, 21, 12, 0, 0).unwrap();
        let boundary = at + Duration::seconds(1);

        let mut future_source = healthy_checkpoint_pair(at);
        future_source[0].source_timestamp = boundary + Duration::milliseconds(1);
        assert!(validate_orderbook_checkpoint_pair(&future_source, boundary).is_err());

        let mut future_receipt = healthy_checkpoint_pair(at);
        future_receipt[1].received_at = boundary + Duration::milliseconds(1);
        assert!(validate_orderbook_checkpoint_pair(&future_receipt, boundary).is_err());
    }

    #[test]
    fn checkpoint_pair_insert_is_one_fixed_two_row_statement() {
        let normalized = INSERT_ORDERBOOK_CHECKPOINT_PAIR_SQL
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase();

        assert!(normalized.contains("values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17), ($18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34)"));
        assert!(!normalized.contains("on conflict"));
    }

    #[test]
    fn directional_model_binance_query_is_causal_fresh_and_single_row() {
        let normalized = LOAD_DIRECTIONAL_MODEL_BINANCE_TICK_SQL
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase();

        assert!(normalized.contains("source = 'direct_binance'"));
        assert!(normalized.contains("symbol = 'btcusd'"));
        assert!(normalized.contains("integrity_status = 'ok'"));
        assert!(normalized.contains("source_timestamp >= $1"));
        assert!(normalized.contains("source_timestamp <= $2"));
        assert!(normalized.contains("received_at >= $1"));
        assert!(normalized.contains("received_at <= $2"));
        assert!(normalized.contains(
            "order by source_timestamp desc, received_at desc, ingest_sequence desc, tick_id desc"
        ));
        assert!(normalized.contains("limit 1"));
        assert!(!normalized.contains("rtds_chainlink"));
        assert!(!normalized.contains("limit 2000"));
        assert!(!normalized.contains("limit 20000"));
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
    fn run_manifest_queries_use_global_identity_and_process_owned_loads() {
        let exists = RUN_MANIFEST_EXISTS_SQL.to_ascii_lowercase();
        assert!(exists.contains("event_type = 'btc_run_manifest'"));
        assert!(exists.contains("event_id = $1"));
        assert!(exists.contains("metadata #>> '{run_id}' = $1::text"));
        assert!(exists.contains("metadata #>> '{run_key}' = $2"));
        assert!(!exists.contains("process_id"));

        let load = LOAD_RUN_MANIFEST_SQL.to_ascii_lowercase();
        assert!(load.contains("where process_id = $1"));
        assert!(load.contains("event_id = $2"));
        assert!(load.contains("metadata #>> '{run_id}' = $2::text"));
        assert!(load.contains("metadata #>> '{run_key}' = $3"));
        assert!(load.contains("metadata #> '{frozen_process_config}'"));
        assert!(load.contains("limit 2"));

        let claim_lock = CLAIM_RUN_MANIFEST_LOCK_SQL.to_ascii_lowercase();
        assert!(claim_lock.contains("pg_advisory_xact_lock"));
        assert!(claim_lock.contains("polymarket.btc_run_manifest.start"));

        let insert = INSERT_RUN_MANIFEST_SQL.to_ascii_lowercase();
        assert!(insert.contains("process_id, event_id"));
        assert!(insert.contains("$1, $2, now()"));
        assert!(insert.contains("'btc_run_manifest'"));
        assert!(!insert.contains("on conflict"));
    }

    #[test]
    fn run_child_queries_are_process_first_with_historical_order_metadata_fallback() {
        let resume = PAPER_VENUE_RESUME_STATE_SQL.to_ascii_lowercase();
        assert!(resume.contains("with order_identity as materialized"));
        assert!(resume.contains("where o.process_id = $1"));
        assert!(resume.contains("bool_or("));
        assert!(resume.contains("as has_identity_conflict"));
        assert!(resume.contains("i.has_identity_conflict"));
        assert!(resume.contains("f.process_id = $1"));
        assert!(resume.contains("where process_id = $1\n    and run_id = $2"));
        assert!(resume.contains("and execution_mode = 'paper'"));
        assert!(resume.matches("metadata,run_id").count() >= 2);
        assert!(resume.matches("experiment_id").count() >= 2);
        assert!(!resume.contains("btc_paper_experiments"));

        let decision_insert = INSERT_STRATEGY_DECISION_SQL.to_ascii_lowercase();
        assert!(decision_insert.contains("process_id, run_id, decision_id"));
        assert!(decision_insert.contains("values (\n  $1,$2,$3"));
        assert!(decision_insert.contains("$21,$22,$23"));
        assert!(!decision_insert.contains("'paper'"));
        assert!(!decision_insert.contains("experiment_id"));

        let decision_update = UPDATE_STRATEGY_DECISION_EXECUTION_SQL.to_ascii_lowercase();
        assert!(decision_update.contains("where process_id = $1"));
        assert!(decision_update.contains("and run_id = $2"));

        let discovery = DISCOVER_PENDING_SETTLEMENTS_SQL.to_ascii_lowercase();
        assert!(discovery.contains("with order_identity as materialized"));
        assert!(discovery.contains("where o.process_id = $1"));
        assert!(discovery.contains("bool_or("));
        assert!(discovery.contains("as has_identity_conflict"));
        assert!(discovery.contains("on f.process_id = $1"));
        assert!(discovery.contains("and f.source = $3"));
        assert!(discovery
            .contains("case when $3 = 'live' then o.order_run_id else $2::uuid end as run_id"));
        assert!(discovery.contains("$3 = 'live' and coalesce("));
        assert!(discovery.contains("process_id, run_id, execution_mode"));
        assert!(discovery.contains("process_id, run_id, $3"));
        assert!(discovery.contains("on conflict (run_id, order_id) do nothing"));
        assert!(discovery.contains("where not (select has_identity_conflict from identity_state)"));
        assert!(discovery.contains(
            "(select has_identity_conflict from identity_state) as has_identity_conflict"
        ));
        assert!(discovery.matches("metadata,run_id").count() >= 2);
        assert!(discovery.matches("experiment_id").count() >= 2);
        assert!(!discovery.contains("btc_paper_experiments"));

        let pending = LOAD_PENDING_SETTLEMENTS_SQL.to_ascii_lowercase();
        assert!(pending.contains("process_id = $1"));
        assert!(pending.contains("execution_mode = $3"));
        assert!(pending.contains("$3 = 'live' or run_id = $2"));
        assert!(!pending.contains("experiment_id"));

        let recognized = MARK_SETTLEMENT_RECOGNIZED_SQL.to_ascii_lowercase();
        assert!(recognized.contains("process_id = $1"));
        assert!(recognized.contains("run_id = $2"));
        assert!(recognized.contains("execution_mode = $4"));
        assert!(!recognized.contains("experiment_id"));
    }

    #[test]
    fn strategy_decision_execution_mode_binding_preserves_paper_and_accepts_live() {
        assert_eq!(BtcExecutionMode::Paper.as_str(), "paper");
        assert_eq!(BtcExecutionMode::Live.as_str(), "live");

        let insert = INSERT_STRATEGY_DECISION_SQL.to_ascii_lowercase();
        assert!(insert.contains("order_plan_id, execution_mode"));
        assert!(insert.contains("$21,$22,$23"));
        assert!(!insert.contains("'paper'"));
        assert!(!insert.contains("'live'"));
    }

    #[test]
    fn settlement_recognition_remains_paper_only_until_redemption_is_proven() {
        validate_settlement_recognition_mode(BtcExecutionMode::Paper).unwrap();

        let error = validate_settlement_recognition_mode(BtcExecutionMode::Live).unwrap_err();
        assert!(error
            .to_string()
            .contains("exact exchange redemption proof"));
    }

    #[test]
    fn conflicting_current_and_legacy_order_identity_fails_closed() {
        let process_id = Uuid::from_u128(101);
        let run_id = Uuid::from_u128(202);

        ensure_unambiguous_order_run_identity(process_id, run_id, false).unwrap();
        let error = ensure_unambiguous_order_run_identity(process_id, run_id, true).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(&process_id.to_string()));
        assert!(message.contains(&run_id.to_string()));
        assert!(message.contains("conflicting current and legacy run identity"));
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
        assert!(normalized.contains("l.execution_mode"));
        assert!(normalized.contains("f.source in ('paper', 'live')"));
        assert!(normalized.contains("l.credited_at <= $2"));
        assert!(normalized.contains("f.timestamp_utc <= $2"));
    }

    #[test]
    fn existing_entry_guard_is_process_owned() {
        let normalized = PROCESS_HAS_ENTRY_SQL.to_ascii_lowercase();
        assert!(normalized.contains("where process_id = $1"));
        assert!(normalized.contains("and market_id = $2"));
        assert!(normalized.contains("and action = 'buy'"));
        assert!(normalized.contains("status in ('approved','submitted','filled')"));
        assert!(!normalized.contains("directional_prediction"));
        assert!(!normalized.contains("status = 'rejected'"));
        assert!(!normalized.contains("experiment_id"));
    }

    #[test]
    fn pending_entry_authorization_is_process_run_and_state_scoped() {
        let normalized = AUTHORIZE_PENDING_STRATEGY_DECISION_SQL.to_ascii_lowercase();
        assert!(normalized.contains("where process_id = $1"));
        assert!(normalized.contains("and run_id = $2"));
        assert!(normalized.contains("and decision_id = $3"));
        assert!(normalized.contains("and decision_at = $4"));
        assert!(normalized.contains("and status = 'execution_pending'"));
        assert!(normalized.contains("set status = 'approved'"));
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
    fn shadow_predictive_regime_v2_query_is_exposure_aligned_process_owned_and_bounded() {
        let normalized = SHADOW_PREDICTIVE_REGIME_V2_CANDIDATES_SQL
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase();

        assert!(normalized.contains("process_boundary as materialized"));
        assert!(normalized.contains("where process_id = $1"));
        assert!(normalized.contains("first_process_fills as materialized"));
        assert!(normalized.contains("distinct on (f.order_id)"));
        assert!(normalized.contains("f.process_id = $1"));
        assert!(normalized.contains("f.source = 'paper'"));
        assert!(normalized.contains("f.size > 0"));
        assert!(normalized.contains("f.timestamp_utc >= boundary.created_at"));
        assert!(normalized.contains("first_market_exposures as materialized"));
        assert!(normalized.contains("distinct on (o.market_id)"));
        assert!(normalized.contains("o.created_at >= boundary.created_at"));
        assert!(normalized.contains("o.created_at <= fill.fill_at"));
        assert!(normalized.contains("boundary.created_at as process_created_at"));
        assert!(normalized.contains("o.order_type = 'fok'"));
        assert!(normalized
            .contains("o.raw_payload #>> '{request,metadata,execution_intent}' = 'entry'"));
        assert!(normalized.contains(
            "o.raw_payload #>> '{request,metadata,paper_execution,non_mutating_preview}' = 'false'"
        ));
        assert!(normalized.contains("order by o.market_id, fill.fill_at, fill.fill_id, o.order_id"));
        assert!(
            normalized
                .find("first_market_exposures as materialized")
                .unwrap()
                < normalized
                    .find("left join polymarket.btc_strategy_decisions")
                    .unwrap()
        );
        assert!(normalized.contains("decision.process_id = $1"));
        assert!(normalized.contains("decision.market_id = exposure.market_id"));
        assert!(normalized.contains("else null::uuid end"));
        assert!(normalized.contains("decision.action = 'buy'"));
        assert!(normalized.contains("decision.outcome in ('up', 'down')"));
        assert!(normalized.contains("decision.fair_probability between 0 and 1"));
        assert!(normalized
            .contains("decision.metadata #>> '{prediction,status}' as decision_prediction_status"));
        assert!(normalized.contains("decision.execution_mode = 'paper'"));
        assert!(normalized.contains("decision.decision_at <= exposure.fill_at"));
        assert!(normalized.contains("exposure.fill_at < exposure.label_available_at"));
        assert!(normalized.contains("left join lateral"));
        assert!(normalized.contains("feature.snapshot_id = decision.snapshot_id"));
        assert!(normalized.contains("feature.market_id = exposure.market_id"));
        assert!(normalized.contains("feature.features #>> '{process_id}' = $1::text"));
        assert!(normalized.contains("feature.feature_as_of = decision.decision_at"));
        assert!(normalized.contains("feature.feature_as_of >= exposure.process_created_at"));
        assert!(normalized.contains("feature.feature_as_of <= $2"));
        assert!(normalized.contains("feature.feature_schema_version"));
        assert!(normalized.contains("feature.feature_hash"));
        assert!(normalized.contains("feature.fair_up_probability"));
        assert!(normalized.contains("feature.fair_up_lower"));
        assert!(normalized.contains("feature.fair_up_upper"));
        assert!(normalized.contains("offset 0"));
        assert!(normalized.contains(
            "order by label_available_at desc, fill_at desc, fill_id desc, market_id desc limit $3"
        ));
        assert!(normalized.contains("order by label_available_at, fill_at, fill_id, market_id"));
        assert!(!normalized.contains("experiment_id"));
        assert!(!normalized.contains("decision.status"));
    }

    #[test]
    fn latest_shadow_predictive_regime_v2_state_query_is_schema_isolated_and_process_owned() {
        let normalized = LATEST_SHADOW_PREDICTIVE_REGIME_V2_STATE_SQL
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase();

        assert!(normalized.contains("d.process_id = $1"));
        assert!(normalized.contains(
            "shadow_predictive_regime_circuit_breaker,schema_version}' = 'shadow_predictive_regime_circuit_breaker_v2'"
        ));
        assert!(normalized.contains("shadow_predictive_regime_circuit_breaker,mode}' = 'shadow'"));
        assert!(normalized
            .contains("shadow_predictive_regime_circuit_breaker,state,config_hash}' = $2"));
        assert!(normalized.contains("d.decision_at <= $3"));
        assert!(normalized.contains("state_checkpoint_eligible}' = 'true'"));
        assert!(normalized.contains("refresh_pending}', 'false' ) = 'false'"));
        assert!(normalized.contains("telemetry_error}"));
        assert!(!normalized.contains("experiment_id"));
        assert!(!normalized.contains("d.config_hash"));
    }

    #[test]
    fn shadow_predictive_regime_v2_row_validates_guard_and_postgres_timestamp_precision() {
        let process_id = Uuid::from_u128(7_000);
        let row = predictive_regime_v2_candidate_row(process_id);
        let candidate = shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(60_000),
            None,
            row,
        )
        .unwrap();

        assert_eq!(candidate.market_id, "v2-exposure-market");
        assert_eq!(candidate.order_id, "paper-v2-exposure-order");
        assert_eq!(candidate.decision_outcome, BtcOutcome::Up);
        assert_eq!(candidate.resolved_outcome, BtcOutcome::Down);
        assert_eq!(candidate.selected_point_probability, dec!(0.65));
        assert_eq!(
            candidate.source,
            ShadowPredictiveRegimeV2CandidateSource::ActualPaperFill
        );
    }

    #[test]
    fn shadow_predictive_regime_v2_row_preserves_fair_value_and_directional_projections() {
        let process_id = Uuid::from_u128(7_000);
        for (outcome, expected_point, expected_conservative) in [
            (BtcOutcome::Up, dec!(0.65), dec!(0.58)),
            (BtcOutcome::Down, dec!(0.35), dec!(0.28)),
        ] {
            let fair_value_row =
                predictive_regime_v2_candidate_row_for_outcome(process_id, outcome);
            let fair_value_candidate = shadow_predictive_regime_v2_candidate_from_row(
                process_id,
                Duration::milliseconds(60_000),
                None,
                fair_value_row,
            )
            .unwrap();
            assert_eq!(
                fair_value_candidate.selected_point_probability,
                expected_point
            );

            let mut directional_row =
                predictive_regime_v2_candidate_row_for_outcome(process_id, outcome);
            directional_row.decision_prediction_status = Some("directional_prediction".to_string());
            directional_row.order_request["metadata"]["prediction"] =
                serde_json::json!({ "status": "directional_prediction" });
            directional_row.decision_fair_probability = Some(expected_point);
            let directional_candidate = shadow_predictive_regime_v2_candidate_from_row(
                process_id,
                Duration::milliseconds(60_000),
                None,
                directional_row,
            )
            .unwrap();
            assert_eq!(
                directional_candidate.selected_point_probability,
                expected_point
            );
            assert_ne!(expected_point, expected_conservative);
        }
    }

    #[test]
    fn shadow_predictive_regime_v2_row_validates_closed_probability_interval() {
        let process_id = Uuid::from_u128(7_000);
        let mut exact_interval = predictive_regime_v2_candidate_row(process_id);
        exact_interval.fair_up_lower = Some(dec!(0.65));
        exact_interval.fair_up_upper = Some(dec!(0.65));
        exact_interval.decision_fair_probability = Some(dec!(0.65));
        let candidate = shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(60_000),
            None,
            exact_interval,
        )
        .unwrap();
        assert_eq!(candidate.selected_point_probability, dec!(0.65));

        for (lower, point, upper) in [
            (dec!(-0.01), dec!(0.65), dec!(0.72)),
            (dec!(0.58), dec!(0.65), dec!(1.01)),
            (dec!(0.66), dec!(0.65), dec!(0.72)),
            (dec!(0.58), dec!(0.73), dec!(0.72)),
        ] {
            let mut malformed = predictive_regime_v2_candidate_row(process_id);
            malformed.fair_up_lower = Some(lower);
            malformed.fair_up_probability = Some(point);
            malformed.fair_up_upper = Some(upper);
            assert!(shadow_predictive_regime_v2_candidate_from_row(
                process_id,
                Duration::milliseconds(60_000),
                None,
                malformed
            )
            .unwrap_err()
            .to_string()
            .contains("feature fair-value interval is inconsistent"));
        }
    }

    #[test]
    fn shadow_predictive_regime_v2_down_projection_allows_only_one_storage_quantum() {
        let process_id = Uuid::from_u128(7_000);
        let mut one_quantum =
            predictive_regime_v2_candidate_row_for_outcome(process_id, BtcOutcome::Down);
        one_quantum.decision_fair_probability = Some(dec!(0.2800000001));
        let candidate = shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(60_000),
            None,
            one_quantum,
        )
        .unwrap();
        assert_eq!(candidate.selected_point_probability, dec!(0.35));

        let mut two_quanta =
            predictive_regime_v2_candidate_row_for_outcome(process_id, BtcOutcome::Down);
        two_quanta.decision_fair_probability = Some(dec!(0.2800000002));
        assert!(shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(60_000),
            None,
            two_quanta
        )
        .unwrap_err()
        .to_string()
        .contains("persisted decision probability projection is inconsistent"));
    }

    #[test]
    fn shadow_predictive_regime_v2_row_fails_closed_on_tampered_or_unlinked_evidence() {
        let process_id = Uuid::from_u128(7_000);
        let mut tampered_guard = predictive_regime_v2_candidate_row(process_id);
        tampered_guard.order_request["metadata"]["reference_execution_guard"]["evidence_sha256"] =
            serde_json::json!("0".repeat(64));
        assert!(shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(60_000),
            None,
            tampered_guard
        )
        .is_err());

        let mut missing_decision = predictive_regime_v2_candidate_row(process_id);
        missing_decision.decision_id = None;
        assert!(shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(60_000),
            None,
            missing_decision
        )
        .unwrap_err()
        .to_string()
        .contains("no exact guarded decision"));

        let mut preview = predictive_regime_v2_candidate_row(process_id);
        preview.order_request["metadata"]["paper_execution"]["non_mutating_preview"] =
            serde_json::json!(true);
        assert!(shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(60_000),
            None,
            preview
        )
        .is_err());

        let mismatched_age = predictive_regime_v2_candidate_row(process_id);
        assert!(shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(2_000),
            None,
            mismatched_age
        )
        .unwrap_err()
        .to_string()
        .contains("execution guard failed validation"));

        let mut mismatched_feature_hash = predictive_regime_v2_candidate_row(process_id);
        mismatched_feature_hash.feature_hash = Some("0".repeat(64));
        assert!(shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(60_000),
            None,
            mismatched_feature_hash
        )
        .unwrap_err()
        .to_string()
        .contains("guarded evidence identity is inconsistent"));

        let mut mismatched_probability = predictive_regime_v2_candidate_row(process_id);
        mismatched_probability.decision_fair_probability = Some(dec!(0.62));
        assert!(shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(60_000),
            None,
            mismatched_probability
        )
        .unwrap_err()
        .to_string()
        .contains("persisted decision probability projection is inconsistent"));

        let mut wrong_directional_projection = predictive_regime_v2_candidate_row(process_id);
        wrong_directional_projection.decision_prediction_status =
            Some("directional_prediction".to_string());
        wrong_directional_projection.order_request["metadata"]["prediction"] =
            serde_json::json!({ "status": "directional_prediction" });
        assert!(shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(60_000),
            None,
            wrong_directional_projection
        )
        .unwrap_err()
        .to_string()
        .contains("persisted decision probability projection is inconsistent"));

        let mut malformed_interval = predictive_regime_v2_candidate_row(process_id);
        malformed_interval.fair_up_lower = Some(dec!(0.66));
        assert!(shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(60_000),
            None,
            malformed_interval
        )
        .unwrap_err()
        .to_string()
        .contains("feature fair-value interval is inconsistent"));

        let mut missing_bound = predictive_regime_v2_candidate_row(process_id);
        missing_bound.fair_up_upper = None;
        assert!(shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(60_000),
            None,
            missing_bound
        )
        .unwrap_err()
        .to_string()
        .contains("no upper probability bound"));

        let mut unsupported_prediction = predictive_regime_v2_candidate_row(process_id);
        unsupported_prediction.decision_prediction_status = Some("no_prediction".to_string());
        unsupported_prediction.order_request["metadata"]["prediction"] =
            serde_json::json!({ "status": "no_prediction" });
        assert!(shadow_predictive_regime_v2_candidate_from_row(
            process_id,
            Duration::milliseconds(60_000),
            None,
            unsupported_prediction
        )
        .unwrap_err()
        .to_string()
        .contains("unsupported prediction status"));
    }

    #[test]
    fn persisted_shadow_predictive_regime_v2_state_round_trips_and_rejects_v1_schema() {
        let process_id = Uuid::from_u128(7_500);
        let config = predictive_regime_v2_config();
        let candidates = (0_i64..20)
            .map(|index| predictive_regime_v2_candidate(index, dec!(0.90), index < 15))
            .collect::<Vec<_>>();
        let state =
            ShadowPredictiveRegimeV2State::from_candidates(process_id, &config, &candidates)
                .unwrap();
        let evaluation_as_of = candidates.last().unwrap().label_available_at
            + Duration::seconds(1)
            + Duration::nanoseconds(999);
        let postgres_decision_at =
            DateTime::<Utc>::from_timestamp_micros(evaluation_as_of.timestamp_micros()).unwrap();
        let mut evaluation = state.evaluate(&config, evaluation_as_of).unwrap();
        evaluation.state_checkpoint_eligible = true;
        let config_hash = config.config_hash().unwrap();
        let persisted = serde_json::to_value(&evaluation).unwrap();

        let restored = shadow_predictive_regime_v2_state_from_persisted_evaluation(
            process_id,
            Some(&config_hash),
            PersistedShadowPredictiveRegimeV2EvaluationRow {
                decision_at: postgres_decision_at,
                evaluation: persisted.clone(),
            },
        )
        .unwrap();
        assert_eq!(restored, state);

        let mut wrong_schema = persisted;
        wrong_schema["schema_version"] =
            serde_json::json!(SHADOW_PREDICTIVE_REGIME_CIRCUIT_BREAKER_SCHEMA_VERSION);
        assert!(shadow_predictive_regime_v2_state_from_persisted_evaluation(
            process_id,
            Some(&config_hash),
            PersistedShadowPredictiveRegimeV2EvaluationRow {
                decision_at: postgres_decision_at,
                evaluation: wrong_schema,
            },
        )
        .is_err());
    }

    #[test]
    fn shadow_predictive_regime_v2_history_limit_is_strictly_bounded() {
        assert!(shadow_predictive_regime_v2_history_limit(0).is_err());
        assert_eq!(shadow_predictive_regime_v2_history_limit(1).unwrap(), 1);
        assert_eq!(
            shadow_predictive_regime_v2_history_limit(
                MAX_SHADOW_PREDICTIVE_REGIME_V2_HISTORY_CANDIDATES
            )
            .unwrap(),
            i64::from(MAX_SHADOW_PREDICTIVE_REGIME_V2_HISTORY_CANDIDATES)
        );
        assert!(shadow_predictive_regime_v2_history_limit(
            MAX_SHADOW_PREDICTIVE_REGIME_V2_HISTORY_CANDIDATES + 1
        )
        .is_err());
    }

    #[test]
    fn persisted_shadow_predictive_regime_state_round_trips_degraded_evidence_at_postgres_precision(
    ) {
        let process_id = Uuid::from_u128(450);
        let candidate_origin = Utc
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
        let candidates = (0_i64..21)
            .map(|index| {
                let label_available_at = candidate_origin + Duration::minutes(index * 5);
                ShadowPredictiveRegimeCandidate {
                    market_id: format!("postgres-precision-market-{index}"),
                    decision_id: Uuid::from_u128(u128::try_from(index + 1).unwrap()),
                    decision_outcome: BtcOutcome::Up,
                    resolved_outcome: BtcOutcome::Down,
                    selected_point_probability: dec!(0.90),
                    decision_at: label_available_at - Duration::minutes(4),
                    label_available_at,
                }
            })
            .collect::<Vec<_>>();
        let state =
            ShadowPredictiveRegimeState::from_candidates(process_id, &config, &candidates).unwrap();
        assert!(state.degraded);
        assert_eq!(state.resolved_markets_observed, 21);
        assert_eq!(state.rolling_candidates.len(), 20);
        assert_eq!(state.consecutive_degradation_markets, 2);

        let evaluation_as_of = candidates.last().unwrap().label_available_at
            + Duration::seconds(1)
            + Duration::nanoseconds(999);
        let postgres_decision_at =
            DateTime::<Utc>::from_timestamp_micros(evaluation_as_of.timestamp_micros()).unwrap();
        assert_ne!(evaluation_as_of, postgres_decision_at);
        assert_eq!(
            evaluation_as_of.timestamp_micros(),
            postgres_decision_at.timestamp_micros()
        );

        let mut evaluation = state.evaluate(&config, evaluation_as_of).unwrap();
        evaluation.state_checkpoint_eligible = true;
        let config_hash = config.config_hash().unwrap();
        let persisted_evaluation = serde_json::to_value(evaluation).unwrap();

        let restored = shadow_predictive_regime_state_from_persisted_evaluation(
            process_id,
            Some(&config_hash),
            PersistedShadowPredictiveRegimeEvaluationRow {
                decision_at: postgres_decision_at,
                evaluation: persisted_evaluation.clone(),
            },
        )
        .unwrap();

        assert_eq!(restored, state);
        let restored_evaluation = restored.evaluate(&config, evaluation_as_of).unwrap();
        assert!(restored_evaluation.sample_ready);
        assert!(restored_evaluation.degraded);
        assert!(restored_evaluation.would_defer);

        let error = shadow_predictive_regime_state_from_persisted_evaluation(
            process_id,
            Some(&config_hash),
            PersistedShadowPredictiveRegimeEvaluationRow {
                decision_at: postgres_decision_at + Duration::microseconds(1),
                evaluation: persisted_evaluation,
            },
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("persisted shadow predictive-regime evaluation is not aligned"));
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
                entry_policy: Default::default(),
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
    fn decision_edge_projection_accepts_signed_validation_edge_only_with_explicit_policy() {
        for outcome in [BtcOutcome::Up, BtcOutcome::Down] {
            let prediction = BtcStrategyPrediction::DirectionalPrediction {
                outcome,
                probability: dec!(0.92),
                conservative_probability: dec!(0.92),
                minimum_conservative_probability: dec!(0.89),
                probability_uncertainty: Decimal::ZERO,
                executable_price: Some(dec!(0.95)),
                direct_taker_fee_per_share: Some(dec!(0.002)),
                direct_net_edge_per_share: Some(dec!(-0.032)),
                entry_policy: BtcDirectionalModelEntryPolicy::ExecuteDirectionalPrediction,
            };
            let mut decision = approved_decision(outcome, Some(prediction), dec!(-0.032));
            decision.approved_intent.as_mut().unwrap().strategy_version =
                BTC_DIRECTIONAL_MODEL_STRATEGY_VERSION.to_string();
            let selected_edge = match outcome {
                BtcOutcome::Up => decision.up_edge.as_mut().unwrap(),
                BtcOutcome::Down => decision.down_edge.as_mut().unwrap(),
            };
            selected_edge.executable_price = dec!(0.95);

            let projection = decision_edge_projection(&decision).unwrap();
            assert_eq!(projection.fair_probability, Some(dec!(0.92)));
            assert_eq!(projection.gross_edge_per_share, Some(dec!(-0.03)));
            assert_eq!(projection.fee_per_share, Some(dec!(0.002)));
            assert_eq!(projection.net_edge_per_share, Some(dec!(-0.032)));

            let mut default_policy = decision.clone();
            let Some(BtcStrategyPrediction::DirectionalPrediction { entry_policy, .. }) =
                default_policy.prediction.as_mut()
            else {
                unreachable!("test decision has a directional prediction")
            };
            *entry_policy = BtcDirectionalModelEntryPolicy::RequirePositiveDirectEdge;
            assert!(decision_edge_projection(&default_policy).is_err());

            let mut non_model_strategy = decision.clone();
            non_model_strategy
                .approved_intent
                .as_mut()
                .unwrap()
                .strategy_version = "non-model-strategy".to_string();
            assert!(decision_edge_projection(&non_model_strategy).is_err());

            let mut tampered_edge = decision;
            let Some(BtcStrategyPrediction::DirectionalPrediction {
                direct_net_edge_per_share,
                ..
            }) = tampered_edge.prediction.as_mut()
            else {
                unreachable!("test decision has a directional prediction")
            };
            *direct_net_edge_per_share = Some(dec!(-0.031));
            assert!(decision_edge_projection(&tampered_edge).is_err());
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

    fn settlement_record() -> BtcSettlementRecord {
        let at = Utc::now();
        BtcSettlementRecord {
            process_id: Uuid::from_u128(3),
            run_id: Uuid::from_u128(2),
            settlement_id: Uuid::from_u128(1),
            execution_mode: "paper".to_string(),
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
        validate_settlement_record(&winner, BtcExecutionMode::Paper).unwrap();

        let mut gamma_winner = winner.clone();
        gamma_winner.official_resolution_source = "gamma_rest_reconciliation".to_string();
        validate_settlement_record(&gamma_winner, BtcExecutionMode::Paper).unwrap();

        let mut loser = winner.clone();
        loser.token_id = "down-token".to_string();
        loser.payout = Decimal::ZERO;
        loser.net_pnl = dec!(-2.1);
        validate_settlement_record(&loser, BtcExecutionMode::Paper).unwrap();

        let mut unsupported = winner.clone();
        unsupported.official_resolution_source = "local_chainlink_label".to_string();
        assert!(validate_settlement_record(&unsupported, BtcExecutionMode::Paper).is_err());

        let mut missing_fill_attribution = winner.clone();
        missing_fill_attribution.fill_ids = serde_json::json!([]);
        assert!(
            validate_settlement_record(&missing_fill_attribution, BtcExecutionMode::Paper).is_err()
        );

        let mut negative = winner.clone();
        negative.payout = dec!(-1);
        assert!(validate_settlement_record(&negative, BtcExecutionMode::Paper).is_err());

        let mut wrong_binary_payout = winner;
        wrong_binary_payout.payout = dec!(4.99);
        assert!(validate_settlement_record(&wrong_binary_payout, BtcExecutionMode::Paper).is_err());

        let mut inconsistent_net_pnl = settlement_record();
        inconsistent_net_pnl.net_pnl = Decimal::ZERO;
        assert!(
            validate_settlement_record(&inconsistent_net_pnl, BtcExecutionMode::Paper).is_err()
        );

        let mut wrong_mode = settlement_record();
        wrong_mode.execution_mode = "live".to_string();
        assert!(validate_settlement_record(&wrong_mode, BtcExecutionMode::Paper).is_err());
        validate_settlement_record(&wrong_mode, BtcExecutionMode::Live).unwrap();
    }

    #[test]
    fn live_zero_payout_evidence_requires_an_exact_official_loss() {
        let mut loser = settlement_record();
        loser.execution_mode = "live".to_string();
        loser.token_id = "down-token".to_string();
        loser.payout = Decimal::ZERO;
        loser.net_pnl = dec!(-2.1);
        let config_hash = "a".repeat(64);

        let evidence = live_zero_payout_settlement_evidence(&loser, &config_hash).unwrap();
        assert_eq!(evidence["proof_type"], "btc_official_zero_payout_loss");
        assert_eq!(evidence["exchange_cash_credit_applied"], false);
        assert_eq!(evidence["payout"], "0");
        assert_eq!(evidence["net_pnl"], "-2.1");

        let mut winner = loser.clone();
        winner.token_id = winner.official_winning_token_id.clone();
        winner.payout = winner.filled_size;
        winner.net_pnl = winner.payout - winner.entry_notional - winner.entry_fees;
        assert!(live_zero_payout_settlement_evidence(&winner, &config_hash).is_err());
        assert!(live_zero_payout_settlement_evidence(&loser, "not-a-hash").is_err());
    }

    #[test]
    fn live_zero_payout_recognition_is_loser_only_and_resolution_scoped() {
        let sql = RECOGNIZE_LIVE_ZERO_PAYOUT_SETTLEMENT_SQL.to_ascii_lowercase();
        assert!(sql.contains("settlement.execution_mode = 'live'"));
        assert!(sql.contains("settlement.credit_status = 'pending'"));
        assert!(sql.contains("settlement.payout = 0"));
        assert!(sql.contains("settlement.token_id <> settlement.official_winning_token_id"));
        assert!(sql.contains("btc_official_resolution_watches"));
        assert!(sql.contains("watch.status in ('resolved', 'resolved_late')"));
    }

    #[test]
    fn official_resolution_source_contract_accepts_exchange_reconciliation_only() {
        assert!(is_supported_official_resolution_source("clob_websocket"));
        assert!(is_supported_official_resolution_source(
            "clob_rest_reconciliation"
        ));
        assert!(is_supported_official_resolution_source(
            "gamma_rest_reconciliation"
        ));
        assert!(!is_supported_official_resolution_source(
            "local_chainlink_label"
        ));
        assert!(!is_supported_official_resolution_source("unknown"));
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
