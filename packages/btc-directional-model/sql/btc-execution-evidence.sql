WITH eligible_markets AS MATERIALIZED (
  SELECT
    market.market_id,
    market.window_start,
    market.window_end,
    market.official_outcome,
    CASE WHEN market.official_outcome = 'up' THEN 1 ELSE 0 END AS label_up,
    market.min_tick_size::double precision AS min_tick_size,
    market.min_order_size::double precision AS min_order_size,
    COALESCE(market.fee_rate, 0)::double precision AS fee_rate
  FROM polymarket.btc_interval_markets market
  WHERE market.window_start >= %(batch_start)s
    AND market.window_start < %(batch_end)s
    AND market.validation_status = 'valid'
    AND market.official_outcome IN ('up', 'down')
),
candidate_snapshots AS MATERIALIZED (
  SELECT
    market.market_id,
    market.window_start,
    market.window_end,
    market.official_outcome,
    market.label_up,
    market.min_tick_size,
    market.min_order_size,
    market.fee_rate,
    snapshot.sampled_at AS observed_at,
    extract(epoch FROM snapshot.sampled_at - market.window_start)::integer
      AS seconds_elapsed,
    snapshot.artifact_id::text AS artifact_id,
    snapshot.schema_version,
    snapshot.up_provider_received_at,
    snapshot.up_best_bid::double precision AS up_best_bid,
    snapshot.up_best_ask::double precision AS up_best_ask,
    snapshot.up_best_bid_size::double precision AS up_best_bid_size,
    snapshot.up_best_ask_size::double precision AS up_best_ask_size,
    snapshot.up_bid_depth::double precision AS up_bid_depth,
    snapshot.up_ask_depth::double precision AS up_ask_depth,
    snapshot.up_ask_vwap_5::double precision AS up_ask_vwap_5,
    snapshot.up_imbalance::double precision AS up_imbalance,
    snapshot.down_provider_received_at,
    snapshot.down_best_bid::double precision AS down_best_bid,
    snapshot.down_best_ask::double precision AS down_best_ask,
    snapshot.down_best_bid_size::double precision AS down_best_bid_size,
    snapshot.down_best_ask_size::double precision AS down_best_ask_size,
    snapshot.down_bid_depth::double precision AS down_bid_depth,
    snapshot.down_ask_depth::double precision AS down_ask_depth,
    snapshot.down_ask_vwap_5::double precision AS down_ask_vwap_5,
    snapshot.down_imbalance::double precision AS down_imbalance,
    snapshot.quality_flags
  FROM eligible_markets market
  JOIN polymarket.btc_market_execution_snapshots snapshot
    ON snapshot.market_id = market.market_id
   AND snapshot.sampled_at >= %(batch_start)s
   AND snapshot.sampled_at < %(batch_end)s
   AND snapshot.sampled_at >=
     market.window_start + (%(min_seconds_after_open)s * interval '1 second')
   AND snapshot.sampled_at <=
     market.window_start + (%(max_seconds_after_open)s * interval '1 second')
   AND mod(
     round(
       extract(epoch FROM snapshot.sampled_at - market.window_start) * 1000
     )::bigint,
     %(sample_interval_milliseconds)s
   ) = 0
   AND snapshot.schema_version = %(snapshot_schema_version)s
  JOIN polymarket.backfill_artifacts snapshot_artifact
    ON snapshot_artifact.artifact_id = snapshot.artifact_id
   AND snapshot_artifact.status = 'completed'
   AND snapshot_artifact.ingester_key =
     'polymarket_btc_five_minute_execution_snapshots'
),
validity AS (
  SELECT
    candidate.*,
    (
      candidate.up_provider_received_at IS NOT NULL
      AND candidate.up_provider_received_at <= candidate.observed_at
    ) AS up_provider_causal,
    (
      candidate.down_provider_received_at IS NOT NULL
      AND candidate.down_provider_received_at <= candidate.observed_at
    ) AS down_provider_causal,
    (
      candidate.up_best_bid IS NOT NULL
      AND candidate.up_best_ask IS NOT NULL
      AND candidate.up_best_bid_size IS NOT NULL
      AND candidate.up_best_ask_size IS NOT NULL
      AND candidate.up_bid_depth IS NOT NULL
      AND candidate.up_ask_depth IS NOT NULL
      AND candidate.up_ask_vwap_5 IS NOT NULL
      AND candidate.up_imbalance IS NOT NULL
    ) AS up_fields_complete,
    (
      candidate.down_best_bid IS NOT NULL
      AND candidate.down_best_ask IS NOT NULL
      AND candidate.down_best_bid_size IS NOT NULL
      AND candidate.down_best_ask_size IS NOT NULL
      AND candidate.down_bid_depth IS NOT NULL
      AND candidate.down_ask_depth IS NOT NULL
      AND candidate.down_ask_vwap_5 IS NOT NULL
      AND candidate.down_imbalance IS NOT NULL
    ) AS down_fields_complete
  FROM candidate_snapshots candidate
),
side_cohorts AS (
  SELECT
    validity.*,
    (
      validity.up_provider_causal
      AND validity.up_fields_complete
      AND (validity.quality_flags & 17) = 0
    ) AS up_side_valid,
    (
      validity.down_provider_causal
      AND validity.down_fields_complete
      AND (validity.quality_flags & 34) = 0
    ) AS down_side_valid
  FROM validity
)
SELECT
  cohort.market_id,
  cohort.window_start,
  cohort.window_end,
  cohort.official_outcome,
  cohort.label_up,
  cohort.min_tick_size,
  cohort.min_order_size,
  cohort.fee_rate,
  cohort.observed_at,
  cohort.seconds_elapsed,
  cohort.artifact_id,
  cohort.schema_version,
  cohort.up_provider_received_at,
  cohort.up_best_bid,
  cohort.up_best_ask,
  cohort.up_best_bid_size,
  cohort.up_best_ask_size,
  cohort.up_bid_depth,
  cohort.up_ask_depth,
  cohort.up_ask_vwap_5,
  cohort.up_imbalance,
  cohort.down_provider_received_at,
  cohort.down_best_bid,
  cohort.down_best_ask,
  cohort.down_best_bid_size,
  cohort.down_best_ask_size,
  cohort.down_bid_depth,
  cohort.down_ask_depth,
  cohort.down_ask_vwap_5,
  cohort.down_imbalance,
  cohort.quality_flags,
  cohort.up_provider_causal,
  cohort.down_provider_causal,
  cohort.up_fields_complete,
  cohort.down_fields_complete,
  cohort.up_side_valid,
  cohort.down_side_valid,
  (
    cohort.up_side_valid
    AND (cohort.quality_flags & 4) = 0
    AND cohort.up_provider_received_at >=
      cohort.observed_at - (%(freshness_seconds)s * interval '1 second')
  ) AS up_side_fresh,
  (
    cohort.down_side_valid
    AND (cohort.quality_flags & 8) = 0
    AND cohort.down_provider_received_at >=
      cohort.observed_at - (%(freshness_seconds)s * interval '1 second')
  ) AS down_side_fresh,
  (
    cohort.up_side_valid
    AND (
      (cohort.quality_flags & 4) <> 0
      OR cohort.up_provider_received_at <
        cohort.observed_at - (%(freshness_seconds)s * interval '1 second')
    )
  ) AS up_stale_initialized,
  (
    cohort.down_side_valid
    AND (
      (cohort.quality_flags & 8) <> 0
      OR cohort.down_provider_received_at <
        cohort.observed_at - (%(freshness_seconds)s * interval '1 second')
    )
  ) AS down_stale_initialized,
  (
    cohort.up_side_valid
    AND cohort.down_side_valid
    AND (cohort.quality_flags & 63) = 0
    AND cohort.up_provider_received_at >=
      cohort.observed_at - (%(freshness_seconds)s * interval '1 second')
    AND cohort.down_provider_received_at >=
      cohort.observed_at - (%(freshness_seconds)s * interval '1 second')
  ) AS strict_both_side_eligible
FROM side_cohorts cohort
ORDER BY cohort.market_id, cohort.observed_at;
