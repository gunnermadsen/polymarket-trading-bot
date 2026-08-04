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
decision_grid AS MATERIALIZED (
  SELECT
    market.*,
    decision.seconds_elapsed,
    market.window_start + decision.seconds_elapsed * interval '1 second'
      AS decision_at
  FROM eligible_markets market
  CROSS JOIN generate_series(
    %(minimum_decision_second)s,
    %(maximum_decision_second)s,
    %(sample_interval_seconds)s
  ) AS decision(seconds_elapsed)
),
scenario_grid AS MATERIALIZED (
  SELECT *
  FROM unnest(
    %(scenario_keys)s::text[],
    %(arrival_latency_milliseconds)s::integer[],
    %(visible_depth_fractions)s::double precision[],
    %(raw_quantities_required)s::double precision[],
    %(price_stress_methods)s::text[],
    %(price_stress_exactness)s::boolean[]
  ) AS scenario(
    scenario_key,
    configured_arrival_latency_ms,
    visible_depth_fraction,
    raw_quantity_required,
    price_stress_method,
    price_stress_exact
  )
)
SELECT
  decision.market_id,
  decision.window_start,
  decision.window_end,
  decision.official_outcome,
  decision.label_up,
  decision.min_tick_size,
  decision.min_order_size,
  decision.fee_rate,
  decision.decision_at,
  decision.seconds_elapsed::integer,
  scenario.scenario_key,
  scenario.configured_arrival_latency_ms,
  scenario.visible_depth_fraction,
  scenario.raw_quantity_required,
  scenario.price_stress_method,
  10::integer AS price_stress_vwap_quantity,
  scenario.price_stress_exact,
  snapshot.sampled_at AS snapshot_at,
  CASE
    WHEN snapshot.sampled_at IS NULL THEN NULL
    ELSE round(
      extract(epoch FROM snapshot.sampled_at - decision.decision_at) * 1000
    )::integer
  END AS realized_arrival_latency_ms,
  snapshot.artifact_id::text AS artifact_id,
  snapshot.schema_version,
  snapshot.up_source_row_number,
  snapshot.up_source_timestamp,
  snapshot.up_provider_received_at,
  snapshot.up_best_bid::double precision AS up_best_bid,
  snapshot.up_best_ask::double precision AS up_best_ask,
  snapshot.up_best_bid_size::double precision AS up_best_bid_size,
  snapshot.up_best_ask_size::double precision AS up_best_ask_size,
  snapshot.up_bid_depth::double precision AS up_bid_depth,
  snapshot.up_ask_depth::double precision AS up_ask_depth,
  snapshot.up_ask_vwap_1::double precision AS up_ask_vwap_1,
  snapshot.up_ask_vwap_5::double precision AS up_ask_vwap_5,
  snapshot.up_ask_vwap_10::double precision AS up_ask_vwap_10,
  snapshot.up_imbalance::double precision AS up_imbalance,
  snapshot.down_source_row_number,
  snapshot.down_source_timestamp,
  snapshot.down_provider_received_at,
  snapshot.down_best_bid::double precision AS down_best_bid,
  snapshot.down_best_ask::double precision AS down_best_ask,
  snapshot.down_best_bid_size::double precision AS down_best_bid_size,
  snapshot.down_best_ask_size::double precision AS down_best_ask_size,
  snapshot.down_bid_depth::double precision AS down_bid_depth,
  snapshot.down_ask_depth::double precision AS down_ask_depth,
  snapshot.down_ask_vwap_1::double precision AS down_ask_vwap_1,
  snapshot.down_ask_vwap_5::double precision AS down_ask_vwap_5,
  snapshot.down_ask_vwap_10::double precision AS down_ask_vwap_10,
  snapshot.down_imbalance::double precision AS down_imbalance,
  snapshot.quality_flags,
  COALESCE(
    snapshot.up_provider_received_at IS NOT NULL
    AND snapshot.up_provider_received_at <= snapshot.sampled_at,
    false
  ) AS up_provider_causal,
  COALESCE(
    snapshot.down_provider_received_at IS NOT NULL
    AND snapshot.down_provider_received_at <= snapshot.sampled_at,
    false
  ) AS down_provider_causal,
  COALESCE(
    snapshot.up_best_bid IS NOT NULL
    AND snapshot.up_best_ask IS NOT NULL
    AND snapshot.up_best_bid_size IS NOT NULL
    AND snapshot.up_best_ask_size IS NOT NULL
    AND snapshot.up_bid_depth IS NOT NULL
    AND snapshot.up_ask_depth IS NOT NULL
    AND snapshot.up_ask_vwap_1 IS NOT NULL
    AND snapshot.up_ask_vwap_5 IS NOT NULL
    AND snapshot.up_ask_vwap_10 IS NOT NULL
    AND snapshot.up_imbalance IS NOT NULL,
    false
  ) AS up_fields_complete,
  COALESCE(
    snapshot.down_best_bid IS NOT NULL
    AND snapshot.down_best_ask IS NOT NULL
    AND snapshot.down_best_bid_size IS NOT NULL
    AND snapshot.down_best_ask_size IS NOT NULL
    AND snapshot.down_bid_depth IS NOT NULL
    AND snapshot.down_ask_depth IS NOT NULL
    AND snapshot.down_ask_vwap_1 IS NOT NULL
    AND snapshot.down_ask_vwap_5 IS NOT NULL
    AND snapshot.down_ask_vwap_10 IS NOT NULL
    AND snapshot.down_imbalance IS NOT NULL,
    false
  ) AS down_fields_complete,
  COALESCE(
    snapshot.up_provider_received_at IS NOT NULL
    AND snapshot.up_provider_received_at <= snapshot.sampled_at
    AND snapshot.up_best_bid IS NOT NULL
    AND snapshot.up_best_ask IS NOT NULL
    AND snapshot.up_best_bid_size IS NOT NULL
    AND snapshot.up_best_ask_size IS NOT NULL
    AND snapshot.up_bid_depth IS NOT NULL
    AND snapshot.up_ask_depth IS NOT NULL
    AND snapshot.up_ask_vwap_5 IS NOT NULL
    AND snapshot.up_imbalance IS NOT NULL
    AND (snapshot.quality_flags & 17) = 0,
    false
  ) AS up_side_valid,
  COALESCE(
    snapshot.down_provider_received_at IS NOT NULL
    AND snapshot.down_provider_received_at <= snapshot.sampled_at
    AND snapshot.down_best_bid IS NOT NULL
    AND snapshot.down_best_ask IS NOT NULL
    AND snapshot.down_best_bid_size IS NOT NULL
    AND snapshot.down_best_ask_size IS NOT NULL
    AND snapshot.down_bid_depth IS NOT NULL
    AND snapshot.down_ask_depth IS NOT NULL
    AND snapshot.down_ask_vwap_5 IS NOT NULL
    AND snapshot.down_imbalance IS NOT NULL
    AND (snapshot.quality_flags & 34) = 0,
    false
  ) AS down_side_valid,
  COALESCE(
    snapshot.up_provider_received_at IS NOT NULL
    AND snapshot.up_provider_received_at <= snapshot.sampled_at
    AND snapshot.up_best_bid IS NOT NULL
    AND snapshot.up_best_ask IS NOT NULL
    AND snapshot.up_best_bid_size IS NOT NULL
    AND snapshot.up_best_ask_size IS NOT NULL
    AND snapshot.up_bid_depth IS NOT NULL
    AND snapshot.up_ask_depth IS NOT NULL
    AND snapshot.up_ask_vwap_5 IS NOT NULL
    AND snapshot.up_imbalance IS NOT NULL
    AND snapshot.up_provider_received_at >=
      snapshot.sampled_at - (%(freshness_seconds)s * interval '1 second')
    AND (snapshot.quality_flags & 21) = 0,
    false
  ) AS up_side_fresh,
  COALESCE(
    snapshot.down_provider_received_at IS NOT NULL
    AND snapshot.down_provider_received_at <= snapshot.sampled_at
    AND snapshot.down_best_bid IS NOT NULL
    AND snapshot.down_best_ask IS NOT NULL
    AND snapshot.down_best_bid_size IS NOT NULL
    AND snapshot.down_best_ask_size IS NOT NULL
    AND snapshot.down_bid_depth IS NOT NULL
    AND snapshot.down_ask_depth IS NOT NULL
    AND snapshot.down_ask_vwap_5 IS NOT NULL
    AND snapshot.down_imbalance IS NOT NULL
    AND snapshot.down_provider_received_at >=
      snapshot.sampled_at - (%(freshness_seconds)s * interval '1 second')
    AND (snapshot.quality_flags & 42) = 0,
    false
  ) AS down_side_fresh,
  COALESCE(
    snapshot.up_provider_received_at IS NOT NULL
    AND snapshot.down_provider_received_at IS NOT NULL
    AND snapshot.up_provider_received_at <= snapshot.sampled_at
    AND snapshot.down_provider_received_at <= snapshot.sampled_at
    AND snapshot.up_provider_received_at >=
      snapshot.sampled_at - (%(freshness_seconds)s * interval '1 second')
    AND snapshot.down_provider_received_at >=
      snapshot.sampled_at - (%(freshness_seconds)s * interval '1 second')
    AND snapshot.up_best_bid IS NOT NULL
    AND snapshot.up_best_ask IS NOT NULL
    AND snapshot.up_best_bid_size IS NOT NULL
    AND snapshot.up_best_ask_size IS NOT NULL
    AND snapshot.up_bid_depth IS NOT NULL
    AND snapshot.up_ask_depth IS NOT NULL
    AND snapshot.up_ask_vwap_5 IS NOT NULL
    AND snapshot.up_imbalance IS NOT NULL
    AND snapshot.down_best_bid IS NOT NULL
    AND snapshot.down_best_ask IS NOT NULL
    AND snapshot.down_best_bid_size IS NOT NULL
    AND snapshot.down_best_ask_size IS NOT NULL
    AND snapshot.down_bid_depth IS NOT NULL
    AND snapshot.down_ask_depth IS NOT NULL
    AND snapshot.down_ask_vwap_5 IS NOT NULL
    AND snapshot.down_imbalance IS NOT NULL
    AND (snapshot.quality_flags & 63) = 0,
    false
  ) AS strict_both_side_eligible_5,
  COALESCE(
    snapshot.up_provider_received_at IS NOT NULL
    AND snapshot.down_provider_received_at IS NOT NULL
    AND snapshot.up_provider_received_at <= snapshot.sampled_at
    AND snapshot.down_provider_received_at <= snapshot.sampled_at
    AND snapshot.up_provider_received_at >=
      snapshot.sampled_at - (%(freshness_seconds)s * interval '1 second')
    AND snapshot.down_provider_received_at >=
      snapshot.sampled_at - (%(freshness_seconds)s * interval '1 second')
    AND snapshot.up_best_bid IS NOT NULL
    AND snapshot.up_best_ask IS NOT NULL
    AND snapshot.up_best_bid_size IS NOT NULL
    AND snapshot.up_best_ask_size IS NOT NULL
    AND snapshot.up_bid_depth IS NOT NULL
    AND snapshot.up_ask_depth IS NOT NULL
    AND snapshot.up_ask_vwap_1 IS NOT NULL
    AND snapshot.up_ask_vwap_5 IS NOT NULL
    AND snapshot.up_ask_vwap_10 IS NOT NULL
    AND snapshot.up_imbalance IS NOT NULL
    AND snapshot.down_best_bid IS NOT NULL
    AND snapshot.down_best_ask IS NOT NULL
    AND snapshot.down_best_bid_size IS NOT NULL
    AND snapshot.down_best_ask_size IS NOT NULL
    AND snapshot.down_bid_depth IS NOT NULL
    AND snapshot.down_ask_depth IS NOT NULL
    AND snapshot.down_ask_vwap_1 IS NOT NULL
    AND snapshot.down_ask_vwap_5 IS NOT NULL
    AND snapshot.down_ask_vwap_10 IS NOT NULL
    AND snapshot.down_imbalance IS NOT NULL
    AND (snapshot.quality_flags & 255) = 0,
    false
  ) AS strict_both_side_eligible_10
FROM decision_grid decision
CROSS JOIN scenario_grid scenario
LEFT JOIN LATERAL (
  SELECT source.*
  FROM polymarket.btc_market_execution_snapshots source
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = source.artifact_id
   AND artifact.ingester_key =
     'polymarket_btc_five_minute_execution_snapshots'
   AND artifact.provider = 'pmxt_v2_execution_snapshots'
   AND artifact.status = 'completed'
  WHERE source.market_id = decision.market_id
    AND source.schema_version = 'btc5m-book-250ms-v1'
    AND source.sampled_at >= %(batch_start)s
    AND source.sampled_at < %(batch_end)s
    AND source.sampled_at >= decision.decision_at
      + scenario.configured_arrival_latency_ms * interval '1 millisecond'
    AND source.sampled_at < decision.decision_at
      + %(sample_interval_seconds)s * interval '1 second'
  ORDER BY source.sampled_at, source.artifact_id
  LIMIT 1
) snapshot ON true
ORDER BY
  decision.window_start,
  decision.market_id,
  decision.seconds_elapsed,
  scenario.configured_arrival_latency_ms;
