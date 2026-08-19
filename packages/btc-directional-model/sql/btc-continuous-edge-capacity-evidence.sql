WITH eligible_markets AS MATERIALIZED (
  SELECT
    market.market_id,
    market.window_start,
    market.window_end,
    CASE WHEN market.official_outcome = 'up' THEN 1 ELSE 0 END AS label_up,
    COALESCE(market.fee_rate, 0)::double precision AS fee_rate
  FROM polymarket.btc_interval_markets market
  WHERE market.window_start >= %(batch_start)s
    AND market.window_start < %(batch_end)s
    AND market.validation_status = 'valid'
    AND market.official_outcome IN ('up', 'down')
)
SELECT
  market.market_id,
  market.window_start,
  market.window_end,
  market.label_up,
  market.fee_rate,
  snapshot.sampled_at AS observed_at,
  extract(epoch FROM snapshot.sampled_at - market.window_start)::integer AS seconds_elapsed,
  snapshot.artifact_id::text AS artifact_id,
  snapshot.schema_version,
  snapshot.up_provider_received_at,
  snapshot.up_best_ask::double precision AS up_best_ask,
  snapshot.up_ask_depth::double precision AS up_ask_depth,
  snapshot.up_ask_vwap_5::double precision AS up_ask_vwap_5,
  snapshot.up_ask_vwap_10::double precision AS up_ask_vwap_10,
  snapshot.up_ask_vwap_15::double precision AS up_ask_vwap_15,
  snapshot.up_ask_vwap_20::double precision AS up_ask_vwap_20,
  snapshot.up_ask_vwap_25::double precision AS up_ask_vwap_25,
  snapshot.up_ask_vwap_30::double precision AS up_ask_vwap_30,
  snapshot.up_ask_vwap_40::double precision AS up_ask_vwap_40,
  snapshot.up_ask_vwap_50::double precision AS up_ask_vwap_50,
  snapshot.up_ask_vwap_75::double precision AS up_ask_vwap_75,
  snapshot.up_ask_vwap_100::double precision AS up_ask_vwap_100,
  snapshot.up_ask_vwap_125::double precision AS up_ask_vwap_125,
  snapshot.up_ask_vwap_150::double precision AS up_ask_vwap_150,
  snapshot.up_ask_vwap_175::double precision AS up_ask_vwap_175,
  snapshot.up_ask_vwap_200::double precision AS up_ask_vwap_200,
  snapshot.down_provider_received_at,
  snapshot.down_best_ask::double precision AS down_best_ask,
  snapshot.down_ask_depth::double precision AS down_ask_depth,
  snapshot.down_ask_vwap_5::double precision AS down_ask_vwap_5,
  snapshot.down_ask_vwap_10::double precision AS down_ask_vwap_10,
  snapshot.down_ask_vwap_15::double precision AS down_ask_vwap_15,
  snapshot.down_ask_vwap_20::double precision AS down_ask_vwap_20,
  snapshot.down_ask_vwap_25::double precision AS down_ask_vwap_25,
  snapshot.down_ask_vwap_30::double precision AS down_ask_vwap_30,
  snapshot.down_ask_vwap_40::double precision AS down_ask_vwap_40,
  snapshot.down_ask_vwap_50::double precision AS down_ask_vwap_50,
  snapshot.down_ask_vwap_75::double precision AS down_ask_vwap_75,
  snapshot.down_ask_vwap_100::double precision AS down_ask_vwap_100,
  snapshot.down_ask_vwap_125::double precision AS down_ask_vwap_125,
  snapshot.down_ask_vwap_150::double precision AS down_ask_vwap_150,
  snapshot.down_ask_vwap_175::double precision AS down_ask_vwap_175,
  snapshot.down_ask_vwap_200::double precision AS down_ask_vwap_200,
  snapshot.quality_flags
FROM polymarket.btc_market_capacity_execution_snapshots snapshot
JOIN polymarket.backfill_artifacts artifact
  ON artifact.artifact_id = snapshot.artifact_id
 AND artifact.status = 'completed'
 AND artifact.provider = 'pmxt_v2_capacity_execution_snapshots_v2'
JOIN eligible_markets market ON market.market_id = snapshot.market_id
WHERE snapshot.sampled_at >= %(batch_start)s
  AND snapshot.sampled_at < %(batch_end)s
  AND snapshot.schema_version = 'btc5m-capacity-book-1-240s-v2'
ORDER BY snapshot.market_id, snapshot.sampled_at;

