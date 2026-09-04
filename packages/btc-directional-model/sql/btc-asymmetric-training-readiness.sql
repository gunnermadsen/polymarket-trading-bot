WITH days AS (
  SELECT generate_series(
    %(range_start)s::timestamptz,
    %(range_end)s::timestamptz - interval '1 day',
    interval '1 day'
  ) AS day_start
),
labels AS (
  SELECT
    date_trunc('day', market.window_start, 'UTC') AS day_start,
    count(*) FILTER (
      WHERE market.validation_status = 'valid'
        AND market.official_outcome IN ('up', 'down')
    )::bigint AS labeled_markets
  FROM polymarket.btc_interval_markets market
  WHERE market.window_start >= %(range_start)s
    AND market.window_start < %(range_end)s
    AND market.validation_status = 'valid'
    AND market.official_outcome IN ('up', 'down')
  GROUP BY 1
),
reference_facts AS (
  SELECT
    date_trunc('day', market.window_start, 'UTC') AS day_start,
    count(DISTINCT fact.market_id) FILTER (
      WHERE fact.fact_type = 'opening_boundary'
    )::bigint AS opening_boundary_markets,
    count(DISTINCT fact.market_id) FILTER (
      WHERE fact.fact_type = 'final_price'
    )::bigint AS final_price_markets
  FROM polymarket.btc_interval_markets market
  JOIN polymarket.btc_market_reference_facts fact
    ON fact.market_id = market.market_id
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = fact.artifact_id
   AND artifact.status = 'completed'
  WHERE market.window_start >= %(range_start)s
    AND market.window_start < %(range_end)s
    AND market.validation_status = 'valid'
    AND market.official_outcome IN ('up', 'down')
  GROUP BY 1
),
binance AS (
  SELECT
    date_trunc('day', kline.open_timestamp, 'UTC') AS day_start,
    count(*)::bigint AS rows,
    count(*) FILTER (
      WHERE kline.close_timestamp >= kline.open_timestamp + interval '1 second'
    )::bigint AS causality_violations,
    array_agg(DISTINCT artifact.provider ORDER BY artifact.provider) AS providers
  FROM market_data.binance_spot_btcusdt_one_second_ohlcv kline
  JOIN ingester.capture_artifacts artifact
    ON artifact.artifact_id = kline.capture_artifact_id
   AND artifact.status = 'completed'
  WHERE kline.symbol = 'BTCUSDT'
    AND kline.open_timestamp >= %(range_start)s
    AND kline.open_timestamp < %(range_end)s
  GROUP BY 1
),
oracle AS (
  SELECT
    date_trunc('day', round.block_timestamp, 'UTC') AS day_start,
    count(*)::bigint AS rows,
    count(*) FILTER (
      WHERE round.source_timestamp > round.block_timestamp
    )::bigint AS causality_violations,
    array_agg(DISTINCT artifact.provider ORDER BY artifact.provider) AS providers
  FROM market_data.polygon_chainlink_btcusd_oracle_rounds round
  JOIN ingester.capture_artifacts artifact
    ON artifact.artifact_id = round.capture_artifact_id
   AND artifact.status = 'completed'
  WHERE round.feed_proxy_address = %(oracle_feed_proxy_address)s
    AND round.block_timestamp >= %(range_start)s
    AND round.block_timestamp < %(range_end)s
  GROUP BY 1
),
pmxt AS (
  SELECT
    artifact.source_date::timestamp AT TIME ZONE 'UTC' AS day_start,
    count(DISTINCT artifact.logical_key)::bigint AS completed_hours,
    COALESCE(sum(artifact.record_count), 0)::bigint AS artifact_rows,
    array_agg(DISTINCT artifact.provider ORDER BY artifact.provider) AS providers,
    array_agg(
      DISTINCT artifact.metadata ->> 'schema_version'
      ORDER BY artifact.metadata ->> 'schema_version'
    ) AS schema_versions
  FROM polymarket.backfill_artifacts artifact
  WHERE artifact.ingester_key =
      'polymarket_btc_five_minute_execution_snapshots'
    AND artifact.provider = 'pmxt_v2_execution_snapshots'
    AND artifact.status = 'completed'
    AND artifact.metadata ->> 'schema_version' = 'btc5m-book-250ms-v1'
    AND artifact.source_date >= %(range_start)s::date
    AND artifact.source_date < %(range_end)s::date
  GROUP BY 1
),
l2 AS (
  SELECT
    date_trunc('day', feature.second_start, 'UTC') AS day_start,
    count(*)::bigint AS rows,
    count(DISTINCT feature.second_start)::bigint AS qualified_seconds,
    count(*) FILTER (
      WHERE feature.source_event_timestamp > feature.available_at
         OR feature.provider_received_at > feature.available_at
         OR feature.second_start > feature.available_at
         OR feature.available_at >= feature.second_start + interval '1 second'
    )::bigint AS causality_violations
  FROM polymarket.binance_spot_btcusdt_l2_training_features feature
  WHERE feature.symbol = 'BTCUSDT'
    AND feature.second_start >= %(range_start)s
    AND feature.second_start < %(range_end)s
  GROUP BY 1
),
l2_lineage AS (
  SELECT
    artifact.source_date::timestamp AT TIME ZONE 'UTC' AS day_start,
    array_agg(DISTINCT artifact.provider ORDER BY artifact.provider) AS providers,
    array_agg(
      DISTINCT artifact.metadata ->> 'materialization_contract'
      ORDER BY artifact.metadata ->> 'materialization_contract'
    ) AS materialization_contracts
  FROM polymarket.backfill_artifacts artifact
  WHERE artifact.ingester_key =
      'binance_spot_btcusdt_l2_one_second_features'
    AND artifact.status = 'completed'
    AND artifact.metadata ->> 'materialization_contract' IN (
      'cryptohft-binance-spot-btcusdt-l2-features-v1',
      'coinapi-binance-spot-btcusdt-l2-snapshots-v1',
      'huggingface-goooddy-binance-spot-btcusdt-l2-features-v1'
    )
    AND artifact.source_date >= %(range_start)s::date
    AND artifact.source_date < %(range_end)s::date
  GROUP BY 1
),
candles AS (
  SELECT
    date_trunc('day', candle.open_timestamp, 'UTC') AS day_start,
    count(*)::bigint AS rows,
    count(*) FILTER (
      WHERE candle.close_timestamp <> candle.open_timestamp + interval '1 minute'
    )::bigint AS causality_violations,
    array_agg(DISTINCT artifact.provider ORDER BY artifact.provider) AS providers
  FROM market_data.chainlink_btcusd_one_minute_candles candle
  JOIN ingester.capture_artifacts artifact
    ON artifact.artifact_id = candle.capture_artifact_id
   AND artifact.status = 'completed'
  WHERE candle.symbol = 'BTCUSD'
    AND candle.open_timestamp >= %(range_start)s
    AND candle.open_timestamp < %(range_end)s
  GROUP BY 1
)
SELECT
  days.day_start::date AS date,
  COALESCE(labels.labeled_markets, 0)::bigint AS labeled_markets,
  COALESCE(reference_facts.opening_boundary_markets, 0)::bigint
    AS opening_boundary_markets,
  COALESCE(reference_facts.final_price_markets, 0)::bigint
    AS final_price_markets,
  COALESCE(binance.rows, 0)::bigint AS binance_one_second_rows,
  COALESCE(binance.causality_violations, 0)::bigint
    AS binance_causality_violations,
  COALESCE(binance.providers, ARRAY[]::text[]) AS binance_providers,
  COALESCE(oracle.rows, 0)::bigint AS oracle_rounds,
  COALESCE(oracle.causality_violations, 0)::bigint
    AS oracle_causality_violations,
  COALESCE(oracle.providers, ARRAY[]::text[]) AS oracle_providers,
  COALESCE(pmxt.completed_hours, 0)::bigint AS pmxt_completed_hours,
  COALESCE(pmxt.artifact_rows, 0)::bigint AS pmxt_artifact_rows,
  COALESCE(pmxt.providers, ARRAY[]::text[]) AS pmxt_providers,
  COALESCE(pmxt.schema_versions, ARRAY[]::text[]) AS pmxt_schema_versions,
  COALESCE(l2.rows, 0)::bigint AS l2_rows,
  COALESCE(l2.qualified_seconds, 0)::bigint AS l2_qualified_seconds,
  COALESCE(l2.causality_violations, 0)::bigint AS l2_causality_violations,
  COALESCE(l2_lineage.providers, ARRAY[]::text[]) AS l2_providers,
  COALESCE(
    l2_lineage.materialization_contracts,
    ARRAY[]::text[]
  ) AS l2_materialization_contracts,
  COALESCE(candles.rows, 0)::bigint AS chainlink_candle_rows,
  COALESCE(candles.causality_violations, 0)::bigint
    AS chainlink_candle_causality_violations,
  COALESCE(candles.providers, ARRAY[]::text[]) AS chainlink_candle_providers
FROM days
LEFT JOIN labels USING (day_start)
LEFT JOIN reference_facts USING (day_start)
LEFT JOIN binance USING (day_start)
LEFT JOIN oracle USING (day_start)
LEFT JOIN pmxt USING (day_start)
LEFT JOIN l2 USING (day_start)
LEFT JOIN l2_lineage USING (day_start)
LEFT JOIN candles USING (day_start)
ORDER BY days.day_start;
