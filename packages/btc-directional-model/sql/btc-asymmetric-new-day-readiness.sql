WITH days AS (
  SELECT generate_series(
    %(range_start)s::timestamptz,
    %(range_end)s::timestamptz - interval '1 day',
    interval '1 day'
  ) AS day_start
),
eligible_markets AS MATERIALIZED (
  SELECT
    market.market_id,
    market.window_start,
    market.official_outcome,
    market.official_resolution_received_at,
    date_trunc('day', market.window_start, 'UTC') AS day_start
  FROM polymarket.btc_interval_markets market
  WHERE market.window_start >= %(range_start)s
    AND market.window_start < %(range_end)s
    AND market.validation_status = 'valid'
    AND market.official_outcome IN ('up', 'down')
),
labels AS (
  SELECT
    market.day_start,
    count(*)::bigint AS labeled_markets,
    count(*) FILTER (WHERE market.official_outcome = 'up')::bigint
      AS up_markets,
    count(*) FILTER (WHERE market.official_outcome = 'down')::bigint
      AS down_markets,
    md5(
      string_agg(
        market.market_id || ':' || market.official_outcome || ':' ||
          COALESCE(market.official_resolution_received_at::text, ''),
        ',' ORDER BY market.market_id
      )
    ) AS label_identity_md5
  FROM eligible_markets market
  GROUP BY 1
),
reference_facts AS (
  SELECT
    market.day_start,
    count(DISTINCT fact.market_id) FILTER (
      WHERE fact.fact_type = 'opening_boundary'
    )::bigint AS opening_boundary_markets,
    count(DISTINCT fact.market_id) FILTER (
      WHERE fact.fact_type = 'final_price'
    )::bigint AS final_price_markets,
    count(*) FILTER (
      WHERE fact.source_effective_at > fact.fetched_at
    )::bigint AS causality_violations,
    md5(
      string_agg(
        fact.market_id || ':' || fact.fact_type || ':' ||
          fact.payload_sha256,
        ',' ORDER BY fact.market_id, fact.fact_type, fact.provider
      )
    ) AS source_identity_md5
  FROM eligible_markets market
  JOIN polymarket.btc_market_reference_facts fact
    ON fact.market_id = market.market_id
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = fact.artifact_id
   AND artifact.status = 'completed'
  GROUP BY 1
),
binance AS (
  SELECT
    date_trunc('day', kline.open_timestamp, 'UTC') AS day_start,
    count(*)::bigint AS rows,
    count(DISTINCT kline.open_timestamp)::bigint AS qualified_seconds,
    count(*) FILTER (
      WHERE kline.close_timestamp >= kline.open_timestamp + interval '1 second'
    )::bigint AS causality_violations
  FROM market_data.binance_spot_btcusdt_one_second_ohlcv kline
  JOIN ingester.capture_artifacts artifact
   ON artifact.artifact_id = kline.capture_artifact_id
   AND artifact.status = 'completed'
   AND artifact.strategy_key = 'binance_spot_btcusdt_one_second_ohlcv'
  WHERE kline.symbol = 'BTCUSDT'
    AND kline.open_timestamp >= %(range_start)s
    AND kline.open_timestamp < %(range_end)s
  GROUP BY 1
),
small_artifact_lineage AS MATERIALIZED (
  SELECT
    artifact.source_date::timestamp AT TIME ZONE 'UTC' AS day_start,
    artifact.ingester_key,
    array_agg(DISTINCT artifact.provider ORDER BY artifact.provider)
      AS providers,
    md5(
      string_agg(
        artifact.artifact_id::text || ':' ||
          COALESCE(artifact.actual_checksum, ''),
        ',' ORDER BY artifact.artifact_id
      )
    ) AS source_identity_md5
  FROM polymarket.backfill_artifacts artifact
  WHERE artifact.status = 'completed'
    AND artifact.ingester_key IN (
      'binance_btcusdt_one_second_klines',
      'polygon_chainlink_btcusd_oracle_rounds',
      'chainlink_btcusd_one_minute_candles'
    )
    AND artifact.source_date >= %(range_start)s::date
    AND artifact.source_date < %(range_end)s::date
  GROUP BY 1, 2
),
pmxt_artifacts AS (
  SELECT
    artifact.source_date::timestamp AT TIME ZONE 'UTC' AS day_start,
    count(DISTINCT artifact.logical_key)::bigint AS completed_hours,
    COALESCE(sum(artifact.record_count), 0)::bigint AS artifact_rows,
    array_agg(DISTINCT artifact.provider ORDER BY artifact.provider)
      AS providers,
    array_agg(
      DISTINCT artifact.metadata ->> 'schema_version'
      ORDER BY artifact.metadata ->> 'schema_version'
    ) AS schema_versions,
    md5(
      string_agg(
        artifact.logical_key || ':' || COALESCE(artifact.actual_checksum, ''),
        ',' ORDER BY artifact.logical_key
      )
    ) AS source_identity_md5
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
pmxt_grid AS (
  SELECT
    market.day_start,
    count(*)::bigint AS exact_grid_rows,
    count(DISTINCT (snapshot.market_id, snapshot.sampled_at))::bigint
      AS exact_grid_keys,
    count(*) FILTER (
      WHERE snapshot.up_provider_received_at IS NOT NULL
        AND snapshot.down_provider_received_at IS NOT NULL
        AND snapshot.up_provider_received_at <= snapshot.sampled_at
        AND snapshot.down_provider_received_at <= snapshot.sampled_at
        AND snapshot.up_provider_received_at >=
          snapshot.sampled_at - interval '2 seconds'
        AND snapshot.down_provider_received_at >=
          snapshot.sampled_at - interval '2 seconds'
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
        AND (snapshot.quality_flags & 63) = 0
    )::bigint AS strict_grid_rows,
    count(*) FILTER (
      WHERE snapshot.up_provider_received_at > snapshot.sampled_at
         OR snapshot.down_provider_received_at > snapshot.sampled_at
    )::bigint AS causality_violations
  FROM eligible_markets market
  JOIN polymarket.btc_market_execution_snapshots snapshot
    ON snapshot.market_id = market.market_id
   AND snapshot.sampled_at >= market.window_start + interval '1 second'
   AND snapshot.sampled_at <= market.window_start + interval '240 seconds'
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = snapshot.artifact_id
   AND artifact.ingester_key =
     'polymarket_btc_five_minute_execution_snapshots'
   AND artifact.provider = 'pmxt_v2_execution_snapshots'
   AND artifact.status = 'completed'
   AND artifact.metadata ->> 'schema_version' = 'btc5m-book-250ms-v1'
  WHERE snapshot.schema_version = 'btc5m-book-250ms-v1'
    AND snapshot.sampled_at >= %(range_start)s
    AND snapshot.sampled_at < %(range_end)s
    AND mod(
      round(
        extract(epoch FROM snapshot.sampled_at - market.window_start) * 1000
      )::bigint,
      1000
    ) = 0
    AND (
      extract(epoch FROM snapshot.sampled_at - market.window_start)::integer
        BETWEEN 1 AND 59
      OR (
        extract(epoch FROM snapshot.sampled_at - market.window_start)::integer
          BETWEEN 60 AND 240
        AND mod(
          extract(
            epoch FROM snapshot.sampled_at - market.window_start
          )::integer,
          5
        ) = 0
      )
    )
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
l2_provider_rows AS (
  SELECT
    date_trunc('day', feature.second_start, 'UTC') AS day_start,
    count(*) FILTER (
      WHERE artifact.metadata ->> 'materialization_contract' =
        'cryptohft-binance-spot-btcusdt-l2-features-v1'
    )::bigint AS cryptohft_rows,
    count(*) FILTER (
      WHERE artifact.metadata ->> 'materialization_contract' =
        'coinapi-binance-spot-btcusdt-l2-snapshots-v1'
    )::bigint AS coinapi_rows,
    count(*) FILTER (
      WHERE artifact.metadata ->> 'materialization_contract' =
        'huggingface-goooddy-binance-spot-btcusdt-l2-features-v1'
    )::bigint AS huggingface_rows
  FROM polymarket.binance_spot_btcusdt_l2_one_second_features feature
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = feature.artifact_id
   AND artifact.ingester_key =
     'binance_spot_btcusdt_l2_one_second_features'
   AND artifact.status = 'completed'
  WHERE feature.symbol = 'BTCUSDT'
    AND feature.second_start >= %(range_start)s
    AND feature.second_start < %(range_end)s
  GROUP BY 1
),
l2_lineage AS (
  SELECT
    artifact.source_date::timestamp AT TIME ZONE 'UTC' AS day_start,
    array_agg(DISTINCT artifact.provider ORDER BY artifact.provider)
      AS providers,
    array_agg(
      DISTINCT artifact.metadata ->> 'materialization_contract'
      ORDER BY artifact.metadata ->> 'materialization_contract'
    ) AS materialization_contracts,
    md5(
      string_agg(DISTINCT
        artifact.artifact_id::text || ':' ||
          COALESCE(artifact.actual_checksum, ''),
        ',' ORDER BY artifact.artifact_id::text || ':' ||
          COALESCE(artifact.actual_checksum, '')
      )
    ) AS source_identity_md5
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
oracle AS (
  SELECT
    date_trunc('day', round.source_timestamp, 'UTC') AS day_start,
    count(*)::bigint AS rows,
    count(*) FILTER (
      WHERE round.source_timestamp > round.block_timestamp
    )::bigint AS causality_violations
  FROM market_data.polygon_chainlink_btcusd_oracle_rounds round
  JOIN ingester.capture_artifacts artifact
   ON artifact.artifact_id = round.capture_artifact_id
   AND artifact.status = 'completed'
   AND artifact.strategy_key = 'polygon_chainlink_btcusd_oracle'
  WHERE round.feed_proxy_address = %(oracle_feed_proxy_address)s
    AND round.source_timestamp >= %(range_start)s
    AND round.source_timestamp < %(range_end)s
  GROUP BY 1
),
candles AS (
  SELECT
    date_trunc('day', candle.open_timestamp, 'UTC') AS day_start,
    count(*)::bigint AS rows,
    count(*) FILTER (
      WHERE candle.close_timestamp <>
        candle.open_timestamp + interval '1 minute'
    )::bigint AS causality_violations
  FROM market_data.chainlink_btcusd_one_minute_candles candle
  JOIN ingester.capture_artifacts artifact
   ON artifact.artifact_id = candle.capture_artifact_id
   AND artifact.status = 'completed'
   AND artifact.strategy_key = 'chainlink_btcusd_one_minute_ohlc'
  WHERE candle.symbol = 'BTCUSD'
    AND candle.open_timestamp >= %(range_start)s
    AND candle.open_timestamp < %(range_end)s
  GROUP BY 1
)
SELECT
  days.day_start::date AS date,
  COALESCE(labels.labeled_markets, 0)::bigint AS labeled_markets,
  COALESCE(labels.up_markets, 0)::bigint AS up_markets,
  COALESCE(labels.down_markets, 0)::bigint AS down_markets,
  COALESCE(labels.label_identity_md5, '') AS label_identity_md5,
  COALESCE(reference_facts.opening_boundary_markets, 0)::bigint
    AS opening_boundary_markets,
  COALESCE(reference_facts.final_price_markets, 0)::bigint
    AS final_price_markets,
  COALESCE(reference_facts.causality_violations, 0)::bigint
    AS reference_causality_violations,
  COALESCE(reference_facts.source_identity_md5, '')
    AS reference_source_identity_md5,
  COALESCE(binance.rows, 0)::bigint AS binance_one_second_rows,
  COALESCE(binance.qualified_seconds, 0)::bigint
    AS binance_qualified_seconds,
  COALESCE(binance.causality_violations, 0)::bigint
    AS binance_causality_violations,
  COALESCE(binance_artifact.providers, ARRAY[]::text[]) AS binance_providers,
  COALESCE(binance_artifact.source_identity_md5, '')
    AS binance_source_identity_md5,
  COALESCE(pmxt_artifacts.completed_hours, 0)::bigint
    AS pmxt_completed_hours,
  COALESCE(pmxt_artifacts.artifact_rows, 0)::bigint AS pmxt_artifact_rows,
  COALESCE(pmxt_artifacts.providers, ARRAY[]::text[]) AS pmxt_providers,
  COALESCE(pmxt_artifacts.schema_versions, ARRAY[]::text[])
    AS pmxt_schema_versions,
  COALESCE(pmxt_artifacts.source_identity_md5, '')
    AS pmxt_source_identity_md5,
  COALESCE(pmxt_grid.exact_grid_rows, 0)::bigint AS pmxt_exact_grid_rows,
  COALESCE(pmxt_grid.exact_grid_keys, 0)::bigint AS pmxt_exact_grid_keys,
  COALESCE(pmxt_grid.strict_grid_rows, 0)::bigint AS pmxt_strict_grid_rows,
  COALESCE(pmxt_grid.causality_violations, 0)::bigint
    AS pmxt_causality_violations,
  COALESCE(l2.rows, 0)::bigint AS l2_rows,
  COALESCE(l2.qualified_seconds, 0)::bigint AS l2_qualified_seconds,
  COALESCE(l2.causality_violations, 0)::bigint AS l2_causality_violations,
  COALESCE(l2_provider_rows.cryptohft_rows, 0)::bigint
    AS l2_cryptohft_rows,
  COALESCE(l2_provider_rows.coinapi_rows, 0)::bigint AS l2_coinapi_rows,
  COALESCE(l2_provider_rows.huggingface_rows, 0)::bigint
    AS l2_huggingface_rows,
  COALESCE(l2_lineage.providers, ARRAY[]::text[]) AS l2_providers,
  COALESCE(
    l2_lineage.materialization_contracts,
    ARRAY[]::text[]
  ) AS l2_materialization_contracts,
  COALESCE(l2_lineage.source_identity_md5, '') AS l2_source_identity_md5,
  COALESCE(oracle.rows, 0)::bigint AS oracle_rounds,
  COALESCE(oracle.causality_violations, 0)::bigint
    AS oracle_causality_violations,
  COALESCE(oracle_artifact.providers, ARRAY[]::text[]) AS oracle_providers,
  COALESCE(oracle_artifact.source_identity_md5, '')
    AS oracle_source_identity_md5,
  COALESCE(candles.rows, 0)::bigint AS chainlink_candle_rows,
  COALESCE(candles.causality_violations, 0)::bigint
    AS chainlink_candle_causality_violations,
  COALESCE(candle_artifact.providers, ARRAY[]::text[])
    AS chainlink_candle_providers,
  COALESCE(candle_artifact.source_identity_md5, '')
    AS chainlink_candle_source_identity_md5
FROM days
LEFT JOIN labels USING (day_start)
LEFT JOIN reference_facts USING (day_start)
LEFT JOIN binance USING (day_start)
LEFT JOIN pmxt_artifacts USING (day_start)
LEFT JOIN pmxt_grid USING (day_start)
LEFT JOIN l2 USING (day_start)
LEFT JOIN l2_provider_rows USING (day_start)
LEFT JOIN l2_lineage USING (day_start)
LEFT JOIN oracle USING (day_start)
LEFT JOIN candles USING (day_start)
LEFT JOIN small_artifact_lineage binance_artifact
  ON binance_artifact.day_start = days.day_start
 AND binance_artifact.strategy_key = 'binance_spot_btcusdt_one_second_ohlcv'
LEFT JOIN small_artifact_lineage oracle_artifact
  ON oracle_artifact.day_start = days.day_start
 AND oracle_artifact.strategy_key = 'polygon_chainlink_btcusd_oracle'
LEFT JOIN small_artifact_lineage candle_artifact
  ON candle_artifact.day_start = days.day_start
 AND candle_artifact.strategy_key = 'chainlink_btcusd_one_minute_ohlc'
ORDER BY days.day_start;
