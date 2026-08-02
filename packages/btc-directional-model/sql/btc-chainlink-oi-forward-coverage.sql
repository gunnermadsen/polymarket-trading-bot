WITH labeled_markets AS MATERIALIZED (
  SELECT
    market.market_id,
    market.window_start,
    market.window_end
  FROM polymarket.btc_interval_markets market
  WHERE market.window_start >= %(range_start)s
    AND market.window_start < %(range_end)s
    AND market.validation_status = 'valid'
    AND market.official_outcome IN ('up', 'down')
),
reference_counts AS MATERIALIZED (
  SELECT
    fact.market_id,
    count(*) FILTER (WHERE fact.fact_type = 'opening_boundary') AS opening_count,
    count(*) FILTER (WHERE fact.fact_type = 'final_price') AS final_count
  FROM polymarket.btc_market_reference_facts fact
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = fact.artifact_id
   AND artifact.status = 'completed'
  JOIN labeled_markets market
    ON market.market_id = fact.market_id
  WHERE fact.source_effective_at >= %(range_start)s
    AND fact.source_effective_at <= %(range_end)s
    AND fact.fact_type IN ('opening_boundary', 'final_price')
  GROUP BY fact.market_id
),
kline_counts AS MATERIALIZED (
  SELECT
    market.market_id,
    count(*) AS history_rows,
    count(DISTINCT kline.open_timestamp) AS history_unique_seconds,
    min(
      extract(epoch FROM (kline.open_timestamp + interval '1 second') - market.window_start)
    )::integer AS history_min_second,
    max(
      extract(epoch FROM (kline.open_timestamp + interval '1 second') - market.window_start)
    )::integer AS history_max_second
  FROM labeled_markets market
  JOIN polymarket.binance_one_second_klines kline
    ON kline.symbol = 'BTCUSDT'
   AND kline.open_timestamp >= market.window_start - interval '1 second'
   AND kline.open_timestamp < market.window_start + interval '240 seconds'
   AND kline.close_timestamp < kline.open_timestamp + interval '1 second'
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = kline.artifact_id
   AND artifact.status = 'completed'
  GROUP BY market.market_id
),
book_counts AS MATERIALIZED (
  SELECT
    snapshot.sampled_at::date AS utc_date,
    count(*)::bigint AS snapshot_rows
  FROM polymarket.btc_market_execution_snapshots snapshot
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = snapshot.artifact_id
   AND artifact.status = 'completed'
  WHERE snapshot.sampled_at >= %(range_start)s
    AND snapshot.sampled_at < %(range_end)s
  GROUP BY snapshot.sampled_at::date
)
SELECT
  market.window_start::date AS utc_date,
  count(*)::bigint AS labeled_markets,
  count(*) FILTER (
    WHERE reference.opening_count = 1
  )::bigint AS opening_reference_markets,
  count(*) FILTER (
    WHERE reference.opening_count = 1
      AND reference.final_count <= 1
  )::bigint AS core_fact_markets,
  count(*) FILTER (
    WHERE reference.opening_count = 1
      AND reference.final_count <= 1
      AND kline.history_rows = 241
      AND kline.history_unique_seconds = 241
      AND kline.history_min_second = 0
      AND kline.history_max_second = 240
  )::bigint AS complete_core_history_markets,
  COALESCE(max(book.snapshot_rows), 0)::bigint AS execution_snapshot_rows
FROM labeled_markets market
LEFT JOIN reference_counts reference USING (market_id)
LEFT JOIN kline_counts kline USING (market_id)
LEFT JOIN book_counts book
  ON book.utc_date = market.window_start::date
GROUP BY market.window_start::date
ORDER BY utc_date;
