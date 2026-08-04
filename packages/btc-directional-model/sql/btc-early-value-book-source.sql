WITH market_points AS (
  SELECT
    market.market_id,
    market.window_start,
    market.window_end,
    CASE WHEN market.official_outcome = 'Up' THEN 1 ELSE 0 END AS label_up,
    market.fee_rate::double precision AS fee_rate,
    offset_seconds::integer AS seconds_elapsed,
    market.window_start + make_interval(secs => offset_seconds) AS observed_at,
    market.up_token_id,
    market.down_token_id
  FROM polymarket.btc_interval_markets market
  CROSS JOIN LATERAL (
    SELECT value AS offset_seconds FROM generate_series(1, 59) value
    UNION ALL
    SELECT value AS offset_seconds FROM generate_series(60, 240, 5) value
  ) offsets
  WHERE market.window_start >= %(batch_start)s
    AND market.window_start < %(batch_end)s
    AND market.official_outcome IN ('Up', 'Down')
), token_points AS (
  SELECT
    point.*,
    side.side,
    side.token_id
  FROM market_points point
  CROSS JOIN LATERAL (
    VALUES ('YES'::text, point.up_token_id), ('NO'::text, point.down_token_id)
  ) side(side, token_id)
), causal_books AS (
  SELECT
    point.*,
    book.received_at,
    book.snapshot_at,
    book.best_ask::double precision AS best_ask,
    book.book
  FROM token_points point
  LEFT JOIN LATERAL (
    SELECT checkpoint.received_at, checkpoint.snapshot_at,
           checkpoint.best_ask, checkpoint.book
    FROM polymarket.orderbook_checkpoints checkpoint
    WHERE checkpoint.market_id = point.market_id
      AND checkpoint.token_id = point.token_id
      AND checkpoint.received_at <= point.observed_at
      AND checkpoint.received_at > point.observed_at - make_interval(secs => %(freshness_seconds)s)
      AND checkpoint.integrity_status = 'ok'
    ORDER BY checkpoint.received_at DESC
    LIMIT 1
  ) book ON TRUE
), priced AS (
  SELECT
    causal.*,
    ladder.ask_vwap_5,
    ladder.ask_depth
  FROM causal_books causal
  LEFT JOIN LATERAL (
    SELECT
      CASE WHEN SUM(size) >= %(quantity)s THEN
        SUM(price * LEAST(size, GREATEST(%(quantity)s - prior_size, 0.0)))
        / %(quantity)s
      END::double precision AS ask_vwap_5,
      SUM(size)::double precision AS ask_depth
    FROM (
      SELECT
        (level->>'price')::double precision AS price,
        (level->>'size')::double precision AS size,
        COALESCE(SUM((level->>'size')::double precision) OVER (
          ORDER BY (level->>'price')::double precision
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0.0) AS prior_size
      FROM jsonb_array_elements(COALESCE(causal.book->'asks', '[]'::jsonb)) level
    ) asks
    WHERE prior_size < %(quantity)s
  ) ladder ON TRUE
)
SELECT
  market_id,
  window_start,
  window_end,
  label_up,
  fee_rate,
  seconds_elapsed,
  observed_at,
  MAX(received_at) FILTER (WHERE side = 'YES') AS yes_received_at,
  MAX(snapshot_at) FILTER (WHERE side = 'YES') AS yes_snapshot_at,
  MAX(best_ask) FILTER (WHERE side = 'YES') AS yes_best_ask,
  MAX(ask_vwap_5) FILTER (WHERE side = 'YES') AS yes_ask_vwap_5,
  MAX(ask_depth) FILTER (WHERE side = 'YES') AS yes_ask_depth,
  MAX(received_at) FILTER (WHERE side = 'NO') AS no_received_at,
  MAX(snapshot_at) FILTER (WHERE side = 'NO') AS no_snapshot_at,
  MAX(best_ask) FILTER (WHERE side = 'NO') AS no_best_ask,
  MAX(ask_vwap_5) FILTER (WHERE side = 'NO') AS no_ask_vwap_5,
  MAX(ask_depth) FILTER (WHERE side = 'NO') AS no_ask_depth
FROM priced
GROUP BY market_id, window_start, window_end, label_up, fee_rate,
         seconds_elapsed, observed_at
ORDER BY window_start, seconds_elapsed;
