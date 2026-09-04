WITH opening_facts AS MATERIALIZED (
  SELECT DISTINCT ON (fact.market_id)
    fact.market_id,
    fact.value::double precision AS opening_boundary
  FROM polymarket.btc_market_reference_facts fact
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = fact.artifact_id
   AND artifact.status = 'completed'
  WHERE fact.fact_type = 'opening_boundary'
    AND fact.source_effective_at >= %(batch_start)s - interval '5 minutes'
    AND fact.source_effective_at < %(batch_end)s + interval '5 minutes'
  ORDER BY fact.market_id, fact.source_effective_at DESC, fact.created_at DESC
),
eligible_markets AS MATERIALIZED (
  SELECT
    market.market_id,
    market.window_start,
    market.window_end,
    market.official_outcome,
    CASE WHEN market.official_outcome = 'up' THEN 1 ELSE 0 END AS label_up,
    COALESCE(fact.opening_boundary, market.reference_price::double precision) AS opening_boundary,
    market.resolution_price::double precision AS final_price
  FROM polymarket.btc_interval_markets market
  LEFT JOIN opening_facts fact USING (market_id)
  WHERE market.window_start >= %(batch_start)s
    AND market.window_start < %(batch_end)s
    AND market.validation_status = 'valid'
    AND market.official_outcome IN ('up', 'down')
    AND COALESCE(fact.opening_boundary, market.reference_price::double precision) > 0
),
legacy_klines AS MATERIALIZED (
  SELECT
    kline.open_timestamp,
    kline.close_timestamp,
    kline.open_price,
    kline.high_price,
    kline.low_price,
    kline.close_price,
    kline.base_volume,
    kline.quote_volume,
    kline.trade_count,
    kline.taker_buy_base_volume,
    kline.taker_buy_quote_volume,
    1 AS source_priority
  FROM market_data.binance_spot_btcusdt_one_second_ohlcv kline
  JOIN ingester.capture_artifacts artifact
    ON artifact.artifact_id = kline.capture_artifact_id
   AND artifact.status = 'completed'
  WHERE kline.symbol = 'BTCUSDT'
    AND kline.open_timestamp >= %(batch_start)s - interval '1 second'
    AND kline.open_timestamp < %(batch_end)s
),
canonical_klines AS MATERIALIZED (
  SELECT
    kline.open_timestamp,
    kline.close_timestamp,
    kline.open_price,
    kline.high_price,
    kline.low_price,
    kline.close_price,
    kline.base_volume,
    kline.quote_volume,
    kline.trade_count,
    kline.taker_buy_base_volume,
    kline.taker_buy_quote_volume,
    2 AS source_priority
  FROM market_data.binance_spot_btcusdt_one_second_ohlcv kline
  WHERE kline.symbol = 'BTCUSDT'
    AND kline.open_timestamp >= %(batch_start)s - interval '1 second'
    AND kline.open_timestamp < %(batch_end)s
    AND kline.received_at IS NOT NULL
    AND kline.close_timestamp <= kline.received_at
),
klines AS MATERIALIZED (
  SELECT DISTINCT ON (open_timestamp)
    open_timestamp,
    close_timestamp,
    open_price,
    high_price,
    low_price,
    close_price,
    base_volume,
    quote_volume,
    trade_count,
    taker_buy_base_volume,
    taker_buy_quote_volume
  FROM (
    SELECT * FROM legacy_klines
    UNION ALL
    SELECT * FROM canonical_klines
  ) source
  ORDER BY open_timestamp, source_priority DESC
)
SELECT
  market.market_id,
  market.window_start,
  market.window_end,
  market.official_outcome,
  market.label_up,
  market.opening_boundary,
  market.final_price,
  kline.open_timestamp + interval '1 second' AS observed_at,
  extract(epoch FROM (kline.open_timestamp + interval '1 second') - market.window_start)::integer
    AS seconds_elapsed,
  kline.open_price::double precision AS btc_open,
  kline.high_price::double precision AS btc_high,
  kline.low_price::double precision AS btc_low,
  kline.close_price::double precision AS btc_close,
  kline.base_volume::double precision AS btc_base_volume,
  kline.quote_volume::double precision AS btc_quote_volume,
  kline.trade_count,
  kline.taker_buy_base_volume::double precision AS btc_taker_buy_base_volume,
  kline.taker_buy_quote_volume::double precision AS btc_taker_buy_quote_volume
FROM eligible_markets market
JOIN klines kline
  ON kline.open_timestamp >= market.window_start - interval '1 second'
 AND kline.open_timestamp < market.window_end - interval '1 second'
 AND kline.close_timestamp < kline.open_timestamp + interval '1 second'
ORDER BY market.window_start, market.market_id, kline.open_timestamp;
