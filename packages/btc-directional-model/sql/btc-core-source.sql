WITH market_facts AS MATERIALIZED (
  SELECT
    fact.market_id,
    max(fact.value) FILTER (WHERE fact.fact_type = 'opening_boundary') AS opening_boundary,
    max(fact.value) FILTER (WHERE fact.fact_type = 'final_price') AS final_price,
    count(*) FILTER (WHERE fact.fact_type = 'opening_boundary') AS opening_fact_count,
    count(*) FILTER (WHERE fact.fact_type = 'final_price') AS final_fact_count
  FROM polymarket.btc_market_reference_facts fact
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = fact.artifact_id
   AND artifact.status = 'completed'
  WHERE fact.source_effective_at >= %(batch_start)s
    AND fact.source_effective_at <= %(batch_end)s
    AND fact.fact_type IN ('opening_boundary', 'final_price')
  GROUP BY fact.market_id
),
eligible_markets AS MATERIALIZED (
  SELECT
    market.market_id,
    market.window_start,
    market.window_end,
    market.official_outcome,
    CASE WHEN market.official_outcome = 'up' THEN 1 ELSE 0 END AS label_up,
    facts.opening_boundary::double precision AS opening_boundary,
    facts.final_price::double precision AS final_price
  FROM polymarket.btc_interval_markets market
  JOIN market_facts facts USING (market_id)
  WHERE market.window_start >= %(batch_start)s
    AND market.window_start < %(batch_end)s
    AND market.validation_status = 'valid'
    AND market.official_outcome IN ('up', 'down')
    AND facts.opening_fact_count = 1
    AND facts.final_fact_count <= 1
    AND (%(strict_final_price_audit)s = false OR facts.final_fact_count = 1)
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
  extract(
    epoch FROM (kline.open_timestamp + interval '1 second') - market.window_start
  )::integer AS seconds_elapsed,
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
JOIN market_data.binance_spot_btcusdt_one_second_ohlcv kline
  ON kline.symbol = 'BTCUSDT'
 AND kline.open_timestamp >= %(batch_start)s - interval '1 second'
 AND kline.open_timestamp < %(batch_end)s - interval '1 second'
 AND kline.open_timestamp >= market.window_start - interval '1 second'
 AND kline.open_timestamp < market.window_end - interval '1 second'
 AND kline.close_timestamp < kline.open_timestamp + interval '1 second'
JOIN polymarket.backfill_artifacts kline_artifact
  ON kline_artifact.artifact_id = kline.capture_artifact_id
 AND kline_artifact.status = 'completed'
ORDER BY market.window_start, kline.open_timestamp;
