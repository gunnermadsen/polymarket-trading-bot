WITH eligible_markets AS MATERIALIZED (
  SELECT
    market.market_id,
    market.window_start,
    market.window_end,
    market.official_outcome,
    CASE WHEN market.official_outcome = 'up' THEN 1 ELSE 0 END AS label_up
  FROM polymarket.btc_interval_markets market
  WHERE market.window_start >= %(batch_start)s
    AND market.window_start < %(batch_end)s
    AND market.window_start >= %(current_start)s
    AND market.validation_status = 'valid'
    AND market.official_outcome IN ('up', 'down')
)
SELECT
  market.market_id,
  market.window_start,
  market.window_end,
  market.official_outcome,
  market.label_up,
  1.0::double precision AS opening_boundary,
  NULL::double precision AS final_price,
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
JOIN polymarket.binance_one_second_klines kline
  ON kline.symbol = 'BTCUSDT'
 AND kline.open_timestamp >= market.window_start - interval '1 second'
 AND kline.open_timestamp < market.window_end - interval '1 second'
 AND kline.open_timestamp >= %(batch_start)s - interval '1 second'
 AND kline.open_timestamp < %(batch_end)s - interval '1 second'
 AND kline.close_timestamp < kline.open_timestamp + interval '1 second'
JOIN polymarket.backfill_artifacts artifact
  ON artifact.artifact_id = kline.artifact_id
 AND artifact.status = 'completed'
ORDER BY market.window_start, kline.open_timestamp;
