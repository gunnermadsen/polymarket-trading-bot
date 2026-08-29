WITH eligible_markets AS MATERIALIZED (
  SELECT market_id, window_start, window_end
  FROM polymarket.btc_interval_markets
  WHERE window_start >= %(batch_start)s
    AND window_start < %(batch_end)s
    AND validation_status = 'valid'
    AND official_outcome IN ('up', 'down')
)
SELECT
  market.market_id,
  market.window_start,
  market.window_end,
  kline.open_timestamp,
  kline.open_timestamp + interval '1 second' AS available_at,
  kline.close_price::double precision AS close_price,
  kline.artifact_id::text AS artifact_id
FROM eligible_markets market
JOIN polymarket.binance_one_second_klines kline
  ON kline.symbol = 'BTCUSDT'
 AND kline.open_timestamp >= market.window_start - interval '60 seconds'
 AND kline.open_timestamp < market.window_end
 AND kline.open_timestamp >= %(batch_start)s - interval '60 seconds'
 AND kline.open_timestamp < %(batch_end)s + interval '5 minutes'
 AND kline.close_timestamp < kline.open_timestamp + interval '1 second'
JOIN polymarket.backfill_artifacts artifact
  ON artifact.artifact_id = kline.artifact_id
 AND artifact.status = 'completed'
ORDER BY market.window_start, market.market_id, kline.open_timestamp;
