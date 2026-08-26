SELECT
  candle.open_timestamp,
  candle.close_timestamp,
  candle.close_timestamp AS available_at,
  candle.open_price::double precision AS open_price,
  candle.high_price::double precision AS high_price,
  candle.low_price::double precision AS low_price,
  candle.close_price::double precision AS close_price
FROM polymarket.chainlink_btcusd_one_minute_candles candle
JOIN polymarket.backfill_artifacts artifact
  ON artifact.artifact_id = candle.artifact_id
 AND artifact.status = 'completed'
WHERE candle.symbol = 'BTCUSD'
  AND candle.close_timestamp >= %(range_start)s - interval '61 minutes'
  AND candle.close_timestamp < %(range_end)s
  AND candle.close_timestamp = candle.open_timestamp + interval '1 minute'
ORDER BY candle.close_timestamp;
