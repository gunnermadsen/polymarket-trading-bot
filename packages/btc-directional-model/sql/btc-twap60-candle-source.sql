SELECT
  candle.open_timestamp,
  candle.close_timestamp,
  candle.close_timestamp AS available_at,
  candle.artifact_id::text AS artifact_id,
  candle.open_price::double precision AS open_price,
  candle.high_price::double precision AS high_price,
  candle.low_price::double precision AS low_price,
  candle.close_price::double precision AS close_price
FROM polymarket.chainlink_btcusd_one_minute_candles candle
WHERE candle.symbol = %(candle_symbol)s
  AND candle.close_timestamp >= %(history_start)s
  AND candle.close_timestamp < %(range_end)s
  AND candle.close_timestamp = candle.open_timestamp + interval '1 minute'
ORDER BY candle.close_timestamp;
