WITH legacy AS (
  SELECT
    candle.open_timestamp,
    candle.close_timestamp,
    candle.close_timestamp AS available_at,
    candle.open_price,
    candle.high_price,
    candle.low_price,
    candle.close_price,
    'market_data.chainlink_btcusd_one_minute_candles'::text AS source_relation,
    candle.capture_artifact_id::text AS source_artifact_id,
    1 AS source_priority
  FROM market_data.chainlink_btcusd_one_minute_candles candle
  JOIN ingester.capture_artifacts artifact
    ON artifact.artifact_id = candle.capture_artifact_id
   AND artifact.status = 'completed'
  WHERE candle.symbol = %(candle_symbol)s
    AND candle.close_timestamp >=
      %(range_start)s - (%(history_minutes)s * interval '1 minute')
    AND candle.close_timestamp < %(range_end)s
    AND candle.close_timestamp = candle.open_timestamp + interval '1 minute'
),
canonical AS (
  SELECT
    candle.open_timestamp,
    candle.close_timestamp,
    greatest(candle.received_at, candle.provider_available_at) AS available_at,
    candle.open_price,
    candle.high_price,
    candle.low_price,
    candle.close_price,
    'market_data.chainlink_btcusd_one_minute_candles'::text AS source_relation,
    candle.capture_artifact_id::text AS source_artifact_id,
    2 AS source_priority
  FROM market_data.chainlink_btcusd_one_minute_candles candle
  WHERE candle.source = 'chainlink_candlestick'
    AND candle.symbol = %(candle_symbol)s
    AND candle.close_timestamp >=
      %(range_start)s - (%(history_minutes)s * interval '1 minute')
    AND candle.close_timestamp < %(range_end)s
    AND candle.close_timestamp = candle.open_timestamp + interval '1 minute'
    AND greatest(candle.received_at, candle.provider_available_at) IS NOT NULL
    AND candle.close_timestamp <= greatest(
      candle.received_at, candle.provider_available_at
    )
)
SELECT DISTINCT ON (open_timestamp)
  open_timestamp,
  close_timestamp,
  available_at,
  open_price::double precision AS open_price,
  high_price::double precision AS high_price,
  low_price::double precision AS low_price,
  close_price::double precision AS close_price,
  source_relation,
  source_artifact_id
FROM (
  SELECT * FROM legacy
  UNION ALL
  SELECT * FROM canonical
) source
ORDER BY open_timestamp, source_priority DESC;
