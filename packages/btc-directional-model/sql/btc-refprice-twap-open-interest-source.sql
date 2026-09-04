SELECT
  source_timestamp,
  COALESCE(received_at, provider_available_at) AS available_at,
  period_seconds,
  sum_open_interest::double precision AS sum_open_interest,
  sum_open_interest_value::double precision AS sum_open_interest_value
FROM market_data.binance_futures_btcusdt_open_interest
WHERE source = 'binance_usd_m_futures'
  AND symbol = 'BTCUSDT'
  AND source_timestamp >= %(range_start)s - interval '65 minutes'
  AND source_timestamp < %(range_end)s
  AND period_seconds = 300
  AND source_timestamp <= received_at
ORDER BY source_timestamp;
