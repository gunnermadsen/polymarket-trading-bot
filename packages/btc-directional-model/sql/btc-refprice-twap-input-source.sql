SELECT
  source_timestamp,
  provider_received_at,
  valid_from_timestamp,
  expires_at,
  window_seconds,
  twap_price::double precision AS twap_price
FROM market_data.pmdata_chainlink_btcusd_twap
WHERE source_timestamp >= %(range_start)s - INTERVAL '65 seconds'
  AND source_timestamp < %(range_end)s
  AND window_seconds IN (30, 60)
  AND provider_received_at IS NOT NULL
  AND valid_from_timestamp IS NOT NULL
  AND expires_at IS NOT NULL
  AND valid_from_timestamp <= source_timestamp
  AND provider_received_at >= source_timestamp
  AND twap_price > 0
ORDER BY window_seconds, source_timestamp;
