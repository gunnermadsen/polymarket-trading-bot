SELECT
  source_timestamp,
  received_at,
  valid_from_timestamp,
  expires_at,
  price::double precision AS price,
  bid::double precision AS bid,
  ask::double precision AS ask,
  report_sha256
FROM market_data.chainlink_btcusd_reference_prices
WHERE source = 'pmdata_chainlink_streams'
  AND source_timestamp >= %(range_start)s - interval '70 seconds'
  AND source_timestamp < %(range_end)s
ORDER BY source_timestamp, received_at;
