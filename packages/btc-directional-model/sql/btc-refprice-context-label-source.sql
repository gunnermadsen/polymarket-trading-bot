SELECT
  market.market_id,
  market.window_start,
  market.window_end,
  market.official_outcome,
  market.official_resolved_at,
  market.official_resolution_received_at,
  market.reference_price::double precision AS legacy_open_price,
  market.resolution_price::double precision AS legacy_close_price,
  CASE
    WHEN market.official_outcome = 'up' THEN 1
    WHEN market.official_outcome = 'down' THEN 0
    ELSE NULL
  END AS official_label_up
FROM polymarket.btc_interval_markets market
WHERE market.window_start >= %(batch_start)s
  AND market.window_start < %(batch_end)s
  AND market.validation_status = 'valid'
ORDER BY market.window_start, market.market_id;
