SELECT
  market.market_id,
  market.window_start,
  start_report.source_timestamp AS twap_start_source_timestamp,
  start_report.provider_received_at AS twap_start_received_at,
  start_report.valid_from_timestamp AS twap_start_valid_from,
  start_report.expires_at AS twap_start_expires_at,
  start_report.twap_price::double precision AS twap_start_price,
  end_report.source_timestamp AS twap_end_source_timestamp,
  end_report.provider_received_at AS twap_end_received_at,
  end_report.valid_from_timestamp AS twap_end_valid_from,
  end_report.expires_at AS twap_end_expires_at,
  end_report.twap_price::double precision AS twap_end_price,
  CASE WHEN end_report.twap_price > start_report.twap_price THEN 1 ELSE 0 END AS twap_label_up
FROM polymarket.btc_interval_markets market
JOIN market_data.pmdata_chainlink_btcusd_twap start_report
  ON start_report.window_seconds = 60
 AND start_report.source_timestamp = market.window_start
JOIN market_data.pmdata_chainlink_btcusd_twap end_report
  ON end_report.window_seconds = 60
 AND end_report.source_timestamp = market.window_end
WHERE market.window_start >= %(range_start)s
  AND market.window_start < %(range_end)s
  AND market.official_outcome IN ('up', 'down')
  AND start_report.provider_received_at IS NOT NULL
  AND start_report.valid_from_timestamp IS NOT NULL
  AND start_report.expires_at IS NOT NULL
  AND end_report.provider_received_at IS NOT NULL
  AND end_report.valid_from_timestamp IS NOT NULL
  AND end_report.expires_at IS NOT NULL
  AND start_report.valid_from_timestamp <= start_report.source_timestamp
  AND end_report.valid_from_timestamp <= end_report.source_timestamp
  AND start_report.provider_received_at >= start_report.source_timestamp
  AND end_report.provider_received_at >= end_report.source_timestamp
  AND start_report.twap_price > 0
  AND end_report.twap_price > 0
ORDER BY market.window_start;
