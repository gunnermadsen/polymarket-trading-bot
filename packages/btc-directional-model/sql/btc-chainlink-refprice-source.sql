SELECT
  report.source_timestamp,
  report.received_at,
  report.valid_from_timestamp,
  report.expires_at,
  report.price::double precision AS price,
  report.bid::double precision AS bid,
  report.ask::double precision AS ask
FROM market_data.chainlink_btcusd_reference_prices report
WHERE report.source = 'pmdata_chainlink_streams'
  AND report.feed_id = %(refprice_feed_id)s
  AND report.source_timestamp >=
    %(range_start)s - (%(history_seconds)s * interval '1 second')
  AND report.source_timestamp < %(range_end)s
ORDER BY report.source_timestamp, report.received_at;
