SELECT
  interest.source_timestamp,
  interest.period_seconds,
  interest.sum_open_interest::double precision AS sum_open_interest,
  interest.sum_open_interest_value::double precision AS sum_open_interest_value
FROM polymarket.binance_btcusdt_five_minute_open_interest interest
JOIN polymarket.backfill_artifacts artifact
  ON artifact.artifact_id = interest.artifact_id
 AND artifact.status = 'completed'
WHERE interest.symbol = %(open_interest_symbol)s
  AND interest.source_timestamp >=
    %(range_start)s - (%(history_minutes)s * interval '1 minute')
  AND interest.source_timestamp < %(range_end)s
  AND interest.period_seconds = 300
ORDER BY interest.source_timestamp;
