SELECT
  report.source_timestamp,
  report.valid_from_timestamp,
  report.provider_available_at,
  report.received_at,
  report.price::double precision AS price,
  report.bid::double precision AS bid,
  report.ask::double precision AS ask,
  report.report_version,
  report.source_date,
  report.archive_row_number,
  report.backfill_artifact_id::text AS artifact_id,
  report.report_sha256
FROM market_data.chainlink_btcusd_reference_prices report
JOIN polymarket.backfill_artifacts artifact
  ON artifact.artifact_id = report.backfill_artifact_id
 AND artifact.status = 'completed'
WHERE report.source = 'pmdata_chainlink_streams'
  AND report.source_timestamp >= %(batch_start)s - interval '125 seconds'
  AND report.source_timestamp < %(batch_end)s
  AND report.valid_from_timestamp IS NOT NULL
  AND report.provider_available_at IS NOT NULL
  AND report.received_at IS NOT NULL
  AND report.price > 0
ORDER BY report.source_timestamp, report.archive_row_number;
