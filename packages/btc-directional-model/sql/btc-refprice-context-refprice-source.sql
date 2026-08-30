WITH eligible AS (
  SELECT
    report.source,
    report.feed_id,
    report.source_timestamp,
    report.provider_available_at,
    report.received_at,
    greatest(
      report.source_timestamp,
      report.valid_from_timestamp,
      coalesce(report.provider_available_at, report.received_at),
      report.received_at
    ) AS available_at,
    report.valid_from_timestamp,
    report.expires_at,
    report.price,
    report.bid,
    report.ask,
    report.report_sha256,
    COALESCE(
      report.backfill_artifact_id,
      report.capture_artifact_id
    ) AS source_artifact_id,
    CASE report.source
      WHEN 'chainlink_data_streams' THEN 2
      WHEN 'pmdata_chainlink_streams' THEN 1
    END AS source_priority
  FROM market_data.chainlink_btcusd_reference_prices report
  LEFT JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = report.backfill_artifact_id
  WHERE report.source IN (
      'pmdata_chainlink_streams',
      'chainlink_data_streams'
    )
    AND report.source_timestamp >= %(range_start)s - interval '125 seconds'
    AND report.source_timestamp < %(range_end)s
    AND report.valid_from_timestamp IS NOT NULL
    AND report.received_at IS NOT NULL
    AND report.price > 0
    AND (
      report.backfill_artifact_id IS NULL
      OR artifact.status = 'completed'
    )
)
SELECT DISTINCT ON (source_timestamp)
  source,
  feed_id,
  source_timestamp,
  provider_available_at,
  received_at,
  available_at,
  valid_from_timestamp,
  expires_at,
  price::double precision AS price,
  bid::double precision AS bid,
  ask::double precision AS ask,
  report_sha256,
  source_artifact_id::text AS source_artifact_id,
  (
    'market_data.chainlink_btcusd_reference_prices:' || source
  )::text AS source_relation
FROM eligible
ORDER BY source_timestamp, source_priority DESC, available_at, report_sha256;
