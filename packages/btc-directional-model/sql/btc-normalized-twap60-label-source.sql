WITH eligible_markets AS MATERIALIZED (
  SELECT
    market.market_id,
    market.window_start,
    market.window_end,
    market.official_outcome,
    market.reference_price::double precision AS legacy_open_price,
    market.resolution_price::double precision AS legacy_close_price,
    market.reference_source_timestamp AS legacy_open_timestamp,
    market.resolution_source_timestamp AS legacy_close_timestamp
  FROM polymarket.btc_interval_markets market
  WHERE market.window_start >= %(batch_start)s
    AND market.window_start < %(batch_end)s
    AND market.validation_status = 'valid'
    AND market.official_outcome IN ('up', 'down')
)
SELECT
  market.*,
  opening.source_timestamp AS twap_open_source_timestamp,
  opening.valid_from_timestamp AS twap_open_valid_from_timestamp,
  opening.provider_received_at AS twap_open_provider_received_at,
  opening.twap_price AS twap_open_price,
  opening.report_version AS twap_open_report_version,
  opening.artifact_id AS twap_open_artifact_id,
  opening.archive_row_number AS twap_open_archive_row_number,
  opening.effective_timestamp_rows AS twap_open_effective_timestamp_rows,
  opening.source_family AS twap_open_source_family,
  closing.source_timestamp AS twap_close_source_timestamp,
  closing.valid_from_timestamp AS twap_close_valid_from_timestamp,
  closing.provider_received_at AS twap_close_provider_received_at,
  closing.twap_price AS twap_close_price,
  closing.report_version AS twap_close_report_version,
  closing.artifact_id AS twap_close_artifact_id,
  closing.archive_row_number AS twap_close_archive_row_number,
  closing.effective_timestamp_rows AS twap_close_effective_timestamp_rows,
  closing.source_family AS twap_close_source_family
FROM eligible_markets market
LEFT JOIN LATERAL (
  SELECT candidate.*
  FROM (
    SELECT
      1 AS source_priority,
      report.source_timestamp,
      report.valid_from_timestamp,
      report.provider_received_at,
      report.twap_price::double precision AS twap_price,
      report.report_version,
      report.artifact_id::text AS artifact_id,
      report.archive_row_number,
      1::integer AS effective_timestamp_rows,
      'pmdata_archive'::text AS source_family
    FROM market_data.pmdata_chainlink_btcusd_twap report
    JOIN polymarket.backfill_artifacts artifact
      ON artifact.artifact_id = report.artifact_id
     AND artifact.status = 'completed'
    WHERE report.window_seconds = 60
      AND report.valid_from_timestamp <= market.window_start
      AND report.source_timestamp >= market.window_start - interval '5 seconds'
      AND report.source_timestamp <= market.window_start + interval '5 seconds'
    UNION ALL
    SELECT
      2 AS source_priority,
      report.source_timestamp,
      report.source_timestamp AS valid_from_timestamp,
      report.received_at AS provider_received_at,
      report.twap_price::double precision AS twap_price,
      coalesce(report.strategy_key, report.source) AS report_version,
      report.capture_artifact_id::text AS artifact_id,
      NULL::bigint AS archive_row_number,
      1::integer AS effective_timestamp_rows,
      'live_capture'::text AS source_family
    FROM market_data.polymarket_chainlink_btcusd_twap report
    WHERE report.symbol = 'btc/usd'
      AND report.window_seconds = 60
      AND report.source_timestamp <= market.window_start
      AND report.source_timestamp >= market.window_start - interval '5 seconds'
  ) candidate
  ORDER BY candidate.source_priority, candidate.valid_from_timestamp DESC,
           candidate.source_timestamp DESC, candidate.archive_row_number DESC NULLS LAST
  LIMIT 1
) opening ON true
LEFT JOIN LATERAL (
  SELECT candidate.*
  FROM (
    SELECT
      1 AS source_priority,
      report.source_timestamp,
      report.valid_from_timestamp,
      report.provider_received_at,
      report.twap_price::double precision AS twap_price,
      report.report_version,
      report.artifact_id::text AS artifact_id,
      report.archive_row_number,
      1::integer AS effective_timestamp_rows,
      'pmdata_archive'::text AS source_family
    FROM market_data.pmdata_chainlink_btcusd_twap report
    JOIN polymarket.backfill_artifacts artifact
      ON artifact.artifact_id = report.artifact_id
     AND artifact.status = 'completed'
    WHERE report.window_seconds = 60
      AND report.valid_from_timestamp <= market.window_end
      AND report.source_timestamp >= market.window_end - interval '5 seconds'
      AND report.source_timestamp <= market.window_end + interval '5 seconds'
    UNION ALL
    SELECT
      2 AS source_priority,
      report.source_timestamp,
      report.source_timestamp AS valid_from_timestamp,
      report.received_at AS provider_received_at,
      report.twap_price::double precision AS twap_price,
      coalesce(report.strategy_key, report.source) AS report_version,
      report.capture_artifact_id::text AS artifact_id,
      NULL::bigint AS archive_row_number,
      1::integer AS effective_timestamp_rows,
      'live_capture'::text AS source_family
    FROM market_data.polymarket_chainlink_btcusd_twap report
    WHERE report.symbol = 'btc/usd'
      AND report.window_seconds = 60
      AND report.source_timestamp <= market.window_end
      AND report.source_timestamp >= market.window_end - interval '5 seconds'
  ) candidate
  ORDER BY candidate.source_priority, candidate.valid_from_timestamp DESC,
           candidate.source_timestamp DESC, candidate.archive_row_number DESC NULLS LAST
  LIMIT 1
) closing ON true
ORDER BY market.window_start, market.market_id;
