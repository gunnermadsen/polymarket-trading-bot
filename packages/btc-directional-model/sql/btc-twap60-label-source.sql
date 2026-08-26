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
  opening.twap_price::double precision AS twap_open_price,
  opening.report_version AS twap_open_report_version,
  opening.artifact_id::text AS twap_open_artifact_id,
  opening.archive_row_number AS twap_open_archive_row_number,
  opening.effective_timestamp_rows AS twap_open_effective_timestamp_rows,
  closing.source_timestamp AS twap_close_source_timestamp,
  closing.valid_from_timestamp AS twap_close_valid_from_timestamp,
  closing.provider_received_at AS twap_close_provider_received_at,
  closing.twap_price::double precision AS twap_close_price,
  closing.report_version AS twap_close_report_version,
  closing.artifact_id::text AS twap_close_artifact_id,
  closing.archive_row_number AS twap_close_archive_row_number,
  closing.effective_timestamp_rows AS twap_close_effective_timestamp_rows
FROM eligible_markets market
LEFT JOIN LATERAL (
  SELECT report.*,
         count(*) OVER (
           PARTITION BY report.valid_from_timestamp
         )::integer AS effective_timestamp_rows
  FROM market_data.pmdata_chainlink_btcusd_twap report
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = report.artifact_id
   AND artifact.status = 'completed'
  WHERE report.window_seconds = 60
    AND report.valid_from_timestamp <= market.window_start
    AND report.source_timestamp >= market.window_start - interval '5 seconds'
    AND report.source_timestamp <= market.window_start + interval '5 seconds'
  ORDER BY report.valid_from_timestamp DESC, report.source_timestamp DESC,
           report.archive_row_number DESC
  LIMIT 1
) opening ON true
LEFT JOIN LATERAL (
  SELECT report.*,
         count(*) OVER (
           PARTITION BY report.valid_from_timestamp
         )::integer AS effective_timestamp_rows
  FROM market_data.pmdata_chainlink_btcusd_twap report
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = report.artifact_id
   AND artifact.status = 'completed'
  WHERE report.window_seconds = 60
    AND report.valid_from_timestamp <= market.window_end
    AND report.source_timestamp >= market.window_end - interval '5 seconds'
    AND report.source_timestamp <= market.window_end + interval '5 seconds'
  ORDER BY report.valid_from_timestamp DESC, report.source_timestamp DESC,
           report.archive_row_number DESC
  LIMIT 1
) closing ON true
ORDER BY market.window_start, market.market_id;
