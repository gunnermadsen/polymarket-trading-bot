WITH market_facts AS MATERIALIZED (
  SELECT
    fact.market_id,
    max(fact.value) FILTER (WHERE fact.fact_type = 'opening_boundary') AS opening_boundary,
    max(fact.value) FILTER (WHERE fact.fact_type = 'final_price') AS final_price,
    count(*) FILTER (WHERE fact.fact_type = 'opening_boundary') AS opening_fact_count,
    count(*) FILTER (WHERE fact.fact_type = 'final_price') AS final_fact_count
  FROM polymarket.btc_market_reference_facts fact
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = fact.artifact_id
   AND artifact.status = 'completed'
  WHERE fact.source_effective_at >= %(batch_start)s
    AND fact.source_effective_at <= %(batch_end)s
    AND fact.fact_type IN ('opening_boundary', 'final_price')
  GROUP BY fact.market_id
),
eligible_markets AS MATERIALIZED (
  SELECT
    market.market_id,
    market.window_start,
    market.window_end,
    market.official_outcome,
    COALESCE(facts.opening_boundary, market.reference_price)::double precision
      AS legacy_open_price,
    COALESCE(facts.final_price, market.resolution_price)::double precision
      AS legacy_close_price
  FROM polymarket.btc_interval_markets market
  LEFT JOIN market_facts facts USING (market_id)
  WHERE market.window_start >= %(batch_start)s
    AND market.window_start < %(batch_end)s
    AND market.validation_status = 'valid'
    AND market.official_outcome IN ('up', 'down')
    AND COALESCE(facts.opening_fact_count, 0) <= 1
    AND COALESCE(facts.final_fact_count, 0) <= 1
)
SELECT
  market.*,
  opening.source_timestamp AS twap_open_source_timestamp,
  opening.valid_from_timestamp AS twap_open_valid_from_timestamp,
  opening.provider_received_at AS twap_open_provider_received_at,
  opening.twap_price::double precision AS twap_open_price,
  opening.effective_timestamp_rows AS twap_open_effective_timestamp_rows,
  closing.source_timestamp AS twap_close_source_timestamp,
  closing.valid_from_timestamp AS twap_close_valid_from_timestamp,
  closing.provider_received_at AS twap_close_provider_received_at,
  closing.twap_price::double precision AS twap_close_price,
  closing.effective_timestamp_rows AS twap_close_effective_timestamp_rows
FROM eligible_markets market
LEFT JOIN LATERAL (
  SELECT report.*,
         count(*) OVER (PARTITION BY report.valid_from_timestamp)::integer
           AS effective_timestamp_rows
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
         count(*) OVER (PARTITION BY report.valid_from_timestamp)::integer
           AS effective_timestamp_rows
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
