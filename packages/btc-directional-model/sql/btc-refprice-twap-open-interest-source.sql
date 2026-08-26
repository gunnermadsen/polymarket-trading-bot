WITH legacy AS (
  SELECT
    interest.source_timestamp,
    interest.source_timestamp AS available_at,
    interest.period_seconds,
    interest.sum_open_interest::double precision AS sum_open_interest,
    interest.sum_open_interest_value::double precision AS sum_open_interest_value,
    1 AS source_priority
  FROM polymarket.binance_btcusdt_five_minute_open_interest interest
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = interest.artifact_id
   AND artifact.status = 'completed'
  WHERE interest.symbol = 'BTCUSDT'
    AND interest.source_timestamp >= %(range_start)s - interval '65 minutes'
    AND interest.source_timestamp < %(range_end)s
    AND interest.period_seconds = 300
),
canonical AS (
  SELECT
    interest.source_timestamp,
    COALESCE(interest.received_at, interest.provider_available_at) AS available_at,
    interest.period_seconds,
    interest.sum_open_interest::double precision AS sum_open_interest,
    interest.sum_open_interest_value::double precision AS sum_open_interest_value,
    2 AS source_priority
  FROM market_data.binance_futures_btcusdt_open_interest interest
  WHERE interest.symbol = 'BTCUSDT'
    AND interest.source_timestamp >= %(range_start)s - interval '65 minutes'
    AND interest.source_timestamp < %(range_end)s
    AND interest.period_seconds = 300
    AND interest.received_at IS NOT NULL
    AND interest.source_timestamp <= interest.received_at
)
SELECT DISTINCT ON (source_timestamp)
  source_timestamp,
  available_at,
  period_seconds,
  sum_open_interest,
  sum_open_interest_value
FROM (
  SELECT * FROM legacy
  UNION ALL
  SELECT * FROM canonical
) source
ORDER BY source_timestamp, source_priority DESC;
