WITH legacy AS (
  SELECT
    round.price::double precision AS oracle_price,
    round.source_timestamp AS oracle_source_timestamp,
    round.block_timestamp AS oracle_block_timestamp,
    round.phase_id AS oracle_phase_id,
    round.aggregator_round_id AS oracle_round_id,
    round.block_number AS oracle_block_number,
    round.log_index AS oracle_log_index,
    1 AS source_priority
  FROM polymarket.polygon_chainlink_btcusd_oracle_rounds round
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = round.artifact_id
   AND artifact.status = 'completed'
  WHERE round.source_timestamp >= %(range_start)s - interval '30 minutes'
    AND round.source_timestamp < %(range_end)s
    AND round.source_timestamp <= round.block_timestamp
),
canonical AS (
  SELECT
    round.price::double precision AS oracle_price,
    round.source_timestamp AS oracle_source_timestamp,
    COALESCE(round.received_at, round.provider_available_at, round.block_timestamp)
      AS oracle_block_timestamp,
    round.phase_id AS oracle_phase_id,
    round.aggregator_round_id AS oracle_round_id,
    round.block_number AS oracle_block_number,
    round.log_index AS oracle_log_index,
    2 AS source_priority
  FROM market_data.polygon_chainlink_btcusd_oracle_rounds round
  WHERE round.source_timestamp >= %(range_start)s - interval '30 minutes'
    AND round.source_timestamp < %(range_end)s
    AND COALESCE(round.received_at, round.provider_available_at, round.block_timestamp) IS NOT NULL
    AND round.source_timestamp <= COALESCE(
      round.received_at, round.provider_available_at, round.block_timestamp
    )
)
SELECT DISTINCT ON (
  oracle_phase_id, oracle_round_id, oracle_block_number, oracle_log_index
)
  oracle_price,
  oracle_source_timestamp,
  oracle_block_timestamp,
  oracle_phase_id,
  oracle_round_id,
  oracle_block_number,
  oracle_log_index
FROM (
  SELECT * FROM legacy
  UNION ALL
  SELECT * FROM canonical
) source
ORDER BY
  oracle_phase_id,
  oracle_round_id,
  oracle_block_number,
  oracle_log_index,
  source_priority DESC;
