WITH daily_rounds AS (
  SELECT
    round.price::double precision AS oracle_price,
    round.source_timestamp AS oracle_source_timestamp,
    round.block_timestamp AS oracle_block_timestamp,
    round.phase_id AS oracle_phase_id,
    round.aggregator_round_id AS oracle_round_id,
    round.block_number AS oracle_block_number,
    round.log_index AS oracle_log_index
  FROM polymarket.polygon_chainlink_btcusd_oracle_rounds round
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = round.artifact_id
   AND artifact.status = 'completed'
  WHERE round.feed_proxy_address = %(oracle_feed_proxy_address)s
    AND round.source_timestamp >=
      %(batch_start)s
      - (%(oracle_max_publication_delay_seconds)s * interval '1 second')
    AND round.source_timestamp < %(batch_end)s
    AND round.source_timestamp <= round.block_timestamp
    AND round.block_timestamp >= %(batch_start)s
    AND round.block_timestamp < %(batch_end)s
),
prior_causal_round AS (
  SELECT
    round.price::double precision AS oracle_price,
    round.source_timestamp AS oracle_source_timestamp,
    round.block_timestamp AS oracle_block_timestamp,
    round.phase_id AS oracle_phase_id,
    round.aggregator_round_id AS oracle_round_id,
    round.block_number AS oracle_block_number,
    round.log_index AS oracle_log_index
  FROM polymarket.polygon_chainlink_btcusd_oracle_rounds round
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = round.artifact_id
   AND artifact.status = 'completed'
  WHERE round.feed_proxy_address = %(oracle_feed_proxy_address)s
    AND round.source_timestamp >=
      %(batch_start)s
      - (%(oracle_max_publication_delay_seconds)s * interval '1 second')
    AND round.source_timestamp < %(batch_start)s
    AND round.source_timestamp <= round.block_timestamp
    AND round.block_timestamp <= %(batch_start)s
  ORDER BY
    round.block_timestamp DESC,
    round.source_timestamp DESC,
    round.block_number DESC,
    round.log_index DESC
  LIMIT 1
)
SELECT *
FROM (
  SELECT * FROM prior_causal_round
  UNION ALL
  SELECT * FROM daily_rounds
) oracle
ORDER BY
  oracle.oracle_block_timestamp,
  oracle.oracle_source_timestamp,
  oracle.oracle_block_number,
  oracle.oracle_log_index;
