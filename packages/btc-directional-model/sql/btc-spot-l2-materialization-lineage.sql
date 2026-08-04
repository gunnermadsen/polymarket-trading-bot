SELECT
  artifact.artifact_id::text AS artifact_id,
  artifact.provider,
  artifact.logical_key,
  artifact.source_date,
  artifact.metadata ->> 'materialization_contract' AS materialization_contract,
  artifact.checksum_algorithm,
  artifact.actual_checksum,
  artifact.record_count,
  artifact.minimum_source_timestamp,
  artifact.maximum_source_timestamp,
  count(*)::bigint AS source_rows,
  count(DISTINCT feature.second_start)::bigint AS qualified_seconds,
  min(feature.second_start) AS minimum_second_start,
  max(feature.second_start) AS maximum_second_start,
  min(feature.available_at) AS minimum_available_at,
  max(feature.available_at) AS maximum_available_at
FROM polymarket.binance_spot_btcusdt_l2_one_second_features feature
JOIN polymarket.backfill_artifacts artifact
  ON artifact.artifact_id = feature.artifact_id
 AND artifact.ingester_key = 'binance_spot_btcusdt_l2_one_second_features'
 AND artifact.status = 'completed'
 AND artifact.metadata ->> 'materialization_contract' IN (
   'cryptohft-binance-spot-btcusdt-l2-features-v1',
   'coinapi-binance-spot-btcusdt-l2-snapshots-v1',
   'huggingface-goooddy-binance-spot-btcusdt-l2-features-v1'
 )
WHERE feature.symbol = 'BTCUSDT'
  AND feature.available_at >= %(batch_start)s
  AND feature.available_at < %(batch_end)s
GROUP BY
  artifact.artifact_id,
  artifact.provider,
  artifact.logical_key,
  artifact.source_date,
  artifact.metadata ->> 'materialization_contract',
  artifact.checksum_algorithm,
  artifact.actual_checksum,
  artifact.record_count,
  artifact.minimum_source_timestamp,
  artifact.maximum_source_timestamp
ORDER BY artifact.logical_key, artifact.artifact_id;
