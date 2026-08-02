SELECT
  tick.source_timestamp,
  tick.valid_from_timestamp,
  tick.price::double precision AS price,
  tick.bid::double precision AS bid,
  tick.ask::double precision AS ask
FROM polymarket.chainlink_btcusd_archive_ticks tick
JOIN polymarket.backfill_artifacts artifact
  ON artifact.artifact_id = tick.artifact_id
 AND artifact.status = 'completed'
WHERE tick.feed_id = %(refprice_feed_id)s
  AND tick.source_timestamp >=
    %(range_start)s - (%(history_seconds)s * interval '1 second')
  AND tick.source_timestamp < %(range_end)s
  AND tick.valid_from_timestamp <= tick.source_timestamp
ORDER BY tick.source_timestamp;
