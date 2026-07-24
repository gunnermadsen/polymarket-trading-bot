import { MigrationInterface, QueryRunner } from 'typeorm';

export class PruneSupersededBtcOrderbookMaterialization1784916000000
  implements MigrationInterface
{
  name = 'PruneSupersededBtcOrderbookMaterialization1784916000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      DECLARE
        raw_chunk_start CONSTANT timestamptz := '2026-07-21 00:00:00+00';
        snapshot_start CONSTANT timestamptz := '2026-07-22 00:00:00+00';
        snapshot_end CONSTANT timestamptz := '2026-07-23 00:00:00+00';
        raw_artifact_count bigint;
        raw_record_count bigint;
        compact_artifact_count bigint;
        compact_artifact_records bigint;
        snapshot_count bigint;
        market_count bigint;
        exact_market_count bigint;
        invalid_snapshot_count bigint;
        replacement_count bigint;
        pruned_evidence_count bigint;
        overlapping_chunk_count bigint;
      BEGIN
        WITH expected_raw(logical_key) AS (
          SELECT 'pmxt:v2:polymarket_orderbook:2026-07-21T23'
          UNION ALL
          SELECT format(
            'pmxt:v2:polymarket_orderbook:2026-07-22T%s',
            lpad(hour::text, 2, '0')
          )
          FROM generate_series(0, 23) AS hours(hour)
        )
        SELECT count(*), COALESCE(sum(a.record_count), 0)
        INTO raw_artifact_count, raw_record_count
        FROM expected_raw e
        JOIN polymarket.backfill_artifacts a
          ON a.logical_key = e.logical_key
         AND a.ingester_key = 'polymarket_btc_five_minute_orderbooks'
         AND a.provider = 'pmxt_v2'
         AND a.status = 'completed';

        IF raw_artifact_count = 0 THEN
          RAISE NOTICE
            'no superseded July 22 BTC orderbook materialization is present; skipping prune';
          RETURN;
        END IF;

        IF raw_artifact_count <> 25 OR raw_record_count <> 92873892 THEN
          RAISE EXCEPTION
            'raw PMXT prune guard failed: expected 25 artifacts and 92873892 rows, found % artifacts and % rows',
            raw_artifact_count, raw_record_count;
        END IF;

        IF EXISTS (
          WITH expected_raw(logical_key) AS (
            SELECT 'pmxt:v2:polymarket_orderbook:2026-07-21T23'
            UNION ALL
            SELECT format(
              'pmxt:v2:polymarket_orderbook:2026-07-22T%s',
              lpad(hour::text, 2, '0')
            )
            FROM generate_series(0, 23) AS hours(hour)
          )
          SELECT 1
          FROM polymarket.backfill_artifacts a
          WHERE a.ingester_key = 'polymarket_btc_five_minute_orderbooks'
            AND a.source_date >= DATE '2026-07-21'
            AND a.source_date <= DATE '2026-07-22'
            AND NOT EXISTS (
              SELECT 1 FROM expected_raw e WHERE e.logical_key = a.logical_key
            )
        ) THEN
          RAISE EXCEPTION
            'raw PMXT prune guard failed: the target chunks contain artifacts outside the replacement set';
        END IF;

        SELECT count(*), COALESCE(sum(record_count), 0)
        INTO compact_artifact_count, compact_artifact_records
        FROM polymarket.backfill_artifacts
        WHERE ingester_key = 'polymarket_btc_five_minute_execution_snapshots'
          AND provider = 'pmxt_v2_execution_snapshots'
          AND source_date = DATE '2026-07-22'
          AND status = 'completed'
          AND record_count = 14400;

        IF compact_artifact_count <> 24 OR compact_artifact_records <> 345600 THEN
          RAISE EXCEPTION
            'compact PMXT prune guard failed: expected 24 artifacts and 345600 rows, found % artifacts and % rows',
            compact_artifact_count, compact_artifact_records;
        END IF;

        WITH per_market AS MATERIALIZED (
          SELECT s.market_id, m.window_start, m.window_end, count(*) AS rows,
            min(sampled_at) AS first_sample,
            max(sampled_at) AS last_sample
          FROM polymarket.btc_market_execution_snapshots s
          JOIN polymarket.btc_interval_markets m USING (market_id)
          WHERE s.sampled_at >= snapshot_start AND s.sampled_at < snapshot_end
          GROUP BY s.market_id, m.window_start, m.window_end
        )
        SELECT
          (SELECT count(*)
             FROM polymarket.btc_market_execution_snapshots
            WHERE sampled_at >= snapshot_start AND sampled_at < snapshot_end),
          count(*),
          count(*) FILTER (
            WHERE rows = 1200
              AND first_sample = window_start
              AND last_sample = window_end - interval '250 milliseconds'
          )
        INTO snapshot_count, market_count, exact_market_count
        FROM per_market;

        SELECT count(*)
        INTO invalid_snapshot_count
        FROM polymarket.btc_market_execution_snapshots s
        JOIN polymarket.backfill_artifacts a USING (artifact_id)
        WHERE s.sampled_at >= snapshot_start
          AND s.sampled_at < snapshot_end
          AND (
            s.schema_version <> 'btc5m-book-250ms-v1'
            OR mod((extract(epoch FROM s.sampled_at) * 1000)::bigint, 250) <> 0
            OR s.up_provider_received_at > s.sampled_at
            OR s.down_provider_received_at > s.sampled_at
            OR a.ingester_key <> 'polymarket_btc_five_minute_execution_snapshots'
            OR a.status <> 'completed'
          );

        IF snapshot_count <> 345600
          OR market_count <> 288
          OR exact_market_count <> 288
          OR invalid_snapshot_count <> 0
        THEN
          RAISE EXCEPTION
            'compact PMXT coverage guard failed: rows %, markets %, exact markets %, invalid rows %',
            snapshot_count, market_count, exact_market_count, invalid_snapshot_count;
        END IF;

        SELECT count(DISTINCT r.source_artifact_id)
        INTO replacement_count
        FROM polymarket.backfill_materialization_retention_events r
        JOIN polymarket.backfill_artifacts a
          ON a.artifact_id = r.source_artifact_id
        JOIN polymarket.backfill_artifacts replacement
          ON replacement.artifact_id = r.replacement_artifact_id
        WHERE r.materialization = 'polymarket.btc_orderbook_archive_events'
          AND r.action = 'replaced'
          AND a.ingester_key = 'polymarket_btc_five_minute_orderbooks'
          AND replacement.ingester_key =
            'polymarket_btc_five_minute_execution_snapshots'
          AND replacement.status = 'completed'
          AND (
            a.logical_key = 'pmxt:v2:polymarket_orderbook:2026-07-21T23'
            OR a.logical_key LIKE 'pmxt:v2:polymarket_orderbook:2026-07-22T__'
          );

        IF replacement_count <> 25 THEN
          RAISE EXCEPTION
            'raw PMXT lineage guard failed: expected 25 replacement events, found %',
            replacement_count;
        END IF;

        IF EXISTS (
          SELECT 1
          FROM polymarket.backfill_jobs
          WHERE ingester_key IN (
            'polymarket_btc_five_minute_orderbooks',
            'polymarket_btc_five_minute_execution_snapshots'
          )
            AND status IN ('queued', 'running', 'cancel_requested')
            AND range_start < snapshot_end
            AND range_end > raw_chunk_start
        ) THEN
          RAISE EXCEPTION
            'raw PMXT prune guard failed: an overlapping ingestion job is active';
        END IF;

        SELECT count(*)
        INTO overlapping_chunk_count
        FROM timescaledb_information.chunks
        WHERE hypertable_schema = 'polymarket'
          AND hypertable_name = 'btc_orderbook_archive_events'
          AND range_start < snapshot_end
          AND range_end > raw_chunk_start;

        IF overlapping_chunk_count <> 2 OR EXISTS (
          SELECT 1
          FROM timescaledb_information.chunks
          WHERE hypertable_schema = 'polymarket'
            AND hypertable_name = 'btc_orderbook_archive_events'
            AND range_start < snapshot_end
            AND range_end > raw_chunk_start
            AND (
              range_start NOT IN (
                raw_chunk_start,
                snapshot_start
              )
              OR range_end NOT IN (
                snapshot_start,
                snapshot_end
              )
            )
        ) THEN
          RAISE EXCEPTION
            'raw PMXT prune guard failed: expected exactly the July 21 and July 22 daily chunks';
        END IF;

        INSERT INTO polymarket.backfill_materialization_retention_events (
          source_artifact_id, replacement_artifact_id, materialization, action,
          source_record_count, metadata
        )
        SELECT
          r.source_artifact_id,
          r.replacement_artifact_id,
          r.materialization,
          'pruned',
          r.source_record_count,
          jsonb_build_object(
            'replacement_schema', 'btc5m-book-250ms-v1',
            'snapshot_range_start', snapshot_start,
            'snapshot_range_end', snapshot_end,
            'snapshot_rows', snapshot_count,
            'market_count', market_count,
            'sample_resolution_ms', 250,
            'validation', jsonb_build_object(
              'exact_market_coverage', true,
              'causal_provider_timestamps', true,
              'completed_replacement_artifacts', true
            )
          )
        FROM polymarket.backfill_materialization_retention_events r
        JOIN polymarket.backfill_artifacts a
          ON a.artifact_id = r.source_artifact_id
        WHERE r.materialization = 'polymarket.btc_orderbook_archive_events'
          AND r.action = 'replaced'
          AND a.ingester_key = 'polymarket_btc_five_minute_orderbooks'
          AND (
            a.logical_key = 'pmxt:v2:polymarket_orderbook:2026-07-21T23'
            OR a.logical_key LIKE 'pmxt:v2:polymarket_orderbook:2026-07-22T__'
          )
        ON CONFLICT (source_artifact_id, materialization, action) DO NOTHING;

        SELECT count(DISTINCT r.source_artifact_id)
        INTO pruned_evidence_count
        FROM polymarket.backfill_materialization_retention_events r
        JOIN polymarket.backfill_artifacts a
          ON a.artifact_id = r.source_artifact_id
        WHERE r.materialization = 'polymarket.btc_orderbook_archive_events'
          AND r.action = 'pruned'
          AND a.ingester_key = 'polymarket_btc_five_minute_orderbooks'
          AND (
            a.logical_key = 'pmxt:v2:polymarket_orderbook:2026-07-21T23'
            OR a.logical_key LIKE 'pmxt:v2:polymarket_orderbook:2026-07-22T__'
          );

        IF pruned_evidence_count <> 25 THEN
          RAISE EXCEPTION
            'raw PMXT prune guard failed: expected 25 immutable prune events, found %',
            pruned_evidence_count;
        END IF;

        PERFORM drop_chunks(
          'polymarket.btc_orderbook_archive_events',
          older_than => snapshot_end,
          newer_than => raw_chunk_start
        );
      END;
      $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        RAISE EXCEPTION
          'superseded raw PMXT chunks cannot be restored by migration rollback';
      END;
      $$;
    `);
  }
}
