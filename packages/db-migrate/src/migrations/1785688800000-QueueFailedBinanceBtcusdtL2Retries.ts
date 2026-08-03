import { MigrationInterface, QueryRunner } from 'typeorm';

const INGESTER_KEY = 'binance_btcusdt_l2_one_second_features';
const MATERIALIZATION_CONTRACT =
  'cryptohft-binance-futures-btcusdt-l2-features-v1';
const RETRY_GENERATION = 3;

export class QueueFailedBinanceBtcusdtL2Retries1785688800000
  implements MigrationInterface
{
  name = 'QueueFailedBinanceBtcusdtL2Retries1785688800000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TEMPORARY TABLE binance_l2_retry_dates (
        source_date date PRIMARY KEY
      ) ON COMMIT DROP;

      INSERT INTO binance_l2_retry_dates (source_date)
      VALUES
        ('2026-05-18'::date),
        ('2026-05-24'::date),
        ('2026-05-25'::date),
        ('2026-05-26'::date),
        ('2026-05-27'::date),
        ('2026-06-01'::date),
        ('2026-06-04'::date),
        ('2026-06-05'::date),
        ('2026-06-30'::date),
        ('2026-07-01'::date),
        ('2026-07-24'::date);

      DO $$
      DECLARE
        missing_dates text;
        missing_completed_dates text;
        unexpected_dates text;
        existing_retry_dates text;
        completed_artifact_dates text;
        missing_failed_artifact_dates text;
        published_feature_dates text;
        staging_feature_dates text;
        completed_retry_count integer;
        failed_artifact_count integer;
      BEGIN
        SELECT string_agg(expected.source_date::text, ', ' ORDER BY expected.source_date)
        INTO missing_dates
        FROM binance_l2_retry_dates AS expected
        WHERE NOT EXISTS (
          SELECT 1
          FROM polymarket.backfill_jobs AS prior
          WHERE prior.ingester_key = '${INGESTER_KEY}'
            AND prior.request_version = 1
            AND prior.status = 'failed'
            AND prior.attempt = 3
            AND prior.max_attempts = 3
            AND prior.range_start = (
              expected.source_date::timestamp AT TIME ZONE 'UTC'
            )
            AND prior.range_end = (
              (expected.source_date + 1)::timestamp AT TIME ZONE 'UTC'
            )
            AND prior.idempotency_key = format(
              'binance-btcusdt-l2:%s:${MATERIALIZATION_CONTRACT}:retry-2',
              to_char(expected.source_date, 'YYYY-MM-DD')
            )
            AND prior.request ->> 'ingester' = '${INGESTER_KEY}'
            AND prior.request ->> 'request_version' = '1'
            AND prior.request -> 'parameters' = '{}'::jsonb
            AND prior.request ->> 'idempotency_key' = prior.idempotency_key
        );

        IF missing_dates IS NOT NULL THEN
          RAISE EXCEPTION
            'refusing to queue Binance L2 retry generation ${RETRY_GENERATION}; exhausted retry-2 failures are missing or changed for: %',
            missing_dates;
        END IF;

        WITH approved_dates AS (
          SELECT '2026-04-14'::date + shard.day_offset AS source_date
          FROM generate_series(0, 109) AS shard(day_offset)
        )
        SELECT string_agg(
          expected.source_date::text,
          ', '
          ORDER BY expected.source_date
        )
        INTO missing_completed_dates
        FROM approved_dates AS expected
        WHERE NOT EXISTS (
          SELECT 1
          FROM binance_l2_retry_dates AS failed
          WHERE failed.source_date = expected.source_date
        )
          AND NOT EXISTS (
            SELECT 1
            FROM polymarket.backfill_jobs AS completed
            WHERE completed.ingester_key = '${INGESTER_KEY}'
              AND completed.request_version = 1
              AND completed.status = 'completed'
              AND completed.range_start = (
                expected.source_date::timestamp AT TIME ZONE 'UTC'
              )
              AND completed.range_end = (
                (expected.source_date + 1)::timestamp AT TIME ZONE 'UTC'
              )
              AND completed.idempotency_key = format(
                'binance-btcusdt-l2:%s:${MATERIALIZATION_CONTRACT}:retry-2',
                to_char(expected.source_date, 'YYYY-MM-DD')
              )
              AND completed.request ->> 'ingester' = '${INGESTER_KEY}'
              AND completed.request ->> 'request_version' = '1'
              AND completed.request -> 'parameters' = '{}'::jsonb
              AND completed.request ->> 'idempotency_key' =
                completed.idempotency_key
          );

        IF missing_completed_dates IS NOT NULL THEN
          RAISE EXCEPTION
            'refusing to queue Binance L2 retries; completed retry-2 jobs are missing or changed for: %',
            missing_completed_dates;
        END IF;

        WITH approved_dates AS (
          SELECT '2026-04-14'::date + shard.day_offset AS source_date
          FROM generate_series(0, 109) AS shard(day_offset)
        )
        SELECT count(*)::integer
        INTO completed_retry_count
        FROM approved_dates AS expected
        JOIN polymarket.backfill_jobs AS completed
          ON completed.ingester_key = '${INGESTER_KEY}'
         AND completed.request_version = 1
         AND completed.status = 'completed'
         AND completed.range_start = (
           expected.source_date::timestamp AT TIME ZONE 'UTC'
         )
         AND completed.range_end = (
           (expected.source_date + 1)::timestamp AT TIME ZONE 'UTC'
         )
         AND completed.idempotency_key = format(
           'binance-btcusdt-l2:%s:${MATERIALIZATION_CONTRACT}:retry-2',
           to_char(expected.source_date, 'YYYY-MM-DD')
         )
        WHERE NOT EXISTS (
          SELECT 1
          FROM binance_l2_retry_dates AS failed
          WHERE failed.source_date = expected.source_date
        );

        IF completed_retry_count <> 99 THEN
          RAISE EXCEPTION
            'expected exactly 99 completed Binance L2 retry-2 jobs outside the retry set; found %',
            completed_retry_count;
        END IF;

        SELECT string_agg(
          (prior.range_start AT TIME ZONE 'UTC')::date::text,
          ', '
          ORDER BY (prior.range_start AT TIME ZONE 'UTC')::date
        )
        INTO unexpected_dates
        FROM polymarket.backfill_jobs AS prior
        WHERE prior.ingester_key = '${INGESTER_KEY}'
          AND prior.request_version = 1
          AND prior.status = 'failed'
          AND prior.attempt = 3
          AND prior.max_attempts = 3
          AND prior.range_start >= '2026-04-14T00:00:00Z'::timestamptz
          AND prior.range_end <= '2026-08-02T00:00:00Z'::timestamptz
          AND prior.idempotency_key LIKE
            'binance-btcusdt-l2:%:${MATERIALIZATION_CONTRACT}:retry-2'
          AND NOT EXISTS (
            SELECT 1
            FROM binance_l2_retry_dates AS expected
            WHERE expected.source_date =
              (prior.range_start AT TIME ZONE 'UTC')::date
          );

        IF unexpected_dates IS NOT NULL THEN
          RAISE EXCEPTION
            'refusing to queue a partial Binance L2 retry set; unexpected exhausted retry-2 failures exist for: %',
            unexpected_dates;
        END IF;

        SELECT string_agg(expected.source_date::text, ', ' ORDER BY expected.source_date)
        INTO existing_retry_dates
        FROM binance_l2_retry_dates AS expected
        WHERE EXISTS (
          SELECT 1
          FROM polymarket.backfill_jobs AS retry
          WHERE retry.ingester_key = '${INGESTER_KEY}'
            AND retry.idempotency_key = format(
              'binance-btcusdt-l2:%s:${MATERIALIZATION_CONTRACT}:retry-${RETRY_GENERATION}',
              to_char(expected.source_date, 'YYYY-MM-DD')
            )
        );

        IF existing_retry_dates IS NOT NULL THEN
          RAISE EXCEPTION
            'refusing to duplicate Binance L2 retry generation ${RETRY_GENERATION}; jobs already exist for: %',
            existing_retry_dates;
        END IF;

        SELECT string_agg(expected.source_date::text, ', ' ORDER BY expected.source_date)
        INTO completed_artifact_dates
        FROM binance_l2_retry_dates AS expected
        WHERE EXISTS (
          SELECT 1
          FROM polymarket.backfill_artifacts AS artifact
          WHERE artifact.ingester_key = '${INGESTER_KEY}'
            AND artifact.provider = 'cryptohftdata'
            AND artifact.logical_key = format(
              'cryptohftdata:binance-futures:BTCUSDT:l2-day:${MATERIALIZATION_CONTRACT}:%s',
              to_char(expected.source_date, 'YYYY-MM-DD')
            )
            AND artifact.status = 'completed'
        );

        IF completed_artifact_dates IS NOT NULL THEN
          RAISE EXCEPTION
            'refusing to retry completed Binance L2 artifacts for: %',
            completed_artifact_dates;
        END IF;

        SELECT string_agg(expected.source_date::text, ', ' ORDER BY expected.source_date)
        INTO missing_failed_artifact_dates
        FROM binance_l2_retry_dates AS expected
        WHERE NOT EXISTS (
          SELECT 1
          FROM polymarket.backfill_jobs AS prior
          JOIN polymarket.backfill_artifacts AS artifact
            ON artifact.job_id = prior.job_id
          WHERE prior.ingester_key = '${INGESTER_KEY}'
            AND prior.status = 'failed'
            AND prior.attempt = 3
            AND prior.max_attempts = 3
            AND prior.range_start = (
              expected.source_date::timestamp AT TIME ZONE 'UTC'
            )
            AND prior.range_end = (
              (expected.source_date + 1)::timestamp AT TIME ZONE 'UTC'
            )
            AND prior.idempotency_key = format(
              'binance-btcusdt-l2:%s:${MATERIALIZATION_CONTRACT}:retry-2',
              to_char(expected.source_date, 'YYYY-MM-DD')
            )
            AND artifact.ingester_key = '${INGESTER_KEY}'
            AND artifact.provider = 'cryptohftdata'
            AND artifact.source_date = expected.source_date
            AND artifact.logical_key = format(
              'cryptohftdata:binance-futures:BTCUSDT:l2-day:${MATERIALIZATION_CONTRACT}:%s',
              to_char(expected.source_date, 'YYYY-MM-DD')
            )
            AND artifact.status = 'failed'
        );

        IF missing_failed_artifact_dates IS NOT NULL THEN
          RAISE EXCEPTION
            'refusing to queue Binance L2 retries; failed retry-2 artifact ownership is missing or changed for: %',
            missing_failed_artifact_dates;
        END IF;

        SELECT count(*)::integer
        INTO failed_artifact_count
        FROM binance_l2_retry_dates AS expected
        JOIN polymarket.backfill_jobs AS prior
          ON prior.ingester_key = '${INGESTER_KEY}'
         AND prior.status = 'failed'
         AND prior.attempt = 3
         AND prior.max_attempts = 3
         AND prior.range_start = (
           expected.source_date::timestamp AT TIME ZONE 'UTC'
         )
         AND prior.range_end = (
           (expected.source_date + 1)::timestamp AT TIME ZONE 'UTC'
         )
         AND prior.idempotency_key = format(
           'binance-btcusdt-l2:%s:${MATERIALIZATION_CONTRACT}:retry-2',
           to_char(expected.source_date, 'YYYY-MM-DD')
         )
        JOIN polymarket.backfill_artifacts AS artifact
          ON artifact.job_id = prior.job_id
         AND artifact.ingester_key = '${INGESTER_KEY}'
         AND artifact.provider = 'cryptohftdata'
         AND artifact.source_date = expected.source_date
         AND artifact.logical_key = format(
           'cryptohftdata:binance-futures:BTCUSDT:l2-day:${MATERIALIZATION_CONTRACT}:%s',
           to_char(expected.source_date, 'YYYY-MM-DD')
         )
         AND artifact.status = 'failed';

        IF failed_artifact_count <> 11 THEN
          RAISE EXCEPTION
            'expected exactly 11 failed Binance L2 artifacts owned by retry-2 jobs; found %',
            failed_artifact_count;
        END IF;

        SELECT string_agg(expected.source_date::text, ', ' ORDER BY expected.source_date)
        INTO published_feature_dates
        FROM binance_l2_retry_dates AS expected
        WHERE EXISTS (
          SELECT 1
          FROM polymarket.backfill_artifacts AS artifact
          JOIN polymarket.binance_btcusdt_l2_one_second_features AS feature
            ON feature.artifact_id = artifact.artifact_id
          WHERE artifact.ingester_key = '${INGESTER_KEY}'
            AND artifact.provider = 'cryptohftdata'
            AND artifact.source_date = expected.source_date
            AND artifact.logical_key = format(
              'cryptohftdata:binance-futures:BTCUSDT:l2-day:${MATERIALIZATION_CONTRACT}:%s',
              to_char(expected.source_date, 'YYYY-MM-DD')
            )
          LIMIT 1
        );

        IF published_feature_dates IS NOT NULL THEN
          RAISE EXCEPTION
            'refusing to retry Binance L2 artifacts with published feature rows for: %',
            published_feature_dates;
        END IF;

        SELECT string_agg(expected.source_date::text, ', ' ORDER BY expected.source_date)
        INTO staging_feature_dates
        FROM binance_l2_retry_dates AS expected
        WHERE EXISTS (
          SELECT 1
          FROM polymarket.backfill_artifacts AS artifact
          JOIN polymarket.binance_btcusdt_l2_one_second_features_staging AS feature
            ON feature.artifact_id = artifact.artifact_id
          WHERE artifact.ingester_key = '${INGESTER_KEY}'
            AND artifact.provider = 'cryptohftdata'
            AND artifact.source_date = expected.source_date
            AND artifact.logical_key = format(
              'cryptohftdata:binance-futures:BTCUSDT:l2-day:${MATERIALIZATION_CONTRACT}:%s',
              to_char(expected.source_date, 'YYYY-MM-DD')
            )
          LIMIT 1
        );

        IF staging_feature_dates IS NOT NULL THEN
          RAISE EXCEPTION
            'refusing to retry Binance L2 artifacts with staging feature rows for: %',
            staging_feature_dates;
        END IF;
      END;
      $$;

      WITH prior_jobs AS (
        SELECT prior.*, expected.source_date
        FROM binance_l2_retry_dates AS expected
        JOIN polymarket.backfill_jobs AS prior
          ON prior.ingester_key = '${INGESTER_KEY}'
         AND prior.request_version = 1
         AND prior.status = 'failed'
         AND prior.attempt = 3
         AND prior.max_attempts = 3
         AND prior.range_start = (
           expected.source_date::timestamp AT TIME ZONE 'UTC'
         )
         AND prior.range_end = (
           (expected.source_date + 1)::timestamp AT TIME ZONE 'UTC'
         )
         AND prior.idempotency_key = format(
           'binance-btcusdt-l2:%s:${MATERIALIZATION_CONTRACT}:retry-2',
           to_char(expected.source_date, 'YYYY-MM-DD')
         )
      ),
      queued_jobs AS (
        INSERT INTO polymarket.backfill_jobs (
          job_id,
          ingester_key,
          request_version,
          status,
          range_start,
          range_end,
          idempotency_key,
          request,
          progress,
          checkpoint,
          summary,
          attempt,
          max_attempts,
          next_attempt_at,
          requested_at,
          updated_at
        )
        SELECT
          gen_random_uuid(),
          prior.ingester_key,
          prior.request_version,
          'queued',
          prior.range_start,
          prior.range_end,
          format(
            'binance-btcusdt-l2:%s:${MATERIALIZATION_CONTRACT}:retry-${RETRY_GENERATION}',
            to_char(prior.source_date, 'YYYY-MM-DD')
          ),
          prior.request || jsonb_build_object(
            'idempotency_key',
            format(
              'binance-btcusdt-l2:%s:${MATERIALIZATION_CONTRACT}:retry-${RETRY_GENERATION}',
              to_char(prior.source_date, 'YYYY-MM-DD')
            )
          ),
          jsonb_build_object(
            'expected_work_units', 1,
            'completed_work_units', 0,
            'records_read', 0,
            'records_committed', 0,
            'bytes_downloaded', 0,
            'details', '{}'::jsonb
          ),
          '{}'::jsonb,
          '{}'::jsonb,
          0,
          3,
          now(),
          now(),
          now()
        FROM prior_jobs AS prior
        RETURNING job_id, ingester_key, request_version, range_start, range_end
      )
      INSERT INTO polymarket.backfill_job_events (
        event_id,
        job_id,
        timestamp_utc,
        level,
        message,
        metadata
      )
      SELECT
        gen_random_uuid(),
        queued.job_id,
        now(),
        'info',
        'backfill job queued',
        jsonb_build_object(
          'ingester', queued.ingester_key,
          'request_version', queued.request_version,
          'range_start', queued.range_start,
          'range_end', queued.range_end,
          'retry_generation', ${RETRY_GENERATION},
          'prior_job_id', prior.job_id,
          'reason', 'cryptohft_hour_boundary_validation_repair'
        )
      FROM queued_jobs AS queued
      JOIN polymarket.backfill_jobs AS prior
        ON prior.ingester_key = queued.ingester_key
       AND prior.range_start = queued.range_start
       AND prior.range_end = queued.range_end
       AND prior.idempotency_key = format(
         'binance-btcusdt-l2:%s:${MATERIALIZATION_CONTRACT}:retry-2',
         to_char(
           (queued.range_start AT TIME ZONE 'UTC')::date,
           'YYYY-MM-DD'
         )
       );

      DO $$
      DECLARE
        queued_count integer;
        queued_event_count integer;
      BEGIN
        SELECT count(*)::integer
        INTO queued_count
        FROM polymarket.backfill_jobs AS retry
        JOIN binance_l2_retry_dates AS expected
          ON retry.range_start = (
            expected.source_date::timestamp AT TIME ZONE 'UTC'
          )
         AND retry.range_end = (
           (expected.source_date + 1)::timestamp AT TIME ZONE 'UTC'
         )
        WHERE retry.ingester_key = '${INGESTER_KEY}'
          AND retry.status = 'queued'
          AND retry.attempt = 0
          AND retry.max_attempts = 3
          AND retry.idempotency_key = format(
            'binance-btcusdt-l2:%s:${MATERIALIZATION_CONTRACT}:retry-${RETRY_GENERATION}',
            to_char(expected.source_date, 'YYYY-MM-DD')
          );

        IF queued_count <> 11 THEN
          RAISE EXCEPTION
            'expected to queue 11 Binance L2 retry-generation-${RETRY_GENERATION} jobs, queued %',
            queued_count;
        END IF;

        SELECT count(*)::integer
        INTO queued_event_count
        FROM polymarket.backfill_job_events AS event
        JOIN polymarket.backfill_jobs AS retry
          ON retry.job_id = event.job_id
        JOIN binance_l2_retry_dates AS expected
          ON retry.range_start = (
            expected.source_date::timestamp AT TIME ZONE 'UTC'
          )
         AND retry.range_end = (
           (expected.source_date + 1)::timestamp AT TIME ZONE 'UTC'
         )
        WHERE retry.ingester_key = '${INGESTER_KEY}'
          AND retry.idempotency_key = format(
            'binance-btcusdt-l2:%s:${MATERIALIZATION_CONTRACT}:retry-${RETRY_GENERATION}',
            to_char(expected.source_date, 'YYYY-MM-DD')
          )
          AND event.message = 'backfill job queued'
          AND event.metadata ->> 'retry_generation' = '${RETRY_GENERATION}';

        IF queued_event_count <> 11 THEN
          RAISE EXCEPTION
            'expected 11 Binance L2 retry-generation-${RETRY_GENERATION} lineage events, recorded %',
            queued_event_count;
        END IF;
      END;
      $$;
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    // Retry jobs and their events are operational lineage and are intentionally retained.
  }
}
