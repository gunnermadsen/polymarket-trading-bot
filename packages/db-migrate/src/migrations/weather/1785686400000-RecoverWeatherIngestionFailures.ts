import { MigrationInterface, QueryRunner } from 'typeorm';

const PROCESS_ID = 'd48e2b47-18df-4c8c-95b9-0f12e6ca7d41';
const HRRR_JOB_ID = 'd210e13f-90df-48f9-bf23-1de9f8f0df43';

export class RecoverWeatherIngestionFailures1785686400000 implements MigrationInterface {
  name = 'RecoverWeatherIngestionFailures1785686400000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      UPDATE weather.ingestion_jobs
      SET status = 'queued',
          attempt = 0,
          next_attempt_at = now(),
          worker_id = NULL,
          lease_token = NULL,
          lease_expires_at = NULL,
          heartbeat_at = NULL,
          completed_at = NULL,
          request = request || jsonb_build_object(
            'process_id', '${PROCESS_ID}',
            'recovery_policy', 'mirror_range_rotation_v1',
            'recovered_error', error
          ),
          error = NULL,
          updated_at = now()
      WHERE job_id = '${HRRR_JOB_ID}'::uuid
        AND ingester_key = 'hrrr_point_forecasts'
        AND range_start = '2020-02-01T00:00:00Z'::timestamptz
        AND range_end = '2020-03-01T00:00:00Z'::timestamptz
        AND status = 'failed'
        AND EXISTS (
          SELECT 1
          FROM polymarket.trading_processes process
          WHERE process.process_id = '${PROCESS_ID}'::uuid
        );
    `);

    await queryRunner.query(`
      WITH replacements (failed_job_id, completed_job_id) AS (
        VALUES
          (
            'd0a0829b-c562-4c4e-8fb7-0adacc3b33ae'::uuid,
            'e5aa549f-2d45-4a95-86fe-df58f06ea1aa'::uuid
          ),
          (
            'ce52d89d-72a5-4f69-93af-3619f2f6a56b'::uuid,
            '1b0abd7b-e42b-4e99-8fb6-3e29f0e3faab'::uuid
          ),
          (
            '9cc78855-da44-4d00-aad0-21587329a8e6'::uuid,
            '3baefd31-3c4b-48fd-82b4-13d3adc23a09'::uuid
          )
      )
      UPDATE weather.ingestion_jobs failed
      SET status = 'cancelled',
          completed_at = COALESCE(failed.completed_at, now()),
          request = failed.request || jsonb_build_object(
            'process_id', '${PROCESS_ID}',
            'reconciliation_policy', 'completed_replacement_v1',
            'superseded_by_job_id', replacement.job_id::text
          ),
          updated_at = now()
      FROM replacements mapping
      JOIN weather.ingestion_jobs replacement
        ON replacement.job_id = mapping.completed_job_id
       AND replacement.status = 'completed'
      WHERE failed.job_id = mapping.failed_job_id
        AND failed.status = 'failed'
        AND failed.ingester_key = replacement.ingester_key
        AND failed.range_start = replacement.range_start
        AND failed.range_end = replacement.range_end
        AND EXISTS (
          SELECT 1
          FROM polymarket.trading_processes process
          WHERE process.process_id = '${PROCESS_ID}'::uuid
        );
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      UPDATE weather.ingestion_jobs
      SET request = request - 'recovery_policy' - 'recovered_error',
          updated_at = now()
      WHERE job_id = '${HRRR_JOB_ID}'::uuid
        AND request ->> 'process_id' = '${PROCESS_ID}';
    `);

    await queryRunner.query(`
      UPDATE weather.ingestion_jobs
      SET request = request - 'reconciliation_policy' - 'superseded_by_job_id',
          updated_at = now()
      WHERE job_id IN (
        'd0a0829b-c562-4c4e-8fb7-0adacc3b33ae'::uuid,
        'ce52d89d-72a5-4f69-93af-3619f2f6a56b'::uuid,
        '9cc78855-da44-4d00-aad0-21587329a8e6'::uuid
      )
        AND request ->> 'process_id' = '${PROCESS_ID}';
    `);
  }
}
