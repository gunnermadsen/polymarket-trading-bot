import { MigrationInterface, QueryRunner } from 'typeorm';

const PROCESS_ID = 'd48e2b47-18df-4c8c-95b9-0f12e6ca7d41';
const JANUARY_JOB_ID = 'a5a5b50e-11fa-4dfa-80dd-e6ba1b4a4627';

export class RecoverHrrrEmptyMirrorInventory1785686600000 implements MigrationInterface {
  name = 'RecoverHrrrEmptyMirrorInventory1785686600000';

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
            'recovery_policy', 'empty_mirror_inventory_rotation_v1',
            'recovered_error', error
          ),
          error = NULL,
          updated_at = now()
      WHERE job_id = '${JANUARY_JOB_ID}'::uuid
        AND ingester_key = 'hrrr_point_forecasts'
        AND range_start = '2022-01-01T00:00:00Z'::timestamptz
        AND range_end = '2022-02-01T00:00:00Z'::timestamptz
        AND status = 'failed'
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
      WHERE job_id = '${JANUARY_JOB_ID}'::uuid
        AND request ->> 'process_id' = '${PROCESS_ID}';
    `);
  }
}
