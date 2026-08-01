import { MigrationInterface, QueryRunner } from 'typeorm';

const PROCESS_ID = 'd48e2b47-18df-4c8c-95b9-0f12e6ca7d41';
const MARCH_JOB_ID = '7d10326b-a1f8-441a-b07b-fabba81fd247';

export class RequeueHrrrArchiveGap1785602400000 implements MigrationInterface {
  name = 'RequeueHrrrArchiveGap1785602400000';

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
          error = NULL,
          request = request || jsonb_build_object(
            'process_id', '${PROCESS_ID}',
            'missing_archive_policy', 'record_and_continue_v1'
          ),
          updated_at = now()
      WHERE job_id = '${MARCH_JOB_ID}'::uuid
        AND ingester_key = 'hrrr_point_forecasts'
        AND range_start = '2019-03-01T00:00:00Z'::timestamptz
        AND range_end = '2019-04-01T00:00:00Z'::timestamptz
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
      SET request = request - 'missing_archive_policy',
          updated_at = now()
      WHERE job_id = '${MARCH_JOB_ID}'::uuid
        AND request ->> 'process_id' = '${PROCESS_ID}';
    `);
  }
}
