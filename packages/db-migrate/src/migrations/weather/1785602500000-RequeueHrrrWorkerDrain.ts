import { MigrationInterface, QueryRunner } from 'typeorm';

const PROCESS_ID = 'd48e2b47-18df-4c8c-95b9-0f12e6ca7d41';
const OCTOBER_JOB_ID = 'c58fa4ce-b1b6-48d0-99dc-4c15760568b1';
const DECEMBER_JOB_ID = '81cdbd78-20a3-4bf4-9aaa-bcd22486b677';

export class RequeueHrrrWorkerDrain1785602500000 implements MigrationInterface {
  name = 'RequeueHrrrWorkerDrain1785602500000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      UPDATE weather.ingestion_jobs
      SET status = 'queued',
          next_attempt_at = now(),
          worker_id = NULL,
          lease_token = NULL,
          lease_expires_at = NULL,
          heartbeat_at = NULL,
          error = NULL,
          request = request || jsonb_build_object(
            'process_id', '${PROCESS_ID}',
            'worker_drain_policy', 'progress_preserving_requeue_v1'
          ),
          updated_at = now()
      WHERE job_id IN ('${OCTOBER_JOB_ID}'::uuid, '${DECEMBER_JOB_ID}'::uuid)
        AND ingester_key = 'hrrr_point_forecasts'
        AND status = 'running'
        AND worker_id IN ('temperature-worker-1', 'temperature-worker-2')
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
      SET request = request - 'worker_drain_policy',
          updated_at = now()
      WHERE job_id IN ('${OCTOBER_JOB_ID}'::uuid, '${DECEMBER_JOB_ID}'::uuid)
        AND request ->> 'process_id' = '${PROCESS_ID}';
    `);
  }
}
