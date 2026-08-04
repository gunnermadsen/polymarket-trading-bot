import { MigrationInterface, QueryRunner } from 'typeorm';

const PROCESS_ID = 'd48e2b47-18df-4c8c-95b9-0f12e6ca7d41';
const MAY_JOB_ID = '13aeca18-c74a-4140-9edf-a6cdc4623448';
const JUNE_JOB_ID = '36e32e88-ab56-4275-a6d9-1169db69c451';

export class RequeueHrrrMirrorRecoveryWorkerDrain1785686500000
  implements MigrationInterface
{
  name = 'RequeueHrrrMirrorRecoveryWorkerDrain1785686500000';

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
            'worker_drain_policy', 'mirror_recovery_deploy_v1'
          ),
          updated_at = now()
      WHERE job_id IN ('${MAY_JOB_ID}'::uuid, '${JUNE_JOB_ID}'::uuid)
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
      WHERE job_id IN ('${MAY_JOB_ID}'::uuid, '${JUNE_JOB_ID}'::uuid)
        AND request ->> 'process_id' = '${PROCESS_ID}';
    `);
  }
}
