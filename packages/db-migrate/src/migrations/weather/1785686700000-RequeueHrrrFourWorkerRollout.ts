import { MigrationInterface, QueryRunner } from 'typeorm';

const PROCESS_ID = 'd48e2b47-18df-4c8c-95b9-0f12e6ca7d41';

export class RequeueHrrrFourWorkerRollout1785686700000 implements MigrationInterface {
  name = 'RequeueHrrrFourWorkerRollout1785686700000';

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
            'worker_drain_policy', 'four_worker_empty_inventory_rollout_v1'
          ),
          updated_at = now()
      WHERE job_id IN (
        'd210e13f-90df-48f9-bf23-1de9f8f0df43'::uuid,
        '36e32e88-ab56-4275-a6d9-1169db69c451'::uuid,
        '46c27d7e-0a41-4a8d-a144-b10c15bf072b'::uuid,
        'dc6295bc-2066-4340-b3c5-9da1651981fc'::uuid
      )
        AND ingester_key = 'hrrr_point_forecasts'
        AND status = 'running'
        AND worker_id IN (
          'temperature-worker-1',
          'temperature-worker-2',
          'temperature-worker-3',
          'temperature-worker-4'
        )
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
      WHERE job_id IN (
        'd210e13f-90df-48f9-bf23-1de9f8f0df43'::uuid,
        '36e32e88-ab56-4275-a6d9-1169db69c451'::uuid,
        '46c27d7e-0a41-4a8d-a144-b10c15bf072b'::uuid,
        'dc6295bc-2066-4340-b3c5-9da1651981fc'::uuid
      )
        AND request ->> 'process_id' = '${PROCESS_ID}';
    `);
  }
}
