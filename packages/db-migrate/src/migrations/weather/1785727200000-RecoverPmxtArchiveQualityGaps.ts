import { MigrationInterface, QueryRunner } from 'typeorm';

const PROCESS_ID = 'd48e2b47-18df-4c8c-95b9-0f12e6ca7d41';
const PMXT_JOB_IDS = [
  '0b706d4d-29de-4ea7-9068-4a211c679a8b',
  '5c979e41-4562-41c0-8fe1-f05918dd8d23',
  'ad2a885c-02ee-4b35-ace7-4db0600cbca1',
] as const;

export class RecoverPmxtArchiveQualityGaps1785727200000 implements MigrationInterface {
  name = 'RecoverPmxtArchiveQualityGaps1785727200000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      UPDATE weather.ingestion_jobs
      SET status = 'queued',
          attempt = 0,
          max_attempts = 6,
          next_attempt_at = now(),
          worker_id = NULL,
          lease_token = NULL,
          lease_expires_at = NULL,
          heartbeat_at = NULL,
          completed_at = NULL,
          request = request || jsonb_build_object(
            'process_id', '${PROCESS_ID}',
            'recovery_policy', 'pmxt_archive_integrity_v2',
            'recovered_error', error
          ),
          error = NULL,
          updated_at = now()
      WHERE job_id = ANY(ARRAY[${PMXT_JOB_IDS.map((jobId) => `'${jobId}'::uuid`).join(', ')}])
        AND ingester_key = 'pmxt_temperature_execution'
        AND status IN ('queued', 'running', 'failed')
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
      WHERE job_id = ANY(ARRAY[${PMXT_JOB_IDS.map((jobId) => `'${jobId}'::uuid`).join(', ')}])
        AND request ->> 'process_id' = '${PROCESS_ID}';
    `);
  }
}
