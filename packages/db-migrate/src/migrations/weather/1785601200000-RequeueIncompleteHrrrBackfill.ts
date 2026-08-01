import { MigrationInterface, QueryRunner } from 'typeorm';

export class RequeueIncompleteHrrrBackfill1785601200000 implements MigrationInterface {
  name = 'RequeueIncompleteHrrrBackfill1785601200000';

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
          error = NULL,
          request = request || '{"resilience_policy":"decision_resume_v1"}'::jsonb,
          updated_at = now()
      WHERE ingester_key = 'hrrr_point_forecasts'
        AND status IN ('queued', 'running', 'failed');
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      UPDATE weather.ingestion_jobs
      SET max_attempts = 3,
          request = request - 'resilience_policy',
          updated_at = now()
      WHERE ingester_key = 'hrrr_point_forecasts'
        AND max_attempts = 6;
    `);
  }
}
