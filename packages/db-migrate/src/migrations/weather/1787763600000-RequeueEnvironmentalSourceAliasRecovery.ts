import { MigrationInterface, QueryRunner } from 'typeorm';

export class RequeueEnvironmentalSourceAliasRecovery1787763600000 implements MigrationInterface {
  name = 'RequeueEnvironmentalSourceAliasRecovery1787763600000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      UPDATE weather.ingestion_jobs
      SET status = 'queued',
          worker_id = NULL,
          lease_token = NULL,
          lease_expires_at = NULL,
          heartbeat_at = NULL,
          next_attempt_at = now(),
          updated_at = now(),
          error = concat_ws(E'\n', error, 'requeued after NOAA environmental source alias correction')
      WHERE status = 'running'
        AND ingester_key IN ('goes_abi_klga_features', 'hrrr_environment_features');
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    // Operational lease recovery is intentionally not reversible.
  }
}
