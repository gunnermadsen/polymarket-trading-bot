import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddTemperatureSpatialGradients1787767200000 implements MigrationInterface {
  name = 'AddTemperatureSpatialGradients1787767200000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE weather.goes_abi_features
        ADD COLUMN infrared_north_south_gradient_k double precision,
        ADD COLUMN infrared_east_west_gradient_k double precision,
        ADD COLUMN cloudy_north_south_gradient double precision,
        ADD COLUMN cloudy_east_west_gradient double precision;

      ALTER TABLE weather.hrrr_environment_features
        ADD COLUMN temperature_2m_north_south_gradient_k double precision,
        ADD COLUMN temperature_2m_east_west_gradient_k double precision;

      UPDATE weather.ingestion_jobs
      SET status = 'cancelled',
          worker_id = NULL,
          lease_token = NULL,
          lease_expires_at = NULL,
          heartbeat_at = now(),
          completed_at = now(),
          updated_at = now(),
          error = concat_ws(E'\n', error, 'v1 preserved; superseded by explicit spatial-gradient schema v2')
      WHERE status IN ('queued', 'running', 'cancel_requested')
        AND ingester_key IN ('goes_abi_klga_features', 'hrrr_environment_features')
        AND request->>'feature_schema_version' IN (
          'goes-abi-klga-v1', 'hrrr-environment-klga-v1'
        );
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE weather.hrrr_environment_features
        DROP COLUMN IF EXISTS temperature_2m_east_west_gradient_k,
        DROP COLUMN IF EXISTS temperature_2m_north_south_gradient_k;
      ALTER TABLE weather.goes_abi_features
        DROP COLUMN IF EXISTS cloudy_east_west_gradient,
        DROP COLUMN IF EXISTS cloudy_north_south_gradient,
        DROP COLUMN IF EXISTS infrared_east_west_gradient_k,
        DROP COLUMN IF EXISTS infrared_north_south_gradient_k;
    `);
  }
}
