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
