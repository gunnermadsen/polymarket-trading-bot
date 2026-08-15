import { MigrationInterface, QueryRunner } from 'typeorm';

export class AllowMultipleCaptureArtifactsPerWindow1786381050000
  implements MigrationInterface
{
  name = 'AllowMultipleCaptureArtifactsPerWindow1786381050000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ingester.capture_artifacts
        DROP CONSTRAINT uq_ingester_capture_artifact_window;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
          FROM ingester.capture_artifacts
          GROUP BY
            strategy_key,
            capture_window_start,
            capture_window_end,
            profile_generation
          HAVING count(*) > 1
          LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to restore capture-artifact window uniqueness while duplicate window generations exist';
        END IF;
      END $$;

      ALTER TABLE ingester.capture_artifacts
        ADD CONSTRAINT uq_ingester_capture_artifact_window UNIQUE (
          strategy_key,
          capture_window_start,
          capture_window_end,
          profile_generation
        );
    `);
  }
}
