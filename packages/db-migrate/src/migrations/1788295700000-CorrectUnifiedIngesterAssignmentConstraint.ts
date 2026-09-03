import { MigrationInterface, QueryRunner } from 'typeorm';

export class CorrectUnifiedIngesterAssignmentConstraint1788295700000
  implements MigrationInterface
{
  name = 'CorrectUnifiedIngesterAssignmentConstraint1788295700000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ingester.backfill_jobs
        DROP CONSTRAINT chk_ingester_backfill_assignment;

      ALTER TABLE ingester.backfill_jobs
        ADD CONSTRAINT chk_ingester_backfill_assignment
        CHECK (
          (job_kind = 'shard'
            AND status IN ('running','cancel_requested')
            AND assigned_worker_id IS NOT NULL
            AND lease_token IS NOT NULL
            AND lease_expires_at IS NOT NULL
            AND heartbeat_at IS NOT NULL)
          OR
          (job_kind = 'request')
          OR
          (status NOT IN ('running','cancel_requested'))
          OR
          legacy_source IS NOT NULL
        );
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ingester.backfill_jobs
        DROP CONSTRAINT chk_ingester_backfill_assignment;

      ALTER TABLE ingester.backfill_jobs
        ADD CONSTRAINT chk_ingester_backfill_assignment
        CHECK (
          (status IN ('running','cancel_requested')
            AND assigned_worker_id IS NOT NULL
            AND lease_token IS NOT NULL
            AND lease_expires_at IS NOT NULL
            AND heartbeat_at IS NOT NULL)
          OR
          (status NOT IN ('running','cancel_requested'))
          OR
          legacy_source IS NOT NULL
        );
    `);
  }
}
