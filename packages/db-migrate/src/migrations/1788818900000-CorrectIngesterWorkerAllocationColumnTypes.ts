import { MigrationInterface, QueryRunner } from 'typeorm';

export class CorrectIngesterWorkerAllocationColumnTypes1788818900000
  implements MigrationInterface
{
  name = 'CorrectIngesterWorkerAllocationColumnTypes1788818900000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ingester.workers
        ALTER COLUMN capacity_units TYPE integer,
        ALTER COLUMN realtime_slot_limit TYPE integer;

      ALTER TABLE ingester.backfill_jobs
        ALTER COLUMN allocation_units TYPE integer;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ingester.backfill_jobs
        ALTER COLUMN allocation_units TYPE smallint;

      ALTER TABLE ingester.workers
        ALTER COLUMN realtime_slot_limit TYPE smallint,
        ALTER COLUMN capacity_units TYPE smallint;
    `);
  }
}
