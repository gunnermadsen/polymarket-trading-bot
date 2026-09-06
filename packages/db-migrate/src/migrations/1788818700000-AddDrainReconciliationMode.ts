import { MigrationInterface, QueryRunner } from "typeorm";

export class AddDrainReconciliationMode1788818700000 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ingester.drain_jobs
        ADD COLUMN mode text NOT NULL DEFAULT 'drain',
        ADD CONSTRAINT ck_ingester_drain_jobs_mode
          CHECK (mode IN ('drain', 'reconcile'));
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ingester.drain_jobs
        DROP CONSTRAINT ck_ingester_drain_jobs_mode,
        DROP COLUMN mode;
    `);
  }
}
