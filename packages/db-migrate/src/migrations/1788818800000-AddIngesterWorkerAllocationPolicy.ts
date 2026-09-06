import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddIngesterWorkerAllocationPolicy1788818800000
  implements MigrationInterface
{
  name = 'AddIngesterWorkerAllocationPolicy1788818800000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ingester.workers
        ADD COLUMN capacity_units smallint NOT NULL DEFAULT 4,
        ADD COLUMN realtime_slot_limit smallint NOT NULL DEFAULT 1,
        ADD COLUMN allocation_contract_version integer NOT NULL DEFAULT 1,
        ADD CONSTRAINT chk_ingester_worker_allocation_capacity
          CHECK (capacity_units BETWEEN 1 AND 32),
        ADD CONSTRAINT chk_ingester_worker_realtime_slots
          CHECK (realtime_slot_limit = 1),
        ADD CONSTRAINT chk_ingester_worker_allocation_contract
          CHECK (allocation_contract_version > 0);

      ALTER TABLE ingester.backfill_jobs
        ADD COLUMN allocation_units smallint NOT NULL DEFAULT 2,
        ADD CONSTRAINT chk_ingester_backfill_allocation_units
          CHECK (allocation_units BETWEEN 1 AND 32);
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ingester.backfill_jobs
        DROP CONSTRAINT chk_ingester_backfill_allocation_units,
        DROP COLUMN allocation_units;

      ALTER TABLE ingester.workers
        DROP CONSTRAINT chk_ingester_worker_allocation_contract,
        DROP CONSTRAINT chk_ingester_worker_realtime_slots,
        DROP CONSTRAINT chk_ingester_worker_allocation_capacity,
        DROP COLUMN allocation_contract_version,
        DROP COLUMN realtime_slot_limit,
        DROP COLUMN capacity_units;
    `);
  }
}
