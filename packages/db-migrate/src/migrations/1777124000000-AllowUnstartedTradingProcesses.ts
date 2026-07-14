import { MigrationInterface, QueryRunner } from 'typeorm';

export class AllowUnstartedTradingProcesses1777124000000 implements MigrationInterface {
  name = 'AllowUnstartedTradingProcesses1777124000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE polymarket.trading_processes
        ALTER COLUMN started_at DROP NOT NULL;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      UPDATE polymarket.trading_processes
      SET started_at = created_at
      WHERE started_at IS NULL;

      ALTER TABLE polymarket.trading_processes
        ALTER COLUMN started_at SET NOT NULL;
    `);
  }
}
