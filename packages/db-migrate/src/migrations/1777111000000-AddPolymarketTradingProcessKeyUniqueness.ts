import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolymarketTradingProcessKeyUniqueness1777111000000 implements MigrationInterface {
  name = 'AddPolymarketTradingProcessKeyUniqueness1777111000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE UNIQUE INDEX IF NOT EXISTS uq_poly_trading_processes_key
        ON polymarket.trading_processes (process_type, process_scope, process_key)
        WHERE process_key IS NOT NULL;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.uq_poly_trading_processes_key;`);
  }
}
