import { MigrationInterface, QueryRunner } from 'typeorm';

export class IndexRecognizedManualLiveExits1786929930000 implements MigrationInterface {
  name = 'IndexRecognizedManualLiveExits1786929930000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_account_trades_recognized_exit_order
        ON polymarket.account_trades (linked_order_id, token_id)
        INCLUDE (size, applied_exit_size, timestamp_utc)
        WHERE linked_order_id IS NOT NULL
          AND side = 'sell'
          AND applied_exit_size > 0;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DROP INDEX IF EXISTS polymarket.idx_poly_account_trades_recognized_exit_order;
    `);
  }
}
