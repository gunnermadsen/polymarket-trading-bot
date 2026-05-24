import { MigrationInterface, QueryRunner } from 'typeorm';

export class DropCopyTradeSignalWalletScore1777114000000 implements MigrationInterface {
  name = 'DropCopyTradeSignalWalletScore1777114000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      UPDATE polymarket.trade_positions
      SET metadata = metadata #- '{copy_signal,wallet_score}'
      WHERE metadata #> '{copy_signal,wallet_score}' IS NOT NULL;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.copy_trade_signals
        DROP COLUMN IF EXISTS wallet_score;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE polymarket.copy_trade_signals
        ADD COLUMN IF NOT EXISTS wallet_score numeric(18,8) NOT NULL DEFAULT 0;
    `);
  }
}
