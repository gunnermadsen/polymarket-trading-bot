import { MigrationInterface, QueryRunner } from 'typeorm';

export class PolymarketBotScannerPersistenceDrift1777103000000 implements MigrationInterface {
  name = 'PolymarketBotScannerPersistenceDrift1777103000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE polymarket.markets
      ADD COLUMN IF NOT EXISTS outcome_group_id text;
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_polymarket_markets_outcome_group
      ON polymarket.markets (outcome_group_id);
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        IF NOT EXISTS (
          SELECT 1
          FROM pg_constraint
          WHERE conname = 'uq_polymarket_positions_market_token'
        ) THEN
          ALTER TABLE polymarket.positions
          ADD CONSTRAINT uq_polymarket_positions_market_token UNIQUE (market_id, token_id);
        END IF;
      END $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE polymarket.positions
      DROP CONSTRAINT IF EXISTS uq_polymarket_positions_market_token;
    `);
    await queryRunner.query(`
      DROP INDEX IF EXISTS polymarket.idx_polymarket_markets_outcome_group;
    `);
    await queryRunner.query(`
      ALTER TABLE polymarket.markets
      DROP COLUMN IF EXISTS outcome_group_id;
    `);
  }
}
