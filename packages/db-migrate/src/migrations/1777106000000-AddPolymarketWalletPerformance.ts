import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolymarketWalletPerformance1777106000000 implements MigrationInterface {
  name = 'AddPolymarketWalletPerformance1777106000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS polymarket;`);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.wallet_performance (
        proxy_wallet text PRIMARY KEY,
        sample_updated_at timestamptz NOT NULL DEFAULT now(),
        realized_pnl_usd numeric(30,10) NOT NULL DEFAULT 0,
        total_bought_usd numeric(30,10) NOT NULL DEFAULT 0,
        roi numeric(30,10) NOT NULL DEFAULT 0,
        closed_positions integer NOT NULL DEFAULT 0,
        winning_positions integer NOT NULL DEFAULT 0,
        win_rate numeric(18,8) NOT NULL DEFAULT 0,
        rank_score numeric(30,10) NOT NULL DEFAULT 0,
        raw_payload jsonb NOT NULL DEFAULT '[]'::jsonb,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_poly_wallet_performance_counts CHECK (
          closed_positions >= 0
          AND winning_positions >= 0
          AND winning_positions <= closed_positions
        )
      );
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_performance_rank
        ON polymarket.wallet_performance (rank_score DESC, realized_pnl_usd DESC, roi DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_performance_thresholds
        ON polymarket.wallet_performance (realized_pnl_usd DESC, roi DESC, closed_positions DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_performance_updated
        ON polymarket.wallet_performance (sample_updated_at DESC);
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.wallet_performance;`);
  }
}
