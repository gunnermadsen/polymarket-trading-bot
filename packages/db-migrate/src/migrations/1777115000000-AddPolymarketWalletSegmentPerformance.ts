import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolymarketWalletSegmentPerformance1777115000000 implements MigrationInterface {
  name = 'AddPolymarketWalletSegmentPerformance1777115000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS polymarket;`);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.wallet_segment_performance (
        proxy_wallet text NOT NULL REFERENCES polymarket.wallets (proxy_wallet) ON DELETE CASCADE,
        segment_key text NOT NULL,
        score_version text NOT NULL,
        classifier_version text NOT NULL,
        score numeric(18,8) NOT NULL DEFAULT 0,
        confidence numeric(18,8) NOT NULL DEFAULT 0,
        closed_positions integer NOT NULL DEFAULT 0,
        winning_positions integer NOT NULL DEFAULT 0,
        losing_positions integer NOT NULL DEFAULT 0,
        win_rate numeric(18,8) NOT NULL DEFAULT 0,
        realized_pnl_usd numeric(30,10) NOT NULL DEFAULT 0,
        total_bought_usd numeric(30,10) NOT NULL DEFAULT 0,
        roi numeric(30,10) NOT NULL DEFAULT 0,
        observed_trade_count integer NOT NULL DEFAULT 0,
        observed_volume_usd numeric(30,10) NOT NULL DEFAULT 0,
        sample_start timestamptz,
        sample_end timestamptz,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_poly_wallet_segment_perf PRIMARY KEY (proxy_wallet, segment_key, score_version),
        CONSTRAINT chk_poly_wallet_segment_perf_score CHECK (score >= 0 AND score <= 100),
        CONSTRAINT chk_poly_wallet_segment_perf_confidence CHECK (confidence >= 0 AND confidence <= 1),
        CONSTRAINT chk_poly_wallet_segment_perf_counts CHECK (
          closed_positions >= 0
          AND winning_positions >= 0
          AND losing_positions >= 0
          AND winning_positions + losing_positions <= closed_positions
          AND observed_trade_count >= 0
        )
      );
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_segment_perf_segment_score
        ON polymarket.wallet_segment_performance (segment_key, score DESC, updated_at DESC);
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_segment_perf_wallet
        ON polymarket.wallet_segment_performance (proxy_wallet, updated_at DESC);
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_wallet_segment_perf_wallet;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_wallet_segment_perf_segment_score;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.wallet_segment_performance;`);
  }
}
