import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolymarketExpectancyFlowCells1777118000000 implements MigrationInterface {
  name = 'AddPolymarketExpectancyFlowCells1777118000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS polymarket;`);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.expectancy_flow_cells (
        process_id uuid NOT NULL REFERENCES polymarket.trading_processes (process_id) ON DELETE CASCADE,
        score_version text NOT NULL,
        cell_key text NOT NULL,
        dimensions jsonb NOT NULL DEFAULT '{}'::jsonb,
        horizon_secs bigint NOT NULL,
        lookback_days integer NOT NULL,
        sample_count integer NOT NULL DEFAULT 0,
        winning_count integer NOT NULL DEFAULT 0,
        losing_count integer NOT NULL DEFAULT 0,
        observed_volume_usd numeric(30,10) NOT NULL DEFAULT 0,
        realized_pnl_usd numeric(30,10) NOT NULL DEFAULT 0,
        mean_price_delta numeric(18,8) NOT NULL DEFAULT 0,
        mean_return numeric(30,10) NOT NULL DEFAULT 0,
        win_rate numeric(18,8) NOT NULL DEFAULT 0,
        expectancy numeric(30,10) NOT NULL DEFAULT 0,
        confidence numeric(18,8) NOT NULL DEFAULT 0,
        sample_start timestamptz,
        sample_end timestamptz,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_poly_expectancy_flow_cells PRIMARY KEY (process_id, score_version, cell_key),
        CONSTRAINT chk_poly_expectancy_flow_cell_counts CHECK (
          sample_count >= 0
          AND winning_count >= 0
          AND losing_count >= 0
          AND winning_count + losing_count <= sample_count
        ),
        CONSTRAINT chk_poly_expectancy_flow_cell_windows CHECK (horizon_secs > 0 AND lookback_days >= 0),
        CONSTRAINT chk_poly_expectancy_flow_cell_confidence CHECK (confidence >= 0 AND confidence <= 1)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.expectancy_flow_wallet_cells (
        process_id uuid NOT NULL REFERENCES polymarket.trading_processes (process_id) ON DELETE CASCADE,
        score_version text NOT NULL,
        proxy_wallet text NOT NULL REFERENCES polymarket.wallets (proxy_wallet) ON DELETE CASCADE,
        cell_key text NOT NULL,
        dimensions jsonb NOT NULL DEFAULT '{}'::jsonb,
        horizon_secs bigint NOT NULL,
        lookback_days integer NOT NULL,
        sample_count integer NOT NULL DEFAULT 0,
        winning_count integer NOT NULL DEFAULT 0,
        losing_count integer NOT NULL DEFAULT 0,
        observed_volume_usd numeric(30,10) NOT NULL DEFAULT 0,
        realized_pnl_usd numeric(30,10) NOT NULL DEFAULT 0,
        mean_price_delta numeric(18,8) NOT NULL DEFAULT 0,
        mean_return numeric(30,10) NOT NULL DEFAULT 0,
        win_rate numeric(18,8) NOT NULL DEFAULT 0,
        expectancy numeric(30,10) NOT NULL DEFAULT 0,
        confidence numeric(18,8) NOT NULL DEFAULT 0,
        sample_start timestamptz,
        sample_end timestamptz,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_poly_expectancy_flow_wallet_cells PRIMARY KEY (process_id, score_version, proxy_wallet, cell_key),
        CONSTRAINT chk_poly_expectancy_flow_wallet_cell_counts CHECK (
          sample_count >= 0
          AND winning_count >= 0
          AND losing_count >= 0
          AND winning_count + losing_count <= sample_count
        ),
        CONSTRAINT chk_poly_expectancy_flow_wallet_cell_windows CHECK (horizon_secs > 0 AND lookback_days >= 0),
        CONSTRAINT chk_poly_expectancy_flow_wallet_cell_confidence CHECK (confidence >= 0 AND confidence <= 1)
      );
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_expectancy_flow_cells_rank
        ON polymarket.expectancy_flow_cells (process_id, score_version, confidence DESC, expectancy DESC, sample_count DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_expectancy_flow_cells_updated
        ON polymarket.expectancy_flow_cells (updated_at DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_expectancy_flow_wallet_cells_wallet
        ON polymarket.expectancy_flow_wallet_cells (process_id, score_version, proxy_wallet, confidence DESC, expectancy DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_expectancy_flow_wallet_cells_rank
        ON polymarket.expectancy_flow_wallet_cells (process_id, score_version, confidence DESC, expectancy DESC, sample_count DESC);
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_expectancy_flow_wallet_cells_rank;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_expectancy_flow_wallet_cells_wallet;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_expectancy_flow_cells_updated;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_expectancy_flow_cells_rank;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.expectancy_flow_wallet_cells;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.expectancy_flow_cells;`);
  }
}
