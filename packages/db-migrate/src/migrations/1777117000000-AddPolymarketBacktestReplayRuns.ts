import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolymarketBacktestReplayRuns1777117000000 implements MigrationInterface {
  name = 'AddPolymarketBacktestReplayRuns1777117000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS polymarket;`);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.backtest_runs (
        backtest_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        status text NOT NULL DEFAULT 'queued',
        range_start timestamptz NOT NULL,
        range_end timestamptz NOT NULL,
        warmup_start timestamptz NOT NULL,
        lookback_days integer NOT NULL,
        warmup_days integer NOT NULL,
        source_process_ids uuid[] NOT NULL DEFAULT '{}'::uuid[],
        backtest_process_ids uuid[] NOT NULL DEFAULT '{}'::uuid[],
        request jsonb NOT NULL DEFAULT '{}'::jsonb,
        summary jsonb NOT NULL DEFAULT '{}'::jsonb,
        error text,
        started_at timestamptz,
        completed_at timestamptz,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_poly_backtest_runs_status CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
        CONSTRAINT chk_poly_backtest_runs_windows CHECK (warmup_start <= range_start AND range_start <= range_end),
        CONSTRAINT chk_poly_backtest_runs_days CHECK (lookback_days >= 0 AND warmup_days >= 0)
      );
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_backtest_runs_status_created
        ON polymarket.backtest_runs (status, created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_backtest_runs_range
        ON polymarket.backtest_runs (range_start, range_end);
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_trades_replay_range
        ON polymarket.wallet_trades (timestamp_utc, cash_value DESC);
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_wallet_trades_replay_range;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_backtest_runs_range;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_backtest_runs_status_created;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.backtest_runs;`);
  }
}
