import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolymarketWhalePersistenceDrift1777105000000 implements MigrationInterface {
  name = 'AddPolymarketWhalePersistenceDrift1777105000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;`);
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS polymarket;`);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.copy_trade_backtest_runs (
        backtest_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id uuid REFERENCES polymarket.backfill_jobs (job_id) ON DELETE SET NULL,
        status text NOT NULL DEFAULT 'running',
        score_version text NOT NULL,
        strategy_name text NOT NULL DEFAULT 'whale_follow_v1',
        range_start timestamptz,
        range_end timestamptz,
        config jsonb NOT NULL DEFAULT '{}'::jsonb,
        started_at timestamptz NOT NULL DEFAULT now(),
        completed_at timestamptz,
        error text,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_poly_copy_trade_backtest_runs_status CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.copy_trade_backtest_results (
        result_id uuid NOT NULL DEFAULT gen_random_uuid(),
        backtest_id uuid NOT NULL REFERENCES polymarket.copy_trade_backtest_runs (backtest_id) ON DELETE CASCADE,
        timestamp_utc timestamptz NOT NULL DEFAULT now(),
        wallet_count integer NOT NULL DEFAULT 0,
        signal_count integer NOT NULL DEFAULT 0,
        trade_count integer NOT NULL DEFAULT 0,
        gross_pnl_usd numeric(30,10) NOT NULL DEFAULT 0,
        net_pnl_usd numeric(30,10) NOT NULL DEFAULT 0,
        roi numeric(30,10),
        max_drawdown numeric(30,10),
        win_rate numeric(18,8),
        result_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_poly_copy_trade_backtest_results PRIMARY KEY (result_id, timestamp_utc)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.wallet_score_calibration_snapshots (
        snapshot_id uuid NOT NULL DEFAULT gen_random_uuid(),
        timestamp_utc timestamptz NOT NULL DEFAULT now(),
        score_version text NOT NULL,
        calibration_version text NOT NULL,
        sample_start timestamptz,
        sample_end timestamptz,
        wallet_count integer NOT NULL DEFAULT 0,
        trade_count integer NOT NULL DEFAULT 0,
        feature_weights jsonb NOT NULL DEFAULT '{}'::jsonb,
        thresholds jsonb NOT NULL DEFAULT '{}'::jsonb,
        metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_poly_wallet_score_calibration_snapshots PRIMARY KEY (snapshot_id, timestamp_utc),
        CONSTRAINT chk_poly_wallet_score_calibration_snapshot_counts CHECK (wallet_count >= 0 AND trade_count >= 0)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.whale_poll_checkpoints (
        checkpoint_name text PRIMARY KEY,
        last_polled_at timestamptz,
        next_cursor text,
        last_trade_timestamp_utc timestamptz,
        last_trade_id uuid,
        pages_seen bigint NOT NULL DEFAULT 0,
        trades_seen bigint NOT NULL DEFAULT 0,
        state jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_poly_whale_poll_checkpoint_counts CHECK (pages_seen >= 0 AND trades_seen >= 0)
      );
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_copy_trade_backtest_runs_status_started
        ON polymarket.copy_trade_backtest_runs (status, started_at DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_copy_trade_backtest_results_backtest_ts
        ON polymarket.copy_trade_backtest_results (backtest_id, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_score_calibration_version_ts
        ON polymarket.wallet_score_calibration_snapshots (score_version, calibration_version, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_whale_poll_checkpoints_trade_ts
        ON polymarket.whale_poll_checkpoints (last_trade_timestamp_utc DESC);
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        PERFORM create_hypertable('polymarket.copy_trade_backtest_results', 'timestamp_utc', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.wallet_score_calibration_snapshots', 'timestamp_utc', chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);
      END $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.whale_poll_checkpoints;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.wallet_score_calibration_snapshots;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.copy_trade_backtest_results;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.copy_trade_backtest_runs;`);
  }
}
