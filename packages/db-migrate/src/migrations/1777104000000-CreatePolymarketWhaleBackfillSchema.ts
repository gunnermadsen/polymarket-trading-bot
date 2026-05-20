import { MigrationInterface, QueryRunner } from 'typeorm';

export class CreatePolymarketWhaleBackfillSchema1777104000000 implements MigrationInterface {
  name = 'CreatePolymarketWhaleBackfillSchema1777104000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;`);
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS polymarket;`);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.backfill_jobs (
        job_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        job_type text NOT NULL,
        status text NOT NULL DEFAULT 'queued',
        requested_at timestamptz NOT NULL DEFAULT now(),
        started_at timestamptz,
        completed_at timestamptz,
        cancel_requested_at timestamptz,
        lookback_days integer NOT NULL DEFAULT 30,
        min_trade_usd numeric(30,10) NOT NULL DEFAULT 1000,
        request jsonb NOT NULL DEFAULT '{}'::jsonb,
        summary jsonb NOT NULL DEFAULT '{}'::jsonb,
        error text,
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_poly_backfill_jobs_type CHECK (job_type IN ('whales')),
        CONSTRAINT chk_poly_backfill_jobs_status CHECK (status IN ('queued', 'running', 'cancel_requested', 'completed', 'failed', 'cancelled')),
        CONSTRAINT chk_poly_backfill_jobs_request CHECK (lookback_days >= 0 AND min_trade_usd >= 0)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.backfill_job_events (
        event_id uuid NOT NULL DEFAULT gen_random_uuid(),
        job_id uuid NOT NULL REFERENCES polymarket.backfill_jobs (job_id) ON DELETE CASCADE,
        timestamp_utc timestamptz NOT NULL DEFAULT now(),
        level text NOT NULL,
        message text NOT NULL,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT pk_poly_backfill_job_events PRIMARY KEY (event_id, timestamp_utc),
        CONSTRAINT chk_poly_backfill_job_events_level CHECK (level IN ('debug', 'info', 'warn', 'error'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.wallets (
        proxy_wallet text PRIMARY KEY,
        first_seen_at timestamptz,
        last_seen_at timestamptz,
        total_observed_volume numeric(30,10) NOT NULL DEFAULT 0,
        total_observed_trades bigint NOT NULL DEFAULT 0,
        status text NOT NULL DEFAULT 'active',
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_poly_wallets_status CHECK (status IN ('active', 'ignored', 'blocked')),
        CONSTRAINT chk_poly_wallets_totals CHECK (total_observed_volume >= 0 AND total_observed_trades >= 0)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.wallet_trades (
        trade_id uuid NOT NULL DEFAULT gen_random_uuid(),
        proxy_wallet text NOT NULL REFERENCES polymarket.wallets (proxy_wallet) ON DELETE CASCADE,
        asset text NOT NULL,
        condition_id text,
        market_id text,
        side text NOT NULL,
        outcome text,
        price numeric(18,8) NOT NULL,
        size numeric(30,10) NOT NULL,
        cash_value numeric(30,10) NOT NULL,
        timestamp_utc timestamptz NOT NULL,
        title text,
        slug text,
        event_slug text,
        transaction_hash text,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_poly_wallet_trades PRIMARY KEY (trade_id, timestamp_utc),
        CONSTRAINT uq_polymarket_wallet_trades_identity UNIQUE (transaction_hash, proxy_wallet, asset, side, price, size, timestamp_utc),
        CONSTRAINT chk_poly_wallet_trades_side CHECK (side IN ('BUY', 'SELL', 'unknown', 'UNKNOWN')),
        CONSTRAINT chk_poly_wallet_trades_amounts CHECK (price >= 0 AND price <= 1 AND size > 0 AND cash_value >= 0)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.wallet_positions (
        position_id uuid NOT NULL DEFAULT gen_random_uuid(),
        proxy_wallet text NOT NULL REFERENCES polymarket.wallets (proxy_wallet) ON DELETE CASCADE,
        market_id text,
        token_id text,
        outcome text,
        size numeric(30,10) NOT NULL DEFAULT 0,
        avg_entry_price numeric(18,8),
        current_price numeric(18,8),
        realized_pnl numeric(30,10) NOT NULL DEFAULT 0,
        unrealized_pnl numeric(30,10) NOT NULL DEFAULT 0,
        snapshot_at timestamptz NOT NULL DEFAULT now(),
        source text NOT NULL DEFAULT 'backfill',
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT pk_poly_wallet_positions PRIMARY KEY (position_id, snapshot_at)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.wallet_scores (
        score_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        proxy_wallet text NOT NULL REFERENCES polymarket.wallets (proxy_wallet) ON DELETE CASCADE,
        score_version text NOT NULL,
        scored_at timestamptz NOT NULL DEFAULT now(),
        resolved_markets integer NOT NULL DEFAULT 0,
        total_trades integer NOT NULL DEFAULT 0,
        total_volume numeric(30,10) NOT NULL DEFAULT 0,
        realized_pnl numeric(30,10) NOT NULL DEFAULT 0,
        roi numeric(30,10) NOT NULL DEFAULT 0,
        win_rate numeric(18,8) NOT NULL DEFAULT 0,
        avg_trade_size numeric(30,10) NOT NULL DEFAULT 0,
        max_drawdown numeric(30,10) NOT NULL DEFAULT 0,
        score numeric(18,8) NOT NULL DEFAULT 0,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT uq_poly_wallet_scores_wallet_version UNIQUE (proxy_wallet, score_version),
        CONSTRAINT chk_poly_wallet_scores_score CHECK (score >= 0 AND score <= 100)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.copy_trade_signals (
        signal_id uuid NOT NULL DEFAULT gen_random_uuid(),
        timestamp_utc timestamptz NOT NULL DEFAULT now(),
        proxy_wallet text NOT NULL REFERENCES polymarket.wallets (proxy_wallet) ON DELETE CASCADE,
        wallet_score numeric(18,8) NOT NULL DEFAULT 0,
        source_trade_id uuid NOT NULL,
        market_id text,
        token_id text,
        side text NOT NULL,
        whale_price numeric(18,8) NOT NULL,
        observed_price numeric(18,8) NOT NULL,
        copy_size_usd numeric(30,10) NOT NULL,
        reason text NOT NULL,
        status text NOT NULL,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT pk_poly_copy_trade_signals PRIMARY KEY (signal_id, timestamp_utc),
        CONSTRAINT uq_poly_copy_trade_source UNIQUE (source_trade_id, timestamp_utc),
        CONSTRAINT chk_poly_copy_trade_signals_status CHECK (status IN ('detected', 'rejected', 'submitted', 'filled', 'closed'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.copy_trade_backtests (
        backtest_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id uuid REFERENCES polymarket.backfill_jobs (job_id) ON DELETE SET NULL,
        status text NOT NULL DEFAULT 'completed',
        score_version text NOT NULL,
        strategy_name text NOT NULL DEFAULT 'whale_follow_v1',
        range_start timestamptz,
        range_end timestamptz,
        wallet_count integer NOT NULL DEFAULT 0,
        signal_count integer NOT NULL DEFAULT 0,
        trade_count integer NOT NULL DEFAULT 0,
        gross_pnl_usd numeric(30,10) NOT NULL DEFAULT 0,
        net_pnl_usd numeric(30,10) NOT NULL DEFAULT 0,
        roi numeric(30,10),
        max_drawdown numeric(30,10),
        win_rate numeric(18,8),
        config jsonb NOT NULL DEFAULT '{}'::jsonb,
        result_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now()
      );
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_backfill_jobs_status_requested
        ON polymarket.backfill_jobs (status, requested_at DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_backfill_job_events_job_ts
        ON polymarket.backfill_job_events (job_id, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_wallets_seen
        ON polymarket.wallets (last_seen_at DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_trades_wallet_ts
        ON polymarket.wallet_trades (proxy_wallet, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_trades_condition_ts
        ON polymarket.wallet_trades (condition_id, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_trades_cash_ts
        ON polymarket.wallet_trades (cash_value DESC, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_scores_score
        ON polymarket.wallet_scores (score DESC, scored_at DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_copy_trade_signals_wallet_ts
        ON polymarket.copy_trade_signals (proxy_wallet, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_copy_trade_signals_status_ts
        ON polymarket.copy_trade_signals (status, timestamp_utc DESC);
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        PERFORM create_hypertable('polymarket.backfill_job_events', 'timestamp_utc', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.wallet_trades', 'timestamp_utc', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.wallet_positions', 'snapshot_at', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.copy_trade_signals', 'timestamp_utc', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
      END $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.copy_trade_backtests;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.copy_trade_signals;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.wallet_scores;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.wallet_positions;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.wallet_trades;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.backfill_job_events;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.wallets;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.backfill_jobs;`);
  }
}
