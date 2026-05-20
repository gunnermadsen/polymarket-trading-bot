import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolymarketTradePnlLifecycle1777107000000 implements MigrationInterface {
  name = 'AddPolymarketTradePnlLifecycle1777107000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;`);
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS polymarket;`);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.trade_positions (
        position_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        source_signal_table text NOT NULL,
        source_signal_id uuid NOT NULL,
        source_trade_id uuid,
        signal_source text NOT NULL,
        strategy_name text NOT NULL,
        strategy_version text NOT NULL,
        strategy_config_hash text,
        execution_mode text NOT NULL,
        venue text NOT NULL,
        is_live_capital boolean NOT NULL DEFAULT false,
        proxy_wallet text REFERENCES polymarket.wallets (proxy_wallet) ON DELETE SET NULL,
        market_id text,
        token_id text NOT NULL,
        side text NOT NULL,
        entry_price numeric(18,8) NOT NULL,
        entry_size numeric(30,10) NOT NULL,
        open_size numeric(30,10) NOT NULL,
        entry_notional numeric(30,10) NOT NULL,
        entry_fee numeric(30,10) NOT NULL DEFAULT 0,
        entry_timestamp timestamptz NOT NULL,
        follow_lag_seconds integer,
        status text NOT NULL DEFAULT 'open',
        latest_mark_price numeric(18,8),
        latest_mark_timestamp timestamptz,
        unrealized_pnl numeric(30,10) NOT NULL DEFAULT 0,
        realized_pnl numeric(30,10) NOT NULL DEFAULT 0,
        roi numeric(30,10) NOT NULL DEFAULT 0,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT uq_poly_trade_positions_signal_token UNIQUE (source_signal_table, source_signal_id, token_id),
        CONSTRAINT chk_poly_trade_positions_mode CHECK (execution_mode IN ('sim', 'paper', 'live')),
        CONSTRAINT chk_poly_trade_positions_venue CHECK (venue IN ('sim', 'paper', 'polymarket_clob')),
        CONSTRAINT chk_poly_trade_positions_live_capital CHECK (is_live_capital = false OR execution_mode = 'live'),
        CONSTRAINT chk_poly_trade_positions_side CHECK (side IN ('buy', 'sell')),
        CONSTRAINT chk_poly_trade_positions_status CHECK (status IN ('open', 'partially_closed', 'closed', 'resolved', 'error')),
        CONSTRAINT chk_poly_trade_positions_amounts CHECK (
          entry_price >= 0 AND entry_price <= 1
          AND entry_size > 0
          AND open_size >= 0
          AND open_size <= entry_size
          AND entry_notional >= 0
          AND entry_fee >= 0
        )
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.trade_marks (
        mark_id uuid NOT NULL DEFAULT gen_random_uuid(),
        position_id uuid NOT NULL REFERENCES polymarket.trade_positions (position_id) ON DELETE CASCADE,
        source_signal_id uuid NOT NULL,
        timestamp_utc timestamptz NOT NULL DEFAULT now(),
        mark_price numeric(18,8) NOT NULL,
        mark_source text NOT NULL,
        mark_age_ms bigint,
        gross_unrealized_pnl numeric(30,10) NOT NULL DEFAULT 0,
        net_unrealized_pnl numeric(30,10) NOT NULL DEFAULT 0,
        roi numeric(30,10) NOT NULL DEFAULT 0,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_poly_trade_marks PRIMARY KEY (mark_id, timestamp_utc),
        CONSTRAINT chk_poly_trade_marks_price CHECK (mark_price >= 0 AND mark_price <= 1),
        CONSTRAINT chk_poly_trade_marks_source CHECK (mark_source IN ('clob_mid', 'clob_bid', 'clob_ask', 'data_api_trade', 'resolution', 'manual', 'last_known'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.trade_exits (
        exit_id uuid NOT NULL DEFAULT gen_random_uuid(),
        position_id uuid NOT NULL REFERENCES polymarket.trade_positions (position_id) ON DELETE CASCADE,
        source_signal_id uuid NOT NULL,
        timestamp_utc timestamptz NOT NULL DEFAULT now(),
        exit_type text NOT NULL,
        is_synthetic boolean NOT NULL DEFAULT false,
        exit_trigger_wallet text,
        exit_source_trade_id uuid,
        exit_price numeric(18,8) NOT NULL,
        exit_size numeric(30,10) NOT NULL,
        exit_notional numeric(30,10) NOT NULL,
        exit_fee numeric(30,10) NOT NULL DEFAULT 0,
        slippage_cost numeric(30,10) NOT NULL DEFAULT 0,
        gross_pnl numeric(30,10) NOT NULL DEFAULT 0,
        net_pnl numeric(30,10) NOT NULL DEFAULT 0,
        roi numeric(30,10) NOT NULL DEFAULT 0,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_poly_trade_exits PRIMARY KEY (exit_id, timestamp_utc),
        CONSTRAINT uq_poly_trade_exits_position_source UNIQUE (position_id, exit_source_trade_id),
        CONSTRAINT chk_poly_trade_exits_type CHECK (exit_type IN ('whale_exit', 'whale_reduce', 'resolution', 'risk_control', 'manual', 'live_fill')),
        CONSTRAINT chk_poly_trade_exits_price CHECK (exit_price >= 0 AND exit_price <= 1),
        CONSTRAINT chk_poly_trade_exits_amounts CHECK (exit_size > 0 AND exit_notional >= 0 AND exit_fee >= 0 AND slippage_cost >= 0)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.wallet_trade_performance (
        proxy_wallet text NOT NULL,
        strategy_name text NOT NULL,
        execution_mode text NOT NULL,
        venue text NOT NULL,
        is_live_capital boolean NOT NULL DEFAULT false,
        open_positions integer NOT NULL DEFAULT 0,
        closed_positions integer NOT NULL DEFAULT 0,
        winning_positions integer NOT NULL DEFAULT 0,
        losing_positions integer NOT NULL DEFAULT 0,
        gross_pnl numeric(30,10) NOT NULL DEFAULT 0,
        net_pnl numeric(30,10) NOT NULL DEFAULT 0,
        unrealized_pnl numeric(30,10) NOT NULL DEFAULT 0,
        total_notional numeric(30,10) NOT NULL DEFAULT 0,
        roi numeric(30,10) NOT NULL DEFAULT 0,
        win_rate numeric(18,8) NOT NULL DEFAULT 0,
        profit_factor numeric(30,10) NOT NULL DEFAULT 0,
        avg_win numeric(30,10) NOT NULL DEFAULT 0,
        avg_loss numeric(30,10) NOT NULL DEFAULT 0,
        max_drawdown numeric(30,10) NOT NULL DEFAULT 0,
        last_updated_at timestamptz NOT NULL DEFAULT now(),
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT pk_poly_wallet_trade_performance PRIMARY KEY (proxy_wallet, strategy_name, execution_mode, venue),
        CONSTRAINT chk_poly_wallet_trade_perf_mode CHECK (execution_mode IN ('sim', 'paper', 'live')),
        CONSTRAINT chk_poly_wallet_trade_perf_venue CHECK (venue IN ('sim', 'paper', 'polymarket_clob')),
        CONSTRAINT chk_poly_wallet_trade_perf_live_capital CHECK (is_live_capital = false OR execution_mode = 'live'),
        CONSTRAINT chk_poly_wallet_trade_perf_counts CHECK (
          open_positions >= 0
          AND closed_positions >= 0
          AND winning_positions >= 0
          AND losing_positions >= 0
          AND winning_positions + losing_positions <= closed_positions
        )
      );
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_trade_positions_status_updated
        ON polymarket.trade_positions (status, updated_at DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_trade_positions_wallet_strategy
        ON polymarket.trade_positions (proxy_wallet, strategy_name, execution_mode, status);
      CREATE INDEX IF NOT EXISTS idx_poly_trade_positions_token_status
        ON polymarket.trade_positions (token_id, status, entry_timestamp DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_trade_marks_position_ts
        ON polymarket.trade_marks (position_id, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_trade_exits_position_ts
        ON polymarket.trade_exits (position_id, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_trade_exits_wallet_source
        ON polymarket.trade_exits (exit_trigger_wallet, exit_source_trade_id);
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_trade_perf_rank
        ON polymarket.wallet_trade_performance (net_pnl DESC, profit_factor DESC, win_rate DESC);
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        PERFORM create_hypertable('polymarket.trade_marks', 'timestamp_utc', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
      END $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.wallet_trade_performance;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.trade_exits;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.trade_marks;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.trade_positions;`);
  }
}
