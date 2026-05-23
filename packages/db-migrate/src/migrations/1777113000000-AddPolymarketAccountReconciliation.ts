import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolymarketAccountReconciliation1777113000000 implements MigrationInterface {
  name = 'AddPolymarketAccountReconciliation1777113000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS polymarket;`);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.account_trades (
        account_trade_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        account_address text NOT NULL,
        token_id text NOT NULL,
        market_id text,
        side text NOT NULL,
        price numeric(18,8) NOT NULL,
        size numeric(30,10) NOT NULL,
        notional numeric(30,10) NOT NULL,
        timestamp_utc timestamptz NOT NULL,
        transaction_hash text,
        venue_order_id text,
        venue_trade_id text,
        source text NOT NULL,
        linked_order_id text REFERENCES polymarket.orders (order_id) ON DELETE SET NULL,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        applied_exit_size numeric(30,10) NOT NULL DEFAULT 0,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_poly_account_trades_side CHECK (side IN ('buy', 'sell')),
        CONSTRAINT chk_poly_account_trades_source CHECK (source IN ('user_ws', 'data_api', 'manual_backfill', 'poll')),
        CONSTRAINT chk_poly_account_trades_amounts CHECK (
          price >= 0 AND price <= 1
          AND size > 0
          AND notional >= 0
          AND applied_exit_size >= 0
          AND applied_exit_size <= size
        )
      );
    `);
    await queryRunner.query(`
      CREATE UNIQUE INDEX IF NOT EXISTS uq_poly_account_trades_identity
        ON polymarket.account_trades (
          account_address,
          token_id,
          side,
          price,
          size,
          timestamp_utc,
          (COALESCE(transaction_hash, '')),
          (COALESCE(venue_trade_id, ''))
        );
    `);
    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_account_trades_token_ts
        ON polymarket.account_trades (account_address, token_id, timestamp_utc DESC);
    `);
    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_account_trades_unapplied
        ON polymarket.account_trades (account_address, token_id, timestamp_utc)
        WHERE applied_exit_size < size;
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.account_position_snapshots (
        snapshot_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        account_address text NOT NULL,
        token_id text NOT NULL,
        market_id text,
        size numeric(30,10) NOT NULL DEFAULT 0,
        avg_price numeric(18,8),
        current_price numeric(18,8),
        current_value numeric(30,10),
        cash_pnl numeric(30,10),
        percent_pnl numeric(30,10),
        snapshot_at timestamptz NOT NULL DEFAULT now(),
        source text NOT NULL,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT chk_poly_account_position_snapshots_source CHECK (source IN ('data_api', 'manual_backfill', 'poll')),
        CONSTRAINT chk_poly_account_position_snapshots_amounts CHECK (
          size >= 0
          AND (avg_price IS NULL OR (avg_price >= 0 AND avg_price <= 1))
          AND (current_price IS NULL OR (current_price >= 0 AND current_price <= 1))
        )
      );
    `);
    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_account_position_snapshots_latest
        ON polymarket.account_position_snapshots (account_address, token_id, snapshot_at DESC);
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.account_reconciliation_runs (
        run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        account_address text NOT NULL,
        source text NOT NULL,
        dry_run boolean NOT NULL DEFAULT true,
        token_id text,
        lookback_hours integer NOT NULL,
        started_at timestamptz NOT NULL DEFAULT now(),
        completed_at timestamptz,
        status text NOT NULL,
        activities_fetched integer NOT NULL DEFAULT 0,
        account_trades_inserted integer NOT NULL DEFAULT 0,
        position_snapshots_inserted integer NOT NULL DEFAULT 0,
        exits_detected integer NOT NULL DEFAULT 0,
        exits_applied integer NOT NULL DEFAULT 0,
        mismatches_found integer NOT NULL DEFAULT 0,
        unmatched_trades integer NOT NULL DEFAULT 0,
        raw_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT chk_poly_account_reconciliation_runs_source CHECK (source IN ('user_ws', 'poll', 'manual_backfill', 'admin')),
        CONSTRAINT chk_poly_account_reconciliation_runs_status CHECK (status IN ('completed', 'dry_run', 'error'))
      );
    `);
    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_account_reconciliation_runs_started
        ON polymarket.account_reconciliation_runs (started_at DESC);
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.trade_exits
        DROP CONSTRAINT IF EXISTS chk_poly_trade_exits_type;
      ALTER TABLE polymarket.trade_exits
        ADD CONSTRAINT chk_poly_trade_exits_type CHECK (
          exit_type IN (
            'whale_exit',
            'whale_reduce',
            'resolution',
            'risk_control',
            'manual',
            'live_fill',
            'manual_ui_exit',
            'manual_ui_exit_backfill'
          )
        );
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE polymarket.trade_exits
        DROP CONSTRAINT IF EXISTS chk_poly_trade_exits_type;
      ALTER TABLE polymarket.trade_exits
        ADD CONSTRAINT chk_poly_trade_exits_type CHECK (exit_type IN ('whale_exit', 'whale_reduce', 'resolution', 'risk_control', 'manual', 'live_fill'));
    `);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.account_reconciliation_runs;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.account_position_snapshots;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.account_trades;`);
  }
}
