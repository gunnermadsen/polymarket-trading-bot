import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolymarketLiveReadiness1777109000000 implements MigrationInterface {
  name = 'AddPolymarketLiveReadiness1777109000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS polymarket;`);

    await queryRunner.query(`
      ALTER TABLE polymarket.orders
        ADD COLUMN IF NOT EXISTS venue_order_id text,
        ADD COLUMN IF NOT EXISTS venue_status text,
        ADD COLUMN IF NOT EXISTS submitted_at timestamptz,
        ADD COLUMN IF NOT EXISTS accepted_at timestamptz,
        ADD COLUMN IF NOT EXISTS last_venue_update_at timestamptz,
        ADD COLUMN IF NOT EXISTS reconciliation_status text;
    `);

    await queryRunner.query(`
      CREATE UNIQUE INDEX IF NOT EXISTS uq_poly_orders_client_order_id
        ON polymarket.orders (client_order_id);
    `);
    await queryRunner.query(`
      CREATE UNIQUE INDEX IF NOT EXISTS uq_poly_orders_venue_order_id
        ON polymarket.orders (venue_order_id)
        WHERE venue_order_id IS NOT NULL;
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.live_venue_events (
        event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        source text NOT NULL,
        event_type text NOT NULL,
        venue_event_id text,
        venue_order_id text,
        venue_trade_id text,
        client_order_id uuid,
        market_id text,
        token_id text,
        event_status text,
        event_timestamp timestamptz,
        event_hash text NOT NULL,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        applied boolean NOT NULL DEFAULT false,
        apply_error text,
        created_at timestamptz NOT NULL DEFAULT now()
      );
    `);
    await queryRunner.query(`
      CREATE UNIQUE INDEX IF NOT EXISTS uq_poly_live_venue_events_hash
        ON polymarket.live_venue_events (event_hash);
    `);
    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_live_venue_events_order_created
        ON polymarket.live_venue_events (venue_order_id, created_at DESC)
        WHERE venue_order_id IS NOT NULL;
    `);
    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_live_venue_events_trade_created
        ON polymarket.live_venue_events (venue_trade_id, created_at DESC)
        WHERE venue_trade_id IS NOT NULL;
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.live_reconciliation_runs (
        run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        started_at timestamptz NOT NULL DEFAULT now(),
        completed_at timestamptz,
        status text NOT NULL,
        open_orders_seen integer NOT NULL DEFAULT 0,
        fills_seen integer NOT NULL DEFAULT 0,
        balances_seen integer NOT NULL DEFAULT 0,
        mismatches_found integer NOT NULL DEFAULT 0,
        mismatches_repaired integer NOT NULL DEFAULT 0,
        unresolved_count integer NOT NULL DEFAULT 0,
        raw_summary jsonb NOT NULL DEFAULT '{}'::jsonb
      );
    `);
    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_live_reconciliation_runs_started
        ON polymarket.live_reconciliation_runs (started_at DESC);
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.live_idempotency_repairs (
        repair_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        created_at timestamptz NOT NULL DEFAULT now(),
        repair_type text NOT NULL,
        client_order_id uuid,
        venue_order_id text,
        source_event_id uuid REFERENCES polymarket.live_venue_events (event_id) ON DELETE SET NULL,
        before_state jsonb NOT NULL DEFAULT '{}'::jsonb,
        after_state jsonb NOT NULL DEFAULT '{}'::jsonb,
        status text NOT NULL
      );
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.live_idempotency_repairs;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.live_reconciliation_runs;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.live_venue_events;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.uq_poly_orders_venue_order_id;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.uq_poly_orders_client_order_id;`);
    await queryRunner.query(`
      ALTER TABLE polymarket.orders
        DROP COLUMN IF EXISTS reconciliation_status,
        DROP COLUMN IF EXISTS last_venue_update_at,
        DROP COLUMN IF EXISTS accepted_at,
        DROP COLUMN IF EXISTS submitted_at,
        DROP COLUMN IF EXISTS venue_status,
        DROP COLUMN IF EXISTS venue_order_id;
    `);
  }
}
