import { MigrationInterface, QueryRunner } from 'typeorm';

export class CreatePolymarketBotSchema1777102000000 implements MigrationInterface {
  name = 'CreatePolymarketBotSchema1777102000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;`);
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS polymarket;`);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.markets (
        market_id text PRIMARY KEY,
        event_id text NOT NULL,
        outcome_group_id text,
        question text NOT NULL,
        category text,
        active boolean NOT NULL DEFAULT false,
        closed boolean NOT NULL DEFAULT false,
        archived boolean NOT NULL DEFAULT false,
        neg_risk boolean NOT NULL DEFAULT false,
        neg_risk_augmented boolean NOT NULL DEFAULT false,
        rules text,
        end_date timestamptz,
        underlying_key text NOT NULL,
        resolution_score integer NOT NULL DEFAULT 0,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now()
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.outcome_tokens (
        token_id text PRIMARY KEY,
        market_id text NOT NULL REFERENCES polymarket.markets (market_id) ON DELETE CASCADE,
        outcome text NOT NULL,
        side text NOT NULL,
        condition_id text,
        tick_size numeric(18,8) NOT NULL DEFAULT 0.01,
        neg_risk boolean NOT NULL DEFAULT false,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_polymarket_outcome_tokens_side CHECK (side IN ('yes', 'no'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.orderbook_snapshots (
        snapshot_id uuid NOT NULL DEFAULT gen_random_uuid(),
        timestamp_utc timestamptz NOT NULL,
        market_id text,
        token_id text NOT NULL,
        best_bid numeric(18,8),
        best_ask numeric(18,8),
        tick_size numeric(18,8),
        stale_level_count integer NOT NULL DEFAULT 0,
        fresh_depth_bid numeric(30,10),
        fresh_depth_ask numeric(30,10),
        book jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_polymarket_orderbook_snapshots PRIMARY KEY (snapshot_id, timestamp_utc)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.signal_candidates (
        signal_id uuid NOT NULL,
        timestamp_utc timestamptz NOT NULL,
        signal_type text NOT NULL,
        market_id text NOT NULL,
        expected_edge numeric(30,10) NOT NULL DEFAULT 0,
        threshold numeric(30,10) NOT NULL DEFAULT 0,
        size numeric(30,10) NOT NULL DEFAULT 0,
        status text NOT NULL,
        reject_reason text,
        worst_case_loss numeric(30,10),
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_polymarket_signal_candidates PRIMARY KEY (signal_id, timestamp_utc)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.orders (
        order_id text PRIMARY KEY,
        client_order_id uuid NOT NULL,
        created_at timestamptz NOT NULL,
        updated_at timestamptz NOT NULL,
        market_id text NOT NULL,
        token_id text NOT NULL,
        side text NOT NULL,
        order_type text NOT NULL,
        price numeric(18,8) NOT NULL,
        size numeric(30,10) NOT NULL,
        state text NOT NULL,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT chk_polymarket_orders_side CHECK (side IN ('buy', 'sell')),
        CONSTRAINT chk_polymarket_orders_order_type CHECK (order_type IN ('fok', 'gtc', 'gtd'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.fills (
        fill_id uuid NOT NULL,
        order_id text NOT NULL,
        token_id text NOT NULL,
        timestamp_utc timestamptz NOT NULL,
        price numeric(18,8) NOT NULL,
        size numeric(30,10) NOT NULL,
        fee numeric(30,10) NOT NULL DEFAULT 0,
        source text NOT NULL,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_polymarket_fills PRIMARY KEY (fill_id, timestamp_utc),
        CONSTRAINT chk_polymarket_fills_source CHECK (source IN ('sim', 'live'))
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.positions (
        position_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        market_id text NOT NULL,
        token_id text NOT NULL,
        underlying_key text NOT NULL,
        status text NOT NULL,
        size numeric(30,10) NOT NULL DEFAULT 0,
        cost_basis numeric(30,10) NOT NULL DEFAULT 0,
        worst_case_loss numeric(30,10) NOT NULL DEFAULT 0,
        opened_at timestamptz NOT NULL DEFAULT now(),
        closed_at timestamptz,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT uq_polymarket_positions_market_token UNIQUE (market_id, token_id)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.conversions (
        conversion_id uuid NOT NULL,
        timestamp_utc timestamptz NOT NULL,
        market_id text NOT NULL,
        no_token_id text NOT NULL,
        size numeric(30,10) NOT NULL,
        status text NOT NULL,
        tx_hash text,
        latency_ms bigint,
        gas_cost_usd numeric(30,10) NOT NULL DEFAULT 0,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_polymarket_conversions PRIMARY KEY (conversion_id, timestamp_utc)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.funnel_events (
        event_id uuid NOT NULL DEFAULT gen_random_uuid(),
        timestamp_utc timestamptz NOT NULL,
        signal_id uuid,
        market_id text,
        stage text NOT NULL,
        status text NOT NULL,
        reason text,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_polymarket_funnel_events PRIMARY KEY (event_id, timestamp_utc)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.risk_events (
        event_id uuid NOT NULL DEFAULT gen_random_uuid(),
        timestamp_utc timestamptz NOT NULL,
        event_type text NOT NULL,
        severity text NOT NULL,
        message text NOT NULL,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_polymarket_risk_events PRIMARY KEY (event_id, timestamp_utc)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.daily_metrics (
        metric_date date NOT NULL,
        metric_name text NOT NULL,
        value jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_polymarket_daily_metrics PRIMARY KEY (metric_date, metric_name)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.underlying_overrides (
        override_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        event_id text,
        market_id text,
        underlying_key text NOT NULL,
        reason text,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT uq_polymarket_underlying_override UNIQUE (event_id, market_id)
      );
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_polymarket_markets_event ON polymarket.markets (event_id);
      CREATE INDEX IF NOT EXISTS idx_polymarket_markets_underlying ON polymarket.markets (underlying_key);
      CREATE INDEX IF NOT EXISTS idx_polymarket_markets_active ON polymarket.markets (active, closed, archived);
      CREATE INDEX IF NOT EXISTS idx_polymarket_outcome_tokens_market ON polymarket.outcome_tokens (market_id);
      CREATE INDEX IF NOT EXISTS idx_polymarket_orderbook_token_ts ON polymarket.orderbook_snapshots (token_id, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_polymarket_signal_market_ts ON polymarket.signal_candidates (market_id, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_polymarket_signal_status_ts ON polymarket.signal_candidates (status, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_polymarket_orders_market_state ON polymarket.orders (market_id, state);
      CREATE INDEX IF NOT EXISTS idx_polymarket_fills_order_ts ON polymarket.fills (order_id, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_polymarket_positions_underlying_status ON polymarket.positions (underlying_key, status);
      CREATE INDEX IF NOT EXISTS idx_polymarket_conversions_market_ts ON polymarket.conversions (market_id, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_polymarket_funnel_stage_ts ON polymarket.funnel_events (stage, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_polymarket_risk_type_ts ON polymarket.risk_events (event_type, timestamp_utc DESC);
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        PERFORM create_hypertable('polymarket.orderbook_snapshots', 'timestamp_utc', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.signal_candidates', 'timestamp_utc', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.fills', 'timestamp_utc', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.conversions', 'timestamp_utc', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.funnel_events', 'timestamp_utc', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
        PERFORM create_hypertable('polymarket.risk_events', 'timestamp_utc', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
      END $$;
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1 FROM timescaledb_information.hypertables
          WHERE hypertable_schema = 'polymarket' AND hypertable_name = 'orderbook_snapshots' AND compression_enabled = false
        ) THEN
          ALTER TABLE polymarket.orderbook_snapshots SET (
            timescaledb.compress = true,
            timescaledb.compress_orderby = 'timestamp_utc DESC, snapshot_id',
            timescaledb.compress_segmentby = 'token_id'
          );
        END IF;

        IF EXISTS (
          SELECT 1 FROM timescaledb_information.hypertables
          WHERE hypertable_schema = 'polymarket' AND hypertable_name = 'signal_candidates' AND compression_enabled = false
        ) THEN
          ALTER TABLE polymarket.signal_candidates SET (
            timescaledb.compress = true,
            timescaledb.compress_orderby = 'timestamp_utc DESC, signal_id',
            timescaledb.compress_segmentby = 'signal_type,status'
          );
        END IF;

        IF EXISTS (
          SELECT 1 FROM timescaledb_information.hypertables
          WHERE hypertable_schema = 'polymarket' AND hypertable_name = 'fills' AND compression_enabled = false
        ) THEN
          ALTER TABLE polymarket.fills SET (
            timescaledb.compress = true,
            timescaledb.compress_orderby = 'timestamp_utc DESC, fill_id',
            timescaledb.compress_segmentby = 'token_id,source'
          );
        END IF;

        PERFORM add_retention_policy('polymarket.orderbook_snapshots', INTERVAL '30 days', if_not_exists => true);
        PERFORM add_retention_policy('polymarket.signal_candidates', INTERVAL '180 days', if_not_exists => true);
        PERFORM add_retention_policy('polymarket.fills', INTERVAL '180 days', if_not_exists => true);
        PERFORM add_retention_policy('polymarket.conversions', INTERVAL '180 days', if_not_exists => true);
        PERFORM add_retention_policy('polymarket.funnel_events', INTERVAL '180 days', if_not_exists => true);
        PERFORM add_retention_policy('polymarket.risk_events', INTERVAL '180 days', if_not_exists => true);

        PERFORM add_compression_policy('polymarket.orderbook_snapshots', INTERVAL '7 days', if_not_exists => true);
        PERFORM add_compression_policy('polymarket.signal_candidates', INTERVAL '14 days', if_not_exists => true);
        PERFORM add_compression_policy('polymarket.fills', INTERVAL '14 days', if_not_exists => true);
      END $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.underlying_overrides;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.daily_metrics;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.risk_events;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.funnel_events;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.conversions;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.positions;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.fills;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.orders;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.signal_candidates;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.orderbook_snapshots;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.outcome_tokens;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.markets;`);
    await queryRunner.query(`DROP SCHEMA IF EXISTS polymarket;`);
  }
}
