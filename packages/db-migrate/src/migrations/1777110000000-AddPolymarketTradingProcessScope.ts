import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolymarketTradingProcessScope1777110000000 implements MigrationInterface {
  name = 'AddPolymarketTradingProcessScope1777110000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;`);
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS polymarket;`);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.trading_processes (
        process_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        name text NOT NULL,
        process_type text NOT NULL,
        process_scope text NOT NULL DEFAULT 'default',
        process_key text,
        status text NOT NULL DEFAULT 'created',
        enabled boolean NOT NULL DEFAULT false,
        hostname text,
        pid integer,
        version text,
        started_at timestamptz NOT NULL DEFAULT now(),
        heartbeat_at timestamptz,
        stopped_at timestamptz,
        stop_reason text,
        last_error text,
        config jsonb NOT NULL DEFAULT '{}'::jsonb,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_poly_trading_processes_name CHECK (length(btrim(name)) > 0),
        CONSTRAINT chk_poly_trading_processes_type CHECK (length(btrim(process_type)) > 0),
        CONSTRAINT chk_poly_trading_processes_scope CHECK (length(btrim(process_scope)) > 0),
        CONSTRAINT chk_poly_trading_processes_status CHECK (length(btrim(status)) > 0),
        CONSTRAINT chk_poly_trading_processes_pid CHECK (pid IS NULL OR pid > 0)
      );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.trading_process_events (
        event_id uuid NOT NULL DEFAULT gen_random_uuid(),
        process_id uuid NOT NULL REFERENCES polymarket.trading_processes (process_id) ON DELETE CASCADE,
        timestamp_utc timestamptz NOT NULL DEFAULT now(),
        level text NOT NULL,
        event_type text NOT NULL,
        message text,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_poly_trading_process_events PRIMARY KEY (event_id, timestamp_utc),
        CONSTRAINT chk_poly_trading_process_events_level CHECK (level IN ('debug', 'info', 'warn', 'error'))
      );
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.signal_candidates
        ADD COLUMN IF NOT EXISTS process_id uuid;
      ALTER TABLE polymarket.orders
        ADD COLUMN IF NOT EXISTS process_id uuid;
      ALTER TABLE polymarket.fills
        ADD COLUMN IF NOT EXISTS process_id uuid;
      ALTER TABLE polymarket.copy_trade_signals
        ADD COLUMN IF NOT EXISTS process_id uuid;
      ALTER TABLE polymarket.trade_positions
        ADD COLUMN IF NOT EXISTS process_id uuid;
      ALTER TABLE polymarket.trade_marks
        ADD COLUMN IF NOT EXISTS process_id uuid;
      ALTER TABLE polymarket.trade_exits
        ADD COLUMN IF NOT EXISTS process_id uuid;
      ALTER TABLE polymarket.wallet_trade_performance
        ADD COLUMN IF NOT EXISTS process_id uuid;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.signal_candidates SET (timescaledb.compress = false);
      ALTER TABLE polymarket.fills SET (timescaledb.compress = false);
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.fills
        DROP CONSTRAINT IF EXISTS chk_polymarket_fills_source;
      ALTER TABLE polymarket.fills
        ADD CONSTRAINT chk_polymarket_fills_source CHECK (source IN ('sim', 'paper', 'live'));
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        IF NOT EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'fk_poly_signal_candidates_process'
        ) THEN
          ALTER TABLE polymarket.signal_candidates
          ADD CONSTRAINT fk_poly_signal_candidates_process
          FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes (process_id)
          ON DELETE SET NULL NOT VALID;
        END IF;

        IF NOT EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'fk_poly_orders_process'
        ) THEN
          ALTER TABLE polymarket.orders
          ADD CONSTRAINT fk_poly_orders_process
          FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes (process_id)
          ON DELETE SET NULL NOT VALID;
        END IF;

        IF NOT EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'fk_poly_fills_process'
        ) THEN
          ALTER TABLE polymarket.fills
          ADD CONSTRAINT fk_poly_fills_process
          FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes (process_id)
          ON DELETE SET NULL NOT VALID;
        END IF;

        IF NOT EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'fk_poly_copy_trade_signals_process'
        ) THEN
          ALTER TABLE polymarket.copy_trade_signals
          ADD CONSTRAINT fk_poly_copy_trade_signals_process
          FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes (process_id)
          ON DELETE SET NULL NOT VALID;
        END IF;

        IF NOT EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'fk_poly_trade_positions_process'
        ) THEN
          ALTER TABLE polymarket.trade_positions
          ADD CONSTRAINT fk_poly_trade_positions_process
          FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes (process_id)
          ON DELETE SET NULL NOT VALID;
        END IF;

        IF NOT EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'fk_poly_trade_marks_process'
        ) THEN
          ALTER TABLE polymarket.trade_marks
          ADD CONSTRAINT fk_poly_trade_marks_process
          FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes (process_id)
          ON DELETE SET NULL NOT VALID;
        END IF;

        IF NOT EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'fk_poly_trade_exits_process'
        ) THEN
          ALTER TABLE polymarket.trade_exits
          ADD CONSTRAINT fk_poly_trade_exits_process
          FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes (process_id)
          ON DELETE SET NULL NOT VALID;
        END IF;

        IF NOT EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'fk_poly_wallet_trade_perf_process'
        ) THEN
          ALTER TABLE polymarket.wallet_trade_performance
          ADD CONSTRAINT fk_poly_wallet_trade_perf_process
          FOREIGN KEY (process_id) REFERENCES polymarket.trading_processes (process_id)
          ON DELETE SET NULL NOT VALID;
        END IF;
      END $$;
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'uq_poly_copy_trade_source'
        ) THEN
          ALTER TABLE polymarket.copy_trade_signals
          DROP CONSTRAINT uq_poly_copy_trade_source;
        END IF;

        IF EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'uq_poly_trade_positions_signal_token'
        ) THEN
          ALTER TABLE polymarket.trade_positions
          DROP CONSTRAINT uq_poly_trade_positions_signal_token;
        END IF;
      END $$;
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_trading_processes_status_heartbeat
        ON polymarket.trading_processes (enabled, status, heartbeat_at DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_trading_processes_updated
        ON polymarket.trading_processes (updated_at DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_trading_processes_type_scope_started
        ON polymarket.trading_processes (process_type, process_scope, started_at DESC);
      CREATE UNIQUE INDEX IF NOT EXISTS uq_poly_trading_processes_active_key
        ON polymarket.trading_processes (process_type, process_scope, process_key)
        WHERE process_key IS NOT NULL AND enabled = true AND status NOT IN ('stopped', 'failed', 'expired');
      CREATE INDEX IF NOT EXISTS idx_poly_trading_process_events_process_ts
        ON polymarket.trading_process_events (process_id, timestamp_utc DESC);
      CREATE INDEX IF NOT EXISTS idx_poly_trading_process_events_type_ts
        ON polymarket.trading_process_events (event_type, timestamp_utc DESC);

      CREATE INDEX IF NOT EXISTS idx_poly_signal_candidates_process_ts
        ON polymarket.signal_candidates (process_id, timestamp_utc DESC)
        WHERE process_id IS NOT NULL;
      CREATE INDEX IF NOT EXISTS idx_poly_orders_process_created
        ON polymarket.orders (process_id, created_at DESC)
        WHERE process_id IS NOT NULL;
      CREATE INDEX IF NOT EXISTS idx_poly_fills_process_ts
        ON polymarket.fills (process_id, timestamp_utc DESC)
        WHERE process_id IS NOT NULL;
      CREATE INDEX IF NOT EXISTS idx_poly_copy_trade_signals_process_ts
        ON polymarket.copy_trade_signals (process_id, timestamp_utc DESC)
        WHERE process_id IS NOT NULL;
      CREATE INDEX IF NOT EXISTS idx_poly_trade_positions_process_status
        ON polymarket.trade_positions (process_id, status, updated_at DESC)
        WHERE process_id IS NOT NULL;
      CREATE INDEX IF NOT EXISTS idx_poly_trade_marks_process_ts
        ON polymarket.trade_marks (process_id, timestamp_utc DESC)
        WHERE process_id IS NOT NULL;
      CREATE INDEX IF NOT EXISTS idx_poly_trade_exits_process_ts
        ON polymarket.trade_exits (process_id, timestamp_utc DESC)
        WHERE process_id IS NOT NULL;
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_trade_perf_process_rank
        ON polymarket.wallet_trade_performance (process_id, net_pnl DESC, profit_factor DESC, win_rate DESC)
        WHERE process_id IS NOT NULL;

      CREATE UNIQUE INDEX IF NOT EXISTS uq_poly_copy_trade_source_legacy
        ON polymarket.copy_trade_signals (source_trade_id, timestamp_utc)
        WHERE process_id IS NULL;
      CREATE UNIQUE INDEX IF NOT EXISTS uq_poly_copy_trade_source_process
        ON polymarket.copy_trade_signals (source_trade_id, timestamp_utc, process_id)
        WHERE process_id IS NOT NULL;
      CREATE UNIQUE INDEX IF NOT EXISTS uq_poly_trade_positions_signal_token_legacy
        ON polymarket.trade_positions (source_signal_table, source_signal_id, token_id)
        WHERE process_id IS NULL;
      CREATE UNIQUE INDEX IF NOT EXISTS uq_poly_trade_positions_signal_token_process
        ON polymarket.trade_positions (source_signal_table, source_signal_id, token_id, process_id)
        WHERE process_id IS NOT NULL;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.signal_candidates SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'timestamp_utc DESC, signal_id',
        timescaledb.compress_segmentby = 'signal_type,status,process_id'
      );
      ALTER TABLE polymarket.fills SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'timestamp_utc DESC, fill_id',
        timescaledb.compress_segmentby = 'token_id,source,process_id'
      );
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        PERFORM create_hypertable('polymarket.trading_process_events', 'timestamp_utc', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);
      END $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE polymarket.signal_candidates SET (timescaledb.compress = false);
      ALTER TABLE polymarket.fills SET (timescaledb.compress = false);
    `);

    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.uq_poly_trade_positions_signal_token_process;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.uq_poly_trade_positions_signal_token_legacy;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.uq_poly_copy_trade_source_process;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.uq_poly_copy_trade_source_legacy;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_wallet_trade_perf_process_rank;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_trade_exits_process_ts;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_trade_marks_process_ts;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_trade_positions_process_status;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_copy_trade_signals_process_ts;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_fills_process_ts;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_orders_process_created;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_signal_candidates_process_ts;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_trading_process_events_type_ts;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_trading_process_events_process_ts;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.uq_poly_trading_processes_active_key;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_trading_processes_type_scope_started;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_trading_processes_updated;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_trading_processes_status_heartbeat;`);

    await queryRunner.query(`
      ALTER TABLE polymarket.wallet_trade_performance
        DROP CONSTRAINT IF EXISTS fk_poly_wallet_trade_perf_process;
      ALTER TABLE polymarket.trade_exits
        DROP CONSTRAINT IF EXISTS fk_poly_trade_exits_process;
      ALTER TABLE polymarket.trade_marks
        DROP CONSTRAINT IF EXISTS fk_poly_trade_marks_process;
      ALTER TABLE polymarket.trade_positions
        DROP CONSTRAINT IF EXISTS fk_poly_trade_positions_process;
      ALTER TABLE polymarket.copy_trade_signals
        DROP CONSTRAINT IF EXISTS fk_poly_copy_trade_signals_process;
      ALTER TABLE polymarket.fills
        DROP CONSTRAINT IF EXISTS fk_poly_fills_process;
      ALTER TABLE polymarket.orders
        DROP CONSTRAINT IF EXISTS fk_poly_orders_process;
      ALTER TABLE polymarket.signal_candidates
        DROP CONSTRAINT IF EXISTS fk_poly_signal_candidates_process;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.fills
        DROP CONSTRAINT IF EXISTS chk_polymarket_fills_source;
      ALTER TABLE polymarket.fills
        ADD CONSTRAINT chk_polymarket_fills_source CHECK (source IN ('sim', 'live'));
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.wallet_trade_performance
        DROP COLUMN IF EXISTS process_id;
      ALTER TABLE polymarket.trade_exits
        DROP COLUMN IF EXISTS process_id;
      ALTER TABLE polymarket.trade_marks
        DROP COLUMN IF EXISTS process_id;
      ALTER TABLE polymarket.trade_positions
        DROP COLUMN IF EXISTS process_id;
      ALTER TABLE polymarket.copy_trade_signals
        DROP COLUMN IF EXISTS process_id;
      ALTER TABLE polymarket.fills
        DROP COLUMN IF EXISTS process_id;
      ALTER TABLE polymarket.orders
        DROP COLUMN IF EXISTS process_id;
      ALTER TABLE polymarket.signal_candidates
        DROP COLUMN IF EXISTS process_id;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.signal_candidates SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'timestamp_utc DESC, signal_id',
        timescaledb.compress_segmentby = 'signal_type,status'
      );
      ALTER TABLE polymarket.fills SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'timestamp_utc DESC, fill_id',
        timescaledb.compress_segmentby = 'token_id,source'
      );
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        IF NOT EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'uq_poly_copy_trade_source'
        ) AND NOT EXISTS (
          SELECT 1
          FROM polymarket.copy_trade_signals
          GROUP BY source_trade_id, timestamp_utc
          HAVING count(*) > 1
        ) THEN
          ALTER TABLE polymarket.copy_trade_signals
          ADD CONSTRAINT uq_poly_copy_trade_source UNIQUE (source_trade_id, timestamp_utc);
        END IF;

        IF NOT EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'uq_poly_trade_positions_signal_token'
        ) AND NOT EXISTS (
          SELECT 1
          FROM polymarket.trade_positions
          GROUP BY source_signal_table, source_signal_id, token_id
          HAVING count(*) > 1
        ) THEN
          ALTER TABLE polymarket.trade_positions
          ADD CONSTRAINT uq_poly_trade_positions_signal_token UNIQUE (source_signal_table, source_signal_id, token_id);
        END IF;
      END $$;
    `);

    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.trading_process_events;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.trading_processes;`);
  }
}
