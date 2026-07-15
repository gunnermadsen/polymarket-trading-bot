import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddMlTrainingBackfillIngestion1777125000000 implements MigrationInterface {
  name = 'AddMlTrainingBackfillIngestion1777125000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.btc_interval_markets (
        market_id text PRIMARY KEY,
        event_id text NOT NULL,
        event_slug text NOT NULL UNIQUE,
        question text NOT NULL,
        series_slug text NOT NULL,
        window_start timestamptz NOT NULL UNIQUE,
        window_end timestamptz NOT NULL,
        condition_id text NOT NULL,
        up_token_id text NOT NULL,
        down_token_id text NOT NULL,
        resolution_source text NOT NULL,
        accepting_orders boolean NOT NULL DEFAULT false,
        active boolean NOT NULL DEFAULT false,
        closed boolean NOT NULL DEFAULT false,
        min_tick_size numeric(18,8) NOT NULL,
        min_order_size numeric(30,10) NOT NULL,
        fee_rate numeric(18,8),
        fee_exponent integer,
        fee_taker_only boolean,
        validation_status text NOT NULL,
        validation_errors jsonb NOT NULL DEFAULT '[]'::jsonb,
        reference_price numeric(30,10),
        reference_source_timestamp timestamptz,
        resolution_price numeric(30,10),
        resolution_source_timestamp timestamptz,
        resolved_outcome text,
        official_outcome text,
        official_resolved_at timestamptz,
        official_winning_token_id text,
        official_resolution_source text,
        official_resolution_received_at timestamptz,
        official_resolution_payload jsonb,
        discovered_at timestamptz NOT NULL,
        last_refreshed_at timestamptz NOT NULL,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_btc_interval_window
          CHECK (window_end = window_start + interval '5 minutes'),
        CONSTRAINT chk_btc_interval_tokens CHECK (up_token_id <> down_token_id),
        CONSTRAINT chk_btc_interval_validation
          CHECK (validation_status IN ('valid','invalid','ineligible')),
        CONSTRAINT chk_btc_interval_outcome
          CHECK (resolved_outcome IS NULL OR resolved_outcome IN ('up','down')),
        CONSTRAINT chk_btc_interval_official_outcome
          CHECK (official_outcome IS NULL OR official_outcome IN ('up','down')),
        CONSTRAINT chk_btc_official_resolution_all_or_none CHECK (
          (official_outcome IS NULL
            AND official_resolved_at IS NULL
            AND official_winning_token_id IS NULL
            AND official_resolution_source IS NULL
            AND official_resolution_received_at IS NULL
            AND official_resolution_payload IS NULL)
          OR
          (official_outcome IS NOT NULL
            AND official_resolved_at IS NOT NULL
            AND official_winning_token_id IS NOT NULL
            AND official_resolution_source IS NOT NULL
            AND official_resolution_received_at IS NOT NULL
            AND official_resolution_payload IS NOT NULL)
        ),
        CONSTRAINT chk_btc_official_resolution_winner CHECK (
          official_outcome IS NULL
          OR (official_outcome = 'up' AND official_winning_token_id = up_token_id)
          OR (official_outcome = 'down' AND official_winning_token_id = down_token_id)
        ),
        CONSTRAINT chk_btc_official_resolution_time CHECK (
          official_resolved_at IS NULL
          OR (
            official_resolved_at >= window_end
            AND official_resolution_received_at >= window_end
          )
        ),
        CONSTRAINT chk_btc_official_resolution_provenance CHECK (
          official_resolution_source IS NULL
          OR official_resolution_source = 'clob_rest_reconciliation'
        ),
        CONSTRAINT chk_btc_official_resolution_payload CHECK (
          official_resolution_payload IS NULL
          OR jsonb_typeof(official_resolution_payload) = 'object'
        )
      );

      CREATE UNIQUE INDEX IF NOT EXISTS uq_btc_interval_condition_id
        ON polymarket.btc_interval_markets (condition_id);
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.btc_interval_markets
        ADD COLUMN IF NOT EXISTS official_outcome text,
        ADD COLUMN IF NOT EXISTS official_resolved_at timestamptz,
        ADD COLUMN IF NOT EXISTS official_winning_token_id text,
        ADD COLUMN IF NOT EXISTS official_resolution_source text,
        ADD COLUMN IF NOT EXISTS official_resolution_received_at timestamptz,
        ADD COLUMN IF NOT EXISTS official_resolution_payload jsonb;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.backfill_jobs
        DROP CONSTRAINT IF EXISTS chk_poly_backfill_jobs_type,
        DROP CONSTRAINT IF EXISTS chk_poly_backfill_jobs_request;

      ALTER TABLE polymarket.backfill_jobs
        RENAME COLUMN job_type TO ingester_key;

      ALTER TABLE polymarket.backfill_jobs
        ALTER COLUMN lookback_days DROP NOT NULL,
        ALTER COLUMN lookback_days DROP DEFAULT,
        ALTER COLUMN min_trade_usd DROP NOT NULL,
        ALTER COLUMN min_trade_usd DROP DEFAULT,
        ADD COLUMN request_version integer NOT NULL DEFAULT 1,
        ADD COLUMN range_start timestamptz,
        ADD COLUMN range_end timestamptz,
        ADD COLUMN idempotency_key text,
        ADD COLUMN progress jsonb NOT NULL DEFAULT '{}'::jsonb,
        ADD COLUMN checkpoint jsonb NOT NULL DEFAULT '{}'::jsonb,
        ADD COLUMN attempt integer NOT NULL DEFAULT 0,
        ADD COLUMN max_attempts integer NOT NULL DEFAULT 3,
        ADD COLUMN next_attempt_at timestamptz NOT NULL DEFAULT now(),
        ADD COLUMN worker_id text,
        ADD COLUMN lease_token uuid,
        ADD COLUMN lease_expires_at timestamptz,
        ADD COLUMN heartbeat_at timestamptz,
        ADD CONSTRAINT chk_poly_backfill_jobs_ingester_key
          CHECK (length(btrim(ingester_key)) > 0),
        ADD CONSTRAINT chk_poly_backfill_jobs_request_version
          CHECK (request_version >= 1),
        ADD CONSTRAINT chk_poly_backfill_jobs_range
          CHECK (
            (range_start IS NULL AND range_end IS NULL)
            OR
            (range_start IS NOT NULL AND range_end IS NOT NULL AND range_end > range_start)
          ),
        ADD CONSTRAINT chk_poly_backfill_jobs_idempotency_key
          CHECK (idempotency_key IS NULL OR length(btrim(idempotency_key)) > 0),
        ADD CONSTRAINT chk_poly_backfill_jobs_documents
          CHECK (
            jsonb_typeof(request) = 'object'
            AND jsonb_typeof(progress) = 'object'
            AND jsonb_typeof(checkpoint) = 'object'
            AND jsonb_typeof(summary) = 'object'
          ),
        ADD CONSTRAINT chk_poly_backfill_jobs_attempts
          CHECK (attempt >= 0 AND max_attempts > 0 AND attempt <= max_attempts),
        ADD CONSTRAINT chk_poly_backfill_jobs_legacy_request
          CHECK (
            (lookback_days IS NULL OR lookback_days >= 0)
            AND (min_trade_usd IS NULL OR min_trade_usd >= 0)
          );
    `);

    await queryRunner.query(`
      CREATE UNIQUE INDEX uq_poly_backfill_jobs_idempotency
        ON polymarket.backfill_jobs (ingester_key, idempotency_key)
        WHERE idempotency_key IS NOT NULL;

      CREATE INDEX idx_poly_backfill_jobs_claim
        ON polymarket.backfill_jobs (next_attempt_at, requested_at, job_id)
        WHERE status = 'queued';

      CREATE INDEX idx_poly_backfill_jobs_expired_lease
        ON polymarket.backfill_jobs (lease_expires_at, requested_at, job_id)
        WHERE status IN ('running', 'cancel_requested');
    `);

    await queryRunner.query(`
      CREATE TABLE polymarket.backfill_artifacts (
        artifact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id uuid NOT NULL
          REFERENCES polymarket.backfill_jobs (job_id) ON DELETE RESTRICT,
        ingester_key text NOT NULL,
        logical_key text NOT NULL,
        provider text NOT NULL,
        source_uri text NOT NULL,
        source_date date,
        checksum_algorithm text NOT NULL DEFAULT 'sha256',
        expected_checksum text,
        actual_checksum text,
        compressed_bytes bigint,
        record_count bigint,
        minimum_source_timestamp timestamptz,
        maximum_source_timestamp timestamptz,
        status text NOT NULL DEFAULT 'pending',
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        completed_at timestamptz,
        CONSTRAINT uq_poly_backfill_artifacts_logical_source
          UNIQUE (provider, logical_key),
        CONSTRAINT chk_poly_backfill_artifacts_identity CHECK (
          length(btrim(ingester_key)) > 0
          AND length(btrim(logical_key)) > 0
          AND length(btrim(provider)) > 0
          AND length(btrim(source_uri)) > 0
        ),
        CONSTRAINT chk_poly_backfill_artifacts_status CHECK (
          status IN (
            'pending', 'downloading', 'downloaded', 'verified',
            'ingesting', 'completed', 'failed'
          )
        ),
        CONSTRAINT chk_poly_backfill_artifacts_checksum_algorithm CHECK (
          checksum_algorithm = 'sha256'
        ),
        CONSTRAINT chk_poly_backfill_artifacts_expected_checksum CHECK (
          expected_checksum IS NULL OR expected_checksum ~ '^[0-9a-f]{64}$'
        ),
        CONSTRAINT chk_poly_backfill_artifacts_actual_checksum CHECK (
          actual_checksum IS NULL OR actual_checksum ~ '^[0-9a-f]{64}$'
        ),
        CONSTRAINT chk_poly_backfill_artifacts_counts CHECK (
          (compressed_bytes IS NULL OR compressed_bytes >= 0)
          AND (record_count IS NULL OR record_count >= 0)
        ),
        CONSTRAINT chk_poly_backfill_artifacts_source_range CHECK (
          (minimum_source_timestamp IS NULL AND maximum_source_timestamp IS NULL)
          OR
          (
            minimum_source_timestamp IS NOT NULL
            AND maximum_source_timestamp IS NOT NULL
            AND maximum_source_timestamp >= minimum_source_timestamp
          )
        ),
        CONSTRAINT chk_poly_backfill_artifacts_metadata CHECK (
          jsonb_typeof(metadata) = 'object'
        ),
        CONSTRAINT chk_poly_backfill_artifacts_completion CHECK (
          (status = 'completed'
            AND actual_checksum IS NOT NULL
            AND record_count IS NOT NULL
            AND completed_at IS NOT NULL)
          OR
          (status <> 'completed' AND completed_at IS NULL)
        ),
        CONSTRAINT chk_poly_backfill_artifacts_times CHECK (
          updated_at >= created_at
          AND (completed_at IS NULL OR completed_at >= created_at)
        )
      );

      CREATE INDEX idx_poly_backfill_artifacts_job
        ON polymarket.backfill_artifacts (job_id, created_at, artifact_id);

      CREATE INDEX idx_poly_backfill_artifacts_status
        ON polymarket.backfill_artifacts (status, updated_at, artifact_id);
    `);

    await queryRunner.query(`
      CREATE FUNCTION polymarket.reject_completed_backfill_artifact_change()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        IF TG_OP = 'DELETE' THEN
          IF OLD.status = 'completed' THEN
            RAISE EXCEPTION
              'completed backfill artifact % is immutable', OLD.artifact_id
              USING ERRCODE = 'integrity_constraint_violation';
          END IF;
          RETURN OLD;
        END IF;
        IF OLD.status = 'completed' AND NEW IS DISTINCT FROM OLD THEN
          RAISE EXCEPTION
            'completed backfill artifact % is immutable', OLD.artifact_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;
        RETURN NEW;
      END;
      $$;

      CREATE TRIGGER trg_reject_completed_backfill_artifact_change
        BEFORE UPDATE OR DELETE ON polymarket.backfill_artifacts
        FOR EACH ROW
        EXECUTE FUNCTION polymarket.reject_completed_backfill_artifact_change();
    `);

    await queryRunner.query(`
      CREATE TABLE polymarket.btc_market_reference_facts (
        fact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        market_id text NOT NULL
          REFERENCES polymarket.btc_interval_markets (market_id) ON DELETE RESTRICT,
        artifact_id uuid NOT NULL
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        fact_type text NOT NULL,
        value numeric(30,10) NOT NULL,
        provider text NOT NULL,
        source_effective_at timestamptz NOT NULL,
        fetched_at timestamptz NOT NULL,
        payload_sha256 text NOT NULL,
        evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT uq_btc_market_reference_fact
          UNIQUE (market_id, fact_type, provider),
        CONSTRAINT chk_btc_market_reference_fact_type CHECK (
          fact_type IN ('opening_boundary', 'final_price')
        ),
        CONSTRAINT chk_btc_market_reference_fact_value CHECK (value > 0),
        CONSTRAINT chk_btc_market_reference_fact_provider CHECK (
          length(btrim(provider)) > 0
        ),
        CONSTRAINT chk_btc_market_reference_fact_payload_sha256 CHECK (
          payload_sha256 ~ '^[0-9a-f]{64}$'
        ),
        CONSTRAINT chk_btc_market_reference_fact_evidence CHECK (
          jsonb_typeof(evidence) = 'object'
        ),
        CONSTRAINT chk_btc_market_reference_fact_times CHECK (
          fetched_at >= source_effective_at
        )
      );

      CREATE INDEX idx_btc_market_reference_facts_artifact
        ON polymarket.btc_market_reference_facts (artifact_id, market_id, fact_type);

      CREATE INDEX idx_btc_market_reference_facts_effective
        ON polymarket.btc_market_reference_facts (
          fact_type, source_effective_at, market_id
        );
    `);

    await queryRunner.query(`
      CREATE FUNCTION polymarket.reject_immutable_btc_reference_fact_change()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        RAISE EXCEPTION
          'BTC market reference fact % is immutable', OLD.fact_id
          USING ERRCODE = 'integrity_constraint_violation';
      END;
      $$;

      CREATE TRIGGER trg_reject_immutable_btc_reference_fact_change
        BEFORE UPDATE OR DELETE ON polymarket.btc_market_reference_facts
        FOR EACH ROW
        EXECUTE FUNCTION polymarket.reject_immutable_btc_reference_fact_change();
    `);

    await queryRunner.query(`
      CREATE TABLE polymarket.binance_aggregate_trades (
        symbol text NOT NULL,
        trade_timestamp timestamptz NOT NULL,
        aggregate_trade_id bigint NOT NULL,
        price numeric(30,10) NOT NULL,
        quantity numeric(30,10) NOT NULL,
        first_trade_id bigint NOT NULL,
        last_trade_id bigint NOT NULL,
        buyer_maker boolean NOT NULL,
        best_match boolean NOT NULL,
        artifact_id uuid NOT NULL
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_binance_aggregate_trades
          PRIMARY KEY (symbol, trade_timestamp, aggregate_trade_id),
        CONSTRAINT chk_binance_aggregate_trades_symbol CHECK (
          length(btrim(symbol)) > 0
        ),
        CONSTRAINT chk_binance_aggregate_trades_values CHECK (
          price > 0 AND quantity > 0
        ),
        CONSTRAINT chk_binance_aggregate_trades_ids CHECK (
          aggregate_trade_id >= 0
          AND first_trade_id >= 0
          AND last_trade_id >= first_trade_id
        )
      );
    `);

    await queryRunner.query(`
      SELECT create_hypertable(
        'polymarket.binance_aggregate_trades',
        'trade_timestamp',
        chunk_time_interval => INTERVAL '1 day',
        if_not_exists => TRUE
      );

      ALTER TABLE polymarket.binance_aggregate_trades SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'trade_timestamp ASC, aggregate_trade_id ASC',
        timescaledb.compress_segmentby = 'symbol, artifact_id'
      );

      CREATE INDEX idx_binance_aggregate_trades_artifact
        ON polymarket.binance_aggregate_trades (
          artifact_id, trade_timestamp, aggregate_trade_id
        );

      CREATE INDEX idx_binance_aggregate_trades_identity
        ON polymarket.binance_aggregate_trades (
          symbol, aggregate_trade_id, trade_timestamp
        );
    `);

    await queryRunner.query(`
      CREATE TABLE polymarket.binance_one_second_klines (
        symbol text NOT NULL,
        open_timestamp timestamptz NOT NULL,
        close_timestamp timestamptz NOT NULL,
        open_price numeric(30,10) NOT NULL,
        high_price numeric(30,10) NOT NULL,
        low_price numeric(30,10) NOT NULL,
        close_price numeric(30,10) NOT NULL,
        base_volume numeric(30,10) NOT NULL,
        quote_volume numeric(30,10) NOT NULL,
        trade_count bigint NOT NULL,
        taker_buy_base_volume numeric(30,10) NOT NULL,
        taker_buy_quote_volume numeric(30,10) NOT NULL,
        artifact_id uuid NOT NULL
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_binance_one_second_klines
          PRIMARY KEY (symbol, open_timestamp),
        CONSTRAINT chk_binance_one_second_klines_symbol CHECK (
          length(btrim(symbol)) > 0
        ),
        CONSTRAINT chk_binance_one_second_klines_window CHECK (
          close_timestamp >= open_timestamp
          AND close_timestamp < open_timestamp + INTERVAL '1 second'
        ),
        CONSTRAINT chk_binance_one_second_klines_prices CHECK (
          open_price > 0
          AND high_price > 0
          AND low_price > 0
          AND close_price > 0
          AND high_price >= open_price
          AND high_price >= low_price
          AND high_price >= close_price
          AND low_price <= open_price
          AND low_price <= high_price
          AND low_price <= close_price
        ),
        CONSTRAINT chk_binance_one_second_klines_volumes CHECK (
          base_volume >= 0
          AND quote_volume >= 0
          AND trade_count >= 0
          AND taker_buy_base_volume >= 0
          AND taker_buy_quote_volume >= 0
          AND taker_buy_base_volume <= base_volume
          AND taker_buy_quote_volume <= quote_volume
        )
      );
    `);

    await queryRunner.query(`
      SELECT create_hypertable(
        'polymarket.binance_one_second_klines',
        'open_timestamp',
        chunk_time_interval => INTERVAL '1 day',
        if_not_exists => TRUE
      );

      ALTER TABLE polymarket.binance_one_second_klines SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'open_timestamp ASC',
        timescaledb.compress_segmentby = 'symbol, artifact_id'
      );

      CREATE INDEX idx_binance_one_second_klines_artifact
        ON polymarket.binance_one_second_klines (artifact_id, open_timestamp);
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (SELECT 1 FROM polymarket.backfill_artifacts)
          OR EXISTS (SELECT 1 FROM polymarket.btc_interval_markets)
          OR EXISTS (SELECT 1 FROM polymarket.btc_market_reference_facts)
          OR EXISTS (SELECT 1 FROM polymarket.binance_aggregate_trades)
          OR EXISTS (SELECT 1 FROM polymarket.binance_one_second_klines)
          OR EXISTS (
            SELECT 1
            FROM polymarket.backfill_jobs
            WHERE ingester_key <> 'whales'
               OR lookback_days IS NULL
               OR min_trade_usd IS NULL
          )
        THEN
          RAISE EXCEPTION
            'refusing to roll back historical ingestion schema while ingestion data exists';
        END IF;
      END $$;
    `);

    await queryRunner.query(`
      DROP TABLE polymarket.binance_one_second_klines;
      DROP TABLE polymarket.binance_aggregate_trades;

      DROP TRIGGER trg_reject_immutable_btc_reference_fact_change
        ON polymarket.btc_market_reference_facts;
      DROP TABLE polymarket.btc_market_reference_facts;
      DROP FUNCTION polymarket.reject_immutable_btc_reference_fact_change();

      DROP TABLE polymarket.btc_interval_markets;

      DROP TRIGGER trg_reject_completed_backfill_artifact_change
        ON polymarket.backfill_artifacts;
      DROP TABLE polymarket.backfill_artifacts;
      DROP FUNCTION polymarket.reject_completed_backfill_artifact_change();
    `);

    await queryRunner.query(`
      DROP INDEX IF EXISTS polymarket.idx_poly_backfill_jobs_expired_lease;
      DROP INDEX IF EXISTS polymarket.idx_poly_backfill_jobs_claim;
      DROP INDEX IF EXISTS polymarket.uq_poly_backfill_jobs_idempotency;

      ALTER TABLE polymarket.backfill_jobs
        DROP CONSTRAINT IF EXISTS chk_poly_backfill_jobs_legacy_request,
        DROP CONSTRAINT IF EXISTS chk_poly_backfill_jobs_attempts,
        DROP CONSTRAINT IF EXISTS chk_poly_backfill_jobs_documents,
        DROP CONSTRAINT IF EXISTS chk_poly_backfill_jobs_idempotency_key,
        DROP CONSTRAINT IF EXISTS chk_poly_backfill_jobs_range,
        DROP CONSTRAINT IF EXISTS chk_poly_backfill_jobs_request_version,
        DROP CONSTRAINT IF EXISTS chk_poly_backfill_jobs_ingester_key,
        DROP COLUMN heartbeat_at,
        DROP COLUMN lease_expires_at,
        DROP COLUMN lease_token,
        DROP COLUMN worker_id,
        DROP COLUMN next_attempt_at,
        DROP COLUMN max_attempts,
        DROP COLUMN attempt,
        DROP COLUMN checkpoint,
        DROP COLUMN progress,
        DROP COLUMN idempotency_key,
        DROP COLUMN range_end,
        DROP COLUMN range_start,
        DROP COLUMN request_version;

      ALTER TABLE polymarket.backfill_jobs
        ALTER COLUMN lookback_days SET DEFAULT 30,
        ALTER COLUMN lookback_days SET NOT NULL,
        ALTER COLUMN min_trade_usd SET DEFAULT 1000,
        ALTER COLUMN min_trade_usd SET NOT NULL;

      ALTER TABLE polymarket.backfill_jobs
        RENAME COLUMN ingester_key TO job_type;

      ALTER TABLE polymarket.backfill_jobs
        ADD CONSTRAINT chk_poly_backfill_jobs_type
          CHECK (job_type IN ('whales')),
        ADD CONSTRAINT chk_poly_backfill_jobs_request
          CHECK (lookback_days >= 0 AND min_trade_usd >= 0);
    `);
  }
}
