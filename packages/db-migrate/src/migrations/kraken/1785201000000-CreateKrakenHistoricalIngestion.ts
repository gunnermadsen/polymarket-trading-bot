import { MigrationInterface, QueryRunner } from 'typeorm';

export class CreateKrakenHistoricalIngestion1785201000000 implements MigrationInterface {
  name = 'CreateKrakenHistoricalIngestion1785201000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;`);
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`
      CREATE SCHEMA kraken;

      CREATE TABLE kraken.backfill_jobs (
        job_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        dataset text NOT NULL,
        symbol text NOT NULL,
        interval_seconds integer NOT NULL,
        range_start timestamptz NOT NULL,
        range_end timestamptz NOT NULL,
        status text NOT NULL DEFAULT 'queued',
        attempt integer NOT NULL DEFAULT 0,
        max_attempts integer NOT NULL DEFAULT 8,
        next_attempt_at timestamptz NOT NULL DEFAULT now(),
        worker_id text,
        lease_token uuid,
        lease_expires_at timestamptz,
        expected_work_units bigint NOT NULL,
        processed_work_units bigint NOT NULL DEFAULT 0,
        rows_written bigint NOT NULL DEFAULT 0,
        error_message text,
        requested_at timestamptz NOT NULL DEFAULT now(),
        started_at timestamptz,
        completed_at timestamptz,
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_kraken_backfill_dataset CHECK (
          dataset IN (
            'instruments',
            'fee_schedules',
            'trade_candles',
            'mark_candles',
            'spot_candles',
            'open_interest',
            'future_basis',
            'aggressor_differential',
            'trade_volume',
            'trade_count',
            'cvd',
            'liquidation_volume',
            'spreads',
            'liquidity',
            'slippage',
            'funding_rates'
          )
        ),
        CONSTRAINT chk_kraken_backfill_symbol CHECK (symbol ~ '^[A-Z0-9_]+$'),
        CONSTRAINT chk_kraken_backfill_interval CHECK (interval_seconds > 0),
        CONSTRAINT chk_kraken_backfill_range CHECK (range_start < range_end),
        CONSTRAINT chk_kraken_backfill_status CHECK (
          status IN ('queued','running','completed','failed')
        ),
        CONSTRAINT chk_kraken_backfill_attempts CHECK (
          attempt >= 0 AND max_attempts > 0 AND attempt <= max_attempts
        ),
        CONSTRAINT chk_kraken_backfill_progress CHECK (
          expected_work_units >= 0
          AND processed_work_units >= 0
          AND rows_written >= 0
        ),
        CONSTRAINT uq_kraken_backfill_identity UNIQUE (
          dataset, symbol, interval_seconds, range_start, range_end
        )
      );

      CREATE INDEX idx_kraken_backfill_claim
        ON kraken.backfill_jobs (status, next_attempt_at, requested_at, job_id);

      CREATE INDEX idx_kraken_backfill_expired_lease
        ON kraken.backfill_jobs (lease_expires_at, requested_at, job_id)
        WHERE status = 'running';

      CREATE TABLE kraken.backfill_job_events (
        event_id uuid NOT NULL DEFAULT gen_random_uuid(),
        job_id uuid NOT NULL REFERENCES kraken.backfill_jobs (job_id) ON DELETE CASCADE,
        recorded_at timestamptz NOT NULL DEFAULT now(),
        level text NOT NULL,
        message text NOT NULL,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT pk_kraken_backfill_job_events PRIMARY KEY (event_id, recorded_at),
        CONSTRAINT chk_kraken_backfill_event_level CHECK (
          level IN ('debug','info','warn','error')
        ),
        CONSTRAINT chk_kraken_backfill_event_metadata CHECK (
          jsonb_typeof(metadata) = 'object'
        )
      );

      SELECT create_hypertable(
        'kraken.backfill_job_events',
        'recorded_at',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
      );

      CREATE INDEX idx_kraken_backfill_events_job_time
        ON kraken.backfill_job_events (job_id, recorded_at DESC);

      CREATE TABLE kraken.worker_status (
        worker_id text PRIMARY KEY,
        state text NOT NULL DEFAULT 'idle',
        current_job_id uuid REFERENCES kraken.backfill_jobs (job_id) ON DELETE SET NULL,
        jobs_completed bigint NOT NULL DEFAULT 0,
        jobs_failed bigint NOT NULL DEFAULT 0,
        rows_written bigint NOT NULL DEFAULT 0,
        started_at timestamptz NOT NULL DEFAULT now(),
        heartbeat_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_kraken_worker_state CHECK (state IN ('idle','running','stopped')),
        CONSTRAINT chk_kraken_worker_counters CHECK (
          jobs_completed >= 0 AND jobs_failed >= 0 AND rows_written >= 0
        )
      );

      CREATE TABLE kraken.api_request_budget (
        budget_key text PRIMARY KEY,
        next_request_at timestamptz NOT NULL DEFAULT now(),
        spacing_milliseconds integer NOT NULL,
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_kraken_api_spacing CHECK (
          spacing_milliseconds BETWEEN 50 AND 60000
        )
      );

      INSERT INTO kraken.api_request_budget (
        budget_key, spacing_milliseconds
      ) VALUES ('futures_public_api', 350);

      CREATE TABLE kraken.backfill_artifacts (
        artifact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id uuid NOT NULL UNIQUE
          REFERENCES kraken.backfill_jobs (job_id) ON DELETE RESTRICT,
        provider text NOT NULL,
        source_url text NOT NULL,
        sha256 text NOT NULL,
        lake_relative_path text NOT NULL UNIQUE,
        row_count bigint NOT NULL,
        source_min_time timestamptz,
        source_max_time timestamptz,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_kraken_artifact_provider CHECK (provider = 'kraken_futures'),
        CONSTRAINT chk_kraken_artifact_hash CHECK (sha256 ~ '^[0-9a-f]{64}$'),
        CONSTRAINT chk_kraken_artifact_path CHECK (
          lake_relative_path <> ''
          AND lake_relative_path NOT LIKE '/%'
          AND lake_relative_path NOT LIKE '%..%'
        ),
        CONSTRAINT chk_kraken_artifact_rows CHECK (row_count >= 0),
        CONSTRAINT chk_kraken_artifact_range CHECK (
          source_min_time IS NULL
          OR source_max_time IS NULL
          OR source_min_time <= source_max_time
        ),
        CONSTRAINT chk_kraken_artifact_metadata CHECK (
          jsonb_typeof(metadata) = 'object'
        )
      );

      CREATE TABLE kraken.parquet_objects (
        object_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        artifact_id uuid NOT NULL UNIQUE
          REFERENCES kraken.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        dataset text NOT NULL,
        symbol text NOT NULL,
        interval_seconds integer NOT NULL,
        lake_relative_path text NOT NULL UNIQUE,
        sha256 text NOT NULL,
        byte_size bigint NOT NULL,
        row_count bigint NOT NULL,
        schema_version integer NOT NULL DEFAULT 1,
        published_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_kraken_parquet_hash CHECK (sha256 ~ '^[0-9a-f]{64}$'),
        CONSTRAINT chk_kraken_parquet_counts CHECK (
          byte_size > 0 AND row_count >= 0 AND schema_version > 0
        )
      );

      CREATE TABLE kraken.instruments (
        symbol text PRIMARY KEY,
        instrument_type text NOT NULL,
        tradeable boolean NOT NULL,
        tick_size numeric(38,18),
        contract_size numeric(38,18),
        base_currency text,
        quote_currency text,
        pair text,
        contract_value_trade_precision integer,
        max_position_size numeric(38,18),
        funding_rate_coefficient numeric(38,18),
        max_relative_funding_rate numeric(38,18),
        fee_schedule_uid text,
        margin_levels jsonb NOT NULL,
        retail_margin_levels jsonb NOT NULL,
        margin_schedules jsonb NOT NULL,
        raw_payload jsonb NOT NULL,
        artifact_id uuid NOT NULL
          REFERENCES kraken.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        source_observed_at timestamptz NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_kraken_instrument_symbol CHECK (symbol ~ '^[A-Z0-9_]+$'),
        CONSTRAINT chk_kraken_instrument_values CHECK (
          (tick_size IS NULL OR tick_size > 0)
          AND (contract_size IS NULL OR contract_size > 0)
          AND (
            contract_value_trade_precision IS NULL
            OR contract_value_trade_precision >= 0
          )
          AND (max_position_size IS NULL OR max_position_size > 0)
          AND (
            max_relative_funding_rate IS NULL
            OR max_relative_funding_rate >= 0
          )
        ),
        CONSTRAINT chk_kraken_instrument_margin_payloads CHECK (
          jsonb_typeof(margin_levels) = 'array'
          AND jsonb_typeof(retail_margin_levels) = 'array'
          AND jsonb_typeof(margin_schedules) = 'object'
        ),
        CONSTRAINT chk_kraken_instrument_payload CHECK (
          jsonb_typeof(raw_payload) = 'object'
        )
      );

      CREATE TABLE kraken.fee_schedules (
        fee_schedule_uid text PRIMARY KEY,
        name text NOT NULL,
        tiers jsonb NOT NULL,
        raw_payload jsonb NOT NULL,
        artifact_id uuid NOT NULL
          REFERENCES kraken.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        source_observed_at timestamptz NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_kraken_fee_schedule_tiers CHECK (
          jsonb_typeof(tiers) = 'array'
        ),
        CONSTRAINT chk_kraken_fee_schedule_payload CHECK (
          jsonb_typeof(raw_payload) = 'object'
        )
      );

      CREATE TABLE kraken.market_candles (
        symbol text NOT NULL,
        candle_kind text NOT NULL,
        interval_seconds integer NOT NULL,
        bucket_start timestamptz NOT NULL,
        open numeric(38,18) NOT NULL,
        high numeric(38,18) NOT NULL,
        low numeric(38,18) NOT NULL,
        close numeric(38,18) NOT NULL,
        volume numeric(38,18) NOT NULL,
        artifact_id uuid NOT NULL
          REFERENCES kraken.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_kraken_market_candles PRIMARY KEY (
          symbol, candle_kind, interval_seconds, bucket_start
        ),
        CONSTRAINT chk_kraken_candle_kind CHECK (
          candle_kind IN ('trade','mark','spot')
        ),
        CONSTRAINT chk_kraken_candle_values CHECK (
          interval_seconds > 0
          AND low >= 0
          AND open >= low AND open <= high
          AND close >= low AND close <= high
          AND volume >= 0
        )
      );

      SELECT create_hypertable(
        'kraken.market_candles',
        'bucket_start',
        chunk_time_interval => INTERVAL '30 days',
        if_not_exists => TRUE
      );

      ALTER TABLE kraken.market_candles SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'bucket_start ASC',
        timescaledb.compress_segmentby =
          'symbol, candle_kind, interval_seconds, artifact_id'
      );

      CREATE TABLE kraken.market_analytics (
        symbol text NOT NULL,
        dataset text NOT NULL,
        interval_seconds integer NOT NULL,
        bucket_start timestamptz NOT NULL,
        values jsonb NOT NULL,
        artifact_id uuid NOT NULL
          REFERENCES kraken.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_kraken_market_analytics PRIMARY KEY (
          symbol, dataset, interval_seconds, bucket_start
        ),
        CONSTRAINT chk_kraken_analytics_dataset CHECK (
          dataset IN (
            'open_interest',
            'future_basis',
            'aggressor_differential',
            'trade_volume',
            'trade_count',
            'cvd',
            'liquidation_volume',
            'spreads',
            'liquidity',
            'slippage'
          )
        ),
        CONSTRAINT chk_kraken_analytics_values CHECK (
          interval_seconds > 0
          AND jsonb_typeof(values) IN ('object','array','number','string')
        )
      );

      SELECT create_hypertable(
        'kraken.market_analytics',
        'bucket_start',
        chunk_time_interval => INTERVAL '30 days',
        if_not_exists => TRUE
      );

      ALTER TABLE kraken.market_analytics SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'bucket_start ASC',
        timescaledb.compress_segmentby =
          'symbol, dataset, interval_seconds, artifact_id'
      );

      CREATE TABLE kraken.funding_rates (
        symbol text NOT NULL,
        funding_time timestamptz NOT NULL,
        funding_rate numeric(38,18) NOT NULL,
        relative_funding_rate numeric(38,18) NOT NULL,
        artifact_id uuid NOT NULL
          REFERENCES kraken.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_kraken_funding_rates PRIMARY KEY (symbol, funding_time)
      );

      SELECT create_hypertable(
        'kraken.funding_rates',
        'funding_time',
        chunk_time_interval => INTERVAL '90 days',
        if_not_exists => TRUE
      );

      ALTER TABLE kraken.funding_rates SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'funding_time ASC',
        timescaledb.compress_segmentby = 'symbol, artifact_id'
      );

      CREATE TABLE kraken.ingestion_coverage (
        dataset text NOT NULL,
        symbol text NOT NULL,
        interval_seconds integer NOT NULL,
        earliest_observation timestamptz,
        latest_observation timestamptz,
        observed_rows bigint NOT NULL DEFAULT 0,
        missing_buckets bigint,
        refreshed_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_kraken_ingestion_coverage PRIMARY KEY (
          dataset, symbol, interval_seconds
        ),
        CONSTRAINT chk_kraken_coverage_counts CHECK (
          observed_rows >= 0 AND (missing_buckets IS NULL OR missing_buckets >= 0)
        ),
        CONSTRAINT chk_kraken_coverage_range CHECK (
          earliest_observation IS NULL
          OR latest_observation IS NULL
          OR earliest_observation <= latest_observation
        )
      );
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP SCHEMA kraken CASCADE;`);
  }
}
