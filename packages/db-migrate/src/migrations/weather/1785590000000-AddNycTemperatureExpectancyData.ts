import { MigrationInterface, QueryRunner } from 'typeorm';

const PROCESS_ID = 'd48e2b47-18df-4c8c-95b9-0f12e6ca7d41';

export class AddNycTemperatureExpectancyData1785590000000 implements MigrationInterface {
  name = 'AddNycTemperatureExpectancyData1785590000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS polymarket;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS weather;`);

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
        started_at timestamptz,
        heartbeat_at timestamptz,
        stopped_at timestamptz,
        stop_reason text,
        last_error text,
        config jsonb NOT NULL DEFAULT '{}'::jsonb,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_weather_process_name CHECK (length(btrim(name)) > 0),
        CONSTRAINT chk_weather_process_type CHECK (length(btrim(process_type)) > 0),
        CONSTRAINT chk_weather_process_scope CHECK (length(btrim(process_scope)) > 0),
        CONSTRAINT chk_weather_process_status CHECK (length(btrim(status)) > 0),
        CONSTRAINT chk_weather_process_config CHECK (jsonb_typeof(config) = 'object'),
        CONSTRAINT chk_weather_process_metadata CHECK (jsonb_typeof(metadata) = 'object')
      );

      CREATE UNIQUE INDEX IF NOT EXISTS uq_weather_process_key
        ON polymarket.trading_processes (process_type, process_scope, process_key)
        WHERE process_key IS NOT NULL;
    `);

    await queryRunner.query(`
      INSERT INTO polymarket.trading_processes (
        process_id, name, process_type, process_scope, process_key,
        status, enabled, config, metadata
      ) VALUES (
        '${PROCESS_ID}',
        'NYC temperature expectancy benchmark',
        'offline_model_benchmark',
        'nyc_daily_high_klga',
        'nyc-temperature-expectancy-v1',
        'created',
        false,
        '{
          "station_id":"KLGA",
          "market_family":"highest_temperature_in_nyc",
          "decision_times_local":["00:00","12:00"],
          "side":"buy_yes",
          "quantities":[1,5,10],
          "order_submission_enabled":false
        }'::jsonb,
        '{"purpose":"offline data and economic qualification only"}'::jsonb
      )
      ON CONFLICT (process_id) DO NOTHING;
    `);

    await queryRunner.query(`
      CREATE TABLE weather.ingestion_jobs (
        job_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        depends_on_job_id uuid REFERENCES weather.ingestion_jobs (job_id) ON DELETE RESTRICT,
        ingester_key text NOT NULL,
        idempotency_key text NOT NULL,
        range_start timestamptz NOT NULL,
        range_end timestamptz NOT NULL,
        request jsonb NOT NULL DEFAULT '{}'::jsonb,
        status text NOT NULL DEFAULT 'queued',
        attempt integer NOT NULL DEFAULT 0,
        max_attempts integer NOT NULL DEFAULT 3,
        next_attempt_at timestamptz NOT NULL DEFAULT now(),
        worker_id text,
        lease_token uuid,
        lease_expires_at timestamptz,
        heartbeat_at timestamptz,
        progress jsonb NOT NULL DEFAULT '{}'::jsonb,
        summary jsonb NOT NULL DEFAULT '{}'::jsonb,
        error text,
        requested_at timestamptz NOT NULL DEFAULT now(),
        started_at timestamptz,
        completed_at timestamptz,
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT uq_weather_ingestion_job UNIQUE (ingester_key, idempotency_key),
        CONSTRAINT chk_weather_ingestion_range CHECK (range_end > range_start),
        CONSTRAINT chk_weather_ingestion_status CHECK (
          status IN ('queued','running','completed','failed','cancel_requested','cancelled')
        ),
        CONSTRAINT chk_weather_ingestion_attempt CHECK (
          attempt >= 0 AND max_attempts > 0 AND attempt <= max_attempts
        ),
        CONSTRAINT chk_weather_ingestion_documents CHECK (
          jsonb_typeof(request) = 'object'
          AND jsonb_typeof(progress) = 'object'
          AND jsonb_typeof(summary) = 'object'
        )
      );

      CREATE INDEX idx_weather_ingestion_claim
        ON weather.ingestion_jobs (next_attempt_at, requested_at, job_id)
        WHERE status = 'queued';
      CREATE INDEX idx_weather_ingestion_dependency
        ON weather.ingestion_jobs (depends_on_job_id, status)
        WHERE depends_on_job_id IS NOT NULL;
      CREATE INDEX idx_weather_ingestion_lease
        ON weather.ingestion_jobs (lease_expires_at, job_id)
        WHERE status IN ('running','cancel_requested');

      CREATE TABLE weather.source_artifacts (
        artifact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        provider text NOT NULL,
        logical_key text NOT NULL,
        source_uri text NOT NULL,
        source_start timestamptz,
        source_end timestamptz,
        sha256 text,
        compressed_bytes bigint,
        record_count bigint NOT NULL DEFAULT 0,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT uq_weather_source_artifact UNIQUE (provider, logical_key),
        CONSTRAINT chk_weather_source_sha256 CHECK (
          sha256 IS NULL OR sha256 ~ '^[0-9a-f]{64}$'
        ),
        CONSTRAINT chk_weather_source_counts CHECK (
          (compressed_bytes IS NULL OR compressed_bytes >= 0) AND record_count >= 0
        ),
        CONSTRAINT chk_weather_source_metadata CHECK (jsonb_typeof(metadata) = 'object')
      );

      CREATE TABLE weather.temperature_markets (
        market_id text PRIMARY KEY,
        event_id text NOT NULL,
        event_slug text NOT NULL,
        market_slug text NOT NULL UNIQUE,
        event_date date NOT NULL,
        station_id text NOT NULL DEFAULT 'KLGA',
        question text NOT NULL,
        condition_id text NOT NULL UNIQUE,
        yes_token_id text NOT NULL,
        no_token_id text NOT NULL,
        bucket_lower_f integer,
        bucket_upper_f integer,
        active boolean NOT NULL,
        closed boolean NOT NULL,
        accepting_orders boolean NOT NULL,
        volume_usd numeric(30,10),
        liquidity_usd numeric(30,10),
        resolved_yes boolean,
        resolved_at timestamptz,
        resolution_source text,
        fee_rate_bps integer,
        source_artifact_id uuid REFERENCES weather.source_artifacts (artifact_id) ON DELETE RESTRICT,
        raw_payload jsonb NOT NULL,
        discovered_at timestamptz NOT NULL DEFAULT now(),
        refreshed_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_weather_market_tokens CHECK (yes_token_id <> no_token_id),
        CONSTRAINT chk_weather_market_bucket CHECK (
          bucket_lower_f IS NULL OR bucket_upper_f IS NULL OR bucket_upper_f >= bucket_lower_f
        ),
        CONSTRAINT chk_weather_market_resolution CHECK (
          (resolved_yes IS NULL AND resolved_at IS NULL)
          OR (resolved_yes IS NOT NULL AND resolved_at IS NOT NULL AND resolution_source IS NOT NULL)
        ),
        CONSTRAINT chk_weather_market_payload CHECK (jsonb_typeof(raw_payload) = 'object')
      );

      CREATE INDEX idx_weather_markets_event_date
        ON weather.temperature_markets (event_date, market_id);
      CREATE INDEX idx_weather_markets_tokens
        ON weather.temperature_markets (yes_token_id, no_token_id);

      CREATE TABLE weather.station_observations (
        station_id text NOT NULL,
        observed_at timestamptz NOT NULL,
        temperature_f numeric(8,3),
        report_type integer,
        provider text NOT NULL,
        source_artifact_id uuid REFERENCES weather.source_artifacts (artifact_id) ON DELETE RESTRICT,
        quality_flags jsonb NOT NULL DEFAULT '[]'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (station_id, observed_at, provider),
        CONSTRAINT chk_weather_observation_flags CHECK (jsonb_typeof(quality_flags) = 'array')
      );

      CREATE INDEX idx_weather_observations_station_time
        ON weather.station_observations (station_id, observed_at DESC);

      CREATE TABLE weather.hrrr_point_forecasts (
        station_id text NOT NULL,
        decision_time timestamptz NOT NULL,
        model_run timestamptz NOT NULL,
        valid_at timestamptz NOT NULL,
        lead_hours integer NOT NULL,
        temperature_f numeric(8,3) NOT NULL,
        latitude numeric(10,6) NOT NULL,
        longitude numeric(10,6) NOT NULL,
        provider text NOT NULL DEFAULT 'noaa_hrrr_aws',
        source_artifact_id uuid REFERENCES weather.source_artifacts (artifact_id) ON DELETE RESTRICT,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (station_id, decision_time, model_run, valid_at),
        CONSTRAINT chk_weather_hrrr_availability CHECK (model_run < decision_time),
        CONSTRAINT chk_weather_hrrr_lead CHECK (lead_hours >= 0)
      );

      CREATE INDEX idx_weather_hrrr_decision
        ON weather.hrrr_point_forecasts (station_id, decision_time, valid_at);

      CREATE TABLE weather.price_history (
        token_id text NOT NULL,
        observed_at timestamptz NOT NULL,
        price numeric(18,8) NOT NULL,
        evidence_tier text NOT NULL DEFAULT 'indicative',
        source_artifact_id uuid REFERENCES weather.source_artifacts (artifact_id) ON DELETE RESTRICT,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (token_id, observed_at),
        CONSTRAINT chk_weather_price CHECK (price >= 0 AND price <= 1),
        CONSTRAINT chk_weather_price_tier CHECK (evidence_tier = 'indicative')
      );

      CREATE TABLE weather.execution_snapshots (
        market_id text NOT NULL REFERENCES weather.temperature_markets (market_id) ON DELETE RESTRICT,
        decision_time timestamptz NOT NULL,
        quantity numeric(12,4) NOT NULL,
        yes_ask_vwap numeric(18,8),
        no_ask_vwap numeric(18,8),
        yes_best_ask numeric(18,8),
        no_best_ask numeric(18,8),
        source_timestamp timestamptz,
        evidence_tier text NOT NULL DEFAULT 'executable_taker',
        quality_flags jsonb NOT NULL DEFAULT '[]'::jsonb,
        source_artifact_id uuid REFERENCES weather.source_artifacts (artifact_id) ON DELETE RESTRICT,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (market_id, decision_time, quantity),
        CONSTRAINT chk_weather_execution_quantity CHECK (quantity > 0),
        CONSTRAINT chk_weather_execution_prices CHECK (
          (yes_ask_vwap IS NULL OR yes_ask_vwap BETWEEN 0 AND 1)
          AND (no_ask_vwap IS NULL OR no_ask_vwap BETWEEN 0 AND 1)
          AND (yes_best_ask IS NULL OR yes_best_ask BETWEEN 0 AND 1)
          AND (no_best_ask IS NULL OR no_best_ask BETWEEN 0 AND 1)
        ),
        CONSTRAINT chk_weather_execution_tier CHECK (evidence_tier = 'executable_taker'),
        CONSTRAINT chk_weather_execution_flags CHECK (jsonb_typeof(quality_flags) = 'array')
      );

      CREATE INDEX idx_weather_execution_decision
        ON weather.execution_snapshots (decision_time, market_id, quantity);

      CREATE TABLE weather.label_reconciliation (
        process_id uuid NOT NULL REFERENCES polymarket.trading_processes (process_id) ON DELETE RESTRICT,
        event_date date NOT NULL,
        station_id text NOT NULL,
        station_daily_max_f numeric(8,3),
        station_rounded_max_f integer,
        winning_market_id text REFERENCES weather.temperature_markets (market_id) ON DELETE RESTRICT,
        winner_matches_station boolean,
        quality_flags jsonb NOT NULL DEFAULT '[]'::jsonb,
        calculated_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (process_id, event_date),
        CONSTRAINT chk_weather_label_flags CHECK (jsonb_typeof(quality_flags) = 'array')
      );

      CREATE TABLE weather.model_runs (
        model_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        process_id uuid NOT NULL REFERENCES polymarket.trading_processes (process_id) ON DELETE RESTRICT,
        candidate text NOT NULL,
        decision_hour_local integer NOT NULL,
        training_start date NOT NULL,
        training_end date NOT NULL,
        calibration_start date NOT NULL,
        calibration_end date NOT NULL,
        feature_schema_version text NOT NULL,
        model_uri text NOT NULL,
        model_sha256 text NOT NULL,
        metrics jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_weather_model_ranges CHECK (
          training_end >= training_start
          AND calibration_start > training_end
          AND calibration_end >= calibration_start
        ),
        CONSTRAINT chk_weather_model_decision CHECK (decision_hour_local IN (0,12)),
        CONSTRAINT chk_weather_model_sha CHECK (model_sha256 ~ '^[0-9a-f]{64}$'),
        CONSTRAINT chk_weather_model_metrics CHECK (jsonb_typeof(metrics) = 'object')
      );

      CREATE TABLE weather.predictions (
        process_id uuid NOT NULL REFERENCES polymarket.trading_processes (process_id) ON DELETE RESTRICT,
        model_run_id uuid NOT NULL REFERENCES weather.model_runs (model_run_id) ON DELETE RESTRICT,
        market_id text NOT NULL REFERENCES weather.temperature_markets (market_id) ON DELETE RESTRICT,
        decision_time timestamptz NOT NULL,
        probability_yes numeric(18,12) NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (process_id, model_run_id, market_id, decision_time),
        CONSTRAINT chk_weather_prediction_probability CHECK (probability_yes BETWEEN 0 AND 1)
      );

      CREATE INDEX idx_weather_predictions_process_decision
        ON weather.predictions (process_id, decision_time, market_id);

      CREATE TABLE weather.benchmark_runs (
        benchmark_run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        process_id uuid NOT NULL REFERENCES polymarket.trading_processes (process_id) ON DELETE RESTRICT,
        model_run_id uuid NOT NULL REFERENCES weather.model_runs (model_run_id) ON DELETE RESTRICT,
        evaluation_start date NOT NULL,
        evaluation_end date NOT NULL,
        quantity numeric(12,4) NOT NULL,
        evidence_tier text NOT NULL,
        policy jsonb NOT NULL,
        metrics jsonb NOT NULL,
        qualified boolean NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_weather_benchmark_range CHECK (evaluation_end >= evaluation_start),
        CONSTRAINT chk_weather_benchmark_quantity CHECK (quantity > 0),
        CONSTRAINT chk_weather_benchmark_tier CHECK (
          evidence_tier IN ('indicative','executable_taker')
        ),
        CONSTRAINT chk_weather_benchmark_documents CHECK (
          jsonb_typeof(policy) = 'object' AND jsonb_typeof(metrics) = 'object'
        )
      );

      CREATE INDEX idx_weather_benchmark_process_created
        ON weather.benchmark_runs (process_id, created_at DESC);
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP SCHEMA IF EXISTS weather CASCADE;`);
    await queryRunner.query(`
      DELETE FROM polymarket.trading_processes WHERE process_id = '${PROCESS_ID}';
    `);
  }
}
