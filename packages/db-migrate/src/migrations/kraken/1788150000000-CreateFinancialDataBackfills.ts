import { MigrationInterface, QueryRunner } from 'typeorm';

export class CreateFinancialDataBackfills1788150000000 implements MigrationInterface {
  name = 'CreateFinancialDataBackfills1788150000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`
      CREATE SCHEMA financial_data;

      CREATE TABLE financial_data.backfill_jobs (
        job_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        provider text NOT NULL,
        dataset text NOT NULL,
        series_id text NOT NULL DEFAULT '',
        range_start timestamptz NOT NULL,
        range_end timestamptz NOT NULL,
        status text NOT NULL DEFAULT 'queued',
        attempt integer NOT NULL DEFAULT 0,
        max_attempts integer NOT NULL DEFAULT 8,
        next_attempt_at timestamptz NOT NULL DEFAULT now(),
        worker_id text,
        lease_token uuid,
        lease_expires_at timestamptz,
        rows_written bigint NOT NULL DEFAULT 0,
        error_message text,
        requested_at timestamptz NOT NULL DEFAULT now(),
        started_at timestamptz,
        completed_at timestamptz,
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_financial_provider CHECK (
          provider IN ('fred','new_york_fed','us_treasury','cftc')
        ),
        CONSTRAINT chk_financial_status CHECK (
          status IN ('queued','running','completed','failed')
        ),
        CONSTRAINT chk_financial_range CHECK (range_start < range_end),
        CONSTRAINT chk_financial_attempts CHECK (
          attempt >= 0 AND max_attempts > 0 AND attempt <= max_attempts
        ),
        CONSTRAINT uq_financial_backfill_identity UNIQUE (
          provider, dataset, series_id, range_start, range_end
        )
      );

      CREATE INDEX idx_financial_backfill_claim
        ON financial_data.backfill_jobs (status, next_attempt_at, requested_at, job_id);

      CREATE TABLE financial_data.backfill_job_events (
        event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id uuid NOT NULL REFERENCES financial_data.backfill_jobs (job_id) ON DELETE CASCADE,
        recorded_at timestamptz NOT NULL DEFAULT now(),
        level text NOT NULL,
        message text NOT NULL,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT chk_financial_event_level CHECK (
          level IN ('debug','info','warn','error')
        )
      );

      CREATE INDEX idx_financial_events_job_time
        ON financial_data.backfill_job_events (job_id, recorded_at DESC);

      CREATE TABLE financial_data.source_artifacts (
        artifact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id uuid NOT NULL REFERENCES financial_data.backfill_jobs (job_id) ON DELETE RESTRICT,
        source_url text NOT NULL,
        relative_path text NOT NULL UNIQUE,
        sha256 text NOT NULL,
        byte_size bigint NOT NULL,
        fetched_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_financial_source_hash CHECK (sha256 ~ '^[0-9a-f]{64}$'),
        CONSTRAINT chk_financial_source_size CHECK (byte_size >= 0)
      );

      CREATE TABLE financial_data.parquet_objects (
        object_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id uuid NOT NULL REFERENCES financial_data.backfill_jobs (job_id) ON DELETE RESTRICT,
        provider text NOT NULL,
        dataset text NOT NULL,
        series_id text NOT NULL DEFAULT '',
        relative_path text NOT NULL UNIQUE,
        sha256 text NOT NULL,
        byte_size bigint NOT NULL,
        row_count bigint NOT NULL,
        minimum_event_at timestamptz,
        maximum_event_at timestamptz,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_financial_parquet_hash CHECK (sha256 ~ '^[0-9a-f]{64}$'),
        CONSTRAINT chk_financial_parquet_counts CHECK (byte_size >= 0 AND row_count >= 0)
      );

      CREATE TABLE financial_data.worker_status (
        worker_id text PRIMARY KEY,
        state text NOT NULL DEFAULT 'idle',
        current_job_id uuid REFERENCES financial_data.backfill_jobs (job_id) ON DELETE SET NULL,
        jobs_completed bigint NOT NULL DEFAULT 0,
        jobs_failed bigint NOT NULL DEFAULT 0,
        rows_written bigint NOT NULL DEFAULT 0,
        started_at timestamptz NOT NULL DEFAULT now(),
        heartbeat_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_financial_worker_state CHECK (state IN ('idle','running','stopped'))
      );
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP SCHEMA financial_data CASCADE;`);
  }
}
