import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddIngesterDrainJobs1788732000000 implements MigrationInterface {
  name = 'AddIngesterDrainJobs1788732000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE ingester.drain_jobs (
        job_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        strategy_key text NOT NULL,
        strategy_contract_version integer NOT NULL,
        cutoff timestamptz NOT NULL,
        dry_run boolean NOT NULL DEFAULT false,
        status text NOT NULL DEFAULT 'queued',
        required_worker_id text,
        required_deployment text,
        assigned_worker_id text,
        lease_token uuid,
        lease_expires_at timestamptz,
        attempt integer NOT NULL DEFAULT 0,
        max_attempts integer NOT NULL DEFAULT 5,
        rows_exported bigint NOT NULL DEFAULT 0,
        rows_removed bigint NOT NULL DEFAULT 0,
        objects_published bigint NOT NULL DEFAULT 0,
        bytes_written bigint NOT NULL DEFAULT 0,
        summary jsonb NOT NULL DEFAULT '{}'::jsonb,
        last_error_code text,
        last_error_message text,
        requested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        started_at timestamptz,
        completed_at timestamptz,
        cancel_requested_at timestamptz,
        updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT chk_ingester_drain_job_status CHECK (status IN ('queued','running','completed','failed','cancelled')),
        CONSTRAINT chk_ingester_drain_job_contract CHECK (strategy_contract_version > 0 AND attempt >= 0 AND max_attempts > 0),
        CONSTRAINT chk_ingester_drain_job_selector CHECK (required_worker_id IS NULL OR required_deployment IS NULL)
      );
      CREATE INDEX idx_ingester_drain_jobs_claim ON ingester.drain_jobs (requested_at,job_id) WHERE status IN ('queued','running');

      CREATE TABLE ingester.drain_objects (
        object_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id uuid NOT NULL REFERENCES ingester.drain_jobs(job_id) ON DELETE RESTRICT,
        strategy_key text NOT NULL,
        source_relation text NOT NULL,
        source_chunk_schema text NOT NULL,
        source_chunk_name text NOT NULL,
        source_start timestamptz NOT NULL,
        source_end timestamptz NOT NULL,
        row_count bigint,
        minimum_aggregate_trade_id bigint,
        maximum_aggregate_trade_id bigint,
        relative_path text,
        sha256 character(64),
        byte_size bigint,
        status text NOT NULL DEFAULT 'staging',
        published_at timestamptz,
        removed_at timestamptz,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT uq_ingester_drain_object_chunk UNIQUE (strategy_key,source_chunk_schema,source_chunk_name),
        CONSTRAINT chk_ingester_drain_object_range CHECK (source_end > source_start),
        CONSTRAINT chk_ingester_drain_object_status CHECK (status IN ('staging','published','removed')),
        CONSTRAINT chk_ingester_drain_object_publication CHECK (
          status='staging' OR (row_count >= 0 AND relative_path IS NOT NULL AND sha256 ~ '^[0-9a-f]{64}$' AND byte_size > 0 AND published_at IS NOT NULL)
        )
      );
      CREATE INDEX idx_ingester_drain_objects_job ON ingester.drain_objects (job_id,source_start,object_id);

      CREATE OR REPLACE FUNCTION ingester.remove_verified_binance_aggregate_trade_chunk(requested_object_id uuid,expected_sha256 text)
      RETURNS bigint LANGUAGE plpgsql SET search_path=pg_catalog,public,ingester,market_data AS $$
      DECLARE object_record ingester.drain_objects%ROWTYPE; matching_chunks integer; dropped_chunks integer;
      BEGIN
        SELECT * INTO object_record FROM ingester.drain_objects WHERE object_id=requested_object_id FOR UPDATE;
        IF NOT FOUND OR object_record.status <> 'published'
          OR object_record.strategy_key <> 'binance_spot_btcusdt_aggregate_trades'
          OR object_record.source_relation <> 'market_data.binance_spot_btcusdt_aggregate_trades'
          OR object_record.sha256 <> expected_sha256 THEN
          RAISE EXCEPTION 'drain object is not a verified Binance aggregate-trade publication';
        END IF;
        SELECT count(*) INTO matching_chunks FROM timescaledb_information.chunks
        WHERE hypertable_schema='market_data' AND hypertable_name='binance_spot_btcusdt_aggregate_trades'
          AND chunk_schema=object_record.source_chunk_schema AND chunk_name=object_record.source_chunk_name
          AND range_start=object_record.source_start AND range_end=object_record.source_end;
        IF matching_chunks <> 1 THEN RAISE EXCEPTION 'source chunk identity or bounds changed before removal'; END IF;
        SELECT count(*) INTO dropped_chunks FROM drop_chunks(
          'market_data.binance_spot_btcusdt_aggregate_trades'::regclass,
          older_than=>object_record.source_end,newer_than=>object_record.source_start
        );
        IF dropped_chunks <> 1 THEN RAISE EXCEPTION 'expected one removed chunk, removed %',dropped_chunks; END IF;
        UPDATE ingester.drain_objects SET status='removed',removed_at=clock_timestamp(),updated_at=clock_timestamp()
        WHERE object_id=requested_object_id;
        RETURN object_record.row_count;
      END $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$ BEGIN IF EXISTS (SELECT 1 FROM ingester.drain_objects WHERE status='removed') THEN
        RAISE EXCEPTION 'refusing to remove drain ledger after source chunks were removed'; END IF; END $$;
      DROP FUNCTION ingester.remove_verified_binance_aggregate_trade_chunk(uuid,text);
      DROP TABLE ingester.drain_objects;
      DROP TABLE ingester.drain_jobs;
    `);
  }
}
