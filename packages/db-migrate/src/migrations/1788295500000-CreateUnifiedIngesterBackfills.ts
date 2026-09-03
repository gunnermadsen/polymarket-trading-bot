import { MigrationInterface, QueryRunner } from 'typeorm';

export class CreateUnifiedIngesterBackfills1788295500000
  implements MigrationInterface
{
  name = 'CreateUnifiedIngesterBackfills1788295500000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`
      CREATE TABLE ingester.backfill_jobs (
        job_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        parent_job_id uuid REFERENCES ingester.backfill_jobs (job_id) ON DELETE RESTRICT,
        job_kind text NOT NULL,
        strategy_key text NOT NULL,
        strategy_contract_version integer NOT NULL,
        request_schema_version integer NOT NULL,
        canonical_request jsonb NOT NULL,
        request_hash text NOT NULL,
        range_start timestamptz NOT NULL,
        range_end timestamptz NOT NULL,
        shard_key text,
        status text NOT NULL DEFAULT 'queued',
        attempt integer NOT NULL DEFAULT 0,
        max_attempts integer NOT NULL DEFAULT 3,
        next_attempt_at timestamptz NOT NULL DEFAULT now(),
        assigned_worker_id text,
        required_worker_id text,
        required_deployment text,
        lease_token uuid,
        lease_expires_at timestamptz,
        heartbeat_at timestamptz,
        progress jsonb NOT NULL DEFAULT '{}'::jsonb,
        checkpoint jsonb NOT NULL DEFAULT '{}'::jsonb,
        verified_coverage jsonb NOT NULL DEFAULT '{}'::jsonb,
        summary jsonb NOT NULL DEFAULT '{}'::jsonb,
        last_error_kind text,
        last_error_code text,
        last_error_message text,
        requested_at timestamptz NOT NULL DEFAULT now(),
        started_at timestamptz,
        completed_at timestamptz,
        cancel_requested_at timestamptz,
        assigned_worker_image_digest text,
        assigned_worker_source_revision text,
        legacy_source text,
        legacy_job_id uuid,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_ingester_backfill_kind
          CHECK (job_kind IN ('request','shard')),
        CONSTRAINT chk_ingester_backfill_strategy
          CHECK (length(btrim(strategy_key)) > 0),
        CONSTRAINT chk_ingester_backfill_versions
          CHECK (strategy_contract_version > 0 AND request_schema_version > 0),
        CONSTRAINT chk_ingester_backfill_request
          CHECK (jsonb_typeof(canonical_request) = 'object'),
        CONSTRAINT chk_ingester_backfill_hash
          CHECK (request_hash ~ '^[0-9a-f]{64}$'),
        CONSTRAINT chk_ingester_backfill_range
          CHECK (range_start < range_end),
        CONSTRAINT chk_ingester_backfill_status
          CHECK (status IN (
            'queued','running','cancel_requested','completed','failed','cancelled'
          )),
        CONSTRAINT chk_ingester_backfill_attempts
          CHECK (attempt >= 0 AND max_attempts > 0 AND attempt <= max_attempts),
        CONSTRAINT chk_ingester_backfill_documents
          CHECK (
            jsonb_typeof(progress) = 'object'
            AND jsonb_typeof(checkpoint) = 'object'
            AND jsonb_typeof(verified_coverage) = 'object'
            AND jsonb_typeof(summary) = 'object'
          ),
        CONSTRAINT chk_ingester_backfill_error_kind
          CHECK (
            last_error_kind IS NULL OR last_error_kind IN (
              'transient_source','transient_database','rate_limited',
              'invalid_request','integrity','lease_lost','cancelled'
            )
          ),
        CONSTRAINT chk_ingester_backfill_parent
          CHECK (
            (job_kind = 'request' AND parent_job_id IS NULL AND shard_key IS NULL)
            OR
            (job_kind = 'shard' AND parent_job_id IS NOT NULL AND shard_key IS NOT NULL)
          ),
        CONSTRAINT chk_ingester_backfill_assignment
          CHECK (
            (job_kind = 'shard'
              AND status IN ('running','cancel_requested')
              AND assigned_worker_id IS NOT NULL
              AND lease_token IS NOT NULL
              AND lease_expires_at IS NOT NULL
              AND heartbeat_at IS NOT NULL)
            OR
            (job_kind = 'request')
            OR
            (status NOT IN ('running','cancel_requested'))
            OR
            legacy_source IS NOT NULL
          ),
        CONSTRAINT uq_ingester_backfill_legacy
          UNIQUE (legacy_source, legacy_job_id)
      );

      CREATE UNIQUE INDEX uq_ingester_backfill_request_hash
        ON ingester.backfill_jobs (request_hash)
        WHERE job_kind = 'request' AND legacy_source IS NULL;

      CREATE UNIQUE INDEX uq_ingester_backfill_shard
        ON ingester.backfill_jobs (parent_job_id, shard_key)
        WHERE job_kind = 'shard';

      CREATE INDEX idx_ingester_backfill_claim
        ON ingester.backfill_jobs (
          strategy_key, next_attempt_at, requested_at, job_id
        )
        WHERE job_kind = 'shard' AND status = 'queued';

      CREATE INDEX idx_ingester_backfill_expired_lease
        ON ingester.backfill_jobs (lease_expires_at, job_id)
        WHERE job_kind = 'shard' AND status IN ('running','cancel_requested');

      CREATE INDEX idx_ingester_backfill_parent
        ON ingester.backfill_jobs (parent_job_id, status, job_id)
        WHERE parent_job_id IS NOT NULL;

      CREATE INDEX idx_ingester_backfill_history
        ON ingester.backfill_jobs (strategy_key, status, requested_at DESC, job_id DESC);

      CREATE INDEX idx_ingester_backfill_worker
        ON ingester.backfill_jobs (assigned_worker_id, status, heartbeat_at)
        WHERE assigned_worker_id IS NOT NULL;

      CREATE TABLE ingester.workers (
        worker_id text PRIMARY KEY,
        hostname text NOT NULL,
        worker_contract_version integer NOT NULL,
        supported_strategies jsonb NOT NULL,
        maximum_backfills integer NOT NULL,
        active_backfills integer NOT NULL DEFAULT 0,
        realtime_strategies jsonb NOT NULL DEFAULT '[]'::jsonb,
        image_digest text NOT NULL,
        source_revision text NOT NULL,
        deployment_id text NOT NULL,
        lifecycle_state text NOT NULL DEFAULT 'active',
        started_at timestamptz NOT NULL DEFAULT now(),
        heartbeat_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_ingester_worker_identity CHECK (
          length(btrim(worker_id)) > 0
          AND length(btrim(hostname)) > 0
          AND length(btrim(image_digest)) > 0
          AND length(btrim(source_revision)) > 0
          AND length(btrim(deployment_id)) > 0
        ),
        CONSTRAINT chk_ingester_worker_contract
          CHECK (worker_contract_version > 0),
        CONSTRAINT chk_ingester_worker_capabilities CHECK (
          jsonb_typeof(supported_strategies) = 'object'
          AND jsonb_typeof(realtime_strategies) = 'array'
        ),
        CONSTRAINT chk_ingester_worker_capacity CHECK (
          maximum_backfills > 0
          AND active_backfills >= 0
          AND active_backfills <= maximum_backfills
        ),
        CONSTRAINT chk_ingester_worker_lifecycle
          CHECK (lifecycle_state IN ('active','draining'))
      );

      CREATE INDEX idx_ingester_workers_available
        ON ingester.workers (lifecycle_state, heartbeat_at DESC, worker_id);

      CREATE TABLE ingester.backfill_job_events (
        event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id uuid NOT NULL REFERENCES ingester.backfill_jobs (job_id) ON DELETE RESTRICT,
        recorded_at timestamptz NOT NULL DEFAULT now(),
        level text NOT NULL,
        event_code text NOT NULL,
        message text NOT NULL,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT chk_ingester_backfill_event_level
          CHECK (level IN ('debug','info','warn','error')),
        CONSTRAINT chk_ingester_backfill_event_metadata
          CHECK (jsonb_typeof(metadata) = 'object')
      );

      CREATE INDEX idx_ingester_backfill_events_job_time
        ON ingester.backfill_job_events (job_id, recorded_at DESC, event_id);

      CREATE TABLE ingester.backfill_artifacts (
        artifact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id uuid NOT NULL REFERENCES ingester.backfill_jobs (job_id) ON DELETE RESTRICT,
        strategy_key text NOT NULL,
        logical_key text NOT NULL,
        provider text NOT NULL,
        source_uri text NOT NULL,
        checksum_algorithm text NOT NULL DEFAULT 'sha256',
        checksum text,
        byte_size bigint,
        record_count bigint,
        minimum_source_timestamp timestamptz,
        maximum_source_timestamp timestamptz,
        durable_target text,
        status text NOT NULL,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        completed_at timestamptz,
        legacy_source text,
        legacy_artifact_id uuid,
        CONSTRAINT uq_ingester_backfill_artifact_logical
          UNIQUE (strategy_key, logical_key),
        CONSTRAINT uq_ingester_backfill_artifact_legacy
          UNIQUE (legacy_source, legacy_artifact_id),
        CONSTRAINT chk_ingester_backfill_artifact_status
          CHECK (status IN (
            'pending','downloading','downloaded','verified','ingesting','completed','failed'
          )),
        CONSTRAINT chk_ingester_backfill_artifact_checksum
          CHECK (
            checksum_algorithm = 'sha256'
            AND (checksum IS NULL OR checksum ~ '^[0-9a-f]{64}$')
          ),
        CONSTRAINT chk_ingester_backfill_artifact_counts
          CHECK (
            (byte_size IS NULL OR byte_size >= 0)
            AND (record_count IS NULL OR record_count >= 0)
          ),
        CONSTRAINT chk_ingester_backfill_artifact_metadata
          CHECK (jsonb_typeof(metadata) = 'object')
      );

      CREATE INDEX idx_ingester_backfill_artifacts_job
        ON ingester.backfill_artifacts (job_id, created_at, artifact_id);
    `);

    await queryRunner.query(`
      INSERT INTO ingester.backfill_jobs (
        job_id, job_kind, strategy_key, strategy_contract_version,
        request_schema_version, canonical_request, request_hash,
        range_start, range_end, status, attempt, max_attempts,
        next_attempt_at, assigned_worker_id, lease_token, lease_expires_at,
        heartbeat_at, progress, checkpoint, summary, last_error_message,
        requested_at, started_at, completed_at, created_at, updated_at,
        legacy_source, legacy_job_id
      )
      SELECT
        job_id,
        'request',
        ingester_key,
        1,
        request_version,
        jsonb_build_object(
          'strategy_key', ingester_key,
          'request_schema_version', request_version,
          'range', jsonb_build_object('start', range_start, 'end', range_end),
          'parameters', request,
          'legacy_idempotency_key', idempotency_key
        ),
        encode(digest('polymarket:' || job_id::text, 'sha256'), 'hex'),
        COALESCE(range_start, requested_at),
        COALESCE(range_end, requested_at + interval '1 microsecond'),
        status,
        attempt,
        max_attempts,
        next_attempt_at,
        worker_id,
        lease_token,
        lease_expires_at,
        heartbeat_at,
        progress,
        checkpoint,
        summary,
        error,
        requested_at,
        started_at,
        completed_at,
        requested_at,
        updated_at,
        'polymarket.backfill_jobs',
        job_id
      FROM polymarket.backfill_jobs
      ON CONFLICT (legacy_source, legacy_job_id) DO NOTHING;
    `);

    await queryRunner.query(`
      INSERT INTO ingester.backfill_job_events (
        event_id, job_id, recorded_at, level, event_code, message, metadata
      )
      SELECT
        event_id, job_id, timestamp_utc, level,
        'legacy_event', message,
        metadata || jsonb_build_object('legacy_source', 'polymarket.backfill_job_events')
      FROM polymarket.backfill_job_events
      ON CONFLICT (event_id) DO NOTHING;
    `);

    await queryRunner.query(`
      INSERT INTO ingester.backfill_artifacts (
        artifact_id, job_id, strategy_key, logical_key, provider, source_uri,
        checksum_algorithm, checksum, byte_size, record_count,
        minimum_source_timestamp, maximum_source_timestamp, status, metadata,
        created_at, completed_at, legacy_source, legacy_artifact_id
      )
      SELECT
        artifact_id, job_id, ingester_key, logical_key, provider, source_uri,
        checksum_algorithm, actual_checksum, compressed_bytes, record_count,
        minimum_source_timestamp, maximum_source_timestamp, status,
        metadata || jsonb_build_object(
          'legacy_expected_checksum', expected_checksum,
          'legacy_source_date', source_date,
          'legacy_source', 'polymarket.backfill_artifacts'
        ),
        created_at, completed_at,
        'polymarket.backfill_artifacts', artifact_id
      FROM polymarket.backfill_artifacts
      ON CONFLICT DO NOTHING;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    const rows = await queryRunner.query(`
      SELECT EXISTS (
        SELECT 1 FROM ingester.backfill_jobs
        WHERE legacy_source IS NULL
        LIMIT 1
      ) AS has_native_jobs;
    `);
    if (rows[0]?.has_native_jobs) {
      throw new Error(
        'cannot remove unified ingester backfills after native jobs have been scheduled',
      );
    }
    await queryRunner.query(`DROP TABLE ingester.backfill_artifacts;`);
    await queryRunner.query(`DROP TABLE ingester.backfill_job_events;`);
    await queryRunner.query(`DROP TABLE ingester.workers;`);
    await queryRunner.query(`DROP TABLE ingester.backfill_jobs;`);
  }
}
