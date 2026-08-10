import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddMarketDataIngesterControlPlane1786381000000
  implements MigrationInterface
{
  name = 'AddMarketDataIngesterControlPlane1786381000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);

    await queryRunner.query(`
      CREATE SCHEMA ingester;
      CREATE SCHEMA market_data;
    `);

    await queryRunner.query(`
      CREATE TABLE ingester.profiles (
        strategy_key text PRIMARY KEY,
        config_schema_version integer NOT NULL,
        config jsonb NOT NULL,
        desired_state text NOT NULL DEFAULT 'stopped',
        desired_generation bigint NOT NULL DEFAULT 1,
        observed_state text NOT NULL DEFAULT 'stopped',
        health_status text NOT NULL DEFAULT 'unknown',
        applied_generation bigint,
        checkpoint_schema_version integer NOT NULL DEFAULT 1,
        checkpoint jsonb NOT NULL DEFAULT '{}'::jsonb,
        lease_owner text,
        lease_token uuid,
        lease_expires_at timestamptz,
        heartbeat_at timestamptz,
        started_at timestamptz,
        stopped_at timestamptz,
        last_source_event_at timestamptz,
        last_provider_available_at timestamptz,
        last_persisted_at timestamptz,
        source_watermark timestamptz,
        availability_watermark timestamptz,
        consecutive_failures integer NOT NULL DEFAULT 0,
        restart_count bigint NOT NULL DEFAULT 0,
        last_error_code text,
        last_error_message text,
        last_error_at timestamptz,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_ingester_profiles_strategy_key CHECK (
          length(btrim(strategy_key)) BETWEEN 1 AND 128
          AND strategy_key ~ '^[a-z0-9_]+$'
        ),
        CONSTRAINT chk_ingester_profiles_config CHECK (
          config_schema_version > 0
          AND jsonb_typeof(config) = 'object'
          AND octet_length(config::text) <= 16384
        ),
        CONSTRAINT chk_ingester_profiles_desired_state CHECK (
          desired_state IN ('running', 'stopped')
        ),
        CONSTRAINT chk_ingester_profiles_observed_state CHECK (
          observed_state IN (
            'starting', 'running', 'degraded', 'restarting',
            'stopping', 'stopped', 'failed', 'unsupported'
          )
        ),
        CONSTRAINT chk_ingester_profiles_health CHECK (
          health_status IN ('unknown', 'healthy', 'degraded', 'unhealthy')
        ),
        CONSTRAINT chk_ingester_profiles_generations CHECK (
          desired_generation > 0
          AND (applied_generation IS NULL OR (
            applied_generation > 0
            AND applied_generation <= desired_generation
          ))
        ),
        CONSTRAINT chk_ingester_profiles_checkpoint CHECK (
          checkpoint_schema_version > 0
          AND jsonb_typeof(checkpoint) = 'object'
          AND octet_length(checkpoint::text) <= 8192
        ),
        CONSTRAINT chk_ingester_profiles_lease CHECK (
          (
            lease_owner IS NULL
            AND lease_token IS NULL
            AND lease_expires_at IS NULL
          ) OR (
            length(btrim(lease_owner)) BETWEEN 1 AND 128
            AND lease_token IS NOT NULL
            AND lease_expires_at IS NOT NULL
            AND heartbeat_at IS NOT NULL
            AND lease_expires_at > heartbeat_at
          )
        ),
        CONSTRAINT chk_ingester_profiles_counters CHECK (
          consecutive_failures >= 0 AND restart_count >= 0
        ),
        CONSTRAINT chk_ingester_profiles_error CHECK (
          (
            last_error_code IS NULL
            OR length(btrim(last_error_code)) BETWEEN 1 AND 128
          )
          AND (last_error_message IS NULL OR octet_length(last_error_message) <= 2048)
          AND (
            (last_error_code IS NULL AND last_error_message IS NULL AND last_error_at IS NULL)
            OR (last_error_code IS NOT NULL AND last_error_at IS NOT NULL)
          )
        ),
        CONSTRAINT chk_ingester_profiles_times CHECK (updated_at >= created_at)
      );

      CREATE INDEX idx_ingester_profiles_expired_lease
        ON ingester.profiles (lease_expires_at, strategy_key)
        WHERE lease_token IS NOT NULL;
    `);

    await queryRunner.query(`
      CREATE TABLE ingester.capture_artifacts (
        artifact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        strategy_key text NOT NULL
          REFERENCES ingester.profiles (strategy_key) ON DELETE RESTRICT,
        profile_generation bigint NOT NULL,
        config_schema_version integer NOT NULL,
        config_sha256 text NOT NULL,
        config_snapshot jsonb NOT NULL,
        capture_window_start timestamptz NOT NULL,
        capture_window_end timestamptz NOT NULL,
        minimum_source_timestamp timestamptz,
        maximum_source_timestamp timestamptz,
        minimum_received_at timestamptz,
        maximum_received_at timestamptz,
        start_cursor text,
        end_cursor text,
        record_count bigint NOT NULL DEFAULT 0,
        content_sha256 text,
        status text NOT NULL DEFAULT 'open',
        failure_code text,
        failure_message text,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        completed_at timestamptz,
        CONSTRAINT uq_ingester_capture_artifact_window UNIQUE (
          strategy_key, capture_window_start, capture_window_end, profile_generation
        ),
        CONSTRAINT uq_ingester_capture_artifact_strategy_id UNIQUE (
          strategy_key, artifact_id
        ),
        CONSTRAINT chk_ingester_capture_artifact_generation CHECK (
          profile_generation > 0 AND config_schema_version > 0
        ),
        CONSTRAINT chk_ingester_capture_artifact_config CHECK (
          config_sha256 ~ '^[0-9a-f]{64}$'
          AND jsonb_typeof(config_snapshot) = 'object'
          AND octet_length(config_snapshot::text) <= 16384
        ),
        CONSTRAINT chk_ingester_capture_artifact_window CHECK (
          capture_window_end > capture_window_start
          AND capture_window_end <= capture_window_start + INTERVAL '7 days'
        ),
        CONSTRAINT chk_ingester_capture_artifact_source_range CHECK (
          (minimum_source_timestamp IS NULL AND maximum_source_timestamp IS NULL)
          OR (
            minimum_source_timestamp IS NOT NULL
            AND maximum_source_timestamp IS NOT NULL
            AND maximum_source_timestamp >= minimum_source_timestamp
          )
        ),
        CONSTRAINT chk_ingester_capture_artifact_receipt_range CHECK (
          (minimum_received_at IS NULL AND maximum_received_at IS NULL)
          OR (
            minimum_received_at IS NOT NULL
            AND maximum_received_at IS NOT NULL
            AND maximum_received_at >= minimum_received_at
          )
        ),
        CONSTRAINT chk_ingester_capture_artifact_cursors CHECK (
          (start_cursor IS NULL OR octet_length(start_cursor) <= 2048)
          AND (end_cursor IS NULL OR octet_length(end_cursor) <= 2048)
        ),
        CONSTRAINT chk_ingester_capture_artifact_counts CHECK (record_count >= 0),
        CONSTRAINT chk_ingester_capture_artifact_content_hash CHECK (
          content_sha256 IS NULL OR content_sha256 ~ '^[0-9a-f]{64}$'
        ),
        CONSTRAINT chk_ingester_capture_artifact_status CHECK (
          status IN ('open', 'completed', 'failed')
        ),
        CONSTRAINT chk_ingester_capture_artifact_completion CHECK (
          (
            status = 'open'
            AND content_sha256 IS NULL
            AND completed_at IS NULL
            AND failure_code IS NULL
            AND failure_message IS NULL
          ) OR (
            status = 'completed'
            AND content_sha256 ~ '^[0-9a-f]{64}$'
            AND completed_at IS NOT NULL
            AND failure_code IS NULL
            AND failure_message IS NULL
          ) OR (
            status = 'failed'
            AND completed_at IS NOT NULL
            AND length(btrim(failure_code)) BETWEEN 1 AND 128
            AND (failure_message IS NULL OR octet_length(failure_message) <= 2048)
          )
        ),
        CONSTRAINT chk_ingester_capture_artifact_times CHECK (
          updated_at >= created_at
          AND (completed_at IS NULL OR completed_at >= created_at)
        )
      );

      CREATE UNIQUE INDEX uq_ingester_capture_artifact_open_strategy
        ON ingester.capture_artifacts (strategy_key)
        WHERE status = 'open';

      CREATE INDEX idx_ingester_capture_artifact_strategy_status
        ON ingester.capture_artifacts (
          strategy_key, status, capture_window_start DESC, artifact_id
        );
    `);

    await queryRunner.query(`
      CREATE TABLE ingester.data_gaps (
        gap_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        gap_fingerprint text NOT NULL UNIQUE,
        strategy_key text NOT NULL
          REFERENCES ingester.profiles (strategy_key) ON DELETE RESTRICT,
        detected_artifact_id uuid,
        repair_artifact_id uuid,
        gap_kind text NOT NULL,
        reason_code text NOT NULL,
        reason_message text,
        source_time_start timestamptz,
        source_time_end timestamptz,
        start_cursor text,
        end_cursor text,
        status text NOT NULL DEFAULT 'open',
        repair_attempts integer NOT NULL DEFAULT 0,
        detected_at timestamptz NOT NULL DEFAULT now(),
        repair_started_at timestamptz,
        resolved_at timestamptz,
        resolution_code text,
        resolution_message text,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT fk_ingester_data_gap_detected_artifact FOREIGN KEY (
          strategy_key, detected_artifact_id
        ) REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT fk_ingester_data_gap_repair_artifact FOREIGN KEY (
          strategy_key, repair_artifact_id
        ) REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_ingester_data_gap_fingerprint CHECK (
          gap_fingerprint ~ '^[0-9a-f]{64}$'
        ),
        CONSTRAINT chk_ingester_data_gap_classification CHECK (
          length(btrim(gap_kind)) BETWEEN 1 AND 64
          AND length(btrim(reason_code)) BETWEEN 1 AND 128
          AND (reason_message IS NULL OR octet_length(reason_message) <= 2048)
        ),
        CONSTRAINT chk_ingester_data_gap_source_time CHECK (
          (source_time_start IS NULL AND source_time_end IS NULL)
          OR (
            source_time_start IS NOT NULL
            AND source_time_end IS NOT NULL
            AND source_time_end >= source_time_start
          )
        ),
        CONSTRAINT chk_ingester_data_gap_cursors CHECK (
          (start_cursor IS NULL OR octet_length(start_cursor) <= 2048)
          AND (end_cursor IS NULL OR octet_length(end_cursor) <= 2048)
          AND (
            source_time_start IS NOT NULL
            OR start_cursor IS NOT NULL
            OR end_cursor IS NOT NULL
          )
        ),
        CONSTRAINT chk_ingester_data_gap_status CHECK (
          status IN ('open', 'repairing', 'repaired', 'unrecoverable')
          AND repair_attempts >= 0
        ),
        CONSTRAINT chk_ingester_data_gap_resolution CHECK (
          (
            status = 'open'
            AND repair_artifact_id IS NULL
            AND repair_started_at IS NULL
            AND resolved_at IS NULL
            AND resolution_code IS NULL
            AND resolution_message IS NULL
          ) OR (
            status = 'repairing'
            AND repair_started_at IS NOT NULL
            AND repair_attempts > 0
            AND resolved_at IS NULL
            AND resolution_code IS NULL
            AND resolution_message IS NULL
          ) OR (
            status = 'repaired'
            AND repair_artifact_id IS NOT NULL
            AND repair_attempts > 0
            AND resolved_at IS NOT NULL
            AND length(btrim(resolution_code)) BETWEEN 1 AND 128
            AND (resolution_message IS NULL OR octet_length(resolution_message) <= 2048)
          ) OR (
            status = 'unrecoverable'
            AND resolved_at IS NOT NULL
            AND length(btrim(resolution_code)) BETWEEN 1 AND 128
            AND (resolution_message IS NULL OR octet_length(resolution_message) <= 2048)
          )
        ),
        CONSTRAINT chk_ingester_data_gap_times CHECK (
          updated_at >= created_at
          AND (repair_started_at IS NULL OR repair_started_at >= detected_at)
          AND (
            resolved_at IS NULL
            OR resolved_at >= COALESCE(repair_started_at, detected_at)
          )
        )
      );

      CREATE INDEX idx_ingester_data_gap_open
        ON ingester.data_gaps (strategy_key, detected_at, gap_id)
        WHERE status IN ('open', 'repairing');

      CREATE INDEX idx_ingester_data_gap_source_range
        ON ingester.data_gaps (
          strategy_key, source_time_start, source_time_end, gap_id
        );
    `);

    await queryRunner.query(`
      CREATE FUNCTION ingester.notify_profile_change()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        IF TG_OP = 'INSERT' THEN
          PERFORM pg_notify('ingester_profile_changed', NEW.strategy_key);
        ELSIF (
          NEW.desired_state,
          NEW.desired_generation,
          NEW.config_schema_version,
          NEW.config
        ) IS DISTINCT FROM (
          OLD.desired_state,
          OLD.desired_generation,
          OLD.config_schema_version,
          OLD.config
        ) THEN
          PERFORM pg_notify('ingester_profile_changed', NEW.strategy_key);
        END IF;
        RETURN NEW;
      END;
      $$;

      CREATE FUNCTION ingester.validate_profile_change()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        IF NEW.strategy_key IS DISTINCT FROM OLD.strategy_key
          OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
          RAISE EXCEPTION 'ingester profile identity is immutable for %', OLD.strategy_key
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF NEW.desired_generation < OLD.desired_generation
          OR NEW.desired_generation::numeric > OLD.desired_generation::numeric + 1 THEN
          RAISE EXCEPTION 'invalid desired generation transition for ingester profile %', OLD.strategy_key
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF (
          NEW.desired_state,
          NEW.config_schema_version,
          NEW.config
        ) IS DISTINCT FROM (
          OLD.desired_state,
          OLD.config_schema_version,
          OLD.config
        ) AND NEW.desired_generation::numeric <> OLD.desired_generation::numeric + 1 THEN
          RAISE EXCEPTION 'profile control changes must advance desired generation for %', OLD.strategy_key
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF OLD.applied_generation IS NOT NULL AND (
          NEW.applied_generation IS NULL
          OR NEW.applied_generation < OLD.applied_generation
        ) THEN
          RAISE EXCEPTION 'applied generation cannot regress for ingester profile %', OLD.strategy_key
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF OLD.lease_token IS NOT NULL
          AND NEW.lease_token IS NOT NULL
          AND NEW.lease_token IS DISTINCT FROM OLD.lease_token
          AND OLD.lease_expires_at > clock_timestamp() THEN
          RAISE EXCEPTION 'active lease cannot be replaced for ingester profile %', OLD.strategy_key
            USING ERRCODE = 'lock_not_available';
        END IF;

        IF OLD.lease_token IS NOT NULL
          AND NEW.lease_token = OLD.lease_token
          AND (
            NEW.lease_owner IS DISTINCT FROM OLD.lease_owner
            OR NEW.heartbeat_at < OLD.heartbeat_at
            OR NEW.lease_expires_at < OLD.lease_expires_at
          ) THEN
          RAISE EXCEPTION 'lease ownership and timestamps cannot regress for ingester profile %', OLD.strategy_key
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        RETURN NEW;
      END;
      $$;

      CREATE TRIGGER trg_validate_ingester_profile_change
        BEFORE UPDATE ON ingester.profiles
        FOR EACH ROW
        EXECUTE FUNCTION ingester.validate_profile_change();

      CREATE TRIGGER trg_notify_ingester_profile_change
        AFTER INSERT OR UPDATE OF desired_state, desired_generation,
          config_schema_version, config
        ON ingester.profiles
        FOR EACH ROW
        EXECUTE FUNCTION ingester.notify_profile_change();

      CREATE FUNCTION ingester.reject_terminal_artifact_change()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        IF TG_OP = 'DELETE' THEN
          RAISE EXCEPTION 'capture artifact % cannot be deleted', OLD.artifact_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF TG_OP = 'UPDATE'
          AND OLD.status IN ('completed', 'failed')
          AND NEW IS DISTINCT FROM OLD THEN
          RAISE EXCEPTION 'terminal capture artifact % is immutable', OLD.artifact_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF TG_OP = 'UPDATE' AND (
          NEW.artifact_id,
          NEW.strategy_key,
          NEW.profile_generation,
          NEW.config_schema_version,
          NEW.config_sha256,
          NEW.config_snapshot,
          NEW.capture_window_start,
          NEW.capture_window_end,
          NEW.created_at
        ) IS DISTINCT FROM (
          OLD.artifact_id,
          OLD.strategy_key,
          OLD.profile_generation,
          OLD.config_schema_version,
          OLD.config_sha256,
          OLD.config_snapshot,
          OLD.capture_window_start,
          OLD.capture_window_end,
          OLD.created_at
        ) THEN
          RAISE EXCEPTION 'capture artifact identity is immutable for %', OLD.artifact_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF TG_OP = 'UPDATE' AND NEW.record_count < OLD.record_count THEN
          RAISE EXCEPTION 'capture artifact record count cannot regress for %', OLD.artifact_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        RETURN NEW;
      END;
      $$;

      CREATE TRIGGER trg_reject_terminal_capture_artifact_change
        BEFORE UPDATE OR DELETE ON ingester.capture_artifacts
        FOR EACH ROW
        EXECUTE FUNCTION ingester.reject_terminal_artifact_change();

      CREATE FUNCTION ingester.reject_data_gap_identity_change()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        IF TG_OP = 'INSERT' THEN
          IF NEW.status = 'repaired' AND NOT EXISTS (
            SELECT 1
            FROM ingester.capture_artifacts artifact
            WHERE artifact.strategy_key = NEW.strategy_key
              AND artifact.artifact_id = NEW.repair_artifact_id
              AND artifact.status = 'completed'
          ) THEN
            RAISE EXCEPTION 'repair artifact must be completed for data gap %', NEW.gap_id
              USING ERRCODE = 'integrity_constraint_violation';
          END IF;
          RETURN NEW;
        END IF;

        IF TG_OP = 'DELETE' THEN
          RAISE EXCEPTION 'data gap % cannot be deleted', OLD.gap_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF OLD.status IN ('repaired', 'unrecoverable')
          AND NEW IS DISTINCT FROM OLD THEN
          RAISE EXCEPTION 'terminal data gap % is immutable', OLD.gap_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF (
          NEW.gap_id,
          NEW.gap_fingerprint,
          NEW.strategy_key,
          NEW.detected_artifact_id,
          NEW.gap_kind,
          NEW.reason_code,
          NEW.reason_message,
          NEW.source_time_start,
          NEW.source_time_end,
          NEW.start_cursor,
          NEW.end_cursor,
          NEW.detected_at,
          NEW.created_at
        ) IS DISTINCT FROM (
          OLD.gap_id,
          OLD.gap_fingerprint,
          OLD.strategy_key,
          OLD.detected_artifact_id,
          OLD.gap_kind,
          OLD.reason_code,
          OLD.reason_message,
          OLD.source_time_start,
          OLD.source_time_end,
          OLD.start_cursor,
          OLD.end_cursor,
          OLD.detected_at,
          OLD.created_at
        ) THEN
          RAISE EXCEPTION 'data gap identity is immutable for %', OLD.gap_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF OLD.status = 'repairing' AND NEW.status = 'open' THEN
          RAISE EXCEPTION 'data gap % cannot return to open after repair begins', OLD.gap_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF NEW.repair_attempts < OLD.repair_attempts THEN
          RAISE EXCEPTION 'data gap repair attempts cannot regress for %', OLD.gap_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        IF NEW.status = 'repaired' AND NOT EXISTS (
          SELECT 1
          FROM ingester.capture_artifacts artifact
          WHERE artifact.strategy_key = NEW.strategy_key
            AND artifact.artifact_id = NEW.repair_artifact_id
            AND artifact.status = 'completed'
        ) THEN
          RAISE EXCEPTION 'repair artifact must be completed for data gap %', OLD.gap_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;

        RETURN NEW;
      END;
      $$;

      CREATE TRIGGER trg_reject_data_gap_identity_change
        BEFORE INSERT OR UPDATE OR DELETE ON ingester.data_gaps
        FOR EACH ROW
        EXECUTE FUNCTION ingester.reject_data_gap_identity_change();

      CREATE FUNCTION market_data.reject_source_fact_change()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        RAISE EXCEPTION 'market source facts are immutable'
          USING ERRCODE = 'integrity_constraint_violation';
      END;
      $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (SELECT 1 FROM ingester.capture_artifacts LIMIT 1)
          OR EXISTS (SELECT 1 FROM ingester.data_gaps LIMIT 1)
          OR EXISTS (SELECT 1 FROM ingester.profiles LIMIT 1) THEN
          RAISE EXCEPTION
            'refusing to remove market-data ingester control plane while profiles, lineage, or gaps exist';
        END IF;
      END $$;

      DROP FUNCTION market_data.reject_source_fact_change();
      DROP TRIGGER trg_reject_data_gap_identity_change ON ingester.data_gaps;
      DROP FUNCTION ingester.reject_data_gap_identity_change();
      DROP TRIGGER trg_reject_terminal_capture_artifact_change
        ON ingester.capture_artifacts;
      DROP FUNCTION ingester.reject_terminal_artifact_change();
      DROP TRIGGER trg_notify_ingester_profile_change ON ingester.profiles;
      DROP FUNCTION ingester.notify_profile_change();
      DROP TRIGGER trg_validate_ingester_profile_change ON ingester.profiles;
      DROP FUNCTION ingester.validate_profile_change();
      DROP TABLE ingester.data_gaps;
      DROP TABLE ingester.capture_artifacts;
      DROP TABLE ingester.profiles;
      DROP SCHEMA market_data;
      DROP SCHEMA ingester;
    `);
  }
}
