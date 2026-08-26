import { MigrationInterface, QueryRunner } from 'typeorm';

export class AllowEmptyPmxtArtifactSupersession1787679600000
  implements MigrationInterface
{
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE OR REPLACE FUNCTION polymarket.reject_completed_backfill_artifact_change()
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
          IF OLD.provider = 'pmxt_v2_capacity_execution_snapshots_v2'
            AND COALESCE(
              (OLD.metadata ->> 'source_events_consumed')::bigint,
              0
            ) = 0
            AND NEW.record_count = 0
            AND NEW.metadata ->> 'superseded_reason'
              = 'reconstructed_from_local_canonical_orderbooks'
            AND NEW.metadata ->> 'superseded_by_artifact_id' IS NOT NULL
            AND NEW.artifact_id = OLD.artifact_id
            AND NEW.job_id = OLD.job_id
            AND NEW.ingester_key = OLD.ingester_key
            AND NEW.logical_key = OLD.logical_key
            AND NEW.provider = OLD.provider
            AND NEW.source_uri = OLD.source_uri
            AND NEW.status = OLD.status
            AND NEW.actual_checksum IS NOT DISTINCT FROM OLD.actual_checksum
            AND NEW.minimum_source_timestamp IS NOT DISTINCT FROM OLD.minimum_source_timestamp
            AND NEW.maximum_source_timestamp IS NOT DISTINCT FROM OLD.maximum_source_timestamp
          THEN
            RETURN NEW;
          END IF;

          RAISE EXCEPTION
            'completed backfill artifact % is immutable', OLD.artifact_id
            USING ERRCODE = 'integrity_constraint_violation';
        END IF;
        RETURN NEW;
      END;
      $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE OR REPLACE FUNCTION polymarket.reject_completed_backfill_artifact_change()
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
    `);
  }
}
