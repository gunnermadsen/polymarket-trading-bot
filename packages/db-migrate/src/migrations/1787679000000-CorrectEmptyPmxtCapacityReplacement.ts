import { MigrationInterface, QueryRunner } from 'typeorm';

export class CorrectEmptyPmxtCapacityReplacement1787679000000
  implements MigrationInterface
{
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE OR REPLACE FUNCTION polymarket.reject_btc_market_execution_snapshot_change()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        IF TG_OP = 'DELETE'
          AND EXISTS (
            SELECT 1
            FROM polymarket.backfill_artifacts artifact
            WHERE artifact.artifact_id = OLD.artifact_id
              AND artifact.provider = 'pmxt_v2_capacity_execution_snapshots_v2'
              AND COALESCE(
                (artifact.metadata ->> 'source_events_consumed')::bigint,
                0
              ) = 0
          )
        THEN
          RETURN OLD;
        END IF;

        RAISE EXCEPTION
          'historical BTC execution snapshot is immutable'
          USING ERRCODE = 'integrity_constraint_violation';
      END;
      $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE OR REPLACE FUNCTION polymarket.reject_btc_market_execution_snapshot_change()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        IF TG_OP = 'DELETE'
          AND TG_TABLE_NAME = 'btc_market_capacity_execution_snapshots'
          AND EXISTS (
            SELECT 1
            FROM polymarket.backfill_artifacts artifact
            WHERE artifact.artifact_id = OLD.artifact_id
              AND artifact.provider = 'pmxt_v2_capacity_execution_snapshots_v2'
              AND COALESCE(
                (artifact.metadata ->> 'source_events_consumed')::bigint,
                0
              ) = 0
          )
        THEN
          RETURN OLD;
        END IF;

        RAISE EXCEPTION
          'historical BTC execution snapshot is immutable'
          USING ERRCODE = 'integrity_constraint_violation';
      END;
      $$;
    `);
  }
}
