import { MigrationInterface, QueryRunner } from 'typeorm';

export class ExpandBtcCapacityVwapTiers1786824000000
  implements MigrationInterface
{
  name = 'ExpandBtcCapacityVwapTiers1786824000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
          FROM polymarket.backfill_jobs
          WHERE ingester_key = 'polymarket_btc_five_minute_execution_snapshots'
            AND idempotency_key LIKE 'btc-vwap-capacity-%'
            AND status NOT IN ('completed', 'failed', 'cancelled')
          LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'cancel active BTC VWAP capacity backfills before expanding the materialization';
        END IF;
      END $$;

      TRUNCATE TABLE polymarket.btc_market_capacity_execution_snapshots;

      ALTER TABLE polymarket.backfill_artifacts
        DISABLE TRIGGER trg_reject_completed_backfill_artifact_change;

      UPDATE polymarket.backfill_artifacts
      SET status = 'failed',
          record_count = 0,
          minimum_source_timestamp = NULL,
          maximum_source_timestamp = NULL,
          metadata = metadata || jsonb_build_object(
            'superseded_by_schema_version', 'btc5m-capacity-book-1-240s-v2',
            'superseded_reason', 'expanded exact VWAP tiers through 200 shares'
          ),
          updated_at = now()
      WHERE provider = 'pmxt_v2_capacity_execution_snapshots';

      ALTER TABLE polymarket.backfill_artifacts
        ENABLE TRIGGER trg_reject_completed_backfill_artifact_change;

      ALTER TABLE polymarket.btc_market_capacity_execution_snapshots
        ADD COLUMN up_ask_vwap_25 numeric(18,8),
        ADD COLUMN up_ask_vwap_30 numeric(18,8),
        ADD COLUMN up_ask_vwap_40 numeric(18,8),
        ADD COLUMN up_ask_vwap_50 numeric(18,8),
        ADD COLUMN up_ask_vwap_75 numeric(18,8),
        ADD COLUMN up_ask_vwap_100 numeric(18,8),
        ADD COLUMN up_ask_vwap_125 numeric(18,8),
        ADD COLUMN up_ask_vwap_150 numeric(18,8),
        ADD COLUMN up_ask_vwap_175 numeric(18,8),
        ADD COLUMN up_ask_vwap_200 numeric(18,8),
        ADD COLUMN down_ask_vwap_25 numeric(18,8),
        ADD COLUMN down_ask_vwap_30 numeric(18,8),
        ADD COLUMN down_ask_vwap_40 numeric(18,8),
        ADD COLUMN down_ask_vwap_50 numeric(18,8),
        ADD COLUMN down_ask_vwap_75 numeric(18,8),
        ADD COLUMN down_ask_vwap_100 numeric(18,8),
        ADD COLUMN down_ask_vwap_125 numeric(18,8),
        ADD COLUMN down_ask_vwap_150 numeric(18,8),
        ADD COLUMN down_ask_vwap_175 numeric(18,8),
        ADD COLUMN down_ask_vwap_200 numeric(18,8),
        ADD CONSTRAINT chk_btc_market_capacity_execution_snapshot_expanded_prices CHECK (
          (up_ask_vwap_25 IS NULL OR up_ask_vwap_25 BETWEEN 0 AND 1)
          AND (up_ask_vwap_30 IS NULL OR up_ask_vwap_30 BETWEEN 0 AND 1)
          AND (up_ask_vwap_40 IS NULL OR up_ask_vwap_40 BETWEEN 0 AND 1)
          AND (up_ask_vwap_50 IS NULL OR up_ask_vwap_50 BETWEEN 0 AND 1)
          AND (up_ask_vwap_75 IS NULL OR up_ask_vwap_75 BETWEEN 0 AND 1)
          AND (up_ask_vwap_100 IS NULL OR up_ask_vwap_100 BETWEEN 0 AND 1)
          AND (up_ask_vwap_125 IS NULL OR up_ask_vwap_125 BETWEEN 0 AND 1)
          AND (up_ask_vwap_150 IS NULL OR up_ask_vwap_150 BETWEEN 0 AND 1)
          AND (up_ask_vwap_175 IS NULL OR up_ask_vwap_175 BETWEEN 0 AND 1)
          AND (up_ask_vwap_200 IS NULL OR up_ask_vwap_200 BETWEEN 0 AND 1)
          AND (down_ask_vwap_25 IS NULL OR down_ask_vwap_25 BETWEEN 0 AND 1)
          AND (down_ask_vwap_30 IS NULL OR down_ask_vwap_30 BETWEEN 0 AND 1)
          AND (down_ask_vwap_40 IS NULL OR down_ask_vwap_40 BETWEEN 0 AND 1)
          AND (down_ask_vwap_50 IS NULL OR down_ask_vwap_50 BETWEEN 0 AND 1)
          AND (down_ask_vwap_75 IS NULL OR down_ask_vwap_75 BETWEEN 0 AND 1)
          AND (down_ask_vwap_100 IS NULL OR down_ask_vwap_100 BETWEEN 0 AND 1)
          AND (down_ask_vwap_125 IS NULL OR down_ask_vwap_125 BETWEEN 0 AND 1)
          AND (down_ask_vwap_150 IS NULL OR down_ask_vwap_150 BETWEEN 0 AND 1)
          AND (down_ask_vwap_175 IS NULL OR down_ask_vwap_175 BETWEEN 0 AND 1)
          AND (down_ask_vwap_200 IS NULL OR down_ask_vwap_200 BETWEEN 0 AND 1)
        ),
        ADD CONSTRAINT chk_btc_market_capacity_execution_snapshot_expanded_vwap CHECK (
          (up_ask_vwap_25 IS NULL OR up_ask_vwap_20 IS NULL OR up_ask_vwap_25 >= up_ask_vwap_20)
          AND (up_ask_vwap_30 IS NULL OR up_ask_vwap_25 IS NULL OR up_ask_vwap_30 >= up_ask_vwap_25)
          AND (up_ask_vwap_40 IS NULL OR up_ask_vwap_30 IS NULL OR up_ask_vwap_40 >= up_ask_vwap_30)
          AND (up_ask_vwap_50 IS NULL OR up_ask_vwap_40 IS NULL OR up_ask_vwap_50 >= up_ask_vwap_40)
          AND (up_ask_vwap_75 IS NULL OR up_ask_vwap_50 IS NULL OR up_ask_vwap_75 >= up_ask_vwap_50)
          AND (up_ask_vwap_100 IS NULL OR up_ask_vwap_75 IS NULL OR up_ask_vwap_100 >= up_ask_vwap_75)
          AND (up_ask_vwap_125 IS NULL OR up_ask_vwap_100 IS NULL OR up_ask_vwap_125 >= up_ask_vwap_100)
          AND (up_ask_vwap_150 IS NULL OR up_ask_vwap_125 IS NULL OR up_ask_vwap_150 >= up_ask_vwap_125)
          AND (up_ask_vwap_175 IS NULL OR up_ask_vwap_150 IS NULL OR up_ask_vwap_175 >= up_ask_vwap_150)
          AND (up_ask_vwap_200 IS NULL OR up_ask_vwap_175 IS NULL OR up_ask_vwap_200 >= up_ask_vwap_175)
          AND (down_ask_vwap_25 IS NULL OR down_ask_vwap_20 IS NULL OR down_ask_vwap_25 >= down_ask_vwap_20)
          AND (down_ask_vwap_30 IS NULL OR down_ask_vwap_25 IS NULL OR down_ask_vwap_30 >= down_ask_vwap_25)
          AND (down_ask_vwap_40 IS NULL OR down_ask_vwap_30 IS NULL OR down_ask_vwap_40 >= down_ask_vwap_30)
          AND (down_ask_vwap_50 IS NULL OR down_ask_vwap_40 IS NULL OR down_ask_vwap_50 >= down_ask_vwap_40)
          AND (down_ask_vwap_75 IS NULL OR down_ask_vwap_50 IS NULL OR down_ask_vwap_75 >= down_ask_vwap_50)
          AND (down_ask_vwap_100 IS NULL OR down_ask_vwap_75 IS NULL OR down_ask_vwap_100 >= down_ask_vwap_75)
          AND (down_ask_vwap_125 IS NULL OR down_ask_vwap_100 IS NULL OR down_ask_vwap_125 >= down_ask_vwap_100)
          AND (down_ask_vwap_150 IS NULL OR down_ask_vwap_125 IS NULL OR down_ask_vwap_150 >= down_ask_vwap_125)
          AND (down_ask_vwap_175 IS NULL OR down_ask_vwap_150 IS NULL OR down_ask_vwap_175 >= down_ask_vwap_150)
          AND (down_ask_vwap_200 IS NULL OR down_ask_vwap_175 IS NULL OR down_ask_vwap_200 >= down_ask_vwap_175)
        );
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
          FROM polymarket.btc_market_capacity_execution_snapshots
          LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to remove expanded VWAP columns while capacity facts exist';
        END IF;
      END $$;

      ALTER TABLE polymarket.btc_market_capacity_execution_snapshots
        DROP CONSTRAINT chk_btc_market_capacity_execution_snapshot_expanded_vwap,
        DROP CONSTRAINT chk_btc_market_capacity_execution_snapshot_expanded_prices,
        DROP COLUMN up_ask_vwap_25,
        DROP COLUMN up_ask_vwap_30,
        DROP COLUMN up_ask_vwap_40,
        DROP COLUMN up_ask_vwap_50,
        DROP COLUMN up_ask_vwap_75,
        DROP COLUMN up_ask_vwap_100,
        DROP COLUMN up_ask_vwap_125,
        DROP COLUMN up_ask_vwap_150,
        DROP COLUMN up_ask_vwap_175,
        DROP COLUMN up_ask_vwap_200,
        DROP COLUMN down_ask_vwap_25,
        DROP COLUMN down_ask_vwap_30,
        DROP COLUMN down_ask_vwap_40,
        DROP COLUMN down_ask_vwap_50,
        DROP COLUMN down_ask_vwap_75,
        DROP COLUMN down_ask_vwap_100,
        DROP COLUMN down_ask_vwap_125,
        DROP COLUMN down_ask_vwap_150,
        DROP COLUMN down_ask_vwap_175,
        DROP COLUMN down_ask_vwap_200;
    `);
  }
}
