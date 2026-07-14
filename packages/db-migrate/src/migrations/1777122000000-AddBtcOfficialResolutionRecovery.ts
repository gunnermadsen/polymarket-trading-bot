import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddBtcOfficialResolutionRecovery1777122000000 implements MigrationInterface {
  name = 'AddBtcOfficialResolutionRecovery1777122000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE polymarket.btc_interval_markets
        ADD COLUMN IF NOT EXISTS official_resolution_source text,
        ADD COLUMN IF NOT EXISTS official_resolution_received_at timestamptz,
        ADD COLUMN IF NOT EXISTS official_resolution_payload jsonb;
    `);

    await queryRunner.query(`
      UPDATE polymarket.btc_interval_markets
      SET official_resolution_source = COALESCE(
            official_resolution_source,
            'clob_websocket_legacy'
          ),
          official_resolution_received_at = COALESCE(
            official_resolution_received_at,
            official_resolved_at
          ),
          official_resolution_payload = COALESCE(
            official_resolution_payload,
            jsonb_build_object('recovered_from', 'pre_v10_official_resolution')
          )
      WHERE official_outcome IS NOT NULL;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.btc_interval_markets
        DROP CONSTRAINT IF EXISTS chk_btc_official_resolution_all_or_none,
        DROP CONSTRAINT IF EXISTS chk_btc_official_resolution_winner,
        DROP CONSTRAINT IF EXISTS chk_btc_official_resolution_time,
        DROP CONSTRAINT IF EXISTS chk_btc_official_resolution_provenance,
        ADD CONSTRAINT chk_btc_official_resolution_all_or_none CHECK (
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
        ADD CONSTRAINT chk_btc_official_resolution_winner CHECK (
          official_outcome IS NULL
          OR (official_outcome = 'up' AND official_winning_token_id = up_token_id)
          OR (official_outcome = 'down' AND official_winning_token_id = down_token_id)
        ),
        ADD CONSTRAINT chk_btc_official_resolution_time CHECK (
          official_resolved_at IS NULL
          OR (
            official_resolved_at >= window_end
            AND official_resolution_received_at >= window_end
          )
        ),
        ADD CONSTRAINT chk_btc_official_resolution_provenance CHECK (
          official_resolution_source IS NULL
          OR official_resolution_source IN (
            'clob_websocket',
            'clob_rest_reconciliation',
            'clob_websocket_legacy'
          )
        ),
        ADD CONSTRAINT chk_btc_official_resolution_payload CHECK (
          official_resolution_payload IS NULL
          OR jsonb_typeof(official_resolution_payload) = 'object'
        );
    `);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.btc_official_resolution_watches (
        market_id text PRIMARY KEY
          REFERENCES polymarket.btc_interval_markets (market_id) ON DELETE CASCADE,
        status text NOT NULL DEFAULT 'pending',
        watch_started_at timestamptz NOT NULL,
        deadline_at timestamptz NOT NULL,
        last_checked_at timestamptz,
        last_subscribed_at timestamptz,
        last_subscription_connection_id uuid,
        subscription_count bigint NOT NULL DEFAULT 0,
        resolution_received_at timestamptz,
        resolution_source text,
        expired_at timestamptz,
        last_error text,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_btc_resolution_watch_status CHECK (
          status IN ('pending','resolved','expired','resolved_late')
        ),
        CONSTRAINT chk_btc_resolution_watch_deadline CHECK (
          deadline_at > watch_started_at
        ),
        CONSTRAINT chk_btc_resolution_watch_subscription_count CHECK (
          subscription_count >= 0
        ),
        CONSTRAINT chk_btc_resolution_watch_source CHECK (
          resolution_source IS NULL
          OR resolution_source IN ('clob_websocket', 'clob_rest_reconciliation')
        ),
        CONSTRAINT chk_btc_resolution_watch_state CHECK (
          (status = 'pending'
            AND resolution_received_at IS NULL
            AND resolution_source IS NULL
            AND expired_at IS NULL)
          OR
          (status = 'resolved'
            AND resolution_received_at IS NOT NULL
            AND resolution_source IS NOT NULL
            AND expired_at IS NULL)
          OR
          (status = 'expired'
            AND resolution_received_at IS NULL
            AND resolution_source IS NULL
            AND expired_at IS NOT NULL)
          OR
          (status = 'resolved_late'
            AND resolution_received_at IS NOT NULL
            AND resolution_source IS NOT NULL
            AND expired_at IS NOT NULL)
        )
      );
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_btc_resolution_watches_pending_deadline
        ON polymarket.btc_official_resolution_watches (deadline_at, market_id)
        WHERE status = 'pending';
      CREATE INDEX IF NOT EXISTS idx_btc_interval_markets_pending_official
        ON polymarket.btc_interval_markets (window_end, window_start, market_id)
        WHERE official_outcome IS NULL;
      CREATE UNIQUE INDEX IF NOT EXISTS idx_btc_interval_markets_condition_id
        ON polymarket.btc_interval_markets (condition_id);
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.btc_official_resolution_watches;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_btc_interval_markets_pending_official;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_btc_interval_markets_condition_id;`);
    await queryRunner.query(`
      ALTER TABLE polymarket.btc_interval_markets
        DROP CONSTRAINT IF EXISTS chk_btc_official_resolution_time,
        DROP CONSTRAINT IF EXISTS chk_btc_official_resolution_winner,
        DROP CONSTRAINT IF EXISTS chk_btc_official_resolution_provenance,
        DROP CONSTRAINT IF EXISTS chk_btc_official_resolution_payload,
        DROP CONSTRAINT IF EXISTS chk_btc_official_resolution_all_or_none,
        DROP COLUMN IF EXISTS official_resolution_payload,
        DROP COLUMN IF EXISTS official_resolution_received_at,
        DROP COLUMN IF EXISTS official_resolution_source;
    `);
  }
}
