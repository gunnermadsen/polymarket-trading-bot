import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolymarketWalletScoreRefreshQueue1777119000000 implements MigrationInterface {
  name = 'AddPolymarketWalletScoreRefreshQueue1777119000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`CREATE EXTENSION IF NOT EXISTS pgcrypto;`);
    await queryRunner.query(`CREATE SCHEMA IF NOT EXISTS polymarket;`);

    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.wallet_score_refresh_jobs (
        queue_id uuid NOT NULL DEFAULT gen_random_uuid(),
        proxy_wallet text NOT NULL REFERENCES polymarket.wallets (proxy_wallet) ON DELETE CASCADE,
        score_version text NOT NULL,
        segment_score_version text NOT NULL,
        status text NOT NULL DEFAULT 'queued',
        refresh_reason text NOT NULL DEFAULT 'unspecified',
        last_seen_trade_id uuid,
        requested_at timestamptz NOT NULL DEFAULT now(),
        available_at timestamptz NOT NULL DEFAULT now(),
        started_at timestamptz,
        completed_at timestamptz,
        attempt_count integer NOT NULL DEFAULT 0,
        max_attempts integer NOT NULL DEFAULT 3,
        last_error text,
        request_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        result_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_poly_wallet_score_refresh_jobs PRIMARY KEY (queue_id),
        CONSTRAINT uq_poly_wallet_score_refresh_pending UNIQUE (
          proxy_wallet,
          score_version,
          segment_score_version
        ),
        CONSTRAINT chk_poly_wallet_score_refresh_status CHECK (
          status IN ('queued', 'running', 'completed', 'failed', 'cancelled')
        ),
        CONSTRAINT chk_poly_wallet_score_refresh_attempts CHECK (
          attempt_count >= 0 AND max_attempts > 0
        )
      );
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_score_refresh_queue_status
        ON polymarket.wallet_score_refresh_jobs (status, available_at ASC, requested_at ASC);
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_score_refresh_queue_updated
        ON polymarket.wallet_score_refresh_jobs (updated_at DESC);
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_wallet_score_refresh_queue_updated;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_wallet_score_refresh_queue_status;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.wallet_score_refresh_jobs;`);
  }
}
