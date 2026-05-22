import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolymarketMarkSourceFailures1777112000000 implements MigrationInterface {
  name = 'AddPolymarketMarkSourceFailures1777112000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE IF NOT EXISTS polymarket.trade_mark_source_failures (
        failure_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        process_id uuid REFERENCES polymarket.trading_processes (process_id) ON DELETE CASCADE,
        position_id uuid REFERENCES polymarket.trade_positions (position_id) ON DELETE CASCADE,
        token_id text NOT NULL,
        market_id text,
        failure_source text NOT NULL,
        failure_reason text NOT NULL,
        failure_count bigint NOT NULL DEFAULT 1,
        first_failed_at timestamptz NOT NULL DEFAULT now(),
        last_failed_at timestamptz NOT NULL DEFAULT now(),
        resolved_at timestamptz,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT chk_poly_trade_mark_failures_token CHECK (length(btrim(token_id)) > 0),
        CONSTRAINT chk_poly_trade_mark_failures_source CHECK (length(btrim(failure_source)) > 0),
        CONSTRAINT chk_poly_trade_mark_failures_reason CHECK (length(btrim(failure_reason)) > 0),
        CONSTRAINT chk_poly_trade_mark_failures_count CHECK (failure_count > 0)
      );
    `);

    await queryRunner.query(`
      CREATE UNIQUE INDEX IF NOT EXISTS uq_poly_trade_mark_failures_scope
        ON polymarket.trade_mark_source_failures (
          COALESCE(process_id, '00000000-0000-0000-0000-000000000000'::uuid),
          COALESCE(position_id, '00000000-0000-0000-0000-000000000000'::uuid),
          token_id,
          failure_source
        );
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_trade_mark_failures_unresolved
        ON polymarket.trade_mark_source_failures (process_id, failure_source, last_failed_at DESC)
        WHERE resolved_at IS NULL;
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_trade_mark_failures_token
        ON polymarket.trade_mark_source_failures (token_id, last_failed_at DESC);
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_trade_mark_failures_token;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_trade_mark_failures_unresolved;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.uq_poly_trade_mark_failures_scope;`);
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.trade_mark_source_failures;`);
  }
}
