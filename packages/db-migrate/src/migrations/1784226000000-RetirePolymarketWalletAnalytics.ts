import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetirePolymarketWalletAnalytics1784226000000 implements MigrationInterface {
  name = 'RetirePolymarketWalletAnalytics1784226000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      DECLARE
        dependency record;
      BEGIN
        SELECT
          source_namespace.nspname AS source_schema,
          source_table.relname AS source_table,
          constraint_record.conname AS constraint_name,
          target_namespace.nspname AS target_schema,
          target_table.relname AS target_table
        INTO dependency
        FROM pg_constraint constraint_record
        JOIN pg_class source_table
          ON source_table.oid = constraint_record.conrelid
        JOIN pg_namespace source_namespace
          ON source_namespace.oid = source_table.relnamespace
        JOIN pg_class target_table
          ON target_table.oid = constraint_record.confrelid
        JOIN pg_namespace target_namespace
          ON target_namespace.oid = target_table.relnamespace
        WHERE constraint_record.contype = 'f'
          AND constraint_record.confrelid = ANY (ARRAY[
            to_regclass('polymarket.wallets'),
            to_regclass('polymarket.wallet_trades'),
            to_regclass('polymarket.wallet_positions'),
            to_regclass('polymarket.wallet_scores'),
            to_regclass('polymarket.wallet_performance'),
            to_regclass('polymarket.wallet_segment_performance'),
            to_regclass('polymarket.wallet_score_refresh_jobs'),
            to_regclass('polymarket.gamma_market_metadata')
          ]::oid[])
          AND constraint_record.conrelid <> ALL (ARRAY[
            to_regclass('polymarket.wallets'),
            to_regclass('polymarket.wallet_trades'),
            to_regclass('polymarket.wallet_positions'),
            to_regclass('polymarket.wallet_scores'),
            to_regclass('polymarket.wallet_performance'),
            to_regclass('polymarket.wallet_segment_performance'),
            to_regclass('polymarket.wallet_score_refresh_jobs'),
            to_regclass('polymarket.gamma_market_metadata')
          ]::oid[])
          AND source_namespace.nspname NOT LIKE '\\_timescaledb\\_%' ESCAPE '\\'
        LIMIT 1;

        IF FOUND THEN
          RAISE EXCEPTION
            'refusing to retire wallet analytics: %.% constraint % still references %.%',
            dependency.source_schema,
            dependency.source_table,
            dependency.constraint_name,
            dependency.target_schema,
            dependency.target_table;
        END IF;

        SELECT
          view_namespace.nspname AS source_schema,
          view_relation.relname AS source_table,
          target_namespace.nspname AS target_schema,
          target_relation.relname AS target_table
        INTO dependency
        FROM pg_depend dependency_record
        JOIN pg_rewrite rewrite_record
          ON rewrite_record.oid = dependency_record.objid
        JOIN pg_class view_relation
          ON view_relation.oid = rewrite_record.ev_class
        JOIN pg_namespace view_namespace
          ON view_namespace.oid = view_relation.relnamespace
        JOIN pg_class target_relation
          ON target_relation.oid = dependency_record.refobjid
        JOIN pg_namespace target_namespace
          ON target_namespace.oid = target_relation.relnamespace
        WHERE dependency_record.classid = 'pg_rewrite'::regclass
          AND view_relation.relkind IN ('v', 'm')
          AND dependency_record.refobjid = ANY (ARRAY[
            to_regclass('polymarket.wallets'),
            to_regclass('polymarket.wallet_trades'),
            to_regclass('polymarket.wallet_positions'),
            to_regclass('polymarket.wallet_scores'),
            to_regclass('polymarket.wallet_performance'),
            to_regclass('polymarket.wallet_segment_performance'),
            to_regclass('polymarket.wallet_score_refresh_jobs'),
            to_regclass('polymarket.gamma_market_metadata')
          ]::oid[])
        LIMIT 1;

        IF FOUND THEN
          RAISE EXCEPTION
            'refusing to retire wallet analytics: view %.% still references %.%',
            dependency.source_schema,
            dependency.source_table,
            dependency.target_schema,
            dependency.target_table;
        END IF;
      END $$;
    `);

    await queryRunner.query(`
      DROP TABLE polymarket.wallet_score_refresh_jobs;
      DROP TABLE polymarket.wallet_segment_performance;
      DROP TABLE polymarket.wallet_scores;
      DROP TABLE polymarket.wallet_positions;
      DROP TABLE polymarket.wallet_trades;
      DROP TABLE polymarket.wallet_performance;
      DROP TABLE polymarket.gamma_market_metadata;
      DROP TABLE polymarket.wallets;
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    throw new Error(
      'RetirePolymarketWalletAnalytics1784226000000 is intentionally irreversible',
    );
  }
}
