import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetirePolymarketLegacyMarketScaffold1784662000000
  implements MigrationInterface
{
  name = 'RetirePolymarketLegacyMarketScaffold1784662000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET LOCAL lock_timeout = '5s';`);

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
          AND constraint_record.confrelid IN (
            SELECT target_oid
            FROM unnest(ARRAY[
              to_regclass('polymarket.markets'),
              to_regclass('polymarket.outcome_tokens'),
              to_regclass('polymarket.underlying_overrides')
            ]::oid[]) AS target(target_oid)
            WHERE target_oid IS NOT NULL
          )
          AND constraint_record.conrelid NOT IN (
            SELECT target_oid
            FROM unnest(ARRAY[
              to_regclass('polymarket.markets'),
              to_regclass('polymarket.outcome_tokens'),
              to_regclass('polymarket.underlying_overrides')
            ]::oid[]) AS target(target_oid)
            WHERE target_oid IS NOT NULL
          )
        LIMIT 1;

        IF FOUND THEN
          RAISE EXCEPTION
            'refusing to retire legacy market scaffold: %.% constraint % still references %.%',
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
          AND dependency_record.refobjid IN (
            SELECT target_oid
            FROM unnest(ARRAY[
              to_regclass('polymarket.markets'),
              to_regclass('polymarket.outcome_tokens'),
              to_regclass('polymarket.underlying_overrides')
            ]::oid[]) AS target(target_oid)
            WHERE target_oid IS NOT NULL
          )
        LIMIT 1;

        IF FOUND THEN
          RAISE EXCEPTION
            'refusing to retire legacy market scaffold: view %.% still references %.%',
            dependency.source_schema,
            dependency.source_table,
            dependency.target_schema,
            dependency.target_table;
        END IF;
      END $$;
    `);

    await queryRunner.query(`
      DROP TABLE IF EXISTS polymarket.outcome_tokens;
      DROP TABLE IF EXISTS polymarket.underlying_overrides;
      DROP TABLE IF EXISTS polymarket.markets;
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    throw new Error(
      'RetirePolymarketLegacyMarketScaffold1784662000000 is intentionally irreversible',
    );
  }
}
