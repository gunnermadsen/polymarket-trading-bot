import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetirePolymarketNegativeRiskConversions1784846090000
  implements MigrationInterface
{
  name = 'RetirePolymarketNegativeRiskConversions1784846090000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET LOCAL lock_timeout = '5s';`);

    await queryRunner.query(`
      DO $$
      DECLARE
        conversions_oid oid := to_regclass('polymarket.conversions');
        actual_columns jsonb;
        expected_columns constant jsonb := '[
          {"name":"conversion_id","type":"uuid","not_null":true,"has_default":false},
          {"name":"timestamp_utc","type":"timestamp with time zone","not_null":true,"has_default":false},
          {"name":"market_id","type":"text","not_null":true,"has_default":false},
          {"name":"no_token_id","type":"text","not_null":true,"has_default":false},
          {"name":"size","type":"numeric(30,10)","not_null":true,"has_default":false},
          {"name":"status","type":"text","not_null":true,"has_default":false},
          {"name":"tx_hash","type":"text","not_null":false,"has_default":false},
          {"name":"latency_ms","type":"bigint","not_null":false,"has_default":false},
          {"name":"gas_cost_usd","type":"numeric(30,10)","not_null":true,"has_default":true},
          {"name":"raw_payload","type":"jsonb","not_null":true,"has_default":true},
          {"name":"created_at","type":"timestamp with time zone","not_null":true,"has_default":true},
          {"name":"updated_at","type":"timestamp with time zone","not_null":true,"has_default":true}
        ]'::jsonb;
        dependency record;
      BEGIN
        IF conversions_oid IS NULL THEN
          RAISE EXCEPTION
            'refusing to retire negative-risk conversions: polymarket.conversions is missing';
        END IF;

        SELECT jsonb_agg(
          jsonb_build_object(
            'name', attribute.attname,
            'type', format_type(attribute.atttypid, attribute.atttypmod),
            'not_null', attribute.attnotnull,
            'has_default', attribute_default.adbin IS NOT NULL
          )
          ORDER BY attribute.attnum
        )
        INTO actual_columns
        FROM pg_attribute attribute
        LEFT JOIN pg_attrdef attribute_default
          ON attribute_default.adrelid = attribute.attrelid
         AND attribute_default.adnum = attribute.attnum
        WHERE attribute.attrelid = conversions_oid
          AND attribute.attnum > 0
          AND NOT attribute.attisdropped;

        IF actual_columns IS DISTINCT FROM expected_columns THEN
          RAISE EXCEPTION
            'refusing to retire negative-risk conversions: unexpected polymarket.conversions columns: %',
            actual_columns;
        END IF;

        IF NOT EXISTS (
          SELECT 1
          FROM pg_constraint constraint_record
          WHERE constraint_record.conrelid = conversions_oid
            AND constraint_record.contype = 'p'
            AND constraint_record.conname = 'pk_polymarket_conversions'
            AND ARRAY(
              SELECT attribute.attname::text
              FROM unnest(constraint_record.conkey)
                WITH ORDINALITY AS key_column(attnum, position)
              JOIN pg_attribute attribute
                ON attribute.attrelid = constraint_record.conrelid
               AND attribute.attnum = key_column.attnum
              ORDER BY key_column.position
            ) = ARRAY['conversion_id', 'timestamp_utc']
        ) THEN
          RAISE EXCEPTION
            'refusing to retire negative-risk conversions: unexpected primary key';
        END IF;

        IF NOT EXISTS (
          SELECT 1
          FROM pg_index index_record
          JOIN pg_class index_relation
            ON index_relation.oid = index_record.indexrelid
          WHERE index_record.indrelid = conversions_oid
            AND index_relation.relname = 'idx_polymarket_conversions_market_ts'
            AND index_record.indisvalid
            AND NOT index_record.indisunique
            AND index_record.indnkeyatts = 2
            AND index_record.indnatts = 2
            AND ARRAY(
              SELECT attribute.attname::text
              FROM unnest(index_record.indkey::smallint[])
                WITH ORDINALITY AS key_column(attnum, position)
              JOIN pg_attribute attribute
                ON attribute.attrelid = index_record.indrelid
               AND attribute.attnum = key_column.attnum
              ORDER BY key_column.position
            ) = ARRAY['market_id', 'timestamp_utc']
            AND NOT pg_index_column_has_property(
              index_record.indexrelid, 1, 'desc'
            )
            AND pg_index_column_has_property(
              index_record.indexrelid, 2, 'desc'
            )
        ) THEN
          RAISE EXCEPTION
            'refusing to retire negative-risk conversions: unexpected market-time index';
        END IF;

        IF to_regclass('timescaledb_information.hypertables') IS NULL
          OR NOT EXISTS (
            SELECT 1
            FROM timescaledb_information.hypertables
            WHERE hypertable_schema = 'polymarket'
              AND hypertable_name = 'conversions'
          )
        THEN
          RAISE EXCEPTION
            'refusing to retire negative-risk conversions: polymarket.conversions is not a hypertable';
        END IF;

        SELECT
          source_namespace.nspname AS source_schema,
          source_table.relname AS source_table,
          constraint_record.conname AS constraint_name
        INTO dependency
        FROM pg_constraint constraint_record
        JOIN pg_class source_table
          ON source_table.oid = constraint_record.conrelid
        JOIN pg_namespace source_namespace
          ON source_namespace.oid = source_table.relnamespace
        WHERE constraint_record.contype = 'f'
          AND constraint_record.confrelid = conversions_oid
          AND constraint_record.conrelid <> conversions_oid
          AND source_namespace.nspname NOT LIKE '\\_timescaledb\\_%' ESCAPE '\\'
        LIMIT 1;

        IF FOUND THEN
          RAISE EXCEPTION
            'refusing to retire negative-risk conversions: %.% constraint % still references polymarket.conversions',
            dependency.source_schema,
            dependency.source_table,
            dependency.constraint_name;
        END IF;

        SELECT
          view_namespace.nspname AS source_schema,
          view_relation.relname AS source_table
        INTO dependency
        FROM pg_depend dependency_record
        JOIN pg_rewrite rewrite_record
          ON rewrite_record.oid = dependency_record.objid
        JOIN pg_class view_relation
          ON view_relation.oid = rewrite_record.ev_class
        JOIN pg_namespace view_namespace
          ON view_namespace.oid = view_relation.relnamespace
        WHERE dependency_record.classid = 'pg_rewrite'::regclass
          AND view_relation.relkind IN ('v', 'm')
          AND dependency_record.refobjid = conversions_oid
        LIMIT 1;

        IF FOUND THEN
          RAISE EXCEPTION
            'refusing to retire negative-risk conversions: view %.% still references polymarket.conversions',
            dependency.source_schema,
            dependency.source_table;
        END IF;
      END $$;
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        PERFORM remove_retention_policy(
          'polymarket.conversions', if_exists => true
        );
        PERFORM remove_compression_policy(
          'polymarket.conversions', if_exists => true
        );
      END $$;
    `);

    await queryRunner.query(`DROP TABLE polymarket.conversions;`);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    throw new Error(
      'RetirePolymarketNegativeRiskConversions1784846090000 is intentionally irreversible',
    );
  }
}
