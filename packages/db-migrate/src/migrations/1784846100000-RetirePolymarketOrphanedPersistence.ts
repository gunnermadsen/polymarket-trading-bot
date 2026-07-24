import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetirePolymarketOrphanedPersistence1784846100000
  implements MigrationInterface
{
  name = 'RetirePolymarketOrphanedPersistence1784846100000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET LOCAL lock_timeout = '5s';`);
    await queryRunner.query(`
      LOCK TABLE
        polymarket.daily_metrics,
        polymarket.live_idempotency_repairs
      IN ACCESS EXCLUSIVE MODE;
    `);

    await queryRunner.query(`
      DO $$
      DECLARE
        daily_metrics_oid oid := to_regclass('polymarket.daily_metrics');
        repairs_oid oid := to_regclass('polymarket.live_idempotency_repairs');
        live_events_oid oid := to_regclass('polymarket.live_venue_events');
        actual_columns jsonb;
        dependency record;
      BEGIN
        IF daily_metrics_oid IS NULL THEN
          RAISE EXCEPTION
            'refusing to retire orphaned persistence: polymarket.daily_metrics is missing';
        END IF;

        IF repairs_oid IS NULL THEN
          RAISE EXCEPTION
            'refusing to retire orphaned persistence: polymarket.live_idempotency_repairs is missing';
        END IF;

        IF live_events_oid IS NULL THEN
          RAISE EXCEPTION
            'refusing to retire orphaned persistence: preserved polymarket.live_venue_events is missing';
        END IF;

        IF (SELECT relkind FROM pg_class WHERE oid = daily_metrics_oid) <> 'r'
          OR (SELECT relkind FROM pg_class WHERE oid = repairs_oid) <> 'r'
        THEN
          RAISE EXCEPTION
            'refusing to retire orphaned persistence: target relation kind has drifted';
        END IF;

        SELECT jsonb_agg(
          jsonb_build_object(
            'name', attribute.attname,
            'type', format_type(attribute.atttypid, attribute.atttypmod),
            'not_null', attribute.attnotnull,
            'default', pg_get_expr(
              attribute_default.adbin,
              attribute_default.adrelid
            )
          )
          ORDER BY attribute.attnum
        )
        INTO actual_columns
        FROM pg_attribute attribute
        LEFT JOIN pg_attrdef attribute_default
          ON attribute_default.adrelid = attribute.attrelid
         AND attribute_default.adnum = attribute.attnum
        WHERE attribute.attrelid = daily_metrics_oid
          AND attribute.attnum > 0
          AND NOT attribute.attisdropped;

        IF actual_columns IS DISTINCT FROM '[
          {"name":"metric_date","type":"date","not_null":true,"default":null},
          {"name":"metric_name","type":"text","not_null":true,"default":null},
          {"name":"value","type":"jsonb","not_null":true,"default":"''{}''::jsonb"},
          {"name":"created_at","type":"timestamp with time zone","not_null":true,"default":"now()"},
          {"name":"updated_at","type":"timestamp with time zone","not_null":true,"default":"now()"}
        ]'::jsonb THEN
          RAISE EXCEPTION
            'refusing to retire orphaned persistence: unexpected polymarket.daily_metrics columns: %',
            actual_columns;
        END IF;

        SELECT jsonb_agg(
          jsonb_build_object(
            'name', attribute.attname,
            'type', format_type(attribute.atttypid, attribute.atttypmod),
            'not_null', attribute.attnotnull,
            'default', pg_get_expr(
              attribute_default.adbin,
              attribute_default.adrelid
            )
          )
          ORDER BY attribute.attnum
        )
        INTO actual_columns
        FROM pg_attribute attribute
        LEFT JOIN pg_attrdef attribute_default
          ON attribute_default.adrelid = attribute.attrelid
         AND attribute_default.adnum = attribute.attnum
        WHERE attribute.attrelid = repairs_oid
          AND attribute.attnum > 0
          AND NOT attribute.attisdropped;

        IF actual_columns IS DISTINCT FROM '[
          {"name":"repair_id","type":"uuid","not_null":true,"default":"gen_random_uuid()"},
          {"name":"created_at","type":"timestamp with time zone","not_null":true,"default":"now()"},
          {"name":"repair_type","type":"text","not_null":true,"default":null},
          {"name":"client_order_id","type":"uuid","not_null":false,"default":null},
          {"name":"venue_order_id","type":"text","not_null":false,"default":null},
          {"name":"source_event_id","type":"uuid","not_null":false,"default":null},
          {"name":"before_state","type":"jsonb","not_null":true,"default":"''{}''::jsonb"},
          {"name":"after_state","type":"jsonb","not_null":true,"default":"''{}''::jsonb"},
          {"name":"status","type":"text","not_null":true,"default":null}
        ]'::jsonb THEN
          RAISE EXCEPTION
            'refusing to retire orphaned persistence: unexpected polymarket.live_idempotency_repairs columns: %',
            actual_columns;
        END IF;

        IF (
          SELECT count(*)
          FROM pg_constraint
          WHERE conrelid = daily_metrics_oid
        ) <> 1
          OR (
            SELECT count(*)
            FROM pg_index
            WHERE indrelid = daily_metrics_oid
          ) <> 1
          OR NOT EXISTS (
            SELECT 1
            FROM pg_constraint constraint_record
            JOIN pg_index backing_index
              ON backing_index.indexrelid = constraint_record.conindid
            JOIN pg_class index_relation
              ON index_relation.oid = backing_index.indexrelid
            JOIN pg_am index_method
              ON index_method.oid = index_relation.relam
            WHERE constraint_record.conrelid = daily_metrics_oid
              AND constraint_record.contype = 'p'
              AND constraint_record.conname = 'pk_polymarket_daily_metrics'
              AND constraint_record.convalidated
              AND NOT constraint_record.condeferrable
              AND NOT constraint_record.condeferred
              AND ARRAY(
                SELECT attribute.attname::text
                FROM unnest(constraint_record.conkey)
                  WITH ORDINALITY AS key_column(attnum, position)
                JOIN pg_attribute attribute
                  ON attribute.attrelid = constraint_record.conrelid
                 AND attribute.attnum = key_column.attnum
                ORDER BY key_column.position
              ) = ARRAY['metric_date', 'metric_name']
              AND index_relation.relname = 'pk_polymarket_daily_metrics'
              AND index_method.amname = 'btree'
              AND backing_index.indisprimary
              AND backing_index.indisunique
              AND backing_index.indisvalid
              AND backing_index.indisready
              AND backing_index.indislive
              AND NOT backing_index.indisexclusion
              AND backing_index.indnkeyatts = 2
              AND backing_index.indnatts = 2
              AND backing_index.indkey[0] = constraint_record.conkey[1]
              AND backing_index.indkey[1] = constraint_record.conkey[2]
              AND backing_index.indoption[0] = 0
              AND backing_index.indoption[1] = 0
              AND backing_index.indexprs IS NULL
              AND backing_index.indpred IS NULL
          )
        THEN
          RAISE EXCEPTION
            'refusing to retire orphaned persistence: unexpected polymarket.daily_metrics constraints';
        END IF;

        IF (
          SELECT count(*)
          FROM pg_constraint
          WHERE conrelid = repairs_oid
        ) <> 2
          OR (
            SELECT count(*)
            FROM pg_index
            WHERE indrelid = repairs_oid
          ) <> 1
          OR NOT EXISTS (
            SELECT 1
            FROM pg_constraint constraint_record
            JOIN pg_index backing_index
              ON backing_index.indexrelid = constraint_record.conindid
            JOIN pg_class index_relation
              ON index_relation.oid = backing_index.indexrelid
            JOIN pg_am index_method
              ON index_method.oid = index_relation.relam
            WHERE constraint_record.conrelid = repairs_oid
              AND constraint_record.contype = 'p'
              AND constraint_record.conname =
                    'live_idempotency_repairs_pkey'
              AND constraint_record.convalidated
              AND NOT constraint_record.condeferrable
              AND NOT constraint_record.condeferred
              AND ARRAY(
                SELECT attribute.attname::text
                FROM unnest(constraint_record.conkey)
                  WITH ORDINALITY AS key_column(attnum, position)
                JOIN pg_attribute attribute
                  ON attribute.attrelid = constraint_record.conrelid
                 AND attribute.attnum = key_column.attnum
                ORDER BY key_column.position
              ) = ARRAY['repair_id']
              AND index_relation.relname = 'live_idempotency_repairs_pkey'
              AND index_method.amname = 'btree'
              AND backing_index.indisprimary
              AND backing_index.indisunique
              AND backing_index.indisvalid
              AND backing_index.indisready
              AND backing_index.indislive
              AND NOT backing_index.indisexclusion
              AND backing_index.indnkeyatts = 1
              AND backing_index.indnatts = 1
              AND backing_index.indkey[0] = constraint_record.conkey[1]
              AND backing_index.indoption[0] = 0
              AND backing_index.indexprs IS NULL
              AND backing_index.indpred IS NULL
          )
          OR NOT EXISTS (
            SELECT 1
            FROM pg_constraint constraint_record
            JOIN pg_attribute source_attribute
              ON source_attribute.attrelid = constraint_record.conrelid
             AND source_attribute.attnum = constraint_record.conkey[1]
            JOIN pg_attribute target_attribute
              ON target_attribute.attrelid = constraint_record.confrelid
             AND target_attribute.attnum = constraint_record.confkey[1]
            WHERE constraint_record.conrelid = repairs_oid
              AND constraint_record.contype = 'f'
              AND constraint_record.conname =
                    'live_idempotency_repairs_source_event_id_fkey'
              AND constraint_record.confrelid = live_events_oid
              AND cardinality(constraint_record.conkey) = 1
              AND cardinality(constraint_record.confkey) = 1
              AND source_attribute.attname = 'source_event_id'
              AND target_attribute.attname = 'event_id'
              AND constraint_record.confdeltype = 'n'
              AND constraint_record.confupdtype = 'a'
              AND constraint_record.confmatchtype = 's'
              AND constraint_record.convalidated
              AND NOT constraint_record.condeferrable
              AND NOT constraint_record.condeferred
          )
        THEN
          RAISE EXCEPTION
            'refusing to retire orphaned persistence: unexpected polymarket.live_idempotency_repairs constraints';
        END IF;

        IF EXISTS (
          SELECT 1
          FROM pg_trigger
          WHERE tgrelid = ANY (ARRAY[daily_metrics_oid, repairs_oid])
            AND NOT tgisinternal
        ) OR EXISTS (
          SELECT 1
          FROM pg_rewrite
          WHERE ev_class = ANY (ARRAY[daily_metrics_oid, repairs_oid])
        ) OR EXISTS (
          SELECT 1
          FROM pg_policy
          WHERE polrelid = ANY (ARRAY[daily_metrics_oid, repairs_oid])
        ) OR EXISTS (
          SELECT 1
          FROM pg_class
          WHERE oid = ANY (ARRAY[daily_metrics_oid, repairs_oid])
            AND (relrowsecurity OR relforcerowsecurity)
        ) THEN
          RAISE EXCEPTION
            'refusing to retire orphaned persistence: target triggers, rules, or row-security policies have drifted';
        END IF;

        SELECT
          source_namespace.nspname AS source_schema,
          source_table.relname AS source_table,
          constraint_record.conname AS constraint_name,
          target_table.relname AS target_table
        INTO dependency
        FROM pg_constraint constraint_record
        JOIN pg_class source_table
          ON source_table.oid = constraint_record.conrelid
        JOIN pg_namespace source_namespace
          ON source_namespace.oid = source_table.relnamespace
        JOIN pg_class target_table
          ON target_table.oid = constraint_record.confrelid
        WHERE constraint_record.contype = 'f'
          AND constraint_record.confrelid = ANY (
            ARRAY[daily_metrics_oid, repairs_oid]
          )
          AND constraint_record.conrelid <> ALL (
            ARRAY[daily_metrics_oid, repairs_oid]
          )
        LIMIT 1;

        IF FOUND THEN
          RAISE EXCEPTION
            'refusing to retire orphaned persistence: %.% constraint % still references polymarket.%',
            dependency.source_schema,
            dependency.source_table,
            dependency.constraint_name,
            dependency.target_table;
        END IF;

        SELECT
          view_namespace.nspname AS source_schema,
          view_relation.relname AS source_table,
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
        WHERE dependency_record.classid = 'pg_rewrite'::regclass
          AND view_relation.relkind IN ('v', 'm')
          AND dependency_record.refobjid = ANY (
            ARRAY[daily_metrics_oid, repairs_oid]
          )
        LIMIT 1;

        IF FOUND THEN
          RAISE EXCEPTION
            'refusing to retire orphaned persistence: view %.% still references polymarket.%',
            dependency.source_schema,
            dependency.source_table,
            dependency.target_table;
        END IF;
      END $$;
    `);

    await queryRunner.query(`
      DROP TABLE polymarket.live_idempotency_repairs;
      DROP TABLE polymarket.daily_metrics;
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    throw new Error(
      'RetirePolymarketOrphanedPersistence1784846100000 is intentionally irreversible',
    );
  }
}
