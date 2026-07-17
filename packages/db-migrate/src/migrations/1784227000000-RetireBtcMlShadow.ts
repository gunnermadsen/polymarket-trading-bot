import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetireBtcMlShadow1784227000000 implements MigrationInterface {
  name = 'RetireBtcMlShadow1784227000000';

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
            to_regclass('polymarket.ml_feature_vectors'),
            to_regclass('polymarket.ml_dataset_manifests'),
            to_regclass('polymarket.ml_model_versions'),
            to_regclass('polymarket.ml_shadow_predictions'),
            to_regclass('polymarket.ml_evaluation_runs'),
            to_regclass('polymarket.ml_evaluation_metrics')
          ]::oid[])
          AND constraint_record.conrelid <> ALL (ARRAY[
            to_regclass('polymarket.ml_feature_vectors'),
            to_regclass('polymarket.ml_dataset_manifests'),
            to_regclass('polymarket.ml_model_versions'),
            to_regclass('polymarket.ml_shadow_predictions'),
            to_regclass('polymarket.ml_evaluation_runs'),
            to_regclass('polymarket.ml_evaluation_metrics')
          ]::oid[])
          AND source_namespace.nspname NOT LIKE '\\_timescaledb\\_%' ESCAPE '\\'
        LIMIT 1;

        IF FOUND THEN
          RAISE EXCEPTION
            'refusing to retire BTC ML shadow: %.% constraint % still references %.%',
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
            to_regclass('polymarket.ml_feature_vectors'),
            to_regclass('polymarket.ml_dataset_manifests'),
            to_regclass('polymarket.ml_model_versions'),
            to_regclass('polymarket.ml_shadow_predictions'),
            to_regclass('polymarket.ml_evaluation_runs'),
            to_regclass('polymarket.ml_evaluation_metrics')
          ]::oid[])
        LIMIT 1;

        IF FOUND THEN
          RAISE EXCEPTION
            'refusing to retire BTC ML shadow: view %.% still references %.%',
            dependency.source_schema,
            dependency.source_table,
            dependency.target_schema,
            dependency.target_table;
        END IF;
      END $$;
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        PERFORM remove_retention_policy(
          'polymarket.ml_feature_vectors', if_exists => true
        );
        PERFORM remove_retention_policy(
          'polymarket.ml_shadow_predictions', if_exists => true
        );
        PERFORM remove_compression_policy(
          'polymarket.ml_feature_vectors', if_exists => true
        );
        PERFORM remove_compression_policy(
          'polymarket.ml_shadow_predictions', if_exists => true
        );
      END $$;
    `);

    await queryRunner.query(`
      DROP TABLE polymarket.ml_evaluation_metrics;
      DROP TABLE polymarket.ml_evaluation_runs;
      DROP TABLE polymarket.ml_shadow_predictions;
      DROP TABLE polymarket.ml_model_versions;
      DROP TABLE polymarket.ml_dataset_manifests;
      DROP TABLE polymarket.ml_feature_vectors;
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    throw new Error('RetireBtcMlShadow1784227000000 is intentionally irreversible');
  }
}
