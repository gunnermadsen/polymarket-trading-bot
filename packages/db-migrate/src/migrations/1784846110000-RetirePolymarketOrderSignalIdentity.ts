import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetirePolymarketOrderSignalIdentity1784846110000
  implements MigrationInterface
{
  name = 'RetirePolymarketOrderSignalIdentity1784846110000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET LOCAL lock_timeout = '5s';`);

    await queryRunner.query(`
      DO $$
      DECLARE
        orders_oid oid := to_regclass('polymarket.orders');
        signal_index_oid oid :=
          to_regclass('polymarket.idx_poly_orders_request_signal_id');
      BEGIN
        IF orders_oid IS NULL THEN
          RAISE EXCEPTION
            'refusing to retire order signal identity: polymarket.orders is missing';
        END IF;

        IF signal_index_oid IS NULL THEN
          RAISE EXCEPTION
            'refusing to retire order signal identity: expected request signal index is missing';
        END IF;

        IF NOT EXISTS (
          SELECT 1
          FROM pg_index index_record
          JOIN pg_class index_relation
            ON index_relation.oid = index_record.indexrelid
          JOIN pg_am index_method
            ON index_method.oid = index_relation.relam
          WHERE index_record.indexrelid = signal_index_oid
            AND index_record.indrelid = orders_oid
            AND index_relation.relkind = 'i'
            AND index_method.amname = 'btree'
            AND NOT index_record.indisunique
            AND NOT index_record.indisprimary
            AND index_record.indisvalid
            AND index_record.indisready
            AND index_record.indislive
            AND NOT index_record.indisexclusion
            AND index_record.indnkeyatts = 1
            AND index_record.indnatts = 1
            AND index_record.indkey[0] = 0
            AND index_record.indoption[0] = 0
            AND index_record.indexprs IS NOT NULL
            AND index_record.indpred IS NOT NULL
            AND regexp_replace(
              pg_get_expr(index_record.indexprs, index_record.indrelid),
              '[[:space:]()]',
              '',
              'g'
            ) = 'raw_payload#>>''{request,signal_id}''::text[]::uuid'
            AND regexp_replace(
              pg_get_expr(index_record.indpred, index_record.indrelid),
              '[[:space:]()]',
              '',
              'g'
            ) = 'raw_payload#>>''{request,signal_id}''::text[]ISNOTNULL'
        ) THEN
          RAISE EXCEPTION
            'refusing to retire order signal identity: request signal index definition has drifted';
        END IF;

        IF EXISTS (
          SELECT 1
          FROM pg_constraint
          WHERE conindid = signal_index_oid
        ) THEN
          RAISE EXCEPTION
            'refusing to retire order signal identity: request signal index backs a constraint';
        END IF;
      END $$;
    `);

    await queryRunner.query(`
      DROP INDEX polymarket.idx_poly_orders_request_signal_id;
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    throw new Error(
      'RetirePolymarketOrderSignalIdentity1784846110000 is intentionally irreversible',
    );
  }
}
