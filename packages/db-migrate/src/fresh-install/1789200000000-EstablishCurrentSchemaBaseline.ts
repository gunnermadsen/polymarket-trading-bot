import { readFileSync } from 'fs';
import { join } from 'path';
import { MigrationInterface, QueryRunner } from 'typeorm';

export class EstablishCurrentSchemaBaseline1789200000000
  implements MigrationInterface
{
  name = 'EstablishCurrentSchemaBaseline1789200000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    const [{ relation_count: relationCount }] = (await queryRunner.query(`
      SELECT count(*)::text AS relation_count
      FROM pg_class relation
      JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
      WHERE relation.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')
        AND namespace.nspname NOT LIKE 'pg_%'
        AND namespace.nspname NOT IN (
          'information_schema',
          '_timescaledb_cache',
          '_timescaledb_catalog',
          '_timescaledb_config',
          '_timescaledb_functions',
          '_timescaledb_internal',
          'timescaledb_experimental',
          'timescaledb_information'
        )
        AND NOT (
          namespace.nspname = 'public'
          AND relation.relname IN ('migrations', 'migrations_id_seq')
        )
    `)) as Array<{ relation_count: string }>;

    if (relationCount !== '0') {
      throw new Error(
        'Refusing to establish the current schema baseline because the database already contains application relations.',
      );
    }

    const schemaSql = readFileSync(
      join(__dirname, 'current-schema.sql'),
      'utf8',
    );
    await queryRunner.query(schemaSql);
  }

  public async down(): Promise<void> {
    throw new Error('The fresh-install schema baseline is irreversible.');
  }
}
