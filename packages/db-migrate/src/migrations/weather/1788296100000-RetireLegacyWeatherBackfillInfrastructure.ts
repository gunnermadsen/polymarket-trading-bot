import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetireLegacyWeatherBackfillInfrastructure1788296100000
  implements MigrationInterface
{
  name = 'RetireLegacyWeatherBackfillInfrastructure1788296100000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    const references = await queryRunner.query(`
      SELECT namespace.nspname AS table_schema,relation.relname AS table_name,
        constraint_record.conname
      FROM pg_constraint constraint_record
      JOIN pg_class relation ON relation.oid=constraint_record.conrelid
      JOIN pg_namespace namespace ON namespace.oid=relation.relnamespace
      WHERE constraint_record.contype='f'
        AND constraint_record.confrelid='weather.source_artifacts'::regclass
      ORDER BY namespace.nspname,relation.relname,constraint_record.conname
    `);
    for (const reference of references) {
      await queryRunner.query(
        `ALTER TABLE ${quoteIdentifier(reference.table_schema)}.${quoteIdentifier(reference.table_name)} ` +
          `DROP CONSTRAINT ${quoteIdentifier(reference.conname)}`,
      );
    }
    await queryRunner.query(`
      DROP TABLE weather.ingestion_jobs;
      DROP TABLE weather.source_artifacts;
    `);
  }

  public async down(): Promise<void> {
    throw new Error('retired weather queues cannot be recreated after verified consolidation');
  }
}

function quoteIdentifier(value: string): string {
  return `"${value.replace(/"/g, '""')}"`;
}
