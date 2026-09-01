import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetireLegacyBackfillInfrastructure1788296000000
  implements MigrationInterface
{
  name = 'RetireLegacyBackfillInfrastructure1788296000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DROP TABLE financial_data.worker_status;
      DROP TABLE financial_data.backfill_job_events;
      DROP TABLE financial_data.parquet_objects;
      DROP TABLE financial_data.source_artifacts;
      DROP TABLE financial_data.backfill_jobs;

      DROP TABLE kraken.worker_status;
      DROP TABLE kraken.backfill_job_events;
      DROP TABLE kraken.parquet_objects;
      ALTER TABLE kraken.backfill_artifacts
        DROP CONSTRAINT backfill_artifacts_job_id_fkey;
      DROP TABLE kraken.backfill_jobs;
    `);
  }

  public async down(): Promise<void> {
    throw new Error('retired archive queues cannot be recreated after verified consolidation');
  }
}
