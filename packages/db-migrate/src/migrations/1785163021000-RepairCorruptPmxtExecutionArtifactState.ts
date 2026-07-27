import { MigrationInterface, QueryRunner } from 'typeorm';

export class RepairCorruptPmxtExecutionArtifactState1785163021000
  implements MigrationInterface
{
  name = 'RepairCorruptPmxtExecutionArtifactState1785163021000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      UPDATE polymarket.backfill_artifacts AS artifact
      SET status = 'failed',
          metadata = artifact.metadata || jsonb_build_object(
            'error',
            COALESCE(
              NULLIF(artifact.metadata ->> 'error', ''),
              NULLIF(job.error, ''),
              'parent backfill job ended before artifact completion'
            ),
            'failure_reconciled_from_parent_job',
            true
          ),
          updated_at = now()
      FROM polymarket.backfill_jobs AS job
      WHERE artifact.artifact_id =
              '2af9b0db-e152-4963-8177-e08a678b79dd'::uuid
        AND artifact.job_id = job.job_id
        AND artifact.ingester_key =
              'polymarket_btc_five_minute_execution_snapshots'
        AND artifact.status IN (
          'pending', 'downloading', 'downloaded', 'verified', 'ingesting'
        )
        AND job.status = 'failed';
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    // The previous in-flight status is unknowable after its parent job is terminal.
  }
}
