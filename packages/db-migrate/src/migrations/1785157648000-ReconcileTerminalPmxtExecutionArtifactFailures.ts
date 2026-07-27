import { MigrationInterface, QueryRunner } from 'typeorm';

export class ReconcileTerminalPmxtExecutionArtifactFailures1785157648000
  implements MigrationInterface
{
  name = 'ReconcileTerminalPmxtExecutionArtifactFailures1785157648000';

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
      WHERE artifact.job_id = job.job_id
        AND artifact.ingester_key = 'polymarket_btc_five_minute_execution_snapshots'
        AND artifact.status IN (
          'pending', 'downloading', 'downloaded', 'verified', 'ingesting'
        )
        AND job.status IN ('failed', 'cancelled');
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    // The previous in-flight status is unknowable after its parent job is terminal.
  }
}
