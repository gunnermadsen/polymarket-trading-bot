import { MigrationInterface, QueryRunner } from 'typeorm';

export class ClearFailedDecisionExecutionSnapshots1785443500000
  implements MigrationInterface
{
  name = 'ClearFailedDecisionExecutionSnapshots1785443500000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      SET LOCAL session_replication_role = replica;

      DELETE FROM polymarket.btc_market_decision_execution_snapshots snapshot
      USING polymarket.backfill_artifacts artifact
      WHERE snapshot.artifact_id = artifact.artifact_id
        AND artifact.status = 'failed';

      SET LOCAL session_replication_role = origin;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        RAISE EXCEPTION
          'failed decision-window snapshots removed by this migration cannot be restored';
      END;
      $$;
    `);
  }
}
