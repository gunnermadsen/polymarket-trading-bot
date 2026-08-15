import { MigrationInterface, QueryRunner } from 'typeorm';

export class AllowSharedLiveExecutionAccounts1786382100000
  implements MigrationInterface
{
  name = 'AllowSharedLiveExecutionAccounts1786382100000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);
    await queryRunner.query(`
      DROP INDEX IF EXISTS polymarket.idx_poly_trading_processes_active_live_account_ref;

      CREATE INDEX idx_poly_trading_processes_active_live_account_ref
        ON polymarket.trading_processes (
          lower(btrim(config #>> '{execution,account_ref}')),
          process_id
        )
        WHERE enabled
          AND status IN ('starting', 'running', 'stopping')
          AND config #>> '{execution,mode}' = 'live';
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);
    await queryRunner.query(`
      DROP INDEX IF EXISTS polymarket.idx_poly_trading_processes_active_live_account_ref;

      CREATE UNIQUE INDEX idx_poly_trading_processes_active_live_account_ref
        ON polymarket.trading_processes (
          lower(btrim(config #>> '{execution,account_ref}'))
        )
        WHERE enabled
          AND status IN ('starting', 'running', 'stopping')
          AND config #>> '{execution,mode}' = 'live';
    `);
  }
}
