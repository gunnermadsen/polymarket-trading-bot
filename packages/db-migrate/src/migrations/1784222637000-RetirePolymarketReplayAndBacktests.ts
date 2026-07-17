import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetirePolymarketReplayAndBacktests1784222637000 implements MigrationInterface {
  name = 'RetirePolymarketReplayAndBacktests1784222637000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TEMP TABLE retired_replay_processes AS
      SELECT process_id
      FROM polymarket.trading_processes
      WHERE process_scope = 'backtest'
         OR metadata ->> 'backtest' = 'true';

      CREATE TEMP TABLE retired_replay_jobs AS
      SELECT job_id
      FROM polymarket.backfill_jobs
      WHERE request ->> 'mode' = 'copy_trade_backtest'
         OR request ->> 'process_id' IN (
           SELECT process_id::text FROM retired_replay_processes
         );
    `);

    await queryRunner.query(`
      DELETE FROM polymarket.backfill_artifacts
      WHERE job_id IN (SELECT job_id FROM retired_replay_jobs);

      DELETE FROM polymarket.backfill_job_events
      WHERE job_id IN (SELECT job_id FROM retired_replay_jobs);

      DELETE FROM polymarket.backfill_jobs
      WHERE job_id IN (SELECT job_id FROM retired_replay_jobs);
    `);

    const retiredReplayProcesses: Array<{ process_id: string }> = await queryRunner.query(`
      SELECT process_id::text AS process_id
      FROM retired_replay_processes;
    `);

    for (const { process_id: replayProcessId } of retiredReplayProcesses) {
      await queryRunner.query(
        `DELETE FROM polymarket.fills WHERE process_id = $1;`,
        [replayProcessId],
      );
      await queryRunner.query(
        `DELETE FROM polymarket.orders WHERE process_id = $1;`,
        [replayProcessId],
      );
      await queryRunner.query(
        `DELETE FROM polymarket.signal_candidates WHERE process_id = $1;`,
        [replayProcessId],
      );
      await queryRunner.query(
        `DELETE FROM polymarket.copy_trade_signals WHERE process_id = $1;`,
        [replayProcessId],
      );
      await queryRunner.query(
        `DELETE FROM polymarket.trade_positions WHERE process_id = $1;`,
        [replayProcessId],
      );
      await queryRunner.query(
        `DELETE FROM polymarket.wallet_trade_performance WHERE process_id = $1;`,
        [replayProcessId],
      );
      await queryRunner.query(
        `DELETE FROM polymarket.expectancy_flow_wallet_cells WHERE process_id = $1;`,
        [replayProcessId],
      );
      await queryRunner.query(
        `DELETE FROM polymarket.expectancy_flow_cells WHERE process_id = $1;`,
        [replayProcessId],
      );
      await queryRunner.query(
        `DELETE FROM polymarket.trading_processes WHERE process_id = $1;`,
        [replayProcessId],
      );
    }

    await queryRunner.query(`
      UPDATE polymarket.trading_processes
      SET config = config #- '{copy_trade,backtest_horizon_secs}',
          metadata = metadata
            - 'backtest'
            - 'backtest_run_id'
            - 'backtest_process_id'
            - 'last_backtest_run_id',
          updated_at = now()
      WHERE config #> '{copy_trade,backtest_horizon_secs}' IS NOT NULL
         OR metadata ?| ARRAY[
           'backtest',
           'backtest_run_id',
           'backtest_process_id',
           'last_backtest_run_id'
         ];
    `);

    await queryRunner.query(`
      DROP INDEX IF EXISTS polymarket.idx_poly_wallet_trades_replay_range;
      DROP TABLE IF EXISTS polymarket.copy_trade_backtest_results;
      DROP TABLE IF EXISTS polymarket.copy_trade_backtest_runs;
      DROP TABLE IF EXISTS polymarket.copy_trade_backtests;
      DROP TABLE IF EXISTS polymarket.wallet_score_calibration_snapshots;
      DROP TABLE IF EXISTS polymarket.backtest_runs;
      DROP TABLE retired_replay_jobs;
      DROP TABLE retired_replay_processes;
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    throw new Error(
      'RetirePolymarketReplayAndBacktests1784222637000 is intentionally irreversible',
    );
  }
}
