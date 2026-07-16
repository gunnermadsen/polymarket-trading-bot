import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetirePolymarketWhaleCopyTrading1784224000000 implements MigrationInterface {
  name = 'RetirePolymarketWhaleCopyTrading1784224000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TEMP TABLE retired_whale_copy_processes ON COMMIT DROP AS
      SELECT process_id
      FROM polymarket.trading_processes
      WHERE process_type = 'copy_trade';

      CREATE TEMP TABLE retired_whale_copy_jobs ON COMMIT DROP AS
      SELECT job_id
      FROM polymarket.backfill_jobs
      WHERE ingester_key = 'whales'
         OR request ->> 'process_id' IN (
           SELECT process_id::text FROM retired_whale_copy_processes
         );
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
          FROM polymarket.btc_strategy_decisions
          WHERE process_id IN (SELECT process_id FROM retired_whale_copy_processes)
        ) OR EXISTS (
          SELECT 1
          FROM polymarket.btc_paper_experiments
          WHERE process_id IN (SELECT process_id FROM retired_whale_copy_processes)
        ) OR EXISTS (
          SELECT 1
          FROM polymarket.btc_paper_settlement_ledger
          WHERE process_id IN (SELECT process_id FROM retired_whale_copy_processes)
        ) THEN
          RAISE EXCEPTION
            'refusing to retire a copy-trade process referenced by the BTC strategy';
        END IF;

        IF EXISTS (
          SELECT 1
          FROM polymarket.backfill_artifacts
          WHERE job_id IN (SELECT job_id FROM retired_whale_copy_jobs)
        ) THEN
          RAISE EXCEPTION
            'refusing to remove legacy whale jobs with immutable ingestion artifacts';
        END IF;
      END $$;
    `);

    await queryRunner.query(`
      DELETE FROM polymarket.backfill_job_events
      WHERE job_id IN (SELECT job_id FROM retired_whale_copy_jobs);

      DELETE FROM polymarket.backfill_jobs
      WHERE job_id IN (SELECT job_id FROM retired_whale_copy_jobs);
    `);

    await queryRunner.query(`
      DELETE FROM polymarket.fills
      WHERE process_id IN (SELECT process_id FROM retired_whale_copy_processes);

      DELETE FROM polymarket.orders
      WHERE process_id IN (SELECT process_id FROM retired_whale_copy_processes);

      DELETE FROM polymarket.signal_candidates
      WHERE process_id IN (SELECT process_id FROM retired_whale_copy_processes);
    `);

    await queryRunner.query(`
      DROP TABLE IF EXISTS polymarket.trade_mark_source_failures;
      DROP TABLE IF EXISTS polymarket.trade_exits;
      DROP TABLE IF EXISTS polymarket.trade_marks;
      DROP TABLE IF EXISTS polymarket.trade_positions;
      DROP TABLE IF EXISTS polymarket.wallet_trade_performance;
      DROP TABLE IF EXISTS polymarket.expectancy_flow_wallet_cells;
      DROP TABLE IF EXISTS polymarket.expectancy_flow_cells;
      DROP TABLE IF EXISTS polymarket.whale_poll_checkpoints;
      DROP TABLE IF EXISTS polymarket.copy_trade_signals;
    `);

    await queryRunner.query(`
      DELETE FROM polymarket.trading_processes
      WHERE process_id IN (SELECT process_id FROM retired_whale_copy_processes);
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    throw new Error(
      'RetirePolymarketWhaleCopyTrading1784224000000 is intentionally irreversible',
    );
  }
}
