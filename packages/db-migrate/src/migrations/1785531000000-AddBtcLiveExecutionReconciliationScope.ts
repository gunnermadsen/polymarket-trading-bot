import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddBtcLiveExecutionReconciliationScope1785531000000
  implements MigrationInterface
{
  name = 'AddBtcLiveExecutionReconciliationScope1785531000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);

    await queryRunner.query(`
      DO $$
      BEGIN
        IF to_regclass('polymarket.btc_strategy_decisions') IS NULL
           OR to_regclass('polymarket.trading_processes') IS NULL
           OR to_regclass('polymarket.live_reconciliation_runs') IS NULL
           OR to_regclass('polymarket.account_reconciliation_runs') IS NULL
           OR to_regclass('polymarket.btc_paper_settlement_ledger') IS NULL THEN
          RAISE EXCEPTION
            'refusing to add BTC live reconciliation scope: required tables are missing';
        END IF;

        IF NOT EXISTS (
          SELECT 1
          FROM pg_constraint
          WHERE conrelid = 'polymarket.btc_strategy_decisions'::regclass
            AND conname = 'chk_btc_decision_mode'
            AND contype = 'c'
        ) THEN
          RAISE EXCEPTION
            'refusing to add BTC live reconciliation scope: decision mode constraint is missing';
        END IF;

        IF EXISTS (
          SELECT 1
          FROM pg_constraint
          WHERE conrelid = 'polymarket.btc_strategy_decisions'::regclass
            AND conname = 'chk_btc_decision_mode'
            AND NOT convalidated
        ) THEN
          RAISE EXCEPTION
            'refusing to expand BTC decision modes: legacy decision constraint is not validated';
        END IF;
      END $$;
    `);

    // The existing validated sim/paper constraint already proves every legacy row satisfies the
    // broader sim/paper/live predicate. Keep both constraints enforced while swapping names so the
    // 3+ million-row decision table is never scanned during this migration.
    await queryRunner.query(`
      ALTER TABLE polymarket.btc_strategy_decisions
        ADD CONSTRAINT chk_btc_decision_mode_live
        CHECK (execution_mode IN ('sim', 'paper', 'live')) NOT VALID;
      ALTER TABLE polymarket.btc_strategy_decisions
        DROP CONSTRAINT chk_btc_decision_mode;
      ALTER TABLE polymarket.btc_strategy_decisions
        RENAME CONSTRAINT chk_btc_decision_mode_live TO chk_btc_decision_mode;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.trading_processes
        ADD CONSTRAINT chk_poly_trading_processes_live_account_ref
        CHECK (
          config #>> '{execution,mode}' <> 'live'
          OR (
            config #>> '{execution,account_ref}' IS NOT NULL
            AND config #>> '{execution,account_ref}'
                = btrim(config #>> '{execution,account_ref}')
            AND config #>> '{execution,account_ref}'
                ~ '^[A-Za-z0-9._:-]{1,128}$'
          )
        ) NOT VALID;

      ALTER TABLE polymarket.trading_processes
        VALIDATE CONSTRAINT chk_poly_trading_processes_live_account_ref;

      CREATE UNIQUE INDEX idx_poly_trading_processes_active_live_account_ref
        ON polymarket.trading_processes (
          lower(btrim(config #>> '{execution,account_ref}'))
        )
        WHERE enabled
          AND status IN ('starting', 'running', 'stopping')
          AND config #>> '{execution,mode}' = 'live';
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.btc_paper_settlement_ledger
        ADD COLUMN execution_mode text NOT NULL DEFAULT 'paper',
        ADD CONSTRAINT chk_btc_settlement_execution_mode
        CHECK (execution_mode IN ('paper', 'live')) NOT VALID;

      ALTER TABLE polymarket.btc_paper_settlement_ledger
        VALIDATE CONSTRAINT chk_btc_settlement_execution_mode;

      CREATE INDEX idx_btc_settlement_process_mode_pending
        ON polymarket.btc_paper_settlement_ledger (
          process_id, run_id, execution_mode,
          official_resolution_received_at, settlement_id
        )
        WHERE credit_status = 'pending';

      CREATE INDEX idx_btc_settlement_process_mode_credited
        ON polymarket.btc_paper_settlement_ledger (
          process_id, execution_mode, credited_at DESC, order_id, settlement_id
        )
        WHERE credit_status = 'credited';
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.live_reconciliation_runs
        ADD COLUMN process_id uuid,
        ADD COLUMN account_ref text;

      ALTER TABLE polymarket.live_reconciliation_runs
        ADD CONSTRAINT fk_poly_live_reconciliation_runs_process
        FOREIGN KEY (process_id)
        REFERENCES polymarket.trading_processes (process_id)
        ON DELETE RESTRICT
        NOT VALID,
        ADD CONSTRAINT chk_poly_live_reconciliation_runs_account_ref
        CHECK (
          account_ref IS NULL
          OR length(btrim(account_ref)) BETWEEN 1 AND 128
        ) NOT VALID,
        ADD CONSTRAINT chk_poly_live_reconciliation_runs_scope_pair
        CHECK ((process_id IS NULL) = (account_ref IS NULL)) NOT VALID;

      ALTER TABLE polymarket.live_reconciliation_runs
        VALIDATE CONSTRAINT fk_poly_live_reconciliation_runs_process;
      ALTER TABLE polymarket.live_reconciliation_runs
        VALIDATE CONSTRAINT chk_poly_live_reconciliation_runs_account_ref;
      ALTER TABLE polymarket.live_reconciliation_runs
        VALIDATE CONSTRAINT chk_poly_live_reconciliation_runs_scope_pair;
    `);

    await queryRunner.query(`
      CREATE INDEX idx_poly_live_reconciliation_runs_process_started
        ON polymarket.live_reconciliation_runs (process_id, started_at DESC)
        WHERE process_id IS NOT NULL;
      CREATE INDEX idx_poly_live_reconciliation_runs_account_started
        ON polymarket.live_reconciliation_runs (account_ref, started_at DESC)
        WHERE account_ref IS NOT NULL;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.account_reconciliation_runs
        ADD COLUMN process_id uuid;

      ALTER TABLE polymarket.account_reconciliation_runs
        ADD CONSTRAINT fk_poly_account_reconciliation_runs_process
        FOREIGN KEY (process_id)
        REFERENCES polymarket.trading_processes (process_id)
        ON DELETE RESTRICT
        NOT VALID;

      ALTER TABLE polymarket.account_reconciliation_runs
        VALIDATE CONSTRAINT fk_poly_account_reconciliation_runs_process;

      CREATE INDEX idx_poly_account_reconciliation_runs_process_started
        ON polymarket.account_reconciliation_runs (process_id, started_at DESC)
        WHERE process_id IS NOT NULL;
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    // Once the runtime can persist process-scoped reconciliation and live execution evidence,
    // removing these columns or narrowing the decision-mode constraint cannot be made atomic
    // without either losing ownership evidence or scanning and exclusively locking the multi-GB
    // decision hypertable. Keep the forward migration explicit and operationally safe.
    throw new Error(
      'AddBtcLiveExecutionReconciliationScope1785531000000 is intentionally irreversible',
    );
  }
}
