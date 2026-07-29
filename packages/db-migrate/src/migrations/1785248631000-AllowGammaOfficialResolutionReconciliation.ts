import { MigrationInterface, QueryRunner } from 'typeorm';

export class AllowGammaOfficialResolutionReconciliation1785248631000
  implements MigrationInterface
{
  name = 'AllowGammaOfficialResolutionReconciliation1785248631000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE polymarket.btc_interval_markets
        ADD CONSTRAINT chk_btc_official_resolution_provenance_gamma
        CHECK (
          official_resolution_source IS NULL
          OR official_resolution_source IN (
            'clob_websocket',
            'clob_rest_reconciliation',
            'clob_websocket_legacy',
            'gamma_rest_reconciliation'
          )
        ) NOT VALID;

      ALTER TABLE polymarket.btc_interval_markets
        VALIDATE CONSTRAINT chk_btc_official_resolution_provenance_gamma;

      ALTER TABLE polymarket.btc_interval_markets
        DROP CONSTRAINT chk_btc_official_resolution_provenance;

      ALTER TABLE polymarket.btc_interval_markets
        RENAME CONSTRAINT chk_btc_official_resolution_provenance_gamma
        TO chk_btc_official_resolution_provenance;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.btc_official_resolution_watches
        ADD CONSTRAINT chk_btc_resolution_watch_source_gamma
        CHECK (
          resolution_source IS NULL
          OR resolution_source IN (
            'clob_websocket',
            'clob_rest_reconciliation',
            'gamma_rest_reconciliation'
          )
        ) NOT VALID;

      ALTER TABLE polymarket.btc_official_resolution_watches
        VALIDATE CONSTRAINT chk_btc_resolution_watch_source_gamma;

      ALTER TABLE polymarket.btc_official_resolution_watches
        DROP CONSTRAINT chk_btc_resolution_watch_source;

      ALTER TABLE polymarket.btc_official_resolution_watches
        RENAME CONSTRAINT chk_btc_resolution_watch_source_gamma
        TO chk_btc_resolution_watch_source;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.btc_paper_settlement_ledger
        ADD CONSTRAINT chk_btc_paper_settlement_source_gamma
        CHECK (
          official_resolution_source IN (
            'clob_websocket',
            'clob_rest_reconciliation',
            'gamma_rest_reconciliation'
          )
        ) NOT VALID;

      ALTER TABLE polymarket.btc_paper_settlement_ledger
        VALIDATE CONSTRAINT chk_btc_paper_settlement_source_gamma;

      ALTER TABLE polymarket.btc_paper_settlement_ledger
        DROP CONSTRAINT chk_btc_paper_settlement_source;

      ALTER TABLE polymarket.btc_paper_settlement_ledger
        RENAME CONSTRAINT chk_btc_paper_settlement_source_gamma
        TO chk_btc_paper_settlement_source;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
          FROM polymarket.btc_interval_markets
          WHERE official_resolution_source = 'gamma_rest_reconciliation'
        ) OR EXISTS (
          SELECT 1
          FROM polymarket.btc_official_resolution_watches
          WHERE resolution_source = 'gamma_rest_reconciliation'
        ) OR EXISTS (
          SELECT 1
          FROM polymarket.btc_paper_settlement_ledger
          WHERE official_resolution_source = 'gamma_rest_reconciliation'
        ) THEN
          RAISE EXCEPTION
            'refusing to remove Gamma official-resolution provenance while Gamma-backed rows exist';
        END IF;
      END $$;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.btc_interval_markets
        ADD CONSTRAINT chk_btc_official_resolution_provenance_without_gamma
        CHECK (
          official_resolution_source IS NULL
          OR official_resolution_source IN (
            'clob_websocket',
            'clob_rest_reconciliation',
            'clob_websocket_legacy'
          )
        ) NOT VALID;

      ALTER TABLE polymarket.btc_interval_markets
        VALIDATE CONSTRAINT chk_btc_official_resolution_provenance_without_gamma;

      ALTER TABLE polymarket.btc_interval_markets
        DROP CONSTRAINT chk_btc_official_resolution_provenance;

      ALTER TABLE polymarket.btc_interval_markets
        RENAME CONSTRAINT chk_btc_official_resolution_provenance_without_gamma
        TO chk_btc_official_resolution_provenance;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.btc_official_resolution_watches
        ADD CONSTRAINT chk_btc_resolution_watch_source_without_gamma
        CHECK (
          resolution_source IS NULL
          OR resolution_source IN (
            'clob_websocket',
            'clob_rest_reconciliation'
          )
        ) NOT VALID;

      ALTER TABLE polymarket.btc_official_resolution_watches
        VALIDATE CONSTRAINT chk_btc_resolution_watch_source_without_gamma;

      ALTER TABLE polymarket.btc_official_resolution_watches
        DROP CONSTRAINT chk_btc_resolution_watch_source;

      ALTER TABLE polymarket.btc_official_resolution_watches
        RENAME CONSTRAINT chk_btc_resolution_watch_source_without_gamma
        TO chk_btc_resolution_watch_source;
    `);

    await queryRunner.query(`
      ALTER TABLE polymarket.btc_paper_settlement_ledger
        ADD CONSTRAINT chk_btc_paper_settlement_source_without_gamma
        CHECK (
          official_resolution_source IN (
            'clob_websocket',
            'clob_rest_reconciliation'
          )
        ) NOT VALID;

      ALTER TABLE polymarket.btc_paper_settlement_ledger
        VALIDATE CONSTRAINT chk_btc_paper_settlement_source_without_gamma;

      ALTER TABLE polymarket.btc_paper_settlement_ledger
        DROP CONSTRAINT chk_btc_paper_settlement_source;

      ALTER TABLE polymarket.btc_paper_settlement_ledger
        RENAME CONSTRAINT chk_btc_paper_settlement_source_without_gamma
        TO chk_btc_paper_settlement_source;
    `);
  }
}
