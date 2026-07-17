import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetirePolymarketLegacyMarketScanner1784225000000
  implements MigrationInterface
{
  name = 'RetirePolymarketLegacyMarketScanner1784225000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TEMP TABLE retired_market_scanner_signals ON COMMIT DROP AS
      SELECT signal_id
      FROM polymarket.signal_candidates
      WHERE process_id IS NULL
        AND signal_type = 'cheap_basket';

      CREATE TEMP TABLE retired_market_scanner_orders ON COMMIT DROP AS
      SELECT order_id
      FROM polymarket.orders
      WHERE process_id IS NULL
        AND raw_payload #>> '{request,signal_id}' IN (
          SELECT signal_id::text FROM retired_market_scanner_signals
        )
        AND raw_payload #>> '{request,metadata,purpose}' = 'signal_entry';
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
          FROM polymarket.signal_candidates signal
          WHERE EXISTS (
            SELECT 1
            FROM polymarket.btc_paper_experiments experiment
            WHERE experiment.process_id = signal.process_id
          )
          OR EXISTS (
            SELECT 1
            FROM polymarket.btc_strategy_decisions decision
            WHERE decision.process_id = signal.process_id
          )
          OR EXISTS (
            SELECT 1
            FROM polymarket.btc_paper_settlement_ledger settlement
            WHERE settlement.process_id = signal.process_id
          )
        ) THEN
          RAISE EXCEPTION
            'refusing to retire signal_candidates referenced by the BTC strategy';
        END IF;

        IF EXISTS (
          SELECT 1
          FROM polymarket.orders
          WHERE order_id IN (SELECT order_id FROM retired_market_scanner_orders)
            AND process_id IS NOT NULL
        ) OR EXISTS (
          SELECT 1
          FROM polymarket.fills
          WHERE order_id IN (SELECT order_id FROM retired_market_scanner_orders)
            AND process_id IS NOT NULL
        ) THEN
          RAISE EXCEPTION
            'refusing to remove scanner-attributed execution rows owned by a trading process';
        END IF;
      END $$;
    `);

    await queryRunner.query(`
      DELETE FROM polymarket.fills
      WHERE order_id IN (SELECT order_id FROM retired_market_scanner_orders);

      DELETE FROM polymarket.orders
      WHERE order_id IN (SELECT order_id FROM retired_market_scanner_orders);
    `);

    await queryRunner.query(`
      DO $$
      BEGIN
        PERFORM remove_retention_policy(
          'polymarket.orderbook_snapshots', if_exists => true
        );
        PERFORM remove_retention_policy(
          'polymarket.signal_candidates', if_exists => true
        );
        PERFORM remove_retention_policy(
          'polymarket.funnel_events', if_exists => true
        );
        PERFORM remove_compression_policy(
          'polymarket.orderbook_snapshots', if_exists => true
        );
        PERFORM remove_compression_policy(
          'polymarket.signal_candidates', if_exists => true
        );
      END $$;
    `);

    await queryRunner.query(`
      DROP TABLE polymarket.orderbook_snapshots;
      DROP TABLE polymarket.signal_candidates;
      DROP TABLE polymarket.positions;
      DROP TABLE polymarket.funnel_events;
    `);
  }

  public async down(_queryRunner: QueryRunner): Promise<void> {
    throw new Error(
      'RetirePolymarketLegacyMarketScanner1784225000000 is intentionally irreversible',
    );
  }
}
