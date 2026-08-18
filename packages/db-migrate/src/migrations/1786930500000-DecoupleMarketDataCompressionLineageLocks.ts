import { MigrationInterface, QueryRunner } from 'typeorm';

const FACT_TABLES = [
  {
    table: 'market_data.binance_spot_btcusdt_aggregate_trades',
    constraint: 'fk_market_data_binance_spot_aggregate_trade_artifact',
    trigger: 'trg_validate_binance_spot_aggregate_trade_artifact',
  },
  {
    table: 'market_data.binance_spot_btcusdt_one_second_ohlcv',
    constraint: 'fk_market_data_binance_spot_one_second_ohlcv_artifact',
    trigger: 'trg_validate_binance_spot_one_second_ohlcv_artifact',
  },
  {
    table: 'market_data.binance_spot_btcusdt_l2_snapshots',
    constraint: 'fk_market_data_binance_spot_l2_snapshot_artifact',
    trigger: 'trg_validate_binance_spot_l2_snapshot_artifact',
  },
  {
    table: 'market_data.binance_futures_btcusdt_open_interest',
    constraint: 'fk_market_data_binance_futures_open_interest_artifact',
    trigger: 'trg_validate_binance_futures_open_interest_artifact',
  },
  {
    table: 'market_data.chainlink_btcusd_reference_prices',
    constraint: 'fk_market_data_chainlink_btcusd_reference_prices_artifact',
    trigger: 'trg_validate_chainlink_btcusd_reference_prices_artifact',
  },
  {
    table: 'market_data.chainlink_btcusd_one_minute_candles',
    constraint: 'fk_market_data_chainlink_btcusd_one_minute_candles_artifact',
    trigger: 'trg_validate_chainlink_btcusd_one_minute_candles_artifact',
  },
  {
    table: 'market_data.polygon_chainlink_btcusd_oracle_rounds',
    constraint: 'fk_market_data_polygon_chainlink_btcusd_oracle_rounds_artifact',
    trigger: 'trg_validate_polygon_chainlink_btcusd_oracle_rounds_artifact',
  },
  {
    table: 'market_data.polymarket_btc_five_minute_orderbook_snapshots',
    constraint: 'fk_market_data_polymarket_btc_five_minute_orderbook_artifact',
    trigger: 'trg_validate_polymarket_btc_five_minute_orderbook_artifact',
  },
] as const;

export class DecoupleMarketDataCompressionLineageLocks1786930500000
  implements MigrationInterface
{
  name = 'DecoupleMarketDataCompressionLineageLocks1786930500000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE FUNCTION ingester.validate_capture_artifact_reference()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        IF NOT EXISTS (
          SELECT 1
          FROM ingester.capture_artifacts artifact
          WHERE artifact.strategy_key = NEW.strategy_key
            AND artifact.artifact_id = NEW.capture_artifact_id
        ) THEN
          RAISE EXCEPTION
            'capture artifact % does not belong to strategy %',
            NEW.capture_artifact_id,
            NEW.strategy_key
            USING ERRCODE = 'foreign_key_violation';
        END IF;
        RETURN NEW;
      END;
      $$;

      CREATE INDEX idx_ingester_data_gap_recent_unrecoverable
        ON ingester.data_gaps (strategy_key, detected_at DESC)
        WHERE status = 'unrecoverable';
    `);

    for (const fact of FACT_TABLES) {
      await queryRunner.query(`
        ALTER TABLE ${fact.table}
          DROP CONSTRAINT ${fact.constraint};

        CREATE TRIGGER ${fact.trigger}
          BEFORE INSERT ON ${fact.table}
          FOR EACH ROW
          EXECUTE FUNCTION ingester.validate_capture_artifact_reference();
      `);
    }
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    for (const fact of [...FACT_TABLES].reverse()) {
      await queryRunner.query(`
        DROP TRIGGER ${fact.trigger} ON ${fact.table};

        ALTER TABLE ${fact.table}
          ADD CONSTRAINT ${fact.constraint}
          FOREIGN KEY (strategy_key, capture_artifact_id)
          REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT;
      `);
    }

    await queryRunner.query(`
      DROP INDEX ingester.idx_ingester_data_gap_recent_unrecoverable;
      DROP FUNCTION ingester.validate_capture_artifact_reference();
    `);
  }
}
