import { MigrationInterface, QueryRunner } from 'typeorm';

export class RegisterTrainingMarketDataDrains1788818600000
  implements MigrationInterface
{
  name = 'RegisterTrainingMarketDataDrains1788818600000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      DECLARE
        definition text;
        replacement text;
      BEGIN
        SELECT pg_get_functiondef('ingester.remove_verified_drain_chunk(uuid,text)'::regprocedure)
          INTO definition;
        replacement := $cases$
          WHEN object_record.strategy_key = 'chainlink_btcusd_one_minute_candles'
            AND object_record.source_relation = 'market_data.chainlink_btcusd_one_minute_candles'
          THEN target_relation := 'market_data.chainlink_btcusd_one_minute_candles'::regclass;
            target_schema := 'market_data'; target_name := 'chainlink_btcusd_one_minute_candles';
          WHEN object_record.strategy_key = 'polymarket_btc_capacity_execution_snapshots'
            AND object_record.source_relation = 'polymarket.btc_market_capacity_execution_snapshots'
          THEN target_relation := 'polymarket.btc_market_capacity_execution_snapshots'::regclass;
            target_schema := 'polymarket'; target_name := 'btc_market_capacity_execution_snapshots';
          WHEN object_record.strategy_key = 'polymarket_btc_feature_snapshots'
            AND object_record.source_relation = 'polymarket.btc_feature_snapshots'
          THEN target_relation := 'polymarket.btc_feature_snapshots'::regclass;
            target_schema := 'polymarket'; target_name := 'btc_feature_snapshots';
          WHEN object_record.strategy_key = 'binance_spot_btcusdt_l2_snapshots'
            AND object_record.source_relation = 'market_data.binance_spot_btcusdt_l2_snapshots'
          THEN target_relation := 'market_data.binance_spot_btcusdt_l2_snapshots'::regclass;
            target_schema := 'market_data'; target_name := 'binance_spot_btcusdt_l2_snapshots';
          WHEN object_record.strategy_key = 'polygon_chainlink_btcusd_oracle_rounds'
            AND object_record.source_relation = 'market_data.polygon_chainlink_btcusd_oracle_rounds'
          THEN target_relation := 'market_data.polygon_chainlink_btcusd_oracle_rounds'::regclass;
            target_schema := 'market_data'; target_name := 'polygon_chainlink_btcusd_oracle_rounds';
          ELSE RAISE EXCEPTION 'dataset is not registered for verified drain removal';$cases$;
        definition := regexp_replace(
          definition,
          $pattern$ELSE[[:space:]]+RAISE EXCEPTION 'dataset is not registered for verified drain removal';$pattern$,
          replacement
        );
        IF definition NOT LIKE '%polymarket_btc_feature_snapshots%' THEN
          RAISE EXCEPTION 'verified drain allowlist function did not match expected definition';
        END IF;
        EXECUTE definition;
      END
      $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1 FROM ingester.drain_objects
          WHERE strategy_key IN (
            'chainlink_btcusd_one_minute_candles',
            'polymarket_btc_capacity_execution_snapshots',
            'polymarket_btc_feature_snapshots',
            'binance_spot_btcusdt_l2_snapshots',
            'polygon_chainlink_btcusd_oracle_rounds'
          ) AND status = 'removed'
        ) THEN
          RAISE EXCEPTION 'refusing rollback after training market-data chunks were removed';
        END IF;
      END
      $$;
    `);
  }
}
