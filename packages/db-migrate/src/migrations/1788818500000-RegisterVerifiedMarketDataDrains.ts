import { MigrationInterface, QueryRunner } from 'typeorm';

export class RegisterVerifiedMarketDataDrains1788818500000
  implements MigrationInterface
{
  name = 'RegisterVerifiedMarketDataDrains1788818500000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE OR REPLACE FUNCTION ingester.remove_verified_drain_chunk(
        requested_object_id uuid,
        expected_sha256 text
      ) RETURNS bigint
      LANGUAGE plpgsql
      SET search_path = pg_catalog, public, ingester, market_data, polymarket
      AS $$
      DECLARE
        object_record ingester.drain_objects%ROWTYPE;
        job_cutoff timestamptz;
        target_relation regclass;
        target_schema text;
        target_name text;
        matching_chunks integer;
        dropped_chunks integer;
      BEGIN
        SELECT object.* INTO object_record
        FROM ingester.drain_objects object
        WHERE object.object_id = requested_object_id
        FOR UPDATE;

        IF NOT FOUND
          OR object_record.status <> 'published'
          OR object_record.sha256 <> expected_sha256 THEN
          RAISE EXCEPTION 'drain object is not a verified publication';
        END IF;

        SELECT job.cutoff INTO job_cutoff
        FROM ingester.drain_jobs job
        WHERE job.job_id = object_record.job_id
          AND job.strategy_key = object_record.strategy_key
          AND job.status = 'running';
        IF NOT FOUND OR object_record.source_end > job_cutoff THEN
          RAISE EXCEPTION 'source chunk is outside the drain cutoff';
        END IF;

        CASE
          WHEN object_record.strategy_key = 'polymarket_btc_five_minute_orderbooks'
            AND object_record.source_relation = 'polymarket.btc_five_minute_orderbook_snapshots'
          THEN target_relation := 'polymarket.btc_five_minute_orderbook_snapshots'::regclass;
            target_schema := 'polymarket'; target_name := 'btc_five_minute_orderbook_snapshots';
          WHEN object_record.strategy_key = 'binance_spot_btcusdt_one_second_ohlcv'
            AND object_record.source_relation = 'market_data.binance_spot_btcusdt_one_second_ohlcv'
          THEN target_relation := 'market_data.binance_spot_btcusdt_one_second_ohlcv'::regclass;
            target_schema := 'market_data'; target_name := 'binance_spot_btcusdt_one_second_ohlcv';
          WHEN object_record.strategy_key = 'pmdata_chainlink_btcusd_reference_price'
            AND object_record.source_relation = 'market_data.pmdata_chainlink_btcusd_reference_prices'
          THEN target_relation := 'market_data.pmdata_chainlink_btcusd_reference_prices'::regclass;
            target_schema := 'market_data'; target_name := 'pmdata_chainlink_btcusd_reference_prices';
          WHEN object_record.strategy_key = 'pmdata_chainlink_btcusd_twap'
            AND object_record.source_relation = 'market_data.pmdata_chainlink_btcusd_twap'
          THEN target_relation := 'market_data.pmdata_chainlink_btcusd_twap'::regclass;
            target_schema := 'market_data'; target_name := 'pmdata_chainlink_btcusd_twap';
          WHEN object_record.strategy_key = 'polymarket_chainlink_btcusd_twap'
            AND object_record.source_relation = 'market_data.polymarket_chainlink_btcusd_twap'
          THEN target_relation := 'market_data.polymarket_chainlink_btcusd_twap'::regclass;
            target_schema := 'market_data'; target_name := 'polymarket_chainlink_btcusd_twap';
          WHEN object_record.strategy_key = 'polymarket_reference_price_ticks'
            AND object_record.source_relation = 'polymarket.reference_price_ticks'
          THEN target_relation := 'polymarket.reference_price_ticks'::regclass;
            target_schema := 'polymarket'; target_name := 'reference_price_ticks';
          ELSE RAISE EXCEPTION 'dataset is not registered for verified drain removal';
        END CASE;

        SELECT count(*) INTO matching_chunks
        FROM timescaledb_information.chunks chunk
        WHERE chunk.hypertable_schema = target_schema
          AND chunk.hypertable_name = target_name
          AND chunk.chunk_schema = object_record.source_chunk_schema
          AND chunk.chunk_name = object_record.source_chunk_name
          AND chunk.range_start = object_record.source_start
          AND chunk.range_end = object_record.source_end;
        IF matching_chunks <> 1 THEN
          RAISE EXCEPTION 'source chunk identity or bounds changed before removal';
        END IF;

        SELECT count(*) INTO dropped_chunks
        FROM drop_chunks(target_relation, older_than => object_record.source_end,
          newer_than => object_record.source_start);
        IF dropped_chunks <> 1 THEN
          RAISE EXCEPTION 'expected one removed chunk, removed %', dropped_chunks;
        END IF;

        UPDATE ingester.drain_objects
        SET status = 'removed', removed_at = clock_timestamp(), updated_at = clock_timestamp()
        WHERE object_id = requested_object_id;
        RETURN object_record.row_count;
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
            'binance_spot_btcusdt_one_second_ohlcv',
            'pmdata_chainlink_btcusd_reference_price',
            'pmdata_chainlink_btcusd_twap',
            'polymarket_chainlink_btcusd_twap',
            'polymarket_reference_price_ticks'
          ) AND status = 'removed'
        ) THEN
          RAISE EXCEPTION 'refusing rollback after registered market-data chunks were removed';
        END IF;
      END
      $$;
    `);
  }
}
