import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'binance_spot_btcusdt_one_second_ohlcv';
const TABLE_NAME = 'market_data.binance_spot_btcusdt_one_second_ohlcv';

export class AddBinanceSpotBtcusdtOneSecondOhlcv1786381200000
  implements MigrationInterface
{
  name = 'AddBinanceSpotBtcusdtOneSecondOhlcv1786381200000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      INSERT INTO ingester.profiles (
        strategy_key, config_schema_version, config,
        desired_state, observed_state, health_status
      ) VALUES (
        '${STRATEGY_KEY}',
        1,
        $config$
        {
          "websocket_url": "wss://stream.binance.com:9443/ws/btcusdt@kline_1s",
          "rest_base_url": "https://data-api.binance.vision",
          "batch_size": 250,
          "flush_interval_ms": 250,
          "rest_page_limit": 1000,
          "recovery_overlap_seconds": 60,
          "read_idle_timeout_ms": 40000,
          "reconnect_initial_delay_ms": 1000,
          "reconnect_max_delay_ms": 30000,
          "artifact_window_seconds": 3600
        }
        $config$::jsonb,
        'stopped',
        'stopped',
        'unknown'
      );

      CREATE TABLE ${TABLE_NAME} (
        source text NOT NULL DEFAULT 'binance_spot',
        symbol text NOT NULL,
        open_timestamp timestamptz NOT NULL,
        close_timestamp timestamptz NOT NULL,
        provider_available_at timestamptz,
        received_at timestamptz NOT NULL,
        open_price numeric(30,10) NOT NULL,
        high_price numeric(30,10) NOT NULL,
        low_price numeric(30,10) NOT NULL,
        close_price numeric(30,10) NOT NULL,
        base_volume numeric(30,10) NOT NULL,
        quote_volume numeric(30,10) NOT NULL,
        trade_count bigint NOT NULL,
        taker_buy_base_volume numeric(30,10) NOT NULL,
        taker_buy_quote_volume numeric(30,10) NOT NULL,
        payload_sha256 character(64) NOT NULL,
        strategy_key text NOT NULL DEFAULT '${STRATEGY_KEY}',
        capture_artifact_id uuid NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_market_data_binance_spot_one_second_ohlcv
          PRIMARY KEY (symbol, open_timestamp),
        CONSTRAINT fk_market_data_binance_spot_one_second_ohlcv_artifact
          FOREIGN KEY (strategy_key, capture_artifact_id)
          REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_market_data_binance_spot_one_second_ohlcv_source CHECK (
          source = 'binance_spot'
          AND symbol = 'BTCUSDT'
          AND strategy_key = '${STRATEGY_KEY}'
        ),
        CONSTRAINT chk_market_data_binance_spot_one_second_ohlcv_window CHECK (
          date_trunc('second', open_timestamp) = open_timestamp
          AND close_timestamp = open_timestamp + INTERVAL '999 milliseconds'
        ),
        CONSTRAINT chk_market_data_binance_spot_one_second_ohlcv_prices CHECK (
          open_price > 0
          AND high_price > 0
          AND low_price > 0
          AND close_price > 0
          AND high_price >= open_price
          AND high_price >= close_price
          AND high_price >= low_price
          AND low_price <= open_price
          AND low_price <= close_price
        ),
        CONSTRAINT chk_market_data_binance_spot_one_second_ohlcv_volume CHECK (
          base_volume >= 0
          AND quote_volume >= 0
          AND trade_count >= 0
          AND taker_buy_base_volume >= 0
          AND taker_buy_quote_volume >= 0
          AND taker_buy_base_volume <= base_volume
          AND taker_buy_quote_volume <= quote_volume
        ),
        CONSTRAINT chk_market_data_binance_spot_one_second_ohlcv_payload CHECK (
          payload_sha256 ~ '^[0-9a-f]{64}$'
        )
      );

      SELECT create_hypertable(
        '${TABLE_NAME}',
        'open_timestamp',
        chunk_time_interval => INTERVAL '1 day',
        create_default_indexes => FALSE,
        if_not_exists => TRUE
      );

      ALTER TABLE ${TABLE_NAME} SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'open_timestamp ASC',
        timescaledb.compress_segmentby =
          'symbol, source, strategy_key, capture_artifact_id'
      );

      SELECT add_compression_policy(
        '${TABLE_NAME}', INTERVAL '7 days', if_not_exists => TRUE
      );

      CREATE INDEX idx_market_data_binance_spot_one_second_ohlcv_artifact
        ON ${TABLE_NAME} (capture_artifact_id, open_timestamp DESC);

      CREATE TRIGGER trg_reject_market_data_binance_spot_one_second_ohlcv_change
        BEFORE UPDATE OR DELETE ON ${TABLE_NAME}
        FOR EACH ROW
        EXECUTE FUNCTION market_data.reject_source_fact_change();
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (SELECT 1 FROM ${TABLE_NAME} LIMIT 1)
          OR EXISTS (
            SELECT 1 FROM ingester.capture_artifacts
            WHERE strategy_key = '${STRATEGY_KEY}' LIMIT 1
          )
          OR EXISTS (
            SELECT 1 FROM ingester.data_gaps
            WHERE strategy_key = '${STRATEGY_KEY}' LIMIT 1
          ) THEN
          RAISE EXCEPTION
            'refusing to remove Binance spot BTCUSDT one-second OHLCV ingestion while facts, artifacts, or gaps exist';
        END IF;
      END $$;

      SELECT remove_compression_policy('${TABLE_NAME}', if_exists => TRUE);
      DROP TRIGGER trg_reject_market_data_binance_spot_one_second_ohlcv_change
        ON ${TABLE_NAME};
      DROP TABLE ${TABLE_NAME};
      DELETE FROM ingester.profiles WHERE strategy_key = '${STRATEGY_KEY}';
    `);
  }
}
