import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'chainlink_btcusd_one_minute_ohlc';
const TABLE_NAME = 'market_data.chainlink_btcusd_one_minute_candles';

export class AddChainlinkBtcusdOneMinuteCandles1786381600000
  implements MigrationInterface
{
  name = 'AddChainlinkBtcusdOneMinuteCandles1786381600000';

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
          "base_url": "https://priceapi.dataengine.chain.link",
          "symbol": "BTCUSD",
          "resolution": "1m",
          "poll_interval_seconds": 15,
          "startup_lookback_minutes": 1440,
          "overlap_minutes": 5,
          "request_window_minutes": 1440,
          "artifact_window_seconds": 3600,
          "request_timeout_seconds": 15
        }
        $config$::jsonb,
        'stopped',
        'stopped',
        'unknown'
      );

      CREATE TABLE ${TABLE_NAME} (
        source text NOT NULL DEFAULT 'chainlink_candlestick',
        symbol text NOT NULL,
        open_timestamp timestamptz NOT NULL,
        close_timestamp timestamptz NOT NULL,
        provider_available_at timestamptz,
        received_at timestamptz NOT NULL,
        open_price numeric(38,18) NOT NULL,
        high_price numeric(38,18) NOT NULL,
        low_price numeric(38,18) NOT NULL,
        close_price numeric(38,18) NOT NULL,
        volume numeric(38,18),
        volume_supported boolean NOT NULL DEFAULT false,
        payload_sha256 character(64) NOT NULL,
        strategy_key text NOT NULL DEFAULT '${STRATEGY_KEY}',
        capture_artifact_id uuid NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_market_data_chainlink_btcusd_one_minute_candles
          PRIMARY KEY (symbol, open_timestamp),
        CONSTRAINT fk_market_data_chainlink_btcusd_one_minute_candles_artifact
          FOREIGN KEY (strategy_key, capture_artifact_id)
          REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_market_data_chainlink_btcusd_one_minute_candles_identity CHECK (
          source = 'chainlink_candlestick'
          AND symbol = 'BTCUSD'
          AND strategy_key = '${STRATEGY_KEY}'
        ),
        CONSTRAINT chk_market_data_chainlink_btcusd_one_minute_candles_window CHECK (
          date_trunc('minute', open_timestamp) = open_timestamp
          AND close_timestamp = open_timestamp + INTERVAL '1 minute'
        ),
        CONSTRAINT chk_market_data_chainlink_btcusd_one_minute_candles_prices CHECK (
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
        CONSTRAINT chk_market_data_chainlink_btcusd_one_minute_candles_volume CHECK (
          volume IS NULL AND volume_supported = false
        ),
        CONSTRAINT chk_market_data_chainlink_btcusd_one_minute_candles_hash CHECK (
          payload_sha256 ~ '^[0-9a-f]{64}$'
        )
      );

      SELECT create_hypertable(
        '${TABLE_NAME}',
        'open_timestamp',
        chunk_time_interval => INTERVAL '7 days',
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

      CREATE INDEX idx_market_data_chainlink_btcusd_one_minute_candles_artifact
        ON ${TABLE_NAME} (capture_artifact_id, open_timestamp DESC);

      CREATE TRIGGER trg_reject_market_data_chainlink_btcusd_one_minute_candles_change
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
            'refusing to remove Chainlink BTCUSD one-minute candle ingestion while facts, artifacts, or gaps exist';
        END IF;
      END $$;

      SELECT remove_compression_policy('${TABLE_NAME}', if_exists => TRUE);
      DROP TRIGGER trg_reject_market_data_chainlink_btcusd_one_minute_candles_change
        ON ${TABLE_NAME};
      DROP TABLE ${TABLE_NAME};
      DELETE FROM ingester.profiles WHERE strategy_key = '${STRATEGY_KEY}';
    `);
  }
}
