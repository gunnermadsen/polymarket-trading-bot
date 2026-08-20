import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'polymarket_chainlink_btcusd_twap';
const TABLE_NAME = 'market_data.polymarket_chainlink_btcusd_twap';

export class AddPolymarketChainlinkBtcusdTwap1787241600000
  implements MigrationInterface
{
  name = 'AddPolymarketChainlinkBtcusdTwap1787241600000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      INSERT INTO ingester.profiles (
        strategy_key, config_schema_version, config,
        desired_state, observed_state, health_status
      ) VALUES (
        '${STRATEGY_KEY}', 1,
        $config$
        {
          "websocket_url": "wss://ws-live-data.polymarket.com",
          "symbol": "btc/usd",
          "connect_timeout_ms": 10000,
          "initial_stream_timeout_ms": 20000,
          "stream_stale_timeout_ms": 30000,
          "ping_interval_ms": 5000,
          "write_timeout_ms": 5000,
          "reconnect_initial_ms": 250,
          "reconnect_max_ms": 30000,
          "artifact_window_seconds": 3600
        }
        $config$::jsonb,
        'stopped', 'stopped', 'unknown'
      );

      CREATE TABLE ${TABLE_NAME} (
        source_timestamp timestamptz NOT NULL,
        published_at timestamptz NOT NULL,
        received_at timestamptz NOT NULL,
        source text NOT NULL DEFAULT 'polymarket_rtds_chainlink_twap',
        symbol text NOT NULL,
        window_seconds smallint NOT NULL,
        twap_price numeric(38,18) NOT NULL,
        full_accuracy_value text NOT NULL,
        source_payload jsonb NOT NULL,
        payload_sha256 character(64) NOT NULL,
        strategy_key text NOT NULL DEFAULT '${STRATEGY_KEY}',
        capture_artifact_id uuid NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_market_data_polymarket_chainlink_btcusd_twap
          PRIMARY KEY (source_timestamp, symbol, window_seconds),
        CONSTRAINT fk_market_data_polymarket_chainlink_btcusd_twap_artifact
          FOREIGN KEY (strategy_key, capture_artifact_id)
          REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_market_data_polymarket_chainlink_btcusd_twap_identity CHECK (
          source = 'polymarket_rtds_chainlink_twap'
          AND strategy_key = '${STRATEGY_KEY}'
          AND symbol = 'btc/usd'
          AND window_seconds IN (30, 60)
        ),
        CONSTRAINT chk_market_data_polymarket_chainlink_btcusd_twap_value CHECK (
          full_accuracy_value ~ '^-?[0-9]{1,29}$'
          AND twap_price * 1000000000000000000::numeric = full_accuracy_value::numeric
          AND twap_price > 0
        ),
        CONSTRAINT chk_market_data_polymarket_chainlink_btcusd_twap_time CHECK (
          source_timestamp <= published_at
          AND published_at <= received_at + INTERVAL '5 seconds'
        ),
        CONSTRAINT chk_market_data_polymarket_chainlink_btcusd_twap_payload CHECK (
          jsonb_typeof(source_payload) = 'object'
          AND octet_length(source_payload::text) <= 4096
          AND payload_sha256 ~ '^[0-9a-f]{64}$'
        )
      );

      SELECT create_hypertable(
        '${TABLE_NAME}', 'source_timestamp',
        chunk_time_interval => INTERVAL '1 day',
        create_default_indexes => FALSE, if_not_exists => TRUE
      );

      ALTER TABLE ${TABLE_NAME} SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'source_timestamp ASC, published_at ASC',
        timescaledb.compress_segmentby =
          'symbol, window_seconds, strategy_key, capture_artifact_id'
      );
      SELECT add_compression_policy(
        '${TABLE_NAME}', INTERVAL '1 day', if_not_exists => TRUE
      );

      CREATE INDEX idx_market_data_polymarket_chainlink_btcusd_twap_latest
        ON ${TABLE_NAME} (symbol, window_seconds, source_timestamp DESC);
      CREATE INDEX idx_market_data_polymarket_chainlink_btcusd_twap_artifact
        ON ${TABLE_NAME} (capture_artifact_id, source_timestamp DESC);

      CREATE TRIGGER trg_reject_md_polymarket_chainlink_btcusd_twap_change
        BEFORE UPDATE OR DELETE ON ${TABLE_NAME}
        FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();
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
            'refusing to remove Polymarket Chainlink BTC/USD TWAP ingestion while facts, artifacts, or gaps exist';
        END IF;
      END $$;
      SELECT remove_compression_policy('${TABLE_NAME}', if_exists => TRUE);
      DROP TRIGGER trg_reject_md_polymarket_chainlink_btcusd_twap_change
        ON ${TABLE_NAME};
      DROP TABLE ${TABLE_NAME};
      DELETE FROM ingester.profiles WHERE strategy_key = '${STRATEGY_KEY}';
    `);
  }
}
