import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'binance_spot_btcusdt_aggregate_trades';
const TABLE_NAME = 'market_data.binance_spot_btcusdt_aggregate_trades';

export class AddBinanceSpotBtcusdtAggregateTrades1786381100000
  implements MigrationInterface
{
  name = 'AddBinanceSpotBtcusdtAggregateTrades1786381100000';

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
          "websocket_url": "wss://stream.binance.com:9443/ws/btcusdt@aggTrade",
          "rest_base_url": "https://data-api.binance.vision",
          "batch_size": 500,
          "flush_interval_ms": 250,
          "rest_page_limit": 1000,
          "recovery_overlap_records": 100,
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
        aggregate_trade_id bigint NOT NULL,
        trade_timestamp timestamptz NOT NULL,
        provider_available_at timestamptz,
        received_at timestamptz NOT NULL,
        price numeric(30,10) NOT NULL,
        quantity numeric(30,10) NOT NULL,
        first_trade_id bigint NOT NULL,
        last_trade_id bigint NOT NULL,
        buyer_maker boolean NOT NULL,
        best_match boolean NOT NULL,
        payload_sha256 character(64) NOT NULL,
        strategy_key text NOT NULL DEFAULT '${STRATEGY_KEY}',
        capture_artifact_id uuid NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_market_data_binance_spot_aggregate_trades
          PRIMARY KEY (symbol, trade_timestamp, aggregate_trade_id),
        CONSTRAINT fk_market_data_binance_spot_aggregate_trade_artifact
          FOREIGN KEY (strategy_key, capture_artifact_id)
          REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_market_data_binance_spot_aggregate_trade_source CHECK (
          source = 'binance_spot'
          AND symbol = 'BTCUSDT'
          AND strategy_key = '${STRATEGY_KEY}'
        ),
        CONSTRAINT chk_market_data_binance_spot_aggregate_trade_values CHECK (
          price > 0 AND quantity > 0
        ),
        CONSTRAINT chk_market_data_binance_spot_aggregate_trade_ids CHECK (
          aggregate_trade_id >= 0
          AND first_trade_id >= 0
          AND last_trade_id >= first_trade_id
        ),
        CONSTRAINT chk_market_data_binance_spot_aggregate_trade_payload CHECK (
          payload_sha256 ~ '^[0-9a-f]{64}$'
        )
      );

      SELECT create_hypertable(
        '${TABLE_NAME}',
        'trade_timestamp',
        chunk_time_interval => INTERVAL '1 day',
        create_default_indexes => FALSE,
        if_not_exists => TRUE
      );

      ALTER TABLE ${TABLE_NAME} SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby =
          'trade_timestamp ASC, aggregate_trade_id ASC',
        timescaledb.compress_segmentby =
          'symbol, source, strategy_key, capture_artifact_id'
      );

      SELECT add_compression_policy(
        '${TABLE_NAME}', INTERVAL '2 days', if_not_exists => TRUE
      );

      CREATE INDEX idx_market_data_binance_spot_aggregate_trade_recovery
        ON ${TABLE_NAME} (
          symbol, aggregate_trade_id DESC, trade_timestamp DESC
        );

      CREATE INDEX idx_market_data_binance_spot_aggregate_trade_artifact
        ON ${TABLE_NAME} (capture_artifact_id, trade_timestamp DESC);

      CREATE TRIGGER trg_reject_market_data_binance_spot_aggregate_trade_change
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
            'refusing to remove Binance spot BTCUSDT aggregate-trade ingestion while facts, artifacts, or gaps exist';
        END IF;
      END $$;

      SELECT remove_compression_policy('${TABLE_NAME}', if_exists => TRUE);
      DROP TRIGGER trg_reject_market_data_binance_spot_aggregate_trade_change
        ON ${TABLE_NAME};
      DROP TABLE ${TABLE_NAME};
      DELETE FROM ingester.profiles WHERE strategy_key = '${STRATEGY_KEY}';
    `);
  }
}
