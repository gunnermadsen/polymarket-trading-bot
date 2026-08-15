import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'binance_futures_btcusdt_open_interest';
const TABLE_NAME = 'market_data.binance_futures_btcusdt_open_interest';

export class AddBinanceFuturesBtcusdtOpenInterest1786381400000
  implements MigrationInterface
{
  name = 'AddBinanceFuturesBtcusdtOpenInterest1786381400000';

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
          "symbol": "BTCUSDT",
          "rest_base_url": "https://fapi.binance.com",
          "period": "5m",
          "poll_interval_seconds": 60,
          "request_limit": 500,
          "startup_lookback_periods": 288,
          "overlap_periods": 2,
          "artifact_window_seconds": 86400,
          "request_timeout_seconds": 10
        }
        $config$::jsonb,
        'stopped',
        'stopped',
        'unknown'
      );

      CREATE TABLE ${TABLE_NAME} (
        source text NOT NULL DEFAULT 'binance_usd_m_futures',
        source_timestamp timestamptz NOT NULL,
        symbol text NOT NULL,
        period_seconds integer NOT NULL,
        sum_open_interest numeric(38,18) NOT NULL,
        sum_open_interest_value numeric(38,18) NOT NULL,
        cmc_circulating_supply numeric(38,18),
        provider_available_at timestamptz,
        received_at timestamptz NOT NULL,
        source_payload jsonb NOT NULL,
        payload_sha256 character(64) NOT NULL,
        strategy_key text NOT NULL DEFAULT '${STRATEGY_KEY}',
        capture_artifact_id uuid NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_market_data_binance_futures_open_interest
          PRIMARY KEY (source_timestamp, symbol, period_seconds),
        CONSTRAINT fk_market_data_binance_futures_open_interest_artifact
          FOREIGN KEY (strategy_key, capture_artifact_id)
          REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_market_data_binance_futures_open_interest_source CHECK (
          source = 'binance_usd_m_futures'
          AND symbol = 'BTCUSDT'
          AND strategy_key = '${STRATEGY_KEY}'
        ),
        CONSTRAINT chk_market_data_binance_futures_open_interest_period CHECK (
          period_seconds = 300
          AND extract(epoch FROM source_timestamp)::bigint % 300 = 0
        ),
        CONSTRAINT chk_market_data_binance_futures_open_interest_values CHECK (
          sum_open_interest >= 0
          AND sum_open_interest_value >= 0
          AND (
            cmc_circulating_supply IS NULL
            OR cmc_circulating_supply >= 0
          )
        ),
        CONSTRAINT chk_market_data_binance_futures_open_interest_payload CHECK (
          jsonb_typeof(source_payload) = 'object'
          AND octet_length(source_payload::text) <= 16384
          AND payload_sha256 ~ '^[0-9a-f]{64}$'
        )
      );

      SELECT create_hypertable(
        '${TABLE_NAME}',
        'source_timestamp',
        chunk_time_interval => INTERVAL '7 days',
        create_default_indexes => FALSE,
        if_not_exists => TRUE
      );

      ALTER TABLE ${TABLE_NAME} SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'source_timestamp ASC',
        timescaledb.compress_segmentby =
          'symbol, source, period_seconds, strategy_key, capture_artifact_id'
      );

      SELECT add_compression_policy(
        '${TABLE_NAME}', INTERVAL '7 days', if_not_exists => TRUE
      );

      CREATE INDEX idx_market_data_binance_futures_open_interest_recovery
        ON ${TABLE_NAME} (symbol, period_seconds, source_timestamp DESC);

      CREATE INDEX idx_market_data_binance_futures_open_interest_artifact
        ON ${TABLE_NAME} (capture_artifact_id, source_timestamp DESC);

      CREATE TRIGGER trg_reject_market_data_binance_futures_open_interest_change
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
            'refusing to remove Binance futures BTCUSDT open-interest ingestion while facts, artifacts, or gaps exist';
        END IF;
      END $$;

      SELECT remove_compression_policy('${TABLE_NAME}', if_exists => TRUE);
      DROP TRIGGER trg_reject_market_data_binance_futures_open_interest_change
        ON ${TABLE_NAME};
      DROP TABLE ${TABLE_NAME};
      DELETE FROM ingester.profiles WHERE strategy_key = '${STRATEGY_KEY}';
    `);
  }
}
