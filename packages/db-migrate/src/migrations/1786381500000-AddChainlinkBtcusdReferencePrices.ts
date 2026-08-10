import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'chainlink_btcusd_reference_price';
const TABLE_NAME = 'market_data.chainlink_btcusd_reference_prices';
const FEED_ID =
  '0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8';

export class AddChainlinkBtcusdReferencePrices1786381500000
  implements MigrationInterface
{
  name = 'AddChainlinkBtcusdReferencePrices1786381500000';

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
          "rest_base_url": "https://api.dataengine.chain.link",
          "feed_id": "${FEED_ID}",
          "poll_interval_ms": 1000,
          "recent_window_seconds": 300,
          "overlap_seconds": 5,
          "page_limit": 100,
          "max_pages_per_poll": 8,
          "artifact_window_seconds": 3600,
          "request_timeout_seconds": 10,
          "max_request_attempts": 3,
          "retry_initial_delay_ms": 250,
          "retry_max_delay_ms": 2000
        }
        $config$::jsonb,
        'stopped',
        'stopped',
        'unknown'
      );

      CREATE TABLE ${TABLE_NAME} (
        source text NOT NULL DEFAULT 'chainlink_data_streams',
        feed_id text NOT NULL,
        source_timestamp timestamptz NOT NULL,
        valid_from_timestamp timestamptz NOT NULL,
        provider_available_at timestamptz,
        received_at timestamptz NOT NULL,
        price numeric(38,18) NOT NULL,
        bid numeric(38,18) NOT NULL,
        ask numeric(38,18) NOT NULL,
        report_sha256 character(64) NOT NULL,
        payload_sha256 character(64) NOT NULL,
        strategy_key text NOT NULL DEFAULT '${STRATEGY_KEY}',
        capture_artifact_id uuid NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_market_data_chainlink_btcusd_reference_prices
          PRIMARY KEY (feed_id, source_timestamp, report_sha256),
        CONSTRAINT fk_market_data_chainlink_btcusd_reference_prices_artifact
          FOREIGN KEY (strategy_key, capture_artifact_id)
          REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_identity CHECK (
          source = 'chainlink_data_streams'
          AND strategy_key = '${STRATEGY_KEY}'
          AND feed_id = '${FEED_ID}'
          AND feed_id ~ '^0x[0-9a-f]{64}$'
        ),
        CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_time CHECK (
          valid_from_timestamp <= source_timestamp
        ),
        CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_values CHECK (
          price > 0
          AND bid > 0
          AND ask > 0
          AND bid <= price
          AND price <= ask
        ),
        CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_hashes CHECK (
          report_sha256 ~ '^[0-9a-f]{64}$'
          AND payload_sha256 ~ '^[0-9a-f]{64}$'
        )
      );

      SELECT create_hypertable(
        '${TABLE_NAME}',
        'source_timestamp',
        chunk_time_interval => INTERVAL '1 day',
        create_default_indexes => FALSE,
        if_not_exists => TRUE
      );

      ALTER TABLE ${TABLE_NAME} SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'source_timestamp ASC, report_sha256 ASC',
        timescaledb.compress_segmentby =
          'feed_id, source, strategy_key, capture_artifact_id'
      );

      SELECT add_compression_policy(
        '${TABLE_NAME}', INTERVAL '2 days', if_not_exists => TRUE
      );

      CREATE INDEX idx_market_data_chainlink_btcusd_reference_prices_recovery
        ON ${TABLE_NAME} (feed_id, source_timestamp DESC);

      CREATE INDEX idx_market_data_chainlink_btcusd_reference_prices_artifact
        ON ${TABLE_NAME} (capture_artifact_id, source_timestamp DESC);

      CREATE TRIGGER trg_reject_market_data_chainlink_btcusd_reference_prices_change
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
            'refusing to remove Chainlink BTCUSD reference-price ingestion while facts, artifacts, or gaps exist';
        END IF;
      END $$;

      SELECT remove_compression_policy('${TABLE_NAME}', if_exists => TRUE);
      DROP TRIGGER trg_reject_market_data_chainlink_btcusd_reference_prices_change
        ON ${TABLE_NAME};
      DROP TABLE ${TABLE_NAME};
      DELETE FROM ingester.profiles WHERE strategy_key = '${STRATEGY_KEY}';
    `);
  }
}
