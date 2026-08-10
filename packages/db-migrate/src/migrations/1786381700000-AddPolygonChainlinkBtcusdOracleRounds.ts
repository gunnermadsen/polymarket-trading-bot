import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'polygon_chainlink_btcusd_oracle';
const TABLE_NAME = 'market_data.polygon_chainlink_btcusd_oracle_rounds';
const FEED_PROXY = '0xc907e116054ad103354f2d350fd2514433d57f6f';

export class AddPolygonChainlinkBtcusdOracleRounds1786381700000
  implements MigrationInterface
{
  name = 'AddPolygonChainlinkBtcusdOracleRounds1786381700000';

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
          "rpc_url": "https://polygon-bor-rpc.publicnode.com",
          "archive_log_rpc_url": "https://tenderly.rpc.polygon.community",
          "feed_proxy_address": "${FEED_PROXY}",
          "poll_interval_seconds": 15,
          "confirmation_depth": 128,
          "maximum_block_range": 30000,
          "startup_lookback_blocks": 43200,
          "overlap_blocks": 256,
          "artifact_window_seconds": 3600,
          "request_timeout_seconds": 30
        }
        $config$::jsonb,
        'stopped',
        'stopped',
        'unknown'
      );

      CREATE TABLE ${TABLE_NAME} (
        source text NOT NULL DEFAULT 'chainlink_polygon_data_feed',
        chain_id bigint NOT NULL,
        feed_proxy_address text NOT NULL,
        aggregator_address text NOT NULL,
        phase_id integer NOT NULL,
        aggregator_round_id bigint NOT NULL,
        source_timestamp timestamptz NOT NULL,
        block_timestamp timestamptz NOT NULL,
        answer_raw numeric(38,0) NOT NULL,
        price numeric(38,18) NOT NULL,
        decimals integer NOT NULL,
        block_number bigint NOT NULL,
        block_hash text NOT NULL,
        transaction_hash text NOT NULL,
        log_index integer NOT NULL,
        provider_available_at timestamptz NOT NULL,
        received_at timestamptz NOT NULL,
        source_payload jsonb NOT NULL,
        payload_sha256 character(64) NOT NULL,
        strategy_key text NOT NULL DEFAULT '${STRATEGY_KEY}',
        capture_artifact_id uuid NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_market_data_polygon_chainlink_btcusd_oracle_rounds
          PRIMARY KEY (
            source_timestamp,
            chain_id,
            feed_proxy_address,
            transaction_hash,
            log_index
          ),
        CONSTRAINT fk_market_data_polygon_chainlink_btcusd_oracle_rounds_artifact
          FOREIGN KEY (strategy_key, capture_artifact_id)
          REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_market_data_polygon_chainlink_btcusd_oracle_identity CHECK (
          source = 'chainlink_polygon_data_feed'
          AND chain_id = 137
          AND feed_proxy_address = '${FEED_PROXY}'
          AND strategy_key = '${STRATEGY_KEY}'
          AND feed_proxy_address ~ '^0x[0-9a-f]{40}$'
          AND aggregator_address ~ '^0x[0-9a-f]{40}$'
          AND block_hash ~ '^0x[0-9a-f]{64}$'
          AND transaction_hash ~ '^0x[0-9a-f]{64}$'
        ),
        CONSTRAINT chk_market_data_polygon_chainlink_btcusd_oracle_round CHECK (
          phase_id > 0
          AND aggregator_round_id > 0
          AND block_number >= 0
          AND log_index >= 0
          AND decimals BETWEEN 0 AND 18
          AND answer_raw > 0
          AND price > 0
        ),
        CONSTRAINT chk_market_data_polygon_chainlink_btcusd_oracle_time CHECK (
          source_timestamp <= block_timestamp
          AND provider_available_at = block_timestamp
        ),
        CONSTRAINT chk_market_data_polygon_chainlink_btcusd_oracle_payload CHECK (
          jsonb_typeof(source_payload) = 'object'
          AND octet_length(source_payload::text) <= 32768
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
        timescaledb.compress_orderby =
          'source_timestamp ASC, block_number ASC, log_index ASC',
        timescaledb.compress_segmentby =
          'chain_id, feed_proxy_address, strategy_key, capture_artifact_id'
      );

      SELECT add_compression_policy(
        '${TABLE_NAME}', INTERVAL '7 days', if_not_exists => TRUE
      );

      CREATE INDEX idx_market_data_polygon_chainlink_btcusd_oracle_identity
        ON ${TABLE_NAME} (
          chain_id,
          feed_proxy_address,
          transaction_hash,
          log_index,
          source_timestamp DESC
        );

      CREATE INDEX idx_market_data_polygon_chainlink_btcusd_oracle_recovery
        ON ${TABLE_NAME} (chain_id, feed_proxy_address, block_number DESC);

      CREATE INDEX idx_market_data_polygon_chainlink_btcusd_oracle_phase_boundary
        ON ${TABLE_NAME} (
          chain_id,
          feed_proxy_address,
          phase_id,
          block_number DESC,
          log_index DESC,
          source_timestamp DESC
        ) INCLUDE (aggregator_round_id);

      CREATE INDEX idx_market_data_polygon_chainlink_btcusd_oracle_artifact
        ON ${TABLE_NAME} (capture_artifact_id, source_timestamp DESC);

      CREATE TRIGGER trg_reject_market_data_polygon_chainlink_btcusd_oracle_change
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
            'refusing to remove Polygon Chainlink BTCUSD oracle ingestion while facts, artifacts, or gaps exist';
        END IF;
      END $$;

      SELECT remove_compression_policy('${TABLE_NAME}', if_exists => TRUE);
      DROP TRIGGER trg_reject_market_data_polygon_chainlink_btcusd_oracle_change
        ON ${TABLE_NAME};
      DROP TABLE ${TABLE_NAME};
      DELETE FROM ingester.profiles WHERE strategy_key = '${STRATEGY_KEY}';
    `);
  }
}
