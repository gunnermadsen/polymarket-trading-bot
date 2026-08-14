import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'polymarket_btc_five_minute_resolutions';
const TABLE_NAME = 'market_data.polymarket_btc_five_minute_resolutions';

export class AddPolymarketBtcFiveMinuteResolutions1786382000000
  implements MigrationInterface
{
  name = 'AddPolymarketBtcFiveMinuteResolutions1786382000000';

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
          "clob_base_url": "https://clob.polymarket.com",
          "clob_market_endpoint": "markets",
          "gamma_base_url": "https://gamma-api.polymarket.com",
          "websocket_url": "wss://ws-subscriptions-clob.polymarket.com/ws/market",
          "poll_interval_seconds": 5,
          "discovery_refresh_seconds": 5,
          "startup_lookback_windows": 288,
          "lookahead_windows": 1,
          "gamma_fallback_grace_seconds": 120,
          "retry_initial_seconds": 30,
          "retry_max_seconds": 300,
          "connect_timeout_ms": 10000,
          "read_timeout_ms": 40000,
          "ping_interval_ms": 10000,
          "pong_timeout_ms": 25000,
          "reconnect_initial_ms": 250,
          "reconnect_max_ms": 30000,
          "artifact_window_seconds": 3600,
          "request_timeout_seconds": 10
        }
        $config$::jsonb,
        'stopped',
        'stopped',
        'unknown'
      );

      CREATE TABLE ${TABLE_NAME} (
        source text NOT NULL,
        market_id text NOT NULL,
        condition_id text NOT NULL,
        event_slug text NOT NULL,
        window_start timestamptz NOT NULL,
        window_end timestamptz NOT NULL,
        up_token_id text NOT NULL,
        down_token_id text NOT NULL,
        winning_token_id text NOT NULL,
        winning_outcome text NOT NULL,
        source_timestamp timestamptz,
        provider_available_at timestamptz,
        received_at timestamptz NOT NULL,
        source_payload jsonb NOT NULL,
        revision_sha256 character(64) NOT NULL,
        payload_sha256 character(64) NOT NULL,
        strategy_key text NOT NULL DEFAULT '${STRATEGY_KEY}',
        capture_artifact_id uuid NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_market_data_polymarket_btc_five_minute_resolutions
          PRIMARY KEY (market_id, source, payload_sha256),
        CONSTRAINT fk_market_data_polymarket_btc_five_minute_resolution_artifact
          FOREIGN KEY (strategy_key, capture_artifact_id)
          REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_resolution_identity CHECK (
          source IN (
            'clob_websocket',
            'clob_rest_reconciliation',
            'gamma_rest_reconciliation'
          )
          AND strategy_key = '${STRATEGY_KEY}'
          AND octet_length(market_id) BETWEEN 1 AND 256
          AND condition_id ~ '^0x[0-9a-f]{64}$'
          AND up_token_id ~ '^[0-9]{1,100}$'
          AND down_token_id ~ '^[0-9]{1,100}$'
          AND up_token_id <> down_token_id
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_resolution_window CHECK (
          window_end = window_start + INTERVAL '5 minutes'
          AND mod(extract(epoch FROM window_start)::bigint, 300) = 0
          AND event_slug =
            'btc-updown-5m-' || (extract(epoch FROM window_start)::bigint)::text
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_resolution_winner CHECK (
          winning_outcome IN ('up', 'down')
          AND (
            (winning_outcome = 'up' AND winning_token_id = up_token_id)
            OR
            (winning_outcome = 'down' AND winning_token_id = down_token_id)
          )
        ),
        CONSTRAINT chk_md_polymarket_btc_five_minute_resolution_source_time CHECK (
          (
            source = 'clob_rest_reconciliation'
            AND source_timestamp IS NULL
            AND provider_available_at IS NULL
          )
          OR
          (
            source IN ('clob_websocket', 'gamma_rest_reconciliation')
            AND source_timestamp IS NOT NULL
            AND provider_available_at = source_timestamp
          )
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_resolution_time CHECK (
          received_at >= window_end
          AND (source_timestamp IS NULL OR source_timestamp >= window_end)
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_resolution_payload CHECK (
          jsonb_typeof(source_payload) = 'object'
          AND octet_length(source_payload::text) <= 1048576
          AND revision_sha256 ~ '^[0-9a-f]{64}$'
          AND payload_sha256 ~ '^[0-9a-f]{64}$'
        )
      );

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_resolution_window
        ON ${TABLE_NAME} (window_end DESC, market_id, source, received_at DESC);

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_resolution_condition
        ON ${TABLE_NAME} (condition_id, received_at DESC);

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_resolution_revision
        ON ${TABLE_NAME} (market_id, revision_sha256, received_at DESC);

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_resolution_artifact
        ON ${TABLE_NAME} (capture_artifact_id, received_at DESC);

      CREATE TRIGGER trg_reject_md_polymarket_btc_five_minute_resolution_change
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
            'refusing to remove Polymarket BTC five-minute resolution ingestion while facts, artifacts, or gaps exist';
        END IF;
      END $$;

      DROP TRIGGER trg_reject_md_polymarket_btc_five_minute_resolution_change
        ON ${TABLE_NAME};
      DROP TABLE ${TABLE_NAME};
      DELETE FROM ingester.profiles WHERE strategy_key = '${STRATEGY_KEY}';
    `);
  }
}
