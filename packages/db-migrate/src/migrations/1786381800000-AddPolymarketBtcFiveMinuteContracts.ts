import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'polymarket_btc_five_minute_market_contracts';
const TABLE_NAME = 'market_data.polymarket_btc_five_minute_contracts';

export class AddPolymarketBtcFiveMinuteContracts1786381800000
  implements MigrationInterface
{
  name = 'AddPolymarketBtcFiveMinuteContracts1786381800000';

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
          "gamma_base_url": "https://gamma-api.polymarket.com",
          "poll_interval_seconds": 5,
          "startup_lookback_windows": 12,
          "lookahead_windows": 1,
          "overlap_windows": 2,
          "artifact_window_seconds": 3600,
          "request_timeout_seconds": 10
        }
        $config$::jsonb,
        'stopped',
        'stopped',
        'unknown'
      );

      CREATE TABLE ${TABLE_NAME} (
        source text NOT NULL DEFAULT 'polymarket_gamma_rest',
        event_id text NOT NULL,
        event_slug text NOT NULL,
        series_slug text NOT NULL,
        market_id text NOT NULL,
        condition_id text NOT NULL,
        window_start timestamptz NOT NULL,
        window_end timestamptz NOT NULL,
        up_token_id text NOT NULL,
        down_token_id text NOT NULL,
        tick_size numeric(18,8) NOT NULL,
        minimum_order_size numeric(38,18),
        resolution_source text NOT NULL,
        active boolean NOT NULL,
        closed boolean NOT NULL,
        accepting_orders boolean NOT NULL,
        fees_enabled boolean NOT NULL,
        fee_schedule jsonb NOT NULL,
        received_at timestamptz NOT NULL,
        source_payload jsonb NOT NULL,
        revision_sha256 character(64) NOT NULL,
        payload_sha256 character(64) NOT NULL,
        strategy_key text NOT NULL DEFAULT '${STRATEGY_KEY}',
        capture_artifact_id uuid NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_market_data_polymarket_btc_five_minute_contracts
          PRIMARY KEY (market_id, revision_sha256),
        CONSTRAINT fk_market_data_polymarket_btc_five_minute_contracts_artifact
          FOREIGN KEY (strategy_key, capture_artifact_id)
          REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_contract_identity CHECK (
          source = 'polymarket_gamma_rest'
          AND strategy_key = '${STRATEGY_KEY}'
          AND series_slug = 'btc-up-or-down-5m'
          AND octet_length(event_id) BETWEEN 1 AND 256
          AND octet_length(market_id) BETWEEN 1 AND 256
          AND condition_id ~ '^0x[0-9a-f]{64}$'
          AND up_token_id ~ '^[0-9]{1,100}$'
          AND down_token_id ~ '^[0-9]{1,100}$'
          AND up_token_id <> down_token_id
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_contract_window CHECK (
          window_end = window_start + INTERVAL '5 minutes'
          AND mod(extract(epoch FROM window_start)::bigint, 300) = 0
          AND event_slug =
            'btc-updown-5m-' || (extract(epoch FROM window_start)::bigint)::text
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_contract_terms CHECK (
          tick_size > 0
          AND tick_size < 1
          AND (minimum_order_size IS NULL OR minimum_order_size > 0)
          AND octet_length(resolution_source) BETWEEN 1 AND 2048
          AND lower(resolution_source) ~ '(chainlink|chain\\.link)'
          AND lower(resolution_source) ~ 'btc'
          AND lower(resolution_source) ~ 'usd'
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_contract_fee CHECK (
          jsonb_typeof(fee_schedule) = 'object'
          AND octet_length(fee_schedule::text) <= 32768
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_contract_payload CHECK (
          jsonb_typeof(source_payload) = 'object'
          AND octet_length(source_payload::text) <= 65536
          AND source_payload ->> 'version' =
            'polymarket-btc-5m-contract-v1'
          AND revision_sha256 ~ '^[0-9a-f]{64}$'
          AND payload_sha256 ~ '^[0-9a-f]{64}$'
          AND revision_sha256 = payload_sha256
        )
      );

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_window
        ON ${TABLE_NAME} (window_start DESC, market_id, received_at DESC);

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_condition
        ON ${TABLE_NAME} (condition_id, received_at DESC);

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_event_id
        ON ${TABLE_NAME} (event_id, received_at DESC, market_id);

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_event_slug
        ON ${TABLE_NAME} (event_slug, received_at DESC, market_id);

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_up_token
        ON ${TABLE_NAME} (up_token_id, window_start DESC);

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_down_token
        ON ${TABLE_NAME} (down_token_id, window_start DESC);

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_contract_artifact
        ON ${TABLE_NAME} (
          capture_artifact_id, window_start, market_id, revision_sha256
        );

      CREATE INDEX idx_ingester_gap_pm_btc5m_contract_retry
        ON ingester.data_gaps (updated_at, detected_at, gap_id)
        WHERE strategy_key = '${STRATEGY_KEY}'
          AND reason_code = 'polymarket_gamma_contract_window_unavailable'
          AND status IN ('open', 'repairing');

      CREATE TRIGGER trg_reject_md_polymarket_btc_five_minute_contract_change
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
            'refusing to remove Polymarket BTC five-minute contract ingestion while facts, artifacts, or gaps exist';
        END IF;
      END $$;

      DROP INDEX ingester.idx_ingester_gap_pm_btc5m_contract_retry;
      DROP TRIGGER trg_reject_md_polymarket_btc_five_minute_contract_change
        ON ${TABLE_NAME};
      DROP TABLE ${TABLE_NAME};
      DELETE FROM ingester.profiles WHERE strategy_key = '${STRATEGY_KEY}';
    `);
  }
}
