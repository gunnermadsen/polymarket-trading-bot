import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'polymarket_btc_five_minute_orderbooks';
const TABLE_NAME =
  'market_data.polymarket_btc_five_minute_orderbook_snapshots';

export class AddPolymarketBtcFiveMinuteOrderbookSnapshots1786381900000
  implements MigrationInterface
{
  name = 'AddPolymarketBtcFiveMinuteOrderbookSnapshots1786381900000';

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
          "websocket_url": "wss://ws-subscriptions-clob.polymarket.com/ws/market",
          "gamma_api_url": "https://gamma-api.polymarket.com",
          "sample_interval_ms": 1000,
          "top_n": 20,
          "gamma_refresh_ms": 5000,
          "lookback_windows": 1,
          "lookahead_windows": 1,
          "successor_lead_ms": 30000,
          "contract_grace_ms": 30000,
          "connect_timeout_ms": 10000,
          "bootstrap_timeout_ms": 15000,
          "read_timeout_ms": 40000,
          "ping_interval_ms": 10000,
          "pong_timeout_ms": 25000,
          "reconnect_initial_ms": 250,
          "reconnect_max_ms": 30000,
          "max_levels_per_side": 10000,
          "artifact_window_seconds": 3600
        }
        $config$::jsonb,
        'stopped',
        'stopped',
        'unknown'
      );

      CREATE FUNCTION market_data.is_valid_polymarket_btc_five_minute_book(
        bids jsonb,
        asks jsonb,
        bid_depth integer,
        ask_depth integer,
        best_bid numeric,
        best_ask numeric,
        maximum_depth integer
      )
      RETURNS boolean
      LANGUAGE plpgsql
      IMMUTABLE
      PARALLEL SAFE
      SET search_path = pg_catalog
      AS $$
      DECLARE
        level jsonb;
        raw_price text;
        raw_quantity text;
        price numeric;
        quantity numeric;
        previous_price numeric;
        derived_best_bid numeric;
        derived_best_ask numeric;
      BEGIN
        IF jsonb_typeof(bids) <> 'array'
          OR jsonb_typeof(asks) <> 'array'
          OR maximum_depth NOT BETWEEN 1 AND 1000
          OR bid_depth NOT BETWEEN 0 AND maximum_depth
          OR ask_depth NOT BETWEEN 0 AND maximum_depth
          OR jsonb_array_length(bids) <> bid_depth
          OR jsonb_array_length(asks) <> ask_depth THEN
          RETURN FALSE;
        END IF;

        previous_price := NULL;
        FOR level IN SELECT value FROM jsonb_array_elements(bids) LOOP
          IF jsonb_typeof(level) <> 'array'
            OR jsonb_array_length(level) <> 2
            OR jsonb_typeof(level -> 0) <> 'string'
            OR jsonb_typeof(level -> 1) <> 'string' THEN
            RETURN FALSE;
          END IF;

          raw_price := level ->> 0;
          raw_quantity := level ->> 1;
          IF length(raw_price) NOT BETWEEN 1 AND 64
            OR length(raw_quantity) NOT BETWEEN 1 AND 64
            OR raw_price !~ '^(0|[1-9][0-9]*)(\\.[0-9]+)?$'
            OR raw_quantity !~ '^(0|[1-9][0-9]*)(\\.[0-9]+)?$' THEN
            RETURN FALSE;
          END IF;

          price := raw_price::numeric;
          quantity := raw_quantity::numeric;
          IF price <= 0
            OR price >= 1
            OR quantity <= 0
            OR (previous_price IS NOT NULL AND price >= previous_price) THEN
            RETURN FALSE;
          END IF;
          IF derived_best_bid IS NULL THEN
            derived_best_bid := price;
          END IF;
          previous_price := price;
        END LOOP;

        previous_price := NULL;
        FOR level IN SELECT value FROM jsonb_array_elements(asks) LOOP
          IF jsonb_typeof(level) <> 'array'
            OR jsonb_array_length(level) <> 2
            OR jsonb_typeof(level -> 0) <> 'string'
            OR jsonb_typeof(level -> 1) <> 'string' THEN
            RETURN FALSE;
          END IF;

          raw_price := level ->> 0;
          raw_quantity := level ->> 1;
          IF length(raw_price) NOT BETWEEN 1 AND 64
            OR length(raw_quantity) NOT BETWEEN 1 AND 64
            OR raw_price !~ '^(0|[1-9][0-9]*)(\\.[0-9]+)?$'
            OR raw_quantity !~ '^(0|[1-9][0-9]*)(\\.[0-9]+)?$' THEN
            RETURN FALSE;
          END IF;

          price := raw_price::numeric;
          quantity := raw_quantity::numeric;
          IF price <= 0
            OR price >= 1
            OR quantity <= 0
            OR (previous_price IS NOT NULL AND price <= previous_price) THEN
            RETURN FALSE;
          END IF;
          IF derived_best_ask IS NULL THEN
            derived_best_ask := price;
          END IF;
          previous_price := price;
        END LOOP;

        RETURN (best_bid IS NOT DISTINCT FROM derived_best_bid)
          AND (best_ask IS NOT DISTINCT FROM derived_best_ask)
          AND (
            derived_best_bid IS NULL
            OR derived_best_ask IS NULL
            OR derived_best_bid < derived_best_ask
          );
      END;
      $$;

      CREATE TABLE ${TABLE_NAME} (
        sampled_at timestamptz NOT NULL,
        source_timestamp timestamptz NOT NULL,
        provider_available_at timestamptz NOT NULL,
        received_at timestamptz NOT NULL,
        source text NOT NULL DEFAULT 'polymarket_clob_market',
        market_id text NOT NULL,
        condition_id text NOT NULL,
        event_slug text NOT NULL,
        window_start timestamptz NOT NULL,
        window_end timestamptz NOT NULL,
        token_id text NOT NULL,
        outcome text NOT NULL,
        connection_epoch uuid NOT NULL,
        ingest_sequence bigint NOT NULL,
        tick_size numeric(18,8) NOT NULL,
        best_bid numeric(18,8),
        best_ask numeric(18,8),
        bid_depth integer NOT NULL,
        ask_depth integer NOT NULL,
        bids jsonb NOT NULL,
        asks jsonb NOT NULL,
        source_hash text,
        book_sha256 character(64) NOT NULL,
        sampling_policy jsonb NOT NULL,
        sampling_policy_sha256 character(64) NOT NULL,
        payload_sha256 character(64) NOT NULL,
        strategy_key text NOT NULL DEFAULT '${STRATEGY_KEY}',
        capture_artifact_id uuid NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_market_data_polymarket_btc_five_minute_orderbook_snapshots
          PRIMARY KEY (
            sampled_at, market_id, token_id, sampling_policy_sha256
          ),
        CONSTRAINT fk_market_data_polymarket_btc_five_minute_orderbook_artifact
          FOREIGN KEY (strategy_key, capture_artifact_id)
          REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_identity CHECK (
          source = 'polymarket_clob_market'
          AND strategy_key = '${STRATEGY_KEY}'
          AND octet_length(market_id) BETWEEN 1 AND 256
          AND condition_id ~ '^0x[0-9a-f]{64}$'
          AND token_id ~ '^[0-9]{1,100}$'
          AND outcome IN ('up', 'down')
          AND ingest_sequence >= 0
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_window CHECK (
          window_end = window_start + INTERVAL '5 minutes'
          AND mod(extract(epoch FROM window_start)::bigint, 300) = 0
          AND event_slug =
            'btc-updown-5m-' || (extract(epoch FROM window_start)::bigint)::text
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_time CHECK (
          provider_available_at = source_timestamp
          AND received_at <= sampled_at
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_prices CHECK (
          tick_size > 0
          AND tick_size < 1
          AND (best_bid IS NULL OR (best_bid > 0 AND best_bid < 1))
          AND (best_ask IS NULL OR (best_ask > 0 AND best_ask < 1))
          AND (best_bid IS NULL OR best_ask IS NULL OR best_bid < best_ask)
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_sampling CHECK (
          jsonb_typeof(sampling_policy) = 'object'
          AND octet_length(sampling_policy::text) <= 4096
          AND sampling_policy ->> 'version' =
            'polymarket-clob-btc-5m-orderbook-top-n-v1'
          AND sampling_policy ->> 'source' = 'polymarket_clob_market'
          AND sampling_policy ->> 'selection' =
            'latest_valid_subscribed_market_book_at_aligned_wall_clock_slot'
          AND jsonb_typeof(
            sampling_policy -> 'market_interval_seconds'
          ) = 'number'
          AND (sampling_policy ->> 'market_interval_seconds')::integer = 300
          AND jsonb_typeof(sampling_policy -> 'top_n') = 'number'
          AND (sampling_policy ->> 'top_n')::integer BETWEEN 1 AND 1000
          AND jsonb_typeof(sampling_policy -> 'sample_interval_ms') = 'number'
          AND (sampling_policy ->> 'sample_interval_ms')::integer
            BETWEEN 100 AND 60000
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_book CHECK (
          octet_length(bids::text) <= 262144
          AND octet_length(asks::text) <= 262144
          AND market_data.is_valid_polymarket_btc_five_minute_book(
            bids,
            asks,
            bid_depth,
            ask_depth,
            best_bid,
            best_ask,
            (sampling_policy ->> 'top_n')::integer
          )
        ),
        CONSTRAINT chk_market_data_polymarket_btc_five_minute_orderbook_payload CHECK (
          (source_hash IS NULL OR octet_length(source_hash) BETWEEN 1 AND 256)
          AND book_sha256 ~ '^[0-9a-f]{64}$'
          AND sampling_policy_sha256 ~ '^[0-9a-f]{64}$'
          AND payload_sha256 ~ '^[0-9a-f]{64}$'
        )
      );

      SELECT create_hypertable(
        '${TABLE_NAME}',
        'sampled_at',
        chunk_time_interval => INTERVAL '1 day',
        create_default_indexes => FALSE,
        if_not_exists => TRUE
      );

      ALTER TABLE ${TABLE_NAME} SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby =
          'sampled_at ASC, source_timestamp ASC, ingest_sequence ASC',
        timescaledb.compress_segmentby =
          'market_id, token_id, sampling_policy_sha256, strategy_key, capture_artifact_id'
      );

      SELECT add_compression_policy(
        '${TABLE_NAME}', INTERVAL '1 day', if_not_exists => TRUE
      );

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_orderbook_recovery
        ON ${TABLE_NAME} (
          market_id, token_id, source_timestamp DESC, sampled_at DESC
        );

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_orderbook_condition
        ON ${TABLE_NAME} (condition_id, sampled_at DESC);

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_orderbook_window
        ON ${TABLE_NAME} (window_start, outcome, sampled_at DESC);

      CREATE INDEX idx_market_data_polymarket_btc_five_minute_orderbook_artifact
        ON ${TABLE_NAME} (capture_artifact_id, sampled_at DESC);

      CREATE TRIGGER trg_reject_md_polymarket_btc_five_minute_orderbook_change
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
            'refusing to remove Polymarket BTC five-minute orderbook ingestion while facts, artifacts, or gaps exist';
        END IF;
      END $$;

      SELECT remove_compression_policy('${TABLE_NAME}', if_exists => TRUE);
      DROP TRIGGER trg_reject_md_polymarket_btc_five_minute_orderbook_change
        ON ${TABLE_NAME};
      DROP TABLE ${TABLE_NAME};
      DROP FUNCTION market_data.is_valid_polymarket_btc_five_minute_book(
        jsonb, jsonb, integer, integer, numeric, numeric, integer
      );
      DELETE FROM ingester.profiles WHERE strategy_key = '${STRATEGY_KEY}';
    `);
  }
}
