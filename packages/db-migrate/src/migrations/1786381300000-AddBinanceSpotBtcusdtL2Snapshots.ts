import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'binance_spot_btcusdt_l2_snapshots';
const TABLE_NAME = 'market_data.binance_spot_btcusdt_l2_snapshots';

export class AddBinanceSpotBtcusdtL2Snapshots1786381300000
  implements MigrationInterface
{
  name = 'AddBinanceSpotBtcusdtL2Snapshots1786381300000';

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
          "websocket_url": "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms",
          "rest_depth_url": "https://api.binance.com/api/v3/depth",
          "rest_depth_limit": 5000,
          "top_n": 20,
          "sample_interval_ms": 1000,
          "max_buffered_updates": 4096,
          "max_buffered_levels": 200000,
          "max_book_levels_per_side": 100000,
          "connect_timeout_ms": 10000,
          "bootstrap_timeout_ms": 10000,
          "read_timeout_ms": 45000,
          "ping_interval_ms": 15000,
          "reconnect_initial_ms": 250,
          "reconnect_max_ms": 30000,
          "artifact_window_seconds": 3600
        }
        $config$::jsonb,
        'stopped',
        'stopped',
        'unknown'
      );

      CREATE FUNCTION market_data.is_valid_binance_spot_btcusdt_l2_book(
        bids jsonb,
        asks jsonb,
        sample_depth integer
      )
      RETURNS boolean
      LANGUAGE plpgsql
      IMMUTABLE
      STRICT
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
        best_bid numeric;
        best_ask numeric;
      BEGIN
        IF jsonb_typeof(bids) <> 'array'
          OR jsonb_typeof(asks) <> 'array'
          OR sample_depth NOT BETWEEN 1 AND 1000
          OR jsonb_array_length(bids) <> sample_depth
          OR jsonb_array_length(asks) <> sample_depth THEN
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
            OR quantity <= 0
            OR (previous_price IS NOT NULL AND price >= previous_price) THEN
            RETURN FALSE;
          END IF;
          IF best_bid IS NULL THEN
            best_bid := price;
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
            OR quantity <= 0
            OR (previous_price IS NOT NULL AND price <= previous_price) THEN
            RETURN FALSE;
          END IF;
          IF best_ask IS NULL THEN
            best_ask := price;
          END IF;
          previous_price := price;
        END LOOP;

        RETURN best_bid IS NOT NULL
          AND best_ask IS NOT NULL
          AND best_bid < best_ask;
      END;
      $$;

      CREATE TABLE ${TABLE_NAME} (
        source_timestamp timestamptz NOT NULL,
        received_at timestamptz NOT NULL,
        source text NOT NULL DEFAULT 'binance_spot',
        symbol text NOT NULL,
        source_update_id bigint NOT NULL,
        connection_epoch uuid NOT NULL,
        sample_depth integer NOT NULL,
        bids jsonb NOT NULL,
        asks jsonb NOT NULL,
        book_sha256 text NOT NULL,
        sampling_policy jsonb NOT NULL,
        sampling_policy_sha256 text NOT NULL,
        payload_sha256 text NOT NULL,
        strategy_key text NOT NULL DEFAULT '${STRATEGY_KEY}',
        capture_artifact_id uuid NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        CONSTRAINT pk_market_data_binance_spot_l2_snapshots PRIMARY KEY (
          source_timestamp, symbol, source_update_id, sampling_policy_sha256
        ),
        CONSTRAINT fk_market_data_binance_spot_l2_snapshot_artifact
          FOREIGN KEY (strategy_key, capture_artifact_id)
          REFERENCES ingester.capture_artifacts (strategy_key, artifact_id)
          ON DELETE RESTRICT,
        CONSTRAINT chk_market_data_binance_spot_l2_snapshot_source CHECK (
          source = 'binance_spot'
          AND symbol = 'BTCUSDT'
          AND strategy_key = '${STRATEGY_KEY}'
        ),
        CONSTRAINT chk_market_data_binance_spot_l2_snapshot_identity CHECK (
          source_update_id >= 0
          AND sample_depth BETWEEN 1 AND 1000
        ),
        CONSTRAINT chk_market_data_binance_spot_l2_snapshot_book CHECK (
          jsonb_typeof(bids) = 'array'
          AND jsonb_typeof(asks) = 'array'
          AND jsonb_array_length(bids) = sample_depth
          AND jsonb_array_length(asks) = sample_depth
          AND octet_length(bids::text) <= 262144
          AND octet_length(asks::text) <= 262144
          AND market_data.is_valid_binance_spot_btcusdt_l2_book(
            bids, asks, sample_depth
          )
        ),
        CONSTRAINT chk_market_data_binance_spot_l2_snapshot_sampling CHECK (
          jsonb_typeof(sampling_policy) = 'object'
          AND octet_length(sampling_policy::text) <= 4096
          AND sampling_policy ->> 'version' =
            'binance-spot-btcusdt-l2-top-n-v1'
          AND sampling_policy ->> 'source' = 'binance_spot_diff_depth'
          AND sampling_policy ->> 'symbol' = 'BTCUSDT'
          AND jsonb_typeof(sampling_policy -> 'sample_depth') = 'number'
          AND (sampling_policy ->> 'sample_depth')::integer = sample_depth
          AND sampling_policy ->> 'selection' =
            'latest_contiguous_update_at_or_after_interval'
        ),
        CONSTRAINT chk_market_data_binance_spot_l2_snapshot_hashes CHECK (
          book_sha256 ~ '^[0-9a-f]{64}$'
          AND sampling_policy_sha256 ~ '^[0-9a-f]{64}$'
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
        timescaledb.compress_orderby =
          'source_timestamp ASC, source_update_id ASC',
        timescaledb.compress_segmentby =
          'symbol, source, sampling_policy_sha256, strategy_key, capture_artifact_id'
      );

      SELECT add_compression_policy(
        '${TABLE_NAME}', INTERVAL '1 day', if_not_exists => TRUE
      );

      CREATE INDEX idx_market_data_binance_spot_l2_snapshot_recovery
        ON ${TABLE_NAME} (
          symbol, source_update_id DESC,
          sampling_policy_sha256, source_timestamp DESC
        );

      CREATE INDEX idx_market_data_binance_spot_l2_snapshot_artifact
        ON ${TABLE_NAME} (capture_artifact_id, source_timestamp DESC);

      CREATE TRIGGER trg_reject_market_data_binance_spot_l2_snapshot_change
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
            'refusing to remove Binance spot BTCUSDT L2 snapshot ingestion while facts, artifacts, or gaps exist';
        END IF;
      END $$;

      SELECT remove_compression_policy('${TABLE_NAME}', if_exists => TRUE);
      DROP TRIGGER trg_reject_market_data_binance_spot_l2_snapshot_change
        ON ${TABLE_NAME};
      DROP TABLE ${TABLE_NAME};
      DROP FUNCTION market_data.is_valid_binance_spot_btcusdt_l2_book(
        jsonb, jsonb, integer
      );
      DELETE FROM ingester.profiles WHERE strategy_key = '${STRATEGY_KEY}';
    `);
  }
}
