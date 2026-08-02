import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddBinanceBtcusdtL2OneSecondFeatures1785685200000
  implements MigrationInterface
{
  name = 'AddBinanceBtcusdtL2OneSecondFeatures1785685200000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE polymarket.binance_btcusdt_l2_one_second_features (
        symbol text NOT NULL,
        second_start timestamptz NOT NULL,
        source_event_timestamp timestamptz NOT NULL,
        provider_received_at timestamptz NOT NULL,
        available_at timestamptz NOT NULL,
        source_update_id bigint NOT NULL,
        feature_schema_version text NOT NULL,
        quality_status text NOT NULL,
        artifact_id uuid NOT NULL
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        midpoint numeric(30,10) NOT NULL,
        microprice numeric(30,10) NOT NULL,
        spread_bps numeric(20,10) NOT NULL,
        bid_depth_5 numeric(30,10) NOT NULL,
        ask_depth_5 numeric(30,10) NOT NULL,
        imbalance_5 numeric(20,10) NOT NULL,
        bid_depth_10 numeric(30,10) NOT NULL,
        ask_depth_10 numeric(30,10) NOT NULL,
        imbalance_10 numeric(20,10) NOT NULL,
        bid_depth_20 numeric(30,10) NOT NULL,
        ask_depth_20 numeric(30,10) NOT NULL,
        imbalance_20 numeric(20,10) NOT NULL,
        bid_depth_slope_20 numeric(20,10) NOT NULL,
        ask_depth_slope_20 numeric(20,10) NOT NULL,
        bid_depth_concentration_20 numeric(20,10) NOT NULL,
        ask_depth_concentration_20 numeric(20,10) NOT NULL,
        bid_quote_replenishment_1s numeric(30,10) NOT NULL,
        ask_quote_replenishment_1s numeric(30,10) NOT NULL,
        bid_quote_churn_1s numeric(30,10) NOT NULL,
        ask_quote_churn_1s numeric(30,10) NOT NULL,
        midpoint_change_bps_1s numeric(30,10) NOT NULL,
        spread_bps_delta_1s numeric(30,10) NOT NULL,
        depth_20_change_bps_1s numeric(30,10) NOT NULL,
        imbalance_20_delta_1s numeric(20,10) NOT NULL,
        midpoint_change_bps_5s numeric(30,10) NOT NULL,
        spread_bps_delta_5s numeric(30,10) NOT NULL,
        depth_20_change_bps_5s numeric(30,10) NOT NULL,
        imbalance_20_delta_5s numeric(20,10) NOT NULL,
        midpoint_change_bps_15s numeric(30,10) NOT NULL,
        spread_bps_delta_15s numeric(30,10) NOT NULL,
        depth_20_change_bps_15s numeric(30,10) NOT NULL,
        imbalance_20_delta_15s numeric(20,10) NOT NULL,
        midpoint_change_bps_30s numeric(30,10) NOT NULL,
        spread_bps_delta_30s numeric(30,10) NOT NULL,
        depth_20_change_bps_30s numeric(30,10) NOT NULL,
        imbalance_20_delta_30s numeric(20,10) NOT NULL,
        midpoint_change_bps_60s numeric(30,10) NOT NULL,
        spread_bps_delta_60s numeric(30,10) NOT NULL,
        depth_20_change_bps_60s numeric(30,10) NOT NULL,
        imbalance_20_delta_60s numeric(20,10) NOT NULL,
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_binance_btcusdt_l2_one_second_features
          PRIMARY KEY (artifact_id, symbol, second_start),
        CONSTRAINT chk_binance_btcusdt_l2_symbol
          CHECK (symbol = 'BTCUSDT'),
        CONSTRAINT chk_binance_btcusdt_l2_second_alignment CHECK (
          second_start = date_trunc('second', second_start)
        ),
        CONSTRAINT chk_binance_btcusdt_l2_source_update_id
          CHECK (source_update_id >= 0),
        CONSTRAINT chk_binance_btcusdt_l2_feature_schema CHECK (
          feature_schema_version =
            'binance-btcusdt-l2-one-second-features-v1'
        ),
        CONSTRAINT chk_binance_btcusdt_l2_quality
          CHECK (quality_status = 'qualified'),
        CONSTRAINT chk_binance_btcusdt_l2_causality CHECK (
          source_event_timestamp <= available_at
          AND provider_received_at <= available_at
          AND second_start <= available_at
          AND available_at < second_start + INTERVAL '1 second'
        ),
        CONSTRAINT chk_binance_btcusdt_l2_prices CHECK (
          midpoint > 0
          AND microprice > 0
          AND spread_bps >= 0
        ),
        CONSTRAINT chk_binance_btcusdt_l2_depth CHECK (
          bid_depth_5 > 0
          AND bid_depth_5 <= bid_depth_10
          AND bid_depth_10 <= bid_depth_20
          AND ask_depth_5 > 0
          AND ask_depth_5 <= ask_depth_10
          AND ask_depth_10 <= ask_depth_20
        ),
        CONSTRAINT chk_binance_btcusdt_l2_imbalance CHECK (
          imbalance_5 BETWEEN -1 AND 1
          AND imbalance_10 BETWEEN -1 AND 1
          AND imbalance_20 BETWEEN -1 AND 1
          AND imbalance_20_delta_1s BETWEEN -2 AND 2
          AND imbalance_20_delta_5s BETWEEN -2 AND 2
          AND imbalance_20_delta_15s BETWEEN -2 AND 2
          AND imbalance_20_delta_30s BETWEEN -2 AND 2
          AND imbalance_20_delta_60s BETWEEN -2 AND 2
        ),
        CONSTRAINT chk_binance_btcusdt_l2_shape CHECK (
          bid_depth_slope_20 >= 0
          AND ask_depth_slope_20 >= 0
          AND bid_depth_concentration_20 BETWEEN 0 AND 1
          AND ask_depth_concentration_20 BETWEEN 0 AND 1
        ),
        CONSTRAINT chk_binance_btcusdt_l2_flow CHECK (
          bid_quote_replenishment_1s >= 0
          AND ask_quote_replenishment_1s >= 0
          AND bid_quote_churn_1s >= 0
          AND ask_quote_churn_1s >= 0
        )
      );

      CREATE TABLE polymarket.binance_btcusdt_l2_one_second_features_staging (
        LIKE polymarket.binance_btcusdt_l2_one_second_features
          INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING STORAGE
      );

      ALTER TABLE polymarket.binance_btcusdt_l2_one_second_features_staging
        ADD CONSTRAINT pk_binance_btcusdt_l2_one_second_features_staging
          PRIMARY KEY (artifact_id, symbol, second_start),
        ADD CONSTRAINT fk_binance_btcusdt_l2_one_second_features_staging_artifact
          FOREIGN KEY (artifact_id)
          REFERENCES polymarket.backfill_artifacts (artifact_id)
          ON DELETE RESTRICT;

      CREATE TABLE polymarket.cryptohft_request_budget (
        provider text PRIMARY KEY,
        next_request_at timestamptz NOT NULL DEFAULT now(),
        spacing_milliseconds integer NOT NULL,
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT chk_cryptohft_request_budget_provider
          CHECK (provider = 'cryptohftdata'),
        CONSTRAINT chk_cryptohft_request_budget_spacing
          CHECK (spacing_milliseconds = 1100)
      );

      INSERT INTO polymarket.cryptohft_request_budget (
        provider, spacing_milliseconds
      ) VALUES ('cryptohftdata', 1100);

      SELECT create_hypertable(
        'polymarket.binance_btcusdt_l2_one_second_features',
        'second_start',
        chunk_time_interval => INTERVAL '1 day',
        create_default_indexes => FALSE,
        if_not_exists => TRUE
      );

      ALTER TABLE polymarket.binance_btcusdt_l2_one_second_features SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'second_start ASC',
        timescaledb.compress_segmentby = 'symbol, artifact_id'
      );

      SELECT add_compression_policy(
        'polymarket.binance_btcusdt_l2_one_second_features',
        INTERVAL '7 days',
        if_not_exists => TRUE
      );

      CREATE INDEX idx_binance_btcusdt_l2_artifact_time
        ON polymarket.binance_btcusdt_l2_one_second_features (
          artifact_id, second_start
        );

      CREATE INDEX idx_binance_btcusdt_l2_available_time
        ON polymarket.binance_btcusdt_l2_one_second_features (
          symbol, available_at DESC, second_start DESC
        );

      CREATE VIEW polymarket.binance_btcusdt_l2_training_features AS
      SELECT
        feature.symbol,
        feature.second_start,
        feature.source_event_timestamp,
        feature.provider_received_at,
        feature.available_at,
        feature.source_update_id,
        feature.midpoint,
        feature.microprice,
        feature.spread_bps,
        feature.bid_depth_5,
        feature.ask_depth_5,
        feature.imbalance_5,
        feature.bid_depth_10,
        feature.ask_depth_10,
        feature.imbalance_10,
        feature.bid_depth_20,
        feature.ask_depth_20,
        feature.imbalance_20,
        feature.bid_depth_slope_20,
        feature.ask_depth_slope_20,
        feature.bid_depth_concentration_20,
        feature.ask_depth_concentration_20,
        feature.bid_quote_replenishment_1s,
        feature.ask_quote_replenishment_1s,
        feature.bid_quote_churn_1s,
        feature.ask_quote_churn_1s,
        feature.midpoint_change_bps_1s,
        feature.spread_bps_delta_1s,
        feature.depth_20_change_bps_1s,
        feature.imbalance_20_delta_1s,
        feature.midpoint_change_bps_5s,
        feature.spread_bps_delta_5s,
        feature.depth_20_change_bps_5s,
        feature.imbalance_20_delta_5s,
        feature.midpoint_change_bps_15s,
        feature.spread_bps_delta_15s,
        feature.depth_20_change_bps_15s,
        feature.imbalance_20_delta_15s,
        feature.midpoint_change_bps_30s,
        feature.spread_bps_delta_30s,
        feature.depth_20_change_bps_30s,
        feature.imbalance_20_delta_30s,
        feature.midpoint_change_bps_60s,
        feature.spread_bps_delta_60s,
        feature.depth_20_change_bps_60s,
        feature.imbalance_20_delta_60s
      FROM polymarket.binance_btcusdt_l2_one_second_features feature
      JOIN polymarket.backfill_artifacts artifact
        ON artifact.artifact_id = feature.artifact_id
       AND artifact.status = 'completed'
       AND artifact.metadata ->> 'materialization_contract' =
         'cryptohft-binance-futures-btcusdt-l2-features-v1';
    `);

    await queryRunner.query(`
      CREATE FUNCTION polymarket.reject_binance_btcusdt_l2_feature_change()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        RAISE EXCEPTION
          'historical Binance BTCUSDT L2 feature row is immutable'
          USING ERRCODE = 'integrity_constraint_violation';
      END;
      $$;

      CREATE TRIGGER trg_reject_binance_btcusdt_l2_feature_change
        BEFORE UPDATE OR DELETE
        ON polymarket.binance_btcusdt_l2_one_second_features
        FOR EACH ROW
        EXECUTE FUNCTION polymarket.reject_binance_btcusdt_l2_feature_change();
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
          FROM polymarket.binance_btcusdt_l2_one_second_features
          LIMIT 1
        ) OR EXISTS (
          SELECT 1
          FROM polymarket.binance_btcusdt_l2_one_second_features_staging
          LIMIT 1
        ) OR EXISTS (
          SELECT 1
          FROM polymarket.backfill_artifacts
          WHERE ingester_key = 'binance_btcusdt_l2_one_second_features'
          LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to remove Binance BTCUSDT L2 features while rows exist';
        END IF;
      END;
      $$;

      DROP VIEW polymarket.binance_btcusdt_l2_training_features;
      DROP TRIGGER trg_reject_binance_btcusdt_l2_feature_change
        ON polymarket.binance_btcusdt_l2_one_second_features;
      DROP FUNCTION polymarket.reject_binance_btcusdt_l2_feature_change();
      DROP TABLE polymarket.binance_btcusdt_l2_one_second_features_staging;
      DROP TABLE polymarket.binance_btcusdt_l2_one_second_features;
      DROP TABLE polymarket.cryptohft_request_budget;
    `);
  }
}
