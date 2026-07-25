import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddBtcHistoricalExecutionSnapshots1784914000000 implements MigrationInterface {
  name = 'AddBtcHistoricalExecutionSnapshots1784914000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE polymarket.btc_market_execution_snapshots (
        market_id text NOT NULL
          REFERENCES polymarket.btc_interval_markets (market_id) ON DELETE RESTRICT,
        sampled_at timestamptz NOT NULL,
        artifact_id uuid NOT NULL
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        schema_version text NOT NULL,
        up_source_row_number bigint,
        up_source_timestamp timestamptz,
        up_provider_received_at timestamptz,
        up_best_bid numeric(18,8),
        up_best_ask numeric(18,8),
        up_best_bid_size numeric(30,10),
        up_best_ask_size numeric(30,10),
        up_bid_depth numeric(30,10),
        up_ask_depth numeric(30,10),
        up_ask_vwap_1 numeric(18,8),
        up_ask_vwap_5 numeric(18,8),
        up_ask_vwap_10 numeric(18,8),
        up_imbalance numeric(18,8),
        down_source_row_number bigint,
        down_source_timestamp timestamptz,
        down_provider_received_at timestamptz,
        down_best_bid numeric(18,8),
        down_best_ask numeric(18,8),
        down_best_bid_size numeric(30,10),
        down_best_ask_size numeric(30,10),
        down_bid_depth numeric(30,10),
        down_ask_depth numeric(30,10),
        down_ask_vwap_1 numeric(18,8),
        down_ask_vwap_5 numeric(18,8),
        down_ask_vwap_10 numeric(18,8),
        down_imbalance numeric(18,8),
        quality_flags integer NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_btc_market_execution_snapshots
          PRIMARY KEY (market_id, sampled_at),
        CONSTRAINT chk_btc_market_execution_snapshot_schema
          CHECK (length(btrim(schema_version)) > 0),
        CONSTRAINT chk_btc_market_execution_snapshot_source_rows CHECK (
          (up_source_row_number IS NULL OR up_source_row_number >= 0)
          AND (down_source_row_number IS NULL OR down_source_row_number >= 0)
        ),
        CONSTRAINT chk_btc_market_execution_snapshot_causality CHECK (
          (up_provider_received_at IS NULL OR up_provider_received_at <= sampled_at)
          AND (down_provider_received_at IS NULL OR down_provider_received_at <= sampled_at)
        ),
        CONSTRAINT chk_btc_market_execution_snapshot_prices CHECK (
          (up_best_bid IS NULL OR (up_best_bid >= 0 AND up_best_bid <= 1))
          AND (up_best_ask IS NULL OR (up_best_ask >= 0 AND up_best_ask <= 1))
          AND (down_best_bid IS NULL OR (down_best_bid >= 0 AND down_best_bid <= 1))
          AND (down_best_ask IS NULL OR (down_best_ask >= 0 AND down_best_ask <= 1))
          AND (up_ask_vwap_1 IS NULL OR (up_ask_vwap_1 >= 0 AND up_ask_vwap_1 <= 1))
          AND (up_ask_vwap_5 IS NULL OR (up_ask_vwap_5 >= 0 AND up_ask_vwap_5 <= 1))
          AND (up_ask_vwap_10 IS NULL OR (up_ask_vwap_10 >= 0 AND up_ask_vwap_10 <= 1))
          AND (down_ask_vwap_1 IS NULL OR (down_ask_vwap_1 >= 0 AND down_ask_vwap_1 <= 1))
          AND (down_ask_vwap_5 IS NULL OR (down_ask_vwap_5 >= 0 AND down_ask_vwap_5 <= 1))
          AND (down_ask_vwap_10 IS NULL OR (down_ask_vwap_10 >= 0 AND down_ask_vwap_10 <= 1))
        ),
        CONSTRAINT chk_btc_market_execution_snapshot_sizes CHECK (
          (up_best_bid_size IS NULL OR up_best_bid_size >= 0)
          AND (up_best_ask_size IS NULL OR up_best_ask_size >= 0)
          AND (up_bid_depth IS NULL OR up_bid_depth >= 0)
          AND (up_ask_depth IS NULL OR up_ask_depth >= 0)
          AND (down_best_bid_size IS NULL OR down_best_bid_size >= 0)
          AND (down_best_ask_size IS NULL OR down_best_ask_size >= 0)
          AND (down_bid_depth IS NULL OR down_bid_depth >= 0)
          AND (down_ask_depth IS NULL OR down_ask_depth >= 0)
        ),
        CONSTRAINT chk_btc_market_execution_snapshot_books CHECK (
          (up_best_bid IS NULL OR up_best_ask IS NULL OR up_best_bid < up_best_ask)
          AND (down_best_bid IS NULL OR down_best_ask IS NULL OR down_best_bid < down_best_ask)
        ),
        CONSTRAINT chk_btc_market_execution_snapshot_vwap CHECK (
          (up_ask_vwap_1 IS NULL OR up_best_ask IS NULL OR up_ask_vwap_1 >= up_best_ask)
          AND (up_ask_vwap_5 IS NULL OR up_ask_vwap_1 IS NULL OR up_ask_vwap_5 >= up_ask_vwap_1)
          AND (up_ask_vwap_10 IS NULL OR up_ask_vwap_5 IS NULL OR up_ask_vwap_10 >= up_ask_vwap_5)
          AND (down_ask_vwap_1 IS NULL OR down_best_ask IS NULL
            OR down_ask_vwap_1 >= down_best_ask)
          AND (down_ask_vwap_5 IS NULL OR down_ask_vwap_1 IS NULL
            OR down_ask_vwap_5 >= down_ask_vwap_1)
          AND (down_ask_vwap_10 IS NULL OR down_ask_vwap_5 IS NULL
            OR down_ask_vwap_10 >= down_ask_vwap_5)
        ),
        CONSTRAINT chk_btc_market_execution_snapshot_imbalance CHECK (
          (up_imbalance IS NULL OR (up_imbalance >= -1 AND up_imbalance <= 1))
          AND (down_imbalance IS NULL OR (down_imbalance >= -1 AND down_imbalance <= 1))
        ),
        CONSTRAINT chk_btc_market_execution_snapshot_quality CHECK (quality_flags >= 0)
      );

      SELECT create_hypertable(
        'polymarket.btc_market_execution_snapshots',
        'sampled_at',
        chunk_time_interval => INTERVAL '1 day',
        create_default_indexes => FALSE,
        if_not_exists => TRUE
      );

      ALTER TABLE polymarket.btc_market_execution_snapshots SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'sampled_at ASC',
        timescaledb.compress_segmentby = 'market_id, artifact_id'
      );

      CREATE INDEX idx_btc_market_execution_snapshots_artifact
        ON polymarket.btc_market_execution_snapshots (artifact_id, sampled_at);

      SELECT add_compression_policy(
        'polymarket.btc_market_execution_snapshots',
        INTERVAL '7 days',
        if_not_exists => true
      );
    `);

    await queryRunner.query(`
      CREATE FUNCTION polymarket.reject_btc_market_execution_snapshot_change()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        RAISE EXCEPTION
          'historical BTC execution snapshot is immutable'
          USING ERRCODE = 'integrity_constraint_violation';
      END;
      $$;

      CREATE TRIGGER trg_reject_btc_market_execution_snapshot_change
        BEFORE UPDATE OR DELETE ON polymarket.btc_market_execution_snapshots
        FOR EACH ROW
        EXECUTE FUNCTION polymarket.reject_btc_market_execution_snapshot_change();

      CREATE VIEW polymarket.btc_market_execution_snapshots_one_second AS
      SELECT *
      FROM polymarket.btc_market_execution_snapshots
      WHERE extract(milliseconds FROM sampled_at)::integer % 1000 = 0;
    `);

    await queryRunner.query(`
      CREATE TABLE polymarket.backfill_materialization_retention_events (
        retention_event_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        source_artifact_id uuid NOT NULL
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        replacement_artifact_id uuid
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        materialization text NOT NULL,
        action text NOT NULL,
        source_record_count bigint NOT NULL,
        occurred_at timestamptz NOT NULL DEFAULT now(),
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT uq_backfill_materialization_retention_event
          UNIQUE (source_artifact_id, materialization, action),
        CONSTRAINT chk_backfill_materialization_retention_identity CHECK (
          length(btrim(materialization)) > 0
          AND action IN ('replaced','pruned')
        ),
        CONSTRAINT chk_backfill_materialization_retention_count
          CHECK (source_record_count >= 0),
        CONSTRAINT chk_backfill_materialization_retention_metadata
          CHECK (jsonb_typeof(metadata) = 'object'),
        CONSTRAINT chk_backfill_materialization_replacement CHECK (
          action <> 'replaced' OR replacement_artifact_id IS NOT NULL
        )
      );

      CREATE INDEX idx_backfill_materialization_retention_replacement
        ON polymarket.backfill_materialization_retention_events (
          replacement_artifact_id, occurred_at
        )
        WHERE replacement_artifact_id IS NOT NULL;

      CREATE FUNCTION polymarket.reject_backfill_retention_event_change()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        RAISE EXCEPTION
          'backfill materialization retention evidence is immutable'
          USING ERRCODE = 'integrity_constraint_violation';
      END;
      $$;

      CREATE TRIGGER trg_reject_backfill_retention_event_change
        BEFORE UPDATE OR DELETE ON polymarket.backfill_materialization_retention_events
        FOR EACH ROW
        EXECUTE FUNCTION polymarket.reject_backfill_retention_event_change();
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1 FROM polymarket.btc_market_execution_snapshots LIMIT 1
        ) OR EXISTS (
          SELECT 1 FROM polymarket.backfill_materialization_retention_events LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to remove historical execution snapshots or retention evidence';
        END IF;
      END;
      $$;

      DROP TRIGGER trg_reject_backfill_retention_event_change
        ON polymarket.backfill_materialization_retention_events;
      DROP FUNCTION polymarket.reject_backfill_retention_event_change();
      DROP TABLE polymarket.backfill_materialization_retention_events;

      DROP TRIGGER trg_reject_btc_market_execution_snapshot_change
        ON polymarket.btc_market_execution_snapshots;
      DROP FUNCTION polymarket.reject_btc_market_execution_snapshot_change();

      DROP VIEW polymarket.btc_market_execution_snapshots_one_second;
      DROP TABLE polymarket.btc_market_execution_snapshots;
    `);
  }
}
