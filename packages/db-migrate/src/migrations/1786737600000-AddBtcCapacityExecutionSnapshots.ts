import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddBtcCapacityExecutionSnapshots1786737600000
  implements MigrationInterface
{
  name = 'AddBtcCapacityExecutionSnapshots1786737600000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE polymarket.btc_market_capacity_execution_snapshots (
        LIKE polymarket.btc_market_execution_snapshots
          INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING STORAGE INCLUDING COMMENTS
      );

      ALTER TABLE polymarket.btc_market_capacity_execution_snapshots
        ADD COLUMN up_ask_vwap_15 numeric(18,8),
        ADD COLUMN up_ask_vwap_20 numeric(18,8),
        ADD COLUMN down_ask_vwap_15 numeric(18,8),
        ADD COLUMN down_ask_vwap_20 numeric(18,8),
        ADD CONSTRAINT pk_btc_market_capacity_execution_snapshots
          PRIMARY KEY (market_id, sampled_at),
        ADD CONSTRAINT fk_btc_market_capacity_execution_snapshots_market
          FOREIGN KEY (market_id)
          REFERENCES polymarket.btc_interval_markets (market_id) ON DELETE RESTRICT,
        ADD CONSTRAINT fk_btc_market_capacity_execution_snapshots_artifact
          FOREIGN KEY (artifact_id)
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        ADD CONSTRAINT chk_btc_market_capacity_execution_snapshot_schema
          CHECK (length(btrim(schema_version)) > 0),
        ADD CONSTRAINT chk_btc_market_capacity_execution_snapshot_prices CHECK (
          (up_ask_vwap_15 IS NULL OR up_ask_vwap_15 BETWEEN 0 AND 1)
          AND (up_ask_vwap_20 IS NULL OR up_ask_vwap_20 BETWEEN 0 AND 1)
          AND (down_ask_vwap_15 IS NULL OR down_ask_vwap_15 BETWEEN 0 AND 1)
          AND (down_ask_vwap_20 IS NULL OR down_ask_vwap_20 BETWEEN 0 AND 1)
        ),
        ADD CONSTRAINT chk_btc_market_capacity_execution_snapshot_vwap CHECK (
          (up_ask_vwap_15 IS NULL OR up_ask_vwap_10 IS NULL
            OR up_ask_vwap_15 >= up_ask_vwap_10)
          AND (up_ask_vwap_20 IS NULL OR up_ask_vwap_15 IS NULL
            OR up_ask_vwap_20 >= up_ask_vwap_15)
          AND (down_ask_vwap_15 IS NULL OR down_ask_vwap_10 IS NULL
            OR down_ask_vwap_15 >= down_ask_vwap_10)
          AND (down_ask_vwap_20 IS NULL OR down_ask_vwap_15 IS NULL
            OR down_ask_vwap_20 >= down_ask_vwap_15)
        );

      SELECT create_hypertable(
        'polymarket.btc_market_capacity_execution_snapshots',
        'sampled_at',
        chunk_time_interval => INTERVAL '1 day',
        create_default_indexes => FALSE
      );

      ALTER TABLE polymarket.btc_market_capacity_execution_snapshots SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'sampled_at ASC',
        timescaledb.compress_segmentby = 'market_id, artifact_id'
      );

      CREATE INDEX idx_btc_market_capacity_execution_snapshots_artifact
        ON polymarket.btc_market_capacity_execution_snapshots (artifact_id, sampled_at);

      SELECT add_compression_policy(
        'polymarket.btc_market_capacity_execution_snapshots',
        INTERVAL '7 days',
        if_not_exists => true
      );

      CREATE TRIGGER trg_reject_btc_market_capacity_execution_snapshot_change
        BEFORE UPDATE OR DELETE
        ON polymarket.btc_market_capacity_execution_snapshots
        FOR EACH ROW
        EXECUTE FUNCTION polymarket.reject_btc_market_execution_snapshot_change();
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
          FROM polymarket.btc_market_capacity_execution_snapshots
          LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to remove BTC capacity execution snapshots while facts exist';
        END IF;
      END $$;

      DROP TABLE polymarket.btc_market_capacity_execution_snapshots;
    `);
  }
}
