import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddPolygonChainlinkBtcusdOracleRounds1785359400000 implements MigrationInterface {
  name = 'AddPolygonChainlinkBtcusdOracleRounds1785359400000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE polymarket.polygon_chainlink_btcusd_oracle_rounds (
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
        artifact_id uuid NOT NULL
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_polygon_chainlink_btcusd_oracle_rounds
          PRIMARY KEY (
            feed_proxy_address, source_timestamp, transaction_hash, log_index
          ),
        CONSTRAINT uq_polygon_chainlink_btcusd_oracle_phase_round
          UNIQUE (
            feed_proxy_address, source_timestamp, phase_id, aggregator_round_id
          ),
        CONSTRAINT chk_polygon_chainlink_btcusd_oracle_chain
          CHECK (chain_id = 137),
        CONSTRAINT chk_polygon_chainlink_btcusd_oracle_addresses CHECK (
          feed_proxy_address ~ '^0x[0-9a-f]{40}$'
          AND aggregator_address ~ '^0x[0-9a-f]{40}$'
        ),
        CONSTRAINT chk_polygon_chainlink_btcusd_oracle_round CHECK (
          phase_id > 0 AND aggregator_round_id > 0
        ),
        CONSTRAINT chk_polygon_chainlink_btcusd_oracle_time CHECK (
          source_timestamp <= block_timestamp
        ),
        CONSTRAINT chk_polygon_chainlink_btcusd_oracle_price CHECK (
          answer_raw > 0
          AND price > 0
          AND decimals BETWEEN 0 AND 18
          AND price = answer_raw / power(10::numeric, decimals)
        ),
        CONSTRAINT chk_polygon_chainlink_btcusd_oracle_block CHECK (
          block_number > 0
          AND block_hash ~ '^0x[0-9a-f]{64}$'
          AND transaction_hash ~ '^0x[0-9a-f]{64}$'
          AND log_index >= 0
        )
      );

      SELECT create_hypertable(
        'polymarket.polygon_chainlink_btcusd_oracle_rounds',
        'source_timestamp',
        chunk_time_interval => INTERVAL '1 day',
        if_not_exists => TRUE
      );

      ALTER TABLE polymarket.polygon_chainlink_btcusd_oracle_rounds SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby =
          'source_timestamp ASC, block_number ASC, log_index ASC',
        timescaledb.compress_segmentby =
          'feed_proxy_address, aggregator_address, phase_id, artifact_id'
      );

      CREATE INDEX idx_polygon_chainlink_btcusd_oracle_time
        ON polymarket.polygon_chainlink_btcusd_oracle_rounds (
          feed_proxy_address, source_timestamp
        );

      CREATE INDEX idx_polygon_chainlink_btcusd_oracle_artifact
        ON polymarket.polygon_chainlink_btcusd_oracle_rounds (
          artifact_id, source_timestamp
        );

      CREATE TRIGGER trg_reject_polygon_chainlink_btcusd_oracle_change
        BEFORE UPDATE OR DELETE
        ON polymarket.polygon_chainlink_btcusd_oracle_rounds
        FOR EACH ROW
        EXECUTE FUNCTION polymarket.reject_historical_market_event_change();
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (
          SELECT 1
          FROM polymarket.polygon_chainlink_btcusd_oracle_rounds
          LIMIT 1
        ) THEN
          RAISE EXCEPTION
            'refusing to remove Polygon Chainlink BTC/USD oracle data while rows exist';
        END IF;
      END $$;

      DROP TRIGGER trg_reject_polygon_chainlink_btcusd_oracle_change
        ON polymarket.polygon_chainlink_btcusd_oracle_rounds;
      DROP TABLE polymarket.polygon_chainlink_btcusd_oracle_rounds;
    `);
  }
}
