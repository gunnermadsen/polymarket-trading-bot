import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddBtcHistoricalReferenceAndOrderbook1784913000000 implements MigrationInterface {
  name = 'AddBtcHistoricalReferenceAndOrderbook1784913000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE polymarket.btc_orderbook_archive_events (
        artifact_id uuid NOT NULL
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        source_row_number bigint NOT NULL,
        provider_received_at timestamptz NOT NULL,
        source_timestamp timestamptz NOT NULL,
        condition_id text NOT NULL,
        asset_id text NOT NULL,
        event_type text NOT NULL,
        bids jsonb,
        asks jsonb,
        price numeric(18,8),
        size numeric(30,10),
        side text,
        best_bid numeric(18,8),
        best_ask numeric(18,8),
        fee_rate_bps integer,
        transaction_hash text,
        old_tick_size numeric(18,8),
        new_tick_size numeric(18,8),
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_btc_orderbook_archive_events
          PRIMARY KEY (artifact_id, source_row_number, provider_received_at),
        CONSTRAINT chk_btc_orderbook_archive_source_row
          CHECK (source_row_number >= 0),
        CONSTRAINT chk_btc_orderbook_archive_identity CHECK (
          condition_id ~ '^0x[0-9a-fA-F]{64}$'
          AND asset_id ~ '^[0-9]+$'
        ),
        CONSTRAINT chk_btc_orderbook_archive_event_type CHECK (
          event_type IN ('book','price_change','last_trade_price','tick_size_change')
        ),
        CONSTRAINT chk_btc_orderbook_archive_prices CHECK (
          (price IS NULL OR (price >= 0 AND price <= 1))
          AND (best_bid IS NULL OR (best_bid >= 0 AND best_bid <= 1))
          AND (best_ask IS NULL OR (best_ask >= 0 AND best_ask <= 1))
          AND (old_tick_size IS NULL OR (old_tick_size > 0 AND old_tick_size <= 1))
          AND (new_tick_size IS NULL OR (new_tick_size > 0 AND new_tick_size <= 1))
          AND (size IS NULL OR size >= 0)
          AND (fee_rate_bps IS NULL OR fee_rate_bps >= 0)
          AND (side IS NULL OR side IN ('buy','sell'))
        ),
        CONSTRAINT chk_btc_orderbook_archive_book CHECK (
          event_type <> 'book'
          OR (
            jsonb_typeof(bids) = 'array'
            AND jsonb_typeof(asks) = 'array'
          )
        )
      );

      SELECT create_hypertable(
        'polymarket.btc_orderbook_archive_events',
        'provider_received_at',
        chunk_time_interval => INTERVAL '1 day',
        if_not_exists => TRUE
      );

      ALTER TABLE polymarket.btc_orderbook_archive_events SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'provider_received_at ASC, source_row_number ASC',
        timescaledb.compress_segmentby = 'condition_id, asset_id, artifact_id'
      );

      CREATE INDEX idx_btc_orderbook_archive_market_time
        ON polymarket.btc_orderbook_archive_events (
          condition_id, asset_id, source_timestamp, provider_received_at
        );

      CREATE INDEX idx_btc_orderbook_archive_artifact
        ON polymarket.btc_orderbook_archive_events (
          artifact_id, provider_received_at, source_row_number
        );
    `);

    await queryRunner.query(`
      CREATE TABLE polymarket.chainlink_btcusd_archive_ticks (
        feed_id text NOT NULL,
        source_timestamp timestamptz NOT NULL,
        valid_from_timestamp timestamptz NOT NULL,
        price numeric(38,18) NOT NULL,
        bid numeric(38,18) NOT NULL,
        ask numeric(38,18) NOT NULL,
        report_sha256 text NOT NULL,
        artifact_id uuid NOT NULL
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        ingested_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT pk_chainlink_btcusd_archive_ticks
          PRIMARY KEY (feed_id, source_timestamp),
        CONSTRAINT chk_chainlink_btcusd_archive_feed CHECK (
          feed_id ~ '^0x[0-9a-f]{64}$'
        ),
        CONSTRAINT chk_chainlink_btcusd_archive_time CHECK (
          valid_from_timestamp <= source_timestamp
        ),
        CONSTRAINT chk_chainlink_btcusd_archive_prices CHECK (
          price > 0 AND bid > 0 AND ask > 0
          AND bid <= price AND price <= ask
        ),
        CONSTRAINT chk_chainlink_btcusd_archive_hash CHECK (
          report_sha256 ~ '^[0-9a-f]{64}$'
        )
      );

      SELECT create_hypertable(
        'polymarket.chainlink_btcusd_archive_ticks',
        'source_timestamp',
        chunk_time_interval => INTERVAL '1 day',
        if_not_exists => TRUE
      );

      ALTER TABLE polymarket.chainlink_btcusd_archive_ticks SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'source_timestamp ASC',
        timescaledb.compress_segmentby = 'feed_id, artifact_id'
      );

      CREATE INDEX idx_chainlink_btcusd_archive_artifact
        ON polymarket.chainlink_btcusd_archive_ticks (artifact_id, source_timestamp);
    `);

    await queryRunner.query(`
      CREATE FUNCTION polymarket.reject_historical_market_event_change()
      RETURNS trigger
      LANGUAGE plpgsql
      AS $$
      BEGIN
        RAISE EXCEPTION 'historical market source rows are immutable';
      END;
      $$;

      CREATE TRIGGER trg_reject_btc_orderbook_archive_event_change
        BEFORE UPDATE OR DELETE ON polymarket.btc_orderbook_archive_events
        FOR EACH ROW
        EXECUTE FUNCTION polymarket.reject_historical_market_event_change();

      CREATE TRIGGER trg_reject_chainlink_btcusd_archive_tick_change
        BEFORE UPDATE OR DELETE ON polymarket.chainlink_btcusd_archive_ticks
        FOR EACH ROW
        EXECUTE FUNCTION polymarket.reject_historical_market_event_change();
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (SELECT 1 FROM polymarket.btc_orderbook_archive_events LIMIT 1)
          OR EXISTS (SELECT 1 FROM polymarket.chainlink_btcusd_archive_ticks LIMIT 1)
        THEN
          RAISE EXCEPTION
            'refusing to remove historical orderbook or Chainlink data while rows exist';
        END IF;
      END $$;

      DROP TRIGGER trg_reject_chainlink_btcusd_archive_tick_change
        ON polymarket.chainlink_btcusd_archive_ticks;
      DROP TRIGGER trg_reject_btc_orderbook_archive_event_change
        ON polymarket.btc_orderbook_archive_events;
      DROP FUNCTION polymarket.reject_historical_market_event_change();

      DROP TABLE polymarket.chainlink_btcusd_archive_ticks;
      DROP TABLE polymarket.btc_orderbook_archive_events;
    `);
  }
}
