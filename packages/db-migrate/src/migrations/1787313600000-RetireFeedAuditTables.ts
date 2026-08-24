import { MigrationInterface, QueryRunner } from 'typeorm';

export class RetireFeedAuditTables1787313600000
  implements MigrationInterface
{
  name = 'RetireFeedAuditTables1787313600000';

  /**
   * Intentionally deletes approximately 14 million market feed events (16 GB)
   * and 26,000 feed sessions (18 MB). Dropping the market-feed hypertable also
   * removes its TimescaleDB chunks, compression settings, indexes, and policies.
   * The down migration can recreate empty definitions only; deleted audit data
   * cannot be restored by rollback.
   */
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      SET LOCAL lock_timeout = '10s';

      DO $$
      BEGIN
        IF to_regclass('polymarket.market_feed_events') IS NULL THEN
          RAISE EXCEPTION 'polymarket.market_feed_events does not exist';
        END IF;
        IF to_regclass('polymarket.feed_sessions') IS NULL THEN
          RAISE EXCEPTION 'polymarket.feed_sessions does not exist';
        END IF;
      END $$;

      DROP TABLE polymarket.market_feed_events;
      DROP TABLE polymarket.feed_sessions;
    `);
  }

  /** Recreates empty table definitions; it cannot restore deleted audit data. */
  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE polymarket.market_feed_events (
        feed_event_id uuid NOT NULL,
        source_timestamp timestamptz NOT NULL,
        received_at timestamptz NOT NULL,
        persisted_at timestamptz NOT NULL DEFAULT now(),
        connection_id uuid NOT NULL,
        ingest_sequence bigint NOT NULL,
        market_id text,
        token_id text,
        event_type text NOT NULL,
        source_hash text,
        applied boolean NOT NULL DEFAULT false,
        integrity_status text NOT NULL,
        raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
        CONSTRAINT pk_market_feed_events
          PRIMARY KEY (feed_event_id, source_timestamp)
      );

      CREATE INDEX idx_feed_events_market_ts
        ON polymarket.market_feed_events (market_id, source_timestamp DESC);
      CREATE INDEX idx_feed_events_integrity_ts
        ON polymarket.market_feed_events (
          integrity_status, source_timestamp DESC
        );

      SELECT create_hypertable(
        'polymarket.market_feed_events', 'source_timestamp',
        chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE
      );

      ALTER TABLE polymarket.market_feed_events SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby =
          'source_timestamp DESC, feed_event_id',
        timescaledb.compress_segmentby = 'event_type'
      );

      SELECT add_retention_policy(
        'polymarket.market_feed_events', INTERVAL '14 days',
        if_not_exists => TRUE
      );
      SELECT add_compression_policy(
        'polymarket.market_feed_events', INTERVAL '1 day',
        if_not_exists => TRUE
      );

      CREATE TABLE polymarket.feed_sessions (
        connection_id uuid PRIMARY KEY,
        feed_name text NOT NULL,
        endpoint text NOT NULL,
        reconnect_ordinal integer NOT NULL,
        started_at timestamptz NOT NULL,
        connected_at timestamptz,
        disconnected_at timestamptz,
        messages_received bigint NOT NULL DEFAULT 0,
        messages_persisted bigint NOT NULL DEFAULT 0,
        decode_errors bigint NOT NULL DEFAULT 0,
        integrity_gaps bigint NOT NULL DEFAULT 0,
        dropped_messages bigint NOT NULL DEFAULT 0,
        disconnect_reason text,
        metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now()
      );
    `);
  }
}
