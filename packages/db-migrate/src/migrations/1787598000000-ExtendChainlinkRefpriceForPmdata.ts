import { MigrationInterface, QueryRunner } from 'typeorm';

const TABLE_NAME = 'market_data.chainlink_btcusd_reference_prices';

export class ExtendChainlinkRefpriceForPmdata1787598000000
  implements MigrationInterface
{
  name = 'ExtendChainlinkRefpriceForPmdata1787598000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      SET LOCAL lock_timeout = '30s';
      LOCK TABLE ${TABLE_NAME} IN ACCESS EXCLUSIVE MODE;
      SELECT remove_compression_policy('${TABLE_NAME}', if_exists => TRUE);
      SELECT decompress_chunk(format('%I.%I', chunk_schema, chunk_name)::regclass)
      FROM timescaledb_information.chunks
      WHERE hypertable_schema = 'market_data'
        AND hypertable_name = 'chainlink_btcusd_reference_prices'
        AND is_compressed;
      ALTER TABLE ${TABLE_NAME} SET (timescaledb.compress = false);

      ALTER TABLE ${TABLE_NAME}
        ADD COLUMN expires_at timestamptz,
        ADD COLUMN report_version text,
        ADD COLUMN source_date date,
        ADD COLUMN archive_row_number bigint,
        ADD COLUMN backfill_artifact_id uuid,
        ADD COLUMN report_hash_kind text NOT NULL DEFAULT 'signed_report',
        ALTER COLUMN valid_from_timestamp DROP NOT NULL,
        ALTER COLUMN bid DROP NOT NULL,
        ALTER COLUMN ask DROP NOT NULL,
        ALTER COLUMN capture_artifact_id DROP NOT NULL;

      ALTER TABLE ${TABLE_NAME}
        DROP CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_identity,
        DROP CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_time,
        DROP CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_values,
        DROP CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_hashes;

      ALTER TABLE ${TABLE_NAME}
        ADD CONSTRAINT fk_market_data_chainlink_btcusd_reference_prices_backfill_artifact
          FOREIGN KEY (backfill_artifact_id)
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        ADD CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_identity CHECK (
          feed_id = '0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8'
          AND strategy_key = 'chainlink_btcusd_reference_price'
          AND source IN ('chainlink_data_streams', 'pmdata_chainlink_streams')
          AND ((source = 'chainlink_data_streams' AND capture_artifact_id IS NOT NULL
                AND backfill_artifact_id IS NULL AND report_hash_kind = 'signed_report')
            OR (source = 'pmdata_chainlink_streams' AND capture_artifact_id IS NULL
                AND backfill_artifact_id IS NOT NULL AND source_date IS NOT NULL
                AND archive_row_number >= 0 AND report_hash_kind = 'canonical_archive_row'))
        ),
        ADD CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_time CHECK (
          (valid_from_timestamp IS NULL OR valid_from_timestamp <= source_timestamp)
          AND (expires_at IS NULL OR expires_at > source_timestamp)
          AND (source_date IS NULL OR (source_timestamp >= source_date::timestamptz
            AND source_timestamp < source_date::timestamptz + INTERVAL '1 day'))
        ),
        ADD CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_values CHECK (
          price > 0 AND ((bid IS NULL AND ask IS NULL)
            OR (bid > 0 AND bid <= price AND price <= ask))
        ),
        ADD CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_hashes CHECK (
          report_sha256 ~ '^[0-9a-f]{64}$'
          AND payload_sha256 ~ '^[0-9a-f]{64}$'
          AND (report_version IS NULL OR length(btrim(report_version)) > 0)
        );

      CREATE INDEX idx_market_data_chainlink_btcusd_reference_prices_backfill_artifact
        ON ${TABLE_NAME} (backfill_artifact_id, archive_row_number)
        WHERE backfill_artifact_id IS NOT NULL;

      ALTER TABLE ${TABLE_NAME} SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'source_timestamp ASC, report_sha256 ASC',
        timescaledb.compress_segmentby =
          'feed_id, source, strategy_key, capture_artifact_id, backfill_artifact_id'
      );
      SELECT add_compression_policy(
        '${TABLE_NAME}', INTERVAL '2 days', if_not_exists => TRUE
      );
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        IF EXISTS (SELECT 1 FROM ${TABLE_NAME}
          WHERE source = 'pmdata_chainlink_streams' LIMIT 1) THEN
          RAISE EXCEPTION 'refusing to remove PMData RefPrice compatibility while facts exist';
        END IF;
      END $$;
      SET LOCAL lock_timeout = '30s';
      LOCK TABLE ${TABLE_NAME} IN ACCESS EXCLUSIVE MODE;
      SELECT remove_compression_policy('${TABLE_NAME}', if_exists => TRUE);
      SELECT decompress_chunk(format('%I.%I', chunk_schema, chunk_name)::regclass)
      FROM timescaledb_information.chunks
      WHERE hypertable_schema = 'market_data'
        AND hypertable_name = 'chainlink_btcusd_reference_prices'
        AND is_compressed;
      ALTER TABLE ${TABLE_NAME} SET (timescaledb.compress = false);
      DROP INDEX idx_market_data_chainlink_btcusd_reference_prices_backfill_artifact;
      ALTER TABLE ${TABLE_NAME}
        DROP CONSTRAINT fk_market_data_chainlink_btcusd_reference_prices_backfill_artifact,
        DROP CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_identity,
        DROP CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_time,
        DROP CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_values,
        DROP CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_hashes,
        DROP COLUMN backfill_artifact_id,
        DROP COLUMN archive_row_number,
        DROP COLUMN source_date,
        DROP COLUMN report_version,
        DROP COLUMN expires_at,
        DROP COLUMN report_hash_kind,
        ALTER COLUMN valid_from_timestamp SET NOT NULL,
        ALTER COLUMN bid SET NOT NULL,
        ALTER COLUMN ask SET NOT NULL,
        ALTER COLUMN capture_artifact_id SET NOT NULL;
      ALTER TABLE ${TABLE_NAME}
        ADD CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_identity CHECK (
          source = 'chainlink_data_streams'
          AND strategy_key = 'chainlink_btcusd_reference_price'
          AND feed_id = '0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8'
          AND feed_id ~ '^0x[0-9a-f]{64}$'
        ),
        ADD CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_time CHECK (
          valid_from_timestamp <= source_timestamp
        ),
        ADD CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_values CHECK (
          price > 0 AND bid > 0 AND ask > 0 AND bid <= price AND price <= ask
        ),
        ADD CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_hashes CHECK (
          report_sha256 ~ '^[0-9a-f]{64}$' AND payload_sha256 ~ '^[0-9a-f]{64}$'
        );
      ALTER TABLE ${TABLE_NAME} SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'source_timestamp ASC, report_sha256 ASC',
        timescaledb.compress_segmentby =
          'feed_id, source, strategy_key, capture_artifact_id'
      );
      SELECT add_compression_policy(
        '${TABLE_NAME}', INTERVAL '2 days', if_not_exists => TRUE
      );
    `);
  }
}
