import { MigrationInterface, QueryRunner } from 'typeorm';

const DIRECT = 'market_data.chainlink_btcusd_reference_prices';
const PMDATA = 'market_data.pmdata_chainlink_btcusd_reference_prices';
const LEGACY = 'polymarket.chainlink_btcusd_archive_ticks';
const DIRECT_WATERMARK = '2026-09-01T19:02:06.000Z';
const LEGACY_WATERMARK = '2026-08-29T23:59:59.000Z';
const FEED_ID =
  '0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8';

export class CanonicalizeChainlinkReferencePriceStorage1788647000000
  implements MigrationInterface
{
  transaction = false;

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET lock_timeout = '5s'`);
    await queryRunner.query(`SET statement_timeout = '5min'`);

    const unsafeProfiles = await queryRunner.query(`
      SELECT strategy_key
      FROM ingester.profiles
      WHERE strategy_key = 'chainlink_btcusd_reference_price'
        AND (desired_state <> 'stopped' OR lease_owner IS NOT NULL)
    `);
    if (unsafeProfiles.length !== 0) {
      throw new Error('Chainlink realtime strategy must be stopped before storage cutover');
    }
    const activeJobs = await queryRunner.query(`
      SELECT job_id
      FROM ingester.backfill_jobs
      WHERE strategy_key IN (
        'chainlink_btcusd_reference_ticks_backfill',
        'pmdata_chainlink_btcusd_refprice_backfill'
      )
        AND status IN ('queued', 'running', 'retrying', 'cancel_requested')
      LIMIT 1
    `);
    if (activeJobs.length !== 0) {
      throw new Error('Chainlink backfill jobs must be terminal before storage cutover');
    }

    await this.requireArchivedWatermark(queryRunner, DIRECT, DIRECT_WATERMARK);
    await this.requireArchivedWatermark(queryRunner, LEGACY, LEGACY_WATERMARK);
    await this.dropHypertable(queryRunner, DIRECT);
    await this.dropHypertable(queryRunner, LEGACY);
    await this.createDirectTable(queryRunner);
    await this.createPmdataTable(queryRunner);

    const finalRelations = await queryRunner.query(
      `SELECT to_regclass($1) AS direct,
              to_regclass($2) AS pmdata,
              to_regclass($3) AS legacy`,
      [DIRECT, PMDATA, LEGACY],
    );
    if (
      !finalRelations[0]?.direct ||
      !finalRelations[0]?.pmdata ||
      finalRelations[0]?.legacy
    ) {
      throw new Error('Chainlink reference-price storage cutover did not complete');
    }
  }

  private async requireArchivedWatermark(
    queryRunner: QueryRunner,
    relation: string,
    watermark: string,
  ): Promise<void> {
    const exists = await queryRunner.query(`SELECT to_regclass($1) AS relation`, [relation]);
    if (!exists[0]?.relation) return;
    const unarchived = await queryRunner.query(
      `SELECT 1 FROM ${relation} WHERE source_timestamp > $1::timestamptz LIMIT 1`,
      [watermark],
    );
    if (unarchived.length !== 0) {
      throw new Error(`${relation} advanced beyond archived watermark ${watermark}`);
    }
  }

  private async dropHypertable(queryRunner: QueryRunner, relation: string): Promise<void> {
    const exists = await queryRunner.query(`SELECT to_regclass($1) AS relation`, [relation]);
    if (!exists[0]?.relation) return;
    const [schema, table] = relation.split('.');
    await queryRunner.query(`SELECT remove_compression_policy($1::regclass, if_exists => true)`, [
      relation,
    ]);
    const cutoffs = await queryRunner.query(
      `WITH ordered AS (
         SELECT range_end,
                row_number() OVER (ORDER BY range_end) AS chunk_number,
                count(*) OVER () AS chunk_count
         FROM timescaledb_information.chunks
         WHERE hypertable_schema = $1 AND hypertable_name = $2
       )
       SELECT range_end FROM ordered
       WHERE chunk_number % 8 = 0 OR chunk_number = chunk_count
       ORDER BY range_end`,
      [schema, table],
    );
    for (const cutoff of cutoffs) {
      await queryRunner.query(
        `SELECT drop_chunks($1::regclass, older_than => $2::timestamptz)`,
        [relation, cutoff.range_end],
      );
    }
    const remaining = await queryRunner.query(
      `SELECT count(*)::integer AS count
       FROM timescaledb_information.chunks
       WHERE hypertable_schema = $1 AND hypertable_name = $2`,
      [schema, table],
    );
    if (remaining[0]?.count !== 0) {
      throw new Error(`${relation} chunks were not fully removed`);
    }
    await queryRunner.query(`DROP TABLE ${relation}`);
  }

  private async createDirectTable(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(this.createTableSql(DIRECT, 'direct'));
  }

  private async createPmdataTable(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(this.createTableSql(PMDATA, 'pmdata'));
  }

  private createTableSql(relation: string, product: 'direct' | 'pmdata'): string {
    const suffix = product === 'direct' ? 'chainlink_btcusd_reference_prices' : 'pmdata_chainlink_btcusd_reference_prices';
    const identity = product === 'direct'
      ? `(source = 'chainlink_data_streams' AND feed_id = '${FEED_ID}'
          AND report_hash_kind = 'signed_report'
          AND ((strategy_key = 'chainlink_btcusd_reference_price'
                AND capture_artifact_id IS NOT NULL AND backfill_artifact_id IS NULL)
            OR (strategy_key = 'chainlink_btcusd_reference_ticks_backfill'
                AND capture_artifact_id IS NULL AND backfill_artifact_id IS NOT NULL)))`
      : `(source = 'pmdata_chainlink_streams' AND feed_id = '${FEED_ID}'
          AND strategy_key = 'pmdata_chainlink_btcusd_refprice_backfill'
          AND report_hash_kind = 'canonical_archive_row'
          AND capture_artifact_id IS NULL AND backfill_artifact_id IS NOT NULL
          AND source_date IS NOT NULL AND archive_row_number >= 0)`;
    return `
      CREATE TABLE IF NOT EXISTS ${relation} (
        source text NOT NULL,
        feed_id text NOT NULL,
        source_timestamp timestamptz NOT NULL,
        valid_from_timestamp timestamptz,
        provider_available_at timestamptz,
        received_at timestamptz NOT NULL,
        price numeric(38,18) NOT NULL,
        bid numeric(38,18),
        ask numeric(38,18),
        report_sha256 character(64) NOT NULL,
        payload_sha256 character(64) NOT NULL,
        strategy_key text NOT NULL,
        capture_artifact_id uuid,
        ingested_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        expires_at timestamptz,
        report_version text,
        source_date date,
        archive_row_number bigint,
        backfill_artifact_id uuid,
        report_hash_kind text NOT NULL,
        CONSTRAINT pk_market_data_${suffix}
          PRIMARY KEY (feed_id, source_timestamp, report_sha256),
        CONSTRAINT fk_market_data_${suffix}_capture_artifact
          FOREIGN KEY (strategy_key, capture_artifact_id)
          REFERENCES ingester.capture_artifacts (strategy_key, artifact_id) ON DELETE RESTRICT,
        CONSTRAINT fk_market_data_${suffix}_backfill_artifact
          FOREIGN KEY (backfill_artifact_id)
          REFERENCES ingester.backfill_artifacts (artifact_id) ON DELETE RESTRICT,
        CONSTRAINT chk_market_data_${suffix}_identity CHECK (${identity}),
        CONSTRAINT chk_market_data_${suffix}_time CHECK (
          (valid_from_timestamp IS NULL OR valid_from_timestamp <= source_timestamp)
          AND (expires_at IS NULL OR expires_at > source_timestamp)
          AND (source_date IS NULL OR (source_timestamp >= source_date::timestamptz
            AND source_timestamp < source_date::timestamptz + INTERVAL '1 day'))
        ),
        CONSTRAINT chk_market_data_${suffix}_values CHECK (
          price > 0 AND ((bid IS NULL AND ask IS NULL)
            OR (bid > 0 AND bid <= price AND price <= ask))
        ),
        CONSTRAINT chk_market_data_${suffix}_hashes CHECK (
          report_sha256 ~ '^[0-9a-f]{64}$'
          AND payload_sha256 ~ '^[0-9a-f]{64}$'
          AND (report_version IS NULL OR length(btrim(report_version)) > 0)
        )
      );
      SELECT create_hypertable(
        '${relation}', 'source_timestamp', chunk_time_interval => INTERVAL '1 day',
        create_default_indexes => FALSE, if_not_exists => TRUE
      );
      ALTER TABLE ${relation} SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'source_timestamp ASC, report_sha256 ASC',
        timescaledb.compress_segmentby =
          'feed_id, source, strategy_key, capture_artifact_id, backfill_artifact_id'
      );
      SELECT add_compression_policy('${relation}', INTERVAL '2 days', if_not_exists => TRUE);
      CREATE INDEX IF NOT EXISTS idx_market_data_${suffix}_recovery
        ON ${relation} (feed_id, source_timestamp DESC);
      CREATE INDEX IF NOT EXISTS idx_market_data_${suffix}_capture_artifact
        ON ${relation} (capture_artifact_id, source_timestamp DESC)
        WHERE capture_artifact_id IS NOT NULL;
      CREATE INDEX IF NOT EXISTS idx_market_data_${suffix}_backfill_artifact
        ON ${relation} (backfill_artifact_id, archive_row_number)
        WHERE backfill_artifact_id IS NOT NULL;
      DROP TRIGGER IF EXISTS trg_reject_market_data_${suffix}_change ON ${relation};
      CREATE TRIGGER trg_reject_market_data_${suffix}_change
        BEFORE UPDATE OR DELETE ON ${relation}
        FOR EACH ROW EXECUTE FUNCTION market_data.reject_source_fact_change();
    `;
  }

  public async down(): Promise<void> {
    throw new Error('irreversible: Chainlink history is preserved in canonical Parquet');
  }
}
