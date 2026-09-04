import { MigrationInterface, QueryRunner } from 'typeorm';

const LEGACY = 'polymarket.chainlink_btcusd_archive_ticks';
const TARGET = 'market_data.chainlink_btcusd_reference_prices';
const HISTORICAL_STRATEGY = 'chainlink_btcusd_reference_ticks';
const BACKFILL_STRATEGY = 'chainlink_btcusd_reference_ticks_backfill';

export class ConsolidateChainlinkReferencePrices1788645700000
  implements MigrationInterface
{
  name = 'ConsolidateChainlinkReferencePrices1788645700000';
  transaction = false;

  public async up(queryRunner: QueryRunner): Promise<void> {
    if (!(await this.tableExists(queryRunner, LEGACY))) return;
    await queryRunner.query(
      'SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0',
    );
    try {
      await this.alignCanonicalConstraint(queryRunner);
      const bounds = await queryRunner.query(`
        SELECT
          (SELECT source_timestamp FROM ${LEGACY}
           ORDER BY source_timestamp ASC LIMIT 1) AS minimum,
          (SELECT source_timestamp FROM ${LEGACY}
           ORDER BY source_timestamp DESC LIMIT 1) AS maximum
      `);
      if (!bounds[0]?.minimum || !bounds[0]?.maximum) {
        await this.replaceLegacyWithCompatibilityView(queryRunner);
        return;
      }

      let legacyCount = 0n;
      let canonicalCount = 0n;
      for (const [start, end] of this.hourRanges(
        new Date(bounds[0].minimum),
        new Date(bounds[0].maximum),
      )) {
        const conflict = await queryRunner.query(`
          SELECT EXISTS (
            SELECT 1 FROM ${LEGACY} legacy
            JOIN ${TARGET} canonical
              ON canonical.feed_id = legacy.feed_id
             AND canonical.source_timestamp = legacy.source_timestamp
            WHERE legacy.source_timestamp >= $1
              AND legacy.source_timestamp < $2
              AND (canonical.price IS DISTINCT FROM legacy.price
                OR canonical.bid IS DISTINCT FROM legacy.bid
                OR canonical.ask IS DISTINCT FROM legacy.ask
                OR canonical.valid_from_timestamp
                  IS DISTINCT FROM legacy.valid_from_timestamp)
            LIMIT 1
          ) AS differs
        `, [start, end]);
        if (conflict[0]?.differs) {
          throw new Error(
            `${TARGET} conflicts with legacy facts in ${start.toISOString()}`,
          );
        }

        const accounting = await queryRunner.query(`
          WITH source_rows AS MATERIALIZED (
            SELECT * FROM ${LEGACY}
            WHERE source_timestamp >= $1 AND source_timestamp < $2
          ), inserted AS (
            INSERT INTO ${TARGET} (
              source, feed_id, source_timestamp, valid_from_timestamp,
              provider_available_at, received_at, price, bid, ask,
              report_sha256, payload_sha256, strategy_key,
              capture_artifact_id, ingested_at, backfill_artifact_id,
              report_hash_kind
            )
            SELECT 'chainlink_data_streams', feed_id, source_timestamp,
              valid_from_timestamp, source_timestamp, ingested_at, price, bid,
              ask, report_sha256, report_sha256, '${HISTORICAL_STRATEGY}',
              NULL, ingested_at, artifact_id, 'signed_report'
            FROM source_rows
            ON CONFLICT (feed_id, source_timestamp, report_sha256) DO NOTHING
            RETURNING 1
          )
          SELECT
            (SELECT count(*)::bigint FROM source_rows) AS legacy_count,
            (SELECT count(*)::bigint FROM ${TARGET} canonical
             JOIN source_rows legacy
               ON canonical.feed_id = legacy.feed_id
              AND canonical.source_timestamp = legacy.source_timestamp
              AND canonical.report_sha256 = legacy.report_sha256) AS canonical_count,
            (SELECT count(*)::bigint FROM inserted) AS inserted_count
        `, [start, end]);
        legacyCount += BigInt(accounting[0].legacy_count);
        canonicalCount += BigInt(accounting[0].canonical_count);
        await this.pause();
      }
      if (legacyCount !== canonicalCount) {
        throw new Error(
          `${LEGACY} accounting differs: ${legacyCount} legacy, ${canonicalCount} canonical`,
        );
      }
      await this.replaceLegacyWithCompatibilityView(queryRunner);
    } finally {
      await queryRunner.query(
        'RESET timescaledb.max_tuples_decompressed_per_dml_transaction',
      );
    }
  }

  private async alignCanonicalConstraint(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ${TARGET}
        DROP CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_identity;
      ALTER TABLE ${TARGET}
        ADD CONSTRAINT chk_market_data_chainlink_btcusd_reference_prices_identity
        CHECK (
          feed_id = '0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8'
          AND source IN ('chainlink_data_streams', 'pmdata_chainlink_streams')
          AND (
            (source = 'chainlink_data_streams'
              AND strategy_key = 'chainlink_btcusd_reference_price'
              AND capture_artifact_id IS NOT NULL
              AND backfill_artifact_id IS NULL
              AND report_hash_kind = 'signed_report')
            OR
            (source = 'chainlink_data_streams'
              AND strategy_key IN (
                '${HISTORICAL_STRATEGY}', '${BACKFILL_STRATEGY}'
              )
              AND capture_artifact_id IS NULL
              AND backfill_artifact_id IS NOT NULL
              AND report_hash_kind = 'signed_report')
            OR
            (source = 'pmdata_chainlink_streams'
              AND strategy_key = 'chainlink_btcusd_reference_price'
              AND capture_artifact_id IS NULL
              AND backfill_artifact_id IS NOT NULL
              AND source_date IS NOT NULL
              AND archive_row_number >= 0
              AND report_hash_kind = 'canonical_archive_row')
          )
        ) NOT VALID;
    `);
  }

  private async replaceLegacyWithCompatibilityView(
    queryRunner: QueryRunner,
  ): Promise<void> {
    await queryRunner.query(`
      DROP TABLE ${LEGACY};
      CREATE VIEW ${LEGACY} AS
      SELECT feed_id, source_timestamp, valid_from_timestamp, price, bid, ask,
        report_sha256, backfill_artifact_id AS artifact_id, ingested_at
      FROM ${TARGET}
      WHERE strategy_key IN ('${HISTORICAL_STRATEGY}', '${BACKFILL_STRATEGY}');
    `);
  }

  private async tableExists(
    queryRunner: QueryRunner,
    relation: string,
  ): Promise<boolean> {
    const result = await queryRunner.query(`
      SELECT EXISTS (
        SELECT 1 FROM pg_class
        WHERE oid = to_regclass('${relation}') AND relkind = 'r'
      ) AS present
    `);
    return Boolean(result[0]?.present);
  }

  private hourRanges(minimum: Date, maximum: Date): Array<[Date, Date]> {
    const result: Array<[Date, Date]> = [];
    let start = new Date(minimum);
    while (start <= maximum) {
      const end = new Date(start.getTime() + 3_600_000);
      result.push([start, end]);
      start = end;
    }
    return result;
  }

  private async pause(): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  public async down(): Promise<void> {
    throw new Error(
      'Chainlink reference-price consolidation is irreversible after legacy removal',
    );
  }
}
