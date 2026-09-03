import { MigrationInterface, QueryRunner } from 'typeorm';

const LEGACY_TABLE = 'polymarket.binance_aggregate_trades';
const CANONICAL_TABLE = 'market_data.binance_spot_btcusdt_aggregate_trades';
const LEGACY_STRATEGY = 'binance_btcusdt_agg_trades';
const CANONICAL_STRATEGY = 'binance_spot_btcusdt_aggregate_trades';

export class ConsolidateBinanceAggregateTrades1788382800000
  implements MigrationInterface
{
  name = 'ConsolidateBinanceAggregateTrades1788382800000';
  transaction = false;

  public async up(queryRunner: QueryRunner): Promise<void> {
    const legacy = await queryRunner.query(
      `SELECT to_regclass('${LEGACY_TABLE}')::text AS relation`,
    );
    if (!legacy[0]?.relation) {
      return;
    }

    await queryRunner.query(`
      INSERT INTO ingester.capture_artifacts (
        artifact_id, strategy_key, profile_generation, config_schema_version,
        config_sha256, config_snapshot, capture_window_start, capture_window_end,
        minimum_source_timestamp, maximum_source_timestamp,
        minimum_received_at, maximum_received_at, record_count, content_sha256,
        status, created_at, updated_at, completed_at
      )
      SELECT
        artifact.artifact_id,
        '${CANONICAL_STRATEGY}',
        profile.desired_generation,
        profile.config_schema_version,
        encode(digest(convert_to(profile.config::text, 'UTF8'), 'sha256'), 'hex'),
        profile.config,
        artifact.source_date::timestamptz,
        (artifact.source_date + 1)::timestamptz,
        artifact.minimum_source_timestamp,
        artifact.maximum_source_timestamp,
        artifact.created_at,
        artifact.completed_at,
        COALESCE(artifact.record_count, 0),
        COALESCE(artifact.checksum, artifact.actual_checksum),
        'completed',
        artifact.created_at,
        artifact.updated_at,
        artifact.completed_at
      FROM ingester.backfill_artifacts artifact
      CROSS JOIN ingester.profiles profile
      WHERE artifact.strategy_key = '${LEGACY_STRATEGY}'
        AND profile.strategy_key = '${CANONICAL_STRATEGY}'
      ON CONFLICT (strategy_key, artifact_id) DO NOTHING
    `);

    const uncoveredArtifacts = await queryRunner.query(`
      SELECT count(*)::bigint AS count
      FROM ingester.backfill_artifacts artifact
      LEFT JOIN ingester.capture_artifacts capture
        ON capture.strategy_key = '${CANONICAL_STRATEGY}'
       AND capture.artifact_id = artifact.artifact_id
      WHERE artifact.strategy_key = '${LEGACY_STRATEGY}'
        AND capture.artifact_id IS NULL
    `);
    if (uncoveredArtifacts[0]?.count !== '0') {
      throw new Error(
        `canonical aggregate-trade lineage is missing ${uncoveredArtifacts[0].count} artifacts`,
      );
    }

    const ranges = await queryRunner.query(`
      SELECT range_start, range_end
      FROM timescaledb_information.chunks
      WHERE hypertable_schema = 'polymarket'
        AND hypertable_name = 'binance_aggregate_trades'
      ORDER BY range_start
    `);

    // Daily overlap verification is a bounded set comparison. Prevent the
    // planner from turning it into millions of point lookups.
    await queryRunner.query(`SET enable_nestloop = off`);
    await queryRunner.query(`SET work_mem = '256MB'`);

    for (const range of ranges) {
      const existing = await queryRunner.query(
        `SELECT EXISTS (
          SELECT 1
          FROM ${CANONICAL_TABLE} canonical
          LEFT JOIN ingester.backfill_artifacts legacy_artifact
            ON legacy_artifact.artifact_id = canonical.capture_artifact_id
           AND legacy_artifact.strategy_key = '${LEGACY_STRATEGY}'
          WHERE canonical.trade_timestamp >= $1
            AND canonical.trade_timestamp < $2
            AND legacy_artifact.artifact_id IS NULL
        ) AS populated`,
        [range.range_start, range.range_end],
      );

      const lineageCounts = async () =>
        queryRunner.query(
          `
            SELECT
              (SELECT count(*)::bigint
               FROM ${LEGACY_TABLE}
               WHERE trade_timestamp >= $1 AND trade_timestamp < $2) AS legacy_count,
              (SELECT count(*)::bigint
               FROM ${CANONICAL_TABLE} canonical
               JOIN ingester.backfill_artifacts artifact
                 ON artifact.artifact_id = canonical.capture_artifact_id
                AND artifact.strategy_key = '${LEGACY_STRATEGY}'
               WHERE canonical.trade_timestamp >= $1
                 AND canonical.trade_timestamp < $2) AS canonical_count
          `,
          [range.range_start, range.range_end],
        );

      if (!existing[0]?.populated) {
        const currentCounts = await lineageCounts();
        if (
          currentCounts[0]?.legacy_count ===
          currentCounts[0]?.canonical_count
        ) {
          continue;
        }
      }

      await queryRunner.query(
        `
          INSERT INTO ${CANONICAL_TABLE} (
            source, symbol, aggregate_trade_id, trade_timestamp,
            provider_available_at, received_at, price, quantity,
            first_trade_id, last_trade_id, buyer_maker, best_match,
            payload_sha256, strategy_key, capture_artifact_id, ingested_at
          )
          SELECT
            'binance_spot', legacy.symbol, legacy.aggregate_trade_id,
            date_trunc('milliseconds', legacy.trade_timestamp), NULL, legacy.ingested_at,
            legacy.price, legacy.quantity, legacy.first_trade_id,
            legacy.last_trade_id, legacy.buyer_maker, legacy.best_match,
            encode(digest(convert_to(format(
              'v1|source=binance_spot|symbol=%s|aggregate_trade_id=%s|trade_timestamp_ms=%s|price=%s|quantity=%s|first_trade_id=%s|last_trade_id=%s|buyer_maker=%s|best_match=%s',
              legacy.symbol,
              legacy.aggregate_trade_id,
              floor(extract(epoch FROM legacy.trade_timestamp) * 1000)::bigint,
              trim(trailing '.' FROM trim(trailing '0' FROM legacy.price::text)),
              trim(trailing '.' FROM trim(trailing '0' FROM legacy.quantity::text)),
              legacy.first_trade_id,
              legacy.last_trade_id,
              CASE WHEN legacy.buyer_maker THEN 'true' ELSE 'false' END,
              CASE WHEN legacy.best_match THEN 'true' ELSE 'false' END
            ), 'UTF8'), 'sha256'), 'hex'),
            '${CANONICAL_STRATEGY}', legacy.artifact_id, legacy.ingested_at
          FROM ${LEGACY_TABLE} legacy
          WHERE legacy.trade_timestamp >= $1
            AND legacy.trade_timestamp < $2
          ON CONFLICT (symbol, trade_timestamp, aggregate_trade_id) DO NOTHING
        `,
        [range.range_start, range.range_end],
      );

      if (existing[0]?.populated) {
        const difference = await queryRunner.query(
          `
          SELECT EXISTS (
            (SELECT
               symbol, aggregate_trade_id,
               date_trunc('milliseconds', trade_timestamp) AS trade_timestamp,
               price, quantity, first_trade_id, last_trade_id,
               buyer_maker, best_match
             FROM ${LEGACY_TABLE}
             WHERE trade_timestamp >= $1 AND trade_timestamp < $2)
            EXCEPT
            (SELECT
               symbol, aggregate_trade_id, trade_timestamp,
               price, quantity, first_trade_id, last_trade_id,
               buyer_maker, best_match
             FROM ${CANONICAL_TABLE}
             WHERE trade_timestamp >= $1 AND trade_timestamp < $2)
            LIMIT 1
          ) AS differs
        `,
          [range.range_start, range.range_end],
        );
        if (difference[0]?.differs) {
          throw new Error(
            `canonical aggregate-trade facts differ in ${range.range_start}..${range.range_end}`,
          );
        }
      } else {
        const counts = await lineageCounts();
        if (counts[0]?.legacy_count !== counts[0]?.canonical_count) {
          throw new Error(
            `canonical aggregate-trade row count differs in ${range.range_start}..${range.range_end}: ` +
              `${counts[0]?.legacy_count} legacy versus ${counts[0]?.canonical_count} canonical`,
          );
        }
      }
    }

    await queryRunner.query(`RESET enable_nestloop`);
    await queryRunner.query(`RESET work_mem`);

    await queryRunner.query(`DROP TABLE ${LEGACY_TABLE}`);
  }

  public async down(): Promise<void> {
    throw new Error(
      'aggregate-trade consolidation is irreversible after verified legacy-table removal',
    );
  }
}
