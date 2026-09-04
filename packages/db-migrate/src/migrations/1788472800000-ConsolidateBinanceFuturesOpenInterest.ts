import { MigrationInterface, QueryRunner } from 'typeorm';

const LEGACY_TABLE = 'polymarket.binance_btcusdt_five_minute_open_interest';
const CANONICAL_TABLE = 'market_data.binance_futures_btcusdt_open_interest';
const LEGACY_STRATEGY = 'binance_btcusdt_five_minute_open_interest';
const CANONICAL_STRATEGY = 'binance_futures_btcusdt_open_interest';

export class ConsolidateBinanceFuturesOpenInterest1788472800000
  implements MigrationInterface
{
  name = 'ConsolidateBinanceFuturesOpenInterest1788472800000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    const legacy = await queryRunner.query(
      `SELECT to_regclass('${LEGACY_TABLE}')::text AS relation`,
    );
    if (!legacy[0]?.relation) {
      return;
    }

    await queryRunner.query(`
      CREATE TEMP TABLE binance_open_interest_artifact_map (
        legacy_artifact_id uuid PRIMARY KEY,
        canonical_artifact_id uuid NOT NULL
      ) ON COMMIT DROP
    `);

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
        artifact.minimum_source_timestamp,
        artifact.maximum_source_timestamp + interval '5 minutes',
        artifact.minimum_source_timestamp,
        artifact.maximum_source_timestamp,
        artifact.created_at,
        artifact.completed_at,
        artifact.record_count,
        artifact.checksum,
        'completed',
        artifact.created_at,
        artifact.completed_at,
        artifact.completed_at
      FROM ingester.backfill_artifacts artifact
      CROSS JOIN ingester.profiles profile
      WHERE artifact.strategy_key = '${LEGACY_STRATEGY}'
        AND profile.strategy_key = '${CANONICAL_STRATEGY}'
        AND artifact.artifact_id IN (
          SELECT DISTINCT artifact_id FROM ${LEGACY_TABLE}
        )
      ON CONFLICT DO NOTHING
    `);

    await queryRunner.query(`
      INSERT INTO binance_open_interest_artifact_map (
        legacy_artifact_id, canonical_artifact_id
      )
      SELECT
        artifact.artifact_id,
        artifact.artifact_id
      FROM ingester.backfill_artifacts artifact
      CROSS JOIN ingester.profiles profile
      JOIN ingester.capture_artifacts capture
        ON capture.strategy_key = '${CANONICAL_STRATEGY}'
       AND capture.artifact_id = artifact.artifact_id
      WHERE artifact.strategy_key = '${LEGACY_STRATEGY}'
        AND profile.strategy_key = '${CANONICAL_STRATEGY}'
        AND artifact.artifact_id IN (
          SELECT DISTINCT artifact_id FROM ${LEGACY_TABLE}
        )
    `);

    const unmapped = await queryRunner.query(`
      SELECT count(DISTINCT legacy.artifact_id)::bigint AS count
      FROM ${LEGACY_TABLE} legacy
      LEFT JOIN binance_open_interest_artifact_map map
        ON map.legacy_artifact_id = legacy.artifact_id
      WHERE map.legacy_artifact_id IS NULL
    `);
    if (unmapped[0]?.count !== '0') {
      throw new Error(
        `canonical open-interest lineage is missing ${unmapped[0].count} artifacts`,
      );
    }

    await queryRunner.query(`
      INSERT INTO ${CANONICAL_TABLE} (
        source, source_timestamp, symbol, period_seconds,
        sum_open_interest, sum_open_interest_value, cmc_circulating_supply,
        provider_available_at, received_at, source_payload, payload_sha256,
        strategy_key, capture_artifact_id, ingested_at
      )
      SELECT
        'binance_usd_m_futures',
        legacy.source_timestamp,
        legacy.symbol,
        legacy.period_seconds,
        legacy.sum_open_interest,
        legacy.sum_open_interest_value,
        legacy.cmc_circulating_supply,
        NULL,
        legacy.ingested_at,
        jsonb_build_object(
          'CMCCirculatingSupply', CASE
            WHEN legacy.cmc_circulating_supply IS NULL THEN NULL
            ELSE trim(trailing '.' FROM trim(trailing '0' FROM legacy.cmc_circulating_supply::text))
          END,
          'sumOpenInterest',
            trim(trailing '.' FROM trim(trailing '0' FROM legacy.sum_open_interest::text)),
          'sumOpenInterestValue',
            trim(trailing '.' FROM trim(trailing '0' FROM legacy.sum_open_interest_value::text)),
          'symbol', legacy.symbol,
          'timestamp', floor(extract(epoch FROM legacy.source_timestamp) * 1000)::bigint
        ),
        encode(digest(convert_to(format(
          '{"cmc_circulating_supply":%s,"sum_open_interest":%s,"sum_open_interest_value":%s,"symbol":%s,"timestamp":%s}',
          CASE
            WHEN legacy.cmc_circulating_supply IS NULL THEN 'null'
            ELSE to_jsonb(trim(trailing '.' FROM trim(trailing '0' FROM legacy.cmc_circulating_supply::text)))::text
          END,
          to_jsonb(trim(trailing '.' FROM trim(trailing '0' FROM legacy.sum_open_interest::text)))::text,
          to_jsonb(trim(trailing '.' FROM trim(trailing '0' FROM legacy.sum_open_interest_value::text)))::text,
          to_jsonb(legacy.symbol)::text,
          floor(extract(epoch FROM legacy.source_timestamp) * 1000)::bigint
        ), 'UTF8'), 'sha256'), 'hex'),
        '${CANONICAL_STRATEGY}',
        map.canonical_artifact_id,
        legacy.ingested_at
      FROM ${LEGACY_TABLE} legacy
      JOIN binance_open_interest_artifact_map map
        ON map.legacy_artifact_id = legacy.artifact_id
      ON CONFLICT (source_timestamp, symbol, period_seconds) DO NOTHING
    `);

    const conflict = await queryRunner.query(`
      SELECT EXISTS (
        SELECT 1
        FROM ${LEGACY_TABLE} legacy
        JOIN ${CANONICAL_TABLE} canonical
          ON canonical.source_timestamp = legacy.source_timestamp
         AND canonical.symbol = legacy.symbol
         AND canonical.period_seconds = legacy.period_seconds
        WHERE canonical.source <> 'binance_usd_m_futures'
           OR canonical.sum_open_interest IS DISTINCT FROM legacy.sum_open_interest
           OR canonical.sum_open_interest_value IS DISTINCT FROM legacy.sum_open_interest_value
           OR canonical.cmc_circulating_supply IS DISTINCT FROM legacy.cmc_circulating_supply
      ) AS differs
    `);
    if (conflict[0]?.differs) {
      throw new Error('canonical open-interest facts conflict with legacy values');
    }

    const accounting = await queryRunner.query(`
      SELECT
        (SELECT count(*)::bigint FROM ${LEGACY_TABLE}) AS legacy_count,
        (SELECT count(*)::bigint
         FROM ${LEGACY_TABLE} legacy
         JOIN ${CANONICAL_TABLE} canonical
           ON canonical.source_timestamp = legacy.source_timestamp
          AND canonical.symbol = legacy.symbol
          AND canonical.period_seconds = legacy.period_seconds) AS represented_count
    `);
    if (accounting[0]?.legacy_count !== accounting[0]?.represented_count) {
      throw new Error(
        `canonical open-interest accounting differs: ${accounting[0]?.legacy_count} legacy versus ` +
          `${accounting[0]?.represented_count} represented`,
      );
    }

    await queryRunner.query(`DROP TABLE ${LEGACY_TABLE}`);
  }

  public async down(): Promise<void> {
    throw new Error(
      'Binance open-interest consolidation is irreversible after verified legacy-table removal',
    );
  }
}
