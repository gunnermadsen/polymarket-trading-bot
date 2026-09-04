import { MigrationInterface, QueryRunner } from 'typeorm';

type Dataset = {
  legacyTable: string;
  canonicalTable: string;
  canonicalStrategy: string;
  timeColumn: string;
  interval: string;
};

const DATASETS: Dataset[] = [
  {
    legacyTable: 'polymarket.binance_one_second_klines',
    canonicalTable: 'market_data.binance_spot_btcusdt_one_second_ohlcv',
    canonicalStrategy: 'binance_spot_btcusdt_one_second_ohlcv',
    timeColumn: 'open_timestamp',
    interval: '1 second',
  },
  {
    legacyTable: 'polymarket.chainlink_btcusd_one_minute_candles',
    canonicalTable: 'market_data.chainlink_btcusd_one_minute_candles',
    canonicalStrategy: 'chainlink_btcusd_one_minute_ohlc',
    timeColumn: 'open_timestamp',
    interval: '1 minute',
  },
  {
    legacyTable: 'polymarket.polygon_chainlink_btcusd_oracle_rounds',
    canonicalTable: 'market_data.polygon_chainlink_btcusd_oracle_rounds',
    canonicalStrategy: 'polygon_chainlink_btcusd_oracle',
    timeColumn: 'source_timestamp',
    interval: '1 second',
  },
];

export class ConsolidateCanonicalMarketDataTables1788559200000
  implements MigrationInterface
{
  name = 'ConsolidateCanonicalMarketDataTables1788559200000';
  transaction = false;

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(
      'SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0',
    );
    try {
      for (const dataset of DATASETS) {
        await this.ensureCaptureLineage(queryRunner, dataset);
      }

      await this.migrateBinanceOhlcv(queryRunner);
      await this.migrateChainlinkCandles(queryRunner);
      await this.migrateOracleRounds(queryRunner);
    } finally {
      await queryRunner.query(
        'RESET timescaledb.max_tuples_decompressed_per_dml_transaction',
      );
    }
  }

  private async ensureCaptureLineage(
    queryRunner: QueryRunner,
    dataset: Dataset,
  ): Promise<void> {
    const relation = await queryRunner.query(
      `SELECT to_regclass('${dataset.legacyTable}')::text AS relation`,
    );
    if (!relation[0]?.relation) return;

    await queryRunner.query(`
      INSERT INTO ingester.capture_artifacts (
        artifact_id, strategy_key, profile_generation, config_schema_version,
        config_sha256, config_snapshot, capture_window_start, capture_window_end,
        minimum_source_timestamp, maximum_source_timestamp,
        minimum_received_at, maximum_received_at, record_count, content_sha256,
        status, created_at, updated_at, completed_at
      )
      SELECT legacy.artifact_id, '${dataset.canonicalStrategy}',
        profile.desired_generation, profile.config_schema_version,
        encode(digest(convert_to(profile.config::text, 'UTF8'), 'sha256'), 'hex'),
        profile.config,
        min(legacy.${dataset.timeColumn}),
        max(legacy.${dataset.timeColumn}) + interval '${dataset.interval}',
        min(legacy.${dataset.timeColumn}), max(legacy.${dataset.timeColumn}),
        min(legacy.ingested_at), max(legacy.ingested_at), count(*)::bigint,
        encode(digest(convert_to(
          '${dataset.canonicalStrategy}:' || legacy.artifact_id::text || ':' || count(*)::text,
          'UTF8'
        ), 'sha256'), 'hex'),
        'completed', min(legacy.ingested_at), max(legacy.ingested_at), max(legacy.ingested_at)
      FROM ${dataset.legacyTable} legacy
      CROSS JOIN ingester.profiles profile
      WHERE profile.strategy_key = '${dataset.canonicalStrategy}'
      GROUP BY legacy.artifact_id, profile.desired_generation,
        profile.config_schema_version, profile.config
      ON CONFLICT DO NOTHING
    `);

    const missing = await queryRunner.query(`
      SELECT count(*)::bigint AS count
      FROM (
        SELECT DISTINCT legacy.artifact_id
        FROM ${dataset.legacyTable} legacy
      ) legacy
      LEFT JOIN ingester.capture_artifacts artifact
        ON artifact.artifact_id = legacy.artifact_id
       AND artifact.strategy_key = '${dataset.canonicalStrategy}'
      WHERE artifact.artifact_id IS NULL
    `);
    if (missing[0]?.count !== '0') {
      throw new Error(
        `${dataset.legacyTable} has ${missing[0].count} artifacts without canonical lineage`,
      );
    }
  }

  private async migrateBinanceOhlcv(queryRunner: QueryRunner): Promise<void> {
    const legacy = 'polymarket.binance_one_second_klines';
    const canonical = 'market_data.binance_spot_btcusdt_one_second_ohlcv';
    if (!(await this.exists(queryRunner, legacy))) return;

    const before = await this.validateExistingOverlap(
      queryRunner, legacy, canonical,
      'canonical.symbol = legacy.symbol AND canonical.open_timestamp = legacy.open_timestamp',
      `canonical.close_timestamp IS DISTINCT FROM legacy.open_timestamp + interval '999 milliseconds'
       OR canonical.open_price IS DISTINCT FROM legacy.open_price
       OR canonical.high_price IS DISTINCT FROM legacy.high_price
       OR canonical.low_price IS DISTINCT FROM legacy.low_price
       OR canonical.close_price IS DISTINCT FROM legacy.close_price
       OR canonical.base_volume IS DISTINCT FROM legacy.base_volume
       OR canonical.quote_volume IS DISTINCT FROM legacy.quote_volume
       OR canonical.trade_count IS DISTINCT FROM legacy.trade_count
       OR canonical.taker_buy_base_volume IS DISTINCT FROM legacy.taker_buy_base_volume
       OR canonical.taker_buy_quote_volume IS DISTINCT FROM legacy.taker_buy_quote_volume`,
      'open_timestamp',
      1 / 24,
    );

    let insertedCount = 0n;
    for (const [rangeStart, rangeEnd] of await this.ranges(
      queryRunner,
      legacy,
      'open_timestamp',
      1 / 24,
    )) {
      const inserted = await queryRunner.query(`
      WITH inserted AS (
      INSERT INTO ${canonical} (
        source, symbol, open_timestamp, close_timestamp,
        provider_available_at, received_at, open_price, high_price, low_price,
        close_price, base_volume, quote_volume, trade_count,
        taker_buy_base_volume, taker_buy_quote_volume, payload_sha256,
        strategy_key, capture_artifact_id, ingested_at
      )
      SELECT 'binance_spot', symbol, open_timestamp,
        open_timestamp + interval '999 milliseconds',
        NULL, ingested_at, open_price, high_price, low_price, close_price,
        base_volume, quote_volume, trade_count,
        taker_buy_base_volume, taker_buy_quote_volume,
        encode(digest(convert_to(format(
          'v1|source=binance_spot|symbol=%s|open_timestamp_ms=%s|close_timestamp_ms=%s|open_price=%s|high_price=%s|low_price=%s|close_price=%s|base_volume=%s|quote_volume=%s|trade_count=%s|taker_buy_base_volume=%s|taker_buy_quote_volume=%s',
          symbol,
          floor(extract(epoch FROM open_timestamp) * 1000)::bigint,
          floor(extract(epoch FROM open_timestamp + interval '999 milliseconds') * 1000)::bigint,
          trim(trailing '.' FROM trim(trailing '0' FROM open_price::text)),
          trim(trailing '.' FROM trim(trailing '0' FROM high_price::text)),
          trim(trailing '.' FROM trim(trailing '0' FROM low_price::text)),
          trim(trailing '.' FROM trim(trailing '0' FROM close_price::text)),
          trim(trailing '.' FROM trim(trailing '0' FROM base_volume::text)),
          trim(trailing '.' FROM trim(trailing '0' FROM quote_volume::text)),
          trade_count,
          trim(trailing '.' FROM trim(trailing '0' FROM taker_buy_base_volume::text)),
          trim(trailing '.' FROM trim(trailing '0' FROM taker_buy_quote_volume::text))
        ), 'UTF8'), 'sha256'), 'hex'),
        'binance_spot_btcusdt_one_second_ohlcv', artifact_id, ingested_at
      FROM ${legacy}
      WHERE open_timestamp >= $1 AND open_timestamp < $2
      ON CONFLICT (symbol, open_timestamp) DO NOTHING
      RETURNING 1
      ) SELECT count(*)::bigint AS count FROM inserted
    `, [rangeStart, rangeEnd]);
      insertedCount += BigInt(inserted[0]?.count ?? '-1');
      await this.pauseBetweenBatches();
    }
    this.assertCompleteAccounting(legacy, before, insertedCount);
    await queryRunner.query(`DROP TABLE ${legacy}`);
  }

  private async migrateChainlinkCandles(queryRunner: QueryRunner): Promise<void> {
    const legacy = 'polymarket.chainlink_btcusd_one_minute_candles';
    const canonical = 'market_data.chainlink_btcusd_one_minute_candles';
    if (!(await this.exists(queryRunner, legacy))) return;

    const unsupported = await queryRunner.query(`
      SELECT EXISTS (
        SELECT 1 FROM ${legacy}
        WHERE volume_supported OR volume IS NOT NULL LIMIT 1
      ) AS differs
    `);
    if (unsupported[0]?.differs) {
      throw new Error('legacy Chainlink candles contain unsupported volume facts');
    }

    const before = await this.validateExistingOverlap(
      queryRunner, legacy, canonical,
      'canonical.symbol = legacy.symbol AND canonical.open_timestamp = legacy.open_timestamp',
      `canonical.close_timestamp IS DISTINCT FROM legacy.close_timestamp
       OR canonical.open_price IS DISTINCT FROM legacy.open_price
       OR canonical.high_price IS DISTINCT FROM legacy.high_price
       OR canonical.low_price IS DISTINCT FROM legacy.low_price
       OR canonical.close_price IS DISTINCT FROM legacy.close_price
       OR canonical.volume IS DISTINCT FROM legacy.volume
       OR canonical.volume_supported IS DISTINCT FROM legacy.volume_supported`,
      'open_timestamp', 7);

    let insertedCount = 0n;
    for (const [rangeStart, rangeEnd] of await this.ranges(queryRunner, legacy, 'open_timestamp', 7)) {
      const inserted = await queryRunner.query(`
      WITH inserted AS (
      INSERT INTO ${canonical} (
        source, symbol, open_timestamp, close_timestamp,
        provider_available_at, received_at, open_price, high_price, low_price,
        close_price, volume, volume_supported, payload_sha256,
        strategy_key, capture_artifact_id, ingested_at
      )
      SELECT 'chainlink_candlestick', symbol, open_timestamp, close_timestamp,
        NULL, ingested_at, open_price, high_price, low_price, close_price,
        volume, volume_supported,
        encode(digest(convert_to(format(
          'v1|source=chainlink_candlestick|symbol=%s|open_timestamp=%s|close_timestamp=%s|open_price=%s|high_price=%s|low_price=%s|close_price=%s|volume=unsupported',
          symbol, extract(epoch FROM open_timestamp)::bigint,
          extract(epoch FROM close_timestamp)::bigint,
          trim(trailing '.' FROM trim(trailing '0' FROM open_price::text)),
          trim(trailing '.' FROM trim(trailing '0' FROM high_price::text)),
          trim(trailing '.' FROM trim(trailing '0' FROM low_price::text)),
          trim(trailing '.' FROM trim(trailing '0' FROM close_price::text))
        ), 'UTF8'), 'sha256'), 'hex'),
        'chainlink_btcusd_one_minute_ohlc', artifact_id, ingested_at
      FROM ${legacy}
      WHERE open_timestamp >= $1 AND open_timestamp < $2
      ON CONFLICT (symbol, open_timestamp) DO NOTHING
      RETURNING 1
      ) SELECT count(*)::bigint AS count FROM inserted
    `, [rangeStart, rangeEnd]);
      insertedCount += BigInt(inserted[0]?.count ?? '-1');
    }
    this.assertCompleteAccounting(legacy, before, insertedCount);
    await queryRunner.query(`DROP TABLE ${legacy}`);
  }

  private async migrateOracleRounds(queryRunner: QueryRunner): Promise<void> {
    const legacy = 'polymarket.polygon_chainlink_btcusd_oracle_rounds';
    const canonical = 'market_data.polygon_chainlink_btcusd_oracle_rounds';
    if (!(await this.exists(queryRunner, legacy))) return;

    const before = await this.validateExistingOverlap(
      queryRunner, legacy, canonical,
      `canonical.source_timestamp = legacy.source_timestamp
       AND canonical.chain_id = legacy.chain_id
       AND canonical.feed_proxy_address = legacy.feed_proxy_address
       AND canonical.transaction_hash = legacy.transaction_hash
       AND canonical.log_index = legacy.log_index`,
      `canonical.aggregator_address IS DISTINCT FROM legacy.aggregator_address
       OR canonical.phase_id IS DISTINCT FROM legacy.phase_id
       OR canonical.aggregator_round_id IS DISTINCT FROM legacy.aggregator_round_id
       OR canonical.block_timestamp IS DISTINCT FROM legacy.block_timestamp
       OR canonical.answer_raw IS DISTINCT FROM legacy.answer_raw
       OR canonical.price IS DISTINCT FROM legacy.price
       OR canonical.decimals IS DISTINCT FROM legacy.decimals
       OR canonical.block_number IS DISTINCT FROM legacy.block_number
       OR canonical.block_hash IS DISTINCT FROM legacy.block_hash`,
      'source_timestamp', 7);

    let insertedCount = 0n;
    for (const [rangeStart, rangeEnd] of await this.ranges(queryRunner, legacy, 'source_timestamp', 7)) {
      const inserted = await queryRunner.query(`
      WITH inserted AS (
      INSERT INTO ${canonical} (
        source, chain_id, feed_proxy_address, aggregator_address, phase_id,
        aggregator_round_id, source_timestamp, block_timestamp, answer_raw,
        price, decimals, block_number, block_hash, transaction_hash, log_index,
        provider_available_at, received_at, source_payload, payload_sha256,
        strategy_key, capture_artifact_id, ingested_at
      )
      SELECT 'chainlink_polygon_data_feed', chain_id, feed_proxy_address,
        aggregator_address, phase_id, aggregator_round_id, source_timestamp,
        block_timestamp, answer_raw, price, decimals, block_number, block_hash,
        transaction_hash, log_index, block_timestamp, ingested_at,
        payload.value,
        encode(digest(convert_to(payload.value::text, 'UTF8'), 'sha256'), 'hex'),
        'polygon_chainlink_btcusd_oracle', artifact_id, ingested_at
      FROM ${legacy} legacy
      CROSS JOIN LATERAL (
        SELECT jsonb_build_object(
          'aggregatorAddress', aggregator_address,
          'aggregatorRoundId', aggregator_round_id,
          'answerRaw', trim(trailing '.' FROM trim(trailing '0' FROM answer_raw::text)),
          'blockHash', block_hash, 'blockNumber', block_number,
          'blockTimestamp', extract(epoch FROM block_timestamp)::bigint,
          'chainId', chain_id, 'decimals', decimals,
          'feedProxyAddress', feed_proxy_address, 'logIndex', log_index,
          'phaseId', phase_id,
          'sourceTimestamp', extract(epoch FROM source_timestamp)::bigint,
          'transactionHash', transaction_hash
        ) AS value
      ) payload
      WHERE source_timestamp >= $1 AND source_timestamp < $2
      ON CONFLICT DO NOTHING
      RETURNING 1
      ) SELECT count(*)::bigint AS count FROM inserted
    `, [rangeStart, rangeEnd]);
      insertedCount += BigInt(inserted[0]?.count ?? '-1');
    }
    this.assertCompleteAccounting(legacy, before, insertedCount);
    await queryRunner.query(`DROP TABLE ${legacy}`);
  }

  private async exists(queryRunner: QueryRunner, relation: string): Promise<boolean> {
    const rows = await queryRunner.query(
      `SELECT to_regclass('${relation}') IS NOT NULL AS present`,
    );
    return Boolean(rows[0]?.present);
  }

  private async validateExistingOverlap(
    queryRunner: QueryRunner,
    legacy: string,
    canonical: string,
    identity: string,
    differs: string,
    timeColumn: string,
    batchDays: number,
  ): Promise<{ legacyCount: bigint; overlapCount: bigint }> {
    const bounds = await queryRunner.query(`
      SELECT min(${timeColumn}) AS minimum, max(${timeColumn}) AS maximum
      FROM ${canonical}
    `);
    let overlapCount = 0n;
    if (bounds[0]?.minimum && bounds[0]?.maximum) {
      let start = new Date(bounds[0].minimum);
      const maximum = new Date(bounds[0].maximum);
      while (start <= maximum) {
        const end = new Date(start.getTime() + batchDays * 86_400_000);
        const conflict = await queryRunner.query(`
          SELECT EXISTS (
            SELECT 1
            FROM ${canonical} canonical JOIN ${legacy} legacy ON ${identity}
            WHERE canonical.${timeColumn} >= $1 AND canonical.${timeColumn} < $2
              AND legacy.${timeColumn} >= $1 AND legacy.${timeColumn} < $2
              AND (${differs})
            LIMIT 1
          ) AS differs
        `, [start, end]);
        if (conflict[0]?.differs) {
          throw new Error(`${canonical} conflicts with legacy factual values`);
        }
        const overlap = await queryRunner.query(`
          SELECT count(*)::bigint AS count
          FROM ${canonical} canonical JOIN ${legacy} legacy ON ${identity}
          WHERE canonical.${timeColumn} >= $1 AND canonical.${timeColumn} < $2
            AND legacy.${timeColumn} >= $1 AND legacy.${timeColumn} < $2
        `, [start, end]);
        overlapCount += BigInt(overlap[0].count);
        start = end;
      }
    }
    const counts = await queryRunner.query(`
      SELECT count(*)::bigint AS legacy_count FROM ${legacy}
    `);
    return {
      legacyCount: BigInt(counts[0].legacy_count),
      overlapCount,
    };
  }

  private assertCompleteAccounting(
    legacy: string,
    before: { legacyCount: bigint; overlapCount: bigint },
    inserted: bigint,
  ): void {
    if (before.overlapCount + inserted !== before.legacyCount) {
      throw new Error(
        `${legacy} accounting differs: ${before.legacyCount} legacy, ` +
          `${before.overlapCount} overlapping, ${inserted} inserted`,
      );
    }
  }

  private async ranges(
    queryRunner: QueryRunner,
    relation: string,
    timeColumn: string,
    batchDays: number,
  ): Promise<Array<[Date, Date]>> {
    const bounds = await queryRunner.query(`
      SELECT min(${timeColumn}) AS minimum, max(${timeColumn}) AS maximum
      FROM ${relation}
    `);
    if (!bounds[0]?.minimum || !bounds[0]?.maximum) return [];
    const ranges: Array<[Date, Date]> = [];
    let start = new Date(bounds[0].minimum);
    const maximum = new Date(bounds[0].maximum);
    while (start <= maximum) {
      const end = new Date(start.getTime() + batchDays * 86_400_000);
      ranges.push([start, end]);
      start = end;
    }
    return ranges;
  }

  private async pauseBetweenBatches(): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  public async down(): Promise<void> {
    throw new Error(
      'canonical market-data consolidation is irreversible after verified legacy-table removal',
    );
  }
}
