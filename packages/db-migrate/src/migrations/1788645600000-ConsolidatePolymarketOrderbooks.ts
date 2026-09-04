import { MigrationInterface, QueryRunner } from 'typeorm';
import { createHash } from 'node:crypto';

const SOURCE = 'market_data.polymarket_btc_five_minute_orderbook_snapshots';
const TARGET = 'polymarket.btc_five_minute_orderbook_snapshots';
const LEGACY = 'polymarket.orderbook_checkpoints';
const STRATEGY = 'polymarket_btc_five_minute_orderbooks';

export class ConsolidatePolymarketOrderbooks1788645600000
  implements MigrationInterface
{
  name = 'ConsolidatePolymarketOrderbooks1788645600000';
  transaction = false;

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(
      'SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0',
    );
    try {
      await this.relocateCanonicalTable(queryRunner);
      if (!(await this.exists(queryRunner, LEGACY, 'r'))) return;
      await this.repairInterruptedArtifact(queryRunner);

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
        const artifactId = this.legacyArtifactId(start);
        await this.ensureLegacyArtifact(queryRunner, artifactId, start, end);
        await this.removeProvisionalLegacyRows(queryRunner, artifactId);
        const result = await queryRunner.query(
          `
          WITH source_rows AS MATERIALIZED (
            SELECT
              checkpoint.checkpoint_id,
              checkpoint.source_timestamp,
              checkpoint.received_at,
              checkpoint.persisted_at,
              checkpoint.connection_id,
              checkpoint.ingest_sequence,
              checkpoint.market_id,
              checkpoint.token_id,
              checkpoint.best_bid,
              checkpoint.best_ask,
              checkpoint.tick_size,
              checkpoint.book,
              checkpoint.source_hash,
              checkpoint.bootstrap_source,
              checkpoint.integrity_status,
              market.condition_id,
              market.event_slug,
              market.window_start,
              market.window_end,
              CASE
                WHEN checkpoint.token_id = market.up_token_id THEN 'up'
                WHEN checkpoint.token_id = market.down_token_id THEN 'down'
              END AS outcome,
              coalesce(jsonb_array_length(checkpoint.book -> 'bids'), 0) AS bid_depth,
              coalesce(jsonb_array_length(checkpoint.book -> 'asks'), 0) AS ask_depth
            FROM ${LEGACY} checkpoint
            LEFT JOIN polymarket.btc_interval_markets market
              ON market.market_id = checkpoint.market_id
             AND checkpoint.token_id IN (market.up_token_id, market.down_token_id)
            WHERE checkpoint.source_timestamp >= $1
              AND checkpoint.source_timestamp < $2
          ), prepared AS MATERIALIZED (
            SELECT source_rows.*,
              coalesce((
                SELECT jsonb_agg(
                  jsonb_build_array(level ->> 'price', level ->> 'size')
                  ORDER BY ordinal
                )
                FROM jsonb_array_elements(source_rows.book -> 'bids')
                  WITH ORDINALITY item(level, ordinal)
              ), '[]'::jsonb) AS bids,
              coalesce((
                SELECT jsonb_agg(
                  jsonb_build_array(level ->> 'price', level ->> 'size')
                  ORDER BY ordinal
                )
                FROM jsonb_array_elements(source_rows.book -> 'asks')
                  WITH ORDINALITY item(level, ordinal)
              ), '[]'::jsonb) AS asks,
              jsonb_build_object(
                'version', 'polymarket-clob-btc-5m-orderbook-top-n-v1',
                'source', 'polymarket_clob_market',
                'selection',
                  'latest_valid_subscribed_market_book_at_aligned_wall_clock_slot',
                'market_interval_seconds', 300,
                'sample_interval_ms', 1000,
                'top_n', least(1000, greatest(1, bid_depth, ask_depth)),
                'legacy_event_driven_checkpoint', true,
                'legacy_checkpoint_id', checkpoint_id
              ) AS policy
            FROM source_rows
          ), inserted AS (
            INSERT INTO ${TARGET} (
              sampled_at, source_timestamp, provider_available_at, received_at,
              source, market_id, condition_id, event_slug, window_start,
              window_end, token_id, outcome, connection_epoch, ingest_sequence,
              tick_size, best_bid, best_ask, bid_depth, ask_depth, bids, asks,
              source_hash, book_sha256, sampling_policy, sampling_policy_sha256,
              payload_sha256, strategy_key, capture_artifact_id, ingested_at,
              legacy_checkpoint_id, legacy_source_payload, bootstrap_source,
              integrity_status
            )
            SELECT
              persisted_at, source_timestamp, source_timestamp, received_at,
              'polymarket_clob_market', market_id, condition_id, event_slug,
              window_start, window_end, token_id, outcome, connection_id,
              ingest_sequence, tick_size, best_bid, best_ask, bid_depth,
              ask_depth, bids, asks, source_hash,
              encode(digest(convert_to(bids::text || '|' || asks::text, 'UTF8'),
                'sha256'), 'hex'),
              policy,
              encode(digest(convert_to(policy::text, 'UTF8'), 'sha256'), 'hex'),
              encode(digest(convert_to(
                checkpoint_id::text || '|' || book::text, 'UTF8'
              ), 'sha256'), 'hex'),
              '${STRATEGY}', $3::uuid, persisted_at,
              checkpoint_id, book, bootstrap_source, integrity_status
            FROM prepared
            WHERE condition_id IS NOT NULL AND outcome IS NOT NULL
            ON CONFLICT DO NOTHING
            RETURNING legacy_checkpoint_id
          ), accounting AS (
            SELECT
              (SELECT count(*)::bigint FROM source_rows) AS legacy_count,
            (SELECT count(*)::bigint FROM ${TARGET} canonical
               JOIN prepared legacy
                 ON canonical.sampled_at = legacy.persisted_at
                AND canonical.market_id = legacy.market_id
                AND canonical.token_id = legacy.token_id
                AND canonical.sampling_policy_sha256 = encode(digest(
                  convert_to(legacy.policy::text, 'UTF8'), 'sha256'
                ), 'hex')) AS canonical_count,
              (SELECT count(*)::bigint FROM source_rows
               WHERE condition_id IS NULL OR outcome IS NULL) AS unmapped_count,
              (SELECT count(*)::bigint FROM inserted) AS inserted_count
          )
          SELECT legacy_count, canonical_count, unmapped_count, inserted_count
          FROM accounting
          `,
          [start, end, artifactId],
        );
        const row = result[0];
        if (row.unmapped_count !== '0') {
          throw new Error(
            `${LEGACY} contains ${row.unmapped_count} unmapped rows in ${start.toISOString()}`,
          );
        }
        legacyCount += BigInt(row.legacy_count);
        canonicalCount += BigInt(row.canonical_count);
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

  private async relocateCanonicalTable(queryRunner: QueryRunner): Promise<void> {
    if (await this.exists(queryRunner, SOURCE, 'r')) {
      await queryRunner.query(`
        ALTER TABLE ${SOURCE} SET SCHEMA polymarket;
        ALTER TABLE polymarket.polymarket_btc_five_minute_orderbook_snapshots
          RENAME TO btc_five_minute_orderbook_snapshots;
        CREATE VIEW ${SOURCE} AS SELECT * FROM ${TARGET};
      `);
    }
    await queryRunner.query(`
      ALTER TABLE ${TARGET} SET (timescaledb.compress = false);

      ALTER TABLE ${TARGET}
        ADD COLUMN IF NOT EXISTS legacy_checkpoint_id uuid,
        ADD COLUMN IF NOT EXISTS legacy_source_payload jsonb,
        ADD COLUMN IF NOT EXISTS bootstrap_source text,
        ADD COLUMN IF NOT EXISTS integrity_status text NOT NULL DEFAULT 'ok';

      ALTER TABLE ${TARGET}
        DROP CONSTRAINT IF EXISTS
          chk_market_data_polymarket_btc_five_minute_orderbook_prices,
        DROP CONSTRAINT IF EXISTS
          chk_market_data_polymarket_btc_five_minute_orderbook_book,
        DROP CONSTRAINT IF EXISTS
          chk_market_data_polymarket_btc_five_minute_orderbook_time;
      ALTER TABLE ${TARGET}
        ADD CONSTRAINT
          chk_market_data_polymarket_btc_five_minute_orderbook_prices
        CHECK (
          legacy_checkpoint_id IS NOT NULL OR (
            tick_size > 0 AND tick_size < 1
            AND (best_bid IS NULL OR (best_bid > 0 AND best_bid < 1))
            AND (best_ask IS NULL OR (best_ask > 0 AND best_ask < 1))
            AND (best_bid IS NULL OR best_ask IS NULL OR best_bid < best_ask)
          )
        ) NOT VALID,
        ADD CONSTRAINT
          chk_market_data_polymarket_btc_five_minute_orderbook_book
        CHECK (
          legacy_checkpoint_id IS NOT NULL OR (
            octet_length(bids::text) <= 262144
            AND octet_length(asks::text) <= 262144
            AND market_data.is_valid_polymarket_btc_five_minute_book(
              bids, asks, bid_depth, ask_depth, best_bid, best_ask,
              (sampling_policy ->> 'top_n')::integer
            )
          )
        ) NOT VALID,
        ADD CONSTRAINT
          chk_market_data_polymarket_btc_five_minute_orderbook_time
        CHECK (
          legacy_checkpoint_id IS NOT NULL OR (
            provider_available_at = source_timestamp
            AND received_at <= sampled_at
          )
        ) NOT VALID;

      ALTER TABLE ${TARGET} SET (
        timescaledb.compress = true,
        timescaledb.compress_segmentby =
          'market_id, token_id, sampling_policy_sha256, strategy_key, capture_artifact_id',
        timescaledb.compress_orderby =
          'sampled_at, source_timestamp, ingest_sequence'
      );
    `);
  }

  private async removeProvisionalLegacyRows(
    queryRunner: QueryRunner,
    artifactId: string,
  ): Promise<void> {
    await queryRunner.query(`
      DELETE FROM ${TARGET}
      WHERE capture_artifact_id = $1::uuid
        AND sampling_policy ->> 'legacy_event_driven_checkpoint' = 'true'
        AND NOT sampling_policy ? 'legacy_checkpoint_id'
    `, [artifactId]);
  }

  private async ensureLegacyArtifact(
    queryRunner: QueryRunner,
    artifactId: string,
    minimum: Date,
    maximum: Date,
  ): Promise<void> {
    await queryRunner.query(`
      INSERT INTO ingester.capture_artifacts (
        artifact_id, strategy_key, profile_generation, config_schema_version,
        config_sha256, config_snapshot, capture_window_start,
        capture_window_end, minimum_source_timestamp,
        maximum_source_timestamp, minimum_received_at, maximum_received_at,
        record_count, content_sha256, status,
        created_at, updated_at, completed_at
      )
      SELECT $3::uuid, '${STRATEGY}', desired_generation,
        config_schema_version,
        encode(digest(convert_to(config::text, 'UTF8'), 'sha256'), 'hex'),
        config, $1::timestamptz, $2::timestamptz,
        facts.minimum_source_timestamp, facts.maximum_source_timestamp,
        facts.minimum_received_at, facts.maximum_received_at,
        facts.record_count,
        encode(digest(convert_to(
          'legacy-orderbook-checkpoints:' || $3::text, 'UTF8'
        ), 'sha256'), 'hex'),
        'completed', now(), now(), now()
      FROM ingester.profiles
      CROSS JOIN LATERAL (
        SELECT min(source_timestamp) AS minimum_source_timestamp,
          max(source_timestamp) AS maximum_source_timestamp,
          min(received_at) AS minimum_received_at,
          max(received_at) AS maximum_received_at,
          count(*)::bigint AS record_count
        FROM ${LEGACY}
        WHERE source_timestamp >= $1 AND source_timestamp < $2
      ) facts
      WHERE strategy_key = '${STRATEGY}'
      ON CONFLICT DO NOTHING
    `, [minimum, maximum, artifactId]);
  }

  private async repairInterruptedArtifact(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ingester.capture_artifacts
        DISABLE TRIGGER trg_reject_terminal_capture_artifact_change;
      UPDATE ingester.capture_artifacts artifact
      SET record_count = facts.record_count,
          minimum_source_timestamp = facts.minimum_source_timestamp,
          maximum_source_timestamp = facts.maximum_source_timestamp,
          minimum_received_at = facts.minimum_received_at,
          maximum_received_at = facts.maximum_received_at,
          updated_at = now(), completed_at = now()
      FROM (
        SELECT count(*)::bigint AS record_count,
          min(source_timestamp) AS minimum_source_timestamp,
          max(source_timestamp) AS maximum_source_timestamp,
          min(received_at) AS minimum_received_at,
          max(received_at) AS maximum_received_at
        FROM ${LEGACY}
        WHERE source_timestamp >= '2026-07-13T01:39:44.757Z'
          AND source_timestamp < '2026-07-13T02:39:44.757Z'
      ) facts
      WHERE artifact.artifact_id = '282eea2b-e54a-0edb-d4c8-71d6d5b99bf6'
        AND artifact.strategy_key = '${STRATEGY}'
        AND artifact.record_count = 0;
      ALTER TABLE ingester.capture_artifacts
        ENABLE TRIGGER trg_reject_terminal_capture_artifact_change;
    `);
  }

  private async replaceLegacyWithCompatibilityView(
    queryRunner: QueryRunner,
  ): Promise<void> {
    await queryRunner.query(`
      DROP TABLE ${LEGACY};
      CREATE VIEW ${LEGACY} AS
      SELECT
        coalesce(
          legacy_checkpoint_id,
          (substr(md5(payload_sha256),1,8) || '-' ||
           substr(md5(payload_sha256),9,4) || '-' ||
           substr(md5(payload_sha256),13,4) || '-' ||
           substr(md5(payload_sha256),17,4) || '-' ||
           substr(md5(payload_sha256),21,12))::uuid
        ) AS checkpoint_id,
        source_timestamp, received_at, sampled_at AS persisted_at,
        connection_epoch AS connection_id, ingest_sequence, market_id, token_id,
        best_bid, best_ask,
        CASE WHEN best_bid IS NULL OR best_ask IS NULL
          THEN NULL ELSE best_ask - best_bid END AS spread,
        tick_size,
        coalesce((
          SELECT sum((level ->> 1)::numeric)
          FROM jsonb_array_elements(snapshot.bids) level
        ), 0) AS depth_bid,
        coalesce((
          SELECT sum((level ->> 1)::numeric)
          FROM jsonb_array_elements(snapshot.asks) level
        ), 0) AS depth_ask,
        coalesce(legacy_source_payload, jsonb_build_object(
          'bids', (SELECT jsonb_agg(jsonb_build_object(
            'price', level ->> 0, 'size', level ->> 1))
            FROM jsonb_array_elements(snapshot.bids) level),
          'asks', (SELECT jsonb_agg(jsonb_build_object(
            'price', level ->> 0, 'size', level ->> 1))
            FROM jsonb_array_elements(snapshot.asks) level)
        )) AS book,
        source_hash, coalesce(bootstrap_source, 'ingester_worker') AS bootstrap_source,
        integrity_status
      FROM ${TARGET} snapshot;
    `);
  }

  private async exists(
    queryRunner: QueryRunner,
    relation: string,
    kind: string,
  ): Promise<boolean> {
    const result = await queryRunner.query(`
      SELECT EXISTS (
        SELECT 1 FROM pg_class
        WHERE oid = to_regclass('${relation}') AND relkind = '${kind}'
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

  private legacyArtifactId(start: Date): string {
    const value = createHash('sha256')
      .update(`polymarket-orderbook-checkpoints:${start.toISOString()}`)
      .digest('hex')
      .slice(0, 32);
    return `${value.slice(0, 8)}-${value.slice(8, 12)}-${value.slice(12, 16)}-${value.slice(16, 20)}-${value.slice(20)}`;
  }

  private async pause(): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  public async down(): Promise<void> {
    throw new Error(
      'Polymarket orderbook consolidation is irreversible after legacy removal',
    );
  }
}
