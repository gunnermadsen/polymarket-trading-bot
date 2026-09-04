import { createHash } from 'node:crypto';
import { MigrationInterface, QueryRunner } from 'typeorm';

const LEGACY = 'polymarket.orderbook_checkpoints';
const MOVED = 'polymarket.btc_five_minute_orderbook_snapshots';
const ORIGINAL = 'market_data.polymarket_btc_five_minute_orderbook_snapshots';
const STRATEGY = 'polymarket_btc_five_minute_orderbooks';

export class RestorePreConsolidationOrderbooks1788645900000
  implements MigrationInterface
{
  name = 'RestorePreConsolidationOrderbooks1788645900000';
  transaction = false;

  public async up(queryRunner: QueryRunner): Promise<void> {
    if (!(await this.isTable(queryRunner, MOVED))) return;
    if (!(await this.isTable(queryRunner, LEGACY))) {
      throw new Error(`${LEGACY} must remain available for the drain migration`);
    }

    await queryRunner.query(
      'SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0',
    );
    try {
      if (await this.hasMigrationArtifacts(queryRunner)) {
        const bounds = await queryRunner.query(`
          SELECT
            (SELECT source_timestamp FROM ${LEGACY}
             ORDER BY source_timestamp ASC LIMIT 1) AS minimum,
            (SELECT source_timestamp FROM ${LEGACY}
             ORDER BY source_timestamp DESC LIMIT 1) AS maximum
        `);
        if (bounds[0]?.minimum && bounds[0]?.maximum) {
          for (const start of this.hourStarts(
            new Date(bounds[0].minimum),
            new Date(bounds[0].maximum),
          )) {
            await this.removeMigrationRows(
              queryRunner,
              this.legacyArtifactId(start),
            );
            await this.pause();
          }
        }
        await this.removeMigrationArtifacts(queryRunner);
      }
      await this.restoreOriginalRelation(queryRunner);
    } finally {
      await queryRunner.query(
        'RESET timescaledb.max_tuples_decompressed_per_dml_transaction',
      );
    }
  }

  private async hasMigrationArtifacts(queryRunner: QueryRunner): Promise<boolean> {
    const result = await queryRunner.query(`
      SELECT EXISTS (
        SELECT 1 FROM ingester.capture_artifacts
        WHERE strategy_key = '${STRATEGY}'
          AND content_sha256 = encode(digest(convert_to(
            'legacy-orderbook-checkpoints:' || artifact_id::text, 'UTF8'
          ), 'sha256'), 'hex')
        LIMIT 1
      ) AS present
    `);
    return Boolean(result[0]?.present);
  }

  private async removeMigrationRows(
    queryRunner: QueryRunner,
    artifactId: string,
  ): Promise<void> {
    await queryRunner.startTransaction();
    try {
      await queryRunner.query(
        'SET LOCAL session_replication_role = replica',
      );
      await queryRunner.query(`
        DELETE FROM ${MOVED}
        WHERE capture_artifact_id = $1::uuid
          AND sampling_policy ->> 'legacy_event_driven_checkpoint' = 'true'
      `, [artifactId]);
      await queryRunner.commitTransaction();
    } catch (error) {
      await queryRunner.rollbackTransaction();
      throw error;
    }
  }

  private async removeMigrationArtifacts(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.startTransaction();
    try {
      await queryRunner.query(
        'SET LOCAL session_replication_role = replica',
      );
      await queryRunner.query(`
        DELETE FROM ingester.capture_artifacts
        WHERE strategy_key = '${STRATEGY}'
          AND content_sha256 = encode(digest(convert_to(
            'legacy-orderbook-checkpoints:' || artifact_id::text, 'UTF8'
          ), 'sha256'), 'hex')
          AND NOT EXISTS (
            SELECT 1 FROM ${MOVED} snapshot
            WHERE snapshot.strategy_key = capture_artifacts.strategy_key
              AND snapshot.capture_artifact_id = capture_artifacts.artifact_id
          )
      `);
      await queryRunner.commitTransaction();
    } catch (error) {
      await queryRunner.rollbackTransaction();
      throw error;
    }
  }

  private async restoreOriginalRelation(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ${MOVED} SET (timescaledb.compress = false);

      ALTER TABLE ${MOVED}
        DROP CONSTRAINT IF EXISTS
          chk_market_data_polymarket_btc_five_minute_orderbook_prices,
        DROP CONSTRAINT IF EXISTS
          chk_market_data_polymarket_btc_five_minute_orderbook_book,
        DROP CONSTRAINT IF EXISTS
          chk_market_data_polymarket_btc_five_minute_orderbook_time;
      ALTER TABLE ${MOVED}
        ADD CONSTRAINT
          chk_market_data_polymarket_btc_five_minute_orderbook_prices
        CHECK (
          tick_size > 0 AND tick_size < 1
          AND (best_bid IS NULL OR (best_bid > 0 AND best_bid < 1))
          AND (best_ask IS NULL OR (best_ask > 0 AND best_ask < 1))
          AND (best_bid IS NULL OR best_ask IS NULL OR best_bid < best_ask)
        ) NOT VALID,
        ADD CONSTRAINT
          chk_market_data_polymarket_btc_five_minute_orderbook_book
        CHECK (
          octet_length(bids::text) <= 262144
          AND octet_length(asks::text) <= 262144
          AND market_data.is_valid_polymarket_btc_five_minute_book(
            bids, asks, bid_depth, ask_depth, best_bid, best_ask,
            (sampling_policy ->> 'top_n')::integer
          )
        ) NOT VALID,
        ADD CONSTRAINT
          chk_market_data_polymarket_btc_five_minute_orderbook_time
        CHECK (
          provider_available_at = source_timestamp
          AND received_at <= sampled_at
        ) NOT VALID;

      ALTER TABLE ${MOVED}
        DROP COLUMN IF EXISTS legacy_checkpoint_id,
        DROP COLUMN IF EXISTS legacy_source_payload,
        DROP COLUMN IF EXISTS bootstrap_source,
        DROP COLUMN IF EXISTS integrity_status;

      ALTER TABLE ${MOVED} SET (
        timescaledb.compress = true,
        timescaledb.compress_segmentby =
          'market_id, token_id, sampling_policy_sha256, strategy_key, capture_artifact_id',
        timescaledb.compress_orderby =
          'sampled_at, source_timestamp, ingest_sequence'
      );

      DROP VIEW ${ORIGINAL};
      ALTER TABLE ${MOVED}
        RENAME TO polymarket_btc_five_minute_orderbook_snapshots;
      ALTER TABLE polymarket.polymarket_btc_five_minute_orderbook_snapshots
        SET SCHEMA market_data;
    `);
  }

  private async isTable(
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

  private hourStarts(minimum: Date, maximum: Date): Date[] {
    const result: Date[] = [];
    let start = new Date(minimum);
    while (start <= maximum) {
      result.push(start);
      start = new Date(start.getTime() + 3_600_000);
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
    throw new Error('The abandoned table consolidation cannot be reapplied');
  }
}
