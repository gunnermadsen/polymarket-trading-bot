import { MigrationInterface, QueryRunner } from 'typeorm';

const SOURCE = 'market_data.polymarket_btc_five_minute_orderbook_snapshots';
const TARGET = 'polymarket.btc_five_minute_orderbook_snapshots';
const LEGACY = 'polymarket.orderbook_checkpoints';
const STRATEGY = 'polymarket_btc_five_minute_orderbooks';
const ARCHIVED_LEGACY_WATERMARK = '2026-09-03T17:26:31.000Z';

export class FinalizePolymarketOrderbookStorage1788646000000
  implements MigrationInterface
{
  name = 'FinalizePolymarketOrderbookStorage1788646000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET LOCAL lock_timeout = '5s'`);
    await queryRunner.query(`SET LOCAL statement_timeout = '5min'`);

    const profile = await queryRunner.query(
      `SELECT desired_state, observed_state
       FROM ingester.profiles
       WHERE strategy_key = $1`,
      [STRATEGY],
    );
    if (
      profile.length !== 1 ||
      profile[0].desired_state !== 'stopped' ||
      profile[0].observed_state !== 'stopped'
    ) {
      throw new Error(`${STRATEGY} must be fully stopped before table cutover`);
    }

    const relations = await queryRunner.query(
      `SELECT to_regclass($1) AS source,
              to_regclass($2) AS target,
              to_regclass($3) AS legacy`,
      [SOURCE, TARGET, LEGACY],
    );
    if (!relations[0]?.source || relations[0]?.target || !relations[0]?.legacy) {
      throw new Error(
        `expected ${SOURCE} and ${LEGACY} as sole pre-cutover tables`,
      );
    }

    const unarchived = await queryRunner.query(
      `SELECT source_timestamp
       FROM ${LEGACY}
       WHERE source_timestamp > $1::timestamptz
       ORDER BY source_timestamp ASC
       LIMIT 1`,
      [ARCHIVED_LEGACY_WATERMARK],
    );
    if (unarchived.length !== 0) {
      throw new Error(
        `${LEGACY} advanced beyond archived watermark ${ARCHIVED_LEGACY_WATERMARK}`,
      );
    }

    await queryRunner.query(`LOCK TABLE ${SOURCE} IN ACCESS EXCLUSIVE MODE NOWAIT`);
    await queryRunner.query(`LOCK TABLE ${LEGACY} IN ACCESS EXCLUSIVE MODE NOWAIT`);
    await queryRunner.query(`ALTER TABLE ${SOURCE} SET SCHEMA polymarket`);
    await queryRunner.query(`
      ALTER TABLE polymarket.polymarket_btc_five_minute_orderbook_snapshots
      RENAME TO btc_five_minute_orderbook_snapshots
    `);
    await queryRunner.query(`DROP TABLE ${LEGACY}`);

    const finalRelations = await queryRunner.query(
      `SELECT to_regclass($1) AS source,
              to_regclass($2) AS target,
              to_regclass($3) AS legacy`,
      [SOURCE, TARGET, LEGACY],
    );
    if (
      finalRelations[0]?.source ||
      !finalRelations[0]?.target ||
      finalRelations[0]?.legacy
    ) {
      throw new Error(`Polymarket orderbook cutover did not reach one physical table`);
    }
  }

  public async down(): Promise<void> {
    throw new Error(
      'irreversible: archived legacy orderbook checkpoints are stored in canonical Parquet',
    );
  }
}
