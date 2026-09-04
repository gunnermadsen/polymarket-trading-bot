import { MigrationInterface, QueryRunner } from 'typeorm';

const TARGET = 'polymarket.btc_five_minute_orderbook_snapshots';
const LEGACY = 'polymarket.orderbook_checkpoints';

export class DropLegacyPolymarketOrderbookStorage1788646001000 implements MigrationInterface {
  transaction = false;

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET lock_timeout = '5s'`);
    await queryRunner.query(`SET statement_timeout = '5min'`);

    const relations = await queryRunner.query(
      `SELECT to_regclass($1) AS target, to_regclass($2) AS legacy`,
      [TARGET, LEGACY],
    );
    if (!relations[0]?.target || !relations[0]?.legacy) {
      throw new Error('Polymarket orderbook legacy removal preconditions are not satisfied');
    }

    const cutoffs = await queryRunner.query(`
      WITH ordered AS (
        SELECT range_end,
               row_number() OVER (ORDER BY range_end) AS chunk_number,
               count(*) OVER () AS chunk_count
        FROM timescaledb_information.chunks
        WHERE hypertable_schema = 'polymarket'
          AND hypertable_name = 'orderbook_checkpoints'
      )
      SELECT range_end
      FROM ordered
      WHERE chunk_number % 16 = 0 OR chunk_number = chunk_count
      ORDER BY range_end
    `);
    for (const cutoff of cutoffs) {
      await queryRunner.query(
        `SELECT drop_chunks($1::regclass, older_than => $2::timestamptz)`,
        [LEGACY, cutoff.range_end],
      );
    }

    const remainingChunks = await queryRunner.query(`
      SELECT count(*)::integer AS count
      FROM timescaledb_information.chunks
      WHERE hypertable_schema = 'polymarket'
        AND hypertable_name = 'orderbook_checkpoints'
    `);
    if (remainingChunks[0]?.count !== 0) {
      throw new Error('Polymarket legacy orderbook chunks were not fully removed');
    }

    await queryRunner.query(`DROP TABLE ${LEGACY}`);

    const finalRelations = await queryRunner.query(
      `SELECT to_regclass($1) AS target, to_regclass($2) AS legacy`,
      [TARGET, LEGACY],
    );
    if (!finalRelations[0]?.target || finalRelations[0]?.legacy) {
      throw new Error('Polymarket orderbook legacy table removal did not complete');
    }
  }

  public async down(): Promise<void> {
    throw new Error(
      'irreversible: archived legacy orderbook checkpoints are stored in canonical Parquet',
    );
  }
}
