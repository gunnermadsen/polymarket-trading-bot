import { MigrationInterface, QueryRunner } from 'typeorm';

const TARGET = 'polymarket.btc_five_minute_orderbook_snapshots';
const LEGACY = 'polymarket.orderbook_checkpoints';

export class DropLegacyPolymarketOrderbookStorage1788646001000 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET LOCAL lock_timeout = '5s'`);
    await queryRunner.query(`SET LOCAL statement_timeout = '5min'`);

    const relations = await queryRunner.query(
      `SELECT to_regclass($1) AS target, to_regclass($2) AS legacy`,
      [TARGET, LEGACY],
    );
    if (!relations[0]?.target || !relations[0]?.legacy) {
      throw new Error('Polymarket orderbook legacy removal preconditions are not satisfied');
    }

    await queryRunner.query(`LOCK TABLE ${LEGACY} IN ACCESS EXCLUSIVE MODE NOWAIT`);
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
