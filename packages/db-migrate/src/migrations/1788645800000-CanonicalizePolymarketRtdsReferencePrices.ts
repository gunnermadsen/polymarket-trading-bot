import { MigrationInterface, QueryRunner } from 'typeorm';

const LEGACY = 'polymarket.reference_price_ticks';
const TARGET = 'polymarket.chainlink_btcusd_reference_prices';

export class CanonicalizePolymarketRtdsReferencePrices1788645800000
  implements MigrationInterface
{
  name = 'CanonicalizePolymarketRtdsReferencePrices1788645800000';
  transaction = false;

  public async up(queryRunner: QueryRunner): Promise<void> {
    const legacyTable = await this.exists(queryRunner, LEGACY, 'r');
    const canonicalTable = await this.exists(queryRunner, TARGET, 'r');
    if (legacyTable && !canonicalTable) {
      await queryRunner.query(`
        ALTER TABLE ${LEGACY} RENAME TO chainlink_btcusd_reference_prices;
        CREATE VIEW ${LEGACY} AS SELECT * FROM ${TARGET};
      `);
    }
    if (!(await this.exists(queryRunner, TARGET, 'r'))) {
      throw new Error(`${TARGET} is not present after canonical rename`);
    }
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

  public async down(): Promise<void> {
    throw new Error(
      'Polymarket RTDS reference-price canonical rename is irreversible',
    );
  }
}
