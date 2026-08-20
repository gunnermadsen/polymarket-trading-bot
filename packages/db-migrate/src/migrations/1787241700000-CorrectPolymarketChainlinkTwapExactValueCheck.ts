import { MigrationInterface, QueryRunner } from 'typeorm';

const TABLE_NAME = 'market_data.polymarket_chainlink_btcusd_twap';
const CONSTRAINT_NAME =
  'chk_market_data_polymarket_chainlink_btcusd_twap_value';

export class CorrectPolymarketChainlinkTwapExactValueCheck1787241700000
  implements MigrationInterface
{
  name = 'CorrectPolymarketChainlinkTwapExactValueCheck1787241700000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ${TABLE_NAME}
        DROP CONSTRAINT ${CONSTRAINT_NAME},
        ADD CONSTRAINT ${CONSTRAINT_NAME} CHECK (
          full_accuracy_value ~ '^-?[0-9]{1,29}$'
          AND twap_price * 1000000000000000000::numeric =
            full_accuracy_value::numeric
          AND twap_price > 0
        );
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE ${TABLE_NAME}
        DROP CONSTRAINT ${CONSTRAINT_NAME},
        ADD CONSTRAINT ${CONSTRAINT_NAME} CHECK (
          full_accuracy_value ~ '^-?[0-9]{1,29}$'
          AND twap_price =
            full_accuracy_value::numeric / 1000000000000000000::numeric
          AND twap_price > 0
        );
    `);
  }
}
