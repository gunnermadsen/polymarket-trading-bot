import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'chainlink_btcusd_reference_price';

export class CorrectChainlinkReferencePricePageLimit1786381750000
  implements MigrationInterface
{
  name = 'CorrectChainlinkReferencePricePageLimit1786381750000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      UPDATE ingester.profiles
      SET config = jsonb_set(config, '{page_limit}', '100'::jsonb, false),
          desired_generation = desired_generation + 1,
          updated_at = clock_timestamp()
      WHERE strategy_key = '${STRATEGY_KEY}'
        AND CASE
          WHEN jsonb_typeof(config -> 'page_limit') = 'number'
            THEN (config ->> 'page_limit')::numeric > 100
          ELSE false
        END;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        RAISE EXCEPTION
          'refusing to restore Chainlink reference-price page_limit because its prior value is not recoverable';
      END $$;
    `);
  }
}
