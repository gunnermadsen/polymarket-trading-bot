import { MigrationInterface, QueryRunner } from 'typeorm';

const STRATEGY_KEY = 'chainlink_btcusd_reference_price';
const REASON_CODE = 'chainlink_reference_second_missing';

export class TerminalizeChainlinkReferencePriceCadenceGaps1786381760000
  implements MigrationInterface
{
  name = 'TerminalizeChainlinkReferencePriceCadenceGaps1786381760000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      UPDATE ingester.data_gaps
      SET status = 'unrecoverable',
          resolved_at = GREATEST(
            clock_timestamp(),
            COALESCE(repair_started_at, detected_at)
          ),
          resolution_code = 'not_a_gap_irregular_provider_cadence',
          resolution_message =
            'Terminalized because Chainlink reference-price reports have irregular cadence; a missing integer second does not establish missing provider data.',
          updated_at = GREATEST(clock_timestamp(), created_at)
      WHERE strategy_key = '${STRATEGY_KEY}'
        AND reason_code = '${REASON_CODE}'
        AND status IN ('open', 'repairing');
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DO $$
      BEGIN
        RAISE EXCEPTION
          'refusing to reopen Chainlink reference-price cadence gaps because their prior operational state is not recoverable';
      END $$;
    `);
  }
}
