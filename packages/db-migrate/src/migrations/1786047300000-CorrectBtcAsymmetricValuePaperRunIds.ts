import { MigrationInterface, QueryRunner } from 'typeorm';

type RunIdentityCorrection = {
  processId: string;
  modelKey: string;
  previousRunId: string;
  deterministicRunId: string;
};

const corrections: readonly RunIdentityCorrection[] = [
  {
    processId: '81f82de7-002b-4ac7-814b-236c6742d81c',
    modelKey: 'btc-5m-asymmetric-core-oracle-paper-20260805-v1',
    previousRunId: '3343ae8a-515a-4aec-971e-b6aaea514d72',
    deterministicRunId: 'ba06d359-5861-5d82-8e29-c97cb1a52757',
  },
  {
    processId: '92298c8b-40e3-4c9f-ad01-61311de1b247',
    modelKey: 'btc-5m-asymmetric-core-binance-l2-paper-20260805-v1',
    previousRunId: '54eeeabc-b2b4-4acc-a3f8-ffad4b4dcd88',
    deterministicRunId: 'f2b0e5dd-eaf2-5c1e-84e7-46300826ff9a',
  },
  {
    processId: '2136d1f0-3530-4480-870f-087147e08b50',
    modelKey: 'btc-5m-asymmetric-core-paper-20260805-v1',
    previousRunId: 'fed328ca-8d9f-47e3-b8c0-889a806639d7',
    deterministicRunId: '913ffa1b-7594-55b8-a16b-b207a9631046',
  },
];

async function replaceRunIds(
  queryRunner: QueryRunner,
  sourceKey: 'previousRunId' | 'deterministicRunId',
  targetKey: 'previousRunId' | 'deterministicRunId',
): Promise<void> {
  await queryRunner.query(`SET LOCAL lock_timeout = '3s';`);
  await queryRunner.query(`LOCK TABLE polymarket.trading_processes IN SHARE ROW EXCLUSIVE MODE;`);

  for (const correction of corrections) {
    const parameters = [
      correction.processId,
      correction.modelKey,
      correction[sourceKey],
      correction[targetKey],
    ];
    const [eligible]: Array<{ count: string }> = await queryRunner.query(
      `SELECT count(*)::text AS count
         FROM polymarket.trading_processes process
        WHERE process.process_id = $1::uuid
          AND process.status = 'created'
          AND NOT process.enabled
          AND process.heartbeat_at IS NULL
          AND process.config #>> '{execution,mode}' = 'paper'
          AND process.config #>> '{execution,live_capital}' = 'false'
          AND process.metadata ->> 'model_key' = $2
          AND process.metadata ->> 'planned_run_id' = $3
          AND NOT EXISTS (
            SELECT 1
              FROM polymarket.trading_process_events event
             WHERE event.process_id = process.process_id
          );`,
      parameters.slice(0, 3),
    );
    if (Number(eligible.count) !== 1) {
      throw new Error(
        `refusing to correct mutated asymmetric paper process ${correction.processId}`,
      );
    }

    await queryRunner.query(
      `UPDATE polymarket.trading_processes process
          SET metadata = jsonb_set(
                process.metadata,
                '{planned_run_id}',
                to_jsonb($4::text),
                false
              ),
              updated_at = now()
        WHERE process.process_id = $1::uuid
          AND process.status = 'created'
          AND NOT process.enabled
          AND process.heartbeat_at IS NULL
          AND process.config #>> '{execution,mode}' = 'paper'
          AND process.config #>> '{execution,live_capital}' = 'false'
          AND process.metadata ->> 'model_key' = $2
          AND process.metadata ->> 'planned_run_id' = $3
          AND NOT EXISTS (
            SELECT 1
              FROM polymarket.trading_process_events event
             WHERE event.process_id = process.process_id
          );`,
      parameters,
    );

    const [updated]: Array<{ count: string }> = await queryRunner.query(
      `SELECT count(*)::text AS count
         FROM polymarket.trading_processes
        WHERE process_id = $1::uuid
          AND metadata ->> 'planned_run_id' = $2;`,
      [correction.processId, correction[targetKey]],
    );
    if (Number(updated.count) !== 1) {
      throw new Error(
        `asymmetric paper process run-id correction failed ${correction.processId}`,
      );
    }
  }
}

export class CorrectBtcAsymmetricValuePaperRunIds1786047300000
  implements MigrationInterface
{
  name = 'CorrectBtcAsymmetricValuePaperRunIds1786047300000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await replaceRunIds(queryRunner, 'previousRunId', 'deterministicRunId');
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await replaceRunIds(queryRunner, 'deterministicRunId', 'previousRunId');
  }
}
