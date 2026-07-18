import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddBtcAdmissionCandidateHistoryIndex1784391000000 implements MigrationInterface {
  name = 'AddBtcAdmissionCandidateHistoryIndex1784391000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_btc_decisions_process_config_candidate
      ON polymarket.btc_strategy_decisions (
        process_id,
        config_hash,
        market_id,
        decision_at,
        decision_id
      )
      INCLUDE (outcome, fair_probability)
      WHERE process_id IS NOT NULL
        AND action = 'buy';
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DROP INDEX IF EXISTS polymarket.idx_btc_decisions_process_config_candidate;
    `);
  }
}
