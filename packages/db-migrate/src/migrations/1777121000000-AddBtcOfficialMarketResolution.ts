import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddBtcOfficialMarketResolution1777121000000 implements MigrationInterface {
  name = 'AddBtcOfficialMarketResolution1777121000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE polymarket.btc_interval_markets
        ADD COLUMN IF NOT EXISTS official_outcome text,
        ADD COLUMN IF NOT EXISTS official_resolved_at timestamptz,
        ADD COLUMN IF NOT EXISTS official_winning_token_id text;
    `);
    await queryRunner.query(`
      DO $$
      BEGIN
        IF NOT EXISTS (
          SELECT 1
          FROM pg_constraint
          WHERE conname = 'chk_btc_interval_official_outcome'
            AND conrelid = 'polymarket.btc_interval_markets'::regclass
        ) THEN
          ALTER TABLE polymarket.btc_interval_markets
            ADD CONSTRAINT chk_btc_interval_official_outcome
            CHECK (official_outcome IS NULL OR official_outcome IN ('up','down'));
        END IF;
      END $$;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      ALTER TABLE polymarket.btc_interval_markets
        DROP CONSTRAINT IF EXISTS chk_btc_interval_official_outcome,
        DROP COLUMN IF EXISTS official_winning_token_id,
        DROP COLUMN IF EXISTS official_resolved_at,
        DROP COLUMN IF EXISTS official_outcome;
    `);
  }
}
