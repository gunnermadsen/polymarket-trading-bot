import { MigrationInterface, QueryRunner } from 'typeorm';

export class AddCanonicalFillIdentity1784661463000 implements MigrationInterface {
  name = 'AddCanonicalFillIdentity1784661463000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE polymarket.fill_identities (
        fill_id uuid PRIMARY KEY,
        process_id uuid,
        timestamp_utc timestamptz NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now()
      );
    `);

    // Deployment keeps the legacy writer quiesced until the new binary starts. Bound this lock so
    // an unexpected secondary writer fails the migration instead of creating an unbounded wait.
    await queryRunner.query(`SET LOCAL lock_timeout = '5s';`);
    await queryRunner.query(`LOCK TABLE polymarket.fills IN SHARE ROW EXCLUSIVE MODE;`);

    // A duplicate deterministic fill_id is already an accounting-integrity breach. The primary
    // key intentionally makes this backfill fail instead of selecting an arbitrary historical row.
    // These compact rows remain as anti-replay tombstones after the authoritative fill expires.
    await queryRunner.query(`
      INSERT INTO polymarket.fill_identities (
        fill_id, process_id, timestamp_utc, created_at
      )
      SELECT
        fill_id, process_id, timestamp_utc, created_at
      FROM polymarket.fills;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP TABLE IF EXISTS polymarket.fill_identities;`);
  }
}
