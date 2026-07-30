import { MigrationInterface, QueryRunner } from 'typeorm';

export class AllowDecisionWindowExecutionSnapshots1785443400000
  implements MigrationInterface
{
  name = 'AllowDecisionWindowExecutionSnapshots1785443400000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE TABLE polymarket.btc_market_decision_execution_snapshots (
        LIKE polymarket.btc_market_execution_snapshots INCLUDING ALL
      );

      ALTER TABLE polymarket.btc_market_decision_execution_snapshots
        ADD CONSTRAINT fk_btc_market_decision_execution_snapshots_market
          FOREIGN KEY (market_id)
          REFERENCES polymarket.btc_interval_markets (market_id) ON DELETE RESTRICT,
        ADD CONSTRAINT fk_btc_market_decision_execution_snapshots_artifact
          FOREIGN KEY (artifact_id)
          REFERENCES polymarket.backfill_artifacts (artifact_id) ON DELETE RESTRICT;

      SELECT create_hypertable(
        'polymarket.btc_market_decision_execution_snapshots',
        'sampled_at',
        chunk_time_interval => INTERVAL '1 day',
        create_default_indexes => FALSE
      );

      ALTER TABLE polymarket.btc_market_decision_execution_snapshots SET (
        timescaledb.compress = true,
        timescaledb.compress_orderby = 'sampled_at ASC',
        timescaledb.compress_segmentby = 'market_id, artifact_id'
      );

      CREATE INDEX idx_btc_market_decision_execution_snapshots_artifact
        ON polymarket.btc_market_decision_execution_snapshots (artifact_id, sampled_at);

      SELECT add_compression_policy(
        'polymarket.btc_market_decision_execution_snapshots',
        INTERVAL '7 days',
        if_not_exists => true
      );

      CREATE TRIGGER trg_reject_btc_market_decision_execution_snapshot_change
        BEFORE UPDATE OR DELETE ON polymarket.btc_market_decision_execution_snapshots
        FOR EACH ROW
        EXECUTE FUNCTION polymarket.reject_btc_market_execution_snapshot_change();
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      DROP TABLE polymarket.btc_market_decision_execution_snapshots;
    `);
  }
}
