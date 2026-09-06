import { MigrationInterface, QueryRunner } from 'typeorm';

const legacySources = [
  {
    schema: 'polymarket',
    table: 'btc_market_execution_snapshots',
    watermark: '2026-08-01T23:59:59.750Z',
  },
  {
    schema: 'polymarket',
    table: 'btc_market_decision_execution_snapshots',
    watermark: '2026-08-10T00:57:20.000Z',
  },
];

export class ConsolidateBtcExecutionSnapshotStorage1788648200000
  implements MigrationInterface
{
  name = 'ConsolidateBtcExecutionSnapshotStorage1788648200000';
  transaction = false;

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET lock_timeout = '5s'`);
    await queryRunner.query(`SET statement_timeout = '5min'`);

    const activeJobs = await queryRunner.query(`
      SELECT 1
      FROM ingester.backfill_jobs
      WHERE strategy_key = 'polymarket_btc_five_minute_execution_snapshots_backfill'
        AND status IN ('queued', 'running', 'stopping')
      LIMIT 1
    `);
    if (activeJobs.length !== 0) {
      throw new Error(
        'Polymarket execution-snapshot backfills must be quiescent before table cleanup',
      );
    }

    const canonical = await queryRunner.query(`
      SELECT 1
      FROM timescaledb_information.hypertables
      WHERE hypertable_schema = 'polymarket'
        AND hypertable_name = 'btc_market_capacity_execution_snapshots'
      LIMIT 1
    `);
    if (canonical.length !== 1) {
      throw new Error(
        'canonical Polymarket capacity execution-snapshot hypertable is missing',
      );
    }

    for (const source of legacySources) {
      const relation = `${source.schema}.${source.table}`;
      const advanced = await queryRunner.query(
        `SELECT sampled_at FROM ${relation}
         WHERE sampled_at > $1::timestamptz
         ORDER BY sampled_at
         LIMIT 1`,
        [source.watermark],
      );
      if (advanced.length !== 0) {
        throw new Error(`${relation} advanced beyond its archived watermark`);
      }
      const cutoffs = await queryRunner.query(
        `WITH ordered AS (
           SELECT range_end,
                  row_number() OVER (ORDER BY range_end) AS chunk_number,
                  count(*) OVER () AS chunk_count
           FROM timescaledb_information.chunks
           WHERE hypertable_schema = $1 AND hypertable_name = $2
         )
         SELECT range_end FROM ordered
         WHERE chunk_number % 16 = 0 OR chunk_number = chunk_count
         ORDER BY range_end`,
        [source.schema, source.table],
      );
      for (const cutoff of cutoffs) {
        await queryRunner.query(
          `SELECT drop_chunks($1::regclass, older_than => $2::timestamptz)`,
          [relation, cutoff.range_end],
        );
      }
      const remaining = await queryRunner.query(
        `SELECT count(*)::integer AS count
         FROM timescaledb_information.chunks
         WHERE hypertable_schema = $1 AND hypertable_name = $2`,
        [source.schema, source.table],
      );
      if (remaining[0]?.count !== 0) {
        throw new Error(`${relation} still has chunks after bounded cleanup`);
      }
    }

    await queryRunner.query(`
      DROP VIEW polymarket.btc_market_execution_snapshots_one_second;
      DROP TABLE polymarket.btc_market_decision_execution_snapshots;
      DROP TABLE polymarket.btc_market_execution_snapshots;

      CREATE VIEW polymarket.btc_market_execution_snapshots AS
      SELECT
        snapshot.market_id, snapshot.sampled_at, snapshot.artifact_id,
        snapshot.schema_version, snapshot.up_source_row_number,
        snapshot.up_source_timestamp, snapshot.up_provider_received_at,
        snapshot.up_best_bid, snapshot.up_best_ask, snapshot.up_best_bid_size,
        snapshot.up_best_ask_size, snapshot.up_bid_depth, snapshot.up_ask_depth,
        snapshot.up_ask_vwap_1, snapshot.up_ask_vwap_5,
        snapshot.up_ask_vwap_10, snapshot.up_imbalance,
        snapshot.down_source_row_number, snapshot.down_source_timestamp,
        snapshot.down_provider_received_at, snapshot.down_best_bid,
        snapshot.down_best_ask, snapshot.down_best_bid_size,
        snapshot.down_best_ask_size, snapshot.down_bid_depth,
        snapshot.down_ask_depth, snapshot.down_ask_vwap_1,
        snapshot.down_ask_vwap_5, snapshot.down_ask_vwap_10,
        snapshot.down_imbalance, snapshot.quality_flags, snapshot.created_at
      FROM polymarket.btc_market_capacity_execution_snapshots snapshot
      OFFSET 0;

      CREATE VIEW polymarket.btc_market_decision_execution_snapshots AS
      SELECT snapshot.*
      FROM polymarket.btc_market_execution_snapshots snapshot
      JOIN polymarket.btc_interval_markets market USING (market_id)
      WHERE extract(epoch FROM snapshot.sampled_at - market.window_start)::integer
              BETWEEN 90 AND 140
        AND mod(
          extract(epoch FROM snapshot.sampled_at - market.window_start)::integer,
          5
        ) = 0;

      CREATE VIEW polymarket.btc_market_execution_snapshots_one_second AS
      SELECT
        snapshot.market_id,
        snapshot.sampled_at,
        snapshot.artifact_id,
        snapshot.schema_version,
        snapshot.up_source_row_number,
        snapshot.up_source_timestamp,
        snapshot.up_provider_received_at,
        snapshot.up_best_bid,
        snapshot.up_best_ask,
        snapshot.up_best_bid_size,
        snapshot.up_best_ask_size,
        snapshot.up_bid_depth,
        snapshot.up_ask_depth,
        snapshot.up_ask_vwap_1,
        snapshot.up_ask_vwap_5,
        snapshot.up_ask_vwap_10,
        snapshot.up_imbalance,
        snapshot.down_source_row_number,
        snapshot.down_source_timestamp,
        snapshot.down_provider_received_at,
        snapshot.down_best_bid,
        snapshot.down_best_ask,
        snapshot.down_best_bid_size,
        snapshot.down_best_ask_size,
        snapshot.down_bid_depth,
        snapshot.down_ask_depth,
        snapshot.down_ask_vwap_1,
        snapshot.down_ask_vwap_5,
        snapshot.down_ask_vwap_10,
        snapshot.down_imbalance,
        snapshot.quality_flags,
        snapshot.created_at
      FROM polymarket.btc_market_execution_snapshots snapshot
      WHERE extract(milliseconds FROM snapshot.sampled_at)::integer % 1000 = 0;
    `);
  }

  public async down(): Promise<void> {
    throw new Error(
      'irreversible: legacy Polymarket execution snapshots are stored in validated canonical Parquet',
    );
  }
}
