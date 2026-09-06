import { MigrationInterface, QueryRunner } from 'typeorm';

const featureColumns = `
  feature.symbol, feature.second_start, feature.source_event_timestamp,
  feature.provider_received_at, feature.available_at, feature.source_update_id,
  feature.midpoint, feature.microprice, feature.spread_bps,
  feature.bid_depth_5, feature.ask_depth_5, feature.imbalance_5,
  feature.bid_depth_10, feature.ask_depth_10, feature.imbalance_10,
  feature.bid_depth_20, feature.ask_depth_20, feature.imbalance_20,
  feature.bid_depth_slope_20, feature.ask_depth_slope_20,
  feature.bid_depth_concentration_20, feature.ask_depth_concentration_20,
  feature.bid_quote_replenishment_1s, feature.ask_quote_replenishment_1s,
  feature.bid_quote_churn_1s, feature.ask_quote_churn_1s,
  feature.midpoint_change_bps_1s, feature.spread_bps_delta_1s,
  feature.depth_20_change_bps_1s, feature.imbalance_20_delta_1s,
  feature.midpoint_change_bps_5s, feature.spread_bps_delta_5s,
  feature.depth_20_change_bps_5s, feature.imbalance_20_delta_5s,
  feature.midpoint_change_bps_15s, feature.spread_bps_delta_15s,
  feature.depth_20_change_bps_15s, feature.imbalance_20_delta_15s,
  feature.midpoint_change_bps_30s, feature.spread_bps_delta_30s,
  feature.depth_20_change_bps_30s, feature.imbalance_20_delta_30s,
  feature.midpoint_change_bps_60s, feature.spread_bps_delta_60s,
  feature.depth_20_change_bps_60s, feature.imbalance_20_delta_60s`;

export class DropLegacyBinanceL2FeatureStorage1788648100000
  implements MigrationInterface
{
  name = 'DropLegacyBinanceL2FeatureStorage1788648100000';
  transaction = false;

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`SET lock_timeout = '5s'`);
    await queryRunner.query(`SET statement_timeout = '5min'`);

    const activeJobs = await queryRunner.query(`
      SELECT 1
      FROM ingester.backfill_jobs
      WHERE strategy_key IN (
        'binance_spot_btcusdt_l2_one_second_features_backfill',
        'coinapi_binance_spot_btcusdt_l2_one_second_features_backfill',
        'binance_futures_btcusdt_l2_one_second_features_backfill'
      )
        AND status IN ('queued', 'running', 'stopping')
      LIMIT 1
    `);
    if (activeJobs.length !== 0) {
      throw new Error('Binance L2 feature backfills must be quiescent before table cutover');
    }

    for (const source of [
      'polymarket.binance_spot_btcusdt_l2_one_second_features',
      'polymarket.binance_btcusdt_l2_one_second_features',
    ]) {
      const unarchived = await queryRunner.query(
        `SELECT second_start FROM ${source}
         WHERE second_start > $1::timestamptz
         ORDER BY second_start LIMIT 1`,
        ['2026-08-01T23:59:59.000Z'],
      );
      if (unarchived.length !== 0) {
        throw new Error(`${source} advanced beyond its archived watermark`);
      }
      const [schema, table] = source.split('.');
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
        [schema, table],
      );
      for (const cutoff of cutoffs) {
        await queryRunner.query(
          `SELECT drop_chunks($1::regclass, older_than => $2::timestamptz)`,
          [source, cutoff.range_end],
        );
      }
      const remaining = await queryRunner.query(
        `SELECT count(*)::integer AS count
         FROM timescaledb_information.chunks
         WHERE hypertable_schema = $1 AND hypertable_name = $2`,
        [schema, table],
      );
      if (remaining[0]?.count !== 0) {
        throw new Error(`${source} still has chunks after bounded cleanup`);
      }
    }

    await queryRunner.query(`
      DROP VIEW polymarket.binance_spot_btcusdt_l2_training_features;
      DROP VIEW polymarket.binance_btcusdt_l2_training_features;

      CREATE VIEW polymarket.binance_spot_btcusdt_l2_training_features AS
      SELECT ${featureColumns}
      FROM market_data.binance_spot_btcusdt_l2_one_second_features feature
      JOIN ingester.backfill_artifacts artifact
        ON artifact.artifact_id = feature.artifact_id
       AND artifact.status = 'completed';

      CREATE VIEW polymarket.binance_btcusdt_l2_training_features AS
      SELECT ${featureColumns}
      FROM market_data.binance_futures_btcusdt_l2_one_second_features feature
      JOIN ingester.backfill_artifacts artifact
        ON artifact.artifact_id = feature.artifact_id
       AND artifact.status = 'completed';

      DROP TABLE polymarket.binance_spot_btcusdt_l2_one_second_features_staging;
      DROP TABLE polymarket.binance_spot_btcusdt_l2_one_second_features;
      DROP TABLE polymarket.binance_btcusdt_l2_one_second_features_staging;
      DROP TABLE polymarket.binance_btcusdt_l2_one_second_features;
    `);
  }

  public async down(): Promise<void> {
    throw new Error(
      'irreversible: legacy Binance L2 features are stored in validated canonical Parquet',
    );
  }
}
