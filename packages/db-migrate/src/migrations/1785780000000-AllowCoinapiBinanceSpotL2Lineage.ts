import { MigrationInterface, QueryRunner } from 'typeorm';

export class AllowCoinapiBinanceSpotL2Lineage1785780000000
  implements MigrationInterface
{
  name = 'AllowCoinapiBinanceSpotL2Lineage1785780000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE OR REPLACE VIEW polymarket.binance_spot_btcusdt_l2_training_features AS
      SELECT
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
        feature.depth_20_change_bps_60s, feature.imbalance_20_delta_60s
      FROM polymarket.binance_spot_btcusdt_l2_one_second_features feature
      JOIN polymarket.backfill_artifacts artifact
        ON artifact.artifact_id = feature.artifact_id
       AND artifact.ingester_key =
         'binance_spot_btcusdt_l2_one_second_features'
       AND artifact.status = 'completed'
       AND artifact.metadata ->> 'materialization_contract' IN (
         'cryptohft-binance-spot-btcusdt-l2-features-v1',
         'coinapi-binance-spot-btcusdt-l2-snapshots-v1'
       );
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE OR REPLACE VIEW polymarket.binance_spot_btcusdt_l2_training_features AS
      SELECT
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
        feature.depth_20_change_bps_60s, feature.imbalance_20_delta_60s
      FROM polymarket.binance_spot_btcusdt_l2_one_second_features feature
      JOIN polymarket.backfill_artifacts artifact
        ON artifact.artifact_id = feature.artifact_id
       AND artifact.ingester_key =
         'binance_spot_btcusdt_l2_one_second_features'
       AND artifact.status = 'completed'
       AND artifact.metadata ->> 'materialization_contract' =
         'cryptohft-binance-spot-btcusdt-l2-features-v1';
    `);
  }
}
