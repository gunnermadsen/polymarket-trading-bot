import { MigrationInterface, QueryRunner } from 'typeorm';

export class OptimizePolymarketTradePnlReadPaths1777108000000 implements MigrationInterface {
  name = 'OptimizePolymarketTradePnlReadPaths1777108000000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_trades_asset_ts_desc
        ON polymarket.wallet_trades (asset, timestamp_utc DESC)
        INCLUDE (price);
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_wallet_trades_wallet_asset_ts
        ON polymarket.wallet_trades (proxy_wallet, asset, timestamp_utc)
        INCLUDE (side, price, cash_value, trade_id);
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_orders_request_signal_id
        ON polymarket.orders (((raw_payload #>> '{request,signal_id}')::uuid))
        WHERE raw_payload #>> '{request,signal_id}' IS NOT NULL;
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_orders_whale_exit_position_source_created
        ON polymarket.orders (
          (raw_payload #>> '{request,metadata,position_id}'),
          (raw_payload #>> '{request,metadata,exit_source_trade_id}'),
          created_at DESC
        )
        WHERE raw_payload #>> '{request,metadata,purpose}' = 'whale_led_exit';
    `);

    await queryRunner.query(`
      CREATE INDEX IF NOT EXISTS idx_poly_trade_positions_open_mark
        ON polymarket.trade_positions (token_id, latest_mark_timestamp, entry_timestamp DESC)
        WHERE status IN ('open', 'partially_closed') AND open_size > 0;
    `);
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_trade_positions_open_mark;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_orders_whale_exit_position_source_created;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_orders_request_signal_id;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_wallet_trades_wallet_asset_ts;`);
    await queryRunner.query(`DROP INDEX IF EXISTS polymarket.idx_poly_wallet_trades_asset_ts_desc;`);
  }
}
