import { MigrationInterface, QueryRunner } from 'typeorm';

export class SeedCanonicalIngesterProfiles1789286400000
  implements MigrationInterface
{
  name = 'SeedCanonicalIngesterProfiles1789286400000';

  public async up(queryRunner: QueryRunner): Promise<void> {
    await queryRunner.query(`
      INSERT INTO ingester.profiles (
        strategy_key,
        config_schema_version,
        config,
        desired_state,
        observed_state,
        health_status
      ) VALUES
        (
          'binance_futures_btcusdt_open_interest', 1,
          '{"period":"5m","symbol":"BTCUSDT","request_limit":500,"rest_base_url":"https://fapi.binance.com","overlap_periods":2,"poll_interval_seconds":60,"artifact_window_seconds":86400,"request_timeout_seconds":10,"startup_lookback_periods":288}'::jsonb,
          'running', 'stopped', 'unknown'
        ),
        (
          'binance_spot_btcusdt_aggregate_trades', 1,
          '{"batch_size":500,"rest_base_url":"https://data-api.binance.vision","websocket_url":"wss://stream.binance.com:9443/ws/btcusdt@aggTrade","rest_page_limit":1000,"flush_interval_ms":250,"read_idle_timeout_ms":40000,"reconnect_max_delay_ms":30000,"artifact_window_seconds":3600,"recovery_overlap_records":100,"reconnect_initial_delay_ms":1000}'::jsonb,
          'stopped', 'stopped', 'unknown'
        ),
        (
          'binance_spot_btcusdt_l2_snapshots', 1,
          '{"top_n":20,"symbol":"BTCUSDT","websocket_url":"wss://stream.binance.com:443/ws/btcusdt@depth@100ms","rest_depth_url":"https://api.binance.com/api/v3/depth","read_timeout_ms":45000,"ping_interval_ms":15000,"reconnect_max_ms":30000,"rest_depth_limit":5000,"connect_timeout_ms":10000,"sample_interval_ms":1000,"max_buffered_levels":200000,"bootstrap_timeout_ms":10000,"max_buffered_updates":4096,"reconnect_initial_ms":250,"artifact_window_seconds":3600,"max_book_levels_per_side":100000}'::jsonb,
          'running', 'stopped', 'unknown'
        ),
        (
          'binance_spot_btcusdt_one_second_ohlcv', 1,
          '{"batch_size":250,"rest_base_url":"https://data-api.binance.vision","websocket_url":"wss://stream.binance.com:9443/ws/btcusdt@kline_1s","rest_page_limit":1000,"flush_interval_ms":250,"read_idle_timeout_ms":40000,"reconnect_max_delay_ms":30000,"artifact_window_seconds":3600,"recovery_overlap_seconds":60,"reconnect_initial_delay_ms":1000}'::jsonb,
          'running', 'stopped', 'unknown'
        ),
        (
          'chainlink_btcusd_one_minute_ohlc', 1,
          '{"symbol":"BTCUSD","base_url":"https://priceapi.dataengine.chain.link","resolution":"1m","overlap_minutes":5,"poll_interval_seconds":15,"request_window_minutes":1440,"artifact_window_seconds":3600,"request_timeout_seconds":15,"startup_lookback_minutes":1440}'::jsonb,
          'stopped', 'stopped', 'unknown'
        ),
        (
          'chainlink_btcusd_reference_price', 1,
          '{"feed_id":"0x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8","page_limit":100,"rest_base_url":"https://api.dataengine.chain.link","overlap_seconds":5,"poll_interval_ms":1000,"max_pages_per_poll":8,"retry_max_delay_ms":2000,"max_request_attempts":3,"recent_window_seconds":300,"retry_initial_delay_ms":250,"artifact_window_seconds":3600,"request_timeout_seconds":10}'::jsonb,
          'stopped', 'stopped', 'unknown'
        ),
        (
          'polygon_chainlink_btcusd_oracle', 1,
          '{"rpc_url":"https://polygon-bor-rpc.publicnode.com","overlap_blocks":256,"confirmation_depth":128,"feed_proxy_address":"0xc907e116054ad103354f2d350fd2514433d57f6f","archive_log_rpc_url":"https://tenderly.rpc.polygon.community","maximum_block_range":30000,"poll_interval_seconds":15,"artifact_window_seconds":3600,"request_timeout_seconds":30,"startup_lookback_blocks":43200}'::jsonb,
          'running', 'stopped', 'unknown'
        ),
        (
          'polymarket_btc_five_minute_market_contracts', 1,
          '{"gamma_base_url":"https://gamma-api.polymarket.com","overlap_windows":2,"lookahead_windows":1,"poll_interval_seconds":5,"artifact_window_seconds":3600,"request_timeout_seconds":10,"startup_lookback_windows":12}'::jsonb,
          'running', 'stopped', 'unknown'
        ),
        (
          'polymarket_btc_five_minute_orderbooks', 1,
          '{"top_n":20,"gamma_api_url":"https://gamma-api.polymarket.com","websocket_url":"wss://ws-subscriptions-clob.polymarket.com/ws/market","pong_timeout_ms":25000,"read_timeout_ms":40000,"gamma_refresh_ms":5000,"lookback_windows":1,"ping_interval_ms":10000,"reconnect_max_ms":30000,"contract_grace_ms":30000,"lookahead_windows":1,"successor_lead_ms":30000,"connect_timeout_ms":10000,"sample_interval_ms":1000,"max_levels_per_side":10000,"bootstrap_timeout_ms":15000,"reconnect_initial_ms":250,"artifact_window_seconds":3600}'::jsonb,
          'running', 'stopped', 'unknown'
        ),
        (
          'polymarket_btc_five_minute_resolutions', 1,
          '{"clob_base_url":"https://clob.polymarket.com","websocket_url":"wss://ws-subscriptions-clob.polymarket.com/ws/market","gamma_base_url":"https://gamma-api.polymarket.com","pong_timeout_ms":25000,"read_timeout_ms":40000,"ping_interval_ms":10000,"reconnect_max_ms":30000,"lookahead_windows":1,"retry_max_seconds":300,"connect_timeout_ms":10000,"clob_market_endpoint":"markets","reconnect_initial_ms":250,"poll_interval_seconds":5,"retry_initial_seconds":30,"artifact_window_seconds":3600,"request_timeout_seconds":10,"startup_lookback_windows":288,"discovery_refresh_seconds":5,"gamma_fallback_grace_seconds":900}'::jsonb,
          'running', 'stopped', 'unknown'
        ),
        (
          'polymarket_chainlink_btcusd_twap', 1,
          '{"symbol":"btc/usd","websocket_url":"wss://ws-live-data.polymarket.com","ping_interval_ms":5000,"reconnect_max_ms":30000,"write_timeout_ms":5000,"connect_timeout_ms":10000,"reconnect_initial_ms":250,"artifact_window_seconds":3600,"stream_stale_timeout_ms":30000,"initial_stream_timeout_ms":20000}'::jsonb,
          'running', 'stopped', 'unknown'
        )
      ON CONFLICT (strategy_key) DO NOTHING;
    `);
  }

  public async down(): Promise<void> {
    throw new Error(
      'SeedCanonicalIngesterProfiles1789286400000 is irreversible because profiles may own runtime data.',
    );
  }
}
