WITH market_facts AS MATERIALIZED (
  SELECT
    market_id,
    max(value) FILTER (WHERE fact_type = 'opening_boundary') AS opening_boundary,
    max(value) FILTER (WHERE fact_type = 'final_price') AS final_price,
    count(*) FILTER (WHERE fact_type = 'opening_boundary') AS opening_fact_count,
    count(*) FILTER (WHERE fact_type = 'final_price') AS final_fact_count
  FROM polymarket.btc_market_reference_facts
  JOIN polymarket.backfill_artifacts fact_artifact USING (artifact_id)
  WHERE source_effective_at >= %(batch_start)s
    AND source_effective_at <= %(batch_end)s
    AND fact_type IN ('opening_boundary', 'final_price')
    AND fact_artifact.status = 'completed'
  GROUP BY market_id
),
eligible_markets AS MATERIALIZED (
  SELECT
    market.market_id,
    market.window_start,
    market.window_end,
    market.official_outcome,
    CASE WHEN market.official_outcome = 'up' THEN 1 ELSE 0 END AS label_up,
    market.min_tick_size::double precision AS min_tick_size,
    market.min_order_size::double precision AS min_order_size,
    COALESCE(market.fee_rate, 0)::double precision AS fee_rate,
    facts.opening_boundary::double precision AS opening_boundary,
    facts.final_price::double precision AS final_price
  FROM polymarket.btc_interval_markets market
  JOIN market_facts facts USING (market_id)
  WHERE market.window_start >= %(batch_start)s
    AND market.window_start < %(batch_end)s
    AND market.validation_status = 'valid'
    AND market.official_outcome IN ('up', 'down')
    AND facts.opening_fact_count = 1
    AND (%(strict_final_price_audit)s = false OR facts.final_fact_count = 1)
)
SELECT
  market.market_id,
  market.window_start,
  market.window_end,
  market.official_outcome,
  market.label_up,
  market.min_tick_size,
  market.min_order_size,
  market.fee_rate,
  market.opening_boundary,
  market.final_price,
  snapshot.sampled_at AS observed_at,
  extract(epoch FROM snapshot.sampled_at - market.window_start)::integer AS seconds_elapsed,
  kline.open_price::double precision AS btc_open,
  kline.high_price::double precision AS btc_high,
  kline.low_price::double precision AS btc_low,
  kline.close_price::double precision AS btc_close,
  kline.base_volume::double precision AS btc_base_volume,
  kline.quote_volume::double precision AS btc_quote_volume,
  kline.trade_count,
  kline.taker_buy_base_volume::double precision AS btc_taker_buy_base_volume,
  kline.taker_buy_quote_volume::double precision AS btc_taker_buy_quote_volume,
  snapshot.up_provider_received_at,
  snapshot.up_best_bid::double precision AS up_best_bid,
  snapshot.up_best_ask::double precision AS up_best_ask,
  snapshot.up_best_bid_size::double precision AS up_best_bid_size,
  snapshot.up_best_ask_size::double precision AS up_best_ask_size,
  snapshot.up_bid_depth::double precision AS up_bid_depth,
  snapshot.up_ask_depth::double precision AS up_ask_depth,
  snapshot.up_ask_vwap_1::double precision AS up_ask_vwap_1,
  snapshot.up_ask_vwap_5::double precision AS up_ask_vwap_5,
  snapshot.up_ask_vwap_10::double precision AS up_ask_vwap_10,
  snapshot.up_imbalance::double precision AS up_imbalance,
  snapshot.down_provider_received_at,
  snapshot.down_best_bid::double precision AS down_best_bid,
  snapshot.down_best_ask::double precision AS down_best_ask,
  snapshot.down_best_bid_size::double precision AS down_best_bid_size,
  snapshot.down_best_ask_size::double precision AS down_best_ask_size,
  snapshot.down_bid_depth::double precision AS down_bid_depth,
  snapshot.down_ask_depth::double precision AS down_ask_depth,
  snapshot.down_ask_vwap_1::double precision AS down_ask_vwap_1,
  snapshot.down_ask_vwap_5::double precision AS down_ask_vwap_5,
  snapshot.down_ask_vwap_10::double precision AS down_ask_vwap_10,
  snapshot.down_imbalance::double precision AS down_imbalance,
  snapshot.quality_flags
FROM eligible_markets market
JOIN polymarket.btc_market_decision_execution_snapshots snapshot
  ON snapshot.market_id = market.market_id
 AND snapshot.sampled_at >= %(batch_start)s
 AND snapshot.sampled_at < %(batch_end)s
 AND snapshot.sampled_at >= market.window_start
 AND snapshot.sampled_at < market.window_end
JOIN polymarket.backfill_artifacts snapshot_artifact
  ON snapshot_artifact.artifact_id = snapshot.artifact_id
 AND snapshot_artifact.status = 'completed'
JOIN polymarket.binance_one_second_klines kline
  ON kline.symbol = 'BTCUSDT'
 AND kline.open_timestamp >= %(batch_start)s - interval '1 second'
 AND kline.open_timestamp < %(batch_end)s
 AND kline.open_timestamp = snapshot.sampled_at - interval '1 second'
JOIN polymarket.backfill_artifacts kline_artifact
  ON kline_artifact.artifact_id = kline.artifact_id
 AND kline_artifact.status = 'completed'
ORDER BY market.window_start, snapshot.sampled_at;
