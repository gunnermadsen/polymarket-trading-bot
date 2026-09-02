SELECT
  date_trunc('second', trade.trade_timestamp) AS second_start,
  sum(trade.price * trade.quantity)::double precision AS quote_volume,
  sum(trade.quantity)::double precision AS base_volume,
  sum(
    CASE WHEN trade.buyer_maker
      THEN -(trade.price * trade.quantity)
      ELSE trade.price * trade.quantity
    END
  )::double precision AS signed_taker_quote_volume,
  count(*)::bigint AS trade_count,
  (
    sum(trade.price * trade.quantity) / nullif(sum(trade.quantity), 0)
  )::double precision AS trade_vwap
FROM market_data.binance_spot_btcusdt_aggregate_trades trade
JOIN ingester.capture_artifacts artifact
  ON artifact.strategy_key = trade.strategy_key
 AND artifact.artifact_id = trade.capture_artifact_id
 AND artifact.status = 'completed'
WHERE trade.source = 'binance_spot'
  AND trade.symbol = 'BTCUSDT'
  AND trade.trade_timestamp >= %(batch_start)s
  AND trade.trade_timestamp < %(batch_end)s
GROUP BY date_trunc('second', trade.trade_timestamp)
ORDER BY second_start;
