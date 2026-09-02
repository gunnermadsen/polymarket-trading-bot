WITH trades AS MATERIALIZED (
  SELECT
    trade.aggregate_trade_id,
    trade.trade_timestamp,
    greatest(trade.received_at, trade.provider_available_at) AS source_available_at,
    trade.price,
    trade.quantity,
    trade.buyer_maker,
    'market_data.binance_spot_btcusdt_aggregate_trades'::text AS source_relation,
    trade.capture_artifact_id::text AS source_artifact_id
  FROM market_data.binance_spot_btcusdt_aggregate_trades trade
  WHERE trade.source = 'binance_spot'
    AND trade.symbol = 'BTCUSDT'
    AND trade.trade_timestamp >= %(batch_start)s - interval '65 seconds'
    AND trade.trade_timestamp < %(batch_end)s
    AND greatest(trade.received_at, trade.provider_available_at) IS NOT NULL
    AND trade.trade_timestamp <= greatest(
      trade.received_at, trade.provider_available_at
    )
)
SELECT
  date_trunc('second', trade_timestamp) AS second_start,
  greatest(
    date_trunc('second', trade_timestamp) + interval '1 second',
    max(source_available_at)
  ) AS available_at,
  sum(price * quantity)::double precision AS quote_volume,
  sum(quantity)::double precision AS base_volume,
  sum(
    CASE WHEN buyer_maker
      THEN -(price * quantity)
      ELSE price * quantity
    END
  )::double precision AS signed_taker_quote_volume,
  count(*)::bigint AS trade_count,
  (sum(price * quantity) / nullif(sum(quantity), 0))::double precision AS trade_vwap,
  min(aggregate_trade_id)::bigint AS first_aggregate_trade_id,
  max(aggregate_trade_id)::bigint AS last_aggregate_trade_id,
  min(source_relation)::text AS source_relation,
  count(DISTINCT source_artifact_id)::integer AS source_artifact_count
FROM trades
GROUP BY date_trunc('second', trade_timestamp)
ORDER BY second_start;
