WITH eligible_markets AS MATERIALIZED (
  SELECT market_id, window_start, window_end
  FROM polymarket.btc_interval_markets
  WHERE window_start >= %(batch_start)s
    AND window_start < %(batch_end)s
    AND validation_status = 'valid'
    AND official_outcome IN ('up', 'down')
),
prices AS MATERIALIZED (
  SELECT kline.open_timestamp, kline.close_price::double precision AS price
  FROM polymarket.binance_one_second_klines kline
  JOIN polymarket.backfill_artifacts artifact
    ON artifact.artifact_id = kline.artifact_id
   AND artifact.status = 'completed'
  WHERE kline.symbol = 'BTCUSDT'
    AND kline.open_timestamp >= %(batch_start)s - interval '60 seconds'
    AND kline.open_timestamp < %(batch_end)s + interval '5 minutes'
    AND kline.close_timestamp < kline.open_timestamp + interval '1 second'
)
SELECT
  market.market_id,
  avg(price.price) FILTER (
    WHERE price.open_timestamp >= market.window_start - interval '60 seconds'
      AND price.open_timestamp < market.window_start
  ) AS binance_twap_open_price,
  count(*) FILTER (
    WHERE price.open_timestamp >= market.window_start - interval '60 seconds'
      AND price.open_timestamp < market.window_start
  )::integer AS binance_twap_open_rows,
  avg(price.price) FILTER (
    WHERE price.open_timestamp >= market.window_end - interval '60 seconds'
      AND price.open_timestamp < market.window_end
  ) AS binance_twap_close_price,
  count(*) FILTER (
    WHERE price.open_timestamp >= market.window_end - interval '60 seconds'
      AND price.open_timestamp < market.window_end
  )::integer AS binance_twap_close_rows
FROM eligible_markets market
JOIN prices price
  ON price.open_timestamp >= market.window_start - interval '60 seconds'
 AND price.open_timestamp < market.window_end
GROUP BY market.market_id
ORDER BY market.market_id;
