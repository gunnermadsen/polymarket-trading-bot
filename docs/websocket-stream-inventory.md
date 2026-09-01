# Runtime websocket stream inventory

Snapshot taken 2026-08-31 from the local Docker Compose runtime, the
`ingester.profiles` control plane, recent container logs, and the websocket
client implementations in this repository.

## Summary

- Two running microservices own public market-data websocket clients:
  `polymarket-bot` and `market-data-ingester`.
- They currently account for 10 configured/running public websocket clients
  across 6 logical feeds.
- Four logical feeds are acquired independently by both microservices:
  Polymarket CLOB orderbooks, Binance aggregate trades, Binance spot L2, and
  Polymarket RTDS Chainlink BTC/USD.
- The Polymarket public CLOB market endpoint has three independent connections:
  the bot orderbook client, the ingester orderbook client, and the ingester
  resolution-event client.
- No enabled `realtime_live` trading process existed at the snapshot time, so
  the configured authenticated Polymarket user websocket was not operating.
- Chainlink one-minute candles, Chainlink RefPrice, Polygon Chainlink oracle,
  and Binance open interest are polling flows, not websocket clients. Their
  cross-service duplication is listed separately.

## Operating websocket clients

| Provider / logical feed | Endpoint | Microservice | Runtime owner | Snapshot state |
|---|---|---|---|---|
| Polymarket CLOB BTC 5m orderbooks | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | `polymarket-bot` | BTC realtime runtime | Operating; connected and producing orderbook checkpoints, with reconnect recovery |
| Binance BTCUSDT aggregate trades | `wss://stream.binance.com/ws/btcusdt@aggTrade` | `polymarket-bot` | BTC reference-price runtime | Operating but reconnecting during snapshot network failures |
| Binance BTCUSDT spot L2 diff depth | `wss://stream.binance.com/ws/btcusdt@depth@100ms` | `polymarket-bot` | BTC spot-L2 runtime | Enabled and operating, but reconnecting during snapshot network failures |
| Polymarket RTDS Chainlink BTC/USD | `wss://ws-live-data.polymarket.com` | `polymarket-bot` | BTC RTDS reference runtime | Operating; supplies RTDS Chainlink price/candle state |
| Binance BTCUSDT aggregate trades | `wss://stream.binance.com:9443/ws/btcusdt@aggTrade` | `market-data-ingester` | `binance_spot_btcusdt_aggregate_trades` | Desired running; degraded during snapshot, with recent source events |
| Binance BTCUSDT spot L2 diff depth | `wss://stream.binance.com:9443/ws/btcusdt@depth@100ms` | `market-data-ingester` | `binance_spot_btcusdt_l2_snapshots` | Running and healthy; recent synchronized book |
| Binance BTCUSDT one-second klines | `wss://stream.binance.com:9443/ws/btcusdt@kline_1s` | `market-data-ingester` | `binance_spot_btcusdt_one_second_ohlcv` | Running and healthy; recent source events |
| Polymarket CLOB BTC 5m orderbooks | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | `market-data-ingester` | `polymarket_btc_five_minute_orderbooks` | Desired running; degraded during snapshot, with recent source events |
| Polymarket CLOB BTC 5m resolution events | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | `market-data-ingester` | `polymarket_btc_five_minute_resolutions` | Desired running; degraded and reconnecting after pong timeouts |
| Polymarket RTDS Chainlink BTC/USD TWAP | `wss://ws-live-data.polymarket.com` | `market-data-ingester` | `polymarket_chainlink_btcusd_twap` | Running and healthy; recent source events |

`stream.binance.com` ports 443 and 9443 are different transport addresses for
the same Binance logical streams. They do not make the acquisitions distinct.

## Duplicate websocket acquisition

| Duplicate logical feed | Independent clients | Duplication finding |
|---|---:|---|
| Polymarket CLOB BTC 5m orderbooks | 2 | Exact logical duplicate: both services independently maintain the same market books |
| Binance BTCUSDT aggregate trades | 2 | Exact logical duplicate: both services consume `btcusdt@aggTrade` |
| Binance BTCUSDT spot L2 | 2 | Exact logical duplicate: both services consume `btcusdt@depth@100ms` and reconstruct books independently |
| Polymarket RTDS Chainlink BTC/USD | 2 | Provider/underlying-feed duplicate: the bot consumes RTDS price state while the ingester captures Chainlink TWAP from the same RTDS connection endpoint |
| Polymarket public CLOB market endpoint | 3 | Endpoint duplicate: two orderbook connections plus one resolution-event connection; the resolution connection has a distinct purpose but still creates a separate upstream session |

The strongest consolidation candidates are the three exact logical duplicates:
CLOB orderbooks, Binance aggregate trades, and Binance spot L2. The RTDS clients
need a topic/subscription comparison before treating them as payload-identical.
The resolution-event connection should not be described as duplicate orderbook
processing merely because it shares the CLOB endpoint.

## Related duplicated polling flows (not websockets)

| Data family | `polymarket-bot` | `market-data-ingester` | Finding |
|---|---|---|---|
| Chainlink RefPrice | Chainlink Data Streams REST poller | `chainlink_btcusd_reference_price` REST strategy | Duplicate independent HTTP acquisition; both were failing/retrying near the snapshot |
| Polygon Chainlink BTC/USD oracle | Polygon JSON-RPC poller | `polygon_chainlink_btcusd_oracle` JSON-RPC strategy | Duplicate independent RPC acquisition |
| Binance futures open interest | Binance Futures REST poller | `binance_futures_btcusdt_open_interest` REST strategy | Duplicate independent HTTP acquisition |
| Chainlink candle history | Bot derives candle windows from RTDS Chainlink ticks | `chainlink_btcusd_one_minute_ohlc` uses Chainlink candlestick HTTP data | Overlapping candle data, but different acquisition mechanisms and potentially different canonical payloads |

## Configured but not operating

The bot is configured for the authenticated endpoint
`wss://ws-subscriptions-clob.polymarket.com/ws/user`. That client is activated
for live execution/account events. The runtime database contained zero enabled
`realtime_live` processes at the snapshot time, so it is excluded from the 10
operating public clients above.

## Active model dependencies

At the snapshot time, five enabled `realtime_paper` processes were running.
All five use a `btc_directional_model` decision strategy:

| Active model | Binance aggregate-trade dependency | Binance spot-L2 dependency |
|---|---|---|
| `btc-5m-payoff-aware-q5-paper-20260820` | Yes, through the bot's aggregate-trade-built one-second BTC candle window | No |
| `btc-5m-chainlink-full-combined-paper-20260820` | Yes, through the same one-second candle window | No |
| `btc-5m-chainlink-stratified-payoff-paper-20260820` | Yes, through the same one-second candle window | No |
| `btc-5m-chainlink-regime-calibrated-paper-20260820` | Yes, through the same one-second candle window | No |
| `btc-5m-specialist-distilled-fair-value-paper-20260823-v1` | Yes, through the same one-second candle window | No |

The active model schemas contain BTC return, momentum, path, basis, and, for
some models, trade-count and quote-volume inputs derived from the completed
one-second Binance candle window. The bot currently constructs that window
from its direct `btcusdt@aggTrade` websocket with bounded sequence-gap recovery.
Therefore, the direct bot aggregate-trade socket can be removed only after an
equivalent ordered, gap-aware stream or canonical one-second candle stream is
delivered from the ingester.

The bot's Binance spot-L2 runtime produces a separate compact L2 feature
history used by the asymmetric-value model path. None of the five enabled
processes selects that model path or has L2 features in its frozen model input
schema. The L2 socket is therefore not required by current active-model
inference. It can still affect readiness/metrics if left globally enabled, so
removal should also remove that unused runtime dependency rather than merely
disconnecting its socket.

## Expected latency from ingester-to-bot streaming

Centralizing acquisition in `market-data-ingester` and forwarding normalized
events over a long-lived streaming gRPC connection should add sub-millisecond
to low-single-digit-millisecond latency on the same Docker host when the path
is implemented without a database round trip or batching. A reasonable design
target—not a measured result for this repository—is:

| Component | Design target |
|---|---:|
| Protobuf encode, enqueue, decode, and state update | about 0.1-0.8 ms p50 |
| Same-host Docker networking and gRPC/HTTP2 scheduling | about 0.1-0.7 ms p50 |
| End-to-end ingester receipt to bot application | under 1 ms p50 and under 5 ms p99 |

That overhead is small relative to the active strategies' 1-second or 5-second
decision cadence and their 1,000-5,000 ms directional-feature age limits. It
is also small relative to the configured 150 ms paper arrival-latency model.
It is not automatically harmless for orderbook execution: queueing, batching,
flow-control stalls, or a persistence-first design can turn sub-millisecond
transport into tens or hundreds of milliseconds.

The low-latency form of this design is a live fan-out path:

1. Timestamp the provider frame immediately when the ingester receives it.
2. Validate sequence and reconstruct canonical state in one owner.
3. Publish normalized events or snapshots directly to the bot over a bounded
   streaming gRPC channel.
4. Persist asynchronously from the same canonical event, without placing the
   database between ingestion and trading.
5. Carry source timestamp, ingester receipt timestamp, sequence/update IDs,
   connection epoch, integrity state, and publish timestamp so the bot can
   retain its existing freshness and fail-closed checks.

The principal tradeoff is availability rather than latency. Centralization
removes duplicate upstream work but makes the ingester and its fan-out channel
a shared dependency. The bot must automatically reconnect, receive a bounded
snapshot plus sequence cursor, reject gaps/stale state locally, and resume
eligibility when healthy evidence returns. A slow consumer must not back up
provider ingestion or other consumers.

## Scope and interpretation

“Operating” means a runtime owner is configured/desired to run and is either
receiving source events or actively reconnecting under its normal recovery
loop. It does not mean every socket was established at the exact inspection
instant: several providers experienced DNS, TLS, timeout, or pong failures
during the snapshot. Backfill workers, Grafana, Prometheus, Loki, Alloy,
PgBouncer, and the database containers showed no market-data websocket role in
this inventory.
