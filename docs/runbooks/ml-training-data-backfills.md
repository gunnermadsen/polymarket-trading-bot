# ML training-data backfills

This subsystem procures historical source data for BTC five-minute model training. It does not
replay strategies, place trades, create trading processes, or replace a trading-process
configuration. A trading process remains the source of truth for paper and live strategy
execution; backfill jobs are durable operational jobs that prepare training inputs.

## Supported ingesters

| Ingester key | Range alignment | Durable output |
|---|---:|---|
| `btc_five_minute_markets` | five minutes | validated market identities plus immutable Gamma artifacts and reference facts |
| `btc_five_minute_resolutions` | five minutes | official CLOB outcomes on the existing market identities |
| `binance_btcusdt_agg_trades` | UTC day | checksummed BTCUSDT aggregate trades |
| `binance_btcusdt_one_second_klines` | UTC day | checksummed BTCUSDT one-second candles |
| `polymarket_btc_five_minute_orderbooks` | UTC hour | PMXT v2 CLOB events filtered to validated BTC five-minute condition and token IDs |
| `chainlink_btcusd_reference_ticks` | UTC day | decoded, signed Chainlink BTC/USD Data Streams v3 reports |

Requests use half-open ranges: `range_start` is included and `range_end` is excluded. Version 1
accepts an empty `parameters` object. An idempotency key is required and remains reserved after a
job completes, so retrying the same request returns the same job instead of duplicating work.
Ranges must end at a completed historical interval; future five-minute windows, incomplete UTC
archive days, and PMXT hours within its ten-minute publication allowance are rejected at enqueue
time. PMXT v2 requests earlier than its `2026-04-13T19:00:00Z` coverage boundary fail explicitly.

## API

All routes require the normal admin bearer token.

- `GET /admin/backfill/ingesters`
- `POST /admin/backfill/jobs`
- `GET /admin/backfill/jobs?limit=50`
- `GET /admin/backfill/jobs/{job_id}`
- `GET /admin/backfill/jobs/{job_id}/events?limit=100`
- `POST /admin/backfill/jobs/{job_id}/cancel`
- `GET /admin/backfill/readiness/btc-five-minute-training?range_start=...&range_end=...`

An enqueue body has this shape:

```json
{
  "ingester": "binance_btcusdt_one_second_klines",
  "request_version": 1,
  "range_start": "2026-07-01T00:00:00Z",
  "range_end": "2026-07-02T00:00:00Z",
  "parameters": {},
  "idempotency_key": "btc-1s-2026-07-01-v1"
}
```

Successful enqueue returns HTTP 200 with `job_id`, `ingester`, `status`, and `requested_at`.
Enqueueing is separate from execution: the Docker worker claims queued jobs from Postgres.

## Execution and integrity

`polymarket-backfill-worker` is a separate Docker Compose service built from the same Rust image
as `polymarket-bot`. It runs one job, one archive parser, and one database writer at a time. Its
Postgres lease carries a random fencing token; progress, artifact transitions, batch writes, and
terminal job transitions reject a stale worker.

Binance archives are downloaded incrementally to the persistent cache volume, limited by byte
count, hashed while streaming, fsynced, and atomically published only after their official SHA-256
checksum matches. ZIP CSV rows are decoded on one blocking thread and passed through a
single-capacity channel in batches of at most 4,000 rows. The worker does not accumulate a day of
trades in memory. Completed artifacts and BTC reference facts are immutable, and repeated writes
must match the original values.

PMXT files use the same bounded, atomic cache path but are Parquet rather than ZIP CSV. A blocking
streaming reader rejects schema drift, retains the global source row number, and sends only events
for validated BTC five-minute condition and outcome-token IDs through the bounded batch channel.
The database stores native events (`book`, `price_change`, `last_trade_price`, and
`tick_size_change`) without sampling. PMXT must run after the market-identity ingester; an empty
identity scope fails instead of creating an incomplete completed artifact. Include the UTC hour
before a study range when initial full-book seed coverage is needed. The source is the
[PMXT Polymarket Orderbook Archive v2](https://archive.pmxt.dev/docs/v2-data-overview), provided by
[pmxt](https://pmxt.dev) under CC BY 4.0.

Chainlink reports are requested from the authenticated sequential-report API, HMAC-verified at
transport authentication, and decoded from their signed v3 report envelope. Envelope and payload
feed IDs and timestamps must agree, prices retain 18 decimal places, and bid/benchmark/ask ordering
must be valid. Set both secrets in `.env`:

```dotenv
POLYMARKET_CHAINLINK_DATA_STREAMS_API_KEY=
POLYMARKET_CHAINLINK_DATA_STREAMS_API_SECRET=
```

The official BTC/USD feed ID, REST endpoint, PMXT endpoint, and bounded page size are non-sensitive
worker configuration in Docker Compose. If credentials are absent, workers remain available for
all other ingesters and Chainlink jobs fail permanently with a configuration error.

Market definitions reuse the same strict Gamma identity parser as realtime execution. Official
outcomes reuse the same strict CLOB resolution parser and persistence path. Missing or ambiguous
source records are recorded as missing; the ingesters do not fabricate order books, Chainlink
ticks, outcomes, or prices.

## Readiness

The readiness endpoint reports coverage rather than claiming model quality. A market is usable
only when it has a valid five-minute identity, an opening boundary, an official outcome, completed
all-300-second Binance candle coverage, Chainlink reports at both window boundaries, full-book
seeds for both outcome tokens, and orderbook events during the market window. Aggregate trades and
final-price coverage are reported separately but are not required for the current strategy
hypothesis. Missing counts and source timestamp bounds make incomplete ranges explicit before
dataset construction or training begins. Readiness establishes data completeness only; the pilot
still has to validate Chainlink overlap against the realtime RTDS feed and orderbook reconstruction
against live checkpoints before a larger backfill is approved.

The intended pilot order is market identities, official outcomes, one-second Binance candles,
Chainlink reports, then PMXT orderbooks with the preceding seed hour. No ingester is automatically
executed by deployment or migration.
