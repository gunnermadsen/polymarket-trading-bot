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

Requests use half-open ranges: `range_start` is included and `range_end` is excluded. Version 1
accepts an empty `parameters` object. An idempotency key is required and remains reserved after a
job completes, so retrying the same request returns the same job instead of duplicating work.
Ranges must end at a completed historical interval; future five-minute windows and incomplete UTC
archive days are rejected at enqueue time.

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

Market definitions reuse the same strict Gamma identity parser as realtime execution. Official
outcomes reuse the same strict CLOB resolution parser and persistence path. Missing or ambiguous
source records are recorded as missing; the ingesters do not fabricate order books, Chainlink
ticks, outcomes, or prices.

## Readiness

The readiness endpoint reports coverage rather than claiming model quality. A market is usable
only when it has a valid five-minute identity, an opening boundary, an official outcome, completed
checksummed Binance aggregate-trade coverage, and all 300 one-second candles from a completed
artifact. Final-price coverage is reported separately. Missing counts and source timestamp bounds
make incomplete ranges explicit before dataset construction or training begins.
