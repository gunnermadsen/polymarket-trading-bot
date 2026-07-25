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
| `polymarket_btc_five_minute_execution_snapshots` | UTC hour | causal 250 ms executable-book snapshots for validated BTC five-minute markets |
| `chainlink_btcusd_reference_ticks` | UTC day | optional decoded, signed Chainlink BTC/USD Data Streams v3 reports |

`polymarket_btc_five_minute_orderbooks` is retained only so already-queued jobs and historical
artifact identities remain readable. The API reports `accepts_new_requests: false` for it and
rejects new raw-materialization requests.

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

Binance archives are downloaded incrementally to the persistent cache directory, limited by byte
count, hashed while streaming, fsynced, and atomically published only after their official SHA-256
checksum matches. ZIP CSV rows are decoded on one blocking thread and passed through a
single-capacity channel in batches of at most 4,000 rows. The worker does not accumulate a day of
trades in memory. Completed artifacts and BTC reference facts are immutable, and repeated writes
must match the original values.

PMXT files use the same bounded, atomic cache path but are Parquet rather than ZIP CSV. The cache is
bound to `/Volumes/docker-data/polymarket-bot/backfill-cache`, separated by worker, capped at 20 GiB
per worker, and removes partial or stale files on worker startup. A blocking streaming reader
rejects schema drift and sends only events for validated BTC five-minute condition and
outcome-token IDs through the bounded batch channel.

The compact ingester reconstructs each token book using only events whose provider receipt time is
at or before the sample. It persists one row per market every 250 ms with both outcomes: best
bid/ask and sizes, total depth, executable ask VWAP for 1, 5, and 10 shares, imbalance, source
timestamps, and explicit missing, stale, crossed-book, and insufficient-depth flags. It does not
fabricate a book. Each five-minute market therefore has exactly 1,200 rows and a full UTC day has
345,600 rows. The preceding UTC hour is read for full-book seeds. When a completed raw
materialization exists, it is reprocessed without downloading the source again; otherwise the
worker streams the PMXT files directly to compact rows and removes the hourly cache as it advances.
Raw database chunks may be pruned only after exact market coverage, cadence, causality, completed
replacement artifacts, and source-artifact lineage have been validated in a database migration.
The raw rows are disposable materialization; immutable `replaced` and `pruned` retention events,
source checksums, record counts, and compact artifact checksums remain as evidence after pruning.
The source is the
[PMXT Polymarket Orderbook Archive v2](https://archive.pmxt.dev/docs/v2-data-overview), provided by
[pmxt](https://pmxt.dev) under CC BY 4.0.

The Gamma market ingester persists `priceToBeat` as the opening boundary and `finalPrice` as the
final boundary. These exact market facts and the official CLOB outcome form the training target.
The final boundary is label-only and must never be used in features available before resolution.
Binance one-second BTCUSDT candles provide the historical intra-window path used for predictive
features; they must retain Binance provenance and must not be represented as historical Chainlink
ticks.

The optional Chainlink ingester requests reports from the authenticated sequential-report API,
HMAC-verifies transport authentication, and decodes the signed v3 report envelope. Envelope and
payload feed IDs and timestamps must agree, prices retain 18 decimal places, and
bid/benchmark/ask ordering must be valid. Set both secrets in `.env` only when this optional source
is available:

```dotenv
POLYMARKET_CHAINLINK_DATA_STREAMS_API_KEY=
POLYMARKET_CHAINLINK_DATA_STREAMS_API_SECRET=
```

The official BTC/USD feed ID, REST endpoint, PMXT endpoint, and bounded page size are non-sensitive
worker configuration in Docker Compose. If credentials are absent, workers remain available for
all other ingesters and Chainlink jobs fail permanently with a configuration error. Historical
Chainlink coverage does not gate the free-source training dataset.

Market definitions reuse the same strict Gamma identity parser as realtime execution. Official
outcomes reuse the same strict CLOB resolution parser and persistence path. Missing or ambiguous
source records are recorded as missing; the ingesters do not fabricate order books, Chainlink
ticks, outcomes, or prices.

## Readiness

The readiness endpoint reports coverage rather than claiming model quality. A market is usable
only when it has a valid five-minute identity, Gamma opening and final boundaries, an official
outcome, complete all-300-second Binance candle coverage, and exactly 1,200 compact execution
snapshots. Aggregate trades and historical Chainlink ticks are optional and do not gate readiness.
Chainlink coverage remains visible for ranges where authenticated reports are available. Quality
flags remain in the dataset so a strategy or later model can learn or abstain under poor liquidity
without treating a missing book as a valid price. Missing counts and source timestamp bounds make
incomplete ranges explicit before dataset construction or training begins. Readiness establishes
data completeness only; the pilot still has to validate label consistency and compact
reconstruction before a larger backfill is approved.

The intended pilot order is market identities, official outcomes, one-second Binance candles,
then compact PMXT execution snapshots. Binance aggregate trades and authenticated Chainlink
reports are optional for separate research and do not gate strategy readiness. No ingester is
automatically executed by deployment or migration.
