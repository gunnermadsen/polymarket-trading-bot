# Binance BTCUSDT L2 ingestion

This ingester materializes causal one-second features for Binance USDⓈ-M `BTCUSDT`
from CryptoHFTData hourly order-book objects. Its historical contract is the half-open
range `[2026-04-14T00:00:00Z, 2026-08-02T00:00:00Z)`.

## Storage contract

`POLYMARKET_BINANCE_L2_ARCHIVE_ROOT` must name an existing, separately mounted SSD
directory. The worker refuses to create this root so an unmounted volume cannot redirect
the archive onto the system disk. Preflight requires at least 25% and 64 GiB free, performs
a synchronous 16 MiB write probe, and requires at least 20 MiB/s. It also requires the file
`.cryptohft-l2-storage` in the mounted root with this exact content, including the trailing
newline:

```text
cryptohft-btcusdt-l2-archive-v1
```

With `POLYMARKET_BINANCE_L2_ARCHIVE_ROOT` set to the mounted SSD directory, provision it
once with:

```bash
printf 'cryptohft-btcusdt-l2-archive-v1\n' > "${POLYMARKET_BINANCE_L2_ARCHIVE_ROOT}/.cryptohft-l2-storage"
chmod 0444 "${POLYMARKET_BINANCE_L2_ARCHIVE_ROOT}/.cryptohft-l2-storage"
```

The sentinel must be created on the mounted SSD itself. If the volume is later unmounted,
the sentinel disappears and the worker refuses to write into a same-named host directory.
The worker deliberately never creates the sentinel itself.

Original objects are retained as:

```text
binance_futures/YYYY-MM-DD/HH/BTCUSDT_orderbook.parquet.zst
```

Each object has a read-only adjacent JSON manifest containing its locally computed
SHA-256, byte count, decoded row count, logical snapshot/update counts, and provider-receipt
coverage. CryptoHFTData does not publish source checksums. Downloads use unique partial
files and no-clobber atomic finalization. Every Parquet row and page is decoded and checked
against the requested provider-receipt hour before finalization. Invalid cached objects are
moved to recoverable quarantine names before a fresh acquisition. Decoded Parquet is written
only under the worker's disposable backfill cache and removed after parsing; stale crash
files are scavenged after six hours.

Use the opt-in Compose overlay so ordinary services do not require the SSD mount:

```bash
POLYMARKET_BINANCE_L2_ARCHIVE_ROOT=/absolute/ssd/path \
  docker compose -f docker-compose.yml -f docker-compose.binance-l2.yml config
```

## Causal replay

Rows belonging to one exchange message are grouped across Parquet row-group boundaries.
A full snapshot is the only valid bootstrap. Updates use absolute quantities, zero removes
a level, and each applied futures update must satisfy Binance `U/u/pu` continuity. A
sequence discontinuity invalidates and clears the book until another full snapshot.
Each daily shard walks backward through contiguous context no earlier than the audited
`2026-04-13T23:00:00Z` archive until it finds a structurally validated 1,000-level snapshot.
This gives every independent shard a deterministic fallback to the known bootstrap instead
of assuming a maximum snapshot age or that the immediately preceding hour contains one.
The fallback is pinned to
`binance_futures/2026-04-13/23/BTCUSDT_orderbook.parquet.zst`, locally verified as
SHA-256 `9e5557a44fe0c414feb353ad64f22ea232bd5e895f01f32f17fb1bd8d36e5174`.
That object contains the audited snapshot at event time `2026-04-13T23:05:19.649Z`
(`1776121519649` milliseconds) with last update ID `10318083192958`. The worker rejects
the anchor if its compressed checksum or that structurally valid snapshot identity changes.

`available_at` is the later of exchange event time and CryptoHFT receive time plus the
immutable 100 ms availability offset. A row is emitted only for the
latest valid, non-crossed state in its availability second, with no more than one second
of staleness and with exact continuous history for all 1/5/15/30/60-second changes.
Unavailable seconds are absent from the feature table and counted in artifact metadata.

The compact feature formulas are:

- midpoint: `(best_bid + best_ask) / 2`
- microprice: `(best_ask * best_bid_qty + best_bid * best_ask_qty) /
  (best_bid_qty + best_ask_qty)`
- spread: `(best_ask - best_bid) / midpoint * 10,000` basis points
- depth: summed quantity at the top 5, 10, and 20 levels
- imbalance: `(bid_depth - ask_depth) / (bid_depth + ask_depth)`
- concentration: top-5 depth divided by top-20 depth
- slope: top-20 price distance in basis points divided by top-20 depth
- replenishment/churn: positive/negative changed quote notional during the second;
  snapshots are excluded

Only rows with `quality_status = 'qualified'` and feature schema
`binance-btcusdt-l2-one-second-features-v1` are stored. Quality and missingness are not
model inputs.

## Sharding and source gate

`binance-l2-backfill-plan` uses one idempotent job per UTC day. On its first invocation it
enqueues only the first day as a representative source audit. It refuses to enqueue the
remaining 109 days unless that completed artifact has a snapshot bootstrap, zero sequence
gaps, a complete 24-object target manifest, no invalid book events, at least 86,000 qualified
seconds, and at most 400 unavailable seconds.

The April 14 boundary audit passed: the April 13 23:00 archive contains a valid 1,000-level
snapshot, its first subsequent update spans the snapshot sequence, and continuity is exact
across the 00:00 and 01:00 hourly boundaries. The representative job still replays and
validates all 24 April 14 hours before the other 109 shards can be queued. If a terminal
representative job must be retried, increment
`POLYMARKET_BINANCE_L2_RETRY_GENERATION`; the generation is part of the job idempotency key.

All workers reserve anonymous CryptoHFT request slots through a single PostgreSQL budget
row. This keeps requests at least 1.1 seconds apart across processes while download, decode,
and bounded database staging remain independently backpressured.
