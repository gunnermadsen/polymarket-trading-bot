# Binance spot BTCUSDT L2 ingestion

This ingester materializes causal one-second features for Binance spot `BTCUSDT` from
CryptoHFTData hourly order-book objects. Its historical contract is the half-open range
`[2026-04-14T00:00:00Z, 2026-08-02T00:00:00Z)`, matching the existing futures L2 backfill
so coverage can be compared over the same 9,504,000 UTC seconds.

The spot asset is independent from the USDⓈ-M futures asset. Spot rows, artifacts, jobs,
feature schema, and materialization contract never replace or publish into futures storage.

## Storage and identity

`POLYMARKET_BINANCE_SPOT_L2_ARCHIVE_ROOT` must name the separately mounted SSD directory
inside each backfill worker. The existing CryptoHFT storage preflight applies: the mount
sentinel must be present, at least 25% and 64 GiB must remain free, and the synchronous
write probe must reach at least 20 MiB/s. The worker refuses to create the archive root or
sentinel.

Original spot objects and adjacent immutable manifests are retained under:

```text
binance_spot/YYYY-MM-DD/HH/BTCUSDT_orderbook.parquet.zst
```

The database identities are:

```text
ingester:                 binance_spot_btcusdt_l2_one_second_features
feature schema:           binance-spot-btcusdt-l2-one-second-features-v1
materialization contract: cryptohft-binance-spot-btcusdt-l2-features-v1
feature table:            polymarket.binance_spot_btcusdt_l2_one_second_features
```

## Causal spot replay

One full-depth spot snapshot is the only valid bootstrap. After bootstrap, update ranges
use Binance spot `U/u` continuity:

```text
u <= current update ID          stale update; ignore
U > current update ID + 1       sequence gap; invalidate and clear the book
U <= current update ID + 1 <= u continuous update; apply absolute quantities
```

Spot does not use the USDⓈ-M futures `pu` field. A sequence gap remains unavailable until a
later valid full snapshot; the ingester does not forward-fill or synthesize state. Crossed,
stale, or discontinuous state is never published.

`available_at` is the later of exchange event time and CryptoHFT receive time plus the
immutable 100 ms availability offset. Only the latest valid state causally available in a
UTC second is eligible. The one-second row uses the same top-5/10/20 feature formulas and
1/5/15/30/60-second exact-history requirements as the futures asset, with the spot-specific
schema identifier above.

## Representative gate and daily plan

`binance-spot-l2-backfill-plan` creates one idempotent job for each of the 110 UTC days. Its
first invocation enqueues only April 14. The remaining 109 shards remain gated until the
representative artifact proves:

- a valid snapshot bootstrap and spot `U/u` replay;
- 24 verified target archives with local SHA-256 manifests;
- no invalid or crossed book event was published;
- at least one qualified second; and
- `qualified_seconds + unavailable_seconds = 86,400`.

After the representative job completes, run the same planner inside one of the already
recreated workers to prepare the entire idempotent daily inventory. This uses the exact worker
image and environment currently serving the backfill:

```bash
docker exec polymarket-backfill-worker \
  /usr/local/bin/binance-spot-l2-backfill-plan
```

If a terminal representative job must be retried, increment
`POLYMARKET_BINANCE_SPOT_L2_RETRY_GENERATION`. The retry generation is part of the spot-only
idempotency key.

Apply the spot migration only through `db-migrate`. Build and recreate only the six
`polymarket-backfill-worker` services; do not rebuild or restart `polymarket-bot`.

## Completion and comparison

The backfill is complete only when all 110 spot artifacts are completed, all 2,640 target
objects have verified manifests, the active retry generation has no queued/running/failed
jobs, staging is empty, and published rows equal the sum of artifact `qualified_seconds`.
Older terminal retry generations remain immutable lineage and are reported separately.

Coverage comparison is performed one UTC day at a time. It reports spot qualified,
futures qualified, both, spot-only, futures-only, and neither, plus contiguous-gap
distributions. Spot-only seconds are complementary training coverage and never repair or
relabel missing futures state.
