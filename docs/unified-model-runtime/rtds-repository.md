# Shared RTDS repository contract

The BTC shared market-data runtime owns one `RtdsRepository` for the product `polymarket_rtds_chainlink_reference_price`. All UMR and existing model consumers read that repository. No consumer seeds a private RTDS history, builds its own RTDS candles, or queries the database during inference.

## Organization and ownership

| Location | Responsibility |
| --- | --- |
| `btc/rtds_repository/mod.rs` | Private bounded point storage, deduplication, causal point reads and candle/readiness interface |
| `btc/rtds_repository/candles.rs` | The existing RTDS-to-OHLC calculation, extracted without changing arithmetic or timestamps |
| `btc/rtds_repository/tests.rs` | Frozen candle outputs, causal gaps, duplicate hydration, bounded retention and snapshot isolation |
| `btc/runtime.rs` | One startup seed and the existing retry/shutdown lifecycle |
| `btc/repository.rs` | Existing bounded database history query; durable persistence remains unchanged |
| `btc/directional_external_runtime.rs` | Shared repository handle and existing source-status compatibility projection |

The repository is held by `DirectionalExternalState` inside the existing shared `RealtimeState`. Observation snapshots share its immutable `Arc`; readers have no public mutation API. A writer uses copy-on-write only when an observation still holds an older snapshot. These transient snapshots are views of one repository lineage, not independently fed caches.

UMR adapters access `FeatureContext.state.directional_external.rtds()`. Existing model feature preparation uses the same accessor. A new compatible model needs only its existing source declaration and read-only adapter binding, never a seeding job or another RTDS connection.

The existing `reference_prices` latest-value map remains a compatibility projection for public readiness and execution interfaces. It contains one current tick per reference source, not another RTDS history or candle builder. Existing durable opening-reference and audit records retain their accounting/provenance roles; this cutover does not create a second database source or redirect historical audit queries.

## Startup, ingestion and recovery

1. Every shared BTC runtime starts by restoring RTDS history, regardless of which model starts first. The existing database query reads a bounded 62-minute range, capped at 5,000 rows, from `polymarket.reference_price_ticks` for `rtds_chainlink`, `BTCUSD`, and valid integrity. No new query, table, migration or worker is introduced.
2. Hydration feeds the repository's same writer entry used by the existing gRPC RTDS ingestion path. Source and original receipt timestamps are retained. Hydration does not publish old ticks into the current-price projection.
3. The existing retry path handles incomplete or unavailable history with backoff capped at 300 seconds. After 61 complete minutes it waits for shutdown. Hydration queries occur outside the state lock; merging is bounded and cannot overwrite a newer live point with older history.
4. One RTDS selector is retained in the existing shared gRPC subscription even if no model currently requires it. This default selector is optional and does not gate unrelated models. An explicitly configured consumer selector takes precedence unchanged. No second feed connection or per-model subscription is added.
5. Later consumers receive the existing repository. The former RTDS dynamic-source hydration flag/task is removed; there is only startup ownership and its retry task. Runtime shutdown owns the task using the existing lifecycle.

Missing history or transport failures do not disable durable process intent. Only actions requiring unavailable evidence remain ineligible under their existing checks. Existing model, feed, execution, capital and accounting controls are preserved.

## Data and read semantics

- Product: RTDS Chainlink BTC/USD midpoint observations, not canonical Chainlink OHLC, direct signed RefPrice, or TWAP.
- Storage remains capped at 4,096 ascending source-timestamp points, preserving the prior retention behavior. This bounds memory; it does not promise arbitrary historical queries or guarantee 61 minutes if source coverage is insufficient.
- Identity and duplicates preserve existing behavior: first accepted point per source timestamp wins. Late historical inserts are sorted; capacity pruning removes the oldest source timestamps. Invalid nonpositive prices are rejected.
- `points_as_of(at)` returns source-ordered immutable points available by `at`, with no future source timestamps. Consumers must declare any additional freshness requirement.
- `closed_candles(at)` returns exactly 61 contiguous closed one-minute candles ending at the minute boundary at or before `at`. Only observations originally available by `at` participate. OHLC arithmetic, first/last source-time selection and maximum constituent availability match the previous implementation.
- Gaps produce the existing `ExternalFeatureUnavailable` diagnostic, including the `chainlink_candles` source key. No forward filling, synthetic candles, relaxed timing or alternate product substitution is allowed.
- `complete_minutes(at)` preserves the existing metric definition: distinct represented closed minutes within the 61-minute window. It does not assert tick-by-tick completeness within a minute. The model independently requests its causal candle window.

## Instrumentation compatibility

`polymarket_btc_rtds_chainlink_candle_complete_minutes` and `polymarket_btc_rtds_chainlink_candle_window_ready` retain their names, units and meanings and now read the repository. Existing hydration diagnostics, source status, process-scoped feature errors, inference counters, and UMR dashboard/alerts remain intact. No dashboard SQL, datasource, monitoring service or new metric family is required.

## Cutover verification

Verify known OHLC/availability outputs, gaps and future-data rejection, overlapping seed/live updates, bounded out-of-order retention, shared snapshot identity, source-selector uniqueness, and unchanged readiness metrics. Run existing model reference vectors and process lifecycle tests. After deploying the committed feature image, check automatic ten-process recovery, model/config identity preservation, 61-minute history when supported by persisted data, and successful inference for RTDS-dependent models. Retain exact image provenance and the preceding immutable image for rollback; do not merge or promote implicitly.
