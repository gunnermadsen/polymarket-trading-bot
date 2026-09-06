# Ingester strategy and backfill contract

The ingester has exactly two runtime roles and exactly one backfill control plane:

- `ingester-master` owns the unversioned HTTP API, canonical request validation, deterministic sharding, assignment, lease recovery, retries, cancellation, and historical status.
- `ingester-worker` owns realtime collection and execution of leased backfill shards. Workers are horizontally scalable and advertise their exact strategy contract versions and immutable deployment provenance.

Both images are built from this package and run the same `ingester` binary with `INGESTER_MODE=master` or `INGESTER_MODE=worker`. A worker processes one backfill shard at a time. Scale throughput by adding workers, not by inventing provider-specific worker runtimes.

## Immutable system rules

1. `ingester.backfill_jobs` is the only job ledger and the only scheduling source. Request rows and shard rows live together. No strategy may add another queue, job table, schema, scheduler, worker executable, or Docker Compose service.
2. `ingester.backfill_job_events`, `ingester.backfill_artifacts`, and `ingester.workers` are the only supporting control-plane tables. Dataset tables remain domain-owned and record the canonical backfill job identifier as lineage.
3. New collection logic implements `RealtimeWorkerStrategy`, `BackfillWorkerStrategy`, or both. A stable strategy key identifies the data product. Contract and request-schema versions change only for a necessary, documented incompatibility.
4. Backfill request bodies are generic: strategy key, time range, strategy-owned parameters, and an optional worker or deployment selector. The master derives the idempotency hash from the canonical validated request.
5. Sharding belongs to `BackfillWorkerStrategy::plan_shards`. It must be deterministic and provider-aware. Realtime strategies are never sharded.
6. Workers never select or mutate queued work directly. They register capabilities, heartbeat, request an assignment from the master, maintain the lease, and report a verified outcome or a fixed failure classification.
7. A job is complete only after the strategy verifies persisted coverage. A lost lease cannot commit or report completion. Expired leases are recovered by the master and either retried, failed at the attempt limit, or cancelled.
8. Strategy deployment is additive: deploy a new `ingester-worker` image under a deployment identifier, wait for registration, optionally target that deployment through the API, then retire old workers after their leases drain.
9. Historical legacy tables remain read-only until every row and lineage record has been reconciled in the canonical ledger. They are not scheduling inputs and must not be deleted as part of strategy work.

## Worker allocation contract

- Realtime strategies are not sharded, and an `ingester-worker` may own at most one current realtime strategy lease.
- Every worker registers allocation contract version 1, a positive capacity-unit budget, and a realtime-slot limit of exactly one.
- Scheduling weights and isolation classes are execution metadata. They do not change dataset, request, checkpoint, or streaming contract versions.
- Realtime profile acquisition and backfill assignment lock the same `ingester.workers` row before evaluating current unexpired leases. Capacity validation and lease mutation commit in the same transaction.
- Polymarket orderbooks and Binance spot L2 are latency-critical realtime workloads and cannot share a worker with a backfill shard. Standard realtime workloads may share only when the combined allocation fits the worker budget.
- PMXT Polymarket orderbook archive work is exclusive. Heavy and exclusive backfills cannot be admitted when their allocation would exceed capacity.
- Capacity rejection leaves a backfill shard queued and does not stop, disable, or mutate a realtime strategy. Expired or released leases restore capacity automatically.
- `ingester.backfill_jobs` remains the sole job ledger, `ingester-master` remains the sole backfill scheduler/API, and `ingester-worker` remains the sole worker runtime.
- The gRPC outbound stream, provider collection paths, canonical persistence, and strategy-owned deterministic sharding are outside this allocation contract and remain unchanged.

## Canonical dataset contract

A strategy identifies how data is collected; it does not define the data model. Every strategy is bound to exactly one `DatasetKey` in `strategies/datasets.rs`. Compatible realtime and backfill strategies bind to the same dataset. A strategy must not define an alternative dataset identity because its transport, provider endpoint, or execution mode differs.

The authoritative dataset definitions live in `domain/dataset.rs`. Each definition fixes:

- the stable dataset key and contract version;
- the canonical `market_data` table;
- the observation's natural identity; and
- the complete persisted field vocabulary.

Source-specific parsers may use private wire types, but normalized records shared by strategies live in the domain contract. Persistence projections may temporarily target a legacy table while a dataset is being reconciled, but that table is not another contract and must not change the canonical model. Changing a canonical field, natural key, or meaning requires an intentional contract-version change and compatibility review.

Realtime and backfill remain separate strategy implementations. They must normalize equivalent source observations to the same dataset contract. `strategy_key` is lineage, not record identity, so records collected through different modes can converge and deduplicate on the dataset natural key.

Direct Chainlink Data Streams reports and PMData reference-price archive rows are separate products even when their timestamps and prices coincide. Direct realtime and direct archive backfill strategies bind to `ChainlinkBtcusdReferencePrices` and persist only through the shared direct repository into `market_data.chainlink_btcusd_reference_prices`. The PMData backfill binds to `PmdataChainlinkBtcusdReferencePrices` and persists only through the PMData repository into `market_data.pmdata_chainlink_btcusd_reference_prices`. Neither strategy may substitute another strategy key, write the other product's table, or issue its own insert SQL.

Historical drain tools are copy-only. They normalize source rows to the exact canonical table-shaped Parquet contract, keep provider products separate, and record complete row accounting and file hashes. They never delete, update, rename, or truncate source relations. A legacy table may be removed only by a guarded migration after the final archive watermark and manifest validation are established.

## API

All control endpoints except health and metrics require the administrative bearer token.

- `GET /strategy/all`
- `GET /strategy/{strategy_key}`
- `POST /backfills`
- `GET /backfills`
- `GET /backfills/{job_id}`
- `GET /backfills/{job_id}/events`
- `POST /backfills/{job_id}/cancel`
- `POST /backfills/{job_id}/retry`
- `GET /workers`

Internal worker endpoints are under `/internal/workers/...` and are not strategy-specific.

## Adding a strategy

Add collection logic and its tests inside this package, register it in the single registry, use existing dataset tables or a migration limited to the dataset itself, and rebuild the two ingester images. Scheduling happens only by calling `POST /backfills`. A strategy change must never add a backfill migration, queue, planner binary, image, or Compose file.
