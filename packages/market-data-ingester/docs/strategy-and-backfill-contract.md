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
