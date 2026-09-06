# Worker allocation acceptance

Acceptance requires the exact committed ingester image and applied allocation migration.

1. Register at least as many healthy workers as desired realtime profiles.
2. Verify every current realtime lease has a distinct `lease_owner` and no owner has more than one lease.
3. Verify every active worker reports `allocated_units <= capacity_units` and `realtime_leases <= 1`.
4. Confirm Polymarket orderbooks and Binance spot L2 have zero colocated backfill leases.
5. Schedule a bounded backfill over the latest detected missing-data interval through `POST /backfills`.
6. Verify the shard remains queued on incompatible or saturated workers and is leased by the first compatible worker with sufficient capacity.
7. Verify the assigned worker's allocation rises by the shard weight, never exceeds capacity, and returns after completion.
8. Verify the parent request reaches `completed` only after persisted coverage verification.
9. Restart one worker and verify its lease expires or drains, another eligible worker acquires at most one realtime profile, and healthy source/persistence timestamps resume automatically.
10. Confirm the worker saturation, capacity exceeded, and distribution-deficit alerts are normal after stabilization; intentionally violating admission in repository tests must be rejected.
11. Confirm existing gRPC subscribers continue receiving unchanged contract-version 1 events throughout placement changes.
12. Monitor worker allocation, orderbook source freshness, persistence freshness, websocket reconnects, critical logs, container restarts, and resource use after deployment.
