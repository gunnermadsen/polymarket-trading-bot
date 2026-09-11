# PostgreSQL connection pooling

The Docker Compose environments route application connections through PgBouncer on port
`6432`. Database migrations remain connected directly to TimescaleDB on port `5432` so
migration sessions retain normal PostgreSQL semantics.

PgBouncer exposes bounded aliases for each service identity while retaining the legacy
`polymarket` route for rollback:

- `polymarket_trading` for `capitonic_trading`
- `polymarket_ingester_master_tx` for `capitonic_ingester_master`
- `polymarket_ingester_worker_tx` for `capitonic_ingester_worker`
- `polymarket_observe_tx` for the read-only `capitonic_grafana`

The aliases isolate connection budgets; they do not substitute PostgreSQL users. PgBouncer
authenticates each dedicated login and PostgreSQL enforces that role's grants. Trading remains
session-pooled. Ingester master, dynamically scaled workers, and Grafana use transaction-pooled
routes. Their prior session-pooled aliases remain available during rollout and rollback:

- `polymarket_ingester_master`
- `polymarket_ingester_worker`
- `polymarket_observe`

Compose defaults to the transaction aliases. Set `INGESTER_MASTER_POSTGRES_ALIAS`,
`INGESTER_WORKER_POSTGRES_ALIAS`, or `GRAFANA_POSTGRES_ALIAS` to the corresponding session alias
to roll back one workload without changing credentials or database grants.

## Why trading session affinity is required

Two BTC repository queries call SQLx `.persistent(false)`:

- `load_directional_external_chainlink_mid_history`
- `load_checkpoint_as_of`

The latter deliberately avoids PostgreSQL's cached generic plan because that plan can expand
the Timescale hypertable across compressed and uncompressed chunks before runtime exclusion.
Keeping a custom plan lets the timestamp predicates prune chunks before relation locks are
taken.

In SQLx 0.8, `.persistent(false)` uses the unnamed PostgreSQL statement slot. SQLx sends
`Parse + Sync`, waits for PostgreSQL, and then sends `Bind + Execute`. A transaction pooler may
release the backend at the first synchronization point and assign a different backend for the
bind. The second backend has no matching unnamed statement and returns:

```text
unnamed prepared statement does not exist
```

Session pooling preserves the physical backend for the client connection and therefore
preserves both the custom-plan optimization and SQLx's protocol assumptions.

The ingester does not use `.persistent(false)`, session advisory locks, PostgreSQL
`LISTEN`/`NOTIFY`, temporary tables, or session-level `SET`. Its advisory locks are
`pg_advisory_xact_lock` calls made inside explicit transactions, so their required affinity ends
at transaction commit. PgBouncer's `max_prepared_statements` support preserves named prepared
statements across transaction-pooled backend changes. Grafana's PostgreSQL datasource does not
depend on backend session state and uses the same transaction-pooled route safely.

## Connection budget

The global PgBouncer ceiling remains 30 server connections while both route sets coexist,
leaving ten PostgreSQL slots for migration, administration, and exceptional direct access. The
transaction-pooled steady-state budgets are:

- trading: 8 session-pooled backends
- ingester master: 2 transaction-pooled backends
- dynamically scaled ingester workers: 8 transaction-pooled backends for up to 96 clients
- Grafana: 2 transaction-pooled backends

The legacy route and three session-pooled service aliases remain configured as rollback paths,
but idle aliases do not reserve servers. PostgreSQL continues to accept 40 total connections.
Each ingester container has one strategy client and one control client; transaction pooling
multiplexes those clients across the bounded backend pool rather than reserving two PostgreSQL
sessions for every worker replica. Additional clients wait behind PgBouncer's bounded pool.

## How a database error fails a trading process

The failure is intentionally fail-closed:

1. `load_checkpoint_as_of` attaches the context `failed to load point-in-time BTC orderbook
   checkpoint` to the database error.
2. The strategy callback in `btc/runtime.rs` records the error in runtime metrics, logs
   `BTC strategy callback failed; terminating the trading process`, and exits the task because
   primary run data is immutable.
3. The process manager's liveness reconciliation in `main.rs` observes the stopped child,
   prefixes the reason with `btc_runtime_failed`, and calls `stop_process_for_generation` with
   `runtime_failed=true`.
4. `stop_process_for_generation` selects terminal status `failed` and calls
   `mark_btc_process_terminal`.
5. `mark_btc_process_terminal` updates `polymarket.trading_processes`: it sets `status` to
   `failed`, disables the process, records `stopped_at`, and stores the reason in `last_error`.

This behavior prevents a process with incomplete primary-path evidence from continuing to
trade. A failed process is not silently resumed after service restart. Recovery must use the
administrative API, supply a globally unique `next_experiment_key`, validate the start preview,
and run process-scoped live preflight before restarting any live-capital process.

## 2026-08-16 cutover incident

### Why all 17 runtimes entered `failed`

The initial PgBouncer cutover used transaction pooling. Every active BTC process eventually
called `load_checkpoint_as_of`, whose SQLx query uses `.persistent(false)` to retain its
TimescaleDB custom-plan behavior. Transaction pooling broke the query's required backend
affinity between SQLx's `Parse + Sync` and `Bind + Execute` messages. PostgreSQL consequently
returned `unnamed prepared statement does not exist` to each runtime.

The error was not caused by connection-pool exhaustion, the number of trading processes, or a
bad trading-process configuration. It was the protocol incompatibility between transaction
pooling and the existing unnamed-statement query. The shared code path in
`btc/repository.rs::load_checkpoint_as_of` propagated the database error into
`btc/runtime.rs::run_strategy_loop`; the strategy loop intentionally terminated, and the
manager's fail-closed terminal transition in `main.rs` marked each affected process `failed`.

The corrective configuration is PgBouncer `pool_mode = session`. Session affinity preserves
the existing SQLx protocol behavior. The pool was also sized to 24 database sessions for the
17-process baseline; after cutover verification, PgBouncer reported 18 active clients, zero
waiting clients, and 18 active server connections.

### Why Grafana later showed only 15/17 trading-enabled

This was separate from lifecycle failure and separate from database pooling. All 17 process
records were `running`, and all 17 runtime-status responses reported `active=true`,
`capability_enabled=true`, `runtime.enabled=true`, and `runtime.running=true`.

The two excluded processes were the live pilots. Grafana's **Trading Entry Status** panel treats
a paper process with `execute_signals=true` as enabled, but for a live process it additionally
requires the latest relevant lifecycle event to be `btc_live_entries_enabled` with
`entries_enabled=true`. Restarting the bot appended `btc_runtime_resumed` after the live pilots'
earlier `btc_live_entries_enabled` events. The panel therefore conservatively evaluated both
live pilots as disabled and displayed 15/17, even though their process lifecycle was running.

The affected processes were:

- `btc-5m-asymmetric-core-oracle-live-pilot-20260814-v1`
- `btc-5m-directional-model-boundary-alignment-live-pilot-v2`

Both were re-enabled through their process-scoped `/live/entries/enable` control endpoints.
The endpoint performed its strict reconciliation and returned `entries_enabled=true`,
`order_submit_enabled=true`, and `process_accounting_proven=true` for each process. Those calls
recorded new `btc_live_entries_enabled` events, after which the exact Grafana query returned
17/17.
