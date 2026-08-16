# PostgreSQL connection pooling

The Docker Compose environments route application connections through PgBouncer on port
`6432`. Database migrations remain connected directly to TimescaleDB on port `5432` so
migration sessions retain normal PostgreSQL semantics.

PgBouncer uses session pooling for the `polymarket` database. Do not change it to transaction
pooling without first redesigning and verifying the SQLx queries described below.

## Why session affinity is required

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
