# Polymarket Bot Standalone Stack

Standalone Docker stack for the Polymarket bot, its Polymarket-only migrations, and a dedicated TimescaleDB database.

## Services

- `timescaledb-0`: dedicated TimescaleDB/Postgres database.
- `db-migrate`: one-shot TypeORM migration runner with only `polymarket` migrations.
- `polymarket-bot`: Rust bot service.
- `grafana`: provisioned Grafana instance with the Postgres datasource and `polymarket-bot` dashboard.

Kafka and pgbouncer are intentionally omitted.

## Local Start

Create local env files from the examples and fill in the secret values:

- `.env`: app and Polymarket secrets.
- `.env.postgres`: canonical source for the PostgreSQL administrator password.
- `.env.postgres.roles`: canonical source for application database-role passwords.
- `.env.postgres.<service>`: generated least-privilege credential mounted into one service.
- `.env.grafana`: Grafana admin credentials and datasource settings.

Generate the application database credentials from the primary checkout without printing
their values:

```bash
./scripts/bootstrap-postgres-service-credentials.sh
```

The generator refuses to overwrite an existing credential set. Keep the generated files out
of linked worktrees and do not commit them.

Generate a local Grafana admin password with:

```bash
openssl rand -hex 32
```

Non-secret runtime configuration belongs in `docker-compose.yml`.

```bash
docker compose up -d timescaledb-0
docker compose up --build db-migrate
./scripts/build-polymarket-bot-image.sh
docker compose up -d polymarket-bot grafana
```

The bot-image build refuses an uncommitted worktree and records the full source
commit in the `org.opencontainers.image.revision` OCI label.

Starting the service leaves BTC trading inactive. Next, create or update the
stopped API-controlled process definition and call that process's `/start`
endpoint. Trading activity remains controlled by persistent process state, not
by starting or stopping the bot container.

The local Compose file publishes Postgres on
`POLYMARKET_POSTGRES_HOST_PORT` (default `55433`) and bot HTTP on
`POLYMARKET_HTTP_HOST_PORT` (default `8098`).

TimescaleDB is capped at 2 GB and starts with matching PostgreSQL memory,
connection, and worker limits. Keep the explicit `postgres -c` settings in the
Compose file: the `TS_TUNE_*` variables tune only a newly initialized data
volume and do not repair an existing volume by themselves. Grafana is limited
to five open and two idle datasource connections so it fits within the
25-connection PostgreSQL budget.

Health:

```bash
curl -fsS "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/health"
curl -fsS "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/metrics"
```

Grafana:

Open `http://127.0.0.1:3030` in a browser.

Log in with `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` from `.env.grafana`.
The provisioned dashboard refreshes every 30 seconds. The BTC five-minute
countdown uses a provisioned Grafana Live channel and updates every second,
independently of that global refresh. The Trading process selector at the top
of the dashboard controls every forecast and lead/lag panel, and supplies the
process used by PnL panels when PnL scope is Selected process. The neighboring
PnL scope selector controls only the first PnL section; All processes renders
per-process chart series and table rows while aggregating the PnL stats.
Forecast and lead/lag panels remain scoped to the Trading process selection.
Stopped and disabled process history remains available. Process health remains
a fleet-wide view, and the market countdown is process-independent. Recreate
Grafana after changing the provisioned JSON:

```bash
docker compose up -d --no-deps --force-recreate grafana
```

## Live Credentials

Set only the CLOB authentication secrets in `.env`:

```bash
POLYMARKET_CLOB_API_KEY=...
POLYMARKET_CLOB_SECRET=...
POLYMARKET_CLOB_PASSPHRASE=...
POLYMARKET_PRIVATE_KEY=...
```

`POLYMARKET_FUNDER_ADDRESS` and `POLYMARKET_SIGNATURE_TYPE` are non-secret
account identity settings. They are mapped explicitly by both Compose files;
the funder address is supplied as deployment-specific Compose configuration and
the signature type defaults to `POLY_1271` for this pilot.

Starting or stopping the `polymarket-bot` container does not select trading
activity. The admin lifecycle supports managed `btc_5m/realtime_paper`
processes with a process-owned `paper` or `live` execution venue.
Create or replace an inactive definition through
`PUT /admin/trading-processes/by-key/{process_key}`; collection `POST` and
generic process activation are intentionally unsupported. Use
`POST /admin/trading-processes/{process_id}/start` and
`POST /admin/trading-processes/{process_id}/stop` for a resumable stop, or
`POST /admin/trading-processes/{process_id}/complete` for an orderly terminal
completion, while leaving the service running. Each BTC paper start creates a
distinct process-owned run, identified by `run_id`. Immutable run evidence
lives in the existing process lifecycle records. Lifecycle and record ownership
remain canonical to the selected `process_id`; the stable process key can be
reused. Trading mode is selected by
`trading_processes.config.execution.mode`; live definitions also require a
bounded `execution.account_ref`. Optional execution limits and the exit-book
requirement are flat process-owned properties accepted identically by paper and
live venues. When omitted, their checks are not configured. Host configuration
is limited to credentials, venue URLs, and transport timing.

The disabled live pilot is installed by migration with process ID
`effa3e5e-2f5a-4f18-98ba-06e4c0da74ef` and process key
`btc-5m-directional-model-boundary-alignment-live-pilot`. It starts in
credential-validation-only mode:

```json
{
  "mode": "live",
  "execute_signals": false,
  "live_capital": false,
  "account_ref": "polymarket-primary",
  "taker_fee_rate": null
}
```

When configured, execution controls remain flat and use the same names in both
venues:

```json
{
  "max_order_notional_usd": "2",
  "max_open_notional_usd": "20",
  "max_open_positions": 6,
  "max_daily_loss_usd": "10",
  "require_exit_book": true
}
```

Each property is independent and optional. An omitted property does not create
an execution check or receive an environment-derived default.

With both process execution flags false, run the non-trading connectivity and
reconciliation check through:

```text
POST /admin/trading-processes/effa3e5e-2f5a-4f18-98ba-06e4c0da74ef/live-preflight
```

The preflight reads authenticated API-key, collateral/allowance, open-order,
geoblock, and account-reconciliation state. It never signs or submits an order.
It reports credential connectivity independently from activation readiness and
requires the process trading flags to remain disabled.

Verify the private key and configured signing identity separately with the
signing-only diagnostic:

```text
POST /admin/live/order-dry-run
Content-Type: application/json

{
  "token_id": "<current-outcome-token-id>",
  "side": "buy",
  "order_type": "fok",
  "price": "0.50",
  "size": "1"
}
```

This endpoint builds and signs locally but has no order-POST path. Its response
redacts the owner and signature; do not persist the diagnostic response in
application logs.

This branch deliberately cannot activate live capital yet. Process accounting
remains `unproven` because Polymarket position responses are wallet aggregates;
no durable zero-exposure/account-ownership baseline has been established for
the pilot. Exchange redemption is also not integrated, so internally observed
market resolution never releases live exposure or credits reusable capital.
The source model remains `paper_only` as well. These are explicit activation
blockers, not warnings: preflight `ready` remains false, process start is denied,
and the entry gate cannot open.

Future live activation requires a distinct immutable directional-model
artifact whose manifest explicitly permits live capital, both live process
flags enabled together, authenticated user websocket evidence, a durable
process-owned accounting baseline, exchange redemption proof, a successful
process start/reconcile, and the process-scoped live-entry endpoint. Live
activation also enforces the
two-second reference/book freshness ceiling. A checked enable grants exactly
one POST attempt; any user-websocket account event, websocket failure,
reconciliation change, or manual halt consumes or invalidates that grant.
Wallet-wide entry enable is intentionally rejected.

## BTC Five-Minute Chainlink Process Contract

Legacy Chainlink definitions and existing durable processes use
`btc_realtime_paper_process_v2`; new definitions that need explicit strategy
selection use the v3 contract below. The stable identity is
`process_type=btc_5m`, `process_scope=realtime_paper`, plus a unique
`process_key`. Paper definitions execute approved signals and never permit live
capital; live definitions use the same strategy/runtime contract and change
only the process-owned execution venue. Unknown fields inside
`config.raw.btc_realtime_paper` are rejected. In particular, the retired
`ml_shadow` setting is not part of this contract.

Save the following request body as `btc-process-v2.json`. Replace the process
name, `next_experiment_key`, and preregistration digest before creating a real
process. The digest must be exactly 64 hexadecimal characters. The existing
`next_experiment_key` field is the legacy compatibility name for the immutable
run key; lifecycle and record ownership still belong to `process_id`.

<!-- btc-5m-process-v2:start -->
```json
{
  "name": "BTC 5m Chainlink paper",
  "process_type": "btc_5m",
  "process_scope": "realtime_paper",
  "enabled": false,
  "status": "created",
  "config": {
    "execution": {
      "mode": "paper",
      "execute_signals": true,
      "live_capital": false
    },
    "raw": {
      "btc_realtime_paper": {
        "schema_version": "btc_realtime_paper_process_v2",
        "playbook_version": "v1.2",
        "sources": [
          "polymarket_btc_five_minute_market_contracts",
          "polymarket_btc_five_minute_orderbooks",
          "polymarket_btc_five_minute_resolutions",
          "polymarket_rtds_chainlink_reference_price",
          "polymarket_chainlink_btcusd_twap",
          "binance_spot_btcusdt_one_second_ohlcv",
          "polygon_chainlink_btcusd_oracle"
        ],
        "next_experiment_key": "btc-5m-chainlink-paper-example-v1",
        "preregistration_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "strategy": {},
        "runtime": {
          "strategy_interval_ms": 1000,
          "official_resolution_audit_grace_secs": 120,
          "official_resolution_watch_retention_secs": 3600
        },
        "paper": {
          "arrival_latency_ms": 150,
          "visible_depth_haircut": "0.80",
          "starting_collateral_usd": "1000",
          "stress_previews": [
            {
              "scenario_key": "latency_300ms_depth_65pct",
              "arrival_latency_ms": 300,
              "visible_depth_haircut": "0.65"
            },
            {
              "scenario_key": "latency_600ms_depth_50pct",
              "arrival_latency_ms": 600,
              "visible_depth_haircut": "0.50"
            }
          ]
        }
      }
    }
  },
  "metadata": {}
}
```
<!-- btc-5m-process-v2:end -->

Entry admission is optional. When `entry_admission` is absent, BTC process
behavior and its frozen process configuration are unchanged. When the following
block is present beside `strategy`, `runtime`, and `paper`, it is enforced; there
is no passive mode or environment-variable control:

```json
"entry_admission": {
  "loss_regime_confidence_floor": {
    "schema_version": "loss_regime_confidence_floor_v1",
    "activation_consecutive_candidate_losses": 2,
    "min_conservative_probability": "0.50",
    "release_consecutive_candidate_wins": 1
  }
}
```

## Selectable BTC Decision Strategies

`btc_realtime_paper_process_v3` requires an explicit
`strategy.decision_strategy`. The supported selectors are the ML-backed
`btc_directional_model` and `btc_asymmetric_value_model` contracts. Each
selection pins the model key, artifact SHA-256, and feature-schema SHA-256.
Runtime readiness, execution validation, entry admission, accounting, and
settlement remain downstream of model evaluation and are unchanged by strategy
selection.


The loss-regime state is reconstructed from the process's immutable,
configuration-scoped decision history. Each resolved market contributes the
earliest strategy-approved buy candidate produced before its label became
available, including candidates deferred by admission. The ordered state is
rebuilt at the first admission evaluation in each new market, so delayed labels
are incorporated without replaying growing history on every one-second
opportunity. After the configured number of consecutive candidate losses,
approved entries below the conservative probability floor are recorded as
`admission_blocked` without creating an order. Entries at or above the floor
continue through the existing paper execution path. The floor releases after
the configured number of consecutive candidate wins. Admission evidence is
stored with each evaluated buy decision.

The optional `daily_realized_pnl_high_water_mark_v1` admission policy protects
a configurable portion of positive paper PnL without changing probability
estimation. Its state is owned and scoped canonically by `process_id`; an
immutable `run_id` is not used to select, partition, or link policy evidence.
For each UTC day, the policy reconstructs credited realized PnL and its running
high-water mark from existing settlement records. It also reserves the full
entry debit of unresolved paper fills. Once the running peak reaches the
activation amount, the protected floor is `peak - max_drawdown`. A proposed
entry is allowed only when current realized PnL minus unresolved entry debit
minus the proposed limit notional and dynamic fee remains at or above that
floor. Equality is allowed. At a new UTC day the state begins a new daily
period, and before activation the policy does not impose a floor.

The HWM policy composes with the existing loss-regime confidence floor inside
the existing entry-admission path. Both policies are evaluated and persisted in
decision evidence; a defer from either policy prevents the entry. No new
service, worker, table, or migration is introduced.

```json
"entry_admission": {
  "loss_regime_confidence_floor": {
    "schema_version": "loss_regime_confidence_floor_v1",
    "activation_consecutive_candidate_losses": 2,
    "min_conservative_probability": "0.50",
    "release_consecutive_candidate_wins": 1
  },
  "daily_realized_pnl_high_water_mark": {
    "schema_version": "daily_realized_pnl_high_water_mark_v1",
    "activation_realized_pnl_usd": "5",
    "max_drawdown_from_high_water_mark_usd": "5"
  }
}
```

Files under `infra/processes` are managed BTC realtime-paper operational request
templates, not runtime configuration watched or read directly by the Rust
application. The bootstrap script rejects any other process identity, active
status, or `enabled=true`. Submitting a template through the stable-key API (or
bootstrap script) creates or updates a persistent inactive process definition
and returns its `process_id`.
Start-preview is the recommended read-only validation step; the Rust runtime
begins trading only after an explicit start for that `process_id`.

Create the definition through the stable-key API, preview the next
process-owned run, then start it explicitly:

```bash
PROCESS_KEY="btc-5m-chainlink-paper"
PROCESS_ID="$({
  curl -fsS -X PUT \
    "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/admin/trading-processes/by-key/${PROCESS_KEY}" \
    -H "Authorization: Bearer ${POLYMARKET_HTTP_ADMIN_TOKEN}" \
    -H "Content-Type: application/json" \
    --data @btc-process-v2.json
} | jq -r '.process.process_id')"

curl -fsS \
  "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/admin/trading-processes/${PROCESS_ID}/start-preview" \
  -H "Authorization: Bearer ${POLYMARKET_HTTP_ADMIN_TOKEN}"

curl -fsS -X POST \
  "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/admin/trading-processes/${PROCESS_ID}/start" \
  -H "Authorization: Bearer ${POLYMARKET_HTTP_ADMIN_TOKEN}"
```

Use the status and stop endpoints for lifecycle management:

```bash
curl -fsS \
  "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/admin/trading-processes/${PROCESS_ID}/status" \
  -H "Authorization: Bearer ${POLYMARKET_HTTP_ADMIN_TOKEN}"

curl -fsS -X POST \
  "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/admin/trading-processes/${PROCESS_ID}/stop" \
  -H "Authorization: Bearer ${POLYMARKET_HTTP_ADMIN_TOKEN}"

curl -fsS -X POST \
  "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/admin/trading-processes/${PROCESS_ID}/complete" \
  -H "Authorization: Bearer ${POLYMARKET_HTTP_ADMIN_TOKEN}"
```

An existing running v1 process may be reattached after a service restart, but
v1 is resume-only. Once stopped, it must be updated to the v2 contract before
another explicit start. Process type, scope, and stable key are immutable;
name and configuration may be updated only while the process is stopped.
