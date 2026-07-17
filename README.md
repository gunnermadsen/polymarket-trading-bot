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
- `.env.postgres`: canonical source for `POSTGRES_PASSWORD`.
- `.env.grafana`: Grafana admin credentials and datasource settings.

Generate a local Grafana admin password with:

```bash
openssl rand -hex 32
```

Non-secret runtime configuration belongs in `docker-compose.yml`.

```bash
docker compose up -d timescaledb-0
docker compose up --build db-migrate
docker compose build polymarket-bot
docker compose up -d polymarket-bot grafana
```

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
independently of that global refresh. Current process and P&L
panels include only enabled processes in a running lifecycle with
a heartbeat/update no older than two minutes. Stopped experiments remain in
the realized-P&L history but do not contribute to current realized or
unrealized totals. Recreate Grafana after changing the provisioned JSON:

```bash
docker compose up -d --no-deps --force-recreate grafana
```

## Live Credentials

Set the CLOB credential secrets in `.env`:

```bash
POLYMARKET_CLOB_API_KEY=...
POLYMARKET_CLOB_SECRET=...
POLYMARKET_CLOB_PASSPHRASE=...
POLYMARKET_PRIVATE_KEY=...
POLYMARKET_FUNDER_ADDRESS=...
POLYMARKET_SIGNATURE_TYPE=...
```

Starting or stopping the `polymarket-bot` container does not select trading
activity. Trading process definitions are mutable and API-driven; use
`POST /admin/trading-processes/{process_id}/start` and
`POST /admin/trading-processes/{process_id}/stop` while leaving the service
running. Each BTC paper start creates a distinct immutable experiment while the
stable process key can be reused. Trading mode is selected by
`trading_processes.config.execution.mode`; host configuration is limited to
credentials, venue URLs, and hard risk caps.
