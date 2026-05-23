# Polymarket Bot Standalone Stack

Standalone Docker stack for the Polymarket bot, its Polymarket-only migrations, and a dedicated TimescaleDB database.

## Services

- `timescaledb-0`: dedicated TimescaleDB/Postgres database.
- `db-migrate`: one-shot TypeORM migration runner with only `polymarket` migrations.
- `polymarket-bot`: Rust bot service.

Kafka and pgbouncer are intentionally omitted.

## Local Start

Create a local `.env` from `.env.example` and fill in the secret values. Non-secret runtime
configuration belongs in `docker-compose.yml`.

```bash
docker compose up -d timescaledb-0
docker compose up --build db-migrate
docker compose up -d --build polymarket-bot
```

Health:

```bash
curl http://127.0.0.1:8097/health
curl http://127.0.0.1:8097/metrics
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

Trading mode is not selected through environment variables. `sim` and `live` are
controlled by `trade_processes.config.execution.mode`, so live and sim processes
can run side by side without restarting the container. Host live configuration is
limited to credentials, venue URLs, and hard risk caps.
