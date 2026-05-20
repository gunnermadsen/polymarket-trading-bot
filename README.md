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

## Reconciliation-Only Live Mode

Set the CLOB credential secrets in `.env`:

```bash
POLYMARKET_CLOB_API_KEY=...
POLYMARKET_CLOB_SECRET=...
POLYMARKET_CLOB_PASSPHRASE=...
```

Then update the corresponding non-secret live-mode flags in `docker-compose.yml`:

```yaml
POLYMARKET_EXECUTION_MODE: "live"
POLYMARKET_LIVE_CONFIRM: "true"
POLYMARKET_LIVE_ORDER_SUBMIT_ENABLED: "false"
POLYMARKET_LIVE_USER_WS_ENABLED: "true"
```

Keep order submission disabled for reconciliation-only validation.
