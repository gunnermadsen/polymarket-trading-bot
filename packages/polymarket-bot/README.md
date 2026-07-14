# polymarket-bot

Rust microservice for Polymarket negative-risk arbitrage scanning, simulation, execution-state management, and Postgres/Timescale persistence.

## v1 scope

- Direct Gamma/CLOB REST and CLOB WebSocket integration.
- Shared execution pipeline for `sim` and `live`.
- Postgres persistence in the `polymarket` schema.
- Kafka is intentionally not required. Future Kafka event publishing should be added behind a disabled adapter only after the service has proven stable.

## Trading Processes

Trading mode is controlled by rows in `trade_processes`, not by a process-wide
environment variable. A single bot runtime can execute `sim` and `live` processes
in parallel. Live processes require CLOB credentials and wallet secrets to be
present in the runtime environment, and host configuration only supplies venue
URLs plus hard risk caps.

## Build and test

```bash
docker compose --profile test build polymarket-bot polymarket-bot-test
docker compose --profile test run --rm --no-deps polymarket-bot-test
```

## Derive CLOB Credentials

From the repository root, add the wallet private key secret to `.env`:

```bash
POLYMARKET_PRIVATE_KEY=...
```

Use the production Compose definition for this separately authorized live-setup
operation; the realtime-paper Compose environment deliberately blanks every
live credential:

```bash
docker compose -f docker-compose.production.yml run --rm --no-deps \
  --entrypoint derive-clob-creds polymarket-bot
```

The command prints:

```bash
POLYMARKET_CLOB_API_KEY=...
POLYMARKET_CLOB_SECRET=...
POLYMARKET_CLOB_PASSPHRASE=...
```

Paste those values back into the root `.env`. The `.env` file is ignored by Git.
`POLYMARKET_FUNDER_ADDRESS` and `POLYMARKET_SIGNATURE_TYPE` are also secrets for
live execution and belong in `.env`.

From the repository root:

```bash
docker compose build polymarket-bot
docker compose up polymarket-bot
```

## Kafka placeholder

Kafka publishing is deliberately inactive in v1. If enabled later, the event boundary should be after persistence and reconciliation:

```rust
// future:
// kafka.publish("polymarket.signals.v1", signal_event).await?;
// kafka.publish("polymarket.orders.v1", order_event).await?;
// kafka.publish("polymarket.fills.v1", fill_event).await?;
// kafka.publish("polymarket.risk_events.v1", risk_event).await?;
```
