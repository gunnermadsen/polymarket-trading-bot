# polymarket-bot

Rust microservice for Polymarket negative-risk arbitrage scanning, simulation, execution-state management, and Postgres/Timescale persistence.

## v1 scope

- Direct Gamma/CLOB REST and CLOB WebSocket integration.
- Shared execution pipeline for `sim` and `live`.
- Postgres persistence in the `polymarket` schema.
- Kafka is intentionally not required. Future Kafka event publishing should be added behind a disabled adapter only after the service has proven stable.

## Run modes

```text
POLYMARKET_EXECUTION_MODE=sim
POLYMARKET_EXECUTION_MODE=live
```

`sim` is the default. `live` refuses to start unless `POLYMARKET_LIVE_CONFIRM=true` is set and wallet/order credentials are present.

## Build

```bash
cargo test
cargo build --release
```

## Derive CLOB Credentials

From the repository root, add the wallet private key secret to `.env`:

```bash
POLYMARKET_PRIVATE_KEY=...
```

Then run:

```bash
cargo run --manifest-path packages/polymarket-bot/Cargo.toml --bin derive-clob-creds
```

The command prints:

```bash
POLYMARKET_CLOB_API_KEY=...
POLYMARKET_CLOB_SECRET=...
POLYMARKET_CLOB_PASSPHRASE=...
```

Paste those values back into the root `.env`. The `.env` file is ignored by Git.
Non-secret live settings, including `POLYMARKET_FUNDER_ADDRESS` and
`POLYMARKET_SIGNATURE_TYPE`, belong in the root `docker-compose.yml`.

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
