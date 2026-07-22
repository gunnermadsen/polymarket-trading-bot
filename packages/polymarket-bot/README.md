# polymarket-bot

Rust microservice for Polymarket negative-risk arbitrage scanning, execution-state management, and Postgres/Timescale persistence.

## v1 scope

- Direct Gamma/CLOB REST and CLOB WebSocket integration.
- Dedicated BTC realtime-paper execution through `btc::paper::PaperVenue`.
- Preserved authenticated Polymarket live-execution and reconciliation foundation.
- Postgres persistence in the `polymarket` schema.
- Kafka is intentionally not required. Future Kafka event publishing should be added behind a disabled adapter only after the service has proven stable.

## Trading Processes

Trading activity is controlled by rows in `trading_processes`, not by a
process-wide environment variable. Managed BTC process definitions are
paper-only and use the dedicated BTC paper venue. The legacy generic simulator
is retired. The admin API can list historical process rows, but definition
upsert, update, start, and stop operations are restricted to
`btc_5m/realtime_paper`; collection `POST` is unsupported. Authenticated
live-execution controls still require CLOB credentials and wallet secrets in the
runtime environment.

## Build and test

The standard bot image build runs its Rust unit, integration, and contract tests
before producing the release binary.

```bash
./scripts/build-polymarket-bot-image.sh
```

The build refuses an uncommitted worktree and records the full source commit in
the `org.opencontainers.image.revision` OCI label.

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
./scripts/build-polymarket-bot-image.sh
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
