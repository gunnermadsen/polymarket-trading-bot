# Market Data Ingester

`market-data-ingester` is a development-oriented service that continuously records causal market-source facts for downstream machine-learning workflows. It does not construct features, labels, datasets, predictions, or trading decisions.

The service is an independent Cargo package and container. It connects directly to public provider feeds in parallel with the trading bot; neither process depends on the other for live market data.

## Boundaries

- TimescaleDB is the only durable store.
- Database-backed profiles control each ingester strategy.
- Every data dimension is implemented as an isolated strategy module.
- Provider facts use source-native identity and causal timestamps.
- Gaps are explicit; watermarks never advance over uncommitted data.
- The package has no dependency on `polymarket-bot`.
- The service is not included in production Compose configuration.

## Organization

- `bootstrap`: process construction and shutdown.
- `control`: profile API and reconciliation.
- `domain`: source-neutral ingestion contracts.
- `persistence`: shared TimescaleDB control-plane access.
- `runtime`: strategy registry and supervision.
- `strategies`: provider and data-dimension implementations.
- `telemetry`: logging, health, and metrics.

Source-specific configuration, decoding, recovery, and fact SQL stay inside the owning strategy directory.

## Development

The canonical image compiles every registered strategy. Runtime start and stop decisions come from `ingester.profiles`, not Cargo features or provider-specific environment variables.

```bash
cargo fmt --check --manifest-path packages/market-data-ingester/Cargo.toml
cargo clippy --manifest-path packages/market-data-ingester/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path packages/market-data-ingester/Cargo.toml --all-targets
```

Create an ignored `.env.market-data-ingester` from the repository template and
set a random administrative token of at least 32 bytes. Database credentials
continue to come from `.env.postgres`. The example files contain names only;
never commit populated secret files.

From the repository root, build and start the opt-in development service with:

```bash
scripts/build-market-data-ingester-image.sh
docker compose \
  -f docker-compose.yml \
  -f packages/market-data-ingester/docker-compose.yml \
  --profile market-data-ingestion \
  up -d market-data-ingester
```

The service joins the existing TimescaleDB network and does not start or depend
on the trading bot. Its API is bound to `127.0.0.1:8099` by Compose, avoiding
the trading bot's development port.

## Control API

Liveness and readiness are unauthenticated:

- `GET /health/live`
- `GET /health/ready`

Administrative routes require `Authorization: Bearer <token>`:

- `GET /v1/ingesters`
- `GET /v1/ingesters/{strategy_key}`
- `GET|PUT /v1/ingesters/{strategy_key}/config`
- `POST /v1/ingesters/{strategy_key}/start`
- `POST /v1/ingesters/{strategy_key}/stop`
- `POST /v1/ingesters/{strategy_key}/restart`

Mutations require `If-Match` set to the row's current
`desired_generation`. Configuration updates are validated against the owning
strategy's typed schema before the database generation advances. A changed
generation causes the supervisor to cancel the old instance, release its lease,
and construct the strategy again without restarting the container.

## Database ownership

The package uses the existing TimescaleDB instance through two additive schemas:

- `ingester` owns profiles, leases, health, capture lineage, and explicit gaps.
- `market_data` owns immutable provider facts.

It does not write to, alter, or reference bot-owned `polymarket` tables. Capture
artifact identity records lineage and is never part of a source fact's natural
identity. Replaying the same provider fact is therefore idempotent; the same
identity with different factual content is an integrity error.
