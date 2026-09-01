# Ingester

The ingester continuously records causal source facts and executes historical backfills. It does not construct features, labels, datasets, predictions, or trading decisions.

The service is an independent Cargo package and container. It connects directly to public provider feeds in parallel with the trading bot; neither process depends on the other for live market data.

## Boundaries

- TimescaleDB is the only durable store.
- Database-backed profiles control each ingester strategy.
- Every data dimension is implemented as an isolated strategy module.
- Provider facts use source-native identity and causal timestamps.
- Gaps are explicit; watermarks never advance over uncommitted data.
- The package has no dependency on `polymarket-bot`.
- `ingester-master` is the sole backfill API, scheduler, and job-ledger owner.
- Horizontally scalable `ingester-worker` containers run realtime and backfill strategies.
- The complete strategy and queue standard is [documented here](docs/strategy-and-backfill-contract.md).

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

The canonical master and worker images compile the same registry. Runtime start and stop decisions come from `ingester.profiles`, not Cargo features or provider-specific environment variables.

```bash
cargo fmt --check --manifest-path packages/market-data-ingester/Cargo.toml
cargo clippy --manifest-path packages/market-data-ingester/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path packages/market-data-ingester/Cargo.toml --all-targets
```

Create an ignored `.env.market-data-ingester` from the repository template and
set a random administrative token of at least 32 bytes. Database credentials
continue to come from `.env.postgres`. The example files contain names only;
never commit populated secret files.

Chainlink strategies additionally require the provider credentials named in
`.env.market-data-ingester.example`. Credentials remain process-level secrets:
they are never accepted through profile JSON, persisted in artifacts, or
returned by the control API. Profiles are seeded stopped, so providers without
configured credentials do not prevent the service from running its other
strategies.

From the repository root, build and start the opt-in development service with:

```bash
docker compose \
  -f docker-compose.yml \
  -f packages/market-data-ingester/docker-compose.yml \
  --profile data-ingestion \
  up -d ingester-master ingester-worker
```

The service joins the existing TimescaleDB network and does not start or depend
on the trading bot. Its API is bound to `127.0.0.1:8099` by Compose, avoiding
the trading bot's development port.

## Control API

Liveness and readiness are unauthenticated:

- `GET /health/live`
- `GET /health/ready`

Administrative routes require `Authorization: Bearer <token>`:

- `GET /strategy/all`
- `GET /strategy/{strategy_key}`
- `GET|POST /backfills`
- `GET /backfills/{job_id}`
- `GET /backfills/{job_id}/events`
- `POST /backfills/{job_id}/cancel`
- `POST /backfills/{job_id}/retry`
- `GET /workers`

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

- `ingester` owns profiles, leases, health, capture lineage, explicit gaps, the sole backfill job ledger, worker heartbeats, events, and artifacts.
- `market_data` owns immutable provider facts.

It does not write to, alter, or reference bot-owned `polymarket` tables. Capture
artifact identity records lineage and is never part of a source fact's natural
identity. Replaying the same provider fact is therefore idempotent; the same
identity with different factual content is an integrity error.
