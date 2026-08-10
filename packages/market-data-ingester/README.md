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

