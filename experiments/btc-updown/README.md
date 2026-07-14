# BTC Up/Down shadow ML experiments

This package contains standard-library-only research scaffolding for point-in-time datasets,
grouped chronological splits, residual logistic models, and probability evaluation.

It does **not** contain a trained alpha model and has no trading authority. The runtime-v2 schema
canary artifacts intentionally reproduce their supplied priors and exist only to verify the feature
and inference contract across Python and Rust.

The trainer is deliberately restricted to ML-A settlement-residual labels and requires an explicit
immutable training cutoff. ML-B currently means FOK fill probability and post-fill toxicity for the
actual taker execution path; its label builders and trained models remain gated on sufficient paper
orders/fills and are not synthesized from settlement labels.

All Python research and test execution runs in the `btc-ml-research` Compose service. The image is
built from this directory's `Dockerfile`, contains no third-party runtime dependencies, runs as an
unprivileged user, and defaults to the complete Python test suite. From the repository root:

```bash
docker compose build btc-ml-research
docker compose run --rm btc-ml-research
```

The service reserves `/datasets` for immutable input datasets and `/artifacts` for candidate model
artifacts. With those paths mounted by Compose, train an ML-A research candidate with an explicit
cutoff entirely inside the container:

```bash
docker compose run --rm btc-ml-research \
  python -m btc_updown_ml.trainer \
  /datasets/ml-a-training.jsonl \
  /artifacts/ml-a-candidate.json \
  --model-version btc-5m-ml-a-candidate-v1 \
  --training-cutoff-ms 1783900800000
```

The emitted artifact remains shadow-only. Training does not promote it into the bot image or grant
it execution authority.

The cross-language contract is pinned in `fixtures/runtime_v2_contract.json`. It contains the
ordered feature schemas, model metadata, immutable hashes, and a deterministic feature vector.
Python validates it in `tests/test_runtime_contract.py`; Rust consumes the same JSON file in
`packages/polymarket-bot/tests/ml_runtime_contract.rs`. The default Compose research job verifies
the Python side:

```bash
docker compose run --rm btc-ml-research
```

The Rust side is verified by the bot's containerized build/test job. Neither Python nor Cargo is
run directly on the host.
