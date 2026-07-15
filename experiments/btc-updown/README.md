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

Python model training and research tests run directly on the development host. The package has no
third-party runtime dependencies. From `experiments/btc-updown`:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

Train an ML-A research candidate with explicit host paths and an immutable cutoff:

```bash
PYTHONPATH=src python3 -m btc_updown_ml.trainer \
  /path/to/ml-a-training.jsonl \
  /path/to/ml-a-candidate.json \
  --model-version btc-5m-ml-a-candidate-v1 \
  --training-cutoff-ms 1783900800000
```

The emitted artifact remains shadow-only. Training does not promote it into the bot image or grant
it execution authority.

The cross-language contract is pinned in `fixtures/runtime_v2_contract.json`. It contains the
ordered feature schemas, model metadata, immutable hashes, and a deterministic feature vector.
Python validates it in `tests/test_runtime_contract.py`; Rust consumes the same JSON file in
`packages/polymarket-bot/tests/ml_runtime_contract.rs`. The standard bot image build verifies the
Rust side before producing the release binary.
