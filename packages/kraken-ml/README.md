# Kraken ML benchmark

This package qualifies whether 15-minute PF_XBTUSD market data contains a
reproducible one-hour classical-ML edge after conservative execution costs.
It neither places orders nor reads from or writes to the existing trading
infrastructure.

The benchmark:

- verifies the SHA-256 suffix of every relevant Kraken lake object before
  materializing content-addressed raw and feature snapshots on the external SSD;
- compares multinomial logistic regression, histogram gradient boosting, and
  extremely randomized trees across six expanding walk-forward folds;
- calibrates probabilities and selects action thresholds only on chronological
  slices that precede each evaluation window;
- freezes model, feature-set, probability calibration, and action thresholds
  before a one-time locked-holdout evaluation;
- reports predictive quality, calibration, cost-stressed net expectancy,
  stability, and held-out feature importance.

The content-addressed Parquet lake is the canonical source. The normalized
PostgreSQL analytics table is explicitly excluded from training because an
integrity audit found a wrong-row child-index lookup and a corresponding
slippage coverage gap. This package does not mutate that database.

## Environment

Python 3.11–3.13 is supported. From this directory:

```bash
python3 -m venv .venv
.venv/bin/pip install --requirement requirements.lock
.venv/bin/pip install --editable . --no-deps --no-build-isolation
```

## Commands

```bash
.venv/bin/python -m kraken_ml prepare \
  --config configs/pf_xbtusd_15m_1h.toml
.venv/bin/python -m kraken_ml develop \
  --config configs/pf_xbtusd_15m_1h.toml
.venv/bin/python -m kraken_ml evaluate \
  --config configs/pf_xbtusd_15m_1h.toml \
  --run-id RUN_ID
```

`develop` materializes deterministic features across the configured range, but
never uses holdout rows for model fitting, calibration, policy selection, or
development evaluation. It compares all three models using the full feature
set, compares price/flow/full feature sets for the selected model, and freezes
a final candidate only when every development and threshold gate passes.
`evaluate` refuses an unqualified run and globally seals the market/time-window
holdout on first access, so a second run cannot reuse the same fixed holdout.

## CPU policy

The checked-in configuration reserves two host cores. Model/fold comparisons
run as up to ten independent processes with one estimator thread apiece.
Only the final pre-holdout refit may use up to ten threads. Joblib, OpenMP,
OpenBLAS, MKL, Accelerate, and NumExpr limits are set by the launcher to prevent
nested oversubscription; Polars is also limited to one thread per comparison
process. These classical estimators use CPU on macOS; Metal is not required.

Datasets, predictions, and model artifacts are written beneath
`/Volumes/docker-data/kraken-ml`. Concise reproducibility and benchmark reports
are written to `reports/latest`.
