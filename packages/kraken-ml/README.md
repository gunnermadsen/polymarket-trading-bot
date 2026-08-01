# Kraken ML benchmark

This package qualifies whether 15-minute PF_XBTUSD market data contains a
reproducible classical-ML edge after conservative execution costs. It retains
the original one-hour classifier as a historical control and adds a direct
net-expectancy regression benchmark across one-hour and four-hour horizons.
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

The net-expectancy benchmark fits independent long and short return regressors.
It runs exactly eight preregistered Ridge, histogram-gradient-boosting, and
ExtraTrees candidates across six expanding walk-forward folds. Price-only,
open-interest-only, and price-plus-open-interest candidates are evaluated under
the same taker-fee policy, with maker and zero-fee outcomes retained only as
fixed-action diagnostics.

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
.venv/bin/python -m kraken_ml backfill-funding \
  --config configs/pf_xbtusd_15m_expectancy.toml
.venv/bin/python -m kraken_ml prepare-expectancy \
  --config configs/pf_xbtusd_15m_expectancy.toml
.venv/bin/python -m kraken_ml develop-expectancy \
  --config configs/pf_xbtusd_15m_expectancy.toml
.venv/bin/python -m kraken_ml evaluate-expectancy \
  --config configs/pf_xbtusd_15m_expectancy.toml \
  --run-id RUN_ID
```

`develop` materializes deterministic features across the configured range, but
never uses holdout rows for model fitting, calibration, policy selection, or
development evaluation. It compares all three models using the full feature
set, compares price/flow/full feature sets for the selected model, and freezes
a final candidate only when every development and threshold gate passes.
`evaluate` refuses an unqualified run and globally seals the market/time-window
holdout on first access, so a second run cannot reuse the same fixed holdout.

`develop-expectancy` predicts long and short net basis-point returns directly,
chooses each fold policy only on a chronological threshold slice, and evaluates
pooled walk-forward economics. A candidate must pass every economic and
stability gate before its modal non-no-trade development policy is written to
a checksum-recorded preregistration artifact. That policy is fixed before the
final pre-holdout confirmation window is loaded or predicted.
`evaluate-expectancy` refuses any run without that checksum-validated freeze
and uses the same market/time-window-global one-use holdout seal as the
classifier benchmark.

`backfill-funding` downloads Kraken's first-party funding-rate ZIP export and
the documented `historical-funding-rates` JSON response. It requires exact
Decimal agreement across their overlap, preserves both source objects and an
immutable provenance manifest, and fills only absent 15-minute lake buckets
using the active continuously accrued per-hour rate. Existing rows are never
overwritten. Existing chart rows must match the relative rate exactly; the
absolute rate permits at most `1e-12` difference because Kraken's charts source
truncates that field. Any larger disagreement fails before Parquet publication,
and the tolerance is recorded in the immutable manifest. For offline or
repeatable execution, pass `--archive-path` and `--recent-json` with previously
downloaded first-party files.

## CPU policy

The checked-in configuration reserves two host cores. The expectancy benchmark
runs 48 candidate/fold jobs as up to ten independent processes with one
estimator thread apiece; long and short estimators are fit sequentially inside
each job.
Only the final pre-holdout refit may use up to ten threads. Joblib, OpenMP,
OpenBLAS, MKL, Accelerate, and NumExpr limits are set by the launcher to prevent
nested oversubscription; Polars is also limited to one thread per comparison
process. These classical estimators use CPU on macOS; Metal is not required.

Datasets, predictions, and model artifacts are written beneath
`/Volumes/docker-data/kraken-ml`. Concise reproducibility and benchmark reports
are written to `reports/latest`. Funding source archives and their immutable
lineage live under the Kraken SSD lake, not in Git.
