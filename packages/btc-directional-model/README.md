# BTC directional model

This package trains and evaluates the supervised BTC five-minute Up/Down prediction source.
It is deliberately separate from the live Rust runtime: Python performs bounded historical
extraction, feature engineering, fitting, calibration, and reporting on the local workstation.
The eventual trading container consumes only a verified model artifact and performs inference
natively in Rust.

The v1 data contract uses only the confirmed backfill sources:

- `polymarket.btc_interval_markets` for market identity and official outcome labels;
- `polymarket.btc_market_reference_facts` for the opening boundary and final-price audit;
- `polymarket.binance_one_second_klines` for point-in-time BTC path and flow features;
- `polymarket.btc_market_execution_snapshots` for optional point-in-time execution-book features.

The April 21-May 20 source cohort contains 8,509 markets with an exact opening boundary and
official outcome. The point-in-time feature contract additionally requires a dense one-second
history through the last candidate decision, leaving 8,508 training markets. Final-price
availability never selects the training cohort, and final prices are never model inputs. Live-only
feature tables, strategy decisions, retired ML tables, and `experiment_id` are not part of the
contract.

## Local environment

Create a package-local virtual environment and install the exact validated environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e . --no-deps
```

Database credentials remain outside Git. The extractor reads `BTC_MODEL_DB_HOST`,
`BTC_MODEL_DB_PORT`, `BTC_MODEL_DB_NAME`, `BTC_MODEL_DB_USER`, and either
`BTC_MODEL_DB_PASSWORD` or `POSTGRES_PASSWORD`.

Run the complete April 21-May 20 workflow:

```bash
.venv/bin/btc-directional-model run \
  --config configs/btc-5m-directional-logistic-v1.toml
```

Generated source data, features, runs, reports, and model artifacts are package-local and ignored
by Git for this implementation pass.

The extractor is read-only, executes daily bounded queries, streams rows to Parquet, and verifies
cached partitions against the SQL/configuration provenance manifest. Feature construction reads
only the partitions named in that manifest. Use `--force` only when intentionally replacing a
changed extraction or feature contract.

## Training contract

Markets are split chronologically and kept disjoint across fitting, calibration, and holdout
evaluation. The calibration period is itself divided chronologically: its first half fits
probability calibration, while its second half selects the feature group and confidence policy.
Preprocessing and regularization selection are fitted without reading later split statistics. Each
market contributes equal total training weight despite having 37 candidate prediction timestamps.

The package compares a BTC-path model with a book-augmented challenger, selects the feature group
and confidence threshold using pre-holdout evidence, and evaluates the frozen choice once on the
chronological holdout. Threshold selection is accuracy-first: after enforcing the minimum market,
accuracy, and confidence-bound contract, it chooses the strongest Wilson lower bound instead of
the earliest broad-coverage threshold. A model is deployable only when `qualification_passed` is true and
`deployment_status` is `qualified` in the model artifact. First-executable accuracy is reported as
a separate timing-policy diagnostic and never overrides the primary qualification contract.

Once a holdout report has been viewed, that date range is consumed. Do not tune against its result
and then describe a rerun on the same markets as independent evidence. Freeze any revised
accuracy/coverage/timing policy on training and calibration data, then use newly backfilled later
dates for the next untouched evaluation. The checked-in configuration therefore sets
`holdout_is_independent = false`: current v2 reruns are exploratory and their artifacts remain
blocked. Change that flag only when the configured final chronological split contains genuinely
unseen, complete four-table coverage.

## Apple Silicon

The canonical model is L2 logistic regression because it is transparent, compact, and directly
portable to native Rust inference. Scikit-learn's solver has no Metal backend, and this workload
fits in seconds on CPU; moving it to MPS would change the numerical implementation without
improving model quality. Apple GPU benchmarking is reserved for a later nonlinear challenger,
where compute volume can justify MPS or MLX. The run evidence records the CPU decision, hardware,
dependency versions, source hash, and convergence state.

## Evidence

Each run writes strict JSON metrics, a checksummed nonbinary model artifact, golden inference
vectors, holdout predictions, a confusion-matrix CSV, and a self-contained Plotly report. Open the
report in a local browser with:

```bash
.venv/bin/btc-directional-model serve --run runs/<run-id> --port 8765
```
