# BTC directional model

This package trains and evaluates the supervised BTC five-minute Up/Down prediction source.
It is deliberately separate from the live Rust runtime: Python performs bounded historical
extraction, feature engineering, fitting, calibration, and reporting on the local workstation.
The eventual trading container consumes only a verified model artifact and performs inference
natively in Rust.

The original four-table data contract uses the confirmed backfill sources:

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

The expanded universal BTC-core contract is intentionally narrower. It uses only interval-market
labels, reference facts, one-second Binance klines, and completed-artifact lineage. It does not
read PMXT snapshots, raw orderbook events, aggregate trades, Chainlink ticks, or any live trading
process data. The historical April 21-July 20 labels have already been accessed during model
development, so results on that interval are development evidence rather than an independent
deployment holdout. A deployment decision now requires a genuinely unseen, complete interval
after July 20.

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

Run the expanded BTC-core workflow:

```bash
export POLARS_MAX_THREADS=6
.venv/bin/btc-directional-model core-run \
  --config configs/btc-5m-directional-core-20260421-20260620.toml
```

Develop the extended April 21-July 20 candidate:

```bash
export POLARS_MAX_THREADS=6
.venv/bin/btc-directional-model core-extract \
  --config configs/btc-5m-directional-core-20260421-20260720.toml \
  --scope pre_holdout
.venv/bin/btc-directional-model core-features \
  --config configs/btc-5m-directional-core-20260421-20260720.toml \
  --scope pre_holdout
.venv/bin/btc-directional-model core-develop \
  --config configs/btc-5m-directional-core-20260421-20260720.toml
```

The extended contract uses `[2026-04-21, 2026-07-21)`. Its configured chronological splits remain
useful for leakage-resistant development comparisons, but they are no longer independent holdout
evidence. Do not promote a candidate from this range. Freeze the candidate and its policy before
extracting or evaluating a later complete interval.

The core workflow evaluates two logistic candidates and two real
histogram-gradient-boosting candidates over five chronological walk-forward folds. The
`histogram_early_weighted` candidate preserves the existing 58-feature native-runtime contract
while assigning more fitting weight to decisions 60-120 seconds into each market. It does not
change live inference inputs or the Rust trading path.

Run the frozen earlier-entry benchmark:

```bash
export POLARS_MAX_THREADS=6
.venv/bin/btc-directional-model entry-benchmark-run \
  --config configs/btc-5m-directional-entry-benchmark-20260421-20260720.toml
```

This compares the control and deploy-compatible early-weighted model with two additional real
offline challengers: a pre-open Binance-context model and a strict-valid-book model. It reads the
compact 250 ms execution snapshots—not raw PMXT archive events—in daily bounded queries, builds
checksummed exact-five-second evidence, and evaluates all candidates on common market/timestamp
checkpoints. Book quality fields route and qualify observations; they are never directional model
features. Offline challengers cannot be deployed through the current 58-feature Rust contract.

Generated source data, features, runs, reports, and model artifacts are package-local and ignored
by Git for this implementation pass.

The extractor is read-only, executes daily bounded queries, streams rows to Parquet, and verifies
cached partitions against the SQL/configuration provenance manifest. Feature construction reads
only the partitions named in that manifest. Use `--force` only when intentionally replacing a
changed extraction or feature contract.

## Native runtime export

The standardized runtime export converts a checksummed frozen histogram candidate into immutable,
Python-free JSON for native inference. It verifies the freeze-manifest checksum, training-joblib
checksum, model-summary checksum, feature order, calibrator, confidence policy, and golden-feature
cache before reading estimator internals. It does not reevaluate or access holdout labels.

Export the selected model:

```bash
.venv/bin/btc-directional-model core-export-runtime \
  --freeze artifacts/btc-core-20260421-20260620/20260726T234955Z-histogram_enriched \
  --golden-features data/btc-core-20260421-20260620/features/holdout.parquet \
  --output-root runtime-models \
  --model-key btc-5m-directional-histogram-enriched-20260421-20260620-v1
```

Each model key owns exactly three files:

- `model.json` contains the ordered feature and median-imputation contract, baseline logit, numeric
  histogram trees, probability calibration, `no_trade` confidence behavior, prediction timing,
  and immutable training provenance;
- `golden-vectors.json` covers Up, Down, `no_trade`, confidence boundaries, and non-finite
  imputation with exact expected raw logits and calibrated probabilities;
- `manifest.json` binds the model and golden vectors by SHA-256 and repeats the feature-schema
  identity needed at container startup.

The feature-schema SHA-256 is calculated from the UTF-8 compact, key-sorted JSON object
`{"names":[...],"schema_version":"..."}`. Feature order is therefore part of the identity.
Exporting the same key from the same inputs is idempotent. A key that already exists with different
bytes is rejected; a newly trained replacement must use a new immutable model key. This keeps model
replacement repeatable without permitting a running process's model identity to change in place.

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

For the expanded BTC core, candidate selection is likewise chronological but completely independent
of orderbook quality. The qualification contract requires at least 65% accuracy and balanced
accuracy, at least 60% recall in both directions, at least 50% coverage, and a 60% Wilson lower
bound. The model must also be non-inferior to the same-time Binance path sign in every fold and at
the lower bound of the hourly block bootstrap. Same-time path uplift remains an explicit diagnostic:
zero uplift means the model is a selective path-persistence predictor—its value is calibrated
confidence and abstention, not reversal identification. The raw Gamma-boundary/Binance price
difference is retained only for source audit and is excluded from every model allowlist because
cross-venue basis drift is not BTC direction. Passing these gates qualifies only the prediction
model. The earlier-entry benchmark separately evaluates executable five-share VWAP, the configured
dynamic fee, direct edge, realized net expectancy, and drawdown without altering the trading
runtime.

The earlier-entry benchmark uses its own stricter advancement contract. A candidate must clear the
absolute accuracy, balanced-accuracy, directional-recall, Wilson, calibration, coverage, and
fixed-five-share economics gates; it must also improve coverage and entry timing without regressing
the control's prediction quality. Native p99 latency and serialized runtime-model size evidence are
also mandatory; missing measurements fail closed. Passing development gates is not deployment
qualification. A genuinely independent post-July-20 holdout remains mandatory.

## Apple Silicon

The canonical model is L2 logistic regression because it is transparent, compact, and directly
portable to native Rust inference. Scikit-learn's solver has no Metal backend, and this workload
fits in seconds on CPU; moving it to MPS would change the numerical implementation without
improving model quality. Apple GPU benchmarking is reserved for a later nonlinear challenger,
where compute volume can justify MPS or MLX. The run evidence records the CPU decision, hardware,
dependency versions, source hash, and convergence state.

## Evidence

The original workflow writes strict JSON metrics, a checksummed nonbinary model artifact, golden
inference vectors, holdout predictions, a confusion-matrix CSV, and a self-contained Plotly report.
The expanded BTC-core workflow writes a training-only joblib artifact, a portable JSON summary when
the selected family is logistic, source and feature hashes, walk-forward/policy predictions,
holdout access evidence, and the same style of self-contained report. The explicit runtime-export
command is the only path from that training artifact to a checked-in native model bundle. The
earlier-entry benchmark also writes candidate timing bands, exact-checkpoint comparisons,
execution-economics evidence, data-quality lineage, gate decisions, and a self-contained report.
Generated reports and extracted data remain ignored by Git. Open a report in a local browser with:

```bash
.venv/bin/btc-directional-model serve --run runs/<run-id> --port 8765
```
