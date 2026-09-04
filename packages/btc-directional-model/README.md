# BTC directional model

This package trains and evaluates the supervised BTC five-minute Up/Down prediction source.
It is deliberately separate from the live Rust runtime: Python performs bounded historical
extraction, feature engineering, fitting, calibration, and reporting on the local workstation.
The eventual trading container consumes only a verified model artifact and performs inference
natively in Rust.

The original four-table data contract uses the confirmed backfill sources:

- `polymarket.btc_interval_markets` for market identity and official outcome labels;
- `polymarket.btc_market_reference_facts` for the opening boundary and final-price audit;
- `market_data.binance_spot_btcusdt_one_second_ohlcv` for point-in-time BTC path and flow features;
- `polymarket.btc_market_execution_snapshots` for optional point-in-time execution-book features.

Execution-book evidence reads the canonical snapshot table one completed PMXT artifact at a time
and selects the 90–140 second decision points from its retained causal 250 ms observations.
`btc_market_decision_execution_snapshots` is a legacy compact materialization and is not a model
input. Both five-share and ten-share executable ask VWAP remain available in the canonical rows;
the evidence manifest records which snapshot schema versions were used.

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
deployment holdout. The current training round is clamped to the exact half-open UTC interval
`[2026-04-21, 2026-07-20)` and makes no deployment decision.

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

Validate the frozen source and feature cache for the current April 21-July 20 development round:

```bash
export POLARS_MAX_THREADS=6
.venv/bin/btc-directional-model core-extract \
  --config configs/btc-5m-directional-core-accuracy-timing-20260421-20260720.toml \
  --scope pre_holdout
.venv/bin/btc-directional-model core-features \
  --config configs/btc-5m-directional-core-accuracy-timing-20260421-20260720.toml \
  --scope pre_holdout
```

The cache covers `[2026-04-21, 2026-07-20)`. Its configured fit, calibration, policy-selection,
and walk-forward splits provide leakage-resistant development comparisons, but none is an
independent holdout. The current round does not configure or access data outside that interval.

Fit the four real accuracy/timing candidates and persist checksummed causal probability evidence:

```bash
export POLARS_MAX_THREADS=6
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
.venv/bin/btc-directional-model persistence-benchmark-run \
  --config configs/btc-5m-directional-accuracy-timing-20260421-20260720.toml
```

The checked-in saved-policy configuration is deliberately a non-runnable template. After the
command above finishes, materialize a temporary config under the package's ignored `data/`
directory and replace
`__PERSISTENCE_RUN_ID__` with the emitted UTC run identifier. Do not invoke the loader while the
placeholder remains:

```bash
PERSISTENCE_RUN_ID="<UTC run identifier>"
sed "s/__PERSISTENCE_RUN_ID__/${PERSISTENCE_RUN_ID}/g" \
  configs/btc-5m-directional-saved-policy-20260421-20260720.toml.template \
  > data/btc-5m-directional-saved-policy-runtime.toml
.venv/bin/btc-directional-model persistence-policy-benchmark-run \
  --config data/btc-5m-directional-saved-policy-runtime.toml
```

This second command does not fit a model. Within each walk-forward fold it selects the four
time-band confidence thresholds only from the earlier policy-selection probabilities, then scores
the corresponding chronological validation probabilities once. The manifest and every Parquet
input are checksum verified.

Run the frequency-only qualification separately:

```bash
PERSISTENCE_RUN_ID="<UTC run identifier>"
sed "s/__PERSISTENCE_RUN_ID__/${PERSISTENCE_RUN_ID}/g" \
  configs/btc-5m-directional-frequency-policy-20260421-20260720.toml.template \
  > data/btc-5m-directional-frequency-policy-runtime.toml
.venv/bin/btc-directional-model frequency-policy-benchmark-run \
  --config data/btc-5m-directional-frequency-policy-runtime.toml
```

This benchmark applies the frequency qualification contract to the enriched control and all
three saved time-calibrated path-persistence candidates. It selects one
deterministic four-band threshold vector from fold zero's policy-selection cohort, verifies that
the policy predates every validation fold, and applies the same vector unchanged to all five
validation folds. Validation never performs a threshold search. Qualification preserves the
87.4% accuracy, balanced-accuracy, and directional-recall floors; the 86.5% Wilson lower bound;
the 5% ECE ceiling; five-fold stability; exact 60/90/120/180/240-second non-regression; coverage
uplift; and positive five-share execution economics with at least 500 executable markets. Median
entry remains visible in every result but is diagnostic-only for this explicitly frequency-focused
model. The early-entry workflow retains the separate 125-second timing requirement.

Train and qualify the fold-robust frequency challenger only after the saved candidate matrix has
no qualifier:

```bash
export POLARS_MAX_THREADS=6
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
.venv/bin/btc-directional-model persistence-benchmark-run \
  --config configs/btc-5m-directional-fold-robust-frequency-20260421-20260720.toml

FOLD_ROBUST_RUN_ID="<UTC run identifier>"
sed "s/__FOLD_ROBUST_RUN_ID__/${FOLD_ROBUST_RUN_ID}/g" \
  configs/btc-5m-directional-fold-robust-frequency-policy-20260421-20260720.toml.template \
  > data/btc-5m-directional-fold-robust-frequency-policy-runtime.toml
.venv/bin/btc-directional-model frequency-policy-benchmark-run \
  --config data/btc-5m-directional-fold-robust-frequency-policy-runtime.toml
```

The challenger retains the enriched control direction at every timestamp and raises confidence
only when a separately fitted pre-window outcome model agrees. Auxiliary hyperparameters maximize
the worst direction metric across exact 60/90/120/180/240-second checkpoints in three expanding
chronological folds within each outer training cohort; coverage and outer validation do not select
the model. The exact-checkpoint direction metrics therefore cannot regress from the control on
common rows. The second command still applies the single anchor-fold policy and all frozen
frequency gates before any holdout or runtime work.

Run the boundary-aligned training session without accessing the external holdout:

```bash
export POLARS_MAX_THREADS=6
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
.venv/bin/btc-directional-model core-features \
  --config configs/btc-5m-directional-core-boundary-aligned-20260421-20260720.toml \
  --scope pre_holdout
.venv/bin/btc-directional-model persistence-benchmark-run \
  --config configs/btc-5m-directional-boundary-alignment-20260421-20260720.toml
```

This session preserves the 58-feature histogram control and adds a separate outcome model with
causal official-boundary distance, crossing, persistence, time-on-side, volatility-normalized
distance, and short-horizon boundary-momentum features. The control retains its global calibration;
the challenger uses frozen 60-89, 90-119, 120-179, and 180-240-second calibration bands. The
output is consumed development evidence only; it does not export a runtime model or access
July 21-August 4.

Run the March 21-July 28 boundary-reversal accuracy session:

```bash
export POLARS_MAX_THREADS=6
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
.venv/bin/btc-directional-model core-extract \
  --config configs/btc-5m-directional-core-boundary-reversal-20260321-20260729.toml \
  --scope pre_holdout
.venv/bin/btc-directional-model core-features \
  --config configs/btc-5m-directional-core-boundary-reversal-20260321-20260729.toml \
  --scope pre_holdout
.venv/bin/btc-directional-model persistence-benchmark-run \
  --config configs/btc-5m-directional-boundary-reversal-accuracy-20260321-20260729.toml
```

This consumed-development session compares the current 58-feature outcome control with one
106-feature continuation-versus-reversal challenger. The additional causal fields describe
longer-horizon momentum, excursion, pullback/recovery, boundary-cross density, volatility shock,
and path-sign-normalized price and flow. Final-price availability does not select the cohort and
final prices are never model inputs. The challenger must preserve all existing accuracy,
direction-recall, timing, coverage, calibration, and execution-economics gates. It must also
reduce the absolute number of wrong out-of-fold first-crossing decisions made at 95% or greater
confidence, without worsening that error rate per selected trade. This requirement qualifies the
trained artifact only; it does not select thresholds, veto individual trades, or add runtime
trading logic. A new forward paper cohort beginning after artifact freeze remains required for
independent qualification.

Run the causal early-entry and NoTrade residual-admission benchmark:

```bash
export POLARS_MAX_THREADS=6
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
.venv/bin/btc-directional-model core-extract \
  --config configs/btc-5m-directional-core-residual-admission-20260321-20260721.toml \
  --scope pre_holdout
.venv/bin/btc-directional-model core-features \
  --config configs/btc-5m-directional-core-residual-admission-20260321-20260721.toml \
  --scope pre_holdout
.venv/bin/btc-directional-model persistence-benchmark-run \
  --config configs/btc-5m-directional-residual-admission-source-20260321-20260721.toml

SOURCE_RUN_ID="<UTC source run identifier>"
sed "s/__SOURCE_RUN_ID__/${SOURCE_RUN_ID}/g" \
  configs/btc-5m-directional-residual-admission-20260321-20260721.toml.template \
  > data/btc-5m-directional-residual-admission-runtime.toml
.venv/bin/btc-directional-model residual-admission-benchmark-run \
  --config data/btc-5m-directional-residual-admission-runtime.toml
```

The source session writes checksum-verified out-of-fold probabilities for the fixed
`histogram_enriched` control and `histogram_boundary_reversal` proposal across seven chronological
folds. The residual runner fits independent early `[60,120)` and rescue `[120,241)` correctness
heads using only older source folds. The immediately prior fold is divided chronologically between
direction-specific calibration and frozen threshold selection. A proposal is eligible only after
two exact five-second control/proposal direction agreements and only while the 0.89 control has not
already crossed. Control decisions always retain timestamp priority.

Both heads use the common frozen q grid `[0.87, 0.89, 0.91, 0.93]` and fail closed to the control
when no threshold qualifies. Qualification requires all five evaluation folds to preserve the
87.4% accuracy, balanced-accuracy, and directional-recall floors, the 86.5% Wilson lower bound,
the 5% calibration ceiling, control non-regression, residual-only quality, and positive
five-share economics. The early head must also improve median entry by at least five seconds,
advance decisions by at least ten seconds, and add at least two percentage points of decisions by
second 120. The rescue head must reduce NoTrade by at least two percentage points. Residual
economics require at least 500 executable decisions; missing execution evidence fails closed.
This consumed-development benchmark never exports a runtime model or changes a trading process.

When a checksum-matched March 21-July 21 source cache already exists elsewhere, it can be copied
without a database read:

```bash
.venv/bin/btc-directional-model core-snapshot-residual-source \
  --config configs/btc-5m-directional-core-residual-admission-20260321-20260721.toml \
  --source-dir <existing-source-cache>
```

The snapshot operation creates a new no-overwrite cache, verifies every source partition, and uses
hard links where the filesystem permits them.

Run the narrower March 21-July 28 mature-reversal accuracy session:

```bash
export POLARS_MAX_THREADS=6
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
.venv/bin/btc-directional-model core-features \
  --config configs/btc-5m-directional-core-boundary-reversal-20260321-20260729.toml \
  --scope pre_holdout
.venv/bin/btc-directional-model persistence-benchmark-run \
  --config configs/btc-5m-directional-mature-reversal-accuracy-20260321-20260729.toml
```

This accuracy-focused session compares the current 58-feature direct-outcome model with one
71-feature direct-outcome challenger under the same five chronological folds, global Platt
calibration, threshold search, and equal-total-per-market weighting. The 13 added fields use only
causal Binance path excursion, pullback, recovery, recency, short-horizon return, and signed-flow
history already present in the compact core dataset. They do not use the official opening
boundary, final price, a correctness selector, an admission model, or a post-prediction veto.

Advancement requires the existing absolute accuracy standards, at least 0.1 percentage-point
aggregate improvements in accuracy, balanced accuracy, and Wilson lower confidence, strictly
positive UP and DOWN recall changes, coverage non-regression, five qualified chronological folds,
and at least one fewer wrong first-crossing prediction at 95% or greater confidence without a
worse selected-trade error rate. Early coverage, exact-time comparisons, entry timing,
path-persistence uplift, and execution economics remain reported diagnostics; this session does
not claim an earlier-entry or trade-frequency improvement. Because March 21-July 28 has been
consumed during development, a paper process starting after artifact freeze is still required for
independent qualification.

Run the corrected rolling regime-robust accuracy session:

```bash
export POLARS_MAX_THREADS=6
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
.venv/bin/btc-directional-model core-features \
  --config configs/btc-5m-directional-core-boundary-reversal-20260321-20260729.toml \
  --scope pre_holdout \
  --force
.venv/bin/btc-directional-model persistence-benchmark-run \
  --config configs/btc-5m-directional-regime-robust-accuracy-20260321-20260729.toml
```

This session uses seven fixed rolling validations from June 9 through July 28. Each fold has an
expanding fit interval followed by separate seven-day calibration, seven-day policy selection,
and validation intervals. The last fold therefore evaluates July 21-28 with a model whose fit,
calibration, and policy roles end before that validation begins. It corrects the older
proportional-fold protocol, which did not reproduce the final model's consecutive seven-day
calibration and policy windows.

The frozen matrix contains the 58-feature control, the 71-feature mature-reversal candidate, and
three isolated training ablations: a 28-day estimator-only recency half-life, a 77-feature variant
with six longer-horizon causal regime fields, and a 71-feature histogram with market-scale leaf
regularization. Probability calibration never receives recency decay. Every challenger remains a
direct-outcome model with global Platt calibration; no gate, admission model, correctness
selector, or execution policy changes its decisions. Advancement additionally requires every
validation fold to meet the absolute accuracy, direction-recall, Wilson, calibration, and coverage
standards. July 21-28 is consumed development evidence, so a newly frozen model still requires a
post-freeze paper cohort for independent qualification.

Run the rolling correctness-admission benchmark from a completed boundary session:

```bash
BOUNDARY_RUN_ID="<UTC run identifier>"
sed "s/__BOUNDARY_RUN_ID__/${BOUNDARY_RUN_ID}/g" \
  configs/btc-5m-directional-correctness-admission-20260421-20260720.toml.template \
  > data/btc-5m-directional-correctness-admission-runtime.toml
.venv/bin/btc-directional-model admission-benchmark-run \
  --config data/btc-5m-directional-correctness-admission-runtime.toml
```

For evaluation folds 2, 3, and 4, the correctness selector fits only older out-of-fold
predictions. It splits the immediately prior fold chronologically between four-band Platt
calibration and exhaustive time-band policy selection, then scores the next fold once. A
`q < 0.5` decision always abstains and never reverses the boundary model's direction. These three
folds can provide development evidence only; five validation folds plus independent holdout
evidence remain mandatory before any runtime export or live-capital decision.

Run the strict-book chronology diagnostic separately:

```bash
export POLARS_MAX_THREADS=6
.venv/bin/btc-directional-model entry-benchmark-run \
  --config configs/btc-5m-directional-strict-book-chronology-20260527-20260720.toml
```

Its compact-book fitting evidence spans `[2026-05-27, 2026-06-12)`, with chronological fit,
calibration, and policy partitions. Its separate consumed evaluation cohort is
`[2026-07-16, 2026-07-20)`. Extraction starts at second 55 solely to seed exact 60-second causal
book deltas; reported decision checkpoints remain 60, 90, 120, 180, and 240 seconds.

Run the compact orderbook-residual challenge:

The checked-in residual configuration intentionally pins the generated
`20260728T151930Z` accuracy/timing benchmark and probability Parquet under the ignored `runs/`
directory. A clean checkout must either restore those exact checksum-matched artifacts or first
run the accuracy/timing workflow, then update the residual config's OOF run id, paths, and both
SHA-256 values as one reviewed provenance change. The runner fails closed when either artifact is
missing, changed, or not a valid chronological walk-forward source.

```bash
export POLARS_MAX_THREADS=6
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
.venv/bin/btc-directional-model entry-benchmark-run \
  --config configs/btc-5m-directional-book-residual-20260527-20260720.toml
```

This workflow keeps the universal BTC model as the prediction source on every eligible market and
adds a compact L2-regularized correction only when both Polymarket books have strict ten-share
validity and an exact prior five-second observation. Non-strict rows preserve the calibrated BTC
core probability bit-for-bit. Residual fitting uses pinned out-of-fold BTC-core probabilities;
regularization selection, stability, direction/time calibration, confidence selection, and the
consumed policy diagnostic remain chronological and disjoint. The post-July-20 cohort remains
sealed and no runtime model, Rust contract, container image, trading process, or database state is
changed by this command.

The accuracy/timing workflow evaluates four real histogram-gradient-boosting candidates over five
chronological walk-forward folds. Every market contributes equal total fitting weight. The two new
challengers apply 1.5x weight at 60-120 seconds and 2x weight at 90-120 seconds respectively,
without changing the existing 58-feature native-runtime contract, live inference inputs, or the
Rust trading path.

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

The regime-robust recency candidate has a separate, explicit paper-only path. It retains the exact
estimator fitted on March 21 through July 13, applies global Platt calibration from July 14 through
July 20, and uses July 21 through July 28 only for its locked `0.87` first-crossing policy. Export
fails if that threshold does not retain at least 55% eligible-market coverage. This command never
changes the production gates: its freeze and native manifests remain
`production_qualified = false` and `live_capital_allowed = false` until an independent post-freeze
cohort is evaluated.

```bash
.venv/bin/btc-directional-model persistence-paper-candidate-export \
  --config configs/btc-5m-directional-regime-robust-accuracy-20260321-20260729.toml \
  --benchmark-run runs/btc-regime-robust-accuracy-20260321-20260729/<run-id> \
  --freeze-root artifacts/btc-directional-recency-paper \
  --runtime-output-root runtime-models \
  --model-key btc-5m-directional-mature-reversal-recency-28d-paper-v1 \
  --authorize-paper-only
```

The exact-120 mature-reversal path preserves the predecessor estimator's five causal
120/125/130/135/140-second training rows, while calibration, causal coverage-threshold selection,
validation, and runtime inference use only the 120-second row. Its 71 model features contain no
oracle or order-book fields; historical book evidence is joined only after scoring for separate
five-share and ten-share execution economics. The primary paper operating point targets 15%
coverage and the secondary diagnostic targets 10%.

```bash
.venv/bin/btc-directional-model fixed-120-benchmark-run \
  --config configs/btc-5m-directional-mature-reversal-fixed-120-20260321-20260729.toml

.venv/bin/btc-directional-model fixed-120-paper-candidate-export \
  --config configs/btc-5m-directional-mature-reversal-fixed-120-20260321-20260729.toml \
  --benchmark-run runs/btc-mature-reversal-fixed-120-20260321-20260729/<run-id> \
  --model-key btc-5m-directional-mature-reversal-fixed-120-paper-v1 \
  --authorize-paper-only
```

The export command fails unless the benchmark's primary operating point passes accuracy,
balanced-accuracy, both directional-recall, Wilson, calibration, hard-confidence-tail, and
ten-share positive-expectancy checks. The exported prediction policy is exactly `120/120/5` and
remains paper-only pending independent forward evidence.

The same exporter supports the two time-banded paper candidates without adding a second
training or runtime subsystem. The frequency candidate must bind the causal four-band policy
selected by its frequency benchmark:

```bash
.venv/bin/btc-directional-model persistence-paper-candidate-export \
  --config configs/btc-5m-directional-accuracy-timing-20260421-20260720.toml \
  --benchmark-run runs/btc-accuracy-timing-20260421-20260720/20260728T151930Z \
  --policy-benchmark-run runs/btc-frequency-policy-20260421-20260720/20260728T201621Z \
  --candidate histogram_path_persistence_time_calibrated_60_120 \
  --freeze-root artifacts/btc-directional-path-persistence-paper \
  --runtime-output-root runtime-models \
  --model-key btc-5m-directional-path-persistence-60-120-frequency-20260421-20260720-paper-v1 \
  --authorize-paper-only
```

Boundary alignment derives its repeated per-band confidence threshold from the final
policy-selection range under the already frozen benchmark configuration:

```bash
.venv/bin/btc-directional-model persistence-paper-candidate-export \
  --config configs/btc-5m-directional-boundary-alignment-20260421-20260720.toml \
  --benchmark-run runs/btc-boundary-alignment-20260421-20260720/20260729T002520Z \
  --candidate histogram_boundary_enriched \
  --freeze-root artifacts/btc-directional-boundary-alignment-paper \
  --runtime-output-root runtime-models \
  --model-key btc-5m-directional-boundary-alignment-20260421-20260720-paper-v1 \
  --authorize-paper-only
```

The frequency artifact keeps all 100 ordered features (58 core plus 42 pre-window features) and
the path-persistence-to-UP conversion. The boundary artifact keeps its exact 68-feature schema.
Both use runtime-model v2 with four calibrated time bands and remain paper-only.

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

Once labels or a report have been viewed, that date range is consumed. Do not tune against its
result and then describe a rerun on the same markets as independent evidence. The checked-in
accuracy/timing configurations therefore set `evaluation_is_independent = false`, disable the core
holdout, and keep every result development-only.

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
qualification.

## Asymmetric-value hunter

The offline asymmetric-value benchmark trains price-aware challengers to find lower-priced YES or
NO claims whose calibrated probability exceeds exact five-share VWAP, fee, and a conservative
execution reserve. Its primary policy search is restricted to raw share prices below 30 cents; the
wider price policies are diagnostics only. Predictions are made every second from seconds 1–59 and
every five seconds from seconds 60–240. Every candidate preserves the runtime contract's 25%
maximum depth participation, so a five-share entry requires at least 20 shares of selected-side
depth. Accuracy is reported but is not an admission gate.

```bash
.venv/bin/btc-directional-model asymmetric-value-benchmark-run \
  --config configs/btc-5m-directional-asymmetric-value-one-second-20260414-20260802.toml
```

Selection is sealed before the frozen evaluation data is opened. Missing exact books are NoTrade,
not losses or proxy prices, and evidence sufficiency is evaluated separately from point-estimate
economics. The matrix contains price logistic, core-plus-PMXT, Binance L2, closed Chainlink candle,
and causal Polygon Chainlink oracle arms, with same-cohort controls for incomplete optional sources.
It deliberately excludes multi-source kitchen-sink models. An enriched sparse-source arm remains
selection-eligible only when it is non-inferior to its same-key control on point expectancy,
opportunity yield, and paired UTC-day net profit. Calibration fits one coherent complementary
YES/NO probability jointly by decision-time band and raw 10-cent side-price cells, with explicit
parent-time fallback when a cell lacks 50 markets, five UTC days, or both outcomes. Historical
RefPrice is excluded because local receipt time is not proven. The frozen current-policy reference
preserves first 89% confidence crossing and its 30–95 cent execution range. A separate diagnostic
varies only the retrained core-plus-price model's 50–89% confidence threshold under a fixed 1–240
second, 70-cent cost, positive-edge, size, and depth contract. The frozen-model low-price diagnostic
uses the value policy only over its artifact-supported 60–240 second interval; it cannot answer the
pre-60 hypothesis. This command never exports a runtime model or changes the trading pipeline. Add
`--force` only when intentionally rebuilding all source caches.

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
