# BTC Candidate Loss-Risk Tournament

Status: **frozen training plan**  
Frozen at: `2026-09-09T14:31:00Z`  
Champion collection: `btc-5m-frozen-champion-collection-20260902`  
Runtime implementation: **not started**

This document freezes the intended training and evaluation contract for a single
candidate loss-risk system. Opening a sealed cohort, changing an evaluation
boundary, adding a feature family, changing a label, or changing a qualification
rule requires a new explicitly versioned plan. This plan does not authorize model
training, runtime refactoring, database mutation, deployment, or promotion.

## Pilot posture

This tournament is a research pilot whose immediate purpose is to learn whether a
risk-oriented model can add value in replay of the existing historical tournament
data. It is not a production-readiness exercise, and its data is expected to be
imperfect.

Do not delay or invalidate the pilot merely because optional sources have gaps,
some champion/time/side slices are small, historical coverage is uneven, or the
first implementation cannot reproduce every desirable feature. Preserve missing
values, report the gaps, use the common subset that can be reproduced causally,
and run the experiment. Low-sample slices are diagnostics rather than independent
qualification gates for this pilot.

The pilot's hard data stops are limited to conditions that would make its result
misleading: future leakage, incorrect champion or artifact identity, unavailable
official outcome labels, invalid executable-price/PnL arithmetic, or an inability
to reproduce the no-risk baseline closely enough to make the comparison
meaningful. Other limitations must be recorded but are not blockers.

The pilot may produce an interesting backtest result without qualifying a model
for runtime integration, shadow operation, live capital, or production. Those
decisions require separate evidence and authorization.

## Objective

Train one standardized risk model that estimates the probability that a proposed
BTC five-minute entry produces negative net PnL after executable price and fees.
The risk model evaluates only entries already admitted by the base champion. It
cannot approve a base-model rejection or bypass market, capital, accounting,
identity, order, or data-readiness controls.

The tournament measures whether risk intervention avoids more loss dollars than
profit dollars while preserving useful opportunity coverage. Classification
accuracy is diagnostic, not the selection objective.

## Frozen champion population

Evaluate the ten champions recorded in
`packages/btc-directional-model/training-results/btc-5m-frozen-champion-collection-20260902/manifest.json`.
Champion identity and artifact SHA-256 are retained for lineage and reporting.
The primary pooled risk model does not use champion identity as a predictive
feature.

## Cohorts

### Construction

- Interval: `[2026-03-21T00:00:00Z, 2026-08-14T00:00:00Z)`.
- Use only causal out-of-fold champion predictions.
- Fit risk estimators and their probability calibration.
- Keep all rows from one market in the same chronological fold.

### Policy selection

- Interval: `[2026-08-14T00:00:00Z, 2026-09-01T00:00:00Z)`.
- Select the risk feature treatment, estimator, probability threshold, and
  coverage constraint.
- Compare every learned policy with the no-risk champion and simple matched-
  coverage controls.
- This interval becomes consumed risk-model development evidence.

### Locked September replay

- Interval: `[2026-09-02T19:30:00Z, 2026-09-09T14:30:00Z]`.
- The start follows the recording of the last artifact in the frozen champion
  population.
- Score this cohort exactly once after the risk artifact, feature schema,
  calibration, threshold, replay rules, and qualification rules are frozen.
- Do not refit, recalibrate, change thresholds, add champion exceptions, or
  substitute feature semantics after opening it.
- Record that aggregate behavior for part of this period was observed before the
  risk tournament, although no candidate risk model was trained or selected from
  it.

### Forward shadow confirmation

- Begins strictly after the selected risk artifact and runtime policy are frozen.
- Persist allow/defer decisions and later official outcomes without allowing the
  risk strategy to alter actual trading during initial confirmation.
- Deferred candidates remain observable and labelable so risk state can recover.

## Canonical Parquet panel

The authoritative training-data source is the manifest-verified Parquet archive
under `/Volumes/docker-data`. Realtime database tables are staging sources only
until their copy-only drains are complete; tournament training must not query the
database as its dataset. Source manifests, partition hashes, schemas, and causal
availability fields must be validated before panel construction.

Create one row per `champion x market x scheduled candidate time`. Preserve:

- champion key and immutable artifact identity;
- market, window, candidate, and causal availability timestamps;
- entry second and the `60_89`, `90_119`, `120_149`, or `150_180` bucket;
- predicted side, raw probability, calibrated probability, conservative
  probability, and confidence;
- original base admission result, reason, policy outputs, and selected-entry
  identity;
- five-share executable VWAP, fees, spread, overround, depth, slippage, book age,
  expected edge, and frozen stress assumptions;
- common causal market features reproducible by the intended UMR contract;
- causally resolved recent performance state, separately for UP and DOWN and for
  the four entry-time buckets;
- official outcome, counterfactual net PnL, stress PnL, and direction correctness.

Unavailable optional inputs remain explicit missing values. Historical and live
missing-value semantics must match. A richer historical feature cannot be used
unless the intended UMR runtime can reproduce its exact source, clock, age, and
missing-value contract.

## Dynamic history features

Construct these only from candidate outcomes whose official labels were available
before the row's candidate timestamp:

- consecutive losses;
- win/loss rates over the prior 5, 10, and 25 resolved candidates;
- exponentially weighted Brier and confidence-weighted error;
- recent hypothetical net PnL and stress PnL;
- expected-versus-observed win-rate gap;
- side-specific and time-bucket-specific versions of the above.

Risk-deferred candidates continue to enter this shadow history after resolution.
Actual execution is not required for the risk model to observe subsequent model
quality.

## Labels

- Primary classification label: `net_pnl <= 0`.
- Economic outcome: counterfactual net PnL after executable cost and fees.
- Stress outcome: counterfactual PnL under the frozen stress execution scenario.
- Direction correctness and loss magnitude are auxiliary diagnostics.

## Tournament contestants

1. No-risk champion baseline.
2. Raised base-confidence control at matched coverage.
3. Raised minimum-edge control at matched coverage.
4. Regularized logistic candidate-loss classifier.
5. Calibrated gradient-boosted classifier using candidate and entry economics.
6. Calibrated gradient-boosted classifier adding common market context.
7. Calibrated gradient-boosted classifier adding causally resolved recent model
   performance.

Each learned contestant emits a continuous loss probability. Threshold candidates
are selected only on the policy-selection cohort.

## Sequential policy replay

Replay candidates in chronological order within each champion and market. A risk
deferral at an earlier candidate does not end the market; a later base-admitted
candidate may be allowed. The first candidate allowed by both the champion and
the risk policy becomes the entry, after which later candidates are not eligible.

## Evaluation

Report pooled results and the complete `10 champions x 4 time buckets x 2 sides`
slice matrix. For every contestant and slice report:

- allowed winners and losses;
- blocked winners and losses;
- avoided loss dollars and missed profit dollars;
- net risk value added;
- resulting net PnL and stress PnL;
- coverage retained;
- profit factor and expectancy;
- maximum drawdown;
- calibration and sample count.

Low-sample slices remain visible and are marked statistically insufficient rather
than optimized independently. Run a leave-one-champion-out diagnostic to measure
whether learned risk behavior generalizes across champion identities.

## Qualification

A risk contestant qualifies only when it:

- increases net PnL over the corresponding no-risk champion baseline;
- improves stress PnL and reduces maximum drawdown;
- avoids more loss dollars than the profit dollars it misses;
- satisfies the coverage floor frozen from policy-selection evidence;
- outperforms matched-coverage confidence and edge controls;
- has no materially destructive UP, DOWN, champion, or entry-time slice; and
- does not derive its pooled improvement from only one champion or one bucket.

The exact numerical coverage floor and loss-probability threshold are selected and
frozen from the policy-selection cohort before the September replay is opened.
For this pilot, these qualification results are comparative research findings, not
production admission gates. A contestant that misses one or more criteria remains
reportable and useful for deciding whether the risk-model idea warrants another
training round.

## Intended runtime boundary

Runtime work follows model qualification. The additive trading-playbook field is
`strategy.risk_strategies`, an array with an empty default for backward
compatibility. The first supported entry is `candidate_loss_risk_v1` and carries
immutable model, feature-schema, and policy identities.

UMR supplies standardized candidate evidence and invokes risk evaluation after
base-model admission and before order creation. Risk state is process-scoped,
bounded, causally rebuildable, and incapable of changing durable process
enablement or another process's eligibility.
