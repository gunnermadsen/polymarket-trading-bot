# Kraken Futures Model Tournament

## Scope and lineage

This tournament qualifies a classical-machine-learning strategy for Kraken
`PF_XBTUSD` futures. It is rooted at Git commit
`4b2cd089b395ddd7b00b3920e94d8163a1c8c4e7` on
`feature/kraken-futures-model-tournament` and is implemented only in
`packages/kraken-ml` plus training-specific documentation. It must not place
orders, mutate trading-process state, or change the Polymarket trading runtime.
The Polymarket bot container and image are not rebuilt for this work.

The tournament has three adaptive generations. A later generation may use
only immutable development and out-of-fold evidence produced by earlier
generations. Confirmation, holdout, and post-holdout results must never select
features, horizons, models, hyperparameters, or action thresholds.

“Self-improvement” means deterministic evidence-driven model selection and
retuning across generations. It does not mean that a model edits code or
silently changes its objective. Every adaptation is recorded in a generation
decision manifest before the next generation trains.

## Desired outcomes

The completed tournament must answer:

1. Which model, horizon, and Kraken feature family has the strongest stable
   out-of-fold predictive signal?
2. Does that signal produce positive net expectancy after fees, funding,
   spread, slippage, and delayed-entry sensitivity?
3. Which data dimensions contribute incremental out-of-fold value rather than
   merely improving in-sample fit?
4. Does recursive refinement improve a lineage across generations, or does
   each apparent edge disappear under chronological validation and cost stress?
5. Is there one frozen candidate worthy of one-time confirmation and holdout
   evaluation, or should the project stop before a trading bot is built?

Success is not “best accuracy.” Success is a reproducible, cost-adjusted edge
that passes every qualification gate. A clean no-edge result is also a valid
business outcome because it prevents deployment of a losing strategy.

## Data qualification

### Required before Generation 1

The canonical source remains the content-addressed Kraken Parquet lake on the
external SSD. PostgreSQL and the Polymarket trading tables are excluded.
Generation 1 requires continuous, causally timestamped 15-minute coverage for:

| Family | Required dimensions |
|---|---|
| Contract and price | instrument versions, trade candles, mark candles, spot/index candles |
| Positioning | open interest, future basis, first-party funding rates |
| Flow | aggressor differential, CVD, trade volume, trade count, liquidation volume |
| Execution | fee schedule, spread, liquidity, and slippage at the configured notional |
| Lineage | event time, available time, ingest time, source identity, object SHA-256, schema version |

Before training, the package must produce a coverage manifest with row counts,
minimum and maximum event/available times, missing buckets, duplicates,
non-finite values, and source-object hashes per dimension. Funding must have a
new pinned first-party import and verified provenance binding spanning the
fixed tournament cutoff. The raw and feature snapshots are immutable and
content addressed.

Long/short ratio, long/short account information, and top-trader positioning
are optional positioning challengers. They enter a later generation only if
their complete causal coverage spans every applicable development fold. Local
rolling volatility is derived from price data rather than ingested.

Kraken L2 does not block Generation 1. L2 features may enter Generation 2 or 3
only after their capture has a fixed availability timestamp, complete coverage,
and a leakage audit. Until then, spread, liquidity, and slippage snapshots are
the microstructure inputs. Partial L2 history is never backfilled with zeros or
mixed into folds where it was unavailable.

### Feature families

Feature families are evaluated as nested additions so incremental value can be
measured:

1. `price`: stationary returns, ranges, volatility, trend, mark/index gaps.
2. `positioning`: price plus OI change/z-score, basis, funding, and optional
   long/short or top-trader measures.
3. `flow`: positioning plus aggressor differential, CVD, volume, trade count,
   and liquidations.
4. `microstructure`: flow plus spread, depth/liquidity, imbalance, and slippage.
5. `cross_venue`: the winning Kraken family plus qualified external features;
   this is disabled unless separately available and causally complete.

Raw price levels, cumulative values, and other non-stationary quantities are
not passed directly to a model. Each feature records its lookback and
`available_at`; all joins are backward as-of joins.

### External and DeFi data

External data is not required for the initial tournament. If Kraken-only
results justify another data family, priority is:

1. major centralized-exchange BTC perpetual price, funding, OI, flow, and
   liquidation divergence;
2. Deribit implied volatility, skew, and term structure;
3. Hyperliquid BTC perpetual mark, funding, OI, liquidations, flow, and L2;
4. DeFi lending utilization/rates/liquidations and WBTC/cbBTC AMM imbalance or
   wrapper-price dislocation.

Hyperliquid is the most relevant DeFi challenger because it is another
perpetual market. AMM and lending inputs are slower regime variables and must
be tested as hourly or daily gates, not assumed to predict the next 15-minute
direction. No external family advances unless it improves paired OOF folds
after multiplicity correction.

## Objective and evaluation design

The primary target is signed forward gross return in basis points from a
causally executable reference price. The evaluator converts each prediction
into long, short, or no-trade net expectancy after contract math, funding,
taker fees, spread, and notional-specific slippage. The existing independent
long/short net-return regression remains a fixed historical control.

The principal horizons are 30 minutes, 1 hour, 2 hours, 4 hours, and 8 hours
(`2`, `4`, `8`, `16`, and `32` 15-minute bars). The 1-hour horizon is the
primary comparison because it had the strongest weak rank signal in the prior
benchmark. Overlapping labels are permitted for fitting but evaluation trades,
bootstrap blocks, and effective sample-size reporting must account for the
horizon overlap.

The existing six expanding chronological development folds remain the common
scoreboard. Calibration and threshold selection use later development slices
that still precede evaluation. The previously sealed confirmation and holdout
boundaries remain unchanged. Newly recovered data after the old holdout is a
separate recency window and is evaluated last, after the final candidate is
frozen and the holdout has been opened once.

All generations use the same fixed seed set, fold boundaries, fee policy,
notional, bootstrap method, and missing-data policy. Hyperparameters are
selected with three chronological inner folds. There is no shuffled split,
random cross-validation, or test-set early stopping.

## Model roster and practical CPU policy

| Model | Role | Generation 1 horizons | Initial tuning emphasis |
|---|---|---|---|
| Ridge | fixed linear control | 30m, 1h, 2h, 4h, 8h | alpha |
| Elastic Net | sparse linear challenger | 30m, 1h, 2h, 4h, 8h | alpha, L1 ratio |
| Histogram Gradient Boosting | compact nonlinear challenger | 30m, 1h, 2h, 4h | learning rate, leaves, leaf size, L2 |
| Extra Trees | interaction/nonlinearity challenger | 1h, 2h, 4h, 8h | depth, leaf size, feature fraction |
| LightGBM | efficient boosted-tree challenger | 30m, 1h, 2h, 4h, 8h | leaves, leaf size, learning rate, feature fraction, L1/L2 |

LightGBM is CPU-only in this package and is added as a pinned package
dependency. Training processes reserve two host cores, use one estimator
thread per parallel comparison, and prevent nested BLAS/OpenMP
oversubscription. The final refit may use the configured remaining cores.
Each job records wall time, CPU time, effective threads, and peak resident
memory. Metal/GPU support is outside scope.

Generation 1 is factorized rather than a full model-by-horizon-by-feature
Cartesian product: the horizon/model screen uses `positioning`, then the best
eligible model/horizon lineages run the nested feature ablation. This limits
multiple testing and compute consumption.

## Recursive generations

### Generation 1: broad causal screen

Train the model/horizon roster with fixed baseline parameters and the
`positioning` family. Retain a fixed Ridge/price control at every horizon.
Then test the nested Kraken feature families only for the three strongest
eligible model/horizon lineages.

A lineage is `promising` when it has positive pooled OOF net expectancy,
positive net P&L in at least four of six folds, profit factor above 1.0, and
positive rank correlation. It is `predictive_only` when rank correlation is
positive with a positive confidence bound but net expectancy is non-positive.
Otherwise it is `failed`.

Generation 1 writes an immutable decision manifest containing the retained
lineages, rejected lineages, paired feature-ablation results, and the exact
search neighborhoods allowed in Generation 2.

### Generation 2: improve or pivot

For each `promising` lineage, run a bounded local search of at most 16
deterministic parameter configurations using three inner chronological folds.
Test threshold and no-trade calibration only after model selection. Add an
optional data family (qualified Kraken L2, long/short/top-trader positioning,
or one cross-venue family) only as a paired ablation against the lineage's
Kraken baseline.

For each `predictive_only` lineage, keep the model/horizon fixed and test
whether volatility-scaled targets, robust loss, and stricter cost-clearing
action thresholds convert signal into net expectancy.

If Generation 1 has no eligible lineage, Generation 2 pivots instead of tuning
noise: it tests 4-hour, 8-hour, and 12-hour volatility-normalized return targets
with Elastic Net, Histogram Gradient Boosting, and LightGBM, plus a calibrated
three-class long/no-trade/short cost-clearing classifier. It does not widen the
feature set at the same time.

Generation 2 advances at most two lineages. Advancement requires improvement
over the lineage parent in paired OOF folds and no material deterioration in
cost stress, trade concentration, or delayed-entry sensitivity.

### Generation 3: robustness and final selection

For improving lineages, freeze the selected feature family and run a narrow
robustness refinement: seed stability, reduced feature subset, calibration
choice, threshold stability, and 1.5x/2.0x execution-cost stress. A blend of
the two Generation 2 finalists is allowed only when it is fit exclusively from
their OOF predictions and improves paired OOF net expectancy after correction.

If Generation 2 remains predictive-only, Generation 3 pivots to a model-gated
strategy: the return regressor proposes direction and a calibrated
cost-clearing classifier decides trade/no-trade. If Generation 2 has no stable
predictive signal, Generation 3 runs the best longer-horizon pivot plus the
Ridge control as a final falsification test; it must not manufacture a winner
by widening the search.

At the end of Generation 3, one candidate or no candidate is selected. The
candidate, action policy, feature list, hyperparameters, seed set, snapshot
hashes, and code revision are frozen before confirmation data is loaded. Only
a candidate passing every development gate proceeds to the one-time
confirmation, then the still-sealed holdout, and finally the post-holdout
recency window without retuning.

## Qualification gates

The final candidate must satisfy all of the following on development OOF data:

- at least five of six positive folds;
- at least `3.0` net basis points per non-overlapping trade;
- profit factor at least `1.15`;
- positive lower bound of the 95% daily block-bootstrap expectancy interval;
- at least five of six positive folds at `1.5x` execution cost and positive
  pooled expectancy at `2.0x` cost;
- at least 60% positive calendar months;
- no more than 40% of positive P&L contributed by one fold;
- at least 300 effective non-overlapping trades;
- OOF Spearman rank correlation at least `0.02` with a positive confidence
  bound;
- positive delayed-entry sensitivity;
- no leakage, coverage, provenance, or non-finite-data failure; and
- statistical significance after Holm correction across all candidates tested
  in all three generations.

Confirmation and holdout use the same economic gates, with the preregistered
minimum trade count appropriate to their shorter durations. A failed gate is a
failed qualification; accuracy cannot override negative net expectancy.

## Reports and artifacts

Each generation produces one machine-readable JSON report and one concise
Markdown report. The report contains one row per trained model/horizon and the
following statistics:

| Category | Required statistics |
|---|---|
| Lineage | generation, parent, run ID, Git revision/tag, config SHA, raw/feature snapshot SHAs |
| Training | model, horizon, feature family/count, hyperparameters, fit rows, validation rows, folds, seeds |
| Predictive | MAE, RMSE, R-squared, Spearman IC and CI, directional/balanced accuracy, calibration slope/intercept |
| Economic | trades, coverage, net bps/trade and CI, total net P&L, profit factor, positive folds/months, max drawdown |
| Robustness | maker/zero-fee diagnostics, 1.5x and 2.0x costs, delayed entry, seed dispersion, fold concentration |
| Compute | wall time, CPU time, effective threads, peak RSS, artifact size |
| Decision | advanced/rejected/pivoted, failed gates, paired improvement versus parent/control |

Reports include failed and no-trade candidates, not only winners. Generation 2
and 3 reports explicitly compare each child with its parent so recursive
improvement is visible. A final tournament report presents all generations in
one scoreboard and records whether confirmation/holdout were opened.

Model binaries and large predictions remain on the SSD. Git contains code,
frozen configs, concise reports, decision manifests, and artifact checksums;
it does not contain raw market data or large serialized estimators.

## Git model tags

After a generation is fully trained and its reports and manifests are
committed, create one annotated tag for every retained model/horizon artifact:

```text
model/kraken-pf-xbtusd-gen-<NN>-<model>-h<horizon>-<run-id>
```

Examples:

```text
model/kraken-pf-xbtusd-gen-01-ridge-h1h-20260821T120000Z
model/kraken-pf-xbtusd-gen-02-lightgbm-h4h-20260822T090000Z
```

Tuning trials that are not retained remain identified in the generation
manifest and do not receive individual Git tags. Each annotated tag records
the parent lineage, artifact URI and SHA-256, raw/feature snapshot SHAs,
configuration SHA-256, training cutoff, qualification status, and report
commit. Tags identify research model artifacts only; they are not golden image
or trading-deployment authorization.

## Implementation boundary and completion

Implementation is limited to tournament configuration, feature/target
construction, classical model adapters, chronological tuning, recursive
decision manifests, evaluation, reporting, and focused tests inside
`packages/kraken-ml`. Existing Kraken ingestion output is read-only. Kraken L2
integration is a conditional feature adapter and is excluded until its own
data qualification passes.

The work is complete when all three generation reports and the consolidated
scoreboard exist, every retained model has its annotated generation tag, all
artifacts have verifiable lineage, and the final report declares either a
qualified frozen candidate or a no-edge result. No paper or live trading bot is
created by this tournament.
