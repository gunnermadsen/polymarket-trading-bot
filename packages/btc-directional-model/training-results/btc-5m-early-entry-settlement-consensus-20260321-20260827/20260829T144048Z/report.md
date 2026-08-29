# Early-Entry Settlement Consensus Tournament

Run: `20260829T144048Z`  
Producing commit: `f86ab895930823840b079f92f17c542ccd4969c9`  
Candidate freeze: `2026-08-27T00:00:00+00:00`  
Deployment: unchanged; training-only artifact.

## Outcome

Top historical candidate: `frozen_bridge_control`.  
Synthetic TWAP incremental value: **not_demonstrated**.  
Constituent consensus incremental value: **not_demonstrated**.
This is a corrected historical rerun. The frozen August 27–28 holdout was already consumed by the prior run and is not new prospective evidence.

## Constituent reproduction gate

| Family | Brier | Accuracy | Probability std | Reference Brier |
|---|---:|---:|---:|---:|
| bridge | 0.181567 | 0.729585 | 0.280297 | 0.184599 |
| latent | 0.189003 | 0.733955 | 0.283272 | — |
| causal | 0.169975 | 0.749808 | 0.300149 | 0.172724 |

Reproduction gate: **passed**. Causal calibration coefficient range: `0.931116`–`1.045067`.

## Paired incremental evidence

- Synthetic history candidate-minus-control Brier: `-0.000015`; 95% market-bootstrap CI `-0.000099` to `0.000063`; 4/6 improving folds; **not_demonstrated**.
- Best consensus candidate-minus-bridge Brier: `-0.009945`; 95% market-bootstrap CI `-0.013423` to `-0.006231`; 5/6 improving folds; **positive**.

## Candidate summary

| Candidate | Policy | Brier | Log loss | ECE | Accuracy | Margin MAE bps | Coverage | Trades | Active days | Cost mean/max | W/L | Stressed PnL | Stressed expectancy | Profit factor | Recovery wins | Losing streak | Max DD | CVaR 5% | Classification |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| frozen_bridge_control | edge_loss_compensation_tail_risk | 0.181344 | 0.540903 | 0.028447 | 0.729585 | 6.012 | 0.0005 | 2 | 2 | 0.5650/0.6100 | 2/0 | 4.0275 | 2.01373 | — | — | 0 | 0.0000 | 1.7907 | promising_sparse_edge |
| bridge_latent_equal_logit_pool | probability_edge_control | 0.180302 | 0.558962 | 0.048008 | 0.741534 | 5.659 | 0.5503 | 2030 | 12 | 0.5697/0.9300 | 1181/849 | -134.6072 | -0.06631 | 0.941 | 1.478 | 9 | 204.2148 | -3.9579 | interesting_pattern |
| three_family_nonnegative_logit_stack | edge_uncertainty_margin | 0.193161 | 0.571419 | 0.124496 | 0.719490 | 5.679 | 0.0114 | 42 | 4 | 0.6855/0.7500 | 31/11 | 5.7930 | 0.13793 | 1.150 | 2.450 | 3 | 12.8230 | -3.8180 | promising_sparse_edge |
| bridge_latent_uncertainty_margin_stack | edge_loss_compensation_tail_risk | 0.171398 | 0.517622 | 0.011653 | 0.748344 | 5.551 | 0.0060 | 22 | 3 | 0.6023/0.6200 | 15/7 | 5.2395 | 0.23816 | 1.234 | 1.736 | 3 | 11.8194 | -3.2583 | promising_sparse_edge |

## History ablation

| History | Brier | Log loss | ECE | Accuracy | Markets |
|---|---:|---:|---:|---:|---:|
| authentic_only | 0.171759 | 0.519354 | 0.016738 | 0.749049 | 3689 |
| chainlink_reconstructed | 0.169932 | 0.512466 | 0.018630 | 0.749742 | 3689 |
| binance_synthetic_extension | 0.169891 | 0.512337 | 0.017887 | 0.750144 | 3689 |
| uncertainty_weighted_hybrid | 0.169917 | 0.512412 | 0.018507 | 0.750165 | 3689 |

## Frozen holdout replication (historically consumed)

The August 27–28 batch was still loaded only after models and policy choices were frozen, with no subsequent tuning or retraining. Because the prior run already evaluated it, these results are a holdout replication and must not be represented as fresh prospective evidence.

| Candidate | Policy | Brier | Trades | Stressed PnL | Expectancy | Classification |
|---|---|---:|---:|---:|---:|---|
| frozen_bridge_control | edge_loss_compensation_tail_risk | — | 0 | 0.0000 | 0.00000 | historically_consumed_holdout_unavailable |
| bridge_latent_equal_logit_pool | probability_edge_control | — | 0 | 0.0000 | 0.00000 | historically_consumed_holdout_unavailable |
| three_family_nonnegative_logit_stack | edge_uncertainty_margin | — | 0 | 0.0000 | 0.00000 | historically_consumed_holdout_unavailable |
| bridge_latent_uncertainty_margin_stack | edge_loss_compensation_tail_risk | — | 0 | 0.0000 | 0.00000 | historically_consumed_holdout_unavailable |

## Integrity and limitations

- Exact schedule: 25 points at 60–180 seconds; `0` incomplete eligible-source markets excluded explicitly.
- All constituent predictions are chronological OOF; preprocessing, missing-column handling, calibration, reliability weighting, and stack fitting occur inside the applicable fold.
- Every regularized estimator fit passed `mean_one_market_band_equal_v1`: finite nonnegative weights, total equal to rows, and mean equal to one. Full fold-level totals, ranges, effective sample sizes, label-source shares, and entry-band shares are in `metrics.json`.
- Immutable-reference reproduction and collapse gates ran before any ensemble fitting.
- Coverage and frequency are descriptive only; no coverage gate was applied.
- Profit factor is reported as undefined when no observed loss exists, alongside an exact accuracy interval and an injected stressed-loss result.
- Poor predictive or economic performance is a finding, not an integrity failure.
- No database writes, migrations, tables, ingesters, runtime exports, deployments, or trading-process changes were made.
