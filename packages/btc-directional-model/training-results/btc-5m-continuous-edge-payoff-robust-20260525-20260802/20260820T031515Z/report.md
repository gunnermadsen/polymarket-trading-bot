# BTC Five-Minute Payoff-Robust Continuous-Edge Training

Status: **development-only evidence; not runtime exported**

Selected probability candidate: `core_oracle_vwap_curve`

The candidate did not pass every frozen fresh-holdout gate; review the failed checks before considering deployment.

## Fresh chronological holdout

| Policy | Trades | Accuracy | Net PnL (VWAP5) | Expectancy | PF | Avg entry | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Without payoff admission | 0 | 0.00% | 0.00 | 0.0000 | 0.000 | — | — |
| Selected payoff-aware edge | 0 | 0.00% | 0.00 | 0.0000 | 0.000 | — | 0.00% |

Stress PnL: **0.00**; stress expectancy: **0.0000**; strict-data coverage: **82.51%**; end-to-end trade coverage: **0.00%**.

Average win: **0.000**; average loss: **0.000**; payoff ratio: **0.000**; active days: **0**; daily PnL concentration: **0.00%**.

Chronological payoff lower-bound calibration coverage: **80.23%** versus **80.00%** target.

All three time bands were disabled before holdout scoring. The loosest frozen lower-bound policy admitted only 8 early, 8 mid, and 5 late validation trades; none reached the 40-trade and 75%-profitable-fold requirements. The calibrated global payoff residual penalty was 0.6019 per share.

## Frozen incumbent control on the same holdout

The previously trained payoff-aware artifact was frozen before this holdout existed. Replaying its unchanged early/late policy produced 35 VWAP5 trades at 68.57% accuracy, -8.25 net PnL, -10.00 stressed PnL, 0.788 profit factor, and 89.86-second average entry. Early contributed 24 trades at 70.83% accuracy and -2.57 PnL; late contributed 11 trades at 63.64% accuracy and -5.68 PnL. All four three-day folds were non-profitable. The robust candidate's abstention therefore avoided carrying this measured negative edge forward, but it did not produce a deployable trading policy.

## Fixed-entry VWAP capacity curve

The side and timestamp are frozen from the VWAP5 policy; only exact execution size changes.

| Shares | Trades | Accuracy | Mean VWAP | Net PnL | Expectancy | PF | Max drawdown |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |
| 10 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |
| 15 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |
| 20 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |
| 25 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |
| 30 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |
| 40 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |
| 50 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |
| 75 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |
| 100 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |
| 125 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |
| 150 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |
| 175 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |
| 200 | 0 | 0.00% | 0.0000 | 0.00 | 0.0000 | 0.000 | 0.00 |

## Test results by entry band (VWAP5)

| Band | Trades | Accuracy | Net PnL | Expectancy | Average entry |
|---|---:|---:|---:|---:|---:|
| early | 0 | 0.00% | 0.00 | 0.0000 | — |
| mid | 0 | 0.00% | 0.00 | 0.0000 | — |
| late | 0 | 0.00% | 0.00 | 0.0000 | — |

## Rolling three-day comparison

| Fold | Trades | Accuracy | Stress PnL | Stress expectancy |
|---|---:|---:|---:|---:|
| 2026-07-23 to 2026-07-26 | 0 | 0.00% | 0.00 | 0.0000 |
| 2026-07-26 to 2026-07-29 | 0 | 0.00% | 0.00 | 0.0000 |
| 2026-07-29 to 2026-08-01 | 0 | 0.00% | 0.00 | 0.0000 |
| 2026-08-01 to 2026-08-02 | 0 | 0.00% | 0.00 | 0.0000 |

## Comparison results by entry-price bucket (VWAP5)

| Entry price | Trades | Accuracy | Net PnL | Stress PnL |
|---|---:|---:|---:|---:|
| 0.00-0.65 | 0 | 0.00% | 0.00 | 0.00 |
| 0.65-0.75 | 0 | 0.00% | 0.00 | 0.00 |
| 0.75-0.85 | 0 | 0.00% | 0.00 | 0.00 |
| 0.85-1.01 | 0 | 0.00% | 0.00 | 0.00 |

## Limitations

- Capacity evidence ends at second 240; seconds 241-299 are not evaluated.
- TWAP was excluded because no persisted, settlement-aligned historical TWAP source was available.
- Open interest begins after the outcome-fit window and is diagnostic only.
- Aggregate trade prints overlap only the end of the test block and are diagnostic only.
- Directional and asymmetric family signals are chronological proxy refits using the frozen family feature contracts; they are not replays of future-trained runtime artifacts.
- Projected PnL assumes the recorded VWAP is fillable at the sampled timestamp and does not model queue position.
