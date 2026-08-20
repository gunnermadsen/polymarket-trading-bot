# BTC Five-Minute Continuous-Edge Directional Training

Status: **development-only evidence; not runtime exported**

Selected probability candidate: `core_oracle_vwap_curve`

The expectancy gates passed, but no fresh post-July-23 capacity holdout exists; this result must not be deployed.

## Retrospective chronological comparison block

| Policy | Trades | Accuracy | Net PnL (VWAP5) | Expectancy | PF | Avg entry | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Without payoff admission | 91 | 70.33% | -15.67 | -0.1722 | 0.843 | 61.637 | — |
| Selected payoff-aware edge | 38 | 84.21% | 20.50 | 0.5395 | 1.933 | 65.658 | 1.68% |

Stress PnL: **18.60**; stress expectancy: **0.4895**; strict-data coverage: **73.28%**; end-to-end trade coverage: **1.23%**.

## Fixed-entry VWAP capacity curve

The side and timestamp are frozen from the VWAP5 policy; only exact execution size changes.

| Shares | Trades | Accuracy | Mean VWAP | Net PnL | Expectancy | PF | Max drawdown |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 38 | 84.21% | 0.7153 | 20.50 | 0.5395 | 1.933 | 10.89 |
| 10 | 38 | 84.21% | 0.7153 | 41.00 | 1.0791 | 1.933 | 21.78 |
| 15 | 38 | 84.21% | 0.7153 | 61.51 | 1.6186 | 1.933 | 32.68 |
| 20 | 38 | 84.21% | 0.7153 | 81.96 | 2.1568 | 1.932 | 43.57 |
| 25 | 38 | 84.21% | 0.7154 | 102.38 | 2.6942 | 1.931 | 54.46 |
| 30 | 38 | 84.21% | 0.7155 | 122.79 | 3.2312 | 1.931 | 65.35 |
| 40 | 38 | 84.21% | 0.7157 | 163.35 | 4.2987 | 1.928 | 87.14 |
| 50 | 38 | 84.21% | 0.7161 | 203.52 | 5.3557 | 1.925 | 108.92 |
| 75 | 38 | 84.21% | 0.7169 | 303.13 | 7.9770 | 1.916 | 163.48 |
| 100 | 38 | 84.21% | 0.7175 | 401.72 | 10.5715 | 1.909 | 218.19 |
| 125 | 38 | 84.21% | 0.7181 | 499.33 | 13.1402 | 1.903 | 273.04 |
| 150 | 38 | 84.21% | 0.7189 | 595.12 | 15.6610 | 1.895 | 327.98 |
| 175 | 38 | 84.21% | 0.7196 | 689.61 | 18.1476 | 1.888 | 382.93 |
| 200 | 38 | 84.21% | 0.7204 | 782.49 | 20.5917 | 1.881 | 437.87 |

## Test results by entry band (VWAP5)

| Band | Trades | Accuracy | Net PnL | Expectancy | Average entry |
|---|---:|---:|---:|---:|---:|
| early | 33 | 81.82% | 14.27 | 0.4324 | 43.788 |
| mid | 0 | 0.00% | 0.00 | 0.0000 | — |
| late | 5 | 100.00% | 6.23 | 1.2468 | 210.000 |

## Rolling three-day comparison

| Fold | Trades | Accuracy | Stress PnL | Stress expectancy |
|---|---:|---:|---:|---:|
| 2026-07-13 to 2026-07-16 | 0 | 0.00% | 0.00 | 0.0000 |
| 2026-07-16 to 2026-07-19 | 3 | 100.00% | 3.87 | 1.2888 |
| 2026-07-19 to 2026-07-22 | 30 | 80.00% | 8.80 | 0.2932 |
| 2026-07-22 to 2026-07-23 | 5 | 100.00% | 5.94 | 1.1878 |

## Comparison results by entry-price bucket (VWAP5)

| Entry price | Trades | Accuracy | Net PnL | Stress PnL |
|---|---:|---:|---:|---:|
| 0.00-0.65 | 1 | 100.00% | 2.84 | 2.79 |
| 0.65-0.75 | 32 | 81.25% | 13.59 | 11.99 |
| 0.75-0.85 | 3 | 100.00% | 2.81 | 2.66 |
| 0.85-1.01 | 2 | 100.00% | 1.27 | 1.17 |

## Limitations

- Capacity evidence ends at second 240; seconds 241-299 are not evaluated.
- No completed PMXT capacity artifacts exist after July 23; the comparison block was observed in the preceding run and is not a fresh untouched holdout.
- Open interest begins after the outcome-fit window and is diagnostic only.
- Aggregate trade prints overlap only the end of the test block and are diagnostic only.
- Directional and asymmetric family signals are chronological proxy refits using the frozen family feature contracts; they are not replays of future-trained runtime artifacts.
- Projected PnL assumes the recorded VWAP is fillable at the sampled timestamp and does not model queue position.
