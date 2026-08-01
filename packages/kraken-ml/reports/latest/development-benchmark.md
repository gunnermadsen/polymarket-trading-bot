# Kraken Futures ML Development Benchmark

## Edge verdict

**FAIL — development evidence does not meet every pre-holdout gate.**

## Reproducibility

| Item | Value |
| --- | --- |
| Run | 20260728T173754Z-3638569e-2b55a336 |
| Generated | 2026-07-28T17:42:12.835420+00:00 |
| Configuration SHA-256 | `2b55a3363b4ecc648b20c4e4230d34c8d7cf9467dcec72331fde6048b139b7a7` |
| Dataset SHA-256 | `1104b512584d2e3d783b479198c1d4f303d017de00409359fd75fce6ba1e0955` |
| Feature SHA-256 | `3638569e60067c8590b02965e43d5dda11242e60579d4f125c9c0e8c0ab66026` |

## Data scope

| Item | Value |
| --- | --- |
| Market | PF_XBTUSD |
| Bar interval | 900 |
| Forecast horizon | 4 |
| Start | 2023-06-01T12:15:00+00:00 |
| End | 2026-07-28T01:30:00+00:00 |
| Rows | 110646 |
| Canonical source | content_addressed_parquet_lake |

## Selected development candidate

| Item | Value |
| --- | --- |
| Model | extra_trees |
| Feature set | price |

## CPU and runtime

| Parameter | Value |
| --- | --- |
| Host | arm |
| Detected cores | 12 |
| Reserved cores | 2 |
| Parallel fits | 10 |
| Comparison threads / estimator | 1 |
| Polars threads / process | 1 |
| Final refit threads | 10 |
| Timing: total development seconds | 259.21 s |

## Model comparison

| Model | Features | Balanced acc. | Macro F1 | Log loss | Net bps/trade | Positive folds | Selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| extra_trees | full | 41.77% | 41.08% | 1.0578 | — | 0 | yes |
| histogram | full | 39.84% | 37.34% | 1.0688 | — | 0 | no |
| logistic | full | 40.79% | 39.23% | 1.068 | — | 0 | no |

## Feature-set comparison

| Model | Features | Balanced acc. | Macro F1 | Log loss | Net bps/trade | Positive folds | Selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| extra_trees | flow | 41.90% | 41.37% | 1.0574 | — | 0 | no |
| extra_trees | full | 41.77% | 41.08% | 1.0578 | — | 0 | no |
| extra_trees | price | 42.22% | 41.84% | 1.0557 | — | 0 | yes |

## Threshold-window policy diagnostics

| Fold | Trades | Best net bps/trade | 80% CI lower | Profit factor | Probability | Margin |
| --- | --- | --- | --- | --- | --- | --- |
| 2024_summer | 86 | 1.2507 | -8.1997 | 1.055 | 0.5 | 0.05 |
| 2024_autumn | 334 | -10.8335 | -14.665 | 0.6709 | 0.4 | 0.05 |
| 2024_winter | 77 | -0.4153 | -7.1107 | 0.9846 | 0.45 | 0.15 |
| 2025_spring | 142 | -7.9182 | -14.7272 | 0.7501 | 0.45 | 0.1 |
| 2025_summer | 58 | -3.1673 | -11.8901 | 0.8285 | 0.45 | 0.15 |
| 2025_autumn | 51 | -0.4067 | -7.1582 | 0.9762 | 0.45 | 0.1 |

A policy qualifies only when its 80% circular-block bootstrap lower bound is positive with at least 50 trades.

## Walk-forward folds

| Fold | Model | Features | Balanced acc. | Uplift | Macro F1 | Log loss | Trades | Net bps/trade | 95% CI lower | Profit factor | Policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024_summer | extra_trees | price | 42.50% | 3.18% | 42.50% | 1.0509 | 0 | — | — | — | no trade |
| 2024_autumn | extra_trees | price | 41.29% | 4.03% | 40.71% | 1.0653 | 0 | — | — | — | no trade |
| 2024_winter | extra_trees | price | 42.58% | 3.72% | 42.17% | 1.0549 | 0 | — | — | — | no trade |
| 2025_spring | extra_trees | price | 41.18% | 4.06% | 41.03% | 1.0624 | 0 | — | — | — | no trade |
| 2025_summer | extra_trees | price | 42.12% | 4.88% | 41.62% | 1.0551 | 0 | — | — | — | no trade |
| 2025_autumn | extra_trees | price | 43.64% | 5.06% | 42.99% | 1.0455 | 0 | — | — | — | no trade |

## Decision gates

| Gate | Result | Observed | Requirement | Detail |
| --- | --- | --- | --- | --- |
| balanced accuracy fold stability | PASS | 6 | 5 | — |
| balanced accuracy uplift | PASS | 0.0416 | 0.03 | — |
| macro f1 fold stability | PASS | 6 | 5 | — |
| paired accuracy confidence | PASS | 0.0388 | > 0 | — |
| positive net expectancy stability | FAIL | 0 | 5 | — |
| proper scoring rule stability | PASS | {'log_loss_wins': 6, 'brier_wins': 6} | 5 | — |

## Locked holdout

| Item | Value |
| --- | --- |
| Status | sealed_not_qualified |
| Holdout opened | False |
| Start | 2026-02-12T13:00:00+00:00 |
| End | 2026-07-28T03:00:00+00:00 |
| Reason | candidate failed the positive net expectancy stability gate |

## Notes

- **Note:** Funding is excluded from every feature set and is applied only to realized target and ledger P&L.
- **Note:** Normalized PostgreSQL analytics rows are excluded because the source audit found an unsafe child index and a slippage gap.
- **Note:** Gross one-hour targets use linear-contract arithmetic returns relative to entry notional.
- **Note:** Execution costs use the last order-book analytics observation in the completed 15-minute bucket at each simulated fill; exact boundary fills would require finer 1-minute or L2 data.
- **Note:** Contract metadata verifies flexible_futures BTC/USD, tick size 1, and the configured 5.0 bps MTF Linear Rebate Fees base taker fee.
- **Note:** Funding archive coverage begins at 2026-02-12 13:00:00+00:00; earlier missing rates are treated as zero and this limitation is not used as a model feature.
- **Note:** Full-range deterministic features are materialized once, but holdout rows are excluded from model fitting, calibration, policy selection, and development evaluation.
- **Note:** Every selected-fold calibration policy resolved to no-trade because no candidate with at least 50 trades had a positive 80% block-bootstrap lower bound.
