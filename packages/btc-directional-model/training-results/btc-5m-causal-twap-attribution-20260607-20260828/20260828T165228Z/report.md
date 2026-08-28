# BTC 5m Causal TWAP Attribution Tournament

- Run: `20260828T165228Z`
- Model family: `btc-5m-counterfactual-twap-state`
- Provisional candidate: `refprice_only`
- Qualification: **prospective_evidence_pending**
- Deployment: **not deployed; training-only and paper-only**

## Source fidelity

- Chainlink reconstruction passed: `True`
- Binance extension passed: `True`

## Causal integrity

- Feature registry: `a81cdee79de28d28979d4ebbad7136dae79a0aad9ff2f445086ce53a61b37644`
- Point-in-time availability audit passed: `True`
- Serialization and batch/single-row parity passed: `True`
- Supervision perturbation parity error: `0.0`

## Predictive attribution

| Layer | Brier | Log loss | ECE |
|---|---:|---:|---:|
| `refprice_only` | 0.17156 | 0.51671 | 0.04158 |
| `refprice_non_twap_basis` | 0.17062 | 0.51422 | 0.04284 |
| `refprice_relative_twap` | 0.17050 | 0.51389 | 0.04212 |
| `refprice_absolute_twap` | 0.17061 | 0.51369 | 0.04039 |

- Best TWAP candidate: `refprice_relative_twap`
- Best TWAP minus basis paired Brier: `-0.000126` (95% CI `[-0.000560, 0.000307]`)
- Date/direction/entry-band/margin-band stability passed: `True`
- TWAP hypothesis supported: `False`

## Candidate development results

| Candidate | Brier | ECE | Economic policy | Trades | Coverage | Accuracy | Stressed PnL | Profit factor | Max drawdown | CVaR 5% |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `refprice_only` | 0.17156 | 0.04158 | fixed | 378 | 10.96% | 87.83% | $-13.93 | 0.93 | $49.36 | $-4.57 |
| `refprice_non_twap_basis` | 0.17062 | 0.04284 | not assigned | — | — | — | — | — | — | — |
| `refprice_relative_twap` | 0.17050 | 0.04212 | not assigned | — | — | — | — | — | — | — |
| `refprice_absolute_twap` | 0.17061 | 0.04039 | not assigned | — | — | — | — | — | — | — |

## Selected-treatment economic folds

| Fold | Trades | Coverage | Stressed PnL | Profit factor | Bootstrap lower | UP | DOWN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `official_20260814_15` | 2 | 0.35% | $-3.79 | 0.11 | $-4.274 | 0 | 2 |
| `official_20260816_17` | 11 | 1.92% | $4.90 | inf | $0.406 | 6 | 5 |
| `official_20260818_19` | 33 | 5.74% | $8.17 | 1.98 | $-0.248 | 19 | 14 |
| `official_20260820_21` | 149 | 25.96% | $-20.67 | 0.76 | $-0.400 | 69 | 80 |
| `official_20260822_23` | 78 | 13.54% | $1.18 | 1.03 | $-0.381 | 51 | 27 |
| `official_20260824_25` | 105 | 18.23% | $-3.73 | 0.94 | $-0.364 | 49 | 56 |

## Prospective qualification

- Window: `2026-08-26T00:00:00+00:00` to `2026-08-28T00:00:00+00:00`
- Authentic markets: `0`
- Trades: `0`
- Stressed PnL: `$0.00`
- Maximum drawdown: `$0.00`
- Post-freeze tuning: `false`

## Required conclusion

- The synthetic TWAP hypothesis is not supported by this frozen tournament: the best TWAP treatment did not improve paired Brier loss beyond the non-TWAP basis baseline with an upper 95% confidence bound below zero.
- Relational TWAP state had a lower point-estimate Brier score than the non-TWAP basis treatment.
- Decisive best-TWAP minus basis Brier interval: [-0.000560, 0.000307].
- Deployment qualification is pending untouched post-freeze evidence.

## Failed qualification gates

- `twap_hypothesis_supported`
- `ece`
- `positive_stressed_pnl`
- `positive_stressed_expectancy`
- `profit_factor`
- `positive_bootstrap_lower`
- `profitable_temporal_folds`
- `market_coverage`
- `both_directions`
- `no_single_day_majority`
- `prospective_markets`
- `prospective_days`
- `prospective_folds`
- `base_capacity_positive`

## Scope controls

No database mutation, migration, table, ingester, data source, runtime export, deployment, or trading-process change was performed.
