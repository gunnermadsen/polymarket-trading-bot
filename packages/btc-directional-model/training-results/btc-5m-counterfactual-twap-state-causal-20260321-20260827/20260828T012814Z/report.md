# BTC 5m Counterfactual TWAP-State Tournament

- Run: `20260828T012814Z`
- Model family: `btc-5m-counterfactual-twap-state`
- Provisional candidate: `dual_twap_state`
- Qualification: **no_deployable_challenger_qualified**
- Deployment: **not deployed; training-only and paper-only**

## Source fidelity

- Chainlink reconstruction passed: `True`
- Binance extension passed: `True`

## Causal integrity

- Feature registry: `1c3aa7694dbbd33fe896424c9e805a744a1021b36530f95b785cbf40fbf4df39`
- Point-in-time availability audit passed: `True`
- Non-Binance labels altered by synthetic uncertainty weighting: `0`
- Serialization and batch/single-row parity passed: `True`
- Supervision perturbation parity error: `0.0`

## Non-qualifying TWAP attribution

| Layer | Brier | Log loss | ECE |
|---|---:|---:|---:|
| `refprice_only` | 0.17223 | 0.51863 | 0.04229 |
| `refprice_non_twap_basis` | 0.17089 | 0.51508 | 0.03955 |
| `refprice_relative_twap` | 0.17082 | 0.51491 | 0.03964 |
| `refprice_absolute_twap` | 0.17109 | 0.51536 | 0.03799 |

## Candidate development results

| Candidate | Brier | Trades | Coverage | Accuracy | Stressed PnL | Profit factor | Max drawdown | CVaR 5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `refprice_state_control` | 0.17230 | 526 | 16.64% | 88.59% | $-63.58 | 0.76 | $85.99 | $-4.67 |
| `dual_twap_state` | 0.17258 | 338 | 10.69% | 89.35% | $-5.34 | 0.97 | $35.05 | $-4.59 |
| `refprice_dual_twap_state` | 0.17263 | 343 | 10.85% | 88.63% | $-26.50 | 0.85 | $46.02 | $-4.63 |
| `refprice_dual_twap_binance_extension` | 0.17588 | 392 | 12.40% | 88.78% | $-42.17 | 0.79 | $55.06 | $-4.66 |
| `refprice_dual_twap_margin_calibrated` | 0.17390 | 441 | 13.95% | 90.02% | $-26.07 | 0.87 | $47.66 | $-4.66 |
| `refprice_dual_twap_uncertainty_guard` | 0.17582 | 367 | 11.61% | 88.83% | $-47.46 | 0.74 | $61.76 | $-4.70 |

## Prospective qualification

- Window: `2026-08-26T00:00:00+00:00` to `2026-08-27T00:00:00+00:00`
- Authentic markets: `239`
- Trades: `0`
- Stressed PnL: `$0.00`
- Maximum drawdown: `$0.00`
- Post-freeze tuning: `false`

## Required conclusion

- Causal relational TWAP state improved Brier score beyond the non-TWAP Binance/Chainlink basis diagnostic.
- Chainlink-reconstructed history helped versus authentic-only history.
- Corrected Binance-only extension did not help versus Chainlink plus authentic history.
- The removed completed-market leakage materially inflated the disagreement treatment.
- No new deployable challenger qualified.

## Failed qualification gates

- `paired_predictive_improvement`
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
