# BTC Five-Minute Middle-Market Payoff Tournament

Provisional champion: `none`

Qualification decision: **no_qualified_model**

All candidates were frozen before the post-August-2 holdout was scored. Holdout results cannot replace the provisional champion.

## New holdout comparison (VWAP5)

| Candidate | Trades | Coverage | Accuracy | Net PnL | Stress PnL | PF | Avg entry | Median entry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mid_core_control | 52 | 44.83% | 75.00% | -3.57 | -6.17 | 0.918 | 120.000 | 117.500 |
| mid_payoff_coverage | 53 | 45.69% | 81.13% | 2.41 | -0.24 | 1.066 | 117.453 | 110.000 |
| mid_failure_risk | 24 | 20.69% | 75.00% | -2.62 | -3.82 | 0.872 | 123.125 | 120.000 |
| mid_direction_qualified | 0 | 0.00% | 0.00% | 0.00 | 0.00 | 0.000 | — | — |
| mid_regime_calibrated | 64 | 55.17% | 78.12% | 1.09 | -2.11 | 1.021 | 118.906 | 117.500 |
| mid_wait_value | 1 | 0.86% | 100.00% | 0.40 | 0.35 | — | 175.000 | 175.000 |
| mid_chainlink | 22 | 18.97% | 81.82% | 2.55 | 1.45 | 1.195 | 125.682 | 122.500 |
| mid_binance_l2 | 0 | 0.00% | 0.00% | 0.00 | 0.00 | 0.000 | — | — |
| mid_open_interest | 0 | 0.00% | 0.00% | 0.00 | 0.00 | 0.000 | — | — |
| mid_trade_prints | 0 | 0.00% | 0.00% | 0.00 | 0.00 | 0.000 | — | — |

## Fixed-entry VWAP PnL

| Candidate | VWAP5 | VWAP10 | VWAP20 | VWAP50 | VWAP100 | VWAP200 |
|---|---:|---:|---:|---:|---:|---:|
| mid_core_control | -3.57 | -7.27 | -14.87 | -40.12 | -90.76 | -210.17 |
| mid_payoff_coverage | 2.41 | 4.73 | 9.20 | 21.10 | 35.53 | 47.68 |
| mid_failure_risk | -2.62 | -5.26 | -10.59 | -27.14 | -57.36 | -126.94 |
| mid_direction_qualified | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| mid_regime_calibrated | 1.09 | 2.11 | 3.86 | 6.36 | 2.34 | -33.77 |
| mid_wait_value | 0.40 | 0.79 | 1.59 | 3.96 | 7.93 | 15.44 |
| mid_chainlink | 2.55 | 5.09 | 10.09 | 23.76 | 43.66 | 77.21 |
| mid_binance_l2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| mid_open_interest | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| mid_trade_prints | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Limitations

- All candidates are offline development artifacts and are not runtime exported.
- The frozen incumbent has no enabled middle-market policy and therefore makes no common-window trades.
- Open-interest history starts July 3 and trade-print history starts July 22, so those candidates use shorter causal fitting windows.
- Projected PnL assumes recorded ask VWAP is fillable and does not model queue position.
