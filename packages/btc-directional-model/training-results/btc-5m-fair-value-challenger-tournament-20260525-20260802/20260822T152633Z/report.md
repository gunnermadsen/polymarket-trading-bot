# BTC Five-Minute Fair-Value Challenger Tournament

Status: **offline development comparison; no runtime export**

Decision: `fresh_holdout_not_ready`
Champion: `None`
Diagnostic leader: `specialist_distilled_fair_value`

| Candidate | Qualified | Trades | Coverage | Accuracy | VWAP5 net | Stress net | PF | Payoff | Avg entry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q5_incumbent | no | 165 | 2.40% | 81.21% | 65.30 | 57.05 | 1.595 | 0.369 | 95.279 |
| chainlink_stratified_payoff | no | 1411 | 20.48% | 73.71% | 13.40 | -57.15 | 1.010 | 0.360 | 125.096 |
| chainlink_full_combined | no | 1372 | 19.92% | 69.75% | -46.54 | -115.14 | 0.966 | 0.419 | 122.617 |
| chainlink_regime_calibrated | no | 1367 | 19.85% | 73.23% | 38.75 | -29.60 | 1.030 | 0.377 | 119.378 |
| exogenous_fair_value | no | 3798 | 55.14% | 66.85% | 162.82 | -27.08 | 1.042 | 0.517 | 80.888 |
| stratified_fair_value | no | 3739 | 54.28% | 71.52% | 253.26 | 66.31 | 1.072 | 0.427 | 110.359 |
| tail_weighted_fair_value | no | 2462 | 35.74% | 60.60% | -59.31 | -182.41 | 0.977 | 0.635 | 74.300 |
| terminal_margin_fair_value | no | 4336 | 62.95% | 75.51% | 77.16 | -139.64 | 1.021 | 0.331 | 175.942 |
| specialist_distilled_fair_value | no | 3340 | 48.49% | 69.94% | 382.75 | 215.75 | 1.119 | 0.481 | 89.226 |
| groupwise_entry_ranker | no | 3195 | 46.39% | 69.33% | 154.40 | -5.35 | 1.049 | 0.464 | 109.622 |

## Fresh holdout readiness

Passed: **False**; coverage: 26.04%.

## Limitations

- Development evidence through August 2 is consumed and is not a fresh holdout.
- The Q5 control is the exact frozen deployed artifact; Chainlink controls are fold-specific algorithm replicas.
- The full-combined replica is eligible only where causal OI evidence exists.
- Terminal-margin targets use the final available Binance path margin and calibrate to canonical labels; no TWAP is used.
- Projected PnL assumes recorded ask VWAP was fillable and does not model queue position.
- The tournament exports no runtime model or trading-process configuration.
