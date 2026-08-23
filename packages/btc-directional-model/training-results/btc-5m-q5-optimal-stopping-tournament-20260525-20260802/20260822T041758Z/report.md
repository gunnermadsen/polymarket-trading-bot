# BTC Q5 and Optimal-Stopping Tournament

Status: **strictly offline development comparison; no runtime export**

Decision: `no_qualified_champion`
Champion: `None`
Diagnostic leader: `q5_incumbent`

| Candidate | Qualified | Trades | Coverage | Accuracy | VWAP5 net | Stress net | PF | Payoff | Avg entry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q5_incumbent | no | 165 | 2.40% | 81.21% | 65.30 | 57.05 | 1.595 | 0.369 | 95.279 |
| q5_loss_veto | no | 159 | 2.31% | 81.13% | 62.47 | 54.52 | 1.592 | 0.370 | 96.478 |
| q5_coverage_expander | no | 120 | 1.74% | 25.00% | -15.29 | -21.29 | 0.873 | 2.620 | 150.917 |
| q5_veto_expander_composite | no | 279 | 4.05% | 56.99% | 47.18 | 33.23 | 1.208 | 0.912 | 119.892 |
| two_sided_direct_value | no | 560 | 8.13% | 38.57% | 25.40 | -2.60 | 1.042 | 1.660 | 116.438 |
| expected_utility_optimal_stopping | no | 1 | 0.01% | 0.00% | -1.08 | -1.13 | 0.000 | 0.000 | 240.000 |
| distributional_optimal_stopping | no | 457 | 6.63% | 91.68% | -4.13 | -26.98 | 0.976 | 0.088 | 232.133 |

## Limitations

- All evidence through August 2 was consumed by prior development work and is not a fresh holdout.
- Future trajectory values are training targets only; evaluation and student inference use causal current-state features.
- Projected PnL assumes recorded ask VWAP was fillable and does not model queue position.
- The tournament is offline only and exports no runtime model or trading-process configuration.
