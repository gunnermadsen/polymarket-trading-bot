# Middle-Strategy Tournament

Run: `20260831T224530Z`
Qualification: **trained_evaluated_not_promoted_negative_expectancy**

## Sealed high-level results

| Candidate | PnL | Stress PnL | Coverage | Wins | Losses | W/L | Recovery wins/loss | Brier | PF | Avg cost | Avg entry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| middle_specialist_refit | -142.96 | -181.11 | 45.44% | 435 | 328 | 1.326 | 1.551 | 0.1540 | 0.855 | 0.586 | 88.1 |
| middle_q5_admission | -72.36 | -96.06 | 28.23% | 264 | 210 | 1.257 | 1.426 | 0.1536 | 0.882 | 0.566 | 96.8 |
| middle_agreement_ensemble | -116.05 | -143.30 | 32.46% | 303 | 242 | 1.252 | 1.494 | 0.1537 | 0.838 | 0.577 | 94.7 |
| crossvenue_middle_specialist | -57.96 | -89.11 | 37.11% | 369 | 254 | 1.453 | 1.571 | 0.1552 | 0.924 | 0.589 | 94.3 |
| price_time_calibrated_middle_ensemble | -89.51 | -147.61 | 69.21% | 739 | 423 | 1.747 | 1.876 | 0.1674 | 0.931 | 0.631 | 86.6 |

## Paired change from the prior normalized tournament

| Candidate | PnL delta | Stress PnL delta | Brier delta | PF delta | Coverage delta |
|---|---:|---:|---:|---:|---:|
| middle_specialist_refit | -54.39 | -53.14 | -0.02149 | -0.055 | -1.49% |
| middle_q5_admission | 8.44 | 16.39 | -0.02098 | -0.017 | -9.47% |
| middle_agreement_ensemble | -31.06 | -23.56 | -0.02111 | -0.063 | -8.93% |
| crossvenue_middle_specialist | 13.39 | 11.24 | -0.02067 | 0.024 | 2.56% |
| price_time_calibrated_middle_ensemble | -21.91 | -47.16 | -0.00750 | 0.016 | 30.08% |

## Prescribed sealed counterfactuals

| Candidate | 150-180 PnL / PF / trades | p80 early PnL / PF / trades |
|---|---:|---:|
| middle_specialist_refit | 2.43 / 1.007 / 303 | -138.99 / 0.829 / 623 |
| middle_q5_admission | -11.04 / 0.953 / 191 | 1.48 / 1.003 / 385 |
| middle_agreement_ensemble | 1.80 / 1.007 / 209 | -52.46 / 0.905 / 440 |
| crossvenue_middle_specialist | 8.98 / 1.032 / 240 | -48.93 / 0.923 / 515 |
| price_time_calibrated_middle_ensemble | -130.30 / 0.852 / 910 | -144.70 / 0.884 / 1128 |

## Settlement bridge

- RefPrice residual: 1983 paired markets, location -0.0000 bps, scale 0.1639 bps, MAE 0.1084 bps, p99 0.6127 bps.
- Binance residual: 1983 paired markets, location 0.0121 bps, scale 0.4742 bps, MAE 0.3615 bps, p99 1.3464 bps.
- Bridge parameters were fitted before policy development and sealed evaluation; raw official, RefPrice, exact TWAP and Binance values remain lineage-distinct.

## Integrity

- Configured source interval: 2026-03-21T00:00:00+00:00 through 2026-08-29T00:00:00+00:00 exclusive.
- Actual retained markets: 2026-03-21T00:05:00+00:00 through 2026-08-26T23:55:00+00:00; optional-source gaps removed no core markets.
- Predictive training, policy development, and sealed testing are chronological and market-disjoint.
- The replay dates were observed in earlier research, so they are computationally sealed here but are not claimed as epistemically untouched.
- Settlement supervision is absent from inference; no row-selection lock was applied to the retained full-history panel.
- Economic results require both recorded books to be no more than two seconds old.
- No database writes, tables, ingesters, sources, runtime exports, deployments, or trading-process changes were made.
