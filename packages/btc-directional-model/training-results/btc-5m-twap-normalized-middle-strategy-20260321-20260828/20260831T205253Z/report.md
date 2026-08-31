# Middle-Strategy Tournament

Run: `20260831T205253Z`
Qualification: **trained_evaluated_not_promoted_negative_expectancy**

## Sealed high-level results

| Candidate | PnL | Stress PnL | Coverage | Wins | Losses | W/L | Recovery wins/loss | Brier | PF | Avg cost | Avg entry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| middle_specialist_refit | -88.56 | -127.96 | 46.93% | 460 | 328 | 1.402 | 1.542 | 0.1755 | 0.910 | 0.585 | 86.8 |
| middle_q5_admission | -80.80 | -112.45 | 37.70% | 368 | 265 | 1.389 | 1.546 | 0.1746 | 0.898 | 0.585 | 92.2 |
| middle_agreement_ensemble | -84.99 | -119.74 | 41.39% | 411 | 284 | 1.447 | 1.605 | 0.1748 | 0.902 | 0.594 | 91.7 |
| crossvenue_middle_specialist | -71.36 | -100.36 | 34.54% | 343 | 237 | 1.447 | 1.607 | 0.1759 | 0.901 | 0.594 | 94.2 |
| price_time_calibrated_middle_ensemble | -67.60 | -100.45 | 39.13% | 394 | 263 | 1.498 | 1.637 | 0.1749 | 0.915 | 0.599 | 108.6 |

## Integrity

- Configured source interval: 2026-03-21T00:00:00+00:00 through 2026-08-29T00:00:00+00:00 exclusive.
- Actual retained markets: 2026-03-21T00:05:00+00:00 through 2026-08-26T23:55:00+00:00; optional-source gaps removed no core markets.
- Predictive training, policy development, and sealed testing are chronological and market-disjoint.
- The replay dates were observed in earlier research, so they are computationally sealed here but are not claimed as epistemically untouched.
- TWAP normalization is supervision only; TWAP is absent from inference and no `authentic_only` filter is applied.
- Economic results require both recorded books to be no more than two seconds old.
- No database writes, tables, ingesters, sources, runtime exports, deployments, or trading-process changes were made.
