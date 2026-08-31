# Middle-Strategy Tournament

Run: `20260831T161810Z`
Qualification: **trained_evaluated_not_promoted_negative_expectancy**

## Sealed high-level results

| Candidate | PnL | Stress PnL | Coverage | Wins | Losses | W/L | Recovery wins/loss | Brier | PF | Avg cost | Avg entry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| middle_specialist_refit | -92.19 | -154.54 | 33.78% | 852 | 395 | 2.157 | 2.327 | 0.1602 | 0.927 | 0.679 | 108.6 |
| middle_q5_admission | -153.44 | -218.99 | 35.51% | 835 | 476 | 1.754 | 1.963 | 0.1547 | 0.893 | 0.641 | 108.5 |
| middle_agreement_ensemble | -172.80 | -237.60 | 35.10% | 852 | 444 | 1.919 | 2.191 | 0.1569 | 0.876 | 0.665 | 107.8 |
| crossvenue_middle_specialist | -67.95 | -121.20 | 28.85% | 710 | 355 | 2.000 | 2.131 | 0.1544 | 0.939 | 0.660 | 108.1 |
| price_time_calibrated_middle_ensemble | -143.30 | -199.90 | 30.66% | 751 | 381 | 1.971 | 2.240 | 0.1537 | 0.880 | 0.670 | 106.7 |

## Integrity

- Configured source interval: 2026-03-21T00:00:00+00:00 through 2026-08-29T00:00:00+00:00 exclusive.
- Actual retained markets: 2026-03-21T00:05:00+00:00 through 2026-08-26T23:55:00+00:00; optional-source gaps removed no core markets.
- Training is chronological and market-disjoint. The August 14–28 replay is opened only after models and policies are frozen.
- The replay dates were observed in earlier research, so they are computationally sealed here but are not claimed as epistemically untouched.
- Official settlement outcomes are labels. TWAP and `authentic_only` are absent from inference and selection.
- Economic results require both recorded books to be no more than two seconds old.
- No database writes, tables, ingesters, sources, runtime exports, deployments, or trading-process changes were made.
