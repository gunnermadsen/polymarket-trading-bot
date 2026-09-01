# Middle-Strategy Tournament

Run: `20260901T010530Z`
Qualification: **trained_policy_frozen_prospective_evidence_pending**

## Prospective high-level results

| Candidate | PnL | Stress PnL | Coverage | Wins | Losses | W/L | Recovery wins/loss | Brier | PF | Avg cost | Avg entry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| external_flow_only_control | 0.00 | 0.00 | 0.00% | 0 | 0 | — | — | — | — | — | — |
| middle_specialist_refit | 0.00 | 0.00 | 0.00% | 0 | 0 | — | — | — | — | — | — |
| middle_q5_admission | 0.00 | 0.00 | 0.00% | 0 | 0 | — | — | — | — | — | — |
| crossvenue_middle_specialist | 0.00 | 0.00 | 0.00% | 0 | 0 | — | — | — | — | — | — |
| latent_boundary_margin_specialist | 0.00 | 0.00 | 0.00% | 0 | 0 | — | — | — | — | — | — |
| middle_agreement_ensemble | 0.00 | 0.00 | 0.00% | 0 | 0 | — | — | — | — | — | — |
| price_time_calibrated_middle_ensemble | 0.00 | 0.00 | 0.00% | 0 | 0 | — | — | — | — | — | — |

Prospective completeness: **False**; 0 of 7 days and 0 of 2016 expected markets observed.

## Known August 21-28 comparison

| Candidate | PnL | Stress PnL | PF | Coverage | Trades | Wins | Losses | Avg entry | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| external_flow_only_control | -147.95 | -184.95 | 0.840 | 44.07% | 740 | 398 | 342 | 79.2 | 0.1620 |
| middle_specialist_refit | -26.16 | -85.31 | 0.978 | 70.46% | 1183 | 802 | 381 | 75.3 | 0.1541 |
| middle_q5_admission | -27.39 | -77.79 | 0.975 | 60.04% | 1008 | 649 | 359 | 82.6 | 0.1541 |
| crossvenue_middle_specialist | -124.48 | -176.93 | 0.893 | 62.48% | 1049 | 656 | 393 | 92.9 | 0.1545 |
| latent_boundary_margin_specialist | -34.82 | -92.92 | 0.972 | 69.21% | 1162 | 759 | 403 | 76.8 | 0.1568 |
| middle_agreement_ensemble | -120.66 | -173.76 | 0.890 | 63.25% | 1062 | 730 | 332 | 98.6 | 0.1536 |
| price_time_calibrated_middle_ensemble | -48.65 | -111.55 | 0.962 | 74.93% | 1258 | 859 | 399 | 81.1 | 0.1765 |

## Paired change from the prior normalized tournament

| Candidate | PnL delta | Stress PnL delta | Brier delta | PF delta | Coverage delta |
|---|---:|---:|---:|---:|---:|
| middle_specialist_refit | 116.80 | 95.80 | 0.00013 | 0.123 | 25.01% |
| middle_q5_admission | 44.97 | 18.27 | 0.00049 | 0.093 | 31.80% |
| crossvenue_middle_specialist | -66.51 | -87.81 | -0.00071 | -0.031 | 25.37% |
| middle_agreement_ensemble | -4.60 | -30.45 | -0.00014 | 0.052 | 30.79% |
| price_time_calibrated_middle_ensemble | 40.85 | 36.05 | 0.00910 | 0.031 | 5.72% |

## Prescribed sealed counterfactuals

| Candidate | 150-180 PnL / PF / trades | p80 early PnL / PF / trades |
|---|---:|---:|
| external_flow_only_control | 0.00 / — / 0 | 0.00 / — / 0 |
| middle_specialist_refit | 0.00 / — / 0 | 0.00 / — / 0 |
| middle_q5_admission | 0.00 / — / 0 | 0.00 / — / 0 |
| crossvenue_middle_specialist | 0.00 / — / 0 | 0.00 / — / 0 |
| latent_boundary_margin_specialist | 0.00 / — / 0 | 0.00 / — / 0 |
| middle_agreement_ensemble | 0.00 / — / 0 | 0.00 / — / 0 |
| price_time_calibrated_middle_ensemble | 0.00 / — / 0 | 0.00 / — / 0 |

## Settlement bridge

- RefPrice residual: 1983 paired markets, location -0.0000 bps, scale 0.1639 bps, MAE 0.1084 bps, p99 0.6127 bps.
- Binance residual: 1983 paired markets, location 0.0121 bps, scale 0.4742 bps, MAE 0.3615 bps, p99 1.3464 bps.
- Bridge parameters were fitted before policy development and sealed evaluation; raw official, RefPrice, exact TWAP and Binance values remain lineage-distinct.

## Integrity

- Configured source interval: 2026-03-21T00:00:00+00:00 through 2026-09-05T00:00:00+00:00 exclusive.
- Actual retained markets: 2026-03-21T00:05:00+00:00 through 2026-08-26T23:55:00+00:00; optional-source gaps removed no core markets.
- Predictive training, rolling policy selection, known comparison, and prospective testing are chronological and market-disjoint.
- August 21-28 is a known comparator; August 29-September 5 is the prospective qualification window.
- Settlement supervision is absent from inference; no row-selection lock was applied to the retained full-history panel.
- Economic results require both recorded books to be no more than two seconds old.
- No database writes, tables, ingesters, sources, runtime exports, deployments, or trading-process changes were made.
