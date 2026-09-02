# Extended Specialist and Orthogonal Strategy Tournament

Run: `20260901T231526Z`
Qualification: **trained_evaluated_not_deployed**

## Sealed high-level results

| Candidate | Preferred admission | Best PnL bucket | PnL | Stress PnL | PF | Expectancy/trade | Coverage | Wins/Losses | Win rate | Recovery wins/loss | Avg entry | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen_specialist_control | programmatic | 150_180 | -0.45 | -7.45 | 0.997 | -0.003 | 8.10% | 95/45 | 0.68 | 2.118 | 159.5 | 0.1762 |
| extended_specialist_official | programmatic | 60_89 | 39.88 | 35.13 | 1.500 | 0.420 | 5.50% | 70/25 | 0.74 | 1.866 | 68.7 | 0.1513 |
| bridge_aware_specialist | programmatic | 60_89 | 37.53 | 18.78 | 1.154 | 0.100 | 21.70% | 314/61 | 0.84 | 4.459 | 69.2 | 0.1512 |
| settlement_agnostic_trajectory | programmatic | 60_89 | -21.27 | -59.12 | 0.970 | -0.028 | 43.81% | 547/210 | 0.72 | 2.684 | 65.3 | 0.1591 |
| crossvenue_lead_lag | programmatic | 60_89 | -120.74 | -169.34 | 0.899 | -0.124 | 56.25% | 465/507 | 0.48 | 1.020 | 65.8 | 0.1781 |
| refprice_residual_specialist | programmatic | 60_89 | -74.39 | -146.19 | 0.951 | -0.052 | 83.10% | 834/602 | 0.58 | 1.456 | 61.6 | 0.2302 |
| specialist_dual_head_economics | learned_enter_now | 60_89 | -28.64 | -92.44 | 0.979 | -0.022 | 73.84% | 832/444 | 0.65 | 1.915 | 64.1 | 0.1513 |

## Integrity

- Predictor fit, development policy selection, and sealed evaluation are chronological and market-disjoint.
- Official outcomes remain canonical for every candidate except the explicitly identified bridge-supervised arm.
- TWAP, outcomes, labels, Polymarket prices, and execution VWAP fields are absent from every directional feature contract.
- The learned admission target is enter-now realized stress edge; it contains no best-later comparison.
- No data row lock, database write, new table, schema, ingester, source, runtime deployment, or image build occurred.
- Limitation: The August 20-25 sealed block is chronological for this run but has been observed during earlier research and is not epistemically fresh.
- Limitation: The reusable full-history panel begins at second 60, so the planned 15-59 Specialist diagnostic is reported from its immutable original tournament rather than retrained here.
- Limitation: Kraken L2 is incremental update-flow rather than reconstructed full-book depth and ends before the sealed block.
- Limitation: Projected PnL assumes recorded ask VWAP5 was fillable and does not model queue position.
