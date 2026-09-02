# Full-August VWAP-Curve Admission Tournament

Run: `20260901T163321Z`
Qualification: **trained_evaluated_not_deployed**

## High-level preferred-mode results: full August, combined 60–180 replay

| Candidate | Admission | PnL | Stress PnL | PF | Coverage | W | L | W/L | Recovery wins/loss | Avg entry | Avg cost | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| external_flow_only_control | hybrid_full_vwap_dual_l2 | -2.04 | -2.24 | 0.506 | 0.04% | 3 | 1 | 3.000 | 5.926 | 180.0 | 0.838 | 0.1825 |
| middle_specialist_refit | hybrid_full_vwap_no_l2 | 58.48 | 19.48 | 1.193 | 8.74% | 712 | 68 | 10.471 | 8.775 | 176.3 | 0.886 | 0.1718 |
| middle_q5_admission | hybrid_full_vwap_dual_l2 | 31.89 | 19.74 | 1.273 | 2.72% | 216 | 27 | 8.000 | 6.284 | 175.7 | 0.849 | 0.1722 |
| crossvenue_middle_specialist | hybrid_full_vwap_dual_l2 | 18.73 | 3.13 | 1.112 | 3.49% | 273 | 39 | 7.000 | 6.295 | 168.4 | 0.849 | 0.1738 |
| latent_boundary_margin_specialist | hybrid_full_vwap_dual_l2 | -9.10 | -28.15 | 0.959 | 4.27% | 329 | 52 | 6.327 | 6.595 | 168.3 | 0.855 | 0.1738 |
| middle_agreement_ensemble | hybrid_full_vwap_dual_l2 | 10.99 | -9.76 | 1.055 | 4.65% | 370 | 45 | 8.222 | 7.791 | 175.6 | 0.874 | 0.1724 |
| price_time_calibrated_middle_ensemble | hybrid_full_vwap_no_l2 | 78.97 | 34.47 | 1.238 | 9.97% | 815 | 75 | 10.867 | 8.777 | 172.7 | 0.886 | 0.1722 |

## Evaluation contract

- Predictive fitting, probability calibration, admission fitting, and policy selection use only markets before August 1.
- August 1–31 is opened once after selection is frozen and is reported as full month, pre-cutover, and post-cutover.
- The 60–89, 90–119, 120–149, and 150–180 results are independent entry replays; combined 60–180 resets and selects the first crossing across the full range.
- Official resolved outcomes are the only predictive supervision. TWAP, RefPrice settlement normalization, and bridge targets are absent.
- All fourteen VWAP sizes are admission-only; directional candidate contracts remain unchanged.
- Optional source gaps preserve rows and are encoded as missingness. No authentic-only filter is present.
- No database writes, tables, schemas, ingesters, sources, runtime exports, deployments, or image rebuilds occurred.
