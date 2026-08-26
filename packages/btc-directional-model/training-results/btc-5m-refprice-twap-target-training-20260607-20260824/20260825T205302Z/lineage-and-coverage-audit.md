# Lineage and coverage audit

Run: `20260825T205302Z`

Model source commit: `56593605c1d29a4527ef814ee05d7447c6babb29`

## Artifact verification

The run contains 15 joblib artifacts: five preserved families for each of R, T, and RT.

| Family | Preserved fitted components | R/T features | RT features |
|---|---|---:|---:|
| Q5 | early/mid/late experts, directional/asymmetric family proxies, price-time calibration, calibration guard, payoff admission model | 124 total; 10 RefPrice; 0 TWAP | 137 total; 10 RefPrice; 13 TWAP |
| Stratified payoff | shared calibrated outcome and stratified correctness calibration | 124 total; 10 RefPrice; 0 TWAP | 137 total; 10 RefPrice; 13 TWAP |
| Regime calibrated | shared calibrated outcome and stratified/regime correctness calibration | 124 total; 10 RefPrice; 0 TWAP | 137 total; 10 RefPrice; 13 TWAP |
| Full combined | shared calibrated outcome, OOF OI probability modifier, stratified/regime correctness, and OOF/meta loss-severity model | 124 total; 10 RefPrice; 0 TWAP | 137 total; 10 RefPrice; 13 TWAP |
| Specialist distilled | base, stratified, tail-weighted, and terminal-margin teachers; logistic selector; distilled HGB regressor | 82 total; 10 RefPrice; 0 TWAP | 95 total; 10 RefPrice; 13 TWAP |

R uses the official-outcome target. T and RT use official outcomes before August 1 as auxiliary learning and complete PMData TWAP-60 labels from August 1 onward. Only RT contains TWAP predictors.

## Dataset coverage

- Watermark: August 24, 2026.
- Cached decision rows: 551,826 across 13,853 markets.
- Dates with eligible rows: 55 of 79.
- RefPrice-eligible rows: 544,331.
- Causal TWAP-input-eligible rows: 199,651.
- TWAP-labeled rows: 182,840 across 4,589 markets.
- OI-eligible rows: 403,424.
- Full PMXT VWAP curve required on every retained row: 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, and 200 shares.
- No interpolation was used and incomplete source rows were excluded individually.

Dates with zero eligible joined rows were June 13–28, July 9–13, and August 11–13. This did not block training.

August 8 retained 12,438 PMXT/RefPrice rows across 288 markets but had no eligible PMData TWAP input or label rows. August 24 retained 11,136 rows across 282 markets; 11,135 had RefPrice and causal TWAP inputs, while 10,329 rows across 260 markets had complete TWAP-60 labels.

## Common evaluation coverage

The sealed August 23–24 cohort contains 487 TWAP-labeled markets and 17,314 decision rows.

| Family | Prediction markets | Prediction coverage | R trades | T trades | RT trades |
|---|---:|---:|---:|---:|---:|
| Q5 | 487 | 100.0% | 0 | 0 | 0 |
| Stratified payoff | 487 | 100.0% | 109 | 98 | 109 |
| Regime calibrated | 487 | 100.0% | 73 | 58 | 58 |
| Full combined | 409 | 84.0% | 163 | 158 | 154 |
| Specialist distilled | 487 | 100.0% | 92 | 106 | 128 |

Full combined is prediction-eligible on 409 markets because its preserved OI modifier requires causal OI evidence. The other four families produce predictions for all 487 markets. No missing-data coverage rule blocks the overall run.

Q5 produces zero admitted trades under the unchanged incumbent thresholds. Rows pass its confidence, edge, price-bucket, and admission gates, but none pass the refitted payoff lower-bound gate. The maximum payoff lower bound after the preceding gates remains below the fixed threshold in every R/T/RT early and late band. No threshold was retuned to manufacture trades.

## Evaluation result

| Arm | Family | Trades | W-L | Net P&L at VWAP5 | Stress P&L |
|---|---|---:|---:|---:|---:|
| R | Q5 | 0 | 0-0 | 0.00 | 0.00 |
| R | Stratified payoff | 109 | 70-39 | -13.62 | -19.07 |
| R | Regime calibrated | 73 | 47-26 | -12.87 | -16.52 |
| R | Full combined | 163 | 113-50 | -26.92 | -35.07 |
| R | Specialist distilled | 92 | 70-22 | 0.73 | -3.87 |
| T | Q5 | 0 | 0-0 | 0.00 | 0.00 |
| T | Stratified payoff | 98 | 62-36 | -8.53 | -13.43 |
| T | Regime calibrated | 58 | 36-22 | -9.24 | -12.14 |
| T | Full combined | 158 | 108-50 | -16.08 | -23.98 |
| T | Specialist distilled | 106 | 78-28 | -12.67 | -17.97 |
| RT | Q5 | 0 | 0-0 | 0.00 | 0.00 |
| RT | Stratified payoff | 109 | 74-35 | 2.14 | -3.31 |
| RT | Regime calibrated | 58 | 40-18 | 4.15 | 1.25 |
| RT | Full combined | 154 | 108-46 | -6.35 | -14.05 |
| RT | Specialist distilled | 128 | 94-34 | -17.99 | -24.39 |

The complete 5–200 share capacity curve for every arm/family is stored under `results.<arm>.<family>.vwap_capacity` in `metrics.json`. RT regime calibrated remains positive across the recorded curve, from +4.15 at five shares to +117.66 at 200 shares. RT stratified is positive through 100 shares but negative at 200 shares. Full combined and specialist distilled remain negative.

These results are development evidence only. They do not establish a new champion, do not replace the incumbent paper models, and do not qualify any artifact for runtime export.
