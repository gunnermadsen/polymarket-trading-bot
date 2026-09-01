# Post-Training Findings

Run: `20260901T194936Z`

Producing commit: `241543751a2c5de2e60048d3394fa057a70bf595`

Artifact SHA-256: `34dcf66ea1dcba6166c6e18386743355cc75fbdb73f864fd860ff4203748ec25`

## Outcome

The seven-candidate tournament completed with intact chronological separation: 42,046 fit markets from March 21 through August 13, 1,725 admission-development markets from August 14 through August 19, and 1,728 sealed markets from August 20 through August 25.

The development-selected hybrid admission modes did not qualify. They admitted no trades before 150 seconds and all seven preferred combined results had negative sealed expectancy. The run must not be promoted or deployed.

Two programmatic admission policies that were independently selected on development and frozen before the sealed reveal did produce positive stressed results. These are valid tournament findings, not post-hoc threshold searches.

## Valid frozen-policy winners

| Candidate | Admission | Bucket | Period | PnL | Stress PnL | PF | Expectancy/trade | Coverage | Wins | Losses | Win rate | Recovery wins/loss | Avg entry | Brier |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Price-Time calibrated ensemble | Programmatic | 60–89 | Development Aug 14–19 | +25.64 | +19.99 | 1.253 | +0.227 | 6.54% | 81 | 32 | 71.68% | 2.020 | 69.8 | — |
| Price-Time calibrated ensemble | Programmatic | 60–89 | Sealed Aug 20–25 | **+17.37** | **+3.12** | **1.060** | **+0.061** | **16.49%** | **197** | **88** | **69.12%** | **2.113** | **67.9** | **0.1729** |
| Middle Q5 | Programmatic | 90–119 | Development Aug 14–19 | +22.57 | +19.22 | 1.373 | +0.337 | 3.88% | 46 | 21 | 68.66% | 1.595 | 98.8 | — |
| Middle Q5 | Programmatic | 90–119 | Sealed Aug 20–25 | **+23.78** | **+15.68** | **1.141** | **+0.147** | **9.38%** | **107** | **55** | **66.05%** | **1.705** | **97.5** | **0.1728** |

No other candidate/admission/time-bucket combination had positive sealed stressed PnL.

## Hybrid-admission failure

The learned hybrid veto was an overly conservative serial `AND` across profitable probability, a lower stress-edge bound, expected shortfall, and enter-now advantage.

For every candidate and every hybrid mode in the 60–89 development bucket:

- The maximum predicted stress-edge lower bound remained below the configured `−0.05` fallback threshold. Candidate maxima ranged from approximately `−0.137` to `−0.322`, so zero rows passed that condition.
- The enter-now regressor compared the current realized edge with the best later executable edge. Early observations were therefore structurally negative. Most candidate/mode maxima remained below the configured `−0.02` threshold.
- The fallback policy was recorded as non-abstaining, but these two continuous veto conditions still eliminated every early row.

This explains why the preferred modes traded only at approximately 176–180 seconds despite the explicit early-entry objective. It is an admission-target and policy-composition defect, not evidence that all early directional models failed.

## Comparison with the preceding tournaments

| Tournament | Candidate | Admission/bucket | PnL | Stress PnL | PF | Expectancy/trade | Coverage | Wins | Losses | Win rate | Recovery wins/loss | Avg entry | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid payoff-admission | Agreement ensemble | Dual-L2, 60–89 | +45.56 | +11.51 | 1.080 | +0.067 | 39.41% | 523 | 158 | 76.80% | 3.070 | 65.9 | 0.2033 |
| Hybrid payoff-admission | Middle Q5 | Dual-L2, 60–89 | +31.15 | −5.95 | 1.048 | +0.042 | 42.94% | 556 | 186 | 74.93% | 2.850 | 66.1 | 0.2028 |
| Full-August VWAP admission | Middle Q5 | Preferred, 150–180 | +31.89 | +19.74 | 1.273 | +0.131 | 2.72% | 216 | 27 | 88.89% | 6.280 | 175.7 | 0.1414 |
| Full-August VWAP admission | Price-Time ensemble | Preferred, 150–180 | +78.97 | +34.47 | 1.238 | +0.089 | 9.97% | 815 | 75 | 91.57% | 8.780 | 172.7 | 0.1414 |
| Current sealed post-cutover | Price-Time ensemble | Programmatic, 60–89 | +17.37 | +3.12 | 1.060 | +0.061 | 16.49% | 197 | 88 | 69.12% | 2.113 | 67.9 | 0.1729 |
| Current sealed post-cutover | Middle Q5 | Programmatic, 90–119 | +23.78 | +15.68 | 1.141 | +0.147 | 9.38% | 107 | 55 | 66.05% | 1.705 | 97.5 | 0.1728 |

The current sealed winners have lower recovery burden than either prior tournament's highlighted candidates. They also remain positive after stress entirely inside the post-cutover period. Coverage is lower than the first tournament, especially for Q5, so these results support continued challenging rather than deployment.

## TWAP and RefPrice diagnostic

Official outcomes were the only supervision. The diagnostic was computed after policy freeze and did not influence model, calibration, admission, or policy selection.

| Diagnostic arm | Development Brier | Delta vs RefPrice-only | Sealed Brier | Delta vs RefPrice-only |
|---|---:|---:|---:|---:|
| RefPrice absolute TWAP | 0.17228 | −0.00162 | 0.16967 | −0.00056 |
| RefPrice non-TWAP basis | 0.17215 | −0.00175 | 0.16937 | −0.00086 |
| RefPrice only | 0.17390 | 0.00000 | 0.17023 | 0.00000 |
| RefPrice relative TWAP | 0.17164 | −0.00226 | 0.16983 | −0.00040 |

The TWAP-derived arms provide only small Brier improvements over RefPrice-only, and the advantage narrows on sealed data. This does not justify restoring synthetic TWAP as canonical supervision. The post-cutover positive programmatic results show that consistent official-outcome training can produce economic edge without settlement normalization controlling inference.

## Required disposition

- Preserve the complete artifact and reports as `not_deployed` and `not_promoted`.
- Do not retrain this completed run.
- Do not use the learned hybrid admission modes for paper or live trading.
- Retain Price-Time 60–89 and Middle Q5 90–119 as the valid frozen-policy findings for a subsequent tournament.
- In a later tournament, remove “current versus best later” as an absolute early-entry veto and calibrate stress-bound residuals separately by time bucket. Keep the programmatic policies as controls and eligible competitors rather than excluding them from preferred-mode selection.

No database writes, tables, schemas, ingesters, runtime exports, deployments, or Docker image rebuilds occurred.
