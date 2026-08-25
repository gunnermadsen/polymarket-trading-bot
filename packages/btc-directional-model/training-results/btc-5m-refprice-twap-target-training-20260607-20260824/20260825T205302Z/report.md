# RefPrice-primary / TWAP-60 target training

Run: `20260825T205302Z`
Source commit: `56593605c1d29a4527ef814ee05d7447c6babb29`
Common evaluation: 487 markets, 17,314 decision rows.

| Arm | Family | Predicted markets | Traded markets | W-L | Net P&L | Stress P&L | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R | q5 | 487 | 0 | 0-0 | 0.00 | 0.00 | 0.1792 | 0.0455 |
| R | stratified_payoff | 487 | 109 | 70-39 | -13.62 | -19.07 | 0.1761 | 0.0377 |
| R | regime_calibrated | 487 | 73 | 47-26 | -12.87 | -16.52 | 0.1761 | 0.0377 |
| R | full_combined | 409 | 163 | 113-50 | -26.92 | -35.07 | 0.1425 | 0.0325 |
| R | specialist_distilled | 487 | 92 | 70-22 | 0.73 | -3.87 | 0.1870 | 0.0210 |
| T | q5 | 487 | 0 | 0-0 | 0.00 | 0.00 | 0.1768 | 0.0353 |
| T | stratified_payoff | 487 | 98 | 62-36 | -8.53 | -13.43 | 0.1763 | 0.0276 |
| T | regime_calibrated | 487 | 58 | 36-22 | -9.24 | -12.14 | 0.1763 | 0.0276 |
| T | full_combined | 409 | 158 | 108-50 | -16.08 | -23.98 | 0.1422 | 0.0302 |
| T | specialist_distilled | 487 | 106 | 78-28 | -12.67 | -17.97 | 0.1883 | 0.0212 |
| RT | q5 | 487 | 0 | 0-0 | 0.00 | 0.00 | 0.1791 | 0.0314 |
| RT | stratified_payoff | 487 | 109 | 74-35 | 2.14 | -3.31 | 0.1776 | 0.0272 |
| RT | regime_calibrated | 487 | 58 | 40-18 | 4.15 | 1.25 | 0.1776 | 0.0272 |
| RT | full_combined | 409 | 154 | 108-46 | -6.35 | -14.05 | 0.1448 | 0.0205 |
| RT | specialist_distilled | 487 | 128 | 94-34 | -17.99 | -24.39 | 0.1877 | 0.0208 |

R is the RefPrice-primary legacy-label refresh. T changes post-August-1 labels to PMData TWAP-60. RT adds causal PMData TWAP-30/60 inputs to T.
All incumbent artifacts, trading processes, runtime bundles, and deployed images remained unchanged.
