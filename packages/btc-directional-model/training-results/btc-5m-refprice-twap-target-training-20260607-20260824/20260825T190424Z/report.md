# RefPrice-primary / TWAP-60 target training

Run: `20260825T190424Z`
Source commit: `1ba5185d0a20ee96bf5fbe1f362e84b8fb0061af`
Common evaluation: 409 markets, 6,468 decision rows.

| Arm | Family | Settled | W-L | Net P&L | Stress P&L | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| W | q5 | 13 | 13-0 | 10.91 | 10.26 | 0.1419 | 0.0290 |
| W | stratified_payoff | 29 | 19-10 | -7.07 | -8.52 | 0.1419 | 0.0307 |
| W | regime_calibrated | 20 | 16-4 | 5.18 | 4.18 | 0.1425 | 0.0279 |
| W | full_combined | 79 | 58-21 | -2.66 | -6.61 | 0.1383 | 0.0203 |
| W | specialist_distilled | 107 | 84-23 | -13.13 | -18.48 | 0.1515 | 0.0412 |
| R | q5 | 11 | 9-2 | -2.17 | -2.72 | 0.1415 | 0.0325 |
| R | stratified_payoff | 30 | 20-10 | -3.83 | -5.33 | 0.1416 | 0.0317 |
| R | regime_calibrated | 19 | 16-3 | 4.45 | 3.50 | 0.1422 | 0.0275 |
| R | full_combined | 62 | 44-18 | 0.55 | -2.55 | 0.1417 | 0.0307 |
| R | specialist_distilled | 96 | 76-20 | -7.22 | -12.02 | 0.1549 | 0.0457 |
| T | q5 | 12 | 11-1 | 3.99 | 3.39 | 0.1590 | 0.0422 |
| T | stratified_payoff | 54 | 35-19 | -18.12 | -20.82 | 0.1587 | 0.0448 |
| T | regime_calibrated | 37 | 27-10 | -10.28 | -12.13 | 0.1557 | 0.0288 |
| T | full_combined | 84 | 54-30 | -22.88 | -27.08 | 0.1531 | 0.0341 |
| T | specialist_distilled | 95 | 76-19 | -1.65 | -6.40 | 0.1606 | 0.0518 |

W is the watermark refresh, R adds causal PMData RefPrice, and T changes the target to PMData TWAP-60.
All incumbent artifacts, trading processes, runtime bundles, and deployed images remained unchanged.
