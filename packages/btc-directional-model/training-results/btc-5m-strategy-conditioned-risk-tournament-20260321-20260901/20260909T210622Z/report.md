# Strategy-Conditioned Candidate Risk Tournament

Run: `20260909T210622Z`

Every PnL, PF, W/L, drawdown, and recovery value below belongs to the named strategy replay with the risk policy applied. Risk estimators themselves are measured by Brier/log loss/AUROC; interventions are measured by blocked-loss and blocked-winner behavior.

## Universal risk-policy comparison

| Risk policy | Threshold | Trades | W/L | Strategy PnL with risk | Delta PnL | Stress PnL | Coverage | PF | Recovery | Max DD | Bad blocked | Good blocked | Alignment | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_risk | — | 2965 | 2074/891 | $-317.71 | $0.00 | $-465.96 | 100.0% | 0.893 | 2.606 | $332.42 | 0.0% | 0.0% | —x | — |
| logistic_economics | -0.210134 | 2608 | 1891/717 | $-220.43 | $97.28 | $-350.83 | 88.0% | 0.912 | 2.893 | $252.08 | 20.1% | 9.4% | 2.137x | 0.2065 |
| boosted_economics | -0.088623 | 2850 | 2028/822 | $-249.60 | $68.11 | $-392.10 | 96.1% | 0.911 | 2.709 | $277.46 | 8.0% | 2.5% | 3.241x | 0.2109 |
| boosted_market_context | -0.323353 | 2606 | 1883/723 | $-278.19 | $39.51 | $-408.49 | 87.9% | 0.890 | 2.927 | $304.20 | 19.6% | 9.4% | 2.089x | 0.2086 |
| boosted_recent_history | -0.062154 | 2839 | 2021/818 | $-265.33 | $52.38 | $-407.28 | 95.8% | 0.905 | 2.730 | $299.91 | 8.3% | 2.7% | 3.132x | 0.2066 |
| economic_harm | -0.495594 | 1618 | 985/633 | $-269.14 | $48.56 | $-350.04 | 54.6% | 0.862 | 1.806 | $284.59 | 33.6% | 57.4% | 0.585x | 0.2066 |
| frozen_pilot_boosted_recent_history | -0.062154 | 2839 | 2021/818 | $-265.33 | $52.38 | $-407.28 | 95.8% | 0.905 | 2.730 | $299.91 | 8.3% | 2.7% | 3.132x | 0.2066 |
| matched_confidence | 0.743271 | 1049 | 840/209 | $38.00 | $355.71 | $-14.45 | 35.4% | 1.049 | 3.830 | $34.56 | 86.9% | 73.0% | 1.190x | — |
| matched_edge | 0.096441 | 1706 | 1041/665 | $-307.01 | $10.70 | $-392.31 | 57.5% | 0.851 | 1.839 | $327.59 | 29.3% | 54.2% | 0.540x | — |

Universal research champion selected without opening the test cohort: `economic_harm`.
Untouched-test qualification: **research_not_qualified**. net_pnl_improved=pass, stress_pnl_improved=pass, max_drawdown_reduced=pass, risk_alignment_above_one=fail, avoided_losses_exceed_missed_profit=pass, aggregate_coverage_floor=pass, per_strategy_coverage_floor=fail, positive_majority_of_strategies=pass, no_materially_destructive_strategy=fail, beats_matched_controls=fail.

## Strategy-by-risk-policy test matrix

### no_risk

| Strategy model | Base PnL | PnL with risk | Delta | Trades | W/L | Coverage | Avg entry | PF | Recovery | Max DD | Loss capture | Opportunity rejection | Alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| btc-5m-specialist-distilled-fair-value-paper-20260823-v1 | $-22.60 | $-22.60 | $0.00 | 488 | 366/122 | 100.0% | 144.4s | 0.947 | 3.168 | $46.67 | 0.0% | 0.0% | —x |
| btc-5m-bridge-aware-specialist-umr-20260902-confidence-080 | $-58.19 | $-58.19 | $0.00 | 539 | 410/129 | 100.0% | 92.5s | 0.874 | 3.635 | $68.53 | 0.0% | 0.0% | —x |
| btc-5m-extended-specialist-official-umr-20260902-confidence-070 | $-56.74 | $-56.74 | $0.00 | 358 | 220/138 | 100.0% | 130.6s | 0.870 | 1.833 | $78.23 | 0.0% | 0.0% | —x |
| btc-5m-official-high-precision-loss-veto-umr-20260902 | $-97.46 | $-97.46 | $0.00 | 377 | 235/142 | 100.0% | 121.1s | 0.794 | 2.085 | $108.57 | 0.0% | 0.0% | —x |
| btc-5m-official-temporal-consensus-umr-20260902-confidence-075 | $-63.56 | $-63.56 | $0.00 | 350 | 211/139 | 100.0% | 126.3s | 0.853 | 1.780 | $86.61 | 0.0% | 0.0% | —x |
| btc-5m-official-vwap-admission-umr-20260902 | $-19.15 | $-19.15 | $0.00 | 853 | 632/221 | 100.0% | 79.7s | 0.974 | 2.936 | $73.35 | 0.0% | 0.0% | —x |

### logistic_economics

| Strategy model | Base PnL | PnL with risk | Delta | Trades | W/L | Coverage | Avg entry | PF | Recovery | Max DD | Loss capture | Opportunity rejection | Alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| btc-5m-specialist-distilled-fair-value-paper-20260823-v1 | $-22.60 | $-22.60 | $0.00 | 488 | 366/122 | 100.0% | 144.4s | 0.947 | 3.168 | $46.67 | 0.0% | 0.0% | —x |
| btc-5m-bridge-aware-specialist-umr-20260902-confidence-080 | $-58.19 | $-35.09 | $23.10 | 524 | 406/118 | 97.2% | 90.5s | 0.919 | 3.745 | $45.91 | 8.5% | 1.0% | 8.740x |
| btc-5m-extended-specialist-official-umr-20260902-confidence-070 | $-56.74 | $-66.86 | $-10.12 | 271 | 168/103 | 75.7% | 121.8s | 0.802 | 2.033 | $72.11 | 25.4% | 23.6% | 1.073x |
| btc-5m-official-high-precision-loss-veto-umr-20260902 | $-97.46 | $-86.77 | $10.70 | 336 | 213/123 | 89.1% | 117.4s | 0.792 | 2.186 | $92.76 | 14.1% | 11.5% | 1.226x |
| btc-5m-official-temporal-consensus-umr-20260902-confidence-075 | $-63.56 | $-37.19 | $26.37 | 257 | 163/94 | 73.4% | 120.7s | 0.878 | 1.974 | $51.47 | 35.3% | 26.1% | 1.352x |
| btc-5m-official-vwap-admission-umr-20260902 | $-19.15 | $28.09 | $47.24 | 732 | 575/157 | 85.8% | 74.9s | 1.049 | 3.492 | $66.98 | 29.0% | 9.0% | 3.211x |

### boosted_economics

| Strategy model | Base PnL | PnL with risk | Delta | Trades | W/L | Coverage | Avg entry | PF | Recovery | Max DD | Loss capture | Opportunity rejection | Alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| btc-5m-specialist-distilled-fair-value-paper-20260823-v1 | $-22.60 | $-22.99 | $-0.39 | 488 | 366/122 | 100.0% | 144.4s | 0.946 | 3.171 | $47.07 | 0.0% | 0.8% | 0.000x |
| btc-5m-bridge-aware-specialist-umr-20260902-confidence-080 | $-58.19 | $-55.68 | $2.51 | 538 | 410/128 | 99.8% | 92.4s | 0.879 | 3.643 | $66.02 | 0.8% | 0.0% | —x |
| btc-5m-extended-specialist-official-umr-20260902-confidence-070 | $-56.74 | $-48.58 | $8.16 | 320 | 201/119 | 89.4% | 127.1s | 0.874 | 1.933 | $70.15 | 13.8% | 8.6% | 1.594x |
| btc-5m-official-high-precision-loss-veto-umr-20260902 | $-97.46 | $-88.12 | $9.35 | 375 | 236/139 | 99.5% | 121.2s | 0.811 | 2.094 | $99.22 | 2.8% | 0.0% | —x |
| btc-5m-official-temporal-consensus-umr-20260902-confidence-075 | $-63.56 | $-46.16 | $17.39 | 330 | 204/126 | 94.3% | 124.7s | 0.884 | 1.832 | $63.97 | 10.1% | 3.8% | 2.656x |
| btc-5m-official-vwap-admission-umr-20260902 | $-19.15 | $11.94 | $31.09 | 799 | 611/188 | 93.7% | 77.2s | 1.018 | 3.192 | $64.50 | 14.9% | 3.3% | 4.494x |

### boosted_market_context

| Strategy model | Base PnL | PnL with risk | Delta | Trades | W/L | Coverage | Avg entry | PF | Recovery | Max DD | Loss capture | Opportunity rejection | Alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| btc-5m-specialist-distilled-fair-value-paper-20260823-v1 | $-22.60 | $-22.60 | $0.00 | 488 | 366/122 | 100.0% | 144.4s | 0.947 | 3.168 | $46.67 | 0.0% | 0.0% | —x |
| btc-5m-bridge-aware-specialist-umr-20260902-confidence-080 | $-58.19 | $-40.83 | $17.37 | 522 | 404/118 | 96.8% | 90.2s | 0.905 | 3.781 | $51.64 | 8.5% | 1.5% | 5.827x |
| btc-5m-extended-specialist-official-umr-20260902-confidence-070 | $-56.74 | $-73.12 | $-16.38 | 257 | 159/98 | 71.8% | 119.0s | 0.775 | 2.094 | $73.12 | 29.7% | 27.7% | 1.072x |
| btc-5m-official-high-precision-loss-veto-umr-20260902 | $-97.46 | $-98.00 | $-0.54 | 331 | 208/123 | 87.8% | 115.8s | 0.766 | 2.209 | $99.96 | 14.1% | 12.3% | 1.141x |
| btc-5m-official-temporal-consensus-umr-20260902-confidence-075 | $-63.56 | $-48.55 | $15.01 | 257 | 162/95 | 73.4% | 118.7s | 0.844 | 2.020 | $61.28 | 35.3% | 24.2% | 1.458x |
| btc-5m-official-vwap-admission-umr-20260902 | $-19.15 | $4.91 | $24.06 | 751 | 584/167 | 88.0% | 74.7s | 1.008 | 3.469 | $66.27 | 24.4% | 7.6% | 3.217x |

### boosted_recent_history

| Strategy model | Base PnL | PnL with risk | Delta | Trades | W/L | Coverage | Avg entry | PF | Recovery | Max DD | Loss capture | Opportunity rejection | Alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| btc-5m-specialist-distilled-fair-value-paper-20260823-v1 | $-22.60 | $-17.14 | $5.46 | 488 | 367/121 | 100.0% | 144.6s | 0.959 | 3.161 | $47.08 | 1.6% | 0.0% | —x |
| btc-5m-bridge-aware-specialist-umr-20260902-confidence-080 | $-58.19 | $-55.04 | $3.16 | 538 | 410/128 | 99.8% | 92.4s | 0.880 | 3.638 | $65.37 | 0.8% | 0.0% | —x |
| btc-5m-extended-specialist-official-umr-20260902-confidence-070 | $-56.74 | $-42.21 | $14.53 | 316 | 201/115 | 88.3% | 126.8s | 0.888 | 1.968 | $58.32 | 16.7% | 9.1% | 1.833x |
| btc-5m-official-high-precision-loss-veto-umr-20260902 | $-97.46 | $-97.84 | $-0.38 | 369 | 230/139 | 97.9% | 120.2s | 0.789 | 2.097 | $103.42 | 2.1% | 2.1% | 0.993x |
| btc-5m-official-temporal-consensus-umr-20260902-confidence-075 | $-63.56 | $-54.92 | $8.64 | 323 | 199/124 | 92.3% | 123.6s | 0.861 | 1.865 | $70.79 | 10.8% | 5.7% | 1.897x |
| btc-5m-official-vwap-admission-umr-20260902 | $-19.15 | $1.82 | $20.97 | 805 | 614/191 | 94.4% | 76.7s | 1.003 | 3.206 | $67.04 | 13.6% | 2.8% | 4.766x |

### economic_harm

| Strategy model | Base PnL | PnL with risk | Delta | Trades | W/L | Coverage | Avg entry | PF | Recovery | Max DD | Loss capture | Opportunity rejection | Alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| btc-5m-specialist-distilled-fair-value-paper-20260823-v1 | $-22.60 | $-9.84 | $12.76 | 292 | 200/92 | 59.8% | 152.4s | 0.967 | 2.247 | $42.32 | 36.1% | 65.0% | 0.555x |
| btc-5m-bridge-aware-specialist-umr-20260902-confidence-080 | $-58.19 | $-63.21 | $-5.02 | 175 | 102/73 | 32.5% | 122.4s | 0.723 | 1.934 | $72.40 | 50.4% | 77.3% | 0.652x |
| btc-5m-extended-specialist-official-umr-20260902-confidence-070 | $-56.74 | $-24.08 | $32.66 | 262 | 157/105 | 73.2% | 128.4s | 0.923 | 1.619 | $49.75 | 26.8% | 30.0% | 0.894x |
| btc-5m-official-high-precision-loss-veto-umr-20260902 | $-97.46 | $-51.33 | $46.13 | 271 | 164/107 | 71.9% | 127.6s | 0.849 | 1.806 | $73.52 | 32.4% | 34.9% | 0.928x |
| btc-5m-official-temporal-consensus-umr-20260902-confidence-075 | $-63.56 | $-53.73 | $9.83 | 310 | 183/127 | 88.6% | 127.9s | 0.861 | 1.674 | $75.28 | 10.8% | 16.1% | 0.670x |
| btc-5m-official-vwap-admission-umr-20260902 | $-19.15 | $-66.94 | $-47.79 | 308 | 179/129 | 36.1% | 99.8s | 0.822 | 1.688 | $84.01 | 41.6% | 71.7% | 0.581x |

### frozen_pilot_boosted_recent_history

| Strategy model | Base PnL | PnL with risk | Delta | Trades | W/L | Coverage | Avg entry | PF | Recovery | Max DD | Loss capture | Opportunity rejection | Alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| btc-5m-specialist-distilled-fair-value-paper-20260823-v1 | $-22.60 | $-17.14 | $5.46 | 488 | 367/121 | 100.0% | 144.6s | 0.959 | 3.161 | $47.08 | 1.6% | 0.0% | —x |
| btc-5m-bridge-aware-specialist-umr-20260902-confidence-080 | $-58.19 | $-55.04 | $3.16 | 538 | 410/128 | 99.8% | 92.4s | 0.880 | 3.638 | $65.37 | 0.8% | 0.0% | —x |
| btc-5m-extended-specialist-official-umr-20260902-confidence-070 | $-56.74 | $-42.21 | $14.53 | 316 | 201/115 | 88.3% | 126.8s | 0.888 | 1.968 | $58.32 | 16.7% | 9.1% | 1.833x |
| btc-5m-official-high-precision-loss-veto-umr-20260902 | $-97.46 | $-97.84 | $-0.38 | 369 | 230/139 | 97.9% | 120.2s | 0.789 | 2.097 | $103.42 | 2.1% | 2.1% | 0.993x |
| btc-5m-official-temporal-consensus-umr-20260902-confidence-075 | $-63.56 | $-54.92 | $8.64 | 323 | 199/124 | 92.3% | 123.6s | 0.861 | 1.865 | $70.79 | 10.8% | 5.7% | 1.897x |
| btc-5m-official-vwap-admission-umr-20260902 | $-19.15 | $1.82 | $20.97 | 805 | 614/191 | 94.4% | 76.7s | 1.003 | 3.206 | $67.04 | 13.6% | 2.8% | 4.766x |

### matched_confidence

| Strategy model | Base PnL | PnL with risk | Delta | Trades | W/L | Coverage | Avg entry | PF | Recovery | Max DD | Loss capture | Opportunity rejection | Alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| btc-5m-specialist-distilled-fair-value-paper-20260823-v1 | $-22.60 | $0.31 | $22.92 | 399 | 314/85 | 81.8% | 158.2s | 1.001 | 3.690 | $31.11 | 70.5% | 61.5% | 1.147x |
| btc-5m-bridge-aware-specialist-umr-20260902-confidence-080 | $-58.19 | $45.54 | $103.74 | 319 | 273/46 | 59.2% | 83.2s | 1.251 | 4.746 | $23.68 | 74.4% | 45.6% | 1.632x |
| btc-5m-extended-specialist-official-umr-20260902-confidence-070 | $-56.74 | $5.27 | $62.01 | 64 | 46/18 | 17.9% | 113.0s | 1.087 | 2.352 | $15.41 | 93.5% | 87.7% | 1.066x |
| btc-5m-official-high-precision-loss-veto-umr-20260902 | $-97.46 | $-13.48 | $83.99 | 60 | 39/21 | 15.9% | 121.7s | 0.809 | 2.295 | $18.98 | 93.7% | 91.5% | 1.024x |
| btc-5m-official-temporal-consensus-umr-20260902-confidence-075 | $-63.56 | $2.78 | $66.34 | 66 | 47/19 | 18.9% | 114.4s | 1.043 | 2.371 | $15.55 | 92.8% | 86.7% | 1.070x |
| btc-5m-official-vwap-admission-umr-20260902 | $-19.15 | $-2.43 | $16.72 | 141 | 121/20 | 16.5% | 66.0s | 0.971 | 6.230 | $22.44 | 91.0% | 80.9% | 1.125x |

### matched_edge

| Strategy model | Base PnL | PnL with risk | Delta | Trades | W/L | Coverage | Avg entry | PF | Recovery | Max DD | Loss capture | Opportunity rejection | Alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| btc-5m-specialist-distilled-fair-value-paper-20260823-v1 | $-22.60 | $-6.03 | $16.58 | 274 | 186/88 | 56.1% | 152.6s | 0.979 | 2.160 | $34.32 | 40.2% | 68.6% | 0.586x |
| btc-5m-bridge-aware-specialist-umr-20260902-confidence-080 | $-58.19 | $-63.83 | $-5.64 | 205 | 125/80 | 38.0% | 115.8s | 0.750 | 2.084 | $77.66 | 41.9% | 71.2% | 0.588x |
| btc-5m-extended-specialist-official-umr-20260902-confidence-070 | $-56.74 | $-41.75 | $14.99 | 294 | 178/116 | 82.1% | 126.2s | 0.883 | 1.737 | $77.47 | 15.9% | 19.5% | 0.816x |
| btc-5m-official-high-precision-loss-veto-umr-20260902 | $-97.46 | $-62.47 | $34.99 | 288 | 174/114 | 76.4% | 127.1s | 0.828 | 1.843 | $82.71 | 28.9% | 28.9% | 0.998x |
| btc-5m-official-temporal-consensus-umr-20260902-confidence-075 | $-63.56 | $-62.55 | $1.00 | 338 | 202/136 | 96.6% | 127.5s | 0.851 | 1.745 | $89.94 | 3.6% | 7.1% | 0.506x |
| btc-5m-official-vwap-admission-umr-20260902 | $-19.15 | $-70.37 | $-51.22 | 307 | 176/131 | 36.0% | 102.0s | 0.816 | 1.647 | $87.98 | 40.7% | 72.2% | 0.564x |

## Selected policy by natural entry bucket

| Isolated candidate bucket | Trades with risk | W/L | PnL with risk | Delta PnL | Coverage | Avg entry | PF | Recovery | Loss capture | Opportunity rejection | Alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15_59 | 0 | 0/0 | $0.00 | $0.00 | —% | —s | — | — | —% | —% | —x |
| 60_89 | 355 | 260/95 | $116.89 | $-77.34 | 30.5% | 68.3s | 1.382 | 1.980 | 57.4% | 72.5% | 0.792x |
| 90_119 | 371 | 240/131 | $-37.63 | $75.02 | 40.4% | 97.2s | 0.911 | 2.012 | 47.6% | 65.1% | 0.731x |
| 120_149 | 436 | 254/182 | $-99.96 | $5.59 | 79.4% | 128.4s | 0.819 | 1.704 | 12.7% | 29.1% | 0.436x |
| 150_179 | 787 | 433/354 | $-263.31 | $54.58 | 78.8% | 156.0s | 0.748 | 1.636 | 16.5% | 28.2% | 0.585x |
| 180_240 | 294 | 188/106 | $-48.37 | $-15.36 | 60.7% | 188.8s | 0.858 | 2.068 | 24.2% | 55.9% | 0.433x |

## Leave-one-strategy-out transfer

The selected estimator was re-thresholded on five strategies in policy calibration and applied to the excluded sixth strategy in the untouched test cohort.

| Held-out strategy | Threshold | Base PnL | PnL with risk | Delta | Coverage | Loss capture | Opportunity rejection | Alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| btc-5m-specialist-distilled-fair-value-paper-20260823-v1 | -0.428914 | $-22.60 | $-20.13 | $2.47 | 67.0% | 26.2% | 54.6% | 0.480x |
| btc-5m-bridge-aware-specialist-umr-20260902-confidence-080 | -0.511398 | $-58.19 | $-71.36 | $-13.16 | 30.4% | 51.2% | 79.3% | 0.645x |
| btc-5m-extended-specialist-official-umr-20260902-confidence-070 | -0.417909 | $-56.74 | $-44.72 | $12.02 | 85.5% | 13.0% | 16.8% | 0.776x |
| btc-5m-official-high-precision-loss-veto-umr-20260902 | -0.420438 | $-97.46 | $-77.04 | $20.42 | 80.9% | 19.7% | 24.3% | 0.813x |
| btc-5m-official-temporal-consensus-umr-20260902-confidence-075 | -0.416952 | $-63.56 | $-69.13 | $-5.57 | 95.1% | 2.9% | 7.1% | 0.405x |
| btc-5m-official-vwap-admission-umr-20260902 | -0.506358 | $-19.15 | $-72.53 | $-53.38 | 35.1% | 42.1% | 72.9% | 0.577x |

## Time-bucket, side, and transition evidence

The complete risk policy × strategy × natural-entry-bucket × side matrix is stored in `ledgers/slice-matrix.parquet` and `metrics.json`. Empty and sub-30-trade cells are retained and marked statistically insufficient. Baseline-bucket to resulting-bucket deferrals and abstentions are stored in `ledgers/bucket-transitions.parquet`.

## Data and integrity

- The risk-estimator construction pool uses causal OOF strategy candidates from the established March-August archive; strategy identity is deliberately excluded from predictive features.
- Exact admission candidate ledgers for all six currently running strategy models overlap only August 20-25, so threshold calibration uses August 20 and the untouched comparison uses August 21-25.
- Archived strategy candidate parity is unavailable for August 26-September 1 for five strategies; those dates were not synthesized or substituted.
- The archived common panel begins at second 60, so the 15-59 bucket is reported as unavailable rather than inferred.
- Fair-value strategy candidates are exact immutable-runtime-artifact replays; the other five are their frozen tournament admission ledgers.
- This is research-only pilot evidence. No model or policy was deployed and no trading process was changed.
