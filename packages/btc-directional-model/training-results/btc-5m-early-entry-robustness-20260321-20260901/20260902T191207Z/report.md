# Early-Entry Robustness Tournament

Run: `20260902T191207Z`
Qualification: **trained_evaluated_not_deployed_l2_challengers_inconclusive**

## Primary 60–89-second results

| Candidate | Status | PnL | Stress PnL | PF | Expectancy | Coverage | W/L | Win rate | Recovery | Avg entry | Avg cost | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official_early_control | evaluated_not_deployed | 39.88 | 35.13 | 1.500 | 0.420 | 2.80% | 70/25 | 0.74 | 1.866 | 68.7 | 0.632 | 0.1518 |
| bridge_early_control | evaluated_not_deployed | 37.53 | 18.78 | 1.154 | 0.100 | 11.06% | 314/61 | 0.84 | 4.459 | 69.2 | 0.802 | 0.1518 |
| official_dualvenue_l2_residual | inconclusive_l2_value_official_probability_fallback | 24.12 | 15.07 | 1.169 | 0.133 | 5.34% | 143/38 | 0.79 | 3.219 | 70.1 | 0.745 | 0.1518 |
| official_high_precision_loss_veto | evaluated_not_deployed | 18.39 | 15.99 | 1.401 | 0.383 | 1.42% | 33/15 | 0.69 | 1.570 | 69.0 | 0.589 | 0.1518 |
| official_capacity_aware_admission | evaluated_not_deployed | 15.98 | 13.53 | 1.331 | 0.326 | 1.45% | 33/16 | 0.67 | 1.550 | 68.8 | 0.587 | 0.1518 |
| official_temporal_consensus | evaluated_not_deployed | 24.81 | 20.91 | 1.349 | 0.318 | 2.30% | 56/22 | 0.72 | 1.887 | 67.7 | 0.633 | 0.1518 |
| official_dualvenue_l2_veto | not_evaluable_no_sealed_l2_evidence | 0.00 | 0.00 | — | — | 0.00% | 0/0 | — | — | — | — | 0.1518 |

## PnL by independent entry bucket

| Candidate | 60–89 | 90–119 | 120–149 | 150–180 | Combined 60–180 |
|---|---:|---:|---:|---:|---:|
| official_early_control | 39.88 | -14.59 | -6.55 | -75.08 | -40.80 |
| bridge_early_control | 37.53 | -56.74 | -10.72 | -52.19 | -85.94 |
| official_dualvenue_l2_residual | 24.12 | -14.59 | -50.89 | -75.08 | -78.77 |
| official_high_precision_loss_veto | 18.39 | -66.85 | -16.21 | -84.78 | -90.58 |
| official_capacity_aware_admission | 15.98 | -8.42 | -2.69 | -61.18 | -31.65 |
| official_temporal_consensus | 24.81 | -14.59 | -53.51 | -77.57 | -55.32 |
| official_dualvenue_l2_veto | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## UP/DOWN behavior in the primary bucket

| Candidate | UP trades | DOWN trades | UP PnL | DOWN PnL | UP stress | DOWN stress |
|---|---:|---:|---:|---:|---:|---:|
| official_early_control | 35 (36.8%) | 60 (63.2%) | -1.39 | 41.27 | -3.14 | 38.27 |
| bridge_early_control | 196 (52.3%) | 179 (47.7%) | 13.76 | 23.77 | 3.96 | 14.82 |
| official_dualvenue_l2_residual | 99 (54.7%) | 82 (45.3%) | -17.12 | 41.24 | -22.07 | 37.14 |
| official_high_precision_loss_veto | 16 (33.3%) | 32 (66.7%) | -13.99 | 32.38 | -14.79 | 30.78 |
| official_capacity_aware_admission | 16 (32.7%) | 33 (67.3%) | -13.99 | 29.97 | -14.79 | 28.32 |
| official_temporal_consensus | 27 (34.6%) | 51 (65.4%) | -6.03 | 30.84 | -7.38 | 28.29 |
| official_dualvenue_l2_veto | 0 (—%) | 0 (—%) | 0.00 | 0.00 | 0.00 | 0.00 |

## Settlement-regime probability quality

| Predictor | Pre-cutover OOF Brier | Sealed post-cutover Brier |
|---|---:|---:|
| official | 0.1472 | 0.1518 |
| bridge | 0.1484 | 0.1518 |

## Primary-bucket VWAP capacity

| Candidate | Quantity | Trades | PnL | Stress PnL | PF | Expectancy |
|---|---:|---:|---:|---:|---:|---:|
| official_early_control | 5 | 95 | 39.88 | 35.13 | 1.500 | 0.420 |
| official_early_control | 10 | 95 | 79.64 | 70.14 | 1.499 | 0.838 |
| official_early_control | 15 | 95 | 119.17 | 104.92 | 1.497 | 1.254 |
| official_early_control | 20 | 95 | 158.54 | 139.54 | 1.496 | 1.669 |
| official_early_control | 25 | 95 | 197.78 | 174.03 | 1.495 | 2.082 |
| official_early_control | 30 | 95 | 236.92 | 208.42 | 1.494 | 2.494 |
| official_early_control | 40 | 95 | 314.77 | 276.77 | 1.492 | 3.313 |
| official_early_control | 50 | 95 | 392.33 | 344.83 | 1.490 | 4.130 |
| official_early_control | 75 | 95 | 583.16 | 511.91 | 1.485 | 6.139 |
| official_early_control | 100 | 95 | 771.59 | 676.59 | 1.481 | 8.122 |
| official_early_control | 125 | 95 | 958.43 | 839.68 | 1.478 | 10.089 |
| official_early_control | 150 | 95 | 1142.72 | 1000.22 | 1.475 | 12.029 |
| official_early_control | 175 | 95 | 1324.02 | 1157.77 | 1.471 | 13.937 |
| official_early_control | 200 | 95 | 1503.50 | 1313.50 | 1.468 | 15.826 |
| bridge_early_control | 5 | 375 | 37.53 | 18.78 | 1.154 | 0.100 |
| bridge_early_control | 10 | 375 | 74.94 | 37.44 | 1.154 | 0.200 |
| bridge_early_control | 15 | 375 | 111.76 | 55.51 | 1.153 | 0.298 |
| bridge_early_control | 20 | 375 | 148.26 | 73.26 | 1.152 | 0.395 |
| bridge_early_control | 25 | 375 | 184.56 | 90.81 | 1.152 | 0.492 |
| bridge_early_control | 30 | 375 | 220.71 | 108.21 | 1.151 | 0.589 |
| bridge_early_control | 40 | 375 | 291.96 | 141.96 | 1.150 | 0.779 |
| bridge_early_control | 50 | 375 | 361.76 | 174.26 | 1.149 | 0.965 |
| bridge_early_control | 75 | 375 | 529.30 | 248.05 | 1.145 | 1.411 |
| bridge_early_control | 100 | 375 | 689.41 | 314.41 | 1.141 | 1.838 |
| bridge_early_control | 125 | 375 | 841.84 | 373.09 | 1.138 | 2.245 |
| bridge_early_control | 150 | 375 | 985.73 | 423.23 | 1.135 | 2.629 |
| bridge_early_control | 175 | 375 | 1120.57 | 464.32 | 1.131 | 2.988 |
| bridge_early_control | 200 | 375 | 1248.64 | 498.64 | 1.128 | 3.330 |
| official_dualvenue_l2_residual | 5 | 181 | 24.12 | 15.07 | 1.169 | 0.133 |
| official_dualvenue_l2_residual | 10 | 181 | 48.23 | 30.13 | 1.169 | 0.266 |
| official_dualvenue_l2_residual | 15 | 181 | 72.17 | 45.02 | 1.169 | 0.399 |
| official_dualvenue_l2_residual | 20 | 181 | 95.89 | 59.69 | 1.168 | 0.530 |
| official_dualvenue_l2_residual | 25 | 181 | 119.50 | 74.25 | 1.168 | 0.660 |
| official_dualvenue_l2_residual | 30 | 181 | 142.94 | 88.64 | 1.167 | 0.790 |
| official_dualvenue_l2_residual | 40 | 181 | 189.27 | 116.87 | 1.166 | 1.046 |
| official_dualvenue_l2_residual | 50 | 181 | 235.12 | 144.62 | 1.165 | 1.299 |
| official_dualvenue_l2_residual | 75 | 181 | 346.32 | 210.57 | 1.162 | 1.913 |
| official_dualvenue_l2_residual | 100 | 181 | 452.96 | 271.96 | 1.159 | 2.503 |
| official_dualvenue_l2_residual | 125 | 181 | 555.89 | 329.64 | 1.156 | 3.071 |
| official_dualvenue_l2_residual | 150 | 181 | 653.61 | 382.11 | 1.153 | 3.611 |
| official_dualvenue_l2_residual | 175 | 181 | 746.79 | 430.04 | 1.149 | 4.126 |
| official_dualvenue_l2_residual | 200 | 181 | 836.24 | 474.24 | 1.146 | 4.620 |
| official_high_precision_loss_veto | 5 | 48 | 18.39 | 15.99 | 1.401 | 0.383 |
| official_high_precision_loss_veto | 10 | 48 | 36.64 | 31.84 | 1.399 | 0.763 |
| official_high_precision_loss_veto | 15 | 48 | 54.66 | 47.46 | 1.396 | 1.139 |
| official_high_precision_loss_veto | 20 | 48 | 72.64 | 63.04 | 1.394 | 1.513 |
| official_high_precision_loss_veto | 25 | 48 | 90.51 | 78.51 | 1.393 | 1.886 |
| official_high_precision_loss_veto | 30 | 48 | 108.30 | 93.90 | 1.391 | 2.256 |
| official_high_precision_loss_veto | 40 | 48 | 143.72 | 124.52 | 1.389 | 2.994 |
| official_high_precision_loss_veto | 50 | 48 | 178.85 | 154.85 | 1.387 | 3.726 |
| official_high_precision_loss_veto | 75 | 48 | 264.46 | 228.46 | 1.381 | 5.510 |
| official_high_precision_loss_veto | 100 | 48 | 348.34 | 300.34 | 1.376 | 7.257 |
| official_high_precision_loss_veto | 125 | 48 | 430.39 | 370.39 | 1.372 | 8.966 |
| official_high_precision_loss_veto | 150 | 48 | 510.87 | 438.87 | 1.367 | 10.643 |
| official_high_precision_loss_veto | 175 | 48 | 590.34 | 506.34 | 1.363 | 12.299 |
| official_high_precision_loss_veto | 200 | 48 | 668.86 | 572.86 | 1.360 | 13.935 |
| official_capacity_aware_admission | 5 | 49 | 15.98 | 13.53 | 1.331 | 0.326 |
| official_capacity_aware_admission | 10 | 49 | 31.81 | 26.91 | 1.329 | 0.649 |
| official_capacity_aware_admission | 15 | 49 | 47.43 | 40.08 | 1.327 | 0.968 |
| official_capacity_aware_admission | 20 | 49 | 62.99 | 53.19 | 1.325 | 1.286 |
| official_capacity_aware_admission | 25 | 49 | 78.46 | 66.21 | 1.323 | 1.601 |
| official_capacity_aware_admission | 30 | 49 | 93.83 | 79.13 | 1.322 | 1.915 |
| official_capacity_aware_admission | 40 | 49 | 124.42 | 104.82 | 1.320 | 2.539 |
| official_capacity_aware_admission | 50 | 49 | 154.70 | 130.20 | 1.318 | 3.157 |
| official_capacity_aware_admission | 75 | 49 | 228.00 | 191.25 | 1.312 | 4.653 |
| official_capacity_aware_admission | 100 | 49 | 299.56 | 250.56 | 1.307 | 6.114 |
| official_capacity_aware_admission | 125 | 49 | 369.30 | 308.05 | 1.303 | 7.537 |
| official_capacity_aware_admission | 150 | 49 | 437.48 | 363.98 | 1.299 | 8.928 |
| official_capacity_aware_admission | 175 | 49 | 504.61 | 418.86 | 1.295 | 10.298 |
| official_capacity_aware_admission | 200 | 49 | 570.57 | 472.57 | 1.292 | 11.644 |
| official_temporal_consensus | 5 | 78 | 24.81 | 20.91 | 1.349 | 0.318 |
| official_temporal_consensus | 10 | 78 | 49.47 | 41.67 | 1.347 | 0.634 |
| official_temporal_consensus | 15 | 78 | 73.90 | 62.20 | 1.346 | 0.947 |
| official_temporal_consensus | 20 | 78 | 98.18 | 82.58 | 1.344 | 1.259 |
| official_temporal_consensus | 25 | 78 | 122.39 | 102.89 | 1.343 | 1.569 |
| official_temporal_consensus | 30 | 78 | 146.53 | 123.13 | 1.342 | 1.879 |
| official_temporal_consensus | 40 | 78 | 194.49 | 163.29 | 1.340 | 2.493 |
| official_temporal_consensus | 50 | 78 | 242.13 | 203.13 | 1.339 | 3.104 |
| official_temporal_consensus | 75 | 78 | 358.69 | 300.19 | 1.334 | 4.599 |
| official_temporal_consensus | 100 | 78 | 473.02 | 395.02 | 1.331 | 6.064 |
| official_temporal_consensus | 125 | 78 | 585.69 | 488.19 | 1.327 | 7.509 |
| official_temporal_consensus | 150 | 78 | 696.45 | 579.45 | 1.324 | 8.929 |
| official_temporal_consensus | 175 | 78 | 804.70 | 668.20 | 1.321 | 10.317 |
| official_temporal_consensus | 200 | 78 | 911.14 | 755.14 | 1.318 | 11.681 |
| official_dualvenue_l2_veto | 5 | 0 | 0.00 | 0.00 | — | — |
| official_dualvenue_l2_veto | 10 | 0 | 0.00 | 0.00 | — | — |
| official_dualvenue_l2_veto | 15 | 0 | 0.00 | 0.00 | — | — |
| official_dualvenue_l2_veto | 20 | 0 | 0.00 | 0.00 | — | — |
| official_dualvenue_l2_veto | 25 | 0 | 0.00 | 0.00 | — | — |
| official_dualvenue_l2_veto | 30 | 0 | 0.00 | 0.00 | — | — |
| official_dualvenue_l2_veto | 40 | 0 | 0.00 | 0.00 | — | — |
| official_dualvenue_l2_veto | 50 | 0 | 0.00 | 0.00 | — | — |
| official_dualvenue_l2_veto | 75 | 0 | 0.00 | 0.00 | — | — |
| official_dualvenue_l2_veto | 100 | 0 | 0.00 | 0.00 | — | — |
| official_dualvenue_l2_veto | 125 | 0 | 0.00 | 0.00 | — | — |
| official_dualvenue_l2_veto | 150 | 0 | 0.00 | 0.00 | — | — |
| official_dualvenue_l2_veto | 175 | 0 | 0.00 | 0.00 | — | — |
| official_dualvenue_l2_veto | 200 | 0 | 0.00 | 0.00 | — | — |

## Integrity

- Frozen controls and the existing champion collection were not modified.
- Learned admission and residual models used chronological OOF inputs ending before development and holdout markets; temporal snapshots fit only pre-development rows.
- Policies were frozen before the sealed panel and Kraken L2 tail were loaded.
- Settlement sources remained distinct; TWAP and future settlement values were not inference features.
- No database write, table, schema, ingester, source, deployment, or image build occurred.
- Limitation: Executable VWAP evidence ends after August 25; August 26-31 contributes predictive metrics but no economic fills.
- Limitation: Binance spot L2 ends after August 1 and Kraken L2 ends after August 19; neither venue has L2 evidence in the sealed window.
- Limitation: The L2 residual used the official probability fallback in the sealed window, while the L2 confirmation veto abstained; neither result qualifies L2 predictive value.
- Limitation: The August holdout has been observed in prior research and is chronological rather than epistemically fresh.
- Limitation: September 1 data is not present in the current immutable panel and is reported as unavailable rather than fabricated.
- Limitation: Projected PnL assumes recorded ask VWAP was fillable and does not model queue position.
