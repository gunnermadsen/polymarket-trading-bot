# Champion Admission Tournament

Run: `20260902T025934Z`
Qualification: **trained_evaluated_not_deployed**

## Fixed-holdout high-level results

| Candidate | Admission | PnL | Stress PnL | PF | Expectancy | Coverage | W/L | Win rate | Recovery wins/loss | Avg entry | Avg cost | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official_specialist_control | programmatic | -40.80 | -62.05 | 0.918 | -0.096 | 12.54% | 266/159 | 0.63 | 1.822 | 130.6 | 0.624 | 0.1518 |
| bridge_aware_control | programmatic | -85.94 | -118.44 | 0.849 | -0.132 | 19.17% | 493/157 | 0.76 | 3.696 | 91.7 | 0.768 | 0.1518 |
| official_vwap_admission | vwap_bucket | -54.32 | -106.02 | 0.941 | -0.053 | 30.50% | 764/270 | 0.74 | 3.008 | 79.0 | 0.732 | 0.1518 |
| bridge_vwap_admission | vwap_bucket_below_080 | -65.20 | -136.20 | 0.958 | -0.046 | 41.89% | 910/510 | 0.64 | 1.863 | 64.3 | 0.630 | 0.1518 |
| official_loss_severity_veto | loss_severity | -103.07 | -162.17 | 0.905 | -0.087 | 34.87% | 874/308 | 0.74 | 3.137 | 85.5 | 0.739 | 0.1518 |
| bridge_capacity_curve_veto | capacity_curve | -57.67 | -138.37 | 0.965 | -0.036 | 47.61% | 1095/519 | 0.68 | 2.187 | 61.1 | 0.666 | 0.1518 |
| specialist_consensus_admission | consensus | -52.52 | -132.97 | 0.968 | -0.033 | 47.46% | 1090/519 | 0.68 | 2.171 | 62.0 | 0.665 | 0.1517 |

## Results by entry-time bucket

| Candidate | Bucket | PnL | Stress PnL | PF | Expectancy | Coverage | Trades | W/L | Win rate | Recovery wins/loss | Avg entry | Avg cost | Max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official_specialist_control | 60–89 | 39.88 | 35.13 | 1.500 | 0.420 | 2.80% | 95 | 70/25 | 0.74 | 1.866 | 68.7 | 0.632 | 15.298 |
| official_specialist_control | 90–119 | -14.59 | -18.89 | 0.848 | -0.170 | 2.54% | 86 | 58/28 | 0.67 | 2.443 | 97.1 | 0.689 | 24.119 |
| official_specialist_control | 120–149 | -6.55 | -10.50 | 0.931 | -0.083 | 2.33% | 79 | 47/32 | 0.59 | 1.577 | 128.5 | 0.590 | 19.766 |
| official_specialist_control | 150–180 | -75.08 | -91.18 | 0.817 | -0.233 | 9.50% | 322 | 189/133 | 0.59 | 1.740 | 159.6 | 0.613 | 94.184 |
| bridge_aware_control | 60–89 | 37.53 | 18.78 | 1.154 | 0.100 | 11.06% | 375 | 314/61 | 0.84 | 4.459 | 69.2 | 0.802 | 31.672 |
| bridge_aware_control | 90–119 | -56.74 | -77.19 | 0.826 | -0.139 | 12.06% | 409 | 326/83 | 0.80 | 4.756 | 96.5 | 0.809 | 61.610 |
| bridge_aware_control | 120–149 | -10.72 | -12.32 | 0.760 | -0.335 | 0.94% | 32 | 17/15 | 0.53 | 1.491 | 130.2 | 0.577 | 12.725 |
| bridge_aware_control | 150–180 | -52.19 | -59.24 | 0.729 | -0.370 | 4.16% | 141 | 77/64 | 0.55 | 1.651 | 160.5 | 0.599 | 72.867 |
| official_vwap_admission | 60–89 | 49.87 | 9.32 | 1.083 | 0.061 | 23.92% | 811 | 655/156 | 0.81 | 3.876 | 66.6 | 0.779 | 50.093 |
| official_vwap_admission | 90–119 | -10.37 | -22.82 | 0.964 | -0.042 | 7.35% | 249 | 155/94 | 0.62 | 1.710 | 95.8 | 0.610 | 36.853 |
| official_vwap_admission | 120–149 | -15.40 | -21.45 | 0.897 | -0.127 | 3.57% | 121 | 71/50 | 0.59 | 1.584 | 128.3 | 0.591 | 32.127 |
| official_vwap_admission | 150–180 | -64.60 | -74.55 | 0.753 | -0.325 | 5.87% | 199 | 95/104 | 0.48 | 1.213 | 160.6 | 0.522 | 77.572 |
| bridge_vwap_admission | 60–89 | -56.88 | -126.33 | 0.962 | -0.041 | 40.97% | 1389 | 894/495 | 0.64 | 1.877 | 63.0 | 0.632 | 126.907 |
| bridge_vwap_admission | 90–119 | -109.88 | -133.73 | 0.820 | -0.230 | 14.07% | 477 | 283/194 | 0.59 | 1.778 | 97.1 | 0.618 | 157.224 |
| bridge_vwap_admission | 120–149 | -79.36 | -90.56 | 0.743 | -0.354 | 6.61% | 224 | 121/103 | 0.54 | 1.580 | 128.2 | 0.589 | 81.465 |
| bridge_vwap_admission | 150–180 | -49.98 | -60.53 | 0.816 | -0.237 | 6.22% | 211 | 108/103 | 0.51 | 1.285 | 160.0 | 0.538 | 77.264 |
| official_loss_severity_veto | 60–89 | 13.82 | -23.93 | 1.026 | 0.018 | 22.27% | 755 | 620/135 | 0.82 | 4.476 | 67.5 | 0.802 | 54.338 |
| official_loss_severity_veto | 90–119 | -82.14 | -103.29 | 0.845 | -0.194 | 12.48% | 423 | 257/166 | 0.61 | 1.831 | 97.2 | 0.625 | 124.158 |
| official_loss_severity_veto | 120–149 | -22.12 | -27.82 | 0.849 | -0.194 | 3.36% | 114 | 66/48 | 0.58 | 1.620 | 128.9 | 0.596 | 33.364 |
| official_loss_severity_veto | 150–180 | -24.76 | -41.91 | 0.937 | -0.072 | 10.12% | 343 | 221/122 | 0.64 | 1.934 | 160.2 | 0.638 | 67.923 |
| bridge_capacity_curve_veto | 60–89 | -46.26 | -126.56 | 0.971 | -0.029 | 47.37% | 1606 | 1092/514 | 0.68 | 2.187 | 60.8 | 0.666 | 115.458 |
| bridge_capacity_curve_veto | 90–119 | -36.89 | -47.29 | 0.848 | -0.177 | 6.14% | 208 | 136/72 | 0.65 | 2.229 | 97.5 | 0.669 | 58.115 |
| bridge_capacity_curve_veto | 120–149 | -115.88 | -129.48 | 0.690 | -0.426 | 8.02% | 272 | 158/114 | 0.58 | 2.009 | 128.4 | 0.645 | 119.627 |
| bridge_capacity_curve_veto | 150–180 | -61.38 | -68.63 | 0.703 | -0.423 | 4.28% | 145 | 77/68 | 0.53 | 1.612 | 161.1 | 0.594 | 78.018 |
| specialist_consensus_admission | 60–89 | -51.62 | -131.37 | 0.968 | -0.032 | 47.05% | 1595 | 1081/514 | 0.68 | 2.173 | 61.3 | 0.665 | 142.874 |
| specialist_consensus_admission | 90–119 | -38.63 | -52.48 | 0.886 | -0.139 | 8.17% | 277 | 168/109 | 0.61 | 1.741 | 96.2 | 0.613 | 66.472 |
| specialist_consensus_admission | 120–149 | -25.00 | -31.55 | 0.852 | -0.191 | 3.86% | 131 | 76/55 | 0.58 | 1.622 | 129.0 | 0.597 | 41.011 |
| specialist_consensus_admission | 150–180 | -10.40 | -27.10 | 0.972 | -0.031 | 9.85% | 334 | 216/118 | 0.65 | 1.883 | 160.2 | 0.632 | 61.407 |

## Predictive metrics

| Candidate | Rows | Markets | Brier | Log loss | Accuracy | ECE-15 |
|---|---:|---:|---:|---:|---:|---:|
| official_specialist_control | 125430 | 3390 | 0.1518 | 0.4623 | 0.7785 | 0.0144 |
| bridge_aware_control | 125430 | 3390 | 0.1518 | 0.4624 | 0.7787 | 0.0126 |
| official_vwap_admission | 125430 | 3390 | 0.1518 | 0.4623 | 0.7785 | 0.0144 |
| bridge_vwap_admission | 125430 | 3390 | 0.1518 | 0.4624 | 0.7787 | 0.0126 |
| official_loss_severity_veto | 125430 | 3390 | 0.1518 | 0.4623 | 0.7785 | 0.0144 |
| bridge_capacity_curve_veto | 125430 | 3390 | 0.1518 | 0.4624 | 0.7787 | 0.0126 |
| specialist_consensus_admission | 125430 | 3390 | 0.1517 | 0.4622 | 0.7787 | 0.0134 |

## Sealed subperiods

| Candidate | Subperiod | PnL | Stress PnL | PF | Expectancy | Coverage | Trades | W/L |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| official_specialist_control | august–20–25 | -40.80 | -62.05 | 0.918 | -0.096 | 24.59% | 425 | 266/159 |
| official_specialist_control | august–26–31 | 0.00 | 0.00 | — | — | 0.00% | 0 | 0/0 |
| bridge_aware_control | august–20–25 | -85.94 | -118.44 | 0.849 | -0.132 | 37.62% | 650 | 493/157 |
| bridge_aware_control | august–26–31 | 0.00 | 0.00 | — | — | 0.00% | 0 | 0/0 |
| official_vwap_admission | august–20–25 | -54.32 | -106.02 | 0.941 | -0.053 | 59.84% | 1034 | 764/270 |
| official_vwap_admission | august–26–31 | 0.00 | 0.00 | — | — | 0.00% | 0 | 0/0 |
| bridge_vwap_admission | august–20–25 | -65.20 | -136.20 | 0.958 | -0.046 | 82.18% | 1420 | 910/510 |
| bridge_vwap_admission | august–26–31 | 0.00 | 0.00 | — | — | 0.00% | 0 | 0/0 |
| official_loss_severity_veto | august–20–25 | -103.07 | -162.17 | 0.905 | -0.087 | 68.40% | 1182 | 874/308 |
| official_loss_severity_veto | august–26–31 | 0.00 | 0.00 | — | — | 0.00% | 0 | 0/0 |
| bridge_capacity_curve_veto | august–20–25 | -57.67 | -138.37 | 0.965 | -0.036 | 93.40% | 1614 | 1095/519 |
| bridge_capacity_curve_veto | august–26–31 | 0.00 | 0.00 | — | — | 0.00% | 0 | 0/0 |
| specialist_consensus_admission | august–20–25 | -52.52 | -132.97 | 0.968 | -0.033 | 93.11% | 1609 | 1090/519 |
| specialist_consensus_admission | august–26–31 | 0.00 | 0.00 | — | — | 0.00% | 0 | 0/0 |

## VWAP capacity curve

| Candidate | Quantity | Trades | PnL | Stress PnL | PF | Expectancy |
|---|---:|---:|---:|---:|---:|---:|
| official_specialist_control | 5 | 425 | -40.80 | -62.05 | 0.918 | -0.096 |
| official_specialist_control | 10 | 425 | -82.23 | -124.73 | 0.918 | -0.193 |
| official_specialist_control | 15 | 425 | -124.60 | -188.35 | 0.917 | -0.293 |
| official_specialist_control | 20 | 425 | -167.61 | -252.61 | 0.916 | -0.394 |
| official_specialist_control | 25 | 425 | -211.26 | -317.51 | 0.916 | -0.497 |
| official_specialist_control | 30 | 425 | -255.58 | -383.08 | 0.915 | -0.601 |
| official_specialist_control | 40 | 425 | -346.26 | -516.26 | 0.914 | -0.815 |
| official_specialist_control | 50 | 425 | -439.48 | -651.98 | 0.912 | -1.034 |
| official_specialist_control | 75 | 425 | -683.66 | -1002.41 | 0.909 | -1.609 |
| official_specialist_control | 100 | 425 | -942.12 | -1367.12 | 0.906 | -2.217 |
| official_specialist_control | 125 | 425 | -1214.47 | -1745.72 | 0.903 | -2.858 |
| official_specialist_control | 150 | 425 | -1499.13 | -2136.63 | 0.901 | -3.527 |
| official_specialist_control | 175 | 425 | -1797.04 | -2540.79 | 0.898 | -4.228 |
| official_specialist_control | 200 | 425 | -2104.75 | -2954.75 | 0.896 | -4.952 |
| bridge_aware_control | 5 | 650 | -85.94 | -118.44 | 0.849 | -0.132 |
| bridge_aware_control | 10 | 650 | -172.57 | -237.57 | 0.849 | -0.265 |
| bridge_aware_control | 15 | 650 | -260.41 | -357.91 | 0.848 | -0.401 |
| bridge_aware_control | 20 | 650 | -349.19 | -479.19 | 0.847 | -0.537 |
| bridge_aware_control | 25 | 650 | -438.41 | -600.91 | 0.847 | -0.674 |
| bridge_aware_control | 30 | 650 | -528.00 | -723.00 | 0.846 | -0.812 |
| bridge_aware_control | 40 | 650 | -709.32 | -969.32 | 0.845 | -1.091 |
| bridge_aware_control | 50 | 650 | -893.34 | -1218.34 | 0.844 | -1.374 |
| bridge_aware_control | 75 | 650 | -1367.13 | -1854.63 | 0.841 | -2.103 |
| bridge_aware_control | 100 | 650 | -1856.19 | -2506.19 | 0.838 | -2.856 |
| bridge_aware_control | 125 | 650 | -2359.88 | -3172.38 | 0.835 | -3.631 |
| bridge_aware_control | 150 | 650 | -2879.17 | -3854.17 | 0.833 | -4.429 |
| bridge_aware_control | 175 | 650 | -3417.57 | -4555.07 | 0.830 | -5.258 |
| bridge_aware_control | 200 | 650 | -3969.10 | -5269.10 | 0.827 | -6.106 |
| official_vwap_admission | 5 | 1034 | -54.32 | -106.02 | 0.941 | -0.053 |
| official_vwap_admission | 10 | 1034 | -109.86 | -213.26 | 0.940 | -0.106 |
| official_vwap_admission | 15 | 1034 | -166.75 | -321.85 | 0.939 | -0.161 |
| official_vwap_admission | 20 | 1034 | -224.63 | -431.43 | 0.939 | -0.217 |
| official_vwap_admission | 25 | 1034 | -283.24 | -541.74 | 0.938 | -0.274 |
| official_vwap_admission | 30 | 1034 | -342.74 | -652.94 | 0.938 | -0.331 |
| official_vwap_admission | 40 | 1034 | -464.96 | -878.56 | 0.937 | -0.450 |
| official_vwap_admission | 50 | 1034 | -591.12 | -1108.12 | 0.935 | -0.572 |
| official_vwap_admission | 75 | 1034 | -925.10 | -1700.60 | 0.933 | -0.895 |
| official_vwap_admission | 100 | 1034 | -1281.54 | -2315.54 | 0.930 | -1.239 |
| official_vwap_admission | 125 | 1034 | -1662.83 | -2955.33 | 0.928 | -1.608 |
| official_vwap_admission | 150 | 1034 | -2068.23 | -3619.23 | 0.925 | -2.000 |
| official_vwap_admission | 175 | 1034 | -2497.66 | -4307.16 | 0.922 | -2.416 |
| official_vwap_admission | 200 | 1034 | -2948.99 | -5016.99 | 0.920 | -2.852 |
| bridge_vwap_admission | 5 | 1420 | -65.20 | -136.20 | 0.958 | -0.046 |
| bridge_vwap_admission | 10 | 1420 | -132.25 | -274.25 | 0.957 | -0.093 |
| bridge_vwap_admission | 15 | 1420 | -201.23 | -414.23 | 0.957 | -0.142 |
| bridge_vwap_admission | 20 | 1420 | -271.58 | -555.58 | 0.956 | -0.191 |
| bridge_vwap_admission | 25 | 1420 | -343.52 | -698.52 | 0.956 | -0.242 |
| bridge_vwap_admission | 30 | 1420 | -417.38 | -843.38 | 0.955 | -0.294 |
| bridge_vwap_admission | 40 | 1420 | -571.13 | -1139.13 | 0.954 | -0.402 |
| bridge_vwap_admission | 50 | 1420 | -730.93 | -1440.93 | 0.953 | -0.515 |
| bridge_vwap_admission | 75 | 1420 | -1152.74 | -2217.74 | 0.950 | -0.812 |
| bridge_vwap_admission | 100 | 1420 | -1609.03 | -3029.03 | 0.948 | -1.133 |
| bridge_vwap_admission | 125 | 1420 | -2102.97 | -3877.97 | 0.946 | -1.481 |
| bridge_vwap_admission | 150 | 1420 | -2631.00 | -4761.00 | 0.944 | -1.853 |
| bridge_vwap_admission | 175 | 1420 | -3192.50 | -5677.50 | 0.941 | -2.248 |
| bridge_vwap_admission | 200 | 1420 | -3783.96 | -6623.96 | 0.939 | -2.665 |
| official_loss_severity_veto | 5 | 1182 | -103.07 | -162.17 | 0.905 | -0.087 |
| official_loss_severity_veto | 10 | 1182 | -207.02 | -325.22 | 0.904 | -0.175 |
| official_loss_severity_veto | 15 | 1182 | -312.59 | -489.89 | 0.904 | -0.264 |
| official_loss_severity_veto | 20 | 1182 | -419.62 | -656.02 | 0.903 | -0.355 |
| official_loss_severity_veto | 25 | 1182 | -527.84 | -823.34 | 0.902 | -0.447 |
| official_loss_severity_veto | 30 | 1182 | -637.27 | -991.87 | 0.902 | -0.539 |
| official_loss_severity_veto | 40 | 1182 | -859.85 | -1332.65 | 0.901 | -0.727 |
| official_loss_severity_veto | 50 | 1182 | -1086.39 | -1677.39 | 0.900 | -0.919 |
| official_loss_severity_veto | 75 | 1182 | -1672.72 | -2559.22 | 0.897 | -1.415 |
| official_loss_severity_veto | 100 | 1182 | -2284.28 | -3466.28 | 0.895 | -1.933 |
| official_loss_severity_veto | 125 | 1182 | -2923.66 | -4401.16 | 0.892 | -2.473 |
| official_loss_severity_veto | 150 | 1182 | -3587.57 | -5360.57 | 0.890 | -3.035 |
| official_loss_severity_veto | 175 | 1182 | -4279.47 | -6347.97 | 0.887 | -3.621 |
| official_loss_severity_veto | 200 | 1182 | -4998.50 | -7362.50 | 0.885 | -4.229 |
| bridge_capacity_curve_veto | 5 | 1614 | -57.67 | -138.37 | 0.965 | -0.036 |
| bridge_capacity_curve_veto | 10 | 1614 | -117.08 | -278.48 | 0.964 | -0.073 |
| bridge_capacity_curve_veto | 15 | 1614 | -178.59 | -420.69 | 0.963 | -0.111 |
| bridge_capacity_curve_veto | 20 | 1614 | -241.63 | -564.43 | 0.963 | -0.150 |
| bridge_capacity_curve_veto | 25 | 1614 | -306.53 | -710.03 | 0.962 | -0.190 |
| bridge_capacity_curve_veto | 30 | 1614 | -373.28 | -857.48 | 0.962 | -0.231 |
| bridge_capacity_curve_veto | 40 | 1614 | -511.55 | -1157.15 | 0.961 | -0.317 |
| bridge_capacity_curve_veto | 50 | 1614 | -655.55 | -1462.55 | 0.960 | -0.406 |
| bridge_capacity_curve_veto | 75 | 1614 | -1042.11 | -2252.61 | 0.957 | -0.646 |
| bridge_capacity_curve_veto | 100 | 1614 | -1466.72 | -3080.72 | 0.955 | -0.909 |
| bridge_capacity_curve_veto | 125 | 1614 | -1929.15 | -3946.65 | 0.953 | -1.195 |
| bridge_capacity_curve_veto | 150 | 1614 | -2428.30 | -4849.30 | 0.950 | -1.505 |
| bridge_capacity_curve_veto | 175 | 1614 | -2961.84 | -5786.34 | 0.948 | -1.835 |
| bridge_capacity_curve_veto | 200 | 1614 | -3527.17 | -6755.17 | 0.946 | -2.185 |
| specialist_consensus_admission | 5 | 1609 | -52.52 | -132.97 | 0.968 | -0.033 |
| specialist_consensus_admission | 10 | 1609 | -106.93 | -267.83 | 0.967 | -0.066 |
| specialist_consensus_admission | 15 | 1609 | -163.62 | -404.97 | 0.966 | -0.102 |
| specialist_consensus_admission | 20 | 1609 | -221.93 | -543.73 | 0.966 | -0.138 |
| specialist_consensus_admission | 25 | 1609 | -282.03 | -684.28 | 0.965 | -0.175 |
| specialist_consensus_admission | 30 | 1609 | -343.91 | -826.61 | 0.965 | -0.214 |
| specialist_consensus_admission | 40 | 1609 | -473.33 | -1116.93 | 0.963 | -0.294 |
| specialist_consensus_admission | 50 | 1609 | -608.87 | -1413.37 | 0.962 | -0.378 |
| specialist_consensus_admission | 75 | 1609 | -974.66 | -2181.41 | 0.960 | -0.606 |
| specialist_consensus_admission | 100 | 1609 | -1378.53 | -2987.53 | 0.958 | -0.857 |
| specialist_consensus_admission | 125 | 1609 | -1820.47 | -3831.72 | 0.955 | -1.131 |
| specialist_consensus_admission | 150 | 1609 | -2297.84 | -4711.34 | 0.953 | -1.428 |
| specialist_consensus_admission | 175 | 1609 | -2809.23 | -5624.98 | 0.951 | -1.746 |
| specialist_consensus_admission | 200 | 1609 | -3354.97 | -6572.97 | 0.948 | -2.085 |

## Integrity

- The two directional champions and the existing seven-model champion collection were not modified.
- Admission models were trained separately from chronological OOF predictor outputs and packaged with immutable champion references.
- VWAP and Polymarket execution fields were admission-only and never entered the directional predictors.
- No best-later target, database write, table, schema, ingester, source, deployment, or image build was used.
- Limitation: Executable VWAP5-200 evidence ends after August 25; August 26-31 contributes predictive metrics but zero economic trades.
- Limitation: The August holdout has been observed in prior research and is chronological rather than epistemically fresh.
- Limitation: Projected PnL assumes recorded ask VWAP was fillable and does not model queue position.
