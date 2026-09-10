# Strategy-Conditioned Entry-Action Risk Tournament

PnL, PF, recovery, drawdown, W/L and entry time below are properties of the named strategy replay after the risk model is applied—not standalone risk-model returns.

Research selection: `boosted_enter_vs_defer` at 85% target coverage.

## Exact six-strategy compatibility comparison

| Risk model | Coverage target | Trades | W/L | Strategy PnL | Delta PnL | Coverage | PF | Recovery | Max DD | Bad blocked | Good blocked | Alignment | Deferred | Better deferrals | Action regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_action_value | 95% | 2837 | 1985/852 | $-329.66 | $-11.95 | 95.7% | 0.884 | 2.635 | $344.29 | 6.7% | 6.1% | 1.10x | 59 | 28.8% | $3120.06 |
| logistic_action_value | 85% | 2506 | 1730/776 | $-378.35 | $-60.64 | 84.5% | 0.853 | 2.613 | $392.98 | 16.0% | 21.1% | 0.76x | 122 | 30.3% | $3168.75 |
| logistic_action_value | 70% | 2047 | 1371/676 | $-388.98 | $-71.27 | 69.0% | 0.823 | 2.464 | $400.85 | 30.0% | 41.7% | 0.72x | 213 | 36.6% | $3179.38 |
| boosted_enter_vs_defer | 95% | 2848 | 1986/862 | $-334.41 | $-16.70 | 96.1% | 0.883 | 2.608 | $349.04 | 4.7% | 5.3% | 0.90x | 34 | 44.1% | $3124.81 |
| boosted_enter_vs_defer | 85% | 2523 | 1739/784 | $-358.70 | $-40.99 | 85.1% | 0.861 | 2.575 | $372.42 | 15.2% | 19.7% | 0.77x | 102 | 36.3% | $3149.10 |
| boosted_enter_vs_defer | 70% | 2072 | 1386/686 | $-381.30 | $-63.59 | 69.9% | 0.829 | 2.437 | $392.03 | 26.2% | 38.0% | 0.69x | 129 | 38.0% | $3171.69 |
| downside_quantile_action | 95% | 2861 | 2024/837 | $-292.92 | $24.78 | 96.5% | 0.897 | 2.697 | $315.41 | 6.1% | 2.4% | 2.51x | 0 | —% | $3083.32 |
| downside_quantile_action | 85% | 2590 | 1851/739 | $-327.55 | $-9.84 | 87.4% | 0.872 | 2.873 | $343.50 | 18.2% | 12.8% | 1.42x | 52 | 30.8% | $3117.95 |
| downside_quantile_action | 70% | 2121 | 1508/613 | $-313.13 | $4.58 | 71.5% | 0.852 | 2.888 | $327.54 | 34.8% | 33.5% | 1.04x | 161 | 29.2% | $3103.53 |
| regime_conditioned_action | 95% | 2845 | 2006/839 | $-286.31 | $31.39 | 96.0% | 0.898 | 2.662 | $300.94 | 6.8% | 4.0% | 1.73x | 23 | 34.8% | $3076.71 |
| regime_conditioned_action | 85% | 2470 | 1785/685 | $-241.00 | $76.71 | 83.3% | 0.897 | 2.904 | $262.17 | 25.1% | 17.9% | 1.41x | 100 | 14.0% | $3031.40 |
| regime_conditioned_action | 70% | 2024 | 1523/501 | $-105.38 | $212.33 | 68.3% | 0.941 | 3.232 | $132.71 | 47.8% | 30.1% | 1.59x | 110 | 7.3% | $2895.78 |
| regime_strategy_health_action | 95% | 2834 | 1988/846 | $-313.73 | $3.98 | 95.6% | 0.889 | 2.643 | $328.36 | 7.1% | 5.5% | 1.28x | 47 | 23.4% | $3104.13 |
| regime_strategy_health_action | 85% | 2529 | 1759/770 | $-375.47 | $-57.76 | 85.3% | 0.854 | 2.676 | $389.18 | 18.3% | 19.5% | 0.94x | 131 | 32.1% | $3165.86 |
| regime_strategy_health_action | 70% | 2052 | 1406/646 | $-368.86 | $-51.16 | 69.2% | 0.826 | 2.634 | $379.69 | 33.1% | 38.7% | 0.86x | 184 | 29.9% | $3159.26 |
| matched_confidence | 95% | 2859 | 2022/837 | $-323.24 | $-5.53 | 96.4% | 0.886 | 2.725 | $341.71 | 6.1% | 2.5% | 2.42x | 0 | —% | $3113.63 |
| matched_confidence | 85% | 2465 | 1785/680 | $-250.66 | $67.05 | 83.1% | 0.894 | 2.936 | $280.30 | 24.6% | 14.4% | 1.71x | 17 | 17.6% | $3041.06 |
| matched_confidence | 70% | 2101 | 1544/557 | $-200.09 | $117.62 | 70.9% | 0.898 | 3.086 | $226.46 | 40.1% | 27.1% | 1.48x | 55 | 14.5% | $2990.48 |
| matched_edge | 95% | 2835 | 1958/877 | $-336.88 | $-19.17 | 95.6% | 0.884 | 2.525 | $351.51 | 1.6% | 5.6% | 0.28x | 0 | —% | $3127.28 |
| matched_edge | 85% | 2611 | 1764/847 | $-356.09 | $-38.39 | 88.1% | 0.872 | 2.388 | $374.25 | 5.2% | 16.3% | 0.32x | 31 | 64.5% | $3146.49 |
| matched_edge | 70% | 2264 | 1471/793 | $-361.61 | $-43.90 | 76.4% | 0.859 | 2.160 | $383.31 | 12.3% | 31.8% | 0.39x | 69 | 84.1% | $3152.01 |
| frozen_logistic_loss_risk | 95% | 2845 | 2021/824 | $-266.66 | $51.05 | 96.0% | 0.905 | 2.711 | $288.83 | 7.5% | 2.6% | 2.94x | 0 | —% | $3057.06 |
| frozen_logistic_loss_risk | 85% | 2528 | 1842/686 | $-216.10 | $101.61 | 85.3% | 0.910 | 2.950 | $259.04 | 24.6% | 12.1% | 2.03x | 33 | 27.3% | $3006.50 |
| frozen_logistic_loss_risk | 70% | 2086 | 1569/517 | $-174.47 | $143.23 | 70.4% | 0.907 | 3.345 | $202.80 | 47.9% | 28.2% | 1.70x | 133 | 12.0% | $2964.87 |
| frozen_boosted_history_loss_risk | 95% | 2839 | 2021/818 | $-265.33 | $52.38 | 95.8% | 0.905 | 2.730 | $299.91 | 8.3% | 2.7% | 3.13x | 3 | 33.3% | $3055.73 |
| frozen_boosted_history_loss_risk | 85% | 2428 | 1779/649 | $-221.19 | $96.52 | 81.9% | 0.903 | 3.035 | $252.17 | 29.2% | 15.5% | 1.88x | 45 | 17.8% | $3011.59 |
| frozen_boosted_history_loss_risk | 70% | 1928 | 1481/447 | $-37.02 | $280.69 | 65.0% | 0.977 | 3.391 | $106.94 | 54.3% | 32.0% | 1.70x | 110 | 16.4% | $2827.42 |
| no_risk | 100% | 2965 | 2074/891 | $-317.71 | $0.00 | 100.0% | 0.893 | 2.606 | $332.42 | 0.0% | 0.0% | —x | 0 | —% | $3108.11 |

## Primary-band strategy × risk-model matrix

| Risk model | Trading process / strategy model | PnL with risk | Delta | Trades | W/L | Coverage | Avg entry | Avg cost | PF | Recovery | Max DD | Alignment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_action_value | BTC 5m specialist distilled fair value native admission paper | $-22.93 | $-0.33 | 488 | 366/122 | 100.0% | 144.5s | $0.747 | 0.946 | 3.170 | $47.20 | 0.00x |
| logistic_action_value | BTC 5m bridge aware specialist UMR paper | $-71.64 | $-13.44 | 526 | 400/126 | 97.6% | 95.7s | $0.771 | 0.842 | 3.769 | $81.97 | 0.48x |
| logistic_action_value | BTC 5m extended specialist official UMR paper | $-50.52 | $6.22 | 328 | 202/126 | 91.6% | 142.1s | $0.626 | 0.873 | 1.837 | $76.21 | 0.82x |
| logistic_action_value | BTC 5m official high precision loss veto UMR paper | $-89.63 | $7.83 | 369 | 232/137 | 97.9% | 125.3s | $0.657 | 0.804 | 2.105 | $101.82 | 1.01x |
| logistic_action_value | BTC 5m official temporal consensus UMR paper | $-58.16 | $5.40 | 327 | 197/130 | 93.4% | 135.4s | $0.617 | 0.855 | 1.772 | $76.84 | 0.76x |
| logistic_action_value | BTC 5m official VWAP admission UMR paper | $-85.47 | $-66.32 | 468 | 333/135 | 54.9% | 91.2s | $0.731 | 0.806 | 3.061 | $98.53 | 0.82x |
| boosted_enter_vs_defer | BTC 5m specialist distilled fair value native admission paper | $-24.65 | $-2.05 | 487 | 365/122 | 99.8% | 144.6s | $0.747 | 0.942 | 3.175 | $47.20 | 1.50x |
| boosted_enter_vs_defer | BTC 5m bridge aware specialist UMR paper | $-63.09 | $-4.90 | 519 | 395/124 | 96.3% | 96.1s | $0.769 | 0.858 | 3.714 | $72.52 | 0.71x |
| boosted_enter_vs_defer | BTC 5m extended specialist official UMR paper | $-53.62 | $3.12 | 330 | 202/128 | 92.2% | 140.3s | $0.624 | 0.867 | 1.820 | $79.63 | 0.89x |
| boosted_enter_vs_defer | BTC 5m official high precision loss veto UMR paper | $-97.23 | $0.24 | 363 | 226/137 | 96.3% | 125.4s | $0.656 | 0.787 | 2.096 | $109.54 | 0.75x |
| boosted_enter_vs_defer | BTC 5m official temporal consensus UMR paper | $-61.24 | $2.32 | 322 | 193/129 | 92.0% | 134.1s | $0.616 | 0.847 | 1.767 | $77.94 | 0.86x |
| boosted_enter_vs_defer | BTC 5m official VWAP admission UMR paper | $-58.87 | $-39.72 | 502 | 358/144 | 58.9% | 89.8s | $0.719 | 0.872 | 2.852 | $76.61 | 0.80x |
| downside_quantile_action | BTC 5m specialist distilled fair value native admission paper | $-22.18 | $0.42 | 488 | 366/122 | 100.0% | 144.7s | $0.747 | 0.948 | 3.165 | $45.78 | 1.50x |
| downside_quantile_action | BTC 5m bridge aware specialist UMR paper | $-62.13 | $-3.94 | 535 | 408/127 | 99.3% | 93.7s | $0.769 | 0.864 | 3.717 | $72.47 | 0.79x |
| downside_quantile_action | BTC 5m extended specialist official UMR paper | $-58.33 | $-1.59 | 276 | 175/101 | 77.1% | 127.8s | $0.656 | 0.826 | 2.097 | $69.62 | 1.10x |
| downside_quantile_action | BTC 5m official high precision loss veto UMR paper | $-85.95 | $11.52 | 350 | 222/128 | 92.8% | 121.4s | $0.663 | 0.801 | 2.166 | $96.22 | 1.30x |
| downside_quantile_action | BTC 5m official temporal consensus UMR paper | $-71.37 | $-7.81 | 296 | 181/115 | 84.6% | 126.0s | $0.639 | 0.806 | 1.953 | $80.87 | 1.02x |
| downside_quantile_action | BTC 5m official VWAP admission UMR paper | $-27.58 | $-8.43 | 645 | 499/146 | 75.6% | 76.5s | $0.765 | 0.948 | 3.603 | $62.41 | 1.61x |
| regime_conditioned_action | BTC 5m specialist distilled fair value native admission paper | $-28.04 | $-5.43 | 487 | 365/122 | 99.8% | 144.7s | $0.749 | 0.935 | 3.201 | $48.69 | 0.50x |
| regime_conditioned_action | BTC 5m bridge aware specialist UMR paper | $-33.43 | $24.76 | 491 | 386/105 | 91.1% | 90.9s | $0.784 | 0.914 | 4.021 | $41.45 | 1.99x |
| regime_conditioned_action | BTC 5m extended specialist official UMR paper | $-46.27 | $10.47 | 256 | 160/96 | 71.5% | 138.0s | $0.641 | 0.850 | 1.961 | $60.69 | 0.94x |
| regime_conditioned_action | BTC 5m official high precision loss veto UMR paper | $-96.49 | $0.98 | 292 | 183/109 | 77.5% | 125.9s | $0.673 | 0.741 | 2.267 | $96.80 | 0.88x |
| regime_conditioned_action | BTC 5m official temporal consensus UMR paper | $-47.68 | $15.88 | 247 | 151/96 | 70.6% | 132.8s | $0.629 | 0.841 | 1.870 | $56.78 | 0.91x |
| regime_conditioned_action | BTC 5m official VWAP admission UMR paper | $10.90 | $30.05 | 697 | 540/157 | 81.7% | 78.0s | $0.755 | 1.020 | 3.372 | $53.40 | 1.99x |
| regime_strategy_health_action | BTC 5m specialist distilled fair value native admission paper | $-23.62 | $-1.02 | 488 | 366/122 | 100.0% | 144.4s | $0.747 | 0.945 | 3.176 | $46.67 | 0.00x |
| regime_strategy_health_action | BTC 5m bridge aware specialist UMR paper | $-58.83 | $-0.64 | 528 | 403/125 | 98.0% | 95.1s | $0.769 | 0.869 | 3.710 | $68.25 | 1.06x |
| regime_strategy_health_action | BTC 5m extended specialist official UMR paper | $-68.22 | $-11.48 | 333 | 202/131 | 93.0% | 140.6s | $0.627 | 0.835 | 1.847 | $82.26 | 0.76x |
| regime_strategy_health_action | BTC 5m official high precision loss veto UMR paper | $-108.82 | $-11.36 | 340 | 207/133 | 90.2% | 131.3s | $0.652 | 0.753 | 2.066 | $114.98 | 0.76x |
| regime_strategy_health_action | BTC 5m official temporal consensus UMR paper | $-64.34 | $-0.78 | 330 | 198/132 | 94.3% | 134.6s | $0.618 | 0.843 | 1.780 | $76.68 | 0.76x |
| regime_strategy_health_action | BTC 5m official VWAP admission UMR paper | $-51.63 | $-32.48 | 510 | 383/127 | 59.8% | 84.3s | $0.755 | 0.879 | 3.429 | $84.72 | 1.08x |
| matched_confidence | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $0.00 | 488 | 366/122 | 100.0% | 144.4s | $0.747 | 0.947 | 3.168 | $46.67 | —x |
| matched_confidence | BTC 5m bridge aware specialist UMR paper | $-41.77 | $16.42 | 521 | 403/118 | 96.7% | 90.0s | $0.773 | 0.903 | 3.781 | $52.33 | 4.99x |
| matched_confidence | BTC 5m extended specialist official UMR paper | $-58.97 | $-2.23 | 253 | 159/94 | 70.7% | 118.4s | $0.655 | 0.811 | 2.087 | $60.61 | 1.15x |
| matched_confidence | BTC 5m official high precision loss veto UMR paper | $-101.57 | $-4.10 | 318 | 198/120 | 84.4% | 114.9s | $0.666 | 0.750 | 2.199 | $110.66 | 0.98x |
| matched_confidence | BTC 5m official temporal consensus UMR paper | $-51.24 | $12.31 | 235 | 147/88 | 67.1% | 117.7s | $0.649 | 0.822 | 2.033 | $60.59 | 1.24x |
| matched_confidence | BTC 5m official VWAP admission UMR paper | $25.50 | $44.65 | 650 | 512/138 | 76.2% | 74.3s | $0.763 | 1.051 | 3.532 | $60.65 | 1.98x |
| matched_edge | BTC 5m specialist distilled fair value native admission paper | $-16.77 | $5.84 | 480 | 360/120 | 98.4% | 145.3s | $0.745 | 0.960 | 3.125 | $44.02 | 0.27x |
| matched_edge | BTC 5m bridge aware specialist UMR paper | $-67.51 | $-9.32 | 480 | 356/124 | 89.1% | 94.4s | $0.753 | 0.847 | 3.389 | $77.07 | 0.33x |
| matched_edge | BTC 5m extended specialist official UMR paper | $-56.74 | $0.00 | 358 | 220/138 | 100.0% | 130.6s | $0.625 | 0.870 | 1.833 | $78.23 | —x |
| matched_edge | BTC 5m official high precision loss veto UMR paper | $-95.94 | $1.52 | 374 | 233/141 | 99.2% | 121.3s | $0.654 | 0.795 | 2.077 | $109.62 | 0.83x |
| matched_edge | BTC 5m official temporal consensus UMR paper | $-63.56 | $0.00 | 350 | 211/139 | 100.0% | 126.3s | $0.618 | 0.853 | 1.780 | $86.61 | —x |
| matched_edge | BTC 5m official VWAP admission UMR paper | $-55.57 | $-36.42 | 569 | 384/185 | 66.7% | 85.9s | $0.675 | 0.906 | 2.291 | $75.19 | 0.42x |
| frozen_logistic_loss_risk | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $0.00 | 488 | 366/122 | 100.0% | 144.4s | $0.747 | 0.947 | 3.168 | $46.67 | —x |
| frozen_logistic_loss_risk | BTC 5m bridge aware specialist UMR paper | $-35.44 | $22.75 | 519 | 403/116 | 96.3% | 89.8s | $0.774 | 0.917 | 3.789 | $45.99 | 5.90x |
| frozen_logistic_loss_risk | BTC 5m extended specialist official UMR paper | $-60.04 | $-3.30 | 263 | 165/98 | 73.5% | 121.8s | $0.653 | 0.815 | 2.066 | $62.29 | 1.12x |
| frozen_logistic_loss_risk | BTC 5m official high precision loss veto UMR paper | $-94.30 | $3.16 | 321 | 202/119 | 85.1% | 115.8s | $0.668 | 0.767 | 2.213 | $96.26 | 1.09x |
| frozen_logistic_loss_risk | BTC 5m official temporal consensus UMR paper | $-37.43 | $26.12 | 233 | 148/85 | 66.6% | 119.6s | $0.647 | 0.866 | 2.011 | $46.33 | 1.35x |
| frozen_logistic_loss_risk | BTC 5m official VWAP admission UMR paper | $33.72 | $52.87 | 704 | 558/146 | 82.5% | 74.4s | $0.766 | 1.062 | 3.599 | $70.31 | 2.90x |
| frozen_boosted_history_loss_risk | BTC 5m specialist distilled fair value native admission paper | $-13.17 | $9.43 | 487 | 367/120 | 99.8% | 144.7s | $0.747 | 0.969 | 3.158 | $41.92 | 6.00x |
| frozen_boosted_history_loss_risk | BTC 5m bridge aware specialist UMR paper | $-19.34 | $38.86 | 505 | 398/107 | 93.7% | 87.8s | $0.779 | 0.952 | 3.909 | $35.68 | 5.83x |
| frozen_boosted_history_loss_risk | BTC 5m extended specialist official UMR paper | $-64.02 | $-7.28 | 210 | 131/79 | 58.7% | 118.4s | $0.665 | 0.759 | 2.186 | $64.02 | 1.08x |
| frozen_boosted_history_loss_risk | BTC 5m official high precision loss veto UMR paper | $-82.26 | $15.20 | 284 | 181/103 | 75.3% | 114.6s | $0.675 | 0.768 | 2.288 | $84.27 | 1.10x |
| frozen_boosted_history_loss_risk | BTC 5m official temporal consensus UMR paper | $-59.62 | $3.94 | 226 | 140/86 | 64.6% | 118.1s | $0.652 | 0.789 | 2.064 | $65.09 | 1.15x |
| frozen_boosted_history_loss_risk | BTC 5m official VWAP admission UMR paper | $17.22 | $36.37 | 716 | 562/154 | 83.9% | 73.5s | $0.763 | 1.030 | 3.541 | $70.49 | 2.74x |
| no_risk | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $0.00 | 488 | 366/122 | 100.0% | 144.4s | $0.747 | 0.947 | 3.168 | $46.67 | —x |
| no_risk | BTC 5m bridge aware specialist UMR paper | $-58.19 | $0.00 | 539 | 410/129 | 100.0% | 92.5s | $0.765 | 0.874 | 3.635 | $68.53 | —x |
| no_risk | BTC 5m extended specialist official UMR paper | $-56.74 | $0.00 | 358 | 220/138 | 100.0% | 130.6s | $0.625 | 0.870 | 1.833 | $78.23 | —x |
| no_risk | BTC 5m official high precision loss veto UMR paper | $-97.46 | $0.00 | 377 | 235/142 | 100.0% | 121.1s | $0.655 | 0.794 | 2.085 | $108.57 | —x |
| no_risk | BTC 5m official temporal consensus UMR paper | $-63.56 | $0.00 | 350 | 211/139 | 100.0% | 126.3s | $0.618 | 0.853 | 1.780 | $86.61 | —x |
| no_risk | BTC 5m official VWAP admission UMR paper | $-19.15 | $0.00 | 853 | 632/221 | 100.0% | 79.7s | $0.728 | 0.974 | 2.936 | $73.35 | —x |

## Selected risk model through each trading strategy

| Trading process / strategy model | Base PnL | PnL with risk | Delta | Trades | W/L | Coverage | Avg entry | Avg cost | PF | Recovery | Max DD | Bad blocked | Good blocked | Alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC 5m specialist distilled fair value native admission paper | $-22.60 | $-24.65 | $-2.05 | 487 | 365/122 | 99.8% | 144.6s | $0.747 | 0.942 | 3.175 | $47.20 | 0.8% | 0.5% | 1.50x |
| BTC 5m bridge aware specialist UMR paper | $-58.19 | $-63.09 | $-4.90 | 519 | 395/124 | 96.3% | 96.1s | $0.769 | 0.858 | 3.714 | $72.52 | 7.8% | 11.0% | 0.71x |
| BTC 5m extended specialist official UMR paper | $-56.74 | $-53.62 | $3.12 | 330 | 202/128 | 92.2% | 140.3s | $0.624 | 0.867 | 1.820 | $79.63 | 14.5% | 16.4% | 0.89x |
| BTC 5m official high precision loss veto UMR paper | $-97.46 | $-97.23 | $0.24 | 363 | 226/137 | 96.3% | 125.4s | $0.656 | 0.787 | 2.096 | $109.54 | 7.0% | 9.4% | 0.75x |
| BTC 5m official temporal consensus UMR paper | $-63.56 | $-61.24 | $2.32 | 322 | 193/129 | 92.0% | 134.1s | $0.616 | 0.847 | 1.767 | $77.94 | 12.2% | 14.2% | 0.86x |
| BTC 5m official VWAP admission UMR paper | $-19.15 | $-58.87 | $-39.72 | 502 | 358/144 | 58.9% | 89.8s | $0.719 | 0.872 | 2.852 | $76.61 | 34.8% | 43.4% | 0.80x |

## Time buckets and sides

The complete risk-model × coverage × strategy × natural time-bucket × side matrix is in `ledgers/slice-matrix.parquet` and `metrics.json`. Empty and sub-30-trade cells are retained.

## Integrity and limitations

- The research winner was selected only from chronological historical walk-forward folds; the six-strategy compatibility results did not choose or replace it.
- The August 21-25 exact six-strategy cohort was viewed in the preceding tournament, so this run correctly reports it as compatibility evidence rather than a pristine sealed qualification cohort.
- Exact candidate ledgers for all six current strategies are unavailable after August 25; no strategy decisions were synthesized from generic execution data.
- The SSD rich panel's causal numeric and boolean dimensions were eligible after explicit outcome, target, timestamp, and identity exclusions; individual contestants use the feature subsets recorded in metrics.json.
- The archived common candidate construction begins at second 60, so 15-59 evidence is unavailable and retained as empty/insufficient slices.
- This is a research-only training run. No runtime model, trading process, database, ingester, data source, schema, or image was changed.
