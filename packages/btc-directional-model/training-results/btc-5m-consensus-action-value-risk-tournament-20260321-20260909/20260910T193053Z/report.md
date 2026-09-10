# Strategy-Consensus Action-Value Risk Tournament

PnL and trading metrics describe each strategy replay after the named risk policy; the risk model has no standalone trading return.

## Historical bucket selections

- 60_89: `conformal_action_value`
- 90_119: `consensus_disagreement_value`
- 120_149: `conformal_action_value`
- 150_179: `pairwise_optimal_stopping`
- 180_240: `conformal_action_value`

## Exact six-strategy compatibility at primary coverage

| Risk policy | Risk PnL | Delta | Stress PnL | Trades | W/L | Coverage | PF | Recovery | Max DD | Bad blocked | Good blocked | Alignment | Wait capture | Catastrophic capture | Portable strategies |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expected_action_value | $-169.81 | $147.90 | $-291.56 | 2435 | 1735/700 | 82.1% | 0.928 | 2.672 | $194.72 | 22.7% | 18.7% | 1.21x | 50.0% | 17.9% | 6/6 |
| distributional_action_value | $-205.14 | $112.56 | $-320.64 | 2310 | 1662/648 | 77.9% | 0.908 | 2.824 | $258.76 | 35.8% | 22.5% | 1.59x | 35.9% | 7.4% | 5/6 |
| consensus_disagreement_value | $-243.27 | $74.44 | $-363.87 | 2412 | 1660/752 | 81.3% | 0.902 | 2.448 | $262.07 | 17.5% | 22.5% | 0.78x | 55.7% | 37.9% | 4/6 |
| pairwise_optimal_stopping | $-378.75 | $-61.04 | $-511.50 | 2655 | 1841/814 | 89.5% | 0.860 | 2.629 | $390.71 | 12.6% | 16.3% | 0.77x | 32.1% | 14.7% | 0/6 |
| conformal_action_value | $-267.60 | $50.11 | $-390.60 | 2460 | 1698/762 | 83.0% | 0.895 | 2.491 | $285.90 | 17.2% | 22.6% | 0.76x | 58.1% | 33.7% | 5/6 |
| frozen_bucket_champion | $-329.98 | $-12.28 | $-460.58 | 2612 | 1821/791 | 88.1% | 0.874 | 2.634 | $343.62 | 15.7% | 18.4% | 0.86x | 39.9% | 14.7% | 4/6 |
| frozen_winner_preserving_loss | $-376.40 | $-58.70 | $-503.40 | 2540 | 1747/793 | 85.7% | 0.856 | 2.572 | $388.07 | 14.6% | 21.4% | 0.68x | 33.8% | 23.2% | 3/6 |
| matched_edge | $-262.35 | $55.36 | $-392.70 | 2607 | 1790/817 | 87.9% | 0.902 | 2.429 | $288.60 | 10.5% | 17.2% | 0.61x | 71.0% | 36.8% | 5/6 |
| frozen_logistic_loss_risk | $-243.67 | $74.03 | $-368.92 | 2505 | 1809/696 | 84.5% | 0.899 | 2.892 | $281.06 | 26.5% | 15.3% | 1.73x | 33.3% | 0.0% | 3/6 |
| frozen_boosted_history_loss_risk | $-193.62 | $124.08 | $-314.67 | 2421 | 1761/660 | 81.7% | 0.915 | 2.915 | $235.65 | 32.1% | 18.0% | 1.78x | 38.3% | 2.1% | 6/6 |
| no_risk | $-317.71 | $0.00 | $-465.96 | 2965 | 2074/891 | 100.0% | 0.893 | 2.606 | $332.42 | 0.0% | 0.0% | —x | —% | 0.0% | 0/6 |

## Strategy × risk policy primary-band matrix

| Risk policy | Trading process / strategy model | Base PnL | Risk PnL | Delta | Stress PnL | Trades | W/L | Coverage | Avg entry | Avg cost | PF | Recovery | Max DD | Alignment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expected_action_value | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $-18.23 | $4.37 | $-41.53 | 466 | 350/116 | 95.5% | 146.2s | $0.747 | 0.955 | 3.160 | $45.01 | 0.60x |
| expected_action_value | BTC 5m bridge aware specialist UMR paper | $-58.19 | $-27.61 | $30.58 | $-49.26 | 433 | 335/98 | 80.3% | 87.0s | $0.770 | 0.922 | 3.708 | $43.42 | 1.31x |
| expected_action_value | BTC 5m extended specialist official UMR paper | $-56.74 | $-37.08 | $19.66 | $-50.43 | 267 | 167/100 | 74.6% | 127.1s | $0.633 | 0.883 | 1.891 | $66.50 | 1.06x |
| expected_action_value | BTC 5m official high precision loss veto UMR paper | $-97.46 | $-50.54 | $46.93 | $-65.04 | 290 | 186/104 | 76.9% | 118.8s | $0.656 | 0.855 | 2.093 | $76.09 | 1.11x |
| expected_action_value | BTC 5m official temporal consensus UMR paper | $-63.56 | $-35.64 | $27.91 | $-49.39 | 275 | 170/105 | 78.6% | 122.9s | $0.623 | 0.890 | 1.818 | $50.46 | 1.28x |
| expected_action_value | BTC 5m official VWAP admission UMR paper | $-19.15 | $-0.71 | $18.44 | $-35.91 | 704 | 527/177 | 82.5% | 78.2s | $0.731 | 0.999 | 2.981 | $47.21 | 1.20x |
| distributional_action_value | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $-16.57 | $6.03 | $-40.82 | 485 | 366/119 | 99.4% | 144.4s | $0.749 | 0.961 | 3.202 | $45.32 | 2.40x |
| distributional_action_value | BTC 5m bridge aware specialist UMR paper | $-58.19 | $-46.65 | $11.54 | $-70.95 | 486 | 369/117 | 90.2% | 90.4s | $0.761 | 0.890 | 3.545 | $56.99 | 1.08x |
| distributional_action_value | BTC 5m extended specialist official UMR paper | $-56.74 | $-61.15 | $-4.42 | $-74.05 | 258 | 160/98 | 72.1% | 126.5s | $0.647 | 0.807 | 2.022 | $71.36 | 1.25x |
| distributional_action_value | BTC 5m official high precision loss veto UMR paper | $-97.46 | $-70.59 | $26.87 | $-85.04 | 289 | 181/108 | 76.7% | 123.6s | $0.655 | 0.803 | 2.088 | $82.11 | 1.51x |
| distributional_action_value | BTC 5m official temporal consensus UMR paper | $-63.56 | $-61.32 | $2.23 | $-73.02 | 234 | 142/92 | 66.9% | 123.4s | $0.639 | 0.789 | 1.955 | $78.38 | 1.26x |
| distributional_action_value | BTC 5m official VWAP admission UMR paper | $-19.15 | $51.15 | $70.30 | $23.25 | 558 | 444/114 | 65.4% | 73.4s | $0.760 | 1.121 | 3.475 | $51.72 | 1.63x |
| consensus_disagreement_value | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $-24.96 | $-2.35 | $-45.66 | 414 | 304/110 | 84.8% | 146.5s | $0.734 | 0.934 | 2.958 | $48.97 | 0.64x |
| consensus_disagreement_value | BTC 5m bridge aware specialist UMR paper | $-58.19 | $-53.42 | $4.78 | $-76.32 | 458 | 341/117 | 85.0% | 93.1s | $0.751 | 0.870 | 3.350 | $67.06 | 0.60x |
| consensus_disagreement_value | BTC 5m extended specialist official UMR paper | $-56.74 | $-10.32 | $46.42 | $-24.97 | 293 | 186/107 | 81.8% | 126.3s | $0.621 | 0.969 | 1.794 | $43.74 | 1.42x |
| consensus_disagreement_value | BTC 5m official high precision loss veto UMR paper | $-97.46 | $-80.11 | $17.36 | $-95.06 | 299 | 184/115 | 79.3% | 119.9s | $0.649 | 0.789 | 2.027 | $89.44 | 0.89x |
| consensus_disagreement_value | BTC 5m official temporal consensus UMR paper | $-63.56 | $-30.37 | $33.19 | $-44.52 | 283 | 174/109 | 80.9% | 123.0s | $0.615 | 0.909 | 1.755 | $56.50 | 1.25x |
| consensus_disagreement_value | BTC 5m official VWAP admission UMR paper | $-19.15 | $-44.10 | $-24.95 | $-77.35 | 665 | 471/194 | 78.0% | 82.1s | $0.703 | 0.930 | 2.609 | $78.44 | 0.48x |
| pairwise_optimal_stopping | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $-30.50 | $-7.90 | $-51.15 | 413 | 307/106 | 84.6% | 152.1s | $0.746 | 0.917 | 3.159 | $44.51 | 0.76x |
| pairwise_optimal_stopping | BTC 5m bridge aware specialist UMR paper | $-58.19 | $-61.30 | $-3.11 | $-87.85 | 531 | 404/127 | 98.5% | 91.7s | $0.767 | 0.866 | 3.674 | $71.64 | 1.82x |
| pairwise_optimal_stopping | BTC 5m extended specialist official UMR paper | $-56.74 | $-77.35 | $-20.61 | $-92.90 | 311 | 186/125 | 86.9% | 128.9s | $0.627 | 0.803 | 1.852 | $93.67 | 0.69x |
| pairwise_optimal_stopping | BTC 5m official high precision loss veto UMR paper | $-97.46 | $-107.09 | $-9.63 | $-125.04 | 359 | 221/138 | 95.2% | 120.2s | $0.655 | 0.767 | 2.089 | $114.61 | 0.62x |
| pairwise_optimal_stopping | BTC 5m official temporal consensus UMR paper | $-63.56 | $-79.32 | $-15.76 | $-95.32 | 320 | 189/131 | 91.4% | 125.8s | $0.619 | 0.804 | 1.794 | $91.54 | 0.73x |
| pairwise_optimal_stopping | BTC 5m official VWAP admission UMR paper | $-19.15 | $-23.18 | $-4.03 | $-59.23 | 721 | 534/187 | 84.5% | 79.9s | $0.729 | 0.963 | 2.965 | $69.80 | 0.99x |
| conformal_action_value | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $-14.46 | $8.14 | $-31.76 | 346 | 249/97 | 70.9% | 160.1s | $0.715 | 0.955 | 2.687 | $37.95 | 0.66x |
| conformal_action_value | BTC 5m bridge aware specialist UMR paper | $-58.19 | $-45.41 | $12.78 | $-69.91 | 490 | 375/115 | 90.9% | 89.9s | $0.767 | 0.891 | 3.659 | $54.06 | 1.34x |
| conformal_action_value | BTC 5m extended specialist official UMR paper | $-56.74 | $-39.58 | $17.16 | $-54.93 | 307 | 193/114 | 85.8% | 126.5s | $0.634 | 0.892 | 1.898 | $57.10 | 1.48x |
| conformal_action_value | BTC 5m official high precision loss veto UMR paper | $-97.46 | $-100.16 | $-2.70 | $-117.41 | 345 | 214/131 | 91.5% | 118.1s | $0.658 | 0.771 | 2.118 | $108.73 | 0.90x |
| conformal_action_value | BTC 5m official temporal consensus UMR paper | $-63.56 | $-59.26 | $4.30 | $-74.76 | 310 | 188/122 | 88.6% | 122.6s | $0.624 | 0.845 | 1.823 | $81.37 | 1.03x |
| conformal_action_value | BTC 5m official VWAP admission UMR paper | $-19.15 | $-8.72 | $10.43 | $-41.82 | 662 | 479/183 | 77.6% | 80.1s | $0.708 | 0.986 | 2.655 | $54.12 | 0.71x |
| frozen_bucket_champion | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $-6.80 | $15.80 | $-28.20 | 428 | 321/107 | 87.7% | 151.2s | $0.741 | 0.981 | 3.057 | $43.59 | 0.68x |
| frozen_bucket_champion | BTC 5m bridge aware specialist UMR paper | $-58.19 | $-49.63 | $8.56 | $-75.98 | 527 | 403/124 | 97.8% | 92.1s | $0.767 | 0.888 | 3.658 | $58.42 | 1.73x |
| frozen_bucket_champion | BTC 5m extended specialist official UMR paper | $-56.74 | $-56.47 | $0.27 | $-72.47 | 320 | 196/124 | 89.4% | 128.5s | $0.627 | 0.855 | 1.848 | $78.01 | 1.00x |
| frozen_bucket_champion | BTC 5m official high precision loss veto UMR paper | $-97.46 | $-92.11 | $5.35 | $-109.01 | 338 | 210/128 | 89.7% | 120.1s | $0.655 | 0.784 | 2.093 | $102.58 | 0.75x |
| frozen_bucket_champion | BTC 5m official temporal consensus UMR paper | $-63.56 | $-71.58 | $-8.02 | $-85.98 | 288 | 170/118 | 82.3% | 126.4s | $0.619 | 0.803 | 1.794 | $86.97 | 1.09x |
| frozen_bucket_champion | BTC 5m official VWAP admission UMR paper | $-19.15 | $-53.39 | $-34.24 | $-88.94 | 711 | 521/190 | 83.4% | 80.1s | $0.730 | 0.916 | 2.995 | $81.69 | 0.80x |
| frozen_winner_preserving_loss | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $-10.48 | $12.12 | $-31.08 | 412 | 308/104 | 84.4% | 152.6s | $0.740 | 0.970 | 3.052 | $46.12 | 0.79x |
| frozen_winner_preserving_loss | BTC 5m bridge aware specialist UMR paper | $-58.19 | $-55.91 | $2.28 | $-81.76 | 517 | 394/123 | 95.9% | 92.8s | $0.767 | 0.873 | 3.668 | $64.56 | 0.79x |
| frozen_winner_preserving_loss | BTC 5m extended specialist official UMR paper | $-56.74 | $-62.74 | $-6.01 | $-78.59 | 317 | 193/124 | 88.5% | 129.9s | $0.628 | 0.839 | 1.855 | $79.48 | 0.92x |
| frozen_winner_preserving_loss | BTC 5m official high precision loss veto UMR paper | $-97.46 | $-91.17 | $6.30 | $-108.27 | 342 | 213/129 | 90.7% | 120.7s | $0.656 | 0.788 | 2.095 | $96.51 | 0.83x |
| frozen_winner_preserving_loss | BTC 5m official temporal consensus UMR paper | $-63.56 | $-73.60 | $-10.04 | $-89.55 | 319 | 190/129 | 91.1% | 126.1s | $0.621 | 0.816 | 1.805 | $83.84 | 0.88x |
| frozen_winner_preserving_loss | BTC 5m official VWAP admission UMR paper | $-19.15 | $-82.50 | $-63.35 | $-114.15 | 633 | 449/184 | 74.2% | 82.1s | $0.718 | 0.864 | 2.823 | $104.94 | 0.58x |
| matched_edge | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $2.55 | $25.15 | $-17.60 | 403 | 298/105 | 82.6% | 147.8s | $0.725 | 1.007 | 2.818 | $41.33 | 0.71x |
| matched_edge | BTC 5m bridge aware specialist UMR paper | $-58.19 | $-53.91 | $4.28 | $-79.11 | 504 | 380/124 | 93.5% | 91.9s | $0.758 | 0.878 | 3.491 | $64.25 | 0.64x |
| matched_edge | BTC 5m extended specialist official UMR paper | $-56.74 | $-37.56 | $19.18 | $-53.01 | 309 | 190/119 | 86.3% | 126.3s | $0.618 | 0.898 | 1.778 | $69.30 | 1.01x |
| matched_edge | BTC 5m official high precision loss veto UMR paper | $-97.46 | $-79.02 | $18.44 | $-96.72 | 354 | 222/132 | 93.9% | 120.2s | $0.651 | 0.819 | 2.052 | $99.49 | 1.30x |
| matched_edge | BTC 5m official temporal consensus UMR paper | $-63.56 | $-61.08 | $2.48 | $-78.18 | 342 | 206/136 | 97.7% | 126.6s | $0.617 | 0.855 | 1.772 | $85.79 | 1.01x |
| matched_edge | BTC 5m official VWAP admission UMR paper | $-19.15 | $-33.33 | $-14.18 | $-68.08 | 695 | 494/201 | 81.5% | 81.7s | $0.702 | 0.949 | 2.589 | $67.85 | 0.41x |
| frozen_logistic_loss_risk | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $-25.71 | $-3.11 | $-50.06 | 487 | 365/122 | 99.8% | 144.2s | $0.748 | 0.940 | 3.184 | $49.43 | 0.00x |
| frozen_logistic_loss_risk | BTC 5m bridge aware specialist UMR paper | $-58.19 | $-43.40 | $14.79 | $-69.85 | 529 | 407/122 | 98.1% | 90.9s | $0.769 | 0.902 | 3.699 | $53.74 | 7.42x |
| frozen_logistic_loss_risk | BTC 5m extended specialist official UMR paper | $-56.74 | $-59.68 | $-2.94 | $-74.33 | 293 | 183/110 | 81.8% | 128.4s | $0.645 | 0.834 | 1.996 | $69.99 | 1.13x |
| frozen_logistic_loss_risk | BTC 5m official high precision loss veto UMR paper | $-97.46 | $-104.51 | $-7.04 | $-120.46 | 319 | 196/123 | 84.6% | 124.2s | $0.660 | 0.748 | 2.130 | $109.96 | 1.01x |
| frozen_logistic_loss_risk | BTC 5m official temporal consensus UMR paper | $-63.56 | $-46.57 | $16.99 | $-59.72 | 263 | 164/99 | 75.1% | 125.1s | $0.638 | 0.853 | 1.941 | $63.70 | 1.35x |
| frozen_logistic_loss_risk | BTC 5m official VWAP admission UMR paper | $-19.15 | $36.19 | $55.34 | $5.49 | 614 | 494/120 | 72.0% | 75.0s | $0.776 | 1.082 | 3.806 | $50.63 | 2.09x |
| frozen_boosted_history_loss_risk | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $-12.82 | $9.78 | $-37.17 | 487 | 367/120 | 99.8% | 144.9s | $0.747 | 0.969 | 3.155 | $42.43 | 6.00x |
| frozen_boosted_history_loss_risk | BTC 5m bridge aware specialist UMR paper | $-58.19 | $-43.94 | $14.25 | $-69.59 | 513 | 396/117 | 95.2% | 90.8s | $0.772 | 0.896 | 3.775 | $53.90 | 2.65x |
| frozen_boosted_history_loss_risk | BTC 5m extended specialist official UMR paper | $-56.74 | $-40.72 | $16.02 | $-53.52 | 256 | 163/93 | 71.5% | 132.3s | $0.648 | 0.867 | 2.022 | $61.17 | 1.22x |
| frozen_boosted_history_loss_risk | BTC 5m official high precision loss veto UMR paper | $-97.46 | $-85.11 | $12.35 | $-99.96 | 297 | 183/114 | 78.8% | 126.7s | $0.653 | 0.776 | 2.070 | $87.00 | 1.10x |
| frozen_boosted_history_loss_risk | BTC 5m official temporal consensus UMR paper | $-63.56 | $-37.63 | $25.93 | $-50.58 | 259 | 163/96 | 74.0% | 126.4s | $0.638 | 0.879 | 1.933 | $56.81 | 1.45x |
| frozen_boosted_history_loss_risk | BTC 5m official VWAP admission UMR paper | $-19.15 | $26.60 | $45.75 | $-3.85 | 609 | 489/120 | 71.4% | 74.2s | $0.778 | 1.059 | 3.847 | $68.53 | 2.02x |
| no_risk | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $-22.60 | $0.00 | $-47.00 | 488 | 366/122 | 100.0% | 144.4s | $0.747 | 0.947 | 3.168 | $46.67 | —x |
| no_risk | BTC 5m bridge aware specialist UMR paper | $-58.19 | $-58.19 | $0.00 | $-85.14 | 539 | 410/129 | 100.0% | 92.5s | $0.765 | 0.874 | 3.635 | $68.53 | —x |
| no_risk | BTC 5m extended specialist official UMR paper | $-56.74 | $-56.74 | $0.00 | $-74.64 | 358 | 220/138 | 100.0% | 130.6s | $0.625 | 0.870 | 1.833 | $78.23 | —x |
| no_risk | BTC 5m official high precision loss veto UMR paper | $-97.46 | $-97.46 | $0.00 | $-116.31 | 377 | 235/142 | 100.0% | 121.1s | $0.655 | 0.794 | 2.085 | $108.57 | —x |
| no_risk | BTC 5m official temporal consensus UMR paper | $-63.56 | $-63.56 | $0.00 | $-81.06 | 350 | 211/139 | 100.0% | 126.3s | $0.618 | 0.853 | 1.780 | $86.61 | —x |
| no_risk | BTC 5m official VWAP admission UMR paper | $-19.15 | $-19.15 | $0.00 | $-61.80 | 853 | 632/221 | 100.0% | 79.7s | $0.728 | 0.974 | 2.936 | $73.35 | —x |

## Granular champions

Champion identity is risk policy + strategy model + time bucket + side + coverage policy. Complete supported and sparse cells are retained in `ledgers/slice-matrix.parquet` and `metrics.json`.

| Bucket | Champion measure | Risk policy | Trading process / strategy model | Side | Target | Base PnL | Risk PnL | Delta | Stress PnL | Trades | W/L | Coverage | PF | Recovery | Max DD | Alignment |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60_89 | pnl | frozen_boosted_history_loss_risk | BTC 5m official VWAP admission UMR paper | ALL | 95% | $71.88 | $85.66 | $13.78 | $54.46 | 624 | 515/109 | 94.8% | 1.205 | 3.922 | $44.29 | 2.72x |
| 60_89 | profit factor | pairwise_optimal_stopping | BTC 5m official temporal consensus UMR paper | ALL | 70% | $17.04 | $30.95 | $13.91 | $28.90 | 41 | 32/9 | 64.1% | 2.148 | 1.655 | $14.88 | 1.82x |
| 90_119 | pnl | frozen_logistic_loss_risk | BTC 5m specialist distilled fair value native admission paper | ALL | 70% | $25.54 | $31.90 | $6.36 | $25.40 | 130 | 112/18 | 91.5% | 1.444 | 4.308 | $9.76 | 2.62x |
| 90_119 | risk contribution | distributional_action_value | BTC 5m official high precision loss veto UMR paper | ALL | 85% | $-58.82 | $0.72 | $59.55 | $-5.48 | 124 | 90/34 | 66.0% | 1.006 | 2.632 | $29.05 | 1.78x |
| 120_149 | pnl | conformal_action_value | BTC 5m extended specialist official UMR paper | ALL | 85% | $-5.52 | $11.39 | $16.91 | $8.54 | 57 | 37/20 | 82.6% | 1.193 | 1.551 | $17.48 | 2.93x |
| 120_149 | profit factor | conformal_action_value | BTC 5m extended specialist official UMR paper | ALL | 70% | $-5.52 | $8.88 | $14.39 | $6.93 | 39 | 25/14 | 56.5% | 1.224 | 1.459 | $12.55 | 1.28x |
| 120_149 | risk contribution | frozen_logistic_loss_risk | BTC 5m official temporal consensus UMR paper | ALL | 70% | $-40.77 | $-2.06 | $38.71 | $-4.51 | 49 | 30/19 | 32.7% | 0.963 | 1.639 | $16.12 | 1.11x |
| 150_179 | pnl | matched_edge | BTC 5m specialist distilled fair value native admission paper | ALL | 85% | $-6.99 | $6.89 | $13.88 | $-1.81 | 174 | 117/57 | 69.9% | 1.039 | 1.976 | $31.28 | 0.63x |
| 150_179 | risk contribution | expected_action_value | BTC 5m extended specialist official UMR paper | ALL | 70% | $-83.07 | $-3.39 | $79.68 | $-7.99 | 92 | 56/36 | 39.7% | 0.967 | 1.608 | $19.91 | 1.14x |
| 180_240 | pnl | consensus_disagreement_value | BTC 5m specialist distilled fair value native admission paper | ALL | 70% | $4.35 | $16.55 | $12.20 | $5.35 | 224 | 170/54 | 64.4% | 1.092 | 2.883 | $24.57 | 0.77x |
| 180_240 | profit factor | conformal_action_value | BTC 5m extended specialist official UMR paper | ALL | 85% | $2.33 | $4.73 | $2.40 | $3.23 | 30 | 21/9 | 83.3% | 1.153 | 2.023 | $10.90 | 2.00x |

## Data integrity and limitations

- The August 21-25 exact six-strategy cohort was already viewed and remains compatibility evidence, not a sealed qualification cohort.
- Historical consensus features use the available frozen champion outputs at each timestamp; exact six-strategy identity is used only for compatibility evaluation and never as a predictive feature.
- September 7-9 remains audit-only because one process admitted zero candidates and the five newer processes admitted only 65 combined buys.
- Rejected live strategy decisions were not converted into synthetic admitted candidates.
- The full established April-August causal SSD-enriched panel was used; the common historical construction begins at second 60.
- This is research-only. No runtime, process, database, ingester, data source, schema, image or deployment was changed.
