# Bucket-Conditioned Harm Risk Tournament

Every PnL, PF, W/L, recovery and drawdown value is a strategy replay with the named risk model applied. The risk model itself does not own standalone trading returns.

## Historical bucket selections

- 60_89: `enhanced_economic_harm`
- 90_119: `winner_preserving_loss`
- 120_149: `bucket_conditioned_harm`
- 150_179: `winner_preserving_loss`
- 180_240: `winner_preserving_loss`

## Exact six-strategy compatibility at primary coverage

| Risk model | PnL with risk | Delta PnL | Trades | W/L | Coverage | PF | Recovery | Max DD | Bad blocked | Good blocked | Alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| enhanced_economic_harm | $-268.89 | $48.82 | 2531 | 1780/751 | 85.4% | 0.892 | 2.656 | $283.28 | 18.4% | 18.9% | 0.98x |
| bucket_conditioned_harm | $-189.59 | $128.12 | 2474 | 1741/733 | 83.4% | 0.921 | 2.579 | $211.60 | 22.0% | 21.5% | 1.02x |
| winner_preserving_loss | $-376.40 | $-58.70 | 2540 | 1747/793 | 85.7% | 0.856 | 2.572 | $388.07 | 14.6% | 21.4% | 0.68x |
| distributional_downside | $-264.58 | $53.13 | 2410 | 1704/706 | 81.3% | 0.890 | 2.711 | $319.52 | 26.7% | 20.3% | 1.32x |
| bucket_regime_harm | $-277.81 | $39.90 | 2412 | 1680/732 | 81.3% | 0.885 | 2.593 | $310.52 | 20.7% | 23.8% | 0.87x |
| matched_confidence | $-294.26 | $23.45 | 2390 | 1708/682 | 80.6% | 0.875 | 2.863 | $308.89 | 26.2% | 18.6% | 1.41x |
| matched_edge | $-262.35 | $55.36 | 2607 | 1790/817 | 87.9% | 0.902 | 2.429 | $288.60 | 10.5% | 17.2% | 0.61x |
| previous_regime_conditioned_action | $-205.83 | $111.88 | 2504 | 1807/697 | 84.5% | 0.913 | 2.839 | $235.99 | 25.1% | 16.1% | 1.56x |
| frozen_logistic_loss_risk | $-243.67 | $74.03 | 2505 | 1809/696 | 84.5% | 0.899 | 2.892 | $281.06 | 26.5% | 15.3% | 1.73x |
| frozen_boosted_history_loss_risk | $-193.62 | $124.08 | 2421 | 1761/660 | 81.7% | 0.915 | 2.915 | $235.65 | 32.1% | 18.0% | 1.78x |
| no_risk | $-317.71 | $0.00 | 2965 | 2074/891 | 100.0% | 0.893 | 2.606 | $332.42 | 0.0% | 0.0% | —x |

## Strategy × risk model primary-band matrix

| Risk model | Strategy model | PnL | Delta | Trades | W/L | Coverage | Avg entry | Avg cost | PF | Recovery | Max DD | Alignment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| enhanced_economic_harm | BTC 5m specialist distilled fair value native admission paper | $-21.32 | $1.28 | 415 | 308/107 | 85.0% | 151.6s | $0.740 | 0.942 | 3.057 | $47.76 | 0.65x |
| enhanced_economic_harm | BTC 5m bridge aware specialist UMR paper | $-56.17 | $2.03 | 530 | 404/126 | 98.3% | 92.2s | $0.767 | 0.876 | 3.660 | $65.54 | 1.41x |
| enhanced_economic_harm | BTC 5m extended specialist official UMR paper | $-49.25 | $7.49 | 290 | 178/112 | 81.0% | 124.5s | $0.627 | 0.860 | 1.848 | $79.27 | 1.00x |
| enhanced_economic_harm | BTC 5m official high precision loss veto UMR paper | $-80.14 | $17.32 | 334 | 210/124 | 88.6% | 117.1s | $0.656 | 0.806 | 2.100 | $107.00 | 1.21x |
| enhanced_economic_harm | BTC 5m official temporal consensus UMR paper | $-42.40 | $21.16 | 283 | 172/111 | 80.9% | 123.6s | $0.617 | 0.876 | 1.770 | $75.84 | 1.16x |
| enhanced_economic_harm | BTC 5m official VWAP admission UMR paper | $-19.60 | $-0.45 | 679 | 508/171 | 79.6% | 79.4s | $0.737 | 0.966 | 3.076 | $82.35 | 1.15x |
| bucket_conditioned_harm | BTC 5m specialist distilled fair value native admission paper | $-12.94 | $9.66 | 467 | 351/116 | 95.7% | 147.0s | $0.745 | 0.968 | 3.126 | $48.08 | 0.76x |
| bucket_conditioned_harm | BTC 5m bridge aware specialist UMR paper | $-30.59 | $27.60 | 489 | 378/111 | 90.7% | 92.2s | $0.769 | 0.923 | 3.689 | $40.93 | 1.10x |
| bucket_conditioned_harm | BTC 5m extended specialist official UMR paper | $-25.09 | $31.65 | 284 | 175/109 | 79.3% | 128.0s | $0.613 | 0.925 | 1.736 | $72.15 | 0.97x |
| bucket_conditioned_harm | BTC 5m official high precision loss veto UMR paper | $-39.54 | $57.93 | 293 | 186/107 | 77.7% | 120.0s | $0.641 | 0.885 | 1.963 | $79.53 | 1.12x |
| bucket_conditioned_harm | BTC 5m official temporal consensus UMR paper | $-53.24 | $10.32 | 266 | 156/110 | 76.0% | 126.4s | $0.605 | 0.841 | 1.687 | $79.44 | 0.96x |
| bucket_conditioned_harm | BTC 5m official VWAP admission UMR paper | $-28.18 | $-9.03 | 675 | 495/180 | 79.1% | 79.9s | $0.724 | 0.952 | 2.888 | $62.39 | 0.86x |
| winner_preserving_loss | BTC 5m specialist distilled fair value native admission paper | $-10.48 | $12.12 | 412 | 308/104 | 84.4% | 152.6s | $0.740 | 0.970 | 3.052 | $46.12 | 0.79x |
| winner_preserving_loss | BTC 5m bridge aware specialist UMR paper | $-55.91 | $2.28 | 517 | 394/123 | 95.9% | 92.8s | $0.767 | 0.873 | 3.668 | $64.56 | 0.79x |
| winner_preserving_loss | BTC 5m extended specialist official UMR paper | $-62.74 | $-6.01 | 317 | 193/124 | 88.5% | 129.9s | $0.628 | 0.839 | 1.855 | $79.48 | 0.92x |
| winner_preserving_loss | BTC 5m official high precision loss veto UMR paper | $-91.17 | $6.30 | 342 | 213/129 | 90.7% | 120.7s | $0.656 | 0.788 | 2.095 | $96.51 | 0.83x |
| winner_preserving_loss | BTC 5m official temporal consensus UMR paper | $-73.60 | $-10.04 | 319 | 190/129 | 91.1% | 126.1s | $0.621 | 0.816 | 1.805 | $83.84 | 0.88x |
| winner_preserving_loss | BTC 5m official VWAP admission UMR paper | $-82.50 | $-63.35 | 633 | 449/184 | 74.2% | 82.1s | $0.718 | 0.864 | 2.823 | $104.94 | 0.58x |
| distributional_downside | BTC 5m specialist distilled fair value native admission paper | $-10.92 | $11.68 | 485 | 365/120 | 99.4% | 144.3s | $0.745 | 0.974 | 3.123 | $41.69 | 1.33x |
| distributional_downside | BTC 5m bridge aware specialist UMR paper | $-66.70 | $-8.50 | 498 | 372/126 | 92.4% | 92.4s | $0.757 | 0.852 | 3.464 | $77.03 | 0.25x |
| distributional_downside | BTC 5m extended specialist official UMR paper | $-74.05 | $-17.31 | 294 | 180/114 | 82.1% | 128.4s | $0.642 | 0.800 | 1.973 | $78.45 | 1.12x |
| distributional_downside | BTC 5m official high precision loss veto UMR paper | $-86.69 | $10.77 | 302 | 183/119 | 80.1% | 129.9s | $0.643 | 0.777 | 1.980 | $93.73 | 1.14x |
| distributional_downside | BTC 5m official temporal consensus UMR paper | $-63.53 | $0.02 | 268 | 163/105 | 76.6% | 126.1s | $0.635 | 0.811 | 1.914 | $78.04 | 1.19x |
| distributional_downside | BTC 5m official VWAP admission UMR paper | $37.31 | $56.46 | 563 | 441/122 | 66.0% | 74.5s | $0.753 | 1.083 | 3.337 | $45.34 | 1.48x |
| bucket_regime_harm | BTC 5m specialist distilled fair value native admission paper | $-3.34 | $19.26 | 452 | 343/109 | 92.6% | 146.3s | $0.748 | 0.991 | 3.175 | $44.30 | 0.89x |
| bucket_regime_harm | BTC 5m bridge aware specialist UMR paper | $-53.56 | $4.63 | 484 | 367/117 | 89.8% | 93.0s | $0.764 | 0.872 | 3.598 | $62.40 | 0.69x |
| bucket_regime_harm | BTC 5m extended specialist official UMR paper | $-44.37 | $12.37 | 287 | 174/113 | 80.2% | 124.7s | $0.616 | 0.873 | 1.763 | $60.89 | 0.88x |
| bucket_regime_harm | BTC 5m official high precision loss veto UMR paper | $-54.39 | $43.07 | 288 | 180/108 | 76.4% | 118.4s | $0.642 | 0.845 | 1.973 | $83.49 | 1.05x |
| bucket_regime_harm | BTC 5m official temporal consensus UMR paper | $-72.77 | $-9.21 | 279 | 161/118 | 79.7% | 123.3s | $0.608 | 0.798 | 1.710 | $90.94 | 0.70x |
| bucket_regime_harm | BTC 5m official VWAP admission UMR paper | $-49.37 | $-30.22 | 622 | 455/167 | 72.9% | 78.7s | $0.730 | 0.912 | 2.988 | $101.29 | 0.87x |
| matched_confidence | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $0.00 | 488 | 366/122 | 100.0% | 144.4s | $0.747 | 0.947 | 3.168 | $46.67 | —x |
| matched_confidence | BTC 5m bridge aware specialist UMR paper | $-44.57 | $13.62 | 531 | 408/123 | 98.5% | 91.2s | $0.768 | 0.900 | 3.685 | $54.91 | 9.53x |
| matched_confidence | BTC 5m extended specialist official UMR paper | $-66.44 | $-9.70 | 285 | 177/108 | 79.6% | 122.1s | $0.647 | 0.812 | 2.018 | $72.19 | 1.11x |
| matched_confidence | BTC 5m official high precision loss veto UMR paper | $-84.79 | $12.67 | 301 | 185/116 | 79.8% | 122.4s | $0.651 | 0.780 | 2.045 | $92.32 | 0.97x |
| matched_confidence | BTC 5m official temporal consensus UMR paper | $-62.01 | $1.54 | 256 | 157/99 | 73.1% | 120.7s | $0.641 | 0.806 | 1.968 | $77.65 | 1.30x |
| matched_confidence | BTC 5m official VWAP admission UMR paper | $-13.84 | $5.31 | 529 | 415/114 | 62.0% | 73.8s | $0.773 | 0.967 | 3.765 | $51.18 | 1.41x |
| matched_edge | BTC 5m specialist distilled fair value native admission paper | $2.55 | $25.15 | 403 | 298/105 | 82.6% | 147.8s | $0.725 | 1.007 | 2.818 | $41.33 | 0.71x |
| matched_edge | BTC 5m bridge aware specialist UMR paper | $-53.91 | $4.28 | 504 | 380/124 | 93.5% | 91.9s | $0.758 | 0.878 | 3.491 | $64.25 | 0.64x |
| matched_edge | BTC 5m extended specialist official UMR paper | $-37.56 | $19.18 | 309 | 190/119 | 86.3% | 126.3s | $0.618 | 0.898 | 1.778 | $69.30 | 1.01x |
| matched_edge | BTC 5m official high precision loss veto UMR paper | $-79.02 | $18.44 | 354 | 222/132 | 93.9% | 120.2s | $0.651 | 0.819 | 2.052 | $99.49 | 1.30x |
| matched_edge | BTC 5m official temporal consensus UMR paper | $-61.08 | $2.48 | 342 | 206/136 | 97.7% | 126.6s | $0.617 | 0.855 | 1.772 | $85.79 | 1.01x |
| matched_edge | BTC 5m official VWAP admission UMR paper | $-33.33 | $-14.18 | 695 | 494/201 | 81.5% | 81.7s | $0.702 | 0.949 | 2.589 | $67.85 | 0.41x |
| previous_regime_conditioned_action | BTC 5m specialist distilled fair value native admission paper | $-25.97 | $-3.37 | 488 | 366/122 | 100.0% | 144.6s | $0.748 | 0.939 | 3.194 | $48.25 | 1.20x |
| previous_regime_conditioned_action | BTC 5m bridge aware specialist UMR paper | $-9.01 | $49.18 | 485 | 386/99 | 90.0% | 88.4s | $0.783 | 0.976 | 3.997 | $22.81 | 3.09x |
| previous_regime_conditioned_action | BTC 5m extended specialist official UMR paper | $-52.68 | $4.06 | 264 | 162/102 | 73.7% | 133.4s | $0.633 | 0.838 | 1.894 | $61.80 | 0.93x |
| previous_regime_conditioned_action | BTC 5m official high precision loss veto UMR paper | $-91.56 | $5.90 | 293 | 181/112 | 77.7% | 124.4s | $0.660 | 0.757 | 2.136 | $93.71 | 0.92x |
| previous_regime_conditioned_action | BTC 5m official temporal consensus UMR paper | $-55.83 | $7.73 | 277 | 166/111 | 79.1% | 129.0s | $0.619 | 0.837 | 1.786 | $68.24 | 0.85x |
| previous_regime_conditioned_action | BTC 5m official VWAP admission UMR paper | $29.24 | $48.39 | 697 | 546/151 | 81.7% | 76.5s | $0.758 | 1.055 | 3.427 | $54.45 | 2.33x |
| frozen_logistic_loss_risk | BTC 5m specialist distilled fair value native admission paper | $-25.71 | $-3.11 | 487 | 365/122 | 99.8% | 144.2s | $0.748 | 0.940 | 3.184 | $49.43 | 0.00x |
| frozen_logistic_loss_risk | BTC 5m bridge aware specialist UMR paper | $-43.40 | $14.79 | 529 | 407/122 | 98.1% | 90.9s | $0.769 | 0.902 | 3.699 | $53.74 | 7.42x |
| frozen_logistic_loss_risk | BTC 5m extended specialist official UMR paper | $-59.68 | $-2.94 | 293 | 183/110 | 81.8% | 128.4s | $0.645 | 0.834 | 1.996 | $69.99 | 1.13x |
| frozen_logistic_loss_risk | BTC 5m official high precision loss veto UMR paper | $-104.51 | $-7.04 | 319 | 196/123 | 84.6% | 124.2s | $0.660 | 0.748 | 2.130 | $109.96 | 1.01x |
| frozen_logistic_loss_risk | BTC 5m official temporal consensus UMR paper | $-46.57 | $16.99 | 263 | 164/99 | 75.1% | 125.1s | $0.638 | 0.853 | 1.941 | $63.70 | 1.35x |
| frozen_logistic_loss_risk | BTC 5m official VWAP admission UMR paper | $36.19 | $55.34 | 614 | 494/120 | 72.0% | 75.0s | $0.776 | 1.082 | 3.806 | $50.63 | 2.09x |
| frozen_boosted_history_loss_risk | BTC 5m specialist distilled fair value native admission paper | $-12.82 | $9.78 | 487 | 367/120 | 99.8% | 144.9s | $0.747 | 0.969 | 3.155 | $42.43 | 6.00x |
| frozen_boosted_history_loss_risk | BTC 5m bridge aware specialist UMR paper | $-43.94 | $14.25 | 513 | 396/117 | 95.2% | 90.8s | $0.772 | 0.896 | 3.775 | $53.90 | 2.65x |
| frozen_boosted_history_loss_risk | BTC 5m extended specialist official UMR paper | $-40.72 | $16.02 | 256 | 163/93 | 71.5% | 132.3s | $0.648 | 0.867 | 2.022 | $61.17 | 1.22x |
| frozen_boosted_history_loss_risk | BTC 5m official high precision loss veto UMR paper | $-85.11 | $12.35 | 297 | 183/114 | 78.8% | 126.7s | $0.653 | 0.776 | 2.070 | $87.00 | 1.10x |
| frozen_boosted_history_loss_risk | BTC 5m official temporal consensus UMR paper | $-37.63 | $25.93 | 259 | 163/96 | 74.0% | 126.4s | $0.638 | 0.879 | 1.933 | $56.81 | 1.45x |
| frozen_boosted_history_loss_risk | BTC 5m official VWAP admission UMR paper | $26.60 | $45.75 | 609 | 489/120 | 71.4% | 74.2s | $0.778 | 1.059 | 3.847 | $68.53 | 2.02x |
| no_risk | BTC 5m specialist distilled fair value native admission paper | $-22.60 | $0.00 | 488 | 366/122 | 100.0% | 144.4s | $0.747 | 0.947 | 3.168 | $46.67 | —x |
| no_risk | BTC 5m bridge aware specialist UMR paper | $-58.19 | $0.00 | 539 | 410/129 | 100.0% | 92.5s | $0.765 | 0.874 | 3.635 | $68.53 | —x |
| no_risk | BTC 5m extended specialist official UMR paper | $-56.74 | $0.00 | 358 | 220/138 | 100.0% | 130.6s | $0.625 | 0.870 | 1.833 | $78.23 | —x |
| no_risk | BTC 5m official high precision loss veto UMR paper | $-97.46 | $0.00 | 377 | 235/142 | 100.0% | 121.1s | $0.655 | 0.794 | 2.085 | $108.57 | —x |
| no_risk | BTC 5m official temporal consensus UMR paper | $-63.56 | $0.00 | 350 | 211/139 | 100.0% | 126.3s | $0.618 | 0.853 | 1.780 | $86.61 | —x |
| no_risk | BTC 5m official VWAP admission UMR paper | $-19.15 | $0.00 | 853 | 632/221 | 100.0% | 79.7s | $0.728 | 0.974 | 2.936 | $73.35 | —x |

## Granular champions

Champion identity is risk model + strategy model + time bucket + side + coverage policy. The full supported and sparse cell inventory is in `ledgers/slice-matrix.parquet`.

## Forward readiness

September audit status: **insufficient_for_six_strategy_post_admission_replay**. The six processes produced 79161 decisions but only 163 admitted buys; one process admitted zero. These rows were not repurposed into synthetic risk candidates.

## Integrity and limitations

- The August 21-25 exact six-strategy cohort was already viewed and is compatibility evidence, not a sealed qualification cohort.
- Five deployed UMR processes were created September 7; the September 7-9 audit has only 65 admitted buys across those five and zero for one process, so it cannot fairly qualify all six post-admission risk pairings.
- Rejected live strategy decisions were not converted into synthetic admitted candidates.
- The full established April-August causal OOF construction range was used; all available numeric and boolean SSD dimensions were eligible without strategy identity.
- The common historical construction begins at second 60, so 15-59 remains unavailable.
- This is research-only. No runtime, process, database, ingester, data source, schema, image or deployment was changed.
