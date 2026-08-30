# Post-training evaluation correction

No model was retrained. The original zero-trade economics were invalid because they queried a retired execution artifact identity. The table below uses the existing capacity execution table. The model artifact remains unqualified and is tagged only to preserve disqualified training provenance; it is not eligible for deployment.

# RefPrice Early-Entry Tournament Report

## Selected sealed result

| Candidate | History arm | Policy | PnL | Coverage | Wins | Losses | Loss-recovery wins | Brier | Profit factor |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| refprice_nonnegative_consensus | authentic_only | edge_loss_compensation_tail | -0.1613 | 2.11% | 14 | 20 | 0.7029 | 0.1969 | 0.9959 |

## Full candidate predictive table

| Candidate | History arm | Markets | Rows | Brier | Log loss | Accuracy | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| refprice_nonnegative_consensus | authentic_only | 1612 | 40300 | 0.1969 | 0.5774 | 0.6993 | 0.0856 |
| refprice_binance_microstructure_context | authentic_only | 1612 | 40300 | 0.2025 | 0.5923 | 0.6959 | 0.1034 |
| refprice_binance_microstructure_context | uncertainty_weighted_hybrid | 1612 | 40300 | 0.2027 | 0.5938 | 0.7006 | 0.0960 |
| refprice_binance_microstructure_context | binance_synthetic_extension | 1612 | 40300 | 0.2045 | 0.5980 | 0.7029 | 0.1029 |
| refprice_path_control | chainlink_reconstructed | 1612 | 40300 | 0.2047 | 0.5977 | 0.7001 | 0.0944 |
| refprice_binance_microstructure_context | chainlink_reconstructed | 1612 | 40300 | 0.2058 | 0.6001 | 0.6913 | 0.1119 |
| refprice_path_control | binance_synthetic_extension | 1612 | 40300 | 0.2074 | 0.6041 | 0.6988 | 0.1063 |
| refprice_chainlink_crossvenue_context | binance_synthetic_extension | 1612 | 40300 | 0.2084 | 0.6062 | 0.6922 | 0.1119 |
| refprice_path_control | uncertainty_weighted_hybrid | 1612 | 40300 | 0.2087 | 0.6069 | 0.6965 | 0.1114 |
| refprice_path_control | authentic_only | 1612 | 40300 | 0.2088 | 0.6072 | 0.6953 | 0.1096 |
| refprice_chainlink_crossvenue_context | chainlink_reconstructed | 1612 | 40300 | 0.2109 | 0.6118 | 0.6914 | 0.1204 |
| refprice_chainlink_crossvenue_context | uncertainty_weighted_hybrid | 1612 | 40300 | 0.2136 | 0.6177 | 0.6888 | 0.1259 |
| refprice_chainlink_crossvenue_context | authentic_only | 1612 | 40300 | 0.2141 | 0.6188 | 0.6857 | 0.1276 |
| refprice_nonnegative_consensus | binance_synthetic_extension | 1612 | 40300 | 0.2158 | 0.6225 | 0.6813 | 0.1344 |
| refprice_nonnegative_consensus | uncertainty_weighted_hybrid | 1612 | 40300 | 0.2159 | 0.6225 | 0.6792 | 0.1339 |
| refprice_nonnegative_consensus | chainlink_reconstructed | 1612 | 40300 | 0.2185 | 0.6278 | 0.6667 | 0.1380 |

## Economic candidate-policy table

| Candidate | History arm | Policy | Dev PnL | Sealed PnL | Coverage | Wins | Losses | Recovery wins | Profit factor |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| refprice_nonnegative_consensus | authentic_only | edge_loss_compensation_tail | 9.9435 | -0.1613 | 2.11% | 14 | 20 | 0.7029 | 0.9959 |
| refprice_nonnegative_consensus | authentic_only | edge_uncertainty_margin | 8.1605 | -1.5336 | 2.92% | 22 | 25 | 0.9049 | 0.9725 |
| refprice_binance_microstructure_context | binance_synthetic_extension | edge_loss_compensation_tail | -9.3270 | -1.5863 | 2.54% | 17 | 24 | 0.7328 | 0.9666 |
| refprice_path_control | uncertainty_weighted_hybrid | edge_loss_compensation_tail | -14.9623 | -7.8779 | 4.40% | 23 | 48 | 0.5333 | 0.8985 |
| refprice_path_control | chainlink_reconstructed | edge_loss_compensation_tail | -14.1661 | -8.5144 | 3.66% | 20 | 39 | 0.5866 | 0.8742 |
| refprice_binance_microstructure_context | binance_synthetic_extension | edge_uncertainty_margin | 1.0747 | -9.1115 | 3.35% | 23 | 31 | 0.8577 | 0.8650 |
| refprice_binance_microstructure_context | uncertainty_weighted_hybrid | edge_loss_compensation_tail | 0.2454 | -9.6618 | 2.67% | 16 | 27 | 0.7224 | 0.8203 |
| refprice_chainlink_crossvenue_context | uncertainty_weighted_hybrid | edge_uncertainty_margin | -14.8409 | -9.9818 | 4.96% | 29 | 51 | 0.6396 | 0.8891 |
| refprice_binance_microstructure_context | authentic_only | probability_edge | -41.0480 | -11.5689 | 53.78% | 401 | 466 | 0.8704 | 0.9887 |
| refprice_nonnegative_consensus | binance_synthetic_extension | edge_loss_compensation_tail | -19.3155 | -11.6096 | 2.67% | 11 | 32 | 0.4579 | 0.7507 |
| refprice_path_control | binance_synthetic_extension | edge_loss_compensation_tail | -17.4352 | -12.2563 | 3.91% | 20 | 43 | 0.5605 | 0.8299 |
| refprice_chainlink_crossvenue_context | uncertainty_weighted_hybrid | edge_loss_compensation_tail | -21.1862 | -13.9808 | 4.16% | 21 | 46 | 0.5590 | 0.8167 |
| refprice_path_control | authentic_only | edge_loss_compensation_tail | -18.0917 | -13.9919 | 4.53% | 22 | 51 | 0.5222 | 0.8260 |
| refprice_nonnegative_consensus | binance_synthetic_extension | edge_uncertainty_margin | -19.3593 | -14.4743 | 2.98% | 13 | 35 | 0.5074 | 0.7320 |
| refprice_path_control | uncertainty_weighted_hybrid | edge_uncertainty_margin | -14.0771 | -15.1796 | 5.09% | 28 | 54 | 0.6170 | 0.8403 |
| refprice_binance_microstructure_context | chainlink_reconstructed | edge_loss_compensation_tail | -2.2680 | -16.5859 | 3.04% | 15 | 34 | 0.6153 | 0.7170 |
| refprice_binance_microstructure_context | uncertainty_weighted_hybrid | edge_uncertainty_margin | -1.2239 | -18.5384 | 3.60% | 23 | 35 | 0.8641 | 0.7605 |
| refprice_binance_microstructure_context | chainlink_reconstructed | probability_edge | -99.3963 | -20.4175 | 54.65% | 397 | 484 | 0.8367 | 0.9803 |
| refprice_binance_microstructure_context | authentic_only | edge_loss_compensation_tail | -5.0992 | -21.0016 | 2.73% | 14 | 30 | 0.7287 | 0.6404 |
| refprice_nonnegative_consensus | uncertainty_weighted_hybrid | edge_loss_compensation_tail | -18.2419 | -22.5443 | 2.98% | 10 | 38 | 0.4585 | 0.5740 |
| refprice_binance_microstructure_context | authentic_only | edge_debit | -56.2855 | -24.3892 | 51.92% | 379 | 458 | 0.8483 | 0.9755 |
| refprice_chainlink_crossvenue_context | binance_synthetic_extension | edge_loss_compensation_tail | -21.3911 | -25.0376 | 4.53% | 21 | 52 | 0.5700 | 0.7085 |
| refprice_binance_microstructure_context | chainlink_reconstructed | edge_uncertainty_margin | 0.4660 | -25.2595 | 3.72% | 20 | 40 | 0.7441 | 0.6719 |
| refprice_nonnegative_consensus | uncertainty_weighted_hybrid | edge_uncertainty_margin | -18.1748 | -25.6198 | 3.29% | 12 | 41 | 0.5074 | 0.5768 |
| refprice_path_control | chainlink_reconstructed | edge_uncertainty_margin | -9.7239 | -26.3485 | 4.65% | 26 | 49 | 0.7293 | 0.7276 |
| refprice_chainlink_crossvenue_context | chainlink_reconstructed | edge_loss_compensation_tail | -22.9926 | -26.5944 | 4.34% | 19 | 51 | 0.5481 | 0.6797 |
| refprice_nonnegative_consensus | chainlink_reconstructed | edge_loss_compensation_tail | -15.6295 | -27.0680 | 3.78% | 13 | 48 | 0.4516 | 0.5998 |
| refprice_chainlink_crossvenue_context | authentic_only | edge_loss_compensation_tail | -21.8006 | -27.3817 | 4.47% | 18 | 54 | 0.4952 | 0.6732 |
| refprice_path_control | binance_synthetic_extension | edge_uncertainty_margin | -13.3106 | -27.4143 | 4.65% | 24 | 51 | 0.6618 | 0.7111 |
| refprice_binance_microstructure_context | binance_synthetic_extension | probability_edge | -60.4227 | -28.2150 | 49.44% | 358 | 439 | 0.8402 | 0.9706 |
| refprice_binance_microstructure_context | authentic_only | edge_uncertainty_margin | 3.2384 | -28.3860 | 3.72% | 22 | 38 | 0.8835 | 0.6553 |
| refprice_binance_microstructure_context | uncertainty_weighted_hybrid | probability_edge | -77.7970 | -29.1029 | 50.62% | 372 | 444 | 0.8635 | 0.9703 |
| refprice_path_control | authentic_only | edge_uncertainty_margin | -13.5681 | -29.1310 | 5.27% | 26 | 59 | 0.6149 | 0.7166 |
| refprice_nonnegative_consensus | chainlink_reconstructed | edge_uncertainty_margin | -15.3957 | -29.7664 | 4.03% | 15 | 50 | 0.5000 | 0.6000 |
| refprice_chainlink_crossvenue_context | binance_synthetic_extension | edge_uncertainty_margin | -14.3052 | -30.6658 | 5.09% | 25 | 57 | 0.6331 | 0.6928 |
| refprice_chainlink_crossvenue_context | authentic_only | edge_uncertainty_margin | -13.6277 | -31.2382 | 5.33% | 25 | 61 | 0.5910 | 0.6934 |
| refprice_chainlink_crossvenue_context | chainlink_reconstructed | edge_uncertainty_margin | -15.8756 | -33.7231 | 5.15% | 25 | 58 | 0.6412 | 0.6722 |
| refprice_binance_microstructure_context | binance_synthetic_extension | edge_debit | -62.6120 | -34.1840 | 47.39% | 337 | 427 | 0.8197 | 0.9628 |
| refprice_binance_microstructure_context | chainlink_reconstructed | edge_debit | -85.5677 | -40.1735 | 52.79% | 374 | 477 | 0.8164 | 0.9604 |
| refprice_path_control | binance_synthetic_extension | probability_edge | -163.6123 | -40.4274 | 51.55% | 361 | 470 | 0.8008 | 0.9591 |
| refprice_path_control | authentic_only | probability_edge | -160.1729 | -46.5964 | 51.12% | 353 | 471 | 0.7868 | 0.9526 |
| refprice_binance_microstructure_context | uncertainty_weighted_hybrid | edge_debit | -72.7547 | -47.6299 | 48.76% | 349 | 437 | 0.8404 | 0.9502 |
| refprice_nonnegative_consensus | binance_synthetic_extension | probability_edge | -183.2527 | -50.7923 | 54.22% | 368 | 506 | 0.7646 | 0.9512 |
| refprice_nonnegative_consensus | binance_synthetic_extension | edge_debit | -181.7802 | -53.7153 | 53.35% | 359 | 501 | 0.7562 | 0.9477 |
| refprice_chainlink_crossvenue_context | binance_synthetic_extension | probability_edge | -158.0235 | -58.4096 | 51.80% | 356 | 479 | 0.7894 | 0.9415 |
| refprice_path_control | uncertainty_weighted_hybrid | probability_edge | -165.0489 | -59.9906 | 51.30% | 351 | 476 | 0.7848 | 0.9396 |
| refprice_nonnegative_consensus | uncertainty_weighted_hybrid | probability_edge | -167.1538 | -60.7171 | 54.71% | 370 | 512 | 0.7670 | 0.9422 |
| refprice_path_control | chainlink_reconstructed | probability_edge | -163.4149 | -62.5729 | 51.86% | 366 | 470 | 0.8303 | 0.9379 |
| refprice_path_control | chainlink_reconstructed | edge_debit | -174.8023 | -63.5425 | 50.12% | 347 | 461 | 0.8051 | 0.9349 |
| refprice_chainlink_crossvenue_context | chainlink_reconstructed | probability_edge | -136.2208 | -66.1815 | 51.36% | 349 | 479 | 0.7806 | 0.9333 |
| refprice_chainlink_crossvenue_context | authentic_only | probability_edge | -139.3432 | -68.2530 | 51.49% | 340 | 490 | 0.7452 | 0.9311 |
| refprice_nonnegative_consensus | authentic_only | edge_debit | -101.0039 | -69.8324 | 54.90% | 411 | 474 | 0.9272 | 0.9352 |
| refprice_nonnegative_consensus | uncertainty_weighted_hybrid | edge_debit | -177.3006 | -71.3917 | 53.85% | 360 | 508 | 0.7608 | 0.9315 |
| refprice_path_control | binance_synthetic_extension | edge_debit | -186.1182 | -73.9799 | 49.50% | 334 | 464 | 0.7793 | 0.9237 |
| refprice_chainlink_crossvenue_context | uncertainty_weighted_hybrid | probability_edge | -149.3804 | -76.7871 | 51.49% | 341 | 489 | 0.7555 | 0.9230 |
| refprice_nonnegative_consensus | authentic_only | probability_edge | -122.3027 | -77.2283 | 60.67% | 486 | 492 | 1.0583 | 0.9334 |
| refprice_chainlink_crossvenue_context | binance_synthetic_extension | edge_debit | -162.6442 | -80.3259 | 50.37% | 338 | 474 | 0.7766 | 0.9182 |
| refprice_chainlink_crossvenue_context | uncertainty_weighted_hybrid | edge_debit | -158.7680 | -81.9859 | 50.43% | 330 | 483 | 0.7456 | 0.9164 |
| refprice_chainlink_crossvenue_context | chainlink_reconstructed | edge_debit | -146.7301 | -84.5722 | 50.06% | 332 | 475 | 0.7651 | 0.9135 |
| refprice_path_control | uncertainty_weighted_hybrid | edge_debit | -159.0832 | -84.7904 | 49.50% | 329 | 469 | 0.7685 | 0.9128 |
| refprice_path_control | authentic_only | edge_debit | -162.3924 | -89.6867 | 49.38% | 328 | 468 | 0.7722 | 0.9076 |
| refprice_chainlink_crossvenue_context | authentic_only | edge_debit | -150.7072 | -90.2086 | 50.25% | 325 | 485 | 0.7381 | 0.9078 |
| refprice_nonnegative_consensus | chainlink_reconstructed | probability_edge | -171.2131 | -98.4646 | 56.58% | 370 | 542 | 0.7499 | 0.9103 |
| refprice_nonnegative_consensus | chainlink_reconstructed | edge_debit | -175.6397 | -104.7219 | 56.14% | 364 | 541 | 0.7440 | 0.9043 |

## Integrity and provenance

- Producing revision: `b2179e1fee40621b271ba173eafe17dc6b445e2d`
- Artifact SHA-256: `4d334d343217d52727d511bc3fa9ff312bba78720186674852b34e240250f901`
- Training observations: 577,250 across 23,090 markets
- TWAP-30/TWAP-60 inference: prohibited
- Runtime export/deployment: none
