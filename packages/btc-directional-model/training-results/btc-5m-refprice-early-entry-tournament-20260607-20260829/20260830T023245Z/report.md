# RefPrice Early-Entry Tournament Report

## Selected sealed result

| Candidate | History arm | Policy | PnL | Coverage | Wins | Losses | Loss-recovery wins | Brier | Profit factor |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| refprice_binance_microstructure_context | authentic_only | probability_edge | 0.0000 | 0.00% | 0 | 0 | — | 0.2025 | — |

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
| refprice_path_control | authentic_only | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | authentic_only | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | authentic_only | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | authentic_only | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | chainlink_reconstructed | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | chainlink_reconstructed | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | chainlink_reconstructed | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | chainlink_reconstructed | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | binance_synthetic_extension | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | binance_synthetic_extension | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | binance_synthetic_extension | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | binance_synthetic_extension | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | uncertainty_weighted_hybrid | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | uncertainty_weighted_hybrid | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | uncertainty_weighted_hybrid | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_path_control | uncertainty_weighted_hybrid | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | authentic_only | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | authentic_only | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | authentic_only | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | authentic_only | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | chainlink_reconstructed | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | chainlink_reconstructed | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | chainlink_reconstructed | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | chainlink_reconstructed | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | binance_synthetic_extension | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | binance_synthetic_extension | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | binance_synthetic_extension | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | binance_synthetic_extension | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | uncertainty_weighted_hybrid | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | uncertainty_weighted_hybrid | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | uncertainty_weighted_hybrid | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_chainlink_crossvenue_context | uncertainty_weighted_hybrid | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | authentic_only | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | authentic_only | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | authentic_only | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | authentic_only | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | chainlink_reconstructed | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | chainlink_reconstructed | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | chainlink_reconstructed | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | chainlink_reconstructed | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | binance_synthetic_extension | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | binance_synthetic_extension | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | binance_synthetic_extension | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | binance_synthetic_extension | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | uncertainty_weighted_hybrid | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | uncertainty_weighted_hybrid | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | uncertainty_weighted_hybrid | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_binance_microstructure_context | uncertainty_weighted_hybrid | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | authentic_only | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | authentic_only | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | authentic_only | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | authentic_only | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | chainlink_reconstructed | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | chainlink_reconstructed | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | chainlink_reconstructed | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | chainlink_reconstructed | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | binance_synthetic_extension | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | binance_synthetic_extension | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | binance_synthetic_extension | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | binance_synthetic_extension | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | uncertainty_weighted_hybrid | probability_edge | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | uncertainty_weighted_hybrid | edge_debit | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | uncertainty_weighted_hybrid | edge_uncertainty_margin | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |
| refprice_nonnegative_consensus | uncertainty_weighted_hybrid | edge_loss_compensation_tail | 0.0000 | 0.0000 | 0.00% | 0 | 0 | — | — |

## Integrity and provenance

- Producing revision: `b2179e1fee40621b271ba173eafe17dc6b445e2d`
- Artifact SHA-256: `4d334d343217d52727d511bc3fa9ff312bba78720186674852b34e240250f901`
- Training observations: 577,250 across 23,090 markets
- TWAP-30/TWAP-60 inference: prohibited
- Runtime export/deployment: none
