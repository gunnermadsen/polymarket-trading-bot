# Kraken Futures Tournament Generation 1

- Run: `20260822T022245Z-cd18aea3-f64fd675`
- Decision: broad causal model/horizon screen and paired feature ablation
- Candidates: 37
- Holdout rows used: **0**

| Candidate | Model | Horizon | Features | Status | Spearman | Net bps/trade | Trades | PF | + folds | 1.5x folds | Delayed bps |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| g1_ablate_3_ridge_4h_flow | ridge | 4h | flow | predictive_only | 0.0128 | -3.9777 | 350 | 0.9125 | 0/6 | 0/6 | -4.9557 |
| g1_elastic_net_4h_positioning | elastic_net | 4h | positioning | predictive_only | 0.0132 | -6.3261 | 310 | 0.8683 | 0/6 | 0/6 | -9.2790 |
| g1_ablate_1_elastic_net_4h_microstructure | elastic_net | 4h | microstructure | predictive_only | 0.0117 | -6.4874 | 372 | 0.8523 | 0/6 | 0/6 | -7.7415 |
| g1_elastic_net_2h_positioning | elastic_net | 2h | positioning | predictive_only | 0.0226 | -7.6784 | 627 | 0.7763 | 0/6 | 0/6 | -9.5619 |
| g1_ridge_4h_positioning | ridge | 4h | positioning | predictive_only | 0.0104 | -9.7272 | 149 | 0.8398 | 0/6 | 0/6 | -11.5868 |
| g1_ablate_3_ridge_4h_price | ridge | 4h | price | predictive_only | 0.0109 | -11.8013 | 222 | 0.7854 | 0/6 | 0/6 | -11.5509 |
| g1_ridge_4h_price | ridge | 4h | price | predictive_only | 0.0109 | -11.8013 | 222 | 0.7854 | 0/6 | 0/6 | -11.5509 |
| g1_elastic_net_8h_positioning | elastic_net | 8h | positioning | predictive_only | 0.0135 | -11.8454 | 269 | 0.8160 | 0/6 | 0/6 | -12.3182 |
| g1_ablate_1_elastic_net_4h_price | elastic_net | 4h | price | predictive_only | 0.0135 | -14.8298 | 266 | 0.7332 | 0/6 | 0/6 | -12.5316 |
| g1_ridge_8h_price | ridge | 8h | price | predictive_only | 0.0229 | -15.7345 | 269 | 0.7493 | 0/6 | 0/6 | -15.7492 |
| g1_ridge_8h_positioning | ridge | 8h | positioning | predictive_only | 0.0131 | -17.9019 | 244 | 0.7305 | 0/6 | 0/6 | -17.6865 |
| g1_elastic_net_30m_positioning | elastic_net | 30m | positioning | predictive_only | 0.0320 | — | 0 | — | 0/6 | 0/6 | — |
| g1_ridge_30m_price | ridge | 30m | price | predictive_only | 0.0293 | — | 0 | — | 0/6 | 0/6 | — |
| g1_ridge_1h_price | ridge | 1h | price | predictive_only | 0.0271 | — | 0 | — | 0/6 | 0/6 | — |
| g1_elastic_net_1h_positioning | elastic_net | 1h | positioning | predictive_only | 0.0257 | — | 0 | — | 0/6 | 0/6 | — |
| g1_ablate_2_elastic_net_2h_price | elastic_net | 2h | price | predictive_only | 0.0240 | — | 0 | — | 0/6 | 0/6 | — |
| g1_ridge_30m_positioning | ridge | 30m | positioning | predictive_only | 0.0238 | — | 0 | — | 0/6 | 0/6 | — |
| g1_ridge_1h_positioning | ridge | 1h | positioning | predictive_only | 0.0221 | — | 0 | — | 0/6 | 0/6 | — |
| g1_ablate_2_elastic_net_2h_flow | elastic_net | 2h | flow | predictive_only | 0.0210 | — | 0 | — | 0/6 | 0/6 | — |
| g1_ridge_2h_price | ridge | 2h | price | predictive_only | 0.0203 | — | 0 | — | 0/6 | 0/6 | — |
| g1_ridge_2h_positioning | ridge | 2h | positioning | predictive_only | 0.0180 | — | 0 | — | 0/6 | 0/6 | — |
| g1_ablate_2_elastic_net_2h_microstructure | elastic_net | 2h | microstructure | predictive_only | 0.0179 | — | 0 | — | 0/6 | 0/6 | — |
| g1_histogram_30m_positioning | histogram | 30m | positioning | predictive_only | 0.0155 | — | 0 | — | 0/6 | 0/6 | — |
| g1_ablate_1_elastic_net_4h_flow | elastic_net | 4h | flow | predictive_only | 0.0148 | — | 0 | — | 0/6 | 0/6 | — |
| g1_lightgbm_30m_positioning | lightgbm | 30m | positioning | predictive_only | 0.0134 | — | 0 | — | 0/6 | 0/6 | — |
| g1_ablate_3_ridge_4h_microstructure | ridge | 4h | microstructure | failed | 0.0084 | -8.3452 | 391 | 0.8223 | 0/6 | 0/6 | -9.2844 |
| g1_extra_trees_8h_positioning | extra_trees | 8h | positioning | failed | -0.0026 | -15.7345 | 269 | 0.7493 | 0/6 | 0/6 | -15.7492 |
| g1_lightgbm_8h_positioning | lightgbm | 8h | positioning | failed | -0.0093 | -15.7345 | 269 | 0.7493 | 0/6 | 0/6 | -15.7492 |
| g1_lightgbm_1h_positioning | lightgbm | 1h | positioning | failed | 0.0003 | -19.1059 | 133 | 0.5735 | 0/6 | 0/6 | -22.4090 |
| g1_extra_trees_1h_positioning | extra_trees | 1h | positioning | failed | 0.0084 | — | 0 | — | 0/6 | 0/6 | — |
| g1_histogram_1h_positioning | histogram | 1h | positioning | failed | 0.0047 | — | 0 | — | 0/6 | 0/6 | — |
| g1_extra_trees_4h_positioning | extra_trees | 4h | positioning | failed | -0.0036 | — | 0 | — | 0/6 | 0/6 | — |
| g1_extra_trees_2h_positioning | extra_trees | 2h | positioning | failed | -0.0058 | — | 0 | — | 0/6 | 0/6 | — |
| g1_histogram_2h_positioning | histogram | 2h | positioning | failed | -0.0110 | — | 0 | — | 0/6 | 0/6 | — |
| g1_lightgbm_4h_positioning | lightgbm | 4h | positioning | failed | -0.0117 | — | 0 | — | 0/6 | 0/6 | — |
| g1_lightgbm_2h_positioning | lightgbm | 2h | positioning | failed | -0.0130 | — | 0 | — | 0/6 | 0/6 | — |
| g1_histogram_4h_positioning | histogram | 4h | positioning | failed | -0.0130 | — | 0 | — | 0/6 | 0/6 | — |

## Retained artifacts

- `g1_ablate_3_ridge_4h_flow` — `6a7fe50c21e9f37d434797ec4ee2e6d54520a1b7371cf52f8e9b4dd77b18aee2` (3272 bytes)
- `g1_elastic_net_4h_positioning` — `9e802692bebdda6552304587e1082b3ca304eaf9190173960cbcacf147030108` (2798 bytes)
- `g1_ablate_1_elastic_net_4h_microstructure` — `482ed8b7339a8254b472cbf84b3413e47f2d5a4c6254b8246766518a8158e1c8` (4147 bytes)
