# Kraken Futures Tournament Generation 2

- Run: `20260822T022245Z-cd18aea3-f64fd675`
- Decision: improve eligible Generation 1 lineages with bounded local tuning
- Candidates: 8
- Holdout rows used: **0**

| Candidate | Model | Horizon | Features | Status | Spearman | Net bps/trade | Trades | PF | + folds | 1.5x folds | Delayed bps |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| g2_1_ridge_4h_flow_v0 | ridge | 4h | flow | predictive_only | 0.0129 | -11.9337 | 438 | 0.7407 | 0/6 | 0/6 | -10.1476 |
| g2_1_ridge_4h_flow_v1 | ridge | 4h | flow | predictive_only | 0.0129 | -11.9593 | 438 | 0.7408 | 0/6 | 0/6 | -10.0605 |
| g2_1_ridge_4h_flow_v3 | ridge | 4h | flow | predictive_only | 0.0129 | -11.9703 | 438 | 0.7401 | 0/6 | 0/6 | -10.1546 |
| g2_1_ridge_4h_flow_v2 | ridge | 4h | flow | predictive_only | 0.0128 | -14.4632 | 282 | 0.7477 | 0/6 | 0/6 | -9.7895 |
| g2_2_elastic_net_4h_positioning_v0 | elastic_net | 4h | positioning | predictive_only | 0.0160 | — | 0 | — | 0/6 | 0/6 | — |
| g2_2_elastic_net_4h_positioning_v1 | elastic_net | 4h | positioning | failed | 0.0057 | -9.8530 | 231 | 0.8358 | 0/6 | 0/6 | -8.9853 |
| g2_2_elastic_net_4h_positioning_v2 | elastic_net | 4h | positioning | failed | 0.0057 | -9.8530 | 231 | 0.8358 | 0/6 | 0/6 | -8.9853 |
| g2_2_elastic_net_4h_positioning_v3 | elastic_net | 4h | positioning | failed | 0.0057 | -9.8530 | 231 | 0.8358 | 0/6 | 0/6 | -8.9853 |

## Retained artifacts

- `g2_1_ridge_4h_flow_v0` — `77a961c6e869e514070c693dd5080b40cefceb76ce60e3779281746fa200d4db` (3272 bytes)
- `g2_1_ridge_4h_flow_v1` — `81182caae0291ac887be7f8491f5eede4c61e59a4b8e89aabe0173aa93a8e535` (3268 bytes)
