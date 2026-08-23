# Kraken Futures Tournament Generation 3

- Run: `20260822T022245Z-cd18aea3-f64fd675`
- Decision: harden eligible Generation 2 lineages across features and deterministic seeds
- Candidates: 6
- Holdout rows used: **0**

| Candidate | Model | Horizon | Features | Status | Spearman | Net bps/trade | Trades | PF | + folds | 1.5x folds | Delayed bps |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| g3_1_ridge_4h_flow_s2 | ridge | 4h | flow | predictive_only | 0.0129 | -9.0203 | 539 | 0.7952 | 0/6 | 0/6 | -10.0604 |
| g3_1_ridge_4h_flow_s1 | ridge | 4h | flow | predictive_only | 0.0129 | -11.9337 | 438 | 0.7407 | 0/6 | 0/6 | -10.1476 |
| g3_2_elastic_net_4h_flow_s1 | elastic_net | 4h | flow | predictive_only | 0.0164 | -13.8553 | 465 | 0.7093 | 0/6 | 0/6 | -13.1280 |
| g3_2_elastic_net_4h_flow_s2 | elastic_net | 4h | flow | predictive_only | 0.0164 | -13.8553 | 465 | 0.7093 | 0/6 | 0/6 | -13.1280 |
| g3_2_elastic_net_4h_positioning_s1 | elastic_net | 4h | positioning | predictive_only | 0.0160 | — | 0 | — | 0/6 | 0/6 | — |
| g3_2_elastic_net_4h_positioning_s2 | elastic_net | 4h | positioning | predictive_only | 0.0160 | — | 0 | — | 0/6 | 0/6 | — |

## Retained artifacts

- `g3_1_ridge_4h_flow_s2` — `1a74bfce182391e7a19e1fbbf4bfdfa8b599584a8ba71eb8c8239a03cc189054` (3262 bytes)
