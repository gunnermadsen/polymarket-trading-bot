# Kraken Futures Net-Expectancy Development Benchmark

## Verdict

**NO_DEVELOPMENT_CANDIDATE_QUALIFIED**

## Reproducibility

- Run: `20260728T200245Z-f64fd675-b9f51aad`
- Generated: 2026-07-28 20:05:36.468287+00:00
- Configuration SHA-256: `b9f51aadd57f695dd734f17d3506ed93ad4aae38822d4c8eeb0a73a09f191a70`
- Source snapshot SHA-256: `f64fd6757d8e8fbfc0cc2615bb6a5c06283a2ca1522c9ae2904f557eca4bffe6`
- Code SHA-256: `38ab1cec876b773899b454374baf3222b7a03f644f51aec5048fbfde68f8038b`

## Training prerequisites

| Item | Value |
| --- | --- |
| Funding complete | PASS |
| Funding rows | 110747 |
| Funding first | 2023-05-31T12:15:00+00:00 |
| Funding last | 2026-07-28T02:45:00+00:00 |
| Funding provenance verified | PASS |
| Funding import id | 1372f049620d5fdc5e735b688302c8752fcaa53419b7c5ec468d2389908e831c |
| Funding binding SHA-256 | 67e6ee420e43811f9ebb82d20a781f674c0c75ebf5041b7e527b93d957f9a19d |
| Funding manifest SHA-256 | bf1e3df33cbee2358761d7a4adbc799d074fb07e29895b5494965ed73db9d6fa |
| Immutable source objects | 2 |
| Verified published objects | 34 |
| Funding used as feature | FAIL |
| Taker fee per side bps | 5.0000 |
| Maker fee per side bps | 2.0000 |

## Execution realism

| Item | Value |
| --- | --- |
| Status | preliminary_research_assumption |
| Feature available at entry | PASS |
| Inference/routing latency seconds | 0 |
| Entry price | next completed 15-minute candle open |
| Deployable-edge claim allowed | FAIL |
| Required follow-up | delayed-entry sensitivity using 1-minute or L2 data before any live-capital interpretation |

## Candidate comparison

| Candidate | Bars | Model | Features | OOF MAE bps | OOF RMSE bps | OOF R² | OOF Spearman | Positive folds | Net bps/trade | Stress bps/trade | Median fold stress | Trades | Qualified |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1_ridge_price | 4 | ridge | price | 32.4530 | 50.4276 | -0.0025 | 0.0303 | 0 | — | — | 0.0000 | 0 | FAIL |
| h1_histogram_price | 4 | histogram | price | 32.4961 | 50.4976 | -0.0052 | 0.0021 | 0 | — | — | 0.0000 | 0 | FAIL |
| h1_extra_trees_price | 4 | extra_trees | price | 32.4760 | 50.4604 | -0.0038 | 0.0113 | 0 | — | — | 0.0000 | 0 | FAIL |
| h4_ridge_price | 16 | ridge | price | 66.5606 | 100.0169 | -0.0052 | 0.0228 | 0 | -13.1431 | -13.5317 | 0.0000 | 173 | FAIL |
| h4_histogram_price | 16 | histogram | price | 66.1659 | 99.9566 | -0.0040 | 0.0095 | 0 | — | — | 0.0000 | 0 | FAIL |
| h4_extra_trees_price | 16 | extra_trees | price | 66.1426 | 99.9293 | -0.0035 | 0.0100 | 0 | — | — | 0.0000 | 0 | FAIL |
| h4_extra_trees_oi | 16 | extra_trees | oi | 66.1831 | 99.9693 | -0.0043 | 0.0088 | 0 | -11.3659 | -11.6647 | 0.0000 | 118 | FAIL |
| h4_extra_trees_price_oi | 16 | extra_trees | price_oi | 66.1698 | 99.9619 | -0.0041 | 0.0107 | 0 | — | — | 0.0000 | 0 | FAIL |

## Selected candidate

| Item | Value |
| --- | --- |
| Candidate | h1_ridge_price |
| Horizon bars | 4 |
| Model | ridge |
| Feature set | price |
| Diagnostic only | PASS |

## Selected candidate folds

| Fold | Hurdle bps | Advantage bps | Trades | Net bps/trade | Stress bps/trade |
| --- | --- | --- | --- | --- | --- |
| 2024_summer | 0.0000 | 0.0000 | 0 | — | — |
| 2024_autumn | 0.0000 | 0.0000 | 0 | — | — |
| 2024_winter | 0.0000 | 0.0000 | 0 | — | — |
| 2025_spring | 0.0000 | 0.0000 | 0 | — | — |
| 2025_summer | 0.0000 | 0.0000 | 0 | — | — |
| 2025_autumn | 0.0000 | 0.0000 | 0 | — | — |

## Selected candidate regression diagnostics

| Fold | Long MAE | Long RMSE | Long R² | Long Spearman | Short MAE | Short RMSE | Short R² | Short Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024_summer | 35.7632 | 57.3512 | -0.0049 | 0.0176 | 35.7526 | 57.2821 | -0.0054 | 0.0215 |
| 2024_autumn | 33.9940 | 50.1295 | -0.0014 | — | 33.9889 | 50.1287 | -0.0010 | — |
| 2024_winter | 37.7275 | 58.1116 | -0.0083 | 0.0528 | 37.7299 | 58.0304 | -0.0041 | 0.0520 |
| 2025_spring | 34.3984 | 53.9213 | 0.0020 | 0.0571 | 34.3811 | 53.8997 | 0.0030 | 0.0635 |
| 2025_summer | 23.2253 | 34.1020 | 0.0015 | 0.0662 | 23.2120 | 34.0781 | 0.0027 | 0.0729 |
| 2025_autumn | 29.7591 | 45.1008 | -0.0077 | -0.0025 | 29.7082 | 45.0310 | -0.0063 | 0.0120 |

## Active-candidate fixed-action fee counterfactuals

| Candidate | Scenario | Round-trip fee bps | Trades | Net bps/trade | 95% lower | Profit factor |
| --- | --- | --- | --- | --- | --- | --- |
| h4_ridge_price | taker_10_bps | 10.0000 | 173 | -13.1431 | -32.2327 | 0.7863 |
| h4_ridge_price | hybrid_7_bps | 7.0000 | 173 | -10.1431 | -28.8359 | 0.8305 |
| h4_ridge_price | maker_4_bps | 4.0000 | 173 | -7.1431 | -26.9158 | 0.8773 |
| h4_ridge_price | zero_fee | 0.0000 | 173 | -3.1431 | -22.3395 | 0.9440 |
| h4_extra_trees_oi | taker_10_bps | 10.0000 | 118 | -11.3659 | -35.7930 | 0.7852 |
| h4_extra_trees_oi | hybrid_7_bps | 7.0000 | 118 | -8.3659 | -32.7987 | 0.8374 |
| h4_extra_trees_oi | maker_4_bps | 4.0000 | 118 | -5.3659 | -29.5238 | 0.8928 |
| h4_extra_trees_oi | zero_fee | 0.0000 | 118 | -1.3659 | -25.6868 | 0.9717 |

## Open-interest qualification

| Candidate | Paired wins | Required | Stressed bps/trade | Qualified |
| --- | --- | --- | --- | --- |
| h4_extra_trees_oi | 0 | 5 | -11.6647 | FAIL |
| h4_extra_trees_price_oi | 0 | 5 | — | FAIL |

## Development gates

| Gate | Pass | Actual | Required |
| --- | --- | --- | --- |
| Nominal positive fold stability | FAIL | 0 | 5 |
| Minimum pooled net expectancy | FAIL | — | 3.0000 |
| Cost stress positive fold stability | FAIL | 0 | 5 |
| Positive daily block bootstrap lower | FAIL | — | > 0 bps/trade |
| Minimum profit factor | FAIL | — | 1.1500 |
| Positive calendar month fraction | FAIL | 0.0000 | 0.6000 |
| Positive fold pnl concentration | FAIL | 1.0000 | <= 0.4 |

## Final pre-holdout confirmation

| Item | Value |
| --- | --- |
| Status | not_run |
| Passed | — |
| Net bps/trade | — |
| Stress bps/trade | — |

## Final confirmation gates

_No rows._

## Historical classifier control

| Item | Value |
| --- | --- |
| Role | historical_control_only |
| Run | 20260728T173754Z-3638569e-2b55a336 |
| Eligible for selection | FAIL |
| Development gates passed | FAIL |
| Holdout status | sealed_not_qualified |

## Holdout state

| Item | Value |
| --- | --- |
| Status | sealed_not_qualified |
| Opened | FAIL |
| Identity | fb5484783ce349d5787ebb898a35d44cbcd84f72b59029f4e3e51145f2cc3d9d |
| Reason | no candidate passed every development gate |

## Notes

- Long and short net returns are modeled separately and calibrated on a later chronological slice.
- Funding is excluded from model features and included in every realized target and economic ledger.
- Only the taker-fee result qualifies an edge; hybrid, maker, and zero-fee results are diagnostics.
- The locked holdout remains unopened unless both development and final confirmation gates pass.
- This preliminary benchmark assumes zero inference/routing latency at the next 15-minute open; a pass is not a deployable-edge claim until delayed-entry sensitivity is evaluated with 1-minute or L2 data.
