# NYC temperature PMXT full-market tournament

Tournament `08709e6a-c4f0-4ccb-8ed5-71224be8578a` tested whether causal PMXT trade-flow features improve a regularized market-log-odds correction model and expose positive-expectancy entries across the full executable price curve. The run used 719 development rows over 71 event days and a sealed 56-row, five-day interval. All decisions used five-share Q5 execution economics, captured taker fees, and one cent of modeled slippage.

## Out-of-fold results

| Model | Log loss | Brier | Full-market trades | Net PnL | RODC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Market only | 0.285368 | 0.091593 | 0 | 0.0000 | n/a |
| Weather + market offset | 0.285568 | 0.091456 | 2 | -3.7644 | -100.0% |
| Market state without PMXT | 0.295383 | 0.095034 | 0 | 0.0000 | n/a |
| PMXT market state | 0.288351 | 0.092441 | 0 | 0.0000 | n/a |

The PMXT challenger did not qualify. Its event-block lower-confidence probabilities found no positive robust-edge out-of-fold entry at the predeclared zero-edge threshold, under either the 25-cent, 50-cent, or full-market policy. That is abstention, not evidence of positive expectancy. The weather/market control selected two contracts in the 25–50 cent band; both lost, for a maximum drawdown of 3.7644 and net PnL of -3.8657 under two-cent slippage stress.

## Sealed diagnostic

The PMXT challenger selected one 65-cent contract, which won: net PnL 1.7000, RODC 51.52%, and net PnL 1.6500 under two-cent slippage stress. The market-state ablation selected one 70-cent winner for net PnL 1.4500. Five sealed event days and one trade are not sufficient evidence of repeatable edge, so the sealed result does not override the negative out-of-fold qualification.

## Outcome

Research qualification: **not supported**. Production qualification: **false**. The artifact is retained as an immutable, training-only no-economic-champion result and was not deployed.

- Producing source revision: `f4196688e6dea6b32c581bfdea1dfc19905007e6`
- Artifact SHA-256: `3b398466a5782162d7d3f9c006f53b3d8efaffaa239c073b617710ae06433e89`
- Input SHA-256: `94750015968835e4a201bdd1a2f54e3fb9941ac8e07d70a75882a4a6125141fb`
