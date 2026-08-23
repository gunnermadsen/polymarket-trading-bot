# BTC Five-Minute Full-Window Payoff Challenger Tournament

Status: **development comparison only; active paper soak remains independent**

Validation-preselected champion: `oof_expert_distilled_admission`

## Reused development comparison

| Candidate | Research gate | Trades | Coverage | Accuracy | Net PnL | Stress PnL | PF | Avg entry | Worst fold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| continuous_payoff_baseline | no | 1935 | 87.64% | 66.41% | -127.85 | -224.60 | 0.938 | 71.281 | -0.3350 |
| time_price_stratified_payoff | no | 675 | 30.57% | 74.96% | -15.81 | -49.56 | 0.973 | 105.877 | -0.3599 |
| chronological_loss_tail_guard | no | 528 | 23.91% | 80.11% | -43.78 | -70.18 | 0.895 | 106.438 | -0.4345 |
| stability_selected_regime | no | 1064 | 48.19% | 65.04% | -56.51 | -109.71 | 0.951 | 61.859 | -0.4091 |
| oof_expert_distilled_admission | no | 1185 | 53.67% | 65.32% | -11.34 | -70.59 | 0.991 | 73.844 | -0.3174 |

## Fixed-entry VWAP5 comparison

PnL includes fees and the frozen execution reserve; stress PnL adds $0.01 per share.

## Limitations

- The August 2 comparison block was consumed by prior work and is not an independent holdout.
- The active paper soak is intentionally excluded from training and selection.
- Projected PnL assumes recorded ask VWAP was fillable and does not model queue position.
- Open interest, Binance L2, trade prints, and TWAP were not admitted as mandatory full-window features.
- The OOF expert candidate distills chronological family-proxy agreement through the existing admission feature contract; it does not run an ensemble at inference.
