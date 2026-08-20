# BTC Five-Minute Middle-Market Ablation Tournament

Development decision: **holdout_not_ready**

Provisional champion: `none`

The independent holdout was not scored unless its frozen readiness contract passed.

## Development comparison (submitted profiles)

| Candidate | Qualified | Trades | Coverage | Accuracy | Net PnL | Stress PnL | PF | Avg entry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| payoff_control | no | 387 | 24.49% | 81.91% | -24.20 | -43.55 | 0.916 | 136.460 |
| chainlink_stratified_payoff | no | 280 | 17.72% | 80.00% | 60.97 | 46.97 | 1.293 | 124.893 |
| chainlink_oof_loss_veto | no | 189 | 11.96% | 81.48% | 27.72 | 18.27 | 1.207 | 123.889 |
| chainlink_regime_calibrated | no | 103 | 6.52% | 80.58% | 29.83 | 24.68 | 1.406 | 131.505 |
| chainlink_oi_modifier | no | 219 | 13.86% | 77.63% | 51.53 | 40.58 | 1.288 | 126.735 |
| chainlink_full_combined | no | 206 | 13.04% | 78.64% | 54.07 | 43.77 | 1.340 | 125.874 |
| chainlink_revised_wait_value | no | 65 | 4.11% | 80.00% | 10.26 | 7.01 | 1.213 | 169.308 |

## Independent holdout readiness

Coverage: 600/2304 (26.04%); passed: **False**.

## Limitations

- The independent holdout was not scored because its frozen readiness contract did not pass.
- August 2-10 was consumed by the prior tournament and is used only as development evidence.
- Projected PnL assumes recorded ask VWAP is fillable and does not model queue position.
- L2 and trade-print candidates are excluded until causal feature coverage satisfies the frozen requirement.
