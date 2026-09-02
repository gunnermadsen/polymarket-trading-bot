# BTC 5m Frozen Champion Collection — 2026-09-02

This collection freezes the ten user-selected champions after removing two exact duplicate control aliases. It does not retrain, deploy, modify, or retag any model.

| # | Champion | Admission | Best bucket | PnL | Stress PnL | PF | Expectancy | Coverage | W/L | Win rate | Recovery | Avg entry | Brier |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `middle_specialist_refit` | Hybrid VWAP, no L2 | 150–180s | +$58.48 | +$19.48 | 1.193 | +$0.075 | 8.74% | 712/68 | 91.28% | 8.775 | 176.3s | 0.1718 |
| 2 | `middle_q5_admission` | Hybrid VWAP + dual L2 | 150–180s | +$31.89 | +$19.74 | 1.273 | +$0.131 | 2.72% | 216/27 | 88.89% | 6.284 | 175.7s | 0.1722 |
| 3 | `crossvenue_middle_specialist` | Hybrid VWAP + dual L2 | 150–180s | +$18.73 | +$3.13 | 1.112 | +$0.060 | 3.49% | 273/39 | 87.50% | 6.295 | 168.4s | 0.1738 |
| 4 | `middle_agreement_ensemble` | Hybrid VWAP + dual L2 | 150–180s | +$10.99 | -$9.76 | 1.055 | +$0.026 | 4.65% | 370/45 | 89.16% | 7.791 | 175.6s | 0.1724 |
| 5 | `price_time_calibrated_middle_ensemble` | Hybrid VWAP, no L2 | 150–180s | +$78.97 | +$34.47 | 1.238 | +$0.089 | 9.97% | 815/75 | 91.57% | 8.777 | 172.7s | 0.1722 |
| 6 | `extended_specialist_official` | Programmatic | 60–89s | +$39.88 | +$35.13 | 1.500 | +$0.420 | 5.50% | 70/25 | 73.68% | 1.866 | 68.7s | 0.1513 |
| 7 | `bridge_aware_specialist` | Programmatic | 60–89s | +$37.53 | +$18.78 | 1.154 | +$0.100 | 21.70% | 314/61 | 83.73% | 4.459 | 69.2s | 0.1512 |
| 8 | `official_vwap_admission` | Learned VWAP admission | 60–89s | +$49.87 | +$9.32 | 1.083 | +$0.061 | 23.92% | 655/156 | 80.76% | 3.876 | 66.6s | 0.1518 |
| 9 | `official_temporal_consensus` | Temporal-consensus veto | 60–89s | +$24.81 | +$20.91 | 1.349 | +$0.318 | 2.30% | 56/22 | 71.79% | 1.887 | 67.7s | 0.1518 |
| 10 | `official_high_precision_loss_veto` | ML loss veto | 60–89s | +$18.39 | +$15.99 | 1.401 | +$0.383 | 1.42% | 33/15 | 68.75% | 1.570 | 69.0s | 0.1518 |

## Replacements

- `official_specialist_control_60_89` was removed because it exactly replayed `extended_specialist_official`; `official_temporal_consensus` takes its collection slot.
- `bridge_aware_control_60_89` was removed because it exactly replayed `bridge_aware_specialist`; `official_high_precision_loss_veto` takes its collection slot.

## Artifact ownership

The ten champions are contained in four existing immutable tournament artifacts. Their exact paths, SHA-256 identities, recording commits, source model tags, evaluation windows, and complete-precision metrics are recorded in `manifest.json`.

The collection itself is a selection manifest, not a newly trained model. Existing model tags remain on their original recording commits and are not recreated or moved.
