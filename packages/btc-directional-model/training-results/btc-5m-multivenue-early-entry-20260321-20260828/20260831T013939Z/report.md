# Multi-Venue Early-Entry Tournament

Run: `20260831T013939Z`
Qualification: **disqualified_artifact_deserialization_failure**
Nominated predictive model: **full_history_price_control**

Post-training validation failed: `tournament.joblib` cannot be deserialized because its
model dataclasses were serialized under `__main__`. The metrics and sealed ledgers below
remain valid evidence, but this model artifact is not reusable and must not be deployed.
No retraining or sealed rerun was performed.

## Sealed high-level results

| Candidate | PnL | Coverage | Wins | Losses | Win rate | Loss recovery wins | Brier | Profit factor | Avg cost | Avg entry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_history_price_control | -14.02 | 32.45% | 626 | 572 | 0.523 | 1.105 | 0.1617 | 0.990 | 0.503 | 105.7 |
| full_history_refprice_residual | -100.39 | 51.30% | 1111 | 783 | 0.587 | 1.487 | 0.1611 | 0.954 | 0.576 | 91.6 |
| binance_flow | -65.14 | 41.28% | 734 | 790 | 0.482 | 0.964 | 0.1640 | 0.964 | 0.469 | 99.2 |
| kraken_crossvenue | -51.66 | 32.80% | 678 | 533 | 0.560 | 1.320 | 0.1614 | 0.964 | 0.547 | 104.6 |
| settlement_aligned_oracle | -26.59 | 28.47% | 620 | 431 | 0.590 | 1.471 | 0.1609 | 0.978 | 0.574 | 106.1 |
| dual_venue_flow_agreement | -46.13 | 37.89% | 763 | 636 | 0.545 | 1.234 | 0.1616 | 0.972 | 0.531 | 99.5 |
| multivenue_consensus | -19.26 | 25.81% | 538 | 415 | 0.565 | 1.319 | 0.1642 | 0.983 | 0.547 | 102.4 |
| time_specialist_ensemble | 12.99 | 25.08% | 546 | 380 | 0.590 | 1.419 | 0.1616 | 1.012 | 0.566 | 124.4 |

## Integrity

- **FAILED:** immutable model artifact deserialization.
- Training and calibration use only markets ending before the sealed boundary.
- The August 14–28 seal was evaluated after candidate and policy selection.
- Optional-source absence does not remove otherwise usable core rows or markets.
- Kraken candles and prints are available only after their one-second bucket closes; Kraken L2 is excluded.
- No database writes, ingesters, source additions, tables, runtime exports, deployments, or trading-process changes were made.
- The prior immutable 60–180 champion is reported as a provenance reference and is not presented as directly comparable at 181–240 seconds.
