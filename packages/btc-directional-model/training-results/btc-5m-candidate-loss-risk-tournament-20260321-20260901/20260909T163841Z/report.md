# BTC Candidate Loss-Risk Tournament

Run: `20260909T163841Z`

## Contestants

| Contestant | Threshold | Trades | W/L | PnL | Stress PnL | PF | Exp./trade | Coverage | Avg entry | Avg cost | Max DD | Recovery wins/loss | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_risk | — | 2972 | 2497/475 | $33.70 | $-114.90 | 1.018 | $0.011 | 1.000 | 122.1s | $0.823 | $84.88 | 5.165 | — |
| logistic_economics | 0.195 | 2749 | 2348/401 | $20.36 | $-117.09 | 1.012 | $0.007 | 0.925 | 126.4s | $0.839 | $95.05 | 5.785 | 0.1347 |
| boosted_economics | 0.162 | 2674 | 2288/386 | $41.52 | $-92.18 | 1.026 | $0.016 | 0.900 | 128.2s | $0.839 | $88.11 | 5.777 | 0.1354 |
| boosted_market_context | 0.179 | 2823 | 2402/421 | $52.35 | $-88.80 | 1.030 | $0.019 | 0.950 | 124.9s | $0.833 | $90.45 | 5.538 | 0.1346 |
| boosted_recent_history | 0.201 | 2823 | 2406/417 | $74.88 | $-66.27 | 1.044 | $0.027 | 0.950 | 124.9s | $0.833 | $88.72 | 5.528 | 0.1340 |
| matched_confidence | 0.568 | 2823 | 2379/444 | $6.64 | $-134.51 | 1.004 | $0.002 | 0.950 | 124.9s | $0.828 | $84.88 | 5.338 | — |
| matched_edge | -0.023 | 2823 | 2368/455 | $33.97 | $-107.18 | 1.019 | $0.012 | 0.950 | 125.0s | $0.822 | $84.88 | 5.109 | — |

## Selection

Research winner: `boosted_recent_history` at loss probability `0.200580`.

This is trained and evaluated research evidence only. It is not deployed or admitted for runtime use.

## Champion comparison

| Champion | Baseline PnL | Risk PnL | Risk W/L | Coverage | Avg entry | PF | Exp./trade | Max DD | Recovery wins/loss | Avoided losses | Missed profit | Net risk value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| middle_specialist_refit | $-44.29 | $-44.29 | 389/51 | 1.000 | 174.1s | 0.806 | $-0.101 | $50.11 | 9.463 | $-0.00 | $0.00 | $0.00 |
| middle_q5_admission | $-5.71 | $-5.71 | 132/22 | 1.000 | 174.0s | 0.940 | $-0.037 | $16.72 | 6.384 | $-0.00 | $0.00 | $0.00 |
| crossvenue_middle_specialist | $-30.57 | $-30.57 | 169/33 | 1.000 | 166.0s | 0.784 | $-0.151 | $40.87 | 6.531 | $-0.00 | $0.00 | $0.00 |
| middle_agreement_ensemble | $-26.97 | $-26.97 | 268/38 | 1.000 | 175.3s | 0.840 | $-0.088 | $39.98 | 8.396 | $-0.00 | $0.00 | $0.00 |
| price_time_calibrated_middle_ensemble | $-29.24 | $-29.24 | 411/52 | 1.000 | 167.1s | 0.873 | $-0.063 | $39.82 | 9.050 | $-0.00 | $0.00 | $0.00 |
| extended_specialist_official | $39.88 | $33.77 | 44/11 | 0.579 | 67.6s | 1.972 | $0.614 | $6.83 | 2.029 | $44.98 | $51.09 | $-6.11 |
| bridge_aware_specialist | $37.53 | $34.43 | 312/61 | 0.995 | 69.1s | 1.142 | $0.092 | $33.32 | 4.480 | $-0.00 | $3.10 | $-3.10 |
| official_vwap_admission | $49.87 | $78.26 | 604/126 | 0.900 | 66.2s | 1.158 | $0.107 | $57.76 | 4.140 | $103.41 | $75.03 | $28.39 |
| official_temporal_consensus | $24.81 | $39.30 | 50/14 | 0.821 | 67.7s | 1.867 | $0.614 | $6.41 | 1.913 | $25.79 | $11.30 | $14.50 |
| official_high_precision_loss_veto | $18.39 | $25.89 | 27/9 | 0.750 | 69.3s | 1.960 | $0.719 | $7.41 | 1.530 | $18.90 | $11.40 | $7.50 |

## Leave-one-champion-out diagnostic

| Champion | Excluded from construction | Baseline PnL | Risk PnL | Coverage | Brier |
|---|---:|---:|---:|---:|---:|
| middle_specialist_refit | yes | $-44.29 | $-44.29 | 1.000 | 0.1043 |
| middle_q5_admission | yes | $-5.71 | $-5.71 | 1.000 | 0.1271 |
| crossvenue_middle_specialist | yes | $-30.57 | $-30.57 | 1.000 | 0.1427 |
| middle_agreement_ensemble | yes | $-26.97 | $-26.97 | 1.000 | 0.1113 |
| price_time_calibrated_middle_ensemble | yes | $-29.24 | $-29.24 | 1.000 | 0.1009 |
| extended_specialist_official | yes | $39.88 | $40.58 | 0.537 | 0.1947 |
| bridge_aware_specialist | yes | $37.53 | $35.88 | 0.997 | 0.1378 |
| official_vwap_admission | unseen alias | $49.87 | $78.26 | 0.900 | 0.1551 |
| official_temporal_consensus | unseen alias | $24.81 | $39.30 | 0.821 | 0.2088 |
| official_high_precision_loss_veto | unseen alias | $18.39 | $25.89 | 0.750 | 0.2282 |

## Published baseline reproduction

All ten prior champion ledgers reproduced their published trade count, W/L, and PnL within $0.00001 before risk evaluation.

Qualification finding: **research_not_qualified**. net_pnl_improved=pass, stress_pnl_improved=pass, max_drawdown_reduced=fail, avoided_losses_exceed_missed_profit=pass, coverage_floor=pass, beats_matched_controls=pass, not_single_champion=pass, not_single_bucket=pass, no_materially_destructive_champion=pass, no_materially_destructive_side=pass, no_materially_destructive_bucket=pass.

## Data limitations

- Causal OOF prediction ledgers begin April 1, so March 21-31 cannot contribute model-fitting rows.
- Construction uses all available chronological periods but a deterministic row stride caps memory; no date range is isolated.
- Exact base-admission ledgers are available only for August 14-25; August 26-31 lacks executable trade evidence.
- Three admission-layer champions share the extended specialist directional stream during construction; champion identity is not a model feature.
- The locked September replay is not opened because the frozen champion candidate/admission ledgers are not archived for that interval.
- Results are a pilot backtest and projected PnL assumes the archived five-share ask VWAP was fillable.
