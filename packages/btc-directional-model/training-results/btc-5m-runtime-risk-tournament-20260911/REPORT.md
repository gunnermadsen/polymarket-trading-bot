# Runtime risk tournament

Fit: 283,138 rows · calibration: 1,402 rows · pristine test: 6,590 rows

| Risk package | Strategy model | Bucket | Trades | W/L | PnL | Delta PnL | Coverage | PF | Recovery | Max DD | Avg entry | Bad blocked | Good blocked | Alignment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| btc-5m-risk-candidate-loss-official-vwap-60-89-c95-20260911 | btc-5m-official-vwap-admission-umr-20260902 | 60–89s | 630 | 514/116 | $+80.58 | $+8.70 | 95.7% | 1.184 | 3.742 | $44.27 | 66.4s | 4.9% | 4.1% | 1.20x |
| btc-5m-risk-candidate-loss-temporal-consensus-60-89-c70-20260911 | btc-5m-official-temporal-consensus-umr-20260902-confidence-075 | 60–89s | 49 | 35/14 | $+18.60 | $+1.56 | 76.6% | 1.419 | 1.762 | $7.10 | 68.1s | 26.3% | 22.2% | 1.18x |
| btc-5m-risk-candidate-loss-specialist-distilled-90-119-c70-20260911 | btc-5m-specialist-distilled-fair-value-paper-20260823-v1 | 90–119s | 116 | 92/24 | $+6.10 | $-19.44 | 81.7% | 1.066 | 3.597 | $20.06 | 98.5s | 0.0% | 22.0% | 0.00x |
| btc-5m-risk-candidate-loss-extended-specialist-120-149-c85-20260911 | btc-5m-extended-specialist-official-umr-20260902-confidence-070 | 120–149s | 57 | 33/24 | $-6.05 | $-0.53 | 82.6% | 0.914 | 1.504 | $15.13 | 128.2s | 14.3% | 19.5% | 0.73x |
| btc-5m-risk-candidate-loss-extended-specialist-180-240-c85-20260911 | btc-5m-extended-specialist-official-umr-20260902-confidence-070 | 180–240s | 32 | 21/11 | $+0.25 | $-2.07 | 88.9% | 1.007 | 1.896 | $10.16 | 180.0s | 8.3% | 12.5% | 0.67x |

PnL, profit factor, recovery, drawdown and coverage describe the trading strategy after applying the named risk package. Delta PnL is the change from that strategy/time-bucket no-risk baseline. All 150 risk-package × strategy-model × time-bucket rows are in `all-results.parquet` and `all-results.json`.
