# BTC 5m TWAP60 challenger tournament

Run: `20260826T021146Z`
Source commit: `79713a7d9eecb3d87a9171cfc1bfebe785291bc6`
Frozen watermark: `2026-08-25T00:00:00+00:00`

This is consumed development evidence. No deployment, trading-process change, database mutation, new table, new ingester, or new data source occurred.

## Proxy-TWAP60 validation

Convention: `source_timestamp`; error band: 0.5260 bps; proxy transfer admitted: True.

## Feature bake-off

Selected treatment: `refprice_path` — one refprice treatment passed every reproducibility gate.

## Tournament

| Candidate | Trades | Coverage | Accuracy | Stressed PnL | Expectancy | Profit factor | CVaR | Avg entry | Qualified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `frozen_chainlink_stratified_payoff` | 583 | 18.42% | 70.33% | -113.3493 | -0.1944 | 0.812 | -4.0150 | 121.15 | False |
| `twap60_native_stratified` | 783 | 24.74% | 76.88% | -144.1774 | -0.1841 | 0.795 | -4.2043 | 109.81 | False |
| `twap60_proxy_transfer_stratified` | 931 | 29.42% | 75.19% | -188.8651 | -0.2029 | 0.780 | -4.2951 | 106.51 | False |
| `twap60_residual_adapter_stratified` | 924 | 29.19% | 77.16% | -137.8139 | -0.1491 | 0.826 | -4.2644 | 106.41 | False |
| `twap60_similarity_weighted_stratified` | 857 | 27.08% | 77.13% | -132.9404 | -0.1551 | 0.819 | -4.2335 | 102.49 | False |
| `twap60_loss_tail_guard_stratified` | 897 | 28.34% | 76.03% | -171.4372 | -0.1911 | 0.787 | -4.3000 | 106.81 | False |

Selection status: `no_deployable_challenger_qualified`.
Provisional challenger: `twap60_similarity_weighted_stratified`.

All data through August 24 is consumed development evidence; an independently validated champion cannot be named from this run.

## Limitations

- All data through August 24 is consumed development evidence; no independent holdout exists.
- August 2-10 core feature coverage is incomplete and is reported rather than imputed.
- Projected PnL assumes recorded executable ask VWAP was fillable at the sampled time.
- No model was deployed and no trading process was changed.
