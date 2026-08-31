# BTC 5m TWAP Conformal-Risk Admission Tournament

Run: `20260828T201718Z`  
Source commit: `0d03639eea6f2d0ebd75f3cf0557b139f0665fc9`  
Selected artifact: `c4bd1b3e75c8f3beb3eac5dd0c37b3c0d73eac69c9326f6ed0d29733ade8450b`  
Conclusion: **upstream_predictor_lacks_sufficient_resolution_for_safe_admission**  
Deployment: **not deployed; training only**

## Development candidate comparison

| Candidate | Trades | Coverage | Accuracy | UP | DOWN | Stressed PnL | Expectancy | PF | Bootstrap lower | Avg entry | Qualified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| frozen_probability_control | 218 | 19.36% | 69.27% | 70.94% | 67.33% | 3.5388 | 0.0162 | 1.0156 | -0.3306 | 69.4037 | False |
| global_market_block_conformal | 0 | 0.00% | n/a | n/a | n/a | 0.0000 | n/a | n/a | n/a | n/a | False |
| direction_price_block_conformal | 0 | 0.00% | n/a | n/a | n/a | 0.0000 | n/a | n/a | n/a | n/a | False |
| direction_time_price_block_conformal | 0 | 0.00% | n/a | n/a | n/a | 0.0000 | n/a | n/a | n/a | n/a | False |

## Untouched test

Candidate: `global_market_block_conformal`  
Qualification: **unqualified_on_untouched_test**  
Trades/coverage: 0 / 0.00%  
Accuracy overall/UP/DOWN: n/a / n/a / n/a  
Gross/net/stressed PnL: 0.0000 / 0.0000 / 0.0000  
Expectancy/profit factor: n/a / n/a  
Bootstrap 95% interval: `{"lower": null, "median": null, "resamples": 2000, "upper": null}`  
Failed gates: coverage, trade_support, overall_accuracy, up_accuracy, down_accuracy, positive_stressed_pnl, positive_stressed_expectancy, profit_factor, positive_bootstrap_lower, average_loss_recovery, every_quote_recovery, average_entry, positive_each_two_day_fold, both_directions_represented, direction_economics, pnl_concentration, positive_five_share_capacity, empirical_conformal_coverage

## Full metric inventory

`metrics.json` contains every candidate's predictive scores, conformal coverage, risk-bound calibration, interval widths, loss distribution, economics, bootstrap intervals, drawdown, CVaR, entries, day/fold/direction/band breakdowns, calibration-cell support, fallback counts, abstention reasons, 5–200 share capacity, concentration, and paired control comparison.

## Integrity

- Calibration, development, and untouched test market IDs are disjoint and chronological.
- The untouched test was loaded only after candidate selection and artifact freezing.
- The upstream predictor was not refit; raw RefPrice was excluded from admission.
- No database, data source, ingester, table, trading process, or runtime service was changed.
