# Binance-Context Latent TWAP Tournament

Run: `20260828T220039Z`
Source commit: `a5309b089601003c1372ccbc08f94863cbc1ff38`
Artifact SHA-256: `e138443d269fb02517cb7c218625b5a6e85a3b63aad76af7f0b8d014e9d3d709`
Development result: **no_candidate_qualified**
Prospective result: **not_applicable_no_development_candidate**

## Candidate results

| Candidate | Role | Trades | Coverage | Accuracy | UP / DOWN | Stressed PnL | Expectancy | PF | Avg entry | Max DD | CVaR | Qualified |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| twap_single_regime | frozen_latent_control | 496 | 13.52% | 68.15% | 266 / 230 | -15.2064 | -0.0307 | 0.9714 | 83.1956 | 79.1783 | -3.6902 | False |
| twap_regime_switching | frozen_latent_control | 419 | 11.42% | 68.02% | 230 / 189 | -7.4648 | -0.0178 | 0.9835 | 86.7184 | 77.8342 | -3.6913 | False |
| twap_refprice_single_regime | frozen_latent_control | 531 | 14.48% | 68.17% | 285 / 246 | -49.0391 | -0.0924 | 0.9162 | 76.1959 | 94.1424 | -3.7132 | False |
| twap_refprice_regime_switching | frozen_latent_control | 507 | 13.82% | 67.46% | 276 / 231 | -64.3715 | -0.1270 | 0.8873 | 77.5838 | 102.1175 | -3.7066 | False |
| twap_regime_kline_residual | binance_context_challenger | 647 | 17.64% | 67.85% | 357 / 290 | -75.0563 | -0.1160 | 0.8966 | 57.2875 | 124.7509 | -3.7211 | False |
| twap_regime_open_interest_residual | binance_context_challenger | 446 | 12.16% | 69.06% | 235 / 211 | -1.0939 | -0.0025 | 0.9977 | 72.0516 | 63.3268 | -3.7111 | False |
| twap_regime_kline_open_interest_residual | binance_context_challenger | 640 | 17.45% | 69.53% | 380 / 260 | -20.0421 | -0.0313 | 0.9706 | 58.9375 | 71.4928 | -3.7245 | False |
| twap_regime_l2_residual_research | binance_context_challenger | 0 | 0.00% | — | 0 / 0 | 0.0000 | — | — | — | 0.0000 | — | False |

## Binance source coverage

### kline

| Evidence block | Markets | Scheduled | Coverage | Rows |
|---|---:|---:|---:|---:|
| historical_fit | 14710 | 14710 | 100.00% | 367750 |
| calibration | 3423 | 3423 | 100.00% | 85575 |
| official_development | 3668 | 3668 | 100.00% | 91700 |
| prospective | 0 | 0 | 0.00% | 0 |

### open_interest

| Evidence block | Markets | Scheduled | Coverage | Rows |
|---|---:|---:|---:|---:|
| historical_fit | 7715 | 14710 | 52.45% | 192875 |
| calibration | 3423 | 3423 | 100.00% | 85575 |
| official_development | 3668 | 3668 | 100.00% | 91700 |
| prospective | 0 | 0 | 0.00% | 0 |

### kline_open_interest

| Evidence block | Markets | Scheduled | Coverage | Rows |
|---|---:|---:|---:|---:|
| historical_fit | 7715 | 14710 | 52.45% | 192875 |
| calibration | 3423 | 3423 | 100.00% | 85575 |
| official_development | 3668 | 3668 | 100.00% | 91700 |
| prospective | 0 | 0 | 0.00% | 0 |

### l2

| Evidence block | Markets | Scheduled | Coverage | Rows |
|---|---:|---:|---:|---:|
| historical_fit | 13509 | 14710 | 91.84% | 337241 |
| calibration | 256 | 3423 | 7.48% | 6370 |
| official_development | 0 | 3668 | 0.00% | 0 |
| prospective | 0 | 0 | 0.00% | 0 |

## Detailed performance

### twap_single_regime

- Wins/losses: 338 / 158; average win 1.5307; average loss -3.3708; wins to recover one average loss 2.2021.
- Worst loss -3.7456; loss percentiles `{"p25": -3.6485, "p5": -3.7455599999999993, "p50": -3.502385, "p75": -3.159, "p95": -2.7049702499999992}`.
- Gross/net/stressed PnL: 60.3689 / 9.5936 / -15.2064.
- Entry seconds mean/median/p10/p90: 83.1956 / 80.0000 / 40.0000 / 132.5000.
- Brier/log loss/ECE: 0.2113 / 0.6702 / 0.0674.
- Qualification failures: official_development_period_complete, accuracy, profit_factor, positive_stressed_pnl, positive_stressed_expectancy, positive_bootstrap_lower, profitable_folds, positive_vwap5_capacity.
- Per-direction, entry-band, price-band, daily, fold, bootstrap, and capacity results from 5 through 200 shares are retained in `metrics.json`.

### twap_regime_switching

- Wins/losses: 285 / 134; average win 1.5604; average loss -3.3744; wins to recover one average loss 2.1626.
- Worst loss -3.7456; loss percentiles `{"p25": -3.6485, "p5": -3.7455599999999993, "p50": -3.502385, "p75": -3.25746, "p95": -2.59492825}`.
- Gross/net/stressed PnL: 56.4844 / 13.4852 / -7.4648.
- Entry seconds mean/median/p10/p90: 86.7184 / 80.0000 / 45.0000 / 135.0000.
- Brier/log loss/ECE: 0.2150 / 0.6681 / 0.0681.
- Qualification failures: official_development_period_complete, accuracy, profit_factor, positive_stressed_pnl, positive_stressed_expectancy, positive_bootstrap_lower, profitable_folds, positive_vwap5_capacity.
- Per-direction, entry-band, price-band, daily, fold, bootstrap, and capacity results from 5 through 200 shares are retained in `metrics.json`.

### twap_refprice_single_regime

- Wins/losses: 362 / 169; average win 1.4804; average loss -3.4612; wins to recover one average loss 2.3380.
- Worst loss -3.7456; loss percentiles `{"p25": -3.6970649999999994, "p5": -3.7455599999999993, "p50": -3.55116, "p75": -3.3065849999999997, "p95": -2.78229}`.
- Gross/net/stressed PnL: 31.3794 / -22.4891 / -49.0391.
- Entry seconds mean/median/p10/p90: 76.1959 / 70.0000 / 35.0000 / 125.0000.
- Brier/log loss/ECE: 0.2065 / 0.6651 / 0.0686.
- Qualification failures: official_development_period_complete, accuracy, profit_factor, positive_stressed_pnl, positive_stressed_expectancy, positive_bootstrap_lower, profitable_folds, positive_vwap5_capacity, refprice_admission.
- Per-direction, entry-band, price-band, daily, fold, bootstrap, and capacity results from 5 through 200 shares are retained in `metrics.json`.

### twap_refprice_regime_switching

- Wins/losses: 342 / 165; average win 1.4818; average loss -3.4615; wins to recover one average loss 2.3360.
- Worst loss -3.7456; loss percentiles `{"p25": -3.6485, "p5": -3.7455599999999993, "p50": -3.55116, "p75": -3.3556399999999997, "p95": -2.8419960000000026}`.
- Gross/net/stressed PnL: 12.4189 / -39.0215 / -64.3715.
- Entry seconds mean/median/p10/p90: 77.5838 / 70.0000 / 40.0000 / 130.0000.
- Brier/log loss/ECE: 0.2078 / 0.6648 / 0.0689.
- Qualification failures: official_development_period_complete, accuracy, profit_factor, positive_stressed_pnl, positive_stressed_expectancy, positive_bootstrap_lower, profitable_folds, positive_vwap5_capacity, refprice_admission.
- Per-direction, entry-band, price-band, daily, fold, bootstrap, and capacity results from 5 through 200 shares are retained in `metrics.json`.

### twap_regime_kline_residual

- Wins/losses: 439 / 208; average win 1.4818; average loss -3.4883; wins to recover one average loss 2.3541.
- Worst loss -3.7456; loss percentiles `{"p25": -3.6970649999999994, "p5": -3.7455599999999993, "p50": -3.55116, "p75": -3.3556399999999997, "p95": -2.9785807500000008}`.
- Gross/net/stressed PnL: 22.9332 / -42.7063 / -75.0563.
- Entry seconds mean/median/p10/p90: 57.2875 / 45.0000 / 30.0000 / 115.0000.
- Brier/log loss/ECE: 0.1998 / 0.6349 / 0.0552.
- Qualification failures: official_development_period_complete, accuracy, profit_factor, positive_stressed_pnl, positive_stressed_expectancy, positive_bootstrap_lower, profitable_folds, pnl_day_concentration, positive_vwap5_capacity, maximum_drawdown_nonworse, binance_context_admission.
- Per-direction, entry-band, price-band, daily, fold, bootstrap, and capacity results from 5 through 200 shares are retained in `metrics.json`.

### twap_regime_open_interest_residual

- Wins/losses: 308 / 138; average win 1.5302; average loss -3.4231; wins to recover one average loss 2.2371.
- Worst loss -3.7456; loss percentiles `{"p25": -3.6849237499999994, "p5": -3.7455599999999993, "p50": -3.502385, "p75": -3.3065849999999997, "p95": -2.8047112499999995}`.
- Gross/net/stressed PnL: 66.7368 / 21.2061 / -1.0939.
- Entry seconds mean/median/p10/p90: 72.0516 / 65.0000 / 30.0000 / 130.0000.
- Brier/log loss/ECE: 0.2123 / 0.6564 / 0.0610.
- Qualification failures: official_development_period_complete, accuracy, profit_factor, positive_stressed_pnl, positive_stressed_expectancy, positive_bootstrap_lower, profitable_folds, positive_vwap5_capacity, binance_context_admission.
- Per-direction, entry-band, price-band, daily, fold, bootstrap, and capacity results from 5 through 200 shares are retained in `metrics.json`.

### twap_regime_kline_open_interest_residual

- Wins/losses: 445 / 195; average win 1.4871; average loss -3.4963; wins to recover one average loss 2.3512.
- Worst loss -3.7456; loss percentiles `{"p25": -3.6970649999999994, "p5": -3.7455599999999993, "p50": -3.55116, "p75": -3.3556399999999997, "p95": -2.96124}`.
- Gross/net/stressed PnL: 76.8825 / 11.9579 / -20.0421.
- Entry seconds mean/median/p10/p90: 58.9375 / 45.0000 / 30.0000 / 120.0000.
- Brier/log loss/ECE: 0.1994 / 0.6350 / 0.0541.
- Qualification failures: official_development_period_complete, accuracy, profit_factor, positive_stressed_pnl, positive_stressed_expectancy, positive_bootstrap_lower, profitable_folds, pnl_day_concentration, positive_vwap5_capacity, binance_context_admission, isolated_sources_admitted.
- Per-direction, entry-band, price-band, daily, fold, bootstrap, and capacity results from 5 through 200 shares are retained in `metrics.json`.

### twap_regime_l2_residual_research

- Wins/losses: 0 / 0; average win —; average loss —; wins to recover one average loss —.
- Worst loss —; loss percentiles `{}`.
- Gross/net/stressed PnL: 0.0000 / 0.0000 / 0.0000.
- Entry seconds mean/median/p10/p90: — / — / — / —.
- Brier/log loss/ECE: — / — / —.
- Qualification failures: official_development_period_complete, accuracy, profit_factor, positive_stressed_pnl, positive_stressed_expectancy, positive_bootstrap_lower, average_loss_recovery, every_quote_recovery, average_entry, profitable_folds, both_directions, pnl_day_concentration, positive_vwap5_capacity, cvar_nonworse, coverage, trade_support, binance_context_admission, not_research_only.
- Per-direction, entry-band, price-band, daily, fold, bootstrap, and capacity results from 5 through 200 shares are retained in `metrics.json`.

## Incremental Binance contribution

```json
{
  "twap_regime_kline_open_interest_residual": {
    "admitted": false,
    "brier_score_difference_challenger_minus_control": {
      "lower": -0.017621302714741084,
      "mean": -0.015593380472100215,
      "resamples": 2000,
      "upper": -0.01365672824844418
    },
    "candidate": "twap_regime_kline_open_interest_residual",
    "checks": {
      "brier_upper_below_zero": true,
      "calibration_not_degraded": true,
      "deployment_eligible_source": true,
      "drawdown_not_worse": true,
      "loss_recovery_not_worse": false,
      "multiple_positive_folds": true,
      "source_development_coverage": true,
      "source_evidence_available": true,
      "stressed_pnl_difference_lower_positive": false
    },
    "feature_family": "kline_open_interest",
    "fold_differences": {
      "official_20260814_15": 3.3065849999999997,
      "official_20260816_17": -3.748775,
      "official_20260818_19": -20.489124999999994,
      "official_20260820_21": 16.212890284080025,
      "official_20260822_23": -8.055884999999998,
      "official_20260824_25": 0.19703823374000407,
      "official_20260826_27": 0.0
    },
    "matching_control": "twap_regime_switching",
    "paired_markets": 3662,
    "research_only": false,
    "source_development_market_coverage": 1.0,
    "stressed_pnl_per_scheduled_market_difference": {
      "lower": -0.019386424236796148,
      "mean": -0.0034345361775477805,
      "resamples": 2000,
      "upper": 0.011559222299884774
    }
  },
  "twap_regime_kline_residual": {
    "admitted": false,
    "brier_score_difference_challenger_minus_control": {
      "lower": -0.01754637571862545,
      "mean": -0.01521657497907366,
      "resamples": 2000,
      "upper": -0.013007333180746114
    },
    "candidate": "twap_regime_kline_residual",
    "checks": {
      "brier_upper_below_zero": true,
      "calibration_not_degraded": true,
      "deployment_eligible_source": true,
      "drawdown_not_worse": false,
      "loss_recovery_not_worse": false,
      "multiple_positive_folds": false,
      "source_development_coverage": true,
      "source_evidence_available": true,
      "stressed_pnl_difference_lower_positive": false
    },
    "feature_family": "kline",
    "fold_differences": {
      "official_20260814_15": 3.3065849999999997,
      "official_20260816_17": -6.87185867624,
      "official_20260818_19": -20.144375,
      "official_20260820_21": -12.948118036159977,
      "official_20260822_23": -14.013019999999997,
      "official_20260824_25": -16.920739673519996,
      "official_20260826_27": 0.0
    },
    "matching_control": "twap_regime_switching",
    "paired_markets": 3662,
    "research_only": false,
    "source_development_market_coverage": 1.0,
    "stressed_pnl_per_scheduled_market_difference": {
      "lower": -0.030821202853192228,
      "mean": -0.018457544070431455,
      "resamples": 2000,
      "upper": -0.0074493172854514155
    }
  },
  "twap_regime_l2_residual_research": {
    "admitted": false,
    "candidate": "twap_regime_l2_residual_research",
    "checks": {
      "source_evidence_available": false
    },
    "feature_family": "l2",
    "matching_control": "twap_regime_switching",
    "paired_markets": 0,
    "research_only": true,
    "source_development_market_coverage": 0.0
  },
  "twap_regime_open_interest_residual": {
    "admitted": false,
    "brier_score_difference_challenger_minus_control": {
      "lower": -0.0040831802868237866,
      "mean": -0.002685242220488042,
      "resamples": 2000,
      "upper": -0.001332969660665854
    },
    "candidate": "twap_regime_open_interest_residual",
    "checks": {
      "brier_upper_below_zero": true,
      "calibration_not_degraded": true,
      "deployment_eligible_source": true,
      "drawdown_not_worse": true,
      "loss_recovery_not_worse": false,
      "multiple_positive_folds": true,
      "source_development_coverage": true,
      "source_evidence_available": true,
      "stressed_pnl_difference_lower_positive": false
    },
    "feature_family": "open_interest",
    "fold_differences": {
      "official_20260814_15": 0.0,
      "official_20260816_17": -2.7249699999999994,
      "official_20260818_19": 5.520709999999998,
      "official_20260820_21": 0.989207544960012,
      "official_20260822_23": 6.9656199999999995,
      "official_20260824_25": -4.37970857984,
      "official_20260826_27": 0.0
    },
    "matching_control": "twap_regime_switching",
    "paired_markets": 3662,
    "research_only": false,
    "source_development_market_coverage": 1.0,
    "stressed_pnl_per_scheduled_market_difference": {
      "lower": -0.007939125965076975,
      "mean": 0.0017397211810813774,
      "resamples": 2000,
      "upper": 0.010012524177424563
    }
  }
}
```

## Evidence and provenance boundaries

- Split audit passed: True.
- Context causality audit passed: True.
- August 14–27 is development evidence; August 28 onward remains untouched prospective evidence.
- No runtime model was exported and no trading process was changed.
- No data source, ingester, table, migration, or database row was created or changed.

## Limitations

- Synthetic Chainlink-reconstructed TWAP evidence was used only for historical fitting and cannot qualify a model.
- Projected economics assume the recorded executable ask VWAP was fillable at each selected checkpoint.
- Binance L2 history ends before official development and is research-only; no missing L2 state was imputed.
- Open-interest history begins July 3, so the OI residual was fitted only on its causally eligible historical rows.
- No model was deployed and no trading process, ingester, SQL source, table, migration, or database row was changed.
- The immutable extract ends at the August 28 freeze boundary, so no prospective market was read or scored.
- Official development evidence is incomplete: 13 of 14 UTC days and 3668 official markets were available.
- No development candidate passed every frozen qualification gate.
