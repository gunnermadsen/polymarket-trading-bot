# Latent TWAP Settlement-Risk Tournament

Run: `20260828T174703Z`
Source commit: `f05e55772bd418688fa003266f0d7c65bff76f5d`
Artifact SHA-256: `18b35a8cec12cfc10dbbd002fbe38eaa16108ee10edc15097f3831557dbe4e80`
Development result: **no_candidate_qualified**
Prospective result: **not_applicable_no_development_candidate**

## Candidate results

| Candidate | Trades | Coverage | Accuracy | UP / DOWN | Stressed PnL | Expectancy | Profit factor | Avg entry | Max DD | CVaR | Qualified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| twap_single_regime | 496 | 13.52% | 68.15% | 266 / 230 | -15.2064 | -0.0307 | 0.9714 | 83.1956 | 79.1783 | -3.6902 | False |
| twap_regime_switching | 419 | 11.42% | 68.02% | 230 / 189 | -7.4648 | -0.0178 | 0.9835 | 86.7184 | 77.8342 | -3.6913 | False |
| twap_refprice_single_regime | 531 | 14.48% | 68.17% | 285 / 246 | -49.0391 | -0.0924 | 0.9162 | 76.1959 | 94.1424 | -3.7132 | False |
| twap_refprice_regime_switching | 507 | 13.82% | 67.46% | 276 / 231 | -64.3715 | -0.1270 | 0.8873 | 77.5838 | 102.1175 | -3.7066 | False |

## Detailed performance

### twap_single_regime

- Wins/losses: 338 / 158; average win 1.5307; average loss -3.3708; wins to recover one average loss 2.2021.
- Worst loss -3.7456; loss percentiles `{"p25": -3.6485, "p5": -3.7455599999999993, "p50": -3.502385, "p75": -3.159, "p95": -2.7049702499999992}`.
- Gross/net/stressed PnL: 60.3689 / 9.5936 / -15.2064.
- Entry seconds mean/median/p10/p90: 83.1956 / 80.0000 / 40.0000 / 132.5000.
- Brier/log loss/ECE: 0.2113 / 0.6702 / 0.0674.
- Positive-fold ratio 28.57%; maximum positive-PnL day share 41.40%.
- Qualification failures: official_development_period_complete, accuracy, profit_factor, positive_stressed_pnl, positive_stressed_expectancy, positive_bootstrap_lower, profitable_folds, positive_vwap5_capacity.

Direction metrics:

```json
{
  "DOWN": {
    "accuracy": 0.6695652173913044,
    "average_loss": -3.4550663157894737,
    "average_win": 1.5184225974025976,
    "losses": 76,
    "mean_entry_second": 84.43478260869566,
    "stressed_expectancy": -0.12499113043478241,
    "stressed_pnl": -28.747959999999953,
    "trades": 230,
    "wins": 154
  },
  "UP": {
    "accuracy": 0.6917293233082706,
    "average_loss": -3.292696890243902,
    "average_win": 1.5409928032659785,
    "losses": 82,
    "mean_entry_second": 82.12406015037594,
    "stressed_expectancy": 0.05090801052984983,
    "stressed_pnl": 13.541530800940054,
    "trades": 266,
    "wins": 184
  }
}
```

Entry bands, price bands, daily/fold results, and capacity from 5 through 200 shares are retained in `metrics.json`.

### twap_regime_switching

- Wins/losses: 285 / 134; average win 1.5604; average loss -3.3744; wins to recover one average loss 2.1626.
- Worst loss -3.7456; loss percentiles `{"p25": -3.6485, "p5": -3.7455599999999993, "p50": -3.502385, "p75": -3.25746, "p95": -2.59492825}`.
- Gross/net/stressed PnL: 56.4844 / 13.4852 / -7.4648.
- Entry seconds mean/median/p10/p90: 86.7184 / 80.0000 / 45.0000 / 135.0000.
- Brier/log loss/ECE: 0.2150 / 0.6681 / 0.0681.
- Positive-fold ratio 57.14%; maximum positive-PnL day share 49.08%.
- Qualification failures: official_development_period_complete, accuracy, profit_factor, positive_stressed_pnl, positive_stressed_expectancy, positive_bootstrap_lower, profitable_folds, positive_vwap5_capacity.

Direction metrics:

```json
{
  "DOWN": {
    "accuracy": 0.6613756613756614,
    "average_loss": -3.4541126562499995,
    "average_win": 1.5476137297880002,
    "losses": 64,
    "mean_entry_second": 88.01587301587301,
    "stressed_expectancy": -0.146092559664021,
    "stressed_pnl": -27.611493776499966,
    "trades": 189,
    "wins": 125
  },
  "UP": {
    "accuracy": 0.6956521739130435,
    "average_loss": -3.301504642857143,
    "average_win": 1.570325208905875,
    "losses": 70,
    "mean_entry_second": 85.65217391304348,
    "stressed_expectancy": 0.08759438445626103,
    "stressed_pnl": 20.146708424940037,
    "trades": 230,
    "wins": 160
  }
}
```

Entry bands, price bands, daily/fold results, and capacity from 5 through 200 shares are retained in `metrics.json`.

### twap_refprice_single_regime

- Wins/losses: 362 / 169; average win 1.4804; average loss -3.4612; wins to recover one average loss 2.3380.
- Worst loss -3.7456; loss percentiles `{"p25": -3.6970649999999994, "p5": -3.7455599999999993, "p50": -3.55116, "p75": -3.3065849999999997, "p95": -2.78229}`.
- Gross/net/stressed PnL: 31.3794 / -22.4891 / -49.0391.
- Entry seconds mean/median/p10/p90: 76.1959 / 70.0000 / 35.0000 / 125.0000.
- Brier/log loss/ECE: 0.2065 / 0.6651 / 0.0686.
- Positive-fold ratio 28.57%; maximum positive-PnL day share 39.79%.
- Qualification failures: official_development_period_complete, accuracy, profit_factor, positive_stressed_pnl, positive_stressed_expectancy, positive_bootstrap_lower, profitable_folds, positive_vwap5_capacity, refprice_admission.

Direction metrics:

```json
{
  "DOWN": {
    "accuracy": 0.6463414634146342,
    "average_loss": -3.4755209770114948,
    "average_win": 1.472348388761258,
    "losses": 87,
    "mean_entry_second": 77.6829268292683,
    "stressed_expectancy": -0.27750785035349573,
    "stressed_pnl": -68.26693118695995,
    "trades": 246,
    "wins": 159
  },
  "UP": {
    "accuracy": 0.712280701754386,
    "average_loss": -3.445957073170731,
    "average_win": 1.4866814857215762,
    "losses": 82,
    "mean_entry_second": 74.91228070175438,
    "stressed_expectancy": 0.06746618105782476,
    "stressed_pnl": 19.227861601480058,
    "trades": 285,
    "wins": 203
  }
}
```

Entry bands, price bands, daily/fold results, and capacity from 5 through 200 shares are retained in `metrics.json`.

### twap_refprice_regime_switching

- Wins/losses: 342 / 165; average win 1.4818; average loss -3.4615; wins to recover one average loss 2.3360.
- Worst loss -3.7456; loss percentiles `{"p25": -3.6485, "p5": -3.7455599999999993, "p50": -3.55116, "p75": -3.3556399999999997, "p95": -2.8419960000000026}`.
- Gross/net/stressed PnL: 12.4189 / -39.0215 / -64.3715.
- Entry seconds mean/median/p10/p90: 77.5838 / 70.0000 / 40.0000 / 130.0000.
- Brier/log loss/ECE: 0.2078 / 0.6648 / 0.0689.
- Positive-fold ratio 28.57%; maximum positive-PnL day share 42.67%.
- Qualification failures: official_development_period_complete, accuracy, profit_factor, positive_stressed_pnl, positive_stressed_expectancy, positive_bootstrap_lower, profitable_folds, positive_vwap5_capacity, refprice_admission.

Direction metrics:

```json
{
  "DOWN": {
    "accuracy": 0.645021645021645,
    "average_loss": -3.4998049390243904,
    "average_win": 1.4749281543624164,
    "losses": 82,
    "mean_entry_second": 78.78787878787878,
    "stressed_expectancy": -0.2909944155844153,
    "stressed_pnl": -67.21970999999994,
    "trades": 231,
    "wins": 149
  },
  "UP": {
    "accuracy": 0.6992753623188406,
    "average_loss": -3.4236063855421683,
    "average_win": 1.4870856777250776,
    "losses": 83,
    "mean_entry_second": 76.57608695652173,
    "stressed_expectancy": 0.010319586235290064,
    "stressed_pnl": 2.8482058009400575,
    "trades": 276,
    "wins": 193
  }
}
```

Entry bands, price bands, daily/fold results, and capacity from 5 through 200 shares are retained in `metrics.json`.

## RefPrice contribution

RefPrice does not provide qualified incremental value under the current TWAP settlement mechanism.

```json
{
  "regime_switching": {
    "admitted": false,
    "brier_score_difference_refprice_minus_twap": {
      "lower": -0.008324572921646942,
      "mean": -0.007187490626097942,
      "resamples": 2000,
      "upper": -0.00600316160901642
    },
    "checks": {
      "both_directions_improve": false,
      "brier_upper_below_zero": true,
      "calibration_not_degraded": false,
      "drawdown_not_worse": false,
      "loss_recovery_not_worse": false,
      "multiple_positive_folds": false,
      "stressed_expectancy_lower_positive": false
    },
    "direction_differences": {
      "DOWN": -39.608216223499994,
      "UP": -17.298502624000015
    },
    "fold_differences": {
      "official_20260814_15": 0.0,
      "official_20260816_17": -4.467509999999999,
      "official_20260818_19": -11.480769999999998,
      "official_20260820_21": -11.517682623999988,
      "official_20260822_23": 6.948035000000004,
      "official_20260824_25": -36.38879122350001,
      "official_20260826_27": 0.0
    },
    "matching_twap_candidate": "twap_regime_switching",
    "paired_markets": 3662,
    "refprice_candidate": "twap_refprice_regime_switching",
    "stressed_pnl_per_scheduled_market_difference": {
      "lower": -0.03147334739505853,
      "mean": -0.015539792148416163,
      "resamples": 2000,
      "upper": -0.0011825407071641353
    }
  },
  "single_regime": {
    "admitted": false,
    "brier_score_difference_refprice_minus_twap": {
      "lower": -0.005300951259451618,
      "mean": -0.004768009393548643,
      "resamples": 2000,
      "upper": -0.004226729627320746
    },
    "checks": {
      "both_directions_improve": false,
      "brier_upper_below_zero": true,
      "calibration_not_degraded": false,
      "drawdown_not_worse": false,
      "loss_recovery_not_worse": false,
      "multiple_positive_folds": true,
      "stressed_expectancy_lower_positive": false
    },
    "direction_differences": {
      "DOWN": -39.518971186960016,
      "UP": 5.686330800540002
    },
    "fold_differences": {
      "official_20260814_15": 0.0,
      "official_20260816_17": 1.21583,
      "official_20260818_19": -9.51236872546,
      "official_20260820_21": -9.681239999999997,
      "official_20260822_23": 2.83779381304,
      "official_20260824_25": -18.692655474,
      "official_20260826_27": 0.0
    },
    "matching_twap_candidate": "twap_single_regime",
    "paired_markets": 3662,
    "refprice_candidate": "twap_refprice_single_regime",
    "stressed_pnl_per_scheduled_market_difference": {
      "lower": -0.01955020057964459,
      "mean": -0.009238842268274166,
      "resamples": 2000,
      "upper": -0.0003737135875224864
    }
  }
}
```

## Evidence separation

```json
{
  "authentic_counterfactual_calibration": {
    "evidentiary_role": "selection and calibration only; cannot qualify",
    "label_sources": [
      "authentic_counterfactual_twap60"
    ],
    "markets": 3423,
    "rows": 85575
  },
  "official_development": {
    "evidentiary_role": "official development qualification",
    "label_sources": [
      "authentic_official_twap60"
    ],
    "markets": 3668,
    "rows": 91700
  },
  "prospective": {
    "evidentiary_role": "untouched prospective qualification only",
    "label_sources": [],
    "markets": 0,
    "rows": 0
  },
  "synthetic_chainlink_historical_fit": {
    "evidentiary_role": "historical fitting only; cannot qualify",
    "label_sources": [
      "chainlink_reconstructed_twap60"
    ],
    "markets": 14710,
    "rows": 367750
  }
}
```

## Limitations

- Synthetic Chainlink-reconstructed TWAP evidence was used only for historical fitting and did not qualify a candidate.
- Projected economics assume the recorded executable ask VWAP was fillable at each selected checkpoint.
- No model was deployed and no trading process, ingester, SQL source, table, migration, or database row was changed.
- The immutable extract ends at the 2026-08-28 freeze boundary, so no prospective market was read or scored.
- Official development evidence is incomplete: 13 of 14 UTC days and 3668 official markets were available.
- No development candidate passed every frozen qualification gate.
