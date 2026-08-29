# BTC 5m TWAP Conformal-Risk Admission Tournament

Run: `20260828T205354Z`  
Source commit: `91205b064acd64e4a2109e8534232e91b5a95f90`  
Selected artifact: `2131c92dec045ac01fb8cae1be8eb10509162d2a3673e348ea9d32fa35ca3b35`  
Conclusion: **upstream_predictor_lacks_sufficient_resolution_for_safe_admission**  
Deployment: **not deployed; training only**

## Calibration

| Candidate | Market blocks | Correctness coverage | Margin coverage | Mean interval width (bps) |
|---|---:|---:|---:|---:|
| frozen_probability_control | 895 | 52.29% | 80.67% | 10.6639 |
| global_market_block_conformal | 895 | 90.17% | 90.17% | 15.9260 |
| direction_price_block_conformal | 2286 | 90.51% | 90.51% | 15.6001 |
| direction_time_price_block_conformal | 4251 | 90.80% | 90.57% | 14.4743 |

### Complete calibration records

```json
{
  "direction_price_block_conformal": {
    "candidate": "direction_price_block_conformal",
    "cells": {
      "direction_price": {
        "DOWN|0.60-0.70": {
          "margin_residual_quantile_bps": 2.2238276036250184,
          "probability_residual_quantile": 0.6096650185786308,
          "support_markets": 236
        },
        "DOWN|0.70-0.80": {
          "margin_residual_quantile_bps": 2.448637686513748,
          "probability_residual_quantile": 0.6522241156178297,
          "support_markets": 252
        },
        "DOWN|above_0.80": {
          "margin_residual_quantile_bps": 4.245664529805466,
          "probability_residual_quantile": 0.6781641293986702,
          "support_markets": 218
        },
        "DOWN|below_0.60": {
          "margin_residual_quantile_bps": 2.949256794401416,
          "probability_residual_quantile": 0.6136828105328482,
          "support_markets": 245
        },
        "UP|0.60-0.70": {
          "margin_residual_quantile_bps": 1.7800747296868078,
          "probability_residual_quantile": 0.599392486727401,
          "support_markets": 324
        },
        "UP|0.70-0.80": {
          "margin_residual_quantile_bps": 2.2264380386316613,
          "probability_residual_quantile": 0.6237817627411949,
          "support_markets": 285
        },
        "UP|above_0.80": {
          "margin_residual_quantile_bps": 3.7510340668569846,
          "probability_residual_quantile": 0.5881756604587833,
          "support_markets": 232
        },
        "UP|below_0.60": {
          "margin_residual_quantile_bps": 0.911366823834822,
          "probability_residual_quantile": 0.5639318505421834,
          "support_markets": 494
        }
      },
      "global": {
        "global": {
          "margin_residual_quantile_bps": 2.631058047519308,
          "probability_residual_quantile": 0.6226618441034003,
          "support_markets": 895
        }
      }
    },
    "coverage": {
      "empirical_conformal_coverage": 0.905074365704287,
      "error_risk_bound_calibration": {
        "actual_error_rate": 0.34951881014873143,
        "bound_minus_actual": 0.6250277870162585,
        "mean_error_risk_upper_bound": 0.9745465971649899
      },
      "margin_interval_coverage": 0.905074365704287,
      "market_blocks": 2286,
      "mean_margin_interval_width_bps": 15.600135175235145,
      "median_margin_interval_width_bps": 15.153684006099734,
      "p90_margin_interval_width_bps": 18.407741979757635
    }
  },
  "direction_time_price_block_conformal": {
    "candidate": "direction_time_price_block_conformal",
    "cells": {
      "direction_price": {
        "DOWN|0.60-0.70": {
          "margin_residual_quantile_bps": 2.2238276036250184,
          "probability_residual_quantile": 0.6096650185786308,
          "support_markets": 236
        },
        "DOWN|0.70-0.80": {
          "margin_residual_quantile_bps": 2.448637686513748,
          "probability_residual_quantile": 0.6522241156178297,
          "support_markets": 252
        },
        "DOWN|above_0.80": {
          "margin_residual_quantile_bps": 4.245664529805466,
          "probability_residual_quantile": 0.6781641293986702,
          "support_markets": 218
        },
        "DOWN|below_0.60": {
          "margin_residual_quantile_bps": 2.949256794401416,
          "probability_residual_quantile": 0.6136828105328482,
          "support_markets": 245
        },
        "UP|0.60-0.70": {
          "margin_residual_quantile_bps": 1.7800747296868078,
          "probability_residual_quantile": 0.599392486727401,
          "support_markets": 324
        },
        "UP|0.70-0.80": {
          "margin_residual_quantile_bps": 2.2264380386316613,
          "probability_residual_quantile": 0.6237817627411949,
          "support_markets": 285
        },
        "UP|above_0.80": {
          "margin_residual_quantile_bps": 3.7510340668569846,
          "probability_residual_quantile": 0.5881756604587833,
          "support_markets": 232
        },
        "UP|below_0.60": {
          "margin_residual_quantile_bps": 0.911366823834822,
          "probability_residual_quantile": 0.5639318505421834,
          "support_markets": 494
        }
      },
      "direction_time_price": {
        "DOWN|30-59|0.60-0.70": {
          "margin_residual_quantile_bps": 1.6176138215902576,
          "probability_residual_quantile": 0.5904791508503044,
          "support_markets": 149
        },
        "DOWN|30-59|0.70-0.80": {
          "margin_residual_quantile_bps": 2.8533369181412795,
          "probability_residual_quantile": 0.6570223481081465,
          "support_markets": 128
        },
        "DOWN|30-59|below_0.60": {
          "margin_residual_quantile_bps": 1.7846879819469414,
          "probability_residual_quantile": 0.5851512627756011,
          "support_markets": 150
        },
        "DOWN|60-89|0.60-0.70": {
          "margin_residual_quantile_bps": 1.6786677969296804,
          "probability_residual_quantile": 0.6039931292679811,
          "support_markets": 136
        },
        "DOWN|60-89|0.70-0.80": {
          "margin_residual_quantile_bps": 1.620710730641024,
          "probability_residual_quantile": 0.6789204141191639,
          "support_markets": 143
        },
        "DOWN|60-89|above_0.80": {
          "margin_residual_quantile_bps": 3.996217984058605,
          "probability_residual_quantile": 0.6475858573188265,
          "support_markets": 152
        },
        "DOWN|60-89|below_0.60": {
          "margin_residual_quantile_bps": 3.56284214877245,
          "probability_residual_quantile": 0.6034432406128751,
          "support_markets": 146
        },
        "DOWN|90-120|0.60-0.70": {
          "margin_residual_quantile_bps": 1.8101772382506063,
          "probability_residual_quantile": 0.6096650185786308,
          "support_markets": 125
        },
        "DOWN|90-120|0.70-0.80": {
          "margin_residual_quantile_bps": 2.054244517588619,
          "probability_residual_quantile": 0.6169711340088266,
          "support_markets": 146
        },
        "DOWN|90-120|above_0.80": {
          "margin_residual_quantile_bps": 1.6984976015922513,
          "probability_residual_quantile": 0.6375950609823925,
          "support_markets": 189
        },
        "DOWN|90-120|below_0.60": {
          "margin_residual_quantile_bps": 3.152216385213853,
          "probability_residual_quantile": 0.6394091617132424,
          "support_markets": 147
        },
        "UP|30-59|0.60-0.70": {
          "margin_residual_quantile_bps": 2.4350125016772735,
          "probability_residual_quantile": 0.5833612000556021,
          "support_markets": 202
        },
        "UP|30-59|0.70-0.80": {
          "margin_residual_quantile_bps": 2.404200506367851,
          "probability_residual_quantile": 0.6121811701968034,
          "support_markets": 169
        },
        "UP|30-59|below_0.60": {
          "margin_residual_quantile_bps": 0.0,
          "probability_residual_quantile": 0.528200099018194,
          "support_markets": 409
        },
        "UP|60-89|0.60-0.70": {
          "margin_residual_quantile_bps": 1.56283915170218,
          "probability_residual_quantile": 0.6251869272626155,
          "support_markets": 196
        },
        "UP|60-89|0.70-0.80": {
          "margin_residual_quantile_bps": 1.9020197715780671,
          "probability_residual_quantile": 0.6301911135198206,
          "support_markets": 176
        },
        "UP|60-89|above_0.80": {
          "margin_residual_quantile_bps": 4.4089045339818895,
          "probability_residual_quantile": 0.5729661591578713,
          "support_markets": 138
        },
        "UP|60-89|below_0.60": {
          "margin_residual_quantile_bps": 0.5737372181613871,
          "probability_residual_quantile": 0.562965260318655,
          "support_markets": 337
        },
        "UP|90-120|0.60-0.70": {
          "margin_residual_quantile_bps": 0.5244261798567305,
          "probability_residual_quantile": 0.5741479064487681,
          "support_markets": 162
        },
        "UP|90-120|0.70-0.80": {
          "margin_residual_quantile_bps": 0.559640241548951,
          "probability_residual_quantile": 0.5925208347061035,
          "support_markets": 166
        },
        "UP|90-120|above_0.80": {
          "margin_residual_quantile_bps": 3.3683907418176293,
          "probability_residual_quantile": 0.5484714533922073,
          "support_markets": 210
        },
        "UP|90-120|below_0.60": {
          "margin_residual_quantile_bps": 0.7046570745057394,
          "probability_residual_quantile": 0.5821556105435699,
          "support_markets": 293
        }
      },
      "global": {
        "global": {
          "margin_residual_quantile_bps": 2.631058047519308,
          "probability_residual_quantile": 0.6226618441034003,
          "support_markets": 895
        }
      }
    },
    "coverage": {
      "empirical_conformal_coverage": 0.9080216419665961,
      "error_risk_bound_calibration": {
        "actual_error_rate": 0.35050576334980005,
        "bound_minus_actual": 0.6184134794129854,
        "mean_error_risk_upper_bound": 0.9689192427627855
      },
      "margin_interval_coverage": 0.9056692542931075,
      "market_blocks": 4251,
      "mean_margin_interval_width_bps": 14.474336665725994,
      "median_margin_interval_width_bps": 13.548410375181374,
      "p90_margin_interval_width_bps": 18.59188309684607
    }
  },
  "frozen_probability_control": {
    "candidate": "frozen_probability_control",
    "cells": {},
    "coverage": {
      "empirical_conformal_coverage": 0.5229050279329609,
      "error_risk_bound_calibration": {
        "actual_error_rate": 0.3560218699218043,
        "bound_minus_actual": 0.05667836566230694,
        "mean_error_risk_upper_bound": 0.41270023558411123
      },
      "margin_interval_coverage": 0.8067039106145252,
      "market_blocks": 895,
      "mean_margin_interval_width_bps": 10.663851316587358,
      "median_margin_interval_width_bps": 10.624555345368336,
      "p90_margin_interval_width_bps": 10.808932706325807
    }
  },
  "global_market_block_conformal": {
    "candidate": "global_market_block_conformal",
    "cells": {
      "global": {
        "global": {
          "margin_residual_quantile_bps": 2.631058047519308,
          "probability_residual_quantile": 0.6226618441034003,
          "support_markets": 895
        }
      }
    },
    "coverage": {
      "empirical_conformal_coverage": 0.9016759776536313,
      "error_risk_bound_calibration": {
        "actual_error_rate": 0.3560218699218043,
        "bound_minus_actual": 0.6134348349344467,
        "mean_error_risk_upper_bound": 0.969456704856251
      },
      "margin_interval_coverage": 0.9016759776536313,
      "market_blocks": 895,
      "mean_margin_interval_width_bps": 15.925967411625992,
      "median_margin_interval_width_bps": 15.886671440406952,
      "p90_margin_interval_width_bps": 16.071048801364423
    }
  }
}
```

## Development candidate comparison

| Candidate | Trades | Coverage | Accuracy | UP | DOWN | Stressed PnL | Expectancy | PF | Bootstrap lower | Avg entry | Qualified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| frozen_probability_control | 218 | 19.36% | 69.27% | 70.94% | 67.33% | 3.5388 | 0.0162 | 1.0156 | -0.3306 | 69.4037 | False |
| global_market_block_conformal | 0 | 0.00% | n/a | n/a | n/a | 0.0000 | n/a | n/a | n/a | n/a | False |
| direction_price_block_conformal | 0 | 0.00% | n/a | n/a | n/a | 0.0000 | n/a | n/a | n/a | n/a | False |
| direction_time_price_block_conformal | 0 | 0.00% | n/a | n/a | n/a | 0.0000 | n/a | n/a | n/a | n/a | False |

## Development detailed records

### frozen_probability_control

Complete candidate metric record:

```json
{
  "abstention_reasons": {
    "admitted": 1218,
    "correctness_lower_bound": 9544,
    "quote_recovery_geometry": 7982,
    "stale_or_unavailable_orderbook": 2650
  },
  "accuracy": 0.6926605504587156,
  "admitted_probability_metrics": {
    "brier_score": 0.2749495083682383,
    "ece": 0.2497180983755473,
    "log_loss": 1.0656931803768117,
    "markets": 218,
    "rows": 218
  },
  "all_quotes_pass_recovery": true,
  "average_loss": -3.3802802985074627,
  "average_win": 1.5232951324503314,
  "bootstrap_stressed_expectancy": {
    "lower": -0.3306352062830686,
    "median": 0.02662562850612135,
    "resamples": 2000,
    "unit": "market_day_block_stressed_pnl_per_observation",
    "upper": 0.6377676391002917
  },
  "by_direction": {
    "DOWN": {
      "accuracy": 0.6732673267326733,
      "average_loss": -3.4540316666666664,
      "average_win": 1.5244180882352947,
      "bootstrap_stressed_expectancy": {
        "lower": -0.6860061168831165,
        "median": -0.07710402873915688,
        "resamples": 2000,
        "unit": "market_day_block_stressed_pnl_per_observation",
        "upper": 1.389985948489011
      },
      "losses": 33,
      "stressed_expectancy": -0.10220410891089093,
      "stressed_pnl": -10.322614999999985,
      "trades": 101,
      "wins": 68
    },
    "UP": {
      "accuracy": 0.7094017094017094,
      "average_loss": -3.308698088235294,
      "average_win": 1.5223751204819278,
      "bootstrap_stressed_expectancy": {
        "lower": -0.31077561359797284,
        "median": 0.1297417743161096,
        "resamples": 2000,
        "unit": "market_day_block_stressed_pnl_per_observation",
        "upper": 0.5164714005255258
      },
      "losses": 34,
      "stressed_expectancy": 0.1184735042735044,
      "stressed_pnl": 13.861400000000016,
      "trades": 117,
      "wins": 83
    }
  },
  "capacity": {
    "10": {
      "net_pnl": 28.714576945430004,
      "stressed_expectancy": 0.031718242868944826,
      "stressed_pnl": 6.914576945429973,
      "supported_trades": 218
    },
    "100": {
      "net_pnl": 250.92647973075594,
      "stressed_expectancy": 0.15103889784750446,
      "stressed_pnl": 32.92647973075597,
      "supported_trades": 218
    },
    "125": {
      "net_pnl": 300.4419695597846,
      "stressed_expectancy": 0.12817417229258982,
      "stressed_pnl": 27.94196955978458,
      "supported_trades": 218
    },
    "15": {
      "net_pnl": 42.745205878473875,
      "stressed_expectancy": 0.04607892604804501,
      "stressed_pnl": 10.045205878473812,
      "supported_trades": 218
    },
    "150": {
      "net_pnl": 345.76858193386937,
      "stressed_expectancy": 0.08609441254068545,
      "stressed_pnl": 18.768581933869427,
      "supported_trades": 218
    },
    "175": {
      "net_pnl": 384.3137274406771,
      "stressed_expectancy": 0.012907006608610282,
      "stressed_pnl": 2.8137274406770416,
      "supported_trades": 218
    },
    "20": {
      "net_pnl": 56.65606149779,
      "stressed_expectancy": 0.05989019035683464,
      "stressed_pnl": 13.056061497789951,
      "supported_trades": 218
    },
    "200": {
      "net_pnl": 418.4456603010079,
      "stressed_expectancy": -0.08052449403207396,
      "stressed_pnl": -17.554339698992123,
      "supported_trades": 218
    },
    "25": {
      "net_pnl": 70.426620125616,
      "stressed_expectancy": 0.07305789048447692,
      "stressed_pnl": 15.926620125615969,
      "supported_trades": 218
    },
    "30": {
      "net_pnl": 84.00580770310542,
      "stressed_expectancy": 0.08534774175736376,
      "stressed_pnl": 18.6058077031053,
      "supported_trades": 218
    },
    "40": {
      "net_pnl": 110.53614072239498,
      "stressed_expectancy": 0.107046517075206,
      "stressed_pnl": 23.33614072239491,
      "supported_trades": 218
    },
    "5": {
      "net_pnl": 14.438784999999998,
      "stressed_expectancy": 0.016232958715596272,
      "stressed_pnl": 3.5387849999999874,
      "supported_trades": 218
    },
    "50": {
      "net_pnl": 136.0004812462761,
      "stressed_expectancy": 0.12385541856089946,
      "stressed_pnl": 27.00048124627608,
      "supported_trades": 218
    },
    "75": {
      "net_pnl": 195.99152891186466,
      "stressed_expectancy": 0.14904371060488392,
      "stressed_pnl": 32.49152891186469,
      "supported_trades": 218
    }
  },
  "conformal": {
    "empirical_conformal_coverage": 0.5189639222941721,
    "error_risk_bound_calibration": {
      "actual_error_rate": 0.3257940642429408,
      "bound_minus_actual": -0.05625402058382023,
      "mean_error_risk_upper_bound": 0.26954004365912054
    },
    "margin_interval_coverage": 0.3561517113783534,
    "market_blocks": 1081,
    "mean_margin_interval_width_bps": 10.585686247592365,
    "median_margin_interval_width_bps": 10.583398609916168,
    "p90_margin_interval_width_bps": 10.739277149805092
  },
  "coverage": 0.1936056838365897,
  "cvar_10": -3.6904138636363624,
  "daily": {
    "2026-08-18": {
      "accuracy": 0.75,
      "average_loss": -2.812185,
      "average_win": 1.6780199999999998,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 1,
      "stressed_expectancy": 0.5554687499999998,
      "stressed_pnl": 2.2218749999999994,
      "trades": 4,
      "wins": 3
    },
    "2026-08-19": {
      "accuracy": 0.75,
      "average_loss": -3.1656539999999995,
      "average_win": 1.6631990000000003,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 10,
      "stressed_expectancy": 0.45598575000000025,
      "stressed_pnl": 18.23943000000001,
      "trades": 40,
      "wins": 30
    },
    "2026-08-20": {
      "accuracy": 0.6721311475409836,
      "average_loss": -3.430264749999999,
      "average_win": 1.4721976829268293,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 20,
      "stressed_expectancy": -0.13516704918032768,
      "stressed_pnl": -8.245189999999988,
      "trades": 61,
      "wins": 41
    },
    "2026-08-21": {
      "accuracy": 0.6814159292035398,
      "average_loss": -3.42791,
      "average_win": 1.4899666233766238,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 36,
      "stressed_expectancy": -0.07679053097345118,
      "stressed_pnl": -8.677329999999984,
      "trades": 113,
      "wins": 77
    }
  },
  "down_accuracy": 0.6732673267326733,
  "down_trade_share": 0.463302752293578,
  "entry_time_bands": {
    "30-59": {
      "accuracy": 0.6627906976744186,
      "average_loss": -3.502859999999999,
      "average_win": 1.5654379824561406,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 29,
      "stressed_expectancy": -0.14363924418604634,
      "stressed_pnl": -12.352974999999985,
      "trades": 86,
      "wins": 57
    },
    "60-89": {
      "accuracy": 0.6901408450704225,
      "average_loss": -3.212933863636364,
      "average_win": 1.4918500000000003,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 22,
      "stressed_expectancy": 0.03402964788732417,
      "stressed_pnl": 2.416105000000016,
      "trades": 71,
      "wins": 49
    },
    "90-120": {
      "accuracy": 0.7377049180327869,
      "average_loss": -3.3882059374999995,
      "average_win": 1.5041544444444446,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 16,
      "stressed_expectancy": 0.22091237704918046,
      "stressed_pnl": 13.475655000000009,
      "trades": 61,
      "wins": 45
    }
  },
  "executable_price_bands": {
    "0.60-0.70": {
      "accuracy": 0.7181818181818181,
      "average_loss": -3.4359279032258057,
      "average_win": 1.5460712025316459,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 31,
      "stressed_expectancy": 0.14205327272727283,
      "stressed_pnl": 15.625860000000012,
      "trades": 110,
      "wins": 79
    },
    "0.70-0.80": {
      "accuracy": 0.7297297297297297,
      "average_loss": -3.699468749999999,
      "average_win": 1.296668055555556,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 20,
      "stressed_expectancy": -0.05363918918918888,
      "stressed_pnl": -3.969299999999977,
      "trades": 74,
      "wins": 54
    },
    "above_0.80": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "below_0.60": {
      "accuracy": 0.5294117647058824,
      "average_loss": -2.8734774999999995,
      "average_win": 2.1032147222222224,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 16,
      "stressed_expectancy": -0.23875808823529385,
      "stressed_pnl": -8.117774999999991,
      "trades": 34,
      "wins": 18
    }
  },
  "fallback_frequency": {
    "not_applicable": 21394
  },
  "fee_adjusted_pnl": 19.888785000000023,
  "gross_pnl": 36.74999999999999,
  "loss_distribution": {
    "average": 3.3802802985074627,
    "median": 3.45354,
    "p90": 3.716462999999999,
    "worst": 3.7455599999999993
  },
  "losses": 67,
  "markets": 1126,
  "maximum_drawdown": 32.84731999999999,
  "maximum_positive_pnl_day_share": 0.8914108850828429,
  "maximum_quote_loss_recovery_ratio": 2.985842288192341,
  "mean_entry_second": 69.40366972477064,
  "median_entry_second": 65.0,
  "net_pnl": 14.438785000000026,
  "p10_entry_second": 35.0,
  "p50_entry_second": 65.0,
  "p90_entry_second": 110.0,
  "predicted_margin_bands": {
    "0-5": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "10-20": {
      "accuracy": 0.7358490566037735,
      "average_loss": -3.392468214285714,
      "average_win": 1.5055693589743588,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 14,
      "stressed_expectancy": 0.21174811320754736,
      "stressed_pnl": 11.22265000000001,
      "trades": 53,
      "wins": 39
    },
    "5-10": {
      "accuracy": 0.6708074534161491,
      "average_loss": -3.377060849056604,
      "average_win": 1.5287752777777774,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 53,
      "stressed_expectancy": -0.08618940993788798,
      "stressed_pnl": -13.876494999999966,
      "trades": 161,
      "wins": 108
    },
    "above_20": {
      "accuracy": 1.0,
      "average_loss": null,
      "average_win": 1.5481575000000003,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": 1.5481575000000003,
      "stressed_pnl": 6.192630000000001,
      "trades": 4,
      "wins": 4
    }
  },
  "predictive_all": {
    "brier_score": 0.22148353965018952,
    "ece": 0.08411146760851683,
    "log_loss": 0.736734818686295,
    "markets": 1126,
    "rows": 21394
  },
  "profit_factor": 1.0156252387088982,
  "qualification": {
    "checks": {
      "average_entry": true,
      "average_loss_recovery": true,
      "both_directions_represented": true,
      "conformal_candidate": false,
      "coverage": true,
      "cvar_not_worse_than_control": true,
      "direction_economics": false,
      "down_accuracy": false,
      "empirical_conformal_coverage": false,
      "every_quote_recovery": true,
      "maximum_drawdown_not_worse_than_control": true,
      "no_material_band_failure": false,
      "overall_accuracy": false,
      "paired_improvement_lower_positive": false,
      "pnl_concentration": false,
      "positive_bootstrap_lower": false,
      "positive_each_two_day_fold": false,
      "positive_five_share_capacity": true,
      "positive_stressed_expectancy": true,
      "positive_stressed_pnl": true,
      "profit_factor": false,
      "trade_support": true,
      "up_accuracy": false
    },
    "passed": false,
    "reasons": [
      "conformal_candidate",
      "empirical_conformal_coverage",
      "overall_accuracy",
      "up_accuracy",
      "down_accuracy",
      "profit_factor",
      "positive_bootstrap_lower",
      "positive_each_two_day_fold",
      "direction_economics",
      "pnl_concentration",
      "no_material_band_failure",
      "paired_improvement_lower_positive"
    ]
  },
  "stressed_expectancy": 0.01623295871559651,
  "stressed_pnl": 3.53878500000004,
  "trades": 218,
  "two_day_folds": {
    "official_20260818_19": {
      "accuracy": 0.75,
      "average_loss": -3.1335204545454545,
      "average_win": 1.6645463636363638,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 11,
      "stressed_expectancy": 0.4650296590909093,
      "stressed_pnl": 20.46130500000001,
      "trades": 44,
      "wins": 33
    },
    "official_20260820_21": {
      "accuracy": 0.6781609195402298,
      "average_loss": -3.4287509821428563,
      "average_win": 1.4837926694915256,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 56,
      "stressed_expectancy": -0.09725586206896535,
      "stressed_pnl": -16.92251999999997,
      "trades": 174,
      "wins": 118
    }
  },
  "up_accuracy": 0.7094017094017094,
  "up_trade_share": 0.536697247706422,
  "wins": 151,
  "wins_to_recover_average_loss": 2.2190580318273816
}
```

Paired comparison against the frozen control:

```json
{
  "bootstrap_improvement": {
    "lower": 0.0,
    "median": 0.0,
    "resamples": 2000,
    "unit": "market_day_block_stressed_pnl_per_observation",
    "upper": 0.0
  },
  "candidate_stressed_pnl": 3.5387850000000416,
  "control_stressed_pnl": 3.5387850000000416,
  "scheduled_markets": 1126,
  "stressed_expectancy_improvement_per_scheduled_market": 0.0
}
```

### global_market_block_conformal

Complete candidate metric record:

```json
{
  "abstention_reasons": {
    "correctness_lower_bound": 10762,
    "quote_recovery_geometry": 7982,
    "stale_or_unavailable_orderbook": 2650
  },
  "accuracy": null,
  "admitted_probability_metrics": {
    "brier_score": null,
    "ece": null,
    "log_loss": null,
    "markets": 0,
    "rows": 0
  },
  "all_quotes_pass_recovery": false,
  "average_loss": null,
  "average_win": null,
  "bootstrap_stressed_expectancy": {
    "lower": null,
    "median": null,
    "resamples": 2000,
    "upper": null
  },
  "by_direction": {
    "DOWN": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "UP": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "capacity": {
    "10": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "100": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "125": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "15": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "150": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "175": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "20": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "200": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "25": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "30": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "40": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "5": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "50": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "75": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    }
  },
  "conformal": {
    "empirical_conformal_coverage": 0.7261794634597595,
    "error_risk_bound_calibration": {
      "actual_error_rate": 0.3257940642429405,
      "bound_minus_actual": 0.5408420090643158,
      "mean_error_risk_upper_bound": 0.8666360733072562
    },
    "margin_interval_coverage": 0.5189639222941721,
    "market_blocks": 1081,
    "mean_margin_interval_width_bps": 15.84780234263096,
    "median_margin_interval_width_bps": 15.845514704954784,
    "p90_margin_interval_width_bps": 16.001393244843708
  },
  "coverage": 0.0,
  "cvar_10": null,
  "daily": {
    "2026-08-18": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "2026-08-19": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "2026-08-20": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "2026-08-21": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "down_accuracy": null,
  "down_trade_share": 0.0,
  "entry_time_bands": {
    "30-59": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "60-89": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "90-120": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "executable_price_bands": {
    "0.60-0.70": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "0.70-0.80": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "above_0.80": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "below_0.60": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "fallback_frequency": {
    "global": 21394
  },
  "fee_adjusted_pnl": 0.0,
  "gross_pnl": 0.0,
  "loss_distribution": {
    "average": null,
    "median": null,
    "p90": null,
    "worst": null
  },
  "losses": 0,
  "markets": 1126,
  "maximum_drawdown": 0.0,
  "maximum_positive_pnl_day_share": null,
  "maximum_quote_loss_recovery_ratio": null,
  "mean_entry_second": null,
  "median_entry_second": null,
  "net_pnl": 0.0,
  "p10_entry_second": null,
  "p50_entry_second": null,
  "p90_entry_second": null,
  "predicted_margin_bands": {
    "0-5": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "10-20": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "5-10": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "above_20": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "predictive_all": {
    "brier_score": 0.22148353965018952,
    "ece": 0.08411146760851683,
    "log_loss": 0.736734818686295,
    "markets": 1126,
    "rows": 21394
  },
  "profit_factor": null,
  "qualification": {
    "checks": {
      "average_entry": false,
      "average_loss_recovery": false,
      "both_directions_represented": false,
      "conformal_candidate": true,
      "coverage": false,
      "cvar_not_worse_than_control": false,
      "direction_economics": false,
      "down_accuracy": false,
      "empirical_conformal_coverage": false,
      "every_quote_recovery": false,
      "maximum_drawdown_not_worse_than_control": true,
      "no_material_band_failure": true,
      "overall_accuracy": false,
      "paired_improvement_lower_positive": false,
      "pnl_concentration": false,
      "positive_bootstrap_lower": false,
      "positive_each_two_day_fold": false,
      "positive_five_share_capacity": false,
      "positive_stressed_expectancy": false,
      "positive_stressed_pnl": false,
      "profit_factor": false,
      "trade_support": false,
      "up_accuracy": false
    },
    "passed": false,
    "reasons": [
      "empirical_conformal_coverage",
      "overall_accuracy",
      "up_accuracy",
      "down_accuracy",
      "average_loss_recovery",
      "every_quote_recovery",
      "positive_stressed_pnl",
      "positive_stressed_expectancy",
      "profit_factor",
      "positive_bootstrap_lower",
      "positive_each_two_day_fold",
      "average_entry",
      "coverage",
      "trade_support",
      "both_directions_represented",
      "direction_economics",
      "pnl_concentration",
      "positive_five_share_capacity",
      "cvar_not_worse_than_control",
      "paired_improvement_lower_positive"
    ]
  },
  "stressed_expectancy": null,
  "stressed_pnl": 0.0,
  "trades": 0,
  "two_day_folds": {
    "official_20260818_19": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "official_20260820_21": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "up_accuracy": null,
  "up_trade_share": 0.0,
  "wins": 0,
  "wins_to_recover_average_loss": null
}
```

Paired comparison against the frozen control:

```json
{
  "bootstrap_improvement": {
    "lower": -0.07118837259038308,
    "median": -0.004661589636157733,
    "resamples": 2000,
    "unit": "market_day_block_stressed_pnl_per_observation",
    "upper": 0.07492812744227348
  },
  "candidate_stressed_pnl": 0.0,
  "control_stressed_pnl": 3.538785000000037,
  "scheduled_markets": 1126,
  "stressed_expectancy_improvement_per_scheduled_market": -0.0031427930728241894
}
```

### direction_price_block_conformal

Complete candidate metric record:

```json
{
  "abstention_reasons": {
    "correctness_lower_bound": 10762,
    "quote_recovery_geometry": 7982,
    "stale_or_unavailable_orderbook": 2650
  },
  "accuracy": null,
  "admitted_probability_metrics": {
    "brier_score": null,
    "ece": null,
    "log_loss": null,
    "markets": 0,
    "rows": 0
  },
  "all_quotes_pass_recovery": false,
  "average_loss": null,
  "average_win": null,
  "bootstrap_stressed_expectancy": {
    "lower": null,
    "median": null,
    "resamples": 2000,
    "upper": null
  },
  "by_direction": {
    "DOWN": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "UP": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "capacity": {
    "10": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "100": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "125": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "15": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "150": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "175": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "20": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "200": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "25": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "30": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "40": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "5": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "50": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "75": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    }
  },
  "conformal": {
    "empirical_conformal_coverage": 0.7787182587666264,
    "error_risk_bound_calibration": {
      "actual_error_rate": 0.32889963724304716,
      "bound_minus_actual": 0.5523541453303469,
      "mean_error_risk_upper_bound": 0.881253782573394
    },
    "margin_interval_coverage": 0.5538089480048367,
    "market_blocks": 3308,
    "mean_margin_interval_width_bps": 15.764959296754345,
    "median_margin_interval_width_bps": 15.539816355047982,
    "p90_margin_interval_width_bps": 18.4037121536112
  },
  "coverage": 0.0,
  "cvar_10": null,
  "daily": {
    "2026-08-18": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "2026-08-19": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "2026-08-20": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "2026-08-21": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "down_accuracy": null,
  "down_trade_share": 0.0,
  "entry_time_bands": {
    "30-59": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "60-89": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "90-120": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "executable_price_bands": {
    "0.60-0.70": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "0.70-0.80": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "above_0.80": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "below_0.60": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "fallback_frequency": {
    "direction_price": 18744,
    "global": 2650
  },
  "fee_adjusted_pnl": 0.0,
  "gross_pnl": 0.0,
  "loss_distribution": {
    "average": null,
    "median": null,
    "p90": null,
    "worst": null
  },
  "losses": 0,
  "markets": 1126,
  "maximum_drawdown": 0.0,
  "maximum_positive_pnl_day_share": null,
  "maximum_quote_loss_recovery_ratio": null,
  "mean_entry_second": null,
  "median_entry_second": null,
  "net_pnl": 0.0,
  "p10_entry_second": null,
  "p50_entry_second": null,
  "p90_entry_second": null,
  "predicted_margin_bands": {
    "0-5": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "10-20": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "5-10": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "above_20": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "predictive_all": {
    "brier_score": 0.22148353965018952,
    "ece": 0.08411146760851683,
    "log_loss": 0.736734818686295,
    "markets": 1126,
    "rows": 21394
  },
  "profit_factor": null,
  "qualification": {
    "checks": {
      "average_entry": false,
      "average_loss_recovery": false,
      "both_directions_represented": false,
      "conformal_candidate": true,
      "coverage": false,
      "cvar_not_worse_than_control": false,
      "direction_economics": false,
      "down_accuracy": false,
      "empirical_conformal_coverage": false,
      "every_quote_recovery": false,
      "maximum_drawdown_not_worse_than_control": true,
      "no_material_band_failure": true,
      "overall_accuracy": false,
      "paired_improvement_lower_positive": false,
      "pnl_concentration": false,
      "positive_bootstrap_lower": false,
      "positive_each_two_day_fold": false,
      "positive_five_share_capacity": false,
      "positive_stressed_expectancy": false,
      "positive_stressed_pnl": false,
      "profit_factor": false,
      "trade_support": false,
      "up_accuracy": false
    },
    "passed": false,
    "reasons": [
      "empirical_conformal_coverage",
      "overall_accuracy",
      "up_accuracy",
      "down_accuracy",
      "average_loss_recovery",
      "every_quote_recovery",
      "positive_stressed_pnl",
      "positive_stressed_expectancy",
      "profit_factor",
      "positive_bootstrap_lower",
      "positive_each_two_day_fold",
      "average_entry",
      "coverage",
      "trade_support",
      "both_directions_represented",
      "direction_economics",
      "pnl_concentration",
      "positive_five_share_capacity",
      "cvar_not_worse_than_control",
      "paired_improvement_lower_positive"
    ]
  },
  "stressed_expectancy": null,
  "stressed_pnl": 0.0,
  "trades": 0,
  "two_day_folds": {
    "official_20260818_19": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "official_20260820_21": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "up_accuracy": null,
  "up_trade_share": 0.0,
  "wins": 0,
  "wins_to_recover_average_loss": null
}
```

Paired comparison against the frozen control:

```json
{
  "bootstrap_improvement": {
    "lower": -0.07018014278408724,
    "median": -0.004546157362504206,
    "resamples": 2000,
    "unit": "market_day_block_stressed_pnl_per_observation",
    "upper": 0.06993339329922571
  },
  "candidate_stressed_pnl": 0.0,
  "control_stressed_pnl": 3.538785000000032,
  "scheduled_markets": 1126,
  "stressed_expectancy_improvement_per_scheduled_market": -0.0031427930728241847
}
```

### direction_time_price_block_conformal

Complete candidate metric record:

```json
{
  "abstention_reasons": {
    "correctness_lower_bound": 10762,
    "quote_recovery_geometry": 7982,
    "stale_or_unavailable_orderbook": 2650
  },
  "accuracy": null,
  "admitted_probability_metrics": {
    "brier_score": null,
    "ece": null,
    "log_loss": null,
    "markets": 0,
    "rows": 0
  },
  "all_quotes_pass_recovery": false,
  "average_loss": null,
  "average_win": null,
  "bootstrap_stressed_expectancy": {
    "lower": null,
    "median": null,
    "resamples": 2000,
    "upper": null
  },
  "by_direction": {
    "DOWN": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "UP": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "capacity": {
    "10": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "100": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "125": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "15": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "150": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "175": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "20": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "200": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "25": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "30": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "40": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "5": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "50": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "75": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    }
  },
  "conformal": {
    "empirical_conformal_coverage": 0.7760135135135136,
    "error_risk_bound_calibration": {
      "actual_error_rate": 0.32804054054054055,
      "bound_minus_actual": 0.5348282099440438,
      "mean_error_risk_upper_bound": 0.8628687504845843
    },
    "margin_interval_coverage": 0.5633445945945946,
    "market_blocks": 5920,
    "mean_margin_interval_width_bps": 14.725414414783002,
    "median_margin_interval_width_bps": 13.793595899036067,
    "p90_margin_interval_width_bps": 18.53959306039819
  },
  "coverage": 0.0,
  "cvar_10": null,
  "daily": {
    "2026-08-18": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "2026-08-19": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "2026-08-20": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "2026-08-21": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "down_accuracy": null,
  "down_trade_share": 0.0,
  "entry_time_bands": {
    "30-59": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "60-89": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "90-120": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "executable_price_bands": {
    "0.60-0.70": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "0.70-0.80": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "above_0.80": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "below_0.60": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "fallback_frequency": {
    "direction_price": 851,
    "direction_time_price": 17893,
    "global": 2650
  },
  "fee_adjusted_pnl": 0.0,
  "gross_pnl": 0.0,
  "loss_distribution": {
    "average": null,
    "median": null,
    "p90": null,
    "worst": null
  },
  "losses": 0,
  "markets": 1126,
  "maximum_drawdown": 0.0,
  "maximum_positive_pnl_day_share": null,
  "maximum_quote_loss_recovery_ratio": null,
  "mean_entry_second": null,
  "median_entry_second": null,
  "net_pnl": 0.0,
  "p10_entry_second": null,
  "p50_entry_second": null,
  "p90_entry_second": null,
  "predicted_margin_bands": {
    "0-5": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "10-20": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "5-10": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "above_20": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "predictive_all": {
    "brier_score": 0.22148353965018952,
    "ece": 0.08411146760851683,
    "log_loss": 0.736734818686295,
    "markets": 1126,
    "rows": 21394
  },
  "profit_factor": null,
  "qualification": {
    "checks": {
      "average_entry": false,
      "average_loss_recovery": false,
      "both_directions_represented": false,
      "conformal_candidate": true,
      "coverage": false,
      "cvar_not_worse_than_control": false,
      "direction_economics": false,
      "down_accuracy": false,
      "empirical_conformal_coverage": false,
      "every_quote_recovery": false,
      "maximum_drawdown_not_worse_than_control": true,
      "no_material_band_failure": true,
      "overall_accuracy": false,
      "paired_improvement_lower_positive": false,
      "pnl_concentration": false,
      "positive_bootstrap_lower": false,
      "positive_each_two_day_fold": false,
      "positive_five_share_capacity": false,
      "positive_stressed_expectancy": false,
      "positive_stressed_pnl": false,
      "profit_factor": false,
      "trade_support": false,
      "up_accuracy": false
    },
    "passed": false,
    "reasons": [
      "empirical_conformal_coverage",
      "overall_accuracy",
      "up_accuracy",
      "down_accuracy",
      "average_loss_recovery",
      "every_quote_recovery",
      "positive_stressed_pnl",
      "positive_stressed_expectancy",
      "profit_factor",
      "positive_bootstrap_lower",
      "positive_each_two_day_fold",
      "average_entry",
      "coverage",
      "trade_support",
      "both_directions_represented",
      "direction_economics",
      "pnl_concentration",
      "positive_five_share_capacity",
      "cvar_not_worse_than_control",
      "paired_improvement_lower_positive"
    ]
  },
  "stressed_expectancy": null,
  "stressed_pnl": 0.0,
  "trades": 0,
  "two_day_folds": {
    "official_20260818_19": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "official_20260820_21": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "up_accuracy": null,
  "up_trade_share": 0.0,
  "wins": 0,
  "wins_to_recover_average_loss": null
}
```

Paired comparison against the frozen control:

```json
{
  "bootstrap_improvement": {
    "lower": -0.07174266345291483,
    "median": -0.004381941819330765,
    "resamples": 2000,
    "unit": "market_day_block_stressed_pnl_per_observation",
    "upper": 0.06926513539262952
  },
  "candidate_stressed_pnl": 0.0,
  "control_stressed_pnl": 3.5387850000000416,
  "scheduled_markets": 1126,
  "stressed_expectancy_improvement_per_scheduled_market": -0.0031427930728241934
}
```

### Development selection record

```json
{
  "qualifications": {
    "direction_price_block_conformal": {
      "checks": {
        "average_entry": false,
        "average_loss_recovery": false,
        "both_directions_represented": false,
        "conformal_candidate": true,
        "coverage": false,
        "cvar_not_worse_than_control": false,
        "direction_economics": false,
        "down_accuracy": false,
        "empirical_conformal_coverage": false,
        "every_quote_recovery": false,
        "maximum_drawdown_not_worse_than_control": true,
        "no_material_band_failure": true,
        "overall_accuracy": false,
        "paired_improvement_lower_positive": false,
        "pnl_concentration": false,
        "positive_bootstrap_lower": false,
        "positive_each_two_day_fold": false,
        "positive_five_share_capacity": false,
        "positive_stressed_expectancy": false,
        "positive_stressed_pnl": false,
        "profit_factor": false,
        "trade_support": false,
        "up_accuracy": false
      },
      "passed": false,
      "reasons": [
        "empirical_conformal_coverage",
        "overall_accuracy",
        "up_accuracy",
        "down_accuracy",
        "average_loss_recovery",
        "every_quote_recovery",
        "positive_stressed_pnl",
        "positive_stressed_expectancy",
        "profit_factor",
        "positive_bootstrap_lower",
        "positive_each_two_day_fold",
        "average_entry",
        "coverage",
        "trade_support",
        "both_directions_represented",
        "direction_economics",
        "pnl_concentration",
        "positive_five_share_capacity",
        "cvar_not_worse_than_control",
        "paired_improvement_lower_positive"
      ]
    },
    "direction_time_price_block_conformal": {
      "checks": {
        "average_entry": false,
        "average_loss_recovery": false,
        "both_directions_represented": false,
        "conformal_candidate": true,
        "coverage": false,
        "cvar_not_worse_than_control": false,
        "direction_economics": false,
        "down_accuracy": false,
        "empirical_conformal_coverage": false,
        "every_quote_recovery": false,
        "maximum_drawdown_not_worse_than_control": true,
        "no_material_band_failure": true,
        "overall_accuracy": false,
        "paired_improvement_lower_positive": false,
        "pnl_concentration": false,
        "positive_bootstrap_lower": false,
        "positive_each_two_day_fold": false,
        "positive_five_share_capacity": false,
        "positive_stressed_expectancy": false,
        "positive_stressed_pnl": false,
        "profit_factor": false,
        "trade_support": false,
        "up_accuracy": false
      },
      "passed": false,
      "reasons": [
        "empirical_conformal_coverage",
        "overall_accuracy",
        "up_accuracy",
        "down_accuracy",
        "average_loss_recovery",
        "every_quote_recovery",
        "positive_stressed_pnl",
        "positive_stressed_expectancy",
        "profit_factor",
        "positive_bootstrap_lower",
        "positive_each_two_day_fold",
        "average_entry",
        "coverage",
        "trade_support",
        "both_directions_represented",
        "direction_economics",
        "pnl_concentration",
        "positive_five_share_capacity",
        "cvar_not_worse_than_control",
        "paired_improvement_lower_positive"
      ]
    },
    "frozen_probability_control": {
      "checks": {
        "average_entry": true,
        "average_loss_recovery": true,
        "both_directions_represented": true,
        "conformal_candidate": false,
        "coverage": true,
        "cvar_not_worse_than_control": true,
        "direction_economics": false,
        "down_accuracy": false,
        "empirical_conformal_coverage": false,
        "every_quote_recovery": true,
        "maximum_drawdown_not_worse_than_control": true,
        "no_material_band_failure": false,
        "overall_accuracy": false,
        "paired_improvement_lower_positive": false,
        "pnl_concentration": false,
        "positive_bootstrap_lower": false,
        "positive_each_two_day_fold": false,
        "positive_five_share_capacity": true,
        "positive_stressed_expectancy": true,
        "positive_stressed_pnl": true,
        "profit_factor": false,
        "trade_support": true,
        "up_accuracy": false
      },
      "passed": false,
      "reasons": [
        "conformal_candidate",
        "empirical_conformal_coverage",
        "overall_accuracy",
        "up_accuracy",
        "down_accuracy",
        "profit_factor",
        "positive_bootstrap_lower",
        "positive_each_two_day_fold",
        "direction_economics",
        "pnl_concentration",
        "no_material_band_failure",
        "paired_improvement_lower_positive"
      ]
    },
    "global_market_block_conformal": {
      "checks": {
        "average_entry": false,
        "average_loss_recovery": false,
        "both_directions_represented": false,
        "conformal_candidate": true,
        "coverage": false,
        "cvar_not_worse_than_control": false,
        "direction_economics": false,
        "down_accuracy": false,
        "empirical_conformal_coverage": false,
        "every_quote_recovery": false,
        "maximum_drawdown_not_worse_than_control": true,
        "no_material_band_failure": true,
        "overall_accuracy": false,
        "paired_improvement_lower_positive": false,
        "pnl_concentration": false,
        "positive_bootstrap_lower": false,
        "positive_each_two_day_fold": false,
        "positive_five_share_capacity": false,
        "positive_stressed_expectancy": false,
        "positive_stressed_pnl": false,
        "profit_factor": false,
        "trade_support": false,
        "up_accuracy": false
      },
      "passed": false,
      "reasons": [
        "empirical_conformal_coverage",
        "overall_accuracy",
        "up_accuracy",
        "down_accuracy",
        "average_loss_recovery",
        "every_quote_recovery",
        "positive_stressed_pnl",
        "positive_stressed_expectancy",
        "profit_factor",
        "positive_bootstrap_lower",
        "positive_each_two_day_fold",
        "average_entry",
        "coverage",
        "trade_support",
        "both_directions_represented",
        "direction_economics",
        "pnl_concentration",
        "positive_five_share_capacity",
        "cvar_not_worse_than_control",
        "paired_improvement_lower_positive"
      ]
    }
  },
  "qualified_candidates": [],
  "selected_candidate": "global_market_block_conformal",
  "selection_order": [
    "highest lower confidence bound for stressed expectancy",
    "highest trade count",
    "earliest average entry",
    "lowest maximum drawdown"
  ],
  "selection_pool": [
    "global_market_block_conformal",
    "direction_price_block_conformal",
    "direction_time_price_block_conformal"
  ],
  "status": "diagnostic_unqualified_candidate_selected",
  "test_read": false
}
```

## Untouched test

Candidate: `global_market_block_conformal`  
Qualification: **unqualified_on_untouched_test**  
Trades/coverage: 0 / 0.00%  
Accuracy overall/UP/DOWN: n/a / n/a / n/a  
Gross/net/stressed PnL: 0.0000 / 0.0000 / 0.0000  
Expectancy/profit factor: n/a / n/a  
Bootstrap 95% interval: `{"lower": null, "median": null, "resamples": 2000, "upper": null}`  
Failed gates: coverage, trade_support, overall_accuracy, up_accuracy, down_accuracy, positive_stressed_pnl, positive_stressed_expectancy, profit_factor, positive_bootstrap_lower, average_loss_recovery, every_quote_recovery, average_entry, positive_each_two_day_fold, both_directions_represented, direction_economics, pnl_concentration, positive_five_share_capacity, empirical_conformal_coverage

### Complete untouched-test metric record

```json
{
  "abstention_reasons": {
    "correctness_lower_bound": 11203,
    "quote_recovery_geometry": 7516,
    "stale_or_unavailable_orderbook": 3131
  },
  "accuracy": null,
  "admitted_probability_metrics": {
    "brier_score": null,
    "ece": null,
    "log_loss": null,
    "markets": 0,
    "rows": 0
  },
  "all_quotes_pass_recovery": false,
  "average_loss": null,
  "average_win": null,
  "bootstrap_stressed_expectancy": {
    "lower": null,
    "median": null,
    "resamples": 2000,
    "upper": null
  },
  "by_direction": {
    "DOWN": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "UP": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "capacity": {
    "10": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "100": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "125": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "15": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "150": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "175": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "20": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "200": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "25": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "30": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "40": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "5": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "50": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    },
    "75": {
      "net_pnl": 0.0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "supported_trades": 0
    }
  },
  "conformal": {
    "empirical_conformal_coverage": 0.8570127504553734,
    "error_risk_bound_calibration": {
      "actual_error_rate": 0.31428324596169166,
      "bound_minus_actual": 0.637191327542808,
      "mean_error_risk_upper_bound": 0.9514745735044997
    },
    "margin_interval_coverage": 0.8834244080145719,
    "market_blocks": 1098,
    "mean_margin_interval_width_bps": 39.75295983453683,
    "median_margin_interval_width_bps": 39.6699184768849,
    "p90_margin_interval_width_bps": 39.774069704051556
  },
  "coverage": 0.0,
  "cvar_10": null,
  "daily": {
    "2026-08-22": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "2026-08-23": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "2026-08-24": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "2026-08-25": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "down_accuracy": null,
  "down_trade_share": 0.0,
  "entry_time_bands": {
    "30-59": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "60-89": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "90-120": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "executable_price_bands": {
    "0.60-0.70": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "0.70-0.80": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "above_0.80": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "below_0.60": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "fallback_frequency": {
    "global": 21850
  },
  "fee_adjusted_pnl": 0.0,
  "gross_pnl": 0.0,
  "loss_distribution": {
    "average": null,
    "median": null,
    "p90": null,
    "worst": null
  },
  "losses": 0,
  "markets": 1150,
  "maximum_drawdown": 0.0,
  "maximum_positive_pnl_day_share": null,
  "maximum_quote_loss_recovery_ratio": null,
  "mean_entry_second": null,
  "median_entry_second": null,
  "net_pnl": 0.0,
  "p10_entry_second": null,
  "p50_entry_second": null,
  "p90_entry_second": null,
  "predicted_margin_bands": {
    "0-5": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "10-20": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "5-10": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "above_20": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "predictive_all": {
    "brier_score": 0.2120930542593269,
    "ece": 0.06956319783773345,
    "log_loss": 0.6756432869780176,
    "markets": 1150,
    "rows": 21850
  },
  "profit_factor": null,
  "stressed_expectancy": null,
  "stressed_pnl": 0.0,
  "trades": 0,
  "two_day_folds": {
    "official_20260822_23": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    },
    "official_20260824_25": {
      "accuracy": null,
      "bootstrap_stressed_expectancy": {
        "lower": null,
        "median": null,
        "resamples": 0,
        "upper": null
      },
      "losses": 0,
      "stressed_expectancy": null,
      "stressed_pnl": 0.0,
      "trades": 0,
      "wins": 0
    }
  },
  "up_accuracy": null,
  "up_trade_share": 0.0,
  "wins": 0,
  "wins_to_recover_average_loss": null
}
```

### Untouched-test qualification record

```json
{
  "checks": {
    "average_entry": false,
    "average_loss_recovery": false,
    "both_directions_represented": false,
    "coverage": false,
    "cvar_not_regressed": true,
    "direction_economics": false,
    "down_accuracy": false,
    "drawdown_not_regressed": true,
    "empirical_conformal_coverage": false,
    "every_quote_recovery": false,
    "overall_accuracy": false,
    "pnl_concentration": false,
    "positive_bootstrap_lower": false,
    "positive_each_two_day_fold": false,
    "positive_five_share_capacity": false,
    "positive_stressed_expectancy": false,
    "positive_stressed_pnl": false,
    "profit_factor": false,
    "trade_support": false,
    "up_accuracy": false
  },
  "passed": false,
  "reasons": [
    "coverage",
    "trade_support",
    "overall_accuracy",
    "up_accuracy",
    "down_accuracy",
    "positive_stressed_pnl",
    "positive_stressed_expectancy",
    "profit_factor",
    "positive_bootstrap_lower",
    "average_loss_recovery",
    "every_quote_recovery",
    "average_entry",
    "positive_each_two_day_fold",
    "both_directions_represented",
    "direction_economics",
    "pnl_concentration",
    "positive_five_share_capacity",
    "empirical_conformal_coverage"
  ],
  "status": "unqualified_on_untouched_test"
}
```

The complete machine-readable metric inventory is also preserved in `metrics.json`.

## Integrity

- Calibration, development, and untouched test market IDs are disjoint and chronological.
- The untouched test was loaded only after candidate selection and artifact freezing.
- The upstream predictor was not refit; raw RefPrice was excluded from admission.
- No database, data source, ingester, table, trading process, or runtime service was changed.
