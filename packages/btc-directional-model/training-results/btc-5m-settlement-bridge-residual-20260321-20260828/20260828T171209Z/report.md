# Settlement-Bridge Residual Tournament

- Run: `20260828T171209Z`
- Model family: `btc-5m-settlement-bridge-residual`
- Source commit: `deec9d5cb315d61b08b72437d228f4352510eaf2`
- Deployment status: **non_deployable**
- Conclusion: **A candidate is promising but lacks prospective evidence.**
- Runtime/trading process/database mutations: none

## RefPrice base admission

```json
{
  "bootstrap_95": {
    "lower": -0.0636315026080091,
    "mean": -0.06162093167909645,
    "upper": -0.059775423027030594
  },
  "constant_frequency": 0.49797462549678995,
  "markets": 26168,
  "paired_brier_difference": -0.06162093167909645,
  "passed": true
}
```

## Candidate probability and margin metrics

```json
{
  "non_twap_settlement_correction": {
    "accuracy": 0.7154733439603259,
    "brier": 0.1985258552817214,
    "by_date": {
      "2026-08-14": {
        "accuracy": 0.7498553240740741,
        "brier": 0.17358329536055395,
        "rows": 6912
      },
      "2026-08-15": {
        "accuracy": 0.692432273262662,
        "brier": 0.20558528083341931,
        "rows": 6792
      },
      "2026-08-16": {
        "accuracy": 0.6665194346289752,
        "brier": 0.20685082977536245,
        "rows": 6792
      },
      "2026-08-17": {
        "accuracy": 0.7433274021352313,
        "brier": 0.17478850929400727,
        "rows": 6744
      },
      "2026-08-18": {
        "accuracy": 0.7719298245614035,
        "brier": 0.16323653407502375,
        "rows": 6840
      },
      "2026-08-19": {
        "accuracy": 0.7305458768873403,
        "brier": 0.17816005572224658,
        "rows": 6888
      },
      "2026-08-20": {
        "accuracy": 0.7257922535211268,
        "brier": 0.1872476107866561,
        "rows": 6816
      },
      "2026-08-21": {
        "accuracy": 0.7145979020979021,
        "brier": 0.19251655012805233,
        "rows": 6864
      },
      "2026-08-22": {
        "accuracy": 0.7307805164319249,
        "brier": 0.18343821780871641,
        "rows": 6816
      },
      "2026-08-23": {
        "accuracy": 0.7532938076416338,
        "brier": 0.16605760738081116,
        "rows": 6072
      },
      "2026-08-24": {
        "accuracy": 0.6980633802816901,
        "brier": 0.2027622873947715,
        "rows": 6816
      },
      "2026-08-25": {
        "accuracy": 0.7322994987468672,
        "brier": 0.18312641555654943,
        "rows": 6384
      },
      "2026-08-26": {
        "accuracy": 0.7208856345885635,
        "brier": 0.18429562832266497,
        "rows": 5736
      },
      "2026-08-27": {
        "accuracy": 0.4968944099378882,
        "brier": 0.5036033350724457,
        "rows": 3864
      }
    },
    "by_direction": {
      "0": {
        "accuracy": 0.6521913415285944,
        "brier": 0.225116561049042,
        "rows": 44904
      },
      "1": {
        "accuracy": 0.7780198978693432,
        "brier": 0.17224418043192904,
        "rows": 45432
      }
    },
    "by_entry_band": {
      "120-149": {
        "accuracy": 0.7367162592986185,
        "brier": 0.1883232040133806,
        "rows": 22584
      },
      "150-179": {
        "accuracy": 0.7646121147715197,
        "brier": 0.1722273688349278,
        "rows": 22584
      },
      "60-89": {
        "accuracy": 0.6617516826071556,
        "brier": 0.22505089967868377,
        "rows": 22584
      },
      "90-119": {
        "accuracy": 0.6988133191640099,
        "brier": 0.20850194859989335,
        "rows": 22584
      }
    },
    "by_price_bucket": {},
    "by_volatility": {
      "q1": {
        "accuracy": 0.679964578259907,
        "brier": 0.21087152633112557,
        "rows": 22585
      },
      "q2": {
        "accuracy": 0.7338381154799858,
        "brier": 0.1900418932971409,
        "rows": 22584
      },
      "q3": {
        "accuracy": 0.7080104503387504,
        "brier": 0.20876532968076034,
        "rows": 22583
      },
      "q4": {
        "accuracy": 0.7400814736096352,
        "brier": 0.1844245785572915,
        "rows": 22584
      }
    },
    "ece": 0.04422944444972447,
    "interval_coverage": null,
    "log_loss": 0.8231371985110435,
    "margin_mae": null,
    "margin_markets": 0,
    "markets": 3764,
    "rows": 90336
  },
  "refprice_bridge_baseline": {
    "accuracy": 0.725436149486362,
    "brier": 0.18459930940926023,
    "by_date": {
      "2026-08-14": {
        "accuracy": 0.7510127314814815,
        "brier": 0.17343550509650385,
        "rows": 6912
      },
      "2026-08-15": {
        "accuracy": 0.692432273262662,
        "brier": 0.2056357756291344,
        "rows": 6792
      },
      "2026-08-16": {
        "accuracy": 0.6659305064782096,
        "brier": 0.20692577090731645,
        "rows": 6792
      },
      "2026-08-17": {
        "accuracy": 0.7440688018979834,
        "brier": 0.17480540027931088,
        "rows": 6744
      },
      "2026-08-18": {
        "accuracy": 0.7717836257309941,
        "brier": 0.16332681463417628,
        "rows": 6840
      },
      "2026-08-19": {
        "accuracy": 0.7312717770034843,
        "brier": 0.17817238254058435,
        "rows": 6888
      },
      "2026-08-20": {
        "accuracy": 0.7256455399061033,
        "brier": 0.18721035201482428,
        "rows": 6816
      },
      "2026-08-21": {
        "accuracy": 0.7143065268065268,
        "brier": 0.19229256280701132,
        "rows": 6864
      },
      "2026-08-22": {
        "accuracy": 0.7304870892018779,
        "brier": 0.1833510246017916,
        "rows": 6816
      },
      "2026-08-23": {
        "accuracy": 0.7527997364953887,
        "brier": 0.16598779189528848,
        "rows": 6072
      },
      "2026-08-24": {
        "accuracy": 0.6976232394366197,
        "brier": 0.20252335915432326,
        "rows": 6816
      },
      "2026-08-25": {
        "accuracy": 0.7319862155388471,
        "brier": 0.18307471605160514,
        "rows": 6384
      },
      "2026-08-26": {
        "accuracy": 0.7217573221757322,
        "brier": 0.18414308808751098,
        "rows": 5736
      },
      "2026-08-27": {
        "accuracy": 0.7285196687370601,
        "brier": 0.17930933072403968,
        "rows": 3864
      }
    },
    "by_direction": {
      "0": {
        "accuracy": 0.6841484054872617,
        "brier": 0.18867190135125553,
        "rows": 44904
      },
      "1": {
        "accuracy": 0.7662440570522979,
        "brier": 0.1805740481712923,
        "rows": 45432
      }
    },
    "by_entry_band": {
      "120-149": {
        "accuracy": 0.7480959971661353,
        "brier": 0.1738350323796634,
        "rows": 22584
      },
      "150-179": {
        "accuracy": 0.7766560396741056,
        "brier": 0.15711594597825246,
        "rows": 22584
      },
      "60-89": {
        "accuracy": 0.6691020191285866,
        "brier": 0.21250620542230936,
        "rows": 22584
      },
      "90-119": {
        "accuracy": 0.7078905419766206,
        "brier": 0.1949400538568156,
        "rows": 22584
      }
    },
    "by_price_bucket": {},
    "by_volatility": {
      "q1": {
        "accuracy": 0.6821784370157183,
        "brier": 0.20651749185203264,
        "rows": 22585
      },
      "q2": {
        "accuracy": 0.7469890187743535,
        "brier": 0.17358822752043207,
        "rows": 22584
      },
      "q3": {
        "accuracy": 0.723287428596732,
        "brier": 0.18811504836802823,
        "rows": 22583
      },
      "q4": {
        "accuracy": 0.7492915338292596,
        "brier": 0.17017565505225618,
        "rows": 22584
      }
    },
    "ece": 0.023263612097459953,
    "interval_coverage": null,
    "log_loss": 0.5493651820909649,
    "margin_mae": null,
    "margin_markets": 0,
    "markets": 3764,
    "rows": 90336
  },
  "relative_twap_settlement_correction": {
    "accuracy": 0.7154733439603259,
    "brier": 0.1985258552817214,
    "by_date": {
      "2026-08-14": {
        "accuracy": 0.7498553240740741,
        "brier": 0.17358329536055395,
        "rows": 6912
      },
      "2026-08-15": {
        "accuracy": 0.692432273262662,
        "brier": 0.20558528083341931,
        "rows": 6792
      },
      "2026-08-16": {
        "accuracy": 0.6665194346289752,
        "brier": 0.20685082977536245,
        "rows": 6792
      },
      "2026-08-17": {
        "accuracy": 0.7433274021352313,
        "brier": 0.17478850929400727,
        "rows": 6744
      },
      "2026-08-18": {
        "accuracy": 0.7719298245614035,
        "brier": 0.16323653407502375,
        "rows": 6840
      },
      "2026-08-19": {
        "accuracy": 0.7305458768873403,
        "brier": 0.17816005572224658,
        "rows": 6888
      },
      "2026-08-20": {
        "accuracy": 0.7257922535211268,
        "brier": 0.1872476107866561,
        "rows": 6816
      },
      "2026-08-21": {
        "accuracy": 0.7145979020979021,
        "brier": 0.19251655012805233,
        "rows": 6864
      },
      "2026-08-22": {
        "accuracy": 0.7307805164319249,
        "brier": 0.18343821780871641,
        "rows": 6816
      },
      "2026-08-23": {
        "accuracy": 0.7532938076416338,
        "brier": 0.16605760738081116,
        "rows": 6072
      },
      "2026-08-24": {
        "accuracy": 0.6980633802816901,
        "brier": 0.2027622873947715,
        "rows": 6816
      },
      "2026-08-25": {
        "accuracy": 0.7322994987468672,
        "brier": 0.18312641555654943,
        "rows": 6384
      },
      "2026-08-26": {
        "accuracy": 0.7208856345885635,
        "brier": 0.18429562832266497,
        "rows": 5736
      },
      "2026-08-27": {
        "accuracy": 0.4968944099378882,
        "brier": 0.5036033350724457,
        "rows": 3864
      }
    },
    "by_direction": {
      "0": {
        "accuracy": 0.6521913415285944,
        "brier": 0.225116561049042,
        "rows": 44904
      },
      "1": {
        "accuracy": 0.7780198978693432,
        "brier": 0.17224418043192904,
        "rows": 45432
      }
    },
    "by_entry_band": {
      "120-149": {
        "accuracy": 0.7367162592986185,
        "brier": 0.1883232040133806,
        "rows": 22584
      },
      "150-179": {
        "accuracy": 0.7646121147715197,
        "brier": 0.1722273688349278,
        "rows": 22584
      },
      "60-89": {
        "accuracy": 0.6617516826071556,
        "brier": 0.22505089967868377,
        "rows": 22584
      },
      "90-119": {
        "accuracy": 0.6988133191640099,
        "brier": 0.20850194859989335,
        "rows": 22584
      }
    },
    "by_price_bucket": {},
    "by_volatility": {
      "q1": {
        "accuracy": 0.679964578259907,
        "brier": 0.21087152633112557,
        "rows": 22585
      },
      "q2": {
        "accuracy": 0.7338381154799858,
        "brier": 0.1900418932971409,
        "rows": 22584
      },
      "q3": {
        "accuracy": 0.7080104503387504,
        "brier": 0.20876532968076034,
        "rows": 22583
      },
      "q4": {
        "accuracy": 0.7400814736096352,
        "brier": 0.1844245785572915,
        "rows": 22584
      }
    },
    "ece": 0.04422944444972447,
    "interval_coverage": null,
    "log_loss": 0.8231371985110435,
    "margin_mae": null,
    "margin_markets": 0,
    "markets": 3764,
    "rows": 90336
  },
  "twap_margin_residual_bridge": {
    "accuracy": 0.737967144881332,
    "brier": 0.18177905034455072,
    "by_date": {
      "2026-08-14": {
        "accuracy": 0.7469618055555556,
        "brier": 0.17706282484825053,
        "rows": 6912
      },
      "2026-08-15": {
        "accuracy": 0.6971436984687868,
        "brier": 0.21051802102185163,
        "rows": 6792
      },
      "2026-08-16": {
        "accuracy": 0.6812426383981154,
        "brier": 0.20981697343335193,
        "rows": 6792
      },
      "2026-08-17": {
        "accuracy": 0.7494068801897983,
        "brier": 0.17399754967101394,
        "rows": 6744
      },
      "2026-08-18": {
        "accuracy": 0.7885964912280702,
        "brier": 0.1652989351195944,
        "rows": 6840
      },
      "2026-08-19": {
        "accuracy": 0.7578397212543554,
        "brier": 0.16883152002552393,
        "rows": 6888
      },
      "2026-08-20": {
        "accuracy": 0.7454518779342723,
        "brier": 0.1791917280907278,
        "rows": 6816
      },
      "2026-08-21": {
        "accuracy": 0.7297494172494172,
        "brier": 0.1957231985322166,
        "rows": 6864
      },
      "2026-08-22": {
        "accuracy": 0.7491197183098591,
        "brier": 0.17080514965554852,
        "rows": 6816
      },
      "2026-08-23": {
        "accuracy": 0.7677865612648221,
        "brier": 0.15756967741939545,
        "rows": 6072
      },
      "2026-08-24": {
        "accuracy": 0.7216842723004695,
        "brier": 0.19764230428710658,
        "rows": 6816
      },
      "2026-08-25": {
        "accuracy": 0.7420112781954887,
        "brier": 0.1704582378869955,
        "rows": 6384
      },
      "2026-08-26": {
        "accuracy": 0.7494769874476988,
        "brier": 0.174184669709681,
        "rows": 5736
      },
      "2026-08-27": {
        "accuracy": 0.6881469979296067,
        "brier": 0.19543960305869018,
        "rows": 3864
      }
    },
    "by_direction": {
      "0": {
        "accuracy": 0.712675930874755,
        "brier": 0.18140158885769372,
        "rows": 44904
      },
      "1": {
        "accuracy": 0.7629644303574573,
        "brier": 0.18215212506293918,
        "rows": 45432
      }
    },
    "by_entry_band": {
      "120-149": {
        "accuracy": 0.7598299681190224,
        "brier": 0.17132553942150044,
        "rows": 22584
      },
      "150-179": {
        "accuracy": 0.789762663832802,
        "brier": 0.15366394320768276,
        "rows": 22584
      },
      "60-89": {
        "accuracy": 0.6816772936592278,
        "brier": 0.21011920009914997,
        "rows": 22584
      },
      "90-119": {
        "accuracy": 0.7205986539142756,
        "brier": 0.19200751864986965,
        "rows": 22584
      }
    },
    "by_price_bucket": {},
    "by_volatility": {
      "q1": {
        "accuracy": 0.6912109807394288,
        "brier": 0.20796722018336955,
        "rows": 22585
      },
      "q2": {
        "accuracy": 0.7529667020899752,
        "brier": 0.17158588706084707,
        "rows": 22584
      },
      "q3": {
        "accuracy": 0.7429039543019085,
        "brier": 0.17789556541090598,
        "rows": 22583
      },
      "q4": {
        "accuracy": 0.7647892313142047,
        "brier": 0.16966619717619869,
        "rows": 22584
      }
    },
    "ece": 0.04759276347718181,
    "interval_coverage": 0.9695996275605214,
    "log_loss": 0.5548990643577015,
    "margin_mae": 5.627189668656409,
    "margin_markets": 3580,
    "markets": 3764,
    "rows": 90336
  }
}
```

## Predictive comparisons

```json
{
  "comparisons": {
    "non_twap_correction_is_useful": {
      "bootstrap_95": {
        "lower": 0.009698444744476582,
        "mean": 0.013926545872461175,
        "upper": 0.018018172906651215
      },
      "improved_folds": 2,
      "left": "refprice_bridge_baseline",
      "paired_brier_difference": 0.013926545872461175,
      "passed": false,
      "right": "non_twap_settlement_correction"
    },
    "settlement_bridge_is_useful": {
      "bootstrap_95": {
        "lower": -0.07117479199077631,
        "mean": -0.06539215003481597,
        "upper": -0.05981189014604867
      },
      "improved_folds": 7,
      "left": "constant_twap_frequency",
      "paired_brier_difference": -0.06539215003481597,
      "passed": true,
      "right": "refprice_bridge_baseline"
    }
  },
  "metrics": {
    "non_twap_settlement_correction": {
      "accuracy": 0.7154733439603259,
      "brier": 0.1985258552817214,
      "by_date": {
        "2026-08-14": {
          "accuracy": 0.7498553240740741,
          "brier": 0.17358329536055395,
          "rows": 6912
        },
        "2026-08-15": {
          "accuracy": 0.692432273262662,
          "brier": 0.20558528083341931,
          "rows": 6792
        },
        "2026-08-16": {
          "accuracy": 0.6665194346289752,
          "brier": 0.20685082977536245,
          "rows": 6792
        },
        "2026-08-17": {
          "accuracy": 0.7433274021352313,
          "brier": 0.17478850929400727,
          "rows": 6744
        },
        "2026-08-18": {
          "accuracy": 0.7719298245614035,
          "brier": 0.16323653407502375,
          "rows": 6840
        },
        "2026-08-19": {
          "accuracy": 0.7305458768873403,
          "brier": 0.17816005572224658,
          "rows": 6888
        },
        "2026-08-20": {
          "accuracy": 0.7257922535211268,
          "brier": 0.1872476107866561,
          "rows": 6816
        },
        "2026-08-21": {
          "accuracy": 0.7145979020979021,
          "brier": 0.19251655012805233,
          "rows": 6864
        },
        "2026-08-22": {
          "accuracy": 0.7307805164319249,
          "brier": 0.18343821780871641,
          "rows": 6816
        },
        "2026-08-23": {
          "accuracy": 0.7532938076416338,
          "brier": 0.16605760738081116,
          "rows": 6072
        },
        "2026-08-24": {
          "accuracy": 0.6980633802816901,
          "brier": 0.2027622873947715,
          "rows": 6816
        },
        "2026-08-25": {
          "accuracy": 0.7322994987468672,
          "brier": 0.18312641555654943,
          "rows": 6384
        },
        "2026-08-26": {
          "accuracy": 0.7208856345885635,
          "brier": 0.18429562832266497,
          "rows": 5736
        },
        "2026-08-27": {
          "accuracy": 0.4968944099378882,
          "brier": 0.5036033350724457,
          "rows": 3864
        }
      },
      "by_direction": {
        "0": {
          "accuracy": 0.6521913415285944,
          "brier": 0.225116561049042,
          "rows": 44904
        },
        "1": {
          "accuracy": 0.7780198978693432,
          "brier": 0.17224418043192904,
          "rows": 45432
        }
      },
      "by_entry_band": {
        "120-149": {
          "accuracy": 0.7367162592986185,
          "brier": 0.1883232040133806,
          "rows": 22584
        },
        "150-179": {
          "accuracy": 0.7646121147715197,
          "brier": 0.1722273688349278,
          "rows": 22584
        },
        "60-89": {
          "accuracy": 0.6617516826071556,
          "brier": 0.22505089967868377,
          "rows": 22584
        },
        "90-119": {
          "accuracy": 0.6988133191640099,
          "brier": 0.20850194859989335,
          "rows": 22584
        }
      },
      "by_price_bucket": {},
      "by_volatility": {
        "q1": {
          "accuracy": 0.679964578259907,
          "brier": 0.21087152633112557,
          "rows": 22585
        },
        "q2": {
          "accuracy": 0.7338381154799858,
          "brier": 0.1900418932971409,
          "rows": 22584
        },
        "q3": {
          "accuracy": 0.7080104503387504,
          "brier": 0.20876532968076034,
          "rows": 22583
        },
        "q4": {
          "accuracy": 0.7400814736096352,
          "brier": 0.1844245785572915,
          "rows": 22584
        }
      },
      "ece": 0.04422944444972447,
      "interval_coverage": null,
      "log_loss": 0.8231371985110435,
      "margin_mae": null,
      "margin_markets": 0,
      "markets": 3764,
      "rows": 90336
    },
    "refprice_bridge_baseline": {
      "accuracy": 0.725436149486362,
      "brier": 0.18459930940926023,
      "by_date": {
        "2026-08-14": {
          "accuracy": 0.7510127314814815,
          "brier": 0.17343550509650385,
          "rows": 6912
        },
        "2026-08-15": {
          "accuracy": 0.692432273262662,
          "brier": 0.2056357756291344,
          "rows": 6792
        },
        "2026-08-16": {
          "accuracy": 0.6659305064782096,
          "brier": 0.20692577090731645,
          "rows": 6792
        },
        "2026-08-17": {
          "accuracy": 0.7440688018979834,
          "brier": 0.17480540027931088,
          "rows": 6744
        },
        "2026-08-18": {
          "accuracy": 0.7717836257309941,
          "brier": 0.16332681463417628,
          "rows": 6840
        },
        "2026-08-19": {
          "accuracy": 0.7312717770034843,
          "brier": 0.17817238254058435,
          "rows": 6888
        },
        "2026-08-20": {
          "accuracy": 0.7256455399061033,
          "brier": 0.18721035201482428,
          "rows": 6816
        },
        "2026-08-21": {
          "accuracy": 0.7143065268065268,
          "brier": 0.19229256280701132,
          "rows": 6864
        },
        "2026-08-22": {
          "accuracy": 0.7304870892018779,
          "brier": 0.1833510246017916,
          "rows": 6816
        },
        "2026-08-23": {
          "accuracy": 0.7527997364953887,
          "brier": 0.16598779189528848,
          "rows": 6072
        },
        "2026-08-24": {
          "accuracy": 0.6976232394366197,
          "brier": 0.20252335915432326,
          "rows": 6816
        },
        "2026-08-25": {
          "accuracy": 0.7319862155388471,
          "brier": 0.18307471605160514,
          "rows": 6384
        },
        "2026-08-26": {
          "accuracy": 0.7217573221757322,
          "brier": 0.18414308808751098,
          "rows": 5736
        },
        "2026-08-27": {
          "accuracy": 0.7285196687370601,
          "brier": 0.17930933072403968,
          "rows": 3864
        }
      },
      "by_direction": {
        "0": {
          "accuracy": 0.6841484054872617,
          "brier": 0.18867190135125553,
          "rows": 44904
        },
        "1": {
          "accuracy": 0.7662440570522979,
          "brier": 0.1805740481712923,
          "rows": 45432
        }
      },
      "by_entry_band": {
        "120-149": {
          "accuracy": 0.7480959971661353,
          "brier": 0.1738350323796634,
          "rows": 22584
        },
        "150-179": {
          "accuracy": 0.7766560396741056,
          "brier": 0.15711594597825246,
          "rows": 22584
        },
        "60-89": {
          "accuracy": 0.6691020191285866,
          "brier": 0.21250620542230936,
          "rows": 22584
        },
        "90-119": {
          "accuracy": 0.7078905419766206,
          "brier": 0.1949400538568156,
          "rows": 22584
        }
      },
      "by_price_bucket": {},
      "by_volatility": {
        "q1": {
          "accuracy": 0.6821784370157183,
          "brier": 0.20651749185203264,
          "rows": 22585
        },
        "q2": {
          "accuracy": 0.7469890187743535,
          "brier": 0.17358822752043207,
          "rows": 22584
        },
        "q3": {
          "accuracy": 0.723287428596732,
          "brier": 0.18811504836802823,
          "rows": 22583
        },
        "q4": {
          "accuracy": 0.7492915338292596,
          "brier": 0.17017565505225618,
          "rows": 22584
        }
      },
      "ece": 0.023263612097459953,
      "interval_coverage": null,
      "log_loss": 0.5493651820909649,
      "margin_mae": null,
      "margin_markets": 0,
      "markets": 3764,
      "rows": 90336
    },
    "relative_twap_settlement_correction": {
      "accuracy": 0.7154733439603259,
      "brier": 0.1985258552817214,
      "by_date": {
        "2026-08-14": {
          "accuracy": 0.7498553240740741,
          "brier": 0.17358329536055395,
          "rows": 6912
        },
        "2026-08-15": {
          "accuracy": 0.692432273262662,
          "brier": 0.20558528083341931,
          "rows": 6792
        },
        "2026-08-16": {
          "accuracy": 0.6665194346289752,
          "brier": 0.20685082977536245,
          "rows": 6792
        },
        "2026-08-17": {
          "accuracy": 0.7433274021352313,
          "brier": 0.17478850929400727,
          "rows": 6744
        },
        "2026-08-18": {
          "accuracy": 0.7719298245614035,
          "brier": 0.16323653407502375,
          "rows": 6840
        },
        "2026-08-19": {
          "accuracy": 0.7305458768873403,
          "brier": 0.17816005572224658,
          "rows": 6888
        },
        "2026-08-20": {
          "accuracy": 0.7257922535211268,
          "brier": 0.1872476107866561,
          "rows": 6816
        },
        "2026-08-21": {
          "accuracy": 0.7145979020979021,
          "brier": 0.19251655012805233,
          "rows": 6864
        },
        "2026-08-22": {
          "accuracy": 0.7307805164319249,
          "brier": 0.18343821780871641,
          "rows": 6816
        },
        "2026-08-23": {
          "accuracy": 0.7532938076416338,
          "brier": 0.16605760738081116,
          "rows": 6072
        },
        "2026-08-24": {
          "accuracy": 0.6980633802816901,
          "brier": 0.2027622873947715,
          "rows": 6816
        },
        "2026-08-25": {
          "accuracy": 0.7322994987468672,
          "brier": 0.18312641555654943,
          "rows": 6384
        },
        "2026-08-26": {
          "accuracy": 0.7208856345885635,
          "brier": 0.18429562832266497,
          "rows": 5736
        },
        "2026-08-27": {
          "accuracy": 0.4968944099378882,
          "brier": 0.5036033350724457,
          "rows": 3864
        }
      },
      "by_direction": {
        "0": {
          "accuracy": 0.6521913415285944,
          "brier": 0.225116561049042,
          "rows": 44904
        },
        "1": {
          "accuracy": 0.7780198978693432,
          "brier": 0.17224418043192904,
          "rows": 45432
        }
      },
      "by_entry_band": {
        "120-149": {
          "accuracy": 0.7367162592986185,
          "brier": 0.1883232040133806,
          "rows": 22584
        },
        "150-179": {
          "accuracy": 0.7646121147715197,
          "brier": 0.1722273688349278,
          "rows": 22584
        },
        "60-89": {
          "accuracy": 0.6617516826071556,
          "brier": 0.22505089967868377,
          "rows": 22584
        },
        "90-119": {
          "accuracy": 0.6988133191640099,
          "brier": 0.20850194859989335,
          "rows": 22584
        }
      },
      "by_price_bucket": {},
      "by_volatility": {
        "q1": {
          "accuracy": 0.679964578259907,
          "brier": 0.21087152633112557,
          "rows": 22585
        },
        "q2": {
          "accuracy": 0.7338381154799858,
          "brier": 0.1900418932971409,
          "rows": 22584
        },
        "q3": {
          "accuracy": 0.7080104503387504,
          "brier": 0.20876532968076034,
          "rows": 22583
        },
        "q4": {
          "accuracy": 0.7400814736096352,
          "brier": 0.1844245785572915,
          "rows": 22584
        }
      },
      "ece": 0.04422944444972447,
      "interval_coverage": null,
      "log_loss": 0.8231371985110435,
      "margin_mae": null,
      "margin_markets": 0,
      "markets": 3764,
      "rows": 90336
    },
    "twap_margin_residual_bridge": {
      "accuracy": 0.737967144881332,
      "brier": 0.18177905034455072,
      "by_date": {
        "2026-08-14": {
          "accuracy": 0.7469618055555556,
          "brier": 0.17706282484825053,
          "rows": 6912
        },
        "2026-08-15": {
          "accuracy": 0.6971436984687868,
          "brier": 0.21051802102185163,
          "rows": 6792
        },
        "2026-08-16": {
          "accuracy": 0.6812426383981154,
          "brier": 0.20981697343335193,
          "rows": 6792
        },
        "2026-08-17": {
          "accuracy": 0.7494068801897983,
          "brier": 0.17399754967101394,
          "rows": 6744
        },
        "2026-08-18": {
          "accuracy": 0.7885964912280702,
          "brier": 0.1652989351195944,
          "rows": 6840
        },
        "2026-08-19": {
          "accuracy": 0.7578397212543554,
          "brier": 0.16883152002552393,
          "rows": 6888
        },
        "2026-08-20": {
          "accuracy": 0.7454518779342723,
          "brier": 0.1791917280907278,
          "rows": 6816
        },
        "2026-08-21": {
          "accuracy": 0.7297494172494172,
          "brier": 0.1957231985322166,
          "rows": 6864
        },
        "2026-08-22": {
          "accuracy": 0.7491197183098591,
          "brier": 0.17080514965554852,
          "rows": 6816
        },
        "2026-08-23": {
          "accuracy": 0.7677865612648221,
          "brier": 0.15756967741939545,
          "rows": 6072
        },
        "2026-08-24": {
          "accuracy": 0.7216842723004695,
          "brier": 0.19764230428710658,
          "rows": 6816
        },
        "2026-08-25": {
          "accuracy": 0.7420112781954887,
          "brier": 0.1704582378869955,
          "rows": 6384
        },
        "2026-08-26": {
          "accuracy": 0.7494769874476988,
          "brier": 0.174184669709681,
          "rows": 5736
        },
        "2026-08-27": {
          "accuracy": 0.6881469979296067,
          "brier": 0.19543960305869018,
          "rows": 3864
        }
      },
      "by_direction": {
        "0": {
          "accuracy": 0.712675930874755,
          "brier": 0.18140158885769372,
          "rows": 44904
        },
        "1": {
          "accuracy": 0.7629644303574573,
          "brier": 0.18215212506293918,
          "rows": 45432
        }
      },
      "by_entry_band": {
        "120-149": {
          "accuracy": 0.7598299681190224,
          "brier": 0.17132553942150044,
          "rows": 22584
        },
        "150-179": {
          "accuracy": 0.789762663832802,
          "brier": 0.15366394320768276,
          "rows": 22584
        },
        "60-89": {
          "accuracy": 0.6816772936592278,
          "brier": 0.21011920009914997,
          "rows": 22584
        },
        "90-119": {
          "accuracy": 0.7205986539142756,
          "brier": 0.19200751864986965,
          "rows": 22584
        }
      },
      "by_price_bucket": {},
      "by_volatility": {
        "q1": {
          "accuracy": 0.6912109807394288,
          "brier": 0.20796722018336955,
          "rows": 22585
        },
        "q2": {
          "accuracy": 0.7529667020899752,
          "brier": 0.17158588706084707,
          "rows": 22584
        },
        "q3": {
          "accuracy": 0.7429039543019085,
          "brier": 0.17789556541090598,
          "rows": 22583
        },
        "q4": {
          "accuracy": 0.7647892313142047,
          "brier": 0.16966619717619869,
          "rows": 22584
        }
      },
      "ece": 0.04759276347718181,
      "interval_coverage": 0.9695996275605214,
      "log_loss": 0.5548990643577015,
      "margin_mae": 5.627189668656409,
      "margin_markets": 3580,
      "markets": 3764,
      "rows": 90336
    }
  },
  "winner": "refprice_bridge_baseline"
}
```

## Economic admission

```json
{
  "bootstrap_lower": -Infinity,
  "coverage": 0.0,
  "gate_count": 0,
  "maximum_day_contribution": Infinity,
  "passed": false,
  "policy_search": [
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.55,
        "stressed_edge": 0.0
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.55,
        "stressed_edge": 0.01
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.55,
        "stressed_edge": 0.02
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.55,
        "stressed_edge": 0.03
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.55,
        "stressed_edge": 0.05
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.6,
        "stressed_edge": 0.0
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.6,
        "stressed_edge": 0.01
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.6,
        "stressed_edge": 0.02
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.6,
        "stressed_edge": 0.03
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.6,
        "stressed_edge": 0.05
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.65,
        "stressed_edge": 0.0
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.65,
        "stressed_edge": 0.01
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.65,
        "stressed_edge": 0.02
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.65,
        "stressed_edge": 0.03
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.65,
        "stressed_edge": 0.05
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.7,
        "stressed_edge": 0.0
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.7,
        "stressed_edge": 0.01
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.7,
        "stressed_edge": 0.02
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.7,
        "stressed_edge": 0.03
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.7,
        "stressed_edge": 0.05
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.75,
        "stressed_edge": 0.0
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.75,
        "stressed_edge": 0.01
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.75,
        "stressed_edge": 0.02
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.75,
        "stressed_edge": 0.03
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.75,
        "stressed_edge": 0.05
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.8,
        "stressed_edge": 0.0
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.8,
        "stressed_edge": 0.01
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.8,
        "stressed_edge": 0.02
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.8,
        "stressed_edge": 0.03
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    },
    {
      "bootstrap_lower": -Infinity,
      "coverage": 0.0,
      "gate_count": 0,
      "maximum_day_contribution": Infinity,
      "passed": false,
      "policy": {
        "confidence": 0.8,
        "stressed_edge": 0.05
      },
      "profit_factor": 0.0,
      "profitable_fold_ratio": 0.0,
      "stressed_expectancy": 0.0,
      "stressed_pnl_5": 0.0,
      "trades": 0
    }
  ],
  "profit_factor": 0.0,
  "profitable_fold_ratio": 0.0,
  "selected_policy": {
    "confidence": 0.55,
    "stressed_edge": 0.0
  },
  "status": "failed",
  "stressed_expectancy": 0.0,
  "stressed_pnl_5": 0.0,
  "trades": 0
}
```

## Prospective qualification

```json
{
  "active_days": 1,
  "checks": {
    "active_days": false,
    "authentic_markets": false,
    "chronological_folds": false
  },
  "deployable": false,
  "markets": 72,
  "metrics": {
    "bootstrap_lower": -Infinity,
    "coverage": 0.0,
    "gate_count": 0,
    "maximum_day_contribution": Infinity,
    "passed": false,
    "profit_factor": 0.0,
    "profitable_fold_ratio": 0.0,
    "stressed_expectancy": 0.0,
    "stressed_pnl_5": 0.0,
    "trades": 0
  },
  "status": "pending_insufficient_evidence"
}
```

## Settlement transition

```json
{
  "agreement_rate": 0.8652354874041621,
  "by_month": {
    "2026-06": {
      "agreement_rate": 0.8463881636205396,
      "markets": 6894
    },
    "2026-07": {
      "agreement_rate": 0.852991452991453,
      "markets": 8775
    },
    "2026-08": {
      "agreement_rate": 0.8984069312465064,
      "markets": 7156
    }
  },
  "by_ref_direction": {
    "0": {
      "agreement_rate": 0.8704726826273788,
      "markets": 11403
    },
    "1": {
      "agreement_rate": 0.8600070040273157,
      "markets": 11422
    }
  },
  "by_reversal_state": {
    "continuation": {
      "agreement_rate": 0.8641298265249021,
      "markets": 17870
    },
    "reversal": {
      "agreement_rate": 0.8692230070635721,
      "markets": 4955
    }
  },
  "by_terminal_margin_band": {
    "0.526-1.052": {
      "agreement_rate": 0.6406544996853367,
      "markets": 1589
    },
    "1.052-1.578": {
      "agreement_rate": 0.7098765432098766,
      "markets": 1458
    },
    "1.578-5": {
      "agreement_rate": 0.8489907421460009,
      "markets": 6589
    },
    "<0.526": {
      "agreement_rate": 0.596228508042152,
      "markets": 1803
    },
    ">=5": {
      "agreement_rate": 0.9715626974812674,
      "markets": 11077
    },
    "None": {
      "agreement_rate": 0.8576051779935275,
      "markets": 309
    }
  },
  "by_volatility": {
    "high": {
      "agreement_rate": 0.8760077111812128,
      "markets": 11412
    },
    "low": {
      "agreement_rate": 0.8544642074826951,
      "markets": 11413
    }
  },
  "disagreement_rate": 0.1347645125958379,
  "margin_residual_distribution": {
    "count": 22192,
    "mean": -0.0017134072893277349,
    "median": -2.2203211491600428e-12,
    "p05": -6.061908794635052,
    "p25": -1.6539207751736629,
    "p75": 1.6530303017830728,
    "p95": 6.0081049199053025,
    "std": 4.257662894157702
  },
  "markets": 22825,
  "predicted_correction_when_agree": {
    "count": 80544,
    "mean": -0.05233774360427,
    "median": -0.008641430116207346,
    "p05": -3.0626966210554474,
    "p25": -0.846417824272128,
    "p75": 0.7076026049422064,
    "p95": 2.7940639822258455,
    "std": 1.7862810691720958
  },
  "predicted_correction_when_disagree": {
    "count": 8568,
    "mean": -0.17363751435134342,
    "median": -0.0652405341108202,
    "p05": -4.5343772566087885,
    "p25": -1.3043829516085483,
    "p75": 0.9123354031528129,
    "p95": 3.9818778341791363,
    "std": 2.4522178774783585
  },
  "transition_errors_by_entry_time": {
    "120-149": {
      "margin_mae": 5.333035063013844,
      "rows": 22584
    },
    "150-179": {
      "margin_mae": 4.707548393813077,
      "rows": 22584
    },
    "60-89": {
      "margin_mae": 6.492891812946248,
      "rows": 22584
    },
    "90-119": {
      "margin_mae": 5.975283404852467,
      "rows": 22584
    }
  }
}
```

The artifact is training-only, was not exported to runtime, and did not change any trading process.
