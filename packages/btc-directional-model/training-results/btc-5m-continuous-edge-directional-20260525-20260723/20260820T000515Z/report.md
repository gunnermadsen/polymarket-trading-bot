# BTC Five-Minute Continuous-Edge Directional Training

Status: **development candidate rejected; not runtime exported**

Selected probability candidate: `core_oracle_vwap_curve`

The candidate failed frozen untouched-test expectancy gates and must not be deployed.

## Untouched chronological test

| Policy | Trades | Accuracy | Net PnL (VWAP5) | Expectancy | PF | Avg entry | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Without correctness veto | 1097 | 77.94% | -79.00 | -0.0720 | 0.914 | 131.286 | — |
| Selected continuous edge | 902 | 81.04% | -41.62 | -0.0461 | 0.937 | 142.868 | 39.91% |

## Fixed-entry VWAP capacity curve

The side and timestamp are frozen from the VWAP5 policy; only exact execution size changes.

| Shares | Trades | Accuracy | Mean VWAP | Net PnL | Expectancy | PF | Max drawdown |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 902 | 81.04% | 0.8043 | -41.62 | -0.0461 | 0.937 | 74.51 |
| 10 | 902 | 81.04% | 0.8044 | -84.47 | -0.0936 | 0.936 | 150.06 |
| 15 | 902 | 81.04% | 0.8046 | -128.58 | -0.1425 | 0.935 | 226.66 |
| 20 | 902 | 81.04% | 0.8048 | -174.61 | -0.1936 | 0.934 | 304.92 |
| 25 | 902 | 81.04% | 0.8049 | -222.05 | -0.2462 | 0.933 | 384.47 |
| 30 | 902 | 81.04% | 0.8051 | -270.49 | -0.2999 | 0.932 | 464.80 |
| 40 | 902 | 81.04% | 0.8054 | -370.55 | -0.4108 | 0.930 | 628.27 |
| 50 | 902 | 81.04% | 0.8057 | -475.26 | -0.5269 | 0.928 | 796.09 |
| 75 | 902 | 81.04% | 0.8064 | -758.41 | -0.8408 | 0.923 | 1234.75 |
| 100 | 902 | 81.04% | 0.8070 | -1064.01 | -1.1796 | 0.919 | 1693.21 |
| 125 | 902 | 81.04% | 0.8076 | -1395.28 | -1.5469 | 0.916 | 2174.52 |
| 150 | 902 | 81.04% | 0.8082 | -1752.24 | -1.9426 | 0.912 | 2679.08 |
| 175 | 902 | 81.04% | 0.8087 | -2131.30 | -2.3629 | 0.908 | 3202.61 |
| 200 | 902 | 81.04% | 0.8093 | -2531.82 | -2.8069 | 0.904 | 3745.33 |

## Test results by entry band (VWAP5)

| Band | Trades | Accuracy | Net PnL | Expectancy | Average entry |
|---|---:|---:|---:|---:|---:|
| early | 270 | 79.63% | 12.07 | 0.0447 | 46.767 |
| mid | 133 | 76.69% | -13.71 | -0.1031 | 124.098 |
| late | 499 | 82.97% | -39.98 | -0.0801 | 199.870 |

## Limitations

- Capacity evidence ends at second 240; seconds 241-299 are not evaluated.
- Open interest begins after the outcome-fit window and is diagnostic only.
- Aggregate trade prints overlap only the end of the test block and are diagnostic only.
- Projected PnL assumes the recorded VWAP is fillable at the sampled timestamp and does not model queue position.
