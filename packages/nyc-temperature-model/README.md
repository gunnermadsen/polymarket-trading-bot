# NYC temperature expectancy benchmark

This package procures public NYC daily-high market data and tests whether a frozen probabilistic
weather policy has positive net expectancy. It does not submit orders, paper trade, load wallet
credentials, or run the Polymarket trading-bot container.

The runtime is isolated in `docker-compose.temperature.yml`:

- a private PostgreSQL database with no published host port;
- a weather-only migration command whose migrations are not loaded by the normal bot migrator;
- four leased weather/archive workers and four dedicated CLOB price-history workers with public
  read-only source access;
- an on-demand offline model runner;
- SSD-backed cache, database, model, and report directories under
  `/Volumes/docker-data/polymarket-bot/temperature-expectancy` by default.

## Start and procure data

```bash
docker compose -f docker-compose.temperature.yml up -d --build
docker compose -f docker-compose.temperature.yml --profile tools run --rm temperature-model \
  enqueue-pilot --weather-start 2019-01-01 --market-start 2025-09-01 --end 2026-08-01
docker compose -f docker-compose.temperature.yml --profile tools run --rm temperature-model jobs
```

`enqueue-pilot` splits both the Wunderground-aligned IEM METAR archive and the IEM-processed NCEI
one-minute ASOS archive into annual jobs, and HRRR, price-history, and PMXT work into monthly
jobs. Market-dependent jobs wait for the market census job. Re-running it is idempotent.

The METAR archive is the canonical resolution-label proxy. The one-minute archive is retained as
auxiliary sensor context only because it can diverge from Wunderground's daily maximum. A market
cannot qualify unless the METAR-derived winning bucket agrees with Polymarket on at least 99% of
complete days.

The HRRR worker assumes a 75-minute publication allowance and selects only model cycles available
before each decision. It retrieves only two-metre-temperature GRIB byte ranges for the KLGA grid
point. Completed decision fields are reused after interruption. Individual downloads rotate through
Google, AWS, and NOMADS archives with bounded exponential retry and request pacing before a monthly
job is retried. The PMXT worker reconstructs the immediately available YES and NO asks using provider
receipt time, never events received after the decision. Each worker has an independent cache
namespace; large PMXT transport files are removed only after compact snapshots and checksums commit.
Workers use validated ingester allowlists so the CLOB price-history pool cannot consume HRRR or PMXT
jobs, while the general pool cannot consume price-history jobs.

## Reconcile, train, and benchmark

```bash
docker compose -f docker-compose.temperature.yml --profile tools run --rm temperature-model \
  reconcile-labels --start 2025-09-01 --end 2026-08-01

docker compose -f docker-compose.temperature.yml --profile tools run --rm temperature-model \
  train --candidate histogram_residual --decision-hour 0 \
  --training-start 2019-01-01 --training-end 2024-12-31 \
  --calibration-start 2025-01-01 --calibration-end 2025-12-31

docker compose -f docker-compose.temperature.yml --profile tools run --rm temperature-model \
  asymmetric-benchmark \
  --midnight-model-run-id MIDNIGHT_MODEL_RUN_ID \
  --noon-model-run-id NOON_MODEL_RUN_ID \
  --discovery-start 2026-04-14 --discovery-end 2026-06-30 \
  --evaluation-start 2026-07-01 --evaluation-end 2026-07-30 \
  --quantity 5 --modeled-slippage 0.01
```

Train all three candidates for midnight and noon. Select each decision-time ML model using the
lowest pre-2026 leave-one-out rounded-temperature distribution log loss, with ranked probability
score secondary and RMSE retained as a point-forecast diagnostic. The executable 2026 policy
discovery and July holdout must not choose the forecast model. July 31 is excluded because its
canonical station day is incomplete in the collected archive. Historical CLOB price history is
diagnostic only; this benchmark uses causal PMXT five-share ask VWAPs.

The asymmetric benchmark predicts the complete mutually exclusive temperature-bucket distribution,
expands every bucket into actual executable YES and NO candidates, and selects at most one position
per event day. It does not require 90% accuracy. The predeclared nine-policy frontier admits a trade
only when the conservative probability estimate exceeds captured-schedule, adverse-slippage,
five-share VWAP break-even cost by the policy's edge and ROI margins. Dynamic fees are recomputed at
the slipped VWAP and rounded to five decimals. Because archived individual match levels are not
persisted, this is a conservative fee-curve approximation at the order VWAP, not an exact per-fill
fee reconstruction. The discovery winner is frozen before the July
holdout is evaluated. Reports include price-conditioned calibration, fixed-five-share PnL, return on
deployed capital, equity and drawdown, worst-loss concentration, an expensive-share comparator, an
uncapped comparator, and frozen-policy slippage stress at 0, 0.5, 1, and 2 cents per share.
Price-cell uncertainty resamples complete event dates in seven-day blocks; it never treats the
mutually exclusive bucket contracts from one day as independent observations. Label qualification
requires at least 100 reconciled days, at least 95% coverage of resolved event days, and at least 99%
agreement with the canonical station result.

`pilot_edge_supported` is an evidence label, not a deployment guarantee. `production_qualified`
also requires a positive one-sided block-bootstrap bound, robustness without the best trade, and a
larger balance of observed wins and losses. If those conditions fail, the valid conclusion is that
the collected history has not demonstrated a deployable edge; thresholds must not be changed after
opening the holdout.
