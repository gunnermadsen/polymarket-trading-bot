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

## Residual opportunity benchmark

The residual benchmark tests one constrained follow-up hypothesis without changing the execution
policy. It starts at the causal two-sided market probability and learns only how much of the frozen
weather-versus-market log-odds disagreement has historically added information. Midnight and noon
each receive one coefficient; both are constrained to `[0, 1]` and ridge-shrunk toward the market
baseline. There is no intercept, so YES and NO remain exact complements.

```bash
WEATHER_MODEL_IMAGE_ID="$(docker image inspect \
  capitonic/nyc-temperature-model:local --format '{{.Id}}')"
docker compose -f docker-compose.temperature.yml --profile tools run --rm temperature-model \
  residual-opportunity-benchmark \
  --source-policy-run-id SOURCE_ASYMMETRIC_POLICY_RUN_ID \
  --weather-model-image-id "${WEATHER_MODEL_IMAGE_ID}" \
  --bootstrap-iterations 1000
```

The required image ID is the immutable local image configuration ID returned by
`docker image inspect`, not a tag or a value invented inside the model container. The report records
it as a runner-declared provenance value alongside the Git revision baked into the image. The
runner must inspect the exact image that Compose will launch immediately before invoking the
benchmark.

Training uses one canonical YES label per contract and gives each event date total weight one. A
contract contributes only when at least one side has a causal five-share 4–25 cent opportunity and
the two-sided market midpoint is available. Labels enter a rolling fit only after the latest
`resolved_at` among every bucket in the ledger contract's canonical event partition is strictly
earlier than the scoring origin; the partition must have exactly one resolved winner. The model
abstains for 30 eligible event dates, refits after every 12 newly available dates, and derives its
conservative probability from both the frozen weather-distribution lower bound and a deterministic
seven-day event-block bootstrap.

The economic rule remains frozen: five shares, one cent of adverse slippage before the captured
dynamic fee, 4–25 cent all-in cost, at least four cents of robust edge, at least 35% robust ROI,
midnight entry before noon fallback, and at most one trade per event day. The report compares this
model with weather-only, market-only, equal-logit-blend, and no-trade baselines and stress-prices the
same selected trades at 0, 0.5, 1, and 2 cents of slippage.

April through July is explicitly labeled `exploratory_post_holdout_redesign`: July generated this
hypothesis and cannot validate it again. These results may reject the model or qualify it for a
forward shadow test, but `production_qualified` is always false. Positive expectancy can be
established only from event decisions occurring after the Git revision, image, and model contract
are frozen, with at least 50 prospective trades and ten observed wins and losses.
