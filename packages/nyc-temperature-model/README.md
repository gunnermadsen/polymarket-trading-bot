# NYC temperature expectancy benchmark

This package procures public NYC daily-high market data and tests whether a frozen probabilistic
weather policy has positive net expectancy. It does not submit orders, paper trade, load wallet
credentials, or run the Polymarket trading-bot container.

The runtime is isolated in `docker-compose.temperature.yml`:

- a private PostgreSQL database with no published host port;
- a weather-only migration command whose migrations are not loaded by the normal bot migrator;
- two leased ingestion workers with public read-only source access;
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
point. The PMXT worker reconstructs the immediately available YES and NO asks using provider receipt
time, never events received after the decision. Each worker has an independent cache namespace;
large PMXT transport files are removed only after compact snapshots and checksums commit.

## Reconcile, train, and benchmark

```bash
docker compose -f docker-compose.temperature.yml --profile tools run --rm temperature-model \
  reconcile-labels --start 2025-09-01 --end 2026-08-01

docker compose -f docker-compose.temperature.yml --profile tools run --rm temperature-model \
  train --candidate histogram_residual --decision-hour 0 \
  --training-start 2019-01-01 --training-end 2024-12-31 \
  --calibration-start 2025-01-01 --calibration-end 2025-08-31

docker compose -f docker-compose.temperature.yml --profile tools run --rm temperature-model \
  benchmark --model-run-id MODEL_RUN_ID \
  --evaluation-start 2026-04-14 --evaluation-end 2026-08-01 \
  --quantity 5 --evidence-tier executable_taker --safety-buffer 0.02
```

Run the midnight and noon policies independently. Train all three candidates, but select the
candidate using only the training and calibration periods. Historical CLOB price history is an
indicative benchmark only. It cannot pass strict qualification; only causal PMXT taker VWAP can.

Strict qualification requires all of the following: at least 99% station/market label agreement,
better log loss than raw HRRR, ECE no greater than 5%, at least 75 independent executable event
days, positive five-share net expectancy after fees and a two-cent safety buffer, a positive lower
90% block-bootstrap bound, positive chronological folds, and positive expectancy after removing
the five best days.
