# Polymarket Bot Standalone Stack

Standalone Docker stack for the Polymarket bot, its Polymarket-only migrations, and a dedicated TimescaleDB database.

## Services

- `timescaledb-0`: dedicated TimescaleDB/Postgres database.
- `db-migrate`: one-shot TypeORM migration runner with only `polymarket` migrations.
- `polymarket-bot`: Rust bot service.
- `grafana`: provisioned Grafana instance with the Postgres datasource and `polymarket-bot` dashboard.

Kafka and pgbouncer are intentionally omitted.

## Local Start

Create local env files from the examples and fill in the secret values:

- `.env`: app and Polymarket secrets.
- `.env.postgres`: canonical source for `POSTGRES_PASSWORD`.
- `.env.grafana`: Grafana admin credentials and datasource settings.

Generate a local Grafana admin password with:

```bash
openssl rand -hex 32
```

Non-secret runtime configuration belongs in `docker-compose.yml`.

```bash
docker compose up -d timescaledb-0
docker compose up --build db-migrate
docker compose build polymarket-bot
docker compose up -d polymarket-bot grafana
```

Starting the service leaves BTC trading inactive. Next, create or update the
stopped API-controlled process definition and call that process's `/start`
endpoint. Trading activity remains controlled by persistent process state, not
by starting or stopping the bot container.

The local Compose file publishes Postgres on
`POLYMARKET_POSTGRES_HOST_PORT` (default `55433`) and bot HTTP on
`POLYMARKET_HTTP_HOST_PORT` (default `8098`).

TimescaleDB is capped at 2 GB and starts with matching PostgreSQL memory,
connection, and worker limits. Keep the explicit `postgres -c` settings in the
Compose file: the `TS_TUNE_*` variables tune only a newly initialized data
volume and do not repair an existing volume by themselves. Grafana is limited
to five open and two idle datasource connections so it fits within the
25-connection PostgreSQL budget.

Health:

```bash
curl -fsS "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/health"
curl -fsS "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/metrics"
```

Grafana:

Open `http://127.0.0.1:3030` in a browser.

Log in with `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` from `.env.grafana`.
The provisioned dashboard refreshes every 30 seconds. The BTC five-minute
countdown uses a provisioned Grafana Live channel and updates every second,
independently of that global refresh. Current process and P&L
panels include only enabled processes in a running lifecycle with
a heartbeat/update no older than two minutes. Stopped experiments remain in
the realized-P&L history but do not contribute to current realized or
unrealized totals. Recreate Grafana after changing the provisioned JSON:

```bash
docker compose up -d --no-deps --force-recreate grafana
```

## Live Credentials

Set the CLOB credential secrets in `.env`:

```bash
POLYMARKET_CLOB_API_KEY=...
POLYMARKET_CLOB_SECRET=...
POLYMARKET_CLOB_PASSPHRASE=...
POLYMARKET_PRIVATE_KEY=...
POLYMARKET_FUNDER_ADDRESS=...
POLYMARKET_SIGNATURE_TYPE=...
```

Starting or stopping the `polymarket-bot` container does not select trading
activity. Trading process definitions are mutable and API-driven; use
`POST /admin/trading-processes/{process_id}/start` and
`POST /admin/trading-processes/{process_id}/stop` while leaving the service
running. Each BTC paper start creates a distinct immutable experiment while the
stable process key can be reused. Trading mode is selected by
`trading_processes.config.execution.mode`; host configuration is limited to
credentials, venue URLs, and hard risk caps.

## BTC Five-Minute Chainlink Process Contract

Legacy Chainlink definitions and existing durable processes use
`btc_realtime_paper_process_v2`; new definitions that need explicit strategy
selection use the v3 contract below. The stable identity is
`process_type=btc_5m`, `process_scope=realtime_paper`, plus a unique
`process_key`. The execution contract is paper-only, executes approved signals,
and never permits live capital. Unknown fields inside
`config.raw.btc_realtime_paper` are rejected. In particular, the retired
`ml_shadow` setting is not part of this contract.

Save the following request body as `btc-process-v2.json`. Replace the process
name, `next_experiment_key`, and preregistration digest before creating a real
process. The digest must be exactly 64 hexadecimal characters.

<!-- btc-5m-process-v2:start -->
```json
{
  "name": "BTC 5m Chainlink paper",
  "process_type": "btc_5m",
  "process_scope": "realtime_paper",
  "enabled": false,
  "status": "created",
  "config": {
    "execution": {
      "mode": "paper",
      "execute_signals": true,
      "live_capital": false
    },
    "raw": {
      "btc_realtime_paper": {
        "schema_version": "btc_realtime_paper_process_v2",
        "next_experiment_key": "btc-5m-chainlink-paper-example-v1",
        "preregistration_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "strategy": {},
        "runtime": {
          "strategy_interval_ms": 1000,
          "official_resolution_audit_grace_secs": 120,
          "official_resolution_watch_retention_secs": 3600
        },
        "paper": {
          "arrival_latency_ms": 150,
          "visible_depth_haircut": "0.80",
          "starting_collateral_usd": "1000",
          "stress_previews": [
            {
              "scenario_key": "latency_300ms_depth_65pct",
              "arrival_latency_ms": 300,
              "visible_depth_haircut": "0.65"
            },
            {
              "scenario_key": "latency_600ms_depth_50pct",
              "arrival_latency_ms": 600,
              "visible_depth_haircut": "0.50"
            }
          ]
        }
      }
    }
  },
  "metadata": {}
}
```
<!-- btc-5m-process-v2:end -->

Entry admission is optional. When `entry_admission` is absent, BTC process
behavior and its frozen process configuration are unchanged. When the following
block is present beside `strategy`, `runtime`, and `paper`, it is enforced; there
is no passive mode or environment-variable control:

```json
"entry_admission": {
  "loss_regime_confidence_floor": {
    "schema_version": "loss_regime_confidence_floor_v1",
    "activation_consecutive_candidate_losses": 2,
    "min_conservative_probability": "0.50",
    "release_consecutive_candidate_wins": 1
  }
}
```

## Selectable BTC Decision Strategies

`btc_realtime_paper_process_v3` requires an explicit
`strategy.decision_strategy`. The selector chooses the probability estimator
and its decision contract. The legacy selectors continue to use the common
reserve-aware edge comparison and intent builder. Entry-admission features run
after strategy evaluation and remain independent downstream gates.

The supported selectors are:

- `chainlink_fair_value`, the existing two-sided Chainlink estimator.
- `volatility_continuation`, the existing one-sided continuation strategy with
  its configuration nested under the selector's `config` field.
- `market_anchored_fair_value`, the two-sided research estimator that starts
  from the normalized Polymarket midpoint and applies bounded external
  Chainlink/Binance evidence.
- `market_anchored_directional_prediction`, a separately versioned research
  strategy that reuses the immutable market-anchored probability profile but
  turns a sufficiently strong estimate into an explicit Up or Down prediction.

The directional-prediction strategy selects Up or Down using the greater
central calibrated probability and requires its conservative probability to be
at least the frozen `0.75` floor. A weaker estimate is recorded as no
prediction. A qualifying prediction is recorded even when no intent can be
built, so research can distinguish prediction quality from an economic or
execution rejection. An intent requires the selected central probability to
exceed the executable price plus the actual dynamic taker fee per share. This
strategy does not apply the legacy reserve-based or minimum-edge admission
thresholds; freshness, runtime readiness, price and depth checks, one entry per
market, and simulated FOK arrival checks remain unchanged.
For its approved decision rows, the canonical edge columns describe that direct
approval contract (central probability, actual fee, and zero contractual
reserve); the hypothetical legacy reserve breakdown remains available in the
serialized decision metadata.

A minimal directional selector is:

```json
"decision_strategy": {
  "type": "market_anchored_directional_prediction",
  "profile_id": "btc5m-market-anchored-research-20260718-v1",
  "profile_sha256": "5f84df367641cd6f6144ef20f8e6044f4de43710f9caa16a7da7e9e8f888c99a",
  "config": {
    "min_conservative_probability": "0.75"
  }
}
```

The candidate profile is compiled into the binary and selected by both ID and
content hash. A profile mismatch fails process validation. Its current
`research_only` profile uses identity calibration and does not claim a trained
or production-calibrated model: the available retrospective evidence covers
only about 4.3 days and roughly 694 independently resolved markets. The initial
live use is therefore paper research.

<!-- btc-5m-process-v3:start -->
```json
{
  "name": "BTC 5m Market Anchored Fair Value paper min entry 0.30 research",
  "process_type": "btc_5m",
  "process_scope": "realtime_paper",
  "enabled": false,
  "status": "created",
  "config": {
    "execution": {
      "mode": "paper",
      "execute_signals": true,
      "live_capital": false
    },
    "raw": {
      "btc_realtime_paper": {
        "schema_version": "btc_realtime_paper_process_v3",
        "next_experiment_key": "btc-5m-market-anchored-research-example-v1",
        "preregistration_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "strategy": {
          "decision_strategy": {
            "type": "market_anchored_fair_value",
            "profile_id": "btc5m-market-anchored-research-20260718-v1",
            "profile_sha256": "5f84df367641cd6f6144ef20f8e6044f4de43710f9caa16a7da7e9e8f888c99a"
          },
          "min_entry_price": "0.30"
        },
        "runtime": {
          "strategy_interval_ms": 1000,
          "official_resolution_audit_grace_secs": 120,
          "official_resolution_watch_retention_secs": 3600
        },
        "paper": {
          "arrival_latency_ms": 150,
          "visible_depth_haircut": "0.80",
          "starting_collateral_usd": "1000",
          "stress_previews": [
            {
              "scenario_key": "latency_300ms_depth_65pct",
              "arrival_latency_ms": 300,
              "visible_depth_haircut": "0.65"
            },
            {
              "scenario_key": "latency_600ms_depth_50pct",
              "arrival_latency_ms": 600,
              "visible_depth_haircut": "0.50"
            }
          ]
        }
      }
    }
  },
  "metadata": {}
}
```
<!-- btc-5m-process-v3:end -->

For an explicit Chainlink v3 process, replace the candidate selector with
`{"type":"chainlink_fair_value"}`. Existing v2 definitions intentionally stay
on the legacy inference contract so durable experiments resume without a
selector or config-hash change.

The loss-regime state is reconstructed from the process's immutable,
configuration-scoped decision history. Each resolved market contributes the
earliest strategy-approved buy candidate produced before its label became
available, including candidates deferred by admission. The ordered state is
rebuilt at the first admission evaluation in each new market, so delayed labels
are incorporated without replaying growing history on every one-second
opportunity. After the configured number of consecutive candidate losses,
approved entries below the conservative probability floor are recorded as
`admission_blocked` without creating an order. Entries at or above the floor
continue through the existing paper execution path. The floor releases after
the configured number of consecutive candidate wins. Admission evidence is
stored with each evaluated buy decision.

Create the definition through the stable-key API, preview the immutable run,
then start it explicitly:

```bash
PROCESS_KEY="btc-5m-chainlink-paper"
PROCESS_ID="$({
  curl -fsS -X PUT \
    "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/admin/trading-processes/by-key/${PROCESS_KEY}" \
    -H "Authorization: Bearer ${POLYMARKET_HTTP_ADMIN_TOKEN}" \
    -H "Content-Type: application/json" \
    --data @btc-process-v2.json
} | jq -r '.process.process_id')"

curl -fsS \
  "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/admin/trading-processes/${PROCESS_ID}/start-preview" \
  -H "Authorization: Bearer ${POLYMARKET_HTTP_ADMIN_TOKEN}"

curl -fsS -X POST \
  "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/admin/trading-processes/${PROCESS_ID}/start" \
  -H "Authorization: Bearer ${POLYMARKET_HTTP_ADMIN_TOKEN}"
```

Use the status and stop endpoints for lifecycle management:

```bash
curl -fsS \
  "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/admin/trading-processes/${PROCESS_ID}/status" \
  -H "Authorization: Bearer ${POLYMARKET_HTTP_ADMIN_TOKEN}"

curl -fsS -X POST \
  "http://127.0.0.1:${POLYMARKET_HTTP_HOST_PORT:-8098}/admin/trading-processes/${PROCESS_ID}/stop" \
  -H "Authorization: Bearer ${POLYMARKET_HTTP_ADMIN_TOKEN}"
```

An existing running v1 process may be reattached after a service restart, but
v1 is resume-only. Once stopped, it must be updated to the v2 contract before
another explicit start. Process type, scope, and stable key are immutable;
name and configuration may be updated only while the process is stopped.
