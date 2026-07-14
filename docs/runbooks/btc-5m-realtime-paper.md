# BTC 5-minute realtime paper experiment runbook

This runbook operates the deterministic BTC Up/Down strategy, its paper venue, and the ML-A/ML-B shadow path. The experiment is research-only. Passing these checks establishes data and execution integrity; it does not establish profitability or authorize live orders.

## Safety invariants

- `POLYMARKET_LIVE_ORDER_SUBMIT_ENABLED=false` is a hard gate for this experiment.
- `POLYMARKET_BTC_REALTIME_ENABLED=true` collects realtime data locally.
- `POLYMARKET_BTC_PAPER_ENABLED=true` permits only the paper venue.
- `POLYMARKET_BTC_ML_SHADOW_ENABLED=true` records predictions but gives ML no trading authority.
- Legacy scan, Signal 2, Signal 3, copy-trade, whale-live, and whale-backfill switches remain `false`.
- Never use `docker compose down -v` during an experiment; it deletes the database volume.

The bot and audit jobs consume one shared Compose environment anchor. The
containerized preflight fails unless every safety switch has the exact value
above and every live credential is blank.

## Realtime-paper experiment cohort

The realtime-paper experiment has one north-star outcome: collect one immutable realtime-paper cohort
that can support an honest decision about deterministic strategy expectancy and
can later use the same order-plan path in a separately authorized live canary.
It does not authorize live capital. The frozen preregistration is
`docs/experiments/btc-5m-chainlink-fair-value-20260713-c/preregistration.json`,
and the only valid experiment key for this cohort is
`btc-5m-chainlink-fair-value-20260713-c`.

The target is 176 hours, with an absolute collection wall-clock budget of 192
hours. A final cohort needs at least 2,000 eligible resolved windows, 99.5%
window/label and official-resolution-SLO coverage, 100 qualifying filled FOK
orders, and seven UTC days. Reaching the clock alone does not pass the experiment;
the preregistered operational, accounting, stress, bootstrap, drawdown, tail,
leave-one-day-out, and regime-concentration gates also have to pass.

After the final image build, freeze the manifest digest in the API-controlled
process configuration. The manifest pins the realtime-paper experiment report,
the independent readiness audit, the archival
wrapper, the capital-ledger and unstarted-process migrations, and the compiled
Rust source identity.
The wrapper verifies every pinned artifact before it accepts evidence. Do not
edit the manifest or a pinned artifact after this point; any edit invalidates
the cohort and requires a new experiment key.

The digest is copied into the experiment's immutable configuration snapshot.
The identity report refuses to treat a cohort with a different digest,
pipeline version, strategy version, feature schema, execution settings, or ML
authority as the preregistered experiment.

Record the evidence filesystem's free space when this cohort is preregistered.
Because every high-volume experiment hypertable compresses after one day, the
launch floor is 20 GiB and the hard halt remains 10 GiB. The audit job checks
the writable evidence mount before launch and on every audit.

Do not launch below 20 GiB. At less than 10 GiB, archive what is safely
available, stop the trading process through its API while leaving the bot
service running, and preserve the cohort as an operational failure; do not
delete experiment evidence to keep it running.

Bot, ML research jobs, TimescaleDB, migrations, Grafana, and audits are Compose
services. Do not execute their Rust, Python, SQL, or audit entrypoints directly
on the host. Infrastructure uptime is an external operating condition; any
engine interruption remains a real coverage failure.

After TimescaleDB, the bot service, and the stopped API-controlled process
definition are ready, execute the read-only preflight; it never starts, stops,
rebuilds, migrates, or recreates a service:

```bash
docker compose run --rm btc-paper-audit preflight
```

The frozen primary venue remains 150 ms arrival latency and 80% visible depth.
Every approved primary order also records two non-mutating online previews:
300 ms/65% depth and 600 ms/50% depth. A preview must never affect the primary
decision, order plan, fill, collateral, position, or P&L. The audit additionally
recomputes a 1.25x probability-uncertainty stress from the decision's recorded fair
probability and `fair_value.probability_uncertainty`, plus the preregistered
1.25x-fee/most-profitable-10%-missed-fill stress. Stressed trades that do not
survive retain zero P&L in the original denominator.

ML-A and ML-B continue in shadow so their datasets can grow, but this remains a
deterministic strategy experiment. ML queue, canary, or task incompleteness is
diagnostic unless it causes primary snapshot/decision loss or resource
instability. Any ML authority or evidence that a shadow prediction influenced
the primary order plan is an immediate operational failure.

## Build, migrate, and start

From the repository root:

```bash
docker compose up -d timescaledb-0
docker compose --profile test --profile ops --profile research build \
  db-migrate polymarket-bot polymarket-bot-test btc-paper-audit btc-ml-research
docker compose --profile test run --rm --no-deps polymarket-bot-test
docker compose --profile research run --rm --no-deps btc-ml-research
docker compose run --rm db-migrate
docker compose up -d --no-deps polymarket-bot
docker compose ps
docker compose logs --since=10m polymarket-bot
```

The first start is explicit: create or update the stable
`btc-5m-chainlink-fair-value-paper` definition through
`PUT /admin/trading-processes/by-key/{process_key}`, then activate it through
`POST /admin/trading-processes/{process_id}/start`. Each start uses the
stopped definition's mutable `next_experiment_key` and preregistration digest;
those values become immutable in the experiment created by `/start`. Stop
trading through the matching process `/stop` endpoint. Once explicitly started,
the durable process remains the desired state across bot or Docker restarts: the
service drains only its in-memory runtime on shutdown and recreates it on startup
when the running experiment identity, frozen config, and compiled source hash
still match exactly. Container lifecycle never changes process or experiment
status, enablement, timestamps, stop reasons, configuration, or accumulated
trading data. An API stop disables that resume intent. If exact reattachment is
not possible, service startup fails visibly and leaves the database unchanged.

Managed BTC definitions use `btc_realtime_paper_process_v1`. The initial
stable-key PUT must use `enabled=false` and `status=created`:

```json
{
  "name": "BTC 5-Minute Chainlink Fair Value Paper",
  "process_type": "btc_5m",
  "process_scope": "realtime_paper",
  "enabled": false,
  "status": "created",
  "config": {
    "execution": {
      "mode": "paper",
      "execute_signals": true,
      "live_capital": false,
      "taker_fee_rate": null
    },
    "raw": {
      "btc_realtime_paper": {
        "schema_version": "btc_realtime_paper_process_v1",
        "next_experiment_key": "btc-5m-chainlink-fair-value-20260713-c",
        "preregistration_sha256": "<sha256-of-frozen-preregistration>",
        "strategy": {
          "strategy_version": "btc_5m_chainlink_fair_value_v1",
          "feature_schema_version": "btc_5m_features_v2",
          "target_size": "5",
          "min_seconds_after_open": 15,
          "min_seconds_before_close": 20,
          "max_reference_age_ms": 2000,
          "max_chainlink_open_delay_ms": 5000,
          "max_book_age_ms": 2000,
          "max_source_skew_ms": 1000,
          "max_fee_age_ms": 3600000,
          "min_entry_price": "0.05",
          "max_entry_price": "0.95",
          "max_depth_participation": "0.25",
          "volatility_floor_per_sqrt_second": "0.00005",
          "probability_floor": "0.01",
          "basis_lead_weight": "0.25",
          "momentum_1s_weight": "0.05",
          "momentum_5s_weight": "0.10",
          "momentum_30s_weight": "0.10",
          "max_lead_sigma_fraction": "0.25",
          "base_probability_uncertainty": "0.015",
          "basis_uncertainty_weight": "1",
          "feed_age_uncertainty_per_second": "0.002",
          "max_probability_uncertainty": "0.10",
          "spread_reserve_fraction": "0.10",
          "slippage_reserve_bps": "25",
          "latency_reserve_per_share": "0.005",
          "min_net_edge_per_share": "0.015",
          "min_net_edge_usd": "0.02",
          "max_fee_rate": "1"
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
        },
        "ml_shadow": { "enabled": true }
      }
    }
  },
  "metadata": {
    "strategy_family": "btc_5m_chainlink_fair_value",
    "legacy_copy_trade": false
  }
}
```

The process schema—not deployment environment variables—is authoritative for
strategy timing, sizing, paper execution assumptions, resolution timing, and
whether the optional ML shadow is requested. Network endpoints and capability
gates remain deployment-owned. Unknown or irrelevant process settings are
rejected instead of silently ignored. The full strategy object is deliberately
explicit: preflight removes only `preregistration_sha256` and requires the
remaining process config to equal
`contracts.process_definition_without_preregistration_sha256` in the frozen
manifest. Do not shorten it to a partial override for this cohort.

While stopped, rotate the next experiment identity with PATCH by sending the
complete `config` object above with a new `next_experiment_key` and digest.
PATCH must omit `enabled` and `status`; type, scope, and stable key cannot
change. A stable-key PUT against an existing definition must preserve its exact
current terminal status. Configuration updates conflict while the process is
starting, running, stopping, or waiting for a terminal database commit.

After the stopped definition matches the frozen manifest, run preflight and
only then call `/start`. Preflight calls the authenticated, read-only
`/admin/trading-processes/{process_id}/start-preview` endpoint and archives its
response. It rejects any resolver drift in the deterministic experiment UUID,
manifest digest, config hash, or exact immutable experiment config before the
experiment key is consumed:

```bash
docker compose run --rm btc-paper-audit preflight
```

After `/start` succeeds, run `docker compose run --rm btc-paper-audit daily`
and confirm that it resolves exactly that experiment.

Authenticated runtime evidence is available at
`/admin/strategy/btc-5m/readiness` and
`/admin/strategy/btc-5m/paper-experiment` using the configured admin bearer token.

Running the migration command again should report no pending migration. Confirm
that `AddBtcRealtimePaperAndMlShadow1777120000000`,
`AddBtcOfficialMarketResolution1777121000000`,
`AddBtcOfficialResolutionRecovery1777122000000`, and
`AddBtcPhase6PaperCapital1777123000000`, plus
`AllowUnstartedTradingProcesses1777124000000`, are in `public.migrations`. The
preflight also verifies the settlement ledger, all 15 frozen ledger/official
identity constraints, five required indexes, four one-day compression
policies, and a nullable `trading_processes.started_at` column. The complete
SELECT-only audit is the one-shot Compose job:

```bash
docker compose run --rm btc-paper-audit daily
```

The job requires exactly one experiment with the frozen key; it never selects an
arbitrary latest experiment. The selected experiment owns one process;
decisions outside that process, duplicate snapshot IDs, process-owned snapshots
without a decision, decisions without a process-owned snapshot, and excess
decisions per snapshot must all be zero. The ML grace excludes snapshots whose
writes may still be in flight. The official-resolution grace excludes
just-ended markets; choose it before the experiment and do not lengthen it
merely to hide delayed or missing outcomes.

The audit preserves whole-cohort diagnostics, including records created during the five-minute window in progress at startup. It reports that startup fragment separately and excludes it from the complete-window streak. Expected complete windows begin at the first aligned boundary at or after `started_at`, end no later than the experiment stop/audit cutoff, and cannot be skipped silently. The initial soak exits only when the latest streak contains at least 13 expected windows, every expected window is complete, and `consecutive_complete_interval_gate_pass` is true. Each complete window needs at least 285 of the expected 300 one-second snapshots, no gap over five seconds, first/last coverage within five seconds of its boundaries, matching decisions, at least one ready snapshot, and one valid boundary label.

Official outcome capture is websocket-primary with public CLOB REST reconciliation as the durable recovery path. The 120-second official-resolution grace is the pre-registered audit SLO; the 3600-second watch retention is a separate hard watchdog. An outcome received after 120 seconds fails the cohort even if it arrives before the watchdog. A market still unresolved at the watchdog deadline terminates the runtime visibly. Pending watches are rehydrated across restarts, and database-recovered markets are never eligible to become the current tradable market.

The script deliberately does not reduce all integrity and profitability evidence to one green/red verdict. Inspect the timestamps and counts against the gates below; an empty experiment can be healthy before the first complete five-minute boundary. The consecutive-window result is only the initial completeness gate, not a profitability verdict. An authoritative settlement requires the `btc_interval_markets` outcome, resolved timestamp, and winning token together with an allowed source, receipt timestamp, object-shaped payload, and a matching resolved durable watch. The Chainlink boundary label remains a provisional research label and is audited for agreement rather than substituted for an official result. A null official outcome is unverified, not agreement. Recomputed P&L excludes unverified fills. Embedded `observed_at` values may differ from the relational timestamp by at most one microsecond to allow the database timestamp precision boundary; larger differences fail the lineage diagnostic.

## First-boundary validation

Observe at least one entire market window, starting before its opening boundary and continuing at least five seconds beyond its close. Re-run the audit after that boundary.

Expected evidence:

- The current exact-slug market is valid and has distinct Up and Down token IDs.
- Direct Binance and RTDS Chainlink ticks remain fresh; both token books have bootstrapped checkpoints.
- The latest checkpoint and both reference sources are no more than two seconds old while feeds are healthy.
- Recent feed events have no unexplained integrity gaps, decode errors, or writer drops.
- A label uses the first Chainlink tick at or after each boundary. Open and close delays are non-negative and no greater than the configured boundary maximum (five seconds by default).
- `label_available_at` is not earlier than the close source timestamp. After the pre-registered official-resolution grace, official coverage is 100%, local receipt is within 120 seconds of close, provenance is `clob_websocket` or `clob_rest_reconciliation`, the durable watch is `resolved`, and disagreement is zero. Missing or late official outcomes remain explicit failures.
- Feature snapshots and deterministic decisions appear only with immutable snapshot IDs, schema versions, config hashes, and point-in-time timestamps.
- Exact slugs decode to their persisted epoch, epochs are divisible by 300, windows are exactly five minutes, embedded snapshot identity/lineage matches its relational columns, and all future-source/future-receipt counters are zero.
- No approved/submitted/filled buy decision occurs at or after its label became available, and no experiment has more than one entered decision per market.
- Every experiment fill has `source='paper'`; there are no live fills, orphan fills, non-buy/non-FOK paper orders, or partial FOK fills. A filled order with zero fills and a non-filled order with any fill both fail. Decision-to-plan-to-order identity, experiment/process metadata, arrival checkpoint causality, frozen latency/depth haircut, and the independently recomputed dynamic fee must match exactly. The paper execution path remains “not exercised” until at least one order plan actually fills.
- After the ML coverage grace, every process-owned cohort snapshot has exactly one ML-B fill vector/prediction and exactly one ML-B toxicity vector/prediction. Every snapshot with a deterministic fair probability also has exactly one ML-A residual vector/prediction. Missing keys, duplicate keys, excess rows, unexpected task keys, contract mismatches, vector/prediction identity mismatches, status/payload mismatches, and future feature lineage counters are all zero.
- The runtime emits only `btc_5m_ml_a_shadow_v2`, `btc_5m_ml_b_fill_shadow_v2`, and `btc_5m_ml_b_toxicity_shadow_v2` with the pinned model/artifact/schema hashes in the audit. `schema_canary=true`, each canary reproduces its provided prior, predictions follow the deterministic decision, worker rejection/failure counters remain zero, and ML has `execution_authority=false`. Shadow output does not alter deterministic decisions or order plans.

When a feed reconnects, the books must bootstrap from a full snapshot before decisions resume. Treat snapshots and decisions during stale, crossed, unbootstrapped, or source-skewed states as integrity failures even if a trade was not placed.

## Monitoring cadence

For the initial soak, inspect logs and the audit at startup, after the first boundary, after 65 minutes, and after any reconnect or persistence error. Do not stop at the thirteenth close: the final market is not audit-eligible until its 120-second official-resolution SLO has elapsed. Earliest final eligibility is the first aligned boundary plus 65 minutes plus 120 seconds—up to almost 72 minutes after an arbitrary launch. Keep the bot running until every selected market has official provenance and no selected watch is pending. Archive authenticated runtime status and cumulative metrics before stopping because not every counter is durable. For the longer experiment, audit at least daily and archive the output with the experiment key, code commit, strategy version, feature schema version, and config hash.

The audit container resolves the exact immutable key and rejects zero or
multiple matches. To pin the UUID explicitly for an incident audit, pass it as
the second argument:

```bash
docker compose run --rm btc-paper-audit daily <experiment-uuid>
```

Run that wrapper at startup/first boundary, daily, after any feed reconnect,
writer/database error, engine interruption, disk warning, OOM event, or
unexpected execution result. It creates a non-overwriting, checksummed evidence
directory under the experiment key. The admin token is used in memory and is
never written.

At 176 hours, first run a daily audit while the service is still available so
volatile health, metrics, and authenticated runtime state are captured. Wait
through the final market's 120-second official-resolution SLO and ensure no
selected watch remains pending. Stop the trading process through the authenticated
process API, leave the bot service running, and run the final audit against the
same UUID:

```bash
docker compose run --rm btc-paper-audit daily
docker compose --profile ops run --rm --no-deps \
  -e PROCESS_ID=<process-uuid> --entrypoint sh btc-paper-audit -ec \
  'curl -fsS -X POST -H "Authorization: Bearer ${POLYMARKET_HTTP_ADMIN_TOKEN:-dev-polymarket-admin}" \
    "$POLYMARKET_HTTP_BASE_URL/admin/trading-processes/$PROCESS_ID/stop"'
docker compose run --rm btc-paper-audit final <experiment-uuid>
```

Stop no later than 192 hours even if a sample gate is still short. Do not extend
the cohort after looking at results. The authoritative final classification is
in `paper-experiment-report.txt` and its machine-readable copy is
`paper-experiment-classification.json`:

- `COLLECTING`: still running or genuinely short of preregistered sample size.
- `OPERATIONAL_FAILURE`: identity, data, official outcome, safety, FOK,
  non-mutating preview, lineage, or accounting evidence is invalid; make no
  profitability claim.
- `NO_VIABLE_EDGE`: operational evidence passes, but normal or stressed point
  expectancy is non-positive.
- `PROMISING_INCONCLUSIVE`: point estimates are positive, but confidence,
  tail, drawdown, day/regime concentration, or leave-one-day-out evidence does
  not pass.
- `EDGE_SUPPORTED_FOR_LIVE_CANARY_PLANNING`: every preregistered gate passes. This
  allows planning a separately gated tiny live FOK canary; it is not production
  authorization.

The wrapper exits `30` for `OPERATIONAL_FAILURE` and exits `31` when a final
audit is still `COLLECTING`. All evidence directories are finalized and
checksummed even on these classified failures. Other nonzero exits are
operational audit failures; inspect `audit-status.json` and the named blocker
file in that immutable evidence directory.

The Grafana current-state panels deliberately require an enabled, non-backtest
process in `starting`, `running`, or `stopping`, with no `stopped_at` and a
heartbeat/update no older than two minutes. Stopped experiments are retained
only in historical realized-P&L panels. Consequently, current P&L is expected
to read zero after a clean experiment stop. BTC paper P&L is sourced from the
experiment, order, paper-fill, official-resolution, and order-book tables;
legacy position P&L is used only for non-BTC processes.

TimescaleDB's 2 GB container limit must stay aligned with the explicit
PostgreSQL settings in Compose. After a resource/configuration change, verify
the active values, cgroup peak, and OOM counters before running a heavy audit:

```bash
docker compose config
docker compose exec -T timescaledb-0 psql -U postgres -d polymarket -c \
  "select name, setting from pg_settings where name in
   ('shared_buffers','effective_cache_size','maintenance_work_mem','work_mem',
    'max_connections','max_parallel_workers','max_parallel_workers_per_gather',
    'max_worker_processes','timescaledb.max_background_workers') order by name"
docker compose exec -T timescaledb-0 sh -c \
  'cat /sys/fs/cgroup/memory.current; cat /sys/fs/cgroup/memory.peak; cat /sys/fs/cgroup/memory.events'
```

Run the exit-gate audit with a statement timeout and a deliberately small
session `work_mem` when validating query-plan safety. A PostgreSQL memory kill
is an audit failure even if crash recovery succeeds.

Do not start another experiment merely to clear an unattractive result. Every
new trading-process start must use a new immutable experiment key;
configuration, schema, data-gap, or model changes likewise define a new
segment. Preserve the old segment and record the stop reason.

## Halt and rollback

Immediately halt trading on any live-order attempt, live fill,
stale/crossed/unbootstrapped book used by an approved decision, boundary-label
violation, epoch/lineage mismatch, future-data violation, unexplained duplicate
entry, persistent missing/duplicate ML key beyond its grace, persistent writer
drops, or database errors. Use the authenticated process API; leave the bot
service running so status, metrics, and evidence remain available:

```bash
docker compose --profile ops run --rm --no-deps \
  -e PROCESS_ID=<process-uuid> --entrypoint sh btc-paper-audit -ec \
  'curl -fsS -X POST -H "Authorization: Bearer ${POLYMARKET_HTTP_ADMIN_TOKEN:-dev-polymarket-admin}" \
    "$POLYMARKET_HTTP_BASE_URL/admin/trading-processes/$PROCESS_ID/stop"'
docker compose ps
docker compose logs --since=30m polymarket-bot
```

Confirm that reference ticks, checkpoints, decisions, and fills stop advancing. Preserve logs and the audit output before changing data.

To keep trading disabled across a restart, stop the stable trading-process
definition through the process API before stopping Docker. Restarting or
recreating the container preserves an explicitly running paper process's resume
intent; it is not a substitute for the process stop endpoint.

Do not run a migration `down` against experiment data as an operational
rollback. Roll back the application image/config while leaving the
forward-compatible schema in place. If a schema rollback is unavoidable, stop
the trading process through the API, take and verify a database backup, and
restore into a separate database first.

## Retention and backfill policy

The migration configures these retention windows: retained market-feed control/anomaly events 14 days, order-book checkpoints 90 days, reference ticks 180 days, and features, ML vectors, and shadow predictions 365 days. Applied price deltas update the in-memory book but are not retained individually; one-second checkpoints are the durable execution/training record. Compression begins earlier. Verify the Timescale jobs in the audit rather than assuming they are scheduled.

No historical market-data backfill is required to begin a forward realtime paper experiment. Old wallet, address, copy-trade, replay, and process data is not an input to this strategy and should not be backfilled for it. Startup does reconcile any previously paper-filled BTC markets whose official result is still absent; this repairs accounting obligations and is not strategy-data backfill.

Do not synthesize or interpolate missing order books or boundary Chainlink ticks. Mark the affected interval incomplete and exclude it from execution and model training. Market metadata may be backfilled from Gamma with its retrieval time and raw provenance. Reference data may be imported for separate offline research only when the source timestamp, receive/import timestamp, source, and integrity status remain explicit; it must not be blended into the realtime paper cohort.

Before retention removes source rows used by a durable ML dataset, freeze a point-in-time dataset manifest, hashes, time range, exclusion rules, label version, and source lineage. Keep training, validation, and test splits grouped by market and chronological.

## Exit gates

Realtime data-plane readiness requires:

- migrations and Timescale policies are present;
- all three feeds persist fresh data and both books re-bootstrap correctly after a tested reconnect;
- `consecutive_complete_interval_gate_pass` is true for at least 13 expected, consecutive, complete intervals (the startup-partial interval is excluded), with valid open/close labels and no unexplained integrity gaps or drops; and
- stale or incomplete inputs demonstrably fail closed.

Deterministic-signal readiness requires:

- feature/decision hashes and lineage are complete and reproducible;
- point-in-time and post-label decision violations are zero;
- the one-entry-per-experiment/market invariant holds; and
- every rejection and approval is attributable to a stable reason/config version.

Paper execution-parity readiness requires:

- approved intents compile through the shared order-plan path;
- every paper fill uses an arrival-time checkpoint, configured latency and depth haircut, and the dynamic fee schedule;
- paper FOK atomicity, order/fill linkage, and independently recomputed P&L reconcile; and
- live submission remains disabled and the paper process has zero live fills.

Experiment-operations readiness requires:

- halt/restart and reconnect drills preserve the invariants above;
- daily audit evidence is retained for the experiment; and
- config or code changes create a new identifiable experiment segment.

ML-A/ML-B schema-canary completion requires the shared runtime-v2 fixture to pass in both languages and cohort-scoped shadow predictions to be complete:

```bash
docker compose run --rm btc-ml-research
docker compose --profile test build polymarket-bot polymarket-bot-test
docker compose --profile test run --rm --no-deps polymarket-bot-test
```

These tests must reproduce the exact v2 schema, artifact, manifest, and vector hashes pinned in `experiments/btc-updown/fixtures/runtime_v2_contract.json`. Any contract edit requires a deliberate version bump and a new fixture; do not update expected hashes merely to make a drifted implementation pass. This is schema/lineage canary completion, not learned-model validation. ML-A training must eventually enforce official CLOB label provenance, and ML-B still needs explicit FOK fill/toxicity label builders. ML remains shadow-only until those definitions and the evidence gates below pass.

## Profitability and promotion evidence

Operational exit gates are not profitability evidence. Pre-register thresholds before inspecting outcomes. At minimum, use a chronological, untouched out-of-sample period spanning multiple volatility, liquidity, weekday, and time-of-day regimes; four weeks is a reasonable first collection horizon, not proof.

Do not promote the deterministic strategy unless:

- net expectancy is computed from arrival-time paper fills, exact fees, and resolved payouts;
- the lower bound of a 95% confidence interval for mean net P&L is above zero, with resampling grouped by UTC day rather than treating correlated five-minute markets as independent;
- the result survives pre-registered latency, visible-depth, fee, spread, and missed-fill stresses;
- expectancy is not concentrated in a small number of days or a single market regime; and
- drawdown and tail loss remain inside pre-registered risk limits.

Require enough resolved filled trades for the confidence interval and regime slices to be meaningful; choose that count before viewing performance. If the interval is wide, the conclusion is “insufficient evidence,” not profitable or unprofitable.

ML-A must improve out-of-sample log loss, Brier score, and calibration over the frozen deterministic prior, then improve the same timestamp-matched after-cost counterfactual without worsening tail risk. ML-B is defined for the actual taker path: FOK fill probability and post-fill toxicity, not maker queue probability. It must demonstrate calibrated probabilities and incremental economic value under the same order eligibility set. Model selection, calibration, and thresholds use training/validation only; the final test set is evaluated once. Passing shadow evaluation authorizes another controlled paper experiment, not live trading.
