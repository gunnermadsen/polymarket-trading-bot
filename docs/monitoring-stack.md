# Capitonic Monitoring Stack

## Purpose and scope

This document defines the operating standards for Capitonic's monitoring stack and records how the stack is currently declared and running. It covers Grafana, Prometheus, Loki, Grafana Alloy, and PostgreSQL/TimescaleDB.

The monitoring system exists to answer three different classes of question:

1. **Is the runtime healthy?** Prometheus metrics describe availability, freshness, throughput, latency, failures, and resource pressure.
2. **Why did an event occur?** Structured logs collected by Alloy and stored in Loki preserve diagnostic context.
3. **What durable outcome occurred?** PostgreSQL/TimescaleDB remains the authoritative source for orders, fills, positions, reconciliation, accounting, trading-process configuration, and other business state.

Grafana is the common query, visualization, correlation, and alert-evaluation layer. It is not the authoritative store for any of these signals.

## Pipeline model

```text
polymarket-bot / other services
  |-- /prometheus/metrics --pull every 20s--> Prometheus --PromQL--+
  |-- structured stdout --Docker API--> Alloy --> Loki --LogQL-----+--> Grafana
  |-- operational records ----------------> PostgreSQL --SQL------+       |
  `-- selected 1s runtime values ----------> Grafana Live -----------------'
```

Each signal has one preferred home:

| Signal | Preferred system | Examples |
|---|---|---|
| Bounded numeric time series | Prometheus | availability, latency, age, counts, rates, queue depth |
| Event and diagnostic context | Loki | error reason, component activity, recovery sequence |
| Durable business and integrity state | PostgreSQL | orders, fills, positions, PnL, reconciliation, settlement |
| Low-latency transient presentation | Grafana Live | one-second countdown or selected runtime display |

Do not duplicate durable business truth into metrics or logs and then treat the duplicate as authoritative. A metric may report that a fill occurred, but PostgreSQL must prove the fill and its accounting effect.

## Current deployment

### Declared production deployment

`docker-compose.production.yml` declares a single-host Docker Compose deployment:

| Component | Declared configuration |
|---|---|
| PostgreSQL | TimescaleDB 2.14.2 on PostgreSQL 14, persistent volume, 2 GB memory, 32 connections, loopback host port `55432` |
| PgBouncer | Session-pooled application and Grafana access on internal port `6432` |
| Prometheus | Version 3.13.2 pinned by digest, 384 MB, 0.20 CPU, persistent volume, 15-day or 5 GB retention |
| Loki | Version 3.7.6 pinned by digest, 384 MB, 0.20 CPU, local filesystem TSDB, 30-day retention, replication factor 1 |
| Grafana | Version 12.4.2 by default, 256 MB, 0.20 CPU, persistent volume, loopback host port `3000` |

Prometheus scrapes itself and `polymarket-bot:8097/prometheus/metrics` every 20 seconds with a 10-second timeout. Its externally attached labels identify `environment`, `service`, and `deployment`.

Grafana provisions four data sources at startup:

- PostgreSQL/TimescaleDB through PgBouncer, with two open and one idle connection in production.
- Prometheus through server-side proxy access and basic authentication.
- Loki through server-side proxy access.
- The bot's authenticated runtime HTTP API through the Infinity data-source plugin for selected readiness alerts.

Dashboards and alert rules are provisioned from version-controlled files and are not editable in the Grafana UI. Grafana stores unified-alerting state history in Loki. The declared production stack does **not** currently include a Grafana Alloy service, so its Compose declaration alone does not deliver application logs to Loki.

### Observed local runtime on 2026-08-27

A read-only `docker compose ps` inspection found Grafana, Prometheus, Loki, Alloy, TimescaleDB, PgBouncer, `polymarket-bot`, `ingester-master`, and `ingester-worker` running in the shared `polymarket-bot` Compose project.

The observed monitoring containers were assembled from more than one checkout:

- TimescaleDB, PgBouncer, Loki, and `polymarket-bot` retained configuration labels from the main repository checkout.
- Grafana, Prometheus, Alloy, and `market-data-ingester` were launched from `target/worktrees/market-data-ingester-observability` on branch `feature/market-data-ingester-observability` at commit `992d5ca`.

That feature deployment differs from the production declaration:

- Prometheus uses 7-day or 2 GB retention and is published on loopback port `9090`.
- Grafana is published on loopback port `3030` and has 512 MB memory and 0.50 CPU.
- Alloy 1.18.0 is pinned by digest and mounts the Docker socket read-only.
- Alloy discovers Docker containers and keeps the canonical `ingester-master` and `ingester-worker` runtime names.
- Alloy parses JSON fields `level`, `target`, `fields.strategy`, and `fields.error_code` into Loki labels, drops discovered log history older than one hour, and sends accepted entries to `http://loki:3100/loki/api/v1/push`.
- Alloy does not currently collect `polymarket-bot`, PostgreSQL, PgBouncer, Prometheus, Grafana, Loki, or other worker logs.

This observed state is useful development evidence, but it is not a canonical production definition because the Alloy configuration exists only in an unmerged feature worktree.

## Usage standards

### Common identity and correlation

- Use `process_id` as the canonical correlation key for trading-process behavior in metrics, logs, and SQL.
- Use stable, bounded dimensions such as `service`, `component`, `environment`, `exchange`, `status`, and a controlled `reason` or `error_code` vocabulary.
- Do not use `experiment_id` for ownership or scoping.
- Propagate a correlation or request identifier in logs for multi-step operations, but do not make unbounded identifiers Prometheus labels or Loki index labels.
- Record service image revision and deployment identity so an operational change can be correlated with the exact executable.

### Prometheus standards

- Expose Prometheus text format on `/prometheus/metrics`; reserve `/metrics` for the existing JSON runtime response.
- Use counters for cumulative events, gauges for current state or age, and histograms for latency or size distributions.
- Name metrics with the `polymarket_` namespace, base units, and standard suffixes such as `_total`, `_seconds`, `_bytes`, or `_info`.
- Counters must be monotonic and queried with `rate()` or `increase()` over an explicit window.
- Readiness gauges should express observable evidence, not permanently disable a trading process. A transient unhealthy value blocks only the unsafe action and must recover automatically when evidence becomes healthy.
- Never put order IDs, transaction hashes, market IDs with unbounded growth, complete URLs, free-form errors, user-controlled strings, or timestamps in labels.
- Before adding a label, estimate its maximum number of values and the cross-product with all other labels. If the bound is unclear, put the detail in logs or PostgreSQL.
- Every scraped service should have an `up` alert and every critical pipeline should expose freshness, failure, and recovery signals.
- Alert windows must account for the current 20-second scrape and evaluation interval.

### Logging and Alloy standards

- Services should emit one structured JSON object per line to stdout/stderr. Containers remain responsible for local log rotation; Alloy is the transport and processing layer.
- Every application event should include a stable level, service/component identity, event name or message, and relevant `process_id`. Errors should use a controlled `error_code` plus a human-readable message.
- Secrets, bearer tokens, private keys, credentials, signed payloads, and unnecessary personal data must never be logged.
- Loki labels must remain low-cardinality. `service_name`, environment, bounded level, and bounded component are suitable. `process_id`, strategy names, and `error_code` require an explicit cardinality bound before becoming indexed labels; otherwise retain them as structured fields and filter them at query time.
- Alloy configuration must declare exactly which containers it collects. New services are not considered covered until discovery/relabel rules and a Loki query verify ingestion.
- Alloy must preserve its positions/state volume across restarts to reduce duplicates and gaps.
- Collection failures must not affect trading liveness. Alloy and Loki are observational dependencies, not trading authorization dependencies.
- Dropping old container history is an ingestion policy, not Loki retention. Changes to either policy must be documented separately.

### Loki standards

- Use LogQL first to narrow by stable stream labels, then parse and filter structured fields.
- Keep high-cardinality and event-specific fields in the log body rather than the index.
- Use bounded time ranges and result limits for dashboards and investigations. The Grafana data source currently caps results at 1,000 lines.
- Retention must be intentional. The current Loki declaration retains 720 hours (30 days); Alloy's current development pipeline independently ignores discovered history older than one hour.
- Loki must not be treated as an audit ledger or the only record of financial activity.
- Alert-state history sharing the Loki instance must be considered when changing Loki availability, retention, or storage.

### PostgreSQL/TimescaleDB monitoring standards

- PostgreSQL is for durable system and business truth, not high-frequency host telemetry.
- Grafana queries must use indexed predicates, bounded time ranges, explicit row limits, and `process_id` scoping where process behavior is involved.
- Do not repeatedly scan large order, market-data, or event tables from dashboard refreshes. Use an existing bounded projection, continuous aggregate, or a purpose-built summary introduced through a migration when necessary.
- Keep dashboard connection pools within the database budget. Production currently allows Grafana two open and one idle PgBouncer connections.
- Monitoring queries must never mutate business data. Non-trading-process database changes require migrations.
- Panels should distinguish current application state from historically reconstructed state and state their freshness expectation.

### Grafana dashboards and alerts

- Version-control dashboards, data-source provisioning, and alert rules. Treat UI edits as temporary exploration unless they are exported and reviewed into provisioning.
- Organize dashboards around operator decisions: infrastructure health, data-pipeline health, trading-path readiness, execution/reconciliation integrity, and business outcomes.
- A panel title, unit, legend, thresholds, data source, scope, and expected freshness must be unambiguous.
- Prefer Prometheus for frequently refreshed health panels, Loki for drill-down context, and PostgreSQL for durable outcomes.
- Alerts must be actionable and identify the affected service, component, environment, and process where applicable. They must distinguish no data, query failure, and an unhealthy measured value.
- `noDataState: OK` is acceptable only where absence genuinely proves safety or normality. Critical readiness signals should generally alert on missing data, as the directional runtime rules currently do.
- Use a `for` duration to suppress transient noise where the condition is not immediately dangerous. Do not turn telemetry alerts into durable trading kill switches.
- Every alert should have an owner-facing summary and a description that says what evidence failed and over what interval.

## Strengths

1. **Appropriate separation of concerns.** Metrics, logs, and durable business data have specialized stores while Grafana provides a common operating surface.
2. **Reproducible configuration.** Dashboards, data sources, alerts, and core back-end configuration are provisioned from files. Most monitoring images are pinned by version and digest.
3. **Resource-conscious database access.** Grafana uses PgBouncer and deliberately small connection pools, protecting a database shared with latency-sensitive trading workloads.
4. **Useful trading-path telemetry.** Existing Prometheus metrics and alert rules cover evidence freshness, oracle readiness, disconnect causes, recovery duration, and flapping rather than only CPU and container uptime.
5. **Bounded storage in a small deployment.** Prometheus has both time and size retention limits, and Loki has explicit retention on persistent local volumes.
6. **Loopback-only operator exposure.** Grafana, Prometheus, PostgreSQL, and PgBouncer host ports are bound to `127.0.0.1` where published, reducing unintended network exposure.

## Weaknesses and risks

1. **The currently running topology is configuration-drifted.** Containers in one Compose project were launched from multiple worktrees and revisions. Recreating the project from the main checkout would omit Alloy and could change Grafana and Prometheus behavior.
2. **Log coverage is extremely narrow.** The running Alloy pipeline collects only the canonical ingester runtimes. Loki therefore cannot yet serve as a complete explanation layer for the bot, database proxy, or monitoring services.
3. **Single-host, single-replica storage.** Prometheus, Loki, Grafana, and PostgreSQL use local Docker volumes. A host or volume failure can remove monitoring history, and Loki explicitly has replication factor 1.
4. **The monitoring plane has shared failure domains.** Grafana alert evaluation depends on Grafana plus its queried data source; alert history also depends on the same Loki used for application logs. There is no independently declared external notification or dead-man path in the inspected configuration.
5. **Loki is unauthenticated internally.** `auth_enabled: false` is reasonable on an isolated Compose network, but compromise of any attached container permits direct Loki access. Docker-socket access also makes Alloy a sensitive component even though the mount is read-only.
6. **Label-cardinality risk in the new Alloy configuration.** `strategy` and `error_code` are promoted to indexed labels. Their vocabularies are not enforced in the collector configuration; uncontrolled values could increase Loki index cost.
7. **Mixed alert data paths.** Some alerts query Prometheus, while older CLOB alerts poll an authenticated JSON API through the Infinity plugin. This creates different semantics for history, no-data behavior, and failure diagnosis.
8. **Limited observability of the observability stack.** Prometheus self-scrapes and has one connectivity alert, but the inspected deployment does not establish comprehensive alerts for scrape coverage, Alloy delivery errors, Loki ingestion/query health, disk usage, PostgreSQL monitoring-query pressure, or Grafana notification delivery.
9. **Short local retention limits investigations.** Seven days/2 GB in the observed development Prometheus and 15 days/5 GB in production may be insufficient for low-frequency trading regressions or release-to-release comparisons. PostgreSQL should remain the source for long-term business analysis, but operational trend retention should be chosen deliberately.

## Source of truth and verification boundary

The canonical deployment definitions are the repository's Compose and provisioning files. `docker compose ps`, container labels, and mounted paths describe the observed runtime but do not make an unmerged worktree canonical. When documentation and runtime differ, record both, reconcile the deployment through the repository's integration policy, and avoid silently treating a development container as an accepted production component.
