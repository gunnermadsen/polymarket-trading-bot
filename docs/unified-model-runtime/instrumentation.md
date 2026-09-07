# UMR instrumentation contract

## Stable ownership and semantics

`process_id` is the canonical scope. Model/run/config identities are attached to durable prediction evidence and the model information metric. Metric names, units, denominators and bounded reason meanings remain stable across model generations. New adapters automatically participate through the common process/scoring wrapper.

`polymarket_umr_*` metrics are exposed by the existing bot `/prometheus/metrics` endpoint. Scrapes read bounded memory only. The dedicated provisioned **Unified Model Runtime** dashboard uses Prometheus and Loki; it performs no PostgreSQL queries. Existing dashboards are retained.

RTDS candle coverage/readiness metrics and existing hydration logs are supplied through the [shared repository contract](rtds-repository.md). Their names and definitions remain unchanged; this extraction introduces no new monitoring datasource or dashboard queries.

## Measurements

| Family | Meaning |
| --- | --- |
| `model_info` | Immutable selected model, feature schema, process configuration and execution mode |
| `enabled`, `runtime_ready` | Active instrumentation registration and latest action readiness |
| `last_observation_timestamp_seconds`, `last_success_timestamp_seconds` | Runtime activity and last successful inference clocks |
| `observations_total`, `markets_total` | Repeated callbacks and distinct observed/inferred/admitted markets |
| `opportunities_total`, `skipped_total` | Claimed scheduled opportunities and skip reasons; not interchangeable denominators |
| `feature_builds_total`, `feature_failures_total`, `missing_feature_fraction`, `feature_age_seconds` | Feature availability and quality |
| `stage_duration_seconds` | Histogram for shared feature, inference, strategy and observation durations |
| `inferences_total`, `inference_failures_total`, `predictions_total`, `actions_total`, `probability_bin_total`, `confidence_bin_total` | Inference success/failure, direction and probability distribution |
| `model_admission_total`, `admission_reasons_total`, `strategy_rejections_total`, `readiness_blocks_total` | Model admission separated from strategy and runtime gates |
| `decisions_total`, `execution_total`, `fills_total` | Persisted decision stages and execution/fill events |
| `prediction_outcomes_total`, `brier_sum`, `brier_count`, `calibration_*` | Officially resolved, evaluation-weighted prediction quality |
| `trade_outcomes_total`, `realized_pnl_usd`, `gross_profit_usd`, `gross_loss_usd`, `fees_usd`, `max_drawdown_usd` | Recognized settlement economics for the instrumentation session |
| `fill_notional_usd`, `filled_shares`, `entry_seconds_sum`, `entry_fill_count`, `slippage_notional_usd` | Actual fill cost, entry timing and quote-to-fill slippage |
| `telemetry_dropped_total`, `registry_dropped_total` | Bounded telemetry capacity losses |

The runtime also exposes last probability/confidence and optional learned admission and consensus outputs. Undefined ratios remain undefined until evidence exists; do not turn absent outcomes into zero Brier or zero loss recovery.

Counters reset on service restart. Per-process aggregates reset when model/run/config identity changes. Session gauges are explicitly labelled as session measurements in the dashboard and must not be presented as lifetime accounting. Durable accounting and prediction records remain authoritative for historical analysis.

Calibration is evaluation-weighted: every retained inferred opportunity, including rejected admissions, is scored against the official market outcome. It is not independent-market statistical evidence. These live opportunity populations differ from historical tournament populations; do not directly label a session Brier change as model degradation without matching the evaluation cohort. Distinct-market coverage is separate. Eligible markets have at least one claimed scheduled opportunity; observed markets include pre-claim input outages. Both coverage denominators are displayed. Pending prediction capacity is bounded; unresolved, dropped and post-restart gaps must remain visible.

Funnel stages are transitions, not a partition to sum. Market coverage is distinct inferred/admitted markets divided by distinct observed markets. Opportunity inference coverage uses successful inferences divided by claimed scheduled opportunities and is displayed alongside skips so unavailable pre-claim inputs are visible.

## Durable evidence

The existing strategy decision metadata includes a versioned `model_evaluation` envelope carrying process/run/config/model identity, probability, confidence, admission result, feature timestamp, input hash and inference timing. Existing feature snapshots retain the feature vector and lineage; a hash alone is not replay data. Existing order-plan/fill/settlement references connect decisions to economics. No duplicate ledger is introduced.

Historical analytical exports remain Parquet on the external SSD using the existing infrastructure. Grafana must not run broad scans of these durable decision tables. If a future visual needs PostgreSQL, scope by process and time, inspect the plan, run bounded `EXPLAIN ANALYZE`, record planning/execution/buffer evidence and verify index pruning before provisioning it. Any required schema/index migration needs explicit authorization.

## Logs and monitoring delivery

Structured UMR lifecycle events flow through the existing bot JSON logs, Alloy and Loki. Process/run/model/decision/market identifiers are fields, not Loki stream labels. Routine predictions belong in durable records and metrics; avoid verbose success logging on every tick. Readiness transitions and errors provide investigation context.

Telemetry is bounded and cannot change model admission or durable enabled intent. Prometheus and Grafana failures do not stop orders. Durable decision persistence retains its existing order-safety role. Capacity drops and unresolved predictions are measurable rather than silently converted into successful observations.

## Dashboard and alerts

The new dashboard separates overview, runtime health, opportunity funnel, inputs/latency, prediction/admission, calibration, execution/economics and investigation. Probability/confidence decile tables, p50/p95/p99 latency, calibration mean/outcome/count tables, and outcome sample counts accompany the summary views. Model selection is dynamic; no champion-specific SQL or hard-coded process IDs are allowed. Units, denominator descriptions and session scope are visible.

Provision alerts for missing runtime activity on enabled registrations, scheduled opportunities without inference, persistent feature/inference failures and telemetry drops. Performance/quality alerts require a declared minimum sample count. Alerts notify; they do not persistently disable trading. Monitor bot scrape availability separately so process-series disappearance does not masquerade as healthy inactivity.

All Grafana, Prometheus, Loki and Alloy changes are provisioned through existing configuration paths and receive the repository's observability deployment provenance record when deployed.
