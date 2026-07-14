#!/usr/bin/env bash

# SELECT-only BTC realtime-paper experiment Compose job. It talks to TimescaleDB and the bot over the
# Compose network, never controls another container, and never archives the admin token.

set -Eeuo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${BTC_PAPER_EXPERIMENT_MANIFEST_PATH:-$ROOT_DIR/docs/experiments/btc-5m-chainlink-fair-value-20260713-c/preregistration.json}"
READINESS_SQL="$ROOT_DIR/docs/sql/btc-5m-paper-experiment-readiness.sql"
REPORT_SQL="$ROOT_DIR/docs/sql/btc-5m-paper-experiment-report.sql"
MODE="${1:-}"
REQUESTED_EXPERIMENT_ID="${2:-}"
DB_HOST="${POSTGRES_HOST:-timescaledb-0}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_USER="${POLYMARKET_DB_USER:-postgres}"
DB_NAME="${POLYMARKET_DB_NAME:-polymarket}"
HTTP_BASE_URL="${POLYMARKET_HTTP_BASE_URL:-http://polymarket-bot:8097}"
PARTIAL_DIR=""
FINAL_DIR=""
export PGPASSWORD="${PGPASSWORD:-${POSTGRES_PASSWORD:-}}"

usage() {
  echo "usage: $0 preflight|daily|final [experiment-uuid]" >&2
}

for command in jq psql pg_isready curl df awk sed grep date find sort cp mv tee tail; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "required command is unavailable: $command" >&2
    exit 2
  }
done

psql_audit() {
  psql -X -v ON_ERROR_STOP=1 -h "$DB_HOST" -p "$DB_PORT" \
    -U "$DB_USER" -d "$DB_NAME" "$@"
}

case "$MODE" in
  preflight|daily|final) ;;
  *) usage; exit 2 ;;
esac

[[ -r "$MANIFEST" && -r "$READINESS_SQL" && -r "$REPORT_SQL" ]] || {
  echo "BTC paper experiment manifest or SQL input is missing" >&2
  exit 2
}
jq -e '.schema_version == "btc_realtime_paper_preregistration_v1" and .status == "preregistered"' \
  "$MANIFEST" >/dev/null

jq -e '
  def keys_exact($expected): (keys | sort) == ($expected | sort);
  def duration_ms($milliseconds): {
    secs: (($milliseconds / 1000) | floor),
    nanos: (($milliseconds % 1000) * 1000000)
  };
  . as $manifest
  | .contracts.process_definition_without_preregistration_sha256 as $definition
  | .contracts.experiment_config_without_preregistration_sha256 as $experiment
  | $definition.raw.btc_realtime_paper as $control
  | $experiment.raw as $frozen
  | ($manifest.contracts | keys_exact([
      "experiment_config_without_preregistration_sha256",
      "process_definition_without_preregistration_sha256"
    ]))
  and ($definition | keys_exact(["execution", "raw"]))
  and ($definition.raw | keys_exact(["btc_realtime_paper"]))
  and ($control | keys_exact([
      "ml_shadow", "next_experiment_key", "paper", "runtime", "schema_version", "strategy"
    ]))
  and ($experiment | keys_exact(["execution", "raw"]))
  and ($frozen | keys_exact([
      "build", "ml_shadow", "paper", "pipeline_version", "process_schema_version", "runtime", "strategy"
    ]))
  and (($control | has("preregistration_sha256")) | not)
  and (($frozen | has("preregistration_sha256")) | not)
  and ($definition.execution == {
      mode: "paper", execute_signals: true, live_capital: false, taker_fee_rate: null
    })
  and ($experiment.execution == $definition.execution)
  and ($control.schema_version == "btc_realtime_paper_process_v1")
  and ($control.next_experiment_key == $manifest.experiment_key)
  and ($control.strategy | keys_exact([
      "base_probability_uncertainty", "basis_lead_weight", "basis_uncertainty_weight",
      "feature_schema_version", "feed_age_uncertainty_per_second", "latency_reserve_per_share",
      "max_book_age_ms", "max_chainlink_open_delay_ms", "max_depth_participation",
      "max_entry_price", "max_fee_age_ms", "max_fee_rate", "max_lead_sigma_fraction",
      "max_probability_uncertainty", "max_reference_age_ms", "max_source_skew_ms",
      "min_entry_price", "min_net_edge_per_share", "min_net_edge_usd",
      "min_seconds_after_open", "min_seconds_before_close", "momentum_1s_weight",
      "momentum_30s_weight", "momentum_5s_weight", "probability_floor",
      "slippage_reserve_bps", "spread_reserve_fraction", "strategy_version", "target_size",
      "volatility_floor_per_sqrt_second"
    ]))
  and ($control.strategy == $frozen.strategy)
  and ($control.runtime | keys_exact([
      "official_resolution_audit_grace_secs",
      "official_resolution_watch_retention_secs",
      "strategy_interval_ms"
    ]))
  and ($control.paper | keys_exact([
      "arrival_latency_ms", "starting_collateral_usd", "stress_previews", "visible_depth_haircut"
    ]))
  and ($control.ml_shadow == {enabled: true})
  and ($frozen.build | keys_exact(["compiled_source_identity", "package_version"]))
  and ($frozen.runtime | keys_exact([
      "binance_heartbeat_interval", "binance_ws_url", "boundary_tick_max_delay",
      "checkpoint_interval", "clob_heartbeat_interval", "clob_rest_base_url", "clob_ws_url",
      "discovery_interval", "enabled", "gamma_base_url", "max_book_age", "max_reference_age",
      "official_resolution_audit_grace", "official_resolution_watch_retention",
      "reconnect_initial_delay", "reconnect_max_delay", "rtds_heartbeat_interval",
      "rtds_ws_url", "strategy_interval", "writer_capacity"
    ]))
  and ($frozen.paper | keys_exact(["execution_enabled", "stress_previews", "venue"]))
  and ($frozen.paper.venue | keys_exact([
      "arrival_latency", "max_book_age", "starting_collateral_usd", "visible_depth_haircut"
    ]))
  and ($frozen.ml_shadow | keys_exact(["execution_authority", "ml_a_enabled", "ml_b_enabled"]))
  and ($frozen.pipeline_version == $manifest.frozen_runtime_contract.pipeline_version)
  and ($frozen.process_schema_version == $control.schema_version)
  and ($frozen.build.compiled_source_identity
       == $manifest.frozen_runtime_contract.compiled_source_identity)
  and ($frozen.strategy.strategy_version == $manifest.frozen_runtime_contract.strategy_version)
  and ($frozen.strategy.feature_schema_version
       == $manifest.frozen_runtime_contract.feature_schema_version)
  and ($frozen.strategy.target_size == $manifest.frozen_runtime_contract.target_size)
  and ($control.runtime.strategy_interval_ms
       == $manifest.frozen_runtime_contract.strategy_interval_ms)
  and ($control.runtime.official_resolution_audit_grace_secs
       == $manifest.cohort.official_resolution_grace_seconds)
  and ($control.runtime.official_resolution_watch_retention_secs
       == $manifest.cohort.resolution_watch_retention_seconds)
  and ($frozen.runtime.strategy_interval == duration_ms($control.runtime.strategy_interval_ms))
  and ($frozen.runtime.max_book_age == duration_ms($frozen.strategy.max_book_age_ms))
  and ($frozen.runtime.max_reference_age == duration_ms($frozen.strategy.max_reference_age_ms))
  and ($frozen.runtime.boundary_tick_max_delay
       == duration_ms($frozen.strategy.max_chainlink_open_delay_ms))
  and ($frozen.runtime.official_resolution_audit_grace
       == duration_ms($control.runtime.official_resolution_audit_grace_secs * 1000))
  and ($frozen.runtime.official_resolution_watch_retention
       == duration_ms($control.runtime.official_resolution_watch_retention_secs * 1000))
  and ($control.paper.arrival_latency_ms
       == $manifest.frozen_runtime_contract.paper_arrival_latency_ms)
  and ($control.paper.visible_depth_haircut
       == $manifest.frozen_runtime_contract.paper_visible_depth_haircut)
  and ($control.paper.starting_collateral_usd
       == $manifest.frozen_runtime_contract.paper_starting_collateral_usd)
  and ($frozen.paper.execution_enabled
       == $manifest.frozen_runtime_contract.paper_execution_enabled)
  and ($frozen.paper.venue.arrival_latency == duration_ms($control.paper.arrival_latency_ms))
  and ($frozen.paper.venue.visible_depth_haircut == $control.paper.visible_depth_haircut)
  and ($frozen.paper.venue.starting_collateral_usd == $control.paper.starting_collateral_usd)
  and ($frozen.paper.venue.max_book_age == [
      (($frozen.strategy.max_book_age_ms / 1000) | floor),
      (($frozen.strategy.max_book_age_ms % 1000) * 1000000)
    ])
  and ($control.paper.stress_previews == $manifest.online_counterfactual_previews.scenarios)
  and (($control.paper.stress_previews | map({
      scenario_key: .scenario_key,
      arrival_latency: duration_ms(.arrival_latency_ms),
      visible_depth_haircut: .visible_depth_haircut
    })) == $frozen.paper.stress_previews)
  and ($frozen.ml_shadow == {
      ml_a_enabled: true, ml_b_enabled: true, execution_authority: false
    })
  and ($control.ml_shadow.enabled == $manifest.frozen_runtime_contract.ml_shadow_enabled)
  and ($frozen.ml_shadow.execution_authority
       == $manifest.frozen_runtime_contract.ml_execution_authority)
  and ($manifest.authority.ml_may_influence_orders == false)
  and ($manifest.authority.authorizes_live_capital == false)
  and ($manifest.artifact_integrity.sha256 | keys_exact([
      "audit_wrapper", "paper_capital_migration", "paper_experiment_report_sql", "readiness_sql",
      "unstarted_trading_process_migration"
    ]))
  and ($manifest.artifact_integrity.sha256.paper_experiment_report_sql.path
       == $manifest.authority.authoritative_report)
  and ($manifest.artifact_integrity.sha256.readiness_sql.path
       == $manifest.authority.operational_integrity_audit)
  and ($manifest.artifact_integrity.sha256.audit_wrapper.path
       == $manifest.authority.archival_wrapper)
  and ($manifest.authority.authoritative_report
       == "docs/sql/btc-5m-paper-experiment-report.sql")
  and ($manifest.authority.operational_integrity_audit
       == "docs/sql/btc-5m-paper-experiment-readiness.sql")
  and ($manifest.authority.archival_wrapper == "scripts/btc-paper-experiment-audit.sh")
  and ($manifest.artifact_integrity.sha256.paper_capital_migration.path
       == "packages/db-migrate/src/migrations/1777123000000-AddBtcPhase6PaperCapital.ts")
  and ($manifest.artifact_integrity.sha256.unstarted_trading_process_migration.path
       == "packages/db-migrate/src/migrations/1777124000000-AllowUnstartedTradingProcesses.ts")
' "$MANIFEST" >/dev/null || {
  echo "preregistration contract coherence check failed" >&2
  exit 2
}

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    LC_ALL=C shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    echo "shasum or sha256sum is required" >&2
    return 2
  fi
}

sha256_stdin() {
  if command -v shasum >/dev/null 2>&1; then
    LC_ALL=C shasum -a 256 | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
  else
    echo "shasum or sha256sum is required" >&2
    return 2
  fi
}

PREREGISTRATION_SHA256="$(sha256_file "$MANIFEST")"

jq -e '
  .artifact_integrity.algorithm == "sha256"
  and (.artifact_integrity.sha256 | type == "object" and length == 5)
  and ([.artifact_integrity.sha256[]
    | (.path | type == "string" and length > 0)
      and (.sha256 | type == "string" and test("^[0-9a-f]{64}$"))]
    | all)
' "$MANIFEST" >/dev/null || {
  echo "invalid or incomplete frozen artifact set in preregistration" >&2
  exit 2
}
ARTIFACT_ROWS="$(jq -er '.artifact_integrity.sha256
  | to_entries[] | [.value.path,.value.sha256] | @tsv' "$MANIFEST")" || {
  echo "unable to read frozen artifact set from preregistration" >&2
  exit 2
}
[[ -n "$ARTIFACT_ROWS" ]] || {
  echo "empty frozen artifact set in preregistration" >&2
  exit 2
}
while IFS=$'\t' read -r artifact_path expected_sha256; do
  [[ -n "$artifact_path" && -n "$expected_sha256" ]] || {
    echo "invalid frozen artifact entry in preregistration" >&2
    exit 2
  }
  [[ "$artifact_path" != /* && "$artifact_path" != *".."* ]] || {
    echo "unsafe frozen artifact path: $artifact_path" >&2
    exit 2
  }
  ACTUAL_SHA256="$(sha256_file "$ROOT_DIR/$artifact_path")"
  [[ "$ACTUAL_SHA256" == "$expected_sha256" ]] || {
    echo "frozen artifact hash mismatch: $artifact_path" >&2
    exit 23
  }
done <<< "$ARTIFACT_ROWS"
unset ARTIFACT_ROWS

EXPERIMENT_KEY="$(jq -er '.experiment_key' "$MANIFEST")"
PROCESS_KEY="$(jq -er '.cohort.process_key | select(type == "string" and length > 0)' "$MANIFEST")"
EXPECTED_STRATEGY_INTERVAL_MS="$(jq -er '.frozen_runtime_contract.strategy_interval_ms' "$MANIFEST")"
EXPECTED_TARGET_SIZE="$(jq -er '.frozen_runtime_contract.target_size' "$MANIFEST")"
EXPECTED_PAPER_LATENCY_MS="$(jq -er '.frozen_runtime_contract.paper_arrival_latency_ms' "$MANIFEST")"
EXPECTED_DEPTH_HAIRCUT="$(jq -er '.frozen_runtime_contract.paper_visible_depth_haircut' "$MANIFEST")"
EXPECTED_STARTING_COLLATERAL="$(jq -er '.frozen_runtime_contract.paper_starting_collateral_usd' "$MANIFEST")"
EXPECTED_RESOLUTION_GRACE="$(jq -er '.cohort.official_resolution_grace_seconds' "$MANIFEST")"
EXPECTED_RESOLUTION_RETENTION="$(jq -er '.cohort.resolution_watch_retention_seconds' "$MANIFEST")"
EXPECTED_PROCESS_DEFINITION_CONTRACT="$(jq -cer \
  '.contracts.process_definition_without_preregistration_sha256' "$MANIFEST")"
EXPECTED_EXPERIMENT_CONFIG_CONTRACT="$(jq -cer \
  '.contracts.experiment_config_without_preregistration_sha256' "$MANIFEST")"
EXPECTED_RTDS_WS_URL="$(jq -er \
  '.contracts.experiment_config_without_preregistration_sha256.raw.runtime.rtds_ws_url' \
  "$MANIFEST")"
EXPECTED_BINANCE_WS_URL="$(jq -er \
  '.contracts.experiment_config_without_preregistration_sha256.raw.runtime.binance_ws_url' \
  "$MANIFEST")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE_ROOT="$ROOT_DIR/docs/evidence/$EXPERIMENT_KEY/audits"
FINAL_DIR="$EVIDENCE_ROOT/${STAMP}-${MODE}"
PARTIAL_DIR="$FINAL_DIR.partial.$$"
[[ ! -e "$FINAL_DIR" && ! -e "$PARTIAL_DIR" ]] || {
  echo "refusing to overwrite an existing evidence directory: $FINAL_DIR" >&2
  exit 3
}
mkdir -p "$PARTIAL_DIR"

finalize() {
  local exit_code=$?
  trap - EXIT
  if [[ -n "$PARTIAL_DIR" && -d "$PARTIAL_DIR" ]]; then
    jq -n \
      --arg mode "$MODE" \
      --arg completed_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --argjson exit_code "$exit_code" \
      '{mode:$mode,completed_at_utc:$completed_at,exit_code:$exit_code}' \
      > "$PARTIAL_DIR/audit-status.json"
    (
      cd "$PARTIAL_DIR"
      find . -type f ! -name SHA256SUMS -print | LC_ALL=C sort | while IFS= read -r file; do
        if command -v shasum >/dev/null 2>&1; then
          LC_ALL=C shasum -a 256 "$file"
        else
          sha256sum "$file"
        fi
      done
    ) > "$PARTIAL_DIR/SHA256SUMS"
    mv "$PARTIAL_DIR" "$FINAL_DIR"
    echo "BTC paper experiment evidence archived at $FINAL_DIR"
  fi
  exit "$exit_code"
}
trap finalize EXIT

cp "$MANIFEST" "$PARTIAL_DIR/preregistration.json"
{
  echo "audit_mode=$MODE"
  echo "experiment_key=$EXPERIMENT_KEY"
  echo "process_key=$PROCESS_KEY"
  echo "preregistration_sha256=$PREREGISTRATION_SHA256"
  echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "artifact_root=$ROOT_DIR"
} > "$PARTIAL_DIR/audit-metadata.txt"

df -Pk "$EVIDENCE_ROOT" > "$PARTIAL_DIR/evidence-storage.txt"
FREE_DISK_KIB="$(df -Pk "$EVIDENCE_ROOT" | awk 'NR==2 {print $4}')"
LAUNCH_FLOOR_GIB="$(jq -er '.operational_gates.minimum_host_free_disk_before_launch_gib' "$MANIFEST")"
HALT_FLOOR_GIB="$(jq -er '.operational_gates.halt_when_host_free_disk_below_gib' "$MANIFEST")"
if (( FREE_DISK_KIB < HALT_FLOOR_GIB * 1024 * 1024 )); then
  echo "HALT REQUIRED: evidence storage is below ${HALT_FLOOR_GIB} GiB free" \
    | tee "$PARTIAL_DIR/HALT-REQUIRED.txt" >&2
  exit 20
fi
if [[ "$MODE" == "preflight" ]] && (( FREE_DISK_KIB < LAUNCH_FLOOR_GIB * 1024 * 1024 )); then
  echo "LAUNCH BLOCKED: evidence storage is below ${LAUNCH_FLOOR_GIB} GiB free" \
    | tee "$PARTIAL_DIR/LAUNCH-BLOCKED.txt" >&2
  exit 21
fi

CONFIG_GATE_PASS=false
if [[ "${POLYMARKET_LIVE_ORDER_SUBMIT_ENABLED:-}" == "false" &&
      "${POLYMARKET_LIVE_USER_WS_ENABLED:-}" == "false" &&
      "${POLYMARKET_SCAN_ENABLED:-}" == "false" &&
      "${POLYMARKET_SIGNAL2_ENABLED:-}" == "false" &&
      "${POLYMARKET_SIGNAL3_ENABLED:-}" == "false" &&
      "${POLYMARKET_COPY_TRADE_ENABLED:-}" == "false" &&
      "${POLYMARKET_COPY_EXECUTE_ENABLED:-}" == "false" &&
      "${POLYMARKET_WHALE_LIVE_ENABLED:-}" == "false" &&
      "${POLYMARKET_WHALE_BACKFILL_ENABLED:-}" == "false" &&
      "${POLYMARKET_BTC_REALTIME_ENABLED:-}" == "true" &&
      "${POLYMARKET_BTC_PAPER_ENABLED:-}" == "true" &&
      "${POLYMARKET_BTC_ML_SHADOW_ENABLED:-}" == "true" ]]; then
  CONFIG_GATE_PASS=true
fi
CREDENTIAL_GATE_PASS=false
if [[ -z "${POLYMARKET_CLOB_API_KEY:-}" && -z "${POLYMARKET_CLOB_SECRET:-}" &&
      -z "${POLYMARKET_CLOB_PASSPHRASE:-}" && -z "${POLYMARKET_PRIVATE_KEY:-}" &&
      -z "${POLYMARKET_FUNDER_ADDRESS:-}" && -z "${POLYMARKET_SIGNATURE_TYPE:-}" ]]; then
  CREDENTIAL_GATE_PASS=true
fi
ENDPOINT_GATE_PASS=false
if [[ "${POLYMARKET_BTC_RTDS_WS_URL:-}" == "$EXPECTED_RTDS_WS_URL" &&
      "${POLYMARKET_BTC_BINANCE_WS_URL:-}" == "$EXPECTED_BINANCE_WS_URL" ]]; then
  ENDPOINT_GATE_PASS=true
fi
jq -n --argjson exact_safe_config "$CONFIG_GATE_PASS" \
  --argjson all_live_credentials_blank "$CREDENTIAL_GATE_PASS" \
  --argjson exact_source_endpoints "$ENDPOINT_GATE_PASS" \
  --arg expected_rtds_ws_url "$EXPECTED_RTDS_WS_URL" \
  --arg expected_binance_ws_url "$EXPECTED_BINANCE_WS_URL" \
  '{exact_safe_config:$exact_safe_config,
    all_live_credentials_blank:$all_live_credentials_blank,
    exact_source_endpoints:$exact_source_endpoints,
    expected_source_endpoints:{rtds_ws_url:$expected_rtds_ws_url,
      binance_ws_url:$expected_binance_ws_url}}' \
  > "$PARTIAL_DIR/safety-gates.json"
if [[ "$MODE" == "preflight" &&
      ( "$CONFIG_GATE_PASS" != "true" || "$CREDENTIAL_GATE_PASS" != "true" ||
        "$ENDPOINT_GATE_PASS" != "true" ) ]]; then
  echo "LAUNCH BLOCKED: safety, blank-credential, or source-endpoint gate failed" >&2
  exit 24
fi

pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
  > "$PARTIAL_DIR/postgres-pg-isready.txt"
psql_audit \
  -P pager=off -c "select name, setting from pg_settings where name in
    ('shared_buffers','effective_cache_size','maintenance_work_mem','work_mem',
     'max_connections','max_parallel_workers','max_parallel_workers_per_gather',
     'max_worker_processes','timescaledb.max_background_workers') order by name" \
  > "$PARTIAL_DIR/postgres-memory-settings.txt"

SCHEMA_GATE_PASS="$(psql_audit -At -c "
  WITH expected_constraints(name) AS (VALUES
    ('uq_btc_interval_market_official_identity'),
    ('btc_paper_settlement_ledger_pkey'),
    ('btc_paper_settlement_ledger_experiment_id_fkey'),
    ('btc_paper_settlement_ledger_process_id_fkey'),
    ('btc_paper_settlement_ledger_order_id_fkey'),
    ('btc_paper_settlement_ledger_market_id_fkey'),
    ('uq_btc_paper_settlement_experiment_order'),
    ('fk_btc_paper_settlement_official_identity'),
    ('chk_btc_paper_settlement_outcome'),('chk_btc_paper_settlement_fill_ids'),
    ('chk_btc_paper_settlement_source'),('chk_btc_paper_settlement_amounts'),
    ('chk_btc_paper_settlement_credit_status'),('chk_btc_paper_settlement_credit_state'),
    ('chk_btc_paper_settlement_evidence')
  ), expected_indexes(name) AS (VALUES
    ('idx_btc_paper_settlement_pending'),('idx_btc_paper_settlement_experiment_credited'),
    ('idx_btc_features_process_window_asof'),('idx_btc_decisions_experiment_process_at'),
    ('idx_book_checkpoints_token_received_source')
  )
  SELECT
    (SELECT count(*) FROM public.migrations WHERE timestamp=1777123000000
      AND name='AddBtcPhase6PaperCapital1777123000000')=1
    AND (SELECT count(*) FROM public.migrations WHERE timestamp=1777124000000
      AND name='AllowUnstartedTradingProcesses1777124000000')=1
    AND (SELECT is_nullable='YES'
      FROM information_schema.columns
      WHERE table_schema='polymarket' AND table_name='trading_processes'
        AND column_name='started_at')
    AND to_regclass('polymarket.btc_paper_settlement_ledger') IS NOT NULL
    AND (SELECT count(*) FROM expected_constraints e JOIN pg_constraint c
      ON c.conname=e.name)=15
    AND (SELECT count(*) FROM expected_indexes e JOIN pg_indexes i
      ON i.schemaname='polymarket' AND i.indexname=e.name)=5
    AND (SELECT count(*) FROM timescaledb_information.jobs
      WHERE hypertable_schema='polymarket'
        AND hypertable_name IN('reference_price_ticks','btc_feature_snapshots',
          'ml_feature_vectors','ml_shadow_predictions')
        AND proc_name='policy_compression' AND scheduled
        AND config->>'compress_after'='1 day')=4;")"
printf 'schema_gate_pass=%s\n' "$SCHEMA_GATE_PASS" > "$PARTIAL_DIR/paper-capital-schema-gate.txt"

if [[ "$MODE" == "preflight" ]]; then
  if [[ "$SCHEMA_GATE_PASS" != "t" ]]; then
    echo "LAUNCH BLOCKED: TimescaleDB migration, schema, index, or policy gate failed" >&2
    exit 25
  fi
  EXISTING_EXPERIMENTS="$(psql_audit -At -v key="$EXPERIMENT_KEY" <<'SQL'
select count(*)
from polymarket.btc_paper_experiments
where name = :'key';
SQL
)"
  if [[ "$EXISTING_EXPERIMENTS" != "0" ]]; then
    echo "LAUNCH BLOCKED: immutable experiment key already exists" >&2
    exit 27
  fi
  read -r ACTIVE_BTC_PROCESSES ACTIVE_BTC_EXPERIMENTS < <(
    psql_audit -At -F ' ' <<'SQL'
select
  (select count(*)
   from polymarket.trading_processes
   where process_type = 'btc_5m'
     and process_scope = 'realtime_paper'
     and (enabled or status in ('starting','running','stopping'))),
  (select count(*)
   from polymarket.btc_paper_experiments
   where status = 'running');
SQL
  )
  jq -n \
    --argjson active_btc_processes "$ACTIVE_BTC_PROCESSES" \
    --argjson active_btc_experiments "$ACTIVE_BTC_EXPERIMENTS" \
    '{active_btc_realtime_paper_processes:$active_btc_processes,
      active_btc_paper_experiments:$active_btc_experiments,
      gate_pass:($active_btc_processes == 0 and $active_btc_experiments == 0)}' \
    > "$PARTIAL_DIR/preflight-lifecycle-gate.json"
  if (( ACTIVE_BTC_PROCESSES != 0 || ACTIVE_BTC_EXPERIMENTS != 0 )); then
    echo "LAUNCH BLOCKED: another BTC realtime-paper process or experiment is active" >&2
    exit 28
  fi
  PROCESS_DEFINITION_GATE_PASS="$(psql_audit -At \
    -v process_key="$PROCESS_KEY" \
    -v experiment_key="$EXPERIMENT_KEY" \
    -v preregistration_sha256="$PREREGISTRATION_SHA256" \
    -v strategy_interval_ms="$EXPECTED_STRATEGY_INTERVAL_MS" \
    -v target_size="$EXPECTED_TARGET_SIZE" \
    -v paper_latency_ms="$EXPECTED_PAPER_LATENCY_MS" \
    -v depth_haircut="$EXPECTED_DEPTH_HAIRCUT" \
    -v starting_collateral="$EXPECTED_STARTING_COLLATERAL" \
    -v resolution_grace="$EXPECTED_RESOLUTION_GRACE" \
    -v resolution_retention="$EXPECTED_RESOLUTION_RETENTION" \
    -v expected_process_definition_contract="$EXPECTED_PROCESS_DEFINITION_CONTRACT" <<'SQL'
select count(*) = 1 and bool_and(
  not enabled
  and status in ('created','stopped','failed','completed')
  and (config #- '{raw,btc_realtime_paper,preregistration_sha256}')
        = :'expected_process_definition_contract'::jsonb
  and config #>> '{execution,mode}' = 'paper'
  and (config #>> '{execution,execute_signals}')::boolean
  and not (config #>> '{execution,live_capital}')::boolean
  and not (config ? 'whale')
  and not (config ? 'copy_trade')
  and not (config ? 'expectancy_flow')
  and not (config ? 'exit_rules')
  and not (config ? 'mark_refresh')
  and config #>> '{raw,btc_realtime_paper,schema_version}' = 'btc_realtime_paper_process_v1'
  and config #>> '{raw,btc_realtime_paper,next_experiment_key}' = :'experiment_key'
  and config #>> '{raw,btc_realtime_paper,preregistration_sha256}' = :'preregistration_sha256'
  and (config #>> '{raw,btc_realtime_paper,strategy,target_size}')::numeric = :'target_size'::numeric
  and (config #>> '{raw,btc_realtime_paper,runtime,strategy_interval_ms}')::bigint = :'strategy_interval_ms'::bigint
  and (config #>> '{raw,btc_realtime_paper,runtime,official_resolution_audit_grace_secs}')::bigint = :'resolution_grace'::bigint
  and (config #>> '{raw,btc_realtime_paper,runtime,official_resolution_watch_retention_secs}')::bigint = :'resolution_retention'::bigint
  and (config #>> '{raw,btc_realtime_paper,paper,arrival_latency_ms}')::bigint = :'paper_latency_ms'::bigint
  and (config #>> '{raw,btc_realtime_paper,paper,visible_depth_haircut}')::numeric = :'depth_haircut'::numeric
  and (config #>> '{raw,btc_realtime_paper,paper,starting_collateral_usd}')::numeric = :'starting_collateral'::numeric
  and (config #>> '{raw,btc_realtime_paper,ml_shadow,enabled}')::boolean
)
from polymarket.trading_processes
where process_type = 'btc_5m'
  and process_scope = 'realtime_paper'
  and process_key = :'process_key';
SQL
)"
  printf 'process_definition_gate_pass=%s\n' "$PROCESS_DEFINITION_GATE_PASS" \
    > "$PARTIAL_DIR/process-definition-gate.txt"
  if [[ "$PROCESS_DEFINITION_GATE_PASS" != "t" ]]; then
    echo "LAUNCH BLOCKED: API-controlled BTC process definition does not match preregistration" >&2
    exit 29
  fi
  PROCESS_ID="$(psql_audit -At -v process_key="$PROCESS_KEY" <<'SQL'
select process_id
from polymarket.trading_processes
where process_type = 'btc_5m'
  and process_scope = 'realtime_paper'
  and process_key = :'process_key';
SQL
)"
  if [[ ! "$PROCESS_ID" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]]; then
    echo "LAUNCH BLOCKED: unable to resolve the unique BTC process definition UUID" >&2
    exit 29
  fi
  EXPECTED_EXPERIMENT_ID="$(psql_audit -At -v experiment_key="$EXPERIMENT_KEY" <<'SQL'
with hashed as (
  select digest(
    decode('6ba7b8119dad11d180b400c04fd430c8','hex')
      || convert_to('polymarket-bot/btc-paper/' || :'experiment_key','UTF8'),
    'sha1'
  ) as bytes
), versioned as (
  select set_byte(
    set_byte(substring(bytes from 1 for 16),6,(get_byte(bytes,6) & 15) | 80),
    8,(get_byte(bytes,8) & 63) | 128
  ) as bytes
  from hashed
), encoded as (
  select encode(bytes,'hex') as value from versioned
)
select substr(value,1,8)||'-'||substr(value,9,4)||'-'||substr(value,13,4)||'-'
  ||substr(value,17,4)||'-'||substr(value,21,12)
from encoded;
SQL
)"
  ADMIN_TOKEN="${POLYMARKET_HTTP_ADMIN_TOKEN:-dev-polymarket-admin}"
  curl -fsS -H "Authorization: Bearer $ADMIN_TOKEN" \
    "$HTTP_BASE_URL/admin/trading-processes/$PROCESS_ID/start-preview" \
    > "$PARTIAL_DIR/process-start-preview.json" || {
      unset ADMIN_TOKEN
      echo "LAUNCH BLOCKED: bot start-preview endpoint is unavailable or rejected the definition" >&2
      exit 29
    }
  unset ADMIN_TOKEN
  PREVIEW_CONFIG_SHA256="$(jq -cj '.frozen_process_config' \
    "$PARTIAL_DIR/process-start-preview.json" | sha256_stdin)"
  jq -e \
    --arg process_id "$PROCESS_ID" \
    --arg experiment_id "$EXPECTED_EXPERIMENT_ID" \
    --arg experiment_key "$EXPERIMENT_KEY" \
    --arg preregistration_sha256 "$PREREGISTRATION_SHA256" \
    --arg config_hash "$PREVIEW_CONFIG_SHA256" \
    --argjson expected_frozen_config "$EXPECTED_EXPERIMENT_CONFIG_CONTRACT" '
      (keys | sort) == ([
        "config_hash", "experiment_id", "experiment_key", "frozen_process_config",
        "preregistration_sha256", "process_id"
      ] | sort)
      and .process_id == $process_id
      and .experiment_id == $experiment_id
      and .experiment_key == $experiment_key
      and .preregistration_sha256 == $preregistration_sha256
      and .config_hash == $config_hash
      and (.config_hash | test("^[0-9a-f]{64}$"))
      and .frozen_process_config.raw.preregistration_sha256 == $preregistration_sha256
      and ((.frozen_process_config | del(.raw.preregistration_sha256))
        == $expected_frozen_config)
    ' "$PARTIAL_DIR/process-start-preview.json" >/dev/null || {
      echo "LAUNCH BLOCKED: bot start preview does not match the frozen manifest contract" >&2
      exit 29
    }
  jq -n \
    --arg process_id "$PROCESS_ID" \
    --arg experiment_id "$EXPECTED_EXPERIMENT_ID" \
    --arg config_hash "$PREVIEW_CONFIG_SHA256" \
    '{gate_pass:true,process_id:$process_id,experiment_id:$experiment_id,
      independently_recomputed_config_hash:$config_hash}' \
    > "$PARTIAL_DIR/process-start-preview-gate.json"
  echo "Preflight is read-only and did not control or mutate another service." \
    > "$PARTIAL_DIR/preflight-result.txt"
  exit 0
fi

if [[ -n "$REQUESTED_EXPERIMENT_ID" ]] &&
   [[ ! "$REQUESTED_EXPERIMENT_ID" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]]; then
  echo "invalid experiment UUID: $REQUESTED_EXPERIMENT_ID" >&2
  exit 4
fi

if [[ -n "$REQUESTED_EXPERIMENT_ID" ]]; then
  EXPERIMENT_ID="$REQUESTED_EXPERIMENT_ID"
else
  EXPERIMENT_ROWS="$(psql_audit -At -v key="$EXPERIMENT_KEY" <<'SQL'
select experiment_id
from polymarket.btc_paper_experiments
where name = :'key'
order by started_at;
SQL
)"
  EXPERIMENT_COUNT="$(printf '%s\n' "$EXPERIMENT_ROWS" | awk 'NF {count++} END {print count+0}')"
  if (( EXPERIMENT_COUNT != 1 )); then
    echo "expected exactly one immutable experiment named $EXPERIMENT_KEY; found $EXPERIMENT_COUNT" >&2
    exit 5
  fi
  EXPERIMENT_ID="$(printf '%s\n' "$EXPERIMENT_ROWS" | awk 'NF {print; exit}')"
fi
echo "$EXPERIMENT_ID" > "$PARTIAL_DIR/experiment-id.txt"

ADMIN_TOKEN="${POLYMARKET_HTTP_ADMIN_TOKEN:-dev-polymarket-admin}"
curl -fsS "$HTTP_BASE_URL/health" > "$PARTIAL_DIR/health.json" || {
  [[ "$MODE" == "final" ]] || exit 6
}
curl -fsS "$HTTP_BASE_URL/metrics" > "$PARTIAL_DIR/metrics.prom" || {
  [[ "$MODE" == "final" ]] || exit 6
}
curl -fsS -H "Authorization: Bearer $ADMIN_TOKEN" \
  "$HTTP_BASE_URL/admin/strategy/btc-5m/readiness" \
  > "$PARTIAL_DIR/readiness.json" || { [[ "$MODE" == "final" ]] || exit 6; }
curl -fsS -H "Authorization: Bearer $ADMIN_TOKEN" \
  "$HTTP_BASE_URL/admin/strategy/btc-5m/paper-experiment" \
  > "$PARTIAL_DIR/paper-experiment.json" || { [[ "$MODE" == "final" ]] || exit 6; }
unset ADMIN_TOKEN

psql_audit \
  -v experiment_id="$EXPERIMENT_ID" \
  -v coverage_grace_seconds=5 \
  -v official_resolution_grace_seconds="$(jq -er '.cohort.official_resolution_grace_seconds' "$MANIFEST")" \
  -v resolution_watch_retention_seconds="$(jq -er '.cohort.resolution_watch_retention_seconds' "$MANIFEST")" \
  -v boundary_max_delay_seconds=5 \
  -v required_consecutive_intervals=13 \
  -v minimum_snapshots_per_interval="$(jq -er '.operational_gates.minimum_snapshots_per_complete_window' "$MANIFEST")" \
  -v maximum_snapshot_gap_seconds="$(jq -er '.operational_gates.maximum_snapshot_gap_seconds' "$MANIFEST")" \
  -v preregistration_sha256="$PREREGISTRATION_SHA256" \
  -v expected_experiment_config_contract="$EXPECTED_EXPERIMENT_CONFIG_CONTRACT" \
  < "$READINESS_SQL" > "$PARTIAL_DIR/readiness-audit.txt"

psql_audit \
  -v experiment_id="$EXPERIMENT_ID" \
  -v audit_mode="$MODE" \
  -v experiment_key="$EXPERIMENT_KEY" \
  -v process_key="$PROCESS_KEY" \
  -v preregistration_sha256="$PREREGISTRATION_SHA256" \
  -v expected_experiment_config_contract="$EXPECTED_EXPERIMENT_CONFIG_CONTRACT" \
  -v expected_compiled_source_identity="$(jq -er '.frozen_runtime_contract.compiled_source_identity' "$MANIFEST")" \
  -v required_complete_windows="$(jq -er '.cohort.minimum_expected_complete_windows' "$MANIFEST")" \
  -v minimum_coverage_pct="$(jq -er '.operational_gates.minimum_window_and_label_coverage_percent' "$MANIFEST")" \
  -v minimum_official_slo_pct="$(jq -er '.operational_gates.minimum_official_resolution_slo_coverage_percent' "$MANIFEST")" \
  -v minimum_snapshots_per_window="$(jq -er '.operational_gates.minimum_snapshots_per_complete_window' "$MANIFEST")" \
  -v maximum_snapshot_gap_seconds="$(jq -er '.operational_gates.maximum_snapshot_gap_seconds' "$MANIFEST")" \
  -v official_resolution_grace_seconds="$(jq -er '.cohort.official_resolution_grace_seconds' "$MANIFEST")" \
  -v resolution_watch_retention_seconds="$(jq -er '.cohort.resolution_watch_retention_seconds' "$MANIFEST")" \
  -v minimum_qualifying_fills="$(jq -er '.cohort.minimum_qualifying_filled_orders' "$MANIFEST")" \
  -v minimum_distinct_utc_days="$(jq -er '.cohort.minimum_distinct_utc_days' "$MANIFEST")" \
  -v bootstrap_replicates="$(jq -er '.bootstrap.replicates' "$MANIFEST")" \
  -v bootstrap_seed="$(jq -er '.bootstrap.seed' "$MANIFEST")" \
  -v bootstrap_lower_quantile="$(jq -er '.bootstrap.lower_quantile' "$MANIFEST")" \
  -v bootstrap_upper_quantile="$(jq -er '.bootstrap.upper_quantile' "$MANIFEST")" \
  -v fee_stress_multiplier="$(jq -er '.pessimistic_stress.fee_multiplier' "$MANIFEST")" \
  -v missed_best_fill_fraction="$(jq -er '.pessimistic_stress.missed_best_fill_fraction' "$MANIFEST")" \
  -v probability_uncertainty_multiplier="$(jq -er '.probability_uncertainty_stress.multiplier' "$MANIFEST")" \
  -v maximum_drawdown_usd="$(jq -er '.economic_gates.maximum_drawdown_usd' "$MANIFEST")" \
  -v minimum_worst_trade_pnl_usd="$(jq -er '.economic_gates.minimum_worst_filled_order_net_pnl_usd' "$MANIFEST")" \
  -v maximum_concentration_fraction="$(jq -er '.economic_gates.maximum_positive_pnl_share_from_one_utc_day' "$MANIFEST")" \
  -v pnl_reconciliation_tolerance_usd="$(jq -er '.operational_gates.maximum_pnl_reconciliation_difference_usd' "$MANIFEST")" \
  < "$REPORT_SQL" > "$PARTIAL_DIR/paper-experiment-report.txt"

echo "The wrapper was SELECT-only and did not mutate services or experiment state." \
  > "$PARTIAL_DIR/audit-safety.txt"

CLASSIFICATION="$(sed -n 's/^EXPERIMENT_CLASSIFICATION= //p' \
  "$PARTIAL_DIR/paper-experiment-report.txt" | tail -n 1)"
case "$CLASSIFICATION" in
  COLLECTING|OPERATIONAL_FAILURE|NO_VIABLE_EDGE|PROMISING_INCONCLUSIVE|EDGE_SUPPORTED_FOR_LIVE_CANARY_PLANNING) ;;
  *) echo "unable to parse the authoritative BTC paper experiment classification" >&2; exit 26 ;;
esac
jq -n --arg classification "$CLASSIFICATION" --arg mode "$MODE" \
  '{classification:$classification,audit_mode:$mode}' \
  > "$PARTIAL_DIR/paper-experiment-classification.json"
if [[ "$CLASSIFICATION" == "OPERATIONAL_FAILURE" ]]; then
  exit 30
fi
if [[ "$MODE" == "final" && "$CLASSIFICATION" == "COLLECTING" ]]; then
  exit 31
fi
