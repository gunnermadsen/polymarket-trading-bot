#!/usr/bin/env bash

# SELECT-only Phase 6 Compose job. It talks to TimescaleDB and the bot over the
# Compose network, never controls another container, and never archives the admin token.

set -Eeuo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${PHASE6_MANIFEST_PATH:-$ROOT_DIR/docs/experiments/btc-5m-chainlink-fair-value-phase6-20260713-b/preregistration.json}"
READINESS_SQL="$ROOT_DIR/docs/sql/btc-5m-paper-readiness.sql"
PHASE6_SQL="$ROOT_DIR/docs/sql/btc-5m-phase6-report.sql"
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

psql_phase6() {
  psql -X -v ON_ERROR_STOP=1 -h "$DB_HOST" -p "$DB_PORT" \
    -U "$DB_USER" -d "$DB_NAME" "$@"
}

case "$MODE" in
  preflight|daily|final) ;;
  *) usage; exit 2 ;;
esac

[[ -r "$MANIFEST" && -r "$READINESS_SQL" && -r "$PHASE6_SQL" ]] || {
  echo "Phase 6 manifest or SQL input is missing" >&2
  exit 2
}
jq -e '.schema_version == "btc_phase6_preregistration_v1" and .status == "preregistered"' \
  "$MANIFEST" >/dev/null

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

PREREGISTRATION_SHA256="$(sha256_file "$MANIFEST")"

jq -e '
  .artifact_integrity.algorithm == "sha256"
  and (.artifact_integrity.sha256 | type == "object" and length == 4)
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
    echo "Phase 6 evidence archived at $FINAL_DIR"
  fi
  exit "$exit_code"
}
trap finalize EXIT

cp "$MANIFEST" "$PARTIAL_DIR/preregistration.json"
{
  echo "audit_mode=$MODE"
  echo "experiment_key=$EXPERIMENT_KEY"
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
      "${POLYMARKET_BTC_ML_SHADOW_ENABLED:-}" == "true" &&
      "${POLYMARKET_BTC_EXPERIMENT_KEY:-}" == "$EXPERIMENT_KEY" &&
      "${POLYMARKET_BTC_PREREGISTRATION_SHA256:-}" == "$PREREGISTRATION_SHA256" &&
      "${POLYMARKET_BTC_STRATEGY_INTERVAL_MS:-}" == "1000" &&
      "${POLYMARKET_BTC_TARGET_SIZE:-}" == "5" &&
      "${POLYMARKET_BTC_PAPER_LATENCY_MS:-}" == "150" &&
      "${POLYMARKET_BTC_PAPER_VISIBLE_DEPTH_HAIRCUT:-}" == "0.80" &&
      "${POLYMARKET_BTC_OFFICIAL_RESOLUTION_AUDIT_GRACE_SECS:-}" == "120" &&
      "${POLYMARKET_BTC_OFFICIAL_RESOLUTION_WATCH_RETENTION_SECS:-}" == "3600" ]]; then
  CONFIG_GATE_PASS=true
fi
CREDENTIAL_GATE_PASS=false
if [[ -z "${POLYMARKET_CLOB_API_KEY:-}" && -z "${POLYMARKET_CLOB_SECRET:-}" &&
      -z "${POLYMARKET_CLOB_PASSPHRASE:-}" && -z "${POLYMARKET_PRIVATE_KEY:-}" &&
      -z "${POLYMARKET_FUNDER_ADDRESS:-}" && -z "${POLYMARKET_SIGNATURE_TYPE:-}" ]]; then
  CREDENTIAL_GATE_PASS=true
fi
jq -n --argjson exact_safe_config "$CONFIG_GATE_PASS" \
  --argjson all_live_credentials_blank "$CREDENTIAL_GATE_PASS" \
  '{exact_safe_config:$exact_safe_config,all_live_credentials_blank:$all_live_credentials_blank}' \
  > "$PARTIAL_DIR/safety-gates.json"
if [[ "$MODE" == "preflight" &&
      ( "$CONFIG_GATE_PASS" != "true" || "$CREDENTIAL_GATE_PASS" != "true" ) ]]; then
  echo "LAUNCH BLOCKED: shared safety configuration or blank-credential gate failed" >&2
  exit 24
fi

pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
  > "$PARTIAL_DIR/postgres-pg-isready.txt"
psql_phase6 \
  -P pager=off -c "select name, setting from pg_settings where name in
    ('shared_buffers','effective_cache_size','maintenance_work_mem','work_mem',
     'max_connections','max_parallel_workers','max_parallel_workers_per_gather',
     'max_worker_processes','timescaledb.max_background_workers') order by name" \
  > "$PARTIAL_DIR/postgres-memory-settings.txt"

SCHEMA_GATE_PASS="$(psql_phase6 -At -c "
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
printf 'schema_gate_pass=%s\n' "$SCHEMA_GATE_PASS" > "$PARTIAL_DIR/phase6-schema-gate.txt"

if [[ "$MODE" == "preflight" ]]; then
  if [[ "$SCHEMA_GATE_PASS" != "t" ]]; then
    echo "LAUNCH BLOCKED: TimescaleDB migration, schema, index, or policy gate failed" >&2
    exit 25
  fi
  EXISTING_EXPERIMENTS="$(psql_phase6 -At -v key="$EXPERIMENT_KEY" <<'SQL'
select count(*)
from polymarket.btc_paper_experiments
where name = :'key';
SQL
)"
  if [[ "$EXISTING_EXPERIMENTS" != "0" ]]; then
    echo "LAUNCH BLOCKED: immutable experiment key already exists" >&2
    exit 27
  fi
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
  EXPERIMENT_ROWS="$(psql_phase6 -At -v key="$EXPERIMENT_KEY" <<'SQL'
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

ADMIN_TOKEN="${POLYMARKET_HTTP_ADMIN_TOKEN:-}"
curl -fsS "$HTTP_BASE_URL/health" > "$PARTIAL_DIR/health.json" || {
  [[ "$MODE" == "final" ]] || exit 6
}
curl -fsS "$HTTP_BASE_URL/metrics" > "$PARTIAL_DIR/metrics.prom" || {
  [[ "$MODE" == "final" ]] || exit 6
}
if [[ -n "$ADMIN_TOKEN" ]]; then
  curl -fsS -H "Authorization: Bearer $ADMIN_TOKEN" \
    "$HTTP_BASE_URL/admin/strategy/btc-5m/readiness" \
    > "$PARTIAL_DIR/readiness.json" || { [[ "$MODE" == "final" ]] || exit 6; }
  curl -fsS -H "Authorization: Bearer $ADMIN_TOKEN" \
    "$HTTP_BASE_URL/admin/strategy/btc-5m/paper-experiment" \
    > "$PARTIAL_DIR/paper-experiment.json" || { [[ "$MODE" == "final" ]] || exit 6; }
fi
unset ADMIN_TOKEN

psql_phase6 \
  -v experiment_id="$EXPERIMENT_ID" \
  -v coverage_grace_seconds=5 \
  -v official_resolution_grace_seconds="$(jq -er '.cohort.official_resolution_grace_seconds' "$MANIFEST")" \
  -v resolution_watch_retention_seconds="$(jq -er '.cohort.resolution_watch_retention_seconds' "$MANIFEST")" \
  -v boundary_max_delay_seconds=5 \
  -v required_consecutive_intervals=13 \
  -v minimum_snapshots_per_interval="$(jq -er '.operational_gates.minimum_snapshots_per_complete_window' "$MANIFEST")" \
  -v maximum_snapshot_gap_seconds="$(jq -er '.operational_gates.maximum_snapshot_gap_seconds' "$MANIFEST")" \
  < "$READINESS_SQL" > "$PARTIAL_DIR/readiness-audit.txt"

psql_phase6 \
  -v experiment_id="$EXPERIMENT_ID" \
  -v audit_mode="$MODE" \
  -v phase6_experiment_key="$EXPERIMENT_KEY" \
  -v preregistration_sha256="$PREREGISTRATION_SHA256" \
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
  < "$PHASE6_SQL" > "$PARTIAL_DIR/phase6-report.txt"

echo "The wrapper was SELECT-only and did not mutate services or experiment state." \
  > "$PARTIAL_DIR/audit-safety.txt"

CLASSIFICATION="$(sed -n 's/^PHASE6_CLASSIFICATION= //p' \
  "$PARTIAL_DIR/phase6-report.txt" | tail -n 1)"
case "$CLASSIFICATION" in
  COLLECTING|OPERATIONAL_FAILURE|NO_VIABLE_EDGE|PROMISING_INCONCLUSIVE|EDGE_SUPPORTED_FOR_PHASE8_PLANNING) ;;
  *) echo "unable to parse the authoritative Phase 6 classification" >&2; exit 26 ;;
esac
jq -n --arg classification "$CLASSIFICATION" --arg mode "$MODE" \
  '{classification:$classification,audit_mode:$mode}' \
  > "$PARTIAL_DIR/phase6-classification.json"
if [[ "$CLASSIFICATION" == "OPERATIONAL_FAILURE" ]]; then
  exit 30
fi
if [[ "$MODE" == "final" && "$CLASSIFICATION" == "COLLECTING" ]]; then
  exit 31
fi
