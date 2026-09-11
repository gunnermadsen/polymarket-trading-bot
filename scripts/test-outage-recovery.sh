#!/usr/bin/env bash
set -euo pipefail

readonly repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly compose_file="$repository_root/packages/market-data-ingester/tests/outage-recovery/docker-compose.yml"
readonly git_revision="$(git -C "$repository_root" rev-parse HEAD)"
readonly project_name="capitonic-recovery-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-$$"
readonly artifact_directory="$repository_root/target/recovery-test/$project_name"
readonly database_network="${project_name}_database"
export RECOVERY_TEST_GIT_REVISION="$git_revision"
export COMPOSE_PROJECT_NAME="$project_name"
export RECOVERY_TEST_DB_MIGRATE_IMAGE="${RECOVERY_TEST_DB_MIGRATE_IMAGE:-${project_name}-db-migrate}"
export RECOVERY_TEST_INGESTER_IMAGE="${RECOVERY_TEST_INGESTER_IMAGE:-${project_name}-ingester}"

compose() {
  docker compose --file "$compose_file" --project-name "$project_name" "$@"
}

capture_diagnostics() {
  mkdir -p "$artifact_directory"
  compose ps --all >"$artifact_directory/compose-ps.txt" 2>&1 || true
  compose logs --no-color >"$artifact_directory/compose.log" 2>&1 || true
  docker network inspect "$database_network" >"$artifact_directory/database-network.json" 2>&1 || true
}

cleanup() {
  local exit_code=$?
  if (( exit_code != 0 )); then
    capture_diagnostics
    echo "Recovery test failed; diagnostics: $artifact_directory" >&2
  elif [[ "${RECOVERY_TEST_KEEP_ARTIFACTS:-false}" == "true" ]]; then
    capture_diagnostics
    echo "Recovery test diagnostics: $artifact_directory"
  fi
  compose down --volumes --remove-orphans >/dev/null 2>&1 || true
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

fail() {
  echo "outage-recovery: $*" >&2
  return 1
}

wait_until() {
  local description=$1
  local timeout_seconds=$2
  shift 2
  local deadline=$((SECONDS + timeout_seconds))
  until "$@"; do
    if (( SECONDS >= deadline )); then
      fail "timed out waiting for $description"
    fi
    sleep 1
  done
}

service_running() {
  [[ "$(compose ps --status running --quiet "$1")" != "" ]]
}

service_healthy() {
  local container
  container="$(compose ps --quiet "$1")"
  [[ -n "$container" ]] \
    && [[ "$(docker inspect --format '{{.State.Health.Status}}' "$container")" == "healthy" ]]
}

logs_contain() {
  compose logs "$1" 2>&1 | grep -Fq "$2"
}

http_status() {
  local url=$1
  compose exec -T probe curl --silent --output /dev/null --write-out '%{http_code}' "$url" 2>/dev/null
}

status_is() {
  local expected=$1
  local url=$2
  [[ "$(http_status "$url")" == "$expected" ]]
}

metric_is() {
  local url=$1
  local metric=$2
  local expected=$3
  compose exec -T probe curl --fail --silent "$url" 2>/dev/null \
    | grep -Eq "^${metric}[[:space:]]+${expected}(\\.0)?$"
}

run_migrations() {
  compose run --rm db-migrate
}

if [[ "${RECOVERY_TEST_SKIP_BUILD:-false}" != "true" ]]; then
  echo "Building isolated recovery-test services"
  compose build db-migrate ingester-master
fi
compose up --detach probe

echo "Verifying database bootstrap retry"
compose up --detach --no-deps ingester-worker
worker_container_before="$(compose ps --quiet ingester-worker)"
wait_until "worker database retry" 15 logs_contain ingester-worker "database connection deferred"
service_running ingester-worker || fail "worker exited while PostgreSQL was unavailable"

compose up --detach postgres
wait_until "PostgreSQL healthcheck" 60 service_healthy postgres
wait_until "database migrations" 30 run_migrations

echo "Verifying master registration retry without process loss"
wait_until "worker telemetry endpoint" 45 status_is 200 http://ingester-worker:8099/health/live
wait_until "worker master-registration retry" 45 logs_contain ingester-worker "registration unavailable"
service_running ingester-worker || fail "worker exited while the master was unavailable"
[[ "$(compose ps --quiet ingester-worker)" == "$worker_container_before" ]] \
  || fail "worker container was replaced during bootstrap recovery"

compose up --detach --no-deps ingester-master
wait_until "master readiness" 45 status_is 200 http://ingester-master:8098/health/ready
wait_until "worker registration" 45 logs_contain ingester-worker "worker registered"
wait_until "worker readiness" 45 status_is 200 http://ingester-worker:8099/health/ready
metric_is http://ingester-worker:8099/prometheus/metrics market_data_ingester_worker_readiness 1 \
  || fail "worker readiness metric did not report healthy"

echo "Verifying master database outage recovery"
master_container="$(compose ps --quiet ingester-master)"
docker network disconnect "$database_network" "$master_container"
wait_until "master readiness to fail" 15 status_is 503 http://ingester-master:8098/health/ready
status_is 200 http://ingester-master:8098/health/live \
  || fail "master liveness failed during a database outage"
service_running ingester-master || fail "master exited during a database outage"
docker network connect "$database_network" "$master_container"
wait_until "master readiness recovery" 30 status_is 200 http://ingester-master:8098/health/ready

echo "Verifying worker reconciliation outage recovery"
worker_container="$(compose ps --quiet ingester-worker)"
docker network disconnect "$database_network" "$worker_container"
wait_until "worker readiness to fail" 15 status_is 503 http://ingester-worker:8099/health/ready
status_is 200 http://ingester-worker:8099/health/live \
  || fail "worker liveness failed during a database outage"
metric_is http://ingester-worker:8099/prometheus/metrics market_data_ingester_worker_readiness 0 \
  || fail "worker readiness metric did not report the outage"
service_running ingester-worker || fail "worker exited during a database outage"
docker network connect "$database_network" "$worker_container"
wait_until "worker readiness recovery" 30 status_is 200 http://ingester-worker:8099/health/ready
metric_is http://ingester-worker:8099/prometheus/metrics market_data_ingester_worker_readiness 1 \
  || fail "worker readiness metric did not recover"

echo "Verifying replacement-worker registration"
compose rm --stop --force ingester-worker
compose up --detach --no-deps ingester-worker
replacement_worker="$(compose ps --quiet ingester-worker)"
[[ "$replacement_worker" != "$worker_container" ]] || fail "worker replacement retained the old container identity"
wait_until "replacement worker readiness" 45 status_is 200 http://ingester-worker:8099/health/ready
wait_until "replacement worker registration" 45 logs_contain ingester-worker "worker registered"

echo "Outage recovery integration test passed"
