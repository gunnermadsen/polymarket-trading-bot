#!/bin/sh
set -eu

SRC_CONFIG="${PROMETHEUS_CONFIG_SOURCE:-/etc/prometheus-src/prometheus.yml}"
RUNTIME_DIR="${PROMETHEUS_RUNTIME_DIR:-/tmp/prometheus-provisioning}"
RUNTIME_CONFIG="${RUNTIME_DIR}/prometheus.yml"
WEB_CONFIG="${RUNTIME_DIR}/web.yml"
PASSWORD_FILE="${RUNTIME_DIR}/basic-auth-password"

require_env() {
  name="$1"
  eval "value=\${$name:-}"
  if [ -z "$value" ]; then
    echo "Missing ${name} for Prometheus provisioning" >&2
    exit 1
  fi
}

require_env PROMETHEUS_BASIC_AUTH_USER
require_env PROMETHEUS_BASIC_AUTH_PASSWORD
require_env PROMETHEUS_BASIC_AUTH_PASSWORD_HASH
require_env PROMETHEUS_ENVIRONMENT

case "${PROMETHEUS_BASIC_AUTH_USER}" in
  *[!A-Za-z0-9._-]* )
    echo "PROMETHEUS_BASIC_AUTH_USER contains unsupported characters" >&2
    exit 1
    ;;
esac

case "${PROMETHEUS_BASIC_AUTH_PASSWORD_HASH}" in
  '$2a$'*|'$2b$'*|'$2y$'*) ;;
  *)
    echo "PROMETHEUS_BASIC_AUTH_PASSWORD_HASH must be a bcrypt hash" >&2
    exit 1
    ;;
esac

if [ ! -f "${SRC_CONFIG}" ]; then
  echo "Missing Prometheus configuration source" >&2
  exit 1
fi

mkdir -p "${RUNTIME_DIR}"
umask 077
printf '%s' "${PROMETHEUS_BASIC_AUTH_PASSWORD}" > "${PASSWORD_FILE}"

cat > "${WEB_CONFIG}" <<EOF
basic_auth_users:
  ${PROMETHEUS_BASIC_AUTH_USER}: '${PROMETHEUS_BASIC_AUTH_PASSWORD_HASH}'
EOF

cat > "${RUNTIME_CONFIG}" <<EOF
global:
  scrape_interval: 20s
  scrape_timeout: 10s
  evaluation_interval: 20s
  external_labels:
    environment: ${PROMETHEUS_ENVIRONMENT}
    service: prometheus
    deployment: docker-compose

scrape_configs:
  - job_name: prometheus
    scheme: http
    basic_auth:
      username: ${PROMETHEUS_BASIC_AUTH_USER}
      password_file: ${PASSWORD_FILE}
    static_configs:
      - targets:
          - 127.0.0.1:9090
  - job_name: polymarket-bot
    metrics_path: /prometheus/metrics
    static_configs:
      - targets:
          - polymarket-bot:8097
  - job_name: market-data-ingester
    metrics_path: /prometheus/metrics
    static_configs:
      - targets:
          - market-data-ingester:8098
EOF

exec /bin/prometheus \
  --config.file="${RUNTIME_CONFIG}" \
  --web.config.file="${WEB_CONFIG}" \
  "$@"
