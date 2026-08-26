#!/bin/sh
set -eu

SRC_DIR="${GRAFANA_PROVISIONING_SRC_DIR:-/etc/grafana/provisioning-src}"
DST_DIR="${GRAFANA_PROVISIONING_DIR:-/tmp/grafana-provisioning}"
POSTGRES_PASSWORD_VALUE="${POSTGRES_PASSWORD:-}"
GRAFANA_ADMIN_USER_VALUE="${GRAFANA_ADMIN_USER:-${GF_SECURITY_ADMIN_USER:-}}"
GRAFANA_ADMIN_PASSWORD_VALUE="${GRAFANA_ADMIN_PASSWORD:-${GF_SECURITY_ADMIN_PASSWORD:-}}"
POLYMARKET_HTTP_ADMIN_TOKEN_VALUE="${POLYMARKET_HTTP_ADMIN_TOKEN:-}"
GRAFANA_POSTGRES_MAX_OPEN_CONNS_VALUE="${GRAFANA_POSTGRES_MAX_OPEN_CONNS:-}"
GRAFANA_POSTGRES_MAX_IDLE_CONNS_VALUE="${GRAFANA_POSTGRES_MAX_IDLE_CONNS:-}"
GRAFANA_PROMETHEUS_URL_VALUE="${GRAFANA_PROMETHEUS_URL:-}"
GRAFANA_LOKI_URL_VALUE="${GRAFANA_LOKI_URL:-}"
PROMETHEUS_BASIC_AUTH_USER_VALUE="${PROMETHEUS_BASIC_AUTH_USER:-}"
PROMETHEUS_BASIC_AUTH_PASSWORD_VALUE="${PROMETHEUS_BASIC_AUTH_PASSWORD:-}"

require_env() {
  name="$1"
  eval "value=\${$name:-}"
  if [ -z "$value" ]; then
    echo "Missing ${name} for Grafana provisioning" >&2
    exit 1
  fi
}

require_env POSTGRES_PASSWORD
require_env GRAFANA_ADMIN_USER
require_env GRAFANA_ADMIN_PASSWORD
require_env GRAFANA_POSTGRES_DATASOURCE_NAME
require_env GRAFANA_POSTGRES_DATASOURCE_UID
require_env GRAFANA_POSTGRES_HOST
require_env GRAFANA_POSTGRES_PORT
require_env GRAFANA_POSTGRES_DATABASE
require_env GRAFANA_POSTGRES_USER
require_env GRAFANA_POSTGRES_SSL_MODE
require_env GRAFANA_POSTGRES_MAX_OPEN_CONNS
require_env GRAFANA_POSTGRES_MAX_IDLE_CONNS
require_env GRAFANA_ALERT_ENVIRONMENT
require_env GRAFANA_PROMETHEUS_URL
require_env GRAFANA_LOKI_URL
require_env PROMETHEUS_BASIC_AUTH_USER
require_env PROMETHEUS_BASIC_AUTH_PASSWORD
require_env POLYMARKET_HTTP_ADMIN_TOKEN

case "${GRAFANA_POSTGRES_MAX_OPEN_CONNS_VALUE}" in
  ''|*[!0-9]*|0)
    echo "GRAFANA_POSTGRES_MAX_OPEN_CONNS must be a positive integer" >&2
    exit 1
    ;;
esac
case "${GRAFANA_POSTGRES_MAX_IDLE_CONNS_VALUE}" in
  ''|*[!0-9]*)
    echo "GRAFANA_POSTGRES_MAX_IDLE_CONNS must be a non-negative integer" >&2
    exit 1
    ;;
esac
if [ "${GRAFANA_POSTGRES_MAX_IDLE_CONNS_VALUE}" -gt "${GRAFANA_POSTGRES_MAX_OPEN_CONNS_VALUE}" ]; then
  echo "GRAFANA_POSTGRES_MAX_IDLE_CONNS must not exceed GRAFANA_POSTGRES_MAX_OPEN_CONNS" >&2
  exit 1
fi

export GF_SECURITY_ADMIN_USER="$GRAFANA_ADMIN_USER_VALUE"
export GF_SECURITY_ADMIN_PASSWORD="$GRAFANA_ADMIN_PASSWORD_VALUE"

mkdir -p \
  "${DST_DIR}/datasources" \
  "${DST_DIR}/dashboards" \
  "${DST_DIR}/plugins" \
  "${DST_DIR}/alerting"

if [ ! -f "${SRC_DIR}/dashboards/dashboards.yml" ]; then
  echo "Missing Grafana dashboard provisioning source" >&2
  exit 1
fi
if [ ! -f "${SRC_DIR}/alerting/rules-clob-market-data.yml" ]; then
  echo "Missing Grafana alert provisioning source" >&2
  exit 1
fi
if [ ! -f "${SRC_DIR}/alerting/rules-prometheus.yml" ]; then
  echo "Missing Grafana Prometheus alert provisioning source" >&2
  exit 1
fi
if [ ! -f "${SRC_DIR}/alerting/rules-directional-runtime-health.yml" ]; then
  echo "Missing Grafana directional runtime alert provisioning source" >&2
  exit 1
fi

cp "${SRC_DIR}/dashboards/dashboards.yml" "${DST_DIR}/dashboards/dashboards.yml"
cp "${SRC_DIR}/alerting/rules-clob-market-data.yml" "${DST_DIR}/alerting/rules-clob-market-data.yml"
cp "${SRC_DIR}/alerting/rules-prometheus.yml" "${DST_DIR}/alerting/rules-prometheus.yml"
cp "${SRC_DIR}/alerting/rules-directional-runtime-health.yml" "${DST_DIR}/alerting/rules-directional-runtime-health.yml"

cat > "${DST_DIR}/datasources/postgres.yml" <<EOF
apiVersion: 1
datasources:
  - name: ${GRAFANA_POSTGRES_DATASOURCE_NAME}
    type: postgres
    uid: ${GRAFANA_POSTGRES_DATASOURCE_UID}
    access: proxy
    url: ${GRAFANA_POSTGRES_HOST}:${GRAFANA_POSTGRES_PORT}
    user: ${GRAFANA_POSTGRES_USER}
    editable: false
    jsonData:
      database: ${GRAFANA_POSTGRES_DATABASE}
      sslmode: ${GRAFANA_POSTGRES_SSL_MODE}
      postgresVersion: 1400
      timescaledb: true
      maxOpenConns: ${GRAFANA_POSTGRES_MAX_OPEN_CONNS_VALUE}
      maxIdleConns: ${GRAFANA_POSTGRES_MAX_IDLE_CONNS_VALUE}
      maxIdleConnsAuto: false
      connMaxLifetime: 14400
    secureJsonData:
      password: ${POSTGRES_PASSWORD_VALUE}
EOF

cat > "${DST_DIR}/datasources/polymarket-bot-runtime.yml" <<EOF
apiVersion: 1
datasources:
  - name: Polymarket Bot Runtime
    type: yesoreyeram-infinity-datasource
    uid: polymarket-bot-runtime
    access: proxy
    editable: false
    jsonData:
      auth_method: bearerToken
      allowedHosts:
        - http://polymarket-bot:8097
      timeoutInSeconds: 2
      allowDangerousHTTPMethods: false
    secureJsonData:
      bearerToken: ${POLYMARKET_HTTP_ADMIN_TOKEN_VALUE}
EOF

cat > "${DST_DIR}/datasources/prometheus.yml" <<EOF
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    uid: prometheus
    access: proxy
    url: ${GRAFANA_PROMETHEUS_URL_VALUE}
    editable: false
    basicAuth: true
    basicAuthUser: ${PROMETHEUS_BASIC_AUTH_USER_VALUE}
    jsonData:
      httpMethod: POST
      timeInterval: 20s
      prometheusType: Prometheus
      prometheusVersion: 3.13.2
    secureJsonData:
      basicAuthPassword: ${PROMETHEUS_BASIC_AUTH_PASSWORD_VALUE}
EOF

cat > "${DST_DIR}/datasources/loki.yml" <<EOF
apiVersion: 1
datasources:
  - name: Loki
    type: loki
    uid: loki
    access: proxy
    url: ${GRAFANA_LOKI_URL_VALUE}
    editable: false
    jsonData:
      timeout: 60
      maxLines: 1000
EOF

export GF_PATHS_PROVISIONING="${DST_DIR}"

exec /run.sh
