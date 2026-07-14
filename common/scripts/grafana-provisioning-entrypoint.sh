#!/bin/sh
set -eu

SRC_DIR="${GRAFANA_PROVISIONING_SRC_DIR:-/etc/grafana/provisioning-src}"
DST_DIR="${GRAFANA_PROVISIONING_DIR:-/tmp/grafana-provisioning}"
POSTGRES_PASSWORD_VALUE="${POSTGRES_PASSWORD:-}"
GRAFANA_ADMIN_USER_VALUE="${GRAFANA_ADMIN_USER:-${GF_SECURITY_ADMIN_USER:-}}"
GRAFANA_ADMIN_PASSWORD_VALUE="${GRAFANA_ADMIN_PASSWORD:-${GF_SECURITY_ADMIN_PASSWORD:-}}"

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

cp "${SRC_DIR}/dashboards/dashboards.yml" "${DST_DIR}/dashboards/dashboards.yml"

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
      maxOpenConns: 5
      maxIdleConns: 2
      maxIdleConnsAuto: false
      connMaxLifetime: 14400
    secureJsonData:
      password: ${POSTGRES_PASSWORD_VALUE}
EOF

export GF_PATHS_PROVISIONING="${DST_DIR}"

exec /run.sh
