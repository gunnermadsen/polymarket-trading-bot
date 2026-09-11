#!/bin/sh
set -eu

: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"
: "${CAPITONIC_TRADING_POSTGRES_PASSWORD:?CAPITONIC_TRADING_POSTGRES_PASSWORD is required}"
: "${CAPITONIC_INGESTER_MASTER_POSTGRES_PASSWORD:?CAPITONIC_INGESTER_MASTER_POSTGRES_PASSWORD is required}"
: "${CAPITONIC_INGESTER_WORKER_POSTGRES_PASSWORD:?CAPITONIC_INGESTER_WORKER_POSTGRES_PASSWORD is required}"
: "${CAPITONIC_GRAFANA_POSTGRES_PASSWORD:?CAPITONIC_GRAFANA_POSTGRES_PASSWORD is required}"

auth_file="/tmp/pgbouncer/userlist.txt"

mkdir -p /tmp/pgbouncer

escape_password() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

{
  printf '"postgres" "%s"\n' "$(escape_password "$POSTGRES_PASSWORD")"
  printf '"capitonic_trading" "%s"\n' "$(escape_password "$CAPITONIC_TRADING_POSTGRES_PASSWORD")"
  printf '"capitonic_ingester_master" "%s"\n' "$(escape_password "$CAPITONIC_INGESTER_MASTER_POSTGRES_PASSWORD")"
  printf '"capitonic_ingester_worker" "%s"\n' "$(escape_password "$CAPITONIC_INGESTER_WORKER_POSTGRES_PASSWORD")"
  printf '"capitonic_grafana" "%s"\n' "$(escape_password "$CAPITONIC_GRAFANA_POSTGRES_PASSWORD")"
} > "$auth_file"
chmod 0600 "$auth_file"

exec "$@"
