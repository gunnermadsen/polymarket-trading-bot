#!/bin/sh
set -eu

: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"

export DB_USER="${POSTGRES_USER:-postgres}"
export DB_PASSWORD="${POSTGRES_PASSWORD}"
export AUTH_TYPE="scram-sha-256"
export AUTH_FILE="/tmp/pgbouncer/userlist.txt"

mkdir -p /tmp/pgbouncer
exec /entrypoint.sh "$@"
