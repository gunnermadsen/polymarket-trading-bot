#!/bin/sh
set -eu

umask 077

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if [ "$(git -C "$repository_root" rev-parse --show-toplevel)" != "$repository_root" ] || [ ! -d "$repository_root/.git" ]; then
  echo "Run this script from the primary repository checkout, not a linked worktree." >&2
  exit 1
fi

roles_file="$repository_root/.env.postgres.roles"
pgbouncer_file="$repository_root/.env.postgres.pgbouncer"
bot_file="$repository_root/.env.postgres.polymarket-bot"
master_file="$repository_root/.env.postgres.ingester-master"
worker_file="$repository_root/.env.postgres.ingester-worker"
grafana_file="$repository_root/.env.postgres.grafana"

for target in "$roles_file" "$pgbouncer_file" "$bot_file" "$master_file" "$worker_file" "$grafana_file"; do
  if [ -e "$target" ]; then
    echo "Refusing to overwrite existing credential file: $target" >&2
    exit 1
  fi
done

command -v openssl >/dev/null 2>&1 || {
  echo "openssl is required to generate PostgreSQL service credentials." >&2
  exit 1
}

temporary_directory=$(mktemp -d "${TMPDIR:-/tmp}/capitonic-postgres-credentials.XXXXXX")
cleanup() {
  rm -rf "$temporary_directory"
}
trap cleanup EXIT HUP INT TERM

trading_password=$(openssl rand -hex 32)
master_password=$(openssl rand -hex 32)
worker_password=$(openssl rand -hex 32)
grafana_password=$(openssl rand -hex 32)

write_secret() {
  destination=$1
  content=$2
  temporary_file="$temporary_directory/$(basename "$destination")"
  printf '%s\n' "$content" > "$temporary_file"
  chmod 0600 "$temporary_file"
  mv "$temporary_file" "$destination"
}

write_secret "$roles_file" "CAPITONIC_TRADING_POSTGRES_PASSWORD=$trading_password
CAPITONIC_INGESTER_MASTER_POSTGRES_PASSWORD=$master_password
CAPITONIC_INGESTER_WORKER_POSTGRES_PASSWORD=$worker_password
CAPITONIC_GRAFANA_POSTGRES_PASSWORD=$grafana_password"

write_secret "$pgbouncer_file" "CAPITONIC_TRADING_POSTGRES_PASSWORD=$trading_password
CAPITONIC_INGESTER_MASTER_POSTGRES_PASSWORD=$master_password
CAPITONIC_INGESTER_WORKER_POSTGRES_PASSWORD=$worker_password
CAPITONIC_GRAFANA_POSTGRES_PASSWORD=$grafana_password"

write_secret "$bot_file" "POSTGRES_PASSWORD=$trading_password"
write_secret "$master_file" "POSTGRES_PASSWORD=$master_password"
write_secret "$worker_file" "POSTGRES_PASSWORD=$worker_password"
write_secret "$grafana_file" "POSTGRES_PASSWORD=$grafana_password"

echo "Generated PostgreSQL service credential files with mode 0600."
