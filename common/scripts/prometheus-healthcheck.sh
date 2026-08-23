#!/bin/sh
set -eu

auth="$(printf '%s:%s' "${PROMETHEUS_BASIC_AUTH_USER}" "${PROMETHEUS_BASIC_AUTH_PASSWORD}" | base64 | tr -d '\n')"
exec wget -q --spider \
  --header="Authorization: Basic ${auth}" \
  http://127.0.0.1:9090/-/healthy
