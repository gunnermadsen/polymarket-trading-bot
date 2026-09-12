#!/usr/bin/env bash
set -euo pipefail

TF_DIR="${TF_DIR:-infra/ec2-compose}"

: "${CLOUDFLARE_API_TOKEN:?CLOUDFLARE_API_TOKEN is required}"
: "${TF_VAR_cloudflare_zone_id:?TF_VAR_cloudflare_zone_id is required}"
: "${TF_VAR_cloudflare_ssh_hostname:?TF_VAR_cloudflare_ssh_hostname is required}"
: "${TF_VAR_cloudflare_monitor_hostname:?TF_VAR_cloudflare_monitor_hostname is required}"

import_record_if_present() {
  local address hostname response count record_id
  address="$1"
  hostname="$2"

  if terraform -chdir="$TF_DIR" state list | grep -qx "$address"; then
    echo "$address is already managed in Terraform state."
    return
  fi

  response="$(
    curl -fsS \
      -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
      -H "Content-Type: application/json" \
      "https://api.cloudflare.com/client/v4/zones/$TF_VAR_cloudflare_zone_id/dns_records?type=CNAME&name=$hostname"
  )"

  count="$(jq -r '.result | length' <<<"$response")"
  case "$count" in
    0)
      echo "No existing Cloudflare CNAME found for $hostname; Terraform will create it."
      ;;
    1)
      record_id="$(jq -r '.result[0].id' <<<"$response")"
      echo "Importing existing Cloudflare CNAME for $hostname into $address."
      terraform -chdir="$TF_DIR" import -input=false "$address" "$TF_VAR_cloudflare_zone_id/$record_id"
      ;;
    *)
      echo "Multiple Cloudflare CNAME records found for $hostname; refusing to choose one." >&2
      exit 1
      ;;
  esac
}

import_record_if_present cloudflare_dns_record.ssh_ops "$TF_VAR_cloudflare_ssh_hostname"
import_record_if_present cloudflare_dns_record.monitor "$TF_VAR_cloudflare_monitor_hostname"
