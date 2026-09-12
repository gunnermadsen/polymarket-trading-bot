#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-eu-west-1}"
APP_SECRET_NAME="${APP_SECRET_NAME:-capitonic/polymarket-bot/production}"
CLOUDFLARE_ZONE_NAME="${CLOUDFLARE_ZONE_NAME:-capitonic.com}"
CLOUDFLARE_SSH_HOSTNAME="${CLOUDFLARE_SSH_HOSTNAME:-ssh.capitonic.com}"
CLOUDFLARE_MONITOR_HOSTNAME="${CLOUDFLARE_MONITOR_HOSTNAME:-monitor.capitonic.com}"

if [ -z "${GITHUB_ENV:-}" ]; then
  echo "GITHUB_ENV is required so secrets are not written to stdout." >&2
  exit 1
fi

secret_json="$(
  aws secretsmanager get-secret-value \
    --region "$AWS_REGION" \
    --secret-id "$APP_SECRET_NAME" \
    --query SecretString \
    --output text
)"

json_value() {
  local key="$1"
  jq -r --arg key "$key" '.[$key] // empty' <<<"$secret_json"
}

cloudflare_api_token="${CLOUDFLARE_API_TOKEN:-$(json_value CLOUDFLARE_API_TOKEN)}"
if [ -z "$cloudflare_api_token" ]; then
  echo "Missing CLOUDFLARE_API_TOKEN in environment or AWS Secrets Manager secret $APP_SECRET_NAME." >&2
  exit 1
fi

cloudflare_zone_id="${CLOUDFLARE_ZONE_ID:-$(json_value CLOUDFLARE_ZONE_ID)}"
if [ -z "$cloudflare_zone_id" ]; then
  cloudflare_zone_id="$(
    curl -fsS \
      -H "Authorization: Bearer $cloudflare_api_token" \
      -H "Content-Type: application/json" \
      "https://api.cloudflare.com/client/v4/zones?name=$CLOUDFLARE_ZONE_NAME&status=active" \
      | jq -r '
          if (.success != true) then
            empty
          else
            .result
            | if length == 1 then .[0].id else empty end
          end
        '
  )"
fi

cloudflare_account_id="${CLOUDFLARE_ACCOUNT_ID:-$(json_value CLOUDFLARE_ACCOUNT_ID)}"
cloudflare_access_email="${CLOUDFLARE_EMAIL_ADDRESS:-$(json_value CLOUDFLARE_EMAIL_ADDRESS)}"
if [ -z "$cloudflare_account_id" ] || [ -z "$cloudflare_access_email" ]; then
  echo "Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_EMAIL_ADDRESS in AWS Secrets Manager." >&2
  exit 1
fi
if [ -z "$cloudflare_zone_id" ]; then
  echo "Unable to resolve Cloudflare zone ID for $CLOUDFLARE_ZONE_NAME." >&2
  exit 1
fi

cloudflare_tunnel_id="${CLOUDFLARE_TUNNEL_ID:-$(json_value CLOUDFLARE_TUNNEL_ID)}"
if [ -z "$cloudflare_tunnel_id" ]; then
  cloudflare_tunnel_id="$(json_value CLOUDFLARED_PRODUCTION_TUNNEL_ID)"
fi
if [ -z "$cloudflare_tunnel_id" ]; then
  credentials_b64="$(json_value CLOUDFLARED_PRODUCTION_TUNNEL_CREDENTIALS_B64)"
  if [ -n "$credentials_b64" ]; then
    cloudflare_tunnel_id="$(
      printf '%s' "$credentials_b64" \
        | base64 -d \
        | jq -r '.TunnelID // .tunnelID // .tunnel_id // empty'
    )"
  fi
fi
if [ -z "$cloudflare_tunnel_id" ]; then
  echo "Missing Cloudflare tunnel ID in CLOUDFLARE_TUNNEL_ID, CLOUDFLARED_PRODUCTION_TUNNEL_ID, or tunnel credentials JSON." >&2
  exit 1
fi

if [ "${GITHUB_ACTIONS:-false}" = "true" ]; then
  {
    echo "::add-mask::$cloudflare_api_token"
    echo "::add-mask::$cloudflare_zone_id"
    echo "::add-mask::$cloudflare_tunnel_id"
    echo "::add-mask::$cloudflare_account_id"
    echo "::add-mask::$cloudflare_access_email"
  } >&2
fi

{
  echo "CLOUDFLARE_API_TOKEN=$cloudflare_api_token"
  echo "TF_VAR_aws_region=$AWS_REGION"
  echo "TF_VAR_cloudflare_account_id=$cloudflare_account_id"
  echo "TF_VAR_cloudflare_access_email=$cloudflare_access_email"
  echo "TF_VAR_cloudflare_zone_id=$cloudflare_zone_id"
  echo "TF_VAR_cloudflare_tunnel_id=$cloudflare_tunnel_id"
  echo "TF_VAR_cloudflare_ssh_hostname=$CLOUDFLARE_SSH_HOSTNAME"
  echo "TF_VAR_cloudflare_monitor_hostname=$CLOUDFLARE_MONITOR_HOSTNAME"
} >> "$GITHUB_ENV"

echo "Loaded Cloudflare Terraform inputs for $CLOUDFLARE_SSH_HOSTNAME and $CLOUDFLARE_MONITOR_HOSTNAME."
