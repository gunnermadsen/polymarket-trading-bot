#!/usr/bin/env bash
set -euo pipefail
set +x

AWS_REGION="${AWS_REGION:-eu-west-1}"
APP_SECRET_NAME="${APP_SECRET_NAME:-capitonic/polymarket-bot/production}"

if [ -z "${GITHUB_ENV:-}" ]; then
  echo "GITHUB_ENV is required so sensitive values are not written to stdout." >&2
  exit 1
fi

secret_json="$(
  aws secretsmanager get-secret-value \
    --region "$AWS_REGION" \
    --secret-id "$APP_SECRET_NAME" \
    --query SecretString \
    --output text
)"

required_value() {
  local key="$1"
  local value
  value="$(jq -r --arg key "$key" '.[$key] // empty' <<<"$secret_json")"
  if [ -z "$value" ]; then
    echo "Missing $key in AWS Secrets Manager secret $APP_SECRET_NAME." >&2
    exit 1
  fi
  printf '%s' "$value"
}

cloudflare_api_token="$(required_value CLOUDFLARE_API_TOKEN)"
cloudflare_account_id="$(required_value CLOUDFLARE_ACCOUNT_ID)"
cloudflare_zone_id="$(required_value CLOUDFLARE_ZONE_ID)"
cloudflare_access_email="$(required_value CLOUDFLARE_EMAIL_ADDRESS)"

{
  echo "::add-mask::$cloudflare_api_token"
  echo "::add-mask::$cloudflare_account_id"
  echo "::add-mask::$cloudflare_zone_id"
  echo "::add-mask::$cloudflare_access_email"
} >&2

{
  echo "CLOUDFLARE_API_TOKEN=$cloudflare_api_token"
  echo "TF_VAR_aws_region=$AWS_REGION"
  echo "TF_VAR_cloudflare_account_id=$cloudflare_account_id"
  echo "TF_VAR_cloudflare_zone_id=$cloudflare_zone_id"
  echo "TF_VAR_cloudflare_access_email=$cloudflare_access_email"
} >> "$GITHUB_ENV"

echo "Loaded masked management-host Terraform inputs from $APP_SECRET_NAME in $AWS_REGION."
