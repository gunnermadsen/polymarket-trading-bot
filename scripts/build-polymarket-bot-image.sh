#!/usr/bin/env bash

set -euo pipefail

repository_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "Run this command from within the polymarket-bot Git repository." >&2
  exit 64
}
cd "$repository_root"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing to build an image from an uncommitted working tree." >&2
  echo "Commit or remove all tracked and untracked source changes first." >&2
  exit 65
fi

git_revision="$(git rev-parse --verify 'HEAD^{commit}')"
if [[ ! "$git_revision" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]]; then
  echo "Git returned an invalid full commit ID: $git_revision" >&2
  exit 66
fi

export POLYMARKET_GIT_REVISION="$git_revision"
docker compose build polymarket-bot

if [[ "$(git rev-parse --verify 'HEAD^{commit}')" != "$git_revision" \
  || -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Repository state changed while the image was building; do not deploy this image." >&2
  exit 68
fi

image_revision="$(
  docker image inspect \
    --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' \
    polymarket/polymarket-bot:btc-paper
)"
if [[ "$image_revision" != "$git_revision" ]]; then
  echo "Built image revision mismatch: expected $git_revision, found $image_revision" >&2
  exit 67
fi

printf 'Built polymarket/polymarket-bot:btc-paper from Git revision %s\n' "$git_revision"
