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

export INGESTER_GIT_REVISION="$git_revision"
master_image="${INGESTER_MASTER_IMAGE:-capitonic/ingester-master:development}"
worker_image="${INGESTER_WORKER_IMAGE:-capitonic/ingester-worker:development}"

docker compose \
  --profile data-ingestion \
  build ingester-master ingester-worker

if [[ "$(git rev-parse --verify 'HEAD^{commit}')" != "$git_revision" \
  || -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Repository state changed while the image was building; do not use this image." >&2
  exit 68
fi

for image in "$master_image" "$worker_image"; do
  image_revision="$(
    docker image inspect \
      --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' \
      "$image"
  )"
  if [[ "$image_revision" != "$git_revision" ]]; then
    echo "Built image revision mismatch: expected $git_revision, found $image_revision" >&2
    exit 67
  fi
  image_id="$(docker image inspect --format '{{ .Id }}' "$image")"
  if [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "Docker returned an invalid image ID for $image: $image_id" >&2
    exit 69
  fi
  printf 'Built %s from Git revision %s\n' "$image" "$git_revision"
  printf 'Image ID: %s\n' "$image_id"
done
