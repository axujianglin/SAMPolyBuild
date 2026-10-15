#!/usr/bin/env sh
set -eu

# Load an offline image archive and start the remote-access Compose service.
if [ "$#" -lt 1 ]; then
    printf 'Usage: %s IMAGE_ARCHIVE [ENV_FILE]\n' "$0" >&2
    exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ARCHIVE="$1"
ENV_FILE="${2:-.env}"

if [ ! -f "$ARCHIVE" ]; then
    printf 'Image archive not found: %s\n' "$ARCHIVE" >&2
    exit 1
fi
if [ ! -f "$ENV_FILE" ]; then
    printf 'Environment file not found: %s\n' "$ENV_FILE" >&2
    exit 1
fi

docker load --input "$ARCHIVE"
docker compose --env-file "$ENV_FILE" -f "$SCRIPT_DIR/compose.deploy.yaml" up -d
docker compose --env-file "$ENV_FILE" -f "$SCRIPT_DIR/compose.deploy.yaml" ps
