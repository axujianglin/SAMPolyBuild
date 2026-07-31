#!/usr/bin/env sh
set -eu

# Export a fully built image for transfer to an offline deployment host.
IMAGE="${1:-sampoly-auto-api:cu117}"
OUTPUT="${2:-sampoly-auto-api-cu117.tar}"

docker image inspect "$IMAGE" >/dev/null
mkdir -p "$(dirname "$OUTPUT")"
docker save --output "$OUTPUT" "$IMAGE"
printf 'Image exported to: %s\n' "$OUTPUT"
