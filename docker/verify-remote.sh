#!/usr/bin/env sh
set -eu

# Verify readiness and optionally perform one real remote inference request.
BASE_URL="${1:-http://127.0.0.1:8000}"
API_KEY="${2:-}"
IMAGE="${3:-}"
OUTPUT="${4:-auto-response.json}"
BASE_URL=${BASE_URL%/}

curl --fail-with-body --silent --show-error "$BASE_URL/health"
printf '\n'

if [ -n "$IMAGE" ]; then
    if [ ! -f "$IMAGE" ]; then
        printf 'Test image not found: %s\n' "$IMAGE" >&2
        exit 1
    fi
    if [ -n "$API_KEY" ]; then
        curl --fail-with-body --silent --show-error \
            -X POST "$BASE_URL/v1/auto/infer" \
            -H "X-API-Key: $API_KEY" \
            -F "image=@$IMAGE" \
            -o "$OUTPUT"
    else
        curl --fail-with-body --silent --show-error \
            -X POST "$BASE_URL/v1/auto/infer" \
            -F "image=@$IMAGE" \
            -o "$OUTPUT"
    fi
    printf 'Inference response saved to: %s\n' "$OUTPUT"
fi
