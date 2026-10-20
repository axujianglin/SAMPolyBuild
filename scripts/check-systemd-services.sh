#!/usr/bin/env bash
set -Eeuo pipefail

auto_url="${1:-http://127.0.0.1:8000}"
interactive_url="${2:-http://127.0.0.1:8010}"

check_unit() {
    local unit="$1"
    if systemctl is-active --quiet "$unit"; then
        echo "$unit: active"
    else
        echo "$unit: inactive or failed" >&2
        systemctl status "$unit" --no-pager || true
        return 1
    fi
}

check_health() {
    local name="$1"
    local url="$2"
    local payload
    payload="$(curl --silent --show-error --max-time 10 "$url/health")" \
        || {
            echo "$name health request failed: $url/health" >&2
            return 1
        }
    printf '%s health: %s\n' "$name" "$payload"
    HEALTH_PAYLOAD="$payload" python3 - "$name" <<'PY'
import json
import os
import sys

name = sys.argv[1]
try:
    payload = json.loads(os.environ["HEALTH_PAYLOAD"])
except (KeyError, json.JSONDecodeError) as exc:
    raise SystemExit(f"{name} returned invalid health JSON: {exc}")
if payload.get("status") != "ready":
    raise SystemExit(f"{name} is not ready: {payload}")
PY
}

failed=0
check_unit sampoly-auto.service || failed=1
check_unit sampoly-interactive.service || failed=1
check_health auto "$auto_url" || failed=1
check_health interactive "$interactive_url" || failed=1

if [[ "$failed" -ne 0 ]]; then
    echo "One or more SAMPoly service checks failed." >&2
    exit 1
fi
echo "All SAMPoly services are active and ready."
