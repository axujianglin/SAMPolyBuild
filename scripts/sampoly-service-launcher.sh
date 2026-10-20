#!/usr/bin/env bash
set -Eeuo pipefail

fail() {
    echo "SAMPoly service startup failed: $*" >&2
    exit 1
}

require_value() {
    local name="$1"
    [[ -n "${!name:-}" ]] || fail "required configuration ${name} is empty"
}

require_file() {
    local name="$1"
    require_value "$name"
    [[ -f "${!name}" ]] || fail "${name} file not found: ${!name}"
}

require_positive_integer() {
    local name="$1"
    require_value "$name"
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] \
        || fail "${name} must be a positive integer: ${!name}"
}

require_nonnegative_integer() {
    local name="$1"
    require_value "$name"
    [[ "${!name}" =~ ^[0-9]+$ ]] \
        || fail "${name} must be a non-negative integer: ${!name}"
}

require_port() {
    local name="$1"
    require_positive_integer "$name"
    (( "${!name}" <= 65535 )) || fail "${name} must not exceed 65535: ${!name}"
}

require_absolute_path() {
    local name="$1"
    require_value "$name"
    [[ "${!name}" == /* ]] || fail "${name} must be an absolute path: ${!name}"
}

service_name="${1:-}"
[[ "$service_name" == "auto" || "$service_name" == "interactive" ]] \
    || fail "usage: sampoly-service-launcher {auto|interactive}"

require_value SAMPOLY_PROJECT_ROOT
require_value SAMPOLY_PYTHON
require_value SAMPOLY_API_KEY
require_positive_integer SAMPOLY_MAX_UPLOAD_MB
require_absolute_path SAMPOLY_PROJECT_ROOT
require_absolute_path SAMPOLY_PYTHON
require_port SAMPOLY_AUTO_PORT
require_port SAMPOLY_REFINE_PORT

[[ -d "$SAMPOLY_PROJECT_ROOT" ]] \
    || fail "project root not found: $SAMPOLY_PROJECT_ROOT"
[[ -x "$SAMPOLY_PYTHON" ]] \
    || fail "configured Python is not executable: $SAMPOLY_PYTHON"
[[ "$SAMPOLY_API_KEY" != "replace-with-a-long-random-secret" ]] \
    || fail "SAMPOLY_API_KEY still contains the example value"
(( ${#SAMPOLY_API_KEY} >= 16 )) || fail "SAMPOLY_API_KEY must contain at least 16 characters"
[[ "$SAMPOLY_AUTO_PORT" != "$SAMPOLY_REFINE_PORT" ]] \
    || fail "Auto and Interactive services cannot use the same port"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SAMPOLY_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$SAMPOLY_PROJECT_ROOT"

if [[ "$service_name" == "auto" ]]; then
    require_value SAMPOLY_AUTO_HOST
    require_absolute_path SAMPOLY_AUTO_CONFIG
    require_absolute_path SAMPOLY_AUTO_CHECKPOINT
    require_absolute_path SAMPOLY_AUTO_RUNTIME_ROOT
    require_file SAMPOLY_AUTO_CONFIG
    require_file SAMPOLY_AUTO_CHECKPOINT
    require_value SAMPOLY_AUTO_DEVICE
    require_nonnegative_integer SAMPOLY_AUTO_GPU
    require_positive_integer SAMPOLY_AUTO_BATCH_SIZE
    require_nonnegative_integer SAMPOLY_AUTO_NUM_WORKERS
    require_value SAMPOLY_AUTO_SCORE_THRESHOLD

    [[ "$SAMPOLY_AUTO_DEVICE" == "cuda" || "$SAMPOLY_AUTO_DEVICE" == "cpu" ]] \
        || fail "SAMPOLY_AUTO_DEVICE must be cuda or cpu"
    mkdir -p "$SAMPOLY_AUTO_RUNTIME_ROOT"

    export SAMPOLY_AUTO_CONFIG
    export SAMPOLY_AUTO_CHECKPOINT
    export SAMPOLY_RUNTIME_ROOT="$SAMPOLY_AUTO_RUNTIME_ROOT"
    export SAMPOLY_DEVICE="$SAMPOLY_AUTO_DEVICE"
    export SAMPOLY_GPU="$SAMPOLY_AUTO_GPU"
    export SAMPOLY_BATCH_SIZE="$SAMPOLY_AUTO_BATCH_SIZE"
    export SAMPOLY_NUM_WORKERS="$SAMPOLY_AUTO_NUM_WORKERS"
    export SAMPOLY_DEFAULT_SCORE_THRESHOLD="$SAMPOLY_AUTO_SCORE_THRESHOLD"

    exec "$SAMPOLY_PYTHON" -m uvicorn api.app:app \
        --host "$SAMPOLY_AUTO_HOST" \
        --port "$SAMPOLY_AUTO_PORT" \
        --workers 1
fi

require_value SAMPOLY_REFINE_HOST
require_absolute_path SAMPOLY_REFINE_CONFIG
require_absolute_path SAMPOLY_REFINE_CHECKPOINT
require_file SAMPOLY_REFINE_CONFIG
require_file SAMPOLY_REFINE_CHECKPOINT
require_value SAMPOLY_REFINE_DEVICE
require_positive_integer SAMPOLY_REFINE_MAX_CONCURRENCY
require_positive_integer SAMPOLY_REFINE_SESSION_TTL_SECONDS
require_positive_integer SAMPOLY_REFINE_MAX_SESSIONS

export SAMPOLY_CONFIG="$SAMPOLY_REFINE_CONFIG"
export SAMPOLY_CHECKPOINT="$SAMPOLY_REFINE_CHECKPOINT"
export SAMPOLY_DEVICE="$SAMPOLY_REFINE_DEVICE"
export SAMPOLY_HOST="$SAMPOLY_REFINE_HOST"
export SAMPOLY_PORT="$SAMPOLY_REFINE_PORT"
export SAMPOLY_MAX_CONCURRENCY="$SAMPOLY_REFINE_MAX_CONCURRENCY"
export SAMPOLY_SESSION_TTL_SECONDS="$SAMPOLY_REFINE_SESSION_TTL_SECONDS"
export SAMPOLY_MAX_SESSIONS="$SAMPOLY_REFINE_MAX_SESSIONS"

exec "$SAMPOLY_PYTHON" -m interactive_api.main
