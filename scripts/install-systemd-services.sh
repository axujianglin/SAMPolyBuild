#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  sudo bash scripts/install-systemd-services.sh --user USER --config /absolute/path/sampoly.env

Installs the native SAMPoly Auto and Interactive systemd services. The source
configuration is copied to /etc/sampoly/sampoly.env with mode 0600.
EOF
}

fail() {
    echo "SAMPoly systemd installation failed: $*" >&2
    exit 1
}

run_user=""
config_path=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --user)
            [[ $# -ge 2 ]] || fail "--user requires a value"
            run_user="$2"
            shift 2
            ;;
        --config)
            [[ $# -ge 2 ]] || fail "--config requires a value"
            config_path="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

[[ "${EUID}" -eq 0 ]] || fail "run this installer with sudo"
command -v systemctl >/dev/null 2>&1 || fail "systemctl was not found on this host"
command -v systemd-analyze >/dev/null 2>&1 || fail "systemd-analyze was not found on this host"
[[ "$run_user" =~ ^[a-z_][a-z0-9_-]*[$]?$ ]] \
    || fail "--user must be an existing Linux account name"
id "$run_user" >/dev/null 2>&1 || fail "Linux user not found: $run_user"
[[ "$config_path" == /* ]] || fail "--config must be an absolute path"
[[ -f "$config_path" ]] || fail "configuration file not found: $config_path"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
asset_dir="$project_root/deploy/systemd"
run_group="$(id -gn "$run_user")"

required_assets=(
    "$asset_dir/sampoly-auto.service.in"
    "$asset_dir/sampoly-interactive.service.in"
    "$asset_dir/sampoly.target"
    "$script_dir/sampoly-service-launcher.sh"
)
for asset in "${required_assets[@]}"; do
    [[ -f "$asset" ]] || fail "installation asset not found: $asset"
done

required_keys=(
    SAMPOLY_PROJECT_ROOT SAMPOLY_PYTHON SAMPOLY_API_KEY SAMPOLY_MAX_UPLOAD_MB
    SAMPOLY_AUTO_HOST SAMPOLY_AUTO_PORT SAMPOLY_AUTO_CONFIG
    SAMPOLY_AUTO_CHECKPOINT SAMPOLY_AUTO_RUNTIME_ROOT SAMPOLY_AUTO_DEVICE
    SAMPOLY_AUTO_GPU SAMPOLY_AUTO_BATCH_SIZE SAMPOLY_AUTO_NUM_WORKERS
    SAMPOLY_AUTO_SCORE_THRESHOLD SAMPOLY_REFINE_HOST SAMPOLY_REFINE_PORT
    SAMPOLY_REFINE_CONFIG SAMPOLY_REFINE_CHECKPOINT SAMPOLY_REFINE_DEVICE
    SAMPOLY_REFINE_MAX_CONCURRENCY SAMPOLY_REFINE_SESSION_TTL_SECONDS
    SAMPOLY_REFINE_MAX_SESSIONS
)
for key in "${required_keys[@]}"; do
    grep -Eq "^[[:space:]]*${key}=" "$config_path" \
        || fail "configuration key is missing: $key"
done

install -d -m 0700 /etc/sampoly
install -m 0600 "$config_path" /etc/sampoly/sampoly.env
install -d -m 0755 /usr/local/libexec
install -m 0755 "$script_dir/sampoly-service-launcher.sh" \
    /usr/local/libexec/sampoly-service-launcher

temporary_dir="$(mktemp -d)"
trap 'rm -rf "$temporary_dir"' EXIT
for service in sampoly-auto sampoly-interactive; do
    sed \
        -e "s|@RUN_USER@|$run_user|g" \
        -e "s|@RUN_GROUP@|$run_group|g" \
        "$asset_dir/${service}.service.in" \
        > "$temporary_dir/${service}.service"
done
systemd-analyze verify \
    "$temporary_dir/sampoly-auto.service" \
    "$temporary_dir/sampoly-interactive.service" \
    "$asset_dir/sampoly.target"

for service in sampoly-auto sampoly-interactive; do
    install -m 0644 "$temporary_dir/${service}.service" \
        "/etc/systemd/system/${service}.service"
done
install -m 0644 "$asset_dir/sampoly.target" /etc/systemd/system/sampoly.target

systemctl daemon-reload
systemctl enable --now sampoly.target

echo "SAMPoly services were installed and enabled for boot."
echo "Configuration: /etc/sampoly/sampoly.env"
echo "Status: sudo systemctl status sampoly.target"
echo "Health: bash $project_root/scripts/check-systemd-services.sh"
