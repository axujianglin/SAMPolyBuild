#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
asset_dir="$project_root/deploy/systemd"

scripts=(
    "$script_dir/sampoly-service-launcher.sh"
    "$script_dir/install-systemd-services.sh"
    "$script_dir/check-systemd-services.sh"
    "$script_dir/uninstall-systemd-services.sh"
)
for script in "${scripts[@]}"; do
    bash -n "$script"
    echo "syntax OK: ${script#$project_root/}"
done

required_keys=(
    SAMPOLY_PROJECT_ROOT SAMPOLY_PYTHON SAMPOLY_API_KEY
    SAMPOLY_AUTO_PORT SAMPOLY_AUTO_CONFIG SAMPOLY_AUTO_CHECKPOINT
    SAMPOLY_REFINE_PORT SAMPOLY_REFINE_CONFIG SAMPOLY_REFINE_CHECKPOINT
)
for key in "${required_keys[@]}"; do
    grep -Eq "^[[:space:]]*${key}=" "$asset_dir/sampoly.env.example" \
        || {
            echo "missing example configuration key: $key" >&2
            exit 1
        }
done

grep -q '@RUN_USER@' "$asset_dir/sampoly-auto.service.in"
grep -q '@RUN_GROUP@' "$asset_dir/sampoly-auto.service.in"
grep -q '@RUN_USER@' "$asset_dir/sampoly-interactive.service.in"
grep -q '@RUN_GROUP@' "$asset_dir/sampoly-interactive.service.in"
grep -q 'sampoly-auto.service' "$asset_dir/sampoly.target"
grep -q 'sampoly-interactive.service' "$asset_dir/sampoly.target"

echo "SAMPoly systemd deployment assets are structurally valid."
