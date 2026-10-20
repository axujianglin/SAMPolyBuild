#!/usr/bin/env bash
set -Eeuo pipefail

purge_config=0
if [[ "${1:-}" == "--purge-config" ]]; then
    purge_config=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: sudo bash scripts/uninstall-systemd-services.sh [--purge-config]" >&2
    exit 2
fi

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this uninstaller with sudo." >&2
    exit 1
fi

systemctl disable --now sampoly.target 2>/dev/null || true
rm -f \
    /etc/systemd/system/sampoly.target \
    /etc/systemd/system/sampoly-auto.service \
    /etc/systemd/system/sampoly-interactive.service \
    /usr/local/libexec/sampoly-service-launcher
systemctl daemon-reload
systemctl reset-failed sampoly-auto.service sampoly-interactive.service 2>/dev/null || true

if [[ "$purge_config" -eq 1 ]]; then
    rm -f /etc/sampoly/sampoly.env
    rmdir /etc/sampoly 2>/dev/null || true
    echo "SAMPoly services and configuration were removed."
else
    echo "SAMPoly services were removed. Configuration was preserved at /etc/sampoly/sampoly.env."
fi
