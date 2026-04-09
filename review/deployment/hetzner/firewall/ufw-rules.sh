#!/usr/bin/env bash
# =============================================================================
# ufw-rules.sh — UFW firewall configuration for Hetzner GEX44
# Platform: Ubuntu 22.04
#
# USAGE:
#   sudo bash ufw-rules.sh
#
# WHAT THIS DOES:
#   1. Sets default policies (deny in, allow out)
#   2. Opens SSH, HTTP, HTTPS, WireGuard
#   3. Restricts Prometheus exporters and engine metrics to WireGuard subnet
#   4. Enables UFW
#
# IMPORTANT: Run AFTER WireGuard (wg0) is up, otherwise you lock out the
#            monitoring VPS until the tunnel comes back.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Safety check — must be root
# ---------------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo)." >&2
    exit 1
fi

echo "=== Metadron Capital — UFW Firewall Setup ==="
echo "Host: $(hostname)  |  Date: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo

# ---------------------------------------------------------------------------
# Reset existing rules to a clean state
# Caution: this will briefly drop all enforced rules during reset.
# If already connected via SSH, the allow SSH rule below will restore access.
# ---------------------------------------------------------------------------
echo "[1/7] Resetting UFW to defaults..."
ufw --force reset

# ---------------------------------------------------------------------------
# Default policies
# ---------------------------------------------------------------------------
echo "[2/7] Setting default policies: deny incoming, allow outgoing..."
ufw default deny incoming
ufw default allow outgoing

# ---------------------------------------------------------------------------
# SSH — allow from anywhere (Hetzner emergency console is always available
#       as a fallback if you lock yourself out)
#       Restrict to specific IP if your management IP is static.
# ---------------------------------------------------------------------------
echo "[3/7] Allowing SSH (port 22)..."
ufw allow 22/tcp comment "SSH"

# Optional: restrict SSH to a management IP
# ufw allow from <YOUR_MGMT_IP> to any port 22 proto tcp comment "SSH (management only)"

# ---------------------------------------------------------------------------
# Web traffic — HTTP and HTTPS
# Both are required because NGINX handles the redirect and Cloudflare
# probes both ports during health checks.
# ---------------------------------------------------------------------------
echo "[4/7] Allowing HTTP (80) and HTTPS (443)..."
ufw allow 80/tcp  comment "HTTP  — NGINX (redirect to HTTPS)"
ufw allow 443/tcp comment "HTTPS — NGINX (Cloudflare Origin)"

# ---------------------------------------------------------------------------
# WireGuard VPN tunnel — UDP only
# ---------------------------------------------------------------------------
echo "[5/7] Allowing WireGuard (51820/udp)..."
ufw allow 51820/udp comment "WireGuard VPN (Contabo monitoring peer)"

# ---------------------------------------------------------------------------
# Internal monitoring — WireGuard subnet only (10.0.0.0/24)
#
# Ports:
#   9100  — node_exporter            (Prometheus Node Exporter)
#   9113  — nginx-prometheus-exporter
#   9209  — pm2-prometheus-exporter
#   8001  — FastAPI engine API        (metrics + health endpoint)
#
# These ports must NOT be reachable from the public internet.
# The `from 10.0.0.0/24` constraint enforces this at kernel level,
# even if an application accidentally binds to 0.0.0.0.
# ---------------------------------------------------------------------------
echo "[6/7] Restricting exporter ports to WireGuard subnet (10.0.0.0/24)..."

ufw allow from 10.0.0.0/24 to any port 9100 proto tcp \
    comment "node_exporter — WireGuard only"

ufw allow from 10.0.0.0/24 to any port 9113 proto tcp \
    comment "nginx-prometheus-exporter — WireGuard only"

ufw allow from 10.0.0.0/24 to any port 9209 proto tcp \
    comment "pm2-prometheus-exporter — WireGuard only"

ufw allow from 10.0.0.0/24 to any port 8001 proto tcp \
    comment "FastAPI engine API — WireGuard only"

# ---------------------------------------------------------------------------
# Loopback — always allow localhost communication
# (PM2 children talk to each other via 127.0.0.1)
# ---------------------------------------------------------------------------
ufw allow in on lo comment "Loopback interface"

# ---------------------------------------------------------------------------
# Enable UFW
# ---------------------------------------------------------------------------
echo "[7/7] Enabling UFW..."
ufw --force enable

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "=== UFW Rules Summary ==="
ufw status verbose

echo
echo "=== Done ==="
echo "Verify connectivity:"
echo "  - SSH still works from your current session"
echo "  - WireGuard peer (10.0.0.2) can reach :9100, :9113, :9209, :8001"
echo "  - Public cannot reach exporter ports (test with: nmap -p 9100 <GEX44_IP>)"
