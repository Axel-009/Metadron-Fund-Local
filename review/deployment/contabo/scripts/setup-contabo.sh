#!/usr/bin/env bash
# =============================================================================
# setup-contabo.sh — Metadron Monitoring Stack Setup Script
# =============================================================================
# Run as root on a fresh Contabo VPS (Ubuntu 22.04)
#
# What this script does:
#   1. System update
#   2. Install Docker + Docker Compose v2
#   3. Install WireGuard and configure tunnel
#   4. Install NGINX + Certbot
#   5. Create directory structure and copy configs
#   6. Configure environment variables
#   7. Deploy the monitoring stack with Docker Compose
#   8. Configure NGINX reverse proxy
#   9. Set up UFW firewall
#  10. Enable systemd services for auto-start
#  11. Print summary
#
# Usage:
#   # 1. Copy repo to server:
#   #    scp -r ./contabo/ root@<CONTABO_IP>:/opt/metadron-monitoring/
#   # 2. SSH in and run:
#   bash /opt/metadron-monitoring/scripts/setup-contabo.sh
#
# PREREQUISITES before running:
#   - Have your WireGuard keys ready (or this script generates them)
#   - Have GEX44 public IP and WireGuard public key
#   - Have Grafana admin password ready
# =============================================================================

set -euo pipefail   # Exit on error, undefined var, pipeline failure
IFS=$'\n\t'

# =============================================================================
# Configuration — Edit these before running
# =============================================================================
DEPLOY_DIR="/opt/metadron-monitoring"
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-}"       # Set via env or prompted
GEX44_PUBLIC_IP="${GEX44_PUBLIC_IP:-}"                     # Hetzner GEX44 public IP
GEX44_WG_PUBLIC_KEY="${GEX44_WG_PUBLIC_KEY:-}"             # GEX44 WireGuard public key
WG_PRIVATE_KEY="${WG_PRIVATE_KEY:-}"                       # Contabo WG private key (generated if empty)
DOMAIN_GRAFANA="monitor.metadroncapital.com"
DOMAIN_STATUS="status.metadroncapital.com"

# Colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================
log()     { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
error()   { echo -e "${RED}[✗]${NC} $*" >&2; }
section() { echo -e "\n${BLUE}${BOLD}━━━ $* ━━━${NC}"; }
prompt()  { read -r -p "$(echo -e "${YELLOW}[?]${NC} $1: ")" "$2"; }

check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root."
        exit 1
    fi
}

# =============================================================================
# Gather required inputs
# =============================================================================
gather_inputs() {
    section "Configuration"

    if [[ -z "$GRAFANA_ADMIN_PASSWORD" ]]; then
        prompt "Grafana admin password (min 8 chars)" GRAFANA_ADMIN_PASSWORD
        if [[ ${#GRAFANA_ADMIN_PASSWORD} -lt 8 ]]; then
            error "Password too short"; exit 1
        fi
    fi

    if [[ -z "$GEX44_PUBLIC_IP" ]]; then
        prompt "GEX44 public IP address" GEX44_PUBLIC_IP
    fi

    if [[ -z "$GEX44_WG_PUBLIC_KEY" ]]; then
        prompt "GEX44 WireGuard public key (leave blank to enter later)" GEX44_WG_PUBLIC_KEY
        [[ -z "$GEX44_WG_PUBLIC_KEY" ]] && GEX44_WG_PUBLIC_KEY="<GEX44_WIREGUARD_PUBLIC_KEY>"
        warn "Remember to update wireguard/wg0.conf with the actual GEX44 public key!"
    fi

    log "Configuration collected."
}

# =============================================================================
# Step 1: System Update
# =============================================================================
system_update() {
    section "Step 1: System Update"
    apt-get update -y
    apt-get upgrade -y
    apt-get install -y \
        curl \
        wget \
        git \
        gnupg \
        ca-certificates \
        lsb-release \
        software-properties-common \
        apt-transport-https \
        htop \
        vim \
        jq \
        net-tools \
        ufw \
        fail2ban
    log "System updated."
}

# =============================================================================
# Step 2: Install Docker + Docker Compose v2
# =============================================================================
install_docker() {
    section "Step 2: Install Docker"

    if command -v docker &>/dev/null; then
        warn "Docker already installed: $(docker --version)"
        return
    fi

    # Add Docker's official GPG key
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    # Add Docker apt repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Enable and start Docker
    systemctl enable docker
    systemctl start docker

    log "Docker installed: $(docker --version)"
    log "Docker Compose: $(docker compose version)"
}

# =============================================================================
# Step 3: Install WireGuard
# =============================================================================
install_wireguard() {
    section "Step 3: Install WireGuard"

    apt-get install -y wireguard wireguard-tools

    # Generate WireGuard keys for Contabo if not provided
    if [[ -z "$WG_PRIVATE_KEY" ]]; then
        log "Generating WireGuard key pair for Contabo..."
        WG_PRIVATE_KEY=$(wg genkey)
        WG_PUBLIC_KEY=$(echo "$WG_PRIVATE_KEY" | wg pubkey)
        log "Contabo WireGuard public key: ${BOLD}${WG_PUBLIC_KEY}${NC}"
        warn "Add this public key as a [Peer] on the GEX44 wg0.conf!"
    fi

    # Install WireGuard config
    mkdir -p /etc/wireguard
    chmod 700 /etc/wireguard

    cat > /etc/wireguard/wg0.conf <<WGEOF
[Interface]
Address = 10.0.0.2/24
PrivateKey = ${WG_PRIVATE_KEY}
PostUp   = iptables -A FORWARD -i %i -j ACCEPT; iptables -A FORWARD -o %i -j ACCEPT
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -D FORWARD -o %i -j ACCEPT

[Peer]
PublicKey = ${GEX44_WG_PUBLIC_KEY}
Endpoint = ${GEX44_PUBLIC_IP}:51820
AllowedIPs = 10.0.0.1/32
PersistentKeepalive = 25
WGEOF

    chmod 600 /etc/wireguard/wg0.conf

    # Enable IP forwarding (needed for WireGuard routing)
    echo "net.ipv4.ip_forward=1" > /etc/sysctl.d/99-wireguard.conf
    sysctl --system

    # Enable WireGuard at boot
    systemctl enable wg-quick@wg0
    systemctl start wg-quick@wg0 || warn "WireGuard failed to start — check GEX44 key/IP"

    log "WireGuard installed. Tunnel status:"
    wg show || true
}

# =============================================================================
# Step 4: Install NGINX
# =============================================================================
install_nginx() {
    section "Step 4: Install NGINX"

    if command -v nginx &>/dev/null; then
        warn "NGINX already installed: $(nginx -v 2>&1)"
    else
        apt-get install -y nginx
        systemctl enable nginx
    fi

    log "NGINX installed: $(nginx -v 2>&1 || true)"
}

# =============================================================================
# Step 5: Create Directory Structure & Copy Configs
# =============================================================================
setup_directories() {
    section "Step 5: Setup Directories & Configs"

    # The script assumes it's run from inside the deployment directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SOURCE_DIR="$(dirname "$SCRIPT_DIR")"   # Parent of scripts/

    if [[ ! -f "$SOURCE_DIR/docker-compose.yml" ]]; then
        error "docker-compose.yml not found at $SOURCE_DIR"
        error "Run this script from the contabo/ deployment directory."
        exit 1
    fi

    # Deploy directory
    mkdir -p "$DEPLOY_DIR"
    cp -r "$SOURCE_DIR"/. "$DEPLOY_DIR/"
    chmod -R 755 "$DEPLOY_DIR"

    # Ensure Prometheus rules dir exists
    mkdir -p "$DEPLOY_DIR/prometheus/rules"

    # Create textfile collector directory for node_exporter custom metrics
    mkdir -p /var/lib/node_exporter/textfile_collector

    # Create Cloudflare cert directory
    mkdir -p /etc/ssl/cloudflare
    chmod 700 /etc/ssl/cloudflare

    log "Files deployed to $DEPLOY_DIR"
    warn "Remember to place Cloudflare Origin Certs in /etc/ssl/cloudflare/"
}

# =============================================================================
# Step 6: Configure Environment Variables
# =============================================================================
setup_env() {
    section "Step 6: Configure Environment"

    cat > "$DEPLOY_DIR/.env" <<ENVEOF
# Metadron Monitoring Stack — Environment Variables
# Generated by setup-contabo.sh on $(date -u '+%Y-%m-%d %H:%M:%S UTC')
# DO NOT commit this file to git — it contains secrets

GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}

# Alertmanager credentials — fill in before starting
SMTP_PASSWORD=<SMTP_APP_PASSWORD>
SLACK_WEBHOOK_CRITICAL=<SLACK_WEBHOOK_URL_CRITICAL>
SLACK_WEBHOOK_WARNINGS=<SLACK_WEBHOOK_URL_WARNINGS>
TELEGRAM_BOT_TOKEN=<TELEGRAM_BOT_TOKEN>
TELEGRAM_CHAT_ID=<TELEGRAM_CHAT_ID>
ENVEOF

    chmod 600 "$DEPLOY_DIR/.env"
    log ".env file created at $DEPLOY_DIR/.env"
    warn "Edit $DEPLOY_DIR/.env to fill in SMTP and Slack credentials!"
}

# =============================================================================
# Step 7: Deploy Monitoring Stack
# =============================================================================
deploy_stack() {
    section "Step 7: Deploy Docker Monitoring Stack"

    cd "$DEPLOY_DIR"

    # Pull all images first (faster subsequent starts)
    log "Pulling Docker images..."
    docker compose pull

    # Start all services
    log "Starting services..."
    docker compose up -d

    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    sleep 15

    # Check status
    docker compose ps
    log "Monitoring stack deployed."
}

# =============================================================================
# Step 8: Configure NGINX Reverse Proxy
# =============================================================================
configure_nginx() {
    section "Step 8: Configure NGINX"

    # Copy NGINX config
    cp "$DEPLOY_DIR/nginx/grafana.conf" /etc/nginx/sites-available/metadron-monitoring.conf

    # Check if SSL certs exist (skip if not)
    if [[ ! -f "/etc/ssl/cloudflare/monitor.metadroncapital.com.pem" ]]; then
        warn "Cloudflare Origin Cert not found at /etc/ssl/cloudflare/"
        warn "NGINX will NOT be enabled until you add the cert files."
        warn "After adding certs, run: nginx -t && systemctl reload nginx"

        # Create a temporary HTTP-only config for testing
        cat > /etc/nginx/sites-available/metadron-temp.conf <<TEMPEOF
server {
    listen 80;
    server_name ${DOMAIN_GRAFANA};
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host \$http_host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
}
server {
    listen 80;
    server_name ${DOMAIN_STATUS};
    location / {
        proxy_pass http://127.0.0.1:3001;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$http_host;
    }
}
TEMPEOF
        ln -sf /etc/nginx/sites-available/metadron-temp.conf /etc/nginx/sites-enabled/
        nginx -t && systemctl reload nginx
        warn "Temporary HTTP-only config enabled. Replace with SSL config once certs are in place."
    else
        ln -sf /etc/nginx/sites-available/metadron-monitoring.conf /etc/nginx/sites-enabled/
        # Remove default site
        rm -f /etc/nginx/sites-enabled/default
        nginx -t && systemctl reload nginx
        log "NGINX configured with SSL."
    fi
}

# =============================================================================
# Step 9: Configure UFW Firewall
# =============================================================================
configure_ufw() {
    section "Step 9: Configure UFW Firewall"

    # Reset UFW to defaults (careful — this may kick you if SSH is not in allow list)
    ufw --force reset

    # Default policies
    ufw default deny incoming
    ufw default allow outgoing

    # Allow SSH (CRITICAL — do this before enabling UFW!)
    ufw allow 22/tcp comment 'SSH'

    # Allow HTTP/HTTPS for NGINX
    ufw allow 80/tcp  comment 'HTTP (redirect to HTTPS)'
    ufw allow 443/tcp comment 'HTTPS (Grafana + Uptime Kuma)'

    # Allow WireGuard UDP
    ufw allow 51820/udp comment 'WireGuard VPN tunnel'

    # Block direct access to internal monitoring ports (Docker handles these on localhost)
    # These are bound to 127.0.0.1 in docker-compose.yml so UFW doesn't need to block them,
    # but we add explicit deny rules as defense-in-depth.
    ufw deny 9090/tcp comment 'Prometheus (internal only)'
    ufw deny 9093/tcp comment 'Alertmanager (internal only)'
    ufw deny 3000/tcp comment 'Grafana (via NGINX)'
    ufw deny 3001/tcp comment 'Uptime Kuma (via NGINX)'
    ufw deny 19999/tcp comment 'Netdata (SSH tunnel only)'
    ufw deny 9100/tcp comment 'node_exporter (internal only)'

    # Enable UFW
    ufw --force enable

    log "UFW configured."
    ufw status verbose
}

# =============================================================================
# Step 10: Configure Fail2Ban
# =============================================================================
configure_fail2ban() {
    section "Step 10: Configure Fail2Ban"

    cat > /etc/fail2ban/jail.local <<F2BEOF
[DEFAULT]
bantime  = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port    = ssh
logpath = %(sshd_log)s
backend = %(sshd_backend)s

[nginx-http-auth]
enabled = true
port    = http,https
logpath = /var/log/nginx/*error.log
F2BEOF

    systemctl enable fail2ban
    systemctl restart fail2ban
    log "Fail2Ban configured."
}

# =============================================================================
# Step 11: Enable Systemd Services & Auto-start
# =============================================================================
enable_services() {
    section "Step 11: Configure Auto-start"

    # Create a systemd service to auto-start the Docker Compose stack
    cat > /etc/systemd/system/metadron-monitoring.service <<SVCEOF
[Unit]
Description=Metadron Monitoring Stack (Docker Compose)
Requires=docker.service
After=docker.service network-online.target wg-quick@wg0.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${DEPLOY_DIR}
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
SVCEOF

    systemctl daemon-reload
    systemctl enable metadron-monitoring.service
    log "Systemd service 'metadron-monitoring' enabled (starts on boot after WireGuard)."
}

# =============================================================================
# Step 12: Print Summary
# =============================================================================
print_summary() {
    section "Setup Complete!"

    WG_PUB=""
    if command -v wg &>/dev/null; then
        WG_PUB=$(echo "$WG_PRIVATE_KEY" | wg pubkey 2>/dev/null || echo "see /etc/wireguard/")
    fi

    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  METADRON MONITORING STACK — DEPLOYMENT SUMMARY${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${BOLD}  Services:${NC}"
    echo -e "  📊 Grafana          https://${DOMAIN_GRAFANA}"
    echo -e "     └─ Username:     admin"
    echo -e "     └─ Password:     [as configured]"
    echo ""
    echo -e "  🟢 Uptime Kuma      https://${DOMAIN_STATUS}"
    echo ""
    echo -e "  📈 Prometheus       http://localhost:9090   (local only)"
    echo -e "  🔔 Alertmanager     http://localhost:9093   (local only)"
    echo -e "  🖥️  Netdata          http://localhost:19999  (SSH tunnel)"
    echo ""
    echo -e "${BOLD}  SSH Tunnel for Netdata:${NC}"
    echo -e "  ssh -L 19999:localhost:19999 root@$(curl -s4 ifconfig.me 2>/dev/null || echo '<CONTABO_IP>')"
    echo ""
    echo -e "${BOLD}  WireGuard:${NC}"
    echo -e "  Contabo WireGuard IP:  10.0.0.2"
    echo -e "  GEX44 WireGuard IP:    10.0.0.1"
    echo -e "  Contabo Public Key:    ${WG_PUB}"
    echo -e "  Status:                $(wg show wg0 2>/dev/null | head -3 || echo 'run: wg show')"
    echo ""
    echo -e "${BOLD}  Next Steps:${NC}"
    echo -e "  1. Add Contabo WG public key as [Peer] on the GEX44"
    echo -e "  2. Place Cloudflare Origin Cert files in /etc/ssl/cloudflare/"
    echo -e "  3. Re-run NGINX config:  nginx -t && systemctl reload nginx"
    echo -e "  4. Fill in SMTP/Slack credentials in $DEPLOY_DIR/alertmanager/alertmanager.yml"
    echo -e "  5. Verify tunnel:  ping 10.0.0.1"
    echo -e "  6. Configure monitors in Uptime Kuma UI"
    echo -e "  7. Import dashboards in Grafana (auto-provisioned from $DEPLOY_DIR/grafana/dashboards/)"
    echo ""
    echo -e "${BOLD}  Useful Commands:${NC}"
    echo -e "  docker compose -f $DEPLOY_DIR/docker-compose.yml ps"
    echo -e "  docker compose -f $DEPLOY_DIR/docker-compose.yml logs -f prometheus"
    echo -e "  wg show"
    echo -e "  systemctl status metadron-monitoring"
    echo -e "  docker compose -f $DEPLOY_DIR/docker-compose.yml pull && docker compose -f $DEPLOY_DIR/docker-compose.yml up -d"
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo ""
    echo -e "${BOLD}${BLUE}Metadron Monitoring Stack — Setup Script${NC}"
    echo -e "${BLUE}Contabo VPS | Ubuntu 22.04${NC}"
    echo ""

    check_root
    gather_inputs
    system_update
    install_docker
    install_wireguard
    install_nginx
    setup_directories
    setup_env
    deploy_stack
    configure_nginx
    configure_ufw
    configure_fail2ban
    enable_services
    print_summary
}

main "$@"
