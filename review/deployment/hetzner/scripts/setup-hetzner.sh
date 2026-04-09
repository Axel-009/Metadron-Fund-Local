#!/usr/bin/env bash
# =============================================================================
# setup-hetzner.sh — Fresh Ubuntu 22.04 setup for Metadron Capital (GEX44)
#
# USAGE:
#   sudo bash setup-hetzner.sh [--repo-url <git-url>] [--branch <branch>]
#
# WHAT THIS DOES (in order):
#   1.  System update + essential packages
#   2.  Create `metadron` application user
#   3.  Install Python 3.11 + venv
#   4.  Install Node.js 20.x (via NodeSource) + PM2
#   5.  Install NGINX, UFW, WireGuard, build tools
#   6.  Clone repo to /opt/metadron
#   7.  Python: create venv, install requirements
#   8.  Node: npm install + npm run build
#   9.  NGINX: install config, create SSL dir, enable site
#   10. Install Prometheus exporters (node, nginx, pm2)
#   11. Start PM2 with ecosystem.config.cjs --env production
#   12. pm2 save && pm2 startup
#   13. Install systemd service unit for PM2
#   14. Configure UFW firewall
#   15. Configure WireGuard
#   16. Create log directories + set permissions
#   17. Print deployment summary
#
# REQUIREMENTS:
#   - Ubuntu 22.04 LTS (fresh Hetzner image)
#   - Root or sudo access
#   - Git repo access (SSH key or HTTPS token)
#   - .env.production file ready (copy from .env.production.example)
#
# ESTIMATED RUN TIME: 15–30 minutes depending on model downloads
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override with CLI flags or environment variables
# ---------------------------------------------------------------------------
REPO_URL="${REPO_URL:-git@github.com:your-org/metadron-fund.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="/opt/metadron"
APP_USER="metadron"
APP_GROUP="metadron"
PYTHON_VERSION="3.11"
NODE_VERSION="20"
DEPLOY_DIR="$(dirname "$(realpath "$0")")"   # directory this script lives in
SCRIPT_ROOT="$(realpath "$DEPLOY_DIR/../..")"  # deployment/ root

# Prometheus exporter versions
NODE_EXPORTER_VERSION="1.8.1"
NGINX_EXPORTER_VERSION="1.1.0"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
section() { echo -e "\n${GREEN}======================================================================${NC}"; \
            echo -e "${GREEN}  $*${NC}"; \
            echo -e "${GREEN}======================================================================${NC}"; }

# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-url) REPO_URL="$2"; shift 2 ;;
        --branch)   BRANCH="$2";   shift 2 ;;
        *) warn "Unknown argument: $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Must be root
# ---------------------------------------------------------------------------
[[ $EUID -eq 0 ]] || error "Run with sudo: sudo bash $0"

section "1 — System update + essential packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    build-essential \
    curl \
    wget \
    git \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common \
    apt-transport-https \
    unzip \
    jq \
    htop \
    iotop \
    net-tools \
    ufw \
    wireguard \
    wireguard-tools \
    openssl \
    fail2ban \
    logrotate

info "Base packages installed."

section "2 — Create application user: $APP_USER"
if id "$APP_USER" &>/dev/null; then
    warn "User $APP_USER already exists — skipping creation."
else
    useradd --system \
            --create-home \
            --home-dir /home/$APP_USER \
            --shell /bin/bash \
            --comment "Metadron Capital application user" \
            "$APP_USER"
    info "Created user $APP_USER."
fi

# Give metadron user ownership of app directory (created later)
mkdir -p "$APP_DIR"
chown "$APP_USER:$APP_GROUP" "$APP_DIR"

section "3 — Install Python $PYTHON_VERSION + venv"
# deadsnakes PPA provides Python 3.11 on Ubuntu 22.04
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq
apt-get install -y -qq \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    python3-pip

# Make python3.11 the default python3 via update-alternatives
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
info "Python $(python3 --version) installed."

section "4 — Install Node.js $NODE_VERSION + PM2"
# Official NodeSource setup
curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash -
apt-get install -y -qq nodejs
info "Node.js $(node --version) | npm $(npm --version)"

# Install PM2 globally — use the system npm so all users can find the binary
npm install -g pm2@latest
info "PM2 $(pm2 --version) installed globally."

section "5 — Install NGINX"
apt-get install -y -qq nginx
systemctl enable nginx
info "NGINX installed."

section "6 — Clone repository to $APP_DIR"
if [[ -d "$APP_DIR/.git" ]]; then
    warn "$APP_DIR already contains a git repo — pulling latest instead."
    sudo -u "$APP_USER" git -C "$APP_DIR" fetch origin
    sudo -u "$APP_USER" git -C "$APP_DIR" checkout "$BRANCH"
    sudo -u "$APP_USER" git -C "$APP_DIR" pull origin "$BRANCH"
else
    # Clone as the metadron user so file ownership is correct
    sudo -u "$APP_USER" git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi
info "Repository at $APP_DIR on branch $BRANCH."

section "7 — Python virtual environment + dependencies"
VENV_DIR="$APP_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
    sudo -u "$APP_USER" python${PYTHON_VERSION} -m venv "$VENV_DIR"
    info "Created venv at $VENV_DIR"
fi

# Upgrade pip inside the venv
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --quiet --upgrade pip setuptools wheel

# Install project dependencies
if [[ -f "$APP_DIR/requirements.txt" ]]; then
    info "Installing Python dependencies (this may take a while for torch/transformers)..."
    sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
    info "Python dependencies installed."
else
    warn "No requirements.txt found at $APP_DIR — skipping pip install."
fi

section "8 — Node.js: npm install + build"
cd "$APP_DIR"

info "Running npm install..."
sudo -u "$APP_USER" npm install --prefer-offline 2>&1 | tail -5

info "Building production assets (React + Express)..."
sudo -u "$APP_USER" npm run build
info "Build complete."

# Ensure marketing static site dir exists
mkdir -p /opt/metadron/marketing
chown -R "$APP_USER:$APP_GROUP" /opt/metadron/marketing
info "Marketing site directory: /opt/metadron/marketing"

section "9 — NGINX configuration"
# SSL certificate directory
mkdir -p /etc/ssl/cloudflare
chmod 750 /etc/ssl/cloudflare

info "SSL directory created at /etc/ssl/cloudflare"
warn "ACTION REQUIRED: Copy your Cloudflare Origin Certificate files:"
warn "  /etc/ssl/cloudflare/origin.pem       (certificate)"
warn "  /etc/ssl/cloudflare/origin-key.pem   (private key — chmod 600)"

# Copy NGINX site config
NGINX_CONF_SRC="$(realpath "$DEPLOY_DIR/../nginx/metadroncapital.conf")"
if [[ -f "$NGINX_CONF_SRC" ]]; then
    cp "$NGINX_CONF_SRC" /etc/nginx/sites-available/metadroncapital.conf
    ln -sf /etc/nginx/sites-available/metadroncapital.conf \
           /etc/nginx/sites-enabled/metadroncapital.conf
    # Remove default site if still present
    rm -f /etc/nginx/sites-enabled/default
    info "NGINX site config installed and enabled."
else
    warn "NGINX config not found at $NGINX_CONF_SRC — copy manually."
fi

# Add rate-limit zones and upstream definitions to nginx.conf http{} block
# (The conf file uses them; they must be in the http context.)
# Check if already patched:
if ! grep -q "limit_req_zone" /etc/nginx/nginx.conf; then
    # Insert before the first include in http block
    sed -i '/http {/a \
\    # Metadron rate limiting (added by setup-hetzner.sh)\
\    limit_req_zone $binary_remote_addr zone=api_limit:10m    rate=30r\/s;\
\    limit_req_zone $binary_remote_addr zone=general_limit:10m rate=60r\/s;' \
        /etc/nginx/nginx.conf
    info "Rate-limit zones added to /etc/nginx/nginx.conf"
fi

# Create NGINX log directory (usually exists, but make sure)
mkdir -p /var/log/nginx
nginx -t && info "NGINX config test passed."
systemctl reload nginx || systemctl start nginx

section "10 — Prometheus exporters"

# -- node_exporter --
ARCH=$(dpkg --print-architecture | sed 's/amd64/amd64/;s/arm64/arm64/')
NE_URL="https://github.com/prometheus/node_exporter/releases/download/v${NODE_EXPORTER_VERSION}/node_exporter-${NODE_EXPORTER_VERSION}.linux-${ARCH}.tar.gz"
info "Downloading node_exporter v${NODE_EXPORTER_VERSION}..."
wget -q "$NE_URL" -O /tmp/node_exporter.tar.gz
tar -xzf /tmp/node_exporter.tar.gz -C /tmp
mv /tmp/node_exporter-${NODE_EXPORTER_VERSION}.linux-${ARCH}/node_exporter /usr/local/bin/
rm -rf /tmp/node_exporter*

# Create systemd service for node_exporter
cat > /etc/systemd/system/node_exporter.service << 'EOF'
[Unit]
Description=Prometheus Node Exporter
After=network.target

[Service]
User=nobody
Group=nogroup
Type=simple
ExecStart=/usr/local/bin/node_exporter \
    --web.listen-address=10.0.0.1:9100 \
    --collector.systemd \
    --collector.processes
Restart=always
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now node_exporter
info "node_exporter running on 10.0.0.1:9100"

# -- nginx-prometheus-exporter --
NE2_URL="https://github.com/nginxinc/nginx-prometheus-exporter/releases/download/v${NGINX_EXPORTER_VERSION}/nginx-prometheus-exporter_${NGINX_EXPORTER_VERSION}_linux_amd64.tar.gz"
info "Downloading nginx-prometheus-exporter v${NGINX_EXPORTER_VERSION}..."
wget -q "$NE2_URL" -O /tmp/nginx_exporter.tar.gz
tar -xzf /tmp/nginx_exporter.tar.gz -C /tmp
mv /tmp/nginx-prometheus-exporter /usr/local/bin/
rm -rf /tmp/nginx_exporter*

cat > /etc/systemd/system/nginx-prometheus-exporter.service << 'EOF'
[Unit]
Description=NGINX Prometheus Exporter
After=nginx.service

[Service]
User=nobody
Group=nogroup
Type=simple
ExecStart=/usr/local/bin/nginx-prometheus-exporter \
    -nginx.scrape-uri=http://127.0.0.1/nginx_status \
    -web.listen-address=10.0.0.1:9113
Restart=always
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now nginx-prometheus-exporter
info "nginx-prometheus-exporter running on 10.0.0.1:9113"

# -- pm2-prometheus-exporter (npm package, run as metadron user via PM2) --
info "Installing pm2-prometheus-exporter npm package..."
npm install -g pm2-prometheus-exporter
# This is added to the PM2 ecosystem — see ecosystem.config.cjs for the process def
info "pm2-prometheus-exporter installed (add to PM2 ecosystem config)"

section "11 — Start PM2 with production ecosystem"
cd "$APP_DIR"

# Copy the production env file if not already present
if [[ ! -f "$APP_DIR/.env.production" ]]; then
    warn ".env.production not found at $APP_DIR"
    warn "Copy $APP_DIR/.env.production.example to $APP_DIR/.env.production"
    warn "and fill in all secrets before continuing."
    warn "Skipping PM2 start — run manually after creating .env.production"
else
    info "Starting PM2 with ecosystem.config.cjs --env production..."
    sudo -u "$APP_USER" pm2 start ecosystem.config.cjs --env production
    info "PM2 processes started."
fi

section "12 — pm2 save + startup"
sudo -u "$APP_USER" pm2 save
info "PM2 process list saved to ~/.pm2/dump.pm2"

# Generate startup script and install it
PM2_STARTUP=$(sudo -u "$APP_USER" pm2 startup systemd -u "$APP_USER" --hp "/home/$APP_USER" | grep "sudo env")
info "Running PM2 startup command..."
eval "$PM2_STARTUP" || warn "PM2 startup command failed — run manually: pm2 startup"

section "13 — Install systemd service for PM2"
SYSTEMD_UNIT_SRC="$(realpath "$DEPLOY_DIR/../systemd/metadron-pm2.service")"
if [[ -f "$SYSTEMD_UNIT_SRC" ]]; then
    cp "$SYSTEMD_UNIT_SRC" /etc/systemd/system/metadron-pm2.service
    systemctl daemon-reload
    systemctl enable metadron-pm2
    info "metadron-pm2.service installed and enabled."
else
    warn "systemd unit not found at $SYSTEMD_UNIT_SRC — copy manually."
fi

section "14 — UFW firewall"
UFW_SCRIPT="$(realpath "$DEPLOY_DIR/../firewall/ufw-rules.sh")"
if [[ -f "$UFW_SCRIPT" ]]; then
    bash "$UFW_SCRIPT"
else
    warn "UFW script not found — apply firewall rules manually."
fi

section "15 — WireGuard"
WG_CONF_SRC="$(realpath "$DEPLOY_DIR/../wireguard/wg0.conf")"
if [[ -f "$WG_CONF_SRC" ]]; then
    cp "$WG_CONF_SRC" /etc/wireguard/wg0.conf
    chmod 600 /etc/wireguard/wg0.conf
    warn "ACTION REQUIRED: Edit /etc/wireguard/wg0.conf:"
    warn "  - Replace GEX44_PRIVATE_KEY_PLACEHOLDER with output of: cat /etc/wireguard/gex44_private.key"
    warn "  - Replace CONTABO_VPS_PUBLIC_KEY_PLACEHOLDER with the Contabo VPS public key"
    warn "  - Replace CONTABO_VPS_PUBLIC_IP with the actual Contabo IP"
    warn "After editing, run:  sudo systemctl enable --now wg-quick@wg0"
else
    warn "WireGuard config not found — configure manually."
fi

# Generate WireGuard keys if they don't exist
if [[ ! -f /etc/wireguard/gex44_private.key ]]; then
    wg genkey | tee /etc/wireguard/gex44_private.key | wg pubkey > /etc/wireguard/gex44_public.key
    chmod 600 /etc/wireguard/gex44_private.key
    info "WireGuard keys generated."
    info "GEX44 Public Key: $(cat /etc/wireguard/gex44_public.key)"
    info "Share the public key with the Contabo VPS operator."
fi

section "16 — Log directories + logrotate"
mkdir -p /var/log/nginx
mkdir -p /var/log/metadron

# Logrotate for application logs
cat > /etc/logrotate.d/metadron << 'EOF'
/var/log/metadron/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 metadron metadron
    sharedscripts
    postrotate
        pm2 reloadLogs 2>/dev/null || true
    endscript
}
EOF

# Logrotate for NGINX
cat > /etc/logrotate.d/nginx-metadron << 'EOF'
/var/log/nginx/metadron-*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        [ -f /run/nginx.pid ] && kill -USR1 $(cat /run/nginx.pid) || true
    endscript
}
EOF

info "Log directories and logrotate configs created."

section "17 — Deployment Summary"
echo
echo "╔══════════════════════════════════════════════════════════╗"
echo "║        Metadron Capital — GEX44 Deployment Summary       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  App directory:    $APP_DIR"
echo "║  App user:         $APP_USER"
echo "║  Python venv:      $VENV_DIR"
echo "║"
echo "║  Services & Ports:"
echo "║    NGINX          :80  (→ HTTPS)  :443"
echo "║    Express app    :5000  (React terminal)"
echo "║    FastAPI engine :8001"
echo "║    LLM Bridge     :8002"
echo "║    Air-LLM        :8003"
echo "║    Qwen server    :7860"
echo "║    MiroFish back  :5001"
echo "║    MiroFish front :5174"
echo "║"
echo "║  Monitoring (WireGuard 10.0.0.1):"
echo "║    node_exporter           :9100"
echo "║    nginx-prometheus-exp    :9113"
echo "║    pm2-prometheus-exp      :9209"
echo "║"
echo "║  WireGuard tunnel: 10.0.0.1/24"
echo "║    Peer (Contabo): 10.0.0.2"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  ACTION REQUIRED (manual steps):"
echo "║  1. Copy Cloudflare Origin Cert to /etc/ssl/cloudflare/"
echo "║  2. Create .env.production from .env.production.example"
echo "║  3. Fill WireGuard placeholders in /etc/wireguard/wg0.conf"
echo "║  4. sudo systemctl enable --now wg-quick@wg0"
echo "║  5. sudo systemctl start metadron-pm2"
echo "╚══════════════════════════════════════════════════════════╝"
echo
info "Setup complete. Check 'pm2 list' and 'systemctl status nginx' to verify."
