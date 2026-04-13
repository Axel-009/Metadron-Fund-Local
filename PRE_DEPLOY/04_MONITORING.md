# Step 4 — Monitoring (Contabo + Grafana + Prometheus + Netdata)

## 4.1 Contabo VPS Setup

SSH into your Contabo VPS:
```bash
ssh root@CONTABO_IP
```

### Install Docker + Docker Compose

```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh

# Add your user to docker group
usermod -aG docker $USER

# Install Docker Compose plugin
apt install -y docker-compose-plugin

# Verify
docker --version
docker compose version
```

### Install WireGuard

```bash
apt install -y wireguard

# Generate keys
cd /etc/wireguard
umask 077
wg genkey | tee private.key | wg pubkey > public.key

# Show your public key (give this to GEX44)
cat public.key
```

Create config:
```bash
nano /etc/wireguard/wg0.conf
```

Contents:
```ini
[Interface]
Address = 10.0.0.2/24
PrivateKey = <CONTABO PRIVATE KEY>

[Peer]
# Hetzner GEX44
PublicKey = <GEX44 PUBLIC KEY from Step 1.9>
Endpoint = <GEX44_PUBLIC_IP>:51820
AllowedIPs = 10.0.0.1/32
PersistentKeepalive = 25
```

Enable:
```bash
systemctl enable --now wg-quick@wg0

# Test connection to GEX44
ping 10.0.0.1
# Should respond
```

**Now go back to GEX44** and add Contabo's public key to `/etc/wireguard/wg0.conf`:
```bash
# On GEX44:
sudo nano /etc/wireguard/wg0.conf
# Replace <CONTABO PUBLIC KEY> with the actual key
sudo systemctl restart wg-quick@wg0

# Test from GEX44
ping 10.0.0.2
```

## 4.2 Deploy Monitoring Stack (Docker Compose)

```bash
# On Contabo
mkdir -p /opt/monitoring
cd /opt/monitoring

# Copy configs from repo (or SCP from GEX44)
# You need these files from the repo's review/deployment/contabo/ directory:
#   docker-compose.yml
#   prometheus/prometheus.yml
#   prometheus/rules/metadron-alerts.yml
#   grafana/provisioning/datasources/prometheus.yml
#   grafana/provisioning/dashboards/dashboard-provider.yml
#   grafana/dashboards/*.json (5 files)
#   alertmanager/alertmanager.yml
#   blackbox/blackbox.yml

# SCP from GEX44 (easiest):
scp -r metadron@GEX44_IP:/opt/metadron/review/deployment/contabo/* /opt/monitoring/
```

### Edit secrets in Alertmanager config:
```bash
nano /opt/monitoring/alertmanager/alertmanager.yml
```

Replace these placeholders:
- `<SMTP_APP_PASSWORD>` — Gmail app password (create at myaccount.google.com → Security → App passwords)
- `<SLACK_WEBHOOK_URL_CRITICAL>` — Slack incoming webhook for critical alerts
- `<SLACK_WEBHOOK_URL_WARNINGS>` — Slack incoming webhook for warnings

### Set Grafana admin password:
```bash
nano /opt/monitoring/docker-compose.yml
# Find GF_SECURITY_ADMIN_PASSWORD and change "changeme_NOW" to a real password
```

### Start everything:
```bash
cd /opt/monitoring
docker compose up -d

# Check all containers are running
docker compose ps

# You should see 7 services: prometheus, grafana, alertmanager,
# blackbox_exporter, node_exporter, uptime-kuma, netdata
```

## 4.3 NGINX for Grafana (Contabo)

```bash
apt install -y nginx

# Copy the monitoring NGINX config
cp /opt/monitoring/nginx/grafana.conf /etc/nginx/sites-available/
ln -sf /etc/nginx/sites-available/grafana.conf /etc/nginx/sites-enabled/

# You need SSL certs for monitor.metadroncapital.com
# Follow same Cloudflare process as Step 3.2 but for monitor subdomain
mkdir -p /etc/ssl/cloudflare
# Place monitor.metadroncapital.com.pem and .key here

nginx -t
systemctl restart nginx
```

## 4.4 Verify Monitoring

```bash
# Prometheus is scraping GEX44
curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool | grep -E "health|job"

# Grafana is accessible
curl -s http://localhost:3000/api/health
# Should return: {"commit":"...","database":"ok","version":"..."}

# Alertmanager is running
curl -s http://localhost:9093/api/v2/status | head -5
```

Access Grafana:
1. Open `https://monitor.metadroncapital.com` in your browser
2. Login: admin / (the password you set)
3. You should see pre-configured dashboards for engines, GPU, PM2, system, trading

## 4.5 Grafana Dashboards

Five dashboards are auto-provisioned:
1. **Metadron Engines** — Engine health, API latency, trade counts
2. **GPU Monitoring** — VRAM usage, temperature, utilization
3. **PM2 Processes** — Per-process CPU, memory, restarts
4. **System Overview** — CPU, RAM, disk, network (from Netdata)
5. **Trading Activity** — NAV, P&L, positions, signals

These load automatically from the JSON files in `grafana/dashboards/`.

## 4.6 Test Alerts

```bash
# Trigger a test alert (temporary high CPU simulation)
# On GEX44:
stress --cpu 8 --timeout 360 &

# Wait 5 minutes, then check:
# - Alertmanager: http://localhost:9093 (on Contabo)
# - Slack channel should receive a HighCPU warning
# - Grafana Alert panel should show firing

# Stop the stress test
killall stress
```
