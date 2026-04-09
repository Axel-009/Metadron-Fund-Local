# Metadron Monitoring Stack — Contabo VPS

Production monitoring infrastructure for the Metadron trading platform, deployed on a Contabo VPS (4 vCPU, 8GB RAM, Ubuntu 22.04). Monitors the Hetzner GEX44 trading server over a WireGuard VPN tunnel.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INTERNET                                                                   │
│                                                                             │
│   Cloudflare CDN / Access                                                   │
│   ├── monitor.metadroncapital.com  ──→ Grafana                             │
│   └── status.metadroncapital.com   ──→ Uptime Kuma                         │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │ HTTPS (443)
                                        │
┌───────────────────────────────────────▼─────────────────────────────────────┐
│  CONTABO VPS (10.0.0.2)  Ubuntu 22.04                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  NGINX (reverse proxy)                                              │   │
│  │  ├── monitor.metadroncapital.com → localhost:3000 (Grafana)        │   │
│  │  └── status.metadroncapital.com  → localhost:3001 (Uptime Kuma)   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Prometheus  │  │   Grafana    │  │ Alertmanager │  │  Uptime Kuma │  │
│  │  :9090       │  │  :3000       │  │  :9093       │  │  :3001       │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘  └──────────────┘  │
│         │                 │                                                  │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Blackbox    │  │   Netdata    │  │ node_exporter│  │  WireGuard   │  │
│  │  :9115       │  │  :19999      │  │  :9100       │  │  wg0         │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────┬───────┘  │
└──────────────────────────────────────────────────────────────────┼──────────┘
                                                                    │
                                              WireGuard tunnel       │
                                              10.0.0.2 ←──────────→ 10.0.0.1
                                                                    │
┌───────────────────────────────────────────────────────────────────▼──────────┐
│  HETZNER GEX44 (10.0.0.1)  Ubuntu 22.04                                     │
│                                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ node_exporter│  │   nginx-prom │  │pm2-prometheus│  │  Engine API    │  │
│  │  :9100       │  │  exp :9113   │  │  exp :9209   │  │  :8001/metrics │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘  │
│                                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────────────┐ │
│  │ nvidia_gpu   │  │   PM2        │  │  Trading Engine + OpenBB + Express │ │
│  │  exp :9835   │  │  processes   │  │  :5000 (Frontend), :8001, :8002    │ │
│  └──────────────┘  └──────────────┘  └────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Scrape Topology

| Prometheus Job | Source | Port | Metrics |
|---|---|---|---|
| `node-gex44` | Hetzner GEX44 | 10.0.0.1:9100 | CPU, RAM, disk, network, load |
| `node-local` | Contabo VPS | localhost:9100 | Contabo system health |
| `nginx` | GEX44 NGINX | 10.0.0.1:9113 | HTTP requests, connections |
| `pm2` | GEX44 PM2 | 10.0.0.1:9209 | Process CPU, memory, restarts |
| `engine-api` | Trading Engine | 10.0.0.1:8001 | NAV, PnL, signals, latency |
| `gpu` | NVIDIA GPU | 10.0.0.1:9835 | Utilization, VRAM, temp, power |
| `blackbox-http` | External | localhost:9115 | Website uptime, SSL cert expiry |
| `blackbox-tcp` | WireGuard | localhost:9115 | Port reachability |
| `prometheus` | Self | localhost:9090 | Prometheus internals |

---

## Prerequisites

### On the Contabo VPS (before running setup)
- Ubuntu 22.04 LTS, fresh install
- Root or sudo access
- Open ports: 22 (SSH), 80, 443, 51820/UDP

### On the Hetzner GEX44 (already deployed)
- WireGuard installed and running (`wg-quick@wg0` active at 10.0.0.1)
- All exporters running:
  - `node_exporter` on :9100
  - `nginx-prometheus-exporter` on :9113
  - `pm2-prometheus-exporter` on :9209
  - Trading engine serving `/api/engine/metrics` on :8001
  - `nvidia_gpu_exporter` on :9835 (if GPU present)

### External services
- Cloudflare account managing `metadroncapital.com` DNS
- Cloudflare Origin Certificates for `monitor.` and `status.` subdomains
- (Optional) Slack workspace with Incoming Webhooks configured
- (Optional) SMTP relay (e.g., Gmail App Password, SendGrid)

---

## Step-by-Step Setup

### 1. Copy deployment files to Contabo

```bash
# From your local machine
scp -r ./contabo/ root@<CONTABO_IP>:/opt/metadron-monitoring/
```

### 2. Generate WireGuard keys on the GEX44 (if not already done)

```bash
# On the GEX44
wg genkey | tee /etc/wireguard/gex44-private.key | wg pubkey > /etc/wireguard/gex44-public.key
cat /etc/wireguard/gex44-public.key   # You'll need this for the Contabo peer config
```

### 3. Run the setup script on Contabo

```bash
ssh root@<CONTABO_IP>
export GEX44_PUBLIC_IP="<your-gex44-public-ip>"
export GEX44_WG_PUBLIC_KEY="<gex44-wg-public-key>"
export GRAFANA_ADMIN_PASSWORD="<your-secure-password>"

bash /opt/metadron-monitoring/scripts/setup-contabo.sh
```

The script will:
- Install Docker, WireGuard, NGINX
- Generate Contabo WireGuard keys
- Deploy all containers
- Configure UFW firewall
- Print a summary with next steps

### 4. Configure the GEX44's WireGuard peer

Add the Contabo VPS as a peer on the GEX44's `/etc/wireguard/wg0.conf`:

```ini
[Peer]
# Contabo Monitoring VPS
PublicKey = <CONTABO_WG_PUBLIC_KEY>   # Output from setup script
AllowedIPs = 10.0.0.2/32
PersistentKeepalive = 25
```

Then reload WireGuard on the GEX44:
```bash
wg-quick down wg0 && wg-quick up wg0
# Or: wg addconf wg0 <(echo "[Peer]...")
```

### 5. Verify WireGuard tunnel

```bash
# On Contabo
wg show
ping 10.0.0.1   # Should reach the GEX44

# Check Prometheus can scrape the GEX44
curl http://10.0.0.1:9100/metrics | head -5
curl http://10.0.0.1:9113/metrics | head -5
curl http://10.0.0.1:8001/api/engine/metrics | head -5
```

### 6. Install Cloudflare Origin Certificates

In the Cloudflare Dashboard:
1. Go to `SSL/TLS → Origin Server`
2. Create a certificate for `*.metadroncapital.com` and `metadroncapital.com`
3. Copy the certificate and private key

```bash
# On Contabo
nano /etc/ssl/cloudflare/monitor.metadroncapital.com.pem   # Paste cert
nano /etc/ssl/cloudflare/monitor.metadroncapital.com.key   # Paste key
nano /etc/ssl/cloudflare/metadroncapital.com-wildcard.pem  # Wildcard cert (for status.)
nano /etc/ssl/cloudflare/metadroncapital.com-wildcard.key  # Wildcard key
chmod 640 /etc/ssl/cloudflare/*.key

# Enable the production NGINX config
ln -sf /etc/nginx/sites-available/metadron-monitoring.conf /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/metadron-temp.conf
nginx -t && systemctl reload nginx
```

### 7. Configure alerting credentials

Edit the Alertmanager config:
```bash
nano /opt/metadron-monitoring/alertmanager/alertmanager.yml
```

Replace all `<PLACEHOLDER>` values:
- `<SMTP_APP_PASSWORD>` — Gmail App Password or SMTP relay secret
- `<SLACK_WEBHOOK_URL_CRITICAL>` — Slack Incoming Webhook for #alerts-critical
- `<SLACK_WEBHOOK_URL_WARNINGS>` — Slack Incoming Webhook for #alerts-warning
- `<TELEGRAM_BOT_TOKEN>` and `<TELEGRAM_CHAT_ID>` — Telegram bot credentials

Reload Alertmanager:
```bash
docker compose -f /opt/metadron-monitoring/docker-compose.yml restart alertmanager
```

---

## Accessing the Services

| Service | URL | Notes |
|---|---|---|
| Grafana | https://monitor.metadroncapital.com | Admin credentials set during setup |
| Uptime Kuma | https://status.metadroncapital.com | First-run creates admin account |
| Prometheus | http://localhost:9090 | SSH tunnel required |
| Alertmanager | http://localhost:9093 | SSH tunnel required |
| Netdata | http://localhost:19999 | SSH tunnel required |

### SSH Tunnels for Internal Services

```bash
# Prometheus + Alertmanager + Netdata in one command
ssh -L 9090:localhost:9090 \
    -L 9093:localhost:9093 \
    -L 19999:localhost:19999 \
    root@<CONTABO_IP>

# Then browse to:
# http://localhost:9090   — Prometheus
# http://localhost:9093   — Alertmanager
# http://localhost:19999  — Netdata
```

---

## Grafana Dashboards

Four dashboards are auto-provisioned at startup from `/opt/metadron-monitoring/grafana/dashboards/`:

| Dashboard | UID | Description |
|---|---|---|
| Metadron — System Health (GEX44) | `metadron-system` | CPU, RAM, disk, network, load |
| Metadron — Trading Dashboard | `metadron-trading` | NAV, PnL, signals, regime, trades, latency |
| Metadron — PM2 Processes | `metadron-pm2` | Per-process memory, CPU, restarts, uptime |
| Metadron — GPU Metrics | `metadron-gpu` | Utilization, VRAM, temperature, power draw |

Dashboards are placed in the **Metadron** folder in the Grafana UI.

---

## Alert Rules

All alert rules are defined in `prometheus/rules/metadron-alerts.yml`.

| Alert | Severity | Condition | Duration |
|---|---|---|---|
| EngineAPIDown | critical | Engine API unreachable | 1m |
| ExpressFrontendDown | critical | PM2 Express process down | 1m |
| NginxDown | critical | NGINX exporter unreachable | 1m |
| WebsiteDown | critical | HTTP probe failing | 2m |
| HighMemory | critical | RAM available < 10% | 2m |
| GPUTempHigh | critical | GPU temp > 85°C | 2m |
| HighCPU | warning | CPU > 85% | 5m |
| DiskFull | critical | Disk free < 10% | 5m |
| PM2ProcessCrashing | warning | > 5 restarts in 5m | immediate |
| HighAPILatency | warning | p95 latency > 5s | 3m |
| GPUMemoryHigh | warning | VRAM > 90% | 5m |
| SSLCertExpiry | warning | Cert expires in < 7 days | immediate |
| CubeSignalStale | warning | Signal not updated for 10m | immediate |
| OpenBBErrors | warning | > 0.1 errors/s | immediate |
| NoTradesExecuted | info | 0 trades in 1h during market hours | immediate |

### Adding a New Alert

1. Edit `prometheus/rules/metadron-alerts.yml`
2. Add your rule to an existing group or create a new group
3. Trigger a hot-reload (no restart needed):

```bash
curl -X POST http://localhost:9090/-/reload
# Or:
docker compose -f /opt/metadron-monitoring/docker-compose.yml \
    exec prometheus sh -c 'kill -HUP 1'
```

4. Verify the rule was loaded:
```bash
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[].name'
```

### Modifying Alert Routing

Edit `alertmanager/alertmanager.yml` and reload:
```bash
curl -X POST http://localhost:9093/-/reload
```

---

## WireGuard Setup Reference

### Topology
```
Contabo VPS  ←────── WireGuard Tunnel ──────→  Hetzner GEX44
10.0.0.2                51820/UDP               10.0.0.1
```

### Key commands

```bash
# Check tunnel status
wg show

# Bring tunnel up/down
wg-quick up wg0
wg-quick down wg0

# Check if GEX44 is reachable
ping 10.0.0.1
curl http://10.0.0.1:9100/metrics | head

# View WireGuard logs
journalctl -u wg-quick@wg0 -f

# Regenerate keys (if compromised)
wg genkey | tee /etc/wireguard/new-private.key | wg pubkey > /etc/wireguard/new-public.key
```

### GEX44 peer config (add to GEX44's wg0.conf)

```ini
[Peer]
# Contabo Monitoring VPS
PublicKey = <CONTABO_WG_PUBLIC_KEY>
AllowedIPs = 10.0.0.2/32
PersistentKeepalive = 25
```

---

## Maintenance Procedures

### Updating All Images

```bash
cd /opt/metadron-monitoring
docker compose pull
docker compose up -d
docker image prune -f   # Remove old images
```

### Updating a Single Service

```bash
docker compose pull prometheus
docker compose up -d prometheus
```

### Reloading Configs Without Restart

```bash
# Prometheus config/rules hot-reload
curl -X POST http://localhost:9090/-/reload

# Alertmanager config hot-reload
curl -X POST http://localhost:9093/-/reload

# NGINX config reload
nginx -t && systemctl reload nginx
```

### Backing Up Persistent Data

```bash
# Prometheus TSDB
docker compose exec prometheus sh -c \
    "promtool tsdb create-blocks-from openmetrics /tmp/snapshot"
# Or snapshot via API:
curl -X POST http://localhost:9090/api/v1/admin/tsdb/snapshot
# Results in /prometheus/snapshots/

# Grafana data (dashboards, users, annotations)
docker run --rm \
    -v monitoring_grafana-data:/data \
    -v $(pwd)/backups:/backup \
    alpine tar czf /backup/grafana-$(date +%Y%m%d).tar.gz -C /data .

# All Docker volumes
for vol in prometheus-data grafana-data alertmanager-data uptime-kuma-data; do
    docker run --rm \
        -v "monitoring_${vol}:/data" \
        -v "$(pwd)/backups:/backup" \
        alpine tar czf "/backup/${vol}-$(date +%Y%m%d).tar.gz" -C /data .
done
```

### Restoring from Backup

```bash
# Stop services first
docker compose down

# Restore a volume
docker run --rm \
    -v monitoring_grafana-data:/data \
    -v $(pwd)/backups:/backup \
    alpine sh -c "cd /data && tar xzf /backup/grafana-20240101.tar.gz"

docker compose up -d
```

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f prometheus
docker compose logs -f grafana
docker compose logs -f alertmanager

# Check for scrape errors in Prometheus
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'
```

### Checking Prometheus Targets

```bash
# All targets and their health
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, instance: .labels.instance, health: .health, lastError: .lastError}'

# Fired alerts
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | {alert: .labels.alertname, state: .state}'
```

---

## Uptime Kuma Configuration

After first login at https://status.metadroncapital.com:

1. Create monitors for:
   - **Engine API**: HTTP `http://10.0.0.1:8001/api/engine/metrics` (via WireGuard)
   - **Trading Frontend**: HTTP `https://metadroncapital.com`
   - **Terminal**: HTTP `https://metadroncapital.com/terminal/`
   - **PM2 Express**: TCP `10.0.0.1:5000`
   - **WebSocket API**: TCP `10.0.0.1:8002`

2. Configure notification channels (Slack, Telegram, Email) in Settings → Notifications

3. Create a public status page if desired

---

## Netdata Notes

Netdata provides per-second metrics with a 1-day retention by default. Access via SSH tunnel only:

```bash
ssh -L 19999:localhost:19999 root@<CONTABO_IP>
# Open: http://localhost:19999
```

To claim the node to Netdata Cloud (optional):
```bash
docker compose exec netdata netdata-claim.sh \
    -token=<CLAIM_TOKEN> \
    -rooms=<ROOM_IDS> \
    -url=https://app.netdata.cloud
```

---

## Security Notes

- All monitoring ports (9090, 9093, 9100, 9115, 19999) are bound to `127.0.0.1` — not publicly accessible
- UFW blocks direct access to these ports as defense-in-depth
- Grafana is behind NGINX with rate-limited `/login` endpoint
- Cloudflare Access can be layered on top for zero-trust SSO (recommended)
- WireGuard uses only the GEX44's specific IP (`10.0.0.1/32`) in AllowedIPs — no full VPN routing
- Fail2Ban is configured to block SSH and NGINX brute force
- Rotate WireGuard keys quarterly or after any suspected compromise

---

## Troubleshooting

### WireGuard tunnel not working

```bash
# Check WireGuard status
wg show
journalctl -u wg-quick@wg0 --no-pager -n 50

# Verify UDP 51820 is reachable on GEX44 (from your local machine)
nc -u -zv <GEX44_PUBLIC_IP> 51820

# Check AllowedIPs on both sides match
```

### Prometheus can't scrape GEX44 targets

```bash
# From Contabo, test direct connectivity
curl -v http://10.0.0.1:9100/metrics   # node_exporter
curl -v http://10.0.0.1:9113/metrics   # nginx exporter
curl -v http://10.0.0.1:8001/api/engine/metrics  # trading engine

# Check Prometheus target status
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'
```

### Grafana dashboards show "No data"

1. Verify Prometheus datasource is working: Grafana → Configuration → Data Sources → Test
2. Check the time range (default: last 6h) — data may not exist yet if freshly deployed
3. Verify PromQL queries in Explore: paste the expr from the dashboard panel
4. Check Prometheus has data: `http://localhost:9090/graph?g0.expr=up`

### Alerts not firing

```bash
# Check rule evaluation
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[] | {name: .name, state: .state, health: .health}'

# Check Alertmanager received alerts
curl -s http://localhost:9093/api/v2/alerts | jq '.'

# Force a test alert
curl -X POST http://localhost:9093/api/v2/alerts \
    -H "Content-Type: application/json" \
    -d '[{"labels":{"alertname":"TestAlert","severity":"warning"},"annotations":{"summary":"Test alert from curl"}}]'
```

### Docker containers failing to start

```bash
docker compose logs prometheus   # Check for config errors
docker compose logs grafana      # Check for permission errors

# Validate Prometheus config
docker run --rm \
    -v /opt/metadron-monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro \
    prom/prometheus:latest promtool check config /etc/prometheus/prometheus.yml

# Validate alert rules
docker run --rm \
    -v /opt/metadron-monitoring/prometheus/rules:/rules:ro \
    prom/prometheus:latest promtool check rules /rules/metadron-alerts.yml
```
