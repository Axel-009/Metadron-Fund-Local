# METADRON CAPITAL — PRODUCTION DEPLOYMENT GUIDE

Complete deployment guide for the Metadron Capital AI investment platform.
Covers fresh Contabo VPS provisioning through to a fully running, monitored production stack.

---

## Table of Contents

1. [Server Provisioning (Contabo VPS)](#a-server-provisioning-contabo-vps)
2. [Network Architecture](#b-network-architecture)
3. [Prometheus Setup](#c-prometheus-setup)
4. [Grafana Setup](#d-grafana-setup)
5. [Full Stack Docker Compose](#e-full-stack-docker-compose)
6. [Service Startup Order & Dependencies](#f-service-startup-order--dependencies)
7. [Environment Variables Reference](#g-environment-variables-reference)
8. [Updating & Redeploying](#h-updating--redeploying)
9. [Troubleshooting](#i-troubleshooting)

---

## A. Server Provisioning (Contabo VPS)

### Recommended Contabo VPS Specs

| Component | Minimum | Recommended | Why |
|-----------|---------|-------------|-----|
| **CPU** | 8 vCPU | 12+ vCPU (AMD EPYC) | ML walk-forward, signal pipeline, 1,000+ securities universe |
| **RAM** | 32 GB | 64 GB | Full universe in memory, pandas/numpy DataFrames, model state, Qwen 2.5-7B inference |
| **Disk** | 400 GB NVMe | 800 GB NVMe | Repo (~3 GB), intelligence_platform (~16K files), market data cache, model artifacts, logs, Prometheus TSDB |
| **Network** | 200 Mbit/s | 400+ Mbit/s | Real-time market data feeds (OpenBB, Alpaca), LLM API calls, Grafana dashboards |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS | Long-term support, Docker-native |

**Recommended plan**: Contabo Cloud VPS XL or higher (12 vCPU, 64 GB RAM, 800 GB NVMe, ~$35-50/mo).
For GPU inference (Qwen, Air-LLM): Contabo GPU VPS with NVIDIA A100/A30 or external GPU server.

### Ubuntu 22.04 LTS Initial Setup

```bash
# 1. Update system
apt update && apt upgrade -y
apt install -y curl wget git build-essential software-properties-common

# 2. Create non-root user
adduser metadron
usermod -aG sudo metadron

# 3. SSH hardening
# Edit /etc/ssh/sshd_config:
#   PermitRootLogin no
#   PasswordAuthentication no
#   PubkeyAuthentication yes
#   AllowUsers metadron
systemctl restart sshd

# 4. Copy your SSH public key
su - metadron
mkdir -p ~/.ssh && chmod 700 ~/.ssh
# Paste your public key into ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# 5. UFW Firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp        # SSH
ufw allow 80/tcp        # HTTP (Nginx → redirect to HTTPS)
ufw allow 443/tcp       # HTTPS (Nginx → React frontend + API)
ufw allow 3000/tcp      # Grafana dashboard
ufw allow 9090/tcp      # Prometheus UI (restrict to VPN/localhost in production)
ufw allow 5000/tcp      # Express frontend (dev mode, close in production)
ufw allow 3001/tcp      # Frontend dev hot-reload (close in production)
ufw enable

# 6. Fail2Ban
apt install -y fail2ban
systemctl enable fail2ban
```

### Install Docker & Docker Compose

```bash
# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker metadron

# Docker Compose plugin
apt install -y docker-compose-plugin

# Verify
docker --version
docker compose version
```

### Install Node.js 20 LTS

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs
npm install -g pm2
node --version  # v20.x
npm --version
```

### Install Python 3.11

```bash
add-apt-repository ppa:deadsnakes/ppa -y
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
python3 --version  # 3.11.x
```

### Clone Repository & Configure Environment

```bash
su - metadron
git clone https://github.com/Axel-009/Metadron-Fund-Local.git ~/metadron
cd ~/metadron

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -e .
pip install -e ".[dev]"

# Install Node dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your actual API keys (see Section G for full reference)
nano .env

# Or decrypt from vault if you have the password:
./setup.sh
```

---

## B. Network Architecture

### ASCII Network Diagram

```
                              ┌──────────────────────────────┐
                              │      INTERNET / CLIENTS       │
                              └──────────────┬───────────────┘
                                             │
                                      ┌──────┴──────┐
                                      │   Contabo   │
                                      │  VPS / WAN  │
                                      │  Public IP  │
                                      └──────┬──────┘
                                             │
                                    ┌────────┴────────┐
                                    │  Nginx Reverse  │
                                    │     Proxy       │
                                    │  :80 → :443     │
                                    │  (SSL/TLS)      │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
          ┌─────────┴──────────┐  ┌─────────┴──────────┐  ┌─────────┴──────────┐
          │  Express + React   │  │   FastAPI Engine    │  │   Monitoring       │
          │  Frontend (SSR)    │  │      API            │  │                    │
          │  :5000             │  │   :8001             │  │  Prometheus :9090  │
          │                    │  │                     │  │  Grafana   :3000   │
          │  React SPA +       │  │  /health            │  │                    │
          │  WebSocket proxy   │  │  /api/hedge-fund/*  │  │  Scrapes all       │
          │  to Engine API     │  │  /api/stream/*      │  │  /metrics          │
          └────────────────────┘  │  /api/allocation/*  │  │  endpoints         │
                                  │  /api/chat/*        │  └────────────────────┘
                                  │  /metrics           │
                                  └─────────┬───────────┘
                                            │
                 ┌──────────────────────────┼──────────────────────────┐
                 │                          │                          │
       ┌─────────┴──────────┐   ┌──────────┴──────────┐   ┌──────────┴──────────┐
       │  MiroFish Backend  │   │  LLM Inference      │   │  Platform           │
       │  (Flask)           │   │  Bridge              │   │  Orchestrator       │
       │  :5001             │   │  :8002               │   │                     │
       └────────────────────┘   │                      │   │  Live Loop          │
                                │  Air-LLM :8003       │   │  Market Open/Close  │
       ┌────────────────────┐   │  Qwen    :7860       │   │  Hourly Tasks       │
       │  MiroFish Frontend │   └──────────────────────┘   └─────────────────────┘
       │  (Vue 3)           │
       │  :5174             │
       └────────────────────┘
```

### Port Map

| Service | Internal Port | External Port | Nginx Path | Protocol |
|---------|--------------|---------------|------------|----------|
| Express + React Frontend | 5000 | 443 | `/` | HTTPS |
| FastAPI Engine API | 8001 | 443 | `/api/engine/*` | HTTPS |
| Prometheus | 9090 | 9090 | `/prometheus` (optional) | HTTP |
| Grafana | 3000 | 3000 | `/grafana` (optional) | HTTP/HTTPS |
| MiroFish Backend (Flask) | 5001 | — | `/api/mirofish/*` | Internal |
| MiroFish Frontend (Vue) | 5174 | — | `/mirofish` | Internal |
| LLM Inference Bridge | 8002 | — | — | Internal |
| Air-LLM Model Server | 8003 | — | — | Internal |
| Qwen 2.5-7B Model Server | 7860 | — | — | Internal |
| News Engine | — | — | — | Internal |

### SSL/TLS Setup with Let's Encrypt + Certbot

```bash
# Install Certbot
apt install -y certbot python3-certbot-nginx

# Obtain certificate (replace with your domain)
certbot --nginx -d metadron.capital -d www.metadron.capital

# Auto-renewal (certbot installs a systemd timer by default)
certbot renew --dry-run

# Verify auto-renewal timer
systemctl status certbot.timer
```

### Nginx Configuration

Create `/etc/nginx/sites-available/metadron`:

```nginx
upstream express_frontend {
    server 127.0.0.1:5000;
}

upstream engine_api {
    server 127.0.0.1:8001;
}

upstream grafana {
    server 127.0.0.1:3000;
}

upstream prometheus {
    server 127.0.0.1:9090;
}

server {
    listen 80;
    server_name metadron.capital www.metadron.capital;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name metadron.capital www.metadron.capital;

    ssl_certificate /etc/letsencrypt/live/metadron.capital/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/metadron.capital/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # React frontend + Express proxy
    location / {
        proxy_pass http://express_frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # FastAPI Engine API
    location /api/engine/ {
        proxy_pass http://engine_api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # SSE streaming endpoints (long-lived connections)
    location /api/stream/ {
        proxy_pass http://engine_api/api/stream/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400s;
    }

    # Grafana (optional external access)
    location /grafana/ {
        proxy_pass http://grafana/;
        proxy_set_header Host $host;
    }

    # Prometheus (restrict to authenticated users)
    location /prometheus/ {
        auth_basic "Prometheus";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://prometheus/;
    }
}
```

```bash
ln -s /etc/nginx/sites-available/metadron /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

---

## C. Prometheus Setup

### prometheus.yml

Create `monitoring/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_timeout: 10s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: []  # Add Alertmanager if deployed

scrape_configs:
  # ─── Prometheus Self-Monitoring ────────────────────
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  # ─── Metadron Engine API (FastAPI + all engines) ───
  - job_name: "metadron-engine-api"
    metrics_path: "/metrics"
    scrape_interval: 10s
    static_configs:
      - targets: ["localhost:8001"]
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        replacement: "engine-api"

  # ─── LLM Inference Bridge ─────────────────────────
  - job_name: "metadron-llm-bridge"
    metrics_path: "/metrics"
    scrape_interval: 30s
    static_configs:
      - targets: ["localhost:8002"]
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        replacement: "llm-bridge"

  # ─── Air-LLM Model Server ─────────────────────────
  - job_name: "metadron-airllm"
    metrics_path: "/metrics"
    scrape_interval: 30s
    static_configs:
      - targets: ["localhost:8003"]
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        replacement: "airllm-server"

  # ─── Node Exporter (host metrics) ─────────────────
  - job_name: "node-exporter"
    static_configs:
      - targets: ["localhost:9100"]

  # ─── Express Frontend ─────────────────────────────
  - job_name: "metadron-frontend"
    metrics_path: "/metrics"
    scrape_interval: 30s
    static_configs:
      - targets: ["localhost:5000"]
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        replacement: "express-frontend"

# Prometheus TSDB retention
storage:
  tsdb:
    retention.time: 30d
    retention.size: 10GB
```

### Scrape Targets Summary

All Prometheus metrics are exposed via the `engine/bridges/prometheus_metrics.py` module, which registers 100+ metrics. The `/metrics` endpoint on the Engine API (`:8001`) exposes:

| Category | Metrics | Description |
|----------|---------|-------------|
| **Engine Health** | `metadron_engine_up` | Engine API liveness (1/0) |
| **API Performance** | `metadron_api_requests_total`, `metadron_api_duration_seconds` | Request count by endpoint/status, latency histogram |
| **Portfolio** | `metadron_portfolio_nav`, `metadron_portfolio_pnl_daily`, `metadron_positions_count` | NAV, daily P&L, open positions |
| **MetadronCube** | `metadron_cube_signal_score`, `metadron_cube_regime`, `metadron_cube_regime_confidence`, `metadron_cube_sleeve_weight` | Regime state, confidence, sleeve allocation |
| **Trades** | `metadron_trades_total` | Executed trades by side |
| **OpenBB Data** | `metadron_openbb_requests_total`, `metadron_openbb_errors_total` | Data request/error counts by endpoint |
| **LLM Inference** | `metadron_llm_requests_total`, `metadron_llm_duration_seconds` | LLM request count and latency by backend |
| **STRAT Engines** | `metadron_strat_engine_health` | Per-engine health (1=healthy, 0=degraded) |
| **Volatility Surface** | `metadron_vol_surface_iv`, `metadron_vol_surface_skew`, `metadron_vol_surface_term_structure` | Implied vol, skew, term structure |
| **StatArb** | `metadron_stat_arb_pairs_count`, `metadron_stat_arb_active_trades`, `metadron_stat_arb_portfolio_beta` | Pairs, active trades, portfolio beta |
| **ML Ensemble** | `metadron_ml_ensemble_vote_bullish`, `metadron_ml_ensemble_vote_bearish`, `metadron_ml_ensemble_confidence` | Ensemble voting breakdown |
| **DecisionMatrix** | `metadron_decision_matrix_gates_passed`, `metadron_decision_matrix_approval_rate` | Gate pass count, approval rate |
| **Agents** | `metadron_agents_total`, `metadron_agents_active`, `metadron_agents_by_tier`, `metadron_agents_herding_risk` | Agent fleet status |
| **Reconciliation** | `metadron_recon_positions_matched`, `metadron_recon_nav_delta` | Broker reconciliation status |
| **Futures** | `metadron_futures_positions_count`, `metadron_futures_beta_current`, `metadron_futures_margin_utilization` | Futures positions and beta |
| **TCA** | `metadron_tca_avg_total_cost_bps`, `metadron_tca_execution_quality` | Transaction cost analysis |
| **TXLOG** | `metadron_txlog_orders_total`, `metadron_txlog_fill_rate`, `metadron_txlog_avg_latency_ms` | Order execution log |
| **Fixed Income** | `metadron_fi_yield_*`, `metadron_fi_ig_oas`, `metadron_fi_hy_oas` | Yields, credit spreads |
| **Macro** | `metadron_macro_vix_current`, `metadron_macro_dxy_current` | VIX, DXY, 2s10s spread |
| **Monte Carlo** | `metadron_mc_var95`, `metadron_mc_var99`, `metadron_mc_prob_profit` | VaR, expected return |
| **ETF** | `metadron_etf_positions_count`, `metadron_etf_total_market_value` | ETF holdings |
| **PM2 Process** | `metadron_pm2_process_memory_bytes`, `metadron_pm2_process_restarts` | PM2 process health |
| **Pattern Recognition** | `metadron_pattern_recognition_patterns_detected`, `metadron_pattern_recognition_confidence` | Active patterns |

### Alert Rules

Create `monitoring/prometheus/alert_rules.yml`:

```yaml
groups:
  - name: metadron_critical
    rules:
      # Kill switch triggered
      - alert: KillSwitchActive
        expr: metadron_cube_regime{regime_name="CRASH"} == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Kill Switch ACTIVE — MetadronCube in CRASH regime"
          description: "The MetadronCube has entered CRASH regime. All positions should be de-risked to beta <= 0.35."

      # Drawdown exceeds 15%
      - alert: DrawdownExceeded
        expr: (1 - metadron_portfolio_nav / metadron_portfolio_nav offset 1d) > 0.15
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Portfolio drawdown exceeds 15%"
          description: "NAV has declined more than 15% from the previous day. Investigate immediately."

      # Daily loss exceeds 3% NAV (circuit breaker)
      - alert: DailyLossCircuitBreaker
        expr: metadron_portfolio_pnl_daily < -0.03 * metadron_portfolio_nav
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Daily loss circuit breaker triggered (>3% NAV)"
          description: "Daily P&L loss exceeds 3% of NAV. Risk gate G3 should halt new trades."

  - name: metadron_performance
    rules:
      # Scan cycle too slow (>300s)
      - alert: ScanCycleSlow
        expr: metadron_api_duration_seconds{endpoint="/api/engine/pipeline/run"} > 300
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "Signal pipeline scan cycle exceeds 300 seconds"
          description: "The full pipeline run is taking more than 5 minutes. Check for data source timeouts or CPU saturation."

      # Engine API down
      - alert: EngineAPIDown
        expr: metadron_engine_up == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Metadron Engine API is DOWN"
          description: "The FastAPI engine API is not responding. Check PM2 or Docker logs."

      # OpenBB data errors spiking
      - alert: OpenBBErrorSpike
        expr: rate(metadron_openbb_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "OpenBB data error rate is elevated"
          description: "More than 0.1 errors/sec from OpenBB. Check API keys and provider availability."

  - name: metadron_agents
    rules:
      # Agent permission blocks > 10/min
      - alert: AgentPermissionBlocksHigh
        expr: rate(metadron_agents_enforcement_events{severity="critical"}[1m]) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Agent permission blocks exceed 10/min"
          description: "Agents are being blocked at an elevated rate. Check NanoClaw permission guard."

      # Agent herding risk high
      - alert: AgentHerdingRisk
        expr: metadron_agents_herding_risk > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Agent herding risk is elevated (>0.8)"
          description: "Agents are converging on similar positions. Review agent diversity."

  - name: metadron_infrastructure
    rules:
      # PM2 process restarts
      - alert: PM2RestartSpike
        expr: increase(metadron_pm2_process_restarts[10m]) > 3
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "PM2 process {{ $labels.process }} has restarted 3+ times in 10 minutes"
          description: "Investigate logs for crash loops."

      # High memory usage
      - alert: HighMemoryUsage
        expr: metadron_pm2_process_memory_bytes > 4e9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "PM2 process {{ $labels.process }} using >4GB RAM"
          description: "Memory usage is high. Check for memory leaks."

      # HY OAS spread stress
      - alert: CreditStressElevated
        expr: metadron_fi_hy_oas > 600
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "HY OAS spread above 600bps — credit stress"
          description: "High-yield credit spreads indicate market stress. Cube should shift to STRESS/CRASH regime."

      # VaR breach
      - alert: VaRBreach
        expr: metadron_mc_var95 > 300000
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Monte Carlo VaR 95% exceeds $300K limit"
          description: "Portfolio VaR has breached the $300K limit on $20M NAV. De-risk immediately."
```

### Retention Policy

Prometheus is configured with a **30-day retention** (`retention.time: 30d`) and a **10 GB size cap** (`retention.size: 10GB`). Whichever limit is reached first triggers data pruning. For longer-term analytics, consider forwarding to a remote write endpoint (Thanos, VictoriaMetrics, or Grafana Mimir).

---

## D. Grafana Setup

### Data Source Configuration

1. Log in to Grafana at `http://<server-ip>:3000` (default: `admin` / `admin`, change on first login).
2. Navigate to **Configuration > Data Sources > Add data source**.
3. Select **Prometheus** and configure:

| Field | Value |
|-------|-------|
| Name | `Metadron-Prometheus` |
| URL | `http://prometheus:9090` (Docker) or `http://localhost:9090` (bare metal) |
| Access | Server (default) |
| Scrape interval | `15s` |

4. Click **Save & Test** — verify "Data source is working."

### Dashboard Provisioning

Create `monitoring/grafana/provisioning/dashboards/dashboards.yml`:

```yaml
apiVersion: 1

providers:
  - name: "Metadron Dashboards"
    orgId: 1
    folder: ""
    type: file
    disableDeletion: false
    editable: true
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: true
```

Create `monitoring/grafana/provisioning/datasources/datasources.yml`:

```yaml
apiVersion: 1

datasources:
  - name: Metadron-Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

### Dashboard Folder Structure

```
monitoring/grafana/dashboards/
├── Core/
│   ├── platform-overview.json         # NAV, P&L, positions, engine health
│   ├── signal-pipeline.json           # Pipeline latency, scan cycle, data errors
│   └── risk-dashboard.json            # VaR, drawdown, kill switch, leverage
├── Allocation/
│   ├── portfolio-allocation.json      # Sleeve weights, sector exposure, beta corridor
│   ├── cube-regime.json               # MetadronCube regime, confidence, gate scores
│   └── decision-matrix.json           # Gate pass rates, approval rates, Kelly sizing
├── Agents/
│   ├── agent-scorecard.json           # Agent fleet: tier distribution, accuracy, Sharpe
│   ├── agent-herding.json             # Herding risk, concentration risk, gradient alignment
│   └── nanoclaw-permissions.json      # Permission blocks, enforcement events
├── Velocity/
│   ├── macro-indicators.json          # VIX, DXY, 2s10s, credit spreads, yields
│   ├── money-velocity.json            # GMTF score, M2V, credit impulse, sector rotation
│   └── fixed-income.json              # Yield curves, OAS spreads, duration, DV01
└── Monitoring/
    ├── infrastructure.json            # PM2 processes, memory, restarts, CPU
    ├── tca-execution.json             # TCA cost decomposition, fill rate, slippage
    ├── llm-inference.json             # LLM request counts, latency by backend
    └── data-sources.json              # OpenBB request/error rates, provider health
```

### Admin Credentials

- **First login**: `admin` / `admin` — Grafana forces a password change.
- **Production**: Set `GF_SECURITY_ADMIN_PASSWORD` environment variable in docker-compose.
- **SMTP** (optional): Configure `GF_SMTP_*` env vars for alert email notifications.

---

## E. Full Stack Docker Compose

The repository currently uses **PM2** (`ecosystem.config.cjs`) for process management in bare-metal deployments. The Docker Compose configuration below provides a containerized alternative for production.

### docker-compose.yml

Create `docker-compose.yml` in the repository root:

```yaml
version: "3.9"

services:
  # ─── React + Express Frontend ──────────────────────
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    container_name: metadron-frontend
    ports:
      - "5000:5000"
    environment:
      - NODE_ENV=production
      - PORT=5000
      - ENGINE_API_URL=http://backend:8001
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    networks:
      - metadron

  # ─── FastAPI Engine API ────────────────────────────
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: metadron-backend
    ports:
      - "8001:8001"
    env_file:
      - .env
    environment:
      - ENGINE_API_PORT=8001
      - PYTHONUNBUFFERED=1
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 60s
    networks:
      - metadron

  # ─── Prometheus ────────────────────────────────────
  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: metadron-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/prometheus/alert_rules.yml:/etc/prometheus/alert_rules.yml:ro
      - prometheus-data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--storage.tsdb.retention.time=30d"
      - "--storage.tsdb.retention.size=10GB"
      - "--web.enable-lifecycle"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 5s
      retries: 3
    networks:
      - metadron

  # ─── Grafana ───────────────────────────────────────
  grafana:
    image: grafana/grafana:10.4.0
    container_name: metadron-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-metadron-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SERVER_ROOT_URL=http://localhost:3000
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
    depends_on:
      prometheus:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    networks:
      - metadron

  # ─── Redis (caching + pub-sub for SSE) ─────────────
  redis:
    image: redis:7-alpine
    container_name: metadron-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    networks:
      - metadron

  # ─── Node Exporter (host metrics) ──────────────────
  node-exporter:
    image: prom/node-exporter:v1.7.0
    container_name: metadron-node-exporter
    ports:
      - "9100:9100"
    restart: unless-stopped
    networks:
      - metadron

  # ─── Nginx Reverse Proxy ───────────────────────────
  nginx:
    image: nginx:1.25-alpine
    container_name: metadron-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./monitoring/nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - frontend
      - backend
      - grafana
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/"]
      interval: 30s
      timeout: 5s
      retries: 3
    networks:
      - metadron

volumes:
  prometheus-data:
    driver: local
  grafana-data:
    driver: local
  redis-data:
    driver: local

networks:
  metadron:
    driver: bridge
```

### Dockerfile.frontend

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 5000
CMD ["node", "dist/index.cjs"]
```

### Dockerfile.backend

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .
COPY . .
EXPOSE 8001
CMD ["python3", "-m", "uvicorn", "engine.api.server:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"]
```

### Running with Docker Compose

```bash
# Build and start all services
docker compose up -d --build

# View logs
docker compose logs -f

# Check service status
docker compose ps

# Stop all services
docker compose down

# Remove volumes (DESTRUCTIVE — deletes Prometheus/Grafana data)
docker compose down -v
```

---

## F. Service Startup Order & Dependencies

### Ordered Startup Sequence

```
1. redis           ← No dependencies. Cache/pub-sub layer.
2. prometheus      ← No dependencies. Starts collecting host metrics.
3. node-exporter   ← No dependencies. Exports host CPU/RAM/disk metrics.
4. backend         ← Depends on: redis (optional). FastAPI engine API + /metrics.
5. frontend        ← Depends on: backend (health check). Express + React SPA.
6. grafana         ← Depends on: prometheus (health check). Visualization layer.
7. nginx           ← Depends on: frontend, backend, grafana. Reverse proxy.
```

### PM2 Startup Sequence (Bare Metal)

When using PM2 instead of Docker, the `ecosystem.config.cjs` defines 14 services:

```
1. engine-api                  ← FastAPI on :8001 (core dependency for all)
2. express-frontend            ← Express + React on :5000
3. mirofish-backend            ← Flask on :5001
4. mirofish-frontend           ← Vue 3 on :5174
5. qwen-model-server           ← Qwen 2.5-7B on :7860 (GPU required)
6. news-engine                 ← Node.js news aggregator
7. live-loop                   ← Continuous trading loop (09:30-16:00 ET)
8. platform-orchestrator       ← Central orchestrator
9. llm-inference-bridge        ← Unified LLM proxy on :8002
10. airllm-model-server        ← Air-LLM on :8003 (GPU required)
11. ainewton-service            ← AI-Newton symbolic regression
12. learning-loop              ← Continuous ML learning
13. metadron-cube              ← 24/7 regime detection
14. market-open (cron 09:30)   ← Scheduled daily pipeline
15. market-close (cron 16:00)  ← Scheduled EOD reconciliation
16. hourly-tasks (cron hourly) ← Scheduled hourly refresh
```

```bash
# Start all PM2 services
pm2 start ecosystem.config.cjs

# Start production mode
pm2 start ecosystem.config.cjs --env production

# Monitor all processes
pm2 monit

# View aggregated logs
pm2 logs

# Restart a specific service
pm2 restart engine-api
```

### Health Check URLs

| Service | Health Check URL | Expected Response |
|---------|-----------------|-------------------|
| FastAPI Engine API | `http://localhost:8001/health` | `{"status": "ok", "service": "Metadron Capital API"}` |
| Express Frontend | `http://localhost:5000/` | 200 OK (HTML page) |
| Prometheus | `http://localhost:9090/-/healthy` | "Prometheus Server is Healthy" |
| Grafana | `http://localhost:3000/api/health` | `{"database": "ok"}` |
| Redis | `redis-cli ping` | `PONG` |
| MiroFish Backend | `http://localhost:5001/` | 200 OK |
| LLM Bridge | `http://localhost:8002/health` | 200 OK |

### Verification Checklist

```bash
# 1. Engine API responds
curl -s http://localhost:8001/health | python3 -m json.tool

# 2. Frontend loads
curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/

# 3. Prometheus targets are up
curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool | grep '"health"'

# 4. Grafana is accessible
curl -s http://localhost:3000/api/health

# 5. Redis is alive
redis-cli ping

# 6. Prometheus scraping engine metrics
curl -s 'http://localhost:9090/api/v1/query?query=metadron_engine_up' | python3 -m json.tool

# 7. PM2 status (bare metal)
pm2 status

# 8. Docker status (containerized)
docker compose ps
```

---

## G. Environment Variables Reference

| Variable | Service | Description | Example | Required |
|----------|---------|-------------|---------|----------|
| `ANTHROPIC_API_KEY` | Engine API, LLM Bridge, Agents | Anthropic Claude API key for all LLM inference | `sk-ant-api03-...` | **Yes** |
| `ALPACA_API_KEY` | Engine API (Alpaca Broker) | Alpaca brokerage API key | `PK...` | Yes (for live/paper trading) |
| `ALPACA_SECRET_KEY` | Engine API (Alpaca Broker) | Alpaca brokerage secret key | `...` | Yes (for live/paper trading) |
| `ALPACA_PAPER_TRADE` | Engine API (Alpaca Broker) | Enable paper trading mode | `True` | Yes |
| `OPENBB_TOKEN` | Engine API (Data Layer) | OpenBB Hub token for premium data providers | `eyJ...` | Recommended |
| `ZEP_API_KEY` | MiroFish, Agents | Zep memory graph API key | `z_...` | Optional |
| `TRADIER_API_KEY` | Engine API (Tradier Broker) | Legacy Tradier API key | `...` | No (legacy) |
| `TRADIER_ACCOUNT_ID` | Engine API (Tradier Broker) | Legacy Tradier account ID | `...` | No (legacy) |
| `TRADIER_ENVIRONMENT` | Engine API (Tradier Broker) | Tradier environment (sandbox/production) | `sandbox` | No (legacy) |
| `ENGINE_API_PORT` | Engine API | FastAPI server port | `8001` | No (default: 8001) |
| `PORT` | Express Frontend | Express server port | `5000` | No (default: 5000) |
| `NODE_ENV` | Frontend, News Engine | Node.js environment | `production` | No (default: development) |
| `FLASK_PORT` | MiroFish Backend | MiroFish Flask port | `5001` | No (default: 5001) |
| `LLM_BRIDGE_PORT` | LLM Bridge | Inference bridge port | `8002` | No (default: 8002) |
| `AIRLLM_PORT` | Air-LLM Server | Air-LLM model server port | `8003` | No (default: 8003) |
| `QWEN_MODEL_PATH` | Qwen Server | Path to Qwen 2.5-7B model weights | `Qwen/Qwen2.5-Omni-7B` | No (default provided) |
| `QWEN_SERVER_PORT` | Qwen Server | Qwen model server port | `7860` | No (default: 7860) |
| `QWEN_GPU_DEVICES` | Qwen Server | CUDA device IDs for Qwen | `0` | No (default: 0) |
| `AIRLLM_MODEL_PATH` | Air-LLM Server | Path to Air-LLM model weights | `meta-llama/Llama-3.1-70B` | No (default provided) |
| `AIRLLM_GPU_DEVICES` | Air-LLM Server | CUDA device IDs for Air-LLM | `1` | No (default: 1) |
| `ANTHROPIC_MODEL` | LLM Bridge | Default Anthropic model ID | `claude-opus-4-6` | No (default provided) |
| `LLM_BRIDGE_URL` | Learning Loop | URL for the LLM inference bridge | `http://localhost:8002` | No (default provided) |
| `METADRON_MODE` | Live Loop | Trading mode (live/paper/backtest) | `live` | No (default: paper) |
| `METADRON_CUBE_MODE` | MetadronCube Service | Cube operating mode | `continuous` | No (default: continuous) |
| `PYTHONUNBUFFERED` | All Python services | Disable stdout buffering | `1` | Yes (always set to 1) |
| `GRAFANA_ADMIN_PASSWORD` | Grafana | Grafana admin password (Docker Compose) | `your-secure-password` | No (default: metadron-admin) |

---

## H. Updating & Redeploying

### Git Pull + Rebuild Workflow

```bash
cd ~/metadron

# 1. Pull latest changes
git pull origin main

# 2. Install any new Python dependencies
source .venv/bin/activate
pip install -e .

# 3. Install any new Node dependencies
npm install

# 4. Rebuild frontend
npm run build

# 5. Restart services
# Docker:
docker compose up -d --build

# PM2:
pm2 restart all
```

### Zero-Downtime Deploy Steps

```bash
# 1. Pull changes
git pull origin main

# 2. Build new frontend bundle (no downtime — old bundle still served)
npm run build

# 3. Restart backend first (health check ensures readiness before traffic)
# Docker:
docker compose up -d --no-deps --build backend
docker compose up -d --no-deps --build frontend

# PM2:
pm2 reload engine-api        # graceful reload
pm2 reload express-frontend  # graceful reload

# 4. Verify health
curl -s http://localhost:8001/health
curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/

# 5. Reload Nginx (if config changed)
nginx -t && systemctl reload nginx
```

### Rolling Back a Bad Commit

```bash
# 1. Identify the last good commit
git log --oneline -10

# 2. Revert to last good state
git revert HEAD     # creates a new commit undoing the last one
# OR for multiple commits:
git revert HEAD~3..HEAD

# 3. Rebuild and restart
npm run build
docker compose up -d --build
# OR
pm2 restart all

# 4. Verify health
curl -s http://localhost:8001/health
```

**Avoid `git reset --hard`** in production — it rewrites history and can cause issues with other developers or CI. Prefer `git revert` which creates a clean undo commit.

---

## I. Troubleshooting

### Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Engine API not starting | Missing `.env` or bad API keys | Check `cat .env`, verify `ANTHROPIC_API_KEY` is set |
| `ModuleNotFoundError` on startup | Dependencies not installed | `pip install -e .` in the venv |
| Frontend returns 502 | Backend not ready yet | Wait for backend health check to pass, check `pm2 logs engine-api` |
| Prometheus targets DOWN | Service not exposing `/metrics` | Verify `prometheus_metrics.py` is mounted on the engine API |
| Grafana shows "No data" | Wrong data source URL | Check Prometheus data source config: `http://prometheus:9090` (Docker) or `http://localhost:9090` |
| OpenBB data errors | Invalid or missing `OPENBB_TOKEN` | Set token in `.env`, verify at https://my.openbb.co |
| PM2 restart loops | Python exceptions on startup | Check `pm2 logs <service>`, fix the Python error |
| Docker build fails on arm64 | Node native modules (better-sqlite3) | Use `--platform linux/amd64` or install build tools |
| WebSocket disconnects | Nginx proxy timeout | Set `proxy_read_timeout 86400s` in Nginx config |
| Kill switch stuck active | CRASH regime persists | Check MetadronCube regime via `/api/engine/cube/status` |

### Log Locations

**PM2 (bare metal)**:
```
logs/pm2/engine-api-out.log          # Engine API stdout
logs/pm2/engine-api-error.log        # Engine API stderr
logs/pm2/express-frontend-out.log    # Frontend stdout
logs/pm2/express-frontend-error.log  # Frontend stderr
logs/pm2/live-loop-out.log           # Trading loop
logs/pm2/llm-bridge-out.log          # LLM inference bridge
logs/pm2/mirofish-backend-out.log    # MiroFish
logs/pm2/metadron-cube-out.log       # MetadronCube regime detection
logs/pm2/learning-loop-out.log       # Continuous learning
logs/pm2/platform-orchestrator-out.log
logs/pm2/market-open-out.log
logs/pm2/market-close-out.log
```

**Docker**:
```bash
docker compose logs backend          # Engine API
docker compose logs frontend         # Express frontend
docker compose logs prometheus       # Prometheus
docker compose logs grafana          # Grafana
docker compose logs redis            # Redis
docker compose logs -f               # Follow all logs
```

### How to Check Prometheus Targets

```bash
# Via API
curl -s http://localhost:9090/api/v1/targets | python3 -m json.tool

# Via UI
# Open http://localhost:9090/targets in browser

# Check specific metric exists
curl -s 'http://localhost:9090/api/v1/query?query=metadron_engine_up'
```

### How to Reset Grafana

```bash
# Docker — remove Grafana volume and restart
docker compose down
docker volume rm metadron-fund-local_grafana-data
docker compose up -d

# Bare metal — delete Grafana SQLite DB
rm /var/lib/grafana/grafana.db
systemctl restart grafana-server
```

### Emergency Procedures

```bash
# Stop all trading immediately (PM2)
pm2 stop live-loop
pm2 stop market-open
pm2 stop market-close

# Stop all trading (Docker)
docker compose stop backend

# Verify kill switch status
curl -s http://localhost:8001/api/engine/cube/status | python3 -m json.tool

# Force paper-only mode
echo "ALPACA_PAPER_TRADE=True" >> .env
pm2 restart engine-api
```

---

*Last updated: 2026-04-10*
*Platform: Metadron Capital v0.1.0*
