# Metadron Capital — Pre-Deploy Overview

## What You're Deploying

A quantitative investment platform that runs 24/7 on a Hetzner GEX44 GPU server.
It ingests market data, runs AI models, generates trading signals, and executes
trades via Alpaca. A separate Contabo VPS monitors everything via Grafana.

## Two Servers

| Server | Role | Specs |
|--------|------|-------|
| **Hetzner GEX44** | Trading + AI inference | RTX 4000 SFF Ada (20GB VRAM), 64GB RAM, Rocky Linux 9 |
| **Contabo VPS** | Monitoring dashboards | 4 vCPU, 8GB RAM, Ubuntu 22.04, Docker |

They connect via WireGuard VPN (encrypted tunnel).

## Port Map — GEX44

| Port | Service | What it does |
|------|---------|-------------|
| 80 | NGINX | Redirects to HTTPS |
| 443 | NGINX | Main site + terminal (SSL) |
| 5000 | Express Frontend | React terminal UI |
| 8001 | Engine API | FastAPI — all trading logic |
| 8002 | LLM Bridge | Orchestrates AI models |
| 8003 | Air-LLM | Llama 3.1-8B (Air-LLM framework) |
| 8004 | Qwen Server | Qwen 2.5-7B model |
| 8005 | Llama Server | Llama 3.1-8B (dedicated) |
| 9100 | Node Exporter | CPU/RAM/disk metrics |
| 9113 | NGINX Exporter | Web server metrics |
| 9209 | PM2 Exporter | Process metrics |
| 9835 | GPU Exporter | NVIDIA GPU metrics |
| 19999 | Netdata | Real-time system monitoring |
| 51820 | WireGuard | Encrypted VPN to Contabo |

## Port Map — Contabo

| Port | Service | What it does |
|------|---------|-------------|
| 443 | NGINX | Grafana + Uptime Kuma (SSL) |
| 3000 | Grafana | Dashboards (behind NGINX) |
| 3001 | Uptime Kuma | Status page (behind NGINX) |
| 9090 | Prometheus | Metrics storage |
| 9093 | Alertmanager | Slack/Email alerts |
| 9115 | Blackbox | HTTP/TCP probes |
| 51820 | WireGuard | VPN to GEX44 |

## 32 Frontend Tabs

```
CORE:         VAULT, SECURITY, LIVE, WRAP, OPENBB, VELOCITY, CUBE
TRANSACTIONS: ALLOC, THINKING, RISK, MARGIN, RECON
PRODUCTS:     ETF, MACRO, FIXED INC, FUTURES
AGENTS:       AGENTS, CHAT, TECH, GRAPHIFY
ANALYSIS:     STRAT, QUANT, ARB, BACKTEST
SIMULATION:   MC SIM, SIM, ML, ML MODELS
REPORTING:    TXLOG, TCA, REPORTS, ARCHIVE
```

## Deployment Order

1. `01_GEX44_SERVER.md` — Set up the server (OS, drivers, Python, Node)
2. `02_APPLICATION.md` — Deploy the code (clone, install, configure, PM2)
3. `03_NGINX_SSL.md` — Set up web access (reverse proxy, SSL, landing page)
4. `04_MONITORING.md` — Set up monitoring (Contabo, Docker, Grafana, WireGuard)
5. `05_API_KEYS.md` — Configure API keys (Alpaca, FMP, Xiaomi, Jarvis)
6. `06_LAUNCH_CHECKLIST.md` — Final checks and first boot
