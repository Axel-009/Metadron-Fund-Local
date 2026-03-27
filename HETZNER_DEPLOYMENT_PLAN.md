# Metadron Capital — Hetzner Deployment Plan

## Resource Analysis

Based on audit of the codebase:

| Requirement | Demand |
|---|---|
| **CPU** | ML training (walk-forward, regime detection), OpenBB data ingestion, signal pipeline |
| **RAM** | 42K files, pandas/numpy dataframes, model state, universe of 1,000+ securities |
| **Storage** | Repo (3GB) + market data cache + model artifacts + logs + reports |
| **Network** | OpenBB API calls, Alpaca broker API, FRED, SEC, real-time data feeds |
| **Uptime** | Must run market hours (9:30 AM – 4 PM ET) + after-hours for reports |

## Recommended Architecture: 2-Server Split

### Server 1: Engine & Execution (Primary)

**Hetzer AX102 Dedicated Server**
- CPU: AMD Ryzen 9 7950X3D (16C/32T, 3D V-Cache for ML latency)
- RAM: 128 GB DDR5 ECC
- Storage: 2 × 1.92 TB NVMe (RAID 1)
- Network: 1 Gbit, unlimited traffic
- Location: Ashburn, VA (lowest latency to US markets)
- **Price: ~€129/mo (~$140/mo)**

**Why this one:**
- 3D V-Cache = lower latency on ML inference (walk-forward, regime models)
- 128GB RAM handles the full universe in memory
- NVMe RAID 1 = fast + redundant for model artifacts and data cache
- Ashburn = sub-1ms to NYSE/NASDAQ colocation

### Server 2: Intelligence Platform & Monitoring

**Hetzner AX42 Dedicated Server**
- CPU: AMD Ryzen 7 7700 (8C/16T)
- RAM: 64 GB DDR5
- Storage: 2 × 512 GB NVMe (RAID 1)
- Location: Nuremberg, DE (or Ashburn if preferred)
- **Price: ~€49/mo (~$53/mo)**

**Why separate:**
- `intelligence_platform/` has 30+ repos, heavy data processing
- Monitoring, reports, dashboards run continuously
- Isolates trading engine from research/analytics workloads
- If intelligence platform crashes, trading engine keeps running

### Storage Add-on (Optional)

**Hetzner Storage Box 5TB**
- For historical data archive, model versioning, report backups
- **Price: ~€10.90/mo (~$12/mo)**

### Total Monthly Cost

| Component | Cost |
|---|---|
| AX102 (Engine) | ~$140/mo |
| AX42 (Intelligence) | ~$53/mo |
| Storage Box 5TB | ~$12/mo |
| **Total** | **~$205/mo** |

---

## Alternative: Single Server (Budget Option)

**AX52 Dedicated Server**
- AMD Ryzen 9 7950X, 128GB DDR5, 2×1TB NVMe
- **Price: ~€79/mo (~$86/mo)**
- Can handle everything on one box
- Risk: single point of failure

---

## Deployment Steps

### Phase 1: Infrastructure Setup (Day 1-2)
1. Order AX102 (Ashburn) + AX42 (Ashburn or Nuremberg)
2. Install Ubuntu 22.04 LTS on both
3. Configure SSH keys, firewall (only 22, 80, 443 + broker API ports)
4. Set up WireGuard VPN between servers
5. Install Docker + docker-compose on both

### Phase 2: Platform Deployment (Day 3-5)
1. Clone Metadron-Capital repo to AX102
2. Set up Python 3.11 + virtual environment
3. Install all dependencies (OpenBB, pandas, numpy, scikit-learn, etc.)
4. Configure `.env` with API keys (OpenBB, Alpaca, FRED)
5. Clone intelligence_platform repos to AX42
6. Set up data pipelines and caching

### Phase 3: Paper Trading Validation (Day 6-14)
1. Run `run_open.py` + `run_close.py` in paper broker mode
2. Verify signal pipeline end-to-end
3. Validate ML model training and walk-forward
4. Check monitoring and report generation
5. Stress test with full universe (1,000+ securities)

### Phase 4: Live Trading (Day 15+)
1. Switch broker from paper to Alpaca
2. Set up kill switches and circuit breakers
3. Configure risk limits (max position size, max drawdown)
4. Start with small capital allocation
5. Scale up as confidence builds

### Phase 5: Frontend & Pattern Recognition (Month 2+)
1. Build React/Next.js dashboard on AX42
2. WebSocket for real-time signals and portfolio
3. Pattern identification engine (chart patterns, regime shifts)
4. Edge detection ML models
5. Alert system (Telegram, email)

---

## System Services (systemd)

```
metadron-engine.service      → run_open.py / run_close.py (scheduled)
metadron-hourly.service      → run_hourly.py (cron: every hour during market)
metadron-monitoring.service  → Live dashboard + report generation
metadron-ml.service          → Model retraining (overnight)
metadron-api.service         → FastAPI backend for frontend
```

---

## Security Checklist

- [ ] SSH key-only auth (no passwords)
- [ ] Firewall: deny all inbound except 22, 443, broker ports
- [ ] API keys in `.env` (never in git), encrypted with GPG
- [ ] Fail2ban for SSH brute force
- [ ] Automated security updates
- [ ] Daily encrypted backups to Storage Box
- [ ] Uptime monitoring (UptimeRobot or Hetzner monitoring)

---

## Scaling Path

Once profitable and stable:
- **AX162 upgrade** (AMD EPYC 9454, 256GB RAM) for larger universe
- **GPU server** (Hetzner EX line) for deep learning models
- **Load balancer** if frontend gets heavy traffic
- **Multi-region** for redundancy (Ashburn + Nuremberg)

---

*Total entry cost: ~$205/mo for a production-grade quant fund infrastructure.*
*This is nothing compared to what a real hedge fund pays for colo. We're getting institutional-grade infra at retail prices.*
