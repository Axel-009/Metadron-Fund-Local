# Metadron Capital — Infrastructure & Hosting Plan

## Codebase Profile (as of 2026-04)

| Metric | Value |
|--------|-------|
| Total repo size | 3.6 GB |
| Total files | 42,253 |
| Python files | 8,963 |
| intelligence_platform (reference repos) | 2.4 GB |
| Engine (core trading logic) | 3.1 MB |
| ML dependencies | PyTorch, XGBoost, hmmlearn, PySR, FinRL (bridges) |
| Web stack | FastAPI (backend) + Vue 3 (MiroFish) + client app |
| Live loop | 1-min heartbeat, 09:30-16:00 ET primary, 24/7 monitoring |

---

## Storage Requirements

| Component | Estimated Size |
|-----------|---------------|
| Codebase + intelligence_platform | 5 GB |
| Python virtualenv + ML deps (PyTorch, OpenBB, XGBoost, PySR) | 15-20 GB |
| Node.js (MiroFish Vue, client app) | 2 GB |
| Market data cache (1,044+ securities, 1-min bars, FRED, options) | 50-100 GB |
| ML model artifacts (HMM, XGBoost ensemble, PPO, walk-forward) | 10-20 GB |
| QSTrader backtesting output + Monte Carlo | 20-30 GB |
| Database (positions, signals, learning loop) | 10-20 GB |
| OS + system overhead | 10 GB |
| Growth headroom | 50 GB |
| **Total recommended** | **250-500 GB NVMe SSD** |

---

## GPU vs CPU Analysis

### Components That DO NOT Need GPU

| Component | File | Reason |
|-----------|------|--------|
| XGBoost 4-model ensemble | engine/ml/universe_classifier.py | CPU-optimized by design |
| Pure-numpy PPO agent | engine/ml/deep_learning_engine.py | No CUDA dependency |
| HMM regime detection | engine/ml/bridges/markov_regime_bridge.py | hmmlearn is CPU-only |
| PySR symbolic regression | engine/signals/pattern_discovery_engine.py | Multi-core CPU |
| Black-Scholes/Greeks | engine/execution/options_engine.py | Pure math |
| OpenBB data ingestion | engine/data/openbb_data.py | I/O bound |
| Alpha optimizer (SLSQP) | engine/ml/alpha_optimizer.py | scipy, CPU |
| All signal engines | engine/signals/*.py | Numerical, CPU |
| LLM inference | Via Anthropic API | API call, not local |

### Components That CAN Use GPU (Optional)

| Component | File | Impact |
|-----------|------|--------|
| FinRL deep RL | engine/ml/bridges/finrl_bridge.py | Has numpy fallback |
| NVIDIA TFT | engine/ml/bridges/nvidia_tft_adapter.py | Optional bridge |
| Air-LLM local inference | intelligence_platform/Air-LLM/ | Not used in prod (Anthropic API) |

### Verdict

**CPU-first. GPU optional for future ML training.**

The architecture has pure-numpy fallbacks by design (rule #7). LLM calls go to
Anthropic's API. The 1-minute live loop is I/O + numerical computation, not
deep learning inference. A 16-core CPU with 128 GB RAM handles the full
pipeline comfortably.

---

## Hetzner Product Comparison

### Web Hosting vs VPS vs Dedicated — What's the Difference?

| Feature | Web Hosting | VPS (Cloud) | Dedicated Server |
|---------|-------------|-------------|------------------|
| Hardware | Shared | Virtual (shared physical) | Physical machine, all yours |
| CPU | Shared, throttled | vCPUs (burstable or dedicated) | Full dedicated cores |
| RAM | 1-4 GB | 4-64 GB | 64-256+ GB |
| Root access | No | Yes | Yes |
| Run Python/Docker | No | Yes | Yes |
| 24/7 sustained load | No | Throttled on shared | Yes, full speed |
| Noisy neighbors | Yes | Yes (shared) / No (dedicated vCPU) | No |
| Use case | WordPress, HTML | Dev, small apps | Production trading |
| **For Metadron?** | **Not suitable** | **Dev/paper only** | **Production** |

Web hosting = shared space for websites. Cannot run Python daemons, Docker,
or sustained workloads. Not an option for a trading platform.

### Recommended Hetzner Products

#### Production (Recommended): AX102 Dedicated Server

- **CPU:** AMD Ryzen 9 7950X3D — 16 cores / 32 threads, 5.7 GHz boost
- **RAM:** 128 GB DDR5 ECC
- **Storage:** 2x 1 TB NVMe SSD (RAID-1 for redundancy)
- **Network:** 1 Gbit/s unmetered
- **Price:** ~EUR 80-90/month
- **Location:** Germany (Falkenstein/Nuremberg) or Finland (Helsinki)

**Why this fits:**
- 16 real cores run all 7 pipeline stages in parallel
- 128 GB RAM holds full universe + ML models in memory
- NVMe handles high-frequency data writes (1-min bars, signal logs)
- Dedicated = consistent latency, no throttling during market hours
- RAID-1 protects against single disk failure

#### Budget / Development: Hetzner Cloud CCX33 or CCX43

- **CPU:** 8-16 dedicated vCPUs
- **RAM:** 32-64 GB
- **Storage:** 240-360 GB NVMe
- **Price:** ~EUR 35-70/month

Adequate for paper trading and development. May struggle under full load
(live loop + backtester + MiroFish + dashboard simultaneously).

#### Future GPU Option: Hetzner GEX44

- **GPU:** NVIDIA dedicated
- **Price:** ~EUR 200+/month
- **When:** Only if adding local FinRL training, LLM fine-tuning, or
  heavy PyTorch workloads to run on-box instead of via API

---

## Deployment Architecture

```
AX102 Dedicated Server (Hetzner)
│
├── Docker Compose
│   ├── metadron-engine        # Python: live_loop_orchestrator.py (24/7 daemon)
│   ├── metadron-api           # FastAPI backend (port 8000)
│   ├── mirofish-backend       # Flask API (port 5001)
│   ├── mirofish-frontend      # Vue 3 via Nginx (port 5174)
│   ├── client-app             # Web frontend (port 3000)
│   ├── postgres               # Market data, positions, signals
│   └── redis                  # Signal cache, pub/sub, session state
│
├── Nginx (reverse proxy + Let's Encrypt SSL)
│
├── systemd
│   ├── docker-compose watchdog (auto-restart)
│   └── market-hours scheduler (pre-market 08:00, close 16:00, overnight)
│
└── Cron / Overnight Jobs
    ├── QSTrader backtesting (walk-forward, Monte Carlo)
    ├── ML model retraining (XGBoost, HMM, PPO)
    └── Data cleanup + log rotation
```

## Latency Considerations

| Route | Latency | Impact |
|-------|---------|--------|
| Hetzner Germany → Alpaca API (US) | ~80-120ms | Irrelevant for 1-min heartbeat |
| Hetzner Germany → OpenBB/FRED | ~80-120ms | Irrelevant for data pulls |
| Hetzner Germany → Anthropic API | ~50-80ms | Irrelevant for LLM calls |
| User browser → Hetzner | ~20-80ms | Fine for dashboard |

For 1-minute cadence trading, transatlantic latency is negligible.
If moving to true sub-second HFT, consider Hetzner Ashburn (US East) or
a US colocation provider near NYSE/NASDAQ.

## Scaling Path

```
Phase 1 (Now):     AX102 dedicated, single server     ~EUR 90/mo
Phase 2 (Growth):  Add cloud VPS for backtesting       ~EUR 130/mo
Phase 3 (GPU):     Add GPU server for ML training      ~EUR 300/mo
Phase 4 (Scale):   Kubernetes cluster (Hetzner Cloud)  ~EUR 500+/mo
```
