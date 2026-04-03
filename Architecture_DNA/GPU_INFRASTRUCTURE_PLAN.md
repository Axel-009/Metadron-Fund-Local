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
| Web stack | FastAPI (backend) + Express.js (proxy) + React (client) + Vue 3 (MiroFish) |
| Live loop | 3-min heartbeat intraday, 30-min after-hours, 1-hr overnight |

---

## Actual Runtime Profile

| Layer | Cadence | Latency | Nature |
|-------|---------|---------|--------|
| L1 Data Ingestion | 3 min | 100-500ms | I/O bound (OpenBB API) |
| L2 Signal Generation | 3 min | 500ms-2s | CPU (numerical) |
| L3 Intelligence/ML | 5 min | 2-5s | CPU (XGBoost, SLSQP, HMM) |
| L4 Decision | On signal delta | 100-200ms | CPU (vote ensemble) |
| L5 Execution | On approval | 50-200ms | I/O (Alpaca API) |
| L6 Learning | Every heartbeat | 100-500ms | CPU (gradient, feedback) |
| L7 Monitoring | 5 min | 100-200ms | CPU + I/O |
| FastAPI SSE | 5-10s push | <100ms | I/O |
| Express Proxy | On-demand | <50ms overhead | I/O |
| React Client | 5-30s polling | <100ms | Browser |

**Key finding: This is a 3-minute batch system, not sub-second HFT.**
No tick-by-tick streams. No WebSocket data feeds. WonderTrader/exchange-core
concepts are present but operate on OHLCV bars, not live ticks.

---

## Storage Requirements

| Component | Estimated Size |
|-----------|---------------|
| Codebase + intelligence_platform | 5 GB |
| Python virtualenv + ML deps (PyTorch, OpenBB, XGBoost, PySR) | 15-20 GB |
| Node.js (Express proxy, React client, MiroFish Vue) | 2 GB |
| Market data cache (1,044+ securities, 1-min bars, FRED, options) | 50-100 GB |
| Mimo v2 model weights (local LLM, if self-hosted) | 20-80 GB |
| ML model artifacts (HMM, XGBoost ensemble, PPO, walk-forward) | 10-20 GB |
| QSTrader backtesting output + Monte Carlo | 20-30 GB |
| Database (positions, signals, learning loop history) | 10-20 GB |
| OS + system overhead | 10 GB |
| Growth headroom | 50 GB |
| **Total recommended** | **500 GB - 1 TB NVMe SSD** |

Note: Storage jumps to 500GB+ if hosting Mimo v2 locally. If Mimo v2 is
API-based (like Anthropic), 250-500 GB is sufficient.

---

## GPU vs CPU — Revised for Mimo v2

### The Mimo v2 Question Changes Everything

If Mimo v2 is a **self-hosted LLM** replacing Anthropic API calls:
- Model weights: 20-80 GB VRAM depending on parameter count
- Inference: GPU-bound (transformer attention is matrix multiply)
- **GPU becomes mandatory** for acceptable latency

If Mimo v2 is an **API service** (like Anthropic):
- Network I/O only, no local compute
- **GPU remains optional**

### Decision Matrix

| Mimo v2 Deployment | GPU Needed? | Recommended Server |
|--------------------|-------------|-------------------|
| API-based (remote) | No | AX102 (CPU, EUR ~90/mo) |
| Local, 7B params | Yes, 8 GB VRAM | GEX44 entry GPU (EUR ~200/mo) |
| Local, 13-30B params | Yes, 24 GB VRAM | GEX44 with A10/L40 (EUR ~250+/mo) |
| Local, 70B+ params | Yes, 48-80 GB VRAM | Multi-GPU or cloud (EUR ~500+/mo) |

---

## Language Performance — Task Allocation Strategy

### Current Stack

```
Python   → Engine (live_loop_orchestrator, all signals, ML, execution)
Python   → FastAPI backend (port 8001)
TypeScript → Express.js proxy server (port 5000)
TypeScript → React client (Vite, browser)
Python   → Flask (MiroFish backend, port 5001)
TypeScript → Vue 3 (MiroFish frontend, port 5174)
```

### Where Language Performance Actually Matters

Your engine runs on a **3-minute heartbeat**. Each phase completes in
milliseconds to low seconds. Python is not the bottleneck — OpenBB API
calls and network I/O are. Rewriting to Rust/Go would save microseconds
on a 180-second cycle. Not worth it.

**However**, there are specific components where language choice matters
for the NEXT evolution of the platform:

| Component | Current | Bottleneck | Recommendation |
|-----------|---------|------------|----------------|
| **Live loop orchestrator** | Python | Not bottlenecked (3-min cycle) | Keep Python |
| **Signal engines** | Python (numpy/scipy) | CPU numerical — already fast via numpy C extensions | Keep Python |
| **ML models** | Python (XGBoost, hmmlearn) | Already C/C++ under the hood | Keep Python |
| **Alpha optimizer** | Python (scipy SLSQP) | Already Fortran under the hood | Keep Python |
| **FastAPI backend** | Python | <100ms response, fine | Keep Python |
| **Express proxy** | TypeScript | <50ms overhead, fine | Keep TypeScript |
| **React client** | TypeScript | Browser, irrelevant | Keep TypeScript |
| **Order matching** | Python (exchange_core) | Simulated, not real matching | Keep Python (or Rust if going real HFT) |
| **Data ingestion** | Python | I/O bound (API calls) | Keep Python |
| **WebSocket/SSE server** | Python SSE (stub) | Could be bottleneck at scale | Consider Go/Rust for high-fan-out |

### What Would Actually Benefit from Rust/Go

Only if you evolve toward real-time:

| Future Component | Why Faster Language Helps | Language |
|-----------------|--------------------------|----------|
| **Tick-by-tick data ingestion** | Millions of messages/sec | Rust |
| **Real order matching engine** | Microsecond latency matters | Rust (or keep Java exchange-core) |
| **WebSocket fan-out** (1000+ clients) | Concurrent connections | Go |
| **Time-series DB writes** | High-throughput append | Rust |

### Practical Recommendation

```
KEEP PYTHON for:
  ├── Engine (live_loop_orchestrator.py) — 3-min batch, not bottlenecked
  ├── All signal engines — numpy/scipy already call C/Fortran
  ├── ML pipeline — XGBoost/hmmlearn are C++ underneath
  ├── FastAPI backend — async, <100ms, adequate
  └── MiroFish backend — Flask, low traffic

KEEP TYPESCRIPT for:
  ├── Express proxy — lightweight, adequate
  ├── React client — browser rendering
  └── MiroFish Vue frontend — browser rendering

CONSIDER ADDING (future, only if needed):
  ├── Go — High-concurrency API gateway / WebSocket server
  │         (if you need 1000+ simultaneous dashboard connections)
  └── Rust — Real tick-by-tick processing / order matching
              (only if moving to true sub-second HFT)
```

**Bottom line: Python's "slowness" is a myth for your use case.**
Your hot path is 3-minute batches where 95% of wall-clock time is
waiting for API responses. numpy/scipy/XGBoost are C/Fortran underneath.
Rewriting to Rust saves you 50ms on a 180,000ms cycle.

---

## PM2 Process Management — 24/7 Architecture

### Why PM2 (Not Docker for Process Management)

PM2 is purpose-built for keeping Node.js + Python processes alive.
It handles restart-on-crash, log rotation, clustering, and monitoring
out of the box. Docker handles isolation. Use both.

### PM2 Ecosystem File

```javascript
// ecosystem.config.js
module.exports = {
  apps: [
    // === CORE ENGINE (Python, 24/7) ===
    {
      name: "metadron-engine",
      script: "python3",
      args: "-u engine/live_loop_orchestrator.py",
      cwd: "/opt/metadron",
      interpreter: "none",
      autorestart: true,
      max_restarts: 50,
      restart_delay: 5000,        // 5s backoff
      max_memory_restart: "8G",   // restart if memory leak
      env: {
        PYTHONUNBUFFERED: "1",
        BROKER_TYPE: "paper",
        OBB_TOKEN: "...",
      },
      // Self-healing: restart on crash, PM2 watches process
    },

    // === FASTAPI BACKEND (Python, port 8001) ===
    {
      name: "metadron-api",
      script: "uvicorn",
      args: "app.backend.main:app --host 0.0.0.0 --port 8001 --workers 2",
      cwd: "/opt/metadron",
      interpreter: "none",
      autorestart: true,
      max_memory_restart: "2G",
    },

    // === EXPRESS PROXY (Node.js, port 5000) ===
    {
      name: "metadron-proxy",
      script: "server/index.ts",
      cwd: "/opt/metadron",
      interpreter: "npx",
      interpreter_args: "tsx",
      autorestart: true,
      env: {
        PORT: 5000,
        ENGINE_API_PORT: 8001,
      },
    },

    // === MIROFISH BACKEND (Python/Flask, port 5001) ===
    {
      name: "mirofish-backend",
      script: "python3",
      args: "mirofish/backend/run.py",
      cwd: "/opt/metadron",
      interpreter: "none",
      autorestart: true,
    },

    // === MIROFISH FRONTEND (Vue/Vite, port 5174) ===
    // In production: serve built static files via Nginx instead
    // Only use PM2 for dev mode:
    {
      name: "mirofish-frontend",
      script: "npm",
      args: "run dev",
      cwd: "/opt/metadron/mirofish/frontend",
      autorestart: true,
    },

    // === LEARNING LOOP (overnight ML retraining) ===
    {
      name: "metadron-learner",
      script: "python3",
      args: "-u engine/monitoring/learning_loop.py",
      cwd: "/opt/metadron",
      interpreter: "none",
      cron_restart: "0 20 * * 1-5",  // Restart nightly at 20:00 ET weekdays
      autorestart: false,             // One-shot, cron triggers next run
    },
  ],
};
```

### PM2 Commands

```bash
# Start all services
pm2 start ecosystem.config.js

# Monitor real-time
pm2 monit

# View logs (all or specific)
pm2 logs
pm2 logs metadron-engine --lines 100

# Restart on deploy
pm2 reload all --update-env

# Save process list (survives reboot)
pm2 save
pm2 startup  # generates systemd service

# Health check
pm2 status
```

### Self-Healing Architecture

```
PM2 Watchdog
├── metadron-engine     → autorestart: true, max_restarts: 50
│   ├── Crash → restart in 5s (restart_delay)
│   ├── Memory leak → restart at 8 GB (max_memory_restart)
│   └── Heartbeat continues from last state (deque persists in Redis)
│
├── metadron-api        → autorestart: true, 2 uvicorn workers
│   └── Worker crash → uvicorn respawns, PM2 catches full exit
│
├── metadron-proxy      → autorestart: true
│   └── Express crash → PM2 restarts, Nginx buffers requests
│
└── metadron-learner    → cron_restart nightly
    └── Walk-forward retrain → save models → engine hot-reloads
```

---

## Latency Analysis — Hetzner Location vs Mimo v2

### API Call Latency Map

| Route | Germany (Falkenstein) | Finland (Helsinki) | US East (Ashburn) |
|-------|----------------------|--------------------|--------------------|
| → Alpaca API (US) | 80-120ms | 90-130ms | 5-15ms |
| → OpenBB/FRED (US) | 80-120ms | 90-130ms | 10-30ms |
| → Anthropic API (US) | 50-80ms | 60-90ms | 5-15ms |
| → Mimo v2 (if US-hosted) | 50-80ms | 60-90ms | 5-15ms |
| → Mimo v2 (if local) | <1ms | <1ms | <1ms |
| User browser (EU) | 5-20ms | 10-30ms | 80-120ms |
| User browser (US) | 80-120ms | 90-130ms | 5-20ms |

### Latency Impact on Your System

Your engine heartbeat is **180 seconds (3 minutes)**. Even from Germany:
- Worst-case API round-trip: ~120ms
- Number of API calls per heartbeat: ~5-20
- Total API latency per cycle: ~600ms-2.4s
- **As % of cycle: 0.3% - 1.3%** — completely irrelevant

### When Latency WOULD Matter

Only if you evolve to:
- Real tick-by-tick execution (sub-second) → need US East colo
- Local Mimo v2 inference in the decision loop → local is always fastest
- WebSocket streaming to 100+ clients → location near users matters

### Recommendation

| Your Users | Mimo v2 | Best Location |
|------------|---------|---------------|
| EU-based | API (remote) | **Hetzner Germany** (cheapest, closest to you) |
| EU-based | Local (self-hosted) | **Hetzner Germany** (GPU server) |
| US-based | API (remote) | **Hetzner Ashburn** or US provider |
| Global | Either | Germany (cheapest) — 120ms to US is fine for 3-min cycles |

---

## Hetzner Product Comparison

### Web Hosting vs VPS vs Dedicated

| Feature | Web Hosting | VPS (Cloud) | Dedicated Server |
|---------|-------------|-------------|------------------|
| Hardware | Shared | Virtual (shared physical) | Physical machine, all yours |
| CPU | Shared, throttled | vCPUs (burstable or dedicated) | Full dedicated cores |
| RAM | 1-4 GB | 4-64 GB | 64-256+ GB |
| Root access | No | Yes | Yes |
| Run Python/Docker/PM2 | No | Yes | Yes |
| 24/7 sustained load | No | Throttled on shared | Yes, full speed |
| Noisy neighbors | Yes | Yes (shared) / No (dedicated vCPU) | No |
| Use case | WordPress, HTML | Dev, small apps | Production trading |
| **For Metadron?** | **Not suitable** | **Dev/paper only** | **Production** |

### Recommended Products

#### Without GPU: AX102 (~EUR 80-90/month)

- AMD Ryzen 9 7950X3D — 16 cores / 32 threads
- 128 GB DDR5 ECC RAM
- 2x 1 TB NVMe SSD (RAID-1)
- **Best for:** Mimo v2 via API, full engine + web stack + PM2

#### With GPU: GEX44 (~EUR 200+/month)

- Dedicated CPU + NVIDIA GPU (A10/L40 tier)
- 64-128 GB RAM
- NVMe storage
- **Best for:** Self-hosted Mimo v2, local FinRL training

---

## Deployment Architecture (PM2 + Nginx)

```
Hetzner AX102 / GEX44
│
├── PM2 (process manager, 24/7)
│   ├── metadron-engine         Python: live_loop_orchestrator (24/7 daemon)
│   │   └── Self-learning: LearningLoop + GSD + Paul plugins
│   │       └── Walk-forward retrain, agent promotion/demotion
│   ├── metadron-api            FastAPI (port 8001, 2 workers)
│   ├── metadron-proxy          Express.js (port 5000, proxies to 8001)
│   ├── mirofish-backend        Flask (port 5001)
│   ├── metadron-learner        Overnight cron: ML retrain + backtest
│   └── mimo-v2-server          (optional) Local LLM inference server
│
├── Nginx (reverse proxy + SSL)
│   ├── metadron.capital → React client (static build)
│   ├── api.metadron.capital → Express proxy → FastAPI
│   ├── mirofish.metadron.capital → Vue static build
│   └── Let's Encrypt auto-renewal
│
├── PostgreSQL (market data, positions, signals, learning history)
├── Redis (signal cache, pub/sub, session state, heartbeat deque)
│
└── systemd
    └── pm2-startup (auto-start PM2 on reboot)
```

### Self-Learning / Self-Managed Flow

```
24/7 CONTINUOUS OPERATION:

INTRADAY (09:30-16:00 ET) — 3-min heartbeat:
  Engine → Data → Signals → ML → Decision → Execute → Learn → Monitor
  └── LearningLoop: signal accuracy tracking, agent scoring
  └── GSDPlugin: gradient signal dynamics, convergence detection
  └── PaulPlugin: pattern memory, context-aware replay

AFTER-HOURS (16:00-20:00 ET) — 30-min heartbeat:
  Engine → Reduced signals → Earnings scan → Agent scoring
  └── AgentMonitor: promote/demote based on daily performance

OVERNIGHT (20:00-08:00 ET) — 1-hr heartbeat:
  Engine → Minimal monitoring
  PM2 cron → metadron-learner:
    ├── QSTrader walk-forward backtesting
    ├── XGBoost/HMM/PPO model retraining on new data
    ├── Pattern evolution (PaulPlugin memory update)
    ├── Agent scorecard recalculation
    └── Model artifacts saved → engine hot-reloads on next heartbeat

PRE-MARKET (08:00-09:30 ET) — 3-min heartbeat:
  Engine → Full data refresh → Overnight signal integration → SEC scan
  └── New models loaded from overnight training
```

---

## Scaling Path

```
Phase 1 (Now):      AX102 dedicated + PM2           ~EUR 90/mo
                    Mimo v2 via API, full engine
                    500 GB NVMe, 128 GB RAM

Phase 2 (Mimo local): Upgrade to GEX44 (GPU)        ~EUR 200-250/mo
                      Self-hosted Mimo v2 inference
                      1 TB NVMe for model weights

Phase 3 (Scale):    Add cloud VPS for backtesting    ~EUR 300-350/mo
                    Separate compute for overnight ML

Phase 4 (HFT):     Add US East node for execution    ~EUR 500+/mo
                    Sub-second order routing
                    Rust order matching engine
```
