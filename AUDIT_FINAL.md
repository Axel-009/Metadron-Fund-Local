# Metadron Platform — Final Audit Report

**Date:** 2026-04-11  
**Branch:** main  
**Status:** READY FOR TESTING (pending Xiaomi API key)

---

## Architecture Overview

| Component | Port | Technology | Role |
|-----------|------|------------|------|
| Engine API | 8001 | FastAPI (Python) | All backend logic — 21 routers, allocation engine, execution, data, ML |
| Frontend | 5000 | Express + React/Vite | UI dashboard, proxies `/api/engine/*`, `/api/allocation`, `/api/chat`, `/api/velocity` → 8001 |
| LLM Bridge | 8002 | FastAPI (Python) | Unified LLM inference via Brain Power (Xiaomi Mimo V2 Pro) |
| Process Manager | — | PM2 (`ecosystem.config.cjs`) | Manages all services |

---

## API Structure (3 Authorized APIs)

| # | API | Purpose | Env Var(s) | Vault Location | Status |
|---|-----|---------|-----------|---------------|--------|
| 1 | **OpenBB / FMP** | All market data (equities, macro, fixed income, news, filings) | `FMP_API_KEY` | `engine/api/vault.py` | Configured |
| 2 | **Alpaca** | Trade execution only (orders, positions, account) | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | `engine/api/vault.py` | Configured |
| 3 | **Xiaomi Mimo V2 Pro** | Brain Power — LLM inference, NanoClaw intelligence, engine actions | `XIAOMI_MIMO_API_KEY` | `engine/api/vault.py` | STUB (awaiting key) |

**No other external APIs are used anywhere in the platform.**  
Removed/replaced: Anthropic, Yahoo Finance, Tradier, ZEP.

---

## Backend Health Check — 21 Routers on Port 8001

| # | Router | Prefix | Purpose |
|---|--------|--------|---------|
| 1 | portfolio | `/api/engine/portfolio` | Portfolio state, holdings, P&L |
| 2 | cube | `/api/engine/cube` | MetadronCube liquidity tensor |
| 3 | macro | `/api/engine/macro` | Macro data (FRED, CPI, GDP) |
| 4 | signals | `/api/engine/signals` | Signal generation and management |
| 5 | risk | `/api/engine/risk` | Risk metrics, VaR, correlation |
| 6 | execution | `/api/engine/execution` | Trade execution via Alpaca |
| 7 | agents | `/api/engine/agents` | NanoClaw, OpenClaw, Ruflo agents |
| 8 | ml | `/api/engine/ml` | ML model serving and inference |
| 9 | monitoring | `/api/engine/monitoring` | Engine health, VPS, logs, errors |
| 10 | universe | `/api/engine/universe` | Universe scan (SP1500 + ETF + FI) |
| 11 | futures | `/api/engine/futures` | Index futures for BetaCorridor |
| 12 | quant | `/api/engine/quant` | Quantitative analytics |
| 13 | etf | `/api/engine/etf` | ETF holdings and analysis |
| 14 | fixed_income | `/api/engine/fixed-income` | Fixed income engine |
| 15 | archive | `/api/engine/archive` | Historical data archive |
| 16 | backtest | `/api/engine/backtest` | SignalBacktester + StrategyBacktester + MonteCarlo |
| 17 | chat | `/api/chat` | NanoClaw chat interface |
| 18 | velocity | `/api/engine/velocity` | Velocity metrics |
| 19 | flows | `/api/engine/flows` | Flow definitions |
| 20 | flow_runs | `/api/engine/flow-runs` | Flow execution runs |
| 21 | api_keys | `/api/engine/api-keys` | API key management |
| — | metrics | `/api/engine` (Prometheus) | Prometheus metrics scraping |

All routers mounted in `engine/api/server.py`.

---

## Frontend Health Check

| Tab | What It Displays | APIs Called |
|-----|------------------|------------|
| Dashboard | Portfolio overview, allocation, P&L | `/api/engine/portfolio/*`, `/api/allocation/*` |
| NanoClaw | Operator chat agent | `/api/chat` → Brain Power (Xiaomi Mimo V2 Pro) |
| Allocation | Engine status, beta corridor, kill switch | `/api/allocation/*` |
| Universe | Scan status, stock universe | `/api/engine/universe/*` |
| Execution | Live orders, positions, Alpaca status | `/api/engine/execution/*` |
| Risk | VaR, correlations, risk metrics | `/api/engine/risk/*` |
| Signals | Signal library, regime classification | `/api/engine/signals/*` |
| Backtest | Backtester UI, Monte Carlo | `/api/engine/backtest/*` |
| TECH | System Status + **API Endpoints** sub-tab | `/monitoring/*` |

### TECH Tab — API Endpoints Sub-Tab (NEW)
Displays all 3 authorized APIs with:
- Endpoint call points from `openbb_data.py`, `alpaca_broker.py`, `brain_power.py`
- Status badges: GREEN (configured) / YELLOW (stub)
- Auth method per API
- Error display with file/line info from `/monitoring/errors`

---

## Data Flow

```
Market Data:    OpenBB/FMP → engine/data/openbb_data.py → allocation engine → signals
Execution:      Allocation decisions → engine/execution/alpaca_broker.py → Alpaca API
Intelligence:   NanoClaw → engine/bridges/brain_power.py → Xiaomi Mimo V2 Pro
LLM Bridge:     All LLM calls → engine/bridges/llm_inference_bridge.py → BrainPowerClient
```

---

## Kill Switch & Reset Flow

- Kill switch in `engine/allocation/allocation_engine.py` — confirmed working
- `pending_reset` flag — operator must explicitly reset after kill switch trigger
- NanoClaw agent has permission-gated write actions (explicit operator approval required)

---

## Universe Scan

- FMP-based via OpenBB (sole data source)
- Covers: S&P 1500 (SP500 + SP400 + SP600) + ETFs + Fixed Income
- Single scan engine in `engine/allocation/universe_scan.py`
- Cross-asset universe defined in `engine/data/cross_asset_universe.py`

---

## Backtester

- `SignalBacktester` — backtest individual signals
- `StrategyBacktester` — backtest full strategies
- `MonteCarloSimulator` — Monte Carlo simulation for risk analysis
- All accessible via `/api/engine/backtest/*`

---

## Known Issues / Warnings

1. **Brain Power in STUB mode** — Xiaomi Mimo V2 Pro API key not yet provided. NanoClaw and LLM bridge return placeholder responses. Platform fully functional otherwise.
2. **yahoo_data.py retained as shim** — `engine/data/universal_pooling.py` imports from it. The shim re-exports everything from `openbb_data.py` (FMP). No Yahoo Finance calls are made.
3. **LLM Bridge port 8002** — Runs as separate PM2 service. All inference now routes through Brain Power instead of Anthropic/Qwen/AirLLM.

---

## Deployment Checklist

- [ ] Provide `XIAOMI_MIMO_API_KEY` (Brain Power)
- [x] `FMP_API_KEY` — configured
- [x] `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` — configured
- [x] All 21 routers mounted on port 8001
- [x] Frontend proxied through port 5000
- [x] PM2 ecosystem configured
- [x] Kill switch + reset flow
- [x] Universe scan (FMP/OpenBB)
- [x] Backtester wired
- [x] TECH tab API Endpoints sub-tab
- [x] API Vault (`engine/api/vault.py`) — single source of truth for all keys
- [x] Brain Power bridge (`engine/bridges/brain_power.py`) — stub mode ready
- [x] NanoClaw agent — swapped from Anthropic to Brain Power
- [x] LLM inference bridge — routes through Brain Power only
- [x] `.env.example` cleaned — unauthorized keys removed
- [x] `AUDIT_FINAL.md` — this document

---

## Files Changed This Final Pass

| Action | File | Description |
|--------|------|-------------|
| **Created** | `engine/bridges/brain_power.py` | Brain Power client (Xiaomi Mimo V2 Pro stub) |
| **Created** | `engine/api/vault.py` | 3-API vault — single source of truth |
| **Created** | `AUDIT_FINAL.md` | This audit report |
| **Modified** | `engine/agents/nanoclaw/nanoclaw_agent.py` | Replaced Anthropic with BrainPowerClient |
| **Modified** | `engine/bridges/llm_inference_bridge.py` | Removed Anthropic/Qwen/AirLLM, routes through Brain Power |
| **Modified** | `engine/data/yahoo_data.py` | Updated deprecation notice, redirects to FMP/OpenBB |
| **Modified** | `client/src/pages/tech-dashboard.tsx` | Added API Endpoints sub-tab with status badges |
| **Modified** | `.env.example` | Removed ANTHROPIC/TRADIER/ZEP, added XIAOMI_MIMO_API_KEY |
