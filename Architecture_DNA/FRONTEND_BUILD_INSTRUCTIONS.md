# Metadron Fund — Frontend Build Instructions

> **UI Name:** Metadron Fund
> **Architecture:** Tabbed UI shell with each tab connected to a functional layer of the intelligence platform
> **Build Target:** Local development, integrated system

---

## PRE-BUILD TASKS

### 1. Build the Stock Prediction Model System

Initialize and build the stock forecasting JS module located at:
- `intelligence_platform/Stock-techincal-prediction-model/stock-forecasting-js/`

This module uses Vanilla JS + Materialize CSS + D3/Plotly/TensorFlow.js for in-browser LSTM stock price prediction. It needs a proper build pipeline (package.json, Vite) to integrate into the consolidated frontend.

### 2. Deduplicate Stock Prediction Modules

Two duplicate stock prediction directories exist:
- `intelligence_platform/Stock-techincal-prediction-model/stock-forecasting-js/`
- `intelligence_platform/Stock-prediction/stock-forecasting-js/`

**Action:** Merge these into a single module under one path. Remove all duplicate files. The consolidated module should contain no redundant code — one source of truth for stock forecasting JS.

### 3. Remove MiroFish Frontend Stub

The MiroFish frontend at `mirofish/frontend/` is an empty stub (no package.json, just an empty `src/assets/` directory). It will not be part of the Metadron Fund UI.

**Action:** Delete the `mirofish/frontend/` directory entirely. MiroFish social prediction data will be consumed via backend APIs through the integrated system, not through a standalone frontend.

### 4. Ruvocal Chat — Openclaw Integration

The Ruvocal Chat UI (`intelligence_platform/Ruflo-agents/ruflo/src/ruvocal/`) will receive its orders from **Openclaw**, the CEO assistant agent. Ruvocal is the chat interface through which Openclaw communicates directives, reviews, and instructions to the system. Configure Ruvocal to accept and display Openclaw command streams.

### 5. AI Hedgefund Frontend — Full Platform Integration

The AI Hedgefund frontend (`intelligence_platform/ai-hedgefund/app/frontend/`) must be integrated with the entire intelligence platform and all its systems. This is the most complete React frontend and serves as the base for the consolidated Metadron Fund UI. Wire it to all engine layers (L1-L7), agents (L6), and monitoring systems.

### 6. OpenBB Components — Shell Integration

The OpenBB Plotly and Tables components:
- `intelligence_platform/open-bb/frontend-components/plotly/`
- `intelligence_platform/open-bb/frontend-components/tables/`

These will be installed and shelled into the overall Metadron Fund frontend build. They provide the reusable charting (Plotly) and data table (React Table + Virtual Scrolling) components used across all tabs.

---

## METADRON FUND — UI TAB LAYOUT

The UI is structured as a tabbed interface. Each tab connects to a different functional layer of the integrated intelligence platform.

---

### Tab 1 — OpenBB (Platform Shell)

The entire OpenBB package in its entirety, serving as the foundational shell for the Metadron Fund UI. This is the base layer that all other tabs sit within.

**Source:** `intelligence_platform/open-bb/`

---

### Tab 2 — Market Wrap & News

| Component | Source | Description |
|-----------|--------|-------------|
| Market Direction | MarketWrap engine | General market direction narrative |
| Scenario Thesis | MacroEngine + MetadronCube | Daily base, up, and down scenarios — short, medium, and long term |
| Sector Heatmap | `engine/monitoring/heatmap_engine.py` | GICS sector heatmap visualization (HeatmapEngine) |
| CVR Dates & News | `engine/signals/cvr_engine.py` | CVR event dates, acquisition news, catalyst calendar |
| Fed Calendar | MacroEngine / FRED | Fed meeting dates, rate decisions, balance sheet updates |
| Incoming Events | EventDrivenEngine + earnings feeds | Earnings releases, dividend dates, ex-dates |

---

### Tab 3 — Live Dashboard

| Component | Source | Description |
|-----------|--------|-------------|
| Live Execution P&L | ExecutionEngine + PaperBroker/AlpacaBroker | Real-time profit & loss display |
| TCA | L7 Unified Execution Surface | Transaction Cost Analysis |
| Slippage | L7 Execution | Slippage tracking per fill |
| Execution Efficiency | L7 Dashboard | Fill rate, latency, routing quality |
| Technical Analysis | `intelligence_platform/quant-trading/` | Chart patterns, indicators, similar to quant-trading module visualizations |

---

### Tab 4 — Asset Allocation

| Component | Source | Description |
|-----------|--------|-------------|
| Live NAV | PaperBroker / AlpacaBroker | Real-time net asset value display |
| Basket Formation | DecisionMatrix + AlphaOptimizer | Pre-execution basket construction view |
| Dynamic Movers Engine | UniverseEngine + signals | Real-time movers, momentum shifts |
| Allocation CSV | Portfolio export | Downloadable CSV of current allocation |
| VIX | OpenBB data / CBOE | Live VIX display |
| Major Indices | SPY, QQQ, IWM, DIA, VT, EFA, EEM | Live index pricing |
| GICS Filter | UniverseEngine GICS 4-tier | Filter portfolio by GICS sector, industry group, industry, sub-industry |
| Instrument Classification | Cross-asset universe | Filter by instrument type (equity, option, future, FI, FX) |

---

### Tab 5 — Risk & Portfolio Monitor

| Component | Source | Description |
|-----------|--------|-------------|
| Options Risk & Greeks | `engine/execution/options_engine.py` | Delta, Gamma, Theta, Vega, Rho — live |
| Portfolio Metrics | PortfolioAnalytics | Sharpe ratio, Sortino, max drawdown, Calmar — all dynamic |
| Filled / No Fills | ExecutionEngine | Order fill status, partial fills, rejects |
| Live Return | PaperBroker / AlpacaBroker | Real-time return calculation |
| Portfolio Risk | `engine/risk/monte_carlo_risk.py` | VaR, CVaR, stress VaR |
| Total Initial Margin | AlpacaBroker account | Margin utilization |
| Greeks Risk | OptionsEngine aggregate | Portfolio-level Greeks exposure |
| Liquidity Risk | ContagionEngine + universe data | Liquidity scoring, bid-ask spread analysis |

---

### Tab 6 — Machine Learning UI

| Component | Source | Description |
|-----------|--------|-------------|
| Stress Testing | `engine/ml/backtester.py` + MonteCarloRiskEngine | Scenario stress test results |
| Backtesting Results | QSTrader backtest bridge | Walk-forward, regime-specific, strategy comparison |
| Trend & Pattern Recognition | `engine/ml/pattern_recognition.py` + PatternDiscoveryEngine | Candlestick, chart patterns, symbolic regression discoveries |
| Statistical Anomalies | `engine/monitoring/anomaly_detector.py` | Z-score anomalies, distribution breaks |
| Relative Value Report | `engine/signals/stat_arb_engine.py` | RV pair spreads, z-scores, cointegration status for 26 pairs |

---

### Tab 7 — Technical Dashboard

| Component | Source | Description |
|-----------|--------|-------------|
| Active Engine Visualization | All engine modules | Visual status of each active engine (L1-L7) with health indicators |
| Memory Consumption | System monitoring | RAM, CPU, process memory per engine |
| VPS Health | System monitoring | Disk, network, uptime, load average |
| Learning Engine | `engine/monitoring/learning_loop.py` | GSD + Paul plugin status, gradient dynamics, pattern evolution |
| Learning Files Usage | ModelStore + learning data | Model versions, training data freshness, storage utilization |
| Latency | L7 Dashboard + heartbeat config | Per-engine latency, heartbeat cadence, API response times |
| Error Log (Live) | Structured logging stream | Real-time error message feed from all engines |
| Fallback Updates | ExecutionEngine | Any orders routed to PaperBroker due to live broker failures |

---

### Tab 8 — Monitoring & Reporting

All reports available in **PDF and CSV** export formats.

| Report | Source | Description |
|--------|--------|-------------|
| Platinum Report | `engine/monitoring/platinum_report.py` | 30-section executive macro state (9 parts) |
| Platinum Report V2 | `engine/monitoring/platinum_report_v2.py` | Enhanced platinum report generator |
| Portfolio Report | `engine/monitoring/portfolio_report.py` | Scenario engine + performance deep-dive (3 parts) |
| Portfolio Analytics | `engine/monitoring/portfolio_analytics.py` | Deep portfolio analytics + scenario engine |
| Daily Report | `engine/monitoring/daily_report.py` | Open/close reports + sector heatmap |
| Sector Tracker | `engine/monitoring/sector_tracker.py` | Sector performance + missed opportunities (>20% movers) — Daily |
| Anomaly Detector | `engine/monitoring/anomaly_detector.py` | Statistical anomaly scanner |
| Market Wrap | `engine/monitoring/market_wrap.py` | Narrative market summary |
| Memory Monitor | `engine/monitoring/memory_monitor.py` | Session tracking + EOD summary |
