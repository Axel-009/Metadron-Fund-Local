# Metadron Capital — AI Investment Platform

## MCP Plugins (Claude Code Agents)

Install these once on your local terminal — not server-side:

### Sequential Thinking
Structured step-by-step reasoning for complex trade/execution decisions.
```
claude mcp add sequential-thinking -- npx -y @modelcontextprotocol/server-sequential-thinking
```
Agents will use this automatically for multi-step analysis tasks.

### Context7 (Library Documentation)
Up-to-date, version-specific docs — no more hallucinated APIs.
```
claude mcp add --scope user context7 -- npx -y @upstash/context7-mcp
```
Add to CLAUDE.md rule: "Always use Context7 when working with library APIs or code examples."

---


## Session Bootstrap

> **START HERE:** Read `Architecture_DNA/AI_SESSION_INSTRUCTIONS.md` first — it explains the two-zone architecture (repos/ archive vs intelligence_platform/ unified system), change flow rules, and how to add new repos. Then continue below.
>
> "Continue building the Metadron Capital investment platform. The master repo is at `/home/user/Metadron-Capital` with a 6-layer architecture. The investment engine is in `engine/`. All 20 component repos are under `repos/`. Core platform code is in `core/`. Run `python3 -m pytest tests/ -v` to verify. Run `python3 run_open.py` to execute the full signal pipeline."

## Investment Thesis

Beta managed within a 7–12% return corridor (S&P 500 historical earnings range).
Alpha extracted through top-down macro → GICS sector selection → bottom-up stock picking.
Money velocity (V = GDP/M2) + cyclical vs secular decomposition drive regime classification.
Platform targets 95%+ alpha via aggressive multi-sleeve allocation with continuous ML walk-forward optimisation.
Paper broker mode — live HFT opportunity-based execution, continuously scanning for alpha throughout the day.
**5% daily compound target (minimum)** — once hit, risk dials down (AGGRESSIVE → MODERATE → DEFENSIVE) to retain gains on close.
Alpha maximized through positioning + selection; beta managed through leverage execution multipliers.
Data sourced via **OpenBB** (34+ providers: FRED, SEC, Polygon, FMP, CBOE, ECB, OECD, etc.) — sole data source, no yfinance dependency.

## Signal Pipeline

```
UniverseEngine → MacroEngine → MetadronCube → SecurityAnalysis → PatternDiscovery → CrossAssetContagion → SocialPrediction → DistressedAssets → CVR → EventDriven → TickerSelection → AlphaOptimizer → BetaCorridor → DecisionMatrix → L7UnifiedExecutionSurface → AlpacaBroker + PaperLog
     (L1)           (L2)          (L2)          (L2/3.1)          (L2/3.2)            (L2/3.5)             (L2/L3)            (L2)          (L2)     (L2)          (L2/4)          (L3)            (L4)           (L5)                    (L7)                       (L7)
```

### Layer Architecture (L1 → L5 → L2 → L3 → L4 → L7)

```
L1 Data           Financial-Data, QLIB, open-bb        → Market data + factor research (FRED via OpenBB)
L2 Signals        quant-trading (also L7), ML-Macro-Market → Strategy library + regime classifier + HFT execution
                  CTA-code, stock-chain
L3 Intelligence   ai-hedgefund, Mav-Analysis,          → Multi-agent decision engine
                  Air-LLM, AI-Newton
L4 Portfolio      hedgefund-tracker, open-bb,           → Institutional tracker + orchestration
                  financial-distressed-repo,
                  sophisticated-distress-analysis
L5 Infrastructure Kserve, nividia-repo, Air-LLM        → ML serving + GPU inference
L6 Agents         Ruflo-agents, MiroFish (AgentSim) → Market microstructure simulation
L7 HFT/Execution quant-trading, wondertrader,           → 12 technical strategies + micro-price + order matching
                  exchange-core, Quant-Dev-Resources
```

### Reference Repo Mapping

| Role | Repo | Function |
|------|------|----------|
| Data | FRB (avelkoski) | **DEPRECATED** — FRED data now via OpenBB `openbb-fred` provider |
| HFT/Execution | wondertrader (C++→Python) | L7 — CTA trend-following, HFT micro-price, TWAP/VWAP routing, multi-timeframe |
| HFT/Execution | exchange-core v2 (Java→Python) | L7 — Ultra-low-latency order matching engine (LMAX Disruptor ring buffer, Python wrapper) |
| Prediction | MiroFish (666ghj) | Agent-based market microstructure → SocialPredictionEngine |
| Reference | Quant-Developers-Resources | Quant research catalog (11 categories) |
| Backend | Installation-Back-end-Files | ML backends: OpenBB SDK, CAMEL-AI/OASIS, PySR, QLIB, FinBERT, Air-LLM |
| Regime ML | markov-model (hmmlearn) | L3 — Hidden Markov Models for data-driven regime detection → MetadronCube RegimeEngine |

### MacroEngine Cadence

```
MacroEngine
    ├── OpenBB/FRED → M2V, WALCL, T10Y2Y, FEDFUNDS, SOFR, CPI, GDP (direct)
    ├── OpenBB/FMP  → GSIB basket (JPM,BAC,GS,C,MS + 5 intl)
    ├── GMTF        → 4 gammas (Liquidity, FX shock, Wage, Reserve)
    ├── CtV signals → carry-to-vol, stop-loss gate
    ├── RV signals  → Z-score on tension-adjusted yields
    └── outputs     → regime | rm_adjustment | alpha_universe | sector_weights
          │
          ▼
AlphaOptimizer (regime-universe)
    ├── tickers = macro.alpha_universe
    ├── EWMA cov + walk-forward ML + Sharpe SLSQP
    ├── 7 features: original 4 + credit_spread_proxy + rv_zscore + capm_residual
    └── outputs → final_weights | sleeve_beta | pure_alpha_annual
          │
          ▼
AlphaBetaUnleashed (Dataset 1) — 1-min cadence
    ├── Rm_adjusted = Rm_realized + macro.rm_adjustment
    ├── target_beta = corridor_fn(Rm_adjusted) × 4.7 × vol_adj
    ├── MES_hedge_beta = target_beta - sleeve_beta
    └── execute via PaperBroker, AlpacaBroker, or TradierBroker (broker_type="alpaca"|"tradier"|"paper")
```

## Architecture

```
Metadron-Capital/                        ← Master monorepo (Layer 0: Hub)
├── engine/                              ← INVESTMENT ENGINE
│   ├── data/                            ← L1: Unified OpenBB data + universe
│   │   ├── universe_engine.py           ← 1,044+ securities (SP500+SP400+SP600), GICS 4-tier, 26 RV pairs
│   │   ├── cross_asset_universe.py     ← Static cross-asset universe data + GICS sector map
│   │   ├── openbb_data.py              ← Primary data source: OpenBB (34+ providers)
│   │   ├── yahoo_data.py               ← Re-exports from openbb_data (backward compat)
│   │   ├── ingestion_orchestrator.py   ← Multi-asset continuous ingestion (no crypto, US+FTSE equities, G10+India FI/FX)
│   │   └── universal_pooling.py        ← Cross-asset data pooling → engine layer routing
│   ├── signals/                         ← L2: Signal processing
│   │   ├── macro_engine.py             ← GMTF: SDR tension, rotation, velocity, FRB
│   │   ├── metadron_cube.py            ← C(t) = f(L,R,F) + 4-Gate + KillSwitch + FCLP
│   │   ├── security_analysis_engine.py ← L2/L2.5 Graham-Dodd-Klarman (top-down + bottom-up, Stage 3.1)
│   │   ├── contagion_engine.py         ← L3 graph topology, 21 nodes, 7 shock scenarios
│   │   ├── stat_arb_engine.py          ← Medallion mean reversion + cointegration pairs
│   │   ├── pattern_discovery_engine.py ← L2 Pattern Discovery (MiroFish + AI-Newton → PatternDiscoveryBus)
│   │   ├── social_prediction_engine.py ← AgentSim market microstructure signals
│   │   ├── distressed_asset_engine.py  ← 5-model distress + Graham-Mielle (fulcrum/liquidation/Marks)
│   │   ├── cvr_engine.py              ← CVR valuation (5 models, 4 instruments)
│   │   ├── event_driven_engine.py      ← 12-category event-driven (M&A arb, PEAD, etc.)
│   │   └── fed_liquidity_plumbing.py  ← Fed balance sheet, reserves, money velocity, sector flow allocation
│   ├── ml/                              ← L3: ML/AI models
│   │   ├── alpha_optimizer.py           ← Walk-forward ML alpha + mean-variance + credit quality
│   │   ├── backtester.py               ← Walk-forward, Monte Carlo, scenario engine
│   │   ├── qstrader_backtest_bridge.py ← QSTrader institutional backtesting integration
│   │   ├── pattern_recognition.py       ← Candlestick, chart patterns, anomalies
│   │   ├── social_features.py          ← Agent sim feature engineering for ML
│   │   ├── universe_classifier.py       ← XGBoost 4-model ensemble, quality tiers A-G
│   │   ├── model_evaluator.py           ← Per-class P/R/F1, tier-aware distance weighting
│   │   ├── deep_learning_engine.py      ← Pure-numpy PPO agent, 50-feature state vector
│   │   └── bridges/                     ← ML model bridge adapters
│   │       ├── finrl_bridge.py          ← FinRL deep RL framework adapter
│   │       ├── nvidia_tft_adapter.py    ← NVIDIA Temporal Fusion Transformer
│   │       ├── monte_carlo_bridge.py    ← Monte Carlo simulation bridge
│   │       ├── stock_prediction_bridge.py ← Stock prediction model bridge
│   │       ├── deep_trading_features.py ← Deep trading feature engineering
│   │       ├── kserve_adapter.py        ← KServe ML model serving adapter
│   │       └── markov_regime_bridge.py ← hmmlearn HMM → MetadronCube RegimeEngine
│   ├── risk/                            ← Risk models
│   │   ├── monte_carlo_risk.py          ← Research MC (per-ticker VaR/CVaR/tails) → calibrates the gate
│   │   └── portfolio_mc_gate.py         ← Shared pre-trade MC gate: 1-day VaR95 ≤ 1.5% NAV, CVaR95 ≤ 2.5% NAV, P(profit) per instrument
│   ├── portfolio/                       ← L4: Portfolio construction
│   │   └── beta_corridor.py            ← Beta corridor 7–12% + vol-normalisation
│   ├── execution/                       ← L5: Execution + L7 HFT routing
│   │   ├── paper_broker.py             ← Simulated broker (OpenBB prices)
│   │   ├── alpaca_broker.py            ← Live broker via Alpaca API (paper/production)
│   │   ├── tradier_broker.py           ← Live broker via Tradier API (legacy, sandbox/production)
│   │   ├── execution_engine.py         ← Full pipeline orchestrator + ML vote ensemble (broker_type="paper"|"alpaca"|"tradier")
│   │   ├── decision_matrix.py          ← 4-gate approval + spread + Monte Carlo VaR/CVaR/P(profit) checks + full-Kelly sizing (1-contract option exception) + ABU beta
│   │   ├── position_protection.py      ← Drawdown protection on live positions + option expiry closes (every heartbeat)
│   │   ├── event_tail_hedge.py         ← Protective puts/calls ahead of earnings / binary events
│   │   ├── options_engine.py           ← Black-Scholes, Greeks, vol surface, θ+Γ optimizer
│   │   ├── conviction_override.py       ← 3-tier conviction override system
│   │   ├── exchange_core_engine.py     ← L7 LMAX Disruptor ring buffer order matching (Python)
│   │   ├── wondertrader_engine.py      ← L7 CTA trend-following + HFT micro-price + TWAP/VWAP
│   │   └── l7_unified_execution_surface.py ← L7 fused execution arm (all products → Tradier + paper log); G11 = Monte Carlo VaR gate; position-reducing orders exempt from G1–G10
│   ├── live_loop_orchestrator.py       ← End-to-end continuous live loop (ingestion → execution → learning)
│   │   # L7 HFT Execution arm:
│   │   # quant_strategy_executor.py  → 12 independent technical strategies (Stage 6.5)
│   │   # exchange-core v2 (source)   → Java reference (repos/layer7_execution/)
│   │   # wondertrader (source)       → C++ reference (repos/layer7_execution/)
│   ├── agents/                          ← L6: Agent orchestration
│   │   ├── sector_bots.py             ← 11 GICS sector micro-bots + scorecard
│   │   ├── research_bots.py           ← 11 GICS research bots + DNA hierarchy
│   │   ├── agent_scorecard.py          ← Agent ranking + promotion/demotion
│   │   ├── gics_sector_agents.py       ← 11 GICS sector agents, 8 scoring dimensions
│   │   ├── agent_monitor.py            ← 4-tier hierarchy (ELITE/STRONG/DEV/UNDER)
│   │   └── investor_personas.py        ← 12 investor personas (Buffett, Munger, etc.)
│   └── monitoring/                      ← Monitoring layer
│       ├── daily_report.py             ← Open/close reports + sector heatmap
│       ├── platinum_report.py           ← 30-section Platinum Report (9 parts)
│       ├── platinum_report_v2.py        ← Enhanced Platinum Report generator
│       ├── portfolio_report.py          ← Portfolio Analytics Report (3 parts)
│       ├── portfolio_analytics.py       ← Scenario engine + performance analytics
│       ├── sector_tracker.py           ← Sector performance + missed opportunities
│       ├── heatmap_engine.py            ← Sector heatmap visualization engine
│       ├── live_dashboard.py            ← Rich-based terminal dashboard (550+ symbols)
│       ├── live_earnings_graph.py       ← Live earnings graph rendering
│       ├── market_wrap.py              ← Market wrap narrative
│       ├── anomaly_detector.py          ← Statistical anomaly scanner
│       ├── memory_monitor.py           ← Session tracking + EOD summary
│       ├── learning_loop.py            ← Closed-loop feedback: signal accuracy → tier weights → regime calibration
│       └── l7_dashboard.py             ← L7 Risk + TCA dashboard panels (Rich + ASCII)
├── app/                                 ← Web application (FastAPI)
│   └── backend/                         ← API server + services
│       ├── main.py                      ← FastAPI entry point
│       ├── api/                         ← REST + SSE endpoints
│       ├── models/                      ← SQLAlchemy models
│       ├── schemas/                     ← Pydantic schemas
│       └── services/                    ← Agent, graph, backtest services
├── core/                                ← Platform orchestrator (original)
│   ├── platform.py                      ← Central orchestrator (20 modules)
│   ├── signals.py                       ← Cyclical vs secular decomposition
│   └── portfolio.py                     ← Portfolio analytics engine
├── tests/
│   ├── test_platform.py                 ← 11 core tests
│   └── test_engine.py                  ← 58 engine tests (69 total)
├── mirofish/                            ← MiroFish agent-based market simulation
│   ├── backend/                         ← Flask API + OASIS simulation
│   │   ├── app/services/                ← Graph builder, simulation runner, report agent
│   │   ├── app/api/                     ← REST endpoints (graph, simulation, report)
│   │   └── run.py                       ← Backend entry (port 5001)
│   ├── frontend/                        ← Vue 3 + Vite + D3.js
│   │   └── src/                         ← Components, views, API layer
│   └── package.json                     ← Monorepo scripts (dev, build, setup)
├── intelligence_platform/               ← 21 reference repos + QSTrader backtester + plugins (Python-coherent, 16K+ files)
│   ├── README.md                       ← Index of all 21 repos
│   ├── AI-Newton/                      ← Physics-informed ML
│   ├── Air-LLM/                        ← LLM inference
│   ├── CTA-code/                       ← CTA trend-following
│   ├── Financial-Data/                 ← Data pipelines
│   ├── FinancialDistressPrediction/    ← GBM distress (ref for DistressedAssetEngine)
│   ├── Kserve/                         ← ML model serving (Python SDK)
│   ├── ML-Macro-Market/                ← Macro ML models
│   ├── Mav-Analysis/                   ← Portfolio analytics
│   ├── MiroFish/                       ← Market microstructure simulation
│   ├── QLIB/                           ← Microsoft QLIB framework
│   ├── Ruflo-agents/                   ← Agent orchestration (Python only)
│   ├── Stock-techincal-prediction-model/ ← Technical analysis
│   ├── TradeTheEvent/                  ← Event-driven ML (ref for EventDrivenEngine)
│   ├── ai-hedgefund/                   ← AI hedge fund simulation
│   ├── financial-distressed-repo/      ← Distress baseline
│   ├── hedgefund-tracker/              ← Fund tracking
│   ├── nividia-repo/                   ← GPU/AI compute
│   ├── open-bb/                        ← OpenBB terminal
│   ├── quant-trading/                  ← Quant strategies
│   ├── sophisticated-distress-analysis/ ← Advanced distress
│   └── stock-chain/                    ← Blockchain+stocks
├── run_open.py                          ← 09:30 ET — full pipeline execution
├── run_close.py                         ← 16:00 ET — EOD reconciliation
└── repos/                               ← All 23 component repositories (6 layers)
```

## Data Ingestion Constraints (NO CRYPTO)

```
DataIngestionOrchestrator → UniversalDataPool → Engine Layers
    ├── Equities: US S&P 1500 + London FTSE 100 ONLY
    ├── Commodities: Major ETFs only (GLD,SLV,USO,UNG,DBA,DBC,COPX,WEAT,CORN)
    │   → Price reference, cyclical patterns, global trade signals
    ├── Indices: SPY,QQQ,IWM,DIA,VT,EFA,EEM → benchmarking + monthly rebalancing
    ├── Fixed Income:
    │   ├── Sovereign: G10 + India + Japan
    │   ├── Corporate: US credit bonds ONLY (IG: LQD,VCIT,VCSH | HY: HYG,JNK)
    │   └── Structured: Major benchmarks only (MBS: MBB,VMBS)
    ├── Currencies: G10 + INR + JPY (via ETFs + FRED)
    ├── Econometrics: 40+ FRED series (GDP,CPI,M2,WALCL,SOFR,etc.)
    ├── SEC Filings: Major updates ONLY (10-K,10-Q,8-K material) — monthly tracking
    ├── Options: Selected securities opportunistically → alpha maximization
    └── Futures: ES,NQ,YM,RTY,VX,ZN,ZB → beta management corridor
```

## Live Loop Architecture

```
CONTINUOUS LOOP (1-min heartbeat, 09:30-16:00 ET):
  1. DATA INGESTION   → DataIngestionOrchestrator → UniversalDataPool
  2. SIGNAL GENERATION → FedLiquidityPlumbing + MacroEngine + MetadronCube + all engines
  3. INTELLIGENCE      → AlphaOptimizer + MLVoteEnsemble + Agent bots
  4. DECISION          → DecisionMatrix + BetaCorridor + Options scan
  5. EXECUTION         → ExecutionEngine + OptionsEngine + Futures beta hedge
  6. LEARNING          → LearningLoop + GSDPlugin + PaulPlugin
  7. MONITORING        → Portfolio P&L + Risk checks + Anomaly detection

Pre-market (08:00): Full refresh + overnight signals + SEC scan
After-hours (16:00-20:00): Reduced frequency + earnings scan
Overnight: QSTrader backtesting + ML training + pattern evolution
```

## QSTrader Backtesting (intelligence_platform/qstrader/)

```
QSTraderBacktestRunner
    ├── MetadronAlphaModel → wraps signal pipeline as QSTrader alpha
    ├── MetadronRiskModel  → wraps BetaCorridor as QSTrader risk model
    ├── MetadronFeeModel   → realistic spread + commission + impact
    ├── StrategyFactory    → pre-built strategies (cube, macro, ensemble, etc.)
    ├── Walk-forward validation → sliding train/test windows
    ├── Regime backtesting → test across TRENDING/RANGE/STRESS/CRASH
    ├── Strategy comparison → side-by-side Sharpe/DD/alpha analysis
    └── Learning loop integration → feed results into continuous improvement
```

## GSD + Paul Learning Plugins

```
GSDPlugin (Gradient Signal Dynamics):
    Signal momentum | convergence | divergence | gradient decay
    Cross-engine alignment | adaptive learning rate per agent

PaulPlugin (Pattern Awareness & Unified Learning):
    Pattern memory | matching | evolution | context-aware replay
    Unified pattern library shared across all agents

AgentLearningWrapper:
    Attaches GSD + Paul to any agent for continuous learning
```

## MetadronCube — C(t) = f(L_t, R_t, F_t)

### 10-Layer Intelligence

| Layer | Name | Function |
|-------|------|----------|
| 0 | FedPlumbingLayer | SOFR, reserves, TGA, ON-RRP |
| 1 | LiquidityTensor | L(t) in [-1,+1] |
| 2 | ReserveFlowKernel | ΔReserves → ΔEquity/Credit |
| Risk | RiskStateModel | R(t) in [0,1] |
| Flow | CapitalFlowModel | F(t) sector rotation |
| 4 | RegimeEngine (HMM) | TRENDING/RANGE/STRESS/CRASH |
| Gate | GateZAllocator | 5-sleeve allocation |
| Entry | GateLogic | 4-gate entry scoring |
| Kill | KillSwitch | Auto-derisking triggers |
| Cal | FCLPLoop | Daily recalibration |

### Regime Parameters (95% Alpha Target)

| Regime | Leverage | Beta Cap | Beta Burst | Crash Floor |
|--------|----------|----------|------------|-------------|
| TRENDING | 3.0x | 0.65 | 0.70 | ≥+25% |
| RANGE | 2.5x | 0.45 | 0.55 | ≥+25% |
| STRESS | 1.5x | 0.15 | 0.20 | ≥+25% |
| CRASH | 0.8x | -0.20 | -0.10 | ≥+25% |

### Portfolio Allocation Mix (Credit-Aware)

| Sleeve | Allocation | Credit Tier | Notes |
|--------|-----------|-------------|-------|
| IG Equities | 40% | A/B (IG) | All caps — no mega cap restriction |
| Options | 25% | Mixed | 10% IG + 10% HY + 5% Distressed |
| Bond/Commodity ETFs | 10% | — | TLT, GLD, USO, HYG, LQD, etc. |
| HY Equities | 10% | C/D (HY) | BB-B rated, leveraged |
| Distressed Equity | 10% | E/F | Fallen angels, recovery plays |
| Cash (Dry Powder) | 5% | — | Never deployed, buying power reserve |

Deploy 95% of NAV + leverage. Only 5% remains as dry powder.

### Pre-Trade Risk Checks (after the 4 gates)

| Check | Limit | Where |
|-------|-------|-------|
| Spread | Equity/ETF ≤ 30 bps · Option ≤ 10% of mid (no quote = skip) · Futures ≤ 5 bps | DecisionMatrix |
| MC VaR95 (1-day) | legacy engine: ≤ 1.5% NAV whole book — SUPERSEDED by the split risk budget (regime ceiling on beta + 2.5% alpha budget), see Risk budget | DecisionMatrix + L7 G11 |
| MC CVaR95 (1-day) | legacy: ≤ 2.5% NAV — new: ≤ 1.5 × regime VaR ceiling | DecisionMatrix + L7 G11 |
| MC P(profit) | Equity/ETF/Future ≥ 52% · Long option ≥ 25% · Short option ≥ 65%, and expected P&L > 0 (hedges exempt) | DecisionMatrix |
| Kelly | Full (1×) Kelly × vol adj, capped at 20% NAV; option whose 1-contract premium exceeds the cap → exactly 1 contract (still must pass MC) | DecisionMatrix |
| ML agreement (overrides) | ≥ 8 of 10 MLVoteEnsemble tiers agree | ConvictionOverride |

GMTF tension percentile and Gaussian-HMM regime probabilities feed the MACRO_REGIME gate (MacroEngine snapshot → `DecisionMatrix.update_macro_context`).

### 4-Gate Entry Logic

| Gate | Weight | Function |
|------|--------|----------|
| G1 Flow/Headlines | 20% | ETF creations + Tensor signal |
| G2 Macro/Beta | 25% | Kernel projections + rates/FX betas |
| G3 Fundamentals | 30% | Quality/ROIC/FCF + supply-chain |
| G4 Momentum/Tech | 25% | Breadth/leadership/gamma/vanna |

### Kill-Switch Matrix

HY OAS +35bp & VIX term flat/inverted & breadth <50% → auto β ≤ 0.35

### New Engines

- **SecurityAnalysisEngine**: L2/L2.5 Graham-Dodd-Klarman framework (Stage 3.1) — top-down (CAPE, ERP, speculative component, max investment P/E) + bottom-up (Graham Number, NCAV, MoS ≥33%, normalized EPS, ROIC-WACC, DuPont ROE, owner earnings, 8-test investment grading, 5-method IV estimation). Feeds Tier-5 of MLVoteEnsemble. Outputs 12 alpha features for ML walk-forward
- **PatternDiscoveryEngine**: L2 Agent-based market simulation + AI-Newton symbolic regression (PySR) → PatternDiscoveryBus → enrichment features for AlphaOptimizer
- **ContagionEngine**: L3 graph topology, 21 nodes, 7 shock scenarios, multi-step propagation
- **StatArbEngine**: Medallion mean reversion + cointegration pairs + factor residuals (Σβ≈0)
- **OptionsEngine**: Black-Scholes Greeks, θ+Γ optimizer, vol surface, P4 sleeve allocation
- **DistressedAssetEngine**: 5-model ensemble (Altman Z, Merton KMV, Ohlson O, Zmijewski, ML GBM) + Graham-Mielle framework (fulcrum security analysis, Ch.42 orderly liquidation rates, BS×IS cross-ref, Howard Marks 8-factor credit), fallen angel detector, LGD estimator, Kelly-sized opportunities
- **CVREngine**: 5-model CVR valuation (binary option, barrier option, milestone tree, Monte Carlo, real options), liquidity/credit adjustments, 4 live instruments
- **EventDrivenEngine**: 12 event categories, Mitchell-Pulvino M&A arb, PEAD SUE drift, Kelly-sized positions, 10 live events
- **UniverseClassifier**: XGBoost 4-model soft-voting ensemble (GaussianNB+GBM+RF+XGB), quality tiers A-G, reconciliation engine
- **CreditQualityClassifier**: 6-factor weighted credit scoring (AAA-D), feeds Tier-10 of MLVoteEnsemble
- **DeepLearningEngine**: Pure-numpy PPO agent, 50-feature state vector, no external ML dependency
- **InvestorPersonaManager**: 12 investor persona agents (Buffett, Munger, Graham, Ackman, Dalio, etc.)

### MLVoteEnsemble (10 Tiers)

| Tier | Name | Weight | Source |
|------|------|--------|--------|
| T1 | Neural | 1.0 | DRL/FinRL bridge |
| T2 | Momentum | 1.2 | Technical signals |
| T3 | Vol Regime | 0.8 | Volatility regime |
| T4 | Monte Carlo | 0.9 | MC simulation |
| T5 | Quality | 1.1 | Quality ranker |
| T6 | Social | 1.0 | Agent sim consensus |
| T7 | Distress | 0.9 | Distressed asset engine |
| T8 | Event | 1.0 | Event-driven engine |
| T9 | CVR | 0.7 | CVR engine |
| T10 | Credit Quality | 0.9 | UniverseClassifier/CreditQualityClassifier |

## Beta Corridor (Dataset 1)

```
ALPHA = 2% (secular alpha headstart)
R_LOW, R_HIGH = 7%, 12% (Gamma Corridor)
BETA_MAX = 2.0, BETA_INV = -0.136
EXECUTION_MULTIPLIER = 4.7
Vol-normalisation: 15% thesis standard
VaR ≤ $0.30M (95%/1-day) on $20M NAV
```

### Smart Beta v2 — CURRENT definition (owner-approved; code: `metadron_lean/corridor.py`)

Chart: `docs/charts/corridor_v2.png` (script: `docs/charts/corridor_v2.py`).

```
L(Rm, σm) = k · s(Rm) · (Rm − rf) / σm²          long market-beta leverage, capped at 4.5x

Rm     annualised S&P 500 return (expected)          rf  risk-free rate = 4%
σm     current annualised market volatility           k   share of full Kelly = 1.0 (parameter)
t      = clip((Rm − 7%) / 5%, 0, 1)
s(Rm)  = 3t² − 2t³                                   corridor participation (smooth 0 → 1)
γ(Rm)  = ds/dRm = 6t(1 − t) / 5%                     corridor gamma — flattened bell,
                                                     peak 0.30 per 1% at 9.5%, area 1
(Rm − rf)/σm²                                        "gamma market": vol-adjusted full Kelly
```

| Zone | Rm (annualised S&P) | Long beta |
|------|---------------------|-----------|
| DEFENSIVE | < 7% (bear / weak market) | **0** — only shorts / hedges |
| DYNAMIC | 7% – 12% | smooth ramp, k · s · (Rm − 4%)/σ² |
| ROTATION | > 12% | full vol-adjusted Kelly + active sector rotation, up to the 4.5x cap |

Leverage (k = 1): σ 15% → 0.43 / 1.22 / 2.79 / 3.56 at Rm 8.5 / 9.5 / 11 / 12%, 4.44 at 14%, 4.5 cap
from ≈14.1%; σ 20% → 2.00 at 12%, 3.00 at 16%; σ 25% → 1.28 at 12%. Rising volatility shrinks
leverage by 1/σ², so the book de-risks itself in volatile tapes without exiting.
Rebalance only when |ΔL| > 0.05. Delivery: Micro E-mini S&P futures (MES) — 4.5x is not reachable with
Reg-T stock margin (≈2x); SPY is the ≤2x alternative. Extra leverage above 12% may come from long calls
when implied vol ≤ realised (delta efficiency Ω = Δ·S/premium), else futures / call spreads.

### Risk budget (owner-approved; code: `metadron_lean/risk_budget.py`) — replaces "VaR ≤ 1.5% NAV"

A single 1.5% 1-day VaR on the whole book caps equity leverage at 0.97x at σ 15% (0.72x at 20%) and
blocks most small/mid caps — superseded. Under vol-adjusted Kelly, beta VaR = 1.645·k·s·Sharpe/√252
(falls as σ rises), so VaR is a regime ceiling, not the sizing tool.

| Layer | Rule |
|-------|------|
| 1 Sizing | corridor k-Kelly leverage above |
| 2 Regime VaR ceiling on market-beta risk (1-day 95%) | TRENDING 7.0% · RANGE 4.5% · STRESS 2.25% · CRASH 1.0% → leverage ceiling = VaR / (1.645·σm/√252). TRENDING 7.0% = the VaR of 4.5x at σ 15%, so the owner's max is reachable |
| 3 Stock-specific (alpha) VaR budget, 1-day 95% | **2.5% of NAV** (parameter) on idiosyncratic risk after removing market beta; per-name ≤ 0.5% NAV (≈ 9.7% weight at 50% vol); sizing ∝ 1/vol; same-sector names correlated 0.3; long options count at premium (defined risk); trades scaled down to fit, not rejected |
| 4 Hard stops | gross ≤ 4.5x; CVaR95 ≤ 1.5 × regime ceiling; drawdown −4% trim (halve), −8% close all but hedges |

Diversified small/mid-cap books fit easily: 40 names × 2.5% at 45% vol → 0.74%; 20 × 5% → 1.04%;
the budget binds only on concentration (10 × 10% at 50% vol → 1.64%; 5 × 20% → 2.32%).

### Smart Beta v1 — corridor derivation (superseded by v2, kept for reference)

Chart: `docs/charts/beta_gamma_smooth.png` (script: `docs/charts/beta_gamma_smooth.py`).

**Labels (use these names everywhere — code, docs, dashboards):**

| Symbol | Name | Meaning |
|--------|------|---------|
| Rm | expected market return | expected annual S&P 500 return; the corridor is Rm ∈ [7%, 12%] (S&P historical earnings range) |
| β(Rm) | corridor beta | target exposure identifier, a smooth S-curve in Rm |
| β_L, β_H | corridor floor / ceiling | β_L = −0.029 at Rm = 7%, β_H = 0.425 at Rm = 12% |
| Δβ | corridor span | β_H − β_L = 0.454 |
| γ(Rm) | corridor gamma | **γ = dβ/dRm**, the rate at which beta changes with Rm (as option gamma is the slope of delta). A flattened bell. |
| λ | risk aversion | the Merton/CRRA coefficient (NOT called gamma, to avoid the clash with corridor gamma) |
| h | hurdle | Rm at which β = 0 |
| σ_std | standard volatility | 15% thesis standard |
| M | execution multiplier | 4.7 — converts β into portfolio leverage |

**1. Endpoints (Merton, Brownian motion).** Under dS/S = μ dt + σ dW a CRRA investor holds
β = (Rm − h)/(λσ²). The two corridor endpoints pin λσ² and h:

```
λσ² = (12% − 7%) / (β_H − β_L) = 0.05 / 0.454 = 0.1101   → λ ≈ 4.89 at σ = 15%
h   = 7% − β_L · λσ²          = 7% + 0.029 × 0.1101 = 7.32%
check: β(7%) = (0.07 − 0.0732)/0.1101 = −0.029 ; β(12%) = (0.12 − 0.0732)/0.1101 = 0.425
```
Only the product λσ² is identified (λ = 0.1101/σ²: 7.65 at 12%, 4.89 at 15%, 3.40 at 18%,
2.75 at 20%). Equivalently β = 1/λ ≈ 20% of full Kelly at σ = 15%.

**2. Smooth beta curve (the shape).** Inside the corridor β follows a smoothstep S-curve; outside
it is flat:

```
t     = clip((Rm − 7%) / 5%, 0, 1)
β(Rm) = β_L + Δβ · (3t² − 2t³)
```
Values: β(7%) = −0.029, β(9.5%) = 0.198, β(12%) = 0.425; flat beyond both edges.

**3. Corridor gamma (the flattened bell).**

```
γ(Rm) = dβ/dRm = Δβ · 6t(1 − t) / 5%      for 7% < Rm < 12%,  0 outside
```
- zero at and beyond 7% and 12% (exposure stops changing at the corridor edges)
- peak at Rm = 9.5%: γ_max = 1.5 · Δβ / 5% = 13.62 per unit Rm = **0.136 per 1% of Rm**
- area under γ over the corridor = Δβ = **0.454**
- the old linear ramp has a flat-box gamma of 0.091 per 1% with jumps at 7% and 12% —
  superseded by the smooth curve.

**4. From beta to leverage.**

```
leverage(Rm) = β(Rm) × M × vol_scale(σ_m)      M = 4.7
             → −0.136 at Rm ≤ 7% (hedge floor, BETA_INV) … 2.00 at Rm ≥ 12% (BETA_MAX)
vol_scale    = σ_std / σ_m  (current, linear)   or  (σ_std / σ_m)²  (Merton-consistent) — OPEN
```
Then: regime cap (or burst cap when momentum confirms) → hard caps (VaR95 ≤ 1.5% NAV,
gross-leverage gate). M ≈ λ, so leverage ≈ full Kelly at σ = 15%; at higher true volatility a
fixed M = 4.7 is over-Kelly (full-Kelly leverage at 12%: 2.08 at σ 15%, 1.44 at 18%, 1.17 at 20%).

**5. Inputs each cycle.** Rm_adj = realised SPY drift + macro adjustment (GMTF tension, HMM regime
probabilities) + Kronos index forecast → β(Rm_adj), γ(Rm_adj) → leverage budget L* (gross, split
across sleeves by Kelly risk-adjusted alpha within sleeve caps) → SPY / index-option overlay closes
the residual net-beta gap. Rebalance only when |Δβ| > 0.05 (throttle).

**Known defects in the legacy engine (fixed in the LEAN build):** linear ramp instead of the smooth
curve; constants duplicated in 3 places; AlphaBetaUnleashed calls a non-existent method and uses a
fallback that saturates at 2.0 near Rm ≈ 7.4%, and its output is never executed; sleeve deployment
ignores the corridor; TRENDING cap 0.65 makes the top of the corridor inert; beta burst never applied.
ARCHITECTURE_DNA.md's "linear(0 → BETA_MAX)" description is wrong.

## Target Architecture — LEAN / QuantConnect migration (agreed design)

Stack: QuantConnect cloud (LEAN, dashboard, backtests, live node) → IBKR (paper now,
live later; QC data for signals, IBKR for execution + cross-check). Agents: Claude
Managed Agents (Anthropic Console = agent dashboard). Readable control room: Obsidian
vault in this repo (agents write notes; Dataview tables; graph view). Kept from the
engine: beta corridor, options engine, Markov HMM, GMTF, MetadronCube, pattern
recognition, security analysis, ML ensemble, decision gates, MC gate, protection,
event hedges. Added: Kronos (forecasting). Reference only: freqtrade (GPLv3 — copy
design, never code): FreqAI-style retraining, protections, dry-run first, chat control.

1. **Forecasting → reallocation.** Kronos + walk-forward ML give per-name return and
   vol forecasts → (a) alpha model expected return, (b) forecast vol in Kelly, vol
   scaling and the MC gate, (c) index forecast into Rm_adj → β*, L*. Rebalance only
   when a weight or the beta gap moves > 0.05 (gamma throttle).
2. **Fundamentals → selection.** SecurityAnalysisEngine (CAPE/ERP/max P/E top-down;
   Graham number, NCAV, MoS ≥ 33%, ROIC−WACC, DuPont, owner earnings, 8-test grade,
   5-method IV bottom-up) + ML quality tier + credit score → Fundamentals gate (40%,
   must pass). Data: QC Morningstar fundamentals + SEC filings.
3. **Universe data.** QC point-in-time S&P 1500 universe (daily refresh); prices/options
   from QC, cross-checked vs IBKR; macro from FRED. "Data health" chart: names scanned
   vs expected, stale/missing data, failed cross-checks, skipped chains → alerts.
4. **Position monitoring.** QC live dashboard; IBKR app/TWS (independent check); daily
   reconciliation by the risk-officer agent; Obsidian daily note per position (why
   held, gate scores, drawdown, protection, upcoming events/hedges); phone alerts; Jarvis.
5. **Profitability gaps.** (a) live paper P&L vs backtest expectation; (b) per-model
   logged forecasts scored (hit rate, IC) + weekly ablation backtests — zero
   contribution 4 weeks → down-weight then remove; (c) per-agent scorecard (accuracy,
   P&L of approved changes, cost, rejection rate) → DIRECTOR…RECRUIT, demote/pause.
6. **Dynamic tilt under hard caps.** Adaptive layer (gate/tier weights, corridor slope
   within band, Kelly vol adj, thresholds) retuned weekly by walk-forward optimisation
   inside fixed min/max, adopted only if it beats current out of sample; regime shifts it
   too. Hard caps (VaR95 1.5%, CVaR 2.5%, position 20%, regime leverage/beta caps,
   drawdown stops −6/−10% position, −4/−8% portfolio, daily-loss breaker) live in a
   locked config — owner-only changes; agents can only propose.
7. **ML models.** All free/open source, run inside the QC algorithm: walk-forward alpha
   (linear + XGBoost, core), 4-classifier quality tier (NB, GBM, RF, XGBoost), HMM,
   Kronos-small (PyTorch; fine-tuned monthly off-platform, weights in QC Object
   Store). The first ablation decides which survive.
8. **Macro ↔ micro.** Macro (GMTF, HMM, Fed liquidity, curve, credit) sets regime, Rm_adj,
   β*, leverage budget, sector and sleeve weights → micro (fundamentals, ML, Kronos,
   patterns, events) picks names inside that budget → decision matrix needs both
   (Fundamentals 40 / Flow 20 / Macro 20 / Momentum 20) → Kelly size capped by regime
   and VaR → corridor overlay closes the beta gap. Macro can shrink/veto; micro cannot
   override a macro cap.
9. **Capital structure.** New monitor for every held/universe name: HY/IG OAS level +
   change (FRED ICE BofA series), debt/EBITDA, interest coverage, 1–3y maturity wall,
   FCF vs capex burn, leases (filings), new facilities / private credit / secured debt
   (8-K, events agent), Merton equity-implied default probability and credit spread vs
   market spread. Deteriorating structure → Fundamentals-gate penalty or block, tighter
   protection, event tail hedge. Issuer-level bond/CDS data = paid vendor, only if the
   free signals prove out in backtests. (Current "credit_spread" is an HY−IG ETF
   volatility difference, not a spread — replace.)

## Key Commands

```bash
# Run full pipeline (morning open)
cd /home/user/Metadron-Capital && python3 run_open.py

# Run all tests (69 tests)
python3 -m pytest tests/ -v

# Platform health check
python3 bootstrap.py

# Core platform status
python3 core/platform.py
```

## Design Rules

1. **All data via OpenBB** (34+ providers: FRED, SEC, CBOE, etc.) — sole data source, no broker dependency
2. **Paper broker default** — AlpacaBroker available for live/paper execution (set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER_TRADE in .env). Switch with `broker_type="alpaca"`. Legacy TradierBroker also supported via `broker_type="tradier"`.
3. **6-layer architecture is immutable** — extend within layers
4. **Beta managed within 7–12% corridor** — vol-normalised
5. **Alpha targeted at 95%+** — aggressive multi-sleeve allocation
6. **All agents use Anthropic API** (claude-opus-4-6) when LLM needed
7. **Pure-numpy fallbacks** — no bridge crashes if ML framework missing
8. **try/except on ALL external imports** — system runs degraded, never broken
9. **Tests must pass** before pushing
10. **Session continuity** — this CLAUDE.md serves as the bootstrap context

## 29 Signal Types

MICRO_PRICE_BUY/SELL, RV_LONG/SHORT, FALLEN_ANGEL_BUY,
ML_AGENT_BUY/SELL, DRL_AGENT_BUY/SELL, TFT_BUY/SELL,
MC_BUY/SELL, QUALITY_BUY/SELL,
SOCIAL_BULLISH, SOCIAL_BEARISH, SOCIAL_MOMENTUM, SOCIAL_REVERSAL,
DISTRESS_FALLEN_ANGEL, DISTRESS_RECOVERY, DISTRESS_AVOID,
CVR_BUY, CVR_SELL,
EVENT_MERGER_ARB, EVENT_PEAD_LONG, EVENT_PEAD_SHORT, EVENT_CATALYST,
HOLD

## MiroFish Social Prediction Engine

### Architecture

```
MiroFish Backend (Flask :5001) + Frontend (Vue :5174)
    ├── Document Input → LLM Ontology → Zep Knowledge Graph
    ├── Entity Extraction → Agent Profile Generation
    ├── OASIS Social Simulation (Twitter/Reddit)
    │   └── actions.jsonl (12 action types per agent per round)
    └── Report Generation (LLM analysis)
         ↓
SocialPredictionEngine (engine/signals/)
    ├── Parse actions.jsonl from latest simulation
    ├── Build agent behavioral profiles
    ├── Compute topic-level sentiment
    ├── Map topics → tickers (TOPIC_TICKER_MAP)
    ├── Detect narrative regime (trending/reversing/stable)
    └── Output: SocialSnapshot
         ↓
SocialFeatureBuilder (engine/ml/)
    ├── Sentiment momentum (EMA fast/slow, MACD, z-score)
    ├── Engagement velocity + acceleration
    ├── Consensus strength + influence Gini
    └── Composite social_alpha signal
         ↓
MLVoteEnsemble Tier-10 (execution_engine.py)
    ├── Ticker-specific social sentiment vote
    ├── Fallback to aggregate market sentiment
    └── Weighted into 10-tier ensemble score
```

### Pipeline Integration

Stage 3.7 in run_pipeline() — runs after MetadronCube, before ticker selection.
Social signals feed into:
- Tier-6 of 10-tier ML Vote Ensemble (weighted vote ±1)
- Ticker-level sentiment available via `social.get_ticker_signal(ticker)`
- Social features available via SocialFeatureBuilder for walk-forward ML

### MiroFish Services

| Service | Port | Function |
|---------|------|----------|
| Backend | 5001 | Flask API + simulation orchestration |
| Frontend | 5174 | Vue 3 + D3.js visualization |

```bash
# Start MiroFish services
cd mirofish/backend && python run.py &      # Backend
cd mirofish/frontend && npm run dev &        # Frontend
```

## Agent Hierarchy (DNA Framework)

| Rank | Requirements | Promotion |
|------|-------------|-----------|
| DIRECTOR | Sharpe >2.5, accuracy >85% | Top performer 8+ weeks |
| GENERAL | Sharpe >2.0, accuracy >80% | 4 consecutive top weeks |
| CAPTAIN | Sharpe >1.5, accuracy >55% | - |
| LIEUTENANT | Sharpe >1.0, accuracy >50% | - |
| RECRUIT | Below thresholds | Demoted after 2 bottom weeks |

Weekly score: 40% accuracy + 30% Sharpe + 30% hit rate

## Conviction Override System

| Tier | Confidence | Multiplier | Agents Required |
|------|-----------|------------|-----------------|
| CONTROLLED | 90-95% | 1.5x | 1 |
| AGGRESSIVE | 95-98% | 2.0x | 2 |
| MAXIMUM | >98% | 2.0x+ | 3 |
