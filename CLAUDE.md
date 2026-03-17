# Metadron Capital — AI Investment Platform

## Session Bootstrap

> **START HERE:** Read `Architecture_DNA/AI_SESSION_INSTRUCTIONS.md` first — it explains the two-zone architecture (repos/ archive vs intelligence_platform/ unified system), change flow rules, and how to add new repos. Then continue below.
>
> "Continue building the Metadron Capital investment platform. The master repo is at `/home/user/Metadron-Capital` with a 6-layer architecture. The investment engine is in `engine/`. All 20 component repos are under `repos/`. Core platform code is in `core/`. Run `python3 -m pytest tests/ -v` to verify. Run `python3 run_open.py` to execute the full signal pipeline."

## Mission

**$1,000 → $100,000 in 100 days** (~4.6% daily compound return).
Target 95%+ alpha. Compete with and outperform the Medallion fund.

## Investment Thesis

Beta managed within a 7–12% return corridor (S&P 500 historical earnings range).
Alpha extracted through top-down macro → GICS sector selection → bottom-up stock picking.
Money velocity (V = GDP/M2) + cyclical vs secular decomposition drive regime classification.
Platform targets 95%+ alpha via aggressive multi-sleeve allocation with continuous ML walk-forward optimisation.
Paper broker mode — not connected to a live broker.
Data sourced via **OpenBB** (34+ providers: FRED, SEC, Polygon, FMP, CBOE, ECB, OECD, etc.) — sole data source, no yfinance dependency.

## Signal Pipeline

```
UniverseEngine → MacroEngine → MetadronCube → CrossAssetContagion → SocialPrediction → DistressedAssets → CVR → EventDriven → TickerSelection → AlphaOptimizer → BetaCorridor → DecisionMatrix → ExecutionEngine
     (L1)           (L2)          (L2)              (L2/3.5)           (L2/L3)              (L2)          (L2)     (L2)          (L2/4)          (L3)            (L4)           (L5)             (L5)
```

### Layer Architecture (L1 → L5 → L2 → L3 → L4 → L7)

```
L1 Data           Financial-Data, QLIB, open-bb        → Market data + factor research (FRED via OpenBB)
L2 Signals        quant-trading, ML-Macro-Market,      → Strategy library + regime classifier
                  CTA-code, stock-chain
L3 Intelligence   ai-hedgefund, Mav-Analysis,          → Multi-agent decision engine
                  Air-LLM, AI-Newton
L4 Portfolio      hedgefund-tracker, open-bb,           → Institutional tracker + orchestration
                  financial-distressed-repo,
                  sophisticated-distress-analysis
L5 Infrastructure Kserve, nividia-repo, Air-LLM        → ML serving + GPU inference
L6 Agents         Ruflo-agents, MiroFish                → Multi-agent orchestration + social prediction
L7 HFT/Execution wondertrader, exchange-core,          → HFT micro-price + order matching
                  Quant-Developers-Resources
```

### Reference Repo Mapping

| Role | Repo | Function |
|------|------|----------|
| Data | FRB (avelkoski) | **DEPRECATED** — FRED data now via OpenBB `openbb-fred` provider |
| Execution | wondertrader (C++→Python) | HFT micro-price signal → MicroPriceEngine |
| Execution | exchange-core (Java→Python) | Order matching / paper order book |
| Prediction | MiroFish (666ghj) | Agent-based social simulation → SocialPredictionEngine |
| Reference | Quant-Developers-Resources | Quant research catalog (11 categories) |

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
    └── execute via PaperBroker (→ IBBroker when live)
```

## Architecture

```
Metadron-Capital/                        ← Master monorepo (Layer 0: Hub)
├── engine/                              ← INVESTMENT ENGINE
│   ├── data/                            ← L1: Unified OpenBB data + universe
│   │   ├── universe_engine.py           ← 150+ securities, GICS 4-tier, 26 RV pairs
│   │   ├── openbb_data.py              ← Primary data source: OpenBB (34+ providers)
│   │   └── yahoo_data.py               ← Re-exports from openbb_data (backward compat)
│   ├── signals/                         ← L2: Signal processing
│   │   ├── macro_engine.py             ← GMTF: SDR tension, rotation, velocity, FRB
│   │   ├── metadron_cube.py            ← C(t) = f(L,R,F) + 4-Gate + KillSwitch + FCLP
│   │   ├── contagion_engine.py         ← L3 graph topology, 21 nodes, 7 shock scenarios
│   │   ├── stat_arb_engine.py          ← Medallion mean reversion + cointegration pairs
│   │   ├── social_prediction_engine.py ← MiroFish bridge → social sentiment signals
│   │   ├── distressed_asset_engine.py  ← 5-model distress ensemble (Z/KMV/O/Zmij/ML)
│   │   ├── cvr_engine.py              ← CVR valuation (5 models, 4 instruments)
│   │   └── event_driven_engine.py      ← 12-category event-driven (M&A arb, PEAD, etc.)
│   ├── ml/                              ← L3: ML/AI models
│   │   ├── alpha_optimizer.py           ← Walk-forward ML alpha + mean-variance + credit quality
│   │   ├── backtester.py               ← Walk-forward, Monte Carlo, scenario engine
│   │   ├── pattern_recognition.py       ← Candlestick, chart patterns, anomalies
│   │   ├── social_features.py          ← MiroFish social feature engineering for ML
│   │   ├── universe_classifier.py       ← XGBoost 4-model ensemble, quality tiers A-G
│   │   ├── model_evaluator.py           ← Per-class P/R/F1, tier-aware distance weighting
│   │   ├── deep_learning_engine.py      ← Pure-numpy PPO agent, 50-feature state vector
│   │   └── bridges/                     ← ML model bridge adapters
│   │       ├── finrl_bridge.py          ← FinRL deep RL framework adapter
│   │       ├── nvidia_tft_adapter.py    ← NVIDIA Temporal Fusion Transformer
│   │       ├── monte_carlo_bridge.py    ← Monte Carlo simulation bridge
│   │       ├── stock_prediction_bridge.py ← Stock prediction model bridge
│   │       ├── deep_trading_features.py ← Deep trading feature engineering
│   │       └── kserve_adapter.py        ← KServe ML model serving adapter
│   ├── portfolio/                       ← L4: Portfolio construction
│   │   └── beta_corridor.py            ← Beta corridor 7–12% + vol-normalisation
│   ├── execution/                       ← L5: Execution
│   │   ├── paper_broker.py             ← Simulated broker (OpenBB prices)
│   │   ├── execution_engine.py         ← Full pipeline orchestrator + ML vote ensemble
│   │   ├── decision_matrix.py          ← 6-gate trade approval + Kelly sizing + ABU beta
│   │   ├── options_engine.py           ← Black-Scholes, Greeks, vol surface, θ+Γ optimizer
│   │   └── conviction_override.py       ← 3-tier conviction override system
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
│       └── memory_monitor.py           ← Session tracking + EOD summary
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
├── mirofish/                            ← MiroFish social prediction engine
│   ├── backend/                         ← Flask API + OASIS simulation
│   │   ├── app/services/                ← Graph builder, simulation runner, report agent
│   │   ├── app/api/                     ← REST endpoints (graph, simulation, report)
│   │   └── run.py                       ← Backend entry (port 5001)
│   ├── frontend/                        ← Vue 3 + Vite + D3.js
│   │   └── src/                         ← Components, views, API layer
│   └── package.json                     ← Monorepo scripts (dev, build, setup)
├── intelligence_platform/               ← 21 reference repos (Python-coherent, 16K+ files)
│   ├── README.md                       ← Index of all 21 repos
│   ├── AI-Newton/                      ← Physics-informed ML
│   ├── Air-LLM/                        ← LLM inference
│   ├── CTA-code/                       ← CTA trend-following
│   ├── Financial-Data/                 ← Data pipelines
│   ├── FinancialDistressPrediction/    ← GBM distress (ref for DistressedAssetEngine)
│   ├── Kserve/                         ← ML model serving (Python SDK)
│   ├── ML-Macro-Market/                ← Macro ML models
│   ├── Mav-Analysis/                   ← Portfolio analytics
│   ├── MiroFish/                       ← Social prediction (ref for SocialEngine)
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

### Gate-Z 5-Sleeve Allocator

| Sleeve | TRENDING | RANGE | STRESS | CRASH |
|--------|----------|-------|--------|-------|
| P1 Carry | 25% | 20% | 15% | 5% |
| P2 Rotation | 25% | 20% | 10% | 5% |
| P3 Trend/LHC | 30% | 20% | 10% | 5% |
| P4 Neutral-Alpha | 10% | 25% | 25% | 20% |
| P5 Down-Offense | 10% | 15% | 40% | 65% |

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

- **ContagionEngine**: L3 graph topology, 21 nodes, 7 shock scenarios, multi-step propagation
- **StatArbEngine**: Medallion mean reversion + cointegration pairs + factor residuals (Σβ≈0)
- **OptionsEngine**: Black-Scholes Greeks, θ+Γ optimizer, vol surface, P4 sleeve allocation
- **DistressedAssetEngine**: 5-model ensemble (Altman Z, Merton KMV, Ohlson O, Zmijewski, ML GBM), fallen angel detector, LGD estimator, Kelly-sized opportunities
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
| T6 | Social | 1.0 | MiroFish sentiment |
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
2. **Paper broker only** — no live execution until broker API connected
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
