# Metadron Capital — AI Investment Platform

## Session Bootstrap

> "Continue building the Metadron Capital investment platform. The master repo is at `/home/user/Metadron-Capital` with a 6-layer architecture. The investment engine is in `engine/`. All 20 component repos are under `repos/`. Core platform code is in `core/`. Run `python3 -m pytest tests/ -v` to verify. Run `python3 run_open.py` to execute the full signal pipeline."

## Investment Thesis

Beta managed within a 7–12% return corridor (S&P 500 historical earnings range).
Alpha extracted through top-down macro → GICS sector selection → bottom-up stock picking.
Money velocity (V = GDP/M2) + cyclical vs secular decomposition drive regime classification.
Platform hunts for 100% alpha with no decay via continuous ML walk-forward optimisation.
Paper broker mode (Yahoo Finance) — not connected to a live broker.

## Signal Pipeline

```
UniverseEngine → MacroEngine → MetadronCube → AlphaOptimizer → ExecutionEngine
     (L1)           (L2)          (L2)            (L3)             (L5)
```

## Architecture

```
Metadron-Capital/                        ← Master monorepo (Layer 0: Hub)
├── engine/                              ← INVESTMENT ENGINE (new)
│   ├── data/                            ← L1: Unified Yahoo data + universe
│   │   ├── universe_engine.py           ← S&P 500/400/600 + GICS taxonomy + 70+ ETFs + 26 RV pairs
│   │   └── yahoo_data.py               ← Single data source: yfinance
│   ├── signals/                         ← L2: Signal processing
│   │   ├── macro_engine.py             ← GMTF money velocity + regime classification (Dataset 3)
│   │   └── metadron_cube.py            ← C(t) = f(L_t, R_t, F_t) + Gate-Z 5-sleeve allocator
│   ├── ml/                              ← L3: ML/AI models
│   │   └── alpha_optimizer.py           ← Walk-forward ML alpha + mean-variance (Dataset 2)
│   ├── portfolio/                       ← L4: Portfolio construction
│   │   └── beta_corridor.py            ← Beta corridor 7–12% + vol-normalisation (Dataset 1)
│   ├── execution/                       ← L5: Execution
│   │   ├── paper_broker.py             ← Simulated broker (Yahoo prices, no live connection)
│   │   └── execution_engine.py         ← Full pipeline orchestrator + ML vote ensemble
│   ├── agents/                          ← L6: Agent orchestration
│   │   └── sector_bots.py             ← 11 GICS sector micro-bots + scorecard
│   └── monitoring/                      ← Monitoring layer
│       └── daily_report.py             ← Open/close reports + sector heatmap
├── core/                                ← Platform orchestrator (original)
│   ├── platform.py                      ← Central orchestrator (20 modules)
│   ├── signals.py                       ← Cyclical vs secular decomposition, money velocity
│   └── portfolio.py                     ← Portfolio analytics engine
├── config/
│   └── repos.yaml                       ← 6-layer module registry
├── tests/
│   ├── test_platform.py                 ← 11 core tests
│   └── test_engine.py                  ← 37 engine tests (48 total)
├── run_open.py                          ← 09:30 ET — full pipeline execution
├── run_close.py                         ← 16:00 ET — EOD reconciliation
├── bootstrap.py                         ← Session health check
└── repos/                               ← All 20 component repositories
    ├── layer1_data/
    │   ├── Financial-Data/              ← yfinance market data pipeline
    │   ├── open-bb/                     ← OpenBB investment research terminal
    │   ├── hedgefund-tracker/           ← SEC 13F institutional flow tracker
    │   ├── FRB/                         ← Federal Reserve Bank data (FRED)
    │   └── EquityLinkedGICPooling/      ← GIC pooling methodology
    ├── layer2_signals/
    │   ├── Mav-Analysis/                ← MaverickMCP technical analysis (39+ tools)
    │   ├── quant-trading/               ← Quant strategies (Bollinger, MACD, SAR)
    │   ├── stock-chain/                 ← Time-series chain analysis
    │   └── CTA-code/                    ← CTA/trend-following (PRML, statistical ML)
    ├── layer3_ml/
    │   ├── QLIB/                        ← Microsoft Qlib quantitative ML framework
    │   ├── Stock-prediction/            ← 30+ DL models, 23 trading agents
    │   ├── ML-Macro-Market/             ← Macro → regime classification
    │   └── AI-Newton/                   ← Physics-inspired financial modeling (Rust+Python)
    ├── layer4_portfolio/
    │   ├── ai-hedgefund/                ← Multi-agent AI hedge fund (LangGraph, 18 agents)
    │   ├── financial-distressed-repo/   ← Company default prediction
    │   └── sophisticated-distress-analysis/ ← Credit scoring (LightGBM)
    ├── layer5_infra/
    │   ├── Kserve/                      ← Kubernetes ML model serving
    │   ├── nividia-repo/                ← NVIDIA GPU-optimized DL
    │   └── Air-LLM/                     ← Memory-efficient LLM inference
    └── layer6_agents/
        └── Ruflo-agents/                ← claude-flow multi-agent orchestration
```

## MetadronCube — C(t) = f(L_t, R_t, F_t)

| Axis | Range | Meaning |
|------|-------|---------|
| L(t) Liquidity | [-1,+1] | Monetary expansion/contraction |
| R(t) Risk | [0,1] | VIX + realized vol + credit spread |
| F(t) Flow | scalar | Sector rotation momentum |

### Regime Parameters

| Regime | Leverage | Beta Cap | Equity % | Hedge % |
|--------|----------|----------|----------|---------|
| TRENDING | 2.5x | 0.65 | 50% | 5% |
| RANGE | 2.0x | 0.30 | 35% | 15% |
| STRESS | 1.5x | 0.10 | 15% | 30% |
| CRASH | 0.8x | -0.20 | 5% | 45% |

### Gate-Z 5-Sleeve Allocator

| Sleeve | TRENDING | RANGE | STRESS | CRASH |
|--------|----------|-------|--------|-------|
| P1 Directional Equities | 50% | 30% | 15% | 5% |
| P2 Factor Rotation | 20% | 25% | 15% | 5% |
| P3 Commodities/Macro | 10% | 15% | 20% | 20% |
| P4 Options Convexity | 10% | 15% | 20% | 25% |
| P5 Hedges/Volatility | 10% | 15% | 30% | 45% |

## Beta Corridor (Dataset 1)

```
ALPHA = 2% (secular alpha headstart)
R_LOW, R_HIGH = 7%, 12% (Gamma Corridor)
BETA_MAX = 2.0, BETA_INV = -0.136
EXECUTION_MULTIPLIER = 4.7
Vol-normalisation: 15% thesis standard
```

## Key Commands

```bash
# Run full pipeline (morning open)
cd /home/user/Metadron-Capital && python3 run_open.py

# Run all tests (48 tests)
python3 -m pytest tests/ -v

# Platform health check
python3 bootstrap.py

# Core platform status
python3 core/platform.py
```

## Design Rules

1. **All data via yfinance** — unified, free, no broker dependency
2. **Paper broker only** — no live execution until broker API connected
3. **6-layer architecture is immutable** — extend within layers
4. **Beta managed within 7–12% corridor** — vol-normalised
5. **Alpha hunted with zero decay** — ML walk-forward, continuous retraining
6. **All agents use Anthropic API** (claude-opus-4-6) when LLM needed
7. **Pure-numpy fallbacks** — no bridge crashes if ML framework missing
8. **try/except on ALL external imports** — system runs degraded, never broken
9. **Tests must pass** before pushing
10. **Session continuity** — this CLAUDE.md serves as the bootstrap context

## 15 Signal Types

MICRO_PRICE_BUY/SELL, RV_LONG/SHORT, FALLEN_ANGEL_BUY,
ML_AGENT_BUY/SELL, DRL_AGENT_BUY/SELL, TFT_BUY/SELL,
MC_BUY/SELL, QUALITY_BUY/SELL, HOLD

## Agent Scorecard

| Tier | Requirements | Promotion |
|------|-------------|-----------|
| TIER_1 General | Sharpe >2.0, accuracy >80% | 4 consecutive top weeks |
| TIER_2 Captain | Sharpe >1.5, accuracy >55% | - |
| TIER_3 Lieutenant | Sharpe >1.0, accuracy >50% | - |
| TIER_4 Recruit | Below thresholds | Demoted after 2 bottom weeks |

Weekly score: 40% accuracy + 30% Sharpe + 30% hit rate
