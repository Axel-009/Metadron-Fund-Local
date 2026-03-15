# Metadron Capital — AI Investment Platform

## Session Bootstrap

When starting a new session, use this prompt to resume:

> "Continue building the Metadron Capital investment platform. The master repo is at `/home/user/Metadron-Capital` with a 6-layer architecture. All 18 component repos are under `repos/`. Core platform code is in `core/`. Config is in `config/repos.yaml`. Run `python3 core/platform.py` for status."

## Architecture

```
Metadron-Capital/                    ← Master monorepo (Layer 0: Hub)
├── core/                            ← Platform orchestrator, signals, portfolio
│   ├── platform.py                  ← Central orchestrator (19 modules)
│   ├── signals.py                   ← Cyclical vs secular decomposition, money velocity
│   └── portfolio.py                 ← Portfolio analytics engine
├── config/
│   └── repos.yaml                   ← 6-layer module registry
├── tests/
│   └── test_platform.py             ← 11 platform tests
├── modules/                         ← Cross-cutting integration modules (future)
├── docs/                            ← Architecture documentation (future)
└── repos/                           ← All 18 component repositories
    ├── layer1_data/                 ← DATA INGESTION & MARKET INTELLIGENCE
    │   ├── Financial-Data/          ← yfinance — market data pipeline
    │   ├── open-bb/                 ← OpenBB — investment research terminal (30+ providers)
    │   └── hedgefund-tracker/       ← SEC 13F/13D — institutional flow intelligence
    ├── layer2_signals/              ← SIGNAL PROCESSING & FEATURE ENGINEERING
    │   ├── Mav-Analysis/            ← MaverickMCP — technical analysis (39+ tools)
    │   ├── quant-trading/           ← Quant strategies (Bollinger, MACD, SAR, etc.)
    │   ├── stock-chain/             ← Time-series chain analysis
    │   └── CTA-code/                ← CTA/trend-following (PRML, statistical ML)
    ├── layer3_ml/                   ← ML/AI MODELS & PREDICTION
    │   ├── QLIB/                    ← Microsoft Qlib — quantitative ML framework
    │   ├── Stock-prediction/        ← 30+ DL models, 23 trading agents
    │   ├── ML-Macro-Market/         ← Macro indicator → regime classification
    │   └── AI-Newton/               ← Physics-inspired financial modeling (Rust+Python)
    ├── layer4_portfolio/            ← PORTFOLIO CONSTRUCTION & RISK MANAGEMENT
    │   ├── ai-hedgefund/            ← Multi-agent AI hedge fund (LangGraph)
    │   ├── financial-distressed-repo/ ← Company default prediction
    │   └── sophisticated-distress-analysis/ ← Credit scoring (LightGBM, 0.867 AUC)
    ├── layer5_infra/                ← EXECUTION & SERVING INFRASTRUCTURE
    │   ├── Kserve/                  ← Kubernetes ML model serving
    │   ├── nividia-repo/            ← NVIDIA GPU-optimized DL examples
    │   └── Air-LLM/                 ← Memory-efficient LLM inference
    └── layer6_agents/               ← AGENT ORCHESTRATION & DECISION INTELLIGENCE
        └── Ruflo-agents/            ← claude-flow v3.5 — multi-agent orchestration
```

## Key Commands

```bash
# Platform status
cd /home/user/Metadron-Capital && python3 core/platform.py

# Run tests
python3 -m pytest tests/ -v

# Run ai-hedgefund tests
cd repos/layer4_portfolio/ai-hedgefund && python3 -m pytest tests/ -v

# Run QLIB tests
cd repos/layer3_ml/QLIB && python3 -m pytest tests/misc/ tests/ops/ -v
```

## Data Flow

```
L1 (Data) → L2 (Signals) → L3 (ML) → L4 (Portfolio) → L5 (Serve) → L6 (Orchestrate)
    ↑                                                                        ↓
    └──────────────────── L0 Hub (Metadron Core) ←──────────────────────────┘
```

## Development Rules

1. **All changes committed to Metadron-Capital** — this is the master repo
2. **6-layer architecture is immutable** — extend within layers, don't reorganize
3. **Core platform code** lives in `core/` — orchestrator, signals, portfolio
4. **Integration modules** go in `modules/` — cross-layer connectors
5. **Tests must pass** before pushing
6. **Session continuity** — this CLAUDE.md serves as the bootstrap context

## API Keys Required (for live data)

- Financial data: Tiingo, FRED, Finnhub, Financial Datasets
- LLM providers: Anthropic, OpenAI, DeepSeek, Groq, Google
- Search: Exa, Tavily

## Tech Stack

- Python 3.11+ (3.12 preferred for Mav-Analysis)
- Node.js 22+ (Ruflo-agents)
- PyTorch 2.10, TensorFlow 2.21, scikit-learn 1.8
- LangChain/LangGraph, FastAPI, OpenBB
- Microsoft Qlib, yfinance, XGBoost, LightGBM
