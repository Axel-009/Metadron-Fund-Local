# Metadron Capital — Full Architecture

```
═══════════════════════════════════════════════════════════════════════════════
                    METADRON CAPITAL — FULL ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

PHASE 1: DATA
  DataIngestionOrchestrator → UniversalDataPool → UniverseEngine
  ├── Equities: S&P 1500 (SP500 + SP400 + SP600) + FTSE 100
  ├── Fixed Income: G10 + India sovereign, US IG/HY corporate
  ├── Commodities: GLD, SLV, USO, UNG, DBA, DBC, COPX, WEAT, CORN
  ├── Indices: SPY, QQQ, IWM, DIA, VT, EFA, EEM
  ├── Currencies: G10 + INR + JPY
  ├── Econometrics: 40+ FRED series (GDP, CPI, M2, WALCL, SOFR)
  └── Futures: ES, NQ, YM, RTY, VX, ZN, ZB
      Source: OpenBB (34+ providers: FMP, FRED, Polygon, SEC, CBOE)
                                    │
                                    ▼
PHASE 2: SIGNALS (1-min cadence)
  ├── FedLiquidityPlumbing    → SOFR, reserves, TGA, ON-RRP
  ├── MacroEngine (GMTF)      → regime, rm_adjustment, sector_weights
  ├── MetadronCube            → C(t)=f(L,R,F) + 4-Gate + KillSwitch
  │                             → cube kill-switch → AllocationEngine
  ├── SecurityAnalysisEngine  → Graham-Dodd-Klarman (top-down + bottom-up)
  ├── ContagionEngine         → 21 nodes, 7 shock scenarios
  ├── StatArbEngine           → Medallion mean-reversion + cointegration pairs
  ├── FixedIncomeEngine       → yield curve, credit spreads, FI signals
  └── News+MiroMomentum       → newsfilter.io (10K+ sources) + FMP fallback
      │                         → MiroMomentumEngine per news-flagged ticker
      │                         → 40% sentiment + 60% agent sim
      ├──→ EventDrivenEngine (shared singleton, enriched signals)
      └──→ CVREngine (shared singleton, enriched signals)
  ▲                                 │
  │ FEEDBACK: learned tier          ▼
  │ weights adjust signal
  │ engine sensitivity        PHASE 3: INTELLIGENCE (5-min cadence)
  │                           ├── FullUniverseScan (async background, 4 runs):
  │                           │   ├── Run 1: SP500      → MiroMomentum per ticker
  │                           │   ├── Run 2: SP400      → MiroMomentum per ticker
  │                           │   ├── Run 3: SP600      → MiroMomentum per ticker
  │                           │   └── Run 4: ETF+FI     → MiroMomentum per ticker
  │                           │       → aggregate_runs() → AllocationSlate
  │                           │
  │                           ├── AlphaOptimizer (dual pipeline, merged):
  │                           │   ├── Standard: XGBoost(60%) + LinearRegression(40%)
  │                           │   │   + CAPM alpha (20% blend) + QualityRanker
  │                           │   │   + 22 features + EWMA cov → SLSQP
  │                           │   └── Enhanced: WalkForward + FactorLibrary + sector MVO
  │                           │       → AlphaOutput (signals + weights + Sharpe)
  │                           │
  │                           ├── PatternDiscoveryEngine  → MiroFish + AI-Newton
  │                           ├── PatternRecognitionEngine → candlestick, chart, breakouts
  │                           ├── StockPredictionBridge   → neural net predictions
  │                           │
  │                           ├── Feed MLVoteEnsemble (all 10 tiers):
  │                           │   ├── T1: Neural Net        T6: MiroMomentum
  │                           │   ├── T2: Momentum          T7: Distressed
  │                           │   ├── T3: Vol Regime        T8: EventDriven
  │                           │   ├── T4: Monte Carlo       T9: CVR
  │                           │   └── T5: Quality           T10: CreditQuality
  │                           │       → ensemble.vote() — all 10 tiers
  │                           │
  │                           └── SectorBotManager (11 GICS + LLM analysis)
  │                                                 │
  │                                                 ▼
  │                           PHASE 4: DECISION (on signal Δ)
  │                           ├── Path A: FullUniverseScan slate
  │                           └── Path B: DecisionMatrix (6-gate) + AllocationEngine:
  │                               ├── G1: Flow/Headlines (20%)
  │   ┌───────────────────────┐   ├── G2: Macro/Beta (25%)
  │   │ FEEDBACK: learned     │   ├── G3: Fundamentals (30%)
  │   │ gate weights adjust   │   ├── G4: Momentum/Tech (25%)
  │   │ approval thresholds   │   ├── G5: Quality tier
  │   │                       │   └── G6: Ensemble vote (10-tier)
  │   │                       │       → AllocationEngine.apply_rules():
  │   │                       │         IG:40% HY:10% Dist:10% TLTW:15%
  │   │                       │         FI:5% CVR:10% Margin:8% MM:2%
  │   │                       │       → BetaCorridor + OptionsEngine
  │   │                       │                   │
  │   │                       │                   ▼
  │   │                       │   PHASE 5: EXECUTION
  │   │                       │   ├── Standard: approved → L7 Execution Surface
  │   │                       │   │   ├── EQUITY → WonderTrader → Alpaca
  │   │                       │   │   ├── OPTION → OptionsEngine → Alpaca
  │   │                       │   │   └── FUTURE → BetaCorridor → Paper (Rithmic later)
  │   │                       │   └── Direct (high-conviction):
  │   │                       │       ├── News+Miro (≥0.3) → L7 [NEWS_MIRO_DIRECT]
  │   │                       │       ├── EventDriven (≥0.7) → L7 [EVENT_DIRECT]
  │   │                       │       └── CVR (STRONG_BUY) → L7 [CVR_DIRECT]
  │   │                       │                   │
  │   │                       │                   │ EXECUTION OUTCOMES
  │   │                       │                   │ (fills, costs, slippage, P&L)
  │   │                       │                   ▼
  │   │                       │
  │   └───────────────────────┘
  │
  │
═══╪═══════════════════════════════════════════════════════════════════════════
   │            UNIFIED LEARNING PIPELINE (Phases 6-7, continuous)
═══╪═══════════════════════════════════════════════════════════════════════════
   │                                ▲
   │                                │ EXECUTION OUTCOMES feed in:
   │                                │ fills, costs, slippage, realized P&L,
   │                                │ signal accuracy, position drawdowns
   │                                │
   │  PHASE 6: LEARNING ◄──────────┘
   │  ├── LearningLoop:
   │  │   ├── record_signal_outcome() ← from execution fills
   │  │   ├── record_regime_feedback() ← from realized regime accuracy
   │  │   ├── record_sector_feedback() ← from sector P&L
   │  │   ├── compute_tier_weight_adjustments() → retune all 10 tiers
   │  │   └── apply_to_ensemble() → push learned weights back to Phase 3
   │  │
   │  ├── GSDPlugin.update_gradients() → signal gradient dynamics
   │  ├── PaulPlugin.store_pattern() → pattern memory + evolution
   │  ├── PaulOrchestrator → agent creation/promotion/demotion/enforcement
   │  ├── GSDWorkflowBridge → integrate gradient + Paul state
   │  ├── AgentScorecard.update() → DIRECTOR/GENERAL/CAPTAIN/LIEUTENANT/RECRUIT
   │  └── ResearchBotManager.run_llm_sector_analysis() → top 3 sectors via LLM
   │                                │
   │  PHASE 7: MONITORING           │
   │  ├── Portfolio P&L tracking    │
   │  │   ├── Total NAV P&L         │
   │  │   ├── Equities P&L (Alpaca) │
   │  │   ├── Options P&L (Alpaca)  │
   │  │   └── Futures P&L (Paper)   │
   │  ├── 3-Layer Profit-Take (20%) │
   │  │   ├── Alpaca >20% → sell options, re-run
   │  │   ├── Futures >20% → sell futures, re-run
   │  │   └── Aggregate >20% → sell all overlays, re-run
   │  ├── Position 20% drawdown     │
   │  │   exit → liquidation orders │
   │  ├── Portfolio 20% drawdown    │
   │  │   → kill switch, liquidate all, operator reset
   │  ├── Broker integrity recon    │
   │  │   (paper vs Alpaca, $0.01 / 1 share tolerance)
   │  ├── Circuit breaker eval      │
   │  ├── AnomalyDetector.scan()    │
   │  └── PortfolioAnalytics        │
   │                                │
   │                                ▼
   │
═══╪═══════════════════════════════════════════════════════════════════════════
   │           CONTINUOUS LEARNING LOOP (PM2 24/7 service)
   │           3 streams — all feed back into the live pipeline
═══╪═══════════════════════════════════════════════════════════════════════════
   │
   │  Stream 1 — Core ML pipeline:
   │    signal_feedback → pattern_recognition → stock_prediction
   │    (overnight: + model_retrain, pattern_integration, regime_calibration)
   │    │
   │    └──→ FEEDS BACK: retrained alpha weights → Phase 3 AlphaOptimizer
   │         learned signal accuracy → Phase 4 DecisionMatrix gate thresholds
   │
   │  Stream 2 — Autoresearch + Agent learning → Graphify:
   │    autoresearch_check ──┐
   │                         ├──→ graphify_refresh
   │    agent_evolution ─────┘    (god nodes + agent scores +
   │    │                          experiment results + learning metrics)
   │    │
   │    └──→ FEEDS BACK: agent tier weights → Phase 3 ensemble
   │         agent promotion/demotion → Phase 6 scorecard
   │         autoresearch discoveries → Phase 3 feature engineering
   │
   │  Stream 3 — LLM review (overnight, fed DIRECTLY):
   │    autoresearch ──┐
   │                   ├──→ llm_market_review (Brain Power /ensemble)
   │    agent_evolution┘    (does NOT go through graphify)
   │    │
   │    └──→ FEEDS BACK: narrative insights → Phase 2 MacroEngine
   │         model anomaly flags → Phase 3 AlphaOptimizer
   │         strategic positioning → Phase 4 AllocationEngine
   │
   │  ML context sent with every /ensemble call:
   │    alpha_signals, regime, patterns, agent_scores,
   │    pattern_recognition, stock_predictions,
   │    graphify (separate), autoresearch (direct)
   │
   └──→ ALL THREE STREAMS FEED BACK INTO PHASE 1-5 ──→ ↑ (top)

═══════════════════════════════════════════════════════════════════════════════
                    6-LAYER SECURITY (engine/security/)
═══════════════════════════════════════════════════════════════════════════════

  1. PhaseChain         — HMAC-signed output between phases (break = halt new trades)
  2. BrokerIntegrityLock — Paper vs Alpaca reconciliation (> $0.01 = freeze entries)
  3. TransactionLedger  — Append-only HMAC-chained trade log (tamper-evident)
  4. CircuitBreaker     — Perimeter lockdown (200 req/10s → API 503, engine running)
  5. HeartbeatIntegrity — Signed PM2 heartbeats (missing > 5min = warn)
  6. PromptGuard        — LLM token cap (8192) + cadence (30s per caller)

  TokenMeter: per-model usage (Qwen, Llama, Air-LLM, Brain Power)
              4M daily cap with manual override on TECH tab
              Daily report → data/archive/token_usage/*.json.gz

═══════════════════════════════════════════════════════════════════════════════
                    OVERNIGHT SCHEDULE (PM2 cron)
═══════════════════════════════════════════════════════════════════════════════

  8:00 PM ET  → overnight-backtest
               │ walk-forward + QSTrader → LearningLoop → LLM review
               └──→ FEEDS BACK: backtest Sharpe/DD → Phase 3 model selection
                    strategy comparison → Phase 4 allocation rules

  9:00 PM ET  → autoresearch-overnight
               │ Karpathy model training (5-min time budget, cuda:0)
               └──→ FEEDS BACK: best_val_bpb → Phase 3 feature selection
                    architecture improvements → AlphaOptimizer

  2:00 AM     → graphify-nightly
               │ codebase knowledge graph regeneration
               └──→ FEEDS BACK: god nodes → agent context
                    architecture awareness → research bots

═══════════════════════════════════════════════════════════════════════════════
                    PORT MAP (GEX44 deployment)
═══════════════════════════════════════════════════════════════════════════════

  PUBLIC (behind NGINX 80/443):
    5000  Express Frontend
    8001  Engine API (FastAPI, API key auth + internal token + rate limit)

  INTELLIGENCE LAYER (CORS-restricted, internal only):
    8002  LLM Inference Bridge (parallel ensemble orchestrator)
    8003  Air-LLM Model Server (Llama 3.1-8B via Air-LLM framework, cuda:0)
    8004  Qwen 2.5-7B-Instruct (cuda:0)
    8005  Llama 3.1-8B-Instruct (dedicated, fast router, cuda:0)
    8006  OpenJarvis Service (voice + text assistant backed by Llama 3.1-8B)

  MONITORING EXPORTERS (WireGuard-reachable):
    9100  node_exporter (CPU/RAM/disk/network)
    9113  nginx-prometheus-exporter
    9209  pm2-prometheus-exporter
    9835  nvidia_gpu_exporter
    19999 Netdata (real-time system + GPU)

  VPN:
    51820 WireGuard (encrypted tunnel to Contabo monitoring)

═══════════════════════════════════════════════════════════════════════════════
                    34 FRONTEND TABS (7 categories)
═══════════════════════════════════════════════════════════════════════════════

  CORE:         VAULT, SECURITY, LIVE, WRAP, OPENBB, VELOCITY, CUBE
  TRANSACTIONS: ALLOC, THINKING, RISK, MARGIN, RECON
  PRODUCTS:     ETF, MACRO, FIXED INC, FUTURES
  AGENTS:       AGENTS, CHAT, TECH, GRAPHIFY, OPEN JARVIS
  ANALYSIS:     STRAT, QUANT, ARB, BACKTEST
  SIMULATION:   MC SIM, SIM, ML, ML MODELS
  REPORTING:    TXLOG, TCA, REPORTS, ARCHIVE

═══════════════════════════════════════════════════════════════════════════════
```
