# METADRON CAPITAL — SINGLE RUN DATA FLOW CHART

Every engine in the system mapped to its position in a single execution cycle.
This is the authoritative reference for how data flows from market scan to
trade execution to learning feedback.

```
═══════════════════════════════════════════════════════════════════════════════
 STAGE 1: SCAN — Data Ingestion (every tick)
═══════════════════════════════════════════════════════════════════════════════

  EXTERNAL SOURCES                    INTERNAL PIPELINE
  ─────────────────                   ──────────────────

  OpenBB (34+ providers)─┐
  FRED (40+ series)──────┤
  FMP (fundamentals)─────┤     ┌──────────────────────────────┐
  IBKR (real-time quotes)┤────→│  DataIngestionOrchestrator   │
  newsfilter.io (10K+)───┤     │  (multi-asset scheduler)     │
  SEC filings────────────┘     └──────────────┬───────────────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │  DataQualityGate   │
                                    │  stale │ complete │ │
                                    │  outlier checks    │
                                    └─────────┬──────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
                ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
                │ UniverseEngine  │ │UniversalDataPool│ │   NewsEngine    │
                │ S&P 1500 + FTSE│ │ cross-asset      │ │ newsfilter.io   │
                │ 1044 securities │ │ routing to       │ │ + FMP fallback  │
                │ GICS 4-tier     │ │ engine layers    │ │ sentiment score │
                │ 26 RV pairs     │ │                  │ │ urgency rating  │
                └────────┬────────┘ └────────┬─────────┘ └───┬───────┬─────┘
                         │                   │               │       │
                         └───────────┬───────┘               │       │
                                     │                       │       │
                              to Track A              to Track B     │
                              (Cube → signals)        (News+Miro)   │
                                     │                       │       │
═════════════════════════════════════╪═══════════════════════════╪═════════
 STAGE 2: LAYERS — Signal Processing (1-min cadence)            │
═════════════════════════════════════╪═══════════════════════════╪═════════
                                     │                           │
          ┌──────────────────────────▼──────────────────┐        │
          │            FedLiquidityPlumbing              │        │
          │  SOFR │ reserves │ TGA │ ON-RRP │ M2V       │        │
          │  → CubeLiquidityTensor │ MoneyVelocityTracker│       │
          └──────────────────────────┬──────────────────┘        │
                                     │                           │
          ┌──────────────────────────▼──────────────────┐        │
          │              MacroEngine (GMTF)              │        │
          │  ┌────────────────────────────────────────┐  │        │
          │  │ MoneyVelocityModule │ SectorRanker     │  │        │
          │  │ CarryToVolatility   │ RegimeTransition │  │        │
          │  │ YieldCurveAnalyzer  │ CreditPulseMonitor│ │        │
          │  │ MacroFeatureBuilder (50+ features)     │  │        │
          │  └────────────────────────────────────────┘  │        │
          │  → regime │ rm_adjustment │ sector_weights   │        │
          └──────────────────────────┬──────────────────┘        │
                                     │                           │
          ┌──────────────────────────▼──────────────────┐        │
          │         MetadronCube  C(t) = f(L,R,F)       │        │
          │  ┌────────────────────────────────────────┐  │        │
          │  │ L0: FedPlumbingLayer                   │  │        │
          │  │ L1: LiquidityTensor    L(t) ∈ [-1,+1] │  │        │
          │  │ L2: ReserveFlowKernel  ΔRes → ΔSector  │  │        │
          │  │ LR: RiskStateModel     R(t) ∈ [0,1]   │  │        │
          │  │ LF: CapitalFlowModel   F(t) rotation   │  │        │
          │  │ L4: RegimeEngine       HMM+RL 4-regime │  │        │
          │  │ LG: GateZAllocator     5-sleeve alloc  │  │        │
          │  │ LE: GateLogic          4-gate entry     │  │        │
          │  │ LK: KillSwitch ◄═══════════════════════╪══╪═► HALT
          │  │ LC: FCLPLoop           daily recal      │  │        │
          │  └────────────────────────────────────────┘  │        │
          │  → regime │ leverage │ beta_cap │ sleeves    │        │
          └──────────────────────────┬──────────────────┘        │
                                     │                           │
  ┌──────────────────────────────────┤
  │                                  │
  │    TRACK A: SIGNAL ENGINES       │
  │    (from MetadronCube output)    │
  │                                  │
  │  ┌───────────────────┐  ┌───────┴──────────┐
  │  │SecurityAnalysis   │  │ ContagionEngine  │
  │  │Graham-Dodd-Klarman│  │ 21 nodes, 7 shock│
  │  │top-down + bottom  │  │ scenarios, multi-│
  │  │up, MoS ≥33%      │  │ step propagation │
  │  └────────┬──────────┘  └────────┬─────────┘
  │           │                      │
  │  ┌────────▼──────────┐  ┌────────▼──────────┐
  │  │  StatArbEngine    │  │FixedIncomeEngine  │
  │  │  Medallion mean-  │  │ yield curve,      │
  │  │  reversion + co-  │  │ credit spreads,   │
  │  │  integration pairs│  │ duration ladder   │
  │  └────────┬──────────┘  └────────┬──────────┘
  │           │                      │
  │  ┌────────▼──────────┐  ┌────────▼──────────┐
  │  │DistressedAsset    │  │PatternDiscovery   │
  │  │5-model ensemble:  │  │MiroFish agent sim │
  │  │Altman Z, Merton   │  │+ AI-Newton PySR   │
  │  │KMV, Ohlson,       │  │symbolic regression│
  │  │Zmijewski, ML GBM  │  │→ PatternBus       │
  │  └────────┬──────────┘  └────────┬──────────┘
  │           │                      │
  │  ┌────────▼──────────┐  ┌────────▼──────────┐
  │  │AdaptiveThreshold  │  │  VelocityEngine   │
  │  │252-day rolling    │  │  money velocity   │
  │  │percentile calib   │  │  tracking         │
  │  └────────┬──────────┘  └────────┬──────────┘
  │           │                      │
  │           └──────────┬───────────┘
  │                      │
  │     TRACK A signals  │
  │                      │
  │                      │
  └──────────────────────┘


                                        TRACK B: NEWS+MIRO
                                        (independent — feeds directly from
                                         NewsEngine in Stage 1, NOT from Cube)

                         │
                         │ (from NewsEngine in Stage 1 — same instance,
                         │  newsfilter.io + FMP fallback + sentiment +
                         │  urgency rating — no duplicate engine)
                         │
          ┌──────────────▼──────────────────────────┐
          │     News+MiroMomentum Pipeline           │
          │                                          │
          │  NewsEngine.run_miro_on_news_tickers():  │
          │    1. Flag tickers with breaking news    │
          │    2. MiroMomentumEngine per ticker       │
          │    3. Combined = 40% sentiment            │
          │                 + 60% agent sim           │
          │                                          │
          │  Outputs to:                             │
          │    → EventDrivenEngine (enriched)         │
          │    → CVREngine (enriched)                │
          │    → MLVoteEnsemble Tier 6               │
          │    → Direct L7 if combined ≥ 0.7         │
          └──────────────┬───────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     EventDrivenEngine       │
          │  12 categories:             │
          │  M&A arb, PEAD, spinoff,    │
          │  activist, buyback,         │
          │  catalyst, etc.             │
          │  Mitchell-Pulvino M&A       │
          │  SUE PEAD model             │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │        CVREngine            │
          │  5-model valuation:         │
          │  binary option, barrier,    │
          │  milestone tree, Monte      │
          │  Carlo, real options         │
          │  liquidity/credit adj       │
          └──────────────┬──────────────┘
                         │
                  TRACK B signals
                         │


  ═══════════════════════╤═══════════════════════════════════════
                         │
        TRACK A signals + TRACK B signals
                         │
                         ▼
                ALL SIGNALS MERGE
                         │
═══════════════════════════╪══════════════════════════════════════════════
 STAGE 3: INTELLIGENCE — ML/AI Decision Layer (5-min cadence)
═══════════════════════════╪══════════════════════════════════════════════
                        │
     ┌──────────────────▼──────────────────────────────────────┐
     │              FullUniverseScan (async, 4 runs)           │
     │                                                         │
     │  Run 1: S&P 500  ──→ MiroMomentum per ticker           │
     │  Run 2: S&P 400  ──→ MiroMomentum per ticker           │
     │  Run 3: S&P 600  ──→ MiroMomentum per ticker           │
     │  Run 4: ETF + FI ──→ MiroMomentum per ticker           │
     │                                                         │
     │  → aggregate_runs() → dedup → cap enforcement           │
     │  → AllocationSlate                                      │
     └──────────────────┬──────────────────────────────────────┘
                        │
     ┌──────────────────▼──────────────────────────────────────┐
     │              AlphaOptimizer (dual pipeline)             │
     │                                                         │
     │  Standard: XGBoost(60%) + LinReg(40%) + CAPM(20% blend)│
     │            + QualityRanker + 22 features + EWMA cov     │
     │            → SLSQP mean-variance optimization           │
     │                                                         │
     │  Enhanced: WalkForward + FactorLibrary(50+ factors)     │
     │            + sector MVO                                 │
     │                                                         │
     │  → AlphaOutput (signals + weights + Sharpe)             │
     └──────────────────┬──────────────────────────────────────┘
                        │
     CONCURRENT ML MODELS (all feed into MLVoteEnsemble):
     ┌──────────────────┼──────────────────────────────────────┐
     │                  │                                      │
     │  ┌───────────────┴───────────────┐                      │
     │  │  PatternRecognitionEngine     │                      │
     │  │  candlestick, chart, breakout │                      │
     │  └───────────────┬───────────────┘                      │
     │  ┌───────────────┴───────────────┐                      │
     │  │  UniverseClassifier           │                      │
     │  │  XGBoost 4-model, tiers A-G  │                      │
     │  └───────────────┬───────────────┘                      │
     │  ┌───────────────┴───────────────┐                      │
     │  │  DeepLearningEngine           │                      │
     │  │  PPO agent, 50-feature state  │                      │
     │  └───────────────┬───────────────┘                      │
     │  ┌───────────────┴───────────────┐                      │
     │  │  ML Bridges (6):              │                      │
     │  │  FinRL │ MonteCarlo │ StockPred│                     │
     │  │  Markov │ TFT │ DeepTrading   │                      │
     │  └───────────────┬───────────────┘                      │
     │  ┌───────────────┴───────────────┐                      │
     │  │  SectorBotManager (11 GICS)   │                      │
     │  │  + SocialFeatureBuilder       │                      │
     │  │  + ModelEvaluator             │                      │
     │  └───────────────┬───────────────┘                      │
     │                  │                                      │
     └──────────────────┼──────────────────────────────────────┘
                        │
     ┌──────────────────▼──────────────────────────────────────┐
     │          MLVoteEnsemble (10-tier weighted vote)          │
     │                                                         │
     │  T1:  Neural Net (FinRL/DRL)              weight 1.0   │
     │  T2:  Momentum/Mean-Reversion             weight 1.2   │
     │  T3:  Volatility Regime                   weight 0.8   │
     │  T4:  Monte Carlo                         weight 0.9   │
     │  T5:  Quality (SecurityAnalysis grade)    weight 1.1   │
     │  T6:  MiroMomentum (News+AgentSim) ◄──── cross-stage   │
     │  T7:  Distressed                          weight 0.9   │
     │  T8:  EventDriven                         weight 1.0   │
     │  T9:  CVR                                 weight 0.7   │
     │  T10: CreditQuality                       weight 0.9   │
     │                                                         │
     │  → weighted vote score [-10, +10]                       │
     │  → minimum edge = 2.0 + max(0, -vote_score) bps        │
     └──────────────────┬──────────────────────────────────────┘
                        │
═══════════════════════════╪══════════════════════════════════════════════
 STAGE 4: DECISION + EXECUTION
═══════════════════════════╪══════════════════════════════════════════════
                        │
          ┌─────────────▼─────────────────────────────────────┐
          │    DECISION: Two paths (both route to L7)         │
          │                                                    │
          │  PATH A: FullUniverseScan slate                    │
          │    → AllocationEngine.apply_rules()                │
          │    → bucket enforcement (IG/HY/Dist/TLTW/FI/CVR)  │
          │    → BetaCorridor (7-12% return corridor)          │
          │                                                    │
          │  PATH B: DecisionMatrix (6-gate)                   │
          │    G1 Flow/Headlines   20%  ──┐                    │
          │    G2 Macro/Beta       25%  ──┤                    │
          │    G3 Fundamentals     30%  ──┼→ composite ≥ 0.55  │
          │    G4 Momentum/Tech    25%  ──┤     to approve     │
          │    G5 Quality tier          ──┤                    │
          │    G6 Ensemble (10-tier)    ──┘                    │
          │    → KellySizer (1.5x multiplier)                  │
          │                                                    │
          │  DIRECT ROUTES (high conviction, still via L7):    │
          │    NEWS_MIRO_DIRECT  (combined ≥ 0.7)              │
          │    EVENT_DIRECT      (conviction ≥ 0.7)            │
          │    CVR_DIRECT        (STRONG_BUY)                  │
          └─────────────┬─────────────────────────────────────┘
                        │
          ┌─────────────▼─────────────────────────────────────┐
          │  OPTIONS SIZING (edge-gated, runs before L7)      │
          │                                                    │
          │  OptionsSizer:                                     │
          │    1. Black-Scholes theoretical price               │
          │    2. Monte Carlo (10K paths) → win prob            │
          │    3. Fair value = avg(BS, MC) vs market price      │
          │    4. Edge < 200bps → REJECTED (no allocation)     │
          │    5. Kelly criterion sizes on detected edge        │
          │    6. Minimum 5 contracts                           │
          │    → contracts + greeks + edge_bps                  │
          └─────────────┬─────────────────────────────────────┘
                        │
   ╔════════════════════▼═════════════════════════════════════╗
   ║        L7 UNIFIED EXECUTION SURFACE                      ║
   ║        (SOLE entry point — ALL orders route here)        ║
   ║                                                          ║
   ║  ┌───────────────────────────────────────────────────┐   ║
   ║  │ Step 1: Research-only guard (reject FI/FX/credit) │   ║
   ║  └───────────────────┬───────────────────────────────┘   ║
   ║  ┌───────────────────▼───────────────────────────────┐   ║
   ║  │ Step 2: MultiProductRouter.classify()             │   ║
   ║  │         EQUITY │ OPTION │ FUTURE                  │   ║
   ║  └────┬──────────────┬──────────────┬────────────────┘   ║
   ║       │              │              │                    ║
   ║  ┌────▼─────┐  ┌─────▼─────┐  ┌────▼──────┐            ║
   ║  │ EQUITY   │  │  OPTION   │  │  FUTURE   │            ║
   ║  │WonderTrdr│  │OptionsSzr │  │BetaCorrdr │            ║
   ║  │micro-    │  │BS+MC+Kelly│  │hedge val  │            ║
   ║  │price adj │  │edge gate  │  │           │            ║
   ║  └────┬─────┘  └─────┬─────┘  └────┬──────┘            ║
   ║       │              │              │                    ║
   ║  ┌────▼──────────────▼──────────────▼────────────────┐   ║
   ║  │ Step 4: L7RiskEngine — 10 gates (ALL must pass)   │   ║
   ║  │ G1 pos 10% │ G2 sector 30% │ G3 loss 3%          │   ║
   ║  │ G4 gross 250% │ G5 net 150% │ G6 throttle 100    │   ║
   ║  │ G7 dd 10% │ G8 cash │ G9 opt Δ 20% │ G10 fut 50% │   ║
   ║  └───────────────────┬───────────────────────────────┘   ║
   ║  ┌───────────────────▼───────────────────────────────┐   ║
   ║  │ Step 5: SlippageModel (pre-trade cost estimate)   │   ║
   ║  └───────────────────┬───────────────────────────────┘   ║
   ║  ┌───────────────────▼───────────────────────────────┐   ║
   ║  │ Step 6: IBKR ALGO ROUTING                         │   ║
   ║  │                                                    │   ║
   ║  │  Notional > $50K ──→ TWAP (server-side splitting) │   ║
   ║  │  VWAP routing ─────→ VWAP (volume participation)  │   ║
   ║  │  Medium urgency ───→ Adaptive (IBKR auto-selects) │   ║
   ║  │  HIGH/CRITICAL ────→ Market order (immediate fill) │   ║
   ║  │                                                    │   ║
   ║  │  ┌─────────────────────────────────────────────┐   │   ║
   ║  │  │           IBKRBroker                        │   │   ║
   ║  │  │  TWS/Gateway via ib_insync                  │   │   ║
   ║  │  │  Equities + Options + Futures               │   │   ║
   ║  │  │  $0.005/share │ $0.65/option │ $1.25/future │   │   ║
   ║  │  └─────────────────────────────────────────────┘   │   ║
   ║  └───────────────────┬───────────────────────────────┘   ║
   ║  ┌───────────────────▼───────────────────────────────┐   ║
   ║  │ Step 7: Post-trade                                │   ║
   ║  │  → L7RiskEngine.post_trade_update()               │   ║
   ║  │  → TransactionCostAnalyzer.analyze()              │   ║
   ║  │  │   spread │ impact │ timing │ commission        │   ║
   ║  │  → ExecutionLearningLoop.record_outcome()         │   ║
   ║  │  → Trade Log (JSONL recon audit)                  │   ║
   ║  │  → Prometheus metrics (17 gauges/counters)        │   ║
   ║  └───────────────────┬───────────────────────────────┘   ║
   ╚══════════════════════╪═══════════════════════════════════╝
                          │
              EXECUTION OUTCOMES
              (fills, costs, slippage, P&L)
                          │
═════════════════════════╪════════════════════════════════════════════
 STAGE 5: LEARNING + MONITORING (continuous, feeds back to Stages 2-4)
═════════════════════════╪════════════════════════════════════════════
                          │
     ┌────────────────────▼────────────────────────────────────┐
     │              LearningLoop (7 channels)                  │
     │                                                         │
     │  CH1: SIGNAL_ACCURACY   → retune tier weights ─────────┼──→ Stage 3
     │  CH2: EXECUTION_QUALITY → retune slippage model ────────┼──→ Stage 4
     │  CH3: REGIME_FEEDBACK   → retune HMM priors ───────────┼──→ Stage 2
     │  CH4: ALPHA_DECAY       → retune half-life ─────────────┼──→ Stage 3
     │  CH5: RISK_CALIBRATION  → retune gate thresholds ───────┼──→ Stage 4
     │  CH6: AGENT_PERFORMANCE → promote/demote agents ────────┼──→ Stage 3
     │  CH7: CROSS_ASSET       → retune sector allocation ─────┼──→ Stage 2
     │                                                         │
     │  Damping: MAX_WEIGHT_CHANGE = 0.05/cycle                │
     │  Oscillation detection: 3-reversal threshold             │
     └────────────────────┬────────────────────────────────────┘
                          │
     CONCURRENT LEARNING SYSTEMS:
     ┌────────────────────┼────────────────────────────────────┐
     │                    │                                    │
     │  ┌─────────────────┴─────────────────┐                  │
     │  │  GSDPlugin (gradient dynamics)    │                  │
     │  │  signal momentum │ convergence    │                  │
     │  │  divergence │ gradient decay      │                  │
     │  └─────────────────┬─────────────────┘                  │
     │  ┌─────────────────┴─────────────────┐                  │
     │  │  PaulPlugin (pattern memory)      │                  │
     │  │  pattern matching │ evolution     │                  │
     │  │  context-aware replay │ library   │                  │
     │  └─────────────────┬─────────────────┘                  │
     │  ┌─────────────────┴─────────────────┐                  │
     │  │  PaulOrchestrator                 │                  │
     │  │  agent create/promote/demote      │                  │
     │  └─────────────────┬─────────────────┘                  │
     │  ┌─────────────────┴─────────────────┐                  │
     │  │  Agent Hierarchy:                 │                  │
     │  │  FullAgentScorecard               │                  │
     │  │  ├─ DIRECTOR (Sharpe>2.5, acc>85%)│                  │
     │  │  ├─ GENERAL  (Sharpe>2.0, acc>80%)│                  │
     │  │  ├─ CAPTAIN  (Sharpe>1.5, acc>55%)│                  │
     │  │  ├─ LIEUTENANT (Sharpe>1.0)       │                  │
     │  │  └─ RECRUIT  (below thresholds)   │                  │
     │  │                                   │                  │
     │  │  SectorBotManager (11 GICS bots)  │                  │
     │  │  ResearchBotManager (11 research) │                  │
     │  │  GICSSectorAgentManager            │                  │
     │  │  AgentMonitor (4-tier hierarchy)  │                  │
     │  │  InvestorPersonaManager (12)      │                  │
     │  └─────────────────┬─────────────────┘                  │
     │                    │                                    │
     └────────────────────┼────────────────────────────────────┘
                          │
     MONITORING:
     ┌────────────────────┼────────────────────────────────────┐
     │                    │                                    │
     │  AnomalyDetector ──┤── statistical anomaly scan         │
     │  PortfolioAnalytics┤── scenario engine + risk metrics   │
     │  MemoryMonitor ────┤── session tracking + EOD summary   │
     │  20% position DD ──┤── auto-liquidation via L7          │
     │  Circuit breakers ─┘── halt on breach                   │
     │                                                         │
     └─────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
 CROSS-STAGE PIPELINES (span multiple stages)
═══════════════════════════════════════════════════════════════════════════════

  1. NEWS+MIRO PIPELINE:
     Stage 1 (NewsEngine) → Stage 2 (run_miro_on_news_tickers: 40% sentiment
     + 60% agent sim) → Stage 2 (enrich EventDriven + CVR) → Stage 3
     (MLVoteEnsemble T6) → Stage 4 (direct L7 if ≥ 0.7)

  2. KILL SWITCH CHAIN:
     Stage 2 (MetadronCube.KillSwitch) → Stage 4 (AllocationEngine.cube_kill
     _switch_override) → Stage 4 (L7RiskEngine.G7) → HALT all execution

  3. LEARNING FEEDBACK LOOP:
     Stage 5 (LearningLoop) → Stage 2 (regime priors, sector weights)
                             → Stage 3 (tier weights, ensemble tuning)
                             → Stage 4 (gate thresholds, slippage model)

  4. OPTIONS EDGE PIPELINE:
     Stage 3 (signal with market_price) → Stage 4 (OptionsSizer: BS + MC →
     fair value vs market → edge ≥ 200bps? → Kelly sizing → 5+ contracts)
     → Stage 4 (L7 IBKR execution)


═══════════════════════════════════════════════════════════════════════════════
 LEGEND
═══════════════════════════════════════════════════════════════════════════════

  ──→    Data flow direction
  ◄──    Feedback / enrichment
  ═══    Stage boundary
  ╔══╗   L7 Execution Surface (mandatory routing point)
  │  │   Engine / component box
  ┌──┐   Sub-component or group
  HALT   Kill switch termination point

  TIMING:
    Stage 1: Every tick (continuous)
    Stage 2: 1-minute cadence
    Stage 3: 5-minute cadence
    Stage 4: On signal delta (event-driven)
    Stage 5: Continuous (per-fill for learning, 5-min for monitoring)

  TOTAL ENGINES: 70+
  TOTAL ML MODELS: 15+ (10-tier ensemble + bridges + PPO + XGBoost)
  TOTAL SIGNAL TYPES: 29
  SINGLE RUN CYCLE: ~20 minutes (4 universe runs × 150s + 5min aggregate + 5min risk)
```
