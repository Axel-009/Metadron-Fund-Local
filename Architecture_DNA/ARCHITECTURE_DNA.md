# METADRON CAPITAL — COMPLETE ARCHITECTURE DNA

## Mission
**$1,000 → $100,000 in 100 days** (~4.6% daily compound return)
Target 95%+ alpha. Compete with and outperform the Medallion fund.
Paper broker mode (OpenBB data). Beta managed within 7–12% corridor.

---

## MASTER SIGNAL PIPELINE (Execution Order)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MORNING OPEN (09:30 ET)                            │
│                                                                            │
│  UniverseEngine ─→ MacroEngine ─→ MetadronCube ─→ CrossAssetCorr          │
│       (L1)            (L2)           (L2)          (L2/3.5)               │
│         │               │              │                │                  │
│         ▼               ▼              ▼                ▼                  │
│  SocialPrediction ─→ DistressedAssets ─→ CVR ─→ EventDriven              │
│     (L2/L3)              (L2)          (L2)       (L2)                    │
│         │                                           │                     │
│         ▼                                           ▼                     │
│  CreditQuality ─→ TickerSelection ─→ AlphaOptimizer ─→ BetaCorridor     │
│     (L3/3.95)         (L2/4)            (L3)              (L4)            │
│                                           │                               │
│                                           ▼                               │
│  DecisionMatrix ─→ ExecutionEngine ─→ PaperBroker                        │
│      (L5)             (L5)             (L5)                               │
│                                        │                                  │
│                                        ▼                                  │
│  ContagionEngine   StatArbEngine   OptionsEngine   SectorBots             │
│  ResearchBots      AgentScorecard  PlatinumReport  PortfolioReport        │
│  SectorTracker     AnomalyDetector MemoryMonitor   MarketWrap             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVENING CLOSE (16:00 ET)                            │
│                                                                            │
│  EOD Reconciliation → Agent Scorecard → Missed Opportunities (>20%)       │
│  Contagion Systemic Risk → Stat Arb Pair Status → Anomaly Detection       │
│  Research Bot DNA Report → Weekly Scorecard (Fridays)                      │
│  Conviction Override Audit → Platinum Report (close) → Market Wrap        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## LAYER 1: DATA INGESTION — UniverseEngine + OpenBB Data

### UniverseEngine (`engine/data/universe_engine.py`)
- **150+ securities** across GICS 4-tier hierarchy
- **11 GICS Sector ETFs**: XLE, XLB, XLI, XLY, XLP, XLV, XLF, XLK, XLC, XLU, XLRE
- **26 Relative Value pairs** (e.g., XLK/XLE, XLF/XLU, XLY/XLP)
- **GSIB basket**: JPM, GS, MS, BAC, C, WFC + 5 international banks
- **Macro tickers**: SPY, QQQ, IWM, TLT, GLD, USO, UUP, HYG, LQD, VXX
- Methods: `get_by_sector()`, `get_rv_pairs()`, `get_sectors()`, `get_gsib_basket()`

### OpenBB Data (`engine/data/yahoo_data.py` → re-exports from `openbb_data.py`)
- **Single data source**: All data via OpenBB (34+ providers) — unified, no broker dependency
- `get_adj_close(tickers, start, end)` → Adjusted close prices
- `get_returns(tickers, start, end)` → Daily log returns
- `get_prices(tickers, start)` → Full OHLCV data
- `get_macro_data(start)` → VIX, S&P 500, 10Y Yield, 5Y Yield, Gold, HY Corporate, IG Corporate
- `get_market_stats()` → SPY return, VIX level, current volatility

---

## LAYER 2: SIGNAL PROCESSING — MacroEngine

### MacroEngine (`engine/signals/macro_engine.py`)
**Purpose**: Regime classification + GMTF money velocity analysis

#### Regime Classification
```
MarketRegime: BULL | BEAR | TRANSITION | STRESS | CRASH
CubeRegime:   TRENDING | RANGE | STRESS | CRASH
```

**Classification logic**:
- VIX > 35 → CRASH
- VIX > 28 AND SPY_3M < -10% → STRESS
- SPY_1M > 3% AND SPY_3M > 5% → BULL
- SPY_1M < -3% AND SPY_3M < -5% → BEAR
- Else → TRANSITION

#### GMTF (Global Monetary Tension Framework)
**SDR basket weights** (IMF 2022): USD=0.4338, EUR=0.2931, CNY=0.1228, JPY=0.0759, GBP=0.0744

**Base tension**: `θ = (M2/GDP) × (1+unemployment) × GDP_growth`

**4 Non-linear gamma multipliers**:
| Gamma | Trigger | Multiplier |
|-------|---------|------------|
| Γ_Liquidity | M2 growth > 2× GDP growth | 1 + sigmoid × 1.5 |
| Γ_FX | USD yield spike > 25bps | 1 - sigmoid × 0.5 |
| Γ_Wage | Unemployment < 4.2% | 1 + sigmoid × 2.0 |
| Γ_Reserve | USD share < 57% | 1 + sigmoid × 1.5 |

**Sigmoid**: `σ(x) = 1 / (1 + exp(-15 × (x - threshold)))`

**Aggregation**: `GMTF = Σ(SDR_weight × θ × Γ_L × Γ_W × Γ_FX × Γ_S)`, smoothed with 5-day rolling mean

#### Sub-Modules (7 total)

1. **MoneyVelocityModule** — Fisher V=GDP/M2, credit impulse, TED spread, SOFR tracking
   - Liquidity Score 0-100: velocity(25%) + credit_impulse(20%) + TED(20%) + VIX(20%) + yield_curve(15%)

2. **SectorRanker** — Macro-adjusted momentum ranking
   - Composite = `Sharpe × regime_multiplier × (1 + relative_strength × 0.5)`
   - Blended momentum: 50% 3M + 30% 1M + 20% 6M
   - Regime multipliers: BULL cyclicals ×1.25, STRESS defensives ×1.4
   - Factor rotation: MOMENTUM | VALUE | QUALITY | DEFENSIVE | NEUTRAL

3. **CarryToVolatility** — G10 FX carry-to-vol signals
   - `CtV = carry_spread / realised_vol(FX)`
   - Gate: CtV > 0.5 AND no stop-loss active
   - Stop-loss: vol spike > 2σ in 1 day
   - SDR-weighted aggregate CtV

4. **RegimeTransitionDetector** — Markov transition with hysteresis (3 days)
   - Base transition probabilities (5×5 matrix)
   - Confidence = 60% frequency + 40% base probability

5. **YieldCurveAnalyzer** — 2s10s, 3m10y, term premium, real rate
   - Curve shapes: STEEP | FLAT | INVERTED | BEAR_FLAT | BULL_STEEP
   - Recession probability via NY Fed probit: `P = Φ(-0.5333 - 0.6330 × spread_3m10y)`

6. **CreditPulseMonitor** — HY-IG spread, z-score, credit impulse
   - Thresholds: NORMAL(<3.0) | ELEVATED(3.0-4.5) | STRESS(4.5-6.0) | CRISIS(>8.0)

7. **MacroFeatureBuilder** — 50+ features for ML
   - VIX(5) + Equity(8) + Yield(8) + Credit(5) + Commodity(6) + Momentum(8) + Volatility(5) + Regime(7+)

#### Output: `MacroSnapshot`
```python
regime, vix, spy_return_1m, spy_return_3m, yield_10y, yield_2y,
yield_spread, credit_spread, gold_momentum, sector_rankings,
gmtf_score, money_velocity_signal, cube_regime
```

---

## LAYER 2: SIGNAL PROCESSING — MetadronCube

### MetadronCube (`engine/signals/metadron_cube.py`)
**Purpose**: `C(t) = f(L_t, R_t, F_t)` — 10-layer intelligence tensor

#### 10 Layers

| # | Layer | Function | Output |
|---|-------|----------|--------|
| 0 | FedPlumbingLayer | SOFR, HY spreads, M2V, TGA, ON-RRP, SOMA | Fed plumbing state |
| 1 | LiquidityTensor | Fed→PD→GSIB→Shadow bank reserve routing | L(t) ∈ [-1, +1] |
| 2 | ReserveFlowKernel | TVP: ΔReserves → ΔSector β at t+1..t+10 | Sector beta projections |
| R | RiskStateModel | VIX + realized vol + credit + skew | R(t) ∈ [0, 1] |
| F | CapitalFlowModel | Sector momentum, leader/laggard, rotation | F(t) flow state |
| 4 | RegimeEngine | HMM+RL → 4 regimes | TRENDING/RANGE/STRESS/CRASH |
| G | GateZAllocator | 5-sleeve capital allocation | Sleeve weights |
| E | GateLogic | 4-gate entry scoring | Entry scores |
| K | KillSwitch | Auto-derisking triggers | Kill signal |
| C | FCLPLoop | Daily recalibration | Updated allocations |

#### Regime Parameters (95% Alpha Target)

| Regime | Max Leverage | β Cap | β Burst | Equity % | Hedge % | Crash Floor | θ Budget/day |
|--------|-------------|-------|---------|----------|---------|-------------|-------------|
| TRENDING | 3.0× | 0.65 | 0.70 | 55% | 5% | ≥+25% | 0.15% |
| RANGE | 2.5× | 0.45 | 0.55 | 40% | 12% | ≥+25% | 0.10% |
| STRESS | 1.5× | 0.15 | 0.20 | 20% | 30% | ≥+25% | 0.05% |
| CRASH | 0.8× | -0.20 | -0.10 | 5% | 50% | ≥+25% | 0.02% |

#### Gate-Z 5-Sleeve Allocator

| Sleeve | Purpose | TRENDING | RANGE | STRESS | CRASH |
|--------|---------|----------|-------|--------|-------|
| P1 Carry | Quality defensives + income | 25% | 20% | 15% | 5% |
| P2 Rotation | Factor/sector RV | 25% | 20% | 10% | 5% |
| P3 Trend/LHC | Durable compounders + collars | 30% | 20% | 10% | 5% |
| P4 Neutral-Alpha | Pairs, basis, dispersion (β≈0) | 10% | 25% | 25% | 20% |
| P5 Down-Offense | Put-spreads + VIX calendars | 10% | 15% | 40% | 65% |

#### 4-Gate Entry Logic

| Gate | Weight | Function |
|------|--------|----------|
| G1 Flow/Headlines | 20% | ETF creations + Tensor signal → shortlist |
| G2 Macro/Beta | 25% | Kernel projections + rates/FX betas → filter |
| G3 Fundamentals | 30% | Quality/ROIC/FCF + supply-chain penalty |
| G4 Momentum/Tech | 25% | Breadth/leadership/gamma/vanna confirms |

#### Kill-Switch Matrix
```
IF HY OAS +35bp AND VIX term flat/inverted AND breadth < 50%
THEN → auto β ≤ 0.35, max tail spend
```

#### FCLP (Full Calibration Learning Protocol)
1. Ingest plumbing → 2. Recompute Tensor/Kernel → 3. Regime detect →
4. Gate scoring → 5. Risk pass → 6. Write allocations

#### Output: `CubeOutput`
```python
regime, max_leverage, beta_cap, beta_burst, risk_budget,
liquidity_state, risk_state, flow_state, sleeve_allocation,
entry_scores, kill_switch_active, crash_floor
```

---

## LAYER 2: ADDITIONAL SIGNAL ENGINES

### SocialPredictionEngine (`engine/signals/social_prediction_engine.py`)
**Reference**: MiroFish repo — OASIS social simulation
- Parses `actions.jsonl` from MiroFish simulation (12 action types per agent per round)
- Builds agent behavioral profiles
- Computes topic-level sentiment (BULLISH/BEARISH/MOMENTUM/REVERSAL)
- Maps topics → tickers via `TOPIC_TICKER_MAP`
- Detects narrative regime (trending/reversing/stable)
- **Output**: `SocialSnapshot` → feeds into Tier-6 of ML Vote Ensemble

### DistressedAssetEngine (`engine/signals/distressed_asset_engine.py`)
**Reference**: FinancialDistressPrediction, financial-distressed-repo, sophisticated-distress-analysis
- **5-model ensemble**:
  1. Altman Z-Score: Z = 1.2×WC/TA + 1.4×RE/TA + 3.3×EBIT/TA + 0.6×MV/TL + 1.0×S/TA
  2. Merton KMV: Distance-to-default via option pricing on firm equity
  3. Ohlson O-Score: Logistic regression on financial ratios
  4. Zmijewski: Probit model on financial ratios
  5. ML GBM: Gradient boosted machine on combined features
- Fallen angel detector (investment grade → high yield transitions)
- LGD (Loss Given Default) estimator
- Kelly-sized opportunity generation
- **Signals**: DISTRESS_FALLEN_ANGEL, DISTRESS_RECOVERY, DISTRESS_AVOID

### CVREngine (`engine/signals/cvr_engine.py`)
**Reference**: ai-hedgefund, Financial-Data
- **5-model CVR valuation**:
  1. Binary Option (Black-Scholes digital)
  2. Barrier Option (knock-in/knock-out)
  3. Milestone Tree (decision tree valuation)
  4. Monte Carlo (path simulation)
  5. Real Options (expansion/abandonment value)
- Liquidity + credit adjustments
- 4 live CVR instruments tracked
- **Signals**: CVR_BUY, CVR_SELL

### EventDrivenEngine (`engine/signals/event_driven_engine.py`)
**Reference**: TradeTheEvent, CTA-code, quant-trading
- **12 event categories**: M&A/merger arb, PEAD (post-earnings drift), spinoffs, restructuring, activist, regulatory, share buyback, secondary offering, index rebalance, management change, dividend, catalyst
- Mitchell-Pulvino M&A arbitrage model
- SUE (Standardized Unexpected Earnings) PEAD model
- Kelly-sized positions
- 10 live event opportunities tracked
- **Signals**: EVENT_MERGER_ARB, EVENT_PEAD_LONG, EVENT_PEAD_SHORT, EVENT_CATALYST

### ContagionEngine (`engine/signals/contagion_engine.py`)
- **21-node graph**: 6 GSIB banks + 11 GICS sectors + 4 macro assets
- Pure adjacency-matrix (no NetworkX)
- **7 shock scenarios**: BANK_RUN, CREDIT_CRUNCH, SOVEREIGN_CRISIS, SECTOR_ROTATION, LIQUIDITY_FREEZE, COMMODITY_SHOCK, CORRELATION_SPIKE
- Multi-step propagation with dampening
- Portfolio contagion risk scoring

### StatArbEngine (`engine/signals/stat_arb_engine.py`)
- Medallion-style mean reversion + cointegration pairs
- Factor residual extraction (Σβ ≈ 0)
- Z-score entry/exit thresholds
- **Signals**: RV_LONG, RV_SHORT, MICRO_PRICE_BUY, MICRO_PRICE_SELL

---

## LAYER 3: ML/AI — AlphaOptimizer

### AlphaOptimizer (`engine/ml/alpha_optimizer.py`)
**Purpose**: Walk-forward ML alpha prediction + mean-variance portfolio construction

#### Pipeline
1. **Feature engineering** (20+ features): market mean/vol, momentum (1w/1m/3m), RSI-14, MACD (12/26/9), Bollinger Band width/position, ATR-14, skewness, kurtosis, cross-sectional dispersion, momentum acceleration, vol ratio
2. **Walk-forward regression**: Linear → Ridge → XGBoost/GBR fallback
3. **EWMA covariance**: λ=0.94, span=60 days
4. **Mean-variance optimization**: SLSQP with turnover constraints (max 50%)
5. **Quality tier classification** (A-G)

#### Quality Tiers
| Tier | Min Sharpe | Min Momentum |
|------|-----------|-------------|
| A | ≥ 2.0 | ≥ 15% |
| B | ≥ 1.5 | ≥ 10% |
| C | ≥ 1.0 | ≥ 5% |
| D | ≥ 0.5 | ≥ 0% |
| E | ≥ 0.0 | ≥ -5% |
| F | ≥ -0.5 | ≥ -10% |
| G | Below all | Below all |

#### Extended Components
- **CAPMAlphaExtractor**: Jensen's alpha, multi-factor decomposition (market, size, value, momentum, quality)
- **FactorLibrary**: 50+ factors across 6 categories (momentum×10, value×10, quality×10, volatility×10, technical×10, fundamental×10)
- **WalkForwardOptimizer**: Rolling window with XGBoost/Ridge fallback
- **MeanVarianceOptimizer**: EWMA cov + turnover + position limits + risk budgeting
- **AlphaDecayModel**: Half-life estimation, projected alpha at 5d/20d
- **TransactionCostModel**: Spread + market impact + commission
- **FeatureImportanceTracker**: Feature importance over time

#### Output: `AlphaOutput`
```python
signals (list[AlphaSignal]), optimal_weights, expected_annual_return,
annual_volatility, sharpe_ratio, max_drawdown, rebalance_cost, alpha_predictions
```

### SocialFeatureBuilder (`engine/ml/social_features.py`)
- Sentiment momentum (EMA fast/slow, MACD, z-score)
- Engagement velocity + acceleration
- Consensus strength + influence Gini coefficient
- Composite `social_alpha` signal → feeds into ML models

### PatternRecognition (`engine/ml/pattern_recognition.py`)
- Candlestick patterns (hammer, engulfing, doji, etc.)
- Chart patterns (head-and-shoulders, double top/bottom)
- Statistical anomaly detection

### Backtester (`engine/ml/backtester.py`)
- Walk-forward backtesting
- Monte Carlo simulation (10,000 paths)
- Scenario engine (historical stress scenarios)

### UniverseClassifier (`engine/ml/universe_classifier.py`)
**Purpose**: XGBoost 4-model soft-voting ensemble for quality tier classification (A-G)
- **4 models**: GaussianNB (15%), GradientBoosting (25%), RandomForest (25%), XGBoost (35%)
- **T3.1 hyperparams**: n_estimators=120, max_depth=6, lr=0.1, gamma=0, reg_lambda=10
- **16 features**: Sharpe, momentum (3M/6M), volatility, max drawdown, ROE, D/E, interest coverage, current ratio, revenue growth, earnings stability, FCF yield, gross margin, Piotroski F, Altman Z, beta
- **CreditQualityClassifier**: 6-factor weighted model (AAA-D credit ratings)
  - Interest coverage (25%), D/E ratio (20%), ROE (15%), earnings stability (15%), Altman Z (15%), FCF yield (10%)
- **Reconciliation engine**: top-down vs bottom-up tier divergence detection
  - Rising stars: observable tier better than fundamental
  - Fallen angels: fundamental tier better than observable
- Feeds credit scores into T10 of MLVoteEnsemble

### ModelEvaluator (`engine/ml/model_evaluator.py`)
- Per-class precision/recall/F1 scoring
- Confusion matrix with tier-aware distance weighting
- Walk-forward evaluation metrics

### DeepLearningEngine (`engine/ml/deep_learning_engine.py`)
**Purpose**: Pure-numpy PPO agent for trading decisions
- 50-feature state vector (no external ML framework dependency)
- Actor-Critic architecture with GAE (Generalized Advantage Estimation)
- Trading environment with position management and transaction costs

### ML Model Bridges (`engine/ml/bridges/`)
Signal bridges that produce SignalType values for MLVoteEnsemble:
- **FinRLBridge**: FinRL deep RL framework adapter → DRL_AGENT signals
- **NvidiaTFTAdapter**: NVIDIA Temporal Fusion Transformer → TFT signals
- **MonteCarloBridge**: Monte Carlo simulation → MC signals
- **StockPredictionBridge**: Stock prediction model → ML_AGENT signals
- **DeepTradingFeatureBuilder**: Deep trading feature engineering
- **KServeAdapter**: KServe ML model serving → remote inference

All bridges follow: try/except on imports, pure-numpy fallbacks, graceful degradation.

---

## LAYER 4: PORTFOLIO — BetaCorridor + DecisionMatrix

### BetaCorridor (`engine/portfolio/beta_corridor.py`)
**Purpose**: Manage portfolio beta within 7%-12% return corridor

#### Core Parameters
```
ALPHA = 2% (secular alpha headstart)
R_LOW = 7%, R_HIGH = 12% (Gamma Corridor)
BETA_MAX = 2.0, BETA_INV = -0.136 (hedge floor)
EXECUTION_MULTIPLIER = 4.7
VOL_STANDARD = 0.15 (thesis standard 15%)
VaR ≤ $0.30M (95%/1-day) on $20M NAV
```

#### Corridor Function (piecewise linear)
```
Rm < R_LOW  → β = BETA_INV (short bias / hedge)
R_LOW ≤ Rm ≤ R_HIGH → β = linear(0 → BETA_MAX)
Rm > R_HIGH → β = BETA_MAX (full throttle)
```

#### Vol Regime Classifier
| Regime | Percentile | β Multiplier | β Cap |
|--------|-----------|-------------|-------|
| LOW_VOL | <25th | 1.20× | 2.0 |
| NORMAL | 25th-75th | 1.00× | 2.0 |
| ELEVATED | 75th-95th | 0.65× | 1.0 |
| CRISIS | >95th | 0.30× | 0.3 |

#### Beta Smoothing Pipeline
Raw target → EMA(α=0.3) → Anti-whipsaw(2% threshold) → Rate limiter(±0.25/cycle) → Kalman filter

### DecisionMatrix (`engine/execution/decision_matrix.py`)
**Purpose**: 6-gate trade approval + Kelly sizing + ABU beta management

#### 6 Approval Gates (weighted)
| Gate | Weight | Threshold | Function |
|------|--------|-----------|----------|
| ALPHA_QUALITY | 25% | 0.50 | Alpha signal strength, quality tier |
| REGIME_ALIGNMENT | 20% | 0.45 | Alignment with MetadronCube regime |
| RISK_BUDGET | 20% | 0.40 | VaR / leverage / drawdown headroom |
| CONVICTION_SCORE | 15% | 0.50 | ML ensemble vote + agent consensus |
| MOMENTUM_CONFIRM | 10% | 0.35 | RSI, MACD, breakout confirmation |
| LIQUIDITY_CHECK | 10% | 0.30 | ADV / spread / executable size |

**MIN_COMPOSITE_SCORE = 0.55** (must pass to approve trade)

#### Regime Alignment
| Regime | Long Modifier | Short Modifier |
|--------|--------------|----------------|
| TRENDING | 1.0 | 0.3 |
| RANGE | 0.6 | 0.6 |
| STRESS | 0.3 | 0.8 |
| CRASH | 0.1 | 1.0 |

#### KellySizer
```
Standard: f* = (p × b - q) / b
  where p = win_prob, b = win/loss ratio, q = 1-p
Aggressive: f_agg = f* × 1.5
Vol-scaled: f_final = f_agg × (VOL_STANDARD / vol)
Capped at: MAX_SINGLE_POSITION_PCT = 20%
```

#### AlphaBetaUnleashed (1-minute cadence beta)
```
Rm_adjusted = Rm_realized + macro.rm_adjustment
target_beta = corridor_fn(Rm_adjusted) × 4.7 × vol_adj
MES_hedge_beta = target_beta - sleeve_beta
```

---

## LAYER 5: EXECUTION ENGINE

### ExecutionEngine (`engine/execution/execution_engine.py`)
**Purpose**: Full pipeline orchestrator + ML vote ensemble + risk gates

#### ML Vote Ensemble (10 tiers, each votes ±1)
| Tier | Name | Weight | Logic |
|------|------|--------|-------|
| 1 | Pure-numpy 2-layer net | 1.0 | Feature → hidden(20) → output, sigmoid activation |
| 2 | Momentum/mean-reversion | 1.2 | Mom_21d > 0 → +1; z-score_63d < -2 → +1 (mean revert) |
| 3 | Volatility regime | 0.8 | Vol_21d < Vol_63d → +1 (vol compression = bullish) |
| 4 | Monte Carlo | 0.9 | ARIMA-like drift + noise → direction vote |
| 5 | Quality tier | 1.1 | Tier A/B → +1; Tier F/G → -1 |
| 6 | Social sentiment | 1.0 | MiroFish ticker sentiment → ±1 |
| 7 | Distress | 0.9 | DistressedAssetEngine signals → ±1 |
| 8 | Event-driven | 1.0 | EventDrivenEngine signals → ±1 |
| 9 | CVR | 0.7 | CVREngine valuation signals → ±1 |
| 10 | Credit quality | 0.9 | UniverseClassifier CQS > 0.7 → +1; < 0.3 → -1 |

**Final vote**: Weighted sum of 10 tiers → [-10, +10]
**Minimum edge**: `effective_min_edge = 2.0 + max(0, -vote_score)` bps

#### Deep Trading Features
- **MicroPriceEngine**: Bid/ask estimation from OHLCV, order flow imbalance, urgency score
- **CrossAssetMonitor**: SPY-TLT, SPY-GLD, SPY-UUP correlations → risk-on/risk-off score
- **DeepTradingFeatures**: 15+ features per ticker (multi-horizon momentum, vol regime, z-score, skew, kurtosis)

#### Risk Gate Manager (8 gates, all must pass)
| Gate | Limit | Description |
|------|-------|-------------|
| G1 Position Size | 10% NAV | Single position limit |
| G2 Sector Concentration | 30% NAV | Sector exposure limit |
| G3 Daily Loss | 3% NAV | Daily loss circuit breaker |
| G4 Gross Exposure | 250% | Gross leverage limit |
| G5 Net Exposure | 150% | Net leverage limit |
| G6 Trade Count | 100/day | Trade throttle |
| G7 Drawdown | 10% | Max drawdown halt |
| G8 Cash Sufficiency | Trade value | Cash check for buys |

### PaperBroker (`engine/execution/paper_broker.py`)
- Simulated broker using OpenBB prices
- Tracks positions, P&L, NAV, cash, exposures
- Supports: BUY, SELL, SHORT, COVER
- Position tracking with sector tagging

### OptionsEngine (`engine/execution/options_engine.py`)
- Black-Scholes pricing (calls + puts)
- Greeks: Delta, Gamma, Theta, Vega, Rho
- Volatility surface construction
- **θ+Γ Optimizer**: `max(Θ + Γ) - c×Vega` for P4 sleeve allocation
- Strategy builder for hedging (collars, put-spreads, VIX calendars)

### ConvictionOverride (`engine/execution/conviction_override.py`)
| Tier | Confidence | Multiplier | Agents Required |
|------|-----------|------------|-----------------|
| CONTROLLED | 90-95% | 1.5× | 1 |
| AGGRESSIVE | 95-98% | 2.0× | 2 |
| MAXIMUM | >98% | 2.0×+ | 3 |

---

## LAYER 6: AGENT ORCHESTRATION

### SectorBots (`engine/agents/sector_bots.py`)
- **11 GICS sector micro-bots**, one per sector
- Each bot: analyzes sector momentum, generates signals, tracks accuracy
- Weekly scoring: 40% accuracy + 30% Sharpe + 30% hit rate
- Promotion/demotion system

### ResearchBots (`engine/agents/research_bots.py`)
- **11 GICS research bots** with DNA hierarchy
- Daily research cycle per sector
- Intelligence reports with conviction levels
- Bot-to-bot communication for cross-sector signals

### Agent Scorecard (`engine/agents/agent_scorecard.py`)
| Rank | Requirements | Promotion |
|------|-------------|-----------|
| DIRECTOR | Sharpe >2.5, accuracy >85% | Top performer 8+ weeks |
| GENERAL | Sharpe >2.0, accuracy >80% | 4 consecutive top weeks |
| CAPTAIN | Sharpe >1.5, accuracy >55% | Default |
| LIEUTENANT | Sharpe >1.0, accuracy >50% | Default |
| RECRUIT | Below thresholds | Demoted after 2 bottom weeks |

### GICS Sector Agents (`engine/agents/gics_sector_agents.py`)
- **11 GICS sector agents** with 8 scoring dimensions each
- Scoring: momentum, quality, value, growth, risk, sentiment, technical, macro alignment
- Sector-specific weight calibration
- Cross-sector signal correlation tracking

### Agent Monitor (`engine/agents/agent_monitor.py`)
**4-tier performance hierarchy**:
| Tier | Sharpe | Win Rate | Status |
|------|--------|----------|--------|
| ELITE | >2.0 | >70% | Full autonomy |
| STRONG | >1.5 | >60% | Standard operation |
| DEVELOPING | >0.5 | >45% | Restricted sizing |
| UNDERPERFORM | <0.5 | <45% | Review + potential shutdown |

- Per-agent memory tracking via `tracemalloc`
- Performance decay detection
- Automatic promotion/demotion with hysteresis

### Investor Personas (`engine/agents/investor_personas.py`)
**12 investor persona agents**, each with unique investment philosophy:
- **Warren Buffett**: Intrinsic value, moat, long-term hold
- **Charlie Munger**: Multi-disciplinary mental models
- **Benjamin Graham**: Deep value, margin of safety
- **Bill Ackman**: Activist + concentrated positions
- **Ray Dalio**: All-weather, macro regime balancing
- **Howard Marks**: Credit cycles, risk awareness
- **Seth Klarman**: Deep value + special situations
- **Peter Lynch**: Growth-at-reasonable-price (GARP)
- **George Soros**: Reflexivity, macro momentum
- **Stanley Druckenmiller**: Macro + growth conviction
- **David Einhorn**: Forensic accounting, short selling
- **Cathie Wood**: Disruptive innovation, hyper-growth

**8 Core Analysis Agents**: fundamentals, technicals, sentiment, risk management, macro, valuation, portfolio optimization, sector rotation

---

## MONITORING & REPORTING

### PlatinumReport — 30-section executive macro state (9 parts)
### PlatinumReportV2 (`engine/monitoring/platinum_report_v2.py`) — Enhanced platinum report generator
### PortfolioReport — Scenario engine + performance deep-dive (3 parts)
### PortfolioAnalytics (`engine/monitoring/portfolio_analytics.py`) — Deep portfolio analytics + scenario engine
### DailyReport — Open/close reports + sector heatmap
### HeatmapEngine (`engine/monitoring/heatmap_engine.py`) — GICS sector heatmap visualization
### SectorTracker — Sector performance + missed opportunities (>20% movers)
### LiveDashboard (`engine/monitoring/live_dashboard.py`) — Rich-based terminal dashboard (550+ symbols)
### LiveEarningsGraph (`engine/monitoring/live_earnings_graph.py`) — Live P&L visualization
### AnomalyDetector — Statistical anomaly scanner
### MarketWrap — Narrative market summary
### MemoryMonitor — Session tracking + EOD summary
### HourlyRecap — Hourly reconciliation

---

## WEB APPLICATION (`app/backend/`)

FastAPI-based web interface for the investment platform:
- **REST API**: Portfolio state, signal pipeline, flow management
- **SSE Streaming**: Real-time pipeline event streaming
- **SQLAlchemy Models**: Flows, FlowRuns, Trades, Portfolio, API keys
- **Pydantic Schemas**: Type-safe request/response validation
- **Services**: Agent orchestration, graph visualization, backtesting

---

## 29 SIGNAL TYPES

```
MICRO_PRICE_BUY    MICRO_PRICE_SELL    RV_LONG         RV_SHORT
FALLEN_ANGEL_BUY   ML_AGENT_BUY        ML_AGENT_SELL
DRL_AGENT_BUY      DRL_AGENT_SELL      TFT_BUY         TFT_SELL
MC_BUY             MC_SELL             QUALITY_BUY      QUALITY_SELL
SOCIAL_BULLISH     SOCIAL_BEARISH      SOCIAL_MOMENTUM  SOCIAL_REVERSAL
DISTRESS_FALLEN_ANGEL  DISTRESS_RECOVERY  DISTRESS_AVOID
CVR_BUY            CVR_SELL
EVENT_MERGER_ARB   EVENT_PEAD_LONG     EVENT_PEAD_SHORT EVENT_CATALYST
HOLD
```

---

## INTELLIGENCE PLATFORM — 28 Reference Repos

### How Each Repo Maps to the Engine

| Repo | Layer | Feeds Into |
|------|-------|-----------|
| Financial-Data | L1 | OpenBB Data, UniverseEngine |
| open-bb | L1 | MacroEngine, SectorRanker |
| hedgefund-tracker | L1 | InstitutionalFlow signals |
| FRB | L1 | **DEPRECATED** — FRED data now routed via OpenBB (`openbb-fred` provider). See `open-bb` |
| EquityLinkedGICPooling | L1 | GIC pooling methodology |
| Quant-Developers-Resources | L1 | Strategy templates library |
| Mav-Analysis | L2 | Technical indicators, backtesting |
| quant-trading | L2 | Strategy library (Bollinger, Dual Thrust) |
| stock-chain | L2 | Chain analysis, flow decomposition |
| CTA-code | L2 | Trend-following, momentum signals |
| TradeTheEvent | L2 | EventDrivenEngine (BERT event detection) |
| QLIB | L3 | AlphaOptimizer (factor mining, pipeline) |
| Stock-techincal-prediction-model | L3 | AlphaOptimizer (LSTM/CNN predictions) |
| Stock-prediction | L3 | Additional prediction models |
| ML-Macro-Market | L3 | MacroEngine (regime classification) |
| AI-Newton | L3 | Physics-informed models (differential equations) |
| ai-hedgefund | L4 | Multi-agent portfolio, CVREngine |
| financial-distressed-repo | L4 | DistressedAssetEngine (baseline) |
| sophisticated-distress-analysis | L4 | DistressedAssetEngine (advanced) |
| FinancialDistressPrediction | L4 | DistressedAssetEngine (GBM reference) |
| Kserve | L5 | Model serving infrastructure |
| nividia-repo | L5 | GPU-accelerated training/inference |
| Air-LLM | L5 | Efficient LLM inference |
| Ruflo-agents | L6 | Agent orchestration framework |
| MiroFish | L6 | SocialPredictionEngine (sentiment) |
| exchange-core | L7 | **HFT Execution** — Ultra-low-latency order matching engine (Java, LMAX Disruptor) |
| wondertrader | L7 | **HFT Execution** — Micro-price engine, CTA execution, low-latency order routing |

---

## DATA FLOW DIAGRAM

```
                    ┌──────────────┐
                    │  OpenBB Data │ ← Sole source (34+ providers, FRED, SEC, CBOE)
                    │  (187 APIs)  │ ← FMP, Intrinio, Polygon, Tiingo, etc.
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Universe    │ ← 150+ securities, 11 sectors, 26 RV pairs
                    │  Engine      │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │      MacroEngine        │ ← GMTF, 4 gammas, 7 sub-modules
              │  regime + sector ranks  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │     MetadronCube        │ ← C(t) = f(L,R,F), 10 layers
              │  regime + sleeves +     │
              │  gates + kill-switch    │
              └────────────┬────────────┘
                           │
        ┌──────────┬───────┴───────┬──────────┐
        ▼          ▼               ▼          ▼
   ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ Social  │ │ Distress │ │   CVR    │ │  Event   │
   │Prediction│ │ 5-model │ │ 5-model  │ │ 12-cat   │
   └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
        │           │             │             │
        └─────┬─────┴─────────────┴─────────────┘
              │
     ┌────────▼────────┐
     │ AlphaOptimizer  │ ← Walk-forward ML, EWMA cov, mean-var
     │ 50+ factors     │
     │ Quality A-G     │
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │  DecisionMatrix │ ← 6 gates, Kelly sizing, ABU beta
     │  MIN_SCORE=0.55 │
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │ExecutionEngine  │ ← 10-tier ML vote, 8 risk gates
     │  + PaperBroker  │ ← MicroPrice, cross-asset, deep features
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │ L7 HFT/Exec    │ ← exchange-core (order matching, LMAX Disruptor)
     │ wondertrader    │ ← micro-price, CTA, low-latency routing
     └────────┬────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌────────┐       ┌──────────────┐
│ Trades │       │   Reports    │
│ + P&L  │       │ Platinum/    │
│ + NAV  │       │ Portfolio/   │
└────────┘       │ Sector/Wrap  │
                 └──────────────┘
```

---

## EXPECTED OUTPUT (Full Pipeline Run)

### Morning Open generates:
1. **REGIME**: BULL/BEAR/TRANSITION/STRESS/CRASH + VIX + SPY returns
2. **SECTOR RANKINGS**: 11 sectors ranked by macro-adjusted composite score
3. **METADRON CUBE**: Regime, target beta, beta cap, max leverage, risk budget, 5 sleeve allocations
4. **ALPHA OPTIMIZER**: Expected return, volatility, Sharpe, top 5 signals with ticker/tier/alpha
5. **TRADES**: N trades executed (side, qty, ticker, vote_score)
6. **PORTFOLIO**: NAV, cash, P&L, positions, gross/net exposure
7. **CONTAGION**: 7 scenario systemic risk scores
8. **STAT ARB**: Pair status, z-scores, signals
9. **DISTRESSED ASSETS**: 5-model ensemble scores, fallen angels, recovery candidates
10. **CVR**: 5-model valuations, trade signals
11. **EVENT-DRIVEN**: Top event opportunities, Kelly-sized positions
12. **OPTIONS θ+Γ**: Best OTM level, theta, gamma, score
13. **RESEARCH BOTS**: 11 sector intelligence reports + DNA hierarchy
14. **PLATINUM REPORT**: 30-section executive macro state
15. **PORTFOLIO REPORT**: Scenario engine + performance analytics
16. **SECTOR HEATMAP**: ASCII visual sector performance
17. **MEMORY STATUS**: Session tracking

### Evening Close generates:
1. **MARKET WRAP**: Narrative summary
2. **MISSED OPPORTUNITIES**: >20% movers not captured
3. **CONTAGION EOD**: Updated systemic risk
4. **STAT ARB EOD**: Pair status update
5. **ANOMALIES**: Statistical anomalies detected
6. **AGENT SCORECARD**: Rankings + promotion/demotion
7. **CONVICTION AUDIT**: Override audit trail
8. **PLATINUM CLOSE**: 30-section with EOD stats
9. **PORTFOLIO CLOSE**: Full performance deep-dive
10. **MEMORY EOD**: Session summary + state persistence

---

## KEY COMMANDS

```bash
# Full morning pipeline
python3 run_open.py

# Evening reconciliation
python3 run_close.py

# Hourly recap
python3 run_hourly.py

# Platform health
python3 bootstrap.py

# Platform orchestrator status
python3 core/platform.py

# Run all tests (69 tests)
python3 -m pytest tests/ -v
```

---

## DESIGN RULES (IMMUTABLE)

1. **All data via OpenBB** (sole source, 34+ providers) — no yfinance dependency
2. **Paper broker only** — no live execution until broker API connected
3. **6-layer architecture is immutable** — extend within layers, not across
4. **Beta managed within 7–12% corridor** — vol-normalised
5. **Alpha targeted at 95%+** — aggressive multi-sleeve allocation
6. **All LLM agents use Anthropic API** (claude-opus-4-6)
7. **Pure-numpy fallbacks** — no bridge crashes if ML framework missing
8. **try/except on ALL external imports** — system runs degraded, never broken
9. **Tests must pass** before pushing
10. **Session continuity** — CLAUDE.md serves as bootstrap context
