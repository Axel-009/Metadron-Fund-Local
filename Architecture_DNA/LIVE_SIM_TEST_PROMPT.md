# METADRON CAPITAL — LIVE SIMULATION TEST PROMPT

> **Copy this entire file as the opening prompt when testing with any AI model
> (Perplexity, Claude CLI, ChatGPT, etc.).** It instructs the model to build,
> wire, and execute the full signal-to-trade pipeline without frontend,
> monitoring, or infrastructure components.

---

## Prompt

You are testing a quantitative investment platform called Metadron Capital. The
repository is at the GitHub link provided. Your job is to wire and execute the
**full signal-to-trade pipeline** — from data ingestion through to live orders
on IBKR — following the architecture exactly as documented. You are NOT building
the frontend, monitoring stack, or server infrastructure. You are testing the
**engine flow only**: data in, signals processed, intelligence scored, decisions
made, trades executed on the broker.

### What you MUST do before writing any code

1. Read `Architecture_DNA/DATA_FLOW_CHART.md` — this is the authoritative
   single-run data flow. Every engine, every connection, every stage.

2. Read `engine/wiring_manifest.py` — this is the machine-readable component
   registry. Run `python3 -m engine.wiring_manifest` to validate all 42
   components import correctly and the execution chain is connected.

3. Read `Architecture_DNA/ARCHITECTURE_DNA.md` — full architecture specification
   including broker config, allocation rules, gate weights, and risk limits.

4. **Do NOT skip components.** Every engine listed in the wiring manifest must
   be instantiated. Do not cherry-pick. Do not use raw HTTP requests to any
   broker API. Do not bypass L7.

### What you ARE testing (in scope)

**Stage 1 — SCAN:** Instantiate UniverseEngine (4-run scan: SP500, SP400,
SP600, ETF/FI + 26 RV pairs), DataIngestionOrchestrator, DataQualityGate,
UniversalDataPool (cross-asset data dispatcher), and NewsEngine (newsfilter.io
WebSocket + FMP fallback). Pull live market data via OpenBB and IBKR quotes.

**Stage 2 — LAYERS (two independent parallel tracks):**

- **Track A** (MetadronCube path): FedLiquidityPlumbing → MacroEngine (GMTF
  regime + sector ranking) → MetadronCube C(t)=f(L,R,F) with 10-layer tensor
  + KillSwitch → then fans out to parallel signal engines: SecurityAnalysis
  (Graham-Dodd-Klarman), ContagionEngine (21 nodes, 7 shocks), StatArbEngine
  (Medallion mean-reversion), FixedIncomeEngine (yield curve), DistressedAsset
  (5-model ensemble), PatternDiscovery (MiroFish + AI-Newton),
  AdaptiveThresholdCalibrator.

- **Track B** (NewsEngine path, independent from Cube): NewsEngine →
  MiroMomentumEngine (40% news sentiment + 60% agent simulation per
  news-flagged ticker) → enriches EventDrivenEngine (12 event categories) +
  CVREngine (5-model valuation). Track B feeds into MLVoteEnsemble Tier 6.

Track A and Track B run independently with zero cross-dependencies. Their
signals converge at Stage 3.

**Stage 3 — INTELLIGENCE:** AlphaOptimizer receives ALL signals from both
tracks and performs: (1) universe merge from 4 runs, (2) aggregate + dedup by
ticker keeping highest confidence, (3) cap enforcement per allocation bucket
(IG 40%, HY 10%, Distressed 10%, TLTW 15%, FI 5%, CVR 10%, Options 25%
notional, Futures 15%), (4) dual alpha scoring pipeline (Standard: XGBoost 60%
+ LinReg 40% + CAPM 20% blend; Enhanced: WalkForward + FactorLibrary + sector
MVO). Then concurrent ML models feed MLVoteEnsemble (10 tiers, all must vote):
T1 Neural, T2 Momentum, T3 VolRegime, T4 MonteCarlo, T5 Quality, T6
MiroMomentum, T7 Distressed, T8 EventDriven, T9 CVR, T10 CreditQuality.
Output: FullUniverseEngine Scan Slate.

**Stage 4 — DECISION + EXECUTION:**

The scan slate passes through DecisionMatrix — 4 cross-asset gates (all
asset-agnostic, applied uniformly to equities/options/futures/ETFs):

| Gate | Weight | ML Tiers | Function |
|------|--------|----------|----------|
| FUNDAMENTALS | 40% | T1,T5,T7,T9,T10 | Quality, ROIC, credit + regime quality modifier |
| FLOW_HEADLINES | 20% | T6,T8 | News sentiment, ETF flow, sector rotation |
| MACRO_REGIME | 20% | T3,T4 | Direction alignment, VaR headroom, drawdown |
| MOMENTUM | 20% | T1,T2 | RSI, MACD, breakout, cross-asset momentum |

Composite ≥ 0.55 to approve. FUNDAMENTALS is the critical gate.

Approved trades → KellySizer (1.5x multiplier) → AllocationEngine bucket
enforcement → BetaCorridor (7-12% return corridor).

Options MUST go through OptionsSizer (Black-Scholes + Monte Carlo 10K paths +
Kelly): no market price → REJECTED, edge < 200bps → REJECTED, < 5 contracts →
REJECTED. Only mispriced options are traded.

ALL orders route through L7UnifiedExecutionSurface.submit_order() — this is
mandatory. Never call IBKRBroker directly. The L7 surface performs:
1. Research-only guard (reject FI/FX/credit)
2. MultiProductRouter classifies EQUITY / OPTION / FUTURE
3. WonderTrader micro-price adjustment (equities)
4. L7RiskEngine 10-gate pre-trade check (all must pass)
5. SlippageModel pre-trade cost estimate
6. IBKR algo routing: >$50K notional → TWAP, else VWAP/Adaptive/Market
7. IBKRBroker executes (sole broker, via ib_insync)
8. Post-trade: risk state update, TCA decomposition, trade log, Prometheus
9. ExecutionLearningLoop records outcome

Trade log records every generated order for reconciliation (generated vs
executed on broker).

### What you are NOT testing (out of scope)

- Frontend (Express, React, all 30 tabs) — skip entirely
- Nginx, SSL, Cloudflare domain — skip entirely
- Hetzner/Contabo server setup — skip entirely
- Docker, PM2 ecosystem — skip entirely
- Prometheus, Grafana dashboards — skip entirely (metrics are registered but
  scraping infrastructure is not needed for this test)
- Overnight backtesting, QSTrader — skip entirely
- LearningLoop feedback, GSD/Paul plugins, agent scorecard — skip entirely
  (these run post-execution and are not needed to verify the trade flow)

### Environment setup

```bash
pip install ib_insync openbb numpy pandas scikit-learn xgboost httpx
```

IBKR TWS or IB Gateway must be running:
```bash
export IBKR_HOST=127.0.0.1
export IBKR_PORT=7497          # 7497 = paper trading, 7496 = live
export IBKR_CLIENT_ID=1
export IBKR_PAPER_TRADE=True
```

OpenBB for market data:
```bash
export OPENBB_TOKEN=your_token
export FMP_API_KEY=your_fmp_key
```

### Validation before first trade

Run the wiring manifest validation:
```bash
python3 -m engine.wiring_manifest
```

All 42 components must import. All 10 routing rules must display. Verdict must
be ALL CHECKS PASSED. If any component fails to import, install the missing
dependency before proceeding.

### Expected output

After running the pipeline, you should see:
1. Universe scan: ~1,600 securities ingested across 4 runs
2. Track A signals: macro regime, cube output, signal engine results
3. Track B signals: news-flagged tickers with MiroMomentum scores
4. AlphaOptimizer: merged, deduped, cap-enforced, scored slate
5. DecisionMatrix: 4-gate composite scores, approved/rejected per ticker
6. L7 execution: orders submitted to IBKR with algo routing
7. Trade log: JSONL entries in logs/l7_execution/trade_log/
8. IBKR dashboard: orders visible on TWS/Gateway

If IBKR is not connected, all orders will appear in the trade log as
NOT_EXECUTED — this is correct behavior. The trade log is the reconciliation
audit trail showing what the platform wanted to do vs what actually executed.

### Critical rules (do not violate)

1. ALL orders through L7UnifiedExecutionSurface — never direct broker calls
2. ALL options through OptionsSizer — rejected without mispricing ≥ 200bps
3. MetadronCube KillSwitch halts ALL execution when triggered
4. DecisionMatrix composite ≥ 0.55 — no trade below this threshold
5. IBKR is the sole broker — no Alpaca, no Tradier, no raw API requests
6. Track A and Track B are independent — no cross-dependencies
7. Every engine in the wiring manifest must be instantiated — no cherry-picking
