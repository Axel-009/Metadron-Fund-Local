# METADRON CAPITAL — COMPREHENSIVE CODE AUDIT REPORT

**Date:** 2026-03-26
**Auditor:** Bobby (CEO, Quant Fund)
**Scope:** Full platform audit — code quality, architecture, security, execution safety, data pipeline, tests, dependencies, ML pipeline, monitoring
**Verdict:** The platform has serious ambition but critical operational gaps. Several findings are fund-killers if not addressed before any capital is deployed.

---

## EXECUTIVE SUMMARY

| Area | Rating | Verdict |
|------|--------|---------|
| 1. Code Quality | ⚠️ Needs work | Massive codebase (61K+ lines engine), clean architecture, but non-optional top-level imports crash the entire engine if any sub-module fails |
| 2. Architecture Health | ⚠️ Needs work | 7-layer pipeline exists on paper, but execution_engine.py is a 1500-line monolith with hard coupling to 12+ modules. Two parallel systems (platform_orchestrator.py vs execution_engine.py) that don't talk to each other |
| 3. Security | ✅ Solid | No hardcoded secrets found. Vault-encrypted .env, proper .env.example, API keys never in code |
| 4. Configuration | 🔴 Critical | repos.yaml missing 6+ repos. Duplicate `stock_prediction` key. `run_hourly.py` imports non-existent module |
| 5. Execution Safety | 🔴 Critical | Paper broker uses hardcoded `entry=100.0`. No crash-recovery. No order persistence. Risk gates exist but sector concentration check is a stub |
| 6. Data Pipeline | ⚠️ Needs work | OpenBB integration is well-architected with graceful fallback, but dependencies not installed — nothing actually runs |
| 7. Tests | 🔴 Critical | 69 tests written but **cannot run** — no dependencies installed (numpy, pytest missing). Tests only cover core modules, not the actual execution engine pipeline |
| 8. Dependencies | 🔴 Critical | pyproject.toml declares 40+ deps including torch, tensorflow, openbb — **none installed**. No setup.sh has ever been run successfully |
| 9. ML Pipeline | 🔴 Critical | No model persistence. No walk-forward validation wired in. AlphaOptimizer is pure-numpy with no model storage. Training is a no-op |
| 10. Monitoring | ⚠️ Needs work | Report generators exist but generate static/mock data. No live data feeds. Learning loop is structurally complete but untested with real outcomes |

**Overall: 🔴 NOT PRODUCTION READY**

---

## DETAILED FINDINGS

### 1. CODE QUALITY — ⚠️ Needs Work

**What's good:**
- Clean module separation across `engine/data/`, `engine/signals/`, `engine/ml/`, `engine/execution/`, `engine/portfolio/`, `engine/monitoring/`, `engine/agents/`
- Consistent dataclass usage throughout (Security, MacroSnapshot, CubeOutput, AlphaSignal, etc.)
- `universe_engine.py` is well-structured with proper enum definitions, GICS taxonomy, and asset class routing
- `macro_engine.py` has 7 sub-modules (MoneyVelocityModule, SectorRanker, CarryToVolatility, RegimeTransitionDetector, YieldCurveAnalyzer, CreditPulseMonitor, MacroFeatureBuilder) — each is cleanly separated

**What's broken:**
- **execution_engine.py** is a 1,500+ line monolith. It contains: MicroPriceEngine, CrossAssetMonitor, RiskGateManager, DeepTradingFeatures, MLVoteEnsemble, TradeAllocator, PipelineTracker, and the main ExecutionEngine class. This should be 8+ separate files.
- **Non-optional imports** at module top level — `from ..signals.social_prediction_engine import SocialPredictionEngine` etc. are NOT wrapped in try/except. If any of these modules fail to import (missing dependency, syntax error, etc.), the entire `execution_engine.py` module crashes, killing the whole platform.
  - Specifically: `TradierBroker`, `QuantStrategyExecutor`, `SocialPredictionEngine`, `DistressedAssetEngine`, `CVREngine`, `EventDrivenEngine`, `SecurityAnalysisEngine`, `PatternDiscoveryEngine` are all imported at top level without protection.
  - Only `L7UnifiedExecutionSurface` has a proper try/except wrapper.
- **Dead code in `run_open.py`**: MacroEngine is instantiated twice (line ~50 and line ~190). The second instance overwrites the first's results.
- **Duplicate key in repos.yaml**: `stock_prediction` appears twice with different paths (`Stock-techincal-prediction-model` and `Stock-prediction`). YAML will silently use the last one.
- **`run_hourly.py`** imports `generate_sector_heatmap` from `engine.monitoring.daily_report` — but that function is defined in `engine.monitoring.heatmap_engine`. This will crash at import time.

### 2. ARCHITECTURE HEALTH — ⚠️ Needs Work

**The documented 7-layer pipeline:**
```
UniverseEngine → MacroEngine → MetadronCube → SecurityAnalysis → AlphaOptimizer → DecisionMatrix → ExecutionEngine
```

**What actually happens:**
- The pipeline in `ExecutionEngine.run_pipeline()` is a 12-stage sequential process that runs synchronously. This is correct for daily open/close but won't work for the "1-minute heartbeat continuous loop" described in CLAUDE.md.
- `platform_orchestrator.py` (InvestmentPlatformOrchestrator) is a COMPLETELY SEPARATE system from the engine. It has its own `_TechnicalAnalyzer`, `_FundamentalAnalyzer`, `_MacroAnalyzer`, `_SentimentAnalyzer`, `_RiskManager`, `_CubeRotation`, `_ExecutionEngine`, `_MLLearner` — none of which use or connect to the actual `engine/` modules. This is ~1,000 lines of dead code that will confuse anyone working on the platform.
- The `distress_scanner` import in `platform_orchestrator.py` references a module that doesn't exist in the repo.

**Orphaned modules:**
- `core/platform.py`, `core/signals.py`, `core/portfolio.py` — the "original" core platform. These load repos.yaml and do status reporting, but they're completely disconnected from the engine. Nobody calls them. Nobody imports them except `bootstrap.py`.

**What connects:**
- `ExecutionEngine` → `UniverseEngine` ✅
- `ExecutionEngine` → `MacroEngine` ✅
- `ExecutionEngine` → `MetadronCube` ✅
- `ExecutionEngine` → `AlphaOptimizer` ✅
- `ExecutionEngine` → `BetaCorridor` ✅
- `ExecutionEngine` → `PaperBroker`/`TradierBroker` ✅
- `ExecutionEngine` → `MLVoteEnsemble` ✅
- `ExecutionEngine` → `LearningLoop` ✅
- `ExecutionEngine` → `DecisionMatrix` ✅

**What's stubbed or broken:**
- `platform_orchestrator.py` → nothing (orphaned system)
- `core/` modules → nothing (orphaned)
- `mirofish/` frontend/backend → nothing calls it (SocialPredictionEngine reads actions.jsonl files, doesn't call the Flask API)
- `intelligence_platform/` repos → all 21 repos are reference code only, none are imported by the engine

### 3. SECURITY — ✅ Solid

**What's good:**
- `.env.example` has placeholder values only — no real keys
- `.env.vault.gpg` exists for encrypted credential storage
- `setup.sh` decrypts the vault with GPG passphrase, never writes keys to stdout
- `setup.sh` propagates keys to sub-module .env files automatically
- No hardcoded API keys found in any Python file (grep for `sk-`, `AKIA`, `api_key=` returned nothing)
- `.gitignore` properly excludes `.env` but keeps `.env.vault.gpg`
- Tradier broker uses environment variables for API key/account ID

**Minor concerns:**
- `.env.vault.gpg` is present in the repo (tracked). If the GPG passphrase is weak, this is a risk. Standard practice, but worth noting.
- `setup.sh` prints partial API keys: `echo -e "  ${GREEN}OK${NC} ANTHROPIC_API_KEY set (${ANTHROPIC_KEY:0:12}...)"` — first 12 chars of a key in terminal output is a minor leak risk in shared environments.

### 4. CONFIGURATION — 🔴 Critical

**repos.yaml analysis:**
- Contains 23 repos total
- **Missing from CLAUDE.md's documented architecture:**
  - `worldmonitor` (L2 — global event monitoring, feeds EventDrivenEngine + MacroEngine)
  - `markov-model` (L3 — HMM regime detection, feeds MetadronCube RegimeEngine)
  - `get-shit-done` (GSD plugin — referenced in learning loop)
  - `wondertrader` (L7 — referenced in CLAUDE.md, exists in repos/ but not in repos.yaml)
  - `exchange-core` (L7 — referenced in CLAUDE.md, exists in repos/ but not in repos.yaml)
  - `qstrader` (intelligence_platform/qstrader/ — referenced in CLAUDE.md but not in repos.yaml)
- **Duplicate key**: `stock_prediction` appears twice (lines referencing `Stock-techincal-prediction-model` and `Stock-prediction`)
- **Layer 0 and Additional sections** are informal — not following the layer numbering scheme

**Other config issues:**
- No `config/` directory for runtime configuration (position limits, regime thresholds, etc.)
- All risk parameters are hardcoded in Python (RiskGateManager defaults, BetaCorridor constants, etc.)

### 5. EXECUTION SAFETY — 🔴 Critical

**Paper broker:**
- Uses `entry = 100.0` as hardcoded placeholder in `platform_orchestrator.py`. In the actual `ExecutionEngine`, prices come from `broker._get_current_price(ticker)` which calls OpenBB — this is correct.
- **No order persistence**: If the process crashes mid-trade, all state is lost. PaperBroker state lives in memory only.
- **No crash recovery**: No serialization of portfolio state, no checkpoint/resume mechanism.
- **No order ID tracking across restarts**: Trade IDs are generated with `uuid4()` and exist only in memory.

**Risk gates (8 gates in RiskGateManager):**
- G1 (Position size) ✅ Implemented
- G2 (Sector concentration) ⚠️ Stub — uses `ticker == sector` which never matches. The sector is always the ticker symbol, not the actual sector name.
- G3 (Daily loss) ✅ Implemented
- G4 (Gross exposure) ✅ Implemented
- G5 (Net exposure) ✅ Implemented
- G6 (Trade count throttle) ✅ Implemented
- G7 (Drawdown circuit breaker) ✅ Implemented
- G8 (Cash sufficiency) ✅ Implemented

**The G2 sector concentration gate is BROKEN.** It compares `p.sector == ticker` which will never be true for sector-level concentration checking. This means a portfolio could go 100% into a single sector with no gate stopping it.

**Tradier broker:**
- Supports sandbox and production environments
- Falls back to paper broker state when API calls fail (graceful degradation)
- No mention of order status polling — orders are fire-and-forget

**Kill switch:**
- The MetadronCube has a kill-switch (`HY OAS +35bp & VIX term flat/inverted & breadth <50% → auto β ≤ 0.35`) but it's not clear this actually gates execution in the run_pipeline flow.
- L7RiskEngine has a proper kill switch that activates on 10%+ drawdown.
- But L7 is an optional component (wrapped in try/except) — if it fails to init, there's no kill switch.

### 6. DATA PIPELINE — ⚠️ Needs Work

**OpenBB integration:**
- `openbb_data.py` is well-architected: try/except on import, retry wrapper, graceful fallback to empty DataFrames
- `yahoo_data.py` is a clean re-export layer — all existing imports continue to work
- FRED series are properly mapped (40+ series including WALCL, T10Y2Y, SOFR, etc.)
- Multiple providers supported: FMP (default), Fred, Polygon, Intrinio

**What's broken:**
- **No dependencies installed.** `numpy`, `pandas`, `openbb` — none available. The platform literally cannot run.
- OpenBB requires API keys for most providers (FMP, Polygon, etc.) — without `.env` configured, `get_adj_close()` returns empty DataFrames.
- The `_try_openbb_constituents()` method tries to fetch S&P 500/400/600 constituents dynamically, but falls back to static universe from `cross_asset_universe.py`. This is good design.

**Fallback chain:**
1. OpenBB dynamic fetch → 2. Static universe → 3. Empty universe (graceful) ✅

### 7. TESTS — 🔴 Critical

**Test coverage:**
- `test_platform.py`: 8 tests — core platform, signal engine, portfolio engine
- `test_engine.py`: ~50 tests — covers L1-L6, CVR, Event-Driven, Distressed Assets, Asset Class Routing, Learning Loop, Tradier Broker
- `test_l7_execution.py`: ~35 tests — L7 Unified Execution Surface, TCA, Risk, Learning Loop

**What's covered:**
- ✅ UniverseEngine (screening, GICS, RV pairs)
- ✅ MacroEngine (regime classification, sigmoid, cube regime mapping)
- ✅ MetadronCube (sleeve allocation sums to 1, regime params, beta target)
- ✅ AlphaOptimizer (quality classification, feature building, EWMA cov)
- ✅ BetaCorridor (boundaries, vol adjustment, rebalance)
- ✅ PaperBroker (basic order flow)
- ✅ SectorBots (init, scorecard, tier promotion/demotion)
- ✅ DistressedAssetEngine (5-model ensemble, Altman Z, Merton KMV)
- ✅ CVREngine (binary option, barrier option, milestone tree)
- ✅ EventDrivenEngine (merger arb, PEAD, signals)
- ✅ Asset Class Routing (tradeable vs macro-only filtering)
- ✅ Learning Loop (signal outcomes, regime feedback, tier weights)
- ✅ Tradier Broker (interface compliance, sandbox init)
- ✅ L7 Execution Surface (order routing, TCA, risk gates)

**What's MISSING:**
- ❌ No tests for `ExecutionEngine.run_pipeline()` (the main pipeline)
- ❌ No tests for `MLVoteEnsemble.vote()` (the core ML voting)
- ❌ No integration test for the full end-to-end pipeline
- ❌ No tests for `MacroEngine.analyze()` with real/realistic data
- ❌ No tests for `run_open.py` or `run_close.py`
- ❌ No tests for any monitoring module (platinum_report, portfolio_report, etc.)
- ❌ No tests for `SocialPredictionEngine`, `PatternDiscoveryEngine`, `SecurityAnalysisEngine`
- ❌ No tests for `quant_strategy_executor` (12 HFT strategies)
- ❌ No tests for `fed_liquidity_plumbing`
- ❌ No tests for `contagion_engine`, `stat_arb_engine`

**Cannot run:** pytest is not installed. Tests are well-written but unverifiable.

### 8. DEPENDENCIES — 🔴 Critical

**pyproject.toml declares:**
- numpy>=2.0, pandas>=2.1, scipy>=1.11, scikit-learn>=1.3
- torch>=2.0, tensorflow>=2.15, xgboost>=2.0, lightgbm>=4.0
- openbb>=4.0 + 11 openbb-* provider packages
- langchain, langchain-anthropic, langchain-openai, langgraph
- anthropic>=0.50, openai>=1.0
- fastapi, uvicorn, httpx, pydantic
- plotly, matplotlib, seaborn
- python-dotenv, rich, pyyaml

**Reality: NONE of these are installed.** The container has Python 3 but no pip packages.

**Issues:**
- `torch` and `tensorflow` together will consume ~5GB+ of disk space. For a platform that uses "pure-numpy fallbacks" and has a DeepLearningEngine that explicitly avoids external ML frameworks, these are massive bloat.
- `langchain`, `langchain-anthropic`, `langgraph` — used nowhere in the actual engine code. The CLAUDE.md says "All agents use Anthropic API" but the engine never calls any LLM. These are dead dependencies.
- `fredapi>=0.5` is listed but OpenBB already provides FRED access. Redundant.
- `openbb>=4.0` + all provider packages — OpenBB 4.x has a different import structure than what the code expects (`from openbb import obb`).

### 9. ML PIPELINE — 🔴 Critical

**What exists:**
- `AlphaOptimizer` — walks through optimization, builds features, computes EWMA covariance, does Sharpe-weighted SLSQP optimization. **No actual ML model.** It's a numerical optimizer, not a learning system.
- `MLVoteEnsemble` — 10-tier voting system. Tiers 1-4 are pure-numpy heuristics. Tiers 5-10 are injected signals. **No model training. No model storage. No walk-forward validation.**
- `DeepLearningEngine` — described as "pure-numpy PPO agent" but not connected to the main pipeline.
- `UniverseClassifier` — XGBoost 4-model ensemble. Imports `xgboost` which isn't installed.

**What's missing:**
- ❌ No model persistence (no `model.save()`, no pickle/joblib of trained weights)
- ❌ No walk-forward validation pipeline (CLAUDE.md describes it, code doesn't implement it)
- ❌ No model retraining triggers
- ❌ No feature store or feature pipeline
- ❌ No backtesting integration with real data (QSTrader bridge exists but not wired in)
- ❌ The `_tier1_neural()` method re-seeds random weights every call with `np.random.seed(abs(hash(r.tobytes())) % (2**31))` — this means the "neural net" produces deterministic but meaningless outputs. It's not learning anything.
- ❌ LearningLoop records outcomes but never actually adjusts anything actionable in the pipeline (tier weights are computed but the apply method's effect is unclear)

### 10. MONITORING — ⚠️ Needs Work

**What exists:**
- `daily_report.py` — generates ASCII reports. Functional but uses static formatting.
- `platinum_report.py` / `platinum_report_v2.py` — 30-section reports. Importable but not tested.
- `portfolio_report.py` / `portfolio_analytics.py` — scenario engine + performance analytics.
- `sector_tracker.py` — sector performance tracking with missed opportunities.
- `heatmap_engine.py` — sector heatmap visualization.
- `live_dashboard.py` — Rich-based terminal dashboard.
- `anomaly_detector.py` — statistical anomaly scanner.
- `memory_monitor.py` — session tracking + EOD summary.
- `learning_loop.py` — closed-loop feedback (structurally complete).
- `l7_dashboard.py` — L7 Risk + TCA dashboard panels.

**What's broken:**
- Most monitoring modules would generate output but with no live data (OpenBB not available), they'd show empty/zeroed reports.
- `live_dashboard.py` depends on `rich` which isn't installed.
- `learning_loop.py` is structurally sound but has no test verifying that `apply_to_ensemble()` actually changes tier weights in a way that affects future votes.

---

## CRITICAL ACTIONS (MUST FIX BEFORE ANY CAPITAL DEPLOYMENT)

### Priority 1 — Platform Cannot Run
1. **Install dependencies**: `pip install -e .` or at minimum `pip install numpy pandas scipy scikit-learn pyyaml pytest`
2. **Create .env**: Decrypt `.env.vault.gpg` or create `.env` with placeholder values
3. **Fix `run_hourly.py` import**: Change `generate_sector_heatmap` import to correct module

### Priority 2 — Execution Safety
4. **Fix Risk Gate G2**: Sector concentration check is broken — `p.sector == ticker` never matches
5. **Add order persistence**: Serialize PaperBroker state to disk (JSON/SQLite). Crash = lost portfolio.
6. **Fix top-level imports in execution_engine.py**: Wrap ALL sub-engine imports in try/except. One broken module kills the entire engine.
7. **Remove the hardcoded `entry = 100.0`** in platform_orchestrator.py or delete the file entirely.

### Priority 3 — Architecture Cleanup
8. **Delete or clearly mark `platform_orchestrator.py` as deprecated**: It's a parallel system that doesn't connect to the engine. Confusing.
9. **Delete or archive `core/` modules**: They're orphaned.
10. **Fix repos.yaml**: Add missing repos (worldmonitor, markov-model, wondertrader, exchange-core, qstrader, get-shit-done). Fix duplicate `stock_prediction` key.
11. **Separate execution_engine.py**: Split the 1500-line monolith into `micro_price.py`, `cross_asset.py`, `risk_gates.py`, `deep_features.py`, `ml_vote.py`, `trade_allocator.py`, `pipeline_tracker.py`.

### Priority 4 — ML Pipeline
12. **Implement model persistence**: Save/load trained models (even if it's just numpy arrays to .npy files)
13. **Wire in actual walk-forward validation**: The backtester.py exists but isn't called by the pipeline
14. **Remove torch/tensorflow from deps** if not actually used — they're 5GB+ of bloat
15. **Remove langchain deps** if not used in the engine

### Priority 5 — Tests
16. **Run the existing test suite** after installing deps
17. **Add pipeline integration tests**: `test_pipeline_full_run()` that exercises `ExecutionEngine.run_pipeline()`
18. **Add MLVoteEnsemble tests**: Test that votes are deterministic, tier injection works
19. **Add monitoring module smoke tests**: Verify report generators produce non-empty output

---

## WHAT'S ACTUALLY GOOD

Don't get me wrong — there's serious work here:

1. **Asset class routing** is well thought out. The TRADEABLE vs MACRO_ONLY distinction prevents bond ETFs from hitting the equity execution pipeline. This is correct.
2. **The GICS taxonomy** is comprehensive (11 sectors, 32 industry groups, 100+ industries).
3. **MetadronCube regime parameters** are properly mapped (TRENDING 3.0x leverage, CRASH 0.8x, etc.)
4. **The learning loop architecture** is sound — signal outcomes, regime feedback, sector allocation tracking. It just needs real data flowing through it.
5. **Graceful degradation pattern** (try/except on imports) is mostly followed — just not in the critical execution_engine.py top-level imports.
6. **L7 Unified Execution Surface** is well-designed with multi-product routing, TCA, and execution learning.
7. **CVREngine, EventDrivenEngine, DistressedAssetEngine** all have proper valuation models (binary options, merger arb spread, Altman Z, Merton KMV). These are not stubs — they have real financial math.

---

## BOTTOM LINE

This platform is a **prototype, not a production system**. The architecture is ambitious and mostly well-designed, but:

- **It cannot run** (no dependencies installed)
- **The risk controls have a broken gate** (sector concentration)
- **There's no crash recovery** (state is memory-only)
- **The ML pipeline doesn't learn** (no model persistence, no walk-forward)
- **Two orphaned codebases** (platform_orchestrator.py, core/) create confusion

The gap between CLAUDE.md's vision ("$1,000 → $100,000 in 100 days", "95%+ alpha", "compete with Medallion") and the actual code state is significant. The financial math in the specialized engines (CVR, Event-Driven, Distressed Assets) is genuinely sophisticated. The core pipeline infrastructure needs another 2-3 months of hardening before any real capital touches it.

**Do not deploy with real money until Priority 1-2 items are resolved.**

---

*Report generated by Bobby. No sugar-coating. Bugs cost money.*
