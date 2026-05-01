# Intelligence Platform Registry

Complete reference for all sub-repos in `intelligence_platform/`, mapped to the
final Metadron Capital architecture. Every entry shows which engine consumes it,
which stage in the data flow it feeds, and its wiring status.

**Architecture reference**: `Architecture_DNA/DATA_FLOW_CHART.md`
**Wiring validation**: `python3 -m engine.wiring_manifest` (43 components, 59 edges)
**Execution broker**: IBKR only (via L7UnifiedExecutionSurface)
**Decision gates**: 4-gate (FUNDAMENTALS 40%, FLOW_HEADLINES 20%, MACRO_REGIME 20%, MOMENTUM 20%)

---

## Data Flow Stage Mapping

| Stage | Focus | Sub-repos |
|-------|-------|-----------|
| Stage 1: SCAN | Data ingestion + universe | Financial-Data, open-bb, hedgefund-tracker, FRB, EquityLinkedGICPooling |
| Stage 2: LAYERS (Track A) | Signal processing via MetadronCube | Mav-Analysis, stock-chain, CTA-code, TradeTheEvent, ML-Macro-Market, wondertrader |
| Stage 2: LAYERS (Track B) | News + MiroMomentum (independent) | MiroFish |
| Stage 3: INTELLIGENCE | ML/AI scoring + ensemble | QLIB, Stock-techincal-prediction-model, Stock-prediction, AI-Newton |
| Stage 4: EXECUTION | Order routing + matching | exchange-core, wondertrader, quant-trading |
| Stage 5: LEARNING | Agent orchestration + feedback | Ruflo-agents, get-shit-done, MiroFish |
| Infrastructure | ML serving + GPU + LLM | Kserve, nividia-repo, Air-LLM |
| Risk | Distress + credit analysis | financial-distressed-repo, sophisticated-distress-analysis, FinancialDistressPrediction, ai-hedgefund |
| Reference | Documentation only | Quant-Developers-Resources, CTA-code |

---

## Sub-Repo Inventory

### Stage 1: SCAN — Data Ingestion

| Sub-repo | Purpose | Engine Consumer | Integration |
|----------|---------|-----------------|-------------|
| **Financial-Data** | OpenBB market data pipeline | `engine/data/openbb_data.py` (superseded by direct OpenBB) | FULLY_WIRED |
| **open-bb** | Vendored OpenBB platform (34+ providers) | `engine/data/openbb_data.py` via `from openbb import obb` | FULLY_WIRED |
| **hedgefund-tracker** | SEC 13F/13D/Form 4, institutional flow, Promise Score | `engine/ml/bridges/hedgefund_tracker_bridge.py` → `HedgefundTrackerBridge` | FULLY_WIRED |
| **FRB** | Federal Reserve FRED API client | `engine/signals/fed_liquidity_plumbing.py` → `FRBFredClient` | FULLY_WIRED |
| **EquityLinkedGICPooling** | GIC pooling methodology reference | `engine/data/universal_pooling.py` (architectural reference) | REFERENCE_ONLY |
| **Quant-Developers-Resources** | Quant finance reference materials (11 categories) | None (documentation only) | REFERENCE_ONLY |

### Stage 2: LAYERS — Track A Signal Processing

| Sub-repo | Purpose | Engine Consumer | Integration |
|----------|---------|-----------------|-------------|
| **Mav-Analysis** | Multi-asset technical analysis (RSI/MACD/BB composite) | `engine/signals/security_analysis_engine.py` → `MultiAssetAnalyzer` | FULLY_WIRED |
| **stock-chain** | Cross-asset correlation, HMM regime, sector rotation | `engine/signals/security_analysis_engine.py` → `AssetClassAnalyzer` | FULLY_WIRED |
| **TradeTheEvent** | Event-driven ML (BERT event detection, credit rating changes) | `engine/signals/event_driven_engine.py` → `EventDrivenStrategy` | FULLY_WIRED |
| **ML-Macro-Market** | HMM regime model, Fama-French factors, GDP nowcasting | `engine/signals/macro_engine.py` → `HiddenMarkovRegimeModel`, `MacroMLEngine` | FULLY_WIRED |
| **CTA-code** | CTA/trend-following reference textbooks and code | None (reference only) | REFERENCE_ONLY |

### Stage 2: LAYERS — Track B (News + MiroMomentum)

| Sub-repo | Purpose | Engine Consumer | Integration |
|----------|---------|-----------------|-------------|
| **MiroFish** | Agent-based market simulation (Kyle Lambda, HAM, stress test) | `engine/signals/agent_sim_engine.py` → `AgentSimEngine`, `engine/signals/social_prediction_engine.py` → `MiroMomentumEngine` | FULLY_WIRED |

### Stage 3: INTELLIGENCE — ML/AI Models

| Sub-repo | Purpose | Engine Consumer | Integration |
|----------|---------|-----------------|-------------|
| **QLIB** | Microsoft QLIB quantitative ML framework | `engine/ml/bridges/qlib_bridge.py` → `QLIBBridge` (OpenBB providers, Alpha158) | FULLY_WIRED |
| **Stock-techincal-prediction-model** | LSTM/XGBoost/RF/Transformer ensemble predictor | `engine/ml/bridges/stock_prediction_bridge.py` → `predict_with_ensemble()` | FULLY_WIRED |
| **Stock-prediction** | BiLSTM/BiGRU/RNN notebooks | None (Jupyter reference only) | REFERENCE_ONLY |
| **AI-Newton** | Physics-inspired symbolic regression (PySR) | `engine/bridges/ainewton_discovery_worker.py` → `PhysicsOptimizer`, `engine/bridges/ainewton_service.py` | FULLY_WIRED |

### Stage 4: EXECUTION — Order Routing + Matching

| Sub-repo | Purpose | Engine Consumer | Integration |
|----------|---------|-----------------|-------------|
| **exchange-core** | Java LMAX Disruptor matching engine blueprint | `engine/execution/exchange_core_engine.py` (full Python reimplementation) | FULLY_WIRED |
| **wondertrader** | C++ HFT framework (CTA + micro-price + TWAP/VWAP) | `engine/execution/wondertrader_engine.py` (Python bridge) | FULLY_WIRED |
| **quant-trading** | Stat arb, triangular FX, index arb, 12 technical strategies | `engine/execution/quant_strategy_executor.py` → `ArbitrageDetector`, `UniverseScanner` | FULLY_WIRED |

### Risk — Distress + Credit Analysis

| Sub-repo | Purpose | Engine Consumer | Integration |
|----------|---------|-----------------|-------------|
| **financial-distressed-repo** | Bond analytics, duration, convexity, DV01, Z-spread, OAS | `engine/signals/distressed_asset_engine.py` → `get_credit_analysis()` | FULLY_WIRED |
| **sophisticated-distress-analysis** | Z-prime, bond-level distress, credit spread scanner | `engine/signals/distressed_asset_engine.py` → `scan_distressed_bonds()` | FULLY_WIRED |
| **FinancialDistressPrediction** | GBM distress prediction, Springate S-Score | `engine/signals/distressed_asset_engine.py` → `get_springate_s_score()` | FULLY_WIRED |
| **ai-hedgefund** | Reference AI hedge fund (agents, backtester, strategies) | `engine/signals/cvr_engine.py` (architectural reference) | REFERENCE_ONLY |

### Infrastructure — ML Serving + GPU + LLM

| Sub-repo | Purpose | Engine Consumer | Integration |
|----------|---------|-----------------|-------------|
| **Kserve** | Model serving platform (A/B testing, batch predict) | `engine/ml/bridges/kserve_adapter.py` → `batch_predict()`, `get_model_server()` | FULLY_WIRED |
| **nividia-repo** | GPU-accelerated Monte Carlo VaR, Black-Scholes, portfolio opt | `engine/risk/monte_carlo_risk.py` → `gpu_portfolio_var()`, `GPUAccelerator` | FULLY_WIRED |
| **Air-LLM** | Efficient LLM inference (70B+ on limited VRAM) | `engine/bridges/airllm_model_server.py` → `/analyze-sentiment`, `/analyze-earnings`, `/generate-thesis` | FULLY_WIRED |

### Stage 5: LEARNING — Agent Orchestration

| Sub-repo | Purpose | Engine Consumer | Integration |
|----------|---------|-----------------|-------------|
| **Ruflo-agents** | Multi-agent orchestration framework (TypeScript) | `engine/agents/investor_personas.py`, `engine/agents/agent_monitor.py` | PARTIALLY_WIRED |
| **get-shit-done** | GSD meta-prompting + workflow orchestration (16 templates) | `intelligence_platform/plugins/gsd_workflow_bridge.py`, `engine/live_loop_orchestrator.py` | FULLY_WIRED |

### Cross-cutting

| Sub-repo | Purpose | Engine Consumer | Integration |
|----------|---------|-----------------|-------------|
| **agent_skills** | Claude Skills API + Files API (financial analysis skills) | `engine/signals/security_analysis_engine.py`, `engine/agents/paul_orchestrator.py`, `engine/agents/sector_bots.py`, `engine/agents/research_bots.py`, `engine/ml/model_evaluator.py` | FULLY_WIRED |
| **plugins** | Multi-language bridge (Rust, Go, CUDA, TypeScript, GSD/Paul) | `engine/live_loop_orchestrator.py`, `engine/agents/paul_orchestrator.py`, `engine/agents/enforcement_engine.py` | FULLY_WIRED |
| **qstrader** | Vendored QSTrader backtesting library | `engine/ml/qstrader_backtest_bridge.py` | FULLY_WIRED |

---

## Integration Summary

| Status | Count | Sub-repos |
|--------|-------|-----------|
| FULLY_WIRED | 25 | Financial-Data, open-bb, hedgefund-tracker, FRB, Mav-Analysis, stock-chain, TradeTheEvent, ML-Macro-Market, MiroFish, QLIB, Stock-techincal-prediction-model, AI-Newton, exchange-core, wondertrader, quant-trading, financial-distressed-repo, sophisticated-distress-analysis, FinancialDistressPrediction, Kserve, nividia-repo, Air-LLM, get-shit-done, agent_skills, plugins, qstrader |
| PARTIALLY_WIRED | 1 | Ruflo-agents |
| REFERENCE_ONLY | 5 | EquityLinkedGICPooling, Quant-Developers-Resources, Stock-prediction, CTA-code, ai-hedgefund |

**Total: 31 sub-repos** (25 fully wired, 1 partially wired, 5 reference only)

All imports use `try/except` with graceful degradation — the system runs in degraded mode if any sub-repo is unavailable.

---

## Execution Path (how intelligence_platform feeds the live pipeline)

```
intelligence_platform/
    │
    ├── Data providers (Stage 1):
    │   open-bb → engine/data/openbb_data.py → UniversalDataPool
    │   FRB → engine/signals/fed_liquidity_plumbing.py → FedLiquidityPlumbing
    │   hedgefund-tracker → engine/ml/bridges/ → institutional flow signals
    │
    ├── Signal enrichment (Stage 2 Track A):
    │   ML-Macro-Market → MacroEngine (HMM regime, factors, nowcasting)
    │   Mav-Analysis → SecurityAnalysisEngine (multi-asset technical)
    │   stock-chain → SecurityAnalysisEngine (cross-asset correlation)
    │   TradeTheEvent → EventDrivenEngine (event-driven ML)
    │
    ├── Signal enrichment (Stage 2 Track B):
    │   MiroFish → MiroMomentumEngine (agent-based market sim)
    │
    ├── ML models (Stage 3):
    │   QLIB → AlphaOptimizer (factor mining, Alpha158)
    │   Stock-techincal-prediction-model → MLVoteEnsemble T1
    │   AI-Newton → PatternDiscoveryEngine (symbolic regression)
    │
    ├── Execution (Stage 4):
    │   exchange-core → ExchangeCoreEngine (order matching)
    │   wondertrader → WonderTraderEngine (micro-price + TWAP/VWAP)
    │   quant-trading → QuantStrategyExecutor (12 strategies)
    │   ALL → L7UnifiedExecutionSurface → IBKRBroker (sole broker)
    │
    ├── Risk:
    │   3 × distress repos → DistressedAssetEngine (5-model ensemble)
    │   nividia-repo → Monte Carlo VaR (GPU-accelerated)
    │
    ├── Infrastructure:
    │   Air-LLM → LLM inference (sentiment, earnings, thesis)
    │   Kserve → model serving (A/B testing, batch predict)
    │
    └── Learning (Stage 5):
        get-shit-done → GSD workflow bridge
        plugins → GSD + Paul learning plugins
        agent_skills → agent capability layer
```
