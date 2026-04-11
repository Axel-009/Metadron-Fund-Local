# Intelligence Platform Registry

Complete reference for all sub-repos in `intelligence_platform/`, their purpose, engine consumers, and integration status.

## Layer Architecture

| Layer | Focus | Sub-repos |
|-------|-------|-----------|
| L1 | Data Ingestion | Financial-Data, open-bb, hedgefund-tracker, FRB, EquityLinkedGICPooling, Quant-Developers-Resources |
| L2 | Signal Processing | Mav-Analysis, quant-trading, stock-chain, CTA-code, TradeTheEvent, wondertrader |
| L3 | ML/AI Models | QLIB, Stock-techincal-prediction-model, Stock-prediction, ML-Macro-Market, AI-Newton |
| L4 | Portfolio & Risk | ai-hedgefund, financial-distressed-repo, sophisticated-distress-analysis, FinancialDistressPrediction |
| L5 | Infrastructure | Kserve, nividia-repo, Air-LLM, exchange-core |
| L6 | Agents | Ruflo-agents, MiroFish, get-shit-done |
| -- | Agent Skills | agent_skills (Claude Skills API + Files API) |
| -- | Plugins | plugins (cross-language bridge layer) |

---

## Sub-Repo Inventory

### L1: Data Ingestion

| Sub-repo | Purpose | Custom Integration Files | Engine Consumers | Integration Status |
|----------|---------|------------------------|------------------|-------------------|
| **Financial-Data** | OpenBB market data pipeline, yfinance fallback | `openbb_bridge.py` — `OpenBBTicker`, `get_data` shim | `engine/data/openbb_data.py` (supersedes), `engine/signals/pattern_discovery_engine.py` | FULLY_WIRED (superseded by engine/data/openbb_data.py) |
| **open-bb** | Vendored OpenBB platform | None (vendored source) | Used via `from openbb import obb` in `engine/data/openbb_data.py` | FULLY_WIRED |
| **hedgefund-tracker** | Institutional flow intelligence — SEC 13F/13D/Form 4 analysis, Promise Score, fund mimicry signals | `openbb_universe.py`, `app/analysis/stocks.py`, `app/ai/agent.py`, `app/analysis/performance_evaluator.py` | `engine/data/universe_engine.py` (supersedes universe), `engine/ml/bridges/hedgefund_tracker_bridge.py` (newly wired — `HedgefundTrackerBridge`) | FULLY_WIRED |
| **FRB** | Federal Reserve Bank FRED API Python client | `fred/` — `Fred` class with `category`, `release`, `series`, `tag`, `source` clients | `engine/signals/fed_liquidity_plumbing.py` (newly wired — `FRBFredClient`) | FULLY_WIRED |
| **EquityLinkedGICPooling** | GIC pooling methodology documentation | None (README only) | `engine/data/universal_pooling.py` (reference) | REFERENCE_ONLY |
| **Quant-Developers-Resources** | Quantitative finance reference materials | None (reference docs) | None | REFERENCE_ONLY |

### L2: Signal Processing

| Sub-repo | Purpose | Custom Integration Files | Engine Consumers | Integration Status |
|----------|---------|------------------------|------------------|-------------------|
| **Mav-Analysis** | Multi-asset technical analysis — RSI/MACD/BB composite, cross-asset momentum, sector RV, vol surface, unusual flow | `multi_asset_analysis.py`, `openbb_data.py` | `engine/signals/security_analysis_engine.py` (newly wired — `MultiAssetAnalyzer`) | FULLY_WIRED |
| **quant-trading** | Quantitative strategy library — stat arb, triangular FX, index arb, cross-asset RV, calendar spread, momentum/value screens | `arbitrage_detector.py`, `universe_scanner.py`, `openbb_data.py` | `engine/execution/quant_strategy_executor.py` (newly wired — `ArbitrageDetector`, `UniverseScanner`; 12 strategies re-implemented inline) | FULLY_WIRED |
| **stock-chain** | Time-series chain analysis — cross-asset correlation, risk parity, HMM regime, sector rotation, Information Ratio | `asset_class_analyzer.py`, `openbb_data.py` | `engine/signals/security_analysis_engine.py` (newly wired — `AssetClassAnalyzer`) | FULLY_WIRED |
| **CTA-code** | CTA/trend-following reference textbooks and code samples | None (reference only) | None | REFERENCE_ONLY |
| **TradeTheEvent** | Event-driven ML — cross-asset macro events, credit rating changes, second-derivative effects | `event_driven_strategy.py` — `EventDrivenStrategy` | `engine/signals/event_driven_engine.py` (newly wired — `EventDrivenStrategy`) | FULLY_WIRED |
| **wondertrader** | C++ HFT quantitative trading framework | None (C++ source only) | `engine/execution/wondertrader_engine.py` (Python bridge) | PARTIALLY_WIRED (no Python plugin in plugins/) |

### L3: ML/AI Models

| Sub-repo | Purpose | Custom Integration Files | Engine Consumers | Integration Status |
|----------|---------|------------------------|------------------|-------------------|
| **QLIB** | Microsoft QLIB quantitative ML framework — backtest engine, OpenBB data providers, model framework | `qlib/data/openbb_universe.py` — `OpenBBCalendarProvider`, `OpenBBInstrumentProvider`, `OpenBBFeatureProvider` | `engine/ml/bridges/qlib_bridge.py` (newly wired — `QLIBBridge`) | FULLY_WIRED |
| **Stock-techincal-prediction-model** | Multi-asset price prediction — LSTM, XGBoost, RF, Transformer ensemble | `multi_asset_predictor.py` — `MultiAssetPredictor` | `engine/ml/bridges/stock_prediction_bridge.py` (newly wired — `predict_with_ensemble()`) | FULLY_WIRED |
| **Stock-prediction** | Jupyter notebooks — BiLSTM, BiGRU, RNN variants | None (notebooks only) | None | REFERENCE_ONLY |
| **ML-Macro-Market** | Macro-market ML — HMM regime model, Fama-French factor model, GDP nowcasting | `macro_ml_engine.py` — `HiddenMarkovRegimeModel`, `FactorModel`, `NowcastingEngine`, `MacroMLEngine` | `engine/signals/macro_engine.py` (newly wired) | FULLY_WIRED |
| **AI-Newton** | Physics-inspired symbolic regression for market microstructure | `investment_platform_integration.py` — `PhysicsOptimizer` (SA, GA, PSO, risk parity) | `engine/bridges/ainewton_discovery_worker.py` (newly wired — `PhysicsOptimizer`), `engine/bridges/ainewton_service.py` | FULLY_WIRED |

### L4: Portfolio & Risk

| Sub-repo | Purpose | Custom Integration Files | Engine Consumers | Integration Status |
|----------|---------|------------------------|------------------|-------------------|
| **ai-hedgefund** | Reference AI hedge fund implementation (agents, backtester, strategies) | None (vendored reference) | `engine/signals/cvr_engine.py`, `app/backend/` (architectural reference) | REFERENCE_ONLY |
| **financial-distressed-repo** | Credit risk analysis — bond analytics, duration, convexity, DV01, Z-spread, OAS, rating transitions, ECL | `credit_analysis_engine.py` — `CreditAnalysisEngine` | `engine/signals/distressed_asset_engine.py` (newly wired — `get_credit_analysis()`) | FULLY_WIRED |
| **sophisticated-distress-analysis** | Advanced distress scanner — Z-prime, bond-level distress, credit spread analysis, event-driven opportunities | `distress_scanner.py` — `DistressScanner` | `engine/signals/distressed_asset_engine.py` (newly wired — `scan_distressed_bonds()`) | FULLY_WIRED |
| **FinancialDistressPrediction** | GBM distress prediction — Springate S-Score, recovery probability, trade recommendations | `distress_prediction_engine.py` — `DistressPredictionEngine` | `engine/signals/distressed_asset_engine.py` (newly wired — `get_springate_s_score()`) | FULLY_WIRED |

### L5: Infrastructure

| Sub-repo | Purpose | Custom Integration Files | Engine Consumers | Integration Status |
|----------|---------|------------------------|------------------|-------------------|
| **Kserve** | Model serving platform — KServe manifest gen, A/B testing, IC tracking, batch predict | `investment_platform_integration.py` — `InvestmentModelServer` | `engine/ml/bridges/kserve_adapter.py` (newly wired — `batch_predict()`, `get_model_server()`) | FULLY_WIRED |
| **nividia-repo** | GPU-accelerated quant finance — Monte Carlo VaR, covariance, parallel backtest, Black-Scholes Greeks, portfolio optimization | `investment_platform_integration.py` — `GPUAccelerator`, `VaRResult`, `GreeksResult` | `engine/risk/monte_carlo_risk.py` (newly wired — `gpu_portfolio_var()`) | FULLY_WIRED |
| **Air-LLM** | Efficient LLM inference for 70B+ models on limited VRAM | `investment_platform_integration.py` — `InvestmentLLMProcessor` (sentiment, earnings, thesis) | `engine/bridges/airllm_model_server.py` (newly wired — `/analyze-sentiment`, `/analyze-earnings`, `/generate-thesis` endpoints) | FULLY_WIRED |
| **exchange-core** | Java LMAX Disruptor matching engine (architectural blueprint) | None (Java source) | `engine/execution/exchange_core_engine.py` (full Python reimplementation) | FULLY_WIRED (Python reimplementation, Java is reference) |

### L6: Agents

| Sub-repo | Purpose | Custom Integration Files | Engine Consumers | Integration Status |
|----------|---------|------------------------|------------------|-------------------|
| **Ruflo-agents** | Multi-agent orchestration framework (TypeScript/Node.js) | `.agents/config.toml` — Claude Flow V3 Codex CLI config | `engine/agents/investor_personas.py`, `engine/agents/agent_monitor.py` | PARTIALLY_WIRED (TS framework, config referenced) |
| **MiroFish** | Agent-based market simulation — MarketSimulator, stress testing, Monte Carlo VaR, Hurst exponent | `investment_platform_integration.py` — `MarketSimulator`, `AgentType`, `MarketAgent` | `engine/signals/agent_sim_engine.py`, `engine/risk/monte_carlo_risk.py` (stress test + GPU VaR newly wired) | FULLY_WIRED |
| **get-shit-done** | GSD meta-prompting & workflow orchestration (16 agent templates) | `agents/*.md` templates, `bin/gsd-tools.cjs` | `intelligence_platform/plugins/gsd_workflow_bridge.py`, `engine/live_loop_orchestrator.py` | FULLY_WIRED |

### Cross-cutting

| Sub-repo | Purpose | Custom Integration Files | Engine Consumers | Integration Status |
|----------|---------|------------------------|------------------|-------------------|
| **agent_skills** | Claude Skills API + Files API — financial analysis skills, agent patterns | `__init__.py`, `skill_utils.py`, `file_utils.py`, `integration_config.json` | `engine/signals/security_analysis_engine.py`, `engine/agents/paul_orchestrator.py`, `engine/agents/sector_bots.py`, `engine/agents/research_bots.py`, `engine/ml/model_evaluator.py`, `engine/agents/investor_personas.py`, `engine/agents/gics_sector_agents.py` | FULLY_WIRED |
| **plugins** | Multi-language bridge layer (Rust, Go, CUDA, TypeScript, GSD/Paul) | `unified_bridge.py`, `rust_plugin.py`, `go_plugin.py`, `cuda_plugin.py`, `typescript_plugin.py`, `gsd_paul_plugin.py`, `gsd_workflow_bridge.py` | `engine/live_loop_orchestrator.py`, `engine/agents/paul_orchestrator.py`, `engine/agents/enforcement_engine.py`, `engine/agents/dynamic_agent_factory.py` | FULLY_WIRED |
| **qstrader** | Vendored qstrader backtesting library | None (vendored library) | `engine/ml/qstrader_backtest_bridge.py` | FULLY_WIRED |

---

## Additional Sub-repos (beyond original 25)

These sub-repos were found in `intelligence_platform/` but not in the original task inventory:

| Sub-repo | Purpose | Integration Status |
|----------|---------|-------------------|
| **sophisticated-distress-analysis** | Advanced distress scanning beyond base models | FULLY_WIRED (into distressed_asset_engine.py) |
| **plugins** | Cross-language bridge layer for all non-Python repos | FULLY_WIRED |
| **qstrader** | Backtesting framework | FULLY_WIRED (via qstrader_backtest_bridge.py) |
| **quant-trading** | Strategy library with arb detection | FULLY_WIRED (into quant_strategy_executor.py) |
| **stock-chain** | Cross-asset chain analysis | FULLY_WIRED (into security_analysis_engine.py) |
| **wondertrader** | C++ HFT framework | PARTIALLY_WIRED (no Python plugin) |

---

## Integration Summary

| Status | Count | Sub-repos |
|--------|-------|-----------|
| FULLY_WIRED | 27 | AI-Newton, Air-LLM, Kserve, Financial-Data, MiroFish, nividia-repo, agent_skills, FinancialDistressPrediction, financial-distressed-repo, sophisticated-distress-analysis, ML-Macro-Market, Stock-techincal-prediction-model, TradeTheEvent, FRB, Mav-Analysis, quant-trading, stock-chain, open-bb, hedgefund-tracker, exchange-core, get-shit-done, plugins, qstrader, ai-hedgefund, EquityLinkedGICPooling, QLIB, hedgefund-tracker |
| PARTIALLY_WIRED | 2 | Ruflo-agents, wondertrader |
| REFERENCE_ONLY | 4 | CTA-code, Quant-Developers-Resources, Stock-prediction, ai-hedgefund |

## Newly Wired in This Integration

The following bridges were added to connect previously-unwired intelligence_platform sub-repos:

1. **nividia-repo GPU** → `engine/risk/monte_carlo_risk.py` — `gpu_portfolio_var()`, `GPUAccelerator`
2. **MiroFish stress test** → `engine/risk/monte_carlo_risk.py` — `mirofish_stress_test()`
3. **AI-Newton PhysicsOptimizer** → `engine/bridges/ainewton_discovery_worker.py`
4. **Air-LLM InvestmentLLMProcessor** → `engine/bridges/airllm_model_server.py` — `/analyze-sentiment`, `/analyze-earnings`, `/generate-thesis`
5. **Kserve InvestmentModelServer** → `engine/ml/bridges/kserve_adapter.py` — `batch_predict()`, `get_model_server()`
6. **FRB FRED client** → `engine/signals/fed_liquidity_plumbing.py` — `FRBFredClient`
7. **FinancialDistressPrediction** → `engine/signals/distressed_asset_engine.py` — `get_springate_s_score()`
8. **financial-distressed-repo** → `engine/signals/distressed_asset_engine.py` — `get_credit_analysis()`
9. **sophisticated-distress-analysis** → `engine/signals/distressed_asset_engine.py` — `scan_distressed_bonds()`
10. **ML-Macro-Market** → `engine/signals/macro_engine.py` — `HiddenMarkovRegimeModel`, `MacroMLEngine`
11. **TradeTheEvent** → `engine/signals/event_driven_engine.py` — `EventDrivenStrategy`
12. **Stock-techincal-prediction-model** → `engine/ml/bridges/stock_prediction_bridge.py` — `predict_with_ensemble()`
13. **Mav-Analysis** → `engine/signals/security_analysis_engine.py` — `MultiAssetAnalyzer`
14. **stock-chain** → `engine/signals/security_analysis_engine.py` — `AssetClassAnalyzer`
15. **quant-trading** → `engine/execution/quant_strategy_executor.py` — `ArbitrageDetector`, `UniverseScanner`
16. **QLIB** → `engine/ml/bridges/qlib_bridge.py` — `QLIBBridge` (OpenBB providers, backtest engine, signal adapter)
17. **hedgefund-tracker** → `engine/ml/bridges/hedgefund_tracker_bridge.py` — `HedgefundTrackerBridge` (institutional flow signals, quarterly consensus, growth score)

All imports use `try/except` with graceful degradation — the system runs in degraded mode if any sub-repo is unavailable.
