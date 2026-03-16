# Intelligence Platform — Reference Repository Collection

All 21 reference repositories consolidated into Metadron Capital as a unified
Python-coherent intelligence platform. Non-Python source code (Go, Rust, C++,
TypeScript, JavaScript, Vue, Java) has been excluded to maintain language coherence
across the entire Metadron Capital codebase.

## Repository Index

| # | Repository | Focus | Key Capabilities |
|---|-----------|-------|-------------------|
| 1 | **AI-Newton** | Physics-informed ML | Neural ODEs, scientific computing |
| 2 | **Air-LLM** | LLM inference | Memory-efficient large model serving |
| 3 | **CTA-code** | Commodity Trading Advisors | Trend-following, momentum strategies |
| 4 | **Financial-Data** | Data pipelines | Financial data ingestion, cleaning |
| 5 | **Kserve** | ML model serving | KServe inference, model deployment (Python SDK only) |
| 6 | **ML-Macro-Market** | Macro ML models | Macro factor prediction, regime detection |
| 7 | **Mav-Analysis** | Maverick analysis | Portfolio analytics, risk modeling |
| 8 | **QLIB** | Quant library | Microsoft QLIB framework, factor research |
| 9 | **Ruflo-agents** | Agent orchestration | Multi-agent workflows (Python components only) |
| 10 | **Stock-techincal-prediction-model** | Technical analysis | Price prediction, pattern recognition |
| 11 | **ai-hedgefund** | AI hedge fund | Full hedge fund simulation, alpha models |
| 12 | **financial-distressed-repo** | Distress analysis | Financial distress prediction baseline |
| 13 | **hedgefund-tracker** | Fund tracking | Hedge fund performance monitoring |
| 14 | **nividia-repo** | GPU/AI compute | NVIDIA ML pipelines, deep learning |
| 15 | **open-bb** | OpenBB terminal | Financial data terminal, research platform |
| 16 | **quant-trading** | Quant strategies | Algorithmic trading strategies |
| 17 | **sophisticated-distress-analysis** | Advanced distress | Enhanced distress modeling |
| 18 | **stock-chain** | Blockchain + stocks | DeFi/TradFi integration |
| 19 | **MiroFish** | Social prediction | Agent-based social sentiment simulation |
| 20 | **FinancialDistressPrediction** | Distress ML | GBM bankruptcy prediction (reference for DistressedAssetEngine) |
| 21 | **TradeTheEvent** | Event-driven ML | BERT event detection (reference for EventDrivenEngine) |

## Integration into Metadron Capital

These repositories serve as reference implementations. The actual production engines
in `engine/signals/` are institutional-grade implementations that far exceed the
reference repos:

- **DistressedAssetEngine** (`engine/signals/distressed_asset_engine.py`)
  - References: `FinancialDistressPrediction/`, `financial-distressed-repo/`, `sophisticated-distress-analysis/`
  - Elevation: Single GBM → 5-model ensemble (Altman Z, Merton KMV, Ohlson O, Zmijewski, ML GBM)

- **CVREngine** (`engine/signals/cvr_engine.py`)
  - References: `ai-hedgefund/`, `Financial-Data/`
  - Elevation: Basic valuation → 5-model CVR (binary option, barrier, milestone tree, Monte Carlo, real options)

- **EventDrivenEngine** (`engine/signals/event_driven_engine.py`)
  - References: `TradeTheEvent/`, `CTA-code/`, `quant-trading/`
  - Elevation: BERT classification → 12-category quantitative models (Mitchell-Pulvino, SUE PEAD, Kelly)

- **SocialPredictionEngine** (`engine/signals/social_prediction_engine.py`)
  - References: `MiroFish/`, `Ruflo-agents/`
  - Elevation: Raw social data → agent-based sentiment simulation with topic modeling

## File Statistics

- **8,643 Python files** — core source code
- **3,120 Markdown files** — documentation
- **1,267 Shell scripts** — build/deploy automation
- **1,114 YAML configs** — configuration
- **963 CSV files** — data samples
- **228 Jupyter notebooks** — research/analysis

All files are Python-ecosystem coherent. No Go, Rust, C++, TypeScript, JavaScript,
or other non-Python source code is included.
