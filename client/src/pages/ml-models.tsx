import { useState, useMemo } from "react";
import { ChevronDown, ChevronRight, Search } from "lucide-react";

// ═══════════ TYPES ═══════════

type ModelType = "LLM" | "ML" | "Statistical" | "Rule-Based" | "Neural Net" | "Ensemble" | "Framework";

interface Engine {
  name: string;
  file: string;
  model: string;
  type: ModelType;
}

interface Layer {
  id: string;
  label: string;
  color: string;
  borderColor: string;
  engines: Engine[];
}

// ═══════════ DATA ═══════════

const LAYERS: Layer[] = [
  {
    id: "l1",
    label: "L1 — Data Ingestion",
    color: "text-[#58a6ff]",
    borderColor: "border-[#58a6ff]/30",
    engines: [
      { name: "OpenBBData", file: "openbb_data.py", model: "OpenBB SDK (34+ providers), FMP API", type: "Framework" },
      { name: "UniverseEngine", file: "universe_engine.py", model: "Rule-based screening", type: "Rule-Based" },
      { name: "CrossAssetUniverse", file: "cross_asset_universe.py", model: "GICS 4-tier classification", type: "Rule-Based" },
      { name: "IngestionOrchestrator", file: "ingestion_orchestrator.py", model: "Data pipeline coordination", type: "Rule-Based" },
      { name: "UniversalPooling", file: "universal_pooling.py", model: "Data aggregation layer", type: "Rule-Based" },
    ],
  },
  {
    id: "l2",
    label: "L2 — Signal Generation",
    color: "text-[#d29922]",
    borderColor: "border-[#d29922]/30",
    engines: [
      { name: "MacroEngine", file: "macro_engine.py", model: "FRED/OpenBB macro indicators, rule-based", type: "Rule-Based" },
      { name: "CVREngine", file: "cvr_engine.py", model: "Binary Option, Barrier Option, Milestone Tree, Monte Carlo (10K paths), Real Options (Black-Scholes)", type: "Statistical" },
      { name: "EventDrivenEngine", file: "event_driven_engine.py", model: "BERT bilevel event classification", type: "ML" },
      { name: "StatArbEngine", file: "stat_arb_engine.py", model: "Cointegration (Engle-Granger), Kalman Filter", type: "Statistical" },
      { name: "DistressedAssetEngine", file: "distressed_asset_engine.py", model: "5-model ensemble: Altman Z-Score, Merton KMV, Ohlson O-Score, Zmijewski, GBM (sklearn)", type: "Ensemble" },
      { name: "ContagionEngine", file: "contagion_engine.py", model: "Monte Carlo stress simulation", type: "Statistical" },
      { name: "SocialPredictionEngine", file: "social_prediction_engine.py", model: "Monte Carlo agent simulation", type: "Statistical" },
      { name: "PatternDiscoveryEngine", file: "pattern_discovery_engine.py", model: "PySR symbolic regression", type: "ML" },
      { name: "SecurityAnalysisEngine", file: "security_analysis_engine.py", model: "Graham-Dodd framework", type: "Rule-Based" },
      { name: "FedLiquidityPlumbing", file: "fed_liquidity_plumbing.py", model: "FRED data, rule-based", type: "Rule-Based" },
      { name: "MetadronCube", file: "metadron_cube.py", model: "Multi-factor synthesis, scenario engine", type: "Ensemble" },
      { name: "AgentSimEngine", file: "agent_sim_engine.py", model: "Monte Carlo N-path simulation", type: "Statistical" },
    ],
  },
  {
    id: "l3",
    label: "L3 — Risk Management",
    color: "text-[#f85149]",
    borderColor: "border-[#f85149]/30",
    engines: [
      { name: "MonteCarloRiskEngine", file: "monte_carlo_risk.py", model: "MiroFish Monte Carlo (VaR, CVaR, stress VaR)", type: "Statistical" },
      { name: "BetaCorridor", file: "beta_corridor.py", model: "Dynamic beta hedging, corridor optimization", type: "Statistical" },
    ],
  },
  {
    id: "l4",
    label: "L4 — Portfolio Construction",
    color: "text-[#bc8cff]",
    borderColor: "border-[#bc8cff]/30",
    engines: [
      { name: "AlphaOptimizer", file: "alpha_optimizer.py", model: "XGBoost/Ridge fallback, sklearn StandardScaler, LinearRegression", type: "ML" },
      { name: "DecisionMatrix", file: "decision_matrix.py", model: "Multi-criteria weighted scoring", type: "Rule-Based" },
      { name: "ConvictionOverride", file: "conviction_override.py", model: "Signal strength override logic", type: "Rule-Based" },
    ],
  },
  {
    id: "l5",
    label: "L5 — ML Pipeline",
    color: "text-[#00d4aa]",
    borderColor: "border-[#00d4aa]/30",
    engines: [
      { name: "DeepLearningEngine", file: "deep_learning_engine.py", model: "Pure-NumPy PPO Agent (RL), no torch/tf", type: "Neural Net" },
      { name: "UniverseClassifier", file: "universe_classifier.py", model: "XGBoost 4-model soft-voting: GaussianNB + GBM + RandomForest + XGBoost (sklearn)", type: "Ensemble" },
      { name: "Backtester", file: "backtester.py", model: "Walk-forward, Monte Carlo, scenario engine", type: "Statistical" },
      { name: "PatternRecognition", file: "pattern_recognition.py", model: "Candlestick/chart pattern detection", type: "Rule-Based" },
      { name: "ModelEvaluator", file: "model_evaluator.py", model: "sklearn metrics (P/R/F1), tier-aware", type: "ML" },
      { name: "ModelStore", file: "model_store.py", model: "sklearn joblib + HMAC-SHA256 integrity", type: "Framework" },
      { name: "SocialFeatures", file: "social_features.py", model: "Feature engineering for social signals", type: "ML" },
    ],
  },
  {
    id: "l5b",
    label: "L5 Bridges — ML Model Adapters",
    color: "text-[#4ecdc4]",
    borderColor: "border-[#4ecdc4]/30",
    engines: [
      { name: "MonteCarloBridge", file: "monte_carlo_bridge.py", model: "ARIMA(1,1,1) + 1000-path Monte Carlo", type: "Statistical" },
      { name: "NvidiaTFTAdapter", file: "nvidia_tft_adapter.py", model: "Temporal Fusion Transformer (multi-horizon)", type: "Neural Net" },
      { name: "StockPredictionBridge", file: "stock_prediction_bridge.py", model: "Evolution Strategy neural net (2-layer, 20→10→1, tanh)", type: "Neural Net" },
      { name: "FinRLBridge", file: "finrl_bridge.py", model: "FinRL deep RL framework adapter", type: "Neural Net" },
      { name: "MarkovRegimeBridge", file: "markov_regime_bridge.py", model: "Hidden Markov Model (hmmlearn)", type: "Statistical" },
      { name: "KServeAdapter", file: "kserve_adapter.py", model: "KServe ML model serving", type: "Framework" },
      { name: "DeepTradingFeatures", file: "deep_trading_features.py", model: "Feature engineering for deep models", type: "ML" },
    ],
  },
  {
    id: "l6",
    label: "L6 — Agent Layer",
    color: "text-[#3fb950]",
    borderColor: "border-[#3fb950]/30",
    engines: [
      { name: "InvestorPersonas", file: "investor_personas.py", model: "Anthropic Claude (claude-opus-4-6), LLM-driven personas with rule-based fallback", type: "LLM" },
      { name: "PaulOrchestrator", file: "paul_orchestrator.py", model: "Agent orchestration, Monte Carlo voter", type: "Statistical" },
      { name: "DynamicAgentFactory", file: "dynamic_agent_factory.py", model: "Agent spawning", type: "Rule-Based" },
      { name: "EnforcementEngine", file: "enforcement_engine.py", model: "Compliance rule engine", type: "Rule-Based" },
      { name: "AgentScorecard", file: "agent_scorecard.py", model: "Performance tracking", type: "Rule-Based" },
      { name: "AgentMonitor", file: "agent_monitor.py", model: "Health monitoring", type: "Rule-Based" },
      { name: "GICSSectorAgents", file: "gics_sector_agents.py", model: "Sector-specific agents", type: "Rule-Based" },
      { name: "ResearchBots", file: "research_bots.py", model: "Research automation", type: "Rule-Based" },
      { name: "SectorBots", file: "sector_bots.py", model: "Sector analysis bots", type: "Rule-Based" },
    ],
  },
  {
    id: "l7",
    label: "L7 — Execution",
    color: "text-[#ff7b72]",
    borderColor: "border-[#ff7b72]/30",
    engines: [
      { name: "ExecutionEngine", file: "execution_engine.py", model: "2-layer neural net (NumPy), Monte Carlo voter, TWAP/VWAP", type: "Neural Net" },
      { name: "L7UnifiedExecutionSurface", file: "l7_unified_execution_surface.py", model: "Unified execution routing", type: "Rule-Based" },
      { name: "OptionsEngine", file: "options_engine.py", model: "Black-Scholes-Merton pricing + Greeks", type: "Statistical" },
      { name: "AlpacaBroker", file: "alpaca_broker.py", model: "Alpaca API execution", type: "Framework" },
      { name: "PaperBroker", file: "paper_broker.py", model: "Paper trading simulation", type: "Rule-Based" },
      { name: "ExchangeCoreEngine", file: "exchange_core_engine.py", model: "Order matching", type: "Rule-Based" },
      { name: "QuantStrategyExecutor", file: "quant_strategy_executor.py", model: "Strategy execution", type: "Rule-Based" },
      { name: "WonderTraderEngine", file: "wondertrader_engine.py", model: "WonderTrader integration", type: "Framework" },
      { name: "MissedOpportunities", file: "missed_opportunities.py", model: "Opportunity tracking", type: "Rule-Based" },
    ],
  },
  {
    id: "monitoring",
    label: "Monitoring Layer",
    color: "text-[#e3b341]",
    borderColor: "border-[#e3b341]/30",
    engines: [
      { name: "LearningLoop", file: "learning_loop.py", model: "GSD + Paul plugin, Monte Carlo voter", type: "Statistical" },
      { name: "LiveDashboard", file: "live_dashboard.py", model: "Real-time monitoring", type: "Rule-Based" },
      { name: "AnomalyDetector", file: "anomaly_detector.py", model: "Z-score anomaly detection", type: "Statistical" },
      { name: "MarketWrap", file: "market_wrap.py", model: "Narrative generation", type: "Rule-Based" },
      { name: "PlatinumReport", file: "platinum_report_v2.py", model: "30-section executive report", type: "Rule-Based" },
      { name: "PortfolioAnalytics", file: "portfolio_analytics.py", model: "Deep analytics", type: "Rule-Based" },
      { name: "SectorTracker", file: "sector_tracker.py", model: "Sector performance", type: "Rule-Based" },
      { name: "L7Dashboard", file: "l7_dashboard.py", model: "Execution dashboard", type: "Rule-Based" },
      { name: "HeatmapEngine", file: "heatmap_engine.py", model: "GICS sector heatmap", type: "Rule-Based" },
      { name: "MemoryMonitor", file: "memory_monitor.py", model: "Session tracking", type: "Rule-Based" },
    ],
  },
  {
    id: "intel",
    label: "Intelligence Platform",
    color: "text-[#f778ba]",
    borderColor: "border-[#f778ba]/30",
    engines: [
      { name: "Air-LLM", file: "air_llm/", model: "LLM inference engine", type: "LLM" },
      { name: "CAMEL-AI/OASIS", file: "camel_ai/", model: "Multi-agent simulation", type: "LLM" },
      { name: "FinBERT", file: "finbert/", model: "Financial sentiment NLP", type: "ML" },
      { name: "QLIB", file: "qlib/", model: "Quantitative investment library", type: "Framework" },
      { name: "KServe", file: "kserve/", model: "ML model serving (GPU inference)", type: "Framework" },
      { name: "Ruflo Agents", file: "ruflo_agents/", model: "Swarm agent framework", type: "LLM" },
      { name: "MiroFish", file: "mirofish/", model: "Agent-based market simulation", type: "Statistical" },
    ],
  },
];

// ═══════════ MODEL TYPE BADGE CONFIG ═══════════

const TYPE_BADGE: Record<ModelType, { bg: string; text: string; dot: string }> = {
  LLM:         { bg: "bg-[#f778ba]/10 border border-[#f778ba]/30", text: "text-[#f778ba]",   dot: "bg-[#f778ba]" },
  ML:          { bg: "bg-[#58a6ff]/10 border border-[#58a6ff]/30", text: "text-[#58a6ff]",   dot: "bg-[#58a6ff]" },
  Statistical: { bg: "bg-[#d29922]/10 border border-[#d29922]/30", text: "text-[#d29922]",   dot: "bg-[#d29922]" },
  "Rule-Based":{ bg: "bg-[#484f58]/20 border border-[#484f58]/50", text: "text-[#8b949e]",   dot: "bg-[#8b949e]" },
  "Neural Net":{ bg: "bg-[#bc8cff]/10 border border-[#bc8cff]/30", text: "text-[#bc8cff]",   dot: "bg-[#bc8cff]" },
  Ensemble:    { bg: "bg-[#00d4aa]/10 border border-[#00d4aa]/30", text: "text-[#00d4aa]",   dot: "bg-[#00d4aa]" },
  Framework:   { bg: "bg-[#4ecdc4]/10 border border-[#4ecdc4]/30", text: "text-[#4ecdc4]",   dot: "bg-[#4ecdc4]" },
};

function ModelTypeBadge({ type }: { type: ModelType }) {
  const cfg = TYPE_BADGE[type];
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-medium font-mono tracking-wider ${cfg.bg} ${cfg.text}`}>
      <span className={`w-1 h-1 rounded-full flex-shrink-0 ${cfg.dot}`} />
      {type}
    </span>
  );
}

// ═══════════ LAYER SECTION ═══════════

function LayerSection({
  layer,
  query,
}: {
  layer: Layer;
  query: string;
}) {
  const [open, setOpen] = useState(true);

  const filtered = useMemo(() => {
    if (!query) return layer.engines;
    const q = query.toLowerCase();
    return layer.engines.filter(
      (e) =>
        e.name.toLowerCase().includes(q) ||
        e.file.toLowerCase().includes(q) ||
        e.model.toLowerCase().includes(q) ||
        e.type.toLowerCase().includes(q)
    );
  }, [layer.engines, query]);

  if (query && filtered.length === 0) return null;

  return (
    <div className={`border ${layer.borderColor} rounded overflow-hidden`} data-testid={`layer-section-${layer.id}`}>
      {/* Header */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-[#0d1117] hover:bg-[#161b22] transition-colors text-left"
        data-testid={`layer-toggle-${layer.id}`}
      >
        <span className="text-[#484f58]">
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </span>
        <span className={`text-[11px] font-semibold font-mono tracking-wider uppercase ${layer.color}`}>
          {layer.label}
        </span>
        <span className="ml-auto text-[9px] text-[#484f58] font-mono tabular-nums">
          {filtered.length} {filtered.length === 1 ? "engine" : "engines"}
        </span>
      </button>

      {/* Table */}
      {open && (
        <div className="overflow-x-auto">
          <table className="w-full text-[9px] font-mono">
            <thead>
              <tr className="border-b border-[#1e2633] bg-[#0a0e17]">
                <th className="text-left px-3 py-1.5 text-[#484f58] uppercase tracking-wider font-medium w-[180px]">Engine</th>
                <th className="text-left px-3 py-1.5 text-[#484f58] uppercase tracking-wider font-medium w-[200px]">File</th>
                <th className="text-left px-3 py-1.5 text-[#484f58] uppercase tracking-wider font-medium">Model / Framework</th>
                <th className="text-left px-3 py-1.5 text-[#484f58] uppercase tracking-wider font-medium w-[110px]">Type</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((engine, i) => (
                <tr
                  key={engine.name}
                  className="border-b border-[#1e2633]/50 hover:bg-[#161b22]/60 transition-colors"
                  data-testid={`engine-row-${engine.name.toLowerCase().replace(/\s+/g, "-")}`}
                >
                  <td className="px-3 py-2">
                    <span className="text-[#e6edf3] font-medium">{engine.name}</span>
                  </td>
                  <td className="px-3 py-2">
                    <span className="text-[#8b949e]">{engine.file}</span>
                  </td>
                  <td className="px-3 py-2">
                    <span className="text-[#8b949e] leading-relaxed">{engine.model}</span>
                  </td>
                  <td className="px-3 py-2">
                    <ModelTypeBadge type={engine.type} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ═══════════ SUMMARY STATS ═══════════

function SummaryStats() {
  const allEngines = LAYERS.flatMap((l) => l.engines);
  const totalEngines = allEngines.length;
  const mlModels = allEngines.filter((e) => e.type === "ML" || e.type === "Neural Net" || e.type === "Ensemble").length;
  const llmModels = allEngines.filter((e) => e.type === "LLM").length;
  const frameworks = allEngines.filter((e) => e.type === "Framework").length;

  const stats = [
    { label: "Total Engines", value: totalEngines, color: "text-[#e6edf3]" },
    { label: "ML Models", value: mlModels, color: "text-[#58a6ff]" },
    { label: "LLM Models", value: llmModels, color: "text-[#f778ba]" },
    { label: "Frameworks", value: frameworks, color: "text-[#4ecdc4]" },
    { label: "Layers", value: LAYERS.length, color: "text-[#00d4aa]" },
  ];

  return (
    <div className="flex items-center gap-4 flex-wrap">
      {stats.map((s, i) => (
        <div key={i} className="flex items-center gap-2 bg-[#0d1117] border border-[#1e2633] rounded px-3 py-1.5" data-testid={`stat-${s.label.toLowerCase().replace(/\s+/g, "-")}`}>
          <span className={`text-xl font-bold font-mono tabular-nums ${s.color}`}>{s.value}</span>
          <span className="text-[9px] text-[#484f58] uppercase tracking-wider font-mono">{s.label}</span>
        </div>
      ))}
    </div>
  );
}

// ═══════════ TYPE FILTER PILLS ═══════════

const ALL_TYPES: ModelType[] = ["LLM", "ML", "Statistical", "Rule-Based", "Neural Net", "Ensemble", "Framework"];

function TypeFilterPills({
  active,
  onToggle,
}: {
  active: ModelType | null;
  onToggle: (t: ModelType | null) => void;
}) {
  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      <button
        onClick={() => onToggle(null)}
        className={`px-2 py-0.5 rounded text-[9px] font-mono font-medium tracking-wider transition-colors ${
          active === null
            ? "bg-[#00d4aa]/15 text-[#00d4aa] border border-[#00d4aa]/30"
            : "bg-[#0d1117] text-[#484f58] border border-[#1e2633] hover:border-[#484f58] hover:text-[#8b949e]"
        }`}
      >
        ALL
      </button>
      {ALL_TYPES.map((t) => {
        const cfg = TYPE_BADGE[t];
        const isActive = active === t;
        return (
          <button
            key={t}
            onClick={() => onToggle(isActive ? null : t)}
            className={`px-2 py-0.5 rounded text-[9px] font-mono font-medium tracking-wider transition-colors ${
              isActive
                ? `${cfg.bg} ${cfg.text}`
                : "bg-[#0d1117] text-[#484f58] border border-[#1e2633] hover:border-[#484f58] hover:text-[#8b949e]"
            }`}
            data-testid={`filter-type-${t.toLowerCase().replace(/\s+/g, "-")}`}
          >
            {t}
          </button>
        );
      })}
    </div>
  );
}

// ═══════════ MAIN PAGE ═══════════

export default function MLModelsPage() {
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<ModelType | null>(null);

  // Combined query string for filtering
  const query = useMemo(() => {
    if (typeFilter && search) return `${search}`;
    return search;
  }, [search, typeFilter]);

  // Filter layers based on type filter
  const filteredLayers = useMemo(() => {
    if (!typeFilter) return LAYERS;
    return LAYERS.map((layer) => ({
      ...layer,
      engines: layer.engines.filter((e) => e.type === typeFilter),
    })).filter((layer) => layer.engines.length > 0);
  }, [typeFilter]);

  return (
    <div className="h-full flex flex-col bg-[#0a0e17] overflow-hidden" data-testid="ml-models-page">
      {/* Header */}
      <div className="flex-shrink-0 px-4 pt-4 pb-3 border-b border-[#1e2633] bg-[#0d1117]">
        <div className="flex items-center gap-3 mb-3">
          {/* Icon */}
          <div className="w-7 h-7 rounded border border-[#00d4aa]/30 bg-[#00d4aa]/5 flex items-center justify-center flex-shrink-0">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <circle cx="7" cy="7" r="2" fill="#00d4aa" fillOpacity="0.8" />
              <circle cx="2" cy="4" r="1.2" fill="#00d4aa" fillOpacity="0.5" />
              <circle cx="12" cy="4" r="1.2" fill="#00d4aa" fillOpacity="0.5" />
              <circle cx="2" cy="10" r="1.2" fill="#00d4aa" fillOpacity="0.5" />
              <circle cx="12" cy="10" r="1.2" fill="#00d4aa" fillOpacity="0.5" />
              <line x1="4.2" y1="5.2" x2="6" y2="6.5" stroke="#00d4aa" strokeWidth="0.8" strokeOpacity="0.5" />
              <line x1="9.8" y1="5.2" x2="8" y2="6.5" stroke="#00d4aa" strokeWidth="0.8" strokeOpacity="0.5" />
              <line x1="4.2" y1="8.8" x2="6" y2="7.5" stroke="#00d4aa" strokeWidth="0.8" strokeOpacity="0.5" />
              <line x1="9.8" y1="8.8" x2="8" y2="7.5" stroke="#00d4aa" strokeWidth="0.8" strokeOpacity="0.5" />
            </svg>
          </div>
          <div>
            <h1 className="text-[13px] font-semibold text-[#e6edf3] font-mono tracking-[0.1em] uppercase">
              ML Models
            </h1>
            <p className="text-[9px] text-[#484f58] font-mono mt-0.5">
              Engine-to-model mapping across L1–L7, Monitoring &amp; Intelligence Platform
            </p>
          </div>
        </div>

        <SummaryStats />
      </div>

      {/* Search + Filter bar */}
      <div className="flex-shrink-0 px-4 py-2.5 border-b border-[#1e2633] bg-[#0a0e17] flex items-center gap-3 flex-wrap">
        <div className="relative flex-shrink-0">
          <Search size={11} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[#484f58]" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search engines, models, files..."
            className="pl-7 pr-3 py-1.5 w-[260px] bg-[#0d1117] border border-[#1e2633] rounded text-[10px] font-mono text-[#e6edf3] placeholder-[#484f58] focus:outline-none focus:border-[#00d4aa]/50 focus:ring-0 transition-colors"
            data-testid="search-input"
          />
        </div>
        <TypeFilterPills active={typeFilter} onToggle={setTypeFilter} />
        {(search || typeFilter) && (
          <button
            onClick={() => { setSearch(""); setTypeFilter(null); }}
            className="text-[9px] font-mono text-[#484f58] hover:text-[#8b949e] transition-colors"
            data-testid="clear-filters"
          >
            CLEAR
          </button>
        )}
      </div>

      {/* Layer sections */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2" data-testid="layers-container">
        {filteredLayers.map((layer) => (
          <LayerSection key={layer.id} layer={layer} query={query} />
        ))}
        {filteredLayers.length === 0 && (
          <div className="flex items-center justify-center h-32 text-[10px] text-[#484f58] font-mono">
            No engines match your filter
          </div>
        )}
      </div>
    </div>
  );
}
