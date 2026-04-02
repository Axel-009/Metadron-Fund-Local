import { useState, useMemo, useEffect } from "react";
import { ChevronDown, ChevronRight, Search } from "lucide-react";
import { useEngineQuery } from "@/hooks/use-engine-api";

// ═══════════ TYPES ═══════════

type ModelType = "LLM" | "ML" | "Statistical" | "Rule-Based" | "Neural Net" | "Ensemble" | "Framework";
type EngineStatus = "ACTIVE" | "STANDBY" | "OFFLINE";

interface Engine {
  name: string;
  file: string;
  model: string;
  type: ModelType;
  status: EngineStatus;
  usage: string;
}

interface Layer {
  id: string;
  label: string;
  color: string;
  borderColor: string;
  engines: Engine[];
}

// ═══════════ USAGE HELPERS ═══════════

function usageForType(type: ModelType, seed: number): string {
  // Deterministic but varied values based on seed
  const vals = [
    ["2.4M tokens", "890K tokens", "1.2M tokens", "3.1M tokens", "540K tokens"],
    ["94.2% eff.", "87.5% eff.", "91.0% eff.", "78.3% eff.", "96.1% eff."],
    ["12.4K sims", "8.2K sims", "24.1K sims", "5.8K sims", "18.7K sims"],
    ["48.2K calls", "124K calls", "31.5K calls", "89.4K calls", "12.1K calls"],
    ["6.8K infer.", "14.2K infer.", "3.9K infer.", "22.4K infer.", "9.1K infer."],
    ["124K calls", "58.3K calls", "201K calls", "77.1K calls", "45.8K calls"],
  ];
  const typeMap: Record<ModelType, number> = {
    LLM: 0,
    ML: 1,
    Statistical: 2,
    "Rule-Based": 3,
    "Neural Net": 4,
    Ensemble: 1,
    Framework: 5,
  };
  const row = vals[typeMap[type]];
  return row[seed % row.length];
}

// ═══════════ DATA ═══════════

const LAYERS: Layer[] = [
  {
    id: "l1",
    label: "L1 — Data Ingestion",
    color: "text-[#58a6ff]",
    borderColor: "border-[#58a6ff]/30",
    engines: [
      { name: "OpenBBData", file: "openbb_data.py", model: "OpenBB SDK (34+ providers), FMP API", type: "Framework", status: "ACTIVE", usage: usageForType("Framework", 0) },
      { name: "UniverseEngine", file: "universe_engine.py", model: "Rule-based screening", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 1) },
      { name: "CrossAssetUniverse", file: "cross_asset_universe.py", model: "GICS 4-tier classification", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 2) },
      { name: "IngestionOrchestrator", file: "ingestion_orchestrator.py", model: "Data pipeline coordination", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 3) },
      { name: "UniversalPooling", file: "universal_pooling.py", model: "Data aggregation layer", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 4) },
    ],
  },
  {
    id: "l2",
    label: "L2 — Signal Generation",
    color: "text-[#d29922]",
    borderColor: "border-[#d29922]/30",
    engines: [
      { name: "MacroEngine", file: "macro_engine.py", model: "FRED/OpenBB macro indicators, rule-based", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 0) },
      { name: "CVREngine", file: "cvr_engine.py", model: "Binary Option, Barrier Option, Milestone Tree, Monte Carlo (10K paths), Real Options (Black-Scholes)", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 0) },
      { name: "EventDrivenEngine", file: "event_driven_engine.py", model: "BERT bilevel event classification", type: "ML", status: "ACTIVE", usage: usageForType("ML", 0) },
      { name: "StatArbEngine", file: "stat_arb_engine.py", model: "Cointegration (Engle-Granger), Kalman Filter", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 1) },
      { name: "DistressedAssetEngine", file: "distressed_asset_engine.py", model: "5-model ensemble: Altman Z-Score, Merton KMV, Ohlson O-Score, Zmijewski, GBM (sklearn)", type: "Ensemble", status: "ACTIVE", usage: usageForType("Ensemble", 0) },
      { name: "ContagionEngine", file: "contagion_engine.py", model: "Monte Carlo stress simulation", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 2) },
      { name: "SocialPredictionEngine", file: "social_prediction_engine.py", model: "Monte Carlo agent simulation", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 3) },
      { name: "PatternDiscoveryEngine", file: "pattern_discovery_engine.py", model: "PySR symbolic regression", type: "ML", status: "ACTIVE", usage: usageForType("ML", 1) },
      { name: "SecurityAnalysisEngine", file: "security_analysis_engine.py", model: "Graham-Dodd framework", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 1) },
      { name: "FedLiquidityPlumbing", file: "fed_liquidity_plumbing.py", model: "FRED data, rule-based", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 2) },
      { name: "MetadronCube", file: "metadron_cube.py", model: "Multi-factor synthesis, scenario engine", type: "Ensemble", status: "ACTIVE", usage: usageForType("Ensemble", 1) },
      { name: "AgentSimEngine", file: "agent_sim_engine.py", model: "Monte Carlo N-path simulation", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 4) },
    ],
  },
  {
    id: "l3",
    label: "L3 — Risk Management",
    color: "text-[#f85149]",
    borderColor: "border-[#f85149]/30",
    engines: [
      { name: "MonteCarloRiskEngine", file: "monte_carlo_risk.py", model: "MiroFish Monte Carlo (VaR, CVaR, stress VaR)", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 0) },
      { name: "BetaCorridor", file: "beta_corridor.py", model: "Dynamic beta hedging, corridor optimization", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 1) },
    ],
  },
  {
    id: "l4",
    label: "L4 — Portfolio Construction",
    color: "text-[#bc8cff]",
    borderColor: "border-[#bc8cff]/30",
    engines: [
      { name: "AlphaOptimizer", file: "alpha_optimizer.py", model: "XGBoost/Ridge fallback, sklearn StandardScaler, LinearRegression", type: "ML", status: "ACTIVE", usage: usageForType("ML", 2) },
      { name: "DecisionMatrix", file: "decision_matrix.py", model: "Multi-criteria weighted scoring", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 3) },
      { name: "ConvictionOverride", file: "conviction_override.py", model: "Signal strength override logic", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 4) },
    ],
  },
  {
    id: "l5",
    label: "L5 — ML Pipeline",
    color: "text-[#00d4aa]",
    borderColor: "border-[#00d4aa]/30",
    engines: [
      { name: "DeepLearningEngine", file: "deep_learning_engine.py", model: "Pure-NumPy PPO Agent (RL), no torch/tf", type: "Neural Net", status: "ACTIVE", usage: usageForType("Neural Net", 0) },
      { name: "UniverseClassifier", file: "universe_classifier.py", model: "XGBoost 4-model soft-voting: GaussianNB + GBM + RandomForest + XGBoost (sklearn)", type: "Ensemble", status: "ACTIVE", usage: usageForType("Ensemble", 2) },
      { name: "Backtester", file: "backtester.py", model: "Walk-forward, Monte Carlo, scenario engine", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 2) },
      { name: "PatternRecognition", file: "pattern_recognition.py", model: "Candlestick/chart pattern detection", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 0) },
      { name: "ModelEvaluator", file: "model_evaluator.py", model: "sklearn metrics (P/R/F1), tier-aware", type: "ML", status: "ACTIVE", usage: usageForType("ML", 3) },
      { name: "ModelStore", file: "model_store.py", model: "sklearn joblib + HMAC-SHA256 integrity", type: "Framework", status: "ACTIVE", usage: usageForType("Framework", 1) },
      { name: "SocialFeatures", file: "social_features.py", model: "Feature engineering for social signals", type: "ML", status: "ACTIVE", usage: usageForType("ML", 4) },
    ],
  },
  {
    id: "l5b",
    label: "L5 Bridges — ML Model Adapters",
    color: "text-[#4ecdc4]",
    borderColor: "border-[#4ecdc4]/30",
    engines: [
      { name: "MonteCarloBridge", file: "monte_carlo_bridge.py", model: "ARIMA(1,1,1) + 1000-path Monte Carlo", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 0) },
      { name: "NvidiaTFTAdapter", file: "nvidia_tft_adapter.py", model: "Temporal Fusion Transformer (multi-horizon)", type: "Neural Net", status: "ACTIVE", usage: usageForType("Neural Net", 1) },
      { name: "StockPredictionBridge", file: "stock_prediction_bridge.py", model: "Evolution Strategy neural net (2-layer, 20→10→1, tanh)", type: "Neural Net", status: "ACTIVE", usage: usageForType("Neural Net", 2) },
      { name: "FinRLBridge", file: "finrl_bridge.py", model: "FinRL deep RL framework adapter", type: "Neural Net", status: "ACTIVE", usage: usageForType("Neural Net", 3) },
      { name: "MarkovRegimeBridge", file: "markov_regime_bridge.py", model: "Hidden Markov Model (hmmlearn)", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 1) },
      { name: "KServeAdapter", file: "kserve_adapter.py", model: "KServe ML model serving", type: "Framework", status: "ACTIVE", usage: usageForType("Framework", 2) },
      { name: "DeepTradingFeatures", file: "deep_trading_features.py", model: "Feature engineering for deep models", type: "ML", status: "ACTIVE", usage: usageForType("ML", 0) },
      { name: "AINewtonBridge", file: "ai_newton_bridge.py", model: "AI-Newton symbolic physics engine", type: "Neural Net", status: "ACTIVE", usage: usageForType("Neural Net", 4) },
    ],
  },
  {
    id: "l6",
    label: "L6 — Agent Layer",
    color: "text-[#3fb950]",
    borderColor: "border-[#3fb950]/30",
    engines: [
      { name: "InvestorPersonas", file: "investor_personas.py", model: "Anthropic Claude (claude-opus-4-6), LLM-driven personas with rule-based fallback", type: "LLM", status: "ACTIVE", usage: usageForType("LLM", 0) },
      { name: "PaulOrchestrator", file: "paul_orchestrator.py", model: "Agent orchestration, Monte Carlo voter", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 3) },
      { name: "DynamicAgentFactory", file: "dynamic_agent_factory.py", model: "Agent spawning", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 0) },
      { name: "EnforcementEngine", file: "enforcement_engine.py", model: "Compliance rule engine", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 1) },
      { name: "AgentScorecard", file: "agent_scorecard.py", model: "Performance tracking", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 2) },
      { name: "AgentMonitor", file: "agent_monitor.py", model: "Health monitoring", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 3) },
      { name: "GICSSectorAgents", file: "gics_sector_agents.py", model: "Sector-specific agents", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 4) },
      { name: "ResearchBots", file: "research_bots.py", model: "Research automation", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 0) },
      { name: "SectorBots", file: "sector_bots.py", model: "Sector analysis bots", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 1) },
    ],
  },
  {
    id: "l7",
    label: "L7 — Execution",
    color: "text-[#ff7b72]",
    borderColor: "border-[#ff7b72]/30",
    engines: [
      { name: "ExecutionEngine", file: "execution_engine.py", model: "2-layer neural net (NumPy), Monte Carlo voter, TWAP/VWAP", type: "Neural Net", status: "ACTIVE", usage: usageForType("Neural Net", 0) },
      { name: "L7UnifiedExecutionSurface", file: "l7_unified_execution_surface.py", model: "Unified execution routing", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 0) },
      { name: "OptionsEngine", file: "options_engine.py", model: "Black-Scholes-Merton pricing + Greeks", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 0) },
      { name: "AlpacaBroker", file: "alpaca_broker.py", model: "Alpaca API execution", type: "Framework", status: "ACTIVE", usage: usageForType("Framework", 0) },
      { name: "PaperBroker", file: "paper_broker.py", model: "Paper trading simulation", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 1) },
      { name: "ExchangeCoreEngine", file: "exchange_core_engine.py", model: "Order matching", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 2) },
      { name: "QuantStrategyExecutor", file: "quant_strategy_executor.py", model: "Strategy execution", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 3) },
      { name: "WonderTraderEngine", file: "wondertrader_engine.py", model: "WonderTrader integration", type: "Framework", status: "STANDBY", usage: usageForType("Framework", 1) },
      { name: "MissedOpportunities", file: "missed_opportunities.py", model: "Opportunity tracking", type: "Rule-Based", status: "STANDBY", usage: usageForType("Rule-Based", 0) },
      { name: "TradierBroker", file: "tradier_broker.py", model: "Tradier API (legacy)", type: "Framework", status: "OFFLINE", usage: "0 calls" },
    ],
  },
  {
    id: "monitoring",
    label: "Monitoring Layer",
    color: "text-[#e3b341]",
    borderColor: "border-[#e3b341]/30",
    engines: [
      { name: "LearningLoop", file: "learning_loop.py", model: "GSD + Paul plugin, Monte Carlo voter", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 0) },
      { name: "LiveDashboard", file: "live_dashboard.py", model: "Real-time monitoring", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 0) },
      { name: "AnomalyDetector", file: "anomaly_detector.py", model: "Z-score anomaly detection", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 1) },
      { name: "MarketWrap", file: "market_wrap.py", model: "Narrative generation", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 1) },
      { name: "PlatinumReport", file: "platinum_report_v2.py", model: "30-section executive report", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 2) },
      { name: "PortfolioAnalytics", file: "portfolio_analytics.py", model: "Deep analytics", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 3) },
      { name: "SectorTracker", file: "sector_tracker.py", model: "Sector performance", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 4) },
      { name: "L7Dashboard", file: "l7_dashboard.py", model: "Execution dashboard", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 0) },
      { name: "HeatmapEngine", file: "heatmap_engine.py", model: "GICS sector heatmap", type: "Rule-Based", status: "ACTIVE", usage: usageForType("Rule-Based", 1) },
      { name: "MemoryMonitor", file: "memory_monitor.py", model: "Session tracking", type: "Rule-Based", status: "STANDBY", usage: usageForType("Rule-Based", 2) },
    ],
  },
  {
    id: "intel",
    label: "Intelligence Platform",
    color: "text-[#f778ba]",
    borderColor: "border-[#f778ba]/30",
    engines: [
      { name: "Air-LLM", file: "air_llm/", model: "LLM inference engine", type: "LLM", status: "ACTIVE", usage: usageForType("LLM", 1) },
      { name: "CAMEL-AI/OASIS", file: "camel_ai/", model: "Multi-agent simulation", type: "LLM", status: "ACTIVE", usage: usageForType("LLM", 2) },
      { name: "FinBERT", file: "finbert/", model: "Financial sentiment NLP", type: "ML", status: "ACTIVE", usage: usageForType("ML", 0) },
      { name: "QLIB", file: "qlib/", model: "Quantitative investment library", type: "Framework", status: "ACTIVE", usage: usageForType("Framework", 3) },
      { name: "KServe", file: "kserve/", model: "ML model serving (GPU inference)", type: "Framework", status: "ACTIVE", usage: usageForType("Framework", 4) },
      { name: "Ruflo Agents", file: "ruflo_agents/", model: "Swarm agent framework", type: "LLM", status: "ACTIVE", usage: usageForType("LLM", 3) },
      { name: "MiroFish", file: "mirofish/", model: "Agent-based market simulation", type: "Statistical", status: "ACTIVE", usage: usageForType("Statistical", 2) },
      { name: "AI-Newton", file: "intelligence_platform/AI-Newton/", model: "Physics-informed neural network, symbolic regression", type: "Neural Net", status: "ACTIVE", usage: usageForType("Neural Net", 2) },
    ],
  },
];

// ═══════════ STATUS CONFIG ═══════════

const STATUS_CONFIG: Record<EngineStatus, { dot: string; label: string; text: string }> = {
  ACTIVE:  { dot: "bg-[#3fb950]", label: "ACTIVE",  text: "text-[#3fb950]" },
  STANDBY: { dot: "bg-[#d29922]", label: "STANDBY", text: "text-[#d29922]" },
  OFFLINE: { dot: "bg-[#f85149]", label: "OFFLINE", text: "text-[#f85149]" },
};

function StatusBadge({ status }: { status: EngineStatus }) {
  const cfg = STATUS_CONFIG[status];
  return (
    <span className={`inline-flex items-center gap-1 text-[9px] font-mono font-medium ${cfg.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${cfg.dot} ${status === "ACTIVE" ? "animate-pulse" : ""}`} />
      {cfg.label}
    </span>
  );
}

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
        e.type.toLowerCase().includes(q) ||
        e.status.toLowerCase().includes(q)
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
                <th className="text-left px-3 py-1.5 text-[#484f58] uppercase tracking-wider font-medium w-[170px]">Engine</th>
                <th className="text-left px-3 py-1.5 text-[#484f58] uppercase tracking-wider font-medium w-[80px]">Status</th>
                <th className="text-left px-3 py-1.5 text-[#484f58] uppercase tracking-wider font-medium w-[180px]">File</th>
                <th className="text-left px-3 py-1.5 text-[#484f58] uppercase tracking-wider font-medium">Model / Framework</th>
                <th className="text-left px-3 py-1.5 text-[#484f58] uppercase tracking-wider font-medium w-[110px]">Type</th>
                <th className="text-left px-3 py-1.5 text-[#484f58] uppercase tracking-wider font-medium w-[100px]">Usage</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((engine) => (
                <tr
                  key={engine.name}
                  className="border-b border-[#1e2633]/50 hover:bg-[#161b22]/60 transition-colors"
                  data-testid={`engine-row-${engine.name.toLowerCase().replace(/\s+/g, "-")}`}
                >
                  <td className="px-3 py-2">
                    <span className="text-[#e6edf3] font-medium">{engine.name}</span>
                  </td>
                  <td className="px-3 py-2">
                    <StatusBadge status={engine.status} />
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
                  <td className="px-3 py-2">
                    <span className={`tabular-nums ${engine.status === "OFFLINE" ? "text-[#484f58]" : "text-[#8b949e]"}`}>
                      {engine.usage}
                    </span>
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
  const activeCount = allEngines.filter((e) => e.status === "ACTIVE").length;
  const standbyCount = allEngines.filter((e) => e.status === "STANDBY").length;
  const offlineCount = allEngines.filter((e) => e.status === "OFFLINE").length;

  const stats = [
    { label: "Total Engines", value: totalEngines, color: "text-[#e6edf3]" },
    { label: "ML Models", value: mlModels, color: "text-[#58a6ff]" },
    { label: "LLM Models", value: llmModels, color: "text-[#f778ba]" },
    { label: "Frameworks", value: frameworks, color: "text-[#4ecdc4]" },
    { label: "Layers", value: LAYERS.length, color: "text-[#00d4aa]" },
    { label: "Active", value: activeCount, color: "text-[#3fb950]" },
    { label: "Standby", value: standbyCount, color: "text-[#d29922]" },
    { label: "Offline", value: offlineCount, color: "text-[#f85149]" },
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

// ═══════════ STATUS FILTER PILLS ═══════════

const ALL_STATUSES: EngineStatus[] = ["ACTIVE", "STANDBY", "OFFLINE"];

function StatusFilterPills({
  active,
  onToggle,
}: {
  active: EngineStatus | null;
  onToggle: (s: EngineStatus | null) => void;
}) {
  return (
    <div className="flex items-center gap-1.5">
      {ALL_STATUSES.map((s) => {
        const cfg = STATUS_CONFIG[s];
        const isActive = active === s;
        return (
          <button
            key={s}
            onClick={() => onToggle(isActive ? null : s)}
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[9px] font-mono font-medium tracking-wider transition-colors border ${
              isActive
                ? `${cfg.text} border-current bg-current/10`
                : "text-[#484f58] border-[#1e2633] hover:border-[#484f58] hover:text-[#8b949e] bg-[#0d1117]"
            }`}
            data-testid={`filter-status-${s.toLowerCase()}`}
          >
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isActive ? cfg.dot : "bg-[#484f58]"}`} />
            {s}
          </button>
        );
      })}
    </div>
  );
}

// ═══════════ MAIN PAGE ═══════════

export default function MLModelsPage() {
  // ─── Engine API — live engine module status ─────────
  const { data: modelsApi } = useEngineQuery<{ modules: Array<{ layer: string; name: string; status: string; error?: string }>; online: number; total: number }>("/ml/models/status", { refetchInterval: 15000 });

  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<ModelType | null>(null);
  const [statusFilter, setStatusFilter] = useState<EngineStatus | null>(null);

  // Combined query string for text filtering
  const query = useMemo(() => search, [search]);

  // Filter layers based on type + status filters
  const filteredLayers = useMemo(() => {
    return LAYERS.map((layer) => ({
      ...layer,
      engines: layer.engines.filter((e) => {
        if (typeFilter && e.type !== typeFilter) return false;
        if (statusFilter && e.status !== statusFilter) return false;
        return true;
      }),
    })).filter((layer) => layer.engines.length > 0);
  }, [typeFilter, statusFilter]);

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
        <div className="w-px h-4 bg-[#1e2633]" />
        <StatusFilterPills active={statusFilter} onToggle={setStatusFilter} />
        {(search || typeFilter || statusFilter) && (
          <button
            onClick={() => { setSearch(""); setTypeFilter(null); setStatusFilter(null); }}
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
