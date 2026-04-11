import { DashboardPanel } from "@/components/dashboard-panel";
import { useState, useEffect, useRef } from "react";
import { useEngineQuery } from "@/hooks/use-engine-api";

const ENGINES = [
  { id: "L1", name: "Data Ingestion", status: "online", latency: 1.2, cpu: 34, memory: 42, errors: 0 },
  { id: "L2", name: "Signal Gen", status: "online", latency: 3.4, cpu: 67, memory: 58, errors: 0 },
  { id: "L3", name: "Risk Engine", status: "online", latency: 2.1, cpu: 45, memory: 51, errors: 1 },
  { id: "L4", name: "Execution", status: "online", latency: 0.8, cpu: 28, memory: 35, errors: 0 },
  { id: "L5", name: "ML Pipeline", status: "degraded", latency: 12.4, cpu: 89, memory: 82, errors: 3 },
  { id: "L6", name: "Backtest", status: "online", latency: 45.2, cpu: 72, memory: 68, errors: 0 },
  { id: "L7", name: "Reporting", status: "online", latency: 8.6, cpu: 22, memory: 31, errors: 0 },
];

const VPS_METRICS = [
  { name: "US-EAST-1", cpu: 45, mem: 62, disk: 34, net: "2.4 Gbps", status: "healthy" },
  { name: "US-WEST-2", cpu: 38, mem: 55, disk: 28, net: "1.8 Gbps", status: "healthy" },
  { name: "EU-WEST-1", cpu: 72, mem: 78, disk: 45, net: "3.1 Gbps", status: "warning" },
  { name: "AP-EAST-1", cpu: 31, mem: 48, disk: 22, net: "1.2 Gbps", status: "healthy" },
];

const LOG_MESSAGES = [
  "[14:32:18.442] INFO  signal_gen: Generated 142 signals for AAPL basket",
  "[14:32:17.891] INFO  execution: Order 0x8f2a filled MSFT 200@420.12",
  "[14:32:17.234] WARN  ml_pipe: Model inference latency exceeded threshold (12.4ms > 10ms)",
  "[14:32:16.987] INFO  risk_eng: VaR recalculated: 1.22% (within limits)",
  "[14:32:16.512] INFO  data_ing: Tick data received: 1,247 symbols",
  "[14:32:15.890] ERROR ml_pipe: GPU memory allocation failed, retrying...",
  "[14:32:15.445] INFO  execution: Order 0x8f2b submitted NVDA BUY 100",
  "[14:32:14.901] INFO  backtest: Completed 252-day backtest in 45.2ms",
  "[14:32:14.234] WARN  risk_eng: Correlation matrix update detected regime change",
  "[14:32:13.567] INFO  reporting: Daily P&L snapshot exported",
  "[14:32:12.890] INFO  signal_gen: ML factor loading updated (α=0.034)",
  "[14:32:12.123] ERROR ml_pipe: TensorRT optimization failed for model v2.4.1",
  "[14:32:11.456] INFO  data_ing: WebSocket reconnected to exchange feed",
  "[14:32:10.789] INFO  execution: Slippage report: avg 0.02bps (target <0.05bps)",
];

/* ─── API Endpoint definitions for the 3 authorized APIs ──────────── */

interface APIEndpoint {
  endpoint: string;
  purpose: string;
  usedIn: string;
}

interface APISection {
  name: string;
  provider: string;
  auth: string;
  status: "configured" | "stub";
  endpoints: APIEndpoint[];
}

const API_SECTIONS: APISection[] = [
  {
    name: "OpenBB / FMP",
    provider: "Market Data",
    auth: "FMP_API_KEY (vaulted)",
    status: "configured",
    endpoints: [
      { endpoint: "obb.equity.price.historical()", purpose: "Historical OHLCV prices", usedIn: "get_prices()" },
      { endpoint: "obb.equity.fundamental.metrics()", purpose: "Company fundamentals (PE, market cap, etc.)", usedIn: "get_fundamentals()" },
      { endpoint: "obb.economy.fred_series()", purpose: "FRED economic data (GDP, CPI, M2)", usedIn: "get_fred_series()" },
      { endpoint: "obb.fixedincome.rate.sofr()", purpose: "SOFR rate", usedIn: "get_sofr_rate()" },
      { endpoint: "obb.fixedincome.rate.effr()", purpose: "Effective Fed Funds Rate", usedIn: "get_effr_rate()" },
      { endpoint: "obb.economy.cpi()", purpose: "Consumer Price Index", usedIn: "get_cpi()" },
      { endpoint: "obb.economy.unemployment()", purpose: "Unemployment rate", usedIn: "get_unemployment()" },
      { endpoint: "obb.equity.fundamental.filings()", purpose: "SEC filings (10-K, 10-Q, 8-K)", usedIn: "get_company_filings()" },
      { endpoint: "obb.equity.ownership.insider_trading()", purpose: "Insider trading data", usedIn: "get_insider_trading()" },
      { endpoint: "obb.news.company()", purpose: "Company news", usedIn: "get_company_news()" },
      { endpoint: "obb.news.world()", purpose: "World news", usedIn: "get_world_news()" },
      { endpoint: "obb.derivatives.options.chains()", purpose: "Options chain data", usedIn: "get_options_chains()" },
      { endpoint: "obb.etf.holdings()", purpose: "ETF holdings", usedIn: "get_etf_holdings()" },
      { endpoint: "obb.economy.calendar()", purpose: "Economic calendar", usedIn: "get_economic_calendar()" },
    ],
  },
  {
    name: "Alpaca",
    provider: "Trade Execution",
    auth: "ALPACA_API_KEY + ALPACA_SECRET_KEY (vaulted)",
    status: "configured",
    endpoints: [
      { endpoint: "/v2/account", purpose: "Account info (equity, cash, buying power)", usedIn: "get_account() / _sync_account()" },
      { endpoint: "/v2/positions", purpose: "All open positions", usedIn: "_sync_positions()" },
      { endpoint: "/v2/orders", purpose: "Submit / list / cancel orders", usedIn: "place_order() / cancel_order()" },
      { endpoint: "/v2/orders (bracket)", purpose: "Bracket orders (stop-loss + take-profit)", usedIn: "place_order(stop_loss=, take_profit=)" },
      { endpoint: "/v2/positions/{ticker}", purpose: "Close specific position", usedIn: "close_position()" },
      { endpoint: "/v2/positions (DELETE)", purpose: "Close all positions", usedIn: "close_all_positions()" },
      { endpoint: "/v2/account/portfolio/history", purpose: "Portfolio history for P&L", usedIn: "get_portfolio_history()" },
    ],
  },
  {
    name: "Brain Power — Xiaomi Mimo V2 Pro",
    provider: "LLM / Intelligence",
    auth: "XIAOMI_MIMO_API_KEY (vaulted — awaiting configuration)",
    status: "stub",
    endpoints: [
      { endpoint: "/v1/chat/completions (placeholder)", purpose: "LLM inference — chat, analysis, reasoning", usedIn: "nanoclaw_agent.py, llm_inference_bridge.py" },
      { endpoint: "analyze()", purpose: "High-level NanoClaw analysis", usedIn: "nanoclaw_agent.py" },
      { endpoint: "engine_action()", purpose: "Engine API actions via Brain Power reasoning", usedIn: "llm_inference_bridge.py" },
    ],
  },
];

/* ─── API Endpoints Sub-Tab Component ─────────────────────────────── */

function APIEndpointsTab() {
  return (
    <div className="space-y-4 p-2 overflow-auto h-full">
      <div className="text-[10px] text-terminal-text-faint uppercase tracking-wider mb-3">
        3 Authorized External APIs — All keys managed via engine/api/vault.py
      </div>

      {API_SECTIONS.map((section) => (
        <div key={section.name} className="border border-terminal-border rounded p-3">
          {/* Section header */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  section.status === "configured"
                    ? "bg-terminal-positive animate-pulse-live"
                    : "bg-terminal-warning animate-pulse-live"
                }`}
              />
              <span className="text-[11px] text-terminal-accent font-mono font-bold">
                {section.name}
              </span>
              <span className="text-[9px] text-terminal-text-muted">— {section.provider}</span>
            </div>
            <span
              className={`text-[8px] font-mono px-1.5 py-0.5 rounded ${
                section.status === "configured"
                  ? "bg-terminal-positive/20 text-terminal-positive"
                  : "bg-terminal-warning/20 text-terminal-warning"
              }`}
            >
              {section.status === "configured" ? "LIVE" : "STUB MODE"}
            </span>
          </div>

          {/* Auth info */}
          <div className="text-[8px] text-terminal-text-faint mb-2">
            Auth: {section.auth}
          </div>

          {/* Endpoints table */}
          <table className="w-full text-[9px] font-mono">
            <thead>
              <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                <th className="text-left px-2 py-1 font-medium">Endpoint</th>
                <th className="text-left px-2 py-1 font-medium">Purpose</th>
                <th className="text-left px-2 py-1 font-medium">Used In</th>
              </tr>
            </thead>
            <tbody>
              {section.endpoints.map((ep, i) => (
                <tr key={i} className="border-b border-terminal-border/20 hover:bg-white/[0.02]">
                  <td className="px-2 py-1 text-terminal-accent">{ep.endpoint}</td>
                  <td className="px-2 py-1 text-terminal-text-muted">{ep.purpose}</td>
                  <td className="px-2 py-1 text-terminal-text-muted">{ep.usedIn}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}


/* ─── Model Ensemble Status Types ─────────────────────────────────── */

interface ModelStatus {
  name: string;
  key: string;
  port: string;
  status: "online" | "offline" | "stub" | "loading";
  status_detail: string;
  last_latency_ms: number | null;
}

interface ModelsStatusResponse {
  models: ModelStatus[];
  ensemble_active: boolean;
  brain_power_orchestrating: boolean;
  timestamp: string;
}

/* ─── Model Ensemble Status Component ─────────────────────────────── */

function ModelEnsembleStatus({ data, onError }: { data: ModelsStatusResponse | null; onError?: (model: string, msg: string) => void }) {
  if (!data) {
    return (
      <DashboardPanel title="MODEL ENSEMBLE" className="col-span-2">
        <div className="p-3 text-[9px] text-terminal-text-faint font-mono">
          Loading model ensemble status...
        </div>
      </DashboardPanel>
    );
  }

  const statusColor = (status: string) => {
    switch (status) {
      case "online": return "#22c55e";
      case "offline": return "#ef4444";
      case "stub":
      case "loading": return "#eab308";
      default: return "#6b7280";
    }
  };

  const allNonStubOnline = data.models
    .filter((m) => m.status !== "stub")
    .every((m) => m.status === "online");
  const anyOffline = data.models.some((m) => m.status === "offline");

  return (
    <DashboardPanel title="MODEL ENSEMBLE" className="col-span-2">
      <div className="p-2">
        {/* Header row */}
        <div className="flex items-center justify-between mb-3">
          <div>
            <span className="text-[10px] text-terminal-accent font-mono font-bold">Model Ensemble</span>
            <span className="text-[8px] text-terminal-text-faint ml-2">Parallel inference — Brain Power orchestration</span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`text-[8px] font-mono px-1.5 py-0.5 rounded ${
                allNonStubOnline && !anyOffline
                  ? "bg-[#22c55e]/20 text-[#22c55e]"
                  : "bg-[#ef4444]/20 text-[#ef4444]"
              }`}
            >
              {allNonStubOnline && !anyOffline ? "ENSEMBLE ACTIVE" : "ENSEMBLE DEGRADED"}
            </span>
            {data.brain_power_orchestrating && (
              <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-terminal-accent/20 text-terminal-accent">
                BP ORCHESTRATING
              </span>
            )}
          </div>
        </div>

        {/* Model rows */}
        <div className="space-y-1">
          {data.models.map((model) => (
            <div
              key={model.key}
              className="flex items-center gap-2 px-2 py-1.5 rounded border border-terminal-border/30 hover:bg-white/[0.02]"
            >
              {/* Status dot with pulse animation for online */}
              <span className="relative flex h-2.5 w-2.5">
                {model.status === "online" && (
                  <span
                    className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-40"
                    style={{ backgroundColor: statusColor(model.status) }}
                  />
                )}
                <span
                  className="relative inline-flex rounded-full h-2.5 w-2.5"
                  style={{ backgroundColor: statusColor(model.status) }}
                />
              </span>

              {/* Model name */}
              <span className="text-[10px] text-terminal-text-primary font-mono font-bold flex-shrink-0 min-w-[200px]">
                {model.name}
              </span>

              {/* Port badge */}
              <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-terminal-surface-2 text-terminal-text-faint flex-shrink-0">
                {model.port}
              </span>

              {/* Orchestrator badge for Brain Power */}
              {model.key === "brain_power" && (
                <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-terminal-accent/15 text-terminal-accent flex-shrink-0">
                  Orchestrator
                </span>
              )}

              {/* Status text */}
              <span
                className="text-[9px] font-mono flex-1"
                style={{ color: statusColor(model.status) }}
              >
                {model.status === "online" ? "Online" : model.status === "stub" ? `Stub — ${model.status_detail}` : model.status === "offline" ? "Offline" : "Loading"}
              </span>

              {/* Latency */}
              <span className="text-[8px] font-mono text-terminal-text-faint text-right min-w-[50px]">
                {model.last_latency_ms != null ? `${model.last_latency_ms}ms` : "—"}
              </span>
            </div>
          ))}
        </div>
      </div>
    </DashboardPanel>
  );
}

/* ─── Main Dashboard ──────────────────────────────────────────────── */

type TechTab = "system" | "api-endpoints";

export default function TechDashboard() {
  const [activeTab, setActiveTab] = useState<TechTab>("system");

  // ─── Engine API — live system health ────────────────
  const { data: healthApi } = useEngineQuery<{ engines: Array<{ id: string; name: string; level: string; status: string; latency: number; errors: number; error_msg?: string }> }>("/monitoring/engines/health", { refetchInterval: 5000 });
  const { data: vpsApi } = useEngineQuery<{ metrics: Array<{ name: string; cpu: number; memory: number; disk: number; network: string }> }>("/monitoring/vps-metrics", { refetchInterval: 10000 });
  const { data: logsApi } = useEngineQuery<{ messages: Array<{ time: string; level: string; message: string; source: string }> }>("/monitoring/logs", { refetchInterval: 5000 });
  const { data: errorsApi } = useEngineQuery<{ errors: Array<{ timestamp: string; engine: string; severity: string; message: string; file?: string; line?: number }>; counts: { total: number; by_engine: Record<string, number> } }>("/monitoring/errors", { refetchInterval: 5000 });

  // ─── Model Ensemble Status ─────────────────
  const [ensembleData, setEnsembleData] = useState<ModelsStatusResponse | null>(null);
  useEffect(() => {
    const fetchModels = () => {
      fetch("/api/models/status")
        .then((r) => r.json())
        .then((d) => setEnsembleData(d))
        .catch(() => {});
    };
    fetchModels();
    const iv = setInterval(fetchModels, 30000);
    return () => clearInterval(iv);
  }, []);



  // Use API engines when available
  const [engines, setEngines] = useState(ENGINES);
  useEffect(() => {
    if (healthApi?.engines?.length) {
      setEngines(healthApi.engines.map((e) => ({
        id: e.id || e.level,
        name: e.name,
        status: e.status as "online" | "degraded" | "offline",
        latency: e.latency,
        cpu: 0,
        memory: 0,
        errors: e.errors,
      })));
    }
  }, [healthApi]);

  // Use API VPS metrics
  const vpsMetrics = vpsApi?.metrics?.length
    ? vpsApi.metrics.map((m) => ({ name: m.name, cpu: m.cpu, mem: m.memory, disk: m.disk, net: m.network || "—", status: m.cpu > 80 ? "warning" : "healthy" }))
    : VPS_METRICS;

  // Use API logs
  const [logs, setLogs] = useState(LOG_MESSAGES);
  useEffect(() => {
    if (logsApi?.messages?.length) {
      setLogs(logsApi.messages.map((m) => m.message));
    }
  }, [logsApi]);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const iv = setInterval(() => {
      setEngines((prev) =>
        prev.map((e) => ({
          ...e,
          latency: +(e.latency + (Math.random() - 0.5) * 0.5).toFixed(1),
          cpu: Math.min(100, Math.max(10, e.cpu + Math.floor(Math.random() * 6 - 3))),
          memory: Math.min(100, Math.max(20, e.memory + Math.floor(Math.random() * 4 - 2))),
        }))
      );
    }, 3000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    const iv = setInterval(() => {
      const levels = ["INFO", "INFO", "INFO", "WARN", "ERROR"];
      const modules = ["signal_gen", "execution", "ml_pipe", "risk_eng", "data_ing"];
      const msgs = [
        "Tick processed in 0.8ms",
        "Order queue depth: 14",
        "Model checkpoint saved",
        "Feed heartbeat OK",
        "Cache hit rate: 94.2%",
      ];
      const lvl = levels[Math.floor(Math.random() * levels.length)];
      const mod = modules[Math.floor(Math.random() * modules.length)];
      const msg = msgs[Math.floor(Math.random() * msgs.length)];
      const now = new Date();
      const ts = `${now.getHours().toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")}:${now.getSeconds().toString().padStart(2, "0")}.${now.getMilliseconds().toString().padStart(3, "0")}`;
      setLogs((prev) => [...prev.slice(-30), `[${ts}] ${lvl.padEnd(5)} ${mod}: ${msg}`]);
    }, 2000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="h-full flex flex-col overflow-hidden" data-testid="tech-dashboard">
      {/* Tab Bar */}
      <div className="flex-shrink-0 flex gap-0 border-b border-terminal-border bg-terminal-surface-1">
        {([
          { key: "system" as TechTab, label: "System Status" },
          { key: "api-endpoints" as TechTab, label: "API Endpoints" },
        ]).map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2 text-[10px] font-mono uppercase tracking-wider border-b-2 transition-colors ${
              activeTab === tab.key
                ? "border-terminal-accent text-terminal-accent bg-terminal-surface-2"
                : "border-transparent text-terminal-text-muted hover:text-terminal-text-primary hover:bg-terminal-surface-2/50"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "api-endpoints" ? (
        <div className="flex-1 overflow-auto">
          <APIEndpointsTab />
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-[1fr_1fr] grid-rows-[auto_1fr_1fr] gap-[2px] p-[2px] overflow-auto">
          {/* Model Ensemble Status — top of system tab */}
          <ModelEnsembleStatus data={ensembleData} />

          {/* Engine Status Grid */}
          <DashboardPanel title="ACTIVE ENGINES" className="col-span-2">
            <div className="flex gap-2 overflow-x-auto pb-1">
              {engines.map((e) => (
                <div key={e.id} className="flex-shrink-0 border border-terminal-border rounded p-2.5 min-w-[130px]">
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className={`w-2 h-2 rounded-full ${
                      e.status === "online" ? "bg-terminal-positive animate-pulse-live" :
                      e.status === "degraded" ? "bg-terminal-warning animate-pulse-live" :
                      "bg-terminal-negative"
                    }`} />
                    <span className="text-[10px] text-terminal-accent font-mono font-bold">{e.id}</span>
                  </div>
                  <div className="text-[9px] text-terminal-text-muted mb-2">{e.name}</div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-[8px]">
                      <span className="text-terminal-text-faint">CPU</span>
                      <span className="text-terminal-text-primary font-mono tabular-nums">{e.cpu}%</span>
                    </div>
                    <div className="h-1 bg-terminal-surface-2 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full transition-all ${e.cpu > 80 ? "bg-terminal-negative" : e.cpu > 60 ? "bg-terminal-warning" : "bg-terminal-accent"}`}
                        style={{ width: `${e.cpu}%` }} />
                    </div>
                    <div className="flex justify-between text-[8px]">
                      <span className="text-terminal-text-faint">MEM</span>
                      <span className="text-terminal-text-primary font-mono tabular-nums">{e.memory}%</span>
                    </div>
                    <div className="h-1 bg-terminal-surface-2 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full transition-all ${e.memory > 80 ? "bg-terminal-negative" : e.memory > 60 ? "bg-terminal-warning" : "bg-terminal-blue"}`}
                        style={{ width: `${e.memory}%` }} />
                    </div>
                    <div className="flex justify-between text-[8px] mt-1">
                      <span className="text-terminal-text-faint">Lat</span>
                      <span className="text-terminal-text-primary font-mono tabular-nums">{e.latency}ms</span>
                    </div>
                    {e.errors > 0 && (
                      <div className="text-[8px] text-terminal-negative mt-0.5">⚠ {e.errors} errors</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </DashboardPanel>

          {/* VPS Health */}
          <DashboardPanel title="VPS HEALTH" noPadding>
            <div className="overflow-auto h-full">
              <table className="w-full text-[9px] font-mono">
                <thead>
                  <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                    <th className="text-left px-2 py-1.5 font-medium">Region</th>
                    <th className="text-right px-2 py-1.5 font-medium">CPU</th>
                    <th className="text-right px-2 py-1.5 font-medium">Mem</th>
                    <th className="text-right px-2 py-1.5 font-medium">Disk</th>
                    <th className="text-right px-2 py-1.5 font-medium">Net</th>
                  </tr>
                </thead>
                <tbody>
                  {vpsMetrics.map((v, i) => (
                    <tr key={i} className="border-b border-terminal-border/20">
                      <td className="px-2 py-1.5 flex items-center gap-1.5">
                        <span className={`w-1.5 h-1.5 rounded-full ${v.status === "healthy" ? "bg-terminal-positive" : "bg-terminal-warning"}`} />
                        <span className="text-terminal-text-primary">{v.name}</span>
                      </td>
                      <td className={`px-2 py-1.5 text-right tabular-nums ${v.cpu > 70 ? "text-terminal-warning" : "text-terminal-text-muted"}`}>{v.cpu}%</td>
                      <td className={`px-2 py-1.5 text-right tabular-nums ${v.mem > 70 ? "text-terminal-warning" : "text-terminal-text-muted"}`}>{v.mem}%</td>
                      <td className="px-2 py-1.5 text-right text-terminal-text-muted tabular-nums">{v.disk}%</td>
                      <td className="px-2 py-1.5 text-right text-terminal-text-muted">{v.net}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </DashboardPanel>

          {/* Error Display */}
          <DashboardPanel title="ERRORS" noPadding>
            <div className="overflow-auto h-full">
              {errorsApi?.errors?.length ? (
                <table className="w-full text-[9px] font-mono">
                  <thead>
                    <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                      <th className="text-left px-2 py-1.5 font-medium">Time</th>
                      <th className="text-left px-2 py-1.5 font-medium">Engine</th>
                      <th className="text-left px-2 py-1.5 font-medium">Message</th>
                      <th className="text-left px-2 py-1.5 font-medium">Location</th>
                    </tr>
                  </thead>
                  <tbody>
                    {errorsApi.errors.map((err, i) => (
                      <tr key={i} className="border-b border-terminal-border/20">
                        <td className="px-2 py-1.5 text-terminal-text-faint whitespace-nowrap">{err.timestamp}</td>
                        <td className="px-2 py-1.5 text-terminal-warning">{err.engine}</td>
                        <td className="px-2 py-1.5 text-terminal-negative">{err.message}</td>
                        <td className="px-2 py-1.5 text-terminal-text-faint">
                          {err.file ? `${err.file}${err.line ? `:${err.line}` : ""}` : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="p-3 text-[9px] text-terminal-text-faint font-mono">
                  {errorsApi ? "No errors reported" : "Waiting for error data..."}
                </div>
              )}
              {errorsApi?.counts && (
                <div className="px-2 py-1.5 border-t border-terminal-border/50 text-[8px] text-terminal-text-faint font-mono">
                  Total: {errorsApi.counts.total} | By engine: {Object.entries(errorsApi.counts.by_engine).map(([k, v]) => `${k}:${v}`).join(", ") || "—"}
                </div>
              )}
            </div>
          </DashboardPanel>

          {/* Memory Consumption & Efficiency */}
          <DashboardPanel title="MEMORY CONSUMPTION & EFFICIENCY" className="col-span-2" noPadding>
            <div className="p-2">
              {/* Mini bar chart */}
              <div className="flex items-end gap-1 h-12 mb-2">
                {[
                  { layer: "L1", rss: 120, color: "bg-terminal-accent" },
                  { layer: "L2", rss: 340, color: "bg-terminal-blue" },
                  { layer: "L3", rss: 85, color: "bg-terminal-accent" },
                  { layer: "L4", rss: 210, color: "bg-terminal-blue" },
                  { layer: "L5", rss: 580, color: "bg-terminal-negative" },
                  { layer: "L6", rss: 250, color: "bg-terminal-warning" },
                  { layer: "L7", rss: 180, color: "bg-terminal-blue" },
                ].map((d) => {
                  const maxRss = 580;
                  const pct = (d.rss / maxRss) * 100;
                  return (
                    <div key={d.layer} className="flex flex-col items-center gap-0.5 flex-1">
                      <span className="text-[7px] font-mono text-terminal-text-faint tabular-nums">{d.rss}</span>
                      <div
                        className={`w-full rounded-t ${d.color} opacity-80`}
                        style={{ height: `${(pct / 100) * 36}px` }}
                      />
                      <span className="text-[7px] font-mono text-terminal-text-faint">{d.layer}</span>
                    </div>
                  );
                })}
              </div>
              {/* Table */}
              <table className="w-full text-[9px] font-mono">
                <thead>
                  <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                    <th className="text-left px-2 py-1 font-medium">Layer</th>
                    <th className="text-right px-2 py-1 font-medium">RSS (MB)</th>
                    <th className="text-right px-2 py-1 font-medium">Heap (MB)</th>
                    <th className="text-right px-2 py-1 font-medium">CPU Time (ms)</th>
                    <th className="text-right px-2 py-1 font-medium">GC Pauses</th>
                    <th className="text-right px-2 py-1 font-medium">Efficiency %</th>
                  </tr>
                </thead>
                <tbody>
                  {([
                    { layer: "L1", rss: 120, heap: 84,  cpu: 1.2,  gc: 2,  eff: 94 },
                    { layer: "L2", rss: 340, heap: 210, cpu: 3.4,  gc: 8,  eff: 87 },
                    { layer: "L3", rss: 85,  heap: 61,  cpu: 2.1,  gc: 1,  eff: 96 },
                    { layer: "L4", rss: 210, heap: 148, cpu: 0.8,  gc: 4,  eff: 91 },
                    { layer: "L5", rss: 580, heap: 412, cpu: 12.4, gc: 22, eff: 78 },
                    { layer: "L6", rss: 250, heap: 171, cpu: 8.6,  gc: 6,  eff: 85 },
                    { layer: "L7", rss: 180, heap: 122, cpu: 2.8,  gc: 3,  eff: 93 },
                  ] as const).map((row) => {
                    const effColor = row.eff >= 90 ? "text-terminal-positive" : row.eff >= 80 ? "text-terminal-warning" : "text-terminal-negative";
                    return (
                      <tr key={row.layer} className="border-b border-terminal-border/20 hover:bg-white/[0.02]">
                        <td className="px-2 py-1 text-terminal-accent font-medium">{row.layer}</td>
                        <td className="px-2 py-1 text-right text-terminal-text-muted tabular-nums">{row.rss}</td>
                        <td className="px-2 py-1 text-right text-terminal-text-muted tabular-nums">{row.heap}</td>
                        <td className="px-2 py-1 text-right text-terminal-text-muted tabular-nums">{row.cpu}</td>
                        <td className={`px-2 py-1 text-right tabular-nums ${row.gc > 10 ? "text-terminal-warning" : "text-terminal-text-muted"}`}>{row.gc}</td>
                        <td className={`px-2 py-1 text-right tabular-nums font-medium ${effColor}`}>{row.eff}%</td>
                      </tr>
                    );
                  })}
                  {/* TOTAL row */}
                  <tr className="border-t border-terminal-border/50 bg-white/[0.02]">
                    <td className="px-2 py-1 text-terminal-text-primary font-bold uppercase text-[8px] tracking-wider">Total</td>
                    <td className="px-2 py-1 text-right text-terminal-text-primary font-bold tabular-nums">1765</td>
                    <td className="px-2 py-1 text-right text-terminal-text-primary font-bold tabular-nums">1208</td>
                    <td className="px-2 py-1 text-right text-terminal-text-primary font-bold tabular-nums">31.3</td>
                    <td className="px-2 py-1 text-right text-terminal-text-primary font-bold tabular-nums">46</td>
                    <td className="px-2 py-1 text-right text-terminal-warning font-bold tabular-nums">89%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </DashboardPanel>

          {/* Latency Table */}
          <DashboardPanel title="ENGINE LATENCY" noPadding>
            <div className="overflow-auto h-full">
              <table className="w-full text-[9px] font-mono">
                <thead>
                  <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                    <th className="text-left px-2 py-1.5 font-medium">Engine</th>
                    <th className="text-right px-2 py-1.5 font-medium">p50</th>
                    <th className="text-right px-2 py-1.5 font-medium">p95</th>
                    <th className="text-right px-2 py-1.5 font-medium">p99</th>
                  </tr>
                </thead>
                <tbody>
                  {engines.map((e) => (
                    <tr key={e.id} className="border-b border-terminal-border/20">
                      <td className="px-2 py-1.5 text-terminal-accent">{e.id} {e.name}</td>
                      <td className="px-2 py-1.5 text-right text-terminal-text-muted tabular-nums">{e.latency.toFixed(1)}ms</td>
                      <td className="px-2 py-1.5 text-right text-terminal-text-muted tabular-nums">{(e.latency * 2.1).toFixed(1)}ms</td>
                      <td className="px-2 py-1.5 text-right text-terminal-warning tabular-nums">{(e.latency * 3.5).toFixed(1)}ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </DashboardPanel>

          {/* Live Log */}
          <DashboardPanel title="LIVE LOG" className="col-span-2" noPadding>
            <div ref={logRef} className="overflow-auto h-full p-2 bg-terminal-bg font-mono text-[9px] leading-relaxed">
              {logs.map((line, i) => {
                const isError = line.includes("ERROR");
                const isWarn = line.includes("WARN");
                return (
                  <div key={i} className={`py-0.5 ${isError ? "text-terminal-negative" : isWarn ? "text-terminal-warning" : "text-terminal-text-muted"}`}>
                    {line}
                  </div>
                );
              })}
            </div>
          </DashboardPanel>
        </div>
      )}
    </div>
  );
}
