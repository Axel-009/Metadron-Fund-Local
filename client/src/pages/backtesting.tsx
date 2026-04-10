import { useState } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { useEngineQuery } from "@/hooks/use-engine-api";

// ═══════════ TYPES ═══════════

interface Mispricing {
  ticker: string;
  current_price: number;
  fair_value: number;
  z_score: number;
  deviation_pct: number;
  signal: string;
  confidence: number;
  source: string;
}

interface RVSignal {
  ticker: string;
  sector: string;
  momentum_12_1: number;
  rsi_14: number;
  signal: string;
  relative_strength: number;
  source: string;
}

interface BacktestResult {
  ticker: string;
  strategy: string;
  direction: string;
  sharpe: number;
  total_return: number;
  annual_return: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  vs_benchmark: number;
  test_days: number;
}

interface Opportunity {
  source: string;
  ticker: string;
  direction: string;
  expected_return: number;
  confidence: number;
  detail: string;
}

interface Pattern {
  ticker: string;
  pattern_type: string;
  direction: string;
  confidence: number;
  entry_price?: number;
}

interface CorrelationBreakdown {
  pair: string;
  long_term_corr: number;
  current_corr: number;
  delta: number;
  alert: string;
}

interface BacktestSummary {
  total_opportunities: number;
  high_conviction: number;
  avg_expected_return: number;
  total_mispricings: number;
  total_rv_signals: number;
  total_patterns: number;
  correlation_breakdowns: number;
}

// ═══════════ KPI CARDS ═══════════

function KpiCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="flex-1 border border-terminal-border/40 rounded bg-terminal-surface p-2">
      <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider">{label}</div>
      <div className={`text-sm font-bold font-mono tabular-nums mt-0.5 ${color || "text-terminal-text-primary"}`}>{value}</div>
      {sub && <div className="text-[8px] text-terminal-text-faint mt-0.5">{sub}</div>}
    </div>
  );
}

// ═══════════ MISPRICINGS TABLE ═══════════

function MispricingsTable({ data }: { data: Mispricing[] }) {
  if (!data.length) {
    return <div className="text-[9px] text-terminal-text-faint font-mono p-3">No mispricing signals detected. Run a backtest to generate data.</div>;
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/40 flex-shrink-0">
        <span className="w-[80px]">Ticker</span>
        <span className="w-[80px] text-right">Price</span>
        <span className="w-[80px] text-right">Fair Value</span>
        <span className="w-[70px] text-right">Z-Score</span>
        <span className="w-[70px] text-right">Dev %</span>
        <span className="w-[70px] text-right">Signal</span>
        <span className="flex-1 text-right">Confidence</span>
      </div>
      <div className="flex-1 overflow-auto">
        {data.map((m, i) => (
          <div key={`${m.ticker}-${i}`} className="flex items-center px-2 py-1.5 border-b border-terminal-border/10 hover:bg-white/[0.02] text-[9px] font-mono">
            <span className="w-[80px] text-terminal-accent font-medium">{m.ticker}</span>
            <span className="w-[80px] text-right tabular-nums text-terminal-text-primary">${m.current_price.toFixed(2)}</span>
            <span className="w-[80px] text-right tabular-nums text-terminal-text-muted">${m.fair_value.toFixed(2)}</span>
            <span className={`w-[70px] text-right tabular-nums font-semibold ${m.z_score > 0 ? "text-terminal-negative" : "text-terminal-positive"}`}>
              {m.z_score.toFixed(2)}
            </span>
            <span className={`w-[70px] text-right tabular-nums ${m.deviation_pct > 0 ? "text-terminal-negative" : "text-terminal-positive"}`}>
              {m.deviation_pct > 0 ? "+" : ""}{m.deviation_pct.toFixed(1)}%
            </span>
            <span className={`w-[70px] text-right font-semibold ${m.signal.includes("LONG") ? "text-terminal-positive" : "text-terminal-negative"}`}>
              {m.signal}
            </span>
            <span className="flex-1 text-right">
              <span className={`px-1.5 py-0.5 rounded text-[7px] font-semibold border ${
                m.confidence > 0.7 ? "text-terminal-positive bg-terminal-positive/10 border-terminal-positive/30" :
                m.confidence > 0.4 ? "text-terminal-warning bg-terminal-warning/10 border-terminal-warning/30" :
                "text-terminal-text-muted bg-terminal-bg border-terminal-border/30"
              }`}>
                {(m.confidence * 100).toFixed(0)}%
              </span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ RELATIVE VALUE TABLE ═══════════

function RVTable({ data }: { data: RVSignal[] }) {
  if (!data.length) {
    return <div className="text-[9px] text-terminal-text-faint font-mono p-3">No relative value signals. Run a backtest to generate data.</div>;
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/40 flex-shrink-0">
        <span className="w-[70px]">Ticker</span>
        <span className="w-[120px]">Sector</span>
        <span className="w-[80px] text-right">Mom 12-1</span>
        <span className="w-[60px] text-right">RSI</span>
        <span className="w-[90px] text-right">Signal</span>
        <span className="flex-1 text-right">RS %</span>
      </div>
      <div className="flex-1 overflow-auto">
        {data.map((rv, i) => (
          <div key={`${rv.ticker}-${i}`} className="flex items-center px-2 py-1.5 border-b border-terminal-border/10 hover:bg-white/[0.02] text-[9px] font-mono">
            <span className="w-[70px] text-terminal-accent font-medium">{rv.ticker}</span>
            <span className="w-[120px] text-terminal-text-muted truncate">{rv.sector}</span>
            <span className={`w-[80px] text-right tabular-nums ${rv.momentum_12_1 >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
              {rv.momentum_12_1 > 0 ? "+" : ""}{rv.momentum_12_1.toFixed(1)}%
            </span>
            <span className={`w-[60px] text-right tabular-nums ${
              rv.rsi_14 > 70 ? "text-terminal-negative" : rv.rsi_14 < 30 ? "text-terminal-positive" : "text-terminal-text-primary"
            }`}>
              {rv.rsi_14.toFixed(0)}
            </span>
            <span className={`w-[90px] text-right font-semibold ${
              rv.signal === "OVERSOLD" || rv.signal === "LONG_SPREAD" ? "text-terminal-positive" :
              rv.signal === "OVERBOUGHT" || rv.signal === "SHORT_SPREAD" ? "text-terminal-negative" :
              "text-terminal-text-muted"
            }`}>
              {rv.signal}
            </span>
            <span className="flex-1 text-right tabular-nums text-terminal-text-primary">{rv.relative_strength.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ CORRELATION MATRIX HEATMAP ═══════════

function CorrelationPanel({ correlations, breakdowns }: { correlations: Record<string, number>; breakdowns: CorrelationBreakdown[] }) {
  const entries = Object.entries(correlations);

  if (!entries.length && !breakdowns.length) {
    return <div className="text-[9px] text-terminal-text-faint font-mono p-3">No correlation data. Run a backtest to generate data.</div>;
  }

  const getCorrColor = (v: number) => {
    if (v > 0.8) return "bg-terminal-positive/40 text-terminal-positive";
    if (v > 0.5) return "bg-terminal-positive/20 text-terminal-text-primary";
    if (v > 0.2) return "bg-terminal-accent/10 text-terminal-text-muted";
    if (v > -0.2) return "bg-terminal-bg text-terminal-text-faint";
    if (v > -0.5) return "bg-terminal-warning/15 text-terminal-warning";
    return "bg-terminal-negative/20 text-terminal-negative";
  };

  return (
    <div className="flex flex-col gap-2 p-2">
      {/* Heatmap grid */}
      {entries.length > 0 && (
        <div className="flex flex-wrap gap-0.5">
          {entries.slice(0, 50).map(([pair, val]) => (
            <div
              key={pair}
              className={`px-1.5 py-0.5 rounded text-[7px] font-mono tabular-nums ${getCorrColor(val)}`}
              title={`${pair}: ${val.toFixed(3)}`}
            >
              {pair.split("/").map(t => t.slice(0, 3)).join("/")} {val.toFixed(2)}
            </div>
          ))}
        </div>
      )}

      {/* Breakdown alerts */}
      {breakdowns.length > 0 && (
        <div className="mt-1">
          <div className="text-[8px] text-terminal-negative tracking-wider font-semibold mb-1">CORRELATION BREAKDOWNS</div>
          {breakdowns.map((b, i) => (
            <div key={i} className="flex items-center gap-2 px-2 py-1 text-[8px] font-mono border-l-2 border-terminal-negative/50 bg-terminal-negative/5">
              <span className="text-terminal-accent font-medium">{b.pair}</span>
              <span className="text-terminal-text-faint">Long-term: {b.long_term_corr.toFixed(2)}</span>
              <span className="text-terminal-negative">Current: {b.current_corr.toFixed(2)}</span>
              <span className="text-terminal-warning">Delta: {b.delta.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ═══════════ PATTERN DETECTION TABLE ═══════════

function PatternsTable({ data }: { data: Pattern[] }) {
  if (!data.length) {
    return <div className="text-[9px] text-terminal-text-faint font-mono p-3">No patterns detected. Run a backtest to generate data.</div>;
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/40 flex-shrink-0">
        <span className="w-[70px]">Ticker</span>
        <span className="flex-1">Pattern</span>
        <span className="w-[70px] text-right">Direction</span>
        <span className="w-[80px] text-right">Confidence</span>
      </div>
      <div className="flex-1 overflow-auto">
        {data.map((p, i) => (
          <div key={`${p.ticker}-${i}`} className="flex items-center px-2 py-1.5 border-b border-terminal-border/10 hover:bg-white/[0.02] text-[9px] font-mono">
            <span className="w-[70px] text-terminal-accent font-medium">{p.ticker}</span>
            <span className="flex-1 text-terminal-text-primary">{p.pattern_type}</span>
            <span className={`w-[70px] text-right font-semibold ${
              p.direction === "LONG" ? "text-terminal-positive" :
              p.direction === "SHORT" ? "text-terminal-negative" :
              "text-terminal-text-muted"
            }`}>
              {p.direction}
            </span>
            <span className="w-[80px] text-right tabular-nums text-terminal-text-primary">{(p.confidence * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ BACKTEST RESULTS TABLE ═══════════

function BacktestResultsTable({ data }: { data: BacktestResult[] }) {
  if (!data.length) {
    return <div className="text-[9px] text-terminal-text-faint font-mono p-3">No backtest results. Run a backtest to generate data.</div>;
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/40 flex-shrink-0">
        <span className="w-[70px]">Ticker</span>
        <span className="w-[140px]">Strategy</span>
        <span className="w-[60px] text-right">Sharpe</span>
        <span className="w-[60px] text-right">Win %</span>
        <span className="w-[70px] text-right">Max DD</span>
        <span className="w-[70px] text-right">Return</span>
        <span className="flex-1 text-right">vs BM</span>
      </div>
      <div className="flex-1 overflow-auto">
        {data.map((bt, i) => (
          <div key={`${bt.ticker}-${i}`} className="flex items-center px-2 py-1.5 border-b border-terminal-border/10 hover:bg-white/[0.02] text-[9px] font-mono">
            <span className="w-[70px] text-terminal-accent font-medium">{bt.ticker}</span>
            <span className="w-[140px] text-terminal-text-muted truncate">{bt.strategy}</span>
            <span className={`w-[60px] text-right tabular-nums font-semibold ${bt.sharpe > 1 ? "text-terminal-positive" : bt.sharpe > 0 ? "text-terminal-text-primary" : "text-terminal-negative"}`}>
              {bt.sharpe.toFixed(2)}
            </span>
            <span className="w-[60px] text-right tabular-nums text-terminal-text-primary">{bt.win_rate.toFixed(0)}%</span>
            <span className="w-[70px] text-right tabular-nums text-terminal-negative">{bt.max_drawdown.toFixed(1)}%</span>
            <span className={`w-[70px] text-right tabular-nums ${bt.total_return >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
              {bt.total_return >= 0 ? "+" : ""}{bt.total_return.toFixed(1)}%
            </span>
            <span className={`flex-1 text-right tabular-nums ${bt.vs_benchmark >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
              {bt.vs_benchmark >= 0 ? "+" : ""}{bt.vs_benchmark.toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ OPPORTUNITIES PANEL ═══════════

function OpportunitiesPanel({ data }: { data: Opportunity[] }) {
  if (!data.length) {
    return <div className="text-[9px] text-terminal-text-faint font-mono p-3">No trade opportunities. Run a backtest to generate data.</div>;
  }

  const sourceColors: Record<string, string> = {
    mispricing: "text-blue-400 bg-blue-400/10 border-blue-400/30",
    relative_value: "text-purple-400 bg-purple-400/10 border-purple-400/30",
    pattern: "text-cyan-400 bg-cyan-400/10 border-cyan-400/30",
    backtest: "text-terminal-accent bg-terminal-accent/10 border-terminal-accent/30",
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/40 flex-shrink-0">
        <span className="w-[80px]">Source</span>
        <span className="w-[70px]">Ticker</span>
        <span className="w-[60px] text-right">Dir</span>
        <span className="w-[80px] text-right">Exp Ret</span>
        <span className="w-[70px] text-right">Conf</span>
        <span className="flex-1 text-right">Detail</span>
      </div>
      <div className="flex-1 overflow-auto">
        {data.map((opp, i) => (
          <div key={i} className="flex items-center px-2 py-1.5 border-b border-terminal-border/10 hover:bg-white/[0.02] text-[9px] font-mono">
            <span className="w-[80px]">
              <span className={`px-1.5 py-0.5 rounded text-[7px] font-semibold border ${sourceColors[opp.source] || "text-terminal-text-muted bg-terminal-bg border-terminal-border/30"}`}>
                {opp.source.toUpperCase()}
              </span>
            </span>
            <span className="w-[70px] text-terminal-accent font-medium">{opp.ticker}</span>
            <span className={`w-[60px] text-right font-semibold ${
              opp.direction.includes("LONG") ? "text-terminal-positive" :
              opp.direction.includes("SHORT") ? "text-terminal-negative" :
              "text-terminal-text-muted"
            }`}>
              {opp.direction}
            </span>
            <span className={`w-[80px] text-right tabular-nums ${opp.expected_return > 0 ? "text-terminal-positive" : "text-terminal-text-primary"}`}>
              {opp.expected_return > 0 ? "+" : ""}{opp.expected_return.toFixed(1)}%
            </span>
            <span className="w-[70px] text-right tabular-nums text-terminal-text-primary">{(opp.confidence * 100).toFixed(0)}%</span>
            <span className="flex-1 text-right text-terminal-text-faint truncate">{opp.detail}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ MAIN BACKTESTING PAGE ═══════════

export default function BacktestingPage() {
  const [running, setRunning] = useState(false);
  const [runStatus, setRunStatus] = useState("");
  const [signalFilter, setSignalFilter] = useState<string>("all");

  // ─── API Queries ───────────────────────────────────
  const { data: latestData, refetch: refetchLatest } = useEngineQuery<{
    summary: BacktestSummary;
    regime: string;
    date: string;
    backtests: BacktestResult[];
    generated_at: string;
  }>("/backtest/latest", { refetchInterval: 60000 });

  const { data: mispricingsData, refetch: refetchMispricings } = useEngineQuery<{
    mispricings: Mispricing[];
    total: number;
    date: string;
  }>("/backtest/mispricings", { refetchInterval: 60000 });

  const { data: rvData, refetch: refetchRV } = useEngineQuery<{
    relative_value: RVSignal[];
    total: number;
  }>("/backtest/relative-value", { refetchInterval: 60000 });

  const { data: corrData, refetch: refetchCorr } = useEngineQuery<{
    correlations: {
      current_30d: Record<string, number>;
      current_60d: Record<string, number>;
      current_90d: Record<string, number>;
      breakdowns: CorrelationBreakdown[];
    };
  }>("/backtest/correlations", { refetchInterval: 60000 });

  const { data: patternsData, refetch: refetchPatterns } = useEngineQuery<{
    patterns: Pattern[];
    total: number;
  }>("/backtest/patterns", { refetchInterval: 60000 });

  const { data: oppsData, refetch: refetchOpps } = useEngineQuery<{
    opportunities: Opportunity[];
    total: number;
    summary: BacktestSummary;
    regime: string;
  }>("/backtest/opportunities", { refetchInterval: 60000 });

  const summary = latestData?.summary || oppsData?.summary;
  const regime = latestData?.regime || oppsData?.regime || "UNKNOWN";
  const lastDate = latestData?.date || "";
  const lastRun = latestData?.generated_at || "";
  const mispricings = mispricingsData?.mispricings || [];
  const rvSignals = rvData?.relative_value || [];
  const correlations = corrData?.correlations?.current_30d || {};
  const breakdowns = corrData?.correlations?.breakdowns || [];
  const patterns = patternsData?.patterns || [];
  const backtests = latestData?.backtests || [];
  const opportunities = oppsData?.opportunities || [];

  // Filter opportunities by source
  const filteredOpps = signalFilter === "all"
    ? opportunities
    : opportunities.filter(o => o.source === signalFilter);

  const handleRunBacktest = async () => {
    setRunning(true);
    setRunStatus("Running backtest...");
    try {
      const res = await fetch("/api/engine/backtest/trigger", { method: "POST" });
      const data = await res.json();
      if (data.status === "completed") {
        setRunStatus(`Completed: ${data.summary?.total_opportunities || 0} opportunities found`);
        refetchLatest();
        refetchMispricings();
        refetchRV();
        refetchCorr();
        refetchPatterns();
        refetchOpps();
      } else {
        setRunStatus(`Error: ${data.error || "unknown"}`);
      }
    } catch {
      setRunStatus("Backtest request failed");
    }
    setRunning(false);
  };

  return (
    <div className="h-full flex flex-col gap-[2px] p-[2px] overflow-hidden" data-testid="backtesting">
      {/* Header KPI Cards */}
      <div className="flex gap-[2px] flex-shrink-0 h-[62px]">
        <KpiCard
          label="Total Opportunities"
          value={String(summary?.total_opportunities ?? 0)}
          sub={`${summary?.total_mispricings ?? 0} mispricings, ${summary?.total_rv_signals ?? 0} RV`}
        />
        <KpiCard
          label="High Conviction"
          value={String(summary?.high_conviction ?? 0)}
          color="text-terminal-positive"
          sub="confidence > 60%"
        />
        <KpiCard
          label="Avg Expected Return"
          value={`${(summary?.avg_expected_return ?? 0).toFixed(1)}%`}
          color={(summary?.avg_expected_return ?? 0) >= 0 ? "text-terminal-positive" : "text-terminal-negative"}
        />
        <KpiCard
          label="Last Backtest"
          value={lastRun ? new Date(lastRun).toLocaleTimeString() : "Never"}
          sub={lastDate || "no runs yet"}
        />
        <KpiCard
          label="Regime"
          value={regime}
          color={
            regime === "BULL" ? "text-terminal-positive" :
            regime === "BEAR" ? "text-terminal-negative" :
            "text-terminal-warning"
          }
        />
      </div>

      {/* Main panels: 2x2 grid + bottom row */}
      <div className="flex-1 flex flex-col gap-[2px] overflow-hidden min-h-0">
        {/* Top row: Mispricings + RV */}
        <div className="flex gap-[2px] flex-1 min-h-0">
          <DashboardPanel
            title="MISPRICINGS"
            className="flex-1"
            headerRight={<span className="text-[8px] text-terminal-text-faint font-mono">{mispricings.length} signals</span>}
            noPadding
          >
            <MispricingsTable data={mispricings} />
          </DashboardPanel>

          <DashboardPanel
            title="RELATIVE VALUE"
            className="flex-1"
            headerRight={<span className="text-[8px] text-terminal-text-faint font-mono">{rvSignals.length} signals</span>}
            noPadding
          >
            <RVTable data={rvSignals} />
          </DashboardPanel>
        </div>

        {/* Middle row: Correlations + Patterns */}
        <div className="flex gap-[2px] flex-1 min-h-0">
          <DashboardPanel
            title="CORRELATION MATRIX"
            className="flex-1"
            headerRight={
              <span className="text-[8px] text-terminal-text-faint font-mono">
                {breakdowns.length > 0 ? `${breakdowns.length} breakdowns` : "30d rolling"}
              </span>
            }
            noPadding
          >
            <CorrelationPanel correlations={correlations} breakdowns={breakdowns} />
          </DashboardPanel>

          <DashboardPanel
            title="PATTERN DETECTION"
            className="flex-[0.6]"
            headerRight={<span className="text-[8px] text-terminal-text-faint font-mono">{patterns.length} patterns</span>}
            noPadding
          >
            <PatternsTable data={patterns} />
          </DashboardPanel>
        </div>

        {/* Bottom row: Backtest Results + Opportunities */}
        <div className="flex gap-[2px] flex-1 min-h-0">
          <DashboardPanel
            title="BACKTEST RESULTS"
            className="flex-[0.5]"
            headerRight={<span className="text-[8px] text-terminal-text-faint font-mono">{backtests.length} strategies</span>}
            noPadding
          >
            <BacktestResultsTable data={backtests} />
          </DashboardPanel>

          <DashboardPanel
            title="TRADE OPPORTUNITIES"
            className="flex-1"
            headerRight={
              <div className="flex items-center gap-1">
                {["all", "mispricing", "relative_value", "pattern", "backtest"].map((f) => (
                  <button
                    key={f}
                    onClick={() => setSignalFilter(f)}
                    className={`px-1.5 py-0.5 text-[7px] font-mono rounded ${
                      signalFilter === f
                        ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/40"
                        : "text-terminal-text-faint border border-terminal-border/20 hover:text-terminal-text-muted"
                    }`}
                  >
                    {f === "all" ? "ALL" : f === "relative_value" ? "RV" : f.toUpperCase()}
                  </button>
                ))}
              </div>
            }
            noPadding
          >
            <OpportunitiesPanel data={filteredOpps} />
          </DashboardPanel>
        </div>
      </div>

      {/* Controls Row */}
      <div className="flex items-center gap-4 px-3 py-2 border border-terminal-border/40 rounded bg-terminal-surface flex-shrink-0 text-[10px] font-mono">
        <button
          onClick={handleRunBacktest}
          disabled={running}
          data-testid="button-run-backtest"
          className="px-3 py-1.5 rounded text-[10px] font-semibold tracking-wider bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/40 hover:bg-terminal-accent/25 transition-colors disabled:opacity-50"
        >
          {running ? "RUNNING..." : "RUN BACKTEST"}
        </button>

        {runStatus && (
          <>
            <div className="w-px h-4 bg-terminal-border/50" />
            <span className="text-[8px] text-terminal-accent">{runStatus}</span>
          </>
        )}

        <div className="ml-auto flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="text-terminal-text-faint">Auto-run:</span>
            <span className="text-terminal-accent">Daily 16:15 ET</span>
          </div>
          <div className="w-px h-4 bg-terminal-border/50" />
          <div className="flex items-center gap-1.5">
            <span className="text-terminal-text-faint">Universe:</span>
            <span className="text-terminal-text-primary font-semibold">19 securities + 26 pairs</span>
          </div>
          <div className="w-px h-4 bg-terminal-border/50" />
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-terminal-positive animate-pulse" />
            <span className="text-terminal-positive text-[8px]">ENGINE READY</span>
          </div>
        </div>
      </div>
    </div>
  );
}
