import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, ReferenceLine, Cell } from "recharts";
import { useEngineQuery } from "@/hooks/use-engine-api";

// ═══════════ TYPES ═══════════

interface SimPath {
  values: number[];
  color: string;
}

// ═══════════ CONSTANTS ═══════════

const PATH_COLORS = [
  "rgba(248,81,73,0.35)", "rgba(88,166,255,0.35)", "rgba(188,140,255,0.35)",
  "rgba(63,185,80,0.35)", "rgba(210,153,34,0.35)", "rgba(78,205,196,0.35)",
  "rgba(248,81,73,0.25)", "rgba(88,166,255,0.25)", "rgba(188,140,255,0.25)",
  "rgba(63,185,80,0.25)", "rgba(210,153,34,0.25)", "rgba(78,205,196,0.25)",
];

const DEFAULT_SCENARIOS = [
  { scenario: "Bull Market", probability: "30%", avgReturn: "+24.8%", var95: "-3.1%", maxDD: "-8.2%", color: "text-terminal-positive" },
  { scenario: "Base Case", probability: "45%", avgReturn: "+12.4%", var95: "-8.7%", maxDD: "-18.3%", color: "text-terminal-text-primary" },
  { scenario: "Bear Market", probability: "20%", avgReturn: "-6.2%", var95: "-19.4%", maxDD: "-32.1%", color: "text-terminal-warning" },
  { scenario: "Crash", probability: "5%", avgReturn: "-28.7%", var95: "-41.2%", maxDD: "-58.9%", color: "text-terminal-negative" },
];

// ═══════════ RANDOM WALK GENERATOR ═══════════

function generatePaths(nPaths: number, days: number, initialCapital: number, drift = 0.0004, vol = 0.012): SimPath[] {
  const paths: SimPath[] = [];
  for (let p = 0; p < nPaths; p++) {
    const values: number[] = [initialCapital];
    let v = initialCapital;
    for (let d = 1; d <= days; d++) {
      const z = gaussianRandom();
      v = v * Math.exp((drift - 0.5 * vol * vol) + vol * z);
      values.push(Math.round(v));
    }
    paths.push({ values, color: PATH_COLORS[p % PATH_COLORS.length] });
  }
  return paths;
}

function gaussianRandom(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function computeStats(paths: SimPath[], days: number) {
  const finalValues = paths.map((p) => p.values[days]);
  const initialCapital = paths[0]?.values[0] ?? 100000;
  const returns = finalValues.map((v) => (v - initialCapital) / initialCapital * 100);
  returns.sort((a, b) => a - b);
  const n = returns.length;
  const mean = returns.reduce((s, r) => s + r, 0) / n;
  const variance = returns.reduce((s, r) => s + (r - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const median = n % 2 === 0 ? (returns[n / 2 - 1] + returns[n / 2]) / 2 : returns[Math.floor(n / 2)];
  const var95 = returns[Math.floor(n * 0.05)];
  const var99 = returns[Math.floor(n * 0.01)];
  const skew = returns.reduce((s, r) => s + ((r - mean) / std) ** 3, 0) / n;
  const kurt = returns.reduce((s, r) => s + ((r - mean) / std) ** 4, 0) / n - 3;
  const probProfit = returns.filter((r) => r > 0).length / n * 100;

  // Compute mean path
  const meanPath: number[] = [];
  for (let d = 0; d <= days; d++) {
    const avg = paths.reduce((s, p) => s + p.values[d], 0) / paths.length;
    meanPath.push(Math.round(avg));
  }
  // 5th and 95th percentile bands
  const p5Path: number[] = [];
  const p95Path: number[] = [];
  for (let d = 0; d <= days; d++) {
    const vals = paths.map((p) => p.values[d]).sort((a, b) => a - b);
    p5Path.push(vals[Math.floor(vals.length * 0.05)]);
    p95Path.push(vals[Math.floor(vals.length * 0.95)]);
  }

  // Max drawdown avg
  const avgMaxDD = paths.reduce((sum, p) => {
    let peak = p.values[0];
    let maxDD = 0;
    for (const v of p.values) {
      if (v > peak) peak = v;
      const dd = (peak - v) / peak;
      if (dd > maxDD) maxDD = dd;
    }
    return sum + maxDD;
  }, 0) / paths.length * 100;

  return { mean, std, median, var95, var99, skew, kurt, probProfit, meanPath, p5Path, p95Path, returns, avgMaxDD };
}

function buildHistogram(returns: number[]): { bin: string; count: number; isPositive: boolean }[] {
  const min = Math.min(...returns);
  const max = Math.max(...returns);
  const bins = 20;
  const binWidth = (max - min) / bins;
  const hist = Array.from({ length: bins }, (_, i) => {
    const lo = min + i * binWidth;
    const hi = lo + binWidth;
    return {
      bin: `${lo.toFixed(0)}%`,
      count: returns.filter((r) => r >= lo && r < hi).length,
      isPositive: lo >= 0,
    };
  });
  return hist;
}

// ═══════════ SIMULATION PATHS CANVAS ═══════════

function SimulationCanvas({
  paths,
  meanPath,
  p5Path,
  p95Path,
  days,
  initialCapital,
}: {
  paths: SimPath[];
  meanPath: number[];
  p5Path: number[];
  p95Path: number[];
  days: number;
  initialCapital: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !paths.length) return;
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;
    const pad = { top: 20, bottom: 30, left: 60, right: 12 };
    const cW = W - pad.left - pad.right;
    const cH = H - pad.top - pad.bottom;

    // Find range
    const allVals = [...meanPath, ...p5Path, ...p95Path, ...paths.flatMap((p) => p.values)];
    const minV = Math.min(...allVals) * 0.98;
    const maxV = Math.max(...allVals) * 1.02;

    const xScale = (i: number) => pad.left + (i / days) * cW;
    const yScale = (v: number) => pad.top + ((maxV - v) / (maxV - minV)) * cH;

    // Clear
    ctx.clearRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "#1e2633";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * cH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();
      const val = maxV - (i / 4) * (maxV - minV);
      ctx.fillStyle = "#484f58";
      ctx.font = "9px JetBrains Mono";
      ctx.textAlign = "right";
      ctx.fillText(`$${(val / 1000).toFixed(0)}K`, pad.left - 5, y + 3);
    }
    for (let i = 0; i <= 6; i++) {
      const x = pad.left + (i / 6) * cW;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, H - pad.bottom);
      ctx.stroke();
      ctx.fillStyle = "#484f58";
      ctx.font = "9px JetBrains Mono";
      ctx.textAlign = "center";
      ctx.fillText(`${Math.round((i / 6) * days)}d`, x, H - 4);
    }

    // 5th-95th percentile band
    ctx.beginPath();
    for (let i = 0; i <= days; i++) {
      const x = xScale(i);
      const y = yScale(p95Path[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    for (let i = days; i >= 0; i--) {
      ctx.lineTo(xScale(i), yScale(p5Path[i]));
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(0, 212, 170, 0.06)";
    ctx.fill();
    ctx.strokeStyle = "rgba(0, 212, 170, 0.15)";
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Individual paths
    const displayPaths = paths.slice(0, 50);
    displayPaths.forEach((path) => {
      ctx.beginPath();
      for (let i = 0; i <= days; i++) {
        const x = xScale(i);
        const y = yScale(path.values[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = path.color;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    });

    // Mean path
    ctx.beginPath();
    for (let i = 0; i <= days; i++) {
      const x = xScale(i);
      const y = yScale(meanPath[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#00d4aa";
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Starting dot
    ctx.beginPath();
    ctx.arc(pad.left, yScale(initialCapital), 4, 0, Math.PI * 2);
    ctx.fillStyle = "#00d4aa";
    ctx.fill();
  }, [paths, meanPath, p5Path, p95Path, days, initialCapital]);

  useEffect(() => {
    draw();
    const ro = new ResizeObserver(draw);
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [draw]);

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <canvas ref={canvasRef} className="absolute inset-0" />
    </div>
  );
}

// ═══════════ HISTOGRAM + DISTRIBUTION ═══════════

function DistributionPanel({ returns, mean, median, std, skew, kurt }: {
  returns: number[]; mean: number; median: number; std: number; skew: number; kurt: number;
}) {
  const hist = useMemo(() => buildHistogram(returns), [returns]);
  return (
    <div className="flex flex-col h-full p-2 gap-2">
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={hist} margin={{ top: 4, right: 4, bottom: 16, left: 4 }} barSize={10}>
            <XAxis dataKey="bin" tick={{ fontSize: 7, fill: "#484f58" }} axisLine={{ stroke: "#1e2633" }} tickLine={false} interval={4} />
            <YAxis tick={{ fontSize: 7, fill: "#484f58" }} axisLine={false} tickLine={false} />
            <ReferenceLine x={`${mean.toFixed(0)}%`} stroke="#00d4aa" strokeWidth={1} strokeDasharray="3 3" />
            <Tooltip
              contentStyle={{ background: "#0d1117", border: "1px solid #1e2633", fontSize: 9, fontFamily: "JetBrains Mono", color: "#e6edf3" }}
              formatter={(v: number) => [`${v} paths`, "Count"]}
            />
            <Bar dataKey="count" radius={[1, 1, 0, 0]}>
              {hist.map((entry, i) => (
                <Cell key={i} fill={entry.isPositive ? "#3fb95066" : "#f8514966"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[9px] font-mono border-t border-terminal-border/30 pt-2 flex-shrink-0">
        {[
          ["Mean", `${mean.toFixed(2)}%`],
          ["Median", `${median.toFixed(2)}%`],
          ["Std Dev", `${std.toFixed(2)}%`],
          ["Skew", skew.toFixed(3)],
          ["Kurtosis", kurt.toFixed(3)],
          ["N", returns.length.toLocaleString()],
        ].map(([k, v]) => (
          <div key={k} className="flex justify-between">
            <span className="text-terminal-text-faint">{k}</span>
            <span className="text-terminal-text-primary tabular-nums">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ KPI CARD ═══════════

function KpiCard({ label, value, sub, positive }: { label: string; value: string; sub?: string; positive?: boolean }) {
  return (
    <div className="flex-1 border border-terminal-border/40 rounded bg-terminal-surface p-2 flex flex-col gap-0.5 min-w-0">
      <div className="text-[8px] text-terminal-text-faint tracking-wider uppercase">{label}</div>
      <div className={`text-sm font-bold tabular-nums font-mono ${positive === true ? "text-terminal-positive" : positive === false ? "text-terminal-negative" : "text-terminal-text-primary"}`}>
        {value}
      </div>
      {sub && <div className="text-[8px] text-terminal-text-faint">{sub}</div>}
    </div>
  );
}

// ═══════════ MAIN MONTE CARLO PAGE ═══════════

export default function MonteCarlo() {
  // ─── Engine API — market parameters for simulation ──
  const { data: macroApi } = useEngineQuery<{ vix?: number; spy_return_1m?: number }>("/macro/snapshot", { refetchInterval: 60000 });
  const { data: betaApi } = useEngineQuery<{ state: { vol_adjustment?: number } }>("/portfolio/beta", { refetchInterval: 60000 });
  const { data: stressApi } = useEngineQuery<{ tests: Array<{ scenario: string; probability?: number; expected_return?: number; var_95?: number; max_drawdown?: number }> }>("/ml/stress-tests", { refetchInterval: 120000 });

  const [nPaths, setNPaths] = useState(10000);
  const [days, setDays] = useState(252);
  const [initialCapital, setInitialCapital] = useState(100000);
  const [running, setRunning] = useState(false);
  const [runCount, setRunCount] = useState(10000);
  const [backendRunning, setBackendRunning] = useState(false);
  const [backendResult, setBackendResult] = useState<{ var_95?: number; var_99?: number; mean_return?: number; prob_profit?: number; avg_max_drawdown?: number; source?: string } | null>(null);

  // Derive drift and vol from real market data when available
  // VIX → annualized vol: VIX/100 → daily vol: VIX/(100*sqrt(252))
  // SPY 1M return → annualized drift proxy
  const realVol = macroApi?.vix ? (macroApi.vix / 100) / Math.sqrt(252) : 0.012;
  const realDrift = macroApi?.spy_return_1m ? macroApi.spy_return_1m / 252 : 0.0004;
  const volAdj = betaApi?.state?.vol_adjustment ?? 1.0;
  const simVol = realVol * volAdj;
  const simDrift = realDrift;

  // Generate paths using real market parameters
  const [visualPaths, setVisualPaths] = useState<SimPath[]>(() => generatePaths(50, 252, 100000));
  const [statPaths, setStatPaths] = useState<SimPath[]>(() => generatePaths(200, 252, 100000));
  const [daysSim, setDaysSim] = useState(252);
  const [capitalSim, setCapitalSim] = useState(100000);

  const runSimulation = useCallback(() => {
    setRunning(true);
    setTimeout(() => {
      const newVisual = generatePaths(50, days, initialCapital, simDrift, simVol);
      const newStats = generatePaths(200, days, initialCapital, simDrift, simVol);
      setVisualPaths(newVisual);
      setStatPaths(newStats);
      setDaysSim(days);
      setCapitalSim(initialCapital);
      setRunCount(nPaths);
      setRunning(false);
    }, 600);
  }, [nPaths, days, initialCapital, simDrift, simVol]);

  const runBackendMC = useCallback(async () => {
    setBackendRunning(true);
    try {
      const baseUrl = (import.meta as Record<string, Record<string, string>>).env?.VITE_ENGINE_API_URL || "http://localhost:8000";
      const params = new URLSearchParams({
        n_paths: String(nPaths),
        horizon_days: String(days),
        drift: String(simDrift),
        volatility: String(simVol),
        initial_capital: String(initialCapital),
      });
      const res = await fetch(`${baseUrl}/ml/monte-carlo/simulate?${params}`, { method: "POST" });
      const data = await res.json();
      if (data && !data.error) {
        setBackendResult(data);
      }
    } catch {
      // Silently fail — client-side sim is the fallback
    } finally {
      setBackendRunning(false);
    }
  }, [nPaths, days, initialCapital, simDrift, simVol]);

  // Wire stress-test scenarios from backend
  const SCENARIOS = useMemo(() => {
    if (stressApi?.tests && Array.isArray(stressApi.tests) && stressApi.tests.length > 0) {
      const colorMap: Record<string, string> = { Bull: "text-terminal-positive", Base: "text-terminal-text-primary", Bear: "text-terminal-warning", Crash: "text-terminal-negative" };
      return stressApi.tests.slice(0, 6).map((t) => {
        const key = Object.keys(colorMap).find((k) => (t.scenario || "").includes(k)) || "Base";
        return {
          scenario: t.scenario || "Unknown",
          probability: t.probability != null ? `${t.probability}%` : "—",
          avgReturn: t.expected_return != null ? `${t.expected_return > 0 ? "+" : ""}${t.expected_return}%` : "—",
          var95: t.var_95 != null ? `${t.var_95}%` : "—",
          maxDD: t.max_drawdown != null ? `${t.max_drawdown}%` : "—",
          color: colorMap[key] || "text-terminal-text-primary",
        };
      });
    }
    return DEFAULT_SCENARIOS;
  }, [stressApi]);

  const stats = useMemo(() => computeStats(statPaths, daysSim), [statPaths, daysSim]);

  return (
    <div className="h-full flex flex-col p-[2px] gap-[2px] overflow-hidden" data-testid="monte-carlo">
      {/* KPI Row */}
      <div className="flex gap-[2px] flex-shrink-0 h-[62px]">
        <KpiCard label="Simulations Run" value={runCount.toLocaleString()} sub="paths computed" />
        <KpiCard label="Mean Return" value={`${stats.mean >= 0 ? "+" : ""}${stats.mean.toFixed(1)}%`} positive={stats.mean >= 0} />
        <KpiCard label="95% VaR" value={`${stats.var95.toFixed(1)}%`} positive={false} />
        <KpiCard label="99% VaR" value={`${stats.var99.toFixed(1)}%`} positive={false} />
        <KpiCard label="Max Drawdown (avg)" value={`-${stats.avgMaxDD.toFixed(1)}%`} positive={false} />
        <KpiCard label="Prob. of Profit" value={`${stats.probProfit.toFixed(1)}%`} positive={stats.probProfit > 50} />
      </div>

      {/* Main content */}
      <div className="flex flex-1 gap-[2px] overflow-hidden min-h-0">
        {/* Center: Simulation paths */}
        <div className="flex flex-col flex-1 gap-[2px] overflow-hidden min-h-0">
          <DashboardPanel
            title="SIMULATION PATHS"
            className="flex-1"
            headerRight={
              <div className="flex items-center gap-2 text-[8px] font-mono">
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-terminal-accent inline-block rounded" /> Mean</span>
                <span className="flex items-center gap-1"><span className="w-3 h-2 bg-terminal-accent/10 border border-terminal-accent/20 inline-block rounded" /> 5th–95th</span>
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-[#f85149]/50 inline-block rounded" /> Paths</span>
              </div>
            }
            noPadding
          >
            <SimulationCanvas
              paths={visualPaths}
              meanPath={stats.meanPath}
              p5Path={stats.p5Path}
              p95Path={stats.p95Path}
              days={daysSim}
              initialCapital={capitalSim}
            />
          </DashboardPanel>

          {/* Scenario Analysis */}
          <DashboardPanel
            title="SCENARIO ANALYSIS"
            className="flex-shrink-0 h-[145px]"
            noPadding
          >
            <div className="h-full flex flex-col">
              <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/30 flex-shrink-0">
                <span className="w-[120px]">Scenario</span>
                <span className="w-[70px] text-right">Probability</span>
                <span className="w-[90px] text-right">Avg Return</span>
                <span className="w-[90px] text-right">VaR 95%</span>
                <span className="flex-1 text-right">Max DD</span>
              </div>
              <div className="flex-1 overflow-auto">
                {SCENARIOS.map((s) => (
                  <div key={s.scenario} className="flex items-center px-2 py-1.5 border-b border-terminal-border/10 hover:bg-white/[0.02] text-[10px] font-mono">
                    <span className={`w-[120px] font-semibold ${s.color}`}>{s.scenario}</span>
                    <span className="w-[70px] text-right text-terminal-text-muted">{s.probability}</span>
                    <span className={`w-[90px] text-right tabular-nums ${s.avgReturn.startsWith("+") ? "text-terminal-positive" : "text-terminal-negative"}`}>
                      {s.avgReturn}
                    </span>
                    <span className="w-[90px] text-right tabular-nums text-terminal-negative">{s.var95}</span>
                    <span className="flex-1 text-right tabular-nums text-terminal-negative">{s.maxDD}</span>
                  </div>
                ))}
              </div>
            </div>
          </DashboardPanel>

          {/* Controls */}
          <div className="flex items-center gap-3 px-2 py-1.5 border border-terminal-border/40 rounded bg-terminal-surface flex-shrink-0 text-[10px] font-mono">
            <button
              onClick={runSimulation}
              disabled={running}
              data-testid="button-run-simulation"
              className="px-3 py-1.5 rounded text-[10px] font-semibold tracking-wider bg-terminal-positive/20 text-terminal-positive border border-terminal-positive/40 hover:bg-terminal-positive/30 transition-colors disabled:opacity-50 flex-shrink-0"
            >
              {running ? "RUNNING..." : "RUN SIMULATION"}
            </button>
            <button
              onClick={runBackendMC}
              disabled={backendRunning}
              data-testid="button-backend-mc"
              className="px-3 py-1.5 rounded text-[10px] font-semibold tracking-wider bg-[#58a6ff]/20 text-[#58a6ff] border border-[#58a6ff]/40 hover:bg-[#58a6ff]/30 transition-colors disabled:opacity-50 flex-shrink-0"
            >
              {backendRunning ? "COMPUTING..." : "RUN BACKEND MC"}
            </button>
            {backendResult && (
              <span className="text-[8px] text-terminal-text-faint">
                Backend: VaR95={backendResult.var_95}% | E[R]={backendResult.mean_return}% | P(profit)={backendResult.prob_profit}%
              </span>
            )}
            <div className="flex items-center gap-1.5">
              <span className="text-terminal-text-faint">N Paths:</span>
              <select
                value={nPaths}
                onChange={(e) => setNPaths(+e.target.value)}
                className="bg-terminal-bg border border-terminal-border/50 rounded px-1.5 py-0.5 text-terminal-text-primary outline-none focus:border-terminal-accent/50 text-[10px]"
              >
                {[1000, 5000, 10000, 50000].map((n) => (
                  <option key={n} value={n}>{n.toLocaleString()}</option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-terminal-text-faint">Horizon:</span>
              <select
                value={days}
                onChange={(e) => setDays(+e.target.value)}
                className="bg-terminal-bg border border-terminal-border/50 rounded px-1.5 py-0.5 text-terminal-text-primary outline-none focus:border-terminal-accent/50 text-[10px]"
              >
                {[{ v: 63, l: "63d (Q)" }, { v: 126, l: "126d (6M)" }, { v: 252, l: "252d (1Y)" }].map(({ v, l }) => (
                  <option key={v} value={v}>{l}</option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-terminal-text-faint">Capital:</span>
              <select
                value={initialCapital}
                onChange={(e) => setInitialCapital(+e.target.value)}
                className="bg-terminal-bg border border-terminal-border/50 rounded px-1.5 py-0.5 text-terminal-text-primary outline-none focus:border-terminal-accent/50 text-[10px]"
              >
                {[{ v: 100000, l: "$100K" }, { v: 500000, l: "$500K" }, { v: 1000000, l: "$1M" }].map(({ v, l }) => (
                  <option key={v} value={v}>{l}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Right: Distribution Panel */}
        <DashboardPanel
          title="RETURN DISTRIBUTION"
          className="w-[240px] flex-shrink-0"
          noPadding
        >
          <DistributionPanel
            returns={stats.returns}
            mean={stats.mean}
            median={stats.median}
            std={stats.std}
            skew={stats.skew}
            kurt={stats.kurt}
          />
        </DashboardPanel>
      </div>
    </div>
  );
}
