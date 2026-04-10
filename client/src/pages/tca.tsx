import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import { useEngineQuery } from "@/hooks/use-engine-api";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, LineChart, Line, ComposedChart, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";

// ═══════════ TYPES ═══════════

interface TCARecord {
  order_id: string;
  ticker: string;
  side: string;
  quantity: number;
  fill_price: number;
  arrival_price: number;
  vwap_price: number;
  spread_cost_bps: number;
  market_impact_bps: number;
  timing_cost_bps: number;
  commission_bps: number;
  total_cost_bps: number;
  implementation_shortfall_usd: number;
  vwap_slippage_bps: number;
  venue: string;
  algo: string;
  sector: string;
  signal_type: string;
  product_type: string;
  participation_rate: number;
  latency_ms: number;
  timestamp: string;
}

interface TCASummary {
  total_trades: number;
  avg_total_cost_bps: number;
  avg_spread_bps: number;
  avg_impact_bps: number;
  avg_timing_bps: number;
  avg_commission_bps: number;
  avg_vwap_slip_bps: number;
  total_is_usd: number;
  total_volume_usd: number;
  execution_quality_score: number;
  best_execution: { ticker: string; cost_bps: number } | null;
  worst_execution: { ticker: string; cost_bps: number } | null;
  cost_trend: string;
}

interface TrendPoint {
  date: string;
  trades: number;
  volume_usd: number;
  spread_bps: number;
  impact_bps: number;
  timing_bps: number;
  commission_bps: number;
  total_bps: number;
  avg_is_usd: number;
}

interface DecompItem { ticker: string; trades: number; spread: number; impact: number; timing: number; commission: number; total: number; }
interface VenueStat { venue: string; fills: number; avg_cost_bps: number; avg_latency_ms: number; avg_impact_bps: number; avg_spread_bps: number; quality_score: number; }
interface AlgoStat { algo: string; trades: number; avg_cost_bps: number; avg_is_usd: number; avg_vwap_slip_bps: number; avg_participation: number; quality_score: number; }
interface SectorStat { sector: string; trades: number; avg_cost_bps: number; avg_impact_bps: number; total_is_usd: number; total_volume_usd: number; }
interface Outlier { order_id: string; ticker: string; side: string; total_cost_bps: number; z_score: number; reason: string; timestamp: string; }
interface Benchmark { benchmark: string; avg_slippage_bps: number; total_shortfall_usd: number; win_rate: number; trades_evaluated: number; }
interface ISBucket { range: string; count: number; }
interface QualityScore { dimensions: string[]; scores: number[]; }

// ═══════════ HOOKS ═══════════

function useTCA() {
  const { data: tcaData } = useEngineQuery<{ trades: TCARecord[]; summary: TCASummary }>("/execution/tca", { refetchInterval: 10000 });
  const { data: trendData } = useEngineQuery<{ trend: TrendPoint[] }>("/execution/tca/trend", { refetchInterval: 30000 });
  const { data: decompData } = useEngineQuery<{ decomposition: DecompItem[] }>("/execution/tca/decomposition", { refetchInterval: 30000 });
  const { data: venueData } = useEngineQuery<{ venues: VenueStat[] }>("/execution/venue-comparison", { refetchInterval: 30000 });
  const { data: algoData } = useEngineQuery<{ algos: AlgoStat[] }>("/execution/algo-comparison", { refetchInterval: 30000 });
  const { data: sectorData } = useEngineQuery<{ sectors: SectorStat[] }>("/execution/tca/sectors", { refetchInterval: 30000 });
  const { data: outlierData } = useEngineQuery<{ outliers: Outlier[] }>("/execution/tca/outliers", { refetchInterval: 30000 });
  const { data: benchData } = useEngineQuery<{ benchmarks: Benchmark[] }>("/execution/tca/benchmarks", { refetchInterval: 30000 });
  const { data: isDistData } = useEngineQuery<{ distribution: ISBucket[] }>("/execution/tca/is-distribution", { refetchInterval: 30000 });
  const { data: qualityData } = useEngineQuery<QualityScore>("/execution/tca/quality-score", { refetchInterval: 30000 });

  return {
    records: tcaData?.trades ?? [],
    summary: tcaData?.summary ?? null,
    trend: trendData?.trend ?? [],
    decomposition: decompData?.decomposition ?? [],
    venues: venueData?.venues ?? [],
    algos: algoData?.algos ?? [],
    sectors: sectorData?.sectors ?? [],
    outliers: outlierData?.outliers ?? [],
    benchmarks: benchData?.benchmarks ?? [],
    isDistribution: isDistData?.distribution ?? [],
    qualityScore: qualityData ?? null,
  };
}

// ═══════════ COMPONENTS ═══════════

function SummaryCards({ summary }: { summary: TCASummary | null }) {
  if (!summary) {
    return (
      <div className="grid grid-cols-4 gap-1.5">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="bg-terminal-bg rounded border border-terminal-border/50 px-2.5 py-2 animate-pulse">
            <div className="h-2 bg-terminal-border/30 rounded w-20 mb-2" />
            <div className="h-4 bg-terminal-border/30 rounded w-16" />
          </div>
        ))}
      </div>
    );
  }

  const trendIcon = summary.cost_trend === "IMPROVING" ? "↓" : summary.cost_trend === "DEGRADING" ? "↑" : "→";
  const trendColor = summary.cost_trend === "IMPROVING" ? "text-terminal-positive" : summary.cost_trend === "DEGRADING" ? "text-terminal-negative" : "text-terminal-text-muted";

  const cards = [
    { label: "AVG TOTAL COST", value: `${summary.avg_total_cost_bps.toFixed(2)} bps`, color: "text-terminal-accent" },
    { label: "AVG SPREAD", value: `${summary.avg_spread_bps.toFixed(2)} bps`, color: "text-terminal-text-primary" },
    { label: "AVG MKT IMPACT", value: `${summary.avg_impact_bps.toFixed(2)} bps`, color: "text-terminal-warning" },
    { label: "AVG VWAP SLIP", value: `${summary.avg_vwap_slip_bps.toFixed(2)} bps`, color: summary.avg_vwap_slip_bps > 0 ? "text-terminal-negative" : "text-terminal-positive" },
    { label: "TOTAL IMPL. SHORTFALL", value: `$${Math.abs(summary.total_is_usd).toLocaleString(undefined, { maximumFractionDigits: 0 })}`, color: summary.total_is_usd > 0 ? "text-terminal-negative" : "text-terminal-positive" },
    { label: "QUALITY SCORE", value: `${summary.execution_quality_score.toFixed(1)} / 100`, color: summary.execution_quality_score >= 70 ? "text-terminal-positive" : summary.execution_quality_score >= 40 ? "text-terminal-warning" : "text-terminal-negative" },
    { label: "COST TREND", value: `${trendIcon} ${summary.cost_trend}`, color: trendColor },
    { label: "TOTAL TRADES", value: `${summary.total_trades}`, color: "text-terminal-text-primary" },
  ];

  return (
    <div className="grid grid-cols-4 gap-1.5">
      {cards.map(c => (
        <div key={c.label} className="bg-terminal-bg rounded border border-terminal-border/50 px-2.5 py-2">
          <div className="text-[9px] text-terminal-text-faint tracking-wider mb-1">{c.label}</div>
          <div className={`text-sm font-mono font-semibold ${c.color}`}>{c.value}</div>
        </div>
      ))}
    </div>
  );
}

function CostDecomposition({ data }: { data: DecompItem[] }) {
  if (!data.length) return <div className="flex items-center justify-center h-full text-terminal-text-faint text-xs">Awaiting trade data...</div>;
  const display = data.slice(0, 25);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={display} margin={{ top: 5, right: 5, left: -15, bottom: 0 }}>
        <XAxis dataKey="ticker" tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} />
        <Tooltip
          contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 10 }}
          labelStyle={{ color: "#8b949e" }}
        />
        <Bar dataKey="spread" stackId="a" fill="#00d4aa" name="Spread (bps)" />
        <Bar dataKey="impact" stackId="a" fill="#f0883e" name="Impact (bps)" />
        <Bar dataKey="timing" stackId="a" fill="#58a6ff" name="Timing (bps)" />
        <Bar dataKey="commission" stackId="a" fill="#8b949e" name="Commission (bps)" />
      </BarChart>
    </ResponsiveContainer>
  );
}

function CostTrendChart({ data }: { data: TrendPoint[] }) {
  if (!data.length) return <div className="flex items-center justify-center h-full text-terminal-text-faint text-xs">Awaiting trend data...</div>;

  const display = data.map(d => ({
    ...d,
    date: d.date.slice(5), // "MM-DD"
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={display} margin={{ top: 5, right: 5, left: -15, bottom: 0 }}>
        <XAxis dataKey="date" tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} interval={Math.max(0, Math.floor(display.length / 8))} />
        <YAxis tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} />
        <Tooltip
          contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 10 }}
          labelStyle={{ color: "#8b949e" }}
        />
        <Area type="monotone" dataKey="total_bps" fill="#00d4aa" fillOpacity={0.08} stroke="#00d4aa" strokeWidth={1.5} name="Total Cost (bps)" />
        <Line type="monotone" dataKey="spread_bps" stroke="#58a6ff" strokeWidth={1} dot={false} name="Spread" />
        <Line type="monotone" dataKey="impact_bps" stroke="#f0883e" strokeWidth={1} dot={false} name="Impact" />
        <Line type="monotone" dataKey="timing_bps" stroke="#d2a8ff" strokeWidth={1} dot={false} name="Timing" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function VWAPSlippageChart({ records }: { records: TCARecord[] }) {
  if (!records.length) return <div className="flex items-center justify-center h-full text-terminal-text-faint text-xs">Awaiting trade data...</div>;

  // Aggregate by ticker
  const tickerMap = new Map<string, { total: number; count: number }>();
  records.forEach(r => {
    const prev = tickerMap.get(r.ticker) || { total: 0, count: 0 };
    tickerMap.set(r.ticker, { total: prev.total + r.vwap_slippage_bps, count: prev.count + 1 });
  });

  const data = Array.from(tickerMap.entries())
    .map(([ticker, v]) => ({ ticker, slippage: +(v.total / v.count).toFixed(2) }))
    .sort((a, b) => Math.abs(b.slippage) - Math.abs(a.slippage))
    .slice(0, 20);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 5, right: 5, left: -15, bottom: 0 }}>
        <XAxis dataKey="ticker" tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} />
        <Tooltip
          contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 10 }}
          labelStyle={{ color: "#8b949e" }}
        />
        <Bar dataKey="slippage" name="VWAP Slip (bps)">
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.slippage > 0 ? "#f85149" : "#3fb950"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function ISDistribution({ data }: { data: ISBucket[] }) {
  if (!data.length) return <div className="flex items-center justify-center h-full text-terminal-text-faint text-xs">Awaiting distribution data...</div>;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 5, right: 5, left: -15, bottom: 0 }}>
        <XAxis dataKey="range" tick={{ fill: "#484f58", fontSize: 8 }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} />
        <Tooltip
          contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 10 }}
        />
        <Bar dataKey="count" fill="#58a6ff" name="Trade Count" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function ExecutionQualityRadar({ data }: { data: QualityScore | null }) {
  if (!data || !data.dimensions.length) {
    return <div className="flex items-center justify-center h-full text-terminal-text-faint text-xs">Awaiting quality data...</div>;
  }

  const chartData = data.dimensions.map((dim, i) => ({
    dimension: dim,
    score: data.scores[i] || 0,
    fullMark: 100,
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <RadarChart data={chartData} cx="50%" cy="50%" outerRadius="70%">
        <PolarGrid stroke="#1e2530" />
        <PolarAngleAxis dataKey="dimension" tick={{ fill: "#8b949e", fontSize: 9 }} />
        <PolarRadiusAxis tick={{ fill: "#484f58", fontSize: 8 }} domain={[0, 100]} />
        <Radar name="Score" dataKey="score" stroke="#00d4aa" fill="#00d4aa" fillOpacity={0.2} strokeWidth={2} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

function VenueAnalysis({ venues }: { venues: VenueStat[] }) {
  if (!venues.length) return <div className="flex items-center justify-center h-full text-terminal-text-faint text-xs">Awaiting venue data...</div>;

  return (
    <div className="text-[10px] overflow-auto h-full">
      <table className="w-full">
        <thead>
          <tr className="text-terminal-text-faint border-b border-terminal-border">
            {["Venue", "Avg Cost (bps)", "Fills", "Latency (ms)", "Quality"].map(h => (
              <th key={h} className="py-1 px-2 text-left font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {venues.map(v => (
            <tr key={v.venue} className="border-b border-terminal-border/50 hover:bg-white/[0.02]">
              <td className="py-1.5 px-2 font-mono font-semibold text-terminal-text-primary">{v.venue}</td>
              <td className="py-1.5 px-2 font-mono">
                <div className="flex items-center gap-2">
                  <span>{v.avg_cost_bps.toFixed(2)}</span>
                  <div className="flex-1 h-1.5 bg-terminal-bg rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min((v.avg_cost_bps / 6) * 100, 100)}%`,
                        backgroundColor: v.avg_cost_bps < 2.8 ? "#3fb950" : v.avg_cost_bps < 3.5 ? "#d29922" : "#f85149",
                      }}
                    />
                  </div>
                </div>
              </td>
              <td className="py-1.5 px-2 font-mono">{v.fills}</td>
              <td className="py-1.5 px-2 font-mono">{v.avg_latency_ms.toFixed(1)}</td>
              <td className="py-1.5 px-2 font-mono">
                <span className={v.quality_score >= 70 ? "text-terminal-positive" : v.quality_score >= 40 ? "text-terminal-warning" : "text-terminal-negative"}>
                  {v.quality_score.toFixed(0)}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AlgoPerformance({ algos }: { algos: AlgoStat[] }) {
  if (!algos.length) return <div className="flex items-center justify-center h-full text-terminal-text-faint text-xs">Awaiting algo data...</div>;

  return (
    <div className="text-[10px] overflow-auto h-full">
      <table className="w-full">
        <thead>
          <tr className="text-terminal-text-faint border-b border-terminal-border">
            {["Algorithm", "Avg Cost (bps)", "Avg IS ($)", "VWAP Slip", "Trades", "Quality"].map(h => (
              <th key={h} className="py-1 px-2 text-left font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {algos.map(a => (
            <tr key={a.algo} className="border-b border-terminal-border/50 hover:bg-white/[0.02]">
              <td className="py-1.5 px-2 font-mono font-semibold text-terminal-accent">{a.algo}</td>
              <td className="py-1.5 px-2 font-mono">
                <div className="flex items-center gap-2">
                  <span>{a.avg_cost_bps.toFixed(2)}</span>
                  <div className="flex-1 h-1.5 bg-terminal-bg rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min((a.avg_cost_bps / 6) * 100, 100)}%`,
                        backgroundColor: a.avg_cost_bps < 2.5 ? "#3fb950" : a.avg_cost_bps < 3.5 ? "#d29922" : "#f85149",
                      }}
                    />
                  </div>
                </div>
              </td>
              <td className={`py-1.5 px-2 font-mono ${a.avg_is_usd > 0 ? "text-terminal-negative" : "text-terminal-positive"}`}>
                ${Math.abs(a.avg_is_usd).toFixed(0)}
              </td>
              <td className={`py-1.5 px-2 font-mono ${a.avg_vwap_slip_bps > 0 ? "text-terminal-negative" : "text-terminal-positive"}`}>
                {a.avg_vwap_slip_bps > 0 ? "+" : ""}{a.avg_vwap_slip_bps.toFixed(1)}
              </td>
              <td className="py-1.5 px-2 font-mono">{a.trades}</td>
              <td className="py-1.5 px-2 font-mono">
                <span className={a.quality_score >= 70 ? "text-terminal-positive" : a.quality_score >= 40 ? "text-terminal-warning" : "text-terminal-negative"}>
                  {a.quality_score.toFixed(0)}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SectorBreakdown({ sectors }: { sectors: SectorStat[] }) {
  if (!sectors.length) return <div className="flex items-center justify-center h-full text-terminal-text-faint text-xs">Awaiting sector data...</div>;

  const colors = ["#00d4aa", "#f0883e", "#58a6ff", "#d2a8ff", "#f85149", "#3fb950", "#d29922", "#8b949e"];

  return (
    <div className="text-[10px] space-y-1 overflow-auto h-full">
      {sectors.map((d, i) => (
        <div key={d.sector} className="flex items-center gap-2 px-1 py-1 rounded hover:bg-white/[0.02]">
          <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: colors[i % colors.length] }} />
          <span className="text-terminal-text-muted w-28 truncate">{d.sector}</span>
          <div className="flex-1 h-1.5 bg-terminal-bg rounded-full overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{ width: `${Math.min((d.avg_cost_bps / 6) * 100, 100)}%`, backgroundColor: colors[i % colors.length] }}
            />
          </div>
          <span className="font-mono text-terminal-text-primary w-14 text-right">{d.avg_cost_bps} bps</span>
          <span className="font-mono text-terminal-text-faint w-8 text-right">{d.trades}</span>
        </div>
      ))}
    </div>
  );
}

function BenchmarkComparison({ benchmarks }: { benchmarks: Benchmark[] }) {
  if (!benchmarks.length) return <div className="flex items-center justify-center h-full text-terminal-text-faint text-xs">Awaiting benchmark data...</div>;

  return (
    <div className="text-[10px] overflow-auto h-full">
      <table className="w-full">
        <thead>
          <tr className="text-terminal-text-faint border-b border-terminal-border">
            {["Benchmark", "Avg Slip (bps)", "Win Rate", "Shortfall ($)", "Trades"].map(h => (
              <th key={h} className="py-1 px-2 text-left font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {benchmarks.map(b => (
            <tr key={b.benchmark} className="border-b border-terminal-border/50 hover:bg-white/[0.02]">
              <td className="py-1.5 px-2 font-mono font-semibold text-terminal-accent">{b.benchmark}</td>
              <td className={`py-1.5 px-2 font-mono ${b.avg_slippage_bps > 0 ? "text-terminal-negative" : "text-terminal-positive"}`}>
                {b.avg_slippage_bps > 0 ? "+" : ""}{b.avg_slippage_bps.toFixed(2)}
              </td>
              <td className="py-1.5 px-2 font-mono">
                <div className="flex items-center gap-1.5">
                  <span className={b.win_rate >= 50 ? "text-terminal-positive" : "text-terminal-negative"}>
                    {b.win_rate.toFixed(1)}%
                  </span>
                  <div className="flex-1 h-1 bg-terminal-bg rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${b.win_rate}%`,
                        backgroundColor: b.win_rate >= 50 ? "#3fb950" : "#f85149",
                      }}
                    />
                  </div>
                </div>
              </td>
              <td className="py-1.5 px-2 font-mono text-terminal-text-muted">
                ${Math.abs(b.total_shortfall_usd).toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </td>
              <td className="py-1.5 px-2 font-mono text-terminal-text-muted">{b.trades_evaluated}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function OutlierTable({ outliers }: { outliers: Outlier[] }) {
  if (!outliers.length) return <div className="flex items-center justify-center h-full text-terminal-text-faint text-xs">No outliers detected</div>;

  const reasonColor: Record<string, string> = {
    HIGH_COST: "text-terminal-negative",
    HIGH_MARKET_IMPACT: "text-terminal-warning",
    ADVERSE_TIMING: "text-terminal-warning",
    NEGATIVE_COST: "text-terminal-positive",
  };

  return (
    <div className="text-[10px] overflow-auto h-full">
      <table className="w-full">
        <thead>
          <tr className="text-terminal-text-faint border-b border-terminal-border">
            {["Ticker", "Side", "Cost (bps)", "Z-Score", "Reason", "Time"].map(h => (
              <th key={h} className="py-1 px-2 text-left font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {outliers.map((o, i) => (
            <tr key={i} className="border-b border-terminal-border/50 hover:bg-white/[0.02]">
              <td className="py-1.5 px-2 font-mono font-semibold text-terminal-text-primary">{o.ticker}</td>
              <td className={`py-1.5 px-2 font-mono font-semibold ${o.side === "BUY" ? "text-terminal-positive" : "text-terminal-negative"}`}>{o.side}</td>
              <td className="py-1.5 px-2 font-mono text-terminal-warning">{o.total_cost_bps.toFixed(1)}</td>
              <td className="py-1.5 px-2 font-mono">{o.z_score.toFixed(2)}σ</td>
              <td className={`py-1.5 px-2 font-mono text-[9px] ${reasonColor[o.reason] || "text-terminal-text-muted"}`}>
                {o.reason.replace(/_/g, " ")}
              </td>
              <td className="py-1.5 px-2 font-mono text-terminal-text-faint">{o.timestamp.slice(11, 19)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PerAssetTable({ records }: { records: TCARecord[] }) {
  const [sortKey, setSortKey] = useState<"total_cost_bps" | "implementation_shortfall_usd" | "vwap_slippage_bps">("total_cost_bps");
  const sorted = useMemo(() => [...records].sort((a, b) => b[sortKey] - a[sortKey]), [records, sortKey]);

  if (!records.length) return <div className="flex items-center justify-center h-full text-terminal-text-faint text-xs">Awaiting trade data...</div>;

  return (
    <div className="text-[10px] overflow-auto h-full">
      <div className="flex items-center gap-2 px-2 py-1 mb-1 sticky top-0 bg-terminal-panel z-10">
        <span className="text-terminal-text-faint">SORT BY:</span>
        {([
          ["total_cost_bps", "Total Cost"],
          ["implementation_shortfall_usd", "Impl. Shortfall"],
          ["vwap_slippage_bps", "VWAP Slip"],
        ] as const).map(([k, label]) => (
          <button
            key={k}
            onClick={() => setSortKey(k)}
            className={`px-2 py-0.5 rounded text-[9px] font-medium transition-colors ${
              sortKey === k
                ? "bg-terminal-accent/20 text-terminal-accent"
                : "text-terminal-text-muted hover:text-terminal-text-primary"
            }`}
          >{label}</button>
        ))}
      </div>
      <table className="w-full">
        <thead>
          <tr className="text-terminal-text-faint border-b border-terminal-border">
            {["Ticker","Side","Qty","Fill","Arrival","VWAP","Spread","Impact","Timing","Comm","Total","IS ($)","VWAP Slip","Venue","Algo"].map(h => (
              <th key={h} className="py-1 px-1 text-left font-medium whitespace-nowrap">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.slice(0, 100).map((r, i) => (
            <tr key={r.order_id || i} className="border-b border-terminal-border/50 hover:bg-white/[0.02]">
              <td className="py-1 px-1 font-mono font-semibold text-terminal-text-primary">{r.ticker}</td>
              <td className={`py-1 px-1 font-mono font-semibold ${r.side === "BUY" || r.side === "COVER" ? "text-terminal-positive" : "text-terminal-negative"}`}>{r.side}</td>
              <td className="py-1 px-1 font-mono">{r.quantity.toLocaleString()}</td>
              <td className="py-1 px-1 font-mono">{r.fill_price.toFixed(2)}</td>
              <td className="py-1 px-1 font-mono">{r.arrival_price.toFixed(2)}</td>
              <td className="py-1 px-1 font-mono">{r.vwap_price > 0 ? r.vwap_price.toFixed(2) : "—"}</td>
              <td className="py-1 px-1 font-mono">{r.spread_cost_bps.toFixed(1)}</td>
              <td className="py-1 px-1 font-mono text-terminal-warning">{r.market_impact_bps.toFixed(1)}</td>
              <td className={`py-1 px-1 font-mono ${r.timing_cost_bps > 0 ? "text-terminal-negative" : "text-terminal-positive"}`}>{r.timing_cost_bps.toFixed(1)}</td>
              <td className="py-1 px-1 font-mono text-terminal-text-muted">{r.commission_bps.toFixed(1)}</td>
              <td className="py-1 px-1 font-mono font-semibold text-terminal-accent">{r.total_cost_bps.toFixed(1)}</td>
              <td className={`py-1 px-1 font-mono ${r.implementation_shortfall_usd > 0 ? "text-terminal-negative" : "text-terminal-positive"}`}>
                {r.implementation_shortfall_usd > 0 ? "+" : ""}${Math.abs(r.implementation_shortfall_usd).toFixed(0)}
              </td>
              <td className={`py-1 px-1 font-mono ${r.vwap_slippage_bps > 0 ? "text-terminal-negative" : "text-terminal-positive"}`}>
                {r.vwap_slippage_bps > 0 ? "+" : ""}{r.vwap_slippage_bps.toFixed(1)}
              </td>
              <td className="py-1 px-1 text-terminal-text-muted">{r.venue}</td>
              <td className="py-1 px-1 font-mono text-terminal-accent">{r.algo}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ═══════════ MAIN PAGE ═══════════

export default function TCAPage() {
  const {
    records, summary, trend, decomposition,
    venues, algos, sectors, outliers, benchmarks,
    isDistribution, qualityScore,
  } = useTCA();
  const [activeView, setActiveView] = useState<"overview" | "detail" | "analytics">("overview");

  return (
    <div className="h-full flex flex-col gap-1 p-1 overflow-hidden">
      {/* Summary Cards Row */}
      <div className="flex-shrink-0">
        <SummaryCards summary={summary} />
      </div>

      {/* Tab Toggle */}
      <div className="flex items-center gap-1 flex-shrink-0 px-1">
        {(["overview", "detail", "analytics"] as const).map(v => (
          <button
            key={v}
            onClick={() => setActiveView(v)}
            className={`px-3 py-1 rounded text-[10px] font-medium tracking-wider transition-colors ${
              activeView === v
                ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30"
                : "text-terminal-text-muted hover:text-terminal-text-primary hover:bg-white/[0.03] border border-transparent"
            }`}
          >{v === "overview" ? "OVERVIEW" : v === "detail" ? "PER-ASSET DETAIL" : "ANALYTICS"}</button>
        ))}
        <div className="flex-1" />
        <span className="text-[9px] text-terminal-text-faint font-mono">
          {summary ? `${summary.total_trades} trades · $${(summary.total_volume_usd / 1e6).toFixed(2)}M vol` : "loading..."}
        </span>
      </div>

      {activeView === "overview" ? (
        <div className="flex-1 min-h-0">
          <ResizableDashboard defaultSizes={[73, 27]} minSizes={[45, 18]}>
            {/* Left: 2 rows of charts */}
            <div className="h-full flex flex-col gap-1 overflow-hidden">
              <div className="flex-1 min-h-0">
                <ResizableDashboard defaultSizes={[50, 50]} minSizes={[30, 30]}>
                  <DashboardPanel title="COST DECOMPOSITION BY ASSET">
                    <CostDecomposition data={decomposition} />
                  </DashboardPanel>
                  <DashboardPanel title="30-DAY COST TREND">
                    <CostTrendChart data={trend} />
                  </DashboardPanel>
                </ResizableDashboard>
              </div>
              <div className="flex-1 min-h-0">
                <ResizableDashboard defaultSizes={[50, 50]} minSizes={[30, 30]}>
                  <DashboardPanel title="VWAP SLIPPAGE BY ASSET">
                    <VWAPSlippageChart records={records} />
                  </DashboardPanel>
                  <DashboardPanel title="IMPL. SHORTFALL DISTRIBUTION">
                    <ISDistribution data={isDistribution} />
                  </DashboardPanel>
                </ResizableDashboard>
              </div>
            </div>

            {/* Right: Sector Costs sidebar */}
            <DashboardPanel title="SECTOR COSTS">
              <SectorBreakdown sectors={sectors} />
            </DashboardPanel>
          </ResizableDashboard>
        </div>
      ) : activeView === "detail" ? (
        <div className="flex-1 min-h-0">
          <ResizableDashboard defaultSizes={[75, 25]} minSizes={[50, 18]}>
            <DashboardPanel title="PER-ASSET TRANSACTION COST ANALYSIS" noPadding>
              <PerAssetTable records={records} />
            </DashboardPanel>
            <div className="h-full flex flex-col gap-1 overflow-hidden">
              <DashboardPanel title="VENUE ANALYSIS" className="flex-1">
                <VenueAnalysis venues={venues} />
              </DashboardPanel>
              <DashboardPanel title="ALGO PERFORMANCE" className="flex-1">
                <AlgoPerformance algos={algos} />
              </DashboardPanel>
            </div>
          </ResizableDashboard>
        </div>
      ) : (
        /* Analytics view: radar + benchmarks + outliers */
        <div className="flex-1 min-h-0">
          <ResizableDashboard defaultSizes={[50, 50]} minSizes={[30, 30]}>
            <div className="h-full flex flex-col gap-1 overflow-hidden">
              <DashboardPanel title="EXECUTION QUALITY RADAR" className="flex-1">
                <ExecutionQualityRadar data={qualityScore} />
              </DashboardPanel>
              <DashboardPanel title="BENCHMARK COMPARISON" className="flex-1">
                <BenchmarkComparison benchmarks={benchmarks} />
              </DashboardPanel>
            </div>
            <DashboardPanel title="EXECUTION OUTLIERS">
              <OutlierTable outliers={outliers} />
            </DashboardPanel>
          </ResizableDashboard>
        </div>
      )}
    </div>
  );
}
