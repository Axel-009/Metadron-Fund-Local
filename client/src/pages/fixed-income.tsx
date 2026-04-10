import { useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
} from "recharts";
import { useEngineQuery } from "@/hooks/use-engine-api";

// ═══════════ TYPES ═══════════

interface FISummary {
  total_exposure?: number;
  avg_duration?: number;
  avg_yield?: number;
  avg_rating?: string;
  dv01?: number;
  convexity?: number;
}

interface BondHolding {
  security: string;
  ticker?: string;
  coupon: number;
  maturity: string;
  rating: string;
  yield: number;
  duration: number;
  dv01: number;
  face_value: number;
  market_value: number;
  pnl: number;
  spread: number;
  sector: string;
}

interface YieldPoint {
  tenor: string;
  rate: number;
  yield?: number;
}

interface CreditBucket {
  name: string;
  pct: number;
  color: string;
}

interface DurationBucket {
  bucket: string;
  dv01: number;
}

interface SpreadPoint {
  date: string;
  ig?: number;
  hy?: number;
}

// ═══════════ COLORS & UTILS ═══════════

const SECTOR_COLORS: Record<string, string> = {
  Govt:   "#3b82f6",
  Corp:   "#00d4aa",
  Agency: "#3fb950",
  Muni:   "#a855f7",
  HY:     "#f85149",
  Aggregate: "#3b82f6",
  TIPS:   "#d29922",
  Other:  "#8b949e",
};

const RATING_COLORS: Record<string, string> = {
  "AAA":"#00d4aa", "AA+":"#22d3ee", "AA":"#3b82f6", "AA-":"#60a5fa",
  "A":"#a78bfa", "A-":"#a78bfa", "BBB+":"#d29922", "BBB":"#f59e0b", "BB":"#f85149",
  "NR":"#8b949e",
};

const TOOLTIP_STYLE = {
  backgroundColor: "#0d1117",
  border: "1px solid #21262d",
  borderRadius: 4,
  fontSize: 10,
  color: "#e6edf3",
  padding: "4px 8px",
};

const fmtM = (n: number) => `$${(n / 1_000_000).toFixed(2)}M`;
const fmtK = (n: number) => `$${n.toLocaleString()}`;

export default function FixedIncomeDashboard() {
  // ─── Engine API — all real data, no static fallbacks ─────
  const { data: summaryData } = useEngineQuery<FISummary>("/fixed-income/summary", { refetchInterval: 30000 });
  const { data: holdingsData } = useEngineQuery<{ holdings: BondHolding[] }>("/fixed-income/holdings", { refetchInterval: 30000 });
  const { data: curveData } = useEngineQuery<{ curve: YieldPoint[] }>("/fixed-income/yield-curve", { refetchInterval: 60000 });
  const { data: creditData } = useEngineQuery<{ quality: CreditBucket[] }>("/fixed-income/credit-quality", { refetchInterval: 60000 });
  const { data: ladderData } = useEngineQuery<{ ladder: DurationBucket[] }>("/fixed-income/duration-ladder", { refetchInterval: 60000 });
  const { data: spreadData } = useEngineQuery<{ data: SpreadPoint[]; ig_current?: number; hy_current?: number }>("/fixed-income/spread-history", { refetchInterval: 120000 });

  // Also pull macro yield curve + credit pulse as secondary overlay
  const { data: macroYield } = useEngineQuery<Record<string, number>>("/macro/yield-curve", { refetchInterval: 30000 });
  const { data: macroCreditPulse } = useEngineQuery<Record<string, number>>("/macro/credit-pulse", { refetchInterval: 30000 });

  // ─── Derived data ─────────────────────────────────────────
  const summary = summaryData || {};
  const holdings = holdingsData?.holdings || [];
  const yieldCurve = useMemo(() => {
    const curve = curveData?.curve || [];
    if (curve.length > 0) return curve;
    // Fallback: try to construct from macro yield-curve endpoint
    if (macroYield && macroYield.yield_2y) {
      return [
        { tenor: "2Y", rate: macroYield.yield_2y },
        { tenor: "5Y", rate: macroYield.yield_5y || 0 },
        { tenor: "10Y", rate: macroYield.yield_10y || 0 },
        { tenor: "30Y", rate: macroYield.yield_30y || 0 },
      ].filter(p => p.rate > 0);
    }
    return [];
  }, [curveData, macroYield]);

  const creditQuality = creditData?.quality || [];
  const durationLadder = ladderData?.ladder || [];
  const spreadHistory = useMemo(() => {
    const raw = spreadData?.data || [];
    return raw.map((d, i) => ({ day: i + 1, ig: d.ig || 0, hy: d.hy || 0 }));
  }, [spreadData]);

  const totalPnL = holdings.reduce((s, b) => s + (b.pnl || 0), 0);

  return (
    <div className="h-full grid gap-1 p-1 overflow-hidden" style={{ gridTemplateColumns: "1fr 1fr 200px", gridTemplateRows: "56px 1fr 170px" }}>
      {/* ─── Row 1: Summary Cards ─── */}
      <div className="col-span-3 grid gap-1" style={{ gridTemplateColumns: "repeat(6, 1fr)" }}>
        {[
          { label:"TOTAL FI EXPOSURE", value: summary.total_exposure ? `$${(summary.total_exposure/1e6).toFixed(1)}M` : "—", color:"text-terminal-accent" },
          { label:"AVG DURATION",      value: summary.avg_duration ? `${summary.avg_duration.toFixed(2)}Y` : "—",           color:"text-terminal-text-primary" },
          { label:"AVG YIELD",         value: summary.avg_yield ? `${summary.avg_yield.toFixed(2)}%` : "—",               color:"text-terminal-warning" },
          { label:"AVG RATING",        value: summary.avg_rating || "—",                               color:"text-terminal-positive" },
          { label:"PORTFOLIO DV01",    value: summary.dv01 ? `$${summary.dv01.toLocaleString()}` : "—",             color:"text-terminal-accent" },
          { label:"CONVEXITY",         value: summary.convexity ? summary.convexity.toFixed(2) : "—",                    color:"text-terminal-text-primary" },
        ].map(c => (
          <div key={c.label} className="terminal-panel p-2 flex flex-col justify-between">
            <span className="text-[9px] text-terminal-text-muted tracking-widest">{c.label}</span>
            <span className={`text-[14px] font-mono font-bold ${c.color}`}>{c.value}</span>
          </div>
        ))}
      </div>

      {/* ─── Row 2: Bond Holdings Table (spans 2 cols) ─── */}
      <DashboardPanel
        title="BOND HOLDINGS"
        className="col-span-2"
        noPadding
        headerRight={
          holdings.length > 0 ? (
            <span className={`text-[10px] font-mono ${totalPnL >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
              P&L: {totalPnL >= 0 ? "+$" : "-$"}{Math.abs(totalPnL).toLocaleString()}
            </span>
          ) : (
            <span className="text-[10px] font-mono text-terminal-text-muted">AWAITING DATA</span>
          )
        }
      >
        <div className="overflow-auto h-full">
          {holdings.length === 0 ? (
            <div className="flex items-center justify-center h-full text-terminal-text-muted text-[11px] font-mono">
              No fixed income positions — data loads from broker + OpenBB
            </div>
          ) : (
            <table className="w-full text-[10px] font-mono border-collapse">
              <thead className="sticky top-0 bg-terminal-surface z-10">
                <tr className="border-b border-terminal-border">
                  <th className="text-terminal-text-muted text-[9px] tracking-widest text-left px-2 py-1">SECURITY</th>
                  <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">COUPON</th>
                  <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">MATURITY</th>
                  <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1">RTG</th>
                  <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">YTM%</th>
                  <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">DUR</th>
                  <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">DV01</th>
                  <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">FACE</th>
                  <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">MKT VAL</th>
                  <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">P&L</th>
                  <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">SPRD</th>
                  <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1">SECTOR</th>
                </tr>
              </thead>
              <tbody>
                {holdings.map((b, i) => (
                  <tr key={b.security} className={`border-b border-terminal-border hover:bg-[#161b22] transition-colors ${i % 2 === 0 ? "" : "bg-[#0d1117]"}`}>
                    <td className="px-2 py-0.5 text-terminal-text-primary max-w-[180px] truncate">{b.security}</td>
                    <td className="px-1 py-0.5 text-terminal-text-muted text-right">{b.coupon.toFixed(3)}%</td>
                    <td className="px-1 py-0.5 text-terminal-text-muted text-right">{b.maturity}</td>
                    <td className="px-1 py-0.5">
                      <span className="px-0.5 rounded text-[8px]" style={{ backgroundColor: (RATING_COLORS[b.rating] || "#888") + "22", color: RATING_COLORS[b.rating] || "#888" }}>
                        {b.rating}
                      </span>
                    </td>
                    <td className="px-1 py-0.5 text-terminal-warning text-right">{b.yield.toFixed(2)}%</td>
                    <td className="px-1 py-0.5 text-terminal-text-primary text-right">{b.duration.toFixed(2)}</td>
                    <td className="px-1 py-0.5 text-terminal-accent text-right">{fmtK(b.dv01)}</td>
                    <td className="px-1 py-0.5 text-terminal-text-muted text-right">{fmtM(b.face_value)}</td>
                    <td className="px-1 py-0.5 text-terminal-text-primary text-right">{fmtM(b.market_value)}</td>
                    <td className={`px-1 py-0.5 text-right font-bold ${b.pnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                      {b.pnl >= 0 ? "+" : ""}{(b.pnl/1000).toFixed(1)}K
                    </td>
                    <td className={`px-1 py-0.5 text-right ${b.spread > 100 ? "text-terminal-negative" : b.spread > 50 ? "text-terminal-warning" : "text-terminal-text-muted"}`}>
                      {b.spread}
                    </td>
                    <td className="px-1 py-0.5">
                      <span className="px-0.5 rounded text-[8px]" style={{ backgroundColor: (SECTOR_COLORS[b.sector] || "#888") + "22", color: SECTOR_COLORS[b.sector] || "#888" }}>
                        {b.sector}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </DashboardPanel>

      {/* ─── Row 2: Right sidebar ─── */}
      <div className="flex flex-col gap-1">
        {/* Yield Curve */}
        <DashboardPanel title="US YIELD CURVE" className="flex-1">
          {yieldCurve.length === 0 ? (
            <div className="flex items-center justify-center h-full text-terminal-text-muted text-[10px] font-mono">Loading yield curve...</div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={yieldCurve} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <XAxis dataKey="tenor" tick={{ fontSize: 8, fill: "#8b949e" }} />
                <YAxis domain={["auto", "auto"]} tickFormatter={v => `${v.toFixed(2)}%`} tick={{ fontSize: 8, fill: "#8b949e" }} />
                <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(3)}%`, "Yield"]} />
                <Line type="monotone" dataKey="rate" stroke="#00d4aa" strokeWidth={2} dot={{ fill: "#00d4aa", r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </DashboardPanel>

        {/* Credit Quality Distribution */}
        <DashboardPanel title="CREDIT QUALITY" className="flex-none" style={{ height: "160px" }}>
          {creditQuality.length === 0 ? (
            <div className="flex items-center justify-center h-full text-terminal-text-muted text-[10px] font-mono">Loading credit data...</div>
          ) : (
            <>
              <div className="flex flex-col gap-1">
                {creditQuality.map(c => (
                  <div key={c.name} className="flex items-center gap-2">
                    <span className="text-[9px] font-mono w-7 text-terminal-text-muted">{c.name}</span>
                    <div className="flex-1 h-3 bg-[#161b22] rounded overflow-hidden">
                      <div className="h-full rounded" style={{ width: `${c.pct}%`, backgroundColor: c.color }} />
                    </div>
                    <span className="text-[9px] font-mono w-8 text-right" style={{ color: c.color }}>{c.pct}%</span>
                  </div>
                ))}
              </div>
              <div className="mt-2">
                <ResponsiveContainer width="100%" height={50}>
                  <PieChart>
                    <Pie data={creditQuality} dataKey="pct" innerRadius={14} outerRadius={24} paddingAngle={1} startAngle={90} endAngle={450}>
                      {creditQuality.map(c => <Cell key={c.name} fill={c.color} />)}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </DashboardPanel>
      </div>

      {/* ─── Row 3: Duration Ladder + Spread Monitor ─── */}
      <DashboardPanel title="DURATION LADDER (DV01 BY BUCKET)">
        {durationLadder.length === 0 ? (
          <div className="flex items-center justify-center h-full text-terminal-text-muted text-[10px] font-mono">Loading duration data...</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={durationLadder} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
              <XAxis dataKey="bucket" tick={{ fontSize: 8, fill: "#8b949e" }} />
              <YAxis tickFormatter={v => `$${(v/1000).toFixed(0)}K`} tick={{ fontSize: 8, fill: "#8b949e" }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`$${v.toLocaleString()}`, "DV01"]} />
              <Bar dataKey="dv01" fill="#3b82f6" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </DashboardPanel>

      <DashboardPanel title="CREDIT SPREAD MONITOR (FRED OAS)" className="col-span-2">
        {spreadHistory.length === 0 ? (
          <div className="flex items-center justify-center h-full text-terminal-text-muted text-[10px] font-mono">Loading spread history...</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={spreadHistory} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
              <XAxis dataKey="day" tick={{ fontSize: 8, fill: "#8b949e" }} tickFormatter={v => `D${v}`} />
              <YAxis yAxisId="ig" orientation="left"  domain={["auto", "auto"]} tickFormatter={v => `${v}bps`} tick={{ fontSize: 8, fill: "#8b949e" }} />
              <YAxis yAxisId="hy" orientation="right" domain={["auto", "auto"]} tickFormatter={v => `${v}bps`} tick={{ fontSize: 8, fill: "#8b949e" }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number, name: string) => [`${v.toFixed(1)} bps`, name === "ig" ? "IG Spread" : "HY Spread"]} />
              <Line yAxisId="ig" type="monotone" dataKey="ig" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="ig" />
              <Line yAxisId="hy" type="monotone" dataKey="hy" stroke="#f85149" strokeWidth={1.5} dot={false} name="hy" />
            </LineChart>
          </ResponsiveContainer>
        )}
      </DashboardPanel>
    </div>
  );
}
