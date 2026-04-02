import { useState, useEffect, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, LineChart, Line, ComposedChart, Cell,
  PieChart, Pie, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";

// ═══════════ SIMULATED TCA DATA ═══════════

interface TCARecord {
  orderId: string;
  ticker: string;
  side: "BUY" | "SELL";
  qty: number;
  avgFillPrice: number;
  arrivalPrice: number;
  vwapPrice: number;
  spreadCostBps: number;
  marketImpactBps: number;
  timingCostBps: number;
  commissionBps: number;
  totalCostBps: number;
  implementationShortfallUsd: number;
  vwapSlippageBps: number;
  venue: string;
  algo: string;
  duration: string;
  participationRate: number;
  sector: string;
}

function generateTCARecords(): TCARecord[] {
  const tickers = [
    { t: "AAPL", s: "Technology" }, { t: "MSFT", s: "Technology" }, { t: "NVDA", s: "Technology" },
    { t: "GOOGL", s: "Communication" }, { t: "AMZN", s: "Consumer Disc." }, { t: "JPM", s: "Financials" },
    { t: "XOM", s: "Energy" }, { t: "JNJ", s: "Healthcare" }, { t: "V", s: "Financials" },
    { t: "PG", s: "Consumer Staples" }, { t: "META", s: "Communication" }, { t: "UNH", s: "Healthcare" },
    { t: "HD", s: "Consumer Disc." }, { t: "BAC", s: "Financials" }, { t: "TSLA", s: "Consumer Disc." },
    { t: "LLY", s: "Healthcare" }, { t: "AVGO", s: "Technology" }, { t: "MRK", s: "Healthcare" },
    { t: "KO", s: "Consumer Staples" }, { t: "CVX", s: "Energy" },
  ];
  const venues = ["NASDAQ", "NYSE", "ARCA", "BATS", "IEX", "EDGX"];
  const algos = ["TWAP", "VWAP", "IS", "POV", "CLOSE", "ADAPTIVE"];

  return tickers.map(({ t, s }, i) => {
    const side = i % 3 === 0 ? "SELL" : "BUY";
    const qty = Math.floor(500 + Math.random() * 9500);
    const base = 50 + Math.random() * 400;
    const spread = 0.3 + Math.random() * 2.2;
    const impact = 0.5 + Math.random() * 4.0;
    const timing = -1.0 + Math.random() * 3.0;
    const commission = 0.15 + Math.random() * 0.4;
    const total = spread + impact + timing + commission;
    const arrival = base;
    const fill = side === "BUY" ? base + (total / 10000) * base : base - (total / 10000) * base;
    const vwap = base + (Math.random() - 0.5) * 0.5;
    const vwapSlip = ((fill - vwap) / vwap) * 10000 * (side === "BUY" ? 1 : -1);
    const isUsd = (fill - arrival) * qty * (side === "BUY" ? 1 : -1);

    return {
      orderId: `TCA-${String(i + 1).padStart(4, "0")}`,
      ticker: t,
      side,
      qty,
      avgFillPrice: +fill.toFixed(2),
      arrivalPrice: +arrival.toFixed(2),
      vwapPrice: +vwap.toFixed(2),
      spreadCostBps: +spread.toFixed(2),
      marketImpactBps: +impact.toFixed(2),
      timingCostBps: +timing.toFixed(2),
      commissionBps: +commission.toFixed(2),
      totalCostBps: +total.toFixed(2),
      implementationShortfallUsd: +isUsd.toFixed(2),
      vwapSlippageBps: +vwapSlip.toFixed(2),
      venue: venues[Math.floor(Math.random() * venues.length)],
      algo: algos[Math.floor(Math.random() * algos.length)],
      duration: `${Math.floor(1 + Math.random() * 45)}m ${Math.floor(Math.random() * 60)}s`,
      participationRate: +(2 + Math.random() * 18).toFixed(1),
      sector: s,
    };
  });
}

function generateCostTrend() {
  const days: any[] = [];
  for (let i = 29; i >= 0; i--) {
    const d = new Date(Date.now() - i * 86400000);
    days.push({
      date: d.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      spreadBps: +(0.8 + Math.random() * 1.4).toFixed(2),
      impactBps: +(1.0 + Math.random() * 2.5).toFixed(2),
      timingBps: +(-0.5 + Math.random() * 2.0).toFixed(2),
      totalBps: +(2.0 + Math.random() * 4.0).toFixed(2),
    });
  }
  return days;
}

function generateVenueComparison() {
  return [
    { venue: "NASDAQ", avgCost: 2.84, fills: 342, avgLatency: 0.42 },
    { venue: "NYSE", avgCost: 3.12, fills: 287, avgLatency: 0.68 },
    { venue: "ARCA", avgCost: 2.71, fills: 198, avgLatency: 0.31 },
    { venue: "BATS", avgCost: 2.94, fills: 156, avgLatency: 0.28 },
    { venue: "IEX", avgCost: 3.48, fills: 89, avgLatency: 1.12 },
    { venue: "EDGX", avgCost: 2.63, fills: 124, avgLatency: 0.35 },
  ];
}

function generateAlgoComparison() {
  return [
    { algo: "TWAP", avgCost: 3.21, avgIS: 412, trades: 84 },
    { algo: "VWAP", avgCost: 2.87, avgIS: 324, trades: 156 },
    { algo: "IS", avgCost: 2.42, avgIS: 218, trades: 102 },
    { algo: "POV", avgCost: 3.08, avgIS: 387, trades: 67 },
    { algo: "CLOSE", avgCost: 3.54, avgIS: 468, trades: 43 },
    { algo: "ADAPTIVE", avgCost: 2.31, avgIS: 196, trades: 78 },
  ];
}

// ═══════════ COMPONENTS ═══════════

function SummaryCards({ records }: { records: TCARecord[] }) {
  const avgTotal = records.reduce((s, r) => s + r.totalCostBps, 0) / records.length;
  const avgSpread = records.reduce((s, r) => s + r.spreadCostBps, 0) / records.length;
  const avgImpact = records.reduce((s, r) => s + r.marketImpactBps, 0) / records.length;
  const avgVwapSlip = records.reduce((s, r) => s + r.vwapSlippageBps, 0) / records.length;
  const totalIS = records.reduce((s, r) => s + Math.abs(r.implementationShortfallUsd), 0);
  const bestTicker = records.reduce((best, r) => r.totalCostBps < best.totalCostBps ? r : best);
  const worstTicker = records.reduce((worst, r) => r.totalCostBps > worst.totalCostBps ? r : worst);

  const cards = [
    { label: "AVG TOTAL COST", value: `${avgTotal.toFixed(2)} bps`, color: "text-terminal-accent" },
    { label: "AVG SPREAD", value: `${avgSpread.toFixed(2)} bps`, color: "text-terminal-text-primary" },
    { label: "AVG MKT IMPACT", value: `${avgImpact.toFixed(2)} bps`, color: "text-terminal-warning" },
    { label: "AVG VWAP SLIP", value: `${avgVwapSlip.toFixed(2)} bps`, color: avgVwapSlip > 0 ? "text-terminal-negative" : "text-terminal-positive" },
    { label: "TOTAL IMPL. SHORTFALL", value: `$${totalIS.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, color: "text-terminal-negative" },
    { label: "BEST EXECUTION", value: `${bestTicker.ticker} (${bestTicker.totalCostBps.toFixed(1)} bps)`, color: "text-terminal-positive" },
    { label: "WORST EXECUTION", value: `${worstTicker.ticker} (${worstTicker.totalCostBps.toFixed(1)} bps)`, color: "text-terminal-negative" },
    { label: "TOTAL TRADES", value: `${records.length}`, color: "text-terminal-text-primary" },
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

function CostDecomposition({ records }: { records: TCARecord[] }) {
  const data = records.map(r => ({
    ticker: r.ticker,
    spread: r.spreadCostBps,
    impact: r.marketImpactBps,
    timing: r.timingCostBps,
    commission: r.commissionBps,
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 5, right: 5, left: -15, bottom: 0 }}>
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

function CostTrendChart({ data }: { data: ReturnType<typeof generateCostTrend> }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 5, right: 5, left: -15, bottom: 0 }}>
        <XAxis dataKey="date" tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} interval={4} />
        <YAxis tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} />
        <Tooltip
          contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 10 }}
          labelStyle={{ color: "#8b949e" }}
        />
        <Area type="monotone" dataKey="totalBps" fill="#00d4aa" fillOpacity={0.08} stroke="#00d4aa" strokeWidth={1.5} name="Total Cost (bps)" />
        <Line type="monotone" dataKey="spreadBps" stroke="#58a6ff" strokeWidth={1} dot={false} name="Spread" />
        <Line type="monotone" dataKey="impactBps" stroke="#f0883e" strokeWidth={1} dot={false} name="Impact" />
        <Line type="monotone" dataKey="timingBps" stroke="#d2a8ff" strokeWidth={1} dot={false} name="Timing" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function VWAPSlippageChart({ records }: { records: TCARecord[] }) {
  const data = records.map(r => ({
    ticker: r.ticker,
    slippage: r.vwapSlippageBps,
  }));

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

function VenueAnalysis({ venues }: { venues: ReturnType<typeof generateVenueComparison> }) {
  return (
    <div className="text-[10px]">
      <table className="w-full">
        <thead>
          <tr className="text-terminal-text-faint border-b border-terminal-border">
            {["Venue", "Avg Cost (bps)", "Fill Count", "Avg Latency (ms)"].map(h => (
              <th key={h} className="py-1 px-2 text-left font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {venues.sort((a, b) => a.avgCost - b.avgCost).map(v => (
            <tr key={v.venue} className="border-b border-terminal-border/50 hover:bg-white/[0.02]">
              <td className="py-1.5 px-2 font-mono font-semibold text-terminal-text-primary">{v.venue}</td>
              <td className="py-1.5 px-2 font-mono">
                <div className="flex items-center gap-2">
                  <span>{v.avgCost.toFixed(2)}</span>
                  <div className="flex-1 h-1.5 bg-terminal-bg rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${(v.avgCost / 4) * 100}%`,
                        backgroundColor: v.avgCost < 2.8 ? "#3fb950" : v.avgCost < 3.2 ? "#d29922" : "#f85149",
                      }}
                    />
                  </div>
                </div>
              </td>
              <td className="py-1.5 px-2 font-mono">{v.fills}</td>
              <td className="py-1.5 px-2 font-mono">{v.avgLatency.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AlgoPerformance({ algos }: { algos: ReturnType<typeof generateAlgoComparison> }) {
  return (
    <div className="text-[10px]">
      <table className="w-full">
        <thead>
          <tr className="text-terminal-text-faint border-b border-terminal-border">
            {["Algorithm", "Avg Cost (bps)", "Avg IS ($)", "Trades"].map(h => (
              <th key={h} className="py-1 px-2 text-left font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {algos.sort((a, b) => a.avgCost - b.avgCost).map(a => (
            <tr key={a.algo} className="border-b border-terminal-border/50 hover:bg-white/[0.02]">
              <td className="py-1.5 px-2 font-mono font-semibold text-terminal-accent">{a.algo}</td>
              <td className="py-1.5 px-2 font-mono">
                <div className="flex items-center gap-2">
                  <span>{a.avgCost.toFixed(2)}</span>
                  <div className="flex-1 h-1.5 bg-terminal-bg rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${(a.avgCost / 4) * 100}%`,
                        backgroundColor: a.avgCost < 2.5 ? "#3fb950" : a.avgCost < 3.2 ? "#d29922" : "#f85149",
                      }}
                    />
                  </div>
                </div>
              </td>
              <td className="py-1.5 px-2 font-mono">${a.avgIS}</td>
              <td className="py-1.5 px-2 font-mono">{a.trades}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ISDistribution({ records }: { records: TCARecord[] }) {
  // Bucket IS values
  const buckets = [
    { range: "< -$500", count: 0 },
    { range: "-$500 to -$100", count: 0 },
    { range: "-$100 to $0", count: 0 },
    { range: "$0 to $100", count: 0 },
    { range: "$100 to $500", count: 0 },
    { range: "> $500", count: 0 },
  ];
  records.forEach(r => {
    const v = r.implementationShortfallUsd;
    if (v < -500) buckets[0].count++;
    else if (v < -100) buckets[1].count++;
    else if (v < 0) buckets[2].count++;
    else if (v < 100) buckets[3].count++;
    else if (v < 500) buckets[4].count++;
    else buckets[5].count++;
  });

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={buckets} margin={{ top: 5, right: 5, left: -15, bottom: 0 }}>
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

function PerAssetTable({ records }: { records: TCARecord[] }) {
  const [sortKey, setSortKey] = useState<"totalCostBps" | "implementationShortfallUsd" | "vwapSlippageBps">("totalCostBps");
  const sorted = useMemo(() => [...records].sort((a, b) => b[sortKey] - a[sortKey]), [records, sortKey]);

  return (
    <div className="text-[10px]">
      <div className="flex items-center gap-2 px-2 py-1 mb-1">
        <span className="text-terminal-text-faint">SORT BY:</span>
        {([
          ["totalCostBps", "Total Cost"],
          ["implementationShortfallUsd", "Impl. Shortfall"],
          ["vwapSlippageBps", "VWAP Slip"],
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
          {sorted.map(r => (
            <tr key={r.orderId} className="border-b border-terminal-border/50 hover:bg-white/[0.02]">
              <td className="py-1 px-1 font-mono font-semibold text-terminal-text-primary">{r.ticker}</td>
              <td className={`py-1 px-1 font-mono font-semibold ${r.side === "BUY" ? "text-terminal-positive" : "text-terminal-negative"}`}>{r.side}</td>
              <td className="py-1 px-1 font-mono">{r.qty.toLocaleString()}</td>
              <td className="py-1 px-1 font-mono">{r.avgFillPrice.toFixed(2)}</td>
              <td className="py-1 px-1 font-mono">{r.arrivalPrice.toFixed(2)}</td>
              <td className="py-1 px-1 font-mono">{r.vwapPrice.toFixed(2)}</td>
              <td className="py-1 px-1 font-mono">{r.spreadCostBps.toFixed(1)}</td>
              <td className="py-1 px-1 font-mono text-terminal-warning">{r.marketImpactBps.toFixed(1)}</td>
              <td className={`py-1 px-1 font-mono ${r.timingCostBps > 0 ? "text-terminal-negative" : "text-terminal-positive"}`}>{r.timingCostBps.toFixed(1)}</td>
              <td className="py-1 px-1 font-mono text-terminal-text-muted">{r.commissionBps.toFixed(1)}</td>
              <td className="py-1 px-1 font-mono font-semibold text-terminal-accent">{r.totalCostBps.toFixed(1)}</td>
              <td className={`py-1 px-1 font-mono ${r.implementationShortfallUsd > 0 ? "text-terminal-negative" : "text-terminal-positive"}`}>
                {r.implementationShortfallUsd > 0 ? "+" : ""}${Math.abs(r.implementationShortfallUsd).toFixed(0)}
              </td>
              <td className={`py-1 px-1 font-mono ${r.vwapSlippageBps > 0 ? "text-terminal-negative" : "text-terminal-positive"}`}>
                {r.vwapSlippageBps > 0 ? "+" : ""}{r.vwapSlippageBps.toFixed(1)}
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

function SectorBreakdown({ records }: { records: TCARecord[] }) {
  const sectorMap = new Map<string, { totalCost: number; count: number; totalIS: number }>();
  records.forEach(r => {
    const prev = sectorMap.get(r.sector) || { totalCost: 0, count: 0, totalIS: 0 };
    sectorMap.set(r.sector, {
      totalCost: prev.totalCost + r.totalCostBps,
      count: prev.count + 1,
      totalIS: prev.totalIS + Math.abs(r.implementationShortfallUsd),
    });
  });

  const data = Array.from(sectorMap.entries())
    .map(([sector, v]) => ({
      sector,
      avgCost: +(v.totalCost / v.count).toFixed(2),
      trades: v.count,
      totalIS: +v.totalIS.toFixed(0),
    }))
    .sort((a, b) => b.avgCost - a.avgCost);

  const colors = ["#00d4aa", "#f0883e", "#58a6ff", "#d2a8ff", "#f85149", "#3fb950", "#d29922"];

  return (
    <div className="text-[10px] space-y-1">
      {data.map((d, i) => (
        <div key={d.sector} className="flex items-center gap-2 px-1 py-1 rounded hover:bg-white/[0.02]">
          <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: colors[i % colors.length] }} />
          <span className="text-terminal-text-muted w-28 truncate">{d.sector}</span>
          <div className="flex-1 h-1.5 bg-terminal-bg rounded-full overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{ width: `${(d.avgCost / 6) * 100}%`, backgroundColor: colors[i % colors.length] }}
            />
          </div>
          <span className="font-mono text-terminal-text-primary w-14 text-right">{d.avgCost} bps</span>
          <span className="font-mono text-terminal-text-faint w-8 text-right">{d.trades}</span>
        </div>
      ))}
    </div>
  );
}

// ═══════════ MAIN PAGE ═══════════

export default function TCAPage() {
  const [records] = useState(generateTCARecords);
  const [costTrend] = useState(generateCostTrend);
  const [venues] = useState(generateVenueComparison);
  const [algos] = useState(generateAlgoComparison);
  const [activeView, setActiveView] = useState<"overview" | "detail">("overview");

  return (
    <div className="h-full flex flex-col gap-1 p-1 overflow-hidden">
      {/* Summary Cards Row */}
      <div className="flex-shrink-0">
        <SummaryCards records={records} />
      </div>

      {/* Tab Toggle */}
      <div className="flex items-center gap-1 flex-shrink-0 px-1">
        {(["overview", "detail"] as const).map(v => (
          <button
            key={v}
            onClick={() => setActiveView(v)}
            className={`px-3 py-1 rounded text-[10px] font-medium tracking-wider transition-colors ${
              activeView === v
                ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30"
                : "text-terminal-text-muted hover:text-terminal-text-primary hover:bg-white/[0.03] border border-transparent"
            }`}
          >{v === "overview" ? "OVERVIEW" : "PER-ASSET DETAIL"}</button>
        ))}
      </div>

      {activeView === "overview" ? (
        <div className="flex-1 grid grid-cols-[1fr_1fr_280px] grid-rows-2 gap-1 overflow-hidden">
          {/* Row 1 */}
          <DashboardPanel title="COST DECOMPOSITION BY ASSET" className="h-full">
            <CostDecomposition records={records} />
          </DashboardPanel>
          <DashboardPanel title="30-DAY COST TREND" className="h-full">
            <CostTrendChart data={costTrend} />
          </DashboardPanel>
          <DashboardPanel title="SECTOR COSTS" className="row-span-2">
            <SectorBreakdown records={records} />
          </DashboardPanel>

          {/* Row 2 */}
          <DashboardPanel title="VWAP SLIPPAGE BY ASSET" className="h-full">
            <VWAPSlippageChart records={records} />
          </DashboardPanel>
          <DashboardPanel title="IMPL. SHORTFALL DISTRIBUTION" className="h-full">
            <ISDistribution records={records} />
          </DashboardPanel>
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-[1fr_280px] gap-1 overflow-hidden">
          <DashboardPanel title="PER-ASSET TRANSACTION COST ANALYSIS" className="h-full" noPadding>
            <PerAssetTable records={records} />
          </DashboardPanel>
          <div className="flex flex-col gap-1">
            <DashboardPanel title="VENUE ANALYSIS" className="flex-1">
              <VenueAnalysis venues={venues} />
            </DashboardPanel>
            <DashboardPanel title="ALGO PERFORMANCE" className="flex-1">
              <AlgoPerformance algos={algos} />
            </DashboardPanel>
          </div>
        </div>
      )}
    </div>
  );
}
