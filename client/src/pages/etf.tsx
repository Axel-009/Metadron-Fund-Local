import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
} from "recharts";
import { useEngineQuery } from "@/hooks/use-engine-api";

// ═══════════ TYPES ═══════════

interface EtfHolding {
  ticker: string;
  name: string;
  category: string;
  quantity: number;
  avg_cost: number;
  current_price: number;
  change_pct: number;
  unrealized_pnl: number;
  market_value: number;
  weight: number;
}

interface SectorEntry {
  ticker: string;
  name: string;
  price: number;
  change_pct: number;
}

interface FlowEntry {
  ticker: string;
  name: string;
  flow: number;
  direction: string;
  weekly_return: number;
  monthly_return: number;
}

interface MoverEntry {
  ticker: string;
  name: string;
  category: string;
  price: number;
  change_pct: number;
}

interface CategoryEntry {
  category: string;
  count: number;
  market_value: number;
  pnl: number;
  weight: number;
}

interface EtfSummary {
  etf_positions: number;
  total_etfs_tracked: number;
  etf_market_value: number;
  etf_unrealized_pnl: number;
  etf_weight_of_portfolio: number;
  categories_active: number;
  portfolio_nav: number;
}

// ═══════════ CONSTANTS ═══════════

const CATEGORY_COLORS: Record<string, string> = {
  Equity:        "#00d4aa",
  Bond:          "#3b82f6",
  Commodity:     "#d29922",
  Sector:        "#a855f7",
  Thematic:      "#f85149",
  Factor:        "#22d3ee",
  Volatility:    "#ec4899",
  International: "#84cc16",
  Other:         "#6b7280",
};

const TOOLTIP_STYLE = {
  backgroundColor: "#0d1117",
  border: "1px solid #21262d",
  borderRadius: 4,
  fontSize: 10,
  color: "#e6edf3",
  padding: "4px 8px",
};

const fmtN = (n: number, d = 2) =>
  n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

const fmtDollar = (n: number) => {
  if (Math.abs(n) >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (Math.abs(n) >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
};

// ═══════════ COMPONENT ═══════════

export default function ETFDashboard() {
  // ─── Live API hooks (all from /etf router) ─────────────────
  const { data: holdingsApi } = useEngineQuery<{
    holdings: EtfHolding[];
    total_positions: number;
    total_market_value: number;
  }>("/etf/holdings", { refetchInterval: 15000 });

  const { data: sectorApi } = useEngineQuery<{
    sectors: SectorEntry[];
  }>("/etf/sector-heatmap", { refetchInterval: 30000 });

  const { data: flowsApi } = useEngineQuery<{
    flows: FlowEntry[];
  }>("/etf/flows", { refetchInterval: 60000 });

  const { data: moversApi } = useEngineQuery<{
    top_movers: MoverEntry[];
    bottom_movers: MoverEntry[];
  }>("/etf/movers", { refetchInterval: 30000 });

  const { data: catApi } = useEngineQuery<{
    categories: CategoryEntry[];
  }>("/etf/categories", { refetchInterval: 30000 });

  const { data: summaryApi } = useEngineQuery<EtfSummary>(
    "/etf/summary", { refetchInterval: 15000 },
  );

  // ─── Derived state ────────────────────────────────────────
  const holdings = holdingsApi?.holdings ?? [];
  const sectors = sectorApi?.sectors ?? [];
  const flows = flowsApi?.flows ?? [];
  const topMovers = moversApi?.top_movers ?? [];
  const botMovers = moversApi?.bottom_movers ?? [];
  const categories = catApi?.categories ?? [];
  const summary = summaryApi ?? null;

  const [sortKey, setSortKey] = useState<string>("weight");
  const [sortDir, setSortDir] = useState<1 | -1>(-1);

  const sorted = useMemo(() => {
    return [...holdings].sort((a, b) => {
      const av = (a as unknown as Record<string, number | string>)[sortKey];
      const bv = (b as unknown as Record<string, number | string>)[sortKey];
      if (typeof av === "number" && typeof bv === "number") return sortDir * (av - bv);
      return sortDir * String(av).localeCompare(String(bv));
    });
  }, [holdings, sortKey, sortDir]);

  const toggleSort = (k: string) => {
    if (k === sortKey) setSortDir(d => (d === 1 ? -1 : 1));
    else { setSortKey(k); setSortDir(-1); }
  };

  // Category pie data
  const pieData = useMemo(() => {
    if (categories.length > 0) {
      return categories.map(c => ({ name: c.category, value: c.weight }));
    }
    // Derive from holdings
    const map: Record<string, number> = {};
    holdings.forEach(h => { map[h.category] = (map[h.category] || 0) + h.weight; });
    return Object.entries(map).map(([name, value]) => ({ name, value: +value.toFixed(1) }));
  }, [categories, holdings]);

  const colHdr = (k: string, label: string) => (
    <th
      className="text-terminal-text-muted text-[9px] tracking-widest cursor-pointer hover:text-terminal-accent transition-colors whitespace-nowrap px-1 py-1 text-right"
      onClick={() => toggleSort(k)}
    >
      {label}{sortKey === k ? (sortDir === 1 ? "▲" : "▼") : ""}
    </th>
  );

  const loading = !holdingsApi;

  return (
    <div className="h-full grid gap-1 p-1 overflow-hidden" style={{ gridTemplateColumns: "1fr 200px", gridTemplateRows: "auto 1fr 190px" }}>

      {/* ─── Summary Strip (top, full width) ─── */}
      <div className="col-span-2">
        <DashboardPanel title="ETF PORTFOLIO OVERVIEW" noPadding>
          <div className="flex items-center gap-4 px-3 py-1.5 overflow-x-auto">
            {[
              { label: "POSITIONS", value: summary?.etf_positions ?? holdings.length, fmt: (v: number) => String(v) },
              { label: "TRACKED", value: summary?.total_etfs_tracked ?? 0, fmt: (v: number) => String(v) },
              { label: "MKT VALUE", value: summary?.etf_market_value ?? holdingsApi?.total_market_value ?? 0, fmt: fmtDollar },
              { label: "UNRLZD P&L", value: summary?.etf_unrealized_pnl ?? 0, fmt: fmtDollar, colored: true },
              { label: "PORT WT%", value: summary?.etf_weight_of_portfolio ?? 0, fmt: (v: number) => `${v.toFixed(1)}%` },
              { label: "CATEGORIES", value: summary?.categories_active ?? categories.length, fmt: (v: number) => String(v) },
            ].map(s => (
              <div key={s.label} className="flex flex-col items-center min-w-[70px]">
                <span className="text-[8px] text-terminal-text-muted tracking-widest">{s.label}</span>
                <span className={`text-[13px] font-mono font-bold ${
                  s.colored ? (s.value >= 0 ? "text-terminal-positive" : "text-terminal-negative") : "text-terminal-text-primary"
                }`}>
                  {loading ? "—" : s.fmt(s.value)}
                </span>
              </div>
            ))}
          </div>
        </DashboardPanel>
      </div>

      {/* ─── Main Holdings Table ─── */}
      <DashboardPanel
        title="ETF HOLDINGS"
        noPadding
        headerRight={<span className="text-[9px] text-terminal-text-muted">{holdings.length} positions</span>}
      >
        {loading ? (
          <div className="flex items-center justify-center h-full text-terminal-text-muted text-xs">Loading ETF data...</div>
        ) : holdings.length === 0 ? (
          <div className="flex items-center justify-center h-full text-terminal-text-muted text-xs">No ETF positions — tracking {summary?.total_etfs_tracked ?? 0} ETFs</div>
        ) : (
          <div className="overflow-auto h-full">
            <table className="w-full text-[10px] font-mono border-collapse">
              <thead className="sticky top-0 bg-terminal-surface z-10">
                <tr className="border-b border-terminal-border">
                  <th className="text-terminal-text-muted text-[9px] tracking-widest text-left px-2 py-1">TICKER</th>
                  <th className="text-terminal-text-muted text-[9px] tracking-widest text-left px-1 py-1">NAME</th>
                  <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1">CAT</th>
                  {colHdr("quantity", "QTY")}
                  {colHdr("avg_cost", "AVG")}
                  {colHdr("current_price", "LAST")}
                  {colHdr("change_pct", "CHG%")}
                  {colHdr("unrealized_pnl", "UNRLZD P&L")}
                  {colHdr("market_value", "MKT VAL")}
                  {colHdr("weight", "WT%")}
                </tr>
              </thead>
              <tbody>
                {sorted.map((h, i) => (
                  <tr key={h.ticker} className={`border-b border-terminal-border hover:bg-[#161b22] transition-colors ${i % 2 === 0 ? "" : "bg-[#0d1117]"}`}>
                    <td className="px-2 py-0.5 text-terminal-accent font-bold">{h.ticker}</td>
                    <td className="px-1 py-0.5 text-terminal-text-primary max-w-[180px] truncate">{h.name}</td>
                    <td className="px-1 py-0.5 text-right">
                      <span className="px-1 rounded text-[8px]" style={{ backgroundColor: (CATEGORY_COLORS[h.category] ?? "#6b7280") + "22", color: CATEGORY_COLORS[h.category] ?? "#6b7280" }}>
                        {h.category}
                      </span>
                    </td>
                    <td className="px-1 py-0.5 text-terminal-text-primary text-right">{h.quantity.toLocaleString()}</td>
                    <td className="px-1 py-0.5 text-terminal-text-muted text-right">{fmtN(h.avg_cost)}</td>
                    <td className="px-1 py-0.5 text-terminal-text-primary text-right">{fmtN(h.current_price)}</td>
                    <td className={`px-1 py-0.5 text-right font-bold ${h.change_pct >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                      {h.change_pct >= 0 ? "+" : ""}{h.change_pct.toFixed(2)}%
                    </td>
                    <td className={`px-1 py-0.5 text-right ${h.unrealized_pnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                      {h.unrealized_pnl >= 0 ? "+$" : "-$"}{Math.abs(h.unrealized_pnl).toLocaleString()}
                    </td>
                    <td className="px-1 py-0.5 text-terminal-text-primary text-right">{fmtDollar(h.market_value)}</td>
                    <td className="px-1 py-0.5 text-terminal-text-primary text-right">{h.weight.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </DashboardPanel>

      {/* ─── Right Column (spans rows 2+3) ─── */}
      <div className="row-span-2 flex flex-col gap-1">
        {/* Category Allocation Donut */}
        <DashboardPanel title="CATEGORY ALLOC" className="flex-none" style={{ height: "230px" }}>
          {pieData.length === 0 ? (
            <div className="flex items-center justify-center h-full text-terminal-text-muted text-[10px]">No data</div>
          ) : (
            <>
              <ResponsiveContainer width="100%" height={120}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" innerRadius={30} outerRadius={52} paddingAngle={2} startAngle={90} endAngle={450}>
                    {pieData.map(entry => (
                      <Cell key={entry.name} fill={CATEGORY_COLORS[entry.name] ?? "#6b7280"} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(1)}%`, ""]} />
                </PieChart>
              </ResponsiveContainer>
              <div className="space-y-0.5">
                {pieData.map(c => (
                  <div key={c.name} className="flex items-center justify-between">
                    <div className="flex items-center gap-1">
                      <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: CATEGORY_COLORS[c.name] ?? "#6b7280" }} />
                      <span className="text-[9px] text-terminal-text-muted">{c.name}</span>
                    </div>
                    <span className="text-[10px] font-mono text-terminal-text-primary">{c.value.toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </DashboardPanel>

        {/* ETF Flow Monitor */}
        <DashboardPanel title="ETF FLOW MONITOR" className="flex-1">
          {flows.length === 0 ? (
            <div className="flex items-center justify-center h-full text-terminal-text-muted text-[10px]">Loading flows...</div>
          ) : (
            <div className="space-y-1 overflow-y-auto">
              {flows.slice(0, 14).map(f => {
                const maxFlow = Math.max(...flows.map(fl => fl.flow), 1);
                return (
                  <div key={f.ticker} className="flex items-center gap-1">
                    <span className="text-[10px] font-mono text-terminal-accent w-9 flex-shrink-0">{f.ticker}</span>
                    <div className="flex-1 relative h-2.5 rounded overflow-hidden bg-[#161b22]">
                      <div
                        className="absolute top-0 h-full rounded"
                        style={{
                          width: `${Math.min(100, (f.flow / maxFlow) * 100)}%`,
                          backgroundColor: f.direction === "in" ? "#3fb950" : "#f85149",
                          left: f.direction === "in" ? "0" : "auto",
                          right: f.direction === "out" ? "0" : "auto",
                        }}
                      />
                    </div>
                    <span className={`text-[9px] font-mono w-16 text-right flex-shrink-0 ${f.direction === "in" ? "text-terminal-positive" : "text-terminal-negative"}`}>
                      {f.direction === "in" ? "+" : "-"}{f.weekly_return.toFixed(2)}%
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </DashboardPanel>
      </div>

      {/* ─── Bottom Row: Sector Heatmap + Movers ─── */}
      <div className="grid gap-1" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {/* Sector Heatmap */}
        <DashboardPanel title="SECTOR ETF HEATMAP" noPadding>
          {sectors.length === 0 ? (
            <div className="flex items-center justify-center h-full text-terminal-text-muted text-[10px]">Loading sectors...</div>
          ) : (
            <div className="grid grid-cols-4 gap-0.5 p-1.5 h-full">
              {sectors.map(s => {
                const intensity = Math.min(Math.abs(s.change_pct) / 2, 1);
                const bg = s.change_pct >= 0
                  ? `rgba(63,185,80,${0.10 + intensity * 0.35})`
                  : `rgba(248,81,73,${0.10 + intensity * 0.35})`;
                const col = s.change_pct >= 0 ? "#3fb950" : "#f85149";
                return (
                  <div key={s.ticker} className="rounded flex flex-col items-center justify-center p-1 cursor-default" style={{ backgroundColor: bg, border: `1px solid ${col}33` }}>
                    <span className="text-[9px] font-mono font-bold" style={{ color: col }}>{s.ticker}</span>
                    <span className="text-[8px] text-terminal-text-muted leading-tight">{s.name}</span>
                    <span className="text-[9px] font-mono font-bold" style={{ color: col }}>{s.change_pct >= 0 ? "+" : ""}{s.change_pct.toFixed(2)}%</span>
                  </div>
                );
              })}
            </div>
          )}
        </DashboardPanel>

        {/* Top/Bottom Movers */}
        <div className="grid gap-1" style={{ gridTemplateRows: "1fr 1fr" }}>
          <DashboardPanel title="TOP MOVERS TODAY" noPadding>
            {topMovers.length === 0 ? (
              <div className="flex items-center justify-center h-full text-terminal-text-muted text-[10px]">Loading...</div>
            ) : (
              <table className="w-full text-[10px] font-mono">
                <tbody>
                  {topMovers.map(m => (
                    <tr key={m.ticker} className="border-b border-terminal-border">
                      <td className="px-2 py-0.5 text-terminal-accent font-bold">{m.ticker}</td>
                      <td className="px-1 py-0.5 text-terminal-text-muted text-[9px] truncate">{m.name}</td>
                      <td className="px-1 py-0.5 text-terminal-text-primary text-right">${m.price.toFixed(2)}</td>
                      <td className="px-2 py-0.5 text-terminal-positive text-right font-bold">+{m.change_pct.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </DashboardPanel>
          <DashboardPanel title="BOTTOM MOVERS TODAY" noPadding>
            {botMovers.length === 0 ? (
              <div className="flex items-center justify-center h-full text-terminal-text-muted text-[10px]">Loading...</div>
            ) : (
              <table className="w-full text-[10px] font-mono">
                <tbody>
                  {botMovers.map(m => (
                    <tr key={m.ticker} className="border-b border-terminal-border">
                      <td className="px-2 py-0.5 text-terminal-accent font-bold">{m.ticker}</td>
                      <td className="px-1 py-0.5 text-terminal-text-muted text-[9px] truncate">{m.name}</td>
                      <td className="px-1 py-0.5 text-terminal-text-primary text-right">${m.price.toFixed(2)}</td>
                      <td className="px-2 py-0.5 text-terminal-negative text-right font-bold">{m.change_pct.toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </DashboardPanel>
        </div>
      </div>
    </div>
  );
}
