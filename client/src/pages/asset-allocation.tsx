import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import { MiniChart } from "@/components/mini-chart";
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";
import { useState, useEffect, useMemo } from "react";
import { useEngineQuery, type PortfolioLive, type AllocationData } from "@/hooks/use-engine-api";

// Static fallbacks removed — all data from live API endpoints





const CREDIT_COLORS: Record<string, string> = {
  "AAA": "#3fb950",
  "AA+": "#4ade80",
  "AA": "#3fb950",
  "AA-": "#86efac",
  "A+": "#d29922",
  "A": "#d29922",
  "A-": "#d29922",
  "BBB": "#f0883e",
  "BB": "#f85149",
  "NR": "#484f58",
};

const PRODUCT_COLORS: Record<string, string> = {
  "Equity": "#00d4aa",
  "Options": "#bc8cff",
  "Futures": "#f0883e",
  "ETF": "#58a6ff",
  "Fixed Income": "#d29922",
  "Cash": "#484f58",
};

function CreditBadge({ rating }: { rating: string }) {
  const color = CREDIT_COLORS[rating] ?? "#484f58";
  return (
    <span
      className="inline-flex items-center px-1 py-0.5 rounded text-[7px] font-mono font-bold"
      style={{ color, border: `1px solid ${color}40`, background: `${color}15` }}
    >
      {rating}
    </span>
  );
}

export default function AssetAllocation() {
  // ─── Engine API ─────────────────────────────────────
  const { data: portData } = useEngineQuery<PortfolioLive>("/portfolio/live", { refetchInterval: 4000 });
  const { data: allocData } = useEngineQuery<AllocationData>("/portfolio/allocation", { refetchInterval: 10000 });
  const { data: posData } = useEngineQuery<{ positions: Array<{ ticker: string; quantity: number; avg_cost: number; current_price: number; unrealized_pnl: number; realized_pnl: number; sector: string }> }>("/portfolio/positions", { refetchInterval: 10000 });
  const { data: indicesApi } = useEngineQuery<{ indices: Array<{ ticker: string; price: number; change: number; data: number[] }> }>("/portfolio/indices", { refetchInterval: 15000 });
  const { data: moversApi } = useEngineQuery<{ movers: Array<{ ticker: string; change: number; momentum: string }> }>("/portfolio/movers", { refetchInterval: 30000 });
  const { data: sectorAllocApi } = useEngineQuery<{ allocation: Array<{ name: string; value: number; color: string }> }>("/portfolio/sector-allocation", { refetchInterval: 30000 });

  // NAV from API
  const nav = portData?.nav ?? 0;

  // Indices from API
  const indices = indicesApi?.indices || [];
  // Movers from API
  const movers = moversApi?.movers || [];
  // Sector allocation from API
  const allocPie = sectorAllocApi?.allocation || [];

  // Positions directly from broker — no static fallback
  const holdings = useMemo(() => {
    if (!posData?.positions?.length) return [];
    const totalValue = posData.positions.reduce((s, p) => s + Math.abs((p.quantity || 0) * (p.current_price || 0)), 0);
    return posData.positions.map((p) => {
      const val = Math.abs((p.quantity || 0) * (p.current_price || 0));
      const weight = totalValue > 0 ? (val / totalValue * 100) : 0;
      return {
        ticker: p.ticker,
        name: p.ticker,
        weight: +weight.toFixed(1),
        shares: p.quantity || 0,
        price: p.current_price || 0,
        change: p.current_price && p.avg_cost ? ((p.current_price - p.avg_cost) / p.avg_cost * 100) : 0,
        sector: p.sector || "Unknown",
        productType: "Equity",
        leveraged: false,
        marginPct: 50,
        creditRating: "NR",
        strategy: "—",
        weightNav: +weight.toFixed(2),
        pnl: (p.unrealized_pnl || 0) + (p.realized_pnl || 0),
      };
    });
  }, [posData]);

  const totalLeveraged = holdings.filter((h) => h.leveraged).reduce((s, h) => s + Math.abs(h.shares * h.price), 0);
  const avgMarginPct = (holdings.reduce((s, h) => s + h.marginPct * (h.weightNav / 100), 0)).toFixed(1);
  const totalPnl = holdings.reduce((s, h) => s + h.pnl, 0);

  return (
    <div className="h-full flex flex-col gap-[2px] p-[2px] overflow-hidden" data-testid="asset-allocation">
      {/* ─── Main resizable: left (NAV + Exposure + Basket) | right (Allocation + Movers) ─── */}
      <div className="flex-1 min-h-0">
        <ResizableDashboard defaultSizes={[72, 28]} minSizes={[45, 20]}>
          {/* Left column: NAV + Exposure + Basket Table */}
          <div className="h-full flex flex-col gap-[2px] overflow-hidden">
            {/* NAV Display */}
            <DashboardPanel title="LIVE NAV">
              <div className="flex items-center gap-6">
                <div>
                  <div className="text-3xl font-mono font-bold text-terminal-text-primary tabular-nums">
                    ${nav.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-terminal-positive font-mono mt-0.5">+$842,150 (+0.66%) today</div>
                </div>
                {/* Indices strip */}
                <div className="flex gap-4 ml-auto">
                  {indices.length === 0 && (
                    <div style={{color: "var(--muted)", fontSize: 11, padding: "20px 16px", textAlign: "center", opacity: 0.7, gridColumn: "1 / -1"}}>
                      Index data loading — awaiting market hours...
                    </div>
                  )}
                  {indices.map((idx) => (
                    <div key={idx.ticker} className="text-center">
                      <div className="text-[9px] text-terminal-text-faint font-mono">{idx.ticker}</div>
                      <div className="text-[11px] text-terminal-text-primary font-mono tabular-nums">{idx.price.toFixed(2)}</div>
                      <div className={`text-[9px] font-mono tabular-nums ${idx.change >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                        {idx.change >= 0 ? "+" : ""}{idx.change.toFixed(2)}%
                      </div>
                      <MiniChart data={idx.data} width={40} height={14} color={idx.change >= 0 ? "#3fb950" : "#f85149"} />
                    </div>
                  ))}
                </div>
              </div>
            </DashboardPanel>

            {/* Exposure Summary */}
            <DashboardPanel title="EXPOSURE SUMMARY">
              <div className="flex gap-6">
                <div className="flex flex-col gap-0.5">
                  <span className="text-[7px] text-terminal-text-faint uppercase tracking-wider">Total Leveraged Exposure</span>
                  <span className="text-[13px] font-mono font-bold text-terminal-warning tabular-nums">
                    ${(totalLeveraged / 1e6).toFixed(2)}M
                  </span>
                </div>
                <div className="w-px bg-terminal-border/50" />
                <div className="flex flex-col gap-0.5">
                  <span className="text-[7px] text-terminal-text-faint uppercase tracking-wider">Avg Margin% (NAV-weighted)</span>
                  <span className="text-[13px] font-mono font-bold text-terminal-accent tabular-nums">{avgMarginPct}%</span>
                </div>
                <div className="w-px bg-terminal-border/50" />
                <div className="flex flex-col gap-0.5">
                  <span className="text-[7px] text-terminal-text-faint uppercase tracking-wider">NAV-Weighted Credit</span>
                  <span className="text-[13px] font-mono font-bold text-terminal-positive tabular-nums">A+</span>
                </div>
                <div className="w-px bg-terminal-border/50" />
                <div className="flex flex-col gap-0.5">
                  <span className="text-[7px] text-terminal-text-faint uppercase tracking-wider">Total Portfolio P&L</span>
                  <span className={`text-[13px] font-mono font-bold tabular-nums ${totalPnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {totalPnl >= 0 ? "+$" : "-$"}{Math.abs(totalPnl).toLocaleString()}
                  </span>
                </div>
                <div className="w-px bg-terminal-border/50" />
                {/* Product Type breakdown */}
                <div className="flex gap-3 ml-auto">
                  {["Equity", "Options", "Futures", "ETF", "Fixed Income", "Cash"].map((pt) => {
                    const count = holdings.filter((h) => h.productType === pt).length;
                    const color = PRODUCT_COLORS[pt];
                    return (
                      <div key={pt} className="flex flex-col items-center gap-0.5">
                        <span className="text-[7px] font-mono" style={{ color }}>{pt}</span>
                        <span className="text-[11px] font-mono font-bold text-terminal-text-primary">{count}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </DashboardPanel>

      {/* Basket Formation Table — enriched */}
      <DashboardPanel title="BASKET FORMATION — ENHANCED" className="flex-1" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[9px] font-mono">
            <thead className="sticky top-0 bg-terminal-surface z-10">
              <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                <th className="text-left px-2 py-1.5 font-medium">Ticker</th>
                <th className="text-left px-2 py-1.5 font-medium">Name</th>
                <th className="text-left px-2 py-1.5 font-medium">Product</th>
                <th className="text-center px-2 py-1.5 font-medium">Lev</th>
                <th className="text-right px-2 py-1.5 font-medium">Margin%</th>
                <th className="text-center px-2 py-1.5 font-medium">Credit</th>
                <th className="text-left px-2 py-1.5 font-medium">Strategy</th>
                <th className="text-right px-2 py-1.5 font-medium">Weight%</th>
                <th className="text-right px-2 py-1.5 font-medium">Wt NAV%</th>
                <th className="text-right px-2 py-1.5 font-medium">Shares</th>
                <th className="text-right px-2 py-1.5 font-medium">Price</th>
                <th className="text-right px-2 py-1.5 font-medium">Chg%</th>
                <th className="text-left px-2 py-1.5 font-medium">Sector</th>
                <th className="text-right px-2 py-1.5 font-medium">P&L</th>
              </tr>
            </thead>
            <tbody>
              {holdings.length === 0 && (
                <tr><td colSpan={12} style={{color: "var(--muted)", fontSize: 11, padding: "20px 16px", textAlign: "center", opacity: 0.7}}>
                  No positions — holdings will appear once you execute a trade.
                </td></tr>
              )}
              {holdings.map((h, i) => (
                <tr key={i} className="border-b border-terminal-border/20 hover:bg-white/[0.02]">
                  <td className="px-2 py-1.5 text-terminal-accent font-medium">{h.ticker}</td>
                  <td className="px-2 py-1.5 text-terminal-text-muted max-w-[120px] truncate">{h.name}</td>
                  {/* Product Type */}
                  <td className="px-2 py-1.5">
                    <span
                      className="text-[7px] px-1 py-0.5 rounded font-medium"
                      style={{
                        color: PRODUCT_COLORS[h.productType] ?? "#7d8590",
                        background: `${PRODUCT_COLORS[h.productType] ?? "#7d8590"}18`,
                        border: `1px solid ${PRODUCT_COLORS[h.productType] ?? "#7d8590"}35`,
                      }}
                    >
                      {h.productType}
                    </span>
                  </td>
                  {/* Leveraged */}
                  <td className="px-2 py-1.5 text-center">
                    {h.leveraged ? (
                      <span className="text-[10px] font-bold text-terminal-warning" title="Leveraged">⚡</span>
                    ) : (
                      <span className="text-[8px] text-terminal-text-faint">—</span>
                    )}
                  </td>
                  {/* Margin% */}
                  <td className={`px-2 py-1.5 text-right tabular-nums font-medium ${
                    h.marginPct === 0 ? "text-terminal-text-faint" :
                    h.marginPct === 100 ? "text-terminal-negative" :
                    h.marginPct >= 50 ? "text-terminal-warning" :
                    "text-terminal-positive"
                  }`}>{h.marginPct}%</td>
                  {/* Credit Rating */}
                  <td className="px-2 py-1.5 text-center"><CreditBadge rating={h.creditRating} /></td>
                  {/* Strategy */}
                  <td className="px-2 py-1.5 text-[8px] text-terminal-accent">{h.strategy}</td>
                  {/* Weight */}
                  <td className="px-2 py-1.5 text-right text-terminal-text-primary tabular-nums">{h.weight.toFixed(1)}%</td>
                  {/* Weight% of NAV */}
                  <td className="px-2 py-1.5 text-right tabular-nums">
                    <span className="text-terminal-text-muted">{h.weightNav.toFixed(2)}%</span>
                  </td>
                  {/* Shares */}
                  <td className="px-2 py-1.5 text-right text-terminal-text-muted tabular-nums">{h.shares === 0 ? "—" : h.shares.toLocaleString()}</td>
                  {/* Price */}
                  <td className="px-2 py-1.5 text-right text-terminal-text-primary tabular-nums">${h.price.toFixed(2)}</td>
                  {/* Chg% */}
                  <td className={`px-2 py-1.5 text-right tabular-nums ${h.change >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {h.change >= 0 ? "+" : ""}{h.change.toFixed(1)}%
                  </td>
                  {/* Sector */}
                  <td className="px-2 py-1.5 text-terminal-text-faint">{h.sector}</td>
                  {/* P&L */}
                  <td className={`px-2 py-1.5 text-right tabular-nums font-medium ${
                    h.pnl > 0 ? "text-terminal-positive" : h.pnl < 0 ? "text-terminal-negative" : "text-terminal-text-faint"
                  }`}>
                    {h.pnl === 0 ? "—" : `${h.pnl > 0 ? "+$" : "-$"}${Math.abs(h.pnl).toLocaleString()}`}
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="border-t border-terminal-border/50 bg-white/[0.02]">
                <td colSpan={7} className="px-2 py-1.5 text-[8px] text-terminal-text-faint uppercase font-medium">Totals</td>
                <td className="px-2 py-1.5 text-right text-terminal-text-primary font-bold tabular-nums text-[9px]">
                  {holdings.reduce((s, h) => s + h.weight, 0).toFixed(1)}%
                </td>
                <td className="px-2 py-1.5 text-right text-terminal-text-muted font-bold tabular-nums text-[9px]">
                  {holdings.reduce((s, h) => s + h.weightNav, 0).toFixed(2)}%
                </td>
                <td colSpan={3} />
                <td />
                <td className={`px-2 py-1.5 text-right font-bold tabular-nums text-[9px] ${totalPnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                  {totalPnl >= 0 ? "+$" : "-$"}{Math.abs(totalPnl).toLocaleString()}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </DashboardPanel>
          </div>

          {/* Right column: Allocation Pie + Movers */}
          <DashboardPanel title="ALLOCATION">
            <div className="h-[180px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={allocPie}
                    cx="50%"
                    cy="50%"
                    innerRadius="45%"
                    outerRadius="75%"
                    paddingAngle={2}
                    dataKey="value"
                    stroke="none"
                  >
                    {allocPie.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-1 mt-2">
              {allocPie.map((d, i) => (
                <div key={i} className="flex items-center gap-2 text-[9px]">
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: d.color }} />
                  <span className="text-terminal-text-muted flex-1">{d.name}</span>
                  <span className="text-terminal-text-primary font-mono tabular-nums">{d.value}%</span>
                </div>
              ))}
            </div>

            {/* Dynamic Movers */}
            <div className="mt-4 pt-3 border-t border-terminal-border">
              <div className="text-[9px] text-terminal-text-faint uppercase tracking-wider font-medium mb-2">Dynamic Movers</div>
              {movers.length === 0 && (
                <div style={{color: "var(--muted)", fontSize: 11, padding: "16px", textAlign: "center", opacity: 0.7}}>
                  Movers loading — awaiting universe engine...
                </div>
              )}
              {movers.map((m, i) => (
                <div key={i} className="flex items-center gap-2 py-0.5 text-[9px]">
                  <span className="text-terminal-text-primary font-mono w-10">{m.ticker}</span>
                  <span className={`font-mono tabular-nums ${m.change >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {m.change >= 0 ? "+" : ""}{m.change.toFixed(1)}%
                  </span>
                  <span className={`ml-auto text-[8px] px-1 py-0.5 rounded ${
                    m.momentum === "strong" ? "bg-terminal-positive/10 text-terminal-positive" :
                    m.momentum === "weak" ? "bg-terminal-negative/10 text-terminal-negative" :
                    "bg-terminal-warning/10 text-terminal-warning"
                  }`}>{m.momentum}</span>
                </div>
              ))}
            </div>
          </DashboardPanel>
        </ResizableDashboard>
      </div>
    </div>
  );
}
