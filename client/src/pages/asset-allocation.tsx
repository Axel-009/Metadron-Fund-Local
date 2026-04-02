import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import { MiniChart } from "@/components/mini-chart";
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";
import { useState, useEffect, useMemo } from "react";
import { useEngineQuery, type PortfolioLive, type AllocationData } from "@/hooks/use-engine-api";

const INDICES = [
  { ticker: "SPY", price: 527.82, change: 0.84, data: [510, 512, 515, 518, 520, 522, 525, 527] },
  { ticker: "QQQ", price: 448.19, change: 1.23, data: [430, 432, 435, 440, 442, 445, 447, 448] },
  { ticker: "IWM", price: 210.45, change: -0.32, data: [212, 211, 210, 209, 210, 211, 210, 210] },
  { ticker: "DIA", price: 398.76, change: 0.51, data: [392, 394, 395, 396, 397, 398, 398, 399] },
  { ticker: "VIX", price: 14.22, change: -3.41, data: [16, 15, 15, 14, 14, 14, 14, 14] },
];

// Extended HOLDINGS with new columns: productType, leveraged, marginPct, creditRating, strategy, weightNav, pnl
const HOLDINGS = [
  { ticker: "AAPL", name: "Apple Inc", weight: 8.5, shares: 1200, price: 189.45, change: 1.2, sector: "Technology", productType: "Equity", leveraged: false, marginPct: 50, creditRating: "AA+", strategy: "Momentum", weightNav: 8.50, pnl: 24380 },
  { ticker: "MSFT", name: "Microsoft Corp", weight: 7.8, shares: 800, price: 420.12, change: 0.8, sector: "Technology", productType: "Equity", leveraged: false, marginPct: 50, creditRating: "AAA", strategy: "Quality", weightNav: 7.80, pnl: 18920 },
  { ticker: "NVDA", name: "NVIDIA Corp", weight: 6.2, shares: 500, price: 875.30, change: 2.4, sector: "Technology", productType: "Equity", leveraged: false, marginPct: 50, creditRating: "A", strategy: "Growth", weightNav: 6.20, pnl: 52140 },
  { ticker: "AMZN", name: "Amazon.com", weight: 5.5, shares: 600, price: 185.67, change: 1.5, sector: "Consumer", productType: "Equity", leveraged: false, marginPct: 50, creditRating: "AA", strategy: "Growth", weightNav: 5.50, pnl: 15680 },
  { ticker: "GOOGL", name: "Alphabet Inc", weight: 4.8, shares: 700, price: 155.89, change: -0.3, sector: "Technology", productType: "Equity", leveraged: false, marginPct: 50, creditRating: "AA+", strategy: "Value", weightNav: 4.80, pnl: -3240 },
  { ticker: "JPM", name: "JPMorgan Chase", weight: 4.2, shares: 450, price: 198.34, change: 0.6, sector: "Financials", productType: "Equity", leveraged: false, marginPct: 50, creditRating: "A+", strategy: "Event-Driven", weightNav: 4.20, pnl: 8910 },
  { ticker: "UNH", name: "UnitedHealth", weight: 3.8, shares: 200, price: 502.15, change: -0.8, sector: "Healthcare", productType: "Equity", leveraged: false, marginPct: 50, creditRating: "A", strategy: "Quality", weightNav: 3.80, pnl: -7240 },
  { ticker: "V", name: "Visa Inc", weight: 3.5, shares: 350, price: 282.90, change: 0.4, sector: "Financials", productType: "Equity", leveraged: false, marginPct: 50, creditRating: "AA-", strategy: "Mean Reversion", weightNav: 3.50, pnl: 4120 },
  { ticker: "META", name: "Meta Platforms", weight: 3.2, shares: 250, price: 505.78, change: 1.8, sector: "Technology", productType: "Equity", leveraged: false, marginPct: 50, creditRating: "A+", strategy: "Momentum", weightNav: 3.20, pnl: 19450 },
  { ticker: "XOM", name: "Exxon Mobil", weight: 2.8, shares: 400, price: 115.23, change: -1.1, sector: "Energy", productType: "Equity", leveraged: false, marginPct: 50, creditRating: "AA", strategy: "Value", weightNav: 2.80, pnl: -9870 },
  { ticker: "SPY PUT", name: "SPY May-31 520P", weight: 1.8, shares: -30, price: 7.80, change: -2.1, sector: "Derivatives", productType: "Options", leveraged: true, marginPct: 20, creditRating: "NR", strategy: "Protective Put", weightNav: 1.80, pnl: 4800 },
  { ticker: "ESM25", name: "E-mini S&P Jun", weight: 1.0, shares: 4, price: 5282.75, change: 0.65, sector: "Futures", productType: "Futures", leveraged: true, marginPct: 100, creditRating: "NR", strategy: "Momentum", weightNav: 1.03, pnl: 34250 },
  { ticker: "GLD", name: "SPDR Gold ETF", weight: 1.5, shares: 600, price: 215.40, change: 0.3, sector: "Commodities", productType: "ETF", leveraged: false, marginPct: 50, creditRating: "AAA", strategy: "Mean Reversion", weightNav: 1.50, pnl: 3240 },
  { ticker: "TLT", name: "iShares 20Y Bond", weight: 1.2, shares: 800, price: 96.80, change: -0.4, sector: "Fixed Income", productType: "Fixed Income", leveraged: false, marginPct: 25, creditRating: "AAA", strategy: "Value", weightNav: 1.20, pnl: -2100 },
  { ticker: "USD CASH", name: "Cash & Equivalents", weight: 5.5, shares: 0, price: 1.00, change: 0, sector: "Cash", productType: "Cash", leveraged: false, marginPct: 0, creditRating: "AAA", strategy: "—", weightNav: 5.50, pnl: 0 },
];

const MOVERS = [
  { ticker: "SMCI", change: 12.4, momentum: "strong" },
  { ticker: "ARM", change: 8.7, momentum: "strong" },
  { ticker: "PLTR", change: 5.2, momentum: "moderate" },
  { ticker: "COIN", change: -6.8, momentum: "weak" },
  { ticker: "RIVN", change: -8.3, momentum: "weak" },
];

const ALLOC_DATA = [
  { name: "Technology", value: 38, color: "#00d4aa" },
  { name: "Financials", value: 15, color: "#58a6ff" },
  { name: "Healthcare", value: 12, color: "#3fb950" },
  { name: "Consumer", value: 11, color: "#bc8cff" },
  { name: "Energy", value: 8, color: "#f0883e" },
  { name: "Industrials", value: 7, color: "#d29922" },
  { name: "Cash", value: 9, color: "#484f58" },
];

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
  const indices = indicesApi?.indices?.length ? indicesApi.indices : INDICES;
  // Movers from API
  const movers = moversApi?.movers?.length ? moversApi.movers : MOVERS;
  // Sector allocation from API
  const allocPie = sectorAllocApi?.allocation?.length ? sectorAllocApi.allocation : ALLOC_DATA;

  // Merge API positions into HOLDINGS format when available
  const holdings = useMemo(() => {
    if (!posData?.positions?.length) return HOLDINGS;
    // Build lookup from API positions
    const apiMap = new Map(posData.positions.map((p) => [p.ticker, p]));
    return HOLDINGS.map((h) => {
      const api = apiMap.get(h.ticker);
      if (!api) return h;
      return {
        ...h,
        shares: api.quantity || h.shares,
        price: api.current_price || h.price,
        pnl: api.unrealized_pnl + api.realized_pnl,
        sector: api.sector || h.sector,
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
                    data={ALLOC_DATA}
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
