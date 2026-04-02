import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
} from "recharts";
import { useEngineQuery } from "@/hooks/use-engine-api";

// ═══════════ MOCK DATA ═══════════

const ETF_HOLDINGS = [
  { ticker: "SPY",  name: "SPDR S&P 500 ETF Trust",            aum: 521.4, er: 0.0945, category: "Equity",    qty: 12400, avgEntry: 478.22, last: 521.08, chg:  0.38, pnl:  531248, weight: 12.8, strategy: "Core"    },
  { ticker: "QQQ",  name: "Invesco QQQ Trust",                  aum: 244.1, er: 0.20,   category: "Equity",    qty:  8200, avgEntry: 412.50, last: 466.31, chg:  0.71, pnl:  441342, weight:  9.2, strategy: "Core"    },
  { ticker: "IWM",  name: "iShares Russell 2000 ETF",           aum:  52.3, er: 0.19,   category: "Equity",    qty:  6100, avgEntry: 194.80, last: 201.44, chg: -0.22, pnl:   40504, weight:  3.1, strategy: "Tactical"},
  { ticker: "EFA",  name: "iShares MSCI EAFE ETF",              aum:  62.8, er: 0.32,   category: "Equity",    qty:  9800, avgEntry:  72.15, last:  78.32, chg:  0.14, pnl:   60466, weight:  3.9, strategy: "Core"    },
  { ticker: "EEM",  name: "iShares MSCI Emerging Markets ETF",  aum:  16.9, er: 0.68,   category: "Equity",    qty:  7500, avgEntry:  38.90, last:  42.17, chg: -0.43, pnl:   24525, weight:  2.0, strategy: "Tactical"},
  { ticker: "AGG",  name: "iShares Core U.S. Aggregate Bond",   aum:  99.1, er: 0.03,   category: "Bond",      qty: 14200, avgEntry:  96.40, last:  95.11, chg:  0.08, pnl:  -18318, weight:  4.4, strategy: "Core"    },
  { ticker: "TLT",  name: "iShares 20+ Year Treasury Bond ETF", aum:  33.8, er: 0.15,   category: "Bond",      qty:  8900, avgEntry:  92.10, last:  88.43, chg:  0.22, pnl:  -32663, weight:  2.5, strategy: "Hedge"   },
  { ticker: "HYG",  name: "iShares iBoxx High Yield Corp Bond", aum:  13.2, er: 0.49,   category: "Bond",      qty:  6400, avgEntry:  77.80, last:  76.92, chg: -0.11, pnl:   -5632, weight:  1.6, strategy: "Income"  },
  { ticker: "GLD",  name: "SPDR Gold Shares",                   aum:  57.9, er: 0.40,   category: "Commodity", qty:  4100, avgEntry: 191.20, last: 230.44, chg:  0.55, pnl:  160884, weight:  3.0, strategy: "Hedge"   },
  { ticker: "USO",  name: "United States Oil Fund",             aum:   1.2, er: 0.83,   category: "Commodity", qty:  3200, avgEntry:  68.40, last:  71.18, chg: -1.24, pnl:    8896, weight:  0.7, strategy: "Tactical"},
  { ticker: "XLK",  name: "Technology Select Sector SPDR Fund", aum: 214.2, er: 0.09,   category: "Sector",    qty:  5800, avgEntry: 182.30, last: 218.47, chg:  0.84, pnl:  209786, weight:  4.0, strategy: "Core"    },
  { ticker: "XLF",  name: "Financial Select Sector SPDR Fund",  aum:  44.2, er: 0.09,   category: "Sector",    qty:  8900, avgEntry:  40.10, last:  45.82, chg:  0.29, pnl:   50908, weight:  1.3, strategy: "Tactical"},
  { ticker: "XLE",  name: "Energy Select Sector SPDR Fund",     aum:  32.1, er: 0.09,   category: "Sector",    qty:  5400, avgEntry:  87.20, last:  90.11, chg: -0.67, pnl:   15714, weight:  1.5, strategy: "Tactical"},
  { ticker: "XLV",  name: "Health Care Select Sector SPDR Fund",aum:  40.8, er: 0.09,   category: "Sector",    qty:  4700, avgEntry: 136.80, last: 139.24, chg:  0.12, pnl:   11468, weight:  2.1, strategy: "Core"    },
  { ticker: "ARKK", name: "ARK Innovation ETF",                 aum:   6.8, er: 0.75,   category: "Thematic",  qty:  3100, avgEntry:  51.20, last:  49.38, chg: -2.14, pnl:   -5642, weight:  0.5, strategy: "Tactical"},
  { ticker: "VWO",  name: "Vanguard FTSE Emerging Markets ETF", aum:  62.1, er: 0.08,   category: "Equity",    qty:  9100, avgEntry:  40.50, last:  43.82, chg: -0.18, pnl:   30212, weight:  1.3, strategy: "Core"    },
  { ticker: "LQD",  name: "iShares iBoxx IG Corp Bond ETF",     aum:  27.4, er: 0.14,   category: "Bond",      qty:  6200, avgEntry: 107.30, last: 108.44, chg:  0.09, pnl:    7068, weight:  2.1, strategy: "Income"  },
  { ticker: "IBIT", name: "iShares Bitcoin Trust ETF",          aum:  41.2, er: 0.25,   category: "Thematic",  qty:  2800, avgEntry:  34.20, last:  55.84, chg:  3.18, pnl:   60592, weight:  0.5, strategy: "Tactical"},
];

const CATEGORY_COLORS: Record<string, string> = {
  Equity:    "#00d4aa",
  Bond:      "#3b82f6",
  Commodity: "#d29922",
  Sector:    "#a855f7",
  Thematic:  "#f85149",
};

const STRATEGY_COLORS: Record<string, string> = {
  Core:    "#00d4aa",
  Tactical:"#d29922",
  Hedge:   "#3b82f6",
  Income:  "#3fb950",
};

const SECTOR_ETFS = [
  { ticker:"XLK",  name:"Tech",        chg:  0.84 },
  { ticker:"XLF",  name:"Financials",  chg:  0.29 },
  { ticker:"XLE",  name:"Energy",      chg: -0.67 },
  { ticker:"XLV",  name:"Healthcare",  chg:  0.12 },
  { ticker:"XLI",  name:"Industrials", chg:  0.33 },
  { ticker:"XLY",  name:"Cons Disc",   chg:  1.02 },
  { ticker:"XLP",  name:"Cons Stpls",  chg: -0.08 },
  { ticker:"XLU",  name:"Utilities",   chg: -0.41 },
  { ticker:"XLRE", name:"Real Estate", chg: -0.55 },
  { ticker:"XLB",  name:"Materials",   chg:  0.19 },
  { ticker:"XLC",  name:"Comm Svcs",   chg:  0.63 },
  { ticker:"XME",  name:"Metals",      chg: -0.92 },
];

const ETF_FLOWS = [
  { ticker:"SPY",  flow:  1842.3, dir:"in"  },
  { ticker:"QQQ",  flow:   923.7, dir:"in"  },
  { ticker:"IBIT", flow:   447.2, dir:"in"  },
  { ticker:"GLD",  flow:   312.8, dir:"in"  },
  { ticker:"IWM",  flow:    88.4, dir:"in"  },
  { ticker:"LQD",  flow:    41.2, dir:"in"  },
  { ticker:"HYG",  flow:    67.9, dir:"in"  },
  { ticker:"USO",  flow:   -55.8, dir:"out" },
  { ticker:"EEM",  flow:  -144.7, dir:"out" },
  { ticker:"TLT",  flow:  -234.1, dir:"out" },
  { ticker:"ARKK", flow:  -178.6, dir:"out" },
  { ticker:"XLE",  flow:   -92.3, dir:"out" },
];

function getCategoryData() {
  const map: Record<string, number> = {};
  ETF_HOLDINGS.forEach(h => { map[h.category] = (map[h.category] || 0) + h.weight; });
  return Object.entries(map).map(([name, value]) => ({ name, value: +value.toFixed(1) }));
}

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

export default function ETFDashboard() {
  // ─── Engine API ─────────────────────────────────────
  const { data: sectorData } = useEngineQuery<{ sectors: Array<{ sector: string; count: number; momentum: number }> }>("/universe/sectors", { refetchInterval: 30000 });

  const [sortKey, setSortKey] = useState<string>("weight");
  const [sortDir, setSortDir] = useState<1 | -1>(-1);

  const sorted = useMemo(() => {
    return [...ETF_HOLDINGS].sort((a, b) => {
      const av = (a as unknown as Record<string, number | string>)[sortKey];
      const bv = (b as unknown as Record<string, number | string>)[sortKey];
      if (typeof av === "number" && typeof bv === "number") return sortDir * (av - bv);
      return sortDir * String(av).localeCompare(String(bv));
    });
  }, [sortKey, sortDir]);

  const toggleSort = (k: string) => {
    if (k === sortKey) setSortDir(d => (d === 1 ? -1 : 1));
    else { setSortKey(k); setSortDir(-1); }
  };

  const topMovers = [...ETF_HOLDINGS].sort((a, b) => b.chg - a.chg).slice(0, 5);
  const botMovers = [...ETF_HOLDINGS].sort((a, b) => a.chg - b.chg).slice(0, 5);
  const catData = getCategoryData();

  const colHdr = (k: string, label: string) => (
    <th
      className="text-terminal-text-muted text-[9px] tracking-widest cursor-pointer hover:text-terminal-accent transition-colors whitespace-nowrap px-1 py-1 text-right"
      onClick={() => toggleSort(k)}
    >
      {label}{sortKey === k ? (sortDir === 1 ? "▲" : "▼") : ""}
    </th>
  );

  return (
    <div className="h-full grid gap-1 p-1 overflow-hidden" style={{ gridTemplateColumns: "1fr 200px", gridTemplateRows: "1fr 190px" }}>
      {/* ─── Main Holdings Table (top-left) ─── */}
      <DashboardPanel
        title="PORTFOLIO ETF HOLDINGS"
        noPadding
        headerRight={<span className="text-[9px] text-terminal-text-muted">{ETF_HOLDINGS.length} positions</span>}
      >
        <div className="overflow-auto h-full">
          <table className="w-full text-[10px] font-mono border-collapse">
            <thead className="sticky top-0 bg-terminal-surface z-10">
              <tr className="border-b border-terminal-border">
                <th className="text-terminal-text-muted text-[9px] tracking-widest text-left px-2 py-1">TICKER</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest text-left px-1 py-1">NAME</th>
                {colHdr("aum",      "AUM $B")}
                {colHdr("er",       "ER%")}
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1">CAT</th>
                {colHdr("qty",      "QTY")}
                {colHdr("avgEntry", "AVG")}
                {colHdr("last",     "LAST")}
                {colHdr("chg",      "CHG%")}
                {colHdr("pnl",      "UNRLZD P&L")}
                {colHdr("weight",   "WT%")}
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1">STRAT</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((h, i) => (
                <tr key={h.ticker} className={`border-b border-terminal-border hover:bg-[#161b22] transition-colors ${i % 2 === 0 ? "" : "bg-[#0d1117]"}`}>
                  <td className="px-2 py-0.5 text-terminal-accent font-bold">{h.ticker}</td>
                  <td className="px-1 py-0.5 text-terminal-text-primary max-w-[180px] truncate">{h.name}</td>
                  <td className="px-1 py-0.5 text-terminal-text-primary text-right">{fmtN(h.aum, 1)}</td>
                  <td className="px-1 py-0.5 text-terminal-text-muted text-right">{h.er.toFixed(2)}</td>
                  <td className="px-1 py-0.5 text-right">
                    <span className="px-1 rounded text-[8px]" style={{ backgroundColor: CATEGORY_COLORS[h.category] + "22", color: CATEGORY_COLORS[h.category] }}>
                      {h.category}
                    </span>
                  </td>
                  <td className="px-1 py-0.5 text-terminal-text-primary text-right">{h.qty.toLocaleString()}</td>
                  <td className="px-1 py-0.5 text-terminal-text-muted text-right">{fmtN(h.avgEntry)}</td>
                  <td className="px-1 py-0.5 text-terminal-text-primary text-right">{fmtN(h.last)}</td>
                  <td className={`px-1 py-0.5 text-right font-bold ${h.chg >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {h.chg >= 0 ? "+" : ""}{h.chg.toFixed(2)}%
                  </td>
                  <td className={`px-1 py-0.5 text-right ${h.pnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {h.pnl >= 0 ? "+$" : "-$"}{Math.abs(h.pnl).toLocaleString()}
                  </td>
                  <td className="px-1 py-0.5 text-terminal-text-primary text-right">{h.weight.toFixed(1)}</td>
                  <td className="px-1 py-0.5 text-right">
                    <span className="px-1 rounded text-[8px]" style={{ backgroundColor: STRATEGY_COLORS[h.strategy] + "22", color: STRATEGY_COLORS[h.strategy] }}>
                      {h.strategy}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>

      {/* ─── Right Column (spans both rows) ─── */}
      <div className="row-span-2 flex flex-col gap-1">
        {/* Category Allocation Donut */}
        <DashboardPanel title="CATEGORY ALLOC" className="flex-none" style={{ height: "230px" }}>
          <ResponsiveContainer width="100%" height={120}>
            <PieChart>
              <Pie data={catData} dataKey="value" innerRadius={30} outerRadius={52} paddingAngle={2} startAngle={90} endAngle={450}>
                {catData.map(entry => (
                  <Cell key={entry.name} fill={CATEGORY_COLORS[entry.name]} />
                ))}
              </Pie>
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(1)}%`, ""]} />
            </PieChart>
          </ResponsiveContainer>
          <div className="space-y-0.5">
            {catData.map(c => (
              <div key={c.name} className="flex items-center justify-between">
                <div className="flex items-center gap-1">
                  <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: CATEGORY_COLORS[c.name] }} />
                  <span className="text-[9px] text-terminal-text-muted">{c.name}</span>
                </div>
                <span className="text-[10px] font-mono text-terminal-text-primary">{c.value.toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </DashboardPanel>

        {/* ETF Flow Monitor */}
        <DashboardPanel title="ETF FLOW MONITOR" className="flex-1">
          <div className="space-y-1">
            {ETF_FLOWS.map(f => (
              <div key={f.ticker} className="flex items-center gap-1">
                <span className="text-[10px] font-mono text-terminal-accent w-9 flex-shrink-0">{f.ticker}</span>
                <div className="flex-1 relative h-2.5 rounded overflow-hidden bg-[#161b22]">
                  <div
                    className="absolute top-0 h-full rounded"
                    style={{
                      width: `${Math.min(100, (Math.abs(f.flow) / 2000) * 100)}%`,
                      backgroundColor: f.dir === "in" ? "#3fb950" : "#f85149",
                      left: f.dir === "in" ? "0" : "auto",
                      right: f.dir === "out" ? "0" : "auto",
                    }}
                  />
                </div>
                <span className={`text-[9px] font-mono w-16 text-right flex-shrink-0 ${f.dir === "in" ? "text-terminal-positive" : "text-terminal-negative"}`}>
                  {f.dir === "in" ? "+" : "-"}${Math.abs(f.flow).toFixed(0)}M
                </span>
              </div>
            ))}
          </div>
        </DashboardPanel>
      </div>

      {/* ─── Bottom Row: Sector Heatmap + Movers ─── */}
      <div className="grid gap-1" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {/* Sector Heatmap */}
        <DashboardPanel title="SECTOR ETF HEATMAP" noPadding>
          <div className="grid grid-cols-4 gap-0.5 p-1.5 h-full">
            {SECTOR_ETFS.map(s => {
              const intensity = Math.min(Math.abs(s.chg) / 2, 1);
              const bg = s.chg >= 0
                ? `rgba(63,185,80,${0.10 + intensity * 0.35})`
                : `rgba(248,81,73,${0.10 + intensity * 0.35})`;
              const col = s.chg >= 0 ? "#3fb950" : "#f85149";
              return (
                <div key={s.ticker} className="rounded flex flex-col items-center justify-center p-1 cursor-default" style={{ backgroundColor: bg, border: `1px solid ${col}33` }}>
                  <span className="text-[9px] font-mono font-bold" style={{ color: col }}>{s.ticker}</span>
                  <span className="text-[8px] text-terminal-text-muted leading-tight">{s.name}</span>
                  <span className="text-[9px] font-mono font-bold" style={{ color: col }}>{s.chg >= 0 ? "+" : ""}{s.chg.toFixed(2)}%</span>
                </div>
              );
            })}
          </div>
        </DashboardPanel>

        {/* Top/Bottom Movers */}
        <div className="grid gap-1" style={{ gridTemplateRows: "1fr 1fr" }}>
          <DashboardPanel title="TOP MOVERS TODAY" noPadding>
            <table className="w-full text-[10px] font-mono">
              <tbody>
                {topMovers.map(m => (
                  <tr key={m.ticker} className="border-b border-terminal-border">
                    <td className="px-2 py-0.5 text-terminal-accent font-bold">{m.ticker}</td>
                    <td className="px-1 py-0.5 text-terminal-text-muted text-[9px] truncate">{m.name.split(" ").slice(0,3).join(" ")}</td>
                    <td className="px-2 py-0.5 text-terminal-positive text-right font-bold">+{m.chg.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </DashboardPanel>
          <DashboardPanel title="BOTTOM MOVERS TODAY" noPadding>
            <table className="w-full text-[10px] font-mono">
              <tbody>
                {botMovers.map(m => (
                  <tr key={m.ticker} className="border-b border-terminal-border">
                    <td className="px-2 py-0.5 text-terminal-accent font-bold">{m.ticker}</td>
                    <td className="px-1 py-0.5 text-terminal-text-muted text-[9px] truncate">{m.name.split(" ").slice(0,3).join(" ")}</td>
                    <td className="px-2 py-0.5 text-terminal-negative text-right font-bold">{m.chg.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </DashboardPanel>
        </div>
      </div>
    </div>
  );
}
