import { DashboardPanel } from "@/components/dashboard-panel";
import { MiniChart } from "@/components/mini-chart";
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";
import { useState, useEffect } from "react";

const INDICES = [
  { ticker: "SPY", price: 527.82, change: 0.84, data: [510, 512, 515, 518, 520, 522, 525, 527] },
  { ticker: "QQQ", price: 448.19, change: 1.23, data: [430, 432, 435, 440, 442, 445, 447, 448] },
  { ticker: "IWM", price: 210.45, change: -0.32, data: [212, 211, 210, 209, 210, 211, 210, 210] },
  { ticker: "DIA", price: 398.76, change: 0.51, data: [392, 394, 395, 396, 397, 398, 398, 399] },
  { ticker: "VIX", price: 14.22, change: -3.41, data: [16, 15, 15, 14, 14, 14, 14, 14] },
];

const HOLDINGS = [
  { ticker: "AAPL", name: "Apple Inc", weight: 8.5, shares: 1200, price: 189.45, change: 1.2, sector: "Technology" },
  { ticker: "MSFT", name: "Microsoft Corp", weight: 7.8, shares: 800, price: 420.12, change: 0.8, sector: "Technology" },
  { ticker: "NVDA", name: "NVIDIA Corp", weight: 6.2, shares: 500, price: 875.30, change: 2.4, sector: "Technology" },
  { ticker: "AMZN", name: "Amazon.com", weight: 5.5, shares: 600, price: 185.67, change: 1.5, sector: "Consumer" },
  { ticker: "GOOGL", name: "Alphabet Inc", weight: 4.8, shares: 700, price: 155.89, change: -0.3, sector: "Technology" },
  { ticker: "JPM", name: "JPMorgan Chase", weight: 4.2, shares: 450, price: 198.34, change: 0.6, sector: "Financials" },
  { ticker: "UNH", name: "UnitedHealth", weight: 3.8, shares: 200, price: 502.15, change: -0.8, sector: "Healthcare" },
  { ticker: "V", name: "Visa Inc", weight: 3.5, shares: 350, price: 282.90, change: 0.4, sector: "Financials" },
  { ticker: "META", name: "Meta Platforms", weight: 3.2, shares: 250, price: 505.78, change: 1.8, sector: "Technology" },
  { ticker: "XOM", name: "Exxon Mobil", weight: 2.8, shares: 400, price: 115.23, change: -1.1, sector: "Energy" },
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

export default function AssetAllocation() {
  const [nav, setNav] = useState(128450320);

  useEffect(() => {
    const iv = setInterval(() => setNav((n) => n + Math.floor(Math.random() * 10000 - 3000)), 4000);
    return () => clearInterval(iv);
  }, []);

  return (
    <div className="h-full grid grid-cols-[1fr_1fr_300px] grid-rows-[auto_1fr] gap-[2px] p-[2px] overflow-auto" data-testid="asset-allocation">
      {/* NAV Display */}
      <DashboardPanel title="LIVE NAV" className="col-span-2">
        <div className="flex items-center gap-6">
          <div>
            <div className="text-3xl font-mono font-bold text-terminal-text-primary tabular-nums">
              ${nav.toLocaleString()}
            </div>
            <div className="text-[10px] text-terminal-positive font-mono mt-0.5">+$842,150 (+0.66%) today</div>
          </div>
          {/* Indices strip */}
          <div className="flex gap-4 ml-auto">
            {INDICES.map((idx) => (
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

      {/* Allocation Pie */}
      <DashboardPanel title="ALLOCATION" className="row-span-2">
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
                {ALLOC_DATA.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-1 mt-2">
          {ALLOC_DATA.map((d, i) => (
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
          {MOVERS.map((m, i) => (
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

      {/* Basket Formation Table */}
      <DashboardPanel title="BASKET FORMATION" className="col-span-2" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[9px] font-mono">
            <thead>
              <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                <th className="text-left px-2 py-1.5 font-medium">Ticker</th>
                <th className="text-left px-2 py-1.5 font-medium">Name</th>
                <th className="text-right px-2 py-1.5 font-medium">Weight</th>
                <th className="text-right px-2 py-1.5 font-medium">Shares</th>
                <th className="text-right px-2 py-1.5 font-medium">Price</th>
                <th className="text-right px-2 py-1.5 font-medium">Chg%</th>
                <th className="text-left px-2 py-1.5 font-medium">Sector</th>
              </tr>
            </thead>
            <tbody>
              {HOLDINGS.map((h, i) => (
                <tr key={i} className="border-b border-terminal-border/20 hover:bg-white/[0.02]">
                  <td className="px-2 py-1.5 text-terminal-accent font-medium">{h.ticker}</td>
                  <td className="px-2 py-1.5 text-terminal-text-muted">{h.name}</td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-primary tabular-nums">{h.weight.toFixed(1)}%</td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-muted tabular-nums">{h.shares.toLocaleString()}</td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-primary tabular-nums">${h.price.toFixed(2)}</td>
                  <td className={`px-2 py-1.5 text-right tabular-nums ${h.change >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {h.change >= 0 ? "+" : ""}{h.change.toFixed(1)}%
                  </td>
                  <td className="px-2 py-1.5 text-terminal-text-faint">{h.sector}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>
    </div>
  );
}
