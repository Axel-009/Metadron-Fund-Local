import { useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
} from "recharts";
import { useEngineQuery } from "@/hooks/use-engine-api";

// ═══════════ MOCK DATA ═══════════

const SUMMARY = {
  totalExposure: 48_230_000,
  avgDuration: 6.42,
  avgYield: 4.87,
  avgRating: "A-",
  dv01: 30_979,
  convexity: 0.72,
};

const BOND_HOLDINGS = [
  { security:"US Treasury 2Y",       coupon: 4.875, maturity:"2026-03", rating:"AAA", yield: 4.62, duration: 1.88, dv01:  3421, faceVal: 18_000_000, mktVal: 18_124_200, pnl:  124200, spread:   0, sector:"Govt"  },
  { security:"US Treasury 5Y",       coupon: 4.250, maturity:"2029-02", rating:"AAA", yield: 4.41, duration: 4.52, dv01:  5812, faceVal: 12_000_000, mktVal: 11_892_000, pnl: -108000, spread:   0, sector:"Govt"  },
  { security:"US Treasury 10Y",      coupon: 4.000, maturity:"2034-02", rating:"AAA", yield: 4.58, duration: 8.14, dv01:  7443, faceVal:  8_000_000, mktVal:  7_728_000, pnl: -272000, spread:   0, sector:"Govt"  },
  { security:"US Treasury 30Y",      coupon: 4.375, maturity:"2053-08", rating:"AAA", yield: 4.82, duration:17.21, dv01:  4128, faceVal:  2_000_000, mktVal:  1_928_400, pnl:  -71600, spread:   0, sector:"Govt"  },
  { security:"AAPL 3.25% 2028",      coupon: 3.250, maturity:"2028-02", rating:"AA+", yield: 4.71, duration: 3.68, dv01:  1847, faceVal:  5_000_000, mktVal:  4_742_500, pnl:  -56000, spread:  62, sector:"Corp"  },
  { security:"MSFT 2.75% 2027",      coupon: 2.750, maturity:"2027-08", rating:"AAA", yield: 4.55, duration: 3.11, dv01:  1562, faceVal:  4_000_000, mktVal:  3_768_000, pnl:  -92000, spread:  46, sector:"Corp"  },
  { security:"JPM 4.50% 2029",       coupon: 4.500, maturity:"2029-01", rating:"A-",  yield: 5.02, duration: 4.28, dv01:  2034, faceVal:  4_500_000, mktVal:  4_328_550, pnl:  -79400, spread:  91, sector:"Corp"  },
  { security:"GOOGL 3.375% 2030",    coupon: 3.375, maturity:"2030-02", rating:"AA",  yield: 4.78, duration: 5.47, dv01:  2218, faceVal:  3_000_000, mktVal:  2_779_200, pnl:  -66000, spread:  67, sector:"Corp"  },
  { security:"BAC 5.00% 2031",       coupon: 5.000, maturity:"2031-01", rating:"BBB+",yield: 5.38, duration: 5.82, dv01:  1891, faceVal:  3_000_000, mktVal:  2_921_700, pnl:  -78300, spread: 127, sector:"Corp"  },
  { security:"Fannie Mae 4.5% MBS",  coupon: 4.500, maturity:"2052-01", rating:"AAA", yield: 5.14, duration: 6.92, dv01:  2414, faceVal:  5_000_000, mktVal:  4_741_000, pnl: -259000, spread:  73, sector:"Agency"},
  { security:"GNMA 30Y Pool",        coupon: 4.000, maturity:"2053-06", rating:"AAA", yield: 4.98, duration: 7.44, dv01:  1876, faceVal:  2_500_000, mktVal:  2_374_500, pnl: -125500, spread:  57, sector:"Agency"},
  { security:"CA Muni 3.0% 2032",    coupon: 3.000, maturity:"2032-09", rating:"AA",  yield: 3.42, duration: 7.18, dv01:   936, faceVal:  2_000_000, mktVal:  1_944_000, pnl:  -56000, spread: -35, sector:"Muni"  },
  { security:"NYC Muni 4.0% 2035",   coupon: 4.000, maturity:"2035-01", rating:"AA-", yield: 3.88, duration: 8.92, dv01:   812, faceVal:  1_500_000, mktVal:  1_531_500, pnl:   31500, spread: -43, sector:"Muni"  },
  { security:"HYG Proxy (HY Basket)",coupon: 6.500, maturity:"2029-06", rating:"BB",  yield: 6.82, duration: 3.44, dv01:   682, faceVal:  3_000_000, mktVal:  2_913_000, pnl:  -87000, spread: 321, sector:"HY"    },
  { security:"LQD Proxy (IG Basket)",coupon: 4.250, maturity:"2031-01", rating:"BBB", yield: 5.11, duration: 5.94, dv01:   908, faceVal:  2_500_000, mktVal:  2_421_750, pnl:  -78250, spread: 110, sector:"Corp"  },
];

const YIELD_CURVE = [
  { tenor:"3M",  yield: 5.28 },
  { tenor:"6M",  yield: 5.24 },
  { tenor:"1Y",  yield: 5.01 },
  { tenor:"2Y",  yield: 4.62 },
  { tenor:"3Y",  yield: 4.52 },
  { tenor:"5Y",  yield: 4.41 },
  { tenor:"7Y",  yield: 4.48 },
  { tenor:"10Y", yield: 4.58 },
  { tenor:"20Y", yield: 4.74 },
  { tenor:"30Y", yield: 4.82 },
];

const CREDIT_QUALITY = [
  { name:"AAA", pct: 38.2, color:"#00d4aa" },
  { name:"AA",  pct: 22.8, color:"#3b82f6" },
  { name:"A",   pct: 18.4, color:"#3fb950" },
  { name:"BBB", pct: 14.1, color:"#d29922" },
  { name:"BB",  pct:  6.5, color:"#f85149" },
];

const DURATION_LADDER = [
  { bucket:"0-1Y",   dv01: 3421 },
  { bucket:"1-3Y",   dv01: 7229 },
  { bucket:"3-5Y",   dv01: 8934 },
  { bucket:"5-7Y",   dv01: 5810 },
  { bucket:"7-10Y",  dv01: 4219 },
  { bucket:"10Y+",   dv01: 4128 },
];

function generateSpreadData() {
  const days = 30;
  const igData: { day: number; ig: number; hy: number }[] = [];
  let ig = 112; let hy = 318;
  for (let i = 0; i < days; i++) {
    ig = Math.max(80, ig + (Math.random() - 0.48) * 4);
    hy = Math.max(260, hy + (Math.random() - 0.47) * 8);
    igData.push({ day: i + 1, ig: +ig.toFixed(1), hy: +hy.toFixed(1) });
  }
  return igData;
}

const SECTOR_COLORS: Record<string, string> = {
  Govt:   "#3b82f6",
  Corp:   "#00d4aa",
  Agency: "#3fb950",
  Muni:   "#a855f7",
  HY:     "#f85149",
};

const RATING_COLORS: Record<string, string> = {
  "AAA":"#00d4aa", "AA+":"#22d3ee", "AA":"#3b82f6", "AA-":"#60a5fa",
  "A-":"#a78bfa", "BBB+":"#d29922", "BBB":"#f59e0b", "BB":"#f85149",
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
  // ─── Engine API ─────────────────────────────────────
  const { data: yieldData } = useEngineQuery<Record<string, number>>("/macro/yield-curve", { refetchInterval: 30000 });
  const { data: creditData } = useEngineQuery<Record<string, number>>("/macro/credit-pulse", { refetchInterval: 30000 });

  const spreadData = useMemo(() => generateSpreadData(), []);
  const totalPnL = BOND_HOLDINGS.reduce((s, b) => s + b.pnl, 0);

  return (
    <div className="h-full grid gap-1 p-1 overflow-hidden" style={{ gridTemplateColumns: "1fr 1fr 200px", gridTemplateRows: "56px 1fr 170px" }}>
      {/* ─── Row 1: Summary Cards ─── */}
      <div className="col-span-3 grid gap-1" style={{ gridTemplateColumns: "repeat(6, 1fr)" }}>
        {[
          { label:"TOTAL FI EXPOSURE", value:`$${(SUMMARY.totalExposure/1e6).toFixed(1)}M`, color:"text-terminal-accent" },
          { label:"AVG DURATION",      value:`${SUMMARY.avgDuration.toFixed(2)}Y`,           color:"text-terminal-text-primary" },
          { label:"AVG YIELD",         value:`${SUMMARY.avgYield.toFixed(2)}%`,               color:"text-terminal-warning" },
          { label:"AVG RATING",        value:SUMMARY.avgRating,                               color:"text-terminal-positive" },
          { label:"PORTFOLIO DV01",    value:`$${SUMMARY.dv01.toLocaleString()}`,             color:"text-terminal-accent" },
          { label:"CONVEXITY",         value:SUMMARY.convexity.toFixed(2),                    color:"text-terminal-text-primary" },
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
          <span className={`text-[10px] font-mono ${totalPnL >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
            P&L: {totalPnL >= 0 ? "+$" : "-$"}{Math.abs(totalPnL).toLocaleString()}
          </span>
        }
      >
        <div className="overflow-auto h-full">
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
              {BOND_HOLDINGS.map((b, i) => (
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
                  <td className="px-1 py-0.5 text-terminal-text-muted text-right">{fmtM(b.faceVal)}</td>
                  <td className="px-1 py-0.5 text-terminal-text-primary text-right">{fmtM(b.mktVal)}</td>
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
        </div>
      </DashboardPanel>

      {/* ─── Row 2: Right sidebar ─── */}
      <div className="flex flex-col gap-1">
        {/* Yield Curve */}
        <DashboardPanel title="US YIELD CURVE" className="flex-1">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={YIELD_CURVE} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <XAxis dataKey="tenor" tick={{ fontSize: 8, fill: "#8b949e" }} />
              <YAxis domain={[4.3, 5.4]} tickFormatter={v => `${v.toFixed(2)}%`} tick={{ fontSize: 8, fill: "#8b949e" }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(3)}%`, "Yield"]} />
              <Line type="monotone" dataKey="yield" stroke="#00d4aa" strokeWidth={2} dot={{ fill: "#00d4aa", r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </DashboardPanel>

        {/* Credit Quality Distribution */}
        <DashboardPanel title="CREDIT QUALITY" className="flex-none" style={{ height: "160px" }}>
          <div className="flex flex-col gap-1">
            {CREDIT_QUALITY.map(c => (
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
                <Pie data={CREDIT_QUALITY} dataKey="pct" innerRadius={14} outerRadius={24} paddingAngle={1} startAngle={90} endAngle={450}>
                  {CREDIT_QUALITY.map(c => <Cell key={c.name} fill={c.color} />)}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </DashboardPanel>
      </div>

      {/* ─── Row 3: Duration Ladder + Spread Monitor ─── */}
      <DashboardPanel title="DURATION LADDER (DV01 BY BUCKET)">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={DURATION_LADDER} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
            <XAxis dataKey="bucket" tick={{ fontSize: 8, fill: "#8b949e" }} />
            <YAxis tickFormatter={v => `$${(v/1000).toFixed(0)}K`} tick={{ fontSize: 8, fill: "#8b949e" }} />
            <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`$${v.toLocaleString()}`, "DV01"]} />
            <Bar dataKey="dv01" fill="#3b82f6" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </DashboardPanel>

      <DashboardPanel title="CREDIT SPREAD MONITOR (30D)" className="col-span-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={spreadData} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
            <XAxis dataKey="day" tick={{ fontSize: 8, fill: "#8b949e" }} tickFormatter={v => `D${v}`} />
            <YAxis yAxisId="ig" orientation="left"  domain={[80, 140]}  tickFormatter={v => `${v}bps`} tick={{ fontSize: 8, fill: "#8b949e" }} />
            <YAxis yAxisId="hy" orientation="right" domain={[260, 380]} tickFormatter={v => `${v}bps`} tick={{ fontSize: 8, fill: "#8b949e" }} />
            <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number, name: string) => [`${v.toFixed(1)} bps`, name === "ig" ? "IG Spread" : "HY Spread"]} />
            <Line yAxisId="ig" type="monotone" dataKey="ig" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="ig" />
            <Line yAxisId="hy" type="monotone" dataKey="hy" stroke="#f85149" strokeWidth={1.5} dot={false} name="hy" />
          </LineChart>
        </ResponsiveContainer>
      </DashboardPanel>
    </div>
  );
}
