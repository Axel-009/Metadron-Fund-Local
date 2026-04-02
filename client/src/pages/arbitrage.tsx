import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";

// ═══════════ MOCK DATA ═══════════

type Signal = "LONG_A_SHORT_B" | "SHORT_A_LONG_B" | "EXIT" | "FLAT";

const SIGNAL_COLORS: Record<Signal, string> = {
  LONG_A_SHORT_B:  "#3fb950",
  SHORT_A_LONG_B:  "#f85149",
  EXIT:            "#d29922",
  FLAT:            "#8b949e",
};

interface RVPair {
  pairA: string; pairB: string; sector: string;
  zscore: number; signal: Signal; halfLife: number; hedgeRatio: number;
  correlation: number; pval: number; spread: number; pnl: number;
}

const RV_PAIRS: RVPair[] = [
  // Tech
  { pairA:"AAPL",  pairB:"MSFT",  sector:"Tech",       zscore:  2.34, signal:"LONG_A_SHORT_B",  halfLife:  8.2, hedgeRatio: 0.847, correlation: 0.921, pval:0.0012, spread:  42.18, pnl:  18420 },
  { pairA:"NVDA",  pairB:"AMD",   sector:"Tech",       zscore: -2.81, signal:"SHORT_A_LONG_B",  halfLife:  5.7, hedgeRatio: 1.312, correlation: 0.903, pval:0.0008, spread: -18.44, pnl:  34210 },
  { pairA:"CRM",   pairB:"ORCL",  sector:"Tech",       zscore:  0.42, signal:"FLAT",            halfLife: 11.4, hedgeRatio: 0.723, correlation: 0.887, pval:0.0024, spread:  12.31, pnl:   3820 },
  { pairA:"GOOGL", pairB:"META",  sector:"Tech",       zscore: -1.87, signal:"FLAT",            halfLife:  9.1, hedgeRatio: 1.041, correlation: 0.876, pval:0.0031, spread:  -8.92, pnl:  -2140 },
  // Financials
  { pairA:"JPM",   pairB:"BAC",   sector:"Financials", zscore:  3.12, signal:"LONG_A_SHORT_B",  halfLife:  6.8, hedgeRatio: 2.184, correlation: 0.944, pval:0.0004, spread:  24.50, pnl:  42810 },
  { pairA:"GS",    pairB:"MS",    sector:"Financials", zscore: -2.44, signal:"SHORT_A_LONG_B",  halfLife:  7.3, hedgeRatio: 1.628, correlation: 0.931, pval:0.0009, spread: -31.22, pnl:  27640 },
  { pairA:"V",     pairB:"MA",    sector:"Financials", zscore:  0.18, signal:"EXIT",            halfLife: 14.2, hedgeRatio: 0.911, correlation: 0.962, pval:0.0002, spread:   4.80, pnl:   9810 },
  // Energy
  { pairA:"XOM",   pairB:"CVX",   sector:"Energy",     zscore: -1.22, signal:"FLAT",            halfLife: 12.1, hedgeRatio: 1.247, correlation: 0.918, pval:0.0017, spread:  -9.14, pnl:  -1240 },
  { pairA:"SLB",   pairB:"HAL",   sector:"Energy",     zscore:  2.09, signal:"LONG_A_SHORT_B",  halfLife:  8.9, hedgeRatio: 1.872, correlation: 0.892, pval:0.0022, spread:  17.42, pnl:  14380 },
  // Healthcare
  { pairA:"PFE",   pairB:"MRK",   sector:"Healthcare", zscore: -3.41, signal:"SHORT_A_LONG_B",  halfLife:  7.4, hedgeRatio: 0.972, correlation: 0.874, pval:0.0038, spread: -22.88, pnl:  51240 },
  { pairA:"JNJ",   pairB:"ABT",   sector:"Healthcare", zscore:  0.61, signal:"FLAT",            halfLife: 15.8, hedgeRatio: 1.418, correlation: 0.861, pval:0.0041, spread:   7.24, pnl:   2180 },
  { pairA:"UNH",   pairB:"CI",    sector:"Healthcare", zscore:  2.78, signal:"LONG_A_SHORT_B",  halfLife:  9.2, hedgeRatio: 2.041, correlation: 0.902, pval:0.0019, spread:  38.91, pnl:  31820 },
  // Consumer
  { pairA:"KO",    pairB:"PEP",   sector:"Consumer",   zscore: -0.94, signal:"FLAT",            halfLife: 21.3, hedgeRatio: 0.879, correlation: 0.953, pval:0.0006, spread:  -5.18, pnl:  -1080 },
  { pairA:"HD",    pairB:"LOW",   sector:"Consumer",   zscore:  1.74, signal:"FLAT",            halfLife: 13.7, hedgeRatio: 1.124, correlation: 0.937, pval:0.0011, spread:  19.42, pnl:   8420 },
  { pairA:"MCD",   pairB:"SBUX",  sector:"Consumer",   zscore: -2.17, signal:"SHORT_A_LONG_B",  halfLife:  8.4, hedgeRatio: 1.634, correlation: 0.894, pval:0.0028, spread: -28.64, pnl:  19840 },
  { pairA:"NKE",   pairB:"LULU",  sector:"Consumer",   zscore:  3.05, signal:"LONG_A_SHORT_B",  halfLife:  6.1, hedgeRatio: 0.748, correlation: 0.868, pval:0.0044, spread:  41.28, pnl:  38120 },
  { pairA:"COST",  pairB:"TGT",   sector:"Consumer",   zscore: -0.37, signal:"EXIT",            halfLife: 17.4, hedgeRatio: 2.318, correlation: 0.912, pval:0.0021, spread:  -3.94, pnl:   7240 },
  // Industrial
  { pairA:"BA",    pairB:"LMT",   sector:"Industrial", zscore:  1.93, signal:"FLAT",            halfLife: 11.6, hedgeRatio: 0.651, correlation: 0.841, pval:0.0057, spread:  22.18, pnl:   6840 },
  { pairA:"CAT",   pairB:"DE",    sector:"Industrial", zscore: -2.62, signal:"SHORT_A_LONG_B",  halfLife:  9.8, hedgeRatio: 1.193, correlation: 0.883, pval:0.0033, spread: -44.82, pnl:  24180 },
  { pairA:"UPS",   pairB:"FDX",   sector:"Industrial", zscore:  0.84, signal:"FLAT",            halfLife: 16.2, hedgeRatio: 0.921, correlation: 0.896, pval:0.0026, spread:  11.54, pnl:   4120 },
];

const MISPRICING = [
  { ticker:"NVDA",  fairValue: 812.40, mktPrice: 878.20, mispct:  7.5, alpha:"MOM",  catalyst:"Blackwell ramp",        confidence: 82 },
  { ticker:"PFE",   fairValue:  34.80, mktPrice:  28.44, mispct: -18.3,alpha:"QUAL", catalyst:"Pipeline undervalued",  confidence: 74 },
  { ticker:"INTC",  fairValue:  52.10, mktPrice:  42.38, mispct: -18.7,alpha:"VAL",  catalyst:"Fab turnaround",        confidence: 61 },
  { ticker:"TSLA",  fairValue: 148.20, mktPrice: 182.64, mispct: 18.9, alpha:"MOM",  catalyst:"Model 3 refresh",       confidence: 44 },
  { ticker:"BA",    fairValue: 198.50, mktPrice: 171.88, mispct: -13.4,alpha:"VAL",  catalyst:"737 MAX recertify",     confidence: 67 },
  { ticker:"ABBV",  fairValue: 192.30, mktPrice: 178.42, mispct:  -7.2,alpha:"QUAL", catalyst:"Skyrizi growth",        confidence: 78 },
  { ticker:"DKNG",  fairValue:  52.80, mktPrice:  44.18, mispct: -16.3,alpha:"GRW",  catalyst:"Sports betting expand", confidence: 55 },
  { ticker:"MRNA",  fairValue:  96.40, mktPrice: 124.82, mispct: 22.8, alpha:"MOM",  catalyst:"RSV vaccine hype",      confidence: 38 },
  { ticker:"WMT",   fairValue: 178.20, mktPrice: 192.44, mispct:  7.4, alpha:"QUAL", catalyst:"Sam's Club momentum",   confidence: 71 },
  { ticker:"GS",    fairValue: 468.20, mktPrice: 428.16, mispct:  -8.5,alpha:"VAL",  catalyst:"M&A advisory boom",    confidence: 69 },
];

const CROSS_ASSET_ARB = [
  { type:"ETF/NAV Arb",            desc:"SPY premium to NAV — buy basket, short SPY",       spread: 4.2, expectedPnl: 12400, risk: "LOW"    },
  { type:"Futures/Spot Basis",     desc:"ES1 futures rich vs cash S&P by 8bps",             spread: 8.1, expectedPnl: 24200, risk: "LOW"    },
  { type:"Index/Constituent",      desc:"QQQ discount to AAPL+MSFT+NVDA basket",            spread:12.4, expectedPnl: 38400, risk: "MEDIUM" },
  { type:"ADR/Ordinary",           desc:"BABA ADR vs HK ordinary shares — 1.8% disloc",    spread:18.2, expectedPnl: 54800, risk: "MEDIUM" },
  { type:"CDS/Bond Basis",         desc:"JPM CDS vs bond spread neg basis trade",           spread:21.8, expectedPnl: 64100, risk: "MEDIUM" },
  { type:"Vol Surface Arb",        desc:"NVDA 90d implied vs realized — vol rich",          spread:31.4, expectedPnl: 92400, risk: "HIGH"   },
  { type:"Calendar Roll",          desc:"ES front/back month roll cost vs fair value",      spread: 2.8, expectedPnl:  8400, risk: "LOW"    },
  { type:"Cross-Exchange Arb",     desc:"MSFT Nasdaq vs BATS — 2bps resting spread",       spread: 2.1, expectedPnl:  6200, risk: "LOW"    },
];

const FACTOR_RADAR = [
  { factor:"MKT",  value: 0.84 },
  { factor:"SMB",  value: 0.12 },
  { factor:"HML",  value:-0.28 },
  { factor:"MOM",  value: 0.61 },
  { factor:"QUAL", value: 0.44 },
];

const RISK_COLORS: Record<string, string> = {
  LOW:    "#3fb950",
  MEDIUM: "#d29922",
  HIGH:   "#f85149",
};

const TOOLTIP_STYLE = {
  backgroundColor: "#0d1117",
  border: "1px solid #21262d",
  borderRadius: 4,
  fontSize: 10,
  color: "#e6edf3",
  padding: "4px 8px",
};

function generateZscoreHistory(currentZ: number, halfLife: number) {
  const data: { day: number; z: number }[] = [];
  let z = -currentZ * 0.7; // starts on other side
  for (let i = 0; i < 60; i++) {
    const reversion = -z * (1 - Math.exp(-1 / halfLife));
    z = z + reversion + (Math.random() - 0.5) * 0.3;
    // Converge to currentZ at end
    if (i === 59) z = currentZ;
    data.push({ day: i + 1, z: +z.toFixed(3) });
  }
  return data;
}

export default function ArbitrageDashboard() {
  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const [sortKey, setSortKey] = useState<string>("zscore");
  const [sortDir, setSortDir] = useState<1 | -1>(-1);

  const selectedPair = RV_PAIRS[selectedIdx];

  const sortedPairs = useMemo(() => {
    return [...RV_PAIRS].sort((a, b) => {
      const av = Math.abs((a as Record<string, number | string>)[sortKey] as number);
      const bv = Math.abs((b as Record<string, number | string>)[sortKey] as number);
      if (typeof av === "number" && typeof bv === "number") return sortDir * (bv - av);
      return 0;
    });
  }, [sortKey, sortDir]);

  const zHistory = useMemo(() => generateZscoreHistory(selectedPair.zscore, selectedPair.halfLife), [selectedIdx]);

  // Normalize factor values to 0-100 for radar
  const radarData = FACTOR_RADAR.map(f => ({
    factor: f.factor,
    value: Math.round((f.value + 1) * 50), // -1..1 → 0..100
    raw: f.value,
  }));

  const activePairs   = RV_PAIRS.filter(p => p.signal !== "FLAT").length;
  const totalPairPnL  = RV_PAIRS.reduce((s, p) => s + p.pnl, 0);
  const highConviction= RV_PAIRS.filter(p => Math.abs(p.zscore) >= 3).length;

  return (
    <div className="h-full flex flex-col gap-1 p-1 overflow-hidden">

      {/* ─── Summary Bar ─── */}
      <div className="flex-shrink-0 grid gap-1" style={{ gridTemplateColumns: "repeat(5, 1fr)" }}>
        {[
          { label:"ACTIVE PAIRS",     value:`${activePairs} / ${RV_PAIRS.length}`,  color:"text-terminal-accent" },
          { label:"HIGH CONVICTION",  value:`${highConviction} pairs`,               color:"text-terminal-warning" },
          { label:"PAIR BOOK P&L",    value:`+$${totalPairPnL.toLocaleString()}`,    color:"text-terminal-positive" },
          { label:"AVG Z-SCORE",      value:(RV_PAIRS.reduce((s,p)=>s+Math.abs(p.zscore),0)/RV_PAIRS.length).toFixed(2), color:"text-terminal-text-primary" },
          { label:"SELECTED PAIR",    value:`${selectedPair.pairA}/${selectedPair.pairB}`, color:"text-terminal-accent" },
        ].map(c => (
          <div key={c.label} className="terminal-panel p-2 flex flex-col justify-between">
            <span className="text-[9px] text-terminal-text-muted tracking-widest">{c.label}</span>
            <span className={`text-[13px] font-mono font-bold ${c.color}`}>{c.value}</span>
          </div>
        ))}
      </div>

      {/* ─── Main: RV Pairs Table + Right — Resizable ─── */}
      <div className="flex-1 min-h-0">
        <ResizableDashboard defaultSizes={[65, 35]} minSizes={[35, 20]}>
      <DashboardPanel
        title="ACTIVE RV PAIRS (STAT ARB)"
        noPadding
        headerRight={
          <span className="text-[9px] text-terminal-text-muted">click row to inspect</span>
        }
      >
        <div className="overflow-auto h-full">
          <table className="w-full text-[10px] font-mono border-collapse">
            <thead className="sticky top-0 bg-terminal-surface z-10">
              <tr className="border-b border-terminal-border">
                <th className="text-terminal-text-muted text-[9px] tracking-widest text-left px-2 py-1">PAIR (A / B)</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1">SECTOR</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">Z-SCORE</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1">SIGNAL</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">HL(d)</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">HEDGE β</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">CORR</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">COINT p</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">SPREAD</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">P&L</th>
              </tr>
            </thead>
            <tbody>
              {sortedPairs.map((p, i) => {
                const realIdx = RV_PAIRS.indexOf(p);
                const isSelected = realIdx === selectedIdx;
                const zAbs = Math.abs(p.zscore);
                const zColor = zAbs >= 3 ? "#f85149" : zAbs >= 2 ? "#d29922" : "#e6edf3";
                return (
                  <tr
                    key={`${p.pairA}/${p.pairB}`}
                    className={`border-b border-terminal-border cursor-pointer transition-colors ${isSelected ? "bg-[#00d4aa11]" : i % 2 === 0 ? "" : "bg-[#0d1117]"} hover:bg-[#161b22]`}
                    onClick={() => setSelectedIdx(realIdx)}
                  >
                    <td className="px-2 py-0.5">
                      <span className="text-terminal-accent font-bold">{p.pairA}</span>
                      <span className="text-terminal-text-faint">/</span>
                      <span className="text-terminal-text-muted">{p.pairB}</span>
                    </td>
                    <td className="px-1 py-0.5 text-terminal-text-muted text-[9px]">{p.sector}</td>
                    <td className="px-1 py-0.5 text-right font-bold" style={{ color: zColor }}>
                      {p.zscore > 0 ? "+" : ""}{p.zscore.toFixed(2)}
                    </td>
                    <td className="px-1 py-0.5">
                      <span className="px-1 rounded text-[8px]" style={{ backgroundColor: SIGNAL_COLORS[p.signal] + "22", color: SIGNAL_COLORS[p.signal] }}>
                        {p.signal === "LONG_A_SHORT_B" ? "L/S" : p.signal === "SHORT_A_LONG_B" ? "S/L" : p.signal}
                      </span>
                    </td>
                    <td className="px-1 py-0.5 text-terminal-text-muted text-right">{p.halfLife.toFixed(1)}</td>
                    <td className="px-1 py-0.5 text-terminal-text-primary text-right">{p.hedgeRatio.toFixed(3)}</td>
                    <td className="px-1 py-0.5 text-terminal-text-primary text-right">{p.correlation.toFixed(3)}</td>
                    <td className="px-1 py-0.5 text-terminal-text-muted text-right">{p.pval.toFixed(4)}</td>
                    <td className={`px-1 py-0.5 text-right ${p.spread >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                      {p.spread >= 0 ? "+" : ""}{p.spread.toFixed(2)}
                    </td>
                    <td className={`px-1 py-0.5 text-right font-bold ${p.pnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                      {p.pnl >= 0 ? "+$" : "-$"}{Math.abs(p.pnl).toLocaleString()}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </DashboardPanel>

      {/* ─── Right: Spread Chart + Factor Radar ─── */}
      <div className="h-full flex flex-col gap-1">
        {/* Z-Score Chart */}
        <DashboardPanel
          title={`Z-SCORE: ${selectedPair.pairA}/${selectedPair.pairB} (60D)`}
          className="flex-1"
        >
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={zHistory} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
              <XAxis dataKey="day" tick={{ fontSize: 7, fill: "#8b949e" }} tickFormatter={v => `D${v}`} interval={9} />
              <YAxis domain={[-4.5, 4.5]} tick={{ fontSize: 7, fill: "#8b949e" }} />
              <ReferenceLine y={ 2.0} stroke="#3fb950" strokeDasharray="4 2" strokeWidth={1} label={{ value:"+2", position:"right", fontSize:7, fill:"#3fb950" }} />
              <ReferenceLine y={-2.0} stroke="#3fb950" strokeDasharray="4 2" strokeWidth={1} label={{ value:"-2", position:"right", fontSize:7, fill:"#3fb950" }} />
              <ReferenceLine y={ 0.5} stroke="#d29922" strokeDasharray="2 2" strokeWidth={1} label={{ value:"+0.5", position:"right", fontSize:6, fill:"#d29922" }} />
              <ReferenceLine y={-0.5} stroke="#d29922" strokeDasharray="2 2" strokeWidth={1} label={{ value:"-0.5", position:"right", fontSize:6, fill:"#d29922" }} />
              <ReferenceLine y={ 4.0} stroke="#f85149" strokeDasharray="4 2" strokeWidth={1} label={{ value:"STOP", position:"right", fontSize:6, fill:"#f85149" }} />
              <ReferenceLine y={-4.0} stroke="#f85149" strokeDasharray="4 2" strokeWidth={1} label={{ value:"STOP", position:"right", fontSize:6, fill:"#f85149" }} />
              <ReferenceLine y={0}    stroke="#444" strokeWidth={1} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [v.toFixed(3), "Z-Score"]} />
              <Line type="monotone" dataKey="z" stroke="#00d4aa" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex gap-2 mt-1 text-[8px] font-mono">
            <span className="text-terminal-text-faint">HL: <span className="text-terminal-accent">{selectedPair.halfLife.toFixed(1)}d</span></span>
            <span className="text-terminal-text-faint">β: <span className="text-terminal-accent">{selectedPair.hedgeRatio.toFixed(3)}</span></span>
            <span className="text-terminal-text-faint">ρ: <span className="text-terminal-accent">{selectedPair.correlation.toFixed(3)}</span></span>
            <span className="text-terminal-text-faint">p: <span className="text-terminal-accent">{selectedPair.pval.toFixed(4)}</span></span>
          </div>
        </DashboardPanel>

        {/* Factor Decomposition Radar */}
        <DashboardPanel title="FACTOR DECOMPOSITION (5F)" className="flex-none" style={{ height: "200px" }}>
          <ResponsiveContainer width="100%" height={150}>
            <RadarChart data={radarData} margin={{ top: 4, right: 20, left: 20, bottom: 4 }}>
              <PolarGrid stroke="#21262d" />
              <PolarAngleAxis dataKey="factor" tick={{ fontSize: 9, fill: "#8b949e" }} />
              <PolarRadiusAxis angle={90} domain={[0, 100]} tick={false} axisLine={false} />
              <Radar dataKey="value" stroke="#00d4aa" fill="#00d4aa" fillOpacity={0.18} strokeWidth={1.5} />
            </RadarChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-5 gap-0.5 mt-0.5">
            {FACTOR_RADAR.map(f => (
              <div key={f.factor} className="flex flex-col items-center">
                <span className="text-[8px] font-mono text-terminal-text-muted">{f.factor}</span>
                <span className={`text-[9px] font-mono font-bold ${f.value >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                  {f.value >= 0 ? "+" : ""}{f.value.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </DashboardPanel>
      </div>
        </ResizableDashboard>
      </div>

      {/* ─── Bottom Row: Mispricing + Cross-Asset Arb ─── */}
      <div className="flex-shrink-0 h-44">
        <ResizableDashboard defaultSizes={[55, 45]} minSizes={[30, 25]}>
      <DashboardPanel title="MISPRICING OPPORTUNITIES (FACTOR RESIDUALS)" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[10px] font-mono border-collapse">
            <thead className="sticky top-0 bg-terminal-surface z-10">
              <tr className="border-b border-terminal-border">
                <th className="text-terminal-text-muted text-[9px] tracking-widest text-left px-2 py-0.5">TICKER</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-0.5 text-right">FAIR VAL</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-0.5 text-right">MKT PX</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-0.5 text-right">MISP%</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-0.5">ALPHA</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-0.5">CATALYST</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-0.5 text-right">CONF</th>
              </tr>
            </thead>
            <tbody>
              {MISPRICING.map((m, i) => (
                <tr key={m.ticker} className={`border-b border-terminal-border hover:bg-[#161b22] ${i % 2 === 0 ? "" : "bg-[#0d1117]"}`}>
                  <td className="px-2 py-0.5 text-terminal-accent font-bold">{m.ticker}</td>
                  <td className="px-1 py-0.5 text-terminal-text-primary text-right">${m.fairValue.toFixed(2)}</td>
                  <td className="px-1 py-0.5 text-terminal-text-muted text-right">${m.mktPrice.toFixed(2)}</td>
                  <td className={`px-1 py-0.5 text-right font-bold ${m.mispct < 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {m.mispct > 0 ? "+" : ""}{m.mispct.toFixed(1)}%
                  </td>
                  <td className="px-1 py-0.5">
                    <span className="px-0.5 rounded text-[8px] bg-[#00d4aa22] text-terminal-accent">{m.alpha}</span>
                  </td>
                  <td className="px-1 py-0.5 text-terminal-text-muted text-[9px] truncate max-w-[120px]">{m.catalyst}</td>
                  <td className="px-1 py-0.5 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <div className="w-12 h-1.5 bg-[#161b22] rounded overflow-hidden">
                        <div className="h-full rounded" style={{ width:`${m.confidence}%`, backgroundColor: m.confidence>70?"#3fb950":m.confidence>50?"#d29922":"#f85149" }} />
                      </div>
                      <span className="text-[9px] text-terminal-text-primary">{m.confidence}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>

      <DashboardPanel title="CROSS-ASSET ARBITRAGE" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[10px] font-mono border-collapse">
            <thead className="sticky top-0 bg-terminal-surface z-10">
              <tr className="border-b border-terminal-border">
                <th className="text-terminal-text-muted text-[9px] tracking-widest text-left px-2 py-0.5">TYPE</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-0.5">DESCRIPTION</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-0.5 text-right">SPRD bps</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-0.5 text-right">EXP P&L</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-0.5">RISK</th>
              </tr>
            </thead>
            <tbody>
              {CROSS_ASSET_ARB.map((c, i) => (
                <tr key={c.type} className={`border-b border-terminal-border hover:bg-[#161b22] ${i % 2 === 0 ? "" : "bg-[#0d1117]"}`}>
                  <td className="px-2 py-0.5 text-terminal-accent text-[9px] whitespace-nowrap">{c.type}</td>
                  <td className="px-1 py-0.5 text-terminal-text-muted text-[9px] max-w-[140px] truncate">{c.desc}</td>
                  <td className="px-1 py-0.5 text-terminal-warning text-right font-bold">{c.spread.toFixed(1)}</td>
                  <td className="px-1 py-0.5 text-terminal-positive text-right">+${c.expectedPnl.toLocaleString()}</td>
                  <td className="px-1 py-0.5">
                    <span className="px-1 rounded text-[8px]" style={{ backgroundColor: RISK_COLORS[c.risk] + "22", color: RISK_COLORS[c.risk] }}>
                      {c.risk}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>
        </ResizableDashboard>
      </div>
    </div>
  );
}
