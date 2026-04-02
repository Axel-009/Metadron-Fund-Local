import { useEffect, useState, useRef, useMemo, useCallback } from "react";
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line, ComposedChart,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Treemap, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend, Scatter,
} from "recharts";
import { useEngineQuery } from "@/hooks/use-engine-api";

/* ═══════════════════════════════════════════════════════════════════════
   METADRON MONEY VELOCITY & US LIQUIDITY INDICATOR TAB
   ═══════════════════════════════════════════════════════════════════════
   Sections:
   1. Fed Balance Sheet & Net Liquidity
   2. Reserve Distribution (TGA, ON-RRP, Bank Reserves, Standing Repo)
   3. Front-End Rates (SOFR, EFFR, T-Bills, Repo)
   4. M2 Money Velocity (V = GDP/M2) with regime detection
   5. Credit Impulse & Collateral Chain
   6. CtV (Carry-to-Volatility) Liquidity Panel
   7. Money Flow Sankey / Distribution
   8. Sector Liquidity Absorption
   9. US Liquidity Composite Indicator
  10. Daily / Weekly / Monthly / Quarterly Change Dashboard
   ═══════════════════════════════════════════════════════════════════════ */

// ── Colour Palette ──
const C = {
  accent: "#00d4aa",
  accentDim: "#00d4aa80",
  warn: "#f59e0b",
  warnDim: "#f59e0b60",
  danger: "#ef4444",
  dangerDim: "#ef444460",
  positive: "#22c55e",
  blue: "#3b82f6",
  blueDim: "#3b82f640",
  purple: "#a855f7",
  purpleDim: "#a855f740",
  cyan: "#06b6d4",
  orange: "#f97316",
  pink: "#ec4899",
  gridLine: "#1e293b",
  tooltipBg: "#0f172a",
  panelBg: "#0a0f1a",
  panelBorder: "#1a2332",
  textPrimary: "#e2e8f0",
  textMuted: "#94a3b8",
  textFaint: "#475569",
};

const PIE_COLORS = [C.accent, C.blue, C.purple, C.orange, C.cyan, C.pink, C.warn, C.danger];

// ── Helpers ──
const fmt = (v: number, dec = 2) => v.toFixed(dec);
const fmtB = (v: number) => `$${(v / 1000).toFixed(1)}T`;
const fmtPct = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
const fmtBps = (v: number) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(0)}bps`;
const jitter = (v: number, pct = 0.003) => v * (1 + (Math.random() - 0.5) * pct * 2);
const ts = () => new Date().toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });

// ── Data Generators ──
function genDateLabels(n: number, unit: "D" | "W" | "M" | "Q" = "D"): string[] {
  const labels: string[] = [];
  const d = new Date();
  for (let i = n - 1; i >= 0; i--) {
    const dd = new Date(d);
    if (unit === "D") dd.setDate(d.getDate() - i);
    else if (unit === "W") dd.setDate(d.getDate() - i * 7);
    else if (unit === "M") dd.setMonth(d.getMonth() - i);
    else dd.setMonth(d.getMonth() - i * 3);
    labels.push(dd.toLocaleDateString("en-US", { month: "short", day: "2-digit" }));
  }
  return labels;
}

function genFedBalanceSheet(n: number) {
  const dates = genDateLabels(n);
  let total = 6820; // $B
  let treasuries = 4220;
  let mbs = 2340;
  let other = 260;
  return dates.map((d) => {
    total += (Math.random() - 0.55) * 12;
    treasuries += (Math.random() - 0.54) * 8;
    mbs += (Math.random() - 0.56) * 4;
    other = total - treasuries - mbs;
    return { date: d, total: +total.toFixed(0), treasuries: +treasuries.toFixed(0), mbs: +mbs.toFixed(0), other: +Math.max(other, 150).toFixed(0) };
  });
}

function genReserves(n: number) {
  const dates = genDateLabels(n);
  let tga = 722;
  let onRrp = 118;
  let bankRes = 3340;
  let standingRepo = 5;
  return dates.map((d) => {
    tga += (Math.random() - 0.48) * 30;
    tga = Math.max(200, Math.min(1200, tga));
    onRrp += (Math.random() - 0.52) * 15;
    onRrp = Math.max(0, Math.min(600, onRrp));
    bankRes += (Math.random() - 0.49) * 25;
    bankRes = Math.max(2800, Math.min(3800, bankRes));
    standingRepo += (Math.random() - 0.5) * 3;
    standingRepo = Math.max(0, Math.min(50, standingRepo));
    return { date: d, tga: +tga.toFixed(0), onRrp: +onRrp.toFixed(0), bankRes: +bankRes.toFixed(0), standingRepo: +standingRepo.toFixed(0) };
  });
}

function genNetLiquidity(fedBs: any[], reserves: any[]) {
  return fedBs.map((f, i) => {
    const r = reserves[i] || { tga: 700, onRrp: 100 };
    const net = f.total - r.tga - r.onRrp;
    return { date: f.date, net: +net.toFixed(0), total: f.total };
  });
}

function genFrontEndRates(n: number) {
  const dates = genDateLabels(n);
  let sofr = 4.32;
  let effr = 4.33;
  let tBill3m = 4.22;
  let tBill1m = 4.27;
  let repo = 4.35;
  let iorb = 4.40;
  return dates.map((d) => {
    sofr += (Math.random() - 0.5) * 0.04;
    effr = sofr + 0.005 + Math.random() * 0.015;
    tBill3m += (Math.random() - 0.5) * 0.05;
    tBill1m = tBill3m + 0.04 + Math.random() * 0.03;
    repo = sofr + 0.02 + Math.random() * 0.02;
    iorb = sofr + 0.07 + Math.random() * 0.02;
    return { date: d, sofr: +sofr.toFixed(3), effr: +effr.toFixed(3), tBill3m: +tBill3m.toFixed(3), tBill1m: +tBill1m.toFixed(3), repo: +repo.toFixed(3), iorb: +iorb.toFixed(3) };
  });
}

function genM2Velocity(n: number) {
  const dates = genDateLabels(n, "M");
  let v = 1.149;
  let m2 = 21200;
  let gdp = 28600;
  return dates.map((d) => {
    m2 += (Math.random() - 0.47) * 80;
    gdp += (Math.random() - 0.46) * 120;
    v = gdp / m2;
    return { date: d, velocity: +v.toFixed(4), m2: +m2.toFixed(0), gdp: +gdp.toFixed(0) };
  });
}

function genCreditImpulse(n: number) {
  const dates = genDateLabels(n, "M");
  let impulse = 0.8;
  let bankCredit = 17400;
  let consumerCredit = 5100;
  return dates.map((d) => {
    impulse += (Math.random() - 0.48) * 0.6;
    impulse = Math.max(-5, Math.min(5, impulse));
    bankCredit += (Math.random() - 0.48) * 60;
    consumerCredit += (Math.random() - 0.46) * 20;
    return { date: d, impulse: +impulse.toFixed(3), bankCredit: +bankCredit.toFixed(0), consumerCredit: +consumerCredit.toFixed(0) };
  });
}

function genCollateralChain() {
  return [
    { tier: "UST Tier-1", reuse: 3.2, haircut: 0.5, velocity: 6.4, quality: 98 },
    { tier: "Agency MBS", reuse: 2.4, haircut: 2.0, velocity: 4.1, quality: 92 },
    { tier: "Corp IG", reuse: 1.8, haircut: 5.0, velocity: 2.8, quality: 78 },
    { tier: "Corp HY", reuse: 1.1, haircut: 12.0, velocity: 1.4, quality: 52 },
    { tier: "ABS/CLO", reuse: 0.9, haircut: 15.0, velocity: 1.0, quality: 41 },
    { tier: "Equities", reuse: 0.7, haircut: 25.0, velocity: 0.6, quality: 35 },
  ];
}

function genCtvPairs() {
  return [
    { pair: "EURUSD", carry: -0.82, vol: 7.2, ctv: -0.114, gate: false, sdr: 0.2931 },
    { pair: "USDJPY", carry: 3.91, vol: 10.8, ctv: 0.362, gate: false, sdr: 0.0759 },
    { pair: "GBPUSD", carry: -0.45, vol: 8.1, ctv: -0.056, gate: false, sdr: 0.0744 },
    { pair: "USDCNY", carry: 1.82, vol: 3.9, ctv: 0.467, gate: false, sdr: 0.1228 },
    { pair: "AUDUSD", carry: 0.15, vol: 9.4, ctv: 0.016, gate: false, sdr: 0 },
    { pair: "USDCAD", carry: 0.38, vol: 6.7, ctv: 0.057, gate: false, sdr: 0 },
    { pair: "USDCHF", carry: 2.90, vol: 7.5, ctv: 0.387, gate: false, sdr: 0 },
    { pair: "NZDUSD", carry: 0.60, vol: 9.8, ctv: 0.061, gate: false, sdr: 0 },
  ].map((p) => ({ ...p, ctv: +p.ctv.toFixed(4), gate: p.ctv > 0.5 }));
}

function genSectorAbsorption() {
  return [
    { sector: "Info Tech", absorption: 0.72, flow: 18.4 },
    { sector: "Financials", absorption: 0.58, flow: 14.2 },
    { sector: "Health Care", absorption: 0.34, flow: 8.8 },
    { sector: "Cons Disc", absorption: 0.29, flow: 7.1 },
    { sector: "Industrials", absorption: 0.25, flow: 6.5 },
    { sector: "Energy", absorption: 0.18, flow: 4.6 },
    { sector: "Comm Svcs", absorption: 0.15, flow: 3.8 },
    { sector: "Materials", absorption: 0.11, flow: 2.7 },
    { sector: "Cons Staples", absorption: 0.08, flow: 2.1 },
    { sector: "Utilities", absorption: 0.05, flow: 1.3 },
    { sector: "Real Estate", absorption: 0.04, flow: 1.0 },
  ];
}

function genFlowDist() {
  return [
    { name: "Equities", value: 48.2, color: C.accent },
    { name: "Bonds", value: 28.1, color: C.blue },
    { name: "Alternatives", value: 13.7, color: C.purple },
    { name: "Cash/MM", value: 10.0, color: C.warn },
  ];
}

function genLiquidityComposite(n: number) {
  const dates = genDateLabels(n);
  let score = 62;
  return dates.map((d) => {
    score += (Math.random() - 0.48) * 3;
    score = Math.max(10, Math.min(95, score));
    const vComp = 15 + Math.random() * 10;
    const ciComp = 10 + Math.random() * 8;
    const tedComp = 12 + Math.random() * 6;
    const vixComp = 10 + Math.random() * 10;
    const ycComp = 8 + Math.random() * 6;
    return { date: d, score: +score.toFixed(1), vComp: +vComp.toFixed(1), ciComp: +ciComp.toFixed(1), tedComp: +tedComp.toFixed(1), vixComp: +vixComp.toFixed(1), ycComp: +ycComp.toFixed(1) };
  });
}

// ── Change Table Data ──
function genChanges() {
  const row = (label: string, current: number, unit: string) => {
    const d1 = (Math.random() - 0.5) * 2 * (current * 0.002);
    const w1 = (Math.random() - 0.5) * 2 * (current * 0.008);
    const m1 = (Math.random() - 0.5) * 2 * (current * 0.025);
    const q1 = (Math.random() - 0.5) * 2 * (current * 0.06);
    return { label, current, unit, d1, w1, m1, q1 };
  };
  return [
    row("Fed Balance Sheet", 6820, "$B"),
    row("Net Liquidity", 5998, "$B"),
    row("TGA Balance", 722, "$B"),
    row("ON-RRP", 118, "$B"),
    row("Bank Reserves", 3340, "$B"),
    row("SOFR", 4.32, "%"),
    row("EFFR", 4.33, "%"),
    row("3M T-Bill", 4.22, "%"),
    row("M2 Supply", 21200, "$B"),
    row("M2 Velocity", 1.149, "x"),
    row("Credit Impulse", 0.8, "σ"),
    row("Liquidity Score", 62.4, "/100"),
    row("10Y Yield", 4.28, "%"),
    row("2s10s Spread", -0.18, "bps"),
    row("HY Spread", 3.42, "%"),
    row("VIX", 16.8, ""),
  ];
}

// ── Gauge Component ──
function LiqGauge({ score, label }: { score: number; label: string }) {
  const angle = (score / 100) * 180;
  const rad = ((180 - angle) * Math.PI) / 180;
  const nx = 50 + 38 * Math.cos(rad);
  const ny = 52 - 38 * Math.sin(rad);
  const color = score > 65 ? C.positive : score > 40 ? C.warn : C.danger;
  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 100 60" className="w-full max-w-[140px]">
        <path d="M10 52 A40 40 0 0 1 90 52" fill="none" stroke={C.gridLine} strokeWidth="6" strokeLinecap="round" />
        <path d="M10 52 A40 40 0 0 1 90 52" fill="none" stroke={color} strokeWidth="6" strokeLinecap="round" strokeDasharray={`${(angle / 180) * 125.6} 200`} />
        <circle cx={nx} cy={ny} r="3" fill={color} />
        <text x="50" y="48" textAnchor="middle" fill={color} fontSize="14" fontFamily="monospace" fontWeight="bold">{score.toFixed(0)}</text>
      </svg>
      <span className="text-[8px] text-terminal-text-faint tracking-wider mt-0.5">{label}</span>
    </div>
  );
}

// ── Tiny Sparkline ──
function Spark({ data, color, w = 60, h = 16 }: { data: number[]; color: string; w?: number; h?: number }) {
  if (!data.length) return null;
  const mn = Math.min(...data);
  const mx = Math.max(...data);
  const range = mx - mn || 1;
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - mn) / range) * h}`).join(" ");
  return (
    <svg width={w} height={h} className="inline-block align-middle">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.2" />
    </svg>
  );
}

// ── Flow Arrow ──
function FlowArrow({ from, to, value, color }: { from: string; to: string; value: string; color: string }) {
  return (
    <div className="flex items-center gap-1 text-[8px] font-mono">
      <span className="text-terminal-text-faint">{from}</span>
      <svg width="20" height="8" className="flex-shrink-0">
        <line x1="0" y1="4" x2="15" y2="4" stroke={color} strokeWidth="1.2" />
        <polygon points="15,1 20,4 15,7" fill={color} />
      </svg>
      <span style={{ color }} className="font-bold">{value}</span>
      <svg width="20" height="8" className="flex-shrink-0">
        <line x1="0" y1="4" x2="15" y2="4" stroke={color} strokeWidth="1.2" />
        <polygon points="15,1 20,4 15,7" fill={color} />
      </svg>
      <span className="text-terminal-text-faint">{to}</span>
    </div>
  );
}

// ── Panel ──
function Panel({ title, children, className = "", cols = "" }: { title: string; children: React.ReactNode; className?: string; cols?: string }) {
  return (
    <div className={`border border-terminal-border rounded-sm bg-[#0a0f1a] flex flex-col overflow-hidden ${cols} ${className}`}>
      <div className="px-2 py-1 border-b border-terminal-border/60 flex items-center justify-between flex-shrink-0">
        <span className="text-[9px] font-semibold tracking-[0.12em] text-terminal-text-muted uppercase">{title}</span>
        <span className="text-[8px] text-terminal-text-faint font-mono">{ts()}</span>
      </div>
      <div className="flex-1 p-2 overflow-auto min-h-0">{children}</div>
    </div>
  );
}

// ── KPI Card ──
function KPI({ label, value, sub, color = C.textPrimary }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[7px] text-terminal-text-faint tracking-wider uppercase">{label}</span>
      <span className="text-[13px] font-bold font-mono tabular-nums" style={{ color }}>{value}</span>
      {sub && <span className="text-[7px] font-mono text-terminal-text-muted">{sub}</span>}
    </div>
  );
}

// ── Custom Tooltip ──
function CTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#0d1520] border border-terminal-border rounded px-2 py-1.5 shadow-xl">
      <div className="text-[8px] text-terminal-text-faint mb-1">{label}</div>
      {payload.map((p: any, i: number) => (
        <div key={i} className="flex items-center gap-1.5 text-[9px] font-mono">
          <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: p.color || p.stroke }} />
          <span className="text-terminal-text-muted">{p.name}:</span>
          <span style={{ color: p.color || p.stroke }} className="font-bold">{typeof p.value === "number" ? p.value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : p.value}</span>
        </div>
      ))}
    </div>
  );
}

const AXIS_STYLE = { fontSize: 8, fill: C.textFaint, fontFamily: "monospace" };

// ══════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ══════════════════════════════════════════════════════════════════════
export default function MoneyVelocityPage() {
  // ─── Engine API ─────────────────────────────────────
  const { data: fedBsApi } = useEngineQuery<{ walcl?: number; soma_treasuries?: number; soma_mbs?: number; reserves?: number; on_rrp?: number; tga?: number }>("/macro/fed-balance-sheet", { refetchInterval: 30000 });
  const { data: reservesApi } = useEngineQuery<{ fed_to_pd?: number; pd_to_gsib?: number; gsib_to_shadow?: number; shadow_to_market?: number; net_market_liquidity?: number; bottleneck?: string }>("/macro/reserves-flow", { refetchInterval: 30000 });
  const { data: velocityApi } = useEngineQuery<{ velocity?: number; velocity_change?: number; credit_impulse?: number; sofr_rate?: number; liquidity_score?: number }>("/macro/velocity", { refetchInterval: 15000 });
  const { data: liqScoreApi } = useEngineQuery<{ score?: number; regime?: string }>("/macro/liquidity-score", { refetchInterval: 15000 });
  const { data: drainApi } = useEngineQuery<{ warning_level?: number; triggers?: string[] }>("/macro/drain-warning", { refetchInterval: 30000 });
  const { data: sectorFlowApi } = useEngineQuery<{ sector_scores?: Record<string, number>; overweight?: string[]; underweight?: string[]; flow_regime?: string }>("/macro/sector-flows", { refetchInterval: 30000 });
  const { data: creditApi } = useEngineQuery<{ impulse?: number; regime?: string; z_score?: number }>("/macro/credit-impulse", { refetchInterval: 30000 });

  // ── State ──
  const [tick, setTick] = useState(0);
  const [timeframe, setTimeframe] = useState<"D" | "W" | "M" | "Q">("D");

  // Live tick
  useEffect(() => {
    const iv = setInterval(() => setTick((t) => t + 1), 4000);
    return () => clearInterval(iv);
  }, []);

  // ── Generate all data (memoised, refreshes each tick) ──
  const fedBs = useMemo(() => genFedBalanceSheet(90), [tick]);
  const reserves = useMemo(() => genReserves(90), [tick]);
  const netLiq = useMemo(() => genNetLiquidity(fedBs, reserves), [fedBs, reserves]);
  const rates = useMemo(() => genFrontEndRates(60), [tick]);
  const m2v = useMemo(() => genM2Velocity(36), [tick]);
  const creditImp = useMemo(() => genCreditImpulse(24), [tick]);
  const collateral = useMemo(() => genCollateralChain(), [tick]);
  const ctvPairs = useMemo(() => genCtvPairs(), [tick]);
  const sectorAbs = useMemo(() => genSectorAbsorption(), [tick]);
  const flowDist = useMemo(() => genFlowDist(), [tick]);
  const liqComp = useMemo(() => genLiquidityComposite(90), [tick]);
  const changes = useMemo(() => genChanges(), [tick]);

  // Derived — override with API data when available
  const latestNet = reservesApi?.net_market_liquidity ?? netLiq[netLiq.length - 1]?.net ?? 5998;
  const latestScore = liqScoreApi?.score != null ? liqScoreApi.score * 100 : (liqComp[liqComp.length - 1]?.score ?? 62);
  const latestVelocity = velocityApi?.velocity ?? m2v[m2v.length - 1]?.velocity ?? 1.149;
  const latestSofr = velocityApi?.sofr_rate ?? rates[rates.length - 1]?.sofr ?? 4.32;
  const latestTga = fedBsApi?.tga ?? reserves[reserves.length - 1]?.tga ?? 722;
  const latestOnRrp = fedBsApi?.on_rrp ?? reserves[reserves.length - 1]?.onRrp ?? 118;
  const latestBankRes = fedBsApi?.reserves ?? reserves[reserves.length - 1]?.bankRes ?? 3340;
  const latestFedBs = fedBsApi?.walcl ?? fedBs[fedBs.length - 1]?.total ?? 6820;
  const velocityRegime = latestVelocity > 1.18 ? "ACCELERATING" : latestVelocity < 1.10 ? "DECELERATING" : "STABLE";
  const liqRegime = liqScoreApi?.regime ?? (latestScore > 65 ? "EXPANSIONARY" : latestScore > 40 ? "NEUTRAL" : "CONTRACTIONARY");
  const sdrCtv = useMemo(() => {
    let s = 0; let w = 0;
    ctvPairs.filter((p) => p.sdr > 0).forEach((p) => { s += p.ctv * p.sdr; w += p.sdr; });
    return w > 0 ? s / w : 0;
  }, [ctvPairs]);

  const drainWarning = drainApi?.warning_level != null ? drainApi.warning_level >= 2 : (latestOnRrp < 50 && latestBankRes < 3000);

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* ── Top KPI Bar ── */}
      <div className="flex-shrink-0 px-3 py-1.5 border-b border-terminal-border bg-[#060a12] flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-1.5">
          <span className="text-[9px] text-terminal-text-faint tracking-wider">MONEY VELOCITY & US LIQUIDITY</span>
          <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded-sm ${liqRegime === "EXPANSIONARY" ? "bg-green-500/15 text-green-400" : liqRegime === "NEUTRAL" ? "bg-yellow-500/15 text-yellow-400" : "bg-red-500/15 text-red-400"}`}>
            {liqRegime}
          </span>
          {drainWarning && (
            <span className="text-[8px] font-bold px-1.5 py-0.5 rounded-sm bg-red-500/20 text-red-400 animate-pulse">DRAIN WARNING</span>
          )}
        </div>
        <div className="flex items-center gap-3 ml-auto text-[9px] font-mono tabular-nums">
          <KPI label="NET LIQ" value={fmtB(latestNet)} color={C.accent} />
          <KPI label="FED BS" value={fmtB(latestFedBs)} color={C.blue} />
          <KPI label="TGA" value={`$${latestTga.toFixed(0)}B`} color={C.warn} />
          <KPI label="ON-RRP" value={`$${latestOnRrp.toFixed(0)}B`} color={C.purple} />
          <KPI label="RESERVES" value={`$${(latestBankRes / 1000).toFixed(2)}T`} color={C.cyan} />
          <KPI label="SOFR" value={`${latestSofr.toFixed(3)}%`} color={C.orange} />
          <KPI label="M2V" value={latestVelocity.toFixed(4)} sub={velocityRegime} color={velocityRegime === "ACCELERATING" ? C.positive : velocityRegime === "DECELERATING" ? C.danger : C.warn} />
          <KPI label="LIQ SCORE" value={`${latestScore.toFixed(0)}/100`} color={latestScore > 65 ? C.positive : latestScore > 40 ? C.warn : C.danger} />
          <KPI label="SDR CtV" value={sdrCtv.toFixed(3)} color={sdrCtv > 0.3 ? C.positive : sdrCtv > 0 ? C.warn : C.danger} />
        </div>
      </div>

      {/* ── Main Grid ── */}
      <div className="flex-1 overflow-auto p-1.5 grid gap-1.5"
        style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr", gridTemplateRows: "1fr 1fr 1fr 1fr", minHeight: 0 }}
      >
        {/* ═══ R1C1: Fed Balance Sheet ═══ */}
        <Panel title="Fed Balance Sheet" cols="col-span-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={fedBs} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
              <CartesianGrid stroke={C.gridLine} strokeDasharray="2 4" />
              <XAxis dataKey="date" tick={AXIS_STYLE} interval={14} />
              <YAxis tick={AXIS_STYLE} domain={["auto", "auto"]} tickFormatter={(v: number) => `${(v / 1000).toFixed(1)}T`} width={38} />
              <Tooltip content={<CTooltip />} />
              <Area type="monotone" dataKey="other" stackId="1" stroke={C.orange} fill={C.orange + "30"} name="Other" />
              <Area type="monotone" dataKey="mbs" stackId="1" stroke={C.purple} fill={C.purple + "30"} name="MBS" />
              <Area type="monotone" dataKey="treasuries" stackId="1" stroke={C.blue} fill={C.blue + "30"} name="Treasuries" />
              <Line type="monotone" dataKey="total" stroke={C.accent} strokeWidth={1.5} dot={false} name="Total" />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>

        {/* ═══ R1C3: Net Liquidity ═══ */}
        <Panel title="Net Liquidity (Fed BS - TGA - ON-RRP)">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={netLiq} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
              <CartesianGrid stroke={C.gridLine} strokeDasharray="2 4" />
              <XAxis dataKey="date" tick={AXIS_STYLE} interval={14} />
              <YAxis tick={AXIS_STYLE} domain={["auto", "auto"]} tickFormatter={(v: number) => `${(v / 1000).toFixed(1)}T`} width={38} />
              <Tooltip content={<CTooltip />} />
              <Area type="monotone" dataKey="net" stroke={C.accent} fill={C.accent + "20"} name="Net Liquidity" />
              <ReferenceLine y={6000} stroke={C.warn + "60"} strokeDasharray="4 4" label={{ value: "Threshold", fill: C.warn, fontSize: 8, position: "right" }} />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>

        {/* ═══ R1C4: US Liquidity Composite Score ═══ */}
        <Panel title="US Liquidity Composite">
          <div className="flex flex-col h-full gap-1">
            <div className="flex items-center justify-center gap-4">
              <LiqGauge score={latestScore} label="COMPOSITE" />
              <div className="flex flex-col gap-1 text-[8px] font-mono">
                <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full" style={{ background: C.accent }} /><span className="text-terminal-text-faint">Velocity</span><span className="text-terminal-text-primary">{liqComp[liqComp.length - 1]?.vComp.toFixed(0)}</span></div>
                <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full" style={{ background: C.blue }} /><span className="text-terminal-text-faint">Credit</span><span className="text-terminal-text-primary">{liqComp[liqComp.length - 1]?.ciComp.toFixed(0)}</span></div>
                <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full" style={{ background: C.warn }} /><span className="text-terminal-text-faint">TED</span><span className="text-terminal-text-primary">{liqComp[liqComp.length - 1]?.tedComp.toFixed(0)}</span></div>
                <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full" style={{ background: C.purple }} /><span className="text-terminal-text-faint">VIX</span><span className="text-terminal-text-primary">{liqComp[liqComp.length - 1]?.vixComp.toFixed(0)}</span></div>
                <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full" style={{ background: C.cyan }} /><span className="text-terminal-text-faint">Y-Curve</span><span className="text-terminal-text-primary">{liqComp[liqComp.length - 1]?.ycComp.toFixed(0)}</span></div>
              </div>
            </div>
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={liqComp.slice(-30)} margin={{ top: 2, right: 2, left: 0, bottom: 0 }}>
                  <XAxis dataKey="date" tick={AXIS_STYLE} interval={6} />
                  <YAxis tick={AXIS_STYLE} domain={[0, 100]} width={24} />
                  <Tooltip content={<CTooltip />} />
                  <Area type="monotone" dataKey="ycComp" stackId="1" stroke={C.cyan} fill={C.cyan + "25"} name="Y-Curve" />
                  <Area type="monotone" dataKey="vixComp" stackId="1" stroke={C.purple} fill={C.purple + "25"} name="VIX" />
                  <Area type="monotone" dataKey="tedComp" stackId="1" stroke={C.warn} fill={C.warn + "25"} name="TED" />
                  <Area type="monotone" dataKey="ciComp" stackId="1" stroke={C.blue} fill={C.blue + "25"} name="Credit" />
                  <Area type="monotone" dataKey="vComp" stackId="1" stroke={C.accent} fill={C.accent + "25"} name="Velocity" />
                  <ReferenceLine y={65} stroke={C.positive + "50"} strokeDasharray="3 3" />
                  <ReferenceLine y={40} stroke={C.danger + "50"} strokeDasharray="3 3" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Panel>

        {/* ═══ R2C1: Reserves Distribution ═══ */}
        <Panel title="Reserve Distribution (TGA / ON-RRP / Bank Reserves)">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={reserves} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
              <CartesianGrid stroke={C.gridLine} strokeDasharray="2 4" />
              <XAxis dataKey="date" tick={AXIS_STYLE} interval={14} />
              <YAxis tick={AXIS_STYLE} domain={["auto", "auto"]} tickFormatter={(v: number) => `${v}B`} width={40} />
              <Tooltip content={<CTooltip />} />
              <Area type="monotone" dataKey="standingRepo" stackId="1" stroke={C.pink} fill={C.pink + "30"} name="Standing Repo" />
              <Area type="monotone" dataKey="onRrp" stackId="1" stroke={C.purple} fill={C.purple + "30"} name="ON-RRP" />
              <Area type="monotone" dataKey="tga" stackId="1" stroke={C.warn} fill={C.warn + "30"} name="TGA" />
              <Area type="monotone" dataKey="bankRes" stackId="1" stroke={C.cyan} fill={C.cyan + "30"} name="Bank Reserves" />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>

        {/* ═══ R2C2: Front-End Rates ═══ */}
        <Panel title="Front-End Rates (SOFR / EFFR / T-Bills / IORB)">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rates} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
              <CartesianGrid stroke={C.gridLine} strokeDasharray="2 4" />
              <XAxis dataKey="date" tick={AXIS_STYLE} interval={9} />
              <YAxis tick={AXIS_STYLE} domain={["auto", "auto"]} tickFormatter={(v: number) => `${v.toFixed(2)}%`} width={42} />
              <Tooltip content={<CTooltip />} />
              <Line type="monotone" dataKey="sofr" stroke={C.accent} strokeWidth={1.5} dot={false} name="SOFR" />
              <Line type="monotone" dataKey="effr" stroke={C.blue} strokeWidth={1.2} dot={false} name="EFFR" />
              <Line type="monotone" dataKey="tBill3m" stroke={C.warn} strokeWidth={1} dot={false} name="3M T-Bill" />
              <Line type="monotone" dataKey="tBill1m" stroke={C.orange} strokeWidth={1} dot={false} name="1M T-Bill" />
              <Line type="monotone" dataKey="repo" stroke={C.purple} strokeWidth={1} dot={false} name="Repo" />
              <Line type="monotone" dataKey="iorb" stroke={C.cyan} strokeWidth={1} dot={false} name="IORB" />
            </LineChart>
          </ResponsiveContainer>
        </Panel>

        {/* ═══ R2C3: M2 Velocity ═══ */}
        <Panel title="M2 Money Velocity (V = GDP / M2)">
          <div className="flex flex-col h-full gap-1">
            <div className="flex items-center gap-3 text-[8px] font-mono">
              <span className="text-terminal-text-faint">V =</span>
              <span className="text-terminal-text-primary font-bold text-[12px]">{latestVelocity.toFixed(4)}</span>
              <span className={`px-1 py-0.5 rounded-sm text-[7px] font-bold ${velocityRegime === "ACCELERATING" ? "bg-green-500/15 text-green-400" : velocityRegime === "DECELERATING" ? "bg-red-500/15 text-red-400" : "bg-yellow-500/15 text-yellow-400"}`}>{velocityRegime}</span>
              <span className="text-terminal-text-faint ml-auto">M2: ${(m2v[m2v.length - 1]?.m2 / 1000).toFixed(1)}T</span>
              <span className="text-terminal-text-faint">GDP: ${(m2v[m2v.length - 1]?.gdp / 1000).toFixed(1)}T</span>
            </div>
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={m2v} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke={C.gridLine} strokeDasharray="2 4" />
                  <XAxis dataKey="date" tick={AXIS_STYLE} interval={5} />
                  <YAxis yAxisId="v" tick={AXIS_STYLE} domain={["auto", "auto"]} tickFormatter={(v: number) => v.toFixed(2)} width={36} />
                  <YAxis yAxisId="m2" orientation="right" tick={AXIS_STYLE} domain={["auto", "auto"]} tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}T`} width={30} />
                  <Tooltip content={<CTooltip />} />
                  <Bar yAxisId="m2" dataKey="m2" fill={C.blue + "30"} name="M2 ($B)" barSize={6} />
                  <Line yAxisId="v" type="monotone" dataKey="velocity" stroke={C.accent} strokeWidth={2} dot={false} name="Velocity" />
                  <ReferenceLine yAxisId="v" y={1.12} stroke={C.warn + "50"} strokeDasharray="4 4" label={{ value: "Baseline", fill: C.warn, fontSize: 7, position: "left" }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Panel>

        {/* ═══ R2C4: Credit Impulse ═══ */}
        <Panel title="Credit Impulse & Bank Credit Growth">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={creditImp} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
              <CartesianGrid stroke={C.gridLine} strokeDasharray="2 4" />
              <XAxis dataKey="date" tick={AXIS_STYLE} interval={3} />
              <YAxis yAxisId="imp" tick={AXIS_STYLE} domain={["auto", "auto"]} width={30} />
              <YAxis yAxisId="credit" orientation="right" tick={AXIS_STYLE} domain={["auto", "auto"]} tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}T`} width={30} />
              <Tooltip content={<CTooltip />} />
              <ReferenceLine yAxisId="imp" y={0} stroke={C.textFaint} strokeDasharray="3 3" />
              <Bar yAxisId="imp" dataKey="impulse" name="Credit Impulse" barSize={8}>
                {creditImp.map((e, i) => (
                  <Cell key={i} fill={e.impulse >= 0 ? C.positive + "80" : C.danger + "80"} />
                ))}
              </Bar>
              <Line yAxisId="credit" type="monotone" dataKey="bankCredit" stroke={C.blue} strokeWidth={1.2} dot={false} name="Bank Credit ($B)" />
              <Line yAxisId="credit" type="monotone" dataKey="consumerCredit" stroke={C.purple} strokeWidth={1} dot={false} name="Consumer Credit ($B)" />
            </ComposedChart>
          </ResponsiveContainer>
        </Panel>

        {/* ═══ R3C1: Collateral Chain ═══ */}
        <Panel title="Collateral Chain (Reuse Rate / Haircut / Velocity)">
          <div className="flex flex-col h-full">
            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={collateral} layout="vertical" margin={{ top: 4, right: 8, left: 4, bottom: 0 }}>
                  <CartesianGrid stroke={C.gridLine} strokeDasharray="2 4" horizontal={false} />
                  <XAxis type="number" tick={AXIS_STYLE} />
                  <YAxis dataKey="tier" type="category" tick={AXIS_STYLE} width={58} />
                  <Tooltip content={<CTooltip />} />
                  <Bar dataKey="reuse" fill={C.accent + "70"} name="Reuse Rate" barSize={6} />
                  <Bar dataKey="velocity" fill={C.blue + "70"} name="Collateral Velocity" barSize={6} />
                  <Scatter dataKey="quality" fill={C.warn} name="Quality Score" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="flex gap-2 mt-1 text-[7px] font-mono text-terminal-text-faint flex-wrap">
              {collateral.map((c) => (
                <span key={c.tier} className="flex items-center gap-0.5">
                  <span className="text-terminal-text-muted">{c.tier}:</span>
                  <span style={{ color: c.quality > 70 ? C.positive : c.quality > 40 ? C.warn : C.danger }}>{c.haircut}% HC</span>
                </span>
              ))}
            </div>
          </div>
        </Panel>

        {/* ═══ R3C2: CtV Liquidity ═══ */}
        <Panel title="Carry-to-Volatility (CtV) Liquidity">
          <div className="flex flex-col h-full gap-1">
            <div className="flex items-center gap-3 text-[8px] font-mono">
              <span className="text-terminal-text-faint">SDR CtV:</span>
              <span className="text-[11px] font-bold" style={{ color: sdrCtv > 0.3 ? C.positive : sdrCtv > 0 ? C.warn : C.danger }}>{sdrCtv.toFixed(4)}</span>
              <span className="text-terminal-text-faint ml-auto">GATE OPEN:</span>
              <span className="text-terminal-text-primary">{ctvPairs.filter((p) => p.gate).length}/{ctvPairs.length}</span>
            </div>
            <div className="flex-1 overflow-auto min-h-0">
              <table className="w-full text-[8px] font-mono tabular-nums">
                <thead>
                  <tr className="text-terminal-text-faint border-b border-terminal-border/40">
                    <th className="text-left py-0.5 px-1">PAIR</th>
                    <th className="text-right px-1">CARRY</th>
                    <th className="text-right px-1">VOL</th>
                    <th className="text-right px-1">CtV</th>
                    <th className="text-center px-1">GATE</th>
                    <th className="text-right px-1">SDR WT</th>
                  </tr>
                </thead>
                <tbody>
                  {ctvPairs.map((p) => (
                    <tr key={p.pair} className="border-b border-terminal-border/20 hover:bg-white/[0.02]">
                      <td className="py-0.5 px-1 text-terminal-text-primary">{p.pair}</td>
                      <td className="text-right px-1" style={{ color: p.carry >= 0 ? C.positive : C.danger }}>{p.carry > 0 ? "+" : ""}{p.carry.toFixed(2)}%</td>
                      <td className="text-right px-1 text-terminal-text-muted">{p.vol.toFixed(1)}%</td>
                      <td className="text-right px-1 font-bold" style={{ color: p.ctv > 0.5 ? C.positive : p.ctv > 0 ? C.warn : C.danger }}>{p.ctv.toFixed(4)}</td>
                      <td className="text-center px-1">
                        <span className={`px-1 py-0.5 rounded-sm text-[7px] ${p.gate ? "bg-green-500/20 text-green-400" : "bg-red-500/15 text-red-400"}`}>{p.gate ? "OPEN" : "SHUT"}</span>
                      </td>
                      <td className="text-right px-1 text-terminal-text-faint">{p.sdr > 0 ? (p.sdr * 100).toFixed(1) + "%" : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Panel>

        {/* ═══ R3C3: Money Flow Distribution ═══ */}
        <Panel title="Money Flow Distribution">
          <div className="flex h-full items-center gap-2">
            <div className="w-1/2 h-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={flowDist} cx="50%" cy="50%" innerRadius="40%" outerRadius="75%" dataKey="value" nameKey="name" strokeWidth={0}>
                    {flowDist.map((e, i) => <Cell key={i} fill={e.color} />)}
                  </Pie>
                  <Tooltip content={<CTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex flex-col gap-1.5 text-[8px] font-mono">
              {flowDist.map((f) => (
                <div key={f.name} className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-sm flex-shrink-0" style={{ background: f.color }} />
                  <span className="text-terminal-text-muted w-16">{f.name}</span>
                  <span className="font-bold" style={{ color: f.color }}>{f.value.toFixed(1)}%</span>
                </div>
              ))}
              <div className="border-t border-terminal-border/40 pt-1 mt-0.5 flex flex-col gap-1">
                <FlowArrow from="Fed" to="Banks" value={`$${(latestBankRes / 100).toFixed(0)}B/d`} color={C.accent} />
                <FlowArrow from="TGA" to="Mkts" value={`$${(latestTga * 0.02).toFixed(0)}B/d`} color={C.warn} />
                <FlowArrow from="RRP" to="Bills" value={`$${(latestOnRrp * 0.05).toFixed(0)}B/d`} color={C.purple} />
              </div>
            </div>
          </div>
        </Panel>

        {/* ═══ R3C4: Sector Liquidity Absorption ═══ */}
        <Panel title="Sector Liquidity Absorption">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sectorAbs} layout="vertical" margin={{ top: 2, right: 8, left: 2, bottom: 0 }}>
              <CartesianGrid stroke={C.gridLine} strokeDasharray="2 4" horizontal={false} />
              <XAxis type="number" tick={AXIS_STYLE} />
              <YAxis dataKey="sector" type="category" tick={AXIS_STYLE} width={55} />
              <Tooltip content={<CTooltip />} />
              <Bar dataKey="absorption" name="Absorption Coeff" barSize={7}>
                {sectorAbs.map((e, i) => (
                  <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length] + "90"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>

        {/* ═══ R4: Changes Table (full width) ═══ */}
        <Panel title="Daily / Weekly / Monthly / Quarterly Changes" cols="col-span-4">
          <div className="overflow-auto h-full">
            <table className="w-full text-[9px] font-mono tabular-nums">
              <thead>
                <tr className="text-terminal-text-faint border-b border-terminal-border/60 sticky top-0 bg-[#0a0f1a]">
                  <th className="text-left py-1 px-2 w-[160px]">INDICATOR</th>
                  <th className="text-right px-2">CURRENT</th>
                  <th className="text-center px-1 w-[60px]">SPARK</th>
                  <th className="text-right px-2">1D CHG</th>
                  <th className="text-right px-2">1W CHG</th>
                  <th className="text-right px-2">1M CHG</th>
                  <th className="text-right px-2">1Q CHG</th>
                  <th className="text-right px-2">1D %</th>
                  <th className="text-right px-2">1W %</th>
                  <th className="text-right px-2">1M %</th>
                  <th className="text-right px-2">1Q %</th>
                </tr>
              </thead>
              <tbody>
                {changes.map((r, idx) => {
                  const spark = Array.from({ length: 20 }, (_, i) => r.current + (Math.random() - 0.5) * Math.abs(r.m1));
                  const chgColor = (v: number) => v >= 0 ? C.positive : C.danger;
                  const pct = (chg: number) => r.current !== 0 ? (chg / Math.abs(r.current)) * 100 : 0;
                  return (
                    <tr key={idx} className="border-b border-terminal-border/20 hover:bg-white/[0.02]">
                      <td className="py-1 px-2 text-terminal-text-primary font-medium">{r.label}</td>
                      <td className="text-right px-2 text-terminal-text-primary font-bold">
                        {r.unit === "$B" ? `$${r.current.toLocaleString()}B` : r.unit === "%" ? `${r.current.toFixed(3)}%` : r.unit === "/100" ? r.current.toFixed(1) : r.unit === "x" ? r.current.toFixed(4) : r.unit === "σ" ? `${r.current.toFixed(2)}σ` : r.unit === "bps" ? `${r.current.toFixed(0)}bps` : r.current.toFixed(1)}
                      </td>
                      <td className="text-center px-1"><Spark data={spark} color={C.accent} w={50} h={14} /></td>
                      <td className="text-right px-2" style={{ color: chgColor(r.d1) }}>{r.d1 >= 0 ? "+" : ""}{r.d1.toFixed(2)}</td>
                      <td className="text-right px-2" style={{ color: chgColor(r.w1) }}>{r.w1 >= 0 ? "+" : ""}{r.w1.toFixed(2)}</td>
                      <td className="text-right px-2" style={{ color: chgColor(r.m1) }}>{r.m1 >= 0 ? "+" : ""}{r.m1.toFixed(2)}</td>
                      <td className="text-right px-2" style={{ color: chgColor(r.q1) }}>{r.q1 >= 0 ? "+" : ""}{r.q1.toFixed(2)}</td>
                      <td className="text-right px-2" style={{ color: chgColor(r.d1) }}>{fmtPct(pct(r.d1))}</td>
                      <td className="text-right px-2" style={{ color: chgColor(r.w1) }}>{fmtPct(pct(r.w1))}</td>
                      <td className="text-right px-2" style={{ color: chgColor(r.m1) }}>{fmtPct(pct(r.m1))}</td>
                      <td className="text-right px-2" style={{ color: chgColor(r.q1) }}>{fmtPct(pct(r.q1))}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </div>
  );
}
