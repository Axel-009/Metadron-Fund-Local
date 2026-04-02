import { useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { useEngineQuery, type MacroSnapshot } from "@/hooks/use-engine-api";

// ═══════════ MOCK DATA ═══════════

const REGIME = { label:"TRANSITION", confidence: 67, prev:"BULL" };

const REGIME_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  BULL:       { bg:"#3fb95022", text:"#3fb950", border:"#3fb950" },
  BEAR:       { bg:"#f8514922", text:"#f85149", border:"#f85149" },
  TRANSITION: { bg:"#d2992222", text:"#d29922", border:"#d29922" },
  STRESS:     { bg:"#a855f722", text:"#a855f7", border:"#a855f7" },
  CRASH:      { bg:"#f8514944", text:"#f85149", border:"#f85149" },
};

const G10_COUNTRIES = [
  { flag:"🇺🇸", country:"United States", code:"US",  gdp: 2.8,  cpi: 3.1,  unemp: 3.9, rate: 5.25, y10: 4.58, currency:"USD", cxChg:  0.00, pmiMfg: 49.8, pmiSvc: 52.3 },
  { flag:"🇬🇧", country:"United Kingdom",code:"UK",  gdp: 0.6,  cpi: 3.8,  unemp: 4.2, rate: 5.25, y10: 4.14, currency:"GBP", cxChg: -0.28, pmiMfg: 47.4, pmiSvc: 53.1 },
  { flag:"🇪🇺", country:"Eurozone",      code:"EUR", gdp: 0.4,  cpi: 2.6,  unemp: 6.4, rate: 4.50, y10: 2.41, currency:"EUR", cxChg: -0.14, pmiMfg: 46.2, pmiSvc: 50.8 },
  { flag:"🇯🇵", country:"Japan",         code:"JP",  gdp: 0.1,  cpi: 2.4,  unemp: 2.6, rate: 0.10, y10: 0.82, currency:"JPY", cxChg:  0.61, pmiMfg: 48.2, pmiSvc: 51.4 },
  { flag:"🇨🇦", country:"Canada",        code:"CA",  gdp: 1.2,  cpi: 2.8,  unemp: 6.1, rate: 5.00, y10: 3.92, currency:"CAD", cxChg:  0.31, pmiMfg: 49.1, pmiSvc: 50.2 },
  { flag:"🇦🇺", country:"Australia",     code:"AU",  gdp: 1.5,  cpi: 3.4,  unemp: 4.1, rate: 4.35, y10: 4.38, currency:"AUD", cxChg:  0.19, pmiMfg: 47.8, pmiSvc: 52.9 },
  { flag:"🇳🇿", country:"New Zealand",   code:"NZ",  gdp: 0.3,  cpi: 4.0,  unemp: 4.0, rate: 5.50, y10: 5.02, currency:"NZD", cxChg: -0.22, pmiMfg: 46.1, pmiSvc: 49.8 },
  { flag:"🇨🇭", country:"Switzerland",   code:"CH",  gdp: 1.4,  cpi: 1.1,  unemp: 2.5, rate: 1.75, y10: 0.89, currency:"CHF", cxChg: -0.08, pmiMfg: 43.8, pmiSvc: 52.1 },
  { flag:"🇸🇪", country:"Sweden",        code:"SE",  gdp:-0.2,  cpi: 2.2,  unemp: 8.8, rate: 4.00, y10: 2.81, currency:"SEK", cxChg:  0.44, pmiMfg: 45.2, pmiSvc: 48.9 },
  { flag:"🇳🇴", country:"Norway",        code:"NO",  gdp: 1.0,  cpi: 3.6,  unemp: 2.1, rate: 4.50, y10: 4.12, currency:"NOK", cxChg:  0.27, pmiMfg: 51.2, pmiSvc: 53.8 },
];

const ECON_CALENDAR = [
  { date:"Apr 02", time:"08:30", event:"US NFP (Non-Farm Payrolls)",   country:"🇺🇸", prev:"303K", consensus:"215K",  importance:3 },
  { date:"Apr 02", time:"08:30", event:"US Unemployment Rate",         country:"🇺🇸", prev:"3.9%", consensus:"3.9%",  importance:2 },
  { date:"Apr 03", time:"10:00", event:"ISM Services PMI",             country:"🇺🇸", prev:"52.6", consensus:"52.8",  importance:2 },
  { date:"Apr 04", time:"08:30", event:"US CPI YoY",                   country:"🇺🇸", prev:"3.1%", consensus:"3.0%",  importance:3 },
  { date:"Apr 04", time:"08:30", event:"US Core CPI MoM",              country:"🇺🇸", prev:"0.4%", consensus:"0.3%",  importance:3 },
  { date:"Apr 07", time:"04:00", event:"ECB Rate Decision",            country:"🇪🇺", prev:"4.50%","consensus":"4.25%",importance:3 },
  { date:"Apr 08", time:"09:30", event:"UK GDP MoM",                   country:"🇬🇧", prev:"0.2%", consensus:"0.2%",  importance:2 },
  { date:"Apr 10", time:"08:30", event:"US PPI MoM",                   country:"🇺🇸", prev:"0.3%", consensus:"0.2%",  importance:2 },
  { date:"Apr 11", time:"00:50", event:"BOJ Monetary Policy Minutes",  country:"🇯🇵", prev:"—",    consensus:"—",     importance:2 },
  { date:"Apr 14", time:"08:30", event:"Canada CPI YoY",               country:"🇨🇦", prev:"2.8%", consensus:"2.9%",  importance:2 },
  { date:"Apr 16", time:"09:00", event:"Eurozone CPI Flash",           country:"🇪🇺", prev:"2.6%", consensus:"2.4%",  importance:2 },
  { date:"Apr 17", time:"08:30", event:"US Retail Sales MoM",          country:"🇺🇸", prev:"0.6%", consensus:"0.4%",  importance:2 },
];

const MACRO_NEWS = [
  { ts:"10:41 EST", src:"Reuters",   headline:"Fed Chair Powell signals higher-for-longer as inflation remains sticky above 3%",   sentiment:"hawkish" },
  { ts:"10:22 EST", src:"Bloomberg", headline:"ECB policymakers debate April cut timing amid diverging inflation data across zone", sentiment:"dovish"  },
  { ts:"09:58 EST", src:"FT",        headline:"NFP expected to show moderation as hiring slows in key service sectors",            sentiment:"neutral"  },
  { ts:"09:31 EST", src:"WSJ",       headline:"BOJ hints at rate path normalization, yen strengthens against dollar",              sentiment:"hawkish" },
  { ts:"09:14 EST", src:"Reuters",   headline:"US 2s10s inversion narrowing — yield curve steepening signals growth optimism",    sentiment:"neutral"  },
  { ts:"08:47 EST", src:"Bloomberg", headline:"China's deflationary pressure spills over to global goods prices, PBOC on hold",   sentiment:"dovish"  },
  { ts:"08:30 EST", src:"CNBC",      headline:"Initial jobless claims come in below expectations, labor market still resilient",   sentiment:"hawkish" },
  { ts:"07:55 EST", src:"FT",        headline:"Eurozone PMI composite rises to 50.8, barely expansionary — recovery fragile",     sentiment:"neutral"  },
  { ts:"07:22 EST", src:"Reuters",   headline:"Oil prices fall as OPEC+ signals production discipline review for Q3 2026",        sentiment:"neutral"  },
  { ts:"06:48 EST", src:"Bloomberg", headline:"Australia RBA holds steady at 4.35%, governor warns against premature easing",     sentiment:"hawkish" },
];

const SENTIMENT_COLORS: Record<string, string> = {
  hawkish:"#f85149",
  dovish:  "#3fb950",
  neutral: "#d29922",
};

const LIQUIDITY_METRICS = [
  { label:"Fed Balance Sheet",   value:"$7.44T",  chg:"-$28B",  dir:"down" },
  { label:"M2 Money Supply YoY", value:"+1.4%",   chg:"vs -3.8%",dir:"up"  },
  { label:"SOFR Rate",           value:"5.31%",   chg:"+2bps",  dir:"up"   },
  { label:"TED Spread",          value:"22.4bps", chg:"-1.2bps",dir:"down" },
  { label:"Net Liquidity Est",   value:"$5.82T",  chg:"-$12B",  dir:"down" },
];

function genChartData(base: number, vol: number, drift = 0, days = 30) {
  const data: { day: number; val: number }[] = [];
  let v = base;
  for (let i = 0; i < days; i++) {
    v = v + (Math.random() - 0.5) * vol + drift;
    data.push({ day: i + 1, val: +v.toFixed(2) });
  }
  return data;
}

const TOOLTIP_STYLE = {
  backgroundColor: "#0d1117",
  border: "1px solid #21262d",
  borderRadius: 4,
  fontSize: 10,
  color: "#e6edf3",
  padding: "4px 8px",
};

export default function MacroDashboard() {
  // ─── Engine API ─────────────────────────────────────
  const { data: macroSnap } = useEngineQuery<MacroSnapshot>("/macro/snapshot", { refetchInterval: 15000 });
  const { data: velocityData } = useEngineQuery<{ velocity?: number; credit_impulse?: number; sofr_rate?: number; ted_spread?: number; liquidity_score?: number }>("/macro/velocity", { refetchInterval: 30000 });
  const { data: yieldCurve } = useEngineQuery<Record<string, number>>("/macro/yield-curve", { refetchInterval: 30000 });
  const { data: tensionData } = useEngineQuery<{ tension_score?: number; stance?: string }>("/macro/monetary-tension", { refetchInterval: 60000 });
  const { data: newsData } = useEngineQuery<{ news: Array<Record<string, string>> }>("/macro/news", { refetchInterval: 60000 });
  const { data: calendarData } = useEngineQuery<{ events: Array<Record<string, string>> }>("/macro/calendar", { refetchInterval: 300000 });
  const { data: g10Data } = useEngineQuery<{ countries: Array<Record<string, string | number | null>> }>("/macro/g10", { refetchInterval: 120000 });

  // Override news from API when available
  const macroNews = useMemo(() => {
    if (!newsData?.news?.length) return MACRO_NEWS;
    return newsData.news.slice(0, 10).map((n) => ({
      ts: n.date || n.published || "",
      src: n.source || n.provider || "OpenBB",
      headline: n.title || n.headline || "",
      sentiment: "neutral" as const,
    }));
  }, [newsData]);

  // G10 countries from FRED API
  const g10Countries = useMemo(() => {
    if (!g10Data?.countries?.length) return G10_COUNTRIES;
    return g10Data.countries.map((c) => ({
      flag: String(c.flag || ""),
      country: String(c.country || ""),
      code: String(c.code || ""),
      gdp: Number(c.gdp ?? 0),
      cpi: Number(c.cpi ?? 0),
      unemp: Number(c.unemp ?? 0),
      rate: Number(c.rate ?? 0),
      y10: Number(c.y10 ?? 0),
      currency: String(c.currency || ""),
      cxChg: 0,
      pmiMfg: 0,
      pmiSvc: 0,
    }));
  }, [g10Data]);

  // Economic calendar from OpenBB
  const econCalendar = useMemo(() => {
    if (!calendarData?.events?.length) return ECON_CALENDAR;
    return calendarData.events.slice(0, 12).map((e) => ({
      date: String(e.date || ""),
      time: String(e.time || ""),
      event: String(e.event || e.name || ""),
      country: String(e.country || ""),
      prev: String(e.previous || e.prev || "—"),
      consensus: String(e.consensus || e.forecast || "—"),
      importance: Number(e.importance || 2),
    }));
  }, [calendarData]);

  // Override regime from API
  const apiRegime = macroSnap?.regime?.toUpperCase() || REGIME.label;
  const regime = { label: apiRegime, confidence: macroSnap ? Math.round((macroSnap.gmtf_score || 0.67) * 100) : REGIME.confidence, prev: REGIME.prev };

  // Override liquidity metrics from API
  const liqMetrics = useMemo(() => {
    if (!velocityData) return LIQUIDITY_METRICS;
    return [
      { label: "Fed Balance Sheet", value: velocityData.liquidity_score != null ? `Score: ${(velocityData.liquidity_score).toFixed(1)}` : LIQUIDITY_METRICS[0].value, chg: "", dir: "up" },
      { label: "Money Velocity", value: velocityData.velocity != null ? `V=${(velocityData.velocity).toFixed(3)}` : LIQUIDITY_METRICS[1].value, chg: "", dir: (velocityData.velocity || 0) > 1.15 ? "up" : "down" },
      { label: "SOFR Rate", value: velocityData.sofr_rate != null ? `${velocityData.sofr_rate.toFixed(2)}%` : LIQUIDITY_METRICS[2].value, chg: "", dir: "up" },
      { label: "TED Spread", value: velocityData.ted_spread != null ? `${(velocityData.ted_spread * 100).toFixed(1)}bps` : LIQUIDITY_METRICS[3].value, chg: "", dir: "down" },
      { label: "Credit Impulse", value: velocityData.credit_impulse != null ? `${velocityData.credit_impulse.toFixed(2)}` : LIQUIDITY_METRICS[4].value, chg: "", dir: (velocityData.credit_impulse || 0) > 0 ? "up" : "down" },
    ];
  }, [velocityData]);

  const spreadData = useMemo(() => genChartData(macroSnap?.yield_spread ?? -0.38, 0.06, 0.004, 30), [macroSnap]);
  const vixData    = useMemo(() => genChartData(macroSnap?.vix ?? 18.4, 1.2, -0.02, 30), [macroSnap]);
  const dxyData    = useMemo(() => genChartData(104.2, 0.3, 0.01, 30), []);

  const r = REGIME_COLORS[regime.label] || REGIME_COLORS["TRANSITION"];

  return (
    <div className="h-full flex flex-col gap-1 p-1 overflow-hidden">

      {/* ─── Regime Banner (full width) ─── */}
      <div className="flex-shrink-0 terminal-panel flex items-center justify-between px-3 py-1" style={{ border: `1px solid ${r.border}`, backgroundColor: r.bg }}>
        <div className="flex items-center gap-3">
          <span className="text-[9px] text-terminal-text-muted tracking-widest">MACRO REGIME</span>
          <span className="text-[18px] font-mono font-bold tracking-widest" style={{ color: r.text }}>{regime.label}</span>
          <span className="text-[10px] font-mono text-terminal-text-muted">CONFIDENCE: <span style={{ color: r.text }}>{regime.confidence}%</span></span>
          <span className="text-[9px] text-terminal-text-faint">PREV: {regime.prev}</span>
        </div>
        <div className="flex items-center gap-3">
          {Object.entries(REGIME_COLORS).map(([label, c]) => (
            <div key={label} className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: c.text, opacity: label === regime.label ? 1 : 0.3 }} />
              <span className="text-[8px] font-mono" style={{ color: label === regime.label ? c.text : "#555" }}>{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ─── Main: G10 + Charts | Economic Calendar ─── */}
      <div className="flex-1 min-h-0">
        <ResizableDashboard defaultSizes={[78, 22]} minSizes={[50, 15]}>
          {/* Left: G10 Table + 3 Charts + News/Liquidity */}
          <div className="h-full flex flex-col gap-1 overflow-hidden">
      <DashboardPanel title="G10 ECONOMIC INDICATORS" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[10px] font-mono border-collapse">
            <thead className="sticky top-0 bg-terminal-surface z-10">
              <tr className="border-b border-terminal-border">
                <th className="text-terminal-text-muted text-[9px] tracking-widest text-left px-2 py-1">COUNTRY</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">GDP%</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">CPI%</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">UNEMP%</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">RATE%</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">10Y%</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">FX</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">FX CHG</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">PMI MFG</th>
                <th className="text-terminal-text-muted text-[9px] tracking-widest px-1 py-1 text-right">PMI SVC</th>
              </tr>
            </thead>
            <tbody>
              {g10Countries.map((c, i) => (
                <tr key={c.code} className={`border-b border-terminal-border hover:bg-[#161b22] transition-colors ${i % 2 === 0 ? "" : "bg-[#0d1117]"}`}>
                  <td className="px-2 py-0.5 text-terminal-text-primary">
                    <span className="mr-1">{c.flag}</span>
                    <span className="text-terminal-accent font-bold">{c.code}</span>
                    <span className="text-terminal-text-muted ml-1 text-[9px]">{c.country}</span>
                  </td>
                  <td className={`px-1 py-0.5 text-right font-bold ${c.gdp >= 1.5 ? "text-terminal-positive" : c.gdp >= 0 ? "text-terminal-warning" : "text-terminal-negative"}`}>
                    {c.gdp >= 0 ? "+" : ""}{c.gdp.toFixed(1)}%
                  </td>
                  <td className={`px-1 py-0.5 text-right ${c.cpi > 4 ? "text-terminal-negative" : c.cpi > 2.5 ? "text-terminal-warning" : "text-terminal-positive"}`}>
                    {c.cpi.toFixed(1)}%
                  </td>
                  <td className={`px-1 py-0.5 text-right ${c.unemp > 6 ? "text-terminal-negative" : c.unemp > 4 ? "text-terminal-warning" : "text-terminal-positive"}`}>
                    {c.unemp.toFixed(1)}%
                  </td>
                  <td className="px-1 py-0.5 text-terminal-accent text-right font-bold">{c.rate.toFixed(2)}%</td>
                  <td className="px-1 py-0.5 text-terminal-text-primary text-right">{c.y10.toFixed(2)}%</td>
                  <td className="px-1 py-0.5 text-terminal-text-muted text-right">{c.currency}</td>
                  <td className={`px-1 py-0.5 text-right ${c.cxChg >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {c.cxChg >= 0 ? "+" : ""}{c.cxChg.toFixed(2)}%
                  </td>
                  <td className={`px-1 py-0.5 text-right font-bold ${c.pmiMfg >= 50 ? "text-terminal-positive" : c.pmiMfg >= 48 ? "text-terminal-warning" : "text-terminal-negative"}`}>
                    {c.pmiMfg.toFixed(1)}
                  </td>
                  <td className={`px-1 py-0.5 text-right font-bold ${c.pmiSvc >= 50 ? "text-terminal-positive" : c.pmiSvc >= 48 ? "text-terminal-warning" : "text-terminal-negative"}`}>
                    {c.pmiSvc.toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>

          </div>

          {/* Right: 3 Charts container */}
          <div className="flex flex-col gap-1 h-full overflow-auto">
            <DashboardPanel title="2s10s YIELD SPREAD (30D)" className="flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={spreadData} margin={{ top: 4, right: 6, left: -20, bottom: 0 }}>
                  <XAxis dataKey="day" tick={{ fontSize: 8, fill: "#8b949e" }} tickFormatter={v => `D${v}`} />
                  <YAxis domain={[-0.6, -0.1]} tickFormatter={v => `${v.toFixed(2)}%`} tick={{ fontSize: 8, fill: "#8b949e" }} />
                  <ReferenceLine y={0} stroke="#555" strokeDasharray="3 3" />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(3)}%`, "2s10s"]} />
                  <Line type="monotone" dataKey="val" stroke="#d29922" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </DashboardPanel>
            <DashboardPanel title="VIX INDEX (30D)" className="flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={vixData} margin={{ top: 4, right: 6, left: -20, bottom: 0 }}>
                  <XAxis dataKey="day" tick={{ fontSize: 8, fill: "#8b949e" }} tickFormatter={v => `D${v}`} />
                  <YAxis domain={[13, 26]} tick={{ fontSize: 8, fill: "#8b949e" }} />
                  <ReferenceLine y={20} stroke="#d29922" strokeDasharray="3 3" label={{ value: "20", position: "right", fontSize: 8, fill: "#d29922" }} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [v.toFixed(2), "VIX"]} />
                  <Line type="monotone" dataKey="val" stroke="#a855f7" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </DashboardPanel>
            <DashboardPanel title="DXY DOLLAR INDEX (30D)" className="flex-1">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dxyData} margin={{ top: 4, right: 6, left: -20, bottom: 0 }}>
                  <XAxis dataKey="day" tick={{ fontSize: 8, fill: "#8b949e" }} tickFormatter={v => `D${v}`} />
                  <YAxis domain={[102, 107]} tick={{ fontSize: 8, fill: "#8b949e" }} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [v.toFixed(2), "DXY"]} />
                  <Line type="monotone" dataKey="val" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </DashboardPanel>
          </div>
        </ResizableDashboard>
      </div>

      {/* ─── Bottom Row: News + Liquidity (resizable) ─── */}
      <div className="flex-shrink-0 h-40">
        <ResizableDashboard defaultSizes={[60, 40]} minSizes={[30, 25]}>
          <DashboardPanel title="MACRO NEWS FEED" noPadding>
            <div className="overflow-auto h-full">
              {macroNews.map((n, i) => (
                <div key={i} className={`px-2 py-1 border-b border-terminal-border flex items-start gap-2 hover:bg-[#161b22] ${i % 2 === 0 ? "" : "bg-[#0d1117]"}`}>
                  <span className="text-[8px] font-mono text-terminal-text-faint w-16 flex-shrink-0">{n.ts}</span>
                  <span className="text-[8px] font-mono text-terminal-text-muted w-14 flex-shrink-0">{n.src}</span>
                  <span className="text-[9px] text-terminal-text-primary flex-1 leading-tight">{n.headline}</span>
                  <span className="text-[8px] font-mono px-1 rounded flex-shrink-0" style={{ backgroundColor: SENTIMENT_COLORS[n.sentiment] + "22", color: SENTIMENT_COLORS[n.sentiment] }}>
                    {n.sentiment}
                  </span>
                </div>
              ))}
            </div>
          </DashboardPanel>

          <DashboardPanel title="MONEY VELOCITY & LIQUIDITY">
            <div className="grid grid-cols-1 gap-2">
              {liqMetrics.map(m => (
                <div key={m.label} className="flex items-center justify-between border-b border-terminal-border pb-1">
                  <span className="text-[9px] text-terminal-text-muted">{m.label}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-[12px] font-mono font-bold text-terminal-accent">{m.value}</span>
                    <span className={`text-[9px] font-mono ${m.dir === "up" ? "text-terminal-positive" : "text-terminal-negative"}`}>{m.chg}</span>
                  </div>
                </div>
              ))}
              <div className="text-[8px] text-terminal-text-faint mt-1">
                GMTF: Global Monetary Tension Framework | V=GDP/M2 | SDR: USD 43.38% EUR 29.31% CNY 12.28% JPY 7.59% GBP 7.44%
              </div>
            </div>
          </DashboardPanel>
        </ResizableDashboard>
      </div>
    </div>
  );
}
