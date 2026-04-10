import { useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { useEngineQuery, type MacroSnapshot } from "@/hooks/use-engine-api";

const REGIME_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  BULL:       { bg:"#3fb95022", text:"#3fb950", border:"#3fb950" },
  BEAR:       { bg:"#f8514922", text:"#f85149", border:"#f85149" },
  TRANSITION: { bg:"#d2992222", text:"#d29922", border:"#d29922" },
  STRESS:     { bg:"#a855f722", text:"#a855f7", border:"#a855f7" },
  CRASH:      { bg:"#f8514944", text:"#f85149", border:"#f85149" },
};

const SENTIMENT_COLORS: Record<string, string> = {
  hawkish:"#f85149",
  dovish:  "#3fb950",
  neutral: "#d29922",
};

const TOOLTIP_STYLE = {
  backgroundColor: "#0d1117",
  border: "1px solid #21262d",
  borderRadius: 4,
  fontSize: 10,
  color: "#e6edf3",
  padding: "4px 8px",
};

type ChartPoint = { day: number; val: number };

export default function MacroDashboard() {
  // ─── Engine API ─────────────────────────────────────
  const { data: macroSnap } = useEngineQuery<MacroSnapshot>("/macro/snapshot", { refetchInterval: 15000 });
  const { data: velocityData } = useEngineQuery<{ velocity?: number; credit_impulse?: number; sofr_rate?: number; ted_spread?: number; liquidity_score?: number }>("/macro/velocity", { refetchInterval: 30000 });
  const { data: yieldCurve } = useEngineQuery<Record<string, number>>("/macro/yield-curve", { refetchInterval: 30000 });
  const { data: tensionData } = useEngineQuery<{ tension_score?: number; stance?: string }>("/macro/monetary-tension", { refetchInterval: 60000 });
  const { data: newsData } = useEngineQuery<{ news: Array<Record<string, string>> }>("/macro/news", { refetchInterval: 60000 });
  const { data: calendarData } = useEngineQuery<{ events: Array<Record<string, string>> }>("/macro/calendar", { refetchInterval: 300000 });
  const { data: g10Data } = useEngineQuery<{ countries: Array<Record<string, string | number | null>> }>("/macro/g10", { refetchInterval: 120000 });

  // ─── Historical time series from FRED ───────────────
  const { data: spreadHistData } = useEngineQuery<{ data: ChartPoint[] }>("/macro/spread-history", { refetchInterval: 60000 });
  const { data: vixHistData } = useEngineQuery<{ data: ChartPoint[] }>("/macro/vix-history", { refetchInterval: 60000 });
  const { data: dxyHistData } = useEngineQuery<{ data: ChartPoint[] }>("/macro/dxy-history", { refetchInterval: 60000 });

  // News from API
  const macroNews = useMemo(() => {
    if (!newsData?.news?.length) return [];
    return newsData.news.slice(0, 10).map((n) => ({
      ts: n.date || n.published || "",
      src: n.source || n.provider || "OpenBB",
      headline: n.title || n.headline || "",
      sentiment: "neutral" as const,
    }));
  }, [newsData]);

  // G10 countries from FRED API
  const g10Countries = useMemo(() => {
    if (!g10Data?.countries?.length) return [];
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
    if (!calendarData?.events?.length) return [];
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

  // Regime from API
  const apiRegime = macroSnap?.regime?.toUpperCase() || "UNKNOWN";
  const regime = { label: apiRegime, confidence: macroSnap ? Math.round((macroSnap.gmtf_score || 0.67) * 100) : 0, prev: "—" };

  // Liquidity metrics from velocity API
  const liqMetrics = useMemo(() => {
    if (!velocityData) return [];
    return [
      { label: "Fed Balance Sheet", value: velocityData.liquidity_score != null ? `Score: ${(velocityData.liquidity_score).toFixed(1)}` : "—", chg: "", dir: "up" },
      { label: "Money Velocity", value: velocityData.velocity != null ? `V=${(velocityData.velocity).toFixed(3)}` : "—", chg: "", dir: (velocityData.velocity || 0) > 1.15 ? "up" : "down" },
      { label: "SOFR Rate", value: velocityData.sofr_rate != null ? `${velocityData.sofr_rate.toFixed(2)}%` : "—", chg: "", dir: "up" },
      { label: "TED Spread", value: velocityData.ted_spread != null ? `${(velocityData.ted_spread * 100).toFixed(1)}bps` : "—", chg: "", dir: "down" },
      { label: "Credit Impulse", value: velocityData.credit_impulse != null ? `${velocityData.credit_impulse.toFixed(2)}` : "—", chg: "", dir: (velocityData.credit_impulse || 0) > 0 ? "up" : "down" },
    ];
  }, [velocityData]);

  // Historical chart data from FRED
  const spreadData = useMemo(() => {
    const d = spreadHistData?.data;
    if (d && d.length > 0) return d.map((p, i) => ({ day: i + 1, val: p.val }));
    return [];
  }, [spreadHistData]);

  const vixData = useMemo(() => {
    const d = vixHistData?.data;
    if (d && d.length > 0) return d.map((p, i) => ({ day: i + 1, val: p.val }));
    return [];
  }, [vixHistData]);

  const dxyData = useMemo(() => {
    const d = dxyHistData?.data;
    if (d && d.length > 0) return d.map((p, i) => ({ day: i + 1, val: p.val }));
    return [];
  }, [dxyHistData]);

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
          {g10Countries.length === 0 ? (
            <div className="flex items-center justify-center h-full text-terminal-text-muted text-[11px] font-mono">Loading G10 data from FRED...</div>
          ) : (
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
          )}
        </div>
      </DashboardPanel>

          </div>

          {/* Right: 3 Charts container */}
          <div className="flex flex-col gap-1 h-full overflow-auto">
            <DashboardPanel title="2s10s YIELD SPREAD (30D)" className="flex-1">
              {spreadData.length === 0 ? (
                <div className="flex items-center justify-center h-full text-terminal-text-muted text-[10px] font-mono">Loading spread data...</div>
              ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={spreadData} margin={{ top: 4, right: 6, left: -20, bottom: 0 }}>
                  <XAxis dataKey="day" tick={{ fontSize: 8, fill: "#8b949e" }} tickFormatter={(v: number) => `D${v}`} />
                  <YAxis domain={["auto", "auto"]} tickFormatter={(v: number) => `${v.toFixed(2)}%`} tick={{ fontSize: 8, fill: "#8b949e" }} />
                  <ReferenceLine y={0} stroke="#555" strokeDasharray="3 3" />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(3)}%`, "2s10s"]} />
                  <Line type="monotone" dataKey="val" stroke="#d29922" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              )}
            </DashboardPanel>
            <DashboardPanel title="VIX INDEX (30D)" className="flex-1">
              {vixData.length === 0 ? (
                <div className="flex items-center justify-center h-full text-terminal-text-muted text-[10px] font-mono">Loading VIX data...</div>
              ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={vixData} margin={{ top: 4, right: 6, left: -20, bottom: 0 }}>
                  <XAxis dataKey="day" tick={{ fontSize: 8, fill: "#8b949e" }} tickFormatter={(v: number) => `D${v}`} />
                  <YAxis domain={["auto", "auto"]} tick={{ fontSize: 8, fill: "#8b949e" }} />
                  <ReferenceLine y={20} stroke="#d29922" strokeDasharray="3 3" label={{ value: "20", position: "right", fontSize: 8, fill: "#d29922" }} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [v.toFixed(2), "VIX"]} />
                  <Line type="monotone" dataKey="val" stroke="#a855f7" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              )}
            </DashboardPanel>
            <DashboardPanel title="DXY DOLLAR INDEX (30D)" className="flex-1">
              {dxyData.length === 0 ? (
                <div className="flex items-center justify-center h-full text-terminal-text-muted text-[10px] font-mono">Loading DXY data...</div>
              ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dxyData} margin={{ top: 4, right: 6, left: -20, bottom: 0 }}>
                  <XAxis dataKey="day" tick={{ fontSize: 8, fill: "#8b949e" }} tickFormatter={(v: number) => `D${v}`} />
                  <YAxis domain={["auto", "auto"]} tick={{ fontSize: 8, fill: "#8b949e" }} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v: number) => [v.toFixed(2), "DXY"]} />
                  <Line type="monotone" dataKey="val" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              )}
            </DashboardPanel>
          </div>
        </ResizableDashboard>
      </div>

      {/* ─── Bottom Row: News + Liquidity (resizable) ─── */}
      <div className="flex-shrink-0 h-40">
        <ResizableDashboard defaultSizes={[60, 40]} minSizes={[30, 25]}>
          <DashboardPanel title="MACRO NEWS FEED" noPadding>
            <div className="overflow-auto h-full">
              {macroNews.length === 0 ? (
                <div className="flex items-center justify-center h-full text-terminal-text-muted text-[10px] font-mono">Loading news...</div>
              ) : (
              macroNews.map((n, i) => (
                <div key={i} className={`px-2 py-1 border-b border-terminal-border flex items-start gap-2 hover:bg-[#161b22] ${i % 2 === 0 ? "" : "bg-[#0d1117]"}`}>
                  <span className="text-[8px] font-mono text-terminal-text-faint w-16 flex-shrink-0">{n.ts}</span>
                  <span className="text-[8px] font-mono text-terminal-text-muted w-14 flex-shrink-0">{n.src}</span>
                  <span className="text-[9px] text-terminal-text-primary flex-1 leading-tight">{n.headline}</span>
                  <span className="text-[8px] font-mono px-1 rounded flex-shrink-0" style={{ backgroundColor: SENTIMENT_COLORS[n.sentiment] + "22", color: SENTIMENT_COLORS[n.sentiment] }}>
                    {n.sentiment}
                  </span>
                </div>
              ))
              )}
            </div>
          </DashboardPanel>

          <DashboardPanel title="MONEY VELOCITY & LIQUIDITY">
            <div className="grid grid-cols-1 gap-2">
              {liqMetrics.length === 0 ? (
                <div className="flex items-center justify-center text-terminal-text-muted text-[10px] font-mono py-4">Loading velocity data...</div>
              ) : (
              liqMetrics.map(m => (
                <div key={m.label} className="flex items-center justify-between border-b border-terminal-border pb-1">
                  <span className="text-[9px] text-terminal-text-muted">{m.label}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-[12px] font-mono font-bold text-terminal-accent">{m.value}</span>
                    <span className={`text-[9px] font-mono ${m.dir === "up" ? "text-terminal-positive" : "text-terminal-negative"}`}>{m.chg}</span>
                  </div>
                </div>
              ))
              )}
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
