import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import { useState, useEffect, useRef, useMemo } from "react";
import { useEngineQuery, type MarketWrapData, type MacroSnapshot } from "@/hooks/use-engine-api";

const SECTORS = [
  { name: "Technology", daily: 1.8, weekly: 3.2, monthly: 5.4, color: "#00d4aa" },
  { name: "Healthcare", daily: 0.6, weekly: 1.1, monthly: 2.3, color: "#3fb950" },
  { name: "Financials", daily: -0.3, weekly: 0.8, monthly: 1.9, color: "#58a6ff" },
  { name: "Energy", daily: -1.2, weekly: -2.1, monthly: -3.8, color: "#f85149" },
  { name: "Consumer Disc.", daily: 0.9, weekly: 1.5, monthly: 3.1, color: "#4ecdc4" },
  { name: "Consumer Stap.", daily: 0.2, weekly: 0.4, monthly: 1.0, color: "#d29922" },
  { name: "Industrials", daily: 0.4, weekly: 0.9, monthly: 2.0, color: "#bc8cff" },
  { name: "Materials", daily: -0.5, weekly: -0.8, monthly: -1.2, color: "#f0883e" },
  { name: "Utilities", daily: -0.1, weekly: 0.3, monthly: 0.7, color: "#7d8590" },
  { name: "Real Estate", daily: -0.7, weekly: -1.3, monthly: -2.1, color: "#da3633" },
  { name: "Comm. Svcs.", daily: 1.2, weekly: 2.4, monthly: 4.1, color: "#58a6ff" },
];

const SCENARIOS = [
  { label: "BASE", prob: "60%", desc: "Soft landing, gradual disinflation. Fed cuts 2x in 2026. SPX +8-12% YoY.", color: "#00d4aa" },
  { label: "BULL", prob: "25%", desc: "AI-driven productivity boom. Strong earnings revisions upward. SPX +18-22%.", color: "#3fb950" },
  { label: "BEAR", prob: "15%", desc: "Recession triggered by credit event. Earnings -15%. SPX -20-25%.", color: "#f85149" },
];

const NEWS = [
  { time: "14:32", headline: "Fed minutes show split on rate path, dovish lean", source: "FOMC" },
  { time: "13:45", headline: "AAPL announces $110B buyback, largest in history", source: "Earnings" },
  { time: "12:20", headline: "China PMI below 50 for third consecutive month", source: "Macro" },
  { time: "11:05", headline: "10Y Treasury yield breaks above 4.35% resistance", source: "Rates" },
  { time: "10:15", headline: "NVDA guidance beats by 12%, AI capex acceleration", source: "Earnings" },
  { time: "09:30", headline: "Initial claims at 210K, labor market remains tight", source: "Labor" },
];

const FED_EVENTS = [
  { date: "Apr 15", event: "Fed Waller speech on inflation outlook" },
  { date: "Apr 22", event: "Beige Book release" },
  { date: "May 1", event: "FOMC Decision — rate hold expected" },
  { date: "May 3", event: "NFP April report" },
  { date: "May 14", event: "CPI April release" },
];

const EARNINGS = [
  { date: "Apr 8", ticker: "TSLA", type: "Earnings", est: "$0.45" },
  { date: "Apr 10", ticker: "JPM", type: "Earnings", est: "$4.12" },
  { date: "Apr 12", ticker: "UNH", type: "Earnings", est: "$6.75" },
  { date: "Apr 14", ticker: "MSFT", type: "Ex-Div", est: "$0.75" },
  { date: "Apr 15", ticker: "BAC", type: "Earnings", est: "$0.82" },
];

// ── Live news feed data ──
const INITIAL_LIVE_NEWS = [
  {
    id: 1,
    timestamp: "2m ago",
    source: "Bloomberg",
    headline: "NVDA surges on record AI chip demand — Blackwell GPU orders double sequentially",
    tickers: ["NVDA"],
    sentiment: "bullish" as const,
    category: "hot" as const,
  },
  {
    id: 2,
    timestamp: "7m ago",
    source: "Reuters",
    headline: "Fed signals potential rate cut in June — meeting minutes reveal dovish shift in tone",
    tickers: ["SPY", "QQQ", "TLT"],
    sentiment: "bullish" as const,
    category: "top" as const,
  },
  {
    id: 3,
    timestamp: "15m ago",
    source: "CNBC",
    headline: "JPMorgan beats Q1 earnings estimates by 8%, raises full-year NII guidance above consensus",
    tickers: ["JPM"],
    sentiment: "bullish" as const,
    category: "top" as const,
  },
  {
    id: 4,
    timestamp: "23m ago",
    source: "WSJ",
    headline: "Tesla recalls 500K vehicles over full-self-driving autopilot lateral control concerns",
    tickers: ["TSLA"],
    sentiment: "bearish" as const,
    category: "breaking" as const,
  },
  {
    id: 5,
    timestamp: "31m ago",
    source: "Reuters",
    headline: "Oil drops 2.3% as OPEC+ weighs surprise production increase at April emergency meeting",
    tickers: ["XOM", "CL"],
    sentiment: "bearish" as const,
    category: "top" as const,
  },
  {
    id: 6,
    timestamp: "44m ago",
    source: "Bloomberg",
    headline: "Microsoft Azure cloud revenue accelerates to +31% YoY — beats Copilot adoption targets",
    tickers: ["MSFT"],
    sentiment: "bullish" as const,
    category: "hot" as const,
  },
  {
    id: 7,
    timestamp: "1h ago",
    source: "FT",
    headline: "China manufacturing PMI contracts for third month, raising global growth slowdown fears",
    tickers: ["XOM", "GLD"],
    sentiment: "bearish" as const,
    category: "top" as const,
  },
  {
    id: 8,
    timestamp: "1h ago",
    source: "CNBC",
    headline: "Apple reportedly in advanced talks to integrate Gemini AI into iOS 19 — partnership widening",
    tickers: ["AAPL", "GOOGL"],
    sentiment: "bullish" as const,
    category: "hot" as const,
  },
  {
    id: 9,
    timestamp: "1h ago",
    source: "Reuters",
    headline: "Meta launches AI-powered ad optimization engine — click-through rates improve 18%",
    tickers: ["META"],
    sentiment: "bullish" as const,
    category: "top" as const,
  },
  {
    id: 10,
    timestamp: "2h ago",
    source: "Bloomberg",
    headline: "10Y Treasury yield breaks 4.35% resistance, risk assets face headwind",
    tickers: ["TLT", "SPY"],
    sentiment: "bearish" as const,
    category: "breaking" as const,
  },
  {
    id: 11,
    timestamp: "2h ago",
    source: "WSJ",
    headline: "Amazon AWS wins $4.2B DoD JEDI follow-on cloud contract, analyst upgrades issued",
    tickers: ["AMZN"],
    sentiment: "bullish" as const,
    category: "hot" as const,
  },
  {
    id: 12,
    timestamp: "3h ago",
    source: "FT",
    headline: "Visa settles DOJ antitrust case for $100M, removes overhang on shares",
    tickers: ["V"],
    sentiment: "bullish" as const,
    category: "top" as const,
  },
];

// New headlines that arrive over time
const INCOMING_HEADLINES = [
  { source: "Bloomberg", headline: "UnitedHealth beats Q1 medical loss ratio estimates, cost pressures ease", tickers: ["UNH"], sentiment: "bullish" as const, category: "hot" as const },
  { source: "Reuters", headline: "Google antitrust remedy trial begins — DOJ seeks search market structural changes", tickers: ["GOOGL"], sentiment: "bearish" as const, category: "breaking" as const },
  { source: "CNBC", headline: "Exxon Permian Basin output exceeds 1.2M barrels/day, efficiency gains on target", tickers: ["XOM"], sentiment: "bullish" as const, category: "top" as const },
  { source: "WSJ", headline: "Amazon Prime membership hits 250M globally, ad revenue up 23% YoY", tickers: ["AMZN"], sentiment: "bullish" as const, category: "hot" as const },
  { source: "FT", headline: "US-China trade tensions flare — new 25% tariff on EV components proposed", tickers: ["TSLA", "SPY"], sentiment: "bearish" as const, category: "breaking" as const },
  { source: "Bloomberg", headline: "Nvidia H200 allocation sold out through Q3 2026 — hyperscaler demand unprecedented", tickers: ["NVDA"], sentiment: "bullish" as const, category: "hot" as const },
  { source: "Reuters", headline: "JPMorgan sees credit card delinquencies stabilizing — consumer resilience intact", tickers: ["JPM", "V"], sentiment: "bullish" as const, category: "top" as const },
  { source: "CNBC", headline: "CPI March comes in at 3.1% YoY, slightly below 3.2% consensus estimate", tickers: ["SPY", "TLT", "QQQ"], sentiment: "bullish" as const, category: "breaking" as const },
  { source: "WSJ", headline: "Meta AI assistant reaches 600M monthly users — monetization roadmap unveiled", tickers: ["META"], sentiment: "bullish" as const, category: "hot" as const },
  { source: "Bloomberg", headline: "Apple Vision Pro 2 enters mass production — holiday 2026 launch confirmed", tickers: ["AAPL"], sentiment: "bullish" as const, category: "top" as const },
];

const SENTIMENT_CONFIG = {
  bullish: { emoji: "🟢", label: "Bullish", color: "#3fb950" },
  bearish: { emoji: "🔴", label: "Bearish", color: "#f85149" },
  neutral: { emoji: "⚪", label: "Neutral", color: "#7d8590" },
} as const;

const CATEGORY_CONFIG = {
  hot: { emoji: "🔥", label: "Hot", color: "#f0883e" },
  top: { emoji: "⭐", label: "Top", color: "#d29922" },
  breaking: { emoji: "🚨", label: "Breaking", color: "#f85149" },
} as const;

interface LiveNewsItem {
  id: number;
  timestamp: string;
  source: string;
  headline: string;
  tickers: string[];
  sentiment: "bullish" | "bearish" | "neutral";
  category: "hot" | "top" | "breaking";
}

// ── CSS for scroll animation (injected once) ──
const SCROLL_CSS = `
@keyframes news-scroll-1x {
  0% { transform: translateY(0); }
  100% { transform: translateY(-50%); }
}
@keyframes news-scroll-2x {
  0% { transform: translateY(0); }
  100% { transform: translateY(-50%); }
}
@keyframes news-scroll-3x {
  0% { transform: translateY(0); }
  100% { transform: translateY(-50%); }
}
.news-scroll-1x {
  animation: news-scroll-1x 30s linear infinite;
}
.news-scroll-2x {
  animation: news-scroll-1x 15s linear infinite;
}
.news-scroll-3x {
  animation: news-scroll-1x 10s linear infinite;
}
.news-scroll-paused {
  animation-play-state: paused !important;
}
@keyframes slide-in-from-bottom {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}
.news-new-item {
  animation: slide-in-from-bottom 0.4s ease-out forwards;
}
`;

function injectScrollCSS() {
  if (document.getElementById("news-scroll-style")) return;
  const el = document.createElement("style");
  el.id = "news-scroll-style";
  el.textContent = SCROLL_CSS;
  document.head.appendChild(el);
}

// ── Live News Feed Component ──
function LiveNewsFeed() {
  const [news, setNews] = useState<LiveNewsItem[]>(INITIAL_LIVE_NEWS);
  const [speed, setSpeed] = useState<1 | 2 | 3>(1);
  const [paused, setPaused] = useState(false);
  const [newIds, setNewIds] = useState<Set<number>>(new Set());
  const incomingIndex = useRef(0);
  const nextId = useRef(100);

  useEffect(() => {
    injectScrollCSS();
  }, []);

  // Add new headlines every 8-12 seconds
  useEffect(() => {
    const getDelay = () => 8000 + Math.random() * 4000;
    let timeout: ReturnType<typeof setTimeout>;
    const scheduleNext = () => {
      timeout = setTimeout(() => {
        const template = INCOMING_HEADLINES[incomingIndex.current % INCOMING_HEADLINES.length];
        incomingIndex.current++;
        const newItem: LiveNewsItem = {
          ...template,
          id: nextId.current++,
          timestamp: "just now",
        };
        setNews(prev => [newItem, ...prev.slice(0, 23)]);
        setNewIds(prev => {
          const next = new Set(prev);
          next.add(newItem.id);
          setTimeout(() => setNewIds(s => { const n = new Set(s); n.delete(newItem.id); return n; }), 1000);
          return next;
        });
        scheduleNext();
      }, getDelay());
    };
    scheduleNext();
    return () => clearTimeout(timeout);
  }, []);

  const scrollClass = `news-scroll-${speed}x${paused ? " news-scroll-paused" : ""}`;

  const bullishCount = news.filter(n => n.sentiment === "bullish").length;
  const bearishCount = news.filter(n => n.sentiment === "bearish").length;

  return (
    <DashboardPanel
      title="LIVE NEWS — PORTFOLIO RELEVANT"
      className="col-span-3"
      noPadding
      headerRight={
        <div className="flex items-center gap-3 text-[8px] font-mono">
          {/* LIVE indicator */}
          <div className="flex items-center gap-1">
            <span
              className="w-2 h-2 rounded-full bg-terminal-positive"
              style={{ animation: "pulse 1.2s ease-in-out infinite", boxShadow: "0 0 0 0 #3fb950" }}
            />
            <span className="text-terminal-positive font-semibold tracking-wider">LIVE</span>
          </div>
          {/* Sentiment counts */}
          <span className="text-terminal-text-faint">
            {bullishCount}🟢 {bearishCount}🔴
          </span>
          {/* Speed controls */}
          <div className="flex items-center gap-0.5 ml-1">
            <span className="text-terminal-text-faint mr-1">SPEED:</span>
            {([1, 2, 3] as const).map(s => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`px-1.5 py-0.5 rounded text-[8px] transition-colors ${speed === s ? "bg-terminal-accent/20 text-terminal-accent border border-terminal-accent/40" : "text-terminal-text-faint hover:text-terminal-text-muted border border-transparent"}`}
              >
                {s}x
              </button>
            ))}
          </div>
          {/* Pause/play */}
          <button
            onClick={() => setPaused(p => !p)}
            className={`px-1.5 py-0.5 rounded text-[8px] border transition-colors ${paused ? "border-terminal-warning/40 text-terminal-warning bg-terminal-warning/10" : "border-terminal-border/40 text-terminal-text-faint hover:text-terminal-text-muted"}`}
          >
            {paused ? "▶ PLAY" : "⏸ PAUSE"}
          </button>
        </div>
      }
    >
      {/* Scroll container — fixed height viewport */}
      <div
        className="overflow-hidden relative"
        style={{ height: 180 }}
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
      >
        {/* Fade masks */}
        <div className="absolute inset-x-0 top-0 h-6 z-10 pointer-events-none" style={{ background: "linear-gradient(to bottom, var(--color-terminal-surface, #0d1117), transparent)" }} />
        <div className="absolute inset-x-0 bottom-0 h-6 z-10 pointer-events-none" style={{ background: "linear-gradient(to top, var(--color-terminal-surface, #0d1117), transparent)" }} />

        {/* Scrolling content — doubled for seamless loop */}
        <div className={scrollClass}>
          {[...news, ...news].map((item, idx) => {
            const sent = SENTIMENT_CONFIG[item.sentiment];
            const cat = CATEGORY_CONFIG[item.category];
            const isNew = newIds.has(item.id) && idx < news.length;
            return (
              <div
                key={`${item.id}-${idx}`}
                className={`flex items-center gap-2 px-2 py-1 border-b border-terminal-border/20 hover:bg-white/[0.02] transition-colors text-[9px] font-mono ${isNew ? "news-new-item bg-terminal-accent/3" : ""}`}
              >
                {/* Timestamp */}
                <span className="text-terminal-text-faint whitespace-nowrap w-14 flex-shrink-0">{item.timestamp}</span>
                {/* Source */}
                <span className="text-terminal-accent text-[8px] font-semibold flex-shrink-0 w-16 truncate">{item.source}</span>
                {/* Headline */}
                <span className="text-terminal-text-muted flex-1 leading-relaxed truncate">{item.headline}</span>
                {/* Tickers */}
                <div className="flex gap-1 flex-shrink-0">
                  {item.tickers.slice(0, 3).map(t => (
                    <span
                      key={t}
                      className="inline-block px-1 py-0.5 rounded text-[7px] font-bold"
                      style={{ color: "#00d4aa", background: "rgba(0,212,170,0.12)", border: "1px solid rgba(0,212,170,0.25)" }}
                    >
                      {t}
                    </span>
                  ))}
                </div>
                {/* Sentiment */}
                <div className="flex items-center gap-1 flex-shrink-0 w-16">
                  <span className="text-[9px]">{sent.emoji}</span>
                  <span className="text-[7px] font-medium" style={{ color: sent.color }}>{sent.label}</span>
                </div>
                {/* Category */}
                <div className="flex items-center gap-1 flex-shrink-0 w-16">
                  <span className="text-[9px]">{cat.emoji}</span>
                  <span className="text-[7px] font-medium" style={{ color: cat.color }}>{cat.label}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </DashboardPanel>
  );
}

export default function MarketWrap() {
  // ─── Engine API ─────────────────────────────────────
  const { data: wrapData } = useEngineQuery<MarketWrapData>("/macro/wrap", { refetchInterval: 30000 });
  const { data: macroData } = useEngineQuery<MacroSnapshot>("/macro/snapshot", { refetchInterval: 15000 });

  // Override sectors from API when available
  const sectors = useMemo(() => {
    if (!wrapData?.sectors?.length) return SECTORS;
    return wrapData.sectors.map((s) => ({
      name: s.sector,
      daily: s.return_1d * 100,
      weekly: s.return_1w * 100,
      monthly: s.return_1m * 100,
      color: SECTORS.find((sec) => sec.name.toLowerCase().startsWith(s.sector.toLowerCase().slice(0, 4)))?.color ?? "#7d8590",
    }));
  }, [wrapData]);

  const regime = macroData?.regime ?? "risk-on";
  const vix = macroData?.vix ?? 14.2;
  const tone = wrapData?.market_tone ?? "risk-on";

  return (
    <div className="h-full flex flex-col gap-[2px] p-[2px] overflow-hidden" data-testid="market-wrap">

      {/* ─── Main resizable area: left content | right sidebar ─── */}
      <div className="flex-1 min-h-0">
        <ResizableDashboard defaultSizes={[72, 28]} minSizes={[45, 18]}>
          {/* Left: Market Direction + GICS Heatmap */}
          <div className="h-full flex flex-col gap-[2px] overflow-hidden">
            {/* Market Direction Narrative */}
            <DashboardPanel title="MARKET DIRECTION">
              <div className="space-y-2 text-[11px] text-terminal-text-muted leading-relaxed">
                <p>
                  Markets are in a <span className="text-terminal-accent font-medium">{tone}</span> regime with{" "}
                  {tone === "risk-on" ? "strong breadth improvement" : tone === "risk-off" ? "deteriorating breadth" : "mixed signals"}.
                  Macro regime: <span className="text-terminal-accent font-medium">{regime}</span>.
                  VIX at {vix.toFixed(1)}{vix < 16 ? " suggests complacency — monitor for mean reversion" : vix > 25 ? " elevated — risk management priority" : " within normal range"}.
                </p>
                <p>
                  Yield spread: {(macroData?.yield_spread ?? 0).toFixed(2)}% | Credit spread: {(macroData?.credit_spread ?? 0).toFixed(2)}% |
                  SPY 1M: {((macroData?.spy_return_1m ?? 0) * 100).toFixed(1)}% | GMTF: {(macroData?.gmtf_score ?? 0).toFixed(2)}
                </p>
              </div>
            </DashboardPanel>

            {/* GICS Sector Heatmap */}
            <DashboardPanel title="GICS SECTOR HEATMAP" className="flex-1">
              <div className="grid grid-cols-4 gap-1.5">
                {sectors.map((s) => {
                  const intensity = Math.min(Math.abs(s.daily) / 2, 1);
                  const bg = s.daily >= 0
                    ? `rgba(0, 212, 170, ${0.08 + intensity * 0.2})`
                    : `rgba(248, 81, 73, ${0.08 + intensity * 0.2})`;
                  return (
                    <div key={s.name} className="rounded p-2 border border-terminal-border/50" style={{ background: bg }}>
                      <div className="text-[9px] text-terminal-text-muted font-medium truncate">{s.name}</div>
                      <div className="flex items-baseline gap-2 mt-0.5">
                        <span className={`text-sm font-mono font-bold tabular-nums ${s.daily >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                          {s.daily >= 0 ? "+" : ""}{s.daily.toFixed(1)}%
                        </span>
                        <span className="text-[8px] text-terminal-text-faint font-mono">
                          W: {s.weekly >= 0 ? "+" : ""}{s.weekly.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </DashboardPanel>
          </div>

          {/* Right sidebar: Scenario Thesis + CVR News + Fed Calendar */}
          <div className="h-full flex flex-col gap-[2px] overflow-hidden">
            <DashboardPanel title="SCENARIO THESIS">
              <div className="space-y-2">
                {SCENARIOS.map((s) => (
                  <div key={s.label} className="border border-terminal-border rounded p-2">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[10px] font-mono font-bold" style={{ color: s.color }}>{s.label}</span>
                      <span className="text-[9px] text-terminal-text-faint font-mono">{s.prob}</span>
                    </div>
                    <p className="text-[9px] text-terminal-text-muted leading-relaxed">{s.desc}</p>
                  </div>
                ))}
              </div>
            </DashboardPanel>

            {/* CVR News */}
            <DashboardPanel title="CVR NEWS" className="flex-1">
              <div className="space-y-1.5">
                {NEWS.map((n, i) => (
                  <div key={i} className="flex gap-2 text-[9px]">
                    <span className="text-terminal-text-faint font-mono flex-shrink-0 w-10">{n.time}</span>
                    <span className="text-terminal-text-muted flex-1">{n.headline}</span>
                    <span className="text-terminal-accent text-[8px] flex-shrink-0">{n.source}</span>
                  </div>
                ))}
              </div>
            </DashboardPanel>

            {/* Fed Calendar */}
            <DashboardPanel title="FED CALENDAR" className="flex-1">
              <div className="space-y-1.5">
                {FED_EVENTS.map((e, i) => (
                  <div key={i} className="flex gap-2 text-[9px]">
                    <span className="text-terminal-warning font-mono flex-shrink-0 w-12">{e.date}</span>
                    <span className="text-terminal-text-muted">{e.event}</span>
                  </div>
                ))}
              </div>
            </DashboardPanel>
          </div>
        </ResizableDashboard>
      </div>

      {/* ── Live News Feed — full width ── */}
      <div className="flex-shrink-0">
        <LiveNewsFeed />
      </div>

      {/* Bottom: Upcoming Earnings/Dividends */}
      <div className="flex-shrink-0">
        <DashboardPanel title="INCOMING EVENTS">
          <div className="flex gap-3 overflow-x-auto pb-1">
            {EARNINGS.map((e, i) => (
              <div key={i} className="flex-shrink-0 border border-terminal-border rounded p-2 min-w-[140px]">
                <div className="flex items-center gap-2 text-[9px]">
                  <span className="text-terminal-warning font-mono">{e.date}</span>
                  <span className={`px-1 py-0.5 rounded text-[7px] font-medium ${
                    e.type === "Earnings" ? "bg-terminal-accent/10 text-terminal-accent" : "bg-terminal-blue/10 text-terminal-blue"
                  }`}>{e.type}</span>
                </div>
                <div className="text-sm font-mono font-bold text-terminal-text-primary mt-1">{e.ticker}</div>
                <div className="text-[8px] text-terminal-text-faint font-mono">Est: {e.est}</div>
              </div>
            ))}
          </div>
        </DashboardPanel>
      </div>
    </div>
  );
}
