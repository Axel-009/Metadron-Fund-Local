import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import { useState, useEffect, useRef, useMemo } from "react";
import { useEngineQuery, type MarketWrapData, type MacroSnapshot } from "@/hooks/use-engine-api";

// TODO: Wire to /cube/state regime probabilities when available
const SCENARIOS = [
  { label: "BASE", prob: "60%", desc: "Soft landing, gradual disinflation. Fed cuts 2x in 2026. SPX +8-12% YoY.", color: "#00d4aa" },
  { label: "BULL", prob: "25%", desc: "AI-driven productivity boom. Strong earnings revisions upward. SPX +18-22%.", color: "#3fb950" },
  { label: "BEAR", prob: "15%", desc: "Recession triggered by credit event. Earnings -15%. SPX -20-25%.", color: "#f85149" },
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
  const [news, setNews] = useState<LiveNewsItem[]>([]);
  const [speed, setSpeed] = useState<1 | 2 | 3>(1);
  const [paused, setPaused] = useState(false);
  const [newIds, setNewIds] = useState<Set<number>>(new Set());
  const nextId = useRef(100);
  const prevFeedLen = useRef(0);

  useEffect(() => {
    injectScrollCSS();
  }, []);

  // Poll real news from NewsEngine every 30s
  const { data: newsFeed } = useEngineQuery<{ feed: Array<{ id: string; timestamp: string; source: string; headline: string; tickers: string[]; sentiment: string; category: string }> }>("/signals/news/live?limit=24", { refetchInterval: 30000 });

  // Sync API news into the feed when new items arrive
  useEffect(() => {
    if (!newsFeed?.feed?.length) return;
    if (newsFeed.feed.length === prevFeedLen.current) return;
    prevFeedLen.current = newsFeed.feed.length;

    const apiItems: LiveNewsItem[] = newsFeed.feed.map((n) => ({
      id: nextId.current++,
      timestamp: n.timestamp || "just now",
      source: n.source || "OpenBB",
      headline: n.headline,
      tickers: n.tickers || [],
      sentiment: (n.sentiment || "neutral") as LiveNewsItem["sentiment"],
      category: (n.category || "top") as LiveNewsItem["category"],
    }));

    setNews(apiItems);

    // Flash the newest items
    const flashIds = new Set(apiItems.slice(0, 3).map(i => i.id));
    setNewIds(flashIds);
    setTimeout(() => setNewIds(new Set()), 1500);
  }, [newsFeed]);

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
          {news.length === 0 ? (
            <div style={{color: "var(--muted)", fontSize: 11, padding: "28px 16px", textAlign: "center", lineHeight: 1.6, opacity: 0.7}}>
              Live news feed connecting — headlines will stream once the NewsEngine initializes.
            </div>
          ) : null}
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
  const { data: newsApi } = useEngineQuery<{ news: Array<Record<string, string>> }>("/macro/news", { refetchInterval: 60000 });
  const { data: calApi } = useEngineQuery<{ events: Array<Record<string, string>> }>("/macro/calendar", { refetchInterval: 300000 });

  // Wire NEWS from API
  const news = useMemo(() => {
    if (!newsApi?.news?.length) return [];
    return newsApi.news.slice(0, 6).map((n) => ({
      time: (n.date || n.published || "").slice(11, 16) || "—",
      headline: n.title || n.headline || "",
      source: n.source || n.provider || "OpenBB",
    }));
  }, [newsApi]);

  // Wire FED_EVENTS from calendar API (filter for Fed/central bank events)
  const fedEvents = useMemo(() => {
    if (!calApi?.events?.length) return [];
    return calApi.events
      .filter((e) => {
        const ev = (e.event || e.name || "").toLowerCase();
        return ev.includes("fed") || ev.includes("fomc") || ev.includes("rate") || ev.includes("cpi") || ev.includes("nfp") || ev.includes("gdp");
      })
      .slice(0, 5)
      .map((e) => ({
        date: (e.date || "").slice(0, 6),
        event: e.event || e.name || "",
      }));
  }, [calApi]);

  // Wire EARNINGS from calendar API
  const earnings = useMemo(() => {
    if (!calApi?.events?.length) return [];
    return calApi.events
      .filter((e) => {
        const ev = (e.event || e.name || "").toLowerCase();
        return ev.includes("earning") || ev.includes("ex-div") || ev.includes("dividend");
      })
      .slice(0, 5)
      .map((e) => ({
        date: (e.date || "").slice(0, 6),
        ticker: e.symbol || e.ticker || "—",
        type: (e.event || "").includes("div") ? "Ex-Div" : "Earnings",
        est: e.consensus || e.forecast || "—",
      }));
  }, [calApi]);

  // Override sectors from API when available
  const sectors = useMemo(() => {
    if (!wrapData?.sectors?.length) return [];
    return wrapData.sectors.map((s) => ({
      name: s.sector,
      daily: s.return_1d * 100,
      weekly: s.return_1w * 100,
      monthly: s.return_1m * 100,
      color: "#7d8590",
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
                {sectors.length === 0 && (
                  <div className="col-span-4 text-[10px] text-terminal-text-faint text-center py-6 opacity-70">
                    Sector data loading — awaiting macro engine...
                  </div>
                )}
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
                {news.length === 0 && (
                  <div className="text-[10px] text-terminal-text-faint text-center py-4 opacity-70">
                    News feed loading — requires Tiingo API key...
                  </div>
                )}
                {news.map((n, i) => (
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
                {fedEvents.length === 0 && (
                  <div className="text-[10px] text-terminal-text-faint text-center py-4 opacity-70">
                    Calendar loading...
                  </div>
                )}
                {fedEvents.map((e, i) => (
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
            {earnings.length === 0 && (
              <div className="text-[10px] text-terminal-text-faint py-3 opacity-70">
                Earnings calendar loading...
              </div>
            )}
            {earnings.map((e, i) => (
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
