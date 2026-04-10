import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, LineChart, Line, ComposedChart,
} from "recharts";
import { useEngineQuery } from "@/hooks/use-engine-api";

const INDICATORS = ["SMA 20", "SMA 50", "RSI", "MACD", "Volume", "Bollinger"];

// ═══════════ CANDLESTICK CHART (Canvas) ═══════════

function CandlestickCanvas({ data, width, height }: { data: any[]; width: number; height: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data.length) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = width * 2;
    canvas.height = height * 2;
    ctx.scale(2, 2);
    ctx.clearRect(0, 0, width, height);

    const visible = data.slice(-80);
    const padding = { top: 10, bottom: 25, left: 50, right: 10 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;
    const candleW = chartW / visible.length;

    const allHighs = visible.map((d: any) => d.high);
    const allLows = visible.map((d: any) => d.low);
    const maxP = Math.max(...allHighs) + 2;
    const minP = Math.min(...allLows) - 2;
    const priceRange = maxP - minP;

    const yScale = (p: number) => padding.top + ((maxP - p) / priceRange) * chartH;
    const xScale = (i: number) => padding.left + i * candleW;

    // Grid lines
    ctx.strokeStyle = "#1e2633";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (i / 5) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      const p = maxP - (i / 5) * priceRange;
      ctx.fillStyle = "#484f58";
      ctx.font = "9px JetBrains Mono";
      ctx.textAlign = "right";
      ctx.fillText(p.toFixed(0), padding.left - 5, y + 3);
    }

    // SMA lines
    ctx.lineWidth = 1;
    [{ key: "sma20", color: "#00d4aa" }, { key: "sma50", color: "#58a6ff" }].forEach(({ key, color }) => {
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      let started = false;
      visible.forEach((d: any, i: number) => {
        if (d[key]) {
          const x = xScale(i) + candleW / 2;
          const y = yScale(d[key]);
          if (!started) { ctx.moveTo(x, y); started = true; }
          else ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
    });

    // Candlesticks
    visible.forEach((d: any, i: number) => {
      const x = xScale(i);
      const isUp = d.close >= d.open;
      const color = isUp ? "#3fb950" : "#f85149";
      const bodyTop = yScale(Math.max(d.open, d.close));
      const bodyBot = yScale(Math.min(d.open, d.close));
      const bodyH = Math.max(1, bodyBot - bodyTop);

      // Wick
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + candleW / 2, yScale(d.high));
      ctx.lineTo(x + candleW / 2, yScale(d.low));
      ctx.stroke();

      // Body
      ctx.fillStyle = isUp ? color : color;
      ctx.fillRect(x + 1, bodyTop, candleW - 2, bodyH);
    });

    // Date labels
    ctx.fillStyle = "#484f58";
    ctx.font = "8px JetBrains Mono";
    ctx.textAlign = "center";
    for (let i = 0; i < visible.length; i += 10) {
      ctx.fillText(visible[i].date, xScale(i) + candleW / 2, height - 5);
    }
  }, [data, width, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height }}
      className="block"
    />
  );
}

// ═══════════ CHART PANEL ═══════════

function ChartPanel({ ticker }: { ticker: string }) {
  // Fetch real OHLCV from OpenBB, fall back to generated data
  const { data: histData } = useEngineQuery<{ data: Array<Record<string, number | string>> }>(`/universe/openbb/historical?ticker=${ticker}&days=200`, { refetchInterval: 60000 });

  const data = useMemo(() => {
    if (histData?.data?.length && histData.data.length > 10) {
      return histData.data.map((r, i, arr) => {
        const close = Number(r.close || r.Close || 0);
        const open = Number(r.open || r.Open || close);
        const high = Number(r.high || r.High || close);
        const low = Number(r.low || r.Low || close);
        const volume = Number(r.volume || r.Volume || 0);
        const date = String(r.date || r.index || r.Date || `Day ${i}`);
        const shortDate = date.length > 10 ? date.slice(5, 10) : date;
        // Calculate simple SMAs
        let sma20 = 0, sma50 = 0;
        if (i >= 19) {
          sma20 = arr.slice(i - 19, i + 1).reduce((s, d) => s + Number(d.close || d.Close || 0), 0) / 20;
        }
        if (i >= 49) {
          sma50 = arr.slice(i - 49, i + 1).reduce((s, d) => s + Number(d.close || d.Close || 0), 0) / 50;
        }
        return { date: shortDate, open, high, low, close, volume, sma20: +sma20.toFixed(2), sma50: +sma50.toFixed(2), rsi: 50, macd: 0, signal: 0, histogram: 0 };
      });
    }
    return [];
  }, [histData, ticker]);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ w: 800, h: 400 });
  const [activeIndicators, setActiveIndicators] = useState<Set<string>>(new Set(["SMA 20", "SMA 50", "Volume"]));
  const [timeframe, setTimeframe] = useState("1D");

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDims({ w: width, h: height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const toggleIndicator = useCallback((ind: string) => {
    setActiveIndicators((prev) => {
      const next = new Set(prev);
      if (next.has(ind)) next.delete(ind);
      else next.add(ind);
      return next;
    });
  }, []);

  const last = data[data.length - 1];
  const prev = data[data.length - 2];
  const chg = last && prev ? last.close - prev.close : 0;
  const chgPct = last && prev && prev.close ? (chg / prev.close) * 100 : 0;

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-2 py-1 border-b border-terminal-border/50 flex-shrink-0">
        <span className="text-terminal-accent font-semibold text-xs">{ticker}</span>
        {last ? (
          <>
            <span className="font-mono text-xs text-terminal-text-primary tabular-nums">{last.close.toFixed(2)}</span>
            <span className={`font-mono text-[10px] tabular-nums ${chg >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
              {chg >= 0 ? "+" : ""}{chg.toFixed(2)} ({chgPct >= 0 ? "+" : ""}{chgPct.toFixed(2)}%)
            </span>
          </>
        ) : (
          <span className="font-mono text-[10px] text-terminal-text-faint">Loading...</span>
        )}
        <div className="flex-1" />
        {["1m", "5m", "15m", "1H", "4H", "1D", "1W"].map((tf) => (
          <button
            key={tf}
            onClick={() => setTimeframe(tf)}
            className={`px-1.5 py-0.5 text-[8px] font-mono rounded-sm ${
              tf === timeframe ? "bg-terminal-accent/15 text-terminal-accent" : "text-terminal-text-faint hover:text-terminal-text-muted"
            }`}
          >
            {tf}
          </button>
        ))}
        <div className="w-px h-4 bg-terminal-border mx-1" />
        {INDICATORS.map((ind) => (
          <button
            key={ind}
            onClick={() => toggleIndicator(ind)}
            className={`px-1.5 py-0.5 text-[8px] rounded-sm ${
              activeIndicators.has(ind) ? "bg-terminal-accent/10 text-terminal-accent" : "text-terminal-text-faint hover:text-terminal-text-muted"
            }`}
          >
            {ind}
          </button>
        ))}
      </div>

      {/* Main chart area */}
      <div ref={containerRef} className="flex-1 min-h-0">
        {data.length === 0 ? (
          <div className="h-full flex items-center justify-center text-terminal-text-faint text-[11px] font-mono">
            Waiting for price data...
          </div>
        ) : dims.w > 0 && (
          <CandlestickCanvas
            data={data}
            width={dims.w}
            height={activeIndicators.has("RSI") || activeIndicators.has("MACD") ? dims.h * 0.65 : dims.h * 0.8}
          />
        )}
      </div>

      {/* RSI */}
      {activeIndicators.has("RSI") && data.length > 0 && (
        <div className="h-[60px] border-t border-terminal-border/30 flex-shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data.slice(-80)} margin={{ top: 4, right: 10, bottom: 0, left: 50 }}>
              <YAxis tick={{ fontSize: 8, fill: "#484f58" }} domain={[20, 80]} axisLine={false} tickLine={false} width={0} />
              <Line type="monotone" dataKey="rsi" stroke="#bc8cff" strokeWidth={1} dot={false} />
              <Line type="monotone" dataKey={() => 70} stroke="#f85149" strokeWidth={0.5} strokeDasharray="3 3" dot={false} />
              <Line type="monotone" dataKey={() => 30} stroke="#3fb950" strokeWidth={0.5} strokeDasharray="3 3" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Volume */}
      {activeIndicators.has("Volume") && data.length > 0 && (
        <div className="h-[50px] border-t border-terminal-border/30 flex-shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.slice(-80)} margin={{ top: 2, right: 10, bottom: 0, left: 50 }}>
              <Bar dataKey="volume" fill="#1e2633" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* MACD */}
      {activeIndicators.has("MACD") && data.length > 0 && (
        <div className="h-[60px] border-t border-terminal-border/30 flex-shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data.slice(-80)} margin={{ top: 4, right: 10, bottom: 0, left: 50 }}>
              <Bar dataKey="histogram" fill="#1e2633">
                {data.slice(-80).map((d: any, i: number) => (
                  <Bar key={i} dataKey="histogram" fill={d.histogram >= 0 ? "#3fb95066" : "#f8514966"} />
                ))}
              </Bar>
              <Line type="monotone" dataKey="macd" stroke="#00d4aa" strokeWidth={1} dot={false} />
              <Line type="monotone" dataKey="signal" stroke="#f85149" strokeWidth={1} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ═══════════ UPGRADED LEFT PANEL: SEARCH + HOLDINGS + WATCHLIST ═══════════

interface SecurityItem {
  ticker: string;
  price: number;
  chg: number;
  pct: number;
  vol: string;
  mktCap: string;
  isHolding: boolean;
  name?: string;
}

function SecurityRow({
  item,
  selected,
  onSelect,
}: {
  item: SecurityItem;
  selected: string;
  onSelect: (t: string) => void;
}) {
  return (
    <div
      onClick={() => onSelect(item.ticker)}
      data-testid={`security-row-${item.ticker}`}
      className={`flex items-center px-2 py-1 cursor-pointer border-b border-terminal-border/10 hover:bg-white/[0.02] ${
        item.ticker === selected ? "bg-terminal-accent/5 border-l-2 border-l-terminal-accent" : ""
      }`}
    >
      <span className={`w-[58px] font-semibold font-mono text-[10px] ${item.ticker === selected ? "text-terminal-accent" : "text-terminal-text-primary"}`}>
        {item.ticker}
      </span>
      <span className="flex-1 text-right font-mono text-[10px] tabular-nums">{item.price.toFixed(2)}</span>
      <span className={`w-[58px] text-right font-mono text-[10px] tabular-nums ${item.chg >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
        {item.pct >= 0 ? "+" : ""}{item.pct.toFixed(2)}%
      </span>
      <span className="w-[52px] text-right font-mono text-[9px] text-terminal-text-faint tabular-nums">{item.vol}</span>
    </div>
  );
}

// Source icons
function BriefcaseIcon() {
  return (
    <svg className="w-2.5 h-2.5 text-terminal-accent flex-shrink-0" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3">
      <rect x="2" y="5" width="12" height="9" rx="1.5" />
      <path d="M5 5V4a3 3 0 016 0v1" strokeLinecap="round" />
    </svg>
  );
}
function EyeIcon() {
  return (
    <svg className="w-2.5 h-2.5 flex-shrink-0" viewBox="0 0 16 16" fill="none" stroke="#58a6ff" strokeWidth="1.3">
      <path d="M1 8s3-5 7-5 7 5 7 5-3 5-7 5-7-5-7-5z" />
      <circle cx="8" cy="8" r="2" />
    </svg>
  );
}
function GlobeIcon() {
  return (
    <svg className="w-2.5 h-2.5 flex-shrink-0" viewBox="0 0 16 16" fill="none" stroke="#bc8cff" strokeWidth="1.3">
      <circle cx="8" cy="8" r="6.5" />
      <path d="M8 1.5c0 0-3 2.5-3 6.5s3 6.5 3 6.5M8 1.5c0 0 3 2.5 3 6.5s-3 6.5-3 6.5M1.5 8h13" />
    </svg>
  );
}

function SecurityListPanel({
  selected,
  onSelect,
  watchlist,
  onAddToWatchlist,
}: {
  selected: string;
  onSelect: (t: string) => void;
  watchlist: string[];
  onAddToWatchlist: (ticker: string) => void;
}) {
  const [search, setSearch] = useState("");
  const [holdingsCollapsed, setHoldingsCollapsed] = useState(false);
  const [watchlistCollapsed, setWatchlistCollapsed] = useState(false);
  const [searchCollapsed, setSearchCollapsed] = useState(false);
  const [liveResults, setLiveResults] = useState<SecurityItem[]>([]);
  const [searching, setSearching] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  // Fetch holdings from portfolio API
  const { data: posData } = useEngineQuery<{ positions: Array<{ ticker: string; current_price: number; unrealized_pnl: number; sector: string }> }>("/portfolio/positions", { refetchInterval: 15000 });

  // Fetch live prices for watchlist tickers
  const watchlistTickers = watchlist.length > 0 ? watchlist.join(",") : "";
  const { data: watchQuotes } = useEngineQuery<{ quotes: Array<{ ticker: string; price: number; change: number; change_pct: number; volume: number; source: string }> }>(
    `/universe/openbb/quotes?tickers=${watchlistTickers}`,
    { refetchInterval: 15000, enabled: watchlistTickers.length > 0 }
  );

  // Build holdings from API
  const holdings = useMemo(() => {
    if (posData?.positions?.length) {
      return posData.positions.map((p) => ({
        ticker: p.ticker,
        price: p.current_price,
        chg: p.unrealized_pnl,
        pct: p.current_price > 0 ? (p.unrealized_pnl / p.current_price) * 100 : 0,
        vol: "—",
        mktCap: "—",
        isHolding: true,
      }));
    }
    return [];
  }, [posData]);

  // Build watchlist items from live quotes
  const watchItems = useMemo(() => {
    const quotesMap = new Map((watchQuotes?.quotes ?? []).map(q => [q.ticker, q]));
    return watchlist.map(ticker => {
      const q = quotesMap.get(ticker);
      return {
        ticker,
        price: q?.price ?? 0,
        chg: q?.change ?? 0,
        pct: q?.change_pct ?? 0,
        vol: q?.volume ? `${(q.volume / 1_000_000).toFixed(1)}M` : "—",
        mktCap: "—",
        isHolding: false,
      };
    });
  }, [watchlist, watchQuotes]);

  const searchLower = search.toLowerCase().trim();

  // Live search via OpenBB API — searches ALL securities, not just our universe
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (searchLower.length < 2) {
      setLiveResults([]);
      setSearching(false);
      return;
    }
    setSearching(true);
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await fetch(`/api/engine/universe/openbb/search?query=${encodeURIComponent(searchLower)}`);
        if (res.ok) {
          const data = await res.json();
          const results = (data.results || []).map((r: Record<string, string>) => ({
            ticker: r.symbol || r.ticker || "",
            price: 0,
            chg: 0,
            pct: 0,
            vol: "—",
            mktCap: r.market_cap || "—",
            isHolding: false,
            name: r.name || r.security_name || "",
          }));
          setLiveResults(results);
        }
      } catch { /* errors visible in TECH tab */ }
      setSearching(false);
    }, 400);
  }, [searchLower]);

  // Filter holdings and watchlist locally
  const filteredHoldings = searchLower
    ? holdings.filter(s => s.ticker.toLowerCase().includes(searchLower))
    : holdings;
  const filteredWatch = searchLower
    ? watchItems.filter(s => s.ticker.toLowerCase().includes(searchLower))
    : watchItems;

  // Search results: live OpenBB results, excluding already-known tickers
  const allKnownTickers = new Set([...holdings.map(h => h.ticker), ...watchlist]);
  const filteredSearch = liveResults.filter(s => s.ticker && !allKnownTickers.has(s.ticker));

  // Show search results section only when actively searching
  const showSearchSection = searchLower.length >= 1;

  // Determine if a searched ticker can be added to watchlist
  const searchedItem = searchLower.length >= 2
    ? liveResults.find(s => s.ticker.toLowerCase() === searchLower)
    : null;
  const canAddToWatchlist = searchedItem && !watchlist.includes(searchedItem.ticker) && !allKnownTickers.has(searchedItem.ticker);

  return (
    <div className="h-full flex flex-col text-[10px] font-mono">
      {/* Search bar */}
      <div className="flex items-center gap-1 px-2 py-1.5 border-b border-terminal-border/50 flex-shrink-0">
        <svg className="w-3 h-3 text-terminal-text-faint flex-shrink-0" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="6.5" cy="6.5" r="4.5" />
          <path d="M10.5 10.5L14 14" strokeLinecap="round" />
        </svg>
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search all securities via OpenBB..."
          data-testid="input-security-search"
          className="flex-1 bg-transparent outline-none text-terminal-text-primary placeholder:text-terminal-text-faint text-[10px]"
        />
        {search && (
          <button
            onClick={() => setSearch("")}
            className="text-terminal-text-faint hover:text-terminal-text-muted text-[10px] flex-shrink-0"
          >
            ✕
          </button>
        )}
      </div>

      {/* ADD TO WATCHLIST button for search results */}
      {canAddToWatchlist && (
        <div className="px-2 py-1 border-b border-terminal-border/30 flex-shrink-0 bg-[#bc8cff]/5">
          <button
            onClick={() => {
              onAddToWatchlist(searchedItem!.ticker);
              setSearch("");
            }}
            data-testid="button-add-to-watchlist"
            className="w-full flex items-center justify-center gap-1.5 py-1 rounded text-[9px] font-semibold transition-colors bg-[#bc8cff]/15 text-[#bc8cff] border border-[#bc8cff]/30 hover:bg-[#bc8cff]/25"
          >
            <GlobeIcon />
            ADD "{searchedItem!.ticker}" TO WATCHLIST
          </button>
        </div>
      )}

      {/* Column headers */}
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50 flex-shrink-0">
        <span className="w-[58px]">Symbol</span>
        <span className="flex-1 text-right">Price</span>
        <span className="w-[58px] text-right">Chg %</span>
        <span className="w-[52px] text-right">Vol</span>
      </div>

      <div className="flex-1 overflow-auto">
        {/* ── HOLDINGS Section ── */}
        <div>
          <button
            onClick={() => setHoldingsCollapsed(c => !c)}
            data-testid="button-collapse-holdings"
            className="w-full flex items-center gap-1.5 px-2 py-1 bg-terminal-bg/60 hover:bg-terminal-bg/80 transition-colors border-b border-terminal-border/40 text-[8px] text-terminal-text-faint uppercase tracking-wider select-none"
          >
            <BriefcaseIcon />
            <span className="text-terminal-accent font-semibold">HOLDINGS</span>
            <span className="text-terminal-text-faint ml-1">({filteredHoldings.length})</span>
            <span className="ml-auto">{holdingsCollapsed ? "▶" : "▼"}</span>
          </button>
          {!holdingsCollapsed && filteredHoldings.length === 0 && (
            <div className="px-2 py-2 text-[9px] text-terminal-text-faint italic">
              {searchLower ? `No holdings match "${search}"` : "No positions"}
            </div>
          )}
          {!holdingsCollapsed && filteredHoldings.map(item => (
            <div key={item.ticker} onClick={() => onSelect(item.ticker)}
              data-testid={`security-row-${item.ticker}`}
              className={`flex items-center px-2 py-1 cursor-pointer border-b border-terminal-border/10 hover:bg-white/[0.02] ${
                item.ticker === selected ? "bg-terminal-accent/5 border-l-2 border-l-terminal-accent" : ""
              }`}
            >
              <BriefcaseIcon />
              <span className={`w-[50px] ml-1 font-semibold text-[10px] ${item.ticker === selected ? "text-terminal-accent" : "text-terminal-text-primary"}`}>{item.ticker}</span>
              <span className="flex-1 text-right font-mono text-[10px] tabular-nums">{item.price.toFixed(2)}</span>
              <span className={`w-[58px] text-right font-mono text-[10px] tabular-nums ${item.chg >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                {item.pct >= 0 ? "+" : ""}{item.pct.toFixed(2)}%
              </span>
              <span className="w-[52px] text-right font-mono text-[9px] text-terminal-text-faint tabular-nums">{item.vol}</span>
            </div>
          ))}
        </div>

        {/* ── WATCHLIST Section ── */}
        <div>
          <button
            onClick={() => setWatchlistCollapsed(c => !c)}
            data-testid="button-collapse-watchlist"
            className="w-full flex items-center gap-1.5 px-2 py-1 bg-terminal-bg/60 hover:bg-terminal-bg/80 transition-colors border-b border-terminal-border/40 text-[8px] text-terminal-text-faint uppercase tracking-wider select-none"
          >
            <EyeIcon />
            <span style={{ color: "#58a6ff" }} className="font-semibold">WATCHLIST</span>
            <span className="text-terminal-text-faint ml-1">({filteredWatch.length})</span>
            <span className="ml-auto">{watchlistCollapsed ? "▶" : "▼"}</span>
          </button>
          {!watchlistCollapsed && filteredWatch.length === 0 && (
            <div className="px-2 py-2 text-[9px] text-terminal-text-faint italic">
              {searchLower ? `No watchlist items match "${search}"` : "Add tickers by searching above"}
            </div>
          )}
          {!watchlistCollapsed && filteredWatch.map(item => (
            <div key={item.ticker} onClick={() => onSelect(item.ticker)}
              data-testid={`security-row-${item.ticker}`}
              className={`flex items-center px-2 py-1 cursor-pointer border-b border-terminal-border/10 hover:bg-white/[0.02] ${
                item.ticker === selected ? "bg-[#58a6ff]/5 border-l-2 border-l-[#58a6ff]" : ""
              }`}
            >
              <EyeIcon />
              <span className={`w-[50px] ml-1 font-semibold text-[10px] ${item.ticker === selected ? "text-[#58a6ff]" : "text-terminal-text-primary"}`}>{item.ticker}</span>
              <span className="flex-1 text-right font-mono text-[10px] tabular-nums">{item.price > 0 ? item.price.toFixed(2) : "—"}</span>
              <span className={`w-[58px] text-right font-mono text-[10px] tabular-nums ${item.chg >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                {item.price > 0 ? `${item.pct >= 0 ? "+" : ""}${item.pct.toFixed(2)}%` : "—"}
              </span>
              <span className="w-[52px] text-right font-mono text-[9px] text-terminal-text-faint tabular-nums">{item.vol}</span>
            </div>
          ))}
        </div>

        {/* ── SEARCH RESULTS Section ── (shown when searching) */}
        {showSearchSection && (
          <div>
            <button
              onClick={() => setSearchCollapsed(c => !c)}
              data-testid="button-collapse-search"
              className="w-full flex items-center gap-1.5 px-2 py-1 bg-terminal-bg/60 hover:bg-terminal-bg/80 transition-colors border-b border-terminal-border/40 text-[8px] text-terminal-text-faint uppercase tracking-wider select-none"
            >
              <GlobeIcon />
              <span style={{ color: "#bc8cff" }} className="font-semibold">SEARCH RESULTS</span>
              {searching && <span className="ml-1 text-[7px] text-terminal-text-faint">searching...</span>}
              <span className="text-terminal-text-faint ml-1">({filteredSearch.length})</span>
              <span className="ml-auto">{searchCollapsed ? "▶" : "▼"}</span>
            </button>
            {!searchCollapsed && filteredSearch.length === 0 && !searching && (
              <div className="px-2 py-2 text-[9px] text-terminal-text-faint italic">No results for "{search}"</div>
            )}
            {!searchCollapsed && filteredSearch.slice(0, 20).map(item => (
              <div key={item.ticker} onClick={() => onSelect(item.ticker)}
                data-testid={`search-row-${item.ticker}`}
                className={`flex items-center px-2 py-1 cursor-pointer border-b border-terminal-border/10 hover:bg-white/[0.02] ${
                  item.ticker === selected ? "bg-[#bc8cff]/5 border-l-2 border-l-[#bc8cff]" : ""
                }`}
              >
                <GlobeIcon />
                <span className={`w-[50px] ml-1 font-semibold text-[10px] ${item.ticker === selected ? "text-[#bc8cff]" : "text-terminal-text-primary"}`}>{item.ticker}</span>
                <span className="flex-1 text-right font-mono text-[10px] tabular-nums text-terminal-text-faint">{item.name ? item.name.slice(0, 14) : "—"}</span>
                <span className="w-[58px] text-right font-mono text-[9px] text-terminal-text-faint">{item.mktCap}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════ SCREENER ═══════════

function Screener() {
  const [filter, setFilter] = useState("");

  const { data: univData } = useEngineQuery<{ securities: Array<{ ticker: string; name: string; sector: string; pe_ratio: number; momentum_3m: number; quality_tier: string }> }>("/universe/securities?limit=50", { refetchInterval: 60000 });

  const rows = useMemo(() => {
    const source: Array<{ ticker: string; name: string; sector: string; pe: number; signal: string; score: string }> =
      univData?.securities?.length
        ? univData.securities.map((s) => {
            const mom = s.momentum_3m ?? 0;
            const signal = mom > 0.05 ? "BUY" : mom < -0.05 ? "SELL" : "HOLD";
            return {
              ticker: s.ticker,
              name: s.name,
              sector: s.sector,
              pe: s.pe_ratio ?? 0,
              signal,
              score: s.quality_tier ?? "?",
            };
          })
        : [];

    if (!filter.trim()) return source;
    const f = filter.toLowerCase();
    return source.filter((s) => s.ticker.toLowerCase().includes(f) || s.sector.toLowerCase().includes(f));
  }, [univData, filter]);

  return (
    <div className="h-full flex flex-col text-[10px] font-mono">
      <div className="flex items-center px-2 py-1.5 gap-2 border-b border-terminal-border/50 flex-shrink-0">
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter by ticker, sector..."
          className="flex-1 bg-terminal-bg border border-terminal-border/50 rounded px-2 py-0.5 text-[10px] text-terminal-text-primary placeholder:text-terminal-text-faint outline-none focus:border-terminal-accent/50"
        />
        <span className="text-terminal-text-faint text-[8px]">{rows.length} results</span>
      </div>
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/30">
        <span className="w-[50px]">Ticker</span>
        <span className="flex-1">Name</span>
        <span className="w-[80px]">Sector</span>
        <span className="w-[45px] text-right">P/E</span>
        <span className="w-[50px] text-right">Signal</span>
        <span className="w-[35px] text-right">Score</span>
      </div>
      <div className="flex-1 overflow-auto">
        {rows.length === 0 && (
          <div className="flex items-center justify-center h-20 text-[9px] text-terminal-text-faint font-mono italic">Waiting for data...</div>
        )}
        {rows.map((s) => (
          <div key={s.ticker} className="flex items-center px-2 py-1.5 border-b border-terminal-border/10 hover:bg-white/[0.02]">
            <span className="w-[50px] text-terminal-accent font-semibold">{s.ticker}</span>
            <span className="flex-1 text-terminal-text-muted truncate pr-2">{s.name}</span>
            <span className="w-[80px] text-terminal-text-faint truncate">{s.sector}</span>
            <span className="w-[45px] text-right tabular-nums">{typeof s.pe === "number" && s.pe > 0 ? s.pe.toFixed(1) : "—"}</span>
            <span className={`w-[50px] text-right font-semibold ${
              s.signal === "BUY" ? "text-terminal-positive" : s.signal === "SELL" ? "text-terminal-negative" : "text-terminal-text-muted"
            }`}>
              {s.signal}
            </span>
            <span className="w-[35px] text-right tabular-nums text-terminal-text-muted">
              {s.score}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ COMMAND LINE ═══════════

const CMD_HELP = [
  "  Available commands:",
  "    stocks/quote TICKER          — live quote for a ticker",
  "    economy/fred SERIES_ID       — FRED economic series (e.g. GDP, UNRATE)",
  "    stocks/search QUERY          — search securities by name/ticker",
  "    portfolio/holdings           — current portfolio positions",
  "    universe/sectors             — sector breakdown with momentum",
  "    help                         — show this message",
];

function formatJsonOutput(data: unknown, maxLines = 20): string[] {
  try {
    const lines: string[] = [];
    const flat = (obj: unknown, prefix = "", depth = 0): void => {
      if (depth > 4) return;
      if (Array.isArray(obj)) {
        obj.slice(0, 10).forEach((item, i) => flat(item, `[${i}] `, depth + 1));
        if (obj.length > 10) lines.push(`  ... (${obj.length - 10} more)`);
      } else if (obj !== null && typeof obj === "object") {
        Object.entries(obj as Record<string, unknown>).forEach(([k, v]) => {
          if (v !== null && typeof v === "object" && !Array.isArray(v)) {
            lines.push(`  ${prefix}${k}:`);
            flat(v, "  ", depth + 1);
          } else if (Array.isArray(v)) {
            lines.push(`  ${prefix}${k}: [${(v as unknown[]).slice(0, 3).map(x => JSON.stringify(x)).join(", ")}${v.length > 3 ? ` ... +${v.length - 3}` : ""}]`);
          } else {
            lines.push(`  ${prefix}${k}: ${v}`);
          }
        });
      } else {
        lines.push(`  ${prefix}${obj}`);
      }
    };
    flat(data);
    return lines.slice(0, maxLines);
  } catch {
    return [`  ${JSON.stringify(data).slice(0, 200)}`];
  }
}

function CommandLine() {
  const [cmd, setCmd] = useState("");
  const [history, setHistory] = useState<string[]>([
    "> economy/fred GDP",
    "  [Type a command or 'help' to see available commands]",
  ]);
  const historyEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  const runCommand = useCallback(async (raw: string) => {
    const trimmed = raw.trim();
    if (!trimmed) return;

    const appendLines = (lines: string[]) =>
      setHistory((prev) => [...prev, `> ${trimmed}`, ...lines]);

    if (trimmed === "help" || trimmed === "?") {
      appendLines(CMD_HELP);
      return;
    }

    // Parse: command arg1 [arg2] [--flag value]
    const parts = trimmed.split(/\s+/);
    const command = parts[0];
    const arg1 = parts[1] ?? "";

    let url: string | null = null;

    if (command === "stocks/quote" && arg1) {
      url = `/api/engine/universe/openbb/quote?ticker=${encodeURIComponent(arg1.toUpperCase())}`;
    } else if (command === "economy/fred" && arg1) {
      url = `/api/engine/universe/openbb/fred?series_id=${encodeURIComponent(arg1.toUpperCase())}`;
    } else if (command === "stocks/search" && arg1) {
      url = `/api/engine/universe/openbb/search?query=${encodeURIComponent(arg1)}`;
    } else if (command === "portfolio/holdings") {
      url = `/api/engine/portfolio/positions`;
    } else if (command === "universe/sectors") {
      url = `/api/engine/universe/sectors`;
    } else {
      appendLines([`  Unknown command: "${command}". Type 'help' for available commands.`]);
      return;
    }

    appendLines([`  Fetching ${url}...`]);

    try {
      const res = await fetch(url);
      if (!res.ok) {
        setHistory((prev) => [...prev, `  Error: HTTP ${res.status} ${res.statusText}`]);
        return;
      }
      const data = await res.json();
      const lines = formatJsonOutput(data);
      setHistory((prev) => [...prev, ...lines]);
    } catch (err) {
      setHistory((prev) => [...prev, `  Error: ${err instanceof Error ? err.message : String(err)}`]);
    }
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!cmd.trim()) return;
    runCommand(cmd);
    setCmd("");
  };

  return (
    <div className="h-full flex flex-col font-mono text-[10px]">
      <div className="flex-1 overflow-auto p-2">
        {history.map((line, i) => (
          <div key={i} className={`py-0.5 ${
            line.startsWith(">") ? "text-terminal-accent" :
            line.includes("Error:") ? "text-terminal-negative" :
            "text-terminal-text-muted"
          }`}>
            {line}
          </div>
        ))}
        <div ref={historyEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="flex items-center border-t border-terminal-border/50 px-2 py-1">
        <span className="text-terminal-accent mr-2">OpenBB &gt;</span>
        <input
          value={cmd}
          onChange={(e) => setCmd(e.target.value)}
          className="flex-1 bg-transparent outline-none text-terminal-text-primary placeholder:text-terminal-text-faint"
          placeholder="stocks/quote AAPL  |  economy/fred GDP  |  help"
        />
      </form>
    </div>
  );
}

// ═══════════ ASSET DETAIL PANEL ═══════════

function AssetDetailPanel({
  selectedTicker,
  selectedData,
  extraWatchlist,
  onAddToWatchlist,
}: {
  selectedTicker: string;
  selectedData: SecurityItem;
  extraWatchlist: string[];
  onAddToWatchlist: (ticker: string) => void;
}) {
  // Live quote — 15s refresh
  const { data: quoteData } = useEngineQuery<{ ticker: string; price: number; change: number; change_pct: number; error?: string }>(
    `/universe/openbb/quote?ticker=${selectedTicker}`,
    { refetchInterval: 15000 }
  );

  // Fundamentals — 5 min refresh
  const { data: fundData } = useEngineQuery<{ ticker: string; fundamentals: Record<string, unknown>; error?: string }>(
    `/universe/openbb/fundamentals?ticker=${selectedTicker}`,
    { refetchInterval: 300000 }
  );

  const fund = fundData?.fundamentals ?? {};

  // Live price values from API
  const livePrice = quoteData?.price ?? selectedData.price;
  const liveChange = quoteData?.change ?? selectedData.chg;
  const liveChangePct = quoteData?.change_pct ?? selectedData.pct;

  const pe = fund.pe_ratio ?? fund.pe ?? fund.forwardPE ?? "—";
  const eps = fund.eps ?? fund.epsTrailingTwelveMonths ?? "—";
  const beta = fund.beta ?? "—";
  const divYield = (fund.dividend_yield ?? fund.dividendYield)
    ? `${((Number(fund.dividend_yield ?? fund.dividendYield)) * 100).toFixed(2)}%`
    : "—";
  const high52 = fund.fifty_two_week_high ?? fund.fiftyTwoWeekHigh ?? "—";
  const low52 = fund.fifty_two_week_low ?? fund.fiftyTwoWeekLow ?? "—";

  const isWatchlisted = extraWatchlist.includes(selectedTicker);

  return (
    <div className="p-2 text-[10px] font-mono space-y-2">
      <div className="text-center pb-2 border-b border-terminal-border/30">
        <div className="text-terminal-accent text-sm font-semibold">{selectedData.ticker}</div>
        <div className="text-xl tabular-nums text-terminal-text-primary">
          {livePrice > 0 ? livePrice.toFixed(2) : "—"}
        </div>
        {livePrice > 0 && (
          <div className={`text-xs tabular-nums ${liveChange >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
            {liveChange >= 0 ? "+" : ""}{liveChange.toFixed(2)} ({liveChangePct >= 0 ? "+" : ""}{liveChangePct.toFixed(2)}%)
          </div>
        )}
        {/* Status badge */}
        <div className="mt-1.5 flex flex-col items-center gap-1.5">
          {selectedData.isHolding ? (
            <span className="px-1.5 py-0.5 rounded text-[7px] font-semibold bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30">
              ◆ PORTFOLIO HOLDING
            </span>
          ) : isWatchlisted ? (
            <span className="px-1.5 py-0.5 rounded text-[7px] font-semibold bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30">
              ✓ WATCHLIST
            </span>
          ) : (
            <>
              <span className="px-1.5 py-0.5 rounded text-[7px] font-semibold bg-[#58a6ff]/15 text-[#58a6ff] border border-[#58a6ff]/30">
                ○ SEARCH RESULT
              </span>
              <button
                onClick={() => onAddToWatchlist(selectedTicker)}
                data-testid="button-add-to-watchlist-detail"
                className="flex items-center gap-1 px-2 py-0.5 rounded text-[7px] font-semibold bg-[#bc8cff]/10 text-[#bc8cff] border border-[#bc8cff]/30 hover:bg-[#bc8cff]/20 transition-colors"
              >
                + ADD TO WATCHLIST
              </button>
            </>
          )}
        </div>
      </div>
      {[
        ["Mkt Cap", selectedData.mktCap],
        ["Volume", selectedData.vol],
        ["P/E", typeof pe === "number" ? pe.toFixed(2) : String(pe)],
        ["EPS", typeof eps === "number" ? eps.toFixed(2) : String(eps)],
        ["Div Yield", divYield],
        ["Beta", typeof beta === "number" ? beta.toFixed(2) : String(beta)],
        ["52W High", typeof high52 === "number" ? high52.toFixed(2) : String(high52)],
        ["52W Low", typeof low52 === "number" ? low52.toFixed(2) : String(low52)],
      ].map(([k, v]) => (
        <div key={k} className="flex justify-between">
          <span className="text-terminal-text-faint">{k}</span>
          <span className="text-terminal-text-primary tabular-nums">{v}</span>
        </div>
      ))}
    </div>
  );
}

// ═══════════ MAIN OPENBB TERMINAL ═══════════

export default function OpenBBTerminal() {
  // ─── Engine API ─────────────────────────────────────
  const { data: universeData } = useEngineQuery<{ securities: Array<{ ticker: string; name: string; sector: string; market_cap: number; pe_ratio: number; beta: number; quality_tier: string }>; total: number }>("/universe/securities?limit=200", { refetchInterval: 60000 });
  const { data: sectorData } = useEngineQuery<{ sectors: Array<{ sector: string; count: number; momentum: number }> }>("/universe/sectors", { refetchInterval: 60000 });

  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const [rightPanel, setRightPanel] = useState<"watchlist" | "screener">("watchlist");
  const [extraWatchlist, setExtraWatchlist] = useState<string[]>([]);

  const handleAddToWatchlist = useCallback((ticker: string) => {
    setExtraWatchlist(prev => prev.includes(ticker) ? prev : [...prev, ticker]);
  }, []);

  // Minimal fallback selectedData — AssetDetailPanel fetches live data itself
  const selectedData: SecurityItem = {
    ticker: selectedTicker,
    price: 0,
    chg: 0,
    pct: 0,
    vol: "—",
    mktCap: "—",
    isHolding: false,
  };

  return (
    <div className="h-full flex flex-col" data-testid="openbb-terminal">
      {/* Main area: top 3-col + bottom command */}
      <div className="flex-1 flex flex-col gap-[2px] p-[2px] overflow-hidden">
        {/* Top 3-column resizable row */}
        <div className="flex-1 min-h-0">
          <ResizableDashboard defaultSizes={[20, 55, 25]} minSizes={[12, 35, 15]}>
        {/* Left: Holdings + Watchlist */}
        <DashboardPanel
          title="SECURITIES"
          className="h-full"
          headerRight={
            <span className="text-[8px] text-terminal-text-faint font-mono">
              {extraWatchlist.length > 0 ? `${extraWatchlist.length} watching` : ""}
            </span>
          }
          noPadding
        >
          <SecurityListPanel
            selected={selectedTicker}
            onSelect={setSelectedTicker}
            watchlist={extraWatchlist}
            onAddToWatchlist={handleAddToWatchlist}
          />
        </DashboardPanel>

        {/* Center: Chart */}
        <DashboardPanel
          title={`${selectedTicker} — CHART`}
          className="h-full"
          headerRight={
            <div className="flex gap-1">
              {["Candle", "Line", "Area"].map((t, i) => (
                <button key={t} className={`px-1.5 py-0.5 text-[8px] rounded-sm ${i === 0 ? "bg-terminal-accent/15 text-terminal-accent" : "text-terminal-text-faint"}`}>
                  {t}
                </button>
              ))}
            </div>
          }
          noPadding
        >
          <ChartPanel ticker={selectedTicker} />
        </DashboardPanel>

        {/* Right: Asset Detail / Screener */}
        <DashboardPanel
          title={rightPanel === "watchlist" ? "ASSET DETAIL" : "SCREENER"}
          className="h-full"
          headerRight={
            <div className="flex gap-1">
              <button
                onClick={() => setRightPanel("watchlist")}
                className={`px-1.5 py-0.5 text-[8px] rounded-sm ${rightPanel === "watchlist" ? "bg-terminal-accent/15 text-terminal-accent" : "text-terminal-text-faint"}`}
              >
                Detail
              </button>
              <button
                onClick={() => setRightPanel("screener")}
                className={`px-1.5 py-0.5 text-[8px] rounded-sm ${rightPanel === "screener" ? "bg-terminal-accent/15 text-terminal-accent" : "text-terminal-text-faint"}`}
              >
                Screen
              </button>
            </div>
          }
          noPadding
        >
          {rightPanel === "screener" ? (
            <Screener />
          ) : (
            <AssetDetailPanel
              selectedTicker={selectedTicker}
              selectedData={selectedData}
              extraWatchlist={extraWatchlist}
              onAddToWatchlist={handleAddToWatchlist}
            />
          )}
        </DashboardPanel>
          </ResizableDashboard>
        </div>

        {/* Bottom: Command line */}
        <div className="flex-shrink-0" style={{ height: 200 }}>
          <DashboardPanel
            title="OPENBB COMMAND LINE"
            headerRight={
              <div className="flex gap-1">
                {["economy", "stocks", "crypto", "forex", "portfolio", "etf"].map((cat) => (
                  <span key={cat} className="px-1 py-0.5 text-[7px] text-terminal-text-faint bg-terminal-surface-2/50 rounded">{cat}</span>
                ))}
              </div>
            }
            className="h-full"
            noPadding
          >
            <CommandLine />
          </DashboardPanel>
        </div>
      </div>
    </div>
  );
}
