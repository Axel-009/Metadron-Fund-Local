import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, LineChart, Line, ComposedChart,
} from "recharts";

// ═══════════ SIMULATED DATA ═══════════

function generateOHLC(days = 200, startPrice = 189) {
  const data: any[] = [];
  let price = startPrice;
  const now = Date.now();
  for (let i = days; i >= 0; i--) {
    const open = price;
    const change = (Math.random() - 0.48) * 4;
    const high = open + Math.abs(change) + Math.random() * 2;
    const low = open - Math.abs(change) - Math.random() * 2;
    const close = open + change;
    const vol = Math.floor(40000000 + Math.random() * 60000000);
    price = close;
    data.push({
      date: new Date(now - i * 86400000).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      open: +open.toFixed(2),
      high: +high.toFixed(2),
      low: +low.toFixed(2),
      close: +close.toFixed(2),
      volume: vol,
      sma20: 0,
      sma50: 0,
      rsi: 50,
      macd: 0,
      signal: 0,
      histogram: 0,
    });
  }
  // Calculate SMAs
  for (let i = 0; i < data.length; i++) {
    if (i >= 19) {
      data[i].sma20 = +(data.slice(i - 19, i + 1).reduce((s: number, d: any) => s + d.close, 0) / 20).toFixed(2);
    }
    if (i >= 49) {
      data[i].sma50 = +(data.slice(i - 49, i + 1).reduce((s: number, d: any) => s + d.close, 0) / 50).toFixed(2);
    }
    // Simple RSI proxy
    if (i > 0) {
      const diff = data[i].close - data[i - 1].close;
      data[i].rsi = Math.min(90, Math.max(10, 50 + diff * 8 + (Math.random() - 0.5) * 10));
    }
    // MACD proxy
    data[i].macd = (Math.random() - 0.5) * 4;
    data[i].signal = data[i].macd * 0.7 + (Math.random() - 0.5) * 1;
    data[i].histogram = data[i].macd - data[i].signal;
  }
  return data;
}

const WATCHLIST = [
  { ticker: "AAPL", price: 189.45, chg: 1.23, pct: 0.65, vol: "62.3M", mktCap: "2.94T" },
  { ticker: "MSFT", price: 420.12, chg: -2.18, pct: -0.52, vol: "24.1M", mktCap: "3.12T" },
  { ticker: "NVDA", price: 875.30, chg: 18.42, pct: 2.15, vol: "48.7M", mktCap: "2.15T" },
  { ticker: "AMZN", price: 185.67, chg: 2.34, pct: 1.28, vol: "35.9M", mktCap: "1.93T" },
  { ticker: "GOOGL", price: 155.89, chg: -0.45, pct: -0.29, vol: "22.8M", mktCap: "1.94T" },
  { ticker: "META", price: 505.78, chg: 8.12, pct: 1.63, vol: "18.4M", mktCap: "1.30T" },
  { ticker: "TSLA", price: 178.22, chg: -5.34, pct: -2.91, vol: "89.1M", mktCap: "567B" },
  { ticker: "JPM", price: 198.34, chg: 1.12, pct: 0.57, vol: "8.7M", mktCap: "572B" },
  { ticker: "V", price: 282.90, chg: 0.78, pct: 0.28, vol: "5.2M", mktCap: "580B" },
  { ticker: "UNH", price: 502.15, chg: -3.45, pct: -0.68, vol: "3.8M", mktCap: "462B" },
  { ticker: "XOM", price: 115.23, chg: -1.87, pct: -1.60, vol: "15.2M", mktCap: "460B" },
  { ticker: "SPY", price: 527.82, chg: 2.34, pct: 0.44, vol: "72.4M", mktCap: "—" },
  { ticker: "QQQ", price: 448.19, chg: 5.12, pct: 1.16, vol: "42.1M", mktCap: "—" },
  { ticker: "IWM", price: 210.45, chg: -0.67, pct: -0.32, vol: "28.3M", mktCap: "—" },
];

const SCREENER_RESULTS = [
  { ticker: "SMCI", name: "Super Micro Computer", sector: "Technology", pe: 31.2, eps: 18.45, rsi: 72, signal: "BUY", score: 87 },
  { ticker: "ARM", name: "ARM Holdings", sector: "Technology", pe: 95.4, eps: 1.32, rsi: 65, signal: "BUY", score: 82 },
  { ticker: "PLTR", name: "Palantir Technologies", sector: "Technology", pe: 68.1, eps: 0.38, rsi: 58, signal: "HOLD", score: 71 },
  { ticker: "COIN", name: "Coinbase Global", sector: "Financials", pe: 42.8, eps: 5.89, rsi: 44, signal: "SELL", score: 38 },
  { ticker: "RIVN", name: "Rivian Automotive", sector: "Consumer Disc.", pe: -12.3, eps: -4.52, rsi: 35, signal: "SELL", score: 22 },
  { ticker: "CRWD", name: "CrowdStrike", sector: "Technology", pe: 55.2, eps: 5.67, rsi: 61, signal: "BUY", score: 78 },
  { ticker: "SNOW", name: "Snowflake", sector: "Technology", pe: -85.0, eps: -1.23, rsi: 42, signal: "HOLD", score: 55 },
  { ticker: "NET", name: "Cloudflare", sector: "Technology", pe: 210.0, eps: 0.45, rsi: 57, signal: "BUY", score: 74 },
];

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
  const data = useMemo(() => generateOHLC(200, ticker === "AAPL" ? 189 : ticker === "NVDA" ? 875 : 150 + Math.random() * 300), [ticker]);
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
  const chg = last.close - prev.close;
  const chgPct = (chg / prev.close) * 100;

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-2 py-1 border-b border-terminal-border/50 flex-shrink-0">
        <span className="text-terminal-accent font-semibold text-xs">{ticker}</span>
        <span className="font-mono text-xs text-terminal-text-primary tabular-nums">{last.close.toFixed(2)}</span>
        <span className={`font-mono text-[10px] tabular-nums ${chg >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
          {chg >= 0 ? "+" : ""}{chg.toFixed(2)} ({chgPct >= 0 ? "+" : ""}{chgPct.toFixed(2)}%)
        </span>
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
        {dims.w > 0 && (
          <CandlestickCanvas
            data={data}
            width={dims.w}
            height={activeIndicators.has("RSI") || activeIndicators.has("MACD") ? dims.h * 0.65 : dims.h * 0.8}
          />
        )}
      </div>

      {/* RSI */}
      {activeIndicators.has("RSI") && (
        <div className="h-[60px] border-t border-terminal-border/30 flex-shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data.slice(-80)} margin={{ top: 4, right: 10, bottom: 0, left: 50 }}>
              <YAxis tick={{ fontSize: 8, fill: "#484f58" }} domain={[20, 80]} axisLine={false} tickLine={false} width={0} />
              <Line type="monotone" dataKey="rsi" stroke="#bc8cff" strokeWidth={1} dot={false} />
              {/* Overbought / Oversold lines */}
              <Line type="monotone" dataKey={() => 70} stroke="#f85149" strokeWidth={0.5} strokeDasharray="3 3" dot={false} />
              <Line type="monotone" dataKey={() => 30} stroke="#3fb950" strokeWidth={0.5} strokeDasharray="3 3" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Volume */}
      {activeIndicators.has("Volume") && (
        <div className="h-[50px] border-t border-terminal-border/30 flex-shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.slice(-80)} margin={{ top: 2, right: 10, bottom: 0, left: 50 }}>
              <Bar dataKey="volume" fill="#1e2633" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* MACD */}
      {activeIndicators.has("MACD") && (
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

// ═══════════ WATCHLIST ═══════════

function Watchlist({ selected, onSelect }: { selected: string; onSelect: (t: string) => void }) {
  return (
    <div className="h-full flex flex-col text-[10px] font-mono">
      <div className="flex items-center px-2 py-1.5 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
        <span className="w-[60px]">Symbol</span>
        <span className="flex-1 text-right">Price</span>
        <span className="w-[60px] text-right">Chg %</span>
        <span className="w-[55px] text-right">Vol</span>
      </div>
      <div className="flex-1 overflow-auto">
        {WATCHLIST.map((w) => (
          <div
            key={w.ticker}
            onClick={() => onSelect(w.ticker)}
            className={`flex items-center px-2 py-1 cursor-pointer border-b border-terminal-border/10 hover:bg-white/[0.02] ${
              w.ticker === selected ? "bg-terminal-accent/5 border-l-2 border-l-terminal-accent" : ""
            }`}
          >
            <span className={`w-[60px] font-semibold ${w.ticker === selected ? "text-terminal-accent" : "text-terminal-text-primary"}`}>
              {w.ticker}
            </span>
            <span className="flex-1 text-right tabular-nums">{w.price.toFixed(2)}</span>
            <span className={`w-[60px] text-right tabular-nums ${w.chg >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
              {w.pct >= 0 ? "+" : ""}{w.pct.toFixed(2)}%
            </span>
            <span className="w-[55px] text-right text-terminal-text-faint tabular-nums">{w.vol}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ SCREENER ═══════════

function Screener() {
  return (
    <div className="h-full flex flex-col text-[10px] font-mono">
      <div className="flex items-center px-2 py-1.5 gap-2 border-b border-terminal-border/50 flex-shrink-0">
        <input
          type="text"
          placeholder="Filter by ticker, sector..."
          className="flex-1 bg-terminal-bg border border-terminal-border/50 rounded px-2 py-0.5 text-[10px] text-terminal-text-primary placeholder:text-terminal-text-faint outline-none focus:border-terminal-accent/50"
        />
        <span className="text-terminal-text-faint text-[8px]">{SCREENER_RESULTS.length} results</span>
      </div>
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/30">
        <span className="w-[50px]">Ticker</span>
        <span className="flex-1">Name</span>
        <span className="w-[80px]">Sector</span>
        <span className="w-[45px] text-right">P/E</span>
        <span className="w-[40px] text-right">RSI</span>
        <span className="w-[45px] text-right">Signal</span>
        <span className="w-[35px] text-right">Score</span>
      </div>
      <div className="flex-1 overflow-auto">
        {SCREENER_RESULTS.map((s) => (
          <div key={s.ticker} className="flex items-center px-2 py-1.5 border-b border-terminal-border/10 hover:bg-white/[0.02]">
            <span className="w-[50px] text-terminal-accent font-semibold">{s.ticker}</span>
            <span className="flex-1 text-terminal-text-muted truncate pr-2">{s.name}</span>
            <span className="w-[80px] text-terminal-text-faint truncate">{s.sector}</span>
            <span className="w-[45px] text-right tabular-nums">{s.pe > 0 ? s.pe.toFixed(1) : "—"}</span>
            <span className={`w-[40px] text-right tabular-nums ${s.rsi > 70 ? "text-terminal-negative" : s.rsi < 30 ? "text-terminal-positive" : ""}`}>
              {s.rsi}
            </span>
            <span className={`w-[45px] text-right font-semibold ${
              s.signal === "BUY" ? "text-terminal-positive" : s.signal === "SELL" ? "text-terminal-negative" : "text-terminal-text-muted"
            }`}>
              {s.signal}
            </span>
            <span className={`w-[35px] text-right tabular-nums ${s.score >= 70 ? "text-terminal-positive" : s.score <= 40 ? "text-terminal-negative" : "text-terminal-text-muted"}`}>
              {s.score}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ COMMAND LINE ═══════════

function CommandLine() {
  const [cmd, setCmd] = useState("");
  const [history, setHistory] = useState<string[]>([
    "> economy/fred --series GDP,UNRATE,CPIAUCSL",
    "  GDP: $27.36T (+2.8% QoQ)  |  UNRATE: 3.7%  |  CPI: 3.2% YoY",
    "> stocks/candle AAPL --ma 20,50 --volume",
    "  Loaded 252 bars for AAPL. SMA20=187.34 SMA50=182.12. Last: 189.45 (+0.65%)",
    "> portfolio/holdings --sort weight",
    "  12 positions | NAV $128,450,320 | Top: AAPL 8.5%, MSFT 7.8%, NVDA 6.2%",
  ]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!cmd.trim()) return;
    setHistory((prev) => [...prev, `> ${cmd}`, `  Processing: ${cmd}...`]);
    setCmd("");
  };

  return (
    <div className="h-full flex flex-col font-mono text-[10px]">
      <div className="flex-1 overflow-auto p-2">
        {history.map((line, i) => (
          <div key={i} className={`py-0.5 ${line.startsWith(">") ? "text-terminal-accent" : "text-terminal-text-muted"}`}>
            {line}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="flex items-center border-t border-terminal-border/50 px-2 py-1">
        <span className="text-terminal-accent mr-2">OpenBB &gt;</span>
        <input
          value={cmd}
          onChange={(e) => setCmd(e.target.value)}
          className="flex-1 bg-transparent outline-none text-terminal-text-primary placeholder:text-terminal-text-faint"
          placeholder="stocks/candle AAPL --ma 20,50"
        />
      </form>
    </div>
  );
}

// ═══════════ MAIN OPENBB TERMINAL ═══════════

export default function OpenBBTerminal() {
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const [rightPanel, setRightPanel] = useState<"watchlist" | "screener">("watchlist");
  const [bottomPanel, setBottomPanel] = useState<"command" | "screener">("command");

  return (
    <div className="h-full flex flex-col" data-testid="openbb-terminal">
      {/* Main grid */}
      <div className="flex-1 grid grid-cols-[240px_1fr_280px] grid-rows-[1fr_200px] gap-[2px] p-[2px] overflow-hidden">
        {/* Left: Watchlist */}
        <DashboardPanel
          title="WATCHLIST"
          className="row-span-2"
          headerRight={<span className="text-[8px] text-terminal-text-faint">{WATCHLIST.length} symbols</span>}
          noPadding
        >
          <Watchlist selected={selectedTicker} onSelect={setSelectedTicker} />
        </DashboardPanel>

        {/* Center: Chart */}
        <DashboardPanel
          title={`${selectedTicker} — CHART`}
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

        {/* Right: Screener / Details */}
        <DashboardPanel
          title={rightPanel === "watchlist" ? "ASSET DETAIL" : "SCREENER"}
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
            <div className="p-2 text-[10px] font-mono space-y-2">
              {(() => {
                const w = WATCHLIST.find((x) => x.ticker === selectedTicker) || WATCHLIST[0];
                return (
                  <>
                    <div className="text-center pb-2 border-b border-terminal-border/30">
                      <div className="text-terminal-accent text-sm font-semibold">{w.ticker}</div>
                      <div className="text-xl tabular-nums text-terminal-text-primary">{w.price.toFixed(2)}</div>
                      <div className={`text-xs tabular-nums ${w.chg >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                        {w.chg >= 0 ? "+" : ""}{w.chg.toFixed(2)} ({w.pct >= 0 ? "+" : ""}{w.pct.toFixed(2)}%)
                      </div>
                    </div>
                    {[
                      ["Mkt Cap", w.mktCap], ["Volume", w.vol],
                      ["P/E", "28.4"], ["EPS", "6.67"],
                      ["Div Yield", "0.52%"], ["Beta", "1.24"],
                      ["52W High", (+w.price * 1.15).toFixed(2)], ["52W Low", (+w.price * 0.72).toFixed(2)],
                      ["Avg Vol", "58.2M"], ["Float", "15.2B"],
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between">
                        <span className="text-terminal-text-faint">{k}</span>
                        <span className="text-terminal-text-primary tabular-nums">{v}</span>
                      </div>
                    ))}
                  </>
                );
              })()}
            </div>
          )}
        </DashboardPanel>

        {/* Bottom: Command line / Screener */}
        <div className="col-span-2 flex flex-col">
          <DashboardPanel
            title="OPENBB COMMAND LINE"
            headerRight={
              <div className="flex gap-1">
                {["economy", "stocks", "crypto", "forex", "portfolio", "etf"].map((cat) => (
                  <span key={cat} className="px-1 py-0.5 text-[7px] text-terminal-text-faint bg-terminal-surface-2/50 rounded">{cat}</span>
                ))}
              </div>
            }
            className="flex-1"
            noPadding
          >
            <CommandLine />
          </DashboardPanel>
        </div>
      </div>
    </div>
  );
}
