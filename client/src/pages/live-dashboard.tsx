import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { LiveTicker } from "@/components/live-ticker";
import { ResizableDashboard } from "@/components/resizable-panel";
import {
  AreaChart, Area, XAxis, YAxis, ResponsiveContainer, LineChart, Line, Tooltip, PieChart, Pie, Cell, BarChart, Bar,
} from "recharts";
import { useEngineQuery, type PortfolioLive, type TradeRecord, type RiskPortfolio, type RiskGreeks } from "@/hooks/use-engine-api";

// ═══════════ SIMULATED DATA GENERATORS ═══════════

interface LiveTx {
  id: string;
  time: string;
  ticker: string;
  side: "BUY" | "SELL" | "SHORT" | "COVER";
  qty: number;
  price: number;
  notional: number;
  fillType: "FULL" | "PARTIAL" | "REJECTED";
  venue: string;
  signal: string;
  latencyMs: number;
}

const TX_TICKERS: Record<string, number> = {
  AAPL: 189, MSFT: 420, NVDA: 875, AMZN: 185, GOOGL: 155,
  META: 505, JPM: 198, TSLA: 178, XOM: 115, UNH: 502,
  V: 282, BAC: 35, GS: 437, LLY: 803, AVGO: 1357,
};
const TX_VENUES = ["ARCA", "NYSE", "NASDAQ", "BATS", "IEX", "EDGX", "DARK"];
const TX_SIGNALS = ["ML_AGENT", "MICRO_PX", "RV_PAIR", "SOCIAL", "CVR", "EVENT", "DRL", "MOM", "TFT"];

let _txSeq = 100000;
function generateLiveTx(): LiveTx {
  const tickers = Object.keys(TX_TICKERS);
  const ticker = tickers[Math.floor(Math.random() * tickers.length)];
  const base = TX_TICKERS[ticker];
  const side = (["BUY", "SELL", "SHORT", "COVER"] as const)[Math.floor(Math.random() * 4)];
  const qty = Math.floor(50 + Math.random() * 950);
  const price = +(base * (1 + (Math.random() - 0.5) * 0.02)).toFixed(2);
  const now = new Date();
  _txSeq++;
  return {
    id: `TX-${_txSeq.toString(36).toUpperCase()}`,
    time: now.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }) + "." + now.getMilliseconds().toString().padStart(3, "0"),
    ticker,
    side,
    qty,
    price,
    notional: +(qty * price),
    fillType: Math.random() > 0.05 ? (Math.random() > 0.1 ? "FULL" : "PARTIAL") : "REJECTED",
    venue: TX_VENUES[Math.floor(Math.random() * TX_VENUES.length)],
    signal: TX_SIGNALS[Math.floor(Math.random() * TX_SIGNALS.length)],
    latencyMs: +(0.5 + Math.random() * 12).toFixed(1),
  };
}

function generateInitialTxs(count: number): LiveTx[] {
  return Array.from({ length: count }, () => generateLiveTx()).reverse();
}

function generatePnlData() {
  const data: { time: string; value: number }[] = [];
  let v = 800;
  for (let i = 0; i < 60; i++) {
    v += Math.random() * 40 - 15;
    const h = Math.floor(i / 4) + 9;
    const m = (i % 4) * 15;
    data.push({ time: `${h}:${m.toString().padStart(2, "0")}`, value: Math.round(v) });
  }
  return data;
}

function generateLiquidityData() {
  const d: { time: string; bid: number; ask: number }[] = [];
  for (let i = 0; i < 30; i++) {
    d.push({
      time: `${(1556 + i * 0.1).toFixed(1)}`,
      bid: Math.random() * 5000 + 8000,
      ask: Math.random() * 5000 + 8000,
    });
  }
  return d;
}

function generateSpreadData() {
  const d: { time: string; spread: number }[] = [];
  for (let i = 0; i < 40; i++) {
    d.push({ time: `${i}`, spread: Math.random() * 0.3 + 0.02 });
  }
  return d;
}

function generateDepthData() {
  const d: { price: string; bidDepth: number; askDepth: number }[] = [];
  for (let i = 0; i < 20; i++) {
    const p = 57000 + i * 200;
    d.push({
      price: p.toFixed(0),
      bidDepth: i < 10 ? (10 - i) * 1000 + Math.random() * 2000 : 0,
      askDepth: i >= 10 ? (i - 9) * 1000 + Math.random() * 2000 : 0,
    });
  }
  return d;
}

const ORDER_DISTRIBUTION = [
  { name: "Market", value: 29.6, color: "#00d4aa" },
  { name: "Limit", value: 3.5, color: "#58a6ff" },
  { name: "Stop", value: 10, color: "#f85149" },
  { name: "Conditional", value: 16, color: "#bc8cff" },
  { name: "TWAP", value: 8.4, color: "#d29922" },
  { name: "Bracket", value: 14, color: "#4ecdc4" },
  { name: "Trailing", value: 18.5, color: "#3fb950" },
];

const EXECUTIONS = [
  { pair: "AAPL", ctrl: 597, sens: 860, status: 506.23, positive: true },
  { pair: "MSFT", ctrl: 398, sens: 956, status: 505.38, positive: true },
  { pair: "BTC/USD", ctrl: 5920.0, sens: 544, status: 508.52, positive: true },
  { pair: "KWTC/DAY", ctrl: 3022.06, sens: 401, status: -796.58, positive: false },
];

const TOP_RISKS = [
  { name: "Tech Concentration", value: "34.2%", severity: "high" },
  { name: "Rising Volatility", value: "VIX 21.8", severity: "medium" },
  { name: "Correlation Breakdown", value: "ρ=0.92", severity: "low" },
];

// ═══════════ NEURAL MARKET MAP (Canvas) ═══════════

interface MapNode {
  id: string; x: number; y: number; radius: number; label: string; momentum: number;
}
interface MapEdge {
  source: number; target: number; weight: number;
}
interface Particle {
  edgeIdx: number; t: number; speed: number;
}

const MAP_NODES: MapNode[] = [
  { id: "TECH", x: 0.3, y: 0.25, radius: 8, label: "TECH", momentum: 0.8 },
  { id: "FIN", x: 0.65, y: 0.55, radius: 7, label: "FIN", momentum: 0.3 },
  { id: "ENERGY", x: 0.75, y: 0.2, radius: 6, label: "ENERGY", momentum: -0.4 },
  { id: "HEALTH", x: 0.2, y: 0.6, radius: 6, label: "HEALTH", momentum: 0.5 },
  { id: "CONS", x: 0.5, y: 0.4, radius: 5, label: "CONS", momentum: 0.2 },
  { id: "IND", x: 0.4, y: 0.7, radius: 5, label: "IND", momentum: 0.1 },
  { id: "UTIL", x: 0.8, y: 0.7, radius: 4, label: "UTIL", momentum: -0.1 },
  { id: "MAT", x: 0.15, y: 0.35, radius: 4, label: "MAT", momentum: 0.4 },
  { id: "RE", x: 0.6, y: 0.15, radius: 4, label: "RE", momentum: -0.2 },
  { id: "COMM", x: 0.85, y: 0.45, radius: 5, label: "COMM", momentum: 0.6 },
];

const MAP_EDGES: MapEdge[] = [
  { source: 0, target: 1, weight: 0.7 }, { source: 0, target: 4, weight: 0.8 },
  { source: 0, target: 7, weight: 0.4 }, { source: 1, target: 2, weight: 0.5 },
  { source: 1, target: 4, weight: 0.6 }, { source: 1, target: 6, weight: 0.3 },
  { source: 2, target: 8, weight: 0.5 }, { source: 2, target: 9, weight: 0.4 },
  { source: 3, target: 5, weight: 0.6 }, { source: 3, target: 7, weight: 0.5 },
  { source: 4, target: 5, weight: 0.5 }, { source: 4, target: 9, weight: 0.4 },
  { source: 5, target: 6, weight: 0.3 }, { source: 6, target: 9, weight: 0.2 },
  { source: 0, target: 9, weight: 0.5 }, { source: 3, target: 4, weight: 0.4 },
  { source: 7, target: 8, weight: 0.3 }, { source: 1, target: 5, weight: 0.3 },
];

const ANNOTATIONS = [
  { text: "Tech showing strong momentum", x: 0.18, y: 0.12 },
  { text: "Energy weakening", x: 0.68, y: 0.10 },
  { text: "Financial flows diverging", x: 0.55, y: 0.42 },
  { text: "TECH Biosilme commons", x: 0.35, y: 0.50 },
];

function NeuralMarketMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animRef = useRef<number>(0);
  const nodesRef = useRef(MAP_NODES.map((n) => ({ ...n })));

  useEffect(() => {
    // Initialize particles
    const parts: Particle[] = [];
    for (let i = 0; i < 80; i++) {
      parts.push({
        edgeIdx: Math.floor(Math.random() * MAP_EDGES.length),
        t: Math.random(),
        speed: 0.001 + Math.random() * 0.003,
      });
    }
    particlesRef.current = parts;
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    // Clear
    ctx.clearRect(0, 0, W, H);

    const nodes = nodesRef.current;

    // Subtle floating animation
    const t = Date.now() * 0.0005;
    nodes.forEach((n, i) => {
      n.x = MAP_NODES[i].x + Math.sin(t + i * 1.5) * 0.008;
      n.y = MAP_NODES[i].y + Math.cos(t + i * 2.1) * 0.006;
    });

    // Draw edges
    MAP_EDGES.forEach((e) => {
      const s = nodes[e.source];
      const tgt = nodes[e.target];
      ctx.beginPath();
      ctx.moveTo(s.x * W, s.y * H);
      ctx.lineTo(tgt.x * W, tgt.y * H);
      ctx.strokeStyle = `rgba(0, 212, 170, ${0.06 + e.weight * 0.08})`;
      ctx.lineWidth = 0.5 + e.weight * 0.5;
      ctx.stroke();
    });

    // Draw particles
    particlesRef.current.forEach((p) => {
      p.t += p.speed;
      if (p.t > 1) {
        p.t = 0;
        p.edgeIdx = Math.floor(Math.random() * MAP_EDGES.length);
      }
      const e = MAP_EDGES[p.edgeIdx];
      const s = nodes[e.source];
      const tgt = nodes[e.target];
      const px = (s.x + (tgt.x - s.x) * p.t) * W;
      const py = (s.y + (tgt.y - s.y) * p.t) * H;

      ctx.beginPath();
      ctx.arc(px, py, 1.2, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 212, 170, ${0.4 + Math.sin(p.t * Math.PI) * 0.5})`;
      ctx.fill();
    });

    // Draw nodes with glow
    nodes.forEach((n) => {
      const nx = n.x * W;
      const ny = n.y * H;
      const r = n.radius;

      // Glow
      const grad = ctx.createRadialGradient(nx, ny, 0, nx, ny, r * 4);
      const glowAlpha = n.momentum > 0 ? 0.08 + n.momentum * 0.12 : 0.04;
      grad.addColorStop(0, `rgba(0, 212, 170, ${glowAlpha})`);
      grad.addColorStop(1, "rgba(0, 212, 170, 0)");
      ctx.beginPath();
      ctx.arc(nx, ny, r * 4, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // Node
      ctx.beginPath();
      ctx.arc(nx, ny, r, 0, Math.PI * 2);
      const nodeColor = n.momentum > 0.3
        ? "rgba(0, 212, 170, 0.8)"
        : n.momentum < -0.1
        ? "rgba(248, 81, 73, 0.6)"
        : "rgba(0, 212, 170, 0.4)";
      ctx.fillStyle = nodeColor;
      ctx.fill();
      ctx.strokeStyle = "rgba(0, 212, 170, 0.5)";
      ctx.lineWidth = 0.8;
      ctx.stroke();

      // Label
      ctx.font = "9px 'JetBrains Mono'";
      ctx.fillStyle = "rgba(230, 237, 243, 0.7)";
      ctx.textAlign = "center";
      ctx.fillText(n.label, nx, ny + r + 12);
    });

    // Annotations
    ctx.font = "9px 'Inter'";
    ANNOTATIONS.forEach((a) => {
      ctx.fillStyle = "rgba(230, 237, 243, 0.45)";
      const ax = a.x * W;
      const ay = a.y * H;
      // Small arrow
      ctx.fillStyle = "rgba(0, 212, 170, 0.5)";
      ctx.fillText("▸", ax - 8, ay);
      ctx.fillStyle = "rgba(230, 237, 243, 0.45)";
      ctx.fillText(a.text, ax + 2, ay);
    });

    // Background grid dots
    for (let gx = 0; gx < W; gx += 40) {
      for (let gy = 0; gy < H; gy += 40) {
        ctx.beginPath();
        ctx.arc(gx, gy, 0.3, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0, 212, 170, 0.06)";
        ctx.fill();
      }
    }

    animRef.current = requestAnimationFrame(draw);
  }, []);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <canvas ref={canvasRef} className="absolute inset-0" />
    </div>
  );
}

// ═══════════ LIVE TRANSACTIONS COMPONENT ═══════════

function LiveTransactions() {
  const { data: tradesData } = useEngineQuery<{ trades: TradeRecord[] }>("/portfolio/trades?limit=200", { refetchInterval: 3000 });
  const [flashId, setFlashId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevCountRef = useRef(0);

  // Map API trades to LiveTx format
  const txns: LiveTx[] = useMemo(() => {
    if (!tradesData?.trades?.length) return [];
    return tradesData.trades.map((t) => {
      const side = (t.side?.toUpperCase() || "BUY") as LiveTx["side"];
      const notional = t.quantity * t.fill_price;
      return {
        id: t.id || `TX-${Math.random().toString(36).slice(2, 8).toUpperCase()}`,
        time: t.timestamp ? new Date(t.timestamp).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "--:--:--",
        ticker: t.ticker,
        side,
        qty: t.quantity,
        price: t.fill_price,
        notional,
        fillType: "FULL" as const,
        venue: "ENGINE",
        signal: t.signal_type || "UNKNOWN",
        latencyMs: 0,
      };
    });
  }, [tradesData]);

  // Flash new entries
  useEffect(() => {
    if (txns.length > prevCountRef.current && txns.length > 0) {
      setFlashId(txns[0].id);
      setTimeout(() => setFlashId(null), 800);
    }
    prevCountRef.current = txns.length;
  }, [txns]);

  // Running stats
  const filled = txns.filter((t) => t.fillType !== "REJECTED");
  const totalNotional = filled.reduce((s, t) => s + t.notional, 0);
  const buys = filled.filter((t) => t.side === "BUY" || t.side === "COVER").length;
  const sells = filled.filter((t) => t.side === "SELL" || t.side === "SHORT").length;
  const avgLatency = filled.length ? filled.reduce((s, t) => s + t.latencyMs, 0) / filled.length : 0;

  // Auto-scroll to top on new tx
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [txns.length]);

  const fmtNotional = (n: number) => n >= 1e6 ? `$${(n / 1e6).toFixed(1)}M` : `$${(n / 1e3).toFixed(0)}K`;

  return (
    <div className="flex flex-col h-full text-[10px] font-mono tabular-nums">
      {/* Summary header */}
      <div className="px-2 py-1.5 border-b border-terminal-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-terminal-positive animate-pulse" />
            <span className="text-[9px] text-terminal-text-faint uppercase tracking-wider">Live Feed</span>
          </div>
          <span className="text-[9px] text-terminal-text-faint">{txns.length} txns</span>
        </div>
        <div className="flex items-baseline gap-2 mt-1">
          <span className="text-lg font-bold text-terminal-text-primary">
            {fmtNotional(totalNotional)}
          </span>
          <span className="text-[9px] text-terminal-text-faint">volume</span>
        </div>
      </div>

      {/* Mini stats row */}
      <div className="grid grid-cols-3 gap-1 px-2 py-1 border-b border-terminal-border/50 text-[8px]">
        <div>
          <div className="text-terminal-text-faint">Fills</div>
          <div className="text-terminal-text-primary">{filled.length}</div>
        </div>
        <div>
          <div className="text-terminal-text-faint">B / S</div>
          <div>
            <span className="text-terminal-positive">{buys}</span>
            <span className="text-terminal-text-faint"> / </span>
            <span className="text-terminal-negative">{sells}</span>
          </div>
        </div>
        <div>
          <div className="text-terminal-text-faint">Avg Lat</div>
          <div className="text-terminal-text-primary">{avgLatency.toFixed(1)}ms</div>
        </div>
      </div>

      {/* Column headers */}
      <div className="flex items-center px-2 py-1 text-[7px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
        <span className="w-[52px]">Time</span>
        <span className="w-[38px]">Ticker</span>
        <span className="w-[30px]">Side</span>
        <span className="w-[32px] text-right">Qty</span>
        <span className="flex-1 text-right">Price</span>
      </div>

      {/* Scrollable transaction rows */}
      <div ref={scrollRef} className="flex-1 overflow-auto">
        {txns.map((tx) => (
          <div
            key={tx.id}
            className={`flex items-center px-2 py-[3px] border-b border-terminal-border/10 transition-colors duration-700 ${
              tx.fillType === "REJECTED" ? "opacity-40" : ""
            } ${flashId === tx.id ? "bg-terminal-accent/10" : "hover:bg-white/[0.02]"}`}
          >
            <span className="w-[52px] text-terminal-text-faint text-[9px]">{tx.time.slice(0, 8)}</span>
            <span className="w-[38px] text-terminal-accent font-semibold text-[9px]">{tx.ticker}</span>
            <span className={`w-[30px] font-semibold text-[8px] ${
              tx.side === "BUY" || tx.side === "COVER" ? "text-terminal-positive" : "text-terminal-negative"
            }`}>
              {tx.side}
            </span>
            <span className="w-[32px] text-right text-terminal-text-muted text-[9px]">{tx.qty}</span>
            <span className="flex-1 text-right text-terminal-text-primary text-[9px]">${tx.price.toFixed(2)}</span>
          </div>
        ))}
      </div>

      {/* Bottom detail strip */}
      <div className="px-2 py-1.5 border-t border-terminal-border">
        <div className="grid grid-cols-3 gap-1 text-[8px]">
          <div>
            <div className="text-terminal-text-faint">Venues</div>
            <div className="text-terminal-text-primary truncate">
              {Array.from(new Set(txns.slice(0, 20).map(t => t.venue))).slice(0, 3).join(", ")}
            </div>
          </div>
          <div>
            <div className="text-terminal-text-faint">Signals</div>
            <div className="text-terminal-accent truncate">
              {Array.from(new Set(txns.slice(0, 20).map(t => t.signal))).slice(0, 2).join(", ")}
            </div>
          </div>
          <div>
            <div className="text-terminal-text-faint">Fill Rate</div>
            <div className="text-terminal-positive">{txns.length ? ((filled.length / txns.length) * 100).toFixed(1) : 0}%</div>
          </div>
        </div>
      </div>

      {/* Mini activity spark */}
      <div className="px-2 pb-1.5 h-[32px]">
        <div className="flex items-end gap-[1px] h-full">
          {txns.slice(0, 40).map((tx, i) => {
            const maxQ = Math.max(...txns.slice(0, 40).map(t => t.qty), 1);
            const h = (tx.qty / maxQ) * 100;
            const isBuy = tx.side === "BUY" || tx.side === "COVER";
            return (
              <div
                key={tx.id + i}
                className="flex-1 rounded-sm"
                style={{
                  height: `${Math.max(h, 5)}%`,
                  backgroundColor: isBuy ? "rgba(63, 185, 80, 0.5)" : "rgba(248, 81, 73, 0.5)",
                }}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ═══════════ ORDER DISTRIBUTION (Donut) ═══════════

function OrderDistributionChart() {
  const { data: distData } = useEngineQuery<{ distribution: Array<{ name: string; value: number; color: string }> }>("/risk/order-distribution", { refetchInterval: 10000 });
  const chartData = distData?.distribution?.length ? distData.distribution : ORDER_DISTRIBUTION;

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 relative">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              innerRadius="55%"
              outerRadius="80%"
              paddingAngle={2}
              dataKey="value"
              stroke="none"
            >
              {chartData.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-lg font-bold text-terminal-text-primary font-mono">25%</span>
          <span className="text-[8px] text-terminal-text-faint font-mono">14 ARRAY</span>
        </div>
      </div>
      {/* Legend */}
      <div className="px-2 pb-1 grid grid-cols-2 gap-x-3 gap-y-0.5 text-[8px]">
        {chartData.map((d, i) => (
          <div key={i} className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: d.color }} />
            <span className="text-terminal-text-muted truncate">{d.name}</span>
            <span className="ml-auto text-terminal-text-primary tabular-nums">{d.value}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ LIQUIDITY CHART ═══════════

function LiquidityChart() {
  const { data: liqApi } = useEngineQuery<{ liquidity: Array<{ time: string; bid: number; ask: number }> }>("/execution/liquidity-data", { refetchInterval: 30000 });
  const data = liqApi?.liquidity?.length ? liqApi.liquidity : useMemo(generateLiquidityData, []);
  return (
    <div className="h-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
          <defs>
            <linearGradient id="bidGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#00d4aa" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#00d4aa" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="askGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f85149" stopOpacity={0.2} />
              <stop offset="100%" stopColor="#f85149" stopOpacity={0} />
            </linearGradient>
          </defs>
          <Area type="monotone" dataKey="bid" stroke="#00d4aa" strokeWidth={1} fill="url(#bidGrad)" />
          <Area type="monotone" dataKey="ask" stroke="#f85149" strokeWidth={1} fill="url(#askGrad)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ═══════════ SPREAD CHART ═══════════

function SpreadChart() {
  const { data: spreadApi } = useEngineQuery<{ spreads: Array<{ time: string; spread: number }> }>("/execution/spread-data", { refetchInterval: 30000 });
  const data = spreadApi?.spreads?.length ? spreadApi.spreads : useMemo(generateSpreadData, []);
  return (
    <div className="h-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
          <Line type="monotone" dataKey="spread" stroke="#58a6ff" strokeWidth={1} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ═══════════ DEPTH CHART ═══════════

function DepthChart() {
  const { data: depthApi } = useEngineQuery<{ depth: Array<{ price: string; bidDepth: number; askDepth: number }> }>("/execution/depth-data", { refetchInterval: 30000 });
  const data = depthApi?.depth?.length ? depthApi.depth : useMemo(generateDepthData, []);
  return (
    <div className="h-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
          <defs>
            <linearGradient id="depthBid" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3fb950" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#3fb950" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="depthAsk" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f85149" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#f85149" stopOpacity={0} />
            </linearGradient>
          </defs>
          <Area type="stepAfter" dataKey="bidDepth" stroke="#3fb950" strokeWidth={1} fill="url(#depthBid)" />
          <Area type="stepAfter" dataKey="askDepth" stroke="#f85149" strokeWidth={1} fill="url(#depthAsk)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ═══════════ RISK PANEL ═══════════

function RiskPanel() {
  const { data: riskData } = useEngineQuery<RiskPortfolio>("/risk/portfolio", { refetchInterval: 5000 });
  const { data: portData } = useEngineQuery<PortfolioLive>("/portfolio/live", { refetchInterval: 5000 });
  const { data: alertsData } = useEngineQuery<{ alerts: Array<{ name: string; value: string; severity: string }> }>("/risk/alerts", { refetchInterval: 10000 });
  const riskAlerts = alertsData?.alerts?.length ? alertsData.alerts : TOP_RISKS;

  const beta = riskData?.current_beta ?? 0;
  const targetBeta = riskData?.target_beta ?? 0;
  const corridor = riskData?.corridor_position ?? "--";
  const pnl = portData?.total_pnl ?? 0;
  const nav = portData?.nav ?? 0;
  const exposure = portData?.gross_exposure ?? 0;
  const wins = portData?.win_count ?? 0;
  const losses = portData?.loss_count ?? 0;
  const winRate = (wins + losses) > 0 ? ((wins / (wins + losses)) * 100).toFixed(1) : "0";

  return (
    <div className="flex flex-col h-full text-[10px] font-mono tabular-nums p-2 gap-2">
      <div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-terminal-text-primary">${Math.abs(pnl).toLocaleString()}</span>
          <span className="text-terminal-text-muted">(β {beta.toFixed(2)})</span>
        </div>
        <div className="flex items-center gap-3 mt-1">
          <span className="text-terminal-text-faint">Corridor</span>
          <span className="text-terminal-text-primary">{corridor}</span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Target Beta</span>
          <span className="text-terminal-text-primary">{targetBeta.toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">NAV</span>
          <span className="text-terminal-text-primary">${nav >= 1e6 ? (nav / 1e6).toFixed(1) + "M" : nav.toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Beta</span>
          <span className="text-terminal-text-primary">{beta.toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Exposure</span>
          <span className="text-terminal-accent">{(exposure * 100).toFixed(1)}%</span>
        </div>
      </div>

      <div className="border-t border-terminal-border pt-1.5">
        <div className="grid grid-cols-3 gap-x-3 gap-y-0.5 text-[10px]">
          <div className="flex justify-between">
            <span className="text-terminal-text-faint">Positions</span>
            <span>{portData?.positions_count ?? 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-terminal-text-faint">W/L</span>
            <span>{wins}/{losses}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-terminal-text-faint">Win%</span>
            <span>{winRate}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-terminal-text-faint">Cash</span>
            <span>${((portData?.cash ?? 0) / 1e3).toFixed(0)}K</span>
          </div>
          <div className="flex justify-between">
            <span className="text-terminal-text-faint">Net Exp</span>
            <span>{((portData?.net_exposure ?? 0) * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-terminal-text-faint">P&L</span>
            <span className={pnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}>{pnl >= 0 ? "+" : ""}{pnl.toLocaleString()}</span>
          </div>
        </div>
      </div>

      <div className="border-t border-terminal-border pt-1.5 flex-1">
        <div className="text-[9px] text-terminal-text-muted font-semibold tracking-wider uppercase mb-1">
          TOP RISKS <span className="text-terminal-text-faint ml-2">{corridor}</span>
        </div>
        {riskAlerts.map((r, i) => (
          <div key={i} className="flex items-center gap-2 py-0.5">
            <span className="text-terminal-text-faint">{i === 0 ? "▲" : "○"}</span>
            <span className="text-terminal-text-muted flex-1">{r.name}</span>
            <span
              className={
                r.severity === "high"
                  ? "text-terminal-negative"
                  : r.severity === "medium"
                  ? "text-terminal-warning"
                  : "text-terminal-text-muted"
              }
            >
              {r.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ PNL / TIME CHART ═══════════

function PnlTimeChart() {
  const { data: pnlApi } = useEngineQuery<{ series: Array<{ time: string; value: number }> }>("/portfolio/pnl-series", { refetchInterval: 10000 });
  const data = pnlApi?.series?.length ? pnlApi.series : useMemo(generatePnlData, []);
  return (
    <div className="h-full flex flex-col">
      <div className="flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: 8 }}>
            <defs>
              <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#00d4aa" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#00d4aa" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="time"
              tick={{ fontSize: 8, fill: "#484f58" }}
              axisLine={{ stroke: "#1e2633" }}
              tickLine={false}
              interval={9}
            />
            <YAxis
              tick={{ fontSize: 8, fill: "#484f58" }}
              axisLine={false}
              tickLine={false}
              width={35}
            />
            <Area type="monotone" dataKey="value" stroke="#00d4aa" strokeWidth={1.5} fill="url(#pnlGrad)" />
            <Tooltip
              contentStyle={{
                background: "#0d1117",
                border: "1px solid #1e2633",
                borderRadius: 4,
                fontSize: 10,
                fontFamily: "JetBrains Mono",
                color: "#e6edf3",
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      {/* Decomposition bars */}
      <div className="px-2 pb-1 flex items-center gap-3 text-[8px] font-mono">
        <span className="text-terminal-text-faint">PNL ESFE</span>
        <div className="flex items-center gap-1">
          <span className="text-terminal-text-faint">GRTE</span>
          <div className="w-16 h-2 bg-terminal-positive/40 rounded-sm" />
        </div>
        <div className="flex items-center gap-1">
          <span className="text-terminal-text-faint">SHORTFALL</span>
          <div className="w-10 h-2 bg-terminal-negative/40 rounded-sm" />
        </div>
        <div className="flex items-center gap-1">
          <span className="text-terminal-text-faint">PORTFOLIO</span>
          <div className="w-12 h-2 bg-terminal-blue/40 rounded-sm" />
        </div>
      </div>
    </div>
  );
}

// ═══════════ EXECUTION TABLE ═══════════

function ExecutionTable() {
  const { data: tcaApi } = useEngineQuery<{ trades: Array<{ ticker: string; quantity: number; fill_price: number; slippage: number; side: string }>; summary: { fill_rate: number; avg_slippage: number; total_trades: number } }>("/execution/tca", { refetchInterval: 10000 });

  const executions = useMemo(() => {
    if (!tcaApi?.trades?.length) return EXECUTIONS;
    return tcaApi.trades.slice(0, 6).map((t) => ({
      pair: t.ticker,
      ctrl: t.quantity,
      sens: Math.round(t.fill_price),
      status: t.fill_price * t.quantity * (t.slippage || 0.001),
      positive: t.side?.toUpperCase() !== "SELL",
    }));
  }, [tcaApi]);

  return (
    <div className="flex flex-col h-full text-[10px] font-mono tabular-nums">
      {/* Header */}
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
        <span className="w-[70px]">Pair</span>
        <span className="w-[55px] text-right">Ctrl</span>
        <span className="w-[45px] text-right">Sens</span>
        <span className="flex-1 text-right">Status</span>
      </div>
      <div className="flex-1 overflow-auto">
        {executions.map((e, i) => (
          <div
            key={i}
            className="flex items-center px-2 py-1.5 border-b border-terminal-border/20 hover:bg-white/[0.02]"
          >
            <span className="w-[70px] text-terminal-text-primary font-medium">{e.pair}</span>
            <span className="w-[55px] text-right text-terminal-text-muted">{e.ctrl}</span>
            <span className="w-[45px] text-right text-terminal-text-muted">{e.sens}</span>
            <span className={`flex-1 text-right font-medium ${e.positive ? "text-terminal-positive" : "text-terminal-negative"}`}>
              {e.positive ? "+" : ""}{e.status.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
      {/* Footer stats */}
      <div className="px-2 py-1.5 border-t border-terminal-border text-[8px]">
        <div className="flex justify-between mb-0.5">
          <span className="text-terminal-text-faint">FONDARTRET IONS</span>
          <span className="text-terminal-text-muted">L TIMING <span className="text-terminal-text-primary">0.48</span></span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">SHREED</span>
          <span className="text-terminal-text-primary">1291</span>
          <span className="text-terminal-text-faint">FRONSIT</span>
          <span className="text-terminal-text-muted">60802%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">ToSS cost</span>
          <span className="text-terminal-text-primary">46.7%</span>
          <span className="text-terminal-text-primary">0.78</span>
        </div>
      </div>
    </div>
  );
}

// ═══════════ MAIN LIVE DASHBOARD ═══════════

export default function LiveDashboard() {
  return (
    <div className="h-full flex flex-col" data-testid="live-dashboard">
      {/* Main 3-column resizable layout */}
      <div className="flex-1 overflow-hidden p-[2px]">
        <ResizableDashboard
          defaultSizes={[22, 52, 26]}
          minSizes={[14, 30, 16]}
          className="gap-0"
        >
          {/* Left column: Live Transactions */}
          <div className="h-full p-[1px]">
            <DashboardPanel title="LIVE TRANSACTIONS" className="h-full" noPadding>
              <LiveTransactions />
            </DashboardPanel>
          </div>

          {/* Center column: Neural Market Map + Risk/PNL */}
          <div className="h-full flex flex-col gap-[2px] px-[1px]">
            {/* Center top: Neural Market Map */}
            <DashboardPanel
              title="NEURAL MARKET MAP"
              className="flex-1"
              headerRight={
                <div className="flex gap-1">
                  {["BIS", "DMS", "NASCR", "PNBL"].map((t, i) => (
                    <button
                      key={t}
                      className={`px-1.5 py-0.5 text-[8px] rounded-sm ${
                        i === 2 ? "bg-terminal-accent/15 text-terminal-accent" : "text-terminal-text-faint hover:text-terminal-text-muted"
                      }`}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              }
              noPadding
            >
              <NeuralMarketMap />
            </DashboardPanel>

            {/* Center bottom: Risk + PNL/TIME */}
            <div className="flex gap-[2px] h-[45%]">
              <DashboardPanel title="RISK" className="w-[200px] flex-shrink-0" noPadding>
                <RiskPanel />
              </DashboardPanel>
              <DashboardPanel title="PNL / TIME" className="flex-1" noPadding>
                <PnlTimeChart />
              </DashboardPanel>
            </div>
          </div>

          {/* Right column: Order Distribution + Liquidity + Spread + Depth + Execution */}
          <div className="h-full flex flex-col gap-[2px] p-[1px]">
            <DashboardPanel title="ORDER DISTRIBUTION" className="flex-[1.2]" noPadding>
              <OrderDistributionChart />
            </DashboardPanel>
            <DashboardPanel
              title="LIQUIDITY"
              className="flex-1"
              headerRight={<span className="text-[8px] text-terminal-accent tabular-nums font-mono">88,888</span>}
              noPadding
            >
              <LiquidityChart />
            </DashboardPanel>
            <DashboardPanel
              title="SPREAD"
              className="flex-1"
              headerRight={<span className="text-[8px] text-terminal-blue font-mono">ESFT</span>}
              noPadding
            >
              <SpreadChart />
            </DashboardPanel>
            <DashboardPanel title="DEPTH" className="flex-1" noPadding>
              <DepthChart />
            </DashboardPanel>
            <DashboardPanel title="EXECUTION" className="flex-[1.3]" noPadding>
              <ExecutionTable />
            </DashboardPanel>
          </div>
        </ResizableDashboard>
      </div>

      {/* Bottom ticker */}
      <LiveTicker />
    </div>
  );
}
