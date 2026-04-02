import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { LiveTicker } from "@/components/live-ticker";
import { ResizableDashboard } from "@/components/resizable-panel";
import {
  AreaChart, Area, XAxis, YAxis, ResponsiveContainer, LineChart, Line, Tooltip, PieChart, Pie, Cell, BarChart, Bar,
} from "recharts";

// ═══════════ SIMULATED DATA GENERATORS ═══════════

function generateOrderBook() {
  const mid = 59238.29;
  const asks: { price: number; size: number; total: number }[] = [];
  const bids: { price: number; size: number; total: number }[] = [];
  let askTotal = 0;
  let bidTotal = 0;
  for (let i = 0; i < 14; i++) {
    const askSize = Math.floor(Math.random() * 3000 + 500);
    askTotal += askSize;
    asks.push({ price: mid + (14 - i) * (Math.random() * 50 + 20), size: askSize, total: askTotal });
    const bidSize = Math.floor(Math.random() * 3000 + 500);
    bidTotal += bidSize;
    bids.push({ price: mid - (i + 1) * (Math.random() * 50 + 20), size: bidSize, total: bidTotal });
  }
  asks.reverse();
  return { asks, bids, mid };
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

// ═══════════ ORDER BOOK COMPONENT ═══════════

function OrderBook() {
  const [book, setBook] = useState(generateOrderBook);
  const [midPrice, setMidPrice] = useState(59238.29);
  const [change, setChange] = useState(0.73);

  useEffect(() => {
    const iv = setInterval(() => {
      setBook(generateOrderBook());
      setMidPrice((p) => p + (Math.random() - 0.48) * 10);
      setChange((c) => c + (Math.random() - 0.5) * 0.1);
    }, 2500);
    return () => clearInterval(iv);
  }, []);

  const maxTotal = Math.max(
    ...book.asks.map((a) => a.total),
    ...book.bids.map((b) => b.total)
  );

  return (
    <div className="flex flex-col h-full text-[10px] font-mono tabular-nums">
      {/* Price header */}
      <div className="px-2 py-1.5 border-b border-terminal-border">
        <div className="flex items-baseline gap-1.5">
          <span className="text-[9px] text-terminal-text-faint">$ Price Market</span>
          <span className="text-[9px] text-terminal-text-faint">● TOTAL</span>
          <span className="ml-auto text-[9px] text-terminal-text-faint">100%</span>
        </div>
        <div className="flex items-baseline gap-2 mt-0.5">
          <span className="text-lg font-bold text-terminal-text-primary">
            {midPrice.toFixed(2)}
          </span>
          <span className={`text-xs ${change >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
            {change >= 0 ? "+" : ""}{change.toFixed(2)}%
          </span>
        </div>
      </div>

      {/* Column headers */}
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
        <span className="w-[60px]">Price</span>
        <span className="w-[50px] text-right">Size</span>
        <span className="w-[60px] text-right">Total</span>
        <span className="flex-1 text-right">Bars</span>
      </div>

      {/* Asks (reversed) */}
      <div className="flex-1 overflow-auto">
        {book.asks.map((a, i) => (
          <div key={`a${i}`} className="flex items-center px-2 py-[2px] relative">
            <div
              className="absolute right-0 top-0 bottom-0 bg-terminal-negative/8"
              style={{ width: `${(a.total / maxTotal) * 100}%` }}
            />
            <span className="w-[60px] text-terminal-negative relative z-10">{a.price.toFixed(2)}</span>
            <span className="w-[50px] text-right text-terminal-text-muted relative z-10">{a.size.toLocaleString()}</span>
            <span className="w-[60px] text-right text-terminal-text-muted relative z-10">{a.total.toLocaleString()}</span>
          </div>
        ))}
        {/* Spread indicator */}
        <div className="flex items-center justify-center py-1 border-y border-terminal-border/30">
          <span className="text-[9px] text-terminal-accent font-medium">
            SPREAD: {(book.asks[book.asks.length - 1]?.price - book.bids[0]?.price || 0).toFixed(2)}
          </span>
        </div>
        {/* Bids */}
        {book.bids.map((b, i) => (
          <div key={`b${i}`} className="flex items-center px-2 py-[2px] relative">
            <div
              className="absolute right-0 top-0 bottom-0 bg-terminal-positive/8"
              style={{ width: `${(b.total / maxTotal) * 100}%` }}
            />
            <span className="w-[60px] text-terminal-positive relative z-10">{b.price.toFixed(2)}</span>
            <span className="w-[50px] text-right text-terminal-text-muted relative z-10">{b.size.toLocaleString()}</span>
            <span className="w-[60px] text-right text-terminal-text-muted relative z-10">{b.total.toLocaleString()}</span>
          </div>
        ))}
      </div>

      {/* Bottom stats */}
      <div className="px-2 py-1.5 border-t border-terminal-border grid grid-cols-4 gap-1 text-[8px]">
        <div>
          <div className="text-terminal-text-faint">Position</div>
          <div className="text-terminal-text-primary">50296</div>
        </div>
        <div>
          <div className="text-terminal-text-faint">Orders</div>
          <div className="text-terminal-accent">97203</div>
        </div>
        <div>
          <div className="text-terminal-text-faint">Seconds</div>
          <div className="text-terminal-text-primary">60057%</div>
        </div>
        <div>
          <div className="text-terminal-text-faint">Positions</div>
          <div className="text-terminal-negative">6.33%</div>
        </div>
      </div>

      {/* Mini volume bars */}
      <div className="px-2 pb-1.5 h-[40px]">
        <div className="flex items-end gap-[1px] h-full">
          {Array.from({ length: 30 }, (_, i) => {
            const h = Math.random() * 100;
            const isGreen = Math.random() > 0.4;
            return (
              <div
                key={i}
                className="flex-1 rounded-sm"
                style={{
                  height: `${h}%`,
                  backgroundColor: isGreen ? "rgba(63, 185, 80, 0.6)" : "rgba(248, 81, 73, 0.6)",
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
  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 relative">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={ORDER_DISTRIBUTION}
              cx="50%"
              cy="50%"
              innerRadius="55%"
              outerRadius="80%"
              paddingAngle={2}
              dataKey="value"
              stroke="none"
            >
              {ORDER_DISTRIBUTION.map((entry, i) => (
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
        {ORDER_DISTRIBUTION.map((d, i) => (
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
  const data = useMemo(generateLiquidityData, []);
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
  const data = useMemo(generateSpreadData, []);
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
  const data = useMemo(generateDepthData, []);
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
  return (
    <div className="flex flex-col h-full text-[10px] font-mono tabular-nums p-2 gap-2">
      <div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-terminal-text-primary">$45,750</span>
          <span className="text-terminal-text-muted">(1.22% VaR)</span>
        </div>
        <div className="flex items-center gap-3 mt-1">
          <span className="text-terminal-text-faint">Calmar Ratio</span>
          <span className="text-terminal-text-primary">155%</span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Max Drawdown</span>
          <span className="text-terminal-negative">8.78%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">DIAS</span>
          <span className="text-terminal-text-primary">$175,125</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Beta</span>
          <span className="text-terminal-text-primary">0.92</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">HoriFlows</span>
          <span className="text-terminal-accent">NASDAQ</span>
        </div>
      </div>

      <div className="border-t border-terminal-border pt-1.5">
        <div className="grid grid-cols-3 gap-x-3 gap-y-0.5 text-[10px]">
          <div className="flex justify-between">
            <span className="text-terminal-text-faint">Sector</span>
            <span>DAM</span>
          </div>
          <div className="flex justify-between">
            <span className="text-terminal-text-faint">DoI</span>
            <span>89.6%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-terminal-text-faint">Risk</span>
            <span>13.67%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-terminal-text-faint">Sens</span>
            <span>18,800</span>
          </div>
          <div className="flex justify-between">
            <span className="text-terminal-text-faint">Bet</span>
            <span>68.8%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-terminal-text-faint">Ret</span>
            <span>25.9%</span>
          </div>
        </div>
      </div>

      <div className="border-t border-terminal-border pt-1.5 flex-1">
        <div className="text-[9px] text-terminal-text-muted font-semibold tracking-wider uppercase mb-1">
          TOP RISKS <span className="text-terminal-text-faint ml-2">68%</span>
        </div>
        {TOP_RISKS.map((r, i) => (
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
  const data = useMemo(generatePnlData, []);
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
        {EXECUTIONS.map((e, i) => (
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
          {/* Left column: Order Book */}
          <div className="h-full p-[1px]">
            <DashboardPanel title="ORDER BOOK" className="h-full" noPadding>
              <OrderBook />
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
