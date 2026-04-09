import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { LiveTicker } from "@/components/live-ticker";
import { ResizableDashboard } from "@/components/resizable-panel";
import {
  AreaChart, Area, XAxis, YAxis, ResponsiveContainer, LineChart, Line, Tooltip, PieChart, Pie, Cell, BarChart, Bar,
} from "recharts";
import { useEngineQuery, type PortfolioLive, type TradeRecord, type RiskPortfolio, type RiskGreeks } from "@/hooks/use-engine-api";

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

interface BlotterTrade {
  ticker: string;
  side: string;
  quantity: number;
  fill_price: number;
  arrival_price: number;
  slippage_bps: number;
  signal_type: string;
  product_type: string;
  routing_strategy: string;
  timestamp: string;
}

interface LiveTransactionsProps {
  onTickerSelect: (ticker: string) => void;
}

function LiveTransactions({ onTickerSelect }: LiveTransactionsProps) {
  const { data: blotterData } = useEngineQuery<{
    trades: Array<BlotterTrade>;
    source: string;
  }>("/execution/l7/blotter?limit=200", { refetchInterval: 3000 });

  const [flashIdx, setFlashIdx] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevCountRef = useRef(0);

  const trades = blotterData?.trades || [];

  // Flash new entries
  useEffect(() => {
    if (trades.length > prevCountRef.current && trades.length > 0) {
      setFlashIdx(0);
      setTimeout(() => setFlashIdx(null), 800);
    }
    prevCountRef.current = trades.length;
  }, [trades.length]);

  // Auto-scroll to top on new trade
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [trades.length]);

  const fmtNotional = (n: number) => n >= 1e6 ? `$${(n / 1e6).toFixed(1)}M` : `$${(n / 1e3).toFixed(0)}K`;
  const fmtTime = (ts: string) => {
    try {
      return new Date(ts).toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch {
      return "--:--:--";
    }
  };

  const totalNotional = trades.reduce((s, t) => s + t.quantity * t.fill_price, 0);
  const buys = trades.filter((t) => t.side?.toUpperCase() === "BUY" || t.side?.toUpperCase() === "COVER").length;
  const sells = trades.filter((t) => t.side?.toUpperCase() === "SELL" || t.side?.toUpperCase() === "SHORT").length;

  return (
    <div className="flex flex-col h-full text-[10px] font-mono tabular-nums">
      {/* Summary header */}
      <div className="px-2 py-1.5 border-b border-terminal-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-terminal-positive animate-pulse" />
            <span className="text-[9px] text-terminal-text-faint uppercase tracking-wider">Live Feed</span>
          </div>
          <span className="text-[9px] text-terminal-text-faint">{trades.length} txns</span>
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
          <div className="text-terminal-text-primary">{trades.length}</div>
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
          <div className="text-terminal-text-faint">Avg Slip</div>
          <div className="text-terminal-text-primary">
            {trades.length ? (trades.reduce((s, t) => s + (t.slippage_bps || 0), 0) / trades.length).toFixed(1) : "0"}bps
          </div>
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
        {trades.length === 0 && (
          <div style={{color: "var(--muted)", fontSize: 12, padding: 20, textAlign: "center"}}>
            Waiting for data...
          </div>
        )}
        {trades.map((tx, i) => {
          const isBuy = tx.side?.toUpperCase() === "BUY" || tx.side?.toUpperCase() === "COVER";
          return (
            <div
              key={i}
              onClick={() => onTickerSelect(tx.ticker)}
              className={`flex items-center px-2 py-[3px] border-b border-terminal-border/10 transition-colors duration-700 cursor-pointer ${
                flashIdx === i ? "bg-terminal-accent/10" : "hover:bg-white/[0.02]"
              }`}
            >
              <span className="w-[52px] text-terminal-text-faint text-[9px]">{fmtTime(tx.timestamp).slice(0, 8)}</span>
              <span className="w-[38px] text-terminal-accent font-semibold text-[9px]">{tx.ticker}</span>
              <span className={`w-[30px] font-semibold text-[8px] ${
                isBuy ? "text-terminal-positive" : "text-terminal-negative"
              }`}>
                {tx.side?.toUpperCase()}
              </span>
              <span className="w-[32px] text-right text-terminal-text-muted text-[9px]">{tx.quantity}</span>
              <span className="flex-1 text-right text-terminal-text-primary text-[9px]">${tx.fill_price.toFixed(2)}</span>
            </div>
          );
        })}
      </div>

      {/* Bottom detail strip */}
      <div className="px-2 py-1.5 border-t border-terminal-border">
        <div className="grid grid-cols-3 gap-1 text-[8px]">
          <div>
            <div className="text-terminal-text-faint">Routing</div>
            <div className="text-terminal-text-primary truncate">
              {Array.from(new Set(trades.slice(0, 20).map(t => t.routing_strategy).filter(Boolean))).slice(0, 3).join(", ") || "--"}
            </div>
          </div>
          <div>
            <div className="text-terminal-text-faint">Signals</div>
            <div className="text-terminal-accent truncate">
              {Array.from(new Set(trades.slice(0, 20).map(t => t.signal_type).filter(Boolean))).slice(0, 2).join(", ") || "--"}
            </div>
          </div>
          <div>
            <div className="text-terminal-text-faint">Fill Rate</div>
            <div className="text-terminal-positive">100%</div>
          </div>
        </div>
      </div>

      {/* Mini activity spark */}
      <div className="px-2 pb-1.5 h-[32px]">
        <div className="flex items-end gap-[1px] h-full">
          {trades.slice(0, 40).map((tx, i) => {
            const maxQ = Math.max(...trades.slice(0, 40).map(t => t.quantity), 1);
            const h = (tx.quantity / maxQ) * 100;
            const isBuy = tx.side?.toUpperCase() === "BUY" || tx.side?.toUpperCase() === "COVER";
            return (
              <div
                key={i}
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
  const chartData = distData?.distribution || [];

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 relative">
        {chartData.length === 0 ? (
          <div style={{color: "var(--muted)", fontSize: 12, padding: 20, textAlign: "center"}}>
            Waiting for data...
          </div>
        ) : (
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
        )}
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

// ═══════════ MARKET DATA CHARTS (Liquidity, Spread, Depth) ═══════════

interface MarketDataProps {
  selectedTicker: string;
}

function LiquidityChart({ selectedTicker }: MarketDataProps) {
  const { data: marketData } = useEngineQuery<{
    quote: { price: number; bid: number; ask: number; spread: number; volume: number };
    ohlcv: Array<{ date: string; open: number; high: number; low: number; close: number; volume: number }>;
    depth: Array<{ price: string; bidDepth: number; askDepth: number }>;
    spread_history: Array<{ time: string; spread: number }>;
    source: string;
    degraded?: boolean;
  }>(`/execution/market-data?ticker=${selectedTicker}`, { refetchInterval: 10000 });

  // Derive liquidity from ohlcv volume data
  const data = useMemo(() => {
    if (!marketData?.ohlcv?.length) return [];
    return marketData.ohlcv.map((bar) => ({
      time: bar.date,
      bid: bar.volume * bar.close * 0.5,
      ask: bar.volume * bar.close * 0.5,
    }));
  }, [marketData]);

  return (
    <div className="h-full">
      {data.length === 0 ? (
        <div style={{color: "var(--muted)", fontSize: 12, padding: 20, textAlign: "center"}}>
          Waiting for data...
        </div>
      ) : (
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
      )}
    </div>
  );
}

function SpreadChart({ selectedTicker }: MarketDataProps) {
  const { data: marketData } = useEngineQuery<{
    quote: { price: number; bid: number; ask: number; spread: number; volume: number };
    ohlcv: Array<{ date: string; open: number; high: number; low: number; close: number; volume: number }>;
    depth: Array<{ price: string; bidDepth: number; askDepth: number }>;
    spread_history: Array<{ time: string; spread: number }>;
    source: string;
    degraded?: boolean;
  }>(`/execution/market-data?ticker=${selectedTicker}`, { refetchInterval: 10000 });

  const data = marketData?.spread_history || [];

  return (
    <div className="h-full">
      {data.length === 0 ? (
        <div style={{color: "var(--muted)", fontSize: 12, padding: 20, textAlign: "center"}}>
          Waiting for data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
            <Line type="monotone" dataKey="spread" stroke="#58a6ff" strokeWidth={1} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

function DepthChart({ selectedTicker }: MarketDataProps) {
  const { data: marketData } = useEngineQuery<{
    quote: { price: number; bid: number; ask: number; spread: number; volume: number };
    ohlcv: Array<{ date: string; open: number; high: number; low: number; close: number; volume: number }>;
    depth: Array<{ price: string; bidDepth: number; askDepth: number }>;
    spread_history: Array<{ time: string; spread: number }>;
    source: string;
    degraded?: boolean;
  }>(`/execution/market-data?ticker=${selectedTicker}`, { refetchInterval: 10000 });

  const data = marketData?.depth || [];

  return (
    <div className="h-full">
      {data.length === 0 ? (
        <div style={{color: "var(--muted)", fontSize: 12, padding: 20, textAlign: "center"}}>
          Waiting for data...
        </div>
      ) : (
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
      )}
    </div>
  );
}

// ═══════════ RISK PANEL ═══════════

function RiskPanel() {
  const { data: riskData } = useEngineQuery<RiskPortfolio>("/risk/portfolio", { refetchInterval: 5000 });
  const { data: portData } = useEngineQuery<PortfolioLive>("/portfolio/live", { refetchInterval: 5000 });
  const { data: alertsData } = useEngineQuery<{ alerts: Array<{ name: string; value: string; severity: string }> }>("/risk/alerts", { refetchInterval: 10000 });
  const riskAlerts = alertsData?.alerts || [];

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
        {riskAlerts.length === 0 && (
          <div style={{color: "var(--muted)", fontSize: 12, padding: "8px 0", textAlign: "center"}}>
            Waiting for data...
          </div>
        )}
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
  const data = pnlApi?.series || [];

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1">
        {data.length === 0 ? (
          <div style={{color: "var(--muted)", fontSize: 12, padding: 20, textAlign: "center"}}>
            Waiting for data...
          </div>
        ) : (
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
        )}
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
  const { data: tobData } = useEngineQuery<{
    positions: Array<{
      ticker: string;
      last_price: number;
      bid: number;
      ask: number;
      spread: number;
      spread_bps: number;
      position_qty: number;
      position_value: number;
      unrealized_pnl: number;
    }>;
  }>("/execution/top-of-book", { refetchInterval: 10000 });

  const positions = tobData?.positions || [];

  return (
    <div className="flex flex-col h-full text-[10px] font-mono tabular-nums">
      {/* Header */}
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
        <span className="w-[70px]">Pair</span>
        <span className="w-[55px] text-right">Bid</span>
        <span className="w-[45px] text-right">Ask</span>
        <span className="flex-1 text-right">Unr. PnL</span>
      </div>
      <div className="flex-1 overflow-auto">
        {positions.length === 0 && (
          <div style={{color: "var(--muted)", fontSize: 12, padding: 20, textAlign: "center"}}>
            Waiting for data...
          </div>
        )}
        {positions.slice(0, 6).map((p, i) => (
          <div
            key={i}
            className="flex items-center px-2 py-1.5 border-b border-terminal-border/20 hover:bg-white/[0.02]"
          >
            <span className="w-[70px] text-terminal-text-primary font-medium">{p.ticker}</span>
            <span className="w-[55px] text-right text-terminal-text-muted">{p.bid?.toFixed(2) ?? "--"}</span>
            <span className="w-[45px] text-right text-terminal-text-muted">{p.ask?.toFixed(2) ?? "--"}</span>
            <span className={`flex-1 text-right font-medium ${p.unrealized_pnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
              {p.unrealized_pnl >= 0 ? "+" : ""}{p.unrealized_pnl?.toFixed(2) ?? "--"}
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
  const [selectedTicker, setSelectedTicker] = useState<string>("SPY");

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
              <LiveTransactions onTickerSelect={setSelectedTicker} />
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
              headerRight={<span className="text-[8px] text-terminal-accent tabular-nums font-mono">{selectedTicker}</span>}
              noPadding
            >
              <LiquidityChart selectedTicker={selectedTicker} />
            </DashboardPanel>
            <DashboardPanel
              title="SPREAD"
              className="flex-1"
              headerRight={<span className="text-[8px] text-terminal-blue font-mono">{selectedTicker}</span>}
              noPadding
            >
              <SpreadChart selectedTicker={selectedTicker} />
            </DashboardPanel>
            <DashboardPanel title="DEPTH" className="flex-1" noPadding>
              <DepthChart selectedTicker={selectedTicker} />
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
