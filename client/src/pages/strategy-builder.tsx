import { DashboardPanel } from "@/components/dashboard-panel";
import { useRef, useEffect, useCallback, useMemo } from "react";
import { useEngineQuery } from "@/hooks/use-engine-api";

const STRATEGIES = [
  { name: "Mean Reversion Alpha", status: "active", sharpe: 1.82, pnl: "+12.4%", trades: 1420 },
  { name: "Momentum Factor", status: "active", sharpe: 1.45, pnl: "+8.7%", trades: 890 },
  { name: "Stat Arb Pairs", status: "active", sharpe: 2.15, pnl: "+15.2%", trades: 3200 },
  { name: "Vol Surface", status: "paused", sharpe: 0.92, pnl: "+3.1%", trades: 450 },
  { name: "Macro Regime", status: "active", sharpe: 1.33, pnl: "+6.8%", trades: 120 },
  { name: "ML Ensemble v3", status: "testing", sharpe: 1.67, pnl: "+9.9%", trades: 2100 },
];

const PERF_CARDS = [
  { label: "Total Return", value: "+24.8%", color: "#00d4aa" },
  { label: "Max Drawdown", value: "-8.78%", color: "#f85149" },
  { label: "Win Rate", value: "62.3%", color: "#58a6ff" },
  { label: "Profit Factor", value: "1.94", color: "#bc8cff" },
];

// Node-based flow diagram
interface FlowNode {
  id: string; x: number; y: number; label: string; type: "input" | "process" | "output" | "decision";
}

const FLOW_NODES: FlowNode[] = [
  { id: "data", x: 80, y: 60, label: "Market Data", type: "input" },
  { id: "features", x: 250, y: 40, label: "Feature Eng", type: "process" },
  { id: "signals", x: 250, y: 110, label: "Signal Gen", type: "process" },
  { id: "filter", x: 420, y: 60, label: "Risk Filter", type: "decision" },
  { id: "size", x: 420, y: 140, label: "Position Size", type: "process" },
  { id: "exec", x: 580, y: 60, label: "Execution", type: "output" },
  { id: "monitor", x: 580, y: 140, label: "Monitor", type: "output" },
];

const FLOW_EDGES = [
  { from: "data", to: "features" }, { from: "data", to: "signals" },
  { from: "features", to: "filter" }, { from: "signals", to: "filter" },
  { from: "filter", to: "size" }, { from: "filter", to: "exec" },
  { from: "size", to: "exec" }, { from: "exec", to: "monitor" },
];

function FlowDiagram() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

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

    ctx.clearRect(0, 0, rect.width, rect.height);

    // Scale factor
    const sx = rect.width / 700;
    const sy = rect.height / 200;

    // Draw edges
    FLOW_EDGES.forEach((e) => {
      const from = FLOW_NODES.find((n) => n.id === e.from)!;
      const to = FLOW_NODES.find((n) => n.id === e.to)!;
      ctx.beginPath();
      ctx.moveTo(from.x * sx + 40, from.y * sy);
      ctx.lineTo(to.x * sx - 40, to.y * sy);
      ctx.strokeStyle = "rgba(0, 212, 170, 0.2)";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Arrow
      const angle = Math.atan2((to.y - from.y) * sy, (to.x - from.x) * sx);
      const ax = to.x * sx - 40;
      const ay = to.y * sy;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - 6 * Math.cos(angle - 0.4), ay - 6 * Math.sin(angle - 0.4));
      ctx.lineTo(ax - 6 * Math.cos(angle + 0.4), ay - 6 * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fillStyle = "rgba(0, 212, 170, 0.3)";
      ctx.fill();
    });

    // Draw nodes
    FLOW_NODES.forEach((n) => {
      const nx = n.x * sx;
      const ny = n.y * sy;
      const w = 75;
      const h = 28;

      const colors = {
        input: { bg: "rgba(0, 212, 170, 0.12)", border: "rgba(0, 212, 170, 0.4)" },
        process: { bg: "rgba(88, 166, 255, 0.12)", border: "rgba(88, 166, 255, 0.4)" },
        output: { bg: "rgba(63, 185, 80, 0.12)", border: "rgba(63, 185, 80, 0.4)" },
        decision: { bg: "rgba(188, 140, 255, 0.12)", border: "rgba(188, 140, 255, 0.4)" },
      };

      const c = colors[n.type];

      // Glow
      const grad = ctx.createRadialGradient(nx, ny, 0, nx, ny, 50);
      grad.addColorStop(0, c.bg);
      grad.addColorStop(1, "rgba(0,0,0,0)");
      ctx.beginPath();
      ctx.arc(nx, ny, 50, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      // Node rectangle
      ctx.beginPath();
      const r = 4;
      ctx.roundRect(nx - w / 2, ny - h / 2, w, h, r);
      ctx.fillStyle = c.bg;
      ctx.fill();
      ctx.strokeStyle = c.border;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Label
      ctx.font = "9px 'JetBrains Mono'";
      ctx.fillStyle = "rgba(230, 237, 243, 0.8)";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(n.label, nx, ny);
    });
  }, []);

  useEffect(() => {
    draw();
    const handleResize = () => draw();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [draw]);

  return (
    <div ref={containerRef} className="w-full h-full">
      <canvas ref={canvasRef} />
    </div>
  );
}

export default function StrategyBuilder() {
  // ─── Engine API — live strategy performance ─────────
  const { data: stratApi } = useEngineQuery<{
    strategies: Array<{ name: string; pnl: number; trades: number; sharpe: number; status: string }>;
    perf_cards: Array<{ name: string; value: string }>;
  }>("/ml/strategy/performance", { refetchInterval: 15000 });

  const strategies = stratApi?.strategies?.length
    ? stratApi.strategies.map((s) => ({
        name: s.name,
        status: s.status,
        sharpe: s.sharpe,
        pnl: s.pnl > 0 ? `+${s.pnl.toLocaleString()}` : `${s.pnl.toLocaleString()}`,
        trades: s.trades,
      }))
    : STRATEGIES;

  const perfCards = stratApi?.perf_cards?.length
    ? stratApi.perf_cards.map((c, i) => ({
        label: c.name,
        value: c.value,
        color: ["#00d4aa", "#f85149", "#58a6ff", "#bc8cff"][i % 4],
      }))
    : PERF_CARDS;

  return (
    <div className="h-full grid grid-cols-[260px_1fr] grid-rows-[auto_1fr] gap-[2px] p-[2px] overflow-auto" data-testid="strategy-builder">
      {/* Performance Summary */}
      <div className="col-span-2 flex gap-[2px]">
        {perfCards.map((p, i) => (
          <DashboardPanel key={i} title={p.label} className="flex-1">
            <div className="text-xl font-mono font-bold tabular-nums text-center" style={{ color: p.color }}>
              {p.value}
            </div>
          </DashboardPanel>
        ))}
        <DashboardPanel title="ACTIONS" className="w-[200px]">
          <div className="flex flex-col gap-1.5">
            <button className="w-full py-1.5 text-[10px] font-mono font-medium bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30 rounded hover:bg-terminal-accent/25 transition-colors">
              ▶ RUN BACKTEST
            </button>
            <button className="w-full py-1.5 text-[10px] font-mono font-medium bg-terminal-blue/15 text-terminal-blue border border-terminal-blue/30 rounded hover:bg-terminal-blue/25 transition-colors">
              + NEW STRATEGY
            </button>
          </div>
        </DashboardPanel>
      </div>

      {/* Strategy List Sidebar */}
      <DashboardPanel title="STRATEGIES" noPadding>
        <div className="overflow-auto h-full">
          {strategies.map((s, i) => (
            <div key={i} className="px-2 py-2 border-b border-terminal-border/20 hover:bg-white/[0.02] cursor-pointer">
              <div className="flex items-center gap-2">
                <span className={`w-1.5 h-1.5 rounded-full ${
                  s.status === "active" ? "bg-terminal-positive" :
                  s.status === "testing" ? "bg-terminal-warning" :
                  "bg-terminal-text-faint"
                }`} />
                <span className="text-[10px] text-terminal-text-primary font-medium">{s.name}</span>
              </div>
              <div className="flex items-center gap-3 mt-1 pl-3.5 text-[8px] font-mono tabular-nums">
                <span className="text-terminal-text-faint">SR: <span className="text-terminal-text-muted">{s.sharpe}</span></span>
                <span className="text-terminal-positive">{s.pnl}</span>
                <span className="text-terminal-text-faint">{s.trades} trades</span>
              </div>
            </div>
          ))}
        </div>
      </DashboardPanel>

      {/* Flow Diagram */}
      <DashboardPanel title="STRATEGY FLOW — Mean Reversion Alpha" noPadding>
        <FlowDiagram />
      </DashboardPanel>
    </div>
  );
}
