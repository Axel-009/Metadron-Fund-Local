import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import {
  AreaChart, Area, XAxis, YAxis, ResponsiveContainer, LineChart, Line,
  Tooltip, PieChart, Pie, Cell, BarChart, Bar,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";

// ═══════════════════════════════════════════════════════════════════════════════
// TYPE DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════════

type Regime = "TRENDING" | "RANGE" | "STRESS" | "CRASH";

interface CoreTensor {
  L: number; // Liquidity [-1, +1]
  R: number; // Risk [0, 1]
  F: number; // Flow [-1, +1]
  Ct: number; // Combined score
}

interface FedPlumbingLayer {
  sofr: number;
  tga: number;
  onRrp: number;
  reserveAdequacy: number;
  fedFundsImpact: number;
  netPlumbing: number;
}

interface LiquidityTensor {
  sofrWeight: number;       // 20%
  creditWeight: number;     // 25%
  m2vWeight: number;        // 15%
  hySpreadWeight: number;   // 15%
  fedPlumbingWeight: number;// 25%
  L: number;
}

interface ReserveFlowKernel {
  deltaReserves: number;
  equityImpact: number;
  creditImpact: number;
  decay: number;
}

interface RiskStateModel {
  vix: number;          // 30%
  realizedVol: number;  // 20%
  credit: number;       // 20%
  correlation: number;  // 15%
  tailRisk: number;     // 15%
  R: number;
}

interface CapitalFlowModel {
  sectorMomentum: number;
  leadersLaggards: number;
  rotationVelocity: number;
  breadth: number;
  persistence: number;
  F: number;
}

interface RegimeState {
  current: Regime;
  confidence: number;
  transitions: {
    TRENDING: number;
    RANGE: number;
    STRESS: number;
    CRASH: number;
  };
  history: { regime: Regime; ts: number }[];
}

interface GateScore {
  id: string;
  ticker: string;
  g1: number; g2: number; g3: number; g4: number;
  weighted: number;
  pass: boolean;
}

interface KillSwitch {
  hyOas: number;    // threshold +35bp
  vixTerm: number;  // flat/inverted
  breadth: number;  // <50%
  active: boolean;
}

interface RiskGovernor {
  positionPct: number;
  sectorPct: number;
  leverage: number;
  varPct: number;
  drawdownPct: number;
  crashFloor: number;
  beta: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOCK DATA GENERATORS
// ═══════════════════════════════════════════════════════════════════════════════

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}

function jitter(v: number, scale: number) {
  return v + (Math.random() - 0.5) * scale;
}

function generateCoreTensor(): CoreTensor {
  const L = clamp(jitter(0.42, 0.04), -1, 1);
  const R = clamp(jitter(0.28, 0.03), 0, 1);
  const F = clamp(jitter(0.61, 0.04), -1, 1);
  return { L, R, F, Ct: L * 0.4 + (1 - R) * 0.35 + F * 0.25 };
}

function generateFedPlumbing(): FedPlumbingLayer {
  return {
    sofr: clamp(jitter(0.38, 0.02), -1, 1),
    tga: clamp(jitter(-0.12, 0.02), -1, 1),
    onRrp: clamp(jitter(0.22, 0.02), -1, 1),
    reserveAdequacy: clamp(jitter(0.55, 0.02), -1, 1),
    fedFundsImpact: clamp(jitter(-0.08, 0.02), -1, 1),
    netPlumbing: clamp(jitter(0.31, 0.02), -1, 1),
  };
}

function generateLiquidityTensor(): LiquidityTensor {
  return {
    sofrWeight: clamp(jitter(0.44, 0.02), -1, 1),
    creditWeight: clamp(jitter(0.38, 0.02), -1, 1),
    m2vWeight: clamp(jitter(0.29, 0.02), -1, 1),
    hySpreadWeight: clamp(jitter(-0.15, 0.02), -1, 1),
    fedPlumbingWeight: clamp(jitter(0.31, 0.02), -1, 1),
    L: clamp(jitter(0.42, 0.02), -1, 1),
  };
}

function generateReserveKernel(): ReserveFlowKernel {
  return {
    deltaReserves: jitter(28.4, 2),
    equityImpact: clamp(jitter(0.31, 0.02), -1, 1),
    creditImpact: clamp(jitter(-0.09, 0.02), -1, 1),
    decay: clamp(jitter(0.87, 0.01), 0, 1),
  };
}

function generateRiskModel(): RiskStateModel {
  return {
    vix: clamp(jitter(0.26, 0.02), 0, 1),
    realizedVol: clamp(jitter(0.19, 0.02), 0, 1),
    credit: clamp(jitter(0.31, 0.02), 0, 1),
    correlation: clamp(jitter(0.38, 0.02), 0, 1),
    tailRisk: clamp(jitter(0.21, 0.02), 0, 1),
    R: clamp(jitter(0.28, 0.02), 0, 1),
  };
}

function generateCapitalFlow(): CapitalFlowModel {
  return {
    sectorMomentum: clamp(jitter(0.67, 0.03), -1, 1),
    leadersLaggards: clamp(jitter(0.55, 0.03), -1, 1),
    rotationVelocity: clamp(jitter(0.41, 0.03), -1, 1),
    breadth: clamp(jitter(0.72, 0.02), 0, 1),
    persistence: clamp(jitter(0.58, 0.02), -1, 1),
    F: clamp(jitter(0.61, 0.02), -1, 1),
  };
}

const REGIME_HISTORY_INITIAL: { regime: Regime; ts: number }[] = [
  { regime: "TRENDING", ts: Date.now() - 3600000 * 2 },
  { regime: "RANGE", ts: Date.now() - 3600000 * 1.5 },
  { regime: "TRENDING", ts: Date.now() - 3600000 * 0.5 },
  { regime: "TRENDING", ts: Date.now() },
];

function generateRegimeState(): RegimeState {
  return {
    current: "TRENDING",
    confidence: clamp(jitter(0.84, 0.02), 0, 1),
    transitions: {
      TRENDING: clamp(jitter(0.72, 0.02), 0, 1),
      RANGE: clamp(jitter(0.18, 0.02), 0, 1),
      STRESS: clamp(jitter(0.07, 0.01), 0, 1),
      CRASH: clamp(jitter(0.03, 0.005), 0, 1),
    },
    history: REGIME_HISTORY_INITIAL,
  };
}

const GATE_TICKERS = ["NVDA", "MSFT", "AAPL", "META", "AMZN", "GOOGL", "JPM", "GS"];

function generateGateScores(): GateScore[] {
  return GATE_TICKERS.map((ticker) => {
    const g1 = clamp(jitter(0.68, 0.08), 0, 1);
    const g2 = clamp(jitter(0.71, 0.08), 0, 1);
    const g3 = clamp(jitter(0.65, 0.08), 0, 1);
    const g4 = clamp(jitter(0.74, 0.08), 0, 1);
    const weighted = g1 * 0.20 + g2 * 0.25 + g3 * 0.30 + g4 * 0.25;
    return {
      id: ticker,
      ticker,
      g1, g2, g3, g4, weighted,
      pass: weighted >= 0.50 && g1 >= 0.30 && g2 >= 0.30 && g3 >= 0.30 && g4 >= 0.30,
    };
  });
}

function generateKillSwitch(): KillSwitch {
  const hyOas = clamp(jitter(24, 3), 0, 100);      // needs +35 to fire
  const vixTerm = clamp(jitter(0.08, 0.03), -0.5, 0.5); // positive = normal, flat/neg fires
  const breadth = clamp(jitter(0.72, 0.04), 0, 1); // <0.5 fires
  return {
    hyOas,
    vixTerm,
    breadth,
    active: hyOas >= 35 && vixTerm <= 0 && breadth < 0.5,
  };
}

function generateRiskGovernor(): RiskGovernor {
  return {
    positionPct: clamp(jitter(3.2, 0.2), 0, 10),
    sectorPct: clamp(jitter(18.4, 0.5), 0, 40),
    leverage: clamp(jitter(2.8, 0.05), 0, 5),
    varPct: clamp(jitter(1.12, 0.05), 0, 3),
    drawdownPct: clamp(jitter(7.8, 0.3), 0, 20),
    crashFloor: clamp(jitter(28.4, 0.5), 0, 50),
    beta: clamp(jitter(0.61, 0.02), -0.5, 1.5),
  };
}

// FCLP step state
const FCLP_STEPS = [
  { id: 1, label: "Ingest Plumbing", short: "INGEST" },
  { id: 2, label: "Recompute Tensor/Kernel", short: "TENSOR" },
  { id: 3, label: "Regime Detect", short: "REGIME" },
  { id: 4, label: "Gate Scoring", short: "GATES" },
  { id: 5, label: "Risk Pass", short: "RISK" },
  { id: 6, label: "Write Allocations", short: "ALLOC" },
];

const STRESS_SCENARIOS = [
  { name: "2008 GFC", regime: "CRASH", beta: -0.18, lev: 0.9, eq: 8, hedge: 70, risk: 9.2 },
  { name: "2020 COVID", regime: "CRASH", beta: -0.22, lev: 0.7, eq: 6, hedge: 75, risk: 9.6 },
  { name: "2022 Rate Hike", regime: "STRESS", beta: 0.12, lev: 1.4, eq: 20, hedge: 45, risk: 7.8 },
  { name: "Bull Run", regime: "TRENDING", beta: 0.63, lev: 3.0, eq: 48, hedge: 12, risk: 3.2 },
  { name: "Range Bound", regime: "RANGE", beta: 0.44, lev: 2.4, eq: 36, hedge: 22, risk: 5.1 },
  { name: "VIX Spike", regime: "STRESS", beta: 0.08, lev: 1.3, eq: 18, hedge: 50, risk: 8.4 },
];

const REGIME_ALLOC: Record<Regime, { P1: number; P2: number; P3: number; P4: number; P5: number; lev: number; beta: string }> = {
  TRENDING: { P1: 25, P2: 25, P3: 30, P4: 10, P5: 10, lev: 3.0, beta: "≤0.65" },
  RANGE:    { P1: 20, P2: 20, P3: 20, P4: 25, P5: 15, lev: 2.5, beta: "≤0.45" },
  STRESS:   { P1: 15, P2: 10, P3: 10, P4: 25, P5: 40, lev: 1.5, beta: "≤0.15" },
  CRASH:    { P1: 5,  P2: 5,  P3: 5,  P4: 20, P5: 65, lev: 0.8, beta: "≤-0.20" },
};

const SLEEVE_COLORS = ["#00d4aa", "#58a6ff", "#bc8cff", "#d29922", "#f85149"];
const SLEEVE_LABELS = ["P1 Dir.Eq", "P2 Factor", "P3 Macro", "P4 Options", "P5 Hedges"];

function generateRegimeHistory() {
  const history: { t: string; regime: Regime }[] = [];
  const regimes: Regime[] = ["TRENDING", "RANGE", "TRENDING", "STRESS", "TRENDING", "TRENDING"];
  const now = Date.now();
  for (let i = 0; i < 6; i++) {
    const ts = new Date(now - (5 - i) * 1200000);
    history.push({
      t: ts.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit" }),
      regime: regimes[i],
    });
  }
  return history;
}

function generateBetaTrend() {
  const d = [];
  let v = 0.55;
  for (let i = 0; i < 30; i++) {
    v = clamp(v + (Math.random() - 0.48) * 0.04, 0.3, 0.85);
    d.push({ t: i, beta: +v.toFixed(3) });
  }
  return d;
}

const REGIME_DIST = [
  { name: "TRENDING", value: 52, color: "#00d4aa" },
  { name: "RANGE", value: 28, color: "#58a6ff" },
  { name: "STRESS", value: 15, color: "#d29922" },
  { name: "CRASH", value: 5, color: "#f85149" },
];

// ═══════════════════════════════════════════════════════════════════════════════
// CUBE CANVAS — 3D animated cube with L/R/F axes
// ═══════════════════════════════════════════════════════════════════════════════

interface CubeCanvasProps {
  tensor: CoreTensor;
}

function CubeCanvas({ tensor }: CubeCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number>(0);
  const angleRef = useRef(0);
  const particlesRef = useRef<{ x: number; y: number; z: number; life: number; maxLife: number }[]>([]);

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
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2;
    const cy = H / 2;
    const size = Math.min(W, H) * 0.32;
    angleRef.current += 0.003;
    const angle = angleRef.current;
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    const cosB = Math.cos(0.45);
    const sinB = Math.sin(0.45);

    // 3D→2D isometric-ish projection
    function project(x: number, y: number, z: number): [number, number] {
      // Rotate around Y axis
      const x1 = x * cosA - z * sinA;
      const z1 = x * sinA + z * cosA;
      // Rotate around X axis slightly
      const y1 = y * cosB - z1 * sinB;
      const z2 = y * sinB + z1 * cosB;
      return [cx + x1 * size, cy - y1 * size + z2 * 0.1];
    }

    // Cube vertices [-1,+1]^3
    const verts: [number, number, number][] = [
      [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
      [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1],
    ];

    const edges = [
      [0,1],[1,2],[2,3],[3,0], // back face
      [4,5],[5,6],[6,7],[7,4], // front face
      [0,4],[1,5],[2,6],[3,7], // connectors
    ];

    // Draw background grid glow
    const gridGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, size * 1.6);
    gridGrad.addColorStop(0, "rgba(0,212,170,0.04)");
    gridGrad.addColorStop(1, "rgba(0,212,170,0)");
    ctx.beginPath();
    ctx.arc(cx, cy, size * 1.6, 0, Math.PI * 2);
    ctx.fillStyle = gridGrad;
    ctx.fill();

    // Draw cube edges
    edges.forEach(([a, b]) => {
      const [ax, ay] = project(...verts[a]);
      const [bx, by] = project(...verts[b]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.strokeStyle = "rgba(0,212,170,0.18)";
      ctx.lineWidth = 0.8;
      ctx.stroke();
    });

    // Draw face planes (subtle fill)
    const faces = [
      [0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5],
    ];
    faces.forEach((f) => {
      const pts = f.map(i => project(...verts[i]));
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      pts.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
      ctx.closePath();
      ctx.fillStyle = "rgba(0,212,170,0.015)";
      ctx.fill();
    });

    // Draw axes with labels
    const axisLen = 1.45;
    const axes: { dir: [number, number, number]; label: string; val: number; color: string }[] = [
      { dir: [axisLen, 0, 0], label: "L(t)", val: tensor.L, color: "#00d4aa" },
      { dir: [0, axisLen, 0], label: "R(t)", val: tensor.R, color: "#f85149" },
      { dir: [0, 0, axisLen], label: "F(t)", val: tensor.F, color: "#58a6ff" },
    ];

    axes.forEach(({ dir, label, val, color }) => {
      const [ox, oy] = project(0, 0, 0);
      const [ex, ey] = project(dir[0], dir[1], dir[2]);
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ex, ey);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.7;
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Arrowhead
      const dx = ex - ox, dy = ey - oy;
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len > 0) {
        const nx = dx / len, ny = dy / len;
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - nx * 6 - ny * 3, ey - ny * 6 + nx * 3);
        ctx.lineTo(ex - nx * 6 + ny * 3, ey - ny * 6 - nx * 3);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
      }

      // Label
      ctx.font = "bold 9px 'JetBrains Mono', monospace";
      ctx.fillStyle = color;
      ctx.textAlign = "center";
      const lx = ex + (ex - ox) * 0.12;
      const ly = ey + (ey - oy) * 0.12;
      ctx.fillText(`${label}=${val >= 0 ? "+" : ""}${val.toFixed(2)}`, lx, ly);
    });

    // Operating point — map L/R/F to cube coords
    const opX = tensor.L;    // [-1,+1]
    const opY = -(tensor.R * 2 - 1); // [0,1] → [-1,+1], inverted for visual
    const opZ = tensor.F;    // [-1,+1]
    const [opPx, opPy] = project(opX, opY, opZ);

    // Glow ring for operating point
    const t = Date.now() * 0.002;
    const pulse = 0.7 + 0.3 * Math.sin(t);

    const glowGrad = ctx.createRadialGradient(opPx, opPy, 0, opPx, opPy, 18 * pulse);
    glowGrad.addColorStop(0, "rgba(0,212,170,0.5)");
    glowGrad.addColorStop(0.4, "rgba(0,212,170,0.15)");
    glowGrad.addColorStop(1, "rgba(0,212,170,0)");
    ctx.beginPath();
    ctx.arc(opPx, opPy, 18 * pulse, 0, Math.PI * 2);
    ctx.fillStyle = glowGrad;
    ctx.fill();

    // Operating point dot
    ctx.beginPath();
    ctx.arc(opPx, opPy, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = "#00d4aa";
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Projection lines from operating point to each face
    const [ox0, oy0] = project(0, 0, 0);
    ctx.beginPath();
    ctx.moveTo(opPx, opPy);
    ctx.lineTo(project(opX, opY, -1)[0], project(opX, opY, -1)[1]);
    ctx.strokeStyle = "rgba(0,212,170,0.12)";
    ctx.lineWidth = 0.8;
    ctx.setLineDash([2, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Particles emanating from operating point
    if (Math.random() < 0.15) {
      particlesRef.current.push({
        x: opX + (Math.random() - 0.5) * 0.3,
        y: opY + (Math.random() - 0.5) * 0.3,
        z: opZ + (Math.random() - 0.5) * 0.3,
        life: 0,
        maxLife: 40 + Math.random() * 40,
      });
    }
    particlesRef.current = particlesRef.current.filter(p => p.life < p.maxLife);
    particlesRef.current.forEach((p) => {
      p.life++;
      p.x += (Math.random() - 0.5) * 0.02;
      p.y += (Math.random() - 0.5) * 0.02;
      p.z += (Math.random() - 0.5) * 0.02;
      const alpha = (1 - p.life / p.maxLife) * 0.6;
      const [px, py] = project(p.x, p.y, p.z);
      ctx.beginPath();
      ctx.arc(px, py, 1.2, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,212,170,${alpha})`;
      ctx.fill();
    });

    // Center origin dot
    ctx.beginPath();
    ctx.arc(ox0, oy0, 2, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0,212,170,0.3)";
    ctx.fill();

    // C(t) label in corner
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(0,212,170,0.8)";
    ctx.textAlign = "left";
    ctx.fillText(`C(t) = ${tensor.Ct >= 0 ? "+" : ""}${tensor.Ct.toFixed(3)}`, 8, 16);

    animRef.current = requestAnimationFrame(draw);
  }, [tensor]);

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

// ═══════════════════════════════════════════════════════════════════════════════
// REGIME PANEL
// ═══════════════════════════════════════════════════════════════════════════════

const REGIME_COLORS: Record<Regime, string> = {
  TRENDING: "#00d4aa",
  RANGE: "#58a6ff",
  STRESS: "#d29922",
  CRASH: "#f85149",
};

const REGIME_BG: Record<Regime, string> = {
  TRENDING: "rgba(0,212,170,0.08)",
  RANGE: "rgba(88,166,255,0.08)",
  STRESS: "rgba(210,153,34,0.08)",
  CRASH: "rgba(248,81,73,0.08)",
};

function RegimePanel({ regime, history }: { regime: RegimeState; history: { t: string; regime: Regime }[] }) {
  const alloc = REGIME_ALLOC[regime.current];
  const sleeveData = [
    { name: "P1", value: alloc.P1, color: SLEEVE_COLORS[0] },
    { name: "P2", value: alloc.P2, color: SLEEVE_COLORS[1] },
    { name: "P3", value: alloc.P3, color: SLEEVE_COLORS[2] },
    { name: "P4", value: alloc.P4, color: SLEEVE_COLORS[3] },
    { name: "P5", value: alloc.P5, color: SLEEVE_COLORS[4] },
  ];

  return (
    <div className="flex flex-col h-full text-[10px] font-mono tabular-nums p-2 gap-1.5">
      {/* Current regime badge */}
      <div
        className="flex items-center justify-between px-2 py-1.5 rounded border"
        style={{ borderColor: REGIME_COLORS[regime.current] + "40", background: REGIME_BG[regime.current] }}
      >
        <div className="flex items-center gap-2">
          <span
            className="w-2 h-2 rounded-full animate-pulse"
            style={{ backgroundColor: REGIME_COLORS[regime.current] }}
          />
          <span
            className="text-[13px] font-bold tracking-widest"
            style={{ color: REGIME_COLORS[regime.current] }}
          >
            {regime.current}
          </span>
        </div>
        <div className="text-right">
          <div className="text-[9px] text-terminal-text-faint">CONFIDENCE</div>
          <div className="text-[12px] font-bold" style={{ color: REGIME_COLORS[regime.current] }}>
            {(regime.confidence * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Regime metrics */}
      <div className="grid grid-cols-3 gap-1 text-[8px]">
        <div className="bg-terminal-surface-2 rounded px-1.5 py-1">
          <div className="text-terminal-text-faint">LEVERAGE</div>
          <div className="text-terminal-text-primary font-bold">{alloc.lev}x</div>
        </div>
        <div className="bg-terminal-surface-2 rounded px-1.5 py-1">
          <div className="text-terminal-text-faint">BETA CAP</div>
          <div className="text-terminal-accent font-bold">{alloc.beta}</div>
        </div>
        <div className="bg-terminal-surface-2 rounded px-1.5 py-1">
          <div className="text-terminal-text-faint">MARKOV</div>
          <div className="text-terminal-text-primary font-bold">
            {(regime.transitions[regime.current] * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Transition probabilities heatmap */}
      <div>
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1">Transition Probabilities</div>
        <div className="grid grid-cols-4 gap-[2px]">
          {(["TRENDING", "RANGE", "STRESS", "CRASH"] as Regime[]).map((r) => {
            const prob = regime.transitions[r];
            const isActive = r === regime.current;
            return (
              <div
                key={r}
                className="flex flex-col items-center py-1 px-0.5 rounded"
                style={{
                  background: isActive
                    ? REGIME_COLORS[r] + "20"
                    : `rgba(${r === "TRENDING" ? "0,212,170" : r === "RANGE" ? "88,166,255" : r === "STRESS" ? "210,153,34" : "248,81,73"},${prob * 0.15})`,
                  border: isActive ? `1px solid ${REGIME_COLORS[r]}50` : "1px solid transparent",
                }}
              >
                <span className="text-[7px] text-terminal-text-faint">{r.slice(0, 4)}</span>
                <span className="text-[9px] font-bold" style={{ color: REGIME_COLORS[r] }}>
                  {(prob * 100).toFixed(0)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Regime history timeline strip */}
      <div>
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1">Regime History</div>
        <div className="flex gap-[2px] h-5">
          {history.map((h, i) => (
            <div
              key={i}
              className="flex-1 rounded-sm flex items-center justify-center relative group"
              style={{ backgroundColor: REGIME_COLORS[h.regime] + "30", border: `1px solid ${REGIME_COLORS[h.regime]}40` }}
            >
              <span className="text-[6px]" style={{ color: REGIME_COLORS[h.regime] }}>
                {h.regime.slice(0, 1)}
              </span>
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block bg-terminal-surface border border-terminal-border rounded px-1 py-0.5 text-[7px] text-terminal-text-muted whitespace-nowrap z-10">
                {h.t} {h.regime}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Sleeve mini bar chart */}
      <div>
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1">Gate-Z Sleeve Alloc</div>
        <div className="flex h-3 rounded overflow-hidden gap-[1px]">
          {sleeveData.map((s) => (
            <div
              key={s.name}
              className="flex items-center justify-center"
              style={{ width: `${s.value}%`, backgroundColor: s.color + "70" }}
            >
              {s.value >= 10 && (
                <span className="text-[6px] font-bold" style={{ color: s.color }}>{s.value}%</span>
              )}
            </div>
          ))}
        </div>
        <div className="flex justify-between mt-0.5">
          {sleeveData.map((s) => (
            <span key={s.name} className="text-[6px]" style={{ color: s.color }}>{s.name}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// LAYER PIPELINE
// ═══════════════════════════════════════════════════════════════════════════════

function SignalBar({ value, range = [-1, 1] }: { value: number; range?: [number, number] }) {
  const [min, max] = range;
  const pct = ((value - min) / (max - min)) * 100;
  const zeroAt = ((-min) / (max - min)) * 100;
  const isPos = value >= 0;

  return (
    <div className="relative h-2 bg-terminal-surface-2 rounded-sm overflow-hidden w-full">
      {/* zero line */}
      <div
        className="absolute top-0 bottom-0 w-px bg-terminal-border"
        style={{ left: `${zeroAt}%` }}
      />
      {/* value bar */}
      <div
        className="absolute top-0 bottom-0 rounded-sm"
        style={{
          left: isPos ? `${zeroAt}%` : `${pct}%`,
          width: isPos ? `${pct - zeroAt}%` : `${zeroAt - pct}%`,
          backgroundColor: isPos ? "#00d4aa" : "#f85149",
          opacity: 0.75,
        }}
      />
    </div>
  );
}

interface LayerPipelineProps {
  fed: FedPlumbingLayer;
  liq: LiquidityTensor;
  kernel: ReserveFlowKernel;
  risk: RiskStateModel;
  flow: CapitalFlowModel;
  regime: RegimeState;
}

function LayerPipeline({ fed, liq, kernel, risk, flow, regime }: LayerPipelineProps) {
  const layers = [
    {
      id: 0,
      name: "FedPlumbing",
      color: "#58a6ff",
      output: fed.netPlumbing,
      outputLabel: "NET_PLUMB",
      signals: [
        { k: "SOFR", v: fed.sofr },
        { k: "TGA", v: fed.tga },
        { k: "ON-RRP", v: fed.onRrp },
        { k: "RESERVE", v: fed.reserveAdequacy },
        { k: "FF_IMP", v: fed.fedFundsImpact },
      ],
    },
    {
      id: 1,
      name: "LiquidityTensor",
      color: "#00d4aa",
      output: liq.L,
      outputLabel: "L(t)",
      signals: [
        { k: "SOFR×0.2", v: liq.sofrWeight },
        { k: "CREDIT×0.25", v: liq.creditWeight },
        { k: "M2V×0.15", v: liq.m2vWeight },
        { k: "HY_SPD×0.15", v: liq.hySpreadWeight },
        { k: "FED_PLMB×0.25", v: liq.fedPlumbingWeight },
      ],
    },
    {
      id: 2,
      name: "ReserveKernel",
      color: "#bc8cff",
      output: kernel.equityImpact,
      outputLabel: "EQ_IMPACT",
      signals: [
        { k: "ΔReserves", v: kernel.deltaReserves / 50 },
        { k: "EQ_IMP", v: kernel.equityImpact },
        { k: "CREDIT_IMP", v: kernel.creditImpact },
        { k: "DECAY", v: kernel.decay },
      ],
    },
    {
      id: 3,
      name: "RiskStateModel",
      color: "#f85149",
      output: risk.R,
      outputLabel: "R(t)",
      signals: [
        { k: "VIX×0.30", v: risk.vix },
        { k: "RVOL×0.20", v: risk.realizedVol },
        { k: "CREDIT×0.20", v: risk.credit },
        { k: "CORR×0.15", v: risk.correlation },
        { k: "TAIL×0.15", v: risk.tailRisk },
      ],
      range: [0, 1] as [number, number],
    },
    {
      id: 4,
      name: "CapitalFlow",
      color: "#4ecdc4",
      output: flow.F,
      outputLabel: "F(t)",
      signals: [
        { k: "SECTOR_MOM", v: flow.sectorMomentum },
        { k: "LDRS/LGDS", v: flow.leadersLaggards },
        { k: "ROT_VEL", v: flow.rotationVelocity },
        { k: "BREADTH", v: flow.breadth * 2 - 1 },
        { k: "PERSIST", v: flow.persistence },
      ],
    },
    {
      id: 5,
      name: "RegimeEngine",
      color: REGIME_COLORS[regime.current],
      output: regime.confidence,
      outputLabel: regime.current,
      signals: [
        { k: "TRENDING", v: regime.transitions.TRENDING },
        { k: "RANGE", v: regime.transitions.RANGE * 2 - 1 },
        { k: "STRESS", v: -(regime.transitions.STRESS) },
        { k: "CRASH", v: -(regime.transitions.CRASH * 4) },
      ],
    },
  ];

  return (
    <div className="flex flex-col h-full gap-[2px] p-1.5 overflow-auto">
      {layers.map((layer, idx) => (
        <div key={layer.id} className="flex flex-col gap-0.5">
          {/* Layer header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="text-[7px] text-terminal-text-faint">L{layer.id}</span>
              <span className="text-[8px] font-bold" style={{ color: layer.color }}>
                {layer.name}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-[7px] text-terminal-text-faint">{layer.outputLabel}</span>
              <span
                className="text-[9px] font-bold tabular-nums"
                style={{ color: layer.output >= 0 ? "#00d4aa" : "#f85149" }}
              >
                {layer.output >= 0 ? "+" : ""}{layer.output.toFixed(3)}
              </span>
            </div>
          </div>

          {/* Signals */}
          <div className="grid grid-cols-1 gap-[2px] pl-3">
            {layer.signals.slice(0, 4).map((sig) => (
              <div key={sig.k} className="flex items-center gap-1.5">
                <span className="text-[7px] text-terminal-text-faint w-[62px] flex-shrink-0">{sig.k}</span>
                <SignalBar value={sig.v} range={layer.range} />
                <span
                  className="text-[7px] tabular-nums w-[30px] text-right flex-shrink-0"
                  style={{ color: sig.v >= 0 ? "#00d4aa" : "#f85149" }}
                >
                  {sig.v >= 0 ? "+" : ""}{sig.v.toFixed(2)}
                </span>
              </div>
            ))}
          </div>

          {/* Flow arrow between layers */}
          {idx < layers.length - 1 && (
            <div className="flex items-center justify-center py-0.5">
              <div className="flex flex-col items-center gap-[1px]">
                <div className="w-px h-2 bg-terminal-border/50" />
                <div
                  className="w-0 h-0"
                  style={{
                    borderLeft: "3px solid transparent",
                    borderRight: "3px solid transparent",
                    borderTop: `4px solid ${layers[idx + 1].color}50`,
                  }}
                />
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// GATE-Z SLEEVE ALLOCATION
// ═══════════════════════════════════════════════════════════════════════════════

function GateZPanel({ regime, gates }: { regime: Regime; gates: GateScore[] }) {
  const alloc = REGIME_ALLOC[regime];
  const sleeveData = [
    { name: "P1: Directional Eq", pct: alloc.P1, color: SLEEVE_COLORS[0] },
    { name: "P2: Factor Rotation", pct: alloc.P2, color: SLEEVE_COLORS[1] },
    { name: "P3: Macro/Trend", pct: alloc.P3, color: SLEEVE_COLORS[2] },
    { name: "P4: Options/Conv", pct: alloc.P4, color: SLEEVE_COLORS[3] },
    { name: "P5: Hedges/Vol", pct: alloc.P5, color: SLEEVE_COLORS[4] },
  ];

  const barData = sleeveData.map(s => ({ name: s.name.split(":")[0], value: s.pct, color: s.color, full: s.name }));

  return (
    <div className="flex flex-col h-full gap-1 p-1.5 text-[9px] font-mono tabular-nums">
      {/* Stacked bar */}
      <div>
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1">
          Gate-Z Allocation — {regime}
        </div>
        <div className="flex h-5 rounded overflow-hidden gap-[1px]">
          {sleeveData.map((s) => (
            <div
              key={s.name}
              className="flex items-center justify-center relative group transition-all duration-1000"
              style={{ width: `${s.pct}%`, backgroundColor: s.color + "60", borderRight: `1px solid ${s.color}80` }}
            >
              {s.pct >= 10 && (
                <span className="text-[7px] font-bold" style={{ color: s.color }}>{s.pct}%</span>
              )}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block bg-terminal-surface border border-terminal-border rounded px-1.5 py-1 text-[8px] text-terminal-text-muted whitespace-nowrap z-10">
                {s.name}: {s.pct}%
              </div>
            </div>
          ))}
        </div>
        {/* Legend */}
        <div className="flex flex-wrap gap-x-2 gap-y-0.5 mt-1">
          {sleeveData.map((s) => (
            <div key={s.name} className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-sm flex-shrink-0" style={{ backgroundColor: s.color }} />
              <span className="text-[7px] text-terminal-text-faint">{s.name}</span>
              <span className="text-[7px] font-bold" style={{ color: s.color }}>{s.pct}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Gate params */}
      <div className="flex gap-2 text-[7px]">
        <div className="bg-terminal-surface-2 rounded px-1.5 py-1 flex-1">
          <span className="text-terminal-text-faint">Leverage</span>
          <span className="ml-1 text-terminal-text-primary font-bold">{alloc.lev}x</span>
        </div>
        <div className="bg-terminal-surface-2 rounded px-1.5 py-1 flex-1">
          <span className="text-terminal-text-faint">β Cap</span>
          <span className="ml-1 text-terminal-accent font-bold">{alloc.beta}</span>
        </div>
        <div className="bg-terminal-surface-2 rounded px-1.5 py-1 flex-1">
          <span className="text-terminal-text-faint">Pass ≥</span>
          <span className="ml-1 text-terminal-text-primary font-bold">0.50 wgt</span>
        </div>
      </div>

      {/* Gate header */}
      <div className="flex items-center text-[7px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border pb-0.5">
        <span className="w-[40px]">TICKER</span>
        <span className="flex-1 text-center">G1×20%</span>
        <span className="flex-1 text-center">G2×25%</span>
        <span className="flex-1 text-center">G3×30%</span>
        <span className="flex-1 text-center">G4×25%</span>
        <span className="w-[32px] text-center">WGT</span>
        <span className="w-[24px] text-center">STS</span>
      </div>

      {/* Gate rows */}
      <div className="flex-1 overflow-auto">
        {gates.map((g) => (
          <div
            key={g.id}
            className="flex items-center py-[3px] border-b border-terminal-border/20 hover:bg-white/[0.02]"
          >
            <span className="w-[40px] text-terminal-accent text-[8px] font-bold">{g.ticker}</span>
            {[g.g1, g.g2, g.g3, g.g4].map((score, i) => (
              <div key={i} className="flex-1 flex items-center justify-center">
                <div className="w-[24px] h-2 bg-terminal-surface-2 rounded-sm overflow-hidden">
                  <div
                    className="h-full rounded-sm"
                    style={{
                      width: `${score * 100}%`,
                      backgroundColor: score >= 0.30 ? "#00d4aa" : "#f85149",
                    }}
                  />
                </div>
                <span
                  className="text-[7px] ml-0.5 tabular-nums"
                  style={{ color: score >= 0.30 ? "#00d4aa" : "#f85149" }}
                >
                  {score.toFixed(2)}
                </span>
              </div>
            ))}
            <div className="w-[32px] text-center">
              <span
                className="text-[8px] font-bold tabular-nums"
                style={{ color: g.weighted >= 0.50 ? "#00d4aa" : "#f85149" }}
              >
                {g.weighted.toFixed(2)}
              </span>
            </div>
            <div className="w-[24px] text-center">
              <span
                className="text-[7px] font-bold px-0.5 py-px rounded"
                style={{
                  backgroundColor: g.pass ? "rgba(0,212,170,0.15)" : "rgba(248,81,73,0.15)",
                  color: g.pass ? "#00d4aa" : "#f85149",
                }}
              >
                {g.pass ? "PASS" : "FAIL"}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// RISK GOVERNOR
// ═══════════════════════════════════════════════════════════════════════════════

function RiskGovernorPanel({ gov, kill }: { gov: RiskGovernor; kill: KillSwitch }) {
  const limits = [
    { label: "Position", val: gov.positionPct, max: 5, fmt: (v: number) => `${v.toFixed(1)}%`, warn: 4, err: 5 },
    { label: "Sector", val: gov.sectorPct, max: 25, fmt: (v: number) => `${v.toFixed(1)}%`, warn: 22, err: 25 },
    { label: "Leverage", val: gov.leverage, max: 3, fmt: (v: number) => `${v.toFixed(2)}x`, warn: 2.7, err: 3.0 },
    { label: "VaR (NAV)", val: gov.varPct, max: 1.5, fmt: (v: number) => `${v.toFixed(2)}%`, warn: 1.3, err: 1.5 },
    { label: "Drawdown", val: gov.drawdownPct, max: 15, fmt: (v: number) => `${v.toFixed(1)}%`, warn: 12, err: 15 },
    { label: "Beta", val: gov.beta, max: 0.65, fmt: (v: number) => `${v.toFixed(3)}β`, warn: 0.60, err: 0.65 },
    { label: "Crash Floor", val: gov.crashFloor, max: 50, fmt: (v: number) => `+${v.toFixed(1)}%`, warn: 0, err: 0, inverse: true },
  ];

  const killConditions = [
    {
      label: "HY OAS +35bp",
      val: `+${kill.hyOas.toFixed(1)}bp`,
      fired: kill.hyOas >= 35,
      threshold: "≥+35bp",
    },
    {
      label: "VIX Term Flat/Inv",
      val: kill.vixTerm >= 0 ? `+${kill.vixTerm.toFixed(3)}` : kill.vixTerm.toFixed(3),
      fired: kill.vixTerm <= 0,
      threshold: "≤0.000",
    },
    {
      label: "Breadth <50%",
      val: `${(kill.breadth * 100).toFixed(1)}%`,
      fired: kill.breadth < 0.5,
      threshold: "<50%",
    },
  ];

  return (
    <div className="flex flex-col h-full gap-1.5 p-1.5 text-[9px] font-mono tabular-nums">
      {/* Limit checks */}
      <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider">Risk Limits</div>
      <div className="flex flex-col gap-1">
        {limits.map((lim) => {
          const pct = lim.inverse
            ? Math.min((lim.val / 50) * 100, 100)
            : Math.min((lim.val / lim.max) * 100, 100);
          const isWarn = !lim.inverse && lim.val >= lim.warn && lim.val < lim.err;
          const isErr = !lim.inverse && lim.val >= lim.err;
          const color = lim.inverse
            ? "#00d4aa"
            : isErr ? "#f85149" : isWarn ? "#d29922" : "#00d4aa";

          return (
            <div key={lim.label}>
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-terminal-text-faint text-[8px]">{lim.label}</span>
                <div className="flex items-center gap-1">
                  <span className="text-[8px]" style={{ color }}>{lim.fmt(lim.val)}</span>
                  {isErr && <span className="text-[6px] font-bold text-terminal-negative px-0.5 bg-terminal-negative/10 rounded">BREACH</span>}
                  {isWarn && <span className="text-[6px] font-bold text-terminal-warning px-0.5 bg-terminal-warning/10 rounded">WARN</span>}
                  {!isErr && !isWarn && <span className="text-[6px] font-bold text-terminal-positive px-0.5 bg-terminal-positive/10 rounded">OK</span>}
                </div>
              </div>
              <div className="h-1.5 bg-terminal-surface-2 rounded-sm overflow-hidden">
                <div
                  className="h-full rounded-sm transition-all duration-500"
                  style={{ width: `${pct}%`, backgroundColor: color + "80" }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="border-t border-terminal-border my-1" />

      {/* Kill switch */}
      <div className="flex items-center justify-between">
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider">Kill Switch</div>
        <div
          className="text-[7px] font-bold px-1 py-0.5 rounded"
          style={{
            backgroundColor: kill.active ? "rgba(248,81,73,0.2)" : "rgba(0,212,170,0.1)",
            color: kill.active ? "#f85149" : "#00d4aa",
          }}
        >
          {kill.active ? "🔴 ACTIVE" : "ARMED"}
        </div>
      </div>
      <div className="flex flex-col gap-1">
        {killConditions.map((cond) => (
          <div
            key={cond.label}
            className="flex items-center gap-1.5 px-1.5 py-1 rounded"
            style={{
              background: cond.fired ? "rgba(248,81,73,0.08)" : "rgba(0,212,170,0.04)",
              border: `1px solid ${cond.fired ? "#f8514940" : "#00d4aa20"}`,
            }}
          >
            <div
              className="w-1.5 h-1.5 rounded-full flex-shrink-0"
              style={{ backgroundColor: cond.fired ? "#f85149" : "#484f58" }}
            />
            <span className="text-[8px] text-terminal-text-faint flex-1">{cond.label}</span>
            <span
              className="text-[8px] font-bold tabular-nums"
              style={{ color: cond.fired ? "#f85149" : "#3fb950" }}
            >
              {cond.val}
            </span>
            <span className="text-[6px] text-terminal-text-faint">{cond.threshold}</span>
          </div>
        ))}
      </div>
      {kill.active && (
        <div className="mt-0.5 px-1.5 py-1 bg-terminal-negative/10 border border-terminal-negative/30 rounded text-[8px] text-terminal-negative text-center animate-pulse font-bold">
          FORCE β≤0.35 — MAX TAIL SPEND ACTIVE
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// FCLP CYCLE
// ═══════════════════════════════════════════════════════════════════════════════

function FCLPCyclePanel({ currentStep }: { currentStep: number }) {
  const now = new Date();
  const lastCalib = new Date(now.getTime() - 47000);
  const driftReport = [
    { signal: "SOFR_DELTA", drift: 0.023, status: "OK" },
    { signal: "VIX_REGIME", drift: 0.041, status: "WARN" },
    { signal: "BREADTH_RAW", drift: 0.012, status: "OK" },
    { signal: "HY_SPREAD", drift: 0.055, status: "OK" },
  ];

  return (
    <div className="flex flex-col h-full gap-1.5 p-1.5 text-[9px] font-mono tabular-nums">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider">FCLP 6-Step Cycle</div>
        <div className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-terminal-positive animate-pulse" />
          <span className="text-[7px] text-terminal-text-faint">RUNNING</span>
        </div>
      </div>

      {/* Step loop */}
      <div className="flex gap-1">
        {FCLP_STEPS.map((step) => {
          const isActive = step.id === currentStep;
          const isDone = step.id < currentStep;
          return (
            <div
              key={step.id}
              className="flex-1 flex flex-col items-center gap-0.5"
            >
              <div
                className="w-6 h-6 rounded-full flex items-center justify-center text-[8px] font-bold border transition-all duration-300"
                style={{
                  backgroundColor: isActive
                    ? "rgba(0,212,170,0.25)"
                    : isDone
                    ? "rgba(63,185,80,0.15)"
                    : "rgba(30,38,51,0.5)",
                  borderColor: isActive
                    ? "#00d4aa"
                    : isDone
                    ? "#3fb950"
                    : "#1e2633",
                  color: isActive ? "#00d4aa" : isDone ? "#3fb950" : "#484f58",
                  boxShadow: isActive ? "0 0 8px rgba(0,212,170,0.4)" : "none",
                }}
              >
                {isDone ? "✓" : step.id}
              </div>
              <span
                className="text-[6px] text-center leading-tight"
                style={{
                  color: isActive ? "#00d4aa" : isDone ? "#3fb950" : "#484f58",
                }}
              >
                {step.short}
              </span>
            </div>
          );
        })}
      </div>

      {/* Current step detail */}
      <div className="bg-terminal-surface-2 rounded px-1.5 py-1">
        <div className="text-[7px] text-terminal-text-faint">Current Step</div>
        <div className="text-[9px] text-terminal-accent font-bold">
          [{currentStep}] {FCLP_STEPS[currentStep - 1]?.label}
        </div>
        <div className="flex items-center gap-1 mt-0.5">
          <div className="flex-1 h-1 bg-terminal-surface rounded-sm overflow-hidden">
            <div
              className="h-full bg-terminal-accent rounded-sm animate-pulse"
              style={{ width: `${((currentStep - 1) / 6) * 100 + 8}%` }}
            />
          </div>
          <span className="text-[7px] text-terminal-text-faint">{Math.round(((currentStep - 1) / 6) * 100)}%</span>
        </div>
      </div>

      {/* Last calibration time */}
      <div className="flex justify-between text-[8px]">
        <span className="text-terminal-text-faint">Last Calib</span>
        <span className="text-terminal-text-primary">
          {lastCalib.toLocaleTimeString("en-US", { hour12: false })}
        </span>
      </div>
      <div className="flex justify-between text-[8px]">
        <span className="text-terminal-text-faint">Cycle Time</span>
        <span className="text-terminal-positive">47.2s</span>
      </div>
      <div className="flex justify-between text-[8px]">
        <span className="text-terminal-text-faint">Next Calib</span>
        <span className="text-terminal-warning">
          {new Date(now.getTime() + 313000).toLocaleTimeString("en-US", { hour12: false })}
        </span>
      </div>

      {/* Drift report */}
      <div className="border-t border-terminal-border pt-1">
        <div className="text-[7px] text-terminal-text-faint uppercase tracking-wider mb-1">Drift Report</div>
        {driftReport.map((d) => (
          <div key={d.signal} className="flex items-center gap-1 py-0.5">
            <span
              className="w-1.5 h-1.5 rounded-full flex-shrink-0"
              style={{ backgroundColor: d.status === "OK" ? "#3fb950" : "#d29922" }}
            />
            <span className="text-[7px] text-terminal-text-faint flex-1">{d.signal}</span>
            <span
              className="text-[7px] font-bold"
              style={{ color: d.status === "OK" ? "#3fb950" : "#d29922" }}
            >
              ±{d.drift.toFixed(3)}
            </span>
            <span
              className="text-[6px] px-0.5 rounded"
              style={{
                color: d.status === "OK" ? "#3fb950" : "#d29922",
                backgroundColor: d.status === "OK" ? "rgba(63,185,80,0.1)" : "rgba(210,153,34,0.1)",
              }}
            >
              {d.status}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRESS SCENARIO TABLE
// ═══════════════════════════════════════════════════════════════════════════════

function StressScenarioTable() {
  return (
    <div className="flex flex-col h-full text-[9px] font-mono tabular-nums p-1.5">
      <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1">Stress Scenarios</div>
      <div className="flex items-center text-[7px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border pb-0.5 mb-0.5">
        <span className="w-[90px]">Scenario</span>
        <span className="w-[48px]">Regime</span>
        <span className="w-[32px] text-right">β</span>
        <span className="w-[28px] text-right">Lev</span>
        <span className="w-[28px] text-right">Eq%</span>
        <span className="w-[32px] text-right">Hdg%</span>
        <span className="flex-1 text-right">Risk</span>
      </div>
      <div className="flex-1 overflow-auto">
        {STRESS_SCENARIOS.map((s) => (
          <div
            key={s.name}
            className="flex items-center py-[3px] border-b border-terminal-border/20 hover:bg-white/[0.02]"
          >
            <span className="w-[90px] text-terminal-text-primary text-[8px] truncate">{s.name}</span>
            <span
              className="w-[48px] text-[7px] font-bold"
              style={{ color: REGIME_COLORS[s.regime as Regime] }}
            >
              {s.regime}
            </span>
            <span className="w-[32px] text-right text-terminal-text-muted">{s.beta.toFixed(2)}</span>
            <span className="w-[28px] text-right text-terminal-text-muted">{s.lev}x</span>
            <span className="w-[28px] text-right text-terminal-positive">{s.eq}%</span>
            <span className="w-[32px] text-right text-terminal-negative">{s.hedge}%</span>
            <div className="flex-1 flex items-center justify-end gap-1">
              <div className="w-12 h-1.5 bg-terminal-surface-2 rounded-sm overflow-hidden">
                <div
                  className="h-full rounded-sm"
                  style={{
                    width: `${(s.risk / 10) * 100}%`,
                    backgroundColor: s.risk > 7 ? "#f85149" : s.risk > 5 ? "#d29922" : "#00d4aa",
                  }}
                />
              </div>
              <span
                className="text-[8px]"
                style={{ color: s.risk > 7 ? "#f85149" : s.risk > 5 ? "#d29922" : "#00d4aa" }}
              >
                {s.risk.toFixed(1)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEARNING LOOP + CUBE HISTORY
// ═══════════════════════════════════════════════════════════════════════════════

function LearningLoopPanel({ betaTrend }: { betaTrend: { t: number; beta: number }[] }) {
  const stats = {
    accuracy: 87.4,
    sampleSize: 1284,
    sharpeDelta: 0.23,
    adjustments: [
      { signal: "VIX_WEIGHT", from: 0.30, to: 0.31, delta: "+0.01" },
      { signal: "HY_SPD_W", from: 0.15, to: 0.14, delta: "-0.01" },
      { signal: "SOFR_COEF", from: 0.20, to: 0.21, delta: "+0.01" },
    ],
  };

  return (
    <div className="flex flex-col h-full gap-1.5 p-1.5 text-[9px] font-mono tabular-nums">
      {/* Learning stats */}
      <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider">Learning Loop</div>
      <div className="grid grid-cols-3 gap-1 text-[8px]">
        <div className="bg-terminal-surface-2 rounded px-1.5 py-1">
          <div className="text-terminal-text-faint text-[7px]">Accuracy</div>
          <div className="text-terminal-positive font-bold">{stats.accuracy}%</div>
        </div>
        <div className="bg-terminal-surface-2 rounded px-1.5 py-1">
          <div className="text-terminal-text-faint text-[7px]">Samples</div>
          <div className="text-terminal-text-primary font-bold">{stats.sampleSize.toLocaleString()}</div>
        </div>
        <div className="bg-terminal-surface-2 rounded px-1.5 py-1">
          <div className="text-terminal-text-faint text-[7px]">ΔSharpe</div>
          <div className="text-terminal-accent font-bold">+{stats.sharpeDelta}</div>
        </div>
      </div>

      {/* Suggested adjustments */}
      <div>
        <div className="text-[7px] text-terminal-text-faint uppercase tracking-wider mb-0.5">Suggested Adjustments</div>
        {stats.adjustments.map((adj) => (
          <div key={adj.signal} className="flex items-center gap-1.5 py-0.5 border-b border-terminal-border/20">
            <span className="text-[7px] text-terminal-text-faint flex-1">{adj.signal}</span>
            <span className="text-[7px] text-terminal-text-muted">{adj.from.toFixed(2)}</span>
            <span className="text-[7px] text-terminal-text-faint">→</span>
            <span className="text-[7px] text-terminal-text-muted">{adj.to.toFixed(2)}</span>
            <span
              className="text-[7px] font-bold"
              style={{ color: adj.delta.startsWith("+") ? "#00d4aa" : "#f85149" }}
            >
              {adj.delta}
            </span>
          </div>
        ))}
      </div>

      {/* Regime distribution pie */}
      <div className="border-t border-terminal-border pt-1">
        <div className="text-[7px] text-terminal-text-faint uppercase tracking-wider mb-0.5">Regime Distribution (90d)</div>
        <div className="flex gap-2 items-center">
          <div className="w-[60px] h-[60px] flex-shrink-0">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={REGIME_DIST}
                  cx="50%"
                  cy="50%"
                  innerRadius="45%"
                  outerRadius="75%"
                  paddingAngle={1}
                  dataKey="value"
                  stroke="none"
                >
                  {REGIME_DIST.map((entry, i) => (
                    <Cell key={i} fill={entry.color} opacity={0.8} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-col gap-0.5 flex-1">
            {REGIME_DIST.map((d) => (
              <div key={d.name} className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-sm flex-shrink-0" style={{ backgroundColor: d.color }} />
                <span className="text-[7px] text-terminal-text-faint flex-1">{d.name}</span>
                <span className="text-[7px] font-bold" style={{ color: d.color }}>{d.value}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Average beta trend sparkline */}
      <div className="border-t border-terminal-border pt-1 flex-1 min-h-0">
        <div className="text-[7px] text-terminal-text-faint uppercase tracking-wider mb-0.5">
          Avg Beta Trend
          <span className="ml-1 text-terminal-accent">{betaTrend[betaTrend.length - 1]?.beta.toFixed(3)}</span>
        </div>
        <div className="h-[40px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={betaTrend} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
              <defs>
                <linearGradient id="betaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#00d4aa" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#00d4aa" stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area type="monotone" dataKey="beta" stroke="#00d4aa" strokeWidth={1} fill="url(#betaGrad)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN PAGE COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export default function MetadronCubePage() {
  // Core state
  const [tensor, setTensor] = useState<CoreTensor>(() => generateCoreTensor());
  const [fed, setFed] = useState<FedPlumbingLayer>(() => generateFedPlumbing());
  const [liq, setLiq] = useState<LiquidityTensor>(() => generateLiquidityTensor());
  const [kernel, setKernel] = useState<ReserveFlowKernel>(() => generateReserveKernel());
  const [risk, setRisk] = useState<RiskStateModel>(() => generateRiskModel());
  const [flow, setFlow] = useState<CapitalFlowModel>(() => generateCapitalFlow());
  const [regime, setRegime] = useState<RegimeState>(() => generateRegimeState());
  const [gates, setGates] = useState<GateScore[]>(() => generateGateScores());
  const [killSwitch, setKillSwitch] = useState<KillSwitch>(() => generateKillSwitch());
  const [govData, setGovData] = useState<RiskGovernor>(() => generateRiskGovernor());
  const [fclpStep, setFclpStep] = useState(3);
  const [regimeHistory] = useState(() => generateRegimeHistory());
  const betaTrend = useMemo(() => generateBetaTrend(), []);
  const [tick, setTick] = useState(0);

  // Core tensor updates every 2s
  useEffect(() => {
    const iv = setInterval(() => {
      setTensor(generateCoreTensor());
      setFed(generateFedPlumbing());
      setLiq(generateLiquidityTensor());
      setKernel(generateReserveKernel());
      setRisk(generateRiskModel());
      setFlow(generateCapitalFlow());
      setRegime(generateRegimeState());
      setKillSwitch(generateKillSwitch());
      setGovData(generateRiskGovernor());
      setTick((t) => t + 1);
    }, 2500);
    return () => clearInterval(iv);
  }, []);

  // Gate scores update every 5s
  useEffect(() => {
    const iv = setInterval(() => {
      setGates(generateGateScores());
    }, 5000);
    return () => clearInterval(iv);
  }, []);

  // FCLP step cycles every 3s
  useEffect(() => {
    const iv = setInterval(() => {
      setFclpStep((s) => (s % 6) + 1);
    }, 3000);
    return () => clearInterval(iv);
  }, []);

  // Tick time display
  const now = new Date();
  const timeStr = now.toLocaleTimeString("en-US", { hour12: false });

  return (
    <div className="h-full flex flex-col bg-terminal-bg overflow-hidden" data-testid="metadron-cube-page">
      {/* ── Header bar ── */}
      <div className="flex-shrink-0 flex items-center justify-between px-3 py-1.5 border-b border-terminal-border bg-terminal-surface">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-terminal-accent animate-pulse" />
            <span className="text-[10px] font-bold text-terminal-accent tracking-widest font-mono">METADRON CUBE</span>
          </div>
          <span className="text-[8px] text-terminal-text-faint font-mono">C(t) = f(L_t, R_t, F_t)</span>
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-terminal-surface-2 border border-terminal-border">
            <span className="text-[8px] text-terminal-text-faint">L</span>
            <span className="text-[9px] font-bold text-terminal-accent tabular-nums">{tensor.L >= 0 ? "+" : ""}{tensor.L.toFixed(3)}</span>
            <span className="text-terminal-border mx-1">|</span>
            <span className="text-[8px] text-terminal-text-faint">R</span>
            <span className="text-[9px] font-bold text-terminal-negative tabular-nums">{tensor.R.toFixed(3)}</span>
            <span className="text-terminal-border mx-1">|</span>
            <span className="text-[8px] text-terminal-text-faint">F</span>
            <span className="text-[9px] font-bold text-terminal-blue tabular-nums">{tensor.F >= 0 ? "+" : ""}{tensor.F.toFixed(3)}</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div
            className="text-[9px] font-bold px-2 py-0.5 rounded border"
            style={{
              color: REGIME_COLORS[regime.current],
              borderColor: REGIME_COLORS[regime.current] + "50",
              backgroundColor: REGIME_BG[regime.current],
            }}
          >
            {regime.current} — {(regime.confidence * 100).toFixed(1)}%
          </div>
          <div className="flex items-center gap-1">
            <span className="text-[8px] text-terminal-text-faint font-mono tabular-nums">{timeStr}</span>
            <span className="text-[7px] text-terminal-text-faint">UTC-7</span>
          </div>
        </div>
      </div>

      {/* ── Main layout: 3 rows ── */}
      <div className="flex-1 overflow-hidden flex flex-col gap-[2px] p-[2px]">

        {/* TOP ROW — 30% height: Cube + Regime */}
        <div className="flex-[3] flex gap-[2px] min-h-0">
          {/* Cube canvas */}
          <div className="flex-[1.4] min-w-0">
            <DashboardPanel
              title="CORE TENSOR — 3D OPERATING SPACE"
              className="h-full"
              noPadding
              headerRight={
                <div className="flex items-center gap-2 font-mono text-[8px]">
                  <span className="text-terminal-text-faint">C(t)</span>
                  <span
                    className="font-bold tabular-nums"
                    style={{ color: tensor.Ct >= 0 ? "#00d4aa" : "#f85149" }}
                  >
                    {tensor.Ct >= 0 ? "+" : ""}{tensor.Ct.toFixed(4)}
                  </span>
                </div>
              }
            >
              <CubeCanvas tensor={tensor} />
            </DashboardPanel>
          </div>

          {/* Regime panel */}
          <div className="flex-1 min-w-0">
            <DashboardPanel
              title="REGIME ENGINE — MARKOV STATE"
              className="h-full"
              noPadding
              headerRight={
                <span className="text-[7px] text-terminal-text-faint font-mono">FCLP:{fclpStep}/6</span>
              }
            >
              <RegimePanel regime={regime} history={regimeHistory} />
            </DashboardPanel>
          </div>
        </div>

        {/* MIDDLE ROW — 40% height: Layers + GateZ + Risk */}
        <div className="flex-[4] flex gap-[2px] min-h-0">
          {/* Layer pipeline */}
          <div className="flex-[1.1] min-w-0">
            <DashboardPanel
              title="LAYER PIPELINE — SIGNAL FLOW"
              className="h-full"
              noPadding
              headerRight={
                <div className="flex items-center gap-1 font-mono text-[7px] text-terminal-text-faint">
                  <span className="w-1.5 h-1.5 rounded-full bg-terminal-positive animate-pulse" />
                  LIVE
                </div>
              }
            >
              <LayerPipeline
                fed={fed}
                liq={liq}
                kernel={kernel}
                risk={risk}
                flow={flow}
                regime={regime}
              />
            </DashboardPanel>
          </div>

          {/* Gate-Z sleeve allocation */}
          <div className="flex-[1.5] min-w-0">
            <DashboardPanel
              title="GATE-Z ALLOCATOR — 5-SLEEVE ENTRY"
              className="h-full"
              noPadding
              headerRight={
                <div className="flex items-center gap-1 font-mono text-[8px]">
                  <span className="text-terminal-text-faint">Pass:</span>
                  <span className="text-terminal-positive font-bold">
                    {gates.filter(g => g.pass).length}/{gates.length}
                  </span>
                </div>
              }
            >
              <GateZPanel regime={regime.current} gates={gates} />
            </DashboardPanel>
          </div>

          {/* Risk Governor */}
          <div className="flex-1 min-w-0">
            <DashboardPanel
              title="RISK GOVERNOR — LIMITS"
              className="h-full"
              noPadding
              headerRight={
                <span
                  className="text-[7px] font-bold px-1 rounded"
                  style={{
                    color: killSwitch.active ? "#f85149" : "#00d4aa",
                    backgroundColor: killSwitch.active ? "rgba(248,81,73,0.15)" : "rgba(0,212,170,0.1)",
                  }}
                >
                  KILL: {killSwitch.active ? "ON" : "OFF"}
                </span>
              }
            >
              <RiskGovernorPanel gov={govData} kill={killSwitch} />
            </DashboardPanel>
          </div>
        </div>

        {/* BOTTOM ROW — 30% height: FCLP + Stress + Learning */}
        <div className="flex-[3] flex gap-[2px] min-h-0">
          {/* FCLP cycle */}
          <div className="flex-1 min-w-0">
            <DashboardPanel
              title="FCLP CALIBRATION CYCLE"
              className="h-full"
              noPadding
              headerRight={
                <span className="text-[7px] text-terminal-text-faint font-mono">
                  Step {fclpStep}/6
                </span>
              }
            >
              <FCLPCyclePanel currentStep={fclpStep} />
            </DashboardPanel>
          </div>

          {/* Stress scenarios */}
          <div className="flex-[1.6] min-w-0">
            <DashboardPanel
              title="STRESS SCENARIOS — PRE-BUILT 6"
              className="h-full"
              noPadding
              headerRight={
                <span className="text-[7px] text-terminal-text-faint font-mono">6 SCENARIOS</span>
              }
            >
              <StressScenarioTable />
            </DashboardPanel>
          </div>

          {/* Learning loop + cube history */}
          <div className="flex-[1.1] min-w-0">
            <DashboardPanel
              title="LEARNING LOOP — CUBE HISTORY"
              className="h-full"
              noPadding
              headerRight={
                <span className="text-[7px] text-terminal-positive font-mono font-bold">87.4% ACC</span>
              }
            >
              <LearningLoopPanel betaTrend={betaTrend} />
            </DashboardPanel>
          </div>
        </div>
      </div>
    </div>
  );
}
