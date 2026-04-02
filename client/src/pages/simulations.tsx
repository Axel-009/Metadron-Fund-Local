import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import { useEngineQuery } from "@/hooks/use-engine-api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, Cell, ReferenceLine,
} from "recharts";

// ═══════════ TYPES ═══════════

type Regime = "BULL" | "BEAR" | "TRANSITION";

// ═══════════ HMM DATA ═══════════

const REGIME_COLORS: Record<Regime, string> = {
  BULL: "#3fb950",
  BEAR: "#f85149",
  TRANSITION: "#d29922",
};

const REGIME_PROBS: { regime: Regime; prob: number }[] = [
  { regime: "BULL", prob: 35 },
  { regime: "BEAR", prob: 22 },
  { regime: "TRANSITION", prob: 43 },
];

const TRANSITION_MATRIX = [
  { from: "BULL", toBull: 0.72, toBear: 0.08, toTrans: 0.20 },
  { from: "BEAR", toBull: 0.12, toBear: 0.63, toTrans: 0.25 },
  { from: "TRANS", toBull: 0.35, toBear: 0.22, toTrans: 0.43 },
];

function generateRegimeHistory(days: number): { day: number; regime: Regime; value: number }[] {
  const regimes: Regime[] = ["BULL", "BEAR", "TRANSITION"];
  let current: Regime = "BULL";
  const result = [];
  let cumReturn = 100;
  for (let d = 0; d < days; d++) {
    const rand = Math.random();
    if (current === "BULL") {
      if (rand < 0.08) current = "BEAR";
      else if (rand < 0.28) current = "TRANSITION";
    } else if (current === "BEAR") {
      if (rand < 0.12) current = "BULL";
      else if (rand < 0.37) current = "TRANSITION";
    } else {
      if (rand < 0.35) current = "BULL";
      else if (rand < 0.57) current = "BEAR";
    }
    const dailyRet = current === "BULL" ? 0.001 + Math.random() * 0.008
      : current === "BEAR" ? -0.008 + Math.random() * 0.006
      : (Math.random() - 0.5) * 0.006;
    cumReturn *= (1 + dailyRet);
    result.push({ day: d, regime: current, value: +cumReturn.toFixed(2) });
  }
  return result;
}

const REGIME_STATS = [
  { regime: "BULL", avgDuration: "22.4d", avgReturn: "+1.82%", volatility: "8.2%", frequency: "48%" },
  { regime: "BEAR", avgDuration: "14.8d", avgReturn: "-1.34%", volatility: "18.7%", frequency: "21%" },
  { regime: "TRANSITION", avgDuration: "8.2d", avgReturn: "+0.12%", volatility: "13.4%", frequency: "31%" },
];

// ═══════════ BLACK-SCHOLES ═══════════

function normCDF(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const t = 1.0 / (1.0 + p * Math.abs(x) / Math.sqrt(2));
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x / 2);
  return 0.5 * (1.0 + sign * y);
}

function normPDF(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

function blackScholes(S: number, K: number, T: number, r: number, sigma: number) {
  if (T <= 0) return { call: Math.max(S - K, 0), put: Math.max(K - S, 0), d1: 0, d2: 0 };
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  const call = S * normCDF(d1) - K * Math.exp(-r * T) * normCDF(d2);
  const put = K * Math.exp(-r * T) * normCDF(-d2) - S * normCDF(-d1);
  // Greeks
  const delta_c = normCDF(d1);
  const delta_p = delta_c - 1;
  const gamma = normPDF(d1) / (S * sigma * Math.sqrt(T));
  const theta_c = (-(S * normPDF(d1) * sigma) / (2 * Math.sqrt(T)) - r * K * Math.exp(-r * T) * normCDF(d2)) / 365;
  const theta_p = (-(S * normPDF(d1) * sigma) / (2 * Math.sqrt(T)) + r * K * Math.exp(-r * T) * normCDF(-d2)) / 365;
  const vega = S * normPDF(d1) * Math.sqrt(T) / 100;
  const rho_c = K * T * Math.exp(-r * T) * normCDF(d2) / 100;
  const rho_p = -K * T * Math.exp(-r * T) * normCDF(-d2) / 100;
  return { call, put, d1, d2, delta_c, delta_p, gamma, theta_c, theta_p, vega, rho_c, rho_p };
}

// Generate volatility surface
function generateVolSurface() {
  const strikes = [170, 175, 180, 185, 190, 195, 200, 205, 210];
  const expiries = ["7d", "14d", "30d", "60d", "90d"];
  const baseIV = 0.285;
  const surface: { strike: number; [key: string]: number }[] = [];
  strikes.forEach((K) => {
    const moneyness = (K - 190) / 190;
    const row: { strike: number; [key: string]: number } = { strike: K };
    expiries.forEach((exp, j) => {
      const termSlope = 1 - j * 0.02;
      const smile = 0.01 * moneyness * moneyness + Math.abs(moneyness) * 0.005;
      row[exp] = +(baseIV * termSlope + smile + (Math.random() - 0.5) * 0.005).toFixed(4);
    });
    surface.push(row);
  });
  return { strikes, expiries, surface };
}

// Delta across strikes
function generateDeltaCurve() {
  const spots = Array.from({ length: 40 }, (_, i) => 160 + i * 1.5);
  const K = 190, T = 30 / 365, r = 0.0458, sigma = 0.285;
  return spots.map((S) => {
    const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
    return { price: +S.toFixed(0), delta: +normCDF(d1).toFixed(4) };
  });
}

// ═══════════ HMM PANEL ═══════════

function HMMPanel() {
  const currentRegime: Regime = "TRANSITION";
  const history = useMemo(() => generateRegimeHistory(60), []);

  return (
    <div className="h-full flex flex-col gap-2 p-2 overflow-auto text-[10px] font-mono">
      {/* Current Regime Badge */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <span className="text-terminal-text-faint text-[9px] tracking-wider">CURRENT REGIME</span>
        <span
          className="px-2 py-0.5 rounded text-[10px] font-bold tracking-wider border"
          style={{
            color: REGIME_COLORS[currentRegime],
            borderColor: REGIME_COLORS[currentRegime] + "60",
            backgroundColor: REGIME_COLORS[currentRegime] + "18",
          }}
        >
          ● {currentRegime}
        </span>
      </div>

      {/* Regime Probability Bars */}
      <div className="flex-shrink-0">
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1.5">REGIME PROBABILITIES</div>
        <div className="flex flex-col gap-1.5">
          {REGIME_PROBS.map(({ regime, prob }) => (
            <div key={regime} className="flex items-center gap-2">
              <span className="w-[80px] text-[9px]" style={{ color: REGIME_COLORS[regime] }}>{regime}</span>
              <div className="flex-1 h-4 bg-terminal-bg rounded-sm overflow-hidden">
                <div
                  className="h-full rounded-sm transition-all"
                  style={{ width: `${prob}%`, backgroundColor: REGIME_COLORS[regime] + "80" }}
                />
              </div>
              <span className="w-[35px] text-right tabular-nums text-terminal-text-primary">{prob}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Transition Matrix */}
      <div className="flex-shrink-0">
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1.5">TRANSITION MATRIX</div>
        <table className="w-full text-[9px]">
          <thead>
            <tr className="text-terminal-text-faint">
              <th className="text-left font-normal py-0.5 pr-2">From \ To</th>
              <th className="text-right font-normal py-0.5 pr-1 text-terminal-positive">BULL</th>
              <th className="text-right font-normal py-0.5 pr-1 text-terminal-negative">BEAR</th>
              <th className="text-right font-normal py-0.5 text-terminal-warning">TRANS</th>
            </tr>
          </thead>
          <tbody>
            {TRANSITION_MATRIX.map((row) => (
              <tr key={row.from} className="border-t border-terminal-border/20">
                <td className="py-0.5 pr-2 text-terminal-text-muted">{row.from}</td>
                <td className="text-right py-0.5 pr-1 tabular-nums text-terminal-positive">{row.toBull.toFixed(2)}</td>
                <td className="text-right py-0.5 pr-1 tabular-nums text-terminal-negative">{row.toBear.toFixed(2)}</td>
                <td className="text-right py-0.5 tabular-nums text-terminal-warning">{row.toTrans.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 60-day regime history chart */}
      <div className="flex-shrink-0">
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1.5">60-DAY REGIME HISTORY</div>
        <div className="h-[40px] flex gap-[1px]">
          {history.map((d, i) => (
            <div
              key={i}
              className="flex-1 rounded-sm"
              style={{ backgroundColor: REGIME_COLORS[d.regime] + "70" }}
              title={`Day ${d.day}: ${d.regime}`}
            />
          ))}
        </div>
        <div className="flex justify-between mt-1 text-[7px] text-terminal-text-faint">
          <span>-60d</span><span>-30d</span><span>Today</span>
        </div>
      </div>

      {/* Regime Stats Table */}
      <div className="flex-shrink-0">
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1.5">REGIME STATISTICS</div>
        <table className="w-full text-[9px]">
          <thead>
            <tr className="text-terminal-text-faint border-b border-terminal-border/30">
              <th className="text-left font-normal py-0.5">Regime</th>
              <th className="text-right font-normal py-0.5">Avg Dur</th>
              <th className="text-right font-normal py-0.5">Avg Ret</th>
              <th className="text-right font-normal py-0.5">Vol</th>
              <th className="text-right font-normal py-0.5">Freq</th>
            </tr>
          </thead>
          <tbody>
            {REGIME_STATS.map((r) => (
              <tr key={r.regime} className="border-b border-terminal-border/10 hover:bg-white/[0.01]">
                <td className="py-0.5" style={{ color: REGIME_COLORS[r.regime as Regime] }}>{r.regime}</td>
                <td className="text-right py-0.5 tabular-nums text-terminal-text-muted">{r.avgDuration}</td>
                <td className={`text-right py-0.5 tabular-nums ${r.avgReturn.startsWith("+") ? "text-terminal-positive" : "text-terminal-negative"}`}>{r.avgReturn}</td>
                <td className="text-right py-0.5 tabular-nums text-terminal-text-muted">{r.volatility}</td>
                <td className="text-right py-0.5 tabular-nums text-terminal-text-muted">{r.frequency}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Model Metrics */}
      <div className="flex-shrink-0 border-t border-terminal-border/30 pt-2">
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1">MODEL METRICS</div>
        <div className="grid grid-cols-3 gap-2">
          {[["Log-Likelihood", "-1,847.3"], ["AIC", "3,718.6"], ["BIC", "3,784.2"]].map(([k, v]) => (
            <div key={k}>
              <div className="text-[7px] text-terminal-text-faint">{k}</div>
              <div className="text-[10px] text-terminal-text-primary tabular-nums font-semibold">{v}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ═══════════ BLACK-SCHOLES PANEL ═══════════

function BlackScholesPanel() {
  const [S, setS] = useState(189.45);
  const [K, setK] = useState(190);
  const [expDays, setExpDays] = useState(30);
  const [r, setR] = useState(0.0458);
  const [iv, setIV] = useState(0.285);

  const T = expDays / 365;
  const bs = useMemo(() => blackScholes(S, K, T, r, iv), [S, K, T, r, iv]);
  const { surface, strikes, expiries } = useMemo(generateVolSurface, []);
  const deltaCurve = useMemo(generateDeltaCurve, []);

  const parity = Math.abs(bs.call - bs.put - S + K * Math.exp(-r * T)) < 0.01;

  const greeks = [
    { name: "Delta (Call)", value: bs.delta_c?.toFixed(4) ?? "—", bar: bs.delta_c ?? 0, max: 1, color: "#00d4aa" },
    { name: "Delta (Put)", value: bs.delta_p?.toFixed(4) ?? "—", bar: Math.abs(bs.delta_p ?? 0), max: 1, color: "#f85149" },
    { name: "Gamma", value: bs.gamma?.toFixed(5) ?? "—", bar: (bs.gamma ?? 0) * 1000, max: 1, color: "#58a6ff" },
    { name: "Theta (C)", value: bs.theta_c?.toFixed(4) ?? "—", bar: Math.abs(bs.theta_c ?? 0) * 10, max: 1, color: "#d29922" },
    { name: "Vega", value: bs.vega?.toFixed(4) ?? "—", bar: (bs.vega ?? 0) * 5, max: 1, color: "#bc8cff" },
    { name: "Rho (C)", value: bs.rho_c?.toFixed(4) ?? "—", bar: (bs.rho_c ?? 0) * 20, max: 1, color: "#4ecdc4" },
  ];

  // IV heatmap color
  const ivColor = (v: number) => {
    if (v < 0.25) return "#3fb95066";
    if (v < 0.30) return "#d2992266";
    return "#f8514966";
  };

  const InputField = ({ label, value, onChange, step = 1 }: { label: string; value: number; onChange: (v: number) => void; step?: number }) => (
    <div className="flex items-center justify-between gap-2">
      <span className="text-terminal-text-faint text-[9px]">{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) => onChange(+e.target.value)}
        className="w-[70px] bg-terminal-bg border border-terminal-border/50 rounded px-1.5 py-0.5 text-[9px] text-terminal-text-primary outline-none focus:border-terminal-accent/50 text-right tabular-nums"
      />
    </div>
  );

  return (
    <div className="h-full flex flex-col gap-2 p-2 overflow-auto text-[10px] font-mono">
      {/* Inputs */}
      <div className="flex-shrink-0">
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1.5">PARAMETERS</div>
        <div className="flex flex-col gap-1">
          <InputField label="Underlying (S)" value={S} onChange={setS} step={0.01} />
          <InputField label="Strike (K)" value={K} onChange={setK} step={0.5} />
          <InputField label="Expiry (days)" value={expDays} onChange={setExpDays} />
          <InputField label="Risk-Free Rate" value={+(r * 100).toFixed(2)} onChange={(v) => setR(v / 100)} step={0.01} />
          <InputField label="IV (%)" value={+(iv * 100).toFixed(1)} onChange={(v) => setIV(v / 100)} step={0.1} />
        </div>
      </div>

      {/* Output cards */}
      <div className="flex gap-2 flex-shrink-0">
        {[
          { label: "Call Price", value: `$${bs.call.toFixed(4)}`, color: "text-terminal-positive" },
          { label: "Put Price", value: `$${bs.put.toFixed(4)}`, color: "text-terminal-negative" },
          { label: "Put-Call Parity", value: parity ? "✓ VALID" : "✗ FAIL", color: parity ? "text-terminal-positive" : "text-terminal-negative" },
        ].map(({ label, value, color }) => (
          <div key={label} className="flex-1 border border-terminal-border/40 rounded bg-terminal-surface p-1.5">
            <div className="text-[7px] text-terminal-text-faint">{label}</div>
            <div className={`text-xs font-bold tabular-nums ${color}`}>{value}</div>
          </div>
        ))}
      </div>

      {/* Greeks */}
      <div className="flex-shrink-0">
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1.5">GREEKS</div>
        <div className="flex flex-col gap-1.5">
          {greeks.map(({ name, value, bar, max, color }) => (
            <div key={name} className="flex items-center gap-2">
              <span className="w-[80px] text-[9px] text-terminal-text-muted">{name}</span>
              <span className="w-[65px] text-right tabular-nums text-terminal-text-primary">{value}</span>
              <div className="flex-1 h-3 bg-terminal-bg rounded-sm overflow-hidden">
                <div
                  className="h-full rounded-sm"
                  style={{ width: `${Math.min(100, Math.abs(bar / max) * 100)}%`, backgroundColor: color + "80" }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Volatility Surface Heatmap */}
      <div className="flex-shrink-0">
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1.5">VOLATILITY SURFACE</div>
        <table className="w-full text-[8px]">
          <thead>
            <tr>
              <th className="text-left font-normal text-terminal-text-faint py-0.5 pr-1">Strike</th>
              {expiries.map((e) => (
                <th key={e} className="text-right font-normal text-terminal-text-faint py-0.5 pr-1">{e}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {surface.map((row) => (
              <tr key={row.strike} className="border-t border-terminal-border/10">
                <td className={`py-0.5 pr-1 tabular-nums ${row.strike === K ? "text-terminal-accent font-semibold" : "text-terminal-text-muted"}`}>
                  {row.strike}
                </td>
                {expiries.map((e) => (
                  <td key={e} className="py-0.5 pr-1 text-right tabular-nums" style={{ color: "#e6edf3", backgroundColor: ivColor(row[e]) }}>
                    {(row[e] * 100).toFixed(1)}%
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Delta Curve */}
      <div className="flex-shrink-0">
        <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mb-1.5">DELTA SENSITIVITY (CALL)</div>
        <div className="h-[80px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={deltaCurve} margin={{ top: 2, right: 4, bottom: 12, left: 28 }}>
              <XAxis dataKey="price" tick={{ fontSize: 7, fill: "#484f58" }} axisLine={{ stroke: "#1e2633" }} tickLine={false} interval={9} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 7, fill: "#484f58" }} axisLine={false} tickLine={false} />
              <ReferenceLine x={K} stroke="#00d4aa" strokeWidth={0.8} strokeDasharray="3 2" />
              <ReferenceLine y={0.5} stroke="#484f58" strokeWidth={0.5} strokeDasharray="2 2" />
              <Line type="monotone" dataKey="delta" stroke="#00d4aa" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ═══════════ MAIN SIMULATIONS PAGE ═══════════

export default function Simulations() {
  // ─── Engine API — regime and vol surface data ───────
  const { data: regimeApi } = useEngineQuery<{ regimes: Array<Record<string, unknown>> }>("/ml/regime-history", { refetchInterval: 30000 });
  const { data: volApi } = useEngineQuery<{ grid: Array<{ strike: number; expiry: number; iv: number }> }>("/ml/vol-surface", { refetchInterval: 60000 });
  const { data: cubeApi } = useEngineQuery<{ regime: string; target_beta: number; max_leverage: number }>("/cube/state", { refetchInterval: 15000 });

  return (
    <div className="h-full flex flex-col p-[2px] overflow-hidden" data-testid="simulations">
      <ResizableDashboard
        defaultSizes={[50, 50]}
        minSizes={[30, 30]}
        className="gap-0 h-full"
      >
        {/* Left: HMM */}
        <div className="h-full p-[1px]">
          <DashboardPanel
            title="HIDDEN MARKOV MODEL — REGIME DETECTION"
            className="h-full"
            headerRight={
              <div className="flex items-center gap-1.5">
                <span className="text-[8px] font-mono text-terminal-text-faint">n_states=3</span>
                <span className="w-1.5 h-1.5 rounded-full bg-terminal-positive animate-pulse" />
                <span className="text-[8px] text-terminal-positive font-mono">FITTED</span>
              </div>
            }
            noPadding
          >
            <HMMPanel />
          </DashboardPanel>
        </div>

        {/* Right: Black-Scholes */}
        <div className="h-full p-[1px]">
          <DashboardPanel
            title="BLACK-SCHOLES OPTIONS PRICING"
            className="h-full"
            headerRight={
              <div className="flex items-center gap-1.5">
                <span className="text-[8px] font-mono text-terminal-text-faint">European</span>
                <span className="w-1.5 h-1.5 rounded-full bg-terminal-accent" />
                <span className="text-[8px] text-terminal-accent font-mono">GBM</span>
              </div>
            }
            noPadding
          >
            <BlackScholesPanel />
          </DashboardPanel>
        </div>
      </ResizableDashboard>
    </div>
  );
}
