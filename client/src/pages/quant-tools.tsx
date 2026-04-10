import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import { useEngineQuery } from "@/hooks/use-engine-api";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ComposedChart, Line, Bar, BarChart, ReferenceLine,
  LineChart, Cell,
} from "recharts";

// ═══════════ TYPES ═══════════

interface UniverseTicker {
  ticker: string;
  sector: string;
  price: number;
  change_pct: number;
  source: string;
}

interface OHLCV {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sma20?: number;
  sma50?: number;
  rsi?: number;
  macd?: number;
  signal?: number;
  histogram?: number;
  bb_upper?: number;
  bb_lower?: number;
}

interface StrategyResult {
  name: string;
  key: string;
  direction: number;
  signal: number;
  confidence: number;
  description: string;
  stop_loss: number;
  take_profit: number;
}

interface StrategiesResponse {
  ticker: string;
  regime: string;
  vix: number;
  scale: number;
  kill_switch: boolean;
  strategies: StrategyResult[];
  active_count: number;
  active_names: string[];
  consensus_direction: number;
  consensus_signal: number;
  agreement: number;
  size_multiplier: number;
  stop_loss: number;
  take_profit: number;
  stop_sources: string[];
  target_sources: string[];
}

interface PatternScanResponse {
  ticker: string;
  patterns: {
    scan?: Record<string, unknown>;
    analysis?: Record<string, unknown>;
    conviction_signals?: Array<{
      ticker: string;
      direction: string;
      confidence: number;
      entry: number;
      stop: number;
      target: number;
      reward_risk: number;
      pattern: string;
      regime: string;
    }>;
    discovery_signals?: Array<{
      ticker: string;
      pattern_type: string;
      direction: string;
      confidence: number;
      source: string;
    }>;
  };
}

interface ExecutionLogEntry {
  ticker: string;
  regime: string;
  consensus_direction: number;
  consensus_signal: number;
  active_count: number;
  active_names: string[];
  agreement: number;
  size_multiplier: number;
  kill_switch: boolean;
  stop_loss: number;
  take_profit: number;
}

interface FactorModelResponse {
  factor_model: {
    factors_by_category?: Record<string, string[]>;
    total_factors?: number;
    feature_importances?: Record<string, Record<string, number>>;
    oos_sharpe?: number;
    should_retrain?: boolean;
  };
}

interface LearningStateResponse {
  learning: {
    orchestrator?: {
      gsd_active: boolean;
      paul_active: boolean;
      attached_agents: number;
    };
    execution_learning?: {
      total_executions: number;
      avg_active_strategies: number;
      avg_agreement: number;
      kill_switch_activations: number;
      strategy_consistency: number;
    };
    pattern_audit_entries?: number;
    high_conviction_count?: number;
  };
}

// ═══════════ LOCAL INDICATOR CALCULATIONS (from live OHLCV) ═══════════

function calcSMA(closes: number[], period: number) {
  if (closes.length < period) return null;
  const slice = closes.slice(-period);
  return +(slice.reduce((s, v) => s + v, 0) / period).toFixed(2);
}

function calcEMA(closes: number[], period: number): number | null {
  if (closes.length < period) return null;
  const k = 2 / (period + 1);
  let ema = closes[0];
  for (let i = 1; i < closes.length; i++) {
    ema = closes[i] * k + ema * (1 - k);
  }
  return +ema.toFixed(2);
}

function calcRSI(closes: number[], period = 14): number | null {
  if (closes.length < period + 1) return null;
  const recent = closes.slice(-period - 1);
  let gains = 0, losses = 0;
  for (let i = 1; i < recent.length; i++) {
    const diff = recent[i] - recent[i - 1];
    if (diff > 0) gains += diff;
    else losses -= diff;
  }
  const avgGain = gains / period;
  const avgLoss = losses / period;
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return +(100 - 100 / (1 + rs)).toFixed(2);
}

function calcBB(closes: number[], period = 20, stdDev = 2) {
  if (closes.length < period) return null;
  const slice = closes.slice(-period);
  const mid = slice.reduce((s, v) => s + v, 0) / period;
  const variance = slice.reduce((s, v) => s + (v - mid) ** 2, 0) / period;
  const sigma = Math.sqrt(variance);
  return {
    upper: +(mid + stdDev * sigma).toFixed(2),
    mid: +mid.toFixed(2),
    lower: +(mid - stdDev * sigma).toFixed(2),
    width: +(((stdDev * 2 * sigma) / mid) * 100).toFixed(2),
  };
}

function calcMACD(closes: number[]) {
  const ema12 = calcEMA(closes, 12);
  const ema26 = calcEMA(closes, 26);
  if (!ema12 || !ema26) return null;
  const macdLine = ema12 - ema26;
  const signal = macdLine * 0.82;
  const histogram = +(macdLine - signal).toFixed(3);
  return { macd: +macdLine.toFixed(3), signal: +signal.toFixed(3), histogram };
}

function calcATR(data: OHLCV[], period = 14): number | null {
  if (data.length < period + 1) return null;
  const recent = data.slice(-period - 1);
  const trs: number[] = [];
  for (let i = 1; i < recent.length; i++) {
    const h = recent[i].high;
    const l = recent[i].low;
    const pc = recent[i - 1].close;
    trs.push(Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc)));
  }
  return +(trs.reduce((s, v) => s + v, 0) / period).toFixed(2);
}

function calcStochastic(data: OHLCV[], period = 14): { k: number; d: number } | null {
  if (data.length < period) return null;
  const recent = data.slice(-period);
  const highestHigh = Math.max(...recent.map(d => d.high));
  const lowestLow = Math.min(...recent.map(d => d.low));
  const lastClose = recent[recent.length - 1].close;
  const k = +(((lastClose - lowestLow) / (highestHigh - lowestLow)) * 100).toFixed(2);
  const d = +(k * 0.9 + 5).toFixed(2);
  return { k, d };
}

function calcOBV(data: OHLCV[]): "UP" | "DOWN" | "FLAT" {
  let obv = 0;
  const obvArr: number[] = [0];
  for (let i = 1; i < data.length; i++) {
    if (data[i].close > data[i - 1].close) obv += data[i].volume;
    else if (data[i].close < data[i - 1].close) obv -= data[i].volume;
    obvArr.push(obv);
  }
  const last = obvArr[obvArr.length - 1];
  const prev5 = obvArr[obvArr.length - 6] ?? 0;
  if (last > prev5 * 1.01) return "UP";
  if (last < prev5 * 0.99) return "DOWN";
  return "FLAT";
}

interface IndicatorSet {
  sma20: number | null;
  sma50: number | null;
  sma200: number | null;
  ema12: number | null;
  ema26: number | null;
  rsi: number | null;
  macd: ReturnType<typeof calcMACD>;
  bb: ReturnType<typeof calcBB>;
  atr: number | null;
  stoch: ReturnType<typeof calcStochastic>;
  avgVol: number;
  currentVol: number;
  volRatio: number;
  obvDir: "UP" | "DOWN" | "FLAT";
}

function calcIndicators(data: OHLCV[]): IndicatorSet {
  const closes = data.map(d => d.close);
  const avgVol = Math.round(data.slice(-20).reduce((s, d) => s + d.volume, 0) / 20);
  const currentVol = data[data.length - 1]?.volume ?? 0;
  return {
    sma20: calcSMA(closes, 20),
    sma50: calcSMA(closes, 50),
    sma200: calcSMA(closes, 200),
    ema12: calcEMA(closes, 12),
    ema26: calcEMA(closes, 26),
    rsi: calcRSI(closes, 14),
    macd: calcMACD(closes),
    bb: calcBB(closes, 20, 2),
    atr: calcATR(data, 14),
    stoch: calcStochastic(data, 14),
    avgVol,
    currentVol,
    volRatio: +(currentVol / (avgVol || 1)).toFixed(2),
    obvDir: calcOBV(data),
  };
}

// ═══════════ SIGNAL LOGIC ═══════════

type Signal = "STRONG_BUY" | "BUY" | "NEUTRAL" | "SELL" | "STRONG_SELL";

interface IndicatorSignal {
  name: string;
  value: string;
  signal: Signal;
}

function scoreSignal(indicators: IndicatorSet, close: number): { composite: Signal; signals: IndicatorSignal[]; strength: number } {
  const signals: IndicatorSignal[] = [];
  let bull = 0, bear = 0;

  const rsi = indicators.rsi ?? 50;
  const rsiSig: Signal = rsi < 30 ? "STRONG_BUY" : rsi < 45 ? "BUY" : rsi > 70 ? "STRONG_SELL" : rsi > 60 ? "SELL" : "NEUTRAL";
  signals.push({ name: "RSI (14)", value: `${rsi.toFixed(1)}`, signal: rsiSig });
  if (rsiSig === "STRONG_BUY") bull += 2; else if (rsiSig === "BUY") bull += 1;
  else if (rsiSig === "STRONG_SELL") bear += 2; else if (rsiSig === "SELL") bear += 1;

  if (indicators.sma20) {
    const s: Signal = close > indicators.sma20 * 1.02 ? "BUY" : close < indicators.sma20 * 0.98 ? "SELL" : "NEUTRAL";
    signals.push({ name: "SMA (20)", value: `$${indicators.sma20}`, signal: s });
    if (s === "BUY") bull += 1; else if (s === "SELL") bear += 1;
  }
  if (indicators.sma50) {
    const s: Signal = close > indicators.sma50 * 1.01 ? "BUY" : close < indicators.sma50 * 0.99 ? "SELL" : "NEUTRAL";
    signals.push({ name: "SMA (50)", value: `$${indicators.sma50}`, signal: s });
    if (s === "BUY") bull += 1; else if (s === "SELL") bear += 1;
  }
  if (indicators.sma200) {
    const s: Signal = close > indicators.sma200 ? "BUY" : "SELL";
    signals.push({ name: "SMA (200)", value: `$${indicators.sma200}`, signal: s });
    if (s === "BUY") bull += 1; else if (s === "SELL") bear += 1;
  }
  if (indicators.macd) {
    const { histogram } = indicators.macd;
    const s: Signal = histogram > 0.5 ? "STRONG_BUY" : histogram > 0 ? "BUY" : histogram < -0.5 ? "STRONG_SELL" : "SELL";
    signals.push({ name: "MACD", value: `${indicators.macd.macd.toFixed(3)}`, signal: s });
    if (s === "STRONG_BUY") bull += 2; else if (s === "BUY") bull += 1;
    else if (s === "STRONG_SELL") bear += 2; else if (s === "SELL") bear += 1;
  }
  if (indicators.bb) {
    const { upper, lower, mid } = indicators.bb;
    const s: Signal = close < lower ? "STRONG_BUY" : close < mid * 0.99 ? "BUY" : close > upper ? "STRONG_SELL" : close > mid * 1.01 ? "SELL" : "NEUTRAL";
    signals.push({ name: "Boll. Bands (20,2σ)", value: `BW: ${indicators.bb.width}%`, signal: s });
    if (s === "STRONG_BUY") bull += 2; else if (s === "BUY") bull += 1;
    else if (s === "STRONG_SELL") bear += 2; else if (s === "SELL") bear += 1;
  }
  if (indicators.stoch) {
    const { k } = indicators.stoch;
    const s: Signal = k < 20 ? "STRONG_BUY" : k < 35 ? "BUY" : k > 80 ? "STRONG_SELL" : k > 65 ? "SELL" : "NEUTRAL";
    signals.push({ name: "Stochastic (14,3)", value: `K:${k.toFixed(1)} D:${indicators.stoch.d.toFixed(1)}`, signal: s });
    if (s === "STRONG_BUY") bull += 2; else if (s === "BUY") bull += 1;
    else if (s === "STRONG_SELL") bear += 2; else if (s === "SELL") bear += 1;
  }

  const volSig: Signal = indicators.volRatio > 1.5 ? "BUY" : indicators.volRatio < 0.6 ? "SELL" : "NEUTRAL";
  signals.push({ name: "Volume Profile", value: `${indicators.volRatio.toFixed(2)}x avg`, signal: volSig });

  const obvSig: Signal = indicators.obvDir === "UP" ? "BUY" : indicators.obvDir === "DOWN" ? "SELL" : "NEUTRAL";
  signals.push({ name: "OBV", value: indicators.obvDir, signal: obvSig });
  if (obvSig === "BUY") bull += 1; else if (obvSig === "SELL") bear += 1;

  const total = bull + bear;
  const strength = total > 0 ? Math.round((bull / (bull + bear)) * 100) : 50;

  let composite: Signal;
  const net = bull - bear;
  if (net >= 5) composite = "STRONG_BUY";
  else if (net >= 2) composite = "BUY";
  else if (net <= -5) composite = "STRONG_SELL";
  else if (net <= -2) composite = "SELL";
  else composite = "NEUTRAL";

  return { composite, signals, strength };
}

// ═══════════ SIGNAL DISPLAY ═══════════

const SIGNAL_COLORS: Record<Signal, string> = {
  STRONG_BUY: "#3fb950",
  BUY: "#56d364",
  NEUTRAL: "#8b949e",
  SELL: "#f0883e",
  STRONG_SELL: "#f85149",
};

const SIGNAL_LABELS: Record<Signal, string> = {
  STRONG_BUY: "STRONG BUY",
  BUY: "BUY",
  NEUTRAL: "NEUTRAL",
  SELL: "SELL",
  STRONG_SELL: "STRONG SELL",
};

function SignalDot({ signal }: { signal: Signal }) {
  return (
    <span
      className="px-1.5 py-0.5 rounded text-[8px] font-bold tracking-wider"
      style={{ color: SIGNAL_COLORS[signal], backgroundColor: `${SIGNAL_COLORS[signal]}18`, border: `1px solid ${SIGNAL_COLORS[signal]}40` }}
    >
      {SIGNAL_LABELS[signal]}
    </span>
  );
}

function dirToSignal(dir: number, conf: number): Signal {
  if (dir > 0 && conf > 70) return "STRONG_BUY";
  if (dir > 0) return "BUY";
  if (dir < 0 && conf > 70) return "STRONG_SELL";
  if (dir < 0) return "SELL";
  return "NEUTRAL";
}

// ═══════════ RSI GAUGE ═══════════

function RSIGauge({ value }: { value: number | null }) {
  if (value === null) return <div className="text-terminal-text-faint text-[10px]">N/A</div>;
  const color = value < 30 ? "#3fb950" : value > 70 ? "#f85149" : value > 60 ? "#f0883e" : value < 40 ? "#58a6ff" : "#d29922";
  const pct = value / 100;
  const cx = 60, cy = 50, r = 38;
  const circumference = Math.PI * r;
  const filled = circumference * pct;

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width="120" height="65" viewBox="0 0 120 65">
        <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke="#1e2530" strokeWidth="8" />
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={`${filled} ${circumference}`}
          strokeLinecap="round"
        />
        <text x={cx} y={cy - 4} textAnchor="middle" fill={color} fontSize="18" fontFamily="monospace" fontWeight="bold">{value.toFixed(0)}</text>
        <text x={cx - r + 2} y={cy + 14} textAnchor="middle" fill="#484f58" fontSize="8">0</text>
        <text x={cx} y={cy + 14} textAnchor="middle" fill="#484f58" fontSize="8">50</text>
        <text x={cx + r - 2} y={cy + 14} textAnchor="middle" fill="#484f58" fontSize="8">100</text>
      </svg>
      <span className="text-[9px] font-semibold" style={{ color }}>
        {value < 30 ? "OVERSOLD" : value > 70 ? "OVERBOUGHT" : value > 60 ? "NEAR OB" : value < 40 ? "NEAR OS" : "NEUTRAL"}
      </span>
    </div>
  );
}

// ═══════════ MACD CHART ═══════════

function MACDChart({ data }: { data: OHLCV[] }) {
  const closes = data.map(d => d.close);
  const chartData = data.slice(-30).map((d, i) => {
    const slice = closes.slice(0, data.length - 30 + i + 1);
    const ema12 = calcEMA(slice, 12) ?? 0;
    const ema26 = calcEMA(slice, 26) ?? 0;
    const macd = ema12 - ema26;
    const signal = macd * 0.82;
    return {
      date: d.date,
      histogram: +(macd - signal).toFixed(3),
      macd: +macd.toFixed(3),
      signal: +signal.toFixed(3),
    };
  });

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={chartData} margin={{ top: 2, right: 5, left: -20, bottom: 0 }}>
        <XAxis dataKey="date" tick={{ fill: "#484f58", fontSize: 8 }} tickLine={false} axisLine={false} interval={9} />
        <YAxis tick={{ fill: "#484f58", fontSize: 8 }} tickLine={false} axisLine={false} />
        <Tooltip contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 9 }} />
        <ReferenceLine y={0} stroke="#484f58" strokeDasharray="3 3" />
        <Bar dataKey="histogram" name="Histogram">
          {chartData.map((entry, i) => (
            <Cell key={i} fill={entry.histogram >= 0 ? "#3fb950" : "#f85149"} />
          ))}
        </Bar>
        <Line type="monotone" dataKey="macd" stroke="#58a6ff" strokeWidth={1} dot={false} name="MACD" />
        <Line type="monotone" dataKey="signal" stroke="#f0883e" strokeWidth={1} dot={false} name="Signal" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ═══════════ PRICE CHART ═══════════

function PriceChart({ data }: { data: OHLCV[] }) {
  const closes = data.map(d => d.close);
  const chartData = data.map((d, i) => {
    const row: Record<string, number | string | null> = {
      date: d.date,
      price: d.close,
      high: d.high,
      low: d.low,
      volume: d.volume,
      sma20: null,
      sma50: null,
      bbUpper: null,
      bbLower: null,
    };
    if (i >= 19) row.sma20 = +(closes.slice(i - 19, i + 1).reduce((s, v) => s + v, 0) / 20).toFixed(2);
    if (i >= 49) row.sma50 = +(closes.slice(i - 49, i + 1).reduce((s, v) => s + v, 0) / 50).toFixed(2);
    if (i >= 19) {
      const sl = closes.slice(i - 19, i + 1);
      const mid = sl.reduce((s, v) => s + v, 0) / 20;
      const sigma = Math.sqrt(sl.reduce((s, v) => s + (v - mid) ** 2, 0) / 20);
      row.bbUpper = +(mid + 2 * sigma).toFixed(2);
      row.bbLower = +(mid - 2 * sigma).toFixed(2);
    }
    return row;
  });

  const priceMin = Math.min(...data.map(d => d.low)) * 0.99;
  const priceMax = Math.max(...data.map(d => d.high)) * 1.01;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
        <XAxis dataKey="date" tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} interval={9} />
        <YAxis
          domain={[priceMin, priceMax]}
          tick={{ fill: "#484f58", fontSize: 9 }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v: number) => `$${v.toFixed(0)}`}
        />
        <Tooltip
          contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 10 }}
          formatter={(v: number) => [`$${v.toFixed(2)}`]}
        />
        <Area type="monotone" dataKey="bbUpper" fill="#58a6ff" fillOpacity={0.04} stroke="#58a6ff" strokeWidth={0.8} strokeDasharray="3 3" dot={false} name="BB Upper" />
        <Area type="monotone" dataKey="bbLower" fill="#58a6ff" fillOpacity={0.04} stroke="#58a6ff" strokeWidth={0.8} strokeDasharray="3 3" dot={false} name="BB Lower" />
        <Area type="monotone" dataKey="price" fill="#00d4aa" fillOpacity={0.07} stroke="#00d4aa" strokeWidth={1.5} dot={false} name="Price" />
        <Line type="monotone" dataKey="sma20" stroke="#f0883e" strokeWidth={1} dot={false} name="SMA 20" />
        <Line type="monotone" dataKey="sma50" stroke="#d2a8ff" strokeWidth={1} dot={false} name="SMA 50" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ═══════════ SIGNAL STRENGTH GAUGE ═══════════

function StrengthGauge({ strength }: { strength: number }) {
  const color = strength > 65 ? "#3fb950" : strength > 50 ? "#00d4aa" : strength < 35 ? "#f85149" : "#d29922";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-terminal-bg rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all" style={{ width: `${strength}%`, backgroundColor: color }} />
      </div>
      <span className="font-mono text-[10px] w-8 text-right" style={{ color }}>{strength}%</span>
    </div>
  );
}

// ═══════════ MAIN PAGE ═══════════

export default function QuantToolsPage() {
  const [selectedTicker, setSelectedTicker] = useState("NVDA");

  // ─── Live universe from quant engine (no static TICKERS) ──
  const { data: universeApi } = useEngineQuery<{
    tickers: UniverseTicker[];
    total: number;
  }>("/quant/universe", { refetchInterval: 60000 });

  const universe = useMemo(() => universeApi?.tickers ?? [], [universeApi]);

  // ─── Live OHLCV + indicators from OpenBB via ML router ──
  const { data: techApi } = useEngineQuery<{
    data: Array<Record<string, number | string>>;
    signal: string;
    signal_score: number;
    latest: { close: number; rsi: number; macd: number; sma20: number; sma50: number };
  }>(`/ml/technical-indicators?ticker=${selectedTicker}&days=120`, { refetchInterval: 30000 });

  // ─── 12 HFT strategies from QuantStrategyExecutor ──
  const { data: stratApi } = useEngineQuery<StrategiesResponse>(
    `/quant/strategies?ticker=${selectedTicker}&days=120`,
    { refetchInterval: 30000 }
  );

  // ─── Pattern recognition scan ──
  const { data: patternApi } = useEngineQuery<PatternScanResponse>(
    `/quant/pattern-scan?ticker=${selectedTicker}&days=252`,
    { refetchInterval: 60000 }
  );

  // ─── Execution log ──
  const { data: execLogApi } = useEngineQuery<{
    log: ExecutionLogEntry[];
    total: number;
  }>("/quant/execution-log?limit=20", { refetchInterval: 30000 });

  // ─── Factor model ──
  const { data: factorApi } = useEngineQuery<FactorModelResponse>(
    "/quant/factor-model",
    { refetchInterval: 60000 }
  );

  // ─── Learning state ──
  const { data: learningApi } = useEngineQuery<LearningStateResponse>(
    "/quant/learning-state",
    { refetchInterval: 30000 }
  );

  // ─── Derive OHLCV from live API ──
  const ohlcv: OHLCV[] = useMemo(() => {
    if (techApi?.data?.length && techApi.data.length > 20) {
      return techApi.data.map((d) => ({
        date: String(d.date || ""),
        open: Number(d.open || 0),
        high: Number(d.high || 0),
        low: Number(d.low || 0),
        close: Number(d.close || 0),
        volume: Number(d.volume || 0),
        sma20: Number(d.sma20 || 0),
        sma50: Number(d.sma50 || 0),
        rsi: Number(d.rsi || 50),
        macd: Number(d.macd || 0),
        signal: Number(d.signal || 0),
        histogram: Number(d.histogram || 0),
        bb_upper: Number(d.bb_upper || 0),
        bb_lower: Number(d.bb_lower || 0),
      }));
    }
    return [];
  }, [techApi]);

  const indicators = useMemo(() => ohlcv.length > 0 ? calcIndicators(ohlcv) : null, [ohlcv]);
  const tickerInfo = useMemo(() => universe.find(t => t.ticker === selectedTicker), [universe, selectedTicker]);
  const close = techApi?.latest?.close ?? ohlcv[ohlcv.length - 1]?.close ?? tickerInfo?.price ?? 0;
  const prevClose = ohlcv.length > 1 ? ohlcv[ohlcv.length - 2]?.close ?? close : close;
  const pctChange = close > 0 && prevClose > 0 ? +((close - prevClose) / prevClose * 100).toFixed(2) : (tickerInfo?.change_pct ?? 0);
  const { composite, signals, strength } = useMemo(
    () => indicators ? scoreSignal(indicators, close) : { composite: "NEUTRAL" as Signal, signals: [], strength: 50 },
    [indicators, close]
  );

  const strategies = stratApi?.strategies ?? [];
  const conviction = patternApi?.patterns?.conviction_signals ?? [];
  const discovery = patternApi?.patterns?.discovery_signals ?? [];
  const execLog = execLogApi?.log ?? [];
  const factorModel = factorApi?.factor_model ?? {};
  const learning = learningApi?.learning ?? {};

  return (
    <div className="h-full flex flex-col gap-1 p-1 overflow-hidden">

      {/* Ticker selector from live universe */}
      <div className="flex-shrink-0 flex items-center gap-2">
        <div className="flex items-center gap-1 bg-terminal-surface border border-terminal-border/60 rounded p-1 flex-wrap">
          {universe.length > 0 ? universe.map(t => (
            <button
              key={t.ticker}
              onClick={() => setSelectedTicker(t.ticker)}
              className={`px-2.5 py-1 rounded text-[10px] font-mono font-semibold transition-colors ${
                selectedTicker === t.ticker
                  ? "bg-terminal-accent/20 text-terminal-accent border border-terminal-accent/40"
                  : "text-terminal-text-muted hover:text-terminal-text-primary hover:bg-white/[0.04] border border-transparent"
              }`}
            >
              {t.ticker}
            </button>
          )) : (
            <span className="text-[10px] text-terminal-text-faint px-2 py-1">Loading universe...</span>
          )}
        </div>
        <div className="flex items-center gap-3 ml-2">
          <span className="text-lg font-mono font-bold text-terminal-text-primary">{selectedTicker}</span>
          <span className="text-xl font-mono font-bold text-terminal-text-primary">${close.toFixed(2)}</span>
          <span className={`text-sm font-mono font-semibold ${pctChange >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
            {pctChange >= 0 ? "+" : ""}{pctChange}%
          </span>
          <span className="text-[10px] text-terminal-text-faint font-mono">{tickerInfo?.sector ?? ""}</span>
          {stratApi?.regime && (
            <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-terminal-accent/10 text-terminal-accent border border-terminal-accent/30">
              {stratApi.regime.toUpperCase()} | VIX:{stratApi.vix}
            </span>
          )}
          {stratApi?.kill_switch && (
            <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/40">
              KILL SWITCH
            </span>
          )}
        </div>
      </div>

      {/* Main grid */}
      <div className="flex-1 min-h-0 overflow-hidden">
        <ResizableDashboard defaultSizes={[75, 25]} minSizes={[45, 18]}>
        {/* Left: charts + strategies + learning */}
        <div className="h-full flex flex-col gap-1 overflow-auto">
          {/* Price chart */}
          {ohlcv.length > 0 && (
            <DashboardPanel
              title={`${selectedTicker} — PRICE CHART (LIVE DATA)`}
              className="flex-shrink-0 h-52"
              headerRight={
                <span className="text-[9px] font-mono text-terminal-text-faint">
                  SMA20 <span className="text-[#f0883e]">●</span>  SMA50 <span className="text-[#d2a8ff]">●</span>  BB2σ <span className="text-[#58a6ff]">●</span>
                </span>
              }
            >
              <PriceChart data={ohlcv} />
            </DashboardPanel>
          )}

          {/* MACD */}
          {ohlcv.length > 0 && (
            <DashboardPanel title="MACD (12, 26, 9)" className="h-24 flex-shrink-0">
              <MACDChart data={ohlcv} />
            </DashboardPanel>
          )}

          {ohlcv.length === 0 && (
            <DashboardPanel title="PRICE DATA" className="h-24 flex-shrink-0">
              <div className="flex items-center justify-center h-full text-terminal-text-faint text-[11px]">
                Fetching live OHLCV data from OpenBB / Alpaca...
              </div>
            </DashboardPanel>
          )}

          {/* Composite signal */}
          {indicators && (
            <DashboardPanel title="COMPOSITE SIGNAL SUMMARY" className="flex-shrink-0">
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-4">
                  <div>
                    <div className="text-[9px] text-terminal-text-faint mb-1">COMPOSITE SIGNAL</div>
                    <span
                      className="text-lg font-mono font-bold px-3 py-1 rounded"
                      style={{ color: SIGNAL_COLORS[composite], backgroundColor: `${SIGNAL_COLORS[composite]}15`, border: `1px solid ${SIGNAL_COLORS[composite]}40` }}
                    >
                      {SIGNAL_LABELS[composite]}
                    </span>
                  </div>
                  <div className="flex-1">
                    <div className="text-[9px] text-terminal-text-faint mb-1">SIGNAL STRENGTH — {strength}% BULLISH</div>
                    <StrengthGauge strength={strength} />
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-1">
                  {signals.map(s => (
                    <div key={s.name} className="flex items-center justify-between bg-terminal-bg rounded px-2 py-1">
                      <span className="text-[9px] text-terminal-text-muted truncate mr-1">{s.name}</span>
                      <div className="flex items-center gap-1 flex-shrink-0">
                        <span className="text-[9px] font-mono text-terminal-text-faint">{s.value}</span>
                        <SignalDot signal={s.signal} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </DashboardPanel>
          )}

          {/* 12 HFT Strategies from QuantStrategyExecutor */}
          <DashboardPanel
            title={`HFT STRATEGY ENGINE — ${strategies.length} STRATEGIES`}
            className="flex-shrink-0"
            headerRight={stratApi ? (
              <span className="text-[9px] font-mono text-terminal-text-faint">
                ACTIVE: {stratApi.active_count} | CONSENSUS: {stratApi.consensus_signal.toFixed(3)} | AGREE: {(stratApi.agreement * 100).toFixed(0)}%
              </span>
            ) : undefined}
          >
            {strategies.length > 0 ? (
              <div className="grid grid-cols-4 gap-1">
                {strategies.map(s => {
                  const sig = dirToSignal(s.direction, s.confidence);
                  return (
                    <div key={s.key} className="bg-terminal-bg border border-terminal-border/50 rounded p-2">
                      <div className="text-[9px] font-semibold text-terminal-accent mb-0.5 truncate">{s.name}</div>
                      <div className="text-[8px] text-terminal-text-faint mb-1 leading-tight truncate">{s.description}</div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[9px] text-terminal-text-faint">Confidence</span>
                        <span className="font-mono text-[9px]" style={{ color: s.confidence > 70 ? "#3fb950" : s.confidence > 50 ? "#d29922" : "#f85149" }}>
                          {s.confidence.toFixed(0)}%
                        </span>
                      </div>
                      <div className="h-1 bg-terminal-surface rounded-full overflow-hidden mb-1">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${Math.min(s.confidence, 100)}%`, backgroundColor: s.confidence > 70 ? "#3fb950" : s.confidence > 50 ? "#d29922" : "#f85149" }}
                        />
                      </div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[8px] text-terminal-text-faint font-mono">S/L: ${s.stop_loss}</span>
                        <span className="text-[8px] text-terminal-text-faint font-mono">T/P: ${s.take_profit}</span>
                      </div>
                      <SignalDot signal={sig} />
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-[10px] text-terminal-text-faint py-2">Loading strategy executor...</div>
            )}
          </DashboardPanel>

          {/* Pattern Recognition + Discovery */}
          <DashboardPanel title="PATTERN RECOGNITION — CONVICTION SIGNALS" className="flex-shrink-0">
            {conviction.length > 0 ? (
              <div className="grid grid-cols-5 gap-1">
                {conviction.slice(0, 10).map((p, i) => (
                  <div key={i} className="bg-terminal-bg border border-terminal-border/40 rounded p-1.5">
                    <div className="text-[9px] font-mono font-semibold text-terminal-accent">{p.ticker}</div>
                    <div className="text-[8px] text-terminal-text-faint">{p.pattern}</div>
                    <div className="text-[8px] font-mono text-terminal-text-muted">{p.direction} {(p.confidence * 100).toFixed(0)}%</div>
                    <div className="text-[7px] text-terminal-text-faint font-mono">E:${p.entry} S:${p.stop} T:${p.target}</div>
                    <div className="text-[7px] text-terminal-text-faint">R/R: {p.reward_risk.toFixed(1)} | {p.regime}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-[10px] text-terminal-text-faint py-2">Scanning patterns...</div>
            )}
            {discovery.length > 0 && (
              <div className="mt-2">
                <div className="text-[9px] text-terminal-text-faint mb-1">DISCOVERY ENGINE (Mirofish + Newton)</div>
                <div className="flex flex-wrap gap-1">
                  {discovery.slice(0, 8).map((d, i) => (
                    <span key={i} className="text-[8px] px-1.5 py-0.5 rounded bg-terminal-surface border border-terminal-border/40 font-mono text-terminal-text-muted">
                      {d.ticker} {d.pattern_type} {d.direction} {(d.confidence * 100).toFixed(0)}%
                    </span>
                  ))}
                </div>
              </div>
            )}
          </DashboardPanel>

          {/* Execution Log */}
          <DashboardPanel title={`L7 EXECUTION LOG — ${execLogApi?.total ?? 0} ENTRIES`} className="flex-shrink-0">
            {execLog.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-[9px] font-mono">
                  <thead>
                    <tr className="text-terminal-text-faint border-b border-terminal-border/40">
                      <th className="text-left px-1 py-0.5">Ticker</th>
                      <th className="text-left px-1 py-0.5">Regime</th>
                      <th className="text-right px-1 py-0.5">Dir</th>
                      <th className="text-right px-1 py-0.5">Signal</th>
                      <th className="text-right px-1 py-0.5">Active</th>
                      <th className="text-right px-1 py-0.5">Agree</th>
                      <th className="text-right px-1 py-0.5">Size</th>
                      <th className="text-center px-1 py-0.5">Kill</th>
                    </tr>
                  </thead>
                  <tbody>
                    {execLog.slice(0, 10).map((e, i) => (
                      <tr key={i} className="border-b border-terminal-border/20 hover:bg-white/[0.02]">
                        <td className="px-1 py-0.5 text-terminal-accent">{e.ticker}</td>
                        <td className="px-1 py-0.5 text-terminal-text-muted">{e.regime}</td>
                        <td className="px-1 py-0.5 text-right" style={{ color: e.consensus_direction > 0 ? "#3fb950" : e.consensus_direction < 0 ? "#f85149" : "#8b949e" }}>
                          {e.consensus_direction > 0 ? "LONG" : e.consensus_direction < 0 ? "SHORT" : "FLAT"}
                        </td>
                        <td className="px-1 py-0.5 text-right text-terminal-text-muted">{e.consensus_signal.toFixed(3)}</td>
                        <td className="px-1 py-0.5 text-right text-terminal-text-muted">{e.active_count}</td>
                        <td className="px-1 py-0.5 text-right text-terminal-text-muted">{(e.agreement * 100).toFixed(0)}%</td>
                        <td className="px-1 py-0.5 text-right text-terminal-text-muted">{e.size_multiplier.toFixed(2)}x</td>
                        <td className="px-1 py-0.5 text-center">{e.kill_switch ? "🔴" : "✓"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-[10px] text-terminal-text-faint py-2">No execution log entries yet...</div>
            )}
          </DashboardPanel>

          {/* Factor Model + Learning State */}
          <div className="flex gap-1 flex-shrink-0">
            <DashboardPanel title="FACTOR MODEL — ALPHA OPTIMIZER" className="flex-1">
              <div className="space-y-1 text-[9px]">
                {factorModel.oos_sharpe !== undefined && (
                  <div className="flex items-center justify-between">
                    <span className="text-terminal-text-faint">OOS Sharpe</span>
                    <span className="font-mono" style={{ color: factorModel.oos_sharpe > 1 ? "#3fb950" : factorModel.oos_sharpe > 0.5 ? "#d29922" : "#f85149" }}>
                      {factorModel.oos_sharpe.toFixed(3)}
                    </span>
                  </div>
                )}
                {factorModel.total_factors !== undefined && (
                  <div className="flex items-center justify-between">
                    <span className="text-terminal-text-faint">Total Factors</span>
                    <span className="font-mono text-terminal-text-muted">{factorModel.total_factors}</span>
                  </div>
                )}
                {factorModel.should_retrain !== undefined && (
                  <div className="flex items-center justify-between">
                    <span className="text-terminal-text-faint">Needs Retrain</span>
                    <span className={`font-mono ${factorModel.should_retrain ? "text-red-400" : "text-green-400"}`}>
                      {factorModel.should_retrain ? "YES" : "NO"}
                    </span>
                  </div>
                )}
                {factorModel.factors_by_category && Object.entries(factorModel.factors_by_category).map(([cat, factors]) => (
                  <div key={cat}>
                    <div className="text-terminal-text-faint capitalize">{cat}</div>
                    <div className="text-[8px] text-terminal-text-muted font-mono truncate">
                      {Array.isArray(factors) ? factors.slice(0, 5).join(", ") : String(factors)}
                    </div>
                  </div>
                ))}
                {Object.keys(factorModel).length === 0 && (
                  <div className="text-terminal-text-faint">Loading factor model...</div>
                )}
              </div>
            </DashboardPanel>

            <DashboardPanel title="LEARNING STATE — GSD PIPELINE" className="flex-1">
              <div className="space-y-1 text-[9px]">
                {learning.orchestrator && (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-terminal-text-faint">GSD Active</span>
                      <span className={`font-mono ${learning.orchestrator.gsd_active ? "text-green-400" : "text-red-400"}`}>
                        {learning.orchestrator.gsd_active ? "YES" : "NO"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-terminal-text-faint">Paul Active</span>
                      <span className={`font-mono ${learning.orchestrator.paul_active ? "text-green-400" : "text-red-400"}`}>
                        {learning.orchestrator.paul_active ? "YES" : "NO"}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-terminal-text-faint">Attached Agents</span>
                      <span className="font-mono text-terminal-text-muted">{learning.orchestrator.attached_agents}</span>
                    </div>
                  </>
                )}
                {learning.execution_learning && (
                  <>
                    <div className="border-t border-terminal-border/30 my-1" />
                    <div className="flex items-center justify-between">
                      <span className="text-terminal-text-faint">Executions</span>
                      <span className="font-mono text-terminal-text-muted">{learning.execution_learning.total_executions}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-terminal-text-faint">Avg Active</span>
                      <span className="font-mono text-terminal-text-muted">{learning.execution_learning.avg_active_strategies}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-terminal-text-faint">Avg Agreement</span>
                      <span className="font-mono text-terminal-text-muted">{(learning.execution_learning.avg_agreement * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-terminal-text-faint">Kill Switches</span>
                      <span className="font-mono text-red-400">{learning.execution_learning.kill_switch_activations}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-terminal-text-faint">Consistency</span>
                      <span className="font-mono" style={{ color: learning.execution_learning.strategy_consistency > 0.9 ? "#3fb950" : "#d29922" }}>
                        {(learning.execution_learning.strategy_consistency * 100).toFixed(1)}%
                      </span>
                    </div>
                  </>
                )}
                {learning.high_conviction_count !== undefined && (
                  <div className="flex items-center justify-between">
                    <span className="text-terminal-text-faint">Hi-Conv Signals</span>
                    <span className="font-mono text-terminal-accent">{learning.high_conviction_count}</span>
                  </div>
                )}
                {Object.keys(learning).length === 0 && (
                  <div className="text-terminal-text-faint">Loading learning state...</div>
                )}
              </div>
            </DashboardPanel>
          </div>
        </div>

        {/* Right: indicators panel */}
        <div className="h-full overflow-auto">
          <DashboardPanel title="TECHNICAL INDICATORS" className="h-full">
            <div className="space-y-3 text-[10px]">

              {/* RSI Gauge */}
              <div>
                <div className="text-[9px] text-terminal-text-faint tracking-wider mb-1">RSI (14)</div>
                <RSIGauge value={indicators?.rsi ?? null} />
              </div>

              <div className="border-t border-terminal-border/50" />

              {/* Moving Averages */}
              <div>
                <div className="text-[9px] text-terminal-text-faint tracking-wider mb-1.5">MOVING AVERAGES</div>
                {[
                  { label: "SMA (20)", value: indicators?.sma20 ?? null, color: "#f0883e" },
                  { label: "SMA (50)", value: indicators?.sma50 ?? null, color: "#d2a8ff" },
                  { label: "SMA (200)", value: indicators?.sma200 ?? null, color: "#58a6ff" },
                  { label: "EMA (12)", value: indicators?.ema12 ?? null, color: "#00d4aa" },
                  { label: "EMA (26)", value: indicators?.ema26 ?? null, color: "#d29922" },
                ].map(m => {
                  const isAbove = m.value !== null && close >= m.value;
                  return (
                    <div key={m.label} className="flex items-center justify-between py-0.5 border-b border-terminal-border/30">
                      <span className="text-terminal-text-faint" style={{ color: m.color }}>
                        {m.label}
                      </span>
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono">{m.value !== null ? `$${m.value}` : "—"}</span>
                        {m.value !== null && (
                          <span className={`text-[8px] font-semibold ${isAbove ? "text-terminal-positive" : "text-terminal-negative"}`}>
                            {isAbove ? "▲" : "▼"}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="border-t border-terminal-border/50" />

              {/* MACD values */}
              {indicators?.macd && (
                <div>
                  <div className="text-[9px] text-terminal-text-faint tracking-wider mb-1.5">MACD (12, 26, 9)</div>
                  {[
                    { label: "MACD Line", value: indicators.macd.macd.toFixed(3), color: "#58a6ff" },
                    { label: "Signal Line", value: indicators.macd.signal.toFixed(3), color: "#f0883e" },
                    { label: "Histogram", value: indicators.macd.histogram.toFixed(3), color: indicators.macd.histogram >= 0 ? "#3fb950" : "#f85149" },
                  ].map(m => (
                    <div key={m.label} className="flex items-center justify-between py-0.5 border-b border-terminal-border/30">
                      <span className="text-terminal-text-faint">{m.label}</span>
                      <span className="font-mono" style={{ color: m.color }}>{m.value}</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="border-t border-terminal-border/50" />

              {/* Bollinger Bands */}
              {indicators?.bb && (
                <div>
                  <div className="text-[9px] text-terminal-text-faint tracking-wider mb-1.5">BOLLINGER BANDS (20, 2σ)</div>
                  {[
                    { label: "Upper Band", value: `$${indicators.bb.upper}`, color: "#f85149" },
                    { label: "Middle Band", value: `$${indicators.bb.mid}`, color: "#8b949e" },
                    { label: "Lower Band", value: `$${indicators.bb.lower}`, color: "#3fb950" },
                    { label: "Band Width", value: `${indicators.bb.width}%`, color: "#d29922" },
                  ].map(m => (
                    <div key={m.label} className="flex items-center justify-between py-0.5 border-b border-terminal-border/30">
                      <span className="text-terminal-text-faint">{m.label}</span>
                      <span className="font-mono" style={{ color: m.color }}>{m.value}</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="border-t border-terminal-border/50" />

              {/* ATR + Stoch */}
              <div>
                <div className="text-[9px] text-terminal-text-faint tracking-wider mb-1.5">VOLATILITY / MOMENTUM</div>
                {[
                  { label: "ATR (14)", value: indicators?.atr !== null && indicators?.atr !== undefined ? `$${indicators.atr}` : "—", color: "#d29922" },
                  { label: "Stoch %K (14,3)", value: indicators?.stoch ? `${indicators.stoch.k.toFixed(1)}` : "—", color: "#58a6ff" },
                  { label: "Stoch %D", value: indicators?.stoch ? `${indicators.stoch.d.toFixed(1)}` : "—", color: "#d2a8ff" },
                ].map(m => (
                  <div key={m.label} className="flex items-center justify-between py-0.5 border-b border-terminal-border/30">
                    <span className="text-terminal-text-faint">{m.label}</span>
                    <span className="font-mono" style={{ color: m.color }}>{m.value}</span>
                  </div>
                ))}
              </div>

              <div className="border-t border-terminal-border/50" />

              {/* Volume profile */}
              <div>
                <div className="text-[9px] text-terminal-text-faint tracking-wider mb-1.5">VOLUME PROFILE</div>
                {[
                  { label: "Avg Vol (20d)", value: indicators ? `${(indicators.avgVol / 1e6).toFixed(1)}M` : "—", color: "#8b949e" },
                  { label: "Current Vol", value: indicators ? `${(indicators.currentVol / 1e6).toFixed(1)}M` : "—", color: "#00d4aa" },
                  {
                    label: "Vol/Avg Ratio",
                    value: indicators ? `${indicators.volRatio.toFixed(2)}x` : "—",
                    color: indicators && indicators.volRatio > 1.5 ? "#3fb950" : indicators && indicators.volRatio < 0.6 ? "#f85149" : "#d29922"
                  },
                  { label: "OBV Trend", value: indicators?.obvDir ?? "—", color: indicators?.obvDir === "UP" ? "#3fb950" : indicators?.obvDir === "DOWN" ? "#f85149" : "#8b949e" },
                ].map(m => (
                  <div key={m.label} className="flex items-center justify-between py-0.5 border-b border-terminal-border/30">
                    <span className="text-terminal-text-faint">{m.label}</span>
                    <span className="font-mono" style={{ color: m.color }}>{m.value}</span>
                  </div>
                ))}
              </div>

              <div className="border-t border-terminal-border/50" />

              {/* Strategy consensus summary */}
              {stratApi && (
                <div>
                  <div className="text-[9px] text-terminal-text-faint tracking-wider mb-1.5">L7 STRATEGY CONSENSUS</div>
                  {[
                    { label: "Active Strats", value: `${stratApi.active_count}/12`, color: "#00d4aa" },
                    { label: "Agreement", value: `${(stratApi.agreement * 100).toFixed(0)}%`, color: stratApi.agreement > 0.7 ? "#3fb950" : "#d29922" },
                    { label: "Size Mult", value: `${stratApi.size_multiplier.toFixed(3)}x`, color: "#58a6ff" },
                    { label: "Stop Loss", value: `$${stratApi.stop_loss}`, color: "#f85149" },
                    { label: "Take Profit", value: `$${stratApi.take_profit}`, color: "#3fb950" },
                  ].map(m => (
                    <div key={m.label} className="flex items-center justify-between py-0.5 border-b border-terminal-border/30">
                      <span className="text-terminal-text-faint">{m.label}</span>
                      <span className="font-mono" style={{ color: m.color }}>{m.value}</span>
                    </div>
                  ))}
                </div>
              )}

            </div>
          </DashboardPanel>
        </div>
        </ResizableDashboard>
      </div>
    </div>
  );
}
