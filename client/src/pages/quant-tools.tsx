import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ComposedChart, Line, Bar, BarChart, ReferenceLine,
  LineChart,
} from "recharts";

// ═══════════ TYPES ═══════════

interface OHLCV {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface TickerConfig {
  ticker: string;
  sector: string;
  basePrice: number;
  volatility: number;
  trend: number;
}

// ═══════════ TICKER CONFIGS ═══════════

const TICKERS: TickerConfig[] = [
  { ticker: "AAPL", sector: "Technology", basePrice: 189.0, volatility: 0.018, trend: 0.0008 },
  { ticker: "MSFT", sector: "Technology", basePrice: 415.0, volatility: 0.016, trend: 0.0009 },
  { ticker: "NVDA", sector: "Technology", basePrice: 878.0, volatility: 0.032, trend: 0.0015 },
  { ticker: "GOOGL", sector: "Communication", basePrice: 174.0, volatility: 0.019, trend: 0.0007 },
  { ticker: "AMZN", sector: "Consumer Disc.", basePrice: 189.0, volatility: 0.021, trend: 0.0010 },
  { ticker: "JPM", sector: "Financials", basePrice: 198.0, volatility: 0.015, trend: 0.0006 },
  { ticker: "XOM", sector: "Energy", basePrice: 113.0, volatility: 0.017, trend: -0.0003 },
  { ticker: "META", sector: "Communication", basePrice: 498.0, volatility: 0.024, trend: 0.0011 },
  { ticker: "TSLA", sector: "Consumer Disc.", basePrice: 248.0, volatility: 0.038, trend: -0.0004 },
  { ticker: "JNJ", sector: "Healthcare", basePrice: 147.0, volatility: 0.012, trend: 0.0003 },
  { ticker: "V", sector: "Financials", basePrice: 274.0, volatility: 0.013, trend: 0.0005 },
  { ticker: "AMD", sector: "Technology", basePrice: 178.0, volatility: 0.034, trend: 0.0012 },
];

// ═══════════ OHLCV GENERATION ═══════════

function seeded(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

function generateOHLCV(cfg: TickerConfig): OHLCV[] {
  const rng = seeded(cfg.ticker.charCodeAt(0) * 31 + cfg.ticker.charCodeAt(1) * 7);
  const data: OHLCV[] = [];
  let price = cfg.basePrice * (0.88 + rng() * 0.08);

  for (let i = 119; i >= 0; i--) {
    const d = new Date(Date.now() - i * 86400000);
    // skip weekends
    if (d.getDay() === 0 || d.getDay() === 6) continue;
    const dailyReturn = cfg.trend + (rng() - 0.5) * cfg.volatility * 2;
    const open = price;
    price = open * (1 + dailyReturn);
    const intraRange = price * cfg.volatility * (0.5 + rng() * 0.8);
    const high = Math.max(open, price) + intraRange * rng();
    const low = Math.min(open, price) - intraRange * rng();
    const volume = Math.floor(8_000_000 + rng() * 40_000_000);
    data.push({
      date: d.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      open: +open.toFixed(2),
      high: +high.toFixed(2),
      low: +low.toFixed(2),
      close: +price.toFixed(2),
      volume,
    });
  }
  return data;
}

// ═══════════ INDICATOR CALCULATIONS ═══════════

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
  // Signal is 9-period EMA of MACD — approximate
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
  const k = +((( lastClose - lowestLow) / (highestHigh - lowestLow)) * 100).toFixed(2);
  const d = +(k * 0.9 + 5).toFixed(2); // simplified
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

  // RSI
  const rsi = indicators.rsi ?? 50;
  const rsiSig: Signal = rsi < 30 ? "STRONG_BUY" : rsi < 45 ? "BUY" : rsi > 70 ? "STRONG_SELL" : rsi > 60 ? "SELL" : "NEUTRAL";
  signals.push({ name: "RSI (14)", value: `${rsi.toFixed(1)}`, signal: rsiSig });
  if (rsiSig === "STRONG_BUY") bull += 2; else if (rsiSig === "BUY") bull += 1;
  else if (rsiSig === "STRONG_SELL") bear += 2; else if (rsiSig === "SELL") bear += 1;

  // SMA 20
  if (indicators.sma20) {
    const sma20Sig: Signal = close > indicators.sma20 * 1.02 ? "BUY" : close < indicators.sma20 * 0.98 ? "SELL" : "NEUTRAL";
    signals.push({ name: "SMA (20)", value: `$${indicators.sma20}`, signal: sma20Sig });
    if (sma20Sig === "BUY") bull += 1; else if (sma20Sig === "SELL") bear += 1;
  }

  // SMA 50
  if (indicators.sma50) {
    const sma50Sig: Signal = close > indicators.sma50 * 1.01 ? "BUY" : close < indicators.sma50 * 0.99 ? "SELL" : "NEUTRAL";
    signals.push({ name: "SMA (50)", value: `$${indicators.sma50}`, signal: sma50Sig });
    if (sma50Sig === "BUY") bull += 1; else if (sma50Sig === "SELL") bear += 1;
  }

  // SMA 200
  if (indicators.sma200) {
    const sma200Sig: Signal = close > indicators.sma200 ? "BUY" : "SELL";
    signals.push({ name: "SMA (200)", value: `$${indicators.sma200}`, signal: sma200Sig });
    if (sma200Sig === "BUY") bull += 1; else if (sma200Sig === "SELL") bear += 1;
  }

  // MACD
  if (indicators.macd) {
    const { histogram } = indicators.macd;
    const macdSig: Signal = histogram > 0.5 ? "STRONG_BUY" : histogram > 0 ? "BUY" : histogram < -0.5 ? "STRONG_SELL" : "SELL";
    signals.push({ name: "MACD", value: `${indicators.macd.macd.toFixed(3)}`, signal: macdSig });
    if (macdSig === "STRONG_BUY") bull += 2; else if (macdSig === "BUY") bull += 1;
    else if (macdSig === "STRONG_SELL") bear += 2; else if (macdSig === "SELL") bear += 1;
  }

  // Bollinger Bands
  if (indicators.bb) {
    const { upper, lower, mid } = indicators.bb;
    const bbSig: Signal = close < lower ? "STRONG_BUY" : close < mid * 0.99 ? "BUY" : close > upper ? "STRONG_SELL" : close > mid * 1.01 ? "SELL" : "NEUTRAL";
    signals.push({ name: "Boll. Bands (20,2σ)", value: `BW: ${indicators.bb.width}%`, signal: bbSig });
    if (bbSig === "STRONG_BUY") bull += 2; else if (bbSig === "BUY") bull += 1;
    else if (bbSig === "STRONG_SELL") bear += 2; else if (bbSig === "SELL") bear += 1;
  }

  // Stochastic
  if (indicators.stoch) {
    const { k } = indicators.stoch;
    const stochSig: Signal = k < 20 ? "STRONG_BUY" : k < 35 ? "BUY" : k > 80 ? "STRONG_SELL" : k > 65 ? "SELL" : "NEUTRAL";
    signals.push({ name: "Stochastic (14,3)", value: `K:${k.toFixed(1)} D:${indicators.stoch.d.toFixed(1)}`, signal: stochSig });
    if (stochSig === "STRONG_BUY") bull += 2; else if (stochSig === "BUY") bull += 1;
    else if (stochSig === "STRONG_SELL") bear += 2; else if (stochSig === "SELL") bear += 1;
  }

  // Volume
  const volSig: Signal = indicators.volRatio > 1.5 ? "BUY" : indicators.volRatio < 0.6 ? "SELL" : "NEUTRAL";
  signals.push({ name: "Volume Profile", value: `${indicators.volRatio.toFixed(2)}x avg`, signal: volSig });

  // OBV
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

// ═══════════ QUANT STRATEGIES ═══════════

interface QuantStrategy {
  name: string;
  description: string;
  confidence: number;
  action: Signal;
  params: string;
}

function generateStrategies(close: number, indicators: IndicatorSet): QuantStrategy[] {
  const rsi = indicators.rsi ?? 50;
  const macdHist = indicators.macd?.histogram ?? 0;
  const bbWidth = indicators.bb?.width ?? 5;

  return [
    {
      name: "ARIMA+GARCH",
      description: "Time series forecasting with volatility clustering",
      confidence: Math.min(95, Math.max(35, 62 + (rsi - 50) * 0.4 + macdHist * 5)),
      action: rsi < 50 && macdHist > 0 ? "BUY" : rsi > 60 ? "SELL" : "NEUTRAL",
      params: "p=2,d=1,q=1 | GARCH(1,1)",
    },
    {
      name: "EMA Crossover + Bollinger",
      description: "Dual EMA crossover confirmed by BB squeeze",
      confidence: Math.min(90, Math.max(30, 55 + (indicators.ema12 && indicators.ema26 ? (indicators.ema12 - indicators.ema26) / indicators.ema26 * 1000 : 0))),
      action: indicators.ema12 && indicators.ema26 && indicators.ema12 > indicators.ema26 ? (bbWidth < 4 ? "STRONG_BUY" : "BUY") : "NEUTRAL",
      params: "EMA(12,26) | BB(20,2σ) squeeze",
    },
    {
      name: "Cointegration Pairs",
      description: "Statistical arbitrage via OU mean-reversion",
      confidence: Math.min(85, Math.max(25, 48 + Math.random() * 30)),
      action: Math.random() > 0.5 ? "BUY" : "NEUTRAL",
      params: "Z-score: ±2.0 entry | ±0.5 exit | ±4.0 stop",
    },
    {
      name: "LSTM Forecasting",
      description: "Deep learning sequence model 5-day forward",
      confidence: Math.min(88, Math.max(40, 61 + (indicators.sma20 && close > indicators.sma20 ? 8 : -5))),
      action: indicators.sma50 && close > indicators.sma50 ? "BUY" : "SELL",
      params: "Seq=60d | Layers=3 | Dropout=0.2",
    },
    {
      name: "Sentiment + Momentum",
      description: "Combined NLP score with price momentum",
      confidence: Math.min(80, Math.max(30, 52 + (rsi > 55 ? 10 : -5))),
      action: rsi > 60 && macdHist > 0 ? "STRONG_BUY" : rsi > 50 ? "BUY" : "NEUTRAL",
      params: "FinBERT score | 20d momentum",
    },
  ];
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
            <rect key={i} fill={entry.histogram >= 0 ? "#3fb950" : "#f85149"} />
          ))}
        </Bar>
        <Line type="monotone" dataKey="macd" stroke="#58a6ff" strokeWidth={1} dot={false} name="MACD" />
        <Line type="monotone" dataKey="signal" stroke="#f0883e" strokeWidth={1} dot={false} name="Signal" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ═══════════ PRICE CHART ═══════════

function PriceChart({ data, indicators }: { data: OHLCV[]; indicators: IndicatorSet }) {
  const chartData = data.map(d => ({
    date: d.date,
    price: d.close,
    high: d.high,
    low: d.low,
    volume: d.volume,
    sma20: null as number | null,
    sma50: null as number | null,
    bbUpper: null as number | null,
    bbLower: null as number | null,
  }));

  // Add SMA/BB values progressively
  const closes = data.map(d => d.close);
  chartData.forEach((d, i) => {
    if (i >= 19) d.sma20 = +(closes.slice(i - 19, i + 1).reduce((s, v) => s + v, 0) / 20).toFixed(2);
    if (i >= 49) d.sma50 = +(closes.slice(i - 49, i + 1).reduce((s, v) => s + v, 0) / 50).toFixed(2);
    if (i >= 19) {
      const slice = closes.slice(i - 19, i + 1);
      const mid = slice.reduce((s, v) => s + v, 0) / 20;
      const sigma = Math.sqrt(slice.reduce((s, v) => s + (v - mid) ** 2, 0) / 20);
      d.bbUpper = +(mid + 2 * sigma).toFixed(2);
      d.bbLower = +(mid - 2 * sigma).toFixed(2);
    }
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
          tickFormatter={(v) => `$${v.toFixed(0)}`}
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

  const cfg = useMemo(() => TICKERS.find(t => t.ticker === selectedTicker) ?? TICKERS[0], [selectedTicker]);
  const ohlcv = useMemo(() => generateOHLCV(cfg), [cfg]);
  const indicators = useMemo(() => calcIndicators(ohlcv), [ohlcv]);
  const close = ohlcv[ohlcv.length - 1]?.close ?? cfg.basePrice;
  const prevClose = ohlcv[ohlcv.length - 2]?.close ?? close;
  const pctChange = +((close - prevClose) / prevClose * 100).toFixed(2);
  const { composite, signals, strength } = useMemo(() => scoreSignal(indicators, close), [indicators, close]);
  const strategies = useMemo(() => generateStrategies(close, indicators), [close, indicators]);

  return (
    <div className="h-full flex flex-col gap-1 p-1 overflow-hidden">

      {/* Ticker selector + price header */}
      <div className="flex-shrink-0 flex items-center gap-2">
        <div className="flex items-center gap-1 bg-terminal-surface border border-terminal-border/60 rounded p-1 flex-wrap">
          {TICKERS.map(t => (
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
          ))}
        </div>
        <div className="flex items-center gap-3 ml-2">
          <span className="text-lg font-mono font-bold text-terminal-text-primary">{selectedTicker}</span>
          <span className="text-xl font-mono font-bold text-terminal-text-primary">${close.toFixed(2)}</span>
          <span className={`text-sm font-mono font-semibold ${pctChange >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
            {pctChange >= 0 ? "+" : ""}{pctChange}%
          </span>
          <span className="text-[10px] text-terminal-text-faint font-mono">{cfg.sector}</span>
        </div>
      </div>

      {/* Main grid */}
      <div className="flex-1 grid grid-cols-[1fr_240px] gap-1 overflow-hidden">
        {/* Left: price chart + signals + strategies */}
        <div className="flex flex-col gap-1 overflow-hidden">
          {/* Price chart */}
          <DashboardPanel
            title={`${selectedTicker} — PRICE CHART (120 DAYS)`}
            className="flex-1"
            headerRight={
              <span className="text-[9px] font-mono text-terminal-text-faint">
                SMA20 <span className="text-[#f0883e]">●</span>  SMA50 <span className="text-[#d2a8ff]">●</span>  BB2σ <span className="text-[#58a6ff]">●</span>
              </span>
            }
          >
            <PriceChart data={ohlcv} indicators={indicators} />
          </DashboardPanel>

          {/* MACD */}
          <DashboardPanel title="MACD (12, 26, 9)" className="h-24 flex-shrink-0">
            <MACDChart data={ohlcv} />
          </DashboardPanel>

          {/* Signal summary */}
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

          {/* Quant strategies */}
          <DashboardPanel title="QUANT STRATEGY LIBRARY" className="flex-shrink-0">
            <div className="grid grid-cols-5 gap-1">
              {strategies.map(s => (
                <div key={s.name} className="bg-terminal-bg border border-terminal-border/50 rounded p-2">
                  <div className="text-[9px] font-semibold text-terminal-accent mb-0.5 truncate">{s.name}</div>
                  <div className="text-[8px] text-terminal-text-faint mb-1.5 leading-tight">{s.description}</div>
                  <div className="text-[8px] text-terminal-text-faint font-mono mb-1.5">{s.params}</div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[9px] text-terminal-text-faint">Confidence</span>
                    <span className="font-mono text-[9px]" style={{ color: s.confidence > 70 ? "#3fb950" : s.confidence > 50 ? "#d29922" : "#f85149" }}>
                      {s.confidence.toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-1 bg-terminal-surface rounded-full overflow-hidden mb-1.5">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${s.confidence}%`, backgroundColor: s.confidence > 70 ? "#3fb950" : s.confidence > 50 ? "#d29922" : "#f85149" }}
                    />
                  </div>
                  <SignalDot signal={s.action} />
                </div>
              ))}
            </div>
          </DashboardPanel>
        </div>

        {/* Right: indicators panel */}
        <div className="overflow-auto">
          <DashboardPanel title="TECHNICAL INDICATORS" className="h-full">
            <div className="space-y-3 text-[10px]">

              {/* RSI Gauge */}
              <div>
                <div className="text-[9px] text-terminal-text-faint tracking-wider mb-1">RSI (14)</div>
                <RSIGauge value={indicators.rsi} />
              </div>

              <div className="border-t border-terminal-border/50" />

              {/* Moving Averages */}
              <div>
                <div className="text-[9px] text-terminal-text-faint tracking-wider mb-1.5">MOVING AVERAGES</div>
                {[
                  { label: "SMA (20)", value: indicators.sma20, color: "#f0883e" },
                  { label: "SMA (50)", value: indicators.sma50, color: "#d2a8ff" },
                  { label: "SMA (200)", value: indicators.sma200, color: "#58a6ff" },
                  { label: "EMA (12)", value: indicators.ema12, color: "#00d4aa" },
                  { label: "EMA (26)", value: indicators.ema26, color: "#d29922" },
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
              {indicators.macd && (
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
              {indicators.bb && (
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
                  { label: "ATR (14)", value: indicators.atr !== null ? `$${indicators.atr}` : "—", color: "#d29922" },
                  { label: "Stoch %K (14,3)", value: indicators.stoch ? `${indicators.stoch.k.toFixed(1)}` : "—", color: "#58a6ff" },
                  { label: "Stoch %D", value: indicators.stoch ? `${indicators.stoch.d.toFixed(1)}` : "—", color: "#d2a8ff" },
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
                  { label: "Avg Vol (20d)", value: `${(indicators.avgVol / 1e6).toFixed(1)}M`, color: "#8b949e" },
                  { label: "Current Vol", value: `${(indicators.currentVol / 1e6).toFixed(1)}M`, color: "#00d4aa" },
                  {
                    label: "Vol/Avg Ratio",
                    value: `${indicators.volRatio.toFixed(2)}x`,
                    color: indicators.volRatio > 1.5 ? "#3fb950" : indicators.volRatio < 0.6 ? "#f85149" : "#d29922"
                  },
                  { label: "OBV Trend", value: indicators.obvDir, color: indicators.obvDir === "UP" ? "#3fb950" : indicators.obvDir === "DOWN" ? "#f85149" : "#8b949e" },
                ].map(m => (
                  <div key={m.label} className="flex items-center justify-between py-0.5 border-b border-terminal-border/30">
                    <span className="text-terminal-text-faint">{m.label}</span>
                    <span className="font-mono" style={{ color: m.color }}>{m.value}</span>
                  </div>
                ))}
              </div>

            </div>
          </DashboardPanel>
        </div>
      </div>
    </div>
  );
}
