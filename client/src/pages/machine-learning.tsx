import { DashboardPanel } from "@/components/dashboard-panel";
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";
import { useMemo } from "react";
import { useEngineQuery } from "@/hooks/use-engine-api";

const STRESS_TESTS = [
  { scenario: "2008 GFC Replay", portfolioImpact: -18.2, var95: -22.5, recovery: "14 days", status: "pass" },
  { scenario: "COVID March 2020", portfolioImpact: -12.4, var95: -16.8, recovery: "8 days", status: "pass" },
  { scenario: "Rate Shock +200bp", portfolioImpact: -8.6, var95: -11.2, recovery: "5 days", status: "pass" },
  { scenario: "Tech Crash -30%", portfolioImpact: -14.1, var95: -19.3, recovery: "11 days", status: "warn" },
  { scenario: "China Devaluation", portfolioImpact: -6.3, var95: -8.9, recovery: "4 days", status: "pass" },
  { scenario: "Oil Spike $150", portfolioImpact: -9.8, var95: -13.1, recovery: "7 days", status: "pass" },
  { scenario: "Liquidity Crisis", portfolioImpact: -21.5, var95: -28.4, recovery: "18 days", status: "fail" },
];

function generateEquityCurve() {
  const data: { day: number; equity: number; benchmark: number }[] = [];
  let eq = 100;
  let bm = 100;
  for (let i = 0; i < 252; i++) {
    eq += (Math.random() - 0.44) * 1.2;
    bm += (Math.random() - 0.46) * 1.0;
    data.push({ day: i, equity: +eq.toFixed(2), benchmark: +bm.toFixed(2) });
  }
  return data;
}

const PATTERNS = [
  { pattern: "Head & Shoulders", asset: "AAPL", confidence: 87, timeframe: "4H", direction: "Bearish" },
  { pattern: "Bull Flag", asset: "NVDA", confidence: 92, timeframe: "1D", direction: "Bullish" },
  { pattern: "Double Bottom", asset: "AMZN", confidence: 78, timeframe: "1D", direction: "Bullish" },
  { pattern: "Rising Wedge", asset: "TSLA", confidence: 71, timeframe: "4H", direction: "Bearish" },
  { pattern: "Cup & Handle", asset: "META", confidence: 84, timeframe: "1W", direction: "Bullish" },
];

const ANOMALIES = [
  { asset: "SMCI", type: "Volume Spike", zScore: 4.2, severity: "critical", detected: "2m ago" },
  { asset: "GME", type: "Options Flow", zScore: 3.8, severity: "high", detected: "15m ago" },
  { asset: "BTC/USD", type: "Correlation Break", zScore: 2.9, severity: "medium", detected: "1h ago" },
  { asset: "RIVN", type: "Spread Widening", zScore: 2.4, severity: "medium", detected: "2h ago" },
  { asset: "VIX", type: "Mean Reversion", zScore: 1.8, severity: "low", detected: "3h ago" },
];

const REL_VALUE_PAIRS = [
  { pair: "AAPL / MSFT", spreadZ: 1.82, halfLife: 12, signal: "long", pnl: 2.4 },
  { pair: "XOM / CVX", spreadZ: -2.15, halfLife: 8, signal: "short", pnl: -0.8 },
  { pair: "JPM / BAC", spreadZ: 0.94, halfLife: 15, signal: "neutral", pnl: 0.3 },
  { pair: "GOOGL / META", spreadZ: 1.67, halfLife: 10, signal: "long", pnl: 1.9 },
  { pair: "HD / LOW", spreadZ: -1.43, halfLife: 14, signal: "short", pnl: 1.1 },
  { pair: "PG / CL", spreadZ: 0.52, halfLife: 22, signal: "neutral", pnl: 0.1 },
  { pair: "V / MA", spreadZ: 2.31, halfLife: 7, signal: "long", pnl: 3.2 },
  { pair: "UNH / CI", spreadZ: -1.88, halfLife: 11, signal: "short", pnl: -0.5 },
  { pair: "AVGO / QCOM", spreadZ: 1.45, halfLife: 9, signal: "long", pnl: 1.7 },
  { pair: "DIS / NFLX", spreadZ: -0.78, halfLife: 18, signal: "neutral", pnl: 0.4 },
  { pair: "COST / WMT", spreadZ: 2.08, halfLife: 13, signal: "long", pnl: 2.1 },
  { pair: "CRM / NOW", spreadZ: -1.92, halfLife: 6, signal: "short", pnl: -1.2 },
  { pair: "AMD / INTC", spreadZ: 3.14, halfLife: 5, signal: "long", pnl: 4.5 },
];

export default function MachineLearning() {
  const { data: stressApi } = useEngineQuery<{ tests: Record<string, unknown> }>("/ml/stress-tests", { refetchInterval: 60000 });
  const { data: alphaApi } = useEngineQuery<{ expected_return: number; volatility: number; sharpe: number; max_drawdown: number }>("/ml/alpha/last", { refetchInterval: 15000 });
  const { data: anomalyApi } = useEngineQuery<{ anomalies: Array<Record<string, string | number>> }>("/monitoring/anomalies", { refetchInterval: 10000 });
  const { data: pairsApi } = useEngineQuery<{ pairs: Array<Record<string, string | number>> }>("/signals/stat-arb/pairs?max_pairs=15", { refetchInterval: 30000 });
  const { data: patternsApi } = useEngineQuery<Record<string, unknown>>("/ml/patterns?ticker=SPY", { refetchInterval: 30000 });

  const equityCurve = useMemo(generateEquityCurve, []);

  // Wire anomalies
  const anomalies = anomalyApi?.anomalies?.length
    ? (anomalyApi.anomalies as Array<Record<string, string | number>>).slice(0, 5).map((a) => ({
        asset: String(a.asset || a.ticker || ""),
        type: String(a.type || ""),
        zScore: Number(a.zScore || a.z_score || 0),
        severity: String(a.severity || "medium"),
        detected: String(a.detected || a.timestamp || ""),
      }))
    : ANOMALIES;

  // Wire RV pairs
  const relValuePairs = pairsApi?.pairs?.length
    ? pairsApi.pairs.slice(0, 13).map((p) => ({
        pair: `${p.ticker_a} / ${p.ticker_b}`,
        spreadZ: Number(p.spread_zscore || 0),
        halfLife: Number(p.half_life || 0),
        signal: Number(p.spread_zscore || 0) > 1.5 ? "long" : Number(p.spread_zscore || 0) < -1.5 ? "short" : "neutral",
        pnl: Number(p.signal_strength || 0) * 10,
      }))
    : REL_VALUE_PAIRS;

  return (
    <div className="h-full grid grid-cols-[1fr_1fr] grid-rows-[auto_1fr_1fr] gap-[2px] p-[2px] overflow-auto" data-testid="ml-dashboard">
      {/* Stress Testing */}
      <DashboardPanel title="STRESS TESTING" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[9px] font-mono">
            <thead>
              <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                <th className="text-left px-2 py-1.5 font-medium">Scenario</th>
                <th className="text-right px-2 py-1.5 font-medium">Impact</th>
                <th className="text-right px-2 py-1.5 font-medium">VaR 95</th>
                <th className="text-right px-2 py-1.5 font-medium">Recovery</th>
                <th className="text-center px-2 py-1.5 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {STRESS_TESTS.map((s, i) => (
                <tr key={i} className="border-b border-terminal-border/20">
                  <td className="px-2 py-1.5 text-terminal-text-muted">{s.scenario}</td>
                  <td className="px-2 py-1.5 text-right text-terminal-negative tabular-nums">{s.portfolioImpact}%</td>
                  <td className="px-2 py-1.5 text-right text-terminal-negative tabular-nums">{s.var95}%</td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-muted">{s.recovery}</td>
                  <td className="px-2 py-1.5 text-center">
                    <span className={`px-1.5 py-0.5 rounded text-[7px] font-bold ${
                      s.status === "pass" ? "bg-terminal-positive/10 text-terminal-positive" :
                      s.status === "warn" ? "bg-terminal-warning/10 text-terminal-warning" :
                      "bg-terminal-negative/10 text-terminal-negative"
                    }`}>{s.status.toUpperCase()}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>

      {/* Backtesting Equity Curve */}
      <DashboardPanel title="BACKTESTING — EQUITY CURVE">
        <div className="h-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={equityCurve} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
              <defs>
                <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#00d4aa" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#00d4aa" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="day" tick={{ fontSize: 7, fill: "#484f58" }} axisLine={{ stroke: "#1e2633" }} tickLine={false} interval={50} />
              <YAxis tick={{ fontSize: 7, fill: "#484f58" }} axisLine={false} tickLine={false} width={30} />
              <Tooltip
                contentStyle={{ background: "#0d1117", border: "1px solid #1e2633", borderRadius: 4, fontSize: 9, fontFamily: "JetBrains Mono", color: "#e6edf3" }}
              />
              <Area type="monotone" dataKey="equity" stroke="#00d4aa" strokeWidth={1.2} fill="url(#eqGrad)" name="Strategy" />
              <Area type="monotone" dataKey="benchmark" stroke="#58a6ff" strokeWidth={0.8} fill="none" name="Benchmark" strokeDasharray="4 2" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </DashboardPanel>

      {/* Pattern Recognition */}
      <DashboardPanel title="PATTERN RECOGNITION" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[9px] font-mono">
            <thead>
              <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                <th className="text-left px-2 py-1.5 font-medium">Pattern</th>
                <th className="text-left px-2 py-1.5 font-medium">Asset</th>
                <th className="text-right px-2 py-1.5 font-medium">Conf%</th>
                <th className="text-left px-2 py-1.5 font-medium">Dir</th>
              </tr>
            </thead>
            <tbody>
              {PATTERNS.map((p, i) => (
                <tr key={i} className="border-b border-terminal-border/20">
                  <td className="px-2 py-1.5 text-terminal-text-muted">{p.pattern}</td>
                  <td className="px-2 py-1.5 text-terminal-accent font-medium">{p.asset}</td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-primary tabular-nums">{p.confidence}%</td>
                  <td className={`px-2 py-1.5 ${p.direction === "Bullish" ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {p.direction === "Bullish" ? "▲" : "▼"} {p.direction}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>

      {/* Anomalies */}
      <DashboardPanel title="STATISTICAL ANOMALIES" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[9px] font-mono">
            <thead>
              <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                <th className="text-left px-2 py-1.5 font-medium">Asset</th>
                <th className="text-left px-2 py-1.5 font-medium">Type</th>
                <th className="text-right px-2 py-1.5 font-medium">Z-Score</th>
                <th className="text-left px-2 py-1.5 font-medium">Sev</th>
              </tr>
            </thead>
            <tbody>
              {anomalies.map((a, i) => (
                <tr key={i} className="border-b border-terminal-border/20">
                  <td className="px-2 py-1.5 text-terminal-text-primary">{a.asset}</td>
                  <td className="px-2 py-1.5 text-terminal-text-muted">{a.type}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-warning">{a.zScore.toFixed(1)}</td>
                  <td className="px-2 py-1.5">
                    <span className={`px-1 py-0.5 rounded text-[7px] font-bold ${
                      a.severity === "critical" ? "bg-terminal-negative/15 text-terminal-negative" :
                      a.severity === "high" ? "bg-terminal-warning/15 text-terminal-warning" :
                      a.severity === "medium" ? "bg-terminal-blue/15 text-terminal-blue" :
                      "bg-terminal-text-faint/15 text-terminal-text-muted"
                    }`}>{a.severity.toUpperCase()}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>

      {/* Relative Value Pairs */}
      <DashboardPanel title="RELATIVE VALUE PAIRS" className="col-span-2" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[9px] font-mono">
            <thead>
              <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                <th className="text-left px-2 py-1.5 font-medium">Pair</th>
                <th className="text-right px-2 py-1.5 font-medium">Spread Z</th>
                <th className="text-right px-2 py-1.5 font-medium">Half-Life</th>
                <th className="text-center px-2 py-1.5 font-medium">Signal</th>
                <th className="text-right px-2 py-1.5 font-medium">P&L %</th>
              </tr>
            </thead>
            <tbody>
              {relValuePairs.map((p, i) => (
                <tr key={i} className="border-b border-terminal-border/20">
                  <td className="px-2 py-1.5 text-terminal-text-primary">{p.pair}</td>
                  <td className={`px-2 py-1.5 text-right tabular-nums ${Math.abs(p.spreadZ) > 2 ? "text-terminal-warning" : "text-terminal-text-muted"}`}>
                    {p.spreadZ > 0 ? "+" : ""}{p.spreadZ.toFixed(2)}
                  </td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-muted tabular-nums">{p.halfLife}d</td>
                  <td className="px-2 py-1.5 text-center">
                    <span className={`px-1.5 py-0.5 rounded text-[7px] font-bold ${
                      p.signal === "long" ? "bg-terminal-positive/10 text-terminal-positive" :
                      p.signal === "short" ? "bg-terminal-negative/10 text-terminal-negative" :
                      "bg-terminal-text-faint/10 text-terminal-text-muted"
                    }`}>{p.signal.toUpperCase()}</span>
                  </td>
                  <td className={`px-2 py-1.5 text-right tabular-nums ${p.pnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {p.pnl >= 0 ? "+" : ""}{p.pnl.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>
    </div>
  );
}
