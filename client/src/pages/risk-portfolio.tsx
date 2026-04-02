import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import { useState, useEffect, useMemo } from "react";
import { useEngineQuery, type RiskGreeks, type BetaState } from "@/hooks/use-engine-api";

const GREEKS = [
  { name: "Delta", value: 0.72, label: "Δ", color: "#00d4aa" },
  { name: "Gamma", value: 0.034, label: "Γ", color: "#58a6ff" },
  { name: "Theta", value: -42.15, label: "Θ", color: "#f85149" },
  { name: "Vega", value: 18.67, label: "ν", color: "#bc8cff" },
  { name: "Rho", value: 5.23, label: "ρ", color: "#d29922" },
];

const METRICS = [
  { name: "Sharpe Ratio", value: "1.82", status: "good" },
  { name: "Sortino Ratio", value: "2.45", status: "good" },
  { name: "Max Drawdown", value: "-8.78%", status: "warning" },
  { name: "Calmar Ratio", value: "1.55", status: "good" },
  { name: "Information Ratio", value: "0.94", status: "neutral" },
  { name: "Treynor Ratio", value: "12.3%", status: "good" },
];

const FILLS = [
  { time: "14:32:18", pair: "AAPL", side: "BUY", qty: 500, price: 189.45, status: "FILLED", strategy: "Momentum" },
  { time: "14:31:45", pair: "MSFT", side: "SELL", qty: 200, price: 420.12, status: "FILLED", strategy: "Mean Reversion" },
  { time: "14:30:22", pair: "NVDA", side: "BUY", qty: 100, price: 875.30, status: "FILLED", strategy: "Growth" },
  { time: "14:29:55", pair: "AMZN", side: "BUY", qty: 300, price: 185.67, status: "PARTIAL", strategy: "Quality" },
  { time: "14:28:10", pair: "GOOGL", side: "SELL", qty: 150, price: 155.89, status: "FILLED", strategy: "Pairs Trade" },
  { time: "14:27:30", pair: "META", side: "BUY", qty: 250, price: 505.78, status: "NO FILL", strategy: "Momentum" },
  { time: "14:25:12", pair: "JPM", side: "SELL", qty: 400, price: 198.34, status: "FILLED", strategy: "Event-Driven" },
];

const LIQUIDITY_RISK = [
  { asset: "AAPL", score: 95, adv: "62.3M", spread: "0.01%", impact: "Low" },
  { asset: "NVDA", score: 92, adv: "45.1M", spread: "0.02%", impact: "Low" },
  { asset: "MSFT", score: 94, adv: "28.7M", spread: "0.01%", impact: "Low" },
  { asset: "BTC/USD", score: 78, adv: "18.2B", spread: "0.05%", impact: "Medium" },
  { asset: "SMCI", score: 55, adv: "8.4M", spread: "0.15%", impact: "High" },
  { asset: "ARM", score: 62, adv: "12.1M", spread: "0.08%", impact: "Medium" },
];

const OPTIONS_POSITIONS = [
  { ticker: "AAPL", type: "PUT", strike: 185, expiry: "May-17", qty: -10, delta: -0.38, gamma: 0.021, theta: -8.45, vega: 12.30, premium: 4.20, mktVal: -4200, pnl: 820, strategy: "Protective Put" },
  { ticker: "AAPL", type: "CALL", strike: 195, expiry: "May-17", qty: 10, delta: 0.31, gamma: 0.018, theta: -7.20, vega: 10.85, premium: 3.15, mktVal: 3150, pnl: -480, strategy: "Covered Call" },
  { ticker: "MSFT", type: "PUT", strike: 410, expiry: "Jun-21", qty: -5, delta: -0.42, gamma: 0.015, theta: -6.80, vega: 9.40, premium: 8.50, mktVal: -4250, pnl: 1250, strategy: "Collar" },
  { ticker: "MSFT", type: "CALL", strike: 435, expiry: "Jun-21", qty: 5, delta: 0.28, gamma: 0.012, theta: -5.60, vega: 8.20, premium: 6.20, mktVal: 3100, pnl: -620, strategy: "Collar" },
  { ticker: "NVDA", type: "CALL", strike: 900, expiry: "Apr-19", qty: 20, delta: 0.55, gamma: 0.008, theta: -18.40, vega: 22.10, premium: 15.80, mktVal: 31600, pnl: 8940, strategy: "Momentum" },
  { ticker: "NVDA", type: "PUT", strike: 840, expiry: "Apr-19", qty: -20, delta: -0.22, gamma: 0.006, theta: -12.30, vega: 15.40, premium: 8.60, mktVal: -17200, pnl: 3200, strategy: "Spread" },
  { ticker: "JPM", type: "CALL", strike: 200, expiry: "May-17", qty: 15, delta: 0.48, gamma: 0.019, theta: -4.20, vega: 6.80, premium: 5.40, mktVal: 8100, pnl: 1350, strategy: "Covered Call" },
  { ticker: "SPY", type: "PUT", strike: 520, expiry: "May-31", qty: -30, delta: -0.35, gamma: 0.012, theta: -9.80, vega: 18.60, premium: 7.80, mktVal: -23400, pnl: 4800, strategy: "Protective Put" },
  { ticker: "QQQ", type: "CALL", strike: 450, expiry: "May-31", qty: 25, delta: 0.52, gamma: 0.014, theta: -11.20, vega: 20.40, premium: 9.20, mktVal: 23000, pnl: 5750, strategy: "Straddle" },
  { ticker: "QQQ", type: "PUT", strike: 450, expiry: "May-31", qty: 25, delta: -0.48, gamma: 0.014, theta: -10.80, vega: 20.40, premium: 8.80, mktVal: 22000, pnl: 4500, strategy: "Straddle" },
  { ticker: "XOM", type: "PUT", strike: 112, expiry: "Jun-21", qty: -8, delta: -0.44, gamma: 0.022, theta: -3.60, vega: 5.20, premium: 3.80, mktVal: -3040, pnl: 960, strategy: "Spread" },
  { ticker: "META", type: "CALL", strike: 520, expiry: "May-17", qty: 10, delta: 0.41, gamma: 0.016, theta: -13.50, vega: 16.80, premium: 11.40, mktVal: 11400, pnl: 2280, strategy: "Butterfly" },
];

const AGG_GREEKS = {
  delta: OPTIONS_POSITIONS.reduce((s, o) => s + o.delta * o.qty, 0),
  gamma: OPTIONS_POSITIONS.reduce((s, o) => s + o.gamma * Math.abs(o.qty), 0),
  theta: OPTIONS_POSITIONS.reduce((s, o) => s + o.theta * Math.abs(o.qty), 0),
  vega: OPTIONS_POSITIONS.reduce((s, o) => s + o.vega * Math.abs(o.qty), 0),
  totalPnl: OPTIONS_POSITIONS.reduce((s, o) => s + o.pnl, 0),
};

const FUTURES_POSITIONS = [
  { contract: "ESM25", desc: "S&P 500 Jun", side: "LONG", qty: 4, entry: 5248.50, last: 5282.75, pnl: 34250, margin: 45000, notional: 1320688, pctNav: 1.03 },
  { contract: "NQM25", desc: "Nasdaq 100 Jun", side: "LONG", qty: 2, entry: 18240.00, last: 18415.50, pnl: 17550, margin: 22000, notional: 736620, pctNav: 0.57 },
  { contract: "CLK25", desc: "Crude Oil May", side: "SHORT", qty: -3, entry: 84.20, last: 82.45, pnl: 5250, margin: 7500, notional: 247350, pctNav: 0.19 },
  { contract: "GCM25", desc: "Gold Jun", side: "LONG", qty: 2, entry: 2285.40, last: 2318.80, pnl: 6680, margin: 15000, notional: 463760, pctNav: 0.36 },
  { contract: "ZBM25", desc: "30Y T-Bond Jun", side: "SHORT", qty: -5, entry: 118.12, last: 117.28, pnl: 4375, margin: 12500, notional: 586400, pctNav: 0.46 },
];

const MARGIN_INFO = {
  regT: 4250000,
  portfolioMargin: 2840000,
  marginUsed: 2840000,
  marginAvailable: 7160000,
  maintenanceMargin: 2130000,
  utilizationPct: 28.4,
  buyingPower: 14320000,
  sma: 3250000,
};

export default function RiskPortfolio() {
  // ─── Engine API ─────────────────────────────────────
  const { data: greeksData } = useEngineQuery<RiskGreeks>("/risk/greeks", { refetchInterval: 5000 });
  const { data: betaData } = useEngineQuery<BetaState>("/portfolio/beta", { refetchInterval: 5000 });
  const { data: stressData } = useEngineQuery<{ scenarios: Array<Record<string, unknown>> }>("/risk/beta/stress", { refetchInterval: 30000 });
  const { data: metricsApi } = useEngineQuery<{ metrics: Array<{ name: string; value: string; status: string }> }>("/risk/metrics", { refetchInterval: 10000 });
  const { data: fillsApi } = useEngineQuery<{ fills: Array<{ time: string; pair: string; side: string; qty: number; price: number; status: string; strategy: string }> }>("/risk/fills", { refetchInterval: 5000 });
  const { data: optionsApi } = useEngineQuery<{ positions: Array<Record<string, number | string>>; aggregate: Record<string, number> }>("/risk/options-positions", { refetchInterval: 10000 });
  const { data: futuresApi } = useEngineQuery<{ positions: Array<Record<string, number | string>>; totals: Record<string, number> }>("/risk/futures-positions", { refetchInterval: 10000 });
  const { data: marginApi } = useEngineQuery<{ margin: Record<string, number> }>("/risk/margin", { refetchInterval: 10000 });
  const { data: liqScoringApi } = useEngineQuery<{ scoring: Array<{ asset: string; score: number; adv: string; spread: string; impact: string }> }>("/risk/liquidity-scoring", { refetchInterval: 30000 });

  // Override Greeks from API when available
  const greeks = useMemo(() => {
    if (!greeksData) return GREEKS;
    return [
      { name: "Delta", value: greeksData.delta ?? GREEKS[0].value, label: "Δ", color: "#00d4aa" },
      { name: "Gamma", value: greeksData.gamma ?? GREEKS[1].value, label: "Γ", color: "#58a6ff" },
      { name: "Theta", value: greeksData.theta ?? GREEKS[2].value, label: "Θ", color: "#f85149" },
      { name: "Vega", value: greeksData.vega ?? GREEKS[3].value, label: "ν", color: "#bc8cff" },
      { name: "Rho", value: greeksData.rho ?? GREEKS[4].value, label: "ρ", color: "#d29922" },
    ];
  }, [greeksData]);

  // VaR from beta analytics
  const varValue = (betaData?.analytics as Record<string, number>)?.var_pct ?? 1.22;
  const cvarValue = (betaData?.analytics as Record<string, number>)?.cvar_pct ?? 2.15;

  // Wire all remaining static data to API
  const metrics = metricsApi?.metrics?.length ? metricsApi.metrics : METRICS;
  const fills = fillsApi?.fills?.length ? fillsApi.fills : FILLS;
  const liquidityRisk = liqScoringApi?.scoring?.length ? liqScoringApi.scoring : LIQUIDITY_RISK;
  const optionsPositions = optionsApi?.positions?.length ? optionsApi.positions as typeof OPTIONS_POSITIONS : OPTIONS_POSITIONS;
  const aggGreeks = optionsApi?.aggregate ?? AGG_GREEKS;
  const futuresPositions = futuresApi?.positions?.length ? futuresApi.positions as typeof FUTURES_POSITIONS : FUTURES_POSITIONS;
  const marginInfo = marginApi?.margin ?? MARGIN_INFO;

  return (
    <div className="h-full flex flex-col gap-[2px] p-[2px] overflow-auto" data-testid="risk-portfolio">

      {/* ── ROW 1: Greeks | VaR (resizable) ── */}
      <div className="flex-shrink-0 h-36">
        <ResizableDashboard defaultSizes={[68, 32]} minSizes={[40, 22]}>
          {/* Greeks */}
          <DashboardPanel title="OPTIONS GREEKS — AGGREGATE">
            <div className="flex gap-4">
              {greeks.map((g) => (
                <div key={g.name} className="flex-1 text-center border border-terminal-border rounded p-3">
                  <div className="text-xl font-mono font-bold" style={{ color: g.color }}>{g.label}</div>
                  <div className="text-lg font-mono font-bold text-terminal-text-primary tabular-nums mt-1">
                    {g.value > 0 && g.name !== "Rho" ? "+" : ""}{g.value}
                  </div>
                  <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider mt-0.5">{g.name}</div>
                </div>
              ))}
            </div>
          </DashboardPanel>

          {/* VaR / CVaR Gauges */}
          <DashboardPanel title="VAR / CVAR">
        <div className="flex flex-col gap-3">
          <div>
            <div className="flex justify-between text-[9px] mb-1">
              <span className="text-terminal-text-muted">Value at Risk (95%)</span>
              <span className="text-terminal-warning font-mono tabular-nums">{varValue.toFixed(2)}%</span>
            </div>
            <div className="h-2 bg-terminal-surface-2 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-terminal-accent to-terminal-warning rounded-full transition-all duration-500"
                style={{ width: `${(varValue / 3) * 100}%` }}
              />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-[9px] mb-1">
              <span className="text-terminal-text-muted">Conditional VaR (95%)</span>
              <span className="text-terminal-negative font-mono tabular-nums">{cvarValue.toFixed(2)}%</span>
            </div>
            <div className="h-2 bg-terminal-surface-2 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-terminal-warning to-terminal-negative rounded-full transition-all duration-500"
                style={{ width: `${(cvarValue / 3) * 100}%` }}
              />
            </div>
          </div>
          <div className="text-[8px] text-terminal-text-faint mt-1">
            Total Initial Margin: <span className="text-terminal-text-primary font-mono">$4,250,000</span>
          </div>
        </div>
      </DashboardPanel>
        </ResizableDashboard>
      </div>

      {/* ── ROW 2: Portfolio Metrics | Fills Table | Liquidity Risk ── */}
      {/* Portfolio Metrics */}
      <DashboardPanel title="PORTFOLIO METRICS">
        <div className="grid grid-cols-2 gap-2">
          {metrics.map((m, i) => (
            <div key={i} className="border border-terminal-border/50 rounded p-2">
              <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider">{m.name}</div>
              <div className={`text-sm font-mono font-bold tabular-nums mt-0.5 ${
                m.status === "good" ? "text-terminal-positive" :
                m.status === "warning" ? "text-terminal-negative" :
                "text-terminal-text-primary"
              }`}>{m.value}</div>
            </div>
          ))}
        </div>
      </DashboardPanel>

      {/* Fills Table — now with Strategy column */}
      <DashboardPanel title="FILLED / NO FILLS" className="col-span-2" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[9px] font-mono">
            <thead>
              <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                <th className="text-left px-2 py-1.5 font-medium">Time</th>
                <th className="text-left px-2 py-1.5 font-medium">Pair</th>
                <th className="text-left px-2 py-1.5 font-medium">Side</th>
                <th className="text-right px-2 py-1.5 font-medium">Qty</th>
                <th className="text-right px-2 py-1.5 font-medium">Price</th>
                <th className="text-left px-2 py-1.5 font-medium">Strategy</th>
                <th className="text-right px-2 py-1.5 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {fills.map((f, i) => (
                <tr key={i} className="border-b border-terminal-border/20 hover:bg-white/[0.02]">
                  <td className="px-2 py-1.5 text-terminal-text-faint">{f.time}</td>
                  <td className="px-2 py-1.5 text-terminal-text-primary font-medium">{f.pair}</td>
                  <td className={`px-2 py-1.5 ${f.side === "BUY" ? "text-terminal-positive" : "text-terminal-negative"}`}>{f.side}</td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-muted tabular-nums">{f.qty}</td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-primary tabular-nums">${f.price.toFixed(2)}</td>
                  <td className="px-2 py-1.5 text-terminal-accent text-[8px]">{f.strategy}</td>
                  <td className={`px-2 py-1.5 text-right font-medium ${
                    f.status === "FILLED" ? "text-terminal-positive" :
                    f.status === "PARTIAL" ? "text-terminal-warning" :
                    "text-terminal-negative"
                  }`}>{f.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>

      {/* ── ROW 3: Liquidity Risk ── */}
      {/* Liquidity Risk */}
      <DashboardPanel title="LIQUIDITY RISK SCORING" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[9px] font-mono">
            <thead>
              <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                <th className="text-left px-2 py-1.5 font-medium">Asset</th>
                <th className="text-right px-2 py-1.5 font-medium">Score</th>
                <th className="text-right px-2 py-1.5 font-medium">Spread</th>
                <th className="text-right px-2 py-1.5 font-medium">Impact</th>
              </tr>
            </thead>
            <tbody>
              {liquidityRisk.map((l, i) => (
                <tr key={i} className="border-b border-terminal-border/20">
                  <td className="px-2 py-1.5 text-terminal-text-primary">{l.asset}</td>
                  <td className="px-2 py-1.5 text-right">
                    <span className={`tabular-nums ${l.score >= 80 ? "text-terminal-positive" : l.score >= 60 ? "text-terminal-warning" : "text-terminal-negative"}`}>
                      {l.score}
                    </span>
                  </td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-muted tabular-nums">{l.spread}</td>
                  <td className={`px-2 py-1.5 text-right ${
                    l.impact === "Low" ? "text-terminal-positive" : l.impact === "Medium" ? "text-terminal-warning" : "text-terminal-negative"
                  }`}>{l.impact}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>

      {/* ── OPTIONS POSITIONS (full-width) ── */}
      <DashboardPanel
        title="OPTIONS POSITIONS"
        className="col-span-3"
        noPadding
        headerRight={
          <div className="flex gap-4 text-[9px] font-mono">
            <span className="text-terminal-text-faint">Net Δ <span className={`font-bold tabular-nums ${aggGreeks.delta >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>{aggGreeks.delta.toFixed(2)}</span></span>
            <span className="text-terminal-text-faint">Γ <span className="text-[#58a6ff] font-bold tabular-nums">{aggGreeks.gamma.toFixed(3)}</span></span>
            <span className="text-terminal-text-faint">Θ/day <span className="text-terminal-negative font-bold tabular-nums">{aggGreeks.theta.toFixed(0)}</span></span>
            <span className="text-terminal-text-faint">ν <span className="text-[#bc8cff] font-bold tabular-nums">{aggGreeks.vega.toFixed(0)}</span></span>
            <span className="text-terminal-text-faint">Total P&L <span className={`font-bold tabular-nums ${aggGreeks.totalPnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>${aggGreeks.totalPnl.toLocaleString()}</span></span>
          </div>
        }
      >
        <div className="overflow-auto" style={{ maxHeight: 220 }}>
          <table className="w-full text-[9px] font-mono">
            <thead className="sticky top-0 bg-terminal-surface z-10">
              <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                <th className="text-left px-2 py-1.5 font-medium">Ticker</th>
                <th className="text-left px-2 py-1.5 font-medium">Type</th>
                <th className="text-right px-2 py-1.5 font-medium">Strike</th>
                <th className="text-left px-2 py-1.5 font-medium">Expiry</th>
                <th className="text-right px-2 py-1.5 font-medium">Qty</th>
                <th className="text-right px-2 py-1.5 font-medium">Delta</th>
                <th className="text-right px-2 py-1.5 font-medium">Gamma</th>
                <th className="text-right px-2 py-1.5 font-medium">Theta</th>
                <th className="text-right px-2 py-1.5 font-medium">Vega</th>
                <th className="text-right px-2 py-1.5 font-medium">Premium</th>
                <th className="text-right px-2 py-1.5 font-medium">Mkt Val</th>
                <th className="text-right px-2 py-1.5 font-medium">P&L</th>
                <th className="text-left px-2 py-1.5 font-medium">Strategy</th>
              </tr>
            </thead>
            <tbody>
              {optionsPositions.map((o, i) => (
                <tr key={i} className="border-b border-terminal-border/20 hover:bg-white/[0.02]">
                  <td className="px-2 py-1 text-terminal-accent font-medium">{o.ticker}</td>
                  <td className={`px-2 py-1 font-bold ${o.type === "CALL" ? "text-terminal-positive" : "text-terminal-negative"}`}>{o.type}</td>
                  <td className="px-2 py-1 text-right text-terminal-text-primary tabular-nums">{o.strike}</td>
                  <td className="px-2 py-1 text-terminal-text-muted">{o.expiry}</td>
                  <td className={`px-2 py-1 text-right tabular-nums font-medium ${o.qty >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>{o.qty > 0 ? "+" : ""}{o.qty}</td>
                  <td className={`px-2 py-1 text-right tabular-nums ${o.delta >= 0 ? "text-[#00d4aa]" : "text-[#f85149]"}`}>{o.delta.toFixed(3)}</td>
                  <td className="px-2 py-1 text-right tabular-nums text-[#58a6ff]">{o.gamma.toFixed(3)}</td>
                  <td className="px-2 py-1 text-right tabular-nums text-[#f85149]">{o.theta.toFixed(2)}</td>
                  <td className="px-2 py-1 text-right tabular-nums text-[#bc8cff]">{o.vega.toFixed(2)}</td>
                  <td className="px-2 py-1 text-right tabular-nums text-terminal-text-primary">${o.premium.toFixed(2)}</td>
                  <td className={`px-2 py-1 text-right tabular-nums ${o.mktVal >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {o.mktVal >= 0 ? "$" : "-$"}{Math.abs(o.mktVal).toLocaleString()}
                  </td>
                  <td className={`px-2 py-1 text-right tabular-nums font-medium ${o.pnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {o.pnl >= 0 ? "+$" : "-$"}{Math.abs(o.pnl).toLocaleString()}
                  </td>
                  <td className="px-2 py-1 text-[8px] text-terminal-warning">{o.strategy}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>

      {/* ── ROW 4: Futures Positions | Margin Panel (resizable) ── */}
      <div className="flex-shrink-0 h-52">
        <ResizableDashboard defaultSizes={[68, 32]} minSizes={[40, 22]}>
          {/* Futures Positions */}
          <DashboardPanel title="FUTURES POSITIONS" noPadding>
        <div className="overflow-auto">
          <table className="w-full text-[9px] font-mono">
            <thead>
              <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                <th className="text-left px-2 py-1.5 font-medium">Contract</th>
                <th className="text-left px-2 py-1.5 font-medium">Desc</th>
                <th className="text-left px-2 py-1.5 font-medium">Side</th>
                <th className="text-right px-2 py-1.5 font-medium">Qty</th>
                <th className="text-right px-2 py-1.5 font-medium">Entry</th>
                <th className="text-right px-2 py-1.5 font-medium">Last</th>
                <th className="text-right px-2 py-1.5 font-medium">P&L</th>
                <th className="text-right px-2 py-1.5 font-medium">Margin</th>
                <th className="text-right px-2 py-1.5 font-medium">Notional</th>
                <th className="text-right px-2 py-1.5 font-medium">% NAV</th>
              </tr>
            </thead>
            <tbody>
              {futuresPositions.map((f, i) => (
                <tr key={i} className="border-b border-terminal-border/20 hover:bg-white/[0.02]">
                  <td className="px-2 py-1.5 text-terminal-accent font-bold">{f.contract}</td>
                  <td className="px-2 py-1.5 text-terminal-text-muted">{f.desc}</td>
                  <td className={`px-2 py-1.5 font-medium ${f.side === "LONG" ? "text-terminal-positive" : "text-terminal-negative"}`}>{f.side}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-text-primary">{Math.abs(f.qty)}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-text-muted">{f.entry.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-text-primary">{f.last.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                  <td className={`px-2 py-1.5 text-right tabular-nums font-medium ${f.pnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {f.pnl >= 0 ? "+$" : "-$"}{Math.abs(f.pnl).toLocaleString()}
                  </td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-text-muted">${f.margin.toLocaleString()}</td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-text-primary">${(f.notional / 1e6).toFixed(2)}M</td>
                  <td className="px-2 py-1.5 text-right tabular-nums text-terminal-warning">{f.pctNav.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="border-t border-terminal-border/50 bg-white/[0.02]">
                <td colSpan={6} className="px-2 py-1.5 text-terminal-text-faint text-[8px] font-medium uppercase">Totals</td>
                <td className="px-2 py-1.5 text-right text-terminal-positive font-bold tabular-nums text-[9px]">
                  +${futuresPositions.reduce((s, f) => s + f.pnl, 0).toLocaleString()}
                </td>
                <td className="px-2 py-1.5 text-right text-terminal-text-muted font-mono tabular-nums text-[9px]">
                  ${futuresPositions.reduce((s, f) => s + f.margin, 0).toLocaleString()}
                </td>
                <td className="px-2 py-1.5 text-right text-terminal-text-primary font-mono tabular-nums text-[9px]">
                  ${(futuresPositions.reduce((s, f) => s + f.notional, 0) / 1e6).toFixed(2)}M
                </td>
                <td className="px-2 py-1.5 text-right text-terminal-warning font-bold tabular-nums text-[9px]">
                  {futuresPositions.reduce((s, f) => s + f.pctNav, 0).toFixed(2)}%
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
          </DashboardPanel>

          {/* Margin Information Panel */}
          <DashboardPanel title="MARGIN INFORMATION">
        <div className="space-y-2">
          {/* Margin Utilization Gauge */}
          <div>
            <div className="flex justify-between text-[9px] mb-1">
              <span className="text-terminal-text-faint uppercase tracking-wider">Margin Utilization</span>
              <span className={`font-mono font-bold tabular-nums ${marginInfo.utilizationPct < 50 ? "text-terminal-positive" : marginInfo.utilizationPct < 75 ? "text-terminal-warning" : "text-terminal-negative"}`}>
                {marginInfo.utilizationPct}%
              </span>
            </div>
            <div className="h-2.5 bg-terminal-surface-2 rounded-full overflow-hidden border border-terminal-border/30">
              <div
                className={`h-full rounded-full transition-all duration-700 ${
                  marginInfo.utilizationPct < 50 ? "bg-gradient-to-r from-terminal-positive to-terminal-accent" :
                  marginInfo.utilizationPct < 75 ? "bg-gradient-to-r from-terminal-accent to-terminal-warning" :
                  "bg-gradient-to-r from-terminal-warning to-terminal-negative"
                }`}
                style={{ width: `${marginInfo.utilizationPct}%` }}
              />
            </div>
            <div className="flex justify-between text-[7px] text-terminal-text-faint mt-0.5">
              <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
            </div>
          </div>

          {/* Margin rows */}
          <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 pt-1">
            {[
              { label: "Reg-T Requirement", val: `$${(marginInfo.regT / 1e6).toFixed(2)}M`, color: "text-terminal-text-primary" },
              { label: "Portfolio Margin", val: `$${(marginInfo.portfolioMargin / 1e6).toFixed(2)}M`, color: "text-terminal-accent" },
              { label: "Margin Used", val: `$${(marginInfo.marginUsed / 1e6).toFixed(2)}M`, color: "text-terminal-warning" },
              { label: "Margin Available", val: `$${(marginInfo.marginAvailable / 1e6).toFixed(2)}M`, color: "text-terminal-positive" },
              { label: "Maintenance Margin", val: `$${(marginInfo.maintenanceMargin / 1e6).toFixed(2)}M`, color: "text-terminal-negative" },
              { label: "Buying Power", val: `$${(marginInfo.buyingPower / 1e6).toFixed(2)}M`, color: "text-terminal-positive" },
              { label: "SMA", val: `$${(marginInfo.sma / 1e6).toFixed(2)}M`, color: "text-[#bc8cff]" },
            ].map((r, i) => (
              <div key={i} className="flex flex-col">
                <span className="text-[7px] text-terminal-text-faint uppercase tracking-wider">{r.label}</span>
                <span className={`text-[10px] font-mono font-bold tabular-nums ${r.color}`}>{r.val}</span>
              </div>
            ))}
          </div>
        </div>
          </DashboardPanel>
        </ResizableDashboard>
      </div>

    </div>
  );
}
