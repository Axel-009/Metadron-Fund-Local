import { DashboardPanel } from "@/components/dashboard-panel";
import { useState, useEffect } from "react";

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
  { time: "14:32:18", pair: "AAPL", side: "BUY", qty: 500, price: 189.45, status: "FILLED" },
  { time: "14:31:45", pair: "MSFT", side: "SELL", qty: 200, price: 420.12, status: "FILLED" },
  { time: "14:30:22", pair: "NVDA", side: "BUY", qty: 100, price: 875.30, status: "FILLED" },
  { time: "14:29:55", pair: "AMZN", side: "BUY", qty: 300, price: 185.67, status: "PARTIAL" },
  { time: "14:28:10", pair: "GOOGL", side: "SELL", qty: 150, price: 155.89, status: "FILLED" },
  { time: "14:27:30", pair: "META", side: "BUY", qty: 250, price: 505.78, status: "NO FILL" },
  { time: "14:25:12", pair: "JPM", side: "SELL", qty: 400, price: 198.34, status: "FILLED" },
];

const LIQUIDITY_RISK = [
  { asset: "AAPL", score: 95, adv: "62.3M", spread: "0.01%", impact: "Low" },
  { asset: "NVDA", score: 92, adv: "45.1M", spread: "0.02%", impact: "Low" },
  { asset: "MSFT", score: 94, adv: "28.7M", spread: "0.01%", impact: "Low" },
  { asset: "BTC/USD", score: 78, adv: "18.2B", spread: "0.05%", impact: "Medium" },
  { asset: "SMCI", score: 55, adv: "8.4M", spread: "0.15%", impact: "High" },
  { asset: "ARM", score: 62, adv: "12.1M", spread: "0.08%", impact: "Medium" },
];

export default function RiskPortfolio() {
  const [varValue, setVarValue] = useState(1.22);
  const [cvarValue, setCvarValue] = useState(2.15);

  useEffect(() => {
    const iv = setInterval(() => {
      setVarValue(1.1 + Math.random() * 0.3);
      setCvarValue(1.9 + Math.random() * 0.5);
    }, 5000);
    return () => clearInterval(iv);
  }, []);

  return (
    <div className="h-full grid grid-cols-3 grid-rows-[auto_1fr_1fr] gap-[2px] p-[2px] overflow-auto" data-testid="risk-portfolio">
      {/* Greeks */}
      <DashboardPanel title="OPTIONS GREEKS" className="col-span-2">
        <div className="flex gap-4">
          {GREEKS.map((g) => (
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

      {/* Portfolio Metrics */}
      <DashboardPanel title="PORTFOLIO METRICS">
        <div className="grid grid-cols-2 gap-2">
          {METRICS.map((m, i) => (
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

      {/* Fills Table */}
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
                <th className="text-right px-2 py-1.5 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {FILLS.map((f, i) => (
                <tr key={i} className="border-b border-terminal-border/20 hover:bg-white/[0.02]">
                  <td className="px-2 py-1.5 text-terminal-text-faint">{f.time}</td>
                  <td className="px-2 py-1.5 text-terminal-text-primary font-medium">{f.pair}</td>
                  <td className={`px-2 py-1.5 ${f.side === "BUY" ? "text-terminal-positive" : "text-terminal-negative"}`}>{f.side}</td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-muted tabular-nums">{f.qty}</td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-primary tabular-nums">${f.price.toFixed(2)}</td>
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
              {LIQUIDITY_RISK.map((l, i) => (
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
    </div>
  );
}
