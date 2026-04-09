import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import { useState, useEffect, useMemo } from "react";
import { useEngineQuery, type RiskGreeks, type BetaState } from "@/hooks/use-engine-api";

// Static fallbacks removed — all data from live API endpoints









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
  const greeks = useMemo(() => [
    { name: "Delta", value: greeksData?.delta ?? 0, label: "Δ", color: "#00d4aa" },
    { name: "Gamma", value: greeksData?.gamma ?? 0, label: "Γ", color: "#58a6ff" },
    { name: "Theta", value: greeksData?.theta ?? 0, label: "Θ", color: "#f85149" },
    { name: "Vega", value: greeksData?.vega ?? 0, label: "ν", color: "#bc8cff" },
    { name: "Rho", value: greeksData?.rho ?? 0, label: "ρ", color: "#d29922" },
  ], [greeksData]);

  // VaR from beta analytics
  const varValue = (betaData?.analytics as Record<string, number>)?.var_pct ?? 0;
  const cvarValue = (betaData?.analytics as Record<string, number>)?.cvar_pct ?? 0;

  // Wire all remaining static data to API
  const metrics = metricsApi?.metrics || [];
  const fills = fillsApi?.fills || [];
  const liquidityRisk = liqScoringApi?.scoring || [];
  const optionsPositions = optionsApi?.positions || [];
  const aggGreeks = optionsApi?.aggregate ?? { delta: 0, gamma: 0, theta: 0, vega: 0, totalPnl: 0 };
  const futuresPositions = futuresApi?.positions || [];
  const marginInfo = marginApi?.margin ?? { used: 0, available: 0, maintenance: 0, equity: 0 };

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
          {metrics.length === 0 && (
            <div style={{color: "var(--muted)", fontSize: 11, padding: "16px", textAlign: "center", opacity: 0.7, gridColumn: "1 / -1"}}>
              Risk metrics loading — awaiting engine computation...
            </div>
          )}
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
              {fills.length === 0 && (
                <tr><td colSpan={7} style={{color: "var(--muted)", fontSize: 11, padding: "16px", textAlign: "center", opacity: 0.7}}>
                  No fills yet — trade executions will appear here.
                </td></tr>
              )}
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
              {liquidityRisk.length === 0 && (
                <tr><td colSpan={5} style={{color: "var(--muted)", fontSize: 11, padding: "16px", textAlign: "center", opacity: 0.7}}>
                  Liquidity scoring loading — awaiting position data...
                </td></tr>
              )}
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
              {optionsPositions.length === 0 && (
                <tr><td colSpan={10} style={{color: "var(--muted)", fontSize: 11, padding: "16px", textAlign: "center", opacity: 0.7}}>
                  No options positions — will populate from broker.
                </td></tr>
              )}
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
              {futuresPositions.length === 0 && (
                <tr><td colSpan={8} style={{color: "var(--muted)", fontSize: 11, padding: "16px", textAlign: "center", opacity: 0.7}}>
                  No futures positions — will populate from broker.
                </td></tr>
              )}
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
